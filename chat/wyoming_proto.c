// Wyoming protocol, protocol part (see wyoming_proto.h).
#include "wyoming_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long num_member(const JNode* pool, int obj, const char* key) {
    int k = json_get(pool, obj, key);
    if (k < 0 || pool[k].type == J_NULL) return 0;
    if (pool[k].type != J_NUM || pool[k].num < 0 || pool[k].num > 1e9) return -1;
    return (long)pool[k].num;
}

long wy_parse(const char* buf, size_t len, WyEvent* ev) {
    const char* nl = memchr(buf, '\n', len < WY_MAX_LINE ? len : WY_MAX_LINE);
    if (!nl) return len >= WY_MAX_LINE ? -1 : 0;
    size_t line = (size_t)(nl - buf);
    ev->head_root = json_parse(buf, line, ev->head, 64);
    if (ev->head_root < 0 || ev->head[ev->head_root].type != J_OBJ) return -1;
    int t = json_get(ev->head, ev->head_root, "type");
    if (t < 0 || ev->head[t].type != J_STR || ev->head[t].slen == 0 || ev->head[t].slen >= sizeof(ev->type)) return -1;
    memcpy(ev->type, ev->head[t].s, ev->head[t].slen);
    ev->type[ev->head[t].slen] = 0;
    ev->head_data = json_get(ev->head, ev->head_root, "data");
    if (ev->head_data >= 0 && ev->head[ev->head_data].type != J_OBJ) ev->head_data = -1;
    long dl = num_member(ev->head, ev->head_root, "data_length");
    long pl = num_member(ev->head, ev->head_root, "payload_length");
    if (dl < 0 || pl < 0 || dl > WY_MAX_DATA || pl > WY_MAX_PAYLOAD) return -1;
    size_t need = line + 1 + (size_t)dl + (size_t)pl;
    if (len < need) return 0;
    ev->data_root = -1;
    if (dl > 0) {
        ev->data_root = json_parse(nl + 1, (size_t)dl, ev->data, WY_NODES);
        if (ev->data_root < 0 || ev->data[ev->data_root].type != J_OBJ) return -1;
    }
    ev->payload = pl ? (const uint8_t*)nl + 1 + dl : NULL;
    ev->payload_len = (size_t)pl;
    return (long)need;
}

int wy_get(const WyEvent* ev, const char* key, const JNode** pool) {
    int k = ev->data_root >= 0 ? json_get(ev->data, ev->data_root, key) : -1;
    if (k >= 0) {
        *pool = ev->data;
        return k;
    }
    k = ev->head_data >= 0 ? json_get(ev->head, ev->head_data, key) : -1;
    *pool = ev->head;
    return k;
}

static void copy_str(const JNode* pool, int k, char* out, size_t cap) {
    out[0] = 0;
    if (k < 0 || pool[k].type != J_STR || cap == 0) return;
    Sb s;
    sb_init(&s);
    json_str_sb(pool, k, &s);
    size_t n = s.n < cap - 1 ? s.n : cap - 1;
    while (n > 0 && n < s.n && ((unsigned char)s.p[n] & 0xC0) == 0x80) n--;   // (no half character at the end)
    if (n) memcpy(out, s.p, n);
    out[n] = 0;
    sb_free(&s);
}

void wy_get_str(const WyEvent* ev, const char* key, char* out, size_t cap) {
    const JNode* pool;
    int k = wy_get(ev, key, &pool);
    copy_str(pool, k, out, cap);
}

long wy_get_int(const WyEvent* ev, const char* key, long def) {
    const JNode* pool;
    int k = wy_get(ev, key, &pool);
    return k >= 0 && pool[k].type == J_NUM ? (long)pool[k].num : def;
}

void wy_get_sub_str(const WyEvent* ev, const char* obj, const char* key, char* out, size_t cap) {
    const JNode* pool;
    out[0] = 0;
    int o = wy_get(ev, obj, &pool);
    if (o < 0 || pool[o].type != J_OBJ) return;
    copy_str(pool, json_get(pool, o, key), out, cap);
}

void wy_write(Sb* out, const char* type, const char* data, const void* payload, size_t n) {
    size_t dl = data ? strlen(data) : 0;
    sb_str(out, "{\"type\":");
    sb_json_str(out, type, strlen(type));
    sb_str(out, ",\"version\":\"" WY_VERSION "\"");
    if (dl) {
        sb_str(out, ",\"data_length\":");
        sb_int(out, (long long)dl);
    }
    if (payload && n) {
        sb_str(out, ",\"payload_length\":");
        sb_int(out, (long long)n);
    }
    sb_str(out, "}\n");
    if (dl) sb_add(out, data, dl);
    if (payload && n) sb_add(out, (const char*)payload, n);
}

void wy_audio_start(Sb* out, int rate) {
    char d[80];
    snprintf(d, sizeof(d), "{\"rate\":%d,\"width\":2,\"channels\":1,\"timestamp\":0}", rate);
    wy_write(out, "audio-start", d, NULL, 0);
}

void wy_audio_chunk(Sb* out, int rate, const int16_t* pcm, int n) {
    char d[80];
    snprintf(d, sizeof(d), "{\"rate\":%d,\"width\":2,\"channels\":1}", rate);
    // (the payload is little-endian, as the console and the PCs that listen are)
    wy_write(out, "audio-chunk", d, pcm, (size_t)n * 2);
}

void wy_audio_stop(Sb* out) { wy_write(out, "audio-stop", "{\"timestamp\":0}", NULL, 0); }

// ---- the info event ---------------------------------------------------------------------------------------------------

static void key_str(Sb* s, const char* key, const char* v, int comma) {
    if (comma) sb_str(s, ",");
    sb_json_str(s, key, strlen(key));
    sb_str(s, ":");
    if (v) sb_json_str(s, v, strlen(v));
    else sb_str(s, "null");
}

// name, attribution, installed, description, version: the fields every Wyoming artifact has
static void artifact(Sb* s, const char* name, const char* who, const char* url, const char* desc, const char* version) {
    key_str(s, "name", name, 0);
    sb_str(s, ",\"attribution\":{");
    key_str(s, "name", who, 0);
    key_str(s, "url", url, 1);
    sb_str(s, "},\"installed\":true");
    key_str(s, "description", desc, 1);
    key_str(s, "version", version, 1);
}

static void langs(Sb* s, const char* const* l, int n) {
    sb_str(s, ",\"languages\":[");
    for (int i = 0; i < n; i++) {
        if (i) sb_str(s, ",");
        sb_json_str(s, l[i], strlen(l[i]));
    }
    sb_str(s, "]");
}

void wy_info_json(Sb* out, const WyInfo* in) {
    const char* app = "https://github.com/leonardo-alegria/garlic-llm";
    sb_str(out, "{\"asr\":[");
    if (in->asr) {
        sb_str(out, "{");
        artifact(out, "garlic-parakeet", "Garlic LLM", app, "Parakeet TDT 0.6B v3 on the PS4's GPU", in->version);
        sb_str(out, ",\"models\":[{");
        artifact(out, "parakeet-tdt-0.6b-v3", "NVIDIA", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3",
                 "Parakeet TDT 0.6B v3", "v3");
        langs(out, in->asr_langs, in->n_asr_langs);
        sb_str(out, "}],\"supports_transcript_streaming\":false,\"requires_external_vad\":true}");
    }
    sb_str(out, "],\"tts\":[");
    if (in->tts) {
        sb_str(out, "{");
        artifact(out, "garlic-chatterbox", "Garlic LLM", app, "Chatterbox Multilingual on the PS4's GPU", in->version);
        sb_str(out, ",\"voices\":[");
        for (int i = 0; i < in->n_voices; i++) {
            const WyVoice* v = &in->voices[i];
            sb_str(out, i ? ",{" : "{");
            artifact(out, v->name, "Resemble AI", "https://github.com/resemble-ai/chatterbox", v->description,
                     in->version);
            langs(out, &v->language, 1);
            sb_str(out, ",\"speakers\":null}");
        }
        sb_str(out, "],\"supports_synthesize_streaming\":true}");
    }
    sb_str(out, "],\"wake\":[");
    if (in->wake) {
        sb_str(out, "{");
        artifact(out, "garlic-openwakeword", "Garlic LLM", app, "openWakeWord on the PS4's CPU", in->version);
        sb_str(out, ",\"models\":[");
        static const char* const en[] = {"en"};
        for (int i = 0; i < in->n_wake; i++) {
            sb_str(out, i ? ",{" : "{");
            artifact(out, in->wake_names[i], "dscripka", "https://github.com/dscripka/openWakeWord", in->wake_phrases[i],
                     "v0.1");
            langs(out, en, 1);
            key_str(out, "phrase", in->wake_phrases[i], 1);
            sb_str(out, "}");
        }
        sb_str(out, "]}");
    }
    sb_str(out, "],\"handle\":[],\"intent\":[],\"mic\":[],\"snd\":[]}");
}

// ---- audio ---------------------------------------------------------------------------------------------------------

int wy_to_16k(const uint8_t* in, size_t n, int rate, int width, int channels, int16_t* out, int cap) {
    if ((width != 2 && width != 4) || channels < 1 || channels > 8 || rate < 4000 || rate > 192000) return -1;
    size_t frame = (size_t)width * channels, frames = n / frame;
    // one mono sample of frame i, as a float in [-1, 1]
#define SAMPLE(i, out_v)                                                            \
    do {                                                                            \
        double acc = 0;                                                             \
        for (int c_ = 0; c_ < channels; c_++) {                                     \
            const uint8_t* p_ = in + (i) * frame + (size_t)c_ * width;              \
            if (width == 2) acc += (int16_t)(p_[0] | p_[1] << 8) / 32768.0;         \
            else acc += (int32_t)((uint32_t)p_[0] | (uint32_t)p_[1] << 8 | (uint32_t)p_[2] << 16 | \
                                  (uint32_t)p_[3] << 24) / 2147483648.0;            \
        }                                                                           \
        out_v = acc / channels;                                                     \
    } while (0)
    int m = 0;
    if (rate == 16000) {
        for (size_t i = 0; i < frames && m < cap; i++) {
            double v;
            SAMPLE(i, v);
            long s = (long)(v * 32768.0 + (v >= 0 ? 0.5 : -0.5));
            out[m++] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
        }
        return m;
    }
    size_t want = frames * 16000 / (size_t)rate;
    for (size_t j = 0; j < want && m < cap; j++) {
        double pos = (double)j * rate / 16000.0, a, b;
        size_t i = (size_t)pos;
        double f = pos - (double)i;
        SAMPLE(i, a);
        if (i + 1 < frames) SAMPLE(i + 1, b);
        else b = a;
        double v = a + (b - a) * f;
        long s = (long)(v * 32768.0 + (v >= 0 ? 0.5 : -0.5));
        out[m++] = (int16_t)(s > 32767 ? 32767 : s < -32768 ? -32768 : s);
    }
#undef SAMPLE
    return m;
}
