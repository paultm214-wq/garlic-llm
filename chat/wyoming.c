// Wyoming server on the console (see wyoming.h). The sockets are the console's sceNet calls, as in api.c.
// Nothing here runs until wy_enable(1).
#include "wyoming.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <orbis/Net.h>
#include <orbis/libkernel.h>

#include "api.h"
#include "oww.h"
#include "ui.h"
#include "wyoming_proto.h"

#ifndef WY_ASSETS
#define WY_ASSETS "/app0/assets"
#endif
#define WY_MAX_CONN 8               // clients served at once (each satellite keeps one open for its wake word)
#define WY_THRESHOLD 0.5f           // wake word: probability that counts (wyoming-openwakeword's default) ...
#define WY_REFRACTORY_MS 2000       // ... and the time before the same word can count again
#define WY_IDLE_S 600               // a connection that sends nothing for this long is closed

// socket constants the SDK's headers do not name (the BSD values; as api.c)
#define WY_SOL_SOCKET 0xffff
#define WY_SO_REUSEADDR 0x0004
#define WY_SO_RCVTIMEO 0x1006
#define WY_SO_SNDTIMEO 0x1005
#define WY_IPPROTO_TCP 6

typedef struct {
    uint8_t len, family;
    uint16_t port;
    uint32_t addr;
    char zero[8];
} WySockaddrIn;
_Static_assert(sizeof(WySockaddrIn) == sizeof(OrbisNetSockaddr), "sockaddr size");

static OrbisPthreadMutex wy_mu;
static int mu_ready, have_thread;
static OrbisPthread wy_th;
static volatile int on, accepting;
static OrbisNetId listen_fd = -1;
static int n_conn;                  // connection threads running
static WyJob *q_head, *q_tail;      // jobs waiting for the main loop
static int svc_asr, svc_tts;
static float asr_max_s = 30;
static OwwModels* oww;              // wake word models (loaded the first time the server goes on)
static uint8_t* oww_data;
static unsigned conn_count;

static const char* const asr_langs[] = {"bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
                                        "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"};
static const WyVoice voices[] = {
    {"garlic-en", "Garlic (English)", "en"},   {"garlic-es", "Garlic (Español)", "es"},
    {"garlic-pt", "Garlic (Português)", "pt"}, {"garlic-fr", "Garlic (Français)", "fr"},
    {"garlic-de", "Garlic (Deutsch)", "de"},   {"garlic-it", "Garlic (Italiano)", "it"},
    {"garlic-ru", "Garlic (Русский)", "ru"},
};
#define N_VOICES (int)(sizeof(voices) / sizeof(voices[0]))

const char* wy_voice_lang(const char* voice) {
    for (int i = 0; voice && i < N_VOICES; i++)
        if (!strcmp(voice, voices[i].name)) return voices[i].language;
    return NULL;
}

void wy_services(int asr, float max_s, int tts) {
    svc_asr = asr, svc_tts = tts;
    asr_max_s = max_s > 1 ? max_s : 1;
}

int wy_enabled(void) { return on; }

// ---- jobs for the main loop -------------------------------------------------------------------------------------------

WyJob* wy_take(void) {
    if (!mu_ready) return NULL;
    scePthreadMutexLock(&wy_mu);
    WyJob* j = q_head;
    if (j) {
        q_head = j->next;
        if (!q_head) q_tail = NULL;
    }
    scePthreadMutexUnlock(&wy_mu);
    return j;
}

void wy_done(WyJob* job) { job->done = 1; }

// Queues a job and waits until the main loop has done it.
static void run_job(WyJob* j) {
    j->next = NULL;
    scePthreadMutexLock(&wy_mu);
    if (q_tail) q_tail->next = j;
    else q_head = j;
    q_tail = j;
    scePthreadMutexUnlock(&wy_mu);
    while (!j->done) sceKernelUsleep(5000);
}

// ---- one connection -------------------------------------------------------------------------------------------------

typedef struct {
    int fd;
    unsigned id;
    char* buf;
    size_t len, cap;
    WyEvent ev;
    int mode;                       // what audio is for: WY_ASR (default) or 'w' (wake words, after detect)
    char lang[8];
    int16_t* audio;                 // speech to text: the samples so far
    int n_audio, cap_audio, cut;
    OwwStream* ws;                  // wake words
    int detected;
    int64_t audio_ms, last_hit_ms[OWW_MAX_WORDS];
    char asked[OWW_MAX_WORDS][64];  // the name each word was asked for by (detections carry it back)
    int tts_stream;                 // between synthesize-start and synthesize-stop
    char voice[48];
    Sb tts_text;
    int16_t conv[16384];
} Conn;

static int send_event(Conn* c, const char* type, const char* data) {
    Sb s;
    sb_init(&s);
    wy_write(&s, type, data, NULL, 0);
    int r = s.p && !s.oom ? api_send(c->fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

static void send_error(Conn* c, const char* text, const char* code) {
    Sb d;
    sb_init(&d);
    sb_str(&d, "{\"text\":");
    sb_json_str(&d, text, strlen(text));
    sb_str(&d, ",\"code\":");
    sb_json_str(&d, code, strlen(code));
    sb_str(&d, "}");
    if (d.p) send_event(c, "error", d.p);
    sb_free(&d);
    say("wyoming #%u: error sent: %s", c->id, text);
}

static void send_info(Conn* c) {
    const char* names[OWW_MAX_WORDS];
    const char* phrases[OWW_MAX_WORDS];
    int nw = oww_count(oww);
    for (int i = 0; i < nw; i++) names[i] = oww_name(oww, i), phrases[i] = oww_phrase(oww, i);
    WyInfo in = {svc_asr, asr_langs, (int)(sizeof(asr_langs) / sizeof(asr_langs[0])), svc_tts, voices, N_VOICES,
                 nw > 0, names, phrases, nw, "1.20"};
    Sb d;
    sb_init(&d);
    wy_info_json(&d, &in);
    if (d.p) send_event(c, "info", d.p);
    sb_free(&d);
}

typedef struct {
    Conn* c;
    int64_t ms;                     // the audio's time at the start of the chunk being fed
} ScoreCtx;

static void on_score(void* user, int word, float p) {
    ScoreCtx* x = user;
    Conn* c = x->c;
    if (p <= WY_THRESHOLD) return;
    if (c->last_hit_ms[word] >= 0 && x->ms - c->last_hit_ms[word] < WY_REFRACTORY_MS) return;
    c->last_hit_ms[word] = x->ms;
    c->detected = 1;
    const char* name = c->asked[word][0] ? c->asked[word] : oww_name(oww, word);
    Sb d;
    sb_init(&d);
    sb_str(&d, "{\"name\":");
    sb_json_str(&d, name, strlen(name));
    sb_str(&d, ",\"timestamp\":");
    sb_int(&d, (long long)x->ms);
    sb_str(&d, "}");
    if (d.p) send_event(c, "detection", d.p);
    sb_free(&d);
    say("wyoming #%u: wake word %s (%.2f) at %.2f s", c->id, oww_name(oww, word), p, x->ms / 1000.0);
}

static void on_detect(Conn* c) {
    c->mode = 'w';
    if (!oww) {
        send_error(c, "Wake word models are not available on the console.", "no-models");
        return;
    }
    if (!c->ws && !(c->ws = oww_stream_new(oww))) {
        send_error(c, "Out of memory for wake word detection.", "no-memory");
        return;
    }
    for (int i = 0; i < oww_count(oww); i++) oww_stream_enable(c->ws, i, 0), c->asked[i][0] = 0;
    int any = 0;
    const JNode* pool;
    int names = wy_get(&c->ev, "names", &pool);
    if (names >= 0 && pool[names].type == J_ARR)
        for (int k = pool[names].first; k >= 0; k = pool[k].next) {
            char nm[64];
            Sb s;
            sb_init(&s);
            json_str_sb(pool, k, &s);
            snprintf(nm, sizeof(nm), "%s", s.p ? s.p : "");
            sb_free(&s);
            int w = oww_find(oww, nm);
            if (w >= 0) {
                oww_stream_enable(c->ws, w, 1), any = 1;
                snprintf(c->asked[w], sizeof(c->asked[w]), "%s", nm);
            }
            else say("wyoming #%u: no wake word called '%s'", c->id, nm);
        }
    if (!any) oww_stream_enable(c->ws, 0, 1);   // (none asked for, or none known: the first, okay_nabu)
}

static void audio_start(Conn* c) {
    if (c->mode == 'w') {
        if (c->ws) oww_stream_reset(c->ws);
        c->detected = 0;
        c->audio_ms = 0;
        for (int i = 0; i < OWW_MAX_WORDS; i++) c->last_hit_ms[i] = -1;
    } else {
        c->n_audio = 0, c->cut = 0;
    }
}

static void audio_chunk(Conn* c) {
    int rate = (int)wy_get_int(&c->ev, "rate", 16000), width = (int)wy_get_int(&c->ev, "width", 2);
    int ch = (int)wy_get_int(&c->ev, "channels", 1);
    if (!c->ev.payload_len) return;
    int n = wy_to_16k(c->ev.payload, c->ev.payload_len, rate, width, ch, c->conv, (int)(sizeof(c->conv) / 2));
    if (n < 0) {
        send_error(c, "Unsupported audio format (16 or 32-bit PCM expected).", "audio-format");
        return;
    }
    if (c->mode == 'w') {
        if (!c->ws) return;
        ScoreCtx x = {c, c->audio_ms};
        oww_stream_feed(c->ws, c->conv, n, on_score, &x);
        c->audio_ms += (int64_t)n * 1000 / 16000;
        return;
    }
    int max = (int)(asr_max_s * 16000);
    if (c->n_audio + n > max) {
        n = max - c->n_audio;   // (the rest is dropped: the model cannot take more)
        if (!c->cut) say("wyoming #%u: speech longer than %.0f s, the rest is not transcribed", c->id, asr_max_s);
        c->cut = 1;
    }
    if (n <= 0) return;
    if (c->n_audio + n > c->cap_audio) {
        int cap = c->cap_audio ? c->cap_audio : 16000 * 8;
        while (cap < c->n_audio + n) cap *= 2;
        int16_t* a = realloc(c->audio, sizeof(int16_t) * cap);
        if (!a) return;
        c->audio = a, c->cap_audio = cap;
    }
    memcpy(c->audio + c->n_audio, c->conv, sizeof(int16_t) * n);
    c->n_audio += n;
}

static void transcribe_now(Conn* c) {
    if (!svc_asr) {
        send_error(c, "Speech to text is not available on the console.", "no-asr");
        return;
    }
    WyJob j;
    memset(&j, 0, sizeof(j));
    j.kind = WY_ASR, j.fd = c->fd, j.audio = c->audio, j.n = c->n_audio;
    snprintf(j.lang, sizeof(j.lang), "%s", c->lang);
    say("wyoming #%u: transcribing %.2f s of audio%s%s", c->id, c->n_audio / 16000.0, c->lang[0] ? ", language " : "",
        c->lang);
    run_job(&j);
    if (j.error[0]) {
        send_error(c, j.error, "asr-failed");
    } else {
        Sb d;
        sb_init(&d);
        const char* t = j.text ? j.text : "";
        sb_str(&d, "{\"text\":");
        sb_json_str(&d, t, strlen(t));
        if (c->lang[0]) {
            sb_str(&d, ",\"language\":");
            sb_json_str(&d, c->lang, strlen(c->lang));
        }
        sb_str(&d, "}");
        if (d.p) send_event(c, "transcript", d.p);
        sb_free(&d);
    }
    free(j.text);
    c->n_audio = 0;
}

static void speak_now(Conn* c, const char* text, const char* voice) {
    if (!svc_tts) {
        send_error(c, "Text to speech is not available on the console.", "no-tts");
        return;
    }
    WyJob j;
    memset(&j, 0, sizeof(j));
    j.kind = WY_TTS, j.fd = c->fd;
    j.text = (char*)text;
    const char* lang = wy_voice_lang(voice);
    snprintf(j.lang, sizeof(j.lang), "%s", lang ? lang : "");
    say("wyoming #%u: speaking %zu bytes%s%s", c->id, strlen(text), lang ? " with voice " : "", lang ? voice : "");
    run_job(&j);   // (the main thread writes audio-start, the chunks and audio-stop)
    if (j.error[0]) send_error(c, j.error, "tts-failed");
}

static void handle_event(Conn* c) {
    const char* t = c->ev.type;
    if (!strcmp(t, "describe")) {
        send_info(c);
    } else if (!strcmp(t, "ping")) {
        char text[128];
        wy_get_str(&c->ev, "text", text, sizeof(text));
        Sb d;
        sb_init(&d);
        sb_str(&d, "{\"text\":");
        sb_json_str(&d, text, strlen(text));
        sb_str(&d, "}");
        if (d.p) send_event(c, "pong", d.p);
        sb_free(&d);
    } else if (!strcmp(t, "transcribe")) {
        c->mode = WY_ASR;
        wy_get_str(&c->ev, "language", c->lang, sizeof(c->lang));
    } else if (!strcmp(t, "detect")) {
        on_detect(c);
    } else if (!strcmp(t, "audio-start")) {
        audio_start(c);
    } else if (!strcmp(t, "audio-chunk")) {
        audio_chunk(c);
    } else if (!strcmp(t, "audio-stop")) {
        if (c->mode == 'w') {
            if (!c->detected) send_event(c, "not-detected", NULL);
        } else {
            transcribe_now(c);
        }
    } else if (!strcmp(t, "synthesize")) {
        if (c->tts_stream) return;   // (sent inside a stream for older servers: the chunks have the same text)
        char voice[48];
        wy_get_sub_str(&c->ev, "voice", "name", voice, sizeof(voice));
        Sb text;
        sb_init(&text);
        const JNode* pool;
        int k = wy_get(&c->ev, "text", &pool);
        if (k >= 0) json_str_sb(pool, k, &text);
        if (text.p && text.n) speak_now(c, text.p, voice);
        else send_error(c, "Nothing to say.", "no-text");
        sb_free(&text);
    } else if (!strcmp(t, "synthesize-start")) {
        c->tts_stream = 1;
        wy_get_sub_str(&c->ev, "voice", "name", c->voice, sizeof(c->voice));
        sb_free(&c->tts_text);
    } else if (!strcmp(t, "synthesize-chunk")) {
        const JNode* pool;
        int k = wy_get(&c->ev, "text", &pool);
        if (c->tts_stream && k >= 0 && c->tts_text.n < (1u << 16)) json_str_sb(pool, k, &c->tts_text);
    } else if (!strcmp(t, "synthesize-stop")) {
        if (c->tts_stream && c->tts_text.p && c->tts_text.n) speak_now(c, c->tts_text.p, c->voice);
        c->tts_stream = 0;
        sb_free(&c->tts_text);
        send_event(c, "synthesize-stopped", NULL);
    }
    // (other events are not for this server: ignored, as the Python servers do)
}

static void* conn_main(void* arg) {
    Conn* c = arg;
    struct {
        long sec, usec;
    } rt = {WY_IDLE_S, 0}, st = {20, 0};
    sceNetSetsockopt(c->fd, WY_SOL_SOCKET, WY_SO_RCVTIMEO, &rt, sizeof(rt));
    sceNetSetsockopt(c->fd, WY_SOL_SOCKET, WY_SO_SNDTIMEO, &st, sizeof(st));
    for (int i = 0; i < OWW_MAX_WORDS; i++) c->last_hit_ms[i] = -1;
    while (on) {
        if (c->len == c->cap) {
            size_t cap = c->cap ? c->cap * 2 : 65536;
            if (cap > WY_MAX_LINE + WY_MAX_DATA + WY_MAX_PAYLOAD + 65536) break;   // (wy_parse would have refused it)
            char* nb = realloc(c->buf, cap);
            if (!nb) break;
            c->buf = nb, c->cap = cap;
        }
        int r = sceNetRecv(c->fd, c->buf + c->len, c->cap - c->len, 0);
        if (r <= 0) break;
        c->len += (size_t)r;
        long used;
        size_t off = 0;
        while ((used = wy_parse(c->buf + off, c->len - off, &c->ev)) > 0) {
            handle_event(c);
            off += (size_t)used;
        }
        if (used < 0) {
            say("wyoming #%u: malformed event, closing", c->id);
            break;
        }
        if (off) memmove(c->buf, c->buf + off, c->len - off), c->len -= off;
    }
    sceNetSocketClose(c->fd);
    oww_stream_free(c->ws);
    free(c->audio);
    free(c->buf);
    sb_free(&c->tts_text);
    unsigned id = c->id;
    free(c);
    scePthreadMutexLock(&wy_mu);
    n_conn--;
    scePthreadMutexUnlock(&wy_mu);
    say("wyoming #%u: closed", id);
    return NULL;
}

// ---- the listening socket -------------------------------------------------------------------------------------------

static void close_if_off(void) {
    scePthreadMutexLock(&wy_mu);
    if (!on && !accepting && listen_fd >= 0) {
        sceNetSocketClose(listen_fd);
        listen_fd = -1;
    }
    scePthreadMutexUnlock(&wy_mu);
}

static void* wy_main(void* arg) {
    (void)arg;
    for (;;) {
        scePthreadMutexLock(&wy_mu);
        OrbisNetId l = on ? listen_fd : -1;
        accepting = l >= 0;
        scePthreadMutexUnlock(&wy_mu);
        if (l < 0) {
            sceKernelUsleep(100000);
            continue;
        }
        WySockaddrIn peer;
        OrbisNetSocklen_t plen = sizeof(peer);
        int fd = sceNetAccept(l, (OrbisNetSockaddr*)&peer, &plen);
        accepting = 0;
        if (fd >= 0) {
            Conn* c = NULL;
            scePthreadMutexLock(&wy_mu);
            int room = on && n_conn < WY_MAX_CONN;
            if (room) n_conn++;
            scePthreadMutexUnlock(&wy_mu);
            if (room) c = calloc(1, sizeof(Conn));
            OrbisPthread th;
            OrbisPthreadAttr attr;
            scePthreadAttrInit(&attr);
            scePthreadAttrSetstacksize(&attr, 256 << 10);
            if (c) {
                c->fd = fd, c->id = ++conn_count, c->mode = WY_ASR;
                sb_init(&c->tts_text);
                const uint8_t* b = (const uint8_t*)&peer.addr;
                say("wyoming #%u: client %u.%u.%u.%u", c->id, b[0], b[1], b[2], b[3]);
            }
            if (c && scePthreadCreate(&th, &attr, conn_main, c, "garlic_wyoming") == 0) {
                scePthreadDetach(th);
            } else {
                if (c) free(c);
                if (room) {
                    scePthreadMutexLock(&wy_mu);
                    n_conn--;
                    scePthreadMutexUnlock(&wy_mu);
                }
                say("wyoming: connection refused (%s)", room ? "no memory or thread" : "too many clients");
                sceNetSocketClose(fd);
            }
        }
        close_if_off();
        if (fd < 0) sceKernelUsleep(100000);
    }
    return NULL;
}

static OrbisNetId wy_listen(void) {
    OrbisNetId s = sceNetSocket("garlic_wyoming", ORBIS_NET_AF_INET, ORBIS_NET_SOCK_STREAM, WY_IPPROTO_TCP);
    if (s < 0) {
        say("wyoming: no socket (%d)", s);
        return -1;
    }
    int one = 1;
    sceNetSetsockopt(s, WY_SOL_SOCKET, WY_SO_REUSEADDR, &one, sizeof(one));
    WySockaddrIn a;
    memset(&a, 0, sizeof(a));
    a.len = sizeof(a);
    a.family = ORBIS_NET_AF_INET;
    a.port = sceNetHtons(WY_PORT);
    if (sceNetBind(s, (const OrbisNetSockaddr*)&a, sizeof(a)) || sceNetListen(s, 8)) {
        say("wyoming: cannot listen on port %d", WY_PORT);
        sceNetSocketClose(s);
        return -1;
    }
    return s;
}

int wy_enable(int want) {
    if (!want) {
        if (!on) return 0;
        scePthreadMutexLock(&wy_mu);
        OrbisNetId l = listen_fd;
        on = 0;
        scePthreadMutexUnlock(&wy_mu);
        if (l >= 0) sceNetSocketAbort(l, 0);
        for (int i = 0; i < 40 && accepting; i++) sceKernelUsleep(5000);
        close_if_off();
        say("wyoming: off (connections close as their clients send again)");
        return 0;
    }
    if (on) return 0;
    if (api_net_setup()) return -1;
    if (!mu_ready) {
        scePthreadMutexInit(&wy_mu, NULL, "garlic_wyoming");
        mu_ready = 1;
    }
    if (!oww) {   // the wake word models: 4.7 MiB of system memory, kept from the first time on
        size_t n = 0;
        char err[96];
        oww_data = read_all(WY_ASSETS "/oww.bin", &n);
        oww = oww_data ? oww_load(oww_data, n, err, sizeof(err)) : NULL;
        if (!oww) {
            say("wyoming: no wake words (%s)", oww_data ? err : "oww.bin missing or no memory");
            free(oww_data);
            oww_data = NULL;
        }
    }
    scePthreadMutexLock(&wy_mu);
    OrbisNetId l = listen_fd;
    if (l < 0) l = listen_fd = wy_listen();
    if (l >= 0) on = 1;
    scePthreadMutexUnlock(&wy_mu);
    if (l < 0) return -1;
    if (!have_thread) {
        OrbisPthreadAttr attr;
        scePthreadAttrInit(&attr);
        scePthreadAttrSetstacksize(&attr, 128 << 10);
        if (scePthreadCreate(&wy_th, &attr, wy_main, NULL, "garlic_wyoming_listen")) {
            say("wyoming: no thread; the server stays off");
            on = 0;
            close_if_off();
            return -1;
        }
        have_thread = 1;
    }
    say("wyoming: server on port %d (speech to text %s, text to speech %s, %d wake words)", WY_PORT,
        svc_asr ? "on" : "off", svc_tts ? "on" : "off", oww_count(oww));
    return 0;
}
