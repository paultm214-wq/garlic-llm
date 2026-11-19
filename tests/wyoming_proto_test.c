// Host tests for chat/wyoming_proto.c (the Wyoming protocol: events, info, audio conversion).
// Build and run: gcc -std=gnu99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/wyoming_proto_test.c chat/wyoming_proto.c chat/api_proto.c -o build/wyoming_proto_test && build/wyoming_proto_test
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wyoming_proto.h"

static int fails, checks;
#define CHECK(c)                                                         \
    do {                                                                 \
        checks++;                                                        \
        if (!(c)) {                                                      \
            fails++;                                                     \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);          \
        }                                                                \
    } while (0)

static WyEvent ev;

static void test_parse(void) {
    // the Python library's framing: data in its own JSON after the line, then the payload
    const char data[] = "{\"rate\":16000,\"width\":2,\"channels\":1,\"timestamp\":64}";
    char buf[512];
    int n = snprintf(buf, sizeof(buf), "{\"type\":\"audio-chunk\",\"version\":\"1.11.0\",\"data_length\":%d,"
                     "\"payload_length\":4}\n%sABCD{\"type\":\"audio-stop\"}\n", (int)strlen(data), data);
    size_t first = strchr(buf, '\n') - buf + 1 + strlen(data) + 4;
    for (size_t k = 0; k < first; k++) CHECK(wy_parse(buf, k, &ev) == 0);   // any prefix: wait for more
    CHECK(wy_parse(buf, (size_t)n, &ev) == (long)first);
    CHECK(!strcmp(ev.type, "audio-chunk"));
    CHECK(wy_get_int(&ev, "rate", 0) == 16000 && wy_get_int(&ev, "timestamp", 0) == 64);
    CHECK(wy_get_int(&ev, "missing", -7) == -7);
    CHECK(ev.payload_len == 4 && !memcmp(ev.payload, "ABCD", 4));
    CHECK(wy_parse(buf + first, (size_t)n - first, &ev) == (long)((size_t)n - first));
    CHECK(!strcmp(ev.type, "audio-stop") && ev.payload_len == 0);

    // data inline in the header (older clients), and merged data winning over it
    const char* inl = "{\"type\":\"transcribe\",\"data\":{\"language\":\"es\",\"name\":\"x\"}}\n";
    CHECK(wy_parse(inl, strlen(inl), &ev) == (long)strlen(inl));
    char s[32];
    wy_get_str(&ev, "language", s, sizeof(s));
    CHECK(!strcmp(s, "es"));
    const char* extra = "{\"text\":\"caf\\u00e9 \\\"ok\\\"\",\"voice\":{\"name\":\"garlic-fr\"}}";
    char both[256];
    snprintf(both, sizeof(both), "{\"type\":\"synthesize\",\"data\":{\"text\":\"old\"},\"data_length\":%d}\n%s",
             (int)strlen(extra), extra);
    CHECK(wy_parse(both, strlen(both), &ev) == (long)strlen(both));
    wy_get_str(&ev, "text", s, sizeof(s));
    CHECK(!strcmp(s, "caf\xc3\xa9 \"ok\""));
    wy_get_sub_str(&ev, "voice", "name", s, sizeof(s));
    CHECK(!strcmp(s, "garlic-fr"));
    wy_get_str(&ev, "text", s, 5);   // cut at a whole character
    CHECK(!strcmp(s, "caf"));

    CHECK(wy_parse("not json\n", 9, &ev) == -1);
    CHECK(wy_parse("{\"data\":{}}\n", 12, &ev) == -1);   // no type
    CHECK(wy_parse("{\"type\":\"x\",\"payload_length\":-1}\n", 33, &ev) == -1);
    CHECK(wy_parse("{\"type\":\"x\",\"payload_length\":99999999}\n", 39, &ev) == -1);
    char* longline = malloc(WY_MAX_LINE + 10);
    memset(longline, ' ', WY_MAX_LINE + 10);
    CHECK(wy_parse(longline, WY_MAX_LINE + 10, &ev) == -1);
    free(longline);
}

static void test_write(void) {
    Sb s;
    sb_init(&s);
    int16_t pcm[3] = {1, -2, 300};
    wy_audio_chunk(&s, 24000, pcm, 3);
    CHECK(s.p && wy_parse(s.p, s.n, &ev) == (long)s.n);
    CHECK(!strcmp(ev.type, "audio-chunk") && wy_get_int(&ev, "rate", 0) == 24000 && ev.payload_len == 6);
    CHECK(ev.payload && !memcmp(ev.payload, pcm, 6));
    sb_free(&s);
    wy_write(&s, "not-detected", NULL, NULL, 0);
    CHECK(s.p && !strcmp(s.p, "{\"type\":\"not-detected\",\"version\":\"" WY_VERSION "\"}\n"));
    sb_free(&s);
}

static void test_info(void) {
    static const char* const langs[] = {"en", "es"};
    static const WyVoice voices[] = {{"garlic-en", "Garlic (English)", "en"}};
    static const char* const names[] = {"okay_nabu"};
    static const char* const phrases[] = {"Okay Nabu"};
    WyInfo in = {1, langs, 2, 1, voices, 1, 1, names, phrases, 1, "1.20"};
    Sb s;
    sb_init(&s);
    wy_info_json(&s, &in);
    static JNode pool[512];
    int root = json_parse(s.p, s.n, pool, 512);
    CHECK(root >= 0);
    int asr = json_get(pool, root, "asr"), tts = json_get(pool, root, "tts"), wake = json_get(pool, root, "wake");
    CHECK(asr >= 0 && pool[asr].count == 1 && tts >= 0 && pool[tts].count == 1 && wake >= 0 && pool[wake].count == 1);
    int m = json_get(pool, pool[asr].first, "models");
    CHECK(m >= 0 && json_get(pool, pool[m].first, "languages") >= 0);
    int inst = json_get(pool, pool[tts].first, "installed");
    CHECK(inst >= 0 && pool[inst].type == J_BOOL && pool[inst].b);
    sb_free(&s);
    WyInfo none = {0};
    wy_info_json(&s, &none);
    CHECK(s.p && !strcmp(s.p, "{\"asr\":[],\"tts\":[],\"wake\":[],\"handle\":[],\"intent\":[],\"mic\":[],\"snd\":[]}"));
    sb_free(&s);
}

static void test_audio(void) {
    int16_t out[64];
    uint8_t mono[8] = {0x10, 0x00, 0xF0, 0xFF, 0xFF, 0x7F, 0x00, 0x80};   // 16, -16, 32767, -32768
    CHECK(wy_to_16k(mono, 8, 16000, 2, 1, out, 64) == 4);
    CHECK(out[0] == 16 && out[1] == -16 && out[2] == 32767 && out[3] == -32768);
    int16_t st[4] = {100, 300, -50, -150};   // two stereo frames -> 200, -100
    CHECK(wy_to_16k((const uint8_t*)st, 8, 16000, 2, 2, out, 64) == 2 && out[0] == 200 && out[1] == -100);
    int16_t f48[48];
    for (int i = 0; i < 48; i++) f48[i] = (int16_t)(i * 10);
    CHECK(wy_to_16k((const uint8_t*)f48, 96, 48000, 2, 1, out, 64) == 16 && out[1] == 30 && out[15] == 450);
    int32_t w4[2] = {1 << 30, -(1 << 30)};
    CHECK(wy_to_16k((const uint8_t*)w4, 8, 16000, 4, 1, out, 64) == 2 && out[0] == 16384 && out[1] == -16384);
    CHECK(wy_to_16k(mono, 8, 16000, 3, 1, out, 64) == -1);
    CHECK(wy_to_16k(mono, 8, 16000, 2, 1, out, 2) == 2);   // the output's room
}

int main(void) {
    test_parse();
    test_write();
    test_info();
    test_audio();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
