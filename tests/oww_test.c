// Host test for chat/oww.c (openWakeWord on the CPU): runs chat/assets/oww.bin over a 16 kHz mono 16-bit WAV.
// Build and run:
//   gcc -std=gnu99 -O2 -Wall -Wextra -I chat tests/oww_test.c chat/oww.c -lm -o build/oww_test
//   build/oww_test speech.wav [--scores] [--chunk N] [--expect okay_nabu]
// Prints each wake word's highest score and when it first passed 0.5; --scores prints every score (to compare with
// the TFLite reference, pc/openwakeword_ref.py); --expect makes the exit code 0 only if that word, and no other, was
// detected.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "oww.h"

static uint8_t* read_file(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* d = malloc(len > 0 ? (size_t)len : 1);
    if (d && fread(d, 1, (size_t)len, f) != (size_t)len) free(d), d = NULL;
    fclose(f);
    *n = (size_t)len;
    return d;
}

static int16_t* read_wav(const char* path, int* n) {
    size_t size;
    uint8_t* d = read_file(path, &size);
    if (!d || size < 44 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) return NULL;
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    for (size_t o = 12; o + 8 <= size;) {
        uint32_t len;
        memcpy(&len, d + o + 4, 4);
        if (!memcmp(d + o, "fmt ", 4)) {
            memcpy(&fmt, d + o + 8, 2), memcpy(&ch, d + o + 10, 2), memcpy(&rate, d + o + 12, 4), memcpy(&bits, d + o + 22, 2);
        } else if (!memcmp(d + o, "data", 4)) {
            if (fmt != 1 || ch != 1 || bits != 16 || rate != 16000) {
                fprintf(stderr, "%s: need 16 kHz mono 16-bit PCM\n", path);
                return NULL;
            }
            if (len > size - o - 8) len = (uint32_t)(size - o - 8);
            *n = (int)(len / 2);
            int16_t* pcm = malloc(len + 2);
            memcpy(pcm, d + o + 8, len);
            free(d);
            return pcm;
        }
        o += 8 + len + (len & 1);
    }
    return NULL;
}

typedef struct {
    int print, calls;
    long pos;               // samples fed so far (the end of the current chunk)
    float best[OWW_MAX_WORDS];
    long first[OWW_MAX_WORDS];
} Res;

static void on_score(void* user, int w, float p) {
    Res* r = user;
    r->calls++;
    if (r->print) printf("%d %.6f\n", w, p);
    if (p > r->best[w]) r->best[w] = p;
    if (p > 0.5f && r->first[w] < 0) r->first[w] = r->pos;
}

int main(int argc, char** argv) {
    const char *wav = NULL, *expect = NULL, *bin = "chat/assets/oww.bin";
    int chunk = 1024, print = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--scores")) print = 1;
        else if (!strcmp(argv[i], "--chunk") && i + 1 < argc) chunk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--expect") && i + 1 < argc) expect = argv[++i];
        else if (!strcmp(argv[i], "--models") && i + 1 < argc) bin = argv[++i];
        else wav = argv[i];
    }
    if (!wav || chunk < 1) {
        fprintf(stderr, "usage: oww_test file.wav [--scores] [--chunk N] [--expect word] [--models oww.bin]\n");
        return 2;
    }
    size_t size;
    uint8_t* data = read_file(bin, &size);
    char err[128];
    OwwModels* m = data ? oww_load(data, size, err, sizeof(err)) : NULL;
    if (!m) {
        fprintf(stderr, "%s\n", data ? err : "cannot read oww.bin");
        return 2;
    }
    int n;
    int16_t* pcm = read_wav(wav, &n);
    if (!pcm) return 2;
    OwwStream* s = oww_stream_new(m);
    for (int i = 0; i < oww_count(m); i++) oww_stream_enable(s, i, 1);
    Res r = {print, 0, 0, {0}, {0}};
    for (int i = 0; i < OWW_MAX_WORDS; i++) r.first[i] = -1;
    clock_t t0 = clock();
    for (int i = 0; i < n; i += chunk) {
        int k = n - i < chunk ? n - i : chunk;
        r.pos = i + k;
        oww_stream_feed(s, pcm + i, k, on_score, &r);
    }
    double cpu = (double)(clock() - t0) / CLOCKS_PER_SEC;
    int ok = 1;
    for (int i = 0; i < oww_count(m); i++) {
        int hit = r.first[i] >= 0;
        fprintf(stderr, "%-12s best %.3f%s", oww_name(m, i), r.best[i], hit ? "" : "\n");
        if (hit) fprintf(stderr, ", detected at %.2f s\n", r.first[i] / 16000.0);
        if (expect && hit != (oww_find(m, expect) == i)) ok = 0;
    }
    fprintf(stderr, "%.1f s of audio, %d scores, %.2f s of CPU (%.1f%% of real time)\n", n / 16000.0, r.calls, cpu,
            100 * cpu / (n / 16000.0));
    oww_stream_free(s);
    oww_free(m);
    free(pcm);
    free(data);
    return expect && !ok ? 1 : 0;
}
