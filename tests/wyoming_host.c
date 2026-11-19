// Host test server for chat/wyoming.c: the real server code over POSIX sockets (tests/shim/orbis), with a mock
// main loop instead of the GPU: speech to text answers "heard <seconds> s", text to speech answers 0.25 s of tone
// per 10 bytes of text. The wake words are the real ones (chat/assets/oww.bin). tests/wyoming_test.py drives it
// with the wyoming Python library, as Home Assistant does.
// Build (one line): gcc -std=gnu99 -O2 -Wall -Wextra -I tests/shim -I chat -DWY_ASSETS='"chat/assets"'
//   tests/wyoming_host.c chat/wyoming.c chat/wyoming_proto.c chat/api_proto.c chat/oww.c -lm -lpthread
//   -o build/wyoming_host
// Run:   WY_TEST_PORT=10399 build/wyoming_host      (then python tests/wyoming_test.py 10399)
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include "api.h"
#include "wyoming.h"
#include "wyoming_proto.h"

void say(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "[server] ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

uint8_t* read_all(const char* path, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* b = malloc(n > 0 ? (size_t)n : 1);
    *size = b ? fread(b, 1, (size_t)n, f) : 0;
    fclose(f);
    return b;
}

int api_send(int fd, const char* p, size_t n) {
    while (n) {
        ssize_t w = send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return -1;
        p += w, n -= (size_t)w;
    }
    return 0;
}

int api_net_setup(void) { return 0; }

static void tts(WyJob* j) {
    if (!strcmp(j->text, "fail")) {
        snprintf(j->error, sizeof(j->error), "mock failure");
        return;
    }
    fprintf(stderr, "[server] TTS job: lang '%s', text '%s'\n", j->lang, j->text);
    int total = (int)(strlen(j->text) / 10 + 1) * 6000;   // 0.25 s at 24 kHz per 10 bytes
    static int16_t pcm[2400];
    Sb s;
    sb_init(&s);
    wy_audio_start(&s, 24000);
    for (int done = 0; done < total; done += 2400) {
        int k = total - done < 2400 ? total - done : 2400;
        for (int i = 0; i < k; i++) pcm[i] = (int16_t)(8000 * sin((done + i) * 2 * M_PI * 440 / 24000));
        wy_audio_chunk(&s, 24000, pcm, k);
    }
    wy_audio_stop(&s);
    api_send(j->fd, s.p, s.n);
    sb_free(&s);
}

int main(void) {
    wy_services(1, 10.0f, 1);
    if (wy_enable(1)) return 1;
    fprintf(stderr, "[server] ready\n");
    for (;;) {
        WyJob* j = wy_take();
        if (!j) {
            usleep(2000);
            continue;
        }
        if (j->kind == WY_ASR) {
            j->text = malloc(64);
            double rms = 0;
            for (int i = 0; i < j->n; i++) rms += (double)j->audio[i] * j->audio[i];
            snprintf(j->text, 64, "heard %.2f s", j->n / 16000.0);
            fprintf(stderr, "[server] ASR job: %d samples, rms %.0f, language '%s'\n", j->n, j->n ? sqrt(rms / j->n) : 0.0,
                    j->lang);
        } else {
            tts(j);
        }
        wy_done(j);
    }
}
