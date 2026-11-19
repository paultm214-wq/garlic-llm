// Home Assistant's speech jobs (wyoming.c queues them; included in main.c after voice_tts.inc, so it sees the engine,
// the voice and the chat). A job runs on the GPU between the chat's turns, as the API's requests do:
//   WY_ASR  Parakeet: the 16 kHz samples Home Assistant sent -> job->text
//   WY_TTS  Chatterbox: job->text, cut into pieces as voice mode cuts a reply (speech_text.c), each piece through T3
//           and S3Gen; every part of the vocoder's output goes to Home Assistant as soon as it is made (24 kHz mono,
//           16-bit), so it can start playing before the reply is all spoken.
// The voice needs T3's weights (which the MTP head borrows outside voice mode) and S3Gen's (in the second half of
// the context): the first spoken reply sets the memory up as voice mode does (~4 s), and it stays so until the
// server goes off (srv_voice_leave). The language model then runs without the MTP head, as in voice mode.

static int srv_voice;   // the memory is laid out for the voice because of the server

// T3's weights back and S3Gen loaded (only what is missing: after the first time, nothing). 0, or -1 if the voice
// cannot be loaded.
static int srv_voice_enter(void) {
    if (!tts_ready) return -1;
#ifdef GARLIC_SPEC
    if (spec_lent) {   // the MTP head had T3's weights memory: T3 reads its weights again (~3.5 s)
        double t0 = now();
        api_cache_drop();   // (the API's kept state was read with the head)
        spec_unload(&eng);
        spec_lent = 0;
        int bad = t3_reload(&eng);
        say("wyoming: T3's weights read again in %.2f s (%s)", now() - t0, bad ? "FAILED" : "ok");
        if (bad) return -1;
    }
#endif
    int had = tts_loaded && eng.s3.loaded;
    if (!tts_enter()) return -1;   // (S3Gen: loaded once, then only the context's second half lent again)
    if (!had && api_seq_n > (int)eng.lend_ctx) api_cache_drop();   // (S3Gen's weights went where those positions were)
    srv_voice = 1;
    return 0;
}

// The server is off: the memory back as the text chat has it (the MTP head in T3's memory).
static void srv_voice_leave(void) {
    if (!srv_voice) return;
    srv_voice = 0;
    tts_leave();
#ifdef GARLIC_SPEC
    spec_lend();
#endif
}

typedef struct {
    int fd, gone;
    int16_t s16[4096];
} WyOut;

// 24 kHz float samples to the client as audio-chunk events, at the voice's loudness (S3Voice.gain), clipped.
static void wy_send_pcm(WyOut* o, const float* pcm, int n, float gain) {
    for (int i0 = 0; i0 < n && !o->gone; i0 += 4096) {
        int k = n - i0 < 4096 ? n - i0 : 4096;
        for (int i = 0; i < k; i++) {
            float v = pcm[i0 + i] * gain;
            v = v > 0.99f ? 0.99f : v < -0.99f ? -0.99f : v;
            o->s16[i] = (int16_t)lrintf(v * 32767.0f);
        }
        Sb ev;
        sb_init(&ev);
        wy_audio_chunk(&ev, 24000, o->s16, k);
        if (!ev.p || ev.oom || api_send(o->fd, ev.p, ev.n)) o->gone = 1;
        sb_free(&ev);
    }
}

// One piece of text (a sentence or a few short ones) spoken: T3, then S3Gen, its audio sent as it comes. lang: the
// voice's language, or "" to guess it from the words. Returns -1 on an engine error (eng.error).
static int wy_speak_piece(WyOut* o, const char* text, int n, const char* lang, const char* fallback) {
    static char words[1024];
    static uint16_t toks[512], speech[512];
    static uint32_t seed_tts = 1000;
    int wl = st_clean(text, n, words, sizeof(words) - 2);
    if (!wl) return 0;   // nothing to say (code, emoji)
    const char* l = lang[0] ? lang : st_lang(words, fallback);
    wl = st_speak(words, wl, (int)sizeof(words) - 2, l);   // numbers, abbreviations, acronyms as words
    if (!wl) return 0;
    // a colon as a comma (T3 can read a colon aloud), except between digits (10:30)
    for (char* c = words; *c; c++)
        if (*c == ':' && !(c > words && c[-1] >= '0' && c[-1] <= '9' && c[1] >= '0' && c[1] <= '9')) *c = ',';
    int nt = ttstok_encode(&tts_tok, words, l, toks, 400);
    int max_out = (int)strlen(words) * 3 + 40;   // T3 ends the sentence itself; this only bounds a runaway
    if (max_out > 450) max_out = 450;
    t3_voice(&eng, l);
    s3g_voice(&eng, l);
    double t0 = now();
    if (t3_start(&eng, toks, nt, max_out, ++seed_tts, 0)) return -1;
    int ns = 0;
    for (;;) {
        int t = t3_next(&eng);
        if (t == -2) return -1;
        if (t < 0) break;
        if (ns < 512) speech[ns++] = (uint16_t)t;
    }
    // T3 ends a piece with 8-12 silence tokens: keep 7 (0.28 s), as voice mode does after a sentence
    static const uint16_t sil[6] = {6324, 6405, 4137, 4218, 6081, 6082};
    int run = 0;
    for (; run < ns; run++) {
        int is = 0;
        for (int q = 0; q < 6; q++) is |= speech[ns - 1 - run] == sil[q];
        if (!is) break;
    }
    if (run > 7) ns -= run - 7;
    if (ns < 2) return 0;
    double t1 = now();
    if (s3g_begin(&eng, speech, ns, seed_tts) <= 0) return -1;
    int samples = 0;
    for (;;) {
        int r = s3g_step(&eng, tts_pcm, (int)(sizeof(tts_pcm) / sizeof(tts_pcm[0])));
        if (r == -2) return -1;
        if (r == -1) break;
        if (r > 0) wy_send_pcm(o, tts_pcm, r, eng.s3.gain), samples += r;
        if (o->gone) break;
    }
    say("wyoming: [%s] %d speech tokens in %.2f s, %.2f s of audio in %.2f s: %.100s", l, ns, t1 - t0,
        samples / 24000.0, now() - t1, words);
    return 0;
}

static void wy_tts_job(WyJob* j) {
    if (srv_voice_enter()) {
        snprintf(j->error, sizeof(j->error), "The voice could not be loaded on the console.");
        return;
    }
    const char* t = j->text;
    int len = (int)strlen(t);
    // the language: the voice's, else what the whole text looks like (the console's language if unclear)
    const char* fallback = j->lang[0] ? j->lang : st_text_lang(t, len, lang_code());
    WyOut o = {j->fd, 0, {0}};
    Sb ev;
    sb_init(&ev);
    wy_audio_start(&ev, 24000);
    if (!ev.p || api_send(o.fd, ev.p, ev.n)) o.gone = 1;
    sb_free(&ev);
    double t0 = now();
    for (int from = 0; from < len && !o.gone;) {
        int end = st_next_piece(t, len, from, 1);
        if (end <= from) break;
        // short sentences go together (fewer pieces sound better), in one language and up to ~150 bytes as said
        while (end < len) {
            int e2 = st_next_piece(t, len, end, 1);
            if (e2 <= end || st_spoken_len(t + from, e2 - from) > 150) break;
            if (!j->lang[0] &&
                strcmp(st_text_lang(t + from, end - from, fallback), st_text_lang(t + end, e2 - end, fallback)))
                break;
            end = e2;
        }
        if (wy_speak_piece(&o, t + from, end - from, j->lang, fallback)) {
            say("wyoming: the voice failed (%s)", eng.error);
            if (!o.gone) snprintf(j->error, sizeof(j->error), "Text to speech failed: %s", eng.error);
            return;
        }
        from = end;
    }
    sb_init(&ev);
    wy_audio_stop(&ev);
    if (ev.p && !o.gone) api_send(o.fd, ev.p, ev.n);
    sb_free(&ev);
    say("wyoming: spoken in %.2f s%s", now() - t0, o.gone ? " (the client went away)" : "");
}

static void wy_asr_job(WyJob* j) {
    asr_wait();   // (after a Python program the dictation weights are read back first)
    if (!eng.asr.ready) {
        snprintf(j->error, sizeof(j->error), "Speech to text is not available on the console.");
        return;
    }
    j->text = calloc(1, 4096);
    if (!j->text) {
        snprintf(j->error, sizeof(j->error), "Out of memory.");
        return;
    }
    int n = j->n, max = (int)((eng.asr.max_frames * 0.08f - 0.5f) * 16000);
    if (n > max) n = max;
    if (n < 16000 / 10) return;   // (under 0.1 s: nothing was said)
    float* a = malloc(sizeof(float) * n);
    if (!a) {
        snprintf(j->error, sizeof(j->error), "Out of memory.");
        return;
    }
    for (int i = 0; i < n; i++) a[i] = j->audio[i] / 32768.0f;
    double t0 = now();
    int r = asr_transcribe(&eng, a, n, j->text, 4095);
    free(a);
    if (r) {
        snprintf(j->error, sizeof(j->error), "Speech to text failed: %s", eng.error);
        return;
    }
    char* s = j->text;   // trimmed
    while (*s == ' ') s++;
    memmove(j->text, s, strlen(s) + 1);
    for (size_t k = strlen(j->text); k > 0 && j->text[k - 1] == ' '; k--) j->text[k - 1] = 0;
    say("wyoming: %.1f s of speech in %.2f s (GPU %.2f s): %s", n / 16000.0, now() - t0, eng.last_gpu_seconds, j->text);
}

// A job of the Wyoming server, on the main loop. The chat's screen shows that the server is busy.
static void wy_job(WyJob* j) {
    api_busy = 1;
    show_chat();
    if (j->kind == WY_ASR) wy_asr_job(j);
    else wy_tts_job(j);
    api_busy = 0;
}
