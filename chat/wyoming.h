// Wyoming server on the console (port WY_PORT, on the LAN): the speech services Home Assistant uses, from one
// address. Speech to text is Parakeet, text to speech is Chatterbox (Garlic's own voice models, on the GPU) and wake
// words are openWakeWord's (oww.c, on the CPU). Home Assistant's Wyoming integration finds all three behind one
// host and port.
// A connection thread per client reads its events (wyoming_proto.h). Wake words are answered there; speech to text
// and text to speech go to the main loop as jobs, one at a time and in order (wyoming_run.inc runs them on the
// GPU between the chat's turns, as the API's requests).
// The server is off until wy_enable turns it on; it goes on and off with the API (L3 on the chat screen, main.c).
#ifndef CHAT_WYOMING_H
#define CHAT_WYOMING_H

#include <stdint.h>

#define WY_PORT 10300

enum { WY_ASR = 1, WY_TTS };

typedef struct WyJob {
    int kind;                   // WY_ASR or WY_TTS
    int fd;                     // the connection (WY_TTS: the main thread writes the audio events to it)
    int16_t* audio;             // WY_ASR: 16 kHz mono samples
    int n;
    char lang[8];               // WY_ASR: the language asked for; WY_TTS: the voice's ("" = from the text)
    char* text;                 // WY_TTS: what to say; WY_ASR: the transcript (malloc'd by the main thread)
    char error[128];            // set if the job failed
    volatile int done;          // set by wy_done
    struct WyJob* next;
} WyJob;

// What the console can do: speech to text (and its longest audio, seconds), text to speech. Set before wy_enable.
void wy_services(int asr, float asr_max_s, int tts);
// Turns the server on (listens on WY_PORT; loads the wake word models the first time) or off. 0, or -1 (logged).
int wy_enable(int on);
int wy_enabled(void);
// Main thread: the next job, or NULL. When it is finished (and for WY_TTS, its audio written), wy_done(job).
WyJob* wy_take(void);
void wy_done(WyJob* job);
// The voices that text to speech offers, with their languages ("garlic-en" -> "en"); NULL if the name is not one.
const char* wy_voice_lang(const char* voice);

#endif
