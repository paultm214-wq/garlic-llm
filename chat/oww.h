// Wake word detection: openWakeWord's models on the CPU (assets/oww.bin, written by pc/convert_openwakeword.py).
// The stream works as pyopen_wakeword's (the library wyoming-openwakeword uses): 16 kHz audio -> 8 log-mel frames
// every 80 ms -> one embedding of 96 features per 80 ms (from the last 76 frames) -> each wake word's classifier on
// the last 16 embeddings. A stream starts with 8 s of silence already heard, as pyopen_wakeword's does (that part is
// computed once per model set, not per stream). No console calls: the same code is tested on Linux
// (tests/oww_test.c). The models are read-only once loaded, so any number of streams in any threads may share them.
#ifndef CHAT_OWW_H
#define CHAT_OWW_H

#include <stddef.h>
#include <stdint.h>

#define OWW_MAX_WORDS 16

typedef struct OwwModels OwwModels;
typedef struct OwwStream OwwStream;

// Parses oww.bin (data must stay alive while the models are used). NULL on error, with the reason in err.
OwwModels* oww_load(const uint8_t* data, size_t size, char* err, size_t errcap);
void oww_free(OwwModels* m);
int oww_count(const OwwModels* m);
const char* oww_name(const OwwModels* m, int i);     // "okay_nabu"
const char* oww_phrase(const OwwModels* m, int i);   // "Okay Nabu"
// The wake word called name: its own name, or an older one ("ok_nabu", "hey_jarvis_v0.1"). -1 if none.
int oww_find(const OwwModels* m, const char* name);

OwwStream* oww_stream_new(const OwwModels* m);
void oww_stream_free(OwwStream* s);
// Which wake words this stream listens for (all off at first).
void oww_stream_enable(OwwStream* s, int word, int on);
// A new stream of audio: what was heard is forgotten (8 s of silence again).
void oww_stream_reset(OwwStream* s);
// 16 kHz mono 16-bit samples. Every classifier result calls score(user, word, probability), in the order of the audio.
typedef void (*OwwScoreFn)(void* user, int word, float probability);
void oww_stream_feed(OwwStream* s, const int16_t* pcm, int n, OwwScoreFn score, void* user);

#endif
