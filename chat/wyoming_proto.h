// Wyoming protocol (github.com/rhasspy/wyoming, what Home Assistant speaks to speech-to-text, text-to-speech and
// wake word services), protocol part: events in and out, the service description, audio conversion. No console
// calls, so it is tested on Linux (tests/wyoming_proto_test.c). The server is wyoming.c; the work on the GPU,
// wyoming_run.inc.
//
// An event is one line of JSON, {"type": ..., "data": {...}, "data_length": N, "payload_length": M}, then N bytes of
// JSON that are merged into data, then M bytes of payload (audio).
#ifndef CHAT_WYOMING_PROTO_H
#define CHAT_WYOMING_PROTO_H

#include <stddef.h>
#include <stdint.h>

#include "api_proto.h"

#define WY_VERSION "1.8.0"          // the protocol version our events carry
#define WY_MAX_LINE 16384           // the longest header line accepted
#define WY_MAX_DATA (256 << 10)     // the most JSON data an event may carry
#define WY_MAX_PAYLOAD (1 << 20)    // the biggest payload (audio chunks are a few KiB)
#define WY_NODES 512

typedef struct {
    char type[40];
    JNode head[64];                 // the header line
    JNode data[WY_NODES];           // the data JSON (data_length bytes), if any
    int head_root, head_data;       // head's root and its "data" member (-1 if none)
    int data_root;                  // root of data (-1 if none)
    const uint8_t* payload;         // points into the buffer given to wy_parse
    size_t payload_len;
} WyEvent;

// One event from buf (len bytes). Returns the bytes it takes (> 0), 0 if buf does not hold a whole event yet, or -1
// if it is malformed (the connection should close). The event points into buf.
long wy_parse(const char* buf, size_t len, WyEvent* ev);
// The node of the data member key (the data JSON first, as the Python library merges it, then the header's data);
// pool is set to the pool the node is in. -1 if there is none.
int wy_get(const WyEvent* ev, const char* key, const JNode** pool);
// A data member as a string (decoded, NUL-terminated, at most cap - 1 bytes) or a number; def if absent.
void wy_get_str(const WyEvent* ev, const char* key, char* out, size_t cap);
long wy_get_int(const WyEvent* ev, const char* key, long def);
// Member key of the data member obj (an object), as a string: e.g. synthesize's voice.name.
void wy_get_sub_str(const WyEvent* ev, const char* obj, const char* key, char* out, size_t cap);

// Appends one event: type, data (a JSON object's text, or NULL for none) and payload (NULL / 0 for none).
void wy_write(Sb* out, const char* type, const char* data, const void* payload, size_t n);

// The audio events of an answer: audio-start, an audio-chunk of n 16-bit samples, audio-stop (mono, rate Hz).
void wy_audio_start(Sb* out, int rate);
void wy_audio_chunk(Sb* out, int rate, const int16_t* pcm, int n);
void wy_audio_stop(Sb* out);

// What the server offers, for the info event (the answer to describe).
typedef struct {
    const char* name;               // "garlic-en"
    const char* description;        // "Garlic (English)"
    const char* language;           // "en"
} WyVoice;

typedef struct {
    int asr;                        // speech to text is offered...
    const char* const* asr_langs;   // ...in these languages
    int n_asr_langs;
    int tts;
    const WyVoice* voices;
    int n_voices;
    int wake;
    const char* const* wake_names;  // "okay_nabu"
    const char* const* wake_phrases;   // "Okay Nabu"
    int n_wake;
    const char* version;            // the app's version
} WyInfo;

void wy_info_json(Sb* out, const WyInfo* info);

// Audio from an audio-chunk (rate Hz, width bytes per sample: 2 or 4, channels) as 16 kHz mono 16-bit samples, into
// out (cap samples). Other rates are resampled linearly, each chunk on its own. Returns the number of samples, or -1
// if the format is not supported.
int wy_to_16k(const uint8_t* in, size_t n, int rate, int width, int channels, int16_t* out, int cap);

#endif
