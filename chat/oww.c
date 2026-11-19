// Wake word detection with openWakeWord's models (see oww.h; the file format is described in
// pc/convert_openwakeword.py).
#include "oww.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum { OP_CONV = 1, OP_LEAKY, OP_MAXC, OP_MAXPOOL, OP_DENSE, OP_NORM, OP_SIGMOID, OP_GATE };

#define N_FFT 512
#define N_BINS (N_FFT / 2 + 1)
#define N_MELS 32
#define HOP 160
#define CALL_SAMPLES 1760   // samples per melspectrogram call (pyopen_wakeword MEL_SAMPLES) ...
#define CALL_STEP 1280      // ... every 80 ms
#define CALL_FRAMES ((CALL_SAMPLES - N_FFT) / HOP + 1)   // 8
#define EMB_FRAMES 76       // mel frames per embedding ...
#define EMB_STEP 8          // ... every 8 frames
#define EMB_FEATS 96
#define AUTOFILL (8 * 16000)   // a stream starts with 8 s of silence (pyopen_wakeword AUTOFILL_SECONDS)
#define MAX_MELS 970        // pyopen_wakeword MAX_MELS (only bounds new_mels)
#define MEL_RING 128        // frames kept (new_mels never passes EMB_FRAMES + EMB_STEP)

typedef struct {
    uint32_t type, np, nf;
    const uint8_t* p;   // np u32 params
    const float* f;     // nf floats
} Op;

typedef struct {
    Op* ops;
    int n;
} Prog;

typedef struct {
    char name[32], phrase[32];
    uint32_t windows, feats;
    Prog prog;
} Word;

struct OwwModels {
    float amin, top_db, db_scale;
    const float* fb;            // [N_BINS][N_MELS]
    Prog emb;
    int n_words;
    Word words[OWW_MAX_WORDS];
    size_t emb_buf, cls_buf;    // floats of the largest activation of the embedding / of any classifier
    float hann[N_FFT];          // periodic Hann window of 400 samples, centred
    double cs[N_FFT / 2], sn[N_FFT / 2];   // FFT twiddles
    float silence_mel[N_MELS];  // a frame of a call whose 1760 samples are all 0 (after x / 10 + 2)
    float silence_emb[EMB_FEATS];
};

static uint32_t u32(const uint8_t* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint32_t prm(const Op* o, int i) { return u32(o->p + 4 * i); }

// ---- programs: parsing and shapes ---------------------------------------------------------------------------------

typedef struct {
    const uint8_t* p;
    size_t n, i;
    int bad;
} Rd;

static const uint8_t* take(Rd* r, size_t n) {
    if (r->bad || n > r->n - r->i) {
        r->bad = 1;
        return NULL;
    }
    const uint8_t* q = r->p + r->i;
    r->i += n;
    return q;
}

static uint32_t take_u32(Rd* r) {
    const uint8_t* q = take(r, 4);
    return q ? u32(q) : 0;
}

static int read_prog(Rd* r, Prog* pg) {
    uint32_t n = take_u32(r);
    if (r->bad || n == 0 || n > 256) return -1;
    pg->ops = calloc(n, sizeof(Op));
    if (!pg->ops) return -1;
    pg->n = (int)n;
    for (uint32_t k = 0; k < n; k++) {
        Op* o = &pg->ops[k];
        o->type = take_u32(r);
        o->np = take_u32(r);
        o->nf = take_u32(r);
        if (r->bad || o->np > 16 || o->nf > (1u << 24)) return -1;
        o->p = take(r, 4u * o->np);
        o->f = (const float*)take(r, 4u * (size_t)o->nf);
        if (r->bad) return -1;
    }
    return 0;
}

// Checks the program against its input shape (h, w, c) and its own sizes; sets the output size and the largest
// activation (floats). Returns 0, or -1 if the program does not fit together.
static int prog_shapes(const Prog* pg, int h, int w, int c, int* out, size_t* biggest) {
    int h0 = h, w0 = w, c0 = c;
    size_t big = (size_t)h * w * c;
    for (int k = 0; k < pg->n; k++) {
        const Op* o = &pg->ops[k];
        switch (o->type) {
        case OP_CONV: {
            if (o->np != 10) return -1;
            int kh = prm(o, 0), kw = prm(o, 1), ci = prm(o, 2), co = prm(o, 3), sh = prm(o, 4), sw = prm(o, 5);
            int pt = prm(o, 6), pb = prm(o, 7), pl = prm(o, 8), pr = prm(o, 9);
            if (ci != c || sh < 1 || sw < 1 || (size_t)o->nf != (size_t)kh * kw * ci * co + co) return -1;
            h = (h + pt + pb - kh) / sh + 1, w = (w + pl + pr - kw) / sw + 1, c = co;
            break;
        }
        case OP_MAXPOOL: {
            if (o->np != 8) return -1;
            int fh = prm(o, 0), fw = prm(o, 1), sh = prm(o, 2), sw = prm(o, 3);
            if (sh < 1 || sw < 1) return -1;
            h = (h + prm(o, 4) + prm(o, 5) - fh) / sh + 1, w = (w + prm(o, 6) + prm(o, 7) - fw) / sw + 1;
            break;
        }
        case OP_LEAKY:
        case OP_MAXC:
        case OP_GATE:
            if (o->nf != 1) return -1;
            if (o->type == OP_GATE) {
                if (h * w * c != 1) return -1;
                h = h0, w = w0, c = c0;   // (the rest runs on the input)
            }
            break;
        case OP_DENSE:
            if (o->np != 3 || (int)prm(o, 0) != h * w * c || (size_t)o->nf != (size_t)prm(o, 0) * prm(o, 1) + prm(o, 1))
                return -1;
            h = w = 1, c = prm(o, 1);
            break;
        case OP_NORM:
            if (o->np != 2 || (int)prm(o, 0) != h * w * c || o->nf != 1 + 2 * prm(o, 0)) return -1;
            break;
        case OP_SIGMOID: break;
        default: return -1;
        }
        if (h < 1 || w < 1 || c < 1) return -1;
        if ((size_t)h * w * c > big) big = (size_t)h * w * c;
    }
    *out = h * w * c;
    *biggest = big;
    return 0;
}

// ---- programs: running -------------------------------------------------------------------------------------------

static void conv(const Op* o, const float* restrict in, int h, int w, int c, float* restrict out, int* ho, int* wo) {
    int kh = prm(o, 0), kw = prm(o, 1), co = prm(o, 3), sh = prm(o, 4), sw = prm(o, 5);
    int pt = prm(o, 6), pb = prm(o, 7), pl = prm(o, 8), pr = prm(o, 9);
    int oh = (h + pt + pb - kh) / sh + 1, ow = (w + pl + pr - kw) / sw + 1;
    const float* wt = o->f;                                   // [kh][kw][cin][cout]
    const float* bias = o->f + (size_t)kh * kw * c * co;
    // Eight output channels at a time (every layer has a multiple of 8): their sums stay in one vector register
    // while all the inputs go by (a = sum of x * W[.., j0..j0+7], the inner loop over the 8 channels).
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            for (int j0 = 0; j0 < co; j0 += 8) {
                int nj = co - j0 < 8 ? co - j0 : 8;
                float a[8] = {0};
                for (int i = 0; i < kh; i++) {
                    int iy = y * sh - pt + i;
                    if (iy < 0 || iy >= h) continue;
                    for (int k = 0; k < kw; k++) {
                        int ix = x * sw - pl + k;
                        if (ix < 0 || ix >= w) continue;
                        const float* xin = in + ((size_t)iy * w + ix) * c;
                        const float* wk = wt + ((size_t)i * kw + k) * c * co + j0;
                        if (nj == 8)
                            for (int ci = 0; ci < c; ci++) {
                                float v = xin[ci];
                                const float* wr = wk + (size_t)ci * co;
                                for (int t = 0; t < 8; t++) a[t] += v * wr[t];
                            }
                        else
                            for (int ci = 0; ci < c; ci++)
                                for (int t = 0; t < nj; t++) a[t] += xin[ci] * wk[(size_t)ci * co + t];
                    }
                }
                float* dst = out + ((size_t)y * ow + x) * co + j0;
                for (int t = 0; t < nj; t++) dst[t] = a[t] + bias[j0 + t];
            }
    *ho = oh, *wo = ow;
}

static void maxpool(const Op* o, const float* in, int h, int w, int c, float* out, int* ho, int* wo) {
    int fh = prm(o, 0), fw = prm(o, 1), sh = prm(o, 2), sw = prm(o, 3), pt = prm(o, 4), pl = prm(o, 6);
    int oh = (h + pt + prm(o, 5) - fh) / sh + 1, ow = (w + pl + prm(o, 7) - fw) / sw + 1;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            for (int j = 0; j < c; j++) {
                float m = -INFINITY;
                for (int i = 0; i < fh; i++)
                    for (int k = 0; k < fw; k++) {
                        int iy = y * sh - pt + i, ix = x * sw - pl + k;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                        float v = in[((size_t)iy * w + ix) * c + j];
                        if (v > m) m = v;
                    }
                out[((size_t)y * ow + x) * c + j] = m;
            }
    *ho = oh, *wo = ow;
}

// Runs a program on x (h, w, c) with two work buffers big enough for its largest activation. Returns the output
// (in a or b; its size is the one prog_shapes gave).
static const float* run(const Prog* pg, const float* x, int h, int w, int c, float* a, float* b) {
    const float* cur = x;
    int h0 = h, w0 = w, c0 = c;
    for (int k = 0; k < pg->n; k++) {
        const Op* o = &pg->ops[k];
        float* dst = cur == a ? b : a;
        size_t n = (size_t)h * w * c;
        switch (o->type) {
        case OP_CONV:
            conv(o, cur, h, w, c, dst, &h, &w);
            c = prm(o, 3);
            cur = dst;
            break;
        case OP_MAXPOOL:
            maxpool(o, cur, h, w, c, dst, &h, &w);
            cur = dst;
            break;
        case OP_LEAKY: {
            float al = o->f[0];
            for (size_t i = 0; i < n; i++) dst[i] = cur[i] < 0 ? cur[i] * al : cur[i];
            cur = dst;
            break;
        }
        case OP_MAXC: {
            float m = o->f[0];
            for (size_t i = 0; i < n; i++) dst[i] = cur[i] > m ? cur[i] : m;
            cur = dst;
            break;
        }
        case OP_DENSE: {
            int ni = prm(o, 0), no = prm(o, 1), relu = prm(o, 2) == 1;
            const float* wt = o->f;   // [in][out]
            const float* bias = o->f + (size_t)ni * no;
            for (int j = 0; j < no; j++) dst[j] = bias[j];
            for (int i = 0; i < ni; i++) {
                float v = cur[i];
                const float* restrict wr = wt + (size_t)i * no;
                float* restrict d = dst;
                for (int j = 0; j < no; j++) d[j] += v * wr[j];
            }
            if (relu)
                for (int j = 0; j < no; j++) dst[j] = dst[j] > 0 ? dst[j] : 0;
            h = w = 1, c = no;
            cur = dst;
            break;
        }
        case OP_NORM: {
            int nn = prm(o, 0), relu = prm(o, 1) == 1;
            float eps = o->f[0];
            const float *g = o->f + 1, *be = o->f + 1 + nn;
            double mean = 0, var = 0;
            for (int i = 0; i < nn; i++) mean += cur[i];
            mean /= nn;
            for (int i = 0; i < nn; i++) var += (cur[i] - mean) * (cur[i] - mean);
            var /= nn;
            float inv = (float)(1.0 / sqrt(var + eps));
            for (int i = 0; i < nn; i++) {
                float v = (float)(cur[i] - mean) * inv * g[i] + be[i];
                dst[i] = relu && v < 0 ? 0 : v;
            }
            cur = dst;
            break;
        }
        case OP_SIGMOID:
            for (size_t i = 0; i < n; i++) dst[i] = 1.0f / (1.0f + expf(-cur[i]));
            cur = dst;
            break;
        case OP_GATE:   // a score under the threshold is the result; else the rest rechecks the input
            if (cur[0] < o->f[0]) return cur;
            cur = x, h = h0, w = w0, c = c0;
            break;
        }
    }
    return cur;
}

// ---- melspectrogram -------------------------------------------------------------------------------------------------

static void fft512(const OwwModels* m, double* re, double* im) {
    for (int i = 1, j = 0; i < N_FFT; i++) {   // bit reversal
        int bit = N_FFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            double t = re[i];
            re[i] = re[j], re[j] = t;
            t = im[i], im[i] = im[j], im[j] = t;
        }
    }
    for (int len = 2; len <= N_FFT; len <<= 1) {
        int half = len >> 1, step = N_FFT / len;
        for (int i = 0; i < N_FFT; i += len)
            for (int k = 0; k < half; k++) {
                double wr = m->cs[k * step], wi = -m->sn[k * step];
                int a = i + k, b = a + half;
                double tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr, im[b] = im[a] - ti;
                re[a] += tr, im[a] += ti;
            }
    }
}

// The 8 frames of one call (1760 samples, int16 scale) into out[8][32], after x / 10 + 2.
static void mel_call(const OwwModels* m, const float* audio, float* out) {
    double re[N_FFT], im[N_FFT];
    float top = -INFINITY;
    for (int f = 0; f < CALL_FRAMES; f++) {
        const float* x = audio + f * HOP;
        for (int i = 0; i < N_FFT; i++) re[i] = (double)x[i] * m->hann[i], im[i] = 0;
        fft512(m, re, im);
        float mel[N_MELS] = {0};
        for (int k = 0; k < N_BINS; k++) {
            float p = (float)(re[k] * re[k] + im[k] * im[k]);
            const float* fr = m->fb + (size_t)k * N_MELS;
            for (int j = 0; j < N_MELS; j++) mel[j] += p * fr[j];
        }
        for (int j = 0; j < N_MELS; j++) {
            float v = mel[j] > m->amin ? mel[j] : m->amin;
            v = logf(v) * m->db_scale;
            out[f * N_MELS + j] = v;
            if (v > top) top = v;
        }
    }
    float floor_db = top - m->top_db;   // at most top_db under the loudest value of the call
    for (int i = 0; i < CALL_FRAMES * N_MELS; i++) {
        float v = out[i] > floor_db ? out[i] : floor_db;
        out[i] = v / 10 + 2;
    }
}

// ---- models ---------------------------------------------------------------------------------------------------------

static void fail(char* err, size_t cap, const char* what) {
    if (err && cap) snprintf(err, cap, "oww.bin: %s", what);
}

void oww_free(OwwModels* m) {
    if (!m) return;
    free(m->emb.ops);
    for (int i = 0; i < m->n_words; i++) free(m->words[i].prog.ops);
    free(m);
}

OwwModels* oww_load(const uint8_t* data, size_t size, char* err, size_t errcap) {
    Rd r = {data, size, 0, 0};
    const uint8_t* magic = take(&r, 4);
    if (!magic || memcmp(magic, "OWW1", 4) || take_u32(&r) != 1) {
        fail(err, errcap, "not an openWakeWord file (version 1)");
        return NULL;
    }
    OwwModels* m = calloc(1, sizeof(OwwModels));
    if (!m) return NULL;
    uint32_t nfft = take_u32(&r), win = take_u32(&r), hop = take_u32(&r), mels = take_u32(&r);
    const uint8_t* f3 = take(&r, 12);
    m->fb = (const float*)take(&r, 4u * N_BINS * N_MELS);
    if (r.bad || nfft != N_FFT || win != 400 || hop != HOP || mels != N_MELS) {
        fail(err, errcap, "unexpected melspectrogram");
        goto bad;
    }
    memcpy(&m->amin, f3, 4), memcpy(&m->top_db, f3 + 4, 4), memcpy(&m->db_scale, f3 + 8, 4);
    int out;
    size_t big;
    if (read_prog(&r, &m->emb) || prog_shapes(&m->emb, EMB_FRAMES, N_MELS, 1, &out, &m->emb_buf) || out != EMB_FEATS) {
        fail(err, errcap, "bad embedding model");
        goto bad;
    }
    uint32_t nw = take_u32(&r);
    if (r.bad || nw > OWW_MAX_WORDS) {
        fail(err, errcap, "bad number of wake words");
        goto bad;
    }
    for (uint32_t i = 0; i < nw; i++) {
        Word* w = &m->words[i];
        const uint8_t* nm = take(&r, 64);
        w->windows = take_u32(&r), w->feats = take_u32(&r);
        if (r.bad || w->feats != EMB_FEATS || w->windows < 1 || w->windows > 64 || read_prog(&r, &w->prog) ||
            prog_shapes(&w->prog, 1, 1, w->windows * w->feats, &out, &big) || out != 1) {
            fail(err, errcap, "bad wake word model");
            goto bad;
        }
        memcpy(w->name, nm, 31), memcpy(w->phrase, nm + 32, 31);
        if (big > m->cls_buf) m->cls_buf = big;
        m->n_words++;
    }
    if (r.i != r.n) {
        fail(err, errcap, "trailing bytes");
        goto bad;
    }
    for (int i = 0; i < N_FFT; i++) {
        int k = i - (N_FFT - 400) / 2;   // the 400-sample window sits in the middle of the 512
        m->hann[i] = k >= 0 && k < 400 ? (float)(0.5 - 0.5 * cos(2 * M_PI * k / 400)) : 0.0f;
    }
    for (int i = 0; i < N_FFT / 2; i++) m->cs[i] = cos(2 * M_PI * i / N_FFT), m->sn[i] = sin(2 * M_PI * i / N_FFT);
    // what silence gives (a stream starts with 8 s of it): computed once here
    {
        float* zero = calloc(CALL_SAMPLES, sizeof(float));
        float* frames = malloc(sizeof(float) * CALL_FRAMES * N_MELS);
        float* in = malloc(sizeof(float) * EMB_FRAMES * N_MELS);
        float* a = malloc(sizeof(float) * m->emb_buf);
        float* b = malloc(sizeof(float) * m->emb_buf);
        int ok = zero && frames && in && a && b;
        if (ok) {
            mel_call(m, zero, frames);
            memcpy(m->silence_mel, frames, sizeof(m->silence_mel));
            for (int f = 0; f < EMB_FRAMES; f++) memcpy(in + f * N_MELS, m->silence_mel, sizeof(m->silence_mel));
            memcpy(m->silence_emb, run(&m->emb, in, EMB_FRAMES, N_MELS, 1, a, b), sizeof(m->silence_emb));
        }
        free(zero), free(frames), free(in), free(a), free(b);
        if (!ok) {
            fail(err, errcap, "out of memory");
            goto bad;
        }
    }
    return m;
bad:
    oww_free(m);
    return NULL;
}

int oww_count(const OwwModels* m) { return m ? m->n_words : 0; }
const char* oww_name(const OwwModels* m, int i) { return m->words[i].name; }
const char* oww_phrase(const OwwModels* m, int i) { return m->words[i].phrase; }

int oww_find(const OwwModels* m, const char* name) {
    char k[64];
    size_t n = 0;
    if (!name) return -1;
    for (; name[n] && n < sizeof(k) - 1; n++) {
        char c = name[n];
        k[n] = c == ' ' || c == '-' ? '_' : c >= 'A' && c <= 'Z' ? c + 32 : c;
    }
    k[n] = 0;
    // older names: "ok_nabu", "hey_jarvis_v0.1", "alexa_v0.1.tflite"
    char* dot = strstr(k, ".tflite");
    if (dot) *dot = 0;
    char* ver = strstr(k, "_v0");
    if (ver) *ver = 0;
    if (!strcmp(k, "ok_nabu")) strcpy(k, "okay_nabu");
    for (int i = 0; i < m->n_words; i++)
        if (!strcmp(k, m->words[i].name)) return i;
    return -1;
}

// ---- streams ---------------------------------------------------------------------------------------------------------

struct OwwStream {
    const OwwModels* m;
    int zeros;                  // silent samples still ahead of pend (the 8 s a stream starts with)
    int n_pend;
    float pend[CALL_SAMPLES + CALL_STEP];   // samples not yet used for a call (int16 scale)
    float window[CALL_SAMPLES];
    float frames[CALL_FRAMES * N_MELS];
    float mel[MEL_RING][N_MELS];
    uint8_t mel_silent[MEL_RING];
    int mel_head, new_mels;     // the next frame goes to mel[mel_head]; frames not yet used up (pyopen_wakeword)
    float emb_in[EMB_FRAMES * N_MELS];
    float *ea, *eb, *ca, *cb, *cls_in;
    struct {
        int on, count, head;
        float* ring;            // [windows][96], the newest at head - 1
    } w[OWW_MAX_WORDS];
};

OwwStream* oww_stream_new(const OwwModels* m) {
    OwwStream* s = calloc(1, sizeof(OwwStream));
    if (!s) return NULL;
    s->m = m;
    size_t cls_in = 0;
    for (int i = 0; i < m->n_words; i++) {
        size_t n = (size_t)m->words[i].windows * EMB_FEATS;
        if (n > cls_in) cls_in = n;
    }
    s->ea = malloc(sizeof(float) * m->emb_buf);
    s->eb = malloc(sizeof(float) * m->emb_buf);
    s->ca = malloc(sizeof(float) * (m->cls_buf + 1));
    s->cb = malloc(sizeof(float) * (m->cls_buf + 1));
    s->cls_in = malloc(sizeof(float) * (cls_in + 1));
    int ok = s->ea && s->eb && s->ca && s->cb && s->cls_in;
    for (int i = 0; ok && i < m->n_words; i++) {
        s->w[i].ring = calloc((size_t)m->words[i].windows * EMB_FEATS, sizeof(float));
        ok = s->w[i].ring != NULL;
    }
    if (!ok) {
        oww_stream_free(s);
        return NULL;
    }
    oww_stream_reset(s);
    return s;
}

void oww_stream_free(OwwStream* s) {
    if (!s) return;
    for (int i = 0; i < OWW_MAX_WORDS; i++) free(s->w[i].ring);
    free(s->ea), free(s->eb), free(s->ca), free(s->cb), free(s->cls_in);
    free(s);
}

void oww_stream_enable(OwwStream* s, int word, int on) {
    if (word >= 0 && word < s->m->n_words) s->w[word].on = on;
}

void oww_stream_reset(OwwStream* s) {
    s->zeros = AUTOFILL;
    s->n_pend = 0;
    s->mel_head = s->new_mels = 0;
    for (int i = 0; i < s->m->n_words; i++) s->w[i].count = s->w[i].head = 0;
}

static void classify(OwwStream* s, const float* emb, OwwScoreFn score, void* user) {
    const OwwModels* m = s->m;
    for (int i = 0; i < m->n_words; i++) {
        if (!s->w[i].on) continue;
        int nw = (int)m->words[i].windows;
        memcpy(s->w[i].ring + (size_t)s->w[i].head * EMB_FEATS, emb, sizeof(float) * EMB_FEATS);
        s->w[i].head = (s->w[i].head + 1) % nw;
        if (s->w[i].count < nw) s->w[i].count++;
        if (s->w[i].count < nw) continue;   // (the first nw embeddings of a stream: not enough yet)
        for (int k = 0; k < nw; k++)        // oldest first
            memcpy(s->cls_in + (size_t)k * EMB_FEATS, s->w[i].ring + (size_t)((s->w[i].head + k) % nw) * EMB_FEATS,
                   sizeof(float) * EMB_FEATS);
        const float* p = run(&m->words[i].prog, s->cls_in, 1, 1, nw * EMB_FEATS, s->ca, s->cb);
        if (score) score(user, i, p[0]);
    }
}

// One call: the window's 8 frames, then every embedding they complete, then the classifiers.
static void one_call(OwwStream* s, int silent, OwwScoreFn score, void* user) {
    const OwwModels* m = s->m;
    if (!silent) mel_call(m, s->window, s->frames);
    for (int f = 0; f < CALL_FRAMES; f++) {
        memcpy(s->mel[s->mel_head], silent ? m->silence_mel : s->frames + f * N_MELS, sizeof(float) * N_MELS);
        s->mel_silent[s->mel_head] = (uint8_t)silent;
        s->mel_head = (s->mel_head + 1) % MEL_RING;
    }
    s->new_mels += CALL_FRAMES;
    if (s->new_mels > MAX_MELS) s->new_mels = MAX_MELS;
    while (s->new_mels >= EMB_FRAMES) {   // frames [-new_mels, -new_mels + 76) of the newest
        int start = ((s->mel_head - s->new_mels) % MEL_RING + MEL_RING) % MEL_RING, all_silent = 1;
        for (int f = 0; f < EMB_FRAMES; f++) {
            int k = (start + f) % MEL_RING;
            all_silent &= s->mel_silent[k];
            memcpy(s->emb_in + f * N_MELS, s->mel[k], sizeof(float) * N_MELS);
        }
        s->new_mels -= EMB_STEP;
        const float* e = all_silent ? m->silence_emb : run(&m->emb, s->emb_in, EMB_FRAMES, N_MELS, 1, s->ea, s->eb);
        classify(s, e, score, user);
    }
}

void oww_stream_feed(OwwStream* s, const int16_t* pcm, int n, OwwScoreFn score, void* user) {
    while (n > 0) {
        int room = (int)(sizeof(s->pend) / sizeof(s->pend[0])) - s->n_pend;
        int k = n < room ? n : room;
        for (int i = 0; i < k; i++) s->pend[s->n_pend + i] = (float)pcm[i];
        s->n_pend += k, pcm += k, n -= k;
        while (s->zeros + s->n_pend >= CALL_SAMPLES) {
            int z = s->zeros < CALL_SAMPLES ? s->zeros : CALL_SAMPLES, silent = 1;
            for (int i = 0; i < z; i++) s->window[i] = 0;
            for (int i = z; i < CALL_SAMPLES; i++) {
                float v = s->pend[i - z];
                s->window[i] = v;
                if (v != 0) silent = 0;
            }
            one_call(s, silent, score, user);
            if (s->zeros >= CALL_STEP) {
                s->zeros -= CALL_STEP;
            } else {
                int used = CALL_STEP - s->zeros;
                s->zeros = 0;
                memmove(s->pend, s->pend + used, sizeof(float) * (s->n_pend - used));
                s->n_pend -= used;
            }
        }
    }
}
