// OpenAI-compatible HTTP API, protocol part (see api_proto.h).
#include "api_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- growing string -----------------------------------------------------------------------------------------------

void sb_init(Sb* s) {
    s->p = NULL;
    s->n = s->cap = 0;
    s->oom = 0;
}

void sb_free(Sb* s) {
    free(s->p);
    sb_init(s);
}

void sb_add(Sb* s, const char* p, size_t n) {
    if (s->oom || n == 0) return;
    if (s->n + n + 1 > s->cap) {
        size_t cap = s->cap ? s->cap : 256;
        while (cap < s->n + n + 1) cap *= 2;
        char* q = realloc(s->p, cap);
        if (!q) {
            s->oom = 1;
            return;
        }
        s->p = q;
        s->cap = cap;
    }
    memcpy(s->p + s->n, p, n);
    s->n += n;
    s->p[s->n] = 0;
}

void sb_str(Sb* s, const char* z) { sb_add(s, z, strlen(z)); }

void sb_json_str(Sb* s, const char* p, size_t n) {
    sb_add(s, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        switch (c) {
        case '"': sb_add(s, "\\\"", 2); break;
        case '\\': sb_add(s, "\\\\", 2); break;
        case '\n': sb_add(s, "\\n", 2); break;
        case '\r': sb_add(s, "\\r", 2); break;
        case '\t': sb_add(s, "\\t", 2); break;
        default:
            if (c < 0x20) {
                char esc[8];
                snprintf(esc, sizeof(esc), "\\u%04x", c);
                sb_add(s, esc, 6);
            } else {
                sb_add(s, (const char*)&p[i], 1);   // UTF-8 bytes pass through
            }
        }
    }
    sb_add(s, "\"", 1);
}

void sb_int(Sb* s, long long v) {
    char b[32];
    snprintf(b, sizeof(b), "%lld", v);
    sb_str(s, b);
}

// ---- HTTP request -------------------------------------------------------------------------------------------------

static int ci_eq(const char* a, size_t n, const char* b) {
    size_t m = strlen(b);
    if (n != m) return 0;
    for (size_t i = 0; i < n; i++) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x += 32;
        if (x != y) return 0;
    }
    return 1;
}

int api_http_parse(const char* buf, size_t len, HttpRequest* req) {
    memset(req, 0, sizeof(*req));
    size_t head = 0, term = 4;
    for (size_t i = 0; i + 1 < len; i++) {
        if (i + 3 < len && !memcmp(buf + i, "\r\n\r\n", 4)) {
            head = i + 4;
            break;
        }
        if (!memcmp(buf + i, "\n\n", 2)) {
            head = i + 2;
            term = 2;
            break;
        }
    }
    if (!head) return len > 16384 ? -1 : 0;
    (void)term;
    // request line: METHOD SP target SP HTTP/1.x
    const char* p = buf;
    const char* end = buf + head;
    const char* eol = memchr(p, '\n', end - p);
    if (!eol) return -1;
    const char* sp1 = memchr(p, ' ', eol - p);
    if (!sp1 || sp1 - p >= (long)sizeof(req->method)) return -1;
    memcpy(req->method, p, sp1 - p);
    const char* sp2 = memchr(sp1 + 1, ' ', eol - sp1 - 1);
    if (!sp2) return -1;
    size_t plen = sp2 - (sp1 + 1);
    const char* q = memchr(sp1 + 1, '?', plen);
    if (q) plen = q - (sp1 + 1);
    if (plen == 0 || plen >= sizeof(req->path)) return -1;
    memcpy(req->path, sp1 + 1, plen);
    req->path[plen] = 0;
    if (memchr(sp2 + 1, 'H', eol - sp2 - 1) == NULL) return -1;
    // headers
    p = eol + 1;
    while (p < end) {
        eol = memchr(p, '\n', end - p);
        if (!eol) break;
        const char* line_end = eol;
        if (line_end > p && line_end[-1] == '\r') line_end--;
        if (line_end == p) break;
        const char* colon = memchr(p, ':', line_end - p);
        if (colon) {
            const char* v = colon + 1;
            while (v < line_end && (*v == ' ' || *v == '\t')) v++;
            size_t vlen = line_end - v;
            if (ci_eq(p, colon - p, "content-length")) {
                char num[24];
                if (vlen == 0 || vlen >= sizeof(num)) return -1;
                memcpy(num, v, vlen);
                num[vlen] = 0;
                char* ep;
                long cl = strtol(num, &ep, 10);
                if (*ep || cl < 0 || cl > 1 << 20) return -1;
                req->content_length = cl;
            } else if (ci_eq(p, colon - p, "authorization")) {
                size_t k = vlen < sizeof(req->auth) - 1 ? vlen : sizeof(req->auth) - 1;
                memcpy(req->auth, v, k);
                req->auth[k] = 0;
            }
        }
        p = eol + 1;
    }
    req->head_len = head;
    if (len < head + (size_t)req->content_length) return 0;
    req->body = buf + head;
    req->body_len = (size_t)req->content_length;
    return 1;
}

// ---- JSON reader --------------------------------------------------------------------------------------------------

typedef struct {
    const char* s;
    size_t n, i;
    JNode* pool;
    int max, used, depth;
} JP;

static void jp_ws(JP* P) {
    while (P->i < P->n && (P->s[P->i] == ' ' || P->s[P->i] == '\t' || P->s[P->i] == '\n' || P->s[P->i] == '\r')) P->i++;
}

static int jp_new(JP* P) {
    if (P->used >= P->max) return -1;
    JNode* d = &P->pool[P->used];
    memset(d, 0, sizeof(*d));
    d->first = d->next = -1;
    return P->used++;
}

// raw string at P->i (on the opening quote): sets node s/slen, moves past the closing quote
static int jp_string(JP* P, int node, const char** s, size_t* len) {
    if (P->i >= P->n || P->s[P->i] != '"') return -1;
    P->i++;
    size_t start = P->i;
    while (P->i < P->n) {
        char c = P->s[P->i];
        if (c == '\\') {
            P->i += 2;
            continue;
        }
        if (c == '"') {
            *s = P->s + start;
            *len = P->i - start;
            P->i++;
            (void)node;
            return 0;
        }
        P->i++;
    }
    return -1;
}

static int jp_value(JP* P, int* out);

static int jp_value(JP* P, int* out) {
    jp_ws(P);
    if (P->i >= P->n || ++P->depth > 64) return -1;
    int r = -1;
    char c = P->s[P->i];
    int idx = jp_new(P);
    if (idx < 0) goto done;
    JNode* d = &P->pool[idx];
    if (c == '{' || c == '[') {
        int obj = c == '{';
        d->type = obj ? J_OBJ : J_ARR;
        P->i++;
        jp_ws(P);
        int last = -1;
        if (P->i < P->n && P->s[P->i] == (obj ? '}' : ']')) {
            P->i++;
            r = 0;
            goto done_ok;
        }
        for (;;) {
            const char* key = NULL;
            size_t klen = 0;
            if (obj) {
                jp_ws(P);
                if (jp_string(P, idx, &key, &klen)) goto done;
                jp_ws(P);
                if (P->i >= P->n || P->s[P->i] != ':') goto done;
                P->i++;
            }
            int child;
            if (jp_value(P, &child)) goto done;
            P->pool[child].key = key;
            P->pool[child].klen = klen;
            if (last < 0) P->pool[idx].first = child;
            else P->pool[last].next = child;
            last = child;
            P->pool[idx].count++;
            jp_ws(P);
            if (P->i >= P->n) goto done;
            if (P->s[P->i] == ',') {
                P->i++;
                continue;
            }
            if (P->s[P->i] == (obj ? '}' : ']')) {
                P->i++;
                break;
            }
            goto done;
        }
    done_ok:
        r = 0;
    } else if (c == '"') {
        d->type = J_STR;
        if (jp_string(P, idx, &d->s, &d->slen)) goto done;
        r = 0;
    } else if (c == 't' || c == 'f' || c == 'n') {
        const char* lit = c == 't' ? "true" : c == 'f' ? "false" : "null";
        size_t m = strlen(lit);
        if (P->n - P->i < m || memcmp(P->s + P->i, lit, m)) goto done;
        P->i += m;
        d->type = c == 'n' ? J_NULL : J_BOOL;
        d->b = c == 't';
        r = 0;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
        size_t start = P->i;
        while (P->i < P->n && strchr("+-0123456789.eE", P->s[P->i])) P->i++;
        size_t m = P->i - start;
        if (m == 0 || m > 40) goto done;
        char num[48];
        memcpy(num, P->s + start, m);
        num[m] = 0;
        char* ep;
        d->num = strtod(num, &ep);
        if (*ep) goto done;
        d->type = J_NUM;
        d->s = P->s + start;   // (the number as written, for api_pyjson)
        d->slen = m;
        r = 0;
    }
done:
    if (r == 0) *out = idx;
    P->depth--;
    return r;
}

int json_parse(const char* s, size_t n, JNode* pool, int max) {
    JP P = {s, n, 0, pool, max, 0, 0};
    int root;
    if (jp_value(&P, &root)) return -1;
    jp_ws(&P);
    if (P.i != n) return -1;
    return root;
}

int json_get(const JNode* pool, int obj, const char* key) {
    if (obj < 0 || pool[obj].type != J_OBJ) return -1;
    size_t m = strlen(key);
    for (int c = pool[obj].first; c >= 0; c = pool[c].next)
        if (pool[c].klen == m && !memcmp(pool[c].key, key, m)) return c;
    return -1;
}

static int hexv(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void utf8_put(Sb* out, uint32_t cp) {
    char b[4];
    if (cp < 0x80) {
        b[0] = (char)cp;
        sb_add(out, b, 1);
    } else if (cp < 0x800) {
        b[0] = (char)(0xC0 | (cp >> 6));
        b[1] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 2);
    } else if (cp < 0x10000) {
        b[0] = (char)(0xE0 | (cp >> 12));
        b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[2] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 3);
    } else {
        b[0] = (char)(0xF0 | (cp >> 18));
        b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[3] = (char)(0x80 | (cp & 0x3F));
        sb_add(out, b, 4);
    }
}

// the 4 hex digits at p (p[0..3]); -1 if not hex
static long hex4(const char* p, const char* end) {
    if (end - p < 4) return -1;
    long v = 0;
    for (int i = 0; i < 4; i++) {
        int h = hexv(p[i]);
        if (h < 0) return -1;
        v = v * 16 + h;
    }
    return v;
}

void json_str_sb(const JNode* pool, int node, Sb* out) {
    if (node < 0 || pool[node].type != J_STR) return;
    const char* p = pool[node].s;
    const char* end = p + pool[node].slen;
    while (p < end) {
        if (*p != '\\') {
            sb_add(out, p++, 1);
            continue;
        }
        p++;
        if (p >= end) break;
        char c = *p++;
        switch (c) {
        case 'n': sb_add(out, "\n", 1); break;
        case 't': sb_add(out, "\t", 1); break;
        case 'r': sb_add(out, "\r", 1); break;
        case 'b': sb_add(out, "\b", 1); break;
        case 'f': sb_add(out, "\f", 1); break;
        case 'u': {
            long cp = hex4(p, end);
            if (cp < 0) return;
            p += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                long lo = hex4(p + 2, end);
                if (lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
            }
            utf8_put(out, (uint32_t)cp);
            break;
        }
        default: sb_add(out, &c, 1); break;   // \" \\ \/
        }
    }
}

int api_message(const JNode* pool, int msg, char* role, size_t rcap, Sb* out) {
    if (msg < 0 || pool[msg].type != J_OBJ) return -1;
    int r = json_get(pool, msg, "role");
    if (r < 0 || pool[r].type != J_STR) return -1;
    Sb rs;
    sb_init(&rs);
    json_str_sb(pool, r, &rs);
    size_t k = rs.n < rcap - 1 ? rs.n : rcap - 1;
    if (rs.n) memcpy(role, rs.p, k);
    role[k] = 0;
    sb_free(&rs);
    int c = json_get(pool, msg, "content");
    if (c < 0 || pool[c].type == J_NULL) return 0;
    if (pool[c].type == J_STR) {
        json_str_sb(pool, c, out);
        return 0;
    }
    if (pool[c].type != J_ARR) return -1;
    for (int part = pool[c].first; part >= 0; part = pool[part].next) {
        if (pool[part].type != J_OBJ) return -1;
        int t = json_get(pool, part, "text");
        if (t >= 0 && pool[t].type == J_STR) json_str_sb(pool, t, out);
    }
    return 0;
}

// ---- JSON writers -------------------------------------------------------------------------------------------------

void api_json_models(Sb* out, const char* model, long long created) {
    sb_str(out, "{\"object\":\"list\",\"data\":[{\"id\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"object\":\"model\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"owned_by\":\"garlic-llm\"}]}");
}

void api_json_completion(Sb* out, const char* id, const char* model, long long created, const char* text, size_t tlen,
                         const char* finish, int prompt_tokens, int completion_tokens) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    sb_json_str(out, text, tlen);
    sb_str(out, "},\"finish_reason\":");
    sb_json_str(out, finish, strlen(finish));
    sb_str(out, "}],\"usage\":{\"prompt_tokens\":");
    sb_int(out, prompt_tokens);
    sb_str(out, ",\"completion_tokens\":");
    sb_int(out, completion_tokens);
    sb_str(out, ",\"total_tokens\":");
    sb_int(out, prompt_tokens + completion_tokens);
    sb_str(out, "}}");
}

void api_json_chunk(Sb* out, const char* id, const char* model, long long created, const char* role, const char* text,
                    size_t tlen, const char* finish) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{");
    int any = 0;
    if (role) {
        sb_str(out, "\"role\":");
        sb_json_str(out, role, strlen(role));
        any = 1;
    }
    if (tlen) {
        if (any) sb_str(out, ",");
        sb_str(out, "\"content\":");
        sb_json_str(out, text, tlen);
    }
    sb_str(out, "},\"finish_reason\":");
    if (finish) sb_json_str(out, finish, strlen(finish));
    else sb_str(out, "null");
    sb_str(out, "}]}");
}

void api_json_error(Sb* out, const char* message, const char* type, const char* code) {
    sb_str(out, "{\"error\":{\"message\":");
    sb_json_str(out, message, strlen(message));
    sb_str(out, ",\"type\":");
    sb_json_str(out, type, strlen(type));
    sb_str(out, ",\"code\":");
    if (code) sb_json_str(out, code, strlen(code));
    else sb_str(out, "null");
    sb_str(out, "}}");
}

// ---- chat request -------------------------------------------------------------------------------------------------

int api_chat_parse(const char* body, size_t len, JNode* pool, int max, ChatRequest* req, const char** err) {
    memset(req, 0, sizeof(*req));
    req->temperature = 0.7;
    req->max_tokens = 512;
    req->messages = -1;
    req->root = json_parse(body, len, pool, max);
    if (req->root < 0 || pool[req->root].type != J_OBJ) {
        *err = "The body must be a JSON object.";
        return -1;
    }
    int m = json_get(pool, req->root, "messages");
    if (m < 0 || pool[m].type != J_ARR || pool[m].count < 1) {
        *err = "'messages' must be a non-empty array.";
        return -1;
    }
    for (int c = pool[m].first; c >= 0; c = pool[c].next) {
        if (pool[c].type != J_OBJ) {
            *err = "Each message must be an object with 'role' and 'content'.";
            return -1;
        }
        req->n_messages++;
    }
    req->messages = m;
    int t = json_get(pool, req->root, "temperature");
    if (t >= 0 && pool[t].type != J_NULL) {
        if (pool[t].type != J_NUM || pool[t].num < 0 || pool[t].num > 2) {
            *err = "'temperature' must be a number from 0 to 2.";
            return -1;
        }
        req->temperature = pool[t].num;
    }
    int mt = json_get(pool, req->root, "max_completion_tokens");
    if (mt < 0) mt = json_get(pool, req->root, "max_tokens");
    if (mt >= 0 && pool[mt].type != J_NULL) {
        if (pool[mt].type != J_NUM || pool[mt].num < 1 || pool[mt].num > 16384) {
            *err = "'max_tokens' must be a number from 1 to 16384.";
            return -1;
        }
        req->max_tokens = (int)pool[mt].num;
    }
    int s = json_get(pool, req->root, "stream");
    if (s >= 0 && pool[s].type != J_NULL) {
        if (pool[s].type != J_BOOL) {
            *err = "'stream' must be true or false.";
            return -1;
        }
        req->stream = pool[s].b;
    }
    int n = json_get(pool, req->root, "n");
    if (n >= 0 && pool[n].type == J_NUM && pool[n].num != 1) {
        *err = "Only n = 1 is supported.";
        return -1;
    }
    int sd = json_get(pool, req->root, "seed");
    if (sd >= 0 && pool[sd].type == J_NUM && pool[sd].num >= 0) {
        req->seed = (unsigned)pool[sd].num;
        req->has_seed = 1;
    }
    req->tools = -1;
    int tl = json_get(pool, req->root, "tools");
    if (tl >= 0 && pool[tl].type != J_NULL) {
        if (pool[tl].type != J_ARR) {
            *err = "'tools' must be an array.";
            return -1;
        }
        for (int k = pool[tl].first; k >= 0; k = pool[k].next) {
            int f = pool[k].type == J_OBJ ? json_get(pool, k, "function") : -1;
            int nm = f >= 0 && pool[f].type == J_OBJ ? json_get(pool, f, "name") : -1;
            if (nm < 0 || pool[nm].type != J_STR) {
                *err = "Each tool must be {\"type\": \"function\", \"function\": {\"name\", ...}}.";
                return -1;
            }
        }
        int tc = json_get(pool, req->root, "tool_choice");
        int none = tc >= 0 && pool[tc].type == J_STR && pool[tc].slen == 4 && !memcmp(pool[tc].s, "none", 4);
        if (pool[tl].count > 0 && !none) req->tools = tl;
    }
    return 0;
}

// ---- Qwen3.5's chat template ----------------------------------------------------------------------------------------

// a string as json.dumps(ensure_ascii=False) writes it
static void py_str(Sb* out, const char* p, size_t n) {
    sb_add(out, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)p[i];
        const char* e = c == '"' ? "\\\"" : c == '\\' ? "\\\\" : c == '\n' ? "\\n" : c == '\r' ? "\\r" : c == '\t' ? "\\t"
                      : c == '\b' ? "\\b" : c == '\f' ? "\\f" : NULL;
        if (e) {
            sb_str(out, e);
        } else if (c < 0x20) {
            char esc[8];
            snprintf(esc, sizeof(esc), "\\u%04x", c);
            sb_add(out, esc, 6);
        } else {
            sb_add(out, (const char*)&p[i], 1);
        }
    }
    sb_add(out, "\"", 1);
}

static void py_jstr(Sb* out, const JNode* pool, int node) {
    Sb d;
    sb_init(&d);
    json_str_sb(pool, node, &d);
    py_str(out, d.p ? d.p : "", d.n);
    sb_free(&d);
}

void api_pyjson(Sb* out, const JNode* pool, int node) {
    const JNode* n = &pool[node];
    switch (n->type) {
    case J_NULL: sb_str(out, "null"); break;
    case J_BOOL: sb_str(out, n->b ? "true" : "false"); break;
    case J_NUM: sb_add(out, n->s, n->slen); break;
    case J_STR: py_jstr(out, pool, node); break;
    default: {
        int obj = n->type == J_OBJ;
        sb_str(out, obj ? "{" : "[");
        for (int c = n->first; c >= 0; c = pool[c].next) {
            if (c != n->first) sb_str(out, ", ");
            if (obj) {
                JNode k = pool[c];   // (the key, as a string node)
                k.type = J_STR, k.s = k.key, k.slen = k.klen;
                py_jstr(out, &k, 0);
                sb_str(out, ": ");
            }
            api_pyjson(out, pool, c);
        }
        sb_str(out, obj ? "}" : "]");
    }
    }
}

static int is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

// Python's str.strip() of s[0, *n): returns the start, sets *n
static const char* strip(const char* s, size_t* n) {
    size_t a = 0, b = *n;
    while (a < b && is_ws(s[a])) a++;
    while (b > a && is_ws(s[b - 1])) b--;
    *n = b - a;
    return s + a;
}

static int has(const char* s, size_t n, const char* sub, const char** at) {
    size_t m = strlen(sub);
    for (size_t i = 0; i + m <= n; i++)
        if (!memcmp(s + i, sub, m)) {
            if (at) *at = s + i;
            return 1;
        }
    return 0;
}

// the last occurrence of sub in s[0, n), or NULL
static const char* last_of(const char* s, size_t n, const char* sub) {
    size_t m = strlen(sub);
    for (size_t i = n >= m ? n - m + 1 : 0; i-- > 0;)
        if (!memcmp(s + i, sub, m)) return s + i;
    return NULL;
}

static int role_is(const JNode* pool, int msg, const char* role) {
    int r = json_get(pool, msg, "role");
    size_t m = strlen(role);
    return r >= 0 && pool[r].type == J_STR && pool[r].slen == m && !memcmp(pool[r].s, role, m);
}

// A value of a tool call's arguments as the template writes it: strings as they are, objects and lists as JSON,
// anything else as Python's str() (True, False, None, numbers).
static void arg_value(Sb* out, const JNode* pool, int v) {
    switch (pool[v].type) {
    case J_STR: json_str_sb(pool, v, out); break;
    case J_OBJ:
    case J_ARR: api_pyjson(out, pool, v); break;
    case J_BOOL: sb_str(out, pool[v].b ? "True" : "False"); break;
    case J_NULL: sb_str(out, "None"); break;
    default: sb_add(out, pool[v].s, pool[v].slen);
    }
}

static const char TOOLS_HEAD[] = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
static const char TOOLS_TAIL[] =
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

// One assistant tool call (OpenAI's {"function": {"name", "arguments": "<JSON>"}}, or name/arguments directly).
static int render_call(Sb* out, const JNode* pool, int tc, int first, int had_content) {
    int f = json_get(pool, tc, "function");
    if (f < 0 || pool[f].type != J_OBJ) f = tc;
    int nm = json_get(pool, f, "name");
    if (nm < 0 || pool[nm].type != J_STR) return -1;
    sb_str(out, first ? (had_content ? "\n\n<tool_call>\n<function=" : "<tool_call>\n<function=") : "\n<tool_call>\n<function=");
    json_str_sb(pool, nm, out);
    sb_str(out, ">\n");
    int a = json_get(pool, f, "arguments");
    if (a >= 0 && pool[a].type != J_NULL) {
        const JNode* ap = pool;
        JNode* own = NULL;
        Sb text;
        sb_init(&text);
        if (pool[a].type == J_STR) {   // OpenAI sends the arguments as a JSON string: read it
            json_str_sb(pool, a, &text);
            own = malloc(sizeof(JNode) * 2048);
            int r = own && text.p ? json_parse(text.p, text.n, own, 2048) : -1;
            if (r < 0 || own[r].type != J_OBJ) {
                free(own);
                sb_free(&text);
                return -1;
            }
            ap = own, a = r;
        }
        if (ap[a].type == J_OBJ)
            for (int k = ap[a].first; k >= 0; k = ap[k].next) {
                JNode key = ap[k];
                key.type = J_STR, key.s = key.key, key.slen = key.klen;
                sb_str(out, "<parameter=");
                json_str_sb(&key, 0, out);
                sb_str(out, ">\n");
                arg_value(out, ap, k);
                sb_str(out, "\n</parameter>\n");
            }
        free(own);
        sb_free(&text);
    }
    sb_str(out, "</function>\n</tool_call>");
    return 0;
}

int api_render(const JNode* pool, const ChatRequest* cr, int think_block, Sb* out, size_t* stable, const char** err) {
    int first = pool[cr->messages].first, n = pool[cr->messages].count;
    int sys_first = role_is(pool, first, "system");
    Sb c;
    sb_init(&c);
    if (cr->tools >= 0) {
        sb_str(out, "<|im_start|>system\n");
        sb_str(out, TOOLS_HEAD);
        for (int t = pool[cr->tools].first; t >= 0; t = pool[t].next) {
            sb_str(out, "\n");
            api_pyjson(out, pool, t);
        }
        sb_str(out, TOOLS_TAIL);
        if (sys_first) {
            char role[16];
            api_message(pool, first, role, sizeof(role), &c);
            size_t k = c.n;
            const char* t = strip(c.p ? c.p : "", &k);
            if (k) sb_str(out, "\n\n"), sb_add(out, t, k);
        }
        sb_str(out, "<|im_end|>\n");
    } else if (sys_first) {
        char role[16];
        api_message(pool, first, role, sizeof(role), &c);
        size_t k = c.n;
        const char* t = strip(c.p ? c.p : "", &k);
        sb_str(out, "<|im_start|>system\n");
        sb_add(out, t, k);
        sb_str(out, "<|im_end|>\n");
    }
    sb_free(&c);
    *stable = out->n;

    // the last user message that is a question (not tool results): assistant turns after it keep their thought block
    int last_query = n - 1, idx = 0, found = -1;
    for (int m = first; m >= 0; m = pool[m].next, idx++) {
        if (!role_is(pool, m, "user")) continue;
        char role[16];
        sb_init(&c);
        api_message(pool, m, role, sizeof(role), &c);
        size_t k = c.n;
        const char* t = strip(c.p ? c.p : "", &k);
        const char *a = "<tool_response>", *b = "</tool_response>";
        if (!(k >= strlen(a) + strlen(b) && !memcmp(t, a, strlen(a)) && !memcmp(t + k - strlen(b), b, strlen(b))))
            found = idx;
        sb_free(&c);
    }
    if (found >= 0) last_query = found;

    idx = 0;
    int prev_tool = 0;
    for (int m = first; m >= 0; m = pool[m].next, idx++) {
        char role[16];
        sb_init(&c);
        if (api_message(pool, m, role, sizeof(role), &c)) {
            sb_free(&c);
            *err = "Each message must have a 'role' and a text 'content'.";
            return -1;
        }
        size_t k = c.n;
        const char* t = strip(c.p ? c.p : "", &k);
        int is_tool = !strcmp(role, "tool");
        if (!strcmp(role, "system")) {
            if (idx != 0) {
                sb_free(&c);
                *err = "A system message must be the first.";
                return -1;
            }
        } else if (!strcmp(role, "user")) {
            sb_str(out, "<|im_start|>user\n");
            sb_add(out, t, k);
            sb_str(out, "<|im_end|>\n");
        } else if (!strcmp(role, "assistant")) {
            const char* reason = "";
            size_t rn = 0;
            int rc = json_get(pool, m, "reasoning_content");
            Sb rs;
            sb_init(&rs);
            const char* te;
            if (rc >= 0 && pool[rc].type == J_STR) {
                json_str_sb(pool, rc, &rs);
                reason = rs.p ? rs.p : "", rn = rs.n;
            } else if (has(t, k, "</think>", &te)) {   // a thought inside the content
                const char* th = last_of(t, (size_t)(te - t), "<think>");
                const char* r0 = th ? th + 7 : t;
                size_t r = (size_t)(te - r0);
                while (r && r0[r - 1] == '\n') r--;   // rstrip('\n') then lstrip('\n')
                while (r && *r0 == '\n') r0++, r--;
                reason = r0, rn = r;
                const char* after = last_of(t, k, "</think>") + 8;
                k -= (size_t)(after - t), t = after;
                while (k && *t == '\n') t++, k--;
            }
            reason = strip(reason, &rn);
            if (idx > last_query) {
                sb_str(out, "<|im_start|>assistant\n<think>\n");
                sb_add(out, reason, rn);
                sb_str(out, "\n</think>\n\n");
            } else {
                sb_str(out, "<|im_start|>assistant\n");
            }
            sb_add(out, t, k);
            size_t kk = k;
            strip(t, &kk);
            int calls = json_get(pool, m, "tool_calls");
            if (calls >= 0 && pool[calls].type == J_ARR)
                for (int tc = pool[calls].first; tc >= 0; tc = pool[tc].next)
                    if (render_call(out, pool, tc, tc == pool[calls].first, kk > 0)) {
                        sb_free(&rs);
                        sb_free(&c);
                        *err = "A tool call must have a function name and JSON object arguments.";
                        return -1;
                    }
            sb_str(out, "<|im_end|>\n");
            sb_free(&rs);
        } else if (is_tool) {
            if (idx > 0 && !prev_tool) sb_str(out, "<|im_start|>user");
            sb_str(out, "\n<tool_response>\n");
            sb_add(out, t, k);
            sb_str(out, "\n</tool_response>");
            int next = pool[m].next;
            if (next < 0 || !role_is(pool, next, "tool")) sb_str(out, "<|im_end|>\n");
        } else {
            sb_free(&c);
            *err = "A message's role must be system, user, assistant or tool.";
            return -1;
        }
        prev_tool = is_tool;
        sb_free(&c);
    }
    sb_str(out, "<|im_start|>assistant\n");
    if (think_block) sb_str(out, "<think>\n\n</think>\n\n");
    if (out->oom) {
        *err = "Out of memory.";
        return -1;
    }
    return 0;
}

// ---- the model's tool calls -------------------------------------------------------------------------------------------

// the type the tool's schema gives parameter p of function fn ("" if it says none)
static void param_type(const JNode* pool, int tools, const char* fn, size_t fl, const char* p, size_t pl, char* type,
                       size_t cap) {
    type[0] = 0;
    if (tools < 0) return;
    for (int t = pool[tools].first; t >= 0; t = pool[t].next) {
        int f = json_get(pool, t, "function");
        int nm = f >= 0 ? json_get(pool, f, "name") : -1;
        Sb name;
        sb_init(&name);
        if (nm >= 0) json_str_sb(pool, nm, &name);
        int same = name.n == fl && name.p && !memcmp(name.p, fn, fl);
        sb_free(&name);
        if (!same) continue;
        int pr = json_get(pool, f, "parameters");
        int props = pr >= 0 ? json_get(pool, pr, "properties") : -1;
        if (props < 0 || pool[props].type != J_OBJ) return;
        for (int q = pool[props].first; q >= 0; q = pool[q].next) {
            JNode key = pool[q];
            key.type = J_STR, key.s = key.key, key.slen = key.klen;
            Sb kn;
            sb_init(&kn);
            json_str_sb(&key, 0, &kn);
            int hit = kn.n == pl && kn.p && !memcmp(kn.p, p, pl);
            sb_free(&kn);
            if (!hit || pool[q].type != J_OBJ) continue;
            int ty = json_get(pool, q, "type");
            if (ty >= 0 && pool[ty].type == J_ARR)   // ["integer", "null"]: the first that is not null
                for (int e = pool[ty].first; e >= 0; e = pool[e].next)
                    if (pool[e].type == J_STR && !(pool[e].slen == 4 && !memcmp(pool[e].s, "null", 4))) {
                        ty = e;
                        break;
                    }
            if (ty >= 0 && pool[ty].type == J_STR && pool[ty].slen < cap) {
                memcpy(type, pool[ty].s, pool[ty].slen);
                type[pool[ty].slen] = 0;
            }
            return;
        }
        return;
    }
}

static int is_json(const char* v, size_t n, int want) {   // want: a JSON type, or -1 for any
    JNode pool[256];
    int r = json_parse(v, n, pool, 256);
    if (r < 0) {   // (too many nodes for the small pool: try a big one)
        JNode* big = malloc(sizeof(JNode) * 8192);
        r = big ? json_parse(v, n, big, 8192) : -1;
        int t = r >= 0 ? big[r].type : -1;
        free(big);
        return r >= 0 && (want < 0 || t == want);
    }
    return want < 0 || pool[r].type == want;
}

// value v (n bytes) as the JSON of its parameter's type
static void param_json(Sb* out, const char* v, size_t n, const char* type) {
    size_t k = n;
    const char* t = strip(v, &k);
    char num[48];
    if (!strcmp(type, "integer") || !strcmp(type, "number")) {
        if (k && k < sizeof(num)) {
            memcpy(num, t, k);
            num[k] = 0;
            char* e;
            double d = strtod(num, &e);
            if (!*e && d == d && d - d == 0) {   // (finite)
                if (!strcmp(type, "integer") && d == (double)(long long)d) snprintf(num, sizeof(num), "%lld", (long long)d);
                else if (!is_json(t, k, J_NUM)) snprintf(num, sizeof(num), "%.17g", d);
                sb_str(out, num);
                return;
            }
        }
    } else if (!strcmp(type, "boolean")) {
        int yes = ci_eq(t, k, "true") || ci_eq(t, k, "1") || ci_eq(t, k, "yes");
        int no = ci_eq(t, k, "false") || ci_eq(t, k, "0") || ci_eq(t, k, "no");
        if (yes || no) {
            sb_str(out, yes ? "true" : "false");
            return;
        }
    } else if (!strcmp(type, "array") || !strcmp(type, "object")) {
        int want = type[0] == 'a' ? J_ARR : J_OBJ;
        if (is_json(t, k, want)) {
            sb_add(out, t, k);
            return;
        }
        if (want == J_ARR && k) {   // a single item written without the brackets: a list of one
            sb_str(out, "[");
            if (is_json(t, k, -1) && !is_json(t, k, J_OBJ) && !is_json(t, k, J_ARR)) sb_add(out, t, k);
            else sb_json_str(out, v, n);
            sb_str(out, "]");
            return;
        }
    } else if (!strcmp(type, "null")) {
        sb_str(out, "null");
        return;
    } else if (!type[0]) {   // no type in the schema: JSON if it is an object or a list, else the text
        if (k && (*t == '{' || *t == '[') && is_json(t, k, -1)) {
            sb_add(out, t, k);
            return;
        }
    }
    sb_json_str(out, v, n);
}

// "<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n...</function>": the arguments' JSON object into args
static int parse_xml_call(const char* b, size_t n, const JNode* pool, int tools, Sb* name, Sb* args) {
    const char* f;
    if (!has(b, n, "<function=", &f)) return -1;
    f += 10;
    const char* gt = memchr(f, '>', (size_t)(b + n - f));
    if (!gt) return -1;
    size_t fl = (size_t)(gt - f);
    const char* fn = strip(f, &fl);
    if (!fl) return -1;
    sb_add(name, fn, fl);
    const char* p = gt + 1;
    const char* end = b + n;
    const char* fe;
    if (has(p, (size_t)(end - p), "</function>", &fe)) end = fe;
    sb_str(args, "{");
    int count = 0;
    const char* q;
    while (has(p, (size_t)(end - p), "<parameter=", &q)) {
        q += 11;
        const char* pg = memchr(q, '>', (size_t)(end - q));
        if (!pg) break;
        size_t pl = (size_t)(pg - q);
        const char* pn = strip(q, &pl);
        const char* v = pg + 1;
        const char *ve = end, *close = NULL, *nextp;
        if (has(v, (size_t)(end - v), "</parameter>", &close)) ve = close;
        if (has(v, (size_t)(ve - v), "<parameter=", &nextp)) ve = nextp;   // (an unclosed parameter ends at the next)
        size_t vn = (size_t)(ve - v);
        if (vn && *v == '\n') v++, vn--;   // the template's newline after the tag and before the closing one
        if (vn && v[vn - 1] == '\n') vn--;
        char type[16];
        param_type(pool, tools, fn, fl, pn, pl, type, sizeof(type));
        if (count++) sb_str(args, ",");
        sb_json_str(args, pn, pl);
        sb_str(args, ":");
        param_json(args, v, vn, type);
        p = ve == close ? close + 12 : ve;
    }
    sb_str(args, "}");
    return 0;
}

// the older form: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
static int parse_json_call(const char* b, size_t n, Sb* name, Sb* args) {
    JNode* pool = malloc(sizeof(JNode) * 4096);
    int r = pool ? json_parse(b, n, pool, 4096) : -1;
    int nm = r >= 0 && pool[r].type == J_OBJ ? json_get(pool, r, "name") : -1;
    if (nm < 0 || pool[nm].type != J_STR) {
        free(pool);
        return -1;
    }
    json_str_sb(pool, nm, name);
    int a = json_get(pool, r, "arguments");
    if (a < 0) a = json_get(pool, r, "parameters");
    if (a >= 0 && pool[a].type == J_OBJ) {
        // compact JSON of the object, from its own text
        Sb s;
        sb_init(&s);
        api_pyjson(&s, pool, a);
        sb_add(args, s.p, s.n);
        sb_free(&s);
    } else if (a >= 0 && pool[a].type == J_STR) {
        json_str_sb(pool, a, args);
    } else {
        sb_str(args, "{}");
    }
    free(pool);
    return 0;
}

int api_parse_reply(const char* text, size_t n, const JNode* pool, int tools, unsigned id_base, Sb* content, Sb* calls) {
    const char* tc;
    size_t before = has(text, n, "<tool_call>", &tc) ? (size_t)(tc - text) : n;
    size_t k = before;
    const char* t = strip(text, &k);
    sb_add(content, t, k);
    int count = 0;
    const char* p = text + before;
    const char* end = text + n;
    while (has(p, (size_t)(end - p), "<tool_call>", &tc)) {
        const char* b = tc + 11;
        const char *be = end, *close = NULL;
        if (has(b, (size_t)(end - b), "</tool_call>", &close)) be = close;
        size_t bn = (size_t)(be - b);
        const char* bs = strip(b, &bn);
        Sb name, args;
        sb_init(&name);
        sb_init(&args);
        int ok = bn && *bs == '{' ? parse_json_call(bs, bn, &name, &args) : parse_xml_call(b, (size_t)(be - b), pool, tools, &name, &args);
        if (ok == 0 && name.p && args.p) {
            char id[48];
            snprintf(id, sizeof(id), "call_%u_%d", id_base, count);
            sb_str(calls, count ? ",{\"index\":" : "[{\"index\":");
            sb_int(calls, count);
            sb_str(calls, ",\"id\":");
            sb_json_str(calls, id, strlen(id));
            sb_str(calls, ",\"type\":\"function\",\"function\":{\"name\":");
            sb_json_str(calls, name.p, name.n);
            sb_str(calls, ",\"arguments\":");
            sb_json_str(calls, args.p, args.n);
            sb_str(calls, "}}");
            count++;
        }
        sb_free(&name);
        sb_free(&args);
        p = be == close ? close + 12 : end;
    }
    if (count) sb_str(calls, "]");
    return count;
}

// ---- JSON writers with tool calls -----------------------------------------------------------------------------------

void api_json_completion_calls(Sb* out, const char* id, const char* model, long long created, const char* text,
                               size_t tlen, const char* calls, const char* finish, int prompt_tokens,
                               int completion_tokens) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":");
    if (calls && !tlen) sb_str(out, "null");
    else sb_json_str(out, text, tlen);
    if (calls) {
        sb_str(out, ",\"tool_calls\":");
        sb_str(out, calls);
    }
    sb_str(out, "},\"finish_reason\":");
    sb_json_str(out, finish, strlen(finish));
    sb_str(out, "}],\"usage\":{\"prompt_tokens\":");
    sb_int(out, prompt_tokens);
    sb_str(out, ",\"completion_tokens\":");
    sb_int(out, completion_tokens);
    sb_str(out, ",\"total_tokens\":");
    sb_int(out, prompt_tokens + completion_tokens);
    sb_str(out, "}}");
}

void api_json_chunk_calls(Sb* out, const char* id, const char* model, long long created, const char* calls) {
    sb_str(out, "{\"id\":");
    sb_json_str(out, id, strlen(id));
    sb_str(out, ",\"object\":\"chat.completion.chunk\",\"created\":");
    sb_int(out, created);
    sb_str(out, ",\"model\":");
    sb_json_str(out, model, strlen(model));
    sb_str(out, ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":");
    sb_str(out, calls);
    sb_str(out, "},\"finish_reason\":null}]}");
}
