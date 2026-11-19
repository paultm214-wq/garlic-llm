// OpenAI-compatible API jobs (api.c queues them; included in main.c, so it sees the engine and the conversation).
// A job runs on the model between the chat's turns: the request's messages are the whole conversation, so the
// conversation on screen is cleared, and the reply is written to the connection (whole, or as server-sent events).
// The reply is generated as the chat generates its own: several tokens per step with the MTP head where the app has
// it (GARLIC_SPEC), and Circle on the controller stops it. What the screen shows meanwhile is main.c's part
// (api_chat_begin / _reply / _step / _end).
// The prompt is the model's own chat template (api_render, api_proto.c), tools included: a reply that calls tools
// comes back as OpenAI tool_calls (Home Assistant's way to control the house).
// main.c gives two tokenizer hooks: api_special (is a special token's text at p? its id) and api_special_text (the
// text of a special token the model wrote, such as <tool_call>).

static uint32_t api_ids[16384];   // the prompt of a job
#define API_MAX_TEXT (256 << 10)  // bytes of prompt a request may make (its tokens must also fit the context)

// ---- the model's state between requests ------------------------------------------------------------------------------
// api_seq[0, api_seq_n): the tokens the model has read for the API, in order, while api_seq_ok. A request whose
// prompt starts with all of them goes on from there instead of reading them again: Home Assistant sends the whole
// conversation every time (and once more after each round of tool calls), so the second request of a command only
// reads the tool results. api_sys_n: the tokens of the last request's system turn (its instructions and tools, the
// same in every conversation). When no request has come for API_REWIND_S, the model reads those again from position
// 0 while idle (api_idle), so the next conversation only reads its own messages. Anything else that uses the model
// (the chat, voice mode, the MTP head coming or going) calls api_cache_drop.
static uint32_t api_seq[16384];
static int api_seq_n, api_seq_ok, api_seq_spec, api_sys_n, api_rewinding;
static double api_last_t, api_rewind_t0;
#define API_REWIND_S 3.0

static void api_cache_drop(void) { api_seq_ok = 0, api_rewinding = 0; }

static int api_spec_on(void) {
#ifdef GARLIC_SPEC
    return spec_active(&eng);
#else
    return 0;
#endif
}

// How many bytes at the start of p (n) are whole UTF-8 characters (the rest waits for the next token).
static size_t api_utf8_whole(const uint8_t* p, size_t n) {
    size_t i = n;
    for (int back = 1; back <= 4 && i > 0; back++) {
        i--;
        uint8_t c = p[i];
        if ((c & 0xC0) == 0x80) continue;   // a continuation byte: keep looking for its lead byte
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        return back < (int)need ? i : n;
    }
    return n;
}

// one server-sent event; -1 if the client went away
static int api_send_event(int fd, const char* json) {
    Sb s;
    sb_init(&s);
    sb_str(&s, "data: ");
    sb_str(&s, json);
    sb_str(&s, "\n\n");
    int r = s.p && !s.oom ? api_send(fd, s.p, s.n) : -1;
    sb_free(&s);
    return r;
}

// Text (n bytes) to ids after ids[0, k): special tokens' texts (<|im_start|>, <tool_call>...) become their ids, as the
// model's own tokenizer does with a rendered chat template; the rest is plain text. Returns the new count, or -1 if
// it does not fit in max.
static int api_encode(char* text, size_t n, uint32_t* ids, int k, int max) {
    size_t a = 0;
    for (size_t i = 0; i <= n; i++) {
        uint32_t id = 0;
        int sl = i < n && text[i] == '<' ? api_special(text + i, n - i, &id) : 0;
        if (!sl && i < n) continue;
        if (i > a) {   // the plain text before it
            char keep = text[i];
            text[i] = 0;
            k = encode_into(ids, k, max, text + a);
            text[i] = keep;
        }
        if (k >= max - 1) return -1;
        if (sl) ids[k++] = id, i += (size_t)sl - 1;
        a = i + 1;
    }
    return k >= max - 1 ? -1 : k;
}

static void api_error(int fd, const char* status_line, const char* message, const char* type, const char* code) {
    Sb s;
    sb_init(&s);
    api_json_error(&s, message, type, code);
    Sb h;
    sb_init(&h);
    sb_str(&h, status_line);
    sb_str(&h, "\r\nContent-Type: application/json\r\nContent-Length: ");
    sb_int(&h, (long long)s.n);
    sb_str(&h, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
    if (h.p && s.p && api_send(fd, h.p, h.n) == 0) api_send(fd, s.p, s.n);
    sb_free(&h);
    sb_free(&s);
}

// Idle (nothing else to do, no conversation on the chat screen): the last system turn read again from position 0,
// a few tokens per call (see api_seq above).
static void api_idle(void) {
    if (!api_seq_ok || api_sys_n <= 0 || pos != 0) return;
    if (!api_rewinding) {
        if (api_seq_n == api_sys_n || now() - api_last_t < API_REWIND_S) return;
        engine_reset(&eng);
        api_seq_n = 0;
        api_rewinding = 1;
        api_rewind_t0 = now();
    }
    int k = api_sys_n - api_seq_n < 16 ? api_sys_n - api_seq_n : 16;
    uint32_t next;
    if (engine_run(&eng, api_seq + api_seq_n, k, api_seq_n, 0.0f, 1, &next)) {
        say("api: reading the system prompt again failed (%s)", eng.error);
        api_cache_drop();
        return;
    }
    api_seq_n += k;
    if (api_seq_n == api_sys_n) {
        api_rewinding = 0;
        say("api: system prompt read again while idle (%d tokens, %.1f s)", api_sys_n, now() - api_rewind_t0);
    }
}

static void api_job(ApiJob* job) {
    static unsigned api_count;
    JNode* pool = malloc(sizeof(JNode) * API_MAX_NODES);
    ChatRequest cr;
    const char* err = NULL;
    if (!pool || api_chat_parse(job->body, job->body_len, pool, API_MAX_NODES, &cr, &err)) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "Out of memory.", "invalid_request_error",
                  "invalid_body");
        return;
    }
    Sb prompt;
    sb_init(&prompt);
    size_t stable = 0;
    int n = -1, ns = 0;
    if (api_render(pool, &cr, eng.qwen35, &prompt, &stable, &err) == 0 && prompt.p && prompt.n <= API_MAX_TEXT) {
        char keep = prompt.p[stable];   // the system turn first, on its own (its tokens are the same either way: it
        prompt.p[stable] = 0;           // ends at a special token's boundary)
        ns = api_encode(prompt.p, stable, api_ids, 0, 16384);
        prompt.p[stable] = keep;
        n = ns < 0 ? -1 : api_encode(prompt.p + stable, prompt.n - stable, api_ids, ns, 16384);
    }
    sb_free(&prompt);
    if (n < 0) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", err ? err : "The messages are too long.",
                  "invalid_request_error", err ? "invalid_body" : "context_length_exceeded");
        return;
    }
    if (n + cr.max_tokens + 2 > (int)eng.ctx_limit) {
        free(pool);
        api_error(job->fd, "HTTP/1.1 400 Bad Request", "The messages and max_tokens do not fit the context window.",
                  "invalid_request_error", "context_length_exceeded");
        return;
    }
    // what the model has already read of this prompt (see api_seq)
    int from = 0;
    if (api_seq_ok && api_seq_spec == api_spec_on() && api_seq_n > 0 && api_seq_n < n &&
        !memcmp(api_seq, api_ids, sizeof(uint32_t) * api_seq_n))
        from = api_seq_n;
    // the model answers this request: the conversation on screen starts again from nothing (a request refused
    // above leaves it as it was)
    pos = 0;
    need_close = 0;
    hist_n = hist_done = 0;
    thinking = think_turn = 0;
    sys_pre = 0;
    view.scroll = 0;
    api_rewinding = 0;
    api_chat_begin();   // (clears the conversation: a note says that a request is being answered)
    if (cr.has_seed) seed = cr.seed;
    char id[32];
    snprintf(id, sizeof(id), "chatcmpl-%u", ++api_count);
    const char* model = api_model();
    long long created = (long long)time(NULL);
    say("api: prompt of %d tokens (%d already read%s), up to %d new, T %.2f%s", n, from,
        cr.tools >= 0 ? ", tools" : "", cr.max_tokens, cr.temperature, cr.stream ? ", streamed" : "");
    double tp = now();
    if (!from) engine_reset(&eng);
    api_seq_ok = 0;
    if (engine_run(&eng, api_ids + from, n - from, from, (float)cr.temperature, seed++, &next_token)) {
        api_error(job->fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
        free(pool);
        pos = 0;
        api_chat_end();
        return;
    }
    say("api: %d prompt tokens read in %.2f s", n - from, now() - tp);
    memcpy(api_seq, api_ids, sizeof(uint32_t) * n);
    api_seq_n = pos = n;
    api_sys_n = ns;
    api_seq_spec = api_spec_on();
    api_seq_ok = 1;

    int fd = job->fd, streamed = 0, client_gone = 0, error = 0, tools = cr.tools >= 0;
    if (cr.stream) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
        if (api_send(fd, h, strlen(h))) client_gone = 1;
        streamed = 1;
        Sb first;
        sb_init(&first);
        api_json_chunk(&first, id, model, created, "assistant", "", 0, NULL);
        if (!client_gone && first.p && api_send_event(fd, first.p)) client_gone = 1;
        sb_free(&first);
    }

    // the reply (all of it), the bytes of the last tokens not yet a whole character, and (with tools) the white space
    // at the end of what was streamed, held back until more text follows: before a tool call it is dropped
    Sb text, pend, held;
    sb_init(&text);
    sb_init(&pend);
    sb_init(&held);
    int produced = 0, stopped = 0, in_call = 0;
    const char* finish = "length";
    // The tokens to write: the one a step drew, after those it kept from the MTP head's draft (the model has read
    // these already). Without the head a step is one token.
    uint32_t q[9];
    int nq = 1, iq = 0;
    q[0] = next_token;
    double t0 = now();
    api_chat_reply();
    while (produced < cr.max_tokens) {
        uint32_t t = q[iq++];
        if (t == IM_END || t == eng.eos2 || t == eng.eos) {
            finish = "stop";
            break;
        }
        const uint8_t* bytes = NULL;
        int nb = tok_piece(&tok, t, &bytes);
        if (nb <= 0) nb = api_special_text(t, &bytes);   // (<tool_call> and its kin, as text)
        produced++;
        if (nb > 0) {
            if (tools && !in_call && nb == 11 && !memcmp(bytes, "<tool_call>", 11)) in_call = 1;
            sb_add(&text, (const char*)bytes, nb);
            if (!in_call) sb_add(&pend, (const char*)bytes, nb);
        }
        if (streamed && pend.n) {
            size_t whole = api_utf8_whole((const uint8_t*)pend.p, pend.n);
            size_t keep = whole;
            if (tools)   // (trailing white space waits: it is not sent if a tool call comes next)
                while (keep > 0 && (pend.p[keep - 1] == ' ' || pend.p[keep - 1] == '\n' || pend.p[keep - 1] == '\t'))
                    keep--;
            if (keep) {
                Sb ev, out;
                sb_init(&ev);
                sb_init(&out);
                sb_add(&out, held.p, held.n);
                sb_add(&out, pend.p, keep);
                sb_free(&held);
                api_json_chunk(&ev, id, model, created, NULL, out.p, out.n, NULL);
                if (ev.p && api_send_event(fd, ev.p)) client_gone = 1;
                sb_free(&ev);
                sb_free(&out);
            }
            sb_add(&held, pend.p + keep, whole - keep);
            memmove(pend.p, pend.p + whole, pend.n - whole);
            pend.n -= whole;
            pend.p[pend.n] = 0;
        }
        if (client_gone || produced == cr.max_tokens) break;
        if (iq < nq) continue;   // (the next token is one the last step kept)
        if (api_chat_step(produced)) {   // Circle: the reply ends here, cut short as by max_tokens
            stopped = 1;
            break;
        }
        int ne = 0;   // tokens of the draft this step keeps
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {   // (never more than max_tokens: the step keeps at most what is left of them)
            if (engine_spec_step(&eng, t, pos, (float)cr.temperature, seed++, cr.max_tokens - produced + 1, q, &ne,
                                 &next_token)) {
                error = 1;
                break;
            }
        } else
#endif
        if (engine_run(&eng, &t, 1, pos, (float)cr.temperature, seed++, &next_token)) {
            error = 1;
            break;
        }
        api_seq[pos] = t;   // what the model has read: the token, then the draft's tokens it kept
        for (int k = 0; k < ne; k++) api_seq[pos + 1 + k] = q[k];
        pos += 1 + ne;
        api_seq_n = pos;
        q[ne] = next_token;
        nq = ne + 1, iq = 0;
    }
    if (error) api_cache_drop();
    api_last_t = now();
    double dt = now() - t0;
    // the reply's text and tool calls
    Sb content, calls;
    sb_init(&content);
    sb_init(&calls);
    int n_calls = 0;
    if (tools) n_calls = api_parse_reply(text.p ? text.p : "", text.n, pool, cr.tools, api_count, &content, &calls);
    else sb_add(&content, text.p, text.n);
    free(pool);
    if (n_calls && !stopped) finish = "tool_calls";
    say("api: %d tokens in %.1f s (%.1f tokens/s), %s", produced, dt, dt > 0 ? produced / dt : 0.0,
        error ? "engine error" : stopped ? "stopped on the console" : finish);
    if (n_calls) say("api: tool calls %.300s", calls.p);

    if (streamed) {
        if (!client_gone && !error) {
            if (!n_calls && held.n) {   // (no tool call came: the white space held back was part of the text)
                Sb ev;
                sb_init(&ev);
                api_json_chunk(&ev, id, model, created, NULL, held.p, held.n, NULL);
                if (ev.p) api_send_event(fd, ev.p);
                sb_free(&ev);
            }
            if (n_calls) {
                Sb ev;
                sb_init(&ev);
                api_json_chunk_calls(&ev, id, model, created, calls.p);
                if (ev.p) api_send_event(fd, ev.p);
                sb_free(&ev);
            }
            Sb last;
            sb_init(&last);
            api_json_chunk(&last, id, model, created, NULL, "", 0, finish);
            if (last.p) api_send_event(fd, last.p);
            sb_free(&last);
            api_send(fd, "data: [DONE]\n\n", 14);
        }
        if (error && !client_gone) {   // the headers are out: the error goes in the stream
            Sb e;
            sb_init(&e);
            api_json_error(&e, eng.error, "server_error", NULL);
            if (e.p) api_send_event(fd, e.p);
            sb_free(&e);
        }
    } else if (error) {
        api_error(fd, "HTTP/1.1 500 Internal Server Error", eng.error, "server_error", NULL);
    } else {
        Sb body, head;
        sb_init(&body);
        sb_init(&head);
        api_json_completion_calls(&body, id, model, created, content.p ? content.p : "", content.n,
                                  n_calls ? calls.p : NULL, finish, n, produced);
        sb_str(&head, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
        sb_int(&head, (long long)body.n);
        sb_str(&head, "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        if (head.p && body.p && api_send(fd, head.p, head.n) == 0) api_send(fd, body.p, body.n);
        sb_free(&body);
        sb_free(&head);
    }
    sb_free(&text);
    sb_free(&pend);
    sb_free(&held);
    sb_free(&content);
    sb_free(&calls);
    // the engine's state is this request's (kept for the next one, api_seq); the conversation on screen starts again
    // at position 0
    pos = 0;
    need_close = 0;
    api_chat_end();
}
