// Host tests for chat/api_proto.c (the OpenAI-compatible API's protocol part).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/api_proto_test.c chat/api_proto.c -o build/api_proto_test && build/api_proto_test
#include <stdio.h>
#include <string.h>

#include "api_proto.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void test_http(void) {
    HttpRequest r;
    const char* post = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: ps4\r\nContent-Type: application/json\r\n"
                       "content-length: 13\r\nAuthorization: Bearer abc\r\n\r\n{\"a\":\"b\"}    ";
    CHECK(api_http_parse(post, strlen(post), &r) == 1);
    CHECK(!strcmp(r.method, "POST"));
    CHECK(!strcmp(r.path, "/v1/chat/completions"));
    CHECK(r.content_length == 13);
    CHECK(r.body_len == 13 && !memcmp(r.body, "{\"a\":\"b\"}    ", 13));
    CHECK(!strcmp(r.auth, "Bearer abc"));

    const char* partial = "POST /x HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc";
    CHECK(api_http_parse(partial, strlen(partial), &r) == 0);
    const char* head_only = "GET /v1/models HTTP/1.1\r\nHost: a\r\n";
    CHECK(api_http_parse(head_only, strlen(head_only), &r) == 0);
    const char* get = "GET /v1/models HTTP/1.1\r\nHost: a\r\n\r\n";
    CHECK(api_http_parse(get, strlen(get), &r) == 1);
    CHECK(!strcmp(r.path, "/v1/models") && r.content_length == 0);
    const char* bad = "GET\r\n\r\n";
    CHECK(api_http_parse(bad, strlen(bad), &r) == -1);
    const char* badcl = "POST / HTTP/1.1\r\nContent-Length: nope\r\n\r\n";
    CHECK(api_http_parse(badcl, strlen(badcl), &r) == -1);
    const char* bigcl = "POST / HTTP/1.1\r\nContent-Length: 99999999\r\n\r\n";
    CHECK(api_http_parse(bigcl, strlen(bigcl), &r) == -1);
}

static void test_json(void) {
    JNode pool[64];
    const char* doc = "{ \"a\" : [1, 2.5, -3e2, true, false, null], \"s\": \"x\\\"y\\\\\\n\\u00e9\\ud83d\\ude00\" }";
    int root = json_parse(doc, strlen(doc), pool, 64);
    CHECK(root >= 0);
    int a = json_get(pool, root, "a");
    CHECK(a >= 0 && pool[a].type == J_ARR && pool[a].count == 6);
    int c = pool[a].first;
    CHECK(pool[c].type == J_NUM && pool[c].num == 1);
    c = pool[c].next;
    CHECK(pool[c].num == 2.5);
    c = pool[c].next;
    CHECK(pool[c].num == -300);
    c = pool[c].next;
    CHECK(pool[c].type == J_BOOL && pool[c].b == 1);
    int s = json_get(pool, root, "s");
    Sb out;
    sb_init(&out);
    json_str_sb(pool, s, &out);
    CHECK(out.n == strlen("x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80"));
    CHECK(out.p && !memcmp(out.p, "x\"y\\\n\xc3\xa9\xf0\x9f\x98\x80", out.n));
    sb_free(&out);

    CHECK(json_parse("{\"a\":}", 6, pool, 64) == -1);
    CHECK(json_parse("[1,2", 4, pool, 64) == -1);
    CHECK(json_parse("{} x", 4, pool, 64) == -1);
    CHECK(json_parse("[1,2]", 5, pool, 2) == -1);   // pool too small
    CHECK(json_get(pool, root, "missing") == -1);
}

static void test_chat(void) {
    JNode pool[API_MAX_NODES];
    ChatRequest req;
    const char* err = NULL;
    const char* body =
        "{\"model\":\"x\",\"messages\":[{\"role\":\"system\",\"content\":\"Be brief.\"},"
        "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"Hi \"},{\"type\":\"text\",\"text\":\"there\"}]}],"
        "\"temperature\":0.2,\"max_tokens\":64,\"stream\":true,\"seed\":7}";
    CHECK(api_chat_parse(body, strlen(body), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.n_messages == 2 && req.stream == 1 && req.max_tokens == 64 && req.has_seed && req.seed == 7);
    CHECK(req.temperature > 0.19 && req.temperature < 0.21);
    char role[16];
    Sb content;
    sb_init(&content);
    int m = pool[req.messages].first;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "system"));
    CHECK(content.n == 9 && !memcmp(content.p, "Be brief.", 9));
    sb_free(&content);
    sb_init(&content);
    m = pool[m].next;
    CHECK(api_message(pool, m, role, sizeof(role), &content) == 0 && !strcmp(role, "user"));
    CHECK(content.n == 8 && !memcmp(content.p, "Hi there", 8));
    sb_free(&content);

    const char* defaults = "{\"messages\":[{\"role\":\"user\",\"content\":\"yo\"}]}";
    CHECK(api_chat_parse(defaults, strlen(defaults), pool, API_MAX_NODES, &req, &err) == 0);
    CHECK(req.max_tokens == 512 && req.stream == 0 && !req.has_seed);

    const char* bad[] = {
        "[]",
        "{\"messages\":[]}",
        "{\"messages\":[1]}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"n\":2}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"temperature\":3}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"max_tokens\":0}",
        "{\"messages\":[{\"role\":\"user\",\"content\":\"a\"}],\"stream\":\"yes\"}",
        "{\"messages\":[{\"role\":\"user\"",
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        CHECK(api_chat_parse(bad[i], strlen(bad[i]), pool, API_MAX_NODES, &req, &err) == -1);
        CHECK(err != NULL);
    }
}

static void test_writers(void) {
    Sb s;
    sb_init(&s);
    api_json_completion(&s, "chatcmpl-1", "Qwen3.5-9B", 1700000000, "Hola \"mundo\"\n", 13, "stop", 5, 4);
    CHECK(s.p && strstr(s.p, "\"content\":\"Hola \\\"mundo\\\"\\n\"") != NULL);
    CHECK(strstr(s.p, "\"total_tokens\":9") != NULL);
    CHECK(strstr(s.p, "\"finish_reason\":\"stop\"") != NULL);
    sb_free(&s);

    sb_init(&s);
    api_json_chunk(&s, "c", "m", 1, "assistant", "", 0, NULL);
    CHECK(!strcmp(s.p, "{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\","
                       "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}"));
    sb_free(&s);

    sb_init(&s);
    api_json_error(&s, "bad \x01 input", "invalid_request_error", "bad_body");
    CHECK(!strcmp(s.p, "{\"error\":{\"message\":\"bad \\u0001 input\",\"type\":\"invalid_request_error\","
                       "\"code\":\"bad_body\"}}"));
    sb_free(&s);

    sb_init(&s);
    api_json_models(&s, "Qwen3.5-9B", 0);
    CHECK(strstr(s.p, "\"data\":[{\"id\":\"Qwen3.5-9B\",\"object\":\"model\"") != NULL);
    sb_free(&s);
}

static JNode tpool[API_MAX_NODES];

// the tool calls of a reply: arguments typed by the tools' schemas
static void test_tool_calls(void) {
    const char* body =
        "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"HassLightSet\",\"parameters\":{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},"
        "\"brightness\":{\"type\":\"integer\"},\"ratio\":{\"type\":\"number\"},\"on\":{\"type\":\"boolean\"},"
        "\"domain\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"extra\":{\"type\":\"object\"},"
        "\"volume\":{\"type\":[\"number\",\"null\"]},\"free\":{}}}}}]}";
    ChatRequest req;
    const char* err = NULL;
    CHECK(api_chat_parse(body, strlen(body), tpool, API_MAX_NODES, &req, &err) == 0 && req.tools >= 0);
    const char* reply =
        "  Setting it.  \n\n<tool_call>\n<function=HassLightSet>\n<parameter=name>\nKitchen \"Light\"\n</parameter>\n"
        "<parameter=brightness>\n50\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n<parameter=on>\nTrue\n"
        "</parameter>\n<parameter=domain>\n[\"light\", \"switch\"]\n</parameter>\n<parameter=extra>\n{\"a\": 1}\n"
        "</parameter>\n<parameter=volume>\n7\n</parameter>\n<parameter=free>\n[1,2]\n</parameter>\n<parameter=gone>\n"
        "line one\nline two\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=HassLightSet>\n"
        "<parameter=domain>\nlight\n</parameter>\n<parameter=brightness>\nlots\n</parameter>\n</function>\n</tool_call>";
    Sb content, calls;
    sb_init(&content);
    sb_init(&calls);
    int n = api_parse_reply(reply, strlen(reply), tpool, req.tools, 9, &content, &calls);
    CHECK(n == 2);
    CHECK(content.p && !strcmp(content.p, "Setting it."));
    CHECK(calls.p && strstr(calls.p, "[{\"index\":0,\"id\":\"call_9_0\",\"type\":\"function\",\"function\":"
                                     "{\"name\":\"HassLightSet\",\"arguments\":") == calls.p);
    // the arguments, as the JSON string they are sent in
    CHECK(strstr(calls.p, "\"{\\\"name\\\":\\\"Kitchen \\\\\\\"Light\\\\\\\"\\\",\\\"brightness\\\":50,\\\"ratio\\\":0.5,"
                          "\\\"on\\\":true,\\\"domain\\\":[\\\"light\\\", \\\"switch\\\"],\\\"extra\\\":{\\\"a\\\": 1},"
                          "\\\"volume\\\":7,\\\"free\\\":[1,2],\\\"gone\\\":\\\"line one\\\\nline two\\\"}\"") != NULL);
    // a list written without brackets becomes a list of one; a number that is not one stays text
    CHECK(strstr(calls.p, "{\"index\":1,\"id\":\"call_9_1\"") != NULL);
    CHECK(strstr(calls.p, "\"{\\\"domain\\\":[\\\"light\\\"],\\\"brightness\\\":\\\"lots\\\"}\"") != NULL);
    sb_free(&content);
    sb_free(&calls);

    // the older JSON form, no text before it, and an unclosed call at the end
    const char* js = "<tool_call>\n{\"name\": \"GetLiveContext\", \"arguments\": {}}\n</tool_call><tool_call>\n"
                     "<function=HassTurnOn>\n<parameter=name>\nFan";
    sb_init(&content);
    sb_init(&calls);
    CHECK(api_parse_reply(js, strlen(js), tpool, req.tools, 1, &content, &calls) == 2);
    CHECK(content.n == 0);
    CHECK(strstr(calls.p, "\"name\":\"GetLiveContext\",\"arguments\":\"{}\"") != NULL);
    CHECK(strstr(calls.p, "\"name\":\"HassTurnOn\",\"arguments\":\"{\\\"name\\\":\\\"Fan\\\"}\"") != NULL);
    sb_free(&content);
    sb_free(&calls);

    // no call: all of it is content; nothing in calls
    sb_init(&content);
    sb_init(&calls);
    CHECK(api_parse_reply("Just text.", 10, tpool, -1, 1, &content, &calls) == 0);
    CHECK(content.p && !strcmp(content.p, "Just text.") && calls.n == 0);
    sb_free(&content);
    sb_free(&calls);

    // tools: refused when malformed, left out with tool_choice none
    const char* bad = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":{}}";
    CHECK(api_chat_parse(bad, strlen(bad), tpool, API_MAX_NODES, &req, &err) == -1);
    const char* noname = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tools\":[{\"type\":\"function\"}]}";
    CHECK(api_chat_parse(noname, strlen(noname), tpool, API_MAX_NODES, &req, &err) == -1);
    const char* none = "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"tool_choice\":\"none\",\"tools\":"
                       "[{\"type\":\"function\",\"function\":{\"name\":\"f\"}}]}";
    CHECK(api_chat_parse(none, strlen(none), tpool, API_MAX_NODES, &req, &err) == 0 && req.tools == -1);

    // the writers
    Sb w;
    sb_init(&w);
    api_json_completion_calls(&w, "c", "m", 1, "", 0, "[{\"index\":0}]", "tool_calls", 3, 4);
    CHECK(w.p && strstr(w.p, "\"message\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0}]},"
                             "\"finish_reason\":\"tool_calls\"") != NULL);
    sb_free(&w);
    api_json_chunk_calls(&w, "c", "m", 1, "[{\"index\":0}]");
    CHECK(w.p && strstr(w.p, "\"delta\":{\"tool_calls\":[{\"index\":0}]},\"finish_reason\":null") != NULL);
    sb_free(&w);
}

// Python's json.dumps of a tool, as the template writes the tools (", " and ": ", numbers as written)
static void test_pyjson(void) {
    const char* doc = "{\"a\":[1,2.50,-3e2,true,null],\"s\":\"caf\\u00e9\\n\\t\\b\\f\\u0001\",\"o\":{}}";
    int r = json_parse(doc, strlen(doc), tpool, 64);
    Sb s;
    sb_init(&s);
    api_pyjson(&s, tpool, r);
    CHECK(s.p && !strcmp(s.p, "{\"a\": [1, 2.50, -3e2, true, null], \"s\": \"caf\xc3\xa9\\n\\t\\b\\f\\u0001\", \"o\": {}}"));
    sb_free(&s);
}

int main(void) {
    test_http();
    test_json();
    test_chat();
    test_writers();
    test_tool_calls();
    test_pyjson();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
