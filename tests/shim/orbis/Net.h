// Host stand-in for the console's <orbis/Net.h> (tests only): the few sceNet calls wyoming.c uses, over POSIX
// sockets. The console's sockaddr and socket options are BSD's, so they are translated to Linux's here.
#pragma once
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

typedef int32_t OrbisNetId;
typedef uint32_t OrbisNetSocklen_t;
typedef struct OrbisNetSockaddr {
    uint8_t len;
    uint8_t sa_family;
    char sa_data[14];
} OrbisNetSockaddr;
enum { ORBIS_NET_AF_INET = 2, ORBIS_NET_SOCK_STREAM = 1, ORBIS_NET_SOCK_DGRAM = 2 };

static inline void shim_to_linux(const OrbisNetSockaddr* a, struct sockaddr_in* l) {
    memset(l, 0, sizeof(*l));
    l->sin_family = AF_INET;
    memcpy(&l->sin_port, (const uint8_t*)a + 2, 2);
    memcpy(&l->sin_addr, (const uint8_t*)a + 4, 4);
}
static inline OrbisNetId sceNetSocket(const char* name, int32_t family, int32_t type, int proto) {
    (void)name, (void)family, (void)proto;
    return socket(AF_INET, type == ORBIS_NET_SOCK_STREAM ? SOCK_STREAM : SOCK_DGRAM, 0);
}
static inline int32_t sceNetSetsockopt(OrbisNetId s, int32_t level, int32_t opt, const void* v, OrbisNetSocklen_t n) {
    (void)level;
    int lo = opt == 0x0004 ? SO_REUSEADDR : opt == 0x1006 ? SO_RCVTIMEO : opt == 0x1005 ? SO_SNDTIMEO : -1;
    return lo < 0 ? -1 : setsockopt(s, SOL_SOCKET, lo, v, n);
}
static inline int32_t sceNetBind(OrbisNetId s, const OrbisNetSockaddr* a, OrbisNetSocklen_t n) {
    (void)n;
    struct sockaddr_in l;
    shim_to_linux(a, &l);
    const char* port = getenv("WY_TEST_PORT");   // (tests listen on a port of their own)
    if (port) l.sin_port = htons((uint16_t)atoi(port));
    return bind(s, (struct sockaddr*)&l, sizeof(l));
}
static inline int32_t sceNetListen(OrbisNetId s, int b) { return listen(s, b); }
static inline OrbisNetId sceNetAccept(OrbisNetId s, OrbisNetSockaddr* a, OrbisNetSocklen_t* n) {
    struct sockaddr_in l;
    socklen_t ln = sizeof(l);
    int fd = accept(s, (struct sockaddr*)&l, &ln);
    if (fd >= 0 && a) {
        memset(a, 0, sizeof(*a));
        a->len = 16, a->sa_family = 2;
        memcpy((uint8_t*)a + 2, &l.sin_port, 2);
        memcpy((uint8_t*)a + 4, &l.sin_addr, 4);
        if (n) *n = 16;
    }
    return fd;
}
static inline int32_t sceNetRecv(OrbisNetId s, void* b, size_t n, int f) { return (int32_t)recv(s, b, n, f); }
static inline int32_t sceNetSend(OrbisNetId s, const void* b, size_t n, int f) {
    return (int32_t)send(s, b, n, f | MSG_NOSIGNAL);
}
static inline int32_t sceNetSocketClose(OrbisNetId s) { return close(s); }
static inline int32_t sceNetSocketAbort(OrbisNetId s, int f) {
    (void)f;
    return shutdown(s, SHUT_RDWR);
}
static inline uint16_t sceNetHtons(uint16_t v) { return htons(v); }
