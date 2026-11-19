// Host stand-in for the console's <orbis/libkernel.h> (tests only): threads, mutexes and sleeps over pthreads.
#pragma once
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

typedef pthread_t OrbisPthread;
typedef pthread_attr_t OrbisPthreadAttr;
typedef pthread_mutex_t OrbisPthreadMutex;
typedef pthread_mutexattr_t OrbisPthreadMutexattr;

static inline int32_t sceKernelUsleep(uint32_t us) { return usleep(us); }
static inline int32_t scePthreadAttrInit(OrbisPthreadAttr* a) { return pthread_attr_init(a); }
static inline int32_t scePthreadAttrSetstacksize(OrbisPthreadAttr* a, size_t n) { return pthread_attr_setstacksize(a, n); }
static inline int32_t scePthreadCreate(OrbisPthread* t, const OrbisPthreadAttr* a, void* (*f)(void*), void* arg,
                                       const char* name) {
    (void)name;
    return pthread_create(t, a, f, arg);
}
static inline int32_t scePthreadDetach(OrbisPthread t) { return pthread_detach(t); }
static inline int32_t scePthreadJoin(OrbisPthread t, void** r) { return pthread_join(t, r); }
static inline int32_t scePthreadMutexInit(OrbisPthreadMutex* m, const OrbisPthreadMutexattr* a, const char* name) {
    (void)name;
    return pthread_mutex_init(m, a);
}
static inline int32_t scePthreadMutexLock(OrbisPthreadMutex* m) { return pthread_mutex_lock(m); }
static inline int32_t scePthreadMutexUnlock(OrbisPthreadMutex* m) { return pthread_mutex_unlock(m); }
