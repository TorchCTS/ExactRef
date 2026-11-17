/* What exactref asks of the operating system (internal). */
#if defined(__linux__)
#define _GNU_SOURCE
#endif
#if defined(_WIN32) && (!defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0601)
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 /* GetLogicalProcessorInformationEx */
#endif

#include "exactref.h"

#include <fenv.h>
#include <string.h>

#if defined(__linux__)
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#elif defined(_WIN32)
#include <stdlib.h>
#include <windows.h> /* after exactref.h, whose names windows.h's macros (small, far) would replace */
#else
#include <unistd.h>
#endif

#if defined(__linux__)
/* The physical cores of the system: distinct (package, core) pairs over its processors, or 0. */
static int linux_cores(void) {
    long count = sysconf(_SC_NPROCESSORS_CONF);
    if (count <= 0) return 0;
    long *ids = malloc(sizeof(long) * 2 * (size_t)count);
    if (!ids) return 0;
    int cores = 0;
    for (long cpu = 0; cpu < count; cpu++) {
        long id[2];
        const char *names[2] = {"physical_package_id", "core_id"};
        int ok = 1;
        for (int k = 0; k < 2 && ok; k++) {
            char path[96];
            snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/topology/%s", cpu, names[k]);
            FILE *f = fopen(path, "r");
            ok = f && fscanf(f, "%ld", &id[k]) == 1;
            if (f) fclose(f);
        }
        if (!ok) continue;
        int seen = 0;
        for (int c = 0; c < cores && !seen; c++) seen = ids[2 * c] == id[0] && ids[2 * c + 1] == id[1];
        if (!seen) {
            ids[2 * cores] = id[0];
            ids[2 * cores + 1] = id[1];
            cores++;
        }
    }
    free(ids);
    return cores;
}
#endif

/* The default number of worker threads, as PyTorch picks its own: on Apple Silicon its performance
   cores; elsewhere the physical cores, at most the processors the process may run on; at least 1. */
int physical_cores(void) {
    int n = 0;
#if defined(__APPLE__)
    size_t size = sizeof n;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &size, NULL, 0) == 0 && n > 1) return n;
    size = sizeof n;
    if (sysctlbyname("hw.physicalcpu", &n, &size, NULL, 0) != 0) n = 0;
#elif defined(__linux__)
    n = linux_cores();
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        int allowed = CPU_COUNT(&set);
        if (n <= 0 || allowed < n) n = allowed;
    }
#elif defined(_WIN32)
    DWORD size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &size);
    char *buffer = size ? malloc(size) : NULL;
    if (buffer && GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &size)) {
        for (DWORD at = 0; at < size; at += ((PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer + at))->Size) n++;
    }
    free(buffer);
    DWORD_PTR process, system;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) {
        int allowed = 0;
        for (; process; process &= process - 1) allowed++;
        if (n <= 0 || allowed < n) n = allowed;
    }
#else
    n = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return n > 0 ? n : 1;
}

/* The calling thread's floating-point environment (_fenv.ieee): saved as fenv_size() bytes, reset
   to the default (FE_DFL_ENV, which flushes no subnormal value) and restored. Each returns 0, or
   nonzero where the C library refused. The bytes go through a local fenv_t, so they need no
   alignment. Nothing here does floating-point arithmetic. */
size_t fenv_size(void) { return sizeof(fenv_t); }

int fenv_save(unsigned char *out) {
    fenv_t env;
    if (fegetenv(&env)) return 1;
    memcpy(out, &env, sizeof env);
    return 0;
}

int fenv_reset(void) { return fesetenv(FE_DFL_ENV) != 0; }

int fenv_restore(const unsigned char *in) {
    fenv_t env;
    memcpy(&env, in, sizeof env);
    return fesetenv(&env) != 0;
}
