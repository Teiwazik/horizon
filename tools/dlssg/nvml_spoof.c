// Process-local experiment. Reports Ada only to clients asking NVML for architecture.
// The real driver, firmware and system libraries remain untouched.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
static _Atomic unsigned scope_depth;
void horizon_dlssg_scope(int enter) {
    if (enter) atomic_fetch_add(&scope_depth,1);
    else atomic_fetch_sub(&scope_depth,1);
}
int horizon_dlssg_in_scope(void) { return atomic_load(&scope_depth)!=0; }
static int (*original_arch)(void *, unsigned *);
static int architecture(void *device, unsigned *out) {
    Dl_info caller = {0};
    dladdr(__builtin_return_address(0), &caller);
    if (getenv("HORIZON_DLSSG_TRACE_CALLERS")) fprintf(stderr,"nvml caller=%s\n",caller.dli_fname?caller.dli_fname:"unknown");
    int status = original_arch(device, out);
    // SR already supports Ampere. Spoof only the FG runtime/its scoped callbacks.
    int fg = horizon_dlssg_in_scope() || (caller.dli_fname && strstr(caller.dli_fname,"libnvidia-ngx-dlssg"));
    if (!status && out && fg) {
        unsigned target=getenv("DLSSG_TEST_ARCH")?atoi(getenv("DLSSG_TEST_ARCH")):10;
        fprintf(stderr, "prototype NVML architecture: %u -> %u\n", *out,target);
        *out = target;
    }
    return status;
}
// A transparent path MUST tail-call libc: RTLD_NEXT is relative to the calling
// object. Calling libc here and then examining its result makes libc see this
// interposer as the caller and can resolve NVIDIA's private symbols incorrectly.
__attribute__((optimize("O2", "optimize-sibling-calls")))
void *dlsym(void *handle, const char *name) {
    void *(*real_dlsym)(void *, const char *) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    if (handle != RTLD_NEXT && getenv("DLSSG_PROTOTYPE_SPOOF_NVML") &&
            !strcmp(name, "nvmlDeviceGetArchitecture")) {
        void *result = real_dlsym(handle, name);
        if (result) { original_arch = result; return (void *)architecture; }
        return result;
    }
    return real_dlsym(handle, name);
}
