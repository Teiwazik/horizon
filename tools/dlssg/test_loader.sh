#!/usr/bin/env bash
# Check the real ELF loader contract, without an NVIDIA driver or GPU.
set -euo pipefail
source_dir=$(cd "$(dirname "$0")" && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
cat > "$scratch/a.c" <<'C'
#define _GNU_SOURCE
#include <dlfcn.h>
int sample_symbol(void) { return 11; }
int route_next(void) {
    int (*next)(void) = dlsym(RTLD_NEXT, "sample_symbol");
    return next ? next() : -1;
}
C
cat > "$scratch/b.c" <<'C'
int sample_symbol(void) { return 22; }
int nvmlDeviceGetArchitecture(void *device, unsigned *out) {
    (void)device; *out=7; return 0;
}
C
cat > "$scratch/main.c" <<'C'
#define _GNU_SOURCE
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
extern int route_next(void);
int main(void) {
    assert(route_next()==22);
    void *library=dlopen("libtest_b.so",RTLD_NOW);
    assert(library);
    int (*architecture)(void *,unsigned *)=dlsym(library,"nvmlDeviceGetArchitecture");
    void (*scope)(int)=dlsym(RTLD_DEFAULT,"horizon_dlssg_scope");
    assert(architecture && scope);
    unsigned value=0;
    assert(architecture(NULL,&value)==0 && value==7);
    scope(1);
    assert(architecture(NULL,&value)==0 && value==10);
    scope(0);
    assert(architecture(NULL,&value)==0 && value==7);
    puts("RTLD_NEXT forwarding and scoped NVML spoof: passed");
    dlclose(library);
}
C
gcc -shared -fPIC -O2 "$source_dir/nvml_spoof.c" -ldl -o "$scratch/shim.so"
gcc -shared -fPIC "$scratch/a.c" -ldl -o "$scratch/libtest_a.so"
gcc -shared -fPIC "$scratch/b.c" -o "$scratch/libtest_b.so"
gcc "$scratch/main.c" -L "$scratch" -Wl,--no-as-needed -ltest_a -ltest_b -ldl \
    -Wl,-rpath,"$scratch" -o "$scratch/test"
DLSSG_PROTOTYPE_SPOOF_NVML=1 LD_PRELOAD="$scratch/shim.so" "$scratch/test"
