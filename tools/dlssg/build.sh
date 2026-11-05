#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sdk="${DLSS_SDK:?Set DLSS_SDK to your local NVIDIA DLSS SDK}"
out="${1:?Pass an absolute runtime output directory}"
mkdir -p "$out"
bash ./test_loader.sh
gcc -shared -fPIC -O2 nvml_spoof.c -ldl -o "$out/nvml_spoof.so"
g++ -shared -fPIC -std=c++17 -O2 cubin_hook.cpp -ldl -o "$out/cubin_hook.so"
g++ -shared -fPIC -std=c++17 -O2 -DVK_ENABLE_BETA_EXTENSIONS -DNGX_SNIPPET_BUILD \
 -DNGX_ENABLE_DEPRECATED_SHUTDOWN bridge.cpp -I "$sdk/include" \
 "$sdk/lib/Linux_x86_64/libnvsdk_ngx.a" -Wl,-Bsymbolic -lvulkan -ldl -lpthread -o "$out/bridge.so"
