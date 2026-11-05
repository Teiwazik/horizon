# Native DLSS-G local runtime

Linux/Vulkan, RTX 3050 Ti SM86 experiment. This is a local runtime build, not
NVIDIA binaries for distribution. `DLSS_SDK=... ./build.sh /absolute/output` builds
the bridge and process-local NVML/CUDA interposers. Prepare exact-byte Ampere
replacements from the user's own matching SDK (prepare_twins.py, bootstrap module
dumps). Configure HORIZON_DLSSG_BRIDGE and DLSSG_PROTOTYPE_REPLACEMENTS, and preload
the two interposers. HORIZON_DLSS_FG=1 enables the optional device extensions.
Frame Generation selection and 2x/3x/4x are saved in horizon-graphics.txt.
FSR fallback is reported as FSR 2x, never as the requested DLSS multiplier.
The ABI is version 2; frame count is the number of interpolated images (1..3).
The bridge stays mapped until process exit because NGX retains callbacks.

The NVML/CUDA hooks are scoped to FG calls. SR keeps its stock Ampere path.
`bash test_loader.sh` checks transparent RTLD_NEXT forwarding and scoped NVML
behavior using independent shared libraries, without a GPU. The build runs it too.
The SR SDK is explicitly released after all features stop, before Vulkan device
and driver teardown; the FG bridge provides a separate final runtime shutdown.

Presentation bounds render-ahead on the GPU and uses two texture groups, without
an extra completed group waiting in a CPU channel. For controlled comparisons,
`HORIZON_FRAMEGEN_LOW_LATENCY=0` restores the older three-group scheduling.
`HORIZON_FRAMEGEN_LATENCY_LOG=1` reports CPU handoff waiting; this is not a
measurement of mouse-to-display latency. `HORIZON_SHADER_PACK_TIMING=1` enables
GPU pass timings, including a separate DLSS SR span; `horizon-ctl bench` includes
both `gpu_passes` and `shader_gpu_passes` while timing is enabled.

DLSS SR uses the SDK texture-detail mip bias, `-log2(upscale) - 1`, while it is
actually running. `HORIZON_DLSS_TEXTURE_DETAIL_BIAS=0` restores the previous
conservative correction for comparisons; the default extra bias is -1. This
changes texture sampling, not the frame-generation model or the sharpening slider.
