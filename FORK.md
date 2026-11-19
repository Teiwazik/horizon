# Teiwazik's Horizon fork: ray tracing for shader packs

A fork of [horizon-rust/horizon](https://github.com/horizon-rust/horizon) with an experimental ray-traced
renderer for shader packs. **It does not track upstream's shader API 0.2**: packs written for it use a prototype
API (`api = "0.5"`, a `[rays]` section and `ray_*` functions) that upstream does not have. No shader pack, mod or
native mod is included; this is the engine and its API only. The branch is a snapshot of work in progress.

## What the fork adds over upstream main

- **Ray functions for packs:** a ray scene of the world, entities and particles; traced shadows, reflections,
  refraction through glass and ice, and the water's own rays; `ray_light*` for the light sources around the camera
  (a light grid), `ray_noise` (a blue-noise tile), bounced light from a probe grid (DDGI), ReSTIR for block light.
- **Entities in the pack's surface targets:** an entity surface pass (Rgba32Uint), entities lit by the pack's own
  lighting like terrain, entity light passes and ray shadows (first-person body and held item included).
- **TAA after a native shader pack,** with upscaling and sharpening, reading entities' own motion.
- **DLSS and frame generation** after a pack's picture (kept out of upstream: there DLSS is a native mod).
- **Pack options:** up to 128 options per pack, 40 post passes and 24 post targets (upstream: 12 and 16),
  `[[presets]]` with a presets page, `[[particle_lights]]`, a `[rays]` switch for the ray scene.
- **Rendering:** a separate `pbrBudgetMiB` for LabPBR maps, line clipping at the near plane in world primitives,
  paged maps, smooth block light, a waterline from the eye's own water column, a motion pass for animated terrain.
- **Integrated branches:** DualSense haptics, joining older (1.21.9 and 1.21.10) servers, and other branches of the
  author that are in flight upstream.

## What it is not

No release, no stability promise, no support. Small independent pieces are sent upstream as separate pull requests
(`pbrBudgetMiB`, near-plane line clipping and others are being prepared); expect this fork to shrink as they land.
Builds are made from source with the usual Horizon toolchain.
