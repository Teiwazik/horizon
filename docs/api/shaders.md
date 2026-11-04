# Shader API

Horizon has no shader packs today: the earlier implementation was removed on 2026-10-24. Two things will replace it,
designed separately: a **Horizon shader API**, described here, and a clean-room loader for the existing pack format
(its own spec; it is not part of this API). Both sit on the same renderer hooks, agreed between the two.

## Overview: what a shader author will be able to do

Write WGSL against documented stages of Horizon's renderer, with typed inputs, options the player can change, a
reload as soon as a file is saved and an overlay that shows what each pass costs and draws. The renderer keeps
everything that makes it fast (culling, batching, LOD, the mesher); a pack changes how surfaces and light look, not
how the world is fed to the GPU.

## The design

- **A pack** is a folder or a zip with `horizon-shaders.toml` (id, name, version, the `api` line, the stages it
  uses, its options) and WGSL files. No preprocessor and no other shading language: WGSL is what wgpu validates and
  compiles for Vulkan, Metal and DirectX 12.
- **Stages, not pipelines.** A stage is a function with a fixed signature that the renderer calls inside its own pass,
  for example a terrain material (`fn material(input: TerrainInput) -> Surface`), the lighting of the visible
  surfaces, the sky, an optional shadow pass, and a chain of full-screen post passes ending in the frame. A pack
  that leaves a stage out gets vanilla's look for it.
- **Typed inputs.** Each stage receives structs (the frame: time, camera, sun and moon, weather, fog; the surface:
  position, normal, block and its light) whose WGSL is generated from the Rust types the renderer uploads. The
  generated reference lists every field with its unit and range, and a test fails when it drifts.
- **Options are WGSL `override` constants.** The manifest declares each option (on/off, a number in a range, a choice)
  with its default; the game shows them on the pack's settings screen and sets them when it builds the pipeline. No
  text substitution, and every option combination is validated the same way.
- **Hot reload.** The game watches the pack's folder. A saved file is validated with naga and the stage rebuilt; on an
  error the last good version keeps drawing and the error, with its file and line, is shown in the overlay.
- **Debug overlay.** GPU time per stage and pass (timestamp queries), the targets each pass writes (show any of them
  full-screen), the pack's options and its compile errors.
- **Budgets.** A pack declares the targets and passes it needs; the renderer refuses more than its limits, and the
  benchmark ([benchmark-report.md](../benchmark-report.md)) records the pack and its cost, so "fast" stays measured.
- **Off by default.** Without a pack the renderer is vanilla's look, and no pack code runs.

## The pack format (schema 1)

A pack is a folder, or the same files zipped as a `.hshader`, with `horizon-shaders.toml` at its root
(crate `horizon-shader-api`; the sample with every key is
`crates/horizon-shader-api/tests/samples/schema-1/`):

```toml
schema = 1                 # the manifest's format; a reader refuses schemas newer than it knows
id = "soft_light"          # [a-z0-9_]{1,64}: the settings key and the registry id
name = "Soft Light"
version = "1.0.0"
api = "0.5"                # the shader API line the WGSL is written for
description = "..."        # optional, like authors and license
shared = ["common.wgsl"]   # optional: WGSL put before every stage's file, in order

[sky]                      # each stage is optional, but a pack has at least one
file = "sky.wgsl"
clouds = true              # optional: false when the pack draws its own clouds
sun = true                 # optional: false when the pack draws its own sun (the moon and stars stay)
stars = true               # optional: false when the pack draws its own stars

[material]
file = "material.wgsl"
fog = true                 # optional: false when the pack fogs its surfaces itself

[shadow]                   # optional: a shadow map from the sun (the moon by night)
resolution = 2048          # its side in texels, 256 to 8192
distance = 128.0           # half the side of the square it covers, in blocks, 32 to 512
solid_materials = [2]      # optional: [[materials]] ids drawn whole, their cutouts ignored (solid leaf shadows)

[displacement]             # optional, needs [material]: LabPBR relief as geometry near the camera
distance = 12.0            # how far, in blocks, 4 to 32; the relief eases out over the last quarter
cells = 16                 # the grid's cells along a face's side, 4 to 64 (no more than the texels)
option = "displacement"    # optional: the bool option that turns it on
cells_option = "relief_cells"  # optional: a choice option of cell counts ("16", "32", ...), replacing cells
steps_option = "relief_steps"  # optional: a bool option: while on, each cell is a flat step with walls

[reflection]               # optional, needs [material]: when the water mirror (`reflection`) is drawn
option = "water_reflections"  # optional: drawn while this option is not 0 (else whenever the stage reads it)

[shadow]                   # optional, needs [material] or [[post]]: a shadow map from the sun (the moon by night)
resolution = 2048          # its side in texels, 256 to 8192
distance = 128.0           # half the side of the square it covers, in blocks, 32 to 512

[[post]]                   # full-screen passes, in the order they run (8 at most)
file = "post/grade.wgsl"
name = "Colour grade"      # optional: the GPU time overlay's label

[[post]]                   # a pass of its own target, at half or a quarter of the world's size
file = "post/glow.wgsl"
output = "glow"            # optional: a target the later passes read by this name; the
scale = 0.5                #   chain's `previous` goes on past it (the last pass writes none)

[[options]]                # 64 at most; `id` is the WGSL `override` it sets
id = "saturation"
name = "Saturation"
type = "float"             # bool, float (min, max, step), int (min, max), choice (values)
default = 1.1
min = 0.0
max = 2.0
step = 0.05

[[pages]]                  # optional: the settings screen's pages, the first the one it opens on
id = "main"
name = "Sample"
items = ["page:colour", ""]   # a button opening a page, an empty slot (or an option's id)

[[pages]]
id = "colour"
name = "Colour"            # the page's title and its button's label
description = "How colourful the picture is."   # optional: the button's tooltip
columns = 1                # optional: 1 or 2 (default) buttons a row
items = ["saturation"]

[budget]
gpu_ms = 1.5               # optional: the GPU time the pack aims to add (0 to 16 ms)

[reflection]               # optional, needs [material]: when the water mirror (`reflection`) is drawn
option = "reflections"     # optional: drawn while this option is not 0 (else whenever the material reads it)
```

An option's `override` has the type of its kind: `bool`, `f32`, `i32`, or `u32` for a choice (the index of the
value). The pack holds only the manifest, the `.wgsl` files it names (each used, UTF-8, 256 KiB at most),
`README.md`, `LICENSE` (`.md`, `.txt`) and `icon.png`, under plain paths, 8 MiB in all. A `.hshader` is built
the same way every time (files in path order, fixed dates), so a pack's SHA-256 depends only on its files.

## Guide, reference and examples

- [shaders-reference.md](shaders-reference.md): every stage's signature, input and output structs, resources and
  helpers, the frame's fields with their units, option types and limits, generated from `crates/horizon-shader-api`.
- [The guide](../shaders.md): making a pack with `horizon-shader new`, `check` and `pack`; the templates and the
  example pack (`crates/horizon-shader-api/packs/golden_hour`).

## Stability

Not started. The first stages will be experimental (opt-in), then preview once a reference, an example and a test
pack exist ([stability.md](stability.md)).

## Not there yet

Option values other than the defaults (the three stages run, with hot reload), then the overlay and lighting
stages.

## Changelog

- 2026-11-04: `[weather] after`: the `[[post]]` pass (by its `file`) after which rain and snow
  draw, over its picture, for a pack whose passes light the terrain again and would paint over
  streaks drawn with the world; `block_light_colors`' alpha is now the block light reaching the
  cell (spread from each source, stopped by walls) in place of how near the sources are.
- 2026-11-03: merged with main: `MaterialInput.tags` (0 on this branch) and `colored_light`
  (`[[particle_lights]]`, and held or dropped items in their `[[lights]]` colour) join the
  material stage's input, in main's order, `displaced` last.
- 2026-10-30: API 0.5 (additions): the water mirror the material stage reads as `reflection`
  is drawn only for a stage that reads it, and with `[reflection] option` only while that
  option is not 0 (`has_reflection` is 0 otherwise): a world drawn again saved when unused.
- 2026-10-30: API 0.5 (additions): `[[post]] output` and `scale`: a pass may write a target
  of the pack's own (up to 4, read by name by the passes after it, last frame's before it is
  written) at the world's size, half or a quarter, without breaking the chain of `previous`,
  so blurs and marches run on a fraction of the pixels.
- 2026-10-29: API 0.5 (additions): `[displacement]`: within its distance, the solid faces
  whose resource pack textures have LabPBR heights are drawn as a grid whose corners the
  material stage's `fn displacement(input: DisplacementInput) -> f32` moves along the normal,
  measured from the texture's mean height and pinned at the face's edges (relief that is
  geometry, with its depth and self-occlusion); `MaterialInput.displaced` says how much of it
  a point has, for a parallax effect to step aside.
- 2026-10-29: API 0.5 (additions): the material prelude's `block_texture`, `block_texture_size`
  and `block_texture_lod` (the surface's own sprite, for normals and detail generated from it);
  `[[pages]]` for the settings screen; `[sky] sun` and `stars`; `[shadow] solid_materials`; in
  first person the player's model casts into the shadow map; the `[weather]` stage
  (`fn weather(input: WeatherInput) -> vec4<f32>`) colours rain and snow streaks; under a pack
  that fogs its terrain itself, entities mark the frame's alpha 0 for its post stage to fog them;
  `block_texture_level` (the sprite at a chosen mip level); `[[materials]] blocks` (ids by block,
  as block.properties writes them, for the textures no `sprites` pattern names), up to 1024
  entries; a stage's `files` (more WGSL files of the pack, joined after the shared ones).
- 2026-10-29: API 0.5 (additions): the material prelude's `set_surface(normal, smoothness, data)`
  and the post stage's `scene_surface` with `surface_at` (the solid terrain's normal, smoothness
  and a value of the pack's own, where the picture still shows that terrain), for reflections
  and other deferred effects; a pack may chain 12 post passes.
- 2026-10-29: API 0.5 (additions): the frame's `wetness`, `biome_rain`, `biome_snow` (eased, as
  Iris's `wetness` and Complementary's `inRainy`/`inSnowy`) and `moon_phase`.
- 2026-10-29: API 0.5 (additions): `[[materials]] translucent = true` draws that id's textures in
  the translucent pass, clear texels included (Iris's `layer.translucent`).
- 2026-10-29: API 0.5 (additions): the material stage's `opaque_color` (the world before
  translucent terrain, for screen-space reflections on glass), `input.tangent`, and the
  resource packs' LabPBR maps: `block_pbr_maps`, `block_normal_map`, `block_specular_map` and
  their `_level` forms, and `block_pbr_size`.
- 2026-10-29: API 0.5 (additions): the frame's `eye_light_smooth` (Iris's
  `eyeBrightnessSmooth`), `in_water` and `in_lava` (eased).
- 2026-10-28: API 0.5: `[[materials]]` (texture globs to the material stage's `material` id),
  the `[vertex]` stage (terrain and shadow positions), `[[textures]]` (PNGs bound for every
  stage), the material stage's `emission` and `opaque_depth`, the post stage's `history` and the
  frame's `previous_view_projection`, `camera_motion` and `delta_time`; `[sky] clouds = false`
  hides Minecraft's clouds for a pack that draws its own; the water plane is looked for 24
  blocks around and below the camera; entities cast into the shadow map.
- 2026-10-28: API 0.4: `[shadow]`: the engine draws the solid terrain the light sees into a
  distorted depth map each frame (the shadow camera of `shaders/` packs), which the material and
  post stages read as `shadow_map` with `shadow_sampler` and the prelude's `shadow_position`; the
  frame's `light_direction`, `shadow_view_proj`, `shadow_distance`, `shadow_distortion` and
  `has_shadow`; the material stage's `normal`; `[material] fog = false` turns Horizon's fog off
  for a pack that fogs its surfaces itself.
- 2026-10-28: API 0.3: the material stage's `screen` input and the frame's `reflection` buffer
  (`water_plane`, `has_reflection`): the engine renders the world as a camera mirrored about the
  water plane into a half-res target, and the material stage samples it (a planar water mirror;
  an analytic sky stands in at the target's edges and far away).
- 2026-10-26: the material stage in the game: terrain's surfaces through the pack's `material`.
- 2026-10-26: the sky stage in the game: the pack's `sky` gets vanilla's colour along each ray.
- 2026-10-26: the post stage in the game (`HORIZON_SHADER_PACK`, hot reload with the last good pack kept).
- 2026-10-25: `horizon-shader` (`new`, `check`, `pack`), three templates, the example pack and the guide.
- 2026-10-25: the check (naga validation, signatures, options, budgets, SPIR-V/Metal/HLSL translation).
- 2026-10-25: the stages of API 0.1 (sky, material, post), their inputs and the generated reference.
- 2026-10-25: the pack format, schema 1: `horizon-shaders.toml` and the `.hshader` package.
- 2026-10-24: the earlier shader pack code was removed; this API is designed from scratch.

## Particles as lights

`[[particle_lights]]` makes particles light their surroundings on a device with ray tracing:

```toml
[[particle_lights]]
particles = ["firefly"]        # names, `firefly` or `minecraft:firefly`
color = [0.30, 0.35, 0.17]     # colour and brightness, 0 to 4 per channel
reach = 4.0                    # blocks at full alpha, up to 16; follows the particle's alpha
size = 0.08                    # half the source's size in blocks, up to 1: smaller is sharper
```

The nearest 64 shine. The first entry naming a particle wins.
