//! The material stage of Horizon's own shader packs (docs/shaders.md): the pack's `material`
//! function gives the lit colour of each point of a block's surface, before the alpha test and
//! fog. Terrain still draws with Horizon's meshes, culling and draw lists; only its pipelines
//! change. Their shader is the pack's stage (`CheckedStage::source`), then a copy of
//! terrain.wgsl's terrain section whose names all take a `horizon_terrain_` prefix
//! (`wgsl_prefix.rs`), then entries that hand the pack what vanilla's `fs_solid` and
//! `fs_translucent` combine: the texture, the vertex tint, the light levels and the lightmap's
//! colour. They read the terrain's group 0, the flat draws' group 1 where those run, and the
//! pack's frame at group 3.

use horizon_shader_api::{check::CheckedStage, stages::GROUP};

use super::native_post::{NativeFrame, present_constants};
use super::wgsl_prefix::{TERRAIN, prefixed, section};

const PREFIX: &str = "horizon_terrain_";

/// The entries around the pack's `material`: vanilla's vertex work without the lightmap
/// product, then the material, the alpha test and fog as `fs_solid` and `fs_translucent` do.
const ENTRIES: &str = "
// The manifest's `[material] fog`: false when the pack fogs its surfaces itself.
override horizon_material_fog: bool = true;
fn horizon_material_fogged(color: vec4<f32>, spherical: f32, cylindrical: f32) -> vec4<f32> {
    if horizon_material_fog {
        return horizon_terrain_apply_fog(color, spherical, cylindrical);
    }
    return color;
}
struct HorizonMaterialOut {
    // Invariant: the depth pre-pass and the material reach the same depth.
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) tint: vec4<f32>,
    @location(3) light: vec2<f32>,
    @location(4) lightmap: vec3<f32>,
    @location(5) position: vec3<f32>,
    @location(6) spherical: f32,
    @location(7) cylindrical: f32,
    @location(8) @interpolate(flat) face: u32,
    @location(9) @interpolate(flat) fluid_quad: u32,
    @location(10) @interpolate(flat) emission: f32,
    // The point on the flat face (`position` without `[displacement]`'s relief), how much of
    // that relief it has (`MaterialInput.displaced`) and the face's normal (0 when the
    // fragment finds it).
    @location(11) base: vec3<f32>,
    @location(12) displaced: f32,
    @location(13) @interpolate(flat) normal: vec3<f32>,
    // terrain.wgsl's `emission`, for HDR output's mark.
    @location(14) @interpolate(flat) hdr_code: u32,
}
fn horizon_material_vertex(position: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>, origin: vec3<f32>) -> HorizonMaterialOut {
    let pos = horizon_material_displace(position + origin, uv, layer, color, light);
    var out = horizon_material_out(pos, uv, layer, color, vec2<f32>(light));
    out.base = pos;
    return out;
}
// `light`: sky, block, 16 per level.
fn horizon_material_out(pos: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<f32>) -> HorizonMaterialOut {
    var out: HorizonMaterialOut;
    out.clip = horizon_terrain_globals.view_proj * vec4(pos, 1.0);
    out.uv = uv;
    out.layer = layer;
    out.tint = vec4(color.rgb, 1.0);
    let block = max(light.y, horizon_terrain_dynamic_block_light(pos));
    let sky = light.x;
    // 16 per level, 15 levels.
    out.light = min(vec2(block, sky) / 240.0, vec2(1.0));
    out.lightmap = horizon_terrain_sample_lightmap(sky / 16.0, block / 16.0).rgb;
    out.position = pos;
    out.spherical = length(pos);
    out.cylindrical = max(length(pos.xz), abs(pos.y));
    // The colour's alpha byte (mesher.rs's `fluid_face` and `emitting`): a fluid quad's
    // high nibble is 8 + its face (0..5), every other quad keeps 0xF there; the low nibble
    // is 15 minus the light the block emits. Every corner of a quad carries the same byte,
    // so the interpolated value rounds back exactly.
    let flags = u32(round(color.a * 255.0));
    let high = flags >> 4u;
    let fluid = high >= 8u && (high & 7u) < 6u;
    out.face = select(0u, high & 7u, fluid);
    out.fluid_quad = u32(fluid);
    // The tier (0 none, 1 low, 2 incandescent, 3 bright, 4 very bright) as how strongly the
    // face shines: a redstone torch 7/15, a torch 14/15, glowstone 1.
    out.hdr_code = 15u - (flags & 15u);
    // Codes above 4 are the same tiers, shining all over rather than in their bright parts.
    let tier = select(out.hdr_code, out.hdr_code - 4u, out.hdr_code > 4u);
    out.emission = select(min(f32(tier) * 7.0 / 15.0, 1.0), 0.0, tier == 0u);
    out.displaced = 0.0;
    out.normal = vec3(0.0);
    return out;
}
@vertex fn horizon_material_vs(
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>,
    @location(5) origin: vec4<f32>) -> HorizonMaterialOut {
    return horizon_material_vertex(position, horizon_terrain_terrain_uv(uv, origin.w), layer,
        color, light, origin.xyz);
}
@vertex fn horizon_material_vs_flat(
    @builtin(vertex_index) vertex: u32,
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>
) -> HorizonMaterialOut {
    let origin = horizon_terrain_slot_origins[horizon_terrain_block_slots[vertex >> 8u]];
    return horizon_material_vertex(position, horizon_terrain_terrain_uv(uv, origin.w), layer,
        color, light, origin.xyz);
}
// The prelude's `block_normal_map` and `block_specular_map`: the resource packs' LabPBR maps
// in the pack's frame group (3), one array each, a blank 1x1 without them.
@group(3) @binding(12) var horizon_pbr_normals: texture_2d_array<f32>;
@group(3) @binding(13) var horizon_pbr_specular: texture_2d_array<f32>;
fn horizon_block_pbr_present() -> bool {
    return textureDimensions(horizon_pbr_normals).x > 1u;
}
fn horizon_block_pbr_size() -> vec2<f32> {
    return vec2<f32>(textureDimensions(horizon_pbr_normals).xy);
}
fn horizon_block_pbr(uv: vec2<f32>, layer: u32, lod: f32, map: u32) -> vec4<f32> {
    if !horizon_block_pbr_present() || layer >= textureNumLayers(horizon_pbr_normals) {
        return select(vec4<f32>(0.5, 0.5, 1.0, 1.0), vec4<f32>(0.0), map == 1u);
    }
    let s = horizon_terrain_sprite_sampler;
    // The maps may be smaller than the colour: the same footprint is fewer of their texels.
    let scale = f32(textureDimensions(horizon_terrain_sprites).x)
        / f32(textureDimensions(horizon_pbr_normals).x);
    let l = max(lod - log2(scale), 0.0);
    if map == 0u {
        return textureSampleLevel(horizon_pbr_normals, s, uv, i32(layer), l);
    }
    return textureSampleLevel(horizon_pbr_specular, s, uv, i32(layer), l);
}
// The prelude's `block_texture`: a sprite of the terrain's arrays at a mip level.
fn horizon_block_size() -> vec2<f32> {
    return vec2<f32>(textureDimensions(horizon_terrain_sprites).xy);
}
fn horizon_block_sample(uv: vec2<f32>, layer: u32, lod: f32) -> vec4<f32> {
    let s = horizon_terrain_sprite_sampler;
    let l = i32(layer % horizon_terrain_SPRITE_SPLIT);
    switch layer / horizon_terrain_SPRITE_SPLIT {
        case 0u: { return textureSampleLevel(horizon_terrain_sprites, s, uv, l, lod); }
        case 1u: { return textureSampleLevel(horizon_terrain_sprites_1, s, uv, l, lod); }
        case 2u: { return textureSampleLevel(horizon_terrain_sprites_2, s, uv, l, lod); }
        case 3u: { return textureSampleLevel(horizon_terrain_sprites_3, s, uv, l, lod); }
        case 4u: { return textureSampleLevel(horizon_terrain_sprites_4, s, uv, l, lod); }
        case 5u: { return textureSampleLevel(horizon_terrain_sprites_5, s, uv, l, lod); }
        case 6u: { return textureSampleLevel(horizon_terrain_sprites_6, s, uv, l, lod); }
        default: { return textureSampleLevel(horizon_terrain_sprites_7, s, uv, l, lod); }
    }
}
fn horizon_material_color(in: HorizonMaterialOut, translucent: bool) -> vec4<f32> {
    let albedo = horizon_terrain_sprite(in.uv, in.layer & 0x7fffu);
    // What the prelude's `block_texture` reads: this sprite, at this point's mip level.
    horizon_block_layer = in.layer & 0x7fffu;
    horizon_ray_pixel = in.clip.xy;
    horizon_material_albedo = albedo.rgb * in.tint.rgb;
    let texels = in.uv * horizon_block_size();
    let footprint = max(dot(dpdx(texels), dpdx(texels)), dot(dpdy(texels), dpdy(texels)));
    horizon_block_lod = max(0.5 * log2(max(footprint, 1e-8)), 0.0);
    // Water draws translucent and lava solid, so the pass resolves the fluid kind.
    let fluid = select(0u, select(2u, 1u, translucent), in.fluid_quad != 0u);
    let screen = in.clip.xy / frame.resolution;
    let material_id = material_ids[min(in.layer & 0x7fffu, arrayLength(&material_ids) - 1u)];
    // Block faces are flat: the derivatives of the position give the face's normal, turned
    // toward the camera. Far away they lose precision, so a normal within about 3.6 degrees of
    // an axis (every full block face) is that axis exactly: no shimmer on distant faces. The
    // gentlest slope of flowing water, one level over a block, tilts about 6 degrees.
    let n = normalize(cross(dpdx(in.position), dpdy(in.position)));
    let a = abs(n);
    let m = max(a.x, max(a.y, a.z));
    let axis = select(select(vec3(0.0, 0.0, sign(n.z)), vec3(0.0, sign(n.y), 0.0), a.y == m),
        vec3(sign(n.x), 0.0, 0.0), a.x == m);
    let flat_n = select(select(n, axis, m > 0.998), in.normal, dot(in.normal, in.normal) > 0.5);
    let normal = select(flat_n, -flat_n, dot(flat_n, in.base) > 0.0);
    horizon_material_normal = normal;
    horizon_surface = vec4(0.0);
    // Dynamic lights (a dropped torch) per pixel: a quad's corners can lie far from one.
    let dynamic = horizon_terrain_dynamic_block_light(in.position) / 240.0;
    let light = vec2(max(in.light.x, min(dynamic, 1.0)), in.light.y);
    let tangent = horizon_material_tangent(in.base, in.uv, normal);
    // No sprite tags on this branch: its materials come from `material_ids`.
    return material(MaterialInput(albedo, in.tint, light, in.lightmap, in.position, in.uv,
        translucent, horizon_sprite_tags(in.layer & 0x7fffu), material_id, screen, in.face, fluid, in.emission, normal, tangent,
        horizon_terrain_native_colored_light(in.position), in.displaced)).color;
}
var<private> horizon_material_normal: vec3<f32>;
var<private> horizon_material_albedo: vec3<f32>;
struct HorizonMaterialSolid {
    @location(0) color: vec4<f32>,
    // The world target's surface data: `set_surface`'s, then the fragment's depth, which the
    // post stage's `surface_at` compares with the final depth.
    @location(1) surface: vec4<f32>,
    // The texture and tint before any light, for light the post stage adds.
    @location(2) albedo: vec4<f32>,
}
// What `fs_solid` leaves in the frame's alpha for HDR output, from the sprite before the pack's
// material: the pack decides the colour, Horizon still knows which pixels are light sources.
fn horizon_material_mark(in: HorizonMaterialOut, texel: vec4<f32>) -> f32 {
    if in.hdr_code > 0u {
        let whole = in.hdr_code > 4u;
        let tier = select(in.hdr_code, in.hdr_code - 4u, whole);
        let l = in.layer & 0x7fffu;
        let t = textureLoad(horizon_terrain_emission_thresholds,
            vec2(i32(l % horizon_terrain_SPRITE_SPLIT), i32(l / horizon_terrain_SPRITE_SPLIT)), 0);
        let y = horizon_terrain_brightness(texel.rgb);
        if whole || y >= t.r {
            return horizon_terrain_hdr_mark(tier, y);
        }
    }
    return 1.0;
}
@fragment fn horizon_material_fs_solid(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    if length(in.base) < horizon_displace_distance - 0.05 && horizon_displaced_layer(in.layer) {
        discard;
    }
    return horizon_material_solid(in);
}
fn horizon_material_solid(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    let color = horizon_material_color(in, false);
    let texel = horizon_terrain_sprite(in.uv, in.layer & 0x7fffu);
    // Bit 15 of the layer marks opaque leaves (forceOpaque): no alpha cutout.
    if color.a < 0.5 && (in.layer & 0x8000u) == 0u { discard; }
    if horizon_surface.w == 0.0 {
        set_surface(horizon_material_normal, 0.0, 0.0);
    }
    var out: HorizonMaterialSolid;
    let mark = horizon_material_mark(in, texel);
    out.color = horizon_material_fogged(vec4(color.rgb, mark), in.spherical, in.cylindrical);
    out.surface = vec4(horizon_surface.xyz, in.clip.z);
    out.albedo = vec4(horizon_material_albedo, 1.0);
    return out;
}
// The depth pre-pass (`NativeMaterial::prepass`): the solid terrain's depth alone, cut out by
// its texture's alpha, before the material shades only the nearest point of each pixel.
@fragment fn horizon_material_fs_depth(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    let alpha = horizon_terrain_sprite(in.uv, in.layer & 0x7fffu).a;
    if length(in.base) < horizon_displace_distance - 0.05 && horizon_displaced_layer(in.layer) {
        discard;
    }
    if alpha < 0.5 && (in.layer & 0x8000u) == 0u { discard; }
    return HorizonMaterialSolid(vec4(0.0), vec4(0.0), vec4(0.0));
}
@fragment fn horizon_material_fs_translucent(in: HorizonMaterialOut) -> @location(0) vec4<f32> {
    let color = horizon_material_color(in, true);
    if color.a < 0.01 { discard; }
    return horizon_material_fogged(color, in.spherical, in.cylindrical);
}
";

/// `[displacement]`'s grid (gfx/displace.rs): one instance a quad the pick chose, its corners
/// read from the copy of the arena, each of its cells two triangles moved along the face's
/// normal by the pack's `displacement`, measured from the texture's mean height so that the
/// face's edges, pinned there, meet their neighbours whatever their relief.
const DISPLACED_ENTRIES: &str = "
@vertex fn horizon_material_vs_displaced(@builtin(vertex_index) v: u32,
    @builtin(instance_index) instance: u32) -> HorizonMaterialOut {
    let p = horizon_displace_point(v, instance);
    var out = horizon_material_out(p.position, p.uv, p.layer, p.color, p.light);
    out.base = p.base;
    out.displaced = p.near;
    out.normal = p.normal;
    return out;
}
@fragment fn horizon_material_fs_displaced(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    if length(in.base) >= horizon_displace_distance + 0.05 {
        discard;
    }
    return horizon_material_solid(in);
}
// The grid's depth alone (`NativeMaterial::grid`), cut where `fs_displaced` cuts it: then the
// material shades only its nearest point a pixel, not each step's wall and cell behind it.
@fragment fn horizon_material_fs_grid_depth(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    if length(in.base) >= horizon_displace_distance + 0.05 {
        discard;
    }
    return HorizonMaterialSolid(vec4(0.0), vec4(0.0), vec4(0.0));
}
";

/// The format of the solid terrain's second target, what the post stage reads as
/// `scene_surface`: x the normal (two snorm16 in its bits), y smoothness, z the pack's data, w
/// depth.
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// The albedo target, for light the post stage adds.
pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// What terrain pipelines are built against: the renderer's layouts and constants.
pub struct TerrainShape<'a> {
    /// The terrain's group 0.
    pub globals: &'a wgpu::BindGroupLayout,
    /// The flat draws' group 1, where they run (`flat.rs`).
    pub flat: Option<&'a wgpu::BindGroupLayout>,
    /// terrain.wgsl's `SPRITE_ARRAYS` and `SPRITE_SPLIT`.
    pub sprites: [f64; 2],
    pub sprite_tags: &'a [(u32, u32)],
    /// The world target's colour format.
    pub format: wgpu::TextureFormat,
    /// The entities' group 0 (their atlas), for the shadow pass's entity draw.
    pub entities: &'a wgpu::BindGroupLayout,
    /// The weather's group 1 (its fall clock), for the pack's weather stage.
    pub weather_time: &'a wgpu::BindGroupLayout,
    /// The resource packs' LabPBR maps (normals, specular), when they ship any.
    pub pbr: Option<&'a [wgpu::Texture; 2]>,
}

/// The pack's terrain pipelines: solid (with the cutout test) and translucent, for the
/// renderer's instanced draws and, where they run, its flat ones.
pub struct NativeMaterial {
    pub solid: wgpu::RenderPipeline,
    pub translucent: wgpu::RenderPipeline,
    pub flat: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    /// `[displacement]`'s grid of the solid terrain near the camera (gfx/displace.rs).
    pub displaced: Option<wgpu::RenderPipeline>,
    /// The grid's depth, then its shading over it (as `prepass` for the flat terrain).
    pub grid: Option<(wgpu::RenderPipeline, wgpu::RenderPipeline)>,
    /// The solid terrain as a depth pre-pass and a material pass after it.
    pub prepass: Prepass,
}

/// The depth pre-pass of the solid terrain: `depth*` draw its depth alone (no colour), then the
/// others shade it without writing depth, so the material runs once a pixel rather than for
/// every surface drawn there before the nearest. Instanced and flat; `[displacement]`'s grid
/// draws with its depth after them.
pub struct Prepass {
    pub depth: wgpu::RenderPipeline,
    pub depth_flat: Option<wgpu::RenderPipeline>,
    pub solid: wgpu::RenderPipeline,
    pub flat: Option<wgpu::RenderPipeline>,
}

/// Which of the solid terrain's passes a pipeline draws (`Prepass`).
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    Shaded,
    Depth,
    After,
}

/// What the material's displaced pipeline is built against: `[displacement]`'s values and
/// its group 1.
pub struct DisplacedShape<'a> {
    pub distance: f32,
    pub cells: u32,
    pub steps: bool,
    pub layout: &'a wgpu::BindGroupLayout,
}

/// terrain.wgsl's terrain section (its globals, bindings, sampling, lighting and fog), with
/// every name it declares prefixed.
fn vanilla_terrain() -> String {
    prefixed(
        section(TERRAIN, "struct Globals {", "// Sky (SkyRenderer)"),
        PREFIX,
    )
}

/// Where a corner of terrain is drawn: the pack's vertex stage when it has one.
const DISPLACE_VANILLA: &str = "
fn horizon_material_displace(pos: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>) -> vec3<f32> {
    return pos;
}
";

/// The vertex stage's call, with its inputs from terrain's vertex.
pub(super) const DISPLACE_PACK: &str = "
fn horizon_material_displace(pos: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>) -> vec3<f32> {
    let material_id = material_ids[min(layer & 0x7fffu, arrayLength(&material_ids) - 1u)];
    let flags = u32(round(color.a * 255.0));
    let high = flags >> 4u;
    let fluid = high >= 8u && (high & 7u) < 6u;
    let lit = min(vec2(f32(light.y), f32(light.x)) / 240.0, vec2(1.0));
    return vertex(VertexInput(pos, uv, material_id, lit, vec4(color.rgb, 1.0),
        select(0u, high & 7u, fluid), u32(fluid)));
}
";

/// The material prelude's `ray_occluded`, where the device has ray queries.
pub(super) const RAYS: &str = "
@group(3) @binding(14) var horizon_scene: acceleration_structure;
@group(3) @binding(15) var<storage, read> horizon_ray_vertices: array<u32>;
struct HorizonRayLight {
    position: vec4<f32>,
    color: vec4<f32>,
}
@group(3) @binding(60) var<storage, read> horizon_ray_sources: array<HorizonRayLight>;
@group(3) @binding(61) var<storage, read> horizon_ray_light_cells: array<u32>;
@group(3) @binding(62) var<storage, read> horizon_ray_blue_tile: array<u32>;
fn horizon_ray_available() -> bool {
    return true;
}
// A candidate's texel: its section's vertices, as `TerrainVertex` (24 bytes, 6 words).
fn horizon_ray_texel(hit: RayIntersection, tint: ptr<function, vec3<f32>>) -> vec4<f32> {
    // Entities: another vertex format, drawn solid.
    if (hit.instance_custom_data & 0x7fffffu) == 0x7fffffu {
        return vec4<f32>(1.0);
    }
    let first = (hit.instance_custom_data & 0x7fffffu) + (hit.primitive_index / 2u) * 4u;
    let second = 1u + (hit.primitive_index & 1u);
    let a = first * 6u;
    let b = (first + second) * 6u;
    let c = (first + second + 1u) * 6u;
    let word = horizon_ray_vertices[a + 4u];
    let layer = word & 0xffffu;
    // Opaque quads need no texture.
    if (layer & 0x8000u) != 0u {
        return vec4<f32>(1.0);
    }
    let w = 1.0 - hit.barycentrics.x - hit.barycentrics.y;
    let uv = horizon_ray_uv_word(horizon_ray_vertices[a + 3u]) * w + horizon_ray_uv_word(horizon_ray_vertices[b + 3u]) * hit.barycentrics.x
        + horizon_ray_uv_word(horizon_ray_vertices[c + 3u]) * hit.barycentrics.y;
    let color = horizon_ray_vertices[a + 5u];
    *tint = vec3<f32>(f32(color & 255u), f32((color >> 8u) & 255u), f32((color >> 16u) & 255u)) / 255.0;
    return horizon_block_sample(uv, layer & 0x7fffu, 0.0);
}
fn horizon_ray_uv_word(word: u32) -> vec2<f32> {
    return vec2<f32>(f32(word & 0xffffu), f32(word >> 16u)) / 32768.0;
}
// Whether water lets the ray through untinted, its absorption being accounted for by length.
var<private> horizon_ray_clear_water = false;
// What a pane of translucent terrain leaves of light: a filter passes its own colour whole and
// holds back the rest. All the light crosses the pane, so half a texture alpha filters it whole.
fn horizon_ray_pane(texel: vec4<f32>, tint: vec3<f32>) -> vec3<f32> {
    let color = texel.rgb * tint;
    let filter_color = color / max(max(color.r, max(color.g, color.b)), 0.001);
    return mix(vec3<f32>(1.0), filter_color * filter_color, saturate(texel.a * 2.0));
}
// The light left of 1 after the ray: 0 past anything solid, tinted by translucent terrain.
fn horizon_ray_trace(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> vec3<f32> {
    var query: ray_query;
    rayQueryInitialize(&query, horizon_scene, RayDesc(0x04u, 0x03u, 0.0, distance, origin, direction));
    var light = vec3<f32>(1.0);
    var steps = 0u;
    while rayQueryProceed(&query) {
        steps += 1u;
        // Through this much foliage hardly anything gets by. Called lit, whole blocks would flick
        // on and off as the order of the candidates changes.
        if steps > 32u {
            return vec3<f32>(0.0);
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        var tint = vec3<f32>(1.0);
        let texel = horizon_ray_texel(hit, &tint);
        if (hit.instance_custom_data & 0x800000u) != 0u {
            if horizon_ray_is_water(hit) {
                if !horizon_ray_clear_water {
                    light *= mix(vec3<f32>(1.0), texel.rgb * tint, texel.a);
                }
                continue;
            }
            light *= horizon_ray_pane(texel, tint);
        } else if texel.a >= 0.5 {
            rayQueryConfirmIntersection(&query);
        }
    }
    if rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE {
        return vec3<f32>(0.0);
    }
    return light;
}
// A translucent quad the mesher marked as a fluid's (`fluid_face`).
fn horizon_ray_is_water(hit: RayIntersection) -> bool {
    if (hit.instance_custom_data & 0x7fffffu) == 0x7fffffu {
        return false;
    }
    let first = (hit.instance_custom_data & 0x7fffffu) + (hit.primitive_index / 2u) * 4u;
    let high = horizon_ray_vertices[first * 6u + 5u] >> 28u;
    return high >= 8u && (high & 7u) < 6u;
}
fn horizon_ray_tint(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> vec3<f32> {
    var query: ray_query;
    rayQueryInitialize(&query, horizon_scene, RayDesc(0x00u, 0x02u, 0.0, distance, origin, direction));
    var light = vec3<f32>(1.0);
    var steps = 0u;
    while rayQueryProceed(&query) {
        steps += 1u;
        if steps > 16u {
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        if (hit.instance_custom_data & 0x800000u) != 0u && !horizon_ray_is_water(hit) {
            var tint = vec3<f32>(1.0);
            let texel = horizon_ray_texel(hit, &tint);
            light *= horizon_ray_pane(texel, tint);
        }
    }
    return light;
}
// The height of the highest water the ray crosses, -1e9 if none. It asks for translucent terrain
// alone: a ray stopping at solid blocks may end before it has been shown the water in front.
fn horizon_ray_water_surface(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> f32 {
    var query: ray_query;
    rayQueryInitialize(&query, horizon_scene, RayDesc(0x00u, 0x02u, 0.0, distance, origin, direction));
    var surface = -1e9;
    var steps = 0u;
    while rayQueryProceed(&query) {
        steps += 1u;
        if steps > 16u {
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        if (hit.instance_custom_data & 0x800000u) != 0u && horizon_ray_is_water(hit) {
            surface = max(surface, origin.y + direction.y * hit.t);
        }
    }
    return surface;
}
// The water's surface at a world place: its slope and curvature, from a few travelling waves.
struct HorizonWave {
    slope: vec2<f32>,
    xx: f32,
    zz: f32,
    xz: f32,
}
fn horizon_ray_wave(world: vec2<f32>) -> HorizonWave {
    const WAVES = array<vec4<f32>, 5>(
        vec4<f32>(3.1, 1.3, 0.030, 0.9), vec4<f32>(-1.7, 3.3, 0.026, 1.1),
        vec4<f32>(4.3, -2.1, 0.016, 1.4), vec4<f32>(-3.7, -4.1, 0.012, 1.7),
        vec4<f32>(1.1, 5.3, 0.012, 1.9));
    var out = HorizonWave(vec2<f32>(0.0), 0.0, 0.0, 0.0);
    for (var i = 0; i < 5; i++) {
        let w = WAVES[i];
        let phase = dot(w.xy, world) + w.w * frame.time;
        out.slope += w.xy * w.z * cos(phase);
        let bend = -w.z * sin(phase);
        out.xx += w.x * w.x * bend;
        out.zz += w.y * w.y * bend;
        out.xz += w.x * w.y * bend;
    }
    return out;
}
// The light of a source reaching `origin` along `direction`, bending where it crosses water:
// Snell's law for the path over the flat surface, Fresnel's for what crosses it (nothing beyond
// the critical angle), absorption along the way in water, and the waves gathering the light
// into caustics: how much a patch of the receiver's surface is squeezed is how much brighter.
fn horizon_ray_seen(origin: vec3<f32>, direction: vec3<f32>, distance: f32, source_at: vec3<f32>, normal: vec3<f32>, salt: f32) -> vec3<f32> {
    // A point of the block, not its middle: a source that size gives soft edges past the water too.
    let target_at = source_at + (vec3<f32>(horizon_ray_noise(83.0 + salt), horizon_ray_noise(89.0 + salt), horizon_ray_noise(97.0 + salt)) - 0.5) * 0.9;
    let surface_y = horizon_ray_water_surface(origin, direction, distance);
    if surface_y < -1e8 || (origin.y - surface_y) * (target_at.y - surface_y) >= 0.0 {
        return horizon_ray_trace(origin, direction, distance);
    }
    let above = origin.y > surface_y;
    let near_h = abs(origin.y - surface_y);
    let far_h = abs(target_at.y - surface_y);
    let near_n = select(1.33, 1.0, above);
    let far_n = select(1.0, 1.33, above);
    let across = target_at.xz - origin.xz;
    let span = length(across);
    let side = across / max(span, 0.0001);
    var low = 0.0;
    var high = span;
    for (var i = 0; i < 10; i++) {
        let x = (low + high) * 0.5;
        let near_sin = x / max(length(vec2<f32>(x, near_h)), 0.0001);
        let far_sin = (span - x) / max(length(vec2<f32>(span - x, far_h)), 0.0001);
        if near_n * near_sin > far_n * far_sin {
            high = x;
        } else {
            low = x;
        }
    }
    // Where on the waves the light that lands here crossed, and how it was gathered.
    let flat_at = origin.xz + side * (low + high) * 0.5;
    let shift = near_h * select(0.25, 0.33, above);
    var at = flat_at;
    for (var i = 0; i < 3; i++) {
        at = flat_at - shift * horizon_ray_wave(at + frame.camera_position.xz).slope;
    }
    let wave = horizon_ray_wave(at + frame.camera_position.xz);
    let squeeze = (1.0 + shift * wave.xx) * (1.0 + shift * wave.zz) - shift * shift * wave.xz * wave.xz;
    let caustic = min(1.0 / max(abs(squeeze), 0.15), 6.0);

    let crossing = vec3<f32>(at.x, surface_y, at.y);
    let near_leg = crossing - origin;
    let near_length = max(length(near_leg), 0.0001);
    // To the block's middle: past the waves a random point of it scatters the caustic into grain.
    let far_leg = source_at - crossing;
    let far_length = max(length(far_leg), 0.0001);
    let far_dir = far_leg / far_length;
    // Up to where the ray enters the source's block.
    let safe = select(far_dir, vec3<f32>(0.001), abs(far_dir) < vec3<f32>(0.001));
    let box_a = (source_at - 0.52 - crossing) / safe;
    let box_b = (source_at + 0.52 - crossing) / safe;
    let box_near = min(box_a, box_b);
    let enters = max(box_near.x, max(box_near.y, box_near.z));
    horizon_ray_clear_water = true;
    let seen_near = horizon_ray_trace(origin, near_leg / near_length, max(near_length - 0.001, 0.001));
    let seen_far = horizon_ray_trace(crossing, far_dir, max(min(enters, far_length), 0.001));
    horizon_ray_clear_water = false;

    let near_cos = near_h / near_length;
    let far_cos = far_h / far_length;
    let water_cos = select(near_cos, far_cos, above);
    let air_cos = select(far_cos, near_cos, above);
    let s = (1.33 * water_cos - air_cos) / (1.33 * water_cos + air_cos);
    let r = (1.33 * air_cos - water_cos) / (1.33 * air_cos + water_cos);
    let through = 1.0 - 0.5 * (s * s + r * r);
    let water_length = select(near_length, far_length, above);
    let absorbed = exp(-vec3<f32>(0.35, 0.07, 0.03) * water_length);
    // The surface is lit from where the light now comes from.
    let turned = min(max(dot(normal, near_leg / near_length), 0.0) / max(dot(normal, direction), 0.05), 3.0);
    return seen_near * seen_far * through * absorbed * caustic * turned;
}
fn horizon_ray_position(word: u32) -> vec3<f32> {
    return vec3<f32>(bitcast<f32>(horizon_ray_vertices[word]),
        bitcast<f32>(horizon_ray_vertices[word + 1u]), bitcast<f32>(horizon_ray_vertices[word + 2u]));
}
fn horizon_ray_hit(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> RayHit {
    var query: ray_query;
    rayQueryInitialize(&query, horizon_scene, RayDesc(0x00u, 0x01u, 0.0, distance, origin, direction));
    var steps = 0u;
    while rayQueryProceed(&query) {
        steps += 1u;
        if steps > 32u {
            break;
        }
        let candidate = rayQueryGetCandidateIntersection(&query);
        var tint = vec3<f32>(1.0);
        if horizon_ray_texel(candidate, &tint).a >= 0.5 {
            rayQueryConfirmIntersection(&query);
        }
    }
    let hit = rayQueryGetCommittedIntersection(&query);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE {
        return RayHit(false, distance, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, true);
    }
    if (hit.instance_custom_data & 0x7fffffu) == 0x7fffffu {
        return RayHit(true, hit.t, vec3<f32>(0.4), -direction, 0.5, 0.0, true);
    }
    let first = (hit.instance_custom_data & 0x7fffffu) + (hit.primitive_index / 2u) * 4u;
    let second = 1u + (hit.primitive_index & 1u);
    let a = first * 6u;
    let b = (first + second) * 6u;
    let c = (first + second + 1u) * 6u;
    var normal = cross(horizon_ray_position(b) - horizon_ray_position(a),
        horizon_ray_position(c) - horizon_ray_position(a));
    let size = length(normal);
    normal = select(-direction, normal / size, size > 1e-6);
    // The quads wind outward, so a normal along the ray means the inner side.
    let front = dot(normal, direction) <= 0.0;
    if !front {
        normal = -normal;
    }
    let word = horizon_ray_vertices[a + 4u];
    let w = 1.0 - hit.barycentrics.x - hit.barycentrics.y;
    let uv = horizon_ray_uv_word(horizon_ray_vertices[a + 3u]) * w
        + horizon_ray_uv_word(horizon_ray_vertices[b + 3u]) * hit.barycentrics.x
        + horizon_ray_uv_word(horizon_ray_vertices[c + 3u]) * hit.barycentrics.y;
    let color = horizon_ray_vertices[a + 5u];
    let tint = vec3<f32>(f32(color & 255u), f32((color >> 8u) & 255u), f32((color >> 16u) & 255u)) / 255.0;
    // A low mip: the bounce wants the block's colour, without its grain.
    let albedo = horizon_block_sample(uv, word & 0x7fffu, 3.0).rgb * tint;
    return RayHit(true, hit.t, albedo, normal,
        f32((word >> 16u) & 255u) / 240.0, f32((word >> 24u) & 255u) / 240.0, front);
}
fn horizon_ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return all(horizon_ray_trace(origin, direction, distance) == vec3<f32>(0.0));
}
// A new number each frame for the pack's temporal filter to average, independent for each salt:
// taken from one noise with an offset, a ray's angle and radius follow each other, the directions
// lie along a curve and a soft shadow comes out as hard edges and arcs.
fn horizon_ray_noise(salt: f32) -> f32 {
    let pixel = vec2<u32>(horizon_ray_pixel);
    let seed = pixel.x + pixel.y * 4099u + u32(salt) * 65537u + (frame.frame_index % 4096u) * 16777259u;
    let state = seed * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32((word >> 22u) ^ word) / 4294967296.0;
}
// Two numbers in 0..1 from the blue noise tile at this pixel: `salt` moves to another place of
// the tile for each ray of a pixel, and each frame turns both by the R2 sequence, so over frames
// every pixel sees the whole range while neighbours stay apart.
fn horizon_ray_blue(salt: f32) -> vec2<f32> {
    let s = u32(salt);
    let at = (vec2<u32>(horizon_ray_pixel) + vec2<u32>(s * 23u, s * 41u)) % 64u;
    let word = horizon_ray_blue_tile[at.y * 64u + at.x];
    let base = vec2<f32>(f32(word & 0xffffu), f32(word >> 16u)) / 65536.0;
    return fract(base + (f32(frame.frame_index % 4096u) + salt * 0.61803399 * 7.0) * vec2<f32>(0.7548776662, 0.5698402910));
}
// A direction within `radius` (the tangent of its half angle) of `direction`.
fn horizon_ray_jitter(direction: vec3<f32>, radius: f32, salt: f32) -> vec3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.9);
    let x = normalize(cross(up, direction));
    let y = cross(direction, x);
    // The angle evenly spread between neighbouring pixels, for a filter to average in few
    // pixels; the radius apart from it, or the directions lie along a curve.
    let moved = horizon_ray_pixel + 5.588238 * f32(frame.frame_index % 64u);
    let even = fract(52.9829189 * fract(dot(moved, vec2<f32>(0.06711056, 0.00583715))) + salt * 0.61803399);
    let angle = even * 6.2831853;
    // White, not the blue tile: with blue noise here a room of lamps flickered a tenth more,
    // their soft shadows' rays turning together from frame to frame.
    let r = sqrt(horizon_ray_noise(salt + 17.0)) * radius;
    return normalize(direction + (x * cos(angle) + y * sin(angle)) * r);
}
fn horizon_ray_light(origin: vec3<f32>, direction: vec3<f32>, distance: f32, radius: f32) -> vec3<f32> {
    return horizon_ray_trace(origin, horizon_ray_jitter(direction, radius, 0.0), distance);
}
fn horizon_ray_lights(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return horizon_ray_lights_budget(position, normal, 4u);
}
// Which word of the source grid starts the list for this place, past the end if none. Not
// dithered: a pack looking up a source it kept must find it, from every pixel of the place.
// The list of the cell at `position`, its edge dithered by up to 3 blocks (the same for a
// pixel's calls in a frame): where a crowd of sources overflows the lists, neighbouring cells
// keep different ones, and read at the exact edge their light changed there in a straight line.
fn horizon_ray_light_cell(position: vec3<f32>) -> u32 {
    let dither = (vec3<f32>(horizon_ray_noise(31.0), horizon_ray_noise(47.0), horizon_ray_noise(59.0)) - 0.5) * 6.0;
    let cell = vec3<i32>(floor((position + dither - horizon_ray_sources[0].position.yzw) / 8.0));
    if any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(20)) {
        return 0xffffffffu;
    }
    return u32((cell.z * 20 + cell.y) * 20 + cell.x) * 32u;
}
fn horizon_ray_light_count(position: vec3<f32>) -> u32 {
    let first = horizon_ray_light_cell(position);
    if first == 0xffffffffu {
        return 0u;
    }
    return min(horizon_ray_light_cells[first] & 0xffffu, 63u);
}
fn horizon_ray_light_id(position: vec3<f32>, index: u32) -> u32 {
    let n = index + 1u;
    let word = horizon_ray_light_cells[horizon_ray_light_cell(position) + n / 2u];
    return (word >> ((n % 2u) * 16u)) & 0xffffu;
}
fn horizon_ray_light_strength(id: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let source = horizon_ray_sources[id];
    let to = source.position.xyz - position;
    let d2 = dot(to, to);
    let reach = source.position.w;
    if d2 >= reach * reach {
        return 0.0;
    }
    let facing = dot(normal, select(to / sqrt(max(d2, 0.0001)), normal, d2 < 0.0001));
    if facing <= 0.0 {
        return 0.0;
    }
    let edge = saturate(1.0 - pow(d2 / (reach * reach), 2.0));
    return reach * reach / 225.0 * edge * edge / (1.0 + d2 * 0.07) * facing;
}
fn horizon_ray_light_color(id: u32) -> vec3<f32> {
    return horizon_ray_sources[id].color.rgb;
}
fn horizon_ray_light_position(id: u32) -> vec3<f32> {
    return horizon_ray_sources[id].position.xyz;
}
fn horizon_ray_light_seen(id: u32, position: vec3<f32>, normal: vec3<f32>, salt: f32) -> vec3<f32> {
    let center = horizon_ray_sources[id].position.xyz;
    let to = center - position;
    let d = max(length(to), 0.0001);
    let size = horizon_ray_sources[id].color.w;
    let dir = horizon_ray_jitter(to / d, size / max(d, size), salt);
    let inside = 1.04 * size * min(min(1.0 / max(abs(dir.x), 0.001), 1.0 / max(abs(dir.y), 0.001)), 1.0 / max(abs(dir.z), 0.001));
    return horizon_ray_seen(position + normal * 0.03, dir, max(d - inside, 0.01), center, normal, salt);
}
fn horizon_ray_lights_budget(position: vec3<f32>, normal: vec3<f32>, budget: u32) -> vec3<f32> {
    // The sources reaching this place. The cell's edge is dithered, so two cells' lists blend.
    let first = horizon_ray_light_cell(position);
    if first == 0xffffffffu {
        return vec3<f32>(0.0);
    }
    let count = min(horizon_ray_light_cells[first] & 0xffffu, 63u);
    // How strong each is here, not at the cell's middle: the budget goes to the strongest at
    // this very point, so what gets a shadow doesn't jump along a surface.
    var strengths: array<f32, 16>;
    var ids: array<u32, 16>;
    var towards: array<vec4<f32>, 16>;
    var found = 0u;
    for (var n = 1u; n <= count; n++) {
        let word = horizon_ray_light_cells[first + n / 2u];
        let id = (word >> ((n % 2u) * 16u)) & 0xffffu;
        let source = horizon_ray_sources[id];
        let to = source.position.xyz - position;
        let d2 = dot(to, to);
        let reach = source.position.w;
        if d2 >= reach * reach {
            continue;
        }
        let d = sqrt(max(d2, 0.0001));
        let toward = select(to / d, normal, d2 < 0.0001);
        let facing = dot(normal, toward);
        if facing <= 0.0 {
            continue;
        }
        // The inverse square at a sphere's size, brought to 0 at the block light's reach.
        let edge = saturate(1.0 - pow(d2 / (reach * reach), 2.0));
        let intensity = reach * reach / 225.0 * edge * edge / (1.0 + d2 * 0.07);
        let strength = intensity * facing;
        if strength < 0.002 {
            continue;
        }
        // The list is strongest first: past 16 the rest add little.
        if found >= 16u {
            break;
        }
        strengths[found] = strength;
        ids[found] = id;
        towards[found] = vec4<f32>(toward, d);
        found += 1u;
    }
    // With more sources than rays, each ray picks one by chance, the stronger the likelier, and
    // counts for what the rest would give. Always tracing the strongest few makes a hard edge
    // wherever that set changes; this makes noise, for the temporal filter to average.
    var total = 0.0;
    for (var i = 0u; i < found; i++) {
        total += strengths[i];
    }
    var sum = vec3<f32>(0.0);
    for (var ray = 0u; ray < min(budget, found); ray++) {
        var pick = ray;
        var weight = 1.0;
        if found > budget {
            var left = horizon_ray_noise(71.0 + f32(ray)) * total;
            for (var i = 0u; i < found; i++) {
                pick = i;
                if left <= strengths[i] {
                    break;
                }
                left -= strengths[i];
            }
            weight = total / (strengths[pick] * f32(budget));
        }
        let source = horizon_ray_sources[ids[pick]];
        let d = towards[pick].w;
        let dir = horizon_ray_jitter(towards[pick].xyz, source.color.w / max(d, source.color.w), f32(ray));
        // The ray ends where it would enter the source's block: a full block doesn't stand in
        // the way of its own light.
        let inside = 1.04 * source.color.w * min(min(1.0 / max(abs(dir.x), 0.001), 1.0 / max(abs(dir.y), 0.001)), 1.0 / max(abs(dir.z), 0.001));
        let seen = horizon_ray_seen(position + normal * 0.03, dir, max(d - inside, 0.01), source.position.xyz, normal, f32(ray));
        sum += source.color.rgb * seen * strengths[pick] * weight;
    }
    // However many pile up the light saturates rather than climbing.
    return sum / (1.0 + max(sum.r, max(sum.g, sum.b)) * 0.5);
}
";
const NO_RAYS: &str = "
fn horizon_ray_available() -> bool {
    return false;
}
fn horizon_ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return false;
}
fn horizon_ray_light(origin: vec3<f32>, direction: vec3<f32>, distance: f32, radius: f32) -> vec3<f32> {
    return vec3<f32>(1.0);
}
fn horizon_ray_lights(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_lights_budget(position: vec3<f32>, normal: vec3<f32>, budget: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_tint(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> vec3<f32> {
    return vec3<f32>(1.0);
}
fn horizon_ray_light_count(position: vec3<f32>) -> u32 {
    return 0u;
}
fn horizon_ray_light_id(position: vec3<f32>, index: u32) -> u32 {
    return 0u;
}
fn horizon_ray_light_strength(id: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    return 0.0;
}
fn horizon_ray_light_color(id: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_light_position(id: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_light_seen(id: u32, position: vec3<f32>, normal: vec3<f32>, salt: f32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_blue(salt: f32) -> vec2<f32> {
    return vec2<f32>(0.5);
}
fn horizon_ray_hit(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> RayHit {
    return RayHit(false, distance, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, true);
}
";

/// The terrain shader for the pack's material stage `source` (with its vertex stage's
/// file, when `vertex`, and `[displacement]`'s grid when `displaced`).
pub(super) fn layer_table(name: &str, table: &[(u32, u32)]) -> String {
    let mut out = format!("fn {name}(layer: u32) -> u32 {{\n    switch layer {{\n");
    let mut table = table.to_vec();
    table.sort_unstable_by_key(|&(layer, value)| (value, layer));
    for group in table.chunk_by(|a, b| a.1 == b.1) {
        let layers: Vec<String> = group.iter().map(|(layer, _)| format!("{layer}u")).collect();
        out += &format!(
            "        case {}: {{ return {}u; }}\n",
            layers.join(", "),
            group[0].1
        );
    }
    out + "        default: { return 0u; }\n    }\n}\n"
}

fn shader_source(source: &str, vertex: bool, displaced: bool, rays: bool) -> String {
    format!(
        "{}{source}\n// Horizon's terrain (terrain.wgsl)\n{}{}{ENTRIES}{}{}{}",
        if rays { "enable wgpu_ray_query;\n" } else { "" },
        vanilla_terrain(),
        if vertex {
            DISPLACE_PACK
        } else {
            DISPLACE_VANILLA
        },
        super::displace::RELIEF,
        if displaced {
            super::displace::GRID.to_owned() + DISPLACED_ENTRIES
        } else {
            String::new()
        },
        (if rays { RAYS } else { NO_RAYS }).to_owned() + &layer_table("horizon_sprite_tags", &[]),
    )
}

/// The material's naga module, its errors in the stage's file.
fn module(
    stage: &CheckedStage,
    vertex: bool,
    displaced: bool,
    rays: bool,
) -> Result<naga::Module, String> {
    let text = shader_source(&stage.source, vertex, displaced, rays);
    naga::front::wgsl::parse_str(&text)
        .map_err(|e| format!("{}: {}", stage.file, e.emit_to_string(&text)))
}

impl NativeMaterial {
    /// The pipelines of the pack's material `stage` with the option values `constants`.
    pub fn new(
        device: &wgpu::Device,
        shape: &TerrainShape<'_>,
        frame: &NativeFrame,
        stage: &CheckedStage,
        constants: &[(String, f64)],
        fog: bool,
        vertex: bool,
        displaced: Option<&DisplacedShape<'_>>,
    ) -> Result<Self, String> {
        let rays = frame.rays.is_some();
        let text = shader_source(&stage.source, vertex, displaced.is_some(), rays).replace(
            &layer_table("horizon_sprite_tags", &[]),
            &layer_table("horizon_sprite_tags", shape.sprite_tags),
        );
        let module = naga::front::wgsl::parse_str(&text)
            .map_err(|e| format!("{}: {}", stage.file, e.emit_to_string(&text)))?;
        let mut constants = present_constants(&module, constants);
        let sprite_names = [
            "horizon_terrain_SPRITE_ARRAYS",
            "horizon_terrain_SPRITE_SPLIT",
        ];
        constants.extend(sprite_names.into_iter().zip(shape.sprites));
        constants.push(("horizon_material_fog", f64::from(u8::from(fog))));
        constants.push((
            "horizon_displace_distance",
            displaced.map_or(0.0, |d| f64::from(d.distance)),
        ));
        if let Some(d) = displaced {
            constants.push(("horizon_displace_steps", f64::from(u8::from(d.steps))));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&stage.label),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let layout = |flat: Option<&wgpu::BindGroupLayout>| {
            let mut groups: [Option<&wgpu::BindGroupLayout>; GROUP as usize + 1] =
                Default::default();
            groups[0] = Some(shape.globals);
            groups[1] = flat;
            groups[GROUP as usize] = Some(&frame.layout);
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Native shader pack terrain"),
                bind_group_layouts: &groups,
                ..Default::default()
            })
        };
        let instance_attributes = wgpu::vertex_attr_array![5=>Float32x4];
        let vertices = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<super::TerrainVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &super::TERRAIN_ATTRIBUTES,
        };
        let instances = wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &instance_attributes,
        };
        // As the renderer's terrain pipelines (gfx.rs): back faces culled, reversed depth.
        // `pass`: Shaded (depth written), Depth (depth alone), After (shaded over a pre-pass).
        let pipeline_in = |layout: &wgpu::PipelineLayout,
                           vertex: &str,
                           buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                           fragment: &str,
                           blend: Option<wgpu::BlendState>,
                           pass: Pass| {
            let write_mask = if pass == Pass::Depth {
                wgpu::ColorWrites::empty()
            } else {
                wgpu::ColorWrites::ALL
            };
            // Solid terrain also writes the surface target (`SURFACE_FORMAT`).
            let color = Some(wgpu::ColorTargetState {
                format: shape.format,
                blend,
                write_mask,
            });
            let surface = Some(wgpu::ColorTargetState {
                format: SURFACE_FORMAT,
                blend: None,
                write_mask,
            });
            let albedo = Some(wgpu::ColorTargetState {
                format: ALBEDO_FORMAT,
                blend: None,
                write_mask,
            });
            let solid = [color.clone(), surface, albedo];
            let translucent = [color];
            let targets: &[Option<wgpu::ColorTargetState>] = if blend.is_none() {
                &solid
            } else {
                &translucent
            };
            let options = wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&stage.label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vertex),
                    compilation_options: options.clone(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(pass != Pass::After),
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fragment),
                    compilation_options: options,
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let instanced = layout(None);
        let buffers = [Some(vertices.clone()), Some(instances)];
        let flat_layout = shape.flat.map(|group| layout(Some(group)));
        let displaced_layout = displaced.map(|d| layout(Some(d.layout)));
        let flat_vertices = [Some(vertices.clone())];
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("{}: {e}", stage.file));
        }
        // Each pipeline compiles the whole material for its entry points: the driver takes
        // seconds over one tracing rays the first time it sees an option's value, so they are
        // built on threads at once (wgpu's error scopes are each thread's own).
        let checked = |make: &(dyn Fn() -> wgpu::RenderPipeline + Sync)| {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = make();
            match pollster::block_on(scope.pop()) {
                Some(e) => Err(format!("{}: {e}", stage.file)),
                None => Ok(pipeline),
            }
        };
        let instanced_in = |fragment: &str, blend, pass| {
            pipeline_in(
                &instanced,
                "horizon_material_vs",
                &buffers,
                fragment,
                blend,
                pass,
            )
        };
        let flat_in = |fragment: &str, blend, pass| {
            flat_layout.as_ref().map(|layout| {
                pipeline_in(
                    layout,
                    "horizon_material_vs_flat",
                    &flat_vertices,
                    fragment,
                    blend,
                    pass,
                )
            })
        };
        let grid_in = |fragment: &str, pass| {
            displaced_layout.as_ref().map(|layout| {
                pipeline_in(
                    layout,
                    "horizon_material_vs_displaced",
                    &[],
                    fragment,
                    None,
                    pass,
                )
            })
        };
        type Built = Result<wgpu::RenderPipeline, String>;
        type Maybe = Result<Option<wgpu::RenderPipeline>, String>;
        let one = |f: &(dyn Fn() -> wgpu::RenderPipeline + Sync)| -> Built { checked(f) };
        let maybe = |f: &(dyn Fn() -> Option<wgpu::RenderPipeline> + Sync)| -> Maybe {
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pipeline = f();
            match pollster::block_on(scope.pop()) {
                Some(e) => Err(format!("{}: {e}", stage.file)),
                None => Ok(pipeline),
            }
        };
        let (
            solid,
            translucent,
            flat_solid,
            flat_translucent,
            displaced,
            grid_depth,
            grid_after,
            depth,
            depth_flat,
            after,
            after_flat,
        ) = std::thread::scope(|s| {
            let solid =
                s.spawn(|| one(&|| instanced_in("horizon_material_fs_solid", None, Pass::Shaded)));
            let translucent = s.spawn(|| {
                one(&|| instanced_in("horizon_material_fs_translucent", blend, Pass::Shaded))
            });
            let flat_solid =
                s.spawn(|| maybe(&|| flat_in("horizon_material_fs_solid", None, Pass::Shaded)));
            let flat_translucent = s.spawn(|| {
                maybe(&|| flat_in("horizon_material_fs_translucent", blend, Pass::Shaded))
            });
            let displaced =
                s.spawn(|| maybe(&|| grid_in("horizon_material_fs_displaced", Pass::Shaded)));
            let grid_depth =
                s.spawn(|| maybe(&|| grid_in("horizon_material_fs_grid_depth", Pass::Depth)));
            let grid_after =
                s.spawn(|| maybe(&|| grid_in("horizon_material_fs_displaced", Pass::After)));
            let depth =
                s.spawn(|| one(&|| instanced_in("horizon_material_fs_depth", None, Pass::Depth)));
            let depth_flat =
                s.spawn(|| maybe(&|| flat_in("horizon_material_fs_depth", None, Pass::Depth)));
            let after =
                s.spawn(|| one(&|| instanced_in("horizon_material_fs_solid", None, Pass::After)));
            let after_flat =
                s.spawn(|| maybe(&|| flat_in("horizon_material_fs_solid", None, Pass::After)));
            fn stopped<T>() -> Result<T, String> {
                Err("a material pipeline's build stopped".to_owned())
            }
            (
                solid.join().unwrap_or_else(|_| stopped()),
                translucent.join().unwrap_or_else(|_| stopped()),
                flat_solid.join().unwrap_or_else(|_| stopped()),
                flat_translucent.join().unwrap_or_else(|_| stopped()),
                displaced.join().unwrap_or_else(|_| stopped()),
                grid_depth.join().unwrap_or_else(|_| stopped()),
                grid_after.join().unwrap_or_else(|_| stopped()),
                depth.join().unwrap_or_else(|_| stopped()),
                depth_flat.join().unwrap_or_else(|_| stopped()),
                after.join().unwrap_or_else(|_| stopped()),
                after_flat.join().unwrap_or_else(|_| stopped()),
            )
        });
        let flat = match (flat_solid?, flat_translucent?) {
            (Some(solid), Some(translucent)) => Some((solid, translucent)),
            _ => None,
        };
        let (solid, translucent, displaced) = (solid?, translucent?, displaced?);
        let grid = match (grid_depth?, grid_after?) {
            (Some(depth), Some(after)) => Some((depth, after)),
            _ => None,
        };
        let prepass = Prepass {
            depth: depth?,
            depth_flat: depth_flat?,
            solid: after?,
            flat: after_flat?,
        };
        Ok(Self {
            solid,
            translucent,
            flat,
            displaced,
            grid,
            prepass,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizon_shader_api::{MANIFEST, Pack, StageKind, check};

    fn stage(material: &str) -> CheckedStage {
        let manifest = "schema = 1\nid = \"t\"\nname = \"T\"\nversion = \"1\"\napi = \"0.5\"\n\
            [material]\nfile = \"material.wgsl\"\n\
            [[options]]\nid = \"gain\"\nname = \"Gain\"\ntype = \"float\"\n\
            default = 1.0\nmin = 0.0\nmax = 2.0\n";
        let files = [(MANIFEST, manifest), ("material.wgsl", material)]
            .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
            .into();
        let mut report = check(&Pack::from_files(files).unwrap()).unwrap();
        assert_eq!(report.stages[0].kind, StageKind::Material);
        report.stages.remove(0)
    }

    #[test]
    fn the_terrain_takes_a_prefix_and_validates_with_a_pack() {
        let terrain = vanilla_terrain();
        for name in [
            "Globals",
            "globals",
            "sprites_7",
            "sprite",
            "SPRITE_SPLIT",
            "block_slots",
            "sample_lightmap",
            "apply_fog",
        ] {
            assert!(terrain.contains(&format!("{PREFIX}{name}")), "{name}");
        }
        assert!(!terrain.contains("fs_sky"), "the sky stays out");
        // A pack may use any name Horizon's terrain uses unprefixed.
        let own = "override gain: f32;\nconst PI: f32 = 3.0;\n\
            fn sprite(c: vec4<f32>) -> vec4<f32> { return c * PI; }\n\
            fn apply_fog(c: vec4<f32>) -> vec4<f32> { return c * gain; }\n\
            fn material(input: MaterialInput) -> Surface {\n    \
                let lit = input.albedo * input.tint * vec4(input.lightmap, 1.0);\n    \
                let t = vec2<i32>(input.screen * frame.resolution);\n    \
                let behind = textureLoad(opaque_color, t, 0) * textureLoad(opaque_depth, t, 0);\n    \
                let mirrored = textureSampleLevel(reflection, reflection_sampler, input.screen, 0.0);\n    \
                let kind = f32(input.face + input.fluid) * input.emission * mirrored.a;\n    \
                let frame_of = vec4(cross(input.normal, input.tangent.xyz) * input.tangent.w, kind);\n    \
                return Surface(apply_fog(sprite(lit)) + (behind + frame_of) * 0.0);\n}\n";
        let stage = stage(own);
        let module = module(&stage, false, false, false).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| {
            panic!(
                "{}",
                e.emit_to_string(&shader_source(&stage.source, false, false, false))
            )
        });
        let entries: Vec<&str> = module
            .entry_points
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        for entry in [
            "horizon_material_vs",
            "horizon_material_vs_flat",
            "horizon_material_fs_solid",
            "horizon_material_fs_translucent",
        ] {
            assert!(entries.contains(&entry), "{entry}");
        }
        drop(info);
        assert_eq!(
            present_constants(&module, &[("gain".into(), 1.5), ("other".into(), 1.0)]),
            [("gain", 1.5)]
        );
    }

    #[test]
    fn the_displaced_grid_validates_with_a_pack() {
        let manifest = "schema = 1\nid = \"t\"\nname = \"T\"\nversion = \"1\"\napi = \"0.5\"\n\
            [material]\nfile = \"material.wgsl\"\n[displacement]\ndistance = 8.0\n";
        let material = "fn displacement(input: DisplacementInput) -> f32 {\n    \
                return (1.0 - input.height) * 0.25;\n}\n\
            fn material(input: MaterialInput) -> Surface {\n    \
                return Surface(input.albedo * (1.0 - input.displaced));\n}\n";
        let files = [(MANIFEST, manifest), ("material.wgsl", material)]
            .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
            .into();
        let mut report = check(&Pack::from_files(files).unwrap()).unwrap();
        let stage = report.stages.remove(0);
        let module = module(&stage, false, true, false).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| {
            panic!(
                "{}",
                e.emit_to_string(&shader_source(&stage.source, false, true, false))
            )
        });
        assert!(
            module
                .entry_points
                .iter()
                .any(|e| e.name == "horizon_material_vs_displaced")
        );
    }
}
