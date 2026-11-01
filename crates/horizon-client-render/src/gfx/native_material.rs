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
    // The face's HDR tier (mesher.rs's `emitting`: 0 none, 1 low, 2 incandescent, 3 bright,
    // 4 very bright) as how strongly it shines: a redstone torch 7/15, a torch 14/15.
    let tier = 15u - (flags & 15u);
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
    return material(MaterialInput(albedo, in.tint, light, in.lightmap, in.position, in.uv,
        translucent, in.face, fluid, screen, material_id, in.emission, normal, tangent,
        in.displaced)).color;
}
var<private> horizon_material_normal: vec3<f32>;
struct HorizonMaterialSolid {
    @location(0) color: vec4<f32>,
    // The world target's surface data: `set_surface`'s, then the fragment's depth, which the
    // post stage's `surface_at` compares with the final depth.
    @location(1) surface: vec4<f32>,
}
@fragment fn horizon_material_fs_solid(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    if length(in.base) < horizon_displace_distance - 0.05 && horizon_displaced_layer(in.layer) {
        discard;
    }
    return horizon_material_solid(in);
}
fn horizon_material_solid(in: HorizonMaterialOut) -> HorizonMaterialSolid {
    let color = horizon_material_color(in, false);
    // Bit 15 of the layer marks opaque leaves (forceOpaque): no alpha cutout.
    if color.a < 0.5 && (in.layer & 0x8000u) == 0u { discard; }
    if horizon_surface.w == 0.0 {
        set_surface(horizon_material_normal, 0.0, 0.0);
    }
    var out: HorizonMaterialSolid;
    out.color = horizon_material_fogged(vec4(color.rgb, 1.0), in.spherical, in.cylindrical);
    out.surface = vec4(horizon_surface.xyz, in.clip.z);
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
    return HorizonMaterialSolid(vec4(0.0), vec4(0.0));
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
";

/// The format of the solid terrain's second target, what the post stage reads as
/// `scene_surface`: x the normal (two snorm16 in its bits), y smoothness, z the pack's data, w
/// depth.
pub const SURFACE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// What terrain pipelines are built against: the renderer's layouts and constants.
pub struct TerrainShape<'a> {
    /// The terrain's group 0.
    pub globals: &'a wgpu::BindGroupLayout,
    /// The flat draws' group 1, where they run (`flat.rs`).
    pub flat: Option<&'a wgpu::BindGroupLayout>,
    /// terrain.wgsl's `SPRITE_ARRAYS` and `SPRITE_SPLIT`.
    pub sprites: [f64; 2],
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
@group(3) @binding(40) var<storage, read> horizon_ray_sources: array<HorizonRayLight>;
@group(3) @binding(41) var<storage, read> horizon_ray_light_cells: array<u32>;
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
// The light left of 1 after the ray: 0 past anything solid, tinted by translucent terrain.
fn horizon_ray_trace(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> vec3<f32> {
    var query: ray_query;
    rayQueryInitialize(&query, horizon_scene, RayDesc(0x04u, 0x03u, 0.0, distance, origin, direction));
    var light = vec3<f32>(1.0);
    var steps = 0u;
    while rayQueryProceed(&query) {
        steps += 1u;
        if steps > 32u {
            break;
        }
        let hit = rayQueryGetCandidateIntersection(&query);
        var tint = vec3<f32>(1.0);
        let texel = horizon_ray_texel(hit, &tint);
        if (hit.instance_custom_data & 0x800000u) != 0u {
            light *= mix(vec3<f32>(1.0), texel.rgb * tint, texel.a);
        } else if texel.a >= 0.5 {
            rayQueryConfirmIntersection(&query);
        }
    }
    if rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE {
        return vec3<f32>(0.0);
    }
    return light;
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
        return RayHit(false, distance, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
    }
    if (hit.instance_custom_data & 0x7fffffu) == 0x7fffffu {
        return RayHit(true, hit.t, vec3<f32>(0.4), -direction, 0.5, 0.0);
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
    if dot(normal, direction) > 0.0 {
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
        f32((word >> 16u) & 255u) / 240.0, f32((word >> 24u) & 255u) / 240.0);
}
fn horizon_ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return all(horizon_ray_trace(origin, direction, distance) == vec3<f32>(0.0));
}
// Interleaved gradient noise, moving each frame for the pack's temporal filter to average.
fn horizon_ray_noise(salt: f32) -> f32 {
    let p = horizon_ray_pixel + vec2<f32>(5.588238 * (f32(frame.frame_index % 64u) + salt));
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}
// A direction within `radius` (the tangent of its half angle) of `direction`.
fn horizon_ray_jitter(direction: vec3<f32>, radius: f32, salt: f32) -> vec3<f32> {
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(direction.y) > 0.9);
    let x = normalize(cross(up, direction));
    let y = cross(direction, x);
    let angle = horizon_ray_noise(salt) * 6.2831853;
    let r = sqrt(horizon_ray_noise(salt + 17.0)) * radius;
    return normalize(direction + (x * cos(angle) + y * sin(angle)) * r);
}
fn horizon_ray_light(origin: vec3<f32>, direction: vec3<f32>, distance: f32, radius: f32) -> vec3<f32> {
    return horizon_ray_trace(origin, horizon_ray_jitter(direction, radius, 0.0), distance);
}
fn horizon_ray_lights(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    // The sources reaching this place, strongest first.
    let cell = vec3<i32>(floor((position - horizon_ray_sources[0].position.yzw) / 8.0));
    if any(cell < vec3<i32>(0)) || any(cell >= vec3<i32>(20)) {
        return vec3<f32>(0.0);
    }
    let first = u32((cell.z * 20 + cell.y) * 20 + cell.x) * 8u;
    let count = horizon_ray_light_cells[first] & 0xffffu;
    var traced = vec3<f32>(0.0);
    var untraced = vec3<f32>(0.0);
    var seen_part = 0.0;
    var whole_part = 0.0;
    var rays = 0u;
    for (var n = 1u; n <= count; n++) {
        let word = horizon_ray_light_cells[first + n / 2u];
        let source = horizon_ray_sources[(word >> ((n % 2u) * 16u)) & 0xffffu];
        let to = source.position.xyz - position;
        let d2 = dot(to, to);
        let reach = source.position.w;
        if d2 >= reach * reach || d2 < 0.0001 {
            continue;
        }
        let d = sqrt(d2);
        let toward = to / d;
        let facing = dot(normal, toward);
        if facing <= 0.0 {
            continue;
        }
        // The inverse square at a sphere's size, brought to 0 at the block light's reach.
        let edge = saturate(1.0 - pow(d2 / (reach * reach), 2.0));
        let intensity = reach * reach / 225.0 * edge * edge / (1.0 + d2 * 0.12);
        let strength = intensity * facing;
        if strength < 0.002 {
            continue;
        }
        if rays < 8u {
            rays += 1u;
            // The source is a small sphere: its near edge is what the ray must reach.
            let dir = horizon_ray_jitter(toward, 0.3 / max(d, 0.5), f32(n));
            let seen = horizon_ray_trace(position + normal * 0.03, dir, max(d - 0.55, 0.01));
            traced += source.color.rgb * seen * strength;
            seen_part += max(seen.r, max(seen.g, seen.b)) * strength;
            whole_part += strength;
        } else {
            untraced += source.color.rgb * strength;
        }
    }
    // Past the budget a source is seen about as much as the strongest ones here were, and
    // however many pile up the light saturates rather than climbing.
    let sum = traced + untraced * seen_part / max(whole_part, 0.0001);
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
fn horizon_ray_hit(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> RayHit {
    return RayHit(false, distance, vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0);
}
";

/// The terrain shader for the pack's material stage `source` (with its vertex stage's
/// file, when `vertex`, and `[displacement]`'s grid when `displaced`).
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
        if rays { RAYS } else { NO_RAYS },
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
        let module = module(stage, vertex, displaced.is_some(), rays)?;
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
            constants.push(("horizon_displace_cells", f64::from(d.cells)));
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
            let solid = [color.clone(), surface];
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
        let pipeline = |layout: &wgpu::PipelineLayout,
                        vertex: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        fragment: &str,
                        blend: Option<wgpu::BlendState>| {
            pipeline_in(layout, vertex, buffers, fragment, blend, Pass::Shaded)
        };
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let instanced = layout(None);
        let buffers = [Some(vertices.clone()), Some(instances)];
        let solid = pipeline(
            &instanced,
            "horizon_material_vs",
            &buffers,
            "horizon_material_fs_solid",
            None,
        );
        let translucent = pipeline(
            &instanced,
            "horizon_material_vs",
            &buffers,
            "horizon_material_fs_translucent",
            blend,
        );
        let flat = shape.flat.map(|group| {
            let layout = layout(Some(group));
            let buffers = [Some(vertices.clone())];
            let vs = "horizon_material_vs_flat";
            (
                pipeline(&layout, vs, &buffers, "horizon_material_fs_solid", None),
                pipeline(
                    &layout,
                    vs,
                    &buffers,
                    "horizon_material_fs_translucent",
                    blend,
                ),
            )
        });
        let displaced_layout = displaced.map(|d| layout(Some(d.layout)));
        let grid = |fragment, pass| {
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
        let displaced = grid("horizon_material_fs_displaced", Pass::Shaded);
        let flat_layout = shape.flat.map(|group| layout(Some(group)));
        let flat_vertices = [Some(vertices.clone())];
        let flat_in = |fragment, pass| {
            flat_layout.as_ref().map(|layout| {
                pipeline_in(
                    layout,
                    "horizon_material_vs_flat",
                    &flat_vertices,
                    fragment,
                    None,
                    pass,
                )
            })
        };
        let prepass = Prepass {
            depth: pipeline_in(
                &instanced,
                "horizon_material_vs",
                &buffers,
                "horizon_material_fs_depth",
                None,
                Pass::Depth,
            ),
            depth_flat: flat_in("horizon_material_fs_depth", Pass::Depth),
            solid: pipeline_in(
                &instanced,
                "horizon_material_vs",
                &buffers,
                "horizon_material_fs_solid",
                None,
                Pass::After,
            ),
            flat: flat_in("horizon_material_fs_solid", Pass::After),
        };
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("{}: {e}", stage.file));
        }
        Ok(Self {
            solid,
            translucent,
            flat,
            displaced,
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
