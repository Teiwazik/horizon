//! The stages a pack can fill, each a WGSL function with a fixed signature
//! that Horizon calls inside its own passes, and the WGSL put around a
//! pack's files to build a stage: the prelude before (the inputs and the
//! resources), the entry points after.
use crate::inputs::{
    DISPLACEMENT_INPUT, Frame, MATERIAL_INPUT, POST_INPUT, SKY_INPUT, SURFACE, Struct,
    VERTEX_INPUT, WEATHER_INPUT,
};
use crate::package::Pack;

/// The bind group of a pack's resources; groups 0 to 2 are the renderer's.
pub const GROUP: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    /// The sky's colour along each view ray, before the sun, moon and stars.
    Sky,
    /// The colour of block surfaces (terrain), before the alpha test and fog.
    Material,
    /// A full-screen pass over the world's picture, under the hand and HUD.
    Post,
    /// Where each corner of a block's surface is drawn (waving plants, rolling water), for
    /// the terrain and the shadow map alike.
    Vertex,
    /// The colour of rain and snow streaks, blended over the world.
    Weather,
}

/// A resource a stage reads, declared by its prelude.
#[derive(Debug)]
pub struct Resource {
    pub binding: u32,
    pub name: &'static str,
    pub ty: &'static str,
    pub doc: &'static str,
}

const FRAME: Resource = Resource {
    binding: 0,
    name: "frame",
    ty: "var<uniform> frame: Frame",
    doc: "The frame (`Frame`).",
};

const REFLECTION: Resource = Resource {
    binding: 1,
    name: "reflection",
    ty: "var reflection: texture_2d<f32>",
    doc: "This frame's planar reflection buffer: the world as a camera mirrored about the horizontal water plane sees it, mapped like the world's picture (`screen`). Holds the scene without its fluids, so a mirror-like surface can sample what it reflects; valid while `frame.has_reflection` is 1, else a 1×1 blank.",
};

const REFLECTION_SAMPLER: Resource = Resource {
    binding: 2,
    name: "reflection_sampler",
    ty: "var reflection_sampler: sampler",
    doc: "Bilinear filtering, clamped to the edges.",
};

/// The shadow map at `binding`.
const fn shadow_map(binding: u32) -> Resource {
    Resource {
        binding,
        name: "shadow_map",
        ty: "var shadow_map: texture_depth_2d",
        doc: "This frame's shadow map: the depth of the solid terrain and entities the light sees, 0 at the light, 1 away from it (and where nothing is), stored at `shadow_position`'s mapping. Valid while `frame.has_shadow` is 1 (the pack asks for it with `[shadow]`), else a 1×1 map that lights everything.",
    }
}

/// The shadow map's comparison sampler at `binding`.
const fn shadow_sampler(binding: u32) -> Resource {
    Resource {
        binding,
        name: "shadow_sampler",
        ty: "var shadow_sampler: sampler_comparison",
        doc: "Depth comparison, bilinear (2×2 percentage-closer filtering): `textureSampleCompareLevel(shadow_map, shadow_sampler, p.xy, p.z)` is 1 where the point is lit, 0 in shadow.",
    }
}

/// The shadow map with translucent terrain at `binding`.
const fn shadow_map_translucent(binding: u32) -> Resource {
    Resource {
        binding,
        name: "shadow_map_translucent",
        ty: "var shadow_map_translucent: texture_depth_2d",
        doc: "`shadow_map` with the translucent terrain (water, stained glass, ice) the light passes through drawn in too, read with `shadow_sampler` the same way: a point lit in `shadow_map` but not here lies under such a surface, which `shadow_color` gives the colour of.",
    }
}

/// The colour of the translucent terrain in the shadow map at `binding`.
const fn shadow_color(binding: u32) -> Resource {
    Resource {
        binding,
        name: "shadow_color",
        ty: "var shadow_color: texture_2d<f32>",
        doc: "The texture colour times the tint (rgb) of the nearest translucent surface the light passes through at each texel of `shadow_map_translucent`, white where there is none; alpha is 1 on a fluid (water) and the texture's alpha, at most 0.99, on anything else. Read it with `textureLoad` at `shadow_position`'s xy times `textureDimensions(shadow_color)`.",
    }
}

const MATERIAL_IDS: Resource = Resource {
    binding: 5,
    name: "material_ids",
    ty: "var<storage, read> material_ids: array<u32>",
    doc: "Each sprite layer's `[[materials]]` id (what `input.material` reads; at least one entry).",
};

/// Functions the material and post preludes give packs.
const SHADOW_HELPERS: &str =
    "// Where the camera-relative point `position` is stored in the shadow map: xy its 0..1 texture
// coordinate (from the top left), z its depth from the light, for `textureSampleCompareLevel`.
fn shadow_position(position: vec3<f32>) -> vec3<f32> {
    let clip = frame.shadow_view_proj * vec4<f32>(position, 1.0);
    let k = frame.shadow_distortion;
    let xy = clip.xy / (length(clip.xy) * k + 1.0 - k);
    return vec3<f32>(xy.x * 0.5 + 0.5, 0.5 - xy.y * 0.5, clip.z);
}
";

const POST_RESOURCES: [Resource; 12] = [
    FRAME,
    Resource {
        binding: 1,
        name: "scene_color",
        ty: "var scene_color: texture_2d<f32>",
        doc: "The world's picture before any pass.",
    },
    Resource {
        binding: 2,
        name: "scene_depth",
        ty: "var scene_depth: texture_depth_2d",
        doc: "The world's depth, reversed (1 near, 0 far and sky).",
    },
    Resource {
        binding: 3,
        name: "previous",
        ty: "var previous: texture_2d<f32>",
        doc: "The previous pass's output (`scene_color` for the first pass).",
    },
    Resource {
        binding: 4,
        name: "linear_sampler",
        ty: "var linear_sampler: sampler",
        doc: "Bilinear filtering, clamped to the edges.",
    },
    Resource {
        binding: 5,
        name: "nearest_sampler",
        ty: "var nearest_sampler: sampler",
        doc: "Nearest texel, clamped to the edges.",
    },
    shadow_map(6),
    shadow_sampler(7),
    Resource {
        binding: 8,
        name: "history",
        ty: "var history: texture_2d<f32>",
        doc: "Last frame's input to the last pass (what the pass before it wrote), for temporal passes (TAA, denoising): put such a pass second to last and it reads its own previous output; reproject with `frame.previous_view_projection`. Black on the first frame and with a single pass.",
    },
    shadow_map_translucent(9),
    shadow_color(10),
    Resource {
        binding: 11,
        name: "scene_surface",
        ty: "var scene_surface: texture_2d<f32>",
        doc: "What the material stage's `set_surface` stored for the solid terrain at each pixel, unfiltered (`textureLoad`): read it through the prelude's `surface_at`, which knows its layout and where something else (an entity, water, glass, a particle) has drawn over the terrain since.",
    },
];

/// Functions the post prelude gives packs.
const POST_HELPERS: &str = "\
// The world's depth at `uv` (reversed: 0 is the sky).
fn depth_at(uv: vec2<f32>) -> f32 {
    let size = vec2<f32>(textureDimensions(scene_depth));
    let texel = vec2<i32>(clamp(uv * size, vec2<f32>(0.0), size - 1.0));
    return textureLoad(scene_depth, texel, 0);
}
// The camera-relative world position, in blocks, of the point at `uv` and `depth`.
fn view_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = frame.projection_inverse * clip;
    return (frame.view_inverse * vec4<f32>(view.xyz / view.w, 0.0)).xyz;
}
// Ray tracing (a prototype, `HORIZON_RT=1`): whether this device traces rays here.
fn ray_tracing() -> bool {
    return horizon_ray_available();
}
// Whether solid terrain near the camera blocks the ray. False without ray tracing.
fn ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return horizon_ray_occluded(origin, direction, distance);
}
// The colour that stained glass and other translucent terrain leave of light along the ray, 1
// through air and water. Solid terrain doesn't stop it: end it where `ray_hit` did.
fn ray_tint(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> vec3<f32> {
    return horizon_ray_tint(origin, direction, distance);
}
// The light sources that may reach `position`, for a pack choosing among them itself.
fn ray_light_count(position: vec3<f32>) -> u32 {
    return horizon_ray_light_count(position);
}
// One of them, `index` below `ray_light_count`. An id holds for this frame alone.
fn ray_light_id(position: vec3<f32>, index: u32) -> u32 {
    return horizon_ray_light_id(position, index);
}
// How strongly the source would light a surface facing `normal` at `position`, seen or not.
fn ray_light_strength(id: u32, position: vec3<f32>, normal: vec3<f32>) -> f32 {
    return horizon_ray_light_strength(id, position, normal);
}
fn ray_light_color(id: u32) -> vec3<f32> {
    return horizon_ray_light_color(id);
}
// Where the source's block is, camera-relative: what names it from one frame to the next.
fn ray_light_position(id: u32) -> vec3<f32> {
    return horizon_ray_light_position(id);
}
// The light left of the source at `position` after one ray: glass tints it, water bends it.
fn ray_light_seen(id: u32, position: vec3<f32>, normal: vec3<f32>, salt: f32) -> vec3<f32> {
    return horizon_ray_light_seen(id, position, normal, salt);
}
// Two random numbers in 0..1 for this pixel and frame, from blue noise: neighbouring pixels get
// numbers far apart, so a few of them averaged cover the range. Another `salt` (a whole number)
// gives another pair. 0.5 without ray tracing.
fn ray_noise(salt: f32) -> vec2<f32> {
    return horizon_ray_blue(salt);
}
var<private> horizon_ray_pixel: vec2<f32>;
// The direct light of the block and dynamic lights at `position`, on a surface facing `normal`,
// with two shadow rays at most: for what a ray met.
fn ray_lights(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return horizon_ray_lights_budget(position, normal, 2u);
}
// The solid terrain's texture and tint at a pixel, before any light.
fn albedo_at(texel: vec2<i32>) -> vec3<f32> {
    return horizon_ray_albedo_at(texel).rgb;
}
// What a ray met: `hit` false past its distance. `albedo` is the texture's colour with the
// block's tint, `normal` faces the ray, `sky` and `block` are the light stored there, 0..1.
struct RayHit {
    hit: bool,
    distance: f32,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    sky: f32,
    block: f32,
    // Whether the ray met the face's outer side: false from inside a block, behind its faces.
    front: bool,
}
// The nearest solid thing along the ray (a prototype, `HORIZON_RT=1`).
fn ray_hit(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> RayHit {
    return horizon_ray_hit(origin, direction, distance);
}
// The solid terrain's surface at a pixel, as the material stage's `set_surface` left it.
struct SurfaceData {
    // false where the pixel shows something else: the sky, an entity, water, glass, a particle.
    present: bool,
    // A unit vector in world axes (the face's own normal where the material set none).
    normal: vec3<f32>,
    smoothness: f32,
    // The pack's own value, passed through (a material class, the sky light, ...).
    data: f32,
}
fn surface_at(texel: vec2<i32>) -> SurfaceData {
    let size = vec2<i32>(textureDimensions(scene_surface));
    let t = clamp(texel, vec2<i32>(0), size - 1);
    let s = textureLoad(scene_surface, t, 0);
    let depth = textureLoad(scene_depth, t, 0);
    // The terrain's own depth is stored with it: anything drawn nearer since hides it.
    let present = s.w > 0.0 && abs(s.w - depth) <= s.w * 1e-5;
    let e = unpack2x16snorm(bitcast<u32>(s.x));
    var n = vec3<f32>(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    if n.y < 0.0 {
        n = vec3<f32>((1.0 - abs(n.z)) * sign(n.x), n.y, (1.0 - abs(n.x)) * sign(n.z));
    }
    return SurfaceData(present, select(vec3<f32>(0.0), normalize(n), present),
        select(0.0, s.y, present), select(0.0, s.z, present));
}
";

/// The post stage's entry points, as the renderer builds them: a full-screen
/// triangle, and a fragment that calls the pack's `post` for each pixel.
macro_rules! post_entry_points {
    () => {
        "\
struct HorizonPostOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@vertex fn horizon_vs_post(@builtin(vertex_index) i: u32) -> HorizonPostOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: HorizonPostOut;
    out.position = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}
@fragment fn horizon_fs_post(in: HorizonPostOut) -> @location(0) vec4<f32> {
    horizon_ray_pixel = in.position.xy;
    let color = textureSampleLevel(previous, linear_sampler, in.uv, 0.0);
    return post(PostInput(in.uv, in.position.xy, color, depth_at(in.uv)));
}
"
    };
}
/// The post stage's entry points alone, for a renderer that gives the prelude real rays.
pub const POST_ENTRY_POINTS: &str = post_entry_points!();
/// What the check and a device without rays give the post prelude's rays.
pub const POST_RAY_STUB: &str = "
fn horizon_ray_available() -> bool {
    return false;
}
fn horizon_ray_lights_budget(position: vec3<f32>, normal: vec3<f32>, budget: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_albedo_at(texel: vec2<i32>) -> vec4<f32> {
    return vec4<f32>(1.0);
}
fn horizon_ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return false;
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
const POST_ENTRIES: &str = concat!(
    "
fn horizon_ray_available() -> bool {
    return false;
}
fn horizon_ray_lights_budget(position: vec3<f32>, normal: vec3<f32>, budget: u32) -> vec3<f32> {
    return vec3<f32>(0.0);
}
fn horizon_ray_albedo_at(texel: vec2<i32>) -> vec4<f32> {
    return vec4<f32>(1.0);
}
fn horizon_ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return false;
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
",
    post_entry_points!()
);

/// The material stage's resources: the frame, the reflection buffer and its sampler, the
/// shadow map and its sampler.
const MATERIAL_RESOURCES: [Resource; 12] = [
    FRAME,
    REFLECTION,
    REFLECTION_SAMPLER,
    shadow_map(3),
    shadow_sampler(4),
    MATERIAL_IDS,
    Resource {
        binding: 6,
        name: "opaque_depth",
        ty: "var opaque_depth: texture_depth_2d",
        doc: "The world's depth before translucent terrain (reversed, like `scene_depth`), for the translucent pass: `textureLoad(opaque_depth, vec2<i32>(input.screen * frame.resolution), 0)` is what lies behind water or glass. Stale while solid terrain draws.",
    },
    shadow_map_translucent(7),
    shadow_color(8),
    Resource {
        binding: 9,
        name: "block_light_colors",
        ty: "var block_light_colors: texture_3d<f32>",
        doc: "The colour of the block light around the camera, from the `[[lights]]` of the light sources nearby: rgb their colours averaged by how near each is, divided by 4 (a `[[lights]]` colour goes up to 4), a the block light reaching the cell over 15 (spread round from each source, a step to an edge or corner neighbour costing its length, stopped by the blocks that stop light), 0 to 1. Read it through the prelude's `block_light_color`.",
    },
    Resource {
        binding: 10,
        name: "block_light_grid",
        ty: "var<uniform> block_light_grid: vec4<f32>",
        doc: "Where `block_light_colors` lies: xyz its corner, camera-relative, w the side of one of its cells in blocks.",
    },
    Resource {
        binding: 11,
        name: "opaque_color",
        ty: "var opaque_color: texture_2d<f32>",
        doc: "The world's picture before translucent terrain (sky, solid terrain, entities), the size of `opaque_depth`, for the translucent pass: what a surface reflects or refracts from the screen. Read it with `reflection_sampler` at `input.screen`-style coordinates. Stale while solid terrain draws; black without a `[[post]]` pass.",
    },
];

/// Functions the material prelude gives packs.
const MATERIAL_HELPERS: &str =
    "// The colour of the block light at the camera-relative `position` (rgb, from the `[[lights]]`
// of the sources nearby) and the block light reaching it (a, 0 to 1 for levels 0 to 15),
// spread round from each source and stopped by walls: a smooth stand-in for `input.light.x`
// near the camera.
fn block_light_color(position: vec3<f32>) -> vec4<f32> {
    let size = vec3<f32>(textureDimensions(block_light_colors));
    let uvw = (position - block_light_grid.xyz) / (block_light_grid.w * size);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return vec4<f32>(0.0);
    }
    let c = textureSampleLevel(block_light_colors, reflection_sampler, uvw, 0.0);
    return vec4<f32>(c.rgb * 4.0, c.a);
}
// The surface's own block texture at `uv` (the sprite's space, as `input.uv`: 0..1 across
// it, repeating), at the mip level of the point being shaded: what the neighbouring texels
// hold, for normals or detail generated from the texture.
var<private> horizon_block_layer: u32;
var<private> horizon_block_lod: f32;
fn block_texture(uv: vec2<f32>) -> vec4<f32> {
    return horizon_block_sample(uv, horizon_block_layer, horizon_block_lod);
}
// `block_texture` at mip level `lod` (0 the full-size texture), wherever the point is.
fn block_texture_level(uv: vec2<f32>, lod: f32) -> vec4<f32> {
    return horizon_block_sample(uv, horizon_block_layer, lod);
}
// The block texture's size, in texels (16 x 16 for vanilla's).
fn block_texture_size() -> vec2<f32> {
    return horizon_block_size();
}
// The mip level `block_texture` reads (0 up close, more far away).
fn block_texture_lod() -> f32 {
    return horizon_block_lod;
}
// LabPBR: whether the resource packs ship normal (`_n`) or specular (`_s`) maps beside their
// block textures. Without them `block_normal_map` and `block_specular_map` give the defaults
// below for every texture.
fn block_pbr_maps() -> bool {
    return horizon_block_pbr_present();
}
// The surface's LabPBR normal map at `uv` (as `block_texture`): rg the normal's x and y in
// the texture's space (x along `input.tangent`, y down the texture; 0..1 for -1..1), b
// ambient occlusion (1 none), a height (1 the top). (0.5, 0.5, 1, 1) where a texture has
// none.
fn block_normal_map(uv: vec2<f32>) -> vec4<f32> {
    return horizon_block_pbr(uv, horizon_block_layer, horizon_block_lod, 0u);
}
// The surface's LabPBR specular map at `uv`: r perceptual smoothness, g F0 up to 229/255 and
// a metal's id above, b porosity or subsurface scattering, a emission (up to 254/255; 1
// none). (0, 0, 0, 0) where a texture has none.
fn block_specular_map(uv: vec2<f32>) -> vec4<f32> {
    return horizon_block_pbr(uv, horizon_block_layer, horizon_block_lod, 1u);
}
// `block_specular_map` at mip level `lod` (0 the full-size map).
fn block_specular_map_level(uv: vec2<f32>, lod: f32) -> vec4<f32> {
    return horizon_block_pbr(uv, horizon_block_layer, lod, 1u);
}
// Ray tracing (a prototype, `HORIZON_RT=1`): whether this device traces rays here.
fn ray_tracing() -> bool {
    return horizon_ray_available();
}
// Whether solid terrain near the camera blocks the ray from the camera-relative `origin`
// along the unit `direction`, up to `distance` blocks. False without ray tracing.
fn ray_occluded(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> bool {
    return horizon_ray_occluded(origin, direction, distance);
}
// What a ray met: `hit` false past its distance. `albedo` is the texture's colour with the
// block's tint, `normal` faces the ray, `sky` and `block` are the light stored there, 0..1.
struct RayHit {
    hit: bool,
    distance: f32,
    albedo: vec3<f32>,
    normal: vec3<f32>,
    sky: f32,
    block: f32,
    // Whether the ray met the face's outer side: false from inside a block, behind its faces.
    front: bool,
}
// The nearest solid thing along the ray (a prototype, `HORIZON_RT=1`).
fn ray_hit(origin: vec3<f32>, direction: vec3<f32>, distance: f32) -> RayHit {
    return horizon_ray_hit(origin, direction, distance);
}
var<private> horizon_ray_pixel: vec2<f32>;
// The light that reaches `origin` along `direction` (a body `radius` wide, the tangent of its
// half angle, for a soft edge): 0 behind anything solid, tinted through translucent terrain.
fn ray_light(origin: vec3<f32>, direction: vec3<f32>, distance: f32, radius: f32) -> vec3<f32> {
    return horizon_ray_light(origin, direction, distance, radius);
}
// The direct light of the block and dynamic lights near `position`, on a surface facing
// `normal`, each shadowed and tinted by the rays to it, added in linear space.
fn ray_lights(position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    return horizon_ray_lights(position, normal);
}
// The LabPBR maps' size, in texels (they may be smaller than `block_texture_size`).
fn block_pbr_size() -> vec2<f32> {
    return horizon_block_pbr_size();
}
// `block_normal_map` at mip level `lod`.
fn block_normal_map_level(uv: vec2<f32>, lod: f32) -> vec4<f32> {
    return horizon_block_pbr(uv, horizon_block_layer, lod, 0u);
}
// The tangent of the surface at `position` (camera-relative) with sprite coordinates `uv` and
// unit `normal`: xyz the direction `uv.x` grows along, w the sign that makes
// `cross(normal, xyz) * w` the direction `uv.y` grows along (down the texture).
fn horizon_material_tangent(position: vec3<f32>, uv: vec2<f32>, normal: vec3<f32>) -> vec4<f32> {
    let dp1 = dpdx(position);
    let dp2 = dpdy(position);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, normal);
    let dp1perp = cross(normal, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let tl = length(t);
    if tl < 1e-12 {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let tangent = normalize(t - normal * dot(normal, t));
    return vec4<f32>(tangent, select(-1.0, 1.0, dot(cross(normal, tangent), b) >= 0.0));
}

// What the post stage's `surface_at` reads for this point of solid terrain: its `normal` (a
// unit vector in world axes), `smoothness` (0..1, say for reflections) and `data`, a value of
// the pack's own passed through. Unset, the point keeps the face's own normal and 0 for both;
// translucent terrain stores none.
var<private> horizon_surface: vec4<f32>;
fn set_surface(normal: vec3<f32>, smoothness: f32, data: f32) {
    // Octahedral, y up: the upper half folds onto xz, the lower half unfolds around it.
    let n = normal / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var e = n.xz;
    if n.y < 0.0 {
        e = (1.0 - abs(n.zx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
    }
    horizon_surface = vec4<f32>(bitcast<f32>(pack2x16snorm(e)), smoothness, data, 1.0);
}
";

/// What the renderer gives the material prelude's `block_texture` (its terrain's sprite
/// arrays); the check and the entries below put a stand-in in its place.
macro_rules! block_sampling_stub {
    () => {
        "fn horizon_block_sample(uv: vec2<f32>, layer: u32, lod: f32) -> vec4<f32> {
    return vec4<f32>(1.0);
}
fn horizon_block_size() -> vec2<f32> {
    return vec2<f32>(16.0);
}
fn horizon_block_pbr(uv: vec2<f32>, layer: u32, lod: f32, map: u32) -> vec4<f32> {
    return select(vec4<f32>(0.5, 0.5, 1.0, 1.0), vec4<f32>(0.0), map == 1u);
}
fn horizon_block_pbr_present() -> bool {
    return false;
}
fn horizon_block_pbr_size() -> vec2<f32> {
    return vec2<f32>(16.0);
}
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
"
    };
}
pub const BLOCK_SAMPLING_STUB: &str = block_sampling_stub!();

/// Entry points with the material stage's inputs: the renderer calls
/// `material` from its own terrain shader in the same way.
const MATERIAL_ENTRIES: &str = concat!(
    block_sampling_stub!(),
    "\
struct HorizonMaterialIn {
    @builtin(position) position: vec4<f32>,
    @location(0) albedo: vec4<f32>,
    @location(1) tint: vec4<f32>,
    @location(2) light: vec2<f32>,
    @location(3) lightmap: vec3<f32>,
    @location(4) world: vec3<f32>,
    @location(5) uv: vec2<f32>,
    @location(6) @interpolate(flat) translucent: u32,
    @location(7) @interpolate(flat) face: u32,
    @location(8) @interpolate(flat) fluid_quad: u32,
    @location(9) @interpolate(flat) layer: u32,
}
@fragment fn horizon_fs_material(in: HorizonMaterialIn) -> @location(0) vec4<f32> {
    let fluid = select(0u, select(2u, 1u, in.translucent != 0u), in.fluid_quad != 0u);
    let screen = in.position.xy / frame.resolution;
    let n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    let a = abs(n);
    let m = max(a.x, max(a.y, a.z));
    let axis = select(select(vec3(0.0, 0.0, sign(n.z)), vec3(0.0, sign(n.y), 0.0), a.y == m),
        vec3(sign(n.x), 0.0, 0.0), a.x == m);
    let flat_n = select(n, axis, m > 0.998);
    let normal = select(flat_n, -flat_n, dot(flat_n, in.world) > 0.0);
    let material_id = material_ids[min(in.layer & 0x7fffu, arrayLength(&material_ids) - 1u)];
    let tangent = horizon_material_tangent(in.world, in.uv, normal);
    let input = MaterialInput(in.albedo, in.tint, in.light, in.lightmap, in.world, in.uv, in.translucent != 0u, 0u, material_id,
        screen, in.face, fluid, 0.0, normal, tangent, vec3<f32>(0.0), 0.0);
    return material(input).color;
}
"
);

/// Entry points with the sky stage's inputs: the renderer calls `sky` from
/// its own sky shader in the same way.
const SKY_ENTRIES: &str = "\
struct HorizonSkyIn {
    @builtin(position) position: vec4<f32>,
    @location(0) direction: vec3<f32>,
    @location(1) color: vec4<f32>,
}
@fragment fn horizon_fs_sky(in: HorizonSkyIn) -> @location(0) vec4<f32> {
    return sky(SkyInput(normalize(in.direction), in.color));
}
";

/// An entry point with the weather stage's inputs: the renderer calls `weather` from its own
/// weather shader in the same way.
const WEATHER_ENTRIES: &str = "struct HorizonWeatherIn {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) light: vec2<f32>,
    @location(2) world: vec3<f32>,
    @location(3) @interpolate(flat) snow: u32,
}
@fragment fn horizon_fs_weather(in: HorizonWeatherIn) -> @location(0) vec4<f32> {
    return weather(WeatherInput(in.color, in.light, in.world, in.snow != 0u));
}
";

/// The vertex stage's resources: the frame and the material ids.
const VERTEX_RESOURCES: [Resource; 2] = [FRAME, MATERIAL_IDS];

/// An entry point with the vertex stage's inputs: the renderer calls `vertex` from its own
/// terrain and shadow vertex shaders in the same way.
const VERTEX_ENTRIES: &str = "\
@vertex fn horizon_vs_vertex(
    @location(0) position: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) light: vec2<f32>,
    @location(3) tint: vec4<f32>, @location(4) layer: u32, @location(5) flags: u32,
) -> @builtin(position) vec4<f32> {
    let material = material_ids[min(layer & 0x7fffu, arrayLength(&material_ids) - 1u)];
    let input = VertexInput(position, uv, material, light, tint, (flags >> 28u) & 7u, flags >> 31u);
    return frame.projection * frame.view * vec4<f32>(vertex(input), 1.0);
}
";

/// `fn displacement(input: DisplacementInput) -> f32`, the material stage's function
/// under `[displacement]`.
pub const DISPLACEMENT_SIGNATURE: &str = "fn displacement(input: DisplacementInput) -> f32";

/// An entry point calling the material stage's `displacement`, after its entries under
/// `[displacement]`: the renderer calls it from its own terrain vertex shader in the same way.
const DISPLACEMENT_ENTRIES: &str = "@vertex fn horizon_vs_displacement(
    @location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
    @location(3) layer: u32, @location(4) height: f32,
) -> @builtin(position) vec4<f32> {
    let material = material_ids[min(layer & 0x7fffu, arrayLength(&material_ids) - 1u)];
    let depth = displacement(DisplacementInput(position, normal, uv, material, height));
    return frame.projection * frame.view * vec4<f32>(position - normal * depth, 1.0);
}
";

impl StageKind {
    pub const ALL: [StageKind; 5] = [
        StageKind::Sky,
        StageKind::Vertex,
        StageKind::Material,
        StageKind::Weather,
        StageKind::Post,
    ];

    /// The manifest's name for the stage (`[sky]`, `[material]`, `[[post]]`).
    pub fn name(self) -> &'static str {
        match self {
            StageKind::Sky => "sky",
            StageKind::Material => "material",
            StageKind::Post => "post",
            StageKind::Vertex => "vertex",
            StageKind::Weather => "weather",
        }
    }

    /// The function a pack defines: it has the stage's name.
    pub fn function(self) -> &'static str {
        self.name()
    }

    pub fn input(self) -> &'static Struct {
        match self {
            StageKind::Sky => &SKY_INPUT,
            StageKind::Material => &MATERIAL_INPUT,
            StageKind::Post => &POST_INPUT,
            StageKind::Vertex => &VERTEX_INPUT,
            StageKind::Weather => &WEATHER_INPUT,
        }
    }

    /// The type the function returns.
    pub fn output(self) -> &'static str {
        match self {
            StageKind::Material => SURFACE.name,
            StageKind::Sky | StageKind::Post | StageKind::Weather => "vec4<f32>",
            StageKind::Vertex => "vec3<f32>",
        }
    }

    /// `fn <name>(input: <Input>) -> <Output>`.
    pub fn signature(self) -> String {
        format!(
            "fn {}(input: {}) -> {}",
            self.function(),
            self.input().name,
            self.output()
        )
    }

    pub fn resources(self) -> &'static [Resource] {
        match self {
            StageKind::Post => &POST_RESOURCES,
            StageKind::Material => &MATERIAL_RESOURCES,
            StageKind::Sky | StageKind::Weather => std::slice::from_ref(&FRAME),
            StageKind::Vertex => &VERTEX_RESOURCES,
        }
    }

    /// Everything the pack's files may use, declared before them.
    pub fn prelude(self) -> String {
        let mut out =
            String::from("// Horizon shader API prelude (docs/api/shaders-reference.md)\n");
        out += &Frame::WGSL.wgsl();
        out += &self.input().wgsl();
        if self == StageKind::Material {
            out += &SURFACE.wgsl();
            // The vertex stage's file joins the material's module (`compose`).
            out += &VERTEX_INPUT.wgsl();
            out += &DISPLACEMENT_INPUT.wgsl();
        }
        for r in self.resources() {
            out += &format!("@group({GROUP}) @binding({}) {};\n", r.binding, r.ty);
        }
        if self == StageKind::Post {
            out += POST_HELPERS;
        }
        if self == StageKind::Material {
            out += MATERIAL_HELPERS;
        }
        if self != StageKind::Sky && self != StageKind::Weather {
            out += SHADOW_HELPERS;
        }
        out
    }

    /// The entry points put after the pack's files.
    pub fn entries(self) -> &'static str {
        match self {
            StageKind::Sky => SKY_ENTRIES,
            StageKind::Weather => WEATHER_ENTRIES,
            StageKind::Material => MATERIAL_ENTRIES,
            StageKind::Post => POST_ENTRIES,
            StageKind::Vertex => VERTEX_ENTRIES,
        }
    }
}

/// The first binding of a pack's `[[textures]]` in group 3; their samplers follow at
/// `TEXTURE_SAMPLER_BINDING`.
pub const TEXTURE_BINDING: u32 = 16;
pub const TEXTURE_SAMPLER_BINDING: u32 = 24;

/// The binding of a pack's first post target (`[[post]] output`) in group 3, the others after.
pub const POST_TARGET_BINDING: u32 = 32;

/// The declarations of a pack's post targets, part of the post stage's prelude.
pub fn post_target_declarations(manifest: &crate::manifest::Manifest) -> String {
    manifest
        .post_targets()
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            format!(
                "@group({GROUP}) @binding({}) var {name}: texture_2d<f32>;\n",
                POST_TARGET_BINDING + i as u32
            )
        })
        .collect()
}

/// The declarations of a pack's `[[textures]]`, part of every stage's prelude.
pub fn texture_declarations(textures: &[crate::manifest::PackTexture]) -> String {
    let mut out = String::new();
    for (i, t) in textures.iter().enumerate() {
        out += &format!(
            "@group({GROUP}) @binding({}) var {}: texture_2d<f32>;\n\
             @group({GROUP}) @binding({}) var {}_sampler: sampler;\n",
            TEXTURE_BINDING + i as u32,
            t.id,
            TEXTURE_SAMPLER_BINDING + i as u32,
            t.id
        );
    }
    out
}

/// A stage's WGSL, and where each part of it came from.
#[derive(Clone, Debug)]
pub struct Composed {
    pub text: String,
    /// (the file, or `<prelude>` / `<entries>`, its byte range in `text`).
    pub parts: Vec<(String, std::ops::Range<usize>)>,
}

impl Composed {
    /// The file, line and column (1-based) of a byte offset in `text`.
    pub fn locate(&self, offset: usize) -> (&str, usize, usize) {
        let (file, range) = self
            .parts
            .iter()
            .find(|(_, r)| r.contains(&offset))
            .or(self.parts.last())
            .expect("a composed stage has parts");
        let before = &self.text[range.start..offset.clamp(range.start, range.end)];
        let line = before.matches('\n').count() + 1;
        let column = before.len() - before.rfind('\n').map_or(0, |i| i + 1) + 1;
        (file, line, column)
    }
}

/// The WGSL of the stage `kind` whose own file is `file`: the prelude, the
/// pack's shared files in order, the file, then the entry points.
pub fn compose(pack: &Pack, kind: StageKind, file: &str) -> Composed {
    let mut composed = Composed {
        text: String::new(),
        parts: Vec::new(),
    };
    let mut push = |name: &str, text: &str| {
        let start = composed.text.len();
        composed.text += text;
        if !text.ends_with('\n') {
            composed.text.push('\n');
        }
        composed
            .parts
            .push((name.to_owned(), start..composed.text.len()));
    };
    let mut prelude = kind.prelude() + &texture_declarations(&pack.manifest.textures);
    if kind == StageKind::Post {
        prelude += &post_target_declarations(&pack.manifest);
    }
    push("<prelude>", &prelude);
    for shared in pack.manifest.shared.iter().filter(|s| *s != file) {
        push(shared, pack.source(shared).unwrap_or_default());
    }
    // The stage's own extra files (`files`), between the shared ones and its file.
    let m = &pack.manifest;
    let stage = match kind {
        StageKind::Sky => m.sky.as_ref(),
        StageKind::Material => m.material.as_ref(),
        StageKind::Vertex => m.vertex.as_ref(),
        StageKind::Weather => m.weather.as_ref(),
        StageKind::Post => m.post.iter().find(|s| s.file == file),
    };
    for extra in stage.into_iter().flat_map(|s| &s.files) {
        if extra != file && !m.shared.contains(extra) {
            push(extra, pack.source(extra).unwrap_or_default());
        }
    }
    push(file, pack.source(file).unwrap_or_default());
    // The terrain's pipelines hold the material and the vertex stage in one module.
    if kind == StageKind::Material
        && let Some(vertex) = &pack.manifest.vertex
        && vertex.file != file
        && !pack.manifest.shared.contains(&vertex.file)
    {
        push(&vertex.file, pack.source(&vertex.file).unwrap_or_default());
    }
    if kind == StageKind::Material && pack.manifest.displacement.is_some() {
        push(
            "<entries>",
            &(kind.entries().to_owned() + DISPLACEMENT_ENTRIES),
        );
    } else {
        push("<entries>", kind.entries());
    }
    composed
}
