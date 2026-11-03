//! The shadow map of Horizon's own shader packs (`[shadow]`, docs/shaders.md): each frame the
//! solid terrain and entities the light (the sun by day, the moon by night) sees are drawn
//! into a square depth map, which the material and post stages read as `shadow_map`. The shadow camera is
//! the one `shaders/` packs use (`pack_uniforms::shadow_camera`), so the renderer's shadow
//! visibility and draw lists (`ShadowView`) serve both; the map's texels are distorted toward
//! the camera as the prelude's `shadow_position` undoes.

use horizon_shader_api::{
    Frame as FrameInputs, check::CheckedStage, manifest::Shadow as Settings, stages::GROUP,
};
use horizon_shaders::Shadow;

use super::native_material::TerrainShape;
use super::pack_post::ShadowView;
use super::wgsl_prefix::{TERRAIN, prefixed, section};

const PREFIX: &str = "horizon_terrain_";

/// The shadow pass's entries: terrain's vertices through the shadow camera, distorted, and the
/// cutout test (leaves' and plants' holes let the light through).
const ENTRIES: &str = "
struct HorizonShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) tint: vec3<f32>,
    @location(3) @interpolate(flat) fluid: u32,
    // The point on the flat face, camera-relative (`horizon_shadow_taken`).
    @location(4) base: vec3<f32>,
}
// `color` is the vertex colour: the tint, and the fluid flag in its alpha's high nibble
// (mesher.rs's `fluid_face`).
fn horizon_shadow_vertex(pos: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>)
    -> HorizonShadowOut {
    let clip = frame.shadow_view_proj * vec4(pos, 1.0);
    let k = frame.shadow_distortion;
    let xy = clip.xy / (length(clip.xy) * k + 1.0 - k);
    var out: HorizonShadowOut;
    out.clip = vec4(xy, clamp(clip.z, 0.0, 1.0), 1.0);
    out.uv = uv;
    out.layer = layer;
    out.tint = color.rgb;
    let high = u32(round(color.a * 255.0)) >> 4u;
    out.fluid = u32(high >= 8u && (high & 7u) < 6u);
    out.base = pos;
    return out;
}
@vertex fn horizon_shadow_vs(
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>,
    @location(2) layer: u32, @location(3) color: vec4<f32>, @location(4) light: vec2<u32>,
    @location(5) origin: vec4<f32>) -> HorizonShadowOut {
    let atlas = horizon_terrain_terrain_uv(uv, origin.w);
    let pos = horizon_material_displace(position + origin.xyz, atlas, layer, color, light);
    return horizon_shadow_vertex(pos, atlas, layer, color);
}
@vertex fn horizon_shadow_vs_flat(
    @builtin(vertex_index) index: u32,
    @location(0) position: vec3<f32>, @location(1) uv: vec2<u32>, @location(2) layer: u32,
    @location(3) color: vec4<f32>, @location(4) light: vec2<u32>
) -> HorizonShadowOut {
    let origin = horizon_terrain_slot_origins[horizon_terrain_block_slots[index >> 8u]];
    let atlas = horizon_terrain_terrain_uv(uv, origin.w);
    let pos = horizon_material_displace(position + origin.xyz, atlas, layer, color, light);
    return horizon_shadow_vertex(pos, atlas, layer, color);
}
@fragment fn horizon_shadow_fs(in: HorizonShadowOut) {
    if horizon_shadow_taken(in.base, in.layer) { discard; }
    horizon_shadow_cutout(in);
}
fn horizon_shadow_cutout(in: HorizonShadowOut) {
    // Bit 15 of the layer marks opaque leaves (forceOpaque): no alpha cutout.
    // A `solid_materials` sprite neither.
    let alpha = horizon_terrain_sprite(in.uv, in.layer & 0x7fffu).a;
    if alpha < 0.5 && (in.layer & 0x8000u) == 0u && !horizon_shadow_solid(in.layer) { discard; }
}
// Translucent terrain: its colour where it is the nearest thing the light passes through;
// alpha 1 marks a fluid, other surfaces keep their alpha below it.
@fragment fn horizon_shadow_fs_translucent(in: HorizonShadowOut) -> @location(0) vec4<f32> {
    let color = horizon_terrain_sprite(in.uv, in.layer & 0x7fffu);
    if color.a < 0.01 { discard; }
    return vec4(color.rgb * in.tint, select(min(color.a, 0.99), 1.0, in.fluid != 0u));
}
";

/// `shadow_color`'s format.
const SHADOW_COLOR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The shader pack's shadow camera for `settings`: orthographic over `distance` each side,
/// staying put while the camera walks a `SNAP`-block cell. The map is distorted about its
/// centre, so a centre that moves with the camera, even by a texel, resamples every
/// shadow but the nearest each frame and they shimmer; a still one maps the world to the
/// same texels until the next cell. The light turns smoothly with the sun: it moves 0.3
/// degrees a second, too slowly for the turn to shimmer, and steps would show. Its depth
/// reaches 256 blocks further toward the light than `shaders/` packs' (near -256), so the
/// ground over a deep cave still casts its shadow.
pub fn camera(settings: &Settings) -> Shadow {
    Shadow {
        resolution: settings.resolution,
        distance: settings.distance,
        render_mul: 1.0,
        interval: SNAP,
        euclidean_snap: true,
        near: 256.0,
        far: super::pack_uniforms::SHADOW_EYE + 2.0 * settings.distance + 384.0,
        ..Shadow::default()
    }
}

/// The shadow map's terrain is drawn again every this many frames; the frames between reuse
/// it, with the light and the shadow camera it was drawn with, and draw only the entities
/// over it. Waving plants' shadows move at that rate; the rest of the terrain stands still.
pub const SHADOW_TERRAIN_FRAMES: u32 = 2;

/// The cell the shadow camera keeps to, in blocks along the shadow view's axes (its
/// corner is the centre): the player stays within 11 blocks of it, where the map is
/// still over three times as fine as at its edge.
const SNAP: f32 = 8.0;

/// The ground one shadow texel covers right under the camera, in blocks: a 32nd, half a
/// texture pixel, so shadow edges near the player stay as sharp as the blocks casting them.
const CENTRE_TEXEL: f32 = 1.0 / 32.0;

/// How strongly the map is distorted toward its centre (`Frame::shadow_distortion`).
/// `xy / (|xy| k + 1 - k)` magnifies the centre by 1 / (1 - k); an undistorted texel
/// covers 2 distance / resolution blocks, so k is what brings that down to
/// `CENTRE_TEXEL`. None when the map is already that fine; at most 0.95, past which the
/// edge of the map gets too few texels.
pub fn distortion(distance: f32, resolution: u32) -> f32 {
    let texel = 2.0 * distance / resolution.max(1) as f32;
    (1.0 - CENTRE_TEXEL / texel).clamp(0.0, 0.95)
}

/// The shadow map's depth: a square map of the pack's resolution, or a 1×1 one (cleared to
/// "far", so everything is lit) for a pack without `[shadow]`; the same with translucent
/// terrain drawn in, and that terrain's colour.
pub struct ShadowMap {
    /// The terrain alone (`begin_terrain`), drawn every `SHADOW_TERRAIN_FRAMES`th frame.
    terrain: wgpu::Texture,
    terrain_view: wgpu::TextureView,
    /// The map the stages read: the terrain's, copied each frame, with the entities over it.
    depth: wgpu::Texture,
    pub view: wgpu::TextureView,
    translucent: wgpu::Texture,
    pub translucent_view: wgpu::TextureView,
    pub color_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// The 1×1 map still holds the zeroes it was made with.
    pub blank_uncleared: bool,
}

impl ShadowMap {
    pub fn new(device: &wgpu::Device, settings: Option<&Settings>) -> Self {
        let size = settings
            .map_or(1, |s| s.resolution)
            .clamp(1, device.limits().max_texture_dimension_2d);
        let texture = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | usage,
                view_formats: &[],
            })
        };
        let terrain = texture(
            "Native shader pack shadow map terrain",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::COPY_SRC,
        );
        let depth = texture(
            "Native shader pack shadow map",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::COPY_DST,
        );
        let translucent = texture(
            "Native shader pack translucent shadow map",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::COPY_DST,
        );
        let color = texture(
            "Native shader pack shadow colour",
            SHADOW_COLOR,
            wgpu::TextureUsages::empty(),
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Native shader pack shadow comparison"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            terrain_view: terrain.create_view(&Default::default()),
            terrain,
            view: depth.create_view(&Default::default()),
            depth,
            translucent_view: translucent.create_view(&Default::default()),
            translucent,
            color_view: color.create_view(&Default::default()),
            sampler,
            blank_uncleared: settings.is_none(),
        }
    }

    /// The translucent pass: the solid terrain's map copied into the translucent one, which
    /// it goes on drawing into, and the colour cleared to white (nothing to pass through).
    /// Entities stay out of it: where one casts, the solid map is the darker and decides.
    pub fn begin_translucent<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'e>>,
    ) -> wgpu::RenderPass<'e> {
        encoder.copy_texture_to_texture(
            self.terrain.as_image_copy(),
            self.translucent.as_image_copy(),
            self.terrain.size(),
        );
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.color_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.translucent_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes,
            ..Default::default()
        })
    }

    /// A depth pass into the terrain's map that clears it to "far".
    pub fn begin_terrain<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'e>>,
    ) -> wgpu::RenderPass<'e> {
        depth_pass(
            encoder,
            label,
            &self.terrain_view,
            wgpu::LoadOp::Clear(1.0),
            timestamp_writes,
        )
    }

    /// A depth pass into the map the stages read, which starts as the terrain's.
    pub fn begin<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        label: &str,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'e>>,
    ) -> wgpu::RenderPass<'e> {
        encoder.copy_texture_to_texture(
            self.terrain.as_image_copy(),
            self.depth.as_image_copy(),
            self.terrain.size(),
        );
        depth_pass(
            encoder,
            label,
            &self.view,
            wgpu::LoadOp::Load,
            timestamp_writes,
        )
    }

    /// Clears the 1×1 map of a pack without `[shadow]` once.
    pub fn clear_blank(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if std::mem::take(&mut self.blank_uncleared) {
            drop(self.begin_terrain(encoder, "Native shader pack shadow map clear", None));
            drop(self.begin(encoder, "Native shader pack shadow map clear", None));
            drop(self.begin_translucent(encoder, "Native shader pack shadow colour clear", None));
        }
    }
}

/// A depth-only pass into `view`.
fn depth_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<f32>,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'e>>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view,
            depth_ops: Some(wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        ..Default::default()
    })
}

/// The pipelines that draw terrain into the shadow map, for the renderer's instanced draws
/// and, where they run, its flat ones, and the frame they read (group 3, the uniform alone:
/// the stages' own group 3 also holds the map being drawn).
pub struct NativeShadow {
    pub camera: Shadow,
    pub solid: wgpu::RenderPipeline,
    pub flat: Option<wgpu::RenderPipeline>,
    /// The translucent terrain's draws into the translucent map and the colour, instanced
    /// and flat.
    pub translucent: wgpu::RenderPipeline,
    pub flat_translucent: Option<wgpu::RenderPipeline>,
    /// The entities' draw, with the entities' group 0 (`ENTITY_ENTRIES`).
    pub entity: wgpu::RenderPipeline,
    pub group: wgpu::BindGroup,
    /// `[displacement]`'s grid (gfx/displace.rs's shadow list).
    pub displaced: Option<wgpu::RenderPipeline>,
}

/// What the shadow pass's grid is built from: `[displacement]`'s values, the material stage
/// and group 1's layout.
pub struct ShadowGrid<'a> {
    pub distance: f32,
    pub cells: u32,
    pub steps: bool,
    pub material: &'a CheckedStage,
    pub layout: &'a wgpu::BindGroupLayout,
}

/// The entity draw of the shadow pass: the renderer's opaque entity quads (`EntityBatch`'s
/// first material) through the shadow camera, with their texture's cutout. The sky's bodies,
/// clouds, beacon beams and breaking cracks share that buffer but are unlit (entity.wgsl's
/// overlays) and cast nothing.
const ENTITY_ENTRIES: &str = "
@group(0) @binding(1) var horizon_atlas: texture_2d<f32>;
@group(0) @binding(2) var horizon_atlas_sampler: sampler;
struct HorizonEntityShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
}
@vertex fn horizon_entity_shadow_vs(@location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>, @location(2) overlay: u32, @location(3) color: vec4<f32>,
    @location(5) normal: vec4<f32>) -> HorizonEntityShadowOut {
    let clip = frame.shadow_view_proj * vec4(position, 1.0);
    let k = frame.shadow_distortion;
    let xy = clip.xy / (length(clip.xy) * k + 1.0 - k);
    // A particle (particles.rs's normal): a quad turned to the camera has no shape to cast.
    let particle = normal.w == 0.0 && normal.y > 0.99;
    let unlit = (overlay >= 2u && overlay <= 4u) || overlay == 6u
        || (overlay >= 9u && overlay <= 12u) || particle;
    var out: HorizonEntityShadowOut;
    // Unlit geometry lands outside the clip volume.
    out.clip = select(vec4(xy, clamp(clip.z, 0.0, 1.0), 1.0), vec4(2.0, 2.0, 2.0, 1.0), unlit);
    out.uv = uv;
    out.alpha = color.a;
    return out;
}
@fragment fn horizon_entity_shadow_fs(in: HorizonEntityShadowOut) {
    let alpha = textureSampleLevel(horizon_atlas, horizon_atlas_sampler, in.uv, 0.0).a;
    if alpha * in.alpha < 0.5 { discard; }
}
";

/// Corners stay where terrain put them, without a vertex stage.
const DISPLACE_VANILLA: &str = "
fn horizon_material_displace(pos: vec3<f32>, uv: vec2<f32>, layer: u32, color: vec4<f32>,
    light: vec2<u32>) -> vec3<f32> {
    return pos;
}
";

/// No `[displacement]` grid: the flat terrain casts everywhere.
const UNTAKEN: &str = "
fn horizon_shadow_taken(base: vec3<f32>, layer: u32) -> bool {
    return false;
}
";

/// `[displacement]`'s grid in the shadow map (gfx/displace.rs), cast in the flat quads' place.
const GRID_ENTRIES: &str = "
@group(3) @binding(12) var horizon_pbr_normals: texture_2d_array<f32>;
fn horizon_shadow_taken(base: vec3<f32>, layer: u32) -> bool {
    return length(base) < horizon_displace_distance - 0.05 && horizon_displaced_layer(layer);
}
@vertex fn horizon_shadow_vs_displaced(@builtin(vertex_index) v: u32,
    @builtin(instance_index) instance: u32) -> HorizonShadowOut {
    let p = horizon_displace_point(v, instance);
    var out = horizon_shadow_vertex(p.position, p.uv, p.layer, p.color);
    out.base = p.base;
    return out;
}
@fragment fn horizon_shadow_fs_displaced(in: HorizonShadowOut) {
    if length(in.base) >= horizon_displace_distance + 0.05 { discard; }
    horizon_shadow_cutout(in);
}
";

/// The shadow pass's shader: the pack's vertex stage (its prelude declares the frame and the
/// material ids) and its call when it has one, else the frame and the material ids alone;
/// `solid`, the `[[materials]]` ids whose cutouts cast whole shadows. Under `[displacement]`
/// (`grid`, the material stage, whose module holds the vertex stage and `displacement`), its
/// grid too.
fn shader_source(
    vertex: Option<&CheckedStage>,
    solid: &[u32],
    grid: Option<&CheckedStage>,
) -> String {
    let (head, displace) = match (grid, vertex) {
        (Some(material), _) => (
            format!(
                "{}{}{}{}{GRID_ENTRIES}",
                material.source,
                horizon_shader_api::stages::BLOCK_SAMPLING_STUB,
                super::displace::RELIEF,
                super::displace::GRID
            ),
            if vertex.is_some() {
                super::native_material::DISPLACE_PACK
            } else {
                DISPLACE_VANILLA
            },
        ),
        (None, Some(stage)) => (
            stage.source.clone() + UNTAKEN,
            super::native_material::DISPLACE_PACK,
        ),
        (None, None) => (
            format!(
                "{}@group({GROUP}) @binding(0) var<uniform> frame: Frame;\n\
                 @group({GROUP}) @binding(5) var<storage, read> material_ids: array<u32>;\n{UNTAKEN}",
                FrameInputs::WGSL.wgsl()
            ),
            DISPLACE_VANILLA,
        ),
    };
    let solid = if solid.is_empty() {
        "false".to_owned()
    } else {
        let ids: Vec<String> = solid.iter().map(|id| format!("m == {id}u")).collect();
        ids.join(" || ")
    };
    format!(
        "{head}\n// Horizon's terrain (terrain.wgsl)\n{}{displace}\
         fn horizon_shadow_solid(layer: u32) -> bool {{\n    \
         let m = material_ids[min(layer & 0x7fffu, arrayLength(&material_ids) - 1u)];\n    \
         return {solid};\n}}\n{ENTRIES}",
        prefixed(
            section(TERRAIN, "struct Globals {", "// Sky (SkyRenderer)"),
            PREFIX
        )
    )
}

/// The shadow pass's entity pipeline: the entities' vertices and group 0, the frame at
/// group 3.
fn entity_pipeline(
    device: &wgpu::Device,
    entities: &wgpu::BindGroupLayout,
    frame_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let text = format!(
        "{}@group({GROUP}) @binding(0) var<uniform> frame: Frame;\n{ENTITY_ENTRIES}",
        FrameInputs::WGSL.wgsl()
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Native shader pack entity shadow"),
        source: wgpu::ShaderSource::Wgsl(text.into()),
    });
    let mut groups: [Option<&wgpu::BindGroupLayout>; GROUP as usize + 1] = Default::default();
    groups[0] = Some(entities);
    groups[GROUP as usize] = Some(frame_layout);
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Native shader pack entity shadow"),
        bind_group_layouts: &groups,
        ..Default::default()
    });
    let attributes = wgpu::vertex_attr_array![0=>Float32x3,1=>Float32x2,2=>Uint32,3=>Unorm8x4,4=>Uint8x4,5=>Snorm8x4,6=>Float32x2];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Native shader pack entity shadow"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("horizon_entity_shadow_vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<super::Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            })],
        },
        // Both sides, as the terrain: thin parts (ears, item quads) cast from either.
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: wgpu::DepthBiasState {
                constant: 0,
                slope_scale: 1.5,
                clamp: 0.0,
            },
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("horizon_entity_shadow_fs"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl NativeShadow {
    pub fn new(
        device: &wgpu::Device,
        shape: &TerrainShape<'_>,
        frame: &super::native_post::NativeFrame,
        settings: &Settings,
        vertex: Option<&CheckedStage>,
        constants: &[(String, f64)],
        grid: Option<ShadowGrid<'_>>,
    ) -> Result<Self, String> {
        let material = grid.as_ref().map(|g| g.material);
        let text = shader_source(vertex, &settings.solid_materials, material);
        let module = naga::front::wgsl::parse_str(&text)
            .map_err(|e| format!("shadow pass: {}", e.emit_to_string(&text)))?;
        let mut constants = super::native_post::present_constants(&module, constants);
        constants.extend([
            ("horizon_terrain_SPRITE_ARRAYS", shape.sprites[0]),
            ("horizon_terrain_SPRITE_SPLIT", shape.sprites[1]),
        ]);
        if let Some(g) = &grid {
            constants.push(("horizon_displace_distance", f64::from(g.distance)));
            constants.push(("horizon_displace_steps", f64::from(u8::from(g.steps))));
        }
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Native shader pack shadow"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Native shader pack shadow frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<FrameInputs>() as u64),
                    },
                    count: None,
                },
                super::native_post::materials_entry(5),
            ]
            .into_iter()
            .chain(grid.is_some().then(|| super::native_post::array_entry(12)))
            .chain(super::native_post::pack_texture_entries(
                frame.textures.len(),
            ))
            .collect::<Vec<_>>(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Native shader pack shadow frame"),
            layout: &frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: frame.materials.as_entire_binding(),
                },
            ]
            .into_iter()
            .chain(grid.is_some().then(|| wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(&frame.pbr[0]),
            }))
            .chain(super::native_post::pack_texture_group_entries(
                &frame.textures,
            ))
            .collect::<Vec<_>>(),
        });
        let layout = |flat: Option<&wgpu::BindGroupLayout>| {
            let mut groups: [Option<&wgpu::BindGroupLayout>; GROUP as usize + 1] =
                Default::default();
            groups[0] = Some(shape.globals);
            groups[1] = flat;
            groups[GROUP as usize] = Some(&frame_layout);
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Native shader pack shadow"),
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
        let color_target = [Some(wgpu::ColorTargetState {
            format: SHADOW_COLOR,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let pipeline = |layout: &wgpu::PipelineLayout,
                        vertex: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        translucent: bool| {
            let options = wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Native shader pack shadow"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vertex),
                    compilation_options: options.clone(),
                    buffers,
                },
                // Both sides: the quads that face the light are drawn (`facing_runs`), and
                // plants' crossed quads cast from either side.
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 1.5,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(if translucent {
                        "horizon_shadow_fs_translucent"
                    } else {
                        "horizon_shadow_fs"
                    }),
                    compilation_options: options,
                    targets: if translucent { &color_target } else { &[] },
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let instanced = layout(None);
        let solid = pipeline(
            &instanced,
            "horizon_shadow_vs",
            &[Some(vertices.clone()), Some(instances.clone())],
            false,
        );
        let translucent = pipeline(
            &instanced,
            "horizon_shadow_vs",
            &[Some(vertices.clone()), Some(instances)],
            true,
        );
        let flat_layout = shape.flat.map(|group| layout(Some(group)));
        let flat_pipeline = |translucent| {
            flat_layout.as_ref().map(|layout| {
                pipeline(
                    layout,
                    "horizon_shadow_vs_flat",
                    &[Some(vertices.clone())],
                    translucent,
                )
            })
        };
        let flat = flat_pipeline(false);
        let flat_translucent = flat_pipeline(true);
        let displaced = grid.as_ref().map(|g| {
            let layout = layout(Some(g.layout));
            let options = wgpu::PipelineCompilationOptions {
                constants: &constants,
                ..Default::default()
            };
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Native shader pack shadow (displaced)"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("horizon_shadow_vs_displaced"),
                    compilation_options: options.clone(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: wgpu::DepthBiasState {
                        constant: 0,
                        slope_scale: 1.5,
                        clamp: 0.0,
                    },
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("horizon_shadow_fs_displaced"),
                    compilation_options: options,
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let entity = entity_pipeline(device, shape.entities, &frame_layout);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(format!("shadow pass: {e}"));
        }
        Ok(Self {
            camera: camera(settings),
            solid,
            flat,
            translucent,
            flat_translucent,
            entity,
            group,
            displaced,
        })
    }

    /// Draws the first `quads` of the renderer's entity buffer (`EntityBatch`'s opaque
    /// ones) and the `shadow_only` (first vertex, quads) run, in a shadow pass that has the
    /// frame's group bound.
    pub fn draw_entities(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        group: &wgpu::BindGroup,
        vertices: &wgpu::Buffer,
        indices: &wgpu::Buffer,
        quads: u32,
        shadow_only: (u32, u32),
    ) {
        if quads == 0 && shadow_only.1 == 0 {
            return;
        }
        pass.set_pipeline(&self.entity);
        pass.set_bind_group(0, group, &[]);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        if quads > 0 {
            pass.draw_indexed(0..quads * 6, 0, 0..1);
        }
        if shadow_only.1 > 0 {
            pass.draw_indexed(0..shadow_only.1 * 6, shadow_only.0 as i32, 0..1);
        }
    }

    /// What the shadow camera sees this frame, for the renderer's shadow lists.
    pub fn view(&self, frame: &super::Frame) -> ShadowView {
        ShadowView {
            cull: super::pack_uniforms::shadow_culling(frame, &self.camera),
            light: super::pack_uniforms::shadow_light(frame, &self.camera),
            // The light's box culls: on the ground it reaches further than `distance`
            // along the sun's azimuth while the sun is low.
            distance: None,
            unculled: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shadow_shader_validates() {
        let text = shader_source(None, &[2, 7], None);
        let module = naga::front::wgsl::parse_str(&text).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    }

    #[test]
    fn the_shadow_grid_validates_with_a_pack() {
        use horizon_shader_api::{MANIFEST, Pack, check};
        let manifest = "schema = 1
id = \"t\"
name = \"T\"
version = \"1\"
api = \"0.5\"
            [material]
file = \"material.wgsl\"
[displacement]
[shadow]
";
        let material = "fn displacement(input: DisplacementInput) -> f32 {
                    return 1.0 - input.height;
}
            fn material(input: MaterialInput) -> Surface {
                    return Surface(input.albedo);
}
";
        let files = [(MANIFEST, manifest), ("material.wgsl", material)]
            .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
            .into();
        let report = check(&Pack::from_files(files).unwrap()).unwrap();
        let text = shader_source(None, &[], Some(&report.stages[0]));
        let module = naga::front::wgsl::parse_str(&text)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
        assert!(
            module
                .entry_points
                .iter()
                .any(|e| e.name == "horizon_shadow_vs_displaced")
        );
    }

    #[test]
    fn the_distortion_brings_the_centre_texel_to_a_32nd_of_a_block() {
        // 2048 texels over 256 blocks: an 8th of a block each, magnified 4 times.
        assert!((distortion(128.0, 2048) - 0.75).abs() < 1e-6);
        // Already that fine: no distortion.
        assert_eq!(distortion(32.0, 2048), 0.0);
        assert_eq!(distortion(10_000.0, 1024), 0.95);
    }
}
