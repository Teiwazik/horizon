//! The post stage of Horizon's own shader packs (docs/shaders.md): each `[[post]]` pass is a
//! full-screen pipeline built from its checked naga module (`horizon_shader_api::check`), with
//! the pack's options as pipeline constants. A pass reads bind group 3: the frame, the world's
//! colour and depth, the previous pass's output and two samplers. Passes before the last write
//! half-float targets of the world's size, so values over 1 carry to the next; the last draws
//! onto the screen. Every stage of a pack reads the same `frame`, written once a frame
//! (`NativeFrame`).

use glam::{Mat4, Vec3};
use horizon_shader_api::{Frame as FrameInputs, check::CheckedStage, stages::GROUP};

use super::Frame;

const INTERMEDIATE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The world's colour, depth and surface views a set of bind groups was made for.
type WorldViews = (wgpu::TextureView, wgpu::TextureView, wgpu::TextureView);

pub struct NativePost {
    passes: Vec<(String, wgpu::RenderPipeline)>,
    layout: wgpu::BindGroupLayout,
    frame: wgpu::Buffer,
    /// The frame's shadow map and its comparison sampler (bindings 6 and 7).
    /// The shadow map, its sampler, the translucent map and its colour.
    shadow: (
        wgpu::TextureView,
        wgpu::Sampler,
        wgpu::TextureView,
        wgpu::TextureView,
    ),
    /// The pack's textures.
    textures: Vec<(wgpu::TextureView, wgpu::Sampler)>,
    /// Last frame's input to the last pass (binding 8), and its size.
    history: Option<([u32; 2], wgpu::Texture)>,
    history_blank: wgpu::Texture,
    /// Whether `history` holds a frame of this size.
    history_valid: bool,
    linear: wgpu::Sampler,
    nearest: wgpu::Sampler,
    /// Two targets of the world's size, written in turn by the chain's passes (none without
    /// one besides the last).
    targets: Option<([u32; 2], [wgpu::TextureView; 2])>,
    /// What each pass writes and reads (`[[post]] output`).
    plan: Vec<PassPlan>,
    /// The pack's targets' scales (`Manifest::post_targets`), and their views for the world's
    /// size.
    scales: Vec<f32>,
    named: Option<([u32; 2], Vec<wgpu::TextureView>)>,
    /// One bind group per pass, for these world views.
    groups: Option<(WorldViews, bool, Vec<wgpu::BindGroup>)>,
    /// The screen's colour format, the world target's too.
    format: wgpu::TextureFormat,
    /// Pass-dump state, when `HORIZON_PASS_DUMP` names a directory.
    dump: Option<Dump>,
    /// Whether the passes trace rays: group 0 has the terrain's sprites, and the frame's group
    /// the scene.
    rays: bool,
}

/// The post prelude's rays where the device has them: the material's, on the terrain's sprites.
const POST_RAY_SPRITES: &str = "
@group(0) @binding(1) var horizon_terrain_sprites: texture_2d_array<f32>;
@group(0) @binding(2) var horizon_terrain_sprite_sampler: sampler;
@group(0) @binding(5) var horizon_terrain_sprites_1: texture_2d_array<f32>;
@group(0) @binding(6) var horizon_terrain_sprites_2: texture_2d_array<f32>;
@group(0) @binding(7) var horizon_terrain_sprites_3: texture_2d_array<f32>;
@group(0) @binding(8) var horizon_terrain_sprites_4: texture_2d_array<f32>;
@group(0) @binding(9) var horizon_terrain_sprites_5: texture_2d_array<f32>;
@group(0) @binding(10) var horizon_terrain_sprites_6: texture_2d_array<f32>;
@group(0) @binding(11) var horizon_terrain_sprites_7: texture_2d_array<f32>;
@group(3) @binding(12) var horizon_ray_albedo: texture_2d<f32>;
fn horizon_ray_albedo_at(texel: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(horizon_ray_albedo));
    return textureLoad(horizon_ray_albedo, clamp(texel, vec2<i32>(0), size - 1), 0);
}
override horizon_terrain_SPRITE_SPLIT: u32 = 2048u;
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
";

/// Where a post pass draws.
#[derive(Clone, Copy, PartialEq)]
enum Writes {
    /// One of the chain's two targets.
    Chain(usize),
    /// A pack target (`[[post]] output`).
    Target(usize),
    /// The screen: the last pass.
    Screen,
}

/// A post pass's target, and the chain's picture it reads as `previous` (the world's for
/// none).
#[derive(Clone, Copy)]
struct PassPlan {
    writes: Writes,
    reads: Option<usize>,
}

/// The passes' plan for `outputs` (each pass's `[[post]] output`, by target index): the chain's
/// passes write its targets in turn, each reading what the one before wrote; a pass with an
/// output writes that and reads the chain's picture, which goes on past it.
fn plan(outputs: &[Option<usize>]) -> Vec<PassPlan> {
    let mut current = None;
    outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            let reads = current;
            let writes = match output {
                _ if i + 1 == outputs.len() => Writes::Screen,
                Some(t) => Writes::Target(*t),
                None => {
                    let slot = usize::from(current == Some(0));
                    current = Some(slot);
                    Writes::Chain(slot)
                }
            };
            PassPlan { writes, reads }
        })
        .collect()
}

/// A pack target's size: `scale` of the world's `size`, at least a texel.
fn scaled(size: [u32; 2], scale: f32) -> [u32; 2] {
    size.map(|s| ((s as f32 * scale).ceil() as u32).max(1))
}

/// The `frame` every stage of a pack reads (group 3, binding 0), written once a frame, and
/// what the material stage reads besides.
pub struct NativeFrame {
    /// This frame's TAA jitter in clip space and the upscale while the renderer's TAA
    /// rebuilds the picture after the post stage, `None` without it.
    pub engine_taa: Option<[f32; 3]>,
    /// The ray tracing prototype's scene (binding 14), with `HORIZON_RT`.
    pub rays: Option<super::native_rt::RayScene>,
    pub(super) buffer: wgpu::Buffer,
    /// Group 3 of the stages that read the frame and the reflection buffer (the sky and
    /// the material).
    pub layout: wgpu::BindGroupLayout,
    pub group: wgpu::BindGroup,
    /// The reflection texture's view `group` was made for, and a blank 1×1 fallback for
    /// frames without a reflection pass.
    reflection: Option<wgpu::TextureView>,
    blank: wgpu::TextureView,
    /// The shadow map the stages read (bindings 3 and 4), and the camera that draws it when
    /// the pack asks for one.
    pub shadow_map: super::native_shadow::ShadowMap,
    shadow_camera: Option<horizon_shaders::Shadow>,
    /// The camera and the shadow camera's view-projection the shadow map's terrain was last
    /// drawn with (`write`); none until it is drawn, or after a frame without it.
    shadow_terrain: Option<(glam::DVec3, Mat4)>,
    /// Whether this frame draws the shadow map's terrain again (`write`).
    pub shadow_refresh: bool,
    /// Whether the water mirror is drawn for the pack (`NativePack::reflection`): without
    /// it `has_reflection` stays 0.
    pub reflects: bool,
    /// Each sprite layer's `[[materials]]` id (binding 5), read by the terrain's stages.
    pub materials: wgpu::Buffer,
    /// The pack's `[[textures]]` and their samplers, bound in every stage.
    pub textures: Vec<(wgpu::TextureView, wgpu::Sampler)>,
    /// The world's depth before translucent terrain (binding 6), copied each frame.
    pub opaque_depth: wgpu::Texture,
    /// The world's colour before translucent terrain (binding 11), copied with it.
    pub opaque_color: wgpu::Texture,
    /// The resource packs' LabPBR maps (bindings 12 and 13: normals, specular), a blank
    /// 1×1 array each without them.
    pub pbr: [wgpu::TextureView; 2],
    /// The solid terrain's second target, the post stage's `scene_surface`: the world's size
    /// while the pack has a material stage, else a blank 1×1.
    pub surface: wgpu::Texture,
    pub surface_view: wgpu::TextureView,
    /// The solid terrain's third target, for light the post stage adds.
    albedo: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    /// The colour of block light around the camera (bindings 9 and 10).
    pub block_light: super::block_light::BlockLight,
    /// Last frame's view-projection, camera and time.
    last: Option<([[f32; 4]; 4], glam::DVec3, std::time::Instant)>,
    /// A post stage must use black history after a long gap until it has rendered a frame.
    history_reset: bool,
    /// `wetness`, `biome_rain` and `biome_snow`, eased from frame to frame.
    weather: [f32; 3],
    /// The eased eye light (sky, block) and being in water and lava (`write`).
    eye: [f32; 4],
    start: std::time::Instant,
    frames: u32,
    /// `frame.wetness`, and when it was last written.
    wetness: (f32, Option<std::time::Instant>),
}

/// A depth texture of `size` the world's depth is copied into.
fn depth_copy(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    copy_target(
        device,
        "Native shader pack opaque depth",
        size,
        wgpu::TextureFormat::Depth32Float,
    )
}

/// A colour texture of `size` and `format` the world's colour is copied into.
fn color_copy(device: &wgpu::Device, size: [u32; 2], format: wgpu::TextureFormat) -> wgpu::Texture {
    copy_target(device, "Native shader pack opaque colour", size, format)
}

fn copy_target(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// The uniform `frame` at binding 0, read by `visibility`.
fn frame_entry(visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size_of::<FrameInputs>() as u64),
        },
        count: None,
    }
}

/// The water mirror's targets at `size` of the world: half of it, rounded up.
pub fn reflection_size(size: [u32; 2]) -> [u32; 2] {
    size.map(|s| s.div_ceil(2).max(1))
}

/// Seconds between frames past which the last one is forgotten (`NativeFrame::write`).
const MAX_FRAME_GAP: f32 = 0.5;

/// `value` moved toward `target` over `dt` seconds by a half-life in seconds; a NaN `value`
/// (the first frame) starts at `target`.
fn eased(value: f32, target: f32, half_life: f32, dt: f32) -> f32 {
    if value.is_nan() {
        target
    } else {
        target + (value - target) * 0.5f32.powf(dt / half_life)
    }
}

impl NativeFrame {
    /// The frame of a pack with the `[shadow]` settings `shadow`, if any.
    pub fn new(
        device: &wgpu::Device,
        shadow: Option<&horizon_shader_api::manifest::Shadow>,
        materials: &[u32],
        textures: Vec<(wgpu::TextureView, wgpu::Sampler)>,
        lights: super::block_light::LightTable,
        pbr: Option<&[wgpu::Texture; 2]>,
        rays: bool,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Native shader pack frame"),
            size: size_of::<FrameInputs>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let mut entries = vec![
            frame_entry(wgpu::ShaderStages::VERTEX_FRAGMENT),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            texture_entry(3, wgpu::TextureSampleType::Depth),
            comparison_entry(4),
            materials_entry(5),
            texture_entry(6, wgpu::TextureSampleType::Depth),
            texture_entry(7, wgpu::TextureSampleType::Depth),
            texture_entry(8, float),
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: float,
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            },
            texture_entry(11, float),
            array_entry(12),
            array_entry(13),
        ];
        entries.extend(pack_texture_entries(textures.len()));
        // Not while the pack's `[rays] option` is off: no scene, and `ray_tracing()` false.
        let rays = rays.then(|| super::native_rt::RayScene::new(device)).flatten();
        let mut entries = entries.to_vec();
        if rays.is_some() {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            });
            for binding in [15, 60, 61, 62] {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Native shader pack frame"),
            entries: &entries,
        });
        let blank = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Native shader pack reflection blank"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let shadow_map = super::native_shadow::ShadowMap::new(device, shadow);
        let materials = materials_buffer(device, materials);
        let opaque_depth = depth_copy(device, [1, 1]);
        let opaque_color = color_copy(device, [1, 1], wgpu::TextureFormat::Rgba8Unorm);
        let array_view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let pbr = match pbr {
            Some(maps) => maps.each_ref().map(array_view),
            None => {
                // Two layers: one would make a 2D texture on the OpenGL backend.
                let blank = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Native shader pack LabPBR (none)"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 2,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                [array_view(&blank), array_view(&blank)]
            }
        };
        let surface = surface_target(device, [1, 1]);
        let surface_view = surface.create_view(&Default::default());
        let albedo = albedo_target(device, [1, 1]);
        let albedo_view = albedo.create_view(&Default::default());
        let block_light = super::block_light::BlockLight::new(device, lights);
        let group = Self::group(
            device,
            &layout,
            &buffer,
            &blank,
            (
                &shadow_map,
                &materials,
                &textures,
                (&opaque_depth, &opaque_color),
                (&block_light, &pbr, rays.as_ref()),
            ),
        );
        Self {
            engine_taa: None,
            rays,
            buffer,
            layout,
            group,
            reflection: None,
            blank,
            shadow_map,
            shadow_camera: shadow.map(super::native_shadow::camera),
            shadow_terrain: None,
            shadow_refresh: true,
            reflects: true,
            materials,
            textures,
            opaque_depth,
            opaque_color,
            pbr,
            surface,
            surface_view,
            albedo,
            albedo_view,
            block_light,
            last: None,
            weather: [0.0; 3],
            eye: [f32::NAN; 4],
            start: std::time::Instant::now(),
            frames: 0,
            history_reset: true,
            wetness: (0.0, None),
        }
    }

    fn group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &wgpu::Buffer,
        view: &wgpu::TextureView,
        (shadow, materials, textures, opaque_depth, block_light): (
            &super::native_shadow::ShadowMap,
            &wgpu::Buffer,
            &[(wgpu::TextureView, wgpu::Sampler)],
            (&wgpu::Texture, &wgpu::Texture),
            (
                &super::block_light::BlockLight,
                &[wgpu::TextureView; 2],
                Option<&super::native_rt::RayScene>,
            ),
        ),
    ) -> wgpu::BindGroup {
        let (opaque_depth, opaque_color) = opaque_depth;
        let (block_light, pbr, rays) = block_light;
        let opaque_view = opaque_depth.create_view(&Default::default());
        let opaque_color_view = opaque_color.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Native shader pack reflection sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&shadow.view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&shadow.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: materials.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&opaque_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&shadow.translucent_view),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&shadow.color_view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&block_light.view),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: block_light.grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(&opaque_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(&pbr[0]),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(&pbr[1]),
            },
        ];
        if let Some(rays) = rays {
            entries.push(wgpu::BindGroupEntry {
                binding: 14,
                resource: wgpu::BindingResource::AccelerationStructure(&rays.tlas),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 15,
                resource: rays.data.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 60,
                resource: rays.lights.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 61,
                resource: rays.light_cells.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 62,
                resource: rays.blue_noise.as_entire_binding(),
            });
        }
        entries.extend(pack_texture_group_entries(textures));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Native shader pack frame"),
            layout,
            entries: &entries,
        })
    }

    /// Binds `view` as this frame's `reflection` texture (a `None` restores the blank
    /// fallback). A frame's reflection pass draws before the stages read it.
    pub fn set_reflection(&mut self, device: &wgpu::Device, view: Option<&wgpu::TextureView>) {
        let view = view.cloned().unwrap_or_else(|| self.blank.clone());
        if self.reflection.as_ref().is_some_and(|bound| bound == &view) {
            return;
        }
        self.group = Self::group(
            device,
            &self.layout,
            &self.buffer,
            &view,
            (
                &self.shadow_map,
                &self.materials,
                &self.textures,
                (&self.opaque_depth, &self.opaque_color),
                (&self.block_light, &self.pbr, self.rays.as_ref()),
            ),
        );
        self.reflection = Some(view);
    }

    /// Keeps the opaque depth and colour copies the size of the world's depth `depth` and
    /// in the format of its colour `color`, when that can be copied (rebinding the stages'
    /// group when they change).
    pub fn fit_opaque_depth(
        &mut self,
        device: &wgpu::Device,
        depth: &wgpu::Texture,
        color: Option<&wgpu::Texture>,
    ) {
        let size = [depth.width(), depth.height()];
        if [self.surface.width(), self.surface.height()] != size {
            self.surface = surface_target(device, size);
            self.surface_view = self.surface.create_view(&Default::default());
            self.albedo = albedo_target(device, size);
            self.albedo_view = self.albedo.create_view(&Default::default());
        }
        let (color_size, format) = match color {
            Some(c) => ([c.width(), c.height()], c.format()),
            None => ([1, 1], self.opaque_color.format()),
        };
        let color_fits = [self.opaque_color.width(), self.opaque_color.height()] == color_size
            && self.opaque_color.format() == format;
        if [self.opaque_depth.width(), self.opaque_depth.height()] == size && color_fits {
            return;
        }
        if !color_fits {
            self.opaque_color = color_copy(device, color_size, format);
        }
        self.opaque_depth = depth_copy(device, size);
        let view = self
            .reflection
            .clone()
            .unwrap_or_else(|| self.blank.clone());
        self.group = Self::group(
            device,
            &self.layout,
            &self.buffer,
            &view,
            (
                &self.shadow_map,
                &self.materials,
                &self.textures,
                (&self.opaque_depth, &self.opaque_color),
                (&self.block_light, &self.pbr, self.rays.as_ref()),
            ),
        );
    }

    /// After a frame that drew no shadow map: the next draws its terrain again.
    pub fn forget_shadow(&mut self) {
        self.shadow_terrain = None;
    }

    /// Writes this frame's inputs, for a world of `size`.
    pub fn write(&mut self, queue: &wgpu::Queue, frame: &Frame, size: [u32; 2]) {
        let seconds = self.start.elapsed().as_secs_f32() % 3600.0;
        let mut inputs = frame_inputs(
            frame,
            size,
            seconds,
            self.frames,
            self.shadow_camera.as_ref(),
        );
        // The shadow map's terrain is drawn every `SHADOW_TERRAIN_FRAMES`th frame (at once when
        // the camera jumps); in between it keeps the light and the shadow camera it was drawn
        // with, moved by the camera since, and the frame's entities are drawn over it.
        if self.shadow_camera.is_some() {
            self.shadow_refresh = self
                .frames
                .is_multiple_of(super::native_shadow::SHADOW_TERRAIN_FRAMES)
                || self
                    .shadow_terrain
                    .is_none_or(|(camera, _)| camera.distance(frame.camera) > 4.0);
            if self.shadow_refresh {
                self.shadow_terrain = Some((
                    frame.camera,
                    Mat4::from_cols_array_2d(&inputs.shadow_view_proj),
                ));
            }
            if let Some((camera, view_proj)) = self.shadow_terrain {
                let moved = (frame.camera - camera).as_vec3();
                inputs.shadow_view_proj =
                    (view_proj * Mat4::from_translation(moved)).to_cols_array_2d();
            }
        }
        if !self.reflects {
            inputs.has_reflection = 0;
        }
        // Drawn with the jitter: the post stage's positions from depth and its reprojection
        // then match what was drawn.
        if let Some([x, y, upscale]) = self.engine_taa {
            let projection = Mat4::from_translation(Vec3::new(x, y, 0.0))
                * Mat4::from_cols_array_2d(&inputs.projection);
            inputs.projection = projection.to_cols_array_2d();
            inputs.projection_inverse = projection.inverse().to_cols_array_2d();
            inputs.engine_taa = [x, y, 1.0, upscale];
        }
        let view_proj = (Mat4::from_cols_array_2d(&inputs.projection)
            * Mat4::from_cols_array_2d(&inputs.view))
        .to_cols_array_2d();
        let now = std::time::Instant::now();
        inputs.previous_view_projection = view_proj;
        // `write` skips title frames, so a gap means a pause or another world: the frame
        // before says nothing about this one, and the eased values start over.
        match self.last {
            Some((previous, camera, time)) if (now - time).as_secs_f32() < MAX_FRAME_GAP => {
                inputs.previous_view_projection = previous;
                inputs.camera_motion = (frame.camera - camera).as_vec3().to_array();
                inputs.delta_time = (now - time).as_secs_f32();
            }
            _ => {
                self.eye = [f32::NAN; 4];
                self.history_reset = true;
            }
        }
        // Eased toward this frame's weather by half-lives in seconds: rising, falling.
        let dt = inputs.delta_time;
        let precipitation = frame.pack.biome_precipitation;
        let targets = [
            (frame.pack.rain, [30.0, 10.0]),
            (f32::from(u8::from(precipitation == 1)), [20.0, 10.0]),
            (f32::from(u8::from(precipitation == 2)), [20.0, 10.0]),
        ];
        for (value, (target, half_lives)) in self.weather.iter_mut().zip(targets) {
            let half_life = if target > *value {
                half_lives[0]
            } else {
                half_lives[1]
            };
            *value = target + (*value - target) * 0.5f32.powf(dt / half_life);
        }
        [inputs.wetness, inputs.biome_rain, inputs.biome_snow] = self.weather;
        // The eye's light and medium, eased the same way (from where they are at first).
        let targets = [
            (inputs.eye_light[0], 2.0),
            (inputs.eye_light[1], 2.0),
            (f32::from(u8::from(inputs.eye_in == 1)), 0.25),
            (f32::from(u8::from(inputs.eye_in == 2)), 0.25),
        ];
        for (value, (target, half_life)) in self.eye.iter_mut().zip(targets) {
            *value = eased(*value, target, half_life, dt);
        }
        let [block, sky, water, lava] = self.eye;
        inputs.eye_light_smooth = [block, sky];
        inputs.in_water = water;
        inputs.in_lava = lava;
        self.last = Some((view_proj, frame.camera, now));
        self.frames = self.frames.wrapping_add(1);
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&inputs));
    }

    /// Takes the post stage's pending history reset. It stays pending if post rendering is
    /// skipped for a frame, so the first frame it does render still sees black history.
    pub(super) fn take_history_reset(&mut self) -> bool {
        let reset = self.history_reset;
        self.history_reset = false;
        reset
    }
}

/// The option values of `constants` that `module` declares as overrides (wgpu refuses others).
pub fn present_constants<'a>(
    module: &naga::Module,
    constants: &'a [(String, f64)],
) -> Vec<(&'a str, f64)> {
    constants
        .iter()
        .filter(|(id, _)| {
            module
                .overrides
                .iter()
                .any(|(_, o)| o.name.as_ref() == Some(id))
        })
        .map(|(id, v)| (id.as_str(), *v))
        .collect()
}

/// A filterable 2D array texture at `binding`, read in the fragment stage.
pub(super) fn array_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        // `[displacement]`'s grid reads the heights where it moves its corners.
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// The solid terrain's surface target of `size` (`SURFACE_FORMAT`), drawn and read.
fn albedo_target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Native shader pack albedo"),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::native_material::ALBEDO_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn surface_target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Native shader pack surface"),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: super::native_material::SURFACE_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

/// The pack's `[[textures]]` (`check::Image`s), uploaded with samplers as the manifest says.
pub fn pack_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    report: &horizon_shader_api::Report,
) -> Vec<(wgpu::TextureView, wgpu::Sampler)> {
    report
        .manifest
        .textures
        .iter()
        .zip(&report.textures)
        .map(|(t, image)| {
            let texture = wgpu::util::DeviceExt::create_texture_with_data(
                device,
                queue,
                &wgpu::TextureDescriptor {
                    label: Some(&t.id),
                    size: wgpu::Extent3d {
                        width: image.width,
                        height: image.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &image.rgba,
            );
            let filter = if t.linear {
                wgpu::FilterMode::Linear
            } else {
                wgpu::FilterMode::Nearest
            };
            let address = if t.repeat {
                wgpu::AddressMode::Repeat
            } else {
                wgpu::AddressMode::ClampToEdge
            };
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(&t.id),
                address_mode_u: address,
                address_mode_v: address,
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            });
            (texture.create_view(&Default::default()), sampler)
        })
        .collect()
}

/// Layout entries of `count` pack textures and their samplers.
pub(super) fn pack_texture_entries(count: usize) -> Vec<wgpu::BindGroupLayoutEntry> {
    use horizon_shader_api::stages::{TEXTURE_BINDING, TEXTURE_SAMPLER_BINDING};
    (0..count as u32)
        .flat_map(|i| {
            [
                wgpu::BindGroupLayoutEntry {
                    binding: TEXTURE_BINDING + i,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: TEXTURE_SAMPLER_BINDING + i,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        })
        .collect()
}

/// Group entries of the pack's textures, as `pack_texture_entries` lays them out.
pub(super) fn pack_texture_group_entries(
    textures: &[(wgpu::TextureView, wgpu::Sampler)],
) -> Vec<wgpu::BindGroupEntry<'_>> {
    use horizon_shader_api::stages::{TEXTURE_BINDING, TEXTURE_SAMPLER_BINDING};
    textures
        .iter()
        .enumerate()
        .flat_map(|(i, (view, sampler))| {
            [
                wgpu::BindGroupEntry {
                    binding: TEXTURE_BINDING + i as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: TEXTURE_SAMPLER_BINDING + i as u32,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ]
        })
        .collect()
}

/// The material ids, read by the terrain's vertex and fragment stages.
pub(super) fn materials_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The material ids by sprite layer (one 0 when there are none).
fn materials_buffer(device: &wgpu::Device, materials: &[u32]) -> wgpu::Buffer {
    let zero = [0u32];
    let ids = if materials.is_empty() {
        &zero[..]
    } else {
        materials
    };
    wgpu::util::DeviceExt::create_buffer_init(
        device,
        &wgpu::util::BufferInitDescriptor {
            label: Some("Native shader pack material ids"),
            contents: bytemuck::cast_slice(ids),
            usage: wgpu::BufferUsages::STORAGE,
        },
    )
}

fn comparison_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

impl NativePost {
    /// Pipelines for `stages` (the pack's post passes, in order) with the option values
    /// `constants`, drawing onto a screen of `format`. Errors name the pass wgpu refused.
    pub fn new(
        device: &wgpu::Device,
        stages: Vec<CheckedStage>,
        manifest: &horizon_shader_api::Manifest,
        constants: &[(String, f64)],
        format: wgpu::TextureFormat,
        frame: &NativeFrame,
        terrain: Option<(&wgpu::BindGroupLayout, [f64; 2])>,
    ) -> Result<Self, String> {
        let targets = manifest.post_targets();
        let outputs: Vec<Option<usize>> = manifest
            .post
            .iter()
            .map(|p| {
                let name = p.output.as_deref()?;
                targets.iter().position(|t| t.0 == name)
            })
            .collect();
        let plan = plan(&outputs);
        let float = wgpu::TextureSampleType::Float { filterable: true };
        let mut entries = vec![
            frame_entry(wgpu::ShaderStages::FRAGMENT),
            texture_entry(1, float),
            texture_entry(2, wgpu::TextureSampleType::Depth),
            texture_entry(3, float),
            sampler_entry(4),
            sampler_entry(5),
            texture_entry(6, wgpu::TextureSampleType::Depth),
            comparison_entry(7),
            texture_entry(8, float),
            texture_entry(9, wgpu::TextureSampleType::Depth),
            texture_entry(10, float),
            texture_entry(11, wgpu::TextureSampleType::Float { filterable: false }),
        ];
        entries.extend(pack_texture_entries(frame.textures.len()));
        let rays = terrain.filter(|_| frame.rays.is_some());
        if rays.is_some() {
            entries.push(texture_entry(
                12,
                wgpu::TextureSampleType::Float { filterable: false },
            ));
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 14,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            });
            for binding in [15, 60, 61, 62] {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
        }
        entries
            .extend((0..targets.len() as u32).map(|j| {
                texture_entry(horizon_shader_api::stages::POST_TARGET_BINDING + j, float)
            }));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Native shader pack post"),
            entries: &entries,
        });
        let mut groups: [Option<&wgpu::BindGroupLayout>; GROUP as usize + 1] = Default::default();
        groups[GROUP as usize] = Some(&layout);
        groups[0] = rays.map(|r| r.0);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Native shader pack post"),
            bind_group_layouts: &groups,
            ..Default::default()
        });
        let count = stages.len();
        // Each pass parses, validates and compiles alone: build them on a few threads at once,
        // which takes the wait for options to apply from the sum of the passes to about a
        // fraction of it (wgpu's device takes calls from any thread).
        let build = |i: usize, stage: CheckedStage| -> Result<(String, wgpu::RenderPipeline), String> {
            let traced = rays
                .map(|_| {
                    let text = format!(
                        "enable wgpu_ray_query;\n{}{POST_RAY_SPRITES}{}{}",
                        stage.source,
                        super::native_material::RAYS,
                        horizon_shader_api::stages::POST_ENTRY_POINTS,
                    );
                    naga::front::wgsl::parse_str(&text)
                        .map_err(|e| format!("{}: {}", stage.file, e.emit_to_string(&text)))
                })
                .transpose()?;
            let ir = traced.unwrap_or(stage.module);
            let mut present = present_constants(&ir, constants);
            if let Some((_, sprites)) = rays {
                present.push(("horizon_terrain_SPRITE_SPLIT", sprites[1]));
            }
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&stage.label),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(ir)),
            });
            let options = wgpu::PipelineCompilationOptions {
                constants: &present,
                ..Default::default()
            };
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&stage.label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("horizon_vs_post"),
                    compilation_options: options.clone(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("horizon_fs_post"),
                    compilation_options: options,
                    targets: &[Some(wgpu::ColorTargetState {
                        format: if plan[i].writes == Writes::Screen {
                            format
                        } else {
                            INTERMEDIATE
                        },
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(format!("{}: {e}", stage.file));
            }
            Ok((stage.label, pipeline))
        };
        let threads = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .clamp(1, 8);
        let mut indexed: Vec<(usize, CheckedStage)> = stages.into_iter().enumerate().collect();
        let mut slots: Vec<Vec<(usize, CheckedStage)>> = (0..threads).map(|_| Vec::new()).collect();
        // Round robin, so the heavier ray-traced passes, which come together, spread out.
        for (n, item) in indexed.drain(..).enumerate() {
            slots[n % threads].push(item);
        }
        let mut built: Vec<(usize, Result<(String, wgpu::RenderPipeline), String>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = slots
                    .into_iter()
                    .map(|slot| {
                        let build = &build;
                        scope.spawn(move || {
                            slot.into_iter()
                                .map(|(i, stage)| (i, build(i, stage)))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap_or_default())
                    .collect()
            });
        built.sort_by_key(|(i, _)| *i);
        if built.len() != count {
            return Err("a post pass's build stopped".to_owned());
        }
        let mut passes = Vec::with_capacity(count);
        for (_, pass) in built {
            passes.push(pass?);
        }
        let sampler = |label, filter| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        Ok(Self {
            passes,
            layout,
            frame: frame.buffer.clone(),
            shadow: (
                frame.shadow_map.view.clone(),
                frame.shadow_map.sampler.clone(),
                frame.shadow_map.translucent_view.clone(),
                frame.shadow_map.color_view.clone(),
            ),
            textures: frame.textures.clone(),
            history: None,
            history_blank: intermediate(device, "Native shader pack history blank", [1, 1]),
            linear: sampler("Native shader pack linear", wgpu::FilterMode::Linear),
            nearest: sampler("Native shader pack nearest", wgpu::FilterMode::Nearest),
            targets: None,
            scales: targets.iter().map(|t| t.1).collect(),
            named: None,
            rays: rays.is_some(),
            plan,
            groups: None,
            history_valid: false,
            format,
            dump: std::env::var_os("HORIZON_PASS_DUMP").map(|dir| Dump {
                dir: dir.into(),
                pending: Vec::new(),
            }),
        })
    }

    /// The passes' labels, in order.
    pub fn labels(&self) -> Vec<&str> {
        self.passes.iter().map(|(l, _)| l.as_str()).collect()
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        world: (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView),
        size: [u32; 2],
        history_reset: bool,
        rays: Option<(
            &wgpu::BindGroup,
            &super::native_rt::RayScene,
            &wgpu::TextureView,
        )>,
    ) {
        let chain = self
            .plan
            .iter()
            .any(|p| matches!(p.writes, Writes::Chain(_)));
        if !self.scales.is_empty() && self.named.as_ref().is_none_or(|n| n.0 != size) {
            let views = self
                .scales
                .iter()
                .map(|&scale| {
                    let size = scaled(size, scale);
                    device
                        .create_texture(&wgpu::TextureDescriptor {
                            label: Some("Native shader pack target"),
                            size: wgpu::Extent3d {
                                width: size[0],
                                height: size[1],
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: INTERMEDIATE,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING
                                | wgpu::TextureUsages::COPY_SRC,
                            view_formats: &[],
                        })
                        .create_view(&Default::default())
                })
                .collect();
            self.named = Some((size, views));
            self.groups = None;
        }
        if chain && self.targets.as_ref().is_none_or(|t| t.0 != size) {
            let target = |label| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: INTERMEDIATE,
                        // COPY_SRC: the last pass's input is kept as next frame's `history`.
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                            | wgpu::TextureUsages::TEXTURE_BINDING
                            | wgpu::TextureUsages::COPY_SRC,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            self.targets = Some((
                size,
                [
                    target("Native shader pack A"),
                    target("Native shader pack B"),
                ],
            ));
            self.history = Some((
                size,
                intermediate(device, "Native shader pack history", size),
            ));
            self.history_valid = false;
            self.groups = None;
        }
        let blank_history = history_reset || !self.history_valid;
        if self
            .groups
            .as_ref()
            .is_some_and(|((color, depth, surface), blank, _)| {
                color == world.0
                    && depth == world.1
                    && surface == world.2
                    && *blank == blank_history
            })
        {
            return;
        }
        let history = if blank_history {
            &self.history_blank
        } else {
            self.history
                .as_ref()
                .map_or(&self.history_blank, |(_, t)| t)
        }
        .create_view(&Default::default());
        let blank = self.history_blank.create_view(&Default::default());
        let groups = (0..self.passes.len())
            .map(|i| {
                // The chain's picture so far: the world's before its first pass.
                let previous = match (&self.targets, self.plan[i].reads) {
                    (Some((_, t)), Some(slot)) => &t[slot],
                    _ => world.0,
                };
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.frame.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(world.0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(world.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(previous),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.nearest),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(&self.shadow.0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&self.shadow.1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(&history),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&self.shadow.2),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&self.shadow.3),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(world.2),
                    },
                ];
                entries.extend(pack_texture_group_entries(&self.textures));
                if let Some(rays) = rays.filter(|_| self.rays) {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::TextureView(rays.2),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 14,
                        resource: wgpu::BindingResource::AccelerationStructure(&rays.1.tlas),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 15,
                        resource: rays.1.data.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 60,
                        resource: rays.1.lights.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 61,
                        resource: rays.1.light_cells.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 62,
                        resource: rays.1.blue_noise.as_entire_binding(),
                    });
                }
                // The pack's targets, a blank one in place of the one this pass writes.
                if let Some((_, views)) = &self.named {
                    entries.extend(views.iter().enumerate().map(|(j, view)| {
                        let view = if self.plan[i].writes == Writes::Target(j) {
                            &blank
                        } else {
                            view
                        };
                        wgpu::BindGroupEntry {
                            binding: horizon_shader_api::stages::POST_TARGET_BINDING + j as u32,
                            resource: wgpu::BindingResource::TextureView(view),
                        }
                    }));
                }
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Native shader pack post"),
                    layout: &self.layout,
                    entries: &entries,
                })
            })
            .collect();
        self.groups = Some((
            (world.0.clone(), world.1.clone(), world.2.clone()),
            blank_history,
            groups,
        ));
    }

    /// Runs the passes over the world target `world` (colour, depth, surface) of `size`, the last onto
    /// `screen`, timed by `timing` when it times this frame.
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView),
        screen: &wgpu::TextureView,
        size: [u32; 2],
        history_reset: bool,
        mut timing: Option<&mut super::pack_timing::PassTiming>,
        rays: Option<(
            &wgpu::BindGroup,
            &super::native_rt::RayScene,
            &wgpu::TextureView,
        )>,
    ) {
        self.prepare(device, world, size, history_reset, rays);
        // Copies made last frame: write them before this frame's copies ask for more.
        if let Some(dump) = &mut self.dump {
            dump.flush(device);
        }
        // The world's passes before the post stage are timed as one.
        if let Some(t) = timing.as_deref_mut() {
            t.span(
                device,
                encoder,
                (world.0, self.format),
                "World (entities, water, the rest)",
            );
        }
        let Some((_, _, groups)) = &self.groups else {
            return;
        };
        let dump_now = self
            .dump
            .as_mut()
            .map(|d| d.capture(device, encoder, (world.0, world.1)))
            .unwrap_or(false);
        let last = self.passes.len() - 1;
        let history_after = self.plan[..last]
            .iter()
            .rposition(|p| matches!(p.writes, Writes::Chain(_)));
        for (i, ((label, pipeline), group)) in self.passes.iter().zip(groups).enumerate() {
            let view = match (self.plan[i].writes, &self.targets, &self.named) {
                (Writes::Chain(slot), Some((_, t)), _) => &t[slot],
                (Writes::Target(j), _, Some((_, n))) => &n[j],
                _ => screen,
            };
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &attachments,
                timestamp_writes: timing.as_deref_mut().and_then(|t| t.writes(label)),
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            if let Some(rays) = rays.filter(|_| self.rays) {
                pass.set_bind_group(0, rays.0, &[]);
            }
            pass.set_bind_group(GROUP, group, &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            if dump_now {
                let clean = |label: &str| label.replace(['/', '\\', ':'], "_");
                let name = if i + 1 < self.passes.len() {
                    format!("02_pass{i}_{}", clean(label))
                } else {
                    format!("03_final_pass{i}_{}", clean(label))
                };
                if let Some(dump) = &mut self.dump {
                    dump.copy(device, encoder, view, name);
                }
            }
        }
        // The chain's last picture becomes next frame's `history`, once every pass has read
        // this frame's.
        if let Some(i) = history_after
            && let Writes::Chain(slot) = self.plan[i].writes
            && let (Some((_, history)), Some((_, targets))) = (&self.history, &self.targets)
        {
            encoder.copy_texture_to_texture(
                targets[slot].texture().as_image_copy(),
                history.as_image_copy(),
                history.size(),
            );
            self.history_valid = true;
        }
    }
}

/// A pass target of `size`: half-float colour, readable and copyable.
fn intermediate(device: &wgpu::Device, label: &str, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: INTERMEDIATE,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// The pass dump, a profiling aid for "which pass draws this": with `HORIZON_PASS_DUMP=<dir>`
/// set, a `go` flag file in `<dir>` makes the next frame's post stage copy the world's colour
/// and depth and every pass's output into readback buffers; the following frame writes them
/// as PNGs and removes the flag. A `NaN` is magenta, other values clamp to [0, 1] and
/// gamma-encode (the depth: boosted, most of a perspective buffer is near 1).
struct Dump {
    dir: std::path::PathBuf,
    /// Copies enqueued this frame, written next frame: mapping needs the GPU to catch up.
    pending: Vec<Pending>,
}

/// One copied texture, with what writing it as a PNG takes.
struct Pending {
    name: String,
    buffer: wgpu::Buffer,
    /// The buffer's padded row, in bytes.
    row: u32,
    pixels: [u32; 2],
    format: DumpFormat,
}

/// How to read a copied texture's bytes.
enum DumpFormat {
    /// Rgba16Float: a pass target, or an HDR world colour.
    Half,
    /// Depth32Float.
    Depth,
    /// An 8-bit colour, `bgra`: swap the channels.
    Srgb { bgra: bool },
}

impl Dump {
    /// Copies the world's colour and depth, when the `go` flag asks for it and no copies
    /// are still awaiting their write; returns whether per-pass copies should follow
    /// (`run` enqueues one after each pass draws).
    fn capture(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        world: (&wgpu::TextureView, &wgpu::TextureView),
    ) -> bool {
        if !self.pending.is_empty() || !self.dir.join("go").exists() {
            return false;
        }
        self.copy(device, encoder, world.0, "00_world_color".into());
        self.copy(device, encoder, world.1, "01_world_depth".into());
        true
    }

    /// Enqueues the readback of `view` as `name`, if the texture allows copies.
    fn copy(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        name: String,
    ) {
        let texture = view.texture();
        if !texture.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            return;
        }
        let format = match texture.format() {
            wgpu::TextureFormat::Rgba16Float => DumpFormat::Half,
            wgpu::TextureFormat::Depth32Float => DumpFormat::Depth,
            f => DumpFormat::Srgb {
                bgra: matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
                ),
            },
        };
        let size = texture.size();
        let pixels = [size.width, size.height];
        let bytes_per_pixel = match format {
            DumpFormat::Half => 8,
            DumpFormat::Depth | DumpFormat::Srgb { .. } => 4,
        };
        let row =
            (pixels[0] * bytes_per_pixel).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Native shader pack pass dump"),
            size: u64::from(row) * u64::from(pixels[1]),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(pixels[1]),
                },
            },
            wgpu::Extent3d {
                width: pixels[0],
                height: pixels[1],
                depth_or_array_layers: 1,
            },
        );
        self.pending.push(Pending {
            name,
            buffer,
            row,
            pixels,
            format,
        });
    }

    /// Writes the copies made last frame as PNGs and removes the `go` flag, if any copy
    /// made it (the flag stays otherwise, to ask again).
    fn flush(&mut self, device: &wgpu::Device) {
        if self.pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.pending);
        let mut wrote = false;
        for p in pending {
            let (sender, receiver) = std::sync::mpsc::channel();
            p.buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = sender.send(r);
            });
            let ok = device.poll(wgpu::PollType::wait_indefinitely()).is_ok()
                && receiver.recv().ok().and_then(|r| r.ok()).is_some();
            if ok {
                let data = p.buffer.slice(..).get_mapped_range().unwrap();
                let rgba = dump_rgba(&data, &p);
                drop(data);
                p.buffer.unmap();
                if let Ok(file) = std::fs::File::create(self.dir.join(&p.name))
                    && let Ok(mut writer) = {
                        let mut encoder = png::Encoder::new(file, p.pixels[0], p.pixels[1]);
                        encoder.set_color(png::ColorType::Rgba);
                        encoder.set_depth(png::BitDepth::Eight);
                        encoder.write_header()
                    }
                    && writer.write_image_data(&rgba).is_ok()
                {
                    wrote = true;
                }
            }
        }
        if wrote {
            let _ = std::fs::remove_file(self.dir.join("go"));
        }
    }
}

/// An 8-bit RGBA copy of `data` (a `Pending`'s buffer), for a PNG.
fn dump_rgba(data: &[u8], pending: &Pending) -> Vec<u8> {
    let (w, h) = (pending.pixels[0] as usize, pending.pixels[1] as usize);
    let row = pending.row as usize;
    let mut out = vec![0u8; 4 * w * h];
    let mut put = |x: usize, y: usize, rgba: [u8; 4]| {
        out[(y * w + x) * 4..][..4].copy_from_slice(&rgba);
    };
    let nan = [255u8, 0, 255, 255];
    match pending.format {
        DumpFormat::Half => {
            for y in 0..h {
                for x in 0..w {
                    let src = &data[y * row + x * 8..][..8];
                    let mut channels = [0.0f32; 4];
                    let mut is_nan = false;
                    for (c, channel) in channels.iter_mut().enumerate() {
                        *channel = half([src[c * 2], src[c * 2 + 1]]);
                        is_nan |= channel.is_nan();
                    }
                    put(x, y, if is_nan { nan } else { channels.map(tone) });
                }
            }
        }
        DumpFormat::Depth => {
            for y in 0..h {
                for x in 0..w {
                    let v = f32::from_le_bytes(data[y * row + x * 4..][..4].try_into().unwrap());
                    put(
                        x,
                        y,
                        if v.is_nan() {
                            nan
                        } else {
                            let v = (1.0 - v.clamp(0.0, 1.0)).powf(0.25);
                            let g = (v * 255.0 + 0.5) as u8;
                            [g, g, g, 255]
                        },
                    );
                }
            }
        }
        DumpFormat::Srgb { bgra } => {
            for y in 0..h {
                for x in 0..w {
                    let src = &data[y * row + x * 4..][..4];
                    put(
                        x,
                        y,
                        if bgra {
                            [src[2], src[1], src[0], src[3]]
                        } else {
                            [src[0], src[1], src[2], src[3]]
                        },
                    );
                }
            }
        }
    }
    out
}

/// A clamped, gamma-encoded byte of a linear value.
fn tone(v: f32) -> u8 {
    (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8
}

/// A half float's value, decoded by hand (`half` isn't a dependency).
fn half(bytes: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(bytes);
    let exp = (h >> 10) & 0x1f;
    let frac = u32::from(h & 0x3ff);
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exp {
        0 => sign * frac as f32 * 2.0f32.powi(-24),
        31 if frac == 0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * 2.0f32.powi(i32::from(exp) - 15) * (1.0 + frac as f32 / 1024.0),
    }
}

/// The frame as packs read it (docs/api/shaders-reference.md), from the renderer's `Frame`.
pub fn frame_inputs(
    frame: &Frame,
    size: [u32; 2],
    seconds: f32,
    index: u32,
    shadow: Option<&horizon_shaders::Shadow>,
) -> FrameInputs {
    let aspect = size[0] as f32 / size[1].max(1) as f32;
    let view = crate::camera::view_matrix(frame.yaw, frame.pitch);
    let projection = super::world_projection(frame.fov_degrees, aspect, frame.depth_far)
        * frame.bob
        * frame.spin;
    // A celestial body's direction at `angle` degrees (SkyRenderer: Y -90°, then X by the
    // angle, of straight up): overhead at 0, rising in the east (+x).
    let body = |angle: f32| {
        let a = angle.to_radians();
        Vec3::new(-a.sin(), a.cos(), 0.0)
    };
    let (fog_color, fog_start, fog_end) = match &frame.fog {
        Some(f) => (
            [f.color[0], f.color[1], f.color[2]],
            f.environmental_start.min(f.render_distance_start),
            f.environmental_end.min(f.render_distance_end),
        ),
        None => (frame.sky.fog_color, frame.distance * 0.9, frame.distance),
    };
    let pack = &frame.pack;
    let level = |l: i32| l as f32 / 15.0;
    let m = |m: Mat4| m.to_cols_array_2d();
    FrameInputs {
        view: m(view),
        projection: m(projection),
        view_inverse: m(view.inverse()),
        projection_inverse: m(projection.inverse()),
        camera_position: frame.camera.as_vec3().to_array(),
        time: seconds,
        sun_direction: body(frame.sky.sun_angle).to_array(),
        day_time: pack.day_time.rem_euclid(24000) as f32 / 24000.0,
        moon_direction: body(frame.sky.moon_angle).to_array(),
        game_time: frame.shader_game_time,
        sky_color: frame.sky.sky_color,
        rain: pack.rain,
        fog_color,
        thunder: pack.thunder,
        resolution: [size[0] as f32, size[1] as f32],
        eye_light: [level(pack.eye_light[0]), level(pack.eye_light[1])],
        fog_start,
        fog_end,
        render_distance: frame.distance,
        far: frame.depth_far,
        night_vision: frame.night_vision,
        blindness: pack.blindness,
        darkness: frame.darkness.1.clamp(0.0, 1.0),
        brightness: pack.brightness,
        eye_in: pack.eye_in as u32,
        dimension: if frame.end_sky {
            1
        } else if frame.nether_lighting {
            -1
        } else {
            0
        },
        frame_index: index,
        hide_gui: u32::from(pack.hide_gui),
        water_plane: frame.water_plane.unwrap_or(0.0),
        has_reflection: u32::from(frame.water_plane.is_some()),
        shadow_distance: shadow.map_or(0.0, |s| s.distance),
        shadow_distortion: shadow.map_or(0.0, |s| {
            super::native_shadow::distortion(s.distance, s.resolution)
        }),
        light_direction: match shadow {
            Some(s) => super::pack_uniforms::shadow_light(frame, s),
            None => body(frame.sky.sun_angle),
        }
        .to_array(),
        has_shadow: u32::from(shadow.is_some()),
        shadow_view_proj: m(shadow.map_or(Mat4::IDENTITY, |s| {
            super::pack_uniforms::shadow_culling(frame, s)
        })),
        previous_view_projection: m(Mat4::IDENTITY),
        camera_motion: [0.0; 3],
        delta_time: 0.0,
        wetness: 0.0,
        biome_rain: 0.0,
        biome_snow: 0.0,
        moon_phase: pack.day_time.div_euclid(24000).rem_euclid(8) as u32,
        eye_light_smooth: [level(pack.eye_light[0]), level(pack.eye_light[1])],
        in_water: f32::from(u8::from(pack.eye_in as u32 == 1)),
        in_lava: f32::from(u8::from(pack.eye_in as u32 == 2)),
        engine_taa: [0.0; 4],
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn eased_values_start_at_the_target_and_halve_the_distance() {
        use super::eased;
        assert_eq!(eased(f32::NAN, 0.25, 2.0, 0.0), 0.25);
        assert_eq!(eased(1.0, 0.0, 2.0, 2.0), 0.5);
        assert_eq!(eased(1.0, 0.0, 2.0, 0.0), 1.0);
    }
    use super::*;
    use horizon_shader_api::{MANIFEST, Pack, check};

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = crate::gfx::test_gpu::instance();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default())).ok()
    }

    fn texture(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    /// The chain goes on past the passes that write a pack target.
    #[test]
    fn outputs_leave_the_chain_to_the_passes_after() {
        let p = plan(&[Some(0), None, Some(1), None, Some(0), None]);
        let w: Vec<_> = p.iter().map(|p| p.writes).collect();
        let r: Vec<_> = p.iter().map(|p| p.reads).collect();
        assert!(
            w == [
                Writes::Target(0),
                Writes::Chain(0),
                Writes::Target(1),
                Writes::Chain(1),
                Writes::Target(0),
                Writes::Screen
            ]
        );
        assert_eq!(r, [None, None, Some(0), Some(0), Some(1), Some(1)]);
        assert_eq!(scaled([1919, 1080], 0.5), [960, 540]);
    }

    /// Two passes: the first doubles the world's colour (over 1, so the half-float target
    /// must keep it) times an option, the second halves it and writes the depth as alpha.
    #[test]
    fn passes_chain_through_half_float_targets_with_options_and_depth() {
        let Some((device, queue)) = device() else {
            return;
        };
        let manifest = "schema = 1\nid = \"t\"\nname = \"T\"\nversion = \"1\"\napi = \"0.5\"\n\
            [[post]]\nfile = \"a.wgsl\"\n[[post]]\nfile = \"b.wgsl\"\n\
            [[options]]\nid = \"gain\"\nname = \"Gain\"\ntype = \"float\"\ndefault = 0.0\nmin = 0.0\nmax = 2.0\n";
        let a = "override gain: f32;\nfn post(input: PostInput) -> vec4<f32> { return vec4<f32>(input.color.rgb * 2.0 * gain, 1.0); }\n";
        let b = "fn post(input: PostInput) -> vec4<f32> { return vec4<f32>(input.color.rgb * 0.5, input.depth); }\n";
        let files = [(MANIFEST, manifest), ("a.wgsl", a), ("b.wgsl", b)]
            .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
            .into();
        let report = check(&Pack::from_files(files).unwrap()).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let frame = NativeFrame::new(&device, None, &[], Vec::new(), Default::default(), None, true);
        queue.write_buffer(
            &frame.buffer,
            0,
            bytemuck::bytes_of(&FrameInputs::default()),
        );
        let mut post = NativePost::new(
            &device,
            report.stages,
            &report.manifest,
            &[("gain".into(), 1.0)],
            format,
            &frame,
            None,
        )
        .unwrap();
        assert_eq!(post.labels(), ["a.wgsl", "b.wgsl"]);

        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let color = texture(&device, format, usage).create_view(&Default::default());
        let depth = texture(&device, wgpu::TextureFormat::Depth32Float, usage)
            .create_view(&Default::default());
        let screen = texture(
            &device,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.8,
                        g: 0.4,
                        b: 0.2,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.5),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        let view = screen.create_view(&Default::default());
        let surface = frame.surface_view.clone();
        post.run(
            &device,
            &mut encoder,
            (&color, &depth, &surface),
            &view,
            [4, 4],
            false,
            None,
            None,
        );
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT * 4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            screen.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                    rows_per_image: Some(4),
                },
            },
            screen.size(),
        );
        queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let pixel = buffer.slice(..).get_mapped_range().unwrap()[..4].to_vec();
        let near = |got: u8, want: u8| got.abs_diff(want) <= 1;
        assert!(
            near(pixel[0], 204) && near(pixel[1], 102) && near(pixel[2], 51) && near(pixel[3], 128),
            "{pixel:?}"
        );
    }

    /// `history` is black on the first frame, then what the pass before the last wrote the
    /// frame before.
    #[test]
    fn the_last_pass_reads_last_frames_input_as_history() {
        let Some((device, queue)) = device() else {
            return;
        };
        let manifest = "schema = 1
id = \"t\"
name = \"T\"
version = \"1\"
api = \"0.5\"
            [[post]]
file = \"a.wgsl\"
[[post]]
file = \"b.wgsl\"
";
        let a = "fn post(input: PostInput) -> vec4<f32> { return vec4<f32>(input.color.rgb * 0.5, 1.0); }
";
        let b = "fn post(input: PostInput) -> vec4<f32> { return textureLoad(history, vec2<i32>(input.pixel), 0); }
";
        let files = [(MANIFEST, manifest), ("a.wgsl", a), ("b.wgsl", b)]
            .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
            .into();
        let report = check(&Pack::from_files(files).unwrap()).unwrap();
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let frame = NativeFrame::new(&device, None, &[], Vec::new(), Default::default(), None, true);
        let mut post = NativePost::new(
            &device,
            report.stages,
            &report.manifest,
            &[],
            format,
            &frame,
            None,
        )
        .unwrap();
        let surface = frame.surface_view.clone();
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let color = texture(&device, format, usage).create_view(&Default::default());
        let depth = texture(&device, wgpu::TextureFormat::Depth32Float, usage)
            .create_view(&Default::default());
        let screen = texture(
            &device,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let view = screen.create_view(&Default::default());
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT * 4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut frame_pixel = |history_reset| {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            post.run(
                &device,
                &mut encoder,
                (&color, &depth, &surface),
                &view,
                [4, 4],
                history_reset,
                None,
                None,
            );
            encoder.copy_texture_to_buffer(
                screen.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT),
                        rows_per_image: Some(4),
                    },
                },
                screen.size(),
            );
            queue.submit([encoder.finish()]);
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |r| r.unwrap());
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let pixel = buffer.slice(..).get_mapped_range().unwrap()[..4].to_vec();
            buffer.unmap();
            pixel
        };
        assert_eq!(
            frame_pixel(true)[..3],
            [0, 0, 0],
            "black on the first frame"
        );
        let second = frame_pixel(false);
        assert!(
            second[..3].iter().all(|&v| v.abs_diff(128) <= 1),
            "{second:?}"
        );
        assert_eq!(frame_pixel(true)[..3], [0, 0, 0], "black after a frame gap");
        let fourth = frame_pixel(false);
        assert!(
            fourth[..3].iter().all(|&v| v.abs_diff(128) <= 1),
            "{fourth:?}"
        );
    }
}
