//! One of Horizon's own shader packs while it runs (docs/shaders.md): the frame its stages
//! read, its sky, material and post stages, and the timer of their passes.

use horizon_shader_api::{Report, StageKind};

use super::displace::Displacement;
use super::native_material::{DisplacedShape, NativeMaterial, TerrainShape};
use super::native_post::{NativeFrame, NativePost};
use super::native_shadow::{NativeShadow, ShadowGrid};
use super::native_sky::NativeSky;
use super::pack_timing::PassTiming;

pub struct NativePack {
    pub frame: NativeFrame,
    pub sky: Option<NativeSky>,
    pub material: Option<NativeMaterial>,
    pub post: Option<NativePost>,
    /// The shadow map's pass, when the pack asks for one (`[shadow]`).
    pub shadow: Option<NativeShadow>,
    /// `[displacement]`'s grid, while it is on and the resource packs have height maps.
    /// `[[particle_lights]]`.
    pub particle_lights: Vec<horizon_shader_api::manifest::ParticleLight>,
    /// `[[lights]]`, for the colour of held and dropped items' light.
    pub light_colors: Vec<horizon_shader_api::manifest::LightColor>,
    /// GPU time of each pass, while the renderer asks for it (`pack_timing.rs`).
    pub timing: Option<PassTiming>,
    /// Whether Minecraft's clouds draw beside the pack's sky (`[sky] clouds`).
    pub vanilla_clouds: bool,
    /// Whether Minecraft's sun draws over the pack's sky (`[sky] sun`).
    pub vanilla_sun: bool,
    /// Whether Minecraft's stars draw over the pack's sky (`[sky] stars`).
    pub vanilla_stars: bool,
    /// The pack's material stage fogs terrain itself (`[material] fog = false`).
    pub own_fog: bool,
    /// The weather stage's pipeline (`[weather]`), drawing rain and snow in vanilla's place.
    pub weather: Option<wgpu::RenderPipeline>,
    /// The post pass after which it draws (`[weather] after`), in place of with the world:
    /// its pipeline is then built for the passes' targets.
    pub weather_after: Option<usize>,
    /// `[displacement]`'s grid, while it is on and the resource packs have height maps.
    pub displacement: Option<Displacement>,
}

/// Whether the option `id`, if any, is on in `constants` (not 0; on when it is not there).
fn option_on(constants: &[(String, f64)], id: Option<&String>) -> bool {
    id.is_none_or(|id| {
        constants
            .iter()
            .find(|(k, _)| k == id)
            .is_none_or(|(_, v)| *v != 0.0)
    })
}

/// `[displacement]`'s cells: the value of its `cells_option`, if any.
fn chosen_cells(
    d: &horizon_shader_api::manifest::Displacement,
    options: &[horizon_shader_api::manifest::PackOption],
    constants: &[(String, f64)],
) -> u32 {
    let chosen = d.cells_option.as_ref().and_then(|id| {
        let option = options.iter().find(|o| o.id == *id)?;
        let horizon_shader_api::manifest::OptionKind::Choice { values, .. } = &option.kind else {
            return None;
        };
        let (_, index) = constants.iter().find(|(k, _)| k == id)?;
        values.get(*index as usize)?.parse().ok()
    });
    chosen.unwrap_or(d.cells)
}

/// `[shadow]` with the option values `constants`: none while its `option` is off, else the
/// resolution and distance its `resolution_option` and `distance_option` pick.
fn chosen_shadow(
    s: &horizon_shader_api::manifest::Shadow,
    options: &[horizon_shader_api::manifest::PackOption],
    constants: &[(String, f64)],
) -> Option<horizon_shader_api::manifest::Shadow> {
    let number = |id: Option<&String>| {
        let option = options.iter().find(|o| Some(&o.id) == id)?;
        let (_, value) = constants.iter().find(|(k, _)| *k == option.id)?;
        option.kind.number(*value)
    };
    option_on(constants, s.option.as_ref()).then(|| horizon_shader_api::manifest::Shadow {
        resolution: number(s.resolution_option.as_ref()).map_or(s.resolution, |r| r as u32),
        distance: number(s.distance_option.as_ref()).map_or(s.distance, |d| d as f32),
        ..s.clone()
    })
}

impl NativePack {
    /// The pack of the checked stages `report` with the option values `constants`, drawing
    /// onto the world's targets as `terrain` describes; the sky reads the terrain's group 0.
    /// Also returns its stages' labels, in the order they draw. Errors name the stage wgpu
    /// refused.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TerrainShape<'_>,
        report: Report,
        constants: &[(String, f64)],
        materials: &[u32],
        lights: super::block_light::LightTable,
    ) -> Result<(Self, Vec<String>), String> {
        // A backend may refuse a layout or pipeline the pack asks for (a stage's bindings
        // past its sampled-texture limit, say, as OpenGL's 16 was before #10095): the pack
        // is refused with that error instead of a validation panic in the renderer.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = Self::build(device, queue, terrain, report, constants, materials, lights);
        let refused = pollster::block_on(scope.pop());
        match (built, refused) {
            (Ok(built), None) => Ok(built),
            (Ok(_), Some(e)) => Err(e.to_string()),
            (Err(e), _) => Err(e),
        }
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &TerrainShape<'_>,
        report: Report,
        constants: &[(String, f64)],
        materials: &[u32],
        lights: super::block_light::LightTable,
    ) -> Result<(Self, Vec<String>), String> {
        let (globals, format) = (terrain.globals, terrain.format);
        let clock = std::time::Instant::now();
        let mut marks: Vec<(&'static str, u128)> = Vec::new();
        let mark = |name: &'static str, marks: &mut Vec<(&'static str, u128)>| {
            marks.push((name, clock.elapsed().as_millis()))
        };
        let textures = super::native_post::pack_textures(device, queue, &report);
        let chosen = report
            .shadow
            .as_ref()
            .and_then(|s| chosen_shadow(s, &report.manifest.options, constants));
        let settings = chosen
            .as_ref()
            .map(|v| super::native_shadow::fitted(device, v));
        let api = report.manifest.api;
        let rays = report.manifest.rays.is_some()
            && option_on(
                constants,
                report
                    .manifest
                    .rays
                    .as_ref()
                    .and_then(|r| r.option.as_ref()),
            );
        let mut frame = NativeFrame::with_rt(
            device,
            settings.as_ref(),
            materials.to_vec(),
            textures,
            lights,
            terrain.pbr,
            api,
            rays,
        );
        // `[displacement]`, unless its option is off; nothing to displace without heights.
        let displacement = report
            .manifest
            .displacement
            .as_ref()
            .filter(|d| option_on(constants, d.option.as_ref()))
            .filter(|_| terrain.pbr.is_some())
            .map(|d| {
                // No more cells than the height maps have texels.
                let texels = terrain.pbr.map_or(16, |p| p[0].width());
                let cells = chosen_cells(d, &report.manifest.options, constants)
                    .min(texels)
                    .max(1);
                let steps =
                    d.steps_option.is_some() && option_on(constants, d.steps_option.as_ref());
                Displacement::new(
                    device,
                    d.distance,
                    cells,
                    steps,
                    &frame.pbr[0],
                    report.shadow.is_some(),
                )
            });
        mark("frame", &mut marks);
        let vertex = report.stages.iter().find(|s| s.kind == StageKind::Vertex);
        let shadow = chosen
            .as_ref()
            .map(|s| {
                let material = report.stages.iter().find(|s| s.kind == StageKind::Material);
                let grid = displacement
                    .as_ref()
                    .and_then(|d| Some((d, d.shadow.as_ref()?, material?)))
                    .map(|(d, picks, material)| ShadowGrid {
                        distance: d.distance,
                        steps: picks.steps,
                        material,
                        layout: &d.draw_layout,
                    });
                NativeShadow::new(device, terrain, &frame, s, vertex, constants, grid)
            })
            .transpose()?;
        mark("shadow", &mut marks);
        let has_vertex = vertex.is_some();
        let reflection = report
            .stages
            .iter()
            .find(|s| s.kind == StageKind::Material)
            .is_some_and(|s| s.reads("reflection"))
            && option_on(
                constants,
                report
                    .manifest
                    .reflection
                    .as_ref()
                    .and_then(|r| r.option.as_ref()),
            );
        let fog = report.material_fog;
        let vanilla_clouds = report
            .manifest
            .sky
            .as_ref()
            .and_then(|s| s.clouds)
            .unwrap_or(true);
        let vanilla_sun = report
            .manifest
            .sky
            .as_ref()
            .and_then(|s| s.sun)
            .unwrap_or(true);
        let vanilla_stars = report
            .manifest
            .sky
            .as_ref()
            .and_then(|s| s.stars)
            .unwrap_or(true);
        let weather_after = report
            .manifest
            .weather
            .as_ref()
            .and_then(|w| w.after.as_ref())
            .and_then(|after| report.manifest.post.iter().position(|p| &p.file == after));
        let weather = report
            .stages
            .iter()
            .find(|s| s.kind == StageKind::Weather)
            .map(|stage| {
                super::weather::native_pipeline(
                    device,
                    if weather_after.is_some() {
                        super::native_post::INTERMEDIATE
                    } else {
                        format
                    },
                    terrain.weather_groups[0],
                    terrain.weather_groups[1],
                    &frame,
                    stage,
                    constants,
                )
            })
            .transpose()?;
        mark("weather", &mut marks);
        let (post, others): (Vec<_>, Vec<_>) = report
            .stages
            .into_iter()
            .partition(|s| s.kind == StageKind::Post);
        let sky = others
            .iter()
            .find(|s| s.kind == StageKind::Sky)
            .map(|stage| NativeSky::new(device, globals, &frame, stage, constants, format))
            .transpose()?;
        mark("sky", &mut marks);
        let material = others
            .iter()
            .find(|s| s.kind == StageKind::Material)
            .map(|stage| {
                let displaced = displacement.as_ref().map(|d| DisplacedShape {
                    distance: d.distance,
                    steps: d.view.steps,
                    layout: &d.draw_layout,
                });
                NativeMaterial::new(
                    device,
                    terrain,
                    &frame,
                    stage,
                    constants,
                    fog,
                    has_vertex,
                    displaced.as_ref(),
                )
            })
            .transpose()?;
        mark("material", &mut marks);
        let tonemap = others.iter().find(|s| s.kind == StageKind::Tonemap);
        let post = (!post.is_empty() || api == horizon_shader_api::Api::V0_2)
            .then(|| {
                NativePost::new_with_rt(
                    device,
                    post,
                    &report.manifest,
                    constants,
                    format,
                    &frame,
                    tonemap,
                    Some((terrain.globals, terrain.sprites)),
                )
            })
            .transpose()?;
        mark("post", &mut marks);
        log::info!("shader pack pipelines, ms since start: {marks:?}");
        let mut labels: Vec<String> = sky.iter().map(|s| s.label.clone()).collect();
        labels.extend(
            others
                .iter()
                .filter(|s| material.is_some() && s.kind == StageKind::Material)
                .map(|s| s.label.clone()),
        );
        if let Some(post) = &post {
            labels.extend(post.labels().into_iter().map(str::to_owned));
        }
        let own_fog = material.is_some() && !fog;
        frame.reflects = reflection;
        let pack = Self {
            particle_lights: report.manifest.particle_lights.clone(),
            light_colors: report.manifest.lights.clone(),
            frame,
            sky,
            material,
            post,
            shadow,
            timing: None,
            vanilla_clouds,
            vanilla_sun,
            vanilla_stars,
            own_fog,
            weather,
            weather_after,
            displacement,
        };
        Ok((pack, labels))
    }

    /// Whether the pack draws anything the game runs today.
    pub fn draws(&self) -> bool {
        self.sky.is_some() || self.material.is_some() || self.post.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{NativePack, TerrainShape, chosen_shadow};
    use horizon_shader_api::manifest::{OptionKind, PackOption, Shadow};

    /// A device with OpenGL's minimum 16 sampled textures per stage refuses a pack whose
    /// sky adds the terrain's 10 to the frame's 9: that is the pack's error to report, not
    /// a validation panic (#10095).
    #[test]
    fn a_pack_the_backend_refuses_is_an_error_not_a_panic() {
        use horizon_shader_api::{MANIFEST, Pack, check};
        let instance = crate::gfx::test_gpu::instance();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            return;
        };
        let mut limits = adapter.limits();
        limits.max_sampled_textures_per_shader_stage = 16;
        let Ok((device, queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_limits: limits,
                ..Default::default()
            }))
        else {
            return;
        };
        // The terrain's group 0: its uniform and the 10 sampled textures (_terrain.wgsl_'s
        // 8 sprite arrays, the lightmap and the emission thresholds).
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty,
            count: None,
        };
        let mut entries = vec![entry(
            0,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        )];
        entries.extend((1..=10).map(|binding| {
            entry(
                binding,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            )
        }));
        let globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[],
        });
        let terrain = TerrainShape {
            globals: &globals,
            flat: None,
            sprites: [1.0, 2048.0],
            format: wgpu::TextureFormat::Rgba8Unorm,
            sprite_tags: &[],
            sprite_materials: &[],
            entities: &empty,
            weather_groups: [&empty, &empty],
            pbr: None,
        };
        let manifest = "schema = 1\nid = \"t\"\nname = \"T\"\nversion = \"1\"\napi = \"0.1\"\n\
            [sky]\nfile = \"sky.wgsl\"\n";
        let files = [
            (MANIFEST, manifest),
            (
                "sky.wgsl",
                "fn sky(input: SkyInput) -> vec4<f32> { return input.color; }\n",
            ),
        ]
        .map(|(n, t)| (n.to_owned(), t.as_bytes().to_vec()))
        .into();
        let report = check(&Pack::from_files(files).unwrap()).unwrap();
        let built = NativePack::new(&device, &queue, &terrain, report, &[], Default::default());
        assert!(built.is_err(), "the pack should be refused");
    }

    fn option(id: &str, kind: OptionKind) -> PackOption {
        PackOption {
            id: id.into(),
            name: id.into(),
            description: String::new(),
            kind,
        }
    }

    #[test]
    fn the_shadow_map_follows_its_options() {
        let options = [
            option("shadows", OptionKind::Bool { default: true }),
            option(
                "quality",
                OptionKind::Choice {
                    values: vec!["1024".into(), "4096".into()],
                    default: 0,
                },
            ),
            option(
                "reach",
                OptionKind::Int {
                    default: 96,
                    min: 32,
                    max: 512,
                },
            ),
        ];
        let shadow = Shadow {
            option: Some("shadows".into()),
            resolution_option: Some("quality".into()),
            distance_option: Some("reach".into()),
            ..Shadow::default()
        };
        let values = |on: f64, quality: f64, reach: f64| {
            [("shadows", on), ("quality", quality), ("reach", reach)].map(|(k, v)| (k.into(), v))
        };
        let size = |values: &[(String, f64)]| {
            chosen_shadow(&shadow, &options, values).map(|s| (s.resolution, s.distance))
        };
        assert_eq!(size(&values(1.0, 1.0, 200.0)), Some((4096, 200.0)));
        assert_eq!(size(&values(1.0, 0.0, 64.0)), Some((1024, 64.0)));
        assert_eq!(size(&values(0.0, 1.0, 200.0)), None);
        // Without the options' keys: the manifest's own numbers, always drawn.
        assert_eq!(
            chosen_shadow(&Shadow::default(), &options, &values(0.0, 1.0, 200.0))
                .map(|s| (s.resolution, s.distance)),
            Some((2048, 128.0))
        );
    }
}
