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
    /// `[displacement]`'s grid, while it is on and the resource packs have height maps.
    pub displacement: Option<Displacement>,
    /// Whether the material stage reads the water mirror (`reflection`) and `[reflection]`'s
    /// option, if any, is on: else the reflection pass is left out.
    pub reflection: bool,
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
        let (globals, format) = (terrain.globals, terrain.format);
        let clock = std::time::Instant::now();
        let mut marks: Vec<(&'static str, u128)> = Vec::new();
        let mark = |name: &'static str, marks: &mut Vec<(&'static str, u128)>| marks.push((name, clock.elapsed().as_millis()));
        let textures = super::native_post::pack_textures(device, queue, &report);
        let mut frame = NativeFrame::new(
            device,
            report.shadow.as_ref(),
            materials,
            textures,
            lights,
            terrain.pbr,
            option_on(
                constants,
                report.manifest.rays.as_ref().and_then(|r| r.option.as_ref()),
            ),
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
        let shadow = report
            .shadow
            .as_ref()
            .map(|s| {
                let material = report.stages.iter().find(|s| s.kind == StageKind::Material);
                let grid = displacement
                    .as_ref()
                    .and_then(|d| Some((d, d.shadow.as_ref()?, material?)))
                    .map(|(d, picks, material)| ShadowGrid {
                        distance: d.distance,
                        cells: picks.cells,
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
        let weather = report
            .stages
            .iter()
            .find(|s| s.kind == StageKind::Weather)
            .map(|stage| {
                super::weather::native_pipeline(
                    device,
                    format,
                    terrain.entities,
                    terrain.weather_time,
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
                    cells: d.view.cells,
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
        let post = (!post.is_empty())
            .then(|| {
                NativePost::new(
                    device,
                    post,
                    &report.manifest,
                    constants,
                    format,
                    &frame,
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
            displacement,
            reflection,
        };
        Ok((pack, labels))
    }

    /// Whether the pack draws anything the game runs today.
    pub fn draws(&self) -> bool {
        self.sky.is_some() || self.material.is_some() || self.post.is_some()
    }
}
