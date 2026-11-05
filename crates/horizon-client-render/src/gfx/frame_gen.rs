//! Frame generation (a Horizon setting, off by default): FSR 3.1's frame interpolation
//! makes a frame halfway between each two real ones. The world, without the hand and
//! HUD, is kept at the screen's size each frame; optical flow and the interpolation run
//! on it, the hand and HUD are drawn over the made frame as over the real one, and a
//! present thread shows the made frame, then the real one half a real frame later.
//!
//! The renderer draws into frames of its own while this is on; the present thread
//! copies each to the swapchain, which it alone acquires and presents.

mod interpolation;
mod motion;
/// The format of the object motion texture (`object_motion`).
pub const MOTION_FORMAT: wgpu::TextureFormat = motion::FORMAT;
mod optical_flow;

use glam::{DMat4, DVec3, Mat4};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};

/// `HORIZON_FRAMEGEN_DEBUG=generated`: show only made frames (AMD's debug view), so a
/// screenshot or an offscreen frame is the made one; `motion`: the same, each made frame
/// showing what moved on its own in blue (its motion vectors in red and green).
pub(super) fn generated_only() -> bool {
    debug().is_some()
}

fn debug() -> Option<&'static str> {
    static DEBUG: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DEBUG
        .get_or_init(|| {
            std::env::var("HORIZON_FRAMEGEN_DEBUG")
                .ok()
                .filter(|v| v == "generated" || v == "motion")
        })
        .as_deref()
}

/// Bound both CPU presentation and GPU work; a texture pool alone does not bound
/// submissions because textures come back as soon as their copies are queued.
pub(super) fn low_latency() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("HORIZON_FRAMEGEN_LOW_LATENCY").as_deref() == Ok("1"))
}

pub struct FrameGen {
    pub on: bool,
    backend: crate::graphics::FrameGeneration,
    multiplier: u32,
    extra_frames: Vec<wgpu::Texture>,
    generated_count: usize,
    dlss_ran: bool,
    /// The present thread failed to show a frame (or died): until the renderer has
    /// acquired a frame itself, the normal path runs and handles the surface.
    pub paused: bool,
    /// Native DLSS frame generation (NVIDIA's DLSS-G, experimental), over FSR's
    /// interpolation: only with `HORIZON_DLSS_FG=1` and the experimental bridge at
    /// `HORIZON_DLSSG_BRIDGE`; a failed evaluation falls back to FSR.
    #[cfg(all(feature = "dlss", target_os = "linux"))]
    fg: Option<DlssFg>,
    /// No native DLSS without the feature; the field keeps the layout stable.
    #[cfg(not(all(feature = "dlss", target_os = "linux")))]
    _fg: (),
    gpu: Option<Gpu>,
    presenter: Option<Presenter>,
    /// When the last real frame was handed over, and the time between real frames.
    last: Option<Instant>,
    interval: Duration,
    /// The last frame's unjittered `view_proj` and camera.
    previous: Option<(DMat4, DVec3)>,
    /// The last frame's hand vertices and the bob and view they were drawn with.
    hand: Vec<crate::mesher::Vertex>,
    hand_view: Mat4,
    /// Offscreen, the frame the made frame is drawn into.
    offscreen_generated: Option<wgpu::Texture>,
    /// Motion vectors for what moves on its own, made on the first frame.
    motion: Option<motion::ObjectMotion>,
    /// This frame's object motion when it was drawn before TAA (`object_motion`), for
    /// `generate` to use rather than draw it again.
    drawn_motion: Option<wgpu::TextureView>,
    /// Real frames and made ones handed to the screen since `shown_since`, and the share
    /// of real frames that had a made frame over the last half second (the F3 screen's),
    /// `None` while off.
    real: u32,
    made: u32,
    shown_since: Instant,
    pub made_share: Option<f32>,
}

/// What runs on the GPU for one screen size.
struct Gpu {
    size: [u32; 2],
    fsr: Option<Fsr>,
    /// This frame's world at the screen's size, and bindings that copy it or the
    /// interpolation's output (`Post::finish_with`).
    world: wgpu::TextureView,
    world_group: wgpu::BindGroup,
    /// The next interpolation has no previous frame.
    reset: bool,
}

struct Fsr {
    flow: optical_flow::OpticalFlow,
    interpolation: interpolation::Interpolation,
    output_group: wgpu::BindGroup,
}

/// This frame's camera, for the interpolation.
pub struct Camera {
    /// Camera view to clip space, without world rotation or temporal jitter.
    pub projection: Mat4,
    /// The world's projection this frame (with TAA's jitter) and without the jitter.
    pub jittered: Mat4,
    pub view_proj: Mat4,
    pub depth: wgpu::TextureView,
    pub render_size: [u32; 2],
    pub jitter: [f32; 2],
    /// This frame's unjittered clip space to the previous frame's, `None` without one.
    pub reproject: Option<Mat4>,
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
}

/// The screen-size DLSS-G resources: input in the format the prototype's driver accepts
/// (RGBA8), and the output the driver draws the made frame into. FSR 3.1 keeps its own
/// flow and interpolation for the fallback.
#[cfg(all(feature = "dlss", target_os = "linux"))]
struct DlssFg {
    bridge: horizon_hdr::dlss_fg::DlssFg,
    input: Option<([u32; 2], wgpu::TextureView)>,
    output: Vec<wgpu::TextureView>,
    motion: Option<super::dlss::MotionVectors>,
}

#[cfg(all(feature = "dlss", target_os = "linux"))]
impl DlssFg {
    /// One when the opt-in environment set both the switch and the bridge's path.
    fn new() -> Option<Self> {
        Some(Self {
            bridge: horizon_hdr::dlss_fg::DlssFg::new()?,
            input: None,
            output: Vec::new(),
            motion: None,
        })
    }

    /// This screen size's input target, made at once if the size changed.
    fn input(&mut self, device: &wgpu::Device, size: [u32; 2]) -> &wgpu::TextureView {
        if self.input.as_ref().is_none_or(|(s, _)| *s != size) {
            self.input = Some((size, target(device, size, "Frame generation DLSS-G input")));
        }
        &self.input.as_ref().expect("just made").1
    }

    fn outputs(
        &mut self,
        device: &wgpu::Device,
        size: [u32; 2],
        count: usize,
    ) -> Vec<wgpu::TextureView> {
        if self
            .output
            .first()
            .is_some_and(|v| [v.texture().width(), v.texture().height()] != size)
        {
            self.output.clear();
        }
        self.output
            .resize_with(count, || target(device, size, "DLSS-G generated output"));
        self.output.clone()
    }
}

/// An RGBA8 target for DLSS-G to read or write, sized to the screen.
#[cfg(all(feature = "dlss", target_os = "linux"))]
fn target(device: &wgpu::Device, size: [u32; 2], label: &str) -> wgpu::TextureView {
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl FrameGen {
    pub fn new() -> Self {
        Self {
            on: false,
            backend: crate::graphics::FrameGeneration::Off,
            multiplier: 2,
            extra_frames: Vec::new(),
            generated_count: 0,
            dlss_ran: false,
            paused: false,
            #[cfg(all(feature = "dlss", target_os = "linux"))]
            fg: None,
            #[cfg(not(all(feature = "dlss", target_os = "linux")))]
            _fg: (),
            gpu: None,
            presenter: None,
            last: None,
            interval: Duration::from_millis(16),
            previous: None,
            hand: Vec::new(),
            hand_view: Mat4::IDENTITY,
            offscreen_generated: None,
            motion: None,
            drawn_motion: None,
            real: 0,
            made: 0,
            shown_since: Instant::now(),
            made_share: None,
        }
    }

    #[cfg(all(feature = "dlss", target_os = "linux"))]
    pub fn shutdown_native(&mut self) {
        self.fg = None;
        horizon_hdr::dlss_fg::shutdown();
    }

    pub fn configure(&mut self, backend: crate::graphics::FrameGeneration, multiplier: u32) {
        let multiplier = multiplier.clamp(2, 4);
        if self.backend == backend && self.multiplier == multiplier {
            return;
        }
        self.stop();
        self.backend = backend;
        self.multiplier = multiplier;
        self.on = backend != crate::graphics::FrameGeneration::Off;
        #[cfg(all(feature = "dlss", target_os = "linux"))]
        {
            self.fg = if backend == crate::graphics::FrameGeneration::Dlss {
                DlssFg::new()
            } else {
                None
            };
        }
    }
    pub fn label(&self) -> String {
        if self.dlss_ran {
            format!("DLSS-G {}x", self.generated_count + 1)
        } else {
            "FSR 3.1 2x".into()
        }
    }
    pub fn extra_generated(&self) -> Vec<wgpu::Texture> {
        self.extra_frames
            .iter()
            .take(self.generated_count.saturating_sub(1))
            .cloned()
            .collect()
    }
    fn requested_count(&self) -> usize {
        if self.backend == crate::graphics::FrameGeneration::Dlss {
            self.multiplier as usize - 1
        } else {
            1
        }
    }

    /// Ends frame generation's frames and present thread: before the surface is
    /// reconfigured or the renderer acquires from it itself.
    pub fn stop(&mut self) {
        self.presenter = None;
        self.extra_frames.clear();
        self.generated_count = 0;
        self.made_share = None;
        (self.real, self.made) = (0, 0);
        self.last = None;
        self.previous = None;
        self.hand.clear();
        if let Some(motion) = &mut self.motion {
            motion.reset();
        }
        if let Some(gpu) = &mut self.gpu {
            gpu.reset = true;
        }
    }

    /// The hand for the made frame, appended to `out`: each vertex halfway between where
    /// it was on screen last frame and where it is now (the hand swings and bobs on its
    /// own, so the world's interpolation can't move it), in this frame's space
    /// (`view`: this frame's bob and view). This frame's hand where the two differ in
    /// shape (another item, the first frame).
    pub fn halfway_hand(
        &mut self,
        hand: &[crate::mesher::Vertex],
        view: Mat4,
        out: &mut Vec<crate::mesher::Vertex>,
    ) {
        let same = self.hand.len() == hand.len()
            && self
                .hand
                .iter()
                .zip(hand)
                .all(|(a, b)| a.layer == b.layer && a.uv == b.uv);
        if same {
            let back = view.inverse();
            out.extend(self.hand.iter().zip(hand).map(|(before, now)| {
                let then = self.hand_view.transform_point3(before.pos.into());
                let mid = (then + view.transform_point3(now.pos.into())) * 0.5;
                crate::mesher::Vertex {
                    pos: back.transform_point3(mid).into(),
                    ..*now
                }
            }));
        } else {
            out.extend_from_slice(hand);
        }
        self.hand.clear();
        self.hand.extend_from_slice(hand);
        self.hand_view = view;
    }

    /// This frame's entity vertices (relative to `camera`) and the entity ids of their
    /// spans (`Frame::entity_motion`), for their own motion vectors; `atlas`: the
    /// entities' textures. Before `generate`, every frame.
    pub fn track_entities(
        &mut self,
        device: &wgpu::Device,
        atlas: &wgpu::TextureView,
        entities: &[crate::mesher::Vertex],
        spans: &[(u32, u64)],
        camera: DVec3,
    ) {
        self.drawn_motion = None;
        self.motion
            .get_or_insert_with(|| motion::ObjectMotion::new(device, atlas.clone()))
            .track(entities, spans, camera);
    }

    /// Draws this frame's object motion now (after the world, before TAA, which reads it too),
    /// at the world's `size`; `previous`: last frame's projection from this camera. `None`
    /// until `track_entities` ran.
    #[allow(clippy::too_many_arguments)]
    pub fn object_motion(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::TextureView,
        size: [u32; 2],
        jittered: Mat4,
        current: Mat4,
        previous: Mat4,
    ) -> Option<wgpu::TextureView> {
        let view = self.motion.as_mut()?.draw(
            device, queue, encoder, depth, size, jittered, current, previous,
        );
        self.drawn_motion = Some(view.clone());
        Some(view)
    }

    /// This frame's unjittered clip space to the last frame's, from their `view_proj`s
    /// (relative to `camera`); `None` on the first frame.
    pub fn camera_motion(&mut self, view_proj: Mat4, camera: DVec3) -> Option<Mat4> {
        let current = view_proj.as_dmat4();
        let previous = self.previous.replace((current, camera));
        previous.map(|(p, pc)| super::world_post::reprojection(p, pc, current, camera).as_mat4())
    }

    /// Frames to draw the real and the made frame into, from the present thread's
    /// (started for this `config`'s surface) or, offscreen, the offscreen target and
    /// one of its own. `None` once the present thread failed or died: it is stopped and
    /// frame generation `paused`.
    pub fn frames(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Option<&Arc<wgpu::Surface<'static>>>,
        offscreen: Option<&wgpu::Texture>,
        window: &Arc<winit::window::Window>,
        config: &wgpu::SurfaceConfiguration,
    ) -> Option<(wgpu::Texture, wgpu::Texture)> {
        let extra_count = self.requested_count() - 1;
        let Some(surface) = surface else {
            let real = offscreen.cloned().unwrap_or_else(|| frame(device, config));
            if self
                .offscreen_generated
                .as_ref()
                .is_none_or(|t| t.size() != real.size())
            {
                self.offscreen_generated = Some(frame(device, config));
            }
            if self
                .extra_frames
                .first()
                .is_some_and(|t| t.size() != real.size())
            {
                self.extra_frames.clear();
            }
            self.extra_frames
                .resize_with(extra_count, || frame(device, config));
            let generated = self.offscreen_generated.clone().expect("made above");
            return Some((real, generated));
        };
        if self.presenter.as_ref().is_some_and(|p| p.config != *config) {
            self.presenter = None;
        }
        let presenter = self.presenter.get_or_insert_with(|| {
            Presenter::start(
                device,
                queue,
                surface.clone(),
                window.clone(),
                config,
                extra_count + 2,
            )
        });
        // A closed channel: the thread died.
        self.extra_frames.clear();
        let pair = if presenter.failed() {
            None
        } else {
            let pair = presenter.free.recv().ok().zip(presenter.free.recv().ok());
            for _ in 0..extra_count {
                if let Ok(t) = presenter.free.recv() {
                    self.extra_frames.push(t);
                }
            }
            pair
        };
        if pair.is_none() {
            self.stop();
            self.paused = true;
        }
        pair
    }

    /// The world's target at the screen's size this frame, where `Post::finish` draws.
    pub fn world(
        &mut self,
        device: &wgpu::Device,
        post: &super::world_post::Post,
        config: &wgpu::SurfaceConfiguration,
    ) -> wgpu::TextureView {
        let size = [config.width, config.height];
        if self.gpu.as_ref().is_none_or(|g| g.size != size) {
            let world = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Frame generation world"),
                    size: wgpu::Extent3d {
                        width: size[0],
                        height: size[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: config.format.remove_srgb_suffix(),
                    // COPY_SRC: the interpolation keeps it as the previous frame.
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&Default::default());
            self.gpu = Some(Gpu {
                size,
                fsr: None,
                world_group: post.bind(device, &world),
                world,
                reset: true,
            });
        }
        self.gpu.as_ref().expect("made above").world.clone()
    }

    /// Copy the world (`world`) onto `screen` for the real frame, and draw the made
    /// frame's world onto `generated`. Returns false when there is no made frame (the
    /// first after a reset): `generated` then holds this frame's world too.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        post: &super::world_post::Post,
        screen: &wgpu::TextureView,
        generated: &wgpu::TextureView,
        camera: &Camera,
        mut timing: Option<(&mut super::pack_timing::PassTiming, wgpu::TextureFormat)>,
    ) -> bool {
        // In a timed frame, each stage's GPU time as a span of its own (marked on `generated`).
        let span =
            |timing: &mut Option<(&mut super::pack_timing::PassTiming, wgpu::TextureFormat)>,
             encoder: &mut wgpu::CommandEncoder,
             name: &str| {
                if let Some((t, format)) = timing.as_mut() {
                    t.span(device, encoder, (generated, *format), name);
                }
            };
        let count = self.requested_count();
        self.generated_count = 1;
        self.dlss_ran = false;
        let gpu = self.gpu.as_mut().expect("world() ran this frame");
        post.finish_with(encoder, screen, &gpu.world_group, None);
        span(&mut timing, encoder, "Frame generation (real frame)");
        let reset = gpu.reset || camera.reproject.is_none();
        let reproject = camera.reproject.unwrap_or(Mat4::IDENTITY);
        let object_motion = match self.drawn_motion.take() {
            Some(view) => view,
            None => self
                .motion
                .as_mut()
                .expect("track_entities() ran this frame")
                .draw(
                    device,
                    queue,
                    encoder,
                    &camera.depth,
                    camera.render_size,
                    camera.jittered,
                    camera.view_proj,
                    reproject * camera.view_proj,
                ),
        };
        span(&mut timing, encoder, "Frame generation (object motion)");
        // Native DLSS-G (experimental): the driver makes the made frame itself, so the
        // optical flow and the interpolation stay idle. A failed evaluation falls back
        // to them, and a reset frame shows like a real one (no made frame to show).
        #[cfg(all(feature = "dlss", target_os = "linux"))]
        log::debug!(target: "frame_gen", "DLSS branch: fg={}", self.fg.is_some());
        #[cfg(all(feature = "dlss", target_os = "linux"))]
        if self.fg.is_some() {
            let dlss = self.fg.as_mut().expect("checked above");
            // Clone the targets: they borrow `dlss` once, the render mutably borrows
            // the bridge object alone.
            let input = dlss.input(device, gpu.size).clone();
            let outputs = dlss.outputs(device, gpu.size, count);
            // DLSS-G's input is plain RGBA8 where the frame is BGRA: the copy runs
            // on its pipeline, not `finish_with`'s (the render pass's format differs).
            post.copy_onto_dlss(
                encoder,
                &input,
                &gpu.world_group,
                timing.as_mut().map(|(t, _)| &mut **t),
            );
            span(&mut timing, encoder, "Frame generation (DLSS-G input)");
            let vectors = dlss
                .motion
                .get_or_insert_with(|| super::dlss::MotionVectors::new(device));
            let vectors = vectors.draw(
                device,
                queue,
                encoder,
                &camera.depth,
                &object_motion,
                camera.render_size,
                super::dlss::MotionUniform {
                    reproject: reproject.to_cols_array_2d(),
                    jitter: [
                        camera.jitter[0] * 2.0 / camera.render_size[0] as f32,
                        -camera.jitter[1] * 2.0 / camera.render_size[1] as f32,
                        0.0,
                        0.0,
                    ],
                },
            );
            let done = dlss.bridge.render(
                device,
                queue,
                encoder,
                &input,
                &camera.depth,
                &vectors,
                &outputs,
                horizon_hdr::dlss_fg::Camera {
                    // Driver matrices go row-major: the C side indexes `[row][col]`,
                    // so a column-major glam matrix sends as its transpose.
                    projection: camera.projection.transpose().to_cols_array(),
                    projection_inverse: camera.projection.inverse().transpose().to_cols_array(),
                    reprojection: reproject.transpose().to_cols_array(),
                    reprojection_inverse: reproject.inverse().transpose().to_cols_array(),
                    jitter: camera.jitter,
                    near: camera.near,
                    far: camera.far,
                    fov_y: camera.fov_y,
                    reset,
                    depth_inverted: true,
                },
            );
            if done {
                for (i, output) in outputs.iter().enumerate() {
                    let group = post.bind(device, output);
                    if i == 0 {
                        post.finish_with(encoder, generated, &group, None);
                    } else {
                        let view =
                            self.extra_frames[i - 1].create_view(&wgpu::TextureViewDescriptor {
                                format: Some(generated.texture().format().remove_srgb_suffix()),
                                ..Default::default()
                            });
                        post.finish_with(encoder, &view, &group, None);
                    }
                }
                self.dlss_ran = true;
                self.generated_count = count;
                span(&mut timing, encoder, "Frame generation (DLSS-G)");
                gpu.reset = false;
                return !reset;
            }
            // Fall through to the FSR path; the bridge stays idle until it succeeds.
        }
        let reset = reset || gpu.fsr.is_none();
        let fsr = gpu.fsr.get_or_insert_with(|| {
            let interpolation = interpolation::Interpolation::new(device, gpu.size);
            Fsr {
                flow: optical_flow::OpticalFlow::new(device, gpu.size),
                output_group: post.bind(device, interpolation.output()),
                interpolation,
            }
        });
        fsr.flow.dispatch(
            device,
            queue,
            encoder,
            &gpu.world,
            reset,
            timing.as_mut().map(|(t, _)| &mut **t),
        );
        span(&mut timing, encoder, "Frame generation (optical flow)");
        fsr.interpolation.dispatch(
            device,
            queue,
            encoder,
            &interpolation::Inputs {
                color: &gpu.world,
                depth: &camera.depth,
                render_size: camera.render_size,
                jitter: camera.jitter,
                reproject,
                object_motion: &object_motion,
                near: camera.near,
                far: camera.far,
                fov_y: camera.fov_y,
                reset,
                flow_vectors: fsr.flow.vectors(),
                scene_change: fsr.flow.scene_change(),
            },
            timing.as_mut().map(|(t, _)| &mut **t),
        );
        span(&mut timing, encoder, "Frame generation (interpolation)");
        gpu.reset = false;
        if debug() == Some("motion") {
            post.finish_with(encoder, generated, &post.bind(device, &object_motion), None);
            return true;
        }
        if reset {
            post.finish_with(encoder, generated, &gpu.world_group, None);
        } else {
            post.finish_with(encoder, generated, &fsr.output_group, None);
        }
        span(&mut timing, encoder, "Frame generation (made frame)");
        !reset
    }

    /// Hand this frame's pair to the present thread, after the frame's submission: the
    /// made frame (when `made`) goes first.
    pub fn present(&mut self, real: wgpu::Texture, generated: wgpu::Texture, made: bool) {
        let now = Instant::now();
        if let Some(last) = self.last {
            // A real frame's time, smoothed: the made frame shows for half of it.
            let dt = (now - last).min(Duration::from_millis(250));
            self.interval = self.interval.mul_f32(0.9) + dt.mul_f32(0.1);
        }
        self.last = Some(now);
        self.real += 1;
        self.made += if made && !generated_only() {
            self.generated_count as u32
        } else {
            0
        };
        if now - self.shown_since >= Duration::from_millis(500) {
            self.made_share = Some(self.made as f32 / self.real as f32);
            (self.real, self.made, self.shown_since) = (0, 0, now);
        }
        let Some(send) = self.presenter.as_ref().and_then(|p| p.send.as_ref()) else {
            return;
        };
        let mut frames = Vec::new();
        let mut spare = Vec::new();
        if generated_only() {
            frames.push(generated);
            spare.push(real);
            spare.append(&mut self.extra_frames);
        } else if made {
            frames.push(generated);
            for (i, t) in self.extra_frames.drain(..).enumerate() {
                if i + 1 < self.generated_count {
                    frames.push(t);
                } else {
                    spare.push(t);
                }
            }
            frames.push(real);
        } else {
            frames.push(real);
            spare.push(generated);
            spare.append(&mut self.extra_frames);
        }
        let pair = Pair {
            frames,
            spare,
            interval: self.interval,
            queued_at: Instant::now(),
        };
        let _ = send.send(pair);
    }
}

/// A frame to draw into, like the swapchain's.
fn frame(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Frame generation frame"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &config.view_formats,
    })
}

/// One real frame's frames for the present thread: `first` (the made frame) now,
/// `second` (the real one) half an `interval` later; `spare` goes straight back.
struct Pair {
    frames: Vec<wgpu::Texture>,
    spare: Vec<wgpu::Texture>,
    interval: Duration,
    queued_at: Instant,
}

/// The thread that copies frames to the swapchain and presents them on time.
struct Presenter {
    config: wgpu::SurfaceConfiguration,
    send: Option<mpsc::SyncSender<Pair>>,
    free: mpsc::Receiver<wgpu::Texture>,
    /// The surface gave something other than a frame: the renderer takes it back.
    failed: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Presenter {
    fn start(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: Arc<wgpu::Surface<'static>>,
        window: Arc<winit::window::Window>,
        config: &wgpu::SurfaceConfiguration,
        frames_per_group: usize,
    ) -> Self {
        // One pair waiting at most: the renderer then waits as for a swapchain image.
        let low_latency = low_latency();
        let (send, receive) = mpsc::sync_channel::<Pair>(usize::from(!low_latency));
        let (give, free) = mpsc::channel();
        for _ in 0..if low_latency { 2 } else { 3 } * frames_per_group {
            let _ = give.send(frame(device, config));
        }
        let (device, queue) = (device.clone(), queue.clone());
        let vsync = config.present_mode == wgpu::PresentMode::Fifo;
        let failed = Arc::new(AtomicBool::new(false));
        let failed_here = failed.clone();
        let thread = thread::Builder::new()
            .name("Frame generation present".into())
            .spawn(move || {
                let show = |texture: &wgpu::Texture| {
                    if failed_here.load(Ordering::Relaxed) {
                        return;
                    }
                    // Anything but a frame (outdated, lost, occluded...): the renderer
                    // stops this thread and acquires itself, handling it as it does
                    // without frame generation. A suboptimal frame still shows.
                    let target = match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(t) => t,
                        wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                            failed_here.store(true, Ordering::Relaxed);
                            t
                        }
                        _ => {
                            failed_here.store(true, Ordering::Relaxed);
                            return;
                        }
                    };
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_texture_to_texture(
                        texture.as_image_copy(),
                        target.texture.as_image_copy(),
                        target.texture.size(),
                    );
                    queue.submit([encoder.finish()]);
                    if vsync {
                        window.pre_present_notify();
                    }
                    queue.present(target);
                };
                let trace = std::env::var_os("HORIZON_FRAMEGEN_LATENCY_LOG").is_some();
                let mut sampled = Instant::now();
                let (mut groups, mut queued_ms, mut oldest_ms) = (0u32, 0.0f64, 0.0f64);
                while let Ok(pair) = receive.recv() {
                    let age = pair.queued_at.elapsed().as_secs_f64() * 1000.0;
                    groups += 1;
                    queued_ms += age;
                    oldest_ms = oldest_ms.max(age);
                    if trace && sampled.elapsed() >= Duration::from_secs(1) {
                        log::info!("FG CPU presentation queue: avg {:.2} ms, max {:.2} ms, low_latency={low_latency}", queued_ms / f64::from(groups), oldest_ms);
                        sampled = Instant::now();
                        (groups, queued_ms, oldest_ms) = (0, 0.0, 0.0);
                    }
                    let step = pair.interval / (pair.frames.len() as u32);
                    let start = Instant::now();
                    for (i, texture) in pair.frames.into_iter().enumerate() {
                        let due = start + step * (i as u32);
                        if let Some(wait) = due.checked_duration_since(Instant::now()) {
                            thread::sleep(wait);
                        }
                        show(&texture);
                        let _ = give.send(texture);
                    }
                    for spare in pair.spare {
                        let _ = give.send(spare);
                    }
                }
            })
            .expect("spawn the frame generation present thread");
        Self {
            config: config.clone(),
            send: Some(send),
            free,
            failed,
            thread: Some(thread),
        }
    }

    /// The surface failed it or the thread died (a panic).
    fn failed(&self) -> bool {
        self.failed.load(Ordering::Relaxed) || self.thread.as_ref().is_none_or(|t| t.is_finished())
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.send = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesher::Vertex;
    use glam::Vec3;

    fn vertex(pos: Vec3) -> Vertex {
        Vertex {
            pos: pos.into(),
            ..bytemuck::Zeroable::zeroed()
        }
    }

    /// A swing lands halfway; a hand still on screen while the camera turns stays still;
    /// another item shows as it is.
    #[test]
    fn the_made_frame_s_hand_is_halfway_on_screen() {
        let mut fg = FrameGen::new();
        let mut out = Vec::new();
        fg.halfway_hand(
            &[vertex(Vec3::new(0.0, 0.0, -1.0))],
            Mat4::IDENTITY,
            &mut out,
        );
        fg.halfway_hand(
            &[vertex(Vec3::new(0.4, 0.0, -1.0))],
            Mat4::IDENTITY,
            &mut out,
        );
        assert!((Vec3::from(out[1].pos) - Vec3::new(0.2, 0.0, -1.0)).length() < 1e-6);
        // The camera turned: the vertex turned with it, the same on screen.
        let turned = Mat4::from_rotation_y(0.3);
        let on_screen = Vec3::new(0.4, 0.0, -1.0);
        let pos = turned.inverse().transform_point3(on_screen);
        fg.halfway_hand(&[vertex(pos)], turned, &mut out);
        assert!((Vec3::from(out[2].pos) - pos).length() < 1e-6);
        let mut other = vertex(Vec3::ZERO);
        other.uv = [0.5, 0.5];
        fg.halfway_hand(&[other], turned, &mut out);
        assert_eq!(out[3].pos, [0.0; 3]);
    }
}
