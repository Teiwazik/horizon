//! NVIDIA DLSS Super Resolution in place of TAA's upscaling (a local build's `dlss` feature,
//! asked for with `HORIZON_DLSS=1`; Vulkan only). It needs NVIDIA's SDK at build time
//! (`DLSS_SDK`) and its `libnvidia-ngx-dlss.so` beside the game; without them the game keeps
//! TAA. Through `dlss_wgpu` (Bevy's, MIT/Apache).
//!
//! DLSS takes the world at its render size, its depth (reversed), its motion vectors (this
//! module draws them: the camera's through the depth, or a mob's own from frame generation's
//! motion) and the jitter, and writes the picture at the output size.

use std::sync::{Arc, Mutex};

use dlss_wgpu::super_resolution::{
    DlssSuperResolution, DlssSuperResolutionExposure, DlssSuperResolutionRenderParameters,
};
use dlss_wgpu::{DlssFeatureFlags, DlssPerfQualityMode, DlssSdk};

/// The motion vectors' format: render pixels, x and y.
const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
/// DLSS's output format (written as a storage image).
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub use horizon_hdr::dlss::{asked, create_instance, open_device};

/// DLSS's quality mode for TAA's upscale ratio (Native: DLAA).
fn mode(upscale: f32) -> DlssPerfQualityMode {
    match upscale {
        u if u <= 1.01 => DlssPerfQualityMode::Dlaa,
        u if u <= 1.55 => DlssPerfQualityMode::Quality,
        u if u <= 1.8 => DlssPerfQualityMode::Balanced,
        u if u <= 2.5 => DlssPerfQualityMode::Performance,
        _ => DlssPerfQualityMode::UltraPerformance,
    }
}

/// `world_post.wgsl`'s `DlssMotion`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionUniform {
    /// This frame's unjittered clip space to the previous frame's.
    pub reproject: [[f32; 4]; 4],
    /// xy: this frame's jitter in clip space.
    pub jitter: [f32; 4],
}

/// DLSS for one output size and mode, its motion vectors and its output.
pub struct Dlss {
    adapter: wgpu::Adapter,
    sdk: Arc<Mutex<DlssSdk>>,
    context: Option<(DlssSuperResolution, [u32; 2], DlssPerfQualityMode)>,
    output: Option<([u32; 2], wgpu::TextureView)>,
    motion: Option<([u32; 2], wgpu::TextureView)>,
    motion_pipeline: wgpu::RenderPipeline,
    motion_layout: wgpu::BindGroupLayout,
    motion_uniform: wgpu::Buffer,
    /// The context failed for this size and mode: TAA keeps it.
    failed: Option<([u32; 2], DlssPerfQualityMode)>,
}

impl Dlss {
    /// DLSS when NGX started on this device.
    pub fn new(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Option<Self> {
        let sdk = horizon_hdr::dlss::sdk()?;
        let shader = &device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Horizon DLSS motion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../dlss_motion.wgsl").into()),
        });
        let motion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Horizon DLSS motion"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Horizon DLSS motion"),
            bind_group_layouts: &[Some(&motion_layout)],
            immediate_size: 0,
        });
        let motion_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Horizon DLSS motion"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some("fs_dlss_motion"),
                compilation_options: Default::default(),
                targets: &[Some(MOTION_FORMAT.into())],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let motion_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Horizon DLSS motion"),
            size: std::mem::size_of::<MotionUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Dlss {
            adapter: adapter.clone(),
            sdk,
            context: None,
            output: None,
            motion: None,
            motion_pipeline,
            motion_layout,
            motion_uniform,
            failed: None,
        })
    }

    /// Whether DLSS can take this output size and upscale with the world at `render`: its
    /// context made (once per size and mode) and `render` within what it accepts.
    pub fn ready(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: [u32; 2],
        render: [u32; 2],
        upscale: f32,
        hdr: bool,
    ) -> bool {
        let mode = mode(upscale);
        if self.failed == Some((output, mode)) {
            return false;
        }
        if self.context.as_ref().is_none_or(|(_, o, m)| *o != output || *m != mode) {
            self.context = None;
            let mut flags = DlssFeatureFlags::LowResolutionMotionVectors
                | DlssFeatureFlags::InvertedDepth;
            if hdr {
                flags |= DlssFeatureFlags::HighDynamicRange | DlssFeatureFlags::AutoExposure;
            }
            match DlssSuperResolution::new(output, mode, flags, self.sdk.clone(), device, queue) {
                Ok(context) => {
                    log::info!(
                        "Horizon: DLSS {mode:?} {}x{} from {:?}..={:?}",
                        output[0],
                        output[1],
                        context.render_resolution_range().start(),
                        context.render_resolution_range().end(),
                    );
                    self.context = Some((context, output, mode));
                }
                Err(e) => {
                    log::warn!("Horizon: DLSS {mode:?} at {output:?}: {e}; TAA upscales");
                    self.failed = Some((output, mode));
                    return false;
                }
            }
        }
        let Some((context, ..)) = &self.context else {
            return false;
        };
        let range = context.render_resolution_range();
        let (lo, hi) = (range.start(), range.end());
        (lo[0]..=hi[0]).contains(&render[0]) && (lo[1]..=hi[1]).contains(&render[1])
    }

    fn texture(
        device: &wgpu::Device,
        label: &str,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::TextureView {
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
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }

    /// Draws the motion vectors and records DLSS: everything `encoder` holds is submitted
    /// with DLSS's commands right after it, and `encoder` starts over empty. Returns the
    /// output (at the context's size) to show, or `None` when it did not run.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        own_motion: &wgpu::TextureView,
        render: [u32; 2],
        motion: MotionUniform,
        jitter_pixels: [f32; 2],
        reset: bool,
    ) -> Option<wgpu::TextureView> {
        let (context, output_size, _) = self.context.as_mut()?;
        let output_size = *output_size;
        if self.output.as_ref().is_none_or(|(s, _)| *s != output_size) {
            self.output = Some((
                output_size,
                Self::texture(
                    device,
                    "DLSS output",
                    output_size,
                    OUTPUT_FORMAT,
                    wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
            ));
        }
        if self.motion.as_ref().is_none_or(|(s, _)| *s != render) {
            self.motion = Some((
                render,
                Self::texture(
                    device,
                    "DLSS motion vectors",
                    render,
                    MOTION_FORMAT,
                    wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                ),
            ));
        }
        let (_, output) = self.output.as_ref()?;
        let (_, vectors) = self.motion.as_ref()?;
        queue.write_buffer(&self.motion_uniform, 0, bytemuck::bytes_of(&motion));
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Horizon DLSS motion"),
            layout: &self.motion_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(own_motion),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.motion_uniform.as_entire_binding(),
                },
            ],
        });
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: vectors,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Horizon DLSS motion"),
                color_attachments: &attachments,
                ..Default::default()
            });
            pass.set_pipeline(&self.motion_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
        let parameters = DlssSuperResolutionRenderParameters {
            color,
            depth,
            motion_vectors: vectors,
            exposure: DlssSuperResolutionExposure::Automatic,
            bias: None,
            dlss_output: output,
            reset,
            jitter_offset: jitter_pixels,
            partial_texture_size: Some(render),
            motion_vector_scale: None,
        };
        match context.render(parameters, encoder, &self.adapter) {
            Ok(dlss) => {
                let before = std::mem::replace(
                    encoder,
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Horizon frame after DLSS"),
                    }),
                );
                queue.submit([before.finish(), dlss]);
                Some(output.clone())
            }
            Err(e) => {
                log::warn!("Horizon: DLSS frame: {e}");
                None
            }
        }
    }
}
