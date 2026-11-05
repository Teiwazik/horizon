//! Experimental native DLSS-G bridge. Enabled only by explicit laboratory environment.
use ash::vk::Handle;
use std::ffi::{CString, c_char, c_void};
use wgpu::hal::api::Vulkan;

#[link(name = "dl")]
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Image {
    image: u64,
    view: u64,
    format: u32,
    width: u32,
    height: u32,
    aspect: u32,
}

#[repr(C)]
struct Frame {
    color: Image,
    depth: Image,
    motion: Image,
    output: Image,
    projection: [f32; 16],
    projection_inverse: [f32; 16],
    reprojection: [f32; 16],
    reprojection_inverse: [f32; 16],
    jitter: [f32; 2],
    near: f32,
    far: f32,
    fov_y: f32,
    reset: u32,
    /// Depth holds 1.0 at the near plane (Horizon's reversed-Z projection).
    depth_inverted: u32,
    multi_frame_count: u32,
    multi_frame_index: u32,
}

pub struct Camera {
    pub projection: [f32; 16],
    pub projection_inverse: [f32; 16],
    pub reprojection: [f32; 16],
    pub reprojection_inverse: [f32; 16],
    pub jitter: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub fov_y: f32,
    pub reset: bool,
    /// Depth holds 1.0 at the near plane (Horizon's reversed-Z projection).
    pub depth_inverted: bool,
}

type Create = unsafe extern "C" fn(u64, u64, u64, u64, u32, u32) -> *mut c_void;
type Evaluate = unsafe extern "C" fn(*mut c_void, u64, *const Frame) -> u32;
type Destroy = unsafe extern "C" fn(*mut c_void);

pub struct DlssFg {
    library: *mut c_void,
    context: *mut c_void,
    create: Create,
    evaluate: Evaluate,
    destroy: Destroy,
    failed: bool,
    size: [u32; 2],
}
// The context is exclusively owned; bridge calls are serialized and its destructor waits
// for device idle before releasing GPU resources. No shared mutable access is exposed.
unsafe impl Send for DlssFg {}

impl DlssFg {
    pub fn new() -> Option<Self> {
        let path = CString::new(std::env::var("HORIZON_DLSSG_BRIDGE").ok()?).ok()?;
        // Safety: the opt-in bridge implements the documented, fixed C ABI.
        unsafe {
            let library = dlopen(path.as_ptr(), 2);
            if library.is_null() {
                log::warn!("DLSS-G: could not open the experimental bridge");
                return None;
            }
            let version = dlsym(library, c"horizon_fg_abi_version".as_ptr());
            if version.is_null()
                || std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> u32>(version)() != 2
            {
                log::warn!("DLSS-G: unsupported bridge ABI");
                dlclose(library);
                return None;
            }
            let create = dlsym(library, c"horizon_fg_create".as_ptr());
            let evaluate = dlsym(library, c"horizon_fg_evaluate".as_ptr());
            let destroy = dlsym(library, c"horizon_fg_destroy".as_ptr());
            if create.is_null() || evaluate.is_null() || destroy.is_null() {
                dlclose(library);
                return None;
            }
            Some(Self {
                library,
                context: std::ptr::null_mut(),
                create: std::mem::transmute::<*mut c_void, Create>(create),
                evaluate: std::mem::transmute::<*mut c_void, Evaluate>(evaluate),
                destroy: std::mem::transmute::<*mut c_void, Destroy>(destroy),
                failed: false,
                size: [0; 2],
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        motion: &wgpu::TextureView,
        output: &[wgpu::TextureView],
        camera: Camera,
    ) -> bool {
        if self.failed {
            return false;
        }
        // Safety: all views belong to this Vulkan device, stay alive through submission,
        // and are tracked/transited by wgpu before the foreign commands consume them.
        let Some((raw_color, raw_depth, raw_motion)) = (unsafe {
            image(color)
                .zip(image(depth))
                .zip(image(motion))
                .map(|((a, b), c)| (a, b, c))
        }) else {
            return false;
        };
        let Some(outputs) =
            (unsafe { output.iter().map(|v| image(v)).collect::<Option<Vec<_>>>() })
        else {
            return false;
        };
        if outputs.is_empty() {
            return false;
        }
        let size = [raw_color.width, raw_color.height];
        if self.size != size && !self.context.is_null() {
            // The bridge waits for pending GPU work before releasing its feature.
            unsafe {
                (self.destroy)(self.context);
            }
            self.context = std::ptr::null_mut();
        }
        self.size = size;
        let first = self.context.is_null();
        let mut frame = Frame {
            color: raw_color,
            depth: raw_depth,
            motion: raw_motion,
            output: outputs[0],
            projection: camera.projection,
            projection_inverse: camera.projection_inverse,
            reprojection: camera.reprojection,
            reprojection_inverse: camera.reprojection_inverse,
            jitter: camera.jitter,
            near: camera.near,
            far: camera.far,
            fov_y: camera.fov_y,
            reset: (camera.reset || first).into(),
            depth_inverted: camera.depth_inverted.into(),
            multi_frame_count: outputs.len() as u32,
            multi_frame_index: 1,
        };
        encoder.transition_resources(
            std::iter::empty(),
            [
                wgpu::TextureTransition {
                    texture: color.texture(),
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                },
                wgpu::TextureTransition {
                    texture: depth.texture(),
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                },
                wgpu::TextureTransition {
                    texture: motion.texture(),
                    selector: None,
                    state: wgpu::TextureUses::RESOURCE,
                },
            ]
            .into_iter()
            .chain(output.iter().map(|v| wgpu::TextureTransition {
                texture: v.texture(),
                selector: None,
                state: wgpu::TextureUses::STORAGE_READ_WRITE,
            })),
        );
        let mut native = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DLSS-G native commands"),
        });
        let status = unsafe {
            native.as_hal_mut::<Vulkan, _, _>(|commands| {
                let Some(commands) = commands else {
                    return 0xbad00001;
                };
                let raw_device = device.as_hal::<Vulkan>().expect("Vulkan views");
                let command = commands.raw_handle().as_raw();
                if self.context.is_null() {
                    self.context = (self.create)(
                        raw_device
                            .shared_instance()
                            .raw_instance()
                            .handle()
                            .as_raw(),
                        raw_device.raw_physical_device().as_raw(),
                        raw_device.raw_device().handle().as_raw(),
                        command,
                        raw_color.width,
                        raw_color.height,
                    );
                }
                if self.context.is_null() {
                    return 0xbad00001;
                }
                let mut result = 1;
                for (i, raw_output) in outputs.iter().enumerate() {
                    frame.output = *raw_output;
                    frame.multi_frame_index = i as u32 + 1;
                    result = (self.evaluate)(self.context, command, &frame);
                    if result & 0xfff00000 == 0xbad00000 {
                        break;
                    }
                }
                result
            })
        };
        let before = std::mem::replace(
            encoder,
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Horizon after DLSS-G"),
            }),
        );
        if self.context.is_null() {
            // A failed CreateFeature may already have released resources referenced by
            // its partial commands. Discard those commands instead of submitting them.
            drop(native);
            queue.submit([before.finish()]);
        } else {
            queue.submit([before.finish(), native.finish()]);
        }
        if status & 0xfff00000 == 0xbad00000 {
            log::warn!("DLSS-G native evaluation failed: {status:#x}; FSR remains available");
            self.failed = true;
            return false;
        }
        true
    }
}

unsafe fn image(view: &wgpu::TextureView) -> Option<Image> {
    let texture = view.texture();
    let (format, aspect) = match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm => (
            ash::vk::Format::R8G8B8A8_UNORM,
            ash::vk::ImageAspectFlags::COLOR,
        ),
        wgpu::TextureFormat::Rg16Float => (
            ash::vk::Format::R16G16_SFLOAT,
            ash::vk::ImageAspectFlags::COLOR,
        ),
        wgpu::TextureFormat::Depth32Float => (
            ash::vk::Format::D32_SFLOAT,
            ash::vk::ImageAspectFlags::DEPTH,
        ),
        _ => return None,
    };
    unsafe {
        // Each HAL guard holds the device's snatch lock. End one borrow before
        // taking the next; keeping both temporaries inside Image recurses the lock.
        let raw_image = { texture.as_hal::<Vulkan>()?.raw_handle().as_raw() };
        let raw_view = { view.as_hal::<Vulkan>()?.raw_handle().as_raw() };
        Some(Image {
            image: raw_image,
            view: raw_view,
            format: format.as_raw() as u32,
            aspect: aspect.as_raw(),
            width: texture.width(),
            height: texture.height(),
        })
    }
}
impl Drop for DlssFg {
    fn drop(&mut self) {
        // Safety: exclusive context ownership and the library outlives its destructor.
        unsafe {
            if !self.context.is_null() {
                (self.destroy)(self.context);
            }
            dlclose(self.library);
        }
    }
}

/// Called with no live FG/SR features, before the Vulkan device is destroyed.
pub fn shutdown() {
    let Some(bridge) = DlssFg::new() else {
        return;
    };
    unsafe {
        let symbol = dlsym(bridge.library, c"horizon_fg_shutdown".as_ptr());
        if !symbol.is_null() {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn()>(symbol)();
        }
    }
}
