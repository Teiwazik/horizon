//! What native mods may ask of Vulkan, and the raw handles they get back.
//!
//! Before the instance and the device are created the renderer collects a
//! [`VulkanRequest`] from the mods (horizon-client-render's `native_vulkan_device`
//! hook). Each extension or feature in it is enabled only when the driver offers
//! it: what is missing is logged and left out, never an error. Once the device
//! exists, [`handles`], [`image`], [`view`] and [`format`] hand out its raw
//! Vulkan objects, so the unsafe code stays in this crate. Every function here is
//! a no-op on other backends; with an empty request the device opens exactly as
//! [`crate::open_vulkan_device`] (or wgpu's own `request_device`) opens it.
use std::ffi::CStr;
use std::sync::Mutex;

/// What the mods ask Vulkan to enable. Fields may be added: the struct is `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanRequest {
    /// Instance extensions, such as `c"VK_KHR_get_surface_capabilities2"`.
    pub instance_extensions: Vec<&'static CStr>,
    /// Device extensions, such as `c"VK_KHR_push_descriptor"`.
    pub device_extensions: Vec<&'static CStr>,
    /// Timeline semaphores. wgpu enables them whenever the GPU has them; asking only makes
    /// [`VulkanGranted::timeline_semaphore`] say whether they are there.
    pub timeline_semaphore: bool,
    /// `VK_KHR_present_id` with its `presentId` feature turned on.
    pub present_id: bool,
}

impl VulkanRequest {
    /// Nothing asked for: the instance and the device open as they would with no mod.
    pub fn is_empty(&self) -> bool {
        self.instance_extensions.is_empty() && !self.asks_device()
    }

    fn asks_device(&self) -> bool {
        !self.device_extensions.is_empty() || self.timeline_semaphore || self.present_id
    }
}

/// What of a [`VulkanRequest`] the driver offered and the engine enabled.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VulkanGranted {
    /// The requested instance extensions the instance was created with.
    pub instance_extensions: Vec<&'static CStr>,
    /// The requested device extensions the device was created with.
    pub device_extensions: Vec<&'static CStr>,
    /// Timeline semaphores were asked for and the device has them.
    pub timeline_semaphore: bool,
    /// `VK_KHR_present_id` and its feature were asked for and enabled.
    pub present_id: bool,
}

/// The raw Vulkan objects of a wgpu device, as Vulkan's handle values. They are valid
/// while the `wgpu::Device` lives; the engine owns them: never destroy them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanHandles {
    /// `VkInstance`.
    pub instance: u64,
    /// `VkPhysicalDevice`.
    pub physical_device: u64,
    /// `VkDevice`.
    pub device: u64,
    /// The `VkQueue` wgpu submits on.
    pub queue: u64,
    /// That queue's family.
    pub queue_family_index: u32,
    /// That queue's index in its family.
    pub queue_index: u32,
    /// `vkGetInstanceProcAddr`, the loader's: cast it to `PFN_vkGetInstanceProcAddr`.
    pub get_instance_proc_addr: usize,
    /// Every device extension the device was created with (wgpu's own too).
    pub device_extensions: Vec<&'static CStr>,
}

/// The `VkImage` of a wgpu texture and the `VkFormat` it was created with.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VulkanImage {
    /// `VkImage`.
    pub image: u64,
    /// `VkFormat`, as its raw value.
    pub format: i32,
}

/// The requested names that are offered, each once, in the order they were requested.
/// What is not offered is logged and left out.
pub fn grant(what: &str, offered: &[&CStr], requested: &[&'static CStr]) -> Vec<&'static CStr> {
    let mut granted: Vec<&'static CStr> = Vec::new();
    for &name in requested {
        if granted.contains(&name) {
            continue;
        }
        if offered.contains(&name) {
            granted.push(name);
        } else {
            log::info!("Vulkan {what} extension {name:?} asked by a mod is not offered here");
        }
    }
    granted
}

/// The adapter the last Vulkan device was opened from (its formats) and what was granted.
struct State {
    adapter: Option<wgpu::Adapter>,
    granted: VulkanGranted,
}

static STATE: Mutex<State> = Mutex::new(State {
    adapter: None,
    granted: VulkanGranted {
        instance_extensions: Vec::new(),
        device_extensions: Vec::new(),
        timeline_semaphore: false,
        present_id: false,
    },
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// What the last Vulkan instance and device were granted, once a device is open.
pub fn granted() -> Option<VulkanGranted> {
    let state = state();
    state.adapter.as_ref()?;
    Some(state.granted.clone())
}

/// A Vulkan instance created with the offered ones of `names`, when `descriptor` asks
/// for Vulkan alone and `names` is not empty. `None` leaves creating it to wgpu.
pub fn create_instance(
    descriptor: &wgpu::InstanceDescriptor,
    names: &[&'static CStr],
) -> Option<wgpu::Instance> {
    if names.is_empty() || descriptor.backends != wgpu::Backends::VULKAN {
        return None;
    }
    imp::create_instance(descriptor, names)
}

/// Opens a Vulkan device with what `request` asks for, where offered (and HDR's
/// metadata extension, as [`crate::open_vulkan_device`] does). `None` leaves opening
/// to wgpu, as with no mod. Remembers the adapter for [`image`] and [`format`].
pub fn open_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
    request: &VulkanRequest,
) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
    if adapter.get_info().backend != wgpu::Backend::Vulkan {
        return None;
    }
    {
        let mut state = state();
        state.adapter = Some(adapter.clone());
        state.granted.device_extensions.clear();
        state.granted.timeline_semaphore = false;
        state.granted.present_id = false;
    }
    if !request.asks_device() {
        return no_mod_device(adapter, desc);
    }
    imp::open_device(adapter, desc, request)
}

#[cfg(not(target_os = "macos"))]
fn no_mod_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
    crate::open_vulkan_device(adapter, desc)
}

#[cfg(target_os = "macos")]
fn no_mod_device(
    _adapter: &wgpu::Adapter,
    _desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
    None
}

/// The raw handles of `device`; `None` unless it is a Vulkan device.
pub fn handles(device: &wgpu::Device) -> Option<VulkanHandles> {
    imp::handles(device)
}

/// The `VkImage` and `VkFormat` of `texture`; `None` unless it is a Vulkan texture.
pub fn image(texture: &wgpu::Texture) -> Option<VulkanImage> {
    let image = imp::image(texture)?;
    Some(VulkanImage {
        image,
        format: format(texture.format())?,
    })
}

/// The `VkImageView` of `view`; `None` unless it is a Vulkan view.
pub fn view(view: &wgpu::TextureView) -> Option<u64> {
    imp::view(view)
}

/// The `VkFormat` wgpu uses for `format` on the open device's adapter (a view may have
/// another format than its texture).
pub fn format(format: wgpu::TextureFormat) -> Option<i32> {
    let adapter = state().adapter.clone()?;
    imp::format(&adapter, format)
}

#[cfg(not(target_os = "macos"))]
mod imp {
    use super::{VulkanHandles, VulkanRequest, grant, state};
    use ash::vk::{self, Handle};
    use std::ffi::CStr;
    use wgpu::hal::api::Vulkan;

    pub fn create_instance(
        descriptor: &wgpu::InstanceDescriptor,
        names: &[&'static CStr],
    ) -> Option<wgpu::Instance> {
        let mut granted = Vec::new();
        // Safety: wgpu's own instance creation; the callback only adds extensions the
        // loader reports.
        let raw = unsafe {
            wgpu::hal::vulkan::Instance::init_with_callback(
                &wgpu::hal::InstanceDescriptor {
                    name: "Horizon",
                    flags: descriptor.flags,
                    memory_budget_thresholds: descriptor.memory_budget_thresholds,
                    backend_options: descriptor.backend_options.clone(),
                    telemetry: None,
                    // Vulkan reads no display handle while it creates the instance.
                    display: None,
                },
                Some(Box::new(|args| {
                    let offered = args
                        .entry
                        .enumerate_instance_extension_properties(None)
                        .unwrap_or_default();
                    let offered: Vec<&CStr> = offered
                        .iter()
                        .filter_map(|e| e.extension_name_as_c_str().ok())
                        .collect();
                    granted = grant("instance", &offered, names);
                    for &name in &granted {
                        if !args.extensions.contains(&name) {
                            args.extensions.push(name);
                        }
                    }
                })),
            )
        };
        match raw {
            Ok(raw) => {
                state().granted.instance_extensions = granted;
                // Safety: a Vulkan instance wgpu-hal made.
                Some(unsafe { wgpu::Instance::from_hal::<Vulkan>(raw) })
            }
            Err(e) => {
                log::warn!("Vulkan instance with the mods' extensions: {e}; opening without");
                None
            }
        }
    }

    pub fn open_device(
        adapter: &wgpu::Adapter,
        desc: &wgpu::DeviceDescriptor<'_>,
        request: &VulkanRequest,
    ) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
        let mut granted_extensions = Vec::new();
        let mut present_id_on = false;
        let timeline;
        let open = {
            // Safety: only read and opened from here; dropped before wgpu takes it back.
            let hal = unsafe { adapter.as_hal::<Vulkan>() }?;
            let shared = hal.shared_instance();
            let instance = shared.raw_instance();
            let physical = hal.raw_physical_device();
            // Safety: the adapter's own physical device on its own instance.
            let offered = unsafe { instance.enumerate_device_extension_properties(physical) }
                .unwrap_or_default();
            let offered: Vec<&CStr> = offered
                .iter()
                .filter_map(|e| e.extension_name_as_c_str().ok())
                .collect();
            let wanted = grant("device", &offered, &request.device_extensions);
            let hdr = offered.contains(&ash::ext::hdr_metadata::NAME);
            let api = shared
                .instance_api_version()
                .min(hal.physical_device_capabilities().properties().api_version);
            // Vulkan 1.2 makes timeline semaphores core and required, and the extension
            // requires its feature; wgpu turns them on in both cases.
            timeline = request.timeline_semaphore
                && (api >= vk::API_VERSION_1_2
                    || offered.contains(&ash::khr::timeline_semaphore::NAME));
            let mut present_id = vk::PhysicalDevicePresentIdFeaturesKHR::default();
            if request.present_id
                && offered.contains(&ash::khr::present_id::NAME)
                && shared.instance_api_version() >= vk::API_VERSION_1_1
            {
                let mut features =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut present_id);
                // Safety: a 1.1 instance's query of its own physical device.
                unsafe { instance.get_physical_device_features2(physical, &mut features) };
            } else if request.present_id {
                log::info!("Vulkan present id asked by a mod is not offered here");
            }
            let wants_present_id = present_id.present_id == vk::TRUE;
            // Safety: what wgpu's own request_device does, with offered extensions and
            // a supported feature added.
            unsafe {
                hal.open_with_callback(
                    desc.required_features,
                    &desc.required_limits,
                    &desc.memory_hints,
                    Some(Box::new(
                        |args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
                            let mut add = |name: &'static CStr| {
                                if !args.extensions.contains(&name) {
                                    args.extensions.push(name);
                                }
                            };
                            if hdr {
                                add(ash::ext::hdr_metadata::NAME);
                            }
                            for &name in &wanted {
                                add(name);
                            }
                            if wants_present_id {
                                add(ash::khr::present_id::NAME);
                                *args.create_info = (*args.create_info).push_next(&mut present_id);
                                present_id_on = true;
                            }
                            granted_extensions = wanted.clone();
                        },
                    )),
                )
            }
            .ok()?
        };
        // Safety: opened from this adapter with exactly `desc`'s features.
        let opened = unsafe { adapter.create_device_from_hal(open, desc) };
        if opened.is_ok() {
            let mut state = state();
            state.granted.device_extensions = granted_extensions;
            state.granted.timeline_semaphore = timeline;
            state.granted.present_id = present_id_on;
        }
        Some(opened)
    }

    pub fn handles(device: &wgpu::Device) -> Option<VulkanHandles> {
        // Safety: read-only access to handles wgpu keeps alive while the guard lives.
        let hal = unsafe { device.as_hal::<Vulkan>() }?;
        let shared = hal.shared_instance();
        Some(VulkanHandles {
            instance: shared.raw_instance().handle().as_raw(),
            physical_device: hal.raw_physical_device().as_raw(),
            device: hal.raw_device().handle().as_raw(),
            queue: hal.raw_queue().as_raw(),
            queue_family_index: hal.queue_family_index(),
            queue_index: hal.queue_index(),
            get_instance_proc_addr: shared.entry().static_fn().get_instance_proc_addr as usize,
            device_extensions: hal.enabled_device_extensions().to_vec(),
        })
    }

    pub fn image(texture: &wgpu::Texture) -> Option<u64> {
        // Safety: read-only; the image lives as long as the texture.
        let hal = unsafe { texture.as_hal::<Vulkan>() }?;
        // Safety: the handle is only read.
        Some(unsafe { hal.raw_handle() }.as_raw())
    }

    pub fn view(view: &wgpu::TextureView) -> Option<u64> {
        // Safety: read-only; the view lives as long as the wgpu view.
        let hal = unsafe { view.as_hal::<Vulkan>() }?;
        // Safety: the handle is only read.
        Some(unsafe { hal.raw_handle() }.as_raw())
    }

    pub fn format(adapter: &wgpu::Adapter, format: wgpu::TextureFormat) -> Option<i32> {
        // Safety: a read-only lookup on the adapter.
        let hal = unsafe { adapter.as_hal::<Vulkan>() }?;
        Some(hal.texture_format_as_raw(format).as_raw())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{VulkanHandles, VulkanRequest};
    use std::ffi::CStr;

    pub fn create_instance(
        _descriptor: &wgpu::InstanceDescriptor,
        _names: &[&'static CStr],
    ) -> Option<wgpu::Instance> {
        None
    }
    pub fn open_device(
        _adapter: &wgpu::Adapter,
        _desc: &wgpu::DeviceDescriptor<'_>,
        _request: &VulkanRequest,
    ) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
        None
    }
    pub fn handles(_device: &wgpu::Device) -> Option<VulkanHandles> {
        None
    }
    pub fn image(_texture: &wgpu::Texture) -> Option<u64> {
        None
    }
    pub fn view(_view: &wgpu::TextureView) -> Option<u64> {
        None
    }
    pub fn format(_adapter: &wgpu::Adapter, _format: wgpu::TextureFormat) -> Option<i32> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_offered_extensions_are_granted_once_in_request_order() {
        let offered = [c"VK_KHR_b", c"VK_KHR_a", c"VK_KHR_c"];
        let requested = [c"VK_KHR_c", c"VK_HORIZON_missing", c"VK_KHR_a", c"VK_KHR_c"];
        assert_eq!(
            grant("device", &offered, &requested),
            vec![c"VK_KHR_c", c"VK_KHR_a"]
        );
    }

    #[test]
    fn nothing_offered_grants_nothing_without_failing() {
        assert!(grant("instance", &[], &[c"VK_HORIZON_missing"]).is_empty());
    }

    /// A Vulkan adapter of this machine, if it has one (tests skip without a GPU).
    #[cfg(not(target_os = "macos"))]
    fn vulkan_adapter() -> Option<wgpu::Adapter> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(descriptor);
        pollster::block_on(instance.request_adapter(&Default::default())).ok()
    }

    /// What the adapter's physical device offers, read straight from the driver.
    #[cfg(not(target_os = "macos"))]
    fn offered(adapter: &wgpu::Adapter) -> Vec<std::ffi::CString> {
        // Safety: read-only queries of the adapter's own physical device.
        let hal = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }.unwrap();
        let properties = unsafe {
            hal.shared_instance()
                .raw_instance()
                .enumerate_device_extension_properties(hal.raw_physical_device())
        }
        .unwrap();
        properties
            .iter()
            .filter_map(|e| e.extension_name_as_c_str().ok().map(CStr::to_owned))
            .collect()
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn a_device_gets_the_offered_extensions_asked_for_and_skips_the_rest() {
        let Some(adapter) = vulkan_adapter() else {
            eprintln!("no Vulkan adapter: skipped");
            return;
        };
        let offered = offered(&adapter);
        let asked = [
            c"VK_HORIZON_not_a_real_extension",
            c"VK_KHR_push_descriptor",
            c"VK_EXT_memory_budget",
            c"VK_KHR_shader_clock",
        ];
        let mut request = VulkanRequest::default();
        request.device_extensions = asked.to_vec();
        let (device, _queue) = open_device(&adapter, &Default::default(), &request)
            .expect("a Vulkan adapter opens here")
            .expect("an extension that is not offered is no error");
        let handles = handles(&device).unwrap();
        for name in asked {
            assert_eq!(
                handles.device_extensions.contains(&name),
                offered.iter().any(|o| o.as_c_str() == name),
                "{name:?} enabled exactly when the driver offers it"
            );
        }
        assert!(handles.instance != 0 && handles.physical_device != 0 && handles.device != 0);
        assert!(handles.queue != 0 && handles.get_instance_proc_addr != 0);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let image = image(&texture).unwrap();
        assert_ne!(image.image, 0);
        assert_eq!(image.format, ash::vk::Format::R8G8B8A8_UNORM.as_raw());
        assert_ne!(view(&texture.create_view(&Default::default())).unwrap(), 0);
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn with_nothing_asked_the_device_opens_as_without_mods() {
        let Some(adapter) = vulkan_adapter() else {
            eprintln!("no Vulkan adapter: skipped");
            return;
        };
        let enabled = |device: &wgpu::Device| {
            let mut names = handles(device).unwrap().device_extensions;
            names.sort();
            names
        };
        let (plain, _q) = crate::open_vulkan_device(&adapter, &Default::default())
            .unwrap()
            .unwrap();
        let (hooked, _q) = open_device(&adapter, &Default::default(), &VulkanRequest::default())
            .unwrap()
            .unwrap();
        assert_eq!(enabled(&hooked), enabled(&plain));
        assert_eq!(hooked.features(), plain.features());
        assert!(VulkanRequest::default().is_empty());
    }
}
