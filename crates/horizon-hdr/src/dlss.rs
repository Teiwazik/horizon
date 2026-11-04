//! DLSS's Vulkan side: the instance and device with its extensions, and NGX started on them
//! (horizon-client-render's `gfx/dlss.rs` uses them; this crate may hold unsafe code).
use std::sync::{Arc, Mutex, OnceLock};

use dlss_wgpu::{DlssSdk, FeatureSupport};

/// Horizon's NGX project id (any UUID of the application's own).
const PROJECT_ID: uuid::Uuid = uuid::Uuid::from_u128(0x6f1c_2b8e_4d3a_4e9b_a7c5_31d0_8e2f_9b64);

/// Whether DLSS was asked for (`HORIZON_DLSS=1`).
pub fn asked() -> bool {
    std::env::var_os("HORIZON_DLSS").is_some_and(|v| v == "1")
}

/// The SDK, once the device is opened with DLSS's extensions and NGX starts on it.
static SDK: OnceLock<Option<Arc<Mutex<DlssSdk>>>> = OnceLock::new();

/// A Vulkan instance with the extensions DLSS needs, when it was asked for; `None` leaves
/// creating it to wgpu.
pub fn create_instance(
    descriptor: &wgpu::InstanceDescriptor,
    window: Option<&Arc<winit::window::Window>>,
) -> Option<wgpu::Instance> {
    use winit::raw_window_handle::HasDisplayHandle;
    if !asked() || descriptor.backends != wgpu::Backends::VULKAN {
        return None;
    }
    let display = window.and_then(|w| w.display_handle().ok());
    let mut support = FeatureSupport::default();
    let mut result = Ok(());
    // Safety: wgpu's own instance creation, with DLSS's extensions added.
    let raw = unsafe {
        wgpu::hal::vulkan::Instance::init_with_callback(
            &wgpu::hal::InstanceDescriptor {
                name: "Horizon",
                flags: descriptor.flags,
                memory_budget_thresholds: descriptor.memory_budget_thresholds,
                backend_options: descriptor.backend_options.clone(),
                telemetry: None,
                display,
            },
            Some(Box::new(|mut args| {
                result = dlss_wgpu::register_instance_extensions(
                    PROJECT_ID,
                    &mut args,
                    &mut support,
                );
            })),
        )
    };
    match (raw, result) {
        (Ok(raw), Ok(())) if support.super_resolution_supported => {
            log::info!("Horizon: DLSS instance extensions enabled");
            // Safety: a Vulkan instance wgpu-hal made.
            Some(unsafe { wgpu::Instance::from_hal::<wgpu::hal::api::Vulkan>(raw) })
        }
        (Ok(_), Ok(())) => {
            log::info!("Horizon: DLSS is not supported here; TAA upscales");
            None
        }
        (Err(e), _) => {
            log::warn!("Horizon: DLSS instance: {e}");
            None
        }
        (_, Err(e)) => {
            log::warn!("Horizon: DLSS instance extensions: {e}");
            None
        }
    }
}

/// Opens the device with DLSS's extensions (and HDR's metadata one, as `horizon_hdr` does),
/// then starts NGX on it. `None` leaves opening to the caller.
pub fn open_device(
    adapter: &wgpu::Adapter,
    desc: &wgpu::DeviceDescriptor<'_>,
) -> Option<Result<(wgpu::Device, wgpu::Queue), wgpu::RequestDeviceError>> {
    use wgpu::hal::api::Vulkan;
    if !asked() {
        return None;
    }
    let mut support = FeatureSupport::default();
    let open = {
        // Safety: only read and opened from here; dropped before wgpu takes it back.
        let hal = unsafe { adapter.as_hal::<Vulkan>() }?;
        let instance = hal.shared_instance().raw_instance();
        // Safety: the adapter's own physical device on its own instance.
        let hdr = unsafe { instance.enumerate_device_extension_properties(hal.raw_physical_device()) }
            .ok()?
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::hdr_metadata::NAME));
        let mut result = Ok(());
        // Safety: what wgpu's own request_device does, with more extensions.
        let open = unsafe {
            hal.open_with_callback(
                desc.required_features,
                &desc.required_limits,
                &desc.memory_hints,
                Some(Box::new(|mut args: wgpu::hal::vulkan::CreateDeviceCallbackArgs| {
                    if hdr {
                        args.extensions.push(ash::ext::hdr_metadata::NAME);
                    }
                    result = dlss_wgpu::register_device_extensions(
                        PROJECT_ID,
                        &mut args,
                        &hal,
                        &mut support,
                    );
                })),
            )
        };
        if let Err(e) = &result {
            log::warn!("Horizon: DLSS device extensions: {e}");
            return None;
        }
        open.ok()?
    };
    // Safety: opened from this adapter with exactly `desc`'s features.
    let opened = unsafe { adapter.create_device_from_hal(open, desc) };
    if let Ok((device, _)) = &opened {
        let sdk = support
            .super_resolution_supported
            .then(|| DlssSdk::new(PROJECT_ID, device.clone()))
            .and_then(|sdk| {
                sdk.map_err(|e| log::warn!("Horizon: DLSS did not start: {e}"))
                    .ok()
            });
        if sdk.is_some() {
            log::info!("Horizon: DLSS Super Resolution ready");
        }
        let _ = SDK.set(sdk);
    }
    Some(opened)
}


/// The SDK, when NGX started on the device `open_device` opened.
pub fn sdk() -> Option<Arc<Mutex<DlssSdk>>> {
    SDK.get()?.clone()
}
