//! NVIDIA Reflex (VK_NV_low_latency2) support for Vulkan.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReflexMode {
    #[default]
    Off,
    On,
    OnBoost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatencyMarker {
    SimulationStart,
    SimulationEnd,
    RendersubmitStart,
    RendersubmitEnd,
    PresentStart,
    PresentEnd,
    InputSample,
    OutOfBandPresentStart,
    OutOfBandPresentEnd,
}

#[cfg(not(target_os = "macos"))]
struct ReflexContext {
    device_handle: ash::vk::Device,
    failed: bool,
    pending: bool,
    loader: ash::nv::low_latency2::Device,
    semaphore: Option<std::sync::Arc<SleepSemaphore>>,
    value: u64,
}

// The wait releases CONTEXT so the FG presenter can submit latency markers.
// Keep the semaphore AND its device alive if shutdown races that wait.
#[cfg(not(target_os = "macos"))]
struct SleepSemaphore {
    handle: ash::vk::Semaphore,
    device: wgpu::Device,
    raw: ash::Device,
    value: std::sync::atomic::AtomicU64,
}

#[cfg(not(target_os = "macos"))]
impl Drop for SleepSemaphore {
    fn drop(&mut self) {
        let value = self.value.load(std::sync::atomic::Ordering::Relaxed);
        if value > 0 {
            let semaphores = [self.handle];
            let values = [value];
            let info = ash::vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            if unsafe { self.raw.wait_semaphores(&info, 1_000_000_000) }
                == Err(ash::vk::Result::TIMEOUT)
            {
                eprintln!(
                    "Horizon Reflex: pending sleep did not finish; retaining its device until process exit"
                );
                std::mem::forget(self.device.clone());
                return;
            }
        }
        unsafe {
            self.raw.destroy_semaphore(self.handle, None);
        }
    }
}

#[cfg(not(target_os = "macos"))]
static CONTEXT: std::sync::Mutex<Option<ReflexContext>> = std::sync::Mutex::new(None);

#[cfg(not(target_os = "macos"))]
fn get_context(
    device: &wgpu::Device,
    hal_device: &wgpu::hal::vulkan::Device,
) -> Option<std::sync::MutexGuard<'static, Option<ReflexContext>>> {
    if !hal_device
        .enabled_device_extensions()
        .iter()
        .any(|&name| name == ash::nv::low_latency2::NAME)
    {
        return None;
    }
    let raw = hal_device.raw_device().handle();
    let mut guard = CONTEXT.lock().ok()?;
    if let Some(ref ctx) = *guard {
        if ctx.device_handle == raw {
            return Some(guard);
        }
    }
    if let Some(old) = guard.take() {
        retire(old);
    }
    let loader = ash::nv::low_latency2::Device::new(
        hal_device.shared_instance().raw_instance(),
        hal_device.raw_device(),
    );
    // Create timeline semaphore required by vkLatencySleepNV
    let mut type_info = ash::vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(ash::vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let create_info = ash::vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
    let semaphore = unsafe { hal_device.raw_device().create_semaphore(&create_info, None) }.ok();
    let semaphore = semaphore.map(|handle| {
        std::sync::Arc::new(SleepSemaphore {
            handle,
            device: device.clone(),
            raw: hal_device.raw_device().clone(),
            value: std::sync::atomic::AtomicU64::new(0),
        })
    });

    log::info!(
        "Horizon Reflex: VK_NV_low_latency2 initialized, timeline semaphore={}",
        semaphore.is_some()
    );
    *guard = Some(ReflexContext {
        device_handle: raw,
        failed: false,
        pending: false,
        loader,
        semaphore,
        value: 0,
    });
    Some(guard)
}

/// Whether the device has the `VK_NV_low_latency2` extension enabled.
pub fn is_supported(device: &wgpu::Device) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        use wgpu::hal::api::Vulkan;
        let Some(hal_device) = (unsafe { device.as_hal::<Vulkan>() }) else {
            return false;
        };
        hal_device
            .enabled_device_extensions()
            .iter()
            .any(|&name| name == ash::nv::low_latency2::NAME)
    }
    #[cfg(target_os = "macos")]
    {
        let _ = device;
        false
    }
}

/// Sets the latency sleep mode on the active swapchain.
pub fn set_latency_sleep_mode(
    device: &wgpu::Device,
    surface: &wgpu::Surface<'_>,
    mode: ReflexMode,
) {
    #[cfg(not(target_os = "macos"))]
    {
        use wgpu::hal::api::Vulkan;
        let (Some(hal_device), Some(hal_surface)) =
            (unsafe { device.as_hal::<Vulkan>() }, unsafe {
                surface.as_hal::<Vulkan>()
            })
        else {
            return;
        };
        let Some(guard) = get_context(device, &hal_device) else {
            return;
        };
        let Some(ref ctx) = *guard else {
            return;
        };
        let Some(swapchain) = hal_surface.raw_native_swapchain() else {
            return;
        };
        if swapchain == ash::vk::SwapchainKHR::null() {
            return;
        }
        let (low_latency_mode, low_latency_boost) = match mode {
            ReflexMode::Off => (false, false),
            ReflexMode::On => (true, false),
            ReflexMode::OnBoost => (true, true),
        };
        let info = ash::vk::LatencySleepModeInfoNV::default()
            .low_latency_mode(low_latency_mode)
            .low_latency_boost(low_latency_boost)
            .minimum_interval_us(0);
        match unsafe { ctx.loader.set_latency_sleep_mode(swapchain, Some(&info)) } {
            Ok(()) => log::info!("Horizon Reflex: sleep mode {mode:?}"),
            Err(error) => log::warn!("Horizon Reflex: setting sleep mode failed: {error:?}"),
        }
    }
    #[cfg(target_os = "macos")]
    let _ = (device, surface, mode);
}

/// Sleeps the thread dynamically until the GPU is ready for the new frame.
pub fn latency_sleep(device: &wgpu::Device, surface: &wgpu::Surface<'_>) {
    #[cfg(not(target_os = "macos"))]
    {
        use wgpu::hal::api::Vulkan;
        let (Some(hal_device), Some(hal_surface)) =
            (unsafe { device.as_hal::<Vulkan>() }, unsafe {
                surface.as_hal::<Vulkan>()
            })
        else {
            return;
        };
        let Some(mut guard) = get_context(device, &hal_device) else {
            return;
        };
        let Some(ref mut ctx) = *guard else {
            return;
        };
        let Some(swapchain) = hal_surface.raw_native_swapchain() else {
            return;
        };
        if swapchain == ash::vk::SwapchainKHR::null() {
            return;
        }
        // vkLatencySleepNV requires a valid timeline semaphore handle.
        // If one could not be created, skip sleeping to prevent driver crash.
        let Some(semaphore) = ctx.semaphore.clone() else {
            return;
        };
        if ctx.failed {
            return;
        }
        // A deadline miss is not a device failure. Do not issue more sleeps
        // against an outstanding signal or wait another 50 ms every frame.
        if ctx.pending {
            match unsafe { semaphore.raw.get_semaphore_counter_value(semaphore.handle) } {
                Ok(value) if value < ctx.value => return,
                Ok(_) => {
                    ctx.pending = false;
                    log::info!("Horizon Reflex: pacing resumed after delayed signal");
                }
                Err(error) => {
                    ctx.failed = true;
                    log::warn!(
                        "Horizon Reflex: counter query failed ({error:?}); pacing disabled for this device"
                    );
                    return;
                }
            }
        }
        let target_value = ctx.value + 1;
        let info = ash::vk::LatencySleepInfoNV::default()
            .signal_semaphore(semaphore.handle)
            .value(target_value);
        let res = unsafe { ctx.loader.latency_sleep(swapchain, &info) };
        if let Err(error) = res {
            ctx.failed = true;
            eprintln!("Horizon Reflex: sleep failed ({error:?}); pacing disabled for this device");
            return;
        }
        ctx.value = target_value;
        semaphore
            .value
            .store(target_value, std::sync::atomic::Ordering::Relaxed);
        // In particular, do not hold this mutex while waiting for a signal
        // whose progress can depend on FG's out-of-band presentation markers.
        drop(guard);
        let semaphores = [semaphore.handle];
        let values = [target_value];
        let wait_info = ash::vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        if let Err(error) = unsafe { semaphore.raw.wait_semaphores(&wait_info, 50_000_000) } {
            if let Ok(mut guard) = CONTEXT.lock() {
                if let Some(ctx) = guard.as_mut() {
                    if ctx
                        .semaphore
                        .as_ref()
                        .is_some_and(|current| std::sync::Arc::ptr_eq(current, &semaphore))
                    {
                        if error == ash::vk::Result::TIMEOUT {
                            ctx.pending = true;
                        } else {
                            ctx.failed = true;
                        }
                    }
                }
            }
            if error == ash::vk::Result::TIMEOUT {
                log::warn!(
                    "Horizon Reflex: sleep deadline exceeded; polling pending signal without blocking subsequent frames"
                );
            } else {
                log::warn!(
                    "Horizon Reflex: sleep wait failed ({error:?}); pacing disabled for this device"
                );
            }
        }
    }
    #[cfg(target_os = "macos")]
    let _ = (device, surface);
}

/// Marks a latency timestamp milestone for NVIDIA Reflex profiling and pacing.
pub fn set_marker(
    device: &wgpu::Device,
    surface: &wgpu::Surface<'_>,
    marker: LatencyMarker,
    frame_id: u64,
) {
    #[cfg(not(target_os = "macos"))]
    {
        use wgpu::hal::api::Vulkan;
        let (Some(hal_device), Some(hal_surface)) =
            (unsafe { device.as_hal::<Vulkan>() }, unsafe {
                surface.as_hal::<Vulkan>()
            })
        else {
            return;
        };
        let Some(guard) = get_context(device, &hal_device) else {
            return;
        };
        let Some(ref ctx) = *guard else {
            return;
        };
        let Some(swapchain) = hal_surface.raw_native_swapchain() else {
            return;
        };
        if swapchain == ash::vk::SwapchainKHR::null() {
            return;
        }
        let vk_marker = match marker {
            LatencyMarker::SimulationStart => ash::vk::LatencyMarkerNV::SIMULATION_START,
            LatencyMarker::SimulationEnd => ash::vk::LatencyMarkerNV::SIMULATION_END,
            LatencyMarker::RendersubmitStart => ash::vk::LatencyMarkerNV::RENDERSUBMIT_START,
            LatencyMarker::RendersubmitEnd => ash::vk::LatencyMarkerNV::RENDERSUBMIT_END,
            LatencyMarker::PresentStart => ash::vk::LatencyMarkerNV::PRESENT_START,
            LatencyMarker::PresentEnd => ash::vk::LatencyMarkerNV::PRESENT_END,
            LatencyMarker::InputSample => ash::vk::LatencyMarkerNV::INPUT_SAMPLE,
            LatencyMarker::OutOfBandPresentStart => {
                ash::vk::LatencyMarkerNV::OUT_OF_BAND_PRESENT_START
            }
            LatencyMarker::OutOfBandPresentEnd => ash::vk::LatencyMarkerNV::OUT_OF_BAND_PRESENT_END,
        };
        let info = ash::vk::SetLatencyMarkerInfoNV::default()
            .present_id(frame_id)
            .marker(vk_marker);
        unsafe { ctx.loader.set_latency_marker(swapchain, &info) };
    }
    #[cfg(target_os = "macos")]
    let _ = (device, surface, marker, frame_id);
}

/// Release device-local state before Vulkan device teardown, including renderer rebuilds.
pub fn shutdown(device: &wgpu::Device) {
    #[cfg(not(target_os = "macos"))]
    {
        use wgpu::hal::api::Vulkan;
        let Some(hal) = (unsafe { device.as_hal::<Vulkan>() }) else {
            return;
        };
        if let Ok(mut guard) = CONTEXT.lock() {
            if guard
                .as_ref()
                .is_some_and(|ctx| ctx.device_handle == hal.raw_device().handle())
            {
                if let Some(ctx) = guard.take() {
                    retire(ctx);
                }
            }
        }
    }
    #[cfg(target_os = "macos")]
    let _ = device;
}

#[cfg(not(target_os = "macos"))]
fn retire(ctx: ReflexContext) {
    // An in-flight CPU wait owns an Arc too; its semaphore is retired only
    // after that wait finishes, always through the original Vulkan device.
    drop(ctx);
}
