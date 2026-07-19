//! wgpu instance/adapter/device setup shared by both render paths.
//!
//! The windowed binary passes `Some(&surface)` as the compatible surface so the
//! adapter is guaranteed presentable; the headless binary passes `None` and
//! never creates a surface at all (DEV_NOTES "Headless capture").

/// Owns the wgpu objects that outlive any single frame.
pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create the instance, request an adapter and device.
    ///
    /// `compatible_surface` is `Some` for the windowed path (so the adapter can
    /// present to that surface) and `None` for the fully headless path.
    pub async fn new(compatible_surface: Option<&wgpu::Surface<'_>>) -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        Self::new_with_instance(instance, compatible_surface).await
    }

    /// Build the context from an already-created instance.
    ///
    /// The windowed path must create the instance and surface together (so the
    /// adapter can be selected as compatible with that surface), then hand the
    /// instance here. The headless path uses [`GpuContext::new`] instead.
    pub async fn new_with_instance(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Self {
        let adapter = Self::request_adapter(&instance, compatible_surface)
            .await
            .expect("no suitable GPU adapter found");

        // The GPU fog atlas (ADR 0007) packs a large single-producer scene's covering
        // chunks into one storage buffer before `copy_buffer_to_texture`. A 50×10×50-block
        // cylinder's atlas is ~200 MiB — over the DEFAULT 128 MiB storage-buffer binding
        // limit, which would validation-error the bind group (#56). Raise the storage-buffer
        // and total-buffer size limits to whatever the adapter actually supports (desktop
        // GPUs report GiBs) so large scenes stay on the GPU path instead of falling back to
        // the 26s CPU densify. Everything else keeps the conservative defaults.
        let adapter_limits = adapter.limits();
        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxel-worker device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                ..Default::default()
            })
            .await
            .expect("request_device failed");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// The single adapter-request the whole crate goes through, so
    /// [`adapter_available`] probes exactly what [`GpuContext::new`] would get.
    async fn request_adapter(
        instance: &wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Option<wgpu::Adapter> {
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface,
            })
            .await
            .ok()
    }
}

/// Whether this machine can hand out a wgpu adapter at all.
///
/// This is the RUNTIME replacement for the deleted `gpu` Cargo feature. The
/// device-dependent integration tests used to be compiled out behind
/// `#![cfg(feature = "gpu")]`, which meant a GPU-less machine did not skip them
/// — they silently vanished, and forgetting the flag made the golden suite pass
/// vacuously. Now every such test is always compiled, calls this once, and
/// skips LOUDLY (printing why) when it returns `false`.
///
/// Creates an instance and requests an adapter — no device, no surface, nothing
/// retained — so it is cheap enough to call once per test binary.
pub fn adapter_available() -> bool {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    pollster::block_on(GpuContext::request_adapter(&instance, None)).is_some()
}
