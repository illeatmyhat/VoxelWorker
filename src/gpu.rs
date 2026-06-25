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
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface,
            })
            .await
            .expect("no suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("voxel-worker device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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
}
