//! VoxelWorker — native Rust port of the Vintage Story chiseling planner.
//!
//! Milestone 1 establishes the rendering foundation shared by both the windowed
//! application (`src/main.rs`) and the headless screenshot harness
//! (`src/bin/shot.rs`):
//!
//!   * A render-target-agnostic frame function ([`render_frame`]) that paints
//!     into any [`wgpu::TextureView`]. It knows nothing about winit or surfaces,
//!     so the same code draws the on-screen surface texture and the offscreen
//!     capture texture — guaranteeing the screenshot matches the window.
//!   * A single egui panel builder ([`build_panel`]) used by both paths so the
//!     captured frame is identical to the live one.
//!   * The colour identity from ARCHITECTURE.md §8 (warm-dark workshop).

pub mod camera;
pub mod gpu;
pub mod panel;
pub mod renderer;
pub mod voxel;

pub use camera::OrbitCamera;
pub use gpu::GpuContext;
pub use panel::{build_panel, PanelState};
pub use renderer::{create_depth_view, VoxelRenderer, DEPTH_FORMAT};
pub use voxel::{SdfShape, ShapeKind, VoxelGrid, VoxelProducer};

/// Surface / offscreen colour format used everywhere in the project.
///
/// Using the same sRGB format for the windowed surface and the headless capture
/// texture keeps the screenshot identical to the window (Hard requirement #9).
pub const COLOR_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// The warm-dark "workshop" clear colour (ARCHITECTURE.md §8).
///
/// These are *linear* component values handed to wgpu; with an sRGB render
/// target the GPU encodes them back to sRGB on write, so the perceived colour is
/// a warm near-black with a faint copper cast.
pub const WORKSHOP_CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.030,
    g: 0.024,
    b: 0.018,
    a: 1.0,
};

/// Everything needed to translate egui output into wgpu draw calls, plus the
/// persistent egui context. Lives for the whole program; reused every frame.
pub struct EguiPaintBridge {
    pub context: egui::Context,
    pub renderer: egui_wgpu::Renderer,
}

impl EguiPaintBridge {
    /// Build the bridge for a given render-target format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let renderer = egui_wgpu::Renderer::new(
            device,
            target_format,
            egui_wgpu::RendererOptions {
                // egui feathers its own AA; 3D MSAA is a separate concern (M4).
                msaa_samples: 1,
                // The shared render pass carries a depth attachment for the voxel
                // pass (M2). egui doesn't write depth, but its pipeline must be
                // compatible with the pass's attachment formats, so declare it.
                depth_stencil_format: Some(renderer::DEPTH_FORMAT),
                dithering: true,
                predictable_texture_filtering: false,
            },
        );
        Self {
            context: egui::Context::default(),
            renderer,
        }
    }
}

/// The fully-prepared egui draw data for one frame.
///
/// Produced by [`run_egui_frame`] and consumed by [`render_frame`]. Keeping it
/// in a struct lets the windowed path interleave winit-specific work (feeding
/// `platform_output` back to the window) between the two steps.
pub struct PreparedEguiFrame {
    pub paint_jobs: Vec<egui::ClippedPrimitive>,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
    pub textures_to_free: Vec<egui::TextureId>,
    pub platform_output: egui::PlatformOutput,
}

/// Run the egui pass for one frame: build the panel, upload changed textures to
/// the GPU, and tessellate the UI into paint jobs.
///
/// This is the render-target-agnostic half of egui integration. Both binaries
/// call it; the windowed binary supplies `raw_input` from `egui_winit`, the
/// headless binary builds `raw_input` by hand.
pub fn run_egui_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    panel_state: &mut PanelState,
    raw_input: egui::RawInput,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
) -> PreparedEguiFrame {
    let full_output = bridge
        .context
        .run_ui(raw_input, |ui| build_panel(ui, panel_state));

    for (texture_id, image_delta) in &full_output.textures_delta.set {
        bridge
            .renderer
            .update_texture(device, queue, *texture_id, image_delta);
    }

    let paint_jobs = bridge
        .context
        .tessellate(full_output.shapes, pixels_per_point);

    PreparedEguiFrame {
        paint_jobs,
        screen_descriptor: egui_wgpu::ScreenDescriptor {
            size_in_pixels,
            pixels_per_point,
        },
        textures_to_free: full_output.textures_delta.free,
        platform_output: full_output.platform_output,
    }
}

/// Render a complete frame into `target_view`.
///
/// This is the render-target-agnostic core (Hard requirement #2): it accepts a
/// colour [`wgpu::TextureView`] plus the prepared egui data and has no knowledge
/// of winit or surfaces. The windowed binary passes the surface texture's view;
/// the headless binary passes the offscreen capture texture's view.
///
/// Milestone 2 draws the instanced voxel cubes (with a depth attachment) into
/// this same render pass *before* the egui pass composites the panel on top.
/// `voxel_renderer` and `depth_view` are render-target-agnostic: the window and
/// the headless capture pass their own depth view sized to the same target.
pub fn render_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    voxel_renderer: &renderer::VoxelRenderer,
    prepared: &PreparedEguiFrame,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("voxel-worker frame encoder"),
    });

    // egui's buffer upload happens on the same encoder; the returned command
    // buffers must be submitted before (or alongside) the main encoder.
    let egui_upload_commands = bridge.renderer.update_buffers(
        device,
        queue,
        &mut encoder,
        &prepared.paint_jobs,
        &prepared.screen_descriptor,
    );

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel-worker main pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(WORKSHOP_CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // === Milestone 2 ===
        // Instanced voxel cubes are drawn into the same render pass, using the
        // depth attachment above and the camera uniform bind group. The egui
        // pass below then composites the panel on top. Nothing about the target
        // view changes — voxels paint into the surface and the capture
        // identically.
        voxel_renderer.draw(&mut render_pass);

        // egui wants a RenderPass<'static>; forget_lifetime converts it.
        bridge.renderer.render(
            &mut render_pass.forget_lifetime(),
            &prepared.paint_jobs,
            &prepared.screen_descriptor,
        );
    }

    queue.submit(egui_upload_commands.into_iter().chain(std::iter::once(encoder.finish())));

    for texture_id in &prepared.textures_to_free {
        bridge.renderer.free_texture(texture_id);
    }
}
