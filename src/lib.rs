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

pub mod assets;
pub mod block_palette;
pub mod camera;
pub mod debug_clouds;
pub mod gpu;
pub mod panel;
pub mod renderer;
pub mod scan_worker;
pub mod settings;
pub mod vox_export;
pub mod voxel;

pub use debug_clouds::DebugCloudField;
pub use camera::{
    nearest_equivalent_theta, CubeFace, OrbitCamera, ProjectionMode, SnapTween, ViewCubeElement,
    CUBE_FACES, POLE_EPSILON,
};
pub use gpu::GpuContext;
pub use panel::{build_panel, GeometryParams, LayerRange, MaterialChoice, PanelResponse, PanelState};
pub use assets::{CubeFaceSlot, FaceProvenance, FaceTextures};
pub use renderer::{
    create_depth_view, create_msaa_color_view, GizmoRenderer, GridLatticeRenderer, LayerBand,
    MaterialSource, OnionFogParams, OnionFogRenderer, ViewCubeRenderer, VoxelRenderer, DEPTH_FORMAT,
    MSAA_SAMPLE_COUNT, VIEW_CUBE_VIEWPORT_PIXELS,
};
pub use renderer::procedural_material_average_color;
pub use settings::AppConfig;
pub use vox_export::VoxExport;
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
                // egui feathers its own AA at 1 sample. M4 splits the frame into a
                // 4× MSAA 3D pass (resolved) followed by a separate egui pass that
                // loads the resolved single-sample target — so egui's pipeline
                // needs neither MSAA nor a depth attachment.
                msaa_samples: 1,
                depth_stencil_format: None,
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
    /// What the user changed in the panel this frame (M3): drives the geometry
    /// rebuild + camera auto-frame in the caller.
    pub panel_response: PanelResponse,
}

/// Run the egui pass for one frame: build the panel, upload changed textures to
/// the GPU, and tessellate the UI into paint jobs.
///
/// This is the render-target-agnostic half of egui integration. Both binaries
/// call it; the windowed binary supplies `raw_input` from `egui_winit`, the
/// headless binary builds `raw_input` by hand.
#[allow(clippy::too_many_arguments)]
pub fn run_egui_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    panel_state: &mut PanelState,
    grid_y: u32,
    measured_diameter: u32,
    palette: &block_palette::BlockPalette,
    raw_input: egui::RawInput,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
) -> PreparedEguiFrame {
    let mut panel_response = PanelResponse::default();
    let full_output = bridge.context.run_ui(raw_input, |ui| {
        panel_response = build_panel(ui, panel_state, grid_y, measured_diameter, palette);
    });

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
        panel_response,
    }
}

/// Render a complete frame into `target_view`.
///
/// This is the render-target-agnostic core (Hard requirement #2): it accepts a
/// resolved single-sample colour [`wgpu::TextureView`] plus the prepared egui
/// data and has no knowledge of winit or surfaces. The windowed binary passes
/// the surface texture's view; the headless binary passes the offscreen capture
/// texture's view.
///
/// Milestone 4 restructures the frame into two passes:
///   1. **3D MSAA pass** — the instanced voxel cubes are drawn into a 4-sample
///      colour texture (`msaa_color_view`) with a 4-sample depth attachment
///      (`depth_view`) and resolved into `target_view` (the single-sample
///      surface / capture texture). `material` selects the bound texture and
///      `grid_overlay_enabled` was already folded into the uniforms by the
///      caller.
///   2. **egui pass** — egui renders at 1 sample directly onto the RESOLVED
///      `target_view` with `LoadOp::Load`, compositing the panel on top.
///
/// `msaa_color_view` and `depth_view` are render-target-agnostic: the window and
/// the headless capture pass their own 4-sample textures sized to the same target.
/// Optional M5 overlays for [`render_frame`]: the origin gizmo (drawn in the
/// MSAA pass, depth-test off) and the corner view cube (its own scissored pass).
/// Each is `None` when its Display toggle is off, so the caller controls
/// visibility without the renderer caring.
pub struct FrameOverlays<'a> {
    pub gizmo: Option<&'a renderer::GizmoRenderer>,
    pub view_cube: Option<&'a renderer::ViewCubeRenderer>,
    /// The block lattice + fine floor grid (M8). Drawn in the MSAA pass (depth-
    /// tested) before the gizmo. `show_lattice`/`show_floor` reflect the toggles;
    /// `None` skips both.
    pub grid_lattice: Option<&'a renderer::GridLatticeRenderer>,
    pub show_lattice: bool,
    pub show_floor: bool,
    /// Face-orientation debug mode: the voxel cubes are drawn with the cull-off
    /// debug pipeline (colour by outward normal + back-facing marker). Must match
    /// the `debug_face_mode` flag passed to `VoxelRenderer::update_uniforms`.
    pub debug_face_mode: bool,
    /// Onion-skin volumetric fog (issue #12): when `Some`, a fullscreen SDF
    /// raymarch composites a faint haze over the resolved scene for the layers
    /// around the displayed band. `None` when onion skin is off. Its uniforms must
    /// already be uploaded via `OnionFogRenderer::update`.
    pub onion_fog: Option<&'a renderer::OnionFogRenderer>,
    /// Target dimensions (needed to place the view-cube corner viewport).
    pub target_width: u32,
    pub target_height: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn render_frame(
    bridge: &mut EguiPaintBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_view: &wgpu::TextureView,
    msaa_color_view: &wgpu::TextureView,
    depth_view: &wgpu::TextureView,
    voxel_renderer: &renderer::VoxelRenderer,
    material: renderer::MaterialSource,
    overlays: &FrameOverlays,
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

    // === Pass 1: 3D voxel pass at 4× MSAA, resolved into the single-sample target.
    {
        let mut voxel_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel-worker 3D msaa pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_color_view,
                // Resolve the multisampled colour into the single-sample target.
                resolve_target: Some(target_view),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(WORKSHOP_CLEAR_COLOR),
                    // The multisampled texture is transient; we only keep the
                    // resolved result. Discarding it is the cheaper store.
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    // Stored (not discarded) so the onion-skin fog pass can sample
                    // this MSAA depth to stop its raymarch at opaque surfaces.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        voxel_renderer.draw(&mut voxel_pass, material, overlays.debug_face_mode);

        // Block lattice + fine floor grid (M8): same MSAA pass, depth-tested so
        // the solid model occludes them (a scaffold around/under it).
        if let Some(grid_lattice) = overlays.grid_lattice {
            grid_lattice.draw(&mut voxel_pass, overlays.show_lattice, overlays.show_floor);
        }

        // Origin gizmo: same MSAA pass, after the voxels, depth-test OFF so it
        // shows through the solid model (ARCHITECTURE.md §5/§6).
        if let Some(gizmo) = overlays.gizmo {
            gizmo.draw(&mut voxel_pass);
        }
    }

    // === Pass 1a: onion-skin volumetric fog (issue #12). A fullscreen raymarch of
    // the resolved voxel grid (as a 3D cloud density) that composites a faint haze
    // over the resolved scene for the layers around the displayed band. Runs after
    // the 3D resolve and before the view cube/egui (so the corner cube and panel
    // aren't fogged). Depth-tested against the 3D pass's MSAA depth so the displayed
    // opaque slice occludes the onion layers behind it (like Minecraft's clouds).
    if let Some(onion_fog) = overlays.onion_fog {
        onion_fog.draw(device, &mut encoder, target_view, depth_view);
    }

    // === Pass 1b: view cube into a scissored top-left corner (its own depth).
    // Drawn after the 3D resolve, before egui (ARCHITECTURE.md §6 layering).
    if let Some(view_cube) = overlays.view_cube {
        view_cube.draw(
            device,
            &mut encoder,
            target_view,
            overlays.target_width,
            overlays.target_height,
        );
    }

    // === Pass 2: egui at 1 sample onto the RESOLVED target (load, don't clear).
    {
        let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("voxel-worker egui pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // egui wants a RenderPass<'static>; forget_lifetime converts it.
        bridge.renderer.render(
            &mut egui_pass.forget_lifetime(),
            &prepared.paint_jobs,
            &prepared.screen_descriptor,
        );
    }

    queue.submit(egui_upload_commands.into_iter().chain(std::iter::once(encoder.finish())));

    for texture_id in &prepared.textures_to_free {
        bridge.renderer.free_texture(texture_id);
    }
}
