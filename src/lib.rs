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
pub mod chunk_cache;
pub mod chunk_storage;
pub mod cuboid;
pub mod cuboid_mesh;
pub mod debug_clouds;
pub mod disk_chunk_store;
pub mod frustum;
pub mod gpu;
pub mod panel;
pub mod renderer;
pub mod scan_worker;
pub mod scene;
pub mod settings;
pub mod spatial_index;
pub mod texture_atlas;
pub mod vox_export;
pub mod voxel;

pub use chunk_cache::{ChunkCacheKey, ChunkResolveCache};
pub use chunk_storage::{compress, decompress, CompressedChunk, Occupancy, SparseCell};
pub use disk_chunk_store::{DiskChunkStore, DiskChunkStoreStats};
pub use cuboid_mesh::{build_cuboid_mesh, CuboidMesh, CuboidMeshRenderer};
pub use texture_atlas::{AtlasSubRect, MaterialAtlas};
pub use debug_clouds::DebugCloudField;
pub use camera::{
    adjacent_face, classify_cube_point, compass_heading_to_theta, nearest_equivalent_theta,
    ArrowDir, CubeChromeZone, CubeFace, CubeRect, Heading, HomeView, OrbitCamera, ProjectionMode,
    RollDir, SnapTween, ViewCubeElement, CUBE_FACES, POLE_EPSILON,
};
pub use gpu::GpuContext;
pub use panel::{
    build_panel, GeometryParams, LayerRange, MaterialChoice, PanelResponse,
    PanelState,
};
pub use assets::{CubeFaceSlot, FaceProvenance, FaceTextures};
pub use renderer::{
    build_per_chunk_fog_occupancy, create_depth_view, create_msaa_color_view, ChunkFogVolume,
    FogMode, InfiniteGridRenderer, LayerBand, MaterialSource, OnionFogParams, PointsRenderer,
    SceneGridRenderer,
    TransformGizmoRenderer,
    OnionFogRenderer, PerChunkFogOccupancy, ViewCubeRenderer, DEPTH_FORMAT,
    MSAA_SAMPLE_COUNT, VIEW_CUBE_VIEWPORT_PIXELS,
};
pub use renderer::procedural_material_average_color;
pub use scene::{
    AssemblyDef, CombineOp, DefId, Node, NodeContent, NodePath, NodeTransform, Part, Point,
    RegionBlocks, Scene,
};
pub use settings::AppConfig;
pub use spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
pub use vox_export::VoxExport;
pub use voxel::{
    widest_run_in_band_over_chunks, SdfShape, ShapeKind, Voxel, VoxelGrid, VoxelProducer,
};

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
    /// The central 3D viewport rect in PHYSICAL PIXELS (issue #25): `[x, y, w, h]`
    /// = the window/target area LEFT of the right side panel and ABOVE the bottom
    /// palette dock. Derived from egui's post-panel `available_rect` × the frame's
    /// `pixels_per_point`, then clamped into the target. The caller computes the
    /// camera aspect from `w/h` and confines the 3D pass (voxels, gizmo, fog, view
    /// cube) to this rect, so the model is centred in the VISIBLE 3D area instead
    /// of the whole window (which the panels would otherwise cover).
    pub viewport_px: [u32; 4],
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
    // Issue #25: the central 3D viewport rect, in egui points. `build_panel` shows
    // the right side panel + bottom palette dock INSIDE `ui`; whatever room those
    // panels leave is the central area where the 3D scene should be centred. We
    // read it AFTER the panels are laid out (`available_rect`), so a resized panel
    // moves the viewport with it.
    let mut central_rect_points = egui::Rect::from_min_size(
        egui::pos2(0.0, 0.0),
        egui::vec2(size_in_pixels[0] as f32, size_in_pixels[1] as f32),
    );
    let full_output = bridge.context.run_ui(raw_input, |ui| {
        panel_response = build_panel(ui, panel_state, grid_y, measured_diameter, palette);
        // After both panels have been shown inside the root ui, the remaining
        // space is the central viewport.
        central_rect_points = ui.available_rect_before_wrap();
    });

    // Convert the central rect from egui points to physical pixels, then clamp it
    // inside the target so the viewport/scissor below are always valid.
    let viewport_px = {
        let to_px = |value: f32| (value * pixels_per_point).round();
        let left = to_px(central_rect_points.min.x).max(0.0) as u32;
        let top = to_px(central_rect_points.min.y).max(0.0) as u32;
        let right = to_px(central_rect_points.max.x).max(0.0) as u32;
        let bottom = to_px(central_rect_points.max.y).max(0.0) as u32;
        let x = left.min(size_in_pixels[0]);
        let y = top.min(size_in_pixels[1]);
        // Always leave at least a 1×1 viewport so set_viewport never gets 0 dims.
        let width = right.min(size_in_pixels[0]).saturating_sub(x).max(1);
        let height = bottom.min(size_in_pixels[1]).saturating_sub(y).max(1);
        [x, y, width, height]
    };

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
        viewport_px,
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
    pub gizmo: Option<&'a renderer::TransformGizmoRenderer>,
    pub view_cube: Option<&'a renderer::ViewCubeRenderer>,
    /// The ViewCube chrome zone under the cursor (#13 Step 2). Drives which hover
    /// arrows the cube draws and which glyph is highlighted. `None` = nothing
    /// hovered (the normal render: compass + Home/Fit only, no arrows).
    pub cube_hovered_zone: Option<camera::CubeChromeZone>,
    /// The per-object block lattice + floor grid (issue #29 S3). Drawn in the MSAA
    /// pass (depth-tested) before the gizmo. The renderer's per-frame batch already
    /// holds only the grid-enabled nodes' lines (master AND per-object), so the draw
    /// is self-gating; `None` skips it entirely.
    pub scene_grid: Option<&'a renderer::SceneGridRenderer>,
    /// The world reference AXES (issue #29 S5): every visible Point's axis lines.
    /// Drawn in the MSAA pass (depth-tested) with the scene-grid line batch, so opaque
    /// voxels occlude them. Its batch already holds only the visible Points' enabled
    /// axes (self-gating); `None` skips it entirely (the `shot` default, so the
    /// existing goldens are unchanged).
    pub points: Option<&'a renderer::PointsRenderer>,
    /// The analytic infinite reference grid (issue #29 Points fast-follow): every
    /// visible Point's enabled PLANES, drawn as fullscreen ray-plane passes in the
    /// MSAA pass after the voxels (depth-tested via `frag_depth`), so opaque objects
    /// occlude the grid. Replaces the old finite tiled-line ground plane. Self-gating
    /// (no enabled plane → no draw); `None` skips it (the `shot` default).
    pub infinite_grid: Option<&'a renderer::InfiniteGridRenderer>,
    /// Onion-skin volumetric fog (issue #12): when `Some`, a fullscreen SDF
    /// raymarch composites a faint haze over the resolved scene for the layers
    /// around the displayed band. `None` when onion skin is off. Its uniforms must
    /// already be uploaded via `OnionFogRenderer::update`.
    pub onion_fog: Option<&'a renderer::OnionFogRenderer>,
    /// The cuboid mesh renderer — the sole voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Draws the voxels as a box-decomposed mesh; its
    /// uniforms must already be uploaded via `CuboidMeshRenderer::update_uniforms`.
    pub cuboid_mesh: &'a cuboid_mesh::CuboidMeshRenderer,
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

        // Issue #25: confine the 3D geometry to the central viewport rect (the
        // window minus the side panel + bottom dock). The MSAA target was still
        // CLEARED to the workshop colour across the WHOLE target above, so any
        // sliver not covered by egui isn't garbage; only the draws are scissored.
        let [viewport_x, viewport_y, viewport_width, viewport_height] = prepared.viewport_px;
        voxel_pass.set_viewport(
            viewport_x as f32,
            viewport_y as f32,
            viewport_width as f32,
            viewport_height as f32,
            0.0,
            1.0,
        );
        voxel_pass.set_scissor_rect(viewport_x, viewport_y, viewport_width, viewport_height);

        // The cuboid mesh path is the sole voxel renderer (part of #20). When a VS
        // block is applied, hand it the block's 6-layer D2Array bind group so it
        // textures the model per-face (selecting the layer by the face normal); no
        // applied block → `None` keeps the procedural-atlas path.
        let loaded_material = match material {
            renderer::MaterialSource::Loaded(bind_group) => Some(bind_group),
            renderer::MaterialSource::Procedural(_) => None,
        };
        overlays.cuboid_mesh.draw(&mut voxel_pass, loaded_material);

        // Per-object block lattice + floor grid (issue #29 S3): same MSAA pass,
        // depth-tested so the solid model occludes them (a scaffold around/under it).
        if let Some(scene_grid) = overlays.scene_grid {
            scene_grid.draw(&mut voxel_pass);
        }

        // Analytic infinite reference grid (issue #29 Points fast-follow): the visible
        // Points' enabled PLANES as fullscreen ray-plane passes, AFTER the voxels (so
        // the depth buffer holds the model) and depth-tested via `frag_depth` so opaque
        // objects occlude the grid. Replaces the old finite tiled-line ground plane —
        // the grid now extends smoothly to the horizon with no finite edge / near-clip
        // cutoff at shallow angles, fading with distance.
        if let Some(infinite_grid) = overlays.infinite_grid {
            infinite_grid.draw(&mut voxel_pass);
        }

        // World reference AXES (issue #29 S5): the visible Points' axis lines, same
        // MSAA pass, depth-tested so opaque voxels occlude them (subtle frame markers
        // behind/under the model, not an overlay on top).
        if let Some(points) = overlays.points {
            points.draw(&mut voxel_pass);
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
        onion_fog.draw(device, &mut encoder, target_view, depth_view, prepared.viewport_px);
    }

    // === Pass 1b: view cube into a scissored top-left corner (its own depth).
    // Drawn after the 3D resolve, before egui (ARCHITECTURE.md §6 layering).
    if let Some(view_cube) = overlays.view_cube {
        view_cube.draw(
            device,
            queue,
            &mut encoder,
            target_view,
            overlays.target_width,
            overlays.target_height,
            prepared.viewport_px,
            overlays.cube_hovered_zone,
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
