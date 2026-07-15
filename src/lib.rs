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

// ADR 0003 keystone: headless orchestrator (scene + store + camera). See app_core.rs.
pub mod app_core;
pub mod assets;
pub mod block_palette;
// ADR 0011 G0: the brick-field BUILD (two-layer boundary set → sorted BrickRecords +
// R8 sculpted-brick atlas), wired to nothing — parity-gated ahead of the G1 raymarch.
pub mod brick_field;
// ADR 0011 G1: the minimal brick raymarch display sink (block DDA + record binary
// search + sculpted voxel DDA, residency-miss contract, per-sample MSAA depth).
pub mod brick_raymarch;
pub mod chunk_cache;
// The display subsystem: the pure per-edit routing policy for the two display pipelines
// (cuboid mesh + brick raymarch). The DisplayOrchestrator state machine joins it later.
pub mod display;
// ADR 0003 bottom layer: dependency-free geometry primitives + the streaming quantum.
pub mod core_geom;
pub mod chunk_storage;
// ADR 0003 Phase C: the linear inverse-command stack behind undo/redo. See command.rs.
pub mod command;
pub mod cuboid;
pub mod cuboid_mesh;
pub mod debug_clouds;
pub mod disk_chunk_store;
pub mod gpu;
// ADR 0003 Phase C: the single serializable mutation boundary (Intent → apply_intent).
pub mod intent;
pub mod panel;
pub mod renderer;
pub mod scene;
pub mod settings;
// ADR 0003 §3i (Slice 2a): the sketch → extrude → volume producer, alongside SdfShape.
pub mod sketch;
pub mod spatial_index;
// ADR 0003 data layer: residency + per-chunk resolve + bound-region reads. See store.rs.
pub mod store;
pub mod texture_atlas;
// ADR 0010 E2: the OFF-by-default boundary-aware two-layer chunk store + block
// classifier (coarse + microblock + seam flags), proven bit-exact vs the dense store.
pub mod two_layer_store;
// ADR 0003 §3f(0) (Slice 2+): the parametric blocks/voxels units parser core.
pub mod units;
pub mod vox_export;
pub mod voxel;
// The background workers, grouped: the generic drain-to-latest/supersede/panic-catch
// Worker in `workers::mod`, with the geometry / diameter / brick / scan domain workers
// as its submodules.
pub mod workers;

#[cfg(test)]
mod windowed_resolve_tests;

// ADR 0010 E1: the standalone exactness parity for the conservative cell-interval
// bound primitive (VoxelProducer::cell_field_interval) + the CSG interval composition.
#[cfg(test)]
mod cell_interval_parity_tests;

pub use app_core::{
    default_replay_seed_scene, replay_intent_script, AppCore, RebuildOutcome, RebuildOutput,
};
pub use store::{ChunkCacheKey, ChunkResolveCache, Store};
pub use brick_field::{
    build_brick_field, build_brick_field_all_blocks, build_brick_field_with_tiles,
    pack_clipmap_level_keys, pack_world_block_key,
    read_back_brick_atlas, unpack_world_block_key, upload_brick_atlas,
    upload_brick_cell_key_atlas, BrickCellKeyTile, BrickFieldBuild,
    BrickFieldUpdate, BrickPayload, BrickRecord, ClipmapLevel, ClipmapPyramid,
    IncrementalBrickField, SculptedAtlasGeometry, SculptedAtlasPayload,
    SculptedCellKeyAtlasGeometry, SculptedCellKeyAtlasPayload,
    CELL_KEY_TEXEL_BYTES, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
    CLIPMAP_LEVEL_3_BLOCKS_PER_CELL,
};
pub use brick_raymarch::{
    cpu_brick_hit_material, cpu_march_brick_field, cpu_march_brick_field_counted,
    cpu_march_levels_counted, cpu_march_exact_occupancy,
    pack_gpu_records, BrickGpuRecord,
    BrickMarchFrame, BrickRaymarchRenderer, CpuMarchHit, NON_RESIDENT_ATLAS_SLOT,
};
pub use workers::brick::{
    build_brick_rebuild, spawn_brick_worker, BrickDisplayInstall, BrickRebuildOutcome,
    BrickRebuildRequest, BrickRebuildResult, BrickWorker,
};
pub use display::orchestrator::{DisplayOrchestrator, DisplayRefreshContext};
pub use display::routing::{
    brick_display_handover, brick_patch_in_place, route_brick_rebuild, route_geometry_rebuild,
    route_mesh_build, BrickDisplayHandover, BrickRebuildAction, EditShape, GenerationTracker,
    MeshBuildRoute, RebuildRoute, ASYNC_REBUILD_CHUNK_THRESHOLD,
};
pub use chunk_storage::{compress, decompress, CompressedChunk, Occupancy, SparseCell};
pub use disk_chunk_store::{DiskChunkStore, DiskChunkStoreStats};
pub use cuboid_mesh::{build_cuboid_mesh, CuboidMesh, CuboidMeshRenderer};
pub use workers::geometry::{
    build_geometry, spawn_geometry_worker, GeometryRebuildRequest, GeometryRebuildResult,
    GeometryWorker,
};
pub use workers::diameter::{
    spawn_diameter_worker, DiameterRequest, DiameterResult, DiameterWorker,
};
pub use workers::export::{
    spawn_vox_export_worker, VoxExportRequest, VoxExportResult, VoxExportSummary, VoxExportWorker,
};
pub use workers::Worker;
pub use texture_atlas::{AtlasSubRect, MaterialAtlas};
pub use debug_clouds::DebugCloudField;
pub use camera::{
    adjacent_face, chrome_zone_left_click_action, classify_cube_point,
    nearest_equivalent_theta, ArrowDir, ChromeClickAction, CubeChromeZone, CubeFace, CubeRect,
    HomeView, OrbitCamera, ProjectionMode,
    RollDir, SnapTween, ViewCubeElement, CUBE_FACES, POLE_EPSILON,
};
pub use gpu::GpuContext;
pub use intent::{Intent, IntentEffect, NodeSpec};
pub use core_geom::MaterialChoice;
pub use panel::{
    build_panel, ExportPanelState, LayerRange, PanelResponse,
    PanelState,
};
pub use assets::{CubeFaceSlot, FaceProvenance, FaceTextures};
pub use renderer::{
    create_depth_view, create_msaa_color_view, InfiniteGridRenderer, LayerBand, MaterialSource,
    OnionFogParams, PointsRenderer, SceneGridRenderer, TransformGizmoRenderer, ViewCubeRenderer,
    DEPTH_FORMAT, MSAA_SAMPLE_COUNT, VIEW_CUBE_VIEWPORT_PIXELS,
};
pub use renderer::procedural_material_average_color;
pub use scene::{
    AssemblyDef, CombineOp, DefId, Node, NodeBuilder, NodeContent, NodeId, NodePath, NodeTransform,
    Part, Point, RegionBlocks, Scene,
};
pub use settings::AppConfig;
pub use two_layer_store::{
    stream_vox_occupancy, streamed_widest_run_in_band, BlockClassification, MicroblockGeometry,
    SeamSolidity, TwoLayerChunk, TwoLayerResidentCache, TwoLayerStore,
};
// The dense whole-region resolve oracle is compile-gated out of production builds
// (see the proof chapter's "Oracles" section, `docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
pub use two_layer_store::resolve_region_two_layer;
pub use sketch::{Operation, PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
pub use spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
pub use vox_export::{VoxExport, VoxExportBuilder};
pub use voxel::{
    widest_run_in_band_over_chunks, GeometryParams, RecentreVoxels, SdfShape, ShapeKind, Voxel,
    VoxelGrid, VoxelProducer,
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

/// A ViewCube right-click context-menu item the user chose this frame (#13
/// Step 3). The windowed caller executes it after `run_egui_frame` returns; egui
/// draws the menu and swallows its own clicks, so these never leak to the
/// left-click snap path. `OrthographicToggle` is handled INSIDE `run_egui_frame`
/// (it just flips `panel_state.projection_mode`, the same field the side panel
/// binds, keeping the two in sync), so it is not surfaced here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewCubeMenuRequest {
    /// "Home" — snap to the saved home view.
    Home,
    /// "Fit" — frame the model.
    Fit,
    /// "Set current as home" — capture the live camera as the home view.
    SetHome,
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
    /// The ViewCube context-menu item chosen this frame (#13 Step 3), if any. The
    /// caller runs Home/Fit/SetHome; the ortho toggle is applied in-place to
    /// `panel_state.projection_mode` and is not reported here.
    pub cube_menu_request: Option<ViewCubeMenuRequest>,
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
    grid_z: u32,
    measured_diameter: u32,
    export: panel::ExportPanelState,
    palette: &block_palette::BlockPalette,
    raw_input: egui::RawInput,
    size_in_pixels: [u32; 2],
    pixels_per_point: f32,
    // #13 Step 3: position (in egui points) of an open ViewCube right-click
    // context menu, or `None`. Drawn inside the egui pass so egui swallows the
    // menu's clicks. The menu clears this (`= None`) on selection or click-away.
    // The headless `shot` path passes `&mut None` (no menu).
    cube_context_menu_at: &mut Option<egui::Pos2>,
) -> PreparedEguiFrame {
    let mut panel_response = PanelResponse::default();
    let mut cube_menu_request: Option<ViewCubeMenuRequest> = None;
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
        panel_response = build_panel(ui, panel_state, grid_z, measured_diameter, export, palette);
        // After both panels have been shown inside the root ui, the remaining
        // space is the central viewport.
        central_rect_points = ui.available_rect_before_wrap();

        // #13 Step 3: the ViewCube right-click context menu. Drawn as a floating
        // egui Area at the press position when open. egui owns its hit-testing, so
        // its buttons swallow the click (no leak to the snap path). A click on an
        // item runs the action and closes the menu; a click anywhere OUTSIDE the
        // menu (detected via the area response) closes it without acting.
        if let Some(menu_pos_px) = *cube_context_menu_at {
            // `cube_context_menu_at` is stored in PHYSICAL pixels (the winit cursor
            // space); egui positions in points, so divide by pixels_per_point.
            let menu_pos = egui::pos2(
                menu_pos_px.x / pixels_per_point,
                menu_pos_px.y / pixels_per_point,
            );
            let context = ui.ctx().clone();
            let area = egui::Area::new(egui::Id::new("view_cube_context_menu"))
                .order(egui::Order::Foreground)
                .fixed_pos(menu_pos)
                .show(&context, |ui| {
                    egui::Frame::menu(ui.style()).show(ui, |ui| {
                        ui.set_min_width(180.0);
                        if ui.button("Home").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::Home);
                        }
                        if ui.button("Fit").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::Fit);
                        }
                        // Ortho ↔ Perspective: toggle the SAME field the side panel
                        // binds, so the menu and the panel stay in sync.
                        let projection_label = match panel_state.projection_mode {
                            ProjectionMode::Perspective => "Orthographic",
                            ProjectionMode::Orthographic => "Perspective",
                        };
                        if ui.button(projection_label).clicked() {
                            panel_state.projection_mode = match panel_state.projection_mode {
                                ProjectionMode::Perspective => ProjectionMode::Orthographic,
                                ProjectionMode::Orthographic => ProjectionMode::Perspective,
                            };
                            *cube_context_menu_at = None;
                        }
                        ui.separator();
                        if ui.button("Set current as home").clicked() {
                            cube_menu_request = Some(ViewCubeMenuRequest::SetHome);
                        }
                    });
                });
            // Close on selection (an item set a request or toggled projection).
            if cube_menu_request.is_some() {
                *cube_context_menu_at = None;
            }
            // Click-away: only a PRIMARY (left) click that lands OUTSIDE the menu's
            // rect closes it. #13 Step 6.5: the previous `any_click()` also fired on
            // the SECONDARY (right) click that OPENS the menu — and on the open frame
            // egui's `interact_pos` is the cursor at the menu's very corner, which the
            // freshly-laid-out rect didn't yet count as "inside", so the menu closed
            // the same frame it appeared (the flicker). Restricting the close to a
            // primary click leaves the opening right-click alone, so the menu stays up
            // until the user picks an item or left-clicks elsewhere.
            let pointer = &context.input(|i| i.pointer.clone());
            if pointer.primary_clicked() {
                let clicked_in_menu = pointer
                    .interact_pos()
                    .map(|p| area.response.rect.contains(p))
                    .unwrap_or(false);
                if !clicked_in_menu {
                    *cube_context_menu_at = None;
                }
            }
        }
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
        cube_menu_request,
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
    /// #13 Step 6 follow-up: draw all four ViewCube rotate arrows PERSISTENTLY (set
    /// when the view is face-constrained), with the hovered one brightened. `false`
    /// (off-face view) draws no rotate arrows. Decoupled from `cube_hovered_zone` so
    /// the arrows are a standing affordance, not a hover-only reveal.
    pub cube_rotate_arrows_visible: bool,
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
    /// ADR 0012: draw the onion GHOST pass this frame. When `true`, immediately
    /// after the solid voxel draw (inside the shared MSAA pass), the engaged display
    /// path (brick raymarch when present, else the cuboid mesh) draws its translucent
    /// ghost of the voxels in the onion slabs — recentred-Z in `[onion_z_min,
    /// band_z_min) ∪ (band_z_max, onion_z_max]`. Depth-tested `Less` + alpha-blended, with
    /// depth WRITE ON so only the NEAREST ghost surface shows (blended once — a
    /// builder-independent render that matches across display paths); the solid, drawn
    /// first, still occludes the ghost. Replaces the volumetric fog pass; a band scrub is a pure uniform (brick)
    /// or thin-slab-remesh (mesh) update, never the fog atlas rebuild. The ghost
    /// uniforms/geometry must already be prepared by the renderers' `update_uniforms`.
    pub onion_ghost_active: bool,
    /// The cuboid mesh renderer — the CPU voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Draws the voxels as a box-decomposed mesh; its
    /// uniforms must already be uploaded via `CuboidMeshRenderer::update_uniforms`.
    /// Kept PERMANENTLY as the headless/no-GPU fallback + A/B reference (ADR 0011
    /// Decision 6) even when the brick path below takes the frame.
    pub cuboid_mesh: &'a cuboid_mesh::CuboidMeshRenderer,
    /// ADR 0011 G1: the brick raymarch display sink. `Some` replaces the cuboid
    /// mesh DRAW for this frame (single ported-producer scenes on the GPU path) —
    /// the pass runs in the same MSAA pass and writes ray-hit depth, so every
    /// overlay/fog/cube/egui pass after it composites unchanged. `None` keeps the
    /// mesh path (multi-producer, loaded materials, debug modes, no-GPU builds).
    pub brick_raymarch: Option<&'a brick_raymarch::BrickRaymarchRenderer>,
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

        // The voxel model: the brick raymarch (ADR 0011 G1) when engaged, else the
        // cuboid mesh path. When a VS block is applied the mesh path binds the
        // block's 6-layer D2Array so it textures per-face; no applied block →
        // `None` keeps the procedural-atlas path. The brick pass writes ray-hit
        // depth into this same MSAA depth attachment, so the depth-tested overlays
        // below (and the fog's depth-stop) composite identically on both paths.
        let loaded_material = match material {
            renderer::MaterialSource::Loaded(bind_group) => Some(bind_group),
            renderer::MaterialSource::Procedural(_) => None,
        };
        if let Some(brick_raymarch) = overlays.brick_raymarch {
            // ADR 0011 G2: a loaded VS block now textures the raymarch too — bind the
            // block's 6-layer D2Array at group(2) so solid hits shade per-face by the
            // owner's lattice rule (the brick renderer's `loaded_material_active` flag,
            // set alongside its uniforms, selects that branch). `None` binds the dummy.
            brick_raymarch.draw(&mut voxel_pass, loaded_material);
        } else {
            overlays.cuboid_mesh.draw(&mut voxel_pass, loaded_material);
        }

        // ADR 0012 (H1) — the onion GHOST pass. Immediately after the SOLID band draw,
        // in the SAME MSAA pass, the engaged display path ghosts the voxels in the onion
        // slabs (recentred-Z outside the band, within ±onion_depth). Depth-tested
        // `Less` + alpha-blended, with depth WRITE ON so only the nearest ghost surface
        // shows (a builder-independent render); the just-drawn solid still occludes it. The
        // brick ghost is two per-slab raymarches; the mesh ghost is two thin per-slab
        // meshes — both shaded flat translucent (the retired fog haze's hue). This
        // REPLACES the former volumetric fog pass (Pass 1a below, now always `None`).
        if overlays.onion_ghost_active {
            if let Some(brick_raymarch) = overlays.brick_raymarch {
                brick_raymarch.draw_ghost(&mut voxel_pass);
            } else {
                overlays.cuboid_mesh.draw_ghost(&mut voxel_pass);
            }
        }

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
            overlays.cube_rotate_arrows_visible,
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
