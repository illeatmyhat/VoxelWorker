//! VoxelWorker — the windowed application (the default binary's logic, now a shell LIB
//! module tree).
//!
//! winit 0.30 `ApplicationHandler` + wgpu 29 surface + egui 0.34 panel. Shows the warm-dark
//! workshop clear colour and the shared right-hand egui side panel. It uses the exact same
//! [`render_frame`]/[`run_egui_frame`] code as the headless `shot` binary, so the live window
//! and the captured PNG match.
//!
//! The thin `src/main.rs` binary just calls [`run`]. The logic is split across this module tree
//! (ADR 0016): the [`WindowedState`] struct lives here, its impl is spread over sibling files
//! (`geometry`, `workers`, `palette`, `export`, `view_cube`, `render`) as descendant modules that
//! reach its private fields, and the winit event pump (`impl ApplicationHandler for App`) lives in
//! `events`.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::block_palette::PaletteHost;
use display::block_texture::LoadedMaterial;
use work::workers::scan::{
    spawn_auto_scan, spawn_custom_folder_scan, FaceResolver, ScanHandle, ScanMessage,
};
use crate::{
    chrome_zone_left_click_action, classify_cube_point, create_depth_view, create_msaa_color_view,
    procedural_material_average_color, render_frame,
    run_egui_frame, AppConfig, AppCore, ChromeClickAction, CubeChromeZone, CubeFace, CubeRect,
    RebuildOutcome, RebuildOutput, RecentreVoxels,
    EguiPaintBridge,
    FrameOverlays,
    TransformGizmoRenderer,
    GpuContext, InfiniteGridRenderer, LayerBand, MaterialSource, PointsRenderer,
    SceneGridRenderer,
    HomeView, NodeSpec, OrbitCamera, PanelState, SdfShape, SnapTween, ViewCubeElement,
    ViewCubeMenuRequest,
    ViewCubeRenderer, COLOR_TARGET_FORMAT,
    view_cube_corner, VIEW_CUBE_VIEWPORT_PIXELS,
};
// The display-state machine (both renderers + both async workers + the install seams) now
// lives in the `DisplayOrchestrator`; the shell holds one and calls it at its integration
// points. See `docs/architecture/03-display.md`.
use crate::{
    spawn_diameter_worker, spawn_vox_export_worker, DiameterRequest, DiameterWorker,
    DisplayOrchestrator, DisplayRefreshContext, GenerationTracker, VoxExportRequest,
    VoxExportWorker,
};

mod events;
mod export;
mod geometry;
mod palette;
mod render;
mod view_cube;
mod workers;

/// Drag threshold (pixels) distinguishing a click (snap) from a drag (orbit) on
/// the view cube, and the general orbit-start threshold.
const VIEW_CUBE_DRAG_THRESHOLD_PIXELS: f64 = 5.0;

/// State that exists only once the window and GPU have been created (on first
/// `resumed`). Kept in its own struct so `App` can start as `None` before then.
struct WindowedState {
    /// Stored as Arc so the surface can be `Surface<'static>` (DEV_NOTES /
    /// Hard requirement #6): the surface is created from `window.clone()`.
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    egui_bridge: EguiPaintBridge,
    egui_winit_state: egui_winit::State,
    panel_state: PanelState,
    /// The display-state machine (map item 2): both display renderers (the cuboid fallback
    /// mesh and the brick raymarch), both async rebuild workers with their generation trackers
    /// and outstanding flags, the `mesh_stale` / brick-handover bookkeeping, and the install
    /// seams that keep them in lock-step. The shell delegates every display decision to it and
    /// keeps only input, surface, egui, and camera. Constructed window-free from cloned wgpu
    /// handles in [`DisplayOrchestrator::first_build`].
    display: DisplayOrchestrator,
    transform_gizmo_renderer: TransformGizmoRenderer,
    /// The boolean-operand ghost (ADR 0018 Decision 6, "Show booleans" mode): every
    /// Subtract/Intersect operand body in the selected subtree, as an operation-coded
    /// x-ray over the composed scene. One renderer instance; its meshes are re-derived
    /// ONLY on selection / geometry / MODE change (see the dirty flag below), never per
    /// frame. Empty in Normal / Onion-fog mode.
    selected_operand_ghost_renderer: crate::SelectedOperandGhostRenderer,
    /// Forces a boolean-operand ghost re-derivation on the next frame. Set at startup and
    /// whenever an applied Intent reports `selection_changed` / `scene_changed`; the
    /// render seam also re-derives when `scene.active` or the view mode differs from what
    /// the ghost was last derived for.
    selected_ghost_dirty: bool,
    /// The selection the ghost meshes were last derived for.
    selected_ghost_selection: Option<crate::NodeId>,
    /// The view mode the ghost meshes were last derived for (re-derive on a mode change).
    selected_ghost_view_mode: crate::ViewMode,
    /// Per-object block lattice + floor grid (issue #29 S3). Its line batch is
    /// rebuilt each frame from the visible nodes' enabled grids.
    scene_grid_renderer: SceneGridRenderer,
    /// The world reference AXES (issue #29 S5): every visible Point's axis lines. Its
    /// line batch is rebuilt each frame from `scene.points`.
    points_renderer: PointsRenderer,
    /// The analytic infinite reference grid (issue #29 Points fast-follow): every
    /// visible Point's enabled PLANES, drawn as fullscreen ray-plane passes. Replaces
    /// the old finite tiled-line ground plane.
    infinite_grid_renderer: InfiniteGridRenderer,
    /// ADR 0022: the armed-tool placement ghost — a translucent analytic SDF drawn where
    /// the armed primitive's voxels would land. Held permanently and armed per-frame from
    /// `PanelState::placement_ghost`; disarmed (no draw) when nothing is armed. The live
    /// cursor/click arming is a later slice — for now it renders whatever a loaded config
    /// (F9 repro) armed.
    placement_ghost_renderer: crate::PlacementGhostRenderer,
    view_cube_renderer: ViewCubeRenderer,
    /// The Signal viewport background gradient (issue #91): a fullscreen radial field
    /// painted first in the 3D pass so the scene composites over it.
    background_gradient_renderer: display::renderer::BackgroundGradientRenderer,
    /// The palette of scanned VS blocks: the UI-facing tiles/status/click counter plus
    /// the shell-side GPU host (thumbnail renderer + texture keep-alives + block groups),
    /// kept index-aligned (M6).
    palette: PaletteHost,
    /// The in-flight background scan (auto-detect on startup, or a custom folder
    /// scan triggered by "Connect folder…"). `None` once finished/idle.
    scan_handle: Option<ScanHandle>,
    /// Groups received from the scan worker but not yet turned into tiles; drained
    /// a few per frame so a few-hundred-block scan doesn't hitch a single frame.
    pending_groups: std::collections::VecDeque<(assets::BlockGroup, assets::DecodedRgba)>,
    /// Final group count from the worker's `Done`, applied to the status line once
    /// the pending queue is fully drained.
    scan_total: Option<usize>,
    /// Source name from the worker's `Done` (for the settled status line).
    scan_source_name: Option<String>,
    /// The active applied VS block, if any (M6/M7). When `Some`, the voxel pass
    /// binds this loaded 6-layer face material instead of the procedural one.
    loaded_material: Option<LoadedMaterial>,
    /// Per-face texture resolver (M7): kept alive beside the palette so a clicked
    /// block resolves its blocktype JSON → per-face PNGs on the main thread.
    /// Rebuilt when "Connect folder…" switches the source.
    face_resolver: FaceResolver,
    /// ADR 0011 G5: the last rebuild's region dimensions (voxels) + composite recentre
    /// (floating origin, ADR 0008). The dense `VoxelGrid` husk is GONE — the camera
    /// auto-frame and layer scrubber read these scalars directly.
    region_dimensions: [u32; 3],
    recentre_voxels: RecentreVoxels,
    /// The headless orchestrator (ADR 0003 keystone): owns the per-chunk resolve
    /// store (issue #27 S2 — the resolve mechanism behind `rebuild_geometry`, with
    /// issue #27 S3's TARGETED invalidation that diffs the scene's leaf spatial
    /// index against the previous one and evicts only the chunks the edit's
    /// world-AABB touched) and the orbit camera. The shell delegates headless work
    /// to it (`self.app_core.store` / `self.app_core.camera`) and keeps the GPU
    /// renderers + winit/egui plumbing.
    app_core: AppCore,
    /// Cached widest-run measurement (the "Ø N vx" readout) shown in the panel. Updated
    /// asynchronously: it holds the PREVIOUS (stale) value until a fresh measurement from the
    /// [`DiameterWorker`] lands, so the UI never blocks on the O(total blocks) query.
    measured_diameter: u32,
    /// The band the most recent measurement was DISPATCHED for (not necessarily landed yet).
    /// A change (band scrub, or the grid-edit reset to `(u32::MAX, u32::MAX)`) re-dispatches.
    measured_band: (u32, u32),
    /// ADR 0010 E5 follow-up: the background diameter / widest-run measurement worker — the
    /// layer-band readout is streamed off the event-loop thread so a huge scene never freezes
    /// the UI on a scrub. The shell shows the stale `measured_diameter` until a fresh result
    /// arrives (`poll_diameter_worker`).
    diameter_worker: DiameterWorker,
    /// Supersede bookkeeping for `diameter_worker`: each dispatch stamps a fresh generation; a
    /// received result is accepted only when its generation is still the newest dispatched (a
    /// mid-measure scrub/edit supersedes the older in-flight measurement — its result is
    /// discarded, exactly as the geometry worker's [`GenerationTracker`]).
    diameter_generation: GenerationTracker,
    /// Slow-paths backlog item 2: the background `.vox` export worker. A `.vox` write
    /// re-streams the whole scene occupancy + serialises it — multi-second on a huge scene
    /// — so it runs off the event-loop thread. Unlike the display workers it carries NO
    /// supersede generation (an export is a user-chosen file, never superseded); the shell
    /// serialises via `export_outstanding` below (see `workers::export`).
    vox_export_worker: VoxExportWorker,
    /// True while an export request is in flight. Disables the export button (so a second
    /// export can never be queued — the worker's drain-to-latest would otherwise silently
    /// drop it) and gates the progress readout. Cleared when the result lands.
    export_outstanding: bool,
    /// While an export is in flight: `(per-chunk counter the worker bumps, total covering
    /// chunks)`. The panel reads it for the "Exporting… done/total chunks" line. A `0`
    /// denominator (empty / VoxelBody-only scene) shows just the count.
    export_progress: Option<(Arc<std::sync::atomic::AtomicU64>, u64)>,
    /// The last export completion or failure message (replaces the old `println!`/
    /// `eprintln!`), plus the large-export warning. Shown as small weak text under the
    /// export button once no export is in flight.
    export_status: Option<String>,
    /// Data-loss guard: set when the user requested a window close WHILE an export was in
    /// flight. The background export worker is detached, so exiting immediately would kill
    /// it mid-build/mid-write; instead we DEFER the close and exit once the result lands
    /// (see `poll_vox_export_worker` / the `RedrawRequested` seam). Escape hatch: a SECOND
    /// close request while already deferring means the user is insisting — the shell exits
    /// immediately, and the atomic `.vox` write bounds the damage to "no file", never a
    /// truncated one.
    close_requested_while_exporting: bool,
    depth_view: wgpu::TextureView,
    /// 4× MSAA colour target for the 3D pass; resolved into the surface texture.
    msaa_color_view: wgpu::TextureView,
    /// The saved Home view (#13): the orbit angles + distance the Home button
    /// returns to. Restored from the persisted config; updated by
    /// `set_home_to_current`. Step 1 only stores it (no input wiring yet).
    home_view: HomeView,
    /// In-progress eased view-cube snap, if any.
    snap_tween: Option<SnapTween>,
    /// Timestamp of the previous frame, for advancing the snap tween.
    last_frame_time: std::time::Instant,
    /// Whether the left mouse button is held (orbit drag in progress).
    left_button_held: bool,
    /// Whether the middle mouse button is held (pan drag in progress).
    middle_button_held: bool,
    /// Last cursor position, for computing drag deltas.
    last_cursor_position: Option<(f64, f64)>,
    /// Where the most recent left-press landed (for view-cube click detection).
    press_position: Option<(f64, f64)>,
    /// Whether the most recent left-press started inside the view-cube viewport.
    press_in_view_cube: bool,
    /// Whether a press that started on the view cube has moved past the drag
    /// threshold and is now orbiting the main camera (so the release snaps nothing).
    view_cube_drag_active: bool,
    /// Issue #25: the central 3D viewport rect ([x, y, w, h], physical pixels) from
    /// the most recent rendered frame. The view-cube hit-testing (run in mouse
    /// events, outside `render`) needs the cube's top-left corner, which is offset
    /// into this central rect — so we cache the rect each frame.
    last_viewport_px: [u32; 4],
    /// Signal (issue #88): the view cube's right inset (physical px) = the floating display
    /// stack's current width, cached from the most recent rendered frame so the cube
    /// hit-testing (run in mouse events, outside `render`) offsets the cube corner by the
    /// SAME amount `run_egui_frame` drew it with. Kept beside `last_viewport_px`.
    last_cube_right_inset: u32,
    /// The Signal chrome hit-rects (`[x, y, w, h]`, physical px) from the most recent
    /// rendered frame: the floating display stack + the icon rail. The camera gate
    /// (orbit / pan / wheel-zoom, run in mouse events) treats pointer input inside them
    /// as chrome, mirroring the cube's reserved region — the stack no longer allocates
    /// in egui's root ui (the #88 full-width dead-band regression), so egui's own
    /// pointer-consumption heuristic no longer covers this chrome and the shell must.
    last_chrome_rects_px: Vec<[f32; 4]>,
    /// #13 Step 3: the screen position (window pixels) of an open ViewCube
    /// right-click context menu, or `None` when no menu is open. Set on a
    /// right-press inside the cube rect; the egui pass draws a small menu there and
    /// clears it on selection or click-away.
    context_menu_open_at: Option<egui::Pos2>,
    /// #13 Step 4: the ViewCube chrome zone currently under the cursor (a rotate
    /// or roll arrow / Home / Fit), driving the live hover highlight in
    /// [`ViewCubeRenderer::draw`]. Recomputed cheaply on every `CursorMoved`; held
    /// at `None` while orbiting/dragging, when the cursor leaves the cube rect, or
    /// when egui consumed the move. The cube body never highlights (we skip its
    /// raycast for hover), so a body hover is treated as `None`.
    hovered_cube_zone: Option<CubeChromeZone>,
    /// ADR 0022 live placement: the tool the user armed from "+ Add", or `None`. While
    /// `Some`, each frame resolves the placement ghost under the cursor and a stationary
    /// left click drops the node — it STAYS armed so several can be placed; Escape or a
    /// right-click disarms. A VIEW/session concern (mirroring the panel's `armed_tool`),
    /// never a document Intent.
    armed_tool: Option<NodeSpec>,
    /// The last rebuild's resident two-layer chunks, kept so the per-frame placement
    /// resolve has a `PickFrame` to march against (Arc refcount bumps — cheap).
    /// Refreshed in `rebuild_geometry` before the chunks move into the display.
    resident_chunks: Vec<([i32; 3], Arc<crate::TwoLayerChunk>)>,
    /// The band the last rebuild drew with, carried into the placement `PickFrame` so a
    /// pick cannot select geometry the viewer mode is not drawing.
    last_pick_band: LayerBand,
    /// The placement intent the armed tool would drop at the current cursor, refreshed
    /// each frame alongside the ghost. `Some` only over a valid drop; a stationary left
    /// click moves it onto `viewport_intents`.
    pending_placement: Option<crate::Intent>,
    /// Whether the current left-press began a placement (armed, on the viewport, not on
    /// egui/cube/chrome). A stationary release with this set drops the pending node; a
    /// drag leaves it and orbits instead (only a stationary click places).
    armed_press: bool,
    /// Placement intents produced by a viewport click this frame, drained into the SAME
    /// apply loop as the panel intents so a drop goes through `apply_intent` + rebuild.
    viewport_intents: Vec<crate::Intent>,
    /// ADR 0028 (#94): the sketch profile's vertex handles, projected to egui points with
    /// their interaction state, refreshed at the END of each frame and drawn on the NEXT
    /// (a one-frame lag, imperceptible for handle chrome). Empty outside sketch mode.
    sketch_overlay_points: Vec<(egui::Pos2, ui::gizmos::HandleState)>,
    /// Every profile vertex's centre in PHYSICAL pixels, **in profile order** — `None` where
    /// projection culled a behind-camera vertex. The press hit-tests (in `events`, outside
    /// `render`) read these: a vertex grab / delete finds the nearest `Some`, and add-point
    /// finds the nearest projected SEGMENT (consecutive `Some` pairs, closing the loop). Kept
    /// in profile order — rather than the old index-paired flat list — precisely because
    /// segments need adjacency, which a culled-and-compacted list loses.
    sketch_vertex_px: Vec<Option<egui::Pos2>>,
    /// The stable point id for each entry in [`sketch_vertex_px`](Self::sketch_vertex_px), in
    /// the SAME order — maps an overlay hit index to the entity to drag or delete (the store has
    /// no positional index, ADR 0030).
    sketch_point_ids: Vec<document::sketch::EntityId>,
    /// Each segment as `(segment id, from index, to index)` into
    /// [`sketch_vertex_px`](Self::sketch_vertex_px) — the add-point hit-test splits the named
    /// segment by id, and the overlay draws a line per entry (ADR 0030, not consecutive pairs).
    sketch_segments: Vec<(document::sketch::EntityId, usize, usize)>,
    /// Each committed segment's two endpoints in egui POINTS for THIS frame plus its interaction
    /// [`HandleState`](ui::gizmos::HandleState), drawn as a line on the NEXT (ADR 0030 — a sketch's
    /// edges, so an open profile reads as connected geometry, not loose dots). The one segment
    /// under the cursor (when no vertex is closer — vertices take priority) carries `Hover` under
    /// the Select tool (brighter line) or `Marked` under Delete (warn-red line + `✕`); every other
    /// segment is `Idle`. One vocabulary with the vertex handles. Only segments whose BOTH
    /// endpoints projected in front of the camera appear; a behind-camera endpoint
    /// (`sketch_vertex_px` `None`) culls its line. Built in
    /// [`refresh_sketch_overlay`](Self::refresh_sketch_overlay) alongside the handles.
    sketch_segment_lines: Vec<(egui::Pos2, egui::Pos2, ui::gizmos::HandleState)>,
    /// The add-point tool's insert-preview marker for THIS frame (egui points): where a click
    /// would drop a vertex on the hovered segment (the foot of the perpendicular from the
    /// cursor), or `None` when the add-point tool is idle / no segment is under the cursor.
    /// Refreshed alongside the handles; drawn as a diamond on the next frame.
    sketch_insert_preview: Option<egui::Pos2>,
    /// The last frame's world→clip matrix, cached so the release handler (in `events`) can
    /// invert a cursor into a profile coordinate for an add-point insert — the same projection
    /// `render` fed the overlay refresh. `None` before the first frame.
    last_view_projection: Option<glam::Mat4>,
    /// Whether the most recent left-press armed a sketch add-point / delete edit (sketch mode,
    /// an edit tool, on the live viewport). A STATIONARY release with this set performs the
    /// edit; a drag leaves it and orbits instead — the placement `armed_press` pattern, so a
    /// click edits and a drag still rotates the view.
    sketch_edit_press: bool,
    /// The in-progress vertex drag (a press landed on a handle), or `None`. While `Some`, each
    /// frame re-projects the cursor onto the sketch plane, grid-snaps, and DIRECTLY updates the
    /// scene node for a live re-resolve preview (no command recorded). On release the `events`
    /// handler commits it synchronously as one edit in the open undo group
    /// (`commit_sketch_vertex_drag`), which clears this back to `None`.
    sketch_drag: Option<SketchVertexDrag>,
}

/// An in-progress sketch point-vertex drag (ADR 0028 #94, id-based per ADR 0030).
#[derive(Debug, Clone)]
struct SketchVertexDrag {
    /// The stable id of the point entity being dragged — NOT a loop index, which is invalid
    /// once the graph opens (ADR 0030).
    point_id: document::sketch::EntityId,
    /// The sketch producer as it stood when the vertex was grabbed — the base every preview
    /// moves the dragged vertex on (a fresh clone), so successive frames never compound, and
    /// the RESTORE-before-commit reverts to exactly this.
    original: document::sketch::SketchSolid,
    /// The node's world voxel offset at grab time. The preview compensates this by the shift in
    /// the profile's bbox-minimum so the NON-dragged vertices stay put in world while the
    /// grabbed one tracks the cursor (the sketch producer re-anchors its bbox-min to the node
    /// origin, so without this the grabbed min-vertex would pin and the rest would lurch).
    original_offset: [i64; 3],
    /// The profile's in-plane bbox-minimum at grab time — the fixed reference the per-frame
    /// compensation measures the bbox-min shift against.
    original_min: [i64; 2],
}

#[derive(Default)]
struct App {
    state: Option<WindowedState>,
}

impl WindowedState {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        // M8: load persisted config (geometry, display, material, camera, window
        // size). Missing/invalid config falls back to defaults (never panics).
        let config = AppConfig::load();

        let mut window_attributes = Window::default_attributes()
            .with_title("VoxelWorker")
            // Open maximized so the 3D view + panels get the full screen.
            .with_maximized(true);
        if let Some(config) = &config {
            window_attributes = window_attributes.with_inner_size(winit::dpi::LogicalSize::new(
                config.window_size[0],
                config.window_size[1],
            ));
        }
        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("failed to create window"),
        );

        // Headless GpuContext also creates the instance, but the windowed path
        // needs the surface to exist before requesting the adapter so the
        // adapter is guaranteed presentable. So we build the instance + surface
        // here, then hand the surface to GpuContext::new as compatible_surface.
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        let gpu = pollster::block_on(async {
            // Reuse the surface-aware adapter selection but keep the same device
            // setup as GpuContext for parity. We construct GpuContext with this
            // surface as the compatibility hint.
            GpuContext::new_with_instance(instance, Some(&surface)).await
        });

        let physical_size = window.inner_size();
        let width = physical_size.width.max(1);
        let height = physical_size.height.max(1);

        let mut surface_config = surface
            .get_default_config(&gpu.adapter, width, height)
            .expect("surface is not supported by the adapter");
        // Force the same sRGB format the headless capture uses so the window and
        // the screenshot are pixel-identical (Hard requirement #9).
        surface_config.format = COLOR_TARGET_FORMAT;
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&gpu.device, &surface_config);

        let egui_bridge = EguiPaintBridge::new(&gpu.device, COLOR_TARGET_FORMAT);

        let egui_winit_state = egui_winit::State::new(
            egui_bridge.context.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        // Resolve the panel geometry into the grid, then build the renderer's
        // instance buffer FROM the grid (the resolved-grid seam, `docs/adr/0006`). The view cube +
        // block lattice are ON by default; the persisted config overrides them.
        let mut panel_state = match &config {
            Some(config) => config.to_panel_state(),
            None => PanelState::with_view_cube_default(),
        };
        let shape = SdfShape::from_geometry(panel_state.geometry.clone());
        // ADR 0011 G5: the startup DOOR constructs NO `VoxelGrid` — it returns only the region
        // dimensions + resolve recentre (the camera auto-frame and layer scrubber
        // consume these scalars), exactly what the per-edit `AppCore::rebuild` yields. This
        // closes the startup OOM on both binaries: the persisted 8000×800×800 scene once
        // resolved a dense ~5.1-billion-cell grid (~28.5 GB → OOM hang before the first print),
        // and the non-gpu binary streamed the same region — now neither materialises occupancy.
        // (The recentre half is recomputed below as `startup_recentre`, reused by the brick
        // install + mesh; discard the tuple's copy here.)
        let (region_dimensions, _) = AppCore::startup_region(
            &panel_state.scene,
            panel_state.geometry.voxels_per_block,
        );
        // Initialise the layer-range band to the full grid height (issue #12). Z-up:
        // layers are Z-slices, so the track spans the Z dimension (index 2).
        let grid_z = region_dimensions[2];
        panel_state
            .layer_range
            .rescale_to_grid_z(0, grid_z, panel_state.geometry.voxels_per_block);
        // ADR 0010 E5 follow-up: the diameter / scrubber readout is measured ASYNCHRONOUSLY
        // (the streamed cacheless query is O(total blocks) — sub-second on a huge solid but not
        // free, and a persisted config could load a large scene at startup). Seed a stale `0`
        // and an impossible band so the first render frame's `current_band != measured_band`
        // guard dispatches the first measurement to the `DiameterWorker`; the readout fills in
        // when it lands. No occupancy is ever resolved synchronously on the main thread here.
        let measured_band = (u32::MAX, u32::MAX);
        let measured_diameter = 0u32;
        let diameter_worker = spawn_diameter_worker();
        let diameter_generation = GenerationTracker::new();
        let vox_export_worker = spawn_vox_export_worker();
        // ADR 0011 G5: no occupancy is ever resolved at startup (dims-only door).
        println!(
            "resolved region {:?} for {:?} {:?}@{} (no dense occupancy)",
            region_dimensions,
            shape.kind,
            shape.size_voxels,
            panel_state.geometry.voxels_per_block,
        );
        // ADR 0010 E5: the cuboid mesh renderer is the sole voxel render path AND it
        // meshes THROUGH the two-layer store (coarse one-box + microblock cuboids +
        // seam-flag culling) — the SAME path `rebuild_geometry` takes on every later
        // edit, so the startup frame (which renders until the first edit re-meshes) is
        // pixel-identical to the two-layer runtime path. `build_covering_chunks` returns
        // empty for a VoxelBody-only scene (the windowed startup default is always chunkable).
        let startup_density = panel_state.geometry.voxels_per_block;
        // Build the startup covering set THROUGH the resident cache that becomes
        // `app_core.two_layer_cache` (async-brick startup follow-up): byte-identical
        // chunks (same `build_two_layer_chunk_from_leaves` over the same coords as the
        // stateless `build_covering_chunks`), but the cache is WARM from frame one — a
        // pre-first-edit display seam (`rebuild_stale_display_mesh` after an async brick
        // build lands `Empty`) hands out these residents as O(chunks) `Arc`
        // bumps instead of synchronously re-resolving the whole set on the main thread.
        let mut startup_two_layer_cache = crate::TwoLayerResidentCache::enabled();
        let startup_two_layer_chunks = startup_two_layer_cache.resident_two_layer_chunks(
            &panel_state.scene,
            startup_density,
            0,
        );
        let startup_recentre = panel_state.scene.recentre_voxels_for_resolve(startup_density);
        // Map item 2: the display-state machine builds itself from the startup covering set —
        // the brick engagement decision, both worker spawns, the (possibly skipped-empty)
        // cuboid mesh, and all display bookkeeping. Cloned wgpu handles keep the shell's
        // `GpuContext` free for its own (non-voxel) renderers below.
        let display = DisplayOrchestrator::first_build(
            gpu.device.clone(),
            gpu.queue.clone(),
            COLOR_TARGET_FORMAT,
            &startup_two_layer_chunks,
            region_dimensions,
            startup_recentre,
            startup_density,
            panel_state.debug_face_orientation,
        );
        // The transform gizmo (issue #29 S2) is rebuilt/positioned to the SELECTED
        // node each frame; seed it at the region size (overwritten on first frame).
        let transform_gizmo_renderer =
            TransformGizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, region_dimensions);
        // The boolean-operand ghost (ADR 0018 Decision 6): built empty; the first render
        // frame derives it for the loaded scene's selection when the view mode is
        // Show-booleans (`selected_ghost_dirty` below).
        let selected_operand_ghost_renderer =
            crate::SelectedOperandGhostRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        // Per-object block lattice + floor grid (issue #29 S3): its line batch is
        // (re)built per frame from the grid-enabled nodes, so it starts empty.
        let scene_grid_renderer = SceneGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        // The world reference grid (issue #29 S5): the visible Points' tiled planes +
        // axes. Its batch is rebuilt per frame from the scene + camera, so empty here.
        let points_renderer = PointsRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        let infinite_grid_renderer = InfiniteGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        // ADR 0022: the armed-tool placement ghost, held permanently (disarmed until a
        // frame arms it from `PanelState::placement_ghost`).
        let placement_ghost_renderer =
            crate::PlacementGhostRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        let view_cube_renderer =
            ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let background_gradient_renderer =
            display::renderer::BackgroundGradientRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);

        // Kick off the VS auto-detect + scan on a background thread immediately;
        // results stream in over the next frames (no startup block).
        let palette = PaletteHost::new(&gpu.device, &gpu.queue, "Scanning…".to_string());
        let scan_handle = Some(spawn_auto_scan());

        let mut camera = OrbitCamera {
            orbit_distance: OrbitCamera::auto_framed_distance(region_dimensions),
            projection_mode: panel_state.projection_mode,
            ..OrbitCamera::default()
        };
        // Restore the saved Home view (#13), or default to the camera defaults.
        let home_view = config
            .as_ref()
            .map(AppConfig::home_view)
            .unwrap_or_default();
        // Startup camera selection: a loaded config (`Some`) means a prior session
        // persisted its live camera — resume it exactly (the existing behavior). A
        // GENUINE FIRST RUN (`AppConfig::load()` returned `None`: no config file, or
        // an unreadable/invalid one) has no live camera to resume, so open at the
        // Home view corner instead — the same `home_view` the Home button snaps to.
        // The signal is `config.is_none()`; a config that merely lacks some keys is
        // still `Some` (serde fills the gaps), so partial configs still resume.
        match &config {
            Some(config) => config.apply_camera(&mut camera),
            None => {
                camera.orbit_theta = home_view.theta;
                camera.orbit_phi = home_view.phi;
            }
        }

        let depth_view = create_depth_view(&gpu.device, width, height);
        let msaa_color_view =
            create_msaa_color_view(&gpu.device, width, height, COLOR_TARGET_FORMAT);

        Self {
            window,
            surface,
            surface_config,
            gpu,
            egui_bridge,
            egui_winit_state,
            panel_state,
            display,
            transform_gizmo_renderer,
            selected_operand_ghost_renderer,
            selected_ghost_dirty: true,
            selected_ghost_selection: None,
            selected_ghost_view_mode: crate::ViewMode::Normal,
            scene_grid_renderer,
            points_renderer,
            infinite_grid_renderer,
            placement_ghost_renderer,
            view_cube_renderer,
            background_gradient_renderer,
            palette,
            scan_handle,
            pending_groups: std::collections::VecDeque::new(),
            scan_total: None,
            scan_source_name: None,
            loaded_material: None,
            face_resolver: FaceResolver::auto(),
            region_dimensions,
            recentre_voxels: startup_recentre,
            // The startup covering set was built through this cache, so it is WARM from
            // frame one (see the `startup_two_layer_cache` comment above).
            app_core: AppCore::with_warm_two_layer_cache(camera, startup_two_layer_cache),
            measured_diameter,
            measured_band,
            diameter_worker,
            diameter_generation,
            vox_export_worker,
            export_outstanding: false,
            export_progress: None,
            export_status: None,
            close_requested_while_exporting: false,
            depth_view,
            msaa_color_view,
            home_view,
            snap_tween: None,
            last_frame_time: std::time::Instant::now(),
            left_button_held: false,
            middle_button_held: false,
            last_cursor_position: None,
            press_position: None,
            press_in_view_cube: false,
            view_cube_drag_active: false,
            // Default to the full target until the first frame fills it in.
            last_viewport_px: [0, 0, width, height],
            // Issue #88: the expanded stack's inset until the first frame refreshes it.
            last_cube_right_inset: crate::cube_right_inset_points(false).round() as u32,
            // Empty until the first frame fills it in (no chrome to reserve yet).
            last_chrome_rects_px: Vec::new(),
            context_menu_open_at: None,
            hovered_cube_zone: None,
            // ADR 0022 live placement: nothing armed until the user picks a "+ Add" chip.
            armed_tool: None,
            // Seed the placement pick-set from the STARTUP covering set — the same chunks the
            // display's `first_build` drew (below). Without this, `resident_chunks` stayed empty
            // until the first edit ran `rebuild_geometry`, so on a fresh launch a pick found no
            // geometry to march and an armed tool could not drop onto the already-loaded scene
            // (adding a node rebuilt and "fixed" it). FULL band = mask nothing, matching a
            // fresh launch's un-clipped view.
            resident_chunks: startup_two_layer_chunks.clone(),
            last_pick_band: LayerBand::FULL,
            pending_placement: None,
            armed_press: false,
            viewport_intents: Vec::new(),
            sketch_overlay_points: Vec::new(),
            sketch_vertex_px: Vec::new(),
            sketch_point_ids: Vec::new(),
            sketch_segments: Vec::new(),
            sketch_segment_lines: Vec::new(),
            sketch_insert_preview: None,
            last_view_projection: None,
            sketch_edit_press: false,
            sketch_drag: None,
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
        // Recreate the depth + MSAA colour textures to match the new target size.
        self.depth_view = create_depth_view(&self.gpu.device, width, height);
        self.msaa_color_view =
            create_msaa_color_view(&self.gpu.device, width, height, COLOR_TARGET_FORMAT);
    }

    /// Bundle the borrows the [`DisplayOrchestrator`] needs to re-mesh the stale fallback
    /// from the resident cache off the main edit path (the per-frame brick poll + the
    /// `ensure_display_mesh_current` seam). An associated function (not `&self`) so the caller
    /// borrows the shell's `panel_state` / `app_core` fields DISJOINTLY from `self.display`,
    /// which the orchestrator call then borrows mutably.
    fn make_refresh_context<'a>(
        panel_state: &'a PanelState,
        two_layer_cache: &'a mut crate::TwoLayerResidentCache,
        region_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        band: LayerBand,
        region: Option<crate::RegionClip>,
    ) -> DisplayRefreshContext<'a> {
        DisplayRefreshContext {
            scene: &panel_state.scene,
            two_layer_cache,
            density: panel_state.geometry.voxels_per_block,
            region_dimensions,
            recentre_voxels,
            band,
            region,
            debug_face_orientation: panel_state.debug_face_orientation,
        }
    }

    /// The EFFECTIVE layer clip (band + onion-fog region) the render path will apply this
    /// frame for a whole-scene grid of `scene_grid_z` layers (issue #12 / #60 M2 / ADR 0018
    /// Decisions 4–5). Delegates to the shared [`AppCore::mesh_clip`] so the async worker,
    /// the fallback rebuild, and the render uniforms all clip identically (the swap frame's
    /// `rebuild_for_band` then no-ops). The band bites only in Onion-fog mode with a
    /// selection; debug-faces / Normal / Show-booleans force FULL + no region.
    fn current_mesh_clip(&self, scene_grid_z: u32) -> crate::MeshClip {
        AppCore::mesh_clip(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            self.panel_state.view_mode,
            self.panel_state.layer_range,
            scene_grid_z,
            self.panel_state.debug_face_orientation,
        )
    }

    /// Persist the current UI + camera + window state to the platform config
    /// (M8). Called on window close / loop exit. Never panics on failure.
    fn save_config(&self) {
        let window_size = [self.surface_config.width, self.surface_config.height];
        let config =
            AppConfig::capture(&self.panel_state, &self.app_core.camera, self.home_view, window_size);
        config.save();
    }

    /// Dump the CURRENT scene + LIVE camera (theta/phi/distance/roll/projection) to a
    /// repro file the `shot` harness loads with `--from-config`, so a bug seen at an exact
    /// live view reproduces headlessly byte-for-byte. Bound to F9. Writes to the system temp
    /// dir (`voxelworker-repro.json`) and prints the absolute path. Unlike `save_config`, this
    /// captures the camera AS IT IS THIS FRAME (config.json only persists on exit), which is the
    /// whole point — an artifact pose is never the last-saved pose.
    fn export_repro(&self) {
        let window_size = [self.surface_config.width, self.surface_config.height];
        let config =
            AppConfig::capture(&self.panel_state, &self.app_core.camera, self.home_view, window_size);
        let path = std::env::temp_dir().join("voxelworker-repro.json");
        // The DUMP, explicitly (ADR 0022): the superset artifact, from which a scene must be
        // completely reproducible. The document projection would be the wrong choice here by
        // construction — it deliberately drops the camera, which is the one thing a repro of a
        // visual bug cannot do without.
        match config
            .to_dump_json()
            .and_then(|json| std::fs::write(&path, json).map_err(|e| e.to_string()))
        {
            Ok(()) => eprintln!("repro: wrote current scene + camera to {}", path.display()),
            Err(error) => eprintln!("repro: failed to write {}: {error}", path.display()),
        }
    }

    /// ADR 0022 live placement: cancel any armed tool — clear the arm, the pending drop,
    /// the ghost preview, and the press latch. Escape and a viewport right-click both call
    /// this; the ghost vanishes on the next frame (nothing armed ⇒ the pass is a no-op).
    fn disarm_placement(&mut self) {
        self.armed_tool = None;
        self.pending_placement = None;
        self.panel_state.placement_ghost = None;
        self.armed_press = false;
    }

    /// The shared shutdown sequence: persist config, then exit the loop. Called from both
    /// the immediate `CloseRequested` path and the deferred-close honour seam so the two
    /// never drift (finding #9).
    fn shutdown(&self, event_loop: &ActiveEventLoop) {
        self.save_config();
        event_loop.exit();
    }
}

/// Run the windowed application: start the (optional) Tracy client, create the winit event
/// loop, and pump the [`App`] handler until exit. The thin `src/main.rs` binary is just a
/// call to this.
pub fn run() {
    // Start the Tracy client and hold the guard alive for the whole program so it
    // stays connectable from the external Tracy profiler app. No-op / absent unless
    // built with `--features tracy` (see docs/profiling.md). CPU zones only for now.
    #[cfg(feature = "tracy")]
    let _tracy_client = tracy_client::Client::start();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
