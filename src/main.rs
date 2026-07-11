//! VoxelWorker — windowed application (default binary).
//!
//! winit 0.30 `ApplicationHandler` + wgpu 29 surface + egui 0.34 panel. Shows
//! the warm-dark workshop clear colour and the shared right-hand egui side
//! panel. It uses the exact same [`render_frame`]/[`build_panel`] code as the
//! headless `shot` binary, so the live window and the captured PNG match.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use voxel_worker::block_palette::{BlockPalette, LoadedMaterial, ThumbnailRenderer};
use voxel_worker::scan_worker::{
    spawn_auto_scan, spawn_custom_folder_scan, FaceResolver, ScanHandle, ScanMessage,
};
use voxel_worker::{
    chrome_zone_left_click_action, classify_cube_point, create_depth_view, create_msaa_color_view,
    procedural_material_average_color, render_frame,
    run_egui_frame, AppConfig, AppCore, ChromeClickAction, CubeChromeZone, CubeFace, CubeRect,
    RebuildOutcome, RebuildOutput,
    EguiPaintBridge, FogMode,
    FrameOverlays,
    TransformGizmoRenderer,
    FogZSlab,
    GpuContext, InfiniteGridRenderer, LayerBand, LayerRange, MaterialSource, PointsRenderer,
    SceneGridRenderer,
    HomeView, OnionFogRenderer, OrbitCamera, PanelState, SdfShape, SnapTween, ViewCubeElement,
    ViewCubeMenuRequest,
    ViewCubeRenderer, VoxelGrid, COLOR_TARGET_FORMAT,
    VIEW_CUBE_VIEWPORT_PIXELS,
};
use voxel_worker::core_geom::CHUNK_BLOCKS;
use voxel_worker::CuboidMeshRenderer;
// ADR 0011 G1: the brick raymarch display sink (engaged for single ported-producer
// scenes under `--features gpu`; the mesh path stays the fallback + A/B reference).
use voxel_worker::BrickRaymarchRenderer;
use voxel_worker::{
    route_geometry_rebuild, route_mesh_build, EditShape, GenerationTracker, GeometryRebuildRequest,
    GeometryWorker, MeshBuildRoute, RebuildRoute, ASYNC_REBUILD_CHUNK_THRESHOLD,
};
use voxel_worker::Scene;
// ADR 0007: the GPU view-resolve pipelines are an opt-in display accelerator (`--features
// gpu`); default builds keep the CPU fog densify so CI / GPU-less runs are unaffected.
#[cfg(feature = "gpu")]
use voxel_worker::{gpu_resolve::GpuResolver, PerChunkAtlasGeometry};

/// Drag threshold (pixels) distinguishing a click (snap) from a drag (orbit) on
/// the view cube, and the general orbit-start threshold.
const VIEW_CUBE_DRAG_THRESHOLD_PIXELS: f64 = 5.0;

/// Margin from the top-left corner to the view-cube viewport (must match the
/// renderer's `VIEW_CUBE_VIEWPORT_MARGIN`).
const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;

/// State that exists only once the window and GPU have been created (on first
/// `resumed`). Kept in its own struct so `App` can start as `None` before then.
/// The onion fog's brick-sourced occupancy inputs, kept together across frames (ADR 0011
/// G5 + interior elision): the brick field build (sculpted records + atlas) AND the
/// two-layer covering chunks it was built from. The record set is SURFACE-ONLY (interiors
/// live in the chunks' coarse layer), so the fog fill box-fills coarse/interior occupancy
/// from `two_layer_chunks` and copies sculpted tiles from `build`'s atlas — see
/// `build_per_chunk_fog_occupancy_from_bricks`. The chunks are `Arc`-shared with the
/// resident cache (an O(chunks) refcount-bump clone per rebuild, never a deep copy).
struct FogBrickSource {
    build: voxel_worker::BrickFieldBuild,
    two_layer_chunks: Vec<([i32; 3], Arc<voxel_worker::TwoLayerChunk>)>,
}

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
    /// The cuboid mesh renderer — the sole voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Rebuilt from the resolve cache's per-chunk
    /// accessor on every geometry change in `rebuild_geometry`.
    cuboid_mesh_renderer: CuboidMeshRenderer,
    /// Brick-display perf follow-up to epic #64: whether `cuboid_mesh_renderer` currently
    /// holds a STALE (skipped / empty) mesh because the ADR 0011 brick raymarch is the live
    /// display and the fallback mesh was not worth the ~333ms serial build. While `true` the
    /// mesh must NOT be drawn (it isn't — the brick pass replaces it) and must NOT be
    /// inline-patched by an incremental edit (its buffers don't reflect the latest resolve);
    /// the next edit that needs the mesh — or [`Self::ensure_display_mesh_current`] on a
    /// debug-face / loaded-material transition — rebuilds it WHOLESALE. Composed into the C1
    /// interlock via [`route_mesh_build`]. Always `false` on non-gpu builds (no brick sink).
    mesh_stale: bool,
    /// ADR 0011 G1: the brick raymarch display sink. Created on first engagement
    /// (a single ported-producer scene under `--features gpu`) and kept — per-edit
    /// work is `install_brick_field` (records + atlas swap, no pipeline rebuild).
    /// When it holds a field and no mesh-only mode is active (debug-faces, a loaded
    /// VS material), the frame's voxel model draws from the brick atlas INSTEAD of
    /// the cuboid mesh; the mesh keeps rebuilding as the fallback + A/B reference
    /// (ADR 0011 Decision 6). `None` on default (no-GPU) builds, always.
    brick_raymarch_renderer: Option<BrickRaymarchRenderer>,
    /// ADR 0011 G3: the PERSISTENT incremental brick field mirroring the boundary set —
    /// the CPU truth an incremental edit patches (dirty chunks re-evaluated, only their
    /// slots written) instead of rebuilding the whole field. `Some` for any chunkable
    /// gpu-gated scene (ADR 0011 G5: it now feeds the FOG too, so it is maintained even
    /// when the DISPLAY falls back to the mesh — a mixed-material scene); reset from a
    /// wholesale `build_brick_field` on a wholesale edit, patched in place on an
    /// incremental edit, and dropped when the scene leaves the gate / empties. `to_build()`
    /// always equals the resident atlas + the fog occupancy source (ADR 0011 G3/G5 gate).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    incremental_brick_field: Option<voxel_worker::IncrementalBrickField>,
    /// ADR 0011 G5: the current brick field + covering chunks the onion FOG sources its
    /// per-chunk occupancy tiles from (`build_per_chunk_fog_occupancy_from_bricks`) — the
    /// boundary set of the last rebuild, kept so the render-path's lazy fog rebuild has a
    /// source across frames without re-streaming a `VoxelGrid`. `Some` iff a chunkable
    /// scene is resident; `None` on loaded-material and Part-only scenes.
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fog_brick_field: Option<FogBrickSource>,
    /// ADR 0011 G2: dedup for the "scene not brick-representable" fallback log — a
    /// chunkable procedural scene whose blocks mix materials / disagree on overlay
    /// keeps the mesh path, reported ONCE per fallback transition (not per drag edit).
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    brick_fallback_reported: bool,
    /// Issue #60 (ADR 0003 §7): the background geometry-rebuild worker. A WHOLESALE
    /// rebuild whose covering-chunk count exceeds [`ASYNC_REBUILD_CHUNK_THRESHOLD`] —
    /// the ~3s large-object build — is dispatched here (cloned `device`/`queue`) instead
    /// of built inline, so the UI never freezes. The main thread keeps rendering the
    /// CURRENT `cuboid_mesh_renderer` (stale-while-rebuilding) until the worker's
    /// freshly-built renderer arrives, then swaps it in. Small / incremental edits stay
    /// synchronous. `None` in the (impossible-in-practice) case the worker failed to spawn.
    geometry_worker: GeometryWorker,
    /// Issue #60: the monotonic generation bookkeeping behind supersede. Each async
    /// dispatch stamps a fresh generation; a received result is swapped in only when its
    /// generation is still the newest dispatched (an edit mid-build supersedes the older
    /// in-flight build, whose result is then discarded — see [`GenerationTracker`]).
    geometry_generation: GenerationTracker,
    /// Issue #60 C1: whether an async WHOLESALE build is OUTSTANDING — dispatched but not
    /// yet accepted/installed. While `true` the currently-installed `cuboid_mesh_renderer`
    /// does NOT reflect the latest resolve (it is still S0 while the worker builds S1), so an
    /// incremental edit must NOT inline-patch it (that strands every chunk that differs
    /// S0→S1 but isn't in the new dirty set — the Frankenstein mesh). The rebuild is routed
    /// to a fresh wholesale-async dispatch instead (see [`route_geometry_rebuild`]). Cleared
    /// when `poll_geometry_worker` accepts + installs a result.
    geometry_async_outstanding: bool,
    transform_gizmo_renderer: TransformGizmoRenderer,
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
    view_cube_renderer: ViewCubeRenderer,
    /// Onion-skin volumetric fog (issue #12).
    onion_fog_renderer: OnionFogRenderer,
    /// ADR 0007 live call-site swap: the GPU view-resolve compute pipelines, built ONCE and
    /// reused every edit (unlike `shot`, which rebuilds them per run). Drives the per-chunk
    /// fog atlas for single-producer scenes on the GPU (no CPU densify); multi-producer
    /// scenes fall back to `upload_fog_occupancy`. Only present under `--features gpu`.
    #[cfg(feature = "gpu")]
    gpu_resolver: GpuResolver,
    /// Onion-fog occupancy mode (issue #28). `PerChunk` is the DEFAULT since S5b
    /// (one apron'd volume per resident chunk packed into a small 3D atlas, so fog
    /// still renders at scale where a single whole-grid 3D texture would exceed
    /// `max_texture_dimension_3d` and disable itself). The legacy `WholeGrid` path is
    /// retained as a fallback. Used by `upload_fog_occupancy` for both the initial
    /// upload and every `rebuild_geometry` re-upload so the two stay in sync.
    fog_mode: FogMode,
    /// Offscreen renderer for the 45° palette cube thumbnails (M6).
    thumbnail_renderer: ThumbnailRenderer,
    /// The palette of scanned VS blocks (tiles + status + click counter, M6).
    palette: BlockPalette,
    /// The in-flight background scan (auto-detect on startup, or a custom folder
    /// scan triggered by "Connect folder…"). `None` once finished/idle.
    scan_handle: Option<ScanHandle>,
    /// Groups received from the scan worker but not yet turned into tiles; drained
    /// a few per frame so a few-hundred-block scan doesn't hitch a single frame.
    pending_groups: std::collections::VecDeque<(voxel_worker::assets::BlockGroup, voxel_worker::scan_worker::DecodedRgba)>,
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
    /// auto-frame, layer scrubber, and fog frame read these scalars directly; fog occupancy
    /// reconstructs from `fog_brick_field` (chunkable) or a transient Part-only resolve.
    region_dimensions: [u32; 3],
    recentre_voxels: [i64; 3],
    /// The headless orchestrator (ADR 0003 keystone): owns the per-chunk resolve
    /// store (issue #27 S2 — the resolve mechanism behind `rebuild_geometry`, with
    /// issue #27 S3's TARGETED invalidation that diffs the scene's leaf spatial
    /// index against the previous one and evicts only the chunks the edit's
    /// world-AABB touched) and the orbit camera. The shell delegates headless work
    /// to it (`self.app_core.store` / `self.app_core.camera`) and keeps the GPU
    /// renderers + winit/egui plumbing.
    app_core: AppCore,
    /// Cached widest-run measurement + the band it was computed for, so we only
    /// re-measure when the band or grid actually changes.
    measured_diameter: u32,
    measured_band: (u32, u32),
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
    /// Issue #58: the onion-fog occupancy is DEMAND-DRIVEN — built/uploaded only when
    /// the fog will actually be drawn (`onion_active`), never on the common edit path
    /// where onion-skin is off (its default). This flag records that the current fog
    /// occupancy is STALE relative to the resolved geometry: set by `rebuild_geometry` (and at
    /// startup) whenever it skips the fog build because onion is inactive, cleared the
    /// moment the render path lazily builds the fog before drawing it. A pure band
    /// scrub does NOT set it (the band clip is applied per-frame; occupancy is reused).
    fog_occupancy_dirty: bool,
    /// Issue #59: the inclusive chunk-Z covering range (Z-up) the fog occupancy was LAST
    /// built for — the identity the band-aware rebuild compares. The fog now covers only
    /// the band-slab `[band ± onion_depth]`, not the whole grid, so a band scrub that
    /// shifts the needed covering chunk-Z rows must rebuild (the old rows aren't in the
    /// atlas), while a scrub that stays within the same rows reuses it. Compared at CHUNK
    /// granularity so a band move within one chunk-Z span does NOT rebuild (no per-voxel-
    /// step atlas rebuilds during a scrub-drag). `None` = fog not built for any slab yet
    /// (or last needed slab was full-range → no fog); the next onion frame builds.
    fog_built_chunk_z_range: Option<[i32; 2]>,
}

#[derive(Default)]
struct App {
    state: Option<WindowedState>,
}

/// The [`CubeFace`] whose outward normal lies along the GEOMETRIC cube `axis`
/// (0=X,1=Y,2=Z) with the given sign. Z-up: +X→Right, −X→Left, +Y→Back, −Y→Front
/// (front = −Y), +Z→Top, −Z→Bottom.
fn face_for_axis_sign(axis: usize, positive: bool) -> CubeFace {
    match (axis, positive) {
        (0, true) => CubeFace::Right,
        (0, false) => CubeFace::Left,
        (1, true) => CubeFace::Back,
        (1, false) => CubeFace::Front,
        (2, true) => CubeFace::Top,
        _ => CubeFace::Bottom,
    }
}

/// The per-`block_id` `.vox` palette over the three procedural materials (ADR 0003
/// §3a): slot `material_id` carries that material's average colour, so a multi-material
/// scene exports each block in its own colour.
fn vox_export_procedural_palette() -> voxel_worker::vox_export::BlockPaletteColors {
    use voxel_worker::core_geom::MaterialChoice;
    let mut palette = [[0u8; 4]; MaterialChoice::MATERIAL_COUNT];
    for (slot, color) in palette.iter_mut().enumerate() {
        *color = procedural_material_average_color(MaterialChoice::from_material_id(slot as u16));
    }
    palette
}

/// Default `.vox` filename from the shape + voxel dims (e.g. `cylinder_80x16x80.vox`).
fn default_vox_filename(shape: &SdfShape, voxels_per_block: u32) -> String {
    let [grid_x, grid_y, grid_z] = shape.grid_dimensions(voxels_per_block);
    let kind = format!("{:?}", shape.kind).to_lowercase();
    format!("{kind}_{grid_x}x{grid_y}x{grid_z}.vox")
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
        // instance buffer FROM the grid (REPRESENTATION.md seam). The view cube +
        // block lattice are ON by default; the persisted config overrides them.
        let mut panel_state = match &config {
            Some(config) => config.to_panel_state(),
            None => PanelState::with_view_cube_default(),
        };
        let shape = SdfShape::from_geometry(panel_state.geometry.clone());
        // ADR 0011 G5: the startup DOOR constructs NO `VoxelGrid` — it returns only the region
        // dimensions + resolve recentre (the camera auto-frame, layer scrubber and fog frame
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
        // ADR 0010 E5: the diameter / scrubber readout STREAMS the cacheless two-layer
        // evaluator (`streamed_widest_run_in_band` — coarse-solid blocks contribute an
        // analytic run, boundary blocks per-voxel), the SAME path the per-frame
        // re-measure takes. The two-layer capability is always ON now, so it never falls
        // back; the retired dense `widest_run_in_band` survives only as the parity
        // oracle. `unwrap_or(0)` covers the Part-only / empty scene (no covering range).
        let measured_band = (panel_state.layer_range.lower, panel_state.layer_range.upper);
        let measured_diameter = voxel_worker::streamed_widest_run_in_band(
            &voxel_worker::TwoLayerStore::enabled(),
            &panel_state.scene,
            panel_state.geometry.voxels_per_block,
            measured_band.0,
            measured_band.1,
        )
        .unwrap_or(0);
        // ADR 0011 G5: no occupancy is ever resolved at startup (dims-only door) — fog sources
        // from the brick sink on the first rebuild.
        println!(
            "resolved region {:?} for {:?} {:?}@{} (no dense occupancy — fog from brick sink)",
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
        // empty for a Part-only scene (the windowed startup default is always chunkable).
        let startup_density = panel_state.geometry.voxels_per_block;
        let startup_two_layer_chunks = voxel_worker::TwoLayerStore::enabled()
            .build_covering_chunks(&panel_state.scene, startup_density, 0);
        let startup_recentre =
            panel_state.scene.recentre_voxels_for_resolve(startup_density);
        // ADR 0011 G2: engage the brick raymarch from the FIRST frame when the startup
        // scene is brick-representable (`--features gpu`, a chunkable procedural scene
        // whose every rendered block is single-material — per-record ids carry per-block
        // materials, so multi-producer distinct-material scenes engage too). Later edits
        // refresh it in `rebuild_geometry`. Perf follow-up to epic #64: this is decided
        // BEFORE the fallback cuboid mesh below so that, when the brick display engages, the
        // ~333ms serial mesh build (and its memory) is SKIPPED at startup — the persisted
        // 8000×800×800 scene installs the brick sink and never meshes.
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut brick_raymarch_renderer: Option<BrickRaymarchRenderer> = None;
        // ADR 0011 G3: the persistent incremental field seeded from the startup wholesale
        // build (kept in lock-step with `brick_raymarch_renderer`).
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut incremental_brick_field: Option<voxel_worker::IncrementalBrickField> = None;
        #[cfg(feature = "gpu")]
        if panel_state.scene.has_chunkable_extent(startup_density) {
            if let Some(overlay_active) =
                voxel_worker::brick_representable_overlay(&startup_two_layer_chunks)
            {
                let build =
                    voxel_worker::build_brick_field(&startup_two_layer_chunks, startup_density);
                if !build.brick_records.is_empty() {
                    let mut renderer =
                        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
                    let pyramid =
                        voxel_worker::ClipmapPyramid::from_chunks(&startup_two_layer_chunks);
                    renderer.install_brick_field(
                        &gpu.device,
                        &gpu.queue,
                        &build,
                        // The record set is surface-only by construction (ADR 0011 interior
                        // elision fused into `build_brick_field`) — a plain 1:1 pack.
                        &voxel_worker::pack_gpu_records(&build, |_| false),
                        &pyramid,
                        startup_recentre,
                        overlay_active,
                    );
                    println!(
                        "brick raymarch: startup field installed ({} records, {} sculpted)",
                        build.brick_records.len(),
                        build.sculpted_brick_count(),
                    );
                    incremental_brick_field =
                        Some(voxel_worker::IncrementalBrickField::from_wholesale(&build));
                    brick_raymarch_renderer = Some(renderer);
                }
            }
        }
        // ADR 0010 E5: the cuboid mesh is the fallback voxel render path AND it meshes THROUGH
        // the two-layer store (coarse one-box + microblock cuboids + seam-flag culling) — the
        // SAME path `rebuild_geometry` takes on every later edit, so the startup frame it draws
        // is pixel-identical to the two-layer runtime path. `build_covering_chunks` returns
        // empty for a Part-only scene (the windowed startup default is always chunkable).
        //
        // Brick-display perf follow-up to epic #64: when the brick raymarch engaged above and no
        // mesh-only mode is active (a config may persist `debug_face_orientation`; a material is
        // never loaded at startup), the mesh is NOT drawn — so SKIP its build entirely and mark
        // it stale. `ensure_display_mesh_current` (or an edit that drops brick engagement) builds
        // the real mesh the moment it is next needed. The empty renderer still carries the
        // pipeline / material bind-group layout / sampler the loaded-material path binds against.
        let brick_engaged_at_startup =
            brick_raymarch_renderer.is_some() && !panel_state.debug_face_orientation;
        let mesh_stale = brick_engaged_at_startup;
        let cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            if brick_engaged_at_startup {
                // Cheap empty renderer: no chunk meshing, just the shared GPU pipeline objects.
                &[]
            } else {
                &startup_two_layer_chunks
            },
            region_dimensions,
            startup_recentre,
            startup_density,
        );
        // The transform gizmo (issue #29 S2) is rebuilt/positioned to the SELECTED
        // node each frame; seed it at the region size (overwritten on first frame).
        let transform_gizmo_renderer =
            TransformGizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, region_dimensions);
        // Per-object block lattice + floor grid (issue #29 S3): its line batch is
        // (re)built per frame from the grid-enabled nodes, so it starts empty.
        let scene_grid_renderer = SceneGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        // The world reference grid (issue #29 S5): the visible Points' tiled planes +
        // axes. Its batch is rebuilt per frame from the scene + camera, so empty here.
        let points_renderer = PointsRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        let infinite_grid_renderer = InfiniteGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        let view_cube_renderer =
            ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let mut onion_fog_renderer = OnionFogRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        // ADR 0007: build the GPU view-resolve pipelines once (reused every edit).
        #[cfg(feature = "gpu")]
        let gpu_resolver = GpuResolver::new(&gpu.device);
        // Upload the resolved grid as the fog's occupancy field. Per-chunk is the
        // DEFAULT (issue #28 S5b): one apron'd volume per resident chunk so fog still
        // renders at scale where a single whole-grid 3D texture would disable itself.
        // ADR 0007: a single-producer scene resolves its atlas on the GPU (no CPU densify)
        // under `--features gpu`; otherwise the CPU densify path.
        let fog_mode = FogMode::PerChunk;
        // Issue #58: DEMAND-DRIVEN fog. Onion-skin defaults OFF, so at startup the fog is
        // never drawn — skip the build entirely and mark the occupancy stale. The render
        // path builds it lazily the first frame onion-skin is enabled. If a startup config
        // ever opened WITH onion-skin on (and not in debug-face mode), build it now.
        let onion_active_at_startup =
            panel_state.layer_range.onion_skin && !panel_state.debug_face_orientation;
        let mut fog_occupancy_dirty = true;
        // Issue #59: the chunk-Z covering range the fog is built for (`None` until built,
        // or when the needed slab is full-range → no fog).
        let mut fog_built_chunk_z_range: Option<[i32; 2]> = None;
        let startup_density = panel_state.geometry.voxels_per_block;
        // Issue #59: the band-slab the onion fog needs at startup (`None` = full-range →
        // nothing outside the band to ghost → skip the fog build even with onion on).
        let startup_slab = if onion_active_at_startup {
            Self::fog_z_slab_for(panel_state.layer_range, region_dimensions[2])
        } else {
            None
        };
        if onion_active_at_startup {
            match startup_slab {
                Some(slab) => {
                    Self::build_fog_occupancy(
                        #[cfg(feature = "gpu")]
                        &gpu_resolver,
                        &mut onion_fog_renderer,
                        &gpu,
                        &panel_state.scene,
                        fog_mode,
                        region_dimensions,
                        startup_recentre,
                        startup_density,
                        Some(slab),
                        // ADR 0011 G5: no brick fog source seeded yet at startup — the first
                        // rebuild wires it. A chunkable scene has no fog source THIS one frame
                        // (fog appears after the first rebuild); a single-producer scene still
                        // resolves its GPU atlas inside `build_fog_occupancy`.
                        None,
                    );
                    fog_occupancy_dirty = false;
                    fog_built_chunk_z_range = Self::fog_covering_chunk_z_range(
                        panel_state.layer_range,
                        region_dimensions[2],
                        startup_density,
                    );
                    println!("fog: built at startup (onion-skin active, band-slab scoped)");
                }
                None => {
                    // Full-range band with onion on: no layers outside to ghost → no fog.
                    // Leave the occupancy stale; the render path skips the draw too.
                    println!("fog: skipped at startup (onion active but band full-range)");
                }
            }
        } else {
            println!("fog: skipped at startup (onion inactive)");
        }
        let thumbnail_renderer = ThumbnailRenderer::new(&gpu.device, &gpu.queue);

        // Kick off the VS auto-detect + scan on a background thread immediately;
        // results stream in over the next frames (no startup block).
        let palette = BlockPalette {
            status: "Scanning…".to_string(),
            ..BlockPalette::default()
        };
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

        // Issue #60 (ADR 0003 §7): spawn the background geometry-rebuild worker with
        // cloned GPU handles (wgpu 29 `Device`/`Queue` are `Send + Sync + Clone`, so the
        // worker builds the mesh's GPU buffers off the main thread). A large wholesale
        // rebuild dispatches here; the shell keeps rendering the current mesh until the
        // worker's result arrives, then swaps it in.
        let geometry_worker =
            GeometryWorker::spawn(gpu.device.clone(), gpu.queue.clone(), COLOR_TARGET_FORMAT);
        let geometry_generation = GenerationTracker::new();

        Self {
            window,
            surface,
            surface_config,
            gpu,
            egui_bridge,
            egui_winit_state,
            panel_state,
            cuboid_mesh_renderer,
            mesh_stale,
            brick_raymarch_renderer,
            incremental_brick_field,
            // ADR 0011 G5: seeded on the first rebuild — the fog reconstructs from it thereafter.
            fog_brick_field: None,
            brick_fallback_reported: false,
            geometry_worker,
            geometry_generation,
            geometry_async_outstanding: false,
            transform_gizmo_renderer,
            scene_grid_renderer,
            points_renderer,
            infinite_grid_renderer,
            view_cube_renderer,
            onion_fog_renderer,
            #[cfg(feature = "gpu")]
            gpu_resolver,
            fog_mode,
            thumbnail_renderer,
            palette,
            scan_handle,
            pending_groups: std::collections::VecDeque::new(),
            scan_total: None,
            scan_source_name: None,
            loaded_material: None,
            face_resolver: FaceResolver::auto(),
            region_dimensions,
            recentre_voxels: startup_recentre,
            app_core: AppCore::new(camera),
            measured_diameter,
            measured_band,
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
            fog_occupancy_dirty,
            fog_built_chunk_z_range,
            // Default to the full target until the first frame fills it in.
            last_viewport_px: [0, 0, width, height],
            context_menu_open_at: None,
            hovered_cube_zone: None,
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

    /// Re-resolve the current panel geometry into a fresh grid and rebuild the
    /// instance buffer. Honours the voxel cap (ARCHITECTURE.md §7): if the grid
    /// is too large the 3D rebuild is skipped and the panel shows a warning.
    /// Upload the resolved grid into the fog renderer's occupancy field, dispatching
    /// on the active [`FogMode`] (issue #28). Per-chunk (the S5b default) builds one
    /// apron'd volume per resident chunk packed into a small 3D atlas — so fog still
    /// renders at scale where the legacy whole-grid single 3D texture would exceed
    /// `max_texture_dimension_3d` and disable itself. Shared by the initial upload and
    /// every `rebuild_geometry` re-upload so both code paths honour the same mode.
    // The CPU fog densify (ADR 0007): now the FALLBACK for multi-producer scenes and
    // default (no `--features gpu`) builds — single-producer scenes resolve on the GPU via
    // `try_install_gpu_per_chunk_fog`. Still deprecated; DELETE this allow + the CPU densify
    // when every producer the live app composes resolves on the GPU (P2+).
    #[allow(deprecated)]
    fn upload_fog_occupancy(
        onion_fog_renderer: &mut OnionFogRenderer,
        fog_mode: FogMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &VoxelGrid,
        voxels_per_block: u32,
        fog_z_slab: Option<FogZSlab>,
    ) {
        match fog_mode {
            // WholeGrid is the legacy single-3D-texture path (issue #12): it densifies
            // the whole grid and cannot be Z-slabbed without changing its texture layout,
            // so it ignores the slab. Only the default PerChunk path is scoped (#59).
            FogMode::WholeGrid => onion_fog_renderer.upload_grid(device, queue, grid),
            FogMode::PerChunk => onion_fog_renderer.upload_grid_per_chunk(
                device,
                queue,
                grid,
                voxels_per_block,
                fog_z_slab,
            ),
        }
    }

    /// ADR 0007 live call-site swap: try the GPU per-chunk fog atlas for a SINGLE ported
    /// producer (`Scene::single_producer` — one SDF Tool, SketchTool, or `DebugClouds`
    /// Part). On a hit the producer's covering-chunk fog atlas is GPU-resolved (option (C)
    /// compaction drops empty-interior tiles) and installed directly — **no CPU densify, no
    /// occupancy readback** (only a tiny per-chunk flags readback to size the atlas).
    /// Returns `true` when the GPU path took over; `false` to keep the CPU densify
    /// (multi-producer scene, a dispatch too large for this device, or a non-empty set that
    /// still overflows the atlas budget). Mirrors `shot::try_install_gpu_per_chunk_fog`, but
    /// reuses the App's persistent `resolver` instead of rebuilding the pipelines per call.
    ///
    /// Taken as an associated fn over explicit fields (not `&mut self`) so the caller's
    /// disjoint borrows of `gpu_resolver` / `onion_fog_renderer` / `gpu` / `scene` don't
    /// collide.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn try_install_gpu_per_chunk_fog(
        resolver: &GpuResolver,
        fog: &mut OnionFogRenderer,
        gpu: &GpuContext,
        scene: &Scene,
        // ADR 0011 G5: the display frame (dimensions + recentre), passed as scalars — the
        // dense `VoxelGrid` this used to read them from is retired.
        region_dimensions: [u32; 3],
        recentre_voxels: [i64; 3],
        voxels_per_block: u32,
        fog_z_slab: Option<FogZSlab>,
    ) -> bool {
        profiling::scope!("fog_gpu_resolve");
        let Some(producer) = scene.single_producer() else {
            return false;
        };
        let Some(atlas) = resolver.resolve_single_producer_fog_atlas(
            &gpu.device,
            &gpu.queue,
            &producer,
            region_dimensions,
            recentre_voxels,
            voxels_per_block,
            fog_z_slab,
        ) else {
            return false;
        };
        fog.install_per_chunk_atlas(
            &gpu.device,
            &gpu.queue,
            &atlas.texture,
            &atlas.world_origins,
            PerChunkAtlasGeometry {
                chunk_extent: atlas.chunk_extent,
                pad: atlas.pad,
                tiles_per_axis: atlas.tiles_per_axis,
                atlas_dim: atlas.atlas_dim,
            },
        );
        // The compacted non-empty set can still exceed the atlas budget; if the install
        // couldn't activate, fall back to the CPU densify rather than show no fog.
        fog.per_chunk_active()
    }

    /// The onion-fog Z-slab (issue #59) the fog occupancy must cover for the current
    /// `layer_range` over a grid of height `grid_z` (Z-up: layers are Z-slices). The band
    /// is derived the SAME way `AppCore::onion_fog_params` derives its onion-Z span —
    /// solid band `[lower, upper.min(grid_z−1)]`, widened by `onion_depth` (clamped 1..8
    /// when onion is on) — so the slab bounds exactly the occupancy the raymarch samples.
    /// `None` when the band is FULL-RANGE (nothing outside to ghost → skip the fog build).
    ///
    /// This assumes onion-skin is active (the only path that draws fog); the caller gates
    /// on `onion_active` before building fog at all.
    fn fog_z_slab_for(layer_range: LayerRange, grid_z: u32) -> Option<FogZSlab> {
        let band = LayerBand {
            band_min: layer_range.lower,
            band_max: layer_range.upper.min(grid_z.saturating_sub(1)),
            onion_depth: layer_range.onion_depth.clamp(1, 8),
        };
        FogZSlab::for_band(band, grid_z)
    }

    /// The inclusive chunk-Z covering range the fog would build for `layer_range` over a
    /// grid of height `grid_z` at `density` (issue #59). This is the CHUNK-granular
    /// identity the band-aware rebuild logic compares: a band scrub that stays within the
    /// same covering chunk-Z rows reuses the built atlas; one that shifts the rows (or
    /// crosses the full-range boundary, `None`) rebuilds. Rebuilding at chunk (not voxel)
    /// granularity avoids a rebuild on every voxel step of a scrub-drag.
    fn fog_covering_chunk_z_range(
        layer_range: LayerRange,
        grid_z: u32,
        density: u32,
    ) -> Option<[i32; 2]> {
        let chunk_extent = CHUNK_BLOCKS * density.max(1);
        Self::fog_z_slab_for(layer_range, grid_z)
            .and_then(|slab| slab.covering_chunk_z_range(chunk_extent))
    }

    /// Build + upload the onion-fog occupancy field for the current display frame. ADR 0011
    /// G5 — the dense-grid retirement: this takes the region dimensions + recentre as SCALARS
    /// (the `VoxelGrid` argument is gone) and sources the occupancy from one of three places,
    /// in order:
    ///
    ///   1. **The brick sink** (`fog_brick = Some`): ANY chunkable scene, on EVERY build. Fog
    ///      is boolean occupancy, so `build_per_chunk_fog_occupancy_from_bricks` reconstructs
    ///      the per-chunk tiles from the boundary set — mixed-material blocks and a loaded VS
    ///      material included (their DISPLAY stays on the mesh). Byte-identical to the retired
    ///      CPU densify (`brick_sourced_fog_matches_cpu_densify_byte_for_byte`), so the fog
    ///      SHADER + the `onion-fog-perchunk` golden are unchanged. This is the common path.
    ///   2. **The GPU single-producer atlas** (`--features gpu`): a lone ported producer
    ///      resolves its per-chunk fog atlas on the GPU (no CPU densify — the ~592ms/edit
    ///      bottleneck, ADR 0007).
    ///   3. **A transient Part-only densify — the SOLE surviving dense resolve.** Reached only
    ///      when there is no brick source AND no GPU atlas: a NON-chunkable (Part-only) scene,
    ///      which resolves to a DEGENERATE, EMPTY region. It is resolved on demand
    ///      (`AppCore::resolve_scene`), densified, uploaded, and the grid is DROPPED — never
    ///      carried in a field. A **chunkable** scene can never legitimately land here (it
    ///      always has a brick source); if one somehow does — an empty brick field — fog is
    ///      SKIPPED rather than O(volume)-resolved, which is the retirement invariant.
    ///
    /// Issue #58: demand-driven — invoked only when the fog will be drawn. Taken as an
    /// associated fn over explicit disjoint fields (not `&mut self`) so the caller's borrows
    /// don't collide. Callers own clearing `fog_occupancy_dirty`.
    #[allow(clippy::too_many_arguments)]
    fn build_fog_occupancy(
        #[cfg(feature = "gpu")] gpu_resolver: &GpuResolver,
        onion_fog_renderer: &mut OnionFogRenderer,
        gpu: &GpuContext,
        // Drives the GPU single-producer atlas (path 2) AND the transient Part-only resolve
        // (path 3) — used on every build now, so no longer `unused` on non-gpu.
        scene: &Scene,
        fog_mode: FogMode,
        // ADR 0011 G5: the display frame as scalars (the dense grid is retired).
        region_dimensions: [u32; 3],
        recentre_voxels: [i64; 3],
        density: u32,
        // Issue #59: the onion-fog Z-slab (band ± onion_depth) to scope the covering
        // chunk set to. `None` covers the whole grid (unused by the live onion path,
        // which always passes a slab, but kept for a would-be whole-grid caller).
        fog_z_slab: Option<FogZSlab>,
        // ADR 0011 G5: the brick field + covering chunks the fog sources its per-chunk
        // occupancy from, for any chunkable scene (path 1). Plain CPU
        // (`build_per_chunk_fog_occupancy_from_bricks`), so the non-gpu binary takes it too
        // (the universal brick-fog law).
        fog_brick: Option<&FogBrickSource>,
    ) {
        profiling::scope!("fog_upload");
        // (1) The primary path — reconstruct the per-chunk fog occupancy from the two-layer
        // chunks (coarse/interior) + the brick field's sculpted records/atlas (boundary),
        // with NO dense `VoxelGrid`. PerChunk only (WholeGrid is the legacy
        // single-3D-texture debug mode, which never rides the brick path).
        if let (Some(source), FogMode::PerChunk) = (fog_brick, fog_mode) {
            profiling::scope!("fog_brick_fill");
            let occupancy = voxel_worker::build_per_chunk_fog_occupancy_from_bricks(
                &source.build,
                &source.two_layer_chunks,
                region_dimensions,
                recentre_voxels,
                fog_z_slab,
            );
            onion_fog_renderer.upload_per_chunk_occupancy(&gpu.device, &gpu.queue, &occupancy);
            println!("fog: per-chunk mode (brick sink)");
            return;
        }
        // (2) The GPU single-producer atlas (no CPU densify).
        #[cfg(feature = "gpu")]
        let gpu_fog_installed = Self::try_install_gpu_per_chunk_fog(
            gpu_resolver,
            onion_fog_renderer,
            gpu,
            scene,
            region_dimensions,
            recentre_voxels,
            density,
            fog_z_slab,
        );
        #[cfg(not(feature = "gpu"))]
        let gpu_fog_installed = false;
        if gpu_fog_installed {
            println!("fog: per-chunk mode (GPU atlas)");
            return;
        }
        // Neither a brick source nor a GPU atlas. A CHUNKABLE scene must never reach the dense
        // resolve below (the retired O(volume) densify) — it always has a brick fog source, so
        // landing here means an empty brick field; skip fog (honest degradation) rather than
        // reintroduce the O(volume) resolve. THIS is the load-bearing retirement invariant.
        if scene.has_chunkable_extent(density) {
            println!(
                "fog: skipped — chunkable scene had no brick fog source (never dense-resolve)"
            );
            return;
        }
        // (3) The SOLE tolerated transient dense resolve (ADR 0011 G5): a NON-chunkable
        // Part-only scene — a degenerate, empty region (issue #58, demand-driven). Resolve it,
        // densify + upload, and DROP the grid immediately; it is never carried in a field.
        profiling::scope!("fog_transient_partonly_densify");
        println!("fog: per-chunk mode (transient Part-only densify)");
        let transient_grid = AppCore::resolve_scene(scene, density);
        Self::upload_fog_occupancy(
            onion_fog_renderer,
            fog_mode,
            &gpu.device,
            &gpu.queue,
            &transient_grid,
            density,
            fog_z_slab,
        );
        // `transient_grid` is dropped here — the retirement holds.
    }

    /// Re-resolve the grid + GPU geometry for the current scene. Camera UX change:
    /// this NEVER moves the camera — edits keep the orbit target + distance fixed.
    /// Explicit framing (startup fit, Home/Fit, Focus) is handled by their own paths.
    /// The EFFECTIVE layer-clip band the render path will apply this frame for a grid of
    /// `grid_z` layers (issue #12 / #60 M2). Mirrors exactly what `update_uniforms` computes
    /// (the scrubber → shader band, plus the debug-faces override that forces FULL), so the
    /// async worker can build the mesh already clipped to THIS band and the swap frame's
    /// `rebuild_for_band` becomes a no-op (no full main-thread re-mesh on the swap).
    fn current_layer_band(&self, grid_z: u32) -> LayerBand {
        // Debug-faces mode bypasses the band (the instanced check sees the whole model), so
        // force FULL — matching `update_uniforms`' `effective_band`.
        if self.panel_state.debug_face_orientation {
            return LayerBand::FULL;
        }
        let layer_range = self.panel_state.layer_range;
        if layer_range.is_full_range(grid_z) && !layer_range.onion_skin {
            LayerBand::FULL
        } else {
            LayerBand {
                band_min: layer_range.lower,
                // `upper` is the last visible layer index; clamp into the grid so a
                // full-range upper (== grid_z) still includes the top layer.
                band_max: layer_range.upper.min(grid_z.saturating_sub(1)),
                onion_depth: if layer_range.onion_skin {
                    layer_range.onion_depth.clamp(1, 8)
                } else {
                    0
                },
            }
        }
    }

    fn rebuild_geometry(&mut self) {
        profiling::scope!("rebuild_geometry");
        let density = self.panel_state.geometry.voxels_per_block;

        // Delegate the headless resolve (S2/S3 targeted invalidation + assemble) to
        // `AppCore::rebuild`, then consume its output here in the shell: build the
        // GPU cuboid mesh and upload the fog (the camera is NOT touched). A density whose
        // single-chunk voxel capacity exceeds the bound is rejected with the store
        // untouched, so we surface the cap warning and bail.
        // ADR 0011 G5: fog is boolean occupancy, so it sources from the brick field for
        // EVERY chunkable procedural scene — mixed-material-block scenes and loaded-VS-material
        // scenes included (their DISPLAY stays on the mesh, but the fog brick occupancy is exact
        // regardless of materials). `AppCore::rebuild` NO LONGER assembles a dense fog
        // `VoxelGrid` at all — the last dense-shaped display consumer is retired. Only a
        // NON-chunkable Part-only scene falls to the shell's transient (degenerate, empty)
        // densify inside `build_fog_occupancy`.
        let chunkable = self.panel_state.scene.has_chunkable_extent(density);
        // Only the gpu brick DISPLAY gate reads this (a loaded material forces the mesh
        // display); the fog mirror + stream decision no longer depend on it.
        #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
        let loaded_material = self.loaded_material.is_some();
        // ADR 0011 G5 — the universal brick-fog gate. The FOG mirror (`fog_brick_field` /
        // `incremental_brick_field`, both plain CPU) is maintained for ANY chunkable scene on
        // ANY build: fog needs only boolean occupancy, so a loaded VS material (mesh-only
        // shading) still fog-sources from bricks. The DISPLAY raymarch keeps its stricter
        // conditions inside the block (`--features gpu` + `!loaded_material` + brick
        // representability) — a mixed-material or textured scene meshes its display while the
        // fog still comes from `build`. The dense fog `VoxelGrid` stream is retired entirely
        // (no `stream_fog_grid` flag): `AppCore::rebuild` never assembles one.
        let brick_fog_gate = chunkable;
        let RebuildOutput {
            region_dimensions,
            two_layer_chunks,
            recentre_voxels,
            recentre_shift_voxels,
            incremental_dirty_chunks,
        } = match self.app_core.rebuild(&self.panel_state.scene, density) {
            RebuildOutcome::DensityRejected {
                chunk_voxels_millions,
            } => {
                self.panel_state.voxel_cap_warning_millions = Some(chunk_voxels_millions);
                return;
            }
            RebuildOutcome::Built(output) => {
                self.panel_state.voxel_cap_warning_millions = None;
                output
            }
        };

        // Read the OLD grid_z before reassigning `self.region_dimensions`, for the layer-band
        // rescale below (Z-up: layers are Z-slices, index 2).
        let previous_grid_z = self.region_dimensions[2];
        // ADR 0010 E5: the cuboid mesh renderer is the sole voxel render path AND it meshes
        // THROUGH the two-layer store — a coarse-solid block is a one-box fast path, a boundary
        // block its microblock cuboids, seam faces culled via the per-face solidity flags (no
        // densified apron). The two-layer chunks are owned, so they outlive `app_core`.
        //
        // Issue #55 — chunk-granular incremental GPU-buffer re-mesh: when the edit LOCALISED
        // (`incremental_dirty_chunks` is `Some`), re-mesh + re-upload ONLY the dirty chunks
        // (dilated by the 26-neighbourhood seam footprint), keeping every untouched chunk's GPU
        // buffer in place — the exact per-edit latency #40 fixed for the dense path, now on the
        // two-layer path. A wholesale re-mesh (`None`: first build, density change, or a
        // region-spanning Part edit) recreates the renderer from the full resident set. Both
        // yield a byte-identical buffer set (proven in `cuboid_mesh`'s incremental parity test).
        // The mesh's frame parameters (the async request moves `two_layer_chunks`, so read
        // `region_dimensions` — a `Copy` scalar — here). Issue #60: a SMALL/incremental edit
        // stays inline; a large WHOLESALE rebuild dispatches to the worker.
        let grid_dimensions = region_dimensions;
        // Issue #60 M2: the effective layer-clip band the render path will apply this frame.
        // The async worker builds the mesh already clipped to this band so the swap frame's
        // `rebuild_for_band` is a no-op (no full main-thread re-mesh — the hitch #60 removed).
        let band = self.current_layer_band(grid_dimensions[2]);
        // Issue #60 C1: classify the edit's shape and route it. The load-bearing rule is that
        // while an async wholesale build is OUTSTANDING (dispatched, not yet installed) the
        // currently-installed renderer is STALE (S0, the worker is building S1), so an
        // incremental edit must NOT inline-patch it (the Frankenstein mesh). `route_geometry_
        // rebuild` sends EVERY edit to a fresh wholesale-async dispatch while outstanding, and
        // resumes the inline fast-paths only once nothing is outstanding.

        // Issue #60 C1: classify the edit ONCE (shared by the brick sink and the mesh path).
        // While an async wholesale build is OUTSTANDING every edit routes to a fresh
        // wholesale-async dispatch (never inline-patch a stale artifact); only with nothing
        // outstanding do the inline fast-paths resume.
        let edit_shape = match &incremental_dirty_chunks {
            Some(_) => EditShape::Incremental,
            None => EditShape::Wholesale {
                chunk_count: two_layer_chunks.len(),
            },
        };
        // The BRICK MIRROR's patch-vs-wholesale decision (below). It shares the C1 interlock
        // with the mesh path but is INDEPENDENT of mesh staleness — the brick FOG mirror is
        // maintained for any chunkable scene on any build, so it uses the plain route. The
        // mesh's own route (`route_mesh_build`, after the brick block) additionally folds in
        // mesh staleness + the brick-display-engaged skip.
        let route = route_geometry_rebuild(
            self.geometry_async_outstanding,
            edit_shape,
            ASYNC_REBUILD_CHUNK_THRESHOLD,
        );

        // ADR 0011 G1/G3/G5: refresh the brick field from THIS rebuild's resident chunk set
        // (the same boundary set the mesher consumes), before the async route can move
        // `two_layer_chunks`. When the mesh route is `InlineIncremental` (an incremental
        // edit, nothing outstanding) the field is PATCHED (G3): only the dirty chunks are
        // re-evaluated and only their atlas slots written. Every other route — wholesale,
        // or ANY edit while an async build is outstanding (the C1 interlock) — rebuilds the
        // field WHOLESALE and resets the incremental mirror, so the brick sink never
        // patches from a state the mesh path treats as stale. A non-chunkable / empty scene
        // clears the field to the mesh fallback.
        //
        // The FOG MIRROR (`build`, `fog_brick_field`, `incremental_brick_field`) is plain CPU
        // and runs for ANY chunkable scene on ANY build (`brick_fog_gate`) — the universal
        // brick-fog law (ADR 0011 G5). The gpu DISPLAY raymarch (installed in section (B)
        // below) keeps its stricter `--features gpu` + `!loaded_material` + representability
        // conditions; a mixed-material or textured scene meshes its display while the fog still
        // reconstructs from `build`. The block YIELDS whether the brick DISPLAY was installed
        // this rebuild — the mesh-skip decision below reads it (always `false` on non-gpu).
        let brick_display_installed = {
            #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
            let mut brick_display_installed = false;
            if brick_fog_gate {
                profiling::scope!("brick_field_build");
                // (A) Maintain the CPU brick MIRROR — the FOG occupancy source (ADR 0011 G5),
                // built for any chunkable scene regardless of display representability. Patch
                // iff the mesh route is InlineIncremental AND a mirror already exists (mirrors
                // `route_geometry_rebuild`, so the C1 interlock composes: outstanding ⇒ route
                // != InlineIncremental ⇒ wholesale here too). `update` (the GPU atlas-slot
                // descriptor) is consumed only by the gpu display in (B); on a non-gpu build
                // the in-place mirror patch is what matters and the descriptor is discarded.
                let patch_mirror = matches!(route, RebuildRoute::InlineIncremental)
                    && self.incremental_brick_field.is_some();
                #[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
                let (build, update): (
                    voxel_worker::BrickFieldBuild,
                    Option<voxel_worker::BrickFieldUpdate>,
                ) = if patch_mirror {
                    let dirty = incremental_dirty_chunks
                        .as_ref()
                        .expect("InlineIncremental ⇒ incremental_dirty_chunks is Some");
                    let field = self
                        .incremental_brick_field
                        .as_mut()
                        .expect("patch_mirror ⇒ Some");
                    debug_assert_eq!(
                        field.brick_edge_voxels(),
                        density,
                        "an incremental edit never changes density (it routes wholesale)"
                    );
                    let update = field.apply_dirty_update(&two_layer_chunks, dirty);
                    (field.to_build(), Some(update))
                } else {
                    // Wholesale (re)build; RESET the mirror so the next incremental edit
                    // patches from a known-good full field.
                    let build = voxel_worker::build_brick_field(&two_layer_chunks, density);
                    self.incremental_brick_field = if build.brick_records.is_empty() {
                        None
                    } else {
                        Some(voxel_worker::IncrementalBrickField::from_wholesale(&build))
                    };
                    (build, None)
                };

                if build.brick_records.is_empty() {
                    // The edit emptied the field — no fog source, no display brick.
                    self.incremental_brick_field = None;
                    self.fog_brick_field = None;
                } else {
                    // (B) DISPLAY: install/patch the GPU raymarch renderer ONLY under
                    // `--features gpu`, with NO loaded VS material (a material needs the mesh's
                    // per-face shading — bricks carry only categorical block ids), and ONLY
                    // when the boundary set is brick-REPRESENTABLE (single-material blocks + one
                    // overlay). A mixed-material or textured scene keeps the mesh display but
                    // still fog-sources from `build` (fog is boolean occupancy — the G5 insight).
                    #[cfg(feature = "gpu")]
                    if !loaded_material {
                    match voxel_worker::brick_representable_overlay(&two_layer_chunks) {
                        Some(overlay_active) => {
                            let pyramid =
                                voxel_worker::ClipmapPyramid::from_chunks(&two_layer_chunks);
                            // ADR 0011 interior elision: the record set is SURFACE-ONLY by
                            // construction (`build_brick_field` fuses the occlusion decision
                            // into emission — a fully-occluded interior block never becomes a
                            // record, so nothing here needs a second mask pass). For a large
                            // solid the per-edit record upload is ∝surface, not ∝volume.
                            // Interiors live in the two-layer chunks: the clip-map (above)
                            // and the fog box-fill both derive from the chunks.
                            let gpu_records = voxel_worker::pack_gpu_records(&build, |_| false);
                            // Patch in place iff we produced an incremental update AND a
                            // renderer already holds a field; otherwise (wholesale, or the
                            // display re-engaging from a mesh fallback) install fresh.
                            if let (Some(update), true) =
                                (update.as_ref(), self.brick_raymarch_renderer.is_some())
                            {
                                if update.atlas_grew {
                                    println!(
                                        "brick: atlas grew — full re-pack ({} sculpted slots)",
                                        build.sculpted_brick_count()
                                    );
                                }
                                let renderer = self
                                    .brick_raymarch_renderer
                                    .as_mut()
                                    .expect("is_some checked");
                                renderer.patch_brick_field(
                                    &self.gpu.device,
                                    &self.gpu.queue,
                                    &build,
                                    update,
                                    &gpu_records,
                                    &pyramid,
                                    recentre_voxels,
                                    overlay_active,
                                );
                            } else {
                                let renderer =
                                    self.brick_raymarch_renderer.get_or_insert_with(|| {
                                        BrickRaymarchRenderer::new(
                                            &self.gpu.device,
                                            &self.gpu.queue,
                                            COLOR_TARGET_FORMAT,
                                        )
                                    });
                                renderer.install_brick_field(
                                    &self.gpu.device,
                                    &self.gpu.queue,
                                    &build,
                                    &gpu_records,
                                    &pyramid,
                                    recentre_voxels,
                                    overlay_active,
                                );
                            }
                            brick_display_installed = true;
                            self.brick_fallback_reported = false;
                        }
                        None => {
                            // Not display-representable: mesh display, fog still from bricks.
                            if !self.brick_fallback_reported {
                                println!(
                                    "brick: scene not representable (a block mixes materials \
                                     or blocks disagree on the on-face grid) — mesh display, \
                                     fog from bricks"
                                );
                                self.brick_fallback_reported = true;
                            }
                        }
                    }
                    } // end `#[cfg(feature = "gpu")] if !loaded_material` (the display gate)
                    // Fog sources from THIS rebuild's boundary set regardless of the display
                    // path — the last dense-shaped display consumer is gone (ADR 0011 G5). This
                    // runs on EVERY build (non-gpu included) and for a loaded VS material, whose
                    // display stays on the mesh above. The covering chunks ride along because
                    // the record set is SURFACE-ONLY (interior fog occupancy is chunk-sourced);
                    // the clone is O(chunks) Arc refcount bumps, never a deep chunk copy.
                    self.fog_brick_field = Some(FogBrickSource {
                        build,
                        two_layer_chunks: two_layer_chunks.clone(),
                    });
                }
            } else {
                // Non-chunkable (a Part-only field): no brick mirror, no brick fog source — the
                // pre-G5 fog path (single-producer GPU atlas, else the CPU densify on the
                // transient/streamed grid) handles it. Runs on every build now.
                self.incremental_brick_field = None;
                self.fog_brick_field = None;
            }
            // Clearing the gpu raymarch display when it did not install is a gpu-only concern —
            // the renderer is `None` on non-gpu builds, so this whole cleanup compiles out.
            #[cfg(feature = "gpu")]
            if !brick_display_installed {
                if let Some(renderer) = &mut self.brick_raymarch_renderer {
                    renderer.clear_brick_field();
                }
            }
            brick_display_installed
        };

        // Brick-display perf follow-up to epic #64: the fallback cuboid mesh is DRAWN only when
        // the brick raymarch is not engaged. Engagement mirrors the per-frame gate
        // (`brick_raymarch_engaged`): a field installed this rebuild AND no debug-face mode.
        // (A loaded VS material skips the display install ⇒ `brick_display_installed` stays
        // false, so it is already covered.) When engaged the mesh is redundant → SKIP the build
        // and mark it stale; the C1 interlock composes via `route_mesh_build` (a stale mesh, like
        // an outstanding async build, is never inline-patched — it rebuilds wholesale when next
        // needed). On non-gpu builds `brick_display_installed` is always false → always Build.
        let brick_display_engaged =
            brick_display_installed && !self.panel_state.debug_face_orientation;
        let mesh_route = route_mesh_build(
            brick_display_engaged,
            self.mesh_stale,
            self.geometry_async_outstanding,
            edit_shape,
            ASYNC_REBUILD_CHUNK_THRESHOLD,
        );

        match mesh_route {
            MeshBuildRoute::Skip => {
                // The brick raymarch is the display — skip the ~333ms mesh build. Mark the mesh
                // stale so the next edit that needs it rebuilds wholesale. Bump the generation
                // and drop any outstanding async so a stale in-flight mesh result is discarded on
                // arrival (`poll_geometry_worker`) instead of being swapped in behind the brick.
                self.geometry_generation.next_generation();
                self.geometry_async_outstanding = false;
                self.mesh_stale = true;
            }
            MeshBuildRoute::Build(RebuildRoute::InlineIncremental) => {
                // Issue #54/#55 fast path: an incremental dirty-chunk re-mesh is already a
                // few chunks — build it inline (no worker hop, no added latency). Reached ONLY
                // when nothing is outstanding, so the installed renderer reflects the latest
                // resolve and patching it in place is sound.
                //
                // Bump the generation so any (phantom) in-flight result is discarded on
                // arrival — the tracker rejects a non-newest generation.
                let dirty = incremental_dirty_chunks
                    .expect("InlineIncremental is only routed for an incremental edit");
                self.geometry_generation.next_generation();
                profiling::scope!("cuboid_incremental_two_layer");
                self.cuboid_mesh_renderer.incremental_rebuild_from_two_layer_chunks(
                    &self.gpu.device,
                    &two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    &dirty,
                );
                // Reached only with `mesh_stale == false` (a stale mesh forces wholesale via
                // `route_mesh_build`), so the in-place patch is sound; keep it non-stale.
                self.mesh_stale = false;
            }
            MeshBuildRoute::Build(RebuildRoute::WholesaleAsync) => {
                // Issue #60: dispatch a WHOLESALE rebuild to the worker so the UI never
                // freezes (the ~3s classify ran above on the main thread; the heavy mesh CPU
                // build + GPU upload is what goes async). Stamp a fresh generation, send the
                // owned FULL covering set (the `AppCore` resident cache is always current on
                // the main thread, so a full wholesale is correct even when the edit itself
                // was incremental — the C1 interlock), and keep the CURRENT renderer drawing
                // (stale-while-rebuilding). Mark the async build OUTSTANDING so the NEXT edit
                // also routes here instead of inline-patching the still-stale renderer. The
                // result is polled + swapped in the event loop (`poll_geometry_worker`).
                let generation = self.geometry_generation.next_generation();
                self.geometry_async_outstanding = true;
                self.geometry_worker.dispatch(GeometryRebuildRequest {
                    generation,
                    two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    band,
                });
                // The worker owns the (re)build now; the outstanding flag carries the C1
                // interlock, so the mesh is no longer treated as skip-stale.
                self.mesh_stale = false;
            }
            MeshBuildRoute::Build(RebuildRoute::WholesaleInline) => {
                // A small wholesale rebuild (at/below the threshold), nothing outstanding:
                // build inline — cheap enough not to hitch a frame, and it avoids the worker's
                // one-frame swap latency. Bump the generation so any phantom in-flight result
                // is discarded on arrival. Build at the active band so the mesh matches the
                // render path immediately (no swap-frame re-mesh — same M2 reasoning).
                self.geometry_generation.next_generation();
                self.cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks_banded(
                    &self.gpu.device,
                    &self.gpu.queue,
                    COLOR_TARGET_FORMAT,
                    &two_layer_chunks,
                    grid_dimensions,
                    recentre_voxels,
                    density,
                    band,
                );
                self.mesh_stale = false;
            }
        }

        // Camera UX invariant: an edit must NEVER re-frame the view. The composite is
        // re-centred on the world origin every rebuild, so any extent change (add /
        // delete / offset) — and any density change, since the recentre is in voxels —
        // shifts the floating origin by `recentre_shift_voxels`. The camera target is
        // pinned in that same recentred render frame (voxels), so without compensation
        // the whole world would slide under the fixed camera (the "jump to centre /
        // fit everything" the user reported). Subtract the shift so the target tracks
        // the SAME world point as the origin floats — net zero view motion. The shift
        // is `[0,0,0]` on the first build, and the explicit Fit/Home/Focus actions
        // OVERWRITE the target afterwards (they run on their own paths, not here), so
        // they keep re-framing exactly as before; orbit/pan/zoom are untouched.
        if recentre_shift_voxels != [0; 3] {
            self.app_core.camera.target -= glam::Vec3::new(
                recentre_shift_voxels[0] as f32,
                recentre_shift_voxels[1] as f32,
                recentre_shift_voxels[2] as f32,
            );
        }
        // Issue #12: clamp/rescale the layer band to the new grid_z (re-snapping to block
        // multiples when snapping is on) BEFORE the fog build below, so the fog slab (#59)
        // is computed against the SAME rescaled band the render path will draw this frame
        // (the render path runs after this rebuild returns and reads the rescaled band).
        // Z-up: index 2. `previous_grid_z` was captured before `grid` was reassigned.
        self.panel_state.layer_range.rescale_to_grid_z(
            previous_grid_z,
            region_dimensions[2],
            density,
        );

        // Re-upload the fog's occupancy field for the new grid — but ONLY when the fog
        // is actually drawn (issue #58). Onion-skin defaults OFF, so the common edit path
        // does ZERO fog work: mark the occupancy stale and let the render path build it
        // lazily the first frame onion-skin is on. When onion IS active, build it now (the
        // edit changed the geometry the fog visualises) so this frame renders correctly.
        //
        // Issue #59: scope the build to the band-slab (band ± onion_depth). A full-range
        // band (`None` slab) has nothing outside to ghost → skip the build (and draw)
        // entirely even with onion on. Record the covering chunk-Z range so the render
        // path's band-aware reuse can tell a within-slab scrub (reuse) from one that
        // shifts the slab (rebuild).
        let onion_active =
            self.panel_state.layer_range.onion_skin && !self.panel_state.debug_face_orientation;
        let new_grid_z = region_dimensions[2];
        if onion_active {
            match Self::fog_z_slab_for(self.panel_state.layer_range, new_grid_z) {
                Some(slab) => {
                    Self::build_fog_occupancy(
                        #[cfg(feature = "gpu")]
                        &self.gpu_resolver,
                        &mut self.onion_fog_renderer,
                        &self.gpu,
                        &self.panel_state.scene,
                        self.fog_mode,
                        region_dimensions,
                        recentre_voxels,
                        density,
                        Some(slab),
                        self.fog_brick_field.as_ref(),
                    );
                    self.fog_occupancy_dirty = false;
                    self.fog_built_chunk_z_range = Self::fog_covering_chunk_z_range(
                        self.panel_state.layer_range,
                        new_grid_z,
                        density,
                    );
                }
                None => {
                    // Full-range band with onion on: no ghost layers → no fog build/draw.
                    self.fog_occupancy_dirty = true;
                    self.fog_built_chunk_z_range = None;
                    println!("fog: skipped on rebuild (band full-range)");
                }
            }
        } else {
            self.fog_occupancy_dirty = true;
            self.fog_built_chunk_z_range = None;
            println!("fog: skipped on rebuild (onion inactive)");
        }
        // The transform gizmo (issue #29 S2) is sized + positioned from the SELECTED
        // node in the per-frame render path (it must track selection changes, which
        // don't trigger a geometry rebuild), not here. The per-object block lattice +
        // floor grid (issue #29 S3) is likewise (re)batched per frame from the
        // grid-enabled nodes — a per-node toggle needs no scene re-resolve.

        self.region_dimensions = region_dimensions;
        self.recentre_voxels = recentre_voxels;
        self.measured_band = (u32::MAX, u32::MAX); // force a re-measure next frame.
    }

    /// Issue #60 (ADR 0003 §7): poll the geometry worker for a finished wholesale
    /// rebuild and, if it is NOT stale, swap it in + request a redraw. Called each frame
    /// in the event loop. Non-blocking — the app never waits on the worker.
    ///
    /// Stale-while-rebuilding: until a fresh result arrives, the current
    /// `cuboid_mesh_renderer` keeps drawing. On arrival, the [`GenerationTracker`] decides
    /// whether the result is still the newest dispatched (accept + swap) or was superseded
    /// by a later edit (discard). The worker drains-to-latest, so at most the newest built
    /// renderer is here; the tracker guards against a build that a mid-flight edit
    /// (wholesale OR incremental — both bump the generation) already superseded.
    fn poll_geometry_worker(&mut self) {
        let Some(result) = self.geometry_worker.try_recv_result() else {
            return;
        };
        if !self.geometry_generation.accepts(result.generation) {
            // A later edit superseded this build — discard it (the stale mesh, or the newer
            // inline/incremental result, is already what's showing). The superseding edit set
            // its own outstanding state (a re-dispatched wholesale keeps it `true`; an inline
            // edit reached only when nothing was outstanding leaves it `false`), so we do NOT
            // touch `geometry_async_outstanding` here.
            return;
        }
        // Issue #60 M1: a `None` renderer means the worker's build PANICKED (it logged to
        // stderr and stayed alive). Keep the current (stale) mesh and leave the outstanding
        // flag SET so the next edit re-dispatches a fresh wholesale — never silently wedge.
        let Some(renderer) = result.renderer else {
            return;
        };
        // Fresh: swap the freshly-built renderer in (GPU buffers already uploaded on the
        // worker) and redraw so the new mesh shows this frame. The newest dispatched build is
        // now installed, so no async build is outstanding — the inline fast-paths resume
        // (issue #60 C1).
        self.cuboid_mesh_renderer = renderer;
        self.geometry_async_outstanding = false;
        // A freshly built worker mesh reflects the latest resolve — never stale.
        self.mesh_stale = false;
        self.window.request_redraw();
    }

    /// Rebuild the fallback cuboid mesh IF it is stale and about to become the display
    /// (brick-display perf follow-up to epic #64). The mesh is skipped while the ADR 0011 brick
    /// raymarch is engaged; a debug-face toggle or a loaded-material change are pure per-frame
    /// display flags that can drop that engagement WITHOUT a `scene_changed` rebuild, so the
    /// skipped mesh would otherwise be drawn stale/empty. This closes that gap: called every
    /// frame before the voxel draw, it is a no-op unless the mesh is stale AND the brick will
    /// not draw. The rebuild is WHOLESALE + inline from the current resident two-layer set
    /// (the scene is unchanged — no re-resolve; same build path as startup), a one-off
    /// ~hundreds-of-ms hitch on a debug/material toggle, never per edit.
    fn ensure_display_mesh_current(&mut self) {
        if !self.mesh_stale {
            return;
        }
        // Will the brick raymarch draw this frame? Mirrors `brick_raymarch_engaged`: a field is
        // installed, not debug-face mode, and no loaded VS material. If so the (stale) mesh stays
        // hidden — leave it stale, keep skipping the build.
        #[cfg(feature = "gpu")]
        let brick_engaged = self
            .brick_raymarch_renderer
            .as_ref()
            .is_some_and(|renderer| renderer.has_brick_field())
            && !self.panel_state.debug_face_orientation
            && self.loaded_material.is_none();
        #[cfg(not(feature = "gpu"))]
        let brick_engaged = false;
        if brick_engaged {
            return;
        }
        // The mesh is about to be the display but is stale — rebuild it wholesale from the
        // resident two-layer set (scene unchanged, so `build_covering_chunks` yields the same
        // set the last resolve produced; identical to the startup mesh build). Bump the
        // generation and drop any outstanding async so a superseded in-flight result is
        // discarded rather than swapped in over this fresh mesh.
        let density = self.panel_state.geometry.voxels_per_block;
        let chunks = voxel_worker::TwoLayerStore::enabled().build_covering_chunks(
            &self.panel_state.scene,
            density,
            0,
        );
        let recentre = self
            .panel_state
            .scene
            .recentre_voxels_for_resolve(density);
        let band = self.current_layer_band(self.region_dimensions[2]);
        self.geometry_generation.next_generation();
        self.geometry_async_outstanding = false;
        self.cuboid_mesh_renderer = CuboidMeshRenderer::new_from_two_layer_chunks_banded(
            &self.gpu.device,
            &self.gpu.queue,
            COLOR_TARGET_FORMAT,
            &chunks,
            self.region_dimensions,
            recentre,
            density,
            band,
        );
        self.mesh_stale = false;
        println!("mesh: rebuilt fallback (brick display disengaged — debug-face / material)");
    }

    /// Drain the background scan channel into a pending queue, then build a
    /// BOUNDED number of thumbnails per frame so a few-hundred-block scan never
    /// stalls a frame. All GPU work (thumbnail render, egui registration) happens
    /// here on the main thread; with the cap it is amortised across frames.
    fn poll_scan(&mut self) {
        // Cap the thumbnail GPU work per frame. The PNG decode already happens on
        // the scan worker; this only bounds the main-thread render+register so a
        // burst of groups arriving at once can't hitch the frame.
        const THUMBNAILS_PER_FRAME: usize = 8;

        // Move everything the worker has produced so far into the pending queue.
        if let Some(handle) = self.scan_handle.as_ref() {
            for message in handle.drain() {
                match message {
                    ScanMessage::Group { group, thumbnail_rgba } => {
                        self.pending_groups.push_back((group, thumbnail_rgba));
                    }
                    ScanMessage::Done { group_count, source_name } => {
                        self.scan_total = Some(group_count);
                        self.scan_source_name = source_name;
                        self.scan_handle = None;
                    }
                }
            }
        }

        // Build at most a few thumbnails this frame; the rest wait for later
        // frames (we keep redrawing each frame via `about_to_wait`).
        for _ in 0..THUMBNAILS_PER_FRAME {
            let Some((group, thumbnail_rgba)) = self.pending_groups.pop_front() else {
                break;
            };
            self.palette.add_group(
                &self.gpu.device,
                &self.gpu.queue,
                &self.thumbnail_renderer,
                &mut self.egui_bridge.renderer,
                group,
                &thumbnail_rgba,
            );
        }

        // Status line: still working while groups are arriving or queued; settle
        // to the final count once the worker is done AND the queue is drained.
        if self.scan_handle.is_none() && self.pending_groups.is_empty() {
            if let Some(total) = self.scan_total.take() {
                self.palette.status = match self.scan_source_name.take() {
                    Some(name) => format!("{total} blocks loaded — {name}"),
                    None => "No VS install found — use Connect folder".to_string(),
                };
            }
        } else {
            self.palette.status = format!("{} blocks loaded…", self.palette.tiles.len());
        }
    }

    /// Apply palette interactions from this frame's [`PanelResponse`] (M6):
    /// applying a block loads + binds its texture; "Connect folder…" opens the OS
    /// picker and starts a custom scan; selecting a procedural material clears the
    /// applied block.
    fn handle_palette_response(&mut self, response: &voxel_worker::PanelResponse) {
        if response.selected_procedural_material {
            self.loaded_material = None;
            self.panel_state.applied_block_label = None;
        }
        if let Some(tile_index) = response.clicked_palette_tile {
            if let Some(variant_path) = self.palette.pick_variant(tile_index) {
                self.apply_block_variant(&variant_path, tile_index);
            }
        }
        if response.clicked_connect_folder {
            if let Some(folder) = rfd::FileDialog::new().pick_folder() {
                // Reset the palette + any in-flight scan state, then start a fresh
                // scan of the picked folder.
                self.palette.tiles.clear();
                self.pending_groups.clear();
                self.scan_total = None;
                self.scan_source_name = None;
                self.palette.status = "Scanning folder…".to_string();
                // Re-point the M7 face resolver at the same folder.
                self.face_resolver = FaceResolver::custom_folder(folder.clone());
                self.scan_handle = Some(spawn_custom_folder_scan(folder));
            }
        }
        if response.clicked_export_vox {
            self.export_vox();
        }
    }

    /// Re-resolve the current geometry and write it to a user-chosen `.vox` file
    /// (M8). The default filename encodes the shape + voxel dims (e.g.
    /// `cylinder_80x16x80.vox`). The palette colour is the active material's
    /// representative colour (a loaded block's average, or the procedural one).
    fn export_vox(&mut self) {
        let density = self.panel_state.geometry.voxels_per_block;
        let shape = SdfShape::from_geometry(self.panel_state.geometry.clone());
        // ADR 0010 E4: the old `exceeds_voxel_cap` guard (the dense whole-region 6M
        // ceiling) is GONE on the export path — the streaming export never materialises
        // a dense interior, so an 800×800-revolve-class solid exports successfully. A
        // pathological per-CHUNK density is still bounded by the resolver itself.

        let representative = match &self.loaded_material {
            Some(loaded) => loaded.average_color,
            None => procedural_material_average_color(self.panel_state.material),
        };
        // ADR 0003 §3a: map each categorical `block_id` to its colour. The palette is
        // the three procedural materials' colours; the ACTIVE material's slot takes the
        // representative (a loaded VS block's average, when applied), so a single-active-
        // material scene exports byte-identically to the old single-colour `.vox`.
        let mut palette_colors = vox_export_procedural_palette();
        palette_colors[self.panel_state.material.material_id() as usize] = representative;

        let default_name = default_vox_filename(&shape, density);
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_name)
            .add_filter("MagicaVoxel", &["vox"])
            .save_file()
        else {
            return;
        };
        // ADR 0010 E4: build the `.vox` by STREAMING the cacheless two-layer evaluator
        // region-scoped — a coarse-solid block is a fast `d³` fill, a boundary block is
        // per-voxel — so no dense whole-region grid is materialised and the 6M cap
        // dissolves on the export path. Each covering chunk's voxels are bucketed
        // DIRECTLY into the `.vox` model set by the incremental `VoxExportBuilder` then
        // DROPPED — peak transient memory is O(one chunk + the output buffers), NEVER the
        // O(all-voxels) `Vec<Vec<Voxel>>` accumulate-then-convert intermediate the button
        // used to build (the owner's peak-memory law: no O(volume) accumulation on any
        // path). The model count/sizes are a pure function of the region dimensions, so
        // the builder pre-creates the model set from `placed_region_dimensions` up front —
        // the SAME value `stream_vox_occupancy` produces — and one streaming pass suffices.
        // The streamed export stays model-set-identical to the dense-path region export
        // (the E4 parity gate). ADR 0010 E5: the two-layer capability is always ON now (the
        // sole runtime path), so the stream always yields — the retired dense
        // `bound_region_occupied` fallback is gone. `stream_vox_occupancy` returns `Some`
        // even for a Part-only / empty scene (an empty but valid `.vox`).
        let two_layer = voxel_worker::TwoLayerStore::enabled();
        let region_dimensions = self.panel_state.scene.placed_region_dimensions(density);
        let mut builder = voxel_worker::VoxExportBuilder::new(region_dimensions, palette_colors);
        voxel_worker::stream_vox_occupancy(
            &two_layer,
            &self.panel_state.scene,
            density,
            |chunk_voxels| builder.ingest_chunk(&chunk_voxels),
        )
        .expect("the two-layer capability is enabled (ADR 0010 E5)");
        let export = builder.finish();
        match export.write(&path) {
            Ok(bytes) => println!(
                "wrote {} ({} voxels, {} model(s), {} bytes)",
                path.display(),
                export.voxel_count(),
                export.model_count(),
                bytes
            ),
            Err(error) => eprintln!("export .vox failed: {error}"),
        }
    }

    /// Resolve `variant_path`'s per-face textures (M7) and bind the 6-layer
    /// material. Uniform blocks resolve to the same PNG on all faces (the M6
    /// path); per-face blocks (e.g. a log: end-grain top, bark sides) bind each
    /// face's own PNG.
    fn apply_block_variant(&mut self, variant_path: &std::path::Path, tile_index: usize) {
        let Some(tile) = self.palette.tiles.get(tile_index) else {
            return;
        };
        let label = tile.label.clone();
        let faces = self.face_resolver.resolve(&tile.group, variant_path);
        self.loaded_material = Some(LoadedMaterial::from_faces(
            &self.gpu.device,
            &self.gpu.queue,
            self.cuboid_mesh_renderer.material_bind_group_layout(),
            self.cuboid_mesh_renderer.material_sampler(),
            &faces,
            label.clone(),
        ));
        self.panel_state.applied_block_label = Some(label);
    }

    /// Persist the current UI + camera + window state to the platform config
    /// (M8). Called on window close / loop exit. Never panics on failure.
    fn save_config(&self) {
        let window_size = [self.surface_config.width, self.surface_config.height];
        let config =
            AppConfig::capture(&self.panel_state, &self.app_core.camera, self.home_view, window_size);
        config.save();
    }

    /// #13: save the live camera orbit as the new Home view (the right-click
    /// "set current view as home" context-menu action; Step 3).
    fn set_home_to_current(&mut self) {
        self.home_view = HomeView::from_camera(&self.app_core.camera);
    }

    /// #13: begin an eased snap toward the saved Home view and set the home
    /// distance directly (the tween animates the orbit angles; distance is a
    /// non-orbit param so it is applied immediately, matching the face-snap
    /// path which never tweens distance). Wired to the Home button + context-menu
    /// Home item in Step 3; pure-ish here so the logic is testable.
    ///
    /// #13 Step 6.4: a USER-set home (`explicitly_set`) restores its saved distance
    /// verbatim. The DEFAULT home (never set by the user) instead FRAMES the model —
    /// the canned default distance (10) zooms in far too close on a large model — so
    /// Home re-fits the auto-framed distance, matching the Fit button's distance.
    fn home_snap_tween(&mut self) -> SnapTween {
        let tween = self.home_view.snap_tween(&self.app_core.camera);
        self.app_core.camera.orbit_distance = if self.home_view.explicitly_set {
            self.home_view.distance
        } else {
            let region_dimensions = AppCore::region_dimensions_for(
                &self.panel_state.scene,
                self.panel_state.geometry.voxels_per_block,
            );
            self.app_core.camera.target = glam::Vec3::ZERO;
            OrbitCamera::auto_framed_distance(region_dimensions)
        };
        tween
    }

    /// #13: frame the model (the "Fit to view" action). Recompute the auto-frame
    /// distance from the scene's region dimensions and recentre the target on the
    /// model centroid — the recentred composite always sits at the world origin
    /// (`resolve_region` centres it), so the centroid is `Vec3::ZERO`. No geometry
    /// rebuild: only the camera distance + target change. The distance math is the
    /// same `auto_framed_distance` covered by camera tests.
    fn fit_to_view(&mut self) {
        let region_dimensions = AppCore::region_dimensions_for(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.app_core.camera.target = glam::Vec3::ZERO;
        self.app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(region_dimensions);
    }

    /// Is the pixel `(x, y)` inside the view-cube viewport? Issue #25: the cube's
    /// corner is offset into the central 3D viewport rect (cached from the last
    /// frame), so the hit rect tracks the cube's actual on-screen position rather
    /// than the window's top-left.
    fn position_in_view_cube(&self, x: f64, y: f64) -> bool {
        let [viewport_x, viewport_y, _, _] = self.last_viewport_px;
        let corner_x = (viewport_x + VIEW_CUBE_VIEWPORT_MARGIN) as f64;
        let corner_y = (viewport_y + VIEW_CUBE_VIEWPORT_MARGIN) as f64;
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f64;
        x >= corner_x && x <= corner_x + size && y >= corner_y && y <= corner_y + size
    }

    /// The ViewCube's on-screen square in window pixels (#13 Step 3), so the chrome
    /// hit-math ([`classify_cube_point`]) shares the SAME rect as
    /// [`Self::position_in_view_cube`] and the renderer. Offset into the central 3D
    /// viewport rect (issue #25), matching `pick_view_cube_element`.
    fn cube_rect(&self) -> CubeRect {
        let [viewport_x, viewport_y, _, _] = self.last_viewport_px;
        CubeRect {
            x: (viewport_x + VIEW_CUBE_VIEWPORT_MARGIN) as f32,
            y: (viewport_y + VIEW_CUBE_VIEWPORT_MARGIN) as f32,
            size: VIEW_CUBE_VIEWPORT_PIXELS as f32,
        }
    }

    /// Execute a [`ChromeClickAction`] resolved from a chrome-zone left-click
    /// (#13 Step 3). The pure mapping lives in `chrome_zone_left_click_action`; this
    /// only carries out the side effects (start a tween, run Home/Fit). A roll-arrow
    /// click resolves to a roll `Snap` tween (#13 Step 5: the real roll DOF).
    fn run_chrome_action(&mut self, action: ChromeClickAction) {
        match action {
            ChromeClickAction::Snap(tween) => self.snap_tween = Some(tween),
            ChromeClickAction::Home => self.snap_tween = Some(self.home_snap_tween()),
            ChromeClickAction::Fit => self.fit_to_view(),
        }
    }

    /// Ray-cast a click inside the view-cube viewport against the cube and return
    /// the hit [`ViewCubeElement`] (face / edge / corner). NDC is computed within
    /// the cube's screen rect, then unprojected through the view-cube matrix; the
    /// entry face is found by a slab intersection, and the 3D hit point's in-plane
    /// coordinates pick one of the face's 9 hot zones (3×3 grid at the 1/3 and 2/3
    /// thresholds): centre → the face, an edge zone → this face + the neighbour the
    /// zone points toward, a corner zone → this face + both neighbours.
    fn pick_view_cube_element(&self, x: f64, y: f64) -> Option<ViewCubeElement> {
        // Issue #25: the cube's corner is offset into the central viewport rect.
        let [viewport_x, viewport_y, _, _] = self.last_viewport_px;
        let corner_x = (viewport_x + VIEW_CUBE_VIEWPORT_MARGIN) as f32;
        let corner_y = (viewport_y + VIEW_CUBE_VIEWPORT_MARGIN) as f32;
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f32;
        // NDC inside the cube rect (y up).
        let ndc_x = ((x as f32 - corner_x) / size) * 2.0 - 1.0;
        let ndc_y = -(((y as f32 - corner_y) / size) * 2.0 - 1.0);

        let view_projection = self.app_core.camera.view_cube_view_projection();
        let inverse = view_projection.inverse();
        let near = inverse * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far = inverse * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        let near = near.truncate() / near.w;
        let far = far.truncate() / far.w;
        let origin = near;
        let direction = (far - near).normalize_or_zero();
        if direction == glam::Vec3::ZERO {
            return None;
        }

        // Slab intersection against the cube [-HALF, HALF]^3; the entry face's
        // dominant axis gives the material index / CubeFace.
        const HALF: f32 = 0.7;
        let mut t_entry = f32::NEG_INFINITY;
        let mut entry_axis = 0usize;
        let mut entry_sign = 1.0f32;
        let mut t_exit = f32::INFINITY;
        for axis in 0..3 {
            let o = origin[axis];
            let d = direction[axis];
            if d.abs() < 1e-6 {
                if !(-HALF..=HALF).contains(&o) {
                    return None; // parallel and outside the slab
                }
                continue;
            }
            let mut t0 = (-HALF - o) / d;
            let mut t1 = (HALF - o) / d;
            let mut sign = -1.0; // entering the -HALF face
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
                sign = 1.0; // entering the +HALF face
            }
            if t0 > t_entry {
                t_entry = t0;
                entry_axis = axis;
                entry_sign = sign;
            }
            t_exit = t_exit.min(t1);
        }
        if t_entry > t_exit || t_exit < 0.0 {
            return None;
        }

        // Map (axis, sign) → material index (+X,-X,+Y,-Y,+Z,-Z) → CubeFace.
        let material_index = entry_axis * 2 + if entry_sign > 0.0 { 0 } else { 1 };
        let face = CubeFace::from_material_index(material_index)?;

        // 3D hit point on the entry face, in cube space.
        let hit = origin + direction * t_entry;

        // The two axes NOT equal to `entry_axis` are the face's in-plane axes.
        // For each, the signed coordinate selects a 3×3 zone column/row: outside
        // ±HALF/3 the zone points toward the neighbouring face whose normal is
        // ±that axis. The combined set of faces resolves the element.
        const ZONE_THRESHOLD: f32 = HALF / 3.0; // 1/3 of the half-extent.
        let mut neighbours: Vec<CubeFace> = Vec::with_capacity(2);
        for axis in 0..3 {
            if axis == entry_axis {
                continue;
            }
            let coordinate = hit[axis];
            if coordinate > ZONE_THRESHOLD {
                // Positive face along this axis (Z-up: +X→Right, +Y→Back, +Z→Top).
                neighbours.push(face_for_axis_sign(axis, true));
            } else if coordinate < -ZONE_THRESHOLD {
                neighbours.push(face_for_axis_sign(axis, false));
            }
        }

        Some(match neighbours.as_slice() {
            [] => ViewCubeElement::from_face(face),
            [a] => ViewCubeElement::from_edge(face, *a),
            [a, b] => ViewCubeElement::from_corner(face, *a, *b),
            _ => ViewCubeElement::from_face(face),
        })
    }

    fn render(&mut self) {
        profiling::scope!("render");
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            // Surface lost / outdated: reconfigure and skip this frame.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                return;
            }
            // Transient conditions: skip this frame, try again next redraw.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded => {
                return;
            }
            other => {
                eprintln!("surface acquisition failed: {other:?}");
                return;
            }
        };

        let target_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Issue #60: poll the geometry worker — swap in a finished (non-stale) wholesale
        // mesh rebuild before drawing so it shows this frame (stale-while-rebuilding).
        self.poll_geometry_worker();

        // M6: drain the background scan channel and turn any new groups into
        // palette tiles (GPU thumbnail + egui texture registration on this thread).
        self.poll_scan();

        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let pixels_per_point = self.egui_winit_state.egui_ctx().pixels_per_point();

        // Issue #12/#20 S6c-1: the layer scrubber's vertical extent comes from the
        // SCENE's region dimensions, not the assembled grid object — identical to
        // `self.region_dimensions[2]` for a chunkable scene. Z-up: layers are Z-slices,
        // so the track spans the Z dimension (index 2).
        let grid_z = AppCore::region_dimensions_for(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        )[2];
        let current_band = (self.panel_state.layer_range.lower, self.panel_state.layer_range.upper);
        if current_band != self.measured_band {
            // ADR 0010 E4: re-measure the diameter by STREAMING the cacheless two-layer
            // evaluator — a coarse-solid block contributes its `d`-long run ANALYTICALLY
            // (no per-voxel expansion), a boundary block is per-voxel. Returns the SAME
            // value as the retired dense region-scoped `widest_run_in_band` (the parity
            // gate) without assembling a dense grid. ADR 0010 E5: the capability is always
            // ON now, so the stream always yields — `unwrap_or(0)` covers the empty scene.
            let density = self.panel_state.geometry.voxels_per_block;
            let two_layer = voxel_worker::TwoLayerStore::enabled();
            self.measured_diameter = voxel_worker::streamed_widest_run_in_band(
                &two_layer,
                &self.panel_state.scene,
                density,
                current_band.0,
                current_band.1,
            )
            .unwrap_or(0);
            self.measured_band = current_band;
        }

        // Issue #29 S5: tell the panel where **+ Add Point** should drop a new Point —
        // the camera target, converted from the recentred render frame back to whole
        // world blocks (`(target_voxels + recentre) / density`), so a new Point lands
        // where the user is looking.
        {
            let density = self.panel_state.geometry.voxels_per_block.max(1) as i64;
            let recentre = self
                .panel_state
                .scene
                .recentre_voxels_for_resolve(self.panel_state.geometry.voxels_per_block);
            let target = self.app_core.camera.target;
            self.panel_state.point_add_position_blocks = [
                ((target.x.round() as i64) + recentre[0]).div_euclid(density),
                ((target.y.round() as i64) + recentre[1]).div_euclid(density),
                ((target.z.round() as i64) + recentre[2]).div_euclid(density),
            ];
        }

        let mut prepared = {
            profiling::scope!("egui_frame");
            run_egui_frame(
                &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &mut self.panel_state,
            grid_z,
            self.measured_diameter,
            &self.palette,
            raw_input,
            [self.surface_config.width, self.surface_config.height],
                pixels_per_point,
                &mut self.context_menu_open_at,
            )
        };

        // Issue #25: cache the central 3D viewport rect so the view-cube
        // hit-testing (run later, in mouse events) can offset the cube corner.
        self.last_viewport_px = prepared.viewport_px;

        // #13 Step 3: execute a context-menu selection (egui drew + closed the
        // menu; the ortho toggle already mutated `panel_state.projection_mode`).
        match prepared.cube_menu_request {
            Some(ViewCubeMenuRequest::Home) => {
                self.snap_tween = Some(self.home_snap_tween());
            }
            Some(ViewCubeMenuRequest::Fit) => self.fit_to_view(),
            Some(ViewCubeMenuRequest::SetHome) => self.set_home_to_current(),
            None => {}
        }

        // Camera UX change: right-click a node row → "Focus" frames that node. This
        // is the ONLY edit-tree action that moves the camera. Set the orbit target to
        // the node's recentred world centre and fit the distance to its AABB (same fit
        // math as Fit, scoped to the node). The orbit ANGLES are held (Focus moves the
        // pivot + distance only). A node with no resolvable extent is a no-op.
        if let Some(focus_id) = prepared.panel_response.focus_node {
            if let Some((pivot, extent)) = AppCore::gizmo_placement_for_id(
                &self.panel_state.scene,
                focus_id,
                self.panel_state.geometry.voxels_per_block,
            ) {
                let (target, distance) = OrbitCamera::focus_target_and_distance(
                    glam::Vec3::from_array(pivot),
                    extent,
                );
                self.app_core.camera.target = target;
                self.app_core.camera.orbit_distance = distance;
            }
        }

        // M6: react to palette interactions (apply a block, connect a folder,
        // revert to a procedural material).
        self.handle_palette_response(&prepared.panel_response);

        // Advance an in-progress view-cube snap tween (eased over ~380ms).
        let now = std::time::Instant::now();
        let delta_seconds = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        if let Some(tween) = self.snap_tween.as_mut() {
            if tween.advance(&mut self.app_core.camera, delta_seconds) {
                self.snap_tween = None;
            }
        }

        // Feed egui's platform output (cursor icon, clipboard, …) back to winit.
        self.egui_winit_state
            .handle_platform_output(&self.window, prepared.platform_output.clone());

        // ADR 0003 Phase C C4a: the panel no longer mutates the scene directly — it
        // DESCRIBES this frame's mutations as a `Vec<Intent>`. Apply each through the
        // single `AppCore::apply_intent` door (in order), merging the returned typed
        // `IntentEffect`s, then fold them into the loop's existing decisions:
        //   * `scene_changed`     → re-resolve the grid (the old `geometry_changed` /
        //                           `scene_changed` rebuild).
        //   * `selection_changed` → re-sync the inspector mirror (the gizmo + node
        //                           highlight are recomputed every frame below from
        //                           `scene.active`, so they already track selection —
        //                           a pure `SelectNode` must NOT force a re-resolve).
        //   * `points_changed`    → the Points overlay is rebuilt every frame anyway
        //                           (camera-relative), so no extra work is needed.
        // Camera UX change: edits NO LONGER auto-frame the camera. The camera orbits
        // a FIXED/floating target (the world origin by default) and never jumps when
        // the user adds/moves/deletes/edits nodes. The panel's `frame_after_apply`
        // hint is intentionally IGNORED here — only the EXPLICIT view controls move
        // the camera now (startup fit, the ViewCube Home/Fit buttons, and the
        // right-click "Focus" action below). Take the intents out of `prepared`
        // (leaving it otherwise intact for the `render_frame` call below).
        let intents = std::mem::take(&mut prepared.panel_response.intents);
        let mut merged_effect = voxel_worker::IntentEffect::none();
        for intent in intents {
            let effect = self
                .app_core
                .apply_intent(&mut self.panel_state.scene, intent);
            merged_effect = merged_effect.merged_with(effect);
        }
        if merged_effect.selection_changed || merged_effect.scene_changed {
            // Re-sync the inspector mirror to the active node. The OLD panel called
            // `sync_mirror_from_active` after EVERY structural action (add / group /
            // make-definition / add-instance / delete — each of which changes the
            // active node) AND on a row select; we reproduce that by syncing on a
            // `selection_changed` (a pure `SelectNode`) OR a `scene_changed` (a
            // structural edit may have moved the active selection to a freshly-added /
            // re-derived node). Syncing after an inspector `SetShape`/`SetDensity` is a
            // harmless no-op (the node now equals the buffer it was written from). The
            // transform gizmo + row highlight read `scene.active` live each frame, so a
            // pure `SelectNode` updates them WITHOUT a re-resolve (the efficiency win).
            self.panel_state.sync_mirror_from_active();
        }
        if merged_effect.scene_changed {
            // A structural / node-field / global-density edit re-resolves the grid.
            // Camera UX change: this NEVER auto-frames any more — `false` keeps the
            // camera target + distance fixed across every edit. Re-framing is now only
            // via explicit controls (Home/Fit/Focus) and the startup fit.
            self.rebuild_geometry();
        }
        // Brick-display perf follow-up to epic #64: a debug-face toggle or a loaded-material
        // change are PURE display flags (they never `scene_changed`, so no rebuild fires) that
        // can turn OFF brick engagement — making the SKIPPED fallback mesh the display. Rebuild
        // it here the frame it is next needed, so a stale/empty mesh is never drawn. A no-op
        // unless the mesh is stale AND about to be shown.
        self.ensure_display_mesh_current();

        // Projection is a display-only param: apply it to the camera each frame
        // (no rebuild).
        self.app_core.camera.projection_mode = self.panel_state.projection_mode;

        // Upload the per-frame uniforms before drawing: camera matrix, grid
        // half-extent + density (per-voxel slice + overlay), and the overlay
        // toggle. The grid dims are the current geometry's voxel-space size.
        // Issue #25: the camera aspect comes from the CENTRAL 3D viewport rect (the
        // window minus the side panel + bottom dock), not the whole window, so the
        // model is centred in the visible 3D area instead of partly hidden behind
        // the side panel. `prepared.viewport_px` = [x, y, w, h] in physical pixels.
        let [_, _, viewport_width, viewport_height] = prepared.viewport_px;
        let aspect_ratio = viewport_width as f32 / viewport_height.max(1) as f32;
        let geometry = self.panel_state.geometry.clone();
        // The grid dims come from the ACTUALLY resolved scene grid (the composited
        // region's extent), not the active node's geometry — with several nodes the
        // region is the per-axis max of their sizes (ADR 0001 step 2).
        let grid_dimensions = self.region_dimensions;
        let view_projection = self.app_core.view_projection(aspect_ratio, grid_dimensions);
        // Issue #12: translate the layer-range scrubber into the shader band. The
        // band is inclusive on both ends; the upper handle is a layer index, so a
        // single-layer band is `lower == upper`. A full range draws everything.
        // Z-up: layers are Z-slices, so the band is a Z-layer range (index 2). The band
        // is computed by the shared `current_layer_band` helper (issue #60 M2) so the async
        // worker builds the mesh at the SAME band the render path applies here.
        let layer_range = self.panel_state.layer_range;
        let band = self.current_layer_band(grid_dimensions[2]);
        // Part of #20: the cuboid mesh path is the sole voxel renderer. Upload its
        // per-frame uniforms (camera + per-material base colours + band clip). A
        // loaded VS block textures it per-face (its 6-layer D2Array is bound at DRAW
        // time in `render_frame`, selecting the loaded pipeline); `bound = None` then
        // just disables the procedural per-box modulation/atlas, which the loaded
        // pipeline ignores.
        let bound = match &self.loaded_material {
            Some(_) => None,
            None => Some(self.panel_state.material),
        };
        self.cuboid_mesh_renderer.update_uniforms(
            &self.gpu.device,
            &self.gpu.queue,
            view_projection,
            grid_dimensions,
            geometry.voxels_per_block,
            // Issue #29 S4: the on-face-grid MASTER (Display checkbox →
            // `scene.master_voxel_grid`). The shader ANDs it with each voxel's
            // per-object flag bit packed into `material_id`.
            self.panel_state.scene.master_voxel_grid,
            bound,
            band,
            self.panel_state.debug_face_orientation,
        );
        // ADR 0011 G1: the brick raymarch takes THIS frame's voxel-model draw when a
        // field is installed and no mesh-only display mode is active — debug-faces
        // and a loaded VS material are per-frame toggles that never rebuild geometry,
        // so the draw decision is per-frame (the field stays installed). Its uniforms
        // mirror the cuboid upload above (camera, viewport, band, overlay master,
        // bound material) so the two paths render pixel-comparable.
        let brick_raymarch_engaged = match &self.brick_raymarch_renderer {
            Some(renderer)
                if renderer.has_brick_field()
                    && !self.panel_state.debug_face_orientation
                    && self.loaded_material.is_none() =>
            {
                renderer.update_uniforms(
                    &self.gpu.queue,
                    view_projection,
                    prepared.viewport_px,
                    grid_dimensions,
                    band,
                    self.panel_state.scene.master_voxel_grid,
                    bound,
                );
                true
            }
            _ => false,
        };
        // Transform gizmo (issue #29 S2): it FOLLOWS the selected node. Size it to
        // the selected node's own extent and bake its recentred pivot into the
        // camera matrix. `None` (nothing selected, or selection has no extent) hides
        // it — visibility is selection-driven, no longer a Display toggle.
        let gizmo_placement = AppCore::gizmo_placement(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        if let Some((pivot, extent)) = gizmo_placement {
            let extent_dims = [
                extent[0].round().max(0.0) as u32,
                extent[1].round().max(0.0) as u32,
                extent[2].round().max(0.0) as u32,
            ];
            self.transform_gizmo_renderer
                .rebuild(&self.gpu.device, &self.gpu.queue, extent_dims);
            self.transform_gizmo_renderer.update_uniforms(
                &self.gpu.queue,
                view_projection,
                glam::Vec3::from_array(pivot),
            );
        }
        // Per-object block lattice + floor grid (issue #29 S3): rebuild this frame's
        // line batch from the scene — for every node whose grids are enabled (the
        // scene master ANDed with the node's own toggle), its enclosing-block lattice
        // / base-plane floor lines. Empty when no node enables a grid (the new
        // default — per-object grids are OFF until the user turns them on).
        self.scene_grid_renderer.rebuild_from_scene(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.scene_grid_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        // World reference AXES (issue #29 S5): rebuild the visible Points' axis lines.
        self.points_renderer.rebuild_from_scene(
            &self.gpu.device,
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.points_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        // Analytic infinite reference grid (issue #29 Points fast-follow): rebuild the
        // visible Points' enabled PLANES with the camera matrices (recentred frame) so
        // the fullscreen ray-plane shader intersects each pixel's ray with the plane —
        // the grid extends to the horizon with no finite edge, fading with distance.
        self.infinite_grid_renderer.rebuild_from_scene(
            &self.gpu.queue,
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            view_projection,
            self.app_core.camera.eye().to_array(),
        );
        self.view_cube_renderer
            .update_uniforms(&self.gpu.queue, self.app_core.camera.view_cube_view_projection());

        // Issue #12: onion-skin volumetric fog. Active only when onion skin is on
        // and not in debug-face mode. Upload the camera + band world-Z ranges (Z-up)
        // so the
        // fullscreen raymarch of the occupancy grid hazes the layers around the band
        // (the grid itself is uploaded on geometry rebuild, not per frame).
        let onion_active = layer_range.onion_skin && !self.panel_state.debug_face_orientation;
        // Issue #59: the fog draws only when onion is active AND the band is NOT full-range
        // (a full-range band ghosts nothing). `false` for a full-range band even with onion
        // on — behaviour-identical to today, where such a frame draws no visible haze.
        let mut fog_should_draw = false;
        if onion_active {
            let fog_density = self.panel_state.geometry.voxels_per_block;
            let grid_z = grid_dimensions[2];
            // Issue #59: the band-slab the fog needs THIS frame (`None` = full-range → no
            // fog), and its covering chunk-Z rows (the reuse identity).
            let needed_slab = Self::fog_z_slab_for(layer_range, grid_z);
            let needed_chunk_z_range =
                Self::fog_covering_chunk_z_range(layer_range, grid_z, fog_density);
            match needed_slab {
                Some(slab) => {
                    // Band-aware rebuild (issue #58 + #59): rebuild the occupancy when it went
                    // stale (geometry changed / onion just toggled on — `fog_occupancy_dirty`)
                    // OR when the needed covering chunk-Z rows differ from the built ones (a
                    // scrub that moved the slab beyond the built region — the whole grid is no
                    // longer covered, so the old atlas is missing the newly-needed chunks).
                    // Compared at CHUNK granularity: a band move within the same chunk-Z rows
                    // reuses the atlas (the band clip is applied per-frame via
                    // `onion_fog_params`), so a scrub-drag doesn't rebuild every voxel step.
                    let slab_moved = self.fog_built_chunk_z_range != needed_chunk_z_range;
                    if self.fog_occupancy_dirty || slab_moved {
                        Self::build_fog_occupancy(
                            #[cfg(feature = "gpu")]
                            &self.gpu_resolver,
                            &mut self.onion_fog_renderer,
                            &self.gpu,
                            &self.panel_state.scene,
                            self.fog_mode,
                            self.region_dimensions,
                            self.recentre_voxels,
                            fog_density,
                            Some(slab),
                            self.fog_brick_field.as_ref(),
                        );
                        self.fog_occupancy_dirty = false;
                        self.fog_built_chunk_z_range = needed_chunk_z_range;
                    }
                    self.onion_fog_renderer.update(
                        &self.gpu.queue,
                        AppCore::onion_fog_params(view_projection, grid_dimensions, layer_range),
                    );
                    fog_should_draw = true;
                }
                None => {
                    // Full-range band (issue #59): nothing outside to ghost. Do NOT build or
                    // draw fog. Leave the occupancy marked stale so the next non-full band
                    // rebuilds (its covering rows also differ, forcing a rebuild regardless).
                    self.fog_occupancy_dirty = true;
                    self.fog_built_chunk_z_range = None;
                }
            }
        }

        let overlays = FrameOverlays {
            gizmo: gizmo_placement
                .is_some()
                .then_some(&self.transform_gizmo_renderer),
            view_cube: if self.panel_state.show_view_cube {
                Some(&self.view_cube_renderer)
            } else {
                None
            },
            // #13 Step 4: live hover — the chrome zone under the cursor (computed
            // cheaply in `CursorMoved`) so the hovered rotate/roll arrow brightens.
            // `None` when nothing's hovered or while orbiting/dragging.
            cube_hovered_zone: self.hovered_cube_zone,
            // #13 Step 6 follow-up: the four rotate arrows are a standing affordance
            // whenever the view is constrained to a face (not hover-gated), with the
            // hovered one brightened. Off-face views show none.
            cube_rotate_arrows_visible: self.app_core.camera.is_face_constrained(),
            scene_grid: Some(&self.scene_grid_renderer),
            // Issue #29 S5: the windowed app always shows the Points (the Origin's
            // ground+axes are on by default); the batch self-gates on hidden/off.
            points: Some(&self.points_renderer),
            // Issue #29 Points fast-follow: the analytic infinite grid (Points' planes);
            // self-gates on no enabled plane.
            infinite_grid: Some(&self.infinite_grid_renderer),
            // Issue #59: draw fog only when there is a non-full-range band to ghost around
            // (a full-range band with onion on draws nothing — `fog_should_draw` is false).
            onion_fog: if fog_should_draw {
                Some(&self.onion_fog_renderer)
            } else {
                None
            },
            cuboid_mesh: &self.cuboid_mesh_renderer,
            // ADR 0011 G1: when engaged (field installed, no mesh-only mode), the
            // brick raymarch replaces the cuboid-mesh DRAW for this frame; the mesh
            // stays built as the fallback + A/B reference (ADR 0011 Decision 6).
            brick_raymarch: if brick_raymarch_engaged {
                self.brick_raymarch_renderer.as_ref()
            } else {
                None
            },
            target_width: self.surface_config.width,
            target_height: self.surface_config.height,
        };

        // M6: an applied VS block overrides the procedural material selection.
        let material = match &self.loaded_material {
            Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
            None => MaterialSource::Procedural(self.panel_state.material),
        };

        {
            profiling::scope!("render_submit");
            render_frame(
                &mut self.egui_bridge,
                &self.gpu.device,
                &self.gpu.queue,
                &target_view,
                &self.msaa_color_view,
                &self.depth_view,
                material,
                &overlays,
                &prepared,
            );

            surface_texture.present();
        }

        // One frame mark per rendered frame (not per event). No-op unless a
        // profiling backend is enabled; under `--features tracy` this delimits the
        // frame on the Tracy timeline.
        profiling::finish_frame!();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(WindowedState::new(event_loop));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };

        // Let egui consume the event first; if it did, don't also use it to
        // drive the camera (so dragging on the panel doesn't orbit the scene).
        let response = state
            .egui_winit_state
            .on_window_event(&state.window, &event);
        let egui_consumed = response.consumed;

        match event {
            WindowEvent::CloseRequested => {
                // M8: persist UI + camera + window size before exiting.
                state.save_config();
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                state.resize(new_size.width, new_size.height);
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Left,
                ..
            } => {
                if button_state == ElementState::Pressed {
                    let position = state.last_cursor_position;
                    let in_cube = state.panel_state.show_view_cube
                        && position
                            .map(|(x, y)| state.position_in_view_cube(x, y))
                            .unwrap_or(false);
                    state.press_position = position;
                    state.press_in_view_cube = in_cube;
                    state.view_cube_drag_active = false;
                    // Pressing on the view cube does NOT start a scene-path orbit
                    // (`left_button_held`): a press on the cube either becomes a
                    // cube-drag orbit (handled in CursorMoved) or, if it stays put,
                    // snaps on release. So the scene orbit path is reserved for
                    // presses that started outside the cube and weren't on egui.
                    state.left_button_held = !egui_consumed && !in_cube;
                } else {
                    // Release: a press that started in the cube and DIDN'T become a
                    // drag (stayed within the threshold) selects the picked hot-zone
                    // element and snaps to it (prototype pointerup). A cube-drag has
                    // already orbited the camera, so it snaps nothing.
                    if state.press_in_view_cube && !state.view_cube_drag_active {
                        if let (Some((down_x, down_y)), Some((up_x, up_y))) =
                            (state.press_position, state.last_cursor_position)
                        {
                            let stationary = (up_x - down_x).abs()
                                < VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                                && (up_y - down_y).abs() < VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                            if stationary && state.position_in_view_cube(up_x, up_y) {
                                // #13 Step 3: classify the stationary release into a
                                // chrome zone (rotate / roll / Home / Fit /
                                // cube body). The body region delegates to the same
                                // raycast picker as before, so a body click still
                                // resolves to an Element snap; the gutters/badges map
                                // to their actions. A drag-orbit never reaches here
                                // (it sets `view_cube_drag_active`, gated above), so
                                // orbiting still wins over a click.
                                let rect = state.cube_rect();
                                let zone = classify_cube_point(
                                    rect,
                                    up_x as f32,
                                    up_y as f32,
                                    || state.pick_view_cube_element(up_x, up_y),
                                );
                                // #13 Step 6.6: a rotate-arrow click only acts when the
                                // view is face-constrained (the arrows are hidden
                                // otherwise, so a stray gutter click is a no-op).
                                let rotate_disabled = matches!(
                                    zone,
                                    Some(CubeChromeZone::RotateArrow(_))
                                ) && !state.app_core.camera.is_face_constrained();
                                if let (Some(zone), false) = (zone, rotate_disabled) {
                                    let action = chrome_zone_left_click_action(
                                        zone,
                                        &state.app_core.camera,
                                    );
                                    state.run_chrome_action(action);
                                }
                            }
                        }
                    }
                    state.left_button_held = false;
                    state.last_cursor_position = None;
                    state.press_in_view_cube = false;
                    state.view_cube_drag_active = false;
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Middle,
                ..
            } => {
                // Middle-drag pans the camera (explicit camera action). A press
                // that egui consumed (over the side panel / dock) doesn't grab the
                // scene, mirroring the left-orbit gate. The view cube doesn't take
                // middle clicks, so no cube gating is needed here.
                state.middle_button_held =
                    button_state == ElementState::Pressed && !egui_consumed;
            }
            WindowEvent::MouseInput {
                state: button_state,
                button: MouseButton::Right,
                ..
            } => {
                // #13 Step 3: a right-press inside the cube rect (not on egui) opens
                // the ViewCube context menu at the cursor. The menu itself is drawn
                // by egui in `run_egui_frame`; egui swallows its own clicks, so the
                // menu items never leak to the left-click snap path. Any other
                // right-press closes a menu that was open.
                if button_state == ElementState::Pressed && !egui_consumed {
                    let position = state.last_cursor_position;
                    let in_cube = state.panel_state.show_view_cube
                        && position
                            .map(|(x, y)| state.position_in_view_cube(x, y))
                            .unwrap_or(false);
                    state.context_menu_open_at = if in_cube {
                        position.map(|(x, y)| egui::pos2(x as f32, y as f32))
                    } else {
                        None
                    };
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);

                // A press that started on the view cube becomes an orbit drag once
                // it moves past the threshold. This routes the SAME delta into
                // `orbit_by_drag` as a scene drag (no double-application: the cube
                // press never sets `left_button_held`, so only one path fires).
                if state.press_in_view_cube && !state.view_cube_drag_active {
                    if let Some((down_x, down_y)) = state.press_position {
                        let moved = (current.0 - down_x).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS
                            || (current.1 - down_y).abs() >= VIEW_CUBE_DRAG_THRESHOLD_PIXELS;
                        if moved {
                            state.view_cube_drag_active = true;
                            // Promote to an orbit drag: cancel any in-progress snap.
                            state.snap_tween = None;
                        }
                    }
                }

                let orbiting = state.left_button_held || state.view_cube_drag_active;
                if orbiting {
                    if let Some((previous_x, previous_y)) = state.last_cursor_position {
                        let mut delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        // #13 Step 6.1: a cube drag GRABS the cube and turns it with
                        // the cursor, so the camera must orbit the OPPOSITE way round
                        // the model from a scene drag (dragging the cube's right edge
                        // leftward spins the model to show its right face). The scene
                        // drag keeps its existing sign; only the cube-drag path flips
                        // the horizontal component.
                        if state.view_cube_drag_active {
                            delta_x = -delta_x;
                        }
                        if delta_x != 0.0 || delta_y != 0.0 {
                            // A manual orbit cancels any in-progress snap tween.
                            state.snap_tween = None;
                            state.app_core.camera.orbit_by_drag(delta_x, delta_y);
                        }
                    }
                }

                // Middle-drag pans the target in the view plane (independent of the
                // orbit path, so the cursor can never both orbit and pan in one
                // move). Like orbit, a manual pan cancels any in-progress snap tween.
                if state.middle_button_held {
                    if let Some((previous_x, previous_y)) = state.last_cursor_position {
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        if delta_x != 0.0 || delta_y != 0.0 {
                            state.snap_tween = None;
                            // The 3D viewport height (cached each frame) makes the
                            // pan cursor-locked: a pixel of drag == a pixel of scene.
                            let viewport_height_px = state.last_viewport_px[3] as f32;
                            state
                                .app_core
                                .camera
                                .pan_by_drag(delta_x, delta_y, viewport_height_px);
                        }
                    }
                }
                state.last_cursor_position = Some(current);

                // #13 Step 4: live hover highlight for the chrome arrows. This runs
                // on every move, so keep it cheap: the chrome zones are pure
                // screen-rect tests, and we DELIBERATELY pass a `None` body picker so
                // the expensive cube raycast never fires for hover — a body-region
                // hover resolves to `None` (the body doesn't highlight anyway). Hover
                // stays `None` while orbiting/dragging, when egui ate the move, when
                // the cube is hidden, or when the cursor is outside the cube rect, so
                // it never interferes with drag-orbit, the click dispatch, or the
                // scene input.
                state.hovered_cube_zone = if orbiting
                    || egui_consumed
                    || !state.panel_state.show_view_cube
                    || !state.position_in_view_cube(current.0, current.1)
                {
                    None
                } else {
                    match classify_cube_point(
                        state.cube_rect(),
                        current.0 as f32,
                        current.1 as f32,
                        || state.pick_view_cube_element(current.0, current.1),
                    ) {
                        // #13 Step 6.6: rotate arrows are a face-relative affordance —
                        // only offer them when the view is constrained to a face
                        // (Fusion behaviour). Off-face hovers over a rotate gutter
                        // don't light up.
                        Some(CubeChromeZone::RotateArrow(_))
                            if !state.app_core.camera.is_face_constrained() =>
                        {
                            None
                        }
                        // #13 Step 6.2: faces/edges/corners DO highlight on hover now
                        // (the body picker resolves the hovered element); arrows and
                        // badges highlight as before.
                        Some(zone) => Some(zone),
                        None => None,
                    }
                };
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let scroll_lines = match delta {
                    MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32,
                };
                state.app_core.camera.zoom_by_wheel(scroll_lines);
            }
            WindowEvent::RedrawRequested => {
                state.render();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }

    /// Loop is exiting (e.g. OS-initiated): persist config as a safety net in
    /// case the exit didn't go through `CloseRequested` (M8).
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.save_config();
        }
    }
}

fn main() {
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
