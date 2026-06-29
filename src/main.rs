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
    GpuContext, InfiniteGridRenderer, LayerBand, MaterialSource, PointsRenderer,
    SceneGridRenderer,
    HomeView, OnionFogRenderer, OrbitCamera, PanelState, SdfShape, SnapTween, ViewCubeElement,
    ViewCubeMenuRequest,
    ViewCubeRenderer, VoxelGrid, COLOR_TARGET_FORMAT,
    VIEW_CUBE_VIEWPORT_PIXELS,
};
use voxel_worker::CuboidMeshRenderer;

/// Drag threshold (pixels) distinguishing a click (snap) from a drag (orbit) on
/// the view cube, and the general orbit-start threshold.
const VIEW_CUBE_DRAG_THRESHOLD_PIXELS: f64 = 5.0;

/// Margin from the top-left corner to the view-cube viewport (must match the
/// renderer's `VIEW_CUBE_VIEWPORT_MARGIN`).
const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;

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
    /// The cuboid mesh renderer — the sole voxel render path (part of #20; the legacy
    /// instanced mesher was removed). Rebuilt from the resolve cache's per-chunk
    /// accessor on every geometry change in `rebuild_geometry`.
    cuboid_mesh_renderer: CuboidMeshRenderer,
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
    /// The resolved voxel grid, kept so the layer-range diameter readout (issue
    /// #12) can re-measure the widest occupied run in the active band on demand.
    grid: VoxelGrid,
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
        let shape = SdfShape::from_geometry(panel_state.geometry);
        let grid = AppCore::resolve_scene(&panel_state.scene, panel_state.geometry.voxels_per_block);
        // Issue #20 S6c-1: the camera auto-frame, origin gizmo, block lattice, fine
        // floor grid and the layer scrubber are sized from the scene's region
        // dimensions DIRECTLY, not by reaching into the assembled grid object. For a
        // chunkable scene (the startup default + every Tool scene) this is
        // BYTE-IDENTICAL to `grid.dimensions` — the assembled grid is literally sized
        // to `placed_region_dimensions` (proven in
        // `scene::tests::placed_region_dimensions_equals_assembled_grid`). The
        // renderer / mesher / fog still consume the assembled `grid` (that's S6c
        // step 4). `region_dimensions_for` keeps the Part-only fallback exact.
        let region_dimensions = AppCore::region_dimensions_for(
            &panel_state.scene,
            panel_state.geometry.voxels_per_block,
            &grid,
        );
        // Initialise the layer-range band to the full grid height (issue #12). Z-up:
        // layers are Z-slices, so the track spans the Z dimension (index 2).
        let grid_z = region_dimensions[2];
        panel_state
            .layer_range
            .rescale_to_grid_z(0, grid_z, panel_state.geometry.voxels_per_block);
        // Issue #20 Step 2: the diameter / scrubber readout reads the region-scoped,
        // per-chunk `widest_run_in_band` (cross-seam stitched) rather than the
        // assembled grid's whole-grid method — returning the SAME value (parity-proven
        // in `chunk_cache::tests`) without depending on the monolithic grid object.
        // We build the resolve cache here so the startup readout uses the same path the
        // per-frame re-measure does; the chunks it resolves are cached for later reuse.
        let mut chunk_resolve_cache = voxel_worker::chunk_cache::ChunkResolveCache::new();
        let measured_band = (panel_state.layer_range.lower, panel_state.layer_range.upper);
        let measured_diameter = chunk_resolve_cache.widest_run_in_band(
            &panel_state.scene,
            panel_state.geometry.voxels_per_block,
            0,
            measured_band.0,
            measured_band.1,
        );
        println!(
            "resolved {} voxels for {:?} {:?}@{}",
            grid.occupied_count(),
            shape.kind,
            shape.size_blocks,
            panel_state.geometry.voxels_per_block
        );
        // The cuboid mesh renderer is the sole voxel render path (part of #20). The
        // whole-grid wrapper buckets `grid` into per-chunk sub-grids internally — the
        // same result the resolve cache's per-chunk accessor produces — so a startup
        // build from `grid` is byte-identical to the per-chunk rebuild path used on
        // every later geometry change in `rebuild_geometry`.
        let cuboid_mesh_renderer = CuboidMeshRenderer::new(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &grid,
            panel_state.geometry.voxels_per_block,
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
        // Upload the resolved grid as the fog's occupancy field. Per-chunk is the
        // DEFAULT (issue #28 S5b): one apron'd volume per resident chunk so fog still
        // renders at scale where a single whole-grid 3D texture would disable itself.
        let fog_mode = FogMode::PerChunk;
        Self::upload_fog_occupancy(
            &mut onion_fog_renderer,
            fog_mode,
            &gpu.device,
            &gpu.queue,
            &grid,
            panel_state.geometry.voxels_per_block,
        );
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
        // Restore the persisted camera orbit + projection if a config was loaded.
        if let Some(config) = &config {
            config.apply_camera(&mut camera);
        }
        // Restore the saved Home view (#13), or default to the camera defaults.
        let home_view = config
            .as_ref()
            .map(AppConfig::home_view)
            .unwrap_or_default();

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
            cuboid_mesh_renderer,
            transform_gizmo_renderer,
            scene_grid_renderer,
            points_renderer,
            infinite_grid_renderer,
            view_cube_renderer,
            onion_fog_renderer,
            fog_mode,
            thumbnail_renderer,
            palette,
            scan_handle,
            pending_groups: std::collections::VecDeque::new(),
            scan_total: None,
            scan_source_name: None,
            loaded_material: None,
            face_resolver: FaceResolver::auto(),
            grid,
            app_core: AppCore::new(chunk_resolve_cache, camera),
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
    fn upload_fog_occupancy(
        onion_fog_renderer: &mut OnionFogRenderer,
        fog_mode: FogMode,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) {
        match fog_mode {
            FogMode::WholeGrid => onion_fog_renderer.upload_grid(device, queue, grid),
            FogMode::PerChunk => {
                onion_fog_renderer.upload_grid_per_chunk(device, queue, grid, voxels_per_block)
            }
        }
    }

    /// Re-resolve the grid + GPU geometry for the current scene. Camera UX change:
    /// this NEVER moves the camera — edits keep the orbit target + distance fixed.
    /// Explicit framing (startup fit, Home/Fit, Focus) is handled by their own paths.
    fn rebuild_geometry(&mut self) {
        let density = self.panel_state.geometry.voxels_per_block;

        // Delegate the headless resolve (S2/S3 targeted invalidation + assemble) to
        // `AppCore::rebuild`, then consume its output here in the shell: build the
        // GPU cuboid mesh and upload the fog (the camera is NOT touched). A density whose
        // single-chunk voxel capacity exceeds the bound is rejected with the store
        // untouched, so we surface the cap warning and bail.
        let RebuildOutput {
            grid,
            region_dimensions,
            render_chunks,
            recentre_shift_voxels,
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

        // Read the OLD grid_z before reassigning `self.grid`, for the layer-band
        // rescale below (Z-up: layers are Z-slices, index 2).
        let previous_grid_z = self.grid.dimensions[2];
        // Part of #20: the cuboid mesh renderer is the sole voxel render path.
        // Rebuild it from the per-chunk accessor (`(absolute_chunk_coord,
        // &rebased_grid)` per covering chunk, 1-voxel apron). `render_chunks` holds
        // an IMMUTABLE borrow of the store, so it is consumed here and dropped BEFORE
        // any later `&mut AppCore` use.
        self.cuboid_mesh_renderer = CuboidMeshRenderer::new_from_chunks(
            &self.gpu.device,
            &self.gpu.queue,
            COLOR_TARGET_FORMAT,
            &render_chunks,
            grid.dimensions,
        );
        drop(render_chunks);

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
        // Re-upload the fog's occupancy field for the new grid, using the active fog
        // mode (per-chunk by default since #28 S5b).
        Self::upload_fog_occupancy(
            &mut self.onion_fog_renderer,
            self.fog_mode,
            &self.gpu.device,
            &self.gpu.queue,
            &grid,
            density,
        );
        // The transform gizmo (issue #29 S2) is sized + positioned from the SELECTED
        // node in the per-frame render path (it must track selection changes, which
        // don't trigger a geometry rebuild), not here. The per-object block lattice +
        // floor grid (issue #29 S3) is likewise (re)batched per frame from the
        // grid-enabled nodes — a per-node toggle needs no scene re-resolve.

        // Issue #12: clamp/rescale the layer band to the new grid_z (re-snapping
        // to block multiples when snapping is on), then invalidate the diameter
        // cache so the readout re-measures against the new grid. Z-up: index 2.
        self.panel_state.layer_range.rescale_to_grid_z(
            previous_grid_z,
            region_dimensions[2],
            density,
        );
        self.grid = grid;
        self.measured_band = (u32::MAX, u32::MAX); // force a re-measure next frame.
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
        let shape = SdfShape::from_geometry(self.panel_state.geometry);
        if shape.exceeds_voxel_cap(density) {
            eprintln!("export .vox: grid exceeds the voxel cap; not exporting");
            return;
        }

        let representative = match &self.loaded_material {
            Some(loaded) => loaded.average_color,
            None => procedural_material_average_color(self.panel_state.material),
        };

        let default_name = default_vox_filename(&shape, density);
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_name)
            .add_filter("MagicaVoxel", &["vox"])
            .save_file()
        else {
            return;
        };
        // Issue #20 Step 2: build the `.vox` from the region-scoped, per-chunk path
        // (`ChunkResolveCache::vox_export`) rather than the monolithic
        // `resolve_scene` + `VoxExport::from_grid`. Each chunk's voxels are rebased
        // to the composite recentre in i64 BEFORE the f32 downcast, so a far-offset
        // scene preserves the voxel-centre `.5` instead of losing it to f32 rounding
        // at large magnitude (the monolithic path's far-offset bug). For a near scene
        // the two paths are byte-identical (proven by the vox-export parity tests).
        let (region_dimensions, occupied) = self.app_core.store.bound_region_occupied(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            0,
        );
        let export =
            voxel_worker::VoxExport::from_region_voxels(region_dimensions, occupied, representative);
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
                &self.grid,
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
            &self.grid,
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

        // M6: drain the background scan channel and turn any new groups into
        // palette tiles (GPU thumbnail + egui texture registration on this thread).
        self.poll_scan();

        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let pixels_per_point = self.egui_winit_state.egui_ctx().pixels_per_point();

        // Issue #12/#20 S6c-1: the layer scrubber's vertical extent comes from the
        // SCENE's region dimensions, not the assembled grid object — identical to
        // `self.grid.dimensions[2]` for a chunkable scene. Z-up: layers are Z-slices,
        // so the track spans the Z dimension (index 2).
        let grid_z = AppCore::region_dimensions_for(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
            &self.grid,
        )[2];
        let current_band = (self.panel_state.layer_range.lower, self.panel_state.layer_range.upper);
        if current_band != self.measured_band {
            // Issue #20 Step 2: re-measure the diameter through the region-scoped,
            // per-chunk `widest_run_in_band` (cross-seam stitched), NOT the assembled
            // `self.grid`. Returns the SAME value as the whole-grid method
            // (parity-proven) while consuming only the per-chunk grids. The chunks are
            // already resident from the geometry rebuild, so this is cache HITs.
            self.measured_diameter = self.app_core.store.widest_run_in_band(
                &self.panel_state.scene,
                self.panel_state.geometry.voxels_per_block,
                0,
                current_band.0,
                current_band.1,
            );
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

        let mut prepared = run_egui_frame(
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
        );

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
        let geometry = self.panel_state.geometry;
        // The grid dims come from the ACTUALLY resolved scene grid (the composited
        // region's extent), not the active node's geometry — with several nodes the
        // region is the per-axis max of their sizes (ADR 0001 step 2).
        let grid_dimensions = self.grid.dimensions;
        let view_projection = self.app_core.view_projection(aspect_ratio, grid_dimensions);
        // Issue #12: translate the layer-range scrubber into the shader band. The
        // band is inclusive on both ends; the upper handle is a layer index, so a
        // single-layer band is `lower == upper`. A full range draws everything.
        let layer_range = self.panel_state.layer_range;
        // Z-up: layers are Z-slices, so the band is a Z-layer range (index 2).
        let band = if layer_range.is_full_range(grid_dimensions[2]) && !layer_range.onion_skin {
            LayerBand::FULL
        } else {
            LayerBand {
                band_min: layer_range.lower,
                // `upper` is the last visible layer index; clamp into the grid so a
                // full-range upper (== grid_z) still includes the top layer.
                band_max: layer_range.upper.min(grid_dimensions[2].saturating_sub(1)),
                onion_depth: if layer_range.onion_skin {
                    layer_range.onion_depth.clamp(1, 8)
                } else {
                    0
                },
            }
        };
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
        if onion_active {
            self.onion_fog_renderer.update(
                &self.gpu.queue,
                AppCore::onion_fog_params(view_projection, grid_dimensions, layer_range),
            );
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
            onion_fog: if onion_active {
                Some(&self.onion_fog_renderer)
            } else {
                None
            },
            cuboid_mesh: &self.cuboid_mesh_renderer,
            target_width: self.surface_config.width,
            target_height: self.surface_config.height,
        };

        // M6: an applied VS block overrides the procedural material selection.
        let material = match &self.loaded_material {
            Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
            None => MaterialSource::Procedural(self.panel_state.material),
        };

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
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
