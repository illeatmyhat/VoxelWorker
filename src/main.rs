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
    create_depth_view, create_msaa_color_view, procedural_material_average_color, render_frame,
    run_egui_frame, AppConfig, CubeFace, EguiPaintBridge, FrameOverlays,
    GizmoRenderer,
    GpuContext, GridLatticeRenderer, LayerBand, MaterialSource, OnionFogParams,
    OnionFogRenderer, OrbitCamera, PanelState, Scene, SdfShape, SnapTween, ViewCubeElement,
    VoxExport, ViewCubeRenderer, VoxelGrid, VoxelRenderer, COLOR_TARGET_FORMAT,
    VIEW_CUBE_VIEWPORT_PIXELS,
};

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
    voxel_renderer: VoxelRenderer,
    gizmo_renderer: GizmoRenderer,
    /// Block lattice + fine floor grid (M8).
    grid_lattice_renderer: GridLatticeRenderer,
    view_cube_renderer: ViewCubeRenderer,
    /// Onion-skin volumetric fog (issue #12).
    onion_fog_renderer: OnionFogRenderer,
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
    /// Cached widest-run measurement + the band it was computed for, so we only
    /// re-measure when the band or grid actually changes.
    measured_diameter: u32,
    measured_band: (u32, u32),
    depth_view: wgpu::TextureView,
    /// 4× MSAA colour target for the 3D pass; resolved into the surface texture.
    msaa_color_view: wgpu::TextureView,
    camera: OrbitCamera,
    /// In-progress eased view-cube snap, if any.
    snap_tween: Option<SnapTween>,
    /// Timestamp of the previous frame, for advancing the snap tween.
    last_frame_time: std::time::Instant,
    /// Whether the left mouse button is held (orbit drag in progress).
    left_button_held: bool,
    /// Last cursor position, for computing drag deltas.
    last_cursor_position: Option<(f64, f64)>,
    /// Where the most recent left-press landed (for view-cube click detection).
    press_position: Option<(f64, f64)>,
    /// Whether the most recent left-press started inside the view-cube viewport.
    press_in_view_cube: bool,
    /// Whether a press that started on the view cube has moved past the drag
    /// threshold and is now orbiting the main camera (so the release snaps nothing).
    view_cube_drag_active: bool,
}

#[derive(Default)]
struct App {
    state: Option<WindowedState>,
}

/// The [`CubeFace`] whose outward normal lies along `axis` (0=X,1=Y,2=Z) with the
/// given sign (`true` = positive → Right/Top/Front, `false` = negative →
/// Left/Bottom/Back).
fn face_for_axis_sign(axis: usize, positive: bool) -> CubeFace {
    match (axis, positive) {
        (0, true) => CubeFace::Right,
        (0, false) => CubeFace::Left,
        (1, true) => CubeFace::Top,
        (1, false) => CubeFace::Bottom,
        (2, true) => CubeFace::Front,
        _ => CubeFace::Back,
    }
}

/// Build the onion-skin fog parameters (issue #12) from the camera, grid, and
/// layer-range scrubber. World-Y of layer `j` spans `[j - grid_y/2, j+1 -
/// grid_y/2]` (voxel centres at `j + 0.5 - grid_y/2`). The solid band is layers
/// `[lower, upper]`; the onion band extends `onion_depth` layers on each side.
fn onion_fog_params(
    view_projection: glam::Mat4,
    grid_dimensions: [u32; 3],
    layer_range: voxel_worker::LayerRange,
) -> OnionFogParams {
    let grid_y = grid_dimensions[1] as f32;
    let half_y = grid_y / 2.0;
    let depth = layer_range.onion_depth.clamp(1, 8) as f32;
    let lower = layer_range.lower as f32;
    let upper = layer_range.upper.min(grid_dimensions[1].saturating_sub(1)) as f32;
    OnionFogParams {
        inverse_view_projection: view_projection.inverse(),
        semi_axes: [
            grid_dimensions[0] as f32 / 2.0,
            grid_dimensions[1] as f32 / 2.0,
            grid_dimensions[2] as f32 / 2.0,
        ],
        // Onion band world-Y: `depth` layers below the band's bottom edge to
        // `depth` layers above its top edge.
        onion_y_min: (lower - depth) - half_y,
        onion_y_max: (upper + 1.0 + depth) - half_y,
        // Solid band world-Y (excluded from the fog).
        band_y_min: lower - half_y,
        band_y_max: (upper + 1.0) - half_y,
    }
}

/// Default `.vox` filename from the shape + voxel dims (e.g. `cylinder_80x16x80.vox`).
fn default_vox_filename(shape: &SdfShape) -> String {
    let [grid_x, grid_y, grid_z] = shape.grid_dimensions();
    let kind = format!("{:?}", shape.kind).to_lowercase();
    format!("{kind}_{grid_x}x{grid_y}x{grid_z}.vox")
}

/// Resolve the whole [`Scene`] into a fresh grid (ADR 0001 step 2). Every visible
/// node composites (union) into one region sized to the per-axis max of the
/// nodes' extents, at full resolution (`lod 0`). `voxels_per_block` is the global
/// app density (the inspector mirror's density). For a one-node scene this is
/// identical to the step-1 behaviour.
fn resolve_scene(scene: &Scene, voxels_per_block: u32) -> VoxelGrid {
    let region = scene.full_extent_blocks(voxels_per_block);
    scene.resolve_region(region, voxels_per_block, 0)
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
        let grid = resolve_scene(&panel_state.scene, panel_state.geometry.voxels_per_block);
        // Initialise the layer-range band to the full grid height (issue #12).
        let grid_y = grid.dimensions[1];
        panel_state
            .layer_range
            .rescale_to_grid_y(0, grid_y, shape.voxels_per_block);
        let measured_diameter = grid.widest_run_in_band(
            panel_state.layer_range.lower,
            panel_state.layer_range.upper,
        );
        let measured_band = (panel_state.layer_range.lower, panel_state.layer_range.upper);
        println!(
            "resolved {} voxels for {:?} {:?}@{}",
            grid.occupied_count(),
            shape.kind,
            shape.size_blocks,
            shape.voxels_per_block
        );
        let voxel_renderer =
            VoxelRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT, &grid);
        let gizmo_renderer =
            GizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, grid.dimensions);
        let grid_lattice_renderer = GridLatticeRenderer::new(
            &gpu.device,
            COLOR_TARGET_FORMAT,
            grid.dimensions,
            shape.voxels_per_block,
        );
        let view_cube_renderer =
            ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let mut onion_fog_renderer = OnionFogRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
        // Upload the resolved grid as the fog's 3D occupancy field (issue #12).
        onion_fog_renderer.upload_grid(&gpu.device, &gpu.queue, &grid);
        let thumbnail_renderer = ThumbnailRenderer::new(&gpu.device, &gpu.queue);

        // Kick off the VS auto-detect + scan on a background thread immediately;
        // results stream in over the next frames (no startup block).
        let palette = BlockPalette {
            status: "Scanning…".to_string(),
            ..BlockPalette::default()
        };
        let scan_handle = Some(spawn_auto_scan());

        let mut camera = OrbitCamera {
            orbit_distance: OrbitCamera::auto_framed_distance(grid.dimensions),
            projection_mode: panel_state.projection_mode,
            ..OrbitCamera::default()
        };
        // Restore the persisted camera orbit + projection if a config was loaded.
        if let Some(config) = &config {
            config.apply_camera(&mut camera);
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
            voxel_renderer,
            gizmo_renderer,
            grid_lattice_renderer,
            view_cube_renderer,
            onion_fog_renderer,
            thumbnail_renderer,
            palette,
            scan_handle,
            pending_groups: std::collections::VecDeque::new(),
            scan_total: None,
            scan_source_name: None,
            loaded_material: None,
            face_resolver: FaceResolver::auto(),
            grid,
            measured_diameter,
            measured_band,
            depth_view,
            msaa_color_view,
            camera,
            snap_tween: None,
            last_frame_time: std::time::Instant::now(),
            left_button_held: false,
            last_cursor_position: None,
            press_position: None,
            press_in_view_cube: false,
            view_cube_drag_active: false,
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
    /// When `auto_frame` is set (size/density change, NOT shape change) the
    /// camera distance is re-framed; shape switches never move the camera.
    fn rebuild_geometry(&mut self, auto_frame: bool) {
        let density = self.panel_state.geometry.voxels_per_block;
        // The cap is evaluated against the resolved region (the per-axis max of the
        // scene's node extents at the app density), not a single shape — multiple
        // nodes share one grid (ADR 0001 step 2).
        let region = self.panel_state.scene.full_extent_blocks(density);
        let region_voxel_count = region.size_blocks[0] as u64
            * region.size_blocks[1] as u64
            * region.size_blocks[2] as u64
            * density as u64
            * density as u64
            * density as u64;
        let shape = SdfShape::from_geometry(self.panel_state.geometry);

        if shape.exceeds_voxel_cap() || region_voxel_count > voxel_worker::voxel::MAX_GRID_VOXELS {
            self.panel_state.voxel_cap_warning_millions =
                Some(region_voxel_count.max(shape.grid_voxel_count()) as f32 / 1_000_000.0);
            return;
        }
        self.panel_state.voxel_cap_warning_millions = None;

        let previous_grid_y = self.grid.dimensions[1];
        let grid = resolve_scene(&self.panel_state.scene, density);
        self.voxel_renderer
            .rebuild_instances(&self.gpu.device, &self.gpu.queue, &grid);
        // Re-upload the fog's 3D occupancy field for the new grid (issue #12).
        self.onion_fog_renderer
            .upload_grid(&self.gpu.device, &self.gpu.queue, &grid);
        // Keep the gizmo sized to the grid.
        self.gizmo_renderer
            .rebuild(&self.gpu.device, &self.gpu.queue, grid.dimensions);
        // Keep the block lattice + floor grid sized to the grid/density.
        self.grid_lattice_renderer.rebuild(
            &self.gpu.device,
            &self.gpu.queue,
            grid.dimensions,
            shape.voxels_per_block,
        );

        // Issue #12: clamp/rescale the layer band to the new grid_y (re-snapping
        // to block multiples when snapping is on), then invalidate the diameter
        // cache so the readout re-measures against the new grid.
        self.panel_state.layer_range.rescale_to_grid_y(
            previous_grid_y,
            grid.dimensions[1],
            shape.voxels_per_block,
        );
        self.grid = grid;
        self.measured_band = (u32::MAX, u32::MAX); // force a re-measure next frame.

        if auto_frame {
            self.camera.orbit_distance = OrbitCamera::auto_framed_distance(self.grid.dimensions);
        }
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
        let shape = SdfShape::from_geometry(self.panel_state.geometry);
        if shape.exceeds_voxel_cap() {
            eprintln!("export .vox: grid exceeds the voxel cap; not exporting");
            return;
        }
        let grid = resolve_scene(&self.panel_state.scene, self.panel_state.geometry.voxels_per_block);

        let representative = match &self.loaded_material {
            Some(loaded) => loaded.average_color,
            None => procedural_material_average_color(self.panel_state.material),
        };

        let default_name = default_vox_filename(&shape);
        let Some(path) = rfd::FileDialog::new()
            .set_file_name(default_name)
            .add_filter("MagicaVoxel", &["vox"])
            .save_file()
        else {
            return;
        };
        let export = VoxExport::from_grid(&grid, representative);
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
            self.voxel_renderer.material_bind_group_layout(),
            self.voxel_renderer.material_sampler(),
            &faces,
            label.clone(),
        ));
        self.panel_state.applied_block_label = Some(label);
    }

    /// Persist the current UI + camera + window state to the platform config
    /// (M8). Called on window close / loop exit. Never panics on failure.
    fn save_config(&self) {
        let window_size = [self.surface_config.width, self.surface_config.height];
        let config = AppConfig::capture(&self.panel_state, &self.camera, window_size);
        config.save();
    }

    /// Is the pixel `(x, y)` inside the top-left view-cube viewport?
    fn position_in_view_cube(&self, x: f64, y: f64) -> bool {
        let margin = VIEW_CUBE_VIEWPORT_MARGIN as f64;
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f64;
        x >= margin && x <= margin + size && y >= margin && y <= margin + size
    }

    /// Ray-cast a click inside the view-cube viewport against the cube and return
    /// the hit [`ViewCubeElement`] (face / edge / corner). NDC is computed within
    /// the cube's screen rect, then unprojected through the view-cube matrix; the
    /// entry face is found by a slab intersection, and the 3D hit point's in-plane
    /// coordinates pick one of the face's 9 hot zones (3×3 grid at the 1/3 and 2/3
    /// thresholds): centre → the face, an edge zone → this face + the neighbour the
    /// zone points toward, a corner zone → this face + both neighbours.
    fn pick_view_cube_element(&self, x: f64, y: f64) -> Option<ViewCubeElement> {
        let margin = VIEW_CUBE_VIEWPORT_MARGIN as f32;
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f32;
        // NDC inside the cube rect (y up).
        let ndc_x = ((x as f32 - margin) / size) * 2.0 - 1.0;
        let ndc_y = -(((y as f32 - margin) / size) * 2.0 - 1.0);

        let view_projection = self.camera.view_cube_view_projection();
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
                // Positive face along this axis (+X→Right, +Y→Top, +Z→Front).
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

        // Issue #12: re-measure the band diameter only when the band changed (the
        // grid rebuild path resets `measured_band` to force a re-measure).
        let grid_y = self.grid.dimensions[1];
        let current_band = (self.panel_state.layer_range.lower, self.panel_state.layer_range.upper);
        if current_band != self.measured_band {
            self.measured_diameter =
                self.grid.widest_run_in_band(current_band.0, current_band.1);
            self.measured_band = current_band;
        }

        let prepared = run_egui_frame(
            &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &mut self.panel_state,
            grid_y,
            self.measured_diameter,
            &self.palette,
            raw_input,
            [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        );

        // M6: react to palette interactions (apply a block, connect a folder,
        // revert to a procedural material).
        self.handle_palette_response(&prepared.panel_response);

        // Advance an in-progress view-cube snap tween (eased over ~380ms).
        let now = std::time::Instant::now();
        let delta_seconds = (now - self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        if let Some(tween) = self.snap_tween.as_mut() {
            if tween.advance(&mut self.camera, delta_seconds) {
                self.snap_tween = None;
            }
        }

        // Feed egui's platform output (cursor icon, clipboard, …) back to winit.
        self.egui_winit_state
            .handle_platform_output(&self.window, prepared.platform_output.clone());

        // React to panel edits (M3): geometry changes rebuild the grid; size or
        // density changes also auto-frame. A shape switch rebuilds but does NOT
        // auto-frame (guard #1). Display/camera params never reach here.
        let panel_response = prepared.panel_response;
        // A geometry edit (inspector) or a scene change (add/delete/select/
        // visibility/seed) both re-resolve the composited scene. Add/delete can
        // change the extent, so a scene change auto-frames like a size change.
        if panel_response.geometry_changed || panel_response.scene_changed {
            let auto_frame =
                panel_response.size_or_density_changed || panel_response.scene_changed;
            self.rebuild_geometry(auto_frame);
        }

        // Projection is a display-only param: apply it to the camera each frame
        // (no rebuild).
        self.camera.projection_mode = self.panel_state.projection_mode;

        // Upload the per-frame uniforms before drawing: camera matrix, grid
        // half-extent + density (per-voxel slice + overlay), and the overlay
        // toggle. The grid dims are the current geometry's voxel-space size.
        let aspect_ratio =
            self.surface_config.width as f32 / self.surface_config.height.max(1) as f32;
        let geometry = self.panel_state.geometry;
        // The grid dims come from the ACTUALLY resolved scene grid (the composited
        // region's extent), not the active node's geometry — with several nodes the
        // region is the per-axis max of their sizes (ADR 0001 step 2).
        let grid_dimensions = self.grid.dimensions;
        let view_projection = self.camera.view_projection(aspect_ratio);
        // Issue #12: translate the layer-range scrubber into the shader band. The
        // band is inclusive on both ends; the upper handle is a layer index, so a
        // single-layer band is `lower == upper`. A full range draws everything.
        let layer_range = self.panel_state.layer_range;
        let band = if layer_range.is_full_range(grid_dimensions[1]) && !layer_range.onion_skin {
            LayerBand::FULL
        } else {
            LayerBand {
                band_min: layer_range.lower,
                // `upper` is the last visible layer index; clamp into the grid so a
                // full-range upper (== grid_y) still includes the top layer.
                band_max: layer_range.upper.min(grid_dimensions[1].saturating_sub(1)),
                onion_depth: if layer_range.onion_skin {
                    layer_range.onion_depth.clamp(1, 8)
                } else {
                    0
                },
            }
        };
        self.voxel_renderer.update_uniforms(
            &self.gpu.queue,
            view_projection,
            grid_dimensions,
            geometry.voxels_per_block,
            self.panel_state.show_grid_overlay,
            self.panel_state.debug_face_orientation,
            band,
        );
        // M5 overlay uniforms: gizmo shares the main camera matrix; the view cube
        // uses its own orientation-mirroring matrix.
        self.gizmo_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.grid_lattice_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.view_cube_renderer
            .update_uniforms(&self.gpu.queue, self.camera.view_cube_view_projection());

        // Issue #12: onion-skin volumetric fog. Active only when onion skin is on
        // and not in debug-face mode. Upload the camera + band world-Y ranges so the
        // fullscreen raymarch of the occupancy grid hazes the layers around the band
        // (the grid itself is uploaded on geometry rebuild, not per frame).
        let onion_active = layer_range.onion_skin && !self.panel_state.debug_face_orientation;
        if onion_active {
            self.onion_fog_renderer.update(
                &self.gpu.queue,
                onion_fog_params(view_projection, grid_dimensions, layer_range),
            );
        }

        let overlays = FrameOverlays {
            gizmo: if self.panel_state.show_origin_gizmo {
                Some(&self.gizmo_renderer)
            } else {
                None
            },
            view_cube: if self.panel_state.show_view_cube {
                Some(&self.view_cube_renderer)
            } else {
                None
            },
            grid_lattice: Some(&self.grid_lattice_renderer),
            show_lattice: self.panel_state.show_block_lattice,
            show_floor: self.panel_state.show_floor_grid,
            debug_face_mode: self.panel_state.debug_face_orientation,
            onion_fog: if onion_active {
                Some(&self.onion_fog_renderer)
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

        render_frame(
            &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &target_view,
            &self.msaa_color_view,
            &self.depth_view,
            &self.voxel_renderer,
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
                                if let Some(element) = state.pick_view_cube_element(up_x, up_y) {
                                    state.snap_tween =
                                        Some(SnapTween::to_element(&state.camera, element));
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
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        if delta_x != 0.0 || delta_y != 0.0 {
                            // A manual orbit cancels any in-progress snap tween.
                            state.snap_tween = None;
                            state.camera.orbit_by_drag(delta_x, delta_y);
                        }
                    }
                }
                state.last_cursor_position = Some(current);
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => {
                let scroll_lines = match delta {
                    MouseScrollDelta::LineDelta(_, vertical) => vertical,
                    MouseScrollDelta::PixelDelta(position) => position.y as f32,
                };
                state.camera.zoom_by_wheel(scroll_lines);
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
