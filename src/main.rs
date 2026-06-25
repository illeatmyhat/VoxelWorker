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

use voxel_worker::{
    create_depth_view, create_msaa_color_view, render_frame, run_egui_frame, CubeFace,
    EguiPaintBridge, FrameOverlays, GizmoRenderer, GpuContext, OrbitCamera, PanelState, SdfShape,
    SliceImage, SnapTween, ViewCubeRenderer, VoxelGrid, VoxelProducer, VoxelRenderer,
    COLOR_TARGET_FORMAT, VIEW_CUBE_VIEWPORT_PIXELS,
};

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
    view_cube_renderer: ViewCubeRenderer,
    /// The current mid-Y 2D slice image, rebuilt whenever the grid rebuilds.
    slice_image: SliceImage,
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
}

#[derive(Default)]
struct App {
    state: Option<WindowedState>,
}

impl WindowedState {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("VoxelWorker"))
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

        // Resolve the panel's default geometry into the grid, then build the
        // renderer's instance buffer FROM the grid (REPRESENTATION.md seam). The
        // view cube is ON by default (prototype `showCube: true`).
        let panel_state = PanelState::with_view_cube_default();
        let shape = SdfShape::from_geometry(panel_state.geometry);
        let mut grid = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut grid);
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
        let view_cube_renderer =
            ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let slice_image = grid.build_slice_image(shape.voxels_per_block);

        let camera = OrbitCamera {
            orbit_distance: OrbitCamera::auto_framed_distance(grid.dimensions),
            projection_mode: panel_state.projection_mode,
            ..OrbitCamera::default()
        };

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
            view_cube_renderer,
            slice_image,
            depth_view,
            msaa_color_view,
            camera,
            snap_tween: None,
            last_frame_time: std::time::Instant::now(),
            left_button_held: false,
            last_cursor_position: None,
            press_position: None,
            press_in_view_cube: false,
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
        let shape = SdfShape::from_geometry(self.panel_state.geometry);

        if shape.exceeds_voxel_cap() {
            self.panel_state.voxel_cap_warning_millions =
                Some(shape.grid_voxel_count() as f32 / 1_000_000.0);
            return;
        }
        self.panel_state.voxel_cap_warning_millions = None;

        let mut grid = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut grid);
        self.voxel_renderer
            .rebuild_instances(&self.gpu.device, &self.gpu.queue, &grid);
        // Keep the gizmo sized to the grid and rebuild the 2D slice (cheap).
        self.gizmo_renderer
            .rebuild(&self.gpu.device, &self.gpu.queue, grid.dimensions);
        self.slice_image = grid.build_slice_image(shape.voxels_per_block);

        if auto_frame {
            self.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid.dimensions);
        }
    }

    /// Is the pixel `(x, y)` inside the top-left view-cube viewport?
    fn position_in_view_cube(&self, x: f64, y: f64) -> bool {
        let margin = VIEW_CUBE_VIEWPORT_MARGIN as f64;
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f64;
        x >= margin && x <= margin + size && y >= margin && y <= margin + size
    }

    /// Ray-cast a click inside the view-cube viewport against the cube and return
    /// the hit [`CubeFace`]. NDC is computed within the cube's screen rect, then
    /// unprojected through the view-cube matrix; the nearest of the six unit
    /// faces (|coord| ≈ 0.7) that the ray crosses is returned.
    fn pick_view_cube_face(&self, x: f64, y: f64) -> Option<CubeFace> {
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
        CubeFace::from_material_index(material_index)
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

        let raw_input = self.egui_winit_state.take_egui_input(&self.window);
        let pixels_per_point = self.egui_winit_state.egui_ctx().pixels_per_point();

        let prepared = run_egui_frame(
            &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &mut self.panel_state,
            &self.slice_image,
            raw_input,
            [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        );

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
        if panel_response.geometry_changed {
            self.rebuild_geometry(panel_response.size_or_density_changed);
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
        let grid_dimensions = [
            geometry.size_blocks[0] * geometry.voxels_per_block,
            geometry.size_blocks[1] * geometry.voxels_per_block,
            geometry.size_blocks[2] * geometry.voxels_per_block,
        ];
        let view_projection = self.camera.view_projection(aspect_ratio);
        self.voxel_renderer.update_uniforms(
            &self.gpu.queue,
            view_projection,
            grid_dimensions,
            geometry.voxels_per_block,
            self.panel_state.show_grid_overlay,
        );
        // M5 overlay uniforms: gizmo shares the main camera matrix; the view cube
        // uses its own orientation-mirroring matrix.
        self.gizmo_renderer
            .update_uniforms(&self.gpu.queue, view_projection);
        self.view_cube_renderer
            .update_uniforms(&self.gpu.queue, self.camera.view_cube_view_projection());

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
            target_width: self.surface_config.width,
            target_height: self.surface_config.height,
        };

        render_frame(
            &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &target_view,
            &self.msaa_color_view,
            &self.depth_view,
            &self.voxel_renderer,
            self.panel_state.material,
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
                    // Pressing on the view cube starts a snap (not an orbit drag).
                    state.left_button_held = !egui_consumed && !in_cube;
                } else {
                    // Release: a small, stationary click that started AND ended in
                    // the cube selects a face and snaps to it (prototype pointerup).
                    if state.press_in_view_cube {
                        if let (Some((down_x, down_y)), Some((up_x, up_y))) =
                            (state.press_position, state.last_cursor_position)
                        {
                            let stationary =
                                (up_x - down_x).abs() < 5.0 && (up_y - down_y).abs() < 5.0;
                            if stationary
                                && state.position_in_view_cube(up_x, up_y)
                            {
                                if let Some(face) = state.pick_view_cube_face(up_x, up_y) {
                                    state.snap_tween =
                                        Some(SnapTween::to_face(&state.camera, face));
                                }
                            }
                        }
                    }
                    state.left_button_held = false;
                    state.last_cursor_position = None;
                    state.press_in_view_cube = false;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if state.left_button_held {
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
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = App::default();
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
