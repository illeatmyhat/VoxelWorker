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
    create_depth_view, render_frame, run_egui_frame, EguiPaintBridge, GpuContext, OrbitCamera,
    PanelState, SdfShape, VoxelGrid, VoxelProducer, VoxelRenderer, COLOR_TARGET_FORMAT,
};

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
    depth_view: wgpu::TextureView,
    camera: OrbitCamera,
    /// Whether the left mouse button is held (orbit drag in progress).
    left_button_held: bool,
    /// Last cursor position, for computing drag deltas.
    last_cursor_position: Option<(f64, f64)>,
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
        // renderer's instance buffer FROM the grid (REPRESENTATION.md seam).
        let panel_state = PanelState::default();
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
        let voxel_renderer = VoxelRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, &grid);

        let camera = OrbitCamera {
            orbit_distance: OrbitCamera::auto_framed_distance(grid.dimensions),
            projection_mode: panel_state.projection_mode,
            ..OrbitCamera::default()
        };

        let depth_view = create_depth_view(&gpu.device, width, height);

        Self {
            window,
            surface,
            surface_config,
            gpu,
            egui_bridge,
            egui_winit_state,
            panel_state,
            voxel_renderer,
            depth_view,
            camera,
            left_button_held: false,
            last_cursor_position: None,
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
        // Recreate the depth texture to match the new target size.
        self.depth_view = create_depth_view(&self.gpu.device, width, height);
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

        if auto_frame {
            self.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid.dimensions);
        }
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
            raw_input,
            [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        );

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

        // Upload the current camera matrix before drawing.
        let aspect_ratio =
            self.surface_config.width as f32 / self.surface_config.height.max(1) as f32;
        self.voxel_renderer
            .update_camera(&self.gpu.queue, self.camera.view_projection(aspect_ratio));

        render_frame(
            &mut self.egui_bridge,
            &self.gpu.device,
            &self.gpu.queue,
            &target_view,
            &self.depth_view,
            &self.voxel_renderer,
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
                state.left_button_held =
                    button_state == ElementState::Pressed && !egui_consumed;
                if button_state == ElementState::Released {
                    state.last_cursor_position = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = (position.x, position.y);
                if state.left_button_held {
                    if let Some((previous_x, previous_y)) = state.last_cursor_position {
                        let delta_x = (current.0 - previous_x) as f32;
                        let delta_y = (current.1 - previous_y) as f32;
                        state.camera.orbit_by_drag(delta_x, delta_y);
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
