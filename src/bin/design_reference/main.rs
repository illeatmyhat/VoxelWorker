//! `design_reference` — the Signal design language, rendered by the app's own code.
//!
//! A printed style guide drifts the moment someone edits a constant. This binary cannot: it
//! paints the palette straight out of [`ui::signal_theme`]'s tokens, and the widgets through
//! the same [`egui::Style`] the application installs — if a token changes, this window changes
//! with it. The glyphs are [`ui::icons`]' data-driven `Mark`s, which is a SEPARATE authoring
//! from the live rail's own hand-painted glyphs in `signal_chrome.rs`; the two are meant to
//! agree with the same design mock but are not the same code, so a glyph illegible here is not
//! proof the live rail draws the same shape (check `signal_chrome.rs` too).
//!
//! It is deliberately a *separate* binary rather than a debug panel inside the app. The
//! reference wants a scene of its own — every glyph, every size, every state at once — which is
//! exactly what a working viewport must never be cluttered with.
//!
//! Run with `cargo run --bin design_reference`. It links no document, no evaluator and no 3D
//! pipeline: one winit window, one wgpu surface, one egui pass.

use std::sync::Arc;

use voxel_worker::gpu::GpuContext;
use voxel_worker::{EguiPaintBridge, COLOR_TARGET_FORMAT};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

mod sheet;

/// Everything the reference window owns once it exists.
struct Reference {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    bridge: EguiPaintBridge,
    egui_winit_state: egui_winit::State,
    sheet: sheet::Sheet,
}

impl Reference {
    /// Create the window, surface and egui bridge — the same construction order the app uses
    /// (surface before adapter, so the adapter is guaranteed presentable) and the same sRGB
    /// target format, so a glyph here is byte-identical to the same glyph in the app.
    fn new(event_loop: &ActiveEventLoop) -> Self {
        // `--width` / `--height` exist so the whole sheet can be captured in one frame: a
        // reference that only ever gets looked at through a 900 pt slot is a reference nobody
        // checks the bottom of. A window taller than the display is legal and PrintWindow
        // captures its full client area.
        let (width, height) = window_size_from_args();
        let attributes = Window::default_attributes()
            .with_title("VoxelWorker — Signal design reference")
            .with_inner_size(winit::dpi::LogicalSize::new(width, height));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create the reference window"),
        );

        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let gpu = pollster::block_on(GpuContext::new_with_instance(instance, Some(&surface)));

        let size = window.inner_size();
        let mut surface_config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by the adapter");
        surface_config.format = COLOR_TARGET_FORMAT;
        surface_config.usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        surface.configure(&gpu.device, &surface_config);

        let bridge = EguiPaintBridge::new(&gpu.device, COLOR_TARGET_FORMAT);
        let egui_winit_state = egui_winit::State::new(
            bridge.context.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        Self {
            window,
            surface,
            surface_config,
            gpu,
            bridge,
            egui_winit_state,
            sheet: sheet::Sheet::default(),
        }
    }

    /// Re-configure the surface after a resize.
    fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width.max(1);
        self.surface_config.height = height.max(1);
        self.surface
            .configure(&self.gpu.device, &self.surface_config);
    }

    /// Draw one frame: run the sheet's egui pass and present it.
    fn render(&mut self) {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            // Lost or outdated: reconfigure and let the next redraw draw.
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface
                    .configure(&self.gpu.device, &self.surface_config);
                return;
            }
            // Transient: skip this frame.
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
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

        // The app-wide Signal style, installed exactly as the application installs it — the
        // whole point of the reference is that it cannot diverge.
        self.bridge
            .context
            .all_styles_mut(ui::signal_theme::apply_app_style);

        let sheet = &mut self.sheet;
        let full_output = self.bridge.context.run_ui(raw_input, |ui| sheet.show(ui));

        self.egui_winit_state
            .handle_platform_output(&self.window, full_output.platform_output.clone());

        for (texture_id, image_delta) in &full_output.textures_delta.set {
            self.bridge.renderer.update_texture(
                &self.gpu.device,
                &self.gpu.queue,
                *texture_id,
                image_delta,
            );
        }
        let paint_jobs = self
            .bridge
            .context
            .tessellate(full_output.shapes, pixels_per_point);
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point,
        };

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("design-reference frame"),
            });
        let upload_commands = self.bridge.renderer.update_buffers(
            &self.gpu.device,
            &self.gpu.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("design-reference egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // The page ground, in linear space for the sRGB target.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0018,
                            g: 0.0022,
                            b: 0.0028,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.bridge.renderer.render(
                &mut pass.forget_lifetime(),
                &paint_jobs,
                &screen_descriptor,
            );
        }
        self.gpu
            .queue
            .submit(upload_commands.into_iter().chain(Some(encoder.finish())));
        surface_texture.present();

        for texture_id in &full_output.textures_delta.free {
            self.bridge.renderer.free_texture(texture_id);
        }
    }
}

/// The winit pump. The window is created lazily on `resumed`, as winit 0.30 requires.
#[derive(Default)]
struct App {
    reference: Option<Reference>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.reference.is_none() {
            self.reference = Some(Reference::new(event_loop));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(reference) = self.reference.as_mut() else {
            return;
        };

        let response = reference
            .egui_winit_state
            .on_window_event(&reference.window, &event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                reference.resize(size.width, size.height);
                reference.window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                reference.render();
                // The sheet is static apart from hover, so redraw on demand rather than
                // spinning: egui asks for the next frame when it needs one.
                if response.repaint {
                    reference.window.request_redraw();
                }
            }
            _ => {
                if response.repaint {
                    reference.window.request_redraw();
                }
            }
        }
    }
}

/// Read `--width` / `--height` (in logical points) from the command line, defaulting to a
/// comfortable reading window.
fn window_size_from_args() -> (f64, f64) {
    let mut width = 1280.0;
    let mut height = 900.0;
    let args: Vec<String> = std::env::args().collect();
    for pair in args.windows(2) {
        let value = pair[1].parse::<f64>().ok();
        match (pair[0].as_str(), value) {
            ("--width", Some(v)) if v >= 480.0 => width = v,
            ("--height", Some(v)) if v >= 360.0 => height = v,
            _ => {}
        }
    }
    (width, height)
}

fn main() {
    let event_loop = EventLoop::new().expect("failed to create the event loop");
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop
        .run_app(&mut App::default())
        .expect("the reference window exited with an error");
}
