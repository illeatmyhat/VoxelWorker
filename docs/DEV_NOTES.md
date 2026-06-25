# DEV NOTES — verified API signatures & decisions

> Project / repo / crate name: **VoxelWorker** (crate `voxel_worker`). The design docs
> (HANDOFF/ARCHITECTURE/DATA) call the tool "Chisel Bench" — same thing, historical name.

This file exists so implementation subagents don't have to re-derive churn-prone API
signatures. Everything below was read directly from the extracted crate sources for the
**exact resolved versions** in `Cargo.toml` (Rust 1.92). If you change a dependency version,
re-verify the relevant section against the registry source.

## Resolved dependency set (single coherent graph, no duplicate wgpu/egui)

```
winit 0.30.13 · wgpu 29.0.3 · egui / egui-wgpu / egui-winit 0.34.3
glam 0.33.1 · bytemuck 1.25 · pollster 0.4 · image 0.25.10
rfd 0.17.2 (m6) · walkdir 2.5 (m6) · serde 1 + serde_json 1 (m7)
```

## Project conventions (from the user)

- **No terse/math identifiers.** Descriptive `snake_case`. `signed_distance_box`, not `sdBox`;
  `voxels_per_block`, not `D`; `semi_axis_x`, not `AX`; `block_local_coord`, not `iLocal`.
  The only allowed short names are loop counters `i, j, k` in the voxel sampling triple-loop.
- Sizes are in **whole blocks**; `density` (voxels_per_block) is fineness ONLY — never changes
  object size or texture scale. See ARCHITECTURE.md "Units & the density bug".
- `isolevel` is a CPU-only `const SURFACE_ISOLEVEL: f32 = 0.0;` — NOT a uniform, NOT a UI slider.
- Renderer must be **render-target-agnostic**: it draws into a `&wgpu::TextureView`, so the same
  code paints the windowed surface AND the headless capture texture. No winit knowledge in render.

## wgpu 29.0.3

```rust
// instance.request_adapter — ONE arg, returns Result (not Option)
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: surface_opt,   // Some(&surface) windowed, None headless
    })
    .await
    .expect("no suitable GPU adapter");

// adapter.request_device — ONE arg now (no trace_path second param). Returns Result<(Device,Queue)>.
let (device, queue) = adapter
    .request_device(&wgpu::DeviceDescriptor {
        label: Some("voxel-worker device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()   // experimental_features, memory_hints, trace: Trace::Off
    })
    .await
    .expect("request_device failed");
```

- `RenderPass` carries the encoder lifetime. egui's `render()` wants `&mut RenderPass<'static>`,
  so call **`render_pass.forget_lifetime()`** to convert before handing it to egui.
- Texture→buffer readback: `bytes_per_row` MUST be padded up to
  `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256). Pad the buffer, strip padding when writing PNG.

## egui-wgpu 0.34.3

```rust
let mut egui_renderer = egui_wgpu::Renderer::new(
    &device,
    surface_format,                 // output_color_format
    egui_wgpu::RendererOptions {
        msaa_samples: 1,            // egui feathers its own AA; 3D MSAA is separate (m4)
        depth_stencil_format: None,
        dithering: true,
        predictable_texture_filtering: false,
    },
);
// ScreenDescriptor { size_in_pixels: [u32;2], pixels_per_point: f32 }
// update_buffers(&device,&queue,&mut encoder,&paint_jobs,&screen_desc) -> Vec<CommandBuffer>
//   submit the returned command buffers (queue.submit) before/with the main encoder.
// render(&self, &mut RenderPass<'static>, &paint_jobs, &screen_desc)
// update_texture(&device,&queue, id, &image_delta) ; free_texture(&id)
```

Per-frame egui flow (shared by windowed + headless):
```rust
let full_output = egui_ctx.run(raw_input, |ctx| build_panel(ctx, &mut params));
for (id, delta) in &full_output.textures_delta.set {
    egui_renderer.update_texture(&device, &queue, *id, delta);
}
let paint_jobs = egui_ctx.tessellate(full_output.shapes, pixels_per_point);
let cmds = egui_renderer.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen_desc);
queue.submit(cmds);
// ... begin render pass (clear), then:
egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen_desc);
// after submit:
for id in &full_output.textures_delta.free { egui_renderer.free_texture(id); }
```

## egui-winit 0.34.3 (windowed input only — NOT used in headless)

```rust
let mut state = egui_winit::State::new(
    egui::Context::default(),
    egui::ViewportId::ROOT,
    &window,                 // &dyn HasDisplayHandle
    Some(window.scale_factor() as f32),  // native_pixels_per_point
    None,                    // theme
    None,                    // max_texture_side
);
let response = state.on_window_event(&window, &event);   // -> EventResponse {consumed, repaint}
let raw_input = state.take_egui_input(&window);
state.handle_platform_output(&window, full_output.platform_output);
```

## Headless capture (the no-window verification path)

- Build instance + adapter (`compatible_surface: None`) + device — **no surface, no window**.
- Offscreen color texture: format `Rgba8UnormSrgb` (match the windowed surface so screenshots
  look identical), usage `RENDER_ATTACHMENT | COPY_SRC`. Depth texture as needed (m2+).
- Drive egui WITHOUT winit: build `egui::RawInput` manually —
  `RawInput { screen_rect: Some(Rect::from_min_size(pos2(0,0), vec2(w,h))), ..Default::default() }`.
  Inject `egui::Event::PointerButton{..}` only if a test needs to exercise a click.
- After render: `copy_texture_to_buffer` (padded row), `buffer.slice(..).map_async(Read, cb)`,
  `device.poll(wgpu::PollType::wait_indefinitely()).unwrap()` (wgpu 29 poll returns Result and
  takes `PollType`, NOT the old `Maintain::Wait`), read mapped range, strip row padding,
  `image::save_buffer`.
- Windowed surface lifetime: store the window as `std::sync::Arc<winit::window::Window>` and call
  `instance.create_surface(window.clone())` so the surface is `Surface<'static>` (no borrow fight).
- CLI (`bin/shot`): accept a scene spec (shape, size x/y/z, density, camera theta/phi/dist,
  toggles, out-path) so a batch of viewpoints can be scripted. m1: just clear color + panel.

## winit 0.30.13 (ApplicationHandler model)

```rust
impl winit::application::ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // create window + surface + gpu + egui here on first resume
        let window = event_loop.create_window(Window::default_attributes()
            .with_title("Chisel Bench")).unwrap();
        // ...
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // feed egui_winit state.on_window_event; handle Resized / RedrawRequested / CloseRequested
    }
    fn about_to_wait(&mut self, _: &ActiveEventLoop) { self.window.request_redraw(); }
}
// main: let el = EventLoop::new()?; el.run_app(&mut App::default())?;
```

## Verification protocol for each milestone

1. `cargo build` (both bins) must succeed with no errors. Warnings: fix or justify.
2. `cargo run --bin shot -- <args> --out shots/mN.png` produces a PNG.
3. Compare against the prototype behavior described in ARCHITECTURE.md / DATA.md.
4. Report: files touched, build status, screenshot path, deviations from spec (+why), blockers,
   what the next milestone needs. Keep the report concise — the orchestrator reads the PNG.

- **Face winding/culling check:** `shot --debug-faces` (or the "Debug: face orientation" Display
  toggle) is the standard way to verify cube face winding/culling — it colours each fragment by its
  outward normal (+X red/−X cyan, +Y green/−Y magenta, +Z blue/−Z yellow) with the cube pipeline's
  culling OFF and flags any back-facing fragment with black-on-white stripes. A correct cube from the
  default 3/4 view shows red/green/blue with no marker; cyan/magenta/yellow or stripes means inverted
  winding.
