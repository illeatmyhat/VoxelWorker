//! `shot` — the headless screenshot harness.
//!
//! Renders the SAME clear colour and the SAME egui panel as the windowed app
//! into an offscreen texture (no window, no surface), reads the pixels back, and
//! writes a PNG. This is the self-verification harness for every later
//! milestone: a milestone is "done" when its `shot` looks right.
//!
//! CLI:
//!   --out <path>     output PNG path        (default: shots/m1.png)
//!   --width <u32>    capture width          (default: 1280)
//!   --height <u32>   capture height         (default: 800)
//!   --shape <cylinder|tube|sphere|torus|box>   (default: cylinder)
//!   --size-x <u32> --size-y <u32> --size-z <u32>   size in blocks (default 5/1/5)
//!   --density <u32>  voxels per block       (default: 16)
//!   --wall <u32>     tube wall in blocks    (default: 1)
//!   --proj <perspective|ortho>              (default: perspective)
//!   --material <stone|wood|plain>           (default: stone)
//!   --grid                                  enable the voxel/block grid overlay
//!   --theta/--phi/--dist                    orbit overrides (auto-framed dist)

use std::path::PathBuf;

use voxel_worker::{
    create_depth_view, create_msaa_color_view, render_frame, run_egui_frame, EguiPaintBridge,
    GeometryParams, GpuContext, MaterialChoice, OrbitCamera, PanelState, ProjectionMode, SdfShape,
    ShapeKind, VoxelGrid, VoxelProducer, VoxelRenderer, COLOR_TARGET_FORMAT,
};

struct ShotOptions {
    output_path: PathBuf,
    width: u32,
    height: u32,
    /// Geometry the panel + producer both use (so the rendered panel reflects
    /// the requested shape/size/density/wall).
    geometry: GeometryParams,
    /// Camera projection.
    projection_mode: ProjectionMode,
    /// Procedural material (Stone/Wood/Plain).
    material: MaterialChoice,
    /// Whether the voxel/block grid overlay is drawn.
    show_grid_overlay: bool,
    /// Orbit azimuth (radians). Default 0.7.
    theta: f32,
    /// Orbit polar angle from +Y (radians). Default 1.05.
    phi: f32,
    /// Orbit distance. `None` = auto-frame from the grid.
    distance: Option<f32>,
}

impl Default for ShotOptions {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("shots/m1.png"),
            width: 1280,
            height: 800,
            geometry: GeometryParams::default(),
            projection_mode: ProjectionMode::Perspective,
            material: MaterialChoice::Stone,
            show_grid_overlay: false,
            theta: 0.7,
            phi: 1.05,
            distance: None,
        }
    }
}

/// Parse a `--shape` value into a [`ShapeKind`].
fn parse_shape(value: &str) -> ShapeKind {
    match value.to_ascii_lowercase().as_str() {
        "cylinder" => ShapeKind::Cylinder,
        "tube" => ShapeKind::Tube,
        "sphere" => ShapeKind::Sphere,
        "torus" => ShapeKind::Torus,
        "box" => ShapeKind::Box,
        other => panic!("--shape must be cylinder|tube|sphere|torus|box, got '{other}'"),
    }
}

/// Parse a `--material` value into a [`MaterialChoice`].
fn parse_material(value: &str) -> MaterialChoice {
    match value.to_ascii_lowercase().as_str() {
        "stone" => MaterialChoice::Stone,
        "wood" => MaterialChoice::Wood,
        "plain" => MaterialChoice::Plain,
        other => panic!("--material must be stone|wood|plain, got '{other}'"),
    }
}

/// Parse a `--proj` value into a [`ProjectionMode`].
fn parse_projection(value: &str) -> ProjectionMode {
    match value.to_ascii_lowercase().as_str() {
        "perspective" | "persp" => ProjectionMode::Perspective,
        "ortho" | "orthographic" => ProjectionMode::Orthographic,
        other => panic!("--proj must be perspective|ortho, got '{other}'"),
    }
}

fn parse_options() -> ShotOptions {
    let mut options = ShotOptions::default();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out" => {
                options.output_path = PathBuf::from(
                    args.next().expect("--out requires a path argument"),
                );
            }
            "--width" => {
                options.width = args
                    .next()
                    .expect("--width requires a value")
                    .parse()
                    .expect("--width must be a positive integer");
            }
            "--height" => {
                options.height = args
                    .next()
                    .expect("--height requires a value")
                    .parse()
                    .expect("--height must be a positive integer");
            }
            "--shape" => {
                options.geometry.shape =
                    parse_shape(&args.next().expect("--shape requires a value"));
            }
            "--size-x" => {
                options.geometry.size_blocks[0] = args
                    .next()
                    .expect("--size-x requires a value")
                    .parse()
                    .expect("--size-x must be a positive integer");
            }
            "--size-y" => {
                options.geometry.size_blocks[1] = args
                    .next()
                    .expect("--size-y requires a value")
                    .parse()
                    .expect("--size-y must be a positive integer");
            }
            "--size-z" => {
                options.geometry.size_blocks[2] = args
                    .next()
                    .expect("--size-z requires a value")
                    .parse()
                    .expect("--size-z must be a positive integer");
            }
            "--density" => {
                options.geometry.voxels_per_block = args
                    .next()
                    .expect("--density requires a value")
                    .parse()
                    .expect("--density must be a positive integer");
            }
            "--wall" => {
                options.geometry.wall_blocks = args
                    .next()
                    .expect("--wall requires a value")
                    .parse()
                    .expect("--wall must be a positive integer");
            }
            "--proj" => {
                options.projection_mode =
                    parse_projection(&args.next().expect("--proj requires a value"));
            }
            "--material" => {
                options.material =
                    parse_material(&args.next().expect("--material requires a value"));
            }
            "--grid" => {
                options.show_grid_overlay = true;
            }
            "--theta" => {
                options.theta = args
                    .next()
                    .expect("--theta requires a value")
                    .parse()
                    .expect("--theta must be a float (radians)");
            }
            "--phi" => {
                options.phi = args
                    .next()
                    .expect("--phi requires a value")
                    .parse()
                    .expect("--phi must be a float (radians)");
            }
            "--dist" => {
                options.distance = Some(
                    args.next()
                        .expect("--dist requires a value")
                        .parse()
                        .expect("--dist must be a float"),
                );
            }
            "--help" | "-h" => {
                println!(
                    "shot — headless VoxelWorker capture\n\
                     \n\
                     Usage: shot [--out <path>] [--width <u32>] [--height <u32>]\n\
                     \x20            [--shape <cylinder|tube|sphere|torus|box>]\n\
                     \x20            [--size-x <u32>] [--size-y <u32>] [--size-z <u32>]\n\
                     \x20            [--density <u32>] [--wall <u32>]\n\
                     \x20            [--proj <perspective|ortho>]\n\
                     \x20            [--material <stone|wood|plain>] [--grid]\n\
                     \x20            [--theta <f32>] [--phi <f32>] [--dist <f32>]\n\
                     Defaults: --out shots/m1.png --width 1280 --height 800\n\
                     \x20         --shape cylinder --size-x 5 --size-y 1 --size-z 5\n\
                     \x20         --density 16 --wall 1 --proj perspective\n\
                     \x20         --material stone (grid off)\n\
                     \x20         --theta 0.7 --phi 1.05 --dist <auto-framed>"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("warning: ignoring unknown argument '{other}'");
            }
        }
    }
    options
}

fn main() {
    let options = parse_options();
    pollster::block_on(run_capture(options));
}

async fn run_capture(options: ShotOptions) {
    assert!(options.width > 0 && options.height > 0, "capture size must be non-zero");

    // Fully headless: no surface, no window.
    let gpu = GpuContext::new(None).await;

    // Offscreen colour target. Same sRGB format as the windowed surface so the
    // screenshot is identical to the window; COPY_SRC so we can read it back.
    let capture_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless capture color"),
        size: wgpu::Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 4× MSAA depth + colour at the offscreen size. The 3D pass renders into the
    // multisampled colour texture and resolves into `capture_texture` (the single
    // -sample COPY_SRC target read back below).
    let depth_view = create_depth_view(&gpu.device, options.width, options.height);
    let msaa_color_view =
        create_msaa_color_view(&gpu.device, options.width, options.height, COLOR_TARGET_FORMAT);

    // Resolve the requested geometry into the grid, then build the renderer's
    // instance buffer FROM the grid (REPRESENTATION.md seam). The voxel cap
    // (ARCHITECTURE.md §7) guards against an enormous CLI request.
    let shape = SdfShape::from_geometry(options.geometry);
    let mut grid = VoxelGrid::new(shape.grid_dimensions());
    let mut panel_state = PanelState {
        geometry: options.geometry,
        projection_mode: options.projection_mode,
        material: options.material,
        show_grid_overlay: options.show_grid_overlay,
        ..PanelState::default()
    };
    if shape.exceeds_voxel_cap() {
        panel_state.voxel_cap_warning_millions =
            Some(shape.grid_voxel_count() as f32 / 1_000_000.0);
        eprintln!(
            "3D paused — {:.1}M voxels exceeds the cap; rendering empty grid",
            shape.grid_voxel_count() as f32 / 1_000_000.0
        );
    } else {
        shape.resolve(&mut grid);
    }
    println!(
        "resolved {} voxels for {:?} {:?}@{}",
        grid.occupied_count(),
        shape.kind,
        shape.size_blocks,
        shape.voxels_per_block
    );
    let voxel_renderer =
        VoxelRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT, &grid);

    // Build the orbit camera from the CLI flags (distance auto-framed if unset).
    let camera = OrbitCamera {
        target: glam::Vec3::ZERO,
        orbit_theta: options.theta,
        orbit_phi: options.phi,
        orbit_distance: options
            .distance
            .unwrap_or_else(|| OrbitCamera::auto_framed_distance(grid.dimensions)),
        projection_mode: options.projection_mode,
    };
    let aspect_ratio = options.width as f32 / options.height as f32;
    voxel_renderer.update_uniforms(
        &gpu.queue,
        camera.view_projection(aspect_ratio),
        shape.grid_dimensions(),
        options.geometry.voxels_per_block,
        options.show_grid_overlay,
    );

    // egui driven WITHOUT winit: build RawInput by hand.
    let mut egui_bridge = EguiPaintBridge::new(&gpu.device, COLOR_TARGET_FORMAT);
    let pixels_per_point = 1.0;
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(options.width as f32, options.height as f32),
        )),
        ..Default::default()
    };

    let prepared = run_egui_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &mut panel_state,
        raw_input,
        [options.width, options.height],
        pixels_per_point,
    );

    // Paint via the exact same render-target-agnostic core the window uses.
    render_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &capture_view,
        &msaa_color_view,
        &depth_view,
        &voxel_renderer,
        options.material,
        &prepared,
    );

    // --- Read back the texture into a PNG ---
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = options.width * bytes_per_pixel;
    let row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row =
        unpadded_bytes_per_row.div_ceil(row_alignment) * row_alignment;

    let readback_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless readback buffer"),
        size: (padded_bytes_per_row * options.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut copy_encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless copy encoder"),
    });
    copy_encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &capture_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(options.height),
            },
        },
        wgpu::Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(copy_encoder.finish()));

    // Map and wait.
    let buffer_slice = readback_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("failed to map readback buffer");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");

    // Strip the per-row padding into a tight RGBA8 image.
    let mut tightly_packed = Vec::with_capacity((unpadded_bytes_per_row * options.height) as usize);
    {
        let mapped = buffer_slice.get_mapped_range();
        for row_index in 0..options.height {
            let row_start = (row_index * padded_bytes_per_row) as usize;
            let row_end = row_start + unpadded_bytes_per_row as usize;
            tightly_packed.extend_from_slice(&mapped[row_start..row_end]);
        }
    }
    readback_buffer.unmap();

    if let Some(parent_dir) = options.output_path.parent() {
        if !parent_dir.as_os_str().is_empty() {
            std::fs::create_dir_all(parent_dir).expect("failed to create output directory");
        }
    }

    image::save_buffer(
        &options.output_path,
        &tightly_packed,
        options.width,
        options.height,
        image::ColorType::Rgba8,
    )
    .expect("failed to write PNG");

    println!(
        "wrote {} ({}x{})",
        options.output_path.display(),
        options.width,
        options.height
    );
}
