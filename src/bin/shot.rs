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
//!   --debug-faces                           face-orientation debug render (colour
//!                                            by outward normal + back-face marker)
//!   --theta/--phi/--dist                    orbit overrides (auto-framed dist)

use std::path::PathBuf;

use voxel_worker::block_palette::{BlockPalette, LoadedMaterial, ThumbnailRenderer};
use voxel_worker::scan_worker::{run_auto_scan_blocking, FaceResolver};
use voxel_worker::{
    create_depth_view, create_msaa_color_view, procedural_material_average_color, render_frame,
    run_egui_frame, CubeFace, EguiPaintBridge, FrameOverlays, GeometryParams, GizmoRenderer,
    GpuContext, GridLatticeRenderer, LayerBand, LayerRange, MaterialChoice, MaterialSource,
    OnionFogParams, OnionFogRenderer, OrbitCamera, PanelState, ProjectionMode, SdfShape, ShapeKind,
    ViewCubeElement, VoxExport, ViewCubeRenderer, VoxelGrid, VoxelProducer, VoxelRenderer,
    COLOR_TARGET_FORMAT,
};

/// Build the onion-skin fog parameters (issue #12) from the camera, grid, and
/// layer-range. World-Y of layer `j` spans `[j - grid_y/2, j+1 - grid_y/2]`; the
/// solid band is layers `[lower, upper]`, the onion band extends `onion_depth`
/// layers on each side.
fn onion_fog_params(
    view_projection: glam::Mat4,
    grid_dimensions: [u32; 3],
    layer_range: LayerRange,
) -> OnionFogParams {
    let half_y = grid_dimensions[1] as f32 / 2.0;
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
        onion_y_min: (lower - depth) - half_y,
        onion_y_max: (upper + 1.0 + depth) - half_y,
        band_y_min: lower - half_y,
        band_y_max: (upper + 1.0) - half_y,
    }
}

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
    /// Whether the origin gizmo is drawn (M5 `--gizmo`).
    show_origin_gizmo: bool,
    /// Whether the block lattice is drawn (M8 `--lattice`).
    show_block_lattice: bool,
    /// Whether the fine floor grid is drawn (M8 `--floor`).
    show_floor_grid: bool,
    /// Whether the voxel cubes render in face-orientation debug mode
    /// (`--debug-faces`): colour by outward face normal + back-facing marker,
    /// cull off. The standard way to verify face winding/culling.
    debug_face_orientation: bool,
    /// When `Some`, write the resolved grid to this `.vox` path (M8
    /// `--export-vox`) instead of (or in addition to) rendering a PNG.
    export_vox_path: Option<PathBuf>,
    /// Whether the view cube is drawn (M5; ON by default, `--no-viewcube` hides).
    show_view_cube: bool,
    /// When `Some`, set the camera directly to this element's snapped angles (M5
    /// `--snap`), overriding `--theta`/`--phi` so the snap table can be verified
    /// headlessly (no tween). Faces, edges (`front-top`) and corners
    /// (`front-top-right`) are all accepted (#13a).
    snap_element: Option<ViewCubeElement>,
    /// Orbit azimuth (radians). Default 0.7.
    theta: f32,
    /// Orbit polar angle from +Y (radians). Default 1.05.
    phi: f32,
    /// Orbit distance. `None` = auto-frame from the grid.
    distance: Option<f32>,
    /// Run the VS auto-detect + scan synchronously before rendering (M6) so the
    /// palette dock is populated in the screenshot.
    scan_vs: bool,
    /// After `--scan-vs`, apply the first scanned block's texture as the active
    /// material (M6) so the model shows a real VS texture (per-voxel sliced).
    apply_first_block: bool,
    /// After `--scan-vs`, apply the first scanned block whose label/key matches
    /// this substring (case-insensitive), going through per-face JSON resolution
    /// (M7 `--apply-block`).
    apply_block_substring: Option<String>,
    /// After `--scan-vs`, print which scanned blocks resolve to genuinely
    /// distinct per-face textures (top != side), then exit (M7 `--list-perface`).
    list_per_face: bool,
    /// Debug escape hatch (M7 verification): resolve an arbitrary blocktype by
    /// its texture stem (e.g. `wood/treetrunk/oak`) even if it is outside the
    /// chiselable allow-list, to demonstrate per-face rendering on a known block.
    force_demo_stem: Option<String>,
    /// Layer-range scrubber lower bound (issue #12), a voxel Y-layer index. When
    /// `None`, defaults to the full range (0). Raw voxel index — no snapping.
    layer_lower: Option<u32>,
    /// Layer-range scrubber upper bound (issue #12), a voxel Y-layer index. When
    /// `None`, defaults to the full range (grid_y). Raw voxel index — no snapping.
    layer_upper: Option<u32>,
    /// Onion-skin depth (issue #12): 0 = off (hard band clip), N = ghost N layers
    /// on each side of the band with screen-door dither.
    onion_depth: u32,
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
            show_origin_gizmo: false,
            show_block_lattice: false,
            show_floor_grid: false,
            debug_face_orientation: false,
            export_vox_path: None,
            show_view_cube: true,
            snap_element: None,
            theta: 0.7,
            phi: 1.05,
            distance: None,
            scan_vs: false,
            apply_first_block: false,
            apply_block_substring: None,
            list_per_face: false,
            force_demo_stem: None,
            layer_lower: None,
            layer_upper: None,
            onion_depth: 0,
        }
    }
}

/// Parse a single face name into a [`CubeFace`].
fn parse_face_name(value: &str) -> CubeFace {
    match value {
        "front" => CubeFace::Front,
        "back" => CubeFace::Back,
        "left" => CubeFace::Left,
        "right" => CubeFace::Right,
        "top" => CubeFace::Top,
        "bottom" => CubeFace::Bottom,
        other => panic!("--snap face must be front|back|left|right|top|bottom, got '{other}'"),
    }
}

/// Parse a `--snap` value into a [`ViewCubeElement`]. Accepts a single face
/// (`front`), a hyphen-joined edge (`front-top`, 2 adjacent faces) or a corner
/// (`front-top-right`, 3 mutually-adjacent faces). Opposite faces (e.g.
/// `front-back`) share no edge/corner and are rejected.
fn parse_snap_element(value: &str) -> ViewCubeElement {
    let lower = value.to_ascii_lowercase();
    let faces: Vec<CubeFace> = lower.split('-').map(parse_face_name).collect();
    // Reject any pair of faces lying on the same axis (opposite or duplicate):
    // their normals don't define a real edge/corner.
    for (i, a) in faces.iter().enumerate() {
        for b in &faces[i + 1..] {
            if a.normal().abs() == b.normal().abs() {
                panic!(
                    "--snap '{value}' combines faces on the same axis; \
                     use adjacent faces (e.g. front-top, front-top-right)"
                );
            }
        }
    }
    match faces.as_slice() {
        [a] => ViewCubeElement::from_face(*a),
        [a, b] => ViewCubeElement::from_edge(*a, *b),
        [a, b, c] => ViewCubeElement::from_corner(*a, *b, *c),
        _ => panic!("--snap must name 1 (face), 2 (edge) or 3 (corner) faces, got '{value}'"),
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
            "--scan-vs" => {
                options.scan_vs = true;
            }
            "--apply-first-block" => {
                options.apply_first_block = true;
            }
            "--apply-block" => {
                options.apply_block_substring =
                    Some(args.next().expect("--apply-block requires a substring"));
            }
            "--list-perface" => {
                options.list_per_face = true;
            }
            "--force-demo-stem" => {
                options.force_demo_stem =
                    Some(args.next().expect("--force-demo-stem requires a texture stem"));
            }
            "--gizmo" => {
                options.show_origin_gizmo = true;
            }
            "--lattice" => {
                options.show_block_lattice = true;
            }
            "--floor" => {
                options.show_floor_grid = true;
            }
            "--debug-faces" => {
                options.debug_face_orientation = true;
            }
            "--layer-lower" => {
                options.layer_lower = Some(
                    args.next()
                        .expect("--layer-lower requires a value")
                        .parse()
                        .expect("--layer-lower must be a non-negative integer"),
                );
            }
            "--layer-upper" => {
                options.layer_upper = Some(
                    args.next()
                        .expect("--layer-upper requires a value")
                        .parse()
                        .expect("--layer-upper must be a non-negative integer"),
                );
            }
            "--onion" => {
                options.onion_depth = args
                    .next()
                    .expect("--onion requires a value")
                    .parse()
                    .expect("--onion must be a non-negative integer (0 = off)");
            }
            "--export-vox" => {
                options.export_vox_path = Some(PathBuf::from(
                    args.next().expect("--export-vox requires a path argument"),
                ));
            }
            "--no-viewcube" => {
                options.show_view_cube = false;
            }
            "--snap" => {
                options.snap_element =
                    Some(parse_snap_element(&args.next().expect("--snap requires a value")));
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
                     \x20            [--scan-vs] [--apply-first-block]\n\
                     \x20            [--apply-block <substring>] [--list-perface]\n\
                     \x20            [--force-demo-stem <texture/stem>]\n\
                     \x20            [--gizmo] [--lattice] [--floor] [--no-viewcube]\n\
                     \x20            [--debug-faces]\n\
                     \x20            [--layer-lower <u32>] [--layer-upper <u32>] [--onion <u32>]\n\
                     \x20            [--export-vox <path.vox>]\n\
                     \x20            [--snap <face|edge|corner>  e.g. front, front-top, front-top-right]\n\
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

/// The file stem (no dir, no extension) of a path, for compact log output.
fn file_stem_of(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Resolve an arbitrary texture stem (e.g. `wood/treetrunk/oak`) to per-face
/// textures via the VS JSON index, bypassing the chiselable allow-list. Used by
/// `--force-demo-stem` to demonstrate per-face rendering on a known block when no
/// chiselable block has distinct faces. Returns `None` if no install is found or
/// the stem can't be located on disk.
fn resolve_demo_stem(stem: &str) -> Option<voxel_worker::FaceTextures> {
    use voxel_worker::assets::registry::detect_all_sources;

    // Find the actual variant PNG on disk for this stem, under each install's
    // textures dir, trying both domains. The detectors give us the block dirs.
    let chosen_variant = locate_stem_png(stem)?;
    // Build a synthetic group keyed by the stem so the resolver can look up the
    // matching blocktype (its `base` entries reference this stem's directory).
    let group = voxel_worker::assets::BlockGroup {
        label: stem.rsplit('/').next().unwrap_or(stem).to_string(),
        key: stem.to_string(),
        variants: vec![chosen_variant.clone()],
    };
    let sources = detect_all_sources();
    let mut fallback: Option<voxel_worker::FaceTextures> = None;
    for source in &sources {
        let faces = source.resolve_faces(&group, &chosen_variant);
        if !faces.is_uniform() {
            return Some(faces);
        }
        if fallback.is_none() {
            fallback = Some(faces);
        }
    }
    fallback
}

/// Locate the PNG for a bare texture stem on disk, scanning the same install
/// roots the detectors use, trying the `game` then `survival` domains.
fn locate_stem_png(stem: &str) -> Option<std::path::PathBuf> {
    let appdata = std::env::var("APPDATA").ok()?;
    let assets_root = std::path::Path::new(&appdata)
        .join("Vintagestory")
        .join("assets");
    for domain in ["game", "survival", "creative"] {
        let candidate = assets_root
            .join(domain)
            .join("textures")
            .join("block")
            .join(format!("{stem}.png"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
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
    let grid_y = shape.grid_dimensions()[1];
    // Issue #12: build the layer-range band from the raw CLI voxel indices (no
    // snapping — flags take raw indices). Defaults to the full range.
    let layer_range = LayerRange {
        lower: options.layer_lower.unwrap_or(0).min(grid_y),
        upper: options.layer_upper.unwrap_or(grid_y).min(grid_y),
        snap_to_blocks: true,
        onion_skin: options.onion_depth > 0,
        onion_depth: options.onion_depth.clamp(1, 8),
    };
    let mut panel_state = PanelState {
        geometry: options.geometry,
        projection_mode: options.projection_mode,
        material: options.material,
        show_grid_overlay: options.show_grid_overlay,
        show_view_cube: options.show_view_cube,
        show_origin_gizmo: options.show_origin_gizmo,
        show_block_lattice: options.show_block_lattice,
        show_floor_grid: options.show_floor_grid,
        debug_face_orientation: options.debug_face_orientation,
        layer_range,
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

    // M8: `--export-vox` writes the resolved grid as a MagicaVoxel .vox and then
    // exits (no render needed — this is the headless verification path).
    if let Some(vox_path) = &options.export_vox_path {
        let representative = procedural_material_average_color(options.material);
        let export = VoxExport::from_grid(&grid, representative);
        match export.write(vox_path) {
            Ok(bytes) => println!(
                "wrote {} ({} voxels, {} model(s), {} bytes)",
                vox_path.display(),
                export.voxel_count(),
                export.model_count(),
                bytes
            ),
            Err(error) => {
                eprintln!("export .vox failed: {error}");
                std::process::exit(1);
            }
        }
        return;
    }

    let voxel_renderer =
        VoxelRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT, &grid);
    let gizmo_renderer = GizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, grid.dimensions);
    let grid_lattice_renderer = GridLatticeRenderer::new(
        &gpu.device,
        COLOR_TARGET_FORMAT,
        grid.dimensions,
        options.geometry.voxels_per_block,
    );
    let view_cube_renderer = ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    let mut onion_fog_renderer = OnionFogRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // Upload the resolved grid as the fog's 3D occupancy field (issue #12).
    onion_fog_renderer.upload_grid(&gpu.device, &gpu.queue, &grid);
    // Issue #12: the layer-range band for the 3D clip + the measured-diameter
    // readout (widest occupied run in the active band).
    let band = if layer_range.is_full_range(grid_y) && !layer_range.onion_skin {
        LayerBand::FULL
    } else {
        LayerBand {
            band_min: layer_range.lower,
            band_max: layer_range.upper.min(grid_y.saturating_sub(1)),
            onion_depth: if layer_range.onion_skin {
                layer_range.onion_depth.clamp(1, 8)
            } else {
                0
            },
        }
    };
    let measured_diameter = grid.widest_run_in_band(layer_range.lower, layer_range.upper);

    // Build the orbit camera from the CLI flags. `--snap` overrides theta/phi
    // with the face's snapped angles directly (no tween in the headless path).
    let (theta, phi) = match options.snap_element {
        Some(element) => element.snap_angles(),
        None => (options.theta, options.phi),
    };
    let camera = OrbitCamera {
        target: glam::Vec3::ZERO,
        orbit_theta: theta,
        orbit_phi: phi,
        orbit_distance: options
            .distance
            .unwrap_or_else(|| OrbitCamera::auto_framed_distance(grid.dimensions)),
        projection_mode: options.projection_mode,
    };
    let aspect_ratio = options.width as f32 / options.height as f32;
    let view_projection = camera.view_projection(aspect_ratio);
    voxel_renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        shape.grid_dimensions(),
        options.geometry.voxels_per_block,
        options.show_grid_overlay,
        options.debug_face_orientation,
        band,
    );
    gizmo_renderer.update_uniforms(&gpu.queue, view_projection);
    grid_lattice_renderer.update_uniforms(&gpu.queue, view_projection);
    view_cube_renderer.update_uniforms(&gpu.queue, camera.view_cube_view_projection());

    // Issue #12: onion-skin volumetric fog params (active when --onion > 0 and not
    // in debug-face mode).
    let onion_active = layer_range.onion_skin && !options.debug_face_orientation;
    if onion_active {
        onion_fog_renderer.update(
            &gpu.queue,
            onion_fog_params(view_projection, shape.grid_dimensions(), layer_range),
        );
    }

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

    // M6: synchronously scan the VS install and populate the palette so the
    // screenshot shows real block thumbnails. Optionally apply the first block.
    let thumbnail_renderer = ThumbnailRenderer::new(&gpu.device, &gpu.queue);
    let mut palette = BlockPalette::default();
    let mut loaded_material: Option<LoadedMaterial> = None;
    if options.scan_vs {
        let (groups, source_name) = run_auto_scan_blocking();
        println!(
            "scan: {} groups from {}",
            groups.len(),
            source_name.as_deref().unwrap_or("(no install found)")
        );
        palette.status = match source_name {
            Some(name) => format!("{} blocks loaded — {}", groups.len(), name),
            None => "No VS install found — use Connect folder".to_string(),
        };

        // M7: per-face resolver, kept alive so each group can resolve its
        // blocktype JSON → per-face PNGs.
        let resolver = FaceResolver::auto();

        // `--list-perface`: report which chiselable blocks have a distinct top vs
        // side face, then exit before any rendering.
        if options.list_per_face {
            let mut distinct = 0usize;
            println!("--- per-face scan (top != side) ---");
            for (group, _) in &groups {
                let variant = group.variants.first().cloned().unwrap_or_default();
                let faces = resolver.resolve(group, &variant);
                if faces.top_differs_from_side() {
                    distinct += 1;
                    println!(
                        "DISTINCT  {:<22} key={}  up={}  side={}  [{:?}]",
                        group.label,
                        group.key,
                        file_stem_of(&faces.paths[2]),
                        file_stem_of(&faces.paths[0]),
                        faces.provenance,
                    );
                }
            }
            println!(
                "--- {distinct}/{} chiselable blocks resolve to distinct per-face textures ---",
                groups.len()
            );
            return;
        }

        // Choose which block to apply (if any) and resolve its per-face textures.
        let chosen: Option<(String, voxel_worker::FaceTextures)> =
            if let Some(stem) = &options.force_demo_stem {
                // Demo escape hatch: resolve an arbitrary texture stem directly via
                // the JSON index, even outside the chiselable allow-list, to prove
                // the per-face mechanism on a known block (e.g. wood/treetrunk/oak).
                resolve_demo_stem(stem).map(|faces| (stem.clone(), faces))
            } else {
                let target = if let Some(substring) = &options.apply_block_substring {
                    let lower = substring.to_ascii_lowercase();
                    groups.iter().find(|(group, _)| {
                        group.label.to_ascii_lowercase().contains(&lower)
                            || group.key.to_ascii_lowercase().contains(&lower)
                    })
                } else if options.apply_first_block {
                    groups.first()
                } else {
                    None
                };
                target.map(|(group, _)| {
                    let variant = group.variants.first().cloned().unwrap_or_default();
                    let faces = resolver.resolve(group, &variant);
                    (group.label.clone(), faces)
                })
            };

        if let Some((label, faces)) = chosen {
            let material = LoadedMaterial::from_faces(
                &gpu.device,
                &gpu.queue,
                voxel_renderer.material_bind_group_layout(),
                voxel_renderer.material_sampler(),
                &faces,
                label.clone(),
            );
            println!(
                "applied block: {label} (per_face={}, provenance={:?})",
                material.is_per_face, faces.provenance
            );
            panel_state.applied_block_label = Some(label);
            loaded_material = Some(material);
        }

        for (group, decoded) in groups {
            palette.add_group(
                &gpu.device,
                &gpu.queue,
                &thumbnail_renderer,
                &mut egui_bridge.renderer,
                group,
                &decoded,
            );
        }
    }

    let prepared = run_egui_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &mut panel_state,
        grid_y,
        measured_diameter,
        &palette,
        raw_input,
        [options.width, options.height],
        pixels_per_point,
    );

    // M6: the active material is a loaded VS block when one was applied,
    // otherwise the procedural choice.
    let material = match &loaded_material {
        Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
        None => MaterialSource::Procedural(options.material),
    };

    let overlays = FrameOverlays {
        gizmo: if options.show_origin_gizmo {
            Some(&gizmo_renderer)
        } else {
            None
        },
        view_cube: if options.show_view_cube {
            Some(&view_cube_renderer)
        } else {
            None
        },
        grid_lattice: Some(&grid_lattice_renderer),
        show_lattice: options.show_block_lattice,
        show_floor: options.show_floor_grid,
        debug_face_mode: options.debug_face_orientation,
        onion_fog: if onion_active {
            Some(&onion_fog_renderer)
        } else {
            None
        },
        target_width: options.width,
        target_height: options.height,
    };

    // Paint via the exact same render-target-agnostic core the window uses.
    render_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &capture_view,
        &msaa_color_view,
        &depth_view,
        &voxel_renderer,
        material,
        &overlays,
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
