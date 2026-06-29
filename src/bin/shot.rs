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
//!   --shape <cylinder|tube|sphere|torus|box|debug-clouds>   (default: cylinder)
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
    run_egui_frame, AppCore, CubeFace, DefId, EguiPaintBridge, FogMode, FrameOverlays,
    GeometryParams,
    GpuContext, InfiniteGridRenderer, LayerBand, LayerRange, MaterialChoice, MaterialSource,
    PointsRenderer, RebuildOutcome, RebuildOutput, SceneGridRenderer,
    Node, NodeBuilder, NodeContent, NodePath, OnionFogRenderer, OrbitCamera, PanelState, Part,
    Point,
    PlaneAxis, ProjectionMode, RegionBlocks, Scene, SdfShape, ShapeKind, Sketch, SketchSolid,
    SketchPoint, TransformGizmoRenderer,
    ViewCubeElement, VoxExport,
    ViewCubeRenderer, VoxelGrid, COLOR_TARGET_FORMAT,
};
use voxel_worker::CuboidMeshRenderer;

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
    /// Whether `--gizmo` was passed: draw the transform gizmo ON the active/
    /// selected node (issue #29 S2). No-op-safe when nothing is selected or the
    /// selection has no extent. The field name is kept for minimal churn.
    show_origin_gizmo: bool,
    /// `--select-node N` (issue #29 S2): override the scene's active selection to
    /// the top-level node at index N, so a headless capture can prove the transform
    /// gizmo FOLLOWS a chosen (non-origin) node. `None` keeps the scene's default
    /// selection. Out-of-range clears the selection (gizmo hidden).
    select_node: Option<usize>,
    /// Whether the block lattice is drawn (M8 `--lattice`).
    show_block_lattice: bool,
    /// Whether the fine floor grid is drawn (M8 `--floor`).
    show_floor_grid: bool,
    /// Whether the world reference grid (the Points: analytic infinite ground plane
    /// plus axes) is drawn (issue #29 S5 `--points`). DEFAULT OFF so the existing
    /// goldens (which never pass `--points`) stay byte-identical; `--points` enables
    /// the Origin Point (and any others) so a deliberate Points golden can be captured.
    show_points: bool,
    /// An OPTIONAL extra Point at the given world BLOCK position (issue #29 Points
    /// fast-follow `--point-at X Y Z`), with its XY ground plane (Z-up) + axes ON, so a
    /// headless capture can verify a second analytic grid plane at a different height
    /// / offset. Only meaningful together with `--points`.
    extra_point_blocks: Option<[i64; 3]>,
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
    /// `--cube-hover <zone>` (#13 Step 2): force a ViewCube chrome zone to read as
    /// hovered so a golden can show a highlighted rotate/roll arrow. `None` = the
    /// normal render (Home/Fit only, no arrows).
    cube_hover: Option<voxel_worker::camera::CubeChromeZone>,
    /// Orbit azimuth (radians). Default 0.7.
    theta: f32,
    /// Orbit polar angle from +Y (radians). Default 1.05.
    phi: f32,
    /// View roll about the forward axis (radians, #13 Step 5). Default 0 (upright).
    /// `--roll <radians>` twists the whole scene AND the ViewCube together.
    roll: f32,
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
    /// Layer-range scrubber lower bound (issue #12), a voxel Z-layer index. When
    /// `None`, defaults to the full range (0). Raw voxel index — no snapping.
    layer_lower: Option<u32>,
    /// Layer-range scrubber upper bound (issue #12), a voxel Z-layer index. When
    /// `None`, defaults to the full range (grid_z). Raw voxel index — no snapping.
    layer_upper: Option<u32>,
    /// Onion-skin depth (issue #12): 0 = off (hard band clip), N = ghost N layers
    /// on each side of the band with screen-door dither.
    onion_depth: u32,
    /// Onion-fog occupancy mode (issue #28). `PerChunk` (DEFAULT since S5b — one apron'd
    /// volume per resident chunk, packed into a small 3D atlas) or `WholeGrid` (the legacy
    /// single whole-grid 3D texture, `--fog=wholegrid`, which disables itself past the
    /// single-3D-texture limit). Per-chunk is A/B-identical on normal scenes and renders
    /// fog at scale where whole-grid cannot.
    fog_mode: FogMode,
    /// `--shape debug-clouds`: replace the parametric producer with the debug
    /// cloud field (several distinct billowy blobs in a mostly-empty volume) at
    /// the requested size/density. The grid dims still come from size×density.
    debug_clouds: bool,
    /// `--demo-scene` (ADR 0001 step 3): ignore the single-shape options and build
    /// a hardcoded multi-node PLACED scene (a sphere at the origin, a box offset
    /// +8 blocks in X, and a clouds Part offset in Z) so the headless capture can
    /// confirm nodes appear separated in space (not overlapping at the origin).
    /// Useful for future headless multi-node checks.
    demo_scene: bool,
    /// `--demo-village` (ADR 0001 step 4): ignore the single-shape options and
    /// build an INSTANCED scene — one "house" `AssemblyDef` placed by four
    /// `Instance` nodes at four offsets — so the headless capture can confirm the
    /// repeated assembly appears at multiple separated locations from a single
    /// stored definition (reuse by reference).
    demo_village: bool,
    /// `--debug-chunks` (ADR 0002 E2, part of #19): after the per-frame frustum
    /// cull, print `chunks: drew X / Y` (visible / total) so the chunking + cull
    /// can be verified headlessly. Zooming/rotating a large scene off-screen draws
    /// fewer chunks; a small scene draws all of them.
    debug_chunks: bool,
    /// `--demo-far-offset` (ADR 0002 streaming S1, part of #18): build a small
    /// recognizable box placed at a LARGE block offset (a block offset of
    /// [100_000, 0, 0]) so the far-lands f32-precision question can be observed.
    /// This is the precision baseline the S4 64-bit/origin-rebasing work regresses
    /// against. NOTE: today's `resolve_region` recentres the composite on its own
    /// centre, so a LONE far node is recentred back to the origin — see the S1
    /// PROGRESS note. The durable artifact is the CPU placement test in `scene.rs`
    /// (the node resolves to absolute coords around 100_000); this render flag is
    /// the visual baseline that S4 must keep jitter-free once the recentre is
    /// removed. Overrides --shape/--size/--density.
    far_offset: bool,
    /// `--demo-far-offset-near` (ADR 0002 streaming S1): the SAME small box as
    /// `--demo-far-offset` but placed at the ORIGIN (a block offset of [0, 0, 0]),
    /// for A/B comparison against the far render. Overrides --shape/--size/--density.
    far_offset_near: bool,
    /// `--demo-sketch-extrude` (ADR 0003 §3i Slice 2a): build a scene containing a
    /// single **sketch → extrude** producer with a RECOGNIZABLE non-box (L-shaped)
    /// footprint extruded up, so the headless capture confirms the new producer
    /// resolves + renders through the same pipeline as `SdfShape`. Overrides
    /// --shape/--size/--density.
    demo_sketch_extrude: bool,
    /// `--demo-groups` (ADR 0001 step 4, UI verification): build a scene with a
    /// top-level `Group` that has two child Tools, plus a sibling top-level Tool
    /// and one `Instance` of a definition — so the headless PANEL capture shows the
    /// INDENTED TREE (a Group with its children nested under it) and the
    /// Definitions list. Overrides --shape/--size/--density.
    demo_groups: bool,
    /// `--synthetic-block` (part of #20 verification): build a LoadedMaterial from
    /// SIX distinct solid-colour faces in-process (no VS install needed) and apply
    /// it as the active material. Lets the headless harness prove the cuboid path
    /// now renders a loaded per-face D2Array (and that cuboid vs instanced match per
    /// face). Overrides --scan-vs/--apply-block material selection.
    synthetic_block: bool,
    /// `--replay <path>` (ADR 0003 Phase C, slice C3): build the scene by REPLAYING a
    /// newline-delimited-JSON Intent script through `AppCore::apply_intent` instead of
    /// from a `--shape`/`--demo-*` source. The file is one [`voxel_worker::Intent`] per
    /// non-empty line, applied IN ORDER to the default seed scene (the same base the
    /// windowed app starts from, via `voxel_worker::default_replay_seed_scene`); the
    /// final post-replay scene flows into the SAME render path (resolve -> offscreen
    /// render -> write PNG to `--out`). `--replay` takes precedence over the demo/shape
    /// scene sources (it is the scene SOURCE); the camera/projection flags
    /// (`--proj`, `--theta`, `--phi`, `--dist`, ...) still apply. `None` keeps the
    /// existing demo/shape behaviour, byte-identical to today.
    replay_path: Option<PathBuf>,
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
            select_node: None,
            show_block_lattice: false,
            show_floor_grid: false,
            show_points: false,
            extra_point_blocks: None,
            debug_face_orientation: false,
            export_vox_path: None,
            show_view_cube: true,
            snap_element: None,
            cube_hover: None,
            theta: 0.7,
            phi: 1.05,
            roll: 0.0,
            distance: None,
            scan_vs: false,
            apply_first_block: false,
            apply_block_substring: None,
            list_per_face: false,
            force_demo_stem: None,
            layer_lower: None,
            layer_upper: None,
            onion_depth: 0,
            // Issue #28 S5b: per-chunk fog is now the DEFAULT. It is visually identical
            // to whole-grid on normal scenes (A/B 0.0000%) and strictly better at scale
            // (whole-grid disables fog past `max_texture_dimension_3d`; per-chunk doesn't).
            // `--fog=wholegrid` selects the legacy whole-grid path.
            fog_mode: FogMode::PerChunk,
            debug_clouds: false,
            debug_chunks: false,
            demo_scene: false,
            demo_village: false,
            demo_sketch_extrude: false,
            demo_groups: false,
            far_offset: false,
            far_offset_near: false,
            synthetic_block: false,
            replay_path: None,
        }
    }
}

/// The block offset of the far-offset demo box (ADR 0002 streaming S1; S4b makes it
/// jitter-free). A large block offset, resolved through the now-`i64` voxel offset
/// (widened in S4a). At
/// density 16 this is **16 million voxels** from the origin — past the f32
/// exact-integer ceiling (2²⁴ ≈ 16.7M), where the old recentre-AFTER-f32-add path
/// lost the voxel-centre `.5` fraction on EVERY voxel (the real precision breakdown
/// the S1 flag exists to expose). The S4b camera-relative rebase (subtract the
/// floating origin in i64 BEFORE the f32 downcast) renders this byte-identical to the
/// near box. (At the previous 100_000 the f32 ULP at 1.6M is 0.125, so `.5` survived
/// and the box never actually jittered — only the demo's UI text differed.)
const FAR_OFFSET_BLOCKS: [i64; 3] = [1_000_000, 0, 0];

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

/// Parse a `--cube-hover` value (#13 Step 2) into the forced hovered chrome zone.
/// Accepts the rotate/roll arrows and the Home/Fit badges so a golden can show
/// any highlighted chrome element.
fn parse_cube_hover(value: &str) -> voxel_worker::camera::CubeChromeZone {
    use voxel_worker::camera::{ArrowDir, CubeChromeZone, RollDir};
    match value.to_ascii_lowercase().as_str() {
        "rotate-up" | "up" => CubeChromeZone::RotateArrow(ArrowDir::Up),
        "rotate-down" | "down" => CubeChromeZone::RotateArrow(ArrowDir::Down),
        "rotate-left" | "left" => CubeChromeZone::RotateArrow(ArrowDir::Left),
        "rotate-right" | "right" => CubeChromeZone::RotateArrow(ArrowDir::Right),
        "roll-cw" | "cw" => CubeChromeZone::RollArrow(RollDir::Cw),
        "roll-ccw" | "ccw" => CubeChromeZone::RollArrow(RollDir::Ccw),
        "home" => CubeChromeZone::HomeButton,
        "fit" => CubeChromeZone::FitButton,
        // #13 Step 6.2: an `element:<spec>` value forces a hovered face/edge/corner
        // so a golden can show the element highlight on the cube body. Reuses the
        // `--snap` element parser (`front`, `front-top`, `front-top-right`).
        other if other.starts_with("element:") => {
            CubeChromeZone::Element(parse_snap_element(&other["element:".len()..]))
        }
        other => panic!(
            "--cube-hover must be one of rotate-up|rotate-down|rotate-left|rotate-right|\
             roll-cw|roll-ccw|home|fit|element:<face|edge|corner>, got '{other}'"
        ),
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
    // The `--size-*` flags are BLOCK counts (the CLI's whole-block ergonomics); the
    // geometry mirror is now voxel-canonical (ADR 0003 §3f(0)), so collect the block
    // sizes here and finalise `size_voxels = blocks · density` AFTER the loop (so the
    // flags are order-independent with `--density`). Default 5×1×5 blocks.
    let mut size_blocks_cli: [u32; 3] = [5, 1, 5];
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
                let value = args.next().expect("--shape requires a value");
                if value == "debug-clouds" || value == "clouds" {
                    // Not an SDF shape: switch the producer to the debug cloud
                    // field. Leave geometry.shape at its default so size/density
                    // (→ grid dims) and the panel still render sensibly.
                    options.debug_clouds = true;
                } else {
                    options.geometry.shape = parse_shape(&value);
                }
            }
            "--size-x" => {
                size_blocks_cli[0] = args
                    .next()
                    .expect("--size-x requires a value")
                    .parse()
                    .expect("--size-x must be a positive integer");
            }
            "--size-y" => {
                size_blocks_cli[1] = args
                    .next()
                    .expect("--size-y requires a value")
                    .parse()
                    .expect("--size-y must be a positive integer");
            }
            "--size-z" => {
                size_blocks_cli[2] = args
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
            "--select-node" => {
                options.select_node = Some(
                    args.next()
                        .expect("--select-node requires an index")
                        .parse()
                        .expect("--select-node index must be a non-negative integer"),
                );
            }
            "--lattice" => {
                options.show_block_lattice = true;
            }
            "--floor" => {
                options.show_floor_grid = true;
            }
            "--points" => {
                options.show_points = true;
            }
            "--point-at" => {
                // Three BLOCK coordinates for an extra Point (XZ plane + axes on).
                let x = args.next().expect("--point-at requires X Y Z").parse()
                    .expect("--point-at X must be an integer");
                let y = args.next().expect("--point-at requires X Y Z").parse()
                    .expect("--point-at Y must be an integer");
                let z = args.next().expect("--point-at requires X Y Z").parse()
                    .expect("--point-at Z must be an integer");
                options.extra_point_blocks = Some([x, y, z]);
            }
            "--debug-faces" => {
                options.debug_face_orientation = true;
            }
            "--debug-chunks" => {
                options.debug_chunks = true;
            }
            "--demo-scene" => {
                options.demo_scene = true;
            }
            "--demo-village" => {
                options.demo_village = true;
            }
            "--demo-sketch-extrude" => {
                options.demo_sketch_extrude = true;
            }
            "--demo-groups" => {
                options.demo_groups = true;
            }
            "--synthetic-block" => {
                options.synthetic_block = true;
            }
            "--replay" => {
                options.replay_path = Some(PathBuf::from(
                    args.next().expect("--replay requires a path argument"),
                ));
            }
            "--demo-far-offset" => {
                options.far_offset = true;
            }
            "--demo-far-offset-near" => {
                options.far_offset_near = true;
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
            // Issue #28: select the onion-fog occupancy source. Accepts both
            // `--fog perchunk` and `--fog=perchunk`. Default is `perchunk` (S5b);
            // `--fog=wholegrid` selects the legacy whole-grid path.
            other_fog if other_fog == "--fog" || other_fog.starts_with("--fog=") => {
                let value = if let Some(eq) = other_fog.strip_prefix("--fog=") {
                    eq.to_string()
                } else {
                    args.next().expect("--fog requires a value (wholegrid|perchunk)")
                };
                options.fog_mode = match value.to_ascii_lowercase().as_str() {
                    "wholegrid" | "whole-grid" | "whole" => FogMode::WholeGrid,
                    "perchunk" | "per-chunk" | "chunk" => FogMode::PerChunk,
                    other => panic!("--fog must be wholegrid|perchunk, got '{other}'"),
                };
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
            "--cube-hover" => {
                options.cube_hover = Some(parse_cube_hover(
                    &args.next().expect("--cube-hover requires a value"),
                ));
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
            "--roll" => {
                options.roll = args
                    .next()
                    .expect("--roll requires a value")
                    .parse()
                    .expect("--roll must be a float (radians)");
            }
            "--roll-quarters" => {
                // Convenience for the headless roll golden: N quarter-turns (×π/2).
                let quarters: f32 = args
                    .next()
                    .expect("--roll-quarters requires a value")
                    .parse()
                    .expect("--roll-quarters must be a number");
                options.roll = quarters * std::f32::consts::FRAC_PI_2;
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
                     \x20            [--shape <cylinder|tube|sphere|torus|box|debug-clouds>]\n\
                     \x20            [--size-x <u32>] [--size-y <u32>] [--size-z <u32>]\n\
                     \x20            [--density <u32>] [--wall <u32>]\n\
                     \x20            [--proj <perspective|ortho>]\n\
                     \x20            [--material <stone|wood|plain>] [--grid]\n\
                     \x20            [--scan-vs] [--apply-first-block]\n\
                     \x20            [--apply-block <substring>] [--list-perface]\n\
                     \x20            [--synthetic-block]\n\
                     \x20            [--replay <script.jsonl>]\n\
                     \x20            [--force-demo-stem <texture/stem>]\n\
                     \x20            [--gizmo] [--select-node <usize>] [--lattice] [--floor] [--points] [--point-at <X Y Z>] [--no-viewcube]\n\
                     \x20            [--debug-faces] [--debug-chunks]\n\
                     \x20            [--demo-scene] [--demo-village] [--demo-groups]\n\
                     \x20            [--demo-sketch-extrude]\n\
                     \x20            [--demo-far-offset] [--demo-far-offset-near]\n\
                     \x20            [--layer-lower <u32>] [--layer-upper <u32>] [--onion <u32>]\n\
                     \x20            [--fog <wholegrid|perchunk>]\n\
                     \x20            [--export-vox <path.vox>]\n\
                     \x20            [--snap <face|edge|corner>  e.g. front, front-top, front-top-right]\n\
                     \x20            [--cube-hover <rotate-up|rotate-down|rotate-left|rotate-right|roll-cw|roll-ccw|home|fit|element:<face|edge|corner>>]\n\
                     \x20            [--theta <f32>] [--phi <f32>] [--roll <f32>] [--roll-quarters <n>] [--dist <f32>]\n\
                     Defaults: --out shots/m1.png --width 1280 --height 800\n\
                     \x20         --shape cylinder --size-x 5 --size-y 1 --size-z 5\n\
                     \x20         --density 16 --wall 1 --proj perspective\n\
                     \x20         --material stone (grid off)\n\
                     \x20         --theta 0.7 --phi 1.05 --dist <auto-framed>\n\
                     \n\
                     \x20  --demo-scene  build a hardcoded multi-node placed scene\n\
                     \x20                (sphere at origin + box offset +8 blocks in X\n\
                     \x20                + clouds offset in Z) to verify separated, non-\n\
                     \x20                overlapping placement (ADR 0001 step 3). Overrides\n\
                     \x20                --shape/--size/--density.\n\
                     \x20  --demo-village build an INSTANCED scene: one 'house' definition\n\
                     \x20                placed by 4 Instances at 4 offsets, proving reuse-\n\
                     \x20                by-reference (ADR 0001 step 4). Overrides\n\
                     \x20                --shape/--size/--density.\n\
                     \x20  --demo-groups build a scene with a top-level Group (2 child\n\
                     \x20                Tools), a sibling Tool and an Instance, so the\n\
                     \x20                captured PANEL shows the indented tree + Definitions\n\
                     \x20                (ADR 0001 step 4 UI). Overrides --shape/--size/--density.\n\
                     \x20  --demo-sketch-extrude build a single 2D-sketch→extrude producer\n\
                     \x20                with an L-shaped footprint extruded up (ADR 0003 §3i\n\
                     \x20                Slice 2a) — a non-box a primitive can't make. Overrides\n\
                     \x20                --shape/--size/--density.\n\
                     \x20  --demo-far-offset      build a small 4³ box at offset [100_000,0,0]\n\
                     \x20                blocks (ADR 0002 streaming S1). Precision baseline:\n\
                     \x20                today's recentre maps it to the origin, so far jitter\n\
                     \x20                is hidden until S4. Overrides --shape/--size/--density.\n\
                     \x20  --demo-far-offset-near the SAME box at the origin, for A/B compare.\n\
                     \x20  --replay <path>  build the scene by replaying a newline-delimited\n\
                     \x20                JSON Intent script (one Intent per line) through\n\
                     \x20                AppCore::apply_intent, applied in order to the default\n\
                     \x20                seed scene; the final post-replay frame is rendered to\n\
                     \x20                --out. Takes precedence over --shape/--demo-* (the scene\n\
                     \x20                SOURCE); camera/projection flags still apply (ADR 0003 C3)."
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("warning: ignoring unknown argument '{other}'");
            }
        }
    }
    // Finalise the voxel-canonical size from the requested BLOCK counts at the final
    // density, retaining each axis as a whole-block measurement (ADR 0003 §3f(0)).
    let built =
        SdfShape::from_blocks(options.geometry.shape, size_blocks_cli, options.geometry.wall_blocks, options.geometry.voxels_per_block);
    options.geometry.size_voxels = built.size_voxels;
    options.geometry.size_measurements = if built.has_retained_size_measurements() {
        Some(Box::new(built.size_measurements()))
    } else {
        None
    };
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

/// Mint stable [`NodeId`]s for a freshly-built demo scene and select its first
/// top-level node by id (ADR 0003 Phase B3: selection is keyed by [`NodeId`], so a
/// demo built with positional intent ("select node 0") must resolve that to an id
/// after minting). The later `ensure_node_ids` on the load path is idempotent.
fn selecting_first_node(mut scene: Scene) -> Scene {
    scene.ensure_node_ids();
    scene.active = scene.roots.first().copied();
    scene
}

/// Build the `--demo-scene` (ADR 0001 step 3): a hardcoded multi-node PLACED
/// scene proving disjoint placement. A sphere at the origin, a box offset +8
/// blocks in X, and a torus offset +6 blocks in Z. Each Tool is 5 blocks, so the
/// offsets open clear gaps and the three solids sit visibly apart (no overlap at
/// the origin) — the headless check the demo exists to confirm.
///
/// NOTE (ADR deviation): the task example named a clouds Part as the third node.
/// The `DebugClouds` Part has no intrinsic bounded size — it fills whatever region
/// it is handed (a bounded stored body is a later Part variant), so as a region-
/// filling fog it would densely OCCLUDE the sphere and box and defeat the very
/// separation the demo verifies. A third SDF Tool (torus) is a crisp, bounded
/// solid that makes the disjoint placement unambiguous in the PNG. Part placement
/// itself is covered by the scene.rs unit tests (a Part stamps under its offset),
/// and the in-app inspector offsets both Tools and Parts.
fn build_demo_scene(voxels_per_block: u32) -> Scene {
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(
            format!("{kind:?}"),
            NodeContent::Tool { shape, material },
        );
        node.transform = voxel_worker::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
    ]));
    // Density is document-level (ADR 0003 §3f(0)).
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-village` (ADR 0001 step 4): an **instanced** scene that
/// proves reuse-by-reference. One small "house" [`AssemblyDef`] (a Box body Tool
/// with a Cylinder "chimney" Tool offset on top, as a `Group`) is stored ONCE in
/// `definitions`; the top-level scene places it by FOUR [`NodeContent::Instance`]
/// nodes at four different X/Z offsets. The four houses appear at four separated
/// locations from a single definition — the village-of-reused-houses case. The
/// headless capture confirms the repeated assembly shows up at multiple disjoint
/// locations.
fn build_demo_village(voxels_per_block: u32) -> Scene {
    let house_def_id = DefId(1);
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = voxel_worker::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };

    // The house: a 2³ stone body with a 1×2×1 wood "chimney" sitting on top, so the
    // chimney's local offset is RELATIVE to the house (it composes down through the
    // instance + group transforms). The body is kept small (2 blocks) so that four
    // instances stay well under the renderer's drawn-instance cap and all four draw.
    // Four instances of the SAME definition in a straight row, 8 blocks apart in X
    // (a 4-block house → 4-block gap between neighbours). A row (not a 2×2 grid, in
    // which diagonal pairs self-occlude from an isometric angle) keeps all four
    // houses non-overlapping in screen space when viewed perpendicular to the row,
    // so the headless PNG unambiguously shows the repeated assembly at four
    // separated locations from a single stored definition.
    let instance = |name: &str, offset: [i64; 3]| {
        let mut node = Node::new(name, NodeContent::Instance(house_def_id));
        node.transform = voxel_worker::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = Scene::from_nodes(vec![
        instance("House 1", [0, 0, 0]),
        instance("House 2", [6, 0, 0]),
        instance("House 3", [12, 0, 0]),
        instance("House 4", [18, 0, 0]),
    ]);
    // The house: a 2³ stone body with a 1×2×1 wood "chimney" sitting on top, so the
    // chimney's local offset is RELATIVE to the house (it composes down through the
    // instance + group transforms). The body is kept small (2 blocks) so that four
    // instances stay well under the renderer's drawn-instance cap and all four draw.
    scene.add_definition(
        house_def_id,
        "House",
        vec![
            tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
            tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
        ],
    );
    scene.voxels_per_block = voxels_per_block;
    selecting_first_node(scene)
}

/// Build the `--demo-sketch-extrude` (ADR 0003 §3i Slice 2a): a single
/// **sketch → extrude → volume** producer with a RECOGNIZABLE non-box footprint —
/// an L-shaped (plus a notch) profile on the GROUND plane, extruded UP several
/// blocks. A box obviously cannot make this footprint, so the headless capture
/// proves the new producer resolves + renders through the SAME pipeline as `SdfShape`.
///
/// The profile is an L: a `4×4`-block square with its top-right `2×2`-block
/// quadrant removed (a reflex vertex), at the document density `d`, extruded
/// `3` blocks (`3·d` voxels) along +Z (Z-up: "up"). The whole footprint is a whole
/// multiple of blocks so it sits cleanly on the lattice in the recentred render frame.
fn build_demo_sketch_extrude(voxels_per_block: u32) -> Scene {
    let density = voxels_per_block.max(1) as i64;
    let two = 2 * density;
    let four = 4 * density;
    // L footprint (CCW), in voxels on the XY ground plane (PlaneAxis::Z in-plane axes
    // are X,Y): outer 0..4×0..2 blocks plus the left 0..2×2..4 block column, leaving
    // the top-right quadrant empty. Extruded UP along +Z.
    let profile = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(four, 0),
        SketchPoint::new(four, two),
        SketchPoint::new(two, two), // reflex vertex (the inside corner of the L)
        SketchPoint::new(two, four),
        SketchPoint::new(0, four),
    ];
    let producer = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), (3 * density) as u32);
    let node = Node::new(
        "Sketch L",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Wood,
        },
    );
    let mut scene = selecting_first_node(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-groups` (ADR 0001 step 4, UI verification): a scene that
/// exercises the indented TREE in the panel. A top-level `Group` ("Cluster") holds
/// two child Tools (a Sphere + a Box at a small offset); a sibling top-level Box
/// Tool sits beside it; and one `Instance` of a small "Widget" definition sits
/// beyond. So the captured panel node list shows: the Group with its two children
/// nested+indented under it, a top-level Tool, and an Instance row, plus the
/// Definitions list — the whole authoring surface this step adds.
fn build_demo_groups(voxels_per_block: u32) -> Scene {
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material, name: &str| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = voxel_worker::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };

    let widget_def_id = DefId(1);

    // A Group with two children, placed at the origin; the children carry their own
    // local offsets relative to the Group.
    let cluster = NodeBuilder::group_at(
        "Cluster",
        [0, 0, 0],
        voxels_per_block,
        vec![
            tool(ShapeKind::Sphere, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone, "Core").into(),
            tool(ShapeKind::Box, [2, 2, 2], [3, 0, 0], MaterialChoice::Wood, "Shell").into(),
        ],
    );

    let lone = tool(ShapeKind::Box, [2, 2, 2], [8, 0, 0], MaterialChoice::Wood, "Lone");
    let mut widget_instance = Node::new("Widget instance", NodeContent::Instance(widget_def_id));
    widget_instance.transform = voxel_worker::scene::NodeTransform::from_blocks([12, 0, 0], voxels_per_block);

    let mut scene = Scene::from_nodes(vec![
        cluster,
        NodeBuilder::Leaf(lone),
        NodeBuilder::Leaf(widget_instance),
    ]);
    scene.add_definition(
        widget_def_id,
        "Widget",
        vec![tool(ShapeKind::Sphere, [2, 2, 2], [0, 0, 0], MaterialChoice::Plain, "Ball")],
    );
    scene.voxels_per_block = voxels_per_block;
    selecting_first_node(scene)
}

/// Build the `--demo-far-offset` / `--demo-far-offset-near` scene (ADR 0002
/// streaming S1, part of #18): a single small recognizable box Tool placed either
/// at the FAR offset ([`FAR_OFFSET_BLOCKS`], i.e. 100_000 blocks in X) or at the
/// ORIGIN, for an A/B precision baseline.
///
/// The box is a 4³-block solid (a crisp, unambiguous shape that frames cleanly).
/// At density 16 the far placement sits 1.6M voxels from the origin in ABSOLUTE
/// composite space — which the CPU placement test in `scene.rs` asserts directly.
///
/// IMPORTANT (today's render math): `Scene::resolve_region` recentres the
/// composite on its OWN centre, so a lone far box is recentred straight back to
/// the origin before rendering. The far and near renders therefore look identical
/// today — f32 jitter from the large offset cannot show up in the live render
/// until S4 removes the recentre / adds origin-rebasing. This flag exists to be
/// the visual regression target for S4 (it must STAY jitter-free once S4 lands).
fn build_far_offset_scene(voxels_per_block: u32, far: bool) -> Scene {
    let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
    let mut node = Node::new(
        if far { "Far box" } else { "Near box" },
        NodeContent::Tool {
            shape,
            material: MaterialChoice::Stone,
        },
    );
    node.transform = voxel_worker::scene::NodeTransform::from_blocks(
        if far { FAR_OFFSET_BLOCKS } else { [0, 0, 0] },
        voxels_per_block,
    );
    let mut scene = Scene::single_node(node);
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// `--replay` (ADR 0003 Phase C, slice C3): **replay-script -> Scene**.
///
/// The script at `replay_path` is **newline-delimited JSON**: one
/// [`voxel_worker::Intent`] per non-empty line. Each line is parsed with
/// `serde_json::from_str::<Intent>` and applied IN ORDER, via
/// [`AppCore::apply_intent`], to the default seed scene (the same base the windowed
/// app starts from — `voxel_worker::default_replay_seed_scene`). Blank /
/// whitespace-only lines are skipped. The returned [`Scene`] is the post-replay
/// document; the caller flows it into the SAME render path (resolve -> offscreen
/// render -> write PNG to `--out`).
///
/// File IO lives here; the parse + apply core is `voxel_worker::replay_intent_script`
/// (lib-tested without a GPU). On a read error, or a JSON parse error on any line,
/// returns `Err` with a clear message naming the offending line — `run_capture`
/// prints it and exits non-zero (no panic).
fn build_scene_from_replay(replay_path: &std::path::Path) -> Result<Scene, String> {
    let script = std::fs::read_to_string(replay_path).map_err(|error| {
        format!("--replay: failed to read '{}': {error}", replay_path.display())
    })?;
    voxel_worker::replay_intent_script(&script)
        .map_err(|error| format!("--replay: '{}': {error}", replay_path.display()))
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
    let shape = SdfShape::from_geometry(options.geometry.clone());
    // Z-up: layers are Z-slices, so the layer track spans the Z dimension (index 2).
    let grid_z = shape.grid_dimensions(options.geometry.voxels_per_block)[2];
    // Issue #12: build the layer-range band from the raw CLI voxel indices (no
    // snapping — flags take raw indices). Defaults to the full range.
    let layer_range = LayerRange {
        lower: options.layer_lower.unwrap_or(0).min(grid_z),
        upper: options.layer_upper.unwrap_or(grid_z).min(grid_z),
        snap_to_blocks: true,
        onion_skin: options.onion_depth > 0,
        onion_depth: options.onion_depth.clamp(1, 8),
    };
    let mut panel_state = PanelState {
        geometry: options.geometry.clone(),
        projection_mode: options.projection_mode,
        material: options.material,
        show_view_cube: options.show_view_cube,
        // Issue #31: the grid masters are no longer mirrored onto PanelState. The CLI
        // `--grid`/`--lattice`/`--floor` flags drive `scene.master_*` directly below
        // (the single source of truth); the scene's masters otherwise default ON.
        debug_face_orientation: options.debug_face_orientation,
        layer_range,
        ..PanelState::default()
    };
    // ADR 0001 step 2/3: resolve through a scene. `--demo-scene` builds a
    // hardcoded multi-node PLACED scene (sphere at origin + box offset +8 in X +
    // clouds offset in Z) to verify separated placement; otherwise a one-node
    // scene — a Tool, or a DebugClouds Part when `--shape debug-clouds`. Seed the
    // panel's scene so the node-list section renders the nodes in the captured
    // panel.
    // `--replay` (ADR 0003 Phase C, slice C3) is the highest-precedence scene SOURCE:
    // when present it REPLACES the demo/shape sources entirely — the scene is built by
    // replaying the JSONL Intent script against the default seed via
    // `AppCore::apply_intent`. A parse/read error is reported (line number + bad line)
    // and the process exits non-zero, rather than panicking. The camera/projection
    // flags below still apply to the replay render.
    let mut scene = if let Some(replay_path) = &options.replay_path {
        match build_scene_from_replay(replay_path) {
            Ok(replayed_scene) => replayed_scene,
            Err(message) => {
                eprintln!("{message}");
                std::process::exit(1);
            }
        }
    } else if options.far_offset || options.far_offset_near {
        build_far_offset_scene(options.geometry.voxels_per_block, options.far_offset)
    } else if options.demo_groups {
        build_demo_groups(options.geometry.voxels_per_block)
    } else if options.demo_sketch_extrude {
        build_demo_sketch_extrude(options.geometry.voxels_per_block)
    } else if options.demo_village {
        build_demo_village(options.geometry.voxels_per_block)
    } else if options.demo_scene {
        build_demo_scene(options.geometry.voxels_per_block)
    } else if options.debug_clouds {
        Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(Part::DebugClouds { seed: 0 }),
        ))
    } else {
        Scene::from_geometry(options.geometry.clone(), options.material)
    };
    // Issue #29 S5: Points are SUPPRESSED unless `--points`. The headless scenes do
    // NOT synthesize an Origin Point (that runs on the windowed load/seed path), so by
    // default `scene.points` is empty → nothing renders and the panel's Points section
    // is zero-height (the 6 existing goldens stay byte-identical). `--points` adds the
    // Origin (ground + axes on by default) so the deliberate Points golden shows the
    // world reference grid.
    if options.show_points {
        scene.ensure_origin_point();
        // An optional extra Point (issue #29 Points fast-follow) at a chosen world
        // block position with its XY ground plane (Z-up) + axes on, so a headless
        // capture can verify a second analytic grid plane at a different height/offset.
        if let Some(position_blocks) = options.extra_point_blocks {
            scene.points.push(Point {
                name: "Extra".to_string(),
                position_blocks,
                plane_xy: true,
                plane_xz: false,
                plane_yz: false,
                ..Point::default()
            });
        }
    }
    // ADR 0003 Phase B: mint a stable NodeId for every node before the scene is
    // consumed (idempotent; nothing reads the id yet in B1).
    scene.ensure_node_ids();
    panel_state.scene = scene.clone();
    // Issue #29 S2: `--select-node N` overrides the active selection so a headless
    // capture can place the transform gizmo on a chosen (non-origin) node and prove
    // it follows the selection. An out-of-range index clears the selection.
    if let Some(index) = options.select_node {
        // ADR 0003 Phase B3: selection is keyed by NodeId. Parse the same top-level
        // index as before, then resolve it to that node's stable id (ids were minted
        // by `ensure_node_ids` above), so the SAME `--select-node N` argument selects
        // the SAME node. An out-of-range index resolves to None → clears selection.
        panel_state.scene.active = panel_state
            .scene
            .id_at_path(&NodePath::root_index(index));
    }
    // Issue #29 S3: the per-object block lattice + floor grid are now gated by a
    // scene master ANDed with each NODE's own toggle (default OFF), so a headless
    // capture must enable them explicitly. `--lattice`/`--floor` set the matching
    // scene master AND turn the per-object flag on for ONE node — the
    // `--select-node N` node (else the top-level node 0). This proves the grid
    // hugs that node's enclosing blocks while a sibling shows none.
    if options.show_block_lattice || options.show_floor_grid {
        panel_state.scene.master_block_lattice = options.show_block_lattice;
        panel_state.scene.master_floor_grid = options.show_floor_grid;
        let grid_node = options.select_node.unwrap_or(0);
        // B5: address the top-level node by position via the public path helper.
        if let Some(node) = panel_state
            .scene
            .node_at_path_mut(&NodePath::root_index(grid_node))
        {
            node.grids.block_lattice = options.show_block_lattice;
            node.grids.floor_grid = options.show_floor_grid;
        }
    }
    // Issue #29 S4: the on-face voxel grid is likewise per-object now (master AND a
    // node's own `voxel_grid_on_faces`). `--grid` sets the scene master AND turns the
    // per-object flag on for ONE node — the `--select-node N` node (else node 0) — so
    // a 2-node capture shows the enabled node's faces bearing bold block-edge grid
    // lines while the sibling's faces show none. The bit is baked at resolve, so this
    // must run BEFORE the resolve below.
    if options.show_grid_overlay {
        // Mutate the LOCAL `scene` — the resolve below reads it (not
        // `panel_state.scene`), so the flag must be baked here for the bit to land
        // on each voxel's `material_id`. Re-sync the panel copy so the inspector
        // and per-frame uniforms agree.
        scene.master_voxel_grid = true;
        let grid_node = options.select_node.unwrap_or(0);
        // B5: address the top-level node by position via the public path helper.
        if let Some(node) = scene.node_at_path_mut(&NodePath::root_index(grid_node)) {
            node.grids.voxel_grid_on_faces = true;
        }
        panel_state.scene = scene.clone();
    }
    // The resolve region: for a placed multi-node scene this is the whole
    // composite extent (per-axis box over all node offsets ± sizes); for a single
    // node it equals the node's own size (the step-2 region).
    // A placed/instanced scene (demo-scene or demo-village) resolves its whole
    // composite extent; a single-node scene uses its own block size (step-2 region).
    // The far-offset demo also resolves its full composite extent (a single 4³
    // box). `full_extent_blocks` returns the box's own size (4³) for a lone node,
    // and the resolve rebases it to the floating origin (= the composite recentre)
    // in i64 BEFORE the f32 downcast (S4b), so even at a 1_000_000-block offset
    // (16M voxels, past the f32 exact-integer ceiling) the grid is BYTE-IDENTICAL
    // to the near box at the origin — the far-lands jitter is gone (S4b proof).
    let placed_scene = options.demo_scene
        || options.demo_village
        || options.demo_groups
        || options.far_offset
        || options.far_offset_near;
    let region = if placed_scene {
        scene.full_extent_blocks(options.geometry.voxels_per_block)
    } else {
        {
            // The geometry mirror is voxel-canonical (ADR 0003 §3f(0)); the explicit
            // single-shape region is whole blocks, so round the voxel size UP to
            // whole blocks (a whole-block size divides cleanly).
            let density = options.geometry.voxels_per_block.max(1);
            let voxels = options.geometry.size_voxels;
            RegionBlocks::new([
                voxels[0].div_ceil(density),
                voxels[1].div_ceil(density),
                voxels[2].div_ceil(density),
            ])
        }
    };
    // Issue #27 S2: the old whole-region `MAX_GRID_VOXELS` total cap is now a
    // PER-CHUNK bound — a scene whose TOTAL voxel count is far beyond 6M resolves
    // fine as long as each chunk is small. Only a pathological density (one chunk's
    // voxel capacity alone exceeds the bound) is rejected.
    let density = options.geometry.voxels_per_block;
    // The headless core: the resolve store + the CLI camera (assigned once the
    // region dimensions are known, below). The CHUNKABLE path — every shape + demo
    // scene the goldens test — resolves through `AppCore::rebuild`, the SAME
    // store-backed resolve + per-chunk path the windowed app drives, so the golden
    // net now exercises the real core instead of a parallel copy (A3 keystone). The
    // density-cap and Part-only branches stay shot-specific (the windowed app never
    // produces them). `render_chunks_for_mesh` carries the per-chunk accessor
    // (chunkable path only) to the cuboid mesh build below; it borrows the store, so
    // `app_core` is left untouched until it is consumed + dropped there.
    let mut app_core = AppCore::new(voxel_worker::Store::new(), OrbitCamera::default());
    // Issue #20 S6c-1: `region_dimensions` (what the camera auto-frame, origin gizmo,
    // block lattice and fine floor grid are sized from) is read from the SCENE, not
    // by reaching into the assembled grid object. For a chunkable scene it equals
    // `grid.dimensions` exactly (the resolve sizes the grid to
    // `placed_region_dimensions`, proven in
    // `scene::tests::placed_region_dimensions_equals_assembled_grid`); for the
    // chunkable path it comes straight off `AppCore::rebuild`'s output (no recompute).
    // A Part-only scene (`--shape debug-clouds`) has no composite extent, so it is
    // resolved through the explicit-region path and sized `region × density` (rather
    // than `placed_region_dimensions`, which is `[0,0,0]` for it).
    let (grid, region_dimensions, mut render_chunks_for_mesh) =
        if voxel_worker::voxel::chunk_extent_exceeds_bound(density) {
            let chunk_extent = (voxel_worker::core_geom::CHUNK_BLOCKS * density.max(1)) as u64;
            let chunk_voxels = chunk_extent * chunk_extent * chunk_extent;
            panel_state.voxel_cap_warning_millions = Some(chunk_voxels as f32 / 1_000_000.0);
            eprintln!(
                "3D paused — one chunk is {:.1}M voxels, exceeding the per-chunk bound; \
                 rendering empty grid",
                chunk_voxels as f32 / 1_000_000.0
            );
            // Render an EMPTY grid + (below) an empty mesh — NO resolve at the
            // pathological density. This matches the windowed app, where
            // `AppCore::rebuild` returns `DensityRejected` and resolves nothing; it
            // also makes shot's mesh consistent with the empty fog grid + the
            // "rendering empty grid" message (the pre-A3 cuboid path keyed on
            // `has_chunkable_extent` and resolved per-chunk anyway, risking the very
            // huge-allocation/hang the cap exists to prevent).
            let grid = VoxelGrid::new([
                region.size_blocks[0] * density,
                region.size_blocks[1] * density,
                region.size_blocks[2] * density,
            ]);
            let region_dimensions = if scene.has_chunkable_extent(density) {
                scene.placed_region_dimensions(density)
            } else {
                grid.dimensions
            };
            (grid, region_dimensions, None)
        } else if scene.has_chunkable_extent(density) {
            // Route the resolve through `AppCore::rebuild` (issue #27 S2/S3): the store
            // lazily resolves each covering chunk and reassembles the SAME recentred
            // grid the windowed app's rebuild produces, then hands back the region
            // dimensions + the per-chunk render accessor — byte-identical to shot's
            // former separate-cache path. (It resolves the scene's full composite
            // extent, which for a single zero-offset shape equals `region`, so
            // single-shape goldens are unchanged.)
            match app_core.rebuild(&scene, density) {
                RebuildOutcome::Built(RebuildOutput {
                    grid,
                    region_dimensions,
                    render_chunks,
                    // The recentre shift compensates the WINDOWED camera across edits;
                    // `shot` sets its camera per-capture from CLI flags, so it ignores it.
                    recentre_shift_voxels: _,
                }) => (grid, region_dimensions, Some(render_chunks)),
                // Unreachable: the density-cap branch above already handled the only
                // rejection case (one chunk exceeding the per-chunk voxel bound).
                RebuildOutcome::DensityRejected { .. } => {
                    unreachable!("chunk_extent_exceeds_bound was false, so rebuild cannot reject")
                }
            }
        } else {
            // A Part-only scene (e.g. `--shape debug-clouds`) has no intrinsic-size
            // leaf, so there is no composite AABB to chunk — the cloud field sizes
            // itself to the EXPLICIT region. Resolve it directly through the monolithic
            // path, exactly as before (unchanged output). The windowed app never
            // produces a Part-only scene.
            let grid = scene.resolve_region(region, density, 0);
            let region_dimensions = [
                region.size_blocks[0] * density,
                region.size_blocks[1] * density,
                region.size_blocks[2] * density,
            ];
            (grid, region_dimensions, None)
        };
    // The voxel-space grid dimensions actually resolved (the composite region for
    // a placed scene), used for the layer track and the uniforms / fog.
    let grid_dimensions = grid.dimensions;
    debug_assert_eq!(
        region_dimensions, grid_dimensions,
        "S6c-1: scene region dimensions must equal the assembled grid the consumers used"
    );
    if options.replay_path.is_some() {
        println!(
            "resolved {} voxels for --replay ({} top-level node(s), {} definition(s), {} point(s))",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            scene.points.len(),
        );
    } else if options.far_offset || options.far_offset_near {
        println!(
            "resolved {} voxels for demo-far-offset ({}, offset {:?} blocks, region {:?} blocks) \
             — S4b: the resolve rebases to the floating origin in i64 before the f32 downcast, so \
             the far box renders BYTE-IDENTICAL to the near box (no far-lands jitter)",
            grid.occupied_count(),
            if options.far_offset { "far" } else { "near" },
            if options.far_offset { FAR_OFFSET_BLOCKS } else { [0, 0, 0] },
            region.size_blocks
        );
    } else if options.demo_groups {
        println!(
            "resolved {} voxels for demo-groups ({} top-level nodes, {} definition(s), region {:?} blocks)",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            region.size_blocks
        );
    } else if options.demo_village {
        println!(
            "resolved {} voxels for demo-village ({} instances of {} definition(s), region {:?} blocks)",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            region.size_blocks
        );
    } else if options.demo_scene {
        println!(
            "resolved {} voxels for demo-scene (region {:?} blocks)",
            grid.occupied_count(),
            region.size_blocks
        );
    } else if options.debug_clouds {
        println!(
            "resolved {} voxels for DebugClouds {:?}@{}",
            grid.occupied_count(),
            shape.size_voxels,
            options.geometry.voxels_per_block
        );
    } else {
        println!(
            "resolved {} voxels for {:?} {:?}@{}",
            grid.occupied_count(),
            shape.kind,
            shape.size_voxels,
            options.geometry.voxels_per_block
        );
    }

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

    // Part of #20: the cuboid mesh path is the sole voxel renderer. Since issue #20
    // S6c-2d it meshes PER CHUNK with a 1-voxel neighbour apron: built from the
    // resolve cache's per-chunk accessor (`resident_render_chunks`) so the goldens
    // exercise the per-chunk path, falling back to the whole-grid wrapper when the
    // scene has no chunkable extent (the wrapper buckets internally → identical mesh).
    let mut cuboid_mesh_renderer = if let Some(render_chunks) = render_chunks_for_mesh.take() {
        // Chunkable path: mesh the per-chunk accessor from `AppCore::rebuild` above
        // (1-voxel neighbour apron per chunk), so the goldens exercise the real
        // per-chunk mesh path. `render_chunks` holds an immutable borrow of the store;
        // consume + drop it here, freeing `app_core` for the camera assignment below.
        let renderer = CuboidMeshRenderer::new_from_chunks(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &render_chunks,
            grid_dimensions,
        );
        drop(render_chunks);
        renderer
    } else {
        // Density-cap (empty grid) or Part-only scene: mesh the whole grid (the
        // wrapper buckets it into per-chunk sub-grids internally → identical mesh).
        CuboidMeshRenderer::new(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &grid,
            options.geometry.voxels_per_block,
        )
    };
    // Transform gizmo (issue #29 S2): when `--gizmo` is passed, place it ON the
    // active/selected node — sized to the node's own extent, positioned at its
    // recentred pivot. `None` (no selection / no extent) keeps `--gizmo` a no-op,
    // and the goldens (which never pass `--gizmo`) are unaffected.
    let gizmo_placement = if options.show_origin_gizmo {
        AppCore::gizmo_placement(&panel_state.scene, options.geometry.voxels_per_block)
    } else {
        None
    };
    let gizmo_extent_dims = gizmo_placement
        .map(|(_, extent)| {
            [
                extent[0].round().max(0.0) as u32,
                extent[1].round().max(0.0) as u32,
                extent[2].round().max(0.0) as u32,
            ]
        })
        .unwrap_or(region_dimensions);
    let transform_gizmo_renderer =
        TransformGizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT, gizmo_extent_dims);
    // Per-object block lattice + floor grid (issue #29 S3): its line batch is built
    // from the scene's grid-enabled nodes below (after the camera is known).
    let mut scene_grid_renderer = SceneGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // The world reference grid (issue #29 S5): SUPPRESSED by default (so the existing
    // goldens are byte-identical); `--points` enables it. Its batch is built below
    // from `scene.points` + the camera once the view matrix is known.
    let mut points_renderer = PointsRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // The analytic infinite reference grid (issue #29 Points fast-follow): the Points'
    // enabled planes. SUPPRESSED by default with the rest of Points; `--points` enables
    // it. Built below from `scene.points` + the camera once the view matrix is known.
    let mut infinite_grid_renderer = InfiniteGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    let view_cube_renderer = ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    let mut onion_fog_renderer = OnionFogRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // Upload the resolved grid as the fog's occupancy field. PerChunk (DEFAULT since #28
    // S5b) builds one apron'd volume per resident chunk so a scene too large for a single
    // whole-grid 3D texture still renders fog; WholeGrid (`--fog=wholegrid`, legacy, issue
    // #12) densifies one whole-grid 3D texture and disables itself past the 3D-texture limit.
    match options.fog_mode {
        FogMode::WholeGrid => {
            onion_fog_renderer.upload_grid(&gpu.device, &gpu.queue, &grid);
        }
        FogMode::PerChunk => {
            onion_fog_renderer.upload_grid_per_chunk(
                &gpu.device,
                &gpu.queue,
                &grid,
                options.geometry.voxels_per_block,
            );
            let occ = voxel_worker::build_per_chunk_fog_occupancy(
                &grid,
                options.geometry.voxels_per_block,
            );
            println!(
                "fog: per-chunk mode — {} resident chunk volume(s){}",
                occ.volumes.len(),
                if onion_fog_renderer.per_chunk_active() {
                    " (atlas active)"
                } else {
                    " (atlas EMPTY/too-large — fog disabled)"
                }
            );
        }
    }
    // Z-up: the voxel-space grid_z of the ACTUALLY resolved grid (the composite for a
    // placed scene), used for the band clip + uniforms so a demo scene that grew
    // past the single-shape `grid_z` is not clipped or mis-sized. Layers are Z-slices.
    let render_grid_z = grid_dimensions[2];
    // Issue #12: the layer-range band for the 3D clip + the measured-diameter
    // readout (widest occupied run in the active band). The demo scene always
    // renders the full band (placement, not layer-scrubbing, is what it verifies).
    let band = if placed_scene
        || (layer_range.is_full_range(render_grid_z) && !layer_range.onion_skin)
    {
        LayerBand::FULL
    } else {
        LayerBand {
            band_min: layer_range.lower,
            band_max: layer_range.upper.min(render_grid_z.saturating_sub(1)),
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
    // The render-chunk borrow from `AppCore::rebuild` was consumed + dropped at the
    // cuboid mesh build above, so `app_core` is free again — install the CLI camera
    // so the headless render sources `view_projection` from the same `AppCore` the
    // windowed shell does.
    app_core.camera = OrbitCamera {
        target: glam::Vec3::ZERO,
        orbit_theta: theta,
        orbit_phi: phi,
        orbit_distance: options
            .distance
            .unwrap_or_else(|| OrbitCamera::auto_framed_distance(region_dimensions)),
        // #13 Step 5: `--roll <radians>` twists the whole view about the view axis.
        roll: options.roll,
        projection_mode: options.projection_mode,
    };
    // Issue #25: ALL uniform uploads (camera matrix → gizmo/lattice/view-cube/fog
    // and the voxel pass) are deferred to AFTER `run_egui_frame`, because the
    // camera aspect must come from the CENTRAL 3D viewport rect (window minus the
    // side panel + bottom dock), which egui only reports once its panels are laid
    // out. The view-cube matrix is aspect-independent but uploaded alongside for
    // simplicity. `onion_active` is needed earlier for the overlays struct.
    let onion_active = layer_range.onion_skin && !options.debug_face_orientation;

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
                cuboid_mesh_renderer.material_bind_group_layout(),
                cuboid_mesh_renderer.material_sampler(),
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

    // Part of #20: synthetic loaded block — six distinct solid-colour faces built
    // in-process (no VS install). Proves the cuboid path now renders a loaded
    // per-face D2Array (layer selected by normal) and matches the instanced path
    // per face. CubeFaceSlot order: 0 +X red, 1 -X green, 2 +Y blue, 3 -Y yellow,
    // 4 +Z magenta, 5 -Z cyan.
    if options.synthetic_block {
        const FACE_SIZE: u32 = 16;
        let face_colors: [[u8; 4]; 6] = [
            [220, 40, 40, 255],   // +X red
            [40, 200, 40, 255],   // -X green
            [40, 80, 220, 255],   // +Y blue
            [230, 210, 40, 255],  // -Y yellow
            [210, 40, 210, 255],  // +Z magenta
            [40, 210, 210, 255],  // -Z cyan
        ];
        let layer_bufs: Vec<Vec<u8>> = face_colors
            .iter()
            .map(|c| c.iter().copied().cycle().take((FACE_SIZE * FACE_SIZE * 4) as usize).collect())
            .collect();
        let layers: [&[u8]; 6] = [
            &layer_bufs[0], &layer_bufs[1], &layer_bufs[2],
            &layer_bufs[3], &layer_bufs[4], &layer_bufs[5],
        ];
        let material = LoadedMaterial::from_face_layers(
            &gpu.device,
            &gpu.queue,
            cuboid_mesh_renderer.material_bind_group_layout(),
            cuboid_mesh_renderer.material_sampler(),
            FACE_SIZE,
            FACE_SIZE,
            &layers,
            "synthetic".to_string(),
        );
        println!("applied synthetic 6-face block (per_face=true)");
        panel_state.applied_block_label = Some("synthetic".to_string());
        loaded_material = Some(material);
    }

    let prepared = run_egui_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &mut panel_state,
        render_grid_z,
        measured_diameter,
        &palette,
        raw_input,
        [options.width, options.height],
        pixels_per_point,
        // #13 Step 3: the headless path never opens the ViewCube context menu.
        &mut None,
    );

    // Issue #25: now that egui has laid out its panels, derive the camera aspect
    // from the CENTRAL 3D viewport rect (window minus side panel + bottom dock) so
    // the model is centred in the visible 3D area instead of partly hidden behind
    // the side panel. Then upload every uniform that depends on the camera matrix.
    let [_, _, viewport_width, viewport_height] = prepared.viewport_px;
    let aspect_ratio = viewport_width as f32 / viewport_height.max(1) as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let gizmo_pivot = gizmo_placement
        .map(|(pivot, _)| glam::Vec3::from_array(pivot))
        .unwrap_or(glam::Vec3::ZERO);
    transform_gizmo_renderer.update_uniforms(&gpu.queue, view_projection, gizmo_pivot);
    // Build this capture's per-object grid batch from the scene's grid-enabled nodes
    // (issue #29 S3), then upload the camera matrix.
    scene_grid_renderer.rebuild_from_scene(
        &gpu.device,
        &gpu.queue,
        &panel_state.scene,
        options.geometry.voxels_per_block,
    );
    scene_grid_renderer.update_uniforms(&gpu.queue, view_projection);
    // World reference grid (issue #29 S5): build the visible Points' tiled planes +
    // axes, centred on the camera's projection onto each plane. Only wired into the
    // overlays when `--points` is passed (default OFF keeps the goldens unchanged).
    if options.show_points {
        points_renderer.rebuild_from_scene(
            &gpu.device,
            &gpu.queue,
            &panel_state.scene,
            options.geometry.voxels_per_block,
        );
        points_renderer.update_uniforms(&gpu.queue, view_projection);
        // The analytic infinite grid (issue #29 Points fast-follow): build the visible
        // Points' planes with the camera matrices (recentred frame) so the fullscreen
        // ray-plane shader can intersect each pixel's view ray with the plane.
        infinite_grid_renderer.rebuild_from_scene(
            &gpu.queue,
            &panel_state.scene,
            options.geometry.voxels_per_block,
            view_projection,
            app_core.camera.eye().to_array(),
        );
    }
    view_cube_renderer.update_uniforms(&gpu.queue, app_core.camera.view_cube_view_projection());
    if onion_active {
        onion_fog_renderer.update(
            &gpu.queue,
            AppCore::onion_fog_params(view_projection, grid_dimensions, layer_range),
        );
    }

    // Part of #20: upload the cuboid path's uniforms (camera + per-material base
    // colours + band clip) and frustum-cull its mesh chunks. A loaded VS block
    // textures the cuboid path per-face (its 6-layer D2Array is bound at draw time in
    // `render_frame`, selecting the loaded pipeline); `bound = None` here just
    // disables the procedural per-box modulation the loaded pipeline ignores.
    let bound = match &loaded_material {
        Some(_) => None,
        None => Some(options.material),
    };
    cuboid_mesh_renderer.update_uniforms(
        &gpu.device,
        &gpu.queue,
        view_projection,
        grid_dimensions,
        options.geometry.voxels_per_block,
        options.show_grid_overlay,
        bound,
        band,
        options.debug_face_orientation,
    );
    println!(
        "cuboid mesher: {} boxes → {} exposed faces ({} triangles), {} chunks",
        cuboid_mesh_renderer.box_count(),
        cuboid_mesh_renderer.face_count(),
        cuboid_mesh_renderer.triangle_count(),
        cuboid_mesh_renderer.chunk_count(),
    );

    // ADR 0002 E2 (#19): the frustum cull ran inside `update_uniforms`. Report the
    // drawn/total chunk counts so the chunking + culling are verifiable headlessly.
    if options.debug_chunks {
        println!(
            "chunks: drew {} / {} ({} boxes total)",
            cuboid_mesh_renderer.visible_chunk_count(),
            cuboid_mesh_renderer.chunk_count(),
            cuboid_mesh_renderer.box_count(),
        );
    }

    // M6: the active material is a loaded VS block when one was applied,
    // otherwise the procedural choice.
    let material = match &loaded_material {
        Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
        None => MaterialSource::Procedural(options.material),
    };

    let overlays = FrameOverlays {
        gizmo: gizmo_placement
            .is_some()
            .then_some(&transform_gizmo_renderer),
        view_cube: if options.show_view_cube {
            Some(&view_cube_renderer)
        } else {
            None
        },
        cube_hovered_zone: options.cube_hover,
        // #13 Step 6 follow-up: the four rotate arrows draw persistently when the view
        // is face-constrained. A forced `--cube-hover rotate-*` also enables them so a
        // golden can pin the arrow render even from a non-face camera.
        cube_rotate_arrows_visible: app_core.camera.is_face_constrained()
            || matches!(
                options.cube_hover,
                Some(voxel_worker::camera::CubeChromeZone::RotateArrow(_))
            ),
        scene_grid: Some(&scene_grid_renderer),
        // Issue #29 S5: Points SUPPRESSED unless `--points` (keeps the 6 goldens
        // byte-identical); the new `demo-village --points` golden enables them.
        points: options.show_points.then_some(&points_renderer),
        // Issue #29 Points fast-follow: the analytic infinite grid (Points' planes),
        // suppressed with the rest of Points unless `--points`.
        infinite_grid: options.show_points.then_some(&infinite_grid_renderer),
        onion_fog: if onion_active {
            Some(&onion_fog_renderer)
        } else {
            None
        },
        cuboid_mesh: &cuboid_mesh_renderer,
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
