//! CLI option parsing for the `shot` harness: the [`ShotOptions`] struct, its
//! defaults, the small value parsers, and the argument loop.

use std::path::PathBuf;

use voxel_worker::{
    CubeFace, GeometryParams, MaterialChoice, ProjectionMode, SdfShape, ShapeKind, ViewCubeElement,
};

pub(crate) struct ShotOptions {
    pub(crate) output_path: PathBuf,
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Geometry the panel + producer both use (so the rendered panel reflects
    /// the requested shape/size/density/wall).
    pub(crate) geometry: GeometryParams,
    /// Camera projection.
    pub(crate) projection_mode: ProjectionMode,
    /// Procedural material (Stone/Wood/Plain).
    pub(crate) material: MaterialChoice,
    /// Whether the voxel/block grid overlay is drawn.
    pub(crate) show_grid_overlay: bool,
    /// Whether `--gizmo` was passed: draw the transform gizmo ON the active/
    /// selected node (issue #29 S2). No-op-safe when nothing is selected or the
    /// selection has no extent. The field name is kept for minimal churn.
    pub(crate) show_origin_gizmo: bool,
    /// `--select-node N` (issue #29 S2): override the scene's active selection to
    /// the top-level node at index N, so a headless capture can prove the transform
    /// gizmo FOLLOWS a chosen (non-origin) node. `None` keeps the scene's default
    /// selection. Out-of-range clears the selection (gizmo hidden).
    pub(crate) select_node: Option<usize>,
    /// Whether the block lattice is drawn (M8 `--lattice`).
    pub(crate) show_block_lattice: bool,
    /// Whether the fine floor grid is drawn (M8 `--floor`).
    pub(crate) show_floor_grid: bool,
    /// Whether the world reference grid (the Points: analytic infinite ground plane
    /// plus axes) is drawn (issue #29 S5 `--points`). DEFAULT OFF so the existing
    /// goldens (which never pass `--points`) stay byte-identical; `--points` enables
    /// the Origin Point (and any others) so a deliberate Points golden can be captured.
    pub(crate) show_points: bool,
    /// An OPTIONAL extra Point at the given world BLOCK position (issue #29 Points
    /// fast-follow `--point-at X Y Z`), with its XY ground plane (Z-up) + axes ON, so a
    /// headless capture can verify a second analytic grid plane at a different height
    /// / offset. Only meaningful together with `--points`.
    pub(crate) extra_point_blocks: Option<[i64; 3]>,
    /// Whether the voxel cubes render in face-orientation debug mode
    /// (`--debug-faces`): colour by outward face normal + back-facing marker,
    /// cull off. The standard way to verify face winding/culling.
    pub(crate) debug_face_orientation: bool,
    /// When `Some`, write the resolved grid to this `.vox` path (M8
    /// `--export-vox`) instead of (or in addition to) rendering a PNG.
    pub(crate) export_vox_path: Option<PathBuf>,
    /// Whether the view cube is drawn (M5; ON by default, `--no-viewcube` hides).
    pub(crate) show_view_cube: bool,
    /// When `Some`, set the camera directly to this element's snapped angles (M5
    /// `--snap`), overriding `--theta`/`--phi` so the snap table can be verified
    /// headlessly (no tween). Faces, edges (`front-top`) and corners
    /// (`front-top-right`) are all accepted (#13a).
    pub(crate) snap_element: Option<ViewCubeElement>,
    /// `--cube-hover <zone>` (#13 Step 2): force a ViewCube chrome zone to read as
    /// hovered so a golden can show a highlighted rotate/roll arrow. `None` = the
    /// normal render (Home/Fit only, no arrows).
    pub(crate) cube_hover: Option<camera::CubeChromeZone>,
    /// Orbit azimuth (radians). Default 0.7.
    pub(crate) theta: f32,
    /// Orbit polar angle from +Y (radians). Default 1.05.
    pub(crate) phi: f32,
    /// View roll about the forward axis (radians, #13 Step 5). Default 0 (upright).
    /// `--roll <radians>` twists the whole scene AND the ViewCube together.
    pub(crate) roll: f32,
    /// Orbit distance. `None` = auto-frame from the grid.
    pub(crate) distance: Option<f32>,
    /// Run the VS auto-detect + scan synchronously before rendering (M6) so the
    /// palette dock is populated in the screenshot.
    pub(crate) scan_vs: bool,
    /// After `--scan-vs`, apply the first scanned block's texture as the active
    /// material (M6) so the model shows a real VS texture (per-voxel sliced).
    pub(crate) apply_first_block: bool,
    /// After `--scan-vs`, apply the first scanned block whose label/key matches
    /// this substring (case-insensitive), going through per-face JSON resolution
    /// (M7 `--apply-block`).
    pub(crate) apply_block_substring: Option<String>,
    /// After `--scan-vs`, print which scanned blocks resolve to genuinely
    /// distinct per-face textures (top != side), then exit (M7 `--list-perface`).
    pub(crate) list_per_face: bool,
    /// Debug escape hatch (M7 verification): resolve an arbitrary blocktype by
    /// its texture stem (e.g. `wood/treetrunk/oak`) even if it is outside the
    /// chiselable allow-list, to demonstrate per-face rendering on a known block.
    pub(crate) force_demo_stem: Option<String>,
    /// Layer-range scrubber lower bound (issue #12), a voxel Z-layer index. When
    /// `None`, defaults to the full range (0). Raw voxel index — no snapping.
    pub(crate) layer_lower: Option<u32>,
    /// Layer-range scrubber upper bound (issue #12), a voxel Z-layer index. When
    /// `None`, defaults to the full range (grid_z). Raw voxel index — no snapping.
    pub(crate) layer_upper: Option<u32>,
    /// Onion-skin depth (issue #12): 0 = off (hard band clip), N = ghost N layers
    /// on each side of the band with screen-door dither.
    pub(crate) onion_depth: u32,
    /// `--shape debug-clouds`: replace the parametric producer with the debug
    /// cloud field (several distinct billowy blobs in a mostly-empty volume) at
    /// the requested size/density. The grid dims still come from size×density.
    pub(crate) debug_clouds: bool,
    /// `--demo-scene` (ADR 0001 step 3): ignore the single-shape options and build
    /// a hardcoded multi-node PLACED scene (a sphere at the origin, a box offset
    /// +8 blocks in X, and a clouds Part offset in Z) so the headless capture can
    /// confirm nodes appear separated in space (not overlapping at the origin).
    /// Useful for future headless multi-node checks.
    pub(crate) demo_scene: bool,
    /// `--demo-village` (ADR 0001 step 4): ignore the single-shape options and
    /// build an INSTANCED scene — one "house" `AssemblyDef` placed by four
    /// `Instance` nodes at four offsets — so the headless capture can confirm the
    /// repeated assembly appears at multiple separated locations from a single
    /// stored definition (reuse by reference).
    pub(crate) demo_village: bool,
    /// `--demo-village-far` (ADR 0010 D0 / ADR 0003 §G3, Phase D0): the SAME
    /// instanced `--demo-village` scene, but its whole composite is offset to the FAR
    /// end of the anisotropic horizontal extent ([`crate::demos::FAR_SCENE_BASE_BLOCKS`] ≈
    /// XZ 10,000 blocks, vertical Y bounded at 0). The near goldens cannot see
    /// far-scene f32 precision loss — at XZ~10k the absolute voxel centre has barely a
    /// mantissa bit below the voxel — so this golden establishes the far-scene baseline
    /// the §3a chunk-local-integer payload move (#48) must preserve. It renders crisp
    /// TODAY because `resolve_chunk_rebased` subtracts the floating origin in i64 BEFORE
    /// the f32 downcast (S4b); a regression of that rebase would smear this golden.
    /// Overrides --shape/--size/--density.
    pub(crate) demo_village_far: bool,
    /// `--debug-chunks` (ADR 0002 E2, part of #19): after the per-frame frustum
    /// cull, print `chunks: drew X / Y` (visible / total) so the chunking + cull
    /// can be verified headlessly. Zooming/rotating a large scene off-screen draws
    /// fewer chunks; a small scene draws all of them.
    pub(crate) debug_chunks: bool,
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
    pub(crate) far_offset: bool,
    /// `--demo-far-offset-near` (ADR 0002 streaming S1): the SAME small box as
    /// `--demo-far-offset` but placed at the ORIGIN (a block offset of [0, 0, 0]),
    /// for A/B comparison against the far render. Overrides --shape/--size/--density.
    pub(crate) far_offset_near: bool,
    /// `--demo-sketch-extrude` (ADR 0003 §3i Slice 2a): build a scene containing a
    /// single **sketch → extrude** producer with a RECOGNIZABLE non-box (L-shaped)
    /// footprint extruded up, so the headless capture confirms the new producer
    /// resolves + renders through the same pipeline as `SdfShape`. Overrides
    /// --shape/--size/--density.
    pub(crate) demo_sketch_extrude: bool,
    /// `--demo-sketch-revolve` (ADR 0003 §3i): build a scene containing a single
    /// **sketch → revolve** producer — a stepped (vase-like) radial profile revolved a
    /// full 360° about the vertical Z axis into a solid of revolution, so the headless
    /// capture confirms the revolve producer resolves + renders through the same
    /// pipeline as `SdfShape`. Overrides --shape/--size/--density.
    pub(crate) demo_sketch_revolve: bool,
    /// `--demo-sketch-box <edge_voxels>`: a solid cube of the given voxel edge, built
    /// via the SKETCH EXTRUDE path (a square profile extruded its own edge) — the
    /// large-scene fixture for the two-layer / brick display + fog at scale. `Some(N)`
    /// overrides --shape/--size/--density-shape (density still sets the voxel grain).
    pub(crate) demo_sketch_box: Option<i64>,
    /// `--demo-groups` (ADR 0001 step 4, UI verification): build a scene with a
    /// top-level `Group` that has two child Tools, plus a sibling top-level Tool
    /// and one `Instance` of a definition — so the headless PANEL capture shows the
    /// INDENTED TREE (a Group with its children nested under it) and the
    /// Definitions list. Overrides --shape/--size/--density.
    pub(crate) demo_groups: bool,
    /// `--synthetic-block` (part of #20 verification): build a LoadedMaterial from
    /// SIX distinct solid-colour faces in-process (no VS install needed) and apply
    /// it as the active material. Lets the headless harness prove the cuboid path
    /// now renders a loaded per-face D2Array (and that cuboid vs instanced match per
    /// face). Overrides --scan-vs/--apply-block material selection.
    pub(crate) synthetic_block: bool,
    /// `--demo-overlap` (ADR 0010 E3 / #50): two solid boxes of DIFFERENT materials that
    /// OVERLAP, so the overlap region resolves last-writer-wins (document order). The golden
    /// pins that an overlapping multi-material scene renders identically on the dense and
    /// two-layer paths (the E2 carry-over). Overrides --shape/--size/--density.
    pub(crate) demo_overlap: bool,
    /// `--demo-subtract` (ADR 0017 / #73): a solid Stone box carved by a smaller
    /// Subtract box placed AFTER it (the ordered document-order fold) — a crisp cubic
    /// notch bitten out of the corner, its newly-exposed faces still STONE (a Subtract
    /// never stamps material). The CSG tracer-bullet golden. Overrides
    /// --shape/--size/--density.
    pub(crate) demo_subtract: bool,
    /// `--demo-group-subtract` (ADR 0017 Decision 3 / #74): a Group holding a Stone body
    /// plus a Subtract cutter (a corner notch carved INSIDE the group), with a Wood
    /// bystander box BEFORE the group overlapping the cutter's volume — rendered intact,
    /// the visible proof that a boolean inside a sealed scope cannot escape it. Overrides
    /// --shape/--size/--density.
    pub(crate) demo_group_subtract: bool,
    /// `--demo-intersect` (ADR 0017 / #75): a Stone body box and an overlapping Intersect
    /// mask box placed AFTER it (the ordered document-order fold) — only the overlap
    /// volume survives, rendered STONE (an Intersect keeps the ACCUMULATED material and
    /// never stamps its own). The intersect golden. Overrides --shape/--size/--density.
    pub(crate) demo_intersect: bool,
    /// `--demo-cutter-def` (ADR 0017 / #76): ONE cutter definition placed by TWO Instance
    /// nodes under Subtract, each carving its own separated Stone host's corner — two
    /// identical notches from a single stored definition (the reusable cutter). The def
    /// body's Wood material appears nowhere (a Subtract instance is an occupancy-only
    /// mask). Overrides --shape/--size/--density.
    pub(crate) demo_cutter_def: bool,
    /// `--demo-window-fixture` (ADR 0017 Decision 4 / #77): a Stone wall plus ONE
    /// placement of a FIXTURE definition [opening cutter Subtract, Wood frame Union]
    /// whose children splice into the wall's scope at the instance's position — the
    /// hole is cut AND the frame filled by a single Instance node (the window
    /// golden). The instance's own operation is inert.
    pub(crate) demo_window_fixture: bool,
    /// `--demo-two-material` (ADR 0011 G2): two solid boxes of DISTINCT materials placed
    /// SEPARATED so no block is shared — every rendered block is single-material, the
    /// brick-representable multi-producer scene the G2 per-record-material golden locks
    /// (brick == mesh). Unlike `--demo-overlap` (which mixes materials INSIDE a block and
    /// correctly stays on the mesh path), this engages the brick sink. Overrides
    /// --shape/--size/--density.
    pub(crate) demo_two_material: bool,
    /// `--demo-mixed-material` (material atlas / ADR 0013): two DISTINCT-material boxes whose
    /// second is offset a SUB-BLOCK voxel amount, so a straddling block MIXES both materials —
    /// the case the deleted representability gate routed to the mesh. Now engages the brick sink
    /// (per-voxel cell-key shading). Overrides --shape/--size/--density.
    pub(crate) demo_mixed_material: bool,
    /// `--two-layer` (ADR 0010 E3 / #50): render the voxel mesh THROUGH the two-layer
    /// path — build each covering chunk's [`evaluation::two_layer_store::TwoLayerChunk`]
    /// (coarse one-box + microblock cuboids + seam-solidity flags) and mesh from it via
    /// [`voxel_worker::CuboidMeshRenderer::new_from_two_layer_chunks`], instead of the dense per-chunk
    /// `VoxelGrid`. PROVES the two-layer mesher renders pixel-identical to the dense path
    /// (the E3 golden gate). DEFAULT OFF — the live renderer stays on the dense path until
    /// E5. Only the voxel MESH source changes; fog / overlays / export are unaffected.
    pub(crate) two_layer: bool,
    /// `--brick` (ADR 0011 G1): source the voxel display from the **brick raymarch**
    /// instead of the CPU cuboid mesh — build the two-layer boundary set, pack it into
    /// the G0 brick field (sorted records + R8 sculpted atlas) and render via the
    /// fullscreen block-DDA pass. Engages only under `--features gpu` for a chunkable
    /// single-producer scene with a uniform render cell (the G1 gate); otherwise it
    /// prints why and falls back to the mesh path. The mesh renderer is built EMPTY
    /// when bricks engage, so the PNG provably comes from the brick atlas.
    pub(crate) brick: bool,
    /// `--brick-force-miss` (implies `--brick`): upload EVERY sculpted record with the
    /// non-resident atlas-slot sentinel, forcing the residency-miss contract's coarse
    /// fallback — the degraded-but-correct all-block-cubes render, for visual checks.
    pub(crate) brick_force_miss: bool,
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
    pub(crate) replay_path: Option<PathBuf>,
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
            debug_clouds: false,
            debug_chunks: false,
            demo_scene: false,
            demo_village: false,
            demo_village_far: false,
            demo_sketch_extrude: false,
            demo_sketch_revolve: false,
            demo_sketch_box: None,
            demo_groups: false,
            far_offset: false,
            far_offset_near: false,
            synthetic_block: false,
            demo_overlap: false,
            demo_subtract: false,
            demo_group_subtract: false,
            demo_intersect: false,
            demo_cutter_def: false,
            demo_window_fixture: false,
            demo_two_material: false,
            demo_mixed_material: false,
            two_layer: false,
            brick: false,
            brick_force_miss: false,
            replay_path: None,
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

/// Parse a `--cube-hover` value (#13 Step 2) into the forced hovered chrome zone.
/// Accepts the rotate/roll arrows and the Home/Fit badges so a golden can show
/// any highlighted chrome element.
fn parse_cube_hover(value: &str) -> camera::CubeChromeZone {
    use camera::{ArrowDir, CubeChromeZone, RollDir};
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

pub(crate) fn parse_options() -> ShotOptions {
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
            "--demo-village-far" => {
                options.demo_village_far = true;
            }
            "--demo-sketch-extrude" => {
                options.demo_sketch_extrude = true;
            }
            "--demo-sketch-revolve" => {
                options.demo_sketch_revolve = true;
            }
            "--demo-sketch-box" => {
                options.demo_sketch_box = Some(
                    args.next()
                        .expect("--demo-sketch-box requires an edge length in voxels")
                        .parse()
                        .expect("--demo-sketch-box must be a positive integer"),
                );
            }
            "--demo-groups" => {
                options.demo_groups = true;
            }
            "--synthetic-block" => {
                options.synthetic_block = true;
            }
            "--two-layer" => {
                options.two_layer = true;
            }
            "--brick" => {
                options.brick = true;
            }
            "--brick-force-miss" => {
                options.brick = true;
                options.brick_force_miss = true;
            }
            "--demo-overlap" => {
                options.demo_overlap = true;
            }
            "--demo-subtract" => {
                options.demo_subtract = true;
            }
            "--demo-group-subtract" => {
                options.demo_group_subtract = true;
            }
            "--demo-intersect" => {
                options.demo_intersect = true;
            }
            "--demo-cutter-def" => {
                options.demo_cutter_def = true;
            }
            "--demo-window-fixture" => {
                options.demo_window_fixture = true;
            }
            "--demo-two-material" => {
                options.demo_two_material = true;
            }
            "--demo-mixed-material" => {
                options.demo_mixed_material = true;
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
                     \x20            [--synthetic-block] [--two-layer]\n\
                     \x20            [--replay <script.jsonl>]\n\
                     \x20            [--force-demo-stem <texture/stem>]\n\
                     \x20            [--gizmo] [--select-node <usize>] [--lattice] [--floor] [--points] [--point-at <X Y Z>] [--no-viewcube]\n\
                     \x20            [--debug-faces] [--debug-chunks]\n\
                     \x20            [--demo-scene] [--demo-overlap] [--demo-subtract] [--demo-group-subtract] [--demo-intersect] [--demo-cutter-def] [--demo-window-fixture] [--demo-two-material] [--demo-village] [--demo-village-far] [--demo-groups]\n\
                     \x20            [--demo-sketch-extrude] [--demo-sketch-revolve]\n\
                     \x20            [--demo-far-offset] [--demo-far-offset-near]\n\
                     \x20            [--layer-lower <u32>] [--layer-upper <u32>] [--onion <u32>]\n\
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
                     \x20  --demo-village-far the SAME instanced village, but its whole\n\
                     \x20                composite is offset to ~XZ 10,000 blocks (vertical\n\
                     \x20                bounded) — the far-scene golden baseline the §3a\n\
                     \x20                chunk-local payload move must preserve (ADR 0010 D0).\n\
                     \x20                Overrides --shape/--size/--density.\n\
                     \x20  --demo-groups build a scene with a top-level Group (2 child\n\
                     \x20                Tools), a sibling Tool and an Instance, so the\n\
                     \x20                captured PANEL shows the indented tree + Definitions\n\
                     \x20                (ADR 0001 step 4 UI). Overrides --shape/--size/--density.\n\
                     \x20  --demo-sketch-extrude build a single 2D-sketch→extrude producer\n\
                     \x20                with an L-shaped footprint extruded up (ADR 0003 §3i\n\
                     \x20                Slice 2a) — a non-box a primitive can't make. Overrides\n\
                     \x20                --shape/--size/--density.\n\
                     \x20  --demo-sketch-revolve build a single 2D-sketch→revolve producer:\n\
                     \x20                a stepped (vase) radial profile revolved 360° about\n\
                     \x20                +Z into a solid of revolution (ADR 0003 §3i). Overrides\n\
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
