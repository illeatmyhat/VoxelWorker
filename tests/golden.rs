//! Golden-image regression harness (issue #24) — the **E0 safety net** for the
//! upcoming engine/renderer rewrite (ADR 0002).
//!
//! Each canonical case renders through the REAL `shot` binary (located via the
//! `CARGO_BIN_EXE_shot` env var Cargo sets for integration tests — it auto-builds
//! the binary) into a temp PNG at a fixed `--width 1280 --height 720` and a fixed
//! camera, then compares the result against a committed reference under
//! `tests/golden/`. When the cuboid mesher replaces the current renderer, these
//! goldens prove the pixels did not change (or, if they intentionally did, the
//! references are refreshed in one step).
//!
//! Run:    `cargo test --features gpu --test golden`
//! Regen:  `UPDATE_GOLDENS=1 cargo test --features gpu --test golden`
//!         (writes the reference PNGs instead of comparing — use after an
//!         intended visual change, then VISUALLY sanity-check each PNG.)
//!
//! GPU-gated (`#![cfg(feature = "gpu")]`) so the GPU-less CI runner skips it; it
//! runs locally and during the renderer rewrite where a real GPU is present.
//!
//! ## Tolerance model
//! GPU rasterisation + MSAA resolve is not bit-exact across runs, so an exact
//! compare would flake. Instead we count a pixel as "different" only when its
//! max per-channel absolute difference exceeds `CHANNEL_DIFF_THRESHOLD` (8/255),
//! and FAIL only when the fraction of such pixels exceeds `MAX_MISMATCH_FRACTION`
//! (0.5%). On this RTX machine the `--debug-faces` case (flat orientation colours)
//! comes out essentially bit-exact, and the shaded cases sit far below 0.5% across
//! repeated runs (observed << 0.1%). On a mismatch we write `<case>-actual.png` and
//! `<case>-diff.png` next to the reference dir's sibling temp output and print the
//! mismatch fraction so a regression is debuggable.
#![cfg(feature = "gpu")]

use std::path::{Path, PathBuf};
use std::process::Command;

use image::RgbaImage;

/// A pixel counts as different when its largest per-channel absolute difference
/// (R/G/B/A) exceeds this many 8-bit levels. Absorbs sub-threshold AA/float jitter.
const CHANNEL_DIFF_THRESHOLD: u8 = 8;

/// The test FAILS when the fraction of differing pixels exceeds this. 0.5% leaves
/// generous headroom over the observed run-to-run jitter on this machine while
/// still catching any real rendering change (which moves whole regions of pixels).
const MAX_MISMATCH_FRACTION: f64 = 0.005;

/// Fixed capture size for every case — small enough to keep the references tiny,
/// large enough that shape silhouettes are unambiguous.
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

/// A canonical golden case: a stable name (→ `tests/golden/<name>.png`) and the
/// shot CLI args that produce it. The width/height/camera are appended uniformly
/// so every case is deterministic and comparable across the renderer rewrite.
struct GoldenCase {
    name: &'static str,
    args: &'static [&'static str],
}

/// The canonical cases. Kept small (6) and chosen to exercise distinct paths:
/// flat face-orientation debug (most deterministic), a default-material shaded
/// solid, a non-trivial SDF (torus), the instanced scene graph (village), the
/// debug cloud field, and (since #28 S5b) per-chunk onion fog.
const CASES: &[GoldenCase] = &[
    GoldenCase {
        name: "sphere-debug-faces",
        args: &["--shape", "sphere", "--debug-faces"],
    },
    GoldenCase {
        name: "cylinder",
        args: &["--shape", "cylinder"],
    },
    GoldenCase {
        name: "torus",
        args: &[
            "--shape", "torus", "--size-x", "8", "--size-y", "2", "--size-z", "8",
        ],
    },
    GoldenCase {
        name: "demo-village",
        args: &["--demo-village"],
    },
    // ADR 0010 D0 (ADR 0003 §G3, Phase D0): the FAR-SCENE baseline. The SAME instanced
    // village, but its whole composite is placed at ~XZ 10,000 blocks (vertical bounded;
    // Z-up, so the far offset is on the two horizontal axes X/Y). Every other golden is
    // near-origin, where the f32 voxel payload is still exact — they cannot see far-scene
    // precision loss. At XZ~10k an absolute f32 voxel centre has barely a fractional bit
    // left, so this golden is the guard the ADR 0003 §3a chunk-local-integer payload move
    // (#48) must preserve. It renders pixel-identical to `demo-village` TODAY because the
    // resolve rebases to the composite floating-origin in i64 BEFORE the f32 downcast
    // (S4b); a regression of that rebase would smear or speckle this golden while leaving
    // the near goldens untouched.
    GoldenCase {
        name: "demo-village-far",
        args: &["--demo-village-far"],
    },
    GoldenCase {
        name: "debug-clouds",
        args: &[
            "--shape",
            "debug-clouds",
            "--size-x",
            "64",
            "--size-y",
            "64",
            "--size-z",
            "64",
            "--density",
            "2",
        ],
    },
    // ADR 0012 (H1): the onion GHOST pass — replaces the retired volumetric fog golden.
    // An 8³-block sphere (grid 128³) with an onion-skinned equatorial band: layers [56,72]
    // render as the crisp solid stone disk, while the sphere's shell ABOVE and BELOW the
    // band ghosts as crisp TRANSLUCENT voxels (8 onion layers each side, the retired fog
    // haze's blue/grey hue). The ghost is two thin per-slab meshes (cuboid path) / two
    // per-slab raymarches (brick path), alpha-blended, depth test `Less` + write ON (nearest
    // ghost surface). This golden pins the DENSE mesh ghost (draws in the onion slabs, solid
    // band unfogged). The translucent ghost is NOT pixel-identical across display paths (the
    // underlying solid's per-path shading shows through the ghost), so it is deliberately
    // EXCLUDED from the two-layer + brick cross-checks — the brick ghost is gated separately by
    // `onion_ghost_marches_only_the_onion_slabs` in `tests/gpu_parity.rs`.
    // ADR 0018 (#84): onion fog is now a VIEWER MODE with a per-object region-scoped clip.
    // The band bites only in `--view-mode onion` with a selection; selecting the sole object
    // (`--select-node 0`) scopes the region to it — which, being the whole scene, makes the
    // ConfineBand clip + ghost slabs identical to the pre-0018 scene-wide band, so this renders
    // pixel-identically to the pre-0018 `onion-ghost` reference (viewport AND panel — the sphere
    // stays the selected/inspected node). The mode gate is what keeps the band alive here
    // (Normal mode would render the full sphere — see `normal-ignores-band`).
    GoldenCase {
        name: "onion-ghost",
        args: &[
            "--view-mode", "onion", "--select-node", "0",
            "--shape", "sphere", "--size-x", "8", "--size-y", "8", "--size-z", "8",
            "--onion", "8", "--layer-lower", "56", "--layer-upper", "72",
        ],
    },
    // Issue #29 S5: the world reference grid (Points). The same instanced village,
    // now with the Origin Point's camera-relative tiled GROUND plane + axis lines on
    // (`--points` enables Points, suppressed by default so the other goldens are
    // unchanged). The ground plane is subtle (low base alpha, fading toward the rim
    // with no hard finite edge), draws BOLD block-cell lines over the dimmer per-block
    // lines, and is DEPTH-TESTED so the four houses occlude it where they sit in front.
    // The origin axes (X/Y/Z) read through the first house. This pins the Point render
    // path: tiled plane + fade + two-tier block lines + depth occlusion + axes.
    GoldenCase {
        name: "demo-village-points",
        args: &["--demo-village", "--points"],
    },
    // #13 Step 2: the ViewCube chrome with a hover. The same village, but the cube
    // corner now carries the always-on Home/Fit glyphs AND a HIGHLIGHTED rotate-left
    // arrow (forced via `--cube-hover rotate-left`). Pins the screen-space chrome
    // overlay path: the glyph quads sit on the Step-1 hit zones, the hovered arrow
    // brightens, and the 3D viewport/panel are untouched.
    GoldenCase {
        name: "cube-chrome-hover",
        args: &["--demo-village", "--cube-hover", "rotate-left"],
    },
    // #13 Step 5: the real roll DOF. The same instanced village, rolled a quarter
    // turn (`--roll-quarters 1` = +π/2) about the view axis. The WHOLE view twists
    // 90° — the house row and the small ViewCube rotate together (the cube's TOP
    // label now points sideways). Pins the roll path: `up_vector` folds roll on top
    // of the pole-aware base up, and BOTH the scene and the ViewCube route through
    // the rolled up so they stay in lockstep. roll=0 (every other golden) is
    // byte-identical to before, proving the fold is a no-op at the default.
    GoldenCase {
        name: "roll-quarter",
        args: &["--demo-village", "--roll-quarters", "1"],
    },
    // ADR 0003 §3i (revolve commit 4): the sketch→revolve render path. A stepped
    // (vase) radial profile revolved a full 360° about the vertical Z axis into a
    // solid of revolution — a round, axially-symmetric body with a foot, a pinched
    // waist and a flared lip that a box / extrude cannot produce. Pins the revolve
    // producer resolving + rendering through the SAME cuboid/instanced pipeline as
    // SdfShape at the fixed golden camera.
    // The revolve was IMPLICITLY band-clipped pre-0018 (its composite grid_z 128 exceeded the
    // default layer-track's 80, so the top third clipped scene-wide). ADR 0018 (#84) retired
    // the scene-wide band: the clip now needs `--view-mode onion` to bite. Selecting the sole
    // revolve node (`--select-node 0`) scopes the (default [0,80]) hard band to it — the whole
    // scene — reproducing the pre-0018 clipped image pixel-for-pixel (viewport AND panel), so
    // this stays the band-clipped SketchSolid case in the two-layer / brick cross-checks.
    GoldenCase {
        name: "sketch-revolve-dome",
        args: &["--demo-sketch-revolve", "--view-mode", "onion", "--select-node", "0"],
    },
    // ADR 0010 E3 (#50): a sketch→extrude (L-footprint) solid — a SketchSolid producer that
    // is NOT band-clipped (its 3-block extrusion fits under the layer-track grid_z), the
    // non-clipped SketchSolid case in the two-layer cross-check. (The revolve golden IS
    // band-clipped via the layer-track's default grid_z; ADR 0010 #53 taught the two-layer
    // path to reclip, so BOTH are now in TWO_LAYER_CASE_NAMES.)
    GoldenCase {
        name: "sketch-extrude-l",
        args: &["--demo-sketch-extrude"],
    },
    // ADR 0010 E3 (#50): an overlapping multi-material scene — two solid boxes of different
    // materials whose corner volumes overlap (the overlap resolves last-writer-wins by
    // document order). Pins that an OVERLAP region renders identically; the two-layer
    // cross-check (`two_layer_golden_matches_dense`) re-renders it through the two-layer
    // path and asserts pixel-identity to THIS dense reference (the E2 carry-over).
    GoldenCase {
        name: "demo-overlap",
        args: &["--demo-overlap"],
    },
    // ADR 0017 (#73): the CSG tracer bullet — a solid Stone box carved by a smaller box
    // placed AFTER it under CombineOp::Subtract (the ordered document-order fold). The
    // render shows a crisp cubic notch bitten out of the box's corner, and the cutter's
    // own material (Wood) never appears: a Subtract is an occupancy-only mask, so every
    // newly-exposed face inside the notch renders STONE. Pins the whole subtract path:
    // document walk → conservative re-classification (coarse-solid corner blocks become
    // boundary/air) → per-voxel boundary resolve → mesh.
    GoldenCase {
        name: "demo-subtract",
        args: &["--demo-subtract"],
    },
    // ADR 0017 Decision 3 (#74): the SEALED-SCOPE golden — a Group holds a Stone body
    // carved by a Subtract cutter, and a Wood bystander box placed BEFORE the group
    // overlaps the cutter's volume. Under a flat (unsealed) fold the cutter — later in
    // depth-first order — would carve the bystander; here it renders INTACT, nestled
    // into the notch: the visible proof that a boolean inside a scope cannot affect
    // geometry outside it. The cutter's Plain material appears nowhere (a Subtract
    // never stamps; the notch faces render Stone).
    GoldenCase {
        name: "demo-group-subtract",
        args: &["--demo-group-subtract"],
    },
    // ADR 0017 (#75): the INTERSECT golden — a Stone body box and an overlapping box
    // placed AFTER it under CombineOp::Intersect. Exactly the overlap volume survives (a
    // 2³-block cube where the boxes met), and the mask's own material (Wood) never
    // appears: an Intersect is an occupancy-only mask, so the surviving cube renders
    // STONE. Pins the whole intersect path: document walk → conservative bound algebra
    // (blocks outside the mask re-classify to air, mask-grazed blocks degrade to
    // boundary) → per-voxel boundary resolve → mesh.
    GoldenCase {
        name: "demo-intersect",
        args: &["--demo-intersect"],
    },
    // ADR 0017 (#76): the REUSABLE CUTTER golden — ONE "corner cutter" definition placed
    // by TWO Instance nodes under CombineOp::Subtract, each carving its own separated
    // Stone host's top corner octant. Two identical notches from a single stored
    // definition is the visible proof of reuse-by-reference cutters (the sealed def body
    // pre-composes, then each instance folds it as a carve at its own transform); the def
    // body's Wood material appears nowhere (a Subtract instance never stamps — every
    // notch face renders Stone).
    GoldenCase {
        name: "demo-cutter-def",
        args: &["--demo-cutter-def"],
    },
    // ADR 0017 Decision 4 (#77): THE WINDOW golden — a Stone wall and ONE placement
    // of a FIXTURE definition [opening cutter Subtract, Wood frame Union]. The def
    // does not pre-compose: its children splice into the wall's scope at the
    // instance's position, so the single placement both CUTS the 3×3-block opening
    // through the wall AND FILLS the Wood frame bar along its bottom (daylight
    // through the hole above a Wood sill). The cutter's Plain material appears
    // nowhere (a spliced Subtract never stamps) and the instance's own operation is
    // inert. This is the epic's finale golden: cut + fill from one Instance node.
    GoldenCase {
        name: "demo-window-fixture",
        args: &["--demo-window-fixture"],
    },
    // ADR 0018 Decision 6: the BURIED-CUTTER golden — a Subtract cutter entirely inside a
    // Stone host (an internal void invisible by success), with the CUTTER selected, in
    // Show-booleans mode. The boolean-operand ghost renders the cutter's whole body in the
    // LOUD occluded red (depth test `Greater` — every ghost fragment is behind the host's
    // surface), so the invisible void x-rays through the unbroken box. Deliberately more
    // obvious than Fusion's treatment of internal voids (the owner's call).
    GoldenCase {
        name: "demo-buried-cutter",
        args: &["--demo-buried-cutter", "--view-mode", "booleans"],
    },
    // ADR 0018 Decision 6: the CORNER-CUTTER golden — the demo-subtract scene with the
    // CUTTER selected (`--select-node 1`), in Show-booleans mode. The cutter's exposed
    // carve faces COINCIDE with the notch's cut surface — the delicate half of the depth
    // split: another mesher's triangulation of the same plane must still classify QUIET
    // (depth `LessEqual` + the shared toward-viewer bias), never loud and never dropped.
    // The whole notch is camera-visible here, so the ghost is all-quiet; the loud half is
    // pinned by demo-buried-cutter above and the window case below.
    GoldenCase {
        name: "demo-subtract-cutter-selected",
        args: &["--demo-subtract", "--select-node", "1", "--view-mode", "booleans"],
    },
    // ADR 0018 Decision 6: the FIXTURE-SELECTION golden — the window scene with the window
    // INSTANCE selected, in Show-booleans mode. Its own operation is inert (ADR 0017
    // Decision 4), so the walk splices its children: only the opening cutter ghosts (red,
    // QUIET on the opening's exposed carve faces, LOUD where the wall thickness / the
    // later-placed frame bury it — both halves of the depth split in one image). The Union
    // frame is already visible and never ghosts (the retired #78 union tint).
    GoldenCase {
        name: "demo-window-fixture-selected",
        args: &["--demo-window-fixture", "--select-node", "1", "--view-mode", "booleans"],
    },
    // ADR 0018 Decision 6: the INTERSECT-mask ghost — the demo-intersect scene with the
    // MASK selected, in Show-booleans mode. The mask's body ghosts AMBER: quiet over the
    // empty space the fold cleared (nothing occludes it there), loud where the surviving
    // Stone cube buries it. Also pins that Intersect never renders the Subtract red.
    GoldenCase {
        name: "demo-intersect-mask-selected",
        args: &["--demo-intersect", "--select-node", "1", "--view-mode", "booleans"],
    },
    // ADR 0018 Decision 6: the ROOT-PART master — a Group whose Stone body carries an
    // exposed corner cutter AND a strictly-interior buried cutter, with the ROOT PART
    // selected in Show-booleans mode (`--select-root --view-mode booleans`). Selecting the
    // root x-rays EVERY boolean in the whole scene: both cutters render as operand ghosts
    // (red; the corner cutter's exposed carve faces quiet, its walled-off remainder and
    // the whole buried cutter loud). The scene-wide master ask (the #79 deferral). No union
    // tint appears anywhere — the mode ghosts only the invisible-by-success boolean masks.
    GoldenCase {
        name: "demo-booleans-root",
        args: &["--demo-child-booleans", "--select-root", "--view-mode", "booleans"],
    },
    // ADR 0018 Decision 4: the SAME scene in NORMAL mode — the finished carved look with
    // ZERO ghosts. Pins that Normal renders no overlay regardless of selection (the mode,
    // not a per-node flag, is what separates this from the master case above).
    GoldenCase {
        name: "demo-child-booleans-normal",
        args: &["--demo-child-booleans", "--view-mode", "normal"],
    },
    // ADR 0018 Decision 5 (#84): the REGION-SCOPING proof. A three-object scene (Sphere at the
    // origin, Box at +8 blocks X, Torus at +6 blocks Z) in `--view-mode onion` with ONLY the
    // Sphere selected (`--select-node 0`). The Sphere clips to its mid-band [30,50] of its own
    // 80-layer Z track with the ghost haze above/below — INSIDE its placed AABB only; the Box
    // and Torus, outside that AABB, render FULLY SOLID/finished. This is what distinguishes the
    // per-object clip from the retired scene-wide band: two untouched neighbours beside a
    // sectioned object.
    GoldenCase {
        name: "onion-region-two-object",
        args: &[
            "--demo-scene", "--view-mode", "onion", "--select-node", "0",
            "--onion", "8", "--layer-lower", "30", "--layer-upper", "50",
        ],
    },
    // ADR 0018 Decision 4 (#84): Normal mode IGNORES the layer band. A sphere with a NARROW
    // band ([56,72]) but `--view-mode normal` renders the FULL finished sphere — the band is
    // Onion-fog's tool alone and does not clip here (contrast `onion-ghost`, the same-shape
    // band alive under `--view-mode onion --select-root`).
    GoldenCase {
        name: "normal-ignores-band",
        args: &[
            "--view-mode", "normal",
            "--shape", "sphere", "--size-x", "8", "--size-y", "8", "--size-z", "8",
            "--layer-lower", "56", "--layer-upper", "72",
        ],
    },
];

/// The subset of [`CASES`] whose scene is CHUNKABLE (has an intrinsic-size leaf), i.e. the
/// cases the two-layer mesher actually meshes through (ADR 0010 E3 / #50). `debug-clouds` is
/// VoxelBody-only (no chunkable extent) so it is excluded — `--two-layer` falls back to the dense
/// path there, which the cross-check would test only trivially. Every name MUST exist in
/// `CASES`.
///
/// ADR 0010 #53: the two LAYER-BAND-clip cases are now INCLUDED — the two-layer mesher honours
/// a layer band (clips coarse blocks to the band one-box, clips microblock cuboids, synthesises
/// cut-plane cap faces at the band edge), so the band slab renders pixel-identical to the dense
/// banded path with no dense source grids:
/// * `sketch-revolve-dome` — IMPLICITLY band-clipped: the layer-track upper bound is taken from
///   the (default-cylinder) `shape` grid_z (80), below the revolve composite grid_z (128), so
///   the dense golden clips the vase's upper third — and the two-layer band reclip now matches.
///
/// ADR 0012 (H1): `onion-ghost` is NOT in this list. The onion ghost is a TRANSLUCENT
/// alpha-blended pass, so the underlying solid's shading shows through the whole ghost cap —
/// and the dense apron mesh, the two-layer mesh, and the brick raymarch shade / decompose the
/// solid differently enough (and the two-layer banded mesh hits the known band-clip×elision
/// seam at a ghost-slab edge) that the translucent composite is NOT pixel-identical across
/// paths, even though the OPAQUE solid is. The ghost is therefore gated PER-PATH: the mesh
/// ghost by `golden_images_match` (the dense reference), the brick ghost by
/// `onion_ghost_marches_only_the_onion_slabs` in `tests/gpu_parity.rs`.
const TWO_LAYER_CASE_NAMES: &[&str] = &[
    "sphere-debug-faces",
    "cylinder",
    "torus",
    "demo-village",
    "demo-village-far",
    "demo-village-points",
    "cube-chrome-hover",
    "roll-quarter",
    "sketch-revolve-dome",
    "sketch-extrude-l",
    "demo-overlap",
    // ADR 0017 (#73): the subtract scene is chunkable and multi-producer, so the
    // two-layer cross-check re-renders the carved box through `--two-layer` and pins it
    // pixel-identical to the dense reference (the carve must classify + resolve the same
    // on both paths).
    "demo-subtract",
    // ADR 0017 Decision 3 (#74): the sealed-scope scene through `--two-layer` — the
    // scoped classification + scoped boundary resolve must render pixel-identical to
    // the dense scoped oracle (the group's cutter carves the group's body only, on
    // both paths).
    "demo-group-subtract",
    // ADR 0017 (#75): the intersect scene through `--two-layer` — the mask's
    // conservative interval fold (never-dropped mask candidates, whole-chunk degrade,
    // per-voxel boundary resolve) must render pixel-identical to the dense oracle.
    "demo-intersect",
    // ADR 0017 (#76): the instanced-cutter scene through `--two-layer` — the
    // definition-scope expansion under each instance's Subtract must classify +
    // resolve pixel-identical to the dense oracle at both placements.
    "demo-cutter-def",
    // ADR 0017 Decision 4 (#77): the window-fixture scene through `--two-layer` —
    // the frameless (spliced) definition expansion must classify + resolve
    // pixel-identical to the dense oracle: the spliced cutter is a root cutter and
    // the spliced frame a root additive leaf to the conservative fast paths.
    "demo-window-fixture",
    // ADR 0018 Decision 6: the boolean-operand ghost cases through `--two-layer`. The
    // ghost GEOMETRY is path-independent (derived from the operand slices' two-layer
    // chunks either way), and the two-layer solid's exposed-face set is proven identical
    // to the dense mesh — so the composite (translucent ghost over solid) must match the
    // dense reference within the tolerance band, unlike the onion ghost's known
    // band-clip×elision seam case above.
    "demo-buried-cutter",
    "demo-subtract-cutter-selected",
    "demo-window-fixture-selected",
    "demo-intersect-mask-selected",
    // The root-part master + its Normal-mode counterpart through `--two-layer` — same
    // path-independence argument (the Normal case is a plain finished solid).
    "demo-booleans-root",
    "demo-child-booleans-normal",
];

/// ADR 0011 G1 (#67): the golden cases whose scene is a chunkable SINGLE producer with a
/// uniform render cell — the ones the brick raymarch actually engages for (`shot --brick`).
/// Each renders brick-path pixel-identical (within the tolerance model) to the SAME committed
/// dense reference: the parity gate's clause (c). The
/// village/overlap cases are multi-producer (not gated); `sphere-debug-faces` disengages
/// bricks (debug-faces is a mesh-only mode). Every name MUST exist in `CASES`.
///
/// ADR 0012 (H1): `onion-ghost` is NOT here — its TRANSLUCENT ghost pass composites the
/// underlying solid's per-path shading through the whole cap, so brick vs dense-mesh is not
/// pixel-identical (≈6% on the ghost cap) even though the OPAQUE solid is. The brick ghost is
/// instead gated by `onion_ghost_marches_only_the_onion_slabs` in `tests/gpu_parity.rs` (the
/// ghost draws ONLY in the onion slabs) + the dense mesh ghost by `golden_images_match`.
const BRICK_CASE_NAMES: &[&str] = &[
    "cylinder",
    "torus",
    "sketch-revolve-dome",
    "sketch-extrude-l",
];

/// Fixed orbit angles so the framing is identical to the committed reference. The
/// distance is deliberately left to shot's auto-frame (it is a pure function of the
/// per-case grid dimensions, which are fixed here, so it stays deterministic) — a
/// hardcoded distance would clip these small grids out of frame. Pinning theta/phi
/// keeps the goldens independent of shot's default view angles.
const CAMERA_ARGS: &[&str] = &["--theta", "0.7", "--phi", "1.05"];

/// Directory holding the committed reference PNGs (`tests/golden/`).
fn golden_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
}

/// A per-run temp output dir (rendered PNGs + actual/diff artifacts on failure).
fn output_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("voxel_worker_golden");
    std::fs::create_dir_all(&dir).expect("failed to create golden temp dir");
    dir
}

/// Run the real `shot` binary for `case`, writing a PNG to `out_path`. `extra_args` appends
/// flags (e.g. `--two-layer` for the ADR 0010 E3 golden cross-check) so the same case can be
/// rendered through an alternate path and compared to the SAME committed reference.
fn render_case_with(case: &GoldenCase, out_path: &Path, extra_args: &[&str]) {
    let shot = env!("CARGO_BIN_EXE_shot");
    let status = Command::new(shot)
        .args(case.args)
        .args(extra_args)
        .args(CAMERA_ARGS)
        .args(["--width", &WIDTH.to_string()])
        .args(["--height", &HEIGHT.to_string()])
        .args(["--out", &out_path.to_string_lossy()])
        .status()
        .unwrap_or_else(|e| panic!("failed to launch shot for case '{}': {e}", case.name));
    assert!(
        status.success(),
        "shot exited with failure for case '{}' (status {status:?})",
        case.name
    );
    assert!(
        out_path.exists(),
        "shot did not produce an output PNG for case '{}' at {}",
        case.name,
        out_path.display()
    );
}

/// Render `case` through the DEFAULT (dense) path.
fn render_case(case: &GoldenCase, out_path: &Path) {
    render_case_with(case, out_path, &[]);
}

/// Run `shot` for `case` + `extra_args` (as [`render_case_with`]) but CAPTURE stdout —
/// used to assert the brick sink actually engaged (its `display=bricks` line) or fell
/// back, so a brick-vs-mesh comparison can't pass vacuously (mesh vs mesh).
fn render_case_capturing(case: &GoldenCase, out_path: &Path, extra_args: &[&str]) -> String {
    let shot = env!("CARGO_BIN_EXE_shot");
    let output = Command::new(shot)
        .args(case.args)
        .args(extra_args)
        .args(CAMERA_ARGS)
        .args(["--width", &WIDTH.to_string()])
        .args(["--height", &HEIGHT.to_string()])
        .args(["--out", &out_path.to_string_lossy()])
        .output()
        .unwrap_or_else(|e| panic!("failed to launch shot for case '{}': {e}", case.name));
    assert!(
        output.status.success(),
        "shot exited with failure for case '{}' (status {:?})",
        case.name,
        output.status
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Compare two same-size RGBA images under the tolerance model. Returns the
/// fraction of differing pixels, and writes a diff image (differing pixels in red,
/// matching pixels dimmed) to `diff_path` so failures are inspectable.
fn compare_images(actual: &RgbaImage, reference: &RgbaImage, diff_path: &Path) -> f64 {
    assert_eq!(
        actual.dimensions(),
        reference.dimensions(),
        "image dimensions differ: actual {:?} vs reference {:?}",
        actual.dimensions(),
        reference.dimensions()
    );

    let (width, height) = actual.dimensions();
    let mut diff = RgbaImage::new(width, height);
    let mut differing = 0u64;
    let total = (width as u64) * (height as u64);

    for (a, r, d) in itertools_zip(actual, reference, &mut diff) {
        let max_channel_diff =
            a.0.iter()
                .zip(r.0.iter())
                .map(|(av, rv)| av.abs_diff(*rv))
                .max()
                .unwrap_or(0);
        if max_channel_diff > CHANNEL_DIFF_THRESHOLD {
            differing += 1;
            *d = image::Rgba([255, 0, 0, 255]);
        } else {
            // Dim the matching pixel so the red diff stands out.
            *d = image::Rgba([r.0[0] / 3, r.0[1] / 3, r.0[2] / 3, 255]);
        }
    }

    diff.save(diff_path).expect("failed to write diff PNG");
    differing as f64 / total as f64
}

/// Iterate three images of identical size in lockstep ((actual, reference, &mut
/// diff) pixel triples). Avoids pulling in the `itertools` crate for one zip.
fn itertools_zip<'a>(
    actual: &'a RgbaImage,
    reference: &'a RgbaImage,
    diff: &'a mut RgbaImage,
) -> impl Iterator<
    Item = (
        &'a image::Rgba<u8>,
        &'a image::Rgba<u8>,
        &'a mut image::Rgba<u8>,
    ),
> {
    actual
        .pixels()
        .zip(reference.pixels())
        .zip(diff.pixels_mut())
        .map(|((a, r), d)| (a, r, d))
}

/// Load a PNG as RGBA8.
fn load_rgba(path: &Path) -> RgbaImage {
    image::open(path)
        .unwrap_or_else(|e| panic!("failed to open image {}: {e}", path.display()))
        .to_rgba8()
}

#[test]
fn golden_images_match() {
    let update = std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1");
    let golden_dir = golden_dir();
    let out_dir = output_dir();
    if update {
        std::fs::create_dir_all(&golden_dir).expect("failed to create tests/golden");
    }

    let mut failures: Vec<String> = Vec::new();

    for case in CASES {
        let reference_path = golden_dir.join(format!("{}.png", case.name));

        if update {
            // Regeneration mode: render straight into the reference path.
            render_case(case, &reference_path);
            println!("UPDATED golden: {}", reference_path.display());
            continue;
        }

        let actual_path = out_dir.join(format!("{}-actual.png", case.name));
        render_case(case, &actual_path);

        if !reference_path.exists() {
            failures.push(format!(
                "[{}] no reference at {} — run with UPDATE_GOLDENS=1 to create it",
                case.name,
                reference_path.display()
            ));
            continue;
        }

        let actual = load_rgba(&actual_path);
        let reference = load_rgba(&reference_path);
        let diff_path = out_dir.join(format!("{}-diff.png", case.name));
        let mismatch = compare_images(&actual, &reference, &diff_path);

        println!(
            "[{}] mismatch fraction = {:.5}% (threshold {:.3}%)",
            case.name,
            mismatch * 100.0,
            MAX_MISMATCH_FRACTION * 100.0
        );

        if mismatch > MAX_MISMATCH_FRACTION {
            failures.push(format!(
                "[{}] mismatch {:.5}% exceeds {:.3}% — actual: {}  diff: {}",
                case.name,
                mismatch * 100.0,
                MAX_MISMATCH_FRACTION * 100.0,
                actual_path.display(),
                diff_path.display()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "golden image regression(s):\n{}",
        failures.join("\n")
    );
}

/// ADR 0010 E3 (#50): render the chunkable golden cases THROUGH the two-layer mesh path
/// (`shot --two-layer`: coarse one-box + microblock cuboids + seam-flag culling) and assert
/// each is PIXEL-IDENTICAL to the SAME committed dense reference PNG. This is the display
/// half of the E3 parity gate — the two-layer mesher is a pure optimization on the data
/// seam, never an observable change. Includes the OVERLAP scene (the E2 carry-over: an
/// overlapping multi-material region must render identically to the dense path).
///
/// Run: `cargo test --features gpu --test golden`. Read the actual PNGs on a mismatch (the
/// large-solid `demo-village`/`demo-overlap` cases prove the one-box coarse path leaves no
/// interior seam or hole).
#[test]
fn two_layer_golden_matches_dense() {
    // Regeneration mode never targets the two-layer path (the references are the dense
    // goldens); skip cleanly so `UPDATE_GOLDENS=1` only refreshes the dense set.
    if std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1") {
        return;
    }
    let golden_dir = golden_dir();
    let out_dir = output_dir();
    let mut failures: Vec<String> = Vec::new();

    for name in TWO_LAYER_CASE_NAMES {
        let case = CASES
            .iter()
            .find(|c| c.name == *name)
            .unwrap_or_else(|| panic!("TWO_LAYER_CASE_NAMES references unknown case '{name}'"));
        let reference_path = golden_dir.join(format!("{}.png", case.name));
        if !reference_path.exists() {
            failures.push(format!(
                "[{}] no dense reference at {} — run the dense golden first",
                case.name,
                reference_path.display()
            ));
            continue;
        }

        // Render the case through the two-layer mesh path.
        let actual_path = out_dir.join(format!("{}-two-layer-actual.png", case.name));
        render_case_with(case, &actual_path, &["--two-layer"]);

        let actual = load_rgba(&actual_path);
        let reference = load_rgba(&reference_path);
        let diff_path = out_dir.join(format!("{}-two-layer-diff.png", case.name));
        let mismatch = compare_images(&actual, &reference, &diff_path);
        println!(
            "[{} two-layer] mismatch fraction = {:.5}% (threshold {:.3}%)",
            case.name,
            mismatch * 100.0,
            MAX_MISMATCH_FRACTION * 100.0
        );
        if mismatch > MAX_MISMATCH_FRACTION {
            failures.push(format!(
                "[{} two-layer] mismatch {:.5}% exceeds {:.3}% — actual: {}  diff: {}",
                case.name,
                mismatch * 100.0,
                MAX_MISMATCH_FRACTION * 100.0,
                actual_path.display(),
                diff_path.display()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "two-layer golden regression(s) vs dense reference:\n{}",
        failures.join("\n")
    );
}

/// ADR 0011 G1 (#67): render the gated single-producer golden cases THROUGH the brick
/// raymarch (`shot --brick`: block-DDA over the G0 sorted records + R8 sculpted atlas, the
/// cuboid mesh built EMPTY so the pixels provably come from the atlas) and assert each is
/// PIXEL-IDENTICAL to the SAME committed dense reference — the parity gate's clause (c). The
/// finest-LOD raymarch is designed (per-sample MSAA rays + pixel-centre face evaluation) to
/// reproduce the rasterized mesh, not merely approximate it, so this reuses the mesh path's
/// own goldens with no new references. The view cube composites over the brick-drawn solid
/// the same as over the mesh, so a byte-for-byte-equivalent render is the depth-compositing
/// evidence (grill Q5 / the one integration point the ADR 0009 benchmark never exercised).
///
/// Run: `cargo test --features gpu --test golden`. On a mismatch, read the `-brick-actual.png`
/// and `-brick-diff.png` artifacts — a silhouette-only diff points at MSAA sample positions,
/// an interior diff at the shading transcription.
#[test]
fn brick_golden_matches_dense() {
    // Regeneration mode targets only the dense references; skip so `UPDATE_GOLDENS=1` never
    // writes a brick render as a reference.
    if std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1") {
        return;
    }
    let golden_dir = golden_dir();
    let out_dir = output_dir();
    let mut failures: Vec<String> = Vec::new();

    for name in BRICK_CASE_NAMES {
        let case = CASES
            .iter()
            .find(|c| c.name == *name)
            .unwrap_or_else(|| panic!("BRICK_CASE_NAMES references unknown case '{name}'"));
        let reference_path = golden_dir.join(format!("{}.png", case.name));
        if !reference_path.exists() {
            failures.push(format!(
                "[{}] no dense reference at {} — run the dense golden first",
                case.name,
                reference_path.display()
            ));
            continue;
        }

        // Render the case through the brick raymarch path.
        let actual_path = out_dir.join(format!("{}-brick-actual.png", case.name));
        render_case_with(case, &actual_path, &["--brick"]);

        let actual = load_rgba(&actual_path);
        let reference = load_rgba(&reference_path);
        let diff_path = out_dir.join(format!("{}-brick-diff.png", case.name));
        let mismatch = compare_images(&actual, &reference, &diff_path);
        println!(
            "[{} brick] mismatch fraction = {:.5}% (threshold {:.3}%)",
            case.name,
            mismatch * 100.0,
            MAX_MISMATCH_FRACTION * 100.0
        );
        if mismatch > MAX_MISMATCH_FRACTION {
            failures.push(format!(
                "[{} brick] mismatch {:.5}% exceeds {:.3}% — actual: {}  diff: {}",
                case.name,
                mismatch * 100.0,
                MAX_MISMATCH_FRACTION * 100.0,
                actual_path.display(),
                diff_path.display()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "brick golden regression(s) vs dense reference (ADR 0011 gate (c)):\n{}",
        failures.join("\n")
    );
}

/// ADR 0011 G2 (#68): the multi-producer per-record-material golden. `--demo-two-material`
/// is two DISTINCT-material boxes placed a whole chunk apart, so every rendered block is
/// single-material — the widened brick gate engages, and the sink shades each hit from its
/// OWN record's packed material id (not a scene-wide uniform), reproducing the mesh's
/// two-tone render. Because the scene has no historical committed reference, this uses the
/// FRESH-render-vs-FRESH-render technique: render the mesh path and the brick path THIS run
/// and compare them under the SAME MSAA tolerance band the dense goldens use. The captured
/// `--brick` stdout must show the sink engaged (`display=bricks`), so the comparison can't
/// pass vacuously (mesh vs mesh).
///
/// (Note: `--demo-overlap`, though multi-material, offsets its boxes in WHOLE blocks, so every
/// block is still single-material. The genuinely-mixed case — a block whose microblocks MIX
/// materials — is `--demo-mixed-material` (a sub-block voxel offset), locked by
/// `brick_golden_mixed_material_matches_mesh` below now that the representability gate is deleted
/// and mixed scenes engage the brick path.)
#[test]
fn brick_golden_multi_material_matches_mesh() {
    if std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1") {
        return;
    }
    let out_dir = output_dir();

    // The gated multi-producer scene: mesh path vs brick path, both rendered fresh.
    let case = GoldenCase {
        name: "demo-two-material",
        args: &["--demo-two-material"],
    };
    let mesh_path = out_dir.join("demo-two-material-mesh.png");
    render_case_with(&case, &mesh_path, &[]);
    let brick_path = out_dir.join("demo-two-material-brick-actual.png");
    let brick_stdout = render_case_capturing(&case, &brick_path, &["--brick"]);
    assert!(
        brick_stdout.contains("display=bricks"),
        "the multi-producer scene must ENGAGE the brick sink (else brick==mesh is vacuous); \
         shot stdout was:\n{brick_stdout}"
    );

    let mesh = load_rgba(&mesh_path);
    let brick = load_rgba(&brick_path);
    let diff_path = out_dir.join("demo-two-material-brick-diff.png");
    let mismatch = compare_images(&brick, &mesh, &diff_path);
    println!(
        "[demo-two-material brick] mismatch fraction = {:.5}% (threshold {:.3}%)",
        mismatch * 100.0,
        MAX_MISMATCH_FRACTION * 100.0
    );
    assert!(
        mismatch <= MAX_MISMATCH_FRACTION,
        "brick != mesh on the multi-producer distinct-material scene: {:.5}% > {:.3}% — \
         brick: {}  mesh: {}  diff: {}",
        mismatch * 100.0,
        MAX_MISMATCH_FRACTION * 100.0,
        brick_path.display(),
        mesh_path.display(),
        diff_path.display()
    );
}

/// Material atlas / ADR 0013 — the MIXED-material golden: the proof the mesh cliff is dead.
/// `--demo-mixed-material` offsets its second (Wood) box by a SUB-BLOCK voxel amount, so a block
/// STRADDLES the boundary and its microblocks MIX Stone + Wood. This is the case the deleted
/// representability gate routed to the mesh; now it engages the brick sink and shades each voxel
/// from its cell-key side atlas. Rendered fresh mesh-vs-brick under the SAME MSAA tolerance band
/// as the dense goldens (no historical committed reference), and the `--brick` stdout must show
/// `display=bricks` so the comparison can't pass vacuously (mesh vs mesh). A per-record-uniform
/// scene would render identically either way; here the block is genuinely mixed, so a byte-equal
/// render proves the per-voxel cell-key shade reproduces the mesh's per-voxel materials.
#[test]
fn brick_golden_mixed_material_matches_mesh() {
    if std::env::var("UPDATE_GOLDENS").is_ok_and(|v| v == "1") {
        return;
    }
    let out_dir = output_dir();

    let case = GoldenCase {
        name: "demo-mixed-material",
        args: &["--demo-mixed-material"],
    };
    let mesh_path = out_dir.join("demo-mixed-material-mesh.png");
    render_case_with(&case, &mesh_path, &[]);
    let brick_path = out_dir.join("demo-mixed-material-brick-actual.png");
    let brick_stdout = render_case_capturing(&case, &brick_path, &["--brick"]);
    assert!(
        brick_stdout.contains("display=bricks"),
        "the mixed-material scene must ENGAGE the brick sink (else brick==mesh is vacuous); \
         shot stdout was:\n{brick_stdout}"
    );

    let mesh = load_rgba(&mesh_path);
    let brick = load_rgba(&brick_path);
    let diff_path = out_dir.join("demo-mixed-material-brick-diff.png");
    let mismatch = compare_images(&brick, &mesh, &diff_path);
    println!(
        "[demo-mixed-material brick] mismatch fraction = {:.5}% (threshold {:.3}%)",
        mismatch * 100.0,
        MAX_MISMATCH_FRACTION * 100.0
    );
    assert!(
        mismatch <= MAX_MISMATCH_FRACTION,
        "brick != mesh on the MIXED-material scene (the cliff proof): {:.5}% > {:.3}% — \
         brick: {}  mesh: {}  diff: {}",
        mismatch * 100.0,
        MAX_MISMATCH_FRACTION * 100.0,
        brick_path.display(),
        mesh_path.display(),
        diff_path.display()
    );
}
