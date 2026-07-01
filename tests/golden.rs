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
    // Issue #28 S5b: lock the PER-CHUNK onion fog (now the default) with a clearly-
    // fogged scene. An 8³-block sphere (grid 128³, 8 resident chunk volumes — 2 per
    // axis) with an onion-skinned equatorial band: layers [56,72] render as the crisp
    // solid stone disk, while the sphere's volume ABOVE and BELOW the band ghosts as a
    // soft blue/grey haze (8 onion layers each side). The haze is sampled from the
    // per-chunk fog atlas (the S5b default) and crosses every chunk seam, so this golden
    // proves the per-chunk path produces CONTINUOUS fog with no seam lines at a fixed
    // camera. `--fog=perchunk` is explicit so the case stays pinned to the per-chunk
    // path even if the default is ever changed again.
    GoldenCase {
        name: "onion-fog-perchunk",
        args: &[
            "--shape", "sphere", "--size-x", "8", "--size-y", "8", "--size-z", "8",
            "--onion", "8", "--layer-lower", "56", "--layer-upper", "72",
            "--fog", "perchunk",
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
    GoldenCase {
        name: "sketch-revolve-dome",
        args: &["--demo-sketch-revolve"],
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
];

/// The subset of [`CASES`] whose scene is CHUNKABLE (has an intrinsic-size leaf), i.e. the
/// cases the two-layer mesher actually meshes through (ADR 0010 E3 / #50). `debug-clouds` is
/// Part-only (no chunkable extent) so it is excluded — `--two-layer` falls back to the dense
/// path there, which the cross-check would test only trivially. Every name MUST exist in
/// `CASES`.
///
/// ADR 0010 #53: the two LAYER-BAND-clip cases are now INCLUDED — the two-layer mesher honours
/// a layer band (clips coarse blocks to the band one-box, clips microblock cuboids, synthesises
/// cut-plane cap faces at the band edge), so the band slab renders pixel-identical to the dense
/// banded path with no dense source grids:
/// * `onion-fog-perchunk` — an explicit `--onion`/`--layer-*` band.
/// * `sketch-revolve-dome` — IMPLICITLY band-clipped: the layer-track upper bound is taken from
///   the (default-cylinder) `shape` grid_z (80), below the revolve composite grid_z (128), so
///   the dense golden clips the vase's upper third — and the two-layer band reclip now matches.
const TWO_LAYER_CASE_NAMES: &[&str] = &[
    "sphere-debug-faces",
    "cylinder",
    "torus",
    "onion-fog-perchunk",
    "demo-village",
    "demo-village-far",
    "demo-village-points",
    "cube-chrome-hover",
    "roll-quarter",
    "sketch-revolve-dome",
    "sketch-extrude-l",
    "demo-overlap",
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
