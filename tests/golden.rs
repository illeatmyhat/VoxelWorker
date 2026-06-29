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

/// Run the real `shot` binary for `case`, writing a PNG to `out_path`.
fn render_case(case: &GoldenCase, out_path: &Path) {
    let shot = env!("CARGO_BIN_EXE_shot");
    let status = Command::new(shot)
        .args(case.args)
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
