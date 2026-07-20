//! What one edit costs — the number that decides whether a manipulator drag can move a
//! COMMITTED node live under the cursor, or has to preview as a ghost and commit on release
//! (`docs/design/direct-manipulation.md`, the open question).
//!
//! `AppCore::rebuild` is the blocking half of an edit: it builds the new leaf spatial index,
//! diffs it for the dirty world-AABB, evicts only the chunks that AABB touches, and
//! re-classifies those. The brick sink then rebuilds on its own worker, so it does not block
//! the frame; the mesh path is skipped while bricks are engaged. So this is what the user
//! waits for.
//!
//! Reported per scene size, because the answer is allowed to be "yes on small scenes": the
//! interesting question is not one number but where the curve crosses a frame budget.
//!
//! Run: `cargo test --release --test edit_cost_probe -- --ignored --nocapture`

use std::time::Instant;

use camera::OrbitCamera;
use document::intent::Intent;
use document::scene::Scene;
use document::voxel::{GeometryParams, SdfShape};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::units::Measurement;
use voxel_core::voxel::ShapeKind;
use voxel_worker::AppCore;

/// Voxels per block — the document default.
const DENSITY: u32 = 16;
/// How many edits to time per scene; the median is reported so one unlucky allocation does
/// not decide an interaction model.
const SAMPLES: usize = 9;

/// A one-node scene of `blocks` extent, as the panel would seed it.
fn scene_of(kind: ShapeKind, blocks: [u32; 3]) -> Scene {
    let shape = SdfShape::from_blocks(kind, blocks, 1, DENSITY);
    Scene::from_geometry(
        GeometryParams {
            shape: kind,
            size_voxels: shape.size_voxels,
            size_measurements: None,
            voxels_per_block: DENSITY,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    )
}

/// The median of a set of millisecond timings.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
    samples[samples.len() / 2]
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn one_edit_rebuild_cost_by_scene_size() {
    println!(
        "\n{:<22} {:>10} {:>12} {:>12} {:>10}",
        "scene", "voxels", "first (ms)", "edit (ms)", "at 60fps"
    );
    println!("{}", "-".repeat(70));

    for (label, kind, blocks) in [
        ("small  5x1x5", ShapeKind::Cylinder, [5u32, 1, 5]),
        ("medium 20x8x20", ShapeKind::Cylinder, [20, 8, 20]),
        ("large  50x10x50", ShapeKind::Tube, [50, 10, 50]),
        ("huge   100x20x100", ShapeKind::Tube, [100, 20, 100]),
    ] {
        let mut scene = scene_of(kind, blocks);
        let mut app_core = AppCore::new(OrbitCamera::default());
        let target = *scene.roots.first().expect("seeded scene has a root node");

        // The FIRST rebuild is wholesale (no previous leaf index), so it is the cold cost —
        // reported separately, because it is not what a drag pays.
        let started = Instant::now();
        {
            let _outcome = app_core.rebuild(&scene, DENSITY);
        }
        let first_ms = started.elapsed().as_secs_f64() * 1000.0;

        // Then the steady-state cost of nudging the node one voxel — exactly what a
        // manipulator drag emits, and the case targeted invalidation is built for.
        let mut samples = Vec::with_capacity(SAMPLES);
        for step in 1..=SAMPLES {
            let offset = [
                Measurement::from_voxels(step as i64),
                Measurement::from_voxels(0),
                Measurement::from_voxels(0),
            ];
            app_core.apply_intent(
                &mut scene,
                Intent::SetOffset {
                    target,
                    offset_measurements: offset,
                },
            );
            let started = Instant::now();
            {
                let _outcome = app_core.rebuild(&scene, DENSITY);
            }
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }

        let edit_ms = median(samples);
        let voxels: u64 = blocks.iter().map(|b| (*b * DENSITY) as u64).product();
        let verdict = if edit_ms <= 16.6 {
            "live ok"
        } else if edit_ms <= 33.3 {
            "30fps"
        } else {
            "GHOST"
        };
        println!(
            "{label:<22} {voxels:>10} {first_ms:>12.1} {edit_ms:>12.1} {verdict:>10}"
        );
    }
    println!(
        "\n'live ok' = a committed node can move under the cursor at 60fps.\n\
         'GHOST'   = the drag must preview and commit on release.\n"
    );
}
