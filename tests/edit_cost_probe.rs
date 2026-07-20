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
use document::intent::{Intent, NodeSpec};
use document::scene::{NodeId, Scene};
use document::voxel::{GeometryParams, SdfShape};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::units::Measurement;
use voxel_core::voxel::ShapeKind;
use voxel_worker::{AppCore, RebuildOutcome};

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

/// The blocks-extent of the node a manipulator actually drags in a real scene: small, and
/// entirely inside the backdrop it sits in. The point of the second probe is that this node
/// is NOT the scene, so its dirty AABB is a fraction of the covering set.
const DRAGGED_BLOCKS: [u32; 3] = [2, 2, 2];

/// Add a small Tool node to `scene` through the same intent the add-flow will emit, and
/// return its minted id (`add_node` selects what it adds, so `scene.active` names it).
fn add_dragged_node(scene: &mut Scene, app_core: &mut AppCore) -> NodeId {
    let shape = SdfShape::from_blocks(ShapeKind::Box, DRAGGED_BLOCKS, 1, DENSITY);
    app_core.apply_intent(
        scene,
        Intent::AddNode {
            content: NodeSpec::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        },
    );
    scene.active.expect("add_node selects the node it mints")
}

/// Time one `SetOffset` + rebuild, reporting the milliseconds and the rebuild's
/// `incremental_dirty_chunks` hint — `Some(n)` = the edit localised to `n` evicted chunks and
/// the resident buffers stayed in frame. `None` covers BOTH non-local outcomes (a wholesale
/// cache clear, and a localised edit whose floating-origin shift reframed every baked buffer);
/// `rebuild` does not distinguish them to its caller, so neither does this probe.
fn timed_offset(
    scene: &mut Scene,
    app_core: &mut AppCore,
    target: NodeId,
    offset_voxels: [i64; 3],
) -> (f64, Option<usize>) {
    app_core.apply_intent(
        scene,
        Intent::SetOffset {
            target,
            offset_measurements: offset_voxels.map(Measurement::from_voxels),
        },
    );
    let started = Instant::now();
    let localised = match app_core.rebuild(scene, DENSITY) {
        RebuildOutcome::Built(output) => output.incremental_dirty_chunks.map(|dirty| dirty.len()),
        RebuildOutcome::DensityRejected { .. } => panic!("the probe's density is in bounds"),
    };
    (started.elapsed().as_secs_f64() * 1000.0, localised)
}

/// **The case the first probe does not measure, and the one that decides the interaction
/// model.** Every scene above holds exactly ONE node, which therefore IS the whole scene, so
/// moving it dirties everything — the worst case. What a manipulator actually drags is a small
/// node inside a big scene, where the dirty AABB is a fraction of the covering set.
///
/// Both columns are measured on the SAME multi-node scene, so the contrast is the edit's
/// locality and nothing else: dragging the small node against dragging the backdrop it sits in.
///
/// The dragged node stays well inside the backdrop's bounds on purpose. A node that moved the
/// scene's overall extent would shift the floating origin, and `rebuild` reframes every baked
/// buffer on a shift — a different (and much more expensive) path than the one a drag inside
/// existing geometry takes. That path is worth measuring too, but it is not this number.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn one_edit_rebuild_cost_by_edit_locality() {
    println!(
        "\n{:<22} {:>10} {:>14} {:>8} {:>14} {:>8}",
        "backdrop", "voxels", "small (ms)", "chunks", "backdrop (ms)", "chunks"
    );
    println!("{}", "-".repeat(82));

    for (label, kind, blocks) in [
        ("small  5x1x5", ShapeKind::Cylinder, [5u32, 1, 5]),
        ("medium 20x8x20", ShapeKind::Cylinder, [20, 8, 20]),
        ("large  50x10x50", ShapeKind::Tube, [50, 10, 50]),
        ("huge   100x20x100", ShapeKind::Tube, [100, 20, 100]),
    ] {
        let mut scene = scene_of(kind, blocks);
        let mut app_core = AppCore::new(OrbitCamera::default());
        let backdrop = *scene.roots.first().expect("seeded scene has a root node");
        let dragged = add_dragged_node(&mut scene, &mut app_core);

        // Park the dragged node near the backdrop's centre, so the whole gesture happens
        // inside existing geometry and the scene's extent never grows.
        let centre = blocks.map(|b| (b as i64 / 2) * DENSITY as i64);
        let _ = app_core.rebuild(&scene, DENSITY);
        let _ = timed_offset(&mut scene, &mut app_core, dragged, centre);

        // The drag that matters: nudge the SMALL node one voxel at a time.
        let mut small_samples = Vec::with_capacity(SAMPLES);
        let mut small_chunks = None;
        for step in 1..=SAMPLES as i64 {
            let offset = [centre[0] + step, centre[1], centre[2]];
            let (ms, localised) = timed_offset(&mut scene, &mut app_core, dragged, offset);
            small_samples.push(ms);
            small_chunks = localised;
        }

        // The same gesture on the BACKDROP, for the contrast — this is the first probe's
        // case, re-measured here so both numbers come off one scene.
        let mut backdrop_samples = Vec::with_capacity(SAMPLES);
        let mut backdrop_chunks = None;
        for step in 1..=SAMPLES as i64 {
            let (ms, localised) = timed_offset(&mut scene, &mut app_core, backdrop, [step, 0, 0]);
            backdrop_samples.push(ms);
            backdrop_chunks = localised;
        }

        let voxels: u64 = blocks.iter().map(|b| (*b * DENSITY) as u64).product();
        let show = |chunks: Option<usize>| match chunks {
            Some(count) => count.to_string(),
            None => "none".to_string(),
        };
        println!(
            "{label:<22} {voxels:>10} {:>14.1} {:>8} {:>14.1} {:>8}",
            median(small_samples),
            show(small_chunks),
            median(backdrop_samples),
            show(backdrop_chunks),
        );
    }
    println!(
        "\n'chunks' = how many chunks the edit's dirty AABB evicted, keeping every other chunk\n\
         resident. 'none' = no incremental hint: the rebuild either cleared wholesale or shifted\n\
         the floating origin, which reframes every baked buffer.\n"
    );
}
