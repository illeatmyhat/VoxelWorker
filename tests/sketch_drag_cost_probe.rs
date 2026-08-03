//! What one FRAME of a sketch drag costs, split into the parts that make it up.
//!
//! The boundary probe next door asks what a finished profile costs to resolve. This one asks the
//! question an author actually feels: while the mouse is down, the shell rebuilds the whole node
//! every frame, and every part of that rebuild scales with what is on the plane rather than with
//! what is being dragged. Which part dominates decides what is worth fixing — a settle that grows
//! with the constraint count needs a different answer from a resolve that grows with the edge
//! count, and neither is visible in a single end-to-end number.
//!
//! Reported per POPULATION, because the interesting number is not one measurement but the slope.
//!
//! Run: `cargo test --release --test sketch_drag_cost_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic
)]

use std::time::Instant;

use camera::OrbitCamera;
use document::scene::{Node, NodeContent, Scene};
use document::sketch::{PlaneAxis, Sketch, SketchLength, SketchPoint, SketchSolid};
use voxel_core::core_geom::MaterialChoice;
use voxel_worker::AppCore;

const DENSITY: u32 = 16;
const DEPTH_VOXELS: u32 = 32;

const fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(DENSITY).expect("probe density is non-zero"),
    )
}

/// `count` arc slots side by side, the shape the owner reported as the slow one.
fn arc_slots(count: usize) -> SketchSolid {
    let mut made = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), DEPTH_VOXELS);
    for index in 0..count {
        let center = (index as i64) * 40;
        made = made
            .with_center_arc_slot(
                SketchPoint::new(center, 0),
                SketchPoint::new(center + 8, 0),
                SketchPoint::new(center, 8),
                parametric::sketch::ArcTurn::CounterClockwise,
                SketchPoint::new(center + 10, 0),
                context(),
            )
            .expect("a quarter-turn arc slot");
    }
    made
}

/// `count` circles, the same entity count's worth of geometry with NO relations — the control
/// that separates "the drawing is big" from "the drawing is held together".
fn circles(count: usize) -> SketchSolid {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    for index in 0..count {
        sketch
            .add_circle(
                SketchPoint::new((index as i64) * 40, 0),
                SketchLength::new(8),
            )
            .expect("a circle of non-zero radius");
    }
    SketchSolid::extrude(sketch, DEPTH_VOXELS)
}

fn scene_of(producer: &SketchSolid) -> Scene {
    Scene::from_nodes(vec![Node::new(
        "Profile",
        NodeContent::SketchTool {
            producer: producer.clone(),
            material: MaterialChoice::Stone,
        },
    )])
}

/// One frame of the drag, timed the way the shell actually spends it.
fn report(label: &str, made: &SketchSolid) {
    let grabbed = made
        .sketch
        .points()
        .first()
        .map(|point| point.id)
        .expect("the drawing has a point to grab");
    let entities = made.sketch.points().len();
    let relations = made.sketch.constraints().len();

    // 1. The preview's base clone — every frame starts from the pre-drag snapshot.
    let started = Instant::now();
    let mut preview = made.clone();
    let clone_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 2. The edit itself: prepare the constraint problem, settle, write back.
    let started = Instant::now();
    let _moved = preview
        .sketch
        .move_point(grabbed, SketchPoint::new(1, 1), context())
        .expect("the drag resolves");
    let move_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 3. The region derive, on the FRESH clone the preview just made — the memo starts empty
    //    every frame, which is exactly the situation a cached one is supposed to avoid.
    let started = Instant::now();
    let loops = preview.sketch.region(context()).len();
    let region_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 4. The re-resolve the shell triggers by mutating the node — WARM, which is the only
    //    honest measurement. A fresh `AppCore` has no previous leaf index, so `rebuild` takes
    //    the wholesale `clear()` arm; the live app is always on the targeted `invalidate_aabb`
    //    arm, and the whole question is how much that arm actually saves on a drag.
    let mut app_core = AppCore::new(OrbitCamera::default());
    drop(app_core.rebuild(&scene_of(made), DENSITY));
    let scene = scene_of(&preview);
    let started = Instant::now();
    let outcome = app_core.rebuild(&scene, DENSITY);
    let rebuild_ms = started.elapsed().as_secs_f64() * 1000.0;
    // `incremental_dirty_chunks == None` conflates two very different failures — no localizable
    // edit AABB (the resident cache is CLEARED) versus a recenter reframe (the cache survived,
    // only the GPU mesh hint was dropped). Ask the index itself which one happened, since only
    // the first explains a whole-node re-resolve.
    let shifted = match &outcome {
        voxel_worker::RebuildOutcome::Built(output) => output.recenter_shift_voxels != [0; 3],
        voxel_worker::RebuildOutcome::DensityRejected { .. } => false,
    };
    let before = scene_of(made).build_leaf_spatial_index(DENSITY);
    let after = scene.build_leaf_spatial_index(DENSITY);
    let dirty = after.edit_aabb_since(&before).map_or_else(
        || "cleared".to_owned(),
        |aabb| {
            let span: [i64; 3] = std::array::from_fn(|axis| aabb.max[axis] - aabb.min[axis]);
            let reframed = if shifted { "+shift" } else { "" };
            format!("{}x{}x{}{reframed}", span[0], span[1], span[2])
        },
    );

    let total = clone_ms + move_ms + region_ms + rebuild_ms;
    println!(
        "{label:<20} {entities:>6} {relations:>6} {loops:>6} {clone_ms:>8.2} {move_ms:>8.2} \
         {region_ms:>8.2} {rebuild_ms:>9.2} {total:>8.2} {dirty:>7}"
    );
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn sketch_drag_frame_cost_by_population() {
    println!(
        "\n{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>9} {:>8} {:>7}",
        "drawing",
        "points",
        "rels",
        "loops",
        "clone",
        "move",
        "region",
        "rebuild",
        "total",
        "dirty"
    );
    println!("{}", "-".repeat(92));
    for count in [1_usize, 8, 16, 32] {
        report(&format!("{count} arc slots"), &arc_slots(count));
    }
    for count in [1_usize, 8, 16, 32] {
        report(&format!("{count} circles"), &circles(count));
    }
}
