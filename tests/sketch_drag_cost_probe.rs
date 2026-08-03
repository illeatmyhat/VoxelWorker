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

fn scene_of(producer: &SketchSolid, offset_voxels: [i64; 3]) -> Scene {
    let mut node = Node::new(
        "Profile",
        NodeContent::SketchTool {
            producer: producer.clone(),
            material: MaterialChoice::Stone,
        },
    );
    node.transform.offset_voxels = offset_voxels;
    Scene::from_nodes(vec![node])
}

/// Which point the gesture grabs — the two cases differ in kind, not degree.
///
/// A CORNER grab moves the profile's bounding box, so the shell answers by shifting the node
/// offset to keep the drawing still, which moves every voxel the node emits. An INTERIOR grab
/// leaves the box alone and is what an author does most of the time. Reporting only the corner
/// case would blame the resolve for work the anchor policy caused.
enum Grab {
    Corner,
    Interior,
}

/// One frame of the drag, timed the way the shell actually spends it.
fn report(label: &str, made: &SketchSolid, grab: &Grab) {
    let points = made.sketch.points();
    let grabbed = match *grab {
        Grab::Corner => points.first(),
        // Mid-drawing, so neither the low nor the high corner of the profile box is the one
        // being moved.
        Grab::Interior => points.get(points.len() / 2),
    }
    .map(|point| point.id)
    .expect("the drawing has a point to grab");
    let entities = points.len();
    let relations = made.sketch.constraints().len();

    // 1. The preview's base clone — every frame starts from the pre-drag snapshot.
    let started = Instant::now();
    let mut preview = made.clone();
    let clone_ms = started.elapsed().as_secs_f64() * 1000.0;

    // 2. The edit itself: prepare the constraint problem, settle, write back. Nudged by one
    //    voxel from wherever the point already is — a drag frame is a small delta, and teleporting
    //    the point to the origin would fold every other shape's distance into the settle.
    let was = preview
        .sketch
        .points()
        .iter()
        .find(|point| point.id == grabbed)
        .map(|point| point.at)
        .expect("the grabbed point is in the drawing");
    let to = SketchPoint::from_continuous(was.in_plane()[0] + 1.0, was.in_plane()[1] + 1.0);
    let started = Instant::now();
    let _moved = preview
        .sketch
        .move_point(grabbed, to, context())
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
    //
    //    The node offset carries the shell's ANCHOR COMPENSATION: `preview_sketch_vertex_drag`
    //    shifts the offset by the profile bbox-min delta so the untouched part of the drawing
    //    holds still on screen. Without it the probe reports a recenter shift on every frame that
    //    the real app cancels, which slanders the resolve for the anchor policy's work.
    let plane = made.sketch.plane.in_plane_axes();
    let original_min = made.profile_bbox_min(context());
    let new_min = preview.profile_bbox_min(context());
    let mut offset = [0i64; 3];
    offset[plane[0]] = new_min[0] - original_min[0];
    offset[plane[1]] = new_min[1] - original_min[1];

    let mut app_core = AppCore::new(OrbitCamera::default());
    drop(app_core.rebuild(&scene_of(made, [0; 3]), DENSITY));
    let scene = scene_of(&preview, offset);
    let started = Instant::now();
    let outcome = app_core.rebuild(&scene, DENSITY);
    let rebuild_ms = started.elapsed().as_secs_f64() * 1000.0;
    // `incremental_dirty_chunks == None` conflates two very different failures — no localizable
    // edit AABB (the resident cache is CLEARED) versus a recenter reframe (the cache survived,
    // only the GPU mesh hint was dropped). Ask the index itself which one happened, since only
    // the first explains a whole-node re-resolve.
    let (shifted, evicted, kept) = match &outcome {
        voxel_worker::RebuildOutcome::Built(output) => (
            output.recenter_shift_voxels != [0; 3],
            output
                .incremental_dirty_chunks
                .as_ref()
                .map_or_else(|| "all".to_owned(), |chunks| chunks.len().to_string()),
            output.two_layer_chunks.len(),
        ),
        voxel_worker::RebuildOutcome::DensityRejected { .. } => (false, "rejected".to_owned(), 0),
    };
    let churn = format!("{evicted}/{kept}");
    let before = scene_of(made, [0; 3]).build_leaf_spatial_index(DENSITY);
    // The broadphase index is rebuilt from scratch inside every `rebuild`, so its own cost is
    // part of the frame — and it fingerprints content, which is string work proportional to the
    // drawing. Worth its own column: a dirty box that no longer grows is no win if the machinery
    // that computes it does.
    let started = Instant::now();
    let after = scene.build_leaf_spatial_index(DENSITY);
    let index_ms = started.elapsed().as_secs_f64() * 1000.0;
    // The resident cache rebuilds the leaf producer list on EVERY call, before it knows whether
    // any chunk is missing — and a sketch leaf's producer is a fresh clone whose region memo
    // starts empty. If that is where the frame goes, no amount of dirty-box precision helps.
    let started = Instant::now();
    let leaves = scene.leaf_producers(DENSITY).len();
    let leaves_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(leaves > 0, "the probe scene has a leaf");
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
         {region_ms:>8.2} {index_ms:>7.2} {leaves_ms:>7.2} {rebuild_ms:>9.2} {total:>8.2} {churn:>8} {dirty:>7}"
    );
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn sketch_drag_frame_cost_by_population() {
    println!(
        "\n{:<20} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>7} {:>7} {:>9} {:>8} {:>8} {:>7}",
        "drawing",
        "points",
        "rels",
        "loops",
        "clone",
        "move",
        "region",
        "index",
        "leaves",
        "rebuild",
        "total",
        "chunks",
        "dirty"
    );
    println!("{}", "-".repeat(92));
    for count in [1_usize, 8, 16, 32] {
        let made = arc_slots(count);
        report(&format!("{count} slots corner"), &made, &Grab::Corner);
        report(&format!("{count} slots inner"), &made, &Grab::Interior);
    }
    for count in [1_usize, 8, 16, 32] {
        let made = circles(count);
        report(&format!("{count} circles corner"), &made, &Grab::Corner);
        report(&format!("{count} circles inner"), &made, &Grab::Interior);
    }
}
