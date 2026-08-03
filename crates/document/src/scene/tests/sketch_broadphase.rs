//! What a sketch edit tells the invalidation broadphase it dirtied.
//!
//! A sketch node is one leaf, and a leaf is one entry — so before the per-loop split, moving one
//! vertex of one shape read as "the whole drawing changed" and re-resolved every chunk the drawing
//! covered. These tests pin the split's two obligations: the dirty box must name only the shape
//! that moved, and the refinement must stand down wherever a loop's cells are not confined to the
//! loop's own in-plane box.

use super::*;
use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchLength, SketchPoint, SketchSolid};

const SKETCH_DENSITY: u32 = 16;
const DEPTH_VOXELS: u32 = 32;

fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(SKETCH_DENSITY).expect("the probe density is non-zero"),
    )
}

/// `count` circles in a row, 40 voxels apart — disjoint shapes on one plane, which is the case
/// the split exists for.
fn circle_row(count: i64) -> Sketch {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    for index in 0..count {
        sketch
            .add_circle(SketchPoint::new(index * 40, 0), SketchLength::new(8))
            .expect("a circle of non-zero radius");
    }
    sketch
}

fn scene_of(producer: SketchSolid) -> Scene {
    scene_at(producer, [0; 3])
}

fn scene_at(producer: SketchSolid, offset_voxels: [i64; 3]) -> Scene {
    let mut node = Node::new(
        "Profile",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Stone,
        },
    );
    node.transform.offset_voxels = offset_voxels;
    Scene::from_nodes(vec![node])
}

/// The scene an edit produces, carrying the same ANCHOR COMPENSATION the shell applies.
///
/// The resolve re-anchors a profile to its bbox minimum, so an edit at the profile's extreme
/// moves the node's whole emitted grid unless the offset absorbs the delta
/// ([`SketchSolid::anchor_preserving_offset`]). Without it every piece genuinely moves in world
/// and dirtying all of them is the CORRECT answer — so a test that skipped the compensation would
/// be measuring the anchor policy, not the broadphase.
fn scene_after_edit(edited: Sketch, previous: &SketchSolid) -> Scene {
    let producer = SketchSolid::extrude(edited, DEPTH_VOXELS);
    let offset = producer.anchor_preserving_offset(previous, [0; 3], context());
    scene_at(producer, offset)
}

/// Widen the circle centered at `center_x` — a change to exactly one shape.
fn with_widened_circle(sketch: &Sketch, center_x: i64) -> Sketch {
    let mut edited = sketch.clone();
    let target = edited
        .circles()
        .iter()
        .find(|circle| {
            edited
                .points()
                .iter()
                .any(|point| point.id == circle.center && point.at.in_plane()[0] == center_x as f64)
        })
        .map(|circle| circle.id)
        .expect("a circle at that center");
    assert!(
        edited.set_circle_radius(target, SketchLength::new(12)),
        "a wider radius resolves"
    );
    edited
}

#[test]
fn editing_one_shape_dirties_only_that_shape() {
    let sketch = circle_row(8);
    let base = SketchSolid::extrude(sketch.clone(), DEPTH_VOXELS);
    let before = scene_of(base.clone()).build_leaf_spatial_index(SKETCH_DENSITY);
    // The LAST circle, so a whole-leaf entry would report a box spanning the full 8-shape row and
    // the difference between the two answers is unmissable.
    let after = scene_after_edit(with_widened_circle(&sketch, 7 * 40), &base)
        .build_leaf_spatial_index(SKETCH_DENSITY);

    let dirty = after
        .edit_aabb_since(&before)
        .expect("a sketch edit is localizable");
    let span_x = dirty.max[0] - dirty.min[0];
    // The row spans 7 gaps of 40 voxels plus the end circles, so a whole-profile answer is ~300
    // voxels wide. One widened circle reaches 24. The bound is deliberately loose — the claim
    // under test is "one shape, not the drawing", not an exact box.
    assert!(
        span_x < 64,
        "a one-shape edit should dirty a one-shape box, got {span_x} voxels across: {dirty:?}"
    );
}

#[test]
fn the_dirty_box_does_not_grow_with_the_rest_of_the_drawing() {
    // The same edit, made twice, against drawings that differ ONLY in how many untouched shapes
    // sit beside it. Anything that scales here is scaling with the drawing rather than the edit.
    let measure = |count: i64| {
        let sketch = circle_row(count);
        let base = SketchSolid::extrude(sketch.clone(), DEPTH_VOXELS);
        let before = scene_of(base.clone()).build_leaf_spatial_index(SKETCH_DENSITY);
        let after = scene_after_edit(with_widened_circle(&sketch, 0), &base)
            .build_leaf_spatial_index(SKETCH_DENSITY);
        let dirty = after
            .edit_aabb_since(&before)
            .expect("a sketch edit is localizable");
        dirty.max[0] - dirty.min[0]
    };
    assert_eq!(
        measure(2),
        measure(16),
        "the dirty box widened when untouched shapes were added beside the edit"
    );
}

#[test]
fn untouched_shapes_keep_their_entries_across_an_edit() {
    let sketch = circle_row(4);
    let base = SketchSolid::extrude(sketch.clone(), DEPTH_VOXELS);
    let before = scene_of(base.clone()).build_leaf_spatial_index(SKETCH_DENSITY);
    let after = scene_after_edit(with_widened_circle(&sketch, 0), &base)
        .build_leaf_spatial_index(SKETCH_DENSITY);

    assert_eq!(
        before.entries.len(),
        4,
        "four disjoint circles should be four broadphase entries, not one leaf"
    );
    assert_eq!(before.entries.len(), after.entries.len());
    // Three of the four entries must survive the edit BYTE-IDENTICAL — that identity is what
    // makes them cancel in the diff, and cancelling is the entire mechanism.
    let survivors = before
        .entries
        .iter()
        .filter(|entry| after.entries.contains(entry))
        .count();
    assert_eq!(
        survivors, 3,
        "every circle but the edited one should be unchanged in both box and fingerprint"
    );
}

#[test]
fn a_revolve_keeps_one_whole_profile_entry() {
    // A revolve carries each loop's cells around the axis, so a loop's in-plane box does not
    // bound where it lands and the split would under-dirty. It must refuse.
    let index = scene_of(SketchSolid::revolve(
        circle_row(4),
        RevolveAxis::InPlane0,
        360,
    ))
    .build_leaf_spatial_index(SKETCH_DENSITY);
    assert_eq!(
        index.entries.len(),
        1,
        "a revolved sketch must stay one conservative whole-leaf entry"
    );
}

#[test]
fn a_rotated_sketch_node_keeps_one_whole_profile_entry() {
    // A piece box is corner-anchored in the producer's own grid. Under a rotated node that box
    // would have to be placed THROUGH the rotation to still bound the leaf's cells, so the
    // refinement stands down rather than emit a box that does not contain what it claims.
    let mut node = Node::new(
        "Profile",
        NodeContent::SketchTool {
            producer: SketchSolid::extrude(circle_row(4), DEPTH_VOXELS),
            material: MaterialChoice::Stone,
        },
    );
    node.transform.rotation_quaternion =
        Some(glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4).to_array());
    let index = Scene::from_nodes(vec![node]).build_leaf_spatial_index(SKETCH_DENSITY);
    assert_eq!(
        index.entries.len(),
        1,
        "a rotated sketch node must stay one conservative whole-leaf entry"
    );
}
