//! The derived region is cached, and the cache is never the reason an answer is wrong.
//!
//! Every mutator invalidates it because the cache compares the entity store it was derived from,
//! so these tests exercise the ways the store can move rather than the mutators one by one.
use super::ctx;

use super::*;

/// A square of `size` from the origin, as four points and four segments.
fn square(sketch: &mut Sketch, origin: i64, size: i64) -> [EntityId; 4] {
    let corners = [
        SketchPoint::new(origin, origin),
        SketchPoint::new(origin + size, origin),
        SketchPoint::new(origin + size, origin + size),
        SketchPoint::new(origin, origin + size),
    ];
    let ids = corners.map(|at| sketch.add_point(at));
    for index in 0..4 {
        sketch.connect(ids[index], ids[(index + 1) % 4]);
    }
    ids
}

/// The same store derives the same region, and asking twice is asking once.
#[test]
fn a_repeated_question_gets_the_same_answer() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    assert_eq!(sketch.region(ctx(16)), sketch.region(ctx(16)));
    assert_eq!(
        sketch.region_field_loops(ctx(16)),
        sketch.region_field_loops(ctx(16))
    );
}

/// Drawing an entity after the region has been derived changes the region.
#[test]
fn a_drawn_curve_after_a_query_is_seen() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    assert_eq!(sketch.region(ctx(16)).len(), 1);
    square(&mut sketch, 4, 4);
    assert_eq!(sketch.region(ctx(16)).len(), 2);
}

/// Moving a point after the region has been derived moves the region with it. The topology is
/// untouched, so nothing but the coordinate comparison can catch this one.
#[test]
fn a_dragged_point_after_a_query_is_seen() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corners = square(&mut sketch, 0, 12);
    let before = sketch.filled_extent(ctx(16)).expect("a filled region");
    sketch
        .move_point(corners[2], SketchPoint::new(20, 20), ctx(16))
        .expect("evaluation context");
    let after = sketch.filled_extent(ctx(16)).expect("a filled region");
    assert_ne!(before, after);
    assert_eq!(after.1, [20.0, 20.0]);
}

/// Deleting an entity after the region has been derived takes its face with it.
#[test]
fn a_deleted_entity_after_a_query_is_seen() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    let inner = square(&mut sketch, 4, 4);
    assert_eq!(sketch.region(ctx(16)).len(), 2);
    sketch.delete_point_cascade(inner[0]);
    assert_eq!(sketch.region(ctx(16)).len(), 1);
}

/// Carving a face after the region has been derived flips that loop's role.
#[test]
fn an_unpick_after_a_query_is_seen() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    square(&mut sketch, 4, 4);
    assert!(sketch
        .region(ctx(16))
        .iter()
        .all(|profile_loop| profile_loop.role == LoopRole::Fill));
    let inner = sketch
        .identified_faces(ctx(16))
        .into_iter()
        .min_by(|first, second| first.0.area_voxels.total_cmp(&second.0.area_voxels))
        .expect("a face")
        .1;
    sketch.set_face_picked(inner, false, ctx(16));
    assert_eq!(
        sketch
            .region(ctx(16))
            .iter()
            .filter(|profile_loop| profile_loop.role == LoopRole::Hole)
            .count(),
        1
    );
}

/// A clone carries the entities, not the cache — and derives the same region from them.
#[test]
fn a_clone_derives_the_same_region() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    square(&mut sketch, 0, 12);
    let _ = sketch.region(ctx(16));
    let copy = sketch.clone();
    assert_eq!(copy, sketch);
    assert_eq!(copy.region(ctx(16)), sketch.region(ctx(16)));
}
