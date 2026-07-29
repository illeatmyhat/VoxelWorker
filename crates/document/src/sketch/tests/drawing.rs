//! #99 — the drawing-tool store mutators: free points, `connect`, coincidence via `point_at`,
//! and the pure `with_point_placed` / `with_segment_between` / `with_rectangle` wrappers the
//! polyline and rectangle gestures commit through. Coincidence IS shared point identity (ADR
//! 0030): placing on an occupied coord reuses the id, never mints a twin.

use crate::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};

fn empty_solid() -> SketchSolid {
    SketchSolid::extrude(Sketch::new(PlaneAxis::Z, vec![]), 3)
}

#[test]
fn connect_rejects_self_loop_unknown_and_duplicate() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    assert_eq!(sketch.connect(a, a), None, "a self-loop is refused");
    assert_eq!(
        sketch.connect(a, 9999),
        None,
        "an unknown endpoint is refused"
    );
    assert!(sketch.connect(a, b).is_some(), "a fresh pair connects");
    assert_eq!(
        sketch.connect(b, a),
        None,
        "the same pair is refused in either direction"
    );
    assert_eq!(sketch.segments().len(), 1, "exactly one segment exists");
}

#[test]
fn point_at_finds_only_an_exact_coincidence() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(2, 3));
    assert_eq!(sketch.point_at(SketchPoint::new(2, 3)), Some(a));
    assert_eq!(
        sketch.point_at(SketchPoint::new(2, 4)),
        None,
        "a neighbouring coord is not a hit — coincidence is exact, proximity lives in the shell"
    );
}

#[test]
fn with_point_placed_reuses_the_occupied_coord() {
    let (one, first) = empty_solid().with_point_placed(SketchPoint::new(1, 1));
    let (two, second) = one.with_point_placed(SketchPoint::new(1, 1));
    assert_eq!(first, second, "the occupied coord answers with the SAME id");
    assert_eq!(two.sketch.points().len(), 1, "no twin point is minted");
    let (three, third) = two.with_point_placed(SketchPoint::new(5, 1));
    assert_ne!(third, first);
    assert_eq!(three.sketch.points().len(), 2);
}

#[test]
fn with_segment_between_tolerates_a_dead_reference() {
    let (solid, id) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    assert_eq!(
        solid.with_segment_between(id, 9999),
        solid,
        "a dead endpoint (mid-gesture delete) is a no-op, never a panic"
    );
    assert_eq!(
        solid.with_segment_between(id, id),
        solid,
        "so is a self-loop"
    );
}

#[test]
fn with_rectangle_closes_a_four_point_loop() {
    let after = empty_solid().with_rectangle(SketchPoint::new(1, 1), SketchPoint::new(4, 3));
    assert_eq!(after.sketch.points().len(), 4);
    assert_eq!(after.sketch.segments().len(), 4);
    let coords: std::collections::BTreeSet<[i64; 2]> = after
        .sketch
        .flattened_loop()
        .iter()
        .map(|p| p.offset_voxels)
        .collect();
    assert_eq!(
        coords,
        [[1, 1], [4, 1], [4, 3], [1, 3]].into_iter().collect(),
        "the four corners close into a real loop — the profile resolves"
    );
}

#[test]
fn with_rectangle_reuses_coincident_corners() {
    // Drawing a second rectangle sharing an edge with the first reuses the shared corners and
    // never doubles the shared segment.
    let one = empty_solid().with_rectangle(SketchPoint::new(0, 0), SketchPoint::new(4, 3));
    let two = one.with_rectangle(SketchPoint::new(4, 0), SketchPoint::new(8, 3));
    assert_eq!(
        two.sketch.points().len(),
        6,
        "the two shared corners are reused"
    );
    assert_eq!(
        two.sketch.segments().len(),
        7,
        "the shared edge exists once, never doubled"
    );
}

#[test]
fn with_rectangle_refuses_a_zero_span() {
    let before = empty_solid();
    assert_eq!(
        before.with_rectangle(SketchPoint::new(2, 2), SketchPoint::new(2, 5)),
        before,
        "a degenerate rectangle (zero span on an axis) changes nothing"
    );
    assert_eq!(
        before.with_rectangle(SketchPoint::new(2, 2), SketchPoint::new(2, 2)),
        before
    );
}
