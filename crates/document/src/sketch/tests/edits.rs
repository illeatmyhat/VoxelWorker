//! ADR 0030 (#98) — the add-point / delete entity edits and their anchor compensation.
//!
//! These pin the PURE producer operations the sketch shell drives on a click: inserting a
//! vertex on a loop edge (splitting a segment), deleting a vertex (cascading to its incident
//! segments — ADR 0030 §6, superseding #95's loop reclose), and the node-offset compensation
//! that keeps the rest of the profile fixed when an edit moves the profile's bbox-minimum (the
//! resolve re-anchors that minimum to the node origin). The frame invariant these serve — that
//! the un-edited handles do not move in the render frame — is checked independently in
//! `sketch_handles`. The profile here is DERIVED from the entity store via `flattened_loop`.

use crate::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};

/// A closed rectangular profile whose bbox-minimum is `[2, 2]`, extruded so it is a real solid.
fn bracket() -> SketchSolid {
    let profile = vec![
        SketchPoint::new(2, 2),
        SketchPoint::new(6, 2),
        SketchPoint::new(6, 5),
        SketchPoint::new(2, 5),
    ];
    SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), 3)
}

#[test]
fn insert_splits_the_edge_after_its_start_vertex() {
    // Inserting after loop position 1 (the edge 1→2) lands the new point between the old
    // vertices 1 and 2 — the segment is split and the loop order is preserved.
    let before = bracket();
    let after = before.with_point_inserted(1, SketchPoint::new(6, 3));
    let coords: Vec<[i64; 2]> = after
        .sketch
        .flattened_loop()
        .iter()
        .map(|p| p.offset_voxels)
        .collect();
    assert_eq!(
        coords,
        [[2, 2], [6, 2], [6, 3], [6, 5], [2, 5]],
        "the new vertex splits edge 1→2, so it sits between them in the loop"
    );
    assert!(
        !before
            .sketch
            .flattened_loop()
            .iter()
            .any(|p| p.offset_voxels == [6, 3]),
        "the source is untouched"
    );
}

#[test]
fn insert_past_the_end_appends() {
    // The closing edge is `last → first`; splitting it inserts a vertex on it, which the loop
    // walk places at the end (the loop closes back to the first vertex either way).
    let before = bracket();
    let last = before.sketch.flattened_loop().len() - 1;
    let after = before.with_point_inserted(last, SketchPoint::new(4, 5));
    let coords = after.sketch.flattened_loop();
    assert_eq!(coords.len(), 5);
    assert_eq!(
        coords.last().unwrap().offset_voxels,
        [4, 5],
        "splitting the closing edge appends the new vertex at the loop's end"
    );
}

#[test]
fn delete_removes_the_indexed_vertex_and_cascades_its_segments() {
    // ADR 0030 §6: deleting a loop vertex removes the POINT and its incident segments — it does
    // NOT reclose the loop (superseding #95). Deleting flattened index 2 ([6, 5]) drops that
    // point and its two edges, opening the loop (which then resolves to nothing).
    let before = bracket();
    let after = before.with_point_removed(2);
    assert_eq!(after.sketch.points().len(), 3, "the vertex is gone");
    assert!(
        !after
            .sketch
            .points()
            .iter()
            .any(|p| p.at.offset_voxels == [6, 5]),
        "the deleted vertex ([6, 5]) is removed"
    );
    assert_eq!(
        after.sketch.segments().len(),
        2,
        "its two incident segments cascade away"
    );
    assert!(
        after.sketch.flattened_loop().len() < 3,
        "the loop is open ⇒ resolves to nothing"
    );
    assert_eq!(
        before.with_point_removed(99),
        before,
        "an out-of-range index changes nothing"
    );
}

#[test]
fn deleting_down_to_a_lone_point_is_allowed_and_degenerate() {
    // Deletes never error; a sketch under a closed triangle simply resolves to nothing. Two
    // cascade-deletes off a triangle leave a single free point and no loop.
    let triangle = before_triangle();
    let after = triangle.with_point_removed(0).with_point_removed(0);
    assert_eq!(after.sketch.points().len(), 1, "two deletes leave one free point");
    assert!(
        after.sketch.flattened_loop().len() < 3,
        "no closed loop remains"
    );
    // A degenerate bbox-minimum is well-defined (no panic).
    let _ = after.profile_bbox_min();
}

fn before_triangle() -> SketchSolid {
    let profile = vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0), SketchPoint::new(0, 4)];
    SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), 3)
}

#[test]
fn anchor_offset_absorbs_a_bbox_min_shift_on_the_in_plane_axes_only() {
    // Inserting a vertex BELOW the current bbox-minimum (in both in-plane axes) moves the
    // minimum from [2, 2] to [0, 1]; the compensated offset must shift by exactly that delta on
    // the plane's in-plane axes (X, Y for PlaneAxis::Z) and never on the normal axis (Z).
    let before = bracket();
    let after = before.with_point_inserted(0, SketchPoint::new(0, 1));
    let offset = after.anchor_preserving_offset(&before, [10, 10, 10]);
    assert_eq!(
        offset,
        [8, 9, 10],
        "offset shifts by the bbox-min delta [-2, -1] on X, Y; Z (the normal) is untouched"
    );
}

#[test]
fn repair_erases_dangling_and_self_segments() {
    use crate::sketch::{EntityRole, Segment};
    let mut solid = bracket(); // 4 points, 4 valid segments
    // A segment to a non-existent point, and a degenerate self-loop.
    solid.sketch.segments_mut_for_test().push(Segment {
        id: 100,
        from: 0,
        to: 9999,
        origin: 100,
        role: EntityRole::Real,
    });
    solid.sketch.segments_mut_for_test().push(Segment {
        id: 101,
        from: 1,
        to: 1,
        origin: 101,
        role: EntityRole::Real,
    });
    let dropped = solid.sketch.repair();
    assert_eq!(dropped, 2, "both the dangling reference and the self-loop are erased");
    assert_eq!(solid.sketch.segments().len(), 4, "the four valid segments remain");
    assert_eq!(solid.sketch.flattened_loop().len(), 4, "the loop still closes");
}

#[test]
fn resolve_tolerates_a_dangling_segment_without_panic() {
    use crate::sketch::{EntityRole, Segment};
    let mut solid = bracket();
    solid.sketch.segments_mut_for_test().push(Segment {
        id: 100,
        from: 0,
        to: 9999,
        origin: 100,
        role: EntityRole::Real,
    });
    // Deriving the loop must not panic — the missing vertex is simply filtered out — and the
    // resolve extent stays sound (a hard load failure never happens, ADR 0030).
    let _ = solid.sketch.flattened_loop();
    let _ = solid.grid_dimensions();
}

#[test]
fn anchor_offset_is_unchanged_when_the_edit_stays_inside_the_bbox() {
    // A vertex added inside the existing bounds does not move the minimum, so nothing to absorb.
    let before = bracket();
    let after = before.with_point_inserted(1, SketchPoint::new(4, 3));
    assert_eq!(
        after.anchor_preserving_offset(&before, [10, 10, 10]),
        [10, 10, 10],
        "an interior edit leaves the anchor — and so the offset — where it was"
    );
}
