//! ADR 0028 (#95) — the add-point / delete profile edits and their anchor compensation.
//!
//! These pin the PURE producer operations the sketch shell drives on a click: inserting a
//! vertex into an edge, removing one, and the node-offset compensation that keeps the rest of
//! the profile fixed when an edit moves the profile's bbox-minimum (the resolve re-anchors that
//! minimum to the node origin). The frame invariant these serve — that the un-edited handles do
//! not move in the render frame — is checked independently in `sketch_handles`.

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
    // Inserting after index 1 (the edge 1→2) lands the new point at index 2, between the old
    // vertices 1 and 2 — the edge is split, the loop order is preserved.
    let before = bracket();
    let after = before.with_point_inserted(1, SketchPoint::new(6, 3));
    let coords: Vec<[i64; 2]> = after.sketch.profile.iter().map(|p| p.offset_voxels).collect();
    assert_eq!(
        coords,
        [[2, 2], [6, 2], [6, 3], [6, 5], [2, 5]],
        "the new vertex splits edge 1→2, so it sits at index 2"
    );
    assert!(!before.sketch.profile.iter().any(|p| p.offset_voxels == [6, 3]), "the source is untouched");
}

#[test]
fn insert_past_the_end_appends() {
    // The closing edge is `last → first`; splitting it inserts after the last index, which
    // clamps to an append (the loop closes back to the first vertex either way).
    let before = bracket();
    let last = before.sketch.profile.len() - 1;
    let after = before.with_point_inserted(last, SketchPoint::new(4, 5));
    assert_eq!(after.sketch.profile.len(), 5);
    assert_eq!(
        after.sketch.profile.last().unwrap().offset_voxels,
        [4, 5],
        "splitting the closing edge appends the new vertex at the end"
    );
}

#[test]
fn remove_drops_the_indexed_vertex_and_out_of_range_is_a_noop() {
    let before = bracket();
    let after = before.with_point_removed(2);
    let coords: Vec<[i64; 2]> = after.sketch.profile.iter().map(|p| p.offset_voxels).collect();
    assert_eq!(coords, [[2, 2], [6, 2], [2, 5]], "index 2 ([6, 5]) is removed");
    assert_eq!(before.with_point_removed(99), before, "an out-of-range index changes nothing");
}

#[test]
fn removing_below_three_vertices_is_allowed_and_degenerate() {
    // The AC: a profile under three vertices resolves to nothing WITHOUT error. The edit itself
    // does not block it — degeneracy is the resolve's business.
    let profile = before_triangle();
    let after = profile.with_point_removed(0).with_point_removed(0);
    assert_eq!(after.sketch.profile.len(), 1, "two deletes leave one vertex");
    assert!(after.profile_bbox_min() == after.sketch.profile[0].offset_voxels || after.sketch.profile.is_empty());
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
