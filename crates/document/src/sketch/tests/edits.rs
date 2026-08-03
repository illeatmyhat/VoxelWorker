//! The id-based add-point / delete entity edits and their anchor compensation.
//!
//! These pin the PURE producer operations the sketch shell drives on a click: splitting a
//! segment by id (add-point), deleting a point by id (which removes its incident edges and nothing
//! else), and the node-offset compensation that keeps the rest of the profile fixed when an edit
//! moves the profile's bbox-minimum (the resolve re-anchors that minimum to the node origin).
//! Everything is keyed by stable `EntityId`, never a loop position — an open graph has no valid
//! loop index. The profile is DERIVED from the entity store via `flattened_loop`, which is empty
//! unless a closed loop exists.

use super::ctx;
use crate::sketch::{
    EntityId, EntityRole, PlaneAxis, PointLifetime, Sketch, SketchLength, SketchPoint, SketchSolid,
};

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

/// The id of the point at in-plane voxel coord `at`.
fn point_id_at(solid: &SketchSolid, at: [i64; 2]) -> EntityId {
    solid
        .sketch
        .points()
        .iter()
        .find(|point| point.at.offset_voxels == at)
        .unwrap_or_else(|| panic!("no point at {at:?}"))
        .id
}

/// The id of the segment joining the points at coords `a` and `b` (either direction).
fn segment_id_between(solid: &SketchSolid, a: [i64; 2], b: [i64; 2]) -> EntityId {
    let (ia, ib) = (point_id_at(solid, a), point_id_at(solid, b));
    solid
        .sketch
        .segments()
        .iter()
        .find(|seg| (seg.from == ia && seg.to == ib) || (seg.from == ib && seg.to == ia))
        .unwrap_or_else(|| panic!("no segment between {a:?} and {b:?}"))
        .id
}

#[test]
fn split_inserts_a_vertex_on_the_named_segment() {
    let before = bracket();
    let seg = segment_id_between(&before, [6, 2], [6, 5]);
    let after = before.with_point_on_segment(seg, SketchPoint::new(6, 3));
    let coords: Vec<[i64; 2]> = after
        .sketch
        .flattened_loop(ctx(16))
        .iter()
        .map(|p| p.offset_voxels)
        .collect();
    assert_eq!(
        coords,
        [[2, 2], [6, 2], [6, 3], [6, 5], [2, 5]],
        "the new vertex splits the named edge, so it sits between its endpoints in the loop"
    );
    assert!(
        !before
            .sketch
            .flattened_loop(ctx(16))
            .iter()
            .any(|p| p.offset_voxels == [6, 3]),
        "the source is untouched"
    );
}

#[test]
fn split_of_an_unknown_segment_is_a_noop() {
    let before = bracket();
    assert_eq!(
        before.with_point_on_segment(9999, SketchPoint::new(6, 3)),
        before,
        "an unknown segment id changes nothing"
    );
}

#[test]
fn delete_removes_the_point_and_cascades_only_its_segments() {
    // Deleting the point at [6, 5] removes it and its TWO incident segments — and nothing else.
    // The two neighbors survive as free points, so the loop opens and resolves to nothing
    // (flattened_loop is empty for an open graph, never a phantom polygon).
    let before = bracket();
    let victim = point_id_at(&before, [6, 5]);
    let after = before.with_point_deleted(victim);
    assert_eq!(
        after.sketch.points().len(),
        3,
        "exactly one point is removed"
    );
    assert!(
        !after
            .sketch
            .points()
            .iter()
            .any(|p| p.at.offset_voxels == [6, 5]),
        "the deleted point is gone"
    );
    assert_eq!(
        after.sketch.segments().len(),
        2,
        "only its two incident segments cascade away — the other two remain"
    );
    assert!(
        after.sketch.flattened_loop(ctx(16)).is_empty(),
        "the loop is open ⇒ no closed region ⇒ resolves to nothing"
    );
    assert_eq!(
        before.with_point_deleted(9999),
        before,
        "an unknown point id changes nothing"
    );
}

#[test]
fn deleting_every_point_leaves_an_empty_sketch() {
    // Deletes never error and never touch an unrelated entity. Deleting all three points of a
    // triangle one by one leaves nothing, resolving to nothing throughout.
    let triangle = before_triangle();
    let a = point_id_at(&triangle, [0, 0]);
    let b = point_id_at(&triangle, [4, 0]);
    let c = point_id_at(&triangle, [0, 4]);
    let after = triangle
        .with_point_deleted(a)
        .with_point_deleted(b)
        .with_point_deleted(c);
    assert_eq!(after.sketch.points().len(), 0, "every point is gone");
    assert!(
        after.sketch.segments().is_empty(),
        "no dangling segment remains"
    );
    assert!(
        after.sketch.flattened_loop(ctx(16)).is_empty(),
        "no loop remains"
    );
    let _ = after.profile_bbox_min(ctx(16)); // well-defined on an empty sketch (no panic)
}

#[test]
fn deleting_a_segment_removes_only_the_line() {
    // Deleting a line removes only that segment; its endpoints survive as free points.
    let before = bracket();
    let seg = segment_id_between(&before, [6, 2], [6, 5]);
    let after = before.with_segment_deleted(seg);
    assert_eq!(
        after.sketch.points().len(),
        4,
        "all four points survive as free points"
    );
    assert_eq!(
        after.sketch.segments().len(),
        3,
        "only the one line is removed"
    );
    assert!(
        after.sketch.flattened_loop(ctx(16)).is_empty(),
        "the loop is open ⇒ resolves to nothing"
    );
}

fn before_triangle() -> SketchSolid {
    let profile = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(4, 0),
        SketchPoint::new(0, 4),
    ];
    SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), 3)
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
    let dropped = solid.sketch.repair(ctx(16));
    assert_eq!(
        dropped, 2,
        "both the dangling reference and the self-loop are erased"
    );
    assert_eq!(
        solid.sketch.segments().len(),
        4,
        "the four valid segments remain"
    );
    assert_eq!(
        solid.sketch.flattened_loop(ctx(16)).len(),
        4,
        "the loop still closes"
    );
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
    // resolve extent stays sound: a malformed store never hard-fails the load.
    let _ = solid.sketch.flattened_loop(ctx(16));
    let _ = solid.grid_dimensions(ctx(16));
}

#[test]
fn anchor_offset_absorbs_a_bbox_min_shift_on_the_in_plane_axes_only() {
    // Splitting an edge with a vertex BELOW the current bbox-minimum (in both in-plane axes)
    // moves the minimum from [2, 2] to [0, 1]; the compensated offset must shift by exactly that
    // delta on the plane's in-plane axes (X, Y for PlaneAxis::Z) and never on the normal (Z).
    let before = bracket();
    let seg = segment_id_between(&before, [2, 2], [6, 2]);
    let after = before.with_point_on_segment(seg, SketchPoint::new(0, 1));
    let offset = after.anchor_preserving_offset(&before, [10, 10, 10], ctx(16));
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
    let seg = segment_id_between(&before, [6, 2], [6, 5]);
    let after = before.with_point_on_segment(seg, SketchPoint::new(4, 3));
    assert_eq!(
        after.anchor_preserving_offset(&before, [10, 10, 10], ctx(16)),
        [10, 10, 10],
        "an interior edit leaves the anchor — and so the offset — where it was"
    );
}

#[test]
fn construction_toggle_retains_identity_and_excludes_geometry_from_regions() {
    let before = before_triangle();
    let edge = before.sketch.segments()[0].id;
    let origin = before.sketch.segments()[0].origin;

    let construction = before
        .with_construction_toggled([edge])
        .expect("a selected edge toggles");
    assert_eq!(
        construction.sketch.segments()[0].role,
        EntityRole::Construction
    );
    assert_eq!(construction.sketch.segments()[0].id, edge);
    assert_eq!(construction.sketch.segments()[0].origin, origin);
    assert!(construction.sketch.flattened_loop(ctx(16)).is_empty());

    let real = construction
        .with_construction_toggled([edge])
        .expect("the same edge toggles back");
    assert_eq!(real.sketch.segments()[0].role, EntityRole::Real);
    assert_eq!(real.sketch.flattened_loop(ctx(16)).len(), 3);
}

/// Construction is a mode a CURVE is in. Neither a point the drawing owns nor one the author
/// placed has a linetype to flip, so both refuse — and refusing the center is what keeps a
/// structural one from being talked into participating as a profile vertex.
#[test]
fn no_point_takes_a_construction_toggle_structural_or_free() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let circle = sketch
        .add_circle(SketchPoint::new(2, 3), SketchLength::new(4))
        .unwrap();
    let center = sketch
        .circles()
        .iter()
        .find(|held| held.id == circle)
        .unwrap()
        .center;
    let free = sketch.add_free_point(SketchPoint::new(9, 9));
    let source = SketchSolid::extrude(sketch, 3);
    let role_of = |id| {
        source
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .unwrap()
            .lifetime
    };

    assert!(source.with_construction_toggled([center]).is_none());
    assert!(source.with_construction_toggled([free]).is_none());
    assert!(source.with_construction_toggled([center, free]).is_none());
    assert_eq!(role_of(center), PointLifetime::CurveAnchored);
    assert_eq!(role_of(free), PointLifetime::Freestanding);
}

/// Every curve kind answers the same door. `set_construction` used to reach segments only and
/// silently drop the other six, so a tool that authored reference geometry got a real boundary.
#[test]
fn set_construction_reaches_every_curve_kind() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let circle = sketch
        .add_circle(SketchPoint::new(2, 3), SketchLength::new(4))
        .unwrap();
    let conic = sketch
        .add_conic(
            SketchPoint::new(-4, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(0, 4),
            0.5,
        )
        .unwrap();
    let spline = sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 5),
            SketchPoint::new(6, 5),
            SketchPoint::new(8, 0),
        ])
        .unwrap();

    for id in [circle, conic, spline] {
        sketch.set_construction(id);
    }

    assert_eq!(sketch.circles()[0].role, EntityRole::Construction);
    assert_eq!(sketch.conics()[0].role, EntityRole::Construction);
    assert_eq!(sketch.splines()[0].role, EntityRole::Construction);
    assert!(sketch.faces(ctx(16)).is_empty());
}
