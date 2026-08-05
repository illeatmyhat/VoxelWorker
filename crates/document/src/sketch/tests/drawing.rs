//! The drawing-tool store mutators: free points, `connect`, coincidence via `point_at`, and the
//! pure `with_point_placed` / `with_segment_between` / `with_rectangle` wrappers the Line and
//! rectangle gestures commit through. Coincidence IS shared point identity: placing on an occupied
//! coord reuses the id, never mints a twin.

use super::ctx;
use crate::sketch::{
    CenterArcRefusal, ConstraintKind, MidpointLineRefusal, PlaneAxis, Sketch, SketchCurve,
    SketchLength, SketchPoint, SketchPointConstructionError, SketchSolid, TangentArcRefusal,
};
use parametric::units::{AngleMeasurement, Measurement};

fn empty_solid() -> SketchSolid {
    SketchSolid::extrude(Sketch::new(PlaneAxis::Z, vec![]), 3)
}

#[test]
fn center_rectangle_preview_and_commit_share_reflected_corners() {
    let solid = empty_solid();
    let center = SketchPoint::from_continuous(2.5, 3.25);
    let corner = SketchPoint::from_continuous(5.75, 8.0);
    let placement = solid.center_rectangle_placement(center, corner).unwrap();
    let made = solid
        .with_center_rectangle(center, corner, ctx(16))
        .unwrap();
    // Four boundary sides plus the two construction diagonals the center hangs from.
    assert_eq!(made.sketch.segments().len(), 6);
    for point in placement.corners {
        assert!(made.sketch.point_at(point).is_some());
    }
    let a = placement.corners[0].in_plane();
    let c = placement.corners[2].in_plane();
    assert!(((a[0] + c[0]) / 2.0 - center.in_plane()[0]).abs() < 1e-6);
    assert!(((a[1] + c[1]) / 2.0 - center.in_plane()[1]).abs() < 1e-6);
}

#[test]
fn three_point_rectangle_projects_width_and_commits_atomically() {
    let solid = empty_solid();
    let placement = solid
        .three_point_rectangle_placement(
            SketchPoint::new(0, 0),
            SketchPoint::new(3, 4),
            SketchPoint::new(0, 5),
        )
        .unwrap();
    let made = solid
        .with_three_point_rectangle(
            SketchPoint::new(0, 0),
            SketchPoint::new(3, 4),
            SketchPoint::new(0, 5),
            ctx(16),
        )
        .unwrap();
    assert_eq!(made.sketch.points().len(), 4);
    assert_eq!(made.sketch.segments().len(), 4);
    // Asserting perpendicularity settles the corners, which quantization left a hair off square.
    // The nudge is bounded well under the 1/256-block quantum the profile flattens to, so it is
    // invisible in resolved occupancy — but it is real, so the corners are compared with a
    // tolerance rather than by exact coincidence. The axis-aligned constructions do not move at
    // all: their corners already satisfy Horizontal and Vertical exactly.
    for corner in placement.corners {
        let corner = corner.in_plane();
        assert!(
            made.sketch.points().iter().any(|point| {
                let at = point.at.in_plane();
                (at[0] - corner[0]).abs() < 1e-3 && (at[1] - corner[1]).abs() < 1e-3
            }),
            "no stored corner near {corner:?}"
        );
    }
    let base = [3.0, 4.0];
    let side = [
        placement.corners[3].in_plane()[0] - placement.corners[0].in_plane()[0],
        placement.corners[3].in_plane()[1] - placement.corners[0].in_plane()[1],
    ];
    assert!(base[0] * side[0] + base[1] * side[1] < 1e-5);
    assert!(solid.sketch.points().is_empty(), "source remains untouched");
}

#[test]
fn center_arc_projects_the_end_direction_and_keeps_the_center_derived() {
    let solid = empty_solid();
    let center = SketchPoint::new(0, 0);
    let start = SketchPoint::new(4, 0);
    let direction = SketchPoint::new(0, 9);
    let placement = solid
        .center_arc_placement(
            center,
            start,
            None,
            direction,
            parametric::sketch::ArcTurn::CounterClockwise,
        )
        .unwrap();
    assert!(placement.endpoint.coincides(&SketchPoint::new(0, 4)));
    assert_eq!(placement.candidate.radius, 4.0);
    assert!((placement.candidate.sweep_radians.to_degrees() - 90.0).abs() < 1e-12);

    let (made, arc_id) = solid
        .with_center_arc(
            center,
            start,
            None,
            direction,
            parametric::sketch::ArcTurn::CounterClockwise,
        )
        .unwrap();
    assert_eq!(
        made.sketch.points().len(),
        3,
        "the derived center is reified"
    );
    let arc = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.id == arc_id)
        .unwrap();
    assert!(made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == arc.to)
        .unwrap()
        .at
        .coincides(&placement.endpoint));
    assert!(made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == arc.center)
        .unwrap()
        .at
        .coincides(&center));
}

#[test]
fn center_arc_reuses_a_stored_start_and_refuses_without_mutating() {
    let (solid, start) = empty_solid().with_point_placed(SketchPoint::new(4, 0));
    let (made, arc_id) = solid
        .with_center_arc(
            SketchPoint::new(0, 0),
            SketchPoint::new(999, 999),
            Some(start),
            SketchPoint::new(0, -8),
            parametric::sketch::ArcTurn::CounterClockwise,
        )
        .unwrap();
    assert_eq!(
        made.sketch
            .arcs()
            .iter()
            .find(|arc| arc.id == arc_id)
            .unwrap()
            .from,
        start
    );
    assert!(
        (made.sketch.arcs()[0]
            .bulge
            .free_value()
            .unwrap()
            .to_degrees_f64()
            - 270.0)
            .abs()
            < 1e-12
    );

    let before = serde_json::to_string(&solid).unwrap();
    assert_eq!(
        solid.with_center_arc(
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 0),
            None,
            SketchPoint::new(1, 0),
            parametric::sketch::ArcTurn::CounterClockwise,
        ),
        Err(CenterArcRefusal::Candidate(
            parametric::sketch::CenterArcCandidateError::CollapsedRadius
        ))
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
}

#[test]
fn center_arc_preview_matches_persisted_geometry_after_endpoint_narrowing() {
    let solid = empty_solid();
    let placement = solid
        .center_arc_placement(
            SketchPoint::from_continuous(0.25, -0.5),
            SketchPoint::from_continuous(4.125, 0.75),
            None,
            SketchPoint::from_continuous(2.7, 8.9),
            parametric::sketch::ArcTurn::CounterClockwise,
        )
        .unwrap();
    let (made, arc) = solid
        .with_center_arc(
            SketchPoint::from_continuous(0.25, -0.5),
            SketchPoint::from_continuous(4.125, 0.75),
            None,
            SketchPoint::from_continuous(2.7, 8.9),
            parametric::sketch::ArcTurn::CounterClockwise,
        )
        .unwrap();
    let parametric::sketch::CurveGeometry::Circular(persisted) = made
        .sketch
        .curve_geometry(SketchCurve::Arc(arc), ctx(16))
        .unwrap()
    else {
        panic!("arc geometry")
    };
    assert!((persisted.center[0] - placement.candidate.center[0]).abs() < 1e-10);
    assert!((persisted.center[1] - placement.candidate.center[1]).abs() < 1e-10);
    assert!((persisted.radius - placement.candidate.radius).abs() < 1e-10);
    assert!(
        (persisted.arc.unwrap().sweep_radians - placement.candidate.sweep_radians).abs() < 1e-10
    );
    let stored_arc = made
        .sketch
        .arcs()
        .iter()
        .find(|candidate| candidate.id == arc)
        .unwrap();
    assert!(made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == stored_arc.center)
        .unwrap()
        .at
        .coincides(&placement.center));
}

#[test]
fn checked_continuous_points_are_canonical_and_round_trip() {
    for [x, y] in [[2.25, -3.75], [-0.125, 0.5], [0.0, 0.0]] {
        let point = SketchPoint::try_from_continuous(x, y).unwrap();
        assert!(point
            .offset_local_voxels
            .into_iter()
            .all(|local| (0.0..1.0).contains(&local)));
        assert_eq!(
            SketchPoint::try_from_continuous(point.in_plane()[0], point.in_plane()[1]).unwrap(),
            point
        );
    }
}

#[test]
fn checked_continuous_points_handle_the_asymmetric_i64_f64_bounds() {
    let lower = i64::MIN as f64;
    let upper = -(i64::MIN as f64);
    let just_below_upper = f64::from_bits(upper.to_bits() - 1);
    let just_below_lower = f64::from_bits(lower.to_bits() + 1);

    assert_eq!(
        SketchPoint::try_from_continuous(lower, just_below_upper)
            .unwrap()
            .offset_voxels,
        [i64::MIN, just_below_upper as i64]
    );
    assert_eq!(
        SketchPoint::try_from_continuous(upper, 0.0),
        Err(SketchPointConstructionError::OutOfCanonicalRange),
        "`i64::MAX as f64` is 2^63 and must not be accepted by a saturating cast"
    );
    assert_eq!(
        SketchPoint::try_from_continuous(just_below_lower, 0.0),
        Err(SketchPointConstructionError::OutOfCanonicalRange)
    );
}

#[test]
fn checked_continuous_points_distinguish_nonfinite_and_range_errors() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            SketchPoint::try_from_continuous(bad, 0.0),
            Err(SketchPointConstructionError::NonFinite)
        );
    }
    assert_eq!(
        SketchPoint::try_from_continuous(f64::MAX, 0.0),
        Err(SketchPointConstructionError::OutOfCanonicalRange)
    );
}

#[test]
fn fractional_narrowing_carries_one_and_refuses_carry_overflow() {
    let rounds_to_one = 1.0 - f64::EPSILON;
    assert_eq!(
        SketchPoint::finish_continuous_split(4, rounds_to_one),
        Ok((5, 0.0))
    );
    assert_eq!(
        SketchPoint::finish_continuous_split(i64::MAX, rounds_to_one),
        Err(SketchPointConstructionError::OutOfCanonicalRange)
    );
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
        "a neighboring coord is not a hit — coincidence is exact, proximity lives in the shell"
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
fn midpoint_line_preview_is_the_exact_document_geometry_that_commit_stores() {
    let empty = empty_solid();
    let placement = empty
        .midpoint_line_placement([5.25, -1.5], [8.75, 3.125], None)
        .unwrap();
    let (made, segment_id) = empty
        .with_midpoint_line([5.25, -1.5], [8.75, 3.125], None)
        .unwrap();
    let segment = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.id == segment_id)
        .unwrap();
    let at = |id| {
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .unwrap()
            .at
    };

    assert_eq!(at(segment.from), placement.endpoint);
    assert_eq!(at(segment.to), placement.reflected);
    assert_eq!(made.sketch.points().len(), 2);
    assert_eq!(made.sketch.segments().len(), 1);
    assert!(made.sketch.constraints().is_empty());
    assert_eq!(made.sketch.point_at(placement.midpoint), None);
}

#[test]
fn midpoint_line_reuses_clicked_reflected_and_both_endpoint_ids() {
    let midpoint = [5.0, 0.0];
    let clicked_at = SketchPoint::new(8, 0);
    let reflected_at = SketchPoint::new(2, 0);

    let mut clicked_sketch = Sketch::empty(PlaneAxis::Z);
    let clicked = clicked_sketch.add_free_point(clicked_at);
    let clicked_solid = SketchSolid::extrude(clicked_sketch, 3);
    let (clicked_reused, segment) = clicked_solid
        .with_midpoint_line(midpoint, [999.0, 999.0], Some(clicked))
        .unwrap();
    assert_eq!(clicked_reused.sketch.segments()[0].id, segment);
    assert_eq!(clicked_reused.sketch.segments()[0].from, clicked);
    assert_eq!(clicked_reused.sketch.points().len(), 2);

    let mut reflected_sketch = Sketch::empty(PlaneAxis::Z);
    let reflected = reflected_sketch.add_free_point(reflected_at);
    let reflected_solid = SketchSolid::extrude(reflected_sketch, 3);
    let (reflected_reused, _) = reflected_solid
        .with_midpoint_line(midpoint, clicked_at.in_plane(), None)
        .unwrap();
    assert_eq!(reflected_reused.sketch.segments()[0].to, reflected);
    assert_eq!(reflected_reused.sketch.points().len(), 2);

    let mut both_sketch = Sketch::empty(PlaneAxis::Z);
    let clicked = both_sketch.add_free_point(clicked_at);
    let reflected = both_sketch.add_free_point(reflected_at);
    let both_solid = SketchSolid::extrude(both_sketch, 3);
    let (both_reused, _) = both_solid
        .with_midpoint_line(midpoint, clicked_at.in_plane(), Some(clicked))
        .unwrap();
    assert_eq!(both_reused.sketch.points().len(), 2, "no coordinate twins");
    assert_eq!(both_reused.sketch.segments()[0].from, clicked);
    assert_eq!(both_reused.sketch.segments()[0].to, reflected);

    let mut coordinate_sketch = Sketch::empty(PlaneAxis::Z);
    let clicked = coordinate_sketch.add_free_point(clicked_at);
    let coordinate_solid = SketchSolid::extrude(coordinate_sketch, 3);
    let (coordinate_reused, _) = coordinate_solid
        .with_midpoint_line(midpoint, clicked_at.in_plane(), None)
        .unwrap();
    assert_eq!(coordinate_reused.sketch.segments()[0].from, clicked);
    assert_eq!(coordinate_reused.sketch.points().len(), 2);
}

#[test]
fn midpoint_line_preview_matches_a_reused_reflected_position_not_its_provenance() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let mut reflected_at = SketchPoint::new(2, 0);
    reflected_at.offset_measurements =
        Some([Measurement::from_voxels(2), Measurement::from_voxels(0)]);
    let reflected = sketch.add_free_point(reflected_at);
    let source = SketchSolid::extrude(sketch, 3);

    let placement = source
        .midpoint_line_placement([5.0, 0.0], [8.0, 0.0], None)
        .unwrap();
    assert!(placement.reflected.coincides(&reflected_at));
    assert_ne!(
        placement.reflected, reflected_at,
        "retained measurements are provenance, not preview geometry"
    );

    let (made, segment_id) = source
        .with_midpoint_line([5.0, 0.0], [8.0, 0.0], None)
        .unwrap();
    let segment = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.id == segment_id)
        .unwrap();
    assert_eq!(
        segment.to, reflected,
        "the coincident stored point is reused"
    );
    let committed = made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == segment.to)
        .unwrap()
        .at;
    assert!(placement.reflected.coincides(&committed));
    assert_eq!(committed, reflected_at, "stored provenance remains intact");
}

#[test]
fn midpoint_line_accepts_extreme_finite_canonical_coordinates() {
    let upper = -(i64::MIN as f64);
    let midpoint = [upper - 4096.0, -4096.0];
    let endpoint = [upper - 3072.0, -3072.0];
    let (made, segment) = empty_solid()
        .with_midpoint_line(midpoint, endpoint, None)
        .unwrap();
    assert_eq!(made.sketch.segments()[0].id, segment);
    assert_eq!(made.sketch.points().len(), 2);
    assert!(made.sketch.points().iter().all(|point| point
        .at
        .in_plane()
        .into_iter()
        .all(f64::is_finite)));
}

#[test]
fn midpoint_line_avoids_large_coordinate_cancellation_and_keeps_valid_extremes() {
    let far = 2.0f64.powi(62);
    for midpoint in [[1.0, 0.0], [0.0, 0.0], [1024.0, -1024.0]] {
        let endpoint = [far, far];
        let placement = empty_solid()
            .midpoint_line_placement(midpoint, endpoint, None)
            .unwrap();
        assert!(placement
            .midpoint
            .is_exact_midpoint_of(&placement.endpoint, &placement.reflected));
    }
    let ordinary_large = empty_solid()
        .midpoint_line_placement([1.0, 0.0], [2.0f64.powi(52), 0.0], None)
        .unwrap();
    assert!(ordinary_large
        .midpoint
        .is_exact_midpoint_of(&ordinary_large.endpoint, &ordinary_large.reflected));
}

#[test]
fn midpoint_line_preserves_an_authoritative_large_split_endpoint() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let endpoint_at = SketchPoint {
        offset_voxels: [1_i64 << 62, -(1_i64 << 62)],
        offset_local_voxels: [0.5, 0.25],
        offset_measurements: None,
    };
    let endpoint = sketch.add_free_point(endpoint_at);
    let source = SketchSolid::extrude(sketch, 3);

    let placement = source
        .midpoint_line_placement([1.0, -1.0], [f64::NAN, f64::NAN], Some(endpoint))
        .unwrap();
    assert_eq!(placement.endpoint, endpoint_at);
    assert!(placement
        .midpoint
        .is_exact_midpoint_of(&placement.endpoint, &placement.reflected));

    let (made, segment) = source
        .with_midpoint_line([1.0, -1.0], [f64::NAN, f64::NAN], Some(endpoint))
        .unwrap();
    assert_eq!(made.sketch.segments()[0].id, segment);
    assert_eq!(made.sketch.segments()[0].from, endpoint);
    assert_eq!(
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == endpoint)
            .unwrap()
            .at,
        placement.endpoint
    );
}

#[test]
fn exact_split_reflection_normalizes_signs_and_refuses_both_range_edges() {
    let midpoint = SketchPoint {
        offset_voxels: [i64::MIN + 1, i64::MAX - 1],
        offset_local_voxels: [0.75, 0.25],
        offset_measurements: None,
    };
    let endpoint = SketchPoint {
        offset_voxels: [i64::MIN, i64::MAX],
        offset_local_voxels: [0.5, 0.5],
        offset_measurements: None,
    };
    let reflected = midpoint.exact_reflection_of(&endpoint).unwrap().unwrap();
    assert_eq!(
        reflected,
        SketchPoint {
            offset_voxels: [i64::MIN + 3, i64::MAX - 2],
            offset_local_voxels: [0.0, 0.0],
            offset_measurements: None,
        }
    );
    assert!(midpoint.is_exact_midpoint_of(&endpoint, &reflected));

    assert_eq!(
        SketchPoint::new(i64::MIN, 0).exact_reflection_of(&SketchPoint::new(i64::MIN + 1, 0)),
        Err(SketchPointConstructionError::OutOfCanonicalRange)
    );
    assert_eq!(
        SketchPoint::new(i64::MAX, 0).exact_reflection_of(&SketchPoint::new(i64::MAX - 1, 0)),
        Err(SketchPointConstructionError::OutOfCanonicalRange)
    );
}

#[test]
fn midpoint_line_refuses_a_fractional_reflection_that_f32_cannot_store_exactly() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let endpoint = sketch.add_free_point(SketchPoint {
        offset_voxels: [0, 0],
        offset_local_voxels: [f32::from_bits(1), 0.0],
        offset_measurements: None,
    });
    let source = SketchSolid::extrude(sketch, 3);

    assert_eq!(
        source.with_midpoint_line([0.5, 0.0], [0.0, 0.0], Some(endpoint)),
        Err(MidpointLineRefusal::CanonicalCollapse)
    );
}

#[test]
fn a_preexisting_midpoint_remains_independent_geometry_not_an_authored_input() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let midpoint_at = SketchPoint::new(5, 0);
    let midpoint_id = sketch.add_free_point(midpoint_at);
    let source = SketchSolid::extrude(sketch, 3);
    let (made, segment_id) = source
        .with_midpoint_line(midpoint_at.in_plane(), [8.0, 0.0], None)
        .unwrap();

    let segment = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.id == segment_id)
        .unwrap();
    assert_ne!(segment.from, midpoint_id);
    assert_ne!(segment.to, midpoint_id);

    assert_eq!(
        made.sketch.points().len(),
        3,
        "one old midpoint plus two ends"
    );
    assert_eq!(made.sketch.point_at(midpoint_at), Some(midpoint_id));
    assert_eq!(
        made.sketch
            .points()
            .iter()
            .filter(|point| point.at.coincides(&midpoint_at))
            .count(),
        1,
        "the construction input minted no midpoint twin"
    );
    assert_eq!(
        made.sketch
            .segments()
            .iter()
            .filter(|segment| segment.from == midpoint_id || segment.to == midpoint_id)
            .count(),
        0,
        "the transient midpoint has no incident segment"
    );
    assert_eq!(
        made.sketch
            .arcs()
            .iter()
            .filter(|arc| arc.from == midpoint_id || arc.to == midpoint_id)
            .count(),
        0,
        "the transient midpoint has no incident curve"
    );
    assert!(made.sketch.constraints().is_empty());
}

#[test]
fn midpoint_line_refuses_stale_duplicate_and_canonical_collapse_atomically() {
    let empty = empty_solid();
    let raw_midpoint = [0.25 + 1.0e-10, 0.0];
    let raw_endpoint = [0.25, 0.0];
    assert!(parametric::sketch::midpoint_line_candidate(raw_midpoint, raw_endpoint).is_ok());

    for refusal in [
        empty.with_midpoint_line([5.0, 0.0], [8.0, 0.0], Some(9999)),
        empty.with_midpoint_line(raw_midpoint, raw_endpoint, None),
    ] {
        assert!(matches!(
            refusal,
            Err(MidpointLineRefusal::UnknownEndpoint | MidpointLineRefusal::CanonicalCollapse)
        ));
    }

    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let endpoint = sketch.add_free_point(SketchPoint::new(8, 0));
    let reflected = sketch.add_free_point(SketchPoint::new(2, 0));
    sketch.connect(endpoint, reflected).unwrap();
    let duplicate = SketchSolid::extrude(sketch, 3);
    let before = serde_json::to_string(&duplicate).unwrap();
    assert_eq!(
        duplicate.with_midpoint_line([5.0, 0.0], [8.0, 0.0], Some(endpoint)),
        Err(MidpointLineRefusal::DuplicateSegment)
    );
    assert_eq!(serde_json::to_string(&duplicate).unwrap(), before);
    let expected_next = duplicate.with_point_placed(SketchPoint::new(100, 100)).1;
    assert_eq!(
        duplicate.with_point_placed(SketchPoint::new(100, 100)).1,
        expected_next,
        "duplicate refusal consumed no id"
    );
}

#[test]
fn every_midpoint_line_refusal_preserves_bytes_and_the_next_id() {
    let source = empty_solid();
    let upper = -(i64::MIN as f64);
    let cases = [
        (
            source.with_midpoint_line([0.0, 0.0], [0.0, 0.0], None),
            MidpointLineRefusal::Candidate(
                parametric::sketch::MidpointLineCandidateError::Collapsed,
            ),
        ),
        (
            source.with_midpoint_line([f64::NAN, 0.0], [1.0, 0.0], None),
            MidpointLineRefusal::Candidate(
                parametric::sketch::MidpointLineCandidateError::NonFinite,
            ),
        ),
        (
            source.with_midpoint_line([f64::MAX, 0.0], [-f64::MAX, 1.0], None),
            MidpointLineRefusal::Candidate(
                parametric::sketch::MidpointLineCandidateError::Overflow,
            ),
        ),
        (
            source.with_midpoint_line([upper, 0.0], [0.0, 1.0], None),
            MidpointLineRefusal::Point(SketchPointConstructionError::OutOfCanonicalRange),
        ),
        (
            source.with_midpoint_line([5.0, 0.0], [8.0, 0.0], Some(7777)),
            MidpointLineRefusal::UnknownEndpoint,
        ),
        (
            source.with_midpoint_line([0.25 + 1.0e-10, 0.0], [0.25, 0.0], None),
            MidpointLineRefusal::CanonicalCollapse,
        ),
    ];
    let before = serde_json::to_string(&source).unwrap();
    let expected_next = source.with_point_placed(SketchPoint::new(100, 100)).1;
    for (refusal, expected) in cases {
        assert_eq!(refusal, Err(expected));
        assert_eq!(serde_json::to_string(&source).unwrap(), before);
        assert_eq!(
            source.with_point_placed(SketchPoint::new(100, 100)).1,
            expected_next,
            "refusal consumed no id"
        );
    }
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
fn traced_segment_returns_the_created_stable_identity_only_on_success() {
    let (solid, from) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, to) = solid.with_point_placed(SketchPoint::new(4, 0));
    let (solid, curve) = solid
        .with_segment_between_traced(from, to)
        .expect("fresh segment");
    let SketchCurve::Segment(id) = curve else {
        panic!("segment identity")
    };
    assert_eq!(solid.sketch.segments()[0].id, id);
    assert_eq!(solid.with_segment_between_traced(to, from), None);
}

fn assert_tangent_arc_from_segment(reverse_storage: bool) {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, target) = solid.with_point_placed(SketchPoint::new(10, 10));
    let (solid, unrelated_id) = solid.with_point_placed(SketchPoint::new(99, 99));
    let unrelated = *solid
        .sketch
        .points()
        .iter()
        .find(|point| point.id == unrelated_id)
        .unwrap();
    let (solid, incoming) = solid
        .with_segment_between_traced(
            if reverse_storage { seam } else { tail },
            if reverse_storage { tail } else { seam },
        )
        .expect("incoming segment");
    let candidate = solid
        .sketch
        .tangent_arc_candidate(incoming, seam, [10.0, 10.0], ctx(16))
        .expect("preview candidate");
    let (after, created) = solid
        .with_tangent_arc_between(incoming, seam, target, ctx(16))
        .expect("atomic tangent arc");
    let SketchCurve::Arc(arc_id) = created else {
        panic!("arc identity")
    };
    let arc = after
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.id == arc_id)
        .unwrap();
    let parametric::sketch::CurveGeometry::Circular(persisted) =
        after.sketch.curve_geometry(created, ctx(16)).unwrap()
    else {
        panic!("circular geometry")
    };
    assert!((persisted.center[0] - candidate.center[0]).abs() < 1.0e-10);
    assert!((persisted.center[1] - candidate.center[1]).abs() < 1.0e-10);
    assert!((persisted.radius - candidate.radius).abs() < 1.0e-10);
    assert!((arc.sweep_degrees().to_radians() - candidate.sweep_radians).abs() < 1.0e-10);
    let constraint = after.sketch.constraints().last().expect("durable tangent");
    let ConstraintKind::Tangent {
        first,
        second,
        branch,
    } = constraint.kind
    else {
        panic!("tangent kind")
    };
    let contact = after
        .sketch
        .tangent_contact(first, second, branch, ctx(16))
        .expect("finite shared contact");
    assert_eq!(contact.at, [10.0, 0.0]);
    assert_eq!(
        after
            .sketch
            .points()
            .iter()
            .find(|point| point.id == unrelated.id),
        Some(&unrelated),
        "an exact candidate does not move unrelated geometry"
    );
    assert!(arc.id < arc.center, "arc precedes its derived center");
    assert!(arc.center < constraint.id, "center precedes the constraint");
    let replayed: SketchSolid =
        serde_json::from_str(&serde_json::to_string(&after).unwrap()).unwrap();
    let replayed_constraint = replayed.sketch.constraints().last().unwrap();
    let ConstraintKind::Tangent {
        first,
        second,
        branch,
    } = replayed_constraint.kind
    else {
        panic!("replayed tangent")
    };
    assert!(replayed
        .sketch
        .tangent_contact(first, second, branch, ctx(16))
        .is_ok());
}

#[test]
fn tangent_arc_from_segment_accepts_both_stored_endpoint_orientations() {
    assert_tangent_arc_from_segment(false);
    assert_tangent_arc_from_segment(true);
}

fn assert_tangent_arc_from_arc(reverse_storage: bool) {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, target) = solid.with_point_placed(SketchPoint::new(0, 10));
    let mut incoming = solid.clone();
    let incoming_id = incoming
        .sketch
        .connect_arc(
            if reverse_storage { seam } else { tail },
            if reverse_storage { tail } else { seam },
            AngleMeasurement::from_degrees(if reverse_storage { -180 } else { 180 }),
        )
        .expect("incoming arc");
    let incoming_curve = SketchCurve::Arc(incoming_id);
    let incoming_before = incoming.sketch.arcs()[0];
    let candidate = incoming
        .sketch
        .tangent_arc_candidate(incoming_curve, seam, [0.0, 10.0], ctx(16))
        .expect("preview candidate");
    let (after, created) = incoming
        .with_tangent_arc_between(incoming_curve, seam, target, ctx(16))
        .expect("arc-to-arc tangent");
    assert_eq!(
        after.sketch.arcs()[0],
        incoming_before,
        "incoming authority stays exact"
    );
    let created_arc = after
        .sketch
        .arcs()
        .iter()
        .find(|arc| SketchCurve::Arc(arc.id) == created)
        .unwrap();
    let parametric::sketch::CurveGeometry::Circular(persisted) =
        after.sketch.curve_geometry(created, ctx(16)).unwrap()
    else {
        panic!("persisted arc")
    };
    assert!((persisted.center[0] - candidate.center[0]).abs() < 1.0e-10);
    assert!((persisted.center[1] - candidate.center[1]).abs() < 1.0e-10);
    assert!((persisted.radius - candidate.radius).abs() < 1.0e-10);
    assert!((created_arc.sweep_degrees().to_radians() - candidate.sweep_radians).abs() < 1.0e-10);
    let constraint = after.sketch.constraints().last().unwrap();
    let ConstraintKind::Tangent {
        first,
        second,
        branch,
    } = constraint.kind
    else {
        panic!("tangent kind")
    };
    let contact = after
        .sketch
        .tangent_contact(first, second, branch, ctx(16))
        .unwrap();
    assert!((contact.at[0] - 10.0).abs() < 1.0e-10);
    assert!(contact.at[1].abs() < 1.0e-10);
    assert!(matches!(created, SketchCurve::Arc(_)));
}

#[test]
fn tangent_arc_from_arc_accepts_both_stored_endpoint_orientations() {
    assert_tangent_arc_from_arc(false);
    assert_tangent_arc_from_arc(true);
}

#[test]
fn tangent_arc_failures_are_atomic_and_do_not_advance_identity() {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, target) = solid.with_point_placed(SketchPoint::new(20, 0));
    let (solid, incoming) = solid
        .with_segment_between_traced(tail, seam)
        .expect("incoming");
    let before_json = serde_json::to_string(&solid).unwrap();
    assert_eq!(
        solid.with_tangent_arc_between(incoming, seam, target, ctx(16)),
        Err(TangentArcRefusal::Candidate(
            parametric::sketch::TangentArcCandidateError::Collinear
        ))
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before_json);
    let mut expected = solid.clone();
    let expected_id = expected.sketch.add_free_point(SketchPoint::new(30, 0));
    let mut actual = solid;
    let actual_id = actual.sketch.add_free_point(SketchPoint::new(30, 0));
    assert_eq!(actual_id, expected_id, "a refusal consumes no stable id");
}

#[test]
fn standalone_tangent_arc_preview_and_commit_share_the_canonical_destination() {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, incoming) = solid
        .with_segment_between_traced(tail, seam)
        .expect("incoming");
    let endpoint = SketchPoint::from_continuous(10.25, 7.5);
    let placement = solid
        .tangent_arc_placement_to(incoming, seam, endpoint, None, ctx(16))
        .unwrap();
    assert!(placement.candidate.radius.is_finite());
    assert!(placement.candidate.radius > 0.0);

    let (made, arc) = solid
        .with_tangent_arc_to(incoming, seam, endpoint, None, ctx(16))
        .unwrap();
    let arc = made
        .sketch
        .arcs()
        .iter()
        .find(|candidate| SketchCurve::Arc(candidate.id) == arc)
        .unwrap();
    let stored_endpoint = made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == arc.to)
        .unwrap()
        .at;
    assert!(placement.endpoint.coincides(&stored_endpoint));
    assert_eq!(made.sketch.constraints().len(), 1);
}

#[test]
fn standalone_tangent_arc_reuses_an_authoritative_endpoint_and_refuses_atomically() {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, endpoint) = solid.with_point_placed(SketchPoint::new(10, 10));
    let (solid, incoming) = solid
        .with_segment_between_traced(tail, seam)
        .expect("incoming");
    let (made, arc) = solid
        .with_tangent_arc_to(
            incoming,
            seam,
            SketchPoint::new(999, 999),
            Some(endpoint),
            ctx(16),
        )
        .unwrap();
    let arc_id = arc.id();
    assert_eq!(
        made.sketch
            .arcs()
            .iter()
            .find(|arc| arc.id == arc_id)
            .unwrap()
            .to,
        endpoint
    );

    let before = serde_json::to_string(&solid).unwrap();
    assert!(solid
        .with_tangent_arc_to(incoming, seam, SketchPoint::new(20, 0), None, ctx(16),)
        .is_err());
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
}

#[test]
fn tangent_arc_refuses_unsupported_dead_nonincident_self_and_duplicate_inputs() {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (solid, target) = solid.with_point_placed(SketchPoint::new(10, 10));
    let (solid, other) = solid.with_point_placed(SketchPoint::new(20, 0));
    let (solid, incoming) = solid
        .with_segment_between_traced(tail, seam)
        .expect("incoming");
    let (solid, nonincident) = solid
        .with_segment_between_traced(tail, other)
        .expect("nonincident curve");
    let mut with_circle = solid.clone();
    let circle = with_circle
        .sketch
        .add_circle(SketchPoint::new(30, 30), SketchLength::new(4))
        .unwrap();
    let before = serde_json::to_string(&solid).unwrap();
    let circle_before = serde_json::to_string(&with_circle).unwrap();

    assert_eq!(
        solid.with_tangent_arc_between(SketchCurve::Segment(9999), seam, target, ctx(16)),
        Err(TangentArcRefusal::UnknownIncoming)
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
    assert_eq!(
        solid.with_tangent_arc_between(nonincident, seam, target, ctx(16)),
        Err(TangentArcRefusal::NonIncidentIncoming)
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
    assert_eq!(
        solid.with_tangent_arc_between(incoming, seam, seam, ctx(16)),
        Err(TangentArcRefusal::SelfLoop)
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
    assert_eq!(
        solid.with_tangent_arc_between(incoming, seam, 9999, ctx(16)),
        Err(TangentArcRefusal::UnknownEndpoint)
    );
    assert_eq!(serde_json::to_string(&solid).unwrap(), before);
    assert_eq!(
        with_circle.with_tangent_arc_between(SketchCurve::Circle(circle), seam, target, ctx(16)),
        Err(TangentArcRefusal::UnsupportedIncoming)
    );
    assert_eq!(serde_json::to_string(&with_circle).unwrap(), circle_before);

    let (after, _) = solid
        .with_tangent_arc_between(incoming, seam, target, ctx(16))
        .expect("first arc");
    let before = serde_json::to_string(&after).unwrap();
    assert_eq!(
        after.with_tangent_arc_between(incoming, seam, target, ctx(16)),
        Err(TangentArcRefusal::ArcRefused)
    );
    assert_eq!(serde_json::to_string(&after).unwrap(), before);
}

#[test]
fn tangent_arc_reads_fixed_incoming_sweep_without_reauthoring_it_or_using_density() {
    let (solid, tail) = empty_solid().with_point_placed(SketchPoint::new(0, 0));
    let (solid, seam) = solid.with_point_placed(SketchPoint::new(10, 0));
    let (mut solid, target) = solid.with_point_placed(SketchPoint::new(0, 10));
    let incoming_id = solid
        .sketch
        .connect_arc(tail, seam, AngleMeasurement::from_degrees(180))
        .unwrap();
    solid.sketch.arcs_mut_for_test()[0].bulge =
        parametric::ArcSweep::fixed(AngleMeasurement::from_degrees(180));
    let incoming = SketchCurve::Arc(incoming_id);
    assert_eq!(
        solid
            .sketch
            .tangent_arc_candidate(incoming, seam, [0.0, 10.0], ctx(8))
            .unwrap(),
        solid
            .sketch
            .tangent_arc_candidate(incoming, seam, [0.0, 10.0], ctx(64))
            .unwrap()
    );
    let (after, _) = solid
        .with_tangent_arc_between(incoming, seam, target, ctx(64))
        .unwrap();
    assert_eq!(
        after.sketch.arcs()[0].bulge.fixed_source().copied(),
        Some(AngleMeasurement::from_degrees(180))
    );
    assert_eq!(after.sketch.arcs()[0].bulge.free_value(), None);
}

#[test]
fn with_rectangle_closes_a_four_point_loop() {
    let after = empty_solid()
        .with_rectangle(SketchPoint::new(1, 1), SketchPoint::new(4, 3), ctx(16))
        .unwrap();
    assert_eq!(after.sketch.points().len(), 4);
    assert_eq!(after.sketch.segments().len(), 4);
    let coords: std::collections::BTreeSet<[i64; 2]> = after
        .sketch
        .flattened_loop(ctx(16))
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
    let one = empty_solid()
        .with_rectangle(SketchPoint::new(0, 0), SketchPoint::new(4, 3), ctx(16))
        .unwrap();
    // The shared side already carries the first rectangle's Vertical, so the second rectangle
    // re-asserting it is idempotent rather than a refusal.
    let two = one
        .with_rectangle(SketchPoint::new(4, 0), SketchPoint::new(8, 3), ctx(16))
        .unwrap();
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
        before.with_rectangle(SketchPoint::new(2, 2), SketchPoint::new(2, 5), ctx(16)),
        Err(crate::sketch::RectangleRefusal::Unrepresentable),
        "a degenerate rectangle (zero span on an axis) draws nothing"
    );
    assert_eq!(
        before.with_rectangle(SketchPoint::new(2, 2), SketchPoint::new(2, 2), ctx(16)),
        Err(crate::sketch::RectangleRefusal::Unrepresentable)
    );
}

/// Each side of a two-point rectangle carries the relation that keeps it on its axis, so
/// dragging a corner later stretches the rectangle instead of shearing it.
#[test]
fn a_two_point_rectangle_constrains_every_side_to_its_axis() {
    let made = empty_solid()
        .with_rectangle(SketchPoint::new(1, 1), SketchPoint::new(5, 4), ctx(16))
        .unwrap();
    let mut horizontal = 0;
    let mut vertical = 0;
    for constraint in made.sketch.constraints() {
        match constraint.kind {
            ConstraintKind::Horizontal { .. } => horizontal += 1,
            ConstraintKind::Vertical { .. } => vertical += 1,
            other => panic!("unexpected relation on a two-point rectangle: {other:?}"),
        }
    }
    assert_eq!((horizontal, vertical), (2, 2));
}

/// A three-point rectangle may turn, so it asserts squareness WITHOUT pinning an axis: opposite
/// sides parallel and one corner perpendicular.
#[test]
fn a_three_point_rectangle_stays_square_without_pinning_its_rotation() {
    let made = empty_solid()
        .with_three_point_rectangle(
            SketchPoint::new(0, 0),
            SketchPoint::new(3, 4),
            SketchPoint::new(0, 5),
            ctx(16),
        )
        .unwrap();
    let mut parallel = 0;
    let mut perpendicular = 0;
    for constraint in made.sketch.constraints() {
        match constraint.kind {
            ConstraintKind::Parallel { .. } => parallel += 1,
            ConstraintKind::Perpendicular { .. } => perpendicular += 1,
            other => panic!("a three-point rectangle must not pin an axis: {other:?}"),
        }
    }
    assert_eq!((parallel, perpendicular), (2, 1));
}

/// The center is real authored geometry: it persists as a construction point held at the
/// crossing of two construction diagonals, and the diagonals never bound the region.
#[test]
fn a_center_rectangle_hangs_its_center_from_two_construction_diagonals() {
    use crate::sketch::EntityRole;
    let center = SketchPoint::new(4, 3);
    let made = empty_solid()
        .with_center_rectangle(center, SketchPoint::new(7, 8), ctx(16))
        .unwrap();

    let diagonals: Vec<_> = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.role == EntityRole::Construction)
        .collect();
    assert_eq!(diagonals.len(), 2, "corner to corner, both ways");

    let center_id = made.sketch.point_at(center).expect("the center persists");
    let held: Vec<_> = made
        .sketch
        .constraints()
        .iter()
        .filter_map(|constraint| match constraint.kind {
            ConstraintKind::Midpoint { point, segment } if point == center_id => Some(segment),
            _ => None,
        })
        .collect();
    assert_eq!(held.len(), 2, "the center is halfway along BOTH diagonals");
    for segment in held {
        assert!(diagonals.iter().any(|held| held.id == segment));
    }

    // The interior stays ONE face: a construction edge never bounds a region.
    assert_eq!(made.sketch.faces(ctx(16)).len(), 1);
}

/// A constrained construction point is referenced. Without this the center would survive its
/// own creation and then vanish the next time any unrelated deletion swept the sketch.
#[test]
fn a_center_rectangle_keeps_its_center_across_an_unrelated_deletion() {
    let center = SketchPoint::new(4, 3);
    let mut made = empty_solid()
        .with_center_rectangle(center, SketchPoint::new(7, 8), ctx(16))
        .unwrap();
    let circle = made
        .sketch
        .add_circle(SketchPoint::new(40, 40), SketchLength::new(3))
        .expect("an unrelated circle");
    made.sketch.delete_circle(circle);
    assert!(
        made.sketch.point_at(center).is_some(),
        "the center is held by its Midpoint assertions, not by a curve"
    );
}

/// A dot marks what the ink cannot say. A joined corner says it already; a loose end does not.
#[test]
fn a_joined_corner_needs_no_dot_and_a_loose_end_does() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let left = sketch.add_free_point(SketchPoint::new(0, 0));
    let corner = sketch.add_free_point(SketchPoint::new(4, 0));
    let right = sketch.add_free_point(SketchPoint::new(4, 4));
    sketch.connect(left, corner).expect("a line");
    sketch.connect(corner, right).expect("a second line off it");

    assert!(
        !sketch.point_draws_at_rest(corner),
        "two ends meet here and the corner is already visible as a corner"
    );
    assert!(
        sketch.point_draws_at_rest(left) && sketch.point_draws_at_rest(right),
        "a loose end is the one thing the ink cannot report"
    );
}

/// Two ends that merely COINCIDE are two loose ends, and both say so.
///
/// This is the case the rule exists for: joined and unjoined look identical, and which one it is
/// decides whether the profile ever closes into a region.
#[test]
fn a_seam_that_only_looks_joined_draws_both_of_its_ends() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let left = sketch.add_free_point(SketchPoint::new(0, 0));
    let first_end = sketch.add_free_point(SketchPoint::new(4, 0));
    let second_end = sketch.add_free_point(SketchPoint::new(4, 0));
    let right = sketch.add_free_point(SketchPoint::new(8, 0));
    sketch.connect(left, first_end).expect("a line");
    sketch
        .connect(second_end, right)
        .expect("a line starting where it ended");

    assert!(
        sketch.point_draws_at_rest(first_end) && sketch.point_draws_at_rest(second_end),
        "the seam is open, and two dots on one place is exactly how that reads"
    );
}

/// A center has no ink on it, so the dot is the only evidence it is there — even though a curve
/// names it, which a corner's two segments also do.
#[test]
fn a_center_draws_where_a_corner_does_not() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(5))
        .expect("a circle");
    let center = sketch.circles()[0].center;

    let a = sketch.add_free_point(SketchPoint::new(20, 0));
    let corner = sketch.add_free_point(SketchPoint::new(24, 0));
    let b = sketch.add_free_point(SketchPoint::new(24, 4));
    sketch.connect(a, corner).expect("a line");
    sketch.connect(corner, b).expect("a second line off it");

    assert!(
        sketch.point_draws_at_rest(center),
        "a circle names its center, but nothing is DRAWN there"
    );
    assert!(
        !sketch.point_draws_at_rest(corner),
        "two segments name this one too, and they draw right through it"
    );
}

/// A spline's ink shows its shape and says nothing about which points made it.
#[test]
fn a_fit_point_draws_because_the_curve_through_it_does_not_reveal_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a spline");
    let spline = sketch.splines()[0].clone();
    for fit in &spline.points {
        assert!(
            sketch.point_draws_at_rest(*fit),
            "a run through five points and a run through seven are one picture"
        );
    }
}
