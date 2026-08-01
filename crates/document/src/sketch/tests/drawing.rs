//! The drawing-tool store mutators: free points, `connect`, coincidence via `point_at`, and the
//! pure `with_point_placed` / `with_segment_between` / `with_rectangle` wrappers the polyline and
//! rectangle gestures commit through. Coincidence IS shared point identity: placing on an occupied
//! coord reuses the id, never mints a twin.

use super::ctx;
use crate::sketch::{
    ConstraintKind, PlaneAxis, Sketch, SketchCurve, SketchLength, SketchPoint, SketchSolid,
    TangentArcRefusal,
};
use parametric::units::AngleMeasurement;

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
    let after = empty_solid().with_rectangle(SketchPoint::new(1, 1), SketchPoint::new(4, 3));
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
