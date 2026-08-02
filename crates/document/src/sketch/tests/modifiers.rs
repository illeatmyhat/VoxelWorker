use super::*;
use ::parametric::units::AngleMeasurement;
use substrate::curve_intersection::PlanarCurve;

fn segment(sketch: &mut Sketch, from: [i64; 2], to: [i64; 2]) -> EntityId {
    let from = sketch.add_free_point(SketchPoint::new(from[0], from[1]));
    let to = sketch.add_free_point(SketchPoint::new(to[0], to[1]));
    sketch.connect(from, to).unwrap()
}

#[test]
fn break_splits_a_line_at_every_interior_crossing_and_preserves_lineage() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = segment(&mut sketch, [0, 0], [12, 0]);
    segment(&mut sketch, [3, -2], [3, 2]);
    segment(&mut sketch, [9, -2], [9, 2]);
    let origin = sketch
        .segments()
        .iter()
        .find(|segment| segment.id == target)
        .unwrap()
        .origin;
    let source = SketchSolid::extrude(sketch, 3);

    let made = source
        .with_curve_broken(SketchCurve::Segment(target), ctx(16))
        .unwrap();
    let pieces: Vec<_> = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.origin == origin)
        .collect();
    assert_eq!(pieces.len(), 3);
    assert_eq!(
        pieces[0].id, target,
        "the first piece retains curve identity"
    );
    assert!(pieces.iter().all(|piece| piece.role == EntityRole::Real));
    assert_eq!(
        source
            .sketch
            .segments()
            .iter()
            .filter(|segment| segment.origin == origin)
            .count(),
        1
    );
}

#[test]
fn break_keeps_arc_pieces_native_and_on_the_same_circle() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(-5, 0));
    let to = sketch.add_free_point(SketchPoint::new(5, 0));
    let target = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(-180))
        .unwrap();
    segment(&mut sketch, [0, -8], [0, 8]);
    let origin = sketch.arcs()[0].origin;
    let source = SketchSolid::extrude(sketch, 3);

    let made = source
        .with_curve_broken(SketchCurve::Arc(target), ctx(16))
        .unwrap();
    let pieces: Vec<_> = made
        .sketch
        .arcs()
        .iter()
        .filter(|arc| arc.origin == origin)
        .collect();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].id, target);
    assert!((pieces.iter().map(|arc| arc.sweep_degrees()).sum::<f64>() + 180.0).abs() < 1.0e-9);
    let geometries: Vec<_> = pieces
        .iter()
        .map(|arc| {
            made.sketch
                .curve_geometry(SketchCurve::Arc(arc.id), ctx(16))
                .unwrap()
        })
        .collect();
    assert!(geometries
        .iter()
        .all(|geometry| matches!(geometry, ::parametric::sketch::CurveGeometry::Circular(_))));
}

#[test]
fn break_turns_a_twice_crossed_circle_into_native_arcs_and_drops_circle_relations() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(5))
        .unwrap();
    let concentric = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(8))
        .unwrap();
    sketch
        .add_constraint(
            ConstraintKind::concentric(
                SketchCurve::Circle(target),
                SketchCurve::Circle(concentric),
            ),
            ctx(16),
        )
        .unwrap();
    segment(&mut sketch, [-10, 0], [10, 0]);
    let origin = sketch
        .circles()
        .iter()
        .find(|circle| circle.id == target)
        .unwrap()
        .origin;
    let source = SketchSolid::extrude(sketch, 3);

    let made = source
        .with_curve_broken(SketchCurve::Circle(target), ctx(16))
        .unwrap();
    assert!(made
        .sketch
        .circles()
        .iter()
        .all(|circle| circle.id != target));
    let pieces: Vec<_> = made
        .sketch
        .arcs()
        .iter()
        .filter(|arc| arc.origin == origin)
        .collect();
    assert_eq!(pieces.len(), 2);
    assert!((pieces.iter().map(|arc| arc.sweep_degrees()).sum::<f64>() - 360.0).abs() < 1.0e-9);
    assert!(made.sketch.constraints().is_empty());
}

#[test]
fn break_refuses_a_curve_without_an_interior_intersection_atomically() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = segment(&mut sketch, [0, 0], [5, 0]);
    segment(&mut sketch, [10, 0], [10, 5]);
    let source = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        source.with_curve_broken(SketchCurve::Segment(target), ctx(16)),
        Err(BreakRefusal::NoInteriorIntersection)
    );
    assert_eq!(source.sketch.segments().len(), 2);
}

#[test]
fn trim_removes_only_the_witnessed_interval_between_neighboring_crossings() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = segment(&mut sketch, [0, 0], [12, 0]);
    segment(&mut sketch, [3, -2], [3, 2]);
    segment(&mut sketch, [9, -2], [9, 2]);
    let origin = sketch.segments()[0].origin;
    let source = SketchSolid::extrude(sketch, 3);

    let placement = source
        .trim_placement(SketchCurve::Segment(target), [6.0, 0.2], ctx(16))
        .unwrap();
    assert_eq!(placement.kept.len(), 2);
    assert_eq!(placement.removed.start(), [3.0, 0.0]);
    assert_eq!(placement.removed.end(), [9.0, 0.0]);
    let made = source
        .with_curve_trimmed(SketchCurve::Segment(target), [6.0, 0.2], ctx(16))
        .unwrap();
    let kept: Vec<_> = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.origin == origin)
        .collect();
    assert_eq!(kept.len(), 2);
    assert!(made
        .sketch
        .segments()
        .iter()
        .all(|segment| segment.id != target));
}

#[test]
fn trim_without_a_crossing_deletes_the_curve_and_circle_trim_stays_curved() {
    let mut lone = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut lone, [0, 0], [5, 0]);
    let source = SketchSolid::extrude(lone, 3);
    let deleted = source
        .with_curve_trimmed(SketchCurve::Segment(line), [2.0, 0.0], ctx(16))
        .unwrap();
    assert!(deleted.sketch.segments().is_empty());

    let mut crossed = Sketch::empty(PlaneAxis::Z);
    let circle = crossed
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(5))
        .unwrap();
    segment(&mut crossed, [-10, 0], [10, 0]);
    let source = SketchSolid::extrude(crossed, 3);
    let trimmed = source
        .with_curve_trimmed(SketchCurve::Circle(circle), [0.0, 4.0], ctx(16))
        .unwrap();
    assert!(trimmed.sketch.circles().is_empty());
    assert_eq!(trimmed.sketch.arcs().len(), 1);
    assert!((trimmed.sketch.arcs()[0].sweep_degrees().abs() - 180.0).abs() < 1.0e-9);
}

#[test]
fn extend_grows_the_witnessed_segment_end_to_the_nearest_finite_crossing() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = segment(&mut sketch, [0, 0], [2, 0]);
    segment(&mut sketch, [5, -2], [5, 2]);
    segment(&mut sketch, [8, -2], [8, 2]);
    segment(&mut sketch, [-3, -2], [-3, 2]);
    let source = SketchSolid::extrude(sketch, 3);

    let placement = source
        .extend_placement(SketchCurve::Segment(target), [2.0, 0.1], ctx(16))
        .unwrap();
    assert_eq!(placement.endpoint, ExtendEndpoint::End);
    assert_eq!(placement.extended.end(), [5.0, 0.0]);
    let made = source
        .with_curve_extended(SketchCurve::Segment(target), [2.0, 0.1], ctx(16))
        .unwrap();
    let held = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.id == target)
        .unwrap();
    assert_eq!(
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == held.to)
            .unwrap()
            .at,
        SketchPoint::new(5, 0)
    );

    let start = source
        .extend_placement(SketchCurve::Segment(target), [0.0, 0.1], ctx(16))
        .unwrap();
    assert_eq!(start.endpoint, ExtendEndpoint::Start);
    assert_eq!(start.extended.start(), [-3.0, 0.0]);
}

#[test]
fn extend_refuses_a_closed_curve_or_a_selected_ray_without_a_hit() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = segment(&mut sketch, [0, 0], [2, 0]);
    segment(&mut sketch, [5, -2], [5, 2]);
    let circle = sketch
        .add_circle(SketchPoint::new(20, 0), SketchLength::new(3))
        .unwrap();
    let source = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        source.extend_placement(SketchCurve::Segment(target), [0.0, 0.0], ctx(16)),
        Err(ExtendRefusal::NoIntersection)
    );
    assert_eq!(
        source.extend_placement(SketchCurve::Circle(circle), [23.0, 0.0], ctx(16)),
        Err(ExtendRefusal::ClosedCurve)
    );
}

#[test]
fn extend_keeps_an_arc_native_and_grows_each_end_around_its_supporting_circle() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(5, 0));
    let to = sketch.add_free_point(SketchPoint::new(0, 5));
    let target = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .unwrap();
    segment(&mut sketch, [-5, -1], [-5, 1]);
    segment(&mut sketch, [-1, -5], [1, -5]);
    let source = SketchSolid::extrude(sketch, 3);

    let mut fixed = source.clone();
    fixed.sketch.arcs_mut_for_test()[0].bulge = ArcSweep::fixed(AngleMeasurement::from_degrees(90));
    assert_eq!(
        fixed.extend_placement(SketchCurve::Arc(target), [0.0, 5.0], ctx(16)),
        Err(ExtendRefusal::FixedSweep)
    );

    let end = source
        .extend_placement(SketchCurve::Arc(target), [0.0, 5.0], ctx(16))
        .unwrap();
    assert_eq!(end.endpoint, ExtendEndpoint::End);
    let extended_end = end.extended.end();
    assert!((extended_end[0] + 5.0).abs() < 1.0e-9);
    assert!(extended_end[1].abs() < 1.0e-9);
    let made = source
        .with_curve_extended(SketchCurve::Arc(target), [0.0, 5.0], ctx(16))
        .unwrap();
    let held = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.id == target)
        .unwrap();
    assert!((held.sweep_degrees() - 180.0).abs() < 1.0e-9);
    let ::parametric::sketch::CurveGeometry::Circular(geometry) = made
        .sketch
        .curve_geometry(SketchCurve::Arc(target), ctx(16))
        .unwrap()
    else {
        panic!("an extended arc must remain circular");
    };
    assert!((geometry.radius - 5.0).abs() < 1.0e-9);
    assert!(geometry.center[0].abs() < 1.0e-9);
    assert!(geometry.center[1].abs() < 1.0e-9);

    let start = source
        .extend_placement(SketchCurve::Arc(target), [5.0, 0.0], ctx(16))
        .unwrap();
    assert_eq!(start.endpoint, ExtendEndpoint::Start);
    let extended_start = start.extended.start();
    assert!(extended_start[0].abs() < 1.0e-9);
    assert!((extended_start[1] + 5.0).abs() < 1.0e-9);
}

#[test]
fn extend_preserves_a_clockwise_arcs_direction_and_radius() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(5, 0));
    let to = sketch.add_free_point(SketchPoint::new(0, -5));
    let target = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(-90))
        .unwrap();
    segment(&mut sketch, [-5, -1], [-5, 1]);
    let source = SketchSolid::extrude(sketch, 3);

    let made = source
        .with_curve_extended(SketchCurve::Arc(target), [0.0, -5.0], ctx(16))
        .unwrap();
    let held = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.id == target)
        .unwrap();
    assert!((held.sweep_degrees() + 180.0).abs() < 1.0e-9);
    let ::parametric::sketch::CurveGeometry::Circular(geometry) = made
        .sketch
        .curve_geometry(SketchCurve::Arc(target), ctx(16))
        .unwrap()
    else {
        panic!("an extended clockwise arc must remain circular");
    };
    assert!((geometry.radius - 5.0).abs() < 1.0e-9);
}

#[test]
fn fillet_rounds_a_two_line_corner_with_a_native_durably_tangent_arc() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first_far = sketch.add_free_point(SketchPoint::new(10, 0));
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let second_far = sketch.add_free_point(SketchPoint::new(0, 10));
    let first = sketch.connect(first_far, corner).unwrap();
    let second = sketch.connect(corner, second_far).unwrap();
    let source = SketchSolid::extrude(sketch, 3);

    let placement = source
        .fillet_placement(SketchCurve::Segment(first), [2.0, 0.1], ctx(16))
        .unwrap();
    assert_eq!(placement.first, SketchCurve::Segment(first));
    assert_eq!(placement.second, SketchCurve::Segment(second));
    assert_eq!(placement.shortened_first.end(), [2.0, 0.0]);
    assert_eq!(placement.shortened_second.start(), [0.0, 2.0]);
    let PlanarCurve::Arc {
        center,
        radius,
        sweep_radians,
        ..
    } = placement.arc
    else {
        panic!("a fillet placement must remain a native arc");
    };
    assert!((center[0] - 2.0).abs() < 1.0e-9);
    assert!((center[1] - 2.0).abs() < 1.0e-9);
    assert!((radius - 2.0).abs() < 1.0e-9);
    assert!((sweep_radians.to_degrees() + 90.0).abs() < 1.0e-9);

    let made = source
        .with_corner_filleted(SketchCurve::Segment(first), [2.0, 0.1], ctx(16))
        .unwrap();
    assert!(made.sketch.segments().iter().any(|held| held.id == first));
    assert!(made.sketch.segments().iter().any(|held| held.id == second));
    assert_eq!(made.sketch.arcs().len(), 1);
    assert_eq!(made.sketch.constraints().len(), 2);
    assert!(made
        .sketch
        .constraints()
        .iter()
        .all(|constraint| matches!(constraint.kind, ConstraintKind::Tangent { .. })));
    let geometry = made
        .sketch
        .curve_geometry(SketchCurve::Arc(made.sketch.arcs()[0].id), ctx(16))
        .unwrap();
    let ::parametric::sketch::CurveGeometry::Circular(geometry) = geometry else {
        panic!("the committed fillet must remain circular");
    };
    assert!((geometry.radius - 2.0).abs() < 1.0e-8);
}

#[test]
fn fillet_refuses_an_ambiguous_or_overlarge_corner_without_editing() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let a = sketch.add_free_point(SketchPoint::new(10, 0));
    let b = sketch.add_free_point(SketchPoint::new(0, 10));
    let c = sketch.add_free_point(SketchPoint::new(-10, 0));
    let first = sketch.connect(corner, a).unwrap();
    sketch.connect(corner, b).unwrap();
    sketch.connect(corner, c).unwrap();
    let source = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        source.fillet_placement(SketchCurve::Segment(first), [2.0, 0.0], ctx(16)),
        Err(FilletRefusal::AmbiguousCorner)
    );

    let mut short = Sketch::empty(PlaneAxis::Z);
    let corner = short.add_free_point(SketchPoint::new(0, 0));
    let a = short.add_free_point(SketchPoint::new(10, 0));
    let b = short.add_free_point(SketchPoint::new(0, 1));
    let first = short.connect(corner, a).unwrap();
    short.connect(corner, b).unwrap();
    let source = SketchSolid::extrude(short, 3);
    assert_eq!(
        source.fillet_placement(SketchCurve::Segment(first), [2.0, 0.0], ctx(16)),
        Err(FilletRefusal::RadiusOutOfRange)
    );
}
