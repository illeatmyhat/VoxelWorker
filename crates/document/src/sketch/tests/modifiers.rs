use super::*;
use ::parametric::units::AngleMeasurement;

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
