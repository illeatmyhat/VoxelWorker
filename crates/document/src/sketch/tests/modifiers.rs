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
