use super::*;

#[test]
fn cubic_bezier_is_durable_profile_geometry() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let curve = sketch
        .add_cubic_bezier([
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 10),
            SketchPoint::new(10, 10),
            SketchPoint::new(10, 0),
        ])
        .expect("valid cubic");
    let [from, _, _, to] = sketch.beziers()[0].controls;
    sketch.connect(to, from).expect("closing chord");

    assert_eq!(sketch.beziers()[0].id, curve);
    assert_eq!(sketch.faces(ctx(16)).len(), 1);

    let json = serde_json::to_string(&sketch).expect("serialize");
    let restored: Sketch = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(restored, sketch);
    assert_eq!(restored.faces(ctx(16)).len(), 1);
}

#[test]
fn bezier_point_delete_cascades_the_curve() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_cubic_bezier([
            SketchPoint::new(0, 0),
            SketchPoint::new(1, 2),
            SketchPoint::new(3, 2),
            SketchPoint::new(4, 0),
        ])
        .expect("valid cubic");
    let handle = sketch.beziers()[0].controls[1];

    sketch.delete_point_cascade(handle);

    assert!(sketch.beziers().is_empty());
    assert!(sketch.points().iter().all(|point| point.id != handle));
}

#[test]
fn repair_erases_a_bezier_with_invalid_weights() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_cubic_bezier([
            SketchPoint::new(0, 0),
            SketchPoint::new(1, 2),
            SketchPoint::new(3, 2),
            SketchPoint::new(4, 0),
        ])
        .expect("valid cubic");
    sketch.beziers[0].weights[2] = 0.0;

    assert_eq!(sketch.repair(ctx(16)), 1);
    assert!(sketch.beziers().is_empty());
}

#[test]
fn older_documents_default_to_no_beziers() {
    let sketch = Sketch::rectangle(PlaneAxis::Z, 4, 3);
    let mut value = serde_json::to_value(sketch).expect("serialize");
    value
        .as_object_mut()
        .expect("sketch object")
        .remove("beziers");

    let restored: Sketch = serde_json::from_value(value).expect("legacy document loads");
    assert!(restored.beziers().is_empty());
}

#[test]
fn break_preserves_rational_bezier_pieces_and_lineage() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = sketch
        .add_cubic_bezier([
            SketchPoint::new(0, 0),
            SketchPoint::new(3, 5),
            SketchPoint::new(7, 5),
            SketchPoint::new(10, 0),
        ])
        .expect("valid cubic");
    let crossing_from = sketch.add_free_point(SketchPoint::new(5, -1));
    let crossing_to = sketch.add_free_point(SketchPoint::new(5, 6));
    sketch
        .connect(crossing_from, crossing_to)
        .expect("crossing segment");
    let origin = sketch.beziers()[0].origin;

    let broken = SketchSolid::extrude(sketch, 2)
        .with_curve_broken(SketchCurve::Bezier(target), ctx(16))
        .expect("interior crossing breaks the curve");

    let pieces: Vec<_> = broken
        .sketch
        .beziers()
        .iter()
        .filter(|piece| piece.origin == origin)
        .collect();
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[0].controls[3], pieces[1].controls[0]);
}

#[test]
fn transforms_and_patterns_treat_bezier_as_one_curve() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let target = sketch
        .add_cubic_bezier([
            SketchPoint::new(0, 0),
            SketchPoint::new(1, 2),
            SketchPoint::new(3, 2),
            SketchPoint::new(4, 0),
        ])
        .expect("valid cubic");
    sketch
        .add_rectangular_pattern(
            [SketchCurve::Bezier(target)],
            [2, 1],
            [
                SketchVector::from_continuous(10.0, 0.0),
                SketchVector::from_continuous(0.0, 0.0),
            ],
        )
        .expect("valid pattern");
    assert!(matches!(
        sketch.derived_pattern_curves(ctx(16))[0].geometry,
        substrate::curve_intersection::PlanarCurve::RationalBezier(_)
    ));

    let moved = SketchSolid::extrude(sketch, 2)
        .with_entities_translated(
            &[SketchTransformEntity::Curve(SketchCurve::Bezier(target))],
            [5.0, -3.0],
            false,
        )
        .expect("unconstrained curve translates");
    let first = moved.sketch.beziers()[0].controls[0];
    let first = moved
        .sketch
        .points()
        .iter()
        .find(|point| point.id == first)
        .expect("control point");
    assert_eq!(first.at.in_plane(), [5.0, -3.0]);
}
