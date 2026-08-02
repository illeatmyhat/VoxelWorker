use super::*;

fn segment(sketch: &mut Sketch, from: [i64; 2], to: [i64; 2]) -> EntityId {
    let from = sketch.add_free_point(SketchPoint::new(from[0], from[1]));
    let to = sketch.add_free_point(SketchPoint::new(to[0], to[1]));
    sketch.connect(from, to).unwrap()
}

fn assert_circle_matches_placement(
    sketch: &Sketch,
    placement: &TangentCirclePlacement,
    context: ::parametric::EvaluationContext,
) {
    let circle = sketch.circles()[0].id;
    let ::parametric::sketch::CurveGeometry::Circular(geometry) = sketch
        .curve_geometry(SketchCurve::Circle(circle), context)
        .expect("the committed circle resolves")
    else {
        panic!("a committed circle resolves as circular geometry");
    };
    let expected_center = placement.center.in_plane();
    // Sketch coordinates cross the canonical `i64 + f32` storage boundary before the durable
    // constraints settle, so preview and commit identity is bounded by one stored scalar ULP.
    let same_stored_scalar = |actual: f64, expected: f64| {
        (actual - expected).abs()
            <= f64::from(f32::EPSILON) * actual.abs().max(expected.abs()).max(1.0)
    };
    assert!(
        same_stored_scalar(geometry.center[0], expected_center[0]),
        "center x: {} != {}",
        geometry.center[0],
        expected_center[0]
    );
    assert!(
        same_stored_scalar(geometry.center[1], expected_center[1]),
        "center y: {} != {}",
        geometry.center[1],
        expected_center[1]
    );
    assert!(
        same_stored_scalar(geometry.radius, placement.radius.value()),
        "radius: {} != {}",
        geometry.radius,
        placement.radius.value()
    );
}

#[test]
fn two_tangent_circle_commits_circle_and_both_durable_relations_atomically() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let horizontal = segment(&mut sketch, [0, 0], [10, 0]);
    let vertical = segment(&mut sketch, [0, 0], [0, 10]);
    let source = SketchSolid::extrude(sketch, 4);
    let placement = source
        .two_tangent_circle_placement([horizontal, vertical], SketchPoint::new(2, 3))
        .unwrap();
    assert_eq!(placement.center, SketchPoint::new(2, 2));
    assert_eq!(placement.radius.value(), 2.0);

    let made = source
        .with_two_tangent_circle([horizontal, vertical], SketchPoint::new(2, 3), ctx(16))
        .unwrap();
    assert_eq!(made.sketch.circles().len(), 1);
    assert_eq!(made.sketch.constraints().len(), 2);
    assert_circle_matches_placement(&made.sketch, &placement, ctx(16));
    let mut solved = made.sketch.clone();
    assert!(solved.solve(ctx(16)).is_ok());
    assert!(source.sketch.circles().is_empty());
    assert!(source.sketch.constraints().is_empty());
}

#[test]
fn three_tangent_circle_commits_the_incircle_and_three_relations() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let bottom = segment(&mut sketch, [0, 0], [10, 0]);
    let diagonal = segment(&mut sketch, [10, 0], [0, 10]);
    let left = segment(&mut sketch, [0, 10], [0, 0]);
    let source = SketchSolid::extrude(sketch, 4);
    let picks = [
        (bottom, SketchPoint::new(5, 0)),
        (diagonal, SketchPoint::new(5, 5)),
        (left, SketchPoint::new(0, 5)),
    ];
    let placement = source.three_tangent_circle_placement(picks).unwrap();
    let made = source.with_three_tangent_circle(picks, ctx(16)).unwrap();
    assert_eq!(made.sketch.circles().len(), 1);
    assert_eq!(made.sketch.constraints().len(), 3);
    assert_circle_matches_placement(&made.sketch, &placement, ctx(16));
    let mut solved = made.sketch.clone();
    assert!(solved.solve(ctx(16)).is_ok());
}

#[test]
fn invalid_sources_and_out_of_span_contacts_leave_the_source_untouched() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let horizontal = segment(&mut sketch, [0, 0], [1, 0]);
    let vertical = segment(&mut sketch, [0, 0], [0, 1]);
    let source = SketchSolid::extrude(sketch, 4);
    assert!(source
        .with_two_tangent_circle([horizontal, vertical], SketchPoint::new(5, 6), ctx(16),)
        .is_err());
    assert!(source
        .with_two_tangent_circle([horizontal, EntityId::MAX], SketchPoint::new(1, 1), ctx(16),)
        .is_err());
    assert!(source.sketch.circles().is_empty());
    assert!(source.sketch.constraints().is_empty());
}
