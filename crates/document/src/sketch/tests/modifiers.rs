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

#[test]
fn chamfer_supports_equal_and_independent_leg_distances_then_commits_one_connector() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first_far = sketch.add_free_point(SketchPoint::new(10, 0));
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let second_far = sketch.add_free_point(SketchPoint::new(0, 10));
    let first = sketch.connect(first_far, corner).unwrap();
    let second = sketch.connect(corner, second_far).unwrap();
    let source = SketchSolid::extrude(sketch, 3);

    let equal = source
        .chamfer_placement(SketchCurve::Segment(first), [2.0, 0.1], None, ctx(16))
        .unwrap();
    assert_eq!(equal.connector.start(), [2.0, 0.0]);
    assert_eq!(equal.connector.end(), [0.0, 2.0]);

    let independent = source
        .chamfer_placement(
            SketchCurve::Segment(first),
            [2.0, 0.1],
            Some([0.1, 4.0]),
            ctx(16),
        )
        .unwrap();
    assert_eq!(independent.connector.start(), [2.0, 0.0]);
    assert_eq!(independent.connector.end(), [0.0, 4.0]);
    let made = source
        .with_corner_chamfered(
            SketchCurve::Segment(first),
            [2.0, 0.1],
            Some([0.1, 4.0]),
            ctx(16),
        )
        .unwrap();
    assert!(made.sketch.segments().iter().any(|held| held.id == first));
    assert!(made.sketch.segments().iter().any(|held| held.id == second));
    assert_eq!(made.sketch.segments().len(), 3);
    let connector = made
        .sketch
        .segments()
        .iter()
        .find(|held| held.id != first && held.id != second)
        .unwrap();
    let from = made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == connector.from)
        .unwrap()
        .at
        .in_plane();
    let to = made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == connector.to)
        .unwrap()
        .at
        .in_plane();
    assert_eq!(from, [2.0, 0.0]);
    assert_eq!(to, [0.0, 4.0]);
}

#[test]
fn offset_keeps_segments_arcs_and_circles_in_their_native_curve_families() {
    let mut line_sketch = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut line_sketch, [0, 0], [5, 0]);
    let source = SketchSolid::extrude(line_sketch, 3);
    let placement = source
        .offset_placement(SketchCurve::Segment(line), [2.0, 3.0], ctx(16))
        .unwrap();
    assert_eq!(placement.offset.start(), [0.0, 3.0]);
    assert_eq!(placement.offset.end(), [5.0, 3.0]);
    let made = source
        .with_curve_offset(SketchCurve::Segment(line), [2.0, 3.0], ctx(16))
        .unwrap();
    assert_eq!(made.sketch.segments().len(), 2);

    let mut arc_sketch = Sketch::empty(PlaneAxis::Z);
    let from = arc_sketch.add_free_point(SketchPoint::new(5, 0));
    let to = arc_sketch.add_free_point(SketchPoint::new(0, 5));
    let arc = arc_sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .unwrap();
    let source = SketchSolid::extrude(arc_sketch, 3);
    let placement = source
        .offset_placement(SketchCurve::Arc(arc), [8.0, 0.0], ctx(16))
        .unwrap();
    let PlanarCurve::Arc {
        radius,
        sweep_radians,
        ..
    } = placement.offset
    else {
        panic!("an arc offset must remain an arc");
    };
    assert!((radius - 8.0).abs() < 1.0e-9);
    assert!((sweep_radians.to_degrees() - 90.0).abs() < 1.0e-9);
    let made = source
        .with_curve_offset(SketchCurve::Arc(arc), [8.0, 0.0], ctx(16))
        .unwrap();
    assert_eq!(made.sketch.arcs().len(), 2);

    let mut circle_sketch = Sketch::empty(PlaneAxis::Z);
    let circle = circle_sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(5))
        .unwrap();
    let source = SketchSolid::extrude(circle_sketch, 3);
    let made = source
        .with_curve_offset(SketchCurve::Circle(circle), [0.0, 8.0], ctx(16))
        .unwrap();
    assert_eq!(made.sketch.circles().len(), 2);
    assert!((made.sketch.circles()[1].resolved_radius(ctx(16)) - 8.0).abs() < 1.0e-9);
}

#[test]
fn offset_refuses_zero_distance_without_mutating_the_source() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut sketch, [0, 0], [5, 0]);
    let source = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        source.with_curve_offset(SketchCurve::Segment(line), [2.0, 0.0], ctx(16)),
        Err(OffsetRefusal::ZeroDistance)
    );
    assert_eq!(source.sketch.segments().len(), 1);
}

#[test]
fn move_and_copy_transform_curve_closures_with_the_right_identity_policy() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut sketch, [0, 0], [5, 0]);
    let source = SketchSolid::extrude(sketch, 3);
    let selected = [SketchTransformEntity::Curve(SketchCurve::Segment(line))];

    let moved = source
        .with_entities_translated(&selected, [3.0, 4.0], false)
        .unwrap();
    assert_eq!(moved.sketch.segments().len(), 1);
    assert_eq!(moved.sketch.segments()[0].id, line);
    let geometry = moved
        .sketch
        .curve_geometry(SketchCurve::Segment(line), ctx(16))
        .unwrap();
    assert_eq!(
        geometry,
        ::parametric::sketch::CurveGeometry::Segment {
            from: [3.0, 4.0],
            to: [8.0, 4.0]
        }
    );

    let copied = source
        .with_entities_translated(&selected, [3.0, 4.0], true)
        .unwrap();
    assert_eq!(copied.sketch.segments().len(), 2);
    assert_ne!(copied.sketch.segments()[1].id, line);
    assert_eq!(
        copied.sketch.segments()[1].origin,
        copied.sketch.segments()[1].id
    );
}

#[test]
fn scale_changes_selected_points_and_free_circle_radius_about_one_center() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut sketch, [1, 1], [3, 1]);
    let circle = sketch
        .add_circle(SketchPoint::new(2, 2), SketchLength::new(2))
        .unwrap();
    let source = SketchSolid::extrude(sketch, 3);
    let selected = [
        SketchTransformEntity::Curve(SketchCurve::Segment(line)),
        SketchTransformEntity::Curve(SketchCurve::Circle(circle)),
    ];
    let scaled = source
        .with_entities_scaled(&selected, [1.0, 1.0], 2.0)
        .unwrap();
    let preview = source
        .scaled_curve_preview(&selected, [1.0, 1.0], 2.0, ctx(16))
        .unwrap();
    let preview_circle = preview.iter().find(|curve| curve.is_closed()).unwrap();
    let PlanarCurve::Arc {
        center: preview_center,
        radius: preview_radius,
        ..
    } = *preview_circle
    else {
        panic!("the scaled preview must keep its circle native");
    };
    assert_eq!(preview_center, [3.0, 3.0]);
    assert!((preview_radius - 4.0).abs() < 1.0e-9);
    let line_geometry = scaled
        .sketch
        .curve_geometry(SketchCurve::Segment(line), ctx(16))
        .unwrap();
    assert_eq!(
        line_geometry,
        ::parametric::sketch::CurveGeometry::Segment {
            from: [1.0, 1.0],
            to: [5.0, 1.0]
        }
    );
    let circle_geometry = scaled
        .sketch
        .curve_geometry(SketchCurve::Circle(circle), ctx(16))
        .unwrap();
    let ::parametric::sketch::CurveGeometry::Circular(circle_geometry) = circle_geometry else {
        panic!("a scaled circle must remain circular");
    };
    assert_eq!(circle_geometry.center, [3.0, 3.0]);
    assert!((circle_geometry.radius - 4.0).abs() < 1.0e-9);
}

#[test]
fn move_and_scale_refuse_constrained_geometry_without_deleting_the_assertion() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let line = segment(&mut sketch, [0, 0], [5, 0]);
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment: line }, ctx(16))
        .unwrap();
    let source = SketchSolid::extrude(sketch, 3);
    let selected = [SketchTransformEntity::Curve(SketchCurve::Segment(line))];
    assert_eq!(
        source.with_entities_translated(&selected, [1.0, 1.0], false),
        Err(SketchTransformRefusal::ConstrainedSelection)
    );
    assert_eq!(
        source.with_entities_scaled(&selected, [0.0, 0.0], 2.0),
        Err(SketchTransformRefusal::ConstrainedSelection)
    );
    assert_eq!(
        source.translated_curve_preview(&selected, [1.0, 1.0], false, ctx(16)),
        Err(SketchTransformRefusal::ConstrainedSelection)
    );
    assert!(source
        .translated_curve_preview(&selected, [1.0, 1.0], true, ctx(16))
        .is_ok());
    assert_eq!(
        source.selection_scale_radius(&selected, [0.0, 0.0], ctx(16)),
        Err(SketchTransformRefusal::ConstrainedSelection)
    );
    assert_eq!(source.sketch.constraints().len(), 1);
}
