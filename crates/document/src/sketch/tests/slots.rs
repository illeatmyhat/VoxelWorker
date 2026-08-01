use super::*;

fn source() -> SketchSolid {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
}

#[test]
fn every_linear_slot_grammar_commits_two_lines_and_two_native_caps() {
    for (kind, first, second) in [
        (
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
        ),
        (
            ::parametric::sketch::LinearSlotKind::Overall,
            SketchPoint::new(0, 0),
            SketchPoint::new(8, 0),
        ),
        (
            ::parametric::sketch::LinearSlotKind::CenterPoint,
            SketchPoint::new(0, 0),
            SketchPoint::new(3, 0),
        ),
    ] {
        let source = source();
        let placement = source
            .linear_slot_placement(kind, first, second, SketchPoint::new(0, 1))
            .unwrap();
        let made = source
            .with_linear_slot(kind, first, second, SketchPoint::new(0, 1))
            .unwrap();
        assert_eq!(made.sketch.segments().len(), 2);
        assert_eq!(made.sketch.arcs().len(), 2);
        assert_eq!(made.sketch.region(ctx(16)).len(), 1);
        for edge in placement.edges {
            let (from, to) = match edge {
                SlotEdgePlacement::Line { from, to } | SlotEdgePlacement::Arc { from, to, .. } => {
                    (from, to)
                }
            };
            assert!(made.sketch.point_at(from).is_some());
            assert!(made.sketch.point_at(to).is_some());
        }
    }
}

#[test]
fn both_arc_slots_commit_four_native_arcs_with_preview_identity() {
    let source = source();
    let three_point = source
        .three_point_arc_slot_placement(
            SketchPoint::new(2, 0),
            SketchPoint::new(0, 2),
            SketchPoint::from_continuous(2.0_f64.sqrt(), 2.0_f64.sqrt()),
            SketchPoint::new(3, 0),
        )
        .unwrap();
    let made = source
        .with_three_point_arc_slot(
            SketchPoint::new(2, 0),
            SketchPoint::new(0, 2),
            SketchPoint::from_continuous(2.0_f64.sqrt(), 2.0_f64.sqrt()),
            SketchPoint::new(3, 0),
        )
        .unwrap();
    assert_eq!(made.sketch.arcs().len(), 4);
    assert_eq!(made.sketch.region(ctx(16)).len(), 1);
    for edge in three_point.edges {
        let SlotEdgePlacement::Arc { from, to, .. } = edge else {
            panic!("arc slot has only curved edges")
        };
        assert!(made.sketch.point_at(from).is_some());
        assert!(made.sketch.point_at(to).is_some());
    }

    let centered = source
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 0),
            SketchPoint::new(0, 2),
            SketchPoint::new(3, 0),
        )
        .unwrap();
    assert_eq!(centered.sketch.arcs().len(), 4);
    assert_eq!(centered.sketch.region(ctx(16)).len(), 1);
}

#[test]
fn invalid_and_duplicate_slots_refuse_atomically() {
    let source = source();
    assert!(source
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 1),
        )
        .is_err());
    assert!(source.sketch.points().is_empty());

    let made = source
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
        )
        .unwrap();
    assert!(made
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
        )
        .is_err());
}
