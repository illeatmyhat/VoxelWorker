use super::*;

fn source() -> SketchSolid {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
}

/// Whether the committed drawing carries a corner where the preview said it would.
///
/// NEAR, not at: a preview is continuous geometry, and canonical storage rounds it. Once the
/// tangencies are asserted the solver takes up that rounding — a few parts in a hundred million,
/// invisible against a voxel — so exact-position lookup is the wrong question to ask of a
/// constrained shape. What must hold is that the shape lands where the author aimed it.
fn holds_a_corner_near(made: &SketchSolid, want: SketchPoint) -> bool {
    let want = want.in_plane();
    made.sketch.points().iter().any(|point| {
        let at = point.at.in_plane();
        (at[0] - want[0]).hypot(at[1] - want[1]) < 1.0e-6
    })
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
            .with_linear_slot(kind, first, second, SketchPoint::new(0, 1), ctx(16))
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
            ctx(16),
        )
        .unwrap();
    assert_eq!(made.sketch.arcs().len(), 4);
    assert_eq!(made.sketch.region(ctx(16)).len(), 1);
    for edge in three_point.edges {
        let SlotEdgePlacement::Arc { from, to, .. } = edge else {
            panic!("arc slot has only curved edges")
        };
        assert!(holds_a_corner_near(&made, from));
        assert!(holds_a_corner_near(&made, to));
    }

    let centered = source
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 0),
            SketchPoint::new(0, 2),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(3, 0),
            ctx(16),
        )
        .unwrap();
    assert_eq!(centered.sketch.arcs().len(), 4);
    assert_eq!(centered.sketch.region(ctx(16)).len(), 1);
}

/// A slot is not four curves that happen to touch — it is four curves held together, around a
/// spine the drawing remembers. Every grammar commits a tangency at each corner, the one relation
/// that keeps the rails rails, and one coincidence per spine handle. The WIDTH is the freedom
/// deliberately left over.
#[test]
fn every_slot_grammar_commits_its_tangencies_rail_relation_and_spine() {
    let source = source();
    let straight = source
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .unwrap();
    let curved = source
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(8, 0),
            SketchPoint::new(0, 8),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(10, 0),
            ctx(16),
        )
        .unwrap();

    // A straight spine has two handles; a turning one also has the center it turns about.
    for (made, handles) in [(&straight, 2), (&curved, 3)] {
        let kinds = |wanted: fn(&ConstraintKind) -> bool| {
            made.sketch
                .constraints()
                .iter()
                .filter(|constraint| wanted(&constraint.kind))
                .count()
        };
        assert_eq!(
            kinds(|kind| matches!(kind, ConstraintKind::Tangent { .. })),
            4
        );
        assert_eq!(
            kinds(|kind| matches!(kind, ConstraintKind::Coincident { .. })),
            handles
        );
        assert_eq!(made.sketch.constraints().len(), 5 + handles);
        // The relations must be true of the geometry the tool just drew — a slot that has to be
        // solved into shape the moment it lands is a slot the tool got wrong.
        assert!(made.sketch.standing_constraints_hold(ctx(16)).unwrap());
    }

    assert!(straight
        .sketch
        .constraints()
        .iter()
        .any(|constraint| matches!(constraint.kind, ConstraintKind::Parallel { .. })));
    assert!(curved
        .sketch
        .constraints()
        .iter()
        .any(|constraint| matches!(constraint.kind, ConstraintKind::Concentric { .. })));
}

/// The handle an author drags to move the slot, found the way the UI finds it: the point standing
/// at that spot that is not one the drawing derives.
fn handle_at(made: &SketchSolid, at: SketchPoint) -> EntityId {
    made.sketch
        .points()
        .iter()
        .find(|point| !made.sketch.is_derived_point(point.id) && point.at.coincides(&at))
        .map(|point| point.id)
        .expect("the slot reifies its spine as draggable handles")
}

fn arc_slot() -> SketchSolid {
    source()
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(8, 0),
            SketchPoint::new(0, 8),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(10, 0),
            ctx(16),
        )
        .expect("a quarter-turn arc slot of half-width two")
}

/// The owner's contract for the center handle: it moves the SLOT, not part of it.
///
/// Least motion on one hand does not produce this and cannot be made to — the freedoms a slot
/// keeps on purpose are cheaper to spend than a translation is — so what this really pins is that
/// the drag policy is still in place. Measured before it existed, the same pull was a dead drag.
#[test]
fn dragging_the_center_translates_the_whole_arc_slot() {
    let mut made = arc_slot();
    let center = handle_at(&made, SketchPoint::new(0, 0));
    let before: Vec<[f64; 2]> = made
        .sketch
        .points()
        .iter()
        .map(|point| point.at.in_plane())
        .collect();

    assert!(made
        .sketch
        .move_point(center, SketchPoint::new(5, 3), ctx(16))
        .unwrap());

    for (was, now) in before.iter().zip(made.sketch.points()) {
        let now = now.at.in_plane();
        let slipped = (now[0] - was[0] - 5.0).hypot(now[1] - was[1] - 3.0);
        assert!(
            slipped < 1.0e-6,
            "{was:?} -> {now:?} is not the translation"
        );
    }
}

/// The other two handles reshape the arc rather than carrying the slot, which is the whole reason
/// the policy asks whether the held point is the shape's CENTER instead of translating on any
/// spine handle it finds.
#[test]
fn dragging_a_spine_end_reshapes_the_slot_instead_of_moving_it() {
    let mut made = arc_slot();
    let end = handle_at(&made, SketchPoint::new(0, 8));
    let center = handle_at(&made, SketchPoint::new(0, 0));

    assert!(made
        .sketch
        .move_point(
            end,
            SketchPoint::from_continuous(8.0 / 2.0_f64.sqrt(), 8.0 / 2.0_f64.sqrt()),
            ctx(16),
        )
        .unwrap());

    let center_at = made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == center)
        .map(|point| point.at.in_plane())
        .expect("the center survives its neighbour moving");
    assert!(center_at[0].hypot(center_at[1]) < 1.0e-6, "{center_at:?}");
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
            ctx(16),
        )
        .is_err());
    assert!(source.sketch.points().is_empty());

    let made = source
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .unwrap();
    assert!(made
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .is_err());
}
