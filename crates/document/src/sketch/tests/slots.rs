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
        // Overall draws a third line: the construction spine down its middle, which the author
        // asked for by clicking the extremes and which never bounds the region.
        let spine_line = usize::from(kind == ::parametric::sketch::LinearSlotKind::Overall);
        assert_eq!(made.sketch.segments().len(), 2 + spine_line);
        assert_eq!(
            made.sketch
                .segments()
                .iter()
                .filter(|segment| segment.role == EntityRole::Construction)
                .count(),
            spine_line
        );
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

/// The width is the one freedom a slot's relations leave open, and dragging a rail is how an
/// author spends it. The rail follows the cursor; the spine it was drawn from does not go with it.
/// An Overall Slot is authored by its two far ends, so the drawing keeps them: a construction line
/// down the middle joining them, the cap centers held on that line, and each end held on its cap.
/// The width is still the one thing nothing pins.
#[test]
fn an_overall_slot_keeps_its_extremes_on_a_construction_line() {
    let made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::Overall,
            SketchPoint::new(0, 0),
            SketchPoint::new(20, 0),
            SketchPoint::new(0, 4),
            ctx(16),
        )
        .expect("a valid overall slot");

    let line = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.role == EntityRole::Construction)
        .copied()
        .expect("the middle is drawn as a construction line");
    let position = |id: EntityId| {
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("a named point exists")
    };
    let (tail, head) = (position(line.from), position(line.to));
    for end in [tail, head] {
        assert!(
            (end[0] - 0.0).abs() < 1.0e-6 || (end[0] - 20.0).abs() < 1.0e-6,
            "the line runs between the authored extremes, not the cap centers: {end:?}"
        );
    }

    // Every cap center is on that line, which is what the relations say and what the author sees.
    let span = [head[0] - tail[0], head[1] - tail[1]];
    let length = span[0].hypot(span[1]);
    for arc in made.sketch.arcs() {
        let center = position(arc.center);
        let off = ((center[1] - tail[1]) * span[0] - (center[0] - tail[0]) * span[1]) / length;
        assert!(
            off.abs() < 1.0e-6,
            "cap center {center:?} off the line by {off}"
        );
    }

    // Four new rows against four new coordinates, so the width is still free to drag.
    let rail = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.role == EntityRole::Real)
        .map(|segment| SketchCurve::Segment(segment.id))
        .expect("a rail");
    let mut widened = made.clone();
    assert!(widened
        .sketch
        .move_curve(rail, [10.0, 7.0], ctx(16))
        .expect("the rail drag is answered"));
}

#[test]
fn dragging_a_rail_widens_the_slot_without_moving_its_spine() {
    let mut made = arc_slot();
    let center = handle_at(&made, SketchPoint::new(0, 0));
    // The outer rail of a quarter-turn slot of half-width two, drawn about the origin at radius 8.
    let outer = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| {
            let from = made
                .sketch
                .points()
                .iter()
                .find(|point| point.id == arc.from)
                .map(|point| point.at.in_plane());
            from.is_some_and(|at| (at[0].hypot(at[1]) - 10.0).abs() < 1.0e-6)
        })
        .map(|arc| SketchCurve::Arc(arc.id))
        .expect("the slot's outer rail");

    assert!(made.sketch.move_curve(outer, [13.0, 0.0], ctx(16)).unwrap());

    let position = |id: EntityId| {
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("the point survives the drag")
    };
    let center_at = position(center);
    assert!(center_at[0].hypot(center_at[1]) < 1.0e-6, "{center_at:?}");
    let SketchCurve::Arc(outer_id) = outer else {
        panic!("the rail is an arc")
    };
    let widened = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.id == outer_id)
        .map(|arc| position(arc.from))
        .expect("the rail survives its own drag");
    assert!(
        (widened[0].hypot(widened[1]) - 13.0).abs() < 1.0e-6,
        "{widened:?} should stand where the cursor left the rail"
    );
}

/// A drag reaches its own shape and stops there.
///
/// This is a COST invariant, not a correctness one — geometry no relation connects could never
/// have moved either way. It is worth pinning because the kernel prices a solve by how big the
/// problem is: measured on eight unrelated arc slots, one drag was 177ms whole-drawing against
/// 1ms scoped. If the walk ever starts dragging in the rest of the plane, that returns silently.
#[test]
fn a_drag_reaches_its_own_slot_and_no_further() {
    let mut made = arc_slot();
    let mine: Vec<EntityId> = made.sketch.points().iter().map(|point| point.id).collect();
    made = made
        .with_center_arc_slot(
            SketchPoint::new(60, 0),
            SketchPoint::new(68, 0),
            SketchPoint::new(60, 8),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(70, 0),
            ctx(16),
        )
        .expect("a second slot, well clear of the first");

    let center = handle_at(&made, SketchPoint::new(0, 0));
    let reach = made.sketch.what_a_drag_of_these_can_reach(&[center]);
    assert!(
        reach.iter().all(|point| mine.contains(point)),
        "the drag reached the other slot: {reach:?} against {mine:?}"
    );
    // The whole of its OWN slot, though — anything less would cut a shape in half.
    assert_eq!(reach.len(), mine.len());
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
