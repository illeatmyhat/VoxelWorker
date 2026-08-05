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

/// A slot's spine handles are the slot's, not the drawing's: deleting the boundary takes them
/// with it instead of leaving loose dots where the slot used to be.
#[test]
fn deleting_a_slots_boundary_takes_its_spine_handles_with_it() {
    let made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .unwrap();
    let mut sketch = made.sketch;
    // The two cap centers are derived, the two handles sit on top of them, and the four boundary
    // corners are shared by a rail and a cap.
    assert_eq!(sketch.points().len(), 8);

    for arc in sketch.arcs().iter().map(|arc| arc.id).collect::<Vec<_>>() {
        sketch.delete_arc(arc);
    }
    for segment in sketch
        .segments()
        .iter()
        .map(|segment| segment.id)
        .collect::<Vec<_>>()
    {
        sketch.delete_segment(segment);
    }

    assert!(
        sketch.points().is_empty(),
        "the slot left {:?} behind",
        sketch.points()
    );
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
        // Every grammar draws its centerline as construction, and exactly one. Overall's runs
        // further out — all the way to the extremes it was authored by — but it is still the
        // middle of the slot, passing through both cap centers on its way.
        assert_eq!(made.sketch.segments().len(), 2 + 1);
        assert_eq!(
            made.sketch
                .segments()
                .iter()
                .filter(|segment| segment.role == EntityRole::Construction)
                .count(),
            1
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
    assert_eq!(native_arcs(&made), 4);
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
    assert_eq!(native_arcs(&centered), 4);
    assert_eq!(centered.sketch.region(ctx(16)).len(), 1);
}

/// The boundary arcs, leaving out the construction one a turning slot draws down its middle.
fn native_arcs(made: &SketchSolid) -> usize {
    made.sketch
        .arcs()
        .iter()
        .filter(|arc| arc.role != EntityRole::Construction)
        .count()
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

/// Every grammar draws the centerline it was authored around, cap center to cap center, as
/// construction — and a turning slot's is an ARC, turning about the slot's own center rather than
/// cutting the corner as a chord would (owner, 2026-08-03).
///
/// The boundary is exactly the curve that does not contain the spine, so before this the
/// centerline was a thing the author reasoned in and the drawing did not have.
#[test]
fn every_slot_grammar_draws_its_centerline_as_construction() {
    let straight = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .expect("a valid center-to-center slot");
    let line = straight
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.role == EntityRole::Construction)
        .expect("the centerline is drawn");
    let at = |id: EntityId| {
        straight
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("a named point exists")
    };
    let ends = [at(line.from), at(line.to)];
    for want in [[0.0, 0.0], [6.0, 0.0]] {
        assert!(
            ends.iter()
                .any(|end| (end[0] - want[0]).hypot(end[1] - want[1]) < 1.0e-6),
            "the centerline runs cap center to cap center: {ends:?}"
        );
    }

    // The quarter-turn slot of `arc_slot`: radius 8 about the origin, so its centerline is a
    // quarter arc of radius 8 and NOT the chord, which would fall short by nearly two and a half
    // blocks in the middle.
    let curved = arc_slot();
    let spine = curved
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.role == EntityRole::Construction)
        .expect("a turning slot's centerline is an arc");
    let center = curved
        .sketch
        .points()
        .iter()
        .find(|point| point.id == spine.center)
        .map(|point| point.at.in_plane())
        .expect("an arc derives its center");
    assert!(
        center[0].hypot(center[1]) < 1.0e-6,
        "the centerline turns about the slot's own center: {center:?}"
    );
    let radius = curved
        .sketch
        .points()
        .iter()
        .find(|point| point.id == spine.from)
        .map(|point| point.at.in_plane())
        .map(|from| (from[0] - center[0]).hypot(from[1] - center[1]))
        .expect("the arc has a start");
    assert!(
        (radius - 8.0).abs() < 1.0e-5,
        "the centerline runs between the cap centers, at their own radius: {radius}"
    );
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
    // A hundred-thousandth of a block, against an eight-block pull. The bound was ten times
    // tighter before the centerline was drawn: a turning slot's spine is an arc, an arc derives a
    // center, and that center joins the least-motion objective at the same spot the slot's own
    // center sits — so the solve is pulled by one more point than it used to be. Six ten-thousandths
    // of a VOXEL, and the claim under test is the difference between reshaping and translating an
    // eight-block slot.
    assert!(center_at[0].hypot(center_at[1]) < 1.0e-5, "{center_at:?}");
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

    let position = |id: EntityId| {
        made.sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("a named point exists")
    };
    // ONE construction line runs down this slot, out to the extremes and through both cap centers
    // on the way. The shorter cap-to-cap curve every other grammar draws would land on top of it.
    let construction: Vec<Segment> = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.role == EntityRole::Construction)
        .copied()
        .collect();
    assert_eq!(construction.len(), 1, "one line down the middle, not two");
    let line = construction[0];
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
    // The far rail MIRRORS: the spine held at radius 8, so a rail pulled 3 out puts the other 3
    // in. Left free it drifted the wrong way instead — 6 to 5.4 for this same pull.
    let inner = made
        .sketch
        .arcs()
        .iter()
        .filter(|arc| arc.id != outer_id)
        .map(|arc| position(arc.from))
        .map(|at| at[0].hypot(at[1]))
        .find(|reach| (*reach - 3.0).abs() < 1.0e-6);
    assert!(
        inner.is_some(),
        "the far rail should have mirrored to radius 3: {:?}",
        made.sketch
            .arcs()
            .iter()
            .map(|arc| position(arc.from))
            .map(|at| at[0].hypot(at[1]))
            .collect::<Vec<_>>()
    );
}

/// A slot widens SYMMETRICALLY about the centerline it was drawn around (owner, 2026-08-04): pull
/// one rail and the other mirrors it, the construction spine does not move, and the length is
/// untouched.
///
/// Nothing asserts the rails are equidistant. The spine is pinned as a second hand and the rest
/// follows from the tangency web: a cap is a circle whose center is equidistant from its own ends
/// by construction, and that center stands on the spine. Left unpinned, least motion spent the
/// slack across everything — measured, a 2.0 pull moved the far rail 0.4 the WRONG way and slid
/// the centerline 0.8, so the slot stayed a slot and stopped being the one the author drew.
#[test]
fn dragging_a_rail_widens_a_linear_slot_about_a_centerline_that_holds() {
    let mut made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 2),
            ctx(16),
        )
        .expect("a center-to-center slot, six long and four wide");
    let rail = made
        .sketch
        .segments()
        .iter()
        .find(|segment| {
            segment.role != EntityRole::Construction
                && made
                    .sketch
                    .point_in_plane(segment.from)
                    .is_some_and(|at| at[1] > 1.0)
        })
        .map(|segment| SketchCurve::Segment(segment.id))
        .expect("the +Y rail");

    assert!(made
        .sketch
        .move_curve(rail, [3.0, 4.0], ctx(16))
        .expect("the rail drag is answered"));

    let corners: Vec<[f64; 2]> = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.role != EntityRole::Construction)
        .filter_map(|segment| made.sketch.point_in_plane(segment.from))
        .collect();
    for want in [4.0_f64, -4.0] {
        assert!(
            corners.iter().any(|at| (at[1] - want).abs() < 1.0e-6),
            "a rail should stand at y={want}: {corners:?}"
        );
    }
    // The spine, cap center to cap center: still on the axis, still six long.
    let spine = made
        .sketch
        .segments()
        .iter()
        .find(|segment| segment.role == EntityRole::Construction)
        .expect("the centerline");
    let (tail, head) = (
        made.sketch
            .point_in_plane(spine.from)
            .expect("the centerline stands"),
        made.sketch
            .point_in_plane(spine.to)
            .expect("the centerline stands"),
    );
    assert!(
        tail[1].abs() < 1.0e-6 && head[1].abs() < 1.0e-6,
        "the centerline slid off the axis: {tail:?} to {head:?}"
    );
    assert!(
        ((head[0] - tail[0]).hypot(head[1] - tail[1]) - 6.0).abs() < 1.0e-6,
        "a widen changed the length: {tail:?} to {head:?}"
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

/// An Overall Slot's two authored EXTREMES are not handles. Each is the end of the middle
/// construction line AND the place that line crosses a cap, so the drawing already says where it
/// is — while the cap centers, which the drawing says nothing about, keep their dots.
#[test]
fn an_overall_slots_extremes_go_quiet_and_its_cap_centers_do_not() {
    let made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::Overall,
            SketchPoint::new(0, 0),
            SketchPoint::new(20, 0),
            SketchPoint::new(0, 4),
            ctx(16),
        )
        .expect("a valid overall slot");
    let sketch = &made.sketch;
    let at = |id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("a named point exists")
    };
    // The extremes are the two picks; the cap centers are inset from them by half the width.
    let extremes: Vec<EntityId> = sketch
        .points()
        .iter()
        .filter(|point| {
            let [x, y] = point.at.in_plane();
            y.abs() < 1.0e-6 && (x.abs() < 1.0e-6 || (x - 20.0).abs() < 1.0e-6)
        })
        .map(|point| point.id)
        .collect();
    let handles: Vec<EntityId> = sketch
        .points()
        .iter()
        .filter(|point| {
            let [x, y] = point.at.in_plane();
            y.abs() < 1.0e-6 && ((x - 4.0).abs() < 1.0e-6 || (x - 16.0).abs() < 1.0e-6)
        })
        .filter(|point| !sketch.is_derived_point(point.id))
        .map(|point| point.id)
        .collect();

    assert_eq!(extremes.len(), 2, "two authored extremes: {extremes:?}");
    assert_eq!(handles.len(), 2, "two draggable cap-center handles");
    for extreme in extremes {
        assert!(
            !sketch.point_draws_at_rest(extreme),
            "the extreme at {:?} is doubly marked and needs no dot",
            at(extreme)
        );
    }
    for handle in handles {
        assert!(
            sketch.point_draws_at_rest(handle),
            "the cap center at {:?} is the grip the slot is held by",
            at(handle)
        );
    }
}

/// Which side the author picks the width on cannot change the shape.
///
/// A cap runs from one rail's corner to the other's and has to bulge OUT past the end of the slot.
/// The half-turn was hardcoded clockwise, which is right for one hand and reads as the inside turn
/// on the mirrored traversal — so picking the width below the spine bit a bite out of both ends. No
/// test that always picks its width on the same side of the spine can see that.
#[test]
fn a_straight_slots_caps_bulge_outward_whichever_side_the_width_was_picked() {
    for width_point in [SketchPoint::new(0, 2), SketchPoint::new(0, -2)] {
        let made = source()
            .with_linear_slot(
                ::parametric::sketch::LinearSlotKind::CenterToCenter,
                SketchPoint::new(0, 0),
                SketchPoint::new(6, 0),
                width_point,
                ctx(16),
            )
            .expect("a valid slot");
        let sketch = &made.sketch;
        let at = |id: EntityId| {
            sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .map(|point| point.at.in_plane())
                .expect("a named point exists")
        };
        // Each cap's crest: its start swung half its own sweep about its own center.
        let mut crests: Vec<f64> = sketch
            .arcs()
            .iter()
            .map(|arc| {
                let center = at(arc.center);
                let from = at(arc.from);
                let half = arc.sweep_degrees().to_radians() / 2.0;
                let (dx, dy) = (from[0] - center[0], from[1] - center[1]);
                dx.mul_add(half.cos(), -(dy * half.sin())) + center[0]
            })
            .collect();
        crests.sort_by(f64::total_cmp);

        assert_eq!(crests.len(), 2, "two caps");
        assert!(
            crests[0] < -1.0,
            "the start cap reaches back past the spine's start, not into the slot: \
             {crests:?} for a width picked at {width_point:?}"
        );
        assert!(
            crests[1] > 7.0,
            "the end cap reaches past the spine's end: \
             {crests:?} for a width picked at {width_point:?}"
        );
    }
}

fn straight_slot() -> SketchSolid {
    source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .expect("a six-block slot of half-width one")
}

fn centerline_of(made: &SketchSolid) -> EntityId {
    made.sketch
        .segments()
        .iter()
        .find(|segment| matches!(segment.role, EntityRole::Construction))
        .map(|segment| segment.id)
        .expect("every slot grammar draws its centerline")
}

/// How far the drawing strayed from carrying every point by exactly `by`.
fn worst_slip(made: &SketchSolid, before: &[(EntityId, [f64; 2])], by: [f64; 2]) -> f64 {
    before.iter().fold(0.0_f64, |worst, (id, was)| {
        let now = made
            .sketch
            .points()
            .iter()
            .find(|point| point.id == *id)
            .map(|point| point.at.in_plane())
            .expect("a translation loses no point");
        worst.max((now[0] - was[0] - by[0]).hypot(now[1] - was[1] - by[1]))
    })
}

fn stood_at(made: &SketchSolid) -> Vec<(EntityId, [f64; 2])> {
    made.sketch
        .points()
        .iter()
        .map(|point| (point.id, point.at.in_plane()))
        .collect()
}

/// The owner's report: "I can only drag a center-to-center slot up and down."
///
/// A body drag used to be a perpendicular OFFSET for every segment, which is the right gesture for
/// a rail — sideways off a boundary means nearer or further — and throws away the whole of what it
/// means for a centerline. Sliding a straight slot along its own length is a real motion the
/// drawing has the freedom for, and half of any diagonal one.
#[test]
fn a_straight_slots_centerline_carries_it_in_any_direction() {
    for by in [[4.0, 0.0], [0.0, 3.0], [4.0, 5.0], [-2.5, -1.5]] {
        let mut made = straight_slot();
        let centerline = centerline_of(&made);
        let before = stood_at(&made);
        assert!(
            made.sketch
                .translate_curve(SketchCurve::Segment(centerline), by, ctx(16))
                .unwrap(),
            "the drawing refused to be carried by {by:?}"
        );
        assert!(
            worst_slip(&made, &before, by) < 1.0e-4,
            "carried by {by:?} left something behind by {}",
            worst_slip(&made, &before, by)
        );
    }
}

/// Translating is a MOVE, so the width — the one freedom a slot leaves open — is not what the
/// motion gets spent on.
///
/// Least motion alone picks the opposite: the cheapest way to satisfy a hand is to move that point
/// and leave the rest, which is maximum deformation for minimum travel. Measured before the
/// gesture said which it was, a diagonal carry of 4 across and 5 up grew the half-width from 1.0
/// to 3.43.
#[test]
fn carrying_a_slot_does_not_spend_its_width() {
    let mut made = straight_slot();
    let centerline = centerline_of(&made);
    made.sketch
        .translate_curve(SketchCurve::Segment(centerline), [4.0, 5.0], ctx(16))
        .unwrap();
    let rails: Vec<f64> = made
        .sketch
        .points()
        .iter()
        .map(|point| point.at.in_plane()[1])
        .filter(|height| (height - 5.0).abs() > 1.0e-6)
        .collect();
    assert_eq!(rails.len(), 4, "four corners off the spine: {rails:?}");
    for height in rails {
        assert!(
            ((height - 5.0).abs() - 1.0).abs() < 1.0e-4,
            "the half-width is no longer one: {height}"
        );
    }
}

/// The owner's report: dragging an Overall Slot's centerline "has a large blast radius, causing all
/// of the other segments to change size."
///
/// The pin that holds a slot's spine while a RAIL widens was finding the cap centers standing on
/// the centerline and holding them there — against the hands carrying the very curve they stand on.
/// Half the hands said move and half said stay, so least squares split the difference: a 5.0 pull
/// arrived as 2.25, stretched the slot by 0.8 and grew its half-width by half again.
#[test]
fn dragging_an_overall_slots_centerline_does_not_fight_its_own_hands() {
    let mut made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::Overall,
            SketchPoint::new(0, 0),
            SketchPoint::new(6, 0),
            SketchPoint::new(0, 1),
            ctx(16),
        )
        .unwrap();
    let centerline = centerline_of(&made);
    let extremes: Vec<EntityId> = made
        .sketch
        .points()
        .iter()
        .filter(|point| {
            let at = point.at.in_plane();
            at[1].abs() < 1.0e-9 && (at[0].abs() < 1.0e-9 || (at[0] - 6.0).abs() < 1.0e-9)
        })
        .map(|point| point.id)
        .collect();
    assert_eq!(
        extremes.len(),
        2,
        "the slot keeps the ends the author picked"
    );

    assert!(made
        .sketch
        .move_curve(SketchCurve::Segment(centerline), [3.0, 5.0], ctx(16))
        .unwrap());

    for id in extremes {
        let at = made
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("an extreme survives the drag");
        assert!(
            (at[1] - 5.0).abs() < 1.0e-4,
            "an extreme went {} of the 5.0 it was pulled",
            at[1]
        );
        assert!(
            at[0].abs() < 1.0e-3 || (at[0] - 6.0).abs() < 1.0e-3,
            "and the slot stretched along its own length: {at:?}"
        );
    }
}

/// An arc slot's middle holds exactly two points, and draws one.
///
/// Its three arcs — both rails and the construction centerline — turn about one place, so they
/// SHARE one center rather than each echoing its own; three of the four dots this used to stack
/// were the same answer written again, and they are gone rather than hidden. The two that remain
/// cannot be collapsed: the shared center is derived, and a handle has to be draggable, so the
/// author is owed the handle and only the handle. Dragging a derived center authors the quantity
/// behind it instead of moving the slot, which would leave the dot most likely to be under the
/// cursor the one least able to answer for the gesture.
///
/// Both arc grammars are checked because the three-point one is sugar for the center-point one and
/// commits the same drawing; a difference between them would mean it had stopped being sugar.
#[test]
fn an_arc_slot_draws_one_dot_at_its_middle_and_it_is_the_draggable_one() {
    let three_point = source()
        .with_three_point_arc_slot(
            SketchPoint::new(8, 0),
            SketchPoint::new(0, 8),
            SketchPoint::from_continuous(8.0 / 2.0_f64.sqrt(), 8.0 / 2.0_f64.sqrt()),
            SketchPoint::new(10, 0),
            ctx(16),
        )
        .expect("a three-point arc slot");
    for (grammar, made) in [("three-point", &three_point), ("center-point", &arc_slot())] {
        let drawn: Vec<&Point> = made
            .sketch
            .points()
            .iter()
            .filter(|point| made.sketch.point_draws_at_rest(point.id))
            .filter(|point| {
                let at = point.at.in_plane();
                at[0].hypot(at[1]) < 1.0e-6
            })
            .collect();
        assert_eq!(
            drawn.len(),
            1,
            "{grammar} draws {} dots at its middle",
            drawn.len()
        );
        for point in drawn {
            assert!(
                !made.sketch.is_derived_point(point.id),
                "{grammar} kept the dot the author cannot drag"
            );
        }

        // Standing there, not merely drawn there: a dot suppressed is a dot the author can still
        // hover, select and wonder about, so the count that matters is of POINTS.
        let standing: Vec<&Point> = made
            .sketch
            .points()
            .iter()
            .filter(|point| point.at.in_plane()[0].hypot(point.at.in_plane()[1]) < 1.0e-3)
            .collect();
        assert_eq!(
            standing.len(),
            2,
            "{grammar} stacks {} points at its middle",
            standing.len()
        );
        let centers: std::collections::BTreeSet<EntityId> = made
            .sketch
            .arcs()
            .iter()
            .filter(|arc| {
                made.sketch
                    .point_in_plane(arc.center)
                    .is_some_and(|at| at[0].hypot(at[1]) < 1.0e-3)
            })
            .map(|arc| arc.center)
            .collect();
        assert_eq!(
            centers.len(),
            1,
            "{grammar} gave its concentric arcs {} centers",
            centers.len()
        );
    }
}

/// Resizing an arc slot must not put the extra dots back.
///
/// The three arcs share one center, and a shared center is only honest while they agree about
/// where it is. Nothing asserts that they do — the centerline's sweep is DERIVED from the center it
/// shares, so it cannot drift off the middle it is named for, and the rails are held concentric by
/// the relation they were always given. What this guards is the failure that hid behind the first
/// attempt at sharing: a tolerance test that split a shared center the moment two arcs disagreed
/// mid-solve, minting a point the author never drew and never putting it back.
#[test]
fn resizing_an_arc_slot_keeps_its_three_arcs_on_one_center() {
    let mut made = arc_slot();
    let rail = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| arc.role != EntityRole::Construction)
        .map(|arc| SketchCurve::Arc(arc.id))
        .expect("a rail to grab");
    let before = made.sketch.points().len();

    assert!(made
        .sketch
        .translate_curve(rail, [1.0, 1.0], ctx(16))
        .expect("the rail drag is answered"));

    assert_eq!(
        made.sketch.points().len(),
        before,
        "resizing minted a point"
    );

    // Five arcs about three centers: a cap at each end, and ONE for the two rails and the
    // centerline running between them.
    let mut about: std::collections::BTreeMap<EntityId, Vec<&Arc>> =
        std::collections::BTreeMap::new();
    for arc in made.sketch.arcs() {
        about.entry(arc.center).or_default().push(arc);
    }
    assert_eq!(made.sketch.arcs().len(), 5, "an arc slot draws five arcs");
    assert_eq!(
        about.len(),
        3,
        "five arcs came apart onto {} centers",
        about.len()
    );
    let shared = about
        .values()
        .find(|arcs| arcs.len() == 3)
        .expect("the rails and their centerline share one center");

    // Derived, not merely coincident: the centerline turns through the middle of what the rails do.
    let sweep = |wanted: EntityRole| {
        shared
            .iter()
            .filter(|arc| arc.role == wanted)
            .map(|arc| arc.sweep_degrees().abs())
            .fold(0.0_f64, f64::max)
    };
    let (rails, spine) = (sweep(EntityRole::Real), sweep(EntityRole::Construction));
    assert!(
        (rails - spine).abs() < 1.0e-2,
        "the centerline kept {spine} while its rails turned {rails}"
    );
}
