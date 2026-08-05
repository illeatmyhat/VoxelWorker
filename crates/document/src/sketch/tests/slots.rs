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
        // Every grammar draws its centerline as construction. Overall draws a SECOND one, further
        // out: the line between the extremes it was authored by, which answers a different
        // question — how long the slot is end to end.
        let reach_line = usize::from(kind == ::parametric::sketch::LinearSlotKind::Overall);
        assert_eq!(made.sketch.segments().len(), 2 + 1 + reach_line);
        assert_eq!(
            made.sketch
                .segments()
                .iter()
                .filter(|segment| segment.role == EntityRole::Construction)
                .count(),
            1 + reach_line
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
    // Two construction lines run down this slot: the centerline every grammar draws, cap center to
    // cap center, and the longer one out to the extremes that only Overall has. The one under test
    // is the latter, picked by the span it covers rather than by being the only one.
    let line = made
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.role == EntityRole::Construction)
        .max_by(|first, second| {
            let span = |segment: &Segment| {
                let (from, to) = (position(segment.from), position(segment.to));
                (to[0] - from[0]).hypot(to[1] - from[1])
            };
            span(first).total_cmp(&span(second))
        })
        .copied()
        .expect("the middle is drawn as a construction line");
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
