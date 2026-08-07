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

/// A slot's spine points are the slot's, not the drawing's: deleting the boundary takes them
/// with it instead of leaving loose dots where the slot used to be.
#[test]
fn deleting_a_slots_boundary_takes_its_spine_with_it() {
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
    // The spine runs between the two cap centers, and the four boundary corners are each shared
    // by a rail and a cap.
    assert_eq!(sketch.points().len(), 6);

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
/// spine the drawing remembers. Every grammar commits a tangency at each corner and the one
/// relation that keeps the rails rails. It commits NO coincidence: the spine is drawn between the
/// boundary's own centers, so there is no second dot anywhere to be held onto a first. The WIDTH
/// is the freedom deliberately left over.
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

    for made in [&straight, &curved] {
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
            0
        );
        assert_eq!(made.sketch.constraints().len(), 5);
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

/// The point an author drags to move the slot, found the way the UI finds it: whatever is
/// standing at that spot. There is exactly one — a slot no longer stacks a handle on its spine.
fn spine_point_at(made: &SketchSolid, at: SketchPoint) -> EntityId {
    let want = at.in_plane();
    made.sketch
        .points()
        .iter()
        .find(|point| {
            let here = point.at.in_plane();
            (here[0] - want[0]).hypot(here[1] - want[1]) < 1.0e-6
        })
        .map(|point| point.id)
        .expect("the slot draws its spine between points that stand there")
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
    let center = spine_point_at(&made, SketchPoint::new(0, 0));
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

/// The other two spine points reshape the arc rather than carrying the slot, which is the whole
/// reason the policy asks whether the held point centers the shape ENTIRE instead of translating
/// on any spine point it finds.
#[test]
fn dragging_a_spine_end_reshapes_the_slot_instead_of_moving_it() {
    let mut made = arc_slot();
    let end = spine_point_at(&made, SketchPoint::new(0, 8));
    let center = spine_point_at(&made, SketchPoint::new(0, 0));

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

/// A slot's WIDTH is its cap radius, and nothing in the drawing dimensions it: four tangents and a
/// parallel force the two caps equal to each other and to half the rail separation, but leave the
/// separation itself free. So a cap carried far enough used to be paid for by widening the cap —
/// two points moved, against the four a rail turn would cost — and the slot changed shape while
/// being moved.
///
/// The radius holds because a hand that carries a cap ENTIRE — its center led, its two ends
/// following — is not authoring the radius, so the solve is given no column to spend on it.
/// Dragging the cap's BODY pins the center instead, and that gesture keeps its column.
///
/// Both halves are measured against what the drawing did before. An `Overall` cap pulled back to
/// twelve came out 8.5630 wide; it now comes out 8.0000, landing in the same place either way,
/// because the far cap is what stops it rather than the width. A `CenterToCenter` cap dragged
/// clean through its partner to −120 used to answer `Ok` with the cap 11.87 wide — the drawing
/// changing shape to buy an inversion — and now declines. Declining is what the same drag already
/// did at −20 and −60 before any of this: a slot cannot turn through itself, and the two targets
/// that answered were the arithmetic finding a shape the author never asked for.
#[test]
fn carrying_a_slots_cap_keeps_the_width_it_was_drawn_with() {
    let slot = |kind| {
        source()
            .with_linear_slot(
                kind,
                SketchPoint::new(8, 0),
                SketchPoint::new(32, 0),
                SketchPoint::new(8, 8),
                ctx(16),
            )
            .expect("a straight slot")
    };
    let position = |sketch: &Sketch, id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("the point survives the drag")
    };
    // The cap furthest along the spine, and the radius its own arc draws.
    let far_cap = |sketch: &Sketch| {
        sketch
            .arcs()
            .iter()
            .map(|arc| arc.center)
            .max_by(|first, second| {
                position(sketch, *first)[0].total_cmp(&position(sketch, *second)[0])
            })
            .expect("a cap center")
    };
    let width_of = |sketch: &Sketch, cap| {
        sketch
            .arcs()
            .iter()
            .find(|arc| arc.center == cap)
            .map(|arc| {
                let (hub, end) = (position(sketch, cap), position(sketch, arc.from));
                (end[0] - hub[0]).hypot(end[1] - hub[1])
            })
            .expect("the cap survives its own drag")
    };

    let mut carried = slot(::parametric::sketch::LinearSlotKind::Overall);
    let cap = far_cap(&carried.sketch);
    let drawn = width_of(&carried.sketch, cap);
    assert!((drawn - 8.0).abs() < 1.0e-6, "the slot drew {drawn} wide");
    assert!(carried
        .sketch
        .move_point(cap, SketchPoint::from_continuous(12.0, 0.0), ctx(16))
        .expect("the cap drag is answered"));
    let after = width_of(&carried.sketch, cap);
    assert!(
        (after - drawn).abs() < 1.0e-6,
        "a carried cap came out {after} wide, not the {drawn} it was drawn"
    );

    // Through its partner, and the drawing declines rather than reshaping itself to allow it.
    let mut inverted = slot(::parametric::sketch::LinearSlotKind::CenterToCenter);
    let cap = far_cap(&inverted.sketch);
    let refused =
        inverted
            .sketch
            .move_point(cap, SketchPoint::from_continuous(-120.0, 0.0), ctx(16));
    assert!(
        refused.is_err(),
        "a slot turned through itself answered {refused:?} with the cap {} wide",
        width_of(&inverted.sketch, cap)
    );
}

/// What the author ASSERTED outranks what the gesture merely keeps.
///
/// An undimensioned radius is held through a carry, but the hold is not a constraint and must never
/// read as one. Here one end of the arc is fixed outright, so carrying the center cannot leave the
/// radius alone — the end is not coming along. The drawing wins, and it wins by measurement rather
/// than by rank: the pass that holds the radius simply cannot converge, so its answer is dropped
/// and the pass without the hold stands.
#[test]
fn a_fixed_end_outranks_the_radius_a_carry_would_have_kept() {
    let arc_about_the_origin = || {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::new(40, 0));
        let to = sketch.add_free_point(SketchPoint::new(0, 40));
        let _ = sketch
            .connect_arc(from, to, AngleMeasurement::from_degrees(90))
            .expect("a quarter turn about the origin");
        let center = sketch.arcs().first().expect("the arc").center;
        (sketch, center, from)
    };
    let position = |sketch: &Sketch, id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("the point survives the drag")
    };
    let carry_the_center = |sketch: &mut Sketch, center, from| {
        assert!(sketch
            .move_point(center, SketchPoint::from_continuous(6.0, 0.0), ctx(16))
            .expect("the carry is answered"));
        let (hub, end) = (position(sketch, center), position(sketch, from));
        ((end[0] - hub[0]).hypot(end[1] - hub[1]), end)
    };

    let (mut loose, center, from) = arc_about_the_origin();
    let (kept, _) = carry_the_center(&mut loose, center, from);
    assert!(
        (kept - 40.0).abs() < 1.0e-6,
        "a carried arc should still be 40 across, not {kept}"
    );

    let (mut asserted, center, from) = arc_about_the_origin();
    asserted
        .add_constraint(
            ConstraintKind::Fix {
                point: from,
                at: SketchPoint::new(40, 0),
            },
            ctx(16),
        )
        .expect("nothing else is asserted, so it cannot conflict");
    let (given_up, stayed) = carry_the_center(&mut asserted, center, from);
    assert!(
        (stayed[0] - 40.0).hypot(stayed[1]) < 1.0e-6,
        "the fixed end moved to {stayed:?}"
    );
    // The drawing's own answer, to the digit: dropping the hold hands the gesture back exactly
    // as it ran before any of this existed, rather than to something reconstructed afterwards.
    assert!(
        (given_up - 36.952_184_332_805_295).abs() < 1.0e-9,
        "the hold should have given way to the fix, not held {given_up} across"
    );
}

#[test]
fn dragging_a_rail_widens_the_slot_without_moving_its_spine() {
    let mut made = arc_slot();
    let center = spine_point_at(&made, SketchPoint::new(0, 0));
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

    let center = spine_point_at(&made, SketchPoint::new(0, 0));
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
        .map(|point| point.id)
        .collect();

    assert_eq!(extremes.len(), 2, "two authored extremes: {extremes:?}");
    assert_eq!(handles.len(), 2, "two draggable cap centers");
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
                let half = sketch
                    .arc_form(arc)
                    .expect("a cap draws a circle")
                    .sweep_degrees
                    .to_radians()
                    / 2.0;
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

/// An arc slot's middle holds exactly ONE point, and it is the center its rails turn about.
///
/// Its three arcs — both rails and the construction centerline — turn about one place, so they
/// SHARE one center rather than each echoing its own; the dots this used to stack were the same
/// answer written again, and they are gone rather than hidden. The last of them to go was a
/// draggable handle standing on the derived center, kept because a derived center could not be
/// dragged. [ADR 0038](../../../../../docs/adr/0038-a-point-is-placed-never-computed.md) made
/// every point authored, so the center answers for the gesture itself and the handle was only a
/// second dot in one place for the rules to trip over.
///
/// Both arc grammars are checked because the three-point one is sugar for the center-point one and
/// commits the same drawing; a difference between them would mean it had stopped being sugar.
#[test]
fn an_arc_slot_draws_one_dot_at_its_middle_and_it_is_the_center() {
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
                made.sketch.is_arc_center(point.id),
                "{grammar} drew a dot at its middle that is not the center itself"
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
            1,
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
            .filter_map(|arc| Some(made.sketch.arc_form(arc)?.sweep_degrees))
            .fold(0.0_f64, f64::max)
    };
    let (rails, spine) = (sweep(EntityRole::Real), sweep(EntityRole::Construction));
    assert!(
        (rails - spine).abs() < 1.0e-2,
        "the centerline kept {spine} while its rails turned {rails}"
    );
}

/// Dragging a rail changes the slot's RADIUS, and the middle it turns about holds.
///
/// A rim is dragged by its radius — the rule a circle here has always had, and an arc is a circle
/// with two ends. It is worth a test because the arithmetic does not want to: carrying a shape
/// costs a least-deformation solve nothing and reshaping it costs something, so with only the grip
/// to go on the solve moved the whole slot and left its width exactly as it found it.
#[test]
fn dragging_an_arc_slots_rail_spends_its_radius_about_a_center_that_holds() {
    let mut made = arc_slot();
    let (rail, center) = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| {
            arc.role != EntityRole::Construction
                && made
                    .sketch
                    .arc_form(arc)
                    .is_some_and(|form| form.sweep_degrees < 170.0)
                && made
                    .sketch
                    .point_in_plane(arc.from)
                    .is_some_and(|at| at[0].hypot(at[1]) > 8.5)
        })
        .map(|arc| (SketchCurve::Arc(arc.id), arc.center))
        .expect("the outer rail, ten out from the middle");
    let stood_at = made.sketch.point_in_plane(center).expect("its middle");
    let out = |radius: f64| {
        [
            stood_at[0] + radius / 2.0_f64.sqrt(),
            stood_at[1] + radius / 2.0_f64.sqrt(),
        ]
    };

    assert!(made
        .sketch
        .drag_curve_through(rail, out(10.0), out(12.0), ctx(16))
        .expect("the rail drag is answered"));

    let now = made.sketch.point_in_plane(center).expect("its middle");
    // A ten-thousandth of a block — two thousandths of a VOXEL — against a two-block widening.
    // The center is held by a relation rather than by being left out of the solve, so it carries
    // the solver's dust; the claim under test is travelling versus widening, and this is neither.
    assert!(
        (now[0] - stood_at[0]).hypot(now[1] - stood_at[1]) < 1.0e-4,
        "the slot travelled instead of widening: {stood_at:?} -> {now:?}"
    );
    let reach = made
        .sketch
        .arcs()
        .iter()
        .find(|arc| SketchCurve::Arc(arc.id) == rail)
        .and_then(|arc| made.sketch.point_in_plane(arc.from))
        .map(|end| (end[0] - now[0]).hypot(end[1] - now[1]))
        .expect("the rail still stands");
    assert!(
        (reach - 12.0).abs() < 1.0e-4,
        "the rail should reach twelve out: {reach}"
    );
}

/// A line has no middle to turn about, so the same gesture carries it — and carries it BOTH ways.
///
/// Dragging a line along its own direction produces the same line, so the along part of a drag
/// cannot mean a deformation. That is not the same as meaning nothing: a line with nothing holding
/// it has every freedom to travel, and travelling is what it does. The distinction the gesture
/// draws is deform versus carry, never move versus stay still — a drawing that answered a body
/// drag by holding still read to the author as the shape being stuck (owner, 2026-08-05).
#[test]
fn dragging_a_lines_body_carries_it_whichever_way_it_is_pulled() {
    for (tag, from, to, want) in [
        ("across", [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]),
        ("along", [3.0, 0.0], [6.0, 0.0], [3.0, 0.0]),
    ] {
        let mut made = source();
        let tail = made.sketch.add_free_point(SketchPoint::new(0, 0));
        let head = made.sketch.add_free_point(SketchPoint::new(6, 0));
        let line = SketchCurve::Segment(made.sketch.connect(tail, head).expect("a line"));

        assert!(made
            .sketch
            .drag_curve_through(line, from, to, ctx(16))
            .expect("the drag is answered"));

        let at = made.sketch.point_in_plane(tail).expect("the tail stands");
        assert!(
            (at[0] - want[0]).abs() < 1.0e-6 && (at[1] - want[1]).abs() < 1.0e-6,
            "dragged {tag}, the tail went to {at:?} rather than {want:?}"
        );
    }
}

/// The whole of what a body drag decides, on the one shape that shows both answers.
///
/// A rectangle's four sides are held square to each other and nothing else, so which way its edge
/// is pulled settles what the drawing can do about it:
///
/// - pulled OUTWARD, moving that edge alone is available and costs two corners, so the rectangle
///   resizes and the far edge stays where the author left it;
/// - pulled SIDEWAYS, the vertical relations forbid the shear that would let one edge go on its
///   own, so the only answer left is to carry all four corners.
///
/// Neither is a rule about rectangles. The gesture states the same thing both times — the across
/// part of the pull seeds a deformation, and the whole of it pulls — and the two answers are the
/// relations', which is the point. See [`Sketch::what_a_body_drag_asks_of`].
#[test]
fn a_rectangles_edge_resizes_it_outward_and_carries_it_sideways() {
    let made = source()
        .with_rectangle(SketchPoint::new(0, 0), SketchPoint::new(40, 20), ctx(16))
        .expect("a rectangle");
    let bottom = made.sketch.segments()[0].id;
    let corners = |sketch: &Sketch| {
        let mut at: Vec<[f64; 2]> = sketch
            .points()
            .iter()
            .map(|point| point.at.in_plane())
            .collect();
        at.sort_by(|first, second| {
            first[0]
                .total_cmp(&second[0])
                .then(first[1].total_cmp(&second[1]))
        });
        at
    };
    // A voxel is 1/16 of a block and the drawing is exact to a ten-thousandth of one, so the claim
    // is about which corners moved, never about the last bit of the arithmetic.
    let stands_at = |got: Vec<[f64; 2]>, want: [[f64; 2]; 4]| {
        got.iter().zip(want).all(|(got, want)| {
            (got[0] - want[0]).abs() < 1.0e-4 && (got[1] - want[1]).abs() < 1.0e-4
        })
    };

    let mut outward = made.sketch.clone();
    assert!(outward
        .drag_curve_through(
            SketchCurve::Segment(bottom),
            [20.0, 0.0],
            [20.0, 6.0],
            ctx(16)
        )
        .expect("the drag is answered"));
    assert!(
        stands_at(
            corners(&outward),
            [[0.0, 6.0], [0.0, 20.0], [40.0, 6.0], [40.0, 20.0]]
        ),
        "pulled outward, the bottom edge rises alone and the top stays put: {:?}",
        corners(&outward)
    );

    let mut sideways = made.sketch.clone();
    assert!(sideways
        .drag_curve_through(
            SketchCurve::Segment(bottom),
            [20.0, 0.0],
            [26.0, 0.0],
            ctx(16)
        )
        .expect("the drag is answered"));
    assert!(
        stands_at(
            corners(&sideways),
            [[6.0, 0.0], [6.0, 20.0], [46.0, 0.0], [46.0, 20.0]]
        ),
        "pulled sideways, every corner carries and the rectangle keeps its size: {:?}",
        corners(&sideways)
    );
}

/// A straight slot's rail, whose two answers the author actually spends.
///
/// Across, the rail is how the slot's width is authored, and it widens SYMMETRICALLY — the spine
/// takes half, so the far rail mirrors the grabbed one. Along, the slot has the freedom to slide
/// and does, with the width EXACTLY unchanged rather than merely nearly so: the earlier reading
/// seeded the whole displacement, which broke the tangency web along the rail as well as across
/// it, and the repair leaked into the width by 0.12, then 0.50, then 0.94 over three equal steps.
#[test]
fn a_slots_rail_widens_it_across_and_slides_it_along() {
    let made = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(100, 100),
            SketchPoint::new(160, 100),
            SketchPoint::new(130, 106),
            ctx(16),
        )
        .expect("a straight slot");
    let rail = made.sketch.segments()[0].id;
    let width = |sketch: &Sketch| {
        sketch
            .arc_form(sketch.arcs().first().expect("a cap"))
            .expect("the cap reads")
            .radius
            * 2.0
    };
    let spine = |sketch: &Sketch| {
        sketch
            .point_in_plane(sketch.arcs().first().expect("a cap").center)
            .expect("the cap center stands")
    };
    let (was_wide, was_spine) = (width(&made.sketch), spine(&made.sketch));

    let mut across = made.sketch.clone();
    assert!(across
        .drag_curve_through(
            SketchCurve::Segment(rail),
            [130.0, 106.0],
            [130.0, 109.0],
            ctx(16)
        )
        .expect("the drag is answered"));
    assert!(
        (width(&across) - was_wide - 3.0).abs() < 1.0e-6,
        "three out on one rail is three of width: {}",
        width(&across) - was_wide
    );
    assert!(
        (spine(&across)[1] - was_spine[1] - 1.5).abs() < 1.0e-6,
        "the spine takes half, so the far rail mirrors: {:?}",
        spine(&across)
    );

    let mut along = made.sketch.clone();
    assert!(along
        .drag_curve_through(
            SketchCurve::Segment(rail),
            [130.0, 106.0],
            [138.0, 106.0],
            ctx(16)
        )
        .expect("the drag is answered"));
    assert!(
        (width(&along) - was_wide).abs() < 1.0e-9,
        "slid along its own rail, a slot keeps its width exactly: {}",
        width(&along) - was_wide
    );
    assert!(
        (spine(&along)[0] - was_spine[0] - 8.0).abs() < 1.0e-4,
        "and the whole of it travels: {:?}",
        spine(&along)
    );
}

/// A curved slot is the shape a long drag used to break, and the reason one is delivered as a walk.
///
/// Its relations have a FAMILY of exact answers — grow the caps outward, or balloon the inner rail
/// inward, both perfectly tangent — so a solve asked for a long motion in one go can cross to a
/// distant member of that family. Measured before the walk: a two-voxel pull on the outer rail of
/// a radius-forty slot threw its inner rail twenty-four voxels inward, and the step after failed a
/// tangency outright.
#[test]
fn a_curved_slot_widens_under_a_long_drag_rather_than_jumping_branch() {
    let made = source()
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(40, 0),
            SketchPoint::new(0, 40),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(44, 0),
            ctx(16),
        )
        .expect("a curved slot");
    let nearest_radius = |sketch: &Sketch, want: f64| {
        sketch
            .arcs()
            .iter()
            .filter_map(|arc| sketch.arc_form(arc))
            .map(|form| form.radius)
            .min_by(|first, second| (first - want).abs().total_cmp(&(second - want).abs()))
            .expect("the slot still has arcs")
    };
    let outer = made
        .sketch
        .arcs()
        .iter()
        .filter_map(|arc| Some((arc.id, made.sketch.arc_form(arc)?.radius)))
        .max_by(|first, second| first.1.total_cmp(&second.1))
        .expect("an outer rail")
        .0;

    let mut sketch = made.sketch.clone();
    assert!(sketch
        .drag_curve_through(SketchCurve::Arc(outer), [44.0, 0.0], [46.0, 0.0], ctx(16))
        .expect("the drag is answered"));
    // A thirty-thousandth of a voxel. The drawing has the LAST word over the hand — the third pass
    // drops the pull and re-solves the relations alone — so the rail lands on the relation manifold
    // nearest the hand rather than on the hand itself, and how near that is depends on the shape.
    assert!(
        (nearest_radius(&sketch, 46.0) - 46.0).abs() < 1.0e-4,
        "the grabbed rail goes where it was pulled: {}",
        nearest_radius(&sketch, 46.0)
    );
    // The inner rail is the one that used to fly. It stays within a fortieth of a voxel of home.
    assert!(
        (nearest_radius(&sketch, 36.0) - 36.0).abs() < 0.025,
        "the far rail holds: {}",
        nearest_radius(&sketch, 36.0)
    );
}

/// A curved slot of rails 36, 40 and 44, all turning about the origin, with its near cap standing
/// at [44, 0]. The gesture the tests below measure is that cap's corner being pulled sideways.
fn curved_slot() -> Sketch {
    source()
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(40, 0),
            SketchPoint::new(0, 40),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(44, 0),
            ctx(16),
        )
        .expect("a curved slot")
        .sketch
        .as_ref()
        .clone()
}

fn rails(sketch: &Sketch) -> Vec<f64> {
    let mut got: Vec<f64> = sketch
        .arcs()
        .iter()
        .filter_map(|arc| sketch.arc_form(arc))
        .map(|form| form.radius)
        .collect();
    got.sort_by(f64::total_cmp);
    got
}

/// The drawn point standing nearest a place.
fn corner_at(sketch: &Sketch, near: [f64; 2]) -> u32 {
    sketch
        .points()
        .iter()
        .min_by(|first, second| {
            let reach = |at: [f64; 2]| (at[0] - near[0]).hypot(at[1] - near[1]);
            reach(first.at.in_plane()).total_cmp(&reach(second.at.in_plane()))
        })
        .expect("a drawn point")
        .id
}

fn stands_at(sketch: &Sketch, point: u32) -> [f64; 2] {
    sketch
        .points()
        .iter()
        .find(|drawn| drawn.id == point)
        .expect("a drawn point")
        .at
        .in_plane()
}

/// Pulling a slot's corner ALONG its rail sweeps the near end round and leaves the far one alone.
///
/// Nothing here is dimensioned, so the hand could be met just as exactly by sliding the whole slot
/// sideways — and before the snap that is what happened, the far end coming 3.6 voxels along for a
/// six-voxel pull. What settles it is reading a hand that moves ALONG a quantity as one that is
/// keeping it: the cursor goes onto the circle the radius draws, which makes the sweep an exact
/// answer too, and a cheaper one than carrying the whole drawing.
#[test]
fn pulling_a_slots_corner_along_its_rail_sweeps_the_near_end_and_leaves_the_far_one() {
    let mut sketch = curved_slot();
    let corner = corner_at(&sketch, [44.0, 0.0]);
    let hub = corner_at(&sketch, [0.0, 0.0]);
    let far = corner_at(&sketch, [0.0, 44.0]);
    let inner = corner_at(&sketch, [36.0, 0.0]);

    assert!(sketch
        .move_point(corner, SketchPoint::from_continuous(44.0, 6.0), ctx(16))
        .expect("answered"));

    let moved = |point: u32, from: [f64; 2]| {
        let at = stands_at(&sketch, point);
        (at[0] - from[0]).hypot(at[1] - from[1])
    };
    assert!(
        moved(corner, [44.0, 0.0]) > 5.5,
        "the hand was not followed"
    );
    assert!(
        moved(inner, [36.0, 0.0]) > 4.0,
        "the near cap did not sweep"
    );
    // Not "hardly moved" — did not move. A corner names the hub it turns about as a hand that
    // stays put, the same way a spine end does, so the sweep has a pivot rather than a preference.
    assert!(moved(far, [0.0, 44.0]) < 1.0e-3, "the far end came along");
    assert!(moved(hub, [0.0, 0.0]) < 1.0e-3, "the hub came along");
    let swept = rails(&sketch);
    for (rail, want) in swept.iter().zip([4.0, 4.0, 36.0, 40.0, 44.0]) {
        assert!((rail - want).abs() < 0.5, "the rails came out {swept:?}");
    }
}

/// A drag's answer must not depend on how fast its frames arrived.
///
/// A snapped drag is a ROTATION, the motion a linearized solve is worst at, so a gesture handed
/// over in one jump used to settle somewhere quite different from the same gesture delivered a
/// frame at a time — the rails collapsing from 36/40/44 to 33.5/38.3/43.2 on the jump. A drag
/// therefore walks its turn in small steps rather than trusting the frame it was given.
#[test]
fn a_drags_answer_does_not_depend_on_how_fast_its_frames_arrived() {
    let walk = |frames: u32| {
        let mut sketch = curved_slot();
        let corner = corner_at(&sketch, [44.0, 0.0]);
        for frame in 1..=frames {
            let height = f64::from(frame) * 6.0 / f64::from(frames);
            sketch
                .move_point(corner, SketchPoint::from_continuous(44.0, height), ctx(16))
                .expect("answered");
        }
        sketch
    };

    let jumped = rails(&walk(1));
    for frames in [2_u32, 8, 24] {
        let walked = rails(&walk(frames));
        for (one, many) in jumped.iter().zip(&walked) {
            assert!(
                (one - many).abs() < 0.5,
                "{frames} frames answered {walked:?} where one answered {jumped:?}"
            );
        }
    }
}

/// Pulling a corner ACROSS its rail is the author setting the radius rather than keeping it, so
/// the snap has to let go — otherwise a slot could never be widened by its own corner.
#[test]
fn pulling_a_slots_corner_across_its_rail_lets_the_snap_go() {
    let mut sketch = curved_slot();
    let corner = corner_at(&sketch, [44.0, 0.0]);
    let before = rails(&sketch);

    assert!(sketch
        .move_point(corner, SketchPoint::from_continuous(50.0, 0.0), ctx(16))
        .expect("answered"));

    let at = stands_at(&sketch, corner);
    assert!(
        (at[0] - 50.0).abs() < 1.0e-6 && at[1].abs() < 1.0e-6,
        "the hand was not met exactly: {at:?}"
    );
    let after = rails(&sketch);
    let outer = |of: &[f64]| of.last().copied().expect("a rail");
    assert!(
        outer(&after) - outer(&before) > 2.0,
        "the rail did not grow with the corner: {after:?}"
    );
}

/// Both arc-slot grammars answer an author's three gestures the same way, and answer them the way
/// the author described: the middle carries the slot, an end sweeps and nothing else does, and the
/// end it sweeps keeps the radius it had.
///
/// The three used to be one gesture with three bad answers. An end handle was a two-hand gesture —
/// the point plus the pivot it turns about — and every rule that asked "is one vertex being
/// reshaped" counted the hands it was NAMED rather than the hand that MOVED, so the snap and the
/// stays both switched themselves off and the far end wandered five voxels for a six-voxel pull.
/// The three-point grammar is sugar for the center-point one, so a difference between the two
/// columns below would mean it had stopped being sugar.
#[test]
fn both_arc_slot_grammars_carry_by_the_middle_and_sweep_by_an_end() {
    let three = source()
        .with_three_point_arc_slot(
            SketchPoint::new(40, 0),
            SketchPoint::new(0, 40),
            SketchPoint::from_continuous(40.0 / 2.0_f64.sqrt(), 40.0 / 2.0_f64.sqrt()),
            SketchPoint::new(44, 0),
            ctx(16),
        )
        .expect("a three-point arc slot")
        .sketch
        .as_ref()
        .clone();
    for (grammar, base) in [("center-arc", curved_slot()), ("three-point", three)] {
        // Carrying it by the middle: every point takes the SAME step, and nothing about the slot
        // changes shape.
        let mut carried = base.clone();
        let hub = spine_dot_near(&carried, [0.0, 0.0]);
        let step = [7.0, 5.0];
        assert!(carried
            .move_point(hub, SketchPoint::from_continuous(step[0], step[1]), ctx(16))
            .expect("answered"));
        for point in base.points() {
            let was = point.at.in_plane();
            let is = stands_at(&carried, point.id);
            let slip = (is[0] - was[0] - step[0]).hypot(is[1] - was[1] - step[1]);
            assert!(
                slip < 1.0e-6,
                "{grammar} left p{} behind by {slip}",
                point.id
            );
        }

        // Sweeping it by an end: that end keeps its radius exactly, and the far end does not move.
        for (label, near, far) in [
            ("near", [40.0, 0.0], [0.0, 40.0]),
            ("far", [0.0, 40.0], [40.0, 0.0]),
        ] {
            let mut swept = base.clone();
            let end = spine_dot_near(&swept, near);
            let stays = [
                spine_dot_near(&swept, far),
                spine_dot_near(&swept, [0.0, 0.0]),
            ];
            // Six voxels straight out, which is ACROSS the radius by a fifteenth — well inside the
            // cone, so the hand is pulled back onto the circle it was already standing on.
            let to = [
                near[0] + 6.0 * near[1] / 40.0,
                near[1] + 6.0 * near[0] / 40.0,
            ];
            assert!(swept
                .move_point(end, SketchPoint::from_continuous(to[0], to[1]), ctx(16))
                .expect("answered"));
            let at = stands_at(&swept, end);
            let radius = at[0].hypot(at[1]);
            assert!(
                (radius - 40.0).abs() < 1.0e-5,
                "{grammar} {label} end left its radius at {radius}"
            );
            // A ten-thousandth of a voxel is the settle's own tolerance, not a motion.
            for held in stays {
                let stood = stands_at(&base, held);
                let now = stands_at(&swept, held);
                let moved = (now[0] - stood[0]).hypot(now[1] - stood[1]);
                assert!(moved < 1.0e-4, "{grammar} {label}: p{held} moved {moved}");
            }
            // And the slot does not FATTEN as it sweeps. The width is the freedom a slot keeps on
            // purpose, so it is the one thing a sweep can spend without breaking a relation, and
            // it used to: the cap center ran ahead of its own two corners and the cap stretched to
            // stay attached, about 5% per six-voxel pull. Carrying a cap as a rigid set — the
            // center and the corners it is the middle of, moving as one — leaves nothing to
            // stretch, and the rails come out exact.
            let rails = rails(&swept);
            for (rail, want) in rails.iter().zip([4.0, 4.0, 36.0, 40.0, 44.0]) {
                assert!(
                    (rail - want).abs() < 1.0e-4,
                    "{grammar} {label} swept its rails to {rails:?}"
                );
            }
        }
    }
}

/// The dot nearest a place, among the ones a slot draws — its spine, in other words.
fn spine_dot_near(sketch: &Sketch, at: [f64; 2]) -> EntityId {
    sketch
        .points()
        .iter()
        .filter(|point| sketch.point_draws_at_rest(point.id))
        .min_by(|first, second| {
            let reach = |stood: [f64; 2]| (stood[0] - at[0]).hypot(stood[1] - at[1]);
            reach(first.at.in_plane()).total_cmp(&reach(second.at.in_plane()))
        })
        .map(|point| point.id)
        .expect("a slot draws its spine")
}

/// A snapped drag says what it kept, so the overlay can draw the circle the hand is sliding along.
///
/// The author could not tell whether the snap was firing — "I can't really tell if it's snapping"
/// — and from the outside there is nothing to tell: a snap puts the point a little off the cursor,
/// which is exactly what a solve that could not reach does. So the drag reports the quantity, and
/// a pull straight off the circle reports none, which is the other half of the affordance.
#[test]
fn a_snapped_drag_reports_the_circle_it_kept() {
    let mut swept = curved_slot();
    let end = spine_dot_near(&swept, [40.0, 0.0]);
    // Six voxels along the sweep and a fifteenth of that across it — well inside the cone.
    let kept = swept
        .move_point_reporting_its_snap(
            end,
            SketchPoint::from_continuous(40.0, 6.0),
            ctx(16),
            SnapReach::UNBOUNDED,
        )
        .expect("answered")
        .kept
        .expect("a sweep along a radius keeps that radius");
    assert!(
        kept.about[0].hypot(kept.about[1]) < 1.0e-6,
        "the slot turns about the origin, not {:?}",
        kept.about
    );
    assert!(
        (kept.radius - 40.0).abs() < 1.0e-6,
        "the spine end stood 40 out, not {}",
        kept.radius
    );

    // Straight out along the radius is a pull ACROSS the quantity, not along it. Nothing is being
    // kept, so nothing is drawn, and the author feels the circle let go.
    let mut grown = curved_slot();
    let end = spine_dot_near(&grown, [40.0, 0.0]);
    let answered = grown
        .move_point_reporting_its_snap(
            end,
            SketchPoint::from_continuous(46.0, 0.0),
            ctx(16),
            SnapReach::UNBOUNDED,
        )
        .expect("answered");
    assert!(answered.moved);
    assert_eq!(
        answered.kept, None,
        "a pull straight off the circle keeps nothing"
    );
}

/// A slot's RAIL keeps the length it was drawn when the hand slides it.
///
/// The same undimensioned quantity as the cap's radius, given to the solve differently: a length
/// is not a column, it is only however far two ends happen to be apart, so nothing holds it and it
/// settles wherever the arithmetic leaves it. Measured on this slot before the hold, sliding a rail
/// out drifted 24.0000 to 23.7126, and to 24.3920 in the other direction for the same shape.
///
/// Nothing is being taken from the author here. A segment dragged by its BODY slides sideways —
/// both ends by the same offset — so its length was never what the gesture was setting, and unlike
/// an arc's radius there is no opposite gesture to tell apart.
#[test]
#[ignore = "the segment hold is switched off at the owner's request; see HOLD_A_CARRIED_SPAN"]
fn sliding_a_slots_rail_keeps_the_length_it_was_drawn() {
    let base = source()
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(8, 0),
            SketchPoint::new(32, 0),
            SketchPoint::new(8, 8),
            ctx(16),
        )
        .expect("a straight slot");
    let position = |sketch: &Sketch, id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("the point survives the drag")
    };
    let length_of = |sketch: &Sketch, id: EntityId| {
        sketch
            .segments()
            .iter()
            .find(|segment| segment.id == id)
            .map(|segment| {
                let (tail, head) = (position(sketch, segment.from), position(sketch, segment.to));
                (head[0] - tail[0]).hypot(head[1] - tail[1])
            })
            .expect("the rail survives its own drag")
    };
    let rails: Vec<EntityId> = base
        .sketch
        .segments()
        .iter()
        .filter(|segment| segment.role == EntityRole::Real)
        .map(|segment| segment.id)
        .collect();
    assert_eq!(rails.len(), 2, "a straight slot has two rails");
    for rail in rails {
        let drawn = length_of(&base.sketch, rail);
        assert!((drawn - 24.0).abs() < 1.0e-6, "the rail drew {drawn} long");
        let mut slid = base.clone();
        assert!(slid
            .sketch
            .move_curve(SketchCurve::Segment(rail), [20.0, 40.0], ctx(16))
            .expect("the rail slide is answered"));
        let after = length_of(&slid.sketch, rail);
        assert!(
            (after - drawn).abs() < 1.0e-6,
            "a slid rail came out {after} long, not the {drawn} it was drawn"
        );
    }
}

/// A CIRCLE needs no hold of its own, and this is the shape that would have shown it if it did.
///
/// Two circular caps and two rails tangent to both, each rail end riding the circle it touches —
/// the arc slot's undimensioned width in the one construction that can spend a circle's radius
/// instead. Carried clean through its partner the radius does not move, because a circle's radius
/// is AUTHORED rather than derived from points a hand can drag: an arc's follows its two endpoints
/// through two equal-radius rows and goes wherever they are pushed, while nothing but a relation
/// naming it can move a circle's. The preference pass keeps it and is never skipped for a carry the
/// way the arc's was — take that preference away and this drawing cannot even be BUILT, the first
/// tangency answering `Degenerate`.
#[test]
fn a_slot_capped_with_circles_keeps_its_width_without_a_hold_of_its_own() {
    let mut base = Sketch::empty(PlaneAxis::Z);
    let near = base
        .add_circle(SketchPoint::new(8, 0), SketchLength::new(8))
        .expect("the near cap");
    let far = base
        .add_circle(SketchPoint::new(32, 0), SketchLength::new(8))
        .expect("the far cap");
    let mut rails = Vec::new();
    for (height, side) in [(8, LineSide::Left), (-8, LineSide::Right)] {
        let tail = base.add_free_point(SketchPoint::new(8, height));
        let head = base.add_free_point(SketchPoint::new(32, height));
        let rail = base.connect(tail, head).expect("a rail");
        for (cap, end) in [(near, tail), (far, head)] {
            base.add_constraint(
                ConstraintKind::PointOnCurve {
                    point: end,
                    curve: SketchCurve::Circle(cap),
                },
                ctx(16),
            )
            .expect("the rail end rides its cap");
            base.add_constraint(
                ConstraintKind::tangent(
                    SketchCurve::Circle(cap),
                    SketchCurve::Segment(rail),
                    TangentBranch::Line(side),
                ),
                ctx(16),
            )
            .expect("the rail touches its cap");
        }
        rails.push(rail);
    }
    base.add_constraint(
        ConstraintKind::Parallel {
            first: rails[0],
            second: rails[1],
        },
        ctx(16),
    )
    .expect("the rails run together");

    let center = base
        .circles()
        .iter()
        .find(|circle| circle.id == near)
        .expect("the near cap")
        .center;
    let width_of = |sketch: &Sketch, id: EntityId| {
        sketch
            .circles()
            .iter()
            .find(|circle| circle.id == id)
            .map(|circle| circle.resolved_radius(ctx(16)))
            .expect("the cap survives the drag")
    };
    for target in [-20.0, -120.0, -300.0_f64] {
        let mut carried = base.clone();
        assert!(carried
            .move_point(center, SketchPoint::from_continuous(target, 0.0), ctx(16))
            .expect("the cap carry is answered"));
        for cap in [near, far] {
            let after = width_of(&carried, cap);
            assert!(
                (after - 8.0).abs() < 1.0e-6,
                "carried to {target}, a cap came out {after} wide"
            );
        }
    }
}
