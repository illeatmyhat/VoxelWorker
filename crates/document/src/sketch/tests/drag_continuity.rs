//! A drag must answer the cursor CONTINUOUSLY.
//!
//! Every frame of a real drag rebuilds the preview from the pre-drag drawing and re-solves with
//! the absolute cursor, so a drag is a pure function of cursor position. That buys
//! path-independence — the same cursor always gives the same drawing — but it buys no smoothness:
//! anything that switches on a threshold shows up undamped, as a jump the author feels as a
//! spring. These measure the map rather than a particular answer.

use super::*;
use crate::sketch::tests::constraints::{add_test_segment, position};
use std::f64::consts::TAU;

/// Where the whole drawing ends up for one cursor position, as a flat list of coordinates.
fn answer(base: &Sketch, held: EntityId, cursor: [f64; 2]) -> Vec<f64> {
    let mut sketch = base.clone();
    assert!(
        sketch
            .move_point(
                held,
                SketchPoint::from_continuous(cursor[0], cursor[1]),
                ctx(16),
            )
            .expect("the drag was answered"),
        "the drag moved nothing at {cursor:?}"
    );
    sketch
        .points()
        .iter()
        .flat_map(|point| point.at.in_plane())
        .collect()
}

fn spread(first: &[f64], second: &[f64]) -> f64 {
    first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// A curved slot: rails at 36, 40 and 44 about a hub at the origin, swept a quarter turn.
fn curved_slot() -> Sketch {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
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

fn spine_end(sketch: &Sketch, at: [f64; 2]) -> EntityId {
    sketch
        .points()
        .iter()
        .min_by(|first, second| {
            let reach = |stood: [f64; 2]| (stood[0] - at[0]).hypot(stood[1] - at[1]);
            reach(first.at.in_plane()).total_cmp(&reach(second.at.in_plane()))
        })
        .expect("a point")
        .id
}

/// Rock the cursor by a fiftieth of a unit across the place a snap gives up, and the drawing must
/// not rock with it.
///
/// This is the shape of the author's complaint, verbatim: "small changes in movement of the mouse
/// result in massive swings of movement back and forth like a spring". With a hard cone it swung
/// **3.79** every time, forever — the snapped and unsnapped answers differ by the whole correction
/// exactly where the hand crosses between them. The falloff arrives at the rim already faded to
/// nothing, so there is no longer a crossing to make.
#[test]
fn rocking_the_cursor_where_a_snap_gives_up_does_not_rock_the_drawing() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let worst = (0..8)
        .map(|step| [40.0 + if step % 2 == 0 { 1.08 } else { 1.10 }, 6.0])
        .map(|cursor| answer(&slot, end, cursor))
        .collect::<Vec<_>>()
        .windows(2)
        .map(|pair| match pair {
            [was, now] => spread(was, now),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 0.25,
        "a fiftieth of a unit of cursor swung the drawing {worst}"
    );
}

/// Sliding ALONG the quantity the hand is holding is the one motion that must stay exact, and the
/// plateau is what keeps it so: a falloff without one is not a snap, only a weak pull toward one.
#[test]
fn a_hand_near_its_own_quantity_still_holds_it_exactly() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let mut sketch = slot.clone();
    // Six units of travel, a fifteenth of it across the radius: plainly on the circle.
    assert!(sketch
        .move_point(end, SketchPoint::from_continuous(40.4, 6.0), ctx(16))
        .expect("answered"));
    let at = sketch
        .points()
        .iter()
        .find(|point| point.id == end)
        .expect("the end")
        .at
        .in_plane();
    let radius = at[0].hypot(at[1]);
    assert!(
        (radius - 40.0).abs() < 1.0e-3,
        "the hand was let off its own radius at {radius}"
    );
}

/// The author's ask, in their words: "try making an arc slot endpoint roughly follow its radius".
///
/// An end of a round curve holds its radius EXACTLY across a generous cone, because the drawing
/// authors a radius by another gesture entirely — dragging the arc's body. The only thing left for
/// an end to mean is a sweep, so it may as well mean it without wobbling.
#[test]
fn an_arc_slot_end_follows_its_radius() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let far = spine_end(&slot, [0.0, 44.0]);
    for step in 0..=5 {
        let out = 40.0 + f64::from(step) * 0.5;
        let mut sketch = slot.clone();
        sketch
            .move_point(end, SketchPoint::from_continuous(out, 6.0), ctx(16))
            .expect("answered");
        let at = |id| {
            sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .expect("a point")
                .at
                .in_plane()
        };
        let (here, there) = (at(end), at(far));
        assert!(
            (here[0].hypot(here[1]) - 40.0).abs() < 1.0e-3,
            "pulled {out} out, the end left its radius at {}",
            here[0].hypot(here[1])
        );
        // Not "hardly moved" — did not move. The whole set travels as one similarity, so the far
        // cap has nothing to reconcile and stays exactly where the author left it.
        assert!(
            (there[0].hypot(there[1]) - 44.0).abs() < 1.0e-3 && there[0].abs() < 1.0e-3,
            "pulled {out} out, the far cap came along to {there:?}"
        );
    }
}

/// The drawing's free sweep is no longer spent at random.
///
/// [ADR 0043](../../../../../docs/adr/0043-a-snap-lets-go-gradually.md) measured the far cap of a
/// swept slot sliding along its own arc by up to **2.7** for a cursor step of 0.005, and named it
/// the bigger of the two instabilities — bigger than the snap threshold that decision removed.
/// Holding the radius is what closed it: held exactly, the whole rigid set moves by one similarity
/// and there is nothing left for the solve to reconcile by spending a freedom. Now 2.8e-25.
#[test]
fn the_free_sweep_of_a_slot_is_no_longer_spent_arbitrarily() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let far = spine_end(&slot, [0.0, 44.0]);
    let sideways = |out: f64| {
        let mut sketch = slot.clone();
        sketch
            .move_point(end, SketchPoint::from_continuous(out, 12.0), ctx(16))
            .expect("answered");
        sketch
            .points()
            .iter()
            .find(|point| point.id == far)
            .expect("the far cap")
            .at
            .in_plane()[0]
    };
    let swing = (0..8)
        .map(|step| sideways(41.135 + f64::from(step) * 0.005))
        .collect::<Vec<_>>()
        .windows(2)
        .map(|pair| match pair {
            [was, now] => (now - was).abs(),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    assert!(swing < 0.01, "the free sweep wandered {swing}");
}

/// A drawing with NOTHING asserted about it still snaps.
///
/// The author found this by using it: "the circle ghost and snapping should apply to any arc-like
/// endpoint". A drag with no standing relation short-circuits — nothing to trade the pull against
/// means the hands are the answer — and the snap used to be skipped along with the solve. So the
/// simplest arc in the world, drawn on an empty plane, was the one place an end followed the
/// cursor freely and no ghost ever appeared. Measured before: radius 40.4, 41.4, 42.4, 43.4 as the
/// hand went out, with no quantity reported at all.
#[test]
fn a_bare_arcs_end_holds_its_radius_and_reports_it() {
    let mut arc = Sketch::empty(PlaneAxis::Z);
    let from = arc.add_free_point(SketchPoint::new(40, 0));
    let to = arc.add_free_point(SketchPoint::new(0, 40));
    arc.connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("an arc");
    for step in 0..=3 {
        let out = 40.0 + f64::from(step);
        let mut drawn = arc.clone();
        let answered = drawn
            .move_point_reporting_its_snap(
                from,
                SketchPoint::from_continuous(out, 6.0),
                ctx(16),
                SnapReach::UNBOUNDED,
            )
            .expect("answered");
        assert!(
            answered.kept.is_some(),
            "pulled {out} out, no quantity was reported for the ghost to draw"
        );
        let at = drawn
            .points()
            .iter()
            .find(|point| point.id == from)
            .expect("the end")
            .at
            .in_plane();
        let radius = at[0].hypot(at[1]);
        // Roughly, not exactly: this arc's center is derived from its ends, so it moves with them
        // and the radius is measured against a center that has itself shifted a little.
        assert!(
            (radius - 40.0).abs() < 1.0,
            "pulled {out} out, the end left its radius at {radius}"
        );
    }
}

/// The radius the slot's end lands on when the shell allows the snap `reach` drawing units.
fn radius_under_a_ceiling(slot: &Sketch, end: EntityId, cursor: [f64; 2], reach: f64) -> f64 {
    let mut sketch = slot.clone();
    sketch
        .move_point_reporting_its_snap(
            end,
            SketchPoint::from_continuous(cursor[0], cursor[1]),
            ctx(16),
            SnapReach::of_length(reach),
        )
        .expect("answered");
    let at = sketch
        .points()
        .iter()
        .find(|point| point.id == end)
        .expect("the end")
        .at
        .in_plane();
    at[0].hypot(at[1])
}

/// A snap reaches no further than the shell says it may.
///
/// The author asked for this twice — "screen pixel limits sound good", "I'd like a fairly generous
/// limit" — and it is a ceiling, nothing more. The cone the gesture opens is an ANGLE, and an angle
/// says nothing about how far the drawing may end up from the cursor; the ceiling is the length
/// that angle is allowed to reach, stated by the one layer that knows how big a pixel is.
///
/// Tightening it narrows the snap smoothly all the way to nothing, which is the property that makes
/// it safe to hand a number to: there is no width at which the drawing changes its mind.
#[test]
fn a_snap_reaches_no_further_than_the_shell_allows() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let cursor = [41.5, 6.0];
    let radius = |reach| radius_under_a_ceiling(&slot, end, cursor, reach);
    // Generous enough not to bite: the gesture's own cone is the narrower of the two.
    assert!((radius(f64::INFINITY) - 40.0).abs() < 1.0e-6);
    assert!((radius(4.0) - 40.0).abs() < 1.0e-6);
    // Tight enough to bite, and it gives the radius up gradually rather than dropping it.
    let ladder = [radius(3.0), radius(2.5), radius(2.0)];
    let raw = radius(1.0);
    assert!(
        ladder.windows(2).all(|pair| pair[1] > pair[0] + 0.05),
        "the ceiling let the radius go in a jump rather than a slope: {ladder:?}"
    );
    // Below the hand's own error the snap is not reached at all, so the cursor is the answer.
    assert!(
        (raw - 41.923).abs() < 1.0e-2 && raw - ladder[2] > 0.0,
        "a ceiling under the hand's error still moved it, to {raw}"
    );
}

/// A ceiling must not bring the spring back.
///
/// It is a threshold, and [ADR 0043](../../../../../docs/adr/0043-a-snap-lets-go-gradually.md) is
/// the record of what a threshold did to this drag the last time there was one. It does not,
/// because it narrows the cone rather than switching the snap off: the falloff still arrives at
/// the rim already faded to nothing, only sooner. Measured at a ceiling tight enough to be the
/// thing letting go, rocking the cursor a fiftieth of a unit swings the drawing 0.08 — against
/// 3.79 for the hard cone this file was written to catch.
#[test]
fn a_ceiling_does_not_bring_the_spring_back() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let worst = (0..8)
        .map(|step| [40.0 + if step % 2 == 0 { 1.68 } else { 1.70 }, 6.0])
        .map(|cursor| {
            let mut sketch = slot.clone();
            sketch
                .move_point_reporting_its_snap(
                    end,
                    SketchPoint::from_continuous(cursor[0], cursor[1]),
                    ctx(16),
                    SnapReach::of_length(2.0),
                )
                .expect("answered");
            sketch
                .points()
                .iter()
                .flat_map(|point| point.at.in_plane())
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<_>>()
        .windows(2)
        .map(|pair| match pair {
            [was, now] => spread(was, now),
            _ => 0.0,
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 0.25,
        "a fiftieth of a unit of cursor swung the drawing {worst} under a ceiling"
    );
}

/// The ghost must name the circle the shape is actually on.
///
/// The author found this by looking at it: "the ghost circle doesn't correctly follow the arc. it's
/// the same center point and radius as the arc so I'm confused". It was not. A snap measures its
/// quantity to a PIVOT, and the pivot — an arc's center — is not a hand, so the kernel fell back to
/// where the pivot stands NOW. By then the caller has written the raw cursor into the drawing and
/// [`Sketch::seat_arc_centers`] has re-seated the center on top of it, so the radius was measured
/// against a center the gesture had already dragged. Pulled two and a half units out, the ghost
/// reported 38.29 about `[1.74, -1.39]` while the arc settled at 39.87 about the origin, and the
/// ghost drifted further with every frame.
///
/// The drawing as the gesture FOUND it is the answer, and `was` is where it travels — it was simply
/// narrowed to the hands on the way down.
#[test]
fn the_ghost_names_the_circle_the_arc_is_on() {
    let mut arc = Sketch::empty(PlaneAxis::Z);
    let from = arc.add_free_point(SketchPoint::new(40, 0));
    let to = arc.add_free_point(SketchPoint::new(0, 40));
    let made = arc
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("an arc");
    let center = arc
        .arcs()
        .iter()
        .find(|arc| arc.id == made)
        .expect("the arc")
        .center;
    for step in 0..=5 {
        let out = 40.0 + f64::from(step) * 0.5;
        let mut drawn = arc.clone();
        let kept = drawn
            .move_point_reporting_its_snap(
                from,
                SketchPoint::from_continuous(out, 6.0),
                ctx(16),
                SnapReach::UNBOUNDED,
            )
            .expect("answered")
            .kept
            .expect("a quantity for the ghost to draw");
        let at = |id| {
            drawn
                .points()
                .iter()
                .find(|point| point.id == id)
                .expect("a point")
                .at
                .in_plane()
        };
        let (hub, end) = (at(center), at(from));
        let radius = (end[0] - hub[0]).hypot(end[1] - hub[1]);
        // The ghost is the circle the gesture opened on, so it does not move as the hand does.
        assert!(
            kept.about[0].hypot(kept.about[1]) < 1.0e-6 && (kept.radius - 40.0).abs() < 1.0e-6,
            "pulled {out} out, the ghost wandered to {:?} r {}",
            kept.about,
            kept.radius
        );
        // And the arc is ON it — exactly, once the center is no longer seated on the raw cursor
        // (`an_arc_keeps_its_circle_around_a_whole_turn`). Before either fix it was out by 2.2.
        assert!(
            (hub[0] - kept.about[0]).hypot(hub[1] - kept.about[1]) < 1.0e-6
                && (radius - kept.radius).abs() < 1.0e-6,
            "pulled {out} out, the ghost said {:?} r {} and the arc is at {hub:?} r {radius}",
            kept.about,
            kept.radius
        );
    }
}

/// An arc keeps its circle all the way round, including where its two ends nearly meet.
///
/// The author, after the ghost was fixed: "towards the end of the full 360, it tends to deform and
/// the radius won't stay consistent; the center point ends up moving." The hand was landing on its
/// radius correctly the whole time — it was the arc that ran away from it.
///
/// [`Sketch::seat_arc_centers`] was being run on the drawing bent by the RAW CURSOR, before the
/// snap had had its say. The seat is a projection back onto the chord's bisector, and it is lossy
/// in exactly the case that matters: as the ends close up the chord shortens until the bisector is
/// nearly parallel to the push, so a small cursor error throws the center a long way and no later
/// projection can bring it back. Measured at a chord of 10, three units of cursor threw the center
/// eleven, and the arc came out at radius 37.09 about `[-0.38, 2.91]`.
///
/// Swept a whole turn, three units of cursor error at every step, before and after:
///
/// | chord | was | now |
/// | --- | --- | --- |
/// | 30.6 | 39.687 | 40.0000 |
/// | 20.7 | 39.259 | 40.0000 |
/// | 10.4 | 37.093 | 40.0000 |
/// | 0.0 | 1.500 | 40.0000 |
#[test]
fn an_arc_keeps_its_circle_around_a_whole_turn() {
    let mut arc = Sketch::empty(PlaneAxis::Z);
    let from = arc.add_free_point(SketchPoint::new(40, 0));
    let to = arc.add_free_point(SketchPoint::new(0, 40));
    let made = arc
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("an arc");
    let center = arc
        .arcs()
        .iter()
        .find(|arc| arc.id == made)
        .expect("the arc")
        .center;
    // From one step off the start — a hand that has not swept at all is pulling straight out, which
    // is the author SETTING the radius and must move the center.
    for step in 1..24 {
        let radians = (-f64::from(step) * 15.0).to_radians();
        // Three units proud of the circle: sloppy, the way a real hand is.
        let cursor = [43.0 * radians.cos(), 43.0 * radians.sin()];
        let mut drawn = arc.clone();
        drawn
            .move_point_reporting_its_snap(
                from,
                SketchPoint::from_continuous(cursor[0], cursor[1]),
                ctx(16),
                SnapReach::UNBOUNDED,
            )
            .expect("answered");
        let at = |id| {
            drawn
                .points()
                .iter()
                .find(|point| point.id == id)
                .expect("a point")
                .at
                .in_plane()
        };
        let (hub, end) = (at(center), at(from));
        let radius = (end[0] - hub[0]).hypot(end[1] - hub[1]);
        assert!(
            hub[0].hypot(hub[1]) < 1.0e-6,
            "swept to {}, the center moved to {hub:?}",
            -f64::from(step) * 15.0
        );
        assert!(
            (radius - 40.0).abs() < 1.0e-6,
            "swept to {}, the radius went to {radius}",
            -f64::from(step) * 15.0
        );
    }
}

/// Two arcs held symmetric, carried through the configuration where each one CLOSES on itself.
///
/// An arc has no stored sweep: the endpoint order is the direction, and how far it turns is read
/// back as the counter-clockwise angle from tail to head, which lives in `(0, 2π]`. That reading
/// necessarily JUMPS by a whole turn as the head crosses the tail — a hair short of closing is a
/// hair short of `2π`, and a hair past is a hair past zero. `Relation::Symmetry` on a pair of arcs
/// subtracts one such reading from the other, which is the one place in the solver where the jump
/// could reach a residual and yank the drawing.
///
/// It does not, and the reason is worth stating: **a symmetric pair crosses together.** The
/// endpoints are held reflected, so both arcs close in the same frame, both readings jump in the
/// same frame, and their difference never sees it. The measured crossing step is ordinary — 0.272
/// against a walk whose steps run 0.13 to 0.29 — so the jump is invisible where it would matter.
///
/// This also says what NOT to do about it. Wrapping the difference into `(-π, π]` would make the
/// residual continuous, and would be wrong: it would call a sliver of an arc equal to one that
/// turns nearly the whole way round, which is the difference the row exists to see.
#[test]
fn a_symmetric_arc_pair_crosses_a_whole_turn_without_a_jump() {
    let (sketch, head, hub, tail_at, head_at) = symmetric_arcs_near_closing();
    let radius = (head_at[0] - hub[0]).hypot(head_at[1] - hub[1]);
    let from = (head_at[1] - hub[1]).atan2(head_at[0] - hub[0]);
    // The counter-clockwise way round from the head to the tail, which is the last sliver of turn
    // the arc has left before it closes.
    let turn = ((tail_at[1] - hub[1]).atan2(tail_at[0] - hub[0]) - from).rem_euclid(TAU);

    let steps = 60;
    let mut swings = Vec::new();
    let mut crossed = None;
    let (mut last_flat, mut last_sweep) = (None::<Vec<f64>>, f64::NAN);
    for step in 0..=steps {
        // Ride the arc's own circle, a little PAST the tail, so the walk is carried through the
        // closing configuration rather than around it.
        let bearing = (turn * f64::from(step) / f64::from(steps)).mul_add(1.02, from);
        let (flat, sweep) = sweeping_answer(
            &sketch,
            head,
            [
                radius.mul_add(bearing.cos(), hub[0]),
                radius.mul_add(bearing.sin(), hub[1]),
            ],
        );
        if let Some(last) = last_flat {
            let swing = spread(&last, &flat);
            if last_sweep > 300.0 && sweep < 60.0 {
                crossed = Some(swing);
            }
            swings.push(swing);
        }
        (last_flat, last_sweep) = (Some(flat), sweep);
    }

    let crossing = crossed.expect("the walk carried the arc through a whole turn");
    let elsewhere = swings
        .iter()
        .copied()
        .filter(|swing| (*swing - crossing).abs() > f64::EPSILON)
        .fold(0.0_f64, f64::max);
    assert!(
        crossing <= elsewhere * 2.0,
        "the closing frame cost {crossing}, against {elsewhere} for the widest step elsewhere"
    );
    assert!(
        elsewhere < 0.5,
        "the walk itself should be smooth throughout, and its widest step was {elsewhere}"
    );
}

/// Two arcs, each turning 340 degrees, held symmetric about the plane's vertical axis, with the
/// first one's TAIL and HUB pinned so the only thing a hand on its head can change is how far it
/// turns. Answers the head, the hub, and where the tail and head stand.
fn symmetric_arcs_near_closing() -> (Sketch, EntityId, [f64; 2], [f64; 2], [f64; 2]) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -40], [0, 40]);
    let first_tail = sketch.add_free_point(SketchPoint::new(-14, 0));
    let first_head = sketch.add_free_point(SketchPoint::new(-8, 0));
    let first = sketch
        .connect_arc(first_tail, first_head, AngleMeasurement::from_degrees(340))
        .expect("an arc a hair short of closing");
    let second_tail = sketch.add_free_point(SketchPoint::new(8, 0));
    let second_head = sketch.add_free_point(SketchPoint::new(14, 0));
    let second = sketch
        .connect_arc(
            second_tail,
            second_head,
            AngleMeasurement::from_degrees(340),
        )
        .expect("its partner");
    sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Arc(first),
                SketchCurve::Arc(second),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        )
        .expect("the pair is symmetric");
    for point in [sketch.arcs()[0].from, sketch.arcs()[0].center] {
        let at = position(&sketch, point);
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("the tail and the hub hold still");
    }
    let hub = position(&sketch, sketch.arcs()[0].center);
    let tail_at = position(&sketch, sketch.arcs()[0].from);
    let head_at = position(&sketch, sketch.arcs()[0].to);
    (sketch, first_head, hub, tail_at, head_at)
}

/// Where the drawing ends up for one cursor position, and how far the first arc turns to get
/// there. Unlike [`answer`] it tolerates a frame that moves nothing: a walk that carries an arc
/// through closing passes configurations the solve may legitimately decline, and a declined frame
/// is a frame that did not move, which is the smoothest answer there is.
fn sweeping_answer(base: &Sketch, held: EntityId, cursor: [f64; 2]) -> (Vec<f64>, f64) {
    let mut sketch = base.clone();
    drop(sketch.move_point(
        held,
        SketchPoint::from_continuous(cursor[0], cursor[1]),
        ctx(16),
    ));
    let sweep = sketch
        .arc_form_of(sketch.arcs()[0].id)
        .map_or(f64::NAN, |form| form.sweep_degrees);
    (
        sketch
            .points()
            .iter()
            .flat_map(|point| point.at.in_plane())
            .collect(),
        sweep,
    )
}

/// The two snap cones, in the degrees they actually come to.
///
/// `SNAP_CONE_KEEPING_A_SPAN` and `SNAP_CONE_KEEPING_A_RADIUS` are shares of how far the hand
/// travelled, which says nothing on its own about what a gesture may look like and still be read as
/// moving ALONG the quantity. This measures that, so the two numbers stop being opaque.
///
/// The angles shrink as the gesture lengthens, and that is not a defect: `across` is measured to
/// the LOCUS, and a straight line tangent to a circle of radius R leaves it by about `travel²/2R`
/// while the cone grows only linearly. A hand that follows the locus is held however far it goes;
/// a hand that strikes out straight is let go the further it commits.
///
/// The bounds here are loose on purpose — they are a guard against the shares being changed without
/// anyone noticing what it did, not a claim that these are the right angles.
#[test]
fn the_two_snap_cones_are_the_angles_they_are_measured_to_be() {
    // A span of 40 from a pinned tail, and an arc of radius 40 about a hub. Both are grabbed at
    // [40, 0] and both keep a locus that is the circle of radius 40, so the two shares are being
    // asked exactly the same question.
    let mut span = Sketch::empty(PlaneAxis::Z);
    let tail = span.add_free_point(SketchPoint::new(0, 0));
    let span_end = span.add_free_point(SketchPoint::new(40, 0));
    span.connect(tail, span_end).expect("a segment");
    span.add_constraint(
        ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        },
        ctx(16),
    )
    .expect("the far end holds still");

    let mut arc = Sketch::empty(PlaneAxis::Z);
    let arc_end = arc.add_free_point(SketchPoint::new(40, 0));
    let arc_far = arc.add_free_point(SketchPoint::new(0, 40));
    arc.connect_arc(arc_end, arc_far, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc");

    // (travel, the angle the span must still hold, the angle the radius must still hold)
    for (travel, span_holds, radius_holds) in [(2.0, 6.0, 24.0), (10.0, 1.0, 19.0)] {
        assert!(
            holds_at(&span, span_end, travel, span_holds),
            "a span stopped holding by {span_holds} degrees on a travel of {travel}"
        );
        assert!(
            holds_at(&arc, arc_end, travel, radius_holds),
            "a radius stopped holding by {radius_holds} degrees on a travel of {travel}"
        );
    }
    // The ordering is the decision, not the numbers: a radius is held harder than a span because
    // the drawing authors a radius by another gesture entirely.
    assert!(
        holds_at(&arc, arc_end, 30.0, 8.0) && !holds_at(&span, span_end, 30.0, 1.0),
        "a radius must outlast a span on a long gesture"
    );
}

/// Whether the grabbed end still lands EXACTLY on the circle of radius 40 after a straight gesture
/// of `travel`, struck `degrees` off the tangent at [40, 0].
fn holds_at(base: &Sketch, grabbed: EntityId, travel: f64, degrees: f64) -> bool {
    let off = degrees.to_radians();
    let mut probe = base.clone();
    drop(probe.move_point(
        grabbed,
        SketchPoint::from_continuous(travel.mul_add(off.sin(), 40.0), travel * off.cos()),
        ctx(16),
    ));
    probe
        .points()
        .iter()
        .find(|point| point.id == grabbed)
        .is_some_and(|point| {
            let at = point.at.in_plane();
            (at[0].hypot(at[1]) - 40.0).abs() < 1.0e-4
        })
}
