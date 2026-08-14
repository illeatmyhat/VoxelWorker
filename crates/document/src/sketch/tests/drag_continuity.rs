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

/// A slot drawn with STRAIGHT rails, the shape whose cap centers name no hub.
fn straight_slot() -> Sketch {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::Overall,
            SketchPoint::new(0, 0),
            SketchPoint::new(40, 0),
            SketchPoint::new(0, 8),
            ctx(16),
        )
        .expect("a straight slot")
        .sketch
        .as_ref()
        .clone()
}

/// Pulling a STRAIGHT slot's end cap lengthens it, rather than carrying the whole slot along.
///
/// The author's report, on three slots side by side: "I can grab an endpoint and resize it" for
/// two of them, and "for the one in the middle, I can only translate it". The middle one was drawn
/// [`Overall`](parametric::sketch::LinearSlotKind::Overall), so it carries the extra
/// `PointOnCurve` relations that tie its extremes to its caps — enough extra structure to tip a
/// solve that was ALREADY reading the gesture as a translation into actually performing one.
///
/// The reading is what this guards. A drag is a reshape when it names a pivot, and
/// [`pivot_a_reshape_turns_about`](Sketch::pivot_a_reshape_turns_about) used to demand that the
/// pivot be a hub — a point that several curves turn about. Only arcs turn about anything, so a
/// slot with SEGMENT rails has no hub anywhere on it and could never be reshaped, whatever the
/// author did to it. The far cap is the pivot now, and the whole slot no longer follows the hand.
#[test]
fn a_straight_slot_lengthens_when_its_end_cap_is_pulled() {
    let slot = straight_slot();
    let far = spine_end(&slot, [8.0, 0.0]);
    let near = spine_end(&slot, [32.0, 0.0]);
    let anchored = slot
        .points()
        .iter()
        .find(|point| point.id == far)
        .map(|point| point.at.in_plane())
        .expect("the far cap");

    let mut moved = slot.clone();
    moved
        .move_point_reporting_its_snap(
            near,
            SketchPoint::from_continuous(60.0, 0.0),
            ctx(16),
            SnapReach::UNBOUNDED,
            &mut [],
        )
        .expect("the drag lands");

    let at = |id: EntityId| {
        moved
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("a point")
    };
    let pulled = at(near);
    assert!(
        (pulled[0] - 60.0).abs() < 1.0,
        "the pulled cap did not reach the cursor: {pulled:?}"
    );
    let drifted = (at(far)[0] - anchored[0]).hypot(at(far)[1] - anchored[1]);
    assert!(
        drifted < 1.0,
        "the far cap travelled {drifted} with the hand, so the slot moved instead of lengthening"
    );
}

/// A scaffold's SPAN is not a quantity a drag can be asked to keep.
///
/// A slot's spine is construction geometry, and for a straight slot its span IS the slot's length.
/// Offering it to the snap put a ghost circle on a cap center and held the length while the author
/// was trying to change it — reported as "the arc circle ghost also happens even though this
/// endpoint is the center-point of the arc, which should never happen".
///
/// A scaffold ARC keeps its radius, which is why the filter is on spans alone: travelling around a
/// center never changes a radius, so a curved slot's curvature survives a gesture that lengthens
/// it. See `Problem::add_scaffolding_segment` in the parametric crate.
#[test]
fn a_scaffold_span_offers_no_quantity_to_hold() {
    let slot = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(8, 0),
            SketchPoint::new(32, 0),
            SketchPoint::new(8, 8),
            ctx(16),
        )
        .expect("a straight slot")
        .sketch
        .as_ref()
        .clone();
    let cap = spine_end(&slot, [32.0, 0.0]);
    let mut moved = slot;
    let answer = moved
        .move_point_reporting_its_snap(
            cap,
            SketchPoint::from_continuous(32.0, 6.0),
            ctx(16),
            SnapReach::UNBOUNDED,
            &mut [],
        )
        .expect("the drag lands");
    assert!(
        answer.kept.is_none(),
        "the spine offered its span as a quantity to hold: {:?}",
        answer.kept
    );
}

/// A shape's handle is DECLARED by the tool that draws it, never inferred from the drawing.
///
/// The inference this replaced read a hub off the curves turning about a point, so it could only
/// ever fire on a shape built from arcs. Nobody in the field infers it: FreeCAD's sketcher stores a
/// handle geoId on a `Group` constraint and swaps any grabbed member for it, and D-Cubed calls the
/// same thing a declared rigid set. A turning slot's rails and centerline share one middle and the
/// tool declares it; a straight slot has no such place, declares nothing, and is translated by its
/// body like any other drawing that names no hub.
#[test]
fn only_the_tool_that_drew_a_shape_says_what_its_handle_is() {
    let curved = curved_slot();
    let hub = curved
        .points()
        .iter()
        .find(|point| point.handle == PointHandle::ShapeHub)
        .map(|point| point.id)
        .expect("a turning slot declares its hub");
    assert!(
        curved.pivot_a_reshape_turns_about(hub).is_none(),
        "the declared hub named a pivot, so dragging it would reshape instead of move"
    );
    let cap = spine_end(&curved, [40.0, 0.0]);
    assert!(
        curved.pivot_a_reshape_turns_about(cap).is_some(),
        "a cap center named no pivot, so the slot would follow the hand whole"
    );

    let straight = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
        .with_linear_slot(
            ::parametric::sketch::LinearSlotKind::CenterToCenter,
            SketchPoint::new(8, 0),
            SketchPoint::new(32, 0),
            SketchPoint::new(8, 8),
            ctx(16),
        )
        .expect("a straight slot")
        .sketch
        .as_ref()
        .clone();
    assert!(
        !straight
            .points()
            .iter()
            .any(|point| point.handle == PointHandle::ShapeHub),
        "a straight slot has no one place its parts turn about, so it may declare no hub"
    );
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

/// **An arc slot wound past its own far end is still one slot.**
///
/// A slot's two rails and its centerline are the same shape three times at three radii about one
/// hub, so they draw the same way round or they are not a slot. The hand names the CENTERLINE — a
/// spine end IS a cap center — while the body an author looks at is the rails, whose ends sit half
/// a width away and are named by nothing the gesture holds.
///
/// The owner's report, against a bare arc that had just been fixed: "it works for a bare 3 point
/// arc but not for an arc slot." Measured at the seam where the two cap centers meet, shrinking:
/// the spine turned onto the other side and read 15 degrees while both rails read 345, having gone
/// the long way. A whole turn later the same thing mirrored — spine 345 against rails 15. Radii
/// held at 44, 36 and 40 the whole way, so nothing was wrong with the shape; the three arcs simply
/// stopped agreeing about which piece of their circles the slot was.
///
/// Wound both ways and far enough each way to cross TWO seams, because the seams alternate.
#[test]
fn an_arc_slots_rails_turn_with_the_spine_they_are_drawn_from() {
    for way in [-1.0_f64, 1.0] {
        let mut sketch = curved_slot();
        let hub = spine_end(&sketch, [0.0, 0.0]);
        let held = spine_end(&sketch, [0.0, 40.0]);
        let mut turns = crate::sketch::ArcTurnUnderAGesture::opening_over(&sketch);
        let step = 15.0_f64;
        let mut drawn: Option<f64> = None;
        let mut stands = 0;
        for taken in 1..=42 {
            let asked = 90.0 + way * step * f64::from(taken);
            let hand = [
                40.0 * asked.to_radians().cos(),
                40.0 * asked.to_radians().sin(),
            ];
            let answered = sketch
                .move_point_reporting_its_snap(
                    held,
                    SketchPoint::from_continuous(hand[0], hand[1]),
                    ctx(16),
                    SnapReach::UNBOUNDED,
                    &mut turns,
                )
                .expect("evaluation context")
                .moved;
            if !answered {
                // A seam: the two cap centers stacked, no piece of the circle to prefer.
                stands += 1;
                assert!(
                    stands <= 2,
                    "more than one frame stood at each of two seams"
                );
                continue;
            }
            let turning: Vec<(EntityId, f64, f64)> = sketch
                .arcs()
                .iter()
                .filter(|arc| arc.center == hub)
                .filter_map(|arc| {
                    sketch
                        .arc_form_of(arc.id)
                        .map(|form| (arc.id, form.radius, form.sweep_degrees))
                })
                .collect();
            assert_eq!(
                turning.len(),
                3,
                "two rails and a centerline turn about the slot's hub"
            );
            let slot = turning[0].2;
            for (id, radius, sweep) in &turning {
                assert!(
                    (sweep - slot).abs() < 1.0e-6,
                    "at {asked} degrees, arc {id:?} of radius {radius} draws {sweep} where the slot draws {slot}"
                );
            }
            if let Some(was) = drawn {
                assert!(
                    (slot - was).abs() < step + 2.0,
                    "at {asked} degrees, the slot jumped from {was} to {slot} in one {step}-degree step"
                );
            }
            drawn = Some(slot);
        }
        assert!(
            stands > 0,
            "the walk never reached a seam, so it proves nothing"
        );
    }
}

/// **A contact is judged against the piece the frame is about to draw, not the one it inherited.**
///
/// Which way round an arc is drawn is decided from the settled solution, because an arc a gesture
/// CARRIES rather than holds cannot be measured until it has moved. That put the decision after the
/// solve — and one validator runs earlier. A tangent CONTACT stands on the arc's DRAWN piece or it
/// does not, and the two readings of an arc whose ends have just crossed are different pieces.
///
/// A segment tangent to this arc's INTERIOR, at bearing 45, and the far end wound clockwise so the
/// sweep grows. Measured with the decision left until after validation: the wind ran clean to 345
/// degrees, stood on the seam exactly as designed, and then died — `InvalidTangent`
/// `OutsideFirstDomain` at the next frame, which in the shell ends the gesture outright. The same
/// move offered to the same drawing with the arc reversed first answered `Ok(true)`. So the
/// refusal was a property of the LABEL: the stale reading is a 15-degree sliver spanning 75 to 90
/// with the contact outside it, and the reading the frame was about to be given is 345 degrees
/// spanning 90 to 435, which contains 45 + 360.
///
/// A contact at an arc's END could never have shown this — both readings share their endpoints,
/// which is why an arc slot's four tangencies cross both seams without noticing.
///
/// The last two frames are the honest boundary: wound far enough, the contact really does leave
/// the drawn piece, and then the drawing refuses and is right to.
#[test]
fn a_tangent_contact_is_judged_against_the_piece_the_frame_will_draw() {
    use crate::sketch::{LineSide, SketchCurve, TangentBranch};
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let held = sketch.add_free_point(SketchPoint::new(40, 0));
    let far = sketch.add_free_point(SketchPoint::new(0, 40));
    let arc = sketch
        .connect_arc(
            held,
            far,
            ::parametric::units::AngleMeasurement::from_degrees(90),
        )
        .expect("a quarter arc");
    let touch = 40.0 / 2.0_f64.sqrt();
    let along = 20.0 / 2.0_f64.sqrt();
    let tail = sketch.add_free_point(SketchPoint::from_continuous(touch - along, touch + along));
    let head = sketch.add_free_point(SketchPoint::from_continuous(touch + along, touch - along));
    let segment = sketch.connect(tail, head).expect("a line");
    // `Left` names the other side of the segment's authored run and nothing touches the arc there.
    sketch
        .add_constraint(
            crate::sketch::ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Arc(arc),
                TangentBranch::Line(LineSide::Right),
            ),
            ctx(16),
        )
        .expect("a line already touching the arc's middle");

    let mut turns = crate::sketch::ArcTurnUnderAGesture::opening_over(&sketch);
    let step = 15.0_f64;
    let mut drawn: Option<f64> = None;
    let mut stands = 0;
    for taken in 1..=21 {
        let asked = -step * f64::from(taken);
        let hand = [
            40.0 * asked.to_radians().cos(),
            40.0 * asked.to_radians().sin(),
        ];
        let answered = sketch
            .move_point_reporting_its_snap(
                held,
                SketchPoint::from_continuous(hand[0], hand[1]),
                ctx(16),
                SnapReach::UNBOUNDED,
                &mut turns,
            )
            .unwrap_or_else(|refused| {
                panic!("at {asked} degrees the drawing refused a frame it can hold: {refused:?}")
            });
        if !answered.moved {
            stands += 1;
            assert!(stands <= 1, "one seam, one stood frame");
            continue;
        }
        let form = sketch.arc_form_of(arc).expect("an arc");
        assert!(
            (form.radius - 40.0).abs() < 1.0e-3,
            "at {asked} degrees the arc left radius 40 for {}",
            form.radius
        );
        if let Some(was) = drawn {
            assert!(
                (form.sweep_degrees - was).abs() < step + 2.0,
                "at {asked} degrees the arc jumped from {was} to {}",
                form.sweep_degrees
            );
        }
        drawn = Some(form.sweep_degrees);
    }
    assert_eq!(stands, 1, "the walk never reached the seam");

    // Wound one step further, the contact is genuinely off the drawn piece, and refusing is the
    // right answer rather than a stale one.
    let asked = -330.0_f64;
    let refused = sketch.move_point_reporting_its_snap(
        held,
        SketchPoint::from_continuous(
            40.0 * asked.to_radians().cos(),
            40.0 * asked.to_radians().sin(),
        ),
        ctx(16),
        SnapReach::UNBOUNDED,
        &mut turns,
    );
    assert!(
        matches!(
            refused,
            Err(crate::sketch::SketchEvaluationError::InvalidTangent { .. })
        ),
        "the contact has been wound past the arc's own end, and got {refused:?}"
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

/// A long sweep does not fatten the slot it is sweeping.
///
/// The shell recomputes the whole gesture from the press on every frame, so a sweep arrives as one
/// wide turn rather than as the frames the author drew it in, and the walk that carries it can only
/// afford so many steps. A step wide enough to see the curvature of the circle the snap turns about
/// lands beside the answer, and because nothing in the drawing prices its own width, that is where
/// the error went: this eight-unit slot came back 9.8 wide at twenty degrees and 23.6 at a hundred
/// and sixty, and FINER steps only slowed the drift. What fixes it is saying the quantity out loud
/// — an arc the gesture drags but does not author keeps the radius the opening measured, so there
/// is no free width left to spend.
///
/// Asked at nine angles rather than at the end, because it is the drawing under the cursor
/// throughout the gesture that the author is watching, not the one they let go of.
#[test]
fn sweeping_a_slots_end_keeps_the_width_it_was_drawn_with() {
    let slot = curved_slot();
    let end = spine_end(&slot, [36.0, 0.0]);
    let reaches = |sketch: &Sketch| {
        sketch
            .points()
            .iter()
            .map(|point| {
                let at = point.at.in_plane();
                at[0].hypot(at[1])
            })
            .collect::<Vec<f64>>()
    };
    // Swept AWAY from the far cap. Turned the other way the near end winds through it, and the
    // drawing refuses the contact rather than answering — which the suite already states above.
    //
    // Replayed from the drawing at rest every time, never nudged from the last answer, because
    // that is what the shell does with a held cursor and it is the only reading a long sweep has.
    // Stopping at 160: a half turn winds this slot's own contact off the piece it was drawn on,
    // and the drawing is right to refuse that rather than answer it.
    for degrees in (20..=160).step_by(20) {
        let asked = -f64::from(degrees);
        let mut sketch = slot.clone();
        let mut turns = crate::sketch::ArcTurnUnderAGesture::opening_over(&sketch);
        sketch
            .move_point_reporting_its_snap(
                end,
                SketchPoint::from_continuous(
                    36.0 * asked.to_radians().cos(),
                    36.0 * asked.to_radians().sin(),
                ),
                ctx(16),
                SnapReach::UNBOUNDED,
                &mut turns,
            )
            .unwrap_or_else(|refused| {
                panic!("swept {degrees} degrees and the drawing refused it: {refused:?}")
            });
        let radii = reaches(&sketch);
        let widest = radii.iter().fold(0.0_f64, |most, reach| most.max(*reach));
        // The hub stands at the origin and is the one reach that is not a rail.
        let narrowest = radii
            .iter()
            .filter(|reach| **reach > 1.0)
            .fold(f64::INFINITY, |least, reach| least.min(*reach));
        assert!(
            (widest - narrowest - 8.0).abs() < 0.05,
            "swept {degrees} degrees, the slot came back {} wide",
            widest - narrowest
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
                &mut [],
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
            &mut [],
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
                    &mut [],
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
                &mut [],
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
                &mut [],
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

/// Ninety screen points means the SAME thing at every zoom.
///
/// [ADR 0045](../../../../../docs/adr/0045-a-snap-reaches-only-as-far-as-the-shell-allows.md) states
/// the ceiling in screen points and has the shell convert. The claim that buys is scale
/// equivariance: scale the drawing by k, scale the gesture by k, scale the ceiling by k, and the
/// answer should be the k-scaled answer — otherwise "ninety points" would mean a generous limit
/// zoomed out and a biting one zoomed in, and the author would have to relearn it at every zoom.
///
/// Measured across a fourfold scale, at five ceilings spanning the whole slope from "does nothing"
/// to "gives the radius up entirely". Agreement is a few parts in a million, which is the solve's
/// own convergence rather than anything about scale — the loosest ceiling reaches three parts and
/// the tightest agrees to one part in two thousand million.
///
/// The bound is a RATIO because the claim is one: an absolute epsilon on a length is itself a
/// statement about scale, and the answers here are lengths near forty. What the residue is not is
/// a tolerance anybody could tighten — dropping `SATISFACTION_TOLERANCE` by four orders of
/// magnitude leaves every figure below bit-identical, so it is the step budget landing at a
/// slightly different place along the same path, not an unconverged solve.
///
/// The one place it is only approximate is the SHELL's conversion, not the kernel's arithmetic:
/// under perspective on a tilted plane, drawing-units-per-pixel varies across the screen, and the
/// shell measures it once at the cursor.
#[test]
fn a_ceiling_in_screen_points_means_the_same_at_every_zoom() {
    for reach in [1.5, 2.0, 2.5, 3.0, 4.0] {
        let answers: Vec<f64> = [1_i64, 2, 4]
            .into_iter()
            .map(|scale| {
                let (slot, span) = (
                    scaled_slot(scale),
                    f64::from(i32::try_from(scale).unwrap_or(1)),
                );
                let end = spine_end(&slot, [40.0 * span, 0.0]);
                radius_under_a_ceiling(&slot, end, [41.5 * span, 6.0 * span], reach * span) / span
            })
            .collect();
        let widest = answers
            .iter()
            .flat_map(|first| answers.iter().map(move |second| (first - second).abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            widest < 1.0e-5 * answers[0],
            "a ceiling of {reach} answered {answers:?} across a fourfold zoom"
        );
    }
}

/// The curved slot of [`curved_slot`], every dimension multiplied by `scale`.
fn scaled_slot(scale: i64) -> Sketch {
    SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
        .with_center_arc_slot(
            SketchPoint::new(0, 0),
            SketchPoint::new(40 * scale, 0),
            SketchPoint::new(0, 40 * scale),
            ::parametric::sketch::ArcTurn::CounterClockwise,
            SketchPoint::new(44 * scale, 0),
            ctx(16),
        )
        .expect("a curved slot")
        .sketch
        .as_ref()
        .clone()
}

/// The snap ring is inked from how much CONE IS LEFT, and that is not how hard the snap holds.
///
/// The two were interchangeable enough that the ring shipped drawing the hold, and the author's
/// read on it was that the fade "is rather inconsistent and dies quickly". Both halves are what the
/// hold does, measured:
///
/// - The hold is exactly one over the plateau (`Problem::SNAP_HOLD`) and spends
///   its entire range in the outer `0.4` of the cone, so the ring sat at full ink until the hand
///   was already 60% of the way out and then collapsed over `0.3 * travel` of cursor. Inking from
///   the room left spends the WHOLE cone: two and a half times the fade for the same gesture, and
///   the ring dims while there is still room to act on.
/// - The hold's steepest slope is `1.5 / 0.4 = 3.75` per cone; this one's is `1`. At the start of a
///   gesture the cone is a few screen points wide and that factor is the ring strobing against the
///   ring dimming.
///
/// The third measured fact is why neither number can be fixed by tuning the falloff: for a hand
/// travelling in a straight line, `across` grows like `travel * sin(heading)` while the cone grows
/// like `share * travel`, so **their ratio is constant in travel**. A straight gesture picks its
/// place in the cone at the outset and holds it. The fade is a function of the author's wrist
/// angle, not of how far they have gone — which is why widening the plateau would not have helped
/// and why the ink has to spend as much of the cone as it can get.
#[test]
fn the_snap_ring_is_inked_from_the_room_left_in_the_cone() {
    let mut arc = Sketch::empty(PlaneAxis::Z);
    let from = arc.add_free_point(SketchPoint::new(40, 0));
    let to = arc.add_free_point(SketchPoint::new(0, 40));
    let _ = arc
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("an arc to slide along");
    // Radially out from one point of the circle, so `across` is the step itself and the ring's
    // whole fade is walked. Fifteen degrees round keeps the gesture off the tangent, where a hand
    // pulling away from a quantity actually is.
    let along = 15.0_f64.to_radians();
    let mut walked: Vec<(f64, f64)> = Vec::new();
    for step in 0..=60 {
        let across = f64::from(step) * 0.2;
        let cursor = [(40.0 + across) * along.cos(), (40.0 + across) * along.sin()];
        let travel = (cursor[0] - 40.0).hypot(cursor[1]);
        let cone = 0.75 * travel;
        let mut drawn = arc.clone();
        let kept = drawn
            .move_point_reporting_its_snap(
                from,
                SketchPoint::from_continuous(cursor[0], cursor[1]),
                ctx(16),
                SnapReach::UNBOUNDED,
                &mut [],
            )
            .expect("answered")
            .kept;
        let Some(kept) = kept else {
            assert!(
                across >= cone,
                "the snap was dropped {across} out of a cone of {cone}"
            );
            continue;
        };
        // The circle the gesture opened on, so the closed form below is measured against the
        // radius candidate and not some other quantity that happened to win.
        assert!(
            kept.about[0].hypot(kept.about[1]) < 1.0e-9 && (kept.radius - 40.0).abs() < 1.0e-9,
            "{across} out, the ring is {:?} r {}",
            kept.about,
            kept.radius
        );
        let ink = kept.ghost_ink();
        // To a millionth rather than exactly: the cursor is stored as a fixed-point coordinate, so
        // the cone recomputed here from the ideal position is a few billionths off the one the
        // solve measured.
        assert!(
            (ink - (1.0 - across / cone)).abs() < 1.0e-6,
            "{across} out of a cone of {cone}, the ring inked {ink}"
        );
        walked.push((across, ink));
    }
    // It dims from the FIRST step off the quantity. The hold is flat at one until 60% out.
    let early = walked
        .iter()
        .find(|(across, _)| *across > 1.5)
        .expect("a step off the quantity");
    assert!(
        early.1 < 0.95,
        "{} out the ring is still at {} ink",
        early.0,
        early.1
    );
    // And it never moves faster than one per cone. The hold peaks at 3.75 per cone.
    let cone_at_the_start = 0.75 * (40.0 * (2.0 * (along / 2.0).sin())).abs();
    for pair in walked.windows(2) {
        let [(was, before), (now, after)] = [pair[0], pair[1]];
        let slope = (before - after) / (now - was);
        assert!(
            slope <= 1.02 / cone_at_the_start,
            "the ring swung {slope} per unit between {was} and {now}, past 1 per cone"
        );
    }
}

/// **Sliding a held point along a spline whose pieces are twenty to one is smooth across a knot.**
///
/// The ADR 0043 gauge, pointed at the station column. A point held to a spline is held by a
/// solver coordinate saying WHERE along it the point stands, and that coordinate is held in one
/// constant unit — the seed curve's total chord over its piece count — rather than being
/// renormalized per piece. A constant keeps the map C1 at every knot, which is the property the
/// finite-difference Jacobian rests on; the price is that on a curve whose pieces differ wildly
/// the unit is right for the average piece and wrong for both extremes.
///
/// So this is the witness that makes the price visible: chords of about 1, 20 and 1, and the held
/// point walked the length of the curve so it crosses both knots. The gauge is the one 0043 uses —
/// whole-drawing displacement per unit of cursor travel — because a conditioning failure in a
/// coordinate the drawing pins down least shows up as a single frame that swings hundreds of
/// times the cursor step, not as a drift.
///
/// **Measured: a worst gain of `1.043` over all 841 frames.** The drawing moves what the cursor
/// moves and very little more, at both knots and across the twenty-to-one change of scale between
/// them. The price the constant unit was supposed to charge does not show up at this ratio, which
/// is what says the deviation from a per-piece normalization cost nothing and bought the C1 join.
/// The ceiling is the 0043 one rather than anything tighter, so the test states the property
/// rather than pinning the number.
///
/// Every point that SHAPES the curve is pinned — the fit points and their forward arms — so the
/// only thing left free to answer is the held point and its station. Pinning the fit points alone
/// is not enough and the first version of this gauge did exactly that: a curve pinned only where
/// it passes through is still free to bend between, and the walk then measured the arms swinging
/// to drag the curve up to a cursor a whole unit above it, which is a different number about a
/// different thing. Read on the unpinned drawing that gain reaches thirty thousand; here it is
/// one, and the difference between the two is the whole reason the shaping points are pinned.
///
/// The per-frame check that the hold is still holding is not decoration. This gauge ran green at
/// `1.0000014` for as long as the relation was being dropped from the scoped problem before the
/// walk ever reached it — a free point tracking a cursor moves exactly what the cursor moves, and
/// answers every frame while doing it, so neither the gain nor the answered-frame count could
/// tell the difference. See
/// `a_line_held_to_a_spline_stays_on_it_when_the_spline_moves_and_when_it_is_reshaped`.
///
/// **Seen red**, which is the only thing that makes any of the above evidence. Emptying
/// [`points_of`](super::Sketch::points_of) for a spline, and separately dropping splines from the
/// stores [`curves_standing_on_any`](super::Sketch::curves_standing_on_any) enumerates, each fail
/// it on frame 0 with `the hold let go`. A least-norm solve reports a MISSING constraint as the
/// smoothest drawing there is, so a gauge that only asks for a small number reads healthiest
/// exactly when it is measuring nothing.
#[test]
fn sliding_a_held_point_along_an_uneven_spline_is_smooth_across_its_knots() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    // Chords of roughly 1.1, 20 and 1.1 — the twenty-to-one witness.
    let places = [[0.0, 0.0], [1.0, 0.5], [21.0, 0.5], [22.0, 0.0]];
    let spline = sketch
        .add_fit_point_spline(
            &places.map(|at| SketchPoint::from_continuous(at[0], at[1])),
            false,
        )
        .expect("a spline");
    let drawn: Vec<EntityId> = sketch
        .splines
        .iter()
        .find(|held| held.id == spline)
        .expect("the spline")
        .points
        .clone();
    let arms: Vec<EntityId> = sketch
        .splines
        .iter()
        .find(|held| held.id == spline)
        .expect("the spline")
        .tangents
        .values()
        .map(|handle| handle.forward)
        .collect();
    // The arms as well as the fit points. A pinned fit point still leaves the curve free to bend
    // through it, and a gauge that let the curve bend would be measuring the SHAPE giving way
    // rather than the station sliding — which is the one thing it is here to isolate. The forward
    // arm only: the back one is its mirror and refuses to be pinned in its own right.
    for point in drawn.iter().chain(&arms) {
        let at = sketch.point_in_plane(*point).expect("a placed point");
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point: *point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("a shaping point can be pinned");
    }
    let standing = sketch.add_free_point(SketchPoint::from_continuous(1.0, 1.5));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: standing,
                onto: CoincidentTarget::Curve(SketchCurve::Spline(spline)),
            },
            ctx(16),
        )
        .expect("a free point can stand on a spline");

    // Walk the cursor the length of the curve, a fortieth of a unit at a time, held above it so
    // the answer is the station sliding rather than the point being pushed off.
    let step = 0.025;
    let mut last: Option<Vec<f64>> = None;
    let mut worst = 0.0_f64;
    let mut measured = 0_u32;
    for tick in 0..=840 {
        let cursor = [0.5 + f64::from(tick) * step, 1.5];
        let mut frame = sketch.clone();
        let Ok(true) = frame.move_point(
            standing,
            SketchPoint::from_continuous(cursor[0], cursor[1]),
            ctx(16),
        ) else {
            continue;
        };
        measured += 1;
        let now: Vec<f64> = frame
            .points()
            .iter()
            .flat_map(|point| point.at.in_plane())
            .collect();
        let landed = frame.point_in_plane(standing).expect("the held point");
        // The cursor rides a unit above a curve that never leaves y = 0.5, so a held point that
        // came back at the cursor's own height is a point nothing is holding. Checked per frame
        // rather than once, because a hold that lets go halfway is the failure worth catching.
        assert!(
            landed[1] < 1.0,
            "frame {tick} answered {landed:?} for a cursor at {cursor:?}, so the hold let go"
        );
        if let Some(was) = last.as_ref() {
            worst = worst.max(spread(was, &now) / step);
        }
        last = Some(now);
    }
    // A walk that answered nothing would report a gain of zero and pass without measuring, so
    // the count of answered frames is asserted beside the gain. The per-frame check above covers
    // the other way to measure nothing — answering every frame with the constraint absent.
    assert!(
        measured > 800,
        "only {measured} of 841 frames answered the cursor"
    );
    assert!(
        worst < 3.0,
        "a cursor step swung the drawing {worst} times over, so the station column is the \
         direction the drawing pins down least"
    );
}

/// The free sweep is STILL spent arbitrarily when nothing snaps, and this measures how badly.
///
/// [ADR 0043](../../../../../docs/adr/0043-a-snap-lets-go-gradually.md) named the free sweep as the
/// bigger of the two instabilities and closed it only where a snap holds the radius —
/// `the_free_sweep_of_a_slot_is_no_longer_spent_arbitrarily`. Outside a cone nothing holds the set
/// together, and walking a slot's end out from its own corner at 0.005 a step:
///
/// | heading | worst gain |
/// | --- | --- |
/// | 0° | 2.46 |
/// | 60° | **191** |
/// | 90° | 1.70 |
/// | 120° | 1.76 |
/// | 180° | **1318** |
///
/// It is not a drift. The answer tracks smoothly — one coordinate creeping `-0.414` to `-0.420`
/// over the whole walk — and then a single cursor value lands at `-1.418`, or `+1.697`, or
/// `-2.532`, and the next value is back on the line. The coordinate that jumps is the far cap
/// sliding along its own arc. Both answers satisfy every authored constraint to `1.2e-9`; the tie
/// is broken by which stationary point the trust region walked into, not by anything the drawing
/// says.
///
/// **This asserts the defect, so the day it is fixed this test fails and gets renamed.** See
/// [ADR 0043](../../../../../docs/adr/0043-a-snap-lets-go-gradually.md) for the four approaches
/// that were measured and rejected, so they are not tried a second time.
/// An unsnapped walk is smooth in EVERY direction, not just the ones that happened to be smooth.
///
/// The curved slot's end is walked radially outward at five headings, a two-hundredth of a unit at
/// a time, and the whole drawing's displacement per step is measured. Two of those directions used
/// to swing by hundreds of times the cursor step, and not as an isolated frame: at sixty degrees,
/// **118 of the 200 steps** moved the drawing by more than ten times the cursor.
///
/// | heading | before | after |
/// | ------- | -----: | ----: |
/// | 0°      |   2.46 |  2.46 |
/// | 60°     | **191.13** |  1.52 |
/// | 90°     |   1.70 |  1.70 |
/// | 120°    |   1.76 |  1.76 |
/// | 180°    | **1317.86** |  2.46 |
///
/// What fixed it was not a preference, a damper or a snap. The solve was forming `JᵀJ`, which
/// squares the condition number past what a `f64` holds, and taking every step out of the damping
/// repair that failure falls through to — so the sweep, being the direction the constraints pinned
/// down least, was where the perturbation landed. See `substrate`'s `gauss_newton_step`. The
/// directions that were already smooth are unchanged to three decimals, which is what says the
/// change removed noise rather than adding a bias.
#[test]
fn an_unsnapped_walk_is_smooth_in_every_direction() {
    let slot = curved_slot();
    let end = spine_end(&slot, [40.0, 0.0]);
    let gain_at = |heading: f64| {
        let ray: f64 = heading.to_radians();
        let mut last: Option<Vec<f64>> = None;
        let mut worst = 0.0_f64;
        for step in 0..=200 {
            let out = 6.0 + f64::from(step) * 0.005;
            let cursor = [40.0 + out * ray.cos(), out * ray.sin()];
            let mut sketch = slot.clone();
            let Ok(_) = sketch.move_point_reporting_its_snap(
                end,
                SketchPoint::from_continuous(cursor[0], cursor[1]),
                ctx(16),
                SnapReach::UNBOUNDED,
                &mut [],
            ) else {
                continue;
            };
            let now: Vec<f64> = sketch
                .points()
                .iter()
                .flat_map(|point| point.at.in_plane())
                .collect();
            if let Some(was) = last.as_ref() {
                worst = worst.max(spread(was, &now) / 0.005);
            }
            last = Some(now);
        }
        worst
    };
    for heading in [0.0, 60.0, 90.0, 120.0, 180.0] {
        let gain = gain_at(heading);
        assert!(
            gain < 3.0,
            "{heading} deg swings {gain} times the cursor step"
        );
    }
}
