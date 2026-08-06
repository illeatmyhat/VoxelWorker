//! A drag must answer the cursor CONTINUOUSLY.
//!
//! Every frame of a real drag rebuilds the preview from the pre-drag drawing and re-solves with
//! the absolute cursor, so a drag is a pure function of cursor position. That buys
//! path-independence — the same cursor always gives the same drawing — but it buys no smoothness:
//! anything that switches on a threshold shows up undamped, as a jump the author feels as a
//! spring. These measure the map rather than a particular answer.

use super::*;

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
