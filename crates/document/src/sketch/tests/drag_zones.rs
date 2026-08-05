//! Where one drag's SETTLE actually spends its time, zone by zone.
//!
//! The settle is scoped: only the part of the drawing the hands can reach enters the kernel, and
//! that made the whole-drawing solve go away. What is left still grows with the drawing, which
//! means the growth is in a caller rather than in the kernel — every zone below either walks the
//! whole entity store or scans a list per lookup, and only a measurement says which one matters.
//!
//! Reported per POPULATION, because the number to read is the slope, not any one row.
//!
//! Run: `cargo test -p document --release --lib drag_zones -- --ignored --nocapture`

use std::num::NonZeroU32;
use std::time::Instant;

use super::super::{PlaneAxis, Sketch, SketchPoint, SketchSolid};
use crate::sketch::constraint;

const DENSITY: u32 = 16;
const DEPTH_VOXELS: u32 = 32;

fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(NonZeroU32::new(DENSITY).expect("a non-zero density"))
}

/// `count` arc slots side by side — disjoint shapes, so the reach of any one drag is one slot
/// however many are drawn. Anything that grows with `count` is reading the rest of the drawing.
fn arc_slots(count: usize) -> Sketch {
    let mut made = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), DEPTH_VOXELS);
    for index in 0..count {
        let center = (index as i64) * 40;
        made = made
            .with_center_arc_slot(
                SketchPoint::new(center, 0),
                SketchPoint::new(center + 8, 0),
                SketchPoint::new(center, 8),
                parametric::sketch::ArcTurn::CounterClockwise,
                SketchPoint::new(center + 10, 0),
                context(),
            )
            .expect("a quarter-turn arc slot");
    }
    *made.sketch
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn drag_settle_zones_by_population() {
    println!(
        "\n{:<10} {:>7} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "drawing", "points", "reach", "hands", "clone", "reach", "prepare", "solve", "total"
    );
    println!("{}", "-".repeat(80));
    for count in [1_usize, 8, 16, 32, 64] {
        let sketch = arc_slots(count);
        let points = sketch.points();
        let grabbed = points[points.len() / 2].id;
        let was = points[points.len() / 2].at;
        let to = SketchPoint::from_continuous(was.in_plane()[0] + 1.0, was.in_plane()[1] + 1.0);

        // 1. Which points the gesture carries — a walk of the relations per hand.
        let started = Instant::now();
        let hands = sketch.hands_moving_with(grabbed, to);
        let hands_ms = milliseconds(started);

        // 2. The rollback snapshot `drag_or_leave_it_alone` takes before anything moves.
        let started = Instant::now();
        let snapshot = (
            sketch.points.clone(),
            sketch.arcs.clone(),
            sketch.circles.clone(),
            sketch.conics.clone(),
        );
        let clone_ms = milliseconds(started);
        drop(snapshot);

        // 3. The reach fixpoint plus the standing-constraint filter — both scan the whole
        //    constraint list, and `constraint_stands_within` scans `reach` per reference.
        let held: Vec<_> = hands.iter().map(|hand| hand.point).collect();
        let started = Instant::now();
        let reach = sketch.what_a_drag_of_these_can_reach(&held);
        let standing: Vec<_> = sketch
            .constraints
            .iter()
            .filter(|constraint| sketch.constraint_stands_within(constraint, &reach))
            .copied()
            .collect();
        let reach_ms = milliseconds(started);

        // 4. Building the local problem. Scoped — but the build still walks EVERY point, segment,
        //    arc and circle in the drawing and asks `reach.contains` about each.
        let started = Instant::now();
        let prepared =
            constraint::prepare_scoped(&sketch, &standing, Some(context()), Some(&reach))
                .expect("the slot's problem builds");
        let prepare_ms = milliseconds(started);

        // 5. The kernel itself, on a problem whose size does NOT depend on `count`.
        let started = Instant::now();
        let outcome = prepared.drag_together(&hands, &[]);
        let solve_ms = milliseconds(started);
        assert!(outcome.is_ok(), "the drag resolves");

        let total = hands_ms + clone_ms + reach_ms + prepare_ms + solve_ms;
        println!(
            "{count:<10} {:>7} {:>6} {hands_ms:>8.3} {clone_ms:>8.3} {reach_ms:>8.3} \
             {prepare_ms:>8.3} {solve_ms:>8.3} {total:>8.3}",
            points.len(),
            reach.len()
        );
    }
}
