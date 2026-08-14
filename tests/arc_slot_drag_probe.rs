//! What dragging an ARC slot costs per frame, against the same slot drawn straight.
//!
//! The owner's report: "working with arc slots feels kind of slow compared to everything else."
//! This measures it, at the speeds a hand actually moves, and against a control that differs in the
//! CURVE and nothing else — same handle count, same relation count, same width.
//!
//! # What it found
//!
//! The whole difference is the walk. A SNAPPED drag is a rotation, and a rotation is the one motion
//! a linearization is worst at, so a snapped drag is arrived at a degree at a time — up to sixteen
//! substeps, each one a full three-pass gesture. A straight slot's snap has no turn in it, so it
//! never walks and pays three passes flat. Counted with temporary instrumentation inside the
//! solver, one frame of the sweep below ran:
//!
//! | hand speed  | arc slot          | straight slot    |
//! | ----------- | ----------------- | ---------------- |
//! | 4 px/frame  | 9 solves, 1.3 ms  | 6 solves, 0.8 ms |
//! | 12 px/frame | 27 solves, 4.2 ms | 6 solves, 1.0 ms |
//! | 40 px/frame | 48 solves, 9.0 ms | 6 solves, 1.7 ms |
//!
//! Nine milliseconds is most of a 60 Hz frame, and the walk is the entire multiplier. Every solve
//! converges — the cost is the NUMBER of them, not any one of them going wrong.
//!
//! # The sweep is a ROTATION, and that is not a detail
//!
//! A rail end travels on a circle; that is what makes the shape an arc slot. Swept in a straight
//! line instead, it walks off its own geometry within a few frames, after which every number is the
//! cost of failing to satisfy a drawing the hand has already broken — the passes stop converging,
//! land at 1e-6 instead of 1e-11, and burn their whole iteration ceiling. That reads as a solver
//! defect and is not one. Measured under a trust radius varied over 160x, which moved the residual
//! only between 1.2e-6 and 2.0e-6 and the clock not at all: the failure is the drawing's, not the
//! numerics'.
//!
//! # Sweeping toward a full circle is the same cost, paid more times
//!
//! The owner's second report was narrower than the first: not that an arc slot is slow but that
//! "sweeping towards a full circle" is. Those would be different faults. The walk is flat in the
//! arc's own sweep; a parameterization degenerating as the arc closes on its own tail would RISE.
//! Carried round in even eight-degree steps from ninety degrees to three hundred and fifty-four,
//! the frame does not move — 3.4 ms at both ends and no trend between. So closure has no fault of
//! its own. What a full-circle sweep costs is the ordinary frame paid forty-five times running,
//! with no cheap frame anywhere in it to hide behind, which is what makes it the sweep a hand
//! notices.
//!
//! Run: `cargo test --release --test arc_slot_drag_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::suboptimal_flops
)]

use std::time::Instant;

use document::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};

const DENSITY: u32 = 16;
const DEPTH_VOXELS: u32 = 32;

const fn context() -> parametric::EvaluationContext {
    parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(DENSITY).expect("probe density is non-zero"),
    )
}

/// `count` center-arc slots side by side — the shape the owner reported as the slow one, built the
/// same way the population probe next door builds it so the two reports are about one drawing.
fn arc_slots(count: usize) -> SketchSolid {
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
            .expect("a slot the plane has room for");
    }
    made
}

/// The control: the same slot drawn straight, so the comparison isolates the curve.
fn linear_slots(count: usize) -> SketchSolid {
    let mut made = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), DEPTH_VOXELS);
    for index in 0..count {
        let center = (index as i64) * 40;
        made = made
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::CenterToCenter,
                SketchPoint::new(center, 0),
                SketchPoint::new(center + 8, 0),
                SketchPoint::new(center + 8, 2),
                context(),
            )
            .expect("a slot the plane has room for");
    }
    made
}

/// WHICH point of an arc slot is the expensive one to drag.
///
/// The population probe next door reports a slot drag averaged over one grabbed point per drawing,
/// and one row of it does not fit the others. A per-point sweep says why: the cost belongs to two
/// of the seven points and is flat in the population, so an average over points reads it as a
/// property of the drawing's size when it is a property of which dot the hand is on.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn which_point_of_an_arc_slot_is_slow_to_drag() {
    for count in [1_usize, 2, 8] {
        let made = arc_slots(count);
        println!(
            "\n{count} center-arc slot(s): {} points, {} relations",
            made.sketch.points().len(),
            made.sketch.constraints().len()
        );
        println!(
            "{:>5} {:>7} {:>9} {:>9} {:>9}",
            "index", "derived", "drag ms", "settle", "hair"
        );
        for (index, point) in made.sketch.points().iter().enumerate() {
            let (id, derived) = (point.id, made.sketch.is_arc_center(point.id));
            let mut preview = made.clone();
            let was = point.at.in_plane();
            let to = SketchPoint::from_continuous(was[0] + 1.0, was[1] + 1.0);

            let started = Instant::now();
            let moved = preview.sketch.move_point(id, to, context());
            let drag_ms = started.elapsed().as_secs_f64() * 1000.0;

            // The bare settle on the drawing the drag just produced, with nothing moving. If the
            // drag costs many times this, the cost is not the constraint problem.
            let mut settled = preview.clone();
            let started = Instant::now();
            drop(settled.sketch.solve(context()));
            let settle_ms = started.elapsed().as_secs_f64() * 1000.0;

            // The same drag with the snap ceiling closed down to a hair. The reach is a MEANINGLESS
            // knob for the cost of settling a constraint problem — if the clock moves when it does,
            // the time is going into the snap path, not into the solve. It moves by fourteen times.
            let mut reached = made.clone();
            let started = Instant::now();
            drop(reached.sketch.move_point_reporting_its_snap(
                id,
                to,
                context(),
                document::sketch::SnapReach::of_length(1e-6),
                &mut [],
            ));
            let reached_ms = started.elapsed().as_secs_f64() * 1000.0;

            let note = if moved.is_err() { " REFUSED" } else { "" };
            println!(
                "{index:>5} {:>7} {drag_ms:>9.2} {settle_ms:>9.2} {reached_ms:>9.2}{note}",
                if derived { "yes" } else { "" }
            );
        }
    }
}

/// What a LIVE drag costs, frame by frame, at the speeds a hand actually moves.
///
/// The per-point probe above hands the whole travel over in one call, which is what a test does and
/// not what a shell does. This one swings a rail end around its arc the way sixty frames a second
/// would, at three hand speeds, against the straight slot swept the same way.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn what_a_live_slot_sweep_costs_per_frame() {
    // Pixels a hand covers between two frames at 60 Hz, turned into drawing units by a viewport
    // showing about sixty units across eight hundred pixels: thirteen pixels to the unit.
    const PER_UNIT: f64 = 13.0;
    for (shape, build) in [
        ("center-arc", arc_slots as fn(usize) -> SketchSolid),
        ("straight", linear_slots as fn(usize) -> SketchSolid),
    ] {
        for pixels_per_frame in [4.0_f64, 12.0, 40.0] {
            let step = pixels_per_frame / PER_UNIT;
            let made = build(1);
            let id = made.sketch.points()[3].id;
            let mut live = made.clone();
            let (mut worst, mut total) = (0.0_f64, 0.0_f64);
            for _ in 0..12 {
                let was = live
                    .sketch
                    .points()
                    .iter()
                    .find(|point| point.id == id)
                    .expect("the point the sweep is holding")
                    .at
                    .in_plane();
                // Swung about the slot's own center rather than pushed in a straight line — see the
                // module header for what a straight sweep measures instead.
                let turn = step / was[0].hypot(was[1]).max(1.0);
                let to = SketchPoint::from_continuous(
                    was[0].mul_add(turn.cos(), -(was[1] * turn.sin())),
                    was[0].mul_add(turn.sin(), was[1] * turn.cos()),
                );
                let started = Instant::now();
                drop(live.sketch.move_point_reporting_its_snap(
                    id,
                    to,
                    context(),
                    document::sketch::SnapReach::of_length(9.0),
                    &mut [],
                ));
                let frame_ms = started.elapsed().as_secs_f64() * 1000.0;
                worst = worst.max(frame_ms);
                total += frame_ms;
            }
            println!(
                "{shape:>11} slot at {pixels_per_frame:>2.0} px/frame: mean {:>5.2} ms, \
                 worst {worst:>5.2} ms",
                total / 12.0
            );
        }
    }
}

/// What the cost does as the arc is swept all the way round toward CLOSING on itself.
///
/// The owner's second report is narrower than the first: not "an arc slot is slow" but "sweeping
/// toward a full circle is". Those are different claims. The first is the walk, which is flat in
/// the arc's own sweep. The second would be a cost that GROWS as the arc approaches its own tail —
/// a parameterization degenerating at closure, which the walk says nothing about. This carries a
/// rail end round in even steps and prints the frame against the angle the arc has reached, so the
/// two claims are told apart by their SHAPE rather than by their average.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn what_a_sweep_costs_as_the_arc_closes_on_itself() {
    const PER_FRAME: f64 = 8.0_f64 * std::f64::consts::PI / 180.0;
    let mut live = arc_slots(1);
    let id = live.sketch.points()[3].id;
    println!("{:>8} {:>10} {:>9}", "reached", "frame ms", "radius");
    let mut swept = 90.0_f64;
    for frame in 0..34 {
        let was = live
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .expect("the point the sweep is holding")
            .at
            .in_plane();
        let to = SketchPoint::from_continuous(
            was[0].mul_add(PER_FRAME.cos(), -(was[1] * PER_FRAME.sin())),
            was[0].mul_add(PER_FRAME.sin(), was[1] * PER_FRAME.cos()),
        );
        let started = Instant::now();
        drop(live.sketch.move_point_reporting_its_snap(
            id,
            to,
            context(),
            document::sketch::SnapReach::of_length(9.0),
            &mut [],
        ));
        let frame_ms = started.elapsed().as_secs_f64() * 1000.0;
        swept += 8.0;
        let now = live
            .sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .expect("the point the sweep is holding")
            .at
            .in_plane();
        if frame % 2 == 0 {
            println!(
                "{swept:>7.0}° {frame_ms:>10.2} {:>9.2}",
                now[0].hypot(now[1])
            );
        }
    }
}
