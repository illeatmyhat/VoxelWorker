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
//! # THE SHELL REPLAYS THE WHOLE GESTURE FROM THE PRESS, EVERY FRAME
//!
//! Read this before believing any other number in this file, because every other probe here nudges
//! the drawing on from where the last frame left it and the shell does not do that.
//! `render.rs` rebuilds the preview from `drag.original` each frame and hands the drag the cursor's
//! whole displacement from the PRESS. So what the walk is asked to turn grows for as long as the
//! gesture lasts, and a frame late in a sweep is doing the entire sweep again.
//!
//! Measured on one hand at one speed, four degrees a frame, both paths side by side:
//!
//! | swept | replayed from the press | nudged from last frame |
//! | ----- | ----------------------- | ---------------------- |
//! | 12°   | 5.2 ms                  | 1.4 ms                 |
//! | 60°   | 8.4 ms                  | 1.9 ms                 |
//! | 120°  | 12.1 ms                 | 1.5 ms                 |
//! | 180°  | 14.8 ms                 | 1.8 ms                 |
//!
//! The nudged column is flat and the replayed one triples. **This is the whole of why a long sweep
//! feels dearer per frame than a short one**, and the owner reported it as such before it was
//! measured. It is not the arc, not the distance from the origin, and not the closure.
//!
//! Two things make it worse than the substep count alone suggests. `walk_of` caps at
//! `MOST_FRAMES = 16`, so past sixteen degrees of gesture the walk is no longer a degree a step but
//! `total / 16` — at 180° swept that is eleven degrees a step, an order coarser than the law it is
//! named for. And each of those steps is a longer throw, so it costs more iterations as well as
//! being cheaper reasoning. Late in a sweep the shell pays the most it can pay for the coarsest
//! walk it can take.
//!
//! Replaying is not thoughtless — see `drag_together` and `point_move_attempt` for why a gesture is
//! measured from its opening: summing frame over frame lets the kept quantity creep, and the snap
//! cone wants to grow with the drag rather than shrink with the step. Which of those survives an
//! incremental preview is a design question and an owner's call, not a speed-up to help oneself to.
//!
//! # A sweep costs its ARC, if it is nudged
//!
//! On the nudged path, cost tracks the turn in the frame and nothing else.
//!
//! What a frame costs is how far it TURNS, because the walk spends about one substep per degree.
//! The same 264 degrees, sliced five ways:
//!
//! | per frame | frames | per frame | whole sweep |
//! | --------- | ------ | --------- | ----------- |
//! | 2°        | 132    | 0.79 ms   | 104 ms      |
//! | 4°        | 66     | 1.61 ms   | 106 ms      |
//! | 8°        | 33     | 3.25 ms   | 107 ms      |
//! | 16°       | 17     | 6.92 ms   | 118 ms      |
//! | 24°       | 11     | 7.25 ms   | 80 ms       |
//!
//! The whole sweep costs the same however it is cut, so there is no amortization left to find: the
//! work is already proportional to the arc and to nothing else. Twenty-four degrees a frame is
//! cheaper only because `MOST_FRAMES` truncates the walk there — sixteen substeps for twenty-four
//! degrees is a step and a half each, coarser than the degree the walk is built on, so that row buys
//! its eighty milliseconds by walking less rather than by walking better.
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
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
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
    let mut every_frame = 0.0_f64;
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
        every_frame += frame_ms;
        if frame % 2 == 0 {
            println!(
                "{swept:>7.0}° {frame_ms:>10.2} {:>9.2}",
                now[0].hypot(now[1])
            );
        }
    }
    println!("mean over ALL 34 frames: {:.2} ms", every_frame / 34.0);
}

/// Whether the same sweep costs less when it arrives in more, smaller frames.
///
/// The owner asked whether the walk could be amortized — nudged on from where the last frame left
/// it instead of restarted. It already is: `was` is read off the drawing at the top of every call,
/// so a frame walks from where the drawing rests to where the cursor now is, and nothing is measured
/// from the grab. What that leaves open is the question this answers: if the walk spends one substep
/// per degree, then a sweep costs its ARC and not its frame count, and slicing it finer buys nothing
/// but more frames to pay in.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn whether_a_sweep_costs_less_when_it_arrives_in_smaller_frames() {
    const SWEEP: f64 = 264.0;
    println!(
        "{:>10} {:>8} {:>11} {:>11}",
        "per frame", "frames", "per frame", "whole sweep"
    );
    for degrees in [2.0_f64, 4.0, 8.0, 16.0, 24.0] {
        let step = degrees * std::f64::consts::PI / 180.0;
        let frames = (SWEEP / degrees).round() as usize;
        let mut live = arc_slots(1);
        let id = live.sketch.points()[3].id;
        let started = Instant::now();
        for _ in 0..frames {
            let was = live
                .sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .expect("the point the sweep is holding")
                .at
                .in_plane();
            let to = SketchPoint::from_continuous(
                was[0].mul_add(step.cos(), -(was[1] * step.sin())),
                was[0].mul_add(step.sin(), was[1] * step.cos()),
            );
            drop(live.sketch.move_point_reporting_its_snap(
                id,
                to,
                context(),
                document::sketch::SnapReach::of_length(9.0),
                &mut [],
            ));
        }
        let whole = started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{degrees:>9.0}° {frames:>8} {:>9.2} ms {whole:>8.0} ms",
            whole / frames as f64
        );
    }
}

/// Whether a LONG sweep gets dearer per frame than a short one at the same hand speed.
///
/// The owner reports that it does, and the probes next door say it should not: cost tracks the turn
/// in the frame, and the turn here is the same every frame. Either something accumulates across the
/// frames of one drag, or the fixture is not the product. This holds the hand speed fixed and prints
/// the cost against how many frames the drag has already run, which is the one thing the other
/// probes average away.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn whether_a_long_sweep_gets_dearer_per_frame_than_a_short_one() {
    const PER_FRAME: f64 = 2.0_f64 * std::f64::consts::PI / 180.0;
    const FRAMES: usize = 180;
    let mut live = arc_slots(1);
    let id = live.sketch.points()[3].id;
    let mut each = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
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
        each.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    println!("{:>12} {:>10} {:>10}", "frames in", "mean ms", "worst ms");
    for (block, run) in each.chunks(30).enumerate() {
        let mean = run.iter().sum::<f64>() / run.len() as f64;
        let worst = run.iter().fold(0.0_f64, |best, ms| best.max(*ms));
        println!(
            "{:>9}-{:<3} {mean:>10.2} {worst:>10.2}",
            block * 30,
            block * 30 + run.len()
        );
    }
}

/// The drag the SHELL actually performs: the whole gesture replayed from the press, every frame.
///
/// Every probe above nudges the drawing on from where the last frame left it, and every one of them
/// reports a flat frame. The shell does not do that. `render.rs` rebuilds the preview from
/// `drag.original` each frame and hands the drag the cursor's whole displacement from the PRESS, so
/// what the walk is asked to turn grows for as long as the gesture lasts. That is a different cost
/// curve, and it is the one an author feels.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn what_the_shells_replay_from_the_press_costs_as_the_sweep_grows() {
    const PER_FRAME: f64 = 4.0_f64 * std::f64::consts::PI / 180.0;
    let original = arc_slots(1);
    let id = original.sketch.points()[3].id;
    let stood = original
        .sketch
        .points()
        .iter()
        .find(|point| point.id == id)
        .expect("the point the sweep is holding")
        .at
        .in_plane();
    println!(
        "{:>8} {:>10} {:>10} {:>9}",
        "swept", "replay ms", "nudge ms", "clone ms"
    );
    // The nudged drawing runs alongside, carried on frame to frame, so both columns are the same
    // hand at the same speed and differ only in what the frame is measured FROM.
    let mut nudged = arc_slots(1);
    for frame in 1..=45_u32 {
        let turn = PER_FRAME * f64::from(frame);
        let to = SketchPoint::from_continuous(
            stood[0].mul_add(turn.cos(), -(stood[1] * turn.sin())),
            stood[0].mul_add(turn.sin(), stood[1] * turn.cos()),
        );

        let started = Instant::now();
        let mut preview = original.clone();
        let clone_ms = started.elapsed().as_secs_f64() * 1000.0;
        drop(preview.sketch.move_point_reporting_its_snap(
            id,
            to,
            context(),
            document::sketch::SnapReach::of_length(9.0),
            &mut [],
        ));
        let replay_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        drop(nudged.sketch.move_point_reporting_its_snap(
            id,
            to,
            context(),
            document::sketch::SnapReach::of_length(9.0),
            &mut [],
        ));
        let nudge_ms = started.elapsed().as_secs_f64() * 1000.0;

        if frame % 3 == 0 {
            println!(
                "{:>7.0}° {replay_ms:>10.2} {nudge_ms:>10.2} {clone_ms:>9.2}",
                turn.to_degrees()
            );
        }
    }
}

/// Whether the sixteen-step cap is degrading the ANSWER on a long sweep, not just the clock.
///
/// `walk_of` exists because a single 7.8-degree step collapsed a slot's rails from 36/40/44 to
/// 33.5/38.3/43.2 — that measurement is in its own doc comment, and it is the whole argument for
/// walking a degree at a time. But `MOST_FRAMES` caps the walk at sixteen steps, and the shell hands
/// it the gesture's WHOLE displacement, so a hundred and eighty degrees is walked in steps of
/// eleven. That is coarser than the step the walk's own justification says loses the width.
///
/// So this asks the same 180 degrees two ways and compares the rails, not the clock: once as the
/// shell asks it, and once at the granularity the law names. It comes back RED, and not narrowly —
/// a slot four units wide is drawn seven and a half wide by the end of a half turn:
///
/// | swept | step taken | slot width | error |
/// | ----- | ---------- | ---------- | ----- |
/// | 20°   | 1.25°      | 4.26       | +7%   |
/// | 60°   | 3.75°      | 5.30       | +33%  |
/// | 100°  | 6.25°      | 6.32       | +58%  |
/// | 160°  | 10.00°     | 7.58       | +90%  |
///
/// Walked a degree at a time over the same travel the width holds to within four percent. So the
/// cost this file spent its first half measuring is not the whole of what the replay is doing: on a
/// long sweep the shell pays the most it can pay AND hands back a drawing the author did not ask
/// for. The invariant the walk exists to protect is already being violated on exactly the gestures
/// the owner reported as slow.
#[test]
#[ignore = "perf probe — run in release with --ignored --nocapture"]
fn whether_the_step_cap_moves_the_answer_on_a_long_sweep() {
    let original = arc_slots(1);
    let id = original.sketch.points()[3].id;
    let stood = original
        .sketch
        .points()
        .iter()
        .find(|point| point.id == id)
        .expect("the point the sweep is holding")
        .at
        .in_plane();
    let target = |turn: f64| {
        SketchPoint::from_continuous(
            stood[0].mul_add(turn.cos(), -(stood[1] * turn.sin())),
            stood[0].mul_add(turn.sin(), stood[1] * turn.cos()),
        )
    };
    let sweep = |made: &SketchSolid| {
        made.sketch
            .points()
            .iter()
            .map(|point| {
                let at = point.at.in_plane();
                at[0].hypot(at[1])
            })
            .collect::<Vec<f64>>()
    };

    // As the shell asks it: the whole gesture from the press, in one call, capped at sixteen steps
    // of eleven degrees each.
    let mut replayed = original.clone();
    drop(replayed.sketch.move_point_reporting_its_snap(
        id,
        target(std::f64::consts::PI),
        context(),
        document::sketch::SnapReach::of_length(9.0),
        &mut [],
    ));

    // At the granularity the walk is named for: a hundred and eighty frames of one degree, each of
    // which the walk covers in a single step of exactly one degree.
    let mut nudged = original.clone();
    for frame in 1..=180_u32 {
        drop(nudged.sketch.move_point_reporting_its_snap(
            id,
            target(f64::from(frame) * std::f64::consts::PI / 180.0),
            context(),
            document::sketch::SnapReach::of_length(9.0),
            &mut [],
        ));
    }

    // The curve the author actually watches: the shell recomputes from the press on EVERY frame, so
    // this is the width on screen at each point of the sweep, not just at the end of it.
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "swept", "step taken", "slot width", "error"
    );
    for degrees in (20..=180).step_by(20) {
        let mut frame = original.clone();
        drop(frame.sketch.move_point_reporting_its_snap(
            id,
            target(f64::from(degrees) * std::f64::consts::PI / 180.0),
            context(),
            document::sketch::SnapReach::of_length(9.0),
            &mut [],
        ));
        let radii = sweep(&frame);
        let width = radii
            .iter()
            .fold(0.0_f64, |widest, radius| widest.max(*radius))
            - 6.0;
        println!(
            "{degrees:>7}° {:>11.2}° {width:>12.3} {:>9.0}%",
            f64::from(degrees) / 16.0,
            (width - 4.0) / 4.0 * 100.0
        );
    }

    println!();
    println!(
        "{:>5} {:>10} {:>12} {:>10} {:>10}",
        "point", "at rest", "replayed 16x", "nudged 1x", "moved by"
    );
    let (rest, coarse, fine) = (sweep(&original), sweep(&replayed), sweep(&nudged));
    for (index, ((was, hard), soft)) in rest.iter().zip(&coarse).zip(&fine).enumerate() {
        println!(
            "{index:>5} {was:>10.4} {hard:>12.4} {soft:>10.4} {:>10.4}",
            hard - soft
        );
    }
}
