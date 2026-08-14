//! What DRAWING the sketch costs, split from what deciding the sketch costs.
//!
//! The owner's report: "the highlighted faces/voxels render faster than the sketch curves. Is it
//! because the sketch curves are written in egui and not as a shader?" The highlight is a GPU pass
//! — the CPU hands it a handful of numbers and the fragment shader does the rest, so its cost does
//! not grow with how much is lit. The sketch overlay is the other kind: every frame it re-projects
//! the drawing, re-flattens every turn into chords, hands the chords to egui as polylines, and egui
//! turns those into triangles ON THE CPU before anything reaches the GPU.
//!
//! So the question has a measurable form: of the per-frame overlay budget, how much is egui's
//! painting and tessellation, and how much is everything upstream of it? This probe times the egui
//! half against a drawing's worth of curves, so the other half has something to be compared to.
//!
//! # What it found: the answer is no
//!
//! At 1024 curves of 32 chords each plus 1024 dots — far past any drawing an author will make —
//! painting costs 0.21 ms and tessellation 0.53 ms. A realistic sketch is under a tenth of a
//! millisecond for both together, so porting the overlay to a shader would buy nothing.
//!
//! The cost that IS there sits upstream, and the second probe below separates it. Cloning the whole
//! `SketchSolid` twenty times a frame — which the overlay refresh does — costs about a tenth of a
//! millisecond even at 448 points, so the clones are not it either. Deriving the FACES is: about
//! 0.01 ms when the region memo answers, and whole milliseconds when it does not — and it does not
//! on any frame that moved the drawing, which is every frame of a drag.
//!
//! That cost grows about quadratically, because the arrangement cuts every curve against every
//! other: 0.03 ms at 28 points, 0.35 at 112, and 4 to 11 at 448. The last figure is deliberately a
//! RANGE. Repeated isolated runs on this machine gave 4.4, 4.6, 7.5, 10.8 and 10.9 ms for the same
//! work, so the reading is not stable to better than a factor of two and quoting one number would
//! claim a precision the measurement does not have. The SHAPE is the robust part, and the shape is
//! quadratic.
//!
//! An axis-aligned bounding-box broadphase was spiked over `cut_at_crossings` — the slots are
//! disjoint, so every cross-slot pair test is wasted — and measured NO win at all, with the face
//! counts unchanged. So the quadratic term is not the crossing solves, and the cost lives somewhere
//! else inside the arrangement. Where, exactly, is not yet measured.
//!
//! Run: `cargo test --release --test sketch_overlay_cost_probe -- --ignored --nocapture`

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::suboptimal_flops
)]

use std::time::Instant;

use egui::{Context, Pos2, RawInput, Rect, Vec2};

/// A screen the size of a real viewport, so the clip work is the work the app does.
fn screen() -> Rect {
    Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0))
}

/// `curves` polylines of `chords` points each, laid out as a grid of arcs across the viewport —
/// the shape of what an overlay hands the painter, at the sizes it hands it.
fn curve_lines(curves: usize, chords: usize) -> Vec<ui::chrome::SketchCurveLine> {
    (0..curves)
        .map(|index| {
            let seat = Pos2::new(
                40.0 + (index % 16) as f32 * 76.0,
                40.0 + (index / 16) as f32 * 76.0,
            );
            ui::chrome::SketchCurveLine {
                chords: (0..chords)
                    .map(|step| {
                        let turn =
                            step as f32 / (chords.max(2) - 1) as f32 * std::f32::consts::FRAC_PI_2;
                        Pos2::new(seat.x + turn.cos() * 32.0, seat.y + turn.sin() * 32.0)
                    })
                    .collect(),
                state: ui::gizmos::HandleState::Idle,
                ink: ui::chrome::SketchCurveInk::Real,
            }
        })
        .collect()
}

/// One dot per curve end, which is roughly what a drawing carries.
fn vertex_handles(count: usize) -> Vec<ui::chrome::SketchVertexHandle> {
    (0..count)
        .map(|index| ui::chrome::SketchVertexHandle {
            at: Pos2::new(
                40.0 + (index % 32) as f32 * 38.0,
                40.0 + (index / 32) as f32 * 38.0,
            ),
            state: ui::gizmos::HandleState::Idle,
            ink: ui::chrome::SketchVertexInk::OnInk,
        })
        .collect()
}

#[test]
#[ignore = "perf probe - run in release with --ignored --nocapture"]
fn what_egui_costs_to_draw_a_sketch() {
    println!(
        "{:>7} {:>7} {:>7} {:>10} {:>12} {:>10} {:>10}",
        "curves", "chords", "dots", "shapes", "paint ms", "tess ms", "triangles"
    );
    for (curves, chords) in [
        (16_usize, 8_usize),
        (64, 8),
        (64, 32),
        (256, 32),
        (1024, 32),
    ] {
        let lines = curve_lines(curves, chords);
        let dots = vertex_handles(curves);
        let context = Context::default();
        let viewport = screen();

        // One warm pass, so neither clock is reading egui's first-frame setup.
        drop(context.run_ui(RawInput::default(), |ui| {
            ui::chrome::sketch_arc_curves(ui, viewport, &lines);
        }));

        let started = Instant::now();
        let output = context.run_ui(
            RawInput {
                screen_rect: Some(viewport),
                ..Default::default()
            },
            |ui| {
                ui::chrome::sketch_arc_curves(ui, viewport, &lines);
                ui::chrome::sketch_vertex_handles(ui, viewport, &dots, &mut Vec::new());
            },
        );
        let paint_ms = started.elapsed().as_secs_f64() * 1000.0;
        let shapes = output.shapes.len();

        let started = Instant::now();
        let primitives = context.tessellate(output.shapes, output.pixels_per_point);
        let tess_ms = started.elapsed().as_secs_f64() * 1000.0;
        let triangles: usize = primitives
            .iter()
            .map(|clipped| match &clipped.primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh.indices.len() / 3,
                egui::epaint::Primitive::Callback(_) => 0,
            })
            .sum();

        println!(
            "{curves:>7} {chords:>7} {:>7} {shapes:>10} {paint_ms:>12.3} {tess_ms:>10.3} \
             {triangles:>10}",
            dots.len()
        );
    }
}

/// What the overlay costs UPSTREAM of egui: the clones the refresh takes, and the face derivation
/// it asks for once a frame.
///
/// The egui half above is under a tenth of a millisecond for any real drawing, so if the overlay
/// costs a frame anything it is spent here. Two things are worth separating: `sketch_node_state`
/// clones the whole `SketchSolid` and is called about twenty times per refresh, and `faces()` is a
/// graph walk whose memo lives ON the sketch — so a clone answers the same question from cold.
#[test]
#[ignore = "perf probe - run in release with --ignored --nocapture"]
fn what_the_overlay_costs_before_egui_sees_it() {
    use document::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};

    let context = parametric::EvaluationContext::new(
        std::num::NonZeroU32::new(16).expect("probe density is non-zero"),
    );
    println!(
        "{:>6} {:>7} {:>12} {:>12} {:>12} {:>12}",
        "slots", "points", "clone us", "x20 ms", "faces warm ms", "faces cold ms"
    );
    for slots in [1_usize, 4, 16, 64] {
        let mut made = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 32);
        for index in 0..slots {
            let center = (index as i64) * 40;
            made = made
                .with_center_arc_slot(
                    SketchPoint::new(center, 0),
                    SketchPoint::new(center + 8, 0),
                    SketchPoint::new(center, 8),
                    parametric::sketch::ArcTurn::CounterClockwise,
                    SketchPoint::new(center + 10, 0),
                    context,
                )
                .expect("a slot the plane has room for");
        }
        // Warm the memo the way an in-place refresh finds it.
        drop(made.sketch.faces(context));

        let started = Instant::now();
        for _ in 0..20 {
            std::hint::black_box(made.clone());
        }
        let twenty_clones_ms = started.elapsed().as_secs_f64() * 1000.0;

        let started = Instant::now();
        std::hint::black_box(made.sketch.faces(context));
        let warm_ms = started.elapsed().as_secs_f64() * 1000.0;

        let copy = made.clone();
        let started = Instant::now();
        std::hint::black_box(copy.sketch.faces(context));
        let cold_ms = started.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{slots:>6} {:>7} {:>12.1} {twenty_clones_ms:>12.3} {warm_ms:>13.3} {cold_ms:>13.3}              faces={}",
            made.sketch.points().len(),
            twenty_clones_ms * 1000.0 / 20.0,
            made.sketch.faces(context).len()
        );
    }
}
