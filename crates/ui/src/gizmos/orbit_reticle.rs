//! `orbit_reticle` — the explicit orbit mode's targeting overlay, filling the viewport.

use egui::{Color32, Painter, Rect, Stroke, Vec2};

use crate::theme::color_palette;

/// The ring's diameter as a fraction of the viewport's **vertical** height. It scales off height
/// alone, not the rect's shorter side: the ring is the mode's horizon, and one that narrowed with
/// the side panel would read as a widget rather than as the frame the view turns inside.
const RING_DIAMETER_OF_HEIGHT: f32 = 0.72;
/// Each cardinal tick's length, again against viewport height. The ticks sit OUTSIDE the ring —
/// they mark the four screen axes, so they must not read as ticks *on* a dial.
const TICK_OF_HEIGHT: f32 = 0.085;
/// The centre cross's arm, against viewport height. Small: it marks the target exactly, and a
/// bigger one would hide the very thing being aimed at.
const CENTER_ARM_OF_HEIGHT: f32 = 0.024;
/// One hairline. The reticle covers most of the frame, so any more weight would compete with the
/// model for the whole duration of the mode.
const STROKE_WIDTH: f32 = 1.0;

/// Draw the **targeting reticle**: a ring most of the viewport tall, four cardinal ticks outside
/// it, and a small cross on the exact centre.
///
/// `viewport` is the live 3D viewport rect (egui points). The camera looks AT `camera.target`, so
/// the target projects to that rect's centre by construction — the reticle is drawn there directly
/// rather than projected, which also means it cannot lag the camera by a frame.
///
/// This is the mode's legibility affordance, and the whole reason it can flip the left button's
/// verb without lying — the reticle is what says "left now turns and re-centres", so the flipped
/// verb is visible for as long as it is in force. It is drawn in a single neutral-gray pass at
/// half alpha ([`color_palette::RETICLE`]), NOT in the accent and NOT double-struck with a dark
/// backing the way [`orbit_center`](super::orbit_center()) is: a mark this large has to sit under
/// the model rather than over it. The shell hides it for the duration of a turn, and only a turn:
/// a press that might still become the re-centring click keeps it, because that click aims at it.
pub fn orbit_reticle(painter: &Painter, viewport: Rect) {
    let height = viewport.height();
    let center = viewport.center();
    let radius = height * RING_DIAMETER_OF_HEIGHT * 0.5;
    let tick = height * TICK_OF_HEIGHT;
    let arm = height * CENTER_ARM_OF_HEIGHT;
    let stroke = Stroke::new(STROKE_WIDTH, color_palette::RETICLE);

    painter.circle_stroke(center, radius, stroke);
    for direction in [
        Vec2::new(0.0, -1.0),
        Vec2::new(1.0, 0.0),
        Vec2::new(0.0, 1.0),
        Vec2::new(-1.0, 0.0),
    ] {
        painter.line_segment(
            [
                center + direction * radius,
                center + direction * (radius + tick),
            ],
            stroke,
        );
        painter.line_segment([center - direction * arm, center + direction * arm], stroke);
    }
}

/// The reticle on its own layer, over the scene and under every panel. Registers no chrome rect —
/// the target is re-aimed by clicking THROUGH the viewport, so a reticle that swallowed clicks
/// would block the one gesture the mode exists for.
///
/// [`egui::Order::Background`], not `Foreground` like the orbit-center marker: this mark spans the
/// frame, and one drawn over the floating chrome would strike a line through the display stack.
pub fn orbit_reticle_overlay(ui: &egui::Ui, viewport: Rect) {
    let painter = ui
        .ctx()
        .layer_painter(egui::LayerId::new(
            egui::Order::Background,
            egui::Id::new("orbit_reticle_gizmo"),
        ))
        .with_clip_rect(viewport);
    orbit_reticle(&painter, viewport);
}

/// The palette entry this gizmo uses, named so the colour lint sees it as a theme token rather
/// than as a raw value.
const _: [Color32; 1] = [color_palette::RETICLE];
