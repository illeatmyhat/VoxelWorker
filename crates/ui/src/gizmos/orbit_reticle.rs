//! `orbit_reticle` — the explicit orbit mode's targeting overlay, drawn on `camera.target`.

use egui::{Color32, Painter, Pos2, Stroke, Vec2};

use super::STROKE_HANDLE;
use crate::theme::color_palette;

/// Half the bracket square's side. Wider than [`ORBIT_CENTER_RADIUS`](super::ORBIT_CENTER_RADIUS)
/// so the two pivots never read as the same mark at a glance — they are different points, moved
/// by different mechanisms, and the design note that keeps them apart is worth a silhouette.
const RETICLE_HALF: f32 = 15.0;
/// How far each bracket arm runs from its corner. Short: a bracket is a corner, and four of them
/// closing into a full square would be a rectangle, which the set already spends on FIT.
const BRACKET_ARM: f32 = 6.0;
/// The centre tick's reach — the target itself, small enough not to hide what sits on it.
const CENTER_TICK: f32 = 3.5;

/// Draw the **targeting reticle**: four corner brackets around a small centre cross, at the
/// camera target's projected position.
///
/// This is the mode's legibility affordance, and the whole reason it can flip the left button's
/// verb without lying — the reticle is what says "left now turns and re-centres", so the flipped
/// verb is visible for as long as it is in force.
///
/// Deliberately unlike [`orbit_center`](super::orbit_center)'s ringed crosshair: brackets are the
/// set's "this is the thing being framed" mark (FIT owns them), and the two pivots are separate
/// points that must never be mistaken for each other.
///
/// Every stroke is laid down twice, a dark backing pass under the accent, so it keeps contrast
/// over pale geometry as well as over the dark background.
pub fn orbit_reticle(painter: &Painter, target: Pos2) {
    for (width, hue) in [
        (STROKE_HANDLE + 2.0, color_palette::BG),
        (STROKE_HANDLE, color_palette::ACCENT),
    ] {
        let stroke = Stroke::new(width, hue);
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let corner = target + Vec2::new(sx * RETICLE_HALF, sy * RETICLE_HALF);
            painter.line_segment([corner, corner - Vec2::new(sx * BRACKET_ARM, 0.0)], stroke);
            painter.line_segment([corner, corner - Vec2::new(0.0, sy * BRACKET_ARM)], stroke);
        }
        painter.line_segment(
            [
                target - Vec2::new(CENTER_TICK, 0.0),
                target + Vec2::new(CENTER_TICK, 0.0),
            ],
            stroke,
        );
        painter.line_segment(
            [
                target - Vec2::new(0.0, CENTER_TICK),
                target + Vec2::new(0.0, CENTER_TICK),
            ],
            stroke,
        );
    }
}

/// The reticle on its own foreground layer, over the scene and the sketch overlay alike. Mirrors
/// [`orbit_center_overlay`](super::orbit_center_overlay)'s layer discipline and, like it,
/// registers no chrome rect — the target is re-aimed by clicking THROUGH the viewport, so a
/// reticle that swallowed clicks would block the one gesture the mode exists for.
pub fn orbit_reticle_overlay(ui: &egui::Ui, target: Pos2) {
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("orbit_reticle_gizmo"),
    ));
    orbit_reticle(&painter, target);
}

/// The palette entries this gizmo uses, named so the colour lint sees them as theme tokens
/// rather than as raw values.
const _: [Color32; 2] = [color_palette::ACCENT, color_palette::BG];
