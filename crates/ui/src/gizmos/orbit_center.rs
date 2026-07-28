//! `orbit_center` — the camera's placed pivot, drawn where it sits in space.

use egui::{Color32, Painter, Pos2, Stroke, Vec2};

use super::STROKE_HANDLE;
use crate::theme::color_palette;

/// Radius (points) of the marker's ring. Large enough to find at a glance without covering the
/// feature it is placed on.
pub const ORBIT_CENTER_RADIUS: f32 = 9.0;

/// How far the crosshair arms reach past the ring.
const ARM_OVERSHOOT: f32 = 5.0;
/// The gap between the centre dot and the inner end of each arm, so the dot stays readable.
const ARM_INNER_GAP: f32 = 3.5;
/// The centre dot's radius — the pivot itself, as exactly as a screen can say it.
const DOT_RADIUS: f32 = 2.0;

/// Draw the **orbit center**: a ringed crosshair around a filled centre dot, at the pivot's
/// projected position.
///
/// `placing` brightens it to the hover step while a placement is armed and the marker is riding
/// the cursor, matching every other manipulator's "you are moving this right now" state.
///
/// Deliberately unlike [`vertex_handle`](super::vertex_handle)'s square thumb: a vertex is a
/// thing you grab, and this is a thing you turn around. It is also never occluded — it draws in
/// the screen-space overlay pass rather than the depth-tested scene, because a pivot placed on
/// the far side of the model is exactly when you most need to see where it went.
///
/// Every stroke is laid down twice, a dark backing pass under the accent, so the marker keeps
/// its contrast over pale geometry as well as over the dark background.
pub fn orbit_center(painter: &Painter, center: Pos2, placing: bool) {
    let color = if placing {
        color_palette::HANDLE_HOVER
    } else {
        color_palette::ACCENT
    };
    let backing = color_palette::BG;
    let arm = ORBIT_CENTER_RADIUS + ARM_OVERSHOOT;

    for (width, hue) in [(STROKE_HANDLE + 2.0, backing), (STROKE_HANDLE, color)] {
        let stroke = Stroke::new(width, hue);
        painter.circle_stroke(center, ORBIT_CENTER_RADIUS, stroke);
        for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
            let direction = Vec2::new(dx, dy);
            painter.line_segment(
                [center + direction * ARM_INNER_GAP, center + direction * arm],
                stroke,
            );
        }
    }
    painter.circle_filled(center, DOT_RADIUS + 1.0, backing);
    painter.circle_filled(center, DOT_RADIUS, color);
}

/// The marker drawn on its own foreground layer, so it sits over the scene and over the sketch
/// overlay alike. Mirrors [`sketch_vertex_handles`](crate::chrome::sketch_vertex_handles)'s
/// layer discipline; it registers no chrome rect, because the pivot is not grabbable — it is
/// moved by the context menu, never by dragging it.
pub fn orbit_center_overlay(ui: &egui::Ui, center: Pos2, placing: bool) {
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("orbit_center_gizmo"),
    ));
    orbit_center(&painter, center, placing);
}

/// The palette entries this gizmo uses, named so the colour lint sees them as theme tokens
/// rather than as raw values.
const _: [Color32; 3] = [
    color_palette::ACCENT,
    color_palette::HANDLE_HOVER,
    color_palette::BG,
];
