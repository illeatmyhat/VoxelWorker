//! `crosshair` — a full cross through a point: the snap tick-cross and the place-point cursor.

use egui::{Color32, Painter, Pos2, Stroke};

use super::{dashed, STROKE_GUIDE};

/// A full **crosshair** through `center`, reaching `reach` points along each axis. `dashed_stroke`
/// picks the kept-ghost / snap-tick idiom (dashed) versus a solid cross. It draws the snap
/// indicator's tick-cross on a locked lattice crossing, and the place-point cursor's aiming cross.
pub fn crosshair(painter: &Painter, center: Pos2, reach: f32, color: Color32, dashed_stroke: bool) {
    let stroke = Stroke::new(STROKE_GUIDE, color);
    let (v0, v1) = (Pos2::new(center.x, center.y - reach), Pos2::new(center.x, center.y + reach));
    let (h0, h1) = (Pos2::new(center.x - reach, center.y), Pos2::new(center.x + reach, center.y));
    if dashed_stroke {
        dashed(painter, v0, v1, stroke);
        dashed(painter, h0, h1, stroke);
    } else {
        painter.line_segment([v0, v1], stroke);
        painter.line_segment([h0, h1], stroke);
    }
}
