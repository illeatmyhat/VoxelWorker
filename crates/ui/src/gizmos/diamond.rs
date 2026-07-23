//! `diamond` — the live-cursor marker at the open end of a segment being placed.

use egui::{Painter, Pos2, Shape, Stroke};

use super::{HANDLE_ACCENT, STROKE_HANDLE};

/// A **hollow diamond** — the pointer marker on an open segment. A diamond, not a square, so the
/// live cursor end reads as distinct from the committed vertex handles it runs between.
pub fn diamond(painter: &Painter, center: Pos2, half: f32) {
    let points = vec![
        Pos2::new(center.x, center.y - half),
        Pos2::new(center.x + half, center.y),
        Pos2::new(center.x, center.y + half),
        Pos2::new(center.x - half, center.y),
    ];
    painter.add(Shape::closed_line(points, Stroke::new(STROKE_HANDLE, HANDLE_ACCENT)));
}
