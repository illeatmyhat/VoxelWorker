//! `snap_ticks` — the four-armed "engaged" decoration drawn over a snapped handle.

use egui::{Color32, Painter, Pos2, Stroke};

use super::STROKE_GUIDE;

/// The **snapped-handle tick decoration**: four short solid arms leaving a gap around the thumb
/// (from `inner` to `outer` out of the centre), the "engaged with the lattice" mark drawn over a
/// selected [`vertex_handle`](super::vertex_handle) in its snapped state.
pub fn snap_ticks(painter: &Painter, center: Pos2, inner: f32, outer: f32, color: Color32) {
    let stroke = Stroke::new(STROKE_GUIDE, color);
    for (dx, dy) in [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
        painter.line_segment(
            [
                Pos2::new(center.x + dx * inner, center.y + dy * inner),
                Pos2::new(center.x + dx * outer, center.y + dy * outer),
            ],
            stroke,
        );
    }
}
