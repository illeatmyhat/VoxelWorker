//! `snap_ticks` — the four-armed "engaged" decoration drawn over a snapped handle.

use egui::{Color32, Painter, Pos2, Stroke};

use super::STROKE_GUIDE;

/// The **snapped-handle tick decoration**: four short solid arms leaving a gap around the thumb
/// (from `inner` to `outer` out of the center), the "engaged with the lattice" mark drawn over a
/// selected [`vertex_handle`](super::vertex_handle()) in its snapped state.
pub fn snap_ticks(painter: &Painter, center: Pos2, inner: f32, outer: f32, color: Color32) {
    snap_ticks_weighted(painter, center, inner, outer, color, STROKE_GUIDE);
}

/// The same four arms at a stated weight.
///
/// A decoration over a filled thumb is read against the thumb and can be hairline; the same mark
/// standing on its own over a lit curve has nothing behind it and disappears at that weight. The
/// arms are the vocabulary and stay identical either way — only how hard they are drawn changes.
pub fn snap_ticks_weighted(
    painter: &Painter,
    center: Pos2,
    inner: f32,
    outer: f32,
    color: Color32,
    width: f32,
) {
    let stroke = Stroke::new(width, color);
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
