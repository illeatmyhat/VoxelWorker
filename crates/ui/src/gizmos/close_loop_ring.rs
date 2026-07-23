//! `close_loop_ring` — the accent ring the start vertex grows when the loop can close.

use egui::{Painter, Pos2, Stroke};

use super::{HANDLE_ACCENT, STROKE_HANDLE};

/// A **close-loop ring** — the accent ring the start vertex grows when the cursor is near enough
/// to close the polygon, the unmistakable "click here to close". Pairs with a dashed closing run
/// ([`dashed_segment`](super::dashed_segment)) until the click commits it.
pub fn close_loop_ring(painter: &Painter, start: Pos2, radius: f32) {
    painter.circle_stroke(start, radius, Stroke::new(STROKE_HANDLE, HANDLE_ACCENT));
}
