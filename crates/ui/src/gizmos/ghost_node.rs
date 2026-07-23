//! `ghost_node` — the pre-first-point plane affordance's vertex, the one kept ghost.

use egui::{Painter, Pos2, Rect, Stroke, Vec2};

use super::{dashed_rect, HANDLE_ACCENT, STROKE_HANDLE};

/// A **hollow ghost node** — a dashed, unfilled square that says "a point WILL land here" without
/// being a real entity yet. The only ghost the mode keeps (ADR 0028 §6); everything else the
/// author manipulates is a real object.
pub fn ghost_node(painter: &Painter, center: Pos2, half: f32) {
    dashed_rect(
        painter,
        Rect::from_center_size(center, Vec2::splat(half * 2.0)),
        Stroke::new(STROKE_HANDLE, HANDLE_ACCENT),
    );
}
