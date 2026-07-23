//! `segment` / `dashed_segment` — a profile edge, committed (solid) or closing (dashed).

use egui::{Painter, Pos2, Stroke};

use super::{dashed, HANDLE_ACCENT, STROKE_SEGMENT};

/// A **committed profile segment** — a real edge between two placed vertices. Solid accent; it is
/// an entity, not a preview.
pub fn segment(painter: &Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}

/// A **dashed closing run** — the uncommitted segment back to the start vertex, in the family's
/// dashed-means-uncommitted idiom. Becomes a solid [`segment`] once the click commits the loop.
pub fn dashed_segment(painter: &Painter, a: Pos2, b: Pos2) {
    dashed(painter, a, b, Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}
