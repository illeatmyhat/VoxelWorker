//! `segment` / `dashed_segment` / `marked_segment` — a profile edge: committed (solid), closing
//! (dashed), or armed for deletion (warn-red with a `✕`).

use egui::{Painter, Pos2, Stroke, Vec2};

use super::{dashed, HANDLE_ACCENT, STROKE_HANDLE, STROKE_SEGMENT};
use crate::signal_theme as tokens;

/// Half-length (points) of the arms of the warn `✕` stamped on a [`marked_segment`] — sized to
/// the vertex handle's own cross so a segment delete-hover and a vertex one read at one scale.
const MARK_CROSS_ARM: f32 = 4.0;

/// A **committed profile segment** — a real edge between two placed vertices. Solid accent; it is
/// an entity, not a preview.
pub fn segment(painter: &Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}

/// A profile segment **armed for deletion** — the Delete tool is hovering this edge (and no
/// vertex, which would take priority). The whole line goes warn-red with a warn `✕` at its
/// midpoint: the line analogue of the vertex handle's [`Marked`](super::HandleState::Marked)
/// state, so a segment delete-hover carries the same destructive vocabulary as a vertex one
/// (colour the line, not just an overlay — the Fusion-style "this edge goes" cue in our warn hue).
pub fn marked_segment(painter: &Painter, a: Pos2, b: Pos2) {
    painter.line_segment([a, b], Stroke::new(STROKE_SEGMENT, tokens::WARN));
    let mid = a + (b - a) * 0.5;
    let cross = Stroke::new(STROKE_HANDLE, tokens::WARN);
    painter.line_segment(
        [mid + Vec2::splat(-MARK_CROSS_ARM), mid + Vec2::splat(MARK_CROSS_ARM)],
        cross,
    );
    painter.line_segment(
        [
            mid + Vec2::new(MARK_CROSS_ARM, -MARK_CROSS_ARM),
            mid + Vec2::new(-MARK_CROSS_ARM, MARK_CROSS_ARM),
        ],
        cross,
    );
}

/// A **dashed closing run** — the uncommitted segment back to the start vertex, in the family's
/// dashed-means-uncommitted idiom. Becomes a solid [`segment`] once the click commits the loop.
pub fn dashed_segment(painter: &Painter, a: Pos2, b: Pos2) {
    dashed(painter, a, b, Stroke::new(STROKE_SEGMENT, HANDLE_ACCENT));
}
