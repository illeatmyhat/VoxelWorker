//! `label_chip` — the one-token readout naming why a vertex locked.

use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

use crate::signal_theme as tokens;

use super::STROKE_GUIDE;

/// A **label chip** — a bordered dark box with an uppercase mono token, its top-left at `at`.
/// Names the quantum a grid snap locked to (`VOXEL` / `BLOCK`), the vertex a coincidence caught
/// (`VERTEX n2`), or the axis a guide follows (`X-AXIS`) — in `accent`, so an axis chip reads in
/// the axis hue. This IS the constraint vocabulary made legible (ADR 0028 §5): the snap says what
/// it caught, the by-product a solver would have named a constraint. Returns the chip's rect so a
/// caller can stack chips.
pub fn label_chip(painter: &Painter, at: Pos2, text: &str, accent: Color32) -> Rect {
    let galley = painter.layout_no_wrap(text.to_uppercase(), FontId::monospace(8.5), accent);
    let pad = Vec2::new(6.0, 3.5);
    let rect = Rect::from_min_size(at, galley.size() + pad * 2.0);
    painter.rect_filled(rect, 0.0, tokens::BG);
    painter.rect_stroke(rect, 0.0, Stroke::new(STROKE_GUIDE, accent), StrokeKind::Inside);
    painter.galley(at + pad, galley, accent);
    rect
}
