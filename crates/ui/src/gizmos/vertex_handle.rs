//! `vertex_handle` — the load-bearing sketch manipulator: a draggable profile vertex.

use egui::{Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

use super::snap_ticks::snap_ticks;
use super::{HANDLE_ACCENT, HANDLE_FILL, HANDLE_HOVER, STROKE_HANDLE};
use crate::theme::color_palette;

/// A profile vertex handle's state — the four the design sheet demonstrates, plus the
/// destructive-hover state the Delete tool arms (#95).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HandleState {
    /// At rest on the working plane.
    Idle,
    /// The pointer is over it — the border brightens to say "draggable".
    Hover,
    /// Picked / being dragged — the thumb fills accent.
    Selected,
    /// Selected AND engaged with the lattice — the filled thumb, ringed by the snap tick-cross.
    Snapped,
    /// The **Delete** tool is armed and the pointer is over this vertex — the border and an
    /// overlaid `✕` go warn-red to say "clicking removes this one" (ADR 0028, #95). The warn
    /// hue is the destructive channel of the palette, distinct from the accent every other
    /// state uses, so a delete-hover can never be mistaken for a draggable hover.
    Marked,
}

/// Draw a **profile vertex handle**. A square thumb of half-extent `half` (points) centred at
/// `center`: dark fill + accent border idle, bright fill on hover, accent fill when selected,
/// and the snap tick-cross around it when snapped. Distinct from the 3D position axis-handles
/// (those move a whole node; this moves one profile vertex).
pub fn vertex_handle(painter: &Painter, center: Pos2, half: f32, state: HandleState) {
    let (fill, border) = match state {
        HandleState::Idle => (HANDLE_FILL, HANDLE_ACCENT),
        // Hover FILLS with the bright hover colour, and Selected fills accent — the same two
        // colours the hovered / selected lines use, so a point and an edge answer alike (owner
        // 2026-07-23). Idle stays hollow (dark fill, accent border), so the three read distinctly.
        HandleState::Hover => (HANDLE_HOVER, HANDLE_HOVER),
        HandleState::Selected | HandleState::Snapped => (HANDLE_ACCENT, HANDLE_ACCENT),
        // Destructive hover: dark thumb, warn-red border, so it reads as "armed to remove"
        // rather than "armed to drag".
        HandleState::Marked => (HANDLE_FILL, color_palette::WARN),
    };
    let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
    painter.rect_filled(rect, 0.0, fill);
    painter.rect_stroke(rect, 0.0, Stroke::new(STROKE_HANDLE, border), StrokeKind::Inside);
    if state == HandleState::Snapped {
        snap_ticks(painter, center, half + 2.5, half + 7.0, HANDLE_ACCENT);
    }
    // The warn `✕` inside the thumb — the unmistakable "this one goes" mark. Drawn as two
    // strokes across the thumb, inset so they sit clear of the border.
    if state == HandleState::Marked {
        let arm = half - 1.0;
        let cross = Stroke::new(STROKE_HANDLE, color_palette::WARN);
        painter.line_segment(
            [center + Vec2::new(-arm, -arm), center + Vec2::new(arm, arm)],
            cross,
        );
        painter.line_segment(
            [center + Vec2::new(arm, -arm), center + Vec2::new(-arm, arm)],
            cross,
        );
    }
}
