//! `vertex_handle` — the load-bearing sketch manipulator: a draggable profile vertex.

use egui::{Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

use super::snap_ticks::snap_ticks;
use super::{HANDLE_ACCENT, HANDLE_FILL, HANDLE_HOVER, STROKE_HANDLE};

/// A profile vertex handle's state — the four the design sheet demonstrates.
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
}

/// Draw a **profile vertex handle**. A square thumb of half-extent `half` (points) centred at
/// `center`: dark fill + accent border idle, brighter border on hover, accent fill when selected,
/// and the snap tick-cross around it when snapped. Distinct from the 3D position axis-handles
/// (those move a whole node; this moves one profile vertex).
pub fn vertex_handle(painter: &Painter, center: Pos2, half: f32, state: HandleState) {
    let (fill, border) = match state {
        HandleState::Idle => (HANDLE_FILL, HANDLE_ACCENT),
        HandleState::Hover => (HANDLE_FILL, HANDLE_HOVER),
        HandleState::Selected | HandleState::Snapped => (HANDLE_ACCENT, HANDLE_ACCENT),
    };
    let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
    painter.rect_filled(rect, 0.0, fill);
    painter.rect_stroke(rect, 0.0, Stroke::new(STROKE_HANDLE, border), StrokeKind::Inside);
    if state == HandleState::Snapped {
        snap_ticks(painter, center, half + 2.5, half + 7.0, HANDLE_ACCENT);
    }
}
