//! `vertex_handle` — the load-bearing sketch manipulator: a draggable profile vertex.

use egui::{Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

use super::snap_ticks::snap_ticks;
use super::{HANDLE_ACCENT, HANDLE_FILL, HANDLE_HOVER, STROKE_HANDLE};
use crate::theme::color_palette;

/// A profile vertex handle's state — the four resting/pointer states, plus the destructive-hover
/// state the Delete tool arms.
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
    /// overlaid `✕` go warn-red to say "clicking removes this one". The warn
    /// hue is the destructive channel of the palette, distinct from the accent every other
    /// state uses, so a delete-hover can never be mistaken for a draggable hover.
    Marked,
}

/// The grabbable end of a tangent handle's lever.
///
/// Green, where the lever is teal: the thing you can take hold of has to be separable at a glance
/// from the line it rides, and every other split this chrome draws (fill versus stroke, hollow
/// versus filled) is already spent saying idle / hover / selected. Filled in every state, because
/// an arm is never "unselected" in the way a profile point is — it is furniture, and it is always
/// live.
pub fn tangent_arm_handle(painter: &Painter, center: Pos2, half: f32, state: HandleState) {
    let ink = match state {
        HandleState::Hover | HandleState::Snapped => HANDLE_HOVER,
        _ => color_palette::SKETCH_TANGENT_POINT,
    };
    let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
    painter.rect_filled(rect, 0.0, ink);
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(STROKE_HANDLE, ink),
        StrokeKind::Inside,
    );
}

/// Draw a **profile vertex handle**. A square thumb of half-extent `half` (points) centered at
/// `center`: dark fill + accent border idle, bright fill on hover, accent fill when selected,
/// and the snap tick-cross around it when snapped. Distinct from the 3D position axis-handles
/// (those move a whole node; this moves one profile vertex).
///
/// `on_ink` says whether a curve is drawn THROUGH this point. Only the idle border reads it, and
/// only downward: a point with no ink through it is scaffolding — a center, a control point — and
/// steps back to [`color_palette::SKETCH_POINT_OFF_INK`] so the shape stays primary over the things
/// that shape it. Hover, selection and the delete-mark are pointer states and answer identically
/// for every point, because the pointer does not care what the dot is for.
pub fn vertex_handle(painter: &Painter, center: Pos2, half: f32, state: HandleState, on_ink: bool) {
    let (fill, border) = match state {
        HandleState::Idle if !on_ink => (HANDLE_FILL, color_palette::SKETCH_POINT_OFF_INK),
        HandleState::Idle => (HANDLE_FILL, HANDLE_ACCENT),
        // Hover FILLS with the bright hover color, and Selected fills accent — the same two
        // colors the hovered / selected lines use, so a point and an edge answer alike. Idle
        // stays hollow (dark fill, accent border), so the three read distinctly.
        HandleState::Hover => (HANDLE_HOVER, HANDLE_HOVER),
        HandleState::Selected | HandleState::Snapped => (HANDLE_ACCENT, HANDLE_ACCENT),
        // Destructive hover: dark thumb, warn-red border, so it reads as "armed to remove"
        // rather than "armed to drag".
        HandleState::Marked => (HANDLE_FILL, color_palette::WARN),
    };
    let rect = Rect::from_center_size(center, Vec2::splat(half * 2.0));
    painter.rect_filled(rect, 0.0, fill);
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(STROKE_HANDLE, border),
        StrokeKind::Inside,
    );
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

#[cfg(test)]
mod tests {
    use super::{vertex_handle, HandleState};
    use egui::{pos2, Color32, Context, RawInput, Rect, Shape, Vec2};

    /// Every stroke color the handle paints, in paint order.
    fn borders(state: HandleState, on_ink: bool) -> Vec<Color32> {
        Context::default()
            .run_ui(
                RawInput {
                    screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, Vec2::splat(32.0))),
                    ..Default::default()
                },
                |ui| vertex_handle(ui.painter(), pos2(16.0, 16.0), 3.5, state, on_ink),
            )
            .shapes
            .into_iter()
            .filter_map(|clipped| match clipped.shape {
                Shape::Rect(rect) if rect.stroke.width > 0.0 => Some(rect.stroke.color),
                _ => None,
            })
            .collect()
    }

    /// The whole claim: at rest, a point with no curve through it recedes.
    #[test]
    fn an_idle_dot_off_the_ink_draws_quieter_than_one_on_it() {
        let on = borders(HandleState::Idle, true);
        let off = borders(HandleState::Idle, false);
        assert_eq!(on.len(), 1, "one bordered thumb");
        assert_ne!(on, off, "the off-ink border is a step back from the accent");
        assert!(
            off[0].r() < on[0].r() && off[0].g() < on[0].g() && off[0].b() < on[0].b(),
            "quieter is DARKER on every channel — a value step, not a second hue"
        );
    }

    /// A pointer state answers the same for every dot: hover means draggable, and what the dot is
    /// for has nothing to do with whether the pointer is on it.
    #[test]
    fn the_pointer_states_do_not_read_the_ink() {
        for state in [
            HandleState::Hover,
            HandleState::Selected,
            HandleState::Snapped,
            HandleState::Marked,
        ] {
            assert_eq!(
                borders(state, true),
                borders(state, false),
                "{state:?} is about the pointer, not about the drawing"
            );
        }
    }
}
