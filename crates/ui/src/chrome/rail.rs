//! The Signal icon rail under the view cube: Home / Fit / viewport-mode-cycle / orbit type.

use camera::OrbitType;
use egui::{Color32, Id, LayerId, Order, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::icons::Icon;
use crate::panel::ViewMode;
use crate::theme;

const RAIL_WIDTH: f32 = 34.0;
const BUTTON_HEIGHT: f32 = 32.0;
const RAIL_GAP: f32 = 6.0;
const GLYPH_BOX: f32 = 18.0;

/// A rail button the user clicked this frame — the shell maps Home / Fit onto the same camera
/// actions the retired cube badges dispatched, and CycleMode onto the next viewport mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailClick {
    Home,
    Fit,
    CycleMode,
    /// The orbit-type button: opens the type menu rather than acting. Fusion's rail carries the
    /// same control, and it is the ONE place the DEFAULT orbit type is written — every other
    /// entry into an orbit either uses the default or overrides it for the session without
    /// changing it (`docs/design/tool-modes-and-navigation.md`, the entry-path table).
    OrbitType,
}

/// The number of rail buttons, in the order [`icon_rail`] draws and dispatches them.
const RAIL_BUTTONS: usize = 4;

/// The full rail height, used to place the readout below the rail.
pub fn rail_height() -> f32 {
    RAIL_BUTTONS as f32 * BUTTON_HEIGHT
}

/// The rail's top Y (points) given the cube's bottom edge.
pub fn rail_top(cube_bottom: f32) -> f32 {
    cube_bottom + RAIL_GAP
}

/// The rail's full rect (egui points) from the cube anchors — the shell's chrome hit-rect for the
/// camera gate, the same geometry [`icon_rail`] draws at.
pub fn rail_rect(cube_left: f32, cube_bottom: f32, cube_size: f32) -> Rect {
    let rail_left = cube_left + (cube_size - RAIL_WIDTH) * 0.5;
    Rect::from_min_size(
        Pos2::new(rail_left, rail_top(cube_bottom)),
        Vec2::new(RAIL_WIDTH, rail_height()),
    )
}

/// The orbit-type button's own rect (egui points). The type menu anchors to it, so the menu and
/// the button it belongs to cannot drift apart as the rail moves with the cube.
pub fn orbit_type_button_rect(cube_left: f32, cube_bottom: f32, cube_size: f32) -> Rect {
    let rail = rail_rect(cube_left, cube_bottom, cube_size);
    Rect::from_min_size(
        Pos2::new(rail.left(), rail.top() + 3.0 * BUTTON_HEIGHT),
        Vec2::new(RAIL_WIDTH, BUTTON_HEIGHT),
    )
}

/// Draw the icon rail centred under the view cube and return a click, if any. `cube_left` /
/// `cube_bottom` / `cube_size` are the cube's screen anchors (egui points). Painted through a
/// foreground `layer_painter` at absolute coordinates (not an `egui::Area`) so it renders on the
/// headless `shot`'s single frame; interaction is `Ui::interact` on the same rects.
pub fn icon_rail(
    ui: &egui::Ui,
    cube_left: f32,
    cube_bottom: f32,
    cube_size: f32,
    view_mode: ViewMode,
    orbit_type: OrbitType,
) -> Option<RailClick> {
    let rail_rect = rail_rect(cube_left, cube_bottom, cube_size);
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("signal_icon_rail")));
    painter.rect_filled(rail_rect, 0.0, theme::BG);

    let mut click = None;
    for index in 0..RAIL_BUTTONS {
        let button_rect = Rect::from_min_size(
            Pos2::new(
                rail_rect.left(),
                rail_rect.top() + index as f32 * BUTTON_HEIGHT,
            ),
            Vec2::new(RAIL_WIDTH, BUTTON_HEIGHT),
        );
        let response = ui.interact(
            button_rect,
            Id::new(("signal_rail_button", index)),
            Sense::click(),
        );
        let hovered = response.hovered();
        // A button lights when it is holding a NON-default state, so the rail reads as "nothing
        // unusual is set" at a glance: the viewport-mode button off Normal, the orbit-type button
        // on Free.
        let lit = (index == 2 && view_mode != ViewMode::Normal)
            || (index == 3 && orbit_type == OrbitType::Free);

        if hovered {
            painter.rect_filled(button_rect, 0.0, theme::ACTIVE_BG);
        } else if lit {
            painter.rect_filled(button_rect, 0.0, theme::HOVER_BG);
        }
        if index > 0 {
            painter.line_segment(
                [
                    Pos2::new(rail_rect.left(), button_rect.top()),
                    Pos2::new(rail_rect.right(), button_rect.top()),
                ],
                Stroke::new(1.0_f32, theme::RULE),
            );
        }
        // Lit mode: a 2 px accent inset bar on the leading edge.
        if lit {
            let bar = Rect::from_min_size(button_rect.left_top(), Vec2::new(2.0, BUTTON_HEIGHT));
            painter.rect_filled(bar, 0.0, theme::ACCENT);
        }

        let glyph_color = if lit {
            theme::ACCENT
        } else if hovered {
            theme::HANDLE_HOVER
        } else {
            theme::TEXT_MUTED
        };
        draw_glyph(&painter, button_rect, index, view_mode, glyph_color);

        let response = response.on_hover_text(match index {
            0 => "Home view",
            1 => "Fit scene",
            2 => "Viewport mode",
            _ => match orbit_type {
                OrbitType::Constrained => "Orbit type: constrained",
                OrbitType::Free => "Orbit type: free",
            },
        });
        if response.clicked() {
            click = Some(match index {
                0 => RailClick::Home,
                1 => RailClick::Fit,
                2 => RailClick::CycleMode,
                _ => RailClick::OrbitType,
            });
        }
    }

    painter.rect_stroke(
        rail_rect,
        0.0,
        Stroke::new(1.0_f32, theme::BORDER),
        StrokeKind::Inside,
    );
    click
}

/// A centred square glyph box inside a rail button — the rail set is authored on a square 18-unit
/// grid, so a square box keeps `IconPainter`'s scale at 1 and the stroke on the design's 1.25 pt.
fn glyph_box(button_rect: Rect) -> Rect {
    Rect::from_center_size(button_rect.center(), Vec2::splat(GLYPH_BOX))
}

/// Draw the glyph for rail button `index` (0 Home, 1 Fit, 2 viewport-mode, 3 orbit-type) in
/// `color`. The marks
/// come from [`crate::icons`], the one authoring the `design_reference` gallery also paints.
fn draw_glyph(
    painter: &egui::Painter,
    button_rect: Rect,
    index: usize,
    view_mode: ViewMode,
    color: Color32,
) {
    let icon = match index {
        0 => Icon::Home,
        1 => Icon::Fit,
        2 => match view_mode {
            ViewMode::Normal => Icon::ModeNormal,
            ViewMode::OnionFog => Icon::ModeOnion,
            ViewMode::ShowBooleans => Icon::ModeBooleans,
        },
        // One glyph for both types: the button always means "orbit", and WHICH type is on is
        // carried by the lit state and the tooltip rather than by a second mark.
        _ => Icon::Orbit,
    };
    icon.draw(painter, glyph_box(button_rect), color);
}
