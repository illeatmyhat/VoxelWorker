//! The Signal icon rail under the view cube: Home / Fit / viewport-mode-cycle / orbit type.

use camera::OrbitType;
use egui::{Color32, Id, LayerId, Order, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::icons::Icon;
use crate::panel::{OrbitMode, ViewMode};
use crate::theme;

const RAIL_WIDTH: f32 = 42.0;
const BUTTON_HEIGHT: f32 = 40.0;
const RAIL_GAP: f32 = 6.0;
/// The rail's glyph box. The marks are authored on an 18-unit grid and this is the size that grid
/// is drawn AT — the answer to "does this mark read?" is always about this number, so it is the one
/// the `design_reference` sheet judges against too.
const GLYPH_BOX: f32 = 24.0;
/// Height of the orbit-type button's dropdown half, which sits BELOW the face rather than beside
/// it. Taking the strip off the side left the mark an off-centre 23 pt of a 34 pt rail to live in,
/// and every legibility problem the pair had was really that: a glyph authored for a square box,
/// judged in a letterbox. The button grows by this much instead, so the face stays a full,
/// centred, rail-width square and the caret costs the mark nothing.
const CARET_HEIGHT: f32 = 13.0;
/// The caret's own glyph box — smaller than [`GLYPH_BOX`], because a chevron is a pointer at the
/// menu and must not read as a second subject beside the face's mark.
const CARET_BOX: f32 = 11.0;
/// The rail index of the orbit-type split button.
const ORBIT_TYPE_BUTTON: usize = 3;

/// A rail button the user clicked this frame — the shell maps Home / Fit onto the same camera
/// actions the retired cube badges dispatched, and CycleMode onto the next viewport mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailClick {
    Home,
    Fit,
    CycleMode,
    /// The orbit-type split button's FACE half — TOGGLE the explicit orbit mode, turning as the
    /// type the face is showing.
    ///
    /// It neither names a type nor writes the default (`docs/design/tool-modes-and-navigation.md`,
    /// the entry-path table): entering from a split button's face means "start an orbit as what
    /// this says", and what it says is already the default. The viewport menu's Constrained Orbit
    /// is the entry that names one.
    OrbitType,
    /// The orbit-type split button's CARET half — opens the type menu, which is the ONE place the
    /// DEFAULT orbit type is written. Every other entry into an orbit either uses the default or
    /// overrides it for the session without changing it.
    OrbitTypeMenu,
}

/// The number of rail buttons, in the order [`icon_rail`] draws and dispatches them.
const RAIL_BUTTONS: usize = 4;

/// The full rail height, used to place the readout below the rail. The orbit-type button carries
/// its caret strip on top of the common button height.
pub fn rail_height() -> f32 {
    RAIL_BUTTONS as f32 * BUTTON_HEIGHT + CARET_HEIGHT
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
        Pos2::new(
            rail.left(),
            rail.top() + ORBIT_TYPE_BUTTON as f32 * BUTTON_HEIGHT,
        ),
        Vec2::new(RAIL_WIDTH, BUTTON_HEIGHT + CARET_HEIGHT),
    )
}

/// Split a rail button into a split button's two halves: `(face, caret)`, the caret UNDER the face.
///
/// They are separate hit targets and light separately, which is the affordance — a control that
/// hovers as one block is a button, and a control that hovers in two pieces is a split button. A
/// hairline between them says the same thing while nothing is hovered.
fn split_halves(button_rect: Rect) -> (Rect, Rect) {
    let divider_y = button_rect.bottom() - CARET_HEIGHT;
    (
        Rect::from_min_max(
            button_rect.left_top(),
            Pos2::new(button_rect.right(), divider_y),
        ),
        Rect::from_min_max(
            Pos2::new(button_rect.left(), divider_y),
            button_rect.right_bottom(),
        ),
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
    default_orbit_type: OrbitType,
    orbit_mode: OrbitMode,
) -> Option<RailClick> {
    // The face shows the ACTIVE type, not the default: while a naming command's override runs,
    // a face showing the default would be naming a type nothing is about to use.
    let active_orbit_type = orbit_mode.active_type(default_orbit_type);
    let rail_rect = rail_rect(cube_left, cube_bottom, cube_size);
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("signal_icon_rail")));
    painter.rect_filled(rail_rect, 0.0, theme::BG);

    let mut click = None;
    for index in 0..RAIL_BUTTONS {
        // The orbit-type button is a SPLIT button: a face with a caret strip under it. Every other
        // button is one target the common height, so its caret half is empty and never hovers.
        let split = index == ORBIT_TYPE_BUTTON;
        let button_rect = Rect::from_min_size(
            Pos2::new(
                rail_rect.left(),
                rail_rect.top() + index as f32 * BUTTON_HEIGHT,
            ),
            Vec2::new(
                RAIL_WIDTH,
                BUTTON_HEIGHT + if split { CARET_HEIGHT } else { 0.0 },
            ),
        );
        let (face_rect, caret_rect) = if split {
            split_halves(button_rect)
        } else {
            (button_rect, Rect::NOTHING)
        };
        let response = ui.interact(
            face_rect,
            Id::new(("signal_rail_button", index)),
            Sense::click(),
        );
        let caret_response = split.then(|| {
            ui.interact(
                caret_rect,
                Id::new(("signal_rail_caret", index)),
                Sense::click(),
            )
        });
        let caret_hovered = caret_response.as_ref().is_some_and(|it| it.hovered());
        let hovered = response.hovered();
        // A button lights when it is holding a NON-default state, so the rail reads as "nothing
        // unusual is set" at a glance: the viewport-mode button off Normal, the orbit-type button
        // while the orbit mode is running or the default is Free. The mode counts because a mode
        // that flips the left button's verb is the least ordinary state the rail can be in.
        let lit = (index == 2 && view_mode != ViewMode::Normal)
            || (index == ORBIT_TYPE_BUTTON
                && (orbit_mode.is_on() || active_orbit_type == OrbitType::Free));

        if lit {
            painter.rect_filled(button_rect, 0.0, theme::HOVER_BG);
        }
        // Each half lights on its own — that separation IS the split-button affordance.
        if hovered {
            painter.rect_filled(face_rect, 0.0, theme::ACTIVE_BG);
        }
        if caret_hovered {
            painter.rect_filled(caret_rect, 0.0, theme::ACTIVE_BG);
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
            let bar =
                Rect::from_min_size(button_rect.left_top(), Vec2::new(2.0, button_rect.height()));
            painter.rect_filled(bar, 0.0, theme::ACCENT);
        }

        let tone = |hovered: bool| {
            if lit {
                theme::ACCENT
            } else if hovered {
                theme::HANDLE_HOVER
            } else {
                theme::TEXT_MUTED
            }
        };
        draw_glyph(
            &painter,
            face_rect,
            index,
            view_mode,
            active_orbit_type,
            tone(hovered),
        );
        if split {
            // The divider, then the chevron: the two marks that say "this opens something".
            painter.line_segment(
                [caret_rect.left_top(), caret_rect.right_top()],
                Stroke::new(1.0_f32, theme::RULE),
            );
            Icon::ChevronDown.draw(
                &painter,
                Rect::from_center_size(caret_rect.center(), Vec2::splat(CARET_BOX)),
                tone(caret_hovered),
            );
        }

        let response = response.on_hover_text(match index {
            0 => "Home view".to_string(),
            1 => "Fit scene".to_string(),
            2 => "Viewport mode".to_string(),
            _ => orbit_face_tooltip(default_orbit_type, orbit_mode),
        });
        if response.clicked() {
            click = Some(match index {
                0 => RailClick::Home,
                1 => RailClick::Fit,
                2 => RailClick::CycleMode,
                _ => RailClick::OrbitType,
            });
        }
        if let Some(caret_response) = caret_response {
            if caret_response.on_hover_text("Choose orbit type").clicked() {
                click = Some(RailClick::OrbitTypeMenu);
            }
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

/// The name of an orbit type, as the chrome says it.
fn orbit_type_name(orbit_type: OrbitType) -> &'static str {
    match orbit_type {
        OrbitType::Constrained => "Constrained",
        OrbitType::Free => "Free",
    }
}

/// The face's tooltip — the one place the rail can say what the glyph cannot.
///
/// The glyph names the ACTIVE type, which is all a face can show. When an override is running,
/// that face is indistinguishable from the same type being the default, so the words carry the
/// difference: which type is running, that it lasts only for this mode, and what the default will
/// go back to being. Dropping the parenthetical when the two agree keeps it from reading as a
/// warning about nothing.
fn orbit_face_tooltip(default_orbit_type: OrbitType, orbit_mode: OrbitMode) -> String {
    let active = orbit_mode.active_type(default_orbit_type);
    let name = orbit_type_name(active);
    match orbit_mode {
        OrbitMode::Off => format!("Start a {} orbit", name.to_lowercase()),
        OrbitMode::UsingDefault => format!("Leave {} orbit", name.to_lowercase()),
        OrbitMode::Named(_) if active == default_orbit_type => {
            format!("Leave {} orbit", name.to_lowercase())
        }
        OrbitMode::Named(_) => format!(
            "Leave {} orbit — this mode only; the default stays {}",
            name.to_lowercase(),
            orbit_type_name(default_orbit_type).to_lowercase()
        ),
    }
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
    orbit_type: OrbitType,
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
        // The face NAMES the type. A split button's face is what it will do, so a rail showing one
        // orbit mark for both types would be showing the noun and hiding the answer.
        _ => match orbit_type {
            OrbitType::Constrained => Icon::OrbitConstrained,
            OrbitType::Free => Icon::OrbitFree,
        },
    };
    icon.draw(painter, glyph_box(button_rect), color);
}
