//! The floating **Signal display stack** (issue #88; ADR 0018 Decision 8,
//! `docs/design/viewport-chrome-signal.md` §Chrome layout — the display-panel bullet).
//!
//! A near-black instrument panel floating at the top-right of the 3D viewport (the view
//! cube + icon rail slide to its left). It hosts the DISPLAY stack the display sections
//! left the sidebar for:
//!
//!   * **VIEWPORT** — the viewer-mode readout (accent) + the camera projection toggle.
//!   * **ONION FOG** — the layer scrubber + onion depth + widest-run stat, mounted ONLY in
//!     [`ViewMode::OnionFog`] (the section AND its folded tab vanish in other modes).
//!   * **GRIDS** — the display master toggles (voxel grid on faces, block lattice, floor
//!     grid, view cube, debug faces).
//!
//! It renders on the headless `shot`'s SINGLE frame the same way the side panel does: a
//! fixed-rect immediate-mode egui child ([`Ui::scope_builder`] at an absolute `max_rect`),
//! NOT a floating [`egui::Area`] (which needs a prior frame to settle its size). The
//! section bodies are ordinary egui widgets restyled through a scoped Signal
//! [`egui::Style`] override (dark fills, zero corner radius, the onion-haze accent).
//!
//! Folded, the whole stack collapses to vertical edge tabs (Blender N-panel style):
//! rotated glyphs, one per section plus a `«` expander; clicking a tab expands the stack
//! with that section opened. Folded/open state is [`SignalStackState`] viewer state (never
//! serialized, like [`ViewMode`]). The cube + rail slide with the stack via
//! [`cube_right_inset_points`], fed back to `view_cube_corner` so the anchor tracks the
//! stack's current width.

use egui::{CornerRadius, Margin, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, UiBuilder, Vec2};

use super::{controls, layers, PanelResponse, PanelState, ViewMode};
use crate::signal_theme::{
    self, ACCENT, BG, BORDER, HOVER_BG, RULE, TEXT_FAINT, TEXT_HOVER, TEXT_MUTED, TEXT_SECONDARY,
};

// --- Layout constants (egui points) ---
/// Expanded stack width.
const STACK_WIDTH: f32 = 226.0;
/// Folded edge-tab strip width.
const TAB_WIDTH: f32 = 22.0;
/// Margin from the viewport's top + right edges to the stack.
const STACK_MARGIN: f32 = 12.0;
/// Gap between the cube (to the left) and the stack's left edge.
const CUBE_GAP: f32 = 10.0;
/// The DISPLAY header bar height.
const HEADER_BAR_HEIGHT: f32 = 24.0;
/// A collapsible section header row height.
const SECTION_HEADER_HEIGHT: f32 = 22.0;
/// Vertical padding (points) above + below the rotated caption inside a folded tab
/// (issue #91 item 5): the tab HEIGHT is the rotated galley's width plus 2× this.
const TAB_TEXT_PAD: f32 = 9.0;

/// The stack's current width (points) — expanded vs the folded tab strip.
fn stack_width(folded: bool) -> f32 {
    if folded {
        TAB_WIDTH
    } else {
        STACK_WIDTH
    }
}

/// The horizontal distance (egui points) from the viewport's RIGHT edge to the view cube's
/// right edge, so the cube + rail slide left of the stack and track its fold state. Fed to
/// `view_cube_corner` (converted to pixels) so the drawn cube, its hit-rect and the egui
/// rail all share one anchor (issue #88 — the slide).
pub fn cube_right_inset_points(folded: bool) -> f32 {
    STACK_MARGIN + stack_width(folded) + CUBE_GAP
}

/// Build the floating Signal display stack into `root_ui` (issue #88). `central_rect` is
/// the post-panel 3D viewport rect (egui points); the stack anchors to its top-right
/// corner. Mutates `state` (fold / section-open toggles, projection, layer band, grid
/// masters) and pushes any `SetGridMasters` intent onto `response`.
///
/// Returns the stack's PAINTED rect (egui points) — the shell's chrome hit-rect, so the
/// windowed camera gate can treat pointer input inside it as chrome (the view-cube idiom).
///
/// The stack draws in a NON-ALLOCATING child ui ([`egui::Ui::new_child`]), never
/// `scope_builder`: egui 0.34's scope advances the PARENT cursor past the child
/// (`advance_cursor_after_rect`), and `Context::run_ui` records the root ui's remaining
/// `available_rect_before_wrap` as the "not over egui" region for input. With the scope,
/// the whole FULL-WIDTH band above the stack's bottom edge fell outside that rect, so
/// `wants_pointer_input` reported it as egui's — and the shell's orbit/pan/zoom (all gated
/// on egui consumption) went DEAD across the top of the viewport, growing with the stack
/// (worst in Onion-fog, whose section is tallest). A non-allocating child paints the
/// identical pixels while leaving the root cursor — and thus the input region — untouched.
pub fn build_signal_stack(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    central_rect: Rect,
    grid_z: u32,
    measured_diameter: u32,
    response: &mut PanelResponse,
) -> Rect {
    let folded = state.stack.folded;
    let width = stack_width(folded);
    let left = central_rect.right() - STACK_MARGIN - width;
    let top = central_rect.top() + STACK_MARGIN;
    // Generous height budget; the immediate-mode content sizes the painted panel to fit.
    let max_rect = Rect::from_min_size(
        Pos2::new(left, top),
        Vec2::new(width, (central_rect.height() - 2.0 * STACK_MARGIN).max(0.0)),
    );

    let mut stack_ui = root_ui.new_child(UiBuilder::new().max_rect(max_rect));
    // The stack's scoped Signal style (promoted to `signal_theme`, issue #89): built
    // from `Style::default` so the floating stack stays byte-identical to its #80
    // rendering regardless of the app-wide restyle around it.
    signal_theme::apply_stack_style(&mut stack_ui);
    if folded {
        build_folded_tabs(&mut stack_ui, state);
    } else {
        build_expanded_stack(&mut stack_ui, state, grid_z, measured_diameter, response);
    }
    stack_ui.min_rect()
}

/// The expanded stack: the DISPLAY header bar (with the `»` fold button) then the
/// collapsible sections. The near-black panel background + outer hairline are painted
/// behind the content via an [`egui::Frame`] (the paint-behind idiom that lets the panel
/// wrap the content on the single frame).
fn build_expanded_stack(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    grid_z: u32,
    measured_diameter: u32,
    response: &mut PanelResponse,
) {
    egui::Frame::new()
        .fill(BG)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::ZERO)
        .show(ui, |ui| {
            ui.set_width(STACK_WIDTH);

            // --- DISPLAY header bar with the » fold control. ---
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(STACK_WIDTH, HEADER_BAR_HEIGHT), Sense::hover());
            let painter = ui.painter_at(bar_rect);
            let title = signal_theme::letter_spaced(ui, "DISPLAY", TEXT_SECONDARY, 10.5, 2.0);
            painter.galley(
                Pos2::new(bar_rect.left() + 8.0, bar_rect.center().y - title.size().y * 0.5),
                title,
                TEXT_SECONDARY,
            );
            // The » fold button (right-aligned).
            let fold_rect = Rect::from_min_size(
                Pos2::new(bar_rect.right() - HEADER_BAR_HEIGHT, bar_rect.top()),
                Vec2::splat(HEADER_BAR_HEIGHT),
            );
            let fold_resp = ui.interact(fold_rect, ui.id().with("stack_fold"), Sense::click());
            if fold_resp.hovered() {
                ui.painter().rect_filled(fold_rect, 0.0, HOVER_BG);
            }
            let fold_glyph = signal_theme::letter_spaced(
                ui,
                "\u{00bb}",
                if fold_resp.hovered() { TEXT_HOVER } else { TEXT_MUTED },
                14.0,
                0.0,
            );
            ui.painter().galley(
                fold_rect.center() - fold_glyph.size() * 0.5,
                fold_glyph,
                TEXT_MUTED,
            );
            if fold_resp.on_hover_text("Fold display panel").clicked() {
                state.stack.folded = true;
            }
            hairline(ui, bar_rect.bottom());

            // --- VIEWPORT: the mode readout + camera projection toggle. ---
            if section_header(ui, "VIEWPORT", "1", state.stack.viewport_open) {
                state.stack.viewport_open = !state.stack.viewport_open;
            }
            if state.stack.viewport_open {
                section_body(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("MODE").monospace().size(9.5).color(TEXT_MUTED));
                        ui.label(
                            egui::RichText::new(state.view_mode.status_label())
                                .monospace()
                                .size(10.0)
                                .color(ACCENT),
                        );
                    });
                    controls::build_camera_body(ui, state);
                });
            }

            // --- ONION FOG: mounted only in Onion-fog mode (ADR 0018 Decision 5). ---
            if state.view_mode == ViewMode::OnionFog {
                if section_header(ui, "ONION FOG", "4", state.stack.onion_open) {
                    state.stack.onion_open = !state.stack.onion_open;
                }
                if state.stack.onion_open {
                    section_body(ui, |ui| {
                        layers::build_onion_body(ui, state, grid_z, measured_diameter);
                    });
                }
            }

            // --- GRIDS: the display master toggles. ---
            if section_header(ui, "GRIDS", "5", state.stack.grids_open) {
                state.stack.grids_open = !state.stack.grids_open;
            }
            if state.stack.grids_open {
                section_body(ui, |ui| {
                    controls::build_display_body(ui, state, response);
                });
            }
        });
}

/// A collapsible section header row: a chevron (rotated when closed), the UPPERCASE
/// letter-spaced name (secondary, hover brightens), and the faint right-aligned
/// control-count. Returns `true` when clicked (the caller toggles the section open).
fn section_header(ui: &mut egui::Ui, name: &str, count: &str, open: bool) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(STACK_WIDTH, SECTION_HEADER_HEIGHT), Sense::click());
    let hovered = resp.hovered();
    if hovered {
        ui.painter().rect_filled(rect, 0.0, HOVER_BG);
    }
    // Chevron.
    chevron(ui.painter(), Pos2::new(rect.left() + 11.0, rect.center().y), open);
    // Name.
    let name_color = if hovered { TEXT_HOVER } else { TEXT_SECONDARY };
    let galley = signal_theme::letter_spaced(ui, name, name_color, 10.0, 1.5);
    ui.painter().galley(
        Pos2::new(rect.left() + 22.0, rect.center().y - galley.size().y * 0.5),
        galley,
        name_color,
    );
    // Count (faint, right-aligned).
    let count_galley = signal_theme::letter_spaced(ui, count, TEXT_FAINT, 9.0, 0.0);
    ui.painter().galley(
        Pos2::new(rect.right() - 10.0 - count_galley.size().x, rect.center().y - count_galley.size().y * 0.5),
        count_galley,
        TEXT_FAINT,
    );
    hairline(ui, rect.bottom());
    resp.clicked()
}

/// Lay out a section body indented under its header, with a small top/bottom pad and a
/// closing inner rule.
fn section_body(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .inner_margin(Margin {
            left: 12,
            right: 8,
            top: 5,
            bottom: 6,
        })
        .show(ui, |ui| {
            ui.set_width(STACK_WIDTH - 20.0);
            add(ui);
        });
    hairline(ui, ui.min_rect().bottom());
}

/// The folded edge-tab strip: a `«` expander tab then one rotated tab per visible section
/// (the ONION FOG tab only in Onion-fog mode). Clicking a section tab expands the stack
/// with that section opened; the `«` tab just expands.
fn build_folded_tabs(ui: &mut egui::Ui, state: &mut PanelState) {
    // The « expander.
    if edge_tab(ui, "\u{00ab}", true) {
        state.stack.folded = false;
    }
    if edge_tab(ui, "VIEWPORT", false) {
        state.stack.folded = false;
        state.stack.viewport_open = true;
    }
    if state.view_mode == ViewMode::OnionFog && edge_tab(ui, "ONION FOG", false) {
        state.stack.folded = false;
        state.stack.onion_open = true;
    }
    if edge_tab(ui, "GRIDS", false) {
        state.stack.folded = false;
        state.stack.grids_open = true;
    }
}

/// One vertical edge tab: a hairline-bordered near-black cell with a rotated (top-to-bottom)
/// caption. Idle muted, hover brightens on the hover fill. Returns `true` when clicked.
///
/// Issue #91 item 5: the tab box is sized from the ROTATED galley's bounds (a 90° rotation
/// swaps width/height, so the box is `TAB_WIDTH` wide and `galley_width + 2·pad` tall) and
/// the galley is positioned so it sits CENTRED INSIDE the box — the old
/// `with_angle_and_anchor` placement dropped the caption outside its rectangle.
fn edge_tab(ui: &mut egui::Ui, caption: &str, expander: bool) -> bool {
    let size = if expander { 13.0 } else { 10.0 };
    let spacing = if expander { 0.0 } else { 1.5 };
    // Measure the caption (galley size is colour-independent) to size the tab box.
    let measured = signal_theme::letter_spaced(ui, caption, TEXT_MUTED, size, spacing);
    let galley_width = measured.size().x;
    let galley_height = measured.size().y;
    let height = galley_width + 2.0 * TAB_TEXT_PAD;

    let (rect, resp) = ui.allocate_exact_size(Vec2::new(TAB_WIDTH, height), Sense::click());
    let hovered = resp.hovered();
    ui.painter().rect_filled(rect, 0.0, if hovered { HOVER_BG } else { BG });
    ui.painter()
        .rect_stroke(rect, 0.0, Stroke::new(1.0_f32, BORDER), StrokeKind::Inside);

    let color = if hovered { TEXT_HOVER } else { TEXT_MUTED };
    let galley = signal_theme::letter_spaced(ui, caption, color, size, spacing);
    // Rotate the pre-laid galley 90° clockwise (egui's `Shape::text` can't letter-space,
    // hence the galley). A TextShape draws the galley from `pos` then rotates it about
    // `pos`; for +90° the galley's rotated bbox centre lands at `pos + (h/2, -w/2)`, so we
    // offset `pos` by the inverse to centre the rotated caption exactly in the tab rect.
    let pos = rect.center() + Vec2::new(galley_height * 0.5, -galley_width * 0.5);
    let text_shape = egui::epaint::TextShape::new(pos, galley, color)
        .with_angle(std::f32::consts::FRAC_PI_2);
    ui.painter().add(text_shape);

    let tip = if expander {
        "Expand display panel".to_string()
    } else {
        format!("Open {caption}")
    };
    resp.on_hover_text(tip).clicked()
}

/// Draw a full-width inner-rule hairline at `y`.
fn hairline(ui: &egui::Ui, y: f32) {
    let rect = ui.max_rect();
    ui.painter().line_segment(
        [Pos2::new(rect.left(), y), Pos2::new(rect.left() + STACK_WIDTH, y)],
        Stroke::new(1.0_f32, RULE),
    );
}

/// Draw a small collapse chevron centred at `center`: pointing down when `open`, right when
/// closed (the "rotates when closed" affordance).
fn chevron(painter: &egui::Painter, center: Pos2, open: bool) {
    let points = if open {
        vec![
            center + Vec2::new(-3.5, -2.0),
            center + Vec2::new(3.5, -2.0),
            center + Vec2::new(0.0, 3.0),
        ]
    } else {
        vec![
            center + Vec2::new(-2.0, -3.5),
            center + Vec2::new(3.0, 0.0),
            center + Vec2::new(-2.0, 3.5),
        ]
    };
    painter.add(Shape::convex_polygon(points, TEXT_FAINT, Stroke::NONE));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::{PanelResponse, PanelState};

    /// ADR 0018 #88 input regression: the floating stack must NOT allocate in the root
    /// ui. egui 0.34's `run_ui` records the root's final `available_rect_before_wrap`
    /// as the "not over egui" input region; a stack drawn via `scope_builder` advances
    /// the root cursor past its bottom edge, carving a FULL-WIDTH band above it out of
    /// that region — and the shell's orbit/pan/zoom (all gated on egui pointer
    /// consumption) go dead across the top of the viewport, growing with the stack
    /// (tallest in Onion-fog, where the bug was reported). Pins the non-allocating
    /// `new_child` draw across view modes and fold states.
    #[test]
    fn stack_leaves_root_cursor_untouched() {
        for (view_mode, folded) in [
            (ViewMode::Normal, false),
            (ViewMode::OnionFog, false),
            (ViewMode::ShowBooleans, false),
            (ViewMode::Normal, true),
        ] {
            let context = egui::Context::default();
            let mut state = PanelState {
                view_mode,
                ..PanelState::default()
            };
            state.stack.folded = folded;
            let mut response = PanelResponse::default();
            let raw_input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(
                    Pos2::ZERO,
                    Vec2::new(1280.0, 800.0),
                )),
                ..Default::default()
            };
            let _ = context.run_ui(raw_input, |ui| {
                let central = ui.available_rect_before_wrap();
                let stack_rect =
                    build_signal_stack(ui, &mut state, central, 800, 0, &mut response);
                assert_eq!(
                    central,
                    ui.available_rect_before_wrap(),
                    "the stack must not advance the root cursor \
                     (view_mode={view_mode:?}, folded={folded})"
                );
                // The returned chrome hit-rect is the painted stack: non-empty,
                // anchored at the viewport's top-right margin (± the 1 px frame stroke).
                assert!(stack_rect.width() > 0.0 && stack_rect.height() > 0.0);
                assert!((stack_rect.right() - (central.right() - STACK_MARGIN)).abs() <= 2.5);
                assert!((stack_rect.top() - (central.top() + STACK_MARGIN)).abs() <= 2.5);
            });
        }
    }
}
