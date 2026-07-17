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

use egui::{
    Align2, Color32, CornerRadius, FontId, Margin, Pos2, Rect, Sense, Shape, Stroke, StrokeKind,
    UiBuilder, Vec2,
};

use super::{controls, layers, PanelResponse, PanelState, ViewMode};

// --- Signal tokens (docs/design/viewport-chrome-signal.md §Tokens) ---
/// Panel background `#0b0d0f` at ~85 % over the viewport.
const BG: Color32 = Color32::from_rgba_unmultiplied_const(0x0b, 0x0d, 0x0f, 217);
/// Hairline outer border `#2b3238`.
const BORDER: Color32 = Color32::from_rgb(0x2b, 0x32, 0x38);
/// Hairline inner rule / separator `#1c2126`.
const RULE: Color32 = Color32::from_rgb(0x1c, 0x21, 0x26);
/// Header / row hover fill `#12161b`.
const HOVER_BG: Color32 = Color32::from_rgb(0x12, 0x16, 0x1b);
/// Text — primary (row values) `#dfe7ef`.
const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xdf, 0xe7, 0xef);
/// Text — secondary (section header names) `#aeb9c4`.
const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xae, 0xb9, 0xc4);
/// Header name hover `#e3ebf3`.
const TEXT_HOVER: Color32 = Color32::from_rgb(0xe3, 0xeb, 0xf3);
/// Text — muted (row labels, idle tabs) `#78828c`.
const TEXT_MUTED: Color32 = Color32::from_rgb(0x78, 0x82, 0x8c);
/// Text — faint (readouts, counts, chevrons) `#4d565f`.
const TEXT_FAINT: Color32 = Color32::from_rgb(0x4d, 0x56, 0x5f);
/// The single accent — the ADR 0012 onion-haze hue `#9cb4d8`.
const ACCENT: Color32 = Color32::from_rgb(0x9c, 0xb4, 0xd8);

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
/// A folded section tab's height (fits the rotated caption).
const TAB_HEIGHT: f32 = 78.0;
/// The `«` expander tab height.
const EXPANDER_TAB_HEIGHT: f32 = 26.0;

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
pub fn build_signal_stack(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    central_rect: Rect,
    grid_z: u32,
    measured_diameter: u32,
    response: &mut PanelResponse,
) {
    let folded = state.stack.folded;
    let width = stack_width(folded);
    let left = central_rect.right() - STACK_MARGIN - width;
    let top = central_rect.top() + STACK_MARGIN;
    // Generous height budget; the immediate-mode content sizes the painted panel to fit.
    let max_rect = Rect::from_min_size(
        Pos2::new(left, top),
        Vec2::new(width, (central_rect.height() - 2.0 * STACK_MARGIN).max(0.0)),
    );

    root_ui.scope_builder(UiBuilder::new().max_rect(max_rect), |ui| {
        apply_signal_style(ui);
        if folded {
            build_folded_tabs(ui, state);
        } else {
            build_expanded_stack(ui, state, grid_z, measured_diameter, response);
        }
    });
}

/// The Signal scoped [`egui::Style`] override for the stack's widgets: zero corner radius
/// everywhere, dark fills, hairline strokes, the onion-haze accent as the selection fill,
/// tight spacing. Pixel-perfect widget cloning is not required (issue #88) — the tokens,
/// the SHAPES (no rounding) and the layout are.
fn apply_signal_style(ui: &mut egui::Ui) {
    let style = ui.style_mut();
    style.spacing.item_spacing = Vec2::new(6.0, 5.0);
    style.spacing.button_padding = Vec2::new(6.0, 2.0);
    style.spacing.interact_size.y = 18.0;
    let v = &mut style.visuals;
    v.override_text_color = Some(TEXT_PRIMARY);
    // Selection (segmented active cell + slider fill) = accent.
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, Color32::from_rgb(0x0b, 0x0d, 0x0f));
    v.hyperlink_color = ACCENT;
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::ZERO;
    }
    v.widgets.noninteractive.bg_fill = Color32::TRANSPARENT;
    v.widgets.inactive.bg_fill = Color32::from_rgb(0x12, 0x16, 0x1b);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(0x12, 0x16, 0x1b);
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_MUTED);
    v.widgets.hovered.bg_fill = HOVER_BG;
    v.widgets.hovered.weak_bg_fill = HOVER_BG;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_HOVER);
    v.widgets.active.bg_fill = Color32::from_rgb(0x16, 0x1a, 0x1e);
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
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
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(CornerRadius::ZERO)
        .inner_margin(Margin::ZERO)
        .show(ui, |ui| {
            ui.set_width(STACK_WIDTH);

            // --- DISPLAY header bar with the » fold control. ---
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(STACK_WIDTH, HEADER_BAR_HEIGHT), Sense::hover());
            let painter = ui.painter_at(bar_rect);
            let title = letter_spaced(ui, "DISPLAY", TEXT_SECONDARY, 10.5, 2.0);
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
            let fold_glyph = letter_spaced(
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
    let galley = letter_spaced(ui, name, name_color, 10.0, 1.5);
    ui.painter().galley(
        Pos2::new(rect.left() + 22.0, rect.center().y - galley.size().y * 0.5),
        galley,
        name_color,
    );
    // Count (faint, right-aligned).
    let count_galley = letter_spaced(ui, count, TEXT_FAINT, 9.0, 0.0);
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
    if edge_tab(ui, "\u{00ab}", EXPANDER_TAB_HEIGHT, true) {
        state.stack.folded = false;
    }
    if edge_tab(ui, "VIEWPORT", TAB_HEIGHT, false) {
        state.stack.folded = false;
        state.stack.viewport_open = true;
    }
    if state.view_mode == ViewMode::OnionFog && edge_tab(ui, "ONION FOG", TAB_HEIGHT, false) {
        state.stack.folded = false;
        state.stack.onion_open = true;
    }
    if edge_tab(ui, "GRIDS", TAB_HEIGHT, false) {
        state.stack.folded = false;
        state.stack.grids_open = true;
    }
}

/// One vertical edge tab: a hairline-bordered near-black cell with a rotated (top-to-bottom)
/// caption. Idle muted, hover brightens on the hover fill. Returns `true` when clicked.
fn edge_tab(ui: &mut egui::Ui, caption: &str, height: f32, expander: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(TAB_WIDTH, height), Sense::click());
    let hovered = resp.hovered();
    ui.painter().rect_filled(rect, 0.0, if hovered { HOVER_BG } else { BG });
    ui.painter()
        .rect_stroke(rect, 0.0, Stroke::new(1.0, BORDER), StrokeKind::Inside);
    let color = if hovered { TEXT_HOVER } else { TEXT_MUTED };
    let size = if expander { 13.0 } else { 10.0 };
    let spacing = if expander { 0.0 } else { 1.5 };
    let galley = letter_spaced(ui, caption, color, size, spacing);
    // Rotate the pre-laid galley 90° clockwise so the caption reads top-to-bottom,
    // centred in the tab (egui's `Shape::text` can't letter-space, hence the galley).
    let text_shape = egui::epaint::TextShape::new(rect.center(), galley, color)
        .with_angle_and_anchor(std::f32::consts::FRAC_PI_2, Align2::CENTER_CENTER);
    ui.painter().add(text_shape);
    let tip = if expander {
        "Expand display panel".to_string()
    } else {
        format!("Open {caption}")
    };
    resp.on_hover_text(tip).clicked()
}

/// Lay out `text` as UPPERCASE monospace with extra letter spacing, returning the galley
/// for painting (and width/height measurement).
fn letter_spaced(
    ui: &egui::Ui,
    text: &str,
    color: Color32,
    size: f32,
    spacing: f32,
) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        egui::TextFormat {
            font_id: FontId::monospace(size),
            color,
            extra_letter_spacing: spacing,
            ..Default::default()
        },
    );
    ui.painter().layout_job(job)
}

/// Draw a full-width inner-rule hairline at `y`.
fn hairline(ui: &egui::Ui, y: f32) {
    let rect = ui.max_rect();
    ui.painter().line_segment(
        [Pos2::new(rect.left(), y), Pos2::new(rect.left() + STACK_WIDTH, y)],
        Stroke::new(1.0, RULE),
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
