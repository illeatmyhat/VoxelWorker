//! The **Signal** design language as one shared egui theme (issue #89; ADR 0018,
//! `docs/design/viewport-chrome-signal.md`).
//!
//! Epic #80 dressed only the viewport chrome (view cube + icon rail + status line) and
//! the floating DISPLAY stack in Signal; the rest of the app kept egui's default look.
//! This module promotes that language to ONE source of truth — the token table, the
//! app-wide [`egui::Style`], and the reusable section-header painting helpers — so the
//! whole app (right sidebar, bottom palette dock, and the DISPLAY stack) wears a single
//! near-black instrument-panel skin.
//!
//! Two entry points feed the two surfaces:
//!
//!   * [`apply_app_style`] mutates the egui context's [`Style`] once per frame (via
//!     [`egui::Context::all_styles_mut`] in the shell's `run_egui_frame`), so every
//!     sidebar + palette-dock widget inherits the Signal visuals: zero corner radius,
//!     `#0b0d0f` panel fills, hairline strokes, monospace type, and the ONE accent
//!     (`#9cb4d8`) as the selection fill — the legacy blue dies.
//!   * [`apply_stack_style`] REPLACES a scoped [`Ui`](egui::Ui)'s style with the DISPLAY
//!     stack's tighter variant (`panel::signal_stack`). It builds from
//!     [`Style::default`](egui::Style::default) rather than inheriting the app style, so
//!     the floating stack stays byte-identical to its epic-#80 rendering regardless of
//!     the app-wide restyle around it.
//!
//! Nothing keeps a private token table: `signal_chrome` (the shell's cube/rail/status
//! painters) is the one exception the extraction map allows, because it lives above this
//! crate and paints with explicit `FontId`s + colours (never widget visuals), so it is
//! immune to — and independent of — the app [`Style`].

use egui::{
    Color32, CornerRadius, FontFamily, FontId, Stroke, Style, TextFormat, TextStyle, Visuals,
};
use std::sync::Arc;

// --- Tokens (docs/design/viewport-chrome-signal.md §Tokens) ---
/// Panel background `#0b0d0f`, OPAQUE — the sidebar + palette-dock fills.
pub const BG: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x0f);
/// Panel background `#0b0d0f` at ~85 % — the floating DISPLAY stack over the viewport.
pub const BG_FLOAT: Color32 = Color32::from_rgba_unmultiplied_const(0x0b, 0x0d, 0x0f, 217);
/// Hairline outer border `#2b3238` (bordered cells, panel edges).
pub const BORDER: Color32 = Color32::from_rgb(0x2b, 0x32, 0x38);
/// Hairline inner rule / separator `#1c2126`.
pub const RULE: Color32 = Color32::from_rgb(0x1c, 0x21, 0x26);
/// Row / header hover fill `#12161b`.
pub const HOVER_BG: Color32 = Color32::from_rgb(0x12, 0x16, 0x1b);
/// Active (pressed / open) fill `#161a1e`.
pub const ACTIVE_BG: Color32 = Color32::from_rgb(0x16, 0x1a, 0x1e);
/// Text — primary (values, live labels) `#dfe7ef`.
pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(0xdf, 0xe7, 0xef);
/// Text — secondary (section header names) `#aeb9c4`.
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(0xae, 0xb9, 0xc4);
/// Header name hover `#e3ebf3`.
pub const TEXT_HOVER: Color32 = Color32::from_rgb(0xe3, 0xeb, 0xf3);
/// Text — muted (idle rows, labels, idle tabs) `#78828c`.
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x78, 0x82, 0x8c);
/// Text — faint (readouts, counts, chevrons, subtitle) `#4d565f`.
pub const TEXT_FAINT: Color32 = Color32::from_rgb(0x4d, 0x56, 0x5f);
/// Text — hint `#3c444c` (the faintest tier).
pub const TEXT_HINT: Color32 = Color32::from_rgb(0x3c, 0x44, 0x4c);
/// The single accent — the ADR 0012 onion-haze hue `#9cb4d8`.
pub const ACCENT: Color32 = Color32::from_rgb(0x9c, 0xb4, 0xd8);
/// Text/glyphs painted ON an accent fill `#0b0d0f` (the near-black, for contrast).
pub const ACCENT_TEXT: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x0f);
/// Warn / subtract red `#d9603f` (the app's warn colour). Doubles as the X spatial axis.
pub const WARN: Color32 = Color32::from_rgb(0xd9, 0x60, 0x3f);

// --- Spatial axes + sketch mode (ADR 0028) ---
// The axis triad the sketch snap guides speak in — X = WARN, Y = AXIS_Y, Z = ACCENT — and the
// working plane's own pale-yellow palette. Axis hues are SPATIAL ONLY: a viewport spends them
// on direction and nothing else, so a snap guide's colour IS the constraint it stands in for
// (ADR 0028 §5; docs/design/colour-vocabulary.md).
/// Y spatial axis — green `#7dba6a` (the only axis hue that is not already a token; X reuses
/// [`WARN`], Z reuses [`ACCENT`]).
pub const AXIS_Y: Color32 = Color32::from_rgb(0x7d, 0xba, 0x6a);
/// The sketch working-plane fill — the accent at a low alpha (owner ruling 2026-07-22: the
/// accent-blue tint of the C2 mock, not a warm plane). Faint enough that the profile drawn on
/// the plane stays primary and is never occluded.
pub const SKETCH_PLANE_FILL: Color32 = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x0f);
/// The sketch working-plane's fine grid lines — accent at a legible-but-quiet weight.
pub const SKETCH_PLANE_GRID: Color32 = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x24);
/// The sketch working-plane's BLOCK grid lines — brighter, so the block lattice reads through the
/// fine grid (every `voxels_per_block` steps).
pub const SKETCH_PLANE_GRID_BLOCK: Color32 = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x55);

// --- Typography sizes (design points; §Tokens: monospace, 10–11 px) ---
/// Body / control text (~10.5 px). Sized so a full blocks+voxels readout
/// (`"10 blocks 0 voxels"`) fits the inspector's value boxes without truncating (issue
/// #90) while staying inside the §Tokens 10–11 px band.
const BODY_SIZE: f32 = 10.5;
/// Small hints / readouts (~9.5 px).
const SMALL_SIZE: f32 = 9.5;
/// The DISPLAY stack's body text (mode readout, projection toggle, grid checkboxes) — mono
/// at the same body tier as the sidebar so the floating stack reads as one instrument panel
/// with it (issue #90; the stack previously fell back to egui's ~14 px proportional Body).
const STACK_BODY_SIZE: f32 = 10.0;
/// The sidebar title block heading.
const HEADING_SIZE: f32 = 15.0;
/// A section header caption (UPPERCASE, letter-spaced).
const SECTION_HEADER_SIZE: f32 = 10.0;
/// Extra letter spacing on section-header captions.
const SECTION_HEADER_SPACING: f32 = 1.5;

/// The Signal WIDGET visuals shared by both the app-wide style and the DISPLAY stack's
/// scoped style. This pins EVERY knob of egui's five-state widget matrix
/// (`noninteractive`/`inactive`/`hovered`/`active`/`open` × `bg_fill`/`weak_bg_fill`/
/// `bg_stroke`/`fg_stroke`/`corner_radius`/`expansion`) so NOTHING falls back to egui's
/// bright grey-white defaults (issue #90 — an unset `open.bg_stroke` or `active.weak_bg_fill`
/// leaks a `gray(60)`/`gray(210)` outline onto combos, buttons and text boxes). Zero corner
/// radius and zero expansion everywhere (flat, aligned cells — no growing-on-hover), hairline
/// `#2b3238` frames at rest, the accent outline on hover/active, and the accent SELECTION with
/// **dark** on-accent text.
///
/// The on-accent contrast lives in [`Selection::stroke`](egui::style::Selection::stroke):
/// `button_style`/`interact_selectable` paint a selected cell's text (and a checkbox tick) in
/// `selection.stroke.color`, but ONLY as a galley fallback — so this is effective only because
/// neither surface sets `override_text_color` (which would bake a light colour into every
/// galley and defeat the fallback; that was the #89 wash). Idle interactable text is left at
/// `inactive.fg_stroke` = [`TEXT_MUTED`]; each surface raises it where its own readouts need
/// the primary tier (see [`apply_app_style`]).
fn apply_widget_visuals(v: &mut Visuals) {
    // Selection (accent-filled segmented cell / selected row / slider fill) with dark
    // text: `button_style` paints the selected text/tick in `selection.stroke.color`, so
    // this is what makes a lit cell read dark-on-accent.
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT_TEXT);
    v.hyperlink_color = ACCENT;

    // Flat + aligned everywhere: zero radius, zero expansion (hover/active must not grow).
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::ZERO;
        w.expansion = 0.0;
    }

    // noninteractive — labels + separators. No fill; `bg_stroke` is what `ui.separator()`
    // reads, so it is the inner rule. `fg_stroke` is the base label tier (muted).
    let ni = &mut v.widgets.noninteractive;
    ni.bg_fill = Color32::TRANSPARENT;
    ni.weak_bg_fill = Color32::TRANSPARENT;
    ni.bg_stroke = Stroke::new(1.0_f32, RULE);
    ni.fg_stroke = Stroke::new(1.0_f32, TEXT_MUTED);

    // inactive — idle interactables at rest (buttons, chips, text boxes, combos, checkbox
    // boxes). Hairline frame, hover-fill interior, muted idle text.
    let ia = &mut v.widgets.inactive;
    ia.bg_fill = HOVER_BG;
    ia.weak_bg_fill = HOVER_BG;
    ia.bg_stroke = Stroke::new(1.0_f32, BORDER);
    ia.fg_stroke = Stroke::new(1.0_f32, TEXT_MUTED);

    // hovered — accent outline on the hover fill, brightened text.
    let hv = &mut v.widgets.hovered;
    hv.bg_fill = HOVER_BG;
    hv.weak_bg_fill = HOVER_BG;
    hv.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    hv.fg_stroke = Stroke::new(1.0_f32, TEXT_HOVER);

    // active — pressed: accent outline on the deeper active fill, primary text.
    let ac = &mut v.widgets.active;
    ac.bg_fill = ACTIVE_BG;
    ac.weak_bg_fill = ACTIVE_BG;
    ac.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    ac.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);

    // open — an open combo/menu button. egui leaves this the brightest default
    // (`gray(210)` text on a `gray(60)` outline); pin it to the hairline frame + primary
    // text so an open picker matches the instrument-panel skin.
    let op = &mut v.widgets.open;
    op.bg_fill = ACTIVE_BG;
    op.weak_bg_fill = ACTIVE_BG;
    op.bg_stroke = Stroke::new(1.0_f32, BORDER);
    op.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
}

/// The app-wide Signal [`Style`] — applied once per frame to the egui context so the
/// right sidebar and the bottom palette dock inherit it. Monospace typography (~11 px
/// body), near-black opaque panel fills, hairline separators, muted text tiers, and the
/// shared accent widget visuals. The DISPLAY stack overrides this in its own scope
/// ([`apply_stack_style`]); the chrome painters are style-immune.
pub fn apply_app_style(style: &mut Style) {
    // Monospace across the app (§Tokens). Each tier is the mono family at its size, so
    // `ui.label`/`.small()`/`ui.heading()` all render monospace.
    style.text_styles = [
        (TextStyle::Small, FontId::new(SMALL_SIZE, FontFamily::Monospace)),
        (TextStyle::Body, FontId::new(BODY_SIZE, FontFamily::Monospace)),
        (TextStyle::Button, FontId::new(BODY_SIZE, FontFamily::Monospace)),
        (TextStyle::Heading, FontId::new(HEADING_SIZE, FontFamily::Monospace)),
        (TextStyle::Monospace, FontId::new(BODY_SIZE, FontFamily::Monospace)),
    ]
    .into();

    // Density pass (issue #90): egui's defaults render cramped rows + chunky buttons. Give
    // the sidebar the mock's airier rhythm — ~9 px row gaps, snug-but-not-tight button
    // padding, compact interact rows — so the sections breathe like the design.
    style.spacing.item_spacing = egui::Vec2::new(8.0, 6.0);
    style.spacing.button_padding = egui::Vec2::new(7.0, 3.0);
    style.spacing.interact_size.y = 20.0;

    let v = &mut style.visuals;
    // Near-black instrument surfaces.
    v.panel_fill = BG;
    v.window_fill = BG;
    v.window_stroke = Stroke::new(1.0_f32, BORDER);
    v.extreme_bg_color = HOVER_BG; // text-edit / drag-value inset cells
    v.faint_bg_color = HOVER_BG; // striped rows
    apply_widget_visuals(v);
    // Text tiers. NO `override_text_color` — it would bake a light colour into every galley
    // and desaturate the dark-on-accent selected text (issue #90). Instead: plain labels ride
    // `noninteractive.fg_stroke` (muted, set in `apply_widget_visuals`); `.weak()` hints ride
    // `weak_text_color`; and the sidebar's idle interactable text (which is where the
    // blocks+voxels VALUE readouts, DragValues and action buttons live) is raised to the
    // primary tier so those readouts stay bright while selected cells still resolve dark.
    v.weak_text_color = Some(TEXT_MUTED);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_PRIMARY);
}

/// The DISPLAY stack's scoped Signal style (`panel::signal_stack`). REPLACES the scoped
/// ui's style with a fresh [`Style::default`]-derived variant so the floating stack is
/// decoupled from the app-wide restyle around it — its tighter spacing + primary-forced
/// text render byte-identically to the epic-#80 stack regardless of [`apply_app_style`].
pub fn apply_stack_style(ui: &mut egui::Ui) {
    // Mono body across the stack's widgets (issue #90): the projection toggle + grid
    // checkboxes previously fell back to egui's ~14 px proportional Body, dwarfing the
    // painter-drawn 10 px section headers. Pin every tier to the stack mono size so the body
    // reads at the same scale as the sidebar and the headers.
    let mut style = Style {
        text_styles: [
            (TextStyle::Small, FontId::new(SMALL_SIZE, FontFamily::Monospace)),
            (TextStyle::Body, FontId::new(STACK_BODY_SIZE, FontFamily::Monospace)),
            (TextStyle::Button, FontId::new(STACK_BODY_SIZE, FontFamily::Monospace)),
            (TextStyle::Heading, FontId::new(HEADING_SIZE, FontFamily::Monospace)),
            (TextStyle::Monospace, FontId::new(STACK_BODY_SIZE, FontFamily::Monospace)),
        ]
        .into(),
        ..Style::default()
    };
    style.spacing.item_spacing = egui::Vec2::new(6.0, 5.0);
    style.spacing.button_padding = egui::Vec2::new(6.0, 2.0);
    style.spacing.interact_size.y = 18.0;
    let v = &mut style.visuals;
    // NO `override_text_color` (see `apply_app_style`): the stack keeps egui's dark-on-accent
    // selected-text fallback so the lit projection cell reads dark, and its idle toggles stay
    // at the muted tier (`inactive.fg_stroke`, from `apply_widget_visuals`) — the mock's idle
    // ORTHO cell. `weak_text_color` keeps `.weak()` stats faint.
    v.weak_text_color = Some(TEXT_MUTED);
    apply_widget_visuals(v);
    *ui.style_mut() = style;
}

/// Lay out `text` as UPPERCASE monospace with extra letter spacing, returning the galley
/// for painting (width/height measurement + `painter.galley`). The stack's header,
/// chevron-row and edge-tab captions use this.
pub fn letter_spaced(
    ui: &egui::Ui,
    text: &str,
    color: Color32,
    size: f32,
    spacing: f32,
) -> Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &text.to_uppercase(),
        0.0,
        TextFormat {
            font_id: FontId::monospace(size),
            color,
            extra_letter_spacing: spacing,
            ..Default::default()
        },
    );
    ui.painter().layout_job(job)
}

/// A sidebar SECTION HEADER in the stack's header voice: `title` UPPERCASE, letter-spaced
/// monospace at ~10 px in the secondary tier. Flows as an ordinary [`egui::Label`] so it
/// participates in the sidebar's vertical layout (unlike the stack's absolute-rect header
/// bar). Replaces the legacy `ui.strong("Scene")` section titles across the sidebar.
pub fn section_heading(ui: &mut egui::Ui, title: &str) {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &title.to_uppercase(),
        0.0,
        TextFormat {
            font_id: FontId::monospace(SECTION_HEADER_SIZE),
            color: TEXT_SECONDARY,
            extra_letter_spacing: SECTION_HEADER_SPACING,
            ..Default::default()
        },
    );
    ui.add(egui::Label::new(job).selectable(false));
}
