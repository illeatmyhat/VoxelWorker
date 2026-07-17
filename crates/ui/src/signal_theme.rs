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
/// Warn / subtract red `#d9603f` (the app's warn colour).
pub const WARN: Color32 = Color32::from_rgb(0xd9, 0x60, 0x3f);

// --- Typography sizes (design points; §Tokens: monospace, 10–11 px) ---
/// Body / control text (~11 px).
const BODY_SIZE: f32 = 11.0;
/// Small hints / readouts (~9.5 px).
const SMALL_SIZE: f32 = 9.5;
/// The sidebar title block heading.
const HEADING_SIZE: f32 = 15.0;
/// A section header caption (UPPERCASE, letter-spaced).
const SECTION_HEADER_SIZE: f32 = 10.0;
/// Extra letter spacing on section-header captions.
const SECTION_HEADER_SPACING: f32 = 1.5;

/// The Signal WIDGET visuals shared by both the app-wide style and the DISPLAY stack's
/// scoped style: zero corner radius everywhere, the accent selection (dark text on the
/// accent fill — the legacy blue dies), hairline-bordered inactive cells, an
/// accent-outlined hover on the hover fill, and an accent-outlined active state. Does not
/// touch typography, panel fills or separators — those differ per surface (see
/// [`apply_app_style`]).
fn apply_widget_visuals(v: &mut Visuals) {
    // Selection (accent-filled segmented cell / selected row / slider fill) with dark
    // text: `interact_selectable`/`button_style` paint the selected text in
    // `selection.stroke.color`, so this is what makes a lit cell read dark-on-accent.
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, ACCENT_TEXT);
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
    v.widgets.inactive.bg_fill = HOVER_BG;
    v.widgets.inactive.weak_bg_fill = HOVER_BG;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT_MUTED);
    v.widgets.hovered.bg_fill = HOVER_BG;
    v.widgets.hovered.weak_bg_fill = HOVER_BG;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, TEXT_HOVER);
    v.widgets.active.bg_fill = ACTIVE_BG;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
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

    let v = &mut style.visuals;
    // Near-black instrument surfaces.
    v.panel_fill = BG;
    v.window_fill = BG;
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.extreme_bg_color = HOVER_BG; // text-edit / drag-value inset cells
    v.faint_bg_color = HOVER_BG; // striped rows
    // Text tiers: primary widget/label text, muted weak hints.
    v.override_text_color = Some(TEXT_PRIMARY);
    v.weak_text_color = Some(TEXT_MUTED);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT_PRIMARY);
    apply_widget_visuals(v);
    // `ui.separator()` reads the noninteractive bg_stroke — make it the inner rule.
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, RULE);
}

/// The DISPLAY stack's scoped Signal style (`panel::signal_stack`). REPLACES the scoped
/// ui's style with a fresh [`Style::default`]-derived variant so the floating stack is
/// decoupled from the app-wide restyle around it — its tighter spacing + primary-forced
/// text render byte-identically to the epic-#80 stack regardless of [`apply_app_style`].
pub fn apply_stack_style(ui: &mut egui::Ui) {
    let mut style = Style::default();
    style.spacing.item_spacing = egui::Vec2::new(6.0, 5.0);
    style.spacing.button_padding = egui::Vec2::new(6.0, 2.0);
    style.spacing.interact_size.y = 18.0;
    let v = &mut style.visuals;
    v.override_text_color = Some(TEXT_PRIMARY);
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
