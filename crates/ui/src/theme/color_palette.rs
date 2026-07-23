//! `theme::palette` — the Signal design language's colour tokens, as one registry
//! (`docs/design/viewport-chrome-signal.md` §Tokens; ADR 0018).
//!
//! Every colour the UI paints with is a `pub const` here, defined through [`color_token!`] so it
//! ALSO lands in [`SWATCHES`] — the design_reference sheet renders that registry, so a colour
//! **cannot exist without a swatch in the sheet** and the two can never drift (owner 2026-07-23,
//! "shown by construction, much like settings show up in a config file by construction"). Add a
//! colour by adding a line to the [`color_token!`] block; the sheet needs no edit.
//!
//! These consts are the Signal (dark) values. When a second theme lands (light / system scheme),
//! the token NAMES and meanings stay — this is the single registry — and the values move behind a
//! resolved-per-theme lookup; the [`crate::theme`] module is the seam for that.
//!
//! Re-exported at [`crate::theme`] (`pub use color_palette::*`), so call sites read `theme::ACCENT`.

use egui::Color32;

/// One colour token as the design_reference sheet renders it: its const name, its value, and the
/// meaning it is permitted to carry. Built only by [`color_token!`], so every entry is a real
/// `pub const` and vice-versa.
#[derive(Debug, Clone, Copy)]
pub struct Swatch {
    /// The token's const identifier (e.g. `"ACCENT"`).
    pub name: &'static str,
    /// Its colour value.
    pub color: Color32,
    /// The meaning it may carry — the sheet's third column and the const's own doc.
    pub meaning: &'static str,
}

/// Define the Signal colour tokens. Each entry emits BOTH a `pub const <NAME>: Color32` (its doc =
/// its meaning) AND an entry in the [`SWATCHES`] registry — so a colour token **cannot exist
/// without a swatch in the design_reference sheet**, which renders by construction. Adding a token
/// here is the only way to add one; the sheet needs no edit.
macro_rules! color_token {
    ($( $name:ident = $color:expr, $meaning:literal );* $(;)?) => {
        $(
            #[doc = $meaning]
            pub const $name: Color32 = $color;
        )*
        /// Every colour token, in declaration order — the ONE registry the design_reference sheet
        /// iterates. By construction a token is here iff it is a `pub const` above.
        pub const SWATCHES: &[Swatch] = &[
            $( Swatch { name: stringify!($name), color: $name, meaning: $meaning } ),*
        ];
    };
}

color_token! {
    BG = Color32::from_rgb(0x0b, 0x0d, 0x0f), "panel fill — the instrument surface (sidebar + palette dock), opaque #0b0d0f";
    BG_FLOAT = Color32::from_rgba_unmultiplied_const(0x0b, 0x0d, 0x0f, 217), "panel fill at ~85% — the floating DISPLAY stack over the viewport";
    BORDER = Color32::from_rgb(0x2b, 0x32, 0x38), "hairline border, 1 px, outer (bordered cells, panel edges)";
    RULE = Color32::from_rgb(0x1c, 0x21, 0x26), "hairline rule, inner divisions / separators";
    HOVER_BG = Color32::from_rgb(0x12, 0x16, 0x1b), "row / header hover fill";
    ACTIVE_BG = Color32::from_rgb(0x16, 0x1a, 0x1e), "active (pressed / open) fill · rail button hover";
    TEXT_PRIMARY = Color32::from_rgb(0xdf, 0xe7, 0xef), "values, live labels — what is read first";
    TEXT_SECONDARY = Color32::from_rgb(0xae, 0xb9, 0xc4), "labels · section-header names";
    TEXT_HOVER = Color32::from_rgb(0xe3, 0xeb, 0xf3), "header name on hover — the brightest text step";
    TEXT_MUTED = Color32::from_rgb(0x78, 0x82, 0x8c), "idle glyphs, secondary labels, idle tabs";
    TEXT_FAINT = Color32::from_rgb(0x4d, 0x56, 0x5f), "readouts, counts, chevrons, subtitles";
    TEXT_HINT = Color32::from_rgb(0x3c, 0x44, 0x4c), "hints — the quietest legible step";
    ACCENT = Color32::from_rgb(0x9c, 0xb4, 0xd8), "ACTIVE · SELECTED · LIVE — and the onion haze. No valence: not 'good', not 'safe'";
    ACCENT_TEXT = Color32::from_rgb(0x0b, 0x0d, 0x0f), "text / glyphs painted ON an accent fill (near-black, for contrast)";
    HANDLE_HOVER = Color32::from_rgb(0xc7, 0xd3, 0xe0), "a hovered manipulator handle / sketch edge fills or strokes this — brighter than the accent, only ever a hover state (ADR 0030)";
    WARN = Color32::from_rgb(0xd9, 0x60, 0x3f), "subtraction and removal, plus genuine warnings · doubles as the X spatial axis";
    AXIS_Y = Color32::from_rgb(0x7d, 0xba, 0x6a), "Y spatial axis — green; the snap-guide triad is X warn · Y this · Z accent (ADR 0028)";
    SKETCH_PLANE_FILL = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x0f), "sketch working-plane fill — accent at low alpha, so the profile stays primary (ADR 0028)";
    SKETCH_PLANE_GRID = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x24), "sketch plane fine grid lines — accent, quiet";
    SKETCH_PLANE_GRID_BLOCK = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x55), "sketch plane block grid lines — accent, brighter, reads through the fine grid";
}
