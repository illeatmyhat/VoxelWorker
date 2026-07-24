//! The Signal colour tokens as one registry. Each is a `pub const` defined via `color_token!`,
//! which also emits its [`SWATCHES`] entry — so the design_reference sheet renders every token by
//! construction and none can drift. Re-exported at [`crate::theme`] (`theme::ACCENT`). Values are
//! the Signal (dark) theme; a second theme resolves the same token names differently.

#![allow(clippy::disallowed_methods)]

use egui::Color32;

/// A colour token: its const name, value, and permitted meaning (the sheet's row).
#[derive(Debug, Clone, Copy)]
pub struct Swatch {
    pub name: &'static str,
    pub color: Color32,
    pub meaning: &'static str,
}

/// Emit each Signal colour token as a `pub const` plus its [`SWATCHES`] entry.
macro_rules! color_token {
    ($( $name:ident = $color:expr, $meaning:literal );* $(;)?) => {
        $(
            #[doc = $meaning]
            pub const $name: Color32 = $color;
        )*
        /// Every colour token, in declaration order — the registry the design_reference iterates.
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
    ACCENT_FAINT = Color32::from_rgba_premultiplied(0x2f, 0x37, 0x43, 0x4d), "a faint accent tint — the rail's lit-cell glow / the DISPLAY-stack accent wash (premultiplied)";
    SCRUBBER_TRACK = Color32::from_rgb(0x1b, 0x17, 0x12), "layer scrubber — the track background (a warm-dark channel the band rides in)";
    SCRUBBER_TICK = Color32::from_rgb(0x3a, 0x5f, 0x57), "layer scrubber — the block-boundary snap ticks (teal)";
    SCRUBBER_BAND = Color32::from_rgba_unmultiplied_const(0x5f, 0xb8, 0xa4, 70), "layer scrubber — the selected-band fill (teal, translucent)";
    SCRUBBER_HANDLE_EDGE = Color32::from_rgb(0x10, 0x0c, 0x08), "layer scrubber — the handle border (near-black warm)";
    DIALOG_BG = Color32::from_rgb(0x12, 0x14, 0x18), "floating dialog background (the Add-shape dialog)";
    DIALOG_BORDER = Color32::from_rgb(0x3c, 0x42, 0x4a), "floating dialog border";
}
