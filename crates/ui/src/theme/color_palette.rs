//! The Signal color tokens as one registry. Each is a `pub const` defined via `color_token!`,
//! which also emits its [`SWATCHES`] entry — so the design_reference sheet renders every token by
//! construction and none can drift. Re-exported at [`crate::theme`] (`theme::ACCENT`). Values are
//! the Signal (dark) theme; a second theme resolves the same token names differently.

#![allow(clippy::disallowed_methods)]

use egui::Color32;

/// A color token: its const name, value, and permitted meaning (the sheet's row).
#[derive(Debug, Clone, Copy)]
pub struct Swatch {
    pub name: &'static str,
    pub color: Color32,
    pub meaning: &'static str,
}

/// Emit each Signal color token as a `pub const` plus its [`SWATCHES`] entry.
macro_rules! color_token {
    ($( $name:ident = $color:expr, $meaning:literal );* $(;)?) => {
        $(
            #[doc = $meaning]
            pub const $name: Color32 = $color;
        )*
        /// Every color token, in declaration order — the registry the design_reference iterates.
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
    HANDLE_HOVER = Color32::from_rgb(0xc7, 0xd3, 0xe0), "a hovered manipulator handle / sketch edge fills or strokes this — brighter than the accent, only ever a hover state";
    WARN = Color32::from_rgb(0xd9, 0x60, 0x3f), "subtraction and removal, plus genuine warnings · doubles as the X spatial axis";
    AXIS_Y = Color32::from_rgb(0x7d, 0xba, 0x6a), "Y spatial axis — green; the snap-guide triad is X warn · Y this · Z accent";
    SKETCH_PLANE_FILL = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x0f), "sketch working-plane fill — accent at low alpha, so the profile stays primary";
    SKETCH_PLANE_GRID = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x24), "sketch plane fine grid lines — accent, quiet";
    SKETCH_PLANE_GRID_BLOCK = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x55), "sketch plane block grid lines — accent, brighter, reads through the fine grid";
    SKETCH_REGION_FILL = Color32::from_rgba_unmultiplied_const(0x35, 0x6f, 0xc9, 0x8a), "a PICKED sketch region's wash — the 2D read of the 3D selection wash: what resolves as material. The accent's own hue, driven deep and saturated: over the viewport's near-black the pale accent at wash alpha composites to a gray that reads as nothing, so this one departs from the accent VALUE to keep the accent HUE";
    SKETCH_CONSTRUCTION = Color32::from_rgb(0xdd, 0xa0, 0x6a), "sketch CONSTRUCTION linetype — geometry that locates but is not part of the shape. Always dashed, and the one ink outside the accent a sketch mark may spend: the Construction tool's icon quotes this color rather than coding for it, which is what keeps the exception from generalising. Warm, so it never reads as a cooler ACCENT step; distinct from WARN, which means removal";
    SKETCH_CONSTRAINT = Color32::from_rgb(0xe2, 0x56, 0x4b), "sketch CONSTRAINT — the entity a relation DRIVES. The second ink outside the accent a sketch mark may spend, and it is spent for the same reason the first is: a constraint glyph has to say which of two entities moved, and drawing the driven one in the accent said only 'this one is picked', which is what every other glyph in the set already means by it. Redder and cooler-dark than WARN, which means removal, and than SKETCH_CONSTRUCTION, which is a linetype rather than a role";
    SKETCH_TANGENT_LEG = Color32::from_rgb(0x4f, 0xb5, 0xa6), "sketch TANGENT HANDLE — the leg from a fit point out to each of its two handle ends. The third ink outside the accent, and the last: a handle is neither a linetype (SKETCH_CONSTRUCTION) nor a driven entity (SKETCH_CONSTRAINT) but a MANIPULATOR that is always present on every fit point, so drawing it in either of those said something false about half the marks on screen. Teal, solid — solid because a handle is not construction, teal because it is the one cool hue no accent step occupies";
    SKETCH_TANGENT_POINT = Color32::from_rgb(0x6f, 0xc9, 0x74), "sketch TANGENT HANDLE end — the grabbable dot at each end of the leg. Green rather than the leg's teal so the thing you can drag is separable at a glance from the thing you cannot, which is the same split the rest of the chrome draws with fill-versus-stroke and cannot here: a handle end is a dot on a line of its own family. Near AXIS_Y without being it — the axis triad never appears inside an edited sketch";
    SKETCH_POINT_OFF_INK = Color32::from_rgb(0x5e, 0x6c, 0x82), "a sketch dot with NO curve drawn through it — a center, a control point, a conic's control, a free point. An accent STEP rather than a fourth ink outside the accent: the point is scaffolding for the shape and has to read as quieter than the shape, which is a value question, not a hue one. The accent's own channels at 60%, so it is the same color seen further away";
    ACCENT_FAINT = Color32::from_rgba_premultiplied(0x2f, 0x37, 0x43, 0x4d), "a faint accent tint — the rail's lit-cell glow / the DISPLAY-stack accent wash (premultiplied)";
    MARQUEE_WINDOW_FILL = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x1e), "marquee WINDOW box fill (drag left→right, fully-enclosed semantic) — the stronger of the pair";
    MARQUEE_CROSSING_FILL = Color32::from_rgba_unmultiplied_const(0x9c, 0xb4, 0xd8, 0x0c), "marquee CROSSING box fill (drag right→left, any-intersection) — lighter, under a dashed outline";
    SCRUBBER_TRACK = Color32::from_rgb(0x1b, 0x17, 0x12), "layer scrubber — the track background (a warm-dark channel the band rides in)";
    SCRUBBER_TICK = Color32::from_rgb(0x3a, 0x5f, 0x57), "layer scrubber — the block-boundary snap ticks (teal)";
    SCRUBBER_BAND = Color32::from_rgba_unmultiplied_const(0x5f, 0xb8, 0xa4, 70), "layer scrubber — the selected-band fill (teal, translucent)";
    SCRUBBER_HANDLE_EDGE = Color32::from_rgb(0x10, 0x0c, 0x08), "layer scrubber — the handle border (near-black warm)";
    RETICLE = Color32::from_rgba_unmultiplied_const(0x8a, 0x93, 0x9c, 128), "orbit-mode targeting reticle — neutral GRAY at 50%, the one mark deliberately outside the accent. It spans most of the viewport, so an accent one would repaint the whole frame; and it is a place-marker, not a live value. The tone is a theme token so it tracks the theme's contrast rather than being pinned to one background";
    DIALOG_BG = Color32::from_rgb(0x12, 0x14, 0x18), "floating dialog background (the Add-shape dialog)";
    DIALOG_BORDER = Color32::from_rgb(0x3c, 0x42, 0x4a), "floating dialog border";
}
