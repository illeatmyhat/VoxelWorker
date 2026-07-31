//! The UI's look: the [`color_palette`] registry, the Signal egui [`style`], and [`text`] painters,
//! re-exported flat (`theme::ACCENT`, `theme::apply_app_style`). [`Theme`] scaffolds theme selection
//! with one variant today; [`active`] is the seam a second theme (light / OS scheme) slots into.

pub mod color_palette;
mod style;
mod text;

pub use color_palette::*;
pub use style::{apply_app_style, apply_stack_style};
pub use text::{letter_spaced, section_heading};

/// A selectable UI theme. One today; a light / OS-scheme theme would join as a variant supplying its
/// own [`color_palette`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// The Signal design language — the dark instrument-panel look (ADR 0018).
    #[default]
    Signal,
}

impl Theme {
    /// Every selectable theme, in menu order — the list a settings picker and the design_reference
    /// iterate, so adding a variant surfaces it everywhere without a second edit.
    pub const ALL: &'static [Theme] = &[Theme::Signal];

    /// The human-readable name for a theme picker.
    pub fn label(self) -> &'static str {
        match self {
            Theme::Signal => "Signal",
        }
    }

    /// This theme's color swatches — the token registry as this theme resolves it.
    pub fn swatches(self) -> &'static [color_palette::Swatch] {
        match self {
            Theme::Signal => color_palette::SWATCHES,
        }
    }
}

/// The active UI theme — the single seam a color lookup resolves through. One theme today, so this
/// is [`Theme::Signal`]; when a second lands, this reads the persisted choice instead.
pub fn active() -> Theme {
    Theme::default()
}

/// A palette token as LINEAR RGB + its own alpha, for a GPU pass that paints in the viewport.
///
/// Some chrome is painted by a render pipeline rather than by egui — the sketch region wash is a
/// fragment shader over the plane. Those still take their color from this registry, converted here,
/// so the 2D and 3D readings of one signal cannot drift apart.
///
/// UNMULTIPLIED, because a pipeline that blends premultiplied does that multiply itself and doing it
/// twice darkens a translucent wash by its own alpha.
pub fn linear_rgba(color: egui::Color32) -> [f32; 4] {
    let [red, green, blue, alpha] = color.to_srgba_unmultiplied();
    [
        substrate::srgb::srgb_component_to_linear(red),
        substrate::srgb::srgb_component_to_linear(green),
        substrate::srgb::srgb_component_to_linear(blue),
        alpha as f32 / 255.0,
    ]
}
