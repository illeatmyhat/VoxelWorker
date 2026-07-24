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

    /// This theme's colour swatches — the token registry as this theme resolves it.
    pub fn swatches(self) -> &'static [color_palette::Swatch] {
        match self {
            Theme::Signal => color_palette::SWATCHES,
        }
    }
}

/// The active UI theme — the single seam a colour lookup resolves through. One theme today, so this
/// is [`Theme::Signal`]; when a second lands, this reads the persisted choice instead.
pub fn active() -> Theme {
    Theme::default()
}
