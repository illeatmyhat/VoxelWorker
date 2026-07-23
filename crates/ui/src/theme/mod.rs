//! `theme` — the UI's look: the colour [`palette`] and (today) the Signal design language's egui
//! `Style`. One home so a second theme — a light mode, or the OS colour scheme — has somewhere to
//! live.
//!
//! **Theme selection is scaffolded here even though only one theme exists** (owner 2026-07-23):
//! [`Theme`] enumerates the choices, [`Theme::ALL`] lists them (a future picker / the
//! design_reference iterate it), and [`active`] is the single seam every colour lookup resolves
//! through. Today there is one theme — Signal, the dark instrument-panel look — so [`active`]
//! returns it and the [`palette`] consts ARE its values. When a second theme lands, the token
//! NAMES + meanings in [`palette`] stay the one registry, the values move behind a per-theme
//! lookup on [`Theme`], and [`active`] starts reading a setting — call sites that already resolve
//! through this seam need no change.
//!
//! Layout: [`color_palette`] is the colour registry; [`style`] applies the Signal egui `Style`;
//! [`text`] holds the reusable caption / section-heading painters. The public API is re-exported at
//! `theme::` so call sites read `theme::ACCENT`, `theme::apply_app_style`, `theme::section_heading`.

pub mod color_palette;
mod style;
mod text;

pub use color_palette::*;
pub use style::{apply_app_style, apply_stack_style};
pub use text::{letter_spaced, section_heading};

/// A selectable UI theme. One today (Signal, the near-black instrument-panel look); a light theme
/// or the OS colour scheme would join as new variants, each supplying its own [`palette`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// The Signal design language — the dark instrument-panel look (ADR 0018). The only theme
    /// today; its values are the [`palette`] consts.
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

    /// This theme's colour swatches — the token registry as this theme resolves it. One theme
    /// today, so this is the [`palette`] registry; a second theme returns its own values here.
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
