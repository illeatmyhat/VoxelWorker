//! `theme::text` — the Signal design language's reusable text painters: UPPERCASE letter-spaced
//! captions and the sidebar section heading (issue #89; ADR 0018).

use egui::{Color32, FontId, TextFormat};
use std::sync::Arc;

use super::color_palette::TEXT_SECONDARY;

/// A section header caption size (UPPERCASE, letter-spaced), ~10 px.
const SECTION_HEADER_SIZE: f32 = 10.0;
/// Extra letter spacing on section-header captions.
const SECTION_HEADER_SPACING: f32 = 1.5;

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
