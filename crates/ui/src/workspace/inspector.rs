//! The inspector column: the selected node's own dialog.
//!
//! For now this region HOSTS the existing sidebar sections rather than replacing them. The
//! new information architecture is the thing being proved here — where the regions are, what
//! belongs in each, and that the viewport keeps the room it needs — and re-authoring the
//! node dialog at the same time would confound the two changes. The sections move into this
//! column's own vocabulary (a per-kind field list, the signed Measurement fields, the
//! authorship probe beneath) in the slices that follow.

use super::{hairline, region_frame, Edge, INSPECTOR_WIDTH};
use crate::palette::BlockPalette;
use crate::panel::{ExportPanelState, PanelResponse, PanelState};
use crate::signal_theme;

/// Build the inspector column.
pub(super) fn build_inspector(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    _palette: &BlockPalette,
    response: &mut PanelResponse,
) {
    egui::Panel::right("workspace_inspector")
        .resizable(false)
        .default_size(INSPECTOR_WIDTH)
        .frame(region_frame())
        .show_inside(root_ui, |ui| {
            let column = ui.max_rect();
            hairline(ui.painter(), column, Edge::Left, signal_theme::BORDER);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.add_space(9.0);
                    let title =
                        signal_theme::letter_spaced(ui, "Inspector", signal_theme::TEXT_MUTED, 9.0, 2.0);
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width(), title.size().y + 7.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().galley(
                        egui::pos2(rect.left() + 11.0, rect.top()),
                        title,
                        signal_theme::TEXT_MUTED,
                    );
                    hairline(ui.painter(), rect, Edge::Bottom, signal_theme::RULE);

                    crate::panel::build_sidebar_sections(ui, state, export, response);

                    if let Some(millions) = state.voxel_cap_warning_millions {
                        ui.add_space(8.0);
                        ui.separator();
                        ui.colored_label(
                            signal_theme::WARN,
                            format!("3D paused — {millions:.1}M voxels; lower size/density"),
                        );
                    }
                });
        });
}
