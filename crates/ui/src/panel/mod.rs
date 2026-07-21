//! The shared egui side panel.
//!
//! Exactly one implementation, used by both the windowed app and the headless
//! screenshot harness (Hard requirement #3), so the captured frame is identical
//! to the live one.
//!
//! The inspector's [`GeometryParams`](document::voxel::GeometryParams) (shape, size,
//! density, wall) are split from display/camera params (projection, material selection) by
//! *consumer*, a split going back to the panel's original Milestone 3 hard requirement #3:
//!
//!   * [`GeometryParams`](document::voxel::GeometryParams) drives a **rebuild-dirty** flag.
//!     Changing it re-resolves the voxel grid.
//!   * Display/camera params live in [`PanelState`] directly and never trigger a voxel
//!     rebuild. (The camera projection toggle itself is no longer built here — it moved to
//!     the floating Signal display stack, issue #88 — but `projection_mode` is still a
//!     `PanelState` display field on this same no-rebuild side of the split.)
//!
//! This split is what enforces the regression guards: selecting a shape only
//! sets [`GeometryParams::shape`](document::voxel::GeometryParams::shape) (never
//! the size or the camera), and changing density only sets
//! [`GeometryParams::voxels_per_block`](document::voxel::GeometryParams::voxels_per_block)
//! (never the block size).
//!
//! The panel is one logical unit split across submodules by section identity:
//! [`state`] holds the mutable state + response types; the `build_*` section
//! builders live in [`nodes`], [`points`], [`inspector`], [`controls`],
//! [`layers`], and [`palette`]; [`build_panel`] (here) is the top-level
//! assembler that lays them out. Every previously-public item is re-exported
//! here so `ui::panel::…` paths resolve unchanged.

mod add_shape_dialog;
mod controls;
mod inspector;
mod layers;
mod nodes;
mod palette;
mod points;
mod signal_stack;
mod state;

pub use add_shape_dialog::build_add_shape_dialog;
pub use signal_stack::{build_signal_stack, cube_right_inset_points};
pub use state::{
    ExportPanelState, LayerRange, OrientationSnap, PanelResponse, PanelState, PlacementGhost,
    PlacementSnap, PositionSnap, SignalStackState, ViewMode,
};

use crate::palette::BlockPalette;
use crate::signal_theme;

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// The sidebar hosts the scene tree, points, inspector and export; the display-related
/// sections (VIEWPORT / ONION FOG / GRIDS) left it for the floating Signal display stack
/// ([`build_signal_stack`], issue #88), which the shell renders separately with the
/// layer-track length + measured diameter. Returns a [`PanelResponse`] describing what the
/// user changed.
/// The sidebar's section stack, independent of which column hosts it.
///
/// Factored out so the [`workspace`](crate::workspace) inspector column can host the same
/// sections while the new information architecture is proved, rather than the sections being
/// duplicated into it and then drifting apart — the failure the rail's glyphs already had.
pub(crate) fn build_sidebar_sections(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    response: &mut PanelResponse,
) {
    nodes::build_node_list_section(ui, state, response);
    points::build_points_section(ui, state, response);
    inspector::build_inspector_section(ui, state, response);
    controls::build_export_section(ui, response, export);
}

pub fn build_panel(
    root_ui: &mut egui::Ui,
    state: &mut PanelState,
    export: ExportPanelState,
    palette: &BlockPalette,
) -> PanelResponse {
    let mut response = PanelResponse::default();

    // The palette dock lives along the bottom (prototype layout); it is its own
    // bottom panel so the right-hand controls keep their width.
    palette::build_palette_dock(root_ui, palette, &mut response);

    egui::Panel::right("voxel_worker_controls")
        .resizable(false)
        .default_size(300.0)
        .show_inside(root_ui, |ui| {
            // The panel outgrows short windows; scroll (wheel or drag) instead
            // of clipping the lower sections.
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    // The title block in the Signal voice (issue #89): primary-tier
                    // wordmark over a faint subtitle.
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new("VoxelWorker")
                            .monospace()
                            .size(15.0)
                            .strong()
                            .color(signal_theme::TEXT_PRIMARY),
                    );
                    ui.label(
                        egui::RichText::new("Vintage Story chiseling planner")
                            .monospace()
                            .size(9.5)
                            .color(signal_theme::TEXT_FAINT),
                    );
                    ui.add_space(6.0);
                    ui.separator();

                    // The display-related sections (VIEWPORT / ONION FOG / GRIDS) left the
                    // sidebar for the floating Signal display stack (issue #88,
                    // `panel::signal_stack`, rendered by `run_egui_frame`). The sidebar keeps
                    // the scene tree, points, inspector and export.
                    build_sidebar_sections(ui, state, export, &mut response);

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

    response
}
