//! The non-inspector control sections: camera projection, the Display toggles, and
//! the .vox export button.

use super::{ExportPanelState, PanelResponse, PanelState};
use crate::theme;
use camera::ProjectionMode;
use document::intent::Intent;

/// The camera projection toggle (display-only: no rebuild) — the **body** of the Signal
/// stack's VIEWPORT section. The section header + framing are drawn by the stack; this only
/// lays out the Perspective / Orthographic segmented control.
pub(super) fn build_camera_body(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.horizontal(|ui| {
        ui.selectable_value(
            &mut state.projection_mode,
            ProjectionMode::Perspective,
            "Perspective",
        );
        ui.selectable_value(
            &mut state.projection_mode,
            ProjectionMode::Orthographic,
            "Orthographic",
        );
    });
}

/// The display MASTER toggles (voxel grid on faces, block lattice, floor grid, view cube,
/// debug faces) — the **body** of the Signal stack's GRIDS section. The section header is drawn
/// by the stack; this only lays out the checkboxes.
pub(super) fn build_display_body(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    // The three grid MASTERS are scene fields, so they bind to LOCAL copies and a change
    // emits ONE `SetGridMasters`. The masters are read live by the per-frame line batch / mesh
    // shader (no re-resolve), so `SetGridMasters`'s effect is `none()` — no rebuild, no
    // auto-frame. `axes_on_top` / `debug_face_orientation` are PanelState DISPLAY fields, not
    // scene mutations, so they mutate in place.
    let mut voxel = state.scene.master_voxel_grid;
    let mut lattice = state.scene.master_block_lattice;
    let mut floor = state.scene.master_floor_grid;
    let mut masters_changed = false;
    // The on-face voxel grid is per-object; this is the scene-wide MASTER, ANDed (in the mesh
    // shaders) with each node's own flag.
    masters_changed |= ui
        .checkbox(&mut voxel, "Voxel grid on faces (master)")
        .changed();
    // Scene-wide MASTERS for the per-object lattice / floor grids.
    masters_changed |= ui
        .checkbox(&mut lattice, "Block lattice (master)")
        .changed();
    masters_changed |= ui.checkbox(&mut floor, "Floor grid (master)").changed();
    if masters_changed {
        response.emit(Intent::SetGridMasters {
            voxel,
            lattice,
            floor,
        });
    }
    // The Points' axes as a nav marker through the model (on) vs occluded scaffold (off).
    ui.checkbox(&mut state.axes_on_top, "Axes on top");
    // The transform gizmo is selection-driven (drawn on the active node), so it has no
    // Display toggle of its own.
    ui.checkbox(&mut state.debug_face_orientation, "Debug: face orientation");
    // Grazing-rim brick diagnostic (face-axis color + UV checkerboard). Keeps the brick
    // path engaged (unlike the mesh-only face-orientation debug above) so the raymarch under
    // investigation is what's shown.
    ui.checkbox(&mut state.debug_brick_faces, "Debug: brick faces");
}

/// Export section: a single "Export .vox" button plus a progress / status line.
/// The click is reported via [`PanelResponse::clicked_export_vox`];
/// the caller opens the OS save dialog and dispatches the write to the background export
/// worker (so the panel stays free of file-system concerns). While an export is in flight
/// the button is disabled — the shell serializes exports — and `export.status_line`
/// carries the "Exporting… done/total" progress; otherwise it is the last completion /
/// failure / large-export message.
pub(super) fn build_export_section(
    ui: &mut egui::Ui,
    response: &mut PanelResponse,
    export: ExportPanelState,
) {
    ui.add_space(8.0);
    theme::section_heading(ui, "Export");
    let button = ui
        .add_enabled(!export.in_flight, egui::Button::new("Export .vox"))
        .on_hover_text("Write the resolved voxels as a MagicaVoxel .vox file")
        .on_disabled_hover_text("An export is already running — it will finish in the background");
    if button.clicked() {
        response.clicked_export_vox = true;
    }
    if let Some(line) = export.status_line {
        ui.label(egui::RichText::new(line).small().weak());
    }
    ui.separator();
}
