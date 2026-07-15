//! The non-inspector control sections: camera projection, the Display toggles, and
//! the .vox export button.

use super::{ExportPanelState, PanelResponse, PanelState};
use camera::ProjectionMode;
use document::intent::Intent;

/// Camera projection toggle (display-only: no rebuild).
pub(super) fn build_camera_section(ui: &mut egui::Ui, state: &mut PanelState) {
    ui.add_space(8.0);
    ui.strong("Camera → Projection");
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
    ui.separator();
}

/// Display section. M4 added the voxel-grid overlay; M5 wired the view cube and
/// the origin gizmo; M8 wires the block lattice and fine floor grid (#10).
pub(super) fn build_display_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(8.0);
    ui.strong("Display");
    // ADR 0003 Phase C C4a: the three grid MASTERS are scene fields, so they bind to
    // LOCAL copies and a change emits ONE `SetGridMasters`. The masters are read live
    // by the per-frame line batch / mesh shader (no re-resolve), so `SetGridMasters`'s
    // effect is `none()` — no rebuild, no auto-frame — matching the old direct writes.
    // `show_view_cube` / `debug_face_orientation` are PanelState DISPLAY fields (not
    // scene mutations), so they keep mutating in place.
    let mut voxel = state.scene.master_voxel_grid;
    let mut lattice = state.scene.master_block_lattice;
    let mut floor = state.scene.master_floor_grid;
    let mut masters_changed = false;
    // Issue #29 S4: the on-face voxel grid is per-object; this is the scene-wide
    // MASTER, ANDed (in the mesh shaders) with each node's own flag.
    masters_changed |= ui
        .checkbox(&mut voxel, "Voxel grid on faces (master)")
        .changed();
    // Issue #29 S3: scene-wide MASTERS for the per-object lattice / floor grids.
    masters_changed |= ui.checkbox(&mut lattice, "Block lattice (master)").changed();
    masters_changed |= ui.checkbox(&mut floor, "Floor grid (master)").changed();
    if masters_changed {
        response.emit(Intent::SetGridMasters { voxel, lattice, floor });
    }
    ui.checkbox(&mut state.show_view_cube, "View cube");
    // Issue #29 S2: the transform gizmo is now selection-driven (drawn on the
    // active node), so it no longer has a Display toggle.
    ui.checkbox(&mut state.debug_face_orientation, "Debug: face orientation");
    ui.separator();
}

/// Export section (M8): a single "Export .vox" button plus a progress / status line
/// (slow-paths item 2). The click is reported via [`PanelResponse::clicked_export_vox`];
/// the caller opens the OS save dialog and dispatches the write to the background export
/// worker (so the panel stays free of file-system concerns). While an export is in flight
/// the button is disabled — the shell serialises exports — and `export.status_line`
/// carries the "Exporting… done/total" progress; otherwise it is the last completion /
/// failure / large-export message.
pub(super) fn build_export_section(ui: &mut egui::Ui, response: &mut PanelResponse, export: ExportPanelState) {
    ui.add_space(8.0);
    ui.strong("Export");
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
