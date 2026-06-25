//! The shared egui side panel.
//!
//! Exactly one implementation, used by both the windowed app and the headless
//! screenshot harness (Hard requirement #3), so the captured frame is identical
//! to the live one. Milestone 1 shows only placeholder section headers — the
//! goal is to confirm egui text and layout render in both render paths. Real
//! widgets (shape chips, size sliders, …) arrive in Milestone 3.

/// Mutable UI state passed to [`build_panel`].
///
/// Empty in Milestone 1 (no functional widgets yet). It exists so the panel
/// signature is stable: later milestones add fields (selected shape, size in
/// blocks, density, display toggles, camera projection) without changing how the
/// binaries call the panel.
#[derive(Debug, Default)]
pub struct PanelState {}

/// Build the right-hand side panel into the root [`egui::Ui`] of the frame.
///
/// Section headers mirror the prototype's panel grouping (see HANDOFF.md "UI"):
/// Shape, Size, Density, Material, Display, Camera.
///
/// Note: in egui 0.34 the top-level `SidePanel::right(..).show(ctx, ..)` form is
/// deprecated in favour of running a root `Ui` (via `Context::run_ui`) and
/// calling `Panel::right(..).show_inside(ui, ..)`. This is the right-hand side
/// panel the spec asks for, expressed in the non-deprecated API.
pub fn build_panel(root_ui: &mut egui::Ui, _state: &mut PanelState) {
    egui::Panel::right("voxel_worker_controls")
        .resizable(false)
        .default_size(260.0)
        .show_inside(root_ui, |ui| {
            ui.add_space(8.0);
            ui.heading("VoxelWorker");
            ui.label("Vintage Story chiseling planner");
            ui.add_space(6.0);
            ui.separator();

            for (section_title, section_hint) in PLACEHOLDER_SECTIONS {
                ui.add_space(8.0);
                ui.strong(*section_title);
                ui.label(*section_hint);
                ui.separator();
            }

            ui.add_space(8.0);
            ui.label("Milestone 1 — foundation only.");
        });
}

/// The placeholder section headers shown in Milestone 1, with a one-line hint of
/// what each will eventually control.
const PLACEHOLDER_SECTIONS: &[(&str, &str)] = &[
    ("Shape", "Cylinder / Tube / Sphere / Torus / Box"),
    ("Size", "X / Y / Z in whole blocks"),
    ("Density", "Voxels per block (chisel fineness)"),
    ("Material", "Stone / Wood / loaded block texture"),
    ("Display", "Voxel grid, block lattice, gizmo, view cube"),
    ("Camera", "Perspective / Orthographic"),
];
