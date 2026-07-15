//! The **Points** section (issue #29 S5): the world reference grid's frames.

use super::{PanelResponse, PanelState};
use document::intent::Intent;

/// The **Points** section (issue #29 S5): the world reference grid's frames. Lists
/// every [`Point`](document::scene::Point) with a visibility checkbox (bound to
/// `!hidden`) and a selectable name; **+ Add Point** appends a Point at the camera
/// target (falling back to the origin); and — for the selected Point — XZ/XY/YZ plane
/// checkboxes, per-axis X/Y/Z checkboxes, a whole-block position editor (HIDDEN for the
/// Origin), and a **Delete** button (hidden for the Origin, which is undeletable).
/// Mirrors the node list's deferred-mutation pattern: selection/delete are applied
/// AFTER the read walk.
pub(super) fn build_points_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    // A scene with NO Points (the headless `shot` path builds scenes WITHOUT the
    // synthesized Origin — `ensure_origin_point` runs only on the windowed load/seed
    // path) renders nothing here, so the section adds zero height and the existing
    // goldens stay byte-identical. The windowed app always carries the Origin Point,
    // so the section always shows there.
    if state.scene.points.is_empty() {
        return;
    }
    ui.add_space(8.0);
    ui.strong("Points");

    let mut select: Option<usize> = None;
    let mut delete: Option<usize> = None;
    let mut toggle_hidden: Option<usize> = None;

    // The Point rows: a visibility checkbox (bound to `!hidden`) + a selectable name.
    for index in 0..state.scene.points.len() {
        let (name, hidden, is_active) = {
            let point = &state.scene.points[index];
            let name = if point.name.is_empty() {
                format!("Point {index}")
            } else {
                point.name.clone()
            };
            (name, point.hidden, state.scene.active_point == Some(index))
        };
        ui.horizontal(|ui| {
            // Visibility is `!hidden`; toggling it flips the Point's `hidden` flag.
            let mut visible = !hidden;
            if ui.checkbox(&mut visible, "").on_hover_text("Visible").changed() {
                toggle_hidden = Some(index);
            }
            if ui.selectable_label(is_active, name).clicked() {
                select = Some(index);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // The Origin is undeletable — no ✕ button for it.
                if !state.scene.points[index].is_origin
                    && ui.small_button("✕").on_hover_text("Delete point").clicked()
                {
                    delete = Some(index);
                }
            });
        });
    }

    // + Add Point — a fresh Point at the camera target (whole blocks), else the
    // origin. ADR 0003 Phase C C4a: described as an `AddPoint` intent; the panel
    // names it after the soon-to-be index (matching the old `format!`), and emits a
    // trailing `SelectPoint` so the new Point becomes active (the old
    // `active_point = len - 1`, which `add_point` itself does not set).
    if ui
        .button("+ Add Point")
        .on_hover_text("Add a reference Point at the camera target")
        .clicked()
    {
        let new_index = state.scene.points.len();
        response.emit(Intent::AddPoint {
            position_blocks: state.point_add_position_blocks,
            name: format!("Point {new_index}"),
        });
        response.emit(Intent::SelectPoint { target: Some(new_index) });
    }

    // The selected Point's editor: plane/axis toggles, position (hidden for Origin),
    // and a delete button (hidden for the Origin). ADR 0003 Phase C C4a: each widget
    // binds to a LOCAL copy of the Point's fields (egui needs the `&mut`); a change
    // emits the matching `SetPoint*` intent instead of mutating the Point. The buffer
    // is read fresh from the scene each frame, so it always reflects the live value.
    if let Some(active) = state.scene.active_point {
        if let Some(point) = state.scene.points.get(active) {
            let point = point.clone();
            ui.add_space(4.0);
            ui.separator();

            // Plane toggles → `SetPointPlanes` (carrying all three current values).
            // Z-up: the GROUND plane is XY (normal +Z) = the `plane_xy` flag; the
            // FRONT plane is XZ (normal +Y) = `plane_xz`; the SIDE plane is YZ.
            let mut plane_xz = point.plane_xz;
            let mut plane_xy = point.plane_xy;
            let mut plane_yz = point.plane_yz;
            let mut planes_changed = false;
            planes_changed |= ui.checkbox(&mut plane_xy, "Ground plane (XY)").changed();
            planes_changed |= ui.checkbox(&mut plane_xz, "Front plane (XZ)").changed();
            planes_changed |= ui.checkbox(&mut plane_yz, "Side plane (YZ)").changed();
            if planes_changed {
                response.emit(Intent::SetPointPlanes {
                    index: active,
                    xz: plane_xz,
                    xy: plane_xy,
                    yz: plane_yz,
                });
            }

            // Per-axis toggles (issue #29 fix): X/Y/Z each toggle independently →
            // `SetPointAxes` (carrying all three).
            let mut axis_x = point.axis_x;
            let mut axis_y = point.axis_y;
            let mut axis_z = point.axis_z;
            let mut axes_changed = false;
            ui.horizontal(|ui| {
                ui.label("Axes");
                axes_changed |= ui.checkbox(&mut axis_x, "X").changed();
                axes_changed |= ui.checkbox(&mut axis_y, "Y").changed();
                axes_changed |= ui.checkbox(&mut axis_z, "Z").changed();
            });
            if axes_changed {
                response.emit(Intent::SetPointAxes {
                    index: active,
                    x: axis_x,
                    y: axis_y,
                    z: axis_z,
                });
            }

            // Position editor — only for a user Point (the Origin is pinned at world 0).
            if !point.is_origin {
                let mut position = point.position_blocks;
                let mut position_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Pos (blocks)");
                    for axis_value in &mut position {
                        position_changed |= ui
                            .add(egui::DragValue::new(axis_value).speed(1.0))
                            .changed();
                    }
                });
                if position_changed {
                    response.emit(Intent::SetPointPosition {
                        index: active,
                        position_blocks: position,
                    });
                }
                if ui.button("Delete point").clicked() {
                    delete = Some(active);
                }
            } else {
                ui.label(
                    egui::RichText::new("Origin — pinned at world origin, undeletable")
                        .small()
                        .weak(),
                );
            }
        }
    }

    // Apply deferred mutations after the read/borrow walk. ADR 0003 Phase C C4a: each
    // is described as an intent the loop applies.
    if let Some(index) = toggle_hidden {
        // The visibility checkbox is bound to `!hidden`; a toggle flips it. Read the
        // current flag and emit the explicit `SetPointHidden` for the new value (the
        // intent path is explicit, unlike `toggle_point_hidden`'s flip).
        if let Some(point) = state.scene.points.get(index) {
            response.emit(Intent::SetPointHidden { index, hidden: !point.hidden });
        }
    }
    if let Some(index) = delete {
        // `RemovePoint` is a no-op on the Origin (the UI already hides its delete
        // affordances). To preserve the old `active_point` fix-up (which `remove_point`
        // does not do), emit a trailing `SelectPoint` re-deriving the selection.
        let was_origin = state.scene.points.get(index).map(|p| p.is_origin).unwrap_or(false);
        if !was_origin {
            response.emit(Intent::RemovePoint { index });
            // After removing index, the list shrinks by one: re-derive the selection
            // exactly as the old code did (clamp to the new last, or clear if empty).
            let remaining = state.scene.points.len().saturating_sub(1);
            let next = if remaining == 0 {
                None
            } else {
                Some(index.min(remaining - 1))
            };
            response.emit(Intent::SelectPoint { target: next });
        }
    } else if let Some(index) = select {
        if state.scene.active_point != Some(index) {
            response.emit(Intent::SelectPoint { target: Some(index) });
        }
    }

    ui.separator();
}
