//! The inspector: the per-node editors (Tool / Sketch / Part / Group / Instance)
//! plus the shared shape / size / density / material / offset / grids sub-sections.

use super::palette::SHAPE_CHIPS;
use super::{PanelResponse, PanelState};
use document::intent::Intent;
use document::scene::{CombineOp, NodeContent, Part};
use document::sketch::{Operation, PlaneAxis, RevolveAxis, Sketch, SketchSolid};
use document::voxel::SdfShape;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::units::{self, DisplayUnit, MeasurementError};
use voxel_core::voxel::ShapeKind;

/// The inspector: switches on the active node. A **Tool** shows the shape chips,
/// size sliders, density slider and material selector (editing the active Tool node;
/// ADR 0003 Phase C C4a routes each edit to a `SetShape`/`SetDensity`/`SetMaterial`
/// intent the loop applies). A **Clouds Part** shows its name + seed instead. With no
/// active node, a hint.
pub(super) fn build_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    /// Which inspector to show for the active node.
    enum ActiveKind {
        Tool,
        Sketch,
        Part,
        Group,
        Instance,
        None,
    }
    let kind = match state.scene.active_node().map(|node| &node.content) {
        Some(NodeContent::Tool { .. }) => ActiveKind::Tool,
        // ADR 0003 §3i: a sketch node shows the rectangle-profile editor
        // (Plane / Width / Depth / Height) — or, for a hand-built non-rectangular
        // profile, a read-only note + Plane/Height — plus the shared material /
        // placement / grids sections (see `build_sketch_inspector_section`).
        Some(NodeContent::SketchTool { .. }) => ActiveKind::Sketch,
        Some(NodeContent::Part(_)) => ActiveKind::Part,
        Some(NodeContent::Group(_)) => ActiveKind::Group,
        Some(NodeContent::Instance(_)) => ActiveKind::Instance,
        None => ActiveKind::None,
    };

    match kind {
        ActiveKind::Tool => {
            // ADR 0017: the combine-operation selector shows on LEAF nodes only
            // (Tool / Sketch / Clouds Part) — Group / Instance operations are inert
            // in this sibling-level slice (sealed scopes are issue #74), so they get
            // no selector.
            build_operation_section(ui, state, response);
            // ADR 0003 Phase C C4a: the inspector still binds the widgets to the
            // `geometry`/`material` mirror buffer (egui needs the `&mut`), but a change
            // now EMITS the matching intent instead of calling `write_mirror_to_active`.
            // The active id is known (this is the Tool arm). `SetShape` carries the FULL
            // updated buffer (`from_geometry`) onto the active node — covering both a
            // shape-chip switch (no auto-frame, guard #1) and a size/wall edit
            // (auto-frame). Density is GLOBAL → `SetDensity` (rewrites every Tool);
            // material → `SetMaterial`. The mirror is now ONLY the widget buffer.
            let active = state.scene.active;
            let shape_changed = build_shape_section(ui, state);
            let size_changed = build_size_section(ui, state);
            let density_changed = build_density_section(ui, state);
            let material_changed = build_material_section(ui, state, response);

            if let Some(target) = active {
                // A shape OR size/wall edit rewrites the active Tool's shape from the
                // buffer. Size/wall auto-frames; a pure shape switch does not.
                if shape_changed || size_changed {
                    let shape = SdfShape::from_geometry(state.geometry.clone());
                    let intent = Intent::SetShape { target, shape };
                    if size_changed {
                        response.emit_and_frame(intent);
                    } else {
                        response.emit(intent);
                    }
                }
                // Density is a document-level attribute (ADR 0003 §3f(0)): the slider's
                // transient value drives the single `scene.voxels_per_block` via
                // SetDensity. Auto-frames like a size change.
                if density_changed {
                    response.emit_and_frame(Intent::SetDensity {
                        voxels_per_block: state.geometry.voxels_per_block,
                    });
                }
                // A material pick updates the active Tool's material (no auto-frame).
                if material_changed {
                    response.emit(Intent::SetMaterial { target, material: state.material });
                }
            }
            // Placement (ADR 0001 step 3) is on the node's transform, common to all
            // node kinds; it emits its own `SetOffset` intent.
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Sketch => {
            build_operation_section(ui, state, response);
            build_sketch_inspector_section(ui, state, response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Part => {
            build_operation_section(ui, state, response);
            build_part_inspector_section(ui, state, response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Group => {
            build_group_inspector_section(ui, state, "Group", response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::Instance => {
            build_group_inspector_section(ui, state, "Instance", response);
            build_offset_section(ui, state, response);
            build_node_grids_section(ui, state, response);
        }
        ActiveKind::None => {
            ui.add_space(8.0);
            ui.label(
                egui::RichText::new("Select or add a node to edit it.")
                    .small()
                    .weak(),
            );
            ui.separator();
        }
    }
}

/// Inspector for a Group or Instance active node (ADR 0001 step 4): its name (and,
/// for an Instance, the definition it references). The offset is edited by the
/// shared [`build_offset_section`], so Group/Instance get at least name + offset. ADR
/// 0003 Phase C C4a: the name widget binds to a LOCAL buffer; a change emits `SetName`
/// WITHOUT an auto-frame (the old rename mutated `node.name` with no response flag, so
/// the camera never moved on rename).
fn build_group_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    heading: &str,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong(heading);
    // Capture the def label (immutable borrow) before taking the active node.
    let def_label = match state.scene.active_node().map(|n| &n.content) {
        Some(NodeContent::Instance(def_id)) => state
            .scene
            .def_by_id(*def_id)
            .map(|def| {
                if def.name.is_empty() {
                    format!("Def {}", def.id.0)
                } else {
                    def.name.clone()
                }
            })
            .or_else(|| Some(format!("Def {} (missing)", def_id.0))),
        _ => None,
    };
    if let (Some(target), Some(node)) = (state.scene.active, state.scene.active_node()) {
        let mut name = node.name.clone();
        ui.horizontal(|ui| {
            ui.label("Name");
            // A rename did NOT auto-frame in the old code (it mutated `node.name`
            // with no response flag), so emit WITHOUT a frame. The `SetName` effect is
            // `scene_changed`, so the loop re-resolves (an identical grid — the name is
            // not geometry) but the camera stays put, matching the old visible result.
            if ui.text_edit_singleline(&mut name).changed() {
                response.emit(Intent::SetName { target, name: name.clone() });
            }
        });
    }
    if let Some(label) = def_label {
        ui.label(
            egui::RichText::new(format!("references: {label}"))
                .small()
                .weak(),
        );
    }
    ui.separator();
}

/// Inspector for a Clouds Part active node: its name and seed (its one knob). A
/// seed change re-resolves the scene. ADR 0003 Phase C C4a: the name/seed widgets bind
/// to LOCAL buffers (read from the active node each frame); a change emits `SetName` /
/// `SetCloudSeed` instead of mutating the node. A seed edit auto-frames like the old
/// `scene_changed`.
fn build_part_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    ui.strong("Clouds (Part)");
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    let mut name = node.name.clone();
    let current_seed = match &node.content {
        NodeContent::Part(Part::DebugClouds { seed }) => Some(*seed),
        _ => None,
    };
    ui.horizontal(|ui| {
        ui.label("Name");
        if ui.text_edit_singleline(&mut name).changed() {
            response.emit(Intent::SetName { target, name: name.clone() });
        }
    });
    if let Some(seed) = current_seed {
        let mut value = seed;
        if ui
            .add(egui::Slider::new(&mut value, 0..=64).text("seed"))
            .changed()
        {
            response.emit_and_frame(Intent::SetCloudSeed { target, seed: value });
        }
    }
    ui.separator();
}

/// Inspector for a sketch→solid active node (ADR 0003 §3i): edits the node's
/// [`SketchSolid`] producer. A change rebuilds the whole producer and emits a
/// `SetSketch` (auto-framed, since the solid's AABB — and thus the composite extent —
/// changes), then the shared material section emits `SetMaterial`. The offset / grids
/// sections are appended by the caller, common to all node kinds.
///
/// An **Operation** picker (Extrude / Revolve) selects how the 2D profile becomes a
/// volume; the editor is OPERATION-AWARE so editing a Revolve node rebuilds a Revolve
/// (it never silently clobbers to Extrude — the rebuild branches on the
/// CURRENTLY-SELECTED operation, not on a hardcoded `extrude`).
///
/// The **Plane** picker and the rectangle Width/Depth detection are
/// operation-independent (the profile is the same shape either way). Each operation
/// then adds its own controls:
///
///   * **Extrude**: a **Height (vx)** field (the extrude span along the plane normal).
///     Rebuilds `SketchSolid::extrude`.
///   * **Revolve**: a **RevolveAxis** picker (the two in-plane world axes, labelled in
///     world-axis terms, e.g. "about X" / "about Y") plus a **Turn (deg)** field
///     (`1..=360`, default 360). Rebuilds `SketchSolid::revolve`.
///
/// Two profile modes, by shape (independent of operation):
///
///   * **Rectangle** (the Add-menu default): editable **Width** / **Depth** (the two
///     in-plane spans, along the plane's [`in_plane_axes`]). A rebuild regenerates a
///     fresh `Sketch::rectangle` on the chosen plane at the edited spans.
///   * **Custom profile** (a hand-built polygon — not authorable from the UI yet, but
///     it can exist in code/tests): a read-only "Custom profile (N points)" note. A
///     rebuild PRESERVES the existing profile points (swapping only plane / operation
///     parameters), so a hand-built polygon is never clobbered into a rectangle.
///
/// DEFERRED (ADR 0003 §3i, Slices 2b/2c): free-polyline point add/move/delete editing,
/// the sweep producer, and on-surface sketching are not built here.
///
/// [`in_plane_axes`]: document::sketch::PlaneAxis::in_plane_axes
fn build_sketch_inspector_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let Some(target) = state.scene.active else {
        return;
    };
    // Read the active node's producer (clone so the borrow of the scene ends before
    // the material section, which takes `&mut state`).
    let Some(producer) = state.scene.active_node().and_then(|node| match &node.content {
        NodeContent::SketchTool { producer, .. } => Some(producer.clone()),
        _ => None,
    }) else {
        return;
    };

    // Which operation kind the active producer currently is — drives the heading and
    // seeds the Operation picker. A SWITCH to the other kind carries sensible defaults
    // (see `OperationKind` / the rebuild below).
    #[derive(PartialEq, Eq, Clone, Copy)]
    enum OperationKind {
        Extrude,
        Revolve,
    }
    let current_kind = match producer.operation {
        Operation::Extrude { .. } => OperationKind::Extrude,
        Operation::Revolve { .. } => OperationKind::Revolve,
    };

    ui.add_space(8.0);
    ui.strong(match current_kind {
        OperationKind::Extrude => "Sketch → Extrude",
        OperationKind::Revolve => "Sketch → Revolve",
    });

    // The label for a plane choice in the picker.
    fn plane_label(plane: PlaneAxis) -> &'static str {
        match plane {
            PlaneAxis::Z => "Z (XY footprint, up)",
            PlaneAxis::X => "X (YZ profile)",
            PlaneAxis::Y => "Y (XZ profile)",
        }
    }

    // The world-axis letter (X/Y/Z) for a world-axis index — used to label the
    // RevolveAxis options in world terms the user can reason about.
    fn world_axis_letter(axis_index: usize) -> char {
        ['X', 'Y', 'Z'][axis_index]
    }

    // The RevolveAxis picker label: which world axis the profile is revolved ABOUT.
    // `RevolveAxis::InPlane0` revolves about `in_plane_axes()[0]`, `InPlane1` about
    // `[1]`, so the label is derived from the plane's in-plane axes (e.g. for the Z
    // footprint plane the in-plane axes are X, Y → "about X" / "about Y").
    fn revolve_axis_label(plane: PlaneAxis, axis: RevolveAxis) -> String {
        let in_plane = plane.in_plane_axes();
        let world_index = match axis {
            RevolveAxis::InPlane0 => in_plane[0],
            RevolveAxis::InPlane1 => in_plane[1],
        };
        format!("about {}", world_axis_letter(world_index))
    }

    let mut plane = producer.sketch.plane;
    let mut kind = current_kind;
    // Per-operation parameters, seeded from the active producer. The OTHER kind's
    // parameters seed from sensible defaults so an operation SWITCH carries them.
    let mut height_voxels = match producer.operation {
        Operation::Extrude { height_voxels } => height_voxels,
        Operation::Revolve { .. } => 0,
    };
    let (mut revolve_axis, mut turn_degrees) = match producer.operation {
        Operation::Revolve { axis, sweep } => (axis, sweep.turn_degrees),
        Operation::Extrude { .. } => (RevolveAxis::InPlane0, 360),
    };
    let mut changed = false;

    // Operation picker — seeds from the current kind. A change flips `kind`; the
    // rebuild below branches on `kind`, so a Revolve stays a Revolve on edit.
    egui::ComboBox::from_label("Operation")
        .selected_text(match kind {
            OperationKind::Extrude => "Extrude",
            OperationKind::Revolve => "Revolve",
        })
        .show_ui(ui, |ui| {
            changed |= ui
                .selectable_value(&mut kind, OperationKind::Extrude, "Extrude")
                .changed();
            changed |= ui
                .selectable_value(&mut kind, OperationKind::Revolve, "Revolve")
                .changed();
        });

    // Plane picker — common to both operations.
    egui::ComboBox::from_label("Plane")
        .selected_text(plane_label(plane))
        .show_ui(ui, |ui| {
            for option in [PlaneAxis::Z, PlaneAxis::X, PlaneAxis::Y] {
                if ui
                    .selectable_value(&mut plane, option, plane_label(option))
                    .changed()
                {
                    changed = true;
                }
            }
        });

    // Rectangle profiles expose editable Width/Depth; a custom polygon is read-only.
    // Operation-independent — the profile shape is the same whether extruded or revolved.
    let rectangle_spans = producer.rectangle_in_plane_spans();
    let mut width_voxels = rectangle_spans.map(|spans| spans[0]).unwrap_or(1).max(1);
    let mut depth_voxels = rectangle_spans.map(|spans| spans[1]).unwrap_or(1).max(1);

    if rectangle_spans.is_some() {
        ui.horizontal(|ui| {
            ui.label("Width (vx)");
            changed |= ui
                .add(egui::DragValue::new(&mut width_voxels).speed(1.0).range(1..=u32::MAX))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("Depth (vx)");
            changed |= ui
                .add(egui::DragValue::new(&mut depth_voxels).speed(1.0).range(1..=u32::MAX))
                .changed();
        });
    } else {
        // A hand-built polygon: read-only note, no Width/Depth (editing them would
        // mean discarding the profile). Only the plane + operation params are editable.
        ui.label(
            egui::RichText::new(format!(
                "Custom profile ({} points)",
                producer.sketch.profile.len()
            ))
            .small()
            .weak(),
        );
    }

    // Per-operation controls.
    match kind {
        OperationKind::Extrude => {
            ui.horizontal(|ui| {
                ui.label("Height (vx)");
                changed |= ui
                    .add(egui::DragValue::new(&mut height_voxels).speed(1.0).range(1..=u32::MAX))
                    .changed();
            });
        }
        OperationKind::Revolve => {
            egui::ComboBox::from_label("Revolve axis")
                .selected_text(revolve_axis_label(plane, revolve_axis))
                .show_ui(ui, |ui| {
                    for option in [RevolveAxis::InPlane0, RevolveAxis::InPlane1] {
                        if ui
                            .selectable_value(
                                &mut revolve_axis,
                                option,
                                revolve_axis_label(plane, option),
                            )
                            .changed()
                        {
                            changed = true;
                        }
                    }
                });
            ui.horizontal(|ui| {
                ui.label("Turn (deg)");
                changed |= ui
                    .add(egui::DragValue::new(&mut turn_degrees).speed(1.0).range(1..=360))
                    .changed();
            });
        }
    }

    if changed {
        // Rebuild the producer. A rectangle profile is regenerated at the edited spans;
        // a custom profile is PRESERVED (only plane / operation params swap), so a
        // hand-built polygon is never clobbered into a rectangle.
        let sketch = if rectangle_spans.is_some() {
            Sketch::rectangle(plane, width_voxels as i64, depth_voxels as i64)
        } else {
            Sketch::new(plane, producer.sketch.profile.clone())
        };
        // Branch on the CURRENTLY-SELECTED operation so editing a Revolve rebuilds a
        // Revolve (no clobber to Extrude). An operation SWITCH carries the defaults
        // seeded above: Extrude→Revolve uses InPlane0 + a full 360° turn; Revolve→
        // Extrude uses a sensible height (the rectangle's depth span if available,
        // else a default 16 — the user adjusts after).
        let rebuilt = match kind {
            OperationKind::Extrude => {
                // When switching FROM Revolve the seeded height is 0 (no extrude span
                // existed); fall back to the rectangle depth, else a default 16.
                let height = if height_voxels >= 1 {
                    height_voxels
                } else {
                    rectangle_spans.map(|spans| spans[1]).unwrap_or(16).max(1)
                };
                SketchSolid::extrude(sketch, height)
            }
            OperationKind::Revolve => SketchSolid::revolve(sketch, revolve_axis, turn_degrees),
        };
        response.emit_and_frame(Intent::SetSketch { target, producer: rebuilt });
    }

    ui.separator();

    // Shared material section (emits SetMaterial). It binds to `state.material`, which
    // the loop syncs from the active node on selection, so it reflects this sketch's
    // material; a pick emits `SetMaterial` (which the dispatch applies to a sketch
    // node's shared material field).
    if build_material_section(ui, state, response) {
        response.emit(Intent::SetMaterial { target, material: state.material });
    }
}

/// Combine-operation selector (ADR 0017): how the active LEAF node folds into the
/// result accumulated before it among its siblings — `Union` adds (later-wins
/// material on overlap), `Subtract` carves (an occupancy-only mask that never
/// stamps material). Shown ONLY on leaf nodes (Tool / Sketch / Clouds Part); Group
/// / Instance operations are inert in this sibling-level slice (sealed scopes are
/// issue #74). A change emits `Intent::SetOperation` WITHOUT an auto-frame (a
/// cutter flip never changes the composite extent — the cutter's AABB already
/// contributes to it — so the camera stays put, like a material pick).
fn build_operation_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    let current = node.operation;

    /// The selector label for a combine operation.
    fn operation_label(operation: CombineOp) -> &'static str {
        match operation {
            CombineOp::Union => "Union",
            CombineOp::Subtract => "Subtract",
        }
    }

    ui.add_space(8.0);
    ui.strong("Operation");
    let mut selected = current;
    egui::ComboBox::from_id_salt(("node_combine_operation", target))
        .selected_text(operation_label(selected))
        .show_ui(ui, |ui| {
            for option in [CombineOp::Union, CombineOp::Subtract] {
                ui.selectable_value(&mut selected, option, operation_label(option));
            }
        });
    if selected != current {
        response.emit(Intent::SetOperation { target, operation: selected });
    }
    ui.separator();
}

/// Offset (placement) section (ADR 0003 §3f(0)): three per-axis text fields
/// (X/Y/Z, signed) accepting blocks+voxels unit expressions (e.g. `"3 blocks 8
/// voxels"`, `"-1b 4v"`, `"3.5 blocks"`). Each field is seeded from the canonical
/// voxel offset formatted as blocks+voxels and, on commit (Enter or focus loss),
/// parsed via [`units::parse`] and validated to land on a whole voxel at the
/// document density; on success it emits a single `SetOffset` carrying the
/// per-axis [`Measurement`](voxel_core::units::Measurement)s (the edited axis plus the
/// two unchanged retained ones). A parse / non-landing error is shown inline (red)
/// and NOTHING is emitted, so the canonical offset never moves on bad input.
///
/// Common to Tools and Parts — placement is on the node's transform, not the
/// producer. A committed edit re-resolves + re-frames the composite (a node moving
/// changes the composite extent), so it auto-frames the whole composited extent.
///
/// The in-progress text + last error live in egui temp memory (keyed per axis by a
/// stable `Id`) so a partial edit and its error survive across frames; an unfocused
/// field re-syncs to the canonical value, so undo / external moves reflect.
fn build_offset_section(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    let density = state.scene.voxels_per_block;
    // The canonical voxel offset (resolve's source of truth) and the RETAINED
    // per-axis measurements (the two unedited axes ride along unchanged in any
    // emitted intent so a single-axis edit does not disturb the others).
    let offset_voxels = node.transform.offset_voxels;
    let retained_measurements = node.transform.offset_measurements();

    ui.add_space(8.0);
    ui.strong("Offset (blocks + voxels)");

    for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
        // Per-axis stable ids for the in-progress text buffer and last error.
        let text_id = egui::Id::new(("offset_axis_text", target, axis_index));
        let error_id = egui::Id::new(("offset_axis_error", target, axis_index));
        // The canonical seed: the current voxel offset as a blocks+voxels string.
        let seed = units::format(offset_voxels[axis_index], density, DisplayUnit::BlocksAndVoxels);

        // The text edit binds to a LOCAL buffer restored from temp memory; an
        // UNFOCUSED field re-syncs to the canonical seed so undo / external moves
        // and density changes reflect, while a focused field keeps the user's
        // in-progress text untouched.
        let mut buffer = ui
            .memory(|memory| memory.data.get_temp::<String>(text_id))
            .unwrap_or_else(|| seed.clone());

        let widget = egui::TextEdit::singleline(&mut buffer)
            .desired_width(110.0)
            .hint_text("blocks + voxels");
        let widget_response = ui.horizontal(|ui| {
            ui.label(format!("{axis_label} "));
            ui.add(widget)
        });
        let edit_response = widget_response.inner;

        // Editing again clears any stale error from a prior failed commit, so the
        // red message tracks the LAST committed attempt, not in-progress typing.
        if edit_response.changed() {
            ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
        }

        // `lost_focus()` fires on Enter AND on click-away, so it is the single
        // commit trigger; the typed `buffer` is still live here (the unfocused
        // re-sync happens AFTER, so a commit reads the user's text, not the seed).
        // Only attempt a parse when the text actually differs from the canonical
        // seed (a focus loss with no edit is a no-op).
        let committed = edit_response.lost_focus() && buffer.trim() != seed;
        if committed {
            match units::parse(&buffer) {
                Ok(measurement) => match measurement.to_voxels(density) {
                    Ok(landed_voxels) => {
                        // Replace only this axis; the other two keep their retained
                        // measurements so a single-axis edit is isolated.
                        let mut next = retained_measurements;
                        next[axis_index] = measurement;
                        ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                        response.emit_and_frame(Intent::SetOffset {
                            target,
                            offset_measurements: next,
                        });
                        // Settle the field on the canonical form of the applied value.
                        buffer =
                            units::format(landed_voxels, density, DisplayUnit::BlocksAndVoxels);
                    }
                    Err(error) => {
                        ui.memory_mut(|memory| {
                            memory.data.insert_temp(error_id, measurement_error_text(&error))
                        });
                    }
                },
                Err(error) => {
                    ui.memory_mut(|memory| {
                        memory.data.insert_temp(error_id, error.to_string())
                    });
                }
            }
        } else if !edit_response.has_focus() {
            // Not being edited and not a commit. If a prior commit FAILED, an error
            // is stored — keep the user's (rejected) text on screen alongside the
            // error so they can see and fix it; do NOT silently revert. With no
            // error, mirror the canonical value so undo / external moves / density
            // changes reflect in the field.
            let has_error = ui.memory(|memory| memory.data.get_temp::<String>(error_id).is_some());
            if !has_error {
                buffer = seed.clone();
            }
        }

        // Persist the buffer for the next frame (the focused, in-progress text).
        ui.memory_mut(|memory| memory.data.insert_temp(text_id, buffer));

        // Inline error (red), cleared on the next successful commit.
        if let Some(message) = ui.memory(|memory| memory.data.get_temp::<String>(error_id)) {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), message);
        }
    }

    ui.separator();
}

/// Render a [`MeasurementError`] for an inline unit-field error label (Offset and
/// Size both use it). A non-landing block fraction reports the nearest representable
/// voxel counts so the user can pick one instead of being silently rounded (ADR 0003
/// §3f(0)).
fn measurement_error_text(error: &MeasurementError) -> String {
    match error {
        MeasurementError::BlockTermNotWholeVoxels {
            density,
            nearest_floor_voxels,
            nearest_ceil_voxels,
        } => format!(
            "doesn't land on a whole voxel at density {density}; nearest are {nearest_floor_voxels} or {nearest_ceil_voxels} voxels"
        ),
        MeasurementError::ZeroDensity => "density must be at least 1".to_string(),
    }
}

/// Per-node grid toggles (issue #29 S3/S4): the active node's own
/// `voxel_grid_on_faces` / `block_lattice` / `floor_grid` flags, each ANDed with
/// its scene-wide master (in the Display section) to decide whether that node draws
/// the grid.
///
/// The block-lattice / floor toggles only need a per-frame batch rebuild — those
/// lines are re-walked from the scene every frame — so they signal NO scene
/// re-resolve, keeping a grid flip cheap. The **voxel-grid-on-faces** toggle (S4) is
/// different: the on-face-grid flag bit is baked onto each voxel's `material_id` at
/// RESOLVE time (so it survives chunk bucketing and the cuboid box-decomposition
/// key), so flipping it MUST re-resolve the scene — it signals `scene_changed`.
fn build_node_grids_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    let Some(target) = state.scene.active else {
        return;
    };
    let Some(node) = state.scene.active_node() else {
        return;
    };
    ui.add_space(8.0);
    ui.strong("Grids (this object)");
    // ADR 0003 Phase C C4a: the three checkboxes bind to a LOCAL copy of the node's
    // grids; a change emits ONE `SetNodeGrids` carrying all three. The on-face-grid
    // flag is baked at RESOLVE time, so flipping it must re-resolve AND it auto-framed
    // before (it set `scene_changed`); the lattice/floor flags are read by the
    // per-frame line batch, so they did NOT auto-frame. We therefore auto-frame ONLY
    // when `voxel_grid_on_faces` flips. (`SetNodeGrids`'s effect is `scene_changed`, so
    // a lattice/floor toggle now also re-resolves — an identical grid, no camera move,
    // so the visible result is unchanged; the cost is a redundant re-resolve.)
    let mut grids = node.grids;
    let mut voxel_grid_changed = false;
    let mut other_changed = false;
    voxel_grid_changed |= ui
        .checkbox(&mut grids.voxel_grid_on_faces, "Voxel grid on faces")
        .changed();
    other_changed |= ui.checkbox(&mut grids.block_lattice, "Block lattice").changed();
    other_changed |= ui.checkbox(&mut grids.floor_grid, "Floor grid").changed();
    if voxel_grid_changed {
        response.emit_and_frame(Intent::SetNodeGrids { target, grids });
    } else if other_changed {
        response.emit(Intent::SetNodeGrids { target, grids });
    }
    ui.separator();
}

/// Shape chips. Selecting a shape sets [`GeometryParams::shape`] ONLY — it never
/// touches the size or the camera (Milestone 3 guard #1). Shown only for a Tool
/// active node. ADR 0003 Phase C C4a: returns `true` when the buffer's shape changed
/// (the inspector then emits a `SetShape` WITHOUT an auto-frame).
///
/// [`GeometryParams::shape`]: document::voxel::GeometryParams::shape
fn build_shape_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Shape");
    let mut changed = false;
    ui.horizontal_wrapped(|ui| {
        for (kind, label) in SHAPE_CHIPS {
            let is_selected = state.geometry.shape == *kind;
            if ui.selectable_label(is_selected, *label).clicked() && !is_selected {
                state.geometry.shape = *kind;
                changed = true;
                // The caller emits a `SetShape` but no auto-frame: a shape switch
                // re-resolves at the same size and must not move the camera.
            }
        }
    });
    ui.separator();
    changed
}

/// Size section (ADR 0003 §3f(0)): three per-axis text fields (X/Y/Z) accepting
/// blocks+voxels unit expressions (e.g. `"5 blocks"`, `"5b 8v"`, `"83 voxels"`),
/// mirroring [`build_offset_section`]. Each field is seeded from the canonical voxel
/// size formatted as blocks+voxels and, on commit (Enter or focus loss), parsed via
/// [`units::parse`] and validated to land on a whole voxel `>= 1` at the document
/// density. On success it writes the edited axis's canonical voxels + retained
/// measurement into the [`GeometryParams`](document::voxel::GeometryParams) mirror
/// (the OTHER two axes keep their retained measurements — single-axis isolation) and
/// returns `true`, so the inspector emits a `SetShape` (built via
/// [`SdfShape::from_geometry`]) AND auto-frames. A parse / non-landing / sub-1 error
/// is shown inline (red) and the size is NOT changed.
///
/// The in-progress text + last error live in egui temp memory (keyed per axis by a
/// stable `Id`) so a partial edit and its error survive across frames; an unfocused
/// field with no error re-syncs to the canonical value, so undo / external edits /
/// density changes reflect.
fn build_size_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Size (blocks + voxels)");

    let mut changed = false;
    let density = state.geometry.voxels_per_block;
    // The canonical voxel size and the RETAINED per-axis measurements (the two
    // unedited axes ride along unchanged so a single-axis edit is isolated).
    let size_voxels = state.geometry.size_voxels;
    let retained_measurements = match &state.geometry.size_measurements {
        Some(measurements) => **measurements,
        None => [
            units::Measurement::from_voxels(size_voxels[0] as i64),
            units::Measurement::from_voxels(size_voxels[1] as i64),
            units::Measurement::from_voxels(size_voxels[2] as i64),
        ],
    };
    // A stable per-active-node key prefix so each selection gets its own buffers
    // (a re-selection re-seeds, like the offset section keys on `target`).
    let key = state.scene.active;

    for (axis_index, axis_label) in ["X", "Y", "Z"].iter().enumerate() {
        let text_id = egui::Id::new(("size_axis_text", key, axis_index));
        let error_id = egui::Id::new(("size_axis_error", key, axis_index));
        let seed = units::format(size_voxels[axis_index] as i64, density, DisplayUnit::BlocksAndVoxels);

        let mut buffer = ui
            .memory(|memory| memory.data.get_temp::<String>(text_id))
            .unwrap_or_else(|| seed.clone());

        let widget = egui::TextEdit::singleline(&mut buffer)
            .desired_width(110.0)
            .hint_text("blocks + voxels");
        let widget_response = ui.horizontal(|ui| {
            ui.label(format!("{axis_label} "));
            ui.add(widget)
        });
        let edit_response = widget_response.inner;

        if edit_response.changed() {
            ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
        }

        let committed = edit_response.lost_focus() && buffer.trim() != seed;
        if committed {
            match units::parse(&buffer) {
                Ok(measurement) => match measurement.to_voxels(density) {
                    Ok(landed_voxels) if landed_voxels >= 1 => {
                        // Replace only this axis; the other two keep their retained
                        // measurements so a single-axis edit is isolated.
                        let mut next = retained_measurements;
                        next[axis_index] = measurement;
                        state.geometry.size_voxels[axis_index] = landed_voxels as u32;
                        state.geometry.size_measurements = Some(Box::new(next));
                        changed = true;
                        ui.memory_mut(|memory| memory.data.remove::<String>(error_id));
                        buffer = units::format(landed_voxels, density, DisplayUnit::BlocksAndVoxels);
                    }
                    Ok(_) => {
                        ui.memory_mut(|memory| {
                            memory
                                .data
                                .insert_temp(error_id, "size must be at least 1 voxel".to_string())
                        });
                    }
                    Err(error) => {
                        ui.memory_mut(|memory| {
                            memory.data.insert_temp(error_id, measurement_error_text(&error))
                        });
                    }
                },
                Err(error) => {
                    ui.memory_mut(|memory| memory.data.insert_temp(error_id, error.to_string()));
                }
            }
        } else if !edit_response.has_focus() {
            let has_error = ui.memory(|memory| memory.data.get_temp::<String>(error_id).is_some());
            if !has_error {
                buffer = seed.clone();
            }
        }

        ui.memory_mut(|memory| memory.data.insert_temp(text_id, buffer));

        if let Some(message) = ui.memory(|memory| memory.data.get_temp::<String>(error_id)) {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), message);
        }
    }

    // Conditional wall row — Tube only.
    if state.geometry.shape == ShapeKind::Tube {
        ui.add_space(4.0);
        let mut wall = state.geometry.wall_blocks;
        let slider = egui::Slider::new(&mut wall, 1..=8).text("Wall");
        if ui.add(slider).changed() {
            state.geometry.wall_blocks = wall;
            changed = true;
        }
        ui.label(
            egui::RichText::new(format!("{wall} block wall"))
                .small()
                .weak(),
        );
    }
    ui.separator();
    changed
}

/// Density slider. Changes fineness ONLY — never the block size (guard #2). ADR 0003
/// Phase C C4a: returns `true` when the buffer's density changed (the inspector then
/// emits a global `SetDensity` AND auto-frames).
fn build_density_section(ui: &mut egui::Ui, state: &mut PanelState) -> bool {
    ui.add_space(8.0);
    ui.strong("Density");
    let mut density = state.geometry.voxels_per_block;
    let slider = egui::Slider::new(&mut density, 2..=32).text("vx/block");
    let changed = ui.add(slider).changed();
    if changed {
        state.geometry.voxels_per_block = density;
    }
    ui.separator();
    changed
}

/// Material selector: selects which procedural texture binds (M4). Selecting any
/// procedural material clears an applied loaded VS block (M6) and reverts to it. ADR
/// 0003 Phase C C4a: returns `true` when the buffer's material changed (the inspector
/// then emits a `SetMaterial`). Still sets `selected_procedural_material` for the
/// caller's M6 palette side-effect (clearing the applied loaded block — NOT a scene
/// mutation, so it stays a response flag, not an intent).
fn build_material_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) -> bool {
    ui.add_space(8.0);
    ui.strong("Material");
    let mut changed = false;
    ui.horizontal(|ui| {
        for (choice, label) in [
            (MaterialChoice::Stone, "Stone"),
            (MaterialChoice::Wood, "Wood"),
            (MaterialChoice::Plain, "Plain"),
        ] {
            if ui.selectable_value(&mut state.material, choice, label).clicked() {
                response.selected_procedural_material = true;
                changed = true;
            }
        }
    });
    if let Some(applied) = &state.applied_block_label {
        ui.label(
            egui::RichText::new(format!("Applied: {applied}"))
                .small()
                .weak(),
        );
    }
    ui.separator();
    changed
}
