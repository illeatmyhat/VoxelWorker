//! The scene node-list section: the tree of nodes, the add/group/definition
//! actions, and the definitions list.

use super::palette::SHAPE_CHIPS;
use super::{PanelResponse, PanelState};
use crate::theme;
use document::intent::{Intent, NodeSpec};
use document::scene::{DefId, Node, NodeContent, NodeId, VoxelBody, ROOT_NODE_ID};
use document::sketch::{PlaneAxis, Sketch, SketchSolid};
use document::voxel::SdfShape;
use voxel_core::voxel::ShapeKind;

/// A label for a node row, switching on its content kind. Falls back to the
/// content-kind name when the node's own name is empty.
fn node_row_label(node: &Node) -> String {
    let kind = match &node.content {
        NodeContent::Tool { shape, .. } => format!("{:?}", shape.kind),
        NodeContent::SketchTool { .. } => "Sketch".to_string(),
        NodeContent::VoxelBody(VoxelBody::DebugClouds { .. }) => "Clouds".to_string(),
        NodeContent::Group(children) => format!("Part ({})", children.len()),
        NodeContent::Instance(_) => "Instance".to_string(),
    };
    if node.name.is_empty() {
        kind
    } else {
        format!("{} · {}", node.name, kind)
    }
}

/// The scene node list (ADR 0001 step 4): the assembly rendered as an INDENTED
/// TREE so [`Group`](NodeContent::Group) children are visible and selectable at
/// any depth (not just top-level nodes). Each row carries a visibility checkbox, a
/// selectable name (indented by depth), and a per-row delete ✕. Beneath the tree:
///
///   * **+ Add** — append a Tool (any shape) or a Clouds VoxelBody at top level.
///   * **Part** — wrap the active node in a new part (then add children to it
///     via "+ Add child").
///   * **+ Add child** — when the active node is a Group, append a Tool/VoxelBody into
///     it.
///   * **Make definition** — turn the active Group/node into a reusable
///     [`AssemblyDef`] and replace it with an `Instance` of it.
///   * a **Definitions** list with an **Add instance** button per definition (the
///     village workflow: one stored body, many placements).
///
/// [`AssemblyDef`]: document::scene::AssemblyDef
///
/// Selecting any node (at any depth) makes it the inspector's active node.
pub(super) fn build_node_list_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    ui.add_space(8.0);
    theme::section_heading(ui, "Scene");

    let mut select: Option<NodeId> = None;
    let mut delete: Option<NodeId> = None;
    let mut set_enabled: Option<(NodeId, bool)> = None;
    // The camera "Focus" view action (right-click a row → frame that node). Deferred
    // like the others and carried out on the `PanelResponse` (a VIEW action, not an
    // undoable document intent).
    let mut focus: Option<NodeId> = None;

    // Walk the tree depth-first; each row is indented by its depth.
    // ADR 0003 Phase B4: each row carries its node's stable NodeId, so the
    // select/delete/visibility ops (now NodeId-typed) are fed it directly — the
    // path is kept only for depth/indentation.
    let rows = state.scene.tree_rows();
    // Selection is keyed by NodeId; compare each row's id against the active id so
    // the highlight tracks the selected node by identity.
    let active_id = state.scene.active;
    for (_path, id, depth) in &rows {
        let is_active = active_id == Some(*id);
        // ADR 0018 Decision 2: the root part is the top row — selectable like any node,
        // but undeletable and with no visibility toggle (it always folds; hiding it is
        // meaningless). Its child count is `roots.len()` (its real spine), since the
        // container node's own `Group` payload is unused.
        if *id == ROOT_NODE_ID {
            ui.horizontal(|ui| {
                ui.add_space(*depth as f32 * 14.0);
                let label = format!("Part ({})", state.scene.roots.len());
                if ui
                    .selectable_label(is_active, label)
                    .on_hover_text("The scene's root part — the assembly everything folds into")
                    .clicked()
                {
                    select = Some(*id);
                }
            });
            continue;
        }
        // Read the node by id; mutate the enabled flag via a deferred op (a separate
        // lookup) so the borrow of `nodes` does not span the whole row.
        let (label, enabled) = match state.scene.node_by_id(*id) {
            Some(node) => (node_row_label(node), node.enabled),
            None => continue,
        };
        ui.horizontal(|ui| {
            ui.add_space(*depth as f32 * 14.0);
            let mut enabled = enabled;
            if ui
                .checkbox(&mut enabled, "")
                .on_hover_text("Contributes to the composed geometry")
                .changed()
            {
                set_enabled = Some((*id, enabled));
            }
            let row_response = ui.selectable_label(is_active, label);
            if row_response.clicked() {
                select = Some(*id);
            }
            // Right-click → Focus: a VIEW action that frames this node (camera target
            // = node centre, distance fitted to its AABB). Carried on the response,
            // not as an Intent (Focus is not undoable).
            row_response.context_menu(|ui| {
                if ui.button("Focus").clicked() {
                    focus = Some(*id);
                    ui.close();
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("✕").on_hover_text("Delete node").clicked() {
                    delete = Some(*id);
                }
            });
        });
    }

    // ADR 0003 Phase C C4a: the toggle is described as a `SetEnabled` intent (the loop
    // applies it), not a direct `set_node_enabled`. It re-resolves + auto-frames exactly
    // as before (the old `scene_changed`) — the flag gates participation in the fold, so
    // a re-resolve is not optional polish, it is what makes the edit visible at all.
    if let Some((id, enabled)) = set_enabled {
        response.emit_and_frame(Intent::SetEnabled { target: id, enabled });
    }

    // Carry a right-click Focus request to the loop (a view action; the loop sets the
    // camera target + distance from this node's AABB). Not an Intent — Focus is not
    // an undoable document mutation.
    if let Some(id) = focus {
        response.focus_node = Some(id);
    }

    if state.scene.roots.is_empty() {
        ui.label(egui::RichText::new("(no nodes — add one below)").small().weak());
    }

    build_node_actions(ui, state, response);
    build_definitions_section(ui, state, response);

    // Apply the deferred selection / delete after the walk. ADR 0003 Phase C C4a:
    // both are now intents (`RemoveNode` / `SelectNode`) the loop applies — the panel
    // no longer touches `scene.active` or calls `remove_node`/`sync_mirror_from_active`
    // here (the loop re-syncs the inspector mirror on the returned effect).
    if let Some(id) = delete {
        response.emit_and_frame(Intent::RemoveNode { target: id });
    } else if let Some(clicked_id) = select {
        // ADR 0003 Phase B4: a clicked row reports its node's stable NodeId; select
        // THAT, so the highlight and inspector follow the node through later
        // structural edits. Only emit when it actually changes the selection (the old
        // guard) — a `SelectNode` is selection-only (no re-resolve, no auto-frame).
        if state.scene.active != Some(clicked_id) {
            response.emit(Intent::SelectNode { target: Some(clicked_id) });
        }
    }

    ui.separator();
}

/// A [`NodeSpec`] for a fresh Tool node of the given shape, sized to that KIND's own
/// sensible default ([`ShapeKind::default_size_blocks`]) rather than the one shared size —
/// a Sphere is a round ball, a Cylinder a pillar, a Torus a flat donut, instead of every
/// kind squashed to the current slab. Density, wall, and material still come from the
/// current state so it renders immediately (ADR 0003 Phase C C4a). The spec's `into_node`
/// names the node after the shape kind.
fn tool_node_spec(kind: ShapeKind, state: &PanelState) -> NodeSpec {
    NodeSpec::Tool {
        // `from_blocks` applies the per-kind default (in blocks) at the current density,
        // retaining whole-block measurements and the ≥1 clamp in one owner.
        shape: SdfShape::from_blocks(
            kind,
            kind.default_size_blocks(),
            state.geometry.wall_blocks,
            state.geometry.voxels_per_block,
        ),
        material: state.material,
    }
}

/// A [`NodeSpec`] for a fresh sketch→extrude node sized to the current Size — a
/// footprint-extrude-up rectangle on the XY ground (ADR 0003 §3i). The current size
/// in voxels `[size_x, size_y, size_z]` maps onto a `PlaneAxis::Z` sketch: the
/// in-plane axes for Z are `[0, 1]` = X, Y, so the rectangle's in-plane width is
/// `size_x` and depth is `size_y`, extruded `size_z` voxels up along +Z. This is the
/// SAME construction the headless `default_sketch_spec_equals_box` test pins
/// (`SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, size_x, size_y), size_z)`), so
/// a freshly-added sketch resolves to exactly the matching `Box` of the current Size.
fn sketch_node_spec(state: &PanelState) -> NodeSpec {
    let [size_x, size_y, size_z] = state.geometry.size_voxels;
    NodeSpec::Sketch {
        producer: SketchSolid::extrude(
            Sketch::rectangle(PlaneAxis::Z, size_x as i64, size_y as i64),
            size_z,
        ),
        material: state.material,
    }
}

/// Build the action buttons under the tree: **+ Add** (top-level), **+ Add child**
/// (into the active Group), **Part** (wrap the active node in a new part), and **Make
/// definition** (turn the active node into a reusable [`AssemblyDef`] + Instance).
///
/// [`AssemblyDef`]: document::scene::AssemblyDef
fn build_node_actions(ui: &mut egui::Ui, state: &mut PanelState, response: &mut PanelResponse) {
    ui.add_space(4.0);

    // Whether the root part is the active selection (ADR 0018 Decision 2): it is a
    // container but NOT a "+ Add child" / Group / Make-definition target — its children
    // are added via the top-level "+ Add", and it can neither be wrapped nor turned
    // into a definition.
    let root_active = state.scene.active == Some(ROOT_NODE_ID);
    // Whether the active node is a Group (gates "+ Add child") — the root part is
    // excluded (adding into it means a top-level "+ Add", and `add_child_to_group`
    // does not resolve the root's reserved id).
    let active_is_group = !root_active
        && matches!(
            state.scene.active_node().map(|node| &node.content),
            Some(NodeContent::Group(_))
        );
    // Group / Make-definition act on a concrete, non-root node.
    let has_active_non_root = state.scene.active.is_some() && !root_active;

    ui.horizontal_wrapped(|ui| {
        // + Add — a top-level Tool or Clouds VoxelBody. ADR 0003 Phase C C4a: described as
        // `AddNode` intents (`NodeSpec` carries the same shape+material/Clouds the old
        // `new_tool_node` / `Node::new` built). The new node becomes active inside the
        // add op, so the loop re-syncs the inspector mirror on the returned effect.
        ui.menu_button("+ Add", |ui| {
            for (kind, label) in SHAPE_CHIPS {
                if ui.button(*label).clicked() {
                    // Live placement (ADR 0022): a primitive chip ARMS the tool rather
                    // than adding immediately — the shell then follows the cursor with a
                    // ghost and drops the node on a click. Sketch/Clouds below stay
                    // immediate (they have no cursor-snap placement yet).
                    response.armed_tool = Some(tool_node_spec(*kind, state));
                    ui.close();
                }
            }
            ui.separator();
            if ui.button("Sketch").clicked() {
                response.emit_and_frame(Intent::AddNode {
                    content: sketch_node_spec(state),
                });
                ui.close();
            }
            if ui.button("Clouds (Body)").clicked() {
                response.emit_and_frame(Intent::AddNode {
                    content: NodeSpec::CloudsPart,
                });
                ui.close();
            }
        });

        // + Add child — into the active Group (only shown when one is selected).
        if active_is_group {
            // ADR 0003 Phase B4: `AddChild` targets a NodeId; this block only shows
            // when a Group is active, so the active selection IS the group's id.
            let group_id = state.scene.active;
            ui.menu_button("+ Add child", |ui| {
                for (kind, label) in SHAPE_CHIPS {
                    if ui.button(*label).clicked() {
                        if let Some(group_id) = group_id {
                            response.emit_and_frame(Intent::AddChild {
                                group: group_id,
                                content: tool_node_spec(*kind, state),
                            });
                        }
                        ui.close();
                    }
                }
                ui.separator();
                if ui.button("Sketch").clicked() {
                    if let Some(group_id) = group_id {
                        response.emit_and_frame(Intent::AddChild {
                            group: group_id,
                            content: sketch_node_spec(state),
                        });
                    }
                    ui.close();
                }
                if ui.button("Clouds (Body)").clicked() {
                    if let Some(group_id) = group_id {
                        response.emit_and_frame(Intent::AddChild {
                            group: group_id,
                            content: NodeSpec::CloudsPart,
                        });
                    }
                    ui.close();
                }
            });
        }

        // Part — wrap the active node in a new part (a fresh composition container) →
        // `GroupNode { target: active }`. Disabled for the root part (it cannot wrap
        // itself).
        if ui
            .add_enabled(has_active_non_root, egui::Button::new("Part"))
            .on_hover_text("Wrap the selected node in a new Part")
            .clicked()
        {
            if let Some(target) = state.scene.active {
                response.emit_and_frame(Intent::GroupNode { target });
            }
        }

        // Make definition — the active node becomes a reusable AssemblyDef and is
        // replaced by an Instance of it → `MakeDefinition { target: active, name }`.
        // Disabled for the root part (a definition of the whole scene is out of scope).
        if ui
            .add_enabled(has_active_non_root, egui::Button::new("Make definition"))
            .on_hover_text("Turn the selected Part/node into a reusable definition, placed by an Instance")
            .clicked()
        {
            if let Some(target) = state.scene.active {
                let def_name = state
                    .scene
                    .active_node()
                    .map(|node| {
                        if node.name.is_empty() {
                            "Definition".to_string()
                        } else {
                            node.name.clone()
                        }
                    })
                    .unwrap_or_else(|| "Definition".to_string());
                response.emit_and_frame(Intent::MakeDefinition { target, name: def_name });
            }
        }
    });
}

/// The **Definitions** list (ADR 0001 step 4): the reusable [`AssemblyDef`]s, each
/// with an **Add instance** button that places another `Instance` of it at a
/// nudged offset (the village workflow: one stored body placed at several offsets)
/// and a **Fixture** toggle (ADR 0017 Decision 4, issue #77) — whether the
/// definition splices its children into each hosting scope's fold (a window that
/// cuts its opening AND fills its frame with one placement) instead of
/// pre-composing sealed. The toggle sits on the DEFINITION row because being a
/// fixture is what the part IS; instances stay pure reference+transform.
///
/// [`AssemblyDef`]: document::scene::AssemblyDef
fn build_definitions_section(
    ui: &mut egui::Ui,
    state: &mut PanelState,
    response: &mut PanelResponse,
) {
    if state.scene.definitions.is_empty() {
        return;
    }
    ui.add_space(6.0);
    theme::section_heading(ui, "Definitions");

    // Collect (id, label, fixture) first so the per-row widgets can mutate the scene
    // without borrowing `definitions` across the click.
    let defs: Vec<(DefId, String, bool)> = state
        .scene
        .definitions
        .iter()
        .map(|def| {
            let label = if def.name.is_empty() {
                format!("Def {}", def.id.0)
            } else {
                def.name.clone()
            };
            (
                def.id,
                format!("{} ({} node)", label, def.children.len()),
                def.fixture,
            )
        })
        .collect();

    let mut add_instance_of: Option<DefId> = None;
    let mut set_fixture_of: Option<(DefId, bool)> = None;
    for (id, label, fixture) in &defs {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(label).small());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .small_button("Add instance")
                    .on_hover_text("Place another instance of this definition")
                    .clicked()
                {
                    add_instance_of = Some(*id);
                }
                let mut is_fixture = *fixture;
                if ui
                    .checkbox(&mut is_fixture, egui::RichText::new("Fixture").small())
                    .on_hover_text(
                        "Splice this definition's children into each hosting scope's \
                         fold at the instance's position (in order, under the \
                         instance's transform) instead of pre-composing a sealed body \
                         — how a window cuts its opening and fills its frame with one \
                         placement",
                    )
                    .changed()
                {
                    set_fixture_of = Some((*id, is_fixture));
                }
            });
        });
    }

    // ADR 0003 Phase C C4a: described as an `AddInstance` intent. The placed Instance
    // becomes active inside the add op, so the loop re-syncs the mirror on the effect.
    if let Some(id) = add_instance_of {
        response.emit_and_frame(Intent::AddInstance { def: id });
    }
    // ADR 0017 Decision 4 (issue #77): a definition-field write — every placement's
    // composition changes at once (no auto-frame; the geometry stays in place).
    if let Some((id, fixture)) = set_fixture_of {
        response.emit(Intent::SetDefinitionFixture { def: id, fixture });
    }
}
