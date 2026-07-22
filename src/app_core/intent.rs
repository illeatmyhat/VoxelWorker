//! Intent dispatch + undo/redo — the serializable mutation core of [`AppCore`].
//!
//! ADR 0003 Phase C. [`AppCore::apply_intent`] (with `capture_inverse`) records each
//! edit on the command stack; [`AppCore::undo`]/[`AppCore::redo`] shuttle commands
//! between its two Vecs; `dispatch` is the single owner of every [`Scene`] field-write
//! / edit op; [`AppCore::effect_of`] classifies the resolve cost of an intent kind.

use document::command::{Command, Inverse};
use document::intent::{Intent, IntentEffect};
use document::scene::{Node, NodeContent, NodeId, NodeTransform, Point, VoxelBody, Scene};
use document::voxel::SdfShape;

use super::AppCore;

/// Dispatch helper for a per-node field write: apply `write` to the addressed node
/// (returning whether it landed), reporting `full_effect` on success and
/// [`IntentEffect::none`] on a missing id / kind-mismatch. Collapses the identical
/// `match node_by_id_mut { Some => {..; true}, None => false }` skeleton the node
/// field-set arms all share (their real content is which field `write` touches).
fn node_write(
    scene: &mut Scene,
    target: NodeId,
    full_effect: IntentEffect,
    write: impl FnOnce(&mut Node) -> bool,
) -> (IntentEffect, Option<NodeId>) {
    let applied = scene.node_by_id_mut(target).map(write).unwrap_or(false);
    (if applied { full_effect } else { IntentEffect::none() }, None)
}

/// Dispatch helper for a per-point field write — the `scene.points` sibling of
/// [`node_write`].
fn point_write(
    scene: &mut Scene,
    index: usize,
    full_effect: IntentEffect,
    write: impl FnOnce(&mut Point) -> bool,
) -> (IntentEffect, Option<NodeId>) {
    let applied = scene.points.get_mut(index).map(write).unwrap_or(false);
    (if applied { full_effect } else { IntentEffect::none() }, None)
}

/// Capture helper: read the addressed node's prior value via `prior` (which returns
/// the reconstructed field-set [`Intent`], or `None` on a kind-mismatch) and wrap it as
/// an [`Inverse::Field`]; a missing id / mismatch yields [`Inverse::NoOp`]. Collapses
/// the identical `match node_by_id { Some => Field(SetX{prior}), None => NoOp }`
/// skeleton the node field-set inverses all share.
fn node_field_inverse(
    scene: &Scene,
    target: NodeId,
    prior: impl FnOnce(&Node) -> Option<Intent>,
) -> Inverse {
    match scene.node_by_id(target).and_then(prior) {
        Some(intent) => Inverse::Field(intent),
        None => Inverse::NoOp,
    }
}

/// Capture helper for a per-point field inverse — the `scene.points` sibling of
/// [`node_field_inverse`]. A Point always yields a field-set when present (no inner
/// kind-match), so `prior` returns the [`Intent`] directly.
fn point_field_inverse(
    scene: &Scene,
    index: usize,
    prior: impl FnOnce(&Point) -> Intent,
) -> Inverse {
    match scene.points.get(index) {
        Some(point) => Inverse::Field(prior(point)),
        None => Inverse::NoOp,
    }
}

impl AppCore {
    /// **The single serializable mutation boundary (ADR 0003 Phase C, slice C1).**
    /// Apply one [`Intent`] to `scene` by dispatching to the SAME edit op / field
    /// write the panel performs today, returning the [`IntentEffect`] (the typed
    /// successor of [`PanelResponse`](ui::panel::PanelResponse)'s effect booleans)
    /// the caller reacts to.
    ///
    /// `apply_intent` borrows the scene (`&mut Scene`) rather than owning it — the
    /// scene still lives in `PanelState` (A2d ownership boundary); it owns no command
    /// stack yet (that is C2), so this is a pure dispatch + effect report. A field
    /// write to a missing id (or a kind-mismatched node — a `SetShape` on a non-Tool,
    /// a `SetCloudSeed` on a non-Clouds) is a no-op returning [`IntentEffect::none`].
    ///
    /// **The active-keyed ops.** [`group_active`](Scene::group_active) /
    /// [`make_definition_from_active`](Scene::make_definition_from_active) operate on
    /// the scene's `active` selection (the panel reaches them via the selected node),
    /// so the matching intents (`GroupNode` / `MakeDefinition`) point `scene.active`
    /// at their `target` first, then call the op — exactly how the panel arrives there
    /// (a clicked row sets `active`, then the action button fires). The intents carry
    /// the target explicitly so the value is self-contained / replayable.
    pub fn apply_intent(&mut self, scene: &mut Scene, intent: Intent) -> IntentEffect {
        // Selection-only intents are a view concern, not an undoable document step
        // (consistent with C1): dispatch + report, push NOTHING.
        if matches!(intent, Intent::SelectNode { .. } | Intent::SelectPoint { .. }) {
            let (effect, _minted) = self.dispatch(scene, intent);
            return effect;
        }

        // Snapshot the pre-state the undo needs (selection + the id counter — see the
        // COUNTER RULE in command.rs), then capture the inverse by reading the scene
        // BEFORE the mutation, then dispatch (which may mint ids the inverse needs).
        let selection_before = scene.active;
        let point_selection_before = scene.active_point;
        let counter_before = scene.next_node_id;
        let inverse = self.capture_inverse(scene, &intent, counter_before);
        let (effect, minted) = self.dispatch(scene, intent.clone());
        // The add family mints exactly one node; its inverse needs that id. We captured
        // a placeholder above for the add family, so patch it with the real minted id.
        let inverse = match (inverse, minted) {
            (Inverse::RemoveAdded { .. }, Some(id)) => Inverse::RemoveAdded { id },
            (other, _) => other,
        };
        self.command_stack.undo.push(Command {
            intent,
            inverse,
            selection_before,
            point_selection_before,
            counter_before,
        });
        // A fresh edit invalidates the redo future (the linear-stack rule).
        self.command_stack.redo.clear();
        effect
    }

    /// Capture the [`Inverse`] of `intent` by reading the scene's pre-mutation state
    /// (ADR 0003 Phase C C2). Called BEFORE [`dispatch`](Self::dispatch) so a field-set
    /// reads the PRIOR value and a structural op reads the soon-to-be-detached shape.
    /// The add family's minted id is not known yet, so it returns an
    /// [`Inverse::RemoveAdded`] placeholder the caller patches with `dispatch`'s minted
    /// id. A forward op that will be a no-op (missing id / kind-mismatch / stale
    /// target) yields [`Inverse::NoOp`], so `undo` of a no-op restores nothing.
    fn capture_inverse(&self, scene: &Scene, intent: &Intent, counter_before: u64) -> Inverse {
        match intent {
            // --- Structural ---
            // The add family mints one node appended to a spine; the caller patches the
            // placeholder id with `dispatch`'s minted id. (AddChild to a stale/non-Group
            // target mints nothing — but `dispatch` then returns `None`, and the
            // unpatched placeholder is never used because no node was added; we guard by
            // checking the minted id in the caller, so a `NoOp` here is cleaner.)
            // PlaceNode is AddNode with a placement — same single-node mint, same
            // RemoveAdded inverse patched with dispatch's minted id.
            Intent::AddNode { .. } | Intent::PlaceNode { .. } => {
                Inverse::RemoveAdded { id: NodeId(0) }
            }
            Intent::AddChild { group, .. } => {
                if matches!(
                    scene.node_by_id(*group).map(|node| &node.content),
                    Some(NodeContent::Group(_))
                ) {
                    Inverse::RemoveAdded { id: NodeId(0) }
                } else {
                    Inverse::NoOp
                }
            }
            Intent::AddInstance { def } => {
                if scene.def_by_id(*def).is_some() {
                    Inverse::RemoveAdded { id: NodeId(0) }
                } else {
                    Inverse::NoOp
                }
            }
            Intent::GroupNode { target } => {
                // group_active wraps `target` (the intent points `active` at it) in a
                // fresh Group minted as the next id. The Group takes the target's slot;
                // the inverse puts `target` back and drops the Group.
                if scene.node_by_id(*target).is_some() {
                    Inverse::UngroupNode {
                        target: *target,
                        group: NodeId(counter_before.max(1)),
                    }
                } else {
                    Inverse::NoOp
                }
            }
            Intent::MakeDefinition { target, .. } => match scene.node_by_id(*target) {
                Some(node) => {
                    let prior_content = node.content.clone();
                    // A Group DONATES its children (no new node); any other content
                    // mints a fresh "Body" node — the only node minted, so its id is
                    // `counter_before` (after the mint's `max(1)` normalization).
                    let minted_body = match &prior_content {
                        NodeContent::Group(_) => None,
                        _ => Some(NodeId(counter_before.max(1))),
                    };
                    Inverse::UndoMakeDefinition {
                        node: *target,
                        prior_content,
                        def: scene.next_def_id(),
                        minted_body,
                    }
                }
                None => Inverse::NoOp,
            },
            Intent::RemoveNode { target } => match scene.parent_and_index_of(*target) {
                Some((parent, index)) => Inverse::InsertSubtree {
                    parent,
                    index,
                    nodes: scene.clone_subtree_nodes(*target),
                },
                None => Inverse::NoOp,
            },

            // --- Node field writes (inverse = same intent carrying the prior value) ---
            Intent::SetEnabled { target, .. } => node_field_inverse(scene, *target, |node| {
                Some(Intent::SetEnabled { target: *target, enabled: node.enabled })
            }),
            Intent::SetShape { target, .. } => node_field_inverse(scene, *target, |node| {
                match &node.content {
                    // `SdfShape` is no longer `Copy` (it owns an optional boxed
                    // retained-size expression), so clone the prior shape so undo
                    // replays the EXACT authored size (ADR 0003 §3f(0)).
                    NodeContent::Tool { shape, .. } => {
                        Some(Intent::SetShape { target: *target, shape: shape.clone() })
                    }
                    _ => None,
                }
            }),
            Intent::SetSketch { target, .. } => node_field_inverse(scene, *target, |node| {
                match &node.content {
                    // Clone the prior producer so undo replays the EXACT sketch +
                    // extrude span (ADR 0003 §3i).
                    NodeContent::SketchTool { producer, .. } => {
                        Some(Intent::SetSketch { target: *target, producer: producer.clone() })
                    }
                    _ => None,
                }
            }),
            Intent::SetMaterial { target, .. } => node_field_inverse(scene, *target, |node| {
                match &node.content {
                    // Sketch nodes share the material field; capture their prior
                    // material too so the shared material edit is undoable.
                    NodeContent::Tool { material, .. }
                    | NodeContent::SketchTool { material, .. } => {
                        Some(Intent::SetMaterial { target: *target, material: *material })
                    }
                    _ => None,
                }
            }),
            // The operation is meaningful on EVERY node kind (ADR 0017): a leaf folds
            // its own body, a Group its sealed composed body (Decision 3, issue #74),
            // and an Instance the referenced definition's finished body — the reusable
            // cutter (issue #76). All capture the same field inverse.
            Intent::SetOperation { target, .. } => node_field_inverse(scene, *target, |node| {
                Some(Intent::SetOperation { target: *target, operation: node.operation })
            }),
            Intent::SetDefinitionFixture { def, .. } => match scene.def_by_id(*def) {
                // A DEFINITION field write (ADR 0017 Decision 4, issue #77): the
                // fixture flag lives on the AssemblyDef, so the inverse captures the
                // definition's prior flag — the same field-inverse shape as the
                // node-targeted writes above.
                Some(definition) => Inverse::Field(Intent::SetDefinitionFixture {
                    def: *def,
                    fixture: definition.fixture,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetOffset { target, .. } => node_field_inverse(scene, *target, |node| {
                // Capture the node's RETAINED per-axis measurements so undo replays the
                // EXACT authored expression — voxel-granular and parametric, not the
                // floored block view (ADR 0003 §3f(0)).
                Some(Intent::SetOffset {
                    target: *target,
                    offset_measurements: node.transform.offset_measurements(),
                })
            }),
            Intent::SetName { target, .. } => node_field_inverse(scene, *target, |node| {
                Some(Intent::SetName { target: *target, name: node.name.clone() })
            }),
            Intent::SetCloudSeed { target, .. } => node_field_inverse(scene, *target, |node| {
                match &node.content {
                    NodeContent::VoxelBody(VoxelBody::DebugClouds { seed }) => {
                        Some(Intent::SetCloudSeed { target: *target, seed: *seed })
                    }
                    _ => None,
                }
            }),
            Intent::SetNodeGrids { target, .. } => node_field_inverse(scene, *target, |node| {
                Some(Intent::SetNodeGrids { target: *target, grids: node.grids })
            }),

            // --- Global ---
            // Density is a single document-level field (ADR 0003 §3f(0)), so the
            // inverse is the same field-set carrying the prior `scene.voxels_per_block`
            // — exactly like `SetGridMasters`, routed back through `dispatch`.
            Intent::SetDensity { .. } => Inverse::Field(Intent::SetDensity {
                voxels_per_block: scene.voxels_per_block,
            }),
            Intent::SetGridMasters { .. } => Inverse::Field(Intent::SetGridMasters {
                voxel: scene.master_voxel_grid,
                lattice: scene.master_block_lattice,
                floor: scene.master_floor_grid,
            }),

            // --- Points ---
            // `add_point` appends, so the new Point lands at the current `len`.
            Intent::AddPoint { .. } => Inverse::RemoveAddedPoint {
                index: scene.points.len(),
            },
            Intent::RemovePoint { index } => match scene.points.get(*index) {
                // The Origin is undeletable (the forward op is a no-op for it).
                Some(point) if !point.is_origin => Inverse::InsertPoint {
                    index: *index,
                    point: point.clone(),
                },
                _ => Inverse::NoOp,
            },
            Intent::SetPointHidden { index, .. } => point_field_inverse(scene, *index, |point| {
                Intent::SetPointHidden { index: *index, hidden: point.hidden }
            }),
            Intent::SetPointPlanes { index, .. } => point_field_inverse(scene, *index, |point| {
                Intent::SetPointPlanes {
                    index: *index,
                    xz: point.plane_xz,
                    xy: point.plane_xy,
                    yz: point.plane_yz,
                }
            }),
            Intent::SetPointAxes { index, .. } => point_field_inverse(scene, *index, |point| {
                Intent::SetPointAxes {
                    index: *index,
                    x: point.axis_x,
                    y: point.axis_y,
                    z: point.axis_z,
                }
            }),
            Intent::SetPointPosition { index, .. } => point_field_inverse(scene, *index, |point| {
                Intent::SetPointPosition { index: *index, position_blocks: point.position_blocks }
            }),

            // Selection-only intents never reach here (handled + returned above).
            Intent::SelectNode { .. } | Intent::SelectPoint { .. } => Inverse::NoOp,
        }
    }

    /// Apply the top `undo` command's [`Inverse`] to `scene`, then restore the captured
    /// selection + id counter (the COUNTER RULE — see command.rs), and move the command
    /// to the `redo` stack (ADR 0003 Phase C C2). Returns the forward intent's own
    /// [`effect_of`](Self::effect_of) (so undoing a rename re-resolves the scene but NOT
    /// the points overlay, and undoing a grid-master toggle re-resolves nothing — the
    /// per-edit cost ADR 0003 optimizes against at 10k nodes), with `selection_changed`
    /// forced on (undo always restores `selection_before`, so the inspector mirror must
    /// re-sync). [`IntentEffect::none`] when the undo stack is empty.
    ///
    /// A [`Inverse::Field`] field-set is reversed by routing the prior-value intent back
    /// through [`dispatch`](Self::dispatch) (the single owner of the field-write
    /// mutations — no parallel copy to drift), so only the structural arms live in
    /// [`Inverse::apply`].
    pub fn undo(&mut self, scene: &mut Scene) -> IntentEffect {
        let Some(command) = self.command_stack.undo.pop() else {
            return IntentEffect::none();
        };
        match &command.inverse {
            Inverse::Field(prior) => {
                // Route the prior-value field-set through the same `dispatch` the forward
                // path uses — no re-implemented field-write copy to silently diverge.
                self.dispatch(scene, prior.clone());
            }
            structural => structural.apply(scene),
        }
        scene.active = command.selection_before;
        scene.active_point = command.point_selection_before;
        scene.next_node_id = command.counter_before;
        let effect = Self::effect_of(&command.intent).merged_with(IntentEffect::selection());
        self.command_stack.redo.push(command);
        effect
    }

    /// Re-apply the top `redo` command's forward `intent` to `scene` (ADR 0003 Phase C
    /// C2). The counter was rewound on undo, so re-`dispatch` re-mints byte-identical
    /// ids. Moves the command back to the `undo` stack. Returns the forward intent's own
    /// [`effect_of`](Self::effect_of) (with `selection_changed` forced on — redo restores
    /// the post-forward selection the caller must re-sync); [`IntentEffect::none`] when
    /// the redo stack is empty.
    pub fn redo(&mut self, scene: &mut Scene) -> IntentEffect {
        let Some(command) = self.command_stack.redo.pop() else {
            return IntentEffect::none();
        };
        self.dispatch(scene, command.intent.clone());
        let effect = Self::effect_of(&command.intent).merged_with(IntentEffect::selection());
        self.command_stack.undo.push(command);
        effect
    }

    /// The [`IntentEffect`] an intent produces **when it applies** — the single source
    /// of truth for "what does this mutation change" (ADR 0003 Phase C C2, code-review
    /// fix). A pure classification keyed only on the intent KIND: structural / field /
    /// global-density edits re-resolve (`scene`); the grid-master toggle is read live
    /// (`none`); a selection switch re-syncs the mirror (`selection`); a Point edit is
    /// overlay-only (`points`).
    ///
    /// [`dispatch`](Self::dispatch) uses this for its success branch (and downgrades to
    /// [`IntentEffect::none`] when the specific mutation could not land — a missing id /
    /// kind-mismatch / stale index). [`undo`](Self::undo) / [`redo`](Self::redo) use it
    /// so undoing a trivial rename reports only `scene_changed`, not a blanket-true
    /// rebuild of points too — the per-edit cost ADR 0003 optimizes against at 10k
    /// nodes.
    pub fn effect_of(intent: &Intent) -> IntentEffect {
        match intent {
            // Structural + node field writes + global density → re-resolve.
            Intent::AddNode { .. }
            | Intent::PlaceNode { .. }
            | Intent::AddChild { .. }
            | Intent::GroupNode { .. }
            | Intent::MakeDefinition { .. }
            | Intent::AddInstance { .. }
            | Intent::RemoveNode { .. }
            | Intent::SetEnabled { .. }
            | Intent::SetShape { .. }
            | Intent::SetSketch { .. }
            | Intent::SetMaterial { .. }
            | Intent::SetOperation { .. }
            | Intent::SetDefinitionFixture { .. }
            | Intent::SetOffset { .. }
            | Intent::SetName { .. }
            | Intent::SetCloudSeed { .. }
            | Intent::SetNodeGrids { .. }
            | Intent::SetDensity { .. } => IntentEffect::scene(),
            // The grid masters are read live by the per-frame line batch — no re-resolve.
            Intent::SetGridMasters { .. } => IntentEffect::none(),
            // Selection is a view concern (re-sync the inspector mirror only).
            Intent::SelectNode { .. } | Intent::SelectPoint { .. } => IntentEffect::selection(),
            // Points are pure overlay (no voxel re-resolve).
            Intent::AddPoint { .. }
            | Intent::RemovePoint { .. }
            | Intent::SetPointHidden { .. }
            | Intent::SetPointPlanes { .. }
            | Intent::SetPointAxes { .. }
            | Intent::SetPointPosition { .. } => IntentEffect::points(),
        }
    }

    /// The raw dispatch of one [`Intent`] to the matching [`Scene`](document::scene::Scene)
    /// edit op / field write (ADR 0003 Phase C — the C1 match, now factored out so both
    /// `apply_intent` and `redo` drive it). The success effect is
    /// [`effect_of`](Self::effect_of) (the single source of truth); a mutation that
    /// could not land (missing id / kind-mismatch / stale index) downgrades to
    /// [`IntentEffect::none`]. Also returns, for the add family (AddNode / AddChild /
    /// AddInstance), the minted node id the inverse needs (`None` for the field /
    /// structural / selection / point intents and for an add that no-ops on a stale
    /// target).
    fn dispatch(&self, scene: &mut Scene, intent: Intent) -> (IntentEffect, Option<NodeId>) {
        let full_effect = Self::effect_of(&intent);
        // The downgraded effect for a mutation that could not land.
        let none = IntentEffect::none();
        match intent {
            // --- Structural ---
            Intent::AddNode { content } => {
                let index = scene.add_node(content.into_node());
                let minted = scene.roots.get(index).copied();
                (full_effect, minted)
            }
            Intent::PlaceNode { content, offset_voxels, offset_local, rotation_quaternion } => {
                // Build the node exactly as AddNode, then override its identity transform with
                // the picked placement (ADR 0008 absolute voxel frame), the sub-voxel pivot
                // remainder (ADR 0027 continuous placement), AND — when the drop supplied one —
                // its continuous rotation (ADR 0027: the exact tilt to the gradient normal),
                // before the same add op mints its id.
                let mut node = content.into_node();
                let mut transform = NodeTransform::from_offset_voxels(offset_voxels);
                transform.offset_local_voxels = offset_local;
                if let Some(quaternion) = rotation_quaternion {
                    transform = transform.with_rotation(glam::Quat::from_array(quaternion));
                }
                node.transform = transform;
                let index = scene.add_node(node);
                let minted = scene.roots.get(index).copied();
                (full_effect, minted)
            }
            Intent::AddChild { group, content } => {
                let added = scene.add_child_to_group(group, content.into_node());
                // `add_child_to_group` selects the new child, so `active` is its id.
                let minted = if added { scene.active } else { None };
                (if added { full_effect } else { none }, minted)
            }
            Intent::GroupNode { target } => {
                // group_active keys off `active`; point it at the target first
                // (mirroring the panel: select the node, then click Group).
                scene.active = Some(target);
                scene.group_active();
                (full_effect, None)
            }
            Intent::MakeDefinition { target, name } => {
                scene.active = Some(target);
                scene.make_definition_from_active(name);
                (full_effect, None)
            }
            Intent::AddInstance { def } => {
                let minted = scene.add_instance(def);
                (if minted.is_some() { full_effect } else { none }, minted)
            }
            Intent::RemoveNode { target } => {
                scene.remove_node(target);
                (full_effect, None)
            }

            // --- Node field writes ---
            Intent::SetEnabled { target, enabled } => {
                let applied = scene.set_node_enabled(target, enabled);
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetShape { target, shape } => {
                node_write(scene, target, full_effect, |node| match &mut node.content {
                    NodeContent::Tool { shape: node_shape, .. } => {
                        *node_shape = shape;
                        true
                    }
                    _ => false,
                })
            }
            Intent::SetSketch { target, producer } => {
                node_write(scene, target, full_effect, |node| match &mut node.content {
                    NodeContent::SketchTool { producer: node_producer, .. } => {
                        *node_producer = producer;
                        true
                    }
                    _ => false,
                })
            }
            Intent::SetMaterial { target, material } => {
                // Sketch nodes carry the same shared material field, so the material
                // edit applies to them too (ADR 0003 §3i).
                node_write(scene, target, full_effect, |node| match &mut node.content {
                    NodeContent::Tool { material: node_material, .. }
                    | NodeContent::SketchTool { material: node_material, .. } => {
                        *node_material = material;
                        true
                    }
                    _ => false,
                })
            }
            Intent::SetOperation { target, operation } => {
                // ADR 0017: the combine operation applies to EVERY node kind — a
                // leaf folds its own body, a Group its sealed composed body
                // (Decision 3, issue #74), and an Instance the referenced
                // definition's finished body: a definition instanced with Subtract
                // is the reusable cutter (issue #76). The resolver honoured the
                // Instance operation since #74; this is its edit surface.
                node_write(scene, target, full_effect, |node| {
                    node.operation = operation;
                    true
                })
            }
            Intent::SetDefinitionFixture { def, fixture } => {
                // ADR 0017 Decision 4 (issue #77): sealed↔spliced is what the part
                // IS, so this writes the DEFINITION's flag. Every placement changes
                // composition at once; the resolver's leaf fingerprints carry the
                // scope path, which this flip changes for every expanded leaf, so
                // the store re-classifies each instance's chunks.
                let applied = scene.set_definition_fixture(def, fixture);
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetOffset { target, offset_measurements } => {
                // The intent carries the per-axis authored measurement (ADR 0003
                // §3f(0)). Derive the canonical voxel offset at the document density
                // and RETAIN the expression — the measurement→voxel rule has one
                // owner in `NodeTransform::from_measurements`. The inspector
                // validated each axis lands on a whole voxel before emitting.
                let density = scene.voxels_per_block;
                node_write(scene, target, full_effect, |node| {
                    node.transform = NodeTransform::from_measurements(offset_measurements, density);
                    true
                })
            }
            Intent::SetName { target, name } => node_write(scene, target, full_effect, |node| {
                node.name = name;
                true
            }),
            Intent::SetCloudSeed { target, seed } => {
                node_write(scene, target, full_effect, |node| match &mut node.content {
                    NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: node_seed }) => {
                        *node_seed = seed;
                        true
                    }
                    _ => false,
                })
            }
            Intent::SetNodeGrids { target, grids } => node_write(scene, target, full_effect, |node| {
                node.grids = grids;
                true
            }),
            // --- Global ---
            Intent::SetDensity { voxels_per_block } => {
                // Density is a document-level attribute (ADR 0003 §3f(0)): one field
                // on the scene that every resolve sources its density param from —
                // no per-Tool fan-out.
                //
                // Placement is stored as canonical voxels at the authoring density
                // (ADR 0003 §3f(0)). A density change must keep every node's
                // placement coherent, but the RIGHT way to do that depends on
                // whether the node carries a retained authored expression:
                //
                // * RETAINED measurement (`Some`): RE-EVALUATE the authored
                //   expression at the new density via `from_measurements`. This is
                //   the ADR's lossless re-target — block terms scale (`3.5 blocks`:
                //   56 vx at d16 → 112 at d32) and voxel terms stay EXACT (`3 blocks
                //   8 voxels`: 56 at d16 → 3*32+8 = 104 at d32, NOT the integer
                //   rescale's 112). A non-dividing re-target (e.g. d16→d15) floors
                //   and resynthesises that axis inside `from_measurements`, so the
                //   retained expression and `offset_voxels` can never disagree.
                // * NO retained measurement (`None` — old docs, drags, pure-voxel
                //   offsets): keep the legacy integer rescale, which PRESERVES the
                //   physical position (and stays on the mating lattice for
                //   block-multiple offsets). The field stays `None`.
                //
                // The explicit, warned, DESTRUCTIVE "re-target to a different game
                // grid" remains a SEPARATE future Slice-2 op, not this.
                let old_density = scene.voxels_per_block.max(1) as i64;
                let new_density = voxels_per_block as i64;
                for node in scene.arena.values_mut() {
                    // Offsets live on EVERY NodeTransform (groups/instances too), not
                    // just Tools, so re-target them all.
                    if node.transform.has_retained_measurements() {
                        node.transform = NodeTransform::from_measurements(
                            node.transform.offset_measurements(),
                            voxels_per_block,
                        );
                    } else {
                        for axis in 0..3 {
                            node.transform.offset_voxels[axis] =
                                node.transform.offset_voxels[axis] * new_density / old_density;
                        }
                    }

                    // A Tool's SIZE is now voxel-granular (ADR 0003 §3f(0)), so it
                    // must be re-targeted on a density change EXACTLY like the offset
                    // — otherwise the physical size would change (an 80-voxel = 5-block
                    // box at d16 would stay 80 voxels = 2.5 blocks at d32). Same split:
                    //  * RETAINED authored size: re-evaluate via `from_measurements`
                    //    (block terms scale, voxel terms stay exact, non-dividing axes
                    //    floor+resynthesise — never disagree with `size_voxels`).
                    //  * NO retained size (old docs / pure-voxel): integer rescale to
                    //    preserve physical size; the field stays `None`.
                    if let NodeContent::Tool { shape, .. } = &mut node.content {
                        if shape.has_retained_size_measurements() {
                            *shape = SdfShape::from_measurements(
                                shape.kind,
                                shape.size_measurements(),
                                shape.wall_blocks,
                                voxels_per_block,
                            );
                        } else {
                            let mut size_voxels = shape.size_voxels;
                            for axis in size_voxels.iter_mut() {
                                // Integer rescale, clamped to ≥1 so a tiny size can't
                                // collapse to a 0-voxel (degenerate) axis.
                                *axis = ((*axis as i64 * new_density / old_density).max(1)) as u32;
                            }
                            *shape = SdfShape::from_voxels(shape.kind, size_voxels, shape.wall_blocks);
                        }
                    }
                }
                scene.voxels_per_block = voxels_per_block;
                (full_effect, None)
            }
            Intent::SetGridMasters { voxel, lattice, floor } => {
                // The masters are read live by the per-frame line batch, so no
                // re-resolve — `full_effect` is already `none()` for this intent.
                scene.master_voxel_grid = voxel;
                scene.master_block_lattice = lattice;
                scene.master_floor_grid = floor;
                (full_effect, None)
            }

            // --- Selection ---
            Intent::SelectNode { target } => {
                scene.active = target;
                (full_effect, None)
            }
            Intent::SelectPoint { target } => {
                scene.active_point = target;
                (full_effect, None)
            }

            // --- Points ---
            Intent::AddPoint { position_blocks, name } => {
                let point = document::scene::Point {
                    name,
                    position_blocks,
                    ..document::scene::Point::default()
                };
                scene.add_point(point);
                (full_effect, None)
            }
            Intent::RemovePoint { index } => {
                scene.remove_point(index);
                (full_effect, None)
            }
            Intent::SetPointHidden { index, hidden } => {
                point_write(scene, index, full_effect, |point| {
                    point.hidden = hidden;
                    true
                })
            }
            Intent::SetPointPlanes { index, xz, xy, yz } => {
                point_write(scene, index, full_effect, |point| {
                    point.plane_xz = xz;
                    point.plane_xy = xy;
                    point.plane_yz = yz;
                    true
                })
            }
            Intent::SetPointAxes { index, x, y, z } => {
                point_write(scene, index, full_effect, |point| {
                    point.axis_x = x;
                    point.axis_y = y;
                    point.axis_z = z;
                    true
                })
            }
            Intent::SetPointPosition { index, position_blocks } => {
                point_write(scene, index, full_effect, |point| {
                    point.position_blocks = position_blocks;
                    true
                })
            }
        }
    }
}
