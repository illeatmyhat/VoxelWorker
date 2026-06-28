//! Headless orchestrator owning store + camera — the AppCore keystone.
//!
//! ADR 0003 (foundation rework). `AppCore` is the headless half of the app: it
//! owns the [`Store`] (residency + per-chunk resolve) and the [`OrbitCamera`],
//! and exposes the headless scene queries both binaries drive. The windowed
//! shell (`WindowedState`) and `bin/shot` keep the GPU renderers + winit/egui
//! plumbing and delegate to `AppCore` for the headless work; in **A3** `shot`
//! re-points here, at which point the golden net tests the real app instead of a
//! parallel render copy.
//!
//! **Ownership boundary (A2d).** `AppCore` owns the store + camera but BORROWS
//! the scene (`&Scene`) — the scene still lives in `PanelState` until Phase B/C
//! moves it here. The scene-query associated functions below therefore take
//! `&Scene` as a parameter; they become `&self` methods once `AppCore` owns the
//! scene. Resolve state + the borrow-sensitive `AppCore::rebuild` land in
//! **A2e**; `render` reads all headless data from here in **A2f**.

use crate::camera::OrbitCamera;
use crate::command::{Command, CommandStack, Inverse};
use crate::core_geom::CHUNK_BLOCKS;
use crate::intent::{Intent, IntentEffect};
use crate::panel::LayerRange;
use crate::renderer::OnionFogParams;
use crate::scene::{NodeContent, NodeId, Part, Scene};
use crate::spatial_index::LeafSpatialIndex;
use crate::store::Store;
use crate::voxel::{chunk_extent_exceeds_bound, VoxelGrid};

/// The headless orchestrator: owns the per-chunk resolve [`Store`] and the
/// [`OrbitCamera`], and answers the headless scene queries the shell renders from.
pub struct AppCore {
    /// Per-chunk resolve cache (issue #27 S2): the resolve mechanism behind the
    /// shell's geometry rebuild and the diameter readout. Lazily resolves each
    /// covering chunk and keeps it resident for reuse.
    pub store: Store,
    /// The orbit camera (orbit angles + distance + projection). The windowed shell
    /// drives it from input; `shot` sets it from CLI flags.
    pub camera: OrbitCamera,
    /// The leaf spatial index (issue #27 S3) the LAST [`rebuild`](Self::rebuild)
    /// resolved from, kept so the next rebuild can diff against it to compute the
    /// edit's dirty world-AABB. `None` before the first rebuild (which clears
    /// wholesale).
    previous_leaf_index: Option<LeafSpatialIndex>,
    /// The composite recentre (floating origin, in voxels) the LAST rebuild resolved
    /// at (issue #20 S6c-2c): the resolve bookkeeping that records whether the
    /// floating origin shifted. `None` before the first rebuild.
    previous_recentre_voxels: Option<[i64; 3]>,
    /// The linear inverse-command stack behind undo/redo (ADR 0003 Phase C C2). Every
    /// non-selection-only `apply_intent` pushes a [`Command`] here; `undo`/`redo`
    /// shuttle commands between its two Vecs. Empty until the first undoable edit.
    command_stack: CommandStack,
}

/// The headless resolve output of a geometry [`rebuild`](AppCore::rebuild) (A2e).
/// Holds the assembled region grid (owned) plus the per-chunk render accessor,
/// which BORROWS the store — so the shell must consume both (build the cuboid mesh
/// + upload the fog occupancy) BEFORE the next `&mut AppCore` call.
pub struct RebuildOutput<'store> {
    /// The assembled monolithic region grid (recentred): feeds the fog upload and
    /// the shell's diameter re-measure.
    pub grid: VoxelGrid,
    /// The region's voxel dimensions, read from the SCENE (see
    /// [`AppCore::region_dimensions_for`]) — what the camera auto-frame, gizmo,
    /// lattice, floor grid and layer scrubber are sized from.
    pub region_dimensions: [u32; 3],
    /// The per-covering-chunk render accessor
    /// (`(absolute_chunk_coord, &rebased_grid)`), borrowing the store. Drop it
    /// before the next `&mut AppCore`.
    pub render_chunks: Vec<([i32; 3], &'store VoxelGrid)>,
}

/// Outcome of [`AppCore::rebuild`]: either the resolve output, or a rejection when
/// the density's PER-CHUNK voxel bound is exceeded. AppCore never writes panel
/// state, so the shell surfaces the cap warning from the returned figure.
pub enum RebuildOutcome<'store> {
    /// The resolve succeeded; the store holds the freshly resolved covering chunks.
    Built(RebuildOutput<'store>),
    /// The density's single-chunk voxel capacity exceeds the bound; the store was
    /// left untouched. `chunk_voxels_millions` is the offending count (millions).
    DensityRejected { chunk_voxels_millions: f32 },
}

impl AppCore {
    /// Assemble the headless core from an already-constructed store + camera. The
    /// shell builds both (the store seeds the startup diameter readout, the camera
    /// restores persisted orbit/projection) and hands them over here.
    pub fn new(store: Store, camera: OrbitCamera) -> Self {
        Self {
            store,
            camera,
            previous_leaf_index: None,
            previous_recentre_voxels: None,
            command_stack: CommandStack::new(),
        }
    }

    /// **The single serializable mutation boundary (ADR 0003 Phase C, slice C1).**
    /// Apply one [`Intent`] to `scene` by dispatching to the SAME edit op / field
    /// write the panel performs today, returning the [`IntentEffect`] (the typed
    /// successor of [`PanelResponse`](crate::panel::PanelResponse)'s effect booleans)
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
            Intent::AddNode { .. } => Inverse::RemoveAdded { id: NodeId(0) },
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
            Intent::SetVisible { target, .. } => match scene.node_by_id(*target) {
                Some(node) => Inverse::Field(Intent::SetVisible {
                    target: *target,
                    visible: node.visible,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetShape { target, .. } => match scene.node_by_id(*target) {
                Some(node) => match &node.content {
                    NodeContent::Tool { shape, .. } => Inverse::Field(Intent::SetShape {
                        target: *target,
                        shape: *shape,
                    }),
                    _ => Inverse::NoOp,
                },
                None => Inverse::NoOp,
            },
            Intent::SetMaterial { target, .. } => match scene.node_by_id(*target) {
                Some(node) => match &node.content {
                    NodeContent::Tool { material, .. } => Inverse::Field(Intent::SetMaterial {
                        target: *target,
                        material: *material,
                    }),
                    _ => Inverse::NoOp,
                },
                None => Inverse::NoOp,
            },
            Intent::SetOffset { target, .. } => match scene.node_by_id(*target) {
                Some(node) => Inverse::Field(Intent::SetOffset {
                    target: *target,
                    offset_blocks: node.transform.offset_blocks,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetName { target, .. } => match scene.node_by_id(*target) {
                Some(node) => Inverse::Field(Intent::SetName {
                    target: *target,
                    name: node.name.clone(),
                }),
                None => Inverse::NoOp,
            },
            Intent::SetCloudSeed { target, .. } => match scene.node_by_id(*target) {
                Some(node) => match &node.content {
                    NodeContent::Part(Part::DebugClouds { seed }) => {
                        Inverse::Field(Intent::SetCloudSeed {
                            target: *target,
                            seed: *seed,
                        })
                    }
                    _ => Inverse::NoOp,
                },
                None => Inverse::NoOp,
            },
            Intent::SetNodeGrids { target, .. } => match scene.node_by_id(*target) {
                Some(node) => Inverse::Field(Intent::SetNodeGrids {
                    target: *target,
                    grids: node.grids,
                }),
                None => Inverse::NoOp,
            },

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
            Intent::SetPointHidden { index, .. } => match scene.points.get(*index) {
                Some(point) => Inverse::Field(Intent::SetPointHidden {
                    index: *index,
                    hidden: point.hidden,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetPointPlanes { index, .. } => match scene.points.get(*index) {
                Some(point) => Inverse::Field(Intent::SetPointPlanes {
                    index: *index,
                    xz: point.plane_xz,
                    xy: point.plane_xy,
                    yz: point.plane_yz,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetPointAxes { index, .. } => match scene.points.get(*index) {
                Some(point) => Inverse::Field(Intent::SetPointAxes {
                    index: *index,
                    x: point.axis_x,
                    y: point.axis_y,
                    z: point.axis_z,
                }),
                None => Inverse::NoOp,
            },
            Intent::SetPointPosition { index, .. } => match scene.points.get(*index) {
                Some(point) => Inverse::Field(Intent::SetPointPosition {
                    index: *index,
                    position_blocks: point.position_blocks,
                }),
                None => Inverse::NoOp,
            },

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
            | Intent::AddChild { .. }
            | Intent::GroupNode { .. }
            | Intent::MakeDefinition { .. }
            | Intent::AddInstance { .. }
            | Intent::RemoveNode { .. }
            | Intent::SetVisible { .. }
            | Intent::SetShape { .. }
            | Intent::SetMaterial { .. }
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

    /// The raw dispatch of one [`Intent`] to the matching [`Scene`](crate::scene::Scene)
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
            Intent::SetVisible { target, visible } => {
                let applied = scene.set_node_visible(target, visible);
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetShape { target, shape } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => match &mut node.content {
                        NodeContent::Tool { shape: node_shape, .. } => {
                            *node_shape = shape;
                            true
                        }
                        _ => false,
                    },
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetMaterial { target, material } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => match &mut node.content {
                        NodeContent::Tool { material: node_material, .. } => {
                            *node_material = material;
                            true
                        }
                        _ => false,
                    },
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetOffset { target, offset_blocks } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => {
                        node.transform.offset_blocks = offset_blocks;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetName { target, name } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => {
                        node.name = name;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetCloudSeed { target, seed } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => match &mut node.content {
                        NodeContent::Part(Part::DebugClouds { seed: node_seed }) => {
                            *node_seed = seed;
                            true
                        }
                        _ => false,
                    },
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetNodeGrids { target, grids } => {
                let applied = match scene.node_by_id_mut(target) {
                    Some(node) => {
                        node.grids = grids;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }

            // --- Global ---
            Intent::SetDensity { voxels_per_block } => {
                // Density is a document-level attribute (ADR 0003 §3f(0)): one field
                // on the scene that every resolve sources its density param from —
                // no per-Tool fan-out.
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
                let point = crate::scene::Point {
                    name,
                    position_blocks,
                    ..crate::scene::Point::default()
                };
                scene.add_point(point);
                (full_effect, None)
            }
            Intent::RemovePoint { index } => {
                scene.remove_point(index);
                (full_effect, None)
            }
            Intent::SetPointHidden { index, hidden } => {
                let applied = match scene.points.get_mut(index) {
                    Some(point) => {
                        point.hidden = hidden;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetPointPlanes { index, xz, xy, yz } => {
                let applied = match scene.points.get_mut(index) {
                    Some(point) => {
                        point.plane_xz = xz;
                        point.plane_xy = xy;
                        point.plane_yz = yz;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetPointAxes { index, x, y, z } => {
                let applied = match scene.points.get_mut(index) {
                    Some(point) => {
                        point.axis_x = x;
                        point.axis_y = y;
                        point.axis_z = z;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
            Intent::SetPointPosition { index, position_blocks } => {
                let applied = match scene.points.get_mut(index) {
                    Some(point) => {
                        point.position_blocks = position_blocks;
                        true
                    }
                    None => false,
                };
                (if applied { full_effect } else { none }, None)
            }
        }
    }

    /// **The headless geometry rebuild (A2e).** Route the resolve through the
    /// per-chunk store with issue #27 S3 TARGETED invalidation: build the new
    /// scene's leaf spatial index, diff it against the last rebuild's to get the
    /// edit's dirty world-AABB, and evict ONLY the chunks that AABB touches (every
    /// other cached chunk stays resident). Fall back to a wholesale `clear()` when a
    /// precise AABB can't be computed — the first rebuild (no previous index), a
    /// density change, or a region-spanning Part edit (no localisable box, see
    /// `LeafSpatialIndex::edit_aabb_since`). The reassembled grid is byte-identical
    /// either way (the same chunks are re-resolved; untouched chunks are reused).
    ///
    /// Returns the assembled grid + region dimensions + the per-chunk render
    /// accessor, which BORROWS the store. The returned [`RebuildOutcome`] therefore
    /// borrows `self`, so the shell must consume it (build the cuboid mesh, upload
    /// the fog occupancy) BEFORE the next `&mut AppCore` call. A density whose
    /// single-chunk voxel capacity exceeds the bound is rejected WITHOUT touching
    /// the store, returning the offending count so the shell can surface the cap
    /// warning (AppCore never writes panel state).
    pub fn rebuild<'a>(&'a mut self, scene: &Scene, density: u32) -> RebuildOutcome<'a> {
        // Issue #27 S2: the resolve is chunked + lazy, so the voxel bound is a
        // PER-CHUNK bound, not a whole-scene total. Only a pathological density
        // (one chunk's voxel capacity alone exceeds the bound) is rejected.
        if chunk_extent_exceeds_bound(density) {
            let chunk_extent = (CHUNK_BLOCKS * density.max(1)) as u64;
            let chunk_voxels = chunk_extent * chunk_extent * chunk_extent;
            return RebuildOutcome::DensityRejected {
                chunk_voxels_millions: chunk_voxels as f32 / 1_000_000.0,
            };
        }

        // S3 targeted invalidation. The cuboid renderer rebuilds every covering
        // chunk wholesale, so it needs no dirty set of its own — but the store's
        // invalidation side effects ARE still required: `invalidate_aabb` evicts the
        // edit's dirty chunks (so `resident_render_chunks` re-resolves them), and
        // `clear()` handles the first build / density change / region-spanning edit
        // where there is no localisable AABB.
        let new_leaf_index = scene.build_leaf_spatial_index(density);
        let new_recentre = scene.recentre_voxels_for_resolve(density);
        match self.previous_leaf_index.as_ref() {
            Some(previous) => match new_leaf_index.edit_aabb_since(previous) {
                Some(edit_aabb) => {
                    self.store.invalidate_aabb(&edit_aabb, density);
                }
                None => self.store.clear(),
            },
            None => self.store.clear(),
        }
        self.previous_recentre_voxels = Some(new_recentre);
        self.previous_leaf_index = Some(new_leaf_index);

        // Resolve the assembled grid (owned), then gather the per-chunk render
        // accessor LAST — it borrows the store, so every `&mut store` call above
        // must already be done. The grid drops straight into the fog upload; the
        // accessor feeds the cuboid mesher, then the shell drops it.
        let grid = self.store.resolve_region(scene, density, 0);
        let region_dimensions = Self::region_dimensions_for(scene, density, &grid);
        let render_chunks = self.store.resident_render_chunks(scene, density, 0);
        RebuildOutcome::Built(RebuildOutput {
            grid,
            region_dimensions,
            render_chunks,
        })
    }

    /// Resolve the whole [`Scene`] into a fresh grid (ADR 0001 step 2). Every
    /// visible node composites (union) into one region sized to the per-axis max of
    /// the nodes' extents, at full resolution (`lod 0`). `voxels_per_block` is the
    /// global app density (the inspector mirror's density). For a one-node scene
    /// this is identical to the step-1 behaviour.
    ///
    /// An associated function for now (it borrows the scene; A2d ownership boundary)
    /// — it becomes a `&self` method once `AppCore` owns the scene in Phase B/C.
    pub fn resolve_scene(scene: &Scene, voxels_per_block: u32) -> VoxelGrid {
        let region = scene.full_extent_blocks(voxels_per_block);
        scene.resolve_region(region, voxels_per_block, 0)
    }

    /// The region dimensions (in voxels) the camera auto-frame, origin gizmo, block
    /// lattice, fine floor grid and layer scrubber are sized from — read from the
    /// SCENE, not by reaching into the assembled `VoxelGrid` (issue #20 S6c-1, prep
    /// for the per-chunk renderer of S6c step 4). This is a behaviour-preserving
    /// substitution: for a chunkable scene (every Tool scene, including the startup
    /// default) the assembled grid is literally sized to
    /// [`Scene::placed_region_dimensions`] — so this returns BYTE-IDENTICAL
    /// dimensions (proven in
    /// `scene::tests::placed_region_dimensions_equals_assembled_grid`).
    ///
    /// A **Part-only** scene (e.g. a lone debug-cloud field) has no composite
    /// extent, so `placed_region_dimensions` would be `[0, 0, 0]`; that scene is
    /// resolved through the explicit-region path instead, so we fall back to the
    /// assembled grid's own dimensions — which (being the grid the consumers used
    /// before) is trivially identical to the old behaviour for that case.
    pub fn region_dimensions_for(scene: &Scene, density: u32, grid: &VoxelGrid) -> [u32; 3] {
        if scene.has_chunkable_extent(density) {
            scene.placed_region_dimensions(density)
        } else {
            grid.dimensions
        }
    }

    /// The camera's view-projection matrix for the given viewport aspect ratio —
    /// the recentred-frame matrix every overlay + the voxel pass draw with. A
    /// `&self` getter (it reads the owned camera) so the shell and `shot` source the
    /// frame matrix identically.
    pub fn view_projection(&self, aspect_ratio: f32) -> glam::Mat4 {
        self.camera.view_projection(aspect_ratio)
    }

    /// Where the transform gizmo (issue #29 S2) should sit: the SELECTED node's
    /// recentred pivot + its extent (in voxels), or `None` when nothing is selected
    /// (or the selection has no extent). An associated function for now (it borrows
    /// the scene; A2d ownership boundary) — becomes `&self` once `AppCore` owns the
    /// scene in Phase B/C.
    pub fn gizmo_placement(scene: &Scene, density: u32) -> Option<([f32; 3], [f32; 3])> {
        scene.active_gizmo_placement(density)
    }

    /// The recentred `(pivot_voxels, extent_voxels)` for an ARBITRARY node id (not
    /// the active selection) — the camera "Focus" view action frames that node. A
    /// thin wrapper over [`Scene::gizmo_placement_for_id`]; `None` when the id no
    /// longer resolves or the node has no extent (Focus is then a no-op).
    pub fn gizmo_placement_for_id(
        scene: &Scene,
        node_id: NodeId,
        density: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        scene.gizmo_placement_for_id(node_id, density)
    }

    /// Build the onion-skin fog parameters (issue #12) from the camera-derived
    /// view-projection, grid, and layer-range scrubber. World-Y of layer `j` spans
    /// `[j - grid_y/2, j+1 - grid_y/2]` (voxel centres at `j + 0.5 - grid_y/2`). The
    /// solid band is layers `[lower, upper]`; the onion band extends `onion_depth`
    /// layers on each side.
    pub fn onion_fog_params(
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        layer_range: LayerRange,
    ) -> OnionFogParams {
        let grid_y = grid_dimensions[1] as f32;
        let half_y = grid_y / 2.0;
        let depth = layer_range.onion_depth.clamp(1, 8) as f32;
        let lower = layer_range.lower as f32;
        let upper = layer_range.upper.min(grid_dimensions[1].saturating_sub(1)) as f32;
        OnionFogParams {
            inverse_view_projection: view_projection.inverse(),
            semi_axes: [
                grid_dimensions[0] as f32 / 2.0,
                grid_dimensions[1] as f32 / 2.0,
                grid_dimensions[2] as f32 / 2.0,
            ],
            // Onion band world-Y: `depth` layers below the band's bottom edge to
            // `depth` layers above its top edge.
            onion_y_min: (lower - depth) - half_y,
            onion_y_max: (upper + 1.0 + depth) - half_y,
            // Solid band world-Y (excluded from the fog).
            band_y_min: lower - half_y,
            band_y_max: (upper + 1.0) - half_y,
        }
    }

    /// The number of commands on the undo stack (ADR 0003 Phase C C2 test support).
    #[cfg(test)]
    pub(crate) fn undo_depth(&self) -> usize {
        self.command_stack.undo.len()
    }

    /// The number of commands on the redo stack (ADR 0003 Phase C C2 test support).
    #[cfg(test)]
    pub(crate) fn redo_depth(&self) -> usize {
        self.command_stack.redo.len()
    }
}

/// The **default seed scene** the windowed app starts from (ADR 0003 Phase C, slice
/// C3 — the base a `shot --replay` script is applied against). A single Tool node
/// from the default geometry/material, the Origin Point synthesized, stable
/// [`NodeId`]s minted — i.e. exactly `PanelState::with_view_cube_default().scene`
/// (which runs `Scene::from_geometry(default)` + `ensure_origin_point` +
/// `ensure_node_ids`). Kept here so both `bin/shot` and the lib tests build the
/// replay base the same way.
pub fn default_replay_seed_scene() -> Scene {
    crate::panel::PanelState::with_view_cube_default().scene
}

/// Replay a **newline-delimited-JSON Intent script** into a [`Scene`] (ADR 0003
/// Phase C, slice C3 — the testable core of `shot --replay`).
///
/// The `script` is one [`Intent`] per line: each non-empty line is parsed with
/// `serde_json::from_str::<Intent>` and applied IN ORDER, via
/// [`AppCore::apply_intent`], to the [`default_replay_seed_scene`]. Blank /
/// whitespace-only lines are skipped. Returns the post-replay scene.
///
/// On a JSON parse error on any line, returns `Err` with a message naming the
/// 1-based line number and the offending line (no panic) — the caller prints it and
/// exits non-zero. `bin/shot` reads the file then calls this; the lib tests feed a
/// string directly (keeping the GPU render out of the unit test).
pub fn replay_intent_script(script: &str) -> Result<Scene, String> {
    let mut scene = default_replay_seed_scene();
    let mut app_core = AppCore::new(Store::new(), OrbitCamera::default());
    for (line_index, raw_line) in script.lines().enumerate() {
        let line_number = line_index + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let intent: Intent = serde_json::from_str(trimmed).map_err(|error| {
            format!("parse error on line {line_number}: {error}\n  line: {trimmed}")
        })?;
        app_core.apply_intent(&mut scene, intent);
    }
    Ok(scene)
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::intent::{Intent, NodeSpec};
    use crate::scene::NodeContent;
    use crate::voxel::{SdfShape, ShapeKind};

    /// A small box Tool shape for the script fixtures.
    fn box_shape() -> SdfShape {
        SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [3, 3, 3],
            wall_blocks: 1,
        }
    }

    /// The replay seed base is the windowed default: exactly one top-level node (a
    /// Tool from the default geometry) and exactly one Point (the Origin), with ids
    /// minted. Scripts are written against this known starting point.
    #[test]
    fn default_seed_scene_matches_windowed_base() {
        let seed = default_replay_seed_scene();
        assert_eq!(seed.roots.len(), 1, "seed has one top-level Tool node");
        assert_eq!(seed.points.len(), 1, "seed carries exactly the Origin Point");
        assert!(
            seed.roots.iter().all(|id| id.0 != 0),
            "every seed node has a minted (non-zero) NodeId"
        );
    }

    /// Plumbing proof: an `AddNode` then a `SetOffset` (targeting the just-added node
    /// by its minted id) replays to a scene whose NEW node sits at the requested
    /// offset — i.e. the script parsed, dispatched through `apply_intent`, and the
    /// mutations landed in order.
    #[test]
    fn replay_add_then_offset_places_new_node() {
        let seed = default_replay_seed_scene();
        let roots_before = seed.roots.len();
        // The add mints the next id past the seed's counter. `apply_intent`'s add op
        // assigns `next_node_id`, which after the seed equals `roots_before + 1` given
        // the seed mints ids 1..=roots_before. Derive it from the seed to stay robust.
        let new_node_id = NodeId(seed.next_node_id);

        let add = Intent::AddNode {
            content: NodeSpec::Tool {
                shape: box_shape(),
                material: MaterialChoice::Wood,
            },
        };
        let set_offset = Intent::SetOffset {
            target: new_node_id,
            offset_blocks: [7, -2, 4],
        };
        let script = format!(
            "{}\n\n{}\n",
            serde_json::to_string(&add).unwrap(),
            serde_json::to_string(&set_offset).unwrap(),
        );

        let scene = replay_intent_script(&script).expect("replay succeeds");
        assert_eq!(
            scene.roots.len(),
            roots_before + 1,
            "AddNode added exactly one top-level node"
        );
        let added = scene
            .node_by_id(new_node_id)
            .expect("the added node exists at its minted id");
        assert_eq!(
            added.transform.offset_blocks,
            [7, -2, 4],
            "SetOffset moved the just-added node to the requested offset"
        );
        assert!(
            matches!(added.content, NodeContent::Tool { .. }),
            "the added node is a Tool"
        );
    }

    /// A malformed line is reported as an `Err` (naming the line number), NOT a panic.
    #[test]
    fn replay_malformed_line_is_reported_not_panicked() {
        let good = Intent::AddNode {
            content: NodeSpec::CloudsPart,
        };
        let script = format!(
            "{}\nthis is not json\n",
            serde_json::to_string(&good).unwrap()
        );
        let error = replay_intent_script(&script).expect_err("malformed line must error");
        assert!(
            error.contains("line 2"),
            "error names the offending 1-based line number, got: {error}"
        );
    }

    /// Blank / whitespace-only lines are skipped (not parse errors).
    #[test]
    fn replay_skips_blank_lines() {
        let add = Intent::AddNode {
            content: NodeSpec::CloudsPart,
        };
        let script = format!("\n   \n{}\n\n", serde_json::to_string(&add).unwrap());
        let scene = replay_intent_script(&script).expect("blank lines skipped");
        // Seed (1 Tool) + 1 Clouds Part = 2 top-level nodes.
        assert_eq!(scene.roots.len(), 2);
    }
}

#[cfg(test)]
mod undo_tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::core_geom::MaterialChoice;
    use crate::intent::{Intent, NodeSpec};
    use crate::scene::{Node, NodeBuilder, NodeContent, NodeGrids, Point, Scene};
    use crate::store::Store;
    use crate::voxel::{SdfShape, ShapeKind};

    /// A headless [`AppCore`] for the undo tests (no GPU — `apply_intent`/`undo`/`redo`
    /// only touch the borrowed scene + the owned command stack).
    fn test_core() -> AppCore {
        AppCore::new(Store::new(), OrbitCamera::default())
    }

    /// A box Tool shape of the given block size at density 8.
    fn box_shape(size: [u32; 3]) -> SdfShape {
        SdfShape {
            kind: ShapeKind::Box,
            size_blocks: size,
            wall_blocks: 1,
        }
    }

    /// A Tool node named after its kind (matching [`NodeSpec::into_node`]).
    fn tool_node(shape: SdfShape, material: MaterialChoice) -> Node {
        Node::new(format!("{:?}", shape.kind), NodeContent::Tool { shape, material })
    }

    /// A normalized two-Tool scene with stable ids minted + an Origin point, the first
    /// node active.
    fn two_tool_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone),
            tool_node(box_shape([3, 1, 4]), MaterialChoice::Wood),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        scene
    }

    /// Apply `intent`, asserting the round-trip invariant: `undo()` restores the
    /// scene byte-for-byte to `before`, and `redo()` restores it byte-for-byte to the
    /// post-apply `after`. Returns the core so the caller can inspect the stacks.
    fn assert_round_trips(scene: &mut Scene, intent: Intent) {
        let mut core = test_core();
        let before = scene.clone();
        core.apply_intent(scene, intent);
        let after = scene.clone();
        assert_ne!(*scene, before, "the forward op must change the scene");
        assert_eq!(core.undo_depth(), 1, "one command pushed");

        core.undo(scene);
        assert_eq!(*scene, before, "undo must restore the scene byte-for-byte");
        assert_eq!(core.undo_depth(), 0);
        assert_eq!(core.redo_depth(), 1);

        core.redo(scene);
        assert_eq!(*scene, after, "redo must restore the post-apply scene byte-for-byte");
        assert_eq!(core.undo_depth(), 1);
        assert_eq!(core.redo_depth(), 0);
    }

    // === Structural inverses (the correctness-critical arms) ===

    #[test]
    fn add_node_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::AddNode {
                content: NodeSpec::Tool {
                    shape: box_shape([5, 5, 5]),
                    material: MaterialChoice::Plain,
                },
            },
        );
    }

    #[test]
    fn add_child_round_trips() {
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into()],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let group = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::AddChild {
                group,
                content: NodeSpec::Tool {
                    shape: box_shape([4, 4, 4]),
                    material: MaterialChoice::Wood,
                },
            },
        );
    }

    #[test]
    fn group_node_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(&mut scene, Intent::GroupNode { target });
    }

    #[test]
    fn group_node_nested_round_trips() {
        // Group a node that already lives inside a Group — exercises the parent-spine
        // (not roots) slot restore.
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![
                tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
                tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
            ],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        // The second child of the Group.
        let group = scene.roots[0];
        let child = match &scene.arena[&group].content {
            NodeContent::Group(children) => children[1],
            _ => unreachable!(),
        };
        assert_round_trips(&mut scene, Intent::GroupNode { target: child });
    }

    #[test]
    fn make_definition_from_leaf_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::MakeDefinition { target, name: "House".to_string() },
        );
    }

    #[test]
    fn make_definition_from_group_round_trips() {
        // A Group active node DONATES its children to the def — the harder inverse
        // (restore the donated spine into the node's content, pop the def, no body).
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![
                tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
                tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
            ],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::MakeDefinition { target, name: "Body".to_string() },
        );
    }

    #[test]
    fn add_instance_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        scene.active = Some(target);
        let def = scene.make_definition_from_active("Body").expect("def made");
        assert_round_trips(&mut scene, Intent::AddInstance { def });
    }

    #[test]
    fn remove_leaf_node_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(&mut scene, Intent::RemoveNode { target });
    }

    #[test]
    fn remove_group_with_children_round_trips() {
        // The critical case: removing a Group detaches a whole subtree; the inverse
        // must re-insert every descendant under its original id at the original slot.
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
            NodeBuilder::group(
                "G",
                vec![
                    tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
                    NodeBuilder::group(
                        "Inner",
                        vec![tool_node(box_shape([1, 1, 1]), MaterialChoice::Plain).into()],
                    ),
                ],
            ),
            tool_node(box_shape([4, 4, 4]), MaterialChoice::Plain).into(),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let group = scene.roots[1];
        assert_round_trips(&mut scene, Intent::RemoveNode { target: group });
    }

    // === Field-set inverses ===

    #[test]
    fn set_visible_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(&mut scene, Intent::SetVisible { target, visible: false });
    }

    #[test]
    fn set_shape_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetShape { target, shape: box_shape([9, 9, 9]) },
        );
    }

    #[test]
    fn set_material_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetMaterial { target, material: MaterialChoice::Plain },
        );
    }

    #[test]
    fn set_offset_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(
            &mut scene,
            Intent::SetOffset { target, offset_blocks: [3, -2, 5] },
        );
    }

    #[test]
    fn set_name_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetName { target, name: "Renamed".to_string() },
        );
    }

    #[test]
    fn set_cloud_seed_round_trips() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let target = scene.roots[0];
        assert_round_trips(&mut scene, Intent::SetCloudSeed { target, seed: 42 });
    }

    #[test]
    fn set_node_grids_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetNodeGrids {
                target,
                grids: NodeGrids {
                    voxel_grid_on_faces: true,
                    block_lattice: true,
                    floor_grid: false,
                },
            },
        );
    }

    #[test]
    fn set_density_round_trips() {
        // Density is a single document-level field now (ADR 0003 §3f(0)); start from a
        // non-default prior so the inverse must restore the exact prior value, not 16.
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
            NodeBuilder::group(
                "G",
                vec![tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into()],
            ),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.voxels_per_block = 5;
        scene.active = scene.roots.first().copied();
        assert_round_trips(&mut scene, Intent::SetDensity { voxels_per_block: 20 });
    }

    #[test]
    fn set_grid_masters_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
        );
    }

    // === Point inverses ===

    #[test]
    fn add_point_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::AddPoint { position_blocks: [4, 0, -3], name: "Anchor".to_string() },
        );
    }

    #[test]
    fn remove_point_round_trips() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [1, 2, 3],
            ..Point::default()
        });
        assert_round_trips(&mut scene, Intent::RemovePoint { index: 1 });
    }

    #[test]
    fn set_point_hidden_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(&mut scene, Intent::SetPointHidden { index: 0, hidden: true });
    }

    #[test]
    fn set_point_planes_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetPointPlanes { index: 0, xz: false, xy: true, yz: true },
        );
    }

    #[test]
    fn set_point_axes_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetPointAxes { index: 0, x: false, y: true, z: false },
        );
    }

    #[test]
    fn set_point_position_round_trips() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [0, 0, 0],
            ..Point::default()
        });
        assert_round_trips(
            &mut scene,
            Intent::SetPointPosition { index: 1, position_blocks: [9, -1, 2] },
        );
    }

    // === Selection intents push NOTHING ===

    #[test]
    fn select_node_pushes_no_command() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[1];
        core.apply_intent(&mut scene, Intent::SelectNode { target: Some(target) });
        assert_eq!(core.undo_depth(), 0, "selection is not an undoable step");
        assert_eq!(scene.active, Some(target));
    }

    #[test]
    fn select_point_pushes_no_command() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SelectPoint { target: Some(0) });
        assert_eq!(core.undo_depth(), 0, "point selection is not an undoable step");
        assert_eq!(scene.active_point, Some(0));
    }

    // === No-op forward → no-op inverse (still pushes a command, undo restores nothing) ===

    #[test]
    fn field_write_to_missing_id_undo_is_noop() {
        let mut scene = two_tool_scene();
        let before = scene.clone();
        let mut core = test_core();
        core.apply_intent(
            &mut scene,
            Intent::SetName { target: crate::scene::NodeId(9999), name: "ghost".to_string() },
        );
        assert_eq!(scene, before, "a no-op forward changes nothing");
        core.undo(&mut scene);
        assert_eq!(scene, before, "undo of a no-op restores nothing");
    }

    // === Scripted realistic sequence ===

    #[test]
    fn scripted_sequence_undo_redo_round_trips() {
        let mut scene = two_tool_scene();
        let seed = scene.clone();
        let mut core = test_core();

        // A realistic authoring sequence.
        core.apply_intent(
            &mut scene,
            Intent::AddNode {
                content: NodeSpec::Tool {
                    shape: box_shape([2, 2, 2]),
                    material: MaterialChoice::Plain,
                },
            },
        );
        let added = scene.active.expect("added node selected");
        core.apply_intent(&mut scene, Intent::GroupNode { target: added });
        // The wrapped child is now active; group IT into a definition.
        let active = scene.active.expect("active after group");
        core.apply_intent(
            &mut scene,
            Intent::MakeDefinition { target: active, name: "Kit".to_string() },
        );
        let def = scene.definitions.last().expect("def made").id;
        core.apply_intent(&mut scene, Intent::AddInstance { def });
        let instance = scene.active.expect("instance selected");
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target: instance, offset_blocks: [7, 0, 0] },
        );
        core.apply_intent(&mut scene, Intent::RemoveNode { target: instance });

        let final_scene = scene.clone();
        assert_eq!(core.undo_depth(), 6, "six undoable steps");

        // Undo all the way back to the seed.
        for _ in 0..6 {
            core.undo(&mut scene);
        }
        assert_eq!(scene, seed, "undo all the way restores the seed byte-for-byte");

        // Redo all the way back to the final scene.
        for _ in 0..6 {
            core.redo(&mut scene);
        }
        assert_eq!(scene, final_scene, "redo all the way restores the final scene");
    }

    #[test]
    fn redo_cleared_after_apply() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetName { target, name: "First".to_string() });
        core.undo(&mut scene);
        assert_eq!(core.redo_depth(), 1, "undo populated redo");
        // A new, different apply must clear the redo future.
        core.apply_intent(&mut scene, Intent::SetName { target, name: "Second".to_string() });
        assert_eq!(core.redo_depth(), 0, "a fresh edit clears the redo stack");
    }

    // === effect_of routing: undo/redo return the per-intent effect, not blanket-true ===

    #[test]
    fn undo_of_field_edit_reports_scene_not_points() {
        // A trivial rename re-resolves the scene but must NOT force a points rebuild
        // (the per-edit cost ADR 0003 optimizes against at 10k nodes).
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetName { target, name: "Renamed".to_string() });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.scene_changed, "rename re-resolves the scene");
        assert!(!undo_effect.points_changed, "rename does not touch the points overlay");
        assert!(undo_effect.selection_changed, "undo always re-syncs the selection mirror");
        // And it is not the old blanket-true effect.
        assert_ne!(
            undo_effect,
            IntentEffect { scene_changed: true, points_changed: true, selection_changed: true },
            "undo no longer returns blanket-true",
        );
        let redo_effect = core.redo(&mut scene);
        assert!(redo_effect.scene_changed);
        assert!(!redo_effect.points_changed, "redo of a rename does not touch points");
    }

    #[test]
    fn undo_of_shape_edit_reports_scene_not_points() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetShape { target, shape: box_shape([9, 9, 9]) });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.scene_changed);
        assert!(!undo_effect.points_changed);
    }

    #[test]
    fn undo_of_point_edit_reports_points_not_scene() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SetPointHidden { index: 0, hidden: true });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.points_changed, "a point edit is overlay-only");
        assert!(!undo_effect.scene_changed, "a point edit triggers no voxel re-resolve");
        assert!(undo_effect.selection_changed);
    }

    #[test]
    fn undo_of_grid_masters_does_not_claim_scene_changed() {
        // The forward SetGridMasters effect is `none()` (masters are read live); undo
        // must match — claiming scene_changed would wrongly force a re-resolve.
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(
            &mut scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
        );
        let undo_effect = core.undo(&mut scene);
        assert!(!undo_effect.scene_changed, "grid masters need no re-resolve");
        assert!(!undo_effect.points_changed, "grid masters do not touch points");
        // Selection is still re-synced (undo restores selection_before).
        assert!(undo_effect.selection_changed);
        let redo_effect = core.redo(&mut scene);
        assert!(!redo_effect.scene_changed, "redo of grid masters needs no re-resolve");
    }

    /// Count the `GRID_OVERLAY_BIT`-flagged voxels in a fresh `rebuild` of `scene` at
    /// `density`. `rebuild` routes through the per-chunk store (the chunk cache), so
    /// this exercises the SAME invalidation path the live app uses — not the
    /// always-full `resolve_region`.
    fn rebuild_grid_overlay_count(core: &mut AppCore, scene: &Scene, density: u32) -> usize {
        match core.rebuild(scene, density) {
            RebuildOutcome::Built(output) => output
                .grid
                .occupied
                .iter()
                .filter(|voxel| voxel.material_id & crate::voxel::GRID_OVERLAY_BIT != 0)
                .count(),
            RebuildOutcome::DensityRejected { .. } => {
                panic!("density {density} unexpectedly rejected")
            }
        }
    }

    /// The occupied-voxel CORNER bounding box of a single `shape` of `size_blocks` at
    /// offset `[0, 0, 0]`, resolved at `density` through **`AppCore::rebuild`** — the
    /// per-chunk store path the WINDOWED APP actually renders. Returns
    /// `(min_corner, max_corner)` per axis in absolute voxel units (the half-open box
    /// `[min, max)`; voxel centres sit at `n + 0.5`, so the corner is `floor(centre)`
    /// for the min and `floor(centre) + 1` for the max).
    fn rebuild_frame_corner_bbox(
        shape: SdfShape,
        density: u32,
    ) -> ([i64; 3], [i64; 3]) {
        let mut scene = Scene::from_nodes(vec![tool_node(shape, MaterialChoice::Stone)]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.voxels_per_block = density;
        scene.active = scene.roots.first().copied();
        let mut core = test_core();
        let RebuildOutcome::Built(output) = core.rebuild(&scene, density) else {
            panic!("density {density} unexpectedly rejected");
        };
        assert!(!output.grid.occupied.is_empty(), "shape resolved empty");
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in &output.grid.occupied {
            for axis in 0..3 {
                let corner = voxel.world_position[axis].floor() as i64;
                min[axis] = min[axis].min(corner);
                max[axis] = max[axis].max(corner + 1); // half-open upper bound
            }
        }
        (min, max)
    }

    /// PERMANENT GUARD (corrects the coordinator's mistaken premise). A shape placed
    /// at world offset `[0, 0, 0]` is rendered CENTRED ON THE WORLD ORIGIN through
    /// the `AppCore::rebuild` / per-chunk store path — the exact path the windowed app
    /// renders. This pins the EMPIRICAL render-frame coordinates so the convention can
    /// never be misdescribed again.
    ///
    /// The per-chunk store applies the composite recentre (`Store::bind_region`
    /// rebases every chunk to the composite's recentre / floating origin), so the
    /// rebuild grid is in the SAME centred frame as the monolithic `resolve_region`
    /// (bit-identical for a near scene — proven by the goldens). The #30 lattice shift
    /// snaps the producer grid onto the block lattice in the PRODUCER-TRUE
    /// (pre-recentre) frame, but the recentre then re-symmetrises the composite about
    /// the origin — so the shape the user sees is centred, NOT corner-at-origin.
    ///
    /// Measured coordinates (this test pins them):
    ///   * 1×1×1 box  @ d16 → `[−8, 8)`  per axis  (d8 → `[−4, 4)`)  — centred, NOT `[0, 16)`.
    ///   * 5×5×5 sphere @ d16 → `[−40, 40)` per axis (d8 → `[−20, 20)`).
    ///   * 5×1×5 box  @ d16 → X/Z `[−40, 40)`, Y `[−8, 8)` (d8 → `[−20, 20)`, `[−4, 4)`).
    ///
    /// We assert the CORNER bbox is symmetric (`min + max == 0`): an odd voxel span
    /// (`size·d` is even here, so the span is even-in-voxels) makes the corner bbox
    /// exactly symmetric, with a voxel BOUNDARY on the origin.
    #[test]
    fn shapes_render_centered_on_origin_in_rebuild_frame() {
        use crate::voxel::ShapeKind;
        let cases: [(ShapeKind, [u32; 3]); 3] = [
            (ShapeKind::Box, [1, 1, 1]),
            (ShapeKind::Sphere, [5, 5, 5]),
            (ShapeKind::Box, [5, 1, 5]),
        ];
        for density in [8u32, 16] {
            for (kind, size) in cases {
                let shape = SdfShape { kind, size_blocks: size, wall_blocks: 1 };
                let (min, max) = rebuild_frame_corner_bbox(shape, density);
                for axis in 0..3 {
                    // Centred: the half-open corner box is symmetric about 0.
                    assert_eq!(
                        min[axis] + max[axis],
                        0,
                        "{kind:?} {size:?}@d{density} axis {axis}: rebuild-frame corner bbox \
                         [{}, {}) must be centred on the origin (min + max == 0)",
                        min[axis], max[axis]
                    );
                    // …and spans exactly size·d voxels (no clipping / no half-block leak).
                    assert_eq!(
                        max[axis] - min[axis],
                        (size[axis] * density) as i64,
                        "{kind:?} {size:?}@d{density} axis {axis}: span must be size·d voxels"
                    );
                }
            }
        }
        // Pin the exact 1×1×1 @ d16 box so the convention is unambiguous: it occupies
        // [−8, 8) per axis (centred), NOT [0, 16) (corner-at-origin).
        let one_block = SdfShape { kind: ShapeKind::Box, size_blocks: [1, 1, 1], wall_blocks: 1 };
        let (min, max) = rebuild_frame_corner_bbox(one_block, 16);
        assert_eq!(min, [-8, -8, -8], "1×1×1 box @ d16 min corner is centred at −8, not 0");
        assert_eq!(max, [8, 8, 8], "1×1×1 box @ d16 max corner is centred at +8, not 16");
    }

    /// Regression (FIX 1): toggling ONLY `voxel_grid_on_faces` must make the on-face
    /// grid appear on the FIRST rebuild — no unrelated edit needed to evict the
    /// stale cached chunks.
    ///
    /// The flag is baked into the resolved voxels as `GRID_OVERLAY_BIT`, but it had
    /// been OMITTED from the leaf content fingerprint. So a lone toggle produced an
    /// identical fingerprint → `edit_aabb_since` found nothing dirty → `rebuild`
    /// evicted no chunks → the cached (grid-less) chunks were reused, and the grid
    /// only "caught up" when a later move/resize/etc. happened to evict them. Folding
    /// the flag into the fingerprint dirties the leaf's AABB on the toggle itself.
    #[test]
    fn toggling_voxel_grid_on_faces_appears_on_first_rebuild() {
        let mut scene = Scene::from_nodes(vec![tool_node(box_shape([3, 3, 3]), MaterialChoice::Stone)]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let target = scene.roots[0];
        let density = 8;

        let mut core = test_core();

        // Seed the chunk cache with a rebuild while the flag is OFF: zero flagged
        // voxels, and (critically) this populates the store + `previous_leaf_index`
        // so the NEXT rebuild diffs against it.
        let before = rebuild_grid_overlay_count(&mut core, &scene, density);
        assert_eq!(before, 0, "with the flag OFF no voxel may carry GRID_OVERLAY_BIT");

        // Flip ONLY voxel_grid_on_faces ON via the intent door (no other edit).
        core.apply_intent(
            &mut scene,
            Intent::SetNodeGrids {
                target,
                grids: NodeGrids { voxel_grid_on_faces: true, ..NodeGrids::default() },
            },
        );

        // Rebuild AGAIN. Before the fix the fingerprint was unchanged → no chunk
        // evicted → this stayed 0. With the flag in the fingerprint the leaf's AABB
        // reports dirty, its chunks re-resolve WITH the bit, and the grid appears now.
        let after = rebuild_grid_overlay_count(&mut core, &scene, density);
        assert!(
            after > 0,
            "after toggling voxel_grid_on_faces ON, the FIRST rebuild must flag voxels \
             (was {before}, now {after}) — no unrelated edit should be needed"
        );
    }
}
