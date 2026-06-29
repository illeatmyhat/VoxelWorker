//! The [`Intent`] boundary — the single serializable description of a mutation
//! (ADR 0003 Phase C, slice C1).
//!
//! ADR 0003 Phase B made identity / selection / edit-ops / storage all key on a
//! stable [`NodeId`](crate::scene::NodeId). Phase C introduces `Intent` as the one
//! **serializable** description of every document mutation, so the live flow
//! becomes `ui → AppCore::apply_intent(Intent) → (later) Command`. An `Intent` is a
//! pure value: it names WHAT to change (by stable id / index), never HOW the panel
//! reached the change, so it survives serialization, scripting (`shot --replay`,
//! C3) and undo (`CommandStack`, C2).
//!
//! **C1 is additive only.** This module + [`AppCore::apply_intent`] sit ALONGSIDE
//! the current panel-mutates-`Scene`-directly flow; nothing in the live path calls
//! `apply_intent` yet (only the lib tests do), so the goldens stay byte-identical.
//! `apply_intent` dispatches each variant to the SAME [`Scene`](crate::scene::Scene)
//! edit op / field write the panel uses today and returns an [`IntentEffect`] — the
//! typed successor of [`PanelResponse`](crate::panel::PanelResponse)'s effect
//! booleans — so a later slice (C4) can drop it in for the panel's flag bag. No
//! `CommandStack` / undo yet (that is C2): `apply_intent` just dispatches + reports.

use serde::{Deserialize, Serialize};

use crate::core_geom::MaterialChoice;
use crate::scene::{DefId, Node, NodeContent, NodeGrids, NodeId, Part};
use crate::units::Measurement;
use crate::voxel::SdfShape;

/// A **by-value node payload** for the structural add intents (ADR 0003 Phase C).
///
/// The add edit ops ([`Scene::add_node`](crate::scene::Scene::add_node) /
/// [`Scene::add_child_to_group`](crate::scene::Scene::add_child_to_group)) take a
/// [`Node`], but a `Node` carries a non-serializable-by-intent id slot + grid flags
/// the caller never sets when adding. `NodeSpec` is the small serializable spec of
/// "what to add"; [`NodeSpec::into_node`] turns it into the exact [`Node`] the panel
/// builds today (the Tool's name is its shape kind's label; the Clouds Part is named
/// `"Clouds"` with seed `0`), so an `AddNode` intent reproduces the panel's add.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeSpec {
    /// A parametric Tool node (an [`SdfShape`] + its single [`MaterialChoice`]),
    /// named after the shape kind exactly as the panel's `new_tool_node` does.
    Tool {
        /// The Tool's parametric primitive.
        shape: SdfShape,
        /// The single material the Tool stamps onto its voxels.
        material: MaterialChoice,
    },
    /// A debug-cloud [`Part`] node, named `"Clouds"` with seed `0` — the panel's
    /// "Clouds (Part)" add.
    CloudsPart,
}

impl NodeSpec {
    /// The shape-kind label the panel's `SHAPE_CHIPS` use as a Tool node's name
    /// (`format!("{:?}", kind)` — the `Debug` rendering of the [`ShapeKind`]). Kept
    /// here so an `AddNode { NodeSpec::Tool }` mints a node byte-identical to the
    /// panel's `new_tool_node` (which labels it with the chip label, i.e. the kind
    /// name).
    ///
    /// [`ShapeKind`]: crate::voxel::ShapeKind
    fn tool_node_name(shape: &SdfShape) -> String {
        format!("{:?}", shape.kind)
    }

    /// Turn the spec into the [`Node`] the add edit ops expect — mirroring how the
    /// panel builds these nodes today (the Tool name = its shape kind label; the
    /// Clouds Part = `"Clouds"` + [`Part::DebugClouds`] seed `0`). The returned node
    /// carries the unassigned [`NodeId(0)`](NodeId) sentinel; the add op mints its
    /// real id.
    pub fn into_node(self) -> Node {
        match self {
            NodeSpec::Tool { shape, material } => {
                let name = NodeSpec::tool_node_name(&shape);
                Node::new(name, NodeContent::Tool { shape, material })
            }
            NodeSpec::CloudsPart => {
                Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 }))
            }
        }
    }
}

/// The single serializable description of one document mutation (ADR 0003 Phase C).
///
/// Every variant names a mutation by **stable identity** ([`NodeId`] / [`DefId`] /
/// a point index), never by the positional path or the panel state the panel
/// happened to reach it through, so an `Intent` round-trips through serde and is
/// replayable. [`AppCore::apply_intent`](crate::AppCore::apply_intent) dispatches
/// each variant to the matching [`Scene`](crate::scene::Scene) edit op / field
/// write — the SAME mutation the panel performs today — and reports an
/// [`IntentEffect`].
///
/// The variants mirror the panel's mutation surface: structural tree edits, node
/// field writes, the two global toggles (density, grid masters), the view-state
/// selection edits, and the reference-point edits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Intent {
    // --- Structural (tree shape) ---
    /// Add a top-level node built from `content`
    /// ([`Scene::add_node`](crate::scene::Scene::add_node)).
    AddNode {
        /// The node to add, by value.
        content: NodeSpec,
    },
    /// Add a child built from `content` into the Group identified by `group`
    /// ([`Scene::add_child_to_group`](crate::scene::Scene::add_child_to_group)).
    AddChild {
        /// The target Group's stable id.
        group: NodeId,
        /// The child node to add, by value.
        content: NodeSpec,
    },
    /// Wrap the node `target` in a new Group
    /// ([`Scene::group_active`](crate::scene::Scene::group_active), reached by
    /// pointing `active` at `target`).
    GroupNode {
        /// The node to wrap.
        target: NodeId,
    },
    /// Turn `target` into a reusable definition named `name` + replace it with an
    /// Instance of it
    /// ([`Scene::make_definition_from_active`](crate::scene::Scene::make_definition_from_active)).
    MakeDefinition {
        /// The node to lift into a definition.
        target: NodeId,
        /// The new definition's name.
        name: String,
    },
    /// Place another Instance of definition `def`
    /// ([`Scene::add_instance`](crate::scene::Scene::add_instance)).
    AddInstance {
        /// The definition to instance.
        def: DefId,
    },
    /// Remove the node `target` and re-derive a sensible selection
    /// ([`Scene::remove_node`](crate::scene::Scene::remove_node)).
    RemoveNode {
        /// The node to remove.
        target: NodeId,
    },

    // --- Node field writes ---
    /// Set the `visible` flag of `target`
    /// ([`Scene::set_node_visible`](crate::scene::Scene::set_node_visible)).
    SetVisible {
        /// The node to retarget.
        target: NodeId,
        /// The new visibility.
        visible: bool,
    },
    /// Set the [`SdfShape`] of the Tool node `target` (a no-op for a non-Tool node).
    SetShape {
        /// The Tool node to edit.
        target: NodeId,
        /// The new shape.
        shape: SdfShape,
    },
    /// Set the [`MaterialChoice`] of the Tool node `target` (a no-op for a non-Tool
    /// node).
    SetMaterial {
        /// The Tool node to edit.
        target: NodeId,
        /// The new material.
        material: MaterialChoice,
    },
    /// Set the offset of `target`'s transform from a per-axis authored unit
    /// expression (ADR 0003 §3f(0)).
    SetOffset {
        /// The node to move.
        target: NodeId,
        /// The new per-axis offset as RETAINED [`Measurement`]s (blocks + voxels,
        /// signed). The apply path derives the canonical voxel offset via
        /// [`Measurement::to_voxels`] at the document density; the inspector
        /// guarantees each axis lands on a whole voxel before emitting. The
        /// measurements are retained on the transform for lossless density
        /// re-targeting and exact-expression undo.
        offset_measurements: [Measurement; 3],
    },
    /// Set the name of `target`.
    SetName {
        /// The node to rename.
        target: NodeId,
        /// The new name.
        name: String,
    },
    /// Set the seed of the Clouds [`Part`] node `target` (a no-op for a non-Clouds
    /// node).
    SetCloudSeed {
        /// The Clouds Part node to edit.
        target: NodeId,
        /// The new seed.
        seed: u32,
    },
    /// Set the per-node grid flags of `target`.
    SetNodeGrids {
        /// The node to edit.
        target: NodeId,
        /// The new per-node grid display settings.
        grids: NodeGrids,
    },

    // --- Global ---
    /// Set the document-level density (voxels per block). Density is a single attribute
    /// on the [`Scene`](crate::scene::Scene) — which block-game grid the plan targets
    /// (ADR 0003 §3f(0)) — so this writes `scene.voxels_per_block`, not a per-Tool field.
    SetDensity {
        /// The new document voxels-per-block.
        voxels_per_block: u32,
    },
    /// Set the three scene-wide grid master toggles.
    SetGridMasters {
        /// The new voxel-grid-on-faces master.
        voxel: bool,
        /// The new block-lattice master.
        lattice: bool,
        /// The new floor-grid master.
        floor: bool,
    },

    // --- Selection (view state, but a valid mutation intent) ---
    /// Set (or clear) the active node selection.
    SelectNode {
        /// The node to select, or `None` to clear.
        target: Option<NodeId>,
    },
    /// Set (or clear) the active point selection.
    SelectPoint {
        /// The point index to select, or `None` to clear.
        target: Option<usize>,
    },

    // --- Points (reference elements) ---
    /// Add a reference [`Point`](crate::scene::Point) at `position_blocks` named
    /// `name` ([`Scene::add_point`](crate::scene::Scene::add_point), which forces the
    /// "+ Add Point" default flags).
    AddPoint {
        /// The new point's whole-block position.
        position_blocks: [i64; 3],
        /// The new point's name.
        name: String,
    },
    /// Remove the point at `index` (a no-op on the Origin / out-of-range index).
    RemovePoint {
        /// The point index to remove.
        index: usize,
    },
    /// Set the `hidden` flag of the point at `index`.
    SetPointHidden {
        /// The point index to edit.
        index: usize,
        /// The new hidden flag.
        hidden: bool,
    },
    /// Set the three plane toggles of the point at `index`.
    SetPointPlanes {
        /// The point index to edit.
        index: usize,
        /// The FRONT (XZ, normal +Y) plane flag (Z-up).
        xz: bool,
        /// The GROUND (XY, normal +Z) plane flag (Z-up).
        xy: bool,
        /// The side (YZ, normal +X) plane flag.
        yz: bool,
    },
    /// Set the three axis toggles of the point at `index`.
    SetPointAxes {
        /// The point index to edit.
        index: usize,
        /// The +X axis flag.
        x: bool,
        /// The +Y axis flag.
        y: bool,
        /// The +Z axis flag.
        z: bool,
    },
    /// Set the whole-block position of the point at `index`.
    SetPointPosition {
        /// The point index to edit.
        index: usize,
        /// The new whole-block position.
        position_blocks: [i64; 3],
    },
}

/// The typed effect of applying an [`Intent`] — the successor of
/// [`PanelResponse`](crate::panel::PanelResponse)'s effect booleans (ADR 0003 Phase
/// C). `apply_intent` returns this so a caller can react exactly as the panel does
/// today: re-resolve the scene on a geometry/scene change, persist on a points
/// change, refresh the inspector mirror on a selection change.
///
/// The flag semantics MATCH the panel's: a structural / field / global-geometry
/// mutation sets [`scene_changed`](Self::scene_changed) (the caller re-resolves +
/// re-frames, exactly as `PanelResponse::scene_changed` drives); a point mutation
/// sets [`points_changed`](Self::points_changed) (overlay-only, no re-resolve); a
/// selection mutation sets [`selection_changed`](Self::selection_changed). A
/// master-toggle / selection mutation needs no re-resolve (the per-frame batch /
/// highlight read the fields live), matching the panel.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IntentEffect {
    /// The scene's geometry changed → the caller re-resolves the grid + re-frames
    /// (the typed successor of [`PanelResponse::scene_changed`] /
    /// `geometry_changed`, which the caller already treats identically as "rebuild").
    ///
    /// [`PanelResponse::scene_changed`]: crate::panel::PanelResponse::scene_changed
    pub scene_changed: bool,
    /// A reference Point changed → the caller may persist (Points are pure overlay,
    /// rebuilt every frame, so this does NOT trigger a voxel re-resolve — matching
    /// [`PanelResponse::points_changed`]).
    ///
    /// [`PanelResponse::points_changed`]: crate::panel::PanelResponse::points_changed
    pub points_changed: bool,
    /// The active node / point selection changed → the caller refreshes the
    /// inspector mirror (the panel folds this into `scene_changed` today, but the
    /// typed effect separates it so a pure selection switch re-resolves nothing).
    pub selection_changed: bool,
}

impl IntentEffect {
    /// The empty effect (nothing changed) — what a no-op intent (a field write to a
    /// missing id, a non-Tool `SetShape`, …) returns.
    pub fn none() -> Self {
        Self::default()
    }

    /// An effect flagging only a scene-geometry change (re-resolve + re-frame).
    pub fn scene() -> Self {
        Self {
            scene_changed: true,
            ..Self::none()
        }
    }

    /// An effect flagging only a points change (persist, no re-resolve).
    pub fn points() -> Self {
        Self {
            points_changed: true,
            ..Self::none()
        }
    }

    /// An effect flagging only a selection change.
    pub fn selection() -> Self {
        Self {
            selection_changed: true,
            ..Self::none()
        }
    }

    /// The OR-merge of two effects — the union of their set flags. Useful when a
    /// later slice batches several intents into one frame's effect.
    pub fn merged_with(self, other: Self) -> Self {
        Self {
            scene_changed: self.scene_changed || other.scene_changed,
            points_changed: self.points_changed || other.points_changed,
            selection_changed: self.selection_changed || other.selection_changed,
        }
    }
}

/// Build a per-axis whole-**block** offset measurement (test helper for the
/// `SetOffset` intent, which now carries `[Measurement; 3]`). Each axis is a pure
/// integer block term, so it derives to `blocks · d` voxels at any density — the
/// same result the old block-granular path produced.
#[cfg(test)]
pub(crate) fn whole_block_offset(blocks: [i64; 3]) -> [Measurement; 3] {
    use crate::units::ExactRational;
    [
        Measurement::new(ExactRational::from_integer(blocks[0] as i128), 0),
        Measurement::new(ExactRational::from_integer(blocks[1] as i128), 0),
        Measurement::new(ExactRational::from_integer(blocks[2] as i128), 0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app_core::AppCore;
    use crate::camera::OrbitCamera;
    use crate::scene::{Node, NodeBuilder, NodeTransform, Point, Scene};
    use crate::store::Store;
    use crate::voxel::{ShapeKind, SdfShape};

    /// A headless [`AppCore`] for the dispatch tests. `apply_intent` reads no AppCore
    /// state (it borrows the scene), so a default store + camera suffice — no GPU.
    fn test_core() -> AppCore {
        AppCore::new(Store::new(), OrbitCamera::default())
    }

    /// A box Tool shape at the given size (the default-ish fixture shape).
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

    /// A normalized two-Tool scene with stable ids minted, the first node active.
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

    /// The stable id of the top-level node at `index`.
    fn root_id(scene: &Scene, index: usize) -> NodeId {
        scene.roots[index]
    }

    /// Assert `apply_intent(intent)` produces the SAME scene as `direct` applied to a
    /// clone — the core `apply_intent ≡ direct op` invariant. Both sides start from
    /// the SAME scene state (so id-minting counters match), so the scenes compare
    /// equal (Scene derives PartialEq).
    fn assert_dispatch_matches(scene: &Scene, intent: Intent, direct: impl FnOnce(&mut Scene)) {
        let mut core = test_core();
        let mut applied = scene.clone();
        core.apply_intent(&mut applied, intent);
        let mut expected = scene.clone();
        direct(&mut expected);
        assert_eq!(applied, expected);
    }

    // === Structural ===

    #[test]
    fn add_node_dispatches_to_add_node() {
        let scene = two_tool_scene();
        let spec = NodeSpec::Tool {
            shape: box_shape([5, 5, 5]),
            material: MaterialChoice::Plain,
        };
        assert_dispatch_matches(
            &scene,
            Intent::AddNode { content: spec.clone() },
            |s| {
                s.add_node(spec.into_node());
            },
        );
    }

    #[test]
    fn add_node_clouds_part_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::AddNode { content: NodeSpec::CloudsPart },
            |s| {
                s.add_node(NodeSpec::CloudsPart.into_node());
            },
        );
    }

    #[test]
    fn add_child_dispatches_to_add_child_to_group() {
        // A scene with a Group so the child has somewhere to land.
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into()],
        )]);
        scene.ensure_node_ids();
        let group_id = root_id(&scene, 0);
        let spec = NodeSpec::Tool {
            shape: box_shape([4, 4, 4]),
            material: MaterialChoice::Wood,
        };
        assert_dispatch_matches(
            &scene,
            Intent::AddChild { group: group_id, content: spec.clone() },
            |s| {
                s.add_child_to_group(group_id, spec.into_node());
            },
        );
    }

    #[test]
    fn group_node_dispatches_via_active() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(&scene, Intent::GroupNode { target }, |s| {
            s.active = Some(target);
            s.group_active();
        });
    }

    #[test]
    fn make_definition_dispatches_via_active() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::MakeDefinition { target, name: "House".to_string() },
            |s| {
                s.active = Some(target);
                s.make_definition_from_active("House".to_string());
            },
        );
    }

    #[test]
    fn add_instance_dispatches() {
        // Build a scene that already has a definition to instance.
        let mut scene = two_tool_scene();
        let target = root_id(&scene, 0);
        scene.active = Some(target);
        let def_id = scene.make_definition_from_active("Body").expect("definition made");
        assert_dispatch_matches(&scene, Intent::AddInstance { def: def_id }, |s| {
            s.add_instance(def_id);
        });
    }

    #[test]
    fn remove_node_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(&scene, Intent::RemoveNode { target }, |s| {
            s.remove_node(target);
        });
    }

    // === Node field writes ===

    #[test]
    fn set_visible_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetVisible { target, visible: false },
            |s| {
                s.set_node_visible(target, false);
            },
        );
    }

    #[test]
    fn set_shape_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let shape = box_shape([7, 7, 7]);
        assert_dispatch_matches(&scene, Intent::SetShape { target, shape }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                if let NodeContent::Tool { shape: node_shape, .. } = &mut node.content {
                    *node_shape = shape;
                }
            }
        });
    }

    #[test]
    fn set_material_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetMaterial { target, material: MaterialChoice::Plain },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    if let NodeContent::Tool { material, .. } = &mut node.content {
                        *material = MaterialChoice::Plain;
                    }
                }
            },
        );
    }

    #[test]
    fn set_offset_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(
            &scene,
            Intent::SetOffset { target, offset_measurements: whole_block_offset([3, -2, 5]) },
            |s| {
                // `apply` derives canonical voxels from the per-axis measurement at
                // the document density (ADR 0003 §3f(0)); mirror that here.
                let density = s.voxels_per_block;
                if let Some(node) = s.node_by_id_mut(target) {
                    node.transform =
                        NodeTransform::from_measurements(whole_block_offset([3, -2, 5]), density);
                }
            },
        );
    }

    #[test]
    fn set_name_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetName { target, name: "Renamed".to_string() },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    node.name = "Renamed".to_string();
                }
            },
        );
    }

    #[test]
    fn set_cloud_seed_dispatches() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(&scene, Intent::SetCloudSeed { target, seed: 42 }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                if let NodeContent::Part(Part::DebugClouds { seed }) = &mut node.content {
                    *seed = 42;
                }
            }
        });
    }

    #[test]
    fn set_node_grids_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: true,
            floor_grid: false,
        };
        assert_dispatch_matches(&scene, Intent::SetNodeGrids { target, grids }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                node.grids = grids;
            }
        });
    }

    #[test]
    fn field_write_to_missing_id_is_noop() {
        let scene = two_tool_scene();
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(
            &mut applied,
            Intent::SetName { target: NodeId(9999), name: "ghost".to_string() },
        );
        assert_eq!(applied, scene);
        assert_eq!(effect, IntentEffect::none());
    }

    #[test]
    fn set_shape_on_non_tool_is_noop() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        let target = root_id(&scene, 0);
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect =
            core.apply_intent(&mut applied, Intent::SetShape { target, shape: box_shape([2, 2, 2]) });
        assert_eq!(applied, scene);
        assert_eq!(effect, IntentEffect::none());
    }

    // === Global ===

    #[test]
    fn set_density_sets_document_field() {
        // Density is a single document-level field (ADR 0003 §3f(0)): the dispatch sets
        // `scene.voxels_per_block`, not a per-Tool fan-out.
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
            NodeBuilder::group(
                "G",
                vec![tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into()],
            ),
            NodeSpec::CloudsPart.into_node().into(),
        ]);
        scene.ensure_node_ids();
        assert_dispatch_matches(&scene, Intent::SetDensity { voxels_per_block: 20 }, |s| {
            // `apply` rescales every node's voxel offset old→new density to preserve
            // block placement (ADR 0003 §3f(0)); mirror that here. (Every node in this
            // scene has a zero offset, so the rescale is a no-op, but mirror it anyway
            // so the equivalence stays honest.)
            let old_density = s.voxels_per_block.max(1) as i64;
            for node in s.arena.values_mut() {
                for axis in 0..3 {
                    node.transform.offset_voxels[axis] =
                        node.transform.offset_voxels[axis] * 20 / old_density;
                }
            }
            s.voxels_per_block = 20;
        });
    }

    #[test]
    fn set_grid_masters_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
            |s| {
                s.master_voxel_grid = false;
                s.master_block_lattice = true;
                s.master_floor_grid = false;
            },
        );
    }

    // === Selection ===

    #[test]
    fn select_node_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(&mut applied, Intent::SelectNode { target: Some(target) });
        let mut expected = scene.clone();
        expected.active = Some(target);
        assert_eq!(applied, expected);
        assert_eq!(effect, IntentEffect::selection());
    }

    #[test]
    fn select_point_dispatches() {
        let scene = two_tool_scene();
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(&mut applied, Intent::SelectPoint { target: Some(0) });
        let mut expected = scene.clone();
        expected.active_point = Some(0);
        assert_eq!(applied, expected);
        assert_eq!(effect, IntentEffect::selection());
    }

    // === Points ===

    #[test]
    fn add_point_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::AddPoint { position_blocks: [4, 0, -3], name: "Anchor".to_string() },
            |s| {
                let point = Point {
                    name: "Anchor".to_string(),
                    position_blocks: [4, 0, -3],
                    ..Point::default()
                };
                s.add_point(point);
            },
        );
    }

    #[test]
    fn remove_point_dispatches() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [1, 2, 3],
            ..Point::default()
        });
        assert_dispatch_matches(&scene, Intent::RemovePoint { index: 1 }, |s| {
            s.remove_point(1);
        });
    }

    #[test]
    fn set_point_hidden_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointHidden { index: 0, hidden: true },
            |s| {
                s.points[0].hidden = true;
            },
        );
    }

    #[test]
    fn set_point_planes_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointPlanes { index: 0, xz: false, xy: true, yz: true },
            |s| {
                s.points[0].plane_xz = false;
                s.points[0].plane_xy = true;
                s.points[0].plane_yz = true;
            },
        );
    }

    #[test]
    fn set_point_axes_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointAxes { index: 0, x: false, y: true, z: false },
            |s| {
                s.points[0].axis_x = false;
                s.points[0].axis_y = true;
                s.points[0].axis_z = false;
            },
        );
    }

    #[test]
    fn set_point_position_dispatches() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [0, 0, 0],
            ..Point::default()
        });
        assert_dispatch_matches(
            &scene,
            Intent::SetPointPosition { index: 1, position_blocks: [9, -1, 2] },
            |s| {
                s.points[1].position_blocks = [9, -1, 2];
            },
        );
    }

    // === serde round-trip: every variant serializes → deserializes to itself ===

    #[test]
    fn every_intent_variant_round_trips_through_json() {
        let shape = box_shape([2, 3, 4]);
        let grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let variants = vec![
            Intent::AddNode {
                content: NodeSpec::Tool { shape, material: MaterialChoice::Wood },
            },
            Intent::AddNode { content: NodeSpec::CloudsPart },
            Intent::AddChild {
                group: NodeId(7),
                content: NodeSpec::Tool { shape, material: MaterialChoice::Plain },
            },
            Intent::GroupNode { target: NodeId(3) },
            Intent::MakeDefinition { target: NodeId(3), name: "House".to_string() },
            Intent::AddInstance { def: DefId(2) },
            Intent::RemoveNode { target: NodeId(5) },
            Intent::SetVisible { target: NodeId(1), visible: false },
            Intent::SetShape { target: NodeId(1), shape },
            Intent::SetMaterial { target: NodeId(1), material: MaterialChoice::Stone },
            Intent::SetOffset { target: NodeId(1), offset_measurements: whole_block_offset([-1, 2, -3]) },
            Intent::SetName { target: NodeId(1), name: "Foo".to_string() },
            Intent::SetCloudSeed { target: NodeId(1), seed: 9 },
            Intent::SetNodeGrids { target: NodeId(1), grids },
            Intent::SetDensity { voxels_per_block: 16 },
            Intent::SetGridMasters { voxel: true, lattice: false, floor: true },
            Intent::SelectNode { target: Some(NodeId(4)) },
            Intent::SelectNode { target: None },
            Intent::SelectPoint { target: Some(2) },
            Intent::SelectPoint { target: None },
            Intent::AddPoint { position_blocks: [1, 2, 3], name: "P".to_string() },
            Intent::RemovePoint { index: 1 },
            Intent::SetPointHidden { index: 0, hidden: true },
            Intent::SetPointPlanes { index: 0, xz: true, xy: false, yz: true },
            Intent::SetPointAxes { index: 0, x: true, y: false, z: true },
            Intent::SetPointPosition { index: 0, position_blocks: [4, 5, 6] },
        ];
        for intent in variants {
            let json = serde_json::to_string(&intent).expect("serialize");
            let back: Intent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(intent, back, "round-trip mismatch for {intent:?}");
        }
    }

    #[test]
    fn node_spec_into_node_matches_panel_tool_naming() {
        // A Tool spec yields a node named after its kind (the panel's chip label),
        // wrapping the same shape + material.
        let shape = box_shape([2, 2, 2]);
        let node = NodeSpec::Tool { shape, material: MaterialChoice::Wood }.into_node();
        assert_eq!(node.name, "Box");
        assert_eq!(node.transform, NodeTransform::identity());
        match node.content {
            NodeContent::Tool { shape: s, material } => {
                assert_eq!(s, shape);
                assert_eq!(material, MaterialChoice::Wood);
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }
}
