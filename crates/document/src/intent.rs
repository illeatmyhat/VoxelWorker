//! The [`Intent`] boundary — the single serializable description of a mutation
//! (ADR 0003 Phase C, slice C1).
//!
//! ADR 0003 Phase B made identity / selection / edit-ops / storage all key on a
//! stable [`NodeId`]. Phase C introduces `Intent` as the one
//! **serializable** description of every document mutation, so the live flow
//! becomes `ui → AppCore::apply_intent(Intent) → (later) Command`. An `Intent` is a
//! pure value: it names WHAT to change (by stable id / index), never HOW the panel
//! reached the change, so it survives serialization, scripting (`shot --replay`,
//! C3) and undo (`CommandStack`, C2).
//!
//! **C1 is additive only.** This module + `AppCore::apply_intent` sit ALONGSIDE
//! the current panel-mutates-`Scene`-directly flow; nothing in the live path calls
//! `apply_intent` yet (only the lib tests do), so the goldens stay byte-identical.
//! `apply_intent` dispatches each variant to the SAME [`Scene`](crate::scene::Scene)
//! edit op / field write the panel uses today and returns an [`IntentEffect`] — the
//! typed successor of `PanelResponse`'s effect
//! booleans — so a later slice (C4) can drop it in for the panel's flag bag. No
//! `CommandStack` / undo yet (that is C2): `apply_intent` just dispatches + reports.

use serde::{Deserialize, Serialize};

use voxel_core::core_geom::MaterialChoice;
use crate::scene::{CombineOp, DefId, Node, NodeContent, NodeGrids, NodeId, Part};
use crate::sketch::SketchSolid;
use voxel_core::units::Measurement;
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
    /// A sketch→operation Tool node (a [`SketchSolid`] producer + its single
    /// [`MaterialChoice`]), named `"Sketch"` — the sketch-authoring add (ADR 0003
    /// §3i). Carries the whole producer by value, mirroring how [`Tool`](Self::Tool)
    /// carries its [`SdfShape`].
    Sketch {
        /// The sketch + operation this node resolves.
        producer: SketchSolid,
        /// The single material the sketch node stamps onto its voxels.
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
    /// [`ShapeKind`]: voxel_core::voxel::ShapeKind
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
            NodeSpec::Sketch { producer, material } => {
                Node::new("Sketch", NodeContent::SketchTool { producer, material })
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
/// replayable. `AppCore::apply_intent` dispatches
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
    /// Set the [`SketchSolid`] producer of the sketch node `target` (a no-op for a
    /// non-sketch node). The sketch-authoring analogue of [`SetShape`](Self::SetShape)
    /// — a separate field-edit intent (not a reuse of `SetShape`) because a sketch
    /// node carries a producer, not an [`SdfShape`].
    SetSketch {
        /// The sketch node to edit.
        target: NodeId,
        /// The new sketch + operation producer.
        producer: SketchSolid,
    },
    /// Set the [`MaterialChoice`] of the Tool node `target` (a no-op for a non-Tool
    /// node).
    SetMaterial {
        /// The Tool node to edit.
        target: NodeId,
        /// The new material.
        material: MaterialChoice,
    },
    /// Set the [`CombineOp`] of the LEAF node `target` (ADR 0017: the node's role
    /// in the ordered document-order fold — `Subtract` carves everything
    /// accumulated before it among its siblings). A no-op for a Group / Instance
    /// node (this sibling-level slice ignores their operations; sealed scopes are
    /// issue #74).
    SetOperation {
        /// The leaf node to edit.
        target: NodeId,
        /// The new combine operation.
        operation: CombineOp,
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
/// `PanelResponse`'s effect booleans (ADR 0003 Phase
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
    /// (the typed successor of `PanelResponse::scene_changed` /
    /// `geometry_changed`, which the caller already treats identically as "rebuild").
    pub scene_changed: bool,
    /// A reference Point changed → the caller may persist (Points are pure overlay,
    /// rebuilt every frame, so this does NOT trigger a voxel re-resolve — matching
    /// `PanelResponse::points_changed`).
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
#[cfg(any(test, feature = "test-support"))]
pub fn whole_block_offset(blocks: [i64; 3]) -> [Measurement; 3] {
    use voxel_core::units::ExactRational;
    [
        Measurement::new(ExactRational::from_integer(blocks[0] as i128), 0),
        Measurement::new(ExactRational::from_integer(blocks[1] as i128), 0),
        Measurement::new(ExactRational::from_integer(blocks[2] as i128), 0),
    ]
}
