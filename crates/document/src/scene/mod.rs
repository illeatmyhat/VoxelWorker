//! The scene (assembly) model: an assembly graph of **nodes**, each wrapping a producer
//! plus a placement. ALL voxel resolution routes through it.
//!
//! **Every `NodeContent` leaf resolves, including recursion and reuse:**
//!
//!   * [`NodeContent::Tool`] — a *parametric* producer (`SdfShape`) that carries
//!     the Tool's single `MaterialChoice`.
//!   * `NodeContent::SketchTool` — the sketch→extrude/revolve producer.
//!   * [`NodeContent::VoxelBody`] — a *static* voxel body ([`VoxelBody::DebugClouds`]).
//!
//! [`NodeContent::Group`] and [`NodeContent::Instance`] carry recursion and reuse by
//! reference: a Group folds its children under its own `CombineOp`, and an Instance
//! resolves the referenced definition under its transform, so the same definition placed
//! by N instances is visited N times (the village-of-reused-houses case) — see
//! `Scene::walk_nodes` / `Scene::for_each_leaf`.
//!
//! ## Compositing
//!
//! Producers emit content centered at the origin; the Scene **composites** — walk the node
//! tree, resolve each enabled leaf into its own local grid, and **stamp** it (under the
//! node's transform) into the output grid. For a one-node scene whose region is the node's
//! full extent with a zero offset the stamp is the identity, so the resulting `VoxelGrid`
//! is bit-for-bit what the bare producer resolves (same dimensions, same occupied set).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

mod extent;
mod graph;
mod operand_body;
mod producers;
mod sketch_handles;
mod spatial;
#[cfg(test)]
mod tests;

pub use extent::{
    block_aabb_exceeds_coordinate_limit, NodeTransform, RegionBlocks, COORDINATE_LIMIT_BLOCKS,
};
pub use graph::{
    AssemblyDef, CombineOp, DefId, LeafOrigin, Node, NodeBuilder, NodeGrids, NodeId, NodePath,
    Point, PointId, ROOT_NODE_ID,
};
pub use producers::{operation_masks_beyond_bounds, quat_from_lattice, LeafProducer, ScopeFrame};
pub use producers::{NodeContent, VoxelBody};
pub use sketch_handles::SketchHandles;

/// Default +X spacing (in blocks) between successive instances of the same
/// definition added via [`Scene::add_instance`], so a freshly-placed village
/// house lands clear of the previous one instead of exactly on top of it.
const DEFAULT_INSTANCE_SPACING_BLOCKS: i32 = 6;

/// Default `true` for the scene-wide grid masters: all three default ON so enabling a
/// per-object toggle shows immediately, while the per-object flags stay default OFF — the
/// default view is still clean.
fn default_master_grid() -> bool {
    true
}

/// The scene (assembly): a list of placed nodes resolved into the shared `VoxelGrid`
/// truth, plus the reusable `definitions` a [`NodeContent::Instance`] resolves under its
/// own transform (reuse by reference: a village of identical houses is one definition
/// placed by N instances). Selection is not part of the document — it is workspace state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// The top-level assembly's nodes, as an **ordered spine of [`NodeId`]s**. Resolved in
    /// this order (later nodes win on overlap under [`CombineOp::Union`]); the `Node`s
    /// themselves live in [`arena`](Self::arena).
    /// **Golden-critical:** every tree walk iterates THIS spine (and each
    /// [`NodeContent::Group`]'s spine) for order, fetching content from the arena —
    /// never iterate the arena to produce a walk (that visits in id order and would
    /// reorder later-wins material on overlap).
    #[serde(default)]
    pub roots: Vec<NodeId>,
    /// The always-present **root part**: the concrete, selectable container node the scene
    /// tree presents as its top row ("Part"). Its children are the top-level nodes — the
    /// ordered spine [`roots`](Self::roots), which stays the source of truth for the fold
    /// entry ([`for_each_leaf`] walks it directly, so root reification changes NO
    /// composition semantics). The node lives HERE (not in the [`arena`](Self::arena)) so
    /// its reserved id [`ROOT_NODE_ID`] never mingles with user ids and every arena scan is
    /// unchanged; its own `Group` payload is left empty (the real children are `roots`).
    /// Undeletable, and never a `MakeDefinition`/`GroupNode` target (a definition of the
    /// whole scene is out of scope — see
    /// [`make_definition_from_node`](Self::make_definition_from_node)).
    ///
    /// [`for_each_leaf`]: Self::for_each_leaf
    /// [`make_definition_from_node`]: Self::make_definition_from_node
    #[serde(default = "default_root_part")]
    pub root: Node,
    /// The id-keyed node storage. A [`BTreeMap`] (not `HashMap`) so it iterates/serializes
    /// in ascending-id order → deterministic, and so the load-path `max_existing` scan in
    /// [`ensure_node_ids`](Self::ensure_node_ids) is stable. Keyed by the monotonic
    /// [`NodeId`] (the counter already prevents stale-id aliasing, so no slotmap
    /// generations are needed). **Get-only inside walks** — see [`roots`](Self::roots).
    #[serde(default)]
    pub arena: BTreeMap<NodeId, Node>,
    /// Reusable sub-assemblies referenced by [`NodeContent::Instance`]. A definition is
    /// stored ONCE here regardless of how many instances place it. Looked up by [`DefId`]
    /// via [`def_by_id`].
    ///
    /// [`def_by_id`]: Self::def_by_id
    #[serde(default)]
    pub definitions: Vec<AssemblyDef>,
    /// World-anchored reference Points. Always contains exactly one Origin Point after
    /// [`ensure_origin_point`](Self::ensure_origin_point) runs on load. A payload without
    /// this field deserializes to an empty list, then gains its Origin on the load path.
    #[serde(default)]
    pub points: Vec<Point>,
    /// Scene-wide master toggle for the block lattice. Default **true**. ANDed with each
    /// node's [`NodeGrids::block_lattice`]. The single source of truth for this master,
    /// persisted directly via the `scene` field.
    #[serde(default = "default_master_grid")]
    pub master_block_lattice: bool,
    /// Scene-wide master toggle for the on-face voxel grid. Default **true**, so a
    /// per-object toggle shows immediately. The single source of truth for this master.
    #[serde(default = "default_master_grid")]
    pub master_voxel_grid: bool,
    /// Scene-wide master toggle for the floor grid. Default **true**, so a per-object
    /// toggle shows immediately. The single source of truth for this master.
    #[serde(default = "default_master_grid")]
    pub master_floor_grid: bool,
    /// Document-owned monotonic counter for minting [`NodeId`]s **and [`PointId`]s** — one
    /// counter so the undo machinery's `counter_before` rewind covers both kinds. `0` is
    /// never minted (it is the unassigned sentinel); the first real id is `1`.
    /// [`ensure_node_ids`](Self::ensure_node_ids) advances it past any ids already
    /// present in a loaded scene before minting new ones.
    #[serde(default)]
    pub next_node_id: u64,
    /// Document-level voxel density (voxels per block): which block grid the plan targets.
    /// Uniform across the document — it is NOT a per-shape attribute. Every resolve /
    /// chunk / export / spatial-index call sources its density param from here;
    /// [`Intent::SetDensity`](crate::intent::Intent::SetDensity) is the single writer.
    #[serde(default = "default_density")]
    pub voxels_per_block: u32,
}

/// The document-level density default (voxels per block) for a fresh or partially
/// deserialized [`Scene`] — matches [`GeometryParams`](crate::voxel::GeometryParams)
/// default 16.
fn default_density() -> u32 {
    16
}

/// The default **root part** node: a `Union`, identity-placed [`NodeContent::Group`]
/// named "Part" carrying the reserved [`ROOT_NODE_ID`]. Its own children `Vec` is left
/// empty — the scene's top-level nodes live on [`Scene::roots`], which is the container's
/// real (and fold-authoritative) spine. Used both by [`Scene::default`] and as the
/// `serde(default)` for the `root` field, so a payload missing it loads with its `roots`
/// adopted as this fresh part's children.
fn default_root_part() -> Node {
    let mut node = Node::new("Part", NodeContent::Group(Vec::new()));
    node.id = ROOT_NODE_ID;
    node
}

impl Default for Scene {
    /// An empty scene with **all three grid masters ON**, while every node's per-object
    /// grid flag stays default OFF, so enabling a per-object toggle shows immediately yet
    /// the default view is clean. No Points (the Origin is synthesized on the load path
    /// via [`ensure_origin_point`](Self::ensure_origin_point)).
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            root: default_root_part(),
            arena: BTreeMap::new(),
            definitions: Vec::new(),
            points: Vec::new(),
            master_block_lattice: true,
            master_voxel_grid: true,
            master_floor_grid: true,
            // Real node ids start at 2; `1` is reserved for the root part
            // ([`ROOT_NODE_ID`]), so a minted user id never collides with it.
            next_node_id: 2,
            voxels_per_block: default_density(),
        }
    }
}
