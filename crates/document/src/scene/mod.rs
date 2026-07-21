//! The scene (assembly) model — ADR 0001.
//!
//! ADR 0001 replaced the old single-producer setup — a lone
//! [`GeometryParams`](crate::voxel::GeometryParams) SDF shape plus a
//! `debug_clouds: bool` selector — with a **Scene**: an assembly graph of **nodes**,
//! each wrapping a producer plus a placement. This module implements that model and
//! routes ALL voxel resolution through it.
//!
//! **Every `NodeContent` leaf resolves, including recursion and reuse:**
//!
//!   * [`NodeContent::Tool`] — a *parametric* producer (`SdfShape`) that carries
//!     the Tool's single `MaterialChoice`.
//!   * `NodeContent::SketchTool` — the sketch→extrude/revolve producer (ADR 0003 §3i).
//!   * [`NodeContent::VoxelBody`] — a *static* voxel body; today the only variant is
//!     [`VoxelBody::DebugClouds`].
//!
//! [`NodeContent::Group`] and [`NodeContent::Instance`] (recursion + reuse by
//! reference, ADR 0001's original "step 4" goal) are fully wired: a Group folds its
//! children under its own `CombineOp` (ADR 0017), and an Instance resolves the
//! referenced definition under its transform, so the same definition placed by N
//! instances is visited N times (the village-of-reused-houses case) — see
//! `Scene::walk_nodes` / `Scene::for_each_leaf`.
//!
//! ## Identical-behaviour guarantee
//!
//! The producer trait (`VoxelProducer`) does **not** change: producers still
//! emit content centred at the origin. The Scene's new job is **compositing** —
//! walk the node tree, resolve each enabled leaf into its own local grid, and
//! **stamp** it (under the node's transform) into the output grid. For a one-node
//! scene whose region is the node's full extent with a zero offset, the stamp is
//! the identity, so the resulting `VoxelGrid` is bit-for-bit what
//! `SdfShape::resolve` / `DebugCloudField::resolve` produce today (same
//! dimensions, same occupied set). See `tool_scene_matches_bare_producer` below.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

mod extent;
mod graph;
mod operand_body;
mod producers;
mod spatial;
#[cfg(test)]
mod tests;

pub use extent::{NodeTransform, RegionBlocks};
pub use graph::{
    AssemblyDef, CombineOp, DefId, Node, NodeBuilder, NodeGrids, NodeId, NodePath, Point,
    ROOT_NODE_ID,
};
pub use producers::{NodeContent, VoxelBody};
pub use producers::{operation_masks_beyond_bounds, quat_from_lattice, LeafProducer, ScopeFrame};

/// Default +X spacing (in blocks) between successive instances of the same
/// definition added via [`Scene::add_instance`], so a freshly-placed village
/// house lands clear of the previous one instead of exactly on top of it.
const DEFAULT_INSTANCE_SPACING_BLOCKS: i32 = 6;

/// Default `true` for the scene-wide grid masters (issue #29 grid-rework fix: all
/// three masters default ON so enabling a per-object toggle shows immediately,
/// while the per-object flags stay default OFF — the default view is still clean).
fn default_master_grid() -> bool {
    true
}

/// The scene (assembly): a list of placed nodes resolved into the shared
/// `VoxelGrid` truth. ADR 0001's full model carries reusable `definitions` too;
/// step 2 added the flat node list plus the `active` selection that drives the
/// inspector. `definitions` is wired up so a [`NodeContent::Instance`]
/// resolves the referenced [`AssemblyDef`] under its transform (reuse by
/// reference: a village of identical houses is one definition placed by N
/// instances).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// The top-level assembly's nodes, as an **ordered spine of [`NodeId`]s**
    /// (ADR 0003 Phase B5). Resolved in this order (later nodes win on overlap under
    /// [`CombineOp::Union`]); the `Node`s themselves live in [`arena`](Self::arena).
    /// **Golden-critical:** every tree walk iterates THIS spine (and each
    /// [`NodeContent::Group`]'s spine) for order, fetching content from the arena —
    /// never iterate the arena to produce a walk (that visits in id order and would
    /// reorder later-wins material on overlap).
    #[serde(default)]
    pub roots: Vec<NodeId>,
    /// The always-present **root part** (ADR 0018 Decision 2): the concrete,
    /// selectable container node the scene tree presents as its top row ("Part"). Its
    /// children are the top-level nodes — the ordered spine [`roots`](Self::roots),
    /// which stays the source of truth for the fold entry ([`for_each_leaf`] walks it
    /// directly, so root reification changes NO composition semantics). The node lives
    /// HERE (not in the [`arena`](Self::arena)) so its reserved id [`ROOT_NODE_ID`]
    /// never mingles with user ids and every arena scan is unchanged; its own `Group`
    /// payload is left empty (the real children are `roots`). Undeletable, and never a
    /// `MakeDefinition`/`GroupNode` target (a definition of the whole scene is out of
    /// scope — see [`make_definition_from_active`](Self::make_definition_from_active)).
    ///
    /// [`for_each_leaf`]: Self::for_each_leaf
    /// [`make_definition_from_active`]: Self::make_definition_from_active
    #[serde(default = "default_root_part")]
    pub root: Node,
    /// The id-keyed node storage (ADR 0003 Phase B5). A [`BTreeMap`] (not `HashMap`)
    /// so it iterates/serializes in ascending-id order → deterministic, and so the
    /// load-path `max_existing` scan in [`ensure_node_ids`](Self::ensure_node_ids) is
    /// stable. Keyed by the monotonic [`NodeId`] (the counter already prevents
    /// stale-id aliasing, so no slotmap generations are needed). **Get-only inside
    /// walks** — see [`roots`](Self::roots).
    #[serde(default)]
    pub arena: BTreeMap<NodeId, Node>,
    /// Reusable sub-assemblies referenced by [`NodeContent::Instance`]. A
    /// definition is stored ONCE here regardless of how many instances place it
    /// (ADR 0001 "Nesting & reuse"). Looked up by [`DefId`] via [`def_by_id`].
    ///
    /// [`def_by_id`]: Self::def_by_id
    #[serde(default)]
    pub definitions: Vec<AssemblyDef>,
    /// The [`NodeId`] of the active/selected node — the one the inspector edits
    /// (ADR 0001 step 4: selection reaches any depth, so a
    /// [`Group`](NodeContent::Group) child is selectable, not just a top-level
    /// node). `None` when nothing is selected.
    ///
    /// **ADR 0003 Phase B3:** selection is keyed by the process-stable [`NodeId`],
    /// not the positional [`NodePath`] it was before. The active node is resolved
    /// on demand via [`node_by_id`](Self::node_by_id) / [`path_of`](Self::path_of),
    /// so a structural edit (add / delete / group / reorder) that shuffles indices
    /// no longer invalidates the selection: it still points at the SAME node by
    /// identity. The edit ops re-point `active` to the [`NodeId`] of their target.
    /// No old-save migration (the user does not keep pre-alpha saves) — a loaded
    /// scene's `active` is read back as a raw id, and any stale id simply resolves
    /// to `None`.
    #[serde(default)]
    pub active: Option<NodeId>,
    /// World-anchored reference Points (issue #29). Always contains exactly one
    /// Origin Point after [`ensure_origin_point`](Self::ensure_origin_point) runs
    /// on load. An older config without this field deserialises to an empty list,
    /// then gains its Origin on the load path.
    #[serde(default)]
    pub points: Vec<Point>,
    /// Scene-wide master toggle for the block lattice (issue #29). Default
    /// **true**. ANDed with each node's [`NodeGrids::block_lattice`] in S3.
    /// The single source of truth for this master (persisted directly via the
    /// `scene` field; the legacy `AppConfig.show_block_lattice` mirror was deleted
    /// in #31).
    #[serde(default = "default_master_grid")]
    pub master_block_lattice: bool,
    /// Scene-wide master toggle for the on-face voxel grid (issue #29). Default
    /// **true** (grid-rework fix: all masters on so a per-object toggle shows
    /// immediately). The single source of truth for this master (the legacy
    /// `AppConfig.show_grid_overlay` mirror was deleted in #31).
    #[serde(default = "default_master_grid")]
    pub master_voxel_grid: bool,
    /// Scene-wide master toggle for the floor grid (issue #29). Default **true**
    /// (grid-rework fix: all masters on so a per-object toggle shows immediately).
    /// The single source of truth for this master (the legacy
    /// `AppConfig.show_floor_grid` mirror was deleted in #31).
    #[serde(default = "default_master_grid")]
    pub master_floor_grid: bool,
    /// The active/selected Point (index into [`points`](Self::points)), or `None`.
    #[serde(default)]
    pub active_point: Option<usize>,
    /// Document-owned monotonic counter for minting [`NodeId`]s (ADR 0003 Phase B).
    /// `0` is never minted (it is the unassigned sentinel); the first real id is `1`.
    /// [`ensure_node_ids`](Self::ensure_node_ids) advances it past any ids already
    /// present in a loaded scene before minting new ones.
    #[serde(default)]
    pub next_node_id: u64,
    /// Document-level voxel density (voxels per block): which block-game grid the
    /// plan targets (ADR 0003 §3f(0)). Uniform across the document — it is NOT a
    /// per-shape attribute. Every resolve / chunk / export / spatial-index call
    /// sources its density param from here; [`Intent::SetDensity`](crate::intent::Intent::SetDensity)
    /// is the single writer.
    #[serde(default = "default_density")]
    pub voxels_per_block: u32,
}

/// The document-level density default (voxels per block) for a fresh or partially
/// deserialised [`Scene`] — matches [`GeometryParams`](crate::voxel::GeometryParams)
/// default 16.
fn default_density() -> u32 {
    16
}

/// The default **root part** node (ADR 0018 Decision 2): a `Union`, identity-placed
/// [`NodeContent::Group`] named "Part" carrying the reserved [`ROOT_NODE_ID`]. Its
/// own children `Vec` is left empty — the scene's top-level nodes live on
/// [`Scene::roots`], which is the container's real (and fold-authoritative) spine.
/// Used both by [`Scene::default`] and as the `serde(default)` for the `root` field,
/// so a document missing it (an older save) loads with its `roots` adopted as this
/// fresh part's children.
fn default_root_part() -> Node {
    let mut node = Node::new("Part", NodeContent::Group(Vec::new()));
    node.id = ROOT_NODE_ID;
    node
}

impl Default for Scene {
    /// An empty scene with the issue-#29 master defaults — **all three masters ON**
    /// (grid-rework fix), while every node's per-object grid flag stays default OFF,
    /// so enabling a per-object toggle shows immediately yet the default view is
    /// clean. No Points yet (the Origin is synthesized on the load path via
    /// [`ensure_origin_point`](Self::ensure_origin_point)).
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            root: default_root_part(),
            arena: BTreeMap::new(),
            definitions: Vec::new(),
            active: None,
            points: Vec::new(),
            master_block_lattice: true,
            master_voxel_grid: true,
            master_floor_grid: true,
            active_point: None,
            // Real node ids start at 2; `1` is reserved for the root part
            // ([`ROOT_NODE_ID`]), so a minted user id never collides with it.
            next_node_id: 2,
            voxels_per_block: default_density(),
        }
    }
}
