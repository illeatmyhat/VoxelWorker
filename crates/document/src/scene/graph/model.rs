//! The node-graph data model: identifiers, the [`Node`] and its by-value
//! [`NodeBuilder`] spec, the [`CombineOp`] fold operation, per-node grids,
//! reusable [`AssemblyDef`]s, and reference [`Point`]s.

use serde::{Deserialize, Serialize};

use super::*;

/// A reusable identifier for a [`Tool`-or-`VoxelBody`](NodeContent) definition that an
/// [`NodeContent::Instance`] points at (ADR 0001: reuse by reference). Definitions and
/// instances are fully live: `Scene::add_definition` / `Scene::make_definition_from_active`
/// mint them and `Scene::add_instance` places references to them (the
/// village-of-reused-houses case).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId(pub u32);

/// A **process-stable node identity** (ADR 0003 Phase B). Minted monotonically from
/// a document-owned counter ([`Scene::next_node_id`]) and durable across structural
/// edits + undo, unlike the positional [`NodePath`] (which invalidates on every
/// add/delete/reorder). `NodeId(0)` is the reserved **unassigned** sentinel a
/// freshly-constructed [`Node`] carries until [`Scene::ensure_node_ids`] mints it a
/// real id on the load/normalization path; real ids start at `1`.
///
/// **Phase B1 was scaffolding only; B2–B5 landed on top of it.** The id is now the
/// identity of record: selection (`Scene::active`), the structural edit ops, and
/// the `Intent`/`Command` boundary all key on it, while `NodePath` has been
/// demoted to an ephemeral render/UI projection derived on demand (see its own
/// doc below).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct NodeId(pub u64);

/// The reserved identity of the always-present **root part** (ADR 0018 Decision 2):
/// the concrete container node whose children are the scene's top-level nodes
/// ([`Scene::roots`]). It is minted once, never handed to a user node (the mint
/// floor starts real ids at `2`, see [`Scene::mint_node_id`]), so
/// `active == Some(ROOT_NODE_ID)` unambiguously means "the root part is selected".
/// The root part lives on [`Scene::root`] (a field, NOT in the [`arena`](Scene::arena))
/// and its children spine IS [`Scene::roots`] — its own `Group` payload is unused.
pub const ROOT_NODE_ID: NodeId = NodeId(1);

/// A path to a node anywhere in the **top-level assembly** (ADR 0001 step 4 UI).
///
/// The path is a list of child indices walked from `Scene::nodes` down through
/// [`NodeContent::Group`] children: an empty-ish single element `[i]` selects the
/// top-level node `i`; `[i, j]` selects the `j`-th child of the Group at top-level
/// `i`; and so on to any depth. A path is **always non-empty** for a real
/// selection (the empty path would be "the whole scene", which has no inspector).
///
/// Selection stops at Group boundaries: an [`NodeContent::Instance`] references a
/// definition stored separately in [`Scene::definitions`], so its *children* are
/// not addressable by a `NodePath` (you edit the definition's nodes by selecting a
/// top-level node that lives in that definition is not possible in this UI — a
/// definition is edited via its instances' shared body). The path therefore never
/// descends through an `Instance`.
// ADR 0003 Phase B6: `NodePath` is now a purely EPHEMERAL render/UI tree
// projection — produced on demand by `path_of`/`tree_rows` and consumed within a
// frame by the renderer + gizmo/extent math. It is never stored on any type, held
// across frames, or serialized (identity/selection/storage are all `NodeId` after
// B3–B5), so the `Default`/`Serialize`/`Deserialize` derives were dropped as
// vestigial (no config back-compat to preserve).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodePath {
    /// Child indices from the top-level node list down through Group children.
    pub indices: Vec<usize>,
}

impl NodePath {
    /// A path selecting the top-level node at `index`.
    pub fn root_index(index: usize) -> Self {
        Self { indices: vec![index] }
    }

    /// Build a path from an explicit list of child indices.
    pub fn from_indices(indices: Vec<usize>) -> Self {
        Self { indices }
    }
}

/// How a node combines with the nodes resolved before it (ADR 0001 decision 1;
/// ADR 0017 the ordered fold). Composition is an ordered document-order fold:
/// within a scope, each node folds into the result accumulated by the nodes
/// BEFORE it under its own `CombineOp` — no operand targeting, ever (ADR 0017
/// Decision 2). Geometry is protected by *placement* (after the cutter), never
/// by per-operation target selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CombineOp {
    /// Additive: the output occupied set is the OR of the contributing nodes; on
    /// overlap the later node wins the material.
    #[default]
    Union,
    /// Subtractive: an **occupancy-only mask** (ADR 0017 Decision 1) — the node's
    /// body REMOVES occupancy from everything accumulated before it among its
    /// siblings. It never stamps material; surviving cells keep the material they
    /// already had.
    Subtract,
    /// Intersective: an **occupancy-only mask** (ADR 0017 Decision 1, issue #75) —
    /// the accumulated result KEEPS ONLY the cells the node's body also occupies;
    /// every accumulated cell OUTSIDE the body dies, including cells far outside
    /// the node's own AABB. Like `Subtract` it never stamps material (surviving
    /// cells keep their accumulated material), and intersecting the EMPTY
    /// accumulator (fold start) yields empty — predictable per the ordering law
    /// (Decision 2).
    Intersect,
    /// Raise or recess the accumulated surface WITHIN the node's footprint (ADR 0020
    /// Decision 4) — normalMagic's Boolean Extrude / Emboss, and the only one of its four
    /// named booleans with content a field world does not already have.
    ///
    /// With the accumulator `A`, this node's body `C` and a signed amount `N`:
    ///
    /// ```text
    /// outward (N > 0)   A' = min(A, max(A − N, C))   ≡  A ∪ (dilate(A, N) ∩ C)
    /// inward  (N < 0)   A' = max(A, min(A − N, −C))  ≡  A \ (dilate(¬A, |N|) ∩ C)
    /// ```
    ///
    /// **It cannot be sugar, which is why it is an arm.** `A` appears TWICE in both
    /// formulas. No node may reference the accumulated result — that would BE operand
    /// targeting, which ADR 0017's law admits no exception to — so this cannot decompose
    /// into a sequence of existing fold steps. It stays law-compatible because, exactly like
    /// `Subtract`, it reads "everything accumulated before it in this scope" and nothing
    /// else.
    ///
    /// Like the other masks it never stamps material: an embossed surface keeps the material
    /// it was raised from.
    Emboss {
        /// How far the surface moves: outward when positive, recessed when negative. A
        /// [`Measurement`](voxel_core::units::Measurement) rather than a voxel count, for
        /// the same reason an outset is one — it is authored geometry, not a derived count.
        amount: voxel_core::units::Measurement,
    },
    // future: Override, …
}

impl CombineOp {
    /// Whether this operation needs the accumulated body as a FIELD rather than a voxel set.
    ///
    /// [`Emboss`](Self::Emboss) does, because `A − N` is only meaningful on a field: its
    /// formulas measure the accumulator, they do not merely test membership of it. A scope
    /// containing one is therefore pre-composed into a
    /// [`CompositeProducer`](crate::voxel::CompositeProducer), which is the ONE
    /// representation both folds can agree on — the same argument that made outset a
    /// producer decorator instead of a second pair of fold arms.
    pub fn needs_accumulated_field(self) -> bool {
        matches!(self, CombineOp::Emboss { .. })
    }
}
/// Per-node grid display settings (issue #29 grid rework). Each grid type a
/// node can show is gated by a scene-wide master ANDed with the node's own flag;
/// these are the per-node flags. All default **off** — a freshly-added object
/// carries no grids until the user turns them on (the spec's "default OFF for new
/// objects"). The scene-wide masters live on [`Scene`] (`master_*`).
///
/// **S3/S4 landed on top of the original S1 data model:** `voxel_grid_on_faces`
/// reaches the resolve (it ORs the on-face-grid bit into the leaf's stamped
/// material id, so it travels with each voxel through chunk bucketing — see
/// `Scene::for_each_leaf`), and `block_lattice` drives the per-object lattice
/// extent (`Scene::node_block_lattice_box_recentred`) the renderer draws from.
/// The old scene-wide `AppConfig.show_*` toggles these replaced were deleted in #31.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NodeGrids {
    /// Whether the on-face voxel grid overlay shows on this node (S4).
    #[serde(default)]
    pub voxel_grid_on_faces: bool,
    /// Whether the per-object block lattice shows on this node (S3).
    #[serde(default)]
    pub block_lattice: bool,
    /// Whether the per-object floor grid shows on this node (S3).
    #[serde(default)]
    pub floor_grid: bool,
}

/// One placed node in the assembly graph: a producer (or sub-assembly) plus its
/// local placement and combine operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Process-stable identity (ADR 0003 Phase B), minted by
    /// [`Scene::ensure_node_ids`]. `NodeId(0)` (the default) until minted. This IS
    /// the identity of record (Phase B2–B5): selection, the edit ops and the
    /// `Intent`/`Command` boundary all key on it.
    #[serde(default)]
    pub id: NodeId,
    /// Human-readable name (for the future node-list UI).
    #[serde(default)]
    pub name: String,
    /// LOCAL transform; composes with ancestors' (`world = parent ∘ local`).
    /// Step 1 only ever uses the identity (zero offset).
    #[serde(default)]
    pub transform: NodeTransform,
    /// How this node folds into the result accumulated before it in its scope (ADR
    /// 0017). For a leaf, the leaf's own role; for a `Group` (and, resolver-side, an
    /// `Instance`), the operation the scope's PRE-COMPOSED body folds under
    /// (Decision 3 — sealed composition scopes, issue #74).
    #[serde(default)]
    pub operation: CombineOp,
    /// How far this node's body is DILATED before it folds (ADR 0019 Decision 7).
    ///
    /// It sits beside [`operation`](Self::operation) deliberately: a leaf, a `Group` and an
    /// `Instance` may all carry one, so a composed cutter dilates **as a whole** rather than
    /// member-by-member. The fold already yields a field, so `d − N` is meaningful at every
    /// level and this needs no new machinery. A NEGATIVE outset insets, shrinking the body —
    /// which is how a deliberate gap between chiselled pieces is authored.
    ///
    /// It is a [`Measurement`](voxel_core::units::Measurement), never an integer voxel
    /// count, so `"1/4 block"` survives a
    /// density change as the authored intent rather than as a stale derived number (ADR 0008
    /// — the frame is carried, never re-derived).
    ///
    /// **The shape of the dilation follows the body's metric, not the outset's.** A group's
    /// metric is the WEAKEST of its members', so a group mixing a box and a sphere outsets
    /// round. That is predictable, but it is a rule the UI must not hide.
    ///
    /// Zero for a node whose producer has no field at all: ADR 0020 Decision 1 bars outset
    /// there rather than fabricating a distance for it.
    ///
    /// # On a scope, it dilates the COMPOSED body
    ///
    /// A Part (`Group`) or a sealed `Instance` body pre-composes its children into one body
    /// (ADR 0017 Decision 3), and its outset dilates THAT — not each member separately. The
    /// walk hands such a scope to the fold as a single
    /// [`CompositeProducer`](crate::voxel::CompositeProducer) leaf.
    ///
    /// The distinction is not cosmetic. Dilation distributes over union, so a pure-union
    /// Part gives the same answer either way — but a Part with an internal `Subtract`
    /// diverges sharply: dilating members individually makes the inner cutter carve MORE,
    /// while dilating the composed Part grows the finished body and partly closes that cut.
    /// Only the latter is what "give this Part clearance" means.
    ///
    /// A scope whose subtree contains a `VoxelBody` declines to compose and keeps its
    /// members' behaviour unchanged — such a body is fieldless, so it could not be outset
    /// anyway (ADR 0020 Decision 1).
    #[serde(default)]
    pub outset: voxel_core::units::Measurement,
    /// Whether the node participates in the composed geometry. This is NOT a display
    /// flag: a disabled node is pruned from the op-stack walk *before* evaluation, so it
    /// stamps nothing and its operation never runs. Disabling a `Subtract` therefore
    /// fills the hole back in rather than merely hiding a cutter — the body you get is
    /// the body the fold would have produced had the node never been authored.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    /// Per-node grid display settings (issue #29). Defaults all-off; an older
    /// config without this field deserialises to the all-off default.
    #[serde(default)]
    pub grids: NodeGrids,
    /// What the node is.
    pub content: NodeContent,
}

/// A node missing its `enabled` flag in an older/partial config defaults to enabled.
/// Authoring a node is itself the statement that it belongs in the composition, so
/// participation is the common case and withdrawing a node is the exception a config
/// has to say out loud.
fn default_enabled() -> bool {
    true
}

impl Node {
    /// An enabled, identity-placed, union node wrapping `content`. A new node
    /// carries NO grids (issue #29: grids default OFF for new objects).
    pub fn new(name: impl Into<String>, content: NodeContent) -> Self {
        Self {
            // Unassigned until `Scene::ensure_node_ids` mints a stable id on the
            // load/normalization path (ADR 0003 Phase B).
            id: NodeId(0),
            name: name.into(),
            transform: NodeTransform::identity(),
            operation: CombineOp::Union,
            outset: voxel_core::units::Measurement::default(),
            enabled: true,
            grids: NodeGrids::default(),
            content,
        }
    }
}

/// A **by-value node-tree spec** for terse construction (ADR 0003 Phase B5).
///
/// Now that [`NodeContent::Group`] stores a `Vec<NodeId>` (ids into the scene
/// [`arena`](Scene::arena)) rather than owning its children, a caller can no longer
/// write `Group(vec![child_node])` to build a subtree by value. `NodeBuilder` restores
/// that ergonomic: a leaf carries its [`Node`] directly; a [`NodeBuilder::group`]
/// carries the (still-by-value) `Node`s/sub-builders of its children, which
/// [`Scene::from_nodes`] / [`Scene::add_definition`] flatten into the arena (minting
/// ids depth-first, building each Group's id-spine) at construction time. A plain
/// [`Node`] converts in via [`From`], so flat fixtures stay `vec![node_a, node_b]`.
pub enum NodeBuilder {
    /// A leaf (or pre-built) node inserted as-is. Its content may NOT be a Group with
    /// by-value children (the spine is ids) — use [`NodeBuilder::group`] for that.
    Leaf(Node),
    /// A Group node (`name` + `transform`) wrapping child specs, inserted as a fresh
    /// arena node whose spine is the children's minted ids.
    Group {
        /// The Group node's name.
        name: String,
        /// The Group node's local transform (offset etc.).
        transform: NodeTransform,
        /// Whether the Group participates in the composed geometry (see [`Node::enabled`]).
        enabled: bool,
        /// The Group's children, in document order.
        children: Vec<NodeBuilder>,
    },
}

impl NodeBuilder {
    /// A Group spec with an identity transform wrapping `children`.
    pub fn group(name: impl Into<String>, children: Vec<NodeBuilder>) -> Self {
        NodeBuilder::Group {
            name: name.into(),
            transform: NodeTransform::identity(),
            enabled: true,
            children,
        }
    }

    /// A Group spec at a whole-block `offset_blocks` (at density
    /// `voxels_per_block`) wrapping `children`. The block-valued param is the UI
    /// placement convenience; it is converted to the canonical voxel offset via
    /// [`NodeTransform::from_blocks`] (ADR 0003 §3f(0)).
    pub fn group_at(
        name: impl Into<String>,
        offset_blocks: [i64; 3],
        voxels_per_block: u32,
        children: Vec<NodeBuilder>,
    ) -> Self {
        NodeBuilder::Group {
            name: name.into(),
            transform: NodeTransform::from_blocks(offset_blocks, voxels_per_block),
            enabled: true,
            children,
        }
    }
}

impl From<Node> for NodeBuilder {
    fn from(node: Node) -> Self {
        NodeBuilder::Leaf(node)
    }
}

/// A reusable sub-assembly (e.g. "house") placed by [`NodeContent::Instance`]
/// (ADR 0001). Definitions are fully live: `Scene::add_definition` /
/// `Scene::make_definition_from_active` mint one and `Scene::add_instance` places a
/// reference to it, so the same definition placed by N instances is visited N times
/// at resolve (the village-of-reused-houses case).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssemblyDef {
    /// The definition's identifier (referenced by an `Instance`).
    pub id: DefId,
    /// Human-readable name.
    #[serde(default)]
    pub name: String,
    /// The nodes that make up this assembly. **ADR 0003 Phase B5:** an ordered spine
    /// of child [`NodeId`]s; the child `Node`s live in the scene-wide
    /// [`Scene::arena`]. The `Vec` order is document order.
    #[serde(default)]
    pub children: Vec<NodeId>,
    /// Whether this definition is a **fixture** (ADR 0017 Decision 4, issue #77): a
    /// fixture does NOT pre-compose — its children splice into the HOSTING scope's
    /// ordered fold at the instance's spine position, in order, under the instance's
    /// transform (how a window both cuts its opening and fills its frame with one
    /// placement). The flag lives HERE because being a fixture is what the part *is*;
    /// instances stay pure reference+transform, and a fixture instance's own
    /// [`CombineOp`] is inert (the resolver never consults it — see
    /// [`Scene::walk_nodes`]). `serde(default)`: pre-fixture documents load sealed.
    #[serde(default)]
    pub fixture: bool,
}

/// A world-anchored **reference element** (issue #29 grid rework): a named point
/// in the world-block lattice that carries optional reference planes (ground /
/// front / side) and axis lines. Distinct from the per-selection transform gizmo
/// (S2) — a Point is a persistent annotation in world space.
///
/// Every scene has exactly one **Origin** Point (`is_origin = true`) synthesized
/// on load ([`Scene::ensure_origin_point`]); it is undeletable but hideable. Users
/// may add further Points.
///
/// Points are rendered as a camera-relative overlay (S5), rebuilt every frame from
/// this list — a hidden/shown Point or a plane/axis toggle takes effect immediately,
/// with no re-resolve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
    /// Human-readable name (e.g. "Origin").
    #[serde(default)]
    pub name: String,
    /// Position in the world-block lattice — the whole-block view of placement
    /// ([`NodeTransform::blocks`], `i64` for far-world addressing).
    #[serde(default)]
    pub position_blocks: [i64; 3],
    /// Sub-block offset in voxels (v1 keeps `[0, 0, 0]`; the field exists so a
    /// future sub-block placement is a data change, not a rewrite).
    #[serde(default)]
    pub offset_voxels: [i32; 3],
    /// Whether the FRONT reference plane (XZ, normal +Y) shows. Default false.
    /// (Z-up: the front view looks along +Y; the front plane spans X and Z.)
    #[serde(default)]
    pub plane_xz: bool,
    /// Whether the GROUND reference plane (XY, normal +Z) shows. Default **true**.
    /// (Z-up: the ground plane is XY — the default reference plane.)
    #[serde(default = "default_true_bool")]
    pub plane_xy: bool,
    /// Whether the SIDE reference plane (YZ, normal +X) shows. Default false.
    #[serde(default)]
    pub plane_yz: bool,
    /// Whether the +X axis line shows. Default **true** (issue #29 fix: the single
    /// `axes` toggle is split into per-axis X/Y/Z so each is independently
    /// toggleable). An older config without this field defaults it true.
    #[serde(default = "default_true_bool")]
    pub axis_x: bool,
    /// Whether the +Y axis line shows. Default **true**.
    #[serde(default = "default_true_bool")]
    pub axis_y: bool,
    /// Whether the +Z axis line shows. Default **true**.
    #[serde(default = "default_true_bool")]
    pub axis_z: bool,
    /// Whether the Point is hidden (renders nothing). Default false. Works for the
    /// Origin too (the Origin is hideable, just not deletable).
    #[serde(default)]
    pub hidden: bool,
    /// Whether this is the (unique, undeletable) Origin Point. Default false.
    #[serde(default)]
    pub is_origin: bool,
}

/// Default `true` for serde defaults on `Point`'s ground/axes flags.
fn default_true_bool() -> bool {
    true
}

impl Default for Point {
    /// A blank Point at the world origin with the spec defaults (ground + axes on,
    /// other planes off, visible, NOT the Origin). Z-up: the ground plane is XY
    /// (`plane_xy`). [`Scene::ensure_origin_point`] clones this and sets
    /// `is_origin`/`name`.
    fn default() -> Self {
        Self {
            name: String::new(),
            position_blocks: [0, 0, 0],
            offset_voxels: [0, 0, 0],
            plane_xz: false,
            plane_xy: true,
            plane_yz: false,
            axis_x: true,
            axis_y: true,
            axis_z: true,
            hidden: false,
            is_origin: false,
        }
    }
}
