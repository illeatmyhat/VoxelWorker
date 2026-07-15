//! Node-graph model and selection (ADR 0001 assembly graph, ADR 0003 Phase B
//! stable ids): the id-keyed arena and root spine, node paths & ids, structural
//! edits (add / remove / group / ungroup / definition / instance), reference
//! Points, and the active selection.


use serde::{Deserialize, Serialize};

use voxel_core::core_geom::MaterialChoice;
use crate::voxel::{GeometryParams, SdfShape};

use super::*;

/// A reusable identifier for a [`Tool`-or-`Part`](NodeContent) definition that an
/// [`NodeContent::Instance`] points at (ADR 0001: reuse by reference). Step 1
/// never constructs an Instance, so this is a forward-declared type only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId(pub u32);

/// A **process-stable node identity** (ADR 0003 Phase B). Minted monotonically from
/// a document-owned counter ([`Scene::next_node_id`]) and durable across structural
/// edits + undo, unlike the positional [`NodePath`] (which invalidates on every
/// add/delete/reorder). `NodeId(0)` is the reserved **unassigned** sentinel a
/// freshly-constructed [`Node`] carries until [`Scene::ensure_node_ids`] mints it a
/// real id on the load/normalization path; real ids start at `1`.
///
/// **Phase B1 is scaffolding only:** the id is minted + persisted but NOT yet the
/// identity of record — `NodePath` still is — so nothing reads it yet (B2/B3 move
/// selection + commands onto it).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub struct NodeId(pub u64);

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

/// How a node combines with the nodes resolved before it. v1 only ever
/// constructs [`CombineOp::Union`]; the enum exists so subtract / intersect /
/// override become a data change on the node rather than a re-architecture
/// (ADR 0001 decision 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CombineOp {
    /// Additive: the output occupied set is the OR of the contributing nodes; on
    /// overlap the later node wins the material.
    #[default]
    Union,
    // future: Subtract, Intersect, Override, …
}
/// Per-node grid display settings (issue #29 grid rework, S1). Each grid type a
/// node can show is gated by a scene-wide master ANDed with the node's own flag;
/// these are the per-node flags. All default **off** — a freshly-added object
/// carries no grids until the user turns them on (the spec's "default OFF for new
/// objects"). The scene-wide masters live on [`Scene`] (`master_*`).
///
/// **S1 is data-model only:** these fields are persisted and tested but NOT yet
/// read by any renderer (that wiring is S3/S4). The existing
/// `PanelState.show_*` toggles keep driving the current renderers unchanged.
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
    /// [`Scene::ensure_node_ids`]. `NodeId(0)` (the default) until minted. NOT yet
    /// the identity of record — `NodePath` still is — so nothing reads this in B1.
    #[serde(default)]
    pub id: NodeId,
    /// Human-readable name (for the future node-list UI).
    #[serde(default)]
    pub name: String,
    /// LOCAL transform; composes with ancestors' (`world = parent ∘ local`).
    /// Step 1 only ever uses the identity (zero offset).
    #[serde(default)]
    pub transform: NodeTransform,
    /// How this node combines with earlier ones. v1: always [`CombineOp::Union`].
    #[serde(default)]
    pub operation: CombineOp,
    /// Whether the node contributes to resolution (a hidden node stamps nothing).
    #[serde(default = "default_visible")]
    pub visible: bool,
    /// Per-node grid display settings (issue #29). Defaults all-off; an older
    /// config without this field deserialises to the all-off default.
    #[serde(default)]
    pub grids: NodeGrids,
    /// What the node is.
    pub content: NodeContent,
}

/// A node missing its `visible` flag in an older/partial config defaults to
/// visible (the common case — a hidden node is the exception, explicitly set).
fn default_visible() -> bool {
    true
}

impl Node {
    /// A visible, identity-placed, union node wrapping `content`. A new node
    /// carries NO grids (issue #29: grids default OFF for new objects).
    pub fn new(name: impl Into<String>, content: NodeContent) -> Self {
        Self {
            // Unassigned until `Scene::ensure_node_ids` mints a stable id on the
            // load/normalization path (ADR 0003 Phase B).
            id: NodeId(0),
            name: name.into(),
            transform: NodeTransform::identity(),
            operation: CombineOp::Union,
            visible: true,
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
        /// Whether the Group is visible.
        visible: bool,
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
            visible: true,
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
            visible: true,
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
/// (ADR 0001). Step 1 never constructs or resolves one; it exists so the model is
/// complete. The top-level assembly is also an `AssemblyDef` (its `root`).
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
/// **S1 is data-model only:** Points are persisted and tested but NOT yet rendered
/// (that is S5).
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

impl Scene {
    /// A scene with a single node — the shape every one-node call site builds. The
    /// lone node is the active selection.
    ///
    /// ADR 0003 Phase B3: selection is keyed by [`NodeId`], so the lone node is
    /// minted a stable id here ([`ensure_node_ids`](Self::ensure_node_ids)) and
    /// `active` is set to that id — the scene is born already-normalised, so the
    /// selection resolves immediately without a separate load-path mint.
    pub fn single_node(node: Node) -> Self {
        let mut scene = Self::from_nodes(vec![node]);
        scene.active = scene.roots.first().copied();
        scene
    }

    /// Build a scene from a list of top-level [`Node`]s (ADR 0003 Phase B5), inserting
    /// each (and its `Group` descendants) into the [`arena`](Self::arena) under a
    /// freshly-minted [`NodeId`] and recording the top-level ids as the
    /// [`roots`](Self::roots) spine in order. The terse constructor the demo builders
    /// and test fixtures use so they keep building `Node` trees by value while the
    /// storage underneath is the id-keyed arena. `active` is left `None` (callers set
    /// it). Equivalent in effect to the old `Scene { nodes, .. }` + `ensure_node_ids`.
    pub fn from_nodes<I, B>(nodes: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<NodeBuilder>,
    {
        let mut scene = Self::default();
        for spec in nodes {
            let id = scene.insert_builder(spec.into());
            scene.roots.push(id);
        }
        scene
    }

    /// Insert a [`Node`] (and, for a [`NodeContent::Group`], its child subtrees) into
    /// the [`arena`](Self::arena) under a freshly-minted [`NodeId`], returning the id
    /// the node itself took. Does NOT touch [`roots`](Self::roots) or any parent spine.
    /// Used by the edit ops (a pre-built `Node` with an already-id spine Group content
    /// is inserted as-is — its descendants already live in the arena).
    fn insert_subtree(&mut self, mut node: Node) -> NodeId {
        let id = self.mint_node_id();
        node.id = id;
        self.arena.insert(id, node);
        id
    }

    /// Flatten a [`NodeBuilder`] spec into the [`arena`](Self::arena), returning the
    /// id the spec's node took (ADR 0003 Phase B5). For a [`NodeBuilder::Group`] the
    /// children are inserted first (depth-first), then the Group node is stored with
    /// its spine of minted child ids. Does NOT touch [`roots`](Self::roots) — the
    /// caller splices the returned id where it belongs.
    fn insert_builder(&mut self, spec: NodeBuilder) -> NodeId {
        match spec {
            NodeBuilder::Leaf(node) => self.insert_subtree(node),
            NodeBuilder::Group {
                name,
                transform,
                visible,
                children,
            } => {
                let child_ids: Vec<NodeId> =
                    children.into_iter().map(|child| self.insert_builder(child)).collect();
                let mut group = Node::new(name, NodeContent::Group(child_ids));
                group.transform = transform;
                group.visible = visible;
                self.insert_subtree(group)
            }
        }
    }

    /// Register a reusable [`AssemblyDef`] from `children` built by value (ADR 0003
    /// Phase B5): each child subtree is inserted into the scene [`arena`](Self::arena)
    /// and the def stores their ids as its spine. The terse test/demo helper mirroring
    /// [`from_nodes`](Self::from_nodes) for definition bodies.
    pub fn add_definition<I, B>(&mut self, id: DefId, name: impl Into<String>, children: I)
    where
        I: IntoIterator<Item = B>,
        B: Into<NodeBuilder>,
    {
        let child_ids: Vec<NodeId> = children
            .into_iter()
            .map(|child| self.insert_builder(child.into()))
            .collect();
        self.definitions.push(AssemblyDef {
            id,
            name: name.into(),
            children: child_ids,
        });
    }

    /// Ensure the scene has exactly one **Origin** Point (issue #29). If no Point
    /// has `is_origin == true`, insert one at index 0 with the spec defaults
    /// (ground plane + axes on; positioned at the world origin). Idempotent: a
    /// second call (or a load of a scene that already carries an Origin) does
    /// nothing. Called on every load path so every scene gains its Origin.
    pub fn ensure_origin_point(&mut self) {
        if self.points.iter().any(|point| point.is_origin) {
            return;
        }
        self.points.insert(
            0,
            Point {
                name: "Origin".to_string(),
                position_blocks: [0, 0, 0],
                offset_voxels: [0, 0, 0],
                // Z-up: the ground plane is XY (`plane_xy`).
                plane_xz: false,
                plane_xy: true,
                plane_yz: false,
                axis_x: true,
                axis_y: true,
                axis_z: true,
                hidden: false,
                is_origin: true,
            },
        );
    }

    /// Mint a stable [`NodeId`] for every still-unassigned node (ADR 0003 Phase B).
    /// Walks the top-level nodes, every [`NodeContent::Group`]'s children, and every
    /// definition's nodes; any node carrying the `NodeId(0)` sentinel gets a fresh id
    /// from [`next_node_id`](Self::next_node_id). The counter is first advanced past
    /// any ids ALREADY present (a loaded scene may carry minted ids) so new ids never
    /// collide. **Idempotent:** a second call mints nothing (every node already has a
    /// non-zero id). Called on the load/normalization path alongside
    /// [`ensure_origin_point`](Self::ensure_origin_point).
    pub fn ensure_node_ids(&mut self) {
        // Advance the counter past any ids already present, so freshly-minted ids
        // never collide with ones a loaded scene already carries. The arena keys ARE
        // every node's id (BTreeMap → ascending order, so the scan is stable), so a
        // single pass over the arena values + the definition spines covers it. Note:
        // a node carrying the `NodeId(0)` sentinel is stored UNDER key 0 in the arena
        // (a fresh-by-value insert always mints, but a deserialized arena could carry
        // a single 0-keyed node), so the `max` ignores 0 naturally.
        let mut max_existing = 0u64;
        for id in self.arena.keys() {
            max_existing = max_existing.max(id.0);
        }
        self.next_node_id = self.next_node_id.max(max_existing + 1).max(1);

        // Re-key any still-unassigned node out of the `NodeId(0)` sentinel slot. With
        // the arena keyed by id, minting a fresh id means MOVING the arena entry AND
        // repointing the one spine slot (`roots`, a Group's children, or a definition's
        // children) that referenced it — otherwise the spine keeps pointing at slot 0
        // while the node lives elsewhere, silently orphaning it on load (it would never
        // render, list, or select). At most one node can sit under key 0 (BTreeMap keys
        // are unique). In practice every arena/def node is minted at insert time
        // (`insert_subtree`), so this is a safety net for a deserialized scene that
        // carries a `NodeId(0)` node.
        if self.arena.contains_key(&NodeId(0)) {
            let fresh = NodeId(self.next_node_id);
            self.next_node_id += 1;
            // Repoint the spine FIRST (while the node is still at key 0), then move the
            // arena entry. Mutating the `Vec<NodeId>` spines never borrows another arena
            // node, so no nested-borrow dance is needed.
            let repointed = self.repoint_spine_id(NodeId(0), fresh);
            debug_assert!(
                repointed,
                "a NodeId(0) arena node must be referenced by some spine slot",
            );
            if let Some(mut node) = self.arena.remove(&NodeId(0)) {
                node.id = fresh;
                self.arena.insert(fresh, node);
            }
        }
    }

    /// Replace every spine reference to `old` with `new` across the top-level
    /// [`roots`](Self::roots), every [`NodeContent::Group`]'s children, and every
    /// definition's children. Returns whether any slot was repointed. Used when
    /// re-keying a node in the arena (its id is its key, so the references that name
    /// it must move with it). Touches only the `Vec<NodeId>` spines — it never looks
    /// up another arena node, so it borrows the arena mutably without nesting.
    fn repoint_spine_id(&mut self, old: NodeId, new: NodeId) -> bool {
        let mut repointed = false;
        for slot in self.roots.iter_mut() {
            if *slot == old {
                *slot = new;
                repointed = true;
            }
        }
        for node in self.arena.values_mut() {
            if let NodeContent::Group(children) = &mut node.content {
                for slot in children.iter_mut() {
                    if *slot == old {
                        *slot = new;
                        repointed = true;
                    }
                }
            }
        }
        for definition in self.definitions.iter_mut() {
            for slot in definition.children.iter_mut() {
                if *slot == old {
                    *slot = new;
                    repointed = true;
                }
            }
        }
        repointed
    }

    /// Append a reference [`Point`] to the scene (issue #29). A newly-added user
    /// Point defaults to **all planes OFF** (XZ/XY/YZ) with its **axes ON** (issue
    /// #29 fix): only the Origin keeps the ground (XY, Z-up) plane on by default (via
    /// [`ensure_origin_point`](Self::ensure_origin_point)). The plane/axis flags on
    /// the passed `point` are overridden here so every "+ Add Point" path gets the
    /// clean default; the caller controls only the point's name/position/identity.
    pub fn add_point(&mut self, mut point: Point) {
        point.plane_xz = false;
        point.plane_xy = false;
        point.plane_yz = false;
        point.axis_x = true;
        point.axis_y = true;
        point.axis_z = true;
        self.points.push(point);
    }

    /// Remove the Point at `index` (issue #29). **No-op if it is the Origin** (the
    /// Origin is undeletable) or the index is out of range. Hiding the Origin is
    /// done by setting its `hidden` flag (see [`toggle_point_hidden`]), not by
    /// removal.
    ///
    /// [`toggle_point_hidden`]: Self::toggle_point_hidden
    pub fn remove_point(&mut self, index: usize) {
        match self.points.get(index) {
            Some(point) if !point.is_origin => {
                self.points.remove(index);
            }
            _ => {}
        }
    }

    /// Toggle the `hidden` flag of the Point at `index` (issue #29). Works for the
    /// Origin too — the Origin is hideable (just not deletable). No-op for an
    /// out-of-range index.
    pub fn toggle_point_hidden(&mut self, index: usize) {
        if let Some(point) = self.points.get_mut(index) {
            point.hidden = !point.hidden;
        }
    }

    /// Look up a reusable definition by its [`DefId`] (ADR 0001 step 4). Returns
    /// `None` when no definition carries that id — an `Instance` pointing at a
    /// missing definition resolves to nothing.
    pub fn def_by_id(&self, id: DefId) -> Option<&AssemblyDef> {
        self.definitions.iter().find(|def| def.id == id)
    }

    /// The node at `path`, walking from `nodes` down through Group
    /// children. `None` when any index along the path is out of range or the path
    /// tries to descend through a non-Group (a Tool / Part / Instance has no
    /// addressable children).
    pub fn node_at_path(&self, path: &NodePath) -> Option<&Node> {
        // Walk the id-spine (`roots`, then each Group's `Vec<NodeId>`) for ORDER,
        // fetching each node's content from the arena. ADR 0003 Phase B5.
        let mut siblings: &[NodeId] = &self.roots;
        let mut found: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                found = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                siblings = children;
            } else {
                return None;
            }
        }
        found
    }

    /// The node at `path`, mutably (the inspector edits through this). ADR 0003
    /// Phase B5: resolve the path to a single [`NodeId`] over the id-spine (a shared
    /// walk), then take ONE mutable arena borrow at the end — so the descent never
    /// holds an aliasing `&mut` into the arena.
    pub fn node_at_path_mut(&mut self, path: &NodePath) -> Option<&mut Node> {
        let id = self.id_at_path(path)?;
        self.arena.get_mut(&id)
    }

    /// The [`NodeId`] of the node at `path` — the top-level-tree inverse of
    /// [`path_of`](Self::path_of) — or `None` if the path doesn't resolve (ADR 0003
    /// Phase B2). A convenience bridge while selection/commands migrate off
    /// [`NodePath`] onto [`NodeId`].
    pub fn id_at_path(&self, path: &NodePath) -> Option<NodeId> {
        self.node_at_path(path).map(|node| node.id)
    }

    /// The node with the given [`NodeId`] in the **top-level assembly tree**
    /// (top-level nodes + [`NodeContent::Group`] children — the same scope
    /// [`NodePath`] addresses), or `None` (ADR 0003 Phase B2). `NodeId(0)` (the
    /// unassigned sentinel) never matches. O(n) DFS; Phase B5 swaps the storage for
    /// an arena so this becomes a direct lookup.
    pub fn node_by_id(&self, id: NodeId) -> Option<&Node> {
        // ADR 0003 Phase B5: the arena IS keyed by NodeId, so this is a direct
        // lookup (was an O(n) DFS). The `NodeId(0)` unassigned sentinel never matches.
        if id == NodeId(0) {
            return None;
        }
        self.arena.get(&id)
    }

    /// The node with the given [`NodeId`], mutably (ADR 0003 Phase B2). Same scope +
    /// caveats as [`node_by_id`](Self::node_by_id).
    pub fn node_by_id_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        // ADR 0003 Phase B5: direct id-keyed arena lookup.
        if id == NodeId(0) {
            return None;
        }
        self.arena.get_mut(&id)
    }

    /// Set the `visible` flag of the node identified by `id` (ADR 0003 Phase B4),
    /// returning whether the id resolved to a node. A NodeId-typed edit op so the
    /// panel's visibility checkbox can mutate by identity rather than by path.
    pub fn set_node_visible(&mut self, id: NodeId, visible: bool) -> bool {
        match self.node_by_id_mut(id) {
            Some(node) => {
                node.visible = visible;
                true
            }
            None => false,
        }
    }

    /// The [`NodePath`] addressing the node with the given [`NodeId`] in the
    /// top-level assembly tree, or `None` (ADR 0003 Phase B2). The inverse of
    /// [`id_at_path`](Self::id_at_path): `path_of(id_at_path(path)) == Some(path)`
    /// for every path that resolves. While `NodePath` is still the identity of
    /// record, this lets callers hold a stable [`NodeId`] and recover its current
    /// position on demand.
    pub fn path_of(&self, id: NodeId) -> Option<NodePath> {
        // ADR 0003 Phase B5: walk the id-spine (`roots`, then each Group's spine) for
        // ORDER, fetching content from the arena — the canonical render-time NodePath
        // projection. The arena is get-only here.
        fn search(scene: &Scene, spine: &[NodeId], id: NodeId, prefix: &mut Vec<usize>) -> bool {
            for (index, &child_id) in spine.iter().enumerate() {
                prefix.push(index);
                if child_id == id {
                    return true;
                }
                if let Some(NodeContent::Group(children)) =
                    scene.arena.get(&child_id).map(|node| &node.content)
                {
                    if search(scene, children, id, prefix) {
                        return true;
                    }
                }
                prefix.pop();
            }
            false
        }
        if id == NodeId(0) {
            return None;
        }
        let mut prefix = Vec::new();
        search(self, &self.roots, id, &mut prefix).then(|| NodePath::from_indices(prefix))
    }

    /// Flatten the top-level assembly into a depth-first list of `(path, id, depth)`
    /// rows for the tree UI (ADR 0001 step 4): every top-level node, and — for a
    /// [`NodeContent::Group`] — its children recursively at increasing depth. The
    /// rows are in display order (a parent immediately precedes its children).
    /// `Instance` nodes are leaves here (their definition's body is stored
    /// separately and rendered in the Definitions list, not inlined into the tree).
    ///
    /// ADR 0003 Phase B4: each row also carries the node's stable [`NodeId`] so the
    /// panel can feed the now-NodeId-typed select/delete/visibility ops directly,
    /// without a `path → id` round-trip; the `NodePath` stays for depth/path display.
    pub fn tree_rows(&self) -> Vec<(NodePath, NodeId, usize)> {
        let mut rows = Vec::new();
        collect_tree_rows(self, &self.roots, &mut Vec::new(), 0, &mut rows);
        rows
    }

    /// The active node, if any. ADR 0003 Phase B3: resolves the selected
    /// [`NodeId`] via [`node_by_id`](Self::node_by_id) (a stale id → `None`).
    pub fn active_node(&self) -> Option<&Node> {
        self.active.and_then(|id| self.node_by_id(id))
    }

    /// The active node mutably, if any (the inspector edits through this). ADR 0003
    /// Phase B3: resolves the selected [`NodeId`] via
    /// [`node_by_id_mut`](Self::node_by_id_mut).
    pub fn active_node_mut(&mut self) -> Option<&mut Node> {
        let id = self.active?;
        self.node_by_id_mut(id)
    }

    /// The [`NodePath`] currently addressing the active node, or `None` when nothing
    /// is selected (or the selected [`NodeId`] no longer resolves). ADR 0003 Phase
    /// B3: a positional bridge for the few call sites + tests that still reason in
    /// paths, now that [`active`](Self::active) stores an id.
    pub fn active_path(&self) -> Option<NodePath> {
        self.active.and_then(|id| self.path_of(id))
    }

    /// The transform gizmo's placement for the **active/selected** node, in the
    /// SAME recentred render frame the resolved voxels live in (issue #29 S2).
    /// `None` when nothing is selected (the gizmo is hidden) or the selection has
    /// no intrinsic extent (e.g. a lone Part with no size).
    ///
    /// Returns `(pivot_voxels, extent_voxels)`:
    /// * `pivot_voxels` — the **centre** of the node's block-aligned AABB in the
    ///   recentred frame: `block_aabb_centre · density − recentre_voxels`. The
    ///   gizmo is anchored here so it sits ON the object rather than at the
    ///   composite origin. (We chose the AABB centre over the node's corner-origin
    ///   so a single-axis-offset child still reads as "on the object".)
    /// * `extent_voxels` — the node's own AABB size in voxels, so the gizmo is
    ///   sized from the SELECTED node's extent (not the whole region).
    ///
    /// For a Group / Instance selection the AABB is the union of all leaves under
    /// it (the same union `placed_extent_blocks` forms scene-wide, but rooted at
    /// the selected node). Single-node scenes recentre that node onto the origin,
    /// so its pivot is `[0, 0, 0]` — the gizmo only visibly *moves* with a
    /// multi-node selection (which is the point of a per-selection manipulator).
    pub fn active_gizmo_placement(
        &self,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let path = self.active_path()?;
        self.gizmo_placement_at_path(&path, voxels_per_block)
    }

    /// The recentred `(pivot_voxels, extent_voxels)` for the node identified by
    /// `node_id` — the SAME computation as [`active_gizmo_placement`](Self::active_gizmo_placement)
    /// but scoped to an arbitrary node rather than the active selection. Used by the
    /// camera "Focus" view action (right-click a tree row → frame that node): the
    /// camera target is set to `pivot` and the distance fitted from `extent`.
    /// `None` when the id no longer resolves or the node's subtree has no extent.
    pub fn gizmo_placement_for_id(
        &self,
        node_id: NodeId,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let path = self.path_of(node_id)?;
        self.gizmo_placement_at_path(&path, voxels_per_block)
    }

    /// Shared body of [`active_gizmo_placement`](Self::active_gizmo_placement) and
    /// [`gizmo_placement_for_id`](Self::gizmo_placement_for_id): the recentred pivot
    /// (centre of the node subtree's block-aligned AABB) + its extent, in voxels.
    fn gizmo_placement_at_path(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        // The gizmo PIVOT is the centre of the node's PRODUCER-TRUE voxel AABB — the
        // exact frame the resolved voxels (and the composite recentre) live in. This
        // makes a lone node of ANY size (even or odd) recentre onto the origin: its
        // producer centre coincides with the composite recentre. (Center-anchoring
        // retirement: we no longer mix the block-floored AABB centre with the voxel
        // recentre, which left odd sizes half a block off.)
        let (min_voxels, max_voxels) = self.node_subtree_extent_voxels(path, voxels_per_block)?;
        // The gizmo SIZE is the node's enclosing-whole-block extent (the visible box
        // snaps to whole blocks), taken from the block-AABB.
        let (min_blocks, max_blocks) = self.node_subtree_extent_blocks(path, voxels_per_block)?;
        let density = voxels_per_block.max(1) as i64;
        let mut pivot = [0.0f32; 3];
        let mut extent = [0.0f32; 3];
        // Unwrap the carried frame at the recentred pivot arithmetic.
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for axis in 0..3 {
            // Producer-true voxel-AABB centre minus the composite recentre — same
            // frame the resolved voxels sit in. `* 1` then `/ 2.0` last avoids a
            // half-voxel rounding bias on an odd voxel span.
            let centre_voxels = min_voxels[axis] + max_voxels[axis];
            let pivot_voxels = centre_voxels - 2 * recentre[axis];
            pivot[axis] = pivot_voxels as f32 / 2.0;
            extent[axis] = ((max_blocks[axis] - min_blocks[axis]) * density) as f32;
        }
        Some((pivot, extent))
    }

    /// Append `node` to the TOP-LEVEL list and make it the active selection.
    /// Returns its top-level index.
    ///
    /// ADR 0003 Phase B3: selection is keyed by [`NodeId`], so the appended node is
    /// minted a stable id here ([`mint_node_id`](Self::mint_node_id)) before
    /// `active` is pointed at it — a freshly-added node is selectable by identity
    /// immediately, surviving any later reorder.
    pub fn add_node(&mut self, node: Node) -> usize {
        // The arena insert (mint id, stamp it, store) is exactly `insert_subtree`.
        let id = self.insert_subtree(node);
        self.roots.push(id);
        let index = self.roots.len() - 1;
        self.active = Some(id);
        index
    }

    /// Mint the next fresh [`NodeId`] from the document counter (ADR 0003 Phase B3),
    /// advancing it past the value handed out. Matches the
    /// [`ensure_node_ids`](Self::ensure_node_ids) convention: ids start at `1`
    /// (`0` is the unassigned sentinel). Used by the `add_*` edit ops so a new node
    /// carries a stable id the moment it joins the tree.
    fn mint_node_id(&mut self) -> NodeId {
        self.next_node_id = self.next_node_id.max(1);
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    /// Append `node` as a child of the Group identified by `group_id` and select
    /// it. Returns `true` if the target was a Group and the node was added. A no-op
    /// (returns `false`) when the id does not resolve to a Group.
    pub fn add_child_to_group(&mut self, group_id: NodeId, mut node: Node) -> bool {
        // ADR 0003 Phase B4: the op targets a NodeId; resolve it to the positional
        // path the internal storage still needs (the positional bridge survives
        // until B5). A stale id → no-op (mirrors the old out-of-range path bail).
        let Some(group_path) = self.path_of(group_id) else {
            return false;
        };
        let group_path = &group_path;
        // Bail before minting if the target is not a Group, so a no-op neither adds
        // a node nor burns a counter value.
        match self.node_at_path(group_path).map(|node| &node.content) {
            Some(NodeContent::Group(_)) => {}
            _ => return false,
        }
        // Mint the child's stable id (ADR 0003 Phase B3) so selection can point at
        // it by identity; minting BEFORE the mutable group borrow releases the
        // `&mut next_node_id` borrow so it can't overlap the arena borrow (B5).
        let id = self.mint_node_id();
        node.id = id;
        // Insert the child into the arena (its `Node` lives there now), then splice
        // its id onto the Group's spine. The arena insert is independent of the group
        // borrow, so the two `&mut arena` accesses are sequential, not overlapping.
        self.arena.insert(id, node);
        let Some(group_node) = self.node_at_path_mut(group_path) else {
            // Unreachable (we checked it is a Group above), but keep the arena clean.
            self.arena.remove(&id);
            return false;
        };
        let NodeContent::Group(children) = &mut group_node.content else {
            self.arena.remove(&id);
            return false;
        };
        children.push(id);
        self.active = Some(id);
        true
    }

    /// Remove the node identified by `target_id` (top-level or a Group child),
    /// keeping the `active` selection sensible: after a removal the selection falls
    /// back to the removed node's parent (so a Group's last child deletion selects
    /// the Group), or to a surviving top-level node, or `None` when the scene
    /// empties. A stale id (no longer in the tree) is ignored.
    pub fn remove_node(&mut self, target_id: NodeId) {
        // ADR 0003 Phase B4/B5: resolve the target NodeId to its positional path (the
        // removal + fallback logic reason in indices). A stale id → no-op.
        let Some(path) = self.path_of(target_id) else {
            return;
        };
        let Some((&last_index, parent_indices)) = path.indices.split_last() else {
            return;
        };
        // Splice the target's id out of its parent spine (top-level `roots` or a
        // Group's `Vec<NodeId>`), capturing the removed id.
        let removed_id = {
            let parent_path = NodePath::from_indices(parent_indices.to_vec());
            match self.siblings_mut(&parent_path) {
                Some(spine) if last_index < spine.len() => Some(spine.remove(last_index)),
                _ => None,
            }
        };
        let Some(removed_id) = removed_id else {
            return;
        };
        // B5: the spine splice only detached the id; the `Node`s still live in the
        // arena. Gather the WHOLE detached subtree's ids (the removed node + every
        // descendant, via a shared-borrow DFS into a `Vec` so no arena borrow is held
        // during removal), then drop each from the arena. Leaving any behind would
        // orphan it (a round-trip / count test would catch it).
        let mut to_remove = Vec::new();
        self.collect_subtree_ids(removed_id, &mut to_remove);
        for id in to_remove {
            self.arena.remove(&id);
        }
        // Re-derive a valid selection. Prefer the sibling now occupying the removed
        // slot (a Group, or the scene root → a surviving top-level node); fall back
        // to the parent Group, then None when empty. ADR 0003 Phase B3: the fallback
        // yields a NodePath, which we resolve to the surviving node's stable id.
        self.active = self
            .fallback_selection_after_remove(parent_indices, last_index)
            .and_then(|path| self.id_at_path(&path));
    }

    /// The parent of the node `id` in the top-level assembly tree, and its index in
    /// that parent's spine (ADR 0003 Phase C C2 undo support): `(Some(parent_id),
    /// index)` for a Group child, `(None, index)` for a top-level node. `None` when the
    /// id does not resolve. Used to CAPTURE a node's slot before a structural edit so
    /// the inverse can splice it back at the same place.
    pub fn parent_and_index_of(&self, id: NodeId) -> Option<(Option<NodeId>, usize)> {
        let path = self.path_of(id)?;
        let (&last_index, parent_indices) = path.indices.split_last()?;
        if parent_indices.is_empty() {
            return Some((None, last_index));
        }
        // The parent is the node the parent-prefix path resolves to (always a Group,
        // since a non-Group has no addressable children).
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let parent_id = self.id_at_path(&parent_path)?;
        Some((Some(parent_id), last_index))
    }

    /// Clone the detached subtree rooted at `root_id` (the node + every descendant
    /// through [`NodeContent::Group`] spines) into a `Vec<Node>`, root first, in the
    /// SAME DFS order as [`collect_subtree_ids`](Self::collect_subtree_ids) (ADR 0003
    /// Phase C C2 undo support). Captured BEFORE a `remove_node` so the inverse can
    /// re-insert every `Node` under its ORIGINAL id. Definition bodies are NOT followed
    /// (an `Instance` references a def stored separately).
    pub fn clone_subtree_nodes(&self, root_id: NodeId) -> Vec<Node> {
        let mut ids = Vec::new();
        self.collect_subtree_ids(root_id, &mut ids);
        ids.into_iter()
            .filter_map(|id| self.arena.get(&id).cloned())
            .collect()
    }

    /// Remove the node `id` (and its whole subtree) from the arena + splice its id out
    /// of its parent spine, WITHOUT re-deriving the `active` selection (ADR 0003 Phase
    /// C C2). The undo path restores selection itself from the command's captured
    /// `selection_before`, so unlike [`remove_node`](Self::remove_node) this must not
    /// touch `active`. Used to reverse a single-node mint (`Inverse::RemoveAdded`). A
    /// stale id is a no-op.
    pub fn remove_node_exact(&mut self, id: NodeId) {
        let Some(path) = self.path_of(id) else {
            return;
        };
        let Some((&last_index, parent_indices)) = path.indices.split_last() else {
            return;
        };
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let removed_id = match self.siblings_mut(&parent_path) {
            Some(spine) if last_index < spine.len() => spine.remove(last_index),
            _ => return,
        };
        let mut to_remove = Vec::new();
        self.collect_subtree_ids(removed_id, &mut to_remove);
        for descendant in to_remove {
            self.arena.remove(&descendant);
        }
    }

    /// Reverse [`group_active`](Self::group_active) (ADR 0003 Phase C C2): the fresh
    /// `group` node took `target`'s spine slot and adopted `target` as its sole child.
    /// Put `target`'s id back in the slot `group` occupies and drop `group` from the
    /// arena. Does NOT touch `active` (the undo path restores it). A no-op if `group`
    /// no longer resolves.
    pub fn ungroup_node(&mut self, group: NodeId, target: NodeId) {
        let Some(path) = self.path_of(group) else {
            return;
        };
        let Some((&last_index, parent_indices)) = path.indices.split_last() else {
            return;
        };
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        if let Some(spine) = self.siblings_mut(&parent_path) {
            if last_index < spine.len() {
                spine[last_index] = target;
            }
        }
        self.arena.remove(&group);
    }

    /// Re-insert a detached subtree captured by [`clone_subtree_nodes`](Self::clone_subtree_nodes)
    /// (ADR 0003 Phase C C2): store every `Node` back in the arena under its ORIGINAL
    /// id (safe — the monotonic counter never reuses an id), then splice the root id
    /// (`nodes[0]`) into `parent`'s spine (`None` = top-level `roots`) at `index`.
    /// Reverses a [`remove_node`](Self::remove_node). Does NOT touch `active`.
    pub fn reinsert_subtree(&mut self, parent: Option<NodeId>, index: usize, nodes: &[Node]) {
        let Some(root) = nodes.first() else {
            return;
        };
        let root_id = root.id;
        for node in nodes {
            self.arena.insert(node.id, node.clone());
        }
        match parent {
            None => {
                let clamped = index.min(self.roots.len());
                self.roots.insert(clamped, root_id);
            }
            Some(parent_id) => {
                if let Some(parent_node) = self.arena.get_mut(&parent_id) {
                    if let NodeContent::Group(children) = &mut parent_node.content {
                        let clamped = index.min(children.len());
                        children.insert(clamped, root_id);
                    }
                }
            }
        }
    }

    /// Collect `root_id` and every descendant id (through [`NodeContent::Group`]
    /// spines) into `out`, via a shared-borrow DFS over the arena (ADR 0003 Phase B5).
    /// Used by [`remove_node`](Self::remove_node) to gather a detached subtree's ids
    /// up front so the arena entries can be dropped without holding a borrow across
    /// the removal. Definition bodies are NOT followed (an `Instance` references a
    /// def stored separately; deleting an instance never deletes the shared body).
    fn collect_subtree_ids(&self, root_id: NodeId, out: &mut Vec<NodeId>) {
        out.push(root_id);
        // Snapshot the Group's spine length, then re-fetch each child id by position
        // for the recursive descent — so no `&self.arena.get` borrow is held across
        // the recursive `&self` call (and no per-group spine clone is allocated).
        let child_count = match self.arena.get(&root_id).map(|node| &node.content) {
            Some(NodeContent::Group(children)) => children.len(),
            _ => return,
        };
        for child_index in 0..child_count {
            let Some(NodeContent::Group(children)) =
                self.arena.get(&root_id).map(|node| &node.content)
            else {
                return;
            };
            let child_id = children[child_index];
            self.collect_subtree_ids(child_id, out);
        }
    }

    /// The mutable id-spine addressed by `parent_path` (the empty path → the
    /// top-level [`roots`](Self::roots); otherwise the [`Vec<NodeId>`] of the Group
    /// the path resolves to). `None` when the path does not resolve to a Group.
    /// ADR 0003 Phase B5: returns the SPINE of child ids, not the child `Node`s.
    fn siblings_mut(&mut self, parent_path: &NodePath) -> Option<&mut Vec<NodeId>> {
        if parent_path.indices.is_empty() {
            return Some(&mut self.roots);
        }
        match self.node_at_path_mut(parent_path) {
            Some(node) => match &mut node.content {
                NodeContent::Group(children) => Some(children),
                _ => None,
            },
            None => None,
        }
    }

    /// Choose a valid `active` path after removing the child at `removed_index`
    /// from the sibling list at `parent_indices`.
    fn fallback_selection_after_remove(
        &self,
        parent_indices: &[usize],
        removed_index: usize,
    ) -> Option<NodePath> {
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let sibling_count = if parent_indices.is_empty() {
            self.roots.len()
        } else {
            match self.node_at_path(&parent_path).map(|n| &n.content) {
                Some(NodeContent::Group(children)) => children.len(),
                _ => 0,
            }
        };
        if sibling_count > 0 {
            // Select the sibling now occupying the removed slot (clamped to last).
            let new_index = removed_index.min(sibling_count - 1);
            let mut indices = parent_indices.to_vec();
            indices.push(new_index);
            Some(NodePath::from_indices(indices))
        } else if parent_indices.is_empty() {
            // The whole scene emptied.
            None
        } else {
            // A Group lost its last child — select the (now empty) Group itself.
            Some(parent_path)
        }
    }

    /// Wrap the active node in a new [`NodeContent::Group`] in place (ADR 0001
    /// step 4 authoring): the active node becomes the sole child of a fresh Group
    /// that takes its slot among its siblings. The Group inherits an identity
    /// transform (the child keeps its own offset, so the composite is unchanged),
    /// and the wrapped child becomes the new active selection. Returns the new
    /// Group's [`NodeId`] on success; `None` when there is no active node.
    ///
    /// Grouping a node that is itself a Group simply nests it one level deeper —
    /// the recursion handles arbitrary depth.
    pub fn group_active(&mut self) -> Option<NodeId> {
        // ADR 0003 Phase B3: selection is a NodeId; resolve it to the child's
        // current position to do the positional wrap. The child keeps its id (and
        // thus stays selected by identity); only the new Group needs a fresh id.
        let path = self.active_path()?;
        let (&index, parent_indices) = path.indices.split_last()?;
        let group_id = self.mint_node_id();
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        // B5: the spine carries child IDS. Swap the child's id at `index` for the new
        // Group's id (capturing the child id), so the child `Node` never leaves the
        // arena (only its id moves down one level into the Group's spine) — it keeps
        // its stable identity and stays the active selection.
        let child_id = {
            let siblings = self.siblings_mut(&parent_path)?;
            if index >= siblings.len() {
                return None;
            }
            let child_id = siblings.remove(index);
            siblings.insert(index, group_id);
            child_id
        };
        // The new Group owns the wrapped child by id; store it in the arena.
        let mut group = Node::new("Group", NodeContent::Group(vec![child_id]));
        group.id = group_id;
        self.arena.insert(group_id, group);
        // ADR 0003 Phase B4: return the new Group's stable id (minted above) rather
        // than its positional path.
        Some(group_id)
    }

    /// The smallest unused [`DefId`] (one past the current max, or `DefId(1)` when
    /// there are no definitions — id 0 is reserved/unused for clarity).
    pub fn next_def_id(&self) -> DefId {
        let max = self
            .definitions
            .iter()
            .map(|def| def.id.0)
            .max()
            .unwrap_or(0);
        DefId(max + 1)
    }

    /// Turn the active node into a reusable [`AssemblyDef`] and REPLACE it with an
    /// [`NodeContent::Instance`] of that definition (ADR 0001 step 4: "make
    /// definition from this Group/node"). The active node's content moves into the
    /// new definition's children (a Group's children become the def body; a single
    /// leaf becomes a one-node def); the active node keeps its transform but its
    /// content becomes an `Instance(new_def_id)`. Returns the new [`DefId`] on
    /// success; `None` when there is no active node.
    ///
    /// After this, the active selection stays on the (now-instance) node, and the
    /// definition can be placed again via [`add_instance`](Self::add_instance) —
    /// the village workflow: one stored body, many placements.
    pub fn make_definition_from_active(&mut self, name: impl Into<String>) -> Option<DefId> {
        let def_id = self.next_def_id();
        // ADR 0003 Phase B3: resolve the selected NodeId to its current position.
        // The node keeps its id while only its content becomes an Instance, so the
        // selection stays valid (still the same node by identity) with no re-point.
        let active_id = self.active?;
        // The edit is by id (B5); the `node_by_id_mut` lookup below already bails
        // (`?`) on a stale selection, so no separate presence guard is needed.
        // The definition body, as a spine of arena ids:
        // * a Group DONATES its child id spine (`mem::take` empties the Group's
        //   `Vec<NodeId>`); the child `Node`s STAY in the arena — the def now owns
        //   them by reference, none are orphaned (B5).
        // * any other content becomes a single-node body: a fresh "Body" node
        //   wrapping a clone of the content, inserted into the arena under a new id.
        // First mutate the node's content to the Instance and extract either the
        // donated child-id spine (Group) or a fresh "Body" node to insert (leaf),
        // dropping the `&mut node` arena borrow before any further `&mut self` use.
        enum Body {
            Donated(Vec<NodeId>),
            Leaf(Node),
        }
        let body = {
            let node = self.node_by_id_mut(active_id)?;
            let body = match &mut node.content {
                NodeContent::Group(children) => Body::Donated(std::mem::take(children)),
                other => Body::Leaf(Node::new("Body", other.clone())),
            };
            node.content = NodeContent::Instance(def_id);
            body
        };
        let child_ids: Vec<NodeId> = match body {
            Body::Donated(ids) => ids,
            Body::Leaf(node) => vec![self.insert_subtree(node)],
        };
        self.definitions.push(AssemblyDef {
            id: def_id,
            name: name.into(),
            children: child_ids,
        });
        Some(def_id)
    }

    /// Place another [`NodeContent::Instance`] of the definition `def_id` as a new
    /// top-level node (ADR 0001 step 4: "Add Instance"). The instance is named
    /// after the definition and gets a default offset that nudges it clear of
    /// earlier instances of the same def (so a freshly-added village house does not
    /// land exactly on top of the previous one). Selects the new node. Returns its
    /// [`NodeId`], or `None` when no definition carries `def_id`.
    pub fn add_instance(&mut self, def_id: DefId) -> Option<NodeId> {
        let def = self.def_by_id(def_id)?;
        let name = format!("{} instance", def.name);
        // Nudge each new instance of this def along +X so it does not overlap the
        // previous one. Count existing top-level instances of this def for the step.
        let existing = self
            .roots
            .iter()
            .filter_map(|id| self.arena.get(id))
            .filter(|node| matches!(node.content, NodeContent::Instance(id) if id == def_id))
            .count();
        let mut node = Node::new(name, NodeContent::Instance(def_id));
        // Block-granular auto-spacing → canonical voxels at the document density.
        let spacing_blocks = (existing as i64 + 1) * DEFAULT_INSTANCE_SPACING_BLOCKS as i64;
        node.transform = NodeTransform::from_blocks([spacing_blocks, 0, 0], self.voxels_per_block);
        let index = self.add_node(node);
        // ADR 0003 Phase B4: return the appended node's stable id rather than its
        // positional path. `add_node` minted its id and pointed `active` at it, and
        // `id_at_path` reads it back from the slot it now occupies.
        self.id_at_path(&NodePath::root_index(index))
    }

    /// Build the one-node Tool scene that reproduces today's single-shape
    /// behaviour from the panel's [`GeometryParams`] plus the active
    /// [`MaterialChoice`]. The node is a [`NodeContent::Tool`] wrapping the SDF
    /// shape, carrying `material` as its single material.
    ///
    /// Step 2 removed the `debug_clouds: bool` selector — "Clouds" is now an
    /// Add-a-Part action in the node list ([`Part::DebugClouds`]), not a mode of
    /// the geometry. So this constructor only ever builds a Tool; the back-compat
    /// config load (a single persisted geometry) routes through here.
    pub fn from_geometry(geometry: GeometryParams, material: MaterialChoice) -> Self {
        // Capture the density before `from_geometry` consumes the params (it is no
        // longer `Copy` — it owns an optional boxed retained-size expression).
        let voxels_per_block = geometry.voxels_per_block;
        let mut scene = Self::single_node(Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_geometry(geometry),
                material,
            },
        ));
        // Density is document-level (ADR 0003 §3f(0)): carry the UI control value
        // onto the scene, not the shape.
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    /// Test helper (ADR 0003 Phase B5): the top-level node at positional `index`, via
    /// the [`roots`](Self::roots) spine + arena. Replaces the old `scene.nodes[index]`
    /// positional read now that storage is id-keyed.
    #[cfg(test)]
    pub(crate) fn root_node(&self, index: usize) -> &Node {
        let id = self.roots[index];
        &self.arena[&id]
    }

    /// Test helper (ADR 0003 Phase B5): the top-level node at positional `index`,
    /// mutably. Replaces the old `scene.nodes[index]` positional `&mut`.
    #[cfg(test)]
    pub(crate) fn root_node_mut(&mut self, index: usize) -> &mut Node {
        let id = self.roots[index];
        self.arena.get_mut(&id).expect("root id present in arena")
    }

}

/// Depth-first worker for [`Scene::tree_rows`]: append `(path, depth)` for each
/// node in `nodes`, descending into Group children (a Group's children follow it
/// at `depth + 1`). `prefix` is the path of the assembly that owns `nodes`.
fn collect_tree_rows(
    scene: &Scene,
    spine: &[NodeId],
    prefix: &mut Vec<usize>,
    depth: usize,
    rows: &mut Vec<(NodePath, NodeId, usize)>,
) {
    // Iterate the id-spine for ORDER, fetching content from the arena (ADR 0003 B5).
    for (index, &child_id) in spine.iter().enumerate() {
        prefix.push(index);
        rows.push((NodePath::from_indices(prefix.clone()), child_id, depth));
        if let Some(NodeContent::Group(children)) =
            scene.arena.get(&child_id).map(|node| &node.content)
        {
            collect_tree_rows(scene, children, prefix, depth + 1, rows);
        }
        prefix.pop();
    }
}
