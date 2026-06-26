//! The scene (assembly) model — ADR 0001, sequence step 1.
//!
//! Today the app has exactly one producer, smuggled in through
//! [`GeometryParams`](crate::panel::GeometryParams) (the SDF shape) plus a
//! `debug_clouds: bool` selector. ADR 0001 replaces that single-producer
//! assumption with a **Scene**: an assembly graph of **nodes**, each wrapping a
//! producer plus a placement. This module introduces that model and routes ALL
//! voxel resolution through it.
//!
//! **Step 1 scope (this file):** the data model exists in full (so later steps
//! are data changes, not rewrites), but only the two leaves that exist today are
//! actually resolved:
//!
//!   * [`NodeContent::Tool`] — a *parametric* producer ([`SdfShape`]) that carries
//!     the Tool's single [`MaterialChoice`].
//!   * [`NodeContent::Part`] — a *static* voxel body; today the only variant is
//!     [`Part::DebugClouds`].
//!
//! [`NodeContent::Group`] and [`NodeContent::Instance`] (recursion + reuse) exist
//! as types but are intentionally not resolved yet — see the `// step 4` markers
//! in [`Scene::resolve_region`].
//!
//! ## Identical-behaviour guarantee
//!
//! The producer trait ([`VoxelProducer`]) does **not** change: producers still
//! emit content centred at the origin. The Scene's new job is **compositing** —
//! walk the node tree, resolve each visible leaf into its own local grid, and
//! **stamp** it (under the node's transform) into the output grid. For a one-node
//! scene whose region is the node's full extent with a zero offset, the stamp is
//! the identity, so the resulting [`VoxelGrid`] is bit-for-bit what
//! `SdfShape::resolve` / `DebugCloudField::resolve` produce today (same
//! dimensions, same occupied set). See `tool_scene_matches_bare_producer` below.

use serde::{Deserialize, Serialize};

use crate::debug_clouds::DebugCloudField;
use crate::panel::{GeometryParams, MaterialChoice};
use crate::spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
use crate::voxel::{SdfShape, VoxelGrid, VoxelProducer};

/// Default +X spacing (in blocks) between successive instances of the same
/// definition added via [`Scene::add_instance`], so a freshly-placed village
/// house lands clear of the previous one instead of exactly on top of it.
const DEFAULT_INSTANCE_SPACING_BLOCKS: i32 = 6;

/// The working volume the scene resolves into, expressed in **whole blocks**
/// (ADR 0001 "Scale": the canvas is the user-set stock / build volume). Step 1
/// always resolves the whole extent as a single region, so this equals the lone
/// node's block extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionBlocks {
    /// Size of the region in whole blocks (X, Y, Z).
    pub size_blocks: [u32; 3],
}

impl RegionBlocks {
    /// A region of the given whole-block size.
    pub fn new(size_blocks: [u32; 3]) -> Self {
        Self { size_blocks }
    }
}

/// A reusable identifier for a [`Tool`-or-`Part`](NodeContent) definition that an
/// [`NodeContent::Instance`] points at (ADR 0001: reuse by reference). Step 1
/// never constructs an Instance, so this is a forward-declared type only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DefId(pub u32);

/// A path to a node anywhere in the **top-level assembly** (ADR 0001 step 4 UI).
///
/// The path is a list of child indices walked from [`Scene::nodes`] down through
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
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NodePath {
    /// Child indices from the top-level node list down through Group children.
    #[serde(default)]
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

    /// Whether this path addresses a top-level node (one index, no descent).
    pub fn is_top_level(&self) -> bool {
        self.indices.len() == 1
    }

    /// The path of this node's parent (drops the last index), or `None` when this
    /// is already a top-level node (its parent is the scene root).
    pub fn parent(&self) -> Option<NodePath> {
        if self.indices.len() <= 1 {
            None
        } else {
            Some(NodePath {
                indices: self.indices[..self.indices.len() - 1].to_vec(),
            })
        }
    }

    /// The last index in the path (the node's position among its siblings), or
    /// `None` for an empty path.
    pub fn last_index(&self) -> Option<usize> {
        self.indices.last().copied()
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

/// A node's LOCAL placement. v1 exposes integer block translation only, but the
/// type targets a full affine (translation + rotation + scale) so rotation /
/// scale (with voxel resampling) slot in later without a rewrite (ADR 0001
/// decision 3). In step 1 the offset is always `[0, 0, 0]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NodeTransform {
    /// Translation in whole blocks (X, Y, Z).
    ///
    /// **64-bit world addressing (S4a, ADR 0002 Decision 2):** the block offset is
    /// `i64` so far-apart nodes compose down the tree without `i32` overflow (a
    /// node placed at ±10⁹ blocks, or a deep nest summing past the i32 range, is
    /// exact). serde widens an older `i32` JSON number into this field transparently
    /// (a JSON integer carries no width), so previously-saved scenes load unchanged
    /// — see the persistence round-trip tests in `settings.rs`.
    #[serde(default)]
    pub offset_blocks: [i64; 3],
    // future: rotation, scale → a general affine.
}

impl NodeTransform {
    /// The identity transform (zero offset) — the only transform step 1 uses.
    pub fn identity() -> Self {
        Self::default()
    }
}

/// A *static* voxel body with no meaningful generation parameters — dropped in
/// as-is (ADR 0001). v1 has one variant; future variants are saved chiseled
/// blocks and imported `.vox` bodies, each carrying baked per-voxel materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Part {
    /// The debug cloud field (several distinct billowy fBm blobs) — "a part with
    /// one trivial knob" (the seed).
    DebugClouds {
        /// Seed for the deterministic placement + noise permutation.
        #[serde(default)]
        seed: u32,
    },
    // future: SavedBody(VoxelBlob), ImportedVox(...).
}

/// What a node *is*: a leaf producer (Tool or Part) or an interior assembly
/// (Group or Instance).
///
/// Step 1 resolves only the two leaf kinds; `Group` / `Instance` are present as
/// types but unimplemented in [`Scene::resolve_region`] (recursion + instancing
/// arrive in step 4).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeContent {
    /// A parametric producer (an [`SdfShape`]) plus the single material the Tool
    /// assigns to every voxel it emits. Step 1 keeps the existing
    /// [`MaterialChoice`]; a richer material table is a later step.
    Tool {
        /// The parametric primitive to resolve.
        shape: SdfShape,
        /// The single material this Tool stamps onto its voxels.
        material: MaterialChoice,
    },
    /// A static voxel body, dropped in as-is.
    Part(Part),
    /// An owned, one-off sub-assembly. **Not resolved in step 1** (step 4).
    Group(Vec<Node>),
    /// A reuse-by-reference of a definition. **Not resolved in step 1** (step 4).
    Instance(DefId),
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
            name: name.into(),
            transform: NodeTransform::identity(),
            operation: CombineOp::Union,
            visible: true,
            grids: NodeGrids::default(),
            content,
        }
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
    /// The nodes that make up this assembly.
    #[serde(default)]
    pub children: Vec<Node>,
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
    /// Position in the world-block lattice — the SAME frame as
    /// [`NodeTransform::offset_blocks`] (whole blocks, `i64` for far-world
    /// addressing).
    #[serde(default)]
    pub position_blocks: [i64; 3],
    /// Sub-block offset in voxels (v1 keeps `[0, 0, 0]`; the field exists so a
    /// future sub-block placement is a data change, not a rewrite).
    #[serde(default)]
    pub offset_voxels: [i32; 3],
    /// Whether the ground reference plane (XZ) shows. Default **true**.
    #[serde(default = "default_true_bool")]
    pub plane_xz: bool,
    /// Whether the front reference plane (XY) shows. Default false.
    #[serde(default)]
    pub plane_xy: bool,
    /// Whether the side reference plane (YZ) shows. Default false.
    #[serde(default)]
    pub plane_yz: bool,
    /// Whether the axis lines show. Default **true**.
    #[serde(default = "default_true_bool")]
    pub axes: bool,
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
    /// other planes off, visible, NOT the Origin). [`Scene::ensure_origin_point`]
    /// clones this and sets `is_origin`/`name`.
    fn default() -> Self {
        Self {
            name: String::new(),
            position_blocks: [0, 0, 0],
            offset_voxels: [0, 0, 0],
            plane_xz: true,
            plane_xy: false,
            plane_yz: false,
            axes: true,
            hidden: false,
            is_origin: false,
        }
    }
}

/// Default `true` for the scene-wide block-lattice master (issue #29).
fn default_master_block_lattice() -> bool {
    true
}

/// The scene (assembly): a list of placed nodes resolved into the shared
/// [`VoxelGrid`] truth. ADR 0001's full model carries reusable `definitions` too;
/// step 2 added the flat node list plus the `active` selection that drives the
/// inspector. **Step 4** wires up `definitions` so a [`NodeContent::Instance`]
/// resolves the referenced [`AssemblyDef`] under its transform (reuse by
/// reference: a village of identical houses is one definition placed by N
/// instances).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    /// The top-level assembly's nodes, resolved in order (later nodes win on
    /// overlap under [`CombineOp::Union`]).
    #[serde(default)]
    pub nodes: Vec<Node>,
    /// Reusable sub-assemblies referenced by [`NodeContent::Instance`]. A
    /// definition is stored ONCE here regardless of how many instances place it
    /// (ADR 0001 "Nesting & reuse"). Looked up by [`DefId`] via [`def_by_id`].
    ///
    /// [`def_by_id`]: Self::def_by_id
    #[serde(default)]
    pub definitions: Vec<AssemblyDef>,
    /// Path to the active/selected node — the one the inspector edits (ADR 0001
    /// step 4: selection reaches any depth, so a [`Group`](NodeContent::Group)
    /// child is selectable, not just a top-level node). `None` when nothing is
    /// selected. Kept valid (re-clamped / dropped) across add / delete / group.
    #[serde(default)]
    pub active: Option<NodePath>,
    /// World-anchored reference Points (issue #29). Always contains exactly one
    /// Origin Point after [`ensure_origin_point`](Self::ensure_origin_point) runs
    /// on load. An older config without this field deserialises to an empty list,
    /// then gains its Origin on the load path.
    #[serde(default)]
    pub points: Vec<Point>,
    /// Scene-wide master toggle for the block lattice (issue #29). Default
    /// **true**. ANDed with each node's [`NodeGrids::block_lattice`] in S3.
    /// Migrated from the legacy `AppConfig.show_block_lattice` on load.
    #[serde(default = "default_master_block_lattice")]
    pub master_block_lattice: bool,
    /// Scene-wide master toggle for the on-face voxel grid (issue #29). Default
    /// false. Migrated from the legacy `AppConfig.show_grid_overlay` on load.
    #[serde(default)]
    pub master_voxel_grid: bool,
    /// Scene-wide master toggle for the floor grid (issue #29). Default false.
    /// Migrated from the legacy `AppConfig.show_floor_grid` on load.
    #[serde(default)]
    pub master_floor_grid: bool,
    /// The active/selected Point (index into [`points`](Self::points)), or `None`.
    #[serde(default)]
    pub active_point: Option<usize>,
}

impl Default for Scene {
    /// An empty scene with the issue-#29 master defaults (block lattice on, voxel
    /// grid + floor grid off) and no Points yet (the Origin is synthesized on the
    /// load path via [`ensure_origin_point`](Self::ensure_origin_point)).
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            definitions: Vec::new(),
            active: None,
            points: Vec::new(),
            master_block_lattice: true,
            master_voxel_grid: false,
            master_floor_grid: false,
            active_point: None,
        }
    }
}

impl Scene {
    /// A scene with a single node — the shape every one-node call site builds. The
    /// lone node is the active selection.
    pub fn single_node(node: Node) -> Self {
        Self {
            nodes: vec![node],
            active: Some(NodePath::root_index(0)),
            ..Self::default()
        }
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
                plane_xz: true,
                plane_xy: false,
                plane_yz: false,
                axes: true,
                hidden: false,
                is_origin: true,
            },
        );
    }

    /// Append a reference [`Point`] to the scene (issue #29).
    pub fn add_point(&mut self, point: Point) {
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

    /// The node at `path`, walking from [`nodes`](Self::nodes) down through Group
    /// children. `None` when any index along the path is out of range or the path
    /// tries to descend through a non-Group (a Tool / Part / Instance has no
    /// addressable children).
    pub fn node_at_path(&self, path: &NodePath) -> Option<&Node> {
        let mut siblings: &[Node] = &self.nodes;
        let mut found: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let node = siblings.get(index)?;
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

    /// The node at `path`, mutably (the inspector edits through this).
    pub fn node_at_path_mut(&mut self, path: &NodePath) -> Option<&mut Node> {
        let mut siblings: &mut Vec<Node> = &mut self.nodes;
        let count = path.indices.len();
        for (depth, &index) in path.indices.iter().enumerate() {
            let is_last = depth + 1 == count;
            if is_last {
                return siblings.get_mut(index);
            }
            match siblings.get_mut(index)?.content {
                NodeContent::Group(ref mut children) => siblings = children,
                _ => return None,
            }
        }
        None
    }

    /// Flatten the top-level assembly into a depth-first list of `(path, depth)`
    /// rows for the tree UI (ADR 0001 step 4): every top-level node, and — for a
    /// [`NodeContent::Group`] — its children recursively at increasing depth. The
    /// rows are in display order (a parent immediately precedes its children).
    /// `Instance` nodes are leaves here (their definition's body is stored
    /// separately and rendered in the Definitions list, not inlined into the tree).
    pub fn tree_rows(&self) -> Vec<(NodePath, usize)> {
        let mut rows = Vec::new();
        collect_tree_rows(&self.nodes, &mut Vec::new(), 0, &mut rows);
        rows
    }

    /// The active node, if any.
    pub fn active_node(&self) -> Option<&Node> {
        self.active.as_ref().and_then(|path| self.node_at_path(path))
    }

    /// The active node mutably, if any (the inspector edits through this).
    pub fn active_node_mut(&mut self) -> Option<&mut Node> {
        let path = self.active.clone()?;
        self.node_at_path_mut(&path)
    }

    /// Append `node` to the TOP-LEVEL list and make it the active selection.
    /// Returns its top-level index.
    pub fn add_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        let index = self.nodes.len() - 1;
        self.active = Some(NodePath::root_index(index));
        index
    }

    /// Append `node` as a child of the Group at `group_path` and select it.
    /// Returns `true` if the target was a Group and the node was added. A no-op
    /// (returns `false`) when the path does not resolve to a Group.
    pub fn add_child_to_group(&mut self, group_path: &NodePath, node: Node) -> bool {
        let Some(group_node) = self.node_at_path_mut(group_path) else {
            return false;
        };
        let NodeContent::Group(children) = &mut group_node.content else {
            return false;
        };
        children.push(node);
        let child_index = children.len() - 1;
        let mut indices = group_path.indices.clone();
        indices.push(child_index);
        self.active = Some(NodePath::from_indices(indices));
        true
    }

    /// Remove the node at `path` (top-level or a Group child), keeping the `active`
    /// selection sensible: after a removal the selection falls back to the removed
    /// node's parent (so a Group's last child deletion selects the Group), or to a
    /// surviving top-level node, or `None` when the scene empties. Out-of-range
    /// paths are ignored.
    pub fn remove_node(&mut self, path: &NodePath) {
        let Some((&last_index, parent_indices)) = path.indices.split_last() else {
            return;
        };
        // Borrow the sibling list that owns the target node.
        let removed = {
            let parent_path = NodePath::from_indices(parent_indices.to_vec());
            let siblings = self.siblings_mut(&parent_path);
            match siblings {
                Some(siblings) if last_index < siblings.len() => {
                    siblings.remove(last_index);
                    true
                }
                _ => false,
            }
        };
        if !removed {
            return;
        }
        // Re-derive a valid selection. Prefer the parent (a Group, or the scene
        // root → a surviving top-level node); fall back to None when empty.
        self.active = self.fallback_selection_after_remove(parent_indices, last_index);
    }

    /// The mutable sibling list addressed by `parent_path` (the empty path → the
    /// top-level [`nodes`](Self::nodes); otherwise the children of the Group the
    /// path resolves to). `None` when the path does not resolve to a Group.
    fn siblings_mut(&mut self, parent_path: &NodePath) -> Option<&mut Vec<Node>> {
        if parent_path.indices.is_empty() {
            return Some(&mut self.nodes);
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
            self.nodes.len()
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
    /// and the wrapped child becomes the new active selection. Returns the Group's
    /// path on success; `None` when there is no active node.
    ///
    /// Grouping a node that is itself a Group simply nests it one level deeper —
    /// the recursion handles arbitrary depth.
    pub fn group_active(&mut self) -> Option<NodePath> {
        let path = self.active.clone()?;
        let (&index, parent_indices) = path.indices.split_last()?;
        let parent_path = NodePath::from_indices(parent_indices.to_vec());
        let siblings = self.siblings_mut(&parent_path)?;
        if index >= siblings.len() {
            return None;
        }
        let child = siblings.remove(index);
        let group = Node::new("Group", NodeContent::Group(vec![child]));
        siblings.insert(index, group);
        // The Group sits at the old slot; its single child is index 0 within it.
        let group_path = NodePath::from_indices({
            let mut v = parent_indices.to_vec();
            v.push(index);
            v
        });
        let mut child_indices = group_path.indices.clone();
        child_indices.push(0);
        self.active = Some(NodePath::from_indices(child_indices));
        Some(group_path)
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
        let path = self.active.clone()?;
        let node = self.node_at_path_mut(&path)?;
        // The definition body: a Group donates its children; any other content
        // becomes a single-node body wrapping a clone of the node's content.
        let children = match &mut node.content {
            NodeContent::Group(children) => std::mem::take(children),
            other => vec![Node::new("Body", other.clone())],
        };
        node.content = NodeContent::Instance(def_id);
        self.definitions.push(AssemblyDef {
            id: def_id,
            name: name.into(),
            children,
        });
        Some(def_id)
    }

    /// Place another [`NodeContent::Instance`] of the definition `def_id` as a new
    /// top-level node (ADR 0001 step 4: "Add Instance"). The instance is named
    /// after the definition and gets a default offset that nudges it clear of
    /// earlier instances of the same def (so a freshly-added village house does not
    /// land exactly on top of the previous one). Selects the new node. Returns its
    /// path, or `None` when no definition carries `def_id`.
    pub fn add_instance(&mut self, def_id: DefId) -> Option<NodePath> {
        let def = self.def_by_id(def_id)?;
        let name = format!("{} instance", def.name);
        // Nudge each new instance of this def along +X so it does not overlap the
        // previous one. Count existing top-level instances of this def for the step.
        let existing = self
            .nodes
            .iter()
            .filter(|node| matches!(node.content, NodeContent::Instance(id) if id == def_id))
            .count();
        let mut node = Node::new(name, NodeContent::Instance(def_id));
        node.transform.offset_blocks = [(existing as i64 + 1) * DEFAULT_INSTANCE_SPACING_BLOCKS as i64, 0, 0];
        let index = self.add_node(node);
        Some(NodePath::root_index(index))
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
        Self::single_node(Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_geometry(geometry),
                material,
            },
        ))
    }

    /// The whole-block extent of the scene: the per-axis size of the bounding box
    /// that encompasses every placed leaf node (ADR 0001 step 3). Each leaf
    /// occupies `offset_blocks ± size/2`; the composite extent is the union of
    /// those boxes (`max_corner - min_corner` per axis). With every node at a zero
    /// offset this reduces to the per-axis MAX of the node sizes (the step-2
    /// behaviour). A Part-only node (the cloud field, which has no intrinsic size)
    /// contributes no box and adopts whatever extent the Tools establish.
    ///
    /// Returns a zero-sized region when no leaf has an intrinsic size.
    pub fn full_extent_blocks(&self, voxels_per_block: u32) -> RegionBlocks {
        match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => RegionBlocks::new([
                (max_corner[0] - min_corner[0]) as u32,
                (max_corner[1] - min_corner[1]) as u32,
                (max_corner[2] - min_corner[2]) as u32,
            ]),
            // NOTE: the corners are `i64` (S4a 64-bit block addressing); the
            // DIFFERENCE (the region size) is bounded by the placed geometry's own
            // extent, never by how far from the origin it sits, so narrowing to u32
            // is safe — a scene whose *span* exceeds 4G blocks is not representable
            // as a single monolithic grid regardless of addressing width.
            None => RegionBlocks::new([0, 0, 0]),
        }
    }

    /// The composite bounding box of all placed leaf nodes, in **whole-block**
    /// coordinates: `(min_corner, max_corner)` where each leaf with intrinsic
    /// `size_blocks` placed at `offset_blocks` spans
    /// `[offset - size/2, offset - size/2 + size]`. `None` when no leaf has an
    /// intrinsic size (a Part-only scene). Drives both [`full_extent_blocks`] (the
    /// size) and the recentre in [`resolve_region`] (centring the composite so its
    /// world positions sit symmetrically about the origin — what the renderer and
    /// camera assume).
    ///
    /// Block extents are split into a low/high half (`floor(size/2)` below the
    /// centre, the remainder above) so an odd block size keeps the same parity the
    /// voxel-space resolution uses, and the returned box is exact in blocks.
    fn placed_extent_blocks(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset, content| {
            let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                return;
            };
            any = true;
            for axis in 0..3 {
                let half_low = (size_blocks[axis] / 2) as i64;
                let low = world_offset[axis] - half_low;
                let high = low + size_blocks[axis] as i64;
                min_corner[axis] = min_corner[axis].min(low);
                max_corner[axis] = max_corner[axis].max(high);
            }
        });
        any.then_some((min_corner, max_corner))
    }

    /// Walk the whole node tree depth-first, invoking `visitor(world_offset, leaf)`
    /// once for every **visible leaf** (`Tool` / `Part`) with its accumulated
    /// **world** block offset (`parent_offset + node.offset_blocks`, summed down the
    /// tree — translation-only composition, ADR 0001 step 4).
    ///
    /// `Group` children inherit the group's world offset; an `Instance(def)` resolves
    /// the referenced [`AssemblyDef`]'s children under the instance's world offset, so
    /// the SAME definition placed by N instances is visited N times at N locations
    /// (the village-of-reused-houses case). The cycle guard (an `Instance` may not
    /// reference an ancestor definition) lives in [`walk_nodes`].
    ///
    /// [`walk_nodes`]: Self::walk_nodes
    fn for_each_leaf(&self, visitor: &mut dyn FnMut([i64; 3], &NodeContent)) {
        let mut def_path: Vec<DefId> = Vec::new();
        self.walk_nodes(&self.nodes, [0, 0, 0], &mut def_path, visitor);
    }

    /// Recursive worker for [`for_each_leaf`](Self::for_each_leaf). `parent_offset`
    /// is the accumulated world block offset of the assembly that owns `nodes`;
    /// `def_path` is the stack of definition ids currently being expanded (for the
    /// cycle guard — an `Instance` that would re-enter a definition already on the
    /// path is skipped instead of recursing forever).
    fn walk_nodes(
        &self,
        nodes: &[Node],
        parent_offset: [i64; 3],
        def_path: &mut Vec<DefId>,
        visitor: &mut dyn FnMut([i64; 3], &NodeContent),
    ) {
        for node in nodes {
            if !node.visible {
                continue;
            }
            let world_offset = [
                parent_offset[0] + node.transform.offset_blocks[0],
                parent_offset[1] + node.transform.offset_blocks[1],
                parent_offset[2] + node.transform.offset_blocks[2],
            ];
            match &node.content {
                NodeContent::Tool { .. } | NodeContent::Part(_) => {
                    visitor(world_offset, &node.content);
                }
                NodeContent::Group(children) => {
                    self.walk_nodes(children, world_offset, def_path, visitor);
                }
                NodeContent::Instance(def_id) => {
                    // Cycle guard: an Instance may not reference an ancestor
                    // definition. If this id is already being expanded on the
                    // current path, skip it (never recurse into a cycle).
                    if def_path.contains(def_id) {
                        eprintln!(
                            "scene: skipping Instance({def_id:?}) — cyclic reference \
                             to an ancestor definition (path {def_path:?})"
                        );
                        continue;
                    }
                    let Some(def) = self.def_by_id(*def_id) else {
                        // An Instance pointing at a missing definition resolves to
                        // nothing (no panic — the model stays robust to dangling ids).
                        continue;
                    };
                    def_path.push(*def_id);
                    self.walk_nodes(&def.children, world_offset, def_path, visitor);
                    def_path.pop();
                }
            }
        }
    }

    /// Resolve `region` into a fresh [`VoxelGrid`] by a union tree-walk: each
    /// visible leaf producer is resolved into its own local grid and **stamped**
    /// into the output under the node's transform.
    ///
    /// `voxels_per_block` is the application density (ADR 0001 "Density": a global
    /// setting, default 16, that the scene reads at resolve time).
    ///
    /// `lod` is the level-of-detail seam required by ADR 0001 ("Deferred: LOD").
    /// It is **always `0`** (full resolution) for now; the parameter exists from
    /// day one so a future LOD level (which would downsample a chunk before
    /// meshing) is a possible change rather than a signature break. Step 1
    /// asserts it is `0`.
    ///
    /// **Identical-behaviour guarantee:** for a one-node scene whose `region`
    /// equals the node's full extent with a zero offset, the stamp is the
    /// identity, so the result equals what the bare producer emits today.
    pub fn resolve_region(
        &self,
        region: RegionBlocks,
        voxels_per_block: u32,
        lod: u32,
    ) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "step 1 only resolves full resolution (lod 0)");

        let region_dimensions = [
            region.size_blocks[0] * voxels_per_block,
            region.size_blocks[1] * voxels_per_block,
            region.size_blocks[2] * voxels_per_block,
        ];
        let mut output = VoxelGrid::new(region_dimensions);

        // Recentre the composite so its world positions sit symmetrically about
        // the origin (what the renderer + camera auto-frame assume). Each producer
        // emits voxels centred on ITS OWN grid; a node's placed centre in the
        // composite's voxel space is `offset_voxels`, and the whole composite's
        // centre is `((min + max) / 2) * voxels_per_block`. Subtracting that centre
        // from every node's translation lands the composite centred in `output`.
        // With a single zero-offset node the composite centre is the node's own
        // centre, so the shift is zero — the step-2 identity is preserved.
        let recentre_voxels = match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => [
                ((min_corner[0] + max_corner[0]) * voxels_per_block as i64) / 2,
                ((min_corner[1] + max_corner[1]) * voxels_per_block as i64) / 2,
                ((min_corner[2] + max_corner[2]) * voxels_per_block as i64) / 2,
            ],
            None => [0i64; 3],
        };

        // Walk the whole tree (groups + instances recurse, composing world
        // translation down — ADR 0001 step 4). Each visited leaf is stamped under
        // its WORLD offset (× density) minus the composite recentre. All of this is
        // in i64 (S4a) so a far-placed node composes without overflow; the result
        // is downcast to f32 inside the stamp (the render frame stays f32 — S4b
        // makes the far case byte-identical via origin rebasing).
        self.for_each_leaf(&mut |world_offset, content| {
            // Snap a sized leaf onto the global block lattice (issue #30): an
            // odd-block leaf's producer grid straddles a block boundary by half a
            // block, so shift it by `leaf_lattice_shift_voxels` before placing. A
            // Part (no intrinsic size) has no lattice frame, so its shift is zero.
            let lattice_shift = match leaf_size_blocks(content, voxels_per_block) {
                Some(size_blocks) => leaf_lattice_shift_voxels(size_blocks, voxels_per_block),
                None => [0i64; 3],
            };
            let translation_voxels = [
                world_offset[0] * voxels_per_block as i64 + lattice_shift[0] - recentre_voxels[0],
                world_offset[1] * voxels_per_block as i64 + lattice_shift[1] - recentre_voxels[1],
                world_offset[2] * voxels_per_block as i64 + lattice_shift[2] - recentre_voxels[2],
            ];
            match content {
                NodeContent::Tool { shape, material } => {
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        material_id_for(*material),
                        shape,
                    );
                }
                NodeContent::Part(Part::DebugClouds { seed }) => {
                    let producer = DebugCloudField {
                        // The cloud field sizes itself from the region (today's
                        // behaviour resolved it at the shape's grid dimensions).
                        dimensions: region_dimensions,
                        voxels_per_block,
                        seed: *seed,
                    };
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        // A Part brings its own per-voxel materials; today the
                        // cloud field emits material 0, so the stamp keeps that.
                        None,
                        &producer,
                    );
                }
                // `for_each_leaf` only ever yields leaf content (Tool / Part); the
                // interior kinds were already recursed through by the walk.
                NodeContent::Group(_) | NodeContent::Instance(_) => {}
            }
        });

        output
    }

    /// Resolve exactly **one chunk** of the scene into a fresh [`VoxelGrid`], in
    /// **absolute (non-recentred) composite voxel coordinates**.
    ///
    /// This is the chunk-addressable counterpart to [`resolve_region`] required by
    /// issue #27 (deep chunked resolve). It is **additive**: the live render path
    /// still goes through [`resolve_region`] (which recentres the composite on the
    /// origin); this path does **not** recentre, so its voxel positions are the
    /// scene's true composite coordinates. The two frames differ by exactly the
    /// recentre offset [`resolve_region`] subtracts (see
    /// [`recentre_voxels`](Self::recentre_voxels)).
    ///
    /// A chunk is a `CHUNK_BLOCKS³`-block cell (`CHUNK_BLOCKS = 4`,
    /// [`crate::renderer::CHUNK_BLOCKS`]); one chunk therefore spans
    /// `CHUNK_BLOCKS * voxels_per_block` voxels per axis. `chunk_coord` is that
    /// cell's integer coordinate, so the chunk covers the **half-open** absolute
    /// voxel box
    /// `[chunk_coord * chunk_extent_voxels, (chunk_coord + 1) * chunk_extent_voxels)`
    /// per axis. Boundary ownership is `floor(world_position / chunk_extent_voxels)`:
    /// because every resolved voxel centre sits at an `n + 0.5` position and chunk
    /// boundaries fall on integer multiples of `chunk_extent_voxels`, the `floor`
    /// is never ambiguous and every voxel lands in **exactly one** chunk.
    ///
    /// The returned grid's `dimensions` are one chunk's voxel extent
    /// (`chunk_extent_voxels³`); the occupied voxels keep their **absolute**
    /// composite `world_position` (they are NOT rebased to the chunk's local origin
    /// — that, like the recentre removal, is a later step). An empty chunk (no leaf
    /// overlaps it) returns an empty grid; it never panics.
    ///
    /// `voxels_per_block` is the application density (ADR 0001). `lod` is the parked
    /// level-of-detail seam (ADR 0002 Decision 2): it is **always `0`** for now and
    /// is asserted so; it exists from day one so a future down-sampling LOD level is
    /// a behavioural change, not a signature break.
    pub fn resolve_chunk(
        &self,
        chunk_coord: [i32; 3],
        voxels_per_block: u32,
        lod: u32,
    ) -> VoxelGrid {
        // The bare `resolve_chunk` keeps the S0 contract: ABSOLUTE composite
        // positions (floating origin `[0, 0, 0]`). The live render path uses
        // `resolve_chunk_rebased` with the floating origin = the composite recentre.
        self.resolve_chunk_rebased(chunk_coord, voxels_per_block, lod, [0, 0, 0])
    }

    /// Resolve one chunk like [`resolve_chunk`](Self::resolve_chunk), but store each
    /// voxel's position **rebased to `floating_origin_voxels`** (ADR 0002 Decision 2,
    /// camera-relative / origin-rebased rendering — S4b).
    ///
    /// The stored `world_position` is `absolute_composite_position −
    /// floating_origin_voxels`, with the subtraction performed in **i64 before the
    /// f32 downcast**, so the rendered f32 magnitude stays small no matter how far the
    /// chunk sits from the absolute origin. The chunk-membership clip is still decided
    /// in **absolute** space (f64), so a far chunk's boundary voxels are never
    /// misclassified by f32 rounding.
    ///
    /// `floating_origin_voxels = [0, 0, 0]` reproduces [`resolve_chunk`] exactly. The
    /// live render passes [`recentre_voxels_for_resolve`](Self::recentre_voxels_for_resolve)
    /// (the composite recentre, an integer-block-aligned point), so for a near scene
    /// the result is bit-identical to today's recentred [`resolve_region`] while a
    /// far-placed scene renders with no f32 jitter (the S1 speckle fix).
    pub fn resolve_chunk_rebased(
        &self,
        chunk_coord: [i32; 3],
        voxels_per_block: u32,
        lod: u32,
        floating_origin_voxels: [i64; 3],
    ) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        // Chunk extent fits i64 trivially; the chunk's absolute-voxel corners can be
        // large (a far-placed chunk), so they are computed in i64 (S4a).
        let chunk_extent_voxels = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

        // The chunk's half-open absolute-voxel box `[min, max)` per axis.
        let chunk_min_voxels = [
            chunk_coord[0] as i64 * chunk_extent_voxels,
            chunk_coord[1] as i64 * chunk_extent_voxels,
            chunk_coord[2] as i64 * chunk_extent_voxels,
        ];
        let chunk_max_voxels = [
            chunk_min_voxels[0] + chunk_extent_voxels,
            chunk_min_voxels[1] + chunk_extent_voxels,
            chunk_min_voxels[2] + chunk_extent_voxels,
        ];

        // The chunk grid is one chunk's voxel extent. (The voxels keep ABSOLUTE
        // positions inside it; `dimensions` describes the chunk's size, not the
        // window of absolute space the positions live in — the consumers that need
        // chunk-local coordinates rebase later, S4.)
        let chunk_dimensions = [
            chunk_extent_voxels as u32,
            chunk_extent_voxels as u32,
            chunk_extent_voxels as u32,
        ];
        let mut output = VoxelGrid::new(chunk_dimensions);

        // Each leaf is resolved into its own origin-centred local grid (exactly as
        // `resolve_region` does), translated by its WORLD offset × density — but
        // WITHOUT the composite recentre, so positions are absolute. We then keep
        // only the voxels whose absolute centre falls in this chunk's box.
        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let density = voxels_per_block.max(1) as i64;
        let chunk_box = VoxelAabb::new(chunk_min_voxels, chunk_max_voxels);
        self.for_each_leaf(&mut |world_offset, content| {
            // Issue #27 S3 optimisation: skip a leaf whose world-AABB doesn't touch
            // this chunk, so resolving one chunk costs ~the leaves that overlap it
            // (not the whole tree). This is BIT-IDENTICAL to stamping-then-clipping:
            // the leaf's AABB `[off·d − grid/2, off·d + grid/2)` is the exact span of
            // its voxel centres, and `stamp_producer_into_chunk` keeps only centres
            // inside `[chunk_min, chunk_max)`; if those two half-open boxes don't
            // intersect, the stamp would have clipped EVERY voxel anyway. A
            // region-spanning leaf (a Part, `leaf_size_blocks` → `None`) has no
            // localisable AABB, so it is never skipped (it may emit anywhere).
            // Lattice snap (issue #30): the producer grid is shifted onto whole
            // blocks before placement, so the leaf's absolute AABB is
            // `[off·d − grid/2 + shift, …)` — equivalently the whole-block range
            // `[(off − floor(size/2))·d, …)`. Both the skip-AABB and the stamp
            // translation use the SAME shift so they stay consistent.
            let lattice_shift = match leaf_size_blocks(content, voxels_per_block) {
                Some(size_blocks) => leaf_lattice_shift_voxels(size_blocks, voxels_per_block),
                None => [0i64; 3],
            };
            if let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) {
                let mut leaf_min = [0i64; 3];
                let mut leaf_max = [0i64; 3];
                for axis in 0..3 {
                    let grid = size_blocks[axis] as i64 * density;
                    let centre = world_offset[axis] * density;
                    leaf_min[axis] = centre - grid / 2 + lattice_shift[axis];
                    leaf_max[axis] = leaf_min[axis] + grid;
                }
                if !VoxelAabb::new(leaf_min, leaf_max).intersects(&chunk_box) {
                    return;
                }
            }
            let translation_voxels = [
                world_offset[0] * voxels_per_block as i64 + lattice_shift[0],
                world_offset[1] * voxels_per_block as i64 + lattice_shift[1],
                world_offset[2] * voxels_per_block as i64 + lattice_shift[2],
            ];
            let (material_override, producer): (Option<u16>, Box<dyn VoxelProducer>) = match content
            {
                NodeContent::Tool { shape, material } => {
                    (material_id_for(*material), Box::new(*shape))
                }
                NodeContent::Part(Part::DebugClouds { seed }) => (
                    None,
                    Box::new(DebugCloudField {
                        dimensions: region_dimensions,
                        voxels_per_block,
                        seed: *seed,
                    }),
                ),
                NodeContent::Group(_) | NodeContent::Instance(_) => return,
            };
            stamp_producer_into_chunk(
                &mut output,
                region_dimensions,
                translation_voxels,
                floating_origin_voxels,
                material_override,
                producer.as_ref(),
                chunk_min_voxels,
                chunk_max_voxels,
            );
        });

        output
    }

    /// Resolve the scene's whole region by **decomposing it into chunks** and
    /// merging them back into one grid, in **absolute (non-recentred) coordinates**.
    ///
    /// This loops over every chunk coordinate covering the composite AABB, calls
    /// [`resolve_chunk`](Self::resolve_chunk) for each, and unions the results. It
    /// proves the chunk decomposition reconstructs the whole scene; it is **not**
    /// wired into rendering (the render path stays on [`resolve_region`], which
    /// recentres — see issue #27 S0). The returned grid is sized to the full
    /// composite extent and its voxels keep their absolute composite positions;
    /// compared against [`resolve_region`]'s output it differs only by the
    /// recentre offset.
    pub fn resolve_region_via_chunks(&self, voxels_per_block: u32, lod: u32) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut output = VoxelGrid::new(region_dimensions);

        let Some(chunk_range) = self.covering_chunk_range(voxels_per_block) else {
            // No leaf has an intrinsic size (a Part-only scene with no Tools): no
            // composite AABB, so there are no chunks to resolve.
            return output;
        };
        let (min_chunk, max_chunk) = chunk_range;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk =
                        self.resolve_chunk([chunk_x, chunk_y, chunk_z], voxels_per_block, lod);
                    output.occupied.extend(chunk.occupied);
                }
            }
        }
        output
    }

    /// The recentre offset (in voxels) that [`resolve_region`] subtracts from every
    /// voxel to centre the composite on the origin. The chunk path does NOT apply
    /// this, so it is the exact translation between the two frames:
    /// `resolve_region.world_position == chunk_path.world_position − recentre_voxels`.
    /// Exposed (crate-internal) so the S0 equivalence tests can normalise one frame
    /// to the other. `[0, 0, 0]` for a scene with no intrinsic-size leaf.
    #[cfg(test)]
    pub(crate) fn recentre_voxels(&self, voxels_per_block: u32) -> [i64; 3] {
        self.recentre_voxels_for_resolve(voxels_per_block)
    }

    /// The recentre offset (in voxels) that [`resolve_region`] subtracts from every
    /// voxel to centre the composite on the origin (issue #27 S2). This is the
    /// SAME computation [`resolve_region`] inlines; the chunk cache
    /// ([`crate::chunk_cache::ChunkResolveCache::resolve_region`]) calls it to apply
    /// the identical offset when reassembling the recentred monolithic grid from
    /// absolute per-chunk pieces, so the assembled output is bit-identical. `[0, 0,
    /// 0]` for a scene with no intrinsic-size leaf.
    pub fn recentre_voxels_for_resolve(&self, voxels_per_block: u32) -> [i64; 3] {
        match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => [
                ((min_corner[0] + max_corner[0]) * voxels_per_block as i64) / 2,
                ((min_corner[1] + max_corner[1]) * voxels_per_block as i64) / 2,
                ((min_corner[2] + max_corner[2]) * voxels_per_block as i64) / 2,
            ],
            None => [0i64; 3],
        }
    }

    /// The full composite extent in voxels (`size_blocks × density`) — the size the
    /// whole-region grids ([`resolve_region`], [`resolve_region_via_chunks`]) and
    /// the per-leaf local grids are seeded with. The chunk cache (issue #27 S2)
    /// seeds its reassembled grid to the same dimensions.
    ///
    /// **This IS the size the assembled render grid takes** for a chunkable scene
    /// (one with at least one intrinsic-size leaf): both [`resolve_region`] and
    /// [`crate::chunk_cache::ChunkResolveCache::resolve_region`] size their output
    /// to exactly this value. So the camera / gizmo / lattice / floor-grid /
    /// layer-scrubber can read the region dimensions straight from the scene
    /// (issue #20 S6c) instead of reaching into the assembled `VoxelGrid` —
    /// the two are provably identical (asserted in
    /// `placed_region_dimensions_equals_assembled_grid` below). `pub` so the `shot`
    /// binary can do the same substitution.
    ///
    /// **Caveat — a Part-only scene** (no intrinsic-size leaf, e.g. a lone
    /// debug-cloud field) returns `[0, 0, 0]` here because it has no composite
    /// extent; such a scene is resolved through the *explicit-region* monolithic
    /// path (sized to the caller's chosen region, not this), so a consumer of a
    /// Part-only scene must use that explicit region — not this — as its dimensions.
    pub fn placed_region_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        let region = self.full_extent_blocks(voxels_per_block);
        [
            region.size_blocks[0] * voxels_per_block,
            region.size_blocks[1] * voxels_per_block,
            region.size_blocks[2] * voxels_per_block,
        ]
    }

    /// Whether the scene has at least one intrinsic-size leaf (a Tool), so it has a
    /// composite AABB that the chunked resolve ([`crate::chunk_cache`]) can cover.
    /// `false` for a Part-only scene (e.g. a lone debug-cloud field), which has no
    /// AABB of its own and must be resolved through the explicit-region monolithic
    /// path instead. Public so the `shot` binary can pick the right resolve path
    /// (issue #27 S2).
    pub fn has_chunkable_extent(&self, voxels_per_block: u32) -> bool {
        self.covering_chunk_range(voxels_per_block).is_some()
    }

    /// The composite occupied AABB in **absolute voxel** space, as the producers
    /// actually emit it. Each leaf producer fills its own grid (`size_blocks ×
    /// density` voxels) **centred on the origin**, then placed at `world_offset ×
    /// density`; so a leaf occupies the half-open absolute-voxel box
    /// `[world_offset·d − grid/2, world_offset·d + grid/2)` per axis, where
    /// `grid = size_blocks · d`. The composite is the union of those boxes.
    ///
    /// This is the **producer-true** frame the chunk ownership (`floor(position /
    /// chunk_extent)`) lives in — distinct from [`placed_extent_blocks`], whose
    /// `floor(size/2)`-per-block split is shifted by up to half a block from the
    /// producer's centre for **odd** block sizes (e.g. a 5- or 1-block axis). The
    /// chunk path must cover the producer-true voxel extent (not the block-AABB
    /// frame) or it drops the voxels living in the chunks the block-AABB misses —
    /// the cause of the flat-shape (Y = 1 block) half-drop. `None` when no leaf has
    /// an intrinsic size.
    fn placed_extent_voxels(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let density = voxels_per_block.max(1) as i64;
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset, content| {
            let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                return;
            };
            any = true;
            let lattice_shift = leaf_lattice_shift_voxels(size_blocks, voxels_per_block);
            for axis in 0..3 {
                let grid = size_blocks[axis] as i64 * density;
                // The producer centres its grid on the origin (voxel centres at
                // `idx + 0.5 − grid/2`), so its occupied voxel span is `[−grid/2,
                // grid/2)`; placed at `world_offset·d` and snapped onto the block
                // lattice (issue #30) it spans `[off·d − grid/2 + shift, …)`, i.e.
                // the whole-block range `[(off − floor(size/2))·d, …)`. Using the
                // SAME shift the resolve path applies keeps every frame consistent.
                let centre = world_offset[axis] * density;
                let low = centre - grid / 2 + lattice_shift[axis];
                let high = low + grid;
                min_corner[axis] = min_corner[axis].min(low);
                max_corner[axis] = max_corner[axis].max(high);
            }
        });
        any.then_some((min_corner, max_corner))
    }

    /// The inclusive range of chunk coordinates `[min_chunk, max_chunk]` whose
    /// half-open boxes cover the composite occupied AABB in **absolute** voxel
    /// space. `None` when no leaf has an intrinsic size (no AABB to cover).
    /// `pub(crate)` so the chunk cache (issue #27 S2) iterates the covering chunks
    /// for reassembly.
    ///
    /// Derived from [`placed_extent_voxels`](Self::placed_extent_voxels) — the
    /// producer-true voxel frame — so it covers every chunk a voxel can land in,
    /// including the chunks an odd/flat block size straddles (which the block-AABB
    /// frame would miss).
    pub(crate) fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        let (min_voxel_corner, max_voxel_corner) = self.placed_extent_voxels(voxels_per_block)?;
        // The voxel corners are i64 (a far-placed leaf), but the chunk extent is
        // small; the block→chunk division therefore happens in i64 and the QUOTIENT
        // (the chunk coordinate) narrows to i32 safely — for offsets up to ±10⁹
        // blocks at density 16 a chunk coord is ≤ ±2.5×10⁸, well inside i32 (S4a).
        let chunk_extent_voxels = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

        let mut min_chunk = [0i32; 3];
        let mut max_chunk = [0i32; 3];
        for axis in 0..3 {
            let min_voxel = min_voxel_corner[axis];
            // The AABB is the half-open box `[min, max)`; its last occupied voxel
            // centre is at `max_voxel - 1 + 0.5`, so the highest chunk is the one
            // owning `max_voxel - 1`.
            let max_voxel = max_voxel_corner[axis];
            min_chunk[axis] = narrow_chunk_coord(min_voxel.div_euclid(chunk_extent_voxels));
            max_chunk[axis] = narrow_chunk_coord((max_voxel - 1).div_euclid(chunk_extent_voxels));
        }
        Some((min_chunk, max_chunk))
    }

    /// Build a [`LeafSpatialIndex`](crate::spatial_index::LeafSpatialIndex) over the
    /// scene's leaves at `voxels_per_block` (issue #27 S3).
    ///
    /// One `for_each_leaf` walk records, per visible leaf, its world-AABB in the
    /// **absolute-voxel producer-true frame** — the SAME frame
    /// [`resolve_chunk`](Self::resolve_chunk) and [`placed_extent_voxels`] use, so a
    /// chunk derived from a leaf's index AABB is exactly a chunk that leaf's voxels
    /// can land in. A leaf with an intrinsic size (a Tool) gets a concrete box
    /// `[off·d − grid/2, off·d + grid/2)`; a region-spanning leaf (a Part, no
    /// intrinsic size) gets an empty box and a
    /// [`RegionSpanning`](crate::spatial_index::LeafFingerprint::RegionSpanning)
    /// fingerprint (it cannot be chunk-localised; an edit touching it forces a
    /// wholesale clear).
    ///
    /// By construction the index's entries ARE the leaves `for_each_leaf` yields, so
    /// a query against the index returns the same leaf set as the full walk filtered
    /// by AABB (proven in the spatial-index tests).
    pub fn build_leaf_spatial_index(&self, voxels_per_block: u32) -> LeafSpatialIndex {
        let density = voxels_per_block.max(1) as i64;
        let mut entries: Vec<LeafEntry> = Vec::new();
        let mut has_region_spanning_leaf = false;
        self.for_each_leaf(&mut |world_offset, content| {
            match leaf_size_blocks(content, voxels_per_block) {
                Some(size_blocks) => {
                    // The producer centres its `size·d` grid on the origin, then it
                    // is placed at `world_offset·d`; its occupied voxel span per axis
                    // is `[off·d − grid/2, off·d + grid/2)` — the producer-true
                    // half-extent (`grid/2 = size·d/2`), identical to
                    // `placed_extent_voxels`. Absolute voxels are i64 (S4a).
                    let mut min = [0i64; 3];
                    let mut max = [0i64; 3];
                    let lattice_shift = leaf_lattice_shift_voxels(size_blocks, voxels_per_block);
                    for axis in 0..3 {
                        let grid = size_blocks[axis] as i64 * density;
                        let centre = world_offset[axis] * density;
                        min[axis] = centre - grid / 2 + lattice_shift[axis];
                        max[axis] = min[axis] + grid;
                    }
                    entries.push(LeafEntry {
                        world_aabb: VoxelAabb::new(min, max),
                        fingerprint: LeafFingerprint::Bounded(leaf_content_fingerprint(
                            world_offset,
                            content,
                        )),
                    });
                }
                None => {
                    // A region-spanning leaf (a Part): no intrinsic box. Record it
                    // with an empty AABB + a region-spanning fingerprint so an edit
                    // touching it forces a wholesale clear (it can't be localised).
                    has_region_spanning_leaf = true;
                    entries.push(LeafEntry {
                        world_aabb: VoxelAabb::new([0; 3], [0; 3]),
                        fingerprint: LeafFingerprint::RegionSpanning(leaf_content_fingerprint(
                            world_offset,
                            content,
                        )),
                    });
                }
            }
        });
        LeafSpatialIndex {
            entries,
            voxels_per_block,
            has_region_spanning_leaf,
        }
    }
}

/// A content fingerprint for a leaf: the bytes (placement + content) that affect the
/// voxels it resolves to. Two leaves with the same fingerprint at the same world
/// position resolve to the same voxels, so the edit diff
/// ([`LeafSpatialIndex::edit_aabb_since`](crate::spatial_index::LeafSpatialIndex::edit_aabb_since))
/// treats them as unchanged. `world_offset` is included so a moved Tool whose box
/// happens to coincide with another's still reads as distinct.
/// Narrow an `i64` chunk coordinate to `i32` (the cache-key / chunk-index width).
///
/// **Audit (S4a, ADR 0002 Decision 2):** the absolute-VOXEL math is i64 so a
/// far-placed node composes without overflow, but the CHUNK coordinate (= voxel /
/// chunk_extent) is much smaller — at density 16 / `CHUNK_BLOCKS = 4` the extent is
/// 64 voxels, so a block offset of ±10⁹ yields a chunk coord of only ±2.5×10⁸,
/// comfortably inside i32 (±2.1×10⁹). Keeping the chunk coord / cache key i32 is
/// therefore safe and avoids widening the whole chunk index. A coordinate that
/// would not fit i32 means a block offset past ~±8×10⁹ — beyond the supported
/// range — and is clamped (debug-asserted) rather than silently wrapping.
fn narrow_chunk_coord(chunk_coord: i64) -> i32 {
    debug_assert!(
        chunk_coord >= i32::MIN as i64 && chunk_coord <= i32::MAX as i64,
        "chunk coordinate {chunk_coord} overflows i32 — block offset is past the \
         supported ±~8×10⁹-block range (S4a)"
    );
    chunk_coord.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn leaf_content_fingerprint(world_offset: [i64; 3], content: &NodeContent) -> String {
    match content {
        NodeContent::Tool { shape, material } => {
            format!("Tool@{world_offset:?}:{shape:?}:{material:?}")
        }
        NodeContent::Part(part) => format!("Part@{world_offset:?}:{part:?}"),
        // for_each_leaf only ever yields leaf content (Tool / Part); Group / Instance
        // are interior and never reach a visitor. Fingerprint defensively anyway.
        NodeContent::Group(_) => format!("Group@{world_offset:?}"),
        NodeContent::Instance(def_id) => format!("Instance@{world_offset:?}:{def_id:?}"),
    }
}

/// Depth-first worker for [`Scene::tree_rows`]: append `(path, depth)` for each
/// node in `nodes`, descending into Group children (a Group's children follow it
/// at `depth + 1`). `prefix` is the path of the assembly that owns `nodes`.
fn collect_tree_rows(
    nodes: &[Node],
    prefix: &mut Vec<usize>,
    depth: usize,
    rows: &mut Vec<(NodePath, usize)>,
) {
    for (index, node) in nodes.iter().enumerate() {
        prefix.push(index);
        rows.push((NodePath::from_indices(prefix.clone()), depth));
        if let NodeContent::Group(children) = &node.content {
            collect_tree_rows(children, prefix, depth + 1, rows);
        }
        prefix.pop();
    }
}

/// The whole-block extent of a leaf node's producer, or `None` for a non-leaf /
/// not-yet-implemented content kind.
fn leaf_size_blocks(content: &NodeContent, voxels_per_block: u32) -> Option<[u32; 3]> {
    let density = voxels_per_block.max(1);
    match content {
        NodeContent::Tool { shape, .. } => Some(shape.size_blocks),
        // The cloud field has no intrinsic size; today it adopts the shape's grid
        // dimensions, so a step-1 Part-only scene has no extent of its own. The
        // call sites that resolve a Part always pass the region explicitly, so
        // this path is unused by them; report whole blocks for completeness.
        NodeContent::Part(Part::DebugClouds { .. }) => {
            // A Part stamped at the app density occupies `dimensions / density`
            // blocks; with no stored body in step 1 it has no size. Returning
            // `None` keeps `full_extent_blocks` deferring to the next leaf.
            let _ = density;
            None
        }
        NodeContent::Group(_) | NodeContent::Instance(_) => None,
    }
}

/// Per-axis voxel shift that snaps a leaf's producer-emitted grid onto the **global
/// block lattice** (issue #30).
///
/// The producer ([`SdfShape::resolve`]) centres its `grid = size·d` voxels on the
/// origin: centres at `idx + 0.5 − grid/2`, so its occupied span is `[−grid/2,
/// grid/2)`. For an **odd** block count `grid/2 = size·d/2` is an odd multiple of
/// `d/2`, so that span straddles a block boundary by **half a block** — a 1-block
/// box at offset 0 lands on `[−d/2, d/2)` instead of a whole block cell. The grids
/// (#29) read this producer-true frame, so the half-block straddle makes them
/// mis-read.
///
/// We snap by translating the leaf so it occupies the whole-block range
/// `[off − floor(size/2), off − floor(size/2) + size)` **blocks** — exactly the
/// frame [`Scene::placed_extent_blocks`] already uses. The shift per axis is the
/// difference between the producer's half-extent and that block-floored half:
///
/// `shift = grid/2 − floor(size/2)·d = (size·d)/2 − floor(size/2)·d`
///
/// which is `0` for an even size and `d/2` for an odd one. Because `d/2` is an
/// **integer** number of voxels, every voxel centre keeps its `n + 0.5` fractional
/// part, so the chunk-storage / index-recovery arithmetic (which depends on that
/// fraction) is untouched. After this shift the producer-true voxel frame and the
/// block-AABB frame coincide, so the recentre, chunk ownership, spatial index and
/// per-object grids all agree and odd sizes are no longer off-centre.
fn leaf_lattice_shift_voxels(size_blocks: [u32; 3], voxels_per_block: u32) -> [i64; 3] {
    let density = voxels_per_block.max(1) as i64;
    let mut shift = [0i64; 3];
    for axis in 0..3 {
        let size = size_blocks[axis] as i64;
        shift[axis] = (size * density) / 2 - (size / 2) * density;
    }
    shift
}

/// Map a Tool's [`MaterialChoice`] to the `material_id` it stamps (ADR 0001 step 3
/// "Materials"). A Tool is single-material by nature: every voxel it emits takes
/// this one id, so distinct nodes render in distinct materials. Stone = 0,
/// Wood = 1, Plain = 2 (see [`MaterialChoice::material_id`]).
fn material_id_for(material: MaterialChoice) -> Option<u16> {
    Some(material.material_id())
}

/// Resolve `producer` into its own local grid (centred at the origin, as the
/// trait guarantees) and **stamp** it into `output`, translated by
/// `translation_voxels` (the node's placement minus the composite recentre, in
/// voxels).
///
/// When `translation_voxels` is zero and no material override applies, the stamp
/// is the identity: the producer's occupied set is moved into `output` unchanged
/// (the one-node, zero-offset path — guarantees a bit-for-bit match with the bare
/// producer). When `material_override` is `Some(id)`, every stamped voxel takes
/// that id (a Tool's single material); when `None`, each voxel keeps the material
/// the producer emitted (a Part's own per-voxel materials).
fn stamp_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    material_override: Option<u16>,
    producer: &dyn VoxelProducer,
) {
    // The producer sizes its own grid (`SdfShape::resolve` overwrites
    // `dimensions` to its own `size_blocks × density`, centred at the origin), so
    // the local grid need only seed the dimensions; the cloud field, which has no
    // intrinsic size, fills the region it is handed.
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local);

    let zero_offset = translation_voxels == [0, 0, 0];

    if zero_offset && material_override.is_none() {
        // Fast path / exact identity: no translation and no material rewrite, so
        // the local occupied set IS the output.
        if output.occupied.is_empty() {
            output.occupied = local.occupied;
            return;
        }
        output.occupied.extend(local.occupied);
        return;
    }

    // General stamp: translate each voxel into the composite (the producer's
    // origin-centred position plus the node's recentred placement) and, for a
    // Tool, overwrite its material id.
    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        if !zero_offset {
            voxel.world_position[0] += translation_voxels[0] as f32;
            voxel.world_position[1] += translation_voxels[1] as f32;
            voxel.world_position[2] += translation_voxels[2] as f32;
        }
        if let Some(id) = material_override {
            voxel.material_id = id;
        }
        output.occupied.push(voxel);
    }
}

/// Resolve `producer` into its own origin-centred local grid, translate it by
/// `translation_voxels` (the node's WORLD placement × density — **no recentre**),
/// and stamp only the voxels whose absolute centre falls in the half-open chunk
/// box `[chunk_min_voxels, chunk_max_voxels)` into `output`.
///
/// This is the chunk-scoped sibling of [`stamp_producer`]: same per-leaf
/// resolution, same material-override rule (a Tool overwrites every voxel's id;
/// `None` keeps the producer's own ids), but it (a) never recentres and (b)
/// clips each voxel to one chunk. Ownership is `floor(world_position /
/// chunk_extent_voxels)` per axis; since centres sit at `n + 0.5` and boundaries
/// at integer multiples of the chunk extent, each voxel lands in exactly one
/// chunk.
/// `floating_origin_voxels` is the **render floating origin** (ADR 0002 Decision 2,
/// camera-relative / origin-rebased rendering — S4b): the integer-voxel point the
/// rendered f32 frame is rebased around. The stored `world_position` is the voxel's
/// absolute composite position **minus the floating origin**, with the subtraction
/// done in **i64 BEFORE the f32 downcast** so the rendered f32 magnitude stays small
/// regardless of how far the chunk sits from the absolute origin (no far-lands
/// jitter). Pass `[0, 0, 0]` to store true absolute positions (the chunk-cache
/// parity tests / `.vox`-style consumers). The chunk-membership clip is computed in
/// **f64 absolute** space (independent of the rebase) so a far chunk's boundary
/// voxels are never misclassified by f32 rounding.
#[allow(clippy::too_many_arguments)]
fn stamp_producer_into_chunk(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    floating_origin_voxels: [i64; 3],
    material_override: Option<u16>,
    producer: &dyn VoxelProducer,
    chunk_min_voxels: [i64; 3],
    chunk_max_voxels: [i64; 3],
) {
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local);

    // The voxel's chunk-local placement, rebased to the floating origin in i64
    // FIRST so the f32 add never sees a large magnitude. For the live render the
    // floating origin equals the composite recentre, so for a near scene this is
    // EXACTLY the small `world_offset·d − recentre` translation `resolve_region`
    // adds in f32 today — bit-identical framing — while a far chunk no longer loses
    // the voxel-centre `.5` to f32 rounding at ~1e6 magnitude (the S1 speckle).
    let rebased_translation = [
        (translation_voxels[0] - floating_origin_voxels[0]) as f32,
        (translation_voxels[1] - floating_origin_voxels[1]) as f32,
        (translation_voxels[2] - floating_origin_voxels[2]) as f32,
    ];

    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        // Chunk membership is decided on the ABSOLUTE centre, computed in f64 so a
        // far chunk's boundary voxels are classified at full precision (the half-open
        // box `[min, max)` with centres at `n + 0.5` → each voxel owns exactly one
        // chunk). f64 carries far more than the ~21 bits an f32 keeps at 1.6M voxels.
        let in_chunk = (0..3).all(|axis| {
            let absolute = voxel.world_position[axis] as f64 + translation_voxels[axis] as f64;
            absolute >= chunk_min_voxels[axis] as f64 && absolute < chunk_max_voxels[axis] as f64
        });
        if !in_chunk {
            continue;
        }

        // Store the rebased (origin-relative) f32 position.
        voxel.world_position[0] += rebased_translation[0];
        voxel.world_position[1] += rebased_translation[1];
        voxel.world_position[2] += rebased_translation[2];

        if let Some(id) = material_override {
            voxel.material_id = id;
        }
        output.occupied.push(voxel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ShapeKind;

    /// The identical-behaviour guarantee (ADR 0001 step 1): a one-node Tool scene
    /// resolved over the node's full extent yields the SAME occupied count as
    /// calling `SdfShape::resolve` directly — and the same grid dimensions.
    #[test]
    fn tool_scene_matches_bare_producer() {
        let geometry = GeometryParams {
            shape: ShapeKind::Sphere,
            size_blocks: [6, 6, 6],
            voxels_per_block: 16,
            wall_blocks: 1,
        };

        // Bare producer (today's path).
        let shape = SdfShape::from_geometry(geometry);
        let mut bare = VoxelGrid::new(shape.grid_dimensions());
        shape.resolve(&mut bare);

        // Through the scene.
        let scene = Scene::from_geometry(geometry, MaterialChoice::Stone);
        let region = scene.full_extent_blocks(geometry.voxels_per_block);
        let resolved = scene.resolve_region(region, geometry.voxels_per_block, 0);

        assert_eq!(
            resolved.dimensions, bare.dimensions,
            "scene grid dimensions must match the bare producer"
        );
        assert_eq!(
            resolved.occupied_count(),
            bare.occupied_count(),
            "scene occupied count must match the bare producer"
        );
    }

    /// **Issue #20 S6c-1 equivalence proof.** `placed_region_dimensions(density)`
    /// is exactly the size the assembled render grid takes — both the monolithic
    /// [`resolve_region`] and the chunk-cache reassembly seed their output to it. So
    /// the camera / gizmo / lattice / floor-grid / layer-scrubber may read the
    /// region dimensions from the SCENE rather than from the assembled `VoxelGrid`,
    /// with zero behavioural change. This pins that substitution across every
    /// representative scene (all SDF shapes, flat/odd sizes, a placed multi-node
    /// scene, and an instanced village) for BOTH resolve paths.
    #[test]
    fn placed_region_dimensions_equals_assembled_grid() {
        use crate::chunk_cache::ChunkResolveCache;

        let assert_equal = |scene: &Scene, vpb: u32, label: &str| {
            let from_scene = scene.placed_region_dimensions(vpb);

            // (1) The monolithic resolve_region (the initial-resolve path).
            let region = scene.full_extent_blocks(vpb);
            let monolithic = scene.resolve_region(region, vpb, 0);
            assert_eq!(
                from_scene, monolithic.dimensions,
                "[{label}] placed_region_dimensions must equal the monolithic assembled grid"
            );

            // (2) The chunk-cache reassembly (the live rebuild path).
            let mut cache = ChunkResolveCache::new();
            let assembled = cache.resolve_region(scene, vpb, 0);
            assert_eq!(
                from_scene, assembled.dimensions,
                "[{label}] placed_region_dimensions must equal the cache-assembled grid"
            );
        };

        // All SDF shapes at the app default density.
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = Scene::from_geometry(
                GeometryParams { shape: kind, size_blocks: [5, 5, 5], voxels_per_block: 16, wall_blocks: 1 },
                MaterialChoice::Stone,
            );
            assert_equal(&scene, 16, &format!("{kind:?}"));
        }

        // Flat / odd sizes (the 5×1×5 app default and friends), several densities.
        for vpb in [1u32, 8, 16] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams { shape: ShapeKind::Cylinder, size_blocks: size, voxels_per_block: vpb, wall_blocks: 1 },
                    MaterialChoice::Stone,
                );
                assert_equal(&scene, vpb, &format!("cylinder {size:?}@{vpb}"));
            }
        }

        // A placed multi-node scene (sphere at origin + box +8X + torus +6Z).
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape { kind, size_blocks: [5, 5, 5], voxels_per_block: 16, wall_blocks: 1 };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let demo_scene = Scene {
            nodes: vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        assert_equal(&demo_scene, 16, "demo-scene");

        // An instanced village (one house definition placed by four instances).
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape { kind, size_blocks: size, voxels_per_block: 16, wall_blocks: 1 };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let house = AssemblyDef {
            id: house_def_id,
            name: "House".to_string(),
            children: vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform.offset_blocks = offset;
            node
        };
        let village = Scene {
            nodes: vec![
                instance("House 1", [0, 0, 0]),
                instance("House 2", [6, 0, 0]),
                instance("House 3", [12, 0, 0]),
                instance("House 4", [18, 0, 0]),
            ],
            definitions: vec![house],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        assert_equal(&village, 16, "demo-village");
    }

    /// The same guarantee for a Part (the debug cloud field): a one-node Part
    /// scene matches `DebugCloudField::resolve` at the same dimensions. Step 2
    /// builds the Part node directly (the `debug_clouds` selector is gone).
    #[test]
    fn part_scene_matches_bare_cloud_field() {
        let size_blocks = [4u32, 4, 4];
        let voxels_per_block = 16u32;
        let dimensions = [
            size_blocks[0] * voxels_per_block,
            size_blocks[1] * voxels_per_block,
            size_blocks[2] * voxels_per_block,
        ];
        let bare_field = DebugCloudField {
            dimensions,
            voxels_per_block,
            seed: 0,
        };
        let mut bare = VoxelGrid::new(dimensions);
        bare_field.resolve(&mut bare);

        let scene =
            Scene::single_node(Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })));
        let region = RegionBlocks::new(size_blocks);
        let resolved = scene.resolve_region(region, voxels_per_block, 0);

        assert_eq!(resolved.dimensions, bare.dimensions);
        assert_eq!(resolved.occupied_count(), bare.occupied_count());
    }

    /// ADR 0001 step 2: several leaf nodes composite into one region under union.
    /// A 2-node scene (a sphere Tool + a box Tool, both centred at origin) yields
    /// the SET-UNION of their occupied voxels: the union count is at least each
    /// node alone, and exactly equals the union of the two single-node sets.
    #[test]
    fn two_node_scene_resolves_to_union() {
        let voxels_per_block = 12u32;
        let region = RegionBlocks::new([6, 6, 6]);

        let sphere = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Sphere,
                    size_blocks: [6, 6, 6],
                    voxels_per_block,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        // A full-extent box: its corners poke outside the inscribed sphere, so the
        // union is strictly larger than the sphere alone (a real composite).
        let cube = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [6, 6, 6],
                    voxels_per_block,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Wood,
            },
        );

        // Each node resolved alone.
        let sphere_only = Scene::single_node(sphere.clone())
            .resolve_region(region, voxels_per_block, 0);
        let cube_only =
            Scene::single_node(cube.clone()).resolve_region(region, voxels_per_block, 0);

        // Both nodes composited.
        let scene = Scene {
            nodes: vec![sphere, cube],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let union = scene.resolve_region(region, voxels_per_block, 0);

        // The expected set-union of the two single-node occupied sets, keyed by
        // integer voxel position (the producers emit voxel-centre world positions).
        use std::collections::HashSet;
        let key = |grid: &VoxelGrid| -> HashSet<[i64; 3]> {
            grid.occupied
                .iter()
                .map(|voxel| {
                    [
                        voxel.world_position[0].round() as i64,
                        voxel.world_position[1].round() as i64,
                        voxel.world_position[2].round() as i64,
                    ]
                })
                .collect()
        };
        let sphere_set = key(&sphere_only);
        let cube_set = key(&cube_only);
        let union_set = key(&union);
        let expected: HashSet<[i64; 3]> = sphere_set.union(&cube_set).copied().collect();

        // Union is at least as occupied as either node alone …
        assert!(union_set.len() >= sphere_set.len());
        assert!(union_set.len() >= cube_set.len());
        // … and equals the set-union exactly (the box pokes outside the sphere, so
        // the union is strictly larger than the sphere alone — a real composite).
        assert_eq!(union_set, expected);
        assert!(union_set.len() > sphere_set.len());
    }

    /// ADR 0001 step 3 (per-voxel material): a Tool with `MaterialChoice::Wood`
    /// stamps voxels whose `material_id` equals the Wood id (1) — every voxel it
    /// emits carries the Tool's single material, so distinct nodes are distinct.
    #[test]
    fn wood_tool_stamps_wood_material_id() {
        let voxels_per_block = 8u32;
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block,
            wall_blocks: 1,
        };
        let scene = Scene::single_node(Node::new(
            "Wood box",
            NodeContent::Tool { shape, material: MaterialChoice::Wood },
        ));
        let grid = scene.resolve_region(RegionBlocks::new([2, 2, 2]), voxels_per_block, 0);
        let wood_id = MaterialChoice::Wood.material_id();
        assert!(grid.occupied_count() > 0, "the box must emit voxels");
        assert!(
            grid.occupied.iter().all(|voxel| voxel.material_id == wood_id),
            "every voxel a Wood Tool stamps must carry the Wood material id"
        );
    }

    /// ADR 0001 step 3 (per-voxel material): a 2-Tool scene (Stone + Wood, placed
    /// disjointly) yields BOTH material ids present — proving the per-voxel id
    /// travels through compositing so the two nodes render in distinct materials.
    #[test]
    fn two_material_scene_has_both_material_ids() {
        let voxels_per_block = 8u32;
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut stone = Node::new("Stone", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        stone.transform.offset_blocks = [0, 0, 0];
        let mut wood = Node::new("Wood", NodeContent::Tool { shape: base, material: MaterialChoice::Wood });
        wood.transform.offset_blocks = [5, 0, 0];
        let scene = Scene {
            nodes: vec![stone, wood],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        assert_ne!(stone_id, wood_id, "Stone and Wood must map to distinct ids");
        assert!(
            grid.occupied.iter().any(|voxel| voxel.material_id == stone_id),
            "the Stone node's voxels must carry the Stone id"
        );
        assert!(
            grid.occupied.iter().any(|voxel| voxel.material_id == wood_id),
            "the Wood node's voxels must carry the Wood id"
        );
    }

    /// A hidden node contributes nothing.
    #[test]
    fn hidden_node_stamps_nothing() {
        let mut node = Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [2, 2, 2],
                    voxels_per_block: 8,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        node.visible = false;
        let scene = Scene::single_node(node);
        let resolved = scene.resolve_region(RegionBlocks::new([2, 2, 2]), 8, 0);
        assert_eq!(resolved.occupied_count(), 0);
    }

    /// A box Tool sized to fill a single block (so the whole block of voxels is
    /// occupied), at the given block offset along X, in a wide region. Returns the
    /// set of occupied voxel positions keyed to integer coordinates.
    fn boxed_block_positions(offset_x: i64, voxels_per_block: u32) -> std::collections::HashSet<[i64; 3]> {
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform.offset_blocks = [offset_x, 0, 0];
        // A region wide enough to hold the offset box without clipping.
        let region = RegionBlocks::new([8, 1, 1]);
        let grid = Scene::single_node(node).resolve_region(region, voxels_per_block, 0);
        grid.occupied
            .iter()
            .map(|voxel| {
                [
                    voxel.world_position[0].round() as i64,
                    voxel.world_position[1].round() as i64,
                    voxel.world_position[2].round() as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 3 (a): a node with `offset_blocks = [N, 0, 0]` places its
    /// voxels shifted by exactly `N × voxels_per_block` in X versus offset 0.
    ///
    /// A two-node scene (a 1-block box at offset 0 and an identical box at offset
    /// N, far enough apart to be disjoint) shares ONE composite recentre, so the
    /// only difference between the two boxes' positions is the N-block placement.
    /// The occupied set splits into two equal clusters whose X-spans are exactly
    /// `N × voxels_per_block` apart; shifting one cluster by that amount reproduces
    /// the other.
    #[test]
    fn offset_node_shifts_voxels_by_blocks_times_density() {
        let voxels_per_block = 8u32;
        let n = 5i64; // 5 blocks apart: a 1-block box leaves a 4-block gap (disjoint).
        let region = RegionBlocks::new([8, 1, 1]);
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut at_zero = Node::new("A", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_zero.transform.offset_blocks = [0, 0, 0];
        let mut at_n = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_n.transform.offset_blocks = [n, 0, 0];

        let scene = Scene {
            nodes: vec![at_zero, at_n],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        // Key each voxel by its EXACT world position (the producers emit voxel-
        // centre positions; the placement is an exact integer-voxel translation, so
        // float comparison is safe and exact — no rounding). The boxes are disjoint
        // in X (5 blocks apart, 1 block wide), so the occupied set splits cleanly at
        // the gap between box A's X-run and box B's X-run.
        let shift = (n * voxels_per_block as i64) as f32; // N blocks → N×density voxels.
        let key = |position: [f32; 3]| -> [i64; 3] {
            // Bit-exact key: positions are k+0.5 half-integers, so ×2 is an exact
            // integer and avoids any float-equality fragility in the HashSet.
            [
                (position[0] * 2.0) as i64,
                (position[1] * 2.0) as i64,
                (position[2] * 2.0) as i64,
            ]
        };

        // The composite centre lies between the two boxes; split there.
        let mut xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let split_x = (xs.first().unwrap() + xs.last().unwrap()) / 2.0;

        let cluster_low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] < split_x)
            .map(|v| key(v.world_position))
            .collect();
        let cluster_high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] >= split_x)
            .map(|v| key(v.world_position))
            .collect();

        assert!(!cluster_low.is_empty() && !cluster_high.is_empty(), "both boxes present");
        assert_eq!(cluster_low.len(), cluster_high.len(), "both boxes fill one block");
        // Shifting the low box by exactly N×density in X reproduces the high box.
        let shifted: std::collections::HashSet<[i64; 3]> = cluster_low
            .iter()
            .map(|c| [c[0] + (shift * 2.0) as i64, c[1], c[2]])
            .collect();
        assert_eq!(shifted, cluster_high, "offset N blocks shifts voxels by exactly N×density");
    }

    /// ADR 0001 step 3 (b): two nodes at non-overlapping offsets give an occupied
    /// count equal to the SUM of each alone (a disjoint union — the placement
    /// genuinely separates them in space, no longer overlapping at the origin).
    #[test]
    fn disjoint_offsets_give_summed_occupancy() {
        let voxels_per_block = 8u32;
        // Two 1-block boxes 5 blocks apart in X — far enough that their voxel sets
        // never touch (each is 1 block = 8 voxels wide, gap is 4 empty blocks).
        let a_alone = boxed_block_positions(0, voxels_per_block).len();
        let b_alone = boxed_block_positions(5, voxels_per_block).len();
        assert!(a_alone > 0 && b_alone > 0);

        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut a = Node::new("A", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        a.transform.offset_blocks = [0, 0, 0];
        let mut b = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        b.transform.offset_blocks = [5, 0, 0];

        let scene = Scene {
            nodes: vec![a, b],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        // Region spans the full composite (offset 0..5, each 1 block) → 6 blocks X.
        let region = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(region.size_blocks, [6, 1, 1], "composite extent encompasses both offsets");
        let grid = scene.resolve_region(region, voxels_per_block, 0);
        assert_eq!(
            grid.occupied_count(),
            a_alone + b_alone,
            "disjoint placement → occupied count is the sum (no overlap)"
        );
    }

    /// ADR 0001 step 3 (c): `full_extent_blocks` grows to encompass an offset node.
    /// A single 2-block box pushed +4 blocks in X spans blocks `[3, 5]` in X (centre
    /// 4, ±1), so the composite X extent is 6 blocks (`0..6` once recentred), while
    /// Y/Z stay at the box's 2 blocks. (A zero-offset single node would be just the
    /// box's own 2×2×2.)
    #[test]
    fn full_extent_encompasses_offset_node() {
        let voxels_per_block = 4u32;
        let base = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new("Box", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        node.transform.offset_blocks = [4, 0, 0];
        let scene = Scene::single_node(node);

        // The box centred at block 4 with half-size 1 spans X blocks [3, 5] → its
        // own size (2) is unchanged but its placement means the bounding box from
        // the origin is wider. `full_extent_blocks` returns the box SIZE of the
        // composite: for a single node that is just the node's own size in every
        // axis (the offset moves it but doesn't enlarge a single box). To prove the
        // extent ACCOUNTS for the offset, compare against a two-node scene where the
        // offset opens a real gap.
        let single = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(single.size_blocks, [2, 2, 2], "a lone offset box keeps its own size");

        // Add a second box at the origin: now the composite must span from the
        // origin box (blocks [-1, 1]) to the offset box (blocks [3, 5]) → X width 6.
        let mut origin_box =
            Node::new("Origin", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        origin_box.transform.offset_blocks = [0, 0, 0];
        let mut offset_box =
            Node::new("Offset", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        offset_box.transform.offset_blocks = [4, 0, 0];
        let two = Scene {
            nodes: vec![origin_box, offset_box],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let extent = two.full_extent_blocks(voxels_per_block);
        assert_eq!(
            extent.size_blocks,
            [6, 2, 2],
            "the offset node widens the composite extent in X from 2 to 6 blocks"
        );
    }

    /// A 1×1×1 box Tool shape, used as a leaf in the step-4 recursion/instancing
    /// tests (the node carries the material; the shape does not).
    fn unit_box_shape(voxels_per_block: u32) -> SdfShape {
        SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        }
    }

    /// Key a grid's occupied voxels by exact half-integer voxel position (×2 → an
    /// exact integer, no float-equality fragility). Used to compare voxel SETS.
    fn position_keys(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
        grid.occupied
            .iter()
            .map(|v| {
                [
                    (v.world_position[0] * 2.0) as i64,
                    (v.world_position[1] * 2.0) as i64,
                    (v.world_position[2] * 2.0) as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 4 (nested transform composition): a leaf inside a `Group`
    /// offset by `+A` blocks, with the leaf itself offset `+B`, lands at world
    /// `A + B` (× density). We compare the grouped scene against a FLAT scene whose
    /// single node sits directly at `A + B` — same composite, so the recentre is
    /// identical and the voxel sets must match exactly.
    #[test]
    fn nested_group_composes_transforms_down() {
        let voxels_per_block = 8u32;
        let region = RegionBlocks::new([10, 1, 1]);
        let a = 3i64; // group offset
        let b = 2i64; // leaf offset within the group

        // Grouped: a Group at +A containing a box at +B.
        let mut leaf = Node::new(
            "Leaf",
            NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
        );
        leaf.transform.offset_blocks = [b, 0, 0];
        let mut group = Node::new("Group", NodeContent::Group(vec![leaf]));
        group.transform.offset_blocks = [a, 0, 0];
        let grouped = Scene {
            nodes: vec![group],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let grouped_grid = grouped.resolve_region(region, voxels_per_block, 0);

        // Flat reference: the same box placed directly at A + B.
        let mut flat_leaf = Node::new(
            "Flat",
            NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
        );
        flat_leaf.transform.offset_blocks = [a + b, 0, 0];
        let flat = Scene::single_node(flat_leaf);
        let flat_grid = flat.resolve_region(region, voxels_per_block, 0);

        assert!(grouped_grid.occupied_count() > 0, "the grouped leaf must emit voxels");
        assert_eq!(
            position_keys(&grouped_grid),
            position_keys(&flat_grid),
            "a leaf at +B inside a Group at +A must land at world A+B (× density)"
        );
    }

    /// ADR 0001 step 4 (instancing): an `Instance` of a 1-node definition placed at
    /// offset `T` resolves to the SAME voxels as that node placed directly at `T`.
    #[test]
    fn instance_matches_direct_placement() {
        let voxels_per_block = 8u32;
        let region = RegionBlocks::new([10, 1, 1]);
        let t = 4i64;
        let def_id = DefId(7);

        // Definition: a single box at the origin (within the def).
        let def = AssemblyDef {
            id: def_id,
            name: "Body".to_string(),
            children: vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Wood },
            )],
        };
        let mut instance = Node::new("I", NodeContent::Instance(def_id));
        instance.transform.offset_blocks = [t, 0, 0];
        let instanced = Scene {
            nodes: vec![instance],
            definitions: vec![def],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let instanced_grid = instanced.resolve_region(region, voxels_per_block, 0);

        // Direct: the same box placed directly at T.
        let mut direct = Node::new(
            "Direct",
            NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Wood },
        );
        direct.transform.offset_blocks = [t, 0, 0];
        let direct_grid = Scene::single_node(direct).resolve_region(region, voxels_per_block, 0);

        assert!(instanced_grid.occupied_count() > 0, "the instance must emit voxels");
        assert_eq!(
            position_keys(&instanced_grid),
            position_keys(&direct_grid),
            "an Instance of a 1-node def at T equals that node placed directly at T"
        );
    }

    /// ADR 0001 step 4 (village): a 2-instance scene (the SAME def placed at two
    /// different offsets) yields `occupied_count == 2 × the def's own count`, at two
    /// DISJOINT locations (the two voxel clusters never overlap).
    #[test]
    fn two_instance_village_doubles_occupancy_disjointly() {
        let voxels_per_block = 8u32;
        let def_id = DefId(2);

        // The "house": a single 1-block box (so its count is easy to reason about).
        let def = AssemblyDef {
            id: def_id,
            name: "House".to_string(),
            children: vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
            )],
        };

        // The def's own occupied count (resolved alone at the origin).
        let def_only = Scene {
            nodes: vec![Node::new("I", NodeContent::Instance(def_id))],
            definitions: vec![def.clone()],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let def_count = def_only
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(def_count > 0);

        // Two instances 6 blocks apart in X (a 1-block house → 5-block gap: disjoint).
        let mut house_a = Node::new("A", NodeContent::Instance(def_id));
        house_a.transform.offset_blocks = [0, 0, 0];
        let mut house_b = Node::new("B", NodeContent::Instance(def_id));
        house_b.transform.offset_blocks = [6, 0, 0];
        let village = Scene {
            nodes: vec![house_a, house_b],
            definitions: vec![def],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        let region = village.full_extent_blocks(voxels_per_block);
        let grid = village.resolve_region(region, voxels_per_block, 0);

        assert_eq!(
            grid.occupied_count(),
            2 * def_count,
            "two disjoint instances of one def → 2× the def's voxel count"
        );

        // Disjoint: split the occupied set at the composite centre; each half is a
        // full house, and the two halves share no voxel position.
        let xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position[0]).collect();
        let split_x = (xs.iter().cloned().fold(f32::MAX, f32::min)
            + xs.iter().cloned().fold(f32::MIN, f32::max))
            / 2.0;
        let low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] < split_x)
            .map(|v| [(v.world_position[0] * 2.0) as i64, (v.world_position[1] * 2.0) as i64, (v.world_position[2] * 2.0) as i64])
            .collect();
        let high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position[0] >= split_x)
            .map(|v| [(v.world_position[0] * 2.0) as i64, (v.world_position[1] * 2.0) as i64, (v.world_position[2] * 2.0) as i64])
            .collect();
        assert_eq!(low.len(), def_count, "the low cluster is one full house");
        assert_eq!(high.len(), def_count, "the high cluster is one full house");
        assert!(low.is_disjoint(&high), "the two houses occupy disjoint locations");
    }

    /// ADR 0001 step 4 (cycle guard): a definition that instances ITSELF resolves
    /// without stack overflow. The self-instance is skipped on re-entry, so the def
    /// contributes only its non-cyclic leaves finitely (here: one box) — never
    /// infinitely.
    #[test]
    fn self_referential_definition_does_not_overflow() {
        let voxels_per_block = 8u32;
        let def_id = DefId(1);

        // A definition whose children are (a) a real box leaf and (b) an Instance of
        // ITSELF — the cycle the guard must break.
        let def = AssemblyDef {
            id: def_id,
            name: "Recursive".to_string(),
            children: vec![
                Node::new(
                    "Box",
                    NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
                ),
                Node::new("Self", NodeContent::Instance(def_id)),
            ],
        };
        let scene = Scene {
            nodes: vec![Node::new("Root", NodeContent::Instance(def_id))],
            definitions: vec![def],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };

        // Resolves (no overflow) and contributes the single box ONCE — the self-
        // instance is skipped, so the count is finite and equals one box's voxels.
        let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
        let one_box = Scene::single_node(Node::new(
            "Box",
            NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
        ))
        .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
        .occupied_count();
        assert_eq!(
            grid.occupied_count(),
            one_box,
            "a self-instancing def contributes its leaves finitely (cycle skipped)"
        );
    }

    /// A small flat scene of two box Tools, the first selected — the fixture the
    /// tree-mutation UI helper tests build on.
    fn two_box_scene(voxels_per_block: u32) -> Scene {
        Scene {
            nodes: vec![
                Node::new(
                    "A",
                    NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
                ),
                Node::new(
                    "B",
                    NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Wood },
                ),
            ],
            definitions: Vec::new(),
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        }
    }

    /// ADR 0001 step 4 (UI helper): `group_active` wraps the active node in a new
    /// Group, so the active node becomes a CHILD of that Group. After grouping, the
    /// top-level node at the old slot is a `Group` whose sole child is the original
    /// node, and the active selection points at that child (path `[0, 0]`).
    #[test]
    fn group_active_nests_node_under_new_group() {
        let mut scene = two_box_scene(8);
        scene.active = Some(NodePath::root_index(0));

        let group_path = scene.group_active().expect("there is an active node to group");
        assert_eq!(group_path, NodePath::root_index(0), "the Group takes the old slot");

        // The top-level node is now a Group with exactly one child (the old "A").
        match &scene.nodes[0].content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1, "the Group holds exactly the wrapped node");
                assert_eq!(children[0].name, "A", "the wrapped child is the original node");
            }
            other => panic!("expected a Group at slot 0, got {other:?}"),
        }
        // The wrapped child is now the active selection.
        assert_eq!(scene.active, Some(NodePath::from_indices(vec![0, 0])));
        // The second node is untouched.
        assert_eq!(scene.nodes.len(), 2);
        assert!(matches!(scene.nodes[1].content, NodeContent::Tool { .. }));
    }

    /// ADR 0001 step 4 (UI helper): `make_definition_from_active` creates an
    /// `AssemblyDef` in `scene.definitions` and replaces the active node with an
    /// `Instance` of it. The resolved occupancy is unchanged (one stored body
    /// resolved via one instance == the original single node).
    #[test]
    fn make_definition_creates_def_and_instance() {
        let voxels_per_block = 8u32;
        let mut scene = two_box_scene(voxels_per_block);
        scene.active = Some(NodePath::root_index(0));

        // Occupancy of just the active node before the change (resolved alone).
        let before = Scene::single_node(scene.nodes[0].clone())
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(before > 0);

        let def_id = scene
            .make_definition_from_active("House")
            .expect("there is an active node to define");

        // A definition now exists, named, with the node's body as its children.
        assert_eq!(scene.definitions.len(), 1, "a definition appears in scene.definitions");
        let def = scene.def_by_id(def_id).expect("the new def is looked up by id");
        assert_eq!(def.name, "House");
        assert_eq!(def.children.len(), 1, "a single leaf becomes a one-node body");

        // The former node is now an Instance of that def.
        assert!(matches!(scene.nodes[0].content, NodeContent::Instance(id) if id == def_id));

        // Resolving the (now-instanced) node reproduces the original occupancy.
        let after = Scene {
            nodes: vec![scene.nodes[0].clone()],
            definitions: scene.definitions.clone(),
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        }
        .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
        .occupied_count();
        assert_eq!(after, before, "an instance of the def equals the original node");
    }

    /// ADR 0001 step 4 (UI helper, the village): after `make_definition_from_active`,
    /// `add_instance` appends another `Instance` node referencing the SAME def, and
    /// the scene resolves with the EXPECTED MULTIPLIED occupancy — two disjoint
    /// instances of a one-box def give 2× the box's voxel count.
    #[test]
    fn add_instance_multiplies_occupancy_via_one_definition() {
        let voxels_per_block = 8u32;
        // Start from a single box node, make it a definition (→ one instance), then
        // add a second instance.
        let mut scene = Scene::single_node(Node::new(
            "House",
            NodeContent::Tool { shape: unit_box_shape(voxels_per_block), material: MaterialChoice::Stone },
        ));
        let def_id = scene.make_definition_from_active("House").expect("active node");
        assert_eq!(scene.definitions.len(), 1);
        assert_eq!(scene.nodes.len(), 1, "the original node became one instance");

        // The def's own voxel count (one box).
        let one = scene
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(one > 0);

        // Add a second instance — an Instance node referencing the same def appears.
        let path = scene.add_instance(def_id).expect("the def exists");
        assert_eq!(scene.nodes.len(), 2, "an Instance node referencing the def appears");
        assert!(matches!(
            scene.node_at_path(&path).map(|n| &n.content),
            Some(NodeContent::Instance(id)) if *id == def_id
        ));
        // Still exactly ONE stored definition (reuse by reference).
        assert_eq!(scene.definitions.len(), 1, "the body is stored once, not copied");

        // The two instances are placed disjointly (add_instance nudges +X), so the
        // scene resolves to 2× the def's occupancy.
        let region = scene.full_extent_blocks(voxels_per_block);
        let total = scene.resolve_region(region, voxels_per_block, 0).occupied_count();
        assert_eq!(total, 2 * one, "two instances of one def → 2× the def's voxel count");
    }

    /// ADR 0001 step 4 (UI helper): `tree_rows` flattens the assembly depth-first,
    /// a parent immediately preceding its Group children at increasing depth, so the
    /// tree UI can render an indented list with selectable child nodes.
    #[test]
    fn tree_rows_lists_group_children_indented() {
        let mut scene = two_box_scene(8);
        // Group node A, then add a child into the Group, so the tree is:
        //   [0]          Group           depth 0
        //   [0, 0]         A (wrapped)    depth 1
        //   [0, 1]         child          depth 1
        //   [1]          B                depth 0
        scene.active = Some(NodePath::root_index(0));
        let group_path = scene.group_active().expect("active node");
        let added = scene.add_child_to_group(
            &group_path,
            Node::new("child", NodeContent::Part(Part::DebugClouds { seed: 0 })),
        );
        assert!(added, "the wrapped node is a Group so a child can be added");

        let rows = scene.tree_rows();
        let paths: Vec<(Vec<usize>, usize)> =
            rows.iter().map(|(p, d)| (p.indices.clone(), *d)).collect();
        assert_eq!(
            paths,
            vec![
                (vec![0], 0),    // Group
                (vec![0, 0], 1), // wrapped A
                (vec![0, 1], 1), // added child
                (vec![1], 0),    // B
            ],
            "tree_rows is depth-first with Group children indented under their parent"
        );
    }

    /// Selecting a node by path reaches a Group child (not just top-level nodes) —
    /// the inspector can therefore edit a node at any depth.
    #[test]
    fn node_at_path_reaches_group_child() {
        let mut scene = two_box_scene(8);
        scene.active = Some(NodePath::root_index(0));
        scene.group_active();
        // The active selection now points at the wrapped child [0, 0].
        let active = scene.active.clone().expect("a child is selected after grouping");
        assert_eq!(active, NodePath::from_indices(vec![0, 0]));
        let node = scene.node_at_path(&active).expect("the child resolves by path");
        assert_eq!(node.name, "A", "the path reaches the wrapped child node");
    }

    // ---- S0: chunk-addressable resolve (issue #27) ---------------------------
    //
    // These tests prove the ADDITIVE chunked resolve path reconstructs EXACTLY
    // what the monolithic `resolve_region` produces, after normalising for the
    // recentre offset that `resolve_region` applies and the chunk path does not.
    // The render path (`resolve_region`) is untouched; only these new functions
    // are exercised.

    /// Canonicalise an occupied set into a multiset of
    /// `(absolute_voxel_index, material_id)` so two resolves can be compared as
    /// the same shape regardless of voxel emission ORDER.
    ///
    /// `recentre_voxels` translates the frame into ABSOLUTE composite space: pass
    /// `[0,0,0]` for the chunked (already-absolute) frame, and the scene's
    /// recentre for the monolithic frame (whose positions are `absolute −
    /// recentre`). A voxel centre sits at an `n + 0.5` position, so `(p − 0.5)`
    /// recovers the integer voxel index exactly.
    fn occupied_multiset(
        grid: &VoxelGrid,
        recentre_voxels: [i64; 3],
    ) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
        let mut multiset = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let key = [
                (voxel.world_position[0] - 0.5).round() as i64 + recentre_voxels[0],
                (voxel.world_position[1] - 0.5).round() as i64 + recentre_voxels[1],
                (voxel.world_position[2] - 0.5).round() as i64 + recentre_voxels[2],
            ];
            *multiset.entry((key, voxel.material_id)).or_insert(0) += 1;
        }
        multiset
    }

    /// Assert the chunk-reassembled occupied set EXACTLY equals the monolithic
    /// `resolve_region`'s set (position + material), after recentre normalisation,
    /// AND that no chunk emits a voxel outside its own chunk AABB.
    fn assert_chunked_matches_monolithic(scene: &Scene, voxels_per_block: u32, label: &str) {
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let chunked = scene.resolve_region_via_chunks(voxels_per_block, 0);

        let recentre = scene.recentre_voxels(voxels_per_block);
        let monolithic_set = occupied_multiset(&monolithic, recentre);
        let chunked_set = occupied_multiset(&chunked, [0, 0, 0]);

        assert_eq!(
            chunked_set, monolithic_set,
            "[{label}] chunked occupied set must equal monolithic resolve (recentre-normalised)"
        );
        // Cross-check the count too (a multiset equality already implies it, but
        // this pins the failure message to the simplest symptom first).
        assert_eq!(
            chunked.occupied_count(),
            monolithic.occupied_count(),
            "[{label}] chunked occupied count must equal monolithic"
        );

        // Each per-chunk resolve must keep every voxel inside its OWN chunk AABB
        // (exactly-one-chunk ownership). Walk the covering range and re-resolve.
        let chunk_extent_voxels =
            (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            let mut total_from_chunks = 0usize;
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk_coord = [chunk_x, chunk_y, chunk_z];
                        let chunk = scene.resolve_chunk(chunk_coord, voxels_per_block, 0);
                        total_from_chunks += chunk.occupied_count();
                        for voxel in &chunk.occupied {
                            for axis in 0..3 {
                                let lo = (chunk_coord[axis] * chunk_extent_voxels) as f32;
                                let hi = lo + chunk_extent_voxels as f32;
                                let position = voxel.world_position[axis];
                                assert!(
                                    position >= lo && position < hi,
                                    "[{label}] voxel {position} on axis {axis} escaped chunk \
                                     {chunk_coord:?} box [{lo}, {hi})"
                                );
                            }
                        }
                    }
                }
            }
            // Every monolithic voxel is accounted for by exactly one chunk (no
            // double-counting, no drops): the chunk total equals the whole count.
            assert_eq!(
                total_from_chunks,
                monolithic.occupied_count(),
                "[{label}] summed per-chunk counts must equal the monolithic count \
                 (each voxel in exactly one chunk)"
            );
        }
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_blocks: [5, 5, 5],
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// Single-shape parity, all five SDF kinds — mirrors the all-shapes coverage
    /// style. (Single-node zero-offset scenes also exercise the recentre
    /// normalisation, since `resolve_region` recentres even a lone node.)
    #[test]
    fn chunked_resolve_matches_monolithic_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_chunked_matches_monolithic(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// A multi-node placed scene (the `--demo-scene` shape: a Sphere + an offset
    /// Box + an offset Torus, three materials) — proves the chunked path composes
    /// several leaves at distinct offsets and materials.
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_scene() {
        let voxels_per_block = 16;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let scene = Scene {
            nodes: vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    /// The `--demo-village` scene: four `Instance`s of one `House` definition (a
    /// Box body + a Cylinder chimney `Group`) — proves the chunked path follows
    /// instance + group transform composition (reuse-by-reference).
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: size,
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let house = AssemblyDef {
            id: house_def_id,
            name: "House".to_string(),
            children: vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform.offset_blocks = offset;
            node
        };
        let scene = Scene {
            nodes: vec![
                instance("House 1", [0, 0, 0]),
                instance("House 2", [6, 0, 0]),
                instance("House 3", [12, 0, 0]),
                instance("House 4", [18, 0, 0]),
            ],
            definitions: vec![house],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// A scene with a single node shifted well OFF the origin (+8 blocks on X) —
    /// proves the chunked path handles off-centre placement (the AABB does not
    /// start at the origin, so the covering chunk range is non-trivial and the
    /// recentre offset is non-zero).
    #[test]
    fn chunked_resolve_matches_monolithic_for_offset_node() {
        let voxels_per_block = 16;
        let shape = SdfShape {
            kind: ShapeKind::Sphere,
            size_blocks: [4, 4, 4],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new(
            "Offset sphere",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        node.transform.offset_blocks = [8, 0, 0];
        let scene = Scene::single_node(node);

        // Sanity: the recentre is genuinely non-zero for this off-centre scene, so
        // the normalisation is actually exercised (a zero recentre would make the
        // test vacuous on that axis).
        let recentre = scene.recentre_voxels(voxels_per_block);
        assert_ne!(
            recentre, [0, 0, 0],
            "an off-centre node must produce a non-zero recentre (else the \
             normalisation is untested)"
        );
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "offset-node");
    }

    /// A chunk that no leaf overlaps resolves to an EMPTY grid (no panic), and its
    /// dimensions are still one chunk's extent.
    #[test]
    fn empty_chunk_resolves_to_empty_grid() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        // A chunk far outside the (origin-area) composite AABB.
        let chunk = scene.resolve_chunk([1000, 1000, 1000], 16, 0);
        assert_eq!(chunk.occupied_count(), 0, "a far-off chunk must be empty");
        let chunk_extent = crate::renderer::CHUNK_BLOCKS * 16;
        assert_eq!(
            chunk.dimensions,
            [chunk_extent, chunk_extent, chunk_extent],
            "an empty chunk still reports one chunk's voxel extent"
        );
    }

    /// Parity holds at a non-default density too (16 is the app default; this pins
    /// that the chunk-extent / ownership math is density-correct).
    #[test]
    fn chunked_resolve_matches_monolithic_at_density_8() {
        let scene = shape_scene(ShapeKind::Torus, 8);
        assert_chunked_matches_monolithic(&scene, 8, "torus@8");
    }

    // ---- S1: far-offset placement (ADR 0002 streaming, part of #18) -----------
    //
    // The durable artifact for streaming S1: a node placed at a LARGE block offset
    // (matching `shot --demo-far-offset`'s 100_000 blocks) really lands far away in
    // ABSOLUTE composite space, independent of the live render recentre. This is
    // proved via the S0 absolute-coordinate chunk path (`resolve_chunk` /
    // `resolve_region_via_chunks`), which — unlike `resolve_region` — does NOT
    // recentre, so its voxel positions ARE the scene's true composite coordinates.
    //
    // `offset_blocks` is `[i64; 3]` (widened in S4a); 100_000 blocks is comfortably
    // in i32 range too, and at density 16 lands the box ~1.6M voxels out. The
    // BEYOND-i32 composition (offsets past ±2.1×10⁹) is proven separately in
    // `i64_composition_beyond_i32_range_is_exact` (pure integer, no f32 precision
    // loss).

    /// A far-offset node resolves to absolute voxel/chunk coordinates around
    /// 100_000 blocks: the box's voxels sit at absolute X ≈ 100_000 × density, the
    /// owning chunks are around `100_000 × density / chunk_extent`, and the box is
    /// genuinely placed far away (the absolute coords are NOT near the origin —
    /// only the recentred render path maps it home). Independent of any render math.
    #[test]
    fn far_offset_node_resolves_to_absolute_coords_near_100k() {
        let voxels_per_block = 16u32;
        let offset_blocks = 100_000i64;
        // A 4³ box — the same recognizable shape `shot --demo-far-offset` builds.
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [4, 4, 4],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform.offset_blocks = [offset_blocks, 0, 0];
        let scene = Scene::single_node(node);

        // The ABSOLUTE-coordinate chunk path (no recentre): these positions are the
        // scene's TRUE composite coordinates, so they reveal the far placement that
        // the render recentre hides.
        let absolute = scene.resolve_region_via_chunks(voxels_per_block, 0);
        assert!(
            absolute.occupied_count() > 0,
            "the far box must resolve to voxels"
        );

        // Every voxel's absolute X centre lands in the far block's voxel span. The
        // 4-block box centred on block 100_000 spans blocks [99_998, 100_002), i.e.
        // absolute voxels [99_998·d, 100_002·d). (Y/Z are centred on 0, unchanged.)
        let density = voxels_per_block as f32;
        let span_lo = (offset_blocks - 2) as f32 * density;
        let span_hi = (offset_blocks + 2) as f32 * density;
        let expected_centre_voxels = offset_blocks as f32 * density; // 1_600_000
        for voxel in &absolute.occupied {
            let x = voxel.world_position[0];
            assert!(
                x >= span_lo && x < span_hi,
                "far-box voxel X={x} must lie in the absolute span [{span_lo}, {span_hi}) \
                 around 100_000 blocks — NOT near the origin"
            );
        }
        // The box is genuinely ~1.6M voxels out (sanity: not collapsed to origin).
        assert!(
            expected_centre_voxels > 1_000_000.0,
            "at density {voxels_per_block}, 100_000 blocks is >1M voxels from the origin"
        );

        // Mean absolute X is within half a block of the far centre (the box is
        // symmetric about block 100_000), confirming the placement, not the recentre.
        let mean_x: f64 = absolute
            .occupied
            .iter()
            .map(|v| v.world_position[0] as f64)
            .sum::<f64>()
            / absolute.occupied_count() as f64;
        assert!(
            (mean_x - expected_centre_voxels as f64).abs() <= (density / 2.0) as f64,
            "the far box's mean absolute X ({mean_x}) must sit at ~{expected_centre_voxels} \
             voxels (block 100_000 × density), proving far placement in absolute space"
        );

        // The owning chunk coordinates are around 100_000 × density / chunk_extent,
        // i.e. far from chunk 0 — the chunk addressing places it far away too.
        let chunk_extent_voxels =
            (crate::renderer::CHUNK_BLOCKS * voxels_per_block) as i64;
        let expected_chunk_x = ((offset_blocks * voxels_per_block as i64) / chunk_extent_voxels) as i32;
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("the far box has an intrinsic size → a covering chunk range");
        assert!(
            min_chunk[0] <= expected_chunk_x && expected_chunk_x <= max_chunk[0],
            "the far box's owning chunk-X range [{}, {}] must bracket chunk {expected_chunk_x} \
             (≈100_000 blocks out), not chunk 0",
            min_chunk[0],
            max_chunk[0]
        );
        assert!(
            min_chunk[0] > 1000,
            "the far box must be owned by a high chunk coordinate (>1000), proving it is \
             far from the origin in chunk space (got {})",
            min_chunk[0]
        );

        // Cross-check: the ABSOLUTE chunk path and the RECENTRED render path agree
        // on the box's SHAPE — they differ ONLY by the recentre offset, which is
        // exactly the far placement. This pins that the render recentre is what maps
        // the far box home (and is the exact thing S4 will remove), while the
        // absolute path keeps it far.
        let recentre = scene.recentre_voxels(voxels_per_block);
        assert_eq!(
            recentre[0],
            offset_blocks * voxels_per_block as i64,
            "the recentre offset equals the full far placement — it is what hides the \
             far offset from the live render today (S4 removes it)"
        );
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        assert_eq!(
            occupied_multiset(&monolithic, recentre),
            occupied_multiset(&absolute, [0, 0, 0]),
            "the recentred render box and the absolute far box are the SAME shape, \
             offset by exactly the recentre (the far placement)"
        );
    }

    /// S4a (64-bit world addressing): nested transforms compose down the tree in
    /// **i64**, so a leaf whose accumulated block offset exceeds the `i32` range
    /// lands at the EXACT absolute coordinate — no overflow, no truncation. This is
    /// the load-bearing data-model guarantee of S4a, proven in PURE INTEGER space
    /// (the producer-true voxel AABB from `build_leaf_spatial_index`) so there is no
    /// f32 precision loss to muddy the result.
    ///
    /// A Group offset `+2_000_000_000` blocks contains a leaf offset `+1_000_000_000`
    /// blocks; their sum `3_000_000_000` is past `i32::MAX` (2_147_483_647). The
    /// composed absolute-voxel centre must be `3_000_000_000 × density` — a value
    /// that would have wrapped to a negative number under the old i32 composition.
    #[test]
    fn i64_composition_beyond_i32_range_is_exact() {
        let voxels_per_block = 16u32;
        let density = voxels_per_block as i64;
        let group_offset: i64 = 2_000_000_000; // ~i32::MAX on its own
        let leaf_offset: i64 = 1_000_000_000;
        let composed_blocks = group_offset + leaf_offset; // 3e9 — past i32::MAX
        assert!(
            composed_blocks > i32::MAX as i64,
            "the composed offset must exceed i32 range to exercise 64-bit addressing"
        );

        // A 1-block box leaf inside a Group; the leaf carries +leaf_offset, the Group
        // +group_offset, so the leaf's world offset composes to their sum.
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [1, 1, 1],
            voxels_per_block,
            wall_blocks: 1,
        };
        let mut leaf = Node::new(
            "Leaf",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        leaf.transform.offset_blocks = [leaf_offset, 0, 0];
        let mut group = Node::new("Group", NodeContent::Group(vec![leaf]));
        group.transform.offset_blocks = [group_offset, 0, 0];
        let scene = Scene {
            nodes: vec![group],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        };

        // The producer-true voxel AABB (pure i64) is block-lattice aligned (issue
        // #30): a 1-block box occupies the whole-block range `[off, off + 1)` in
        // blocks, so in voxels `[off·d, off·d + d)` — the min sits ON a block
        // multiple, NOT half a block below the composed centre.
        let index = scene.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(index.entries.len(), 1, "exactly one leaf is indexed");
        let aabb = index.entries[0].world_aabb;
        let composed_voxels = composed_blocks * density; // 48_000_000_000 — past i32 too
        assert_eq!(
            aabb.min[0], composed_voxels,
            "the composed leaf min-X must equal (group+leaf)·density (block-aligned), \
             with NO i32 overflow (got {}, want {})",
            aabb.min[0], composed_voxels
        );
        assert_eq!(
            aabb.max[0], composed_voxels + density,
            "the composed leaf max-X must be exact in i64"
        );
        // Sanity: this absolute voxel coordinate genuinely exceeds the i32 range, so
        // the test would have FAILED (wrapped negative) under i32 composition.
        assert!(
            composed_voxels > i32::MAX as i64,
            "the absolute voxel coordinate ({composed_voxels}) is past i32::MAX — the \
             exact case 64-bit addressing exists to handle"
        );

        // The covering chunk range also derives correctly (chunk coord narrows to i32
        // safely): chunk-X = composed_voxels / chunk_extent, well inside i32.
        let chunk_extent = (crate::renderer::CHUNK_BLOCKS as i64) * density;
        let expected_chunk_x = composed_voxels.div_euclid(chunk_extent);
        assert!(
            expected_chunk_x <= i32::MAX as i64,
            "the derived chunk coordinate stays inside i32 even for a 3e9-block offset"
        );
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("the far leaf has an intrinsic size");
        assert!(
            (min_chunk[0] as i64) <= expected_chunk_x && expected_chunk_x <= (max_chunk[0] as i64),
            "the covering chunk-X range must bracket the composed chunk {expected_chunk_x}"
        );
    }

    // ===== Issue #27 S3: leaf spatial index =====================================

    /// The ground-truth leaf set a query AABB selects: a FULL `for_each_leaf` walk,
    /// recomputing each leaf's producer-true voxel AABB inline (the same maths
    /// `build_leaf_spatial_index` uses), filtered by overlap with `query`. The
    /// spatial index must return exactly this set; that equality is the S3
    /// correctness contract.
    fn walk_leaf_aabbs_intersecting(
        scene: &Scene,
        voxels_per_block: u32,
        query: &crate::spatial_index::VoxelAabb,
    ) -> Vec<crate::spatial_index::VoxelAabb> {
        let density = voxels_per_block.max(1) as i64;
        let mut matched = Vec::new();
        scene.for_each_leaf(&mut |world_offset, content| {
            let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                return; // region-spanning leaf — not an AABB match.
            };
            let mut min = [0i64; 3];
            let mut max = [0i64; 3];
            let lattice_shift = leaf_lattice_shift_voxels(size_blocks, voxels_per_block);
            for axis in 0..3 {
                let grid = size_blocks[axis] as i64 * density;
                let centre = world_offset[axis] * density;
                min[axis] = centre - grid / 2 + lattice_shift[axis];
                max[axis] = min[axis] + grid;
            }
            let aabb = crate::spatial_index::VoxelAabb::new(min, max);
            if aabb.intersects(query) {
                matched.push(aabb);
            }
        });
        matched
    }

    fn sorted_aabbs(
        mut boxes: Vec<crate::spatial_index::VoxelAabb>,
    ) -> Vec<([i64; 3], [i64; 3])> {
        boxes.sort_by_key(|b| (b.min, b.max));
        boxes.into_iter().map(|b| (b.min, b.max)).collect()
    }

    fn demo_three_tool_scene(voxels_per_block: u32) -> Scene {
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        Scene {
            nodes: vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        }
    }

    fn demo_village_scene(voxels_per_block: u32) -> Scene {
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: size,
                voxels_per_block,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform.offset_blocks = offset;
            node
        };
        let house = AssemblyDef {
            id: house_def_id,
            name: "House".to_string(),
            children: vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform.offset_blocks = offset;
            node
        };
        Scene {
            nodes: vec![
                instance("House 1", [0, 0, 0]),
                instance("House 2", [6, 0, 0]),
                instance("House 3", [12, 0, 0]),
                instance("House 4", [18, 0, 0]),
            ],
            definitions: vec![house],
            active: Some(NodePath::root_index(0)),
            ..Scene::default()
        }
    }

    /// The index query returns EXACTLY the leaves a full walk + AABB filter returns,
    /// across several query boxes and several scenes (incl. instanced/recursive
    /// `--demo-village`). This is the S3 spatial-index correctness proof.
    #[test]
    fn spatial_index_query_matches_full_walk() {
        use crate::spatial_index::VoxelAabb;
        let voxels_per_block = 16;
        let scenes = [
            ("single", Scene::from_geometry(
                GeometryParams { shape: ShapeKind::Sphere, size_blocks: [5, 5, 5], voxels_per_block, wall_blocks: 1 },
                MaterialChoice::Stone,
            )),
            ("three-tool", demo_three_tool_scene(voxels_per_block)),
            ("village", demo_village_scene(voxels_per_block)),
        ];
        // A spread of query boxes: empty, tiny near origin, a slab, the whole scene,
        // and a far-away box that should match nothing.
        let queries = [
            VoxelAabb::new([0, 0, 0], [0, 0, 0]),
            VoxelAabb::new([-8, -8, -8], [8, 8, 8]),
            VoxelAabb::new([0, -200, -200], [64, 200, 200]),
            VoxelAabb::new([-5000, -5000, -5000], [5000, 5000, 5000]),
            VoxelAabb::new([100_000, 0, 0], [100_064, 64, 64]),
        ];
        for (label, scene) in &scenes {
            let index = scene.build_leaf_spatial_index(voxels_per_block);
            for query in &queries {
                let from_index: Vec<VoxelAabb> = index
                    .leaves_intersecting(query)
                    .into_iter()
                    .map(|entry| entry.world_aabb)
                    .collect();
                let from_walk = walk_leaf_aabbs_intersecting(scene, voxels_per_block, query);
                assert_eq!(
                    sorted_aabbs(from_index),
                    sorted_aabbs(from_walk),
                    "[{label}] index query {query:?} must match the full walk + AABB filter"
                );
            }
        }
    }

    /// The diff that drives invalidation: an edit's AABB is the union of the old and
    /// new boxes of whatever changed.
    #[test]
    fn edit_aabb_diff_covers_old_and_new() {
        let voxels_per_block = 16;
        let scene_a = demo_three_tool_scene(voxels_per_block);
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);

        // No change: empty edit AABB.
        let index_a2 = scene_a.build_leaf_spatial_index(voxels_per_block);
        let no_edit = index_a2.edit_aabb_since(&index_a).expect("same density");
        assert!(no_edit.is_empty(), "an identical scene dirties nothing");

        // Move the Box (node 1) from +8X to +40X: the edit AABB must span BOTH the
        // old (+8) and new (+40) boxes.
        let mut scene_b = scene_a.clone();
        scene_b.nodes[1].transform.offset_blocks = [40, 0, 0];
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        let moved = index_b.edit_aabb_since(&index_a).expect("same density");
        assert!(!moved.is_empty());
        // A 5-block (odd) box is block-lattice aligned (issue #30): it occupies the
        // block range `[off − 2, off + 3)`, i.e. voxels `[(off−2)·16, (off+3)·16)`.
        // Old at +8: [(8−2)·16, (8+3)·16) = [96, 176). New at +40: [608, 688). The
        // union must contain both.
        assert!(moved.min[0] <= (8 - 2) * 16, "edit AABB must cover the OLD location");
        assert!(moved.max[0] >= (40 + 3) * 16, "edit AABB must cover the NEW location");

        // Recolour the Sphere (node 0, same box): edit AABB is just that box.
        let mut scene_c = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_c.nodes[0].content {
            *material = MaterialChoice::Wood;
        }
        let index_c = scene_c.build_leaf_spatial_index(voxels_per_block);
        let recoloured = index_c.edit_aabb_since(&index_a).expect("same density");
        assert!(!recoloured.is_empty(), "a same-box content change is still dirty");
        // Sphere at origin, 5 blocks (odd) → block range [-2, 3) → voxels [-32, 48)
        // (block-lattice aligned, issue #30).
        assert_eq!(recoloured, crate::spatial_index::VoxelAabb::new([-32, -32, -32], [48, 48, 48]));
    }

    /// A density change can't be localised: the diff returns `None` (clear).
    #[test]
    fn edit_aabb_diff_density_change_is_none() {
        let scene = demo_three_tool_scene(16);
        let index_16 = scene.build_leaf_spatial_index(16);
        let index_8 = scene.build_leaf_spatial_index(8);
        assert_eq!(
            index_8.edit_aabb_since(&index_16),
            None,
            "a density change forces a wholesale clear"
        );
    }

    /// A region-spanning Part edit can't be localised: the diff returns `None`.
    #[test]
    fn edit_aabb_diff_part_edit_is_none() {
        let voxels_per_block = 16;
        // A scene with a Tool plus a debug-cloud Part.
        let mut tool = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape { kind: ShapeKind::Sphere, size_blocks: [5, 5, 5], voxels_per_block, wall_blocks: 1 },
                material: MaterialChoice::Stone,
            },
        );
        tool.transform.offset_blocks = [0, 0, 0];
        let part = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 1 }));
        let scene_a = Scene { nodes: vec![tool.clone(), part], active: Some(NodePath::root_index(0)), ..Scene::default() };
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);
        assert!(index_a.has_region_spanning_leaf);

        // Change the Part's seed (a region-spanning content change).
        let mut scene_b = scene_a.clone();
        scene_b.nodes[1].content = NodeContent::Part(Part::DebugClouds { seed: 2 });
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(
            index_b.edit_aabb_since(&index_a),
            None,
            "editing a region-spanning Part forces a wholesale clear"
        );
    }

    // ===== Issue #30: shape generation aligns to the global block lattice ========

    /// Resolve a single Box leaf of `size_blocks` at the origin and return its
    /// occupied voxels' **absolute** (producer-true, non-recentred) integer-index
    /// bounding box `(min_corner, max_corner_exclusive)` plus the occupied count. A
    /// Box fully fills its bounding box, so the count is `prod(size·d)` and the box
    /// is the exact placed extent — letting the lattice-alignment tests read where
    /// generation actually lands relative to block multiples (multiples of `d`).
    fn absolute_box_extent(
        size_blocks: [u32; 3],
        voxels_per_block: u32,
    ) -> ([i64; 3], [i64; 3], usize) {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_blocks,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        // `resolve_region_via_chunks` keeps ABSOLUTE (non-recentred) positions, so
        // its voxels are in the producer-true frame the per-object grids (#29) read.
        let grid = scene.resolve_region_via_chunks(voxels_per_block, 0);
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in &grid.occupied {
            for axis in 0..3 {
                // Voxel centres sit at `n + 0.5`; `floor` recovers the cell index.
                let index = voxel.world_position[axis].floor() as i64;
                min[axis] = min[axis].min(index);
                max[axis] = max[axis].max(index + 1); // half-open upper bound
            }
        }
        (min, max, grid.occupied.len())
    }

    /// Assert that a box of `size_blocks` resolved at `density` is block-lattice
    /// aligned: it generates exactly `prod(size·d)` voxels and every occupied
    /// block boundary (both the min corner and the max boundary, per axis) lands
    /// on a global block multiple (a multiple of `d`) — no half-block straddle.
    /// Shared by the density-parametrized generation tests below.
    fn assert_box_block_aligned(size: [u32; 3], density: u32) {
        let (min, max, count) = absolute_box_extent(size, density);
        let expected_count = (size[0] * density) as usize
            * (size[1] * density) as usize
            * (size[2] * density) as usize;
        assert_eq!(
            count, expected_count,
            "a {size:?}-block box at density {density} must generate prod(size·d) voxels"
        );
        for (axis, &size_axis) in size.iter().enumerate() {
            assert_eq!(
                min[axis].rem_euclid(density as i64),
                0,
                "axis {axis}: min corner ({}) must sit on a block multiple of {density} \
                 — no half-block straddle (size {size:?})",
                min[axis]
            );
            assert_eq!(
                max[axis].rem_euclid(density as i64),
                0,
                "axis {axis}: max boundary ({}) must fall on a block multiple of {density} \
                 (size {size:?})",
                max[axis]
            );
            assert_eq!(
                max[axis] - min[axis],
                (size_axis * density) as i64,
                "axis {axis}: the span must be size·d voxels (size {size:?} @ {density})"
            );
        }
    }

    /// A 1×1×1-block box at density `d` generates exactly `d³` voxels occupying
    /// exactly ONE block-aligned cell `[k·d, (k+1)·d)` per axis (min corner on a
    /// multiple of `d`, span exactly one block). The user's concrete acceptance
    /// test for issue #30, generalized across the representative density set —
    /// INCLUDING the requested **d=1** (→ 1 voxel) and **d=15** (→ 15³ = 3375),
    /// plus d=2, d=16 (default, → 4096), d=32.
    #[test]
    fn one_block_box_aligns_across_densities() {
        for density in [1u32, 2, 15, 16, 32] {
            let (min, max, count) = absolute_box_extent([1, 1, 1], density);
            assert_eq!(
                count,
                (density as usize).pow(3),
                "a 1×1×1 box at density {density} must generate exactly density³ voxels"
            );
            for axis in 0..3 {
                assert_eq!(
                    min[axis].rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: min voxel corner ({}) must sit on a block \
                     multiple — no half-block straddle",
                    min[axis]
                );
                assert_eq!(
                    max[axis] - min[axis],
                    density as i64,
                    "axis {axis} @ d{density}: a 1-block box spans exactly one block"
                );
                assert_eq!(
                    max[axis].rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: max boundary must fall on a block multiple"
                );
            }
        }
    }

    /// An odd-sized shape (5×5×2) is block-lattice aligned across densities: it
    /// generates `(5d)×(5d)×(2d)` voxels whose every block boundary coincides with
    /// a global block boundary — the off-centre-odd-size bug fix (#30) — at
    /// d ∈ {1, 15, 16}.
    #[test]
    fn odd_size_shape_is_block_lattice_aligned() {
        for density in [1u32, 15, 16] {
            assert_box_block_aligned([5, 5, 2], density);
        }
    }

    /// An even-sized shape (2×4×6) is also block-lattice aligned across densities
    /// (it straddles the origin symmetrically), confirming the convention holds for
    /// both parities at d ∈ {1, 15, 16}.
    #[test]
    fn even_size_shape_is_block_lattice_aligned() {
        for density in [1u32, 15, 16] {
            assert_box_block_aligned([2, 4, 6], density);
            // An even size straddles the origin symmetrically: [-size/2·d, +size/2·d).
            let size = [2u32, 4, 6];
            let (min, _max, _count) = absolute_box_extent(size, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                assert_eq!(
                    min[axis],
                    -((size_axis / 2 * density) as i64),
                    "axis {axis} @ d{density}: an even box is symmetric about the origin"
                );
            }
        }
    }

    // ===== Issue #29 foundation: per-object block-aligned voxel AABB + pivot =====
    //
    // The grid rework (#29) positions each object's block lattice / floor / voxel
    // grid and the transform gizmo from the node's BLOCK-ALIGNED VOXEL AABB and its
    // pivot/origin, in the recentred frame, across densities. The renderers don't
    // exist yet, but the geometry SOURCE does — `build_leaf_spatial_index` (the
    // per-leaf world AABB) and `recentre_voxels_for_resolve` (the recentre). These
    // tests pin that source. The RENDERER-level grid/lattice/gizmo-follow tests
    // (drawing the actual lines and the gizmo) will be added with #29 sub-steps
    // S3/S5, parametrized over the SAME density set {1, 15, 16}, once those
    // renderers exist.

    /// The single leaf's block-aligned voxel AABB, as `build_leaf_spatial_index`
    /// records it (the #29 grids' geometry source).
    fn single_leaf_aabb(size_blocks: [u32; 3], offset_blocks: [i64; 3], density: u32) -> VoxelAabb {
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks,
            voxels_per_block: density,
            wall_blocks: 1,
        };
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform.offset_blocks = offset_blocks;
        let scene = Scene { nodes: vec![node], active: Some(NodePath::root_index(0)), ..Scene::default() };
        let index = scene.build_leaf_spatial_index(density);
        assert_eq!(index.entries.len(), 1, "one Tool leaf → one index entry");
        index.entries[0].world_aabb
    }

    /// A `B`-block extent → a `B·d`-voxel AABB whose corners land on block
    /// multiples of `d`, at each density. This is the geometry the per-object
    /// block lattice / floor / voxel grid (#29) will span (expanded out to
    /// enclosing whole blocks — here already whole-block, since generation is
    /// block-aligned).
    #[test]
    fn node_block_aabb_scales_and_aligns_across_densities() {
        let size = [5u32, 5, 2]; // a representative odd extent
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let aabb = single_leaf_aabb(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                // Scales with density: a B-block extent → B·d voxels.
                assert_eq!(
                    aabb.max[axis] - aabb.min[axis],
                    (size_axis * density) as i64,
                    "axis {axis} @ d{density}: AABB extent must be size·d voxels"
                );
                // Corners land on block multiples of d (block-aligned, no half-block).
                assert_eq!(
                    aabb.min[axis].rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: AABB min corner ({}) must be a block multiple",
                    aabb.min[axis]
                );
                assert_eq!(
                    aabb.max[axis].rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: AABB max corner ({}) must be a block multiple",
                    aabb.max[axis]
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its AABB by
    /// exactly `d` voxels per axis (the grids/gizmo follow it), and the AABB stays
    /// block-aligned, at each density. Offsets are whole blocks (`offset_blocks:
    /// [i64; 3]`), so sub-block translation is not representable — whole-block
    /// translation is the unit tested.
    #[test]
    fn node_aabb_follows_translation_at_each_density() {
        let size = [5u32, 5, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let before = single_leaf_aabb(size, base, density);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let after = single_leaf_aabb(size, shifted, density);
                for axis in 0..3 {
                    let expected = if axis == moved_axis { density as i64 } else { 0 };
                    assert_eq!(
                        after.min[axis] - before.min[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block on axis {moved_axis} must shift \
                         the AABB min by exactly d on that axis (0 elsewhere)"
                    );
                    assert_eq!(
                        after.max[axis] - before.max[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block must shift the AABB max by d"
                    );
                    // Still block-aligned after the move.
                    assert_eq!(
                        after.min[axis].rem_euclid(density as i64),
                        0,
                        "axis {axis} @ d{density}: AABB stays block-aligned after translate"
                    );
                }
            }
        }
    }

    /// The node pivot/origin the selection transform gizmo (#29) will track: the
    /// node's world origin = `offset_blocks·d − recentre`, in the recentred frame.
    /// Pinned across densities for two facets:
    ///
    /// 1. **Recentred-frame value.** For a SINGLE-node scene the recentre always
    ///    re-centres that one node, so its pivot in the recentred frame is the
    ///    node's own centre offset from the recentre — INVARIANT under translation
    ///    (translating the lone node drags the auto-recentre with it). We pin the
    ///    concrete value `offset·d − recentre` and assert it does NOT move when the
    ///    node is translated alone. (This is why #29 positions grids in the GLOBAL
    ///    lattice frame, not this auto-recentred composite — only a fixed frame
    ///    makes "the gizmo follows the object" observable.)
    /// 2. **Absolute-frame follow.** In the producer-true ABSOLUTE frame the node
    ///    origin is `offset_blocks·d`; this DOES follow a `+1 block` translate by
    ///    exactly `d` voxels per axis (the property the global-frame gizmo inherits).
    #[test]
    fn node_pivot_origin_tracks_offset_across_densities() {
        let size = [5u32, 5, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let recentre_of = |offset: [i64; 3]| {
                let shape = SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: size,
                    voxels_per_block: density,
                    wall_blocks: 1,
                };
                let mut node =
                    Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
                node.transform.offset_blocks = offset;
                let scene = Scene {
                    nodes: vec![node],
                    active: Some(NodePath::root_index(0)),
                    ..Scene::default()
                };
                scene.recentre_voxels_for_resolve(density)
            };
            // Pivot in the recentred frame = offset·d − recentre.
            let recentred_pivot = |offset: [i64; 3]| {
                let recentre = recentre_of(offset);
                [
                    offset[0] * density as i64 - recentre[0],
                    offset[1] * density as i64 - recentre[1],
                    offset[2] * density as i64 - recentre[2],
                ]
            };
            // Absolute-frame node origin = offset·d (no recentre).
            let absolute_origin =
                |offset: [i64; 3]| [offset[0] * density as i64, offset[1] * density as i64, offset[2] * density as i64];

            let base_recentred = recentred_pivot(base);
            let base_absolute = absolute_origin(base);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let moved_recentred = recentred_pivot(shifted);
                let moved_absolute = absolute_origin(shifted);
                for axis in 0..3 {
                    // (1) Single-node recentred pivot is invariant under self-translation.
                    assert_eq!(
                        moved_recentred[axis], base_recentred[axis],
                        "axis {axis} @ d{density}: a lone node's recentred pivot is invariant \
                         under self-translation (the auto-recentre follows it)"
                    );
                    // (2) Absolute origin follows +1 block by exactly d on that axis.
                    let expected = if axis == moved_axis { density as i64 } else { 0 };
                    assert_eq!(
                        moved_absolute[axis] - base_absolute[axis],
                        expected,
                        "axis {axis} @ d{density}: absolute node origin must follow a +1-block \
                         translate on axis {moved_axis} by exactly d voxels (0 elsewhere)"
                    );
                }
            }
        }
    }

    // ---- issue #29 (grid rework S1): per-node grids, Points, masters ----

    /// A freshly-built node carries NO grids (issue #29: grids default OFF for new
    /// objects). `NodeGrids::default()` is all-false, and `Node::new` adopts it.
    #[test]
    fn new_node_has_all_grids_off() {
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [1, 1, 1],
                    voxels_per_block: 8,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        assert!(!node.grids.voxel_grid_on_faces);
        assert!(!node.grids.block_lattice);
        assert!(!node.grids.floor_grid);
        assert_eq!(node.grids, NodeGrids::default());
    }

    /// An empty `Scene::default()` has the issue-#29 master defaults (block lattice
    /// ON, voxel grid + floor grid OFF) and no Points yet.
    #[test]
    fn scene_default_master_grids() {
        let scene = Scene::default();
        assert!(scene.master_block_lattice, "block lattice master defaults ON");
        assert!(!scene.master_voxel_grid, "voxel grid master defaults OFF");
        assert!(!scene.master_floor_grid, "floor grid master defaults OFF");
        assert!(scene.points.is_empty(), "no Points until ensure_origin_point");
        assert_eq!(scene.active_point, None);
    }

    /// `ensure_origin_point` is idempotent and creates EXACTLY one Origin at index 0
    /// with the spec defaults (ground plane + axes on); a second call (or a scene
    /// that already has an Origin) does not duplicate it.
    #[test]
    fn ensure_origin_point_is_idempotent_and_creates_one_origin() {
        let mut scene = Scene::default();
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "exactly one Point after first call");
        let origin = &scene.points[0];
        assert!(origin.is_origin, "the synthesized Point is the Origin");
        assert_eq!(origin.name, "Origin");
        assert_eq!(origin.position_blocks, [0, 0, 0]);
        assert!(origin.plane_xz, "ground plane on by default");
        assert!(origin.axes, "axes on by default");
        assert!(!origin.plane_xy && !origin.plane_yz);
        assert!(!origin.hidden);

        // Idempotent: a second call does not add another Origin.
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "second call adds nothing");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// An existing Origin (anywhere in the list) is NOT duplicated by
    /// `ensure_origin_point`; a scene that already carries one is left untouched.
    #[test]
    fn ensure_origin_point_does_not_duplicate_existing_origin() {
        let mut scene = Scene::default();
        // Seed a non-origin Point first, then an Origin at index 1.
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });
        scene.add_point(Point { name: "Origin".to_string(), is_origin: true, ..Point::default() });
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 2, "no Origin inserted when one exists");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// `remove_point` deletes a normal Point but NO-OPS on the Origin (undeletable),
    /// and `toggle_point_hidden` hides the Origin (hideable).
    #[test]
    fn remove_point_spares_origin_which_is_hideable() {
        let mut scene = Scene::default();
        scene.ensure_origin_point(); // Origin at index 0
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() }); // index 1

        // Removing the Origin is a no-op.
        scene.remove_point(0);
        assert_eq!(scene.points.len(), 2, "the Origin is undeletable");
        assert!(scene.points[0].is_origin);

        // Removing a normal Point works.
        scene.remove_point(1);
        assert_eq!(scene.points.len(), 1, "a normal Point is removable");
        assert!(scene.points[0].is_origin);

        // Out-of-range removal is a no-op (never panics).
        scene.remove_point(99);
        assert_eq!(scene.points.len(), 1);

        // The Origin is hideable: toggling its hidden flag works.
        assert!(!scene.points[0].hidden);
        scene.toggle_point_hidden(0);
        assert!(scene.points[0].hidden, "the Origin can be hidden");
        scene.toggle_point_hidden(0);
        assert!(!scene.points[0].hidden, "and un-hidden");
    }

    /// Serde round-trip: a Scene whose node carries non-default `NodeGrids` plus a
    /// custom Point round-trips through JSON byte-equal (structurally).
    #[test]
    fn scene_with_grids_and_points_round_trips() {
        let mut node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape {
                    kind: ShapeKind::Box,
                    size_blocks: [1, 1, 1],
                    voxels_per_block: 8,
                    wall_blocks: 1,
                },
                material: MaterialChoice::Stone,
            },
        );
        node.grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let mut scene = Scene {
            nodes: vec![node],
            active: Some(NodePath::root_index(0)),
            master_block_lattice: false,
            master_voxel_grid: true,
            master_floor_grid: true,
            active_point: Some(1),
            ..Scene::default()
        };
        scene.ensure_origin_point();
        scene.add_point(Point {
            name: "Corner".to_string(),
            position_blocks: [3, 4, 5],
            plane_xz: false,
            plane_xy: true,
            plane_yz: true,
            axes: false,
            hidden: true,
            ..Point::default()
        });

        let json = serde_json::to_string_pretty(&scene).expect("serialise");
        let restored: Scene = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(scene, restored, "scene with grids + points round-trips");
        assert!(restored.nodes[0].grids.voxel_grid_on_faces);
        assert!(restored.nodes[0].grids.floor_grid);
        assert!(!restored.master_block_lattice);
        assert!(restored.master_voxel_grid);
        assert_eq!(restored.points.len(), 2);
        assert_eq!(restored.points[1].position_blocks, [3, 4, 5]);
    }

    /// Back-compat: an OLD serialized scene (no `grids`, no `points`, no masters)
    /// deserialises with the correct defaults — node grids all-off, masters at
    /// their struct defaults (lattice on, others off), empty points.
    #[test]
    fn old_scene_json_loads_with_grid_defaults() {
        let old_json = r#"{
            "nodes": [
                {
                    "name": "Box",
                    "transform": { "offset_blocks": [0, 0, 0] },
                    "operation": "Union",
                    "visible": true,
                    "content": {
                        "Tool": {
                            "shape": { "kind": "Box", "size_blocks": [2,2,2], "voxels_per_block": 8, "wall_blocks": 1 },
                            "material": "Stone"
                        }
                    }
                }
            ],
            "active": { "indices": [0] }
        }"#;
        let scene: Scene = serde_json::from_str(old_json).expect("old scene parses");
        assert_eq!(scene.nodes.len(), 1);
        assert_eq!(scene.nodes[0].grids, NodeGrids::default(), "grids default off");
        assert!(scene.master_block_lattice, "lattice master default on");
        assert!(!scene.master_voxel_grid && !scene.master_floor_grid);
        assert!(scene.points.is_empty(), "no points in the old document");
        assert_eq!(scene.active_point, None);
    }
}
