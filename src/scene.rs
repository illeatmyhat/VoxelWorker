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
    #[serde(default)]
    pub offset_blocks: [i32; 3],
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
    /// What the node is.
    pub content: NodeContent,
}

/// A node missing its `visible` flag in an older/partial config defaults to
/// visible (the common case — a hidden node is the exception, explicitly set).
fn default_visible() -> bool {
    true
}

impl Node {
    /// A visible, identity-placed, union node wrapping `content`.
    pub fn new(name: impl Into<String>, content: NodeContent) -> Self {
        Self {
            name: name.into(),
            transform: NodeTransform::identity(),
            operation: CombineOp::Union,
            visible: true,
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

/// The scene (assembly): a list of placed nodes resolved into the shared
/// [`VoxelGrid`] truth. ADR 0001's full model carries reusable `definitions` too;
/// step 2 added the flat node list plus the `active` selection that drives the
/// inspector. **Step 4** wires up `definitions` so a [`NodeContent::Instance`]
/// resolves the referenced [`AssemblyDef`] under its transform (reuse by
/// reference: a village of identical houses is one definition placed by N
/// instances).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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
}

impl Scene {
    /// A scene with a single node — the shape every one-node call site builds. The
    /// lone node is the active selection.
    pub fn single_node(node: Node) -> Self {
        Self {
            nodes: vec![node],
            definitions: Vec::new(),
            active: Some(NodePath::root_index(0)),
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
        node.transform.offset_blocks = [(existing as i32 + 1) * DEFAULT_INSTANCE_SPACING_BLOCKS, 0, 0];
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
    fn placed_extent_blocks(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        let mut min_corner = [i32::MAX; 3];
        let mut max_corner = [i32::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset, content| {
            let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                return;
            };
            any = true;
            for axis in 0..3 {
                let half_low = (size_blocks[axis] / 2) as i32;
                let low = world_offset[axis] - half_low;
                let high = low + size_blocks[axis] as i32;
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
    fn for_each_leaf(&self, visitor: &mut dyn FnMut([i32; 3], &NodeContent)) {
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
        parent_offset: [i32; 3],
        def_path: &mut Vec<DefId>,
        visitor: &mut dyn FnMut([i32; 3], &NodeContent),
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
                ((min_corner[0] + max_corner[0]) * voxels_per_block as i32) / 2,
                ((min_corner[1] + max_corner[1]) * voxels_per_block as i32) / 2,
                ((min_corner[2] + max_corner[2]) * voxels_per_block as i32) / 2,
            ],
            None => [0, 0, 0],
        };

        // Walk the whole tree (groups + instances recurse, composing world
        // translation down — ADR 0001 step 4). Each visited leaf is stamped under
        // its WORLD offset (× density) minus the composite recentre.
        self.for_each_leaf(&mut |world_offset, content| {
            let translation_voxels = [
                world_offset[0] * voxels_per_block as i32 - recentre_voxels[0],
                world_offset[1] * voxels_per_block as i32 - recentre_voxels[1],
                world_offset[2] * voxels_per_block as i32 - recentre_voxels[2],
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
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        let chunk_extent_voxels = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;

        // The chunk's half-open absolute-voxel box `[min, max)` per axis.
        let chunk_min_voxels = [
            chunk_coord[0] * chunk_extent_voxels,
            chunk_coord[1] * chunk_extent_voxels,
            chunk_coord[2] * chunk_extent_voxels,
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
        self.for_each_leaf(&mut |world_offset, content| {
            let translation_voxels = [
                world_offset[0] * voxels_per_block as i32,
                world_offset[1] * voxels_per_block as i32,
                world_offset[2] * voxels_per_block as i32,
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
    pub(crate) fn recentre_voxels(&self, voxels_per_block: u32) -> [i32; 3] {
        match self.placed_extent_blocks(voxels_per_block) {
            Some((min_corner, max_corner)) => [
                ((min_corner[0] + max_corner[0]) * voxels_per_block as i32) / 2,
                ((min_corner[1] + max_corner[1]) * voxels_per_block as i32) / 2,
                ((min_corner[2] + max_corner[2]) * voxels_per_block as i32) / 2,
            ],
            None => [0, 0, 0],
        }
    }

    /// The full composite extent in voxels (`size_blocks × density`) — the size the
    /// whole-region grids ([`resolve_region`], [`resolve_region_via_chunks`]) and
    /// the per-leaf local grids are seeded with.
    fn placed_region_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        let region = self.full_extent_blocks(voxels_per_block);
        [
            region.size_blocks[0] * voxels_per_block,
            region.size_blocks[1] * voxels_per_block,
            region.size_blocks[2] * voxels_per_block,
        ]
    }

    /// The inclusive range of chunk coordinates `[min_chunk, max_chunk]` whose
    /// half-open boxes cover the composite AABB in **absolute** voxel space.
    /// `None` when no leaf has an intrinsic size (no AABB to cover).
    fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        let (min_corner_blocks, max_corner_blocks) =
            self.placed_extent_blocks(voxels_per_block)?;
        let chunk_extent_voxels = (crate::renderer::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;

        let mut min_chunk = [0i32; 3];
        let mut max_chunk = [0i32; 3];
        for axis in 0..3 {
            let min_voxel = min_corner_blocks[axis] * voxels_per_block as i32;
            // The AABB is the half-open box `[min, max)`; its last occupied voxel
            // centre is at `max_voxel - 1 + 0.5`, so the highest chunk is the one
            // owning `max_voxel - 1`.
            let max_voxel = max_corner_blocks[axis] * voxels_per_block as i32;
            min_chunk[axis] = min_voxel.div_euclid(chunk_extent_voxels);
            max_chunk[axis] = (max_voxel - 1).div_euclid(chunk_extent_voxels);
        }
        Some((min_chunk, max_chunk))
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
    translation_voxels: [i32; 3],
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
#[allow(clippy::too_many_arguments)]
fn stamp_producer_into_chunk(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i32; 3],
    material_override: Option<u16>,
    producer: &dyn VoxelProducer,
    chunk_min_voxels: [i32; 3],
    chunk_max_voxels: [i32; 3],
) {
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local);

    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        // Absolute composite position (no recentre).
        voxel.world_position[0] += translation_voxels[0] as f32;
        voxel.world_position[1] += translation_voxels[1] as f32;
        voxel.world_position[2] += translation_voxels[2] as f32;

        // Keep only voxels whose centre lands in this chunk's half-open box. The
        // centre is at an `n + 0.5` position, so `floor` never lands on a boundary
        // integer — each voxel belongs to exactly one chunk.
        let in_chunk = (0..3).all(|axis| {
            let position = voxel.world_position[axis];
            position >= chunk_min_voxels[axis] as f32 && position < chunk_max_voxels[axis] as f32
        });
        if !in_chunk {
            continue;
        }

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
    fn boxed_block_positions(offset_x: i32, voxels_per_block: u32) -> std::collections::HashSet<[i64; 3]> {
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
        let n = 5i32; // 5 blocks apart: a 1-block box leaves a 4-block gap (disjoint).
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
        let shift = (n * voxels_per_block as i32) as f32; // N blocks → N×density voxels.
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
        let a = 3i32; // group offset
        let b = 2i32; // leaf offset within the group

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
        let t = 4i32;
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
        recentre_voxels: [i32; 3],
    ) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
        let mut multiset = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let key = [
                (voxel.world_position[0] - 0.5).round() as i64 + recentre_voxels[0] as i64,
                (voxel.world_position[1] - 0.5).round() as i64 + recentre_voxels[1] as i64,
                (voxel.world_position[2] - 0.5).round() as i64 + recentre_voxels[2] as i64,
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
        let make_tool = |kind, offset: [i32; 3], material| {
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
        let tool = |kind, size: [u32; 3], offset: [i32; 3], material| {
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
        let instance = |name: &str, offset: [i32; 3]| {
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
    // `offset_blocks` is `[i32; 3]` today (S4 widens to i64); 100_000 blocks is
    // comfortably in i32 range, and at density 16 lands the box ~1.6M voxels out.

    /// A far-offset node resolves to absolute voxel/chunk coordinates around
    /// 100_000 blocks: the box's voxels sit at absolute X ≈ 100_000 × density, the
    /// owning chunks are around `100_000 × density / chunk_extent`, and the box is
    /// genuinely placed far away (the absolute coords are NOT near the origin —
    /// only the recentred render path maps it home). Independent of any render math.
    #[test]
    fn far_offset_node_resolves_to_absolute_coords_near_100k() {
        let voxels_per_block = 16u32;
        let offset_blocks = 100_000i32;
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
            (crate::renderer::CHUNK_BLOCKS * voxels_per_block) as i32;
        let expected_chunk_x = (offset_blocks * voxels_per_block as i32) / chunk_extent_voxels;
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
            offset_blocks * voxels_per_block as i32,
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
}
