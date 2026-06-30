//! The scene (assembly) model — ADR 0001, sequence step 1.
//!
//! Today the app has exactly one producer, smuggled in through
//! [`GeometryParams`](crate::voxel::GeometryParams) (the SDF shape) plus a
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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::core_geom::MaterialChoice;
use crate::debug_clouds::DebugCloudField;
use crate::spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};
use crate::sketch::SketchSolid;
use crate::units::{ExactRational, Measurement};
use crate::voxel::{GeometryParams, SdfShape, VoxelGrid, VoxelProducer};

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

/// A node's LOCAL placement. v1 exposes integer block translation only, but the
/// type targets a full affine (translation + rotation + scale) so rotation /
/// scale (with voxel resampling) slot in later without a rewrite (ADR 0001
/// decision 3). In step 1 the offset is always `[0, 0, 0]`.
///
/// NOT `Copy`: it owns an optional boxed retained-measurement expression (the
/// parametric units layer, ADR 0003 §3f(0)), so it is `Clone` only. The canonical
/// `offset_voxels` is read by-field everywhere; the few sites that moved a whole
/// transform out of a `&Node` now `.clone()` it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NodeTransform {
    /// Translation in **voxels** at the document's density `d`
    /// ([`Scene::voxels_per_block`]) — the single canonical placement field
    /// (ADR 0003 §3f(0)). The planning unit is the voxel; "blocks" are a DERIVED
    /// overlay (a grid line every `d` voxels), exposed via the [`blocks`] /
    /// [`block_aligned`] accessors below — **not** a stored field. Sub-block
    /// placement (an offset not divisible by `d`) is the kit-authoring primitive;
    /// inter-part mating stays block-aligned via `offset_voxels % d == 0`.
    ///
    /// **64-bit world addressing (S4a, ADR 0002 Decision 2):** the offset is `i64`
    /// so far-apart nodes compose down the tree without overflow (a node placed at
    /// ±10⁹ blocks, or a deep nest summing past the i32 range, is exact). It enters
    /// the i64 placement sum at resolve as-is, with no rounding (the resolved grid
    /// *is* `d`).
    ///
    /// [`blocks`]: NodeTransform::blocks
    /// [`block_aligned`]: NodeTransform::block_aligned
    #[serde(default)]
    pub offset_voxels: [i64; 3],

    /// The RETAINED authored unit expression per axis (ADR 0003 §3f(0)).
    ///
    /// `offset_voxels` stays the canonical source of truth for ALL geometry /
    /// resolve; this is the parametric expression the user typed (e.g. `"3.5
    /// blocks"`), kept ALONGSIDE the voxels so a later density re-target is
    /// lossless (the same measurement re-evaluates at the new `d`). It is NOT read
    /// by resolve — only by the inspector (seed/undo) and a future density change.
    ///
    /// **Versioning:** `#[serde(default)]` makes this `None` on an OLD scene that
    /// predates the field, so old documents still load. The accessor
    /// [`offset_measurements`](NodeTransform::offset_measurements) SYNTHESISES a
    /// pure-voxel measurement from `offset_voxels` when this is `None`, so the
    /// retained expression is always correct (just non-parametric — a whole-voxel
    /// count — for a placement authored before the field existed or via a path
    /// that has no expression, e.g. a drag gizmo).
    ///
    /// **Boxed** so the common (`None`) case keeps [`NodeTransform`] pointer-small:
    /// three `Measurement`s are ~120 bytes, which would otherwise bloat every
    /// `Node` (and the arena's `Leaf(Node)` variant). The box is allocated only
    /// when a real authored block expression is retained. `serde` treats
    /// `Option<Box<T>>` transparently, so the on-disk shape is unchanged (`null`
    /// or the three-measurement array).
    #[serde(default)]
    offset_measurements: Option<Box<[Measurement; 3]>>,
    // future: rotation, scale → a general affine.
}

impl NodeTransform {
    /// The identity transform (zero offset) — the only transform step 1 uses.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Build a transform from a whole-**block** translation at density
    /// `voxels_per_block` (`offset_voxels = blocks · d`). The block-valued
    /// convenience constructor used by demos, tests and `GroupSpec` placement
    /// (ADR 0003 §3f(0)). The inspector's Offset path now authors through
    /// [`from_measurements`](NodeTransform::from_measurements) (blocks + voxels);
    /// this remains the terse whole-block entry point. It retains each axis as a
    /// whole-block measurement, so a later density re-target scales it losslessly.
    pub fn from_blocks(blocks: [i64; 3], voxels_per_block: u32) -> Self {
        // Clamp density to ≥1 like every resolve site, so a 0-density doc can't
        // multiply placement to zero / mis-scale.
        let density = voxels_per_block.max(1) as i64;
        let offset_voxels = [blocks[0] * density, blocks[1] * density, blocks[2] * density];
        // Retain a whole-BLOCK measurement per axis (no voxel remainder), so a later
        // density re-target scales the block count losslessly — but normalise the
        // all-zero case to `None` so a zero placement matches a fresh identity.
        let measurements = [
            Measurement::new(ExactRational::from_integer(blocks[0] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[1] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[2] as i128), 0),
        ];
        Self {
            offset_voxels,
            offset_measurements: Self::retained_or_none(measurements, offset_voxels),
        }
    }

    /// Build a transform from a per-axis authored [`Measurement`] at density
    /// `voxels_per_block` (ADR 0003 §3f(0)). The canonical voxel offset is DERIVED
    /// via [`Measurement::to_voxels`]; the measurements are RETAINED for lossless
    /// density re-targeting and for the inspector to seed/undo the exact authored
    /// expression.
    ///
    /// **Self-consistency invariant:** the result NEVER carries a retained
    /// measurement that disagrees with `offset_voxels`. On the UI path every axis
    /// lands on a whole voxel (the inspector validates before emitting), so the
    /// authored measurement is kept verbatim. On the LOSSY density-retarget path
    /// (`SetDensity` re-evaluating an expression at a `d` it no longer divides
    /// cleanly, e.g. `3.5 blocks` at `d = 15`), the offending axis is floored to a
    /// whole voxel AND its retained measurement is RESYNTHESISED to the pure-voxel
    /// form of that floored value — so the canonical voxels and the retained
    /// expression always agree (the block-term parametricity is lost for that axis,
    /// which is the honest outcome of a non-dividing re-target). Landing axes keep
    /// their authored (block-parametric) measurement.
    pub fn from_measurements(measurements: [Measurement; 3], voxels_per_block: u32) -> Self {
        // Per axis, derive the voxel count AND the measurement to retain. A landing
        // axis keeps its authored measurement; a non-landing axis floors and
        // resynthesises to the pure-voxel form of the floored value so the two can
        // never disagree.
        let resolve_axis = |measurement: Measurement| -> (i64, Measurement) {
            match measurement.to_voxels(voxels_per_block) {
                Ok(voxels) => (voxels, measurement),
                Err(crate::units::MeasurementError::BlockTermNotWholeVoxels {
                    nearest_floor_voxels,
                    ..
                }) => (nearest_floor_voxels, Measurement::from_voxels(nearest_floor_voxels)),
                Err(crate::units::MeasurementError::ZeroDensity) => {
                    let voxels = measurement.voxel_term();
                    (voxels, Measurement::from_voxels(voxels))
                }
            }
        };
        let (voxels_x, retained_x) = resolve_axis(measurements[0]);
        let (voxels_y, retained_y) = resolve_axis(measurements[1]);
        let (voxels_z, retained_z) = resolve_axis(measurements[2]);
        let offset_voxels = [voxels_x, voxels_y, voxels_z];
        let retained = [retained_x, retained_y, retained_z];
        Self {
            offset_voxels,
            offset_measurements: Self::retained_or_none(retained, offset_voxels),
        }
    }

    /// Normalise the retained measurements to `None` when they carry NO parametric
    /// content beyond the derived voxel count — i.e. every axis is exactly the
    /// pure-voxel measurement [`Measurement::from_voxels`] of its derived voxels.
    /// This keeps a placement with no real authored block expression (a zero
    /// offset, a drag, a `from_voxels` round-trip) in the SAME canonical form as a
    /// freshly-built / freshly-loaded transform (`None`), so apply→undo→apply is
    /// byte-identical and serde does not gain a redundant `Some([...])` husk. A
    /// real block expression (e.g. `3 blocks`, `3.5 blocks`) does NOT synthesise
    /// from its voxel count, so it is retained as `Some` for lossless re-targeting.
    fn retained_or_none(
        measurements: [Measurement; 3],
        offset_voxels: [i64; 3],
    ) -> Option<Box<[Measurement; 3]>> {
        let is_synthesisable = (0..3).all(|axis| {
            measurements[axis] == Measurement::from_voxels(offset_voxels[axis])
        });
        if is_synthesisable {
            None
        } else {
            Some(Box::new(measurements))
        }
    }

    /// The RETAINED per-axis authored measurement (ADR 0003 §3f(0)).
    ///
    /// When the placement carries no stored expression (an OLD scene predating the
    /// field, or a transform built without one), this SYNTHESISES a pure-voxel
    /// measurement equal to `offset_voxels` per axis — correct (it re-evaluates
    /// back to the same voxels at any density), just non-parametric for a block
    /// re-target. The canonical `offset_voxels` always wins for geometry; this is
    /// retention/display only.
    pub fn offset_measurements(&self) -> [Measurement; 3] {
        match &self.offset_measurements {
            Some(measurements) => **measurements,
            None => [
                Measurement::from_voxels(self.offset_voxels[0]),
                Measurement::from_voxels(self.offset_voxels[1]),
                Measurement::from_voxels(self.offset_voxels[2]),
            ],
        }
    }

    /// Whether this transform carries a GENUINELY retained authored expression
    /// (the stored field is `Some`) versus a placement whose measurement is only
    /// SYNTHESISED from `offset_voxels` (the field is `None` — an old document, a
    /// drag, a pure-voxel offset). The density re-target (`SetDensity`) uses this
    /// to decide between RE-EVALUATING the authored block expression at the new
    /// density (lossless block scaling, exact voxel terms) and the legacy integer
    /// rescale that preserves a non-parametric offset's physical position.
    pub fn has_retained_measurements(&self) -> bool {
        self.offset_measurements.is_some()
    }

    /// The whole-**block** view of this placement (the derived block overlay,
    /// ADR 0003 §3f(0)): the floor of `offset_voxels / d` componentwise (the same
    /// single floor rule the extent derivations use, see
    /// [`world_block_corner_floor`]). EXACT while placement is block-aligned — which
    /// it is today; for future negative sub-voxel offsets the floor is the correct
    /// (block-containing) view.
    pub fn blocks(&self, voxels_per_block: u32) -> [i64; 3] {
        world_block_corner_floor(self.offset_voxels, voxels_per_block)
    }

    /// Whether this placement sits on the whole-block lattice — the connector /
    /// joint mating predicate `offset_voxels % d == 0` per axis (ADR 0003 §3f(0)
    /// / §3i "block-aligned where you mate").
    pub fn block_aligned(&self, voxels_per_block: u32) -> bool {
        // Clamp density to ≥1 so a 0-density doc can't panic on `% 0`.
        let density = voxels_per_block.max(1) as i64;
        self.offset_voxels.iter().all(|&v| v.rem_euclid(density) == 0)
    }
}

/// The whole-**block** corner of a world VOXEL offset: `floor(offset_voxels / d)`
/// per axis via `div_euclid` (the codebase convention, e.g. `main.rs`'s
/// `point_add_position_blocks`). The single owner of the voxel→block-corner rule —
/// [`NodeTransform::blocks`] and both extent derivations
/// ([`Scene::placed_extent_blocks`], [`Scene::node_subtree_extent_blocks`]) route
/// through it.
///
/// This is EXACT while placement is block-aligned — which it is today (every offset
/// is a block multiple); Slice 2's sub-voxel placement makes it a truncating
/// (floor) view, correct for the LOW corner of a leaf box but requiring outward
/// (ceil) rounding for the HIGH corner at the call sites (see those).
fn world_block_corner_floor(world_offset_voxels: [i64; 3], voxels_per_block: u32) -> [i64; 3] {
    let density = voxels_per_block.max(1) as i64;
    [
        world_offset_voxels[0].div_euclid(density),
        world_offset_voxels[1].div_euclid(density),
        world_offset_voxels[2].div_euclid(density),
    ]
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
    /// A **sketch → extrude → volume** producer (ADR 0003 §3i, Slice 2a): a
    /// grid-aligned plane + closed polygon profile extruded a whole number of
    /// voxels, plus the single material it stamps. Added **alongside** [`Tool`]
    /// (not replacing it) — the §3i sketch-to-volume authoring atom over which
    /// primitives become sugar later. It resolves through the SAME stamp /
    /// `CombineOp` / chunk path as [`Tool`]. Both producers center-emit their grids
    /// at the origin and are placed purely by their world voxel offset (no per-leaf
    /// lattice shift) — see [`Scene::recentre_voxels_for_resolve`].
    ///
    /// [`Tool`]: NodeContent::Tool
    /// [`Part`]: NodeContent::Part
    SketchTool {
        /// The sketch + operation to resolve.
        producer: SketchSolid,
        /// The single material this node stamps onto its voxels.
        material: MaterialChoice,
    },
    /// A static voxel body, dropped in as-is.
    Part(Part),
    /// An owned, one-off sub-assembly. **ADR 0003 Phase B5:** a Group owns its
    /// children by **identity** — the ordered spine of child [`NodeId`]s — while the
    /// child `Node`s themselves live in the scene-wide [`Scene::arena`]. The `Vec`
    /// order IS document order (resolved later-wins on overlap); the arena is fetched
    /// from but never iterated to produce a walk. **Not resolved in step 1** (step 4).
    Group(Vec<NodeId>),
    /// A reuse-by-reference of a definition. **Not resolved in step 1** (step 4).
    Instance(DefId),
}

/// The single ported producer of a one-leaf scene — the scope of the GPU fog path
/// (ADR 0007 P1). `Clouds` carries only its seed; the resolver fills the cloud's
/// region-derived dimensions from the resolved grid. Anything else (empty / multi-leaf /
/// a generic `Part` / `Group` / `Instance`) yields `None` from [`Scene::single_producer`]
/// and stays on the CPU fog path.
#[derive(Debug, Clone)]
pub enum SingleProducerKind {
    /// An SDF primitive Tool.
    Sdf(SdfShape),
    /// A sketch extrude/revolve solid.
    Sketch(SketchSolid),
    /// A `DebugClouds` Part (corner-anchored, recentre `[0,0,0]`).
    Clouds { seed: u32 },
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

/// Default `true` for the scene-wide grid masters (issue #29 grid-rework fix: all
/// three masters default ON so enabling a per-object toggle shows immediately,
/// while the per-object flags stay default OFF — the default view is still clean).
fn default_master_grid() -> bool {
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
    /// The top-level assembly's nodes, as an **ordered spine of [`NodeId`]s**
    /// (ADR 0003 Phase B5). Resolved in this order (later nodes win on overlap under
    /// [`CombineOp::Union`]); the `Node`s themselves live in [`arena`](Self::arena).
    /// **Golden-critical:** every tree walk iterates THIS spine (and each
    /// [`NodeContent::Group`]'s spine) for order, fetching content from the arena —
    /// never iterate the arena to produce a walk (that visits in id order and would
    /// reorder later-wins material on overlap).
    #[serde(default)]
    pub roots: Vec<NodeId>,
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

impl Default for Scene {
    /// An empty scene with the issue-#29 master defaults — **all three masters ON**
    /// (grid-rework fix), while every node's per-object grid flag stays default OFF,
    /// so enabling a per-object toggle shows immediately yet the default view is
    /// clean. No Points yet (the Origin is synthesized on the load path via
    /// [`ensure_origin_point`](Self::ensure_origin_point)).
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            arena: BTreeMap::new(),
            definitions: Vec::new(),
            active: None,
            points: Vec::new(),
            master_block_lattice: true,
            master_voxel_grid: true,
            master_floor_grid: true,
            active_point: None,
            next_node_id: 0,
            voxels_per_block: default_density(),
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

    /// The node at `path`, walking from [`nodes`](Self::nodes) down through Group
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
    /// it (the same union [`placed_extent_blocks`] forms scene-wide, but rooted at
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
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block);
        let density = voxels_per_block.max(1) as i64;
        let mut pivot = [0.0f32; 3];
        let mut extent = [0.0f32; 3];
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

    /// The per-object **block lattice box** for the node at `path`, in the SAME
    /// recentred render frame the resolved voxels live in (issue #29 S3). Returns
    /// `(min_corner, max_corner)` in voxels.
    ///
    /// The box is the node's block-aligned voxel AABB **expanded out to enclosing
    /// whole blocks** — i.e. the union of every visible leaf under the node, each
    /// leaf snapped to the whole-block range `[off − floor(size/2), … + size)` (the
    /// same split [`node_subtree_extent_blocks`] forms), then scaled by `density`
    /// and shifted by `− recentre_voxels_for_resolve`. Because the corners are taken
    /// in WHOLE blocks before scaling, a sub-block (1-voxel) translate that crosses a
    /// block boundary moves the enclosing-block box by exactly one whole block — the
    /// spec's "a 1-voxel translate adds/removes a whole block" requirement falls out
    /// of the expand-to-block on the shifted box.
    ///
    /// For a Group / Instance node the box is the union of all leaves under it.
    /// A size-less node (a Part-only / empty subtree, or a path that descends
    /// through a non-Group) returns `None` — there is no block lattice to draw.
    pub fn node_block_lattice_box_recentred(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let (min_corner, max_corner) = self.node_subtree_extent_blocks(path, voxels_per_block)?;
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block);
        let density = voxels_per_block.max(1) as i64;
        let mut min_box = [0.0f32; 3];
        let mut max_box = [0.0f32; 3];
        for axis in 0..3 {
            // Whole-block corners → voxels (exact), then into the recentred frame.
            min_box[axis] = (min_corner[axis] * density - recentre[axis]) as f32;
            max_box[axis] = (max_corner[axis] * density - recentre[axis]) as f32;
        }
        Some((min_box, max_box))
    }

    /// The block-aligned AABB (`min_corner, max_corner`, whole blocks) of the
    /// subtree rooted at `path` — the union of every visible leaf under that node,
    /// each leaf spanning `[off − floor(size/2), off − floor(size/2) + size)` (the
    /// same split [`placed_extent_blocks`] uses scene-wide). The accumulated world
    /// offset down to `path` seeds the walk so a Group/Instance child is measured at
    /// its world location. `None` when the subtree has no intrinsic-size leaf.
    fn node_subtree_extent_blocks(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([i64; 3], [i64; 3])> {
        // Accumulate the world VOXEL offset of every node ABOVE the target (the
        // parent offset), and grab the target node itself. `walk_nodes` below
        // re-adds the target's own offset (also voxels), so we must stop
        // accumulating at its parent. Walk the id-spine for ORDER, fetch content
        // from the arena (ADR 0003 B5).
        let mut siblings: &[NodeId] = &self.roots;
        let mut parent_offset_voxels = [0i64; 3];
        let mut target: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                target = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                parent_offset_voxels = [
                    parent_offset_voxels[0] + node.transform.offset_voxels[0],
                    parent_offset_voxels[1] + node.transform.offset_voxels[1],
                    parent_offset_voxels[2] + node.transform.offset_voxels[2],
                ];
                siblings = children;
            } else {
                return None;
            }
        }
        let target = target?;
        if !target.visible {
            return None;
        }
        let target_id = target.id;

        // Union the leaf boxes under the target. `walk_nodes` adds the target's own
        // voxel offset to `parent_offset_voxels`, giving the leaf its true world
        // location. The single-element id spine carries the target itself (ADR 0003
        // B5).
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        let mut def_path: Vec<DefId> = Vec::new();
        self.walk_nodes(
            &[target_id],
            parent_offset_voxels,
            &mut def_path,
            &mut |world_offset_voxels, content, _grid_on_faces| {
                let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                    return;
                };
                any = true;
                // The leaf's whole-block offset, via the single floor rule.
                let world_blocks = world_block_corner_floor(world_offset_voxels, voxels_per_block);
                for axis in 0..3 {
                    // Corner-anchored: the offset block IS the low corner, so the leaf
                    // spans `[off_block, off_block + size)` blocks.
                    let low = world_blocks[axis];
                    let high = low + size_blocks[axis] as i64;
                    min_corner[axis] = min_corner[axis].min(low);
                    max_corner[axis] = max_corner[axis].max(high);
                }
            },
        );
        any.then_some((min_corner, max_corner))
    }

    /// The PRODUCER-TRUE voxel AABB (`min_corner, max_corner`, in voxels) of the
    /// subtree rooted at `path` — the union of every visible leaf under that node,
    /// each leaf spanning the center-emitted `[off·d − grid/2, off·d + grid/2)` (the
    /// exact frame [`placed_extent_voxels`] forms scene-wide). This is the frame the
    /// composite recentre and the resolved voxels live in, so the gizmo pivot derived
    /// from it lands exactly on the object. `None` when the subtree has no
    /// intrinsic-size leaf. Mirrors [`node_subtree_extent_blocks`] but in voxels with
    /// no block-floor split (so odd sizes are centred, not snapped).
    fn node_subtree_extent_voxels(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([i64; 3], [i64; 3])> {
        let mut siblings: &[NodeId] = &self.roots;
        let mut parent_offset_voxels = [0i64; 3];
        let mut target: Option<&Node> = None;
        for (depth, &index) in path.indices.iter().enumerate() {
            let &child_id = siblings.get(index)?;
            let node = self.arena.get(&child_id)?;
            let is_last = depth + 1 == path.indices.len();
            if is_last {
                target = Some(node);
            } else if let NodeContent::Group(children) = &node.content {
                parent_offset_voxels = [
                    parent_offset_voxels[0] + node.transform.offset_voxels[0],
                    parent_offset_voxels[1] + node.transform.offset_voxels[1],
                    parent_offset_voxels[2] + node.transform.offset_voxels[2],
                ];
                siblings = children;
            } else {
                return None;
            }
        }
        let target = target?;
        if !target.visible {
            return None;
        }
        let target_id = target.id;

        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        let mut def_path: Vec<DefId> = Vec::new();
        self.walk_nodes(
            &[target_id],
            parent_offset_voxels,
            &mut def_path,
            &mut |world_offset_voxels, content, _grid_on_faces| {
                let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block) else {
                    return;
                };
                any = true;
                for axis in 0..3 {
                    // Corner-anchored span `[off, off + grid)` (offset is the low corner).
                    let grid = grid_voxels[axis];
                    let low = world_offset_voxels[axis];
                    let high = low + grid;
                    min_corner[axis] = min_corner[axis].min(low);
                    max_corner[axis] = max_corner[axis].max(high);
                }
            },
        );
        any.then_some((min_corner, max_corner))
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

    /// The whole-block extent of the scene: the per-axis size of the bounding box
    /// that encompasses every placed leaf node (ADR 0001 step 3). Each leaf
    /// occupies `block-offset ± size/2` (its placement's derived block view,
    /// ADR 0003 §3f(0)); the composite extent is the union of
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
    /// `size_blocks` is CORNER-ANCHORED at its block-offset (the derived block view of
    /// its voxel placement, ADR 0003 §3f(0)) and so spans `[offset, offset + size]`.
    /// `None` when no leaf has an intrinsic size (a Part-only scene). Drives
    /// [`full_extent_blocks`] (the whole-block size readout) and the block-lattice
    /// overlay extent.
    ///
    /// CORNER-ANCHORING: the offset block is the LOW corner (no `± size/2` split), so
    /// the block frame matches the corner-anchored producer voxel frame exactly.
    fn placed_extent_blocks(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces| {
            let Some(size_blocks) = leaf_size_blocks(content, voxels_per_block) else {
                return;
            };
            any = true;
            // The leaf's whole-block offset, via the single floor rule.
            let world_blocks = world_block_corner_floor(world_offset_voxels, voxels_per_block);
            for axis in 0..3 {
                // Corner-anchored: the offset block IS the low corner.
                let low = world_blocks[axis];
                let high = low + size_blocks[axis] as i64;
                min_corner[axis] = min_corner[axis].min(low);
                max_corner[axis] = max_corner[axis].max(high);
            }
        });
        any.then_some((min_corner, max_corner))
    }

    /// Walk the whole node tree depth-first, invoking
    /// `visitor(world_offset_voxels, leaf)` once for every **visible leaf** (`Tool`
    /// / `Part`) with its accumulated **world** VOXEL offset (`parent_offset +
    /// node.offset_voxels`, summed down the tree — translation-only composition,
    /// ADR 0001 step 4; voxels at the document density, ADR 0003 §3f(0)).
    ///
    /// `Group` children inherit the group's world offset; an `Instance(def)` resolves
    /// the referenced [`AssemblyDef`]'s children under the instance's world offset, so
    /// the SAME definition placed by N instances is visited N times at N locations
    /// (the village-of-reused-houses case). The cycle guard (an `Instance` may not
    /// reference an ancestor definition) lives in [`walk_nodes`].
    ///
    /// [`walk_nodes`]: Self::walk_nodes
    /// The visitor receives, per visible leaf: its accumulated world VOXEL
    /// offset, its content, and its own `grids.voxel_grid_on_faces` flag (issue
    /// #29 S4 — the resolver ORs [`crate::voxel::GRID_OVERLAY_BIT`] into the
    /// leaf's stamped `material_id` when this is set, so the on-face voxel grid
    /// travels with each voxel through chunk bucketing).
    fn for_each_leaf(&self, visitor: &mut dyn FnMut([i64; 3], &NodeContent, bool)) {
        let mut def_path: Vec<DefId> = Vec::new();
        self.walk_nodes(&self.roots, [0, 0, 0], &mut def_path, visitor);
    }

    /// `Some` iff this scene resolves to EXACTLY ONE ported producer leaf — an SDF Tool,
    /// a sketch solid, or a `DebugClouds` Part — the scope of the GPU fog path (ADR 0007
    /// P1). Returns `None` for empty, multi-leaf, or unported (generic `Part` / `Group` /
    /// `Instance`) scenes, which stay on the CPU fog path.
    pub fn single_producer(&self) -> Option<SingleProducerKind> {
        let mut leaf_count = 0usize;
        let mut producer: Option<SingleProducerKind> = None;
        self.for_each_leaf(&mut |_offset, content, _grid_on_faces| {
            leaf_count += 1;
            producer = match content {
                NodeContent::Tool { shape, .. } => Some(SingleProducerKind::Sdf(shape.clone())),
                NodeContent::SketchTool { producer, .. } => {
                    Some(SingleProducerKind::Sketch(producer.clone()))
                }
                NodeContent::Part(Part::DebugClouds { seed }) => {
                    Some(SingleProducerKind::Clouds { seed: *seed })
                }
                // An unported leaf (a generic voxel Part); not GPU-resolvable in P1.
                _ => None,
            };
        });
        // Exactly one leaf AND it was a ported kind.
        if leaf_count == 1 {
            producer
        } else {
            None
        }
    }

    /// Collect every visible leaf as a [`LeafProducer`] (ADR 0010 E2): its world voxel
    /// offset, a boxed [`VoxelProducer`], and its single-material override id. This is the
    /// op-stack the two-layer classifier / boundary-resolve evaluate over — the SAME
    /// leaves [`resolve_chunk_rebased`](Self::resolve_chunk_rebased) stamps, in the SAME
    /// document (walk) order, so the two-layer round-trip composes identically (later-wins
    /// Union on overlap). A region-sized Part (the cloud field) is sized to the composite
    /// `placed_region_dimensions` exactly as the dense chunk resolve sizes it.
    ///
    /// `pub(crate)` — the evaluator seam ADR 0010 E2's [`crate::two_layer_store`] reads;
    /// the dense store keeps using the private [`for_each_leaf`](Self::for_each_leaf).
    pub(crate) fn leaf_producers(&self, voxels_per_block: u32) -> Vec<LeafProducer> {
        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut leaves = Vec::new();
        self.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces| {
            let (material, producer): (Option<crate::core_geom::BlockId>, Box<dyn VoxelProducer>) =
                match content {
                    NodeContent::Tool { shape, material } => {
                        (material_id_for(*material), Box::new(shape.clone()))
                    }
                    NodeContent::SketchTool { producer, material } => {
                        (material_id_for(*material), Box::new(producer.clone()))
                    }
                    NodeContent::Part(Part::DebugClouds { seed }) => (
                        None,
                        Box::new(DebugCloudField {
                            dimensions: region_dimensions,
                            seed: *seed,
                        }),
                    ),
                    NodeContent::Group(_) | NodeContent::Instance(_) => return,
                };
            leaves.push(LeafProducer {
                world_offset_voxels,
                producer,
                material,
            });
        });
        leaves
    }

    /// Recursive worker for [`for_each_leaf`](Self::for_each_leaf). `parent_offset`
    /// is the accumulated world VOXEL offset of the assembly that owns `nodes`;
    /// `def_path` is the stack of definition ids currently being expanded (for the
    /// cycle guard — an `Instance` that would re-enter a definition already on the
    /// path is skipped instead of recursing forever).
    fn walk_nodes(
        &self,
        spine: &[NodeId],
        parent_offset: [i64; 3],
        def_path: &mut Vec<DefId>,
        visitor: &mut dyn FnMut([i64; 3], &NodeContent, bool),
    ) {
        // GOLDEN-CRITICAL (ADR 0003 B5): iterate the id-spine for ORDER (document
        // order = later-wins on overlap), fetching each node's content from the
        // arena. NEVER iterate the arena to produce this walk — that visits in id
        // order and would reorder Union material on overlap, moving the goldens.
        for &node_id in spine {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.visible {
                continue;
            }
            let world_offset_voxels = [
                parent_offset[0] + node.transform.offset_voxels[0],
                parent_offset[1] + node.transform.offset_voxels[1],
                parent_offset[2] + node.transform.offset_voxels[2],
            ];
            match &node.content {
                NodeContent::Tool { .. }
                | NodeContent::SketchTool { .. }
                | NodeContent::Part(_) => {
                    visitor(world_offset_voxels, &node.content, node.grids.voxel_grid_on_faces);
                }
                NodeContent::Group(children) => {
                    self.walk_nodes(children, world_offset_voxels, def_path, visitor);
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
                    self.walk_nodes(&def.children, world_offset_voxels, def_path, visitor);
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

        // The region grid is sized in the PRODUCER VOXEL FRAME (corner-anchoring):
        // the recentred composite occupies exactly `[region_low, region_low + D)` with
        // `D = max_v − min_v` (`placed_extent_voxels`) and `region_low = min_v −
        // recentre`, so a block-framed region (`size·d`) would clip a parity-mismatched
        // multi-leaf composite. For a chunkable scene we IGNORE the passed-in block
        // `region` for sizing and use the voxel span; the explicit `region` argument
        // still sizes a Part-only scene (which has no composite voxel extent).
        let region_dimensions = match self.placed_extent_voxels(voxels_per_block) {
            Some(_) => self.placed_region_dimensions(voxels_per_block),
            None => [
                region.size_blocks[0] * voxels_per_block,
                region.size_blocks[1] * voxels_per_block,
                region.size_blocks[2] * voxels_per_block,
            ],
        };
        let mut output = VoxelGrid::new(region_dimensions);

        // Recentre the composite so its world positions sit symmetrically about
        // the origin (what the renderer + camera auto-frame assume). Each producer
        // CORNER-ANCHORS its grid (local span `[0, grid)`); a leaf's low corner in the
        // composite's voxel space is `offset_voxels`, and the whole composite's centre
        // is `(min + max).div_euclid(2)` (producer-true voxel frame). Subtracting that
        // centre from every node's translation lands the composite centred in `output`.
        // A Part-only scene (e.g. `DebugClouds`) has no composite extent, so this is
        // `[0,0,0]` and the field stays CORNER-anchored at `[0, region)` — the shipped
        // convention (see `part_only_cloud_at_odd_density_drops_no_voxels` /
        // `mixed_tool_and_cloud_resolve_in_one_frame`). ADR 0008: the recentre is CARRIED on
        // the grid (below), so every consumer decodes correctly without re-deriving the
        // frame as `floor(dim/2)` (the assumption that dropped the corner-anchored cloud fog).
        let recentre_voxels = self.recentre_voxels_for_resolve(voxels_per_block);
        output.recentre_voxels = recentre_voxels;

        // Walk the whole tree (groups + instances recurse, composing world
        // translation down — ADR 0001 step 4). Each visited leaf is stamped under
        // its WORLD voxel offset minus the composite recentre. The offset is
        // already voxels at the document density (ADR 0003 §3f(0)), so it enters
        // the sum as-is. All of this is in i64 (S4a) so a far-placed node composes
        // without overflow; the result is downcast to f32 inside the stamp (the
        // render frame stays f32 — S4b makes the far case byte-identical via origin
        // rebasing).
        self.for_each_leaf(&mut |world_offset_voxels, content, grid_on_faces| {
            // Every producer corner-anchors its grid at its world voxel offset (the low
            // corner); the recentre (from the producer-true voxel frame) symmetrises the
            // composite about the origin for ALL size·d parities, so no per-leaf lattice
            // shift is needed — a leaf simply sits at its world voxel offset.
            let translation_voxels = [
                world_offset_voxels[0] - recentre_voxels[0],
                world_offset_voxels[1] - recentre_voxels[1],
                world_offset_voxels[2] - recentre_voxels[2],
            ];
            match content {
                NodeContent::Tool { shape, material } => {
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        material_id_for(*material),
                        // Issue #29 S4: OR the on-face-grid flag bit onto every
                        // stamped voxel iff this node opted in, so the bit travels
                        // with each voxel (and survives chunk bucketing).
                        grid_on_faces,
                        shape,
                        voxels_per_block,
                    );
                }
                NodeContent::SketchTool { producer, material } => {
                    // The sketch producer self-sizes its origin-centred grid exactly
                    // like SdfShape, so it stamps through the same path at the plain
                    // `translation_voxels` (world offset minus recentre, no shift).
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        material_id_for(*material),
                        grid_on_faces,
                        producer,
                        voxels_per_block,
                    );
                }
                NodeContent::Part(Part::DebugClouds { seed }) => {
                    let producer = DebugCloudField {
                        // The cloud field sizes itself from the region (today's
                        // behaviour resolved it at the shape's grid dimensions).
                        dimensions: region_dimensions,
                        seed: *seed,
                    };
                    stamp_producer(
                        &mut output,
                        region_dimensions,
                        translation_voxels,
                        // A Part brings its own per-voxel materials; today the
                        // cloud field emits material 0, so the stamp keeps that.
                        None,
                        // Issue #29 S4: still OR the flag bit per-voxel when this
                        // node wants the on-face grid (independent of material).
                        grid_on_faces,
                        &producer,
                        voxels_per_block,
                    );
                }
                // `for_each_leaf` only ever yields leaf content (Tool / SketchTool /
                // Part); the interior kinds were already recursed through by the walk.
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
    /// [`crate::core_geom::CHUNK_BLOCKS`]); one chunk therefore spans
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
        let chunk_extent_voxels = (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

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
        let chunk_box = VoxelAabb::new(chunk_min_voxels, chunk_max_voxels);
        self.for_each_leaf(&mut |world_offset_voxels, content, grid_on_faces| {
            // Issue #27 S3 optimisation: skip a leaf whose world-AABB doesn't touch
            // this chunk, so resolving one chunk costs ~the leaves that overlap it
            // (not the whole tree). This is BIT-IDENTICAL to stamping-then-clipping:
            // the leaf's AABB `[off·d − grid/2, off·d + grid/2)` is the exact span of
            // its voxel centres, and `stamp_producer_into_chunk` keeps only centres
            // inside `[chunk_min, chunk_max)`; if those two half-open boxes don't
            // intersect, the stamp would have clipped EVERY voxel anyway. A
            // region-spanning leaf (a Part, `leaf_size_blocks` → `None`) has no
            // localisable AABB, so it is never skipped (it may emit anywhere).
            if let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block) {
                let mut leaf_min = [0i64; 3];
                let mut leaf_max = [0i64; 3];
                for axis in 0..3 {
                    // The producer corner-anchors its grid, so placed at the world
                    // voxel offset (its low corner) it spans `[off, off + grid)`. Using
                    // the producer-true grid (exact emitted voxels, NOT block-rounded)
                    // keeps the skip AABB bit-identical to stamping-then-clipping.
                    let grid = grid_voxels[axis];
                    leaf_min[axis] = world_offset_voxels[axis];
                    leaf_max[axis] = leaf_min[axis] + grid;
                }
                if !VoxelAabb::new(leaf_min, leaf_max).intersects(&chunk_box) {
                    return;
                }
            }
            let translation_voxels = world_offset_voxels;
            let (material_override, producer): (
                Option<crate::core_geom::BlockId>,
                Box<dyn VoxelProducer>,
            ) = match content
            {
                NodeContent::Tool { shape, material } => {
                    // `SdfShape` is no longer `Copy` (owns an optional boxed retained
                    // size); clone it into the producer box.
                    (material_id_for(*material), Box::new(shape.clone()))
                }
                NodeContent::SketchTool { producer, material } => {
                    (material_id_for(*material), Box::new(producer.clone()))
                }
                NodeContent::Part(Part::DebugClouds { seed }) => (
                    None,
                    Box::new(DebugCloudField {
                        dimensions: region_dimensions,
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
                // Issue #29 S4: OR the on-face-grid flag bit onto each kept voxel
                // iff this node opted in, so the bit travels through the chunked
                // render path exactly as it does through `resolve_region`.
                grid_on_faces,
                producer.as_ref(),
                voxels_per_block,
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
    ///
    /// Derived from the **producer-true voxel frame** ([`placed_extent_voxels`]) —
    /// the exact span the producers center-emit — NOT the block-floored frame. This
    /// makes the composite centre coincide with the producers' own centres for ALL
    /// `size·d` parities (including odd size at density 1), so no per-leaf lattice
    /// shift is needed.
    ///
    /// [`placed_extent_voxels`]: Self::placed_extent_voxels
    pub fn recentre_voxels_for_resolve(&self, voxels_per_block: u32) -> [i64; 3] {
        match self.placed_extent_voxels(voxels_per_block) {
            // FLOOR division (`div_euclid`), NOT truncation: for an odd composite span
            // `(min + max)` is odd, and `/` rounds toward zero — which biases a
            // negative-X composite the OPPOSITE way from a positive-X one, breaking
            // +X/−X symmetry. `div_euclid(2)` always rounds toward −∞, so the recentre
            // direction is consistent regardless of where the composite sits.
            Some((min_corner, max_corner)) => [
                (min_corner[0] + max_corner[0]).div_euclid(2),
                (min_corner[1] + max_corner[1]).div_euclid(2),
                (min_corner[2] + max_corner[2]).div_euclid(2),
            ],
            None => [0i64; 3],
        }
    }

    /// The full composite extent in voxels — the size the whole-region grids
    /// ([`resolve_region`], [`resolve_region_via_chunks`]) are seeded with. The chunk
    /// cache (issue #20 S2) seeds its reassembled grid to the same dimensions.
    ///
    /// **Producer voxel frame (center-anchoring retirement).** This is the EXACT
    /// occupied span `max_v − min_v` from [`placed_extent_voxels`] — NOT
    /// `size_blocks·d`. The region MUST share the placement frame: producers
    /// center-emit and are recentred by `(min_v + max_v)/2` (see
    /// [`recentre_voxels_for_resolve`]); the recentred composite occupies exactly
    /// `[−D/2, D/2)` with `D = max_v − min_v`, so a block-framed region (`size·d`)
    /// would be too SMALL for a parity-mismatched multi-leaf composite and silently
    /// clip voxels off each end. (The whole-block SIZE readout / block-lattice
    /// overlay still read [`full_extent_blocks`] — that is the only legitimate
    /// block-frame consumer.)
    ///
    /// **This IS the size the assembled render grid takes** for a chunkable scene:
    /// both [`resolve_region`] and the chunk-cache reassembly size their output to
    /// exactly this value (asserted in `placed_region_dimensions_equals_assembled_grid`).
    /// `pub` so the `shot` binary can do the same substitution.
    ///
    /// **Caveat — a Part-only scene** (no intrinsic-size leaf, e.g. a lone
    /// debug-cloud field) returns `[0, 0, 0]` here because it has no composite
    /// extent; such a scene is resolved through the *explicit-region* monolithic
    /// path (sized to the caller's chosen region, not this), so a consumer of a
    /// Part-only scene must use that explicit region — not this — as its dimensions.
    ///
    /// [`placed_extent_voxels`]: Self::placed_extent_voxels
    /// [`recentre_voxels_for_resolve`]: Self::recentre_voxels_for_resolve
    /// [`full_extent_blocks`]: Self::full_extent_blocks
    pub fn placed_region_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        match self.placed_extent_voxels(voxels_per_block) {
            // The EXACT voxel span (`max − min`). Corner-anchored producers emit
            // half-integer centres, so the region-relative decode
            // (`floor(world − region_low)`, see `resolve_region`) is exact for any
            // span parity — no even-padding is needed.
            Some((min_corner, max_corner)) => [
                (max_corner[0] - min_corner[0]) as u32,
                (max_corner[1] - min_corner[1]) as u32,
                (max_corner[2] - min_corner[2]) as u32,
            ],
            None => [0, 0, 0],
        }
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
    /// density` voxels) **corner-anchored** (local span `[0, grid)`, centres at
    /// `idx + 0.5`), placed so its `world_offset` is its LOW CORNER; so a leaf
    /// occupies the half-open absolute-voxel box `[world_offset, world_offset + grid)`
    /// per axis, where `grid = size_blocks · d`. The composite is the union of those
    /// boxes.
    ///
    /// This is the **producer-true** frame the chunk ownership (`floor(position /
    /// chunk_extent)`) lives in — distinct from [`placed_extent_blocks`] (the
    /// whole-block size readout). `None` when no leaf has an intrinsic size.
    fn placed_extent_voxels(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces| {
            let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block) else {
                return;
            };
            any = true;
            for axis in 0..3 {
                // The producer-true emitted grid (`size·d` for an SDF Tool, the exact
                // prism AABB for a SketchTool), corner-anchored so its world offset is
                // the LOW corner: it spans `[off, off + grid)`.
                let grid = grid_voxels[axis];
                let low = world_offset_voxels[axis];
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
        let chunk_extent_voxels = (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

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
        let mut entries: Vec<LeafEntry> = Vec::new();
        let mut has_region_spanning_leaf = false;
        self.for_each_leaf(&mut |world_offset_voxels, content, grid_on_faces| {
            match leaf_producer_grid_voxels(content, voxels_per_block) {
                Some(grid_voxels) => {
                    // The producer-true emitted grid (`size·d` for an SDF Tool, the
                    // exact prism AABB for a SketchTool), corner-anchored: its world
                    // voxel offset is the LOW corner, so the span per axis is
                    // `[off, off + grid)` — identical to `placed_extent_voxels`.
                    // Absolute voxels are i64 (S4a).
                    let mut min = [0i64; 3];
                    let mut max = [0i64; 3];
                    for axis in 0..3 {
                        let grid = grid_voxels[axis];
                        min[axis] = world_offset_voxels[axis];
                        max[axis] = min[axis] + grid;
                    }
                    entries.push(LeafEntry {
                        world_aabb: VoxelAabb::new(min, max),
                        fingerprint: LeafFingerprint::Bounded(leaf_content_fingerprint(
                            world_offset_voxels,
                            content,
                            grid_on_faces,
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
                            world_offset_voxels,
                            content,
                            grid_on_faces,
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

fn leaf_content_fingerprint(
    world_offset_voxels: [i64; 3],
    content: &NodeContent,
    grid_on_faces: bool,
) -> String {
    // The on-face-grid flag is baked into the resolved voxels as `GRID_OVERLAY_BIT`
    // (issue #29 S4), so two otherwise-identical leaves that differ only in this flag
    // resolve to DIFFERENT voxels. It must therefore be part of the fingerprint, or a
    // lone toggle of `voxel_grid_on_faces` produces an identical fingerprint and the
    // chunk-cache diff (`edit_aabb_since`) sees nothing dirty — leaving the stale
    // grid-less chunks in place until an unrelated edit evicts them. The embedded
    // offset is voxels (the canonical placement unit, ADR 0003 §3f(0)); it is an
    // opaque cache key, all leaves on the same unit for consistency.
    let grid = if grid_on_faces { ":grid=1" } else { ":grid=0" };
    match content {
        NodeContent::Tool { shape, material } => {
            format!("Tool@{world_offset_voxels:?}:{shape:?}:{material:?}{grid}")
        }
        NodeContent::SketchTool { producer, material } => {
            format!("SketchTool@{world_offset_voxels:?}:{producer:?}:{material:?}{grid}")
        }
        NodeContent::Part(part) => format!("Part@{world_offset_voxels:?}:{part:?}{grid}"),
        // for_each_leaf only ever yields leaf content (Tool / SketchTool / Part);
        // Group / Instance are interior and never reach a visitor. Fingerprint
        // defensively anyway.
        NodeContent::Group(_) => format!("Group@{world_offset_voxels:?}{grid}"),
        NodeContent::Instance(def_id) => format!("Instance@{world_offset_voxels:?}:{def_id:?}{grid}"),
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

/// The whole-block extent of a leaf node's producer, or `None` for a non-leaf /
/// not-yet-implemented content kind.
fn leaf_size_blocks(content: &NodeContent, voxels_per_block: u32) -> Option<[u32; 3]> {
    let density = voxels_per_block.max(1);
    match content {
        // A Tool's size is now voxel-granular (ADR 0003 §3f(0)). The composite region
        // SIZING reports whole blocks, so round the exact voxel span UP to whole
        // blocks (a sub-block remainder still claims its block, exactly like a
        // SketchTool prism) — a whole-block size divides cleanly and is unchanged.
        NodeContent::Tool { shape, .. } => {
            let ceil_blocks = |voxels: u32| voxels.div_ceil(density);
            Some([
                ceil_blocks(shape.size_voxels[0]),
                ceil_blocks(shape.size_voxels[1]),
                ceil_blocks(shape.size_voxels[2]),
            ])
        }
        // A sketch→extrude prism reports its AABB rounded UP to whole blocks so the
        // composite region SIZING (`full_extent_blocks`) sees its extent — exactly
        // like a Tool. The recentre / chunk-coverage / spatial-index use the exact
        // producer voxel frame (`leaf_producer_grid_voxels`) instead.
        NodeContent::SketchTool { producer, .. } => {
            let [grid_x, grid_y, grid_z] = producer.grid_dimensions();
            let ceil_blocks = |voxels: u32| voxels.div_ceil(density);
            Some([
                ceil_blocks(grid_x),
                ceil_blocks(grid_y),
                ceil_blocks(grid_z),
            ])
        }
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

/// The producer's exact **emitted grid** in voxels per axis (the producer-true
/// frame the chunk ownership lives in), or `None` for a sizeless / interior leaf.
///
/// This is `size_blocks · d` for an [`SdfShape`] `Tool` (a whole-block grid), but
/// the EXACT prism AABB for a [`SketchTool`] — which may NOT be a whole multiple
/// of `d` (a sub-block profile). The chunk-coverage / spatial-index / AABB-skip
/// math must use this true span, not the block-rounded [`leaf_size_blocks`], so a
/// sub-block sketch's voxels are never dropped by a too-small cover.
///
/// [`SketchTool`]: NodeContent::SketchTool
/// One visible leaf of the op-stack as a resolvable producer (ADR 0010 E2). The
/// two-layer classifier + boundary-resolve evaluate this list (in document order, Union
/// on overlap) exactly as [`Scene::resolve_chunk_rebased`] stamps it. Yielded by
/// [`Scene::leaf_producers`].
pub(crate) struct LeafProducer {
    /// The leaf's accumulated WORLD voxel offset (its corner-anchored low corner in the
    /// scene's absolute voxel frame). A local cell `idx` has absolute index
    /// `world_offset_voxels + idx` (ADR 0008 — the frame is carried).
    pub world_offset_voxels: [i64; 3],
    /// The boxed producer that resolves / bounds this leaf in its own `[0, full_dim)`
    /// local voxel-index frame.
    pub producer: Box<dyn VoxelProducer>,
    /// The single-material override id a Tool stamps onto every voxel (`Some`), or `None`
    /// for a Part that brings its own per-voxel materials (the cloud field emits id 0).
    pub material: Option<crate::core_geom::BlockId>,
}

fn leaf_producer_grid_voxels(content: &NodeContent, _voxels_per_block: u32) -> Option<[i64; 3]> {
    match content {
        // The Tool's exact emitted grid is its canonical voxel size directly (ADR
        // 0003 §3f(0); `size_voxels` already IS `blocks · d` for a whole-block size).
        NodeContent::Tool { shape, .. } => Some([
            shape.size_voxels[0] as i64,
            shape.size_voxels[1] as i64,
            shape.size_voxels[2] as i64,
        ]),
        NodeContent::SketchTool { producer, .. } => {
            let [grid_x, grid_y, grid_z] = producer.grid_dimensions();
            Some([grid_x as i64, grid_y as i64, grid_z as i64])
        }
        NodeContent::Part(_) | NodeContent::Group(_) | NodeContent::Instance(_) => None,
    }
}

/// Map a Tool's [`MaterialChoice`] to the categorical [`BlockId`](crate::core_geom::BlockId)
/// it stamps (ADR 0001 step 3 "Materials"; ADR 0003 §3a). A Tool is single-material by
/// nature: every voxel it emits takes this one block id, so distinct nodes render in
/// distinct materials. Stone = 0, Wood = 1, Plain = 2 (see [`MaterialChoice::block_id`]).
fn material_id_for(material: MaterialChoice) -> Option<crate::core_geom::BlockId> {
    Some(material.block_id())
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
    material_override: Option<crate::core_geom::BlockId>,
    grid_overlay: bool,
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
) {
    // The producer sizes its own grid (`SdfShape::resolve` overwrites
    // `dimensions` to its own canonical `size_voxels`, centred at the origin), so
    // the local grid need only seed the dimensions; the cloud field, which has no
    // intrinsic size, fills the region it is handed.
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local, voxels_per_block);

    let zero_offset = translation_voxels == [0, 0, 0];

    if zero_offset && material_override.is_none() && !grid_overlay {
        // Fast path / exact identity: no translation, no material rewrite and no
        // on-face-grid flag bit, so the local occupied set IS the output.
        if output.occupied.is_empty() {
            output.occupied = local.occupied;
            return;
        }
        output.occupied.extend(local.occupied);
        return;
    }

    // General stamp: translate each voxel into the composite (the producer's
    // origin-centred position plus the node's recentred placement), overwrite its
    // material id for a Tool, then OR the on-face-grid flag bit (issue #29 S4) when
    // this node opted in so it travels with each voxel.
    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        if !zero_offset {
            // ADR 0003 §3a / ADR 0008: translate the INTEGER index in the grid's frame
            // (the absolute origin lives on the grid), never an f32 position. The add is
            // i64 then downcast, so the placement is exact for any magnitude.
            voxel.local_index[0] = (voxel.local_index[0] as i64 + translation_voxels[0]) as i32;
            voxel.local_index[1] = (voxel.local_index[1] as i64 + translation_voxels[1]) as i32;
            voxel.local_index[2] = (voxel.local_index[2] as i64 + translation_voxels[2]) as i32;
        }
        if let Some(id) = material_override {
            voxel.block_id = id;
        }
        // ADR 0003 §3c: the on-face-grid flag is a transient render marker on the cell,
        // NOT the categorical `block_id` — the cuboid mesher reads it (splitting boxes on
        // it) and the draw enables the overlay; it never enters the categorical id.
        voxel.grid_overlay = grid_overlay;
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
    material_override: Option<crate::core_geom::BlockId>,
    grid_overlay: bool,
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    chunk_min_voxels: [i64; 3],
    chunk_max_voxels: [i64; 3],
) {
    // Resolve ONLY the cells this chunk owns, in the producer's LOCAL voxel-index
    // frame `[0, full_dim)`. A producer's local cell `idx` has absolute centre
    // `translation_voxels[axis] + idx + 0.5`; the historical chunk-membership clip
    // kept `chunk_min ≤ translation + idx + 0.5 < chunk_max`. The `+ 0.5` cancels on
    // half-open INTEGER chunk edges:
    //   idx + 0.5 ≥ chunk_min  ⟺  idx ≥ chunk_min − translation
    //   idx + 0.5 <  chunk_max  ⟺  idx <  chunk_max − translation
    // so the chunk window in the local frame is the integer half-open box below.
    // `resolve_into` clamps it to `[0, full_dim)` internally, so an out-of-range
    // window is safe, and it returns EXACTLY the cells the old per-voxel clip kept —
    // a producer spanning N chunks now resolves each chunk's cells once instead of
    // re-resolving its full extent N×.
    let mut local = VoxelGrid::new(region_dimensions);
    let window_local = crate::spatial_index::VoxelAabb::new(
        [
            chunk_min_voxels[0] - translation_voxels[0],
            chunk_min_voxels[1] - translation_voxels[1],
            chunk_min_voxels[2] - translation_voxels[2],
        ],
        [
            chunk_max_voxels[0] - translation_voxels[0],
            chunk_max_voxels[1] - translation_voxels[1],
            chunk_max_voxels[2] - translation_voxels[2],
        ],
    );
    producer.resolve_into(&mut local, voxels_per_block, window_local);

    // The voxel's chunk-local placement, rebased to the floating origin in i64
    // FIRST so the f32 add never sees a large magnitude. For the live render the
    // floating origin equals the composite recentre, so for a near scene this is
    // EXACTLY the small `world_offset·d − recentre` translation `resolve_region`
    // adds in f32 today — bit-identical framing — while a far chunk no longer loses
    // the voxel-centre `.5` to f32 rounding at ~1e6 magnitude (the S1 speckle).
    let rebased_translation = [
        translation_voxels[0] - floating_origin_voxels[0],
        translation_voxels[1] - floating_origin_voxels[1],
        translation_voxels[2] - floating_origin_voxels[2],
    ];

    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        // Store the rebased (origin-relative) INTEGER index (ADR 0003 §3a). The rebase
        // is a pure i64 subtraction done here BEFORE the downcast, so the far chunk's
        // index keeps full precision — the f32 magnitude loss the old f32 payload took
        // at ~1e6 (the S1 speckle) is gone, and `world_position()` (= index + 0.5)
        // reproduces the small rebased centre exactly for a near scene.
        voxel.local_index[0] = (voxel.local_index[0] as i64 + rebased_translation[0]) as i32;
        voxel.local_index[1] = (voxel.local_index[1] as i64 + rebased_translation[1]) as i32;
        voxel.local_index[2] = (voxel.local_index[2] as i64 + rebased_translation[2]) as i32;

        if let Some(id) = material_override {
            voxel.block_id = id;
        }
        // ADR 0003 §3c: transient render marker, not the categorical id (see stamp_producer).
        voxel.grid_overlay = grid_overlay;
        output.occupied.push(voxel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::ShapeKind;

    /// `from_measurements` derives the canonical voxel offset from the per-axis
    /// authored expression and retains the expression (ADR 0003 §3f(0)). A `3.5
    /// blocks` axis lands on `3.5 · d` voxels (56 at d16, 112 at d32 — the lossless
    /// parametric refine).
    #[test]
    fn transform_from_measurements_derives_voxels_and_retains_expression() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(-4),                           // -4 voxels
            Measurement::new(ExactRational::from_integer(2), 8),    // 2 blocks 8 voxels
        ];
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels, [56, -4, 40]);
        // The expression is retained verbatim.
        assert_eq!(at_sixteen.offset_measurements(), measurements);
        // The SAME measurements re-evaluate at a denser document (lossless refine).
        let at_thirty_two = NodeTransform::from_measurements(measurements, 32);
        assert_eq!(at_thirty_two.offset_voxels, [112, -4, 72]);
    }

    /// A retained NON-block-multiple offset (`3.5 blocks` on X = 56 vx at d16)
    /// re-evaluated at a NON-dividing density (d15, where 3.5·15 = 52.5) must not
    /// panic, floors X to a whole voxel, and keeps the retained measurement
    /// CONSISTENT with `offset_voxels` (the seam bug: they used to disagree). This
    /// is the lossy density-retarget path inside `from_measurements`.
    #[test]
    fn from_measurements_non_dividing_density_stays_self_consistent() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        // 3.5 blocks lands cleanly at d16 (= 56 voxels).
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels[0], 56);

        // Re-evaluate the SAME authored expression at the non-dividing d15.
        let at_fifteen = NodeTransform::from_measurements(at_sixteen.offset_measurements(), 15);
        // 3.5·15 = 52.5 → floored to 52 voxels (no panic).
        assert_eq!(at_fifteen.offset_voxels[0], 52);
        // The retained measurement now AGREES with the floored voxels: re-evaluating
        // it at d15 yields exactly offset_voxels[0] (no silent disagreement).
        let retained = at_fifteen.offset_measurements();
        assert_eq!(
            retained[0].to_voxels(15).unwrap(),
            at_fifteen.offset_voxels[0],
            "retained measurement must be consistent with the floored canonical voxels"
        );
    }

    /// A retained `3 blocks 8 voxels` (= 56 vx at d16) re-evaluated at the
    /// integer-multiple d32 keeps the VOXEL TERM EXACT: 3·32 + 8 = 104, NOT the
    /// integer rescale's 56·2 = 112. The authored expression is preserved.
    #[test]
    fn from_measurements_integer_multiple_density_keeps_voxel_term_exact() {
        let measurements = [
            Measurement::new(ExactRational::from_integer(3), 8), // 3 blocks 8 voxels
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels[0], 56);

        let at_thirty_two =
            NodeTransform::from_measurements(at_sixteen.offset_measurements(), 32);
        assert_eq!(
            at_thirty_two.offset_voxels[0], 104,
            "voxel term stays exact (3*32 + 8), NOT the integer rescale 112"
        );
        // The authored expression is preserved verbatim.
        assert_eq!(at_thirty_two.offset_measurements()[0], measurements[0]);
    }

    /// An OLD `NodeTransform` JSON that predates `offset_measurements` still
    /// deserialises (serde default → `None`), and the accessor SYNTHESISES a
    /// pure-voxel measurement equal to `offset_voxels` per axis — which
    /// re-evaluates back to exactly those voxels at any density (versioning:
    /// shared documents must load forward, ADR 0003 §3f(0)).
    #[test]
    fn transform_serde_back_compat_synthesises_measurements_from_voxels() {
        let old_json = r#"{ "offset_voxels": [48, -16, 7] }"#;
        let restored: NodeTransform =
            serde_json::from_str(old_json).expect("old transform without measurements must load");
        assert_eq!(restored.offset_voxels, [48, -16, 7]);
        let synthesised = restored.offset_measurements();
        for (axis, &voxels) in restored.offset_voxels.iter().enumerate() {
            assert_eq!(synthesised[axis], Measurement::from_voxels(voxels));
            assert_eq!(synthesised[axis].to_voxels(16).unwrap(), voxels);
            assert_eq!(synthesised[axis].to_voxels(32).unwrap(), voxels);
        }
    }

    /// A `NodeTransform` carrying retained measurements round-trips through serde
    /// unchanged (the new field persists for a forward-saved document).
    #[test]
    fn transform_serde_round_trips_with_retained_measurements() {
        let transform = NodeTransform::from_measurements(
            [
                Measurement::new(ExactRational::new(7, 2).unwrap(), 0),
                Measurement::from_voxels(-4),
                Measurement::new(ExactRational::from_integer(2), 8),
            ],
            16,
        );
        let json = serde_json::to_string(&transform).expect("serialises");
        let restored: NodeTransform = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(restored, transform);
        assert_eq!(restored.offset_measurements(), transform.offset_measurements());
        assert_eq!(restored.offset_voxels, transform.offset_voxels);
    }

    /// The identical-behaviour guarantee (ADR 0001 step 1): a one-node Tool scene
    /// resolved over the node's full extent yields the SAME occupied count as
    /// calling `SdfShape::resolve` directly — and the same grid dimensions.
    #[test]
    fn tool_scene_matches_bare_producer() {
        let geometry = GeometryParams {
            shape: ShapeKind::Sphere,
            size_voxels: [6 * 16, 6 * 16, 6 * 16],
            size_measurements: None,
            voxels_per_block: 16,
            wall_blocks: 1,
        };

        // Bare producer (today's path).
        let shape = SdfShape::from_geometry(geometry.clone());
        let mut bare = VoxelGrid::new(shape.grid_dimensions(geometry.voxels_per_block));
        shape.resolve(&mut bare, geometry.voxels_per_block);

        // Through the scene.
        let scene = Scene::from_geometry(geometry.clone(), MaterialChoice::Stone);
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
                GeometryParams { shape: kind, size_voxels: [5 * 16, 5 * 16, 5 * 16], size_measurements: None, voxels_per_block: 16, wall_blocks: 1 },
                MaterialChoice::Stone,
            );
            assert_equal(&scene, 16, &format!("{kind:?}"));
        }

        // Flat / odd sizes (the 5×1×5 app default and friends), several densities.
        for vpb in [1u32, 8, 16] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams { shape: ShapeKind::Cylinder, size_voxels: [size[0] * vpb, size[1] * vpb, size[2] * vpb], size_measurements: None, voxels_per_block: vpb, wall_blocks: 1 },
                    MaterialChoice::Stone,
                );
                assert_equal(&scene, vpb, &format!("cylinder {size:?}@{vpb}"));
            }
        }

        // A placed multi-node scene (sphere at origin + box +8X + torus +6Z).
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, 16);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let demo_scene = scene_with_top_level_selected(Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]), 0);
        assert_equal(&demo_scene, 16, "demo-scene");

        // An instanced village (one house definition placed by four instances).
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, 16);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let mut village = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        village.add_definition(
            house_def_id,
            "House",
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        let village = scene_with_top_level_selected(village, 0);
        assert_equal(&village, 16, "demo-village");
    }

    /// Build the review's parity-mismatched composite: Tool A `size [1,1,1] @ offset
    /// 0` + Tool B `size [2,1,1] @ offset +1 block` at density `vpb` — the exact
    /// X-axis parity mismatch (odd 1 vs even 2) the adversarial review caught.
    fn parity_mismatch_scene(vpb: u32) -> Scene {
        let mut node_a = Node::new(
            "A",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, vpb),
                material: MaterialChoice::Stone,
            },
        );
        node_a.transform = NodeTransform::from_blocks([0, 0, 0], vpb);
        let mut node_b = Node::new(
            "B",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 1, 1], 1, vpb),
                material: MaterialChoice::Wood,
            },
        );
        node_b.transform = NodeTransform::from_blocks([1, 0, 0], vpb);
        scene_with_top_level_selected(Scene::from_nodes(vec![node_a, node_b]), 0)
    }

    /// THE BUG-CLASS MATRIX (corner-anchoring): across size ∈ {1,2,3,5,6} ×
    /// density ∈ {1,2,5,15,16}, for BOTH a single shape AND a 2-leaf mixed-parity
    /// composite, assert the four invariants that the old center-emit broke:
    ///
    /// (a) every occupied voxel CENTRE is a HALF-INTEGER (`fract()==0.5`) — on the
    ///     voxel lattice, inside a cell, for ANY size·d parity (the win: odd grids no
    ///     longer land on integers and straddle cell boundaries);
    /// (b) ZERO voxels dropped — occupied count == the expected filled-cell count;
    /// (c) every DECODED index is in `[0, dim)` (no clipped slab, none at `== dim`),
    ///     using the production decode `round(world + floor(dim/2) − 0.5)`;
    /// (d) the monolithic and chunk paths emit the IDENTICAL voxel set.
    ///
    /// Crucially this passes at ODD density (d ∈ {1,5,15}) and MIXED parity — the
    /// cases the center-emit convention could not represent.
    #[test]
    fn corner_anchoring_parity_matrix() {
        use crate::chunk_cache::ChunkResolveCache;

        // Decode an occupied set to integer cell indices with the production rule.
        let decode_cells = |grid: &VoxelGrid| -> std::collections::BTreeSet<[i64; 3]> {
            let [dx, dy, dz] = grid.dimensions;
            let half = [(dx / 2) as f32, (dy / 2) as f32, (dz / 2) as f32];
            grid.occupied
                .iter()
                .map(|voxel| {
                    let position = voxel.world_position();
                    [
                        (position[0] + half[0] - 0.5).round() as i64,
                        (position[1] + half[1] - 0.5).round() as i64,
                        (position[2] + half[2] - 0.5).round() as i64,
                    ]
                })
                .collect()
        };
        // The exact f32-bit + material multiset (order-independent path comparison).
        let multiset = |grid: &VoxelGrid| {
            let mut set = std::collections::BTreeMap::<([u32; 3], u16), usize>::new();
            for voxel in &grid.occupied {
                let position = voxel.world_position();
                let key = (
                    [
                        position[0].to_bits(),
                        position[1].to_bits(),
                        position[2].to_bits(),
                    ],
                    voxel.color_index(),
                );
                *set.entry(key).or_insert(0) += 1;
            }
            set
        };

        // Run the four-invariant battery on one scene, returning its decoded cell set.
        let check = |scene: &Scene, vpb: u32, label: &str| -> std::collections::BTreeSet<[i64; 3]> {
            let dims = scene.placed_region_dimensions(vpb);
            let monolithic = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
            let mut cache = ChunkResolveCache::new();
            let assembled = cache.resolve_region(scene, vpb, 0);

            assert_eq!(monolithic.dimensions, dims, "[{label}] monolithic dims voxel-framed");
            assert_eq!(assembled.dimensions, dims, "[{label}] assembled dims voxel-framed");

            // (a) every centre is a half-integer.
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for axis in 0..3 {
                    assert_eq!(
                        position[axis].fract().abs(),
                        0.5,
                        "[{label}] centre {:?} axis {axis} must be a half-integer (on the lattice)",
                        position
                    );
                }
            }
            // (c) every decoded index is in [0, dim).
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for (axis, &dim) in dims.iter().enumerate() {
                    let half = (dim / 2) as f32;
                    let index = (position[axis] + half - 0.5).round() as i64;
                    assert!(
                        index >= 0 && index < dim as i64,
                        "[{label}] voxel {:?} axis {axis} decodes to {index} OUTSIDE [0, {dim})",
                        position
                    );
                }
            }
            // (d) the two paths emit the identical voxel set.
            assert_eq!(
                multiset(&monolithic),
                multiset(&assembled),
                "[{label}] monolithic and chunk paths must emit the identical voxel set"
            );
            assert!(!monolithic.occupied.is_empty(), "[{label}] non-empty");
            decode_cells(&monolithic)
        };

        for vpb in [1u32, 2, 5, 15, 16] {
            // --- single shape: a Box fully fills `size·d`³ cells, zero dropped (b). ---
            for size in [1u32, 2, 3, 5, 6] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: ShapeKind::Box,
                        size_voxels: [size * vpb, size * vpb, size * vpb],
                        size_measurements: None,
                        voxels_per_block: vpb,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                let label = format!("box {size}³ @ d{vpb}");
                let cells = check(&scene, vpb, &label);
                let expected = (size * vpb).pow(3) as usize;
                assert_eq!(
                    cells.len(), expected,
                    "[{label}] (b) zero dropped: distinct cells {} must equal size·d cubed {expected}",
                    cells.len()
                );
                let monolithic = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
                assert_eq!(
                    monolithic.occupied_count(), expected,
                    "[{label}] (b) occupied count must equal the filled-cell count"
                );
            }

            // --- 2-leaf mixed-parity composite: A [1,1,1]@0 + B [2,1,1]@+1 block. ---
            let scene = parity_mismatch_scene(vpb);
            let label = format!("parity-composite @ d{vpb}");
            let cells = check(&scene, vpb, &label);
            // (b) distinct cells = |A| + |B| − overlap. A spans X[0,d), B spans
            // X[d, 3d) (off=1 block=d voxels, grid 2d) → DISJOINT on X (no overlap),
            // both full d×d in Y,Z. So distinct = d³ + 2d³ = 3d³.
            let d = vpb as i64;
            let expected_distinct = d * d * d + 2 * d * d * d;
            assert_eq!(
                cells.len() as i64, expected_distinct,
                "[{label}] (b) distinct occupied cells {} must equal |A|+|B| (disjoint) {expected_distinct}",
                cells.len()
            );
        }
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
            seed: 0,
        };
        let mut bare = VoxelGrid::new(dimensions);
        bare_field.resolve(&mut bare, voxels_per_block);

        let scene =
            Scene::single_node(Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })));
        let region = RegionBlocks::new(size_blocks);
        let resolved = scene.resolve_region(region, voxels_per_block, 0);

        assert_eq!(resolved.dimensions, bare.dimensions);
        assert_eq!(resolved.occupied_count(), bare.occupied_count());
    }

    /// CORNER-ANCHORING (cloud producer): a PART-ONLY cloud at ODD density drops ZERO
    /// voxels — every occupied centre is a HALF-INTEGER (on the voxel lattice) and
    /// every decoded index ∈ [0, dim). This is the case center-emit broke: at an odd
    /// region dim the centred bottom voxel decoded to index −1 and was dropped.
    /// Tested at d=1 and d=5 (odd densities → odd region dims for an odd block size).
    #[test]
    fn part_only_cloud_at_odd_density_drops_no_voxels() {
        // 5×5×5 blocks at odd density → region dims 5·d (odd). A 64-vx field has plenty
        // of solid voxels so the boundary cells are genuinely exercised.
        for (size_blocks, vpb) in [([5u32, 5, 5], 1u32), ([5, 5, 5], 5)] {
            let scene = Scene::single_node(Node::new(
                "Clouds",
                NodeContent::Part(Part::DebugClouds { seed: 7 }),
            ));
            let region = RegionBlocks::new(size_blocks);
            let dims = [
                size_blocks[0] * vpb,
                size_blocks[1] * vpb,
                size_blocks[2] * vpb,
            ];
            let label = format!("part-only cloud {size_blocks:?}@d{vpb}");

            let monolithic = scene.resolve_region(region, vpb, 0);
            assert_eq!(monolithic.dimensions, dims, "[{label}] dims = region·d");
            assert!(!monolithic.occupied.is_empty(), "[{label}] non-empty cloud");

            // (a) every centre is a half-integer; (c) every decoded index ∈ [0, dim).
            // A Part-only cloud is corner-anchored at the explicit region (low corner 0,
            // recentre 0), so the decode is `floor(world)`.
            let mut decoded = 0usize;
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for (axis, &dim) in dims.iter().enumerate() {
                    let pos = position[axis];
                    assert_eq!(
                        pos.fract().abs(), 0.5,
                        "[{label}] centre {pos} axis {axis} must be a half-integer"
                    );
                    let index = pos.floor() as i64;
                    assert!(
                        index >= 0 && index < dim as i64,
                        "[{label}] voxel {pos} axis {axis} decodes to {index} OUTSIDE [0, {dim})"
                    );
                }
                decoded += 1;
            }
            assert_eq!(
                decoded, monolithic.occupied_count(),
                "[{label}] every emitted voxel decodes in-range (no silent drop)"
            );

            // A Part-only scene has no chunkable extent, so the monolithic path above
            // IS the resolve path (the chunk reassembly is for Tool-bearing scenes).
            assert!(
                !scene.has_chunkable_extent(vpb),
                "[{label}] a Part-only cloud has no chunkable extent"
            );
        }
    }

    /// CORNER-ANCHORING (mixed frame): a Tool and a Cloud in the SAME scene resolve in
    /// ONE frame — the cloud's voxels are NOT offset by `region_dim/2` from the Tool.
    /// Center-emit broke this: the Tool corner-anchored but the cloud center-emitted,
    /// so they sat in different frames. Now BOTH corner-anchor at `[0, region_dim)`, so
    /// a Tool placed at offset 0 and the region-filling cloud share the same low corner.
    #[test]
    fn mixed_tool_and_cloud_resolve_in_one_frame() {
        // A Box Tool at offset 0 (size 3³) plus a Cloud. The Tool's voxel span and the
        // cloud's region span share the SAME low corner in the resolved frame.
        let vpb = 4u32;
        let mut tool = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb),
                material: MaterialChoice::Stone,
            },
        );
        tool.transform = NodeTransform::from_blocks([0, 0, 0], vpb);
        let cloud = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 3 }));
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![tool, cloud]),
            0,
        );

        // The Tool gives the scene a chunkable extent; the region is its voxel span.
        let region = scene.full_extent_blocks(vpb);
        let dims = scene.placed_region_dimensions(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        assert_eq!(grid.dimensions, dims, "region is the Tool's voxel-framed span");

        // Decode in the recentred frame (low corner −floor(dim/2)). EVERY voxel —
        // whether from the Tool or the Cloud — must decode to an index in [0, dim) with
        // a half-integer centre. If the cloud were still center-emitting it would be
        // offset by ~region_dim/2 and a slab would decode out of range.
        let recentre = scene.recentre_voxels_for_resolve(vpb);
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            for (axis, &dim) in dims.iter().enumerate() {
                let pos = position[axis];
                assert_eq!(
                    pos.fract().abs(), 0.5,
                    "mixed scene: centre {pos} axis {axis} must be a half-integer (same frame)"
                );
                let half = (dim / 2) as f32;
                let index = (pos + half - 0.5).round() as i64;
                assert!(
                    index >= 0 && index < dim as i64,
                    "mixed scene: voxel {pos} axis {axis} decodes to {index} OUTSIDE [0, {dim}) \
                     — a cloud offset by region_dim/2 would land here"
                );
            }
        }

        // The Tool's voxels land EXACTLY where corner-anchored math says: a 3³ box at
        // offset 0 occupies absolute `[0, 3d)`; recentred, its low corner is
        // `0 − recentre`. At least one voxel sits at that low corner (the box fully
        // fills its AABB). This pins the cloud sharing the Tool's frame, not an offset.
        let expected_low = [
            (0 - recentre[0]) as f32 + 0.5,
            (0 - recentre[1]) as f32 + 0.5,
            (0 - recentre[2]) as f32 + 0.5,
        ];
        let has_box_low_corner = grid.occupied.iter().any(|v| {
            let position = v.world_position();
            (position[0] - expected_low[0]).abs() < 1e-3
                && (position[1] - expected_low[1]).abs() < 1e-3
                && (position[2] - expected_low[2]).abs() < 1e-3
        });
        assert!(
            has_box_low_corner,
            "the corner-anchored Box must place a voxel at its recentred low corner {expected_low:?}"
        );
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
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [6, 6, 6], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        // A full-extent box: its corners poke outside the inscribed sphere, so the
        // union is strictly larger than the sphere alone (a real composite).
        let cube = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [6, 6, 6], 1, voxels_per_block),
                material: MaterialChoice::Wood,
            },
        );

        // Each node resolved alone.
        let sphere_only = Scene::single_node(sphere.clone())
            .resolve_region(region, voxels_per_block, 0);
        let cube_only =
            Scene::single_node(cube.clone()).resolve_region(region, voxels_per_block, 0);

        // Both nodes composited.
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![sphere, cube]), 0);
        let union = scene.resolve_region(region, voxels_per_block, 0);

        // The expected set-union of the two single-node occupied sets, keyed by
        // integer voxel position (the producers emit voxel-centre world positions).
        use std::collections::HashSet;
        let key = |grid: &VoxelGrid| -> HashSet<[i64; 3]> {
            grid.occupied
                .iter()
                .map(|voxel| {
                    let position = voxel.world_position();
                    [
                        position[0].round() as i64,
                        position[1].round() as i64,
                        position[2].round() as i64,
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
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
        let scene = Scene::single_node(Node::new(
            "Wood box",
            NodeContent::Tool { shape, material: MaterialChoice::Wood },
        ));
        let grid = scene.resolve_region(RegionBlocks::new([2, 2, 2]), voxels_per_block, 0);
        let wood_id = MaterialChoice::Wood.material_id();
        assert!(grid.occupied_count() > 0, "the box must emit voxels");
        assert!(
            grid.occupied.iter().all(|voxel| voxel.color_index() == wood_id),
            "every voxel a Wood Tool stamps must carry the Wood material id"
        );
    }

    /// ADR 0001 step 3 (per-voxel material): a 2-Tool scene (Stone + Wood, placed
    /// disjointly) yields BOTH material ids present — proving the per-voxel id
    /// travels through compositing so the two nodes render in distinct materials.
    #[test]
    fn two_material_scene_has_both_material_ids() {
        let voxels_per_block = 8u32;
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut stone = Node::new("Stone", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        stone.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut wood = Node::new("Wood", NodeContent::Tool { shape: base, material: MaterialChoice::Wood });
        wood.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![stone, wood]), 0);
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        assert_ne!(stone_id, wood_id, "Stone and Wood must map to distinct ids");
        assert!(
            grid.occupied.iter().any(|voxel| voxel.color_index() == stone_id),
            "the Stone node's voxels must carry the Stone id"
        );
        assert!(
            grid.occupied.iter().any(|voxel| voxel.color_index() == wood_id),
            "the Wood node's voxels must carry the Wood id"
        );
    }

    /// Issue #29 S4 (per-object on-face grid): the resolver ORs
    /// [`crate::voxel::GRID_OVERLAY_BIT`] into a node's stamped `material_id`
    /// **iff** that node's `grids.voxel_grid_on_faces` is set — and the masked
    /// material id still round-trips to the real handle (≤2). Parametrized over
    /// density {1, 15, 16} so the bit survives every density's chunk bucketing.
    #[test]
    fn voxel_grid_flag_bit_set_iff_node_opts_in() {
        for &voxels_per_block in &[1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
            let wood_id = MaterialChoice::Wood.material_id();

            // Node with the on-face grid ON → every voxel carries the flag bit, and
            // the masked id is still the real Wood handle (the bit never corrupts it).
            let mut on = Node::new(
                "On",
                NodeContent::Tool { shape: shape.clone(), material: MaterialChoice::Wood },
            );
            on.grids.voxel_grid_on_faces = true;
            let scene = Scene::single_node(on);
            let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
            assert!(grid.occupied_count() > 0);
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| v.grid_overlay),
                "density {voxels_per_block}: a node with voxel_grid_on_faces must flag every voxel"
            );
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| v.color_index() == wood_id),
                "density {voxels_per_block}: the colour index must round-trip to Wood (≤2)"
            );

            // Same node with the flag OFF → no voxel carries the bit (the default).
            let off = Node::new(
                "Off",
                NodeContent::Tool { shape, material: MaterialChoice::Wood },
            );
            let scene = Scene::single_node(off);
            let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
            assert!(grid.occupied_count() > 0);
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| !v.grid_overlay),
                "density {voxels_per_block}: a node WITHOUT the flag must leave the bit clear"
            );
        }
    }

    /// Issue #29 S4: in a 2-node scene with the on-face grid enabled on ONE node
    /// only, exactly that node's voxels carry the flag bit; the other node's don't —
    /// the per-object gating the headless capture verifies. Also confirms the bit
    /// travels through the chunked resolve path (`resolve_chunk`) identically.
    #[test]
    fn voxel_grid_flag_bit_is_per_object() {
        let voxels_per_block = 8u32;
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        // Stone node opts IN; Wood node opts OUT, placed disjointly.
        let mut stone = Node::new(
            "Stone",
            NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone },
        );
        stone.grids.voxel_grid_on_faces = true;
        let wood = Node::new(
            "Wood",
            NodeContent::Tool { shape: base, material: MaterialChoice::Wood },
        );
        let mut wood = wood;
        wood.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![stone, wood]), 0);
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        // Every flagged voxel is a Stone voxel; every Wood voxel is unflagged.
        let stone_flagged = grid
            .occupied
            .iter()
            .filter(|v| v.color_index() == stone_id)
            .all(|v| v.grid_overlay);
        let wood_unflagged = grid
            .occupied
            .iter()
            .filter(|v| v.color_index() == wood_id)
            .all(|v| !v.grid_overlay);
        assert!(stone_flagged, "the enabled (Stone) node's voxels must all be flagged");
        assert!(wood_unflagged, "the disabled (Wood) node's voxels must all be unflagged");
        assert!(
            grid.occupied.iter().any(|v| v.grid_overlay),
            "at least one voxel (the Stone node's) must carry the flag"
        );
    }

    /// A hidden node contributes nothing.
    #[test]
    fn hidden_node_stamps_nothing() {
        let mut node = Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 8),
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
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks([offset_x, 0, 0], voxels_per_block);
        // A region wide enough to hold the offset box without clipping.
        let region = RegionBlocks::new([8, 1, 1]);
        let grid = Scene::single_node(node).resolve_region(region, voxels_per_block, 0);
        grid.occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    position[0].round() as i64,
                    position[1].round() as i64,
                    position[2].round() as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 3 (a): a node placed at a whole-block offset `[N, 0, 0]` places
    /// its voxels shifted by exactly `N × voxels_per_block` in X versus offset 0.
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
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut at_zero = Node::new("A", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        at_zero.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut at_n = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_n.transform = NodeTransform::from_blocks([n, 0, 0], voxels_per_block);

        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![at_zero, at_n]), 0);
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
        let mut xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position()[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let split_x = (xs.first().unwrap() + xs.last().unwrap()) / 2.0;

        let cluster_low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] < split_x)
            .map(|v| key(v.world_position()))
            .collect();
        let cluster_high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] >= split_x)
            .map(|v| key(v.world_position()))
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

        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut a = Node::new("A", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        a.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut b = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        b.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);

        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![a, b]), 0);
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
        let base = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
        let mut node = Node::new("Box", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks([4, 0, 0], voxels_per_block);
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
            Node::new("Origin", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        origin_box.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut offset_box =
            Node::new("Offset", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        offset_box.transform = NodeTransform::from_blocks([4, 0, 0], voxels_per_block);
        let two = scene_with_top_level_selected(Scene::from_nodes(vec![origin_box, offset_box]), 0);
        let extent = two.full_extent_blocks(voxels_per_block);
        assert_eq!(
            extent.size_blocks,
            [6, 2, 2],
            "the offset node widens the composite extent in X from 2 to 6 blocks"
        );
    }

    /// A 1×1×1 box Tool shape, used as a leaf in the step-4 recursion/instancing
    /// tests (the node carries the material; the shape does not).
    fn unit_box_shape() -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 8)
    }

    /// Key a grid's occupied voxels by exact half-integer voxel position (×2 → an
    /// exact integer, no float-equality fragility). Used to compare voxel SETS.
    fn position_keys(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
        grid.occupied
            .iter()
            .map(|v| {
                let position = v.world_position();
                [
                    (position[0] * 2.0) as i64,
                    (position[1] * 2.0) as i64,
                    (position[2] * 2.0) as i64,
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
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        );
        leaf.transform = NodeTransform::from_blocks([b, 0, 0], voxels_per_block);
        let grouped = scene_with_top_level_selected(
            Scene::from_nodes(vec![NodeBuilder::group_at("Group", [a, 0, 0], voxels_per_block, vec![leaf.into()])]),
            0,
        );
        let grouped_grid = grouped.resolve_region(region, voxels_per_block, 0);

        // Flat reference: the same box placed directly at A + B.
        let mut flat_leaf = Node::new(
            "Flat",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        );
        flat_leaf.transform = NodeTransform::from_blocks([a + b, 0, 0], voxels_per_block);
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

        let mut instance = Node::new("I", NodeContent::Instance(def_id));
        instance.transform = NodeTransform::from_blocks([t, 0, 0], voxels_per_block);
        let mut instanced_scene = Scene::from_nodes(vec![instance]);
        // Definition: a single box at the origin (within the def).
        instanced_scene.add_definition(
            def_id,
            "Body".to_string(),
            vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
            )],
        );
        let instanced = scene_with_top_level_selected(instanced_scene, 0);
        let instanced_grid = instanced.resolve_region(region, voxels_per_block, 0);

        // Direct: the same box placed directly at T.
        let mut direct = Node::new(
            "Direct",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
        );
        direct.transform = NodeTransform::from_blocks([t, 0, 0], voxels_per_block);
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
        let house_body = || {
            vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
            )]
        };

        // The def's own occupied count (resolved alone at the origin).
        let mut def_only_scene =
            Scene::from_nodes(vec![Node::new("I", NodeContent::Instance(def_id))]);
        def_only_scene.add_definition(def_id, "House".to_string(), house_body());
        let def_only = scene_with_top_level_selected(def_only_scene, 0);
        let def_count = def_only
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(def_count > 0);

        // Two instances 6 blocks apart in X (a 1-block house → 5-block gap: disjoint).
        let mut house_a = Node::new("A", NodeContent::Instance(def_id));
        house_a.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut house_b = Node::new("B", NodeContent::Instance(def_id));
        house_b.transform = NodeTransform::from_blocks([6, 0, 0], voxels_per_block);
        let mut village_scene = Scene::from_nodes(vec![house_a, house_b]);
        village_scene.add_definition(def_id, "House".to_string(), house_body());
        let village = scene_with_top_level_selected(village_scene, 0);
        let region = village.full_extent_blocks(voxels_per_block);
        let grid = village.resolve_region(region, voxels_per_block, 0);

        assert_eq!(
            grid.occupied_count(),
            2 * def_count,
            "two disjoint instances of one def → 2× the def's voxel count"
        );

        // Disjoint: split the occupied set at the composite centre; each half is a
        // full house, and the two halves share no voxel position.
        let xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position()[0]).collect();
        let split_x = (xs.iter().cloned().fold(f32::MAX, f32::min)
            + xs.iter().cloned().fold(f32::MIN, f32::max))
            / 2.0;
        let low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] < split_x)
            .map(|v| { let p = v.world_position(); [(p[0] * 2.0) as i64, (p[1] * 2.0) as i64, (p[2] * 2.0) as i64] })
            .collect();
        let high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] >= split_x)
            .map(|v| { let p = v.world_position(); [(p[0] * 2.0) as i64, (p[1] * 2.0) as i64, (p[2] * 2.0) as i64] })
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

        let mut scene_build =
            Scene::from_nodes(vec![Node::new("Root", NodeContent::Instance(def_id))]);
        // A definition whose children are (a) a real box leaf and (b) an Instance of
        // ITSELF — the cycle the guard must break.
        scene_build.add_definition(
            def_id,
            "Recursive".to_string(),
            vec![
                Node::new(
                    "Box",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
                ),
                Node::new("Self", NodeContent::Instance(def_id)),
            ],
        );
        let scene = scene_with_top_level_selected(scene_build, 0);

        // Resolves (no overflow) and contributes the single box ONCE — the self-
        // instance is skipped, so the count is finite and equals one box's voxels.
        let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
        let one_box = Scene::single_node(Node::new(
            "Box",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        ))
        .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
        .occupied_count();
        assert_eq!(
            grid.occupied_count(),
            one_box,
            "a self-instancing def contributes its leaves finitely (cycle skipped)"
        );
    }

    /// Mint stable [`NodeId`]s for a freshly-built test scene and select the
    /// top-level node at `index` by id (ADR 0003 Phase B3: selection is keyed by
    /// [`NodeId`], so a fixture built with positional intent must resolve "select
    /// node `index`" to that node's id after minting). Returns the scene with its
    /// ids minted and the chosen node active — the id-era equivalent of the old
    /// `active: Some(NodePath::root_index(index))` struct-literal fixtures.
    fn scene_with_top_level_selected(mut scene: Scene, index: usize) -> Scene {
        scene.ensure_node_ids();
        scene.active = scene
            .id_at_path(&NodePath::root_index(index));
        scene
    }

    /// A small flat scene of two box Tools, the first selected — the fixture the
    /// tree-mutation UI helper tests build on. ADR 0003 Phase B3: ids are minted so
    /// the selection (and the `group_active` it drives) resolves by identity.
    fn two_box_scene(voxels_per_block: u32) -> Scene {
        let mut scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                Node::new(
                    "A",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
                ),
                Node::new(
                    "B",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
                ),
            ]),
            0,
        );
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    /// ADR 0001 step 4 (UI helper): `group_active` wraps the active node in a new
    /// Group, so the active node becomes a CHILD of that Group. After grouping, the
    /// top-level node at the old slot is a `Group` whose sole child is the original
    /// node, and the active selection points at that child (path `[0, 0]`).
    #[test]
    fn group_active_nests_node_under_new_group() {
        let mut scene = two_box_scene(8);
        // Node "A" (top-level 0) is the active selection; remember its stable id so
        // we can confirm the wrap keeps that SAME node selected by identity.
        let node_a_id = scene.id_at_path(&NodePath::root_index(0)).expect("A has an id");
        assert_eq!(scene.active, Some(node_a_id));

        let group_id = scene.group_active().expect("there is an active node to group");
        // B4: `group_active` now returns the new Group's stable id; it resolves to
        // the old top-level slot the Group took (path [0]).
        assert_eq!(
            scene.path_of(group_id),
            Some(NodePath::root_index(0)),
            "the Group takes the old slot"
        );

        // The top-level node is now a Group with exactly one child (the old "A").
        match &scene.root_node(0).content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1, "the Group holds exactly the wrapped node");
                assert_eq!(
                    scene.arena[&children[0]].name, "A",
                    "the wrapped child is the original node"
                );
            }
            other => panic!("expected a Group at slot 0, got {other:?}"),
        }
        // The wrapped child is still the active selection — by identity it is the
        // SAME node "A", now living at path [0, 0] inside the new Group.
        assert_eq!(scene.active, Some(node_a_id), "the wrapped node stays selected by id");
        assert_eq!(
            scene.active_path(),
            Some(NodePath::from_indices(vec![0, 0])),
            "the selection now resolves to the child slot inside the Group"
        );
        // The second node is untouched.
        assert_eq!(scene.roots.len(), 2);
        assert!(matches!(scene.root_node(1).content, NodeContent::Tool { .. }));
    }

    /// ADR 0001 step 4 (UI helper): `make_definition_from_active` creates an
    /// `AssemblyDef` in `scene.definitions` and replaces the active node with an
    /// `Instance` of it. The resolved occupancy is unchanged (one stored body
    /// resolved via one instance == the original single node).
    #[test]
    fn make_definition_creates_def_and_instance() {
        let voxels_per_block = 8u32;
        // The fixture already selects top-level node 0 (by id).
        let mut scene = two_box_scene(voxels_per_block);

        // Occupancy of just the active node before the change (resolved alone).
        let before = Scene::single_node(scene.root_node(0).clone())
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
        assert!(matches!(scene.root_node(0).content, NodeContent::Instance(id) if id == def_id));

        // Resolving the (now-instanced) node reproduces the original occupancy.
        // Reuse the live scene's arena + definitions, keeping only the single root.
        let mut after_scene = scene.clone();
        let kept_root = after_scene.roots[0];
        after_scene.roots = vec![kept_root];
        after_scene.active = Some(kept_root);
        let after = after_scene
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
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        ));
        let def_id = scene.make_definition_from_active("House").expect("active node");
        assert_eq!(scene.definitions.len(), 1);
        assert_eq!(scene.roots.len(), 1, "the original node became one instance");

        // The def's own voxel count (one box).
        let one = scene
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(one > 0);

        // Add a second instance — an Instance node referencing the same def appears.
        // B4: `add_instance` now returns the new node's stable id; resolve it by id.
        let instance_id = scene.add_instance(def_id).expect("the def exists");
        assert_eq!(scene.roots.len(), 2, "an Instance node referencing the def appears");
        assert!(matches!(
            scene.node_by_id(instance_id).map(|n| &n.content),
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
        // Node 0 ("A") is already the active selection (the fixture selects it).
        let group_id = scene.group_active().expect("active node");
        let added = scene.add_child_to_group(
            group_id,
            Node::new("child", NodeContent::Part(Part::DebugClouds { seed: 0 })),
        );
        assert!(added, "the wrapped node is a Group so a child can be added");

        let rows = scene.tree_rows();
        let paths: Vec<(Vec<usize>, usize)> =
            rows.iter().map(|(p, _id, d)| (p.indices.clone(), *d)).collect();
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
        // Node 0 ("A") is already the active selection (the fixture selects it).
        let mut scene = two_box_scene(8);
        scene.group_active();
        // The active selection now resolves to the wrapped child at path [0, 0].
        let active_path = scene
            .active_path()
            .expect("a child is selected after grouping");
        assert_eq!(active_path, NodePath::from_indices(vec![0, 0]));
        let node = scene.node_at_path(&active_path).expect("the child resolves by path");
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
            let position = voxel.world_position();
            let key = [
                (position[0] - 0.5).round() as i64 + recentre_voxels[0],
                (position[1] - 0.5).round() as i64 + recentre_voxels[1],
                (position[2] - 0.5).round() as i64 + recentre_voxels[2],
            ];
            *multiset.entry((key, voxel.color_index())).or_insert(0) += 1;
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
            (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            let mut total_from_chunks = 0usize;
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk_coord = [chunk_x, chunk_y, chunk_z];
                        let chunk = scene.resolve_chunk(chunk_coord, voxels_per_block, 0);
                        total_from_chunks += chunk.occupied_count();
                        for voxel in &chunk.occupied {
                            let world_position = voxel.world_position();
                            for axis in 0..3 {
                                let lo = (chunk_coord[axis] * chunk_extent_voxels) as f32;
                                let hi = lo + chunk_extent_voxels as f32;
                                let position = world_position[axis];
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
                size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block],
                size_measurements: None,
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
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ]),
            0,
        );
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
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene_build = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene_build.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        let scene = scene_with_top_level_selected(scene_build, 0);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// ADR 0003 §3i Slice 2a: the new sketch→extrude producer composes through the
    /// chunked resolve identically to the monolithic one — mirrors the SDF parity
    /// harness for a SketchTool leaf. Two cases: a plain rectangle extrude (the box
    /// sugar) and a concave L-shape extrude (the added-value path), both at the app
    /// density and at an off-origin placement so the recentre/cover math is real.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_extrude() {
        use crate::sketch::{PlaneAxis, Sketch, SketchPoint};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // (a) Rectangle extrude (box sugar), placed off-origin on X. Z-up:
        // footprint-extrude-up uses PlaneAxis::Z (profile in XY, extruded along +Z).
        let rect = SketchSolid::extrude(
            Sketch::rectangle(PlaneAxis::Z, 3 * density, 2 * density),
            2 * density as u32,
        );
        let mut rect_node = Node::new(
            "Sketch rect",
            NodeContent::SketchTool {
                producer: rect,
                material: MaterialChoice::Stone,
            },
        );
        rect_node.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let rect_scene = Scene::single_node(rect_node);
        assert_chunked_matches_monolithic(&rect_scene, voxels_per_block, "sketch-rect");

        // (b) Concave L-shape extrude (the added value a box can't make).
        let two = 2 * density;
        let four = 4 * density;
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(four, 0),
            SketchPoint::new(four, two),
            SketchPoint::new(two, two),
            SketchPoint::new(two, four),
            SketchPoint::new(0, four),
        ];
        let l_extrude =
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 3 * density as u32);
        let mut l_node = Node::new(
            "Sketch L",
            NodeContent::SketchTool {
                producer: l_extrude,
                material: MaterialChoice::Wood,
            },
        );
        // Off-origin (crossing chunk boundaries on both in-plane axes X and Y) so the
        // off-origin chunked path is proven on the concave/reflex shape, not just the
        // convex rectangle above. (Z-up: the L footprint lives in the XY ground plane.)
        l_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let l_scene = Scene::single_node(l_node);
        assert_chunked_matches_monolithic(&l_scene, voxels_per_block, "sketch-L");
    }

    /// ADR 0003 §3i: the revolve operation composes through the chunked resolve
    /// identically to the monolithic one — mirrors the extrude parity harness for a
    /// solid of revolution. A rectangle revolved 360° about Z (a cylinder) placed
    /// off-origin on X+Y so the recentre/cover math is real and the disc crosses
    /// chunk boundaries on both radial axes.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // PlaneAxis::X + RevolveAxis::InPlane1 ⇒ axial = Z (vertical), radial = {X, Y}.
        // (a) Profile (radial, axial) = rectangle(radial = 2 blocks, axial = 3 blocks)
        // ⇒ a 4-block-diameter, 3-block-tall cylinder. EVEN radial + whole-block axial.
        let revolve = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 2 * density, 3 * density),
            RevolveAxis::InPlane1,
            360,
        );
        let mut node = Node::new(
            "Sketch revolve",
            NodeContent::SketchTool {
                producer: revolve,
                material: MaterialChoice::Stone,
            },
        );
        // Off-origin so the covering chunk range and recentre offset are non-trivial.
        node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let scene = Scene::single_node(node);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "sketch-revolve");

        // (b) ODD axial extent (NOT a whole number of blocks) with an even radial, so
        // the even-radial diameter + odd-axial block-rounding combo is exercised through
        // the chunked path. Radial 30 voxels (diameter 60), axial 2·16 + 5 = 37 voxels.
        let revolve_odd_axial = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 30, 2 * density + 5),
            RevolveAxis::InPlane1,
            360,
        );
        let mut odd_node = Node::new(
            "Sketch revolve odd axial",
            NodeContent::SketchTool {
                producer: revolve_odd_axial,
                material: MaterialChoice::Wood,
            },
        );
        odd_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let odd_scene = Scene::single_node(odd_node);
        assert_chunked_matches_monolithic(&odd_scene, voxels_per_block, "sketch-revolve-odd-axial");
    }

    /// A scene with a single node shifted well OFF the origin (+8 blocks on X) —
    /// proves the chunked path handles off-centre placement (the AABB does not
    /// start at the origin, so the covering chunk range is non-trivial and the
    /// recentre offset is non-zero).
    #[test]
    fn chunked_resolve_matches_monolithic_for_offset_node() {
        let voxels_per_block = 16;
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Offset sphere",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        node.transform = NodeTransform::from_blocks([8, 0, 0], voxels_per_block);
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
        let chunk_extent = crate::core_geom::CHUNK_BLOCKS * 16;
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
    // A node's whole-block offset is `[i64; 3]` (widened in S4a); 100_000 blocks is comfortably
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
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks([offset_blocks, 0, 0], voxels_per_block);
        let scene = Scene::single_node(node);

        // The ABSOLUTE-coordinate chunk path (no recentre): these positions are the
        // scene's TRUE composite coordinates, so they reveal the far placement that
        // the render recentre hides.
        let absolute = scene.resolve_region_via_chunks(voxels_per_block, 0);
        assert!(
            absolute.occupied_count() > 0,
            "the far box must resolve to voxels"
        );

        // CORNER-ANCHORING: every voxel's absolute X centre lands in the far block's
        // voxel span. The 4-block box CORNER-ANCHORED at block 100_000 spans blocks
        // [100_000, 100_004), i.e. absolute voxels [100_000·d, 100_004·d). (Y/Z start
        // at 0.) The box's geometric centre is `off·d + 2·d`.
        let density = voxels_per_block as f32;
        let span_lo = offset_blocks as f32 * density;
        let span_hi = (offset_blocks + 4) as f32 * density;
        let expected_centre_voxels = (offset_blocks as f32 + 2.0) * density; // 1_600_032
        for voxel in &absolute.occupied {
            let x = voxel.world_position()[0];
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
            .map(|v| v.world_position()[0] as f64)
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
            (crate::core_geom::CHUNK_BLOCKS * voxels_per_block) as i64;
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
            offset_blocks * voxels_per_block as i64 + 2 * voxels_per_block as i64,
            "CORNER-ANCHORING: the recentre is the box's geometric CENTRE `off·d + 2·d` \
             (corner `off·d` + half the 4-block extent) — it is what hides the far \
             offset from the live render today (S4 removes it)"
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
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut leaf = Node::new(
            "Leaf",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        leaf.transform = NodeTransform::from_blocks([leaf_offset, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![NodeBuilder::group_at(
                "Group",
                [group_offset, 0, 0],
                voxels_per_block,
                vec![leaf.into()],
            )]),
            0,
        );

        // CORNER-ANCHORING: the producer-true voxel AABB (pure i64) is `[off·d,
        // off·d + grid)` — the composed offset IS the low corner (block-aligned for a
        // whole-block offset). The point of THIS test is the exact i64 composition
        // (no i32 overflow).
        let index = scene.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(index.entries.len(), 1, "exactly one leaf is indexed");
        let aabb = index.entries[0].world_aabb;
        let composed_voxels = composed_blocks * density; // 48_000_000_000 — past i32 too
        assert_eq!(
            aabb.min[0], composed_voxels,
            "the composed leaf min-X must equal (group+leaf)·d (corner-anchored), \
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
        let chunk_extent = (crate::core_geom::CHUNK_BLOCKS as i64) * density;
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
        let mut matched = Vec::new();
        scene.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces| {
            let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block) else {
                return; // region-spanning leaf — not an AABB match.
            };
            let mut min = [0i64; 3];
            let mut max = [0i64; 3];
            for axis in 0..3 {
                // Corner-anchored span `[off, off + grid)` (offset is the low corner),
                // the same maths `build_leaf_spatial_index` now uses.
                let grid = grid_voxels[axis];
                min[axis] = world_offset_voxels[axis];
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
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ]),
            0,
        );
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    fn demo_village_scene(voxels_per_block: u32) -> Scene {
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        scene.voxels_per_block = voxels_per_block;
        scene_with_top_level_selected(scene, 0)
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
                GeometryParams { shape: ShapeKind::Sphere, size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block], size_measurements: None, voxels_per_block, wall_blocks: 1 },
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
        scene_b.root_node_mut(1).transform = NodeTransform::from_blocks([40, 0, 0], voxels_per_block);
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        let moved = index_b.edit_aabb_since(&index_a).expect("same density");
        assert!(!moved.is_empty());
        // CORNER-ANCHORING: a 5-block box spans `[off·d, off·d + 5d)` = `[off·16,
        // off·16 + 80)`. Old at +8: [128, 208). New at +40: [640, 720). The union
        // must contain both.
        assert!(moved.min[0] <= 8 * 16, "edit AABB must cover the OLD location");
        assert!(moved.max[0] >= 40 * 16 + 80, "edit AABB must cover the NEW location");

        // Recolour the Sphere (node 0, same box): edit AABB is just that box.
        let mut scene_c = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_c.root_node_mut(0).content {
            *material = MaterialChoice::Wood;
        }
        let index_c = scene_c.build_leaf_spatial_index(voxels_per_block);
        let recoloured = index_c.edit_aabb_since(&index_a).expect("same density");
        assert!(!recoloured.is_empty(), "a same-box content change is still dirty");
        // CORNER-ANCHORING: Sphere at origin, 5 blocks → span [0, 5·16) = [0, 80).
        assert_eq!(recoloured, crate::spatial_index::VoxelAabb::new([0, 0, 0], [80, 80, 80]));
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
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [5, 5, 5], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        tool.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let part = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 1 }));
        let scene_a = scene_with_top_level_selected(Scene::from_nodes(vec![tool.clone(), part]), 0);
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);
        assert!(index_a.has_region_spanning_leaf);

        // Change the Part's seed (a region-spanning content change).
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).content = NodeContent::Part(Part::DebugClouds { seed: 2 });
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
                size_voxels: [size_blocks[0] * voxels_per_block, size_blocks[1] * voxels_per_block, size_blocks[2] * voxels_per_block],
                size_measurements: None,
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
            let world_position = voxel.world_position();
            for axis in 0..3 {
                // Voxel centres sit at `n + 0.5`; `floor` recovers the cell index.
                let index = world_position[axis].floor() as i64;
                min[axis] = min[axis].min(index);
                max[axis] = max[axis].max(index + 1); // half-open upper bound
            }
        }
        (min, max, grid.occupied.len())
    }

    /// Assert that a box of `size_blocks` resolved at `density` (at world offset 0) is
    /// CORNER-ANCHORED at the origin in the ABSOLUTE frame: it generates exactly
    /// `prod(size·d)` voxels occupying the span `[0, size·d)` per axis, with every
    /// voxel centre on a half-integer (the global voxel lattice).
    ///
    /// CHANGED (corner-anchoring): the producer corner-emits, so its world offset is
    /// the LOW CORNER. In the ABSOLUTE (non-recentred) frame a zero-offset box spans
    /// `[0, size·d)`, NOT the old centred `[−size·d/2, size·d/2)`. The recentre then
    /// symmetrises it for the render frame (see the recentred-frame tests).
    fn assert_box_corner_at_origin(size: [u32; 3], density: u32) {
        let (min, max, count) = absolute_box_extent(size, density);
        let expected_count = (size[0] * density) as usize
            * (size[1] * density) as usize
            * (size[2] * density) as usize;
        assert_eq!(
            count, expected_count,
            "a {size:?}-block box at density {density} must generate prod(size·d) voxels"
        );
        for (axis, &size_axis) in size.iter().enumerate() {
            let grid = (size_axis * density) as i64;
            assert_eq!(
                min[axis], 0,
                "axis {axis}: corner-anchored min is 0 (size {size:?} @ {density})"
            );
            assert_eq!(
                max[axis], grid,
                "axis {axis}: corner-anchored max is size·d (size {size:?} @ {density})"
            );
        }
    }

    /// A 1×1×1-block box (size 1, ODD) at density `d` and offset 0 generates exactly
    /// `d³` voxels CORNER-ANCHORED at the origin: the absolute span is `[0, d)` per
    /// axis. Across the representative density set — d=1 (→ 1 voxel), d=2, d=15
    /// (→ 15³ = 3375), d=16 (default → 4096), d=32.
    ///
    /// CHANGED (corner-anchoring): the absolute span is `[0, d)` (offset = low corner),
    /// not the old centred `[−d/2, d/2)`.
    #[test]
    fn one_block_box_corner_anchored_across_densities() {
        for density in [1u32, 2, 15, 16, 32] {
            assert_box_corner_at_origin([1, 1, 1], density);
        }
    }

    /// An odd-sized shape (5×5×2) is CORNER-ANCHORED at the origin across densities:
    /// it generates `(5d)×(5d)×(2d)` voxels spanning `[0, size·d)`.
    ///
    /// CHANGED (corner-anchoring): the absolute span is `[0, size·d)` (offset = low
    /// corner), at d ∈ {1, 15, 16}. ODD `size·d` (d=15) no longer straddles voxel
    /// cells — every centre is a half-integer.
    #[test]
    fn odd_size_shape_corner_anchored_at_origin() {
        for density in [1u32, 15, 16] {
            assert_box_corner_at_origin([5, 5, 2], density);
        }
    }

    /// An even-sized shape (2×4×6) corner-anchored at the origin spans `[0, size·d)`,
    /// at d ∈ {1, 15, 16}.
    #[test]
    fn even_size_shape_corner_anchored_at_origin() {
        for density in [1u32, 15, 16] {
            assert_box_corner_at_origin([2, 4, 6], density);
            // Corner-anchored: the absolute min corner is 0 (offset = low corner).
            let size = [2u32, 4, 6];
            let (min, _max, _count) = absolute_box_extent(size, density);
            for (axis, &min_axis) in min.iter().enumerate() {
                assert_eq!(
                    min_axis, 0,
                    "axis {axis} @ d{density}: a corner-anchored box starts at index 0"
                );
            }
        }
    }

    /// The bounding box of the OCCUPIED VOXEL CENTRES for a single `shape` of
    /// `size_blocks` placed at world offset `[0, 0, 0]`, resolved at `density` in the
    /// **recentred render frame** ([`resolve_region`] — the frame the camera, gizmo
    /// and renderer use, which centres the composite on the origin). Returns
    /// `(min_centre, max_centre)` per axis (centres sit at `n + 0.5`). A shape is
    /// centred on the origin iff `min_centre + max_centre == 0` per axis.
    ///
    /// We assert on voxel CENTRES, not corners. CORNER-ANCHORING: the producer
    /// corner-emits (`[0, grid)`) and the recentre `floor(grid/2)` lands the composite
    /// in the render frame. For an EVEN voxel span the centre bbox is exactly symmetric
    /// (`min + max == 0`); for an ODD span the floor-recentre leaves it off by exactly
    /// one voxel (`min + max == 1`), since an odd extent has no voxel-centred origin.
    fn occupied_voxel_centre_bbox(
        shape: ShapeKind,
        size_blocks: [u32; 3],
        density: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape,
                size_voxels: [size_blocks[0] * density, size_blocks[1] * density, size_blocks[2] * density],
                size_measurements: None,
                voxels_per_block: density,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        // The recentred frame (what the renderer/camera/gizmo see): the composite's
        // block AABB is centred on the origin.
        let region = scene.full_extent_blocks(density);
        let grid = scene.resolve_region(region, density, 0);
        assert!(!grid.occupied.is_empty(), "shape {shape:?} {size_blocks:?}@{density} resolved empty");
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for voxel in &grid.occupied {
            let world_position = voxel.world_position();
            for axis in 0..3 {
                min[axis] = min[axis].min(world_position[axis]);
                max[axis] = max[axis].max(world_position[axis]);
            }
        }
        (min, max)
    }

    /// PERMANENT GUARD: a shape placed at world offset `[0, 0, 0]` is centred (to
    /// sub-voxel precision) on the origin in the **recentred monolithic
    /// `resolve_region` frame** — its occupied-voxel-CENTRE bounding box is symmetric
    /// about 0 within half a voxel on every axis. CORNER-ANCHORING: the centre-sum is
    /// `grid mod 2` — exactly 0 for an EVEN voxel span, exactly 1 for an ODD one (an
    /// odd extent has no voxel-centred origin; the floor-recentre biases it one voxel).
    ///
    /// This is the MONOLITHIC resolve frame; it is BIT-IDENTICAL to the windowed-app
    /// render path for a near scene (the per-chunk store applies the same recentre).
    /// Covers a 5×5×5 sphere (odd) and a 5×1×5 box (odd-X/Z, 1-block-Y).
    #[test]
    fn shape_centered_within_half_voxel_in_resolve_region_frame() {
        let cases: [(ShapeKind, [u32; 3]); 2] =
            [(ShapeKind::Sphere, [5, 5, 5]), (ShapeKind::Box, [5, 1, 5])];
        for density in [1u32, 8, 16] {
            for (shape, size) in cases {
                let (min, max) = occupied_voxel_centre_bbox(shape, size, density);
                for axis in 0..3 {
                    let grid = size[axis] * density;
                    let expected = (grid % 2) as f32; // 0 even, 1 odd
                    let centre_sum = min[axis] + max[axis];
                    assert_eq!(
                        centre_sum, expected,
                        "{shape:?} {size:?}@d{density} axis {axis}: voxel-centre bbox \
                         [{}, {}] sum must be grid%2 = {expected} (corner-anchored recentre)",
                        min[axis], max[axis]
                    );
                }
            }
        }
    }

    /// HEADLINE WIN (corner-anchoring): an ODD extent at ODD DENSITY (d=1) lands on the
    /// voxel lattice — every centre is a HALF-INTEGER, sitting strictly INSIDE its
    /// voxel cell `[k, k+1)`. This is the exact case the old centred-emit got wrong:
    /// at odd grid the centred convention put centres on INTEGERS (`idx + 0.5 − grid/2`
    /// = whole numbers), straddling cell boundaries — visibly off the global voxel
    /// grid. Corner-emit (`idx + 0.5`) makes every centre a half-integer for ANY parity.
    ///
    /// A 3×1×3 box @ d=1, recentred (recentre = floor(grid/2) = 1 on X/Z): X/Z centres
    /// are `idx + 0.5 − 1` = {−0.5, 0.5, 1.5}; Y centre = 0.5. Nine voxels, every centre
    /// a half-integer.
    #[test]
    fn odd_extent_at_odd_density_lands_on_voxel_lattice() {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 3], 1, 1);
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            )]),
            0,
        );
        let density = 1u32;
        let region = scene.full_extent_blocks(density);
        let grid = scene.resolve_region(region, density, 0);

        assert_eq!(grid.occupied.len(), 9, "3×1×3 box @ d=1 is a full 9-cell prism (3·1·3)");

        // THE WIN: every voxel centre is a half-integer (frac == 0.5) — on the lattice,
        // inside a cell — NOT an integer straddling a boundary (the old odd-grid bug).
        for voxel in &grid.occupied {
            for (axis, pos) in voxel.world_position().into_iter().enumerate() {
                assert_eq!(
                    pos.fract().abs(), 0.5,
                    "axis {axis} centre {pos} must be a HALF-INTEGER (on the voxel lattice) at d=1 odd extent"
                );
            }
        }

        // The recovered cells (floor of the recentred centre) are the symmetric set
        // {−1, 0, 1} on X/Z and {0} on Y.
        use std::collections::BTreeSet;
        let cells: BTreeSet<[i64; 3]> = grid
            .occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    position[0].floor() as i64,
                    position[1].floor() as i64,
                    position[2].floor() as i64,
                ]
            })
            .collect();
        let mut expected: BTreeSet<[i64; 3]> = BTreeSet::new();
        for x in [-1i64, 0, 1] {
            for z in [-1i64, 0, 1] {
                expected.insert([x, 0, z]);
            }
        }
        assert_eq!(cells, expected, "odd 3×1×3 @ d=1 cells are {{−1,0,1}}²×{{0}}");
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
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
        let index = scene.build_leaf_spatial_index(density);
        assert_eq!(index.entries.len(), 1, "one Tool leaf → one index entry");
        index.entries[0].world_aabb
    }

    /// The `NodeTransform` block/voxel accessors round-trip (incl. negatives), the
    /// mating predicate distinguishes block-aligned from sub-block offsets, and a
    /// 0-density document cannot panic (density clamped to ≥1 like the resolve
    /// sites). ADR 0003 §3f(0).
    #[test]
    fn node_transform_accessors_round_trip_and_guard() {
        // Round-trip through canonical voxels, including negative components.
        let transform = NodeTransform::from_blocks([-3, 2, -7], 16);
        assert_eq!(transform.offset_voxels, [-48, 32, -112], "blocks·d = voxels");
        assert_eq!(transform.blocks(16), [-3, 2, -7], "blocks(d) inverts from_blocks");
        assert!(transform.block_aligned(16), "a whole-block offset is on the lattice");

        // A hand-set SUB-block offset is NOT block-aligned (the mating predicate).
        let sub_block = NodeTransform { offset_voxels: [1, 0, 0], ..Default::default() };
        assert!(
            !sub_block.block_aligned(16),
            "an offset of 1 voxel at d=16 is off the block lattice"
        );

        // A 0-density document must not panic: density is clamped to ≥1.
        let _ = NodeTransform::from_blocks([2, 0, 0], 0);
        let zero_density = NodeTransform { offset_voxels: [2, 0, 0], ..Default::default() };
        let _ = zero_density.blocks(0);
        let _ = zero_density.block_aligned(0);
    }

    /// A `B`-block extent → a `B·d`-voxel AABB CORNER-ANCHORED at the node's world
    /// offset, at each density. This is the geometry the per-object block lattice /
    /// floor / voxel grid (#29) will span.
    ///
    /// CHANGED (corner-anchoring): the AABB is the producer-true span
    /// `[off·d, off·d + size·d)` — the offset IS the low corner. For a whole-block
    /// offset the corner is a block multiple of `d` at ANY size parity (no more
    /// half-block straddle for odd sizes).
    #[test]
    fn node_block_aabb_scales_and_corner_anchors_across_densities() {
        let size = [5u32, 5, 2]; // a representative mixed (odd X/Y, even Z) extent
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let aabb = single_leaf_aabb(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                let grid = (size_axis * density) as i64;
                let off_voxels = offset[axis] * density as i64;
                // Scales with density: a B-block extent → B·d voxels.
                assert_eq!(
                    aabb.max[axis] - aabb.min[axis],
                    grid,
                    "axis {axis} @ d{density}: AABB extent must be size·d voxels"
                );
                // Corner-anchored: the offset is the LOW corner.
                assert_eq!(
                    aabb.min[axis], off_voxels,
                    "axis {axis} @ d{density}: AABB min corner is off·d (corner-anchored)"
                );
                assert_eq!(
                    aabb.max[axis], off_voxels + grid,
                    "axis {axis} @ d{density}: AABB max corner is off·d + size·d"
                );
                // A whole-block offset → block-aligned corner at ANY size parity.
                assert_eq!(
                    aabb.min[axis].rem_euclid(density as i64), 0,
                    "axis {axis} @ d{density}: a whole-block offset is block-aligned"
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its AABB by
    /// exactly `d` voxels per axis (the grids/gizmo follow it), and the AABB stays
    /// block-aligned, at each density. A node's placement here is a whole-block
    /// offset (`[i64; 3]` blocks), so sub-block translation is not representable —
    /// whole-block translation is the unit tested.
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
                    // The corner's lattice RESIDUE is preserved by a whole-block move
                    // (a +d shift can't change `min mod d`). We no longer require it to
                    // be 0 — an odd extent is centred on the offset, half a block off
                    // the lattice (center-anchoring retirement) — only that the move
                    // doesn't perturb it.
                    assert_eq!(
                        after.min[axis].rem_euclid(density as i64),
                        before.min[axis].rem_euclid(density as i64),
                        "axis {axis} @ d{density}: a whole-block translate preserves the corner's lattice residue"
                    );
                }
            }
        }
    }

    /// The node pivot/origin the selection transform gizmo (#29) will track: the
    /// node's world origin = `offset_in_blocks·d − recentre`, in the recentred frame.
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
    ///    origin is `offset_in_blocks·d`; this DOES follow a `+1 block` translate by
    ///    exactly `d` voxels per axis (the property the global-frame gizmo inherits).
    #[test]
    fn node_pivot_origin_tracks_offset_across_densities() {
        let size = [5u32, 5, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let recentre_of = |offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
                let mut node =
                    Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
                node.transform = NodeTransform::from_blocks(offset, density);
                let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
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

    // ---- issue #29 (grid rework S3): per-object block lattice box (renderer-follow) ----

    /// Build a single-Box-node scene at `offset`, return its
    /// `node_block_lattice_box_recentred` for node 0 at `density`.
    fn single_node_lattice_box(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        density: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
        scene
            .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
            .expect("a sized Box node has a lattice box")
    }

    /// The per-object lattice box spans the node's enclosing-block AABB and SCALES
    /// with density: a `B`-block extent → a `B·d`-voxel box, at each density
    /// {1, 15, 16} (the explicit user ask).
    ///
    /// The producer-true corner geometry is asserted in
    /// `node_block_aabb_scales_and_centres_across_densities` — in the RECENTRED frame
    /// the box is shifted by the composite recentre, so the recentred corners need not
    /// be block multiples; the block-aligned STRUCTURE (extent = B·d, planes step d)
    /// is what survives the recentre, and that is what this asserts.
    #[test]
    fn lattice_box_spans_enclosing_blocks_and_scales_with_density() {
        let size = [5u32, 3, 2];
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let (min, max) = single_node_lattice_box(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                // Box extent = size · density voxels (B-block extent → B·d voxels).
                assert_eq!(
                    (max[axis] - min[axis]) as i64,
                    (size_axis * density) as i64,
                    "axis {axis} @ d{density}: lattice box extent must be size·d voxels"
                );
                // The extent is an exact multiple of a block, so the box encloses
                // exactly `size_axis` whole blocks along each axis.
                assert_eq!(
                    ((max[axis] - min[axis]) as i64).rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: box extent spans whole blocks"
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its lattice box
    /// by exactly `density` voxels per axis (the lattice follows the object), at each
    /// density {1, 15, 16}. Because the node offset is whole-block, a SUB-block
    /// (1-voxel) translate is NOT representable at the node level, so the
    /// "add/remove a whole block on a sub-block move" requirement cannot be
    /// constructed here; the whole-block follow IS the unit tested. (The
    /// expand-to-block that WOULD turn a sub-block shift into a whole-block box
    /// change is exercised directly on `block_boundaries`/`*_vertices_into` in the
    /// renderer tests.)
    #[test]
    fn lattice_box_follows_whole_block_translate_at_each_density() {
        let size = [5u32, 3, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            // A SECOND, LARGE anchor node (centred at the origin, ±100 blocks on
            // every axis) dominates the composite AABB on all axes, so the small
            // moving node never touches a composite corner and the recentre stays
            // FIXED. Observed in that fixed frame, moving the node by +1 block shifts
            // its box by exactly d — the "lattice follows the object in the global
            // lattice frame" property. (A lone node would drag its own recentre, so
            // the box would NOT appear to move — see `node_pivot_origin_*`.)
            let make_scene = |offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
                let mut moving = Node::new(
                    "Moving",
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                moving.transform = NodeTransform::from_blocks(offset, density);
                let anchor_shape = SdfShape::from_blocks(ShapeKind::Box, [200, 200, 200], 1, density);
                let mut anchor = Node::new(
                    "Anchor",
                    NodeContent::Tool { shape: anchor_shape, material: MaterialChoice::Stone },
                );
                // CORNER-ANCHORING: a leaf spans `[off, off+size)` blocks, so to make
                // the 200³ anchor BRACKET the small moving node on every axis (and so
                // dominate the composite AABB, fixing the recentre) it must be offset to
                // `[−100, 100)` blocks, not corner-anchored at the origin.
                anchor.transform = NodeTransform::from_blocks([-100, -100, -100], density);
                scene_with_top_level_selected(Scene::from_nodes(vec![moving, anchor]), 0)
            };
            let box_of = |offset: [i64; 3]| {
                make_scene(offset)
                    .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
                    .expect("moving node has a lattice box")
            };
            let before = box_of(base);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let after = box_of(shifted);
                for axis in 0..3 {
                    let expected = if axis == moved_axis { density as f32 } else { 0.0 };
                    assert_eq!(
                        after.0[axis] - before.0[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block on axis {moved_axis} must shift the \
                         lattice box min by exactly d (0 elsewhere)"
                    );
                    assert_eq!(
                        after.1[axis] - before.1[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block must shift the lattice box max by d"
                    );
                }
            }
        }
    }

    /// A size-less node (a Part with no intrinsic extent — `DebugClouds`) has NO
    /// lattice box: `node_block_lattice_box_recentred` returns `None` (nothing to
    /// draw), at each density.
    #[test]
    fn sizeless_node_has_no_lattice_box() {
        for density in [1u32, 15, 16] {
            let scene = Scene::single_node(Node::new(
                "Clouds",
                NodeContent::Part(Part::DebugClouds { seed: 0 }),
            ));
            assert_eq!(
                scene.node_block_lattice_box_recentred(&NodePath::root_index(0), density),
                None,
                "@ d{density}: a size-less node yields no lattice box"
            );
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
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        assert!(!node.grids.voxel_grid_on_faces);
        assert!(!node.grids.block_lattice);
        assert!(!node.grids.floor_grid);
        assert_eq!(node.grids, NodeGrids::default());
    }

    /// An empty `Scene::default()` has the issue-#29 grid-rework master defaults:
    /// ALL THREE masters ON (per-object flags stay OFF), and no Points yet.
    #[test]
    fn scene_default_master_grids() {
        let scene = Scene::default();
        assert!(scene.master_block_lattice, "block lattice master defaults ON");
        assert!(scene.master_voxel_grid, "voxel grid master defaults ON");
        assert!(scene.master_floor_grid, "floor grid master defaults ON");
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
        // Z-up: the ground plane is XY (`plane_xy`).
        assert!(origin.plane_xy, "ground plane (XY) on by default");
        assert!(origin.axis_x && origin.axis_y && origin.axis_z, "all axes on by default");
        assert!(!origin.plane_xz && !origin.plane_yz);
        assert!(!origin.hidden);

        // Idempotent: a second call does not add another Origin.
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "second call adds nothing");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// ADR 0003 Phase B: `ensure_node_ids` mints a unique non-zero id for every
    /// node — top-level, Group children, and definition nodes — and is idempotent.
    #[test]
    fn ensure_node_ids_mints_unique_stable_ids() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::Part(Part::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group("G", vec![clouds("B").into(), clouds("C").into()]),
        ]);
        scene.add_definition(DefId(1), "Def".to_string(), vec![clouds("D")]);

        scene.ensure_node_ids();

        // Collect every id (top-level + Group children + definition nodes). Every node
        // lives in the arena keyed by its id, so the arena keys ARE the full id set.
        let ids: Vec<NodeId> = scene.arena.keys().copied().collect();
        assert_eq!(ids.len(), 5, "A, G, B, C, D all visited");
        assert!(ids.iter().all(|&id| id != NodeId(0)), "no node keeps the 0 sentinel");
        let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "every minted id is unique");

        // Idempotent: a second pass mints nothing and changes no id.
        let before = scene.clone();
        scene.ensure_node_ids();
        assert_eq!(scene, before, "second call is a no-op");
    }

    /// A loaded scene that already carries an id keeps it, and the counter advances
    /// past it so a newly-minted node never collides.
    #[test]
    fn ensure_node_ids_preserves_existing_and_advances_counter() {
        // A loaded scene: the arena is keyed by id, so a node that already carries a
        // minted id (the "preset", id 5) lives under key NodeId(5), while a still-
        // unminted node sits under the NodeId(0) sentinel. `next_node_id` starts at 0,
        // as it would for a freshly-deserialized scene before normalization.
        let mut preset = Node::new("preset", NodeContent::Part(Part::DebugClouds { seed: 0 }));
        preset.id = NodeId(5);
        let mut fresh = Node::new("fresh", NodeContent::Part(Part::DebugClouds { seed: 0 }));
        fresh.id = NodeId(0);
        let mut scene = Scene::default();
        scene.arena.insert(NodeId(5), preset);
        scene.arena.insert(NodeId(0), fresh);
        scene.roots = vec![NodeId(5), NodeId(0)];

        scene.ensure_node_ids();

        // The preset id is preserved verbatim.
        assert!(scene.arena.contains_key(&NodeId(5)), "existing id preserved");
        assert_eq!(scene.arena[&NodeId(5)].name, "preset");
        // The unminted node was re-keyed out of the 0 sentinel into a fresh, distinct id.
        assert!(!scene.arena.contains_key(&NodeId(0)), "the 0 sentinel is gone");
        let fresh_id = scene
            .arena
            .iter()
            .find(|(_, node)| node.name == "fresh")
            .map(|(id, _)| *id)
            .expect("the fresh node still exists under a minted id");
        assert_ne!(fresh_id, NodeId(0), "fresh node minted");
        assert_ne!(fresh_id, NodeId(5), "fresh id does not collide with the existing one");
        assert!(scene.next_node_id > 5, "counter advanced past the loaded id");
        // Re-keying must repoint the SPINE, not just move the arena entry: the root slot
        // that referenced the sentinel now names the fresh id, so the node is still
        // reachable through `roots` (a stale NodeId(0) here would silently orphan it).
        assert_eq!(scene.roots[1], fresh_id, "the root spine slot was repointed off the sentinel");
        assert_eq!(
            scene.node_at_path(&NodePath::root_index(1)).map(|node| node.name.as_str()),
            Some("fresh"),
            "the re-keyed node still resolves through the spine, not orphaned",
        );
    }

    /// ADR 0003 Phase B2: `id_at_path` / `path_of` / `node_by_id` agree with the
    /// positional `node_at_path` for EVERY node in the tree (the ⇄ equivalence the
    /// later selection/command migration relies on).
    #[test]
    fn node_id_and_path_resolution_round_trip() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::Part(Part::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group(
                "G",
                vec![
                    clouds("B").into(),
                    NodeBuilder::group("H", vec![clouds("C").into()]),
                ],
            ),
            NodeBuilder::Leaf(clouds("D")),
        ]);
        scene.ensure_node_ids();

        // Every tree row resolves both ways, consistently.
        for (path, row_id, _depth) in scene.tree_rows() {
            let id = scene.id_at_path(&path).expect("path resolves to an id");
            assert_eq!(id, row_id, "the row's carried id matches id_at_path");
            assert_ne!(id, NodeId(0), "a minted node never has the 0 sentinel");
            assert_eq!(
                scene.path_of(id),
                Some(path.clone()),
                "path_of inverts id_at_path"
            );
            // node_by_id and node_at_path reach the SAME node.
            let by_id = scene.node_by_id(id).expect("id resolves to a node");
            let by_path = scene.node_at_path(&path).expect("path resolves to a node");
            assert_eq!(by_id.id, by_path.id);
            assert_eq!(by_id.name, by_path.name);
        }

        // Sentinel + unknown ids resolve to nothing.
        assert!(scene.node_by_id(NodeId(0)).is_none());
        assert!(scene.path_of(NodeId(0)).is_none());
        assert!(scene.node_by_id(NodeId(9_999)).is_none());
        assert!(scene.path_of(NodeId(9_999)).is_none());

        // Mutable lookup reaches the same node.
        let first_id = scene.id_at_path(&NodePath::root_index(0)).unwrap();
        scene.node_by_id_mut(first_id).unwrap().name = "renamed".to_string();
        assert_eq!(scene.node_at_path(&NodePath::root_index(0)).unwrap().name, "renamed");
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

    /// `add_point` gives a newly-added user Point the clean default (issue #29 fix):
    /// **all planes OFF** with **all three axes ON** — even if the caller passes a
    /// Point with planes enabled. Only the Origin (built by `ensure_origin_point`,
    /// not `add_point`) keeps the ground (XY, Z-up) plane on.
    #[test]
    fn add_point_defaults_planes_off_axes_on() {
        let mut scene = Scene::default();
        // Pass a Point with EVERY plane on; add_point must override them off.
        scene.add_point(Point {
            name: "User".to_string(),
            plane_xz: true,
            plane_xy: true,
            plane_yz: true,
            axis_x: false,
            axis_y: false,
            axis_z: false,
            ..Point::default()
        });
        let point = &scene.points[0];
        assert!(!point.plane_xz && !point.plane_xy && !point.plane_yz, "new point: all planes OFF");
        assert!(point.axis_x && point.axis_y && point.axis_z, "new point: all axes ON");

        // The Origin (via ensure_origin_point) still keeps the ground plane on
        // (Z-up: ground = XY = `plane_xy`).
        let mut origin_scene = Scene::default();
        origin_scene.ensure_origin_point();
        assert!(origin_scene.points[0].plane_xy, "Origin keeps the ground plane (XY)");
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
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        node.grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let mut built = Scene::from_nodes(vec![node]);
        built.master_block_lattice = false;
        built.master_voxel_grid = true;
        built.master_floor_grid = true;
        built.active_point = Some(1);
        let mut scene = scene_with_top_level_selected(built, 0);
        scene.ensure_origin_point();
        // Push directly (not via `add_point`, which overrides plane/axis flags to the
        // new-point default) so the round-trip exercises non-default per-axis flags.
        scene.points.push(Point {
            name: "Corner".to_string(),
            position_blocks: [3, 4, 5],
            plane_xz: false,
            plane_xy: true,
            plane_yz: true,
            axis_x: true,
            axis_y: false,
            axis_z: true,
            hidden: true,
            ..Point::default()
        });

        let json = serde_json::to_string_pretty(&scene).expect("serialise");
        let restored: Scene = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(scene, restored, "scene with grids + points round-trips");
        assert!(restored.root_node(0).grids.voxel_grid_on_faces);
        assert!(restored.root_node(0).grids.floor_grid);
        assert!(!restored.master_block_lattice);
        assert!(restored.master_voxel_grid);
        assert_eq!(restored.points.len(), 2);
        assert_eq!(restored.points[1].position_blocks, [3, 4, 5]);
        // Per-axis flags survive the round-trip (issue #29 fix: split axes).
        assert!(restored.points[1].axis_x && !restored.points[1].axis_y && restored.points[1].axis_z);
    }

    /// Back-compat: an OLD serialized scene (no `grids`, no `points`, no masters)
    /// deserialises with the correct defaults — node grids all-off, all three
    /// masters at their struct default (ON, issue #29 grid-rework fix), empty points.
    #[test]
    fn old_scene_json_loads_with_grid_defaults() {
        // Build a one-Box scene, serialize it, then STRIP the optional fields that an
        // old document would not carry (the per-node `grids`, the scene-wide masters,
        // `points`, `active_point`). Deserializing the trimmed JSON must fill every
        // missing field with its struct default.
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let mut value = serde_json::to_value(&scene).expect("serialise");
        let object = value.as_object_mut().expect("scene serializes to an object");
        // Drop the optional/defaulted fields so the load path must synthesize them.
        object.remove("master_block_lattice");
        object.remove("master_voxel_grid");
        object.remove("master_floor_grid");
        object.remove("points");
        object.remove("active_point");
        // Strip every node's `grids` so the per-node default (#29 all-off) is exercised.
        if let Some(arena) = object.get_mut("arena").and_then(|a| a.as_object_mut()) {
            for stored in arena.values_mut() {
                if let Some(node_obj) = stored.as_object_mut() {
                    node_obj.remove("grids");
                }
            }
        }
        let old_json = serde_json::to_string(&value).expect("re-serialise trimmed doc");

        let scene: Scene = serde_json::from_str(&old_json).expect("old scene parses");
        assert_eq!(scene.roots.len(), 1);
        assert_eq!(scene.root_node(0).grids, NodeGrids::default(), "grids default off");
        assert!(scene.master_block_lattice, "lattice master default on");
        assert!(scene.master_voxel_grid && scene.master_floor_grid, "all masters default on");
        assert!(scene.points.is_empty(), "no points in the old document");
        assert_eq!(scene.active_point, None);
    }

    /// Issue #29 S2: the transform gizmo's pivot is the SELECTED node's block-AABB
    /// centre in the recentred render frame — `block_aabb_centre·d − recentre` —
    /// `None` when nothing is selected, across densities.
    #[test]
    fn active_gizmo_placement_follows_selected_node() {
        for vpb in [1u32, 15, 16] {
            // Bake each node's whole-block offset at the resolve density `vpb` so the
            // stored voxel offset divides back to the same block offset under this
            // resolution (the gizmo reads `offset_voxels / vpb` → blocks).
            let make_tool = |kind, size: [u32; 3], offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(kind, size, 1, vpb);
                let mut node = Node::new(
                    format!("{kind:?}"),
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                node.transform = NodeTransform::from_blocks(offset, vpb);
                node
            };
            // Three even-sized boxes; box B sits +8X, box C sits +6Z. CORNER-ANCHORING:
            // a 4-block box at offset `off` spans `[off, off+4]` blocks, centre `off+2`.
            let mut scene = Scene::from_nodes(vec![
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [8, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 6]),
            ]);
            scene.active = None;
            // ADR 0003 Phase B3: mint ids so selecting a node by id resolves.
            scene.ensure_node_ids();

            // Nothing selected → no gizmo.
            assert_eq!(
                scene.active_gizmo_placement(vpb),
                None,
                "no selection hides the gizmo (vpb={vpb})"
            );

            let recentre = scene.recentre_voxels_for_resolve(vpb);
            let density = vpb as i64;

            // Expected pivot for a 4-block box at block OFFSET `off`: its geometric
            // centre is `(off + 2)·d` voxels (corner-anchored), minus the recentre.
            let half_extent_voxels = 2 * density; // half of the 4-block extent
            let expected_pivot = |off_blocks: [i64; 3]| {
                [
                    (off_blocks[0] * density + half_extent_voxels - recentre[0]) as f32,
                    (off_blocks[1] * density + half_extent_voxels - recentre[1]) as f32,
                    (off_blocks[2] * density + half_extent_voxels - recentre[2]) as f32,
                ]
            };

            // Select each node in turn; the gizmo pivot tracks it.
            for (index, centre) in [([0, 0, 0]), ([8, 0, 0]), ([0, 0, 6])].into_iter().enumerate() {
                scene.active = scene.id_at_path(&NodePath::root_index(index));
                let (pivot, extent) =
                    scene.active_gizmo_placement(vpb).expect("selection shows the gizmo");
                assert_eq!(
                    pivot,
                    expected_pivot(centre),
                    "pivot == centre·d − recentre for node {index} (vpb={vpb})"
                );
                // Extent is the node's OWN 4-block AABB (not the whole region).
                assert_eq!(
                    extent,
                    [(4 * density) as f32; 3],
                    "gizmo sized from the node's own extent (vpb={vpb})"
                );
            }
        }
    }

    /// Issue #29 S2: a SINGLE selected node recentres onto the origin, so its gizmo
    /// pivot is exactly `[0, 0, 0]` (for an EVEN-sized node, whose block-AABB centre
    /// lands on an integer voxel). The gizmo only visibly moves with a multi-node
    /// selection. Guards against reading the pivot from absolute (un-recentred) space.
    #[test]
    fn single_even_selected_node_gizmo_sits_at_origin() {
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 2, 6], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            assert_eq!(
                pivot,
                [0.0, 0.0, 0.0],
                "the lone even-sized selected node recentres onto the origin (vpb={vpb})"
            );
        }
    }

    /// CHANGED (center-anchoring retirement): for an ODD-sized lone node the gizmo
    /// pivot now sits at WITHIN HALF A VOXEL of the origin for ALL densities —
    /// including the odd-size/odd-density case the old block-lattice shift got wrong
    /// (it left the pivot half a BLOCK off). The gizmo pivot and the composite
    /// recentre are now BOTH derived from the producer-true voxel frame, so a lone
    /// node's centre coincides with the recentre: pivot is exactly 0 for an even voxel
    /// span and ±0.5 voxel for an odd one (the truncation of a half-voxel centre).
    #[test]
    fn single_odd_selected_node_gizmo_is_at_most_half_voxel_off_origin() {
        // Sizes (3, 1, 5) are all odd. The lone node's pivot stays WITHIN half a voxel
        // of origin (NOT half a block, as the retired #30 shift produced) — exactly 0
        // when the voxel span size·d is even, ±0.5 voxel when odd.
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 5], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            for (axis, &component) in pivot.iter().enumerate() {
                assert!(
                    component.abs() <= 0.5,
                    "lone odd-sized node pivot within half a voxel of origin \
                     (axis {axis}, vpb={vpb}, got {component})"
                );
            }
            if vpb % 2 == 0 {
                assert_eq!(
                    pivot, [0.0, 0.0, 0.0],
                    "even density makes the lone-node recentre exact (vpb={vpb})"
                );
            }
        }
    }
}
