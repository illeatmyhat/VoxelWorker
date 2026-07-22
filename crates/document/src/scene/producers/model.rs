//! Leaf model + keying: the leaf content kinds ([`VoxelBody`] / [`NodeContent`]),
//! the walk's scope frames and visitor body ([`ScopeFrame`] / [`LeafBody`] /
//! [`LeafVisitor`] / [`ComposedScope`]), the flat op-stack entry ([`LeafProducer`]),
//! and the pure keying / sizing helpers ([`leaf_content_fingerprint`],
//! [`quat_from_lattice`], [`operation_masks_beyond_bounds`],
//! [`leaf_producer_grid_voxels`], [`outset_voxels_at`]).

use serde::{Deserialize, Serialize};

use voxel_core::core_geom::MaterialChoice;
use crate::debug_clouds::DebugCloudField;
use crate::sketch::SketchSolid;
use crate::voxel::{SdfShape, VoxelProducer};

use crate::scene::*;

/// A *static* voxel body with no meaningful generation parameters — dropped in
/// as-is (ADR 0001). v1 has one variant; future variants are saved chiseled
/// blocks and imported `.vox` bodies, each carrying baked per-voxel materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoxelBody {
    /// The debug cloud field (several distinct billowy fBm blobs) — "a part with
    /// one trivial knob" (the seed).
    DebugClouds {
        /// Seed for the deterministic placement + noise permutation.
        #[serde(default)]
        seed: u32,
    },
    // future: SavedBody(VoxelBlob), ImportedVox(...).
}

/// What a node *is*: a leaf producer (Tool, SketchTool or VoxelBody) or an interior
/// assembly (Group or Instance).
///
/// Every arm resolves: a leaf stamps its own producer, a `Group` folds its children
/// under its own `CombineOp` (ADR 0017), and an `Instance` resolves the referenced
/// definition under its transform (recursion + instancing, ADR 0001's original
/// "step 4" goal) — see `Scene::walk_nodes` / `Scene::for_each_leaf`.
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
    /// [`VoxelBody`]: NodeContent::VoxelBody
    SketchTool {
        /// The sketch + operation to resolve.
        producer: SketchSolid,
        /// The single material this node stamps onto its voxels.
        material: MaterialChoice,
    },
    /// A static voxel body, dropped in as-is.
    VoxelBody(VoxelBody),
    /// An owned, one-off sub-assembly. **ADR 0003 Phase B5:** a Group owns its
    /// children by **identity** — the ordered spine of child [`NodeId`]s — while the
    /// child `Node`s themselves live in the scene-wide [`Scene::arena`]. The `Vec`
    /// order IS document order (resolved later-wins on overlap); the arena is fetched
    /// from but never iterated to produce a walk. Resolved by `Scene::walk_nodes`,
    /// which folds the children under the Group's own `CombineOp` (ADR 0017).
    Group(Vec<NodeId>),
    /// A reuse-by-reference of a definition. Resolved by `Scene::walk_nodes`, which
    /// expands the referenced `AssemblyDef`'s children under the instance's transform
    /// (the cycle guard bars an Instance from re-entering an ancestor definition).
    Instance(DefId),
}

/// One enclosing **sealed composition scope** of a leaf (ADR 0017 Decision 3): a `Group`
/// node, or a definition body expanded under an `Instance` node. The leaf's chain of
/// frames — outermost first — is its **scope path**: the walk emits leaves in depth-first
/// document order, and two consecutive leaves are in the same scope iff their paths are
/// equal, so a consumer reconstructs the scope-open/scope-close markers of the depth-first
/// fold (the push/pop evaluation of the SDF-editor prior art, `docs/design/
/// csg-prior-art-study.md` round 2) by comparing adjacent paths. Carrying the path on each
/// leaf — instead of interleaving marker entries — keeps the flat `LeafProducer` list a
/// plain document-order sequence, so the edit broadphase's positional indexing and the
/// candidate-subsequence filtering stay valid unchanged (dropping a leaf drops nothing but
/// that leaf; an emptied scope simply never opens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeFrame {
    /// The scope node's stable [`NodeId`] — the `Group` node itself, or the `Instance`
    /// node whose referenced definition body this expansion is. Distinguishes two sibling
    /// scopes (and two expansions of the same definition: their instance nodes differ), and
    /// is stable across walks so it can enter the leaf fingerprint (a regroup or a scope-op
    /// flip must dirty the leaves inside — see [`leaf_content_fingerprint`]).
    pub scope_node: NodeId,
    /// The scope node's own [`CombineOp`] — the operation the scope's COMPOSED body folds
    /// under into its parent scope (ADR 0017 Decision 3: a Group/definition pre-composes
    /// its children into one body; that body then folds as a unit).
    pub operation: CombineOp,
}

/// The [`Scene::for_each_leaf`] / [`Scene::walk_nodes`] visitor callback: invoked once per
/// enabled leaf with `(world_offset_voxels, offset_local_voxels, rotation, content,
/// grid_on_faces, operation, outset, scope_path)` — the accumulated world VOXEL offset
/// (integer), the accumulated **continuous** local float offset relative to it (ADR 0027),
/// the leaf's continuous `Quat` rotation (ADR 0027), the leaf content, the node's on-face-grid
/// flag (issue #29 S4), the node's own [`CombineOp`] role in the ordered fold (ADR 0017), the
/// node's outset (ADR 0019 Decision 7), and the chain of enclosing sealed scopes (outermost
/// first — see [`ScopeFrame`]).
///
/// **ADR 0027.** Placement, rotation and off-lattice slide are all carried by the continuous
/// `Quat` and the `[f32; 3]` float offset; the classifier reads that quaternion directly. The
/// `Quat` a leaf carries is `node.transform.rotation()` — the whole tilt seated against the
/// surface it was dropped on.
///
/// The outset arrives as an unevaluated [`Measurement`] because the walk carries no density.
/// Each consumer resolves it against its own `voxels_per_block`, which is what keeps the
/// authored intent (`"1/4 block"`) rather than a number derived at the wrong moment.
///
/// [`Measurement`]: voxel_core::units::Measurement
pub(crate) type LeafVisitor<'walk> = dyn FnMut(
        [i64; 3],
        [f32; 3],
        glam::Quat,
        LeafBody<'_>,
        bool,
        CombineOp,
        voxel_core::units::Measurement,
        &[ScopeFrame],
    ) + 'walk;

/// What a visited leaf actually IS: ordinary document content, or a sealed scope that has
/// been pre-composed into one producer because it carries an outset.
///
/// The second arm exists because ADR 0019 Decision 7 puts an outset on a **scope** — a Part
/// (ADR 0018 Decision 1) or a sealed definition body — and requires it to dilate the scope's
/// COMPOSED body. A scope is already defined as "pre-compose the children into one body"
/// (ADR 0017 Decision 3), so the walk hands it over as a single leaf and every consumer
/// treats it like any other producer. There is no `NodeContent` for it: it is a derived
/// runtime body, never document data.
pub(crate) enum LeafBody<'walk> {
    Content(&'walk NodeContent),
    Composed {
        producer: crate::voxel::CompositeProducer,
        /// The composed subtree's cache key. A scope has no `NodeContent` to fingerprint, so
        /// this is built from its members' fingerprints at compose time.
        fingerprint: String,
    },
}

/// A sealed scope pre-composed into one producer, with the world offset of its low corner.
pub(crate) struct ComposedScope {
    pub origin_voxels: [i64; 3],
    pub fingerprint: String,
    pub producer: crate::voxel::CompositeProducer,
}

impl LeafBody<'_> {
    /// The producer this leaf resolves through, plus its single-material override (`None`
    /// for a body carrying its own per-voxel materials).
    ///
    /// This is the ONE place content maps to a producer. It used to be an identical `match`
    /// repeated in `leaf_producers`, `resolve_region` and `resolve_chunk`, which is exactly
    /// the shape of duplication a new body kind would have had to be added to three times.
    pub(crate) fn into_producer(
        self,
        region_dimensions: [u32; 3],
        voxels_per_block: u32,
        outset_voxels: i64,
    ) -> Option<(Option<voxel_core::core_geom::BlockId>, Box<dyn VoxelProducer>)> {
        let _ = voxels_per_block;
        let (material, producer): (Option<voxel_core::core_geom::BlockId>, Box<dyn VoxelProducer>) =
            match self {
                LeafBody::Content(NodeContent::Tool { shape, material }) => {
                    (material_id_for(*material), Box::new(shape.clone()))
                }
                LeafBody::Content(NodeContent::SketchTool { producer, material }) => {
                    (material_id_for(*material), Box::new(producer.clone()))
                }
                LeafBody::Content(NodeContent::VoxelBody(VoxelBody::DebugClouds { seed })) => (
                    // A VoxelBody brings its own per-voxel materials; today the cloud field
                    // emits material 0, so the stamp keeps that.
                    None,
                    Box::new(DebugCloudField { dimensions: region_dimensions, seed: *seed }),
                ),
                LeafBody::Content(NodeContent::Group(_) | NodeContent::Instance(_)) => return None,
                // A composed scope's materials vary across its body, so it stamps per-voxel
                // rather than through a single override.
                LeafBody::Composed { producer, .. } => (None, Box::new(producer)),
            };
        // ADR 0019 Decision 7: the outset dilates the body BEFORE it folds.
        Some((material, crate::voxel::OutsetProducer::wrap(producer, outset_voxels)))
    }

    /// The leaf's emitted grid extent in voxels, grown by its outset — `None` for a body with
    /// no localisable extent.
    pub(crate) fn grid_voxels(&self, voxels_per_block: u32, outset_voxels: i64) -> Option<[i64; 3]> {
        let dimensions = match self {
            LeafBody::Content(content) => {
                return leaf_producer_grid_voxels(content, voxels_per_block, outset_voxels)
            }
            LeafBody::Composed { producer, .. } => producer.full_dimensions(voxels_per_block),
        };
        Some(std::array::from_fn(|axis| {
            (dimensions[axis] as i64 + 2 * outset_voxels).max(0)
        }))
    }
}

/// A content fingerprint for a leaf: the bytes (placement + content) that affect the
/// voxels it resolves to. Two leaves with the same fingerprint at the same world
/// position resolve to the same voxels, so the edit diff
/// ([`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since))
/// treats them as unchanged. `world_offset` is included so a moved Tool whose box
/// happens to coincide with another's still reads as distinct.
pub(crate) fn leaf_content_fingerprint(
    world_offset_voxels: [i64; 3],
    body: &LeafBody<'_>,
    grid_on_faces: bool,
    operation: CombineOp,
    scope_path: &[ScopeFrame],
) -> String {
    // The on-face-grid flag is baked into the resolved voxels as `GRID_OVERLAY_BIT`
    // (issue #29 S4), so two otherwise-identical leaves that differ only in this flag
    // resolve to DIFFERENT voxels. It must therefore be part of the fingerprint, or a
    // lone toggle of `voxel_grid_on_faces` produces an identical fingerprint and the
    // chunk-cache diff (`edit_aabb_since`) sees nothing dirty — leaving the stale
    // grid-less chunks in place until an unrelated edit evicts them. The embedded
    // offset is voxels (the canonical placement unit, ADR 0003 §3f(0)); it is an
    // opaque cache key, all leaves on the same unit for consistency.
    //
    // The leaf's `CombineOp` (ADR 0017) is fingerprinted for the same reason: a
    // Union↔Subtract flip changes the composite's voxels WITHIN the leaf's own
    // AABB (a cutter only ever removes cells its body covers), so the flip must
    // dirty exactly that AABB — and the store's dirtied chunks are RE-CLASSIFIED
    // (a Subtract can turn coarse-solid blocks into boundary or air), not merely
    // re-meshed.
    //
    // The leaf's SCOPE PATH (ADR 0017 Decision 3, issue #74) is fingerprinted too:
    // a Group's operation flip, or a restructure that moves the leaf into a
    // different scope, changes how the leaf composes — and the change is confined
    // to the leaves inside the scope (a scope-op flip re-folds exactly the scope's
    // composed body, whose cells all lie within its leaves' AABBs), so dirtying
    // every enclosed leaf's AABB dirties precisely the scope's subtree AABB. The
    // frame's stable `NodeId` (not a walk-order counter) keeps the fingerprint
    // stable across unrelated edits. The same mechanism absorbs a definition's
    // FIXTURE flip (ADR 0017 Decision 4, issue #77): sealed↔spliced changes every
    // expanded leaf's carried path (the instance frame appears/disappears), so
    // every placement's leaves re-fingerprint and their AABBs — which bound every
    // cell the splice can differ in — are dirtied.
    //
    // NOTE the `Intersect` asymmetry (ADR 0017 / issue #75): the two locality claims
    // above hold for Union/Subtract only. An Intersect mask kills accumulated cells
    // ANYWHERE OUTSIDE its own body, so an edit involving an Intersect-influence leaf
    // (see [`operation_masks_beyond_bounds`]) is NOT confined to the changed leaves'
    // AABBs — the spatial index records such leaves under a distinct fingerprint kind
    // (`LeafFingerprint::MasksBeyondItsBox`, chosen in `build_leaf_spatial_index`) so
    // the edit diff degrades to a wholesale clear instead of trusting the box union.
    let grid = if grid_on_faces { ":grid=1" } else { ":grid=0" };
    let op_token = |operation: CombineOp| match operation {
        CombineOp::Union => "union".to_string(),
        CombineOp::Subtract => "subtract".to_string(),
        CombineOp::Intersect => "intersect".to_string(),
        // The AMOUNT is part of the key: changing how far a surface is embossed changes the
        // resolved body, so it must dirty the leaf like any other geometry edit.
        CombineOp::Emboss { amount } => format!("emboss({amount:?})"),
    };
    let op = format!(":op={}", op_token(operation));
    let scopes = {
        let mut token = String::from(":scopes=[");
        for (depth, frame) in scope_path.iter().enumerate() {
            if depth > 0 {
                token.push(',');
            }
            token.push_str(&format!("{}:{}", frame.scope_node.0, op_token(frame.operation)));
        }
        token.push(']');
        token
    };
    let content = match body {
        LeafBody::Content(content) => *content,
        // A composed scope has no `NodeContent` to hash. Its key is built at compose time
        // from its members' ids and their own keys, so any edit inside the scope — a member
        // moving, changing shape, or changing its own outset — changes it, exactly as an
        // edit to an ordinary leaf changes that leaf's.
        LeafBody::Composed { fingerprint, .. } => {
            return format!("Composed@{world_offset_voxels:?}{fingerprint}{grid}{op}{scopes}")
        }
    };
    match content {
        NodeContent::Tool { shape, material } => {
            format!("Tool@{world_offset_voxels:?}:{shape:?}:{material:?}{grid}{op}{scopes}")
        }
        NodeContent::SketchTool { producer, material } => {
            format!("SketchTool@{world_offset_voxels:?}:{producer:?}:{material:?}{grid}{op}{scopes}")
        }
        NodeContent::VoxelBody(voxel_body) => format!("VoxelBody@{world_offset_voxels:?}:{voxel_body:?}{grid}{op}{scopes}"),
        // for_each_leaf only ever yields leaf content (Tool / SketchTool / VoxelBody);
        // Group / Instance are interior and never reach a visitor. Fingerprint
        // defensively anyway.
        NodeContent::Group(_) => format!("Group@{world_offset_voxels:?}{grid}{op}{scopes}"),
        NodeContent::Instance(def_id) => {
            format!("Instance@{world_offset_voxels:?}:{def_id:?}{grid}{op}{scopes}")
        }
    }
}

/// The producer's exact **emitted grid** in voxels per axis (the producer-true
/// frame the chunk ownership lives in), or `None` for a sizeless / interior leaf.
///
/// This is `size_blocks · d` for an [`SdfShape`] `Tool` (a whole-block grid), but
/// the EXACT prism AABB for a [`SketchTool`] — which may NOT be a whole multiple
/// of `d` (a sub-block profile). The chunk-coverage / spatial-index / AABB-skip
/// math must use this true span, not the block-rounded `leaf_size_blocks`, so a
/// sub-block sketch's voxels are never dropped by a too-small cover.
///
/// [`SketchTool`]: NodeContent::SketchTool
/// One enabled leaf of the op-stack as a resolvable producer (ADR 0010 E2). The
/// two-layer classifier + boundary-resolve evaluate this list (in document order, Union
/// on overlap) exactly as [`Scene::resolve_chunk_rebased`] stamps it. Yielded by
/// [`Scene::leaf_producers`].
pub struct LeafProducer {
    /// The leaf's accumulated WORLD voxel offset (its corner-anchored low corner in the
    /// scene's absolute voxel frame). A local cell `idx` has absolute index
    /// `world_offset_voxels + rotation·idx` (ADR 0008 — the frame is carried; the turn too,
    /// see [`rotation`](Self::rotation)).
    pub world_offset_voxels: [i64; 3],
    /// The leaf's **continuous rotation** (ADR 0027), the `Quat` the classifier reads to map a
    /// world voxel back into the producer's unturned local frame — a lattice turn is just a
    /// rotation that lands on the exact classifier path (§4). Populated as
    /// `node.transform.rotation()`: the whole tilt seated against the surface the node was
    /// dropped on (identity for an upright / world-plane drop).
    pub rotation: glam::Quat,
    /// The leaf's **continuous local offset** in voxels (ADR 0027), the accumulated float
    /// slide relative to the integer [`world_offset_voxels`](Self::world_offset_voxels)
    /// wandering origin — the field's continuous world position is
    /// `world_offset_voxels + offset_local_voxels` per axis. Zero for every voxel-snapped
    /// placement (the default), so a snapped scene's value is `[0.0, 0.0, 0.0]` and resolves
    /// exactly as the integer offset does.
    pub offset_local_voxels: [f32; 3],
    /// The boxed producer that resolves / bounds this leaf in its own `[0, full_dim)`
    /// local voxel-index frame.
    pub producer: Box<dyn VoxelProducer>,
    /// The single-material override id a Tool stamps onto every voxel (`Some`), or `None`
    /// for a VoxelBody that brings its own per-voxel materials (the cloud field emits id 0).
    pub material: Option<voxel_core::core_geom::BlockId>,
    /// The owning node's `grids.voxel_grid_on_faces` flag (issue #29 S4 / ADR 0003 §3c) —
    /// the transient on-face-grid render marker. Carried so the two-layer mesher (ADR 0010
    /// E3) can attach the per-box overlay flag exactly as the dense resolve bakes
    /// [`voxel_core::voxel::Voxel::grid_overlay`]. It is a RENDER hint only: it never enters the
    /// categorical `block_id`, the chunk codec, or `.vox` export (§3c).
    pub grid_overlay: bool,
    /// The leaf's [`CombineOp`] role in the ordered fold (ADR 0017): `Union` stamps
    /// (later-wins material on overlap); `Subtract` is an occupancy-only mask that
    /// removes cells accumulated before it **within its scope** and never stamps
    /// material. This is the owning NODE's operation; the scope structure it folds
    /// inside is `scope_path`.
    pub operation: CombineOp,
    /// The chain of enclosing sealed composition scopes (ADR 0017 Decision 3, issue
    /// #74), outermost first — every `Group` and every `Instance`-expanded SEALED
    /// definition body above this leaf, each frame carrying the SCOPE node's own
    /// [`CombineOp`]. A FIXTURE definition's expansion adds no frame (Decision 4,
    /// issue #77): its leaves carry the HOSTING scope's path unchanged, which is
    /// exactly what makes them splice into the host's fold.
    /// The flat list stays plain document order; a consumer reconstructs the
    /// depth-first fold's scope-open / scope-close markers by comparing adjacent
    /// leaves' paths (see [`ScopeFrame`]). Empty for a root-level leaf, which folds
    /// directly into the scene's root accumulator — the pre-#74 behaviour.
    pub scope_path: Vec<ScopeFrame>,
}

impl LeafProducer {
    /// Whether this leaf can remove occupancy at cells its own body does NOT cover —
    /// see [`operation_masks_beyond_bounds`]. Consumers that filter leaf subsequences
    /// by AABB overlap (per-block classify, per-chunk broadphase, the chunk-resolve
    /// skip) MUST keep such a leaf regardless of overlap, or a mask would silently
    /// stop applying outside its box (erring toward SOLID — never conservative).
    pub fn masks_beyond_bounds(&self) -> bool {
        operation_masks_beyond_bounds(self.operation, &self.scope_path)
    }
}

/// The continuous `glam::Quat` equivalent of a discrete [`LatticeOrientation`] (ADR 0027
/// §4). A lattice turn is one of the 24 axis-aligned rotations — a proper rotation (det `+1`,
/// group *O*) — so it maps exactly onto a clean unit quaternion. The bridge a caller holding a
/// face-normal turn uses to obtain the equivalent `Quat` the ghost / classifier speak (e.g.
/// `shot --ghost-face`).
///
/// The rotation matrix is built from the turn's action on the three basis axes: column `axis`
/// is where the turn sends the unit vector `e_axis`, so `matrix * v == orientation.apply(v)`
/// for every `v`. [`glam::Quat::from_mat3`] then reads the quaternion off that proper-rotation
/// matrix.
///
/// [`LatticeOrientation`]: substrate::spatial::LatticeOrientation
///
/// `pub` — the discrete→continuous bridge (ADR 0027) any caller holding a
/// [`LatticeOrientation`] uses to obtain the equivalent [`glam::Quat`] the ghost / classifier
/// speak (e.g. `shot --ghost-face`).
pub fn quat_from_lattice(orientation: substrate::spatial::LatticeOrientation) -> glam::Quat {
    let matrix = glam::Mat3::from_cols(
        orientation.apply_f32([1.0, 0.0, 0.0]).into(),
        orientation.apply_f32([0.0, 1.0, 0.0]).into(),
        orientation.apply_f32([0.0, 0.0, 1.0]).into(),
    );
    glam::Quat::from_mat3(&matrix)
}

/// Whether a leaf carrying `operation` under `scope_path` can remove occupancy at cells
/// its own body does NOT cover (ADR 0017 / issue #75). True exactly when `Intersect` is
/// involved anywhere on the leaf's fold path:
///
/// * the leaf's OWN operation is `Intersect` — its mask kills every accumulated cell
///   outside its body, at any distance from its AABB; or
/// * ANY enclosing scope folds under `Intersect` — the scope's composed body (which the
///   leaf contributes to) masks the parent accumulator everywhere outside it, so the
///   scope must open (and close under `Intersect`) even where none of its leaves emit.
///
/// `Union` and `Subtract` influence, by contrast, is confined to the contributing
/// leaves' own AABBs (a union adds cells only within its body; a subtract removes only
/// cells its body covers), which is what licenses every AABB-overlap filter for them.
pub fn operation_masks_beyond_bounds(operation: CombineOp, scope_path: &[ScopeFrame]) -> bool {
    operation == CombineOp::Intersect
        || scope_path
            .iter()
            .any(|frame| frame.operation == CombineOp::Intersect)
}

/// The leaf's emitted grid extent in voxels, GROWN by its outset (ADR 0019 Decision 7).
///
/// The outset belongs here rather than at the call sites because this one function feeds
/// both the region sizing and — through [`Scene::build_leaf_spatial_index`] — the
/// edit-broadphase AABB. ADR 0020's Consequences require the dirty region to be the OUTSET
/// bounds, not the producer bounds: an outset cutter dirties more than its own extent, and
/// invalidating only the undilated box would leave a stale rim behind after an edit.
///
/// [`Scene::build_leaf_spatial_index`]: crate::scene::Scene::build_leaf_spatial_index
pub(crate) fn leaf_producer_grid_voxels(
    content: &NodeContent,
    _voxels_per_block: u32,
    outset_voxels: i64,
) -> Option<[i64; 3]> {
    let grown = |dimensions: [i64; 3]| {
        // Grown by `N` on BOTH sides of every axis; an inset deeper than the half-extent
        // erodes the body away, so the floor is zero rather than a negative extent.
        Some(std::array::from_fn(|axis| {
            (dimensions[axis] + 2 * outset_voxels).max(0)
        }))
    };
    match content {
        // The Tool's exact emitted grid is its canonical voxel size directly (ADR
        // 0003 §3f(0); `size_voxels` already IS `blocks · d` for a whole-block size).
        NodeContent::Tool { shape, .. } => grown([
            shape.size_voxels[0] as i64,
            shape.size_voxels[1] as i64,
            shape.size_voxels[2] as i64,
        ]),
        NodeContent::SketchTool { producer, .. } => {
            let [grid_x, grid_y, grid_z] = producer.grid_dimensions();
            grown([grid_x as i64, grid_y as i64, grid_z as i64])
        }
        NodeContent::VoxelBody(_) | NodeContent::Group(_) | NodeContent::Instance(_) => None,
    }
}

/// The node's outset resolved to whole voxels at `voxels_per_block`, or `0` if it cannot be
/// (a fractional-voxel block term, or a zero density).
///
/// Falling back to zero is the safe direction: an unresolvable outset leaves the body
/// undilated rather than dilating it by a wrong amount.
pub(crate) fn outset_voxels_at(
    outset: voxel_core::units::Measurement,
    voxels_per_block: u32,
) -> i64 {
    outset.to_voxels(voxels_per_block).unwrap_or(0)
}

/// Map a Tool's [`MaterialChoice`] to the categorical [`BlockId`](voxel_core::core_geom::BlockId)
/// it stamps (ADR 0001 step 3 "Materials"; ADR 0003 §3a). A Tool is single-material by
/// nature: every voxel it emits takes this one block id, so distinct nodes render in
/// distinct materials. Stone = 0, Wood = 1, Plain = 2 (see [`MaterialChoice::block_id`]).
fn material_id_for(material: MaterialChoice) -> Option<voxel_core::core_geom::BlockId> {
    Some(material.block_id())
}
