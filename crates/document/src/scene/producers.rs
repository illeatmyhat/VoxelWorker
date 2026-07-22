//! Leaf producers and resolution: the [`VoxelBody`] / [`NodeContent`] leaf kinds, the
//! tree walk that composes placed leaves, the monolithic and chunk-scoped resolve
//! paths (region resolve is a test/oracle-gated oracle), and the per-leaf stamp
//! helpers that write a producer's voxels into an output grid or chunk.

use serde::{Deserialize, Serialize};

use voxel_core::core_geom::MaterialChoice;
use crate::debug_clouds::DebugCloudField;
use crate::sketch::SketchSolid;
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{VoxelGrid};
use crate::voxel::{SdfShape, VoxelProducer};

use super::*;

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
pub(super) type LeafVisitor<'walk> = dyn FnMut(
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
pub(super) enum LeafBody<'walk> {
    Content(&'walk NodeContent),
    Composed {
        producer: crate::voxel::CompositeProducer,
        /// The composed subtree's cache key. A scope has no `NodeContent` to fingerprint, so
        /// this is built from its members' fingerprints at compose time.
        fingerprint: String,
    },
}

/// A sealed scope pre-composed into one producer, with the world offset of its low corner.
pub(super) struct ComposedScope {
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
    pub(super) fn into_producer(
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
    pub(super) fn grid_voxels(&self, voxels_per_block: u32, outset_voxels: i64) -> Option<[i64; 3]> {
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

impl Scene {
    /// Walk the whole node tree depth-first, invoking
    /// `visitor(world_offset_voxels, leaf)` once for every **enabled leaf** (`Tool`
    /// / `VoxelBody`) with its accumulated **world** VOXEL offset (`parent_offset +
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
    /// The visitor receives, per enabled leaf: its accumulated world VOXEL
    /// offset, its content, its own `grids.voxel_grid_on_faces` flag (issue
    /// #29 S4 — the resolver ORs [`crate::voxel::GRID_OVERLAY_BIT`] into the
    /// leaf's stamped `material_id` when this is set, so the on-face voxel grid
    /// travels with each voxel through chunk bucketing), its own
    /// [`CombineOp`] (ADR 0017: the leaf's role in the ordered document-order
    /// fold — a `Subtract` leaf removes occupancy from everything accumulated
    /// before it *in its scope*), and its **scope path** — the chain of
    /// enclosing sealed composition scopes (ADR 0017 Decision 3, issue #74):
    /// every `Group` node and every `Instance`-expanded SEALED definition body
    /// on the way down, outermost first, each carrying the SCOPE node's own
    /// operation (see [`ScopeFrame`]). A scope pre-composes its leaves into one
    /// body; a boolean inside it can never affect geometry outside it. A
    /// FIXTURE definition's expansion contributes NO frame (ADR 0017 Decision
    /// 4, issue #77): its leaves splice into the hosting scope's fold carrying
    /// the HOST's path, so they compose directly at the instance's position.
    pub(super) fn for_each_leaf(&self, visitor: &mut LeafVisitor<'_>) {
        let mut def_path: Vec<DefId> = Vec::new();
        let mut scope_path: Vec<ScopeFrame> = Vec::new();
        // The ROOT PART is a scope too (ADR 0018 Decision 2 makes it a field rather than an
        // arena entry, but it is still the container the top-level nodes compose in). So it
        // pre-composes on the same two triggers a Group does: its own outset dilates the
        // whole scene, and a top-level Emboss needs the accumulated field its siblings form.
        //
        // Without this a top-level emboss reaches the voxel-set fold, which has no `A − N`
        // to read, and no-ops — and most scenes put their nodes at the top level.
        if let Some(composed) =
            self.composed_scope_leaf(&self.root, &self.roots, [0, 0, 0], &mut def_path)
        {
            visitor(
                composed.origin_voxels,
                // ADR 0027: the root composite sits at its integer origin with no continuous
                // slide of its own (its members' float offsets are baked into the composed
                // producer, not carried here).
                [0.0, 0.0, 0.0],
                // ADR 0027: a composed scope carries no continuous rotation of its own — its
                // members bake into the composite producer, so the composite is upright.
                glam::Quat::IDENTITY,
                LeafBody::Composed {
                    producer: composed.producer,
                    fingerprint: composed.fingerprint,
                },
                self.root.grids.voxel_grid_on_faces,
                CombineOp::Union,
                self.root.outset,
                &scope_path,
            );
            return;
        }
        self.walk_nodes(&self.roots, [0, 0, 0], [0.0, 0.0, 0.0], &mut def_path, &mut scope_path, visitor);
    }


    /// Collect every enabled leaf as a [`LeafProducer`] (ADR 0010 E2): its world voxel
    /// offset, a boxed [`VoxelProducer`], and its single-material override id. This is the
    /// op-stack the two-layer classifier / boundary-resolve evaluate over — the SAME
    /// leaves [`resolve_chunk_rebased`](Self::resolve_chunk_rebased) stamps, in the SAME
    /// document (walk) order, so the two-layer round-trip composes identically (later-wins
    /// Union on overlap). A region-sized VoxelBody (the cloud field) is sized to the composite
    /// `placed_region_dimensions` exactly as the dense chunk resolve sizes it.
    ///
    /// `pub` — the evaluator seam ADR 0010 E2's `two_layer_store` (up in the app crate) reads;
    /// the dense store keeps using the private [`for_each_leaf`](Self::for_each_leaf).
    pub fn leaf_producers(&self, voxels_per_block: u32) -> Vec<LeafProducer> {
        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut leaves = Vec::new();
        self.for_each_leaf(&mut |world_offset_voxels, offset_local_voxels, rotation, body, grid_on_faces, operation, outset, scope_path| {
            // ADR 0019 Decision 7: the outset dilates the body BEFORE it folds. Wrapping the
            // producer (rather than teaching the fold a new arm) means the classifier's
            // `cell_field_interval` call below and the voxel-set fold both see one definition
            // of what outset means — see `OutsetProducer`.
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let Some((material, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };
            leaves.push(LeafProducer {
                // The dilated body grows on every side, so its low corner moves DOWN by the
                // outset — the wrapper's frame origin sits `N` below the inner producer's
                // (ADR 0008: the frame is carried, never re-derived).
                world_offset_voxels: std::array::from_fn(|axis| {
                    world_offset_voxels[axis] - outset_voxels
                }),
                // ADR 0027: the continuous rotation and the float local offset. The
                // `- outset_voxels` frame adjustment is an INTEGER-frame move only — the float
                // slide is relative to the (unadjusted) integer origin, so it is NOT re-based here.
                rotation,
                offset_local_voxels,
                producer,
                material,
                grid_overlay: grid_on_faces,
                operation,
                scope_path: scope_path.to_vec(),
            });
        });
        leaves
    }

    /// Pre-compose a sealed scope into ONE producer, when it carries an outset and can be
    /// composed. Returns the composed body and the world offset of its low corner.
    ///
    /// `None` means "walk it normally": either the scope carries no outset (the overwhelming
    /// common case — nothing to dilate, so nothing to change) or its subtree contains a body
    /// that cannot participate. A `VoxelBody` is the latter: the cloud field sizes itself
    /// from the region, which a scope has no notion of, and it is fieldless anyway — so a
    /// Part containing one could not be outset even if it were composed (ADR 0020 Decision
    /// 1). Declining leaves that Part's existing behaviour exactly as it was.
    /// The shared tail of [`composed_scope_leaf`](Self::composed_scope_leaf) and
    /// [`composed_subtree`](Self::composed_subtree): given the collected `members` and a
    /// caller-built `fingerprint`, derive the composite's frame origin as the low corner
    /// of its UNION members (the only ones that can bound the body, ADR 0020 Decision 3 —
    /// voxel-valued so density-independent), rebase every member onto it (ADR 0008), and
    /// wrap it in a [`ComposedScope`]. `None` when there are no members, or no union
    /// member (the scope composes to nothing it could dilate). One definition of how a
    /// composite's origin is derived; the two callers differ only in the fingerprint.
    fn finish_composite(
        mut members: Vec<crate::voxel::CompositeMember>,
        fingerprint: String,
    ) -> Option<ComposedScope> {
        if members.is_empty() {
            return None;
        }
        let mut origin = [i64::MAX; 3];
        for member in &members {
            if member.operation != CombineOp::Union {
                continue;
            }
            for (lowest, member_offset) in origin.iter_mut().zip(member.offset_voxels) {
                *lowest = (*lowest).min(member_offset);
            }
        }
        if origin[0] == i64::MAX {
            return None;
        }
        for member in &mut members {
            member.offset_voxels =
                std::array::from_fn(|axis| member.offset_voxels[axis] - origin[axis]);
        }
        Some(ComposedScope {
            origin_voxels: origin,
            fingerprint,
            producer: crate::voxel::CompositeProducer::new(members),
        })
    }

    fn composed_scope_leaf(
        &self,
        scope: &Node,
        children: &[NodeId],
        world_offset_voxels: [i64; 3],
        def_path: &mut Vec<DefId>,
    ) -> Option<ComposedScope> {
        // Two reasons to pre-compose: the scope is DILATED as a whole (ADR 0019 Decision 7),
        // or one of its members EMBOSSES and so needs the accumulated body as a field rather
        // than as a voxel set (ADR 0020 Decision 4).
        let embosses = children.iter().any(|child| {
            self.arena
                .get(child)
                .is_some_and(|node| node.operation.needs_accumulated_field())
        });
        if scope.outset == voxel_core::units::Measurement::default() && !embosses {
            return None;
        }
        let mut members = Vec::new();
        let mut fingerprints = Vec::new();
        self.collect_composite_members(
            children,
            world_offset_voxels,
            def_path,
            &mut members,
            &mut fingerprints,
        )?;
        Self::finish_composite(
            members,
            format!(":composed[{}]={}", scope.id.0, fingerprints.join("|")),
        )
    }

    /// Collect a scope's members in document order, mirroring [`walk_nodes`]'s expansion
    /// rules exactly — nested Groups and sealed definition bodies become nested composites,
    /// a FIXTURE definition splices its children inline (ADR 0017 Decision 4), and the cycle
    /// guard is the same. `None` aborts the whole composition (see
    /// [`composed_scope_leaf`](Self::composed_scope_leaf)).
    ///
    /// [`walk_nodes`]: Self::walk_nodes
    fn collect_composite_members(
        &self,
        spine: &[NodeId],
        parent_offset: [i64; 3],
        def_path: &mut Vec<DefId>,
        members: &mut Vec<crate::voxel::CompositeMember>,
        fingerprints: &mut Vec<String>,
    ) -> Option<()> {
        for &node_id in spine {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.enabled {
                continue;
            }
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| parent_offset[axis] + node.transform.offset_voxels[axis]);
            match &node.content {
                // A fieldless / region-sized body cannot compose (see the caller).
                NodeContent::VoxelBody(_) => return None,
                NodeContent::Tool { .. } | NodeContent::SketchTool { .. } => {
                    let (material, producer) = LeafBody::Content(&node.content).into_producer(
                        [0, 0, 0],
                        0,
                        // A member's OWN outset still applies inside the scope: it dilates
                        // that member before the scope's fold sees it, and the scope's outset
                        // then dilates the composed result.
                        node.outset.to_voxels(1).unwrap_or(0),
                    )?;
                    fingerprints.push(format!("{}", node.id.0));
                    members.push(crate::voxel::CompositeMember {
                        offset_voxels: world_offset_voxels,
                        operation: node.operation,
                        material,
                        producer,
                    });
                }
                NodeContent::Group(children) => {
                    let nested = self.composed_subtree(children, world_offset_voxels, def_path)?;
                    fingerprints.push(format!("{}({})", node.id.0, nested.fingerprint));
                    members.push(crate::voxel::CompositeMember {
                        offset_voxels: nested.origin_voxels,
                        operation: node.operation,
                        material: None,
                        producer: crate::voxel::OutsetProducer::wrap(
                            Box::new(nested.producer),
                            node.outset.to_voxels(1).unwrap_or(0),
                        ),
                    });
                }
                NodeContent::Instance(def_id) => {
                    if def_path.contains(def_id) {
                        continue;
                    }
                    let def = self.def_by_id(*def_id)?;
                    def_path.push(*def_id);
                    let outcome = if def.fixture {
                        // ADR 0017 Decision 4: a fixture does NOT pre-compose — its children
                        // splice into the hosting scope's fold under their own operations.
                        self.collect_composite_members(
                            &def.children,
                            world_offset_voxels,
                            def_path,
                            members,
                            fingerprints,
                        )
                    } else {
                        self.composed_subtree(&def.children, world_offset_voxels, def_path)
                            .map(|nested| {
                                fingerprints.push(format!("{}[{}]", node.id.0, nested.fingerprint));
                                members.push(crate::voxel::CompositeMember {
                                    offset_voxels: nested.origin_voxels,
                                    operation: node.operation,
                                    material: None,
                                    producer: crate::voxel::OutsetProducer::wrap(
                                        Box::new(nested.producer),
                                        node.outset.to_voxels(1).unwrap_or(0),
                                    ),
                                });
                            })
                    };
                    def_path.pop();
                    outcome?;
                }
            }
        }
        Some(())
    }

    /// Compose a sub-scope unconditionally (its outset is applied by the CALLER, which owns
    /// the member entry) — the recursive half of [`composed_scope_leaf`].
    ///
    /// [`composed_scope_leaf`]: Self::composed_scope_leaf
    fn composed_subtree(
        &self,
        children: &[NodeId],
        world_offset_voxels: [i64; 3],
        def_path: &mut Vec<DefId>,
    ) -> Option<ComposedScope> {
        let mut members = Vec::new();
        let mut fingerprints = Vec::new();
        self.collect_composite_members(
            children,
            world_offset_voxels,
            def_path,
            &mut members,
            &mut fingerprints,
        )?;
        Self::finish_composite(members, fingerprints.join("|"))
    }

    /// Recursive worker for [`for_each_leaf`](Self::for_each_leaf). `parent_offset`
    /// is the accumulated world VOXEL offset of the assembly that owns `nodes`;
    /// `parent_offset_local` is the accumulated **continuous** local float offset (ADR
    /// 0027) summed the same way from each ancestor's `offset_local_voxels`, carried
    /// alongside the integer offset and handed to the visitor (additive — resolve still
    /// reads the integer offset, so occupancy is unchanged);
    /// `def_path` is the stack of definition ids currently being expanded (for the
    /// cycle guard — an `Instance` that would re-enter a definition already on the
    /// path is skipped instead of recursing forever); `scope_path` is the stack of
    /// enclosing sealed-scope frames (ADR 0017 Decision 3 — pushed on entering a
    /// `Group` or an `Instance`'s definition body, popped on leaving) handed to the
    /// visitor per leaf.
    pub(super) fn walk_nodes(
        &self,
        spine: &[NodeId],
        parent_offset: [i64; 3],
        parent_offset_local: [f32; 3],
        def_path: &mut Vec<DefId>,
        scope_path: &mut Vec<ScopeFrame>,
        visitor: &mut LeafVisitor<'_>,
    ) {
        // GOLDEN-CRITICAL (ADR 0003 B5): iterate the id-spine for ORDER (document
        // order = later-wins on overlap), fetching each node's content from the
        // arena. NEVER iterate the arena to produce this walk — that visits in id
        // order and would reorder Union material on overlap, moving the goldens.
        for &node_id in spine {
            let Some(node) = self.arena.get(&node_id) else {
                continue;
            };
            if !node.enabled {
                continue;
            }
            let world_offset_voxels = [
                parent_offset[0] + node.transform.offset_voxels[0],
                parent_offset[1] + node.transform.offset_voxels[1],
                parent_offset[2] + node.transform.offset_voxels[2],
            ];
            // ADR 0027: accumulate the continuous local float offset exactly like the integer
            // offset above — additive, carried to the visitor, unread by resolve this slice.
            let world_offset_local: [f32; 3] = std::array::from_fn(|axis| {
                parent_offset_local[axis] + node.transform.offset_local_voxels[axis]
            });
            match &node.content {
                NodeContent::Tool { .. }
                | NodeContent::SketchTool { .. }
                | NodeContent::VoxelBody(_) => {
                    // ADR 0017: the leaf carries its OWN `operation` plus the chain
                    // of enclosing sealed-scope frames (issue #74) into the flat
                    // walk — consumers reconstruct the scoped fold from the paths.
                    visitor(
                        world_offset_voxels,
                        world_offset_local,
                        // ADR 0027: the leaf's continuous rotation, the whole tilt seated
                        // against the surface it was dropped on. The classifier reads this
                        // quaternion directly (a lattice turn is just a rotation on the exact
                        // classifier path).
                        node.transform.rotation(),
                        LeafBody::Content(&node.content),
                        node.grids.voxel_grid_on_faces,
                        node.operation,
                        node.outset,
                        scope_path,
                    );
                }
                NodeContent::Group(children) => {
                    // ADR 0019 Decision 7: an outset on a SCOPE dilates the scope's COMPOSED
                    // body, so the scope is evaluated as one producer and handed to the
                    // visitor as a single leaf rather than recursed into. Per-member
                    // dilation is a different operation and the ADR rejects it: it would
                    // make an internal Subtract cutter carve MORE, where dilating the
                    // composed Part grows the finished body and partly closes that cut.
                    if let Some(composed) =
                        self.composed_scope_leaf(node, children, world_offset_voxels, def_path)
                    {
                        visitor(
                            composed.origin_voxels,
                            // ADR 0027: a composed scope carries no continuous slide of its own
                            // (members' float offsets are baked into the composite producer).
                            [0.0, 0.0, 0.0],
                            // ADR 0027: identity continuous rotation — the composite is upright.
                            glam::Quat::IDENTITY,
                            LeafBody::Composed {
                                producer: composed.producer,
                                fingerprint: composed.fingerprint,
                            },
                            node.grids.voxel_grid_on_faces,
                            node.operation,
                            node.outset,
                            scope_path,
                        );
                        continue;
                    }
                    // ADR 0017 Decision 3 (issue #74): a Group is a SEALED
                    // composition scope — its frame (identity + the GROUP node's own
                    // operation) encloses every leaf below it, so the group's
                    // children pre-compose into one body that folds into the parent
                    // under the group's operation.
                    scope_path.push(ScopeFrame {
                        scope_node: node.id,
                        operation: node.operation,
                    });
                    self.walk_nodes(
                        children,
                        world_offset_voxels,
                        world_offset_local,
                        def_path,
                        scope_path,
                        visitor,
                    );
                    scope_path.pop();
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
                    if def.fixture {
                        // ADR 0017 Decision 4 (issue #77): a FIXTURE definition does
                        // NOT pre-compose — NO scope frame is pushed, so its children
                        // SPLICE into the HOSTING scope's fold at this instance's
                        // spine position, in order, under the instance's transform
                        // (`world_offset_voxels` composes exactly as for a sealed
                        // body — the carried-frame discipline of ADR 0008; the host
                        // is POSITIONAL, never a stored reference). The instance's
                        // own `operation` is INERT (never consulted): each spliced
                        // leaf folds under its OWN operation. The fixture pierces
                        // exactly this ONE level of pre-composition — every scope
                        // frame already on `scope_path` (the host scope's seal and
                        // above) stays absolute. Invalidation: flipping a def's
                        // fixture flag changes every expanded leaf's carried scope
                        // path (the instance frame appears/disappears), so their
                        // fingerprints change and the store re-classifies exactly
                        // those leaves' AABBs — which contain every cell a splice
                        // can differ in (Union adds / Subtract carves only within a
                        // leaf's own body; an Intersect-influence leaf is already a
                        // wholesale-clear fingerprint kind either way).
                        self.walk_nodes(
                            &def.children,
                            world_offset_voxels,
                            world_offset_local,
                            def_path,
                            scope_path,
                            visitor,
                        );
                    } else {
                        // ADR 0017 Decision 3 (issue #74): a definition body is a
                        // SEALED scope — it pre-composes (internal booleans are fully
                        // spent inside it), and the finished body folds into the
                        // parent under the INSTANCE node's operation. The frame's
                        // identity is the INSTANCE node (unique per placement), so
                        // two expansions of the same definition are distinct scopes.
                        scope_path.push(ScopeFrame {
                            scope_node: node.id,
                            operation: node.operation,
                        });
                        self.walk_nodes(
                            &def.children,
                            world_offset_voxels,
                            world_offset_local,
                            def_path,
                            scope_path,
                            visitor,
                        );
                        scope_path.pop();
                    }
                    def_path.pop();
                }
            }
        }
    }

    /// Resolve `region` into a fresh [`VoxelGrid`] by a union tree-walk: each
    /// enabled leaf producer is resolved into its own local grid and **stamped**
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
    ///
    /// **Oracle — compile-gated.** This is a dense, O(volume) whole-region resolver:
    /// the measuring stick the sparse runtime path is held against, never a runtime
    /// path itself. It is excluded from production builds behind the `oracle` feature
    /// (tests reach it via `cfg(test)`), so "memory follows the surface" is enforced by
    /// the compiler, not by review — see the proof chapter's "Oracles" section
    /// (`docs/architecture/05-proof.md`).
    #[cfg(any(test, feature = "oracle"))]
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
        // still sizes a VoxelBody-only scene (which has no composite voxel extent).
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
        // A VoxelBody-only scene (e.g. `DebugClouds`) has no composite extent, so this is
        // `[0,0,0]` and the field stays CORNER-anchored at `[0, region)` — the shipped
        // convention (see `part_only_cloud_at_odd_density_drops_no_voxels` /
        // `mixed_tool_and_cloud_resolve_in_one_frame`). ADR 0008: the recentre is CARRIED on
        // the grid (below), so every consumer decodes correctly without re-deriving the
        // frame as `floor(dim/2)` (the assumption that dropped the corner-anchored cloud fog).
        let recentre_voxels = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        output.recentre_voxels = recentre_voxels;

        // Walk the whole tree (groups + instances recurse, composing world
        // translation down — ADR 0001 step 4). Each visited leaf is stamped under
        // its WORLD voxel offset minus the composite recentre. The offset is
        // already voxels at the document density (ADR 0003 §3f(0)), so it enters
        // the sum as-is. All of this is in i64 (S4a) so a far-placed node composes
        // without overflow; the result is downcast to f32 inside the stamp (the
        // render frame stays f32 — S4b makes the far case byte-identical via origin
        // rebasing).
        // ADR 0017 Decision 3 (issue #74): the walk is evaluated as a SCOPED depth-first
        // fold — each open Group / definition-body scope composes its leaves into its own
        // scratch grid, and a closing scope folds that composed body into its parent under
        // the SCOPE's operation (`sync_grid_scope_stack`), so a boolean inside a scope can
        // never affect geometry outside it.
        let mut scope_stack: Vec<(ScopeFrame, VoxelGrid)> = Vec::new();
        // ADR 0026: the discrete lattice `orientation` is still not applied here (an oriented
        // leaf is checked through the two-layer classifier against a hand-derived expectation, not
        // against this oracle) — every parity-gate scene is lattice-identity. ADR 0027 "Step 2":
        // the CONTINUOUS `rotation` quaternion and the fractional `offset_local_voxels` ARE now
        // applied, by routing a genuinely out-of-phase FIELD leaf through the shared inverse-gather
        // ([`gather_placed_field_into_grid`], substrate's ONE placement affine) so the dense
        // reference agrees with the live path on rotated / sub-voxel seats. A whole-phase leaf
        // (integer offset, axis-aligned rotation) keeps the exact translate-and-stamp path below.
        self.for_each_leaf(&mut |world_offset_voxels, offset_local_voxels, rotation, body, grid_on_faces, operation, outset, scope_path| {
            sync_grid_scope_stack(&mut scope_stack, &mut output, scope_path, region_dimensions);
            let target: &mut VoxelGrid = match scope_stack.last_mut() {
                Some((_, scratch)) => scratch,
                None => &mut output,
            };
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            // Every producer corner-anchors its grid at its world voxel offset (the low
            // corner); the recentre (from the producer-true voxel frame) symmetrises the
            // composite about the origin for ALL size·d parities, so no per-leaf lattice
            // shift is needed — a leaf simply sits at its world voxel offset.
            //
            // An outset body grows on every side, so its low corner moves DOWN by the outset
            // (ADR 0008 — the frame is carried, never re-derived).
            let translation_voxels = [
                world_offset_voxels[0] - recentre_voxels[0] - outset_voxels,
                world_offset_voxels[1] - recentre_voxels[1] - outset_voxels,
                world_offset_voxels[2] - recentre_voxels[2] - outset_voxels,
            ];
            // ADR 0017: Subtract and Intersect leaves are occupancy-only masks — they
            // never stamp material, so they take a mask path instead of a stamp. A
            // Subtract CARVES its body out of everything stamped before it (document
            // order, within its scope); an Intersect (issue #75) keeps ONLY the cells
            // its body covers, killing accumulated cells anywhere OUTSIDE its body —
            // including an empty result when nothing accumulated yet (fold start).
            // ONE producer serves both the mask and the stamp paths, so the outset wrapper
            // applies at a single point (ADR 0019 Decision 7 — the outset dilates the body
            // before it folds, whatever the fold role).
            let Some((material, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };

            // ADR 0027 "Step 2": a genuinely out-of-phase FIELD leaf (a continuous rotation or a
            // fractional sub-voxel seat) cannot be emitted one-cell-per-abs-cell by the integer
            // translation below, so resample it by inverse gather through substrate's shared
            // placement affine — the SAME map (and the same per-cell field test) the two-layer
            // classifier folds through, so the dense oracle agrees with the live path. The output
            // grid index `oi` denotes absolute cell `oi + recentre_voxels`, and the leaf's low
            // corner in the absolute frame is `world_offset_voxels − outset` (matching the
            // two-layer leaf's `world_offset_voxels`).
            if leaf_is_out_of_phase(rotation, offset_local_voxels) && producer.as_field().is_some() {
                let leaf_abs_low: [i64; 3] =
                    std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
                let placement = dense_leaf_placement(
                    rotation,
                    offset_local_voxels,
                    leaf_abs_low,
                    producer.as_ref(),
                    voxels_per_block,
                );
                gather_placed_field_into_grid(
                    target,
                    &placement,
                    producer.as_ref(),
                    material,
                    grid_on_faces,
                    operation,
                    recentre_voxels,
                    None,
                    voxels_per_block,
                );
                return;
            }

            // ADR 0017: Subtract and Intersect leaves are occupancy-only masks — they
            // never stamp material, so they take a mask path instead of a stamp.
            match operation {
                CombineOp::Subtract => mask_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    false,
                ),
                CombineOp::Intersect => mask_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    true,
                ),
                // Unreachable in practice: a scope containing an Emboss node is pre-composed
                // into a CompositeProducer (`CombineOp::needs_accumulated_field`), which
                // evaluates the formulas on the accumulated FIELD — the only representation
                // the voxel-set fold and the interval fold can agree on. A voxel-set
                // accumulator has no `A − N` to read. Skipping rather than falling back to
                // Union keeps an unevaluable emboss VISIBLE as a missing feature instead of
                // silently resolving as the wrong operation.
                CombineOp::Emboss { .. } => {
                    eprintln!(
                        "scene: skipping an Emboss node whose scope could not be composed                          (an un-composable scope has no accumulated field to emboss)"
                    );
                }
                CombineOp::Union => stamp_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    material,
                    // Issue #29 S4: OR the on-face-grid flag bit onto every
                    // stamped voxel iff this node opted in, so the bit travels
                    // with each voxel (and survives chunk bucketing).
                    grid_on_faces,
                    producer.as_ref(),
                    voxels_per_block,
                ),
            }
        });
        // Close every scope still open after the last leaf (folding each composed
        // body down into `output` under its scope's operation).
        sync_grid_scope_stack(&mut scope_stack, &mut output, &[], region_dimensions);

        output
    }

    /// Resolve exactly **one chunk** of the scene into a fresh [`VoxelGrid`], in
    /// **absolute (non-recentred) composite voxel coordinates**.
    ///
    /// This is the chunk-addressable counterpart to `resolve_region` required by
    /// issue #27 (deep chunked resolve). `resolve_region` is now the test/oracle-only
    /// dense measuring stick (ADR 0010 boundary residency retired it from the live
    /// render path; it is compile-gated behind `cfg(test)`/`oracle`) — the two-layer
    /// store (`evaluation::two_layer_store`) is the sole runtime path, and it calls
    /// THIS resolver per chunk. `resolve_region` recentres the composite on the
    /// origin; this path does **not** recentre, so its voxel positions are the
    /// scene's true composite coordinates. The two frames differ by exactly the
    /// recentre offset `resolve_region` subtracts (see
    /// `recentre_voxels`).
    ///
    /// A chunk is a `CHUNK_BLOCKS³`-block cell (`CHUNK_BLOCKS = 4`,
    /// [`voxel_core::core_geom::CHUNK_BLOCKS`]); one chunk therefore spans
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
    /// `floating_origin_voxels = [0, 0, 0]` reproduces `resolve_chunk` exactly. The
    /// live render passes [`recentre_voxels_for_resolve`](Self::recentre_voxels_for_resolve)
    /// (the composite recentre, an integer-block-aligned point), so for a near scene
    /// the result is bit-identical to today's recentred `resolve_region` while a
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
        let chunk_extent_voxels = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

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
        // ADR 0017 Decision 3 (issue #74): the same scoped depth-first fold as
        // `resolve_region`, restricted to this chunk. Composition is cell-local (a
        // union appends a cell, a subtract removes a cell), so restricting every
        // stamp / carve / scope-close to the chunk's cells commutes with the fold —
        // the reassembled chunks equal the monolithic scoped resolve exactly. A leaf
        // whose AABB misses the chunk is skipped WITHOUT syncing the stack: it
        // contributes no cells here, and a scope none of whose leaves touch the
        // chunk simply never opens (an empty scope folds to nothing under Union /
        // Subtract). EXCEPTION (ADR 0017 #75): an Intersect-influence leaf is never
        // skipped — its mask applies precisely where its body has no cells, and an
        // Intersect-closing scope must open even here so its ∅-in-chunk body
        // annihilates the parent on close (see the skip guard below).
        let mut scope_stack: Vec<(ScopeFrame, VoxelGrid)> = Vec::new();
        // ADR 0026: the discrete lattice `orientation` is still not applied here (identity for
        // every gate scene). ADR 0027 "Step 2": the CONTINUOUS `rotation` and the fractional
        // `offset_local_voxels` ARE applied — a genuinely out-of-phase FIELD leaf is resampled by
        // the shared inverse-gather ([`gather_placed_field_into_grid`]) AND its chunk-skip AABB is
        // taken from the ROTATED world box (the placement affine), so a tilted body is neither
        // truncated by the upright skip nor stamped upright.
        self.for_each_leaf(&mut |world_offset_voxels, offset_local_voxels, rotation, body, grid_on_faces, operation, outset, scope_path| {
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            // An outset body grows on every side, so its low corner moves DOWN by the
            // outset — and the skip AABB below must use the DILATED span, or a cutter whose
            // dilation reaches into this chunk would be skipped and its mask silently lost.
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            // Issue #27 S3 optimisation: skip a leaf whose world-AABB doesn't touch
            // this chunk, so resolving one chunk costs ~the leaves that overlap it
            // (not the whole tree). This is BIT-IDENTICAL to stamping-then-clipping:
            // the leaf's AABB `[off·d − grid/2, off·d + grid/2)` is the exact span of
            // its voxel centres, and `stamp_producer_into_chunk` keeps only centres
            // inside `[chunk_min, chunk_max)`; if those two half-open boxes don't
            // intersect, the stamp would have clipped EVERY voxel anyway. A
            // region-spanning leaf (a VoxelBody, `leaf_size_blocks` → `None`) has no
            // localisable AABB, so it is never skipped (it may emit anywhere).
            //
            // ADR 0017 (#75): an Intersect-INFLUENCE leaf (its own operation is
            // Intersect, or any enclosing scope closes under Intersect) is NEVER
            // skipped either: its mask kills accumulated cells anywhere OUTSIDE its
            // body, so a chunk its AABB misses is exactly where the mask must still
            // apply (its body has no cells here ⇒ everything accumulated in this
            // chunk within its scope dies). Keeping it also guarantees every
            // Intersect-closing scope OPENS in this chunk's fold (its leaves all
            // carry the Intersect frame), so the ∅-body scope close annihilates the
            // parent here exactly as the monolithic fold does.
            if !operation_masks_beyond_bounds(operation, scope_path) {
                if let Some(grid_voxels) = body.grid_voxels(voxels_per_block, outset_voxels) {
                    // The leaf's true footprint in the absolute frame. For a whole-phase leaf
                    // (axis-aligned rotation, integer offset — every gate scene) the producer
                    // corner-anchors its grid, so this is `[off, off + grid)`, bit-identical to
                    // stamping-then-clipping. ADR 0027: a genuinely rotated / sub-voxel-seated
                    // leaf's footprint is the ROTATED box, so it is taken from the SAME placement
                    // affine the gather stamps through — otherwise the upright box would skip the
                    // chunks the tilted body occupies and TRUNCATE it (the tubes-render-upright bug).
                    let leaf_box = if leaf_is_out_of_phase(rotation, offset_local_voxels) {
                        let full = glam::Vec3::new(
                            grid_voxels[0] as f32,
                            grid_voxels[1] as f32,
                            grid_voxels[2] as f32,
                        );
                        let world_offset = glam::Vec3::new(
                            world_offset_voxels[0] as f32,
                            world_offset_voxels[1] as f32,
                            world_offset_voxels[2] as f32,
                        ) + glam::Vec3::from_array(offset_local_voxels);
                        let (min, max) = substrate::spatial::LeafPlacement::new(
                            rotation,
                            full,
                            substrate::spatial::TrueWorldVoxelPoint::from_voxels(world_offset),
                        )
                        .world_aabb();
                        VoxelAabb::new(min, max)
                    } else {
                        let leaf_min = world_offset_voxels;
                        let leaf_max: [i64; 3] =
                            std::array::from_fn(|axis| leaf_min[axis] + grid_voxels[axis]);
                        VoxelAabb::new(leaf_min, leaf_max)
                    };
                    if !leaf_box.intersects(&chunk_box) {
                        return;
                    }
                }
            }
            let translation_voxels = world_offset_voxels;
            // ADR 0019 Decision 7: dilate before folding, exactly as the dense path does.
            let Some((material_override, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };
            // The leaf overlaps the chunk: sync the scope stack to its path (closing /
            // opening scopes exactly where the depth-first fold does) and compose into
            // the innermost open scope's scratch grid — or `output` at root level.
            sync_grid_scope_stack(&mut scope_stack, &mut output, scope_path, chunk_dimensions);
            let target: &mut VoxelGrid = match scope_stack.last_mut() {
                Some((_, scratch)) => scratch,
                None => &mut output,
            };
            // ADR 0027 "Step 2": a genuinely out-of-phase FIELD leaf is resampled by the shared
            // inverse-gather through substrate's placement affine — the SAME map (and per-cell
            // field test) the two-layer classifier folds through, so the dense chunk oracle agrees
            // with the live path on rotated / sub-voxel seats. Here the output grid holds ABSOLUTE
            // positions (floating origin `[0,0,0]` for the bare `resolve_chunk`, the recentre for
            // the rebased render path), so `oi` denotes absolute cell `oi + floating_origin_voxels`,
            // and the chunk membership clip keeps only cells in `[chunk_min, chunk_max)`.
            if leaf_is_out_of_phase(rotation, offset_local_voxels) && producer.as_field().is_some() {
                let placement = dense_leaf_placement(
                    rotation,
                    offset_local_voxels,
                    world_offset_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                );
                gather_placed_field_into_grid(
                    target,
                    &placement,
                    producer.as_ref(),
                    material_override,
                    grid_on_faces,
                    operation,
                    floating_origin_voxels,
                    Some(chunk_box),
                    voxels_per_block,
                );
                return;
            }
            // ADR 0017: a Subtract leaf carves its body's cells OUT of the voxels
            // stamped so far in this chunk WITHIN ITS SCOPE (occupancy-only — no
            // material, no stamp). A leaf whose AABB missed the chunk was already
            // skipped above (it carves nothing here), so this sees only
            // genuinely-overlapping cutters.
            if operation == CombineOp::Subtract {
                mask_producer_in_chunk(
                    target,
                    region_dimensions,
                    translation_voxels,
                    floating_origin_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    chunk_min_voxels,
                    chunk_max_voxels,
                    false,
                );
                return;
            }
            // ADR 0017 (#75): an Intersect leaf keeps ONLY the cells its body covers
            // in this chunk within its scope (occupancy-only). It is never skipped by
            // the AABB guard, so a mask whose box misses the chunk resolves an EMPTY
            // window here and correctly kills everything accumulated so far — the
            // restriction to this chunk's cells still commutes with the fold, because
            // a cell survives iff the mask occupies THAT cell.
            if operation == CombineOp::Intersect {
                mask_producer_in_chunk(
                    target,
                    region_dimensions,
                    translation_voxels,
                    floating_origin_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    chunk_min_voxels,
                    chunk_max_voxels,
                    true,
                );
                return;
            }
            stamp_producer_into_chunk(
                target,
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
        // Close every scope still open after the last overlapping leaf.
        sync_grid_scope_stack(&mut scope_stack, &mut output, &[], chunk_dimensions);

        output
    }

    /// Resolve the scene's whole region by **decomposing it into chunks** and
    /// merging them back into one grid, in **absolute (non-recentred) coordinates**.
    ///
    /// This loops over every chunk coordinate covering the composite AABB, calls
    /// [`resolve_chunk`](Self::resolve_chunk) for each, and unions the results. It
    /// proves the chunk decomposition reconstructs the whole scene; it is **not**
    /// wired into rendering (the render path stays on `resolve_region`, which
    /// recentres — see issue #27 S0). The returned grid is sized to the full
    /// composite extent and its voxels keep their absolute composite positions;
    /// compared against `resolve_region`'s output it differs only by the
    /// recentre offset.
    ///
    /// **Oracle — compile-gated.** A dense whole-region resolver kept only to prove the
    /// chunk decomposition reconstructs the scene; it is excluded from production builds
    /// behind the `oracle` feature (tests reach it via `cfg(test)`) so a dense path is a
    /// compile error, not a review catch — see the proof chapter's "Oracles" section
    /// (`docs/architecture/05-proof.md`).
    #[cfg(any(test, feature = "oracle"))]
    pub fn resolve_region_via_chunks(&self, voxels_per_block: u32, lod: u32) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut output = VoxelGrid::new(region_dimensions);

        let Some(chunk_range) = self.covering_chunk_range(voxels_per_block) else {
            // No leaf has an intrinsic size (a VoxelBody-only scene with no Tools): no
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

}

/// A content fingerprint for a leaf: the bytes (placement + content) that affect the
/// voxels it resolves to. Two leaves with the same fingerprint at the same world
/// position resolve to the same voxels, so the edit diff
/// ([`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since))
/// treats them as unchanged. `world_offset` is included so a moved Tool whose box
/// happens to coincide with another's still reads as distinct.
pub(super) fn leaf_content_fingerprint(
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
pub(super) fn leaf_producer_grid_voxels(
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
pub(super) fn outset_voxels_at(
    outset: voxel_core::units::Measurement,
    voxels_per_block: u32,
) -> i64 {
    outset.to_voxels(voxels_per_block).unwrap_or(0)
}

/// Sync the dense resolvers' **scope stack** to `target_path` — the stack-evaluated
/// depth-first fold of ADR 0017 Decision 3 (issue #74), reconstructed from each leaf's
/// carried [`ScopeFrame`] path (scopes are contiguous in the depth-first walk, so
/// comparing the open stack against the next leaf's path recovers the exact
/// scope-close / scope-open marker sequence).
///
/// Frames deeper than the common prefix CLOSE (innermost first): the popped scratch
/// grid — the scope's fully composed body so far — folds into its parent (the next
/// stack entry, or `root`) under the SCOPE's own operation via
/// [`fold_closed_scope_into`]. Frames beyond the common prefix OPEN: a fresh scratch
/// grid is pushed, so the scope's leaves compose sealed until it closes. Called once
/// per visited leaf and once with the empty path after the walk (closing everything).
///
/// For a pure-`Union` scene this is provably the identity transformation on the output
/// occupied list: a union close APPENDS the scratch voxels at exactly the walk position
/// the scope closed (before any later sibling stamped), preserving both the element
/// order and the later-wins material resolution of the flat pre-#74 walk — which is why
/// the pure-Union goldens hold byte-identical.
fn sync_grid_scope_stack(
    stack: &mut Vec<(ScopeFrame, VoxelGrid)>,
    root: &mut VoxelGrid,
    target_path: &[ScopeFrame],
    accumulator_dimensions: [u32; 3],
) {
    // The longest prefix of open frames the target path keeps open.
    let mut common = 0;
    while common < stack.len()
        && common < target_path.len()
        && stack[common].0 == target_path[common]
    {
        common += 1;
    }
    // Close the scopes deeper than the common prefix, innermost first.
    while stack.len() > common {
        let (frame, closed) = stack.pop().expect("len checked by the loop condition");
        let parent: &mut VoxelGrid = match stack.last_mut() {
            Some((_, scratch)) => scratch,
            None => root,
        };
        fold_closed_scope_into(parent, frame.operation, closed);
    }
    // Open the target path's scopes beyond the common prefix, outermost first.
    for frame in &target_path[common..] {
        stack.push((*frame, VoxelGrid::new(accumulator_dimensions)));
    }
}

/// Fold one CLOSED scope's composed body into its parent accumulator under the scope's
/// own [`CombineOp`] (ADR 0017 Decision 3):
///
/// * `Union` — append the body's voxels. The parent's occupied list is later-wins on
///   overlap (last write persists downstream), and the body's voxels are appended at
///   the walk position the scope closed, so the union close reproduces the flat
///   depth-first later-wins order exactly.
/// * `Subtract` — an occupancy-only mask (ADR 0017 Decision 1): every parent voxel
///   whose integer index coincides with one of the body's occupied cells is REMOVED;
///   surviving voxels keep their material and overlay, and the body's materials never
///   enter the parent.
/// * `Intersect` — the complementary occupancy-only mask (issue #75): the parent KEEPS
///   ONLY the voxels whose index coincides with one of the body's occupied cells;
///   everything else dies, including cells far outside the body's AABB. A scope that
///   closed at the EMPTY body therefore annihilates its parent (`A ∩ ∅ = ∅`), matching
///   the substrate kernel's ∅ identity. Surviving voxels keep their material/overlay.
fn fold_closed_scope_into(parent: &mut VoxelGrid, operation: CombineOp, closed: VoxelGrid) {
    match operation {
        // A scope that folds under Emboss is pre-composed with its siblings into one
        // CompositeProducer, so a composed body never arrives here needing to read the
        // parent's field (`CombineOp::needs_accumulated_field`). Reaching this arm means the
        // scope declined to compose — see the matching arm in the leaf fold.
        CombineOp::Emboss { .. } => {
            eprintln!(
                "scene: skipping an Emboss scope close whose siblings could not be composed \
                 (there is no accumulated field to emboss)"
            );
        }
        CombineOp::Union => parent.occupied.extend(closed.occupied),
        CombineOp::Subtract => {
            let carved: std::collections::HashSet<[i32; 3]> = closed
                .occupied
                .iter()
                .map(|voxel| voxel.local_index)
                .collect();
            parent
                .occupied
                .retain(|voxel| !carved.contains(&voxel.local_index));
        }
        CombineOp::Intersect => {
            let kept: std::collections::HashSet<[i32; 3]> = closed
                .occupied
                .iter()
                .map(|voxel| voxel.local_index)
                .collect();
            parent
                .occupied
                .retain(|voxel| kept.contains(&voxel.local_index));
        }
    }
}

/// Map a Tool's [`MaterialChoice`] to the categorical [`BlockId`](voxel_core::core_geom::BlockId)
/// it stamps (ADR 0001 step 3 "Materials"; ADR 0003 §3a). A Tool is single-material by
/// nature: every voxel it emits takes this one block id, so distinct nodes render in
/// distinct materials. Stone = 0, Wood = 1, Plain = 2 (see [`MaterialChoice::block_id`]).
fn material_id_for(material: MaterialChoice) -> Option<voxel_core::core_geom::BlockId> {
    Some(material.block_id())
}

/// Build the ADR 0027 continuous placement [`substrate::spatial::LeafPlacement`] for a leaf the
/// dense oracle is stamping — the SAME corner-anchored world↔producer-local affine the
/// two-layer classifier folds through (its evaluation-layer `leaf_affine` constructs an
/// identical `LeafPlacement`). Sharing substrate's ONE map — rather than the dense path's old
/// translation-only copy — is what stops the reference oracle silently disagreeing with the
/// live path on where a rotated / sub-voxel-seated producer's cells land (the deferred "Step 2").
///
/// `leaf_abs_low_voxels` is the OUTSET producer's low corner in the scene's ABSOLUTE voxel frame
/// (the visitor's `world_offset_voxels` minus the outset), matching the two-layer leaf's
/// `world_offset_voxels`; `offset_local_voxels` is the ADR 0027 continuous sub-voxel slide added
/// on top. `producer` is the same boxed producer the stamp resolves, so `full_dimensions` matches.
fn dense_leaf_placement(
    rotation: glam::Quat,
    offset_local_voxels: [f32; 3],
    leaf_abs_low_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
) -> substrate::spatial::LeafPlacement {
    let full_dimensions = producer.full_dimensions(voxels_per_block);
    let full = glam::Vec3::new(
        full_dimensions[0] as f32,
        full_dimensions[1] as f32,
        full_dimensions[2] as f32,
    );
    let world_offset = glam::Vec3::new(
        leaf_abs_low_voxels[0] as f32,
        leaf_abs_low_voxels[1] as f32,
        leaf_abs_low_voxels[2] as f32,
    ) + glam::Vec3::from_array(offset_local_voxels);
    substrate::spatial::LeafPlacement::new(
        rotation,
        full,
        substrate::spatial::TrueWorldVoxelPoint::from_voxels(world_offset),
    )
}

/// Whether a leaf is OUT OF PHASE with the absolute voxel lattice (ADR 0027): a genuine
/// (non-axis-aligned) rotation, OR a fractional `offset_local_voxels` sub-voxel seat. An
/// out-of-phase FIELD leaf cannot be emitted one-cell-per-abs-cell by a translation, so the dense
/// oracle resamples it by inverse gather ([`gather_placed_field_into_grid`]) — mirroring the
/// two-layer classifier's `gather_rotated_leaf_into_region`. A whole-phase leaf (integer offset,
/// axis-aligned rotation — every gate scene) keeps the exact translate-and-stamp path, so the
/// existing goldens stay byte-identical.
fn leaf_is_out_of_phase(rotation: glam::Quat, offset_local_voxels: [f32; 3]) -> bool {
    let axis_aligned = substrate::spatial::is_axis_aligned(rotation);
    let integer_offset = offset_local_voxels.iter().all(|slide| slide.fract() == 0.0);
    !(axis_aligned && integer_offset)
}

/// The ADR 0027 **inverse-resample gather** for a genuinely out-of-phase (rotated or sub-voxel-
/// seated) FIELD leaf, writing into the dense oracle's output [`VoxelGrid`]. The single-leaf
/// occupancy definition BOTH dense paths ([`Scene::resolve_region`] and
/// [`Scene::resolve_chunk_rebased`]) share, and the exact `VoxelGrid` mirror of the two-layer
/// classifier's `gather_rotated_leaf_into_region` — both fold through substrate's ONE
/// [`substrate::spatial::LeafPlacement`], so the dense reference can no longer drop the rotation
/// the live path applies.
///
/// For every output cell in the placed box, its centre is inverse-mapped into the producer-local
/// frame and the field is sampled: inside-or-on-surface cells are covered. The leaf's `operation`
/// is then applied to `output` exactly as the forward stamp path does — `Union` stamps the covered
/// cells (later document-order write wins on overlap), `Subtract` clears every covered cell, and
/// `Intersect` keeps ONLY the covered cells (killing accumulated cells anywhere outside the body,
/// including the whole grid when the body covers nothing — `A ∩ ∅ = ∅`).
///
/// `output_origin_abs` is the absolute voxel the output grid's index `[0,0,0]` denotes (the
/// recentre for `resolve_region`; the floating origin for `resolve_chunk_rebased`), so output
/// index `oi` denotes absolute cell `oi + output_origin_abs`. `clip_abs`, when `Some`, keeps only
/// cells whose absolute index lies in the half-open box (the chunk membership clip — the voxel
/// centre `+0.5` cancels on integer chunk edges exactly as the forward chunk stamp derives).
#[allow(clippy::too_many_arguments)]
fn gather_placed_field_into_grid(
    output: &mut VoxelGrid,
    placement: &substrate::spatial::LeafPlacement,
    producer: &dyn VoxelProducer,
    material_override: Option<voxel_core::core_geom::BlockId>,
    grid_overlay: bool,
    operation: CombineOp,
    output_origin_abs: [i64; 3],
    clip_abs: Option<VoxelAabb>,
    voxels_per_block: u32,
) {
    use voxel_core::voxel::{BlockAttrs, Voxel, SURFACE_ISOLEVEL};

    let field = producer
        .as_field()
        .expect("the dense gather is only reached for field producers (ADR 0027)");
    let (world_min, world_max) = placement.world_aabb();

    // The output-index box the leaf can touch: its absolute world AABB rebased to the output
    // frame (`abs − output_origin_abs`), intersected with the optional absolute clip box. Both the
    // world box and the clip are half-open, so the per-axis min/max of their rebased edges is the
    // exact overlap. The result is NOT clamped to the grid dimensions: a recentred dense grid
    // stores `i32` indices whose origin sits at a negative position (see `Voxel::local_index`), so
    // the stamp path never bounds the index to `[0, dimensions)`, and neither may the gather.
    let mut lo = [0i64; 3];
    let mut hi = [0i64; 3];
    for axis in 0..3 {
        let mut min_index = world_min[axis] - output_origin_abs[axis];
        let mut max_index = world_max[axis] - output_origin_abs[axis];
        if let Some(clip) = clip_abs {
            min_index = min_index.max(clip.min[axis] - output_origin_abs[axis]);
            max_index = max_index.min(clip.max[axis] - output_origin_abs[axis]);
        }
        lo[axis] = min_index;
        hi[axis] = max_index.max(min_index);
    }

    // Sample the field at every candidate cell centre, collecting the covered output cells and
    // the material each takes (the leaf's single-material override, else the producer's per-voxel
    // material, else the default id — the same precedence the forward stamp uses).
    let mut covered: Vec<([i32; 3], voxel_core::core_geom::BlockId)> = Vec::new();
    for z in lo[2]..hi[2] {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                let output_index = [x, y, z];
                let abs_centre = glam::Vec3::new(
                    (output_index[0] + output_origin_abs[0]) as f32 + 0.5,
                    (output_index[1] + output_origin_abs[1]) as f32 + 0.5,
                    (output_index[2] + output_origin_abs[2]) as f32 + 0.5,
                );
                let local = placement
                    .local_of(substrate::spatial::TrueWorldVoxelPoint::from_voxels(abs_centre))
                    .voxels()
                    .to_array();
                if field.signed_distance(local, voxels_per_block) <= SURFACE_ISOLEVEL {
                    let block_id = material_override
                        .or_else(|| producer.material_at(local, voxels_per_block))
                        .unwrap_or(voxel_core::core_geom::BlockId::DEFAULT);
                    // The recentred dense grid stores i32 indices (ADR 0008): the rebased output
                    // index fits i32 for every representable scene, as the stamp path assumes.
                    covered.push((
                        [output_index[0] as i32, output_index[1] as i32, output_index[2] as i32],
                        block_id,
                    ));
                }
            }
        }
    }

    match operation {
        // Later document-order leaf wins on overlap: appending the covered voxels reproduces the
        // dense Union (the resolved occupancy set keeps the last writer at each cell).
        CombineOp::Union => {
            output.occupied.reserve(covered.len());
            for (output_index, block_id) in covered {
                output.occupied.push(Voxel {
                    local_index: output_index,
                    block_local_coord: std::array::from_fn(|axis| {
                        (output_index[axis] as i64 + output_origin_abs[axis])
                            .rem_euclid(voxels_per_block.max(1) as i64) as u8
                    }),
                    block_id,
                    attrs: BlockAttrs::DEFAULT,
                    grid_overlay,
                });
            }
        }
        // Occupancy-only masks (ADR 0017 Decision 1): the covered cells are removed (Subtract) or
        // are the ONLY survivors (Intersect); surviving voxels keep their own material/overlay.
        CombineOp::Subtract => {
            let carved: std::collections::HashSet<[i32; 3]> =
                covered.iter().map(|(index, _)| *index).collect();
            output.occupied.retain(|voxel| !carved.contains(&voxel.local_index));
        }
        CombineOp::Intersect => {
            let kept: std::collections::HashSet<[i32; 3]> =
                covered.iter().map(|(index, _)| *index).collect();
            output.occupied.retain(|voxel| kept.contains(&voxel.local_index));
        }
        // Unreachable: an Emboss scope is pre-composed into a CompositeProducer before it reaches
        // a visitor, and a composed root sits at identity rotation / integer offset (in phase), so
        // it never routes to this gather.
        CombineOp::Emboss { .. } => {}
    }
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
/// the producer emitted (a VoxelBody's own per-voxel materials).
///
/// Private helper of the dense [`Scene::resolve_region`] oracle only (the per-chunk
/// path uses [`stamp_producer_into_chunk`]), so it carries the same `oracle` compile
/// gate — see the proof chapter's "Oracles" section (`docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
fn stamp_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    material_override: Option<voxel_core::core_geom::BlockId>,
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

/// Resolve `producer` into its own local grid and **occupancy-mask** `output` with it
/// (ADR 0017 Decision 1 — `Subtract`/`Intersect` are occupancy-only, never stamping
/// material). Each output voxel whose index coincides with one of the producer's
/// occupied cells (translated by `translation_voxels`) is *covered*; whether covered
/// voxels are the ones KEPT or the ones REMOVED is the single varying bit:
///
/// * `keep_if_covered = false` → **Subtract** (carve): covered voxels are removed.
/// * `keep_if_covered = true`  → **Intersect** (issue #75): only covered voxels
///   survive, so every accumulated voxel outside the mask's body dies — however far
///   from its AABB.
///
/// Surviving voxels keep their material and overlay; the cutter/mask's own material
/// never enters the output. The mask sibling of [`stamp_producer`], and like it a
/// private helper of the dense [`Scene::resolve_region`] oracle only, so it carries
/// the same `oracle` compile gate (see the proof chapter's "Oracles" section,
/// `docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
fn mask_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    keep_if_covered: bool,
) {
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local, voxels_per_block);

    // The mask's occupied INTEGER indices in the output's frame (the same
    // i64-then-downcast translation the stamp applies, so a covered cell coincides
    // bit-exactly with the stamped cell it keeps or removes).
    let covered: std::collections::HashSet<[i32; 3]> = local
        .occupied
        .iter()
        .map(|voxel| {
            [
                (voxel.local_index[0] as i64 + translation_voxels[0]) as i32,
                (voxel.local_index[1] as i64 + translation_voxels[1]) as i32,
                (voxel.local_index[2] as i64 + translation_voxels[2]) as i32,
            ]
        })
        .collect();
    output
        .occupied
        .retain(|voxel| covered.contains(&voxel.local_index) == keep_if_covered);
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
    material_override: Option<voxel_core::core_geom::BlockId>,
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
    let window_local = voxel_core::spatial_index::VoxelAabb::new(
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

/// Resolve `producer`'s cells inside the chunk window and **occupancy-mask** `output`
/// with them (ADR 0017 Decision 1). Each already-stamped voxel whose (rebased) index
/// coincides with one of the mask's cells is *covered*; `keep_if_covered` picks which
/// side of the mask survives — the chunk-scoped sibling of [`mask_producer`]:
///
/// * `keep_if_covered = false` → **Subtract** (carve): covered voxels are removed.
/// * `keep_if_covered = true`  → **Intersect** (issue #75): only covered voxels
///   survive. Restricting the mask to the chunk window is EXACT (not merely
///   conservative): a cell survives iff the mask occupies that very cell, and every
///   output voxel here lies inside the chunk — a mask cell in another chunk can only
///   affect that other chunk. A mask whose box misses this chunk entirely resolves an
///   EMPTY window and thus clears everything accumulated so far, exactly
///   `accumulated ∩ ∅ = ∅` restricted here.
///
/// Like [`stamp_producer_into_chunk`], uses the same local resolve window
/// (`[chunk_min, chunk_max)` mapped into the producer's local frame — a mask cell
/// outside this chunk can only affect OTHER chunks) and the same
/// i64-before-f32-downcast rebase to `floating_origin_voxels`, so the covered index
/// coincides bit-exactly with the stamped index it keeps or removes.
#[allow(clippy::too_many_arguments)]
fn mask_producer_in_chunk(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    floating_origin_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    chunk_min_voxels: [i64; 3],
    chunk_max_voxels: [i64; 3],
    keep_if_covered: bool,
) {
    // Resolve ONLY the mask cells this chunk owns, in the producer's LOCAL
    // voxel-index frame — the identical window arithmetic as the stamp (see
    // `stamp_producer_into_chunk` for the half-open-edge derivation).
    let mut local = VoxelGrid::new(region_dimensions);
    let window_local = voxel_core::spatial_index::VoxelAabb::new(
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

    // Rebase the mask's indices exactly as the stamp rebases stamped ones (pure
    // i64 subtraction BEFORE the downcast), so mask and stamp agree bit-exactly.
    let rebased_translation = [
        translation_voxels[0] - floating_origin_voxels[0],
        translation_voxels[1] - floating_origin_voxels[1],
        translation_voxels[2] - floating_origin_voxels[2],
    ];
    let covered: std::collections::HashSet<[i32; 3]> = local
        .occupied
        .iter()
        .map(|voxel| {
            [
                (voxel.local_index[0] as i64 + rebased_translation[0]) as i32,
                (voxel.local_index[1] as i64 + rebased_translation[1]) as i32,
                (voxel.local_index[2] as i64 + rebased_translation[2]) as i32,
            ]
        })
        .collect();
    output
        .occupied
        .retain(|voxel| covered.contains(&voxel.local_index) == keep_if_covered);
}
