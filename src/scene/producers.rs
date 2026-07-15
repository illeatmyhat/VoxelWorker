//! Leaf producers and resolution: the [`Part`] / [`NodeContent`] leaf kinds, the
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
/// types but unimplemented in `Scene::resolve_region` (recursion + instancing
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

impl Scene {
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
    pub(super) fn for_each_leaf(&self, visitor: &mut dyn FnMut([i64; 3], &NodeContent, bool)) {
        let mut def_path: Vec<DefId> = Vec::new();
        self.walk_nodes(&self.roots, [0, 0, 0], &mut def_path, visitor);
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
        self.for_each_leaf(&mut |world_offset_voxels, content, grid_on_faces| {
            let (material, producer): (Option<voxel_core::core_geom::BlockId>, Box<dyn VoxelProducer>) =
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
                grid_overlay: grid_on_faces,
            });
        });
        leaves
    }

    /// Recursive worker for [`for_each_leaf`](Self::for_each_leaf). `parent_offset`
    /// is the accumulated world VOXEL offset of the assembly that owns `nodes`;
    /// `def_path` is the stack of definition ids currently being expanded (for the
    /// cycle guard — an `Instance` that would re-enter a definition already on the
    /// path is skipped instead of recursing forever).
    pub(super) fn walk_nodes(
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
    /// This is the chunk-addressable counterpart to `resolve_region` required by
    /// issue #27 (deep chunked resolve). It is **additive**: the live render path
    /// still goes through `resolve_region` (which recentres the composite on the
    /// origin); this path does **not** recentre, so its voxel positions are the
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
                Option<voxel_core::core_geom::BlockId>,
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

}

/// A content fingerprint for a leaf: the bytes (placement + content) that affect the
/// voxels it resolves to. Two leaves with the same fingerprint at the same world
/// position resolve to the same voxels, so the edit diff
/// ([`LeafSpatialIndex::edit_aabb_since`](voxel_core::spatial_index::LeafSpatialIndex::edit_aabb_since))
/// treats them as unchanged. `world_offset` is included so a moved Tool whose box
/// happens to coincide with another's still reads as distinct.
pub(super) fn leaf_content_fingerprint(
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
    pub material: Option<voxel_core::core_geom::BlockId>,
    /// The owning node's `grids.voxel_grid_on_faces` flag (issue #29 S4 / ADR 0003 §3c) —
    /// the transient on-face-grid render marker. Carried so the two-layer mesher (ADR 0010
    /// E3) can attach the per-box overlay flag exactly as the dense resolve bakes
    /// [`voxel_core::voxel::Voxel::grid_overlay`]. It is a RENDER hint only: it never enters the
    /// categorical `block_id`, the chunk codec, or `.vox` export (§3c).
    pub grid_overlay: bool,
}

pub(super) fn leaf_producer_grid_voxels(content: &NodeContent, _voxels_per_block: u32) -> Option<[i64; 3]> {
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

/// Map a Tool's [`MaterialChoice`] to the categorical [`BlockId`](voxel_core::core_geom::BlockId)
/// it stamps (ADR 0001 step 3 "Materials"; ADR 0003 §3a). A Tool is single-material by
/// nature: every voxel it emits takes this one block id, so distinct nodes render in
/// distinct materials. Stone = 0, Wood = 1, Plain = 2 (see [`MaterialChoice::block_id`]).
fn material_id_for(material: MaterialChoice) -> Option<voxel_core::core_geom::BlockId> {
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
