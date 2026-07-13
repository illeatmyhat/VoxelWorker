//! ADR 0010 E2 — the boundary-aware **two-layer chunk** representation + the block
//! classifier, an **OFF-by-default** capability proven bit-exact against the dense
//! [`crate::store::Store`] path (CONTEXT.md "Boundary residency").
//!
//! This is the heart of the ADR 0009 → ADR 0010 port: the **one evaluator** classifies
//! each BLOCK of a covering chunk **air / coarse-solid / boundary** via the E1 interval
//! bound ([`VoxelProducer::cell_field_interval`]), then materialises only the boundary
//! blocks per-voxel. A solid interior carries block ids with **NO voxel data** — the
//! whole point of the port (an 800×800-revolve-class solid stops densifying its interior).
//!
//! ## What this slice IS
//!
//! * [`BlockClassification`] / [`classify_chunk_block`] — the conservative classifier:
//!   compose every leaf's field interval over a block cell by CSG interval arithmetic
//!   (v1 only has [`crate::scene::CombineOp::Union`], so the composition is
//!   [`crate::voxel::union_field_intervals`]), then [`FieldInterval::classify`]. An
//!   unboundable producer (`cell_field_interval == None`) forces the block BOUNDARY.
//! * [`TwoLayerChunk`] — the per-chunk store: a coarse per-block [`BlockId`] grid
//!   (coarse-solid blocks carry their id, no voxels) + a SPARSE map of boundary blocks
//!   to their decomposed [`VoxelBox`] geometry + per-face [`SeamSolidity`] flags.
//! * [`TwoLayerStore::build_chunk`] — runs the evaluator for one chunk behind the
//!   capability flag; [`TwoLayerChunk::expand_occupancy_into`] streams it back to full
//!   occupancy (coarse fast-fill + boundary per-voxel) for the parity gate + as a
//!   transition shim.
//!
//! ## Status (ADR 0010 E5 LANDED — the two-layer path is the SOLE runtime display path)
//!
//! * **The mesher consumes the layers** (E3 / #50): `new_from_two_layer_chunks` (coarse
//!   one-box + microblock cuboids + seam-flag culling) is the live display mesh path.
//! * **Export + the diameter query stream cacheless from the evaluator** (E4 / #51):
//!   `stream_vox_occupancy` / `streamed_widest_run_in_band`, no dense fallback.
//! * **The live display cache is the [`TwoLayerResidentCache`]** (E5 / #54): chunk-granular
//!   incremental edits; the mesher + brick fog sink read its resident set directly (ADR 0011
//!   G5 retired the dense fog-grid stream — `expand_resident_chunks_into_grid` is now a
//!   `#[cfg(test)]` parity oracle).
//!   The dense [`Store::resolve_region`](crate::store::Store::resolve_region) is retired from
//!   every RUNTIME path and kept ONLY as the test parity + golden reference oracle.
//!
//! ## Frame (ADR 0008 — the voxel-frame invariant)
//!
//! The coarse grid is **chunk-local integer**: a coarse cell is addressed by its
//! chunk-local block index `[0, CHUNK_BLOCKS)`, and the absolute origin lives in the
//! chunk key ([`crate::store::ChunkCacheKey::chunk_coord`]). The boundary blocks'
//! [`VoxelBox`]es are in **chunk-local voxel** indices `[0, chunk_extent_voxels)`. The
//! expansion stamps voxels into the SAME (recentred / floating-origin-rebased) frame
//! [`Scene::resolve_chunk_rebased`](crate::scene::Scene::resolve_chunk_rebased) produces,
//! so the round-trip is occupancy-identical to the dense path.

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;

use crate::core_geom::{BlockAttrs, BlockId, CHUNK_BLOCKS};
use crate::cuboid::{decompose_into_boxes, VoxelBox, VoxelRegion};
use crate::scene::{LeafProducer, Scene};
use crate::spatial_index::{EditBroadphaseBvh, VoxelAabb};
use crate::voxel::{
    union_field_intervals, FieldClassification, RecentreVoxels, Voxel, VoxelGrid, SURFACE_ISOLEVEL,
};

/// The coarse verdict for a single BLOCK of a chunk (ADR 0010 Decision 2). Distinct from
/// [`FieldClassification`] (the per-producer interval verdict) because the BLOCK verdict
/// is the COMPOSED result over the whole op-stack plus the sculpt-touched override:
/// any block a sculpt delta touches is forced [`BlockClassification::Boundary`], and an
/// unboundable op (`cell_field_interval == None`) collapses the block to boundary too.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockClassification {
    /// Every voxel in the block is empty — the block holds nothing (no coarse id, no
    /// microblock entry).
    Air,
    /// Every voxel in the block is occupied by a single material — the block lives in
    /// the COARSE layer as one [`BlockId`], with NO per-voxel data (interior elision).
    CoarseSolid(BlockId),
    /// The block straddles the surface (or an op could not bound it, or a sculpt delta
    /// touched it): it lives in the MICROBLOCK layer, resolved per-voxel and decomposed
    /// to cuboids. Always the SAFE verdict.
    Boundary,
}

/// Per-face solidity flags for a boundary block — the coarse/microblock analogue of the
/// dense-fog **apron** (CONTEXT.md "Seam solidity"; VS `sideAlmostSolid`). Each face flag
/// is `true` iff that whole face of the block is solid (every voxel of the `density²`
/// face cells is occupied), so E3's mesher can cull a seam face against a fully-solid
/// neighbour face without expanding the neighbour's voxels.
///
/// Faces are indexed by `(axis, side)`: axis 0/1/2 = X/Y/Z (Z-up), side
/// 0 = the LOW face (`coord == 0`), side 1 = the HIGH face (`coord == density - 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SeamSolidity {
    /// `[axis][side]`: `solid[axis][0]` is the low face, `solid[axis][1]` the high face.
    pub solid: [[bool; 2]; 3],
}

impl SeamSolidity {
    /// Whether the face on `axis` (0/1/2 = X/Y/Z), `side` (0 = low, 1 = high), is solid.
    pub fn face_is_solid(&self, axis: usize, side: usize) -> bool {
        self.solid[axis][side]
    }
}

/// The geometry of one boundary block (CONTEXT.md "Microblock layer"): its sub-block
/// voxels already decomposed to cuboids, plus its per-face seam-solidity flags.
///
/// The [`VoxelBox`]es are in **block-local voxel** indices `[0, density)` per axis (the
/// frame [`decompose_into_boxes`] yields over a `density³` [`VoxelRegion`]); the owning
/// block's chunk-local position is the map key in [`TwoLayerChunk::microblocks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroblockGeometry {
    /// The block's solid voxels, greedy-decomposed to single-material cuboids
    /// (block-local voxel indices `[0, density)`). Empty when the block resolved to no
    /// voxels (a boundary verdict the per-voxel pass found empty — still exact).
    ///
    /// Each cuboid's `material_id` is the cuboid mesher's **render-cell key** (ADR 0003
    /// §3c): the clean categorical `block_id` in the low bits, the transient on-face-grid
    /// overlay marker in [`crate::cuboid_mesh::MESH_GRID_OVERLAY_BIT`]. The decomposition
    /// therefore splits a box across differing overlay states exactly like the dense
    /// mesher, and E3's mesher reads the box's clean id + overlay back out of this key
    /// without the render flag ever entering the categorical cell. Consumers that want the
    /// clean id (E2 occupancy expansion) mask the overlay bit off.
    pub cuboids: Vec<VoxelBox>,
    /// Per-face solidity flags (the seam apron analogue) for this block.
    pub seam_solidity: SeamSolidity,
}

/// The boundary-aware two-layer representation of ONE chunk (ADR 0010 Decision 1).
///
/// * `coarse` — a per-BLOCK [`Option<BlockId>`] grid over the chunk's `CHUNK_BLOCKS³`
///   blocks (chunk-local integer, ADR 0008). `Some(id)` is a coarse-solid block (id, no
///   voxels); `None` is air OR a boundary block (boundary geometry lives in `microblocks`).
/// * `microblocks` — a SPARSE map of boundary blocks (keyed by chunk-local block index)
///   to their [`MicroblockGeometry`]. Only surface blocks appear.
///
/// A coarse-solid block stores ZERO voxels (interior elision); only boundary blocks carry
/// per-voxel geometry, and even those are stored as cuboids, never a dense `density³` grid.
///
/// Derives [`PartialEq`]/[`Eq`] so the incremental-vs-full parity gate (ADR 0010 #54) can
/// assert an incrementally-rebuilt chunk is IDENTICAL — coarse layer + overlay + microblock
/// map + seam flags — to a full from-scratch rebuild of the same chunk.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TwoLayerChunk {
    /// The chunk's voxels-per-block density (chunk extent in voxels is
    /// `CHUNK_BLOCKS * density`). Carried so the expansion / interior count can size the
    /// per-block fill without re-deriving it.
    pub voxels_per_block: u32,
    /// Per-block coarse layer, row-major over the chunk's `CHUNK_BLOCKS³` blocks
    /// (X fastest, then Y, then Z): `Some(id)` = coarse-solid, `None` = air or boundary.
    pub coarse: Vec<Option<BlockId>>,
    /// Per-block on-face-grid overlay marker (ADR 0003 §3c), parallel to `coarse`: `true`
    /// iff the coarse-solid block's single owning leaf had `grid_overlay` set. Meaningful
    /// only where `coarse[i].is_some()`; the E3 mesher reads it to flag a coarse block's
    /// one-box draw. A RENDER hint only — never part of the categorical id / occupancy.
    pub coarse_overlay: Vec<bool>,
    /// Sparse boundary-block geometry, keyed by chunk-local block index `[bx, by, bz]`
    /// (each component `< CHUNK_BLOCKS`).
    pub microblocks: BTreeMap<[u32; 3], MicroblockGeometry>,
}

/// Chunk-local block index → flat row-major index over `CHUNK_BLOCKS³` (X fastest).
#[inline]
fn coarse_flat_index(block: [u32; 3]) -> usize {
    let n = CHUNK_BLOCKS as usize;
    (block[2] as usize * n + block[1] as usize) * n + block[0] as usize
}

impl TwoLayerChunk {
    /// An all-air chunk at `voxels_per_block` (no coarse ids, no microblocks).
    fn empty(voxels_per_block: u32) -> Self {
        let block_count = (CHUNK_BLOCKS as usize).pow(3);
        Self {
            voxels_per_block,
            coarse: vec![None; block_count],
            coarse_overlay: vec![false; block_count],
            microblocks: BTreeMap::new(),
        }
    }

    /// The coarse-layer id at chunk-local block index `block` (`Some` only for a
    /// coarse-solid block; `None` for air or boundary).
    pub fn coarse_block(&self, block: [u32; 3]) -> Option<BlockId> {
        self.coarse[coarse_flat_index(block)]
    }

    /// The on-face-grid overlay marker (ADR 0003 §3c) of the coarse-solid block at
    /// `block` — `true` iff that block's owning leaf had `grid_overlay` set. Only
    /// meaningful when [`coarse_block`](Self::coarse_block) is `Some`.
    pub fn coarse_block_overlay(&self, block: [u32; 3]) -> bool {
        self.coarse_overlay[coarse_flat_index(block)]
    }

    /// Whether this chunk holds ANY geometry (at least one coarse-solid block OR one
    /// boundary block) — i.e. it would produce a mesh / occupancy. An all-air chunk returns
    /// `false`. The two-layer analogue of the dense `!grid.occupied.is_empty()` the cuboid
    /// incremental plan keys "occupied" off (issue #55): only non-empty chunks are meshed,
    /// so a chunk that an edit turned all-air drops out of the rebuild set and is evicted.
    pub fn has_geometry(&self) -> bool {
        self.coarse.iter().any(Option::is_some) || !self.microblocks.is_empty()
    }

    /// The TOTAL voxel count this chunk STORES — the sum of every boundary block's
    /// decomposed cuboid voxels. A coarse-solid block contributes ZERO (interior
    /// elision); this is the measured "surface-only" residency the ADR demands (an
    /// 800×800 revolve's interior holds no voxels here, unlike the dense path).
    pub fn stored_voxel_count(&self) -> u64 {
        self.microblocks
            .values()
            .flat_map(|geometry| geometry.cuboids.iter())
            .map(VoxelBox::voxel_count)
            .sum()
    }

    /// Stream this chunk back to full occupancy into `output` (ADR 0010 Decision 3 /
    /// parity gate (a)): a coarse-solid block is a fast `density³` fill at its block id;
    /// a boundary block expands its cuboids per-voxel. Each voxel is stamped at its
    /// CHUNK-LOCAL voxel index (the chunk's own `[0, chunk_extent_voxels)` frame) +
    /// `index_offset`, so the caller can rebase the whole chunk into the recentred /
    /// floating-origin frame in one integer add (mirroring
    /// [`Scene::resolve_chunk_rebased`](crate::scene::Scene::resolve_chunk_rebased)).
    ///
    /// `index_offset` is added (in i64, before the i32 downcast) to every emitted voxel
    /// index: pass `chunk_min_voxels − floating_origin_voxels` to land the chunk in the
    /// exact frame the dense store assembles, or `[0,0,0]` for chunk-local indices.
    pub fn expand_occupancy_into(&self, output: &mut Vec<Voxel>, index_offset: [i64; 3]) {
        let density = self.voxels_per_block.max(1);
        let block_extent = density as i64;

        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let block_low_voxels = [
                        block_x as i64 * block_extent,
                        block_y as i64 * block_extent,
                        block_z as i64 * block_extent,
                    ];
                    if let Some(block_id) = self.coarse_block(block) {
                        // Coarse-solid: fast d³ fill at the single block id, carrying the
                        // block's on-face-grid overlay marker (ADR 0003 §3c) so the
                        // expanded grid matches the dense resolve's `grid_overlay` bit
                        // (E5 — the fog grid + the grid-overlay parity read).
                        Self::fill_solid_block(
                            output,
                            block_low_voxels,
                            density,
                            block_id,
                            self.coarse_block_overlay(block),
                            index_offset,
                        );
                    } else if let Some(geometry) = self.microblocks.get(&block) {
                        // Boundary: expand each cuboid per-voxel at its material id.
                        Self::expand_boundary_block(
                            output,
                            block_low_voxels,
                            geometry,
                            index_offset,
                        );
                    }
                    // else: air block, nothing to emit.
                }
            }
        }
    }

    /// Fast-fill a coarse-solid block: every `density³` voxel at `block_id`, all carrying
    /// the block's `grid_overlay` render marker (ADR 0003 §3c).
    fn fill_solid_block(
        output: &mut Vec<Voxel>,
        block_low_voxels: [i64; 3],
        density: u32,
        block_id: BlockId,
        grid_overlay: bool,
        index_offset: [i64; 3],
    ) {
        for voxel_z in 0..density {
            for voxel_y in 0..density {
                for voxel_x in 0..density {
                    let chunk_local = [
                        block_low_voxels[0] + voxel_x as i64,
                        block_low_voxels[1] + voxel_y as i64,
                        block_low_voxels[2] + voxel_z as i64,
                    ];
                    output.push(stamped_voxel(
                        chunk_local,
                        [voxel_x as u8, voxel_y as u8, voxel_z as u8],
                        block_id,
                        grid_overlay,
                        index_offset,
                    ));
                }
            }
        }
    }

    /// Expand a boundary block's cuboids back to per-voxel occupancy at their material.
    fn expand_boundary_block(
        output: &mut Vec<Voxel>,
        block_low_voxels: [i64; 3],
        geometry: &MicroblockGeometry,
        index_offset: [i64; 3],
    ) {
        for cuboid in &geometry.cuboids {
            // The cuboid's `material_id` is the render-cell key (block_id | overlay<<15);
            // occupancy is the CLEAN categorical id, so mask the overlay bit off (ADR
            // 0003 §3c — the overlay never enters the occupancy / categorical cell) but
            // carry it onto the expanded voxel's `grid_overlay` render marker (E5 — so
            // the grid matches the dense resolve's per-voxel overlay bit).
            let block_id = BlockId(crate::cuboid_mesh::clean_block_id(cuboid.material_id));
            let grid_overlay = crate::cuboid_mesh::cell_key_has_overlay(cuboid.material_id);
            for voxel_z in cuboid.min[2]..=cuboid.max[2] {
                for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                    for voxel_x in cuboid.min[0]..=cuboid.max[0] {
                        let chunk_local = [
                            block_low_voxels[0] + voxel_x as i64,
                            block_low_voxels[1] + voxel_y as i64,
                            block_low_voxels[2] + voxel_z as i64,
                        ];
                        output.push(stamped_voxel(
                            chunk_local,
                            [voxel_x as u8, voxel_y as u8, voxel_z as u8],
                            block_id,
                            grid_overlay,
                            index_offset,
                        ));
                    }
                }
            }
        }
    }
}

/// Build one [`Voxel`] at chunk-local voxel index `chunk_local + index_offset` (i64 add
/// before the i32 downcast, ADR 0008), with `block_local_coord`, `block_id`, and the
/// `grid_overlay` render marker (ADR 0003 §3c — carried through so the expanded grid
/// matches the dense resolve's per-voxel overlay bit; E5).
#[inline]
fn stamped_voxel(
    chunk_local: [i64; 3],
    block_local_coord: [u8; 3],
    block_id: BlockId,
    grid_overlay: bool,
    index_offset: [i64; 3],
) -> Voxel {
    Voxel {
        local_index: [
            (chunk_local[0] + index_offset[0]) as i32,
            (chunk_local[1] + index_offset[1]) as i32,
            (chunk_local[2] + index_offset[2]) as i32,
        ],
        block_local_coord,
        block_id,
        attrs: BlockAttrs::DEFAULT,
        grid_overlay,
    }
}

/// Classify ONE block (the absolute-voxel box `block_abs_voxels`) against the scene's
/// leaves (ADR 0010 Decision 2), composing each boundable leaf's field interval by CSG
/// interval arithmetic and overriding to BOUNDARY for any unboundable leaf.
///
/// `block_abs_voxels` is the block's half-open `[min, max)` box in the SCENE's ABSOLUTE
/// voxel frame (the frame [`Scene::resolve_chunk`](crate::scene::Scene::resolve_chunk)
/// clips against — recentre is applied later as a pure index offset, so classification is
/// frame-independent). Each leaf's interval is taken in its OWN local voxel-index frame by
/// subtracting the leaf's `world_offset_voxels` (the same map
/// [`stamp_producer_into_chunk`](crate::scene) uses for its resolve window).
///
/// Returns:
/// * [`BlockClassification::Air`] iff EVERY overlapping leaf provably misses the block
///   (and no leaf is unboundable) — the conservative interval guarantees brute force
///   finds zero voxels.
/// * [`BlockClassification::CoarseSolid`] iff a single leaf provably fills the WHOLE block
///   solid (a Union's nearer surface wins) — every voxel occupied, one material.
/// * [`BlockClassification::Boundary`] otherwise (straddling, multi-leaf overlap that
///   can't be proven uniformly solid, or any unboundable leaf) — the always-safe verdict.
///
/// The classifier is CONSERVATIVE: a block it calls AIR or COARSE-SOLID is occupancy-
/// IDENTICAL to brute force; a block it calls BOUNDARY is resolved per-voxel and is exact
/// regardless. This is what makes the round-trip bit-identical to the dense path.
pub(crate) fn classify_chunk_block(
    leaves: &[&LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> BlockClassification {
    // Gather the leaves whose own grid AABB overlaps this block (others contribute
    // nothing). A leaf's absolute span is `[off, off + grid)` (corner-anchored).
    let mut overlapping: Vec<&LeafProducer> = Vec::new();
    for &leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        if leaf_box.intersects(&block_abs_voxels) {
            overlapping.push(leaf);
        }
    }

    if overlapping.is_empty() {
        // No leaf touches the block ⇒ provably empty.
        return BlockClassification::Air;
    }

    // v1 composes leaves by Union (CombineOp::Union, later-wins material on overlap).
    // Compose the conservative field intervals by `union_field_intervals` (min-of-fields):
    // any unboundable operand collapses the union to `None` ⇒ BOUNDARY.
    let composed = union_field_intervals(overlapping.iter().map(|leaf| {
        // Map the absolute block box into THIS leaf's local voxel-index frame `[0, full)`
        // by subtracting its world offset — the exact frame `cell_field_interval` expects
        // (ADR 0008: the frame is carried, never re-derived).
        let cell_local = VoxelAabb::new(
            [
                block_abs_voxels.min[0] - leaf.world_offset_voxels[0],
                block_abs_voxels.min[1] - leaf.world_offset_voxels[1],
                block_abs_voxels.min[2] - leaf.world_offset_voxels[2],
            ],
            [
                block_abs_voxels.max[0] - leaf.world_offset_voxels[0],
                block_abs_voxels.max[1] - leaf.world_offset_voxels[1],
                block_abs_voxels.max[2] - leaf.world_offset_voxels[2],
            ],
        );
        leaf.producer.cell_field_interval(cell_local, voxels_per_block)
    }));

    let Some(interval) = composed else {
        // An unboundable leaf in the union ⇒ resolve the block per-voxel (BOUNDARY).
        return BlockClassification::Boundary;
    };

    match interval.classify(SURFACE_ISOLEVEL) {
        FieldClassification::Air => BlockClassification::Air,
        FieldClassification::Boundary => BlockClassification::Boundary,
        FieldClassification::CoarseSolid => {
            // A composed-solid verdict is only SAFELY coarse when EXACTLY ONE leaf
            // overlaps the block: with two leaves the Union's per-voxel MATERIAL is
            // "later wins on overlap", which the composed interval cannot resolve (it
            // proves geometric solidity, not which id each voxel takes). A multi-leaf
            // solid block is therefore forced BOUNDARY so the per-voxel pass assigns the
            // correct (later-wins) material — still exact, just unelided. (A single-leaf
            // solid block is uniform-material by construction: a Tool is single-material.)
            match (overlapping.len(), overlapping[0].material) {
                // Single single-material leaf provably filling the block ⇒ elide to coarse.
                (1, Some(block_id)) => BlockClassification::CoarseSolid(block_id),
                // Multi-leaf overlap (Union material is per-voxel later-wins, not coarsely
                // decidable) OR a single leaf with no single-material id (a Part's
                // per-voxel materials) ⇒ resolve per-voxel. Still exact, just unelided.
                _ => BlockClassification::Boundary,
            }
        }
    }
}

/// The leaf's grid AABB in the SCENE's absolute voxel frame: `[off, off + grid)`,
/// corner-anchored at its `world_offset_voxels`. The single box construction shared by the
/// classify / overlap / whole-chunk paths (they must all test the SAME leaf extent).
fn leaf_world_box(leaf: &LeafProducer, voxels_per_block: u32) -> VoxelAabb {
    let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
    VoxelAabb::new(
        leaf.world_offset_voxels,
        [
            leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
            leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
            leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
        ],
    )
}

/// The FIRST leaf whose grid AABB overlaps `block_abs_voxels`, or `None` if none does. The
/// overlap test mirrors [`classify_chunk_block`]'s exactly, so the same leaf is found. A
/// coarse-solid block is owned by exactly one leaf (the classifier forces any multi-leaf
/// overlap to boundary), so for a coarse verdict this first hit IS the single owning leaf.
fn single_overlapping_leaf<'a>(
    leaves: &[&'a LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> Option<&'a LeafProducer> {
    leaves
        .iter()
        .copied()
        .find(|leaf| leaf_world_box(leaf, voxels_per_block).intersects(&block_abs_voxels))
}

/// The on-face-grid overlay (ADR 0003 §3c) of the SINGLE leaf overlapping `block_abs_voxels`,
/// or `false` if none overlaps (an unreachable case for a coarse-solid verdict — guarded
/// defensively).
fn single_overlapping_leaf_overlay(
    leaves: &[&LeafProducer],
    block_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> bool {
    single_overlapping_leaf(leaves, block_abs_voxels, voxels_per_block)
        .is_some_and(|leaf| leaf.grid_overlay)
}

/// The whole-CHUNK fast-path verdict (ADR 0010 Decision 2 — chunk-granular interval
/// elision). Evaluating the composed field interval ONCE at the whole-chunk cell can
/// decide the ENTIRE chunk without the 64 per-block calls, but only when the chunk verdict
/// PROVABLY implies the identical per-block outcome (CONSERVATIVE-NEVER-NARROW).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WholeChunkVerdict {
    /// The whole chunk is provably AIR: every block is empty, so the chunk is the empty
    /// [`TwoLayerChunk`] (no coarse ids, no microblocks).
    AllAir,
    /// The whole chunk is provably one coarse-solid material: every block is
    /// [`BlockClassification::CoarseSolid`] at `block_id` with the same `overlay`.
    AllCoarse { block_id: BlockId, overlay: bool },
    /// The chunk straddles the surface / is multi-leaf ambiguous / unboundable: fall back
    /// to the per-block classify (the always-safe path).
    PerBlock,
}

/// Classify a whole chunk from ONE composed interval at the chunk cell (ADR 0010 Decision 2).
///
/// **Why the chunk verdict is byte-identical to the per-block sweep.** The composed field
/// interval is *inclusion-monotone*: for a sub-block box `B ⊆ chunk` every operand's bound
/// over `B` nests inside its bound over the chunk (the Lipschitz-centre bound because
/// `dist(centre_chunk, centre_B) + circumradius(B) ≤ circumradius(chunk)` for nested
/// axis-aligned boxes; the sketch discrete bound because `B`'s footprint rectangle is a
/// SUBSET of the chunk's), and a sub-block's overlapping-leaf set is a SUBSET of the chunk's
/// (`B ⊆ chunk`). Therefore:
///
/// * **AIR** (`minimum > isolevel`) ⇒ every block's `minimum` is `≥` the chunk's ⇒ every
///   block is AIR ⇒ the empty chunk. (Even a block the per-block sweep called BOUNDARY would
///   resolve to zero voxels here, since the chunk is provably all-outside — still empty.)
/// * **COARSE-SOLID** — [`classify_chunk_block`] only yields it for a SINGLE single-material
///   leaf, so the resolution is provably uniform across the chunk. We ADDITIONALLY require
///   that leaf's grid AABB to CONTAIN the whole chunk, so every sub-block lies inside the
///   leaf's extent ⇒ that one leaf overlaps every block (no block collapses to AIR) ⇒ every
///   block's `maximum` is `≤` the chunk's ⇒ every block is `CoarseSolid(block_id)` with the
///   same `overlay`. (For `SketchSolid`/`SdfShape` a coarse chunk is always fully interior,
///   so the containment guard never rejects a legitimate coarse chunk; it only defends
///   against a hypothetical producer whose field is negative OUTSIDE its own AABB.)
/// * anything else ⇒ `PerBlock`.
fn classify_whole_chunk(
    leaves: &[&LeafProducer],
    chunk_abs_voxels: VoxelAabb,
    voxels_per_block: u32,
) -> WholeChunkVerdict {
    match classify_chunk_block(leaves, chunk_abs_voxels, voxels_per_block) {
        BlockClassification::Air => WholeChunkVerdict::AllAir,
        BlockClassification::Boundary => WholeChunkVerdict::PerBlock,
        BlockClassification::CoarseSolid(block_id) => {
            // classify_chunk_block returns CoarseSolid ONLY for a single single-material
            // leaf; recover it (the sole overlapping leaf) to read its overlay AND to prove
            // its grid AABB encloses the whole chunk.
            match single_overlapping_leaf(leaves, chunk_abs_voxels, voxels_per_block) {
                Some(leaf)
                    if leaf_world_box(leaf, voxels_per_block).contains_box(&chunk_abs_voxels) =>
                {
                    WholeChunkVerdict::AllCoarse { block_id, overlay: leaf.grid_overlay }
                }
                _ => WholeChunkVerdict::PerBlock,
            }
        }
    }
}

/// The OFF-by-default capability that builds the [`TwoLayerChunk`] display cache from the
/// one evaluator (ADR 0010 Decision 3 / 6). When the capability is OFF the live store
/// stays on the dense [`crate::store::Store`] path; this type is only constructed when a
/// caller opts in (the parity gate, and — later — E3's mesher).
///
/// It is a thin, stateless builder (no resident cache of its own in this slice — the
/// dense `Store` remains the live cache); a chunk is built on demand from the scene.
#[derive(Debug, Clone, Copy, Default)]
pub struct TwoLayerStore {
    /// The capability flag (ADR 0010 Decision 6). `false` (the default) means the
    /// two-layer path is OFF and [`build_chunk`](Self::build_chunk) returns `None`, so a
    /// caller falls back to the dense path; `true` engages the boundary-aware build.
    enabled: bool,
}

impl TwoLayerStore {
    /// A store with the two-layer capability ENABLED. The default ([`Default`]) is
    /// DISABLED, matching the ADR's "OFF by default, dense fallback" coexistence.
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Whether the two-layer capability is engaged.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Build the [`TwoLayerChunk`] for EVERY covering chunk of `scene` (ADR 0010 E3): the
    /// `(absolute_chunk_coord, chunk)` list the two-layer mesher
    /// ([`crate::cuboid_mesh::CuboidMeshRenderer::new_from_two_layer_chunks`]) consumes,
    /// visited in the SAME z,y,x order the dense store assembles. Returns an empty list when
    /// the capability is OFF or the scene has no covering chunk range (a Part-only scene —
    /// the caller falls back to the dense path). This keeps the `pub(crate)` chunk-range
    /// logic inside the crate while exposing the covering-chunk build to the `shot` binary.
    pub fn build_covering_chunks(
        &self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        if !self.enabled {
            return Vec::new();
        }
        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        // #63: HOIST the leaf list out of the per-chunk build — compute it ONCE (re-walking
        // the node tree + cloning every producer per covering chunk was the O(objects²) sink)
        // and share the read-only slice across the parallel build. Then the EDIT BROADPHASE
        // (#66, ADR 0011 Decision 4b): a stateless per-build BVH over the leaf world-AABBs;
        // each chunk queries its own box and is classified against only the overlapping
        // candidates, keeping the build ~O(chunks × (log leaves + candidates)).
        let leaves = scene.leaf_producers(voxels_per_block);
        let broadphase = leaf_edit_broadphase(&leaves, voxels_per_block);
        // Each covering chunk is built independently from the (read-only, `Sync`) leaf slice
        // + broadphase, so the wholesale build is embarrassingly parallel (#57). Enumerate
        // the coords in the SAME z,y,x order the dense store assembles (X fastest), then map
        // each coord → its chunk with rayon. A parallel `.collect()` PRESERVES ordering, so
        // the output Vec is byte-identical to the serial nested loop regardless of thread
        // count.
        let coords = enumerate_covering_chunk_coords(min_chunk, max_chunk);
        coords
            .into_par_iter()
            .map(|coord| {
                let candidates =
                    chunk_candidate_leaves(&broadphase, &leaves, coord, voxels_per_block);
                let chunk =
                    build_two_layer_chunk_from_leaves(coord, &candidates, voxels_per_block);
                // `Arc`-wrap the freshly built chunk so this owned covering set can be
                // handed to the mesh / brick / fog readers, retained in the mesh
                // renderer, AND moved into the async `GeometryRebuildRequest` with only
                // O(1) refcount bumps — never a deep `TwoLayerChunk` copy (the per-edit
                // clone this cleanup killed; ADR 0011 G3 record/atlas territory).
                (coord, Arc::new(chunk))
            })
            .collect()
    }

    /// Build the [`TwoLayerChunk`] for `chunk_coord` from the scene's one evaluator, or
    /// `None` when the capability is OFF (the caller then uses the dense path). The
    /// returned chunk is in chunk-local frame; [`TwoLayerChunk::expand_occupancy_into`]
    /// rebases it to match the dense store.
    ///
    /// `lod` is the parked LOD seam (always `0`), kept for call-site symmetry with the
    /// dense store.
    pub fn build_chunk(
        &self,
        chunk_coord: [i32; 3],
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Option<TwoLayerChunk> {
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        if !self.enabled {
            return None;
        }
        Some(build_two_layer_chunk(chunk_coord, scene, voxels_per_block))
    }
}

/// **The resident two-layer display cache (ADR 0010 #54 — chunk-granular incremental edits).**
///
/// The [`TwoLayerStore`] above is a *stateless* builder (every call re-classifies a chunk from
/// the scene); this is its **incremental-edit counterpart**, the two-layer analogue of the dense
/// [`crate::store::Store`]. It holds the resident [`TwoLayerChunk`]s across edits and re-derives
/// **only the chunks an edit's world-AABB intersects** (chunk-granular, ADR 0002 Decision 3),
/// mirroring [`Store::invalidate_aabb`](crate::store::Store::invalidate_aabb) exactly. Untouched
/// chunks stay resident.
///
/// ## Why a dirty chunk re-runs the whole build
///
/// A dirty chunk drops its cached [`TwoLayerChunk`] and, on next access, re-runs the block
/// classifier + two-layer build ([`build_two_layer_chunk`]) from scratch. Chunk-granular is
/// sufficient to unblock E5 (retire the dense path); a **block-granular dirty-brick recompute**
/// (re-classify only the blocks the edit AABB touches, keeping the rest of the chunk's coarse
/// layer) is a later optimization, NOT this slice (ADR 0010 Consequences).
///
/// ## Frame (ADR 0008) — why a recentre shift does NOT invalidate the cache
///
/// A [`TwoLayerChunk`] is stored in **chunk-local integer** frame (its coarse ids + block-local
/// cuboids never mention the absolute origin — that lives in the chunk COORD key). The recentre /
/// floating origin is applied only at *expand* time as a pure index offset
/// ([`TwoLayerChunk::expand_occupancy_into`]). So — unlike the dense [`Store`], which caches
/// PRE-REBASED grids and must clear on a floating-origin shift — a recentre shift leaves every
/// resident two-layer chunk VALID. Only a **density change** (which resizes each chunk's voxel
/// extent) forces a wholesale clear; that is the one binding this cache tracks.
///
/// [`Store`]: crate::store::Store
#[derive(Debug, Clone, Default)]
pub struct TwoLayerResidentCache {
    /// The two-layer capability flag (ADR 0010 Decision 6), forwarded to the stateless builder.
    /// `false` (the default) means the cache stays empty and [`resident_two_layer_chunks`] is a
    /// no-op, so a caller falls back to the dense path.
    ///
    /// [`resident_two_layer_chunks`]: Self::resident_two_layer_chunks
    enabled: bool,
    /// Resident chunks keyed by ABSOLUTE chunk coord (the only LOD in use is 0, ADR 0002 S4a).
    ///
    /// Stored as `Arc<TwoLayerChunk>` so [`resident_two_layer_chunks`](Self::resident_two_layer_chunks)
    /// can hand the owned covering set out to the readers (mesh / brick / fog) and into the async
    /// `GeometryRebuildRequest` with an O(1) refcount bump each, never a deep chunk copy per rebuild.
    ///
    /// **Mutation discipline (why an `Arc` is safe here).** A dirty chunk is never mutated
    /// through its `Arc` while shell copies are alive: the cache only ever REPLACES a chunk's
    /// entry with a freshly built `Arc` (evict via [`invalidate_aabb`](Self::invalidate_aabb) /
    /// [`clear`](Self::clear), then re-`insert` in [`resident_two_layer_chunks`]), so an
    /// outstanding shared copy keeps seeing the exact chunk it was handed. No `Arc::make_mut` /
    /// in-place edit path exists — the resident chunk is immutable once built.
    resident: BTreeMap<[i32; 3], Arc<TwoLayerChunk>>,
    /// The density the resident chunks were built at. A change resizes every chunk's voxel
    /// extent, so it forces a wholesale [`clear`](Self::clear) (mirrors
    /// [`Store::rebind_if_changed`](crate::store::Store)'s density guard).
    bound_density: Option<u32>,
}

impl TwoLayerResidentCache {
    /// A resident cache with the two-layer capability ENABLED. The default ([`Default`]) is
    /// DISABLED (empty, no-op), matching the ADR's "OFF by default, dense fallback" coexistence.
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            resident: BTreeMap::new(),
            bound_density: None,
        }
    }

    /// Whether the two-layer capability is engaged.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The number of chunks currently resident (a diagnostic / test-observability count).
    pub fn resident_len(&self) -> usize {
        self.resident.len()
    }

    /// Drop every cached chunk (the all-or-nothing invalidation seam) — the two-layer analogue
    /// of [`Store::clear`](crate::store::Store::clear). Used for the first build (no previous
    /// scene to diff) and the edit kinds [`invalidate_aabb`](Self::invalidate_aabb) can't
    /// localise (a density change, or a region-spanning Part edit).
    pub fn clear(&mut self) {
        self.resident.clear();
        self.bound_density = None;
    }

    /// **Targeted invalidation (ADR 0010 #54, mirroring
    /// [`Store::invalidate_aabb`](crate::store::Store::invalidate_aabb)).** Drop exactly the
    /// cached chunks whose half-open box intersects the edit world-AABB `edit_aabb` (absolute
    /// voxels, producer-true frame), at `voxels_per_block` — ADR 0002 Decision 3's whole-chunk
    /// dirty granularity. Every other cached chunk stays resident untouched, so the next
    /// [`resident_two_layer_chunks`](Self::resident_two_layer_chunks) re-runs the classifier +
    /// build only for the evicted (dirty) chunks.
    ///
    /// `edit_aabb` is what
    /// [`LeafSpatialIndex::edit_aabb_since`](crate::spatial_index::LeafSpatialIndex::edit_aabb_since)
    /// returns: the union of an edit's old and new leaf boxes, so a moved node dirties chunks
    /// around BOTH its source and destination. An empty `edit_aabb` evicts nothing.
    ///
    /// A density mismatch against the bound density is treated conservatively (the AABB was
    /// computed at a different chunk size) by clearing everything — belt-and-braces, as the
    /// caller already falls back to [`clear`](Self::clear) for a density change.
    ///
    /// **Returns the chunk coords actually evicted** (resident AND intersecting the edit AABB),
    /// so the mesher's incremental plan ([`crate::cuboid_mesh::cuboid_incremental_plan`]) can
    /// dilate exactly this dirty set by the 26-neighbourhood. The density-mismatch path returns
    /// every previously-resident coord.
    pub fn invalidate_aabb(
        &mut self,
        edit_aabb: &VoxelAabb,
        voxels_per_block: u32,
    ) -> Vec<[i32; 3]> {
        if let Some(bound) = self.bound_density {
            if bound != voxels_per_block {
                let evicted: Vec<[i32; 3]> = self.resident.keys().copied().collect();
                self.clear();
                return evicted;
            }
        }
        let Some((min_chunk, max_chunk)) = edit_aabb.covering_chunk_range(voxels_per_block) else {
            return Vec::new(); // empty edit AABB — nothing to invalidate.
        };
        let mut evicted = Vec::new();
        self.resident.retain(|coord, _| {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                evicted.push(*coord);
            }
            !inside
        });
        evicted
    }

    /// **Per-chunk two-layer accessor — the incremental analogue of
    /// [`Store::resident_render_chunks`](crate::store::Store::resident_render_chunks).** Ensure
    /// every covering chunk of `(scene, voxels_per_block, lod)` is resident (re-run the
    /// classifier + build for any DIRTY or MISSING chunk, reuse resident HITs verbatim), then
    /// return every covering chunk as `([i32; 3] absolute_chunk_coord, Arc<TwoLayerChunk>)` in the
    /// SAME z,y,x order the dense store assembles.
    ///
    /// Because a two-layer chunk is chunk-local-integer (frame-independent), a resident HIT is
    /// reused across a recentre shift; only [`invalidate_aabb`](Self::invalidate_aabb) (a dirty
    /// edit) or a density change ([`clear`](Self::clear)) re-derives a chunk. The returned chunks
    /// are `Arc`-SHARED (an O(1) refcount bump per covering chunk, NOT a deep copy), so the caller
    /// owns a covering set that outlives this `&mut self` borrow and can be meshed, fog-expanded,
    /// brick-packed AND moved into the async mesh request without cloning a single chunk. The fill
    /// (needing `&mut self`) runs FIRST, then the gather clones the resident `Arc`s.
    ///
    /// Returns an empty `Vec` when the capability is OFF (dense fallback) or the scene has no
    /// covering chunk range (a Part-only scene).
    pub fn resident_two_layer_chunks(
        &mut self,
        scene: &Scene,
        voxels_per_block: u32,
        lod: u32,
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        debug_assert_eq!(lod, 0, "E2 only builds full resolution (lod 0)");
        if !self.enabled {
            return Vec::new();
        }
        // A density change resizes every chunk's voxel extent; drop the stale residents.
        if self.bound_density != Some(voxels_per_block) {
            self.resident.clear();
            self.bound_density = Some(voxels_per_block);
        }

        let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
            return Vec::new();
        };

        // Fill misses (dirty-evicted or never-built). The build (`build_two_layer_chunk`)
        // is the ~3.5s cost and is pure given the scene, so parallelise the WHOLESALE fill
        // (#57): gather the missing coords, build them in parallel into a Vec, THEN insert
        // serially (the insert is cheap next to the build). This keeps the incremental
        // dirty-set path (#54) intact — only chunks actually absent are (re)built, resident
        // HITs are reused verbatim — while the initial build / density-change / recentre
        // fallback (which re-fills many chunks at once) now runs across threads. Each chunk
        // is deterministic given the scene, so the resident map is identical to the serial
        // one-by-one fill regardless of thread count.
        //
        // #63: HOIST the leaf list out of the per-chunk build (compute ONCE, not per missing
        // chunk); #66: the EDIT BROADPHASE (ADR 0011 Decision 4b) — a stateless per-build BVH
        // over the leaf world-AABBs, queried per missing chunk, so each is built from only
        // its overlapping candidate leaves. Only chunks actually absent are (re)built,
        // resident HITs are reused verbatim (the #54 dirty-set path is intact).
        let leaves = scene.leaf_producers(voxels_per_block);
        let broadphase = leaf_edit_broadphase(&leaves, voxels_per_block);
        let missing_coords: Vec<[i32; 3]> =
            enumerate_covering_chunk_coords(min_chunk, max_chunk)
                .into_iter()
                .filter(|coord| !self.resident.contains_key(coord))
                .collect();
        let freshly_built: Vec<([i32; 3], Arc<TwoLayerChunk>)> = missing_coords
            .into_par_iter()
            .map(|coord| {
                let candidates =
                    chunk_candidate_leaves(&broadphase, &leaves, coord, voxels_per_block);
                (
                    coord,
                    Arc::new(build_two_layer_chunk_from_leaves(
                        coord,
                        &candidates,
                        voxels_per_block,
                    )),
                )
            })
            .collect();
        for (coord, chunk) in freshly_built {
            self.resident.insert(coord, chunk);
        }

        // Gather the covering chunks as O(1) `Arc` clones (all HITs after the fill above) — the
        // caller gets an owned, shareable covering set with no deep chunk copy.
        let resident = &self.resident;
        let mut chunks = Vec::new();
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    if let Some(chunk) = resident.get(&coord) {
                        chunks.push((coord, Arc::clone(chunk)));
                    }
                }
            }
        }
        chunks
    }
}

/// Enumerate every covering chunk coord in the inclusive `[min_chunk, max_chunk]` range,
/// in the SAME z,y,x order (X fastest, then Y, then Z) the dense store assembles them. This
/// materialises the coords into a `Vec` so the wholesale build (#57) can `into_par_iter()`
/// them and `.collect()` back into an identically-ordered result.
fn enumerate_covering_chunk_coords(min_chunk: [i32; 3], max_chunk: [i32; 3]) -> Vec<[i32; 3]> {
    let mut coords = Vec::new();
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                coords.push([chunk_x, chunk_y, chunk_z]);
            }
        }
    }
    coords
}

/// The leaf's world-AABB in absolute voxels: `[world_offset, world_offset + full_dimensions)`,
/// corner-anchored — the SAME box [`classify_chunk_block`] / [`resolve_boundary_block`] test
/// each block against. A region-spanning Part (the cloud field) reports its composite-region
/// `full_dimensions`, so its box correctly spans every chunk it fills.
fn leaf_world_aabb(leaf: &LeafProducer, voxels_per_block: u32) -> VoxelAabb {
    let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
    VoxelAabb::new(
        leaf.world_offset_voxels,
        [
            leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
            leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
            leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
        ],
    )
}

/// **The edit broadphase over a scene's leaves (#66, ADR 0011 Decision 4b).** Build the
/// stateless per-build [`EditBroadphaseBvh`] over every leaf's world AABB, indexed by the
/// leaf's position in `leaves` (document order). Rebuilt from scratch on every wholesale
/// build / edit — never persisted across edits (no invalidation obligation, the C1 lesson).
fn leaf_edit_broadphase(leaves: &[LeafProducer], voxels_per_block: u32) -> EditBroadphaseBvh {
    let leaf_aabbs: Vec<VoxelAabb> = leaves
        .iter()
        .map(|leaf| leaf_world_aabb(leaf, voxels_per_block))
        .collect();
    EditBroadphaseBvh::build(&leaf_aabbs)
}

/// The half-open absolute-voxel box of the chunk at `chunk_coord` — the query box the edit
/// broadphase answers per covering chunk.
fn chunk_world_voxel_aabb(chunk_coord: [i32; 3], voxels_per_block: u32) -> VoxelAabb {
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    let min = [
        chunk_coord[0] as i64 * chunk_extent_voxels,
        chunk_coord[1] as i64 * chunk_extent_voxels,
        chunk_coord[2] as i64 * chunk_extent_voxels,
    ];
    VoxelAabb::new(
        min,
        [
            min[0] + chunk_extent_voxels,
            min[1] + chunk_extent_voxels,
            min[2] + chunk_extent_voxels,
        ],
    )
}

/// Resolve one chunk's broadphase candidates to borrows of the shared leaf slice (no clone
/// — the classifier reads them read-only). The BVH returns indices sorted ascending, i.e. a
/// document-order subsequence of `leaves` (a filter, never a reorder), so later-wins Union
/// material resolution is unchanged.
///
/// EXACTNESS: a leaf whose AABB does not overlap the chunk cannot affect any block in it, so
/// a chunk classified against only its overlapping candidates is byte-identical to one
/// classified against all leaves (the per-block AABB tests inside the classifier already
/// narrow further per block — the broadphase just hands them a smaller exact-superset set).
fn chunk_candidate_leaves<'leaf_slice>(
    broadphase: &EditBroadphaseBvh,
    leaves: &'leaf_slice [LeafProducer],
    chunk_coord: [i32; 3],
    voxels_per_block: u32,
) -> Vec<&'leaf_slice LeafProducer> {
    broadphase
        .overlapping_input_indices(&chunk_world_voxel_aabb(chunk_coord, voxels_per_block))
        .into_iter()
        .map(|leaf_index| &leaves[leaf_index])
        .collect()
}

/// Build one chunk's two-layer representation by classifying every block and resolving the
/// boundary blocks per-voxel (the evaluator → display-cache step, ADR 0010 Decision 3).
///
/// Stateless single-chunk entry (tests, incremental single-chunk rebuild): it computes the
/// scene's leaf list itself, then delegates to [`build_two_layer_chunk_from_leaves`]. The
/// BULK paths ([`TwoLayerStore::build_covering_chunks`],
/// [`TwoLayerResidentCache::resident_two_layer_chunks`]) hoist `leaf_producers` out of the
/// per-chunk loop (#63) and pass a pre-filtered candidate slice into
/// [`build_two_layer_chunk_from_leaves`] directly — so the O(chunks) tree-walk + producer
/// clone never happens per chunk there.
fn build_two_layer_chunk(
    chunk_coord: [i32; 3],
    scene: &Scene,
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let leaves = scene.leaf_producers(voxels_per_block);
    let candidates: Vec<&LeafProducer> = leaves.iter().collect();
    build_two_layer_chunk_from_leaves(chunk_coord, &candidates, voxels_per_block)
}

/// Build one chunk's two-layer representation from a pre-computed leaf candidate slice — the
/// hoisted core of [`build_two_layer_chunk`] (#63).
///
/// `leaves` MUST be a document-order subsequence of `scene.leaf_producers(voxels_per_block)`
/// (a filter, never a reorder) that INCLUDES every leaf whose world AABB overlaps this chunk.
/// The edit broadphase ([`chunk_candidate_leaves`]) guarantees exactly that: a leaf whose AABB
/// does NOT overlap the chunk cannot affect ANY block in it (the per-block AABB tests inside
/// [`classify_chunk_block`] / [`resolve_boundary_block`] would skip it regardless), so passing
/// only the chunk-overlapping candidates yields IDENTICAL coarse / microblock / seam output
/// while preserving later-wins Union material resolution (document order kept).
fn build_two_layer_chunk_from_leaves(
    chunk_coord: [i32; 3],
    leaves: &[&LeafProducer],
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let density = voxels_per_block.max(1);
    let chunk_extent_voxels = (CHUNK_BLOCKS * density) as i64;
    let chunk_min_voxels = [
        chunk_coord[0] as i64 * chunk_extent_voxels,
        chunk_coord[1] as i64 * chunk_extent_voxels,
        chunk_coord[2] as i64 * chunk_extent_voxels,
    ];

    // CHUNK-GRANULAR INTERVAL FAST PATH (ADR 0010 Decision 2): decide the whole chunk from
    // ONE composed interval at the chunk cell. A solid interior chunk is 1 interval call
    // instead of 64 per-block calls — the O(volume) → O(surface) win for large solids.
    // Only a verdict that PROVABLY implies the identical per-block outcome short-circuits;
    // any ambiguity falls back to the byte-identical per-block sweep below.
    let chunk_abs = VoxelAabb::new(
        chunk_min_voxels,
        [
            chunk_min_voxels[0] + chunk_extent_voxels,
            chunk_min_voxels[1] + chunk_extent_voxels,
            chunk_min_voxels[2] + chunk_extent_voxels,
        ],
    );
    match classify_whole_chunk(leaves, chunk_abs, voxels_per_block) {
        WholeChunkVerdict::AllAir => return TwoLayerChunk::empty(density),
        WholeChunkVerdict::AllCoarse { block_id, overlay } => {
            let mut chunk = TwoLayerChunk::empty(density);
            for flat in 0..chunk.coarse.len() {
                chunk.coarse[flat] = Some(block_id);
                chunk.coarse_overlay[flat] = overlay;
            }
            return chunk;
        }
        WholeChunkVerdict::PerBlock => {}
    }

    build_two_layer_chunk_per_block(chunk_min_voxels, leaves, density, voxels_per_block)
}

/// The per-block classify sweep — the always-correct fallback of
/// [`build_two_layer_chunk_from_leaves`] (and the parity oracle the fast-path test compares
/// against). Classifies every one of the chunk's `CHUNK_BLOCKS³` blocks independently and
/// resolves boundary blocks per-voxel. `chunk_min_voxels` is the chunk's low corner in
/// absolute voxels.
fn build_two_layer_chunk_per_block(
    chunk_min_voxels: [i64; 3],
    leaves: &[&LeafProducer],
    density: u32,
    voxels_per_block: u32,
) -> TwoLayerChunk {
    let block_extent = density as i64;

    let mut chunk = TwoLayerChunk::empty(density);

    for block_z in 0..CHUNK_BLOCKS {
        for block_y in 0..CHUNK_BLOCKS {
            for block_x in 0..CHUNK_BLOCKS {
                let block = [block_x, block_y, block_z];
                // The block's half-open box in the SCENE's absolute voxel frame.
                let block_min = [
                    chunk_min_voxels[0] + block_x as i64 * block_extent,
                    chunk_min_voxels[1] + block_y as i64 * block_extent,
                    chunk_min_voxels[2] + block_z as i64 * block_extent,
                ];
                let block_abs = VoxelAabb::new(
                    block_min,
                    [
                        block_min[0] + block_extent,
                        block_min[1] + block_extent,
                        block_min[2] + block_extent,
                    ],
                );

                match classify_chunk_block(leaves, block_abs, voxels_per_block) {
                    BlockClassification::Air => {}
                    BlockClassification::CoarseSolid(block_id) => {
                        let flat = coarse_flat_index(block);
                        chunk.coarse[flat] = Some(block_id);
                        // A coarse-solid block is owned by EXACTLY ONE leaf (the classifier
                        // forces multi-leaf overlaps to boundary), so its on-face-grid
                        // overlay (ADR 0003 §3c) is that single leaf's `grid_overlay`.
                        chunk.coarse_overlay[flat] =
                            single_overlapping_leaf_overlay(leaves, block_abs, voxels_per_block);
                    }
                    BlockClassification::Boundary => {
                        let geometry =
                            resolve_boundary_block(leaves, block_min, density, voxels_per_block);
                        // A boundary verdict the per-voxel pass found EMPTY contributes
                        // nothing (still exact — the interval was merely conservative).
                        if !geometry.cuboids.is_empty() {
                            chunk.microblocks.insert(block, geometry);
                        }
                    }
                }
            }
        }
    }

    chunk
}

/// Resolve a boundary block per-voxel into a dense `density³` [`VoxelRegion`] (the
/// material at each occupied voxel), decompose it to cuboids, and compute its per-face
/// seam-solidity flags. `block_min_abs` is the block's low corner in absolute voxels.
///
/// Per-voxel resolution reuses each overlapping leaf's [`VoxelProducer::resolve_into`]
/// over the block window, composed by the SAME Union semantics the dense path uses
/// (document order, later-wins on overlap) — so the materialised block is bit-identical
/// to the dense store's voxels for that block.
fn resolve_boundary_block(
    leaves: &[&LeafProducer],
    block_min_abs: [i64; 3],
    density: u32,
    voxels_per_block: u32,
) -> MicroblockGeometry {
    let extent = [density, density, density];
    let mut region = VoxelRegion::new_empty(extent);

    // Compose leaves in DOCUMENT ORDER (the order `leaf_producers` yields them, which is
    // `for_each_leaf`'s walk order), later-wins on overlap — exactly the dense Union.
    for &leaf in leaves {
        let grid_dimensions = leaf.producer.full_dimensions(voxels_per_block);
        let leaf_box = VoxelAabb::new(
            leaf.world_offset_voxels,
            [
                leaf.world_offset_voxels[0] + grid_dimensions[0] as i64,
                leaf.world_offset_voxels[1] + grid_dimensions[1] as i64,
                leaf.world_offset_voxels[2] + grid_dimensions[2] as i64,
            ],
        );
        let block_abs = VoxelAabb::new(
            block_min_abs,
            [
                block_min_abs[0] + density as i64,
                block_min_abs[1] + density as i64,
                block_min_abs[2] + density as i64,
            ],
        );
        if !leaf_box.intersects(&block_abs) {
            continue;
        }

        // Resolve JUST this block's window in the leaf's local voxel-index frame.
        let window_local = VoxelAabb::new(
            [
                block_min_abs[0] - leaf.world_offset_voxels[0],
                block_min_abs[1] - leaf.world_offset_voxels[1],
                block_min_abs[2] - leaf.world_offset_voxels[2],
            ],
            [
                block_min_abs[0] + density as i64 - leaf.world_offset_voxels[0],
                block_min_abs[1] + density as i64 - leaf.world_offset_voxels[1],
                block_min_abs[2] + density as i64 - leaf.world_offset_voxels[2],
            ],
        );
        let mut local = VoxelGrid::default();
        leaf.producer
            .resolve_into(&mut local, voxels_per_block, window_local);

        // Stamp each emitted voxel into the block-local region at its material (a Tool
        // overrides every voxel's id; a Part keeps its own per-voxel id). The voxel's
        // local index is in the LEAF's frame, so shift back to block-local by adding the
        // leaf offset and subtracting the block's absolute low corner.
        for voxel in &local.occupied {
            let block_local = [
                voxel.local_index[0] as i64 + leaf.world_offset_voxels[0] - block_min_abs[0],
                voxel.local_index[1] as i64 + leaf.world_offset_voxels[1] - block_min_abs[1],
                voxel.local_index[2] as i64 + leaf.world_offset_voxels[2] - block_min_abs[2],
            ];
            if block_local.iter().any(|&c| c < 0 || c >= density as i64) {
                continue; // Outside this block (the window clamps, but guard anyway).
            }
            let block_id = match leaf.material {
                Some(id) => id.0,
                None => voxel.block_id.0,
            };
            // Stamp the cuboid mesher's RENDER-CELL key (ADR 0003 §3c): the clean
            // categorical id in the low bits, this leaf's on-face-grid overlay in the
            // dedicated bit. So `decompose_into_boxes` splits a box across differing
            // overlay states exactly like the dense mesher, and the E3 mesher reads the
            // box's clean id + overlay back out — without the render flag ever entering
            // the categorical cell (the E2 occupancy expansion masks the bit off).
            let render_key = crate::cuboid_mesh::compose_cell_key(block_id, leaf.grid_overlay);
            // Later document-order leaf wins on overlap: a plain overwrite reproduces
            // the dense Union (the walk visits in document order, last write persists).
            region.set(
                block_local[0] as u32,
                block_local[1] as u32,
                block_local[2] as u32,
                Some(render_key),
            );
        }
    }

    let cuboids = decompose_into_boxes(&region);
    let seam_solidity = compute_seam_solidity(&region);
    MicroblockGeometry {
        cuboids,
        seam_solidity,
    }
}

/// Compute the per-face seam-solidity flags for a resolved boundary block: a face is solid
/// iff EVERY voxel cell on that face of the `density³` region is occupied.
fn compute_seam_solidity(region: &VoxelRegion) -> SeamSolidity {
    let [width, height, depth] = region.extent;
    let mut solid = [[true; 2]; 3];
    // Degenerate (zero-extent) region: no face can be solid.
    if width == 0 || height == 0 || depth == 0 {
        return SeamSolidity {
            solid: [[false; 2]; 3],
        };
    }

    // X faces (axis 0): low x == 0, high x == width - 1.
    for &(side, x) in &[(0usize, 0u32), (1usize, width - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for y in 0..height {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[0][side] = face_solid;
    }
    // Y faces (axis 1): low y == 0, high y == height - 1.
    for &(side, y) in &[(0usize, 0u32), (1usize, height - 1)] {
        let mut face_solid = true;
        'scan: for z in 0..depth {
            for x in 0..width {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[1][side] = face_solid;
    }
    // Z faces (axis 2): low z == 0, high z == depth - 1.
    for &(side, z) in &[(0usize, 0u32), (1usize, depth - 1)] {
        let mut face_solid = true;
        'scan: for y in 0..height {
            for x in 0..width {
                if region.material_at(x, y, z).is_none() {
                    face_solid = false;
                    break 'scan;
                }
            }
        }
        solid[2][side] = face_solid;
    }

    SeamSolidity { solid }
}

/// Stream a whole scene's two-layer chunks back to one recentred [`VoxelGrid`], in the
/// EXACT frame the dense [`Store::resolve_region`](crate::store::Store::resolve_region)
/// assembles — the parity gate's "two-layer round-trip" (ADR 0010 parity (a)). Builds
/// each covering chunk via the capability, expands it (coarse fast-fill + boundary
/// per-voxel), and rebases by `chunk_min_voxels − recentre` so the occupied SET matches
/// the dense path bit-for-bit (position + block id).
///
/// Returns `None` when the capability is OFF (the caller stays on the dense path) or the
/// scene has no covering chunk range (a Part-only scene — handled by the dense path).
///
/// **Oracle — compile-gated.** This streams a whole-region dense [`VoxelGrid`] purely so
/// the parity gate can compare it against `Store::resolve_region`; no runtime display
/// path assembles a whole-region grid. It is excluded from production builds behind the
/// `oracle` feature (tests reach it via `cfg(test)`), so a dense whole-region resolve is
/// a compile error in production — see the proof chapter's "Oracles" section
/// (`docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
pub fn resolve_region_two_layer(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    lod: u32,
) -> Option<VoxelGrid> {
    if !store.is_enabled() {
        return None;
    }
    debug_assert_eq!(lod, 0, "E2 only resolves full resolution (lod 0)");

    let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
    let mut output = VoxelGrid::new(region_dimensions);
    // ADR 0008: carry the recentre so consumers decode world→index without re-deriving it
    // — the same value the dense `resolve_region` stamps (the parity gate compares it).
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    output.recentre_voxels = recentre.voxels();

    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        return Some(output); // No composite extent (Part-only): empty region.
    };

    // Unwrap the frame at the per-chunk rebase arithmetic below.
    let recentre_voxels = recentre.voxels();
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    // Visit chunks in the SAME z,y,x order the dense store assembles them in, so the
    // emitted voxel order matches too (the multiset compare is order-independent, but
    // keeping the order identical makes a future ordered compare cheap).
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                let chunk_coord = [chunk_x, chunk_y, chunk_z];
                let chunk = store
                    .build_chunk(chunk_coord, scene, voxels_per_block, lod)
                    .expect("capability is enabled");
                // Rebase chunk-local indices into the recentred frame: a voxel at
                // chunk-local index `l` has absolute index `chunk_min + l`, and the
                // recentred frame subtracts `recentre` — so the offset is
                // `chunk_min − recentre` (i64, before the i32 downcast). This is exactly
                // `resolve_chunk_rebased`'s rebase with floating origin = recentre.
                let index_offset = [
                    chunk_x as i64 * chunk_extent_voxels - recentre_voxels[0],
                    chunk_y as i64 * chunk_extent_voxels - recentre_voxels[1],
                    chunk_z as i64 * chunk_extent_voxels - recentre_voxels[2],
                ];
                chunk.expand_occupancy_into(&mut output.occupied, index_offset);
            }
        }
    }

    Some(output)
}

/// Expand an already-resident two-layer chunk set (ADR 0010 E5 — the
/// [`TwoLayerResidentCache`] display path) into one recentred [`VoxelGrid`], in the
/// EXACT frame the retired dense `Store::resolve_region` assembled. Unlike
/// [`resolve_region_two_layer`] (which re-classifies each chunk from the scene via a
/// stateless [`TwoLayerStore`]), this reuses the caller's ALREADY-BUILT resident chunks.
///
/// `chunks` is `(absolute_chunk_coord, Arc<TwoLayerChunk>)` per covering chunk (the
/// [`TwoLayerResidentCache::resident_two_layer_chunks`] output); `region_dimensions` is
/// the composite voxel extent ([`Scene::placed_region_dimensions`]); `recentre`
/// is the composite recentre frame (ADR 0008). The occupied SET is bit-identical to
/// [`resolve_region_two_layer`]'s (the E2 round-trip parity gate proves the shared expand
/// path).
///
/// **ADR 0011 G5 — demoted to a TEST-ONLY oracle (`#[cfg(test)]`).** With the fog
/// `VoxelGrid` stream retired, no runtime path expands the resident set into a dense grid;
/// this survives only so parity tests (the brick-vs-densify fog nets, the render-frame
/// coordinate guard, the grid-overlay invalidation test) can materialise the same occupancy
/// the retired stream produced, and assert against it. A runtime caller reappearing here would
/// be the O(volume) densify coming back.
#[cfg(test)]
pub fn expand_resident_chunks_into_grid(
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    region_dimensions: [u32; 3],
    recentre: RecentreVoxels,
    voxels_per_block: u32,
) -> VoxelGrid {
    let mut output = VoxelGrid::new(region_dimensions);
    // Unwrap at the chunk-rebase arithmetic (the index offset below) and the grid's carried
    // raw frame field.
    let recentre_voxels = recentre.voxels();
    output.recentre_voxels = recentre_voxels;
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    for (chunk_coord, chunk) in chunks {
        // Rebase chunk-local indices into the recentred frame: a voxel at chunk-local
        // index `l` sits at absolute `chunk_min + l`, and the recentred frame subtracts
        // the recentre — the SAME `chunk_min − recentre` offset `resolve_region_two_layer`
        // applies (ADR 0008).
        let index_offset = [
            chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
            chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
            chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
        ];
        chunk.expand_occupancy_into(&mut output.occupied, index_offset);
    }
    output
}

// ===== ADR 0010 E4 — the cacheless STREAMING exact sinks =====================
//
// The display sink (E3) CACHES the two layers; the exact sinks (`.vox` export and
// the diameter/widest-run query) read the SAME evaluator region-scoped, cacheless,
// streaming (ADR 0010 Decision 3 — "many sinks by policy"). They drive the E2
// classifier ([`classify_chunk_block`]) block-by-block and NEVER assemble a dense
// whole-region `VoxelGrid`: a coarse-solid block is a fast `d³` fill (export) or an
// analytic run contribution (query: `run += d`, no per-voxel expansion); a boundary
// block is per-voxel field eval. This is why the `.vox` 6M whole-region cap
// dissolves on the export path — no dense interior is ever materialised.

/// Build ONE covering chunk in the recentred frame and stream its occupancy into a
/// FRESH `Vec<Voxel>` (coarse `d³` fast-fill + boundary per-voxel), in the EXACT
/// frame the dense [`Store::resolve_region`](crate::store::Store::resolve_region)
/// assembles. Returns `None` when the capability is OFF.
///
/// This is the per-chunk streaming primitive both exact sinks share: the export
/// buckets each chunk's `Vec` then DROPS it (never holding the whole region), and a
/// caller wanting the whole occupancy just concatenates. No dense whole-region grid
/// is ever allocated — the chunk buffer is the only transient, bounded to one
/// chunk's surface + (for export) its coarse interior.
fn stream_chunk_recentred(
    store: &TwoLayerStore,
    scene: &Scene,
    chunk_coord: [i32; 3],
    voxels_per_block: u32,
    recentre: RecentreVoxels,
) -> Option<Vec<Voxel>> {
    let chunk = store.build_chunk(chunk_coord, scene, voxels_per_block, 0)?;
    let chunk_extent_voxels = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    // Rebase chunk-local indices into the recentred frame (mirrors
    // `resolve_region_two_layer` / `resolve_chunk_rebased`): a chunk-local voxel `l`
    // has absolute index `chunk_min + l`, and the recentred frame subtracts the
    // recentre, so the offset is `chunk_min − recentre` (i64, before the i32 downcast).
    // Unwrap at this arithmetic.
    let recentre_voxels = recentre.voxels();
    let index_offset = [
        chunk_coord[0] as i64 * chunk_extent_voxels - recentre_voxels[0],
        chunk_coord[1] as i64 * chunk_extent_voxels - recentre_voxels[1],
        chunk_coord[2] as i64 * chunk_extent_voxels - recentre_voxels[2],
    ];
    let mut output = Vec::new();
    chunk.expand_occupancy_into(&mut output, index_offset);
    Some(output)
}

/// **Cacheless `.vox` streaming source (ADR 0010 E4).** Stream the scene's exact
/// occupancy region-scoped, ONE covering chunk at a time, in the recentred frame the
/// dense path produces, invoking `sink` with each chunk's freshly-expanded
/// `Vec<Voxel>` (coarse `d³` fast-fill + boundary per-voxel) before dropping it. The
/// caller's `sink` buckets each chunk into the `.vox` model set (so no whole-region
/// dense grid is ever assembled — the 6M whole-region cap dissolves on this path).
///
/// Returns the region's voxel `dimensions` (the SAME value
/// [`Scene::placed_region_dimensions`](crate::scene::Scene::placed_region_dimensions)
/// produces — the `.vox` tiling/decode frame) and the carried `recentre`. Returns
/// `None` when the capability is OFF (the caller falls back to the dense path).
pub fn stream_vox_occupancy<Sink: FnMut(Vec<Voxel>)>(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    mut sink: Sink,
) -> Option<([u32; 3], [i64; 3])> {
    if !store.is_enabled() {
        return None;
    }
    let region_dimensions = scene.placed_region_dimensions(voxels_per_block);
    // Carry the frame newtype through the per-chunk stream; unwrap only at the raw
    // `([u32; 3], [i64; 3])` return contract the `.vox` decode frame consumes.
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        // No composite extent (Part-only): an empty occupancy is still a valid export.
        return Some((region_dimensions, recentre.voxels()));
    };
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                if let Some(chunk_voxels) = stream_chunk_recentred(
                    store,
                    scene,
                    [chunk_x, chunk_y, chunk_z],
                    voxels_per_block,
                    recentre,
                ) {
                    sink(chunk_voxels);
                }
            }
        }
    }
    Some((region_dimensions, recentre.voxels()))
}

/// Insert the half-open run `[lo, hi)` into a row's sorted, disjoint, **non-touching**
/// interval list, coalescing with every interval it overlaps OR abuts (`end == lo` /
/// `start == hi` are adjacent — the corresponding dense bitset cells would be contiguous,
/// so they must fuse into one run). Keeps the list minimal, so the widest contiguous run
/// in the row is exactly `max(hi − lo)` over its intervals. In place; the dominant
/// ascending-arrival case (spans stream in increasing X within a row) hits the fast path
/// in O(1) and a solid row stays length 1.
fn insert_run(row: &mut Vec<(i64, i64)>, lo: i64, hi: i64) {
    // Fast path: append after, or extend, the last interval. Spans arrive in ascending X
    // within a row (chunk_x, block_x and the local indices all increase), so the coarse
    // solid sweep coalesces here with no shifting.
    if let Some(&mut (last_lo, ref mut last_hi)) = row.last_mut() {
        if lo > *last_hi {
            row.push((lo, hi)); // strictly right of the last, with a gap
            return;
        }
        if lo >= last_lo {
            if hi > *last_hi {
                *last_hi = hi; // overlaps / abuts the last, extends it right
            }
            return;
        }
    } else {
        row.push((lo, hi));
        return;
    }
    // General merge (rare: an out-of-order boundary cuboid starting left of the last run).
    let mut start = 0;
    while start < row.len() && row[start].1 < lo {
        start += 1; // skip intervals strictly left of the run (a real gap)
    }
    let mut merged_lo = lo;
    let mut merged_hi = hi;
    let mut end = start;
    while end < row.len() && row[end].0 <= merged_hi {
        merged_lo = merged_lo.min(row[end].0);
        merged_hi = merged_hi.max(row[end].1);
        end += 1;
    }
    row.splice(start..end, std::iter::once((merged_lo, merged_hi)));
}

/// **Cacheless diameter / widest-run query (ADR 0010 E4).** Compute the widest
/// occupied run in the layer band `[band_min, band_max]` (Z-slices, Z-up) by
/// streaming the classifier block-by-block — accounting a **coarse-solid block
/// ANALYTICALLY** (a fully-solid block sets a contiguous `density`-long X span in
/// every `(y, z)` row it covers, with NO per-voxel expansion) and a boundary block
/// per-voxel. Returns the SAME value
/// [`Store::widest_run_in_band`](crate::store::Store::widest_run_in_band) /
/// [`VoxelGrid::widest_run_in_band`](crate::voxel::VoxelGrid::widest_run_in_band)
/// returns for the assembled region, but never assembles a dense grid.
///
/// Returns `None` when the capability is OFF (the caller falls back to the dense
/// path).
///
/// ## Frame / decode (identical to the dense readout, ADR 0008)
///
/// The shared per-`(y, z)` occupancy rows are keyed by the GLOBAL X index the dense
/// [`VoxelGrid::widest_run_in_band`] computes
/// (`i = round(world_x + floor(grid_x/2) − 0.5)`). A coarse-solid block is stamped
/// at the SAME indices its per-voxel expansion would land — the recentred chunk-local
/// voxel index `chunk_min + block_low + local − recentre`, whose `world = index + 0.5`
/// decodes back to `index − region_low = index + floor(dim/2)` — so the analytic span
/// is bit-identical to expanding the block and scanning. A boundary block's per-voxel
/// fill uses the identical decode, so a run crossing a coarse↔boundary seam is one
/// contiguous span in the shared bitset.
pub fn streamed_widest_run_in_band(
    store: &TwoLayerStore,
    scene: &Scene,
    voxels_per_block: u32,
    band_min: u32,
    band_max: u32,
) -> Option<u32> {
    if !store.is_enabled() {
        return None;
    }
    let [grid_x, grid_y, grid_z] = scene.placed_region_dimensions(voxels_per_block);
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return Some(0);
    }
    // Unwrap the frame at this cacheless query's per-block rebase arithmetic.
    let recentre_voxels = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
    let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) else {
        return Some(0);
    };

    let width = grid_x as usize;
    // The dense decode: `idx = round(world + floor(dim/2) − 0.5)`. Because every
    // streamed voxel index `n` decodes `world = n + 0.5` ⇒ `idx = n + floor(dim/2)`,
    // the recentred index maps to the global grid index by ADDING `floor(dim/2)`.
    let half = [
        (grid_x / 2) as i64,
        (grid_y / 2) as i64,
        (grid_z / 2) as i64,
    ];
    let density = voxels_per_block.max(1) as i64;
    let chunk_extent_voxels = CHUNK_BLOCKS as i64 * density;
    let grid_y_i = grid_y as i64;
    let band_min_i = band_min as i64;
    let band_max_i = band_max as i64;

    // ADR 0010 E5 — BLOCK-ROW DEDUP (the O(volume)→O(total blocks) diameter fix).
    //
    // The widest run is folded one **(chunk_z, chunk_y) band at a time**: every global
    // (z, y) row maps to EXACTLY ONE band (the recentred z/y ranges partition disjointly),
    // so a band is complete once its `chunk_x` sweep finishes. Bands are independent, so the
    // fold runs across them in PARALLEL with a deterministic `max` reduction — live state is
    // bounded per band, never a region-wide bitset (the prior `Vec<bool>` per (z,y) row was
    // O(volume): a 8000×800×800 solid was ≈5 GB and OOM-hung startup).
    //
    // The fatal remaining cost was per-VOXEL-row work WITHIN a band: a coarse-solid block
    // stamped its `[x0, x0+d)` span into every one of the `d²` voxel rows it covers, so a
    // solid cube was O(voxels) (edge² rows × the blocks per row) — 64M rows at 8000³, a 127s
    // main-thread freeze. THE DEDUP: every `d²` voxel row under the SAME BLOCK-ROW (same
    // block_y, block_z) receives IDENTICAL coarse spans, so each block-row's coarse span set
    // is accumulated ONCE — block-granular (one `[x0, x0+d)` per coarse block, ×density
    // widths) — and its widest contiguous run is counted once for the whole block-row (valid
    // iff its z-range meets the band and its y-range meets `[0, grid_y)`, exactly the old
    // per-voxel band/grid clip). Per-voxel work survives ONLY for the voxel rows a BOUNDARY
    // (microblock) block actually intersects: those are expanded per voxel and then MERGED
    // with their block-row's coarse spans (so a run crossing a coarse↔boundary seam is one
    // contiguous span). A solid cube is then O(total blocks) — the chunk-enumeration floor —
    // not O(voxels).
    //
    // Each accumulator holds occupied X as a sorted, disjoint, non-touching INTERVAL list
    // (half-open `[lo, hi)`); the widest contiguous run in a row is `max(hi − lo)`.
    // Byte-identical to the dense `widest_run_in_band` oracle (the
    // `streamed_widest_run_matches_dense_*` parity gate) — the same global X indices,
    // stitched the same way across seams, folded to the same max.
    const BLOCK_ROWS: usize = CHUNK_BLOCKS as usize * CHUNK_BLOCKS as usize;

    let mut bands: Vec<(i32, i32)> = Vec::new();
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            bands.push((chunk_z, chunk_y));
        }
    }

    // The widest occupied run within ONE (chunk_z, chunk_y) band. Pure over the shared
    // immutable inputs (store/scene/frame scalars), so it parallelises across bands.
    let band_widest = |(chunk_z, chunk_y): (i32, i32)| -> u32 {
        // The band's global-row origins (`to_global[1..=2]`, constant over `chunk_x`).
        let z_base = chunk_z as i64 * chunk_extent_voxels - recentre_voxels[2] + half[2];
        let y_base = chunk_y as i64 * chunk_extent_voxels - recentre_voxels[1] + half[1];

        // Per block-row (index `block_z * CHUNK_BLOCKS + block_y`) coarse X spans — the ONE
        // span set shared by every voxel row the block-row covers. Boundary cuboid spans are
        // expanded per voxel row, keyed by their global `(z, y)` (sparse — surface rows only),
        // and merged with the block-row's coarse spans at fold time.
        let mut coarse_block_runs: [Vec<(i64, i64)>; BLOCK_ROWS] =
            std::array::from_fn(|_| Vec::new());
        let mut boundary_rows: std::collections::HashMap<(i64, i64), Vec<(i64, i64)>> =
            std::collections::HashMap::new();

        for chunk_x in min_chunk[0]..=max_chunk[0] {
            let chunk_coord = [chunk_x, chunk_y, chunk_z];
            let Some(chunk) = store.build_chunk(chunk_coord, scene, voxels_per_block, 0) else {
                continue;
            };
            // The recentred→global X origin: global_x = chunk_min_x + block_low_x + local
            // − recentre + half. Spans arrive in ascending X (chunk_x, block_x, local all
            // increase), so `insert_run`'s append fast path coalesces a solid row in O(1).
            let to_global_x = chunk_x as i64 * chunk_extent_voxels - recentre_voxels[0] + half[0];
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    let block_row =
                        block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let block_low_x = block_x as i64 * density;
                        if chunk.coarse_block(block).is_some() {
                            // ANALYTIC / block-granular: the contiguous span `[x0, x0+d)` is
                            // identical for every voxel row this coarse block covers — recorded
                            // ONCE for the whole block-row (no per-voxel expansion).
                            let x0 = to_global_x + block_low_x;
                            let lo = x0.max(0);
                            let hi = (x0 + density).min(width as i64);
                            if lo < hi {
                                insert_run(&mut coarse_block_runs[block_row], lo, hi);
                            }
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            // BOUNDARY: per-cuboid spans expanded into every voxel row they
                            // cover, band/grid/width-clipped exactly as the dense readout.
                            for cuboid in &geometry.cuboids {
                                for local_z in cuboid.min[2]..=cuboid.max[2] {
                                    let gz = z_base + block_z as i64 * density + local_z as i64;
                                    if gz < band_min_i || gz > band_max_i {
                                        continue;
                                    }
                                    for local_y in cuboid.min[1]..=cuboid.max[1] {
                                        let gy =
                                            y_base + block_y as i64 * density + local_y as i64;
                                        if gy < 0 || gy >= grid_y_i {
                                            continue;
                                        }
                                        let x_lo = (to_global_x + block_low_x
                                            + cuboid.min[0] as i64)
                                            .max(0);
                                        let x_hi = (to_global_x
                                            + block_low_x
                                            + cuboid.max[0] as i64
                                            + 1)
                                            .min(width as i64);
                                        if x_lo >= x_hi {
                                            continue;
                                        }
                                        insert_run(
                                            boundary_rows.entry((gz, gy)).or_default(),
                                            x_lo,
                                            x_hi,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut widest = 0u32;
        // (1) All-coarse contribution: each block-row with coarse runs AND ≥1 valid voxel row
        // (its z-range meets the band, its y-range meets `[0, grid_y)`) carries that widest
        // coarse run in every one of its voxel rows.
        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                let block_row = block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
                let runs = &coarse_block_runs[block_row];
                if runs.is_empty() {
                    continue;
                }
                let z_lo = z_base + block_z as i64 * density;
                if z_lo + density - 1 < band_min_i || z_lo > band_max_i {
                    continue;
                }
                let y_lo = y_base + block_y as i64 * density;
                if y_lo + density - 1 < 0 || y_lo >= grid_y_i {
                    continue;
                }
                for &(lo, hi) in runs {
                    widest = widest.max((hi - lo) as u32);
                }
            }
        }
        // (2) Boundary rows: merge each with its block-row's coarse spans, fold the widest —
        // a run crossing a coarse↔boundary seam is one contiguous span. The global (z, y) key
        // recovers its block-row: `block_z = (gz − z_base) / d`, `block_y = (gy − y_base) / d`.
        for (&(gz, gy), spans) in &boundary_rows {
            let block_z = (gz - z_base) / density;
            let block_y = (gy - y_base) / density;
            let block_row = block_z as usize * CHUNK_BLOCKS as usize + block_y as usize;
            let mut merged = coarse_block_runs[block_row].clone();
            for &(lo, hi) in spans {
                insert_run(&mut merged, lo, hi);
            }
            for &(lo, hi) in &merged {
                widest = widest.max((hi - lo) as u32);
            }
        }
        widest
    };

    let widest = bands.into_par_iter().map(band_widest).max().unwrap_or(0);
    Some(widest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::{DefId, Node, NodeContent, NodeTransform};
    use crate::voxel::{GeometryParams, SdfShape, ShapeKind};

    /// Canonicalise an occupied set into the **resolved occupancy SET**: a map from each
    /// bit-exact voxel position to the block id of the LAST (document-order) writer at that
    /// position. This is the ADR 0010 parity-gate canonical form (the resolved occupancy
    /// SET keyed by position+block_id) — it differs from the dense store's
    /// `cache_region_matches_monolithic_*` MULTISET only at positions where leaves overlap.
    ///
    /// The two-layer store is a one-id-per-cell representation (a boundary block resolves
    /// to a dense region where the later leaf overwrites the earlier — Union "later wins"),
    /// so it never carries the dense path's DUPLICATE Vec entries at a shared position. The
    /// dense `Scene::resolve_region` emits leaves in document order, so the LAST entry at a
    /// position is the winner there too — taking the last writer on BOTH sides compares the
    /// true resolved occupancy. For every non-overlapping scene (all the SDF-shape /
    /// flat-odd cases) each position has exactly one writer, so this is byte-identical to
    /// the dense multiset; only genuinely-overlapping leaves (cloud-over-box) differ, and
    /// there the resolved-set is the correct comparison.
    ///
    /// Keying on the raw `f32` bits (`to_bits`) asserts the BYTES a consumer reads are
    /// identical, not merely the rounded voxel set.
    fn resolved_occupancy_set(
        grid: &VoxelGrid,
    ) -> std::collections::BTreeMap<[u32; 3], u16> {
        let mut set = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
            ];
            // Last document-order writer wins (Union later-wins material).
            set.insert(key, voxel.color_index());
        }
        set
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// THE GATE (parity (a)): the two-layer round-trip occupancy (coarse fast-fill +
    /// boundary per-voxel) is BIT-IDENTICAL (position + block id) to the dense
    /// `Scene::resolve_region`, for the gated scene. Mirrors
    /// `store.rs::cache_region_matches_monolithic_*`. Returns the chunk + cell counts the
    /// build classified (so the harness can report coverage).
    fn assert_two_layer_round_trip_matches_dense(
        scene: &Scene,
        voxels_per_block: u32,
        label: &str,
    ) -> (usize, u64) {
        let dense = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let store = TwoLayerStore::enabled();
        let assembled = resolve_region_two_layer(&store, scene, voxels_per_block, 0)
            .expect("the capability is enabled");

        assert_eq!(
            assembled.dimensions, dense.dimensions,
            "[{label}] two-layer round-trip dimensions must match dense resolve_region"
        );
        assert_eq!(
            assembled.recentre_voxels, dense.recentre_voxels,
            "[{label}] two-layer round-trip must carry the SAME recentre as dense"
        );
        let dense_set = resolved_occupancy_set(&dense);
        let assembled_set = resolved_occupancy_set(&assembled);
        assert_eq!(
            assembled_set.len(),
            dense_set.len(),
            "[{label}] two-layer resolved occupancy count must match dense (the dense Vec \
             may hold duplicate entries at overlap positions; the resolved SET must agree)"
        );
        assert_eq!(
            assembled_set, dense_set,
            "[{label}] two-layer round-trip resolved occupancy SET (position + block id, \
             last-writer-wins) must be BIT-IDENTICAL to the dense resolve_region"
        );

        // Coverage accounting: count chunks + blocks the build classified.
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .unwrap_or(([0; 3], [-1; 3]));
        let chunks = if max_chunk[0] < min_chunk[0] {
            0
        } else {
            ((max_chunk[0] - min_chunk[0] + 1)
                * (max_chunk[1] - min_chunk[1] + 1)
                * (max_chunk[2] - min_chunk[2] + 1)) as usize
        };
        let cells = chunks as u64 * (CHUNK_BLOCKS as u64).pow(3);
        (chunks, cells)
    }

    #[test]
    fn round_trip_matches_dense_for_all_shapes() {
        let mut total_chunks = 0usize;
        let mut total_cells = 0u64;
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            let (chunks, cells) =
                assert_two_layer_round_trip_matches_dense(&scene, 16, &format!("{kind:?}"));
            total_chunks += chunks;
            total_cells += cells;
        }
        eprintln!(
            "two-layer parity (all shapes): {total_chunks} chunks, {total_cells} block cells"
        );
    }

    /// FLAT / odd-sized shapes — the S0 covering-range regression case (a 1-block axis
    /// straddles two chunks). The classifier must cover the producer-true voxel extent
    /// and round-trip bit-identically, just as the dense net pins.
    #[test]
    fn round_trip_matches_dense_for_flat_and_odd_shapes() {
        let mut total_cells = 0u64;
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: kind,
                        size_voxels: [size[0] * 16, size[1] * 16, size[2] * 16],
                        size_measurements: None,
                        voxels_per_block: 16,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                let (_chunks, cells) = assert_two_layer_round_trip_matches_dense(
                    &scene,
                    16,
                    &format!("{kind:?} {size:?}"),
                );
                total_cells += cells;
            }
        }
        eprintln!("two-layer parity (flat/odd): {total_cells} block cells");
    }

    fn make_tool(kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    #[test]
    fn round_trip_matches_dense_for_demo_scene() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, density),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-scene");
    }

    #[test]
    fn round_trip_matches_dense_for_demo_village() {
        let density = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, density);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, density);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, density);
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
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-village");
    }

    /// A sketch-revolve solid (the 800×800-revolve CLASS that stressed the dense cap): the
    /// interior now ELIDES to coarse-solid blocks (ADR 0010 rollout) while the round-trip
    /// stays bit-identical to the dense store — pinning the coarse + boundary composition
    /// exact.
    #[test]
    fn round_trip_matches_dense_for_sketch_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let density = 16;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve");
    }

    /// A PARTIAL-turn revolve with an AXIS-STRADDLING profile (radial spans negative→positive,
    /// so the resolve's mirrored `−radius` union is live) — the ADR 0010 partial-sweep coarse
    /// test must round-trip bit-identically to the dense oracle: interior blocks inside the
    /// swept arc elide to coarse, the excluded wedge stays boundary/air, and the mirrored
    /// occupancy is reproduced exactly.
    #[test]
    fn round_trip_matches_dense_for_partial_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
        let density = 16;
        // Radial (c1) straddles the axis: [-20, 20]; axial (c0) [8, 56].
        let profile = Sketch::new(
            PlaneAxis::Z,
            vec![
                SketchPoint::new(8, -20),
                SketchPoint::new(56, -20),
                SketchPoint::new(56, 20),
                SketchPoint::new(8, 20),
            ],
        );
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 135);
        let node = Node::new(
            "PartialRevolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-partial-revolve");
    }

    /// A region-spanning UNBOUNDABLE producer (the fBm cloud field) forces every covering
    /// block BOUNDARY (its `cell_field_interval` is `None`) and STILL round-trips
    /// bit-identically — the "unboundable ops fall back, still exact" acceptance criterion.
    /// (Mixed with a Tool so the scene has a composite chunk extent.)
    #[test]
    fn round_trip_matches_dense_with_unboundable_cloud() {
        use crate::scene::Part;
        let density = 16;
        let mut cloud = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 7 }));
        cloud.transform = NodeTransform::from_blocks([0, 0, 0], density);
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density),
            cloud,
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "tool+cloud");
    }

    /// INTERIOR ELISION (the whole point): a LARGE solid box stores ZERO interior voxels
    /// under the two-layer path — only its surface shell lives in the microblock layer,
    /// while its interior is coarse block ids. The dense path would densify all
    /// ~`(size·d)³` interior voxels (and a revolve-class size blows the 6M cap); the
    /// two-layer stored count is surface-only.
    #[test]
    fn large_solid_box_stores_zero_interior_voxels() {
        let density = 16;
        // 50×50×50 BLOCKS @ d16 = 800×800×800 voxels — the revolve-class size the ADR
        // calls out. Dense interior would be 800³ ≈ 5.1e8 voxels (far past the 6M cap);
        // the two-layer interior holds NONE.
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, density);
        let node = Node::new(
            "BigBox",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let store = TwoLayerStore::enabled();

        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(density)
            .expect("a placed box has a covering chunk range");

        let mut total_stored = 0u64;
        let mut interior_chunks = 0u64;
        let mut total_chunks = 0u64;
        // An interior chunk (no block of it touches a face of the box) is entirely
        // coarse-solid, so it must store ZERO voxels.
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = store
                        .build_chunk([chunk_x, chunk_y, chunk_z], &scene, density, 0)
                        .unwrap();
                    let stored = chunk.stored_voxel_count();
                    total_stored += stored;
                    total_chunks += 1;
                    let is_interior_chunk = chunk_x > min_chunk[0]
                        && chunk_x < max_chunk[0]
                        && chunk_y > min_chunk[1]
                        && chunk_y < max_chunk[1]
                        && chunk_z > min_chunk[2]
                        && chunk_z < max_chunk[2];
                    if is_interior_chunk {
                        interior_chunks += 1;
                        assert_eq!(
                            stored, 0,
                            "interior chunk ({chunk_x},{chunk_y},{chunk_z}) of a solid box \
                             must store ZERO voxels (interior elision), got {stored}"
                        );
                    }
                }
            }
        }
        let dense_interior_voxels = (blocks as u64 * density as u64).pow(3);
        assert!(
            interior_chunks > 0,
            "the box must be large enough to have fully-interior chunks"
        );
        // The stored voxels are the 1-block-thick SURFACE SHELL only (each surface block
        // is d³ = 4096 voxels at d16, so a 50²-face shell is legitimately ~12% of the
        // volume) — a fraction of the dense interior, and every FULLY-interior chunk
        // (asserted above) holds ZERO. The dense path would densify the whole volume and
        // blow the 6M cap; the two-layer path never builds the interior.
        assert!(
            total_stored < dense_interior_voxels / 4,
            "two-layer stored voxels ({total_stored}) must be well below the dense interior \
             volume ({dense_interior_voxels}) — surface-shell-only residency"
        );
        eprintln!(
            "interior elision: {total_chunks} chunks ({interior_chunks} fully interior); \
             two-layer stored {total_stored} voxels vs dense interior {dense_interior_voxels}"
        );
    }

    /// INTERIOR ELISION for the SKETCH producer — the completion of the ADR 0010 rollout.
    /// A SOLID extrude box and a full 360° revolve now classify their interiors
    /// COARSE-SOLID (dominating the surface-only boundary shell), and a CONCAVE L extrude
    /// elides its interior while keeping the reflex-corner block BOUNDARY and the removed
    /// quadrant AIR (proving the polygon test, not just axis-aligned rectangles). Every
    /// case also round-trips bit-identically to the dense oracle (the over-claim police).
    #[test]
    fn sketch_interior_elides_to_coarse_solid() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
        let density = 8u32;

        // Count (coarse, boundary) blocks across a producer's covering chunk range by
        // classifying every block directly (no per-voxel resolve → fast).
        let classify_scene = |scene: &Scene| -> (u64, u64) {
            let leaves = scene.leaf_producers(density);
            let leaves: Vec<&LeafProducer> = leaves.iter().collect();
            let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
            let chunk_extent = (CHUNK_BLOCKS * density) as i64;
            let block = density as i64;
            let (mut coarse, mut boundary) = (0u64, 0u64);
            for cz in min_chunk[2]..=max_chunk[2] {
                for cy in min_chunk[1]..=max_chunk[1] {
                    for cx in min_chunk[0]..=max_chunk[0] {
                        for bz in 0..CHUNK_BLOCKS {
                            for by in 0..CHUNK_BLOCKS {
                                for bx in 0..CHUNK_BLOCKS {
                                    let low = [
                                        cx as i64 * chunk_extent + bx as i64 * block,
                                        cy as i64 * chunk_extent + by as i64 * block,
                                        cz as i64 * chunk_extent + bz as i64 * block,
                                    ];
                                    let cell = VoxelAabb::new(
                                        low,
                                        [low[0] + block, low[1] + block, low[2] + block],
                                    );
                                    match classify_chunk_block(&leaves, cell, density) {
                                        BlockClassification::CoarseSolid(_) => coarse += 1,
                                        BlockClassification::Boundary => boundary += 1,
                                        BlockClassification::Air => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (coarse, boundary)
        };

        // (1) SOLID extrude box, 8 blocks per axis (64³ voxels), BLOCK-ALIGNED: every block
        // is fully solid (the axis-aligned wall blocks too — their face lattice line is
        // collinear with the profile edge but every voxel centre is inside), so the whole
        // box is COARSE with ZERO boundary blocks (the sample-centre rectangle win).
        let edge = 8 * density as i64;
        let extrude =
            SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
        )]);
        let (coarse, boundary) = classify_scene(&scene);
        assert_eq!(
            boundary, 0,
            "a block-aligned solid box has NO boundary blocks (walls are fully solid ⇒ coarse)"
        );
        assert_eq!(
            coarse,
            (CHUNK_BLOCKS as u64 * 2).pow(3),
            "every block of the 8-block-per-axis box must be coarse-solid, got {coarse}"
        );
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-extrude-box");

        // (2) FULL 360° revolve (a solid cylinder, radial 3 blocks × axial 4 blocks):
        // interior near the axis elides to coarse.
        let revolve = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 3 * density as i64, 4 * density as i64),
            RevolveAxis::InPlane1,
            360,
        );
        let scene = Scene::from_nodes(vec![Node::new(
            "Cyl",
            NodeContent::SketchTool { producer: revolve, material: MaterialChoice::Stone },
        )]);
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "full 360 revolve must elide interior blocks to coarse-solid");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve-cyl");

        // (3) CONCAVE L extrude (notch corner at voxel 20 = mid-block at d8, so the reflex
        // edges CUT a block): interior elides, the reflex-corner block stays boundary, and
        // the removed quadrant is NOT coarse (a plain rectangle would over-claim it solid).
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(32, 0),
            SketchPoint::new(32, 20),
            SketchPoint::new(20, 20), // reflex vertex, mid-block
            SketchPoint::new(20, 32),
            SketchPoint::new(0, 32),
        ];
        let l = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 24);
        let scene = Scene::from_nodes(vec![Node::new(
            "L",
            NodeContent::SketchTool { producer: l, material: MaterialChoice::Wood },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        // Deep inside the bottom bar (not touching any face) ⇒ coarse.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([8, 8, 8], [16, 16, 16]), density),
            BlockClassification::CoarseSolid(MaterialChoice::Wood.block_id()),
            "an interior L block must be coarse-solid"
        );
        // The block the reflex edges cut through ([16,24)² in-plane, spanning y=20 & x=20)
        // ⇒ boundary (a coarse claim would over-fill the notch).
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([16, 16, 8], [24, 24, 16]), density),
            BlockClassification::Boundary,
            "the L reflex-corner block must stay boundary"
        );
        // The removed top-right quadrant ([24,32)² in-plane) overlaps the producer AABB so
        // it classifies BOUNDARY (resolves per-voxel to EMPTY) — crucially NOT coarse-solid,
        // which is exactly what a naive bbox-solid claim would have wrongly returned.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([24, 24, 8], [32, 32, 16]), density),
            BlockClassification::Boundary,
            "the removed L quadrant must NOT be coarse-solid (the polygon excludes it)"
        );
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "the L extrude must still elide its solid interior");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-L-extrude");

        // (4) NON-BLOCK-ALIGNED interior edge: a right-triangle profile whose hypotenuse
        // (x + y = 24) cuts through block INTERIORS at d8. A block the hypotenuse crosses
        // stays BOUNDARY; a block fully below it goes coarse — proving the sample-centre
        // test still distinguishes true-boundary blocks from fully-solid axis-aligned walls.
        let triangle = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(24, 0),
            SketchPoint::new(0, 24),
        ];
        let tri = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, triangle), 24);
        let scene = Scene::from_nodes(vec![Node::new(
            "Tri",
            NodeContent::SketchTool { producer: tri, material: MaterialChoice::Stone },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        // Fully below the hypotenuse (max x+y = 15 < 24) ⇒ coarse.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([0, 0, 8], [8, 8, 16]), density),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "a block fully below the triangle hypotenuse must be coarse-solid"
        );
        // The hypotenuse passes through this block's interior ⇒ boundary (not coarse).
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([8, 8, 8], [16, 16, 16]), density),
            BlockClassification::Boundary,
            "a block the hypotenuse cuts through the interior of must stay boundary"
        );
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "the triangle extrude must still elide its solid interior");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-triangle-extrude");

        // (5) PARTIAL 270° revolve (a fat cylinder WEDGE, radial 6 blocks × axial 4 blocks):
        // closing the ADR 0010 deferral — a partial sweep now elides its interior via the
        // angular-containment coarse test. Before this fix `revolve_cell_is_solid` returned
        // false for every partial-turn cell, so a wedge densified its WHOLE interior (0 coarse
        // blocks); now interior blocks fully inside the [0°, 270°] arc AND the radial/axial
        // profile classify coarse-solid, while the excluded fourth quadrant (270°–360°) stays
        // boundary/air. The round-trip stays bit-identical to the dense oracle.
        let wedge = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 6 * density as i64, 4 * density as i64),
            RevolveAxis::InPlane1,
            270,
        );
        let scene = Scene::from_nodes(vec![Node::new(
            "Wedge",
            NodeContent::SketchTool { producer: wedge, material: MaterialChoice::Stone },
        )]);
        let (coarse, boundary) = classify_scene(&scene);
        assert!(
            coarse > 0,
            "a PARTIAL 270° revolve wedge must now elide interior blocks to coarse-solid \
             (the ADR 0010 partial-sweep deferral is closed), got {coarse} coarse / {boundary} boundary"
        );
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve-wedge");
    }

    /// A fully-interior block of a solid box classifies COARSE-SOLID (no voxels); a block
    /// straddling the box face classifies BOUNDARY; a block well outside classifies AIR.
    #[test]
    fn classifier_sorts_air_coarse_and_boundary() {
        let density = 8u32;
        // A 5×5×5-block box at the origin → voxel extent [0, 40) per axis.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = density as i64;

        // A deep-interior block ([16,24) on each axis, well inside [0,40)) is coarse-solid.
        let interior = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, interior, density),
            BlockClassification::CoarseSolid(MaterialChoice::Wood.block_id()),
            "a deep-interior block of a solid box must be coarse-solid at its material"
        );

        // A block straddling the +X face (the box ends at voxel 40; block [40−4,40+4)).
        let straddle = VoxelAabb::new([36, 16, 16], [36 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, straddle, density),
            BlockClassification::Boundary,
            "a block straddling the box surface must be boundary"
        );

        // A block far outside the box ([200, 208) on X) is air.
        let outside = VoxelAabb::new([200, 16, 16], [200 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, outside, density),
            BlockClassification::Air,
            "a block well outside every leaf must be air"
        );
    }

    /// Sculpt-touched / multi-leaf-overlap conservatism: a block where TWO Tools overlap
    /// is forced BOUNDARY (the Union's per-voxel later-wins material is not coarsely
    /// decidable), even if geometrically solid — still exact after per-voxel.
    #[test]
    fn overlapping_leaves_force_boundary() {
        let density = 8u32;
        // Two boxes overlapping at the origin region, different materials.
        let scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 5),
            make_tool_density(ShapeKind::Box, [1, 0, 0], MaterialChoice::Wood, density, 5),
        ]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = density as i64;
        // A block in the overlap region of both boxes.
        let overlap = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, overlap, density),
            BlockClassification::Boundary,
            "a block two leaves both fill must be boundary (per-voxel material resolution)"
        );
    }

    /// **CHUNK-GRANULAR FAST-PATH BYTE-IDENTITY (ADR 0010 Decision 2).** Over every covering
    /// chunk of a battery of mixed scenes, the whole-chunk interval fast path
    /// ([`build_two_layer_chunk_from_leaves`]) produces a `TwoLayerChunk` BYTE-IDENTICAL to
    /// the forced per-block sweep ([`build_two_layer_chunk_per_block`]) — coarse layer +
    /// overlay + microblock maps + seam flags. This pins the fast path's
    /// CONSERVATIVE-NEVER-NARROW contract directly (the round-trip-vs-dense gates check
    /// occupancy; this checks the exact two-layer STRUCTURE the fast path claims).
    ///
    /// The scenes exercise every fast-path arm: solid interiors (whole-chunk COARSE), the
    /// surface shell + concave/diagonal profiles (whole-chunk BOUNDARY → per-block),
    /// multi-leaf overlaps with DIFFERENT materials (uniformity guard forces per-block),
    /// `DebugClouds` (unboundable → per-block), and a partial revolve (angular ambiguity).
    #[test]
    fn whole_chunk_fast_path_matches_per_block_sweep() {
        use crate::scene::Part;
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};

        // Assert fast-path == per-block over EVERY covering chunk of `scene`.
        fn assert_identical(scene: &Scene, density: u32, label: &str) {
            let leaves = scene.leaf_producers(density);
            let leaves: Vec<&LeafProducer> = leaves.iter().collect();
            let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(density) else {
                return;
            };
            let chunk_extent = (CHUNK_BLOCKS * density) as i64;
            let mut coarse_chunks = 0u64;
            for cz in min_chunk[2]..=max_chunk[2] {
                for cy in min_chunk[1]..=max_chunk[1] {
                    for cx in min_chunk[0]..=max_chunk[0] {
                        let coord = [cx, cy, cz];
                        let fast = build_two_layer_chunk_from_leaves(coord, &leaves, density);
                        let chunk_min = [
                            cx as i64 * chunk_extent,
                            cy as i64 * chunk_extent,
                            cz as i64 * chunk_extent,
                        ];
                        let per_block =
                            build_two_layer_chunk_per_block(chunk_min, &leaves, density, density);
                        assert_eq!(
                            fast, per_block,
                            "[{label}] chunk {coord:?}: fast-path classification must be \
                             BYTE-IDENTICAL to the per-block sweep"
                        );
                        if fast.coarse.iter().all(Option::is_some) && !fast.coarse.is_empty() {
                            coarse_chunks += 1;
                        }
                    }
                }
            }
            eprintln!("[{label}] fast-path==per-block over all chunks ({coarse_chunks} all-coarse)");
        }

        let density = 8u32;

        // (a) SOLID sketch-extrude box — the whole-CHUNK-COARSE perf target (interior chunks
        // resolve in ONE interval call). Block-aligned so interiors AND walls are coarse.
        let edge = 8 * density as i64;
        let box_scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32),
                material: MaterialChoice::Stone,
            },
        )]);
        assert_identical(&box_scene, density, "sketch-extrude-box");

        // (b) SDF shapes — curved surfaces give a real boundary shell + coarse interiors,
        // and exercise the Lipschitz-centre interval's inclusion-monotonicity.
        for kind in [ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Cylinder, ShapeKind::Torus] {
            let scene = Scene::from_nodes(vec![make_tool_density(
                kind,
                [0, 0, 0],
                MaterialChoice::Stone,
                density,
                6,
            )]);
            assert_identical(&scene, density, &format!("sdf-{kind:?}"));
        }

        // (c) MULTI-LEAF overlap, DIFFERENT materials — the uniformity guard must force the
        // overlap chunks to per-block (Union later-wins material is not coarsely decidable).
        let overlap_scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 6),
            make_tool_density(ShapeKind::Box, [3, 0, 0], MaterialChoice::Wood, density, 6),
        ]);
        assert_identical(&overlap_scene, density, "multi-leaf-materials");

        // (d) DebugClouds — unboundable (`cell_field_interval == None`) ⇒ the whole chunk
        // falls back to per-block; every chunk must still match.
        let cloud_scene = Scene::from_nodes(vec![Node::new(
            "Clouds",
            NodeContent::Part(Part::DebugClouds { seed: 7 }),
        )]);
        assert_identical(&cloud_scene, density, "debug-clouds");

        // (e) CONCAVE L extrude — reflex-corner + removed quadrant keep boundary/air chunks
        // adjacent to coarse interiors.
        let l_scene = Scene::from_nodes(vec![Node::new(
            "L",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(
                    Sketch::new(
                        PlaneAxis::Z,
                        vec![
                            SketchPoint::new(0, 0),
                            SketchPoint::new(32, 0),
                            SketchPoint::new(32, 20),
                            SketchPoint::new(20, 20),
                            SketchPoint::new(20, 32),
                            SketchPoint::new(0, 32),
                        ],
                    ),
                    24,
                ),
                material: MaterialChoice::Wood,
            },
        )]);
        assert_identical(&l_scene, density, "sketch-L-extrude");

        // (f) PARTIAL 270° revolve — angular ambiguity keeps the excluded wedge boundary/air
        // while the swept interior elides to coarse.
        let wedge_scene = Scene::from_nodes(vec![Node::new(
            "Wedge",
            NodeContent::SketchTool {
                producer: SketchSolid::revolve(
                    Sketch::rectangle(PlaneAxis::X, 6 * density as i64, 4 * density as i64),
                    RevolveAxis::InPlane1,
                    270,
                ),
                material: MaterialChoice::Stone,
            },
        )]);
        assert_identical(&wedge_scene, density, "sketch-revolve-wedge");
    }

    /// **#66 edit-broadphase exactness (belt-and-braces, the #63 gate carried over).** The
    /// per-chunk candidate set the BVH ([`leaf_edit_broadphase`]) hands each chunk MUST
    /// equal the naive "all leaves filtered by AABB-overlaps-chunk" set — leaf-index-
    /// identical, in document order. If they ever diverge, a chunk could be classified
    /// against the wrong candidate set and the two-layer output would drift from the dense
    /// path (which the parity gate would catch, but this pins the invariant directly at the
    /// broadphase boundary).
    #[test]
    fn broadphase_candidate_set_equals_naive_filter() {
        let density = 8u32;
        // A 4×4×4 grid of small boxes spaced 3 blocks apart — leaves land in many chunks,
        // some sharing a chunk (adjacency), so the candidate sets are non-trivial.
        let mut nodes = Vec::new();
        for grid_z in 0..4i64 {
            for grid_y in 0..4i64 {
                for grid_x in 0..4i64 {
                    nodes.push(make_tool_density(
                        ShapeKind::Box,
                        [grid_x * 3, grid_y * 3, grid_z * 3],
                        MaterialChoice::Stone,
                        density,
                        2,
                    ));
                }
            }
        }
        let scene = Scene::from_nodes(nodes);
        let leaves = scene.leaf_producers(density);
        let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
        let broadphase = leaf_edit_broadphase(&leaves, density);

        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    let chunk_box = chunk_world_voxel_aabb(coord, density);
                    // Naive: every leaf whose world AABB overlaps this chunk's box, in
                    // document order (a filter — never a reorder).
                    let naive: Vec<usize> = leaves
                        .iter()
                        .enumerate()
                        .filter(|(_, leaf)| {
                            leaf_world_aabb(leaf, density).intersects(&chunk_box)
                        })
                        .map(|(index, _)| index)
                        .collect();
                    assert_eq!(
                        broadphase.overlapping_input_indices(&chunk_box),
                        naive,
                        "edit-broadphase candidates for chunk {coord:?} must equal the \
                         naive all-leaves-filtered set, in document order"
                    );
                }
            }
        }
    }

    fn make_tool_density(
        kind: ShapeKind,
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
        size_blocks: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, [size_blocks; 3], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// SEAM-SOLIDITY flags: a boundary block's per-face flag matches its ACTUAL face
    /// occupancy. We resolve a boundary block of a solid box and assert the face that lies
    /// INSIDE the box is solid while the face that pokes OUT of the box is not.
    #[test]
    fn seam_solidity_flags_match_face_occupancy() {
        let density = 8u32;
        // A solid box [0,40) per axis. Take the block straddling the +X face: block
        // [32,40) on X (the last fully-inside block column is [32,40); the face at X=39
        // is the box's last solid layer, and X=40+ is air). To get a STRADDLING block on
        // a different axis we instead take a block at the +X edge whose low-X face is
        // solid (inside the box) and whose geometry is the surface shell.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        // The block [32,40) on X, interior on Y/Z ([16,24)). The box fills X∈[0,40), so
        // this whole block is solid — BUT it touches the +X face of the box, so its
        // classification depends on the conservative bound. Resolve it directly to read
        // its seam flags regardless of the coarse verdict.
        let block_min = [32i64, 16, 16];
        let geometry = resolve_boundary_block(&leaves, block_min, density, density);
        // The low-X face (X=32, the 0th local layer) is deep inside the box ⇒ fully solid.
        assert!(
            geometry.seam_solidity.face_is_solid(0, 0),
            "the low-X face of an interior-touching block must be solid"
        );
        // Every face of this all-solid block is solid (the box fully covers [32,40)³ here,
        // since Y,Z ∈ [16,24) ⊂ [0,40) and X ∈ [32,40) ⊂ [0,40)).
        for axis in 0..3 {
            for side in 0..2 {
                assert!(
                    geometry.seam_solidity.face_is_solid(axis, side),
                    "a fully-solid block must report every face solid (axis {axis}, side {side})"
                );
            }
        }

        // A block straddling the +X surface (X∈[36,44), so X∈[40,44) is OUTSIDE the box):
        // its low-X face (X=36, inside) is solid; its high-X face (X=43, outside) is NOT.
        let straddle_min = [36i64, 16, 16];
        let straddle = resolve_boundary_block(&leaves, straddle_min, density, density);
        assert!(
            straddle.seam_solidity.face_is_solid(0, 0),
            "the inside (low-X) face of a +X-straddling block must be solid"
        );
        assert!(
            !straddle.seam_solidity.face_is_solid(0, 1),
            "the outside (high-X) face of a +X-straddling block must NOT be solid"
        );
    }

    /// The capability is OFF by default: `build_chunk` / `resolve_region_two_layer` return
    /// `None` so the caller falls back to the dense path (the coexistence contract).
    #[test]
    fn capability_off_by_default_returns_none() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let store = TwoLayerStore::default();
        assert!(!store.is_enabled());
        assert!(store.build_chunk([0, 0, 0], &scene, 16, 0).is_none());
        assert!(resolve_region_two_layer(&store, &scene, 16, 0).is_none());
        // E4 exact sinks also return None when the capability is OFF (dense fallback).
        assert!(streamed_widest_run_in_band(&store, &scene, 16, 0, 0).is_none());
        assert!(stream_vox_occupancy(&store, &scene, 16, |_| {}).is_none());
    }

    // ===== ADR 0010 E4: cacheless STREAMING diameter / widest-run query ===========

    /// The whole-grid diameter readout — today's reference value the streamed query
    /// must reproduce (same as `store.rs::whole_grid_widest_run`).
    fn whole_grid_widest_run(scene: &Scene, vpb: u32, band: (u32, u32)) -> u32 {
        let region = scene.full_extent_blocks(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        grid.widest_run_in_band(band.0, band.1)
    }

    /// **THE E4 diameter PARITY GATE:** the STREAMED widest-run (coarse blocks accounted
    /// ANALYTICALLY, boundary per-voxel) equals today's dense
    /// `VoxelGrid::widest_run_in_band` for the gated scene, across a spread of bands.
    /// Mirrors `store.rs::assert_region_widest_run_matches_whole_grid`.
    fn assert_streamed_widest_run_matches_dense(scene: &Scene, vpb: u32, label: &str) {
        let dims = scene.placed_region_dimensions(vpb);
        let grid_z = dims[2];
        let mid = grid_z.saturating_sub(1) / 2;
        let bands = [
            (0, grid_z.saturating_sub(1)),
            (0, 0),
            (grid_z.saturating_sub(1), grid_z.saturating_sub(1)),
            (mid, mid),
            (mid, (mid + 2).min(grid_z.saturating_sub(1))),
            (grid_z + 10, grid_z + 20),
        ];
        let store = TwoLayerStore::enabled();
        for band in bands {
            let expected = whole_grid_widest_run(scene, vpb, band);
            let actual = streamed_widest_run_in_band(&store, scene, vpb, band.0, band.1)
                .expect("the two-layer capability is enabled");
            assert_eq!(
                actual, expected,
                "[{label}] streamed widest_run band {band:?} must equal the dense readout"
            );
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_streamed_widest_run_matches_dense(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_flat_and_odd_shapes() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: kind,
                        size_voxels: [size[0] * 16, size[1] * 16, size[2] * 16],
                        size_measurements: None,
                        voxels_per_block: 16,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                assert_streamed_widest_run_matches_dense(&scene, 16, &format!("{kind:?} {size:?}"));
            }
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_demo_scene() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, density),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "demo-scene");
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_demo_village() {
        let density = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, density);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, density);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, density);
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
        assert_streamed_widest_run_matches_dense(&scene, density, "demo-village");
    }

    /// A sketch-revolve solid (boundary-only) — its diameter streams identically.
    #[test]
    fn streamed_widest_run_matches_dense_for_sketch_solid() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let density = 16;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_streamed_widest_run_matches_dense(&scene, density, "sketch-revolve");
    }

    /// An OVERLAP multi-material scene (overlap blocks classify boundary) streams the
    /// same widest-run as the dense readout — a run crossing a coarse↔boundary seam is
    /// one contiguous span.
    #[test]
    fn streamed_widest_run_matches_dense_for_overlap_multi_material() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 4),
            make_tool_density(ShapeKind::Box, [2, 0, 0], MaterialChoice::Wood, density, 4),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "overlap-multi-material");
    }

    /// **Band-at-a-time interval fold parity (the OOM-fix guard):** two solid boxes
    /// separated along X give every covering row TWO disjoint occupied runs (a coalescing
    /// bug would merge them across the gap and report a doubled diameter); a torus adds
    /// boundary blocks that seam with the coarse interiors; and the helper's single-Z-slice
    /// bands clip blocks mid-row. The streamed interval fold must still match the dense
    /// oracle exactly across every band.
    #[test]
    fn streamed_widest_run_matches_dense_for_disjoint_runs_and_mixed_blocks() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [10, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 8, 0], MaterialChoice::Plain, density),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "disjoint-runs-mixed-blocks");
    }

    /// **COARSE + BOUNDARY in the SAME block-row, band slicing mid-block (the block-row
    /// dedup guard).** A small SOLID box has, along an interior `(block_y, block_z)`
    /// block-row, boundary FACE blocks at both X extremes flanking coarse INTERIOR blocks —
    /// so the dedup must both (a) count the block-row's coarse run once and (b) refine the
    /// boundary voxel rows per-voxel and MERGE them across the coarse↔boundary seam. The
    /// helper's single-Z-slice bands `(0,0)` / `(mid,mid)` cut a 16-voxel-tall block mid-height
    /// (a partial block layer), exercising the block-row's band clip. Must match the dense
    /// oracle exactly across every band.
    #[test]
    fn streamed_widest_run_matches_dense_for_coarse_and_boundary_in_same_block_row() {
        let density = 16;
        // A 4-block solid cube: interior blocks classify coarse, the six faces boundary, so an
        // interior block-row is boundary-coarse-…-coarse-boundary along X.
        let scene = Scene::from_nodes(vec![make_tool_density(
            ShapeKind::Box,
            [0, 0, 0],
            MaterialChoice::Stone,
            density,
            4,
        )]);
        assert_streamed_widest_run_matches_dense(&scene, density, "coarse+boundary-same-block-row");
    }

    /// **6M-CAP DISSOLUTION (query side):** the streamed diameter of an
    /// 800×800-revolve-class solid is accounted with coarse blocks ANALYTICALLY (no
    /// per-voxel expansion), so the whole-region densify the dense path needs never
    /// happens. We assert the streamed widest run equals the box's true 800-voxel face
    /// width and quantify the analytic saving (coarse cells vs per-voxel cells avoided).
    #[test]
    fn streamed_widest_run_dissolves_6m_cap_with_analytic_coarse() {
        let density = 16u32;
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, density);
        assert!(
            shape.exceeds_voxel_cap(density),
            "the large solid must exceed the dense 6M cap to prove the point"
        );
        let node = Node::new("BigBox", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        let scene = Scene::from_nodes(vec![node]);
        let dims = scene.placed_region_dimensions(density);
        let band = (0, dims[2].saturating_sub(1));
        let true_width = blocks * density; // 800-voxel face row.

        let store = TwoLayerStore::enabled();
        let widest = streamed_widest_run_in_band(&store, &scene, density, band.0, band.1)
            .expect("the two-layer capability is enabled");
        assert_eq!(
            widest, true_width,
            "the streamed diameter must report the solid box's true 800-voxel width"
        );

        // Quantify the analytic saving: count coarse-solid blocks (accounted by run +=
        // d, NO per-voxel expansion) vs boundary blocks (per-voxel). Each coarse block
        // elides d³ per-voxel cells from the scan.
        let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
        let mut coarse_blocks = 0u64;
        let mut boundary_blocks = 0u64;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = store
                        .build_chunk([chunk_x, chunk_y, chunk_z], &scene, density, 0)
                        .unwrap();
                    for bz in 0..CHUNK_BLOCKS {
                        for by in 0..CHUNK_BLOCKS {
                            for bx in 0..CHUNK_BLOCKS {
                                let block = [bx, by, bz];
                                if chunk.coarse_block(block).is_some() {
                                    coarse_blocks += 1;
                                } else if chunk.microblocks.contains_key(&block) {
                                    boundary_blocks += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let cells_per_block = (density as u64).pow(3);
        let analytic_cells_elided = coarse_blocks * cells_per_block;
        assert!(
            coarse_blocks > boundary_blocks,
            "a large solid box must be mostly coarse blocks (coarse {coarse_blocks} > \
             boundary {boundary_blocks})"
        );
        eprintln!(
            "E4 analytic diameter: {coarse_blocks} coarse blocks (accounted run += d, \
             {analytic_cells_elided} per-voxel cells ELIDED) vs {boundary_blocks} boundary \
             blocks (per-voxel); dense path would densify all {} region voxels",
            (blocks as u64 * density as u64).pow(3)
        );
    }

    // ===== ADR 0010 #54: chunk-granular INCREMENTAL edits on the two-layer path ======
    //
    // Mirrors `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`:
    // for every edit kind, the two-layer resident cache after an INCREMENTAL edit
    // (invalidate the dirty AABB's chunks, re-derive only those) is IDENTICAL — the
    // coarse layer + overlay + microblock maps + seam flags, via the derived
    // `TwoLayerChunk: PartialEq` — to a full from-scratch two-layer rebuild of scene B.

    /// A tool node for the incremental edit scenes (mirrors `store.rs::tool_node`).
    fn incr_tool_node(
        kind: ShapeKind,
        size: [u32; 3],
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// The full resident map a WHOLESALE two-layer rebuild produces for `scene`: every
    /// covering chunk built from scratch, keyed by absolute coord. This is the parity
    /// gate's ground truth — the "full rebuild" every incremental edit must equal.
    fn full_two_layer_resident(
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        let mut cache = TwoLayerResidentCache::enabled();
        let chunks = cache.resident_two_layer_chunks(scene, density, 0);
        chunks
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Snapshot a resident cache's covering chunks (post-edit) as an owned coord→chunk
    /// map, for the `== full` comparison.
    fn resident_snapshot(
        cache: &mut TwoLayerResidentCache,
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        cache
            .resident_two_layer_chunks(scene, density, 0)
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `cache` in place, driving the
    /// dirty set exactly as `app_core::rebuild`: build the leaf spatial index for both
    /// scenes, diff for the edit AABB, and `invalidate_aabb` the dirty chunks (or
    /// `clear()` for the non-localisable fallback). Returns `(evicted_count, took_aabb_path)`
    /// so the harness can assert the localisable edits touch a strict subset.
    fn apply_two_layer_incremental_edit(
        cache: &mut TwoLayerResidentCache,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> (usize, bool) {
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        match index_b.edit_aabb_since(&index_a) {
            Some(edit_aabb) => {
                let evicted = cache.invalidate_aabb(&edit_aabb, density);
                (evicted.len(), true)
            }
            None => {
                // The wholesale fallback: a density change or a region-spanning Part edit
                // has no localisable box (mirrors `app_core::rebuild`'s `clear()` arm).
                cache.clear();
                (0, false)
            }
        }
    }

    /// **THE #54 GATE — incremental == full for every LOCALISABLE edit kind.** For each of
    /// add / remove / move / resize / recolor, the two-layer resident cache after the
    /// incremental edit is IDENTICAL (coarse layer + overlay + microblock maps + seam
    /// flags) to a full from-scratch two-layer rebuild of scene B, AND the edit touched a
    /// strict SUBSET of the scene's chunks (proving it is genuinely incremental, not a
    /// disguised full rebuild). Mirrors
    /// `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`.
    #[test]
    fn incremental_two_layer_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // Three tools spread far apart in X so each occupies chunks the others don't
        // touch (clean localised edits). The interior "subject" box sits between two
        // static anchors that pin the composite extent (as in the dense net) — though
        // note a recentre shift does NOT invalidate the two-layer cache (chunk-local
        // frame), the anchors keep the setup parallel to the dense parity net.
        let anchor_lo =
            || incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi =
            || incr_tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            incr_tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood, density),
            anchor_hi(),
        ]);

        let recolor = {
            let mut b = scene_a.clone();
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            let replacement =
                incr_tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood, density);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            b.root_node_mut(1).transform = NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            b.add_node(incr_tool_node(
                ShapeKind::Box,
                [3, 3, 3],
                [90, 0, 0],
                MaterialChoice::Stone,
                density,
            ));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node] {
            // Incremental: wholesale-build A, then apply the single edit and re-fill.
            let mut cache = TwoLayerResidentCache::enabled();
            let total_before = {
                let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
                cache.resident_len()
            };
            let (evicted, took_aabb_path) =
                apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
            assert!(
                took_aabb_path,
                "[{label}] this edit kind must be localisable (the AABB path, not clear())"
            );
            let incremental = resident_snapshot(&mut cache, &scene_b, density);

            // The full from-scratch rebuild for scene B (the truth).
            let full = full_two_layer_resident(&scene_b, density);

            assert_eq!(
                incremental, full,
                "[{label}] incremental two-layer cache (coarse layer + overlay + microblock \
                 maps + seam flags per covering chunk) MUST equal a full from-scratch rebuild \
                 of scene B — a stale chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: the edit evicted strictly fewer chunks than the scene's
            // total resident count (so it is genuinely incremental, not a full rebuild).
            let scene_chunks = total_before.max(full.len());
            assert!(
                evicted < scene_chunks,
                "[{label}] a localised edit must evict strictly FEWER chunks ({evicted}) than \
                 the scene's total ({scene_chunks}) — else it is a disguised full rebuild"
            );
        }
    }

    /// Perf probe (block-row-dedup regression guard): the full-band diameter re-measure —
    /// the query that fires when the layer band or grid changes. Before the ADR 0010 E5
    /// block-row dedup this was O(volume) (a coarse block stamped all `d²` of its voxel rows):
    /// 130ms @800³ → 127s @8000³, freezing the main thread. After, it is O(total blocks) and
    /// runs on the background diameter worker (never the UI thread). Reports wall-clock across
    /// four solid-cube edge lengths. Run:
    /// `cargo test --release widest_run_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn widest_run_scaling_probe() {
        use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [50i64, 125, 250, 500] {
            let edge = blocks * density as i64;
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let widest = streamed_widest_run_in_band(
                &TwoLayerStore::enabled(),
                &scene,
                density,
                0,
                edge as u32,
            );
            let elapsed = start.elapsed();
            println!("widest-run {edge}^3 vx full band: {widest:?} in {elapsed:?}");
        }
    }

    /// Perf probe (interior-elision win): time the LIVE two-layer build for a large
    /// SOLID sketch-extrude box — the path the app actually runs (NOT shot's dense
    /// `resolve_region` golden oracle). Before elision every interior block resolved
    /// per-voxel (O(volume)); after, interiors classify coarse (O(surface)). Reports the
    /// coarse/sculpted split + wall-clock. Run:
    /// `cargo test --release two_layer_sketch_box_build_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn two_layer_sketch_box_build_probe() {
        use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [25i64, 50] {
            let edge = blocks * density as i64; // 400, then 800 voxels/axis (block-aligned)
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let elapsed = start.elapsed();
            let coarse: u64 = chunks
                .iter()
                .map(|(_, chunk)| chunk.coarse.iter().filter(|id| id.is_some()).count() as u64)
                .sum();
            let sculpted: u64 = chunks.iter().map(|(_, chunk)| chunk.microblocks.len() as u64).sum();
            println!(
                "sketch box {edge}³ voxels ({blocks} blocks/axis): two-layer build {:?} — \
                 {coarse} coarse + {sculpted} sculpted blocks over {} chunks",
                elapsed,
                chunks.len()
            );
        }
    }

    /// Perf probe (`#[ignore]`d — run in release): per-stage timing of the FULL brick
    /// pipeline a wholesale rebuild runs after the two-layer classify, on solid
    /// sketch-extrude cubes of growing block span. This is the interior-elision regression
    /// guard for the 8000³-cube freeze fix (ADR 0011 surface-only record contract): at
    /// density 16 the 500-blk/axis cube is 125M blocks, and before the surface-only build
    /// every stage was O(all blocks) — ~12.5s of serial main-thread work and ~6 GB of
    /// transient record traffic per rebuild. With the record set ∝ surface, every stage
    /// must stay sub-second and the record count ~1.5M (the shell), not 125M.
    ///
    /// `cargo test --release brick_pipeline_scaling_probe -- --ignored --nocapture`
    /// The 500-blk/axis case (the actual 8000³ user scene) is opt-in via
    /// `VOXELWORKER_PROBE_LARGE=1` — it is the slowest case and the smaller spans already
    /// expose any O(volume) regression as a super-quadratic jump between rows.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn brick_pipeline_scaling_probe() {
        use crate::brick_field::{
            build_brick_field, BrickRecord, ClipmapPyramid, IncrementalBrickField,
        };
        use crate::brick_raymarch::pack_gpu_records;
        use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        let mut block_spans = vec![50i64, 125, 250];
        if std::env::var_os("VOXELWORKER_PROBE_LARGE").is_some() {
            block_spans.push(500); // the 8000³-voxel user cube
        }
        for blocks in block_spans {
            let edge = blocks * density as i64;
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let stage_start = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let classify_elapsed = stage_start.elapsed();
            let stage_start = std::time::Instant::now();
            let build = build_brick_field(&chunks, density);
            let field_elapsed = stage_start.elapsed();
            let record_count = build.brick_records.len();
            let record_megabytes =
                (record_count * std::mem::size_of::<BrickRecord>()) as f64 / 1.0e6;
            let stage_start = std::time::Instant::now();
            let gpu_records = pack_gpu_records(&build.brick_records, |_| false);
            let pack_elapsed = stage_start.elapsed();
            let stage_start = std::time::Instant::now();
            let pyramid = ClipmapPyramid::from_chunks(&chunks);
            let pyramid_elapsed = stage_start.elapsed();
            let stage_start = std::time::Instant::now();
            // Single-owner rework (item 9): from_wholesale now MOVES the records + atlas bytes
            // (no records clone), seeding only the bit tiles.
            let (incremental_mirror, _atlas) = IncrementalBrickField::from_wholesale(build);
            let wholesale_elapsed = stage_start.elapsed();
            println!(
                "brick pipeline probe {edge}^3 vx ({blocks} blk/axis): classify {} chunks \
                 {classify_elapsed:?} | brick_field {} surface records ({record_megabytes:.0} MB) \
                 {field_elapsed:?} | gpu_pack {} records {pack_elapsed:?} | \
                 pyramid(from_chunks) {pyramid_elapsed:?} | \
                 from_wholesale (records move + tile seed) {wholesale_elapsed:?}",
                chunks.len(),
                record_count,
                gpu_records.len(),
            );
            drop(incremental_mirror);
            drop(pyramid);
        }
    }

    /// A localised recolor of one small far-flung node dirties only the handful of chunks
    /// that node occupies, NOT the whole scene — the two-layer analogue of
    /// `store.rs::localized_recolor_rebuilds_few_chunks`.
    #[test]
    fn incremental_two_layer_localized_recolor_evicts_few_chunks() {
        let density = 16u32;
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "an in-place recolor must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            evicted * 2 < total,
            "a localised recolor of a small node must evict far fewer than half the chunks: \
             evicted {evicted} of {total}"
        );
        assert_eq!(incremental, full_two_layer_resident(&scene_b, density));
    }

    /// **Localisable move re-derives BOTH endpoints.** A moved node's dirty AABB spans its
    /// source AND destination (the `edit_aabb_since` union), so the two-layer cache vacates
    /// the source chunks and rebuilds the destination — and the result equals a full
    /// rebuild (no stale geometry left at the old location).
    #[test]
    fn incremental_two_layer_move_clears_source_and_fills_destination() {
        let density = 16u32;
        // A wide anchor keeps many chunks resident that the moved box never touches, so a
        // move touching a strict subset is meaningful.
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [2, 2, 2], [70, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = NodeTransform::from_blocks([85, 0, 0], density);

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "a move must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "a move must leave no stale geometry at the source and match a full rebuild"
        );
        assert!(evicted < total, "a move touches a strict subset ({evicted} of {total})");
    }

    /// **WHOLESALE FALLBACK — a density change re-derives everything.** A density change
    /// resizes every chunk's voxel extent, so `edit_aabb_since` returns `None` and the
    /// cache clears (belt-and-braces: `invalidate_aabb` also clears on a density mismatch).
    /// After the fallback the cache still equals a full rebuild at the NEW density.
    #[test]
    fn incremental_two_layer_density_change_falls_back_to_wholesale() {
        let density_a = 16u32;
        let density_b = 8u32;
        let scene = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density_a),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene, density_a, 0);
        // The density-change diff: the same scene rebuilt at a different density has no
        // localisable AABB (the indices differ in density), so `edit_aabb_since` is None.
        let index_a = scene.build_leaf_spatial_index(density_a);
        let index_b = scene.build_leaf_spatial_index(density_b);
        assert!(
            index_b.edit_aabb_since(&index_a).is_none(),
            "a density change must have no localisable edit AABB (the wholesale fallback)"
        );
        cache.clear();
        let incremental = resident_snapshot(&mut cache, &scene, density_b);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene, density_b),
            "after the density-change wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// **WHOLESALE FALLBACK — editing an unbounded (region-spanning) producer.** Editing a
    /// `DebugClouds` Part (its dirty region is "everywhere", `edit_aabb_since` returns
    /// `None`) forces a wholesale clear; the rebuilt cache still equals a full rebuild.
    /// This is the "unboundable-producer edit falls back to wholesale" acceptance case.
    #[test]
    fn incremental_two_layer_cloud_edit_falls_back_to_wholesale() {
        use crate::scene::Part;
        let density = 16u32;
        let cloud = |seed: u32| {
            let mut node = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed }));
            node.transform = NodeTransform::from_blocks([0, 0, 0], density);
            node
        };
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Box, [3, 3, 3], [0, 0, 0], MaterialChoice::Stone, density),
            cloud(7),
        ]);
        // Edit the cloud's seed (a region-spanning content change; root index 1).
        let mut scene_b = scene_a.clone();
        if let NodeContent::Part(Part::DebugClouds { seed }) =
            &mut scene_b.root_node_mut(1).content
        {
            *seed = 42;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
        let (_evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(
            !took_aabb_path,
            "editing a region-spanning Part must take the wholesale fallback, not the AABB path"
        );
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "after the cloud-edit wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// Wholesale-build timing probe across a WIDE object-count range (#66; the #63 lesson —
    /// a small N hides a super-linear asymptote). Not a correctness gate: run manually with
    /// `cargo test --release --lib wholesale_build_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe, run manually with --release --ignored --nocapture"]
    fn wholesale_build_scaling_probe() {
        let density = 16u32;
        for boxes_per_axis in [5i64, 12, 22] {
            let mut nodes = Vec::new();
            for grid_z in 0..boxes_per_axis {
                for grid_y in 0..boxes_per_axis {
                    for grid_x in 0..boxes_per_axis {
                        nodes.push(make_tool_density(
                            ShapeKind::Box,
                            [grid_x * 4, grid_y * 4, grid_z * 4],
                            MaterialChoice::Stone,
                            density,
                            2,
                        ));
                    }
                }
            }
            let object_count = boxes_per_axis.pow(3);
            let scene = Scene::from_nodes(nodes);
            let leaves_started = std::time::Instant::now();
            let leaves = scene.leaf_producers(density);
            let leaves_elapsed = leaves_started.elapsed();
            let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
            let chunk_count = (0..3)
                .map(|axis| (max_chunk[axis] - min_chunk[axis] + 1) as i64)
                .product::<i64>();
            let broadphase_started = std::time::Instant::now();
            let broadphase = leaf_edit_broadphase(&leaves, density);
            let broadphase_elapsed = broadphase_started.elapsed();
            std::hint::black_box(&broadphase);
            let build_started = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let build_elapsed = build_started.elapsed();
            eprintln!(
                "N={object_count} objects, {chunk_count} covering chunks: leaf hoist \
                 {leaves_elapsed:?}, edit-broadphase BVH rebuild {broadphase_elapsed:?}, \
                 wholesale build {build_elapsed:?} ({} chunks emitted)",
                chunks.len()
            );
        }
    }

    /// The capability OFF (the default): the resident cache is a no-op — it never fills and
    /// `resident_two_layer_chunks` returns empty, so a caller falls back to the dense path.
    #[test]
    fn incremental_two_layer_capability_off_is_noop() {
        let density = 16u32;
        let scene = shape_scene(ShapeKind::Sphere, density);
        let mut cache = TwoLayerResidentCache::default();
        assert!(!cache.is_enabled());
        assert!(cache.resident_two_layer_chunks(&scene, density, 0).is_empty());
        assert_eq!(cache.resident_len(), 0);
    }
}
