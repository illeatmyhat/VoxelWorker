//! The two-layer chunk representation: coarse block-ID grid + sparse microblock cuboids + seam flags.

use std::collections::BTreeMap;


use voxel_core::core_geom::{BlockAttrs, BlockId, CellKey, CHUNK_BLOCKS};
use crate::cuboid::{VoxelBox, VoxelBoxMaterial};
use voxel_core::voxel::Voxel;

#[allow(unused_imports)]
use super::*;

/// The coarse verdict for a single BLOCK of a chunk (ADR 0010 Decision 2). Distinct from
/// [`FieldClassification`](document::voxel::FieldClassification) (the per-producer interval verdict) because the BLOCK verdict
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
/// frame [`decompose_into_boxes`](crate::cuboid::decompose_into_boxes) yields over a `density³` [`VoxelRegion`](crate::cuboid::VoxelRegion)); the owning
/// block's chunk-local position is the map key in [`TwoLayerChunk::microblocks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroblockGeometry {
    /// The block's solid voxels, greedy-decomposed to single-material cuboids
    /// (block-local voxel indices `[0, density)`). Empty when the block resolved to no
    /// voxels (a boundary verdict the per-voxel pass found empty — still exact).
    ///
    /// Each cuboid's `material_id` is the cuboid mesher's **render-cell key** (ADR 0003
    /// §3c): the clean categorical `block_id` in the low bits, the transient on-face-grid
    /// overlay marker in the high bit (see [`voxel_core::core_geom::CellKey`]). The decomposition
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
pub(crate) fn coarse_flat_index(block: [u32; 3]) -> usize {
    let n = CHUNK_BLOCKS as usize;
    (block[2] as usize * n + block[1] as usize) * n + block[0] as usize
}

impl TwoLayerChunk {
    /// An all-air chunk at `voxels_per_block` (no coarse ids, no microblocks).
    pub(crate) fn empty(voxels_per_block: u32) -> Self {
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
            .map(VoxelBox::cell_count)
            .sum()
    }

    /// Stream this chunk back to full occupancy into `output` (ADR 0010 Decision 3 /
    /// parity gate (a)): a coarse-solid block is a fast `density³` fill at its block id;
    /// a boundary block expands its cuboids per-voxel. Each voxel is stamped at its
    /// CHUNK-LOCAL voxel index (the chunk's own `[0, chunk_extent_voxels)` frame) +
    /// `index_offset`, so the caller can rebase the whole chunk into the recentred /
    /// floating-origin frame in one integer add (mirroring
    /// [`Scene::resolve_chunk_rebased`](document::scene::Scene::resolve_chunk_rebased)).
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
            let block_id = BlockId(CellKey::from_raw(cuboid.material_id()).block_id());
            let grid_overlay = CellKey::from_raw(cuboid.material_id()).has_overlay();
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
pub(crate) fn stamped_voxel(
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

