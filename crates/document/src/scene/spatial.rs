//! Spatial index and chunk covering: the composite voxel AABB, the covering
//! chunk range (ADR 0002 far-placement i64 math narrowed to the i32 chunk key),
//! and the per-leaf [`LeafSpatialIndex`] the edit-diff broadphase reads.

use voxel_core::spatial_index::{LeafEntry, LeafFingerprint, LeafSpatialIndex, VoxelAabb};

use super::extent::rotated_grid_extent_voxels;
use super::producers::{
    leaf_content_fingerprint, operation_masks_beyond_bounds, outset_voxels_at,
};
use super::*;

impl Scene {
    /// Whether the scene has at least one intrinsic-size leaf (a Tool), so it has a
    /// composite AABB that the chunked resolve (`chunk_cache`) can cover.
    /// `false` for a VoxelBody-only scene (e.g. a lone debug-cloud field), which has no
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
    pub(super) fn placed_extent_voxels(&self, voxels_per_block: u32) -> Option<([i64; 3], [i64; 3])> {
        let mut min_corner = [i64::MAX; 3];
        let mut max_corner = [i64::MIN; 3];
        let mut any = false;
        self.for_each_leaf(&mut |world_offset_voxels, _offset_local_voxels, _orientation, rotation, body, _grid_on_faces, _operation, outset, _scope_path| {
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            let Some(grid_voxels) = body.grid_voxels(voxels_per_block, outset_voxels) else {
                return;
            };
            // ADR 0026: an oriented leaf occupies its TURNED grid in the world (a 4×4×20
            // cylinder stood on a +X wall spans 20×4×4), still corner-anchored at its world
            // offset. Turn the local extent into world axes before the span.
            let grid_voxels = rotated_grid_extent_voxels(rotation, grid_voxels);
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
    /// `pub` so the chunk cache (issue #27 S2, up in the app crate) iterates the covering
    /// chunks for reassembly.
    ///
    /// Derived from [`placed_extent_voxels`](Self::placed_extent_voxels) — the
    /// producer-true voxel frame — so it covers every chunk a voxel can land in,
    /// including the chunks an odd/flat block size straddles (which the block-AABB
    /// frame would miss).
    pub fn covering_chunk_range(&self, voxels_per_block: u32) -> Option<([i32; 3], [i32; 3])> {
        let (min_voxel_corner, max_voxel_corner) = self.placed_extent_voxels(voxels_per_block)?;
        // The voxel corners are i64 (a far-placed leaf), but the chunk extent is
        // small; the block→chunk division therefore happens in i64 and the QUOTIENT
        // (the chunk coordinate) narrows to i32 safely — for offsets up to ±10⁹
        // blocks at density 16 a chunk coord is ≤ ±2.5×10⁸, well inside i32 (S4a).
        let chunk_extent_voxels = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

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

    /// The number of covering chunks the `.vox` streaming export will visit at
    /// `voxels_per_block` — the product of the per-axis chunk-range extents from
    /// [`covering_chunk_range`](Self::covering_chunk_range), or `0` for a VoxelBody-only /
    /// empty scene (no covering range). Public so the shell can size the export progress
    /// readout's denominator without materialising any occupancy; the async export worker
    /// increments its per-chunk counter to exactly this total.
    pub fn covering_chunk_count(&self, voxels_per_block: u32) -> u64 {
        let Some((min_chunk, max_chunk)) = self.covering_chunk_range(voxels_per_block) else {
            return 0;
        };
        (0..3)
            .map(|axis| (max_chunk[axis] - min_chunk[axis] + 1) as u64)
            .product()
    }

    /// Build a [`LeafSpatialIndex`] over the
    /// scene's leaves at `voxels_per_block` (issue #27 S3).
    ///
    /// One `for_each_leaf` walk records, per enabled leaf, its world-AABB in the
    /// **absolute-voxel producer-true frame** — the SAME frame
    /// [`resolve_chunk`](Self::resolve_chunk) and `placed_extent_voxels` use, so a
    /// chunk derived from a leaf's index AABB is exactly a chunk that leaf's voxels
    /// can land in. A leaf with an intrinsic size (a Tool) gets a concrete box
    /// `[off·d − grid/2, off·d + grid/2)`; a region-spanning leaf (a VoxelBody, no
    /// intrinsic size) gets an empty box and a
    /// [`RegionSpanning`](voxel_core::spatial_index::LeafFingerprint::RegionSpanning)
    /// fingerprint (it cannot be chunk-localised; an edit touching it forces a
    /// wholesale clear).
    ///
    /// By construction the index's entries ARE the leaves `for_each_leaf` yields, so
    /// a query against the index returns the same leaf set as the full walk filtered
    /// by AABB (proven in the spatial-index tests).
    pub fn build_leaf_spatial_index(&self, voxels_per_block: u32) -> LeafSpatialIndex {
        let mut entries: Vec<LeafEntry> = Vec::new();
        let mut has_region_spanning_leaf = false;
        self.for_each_leaf(&mut |world_offset_voxels, _offset_local_voxels, _orientation, rotation, body, grid_on_faces, operation, outset, scope_path| {
            // ADR 0020 Consequences: the edit-broadphase AABB must be the OUTSET bounds, not
            // the producer bounds — an outset cutter dirties a larger region than its own
            // extent, and invalidating only the undilated box leaves a stale rim behind.
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            match body.grid_voxels(voxels_per_block, outset_voxels) {
                Some(grid_voxels) => {
                    // ADR 0026: the leaf's world box is its TURNED grid, corner-anchored at its
                    // world offset — so the broadphase covers the axes it actually occupies.
                    let grid_voxels = rotated_grid_extent_voxels(rotation, grid_voxels);
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
                    let payload = leaf_content_fingerprint(
                        world_offset_voxels,
                        &body,
                        grid_on_faces,
                        operation,
                        scope_path,
                    );
                    // ADR 0017 (#75): an Intersect-influence leaf's edits cannot be
                    // localised to its box (its mask kills cells anywhere outside its
                    // body), so it carries the fingerprint kind whose presence in an
                    // edit diff forces a wholesale clear. The box itself stays real
                    // for overlap queries.
                    let fingerprint =
                        if operation_masks_beyond_bounds(operation, scope_path) {
                            LeafFingerprint::MasksBeyondItsBox(payload)
                        } else {
                            LeafFingerprint::Bounded(payload)
                        };
                    entries.push(LeafEntry {
                        world_aabb: VoxelAabb::new(min, max),
                        fingerprint,
                    });
                }
                None => {
                    // A region-spanning leaf (a VoxelBody): no intrinsic box. Record it
                    // with an empty AABB + a region-spanning fingerprint so an edit
                    // touching it forces a wholesale clear (it can't be localised).
                    has_region_spanning_leaf = true;
                    entries.push(LeafEntry {
                        world_aabb: VoxelAabb::new([0; 3], [0; 3]),
                        fingerprint: LeafFingerprint::RegionSpanning(leaf_content_fingerprint(
                            world_offset_voxels,
                            &body,
                            grid_on_faces,
                            operation,
                            scope_path,
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

/// Narrow an `i64` chunk coordinate to `i32` (the cache-key / chunk-index width).
///
/// **Audit (S4a, ADR 0002 Decision 2):** the absolute-VOXEL math is i64 so a
/// far-placed node composes without overflow, but the CHUNK coordinate (= voxel /
/// chunk_extent) is much smaller — at density 16 / `CHUNK_BLOCKS = 4` the extent is
/// 64 voxels, so a block offset of ±10⁹ yields a chunk coord of only ±2.5×10⁸,
/// comfortably inside i32 (±2.1×10⁹). Keeping the chunk coord / cache key i32 is
/// therefore safe and avoids widening the whole chunk index. A coordinate that
/// would not fit i32 means a block offset past ~±8×10⁹ and is clamped
/// (debug-asserted) rather than silently wrapping.
///
/// **Correction 2026-07-18 — ±8×10⁹ is NOT the supported placement range.** This audit is
/// sound about what it bounds (the chunk coordinate) but that is a voxel index DIVIDED by
/// the chunk extent, and the two-layer expansion multiplies back by that extent to rebase
/// each voxel into an `i32` `local_index`. Bounding the quotient says nothing about the
/// product: at ±8×10⁹ blocks the voxel index overruns `i32` by 4× at density 1 and 238× at
/// density 64. The real range is [`voxel_core::core_geom::max_supported_block_offset`]
/// (±3.3×10⁷ blocks at density 64), which is what the expansion's unchecked `as i32`
/// actually tolerates — Kani-proved there. See ADR 0008's 2026-07-18 amendment.
fn narrow_chunk_coord(chunk_coord: i64) -> i32 {
    debug_assert!(
        chunk_coord >= i32::MIN as i64 && chunk_coord <= i32::MAX as i64,
        "chunk coordinate {chunk_coord} overflows i32 — block offset is past the \
         supported ±~8×10⁹-block range (S4a)"
    );
    chunk_coord.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}
