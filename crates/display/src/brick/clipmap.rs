use super::*;

/// One mixed block's per-voxel cell-key tile: `edge³` render-cell keys (clean block id +
/// overlay bit), block-local x-fastest — the sibling of [`BrickOccupancyTile`]. Only a block
/// whose microblocks disagree on their cell key carries one; a uniform block's single key
/// lives on its record.
pub type BrickCellKeyTile = ValueTile<u16>;

/// The cell key an AIR voxel of a mixed block's cell-key tile holds — a documented
/// **don't-care**: occupancy gates every read of the tile (a cleared occupancy bit means the
/// voxel is not there at all), so no consumer may attribute meaning to it. `0` is chosen only
/// because it is the cheapest fill.
pub(crate) const AIR_CELL_KEY_DONT_CARE: u16 = 0;

// The clip-map occupancy levels ARE substrate's `SparseMinMipPyramid`: a sparse min-mip that folds
// a set of packed lattice keys to coarser cells (edge 8, then 64, then 512 blocks), keeping the
// folded cell keys sorted + deduplicated as a conservative-superset occupancy the raymarch's
// hierarchical DDA skips against. The domain keeps the "clip-map" name and the CHUNK traversal
// (`ClipmapLevel::from_chunks`, with its solid-chunk bulk fast path) at this seam; the pure fold,
// the multi-level assembly, and the binary-search lookup live in the substrate module. The
// three-level edge progression (8/64/512) is domain configuration, passed to the fold. See
// docs/architecture/03-display.md (the brick-field clip-map) for how the levels drive the march.
use substrate::spatial::min_mip_pyramid::{fold_coordinate_to_cell, MinMipLevel};

// ============================================================================
// Clip-map occupancy pyramid (ADR 0011 Decision 4a / slice G2+G4) — THREE
// WORLD-FIXED coarse "any-brick-inside" levels above the brick set, a min-mip of
// the record keys on an 8× cell progression (8 → 64 → 512 blocks/cell). The
// hierarchical DDA (brick_raymarch.wgsl) jumps a ray to the exit of the coarsest
// EMPTY level covering its position — one stride through empty space — descending
// to per-block brick work only where a level reports occupancy. This is the port
// of ADR 0009's measured 160→10240 (~64×) scattered-ceiling lift; G4 adds the
// third level (512-block cells) so a wide scatter skips whole 512-block voids in
// one stride instead of eight L2 strides, closing most of the raw scattered
// ceiling gap vs the rasterized mesh (frustum/Z cull it gets for free).
//
// Why stop at three: the packed key is 21 bits/axis (±2^20 blocks), so a fourth
// level (4096 blocks/cell) has at most ~512 cells of span to skip; on realistic
// 10k-block-span scenes L3 already caps the empty-void skip at a handful of
// strides and an L4 stride only replaces ~8 already-cheap L3 strides — measured
// not to pay (see `clipmap_scattered_scene_skips_empty_space`'s +L4 column).
// ============================================================================

/// Level 1 (fine) clip-map cell edge, in BLOCKS — the benchmark's proven config
/// (ADR 0011 Decision 4a). Block-denominated (density-agnostic by construction),
/// never a hard-coded voxel count.
pub const CLIPMAP_LEVEL_1_BLOCKS_PER_CELL: u32 = 8;
/// Level 2 (middle) clip-map cell edge, in BLOCKS (the benchmark's L2) — 8× L1.
pub const CLIPMAP_LEVEL_2_BLOCKS_PER_CELL: u32 = 64;
/// Level 3 (coarse) clip-map cell edge, in BLOCKS (G4) — 8× L2, checked first by
/// the hierarchical DDA so a wide empty void skips in one 512-block stride.
pub const CLIPMAP_LEVEL_3_BLOCKS_PER_CELL: u32 = 512;

/// One clip-map occupancy level: cells of `blocks_per_cell` blocks per axis, each
/// a packed cell key (the SAME 21-bit z-major packing as a brick record's block
/// key, applied to the CELL coordinate = `floor_div(absolute_block,
/// blocks_per_cell)`). `cell_keys` is sorted strictly ascending + unique — the
/// order the in-shader binary search relies on, exactly like the record array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipmapLevel {
    /// Cell edge in blocks (8 for L1, 64 for L2). Block-denominated.
    pub blocks_per_cell: u32,
    /// The occupied cells' packed keys, sorted ascending + deduplicated — a
    /// SUPERSET of the true occupied cells by construction (every record's cell is
    /// present), so the hierarchical DDA only ever skips provably-empty space.
    pub cell_keys: Vec<u64>,
}

/// Wrap a substrate [`MinMipLevel`] as a domain [`ClipmapLevel`] (the kernel's `cell_edge` IS this
/// level's `blocks_per_cell`; the sorted-deduplicated `cell_keys` carry across byte-identically).
pub(crate) fn clipmap_level_from_kernel(level: MinMipLevel) -> ClipmapLevel {
    ClipmapLevel {
        blocks_per_cell: level.cell_edge,
        cell_keys: level.cell_keys,
    }
}

impl ClipmapLevel {
    /// An empty level (no occupied cells) — the "pyramid off" form the renderer
    /// installs to A/B the hierarchical skip (`record_count == 0` ⇒ the shader
    /// never skips, so the march is the flat G1 block-DDA).
    pub fn empty(blocks_per_cell: u32) -> Self {
        clipmap_level_from_kernel(MinMipLevel::empty(blocks_per_cell))
    }

    /// Fold a record set's block keys into this level's occupied-cell set: every
    /// record's block maps to exactly one cell; the deduplicated, sorted set is
    /// the min-mip. A thin adapter over the substrate fold ([`MinMipLevel::from_keys`]) —
    /// extract the packed block keys, hand them to the kernel (ADR 0011 4a).
    pub fn from_records(records: &[BrickRecord], blocks_per_cell: u32) -> Self {
        // Fold the record block keys straight through the kernel's single-pass entry — no
        // intermediate `Vec<u64>` (only the folded cell-key output is allocated).
        clipmap_level_from_kernel(MinMipLevel::from_key_iter(
            records.iter().map(|record| record.packed_world_block_key),
            blocks_per_cell,
        ))
    }

    /// Fold every non-air block of the two-layer chunk set into this level's occupied-cell
    /// set — the **chunk-sourced** min-mip that replaces [`from_records`](Self::from_records)
    /// now that the record set is SURFACE-ONLY (ADR 0011 interior elision, this epic). The
    /// pyramid must stay a conservative superset over EVERY occupied block (interior included,
    /// so the DDA never strides past an occupied cell), which the surface record set no longer
    /// enumerates — but the chunks do (their coarse layer holds the interior).
    ///
    /// This is the DOMAIN traversal half of the extraction: it walks the chunks and emits the
    /// per-cell keys, then hands the raw key list to the substrate sort+dedup sink
    /// ([`MinMipLevel::from_folded_cell_keys`]) — the pure fold lives in the kernel, the chunk
    /// walk and its fast path stay here.
    ///
    /// **Solid-chunk bulk fast path (the interior-elision win carried to the pyramid):** a
    /// fully-solid chunk (all `CHUNK_BLOCKS³` coarse-solid, no microblocks) covers one aligned
    /// block box, so its occupied cells are the cell range that box spans — bulk-added WITHOUT
    /// visiting its 64 blocks. A boundary / partial chunk adds one cell per occupied (coarse or
    /// microblock) block. The resulting cell set is BYTE-IDENTICAL to `from_records` over the
    /// full, interior-inclusive record set (proven by
    /// `clipmap_from_chunks_equals_from_full_records`): every occupied block's cell is present,
    /// no others.
    pub fn from_chunks(chunks: &[([i32; 3], Arc<TwoLayerChunk>)], blocks_per_cell: u32) -> Self {
        let cell_edge = blocks_per_cell.max(1);
        let chunk_blocks = CHUNK_BLOCKS as i64;
        let mut cell_keys: Vec<u64> = Vec::new();
        for (chunk_coord, chunk) in chunks {
            let chunk_block_low = [
                chunk_coord[0] as i64 * chunk_blocks,
                chunk_coord[1] as i64 * chunk_blocks,
                chunk_coord[2] as i64 * chunk_blocks,
            ];
            let fully_solid =
                chunk.microblocks.is_empty() && chunk.coarse.iter().all(Option::is_some);
            if fully_solid {
                // The chunk's whole block box [low, low + CHUNK_BLOCKS) maps to this aligned
                // cell range — add each cell once, no per-block visit (the bulk fast path).
                let cell_lo = fold_coordinate_to_cell(chunk_block_low, cell_edge);
                let cell_hi = fold_coordinate_to_cell(
                    [
                        chunk_block_low[0] + chunk_blocks - 1,
                        chunk_block_low[1] + chunk_blocks - 1,
                        chunk_block_low[2] + chunk_blocks - 1,
                    ],
                    cell_edge,
                );
                for cell_z in cell_lo[2]..=cell_hi[2] {
                    for cell_y in cell_lo[1]..=cell_hi[1] {
                        for cell_x in cell_lo[0]..=cell_hi[0] {
                            cell_keys.push(pack_world_block_key([cell_x, cell_y, cell_z]));
                        }
                    }
                }
            } else {
                for block_z in 0..CHUNK_BLOCKS {
                    for block_y in 0..CHUNK_BLOCKS {
                        for block_x in 0..CHUNK_BLOCKS {
                            let block = [block_x, block_y, block_z];
                            let occupied = chunk.coarse_block(block).is_some()
                                || chunk.microblocks.contains_key(&block);
                            if !occupied {
                                continue;
                            }
                            let cell = fold_coordinate_to_cell(
                                [
                                    chunk_block_low[0] + block_x as i64,
                                    chunk_block_low[1] + block_y as i64,
                                    chunk_block_low[2] + block_z as i64,
                                ],
                                cell_edge,
                            );
                            cell_keys.push(pack_world_block_key(cell));
                        }
                    }
                }
            }
        }
        clipmap_level_from_kernel(MinMipLevel::from_folded_cell_keys(cell_keys, cell_edge))
    }
}

/// The three-level clip-map pyramid (L1 = 8-block cells, L2 = 64-block cells, L3
/// = 512-block cells; ADR 0011 Decision 4a + G4). A derived, rebuildable min-mip
/// of the brick records — never truth (ADR 0006/0009 4c). The DDA descends the
/// levels coarsest-first (L3 → L2 → L1) via [`Self::levels_coarse_to_fine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipmapPyramid {
    /// Fine level (8-block cells).
    pub level_1: ClipmapLevel,
    /// Middle level (64-block cells).
    pub level_2: ClipmapLevel,
    /// Coarse level (512-block cells) — checked first by the hierarchical DDA.
    pub level_3: ClipmapLevel,
    /// The block-granular interior-occupancy signal ([`BlockOccupancyMasks`]) — the
    /// band-clip cross-section fix (ADR 0011). Populated by [`from_chunks`](Self::from_chunks);
    /// EMPTY for the record-sourced / off constructors (they run FULL-band only, where the
    /// fallback never fires). Not part of the clip-map SKIP contract — the DDA never reads it;
    /// it rides here so the live install sites carry it without a new argument.
    pub interior_masks: BlockOccupancyMasks,
}

impl ClipmapPyramid {
    /// Build all levels from a brick-field's sorted records (a pure function of the record
    /// keys). **Oracle/legacy:** now that the live record set is surface-only, a pyramid the
    /// DDA can skip against must cover interior cells too, so the live sinks build from the
    /// CHUNKS via [`from_chunks`](Self::from_chunks); this constructor stays as the parity
    /// oracle (fed a full, interior-inclusive record set) and for the pyramid-shape unit tests.
    pub fn from_records(records: &[BrickRecord]) -> Self {
        // The record path folds the SAME block-key set at all three edges — exactly the substrate
        // multi-level assembly ([`SparseMinMipPyramid::from_key_iter`]). Stream the record keys to
        // the kernel (it buffers them ONCE internally for the three folds — the domain builds no
        // intermediate `Vec<u64>`), then wrap each level.
        let assembled = substrate::spatial::SparseMinMipPyramid::from_key_iter(
            records.iter().map(|record| record.packed_world_block_key),
            &[
                CLIPMAP_LEVEL_1_BLOCKS_PER_CELL,
                CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
                CLIPMAP_LEVEL_3_BLOCKS_PER_CELL,
            ],
        );
        let [level_1, level_2, level_3] = assembled
            .levels
            .try_into()
            .expect("three edges yield three levels");
        ClipmapPyramid {
            level_1: clipmap_level_from_kernel(level_1),
            level_2: clipmap_level_from_kernel(level_2),
            level_3: clipmap_level_from_kernel(level_3),
            // The oracle pyramid is a SKIP min-mip only (its record set is interior-inclusive,
            // so no band-clip miss-fallback signal is needed); the live from_chunks path carries it.
            interior_masks: BlockOccupancyMasks::empty(),
        }
    }

    /// Build all levels from the two-layer chunk set — the LIVE pyramid constructor (ADR 0011
    /// interior elision). The surface-only record set omits interior blocks the DDA's skip
    /// pyramid must still cover, so the min-mip is derived from the chunks (which retain the
    /// interior in their coarse layer), with a solid-chunk bulk fast path. Conservative-superset
    /// identical to [`from_records`](Self::from_records) over a full record set — see
    /// [`ClipmapLevel::from_chunks`].
    pub fn from_chunks(chunks: &[([i32; 3], Arc<TwoLayerChunk>)]) -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::from_chunks(chunks, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::from_chunks(chunks, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            level_3: ClipmapLevel::from_chunks(chunks, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
            // The band-clip interior-occupancy signal (this epic): block-granular, bitpacked,
            // consulted only on a record miss under an active band clip (ADR 0011).
            interior_masks: BlockOccupancyMasks::from_chunks(chunks),
        }
    }

    /// The "pyramid off" form — every level empty, so the shader's hierarchical
    /// skip never fires (the flat G1 block-DDA). Used by the pyramid-on == off
    /// parity assertion and the perf probe's baseline.
    pub fn empty() -> Self {
        ClipmapPyramid {
            level_1: ClipmapLevel::empty(CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::empty(CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            level_3: ClipmapLevel::empty(CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
            interior_masks: BlockOccupancyMasks::empty(),
        }
    }

    /// The levels ordered COARSEST → FINEST (L3, L2, L1) — the order the
    /// hierarchical DDA descends (skip by the coarsest empty level covering the
    /// ray's block). The CPU march mirror ([`crate::brick::cpu_march_brick_field`])
    /// and the perf probe iterate this slice.
    pub fn levels_coarse_to_fine(&self) -> [&ClipmapLevel; 3] {
        [&self.level_3, &self.level_2, &self.level_1]
    }
}

/// Split a level's sorted u64 cell keys into the `(hi, lo)` u32 pairs the WGSL
/// binary search consumes (no u64 in WGSL) — the pyramid analogue of
/// `pack_gpu_records`' key split.
pub fn pack_clipmap_level_keys(level: &ClipmapLevel) -> Vec<[u32; 2]> {
    level
        .cell_keys
        .iter()
        .map(|&key| substrate::spatial::lattice_key::split_key_hi_lo(key))
        .collect()
}
