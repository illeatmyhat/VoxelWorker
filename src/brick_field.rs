//! ADR 0011 G0 — the **brick-field BUILD**: pack ADR 0010's two-layer boundary set into
//! a sorted [`BrickRecord`] array (keyed by a packed world-block key) + an R8 3D
//! texture atlas of sculpted-brick occupancy. **Wired to NOTHING** — no renderer reads
//! this yet; it is the standalone build + parity harness (the analogue of ADR 0007's
//! atlas-mechanic-proven step and ADR 0010's E1), gated by `tests/gpu_parity.rs`
//! before any live raymarch (G1) consumes it.
//!
//! The mapping is ADR 0011 Decision 2 onto the [`TwoLayerChunk`] partition, **surface-only**
//! (ADR 0011 interior elision — the record-set contract since the 8000³-freeze fix):
//!
//! * **air block** → no record (the ray skips it via the clip-map, later).
//! * **coarse-solid block** → one [`BrickPayload::CoarseSolid`] record: a solid
//!   block-cube at its [`BlockId`], **no atlas slot, no per-voxel data** — UNLESS the
//!   block is fully occluded (all six face-neighbours present + solid on the shared
//!   face), in which case it emits NOTHING: a ray can never reach it first, so its
//!   record would be dead weight hauled through every stage. Interiors live in the
//!   two-layer chunks' coarse layer, not in the record set.
//! * **boundary block** (a `microblocks` entry) → one [`BrickPayload::Sculpted`] record
//!   whose atlas slot holds the block's voxel occupancy, rasterized from its cuboids.
//!   Every boundary block keeps its record (the sculpted set is never elided), so the
//!   atlas's sculpted tiles stay complete.
//!
//! **Consumers under this contract:** the shader record buffer binary-searches exactly
//! this set (no second elision pass); the clip-map pyramid derives from the CHUNKS
//! ([`ClipmapPyramid::from_chunks`], interiors included). The interior-inclusive build
//! survives as the parity oracle ([`build_brick_field_all_blocks`]).
//!
//! **The brick granule is ONE BLOCK** (ADR 0011 Decision 1): the brick edge is
//! `voxels_per_block` — block-denominated, correct at ANY density; nothing here may
//! hard-code 16. Per-face [`SeamSolidity`] flags carry across unchanged (they are the
//! brick-field's apron analogue).
//!
//! ## Frame (ADR 0008)
//!
//! A brick key is the block's **absolute world-block coordinate**
//! (`chunk_coord * CHUNK_BLOCKS + chunk_local_block_index`) — the same world-fixed
//! integer lattice the chunk keys live on; recentre/floating-origin never enters the
//! key (a recentre shift leaves every record valid, exactly like the chunk cache).
//!
//! ## Exactness (the ADR 0011 parity gate, clause (a))
//!
//! Packing is pure integer work: a sculpted brick's atlas bytes must be BYTE-IDENTICAL
//! to the CPU two-layer boundary set's occupancy for that block, and a coarse-solid
//! block must emit exactly one coarse record and consume no atlas slot. The
//! `--features gpu` parity tests assert this through the full texture round-trip.

use std::sync::Arc;

use rayon::prelude::*;

use voxel_core::core_geom::{BlockId, CellKey, CHUNK_BLOCKS};
use evaluation::cuboid::VoxelBoxMaterial;
use evaluation::two_layer_store::{SeamSolidity, TwoLayerChunk};

// The brick-record key codec IS substrate's `lattice_key`: an absolute world-block
// coordinate packed into one sortable `u64` in z-major lexicographic (z, y, x) order,
// 21 bits/axis (±2^20 blocks — far beyond the anisotropic 10k+-block target), so the
// record array's integer order IS block order (sortable on the CPU, binary-searchable
// as a `(hi, lo)` u32 pair on the GPU). The domain keeps the "world-block key" name at
// this seam; the space-filling-curve definition and citations live in the substrate
// module. See docs/architecture/data-structures.md (Substrate) for the codec itself.
pub use substrate::spatial::lattice_key::{
    pack_lattice_key as pack_world_block_key, unpack_lattice_key as unpack_world_block_key,
};

// A boundary block's occupancy tile IS substrate's `BitCube`: an edge-≤64 (density-bounded)
// 3D bitset stored one `u64` per X-row (one voxel per bit, x-fastest). The domain keeps the
// "occupancy tile" name at this seam; the word-packed-bitset definition, the run-set mask
// math, and its citations live in the substrate module. The sculpted-atlas scatter reuses
// substrate's `CubeTilePacking` (linear slot → cubic tile grid) and the per-slot store is a
// `SlotFreeList` (stable-index free-list). See docs/architecture/03-display.md (the
// brick-field atlas) for how these pack into the R8 atlas the raymarch samples.
use substrate::occupancy::{CubeTilePacking, SlotFreeList};
pub use substrate::occupancy::BitCube as BrickOccupancyTile;

// A MIXED block's per-voxel cell-key tile IS substrate's `ValueCube<u16>`: the payload sibling
// of the occupancy `BitCube` — the same cube edge, the same X-row layout (row = z*edge + y),
// one `u16` per voxel instead of one bit, so the occupancy tile gates the cell-key tile
// cell-for-cell and ONE rasterizing walk fills both. The domain keeps the "cell-key tile" name
// at this seam; the dense row-major cube, its row seam, and why it is not a `CellGrid` live in
// the substrate module. A cell key is the render-cell key of `cuboid_mesh` (clean block-palette
// id + the on-face-grid overlay bit), stored verbatim. See docs/architecture/03-display.md (the
// brick-field atlas) for the per-voxel material side atlas these tiles feed.
pub use substrate::occupancy::ValueCube as ValueTile;

/// One mixed block's per-voxel cell-key tile: `edge³` render-cell keys (clean block id +
/// overlay bit), block-local x-fastest — the sibling of [`BrickOccupancyTile`]. Only a block
/// whose microblocks disagree on their cell key carries one; a uniform block's single key
/// lives on its record.
pub type BrickCellKeyTile = ValueTile<u16>;

/// The cell key an AIR voxel of a mixed block's cell-key tile holds — a documented
/// **don't-care**: occupancy gates every read of the tile (a cleared occupancy bit means the
/// voxel is not there at all), so no consumer may attribute meaning to it. `0` is chosen only
/// because it is the cheapest fill.
const AIR_CELL_KEY_DONT_CARE: u16 = 0;

// The clip-map occupancy levels ARE substrate's `SparseMinMipPyramid`: a sparse min-mip that folds
// a set of packed lattice keys to coarser cells (edge 8, then 64, then 512 blocks), keeping the
// folded cell keys sorted + deduplicated as a conservative-superset occupancy the raymarch's
// hierarchical DDA skips against. The domain keeps the "clip-map" name and the CHUNK traversal
// (`ClipmapLevel::from_chunks`, with its solid-chunk bulk fast path) at this seam; the pure fold,
// the multi-level assembly, and the binary-search lookup live in the substrate module. The
// three-level edge progression (8/64/512) is domain configuration, passed to the fold. See
// docs/architecture/03-display.md (the brick-field clip-map) for how the levels drive the march.
use substrate::spatial::min_mip_pyramid::{fold_coordinate_to_cell, MinMipLevel};

// The block-occupancy masks' STORAGE is substrate's `SortedKeyBitmaskMap`: a sorted parallel-array
// map (keys ∥ fixed-width bitmasks ∥ per-key fallback scalar), binary-searchable, with the textbook
// word/bit indexing. The domain keeps the "occupancy masks" name and the CHUNK traversal
// (`BlockOccupancyMasks::from_chunks`, its solid-chunk bulk fast path, and its first-writer-wins
// fallback-material policy) at this seam; the parallel-array shape, the sort-by-key construction,
// the binary search, and the bit set/test live in the substrate module (fallback = a caller-defined
// `u32`, here the render-cell material colour index). See docs/architecture/03-display.md (the
// band-clip interior fallback) for how the packed cells feed the raymarch.
use substrate::occupancy::bitmask_map::{set_mask_bit, SortedKeyBitmaskMap};

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
fn clipmap_level_from_kernel(level: MinMipLevel) -> ClipmapLevel {
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
    /// ray's block). The CPU march mirror ([`crate::brick_raymarch::cpu_march_brick_field`])
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

/// The clip-map cell edge (in blocks) the [`BlockOccupancyMasks`] bitmask cells use —
/// the same 8-block granule as the pyramid's [`ClipmapLevel`] L1, so a `512`-block
/// interior-occupancy cell is one `u32[16]` bitmask.
pub const BLOCK_OCCUPANCY_CELL_BLOCKS: u32 = CLIPMAP_LEVEL_1_BLOCKS_PER_CELL;
/// Blocks per [`BlockOccupancyMasks`] cell (`8³ = 512`) — the bitmask's bit count.
const BLOCK_OCCUPANCY_BITS_PER_CELL: usize =
    (BLOCK_OCCUPANCY_CELL_BLOCKS * BLOCK_OCCUPANCY_CELL_BLOCKS * BLOCK_OCCUPANCY_CELL_BLOCKS)
        as usize;
/// `u32` words in one cell's occupancy bitmask (`512 / 32 = 16`).
pub const BLOCK_OCCUPANCY_MASK_WORDS: usize = BLOCK_OCCUPANCY_BITS_PER_CELL / 32;

/// **ADR 0011 — the band-clip interior-occupancy signal (this fix).** A block-granular,
/// bitpacked occupancy map over the two-layer chunks, consulted by the raymarch ONLY when a
/// LAYER-BAND clip is active AND the surface-only record search misses.
///
/// The surface-only record set (interior elision, `b1cadb7`/`6f0718e`) omits fully-occluded
/// interior blocks. Under a FULL band that is hit-identical (a ray reaches an interior block
/// only through a solid surface neighbour that keeps its record, stopping the ray first —
/// [`BrickOcclusionOracle`]). But a band cut-plane SLICES a solid, so a ray can start/enter
/// INSIDE the solid at a block whose record was elided: the record search misses,
/// indistinguishable at the record level from genuine air, and the cross-section renders
/// hollow. This map is the distinguishing signal: an occupied bit + a record miss ⇒ an
/// elided coarse interior ⇒ render its coarse block-cube (exactly the record the interior-
/// inclusive oracle build would have carried).
///
/// **Why bitpacked, not more records (the owner's no-dense-grid / no-O(volume)-records law):**
/// storage is one bit per occupied-region block (a `u32[16]` per PRESENT 8-block cell, empty
/// cells stored nothing), i.e. `volume/8` bytes at worst — ~192× leaner than the 24-byte
/// records the surface-only contract deleted, and never a dense whole-region volume. Built
/// from the chunks (a fully-solid chunk sets its `CHUNK_BLOCKS³` bits in bulk, no O(volume)
/// hashing), rebuilt per EDIT like the pyramid — never per band scrub (the band is a uniform).
///
/// The cell keys are the 8-block clip-map cell keys ([`pack_world_block_key`] of
/// `floor_div(block, 8)`), sorted ascending — the same order the shader binary-searches.
///
/// **Storage:** the sorted-key / bitmask / fallback shape is substrate's
/// [`SortedKeyBitmaskMap`] (see the seam comment above); this domain type wraps it, keeping the
/// occupancy vocabulary at the GPU seam (`cell_keys` ∥
/// `cell_masks` ∥ `cell_materials` — the fallback
/// scalar IS the render-cell material colour index) and owning the domain `from_chunks` builder.
/// The bit of a fallback word carrying the interior block's on-face-grid overlay flag, above
/// the material colour index (which is tiny — 0..MATERIAL_COUNT). The interior-elision fallback
/// stores one `u32` per cell, so material and overlay are packed together and split apart at the
/// GPU seam ([`OccupancyCellPod`](crate::brick_raymarch) reads them as two fields). With the
/// scene-wide overlay bool deleted, an interior-elision coarse hit sources its overlay from here.
pub const OCCUPANCY_FALLBACK_OVERLAY_BIT: u32 = 1 << 16;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockOccupancyMasks {
    /// The substrate storage: present 8-block cell keys (sorted ascending) ∥ per-cell `512`-bit
    /// occupancy bitmask (`bit = (local_z*8 + local_y)*8 + local_x`, `local = block.rem_euclid(8)`)
    /// ∥ per-cell fallback WORD: the first occupied block's material colour index packed with its
    /// on-face-grid overlay bit (`OCCUPANCY_FALLBACK_OVERLAY_BIT`) — the coarse-cube's shade AND
    /// overlay when the record-miss fallback fires. Exact for a uniform interior cell (every
    /// current band golden); best-effort where a cell mixes material/overlay (the documented
    /// tolerance edge — the R8 atlas is occupancy-only, so per-interior-block detail would
    /// re-introduce the O(volume) record set this contract deleted).
    map: SortedKeyBitmaskMap<BLOCK_OCCUPANCY_MASK_WORDS>,
}

impl BlockOccupancyMasks {
    /// The empty map (no occupied cells) — the "off" form for the record-sourced /
    /// pyramid-off constructors that never carry an interior signal (they run FULL-band only).
    pub fn empty() -> Self {
        BlockOccupancyMasks::default()
    }

    /// Present 8-block cells' packed keys, sorted strictly ascending (the shader's binary-search
    /// order). Parallel to [`cell_masks`](Self::cell_masks) / [`cell_materials`](Self::cell_materials).
    pub fn cell_keys(&self) -> &[u64] {
        &self.map.keys
    }

    /// Per-cell `512`-bit occupancy bitmasks, one `[u32; 16]` per key, parallel to
    /// [`cell_keys`](Self::cell_keys).
    pub fn cell_masks(&self) -> &[[u32; BLOCK_OCCUPANCY_MASK_WORDS]] {
        &self.map.masks
    }

    /// Per-cell fallback material colour index, parallel to [`cell_keys`](Self::cell_keys).
    pub fn cell_materials(&self) -> &[u32] {
        &self.map.fallbacks
    }

    /// Set one block's bit (and, first-writer-wins, its cell fallback word) in the accumulation
    /// map. The `fallback` word packs the block's material colour index with its overlay bit
    /// (`OCCUPANCY_FALLBACK_OVERLAY_BIT`) — the coarse-cube shade + overlay a record-miss
    /// interior hit resolves to. The cell fold (`floor_div` by the 8-block cell edge) and the
    /// cell-local z-major linear bit index are domain cell geometry; the bit-set on the
    /// fixed-width mask is substrate's [`set_mask_bit`].
    fn insert_block(
        cells: &mut std::collections::BTreeMap<u64, ([u32; BLOCK_OCCUPANCY_MASK_WORDS], u32)>,
        world_block: [i64; 3],
        fallback: u32,
    ) {
        let cell_size = BLOCK_OCCUPANCY_CELL_BLOCKS as i64;
        let cell = [
            world_block[0].div_euclid(cell_size),
            world_block[1].div_euclid(cell_size),
            world_block[2].div_euclid(cell_size),
        ];
        let local = [
            world_block[0].rem_euclid(cell_size) as usize,
            world_block[1].rem_euclid(cell_size) as usize,
            world_block[2].rem_euclid(cell_size) as usize,
        ];
        let bit = (local[2] * BLOCK_OCCUPANCY_CELL_BLOCKS as usize + local[1])
            * BLOCK_OCCUPANCY_CELL_BLOCKS as usize
            + local[0];
        let entry = cells
            .entry(pack_world_block_key(cell))
            .or_insert(([0u32; BLOCK_OCCUPANCY_MASK_WORDS], fallback));
        set_mask_bit(&mut entry.0, bit);
    }

    /// Build the block-granular occupancy map from the two-layer chunks — the interior-elision
    /// companion of [`ClipmapPyramid::from_chunks`], marking EVERY non-air block (coarse-solid or
    /// microblock), so a record-miss inside a band-clipped solid resolves to its coarse cube.
    ///
    /// A fully-solid chunk (all `CHUNK_BLOCKS³` coarse, no microblocks) sets its block bits in
    /// BULK from the chunk's first block colour — one map lookup + a constant bit-set, no
    /// per-block hashing (the interior-elision cost discipline). Partial/boundary chunks set one
    /// bit per occupied block. Cell keys are the same 8-block keys the pyramid's L1 carries.
    pub fn from_chunks(chunks: &[([i32; 3], Arc<TwoLayerChunk>)]) -> Self {
        let mut cells: std::collections::BTreeMap<
            u64,
            ([u32; BLOCK_OCCUPANCY_MASK_WORDS], u32),
        > = std::collections::BTreeMap::new();
        let chunk_blocks = CHUNK_BLOCKS as i64;
        for (chunk_coord, chunk) in chunks {
            let base = [
                chunk_coord[0] as i64 * chunk_blocks,
                chunk_coord[1] as i64 * chunk_blocks,
                chunk_coord[2] as i64 * chunk_blocks,
            ];
            let fully_solid =
                chunk.microblocks.is_empty() && chunk.coarse.iter().all(Option::is_some);
            if fully_solid {
                // Bulk: the whole `CHUNK_BLOCKS³` block box is occupied at the chunk's first
                // block colour + overlay — no per-block visit beyond the constant bit-set (a
                // 4-aligned chunk box lands wholly inside one 8-block cell per axis).
                let material = chunk
                    .coarse_block([0, 0, 0])
                    .map(|block_id| block_id.color_index() as u32)
                    .unwrap_or(0);
                let fallback = material
                    | (u32::from(chunk.coarse_block_overlay([0, 0, 0]))
                        * OCCUPANCY_FALLBACK_OVERLAY_BIT);
                for block_z in 0..CHUNK_BLOCKS {
                    for block_y in 0..CHUNK_BLOCKS {
                        for block_x in 0..CHUNK_BLOCKS {
                            Self::insert_block(
                                &mut cells,
                                [
                                    base[0] + block_x as i64,
                                    base[1] + block_y as i64,
                                    base[2] + block_z as i64,
                                ],
                                fallback,
                            );
                        }
                    }
                }
            } else {
                for block_z in 0..CHUNK_BLOCKS {
                    for block_y in 0..CHUNK_BLOCKS {
                        for block_x in 0..CHUNK_BLOCKS {
                            let block = [block_x, block_y, block_z];
                            // The fallback word: the block's material colour index packed with its
                            // overlay bit (first cuboid's, or the coarse block's overlay marker).
                            let fallback = if let Some(block_id) = chunk.coarse_block(block) {
                                block_id.color_index() as u32
                                    | (u32::from(chunk.coarse_block_overlay(block))
                                        * OCCUPANCY_FALLBACK_OVERLAY_BIT)
                            } else if let Some(geometry) = chunk.microblocks.get(&block) {
                                geometry
                                    .cuboids
                                    .first()
                                    .map(|cuboid| {
                                        let key = CellKey::from_raw(cuboid.material_id());
                                        key.block_id() as u32
                                            | (u32::from(key.has_overlay())
                                                * OCCUPANCY_FALLBACK_OVERLAY_BIT)
                                    })
                                    .unwrap_or(0)
                            } else {
                                continue;
                            };
                            Self::insert_block(
                                &mut cells,
                                [
                                    base[0] + block_x as i64,
                                    base[1] + block_y as i64,
                                    base[2] + block_z as i64,
                                ],
                                fallback,
                            );
                        }
                    }
                }
            }
        }
        // The accumulation is done: hand the cells to the substrate constructor as
        // `(key, mask, fallback)` triples. The `BTreeMap` already drained them strictly
        // ascending by key and unique, so use the no-re-sort constructor — the stored shape
        // is byte-identical to a `from_triples` sort of the same already-ordered input.
        let triples: Vec<(u64, [u32; BLOCK_OCCUPANCY_MASK_WORDS], u32)> = cells
            .into_iter()
            .map(|(key, (mask, material))| (key, mask, material))
            .collect();
        BlockOccupancyMasks {
            map: SortedKeyBitmaskMap::from_sorted_unique_triples(triples),
        }
    }

    /// The present-cell count (== the shader's occupancy binary-search span; 0 ⇒ the
    /// band-clip interior fallback never fires).
    pub fn cell_count(&self) -> u32 {
        self.map.len() as u32
    }

    /// Whether the map holds no occupied cell — the "off" form (the band-clip interior fallback
    /// never fires). A thin delegate over the substrate map so consumers use the accessor surface,
    /// not the private storage.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Whether the block at cell-local `linear_index` is occupied in the cell `cell_key` — a
    /// delegate over the substrate map's [`SortedKeyBitmaskMap::contains_bit`] (binary-search the
    /// key, then test the bit). `false` for an absent cell. The occupancy read the raymarch mirror
    /// resolves a record-miss fallback with.
    pub fn contains_bit(&self, cell_key: u64, linear_index: usize) -> bool {
        self.map.contains_bit(cell_key, linear_index)
    }
}

/// What a brick holds — the record kinds of the brick partition. The enum makes "a coarse
/// record consumes no atlas slot" and "only a MIXED block owns a cell-key tile" structural,
/// not conventions.
///
/// A sculpted block is **uniform** when every one of its microblock cuboids carries the same
/// cell key (block-palette id + on-face-grid overlay bit) — then the key lives once, on the
/// record ([`BrickRecord::material_id`] + [`BrickRecord::overlay`]). It is **mixed** when the
/// cuboids disagree; then its per-voxel keys live in a cell-key tile of the separately-pooled
/// material side atlas, and the record's own material/overlay are don't-care. See
/// docs/architecture/03-display.md (the brick-field atlas).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPayload {
    /// **Kind 0** — an analytic coarse brick: the whole block is solid at `block_id`,
    /// stored as this one record with no per-voxel data (interior elision on the GPU;
    /// also the residency-miss fallback form the G1 contract renders).
    CoarseSolid { block_id: BlockId },
    /// **Kind 1** — a sculpted brick whose voxels all share ONE cell key: the block's voxel
    /// occupancy lives in atlas slot `atlas_slot` (an `edge³` R8 tile, edge =
    /// `voxels_per_block`); its material + overlay live on the record.
    Sculpted { atlas_slot: u32 },
    /// **Kind 1, mixed** — a sculpted brick whose microblocks disagree on their cell key:
    /// occupancy in `atlas_slot` exactly as [`Sculpted`](Self::Sculpted), PLUS a per-voxel
    /// cell-key tile in `cell_key_slot` of the material side atlas (an independent pool with
    /// its own free-list — a cell-key slot number is unrelated to an occupancy slot number).
    SculptedMixed {
        atlas_slot: u32,
        cell_key_slot: u32,
    },
}

impl BrickPayload {
    /// The GPU-side record-kind discriminant: **0** = coarse, **1** = sculpted-uniform,
    /// **2** = sculpted-MIXED. Pinned here — like `shape_kind_discriminant` — so a future enum
    /// reorder can't silently desync the shader: `pack_gpu_records` packs THIS value into the
    /// GPU record's `kind` bits and the WGSL decodes it there.
    ///
    /// Kinds 1 and 2 TRAVERSE identically (both descend into an occupancy atlas slot); they
    /// differ only in where the hit's SHADE comes from — the record's own material + overlay
    /// (1), or the per-voxel cell-key texel of the material side atlas (2).
    pub fn kind_discriminant(&self) -> u32 {
        match self {
            BrickPayload::CoarseSolid { .. } => 0,
            BrickPayload::Sculpted { .. } => 1,
            BrickPayload::SculptedMixed { .. } => 2,
        }
    }

    /// The occupancy atlas slot of a sculpted brick (uniform or mixed); `None` for a coarse
    /// record (which consumes no slot). The ONE reader of "does this record own an occupancy
    /// tile", so a new sculpted arm can never be missed by a slot-bookkeeping site.
    pub fn occupancy_atlas_slot(&self) -> Option<u32> {
        match *self {
            BrickPayload::CoarseSolid { .. } => None,
            BrickPayload::Sculpted { atlas_slot }
            | BrickPayload::SculptedMixed { atlas_slot, .. } => Some(atlas_slot),
        }
    }

    /// The material side-atlas slot holding this brick's per-voxel cell-key tile — `Some`
    /// only for a MIXED sculpted brick (a uniform or coarse block carries its one cell key on
    /// the record and owns no tile).
    pub fn cell_key_slot(&self) -> Option<u32> {
        match *self {
            BrickPayload::SculptedMixed { cell_key_slot, .. } => Some(cell_key_slot),
            _ => None,
        }
    }
}

/// One resident brick: a non-air block of the two-layer boundary set, keyed for the
/// sorted-array binary search the G1 raymarch resolves residency with (ADR 0011 4b).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickRecord {
    /// [`pack_world_block_key`] of the block's absolute world-block coordinate.
    pub packed_world_block_key: u64,
    /// The block's clean render-cell material colour index (`0..MATERIAL_COUNT`) — the
    /// `block_id`'s colour index for a coarse block, the single microblock material for a
    /// UNIFORM boundary block. The occupancy atlas is occupancy-only, so this is the
    /// per-BLOCK material the raymarch shades with, packed into the GPU record's `kind`
    /// high bits by `pack_gpu_records`. **Don't-care for a
    /// [`SculptedMixed`](BrickPayload::SculptedMixed) block** — its per-voxel keys are the
    /// truth (this holds the first cuboid's clean id there, never read as the block's).
    pub material_id: u16,
    /// The block's on-face-grid overlay bit — the other half of its cell key (the render-cell
    /// key is `material_id | overlay`, see [`voxel_core::core_geom::CellKey`]). Carried
    /// per-RECORD so a scene whose blocks disagree on the overlay is still one brick field.
    /// Meaningful for coarse + UNIFORM sculpted blocks; don't-care for a
    /// [`SculptedMixed`](BrickPayload::SculptedMixed) block (its tile's per-voxel keys carry
    /// the overlay bit themselves).
    pub overlay: bool,
    /// Coarse (kind 0), sculpted-uniform or sculpted-mixed (kind 1) — see [`BrickPayload`].
    pub payload: BrickPayload,
    /// Per-face seam-solidity flags, carried UNCHANGED from the boundary set for a
    /// sculpted brick. A coarse-solid block is solid through, so every face flag is
    /// `true` by construction (the block-DDA culls against it identically either way).
    pub seam_solidity: SeamSolidity,
}

/// The built brick field: the sorted record array + the sculpted-brick occupancy atlas
/// bytes in the ADR 0007 tile-cube layout (`bricks_per_axis³` slots of `edge³` texels,
/// linear slot index → 3D tile coord exactly as `upload_grid_per_chunk` packs fog
/// tiles). [`upload_brick_atlas`] lands the bytes in an R8 3D texture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrickFieldBuild {
    /// Every non-air block's record, sorted strictly ascending by
    /// `packed_world_block_key` (unique — a block is coarse XOR boundary).
    pub brick_records: Vec<BrickRecord>,
    /// `atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed;
    /// tile slots past the last sculpted brick stay all-zero.
    pub sculpted_atlas_bytes: Vec<u8>,
    /// The MIXED bricks' per-voxel cell-key tiles, indexed by the `cell_key_slot` their
    /// records carry (dense `0..mixed_count` in this build's traversal order). EMPTY for a
    /// scene whose every sculpted block is uniform — the sparse-side-atlas contract: only a
    /// block that mixes cell keys pays per-voxel material cost.
    ///
    /// Unlike the occupancy atlas, these are **not** packed to a byte blob here: the material
    /// side atlas has no GPU pool yet, so the CPU mirror is the only consumer and the tiles
    /// travel as tiles (the single-owner tile law — moved into
    /// [`IncrementalBrickField`], never cloned per edit).
    pub cell_key_tiles: Vec<BrickCellKeyTile>,
    /// The brick edge in voxels — `voxels_per_block`, the ONE-BLOCK granule
    /// (ADR 0011 Decision 1). Block-denominated: never a hard-coded voxel count.
    pub brick_edge_voxels: u32,
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(sculpted_count))`).
    pub bricks_per_axis: u32,
    /// `bricks_per_axis * brick_edge_voxels` — the atlas texture dimension per axis
    /// (0 when the build has no sculpted brick).
    pub atlas_dim_voxels: u32,
}

/// The GPU upload payload for the sculpted-brick atlas — the ONE place the flat R8 byte
/// blob still lives after item 9's single-owner rework (see `docs/architecture/`, the
/// brick-field display chapter). A wholesale build hands this to
/// [`BrickRaymarchRenderer::install_brick_field`](crate::brick_raymarch::BrickRaymarchRenderer::install_brick_field)
/// by MOVE ([`IncrementalBrickField::from_wholesale`]); the incremental patch path never
/// materialises one except on the legitimate atlas-grow re-pack
/// ([`IncrementalBrickField::pack_atlas_payload`]). Carries the atlas GEOMETRY alongside
/// the bytes so the install seam sets its frame scalars without a `BrickFieldBuild`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SculptedAtlasPayload {
    /// `geometry.atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed —
    /// the bytes [`upload_brick_atlas`] lands in the R8 3D texture.
    pub bytes: Vec<u8>,
    /// The atlas tile geometry (tile-grid edge, texture dimension, brick edge) — shared with
    /// the incremental owner's [`IncrementalBrickField::atlas_geometry`] so the two never
    /// drift on the tile-cube math.
    pub geometry: SculptedAtlasGeometry,
    /// Live sculpted-brick count (the wholesale install's `last_atlas_slots_written`).
    pub sculpted_slot_count: u32,
}

/// The sculpted atlas's tile geometry — `bricks_per_axis` / `atlas_dim_voxels` / brick edge,
/// factored so the incremental owner and the packer never drift on the tile-cube math.
/// ([`IncrementalBrickField::atlas_geometry`].)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SculptedAtlasGeometry {
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(slot_high_water))`).
    pub bricks_per_axis: u32,
    /// The atlas texture dimension per axis (`bricks_per_axis * brick_edge_voxels`).
    pub atlas_dim_voxels: u32,
    /// The brick edge in voxels (`voxels_per_block`).
    pub brick_edge_voxels: u32,
}

/// The GPU upload payload for the **material side atlas**: the MIXED bricks' per-voxel
/// cell-key tiles packed into one 16-bit-texel cube, landed by
/// [`upload_brick_cell_key_atlas`] in an R16Uint 3D texture. The sibling of
/// [`SculptedAtlasPayload`] — a SECOND, independently pooled atlas (its own slot numbering,
/// its own free-list, its own tile-grid edge), sparse by construction: only a block whose
/// microblocks disagree on their cell key owns a slot here, so a scene of uniform blocks packs
/// ZERO bytes. See docs/architecture/03-display.md (the brick-field atlas).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SculptedCellKeyAtlasPayload {
    /// `2 · geometry.atlas_dim_voxels³` bytes: one **little-endian u16 cell key per voxel**
    /// (clean block-palette id + the on-face-grid overlay bit, verbatim — no indirection, no
    /// per-brick palette). An air voxel's texel is a documented don't-care (occupancy gates
    /// every read). EMPTY when no brick is mixed.
    pub bytes: Vec<u8>,
    /// The side atlas's OWN tile geometry — derived from the cell-key slot count, never from
    /// the occupancy pool's.
    pub geometry: SculptedCellKeyAtlasGeometry,
    /// Live cell-key slot count (== the mixed-brick count of a wholesale build).
    pub cell_key_slot_count: u32,
}

/// The material side atlas's tile geometry — the twin of [`SculptedAtlasGeometry`] computed
/// from the CELL-KEY slot count (the pools size independently: a scene of 10k sculpted bricks
/// with 3 mixed ones has a 22-tile occupancy grid and a 2-tile material grid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SculptedCellKeyAtlasGeometry {
    /// Cell-key tile slots per atlas axis (`ceil(cbrt(cell_key_slot_high_water))`).
    pub bricks_per_axis: u32,
    /// The side atlas's texture dimension per axis in voxels (`bricks_per_axis *
    /// brick_edge_voxels`); 0 when no brick is mixed.
    pub atlas_dim_voxels: u32,
    /// The brick edge in voxels (`voxels_per_block`) — the same granule the occupancy atlas
    /// tiles at (one cell-key texel per occupancy voxel, cell-for-cell).
    pub brick_edge_voxels: u32,
}

impl SculptedCellKeyAtlasPayload {
    /// The side atlas of a field with NO mixed brick: zero bytes, zero slots, zero dimension —
    /// what every scene the representability gate admits today packs to.
    pub fn empty(brick_edge_voxels: u32) -> Self {
        Self {
            bytes: Vec::new(),
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis: 0,
                atlas_dim_voxels: 0,
                brick_edge_voxels: brick_edge_voxels.max(1),
            },
            cell_key_slot_count: 0,
        }
    }
}

/// Pack a slot-indexed set of cell-key tiles into the side atlas's little-endian u16 texel cube
/// — substrate's [`CubeTilePacking::pack_u16_value_cubes`], the payload sibling of the
/// occupancy scatter, at its own (cell-key) slot count. Shared by the wholesale build and
/// [`IncrementalBrickField::pack_cell_key_atlas_payload`] so the two are byte-identical for the
/// same tile vector. A FREED (dead) slot's tile is scattered as-is: unreachable from any live
/// record, so its texels may be garbage.
fn pack_cell_key_atlas(
    slot_tiles: &[BrickCellKeyTile],
    brick_edge_voxels: u32,
) -> (u32, u32, Vec<u8>) {
    CubeTilePacking::pack_u16_value_cubes(slot_tiles, brick_edge_voxels)
}

/// The occupancy byte a solid voxel packs to — the fog atlas's 0/255 R8 convention. Injected
/// into [`BrickOccupancyTile::expand_to_bytes`] / [`CubeTilePacking::pack_bit_cubes`] at the
/// atlas seam (substrate names no such byte — a set bit reads as whatever the caller passes).
const SCULPTED_BRICK_OCCUPIED: u8 = 255;

impl BrickFieldBuild {
    /// Materialise this build's sculpted atlas as an upload [`SculptedAtlasPayload`] — the
    /// wholesale-build → install adapter for the callers that keep the `BrickFieldBuild`
    /// around (the `shot` golden tool and the parity tests). CLONES the atlas bytes; the
    /// live worker/orchestrator paths move them instead via
    /// [`IncrementalBrickField::from_wholesale`].
    pub fn atlas_payload(&self) -> SculptedAtlasPayload {
        SculptedAtlasPayload {
            bytes: self.sculpted_atlas_bytes.clone(),
            geometry: SculptedAtlasGeometry {
                bricks_per_axis: self.bricks_per_axis,
                atlas_dim_voxels: self.atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            sculpted_slot_count: self.sculpted_brick_count() as u32,
        }
    }

    /// Materialise this build's MATERIAL SIDE ATLAS as an upload
    /// [`SculptedCellKeyAtlasPayload`] — the second pool's install adapter, packed from the
    /// mixed bricks' cell-key tiles at their own dense slot numbering. A build with no mixed
    /// brick yields the empty payload (zero bytes: the sparse-side-atlas contract).
    pub fn cell_key_atlas_payload(&self) -> SculptedCellKeyAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_cell_key_atlas(&self.cell_key_tiles, self.brick_edge_voxels);
        SculptedCellKeyAtlasPayload {
            bytes,
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            cell_key_slot_count: self.mixed_brick_count() as u32,
        }
    }

    /// Resolve the record for an absolute world-block coordinate by binary search over
    /// the sorted array — the CPU mirror of the in-shader residency lookup (ADR 0011
    /// 4b), and the parity harness's per-block accessor. `None` = air.
    pub fn find_record(&self, world_block: [i64; 3]) -> Option<&BrickRecord> {
        let key = pack_world_block_key(world_block);
        self.brick_records
            .binary_search_by_key(&key, |record| record.packed_world_block_key)
            .ok()
            .map(|index| &self.brick_records[index])
    }

    /// How many records are sculpted bricks — uniform AND mixed (== occupancy atlas slots in
    /// use; slots are assigned densely `0..count`).
    pub fn sculpted_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| record.payload.occupancy_atlas_slot().is_some())
            .count()
    }

    /// How many records are MIXED sculpted bricks (== cell-key tiles, i.e. material
    /// side-atlas slots in use; densely `0..count` in a wholesale build).
    pub fn mixed_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| record.payload.cell_key_slot().is_some())
            .count()
    }

    /// The low-corner texel of `atlas_slot`'s tile in the atlas cube (linear slot →
    /// 3D tile coord, x-fastest — the `upload_grid_per_chunk` tile layout).
    fn atlas_slot_origin_texels(&self, atlas_slot: u32) -> [usize; 3] {
        let tiles = self.bricks_per_axis.max(1);
        let tile = [
            atlas_slot % tiles,
            (atlas_slot / tiles) % tiles,
            atlas_slot / (tiles * tiles),
        ];
        let edge = self.brick_edge_voxels as usize;
        [
            tile[0] as usize * edge,
            tile[1] as usize * edge,
            tile[2] as usize * edge,
        ]
    }

    /// Copy one sculpted brick's `edge³` occupancy bytes out of the atlas (block-local
    /// x-fastest order — the order the boundary set's per-block occupancy expands in).
    pub fn sculpted_brick_occupancy(&self, atlas_slot: u32) -> Vec<u8> {
        let edge = self.brick_edge_voxels as usize;
        let atlas_dim = self.atlas_dim_voxels as usize;
        let origin = self.atlas_slot_origin_texels(atlas_slot);
        let mut brick_bytes = vec![0u8; edge * edge * edge];
        for local_z in 0..edge {
            for local_y in 0..edge {
                for local_x in 0..edge {
                    let atlas_index = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                        * atlas_dim
                        + origin[0]
                        + local_x;
                    brick_bytes[(local_z * edge + local_y) * edge + local_x] =
                        self.sculpted_atlas_bytes[atlas_index];
                }
            }
        }
        brick_bytes
    }
}

/// Build the **surface-only** brick field from a scene's two-layer boundary set (the
/// `build_covering_chunks` / resident-cache output): walk every chunk's block partition,
/// emit one record per SURFACE non-air block — a fully-occluded coarse interior block
/// emits nothing (ADR 0011 interior elision, fused into the build via
/// [`BrickOcclusionOracle`]; interiors stay queryable through the chunks) — rasterize each
/// boundary block's cuboids into its atlas slot, and sort the records by packed
/// world-block key.
///
/// **O(surface), not O(volume) (the 8000³-freeze fix):** an all-interior chunk (fully
/// solid, fully-solid face-neighbours) is skipped whole without visiting its blocks, so a
/// 125M-block solid emits ~1.5M records and the build touches only the ~1-chunk-thick
/// boundary shell. Every consumer downstream (sort, GPU pack, incremental mirror clone)
/// inherits the ∝-surface cost. The interior-INCLUSIVE build survives as
/// [`build_brick_field_all_blocks`], the parity oracle.
///
/// `voxels_per_block` is the document density every chunk was built at (each chunk
/// carries it; a mismatch is a caller bug, asserted in debug).
///
/// **Why the classify pass stays SERIAL (measured).** The per-block classify + slot
/// assignment is coarse-dominated and memory-bound, so a rayon per-chunk split measured NO
/// net win: the parallel classify gain was cancelled by the extra ordered-merge pass needed
/// to keep the sculpted atlas-slot numbering byte-identical (slots are assigned in
/// traversal order — ADR 0011 G3's incremental-atlas contract — so a parallel build must
/// re-derive that exact order, adding an O(records) merge). Only the final key sort and the
/// oracle's chunk classification are worth parallelising. The record ORDER + sculpted slot
/// numbering are produced by the same serial traversal as the oracle build — sculpted slots
/// bit-for-bit identical (the sculpted set is never elided).
pub fn build_brick_field(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> BrickFieldBuild {
    // The build-only entry (lib re-export; the golden `shot` tool, parity/perf tests, the
    // non-tile-carrying orchestrator/startup paths). Drops the rasterised tiles; the live
    // worker/orchestrator wholesale path calls `build_brick_field_with_tiles` to keep and
    // MOVE them into the mirror (skipping the from-atlas-bytes re-derive).
    build_brick_field_with_tiles(two_layer_chunks, voxels_per_block).0
}

/// Like [`build_brick_field`] but ALSO returns the per-sculpted-slot occupancy tiles it
/// rasterised (dense slot order — the `atlas_slot` numbering baked into the records), so a
/// wholesale reset can MOVE them straight into the incremental mirror
/// ([`IncrementalBrickField::from_wholesale_with_tiles`]) instead of re-gathering + re-bit-
/// packing them out of the flat atlas bytes the packer just produced. The `BrickFieldBuild`
/// is byte-identical to [`build_brick_field`]'s (same records, same packed atlas bytes) —
/// this only hands back the intermediate the plain entry discards.
pub fn build_brick_field_with_tiles(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> (BrickFieldBuild, Vec<BrickOccupancyTile>) {
    let brick_edge_voxels = voxels_per_block.max(1);
    let oracle = BrickOcclusionOracle::new(two_layer_chunks);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One bit-packed `edge²`-word tile per sculpted brick, in slot order; unpacked into
    // the atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<BrickOccupancyTile> = Vec::new();
    // One `edge³` cell-key tile per MIXED sculpted brick, in cell-key-slot order — an
    // INDEPENDENT dense numbering (a mixed brick's two slots are unrelated numbers).
    let mut cell_key_tiles: Vec<BrickCellKeyTile> = Vec::new();

    for (chunk_coord, chunk) in two_layer_chunks {
        debug_assert_eq!(
            chunk.voxels_per_block, brick_edge_voxels,
            "every chunk of one build shares the document density"
        );
        // Interior-chunk fast path (∝ surface, the 8000³-freeze fix): a fully-solid chunk
        // ringed by fully-solid face-neighbours is all-occluded — emit NOTHING for it
        // without visiting a single block. An interior chunk has no microblocks by
        // definition, so no sculpted record is skipped here.
        if oracle.chunk_is_all_interior(*chunk_coord) {
            continue;
        }
        let occlusion = oracle.context_for_chunk(*chunk_coord, chunk.as_ref());
        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let world_block = [
                        chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                        chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                        chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                    ];
                    // Classify the block once (shared with the G3 incremental update so
                    // both paths emit identical records); the wholesale build assigns
                    // sculpted slots densely in record order.
                    match classify_block_brick(chunk, block, world_block, brick_edge_voxels) {
                        BlockBrick::Air => {}
                        // A coarse-solid block emits ONLY when a ray could reach it: the
                        // fused interior elision (never emitted ⇒ never sorted, packed,
                        // uploaded). Interiors stay queryable through the chunks.
                        BlockBrick::Coarse(record) => {
                            if !occlusion.coarse_block_occluded(block) {
                                brick_records.push(record);
                            }
                        }
                        // A boundary (sculpted) block is surface by definition here: its
                        // record — and thus the atlas + fog tile set — is NEVER elided,
                        // so sculpted slot numbering matches the interior-inclusive
                        // oracle build tile-for-tile.
                        BlockBrick::Sculpted {
                            material_id,
                            overlay,
                            seam_solidity,
                            tile,
                            cell_keys,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                overlay,
                                payload: sculpted_payload_dense(
                                    atlas_slot,
                                    cell_keys,
                                    &mut cell_key_tiles,
                                ),
                                seam_solidity,
                            });
                        }
                    }
                }
            }
        }
    }

    // The keys are UNIQUE (each world block appears in exactly one chunk — asserted below),
    // so a parallel unstable sort yields the byte-identical order a serial sort would, at any
    // thread count. (A filtered emission of the serial traversal stays traversal-ordered, so
    // this is the same sort the interior-inclusive build performs — the shader binary search
    // and the G3 patch protocol see a sorted, unique array either way.)
    brick_records.par_sort_unstable_by_key(|record| record.packed_world_block_key);
    debug_assert!(
        brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "brick keys must be unique (each world block appears in exactly one chunk)"
    );

    let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
        pack_sculpted_atlas(&sculpted_brick_tiles, brick_edge_voxels);

    let build = BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        cell_key_tiles,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    };
    (build, sculpted_brick_tiles)
}

/// Assign a sculpted block's payload in a WHOLESALE build's dense numbering: the occupancy
/// slot is the caller's (already pushed), and a MIXED block's cell-key tile appends to
/// `cell_key_tiles` at the next dense material slot. The two pools are independent — a mixed
/// brick's `atlas_slot` and `cell_key_slot` are unrelated numbers. Shared by the surface-only
/// build and the interior-inclusive oracle build so both number the pools identically.
fn sculpted_payload_dense(
    atlas_slot: u32,
    cell_keys: Option<BrickCellKeyTile>,
    cell_key_tiles: &mut Vec<BrickCellKeyTile>,
) -> BrickPayload {
    match cell_keys {
        None => BrickPayload::Sculpted { atlas_slot },
        Some(tile) => {
            let cell_key_slot = cell_key_tiles.len() as u32;
            cell_key_tiles.push(tile);
            BrickPayload::SculptedMixed {
                atlas_slot,
                cell_key_slot,
            }
        }
    }
}

/// The interior-INCLUSIVE brick-field build: one record per NON-AIR block (coarse-solid or
/// boundary), the pre-elision reference. **Oracle only** — the live sink uses the surface-only
/// [`build_brick_field`]; this stays as the parity oracle for the interior-elision gates
/// (`brick_surface_elision_hit_set_unchanged`, `clipmap_from_chunks_equals_from_full_records`)
/// and any consumer that genuinely needs every block (none on the live path).
pub fn build_brick_field_all_blocks(
    two_layer_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    voxels_per_block: u32,
) -> BrickFieldBuild {
    let brick_edge_voxels = voxels_per_block.max(1);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One bit-packed `edge²`-word tile per sculpted brick, in slot order; unpacked into
    // the atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<BrickOccupancyTile> = Vec::new();
    // One `edge³` cell-key tile per MIXED sculpted brick, in cell-key-slot order.
    let mut cell_key_tiles: Vec<BrickCellKeyTile> = Vec::new();

    for (chunk_coord, chunk) in two_layer_chunks {
        debug_assert_eq!(
            chunk.voxels_per_block, brick_edge_voxels,
            "every chunk of one build shares the document density"
        );
        for block_z in 0..CHUNK_BLOCKS {
            for block_y in 0..CHUNK_BLOCKS {
                for block_x in 0..CHUNK_BLOCKS {
                    let block = [block_x, block_y, block_z];
                    let world_block = [
                        chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                        chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                        chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                    ];
                    // Classify the block once (shared with the G3 incremental update so
                    // both paths emit identical records); the wholesale build assigns
                    // sculpted slots densely in record order.
                    match classify_block_brick(chunk, block, world_block, brick_edge_voxels) {
                        BlockBrick::Air => {}
                        BlockBrick::Coarse(record) => brick_records.push(record),
                        BlockBrick::Sculpted {
                            material_id,
                            overlay,
                            seam_solidity,
                            tile,
                            cell_keys,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                overlay,
                                payload: sculpted_payload_dense(
                                    atlas_slot,
                                    cell_keys,
                                    &mut cell_key_tiles,
                                ),
                                seam_solidity,
                            });
                        }
                    }
                }
            }
        }
    }

    // The keys are UNIQUE (each world block appears in exactly one chunk — asserted below),
    // so a parallel unstable sort yields the byte-identical order a serial sort would, at any
    // thread count. This is the one part of the build that measurably parallelises.
    brick_records.par_sort_unstable_by_key(|record| record.packed_world_block_key);
    debug_assert!(
        brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "brick keys must be unique (each world block appears in exactly one chunk)"
    );

    // Tile geometry mirrors `upload_grid_per_chunk`: a cubic-ish slot grid bounded by
    // the SCULPTED count (coarse records consume none of it), then scatter each tile.
    let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
        pack_sculpted_atlas(&sculpted_brick_tiles, brick_edge_voxels);

    BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        cell_key_tiles,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
    }
}

/// The six face-neighbour chunk offsets — the reach of a block's occlusion verdict at the
/// chunk granularity (a block's six face-neighbours land in its own chunk or one of these).
const FACE_NEIGHBOUR_CHUNK_OFFSETS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// **The occlusion oracle over a two-layer covering set (ADR 0011 interior elision — the
/// brick sink's analogue of the mesh's interior-face culling).** Decides which coarse-solid
/// blocks are FULLY OCCLUDED — all six face-neighbours present AND solid on the shared face
/// — so [`build_brick_field`] / [`IncrementalBrickField::apply_dirty_update`] can fuse the
/// interior elision INTO record emission (the record set is surface-only by construction;
/// no post-hoc mask pass over an O(volume) record array).
///
/// A fully-occluded block is never a ray's first hit: the block-DDA
/// ([`cpu_march_brick_field`](crate::brick_raymarch::cpu_march_brick_field)) returns at the
/// FIRST block carrying a record, and a ray reaching an occluded block must first pass
/// through the solid neighbour surrounding it (which keeps its record). So never emitting it
/// is **hit-identical** — proven against the interior-inclusive oracle build in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`.
///
/// **Chunk-level fast path ([`Self::chunk_is_all_interior`]):** a chunk that is itself fully
/// coarse-solid AND whose six FACE-neighbour chunks are all fully coarse-solid has every one
/// of its blocks occluded (each block's six neighbours land in this chunk or a full
/// neighbour, all solid) — the builder skips the whole chunk with one set lookup, visiting
/// none of its blocks. Only the ~1-chunk-thick boundary shell (and any chunk carrying
/// microblocks) does per-block work, so the build is ∝ surface, not volume.
///
/// **Conservative direction:** a neighbour that is ABSENT (air) or only PARTIALLY solid on
/// the shared face keeps the block. The emitted set is thus always a superset of the
/// truly-visible blocks — elision can never drop a block a ray can see.
struct BrickOcclusionOracle<'a> {
    /// Every covering chunk by absolute chunk coord (the neighbour-resolution index).
    chunk_by_coord: std::collections::HashMap<[i32; 3], &'a TwoLayerChunk>,
    /// Chunks that are fully coarse-solid AND ringed by fully coarse-solid face-neighbours —
    /// every block of these is provably occluded (the bulk fast path).
    interior_chunk: std::collections::HashSet<[i32; 3]>,
}

impl<'a> BrickOcclusionOracle<'a> {
    /// Classify the chunk set once (parallel — the full-solidity scan is a pure per-chunk
    /// fold, and set membership is order-free).
    fn new(chunks: &'a [([i32; 3], Arc<TwoLayerChunk>)]) -> Self {
        let chunk_by_coord: std::collections::HashMap<[i32; 3], &TwoLayerChunk> = chunks
            .iter()
            .map(|(coord, chunk)| (*coord, chunk.as_ref()))
            .collect();
        // A chunk is "full-solid" iff every one of its CHUNK_BLOCKS³ blocks is coarse-solid
        // and it carries no microblocks — then every block of it is solid on every face.
        let full_solid: std::collections::HashSet<[i32; 3]> = chunks
            .par_iter()
            .filter(|(_, chunk)| {
                chunk.microblocks.is_empty() && chunk.coarse.iter().all(Option::is_some)
            })
            .map(|(coord, _)| *coord)
            .collect();
        let interior_chunk: std::collections::HashSet<[i32; 3]> = full_solid
            .par_iter()
            .filter(|coord| {
                FACE_NEIGHBOUR_CHUNK_OFFSETS.iter().all(|d| {
                    full_solid.contains(&[coord[0] + d[0], coord[1] + d[1], coord[2] + d[2]])
                })
            })
            .copied()
            .collect();
        Self {
            chunk_by_coord,
            interior_chunk,
        }
    }

    /// Whether every block of `chunk_coord` is provably occluded (the bulk fast path): the
    /// chunk and its six face-neighbours are all fully coarse-solid. The builder emits
    /// nothing for such a chunk without visiting a single block.
    fn chunk_is_all_interior(&self, chunk_coord: [i32; 3]) -> bool {
        self.interior_chunk.contains(&chunk_coord)
    }

    /// The per-chunk occlusion context: this chunk plus its six face-neighbour chunk refs,
    /// hoisted ONCE per chunk so the per-block six-neighbour test needs no hashing.
    fn context_for_chunk(
        &self,
        chunk_coord: [i32; 3],
        chunk: &'a TwoLayerChunk,
    ) -> ChunkOcclusionContext<'a> {
        // [axis][side]: side 0 = the low-face neighbour (coord − 1), side 1 = high (+1).
        let mut face_neighbours: [[Option<&TwoLayerChunk>; 2]; 3] = [[None; 2]; 3];
        for (axis, sides) in face_neighbours.iter_mut().enumerate() {
            for (side, slot) in sides.iter_mut().enumerate() {
                let mut coord = chunk_coord;
                coord[axis] += if side == 0 { -1 } else { 1 };
                *slot = self.chunk_by_coord.get(&coord).copied();
            }
        }
        ChunkOcclusionContext {
            chunk,
            face_neighbours,
        }
    }
}

/// One chunk's occlusion window: the chunk itself + its six face-neighbour chunks (resolved
/// once — see [`BrickOcclusionOracle::context_for_chunk`]). Answers the per-block
/// six-neighbour occlusion test in O(1) chunk resolution (a block's neighbours land in this
/// chunk or a face-adjacent one, never farther).
struct ChunkOcclusionContext<'a> {
    chunk: &'a TwoLayerChunk,
    /// `[axis][side]`: side 0 = the low-face neighbour chunk, 1 = high. `None` = absent (air).
    face_neighbours: [[Option<&'a TwoLayerChunk>; 2]; 3],
}

impl ChunkOcclusionContext<'_> {
    /// Whether the coarse-solid block at chunk-local `block` is FULLY OCCLUDED: each axis
    /// capped on BOTH sides — the +1 neighbour's LOW face covers this block's HIGH face, and
    /// the −1 neighbour's HIGH face covers its LOW. Occluded ⇒ no record is emitted.
    fn coarse_block_occluded(&self, block: [u32; 3]) -> bool {
        (0..3).all(|axis| {
            self.neighbour_face_solid(block, axis, 1) && self.neighbour_face_solid(block, axis, -1)
        })
    }

    /// Is the neighbour of chunk-local `block` across `(axis, delta)` present AND solid on
    /// the face it shares with `block`? A coarse-solid neighbour is solid on every face; a
    /// boundary neighbour consults its per-face seam flag; an air block / absent chunk is
    /// not solid (the conservative direction). Semantics identical to resolving through the
    /// absolute-coordinate chunk map — only the chunk lookup is hoisted.
    fn neighbour_face_solid(&self, block: [u32; 3], axis: usize, delta: i64) -> bool {
        // The face the NEIGHBOUR shares with `block`: stepping +1 lands on the neighbour's
        // LOW face (side 0); stepping −1 on its HIGH face (side 1).
        let facing_side = if delta > 0 { 0 } else { 1 };
        let stepped = block[axis] as i64 + delta;
        let mut local = block;
        let neighbour_chunk = if (0..CHUNK_BLOCKS as i64).contains(&stepped) {
            local[axis] = stepped as u32;
            Some(self.chunk)
        } else {
            local[axis] = stepped.rem_euclid(CHUNK_BLOCKS as i64) as u32;
            self.face_neighbours[axis][if delta > 0 { 1 } else { 0 }]
        };
        let Some(chunk) = neighbour_chunk else {
            return false;
        };
        if chunk.coarse_block(local).is_some() {
            true
        } else if let Some(geometry) = chunk.microblocks.get(&local) {
            geometry.seam_solidity.face_is_solid(axis, facing_side)
        } else {
            false
        }
    }
}

// The occupancy tile itself is substrate's `BitCube` (aliased to `BrickOccupancyTile` at the
// top-of-module seam): edge-≤64 word-packed 3D bitset, one voxel per bit. `empty`, `set_x_run`,
// `expand_to_bytes(byte)`, `from_bytes`, `is_set`, `popcount`, `edge()` live there. The atlas
// seam injects `SCULPTED_BRICK_OCCUPIED` as the "set-bit byte"; substrate names no such byte.

/// The single cell key shared by every cuboid of a boundary block, or `None` when they
/// disagree — the **uniform vs MIXED** classification, made at emission (the one place that
/// decides whether a block owns a cell-key tile). An empty block (no cuboids) is trivially
/// uniform at the fallback key `0`.
fn uniform_cell_key(geometry: &evaluation::two_layer_store::MicroblockGeometry) -> Option<u16> {
    let mut cuboids = geometry.cuboids.iter();
    let first = match cuboids.next() {
        Some(cuboid) => cuboid.material_id(),
        None => return Some(AIR_CELL_KEY_DONT_CARE),
    };
    cuboids
        .all(|cuboid| cuboid.material_id() == first)
        .then_some(first)
}

/// Rasterize one boundary block's cuboids into its `edge³` occupancy tile (block-local
/// x-fastest) and — for a MIXED block only — its per-voxel cell-key tile, in ONE walk over
/// the cuboids (the occupancy bit and the cell key of a voxel are written by the same X-run;
/// the tiles share their row layout, so a second pass would re-derive the same indices).
///
/// A cuboid's `material_id` IS its render-cell key (clean block id + overlay bit); the
/// occupancy tile never sees it (any voxel a cuboid covers is occupied), the cell-key tile
/// stores it verbatim. Air voxels of the cell-key tile keep [`AIR_CELL_KEY_DONT_CARE`] —
/// occupancy gates every read. `mixed` is the [`uniform_cell_key`] verdict: a uniform block
/// gets no tile (its one key rides on the record).
fn rasterize_brick_tiles(
    geometry: &evaluation::two_layer_store::MicroblockGeometry,
    brick_edge_voxels: u32,
    mixed: bool,
) -> (BrickOccupancyTile, Option<BrickCellKeyTile>) {
    let mut occupancy = BrickOccupancyTile::empty(brick_edge_voxels);
    let mut cell_keys = mixed
        .then(|| BrickCellKeyTile::new_filled(brick_edge_voxels, AIR_CELL_KEY_DONT_CARE));
    for cuboid in &geometry.cuboids {
        let cell_key = cuboid.material_id();
        for voxel_z in cuboid.min[2]..=cuboid.max[2] {
            for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                occupancy.set_x_run(voxel_y, voxel_z, cuboid.min[0], cuboid.max[0]);
                if let Some(tile) = cell_keys.as_mut() {
                    tile.fill_x_run(voxel_y, voxel_z, cuboid.min[0], cuboid.max[0], cell_key);
                }
            }
        }
    }
    (occupancy, cell_keys)
}

/// One block's brick contribution, INDEPENDENT of atlas-slot assignment — the shared
/// classifier both the wholesale [`build_brick_field`] and the G3 incremental update
/// ([`IncrementalBrickField::apply_dirty_update`]) run, so a block classifies to the
/// exact same record kind + material + occupancy either way (only the slot NUMBER
/// differs: wholesale packs `0..count` in record order, incremental allocates from a
/// free-list). Keeping ONE classifier is what makes "incremental == wholesale byte-exact"
/// structural rather than a convention two code paths must independently uphold.
enum BlockBrick {
    /// Air — no record (ADR 0011 Decision 2).
    Air,
    /// A coarse-solid block: the whole record (no atlas slot).
    Coarse(BrickRecord),
    /// A boundary block: the record MINUS its slots (the caller's allocators assign them),
    /// the occupancy tile to land in its atlas slot, and — iff the block is MIXED — the
    /// per-voxel cell-key tile to land in its (independently pooled) material slot. A uniform
    /// block yields `cell_keys: None` and its one cell key as `material_id` + `overlay`.
    Sculpted {
        material_id: u16,
        overlay: bool,
        seam_solidity: SeamSolidity,
        tile: BrickOccupancyTile,
        cell_keys: Option<BrickCellKeyTile>,
    },
}

/// Classify one block of a [`TwoLayerChunk`] into its [`BlockBrick`] — the coarse XOR
/// boundary XOR air partition (ADR 0011 Decision 2). `world_block` is the block's
/// absolute world-block coordinate (its packed key).
fn classify_block_brick(
    chunk: &TwoLayerChunk,
    block: [u32; 3],
    world_block: [i64; 3],
    brick_edge_voxels: u32,
) -> BlockBrick {
    if let Some(block_id) = chunk.coarse_block(block) {
        // Coarse XOR boundary is the classifier's invariant; a block in both layers
        // would double-emit its key.
        debug_assert!(
            !chunk.microblocks.contains_key(&block),
            "a block must be coarse XOR boundary"
        );
        BlockBrick::Coarse(BrickRecord {
            packed_world_block_key: pack_world_block_key(world_block),
            material_id: block_id.color_index(),
            // A coarse block's cell key is its id + the chunk's per-block overlay marker.
            overlay: chunk.coarse_block_overlay(block),
            payload: BrickPayload::CoarseSolid { block_id },
            // Fully solid through ⇒ every face is solid.
            seam_solidity: SeamSolidity {
                solid: [[true; 2]; 3],
            },
        })
    } else if let Some(geometry) = chunk.microblocks.get(&block) {
        // Uniform vs MIXED, decided here and nowhere else: all cuboids sharing ONE cell key
        // ⇒ that key rides on the record (no cell-key tile); disagreeing cuboids ⇒ a
        // per-voxel cell-key tile, and the record's material/overlay become don't-care (kept
        // as the first cuboid's, exactly as before, so a uniform scene's records are
        // byte-identical to the pre-material-atlas ones).
        let uniform = uniform_cell_key(geometry);
        let record_cell_key = uniform.unwrap_or_else(|| {
            geometry
                .cuboids
                .first()
                .map(|cuboid| cuboid.material_id())
                .unwrap_or(AIR_CELL_KEY_DONT_CARE)
        });
        let (tile, cell_keys) =
            rasterize_brick_tiles(geometry, brick_edge_voxels, uniform.is_none());
        BlockBrick::Sculpted {
            material_id: CellKey::from_raw(record_cell_key).block_id(),
            overlay: CellKey::from_raw(record_cell_key).has_overlay(),
            seam_solidity: geometry.seam_solidity,
            tile,
            cell_keys,
        }
    } else {
        BlockBrick::Air
    }
}

/// Scatter a slot-indexed set of `edge³` occupancy tiles into the ADR 0007 tile-cube
/// atlas layout: a cubic-ish `bricks_per_axis³` slot grid (bounded by the slot count,
/// linear slot → 3D tile x-fastest), returning `(bricks_per_axis, atlas_dim_voxels,
/// bytes)`. Shared by the wholesale build and [`IncrementalBrickField::to_build`] so the
/// two produce byte-identical layouts for the same tile vector. A slot with a FREED
/// (dead) tile is scattered as-is — its bytes are unreachable from any live record, so
/// they may be garbage (the free-slot discipline).
///
/// The tile-cube geometry + the per-row expand ARE substrate's [`CubeTilePacking`]; this seam
/// only injects [`SCULPTED_BRICK_OCCUPIED`] as the "set-bit byte" so everything GPU-facing keeps
/// consuming `sculpted_atlas_bytes` unchanged. Returns `(bricks_per_axis, atlas_dim_voxels,
/// bytes)`.
fn pack_sculpted_atlas(
    slot_tiles: &[BrickOccupancyTile],
    brick_edge_voxels: u32,
) -> (u32, u32, Vec<u8>) {
    CubeTilePacking::pack_bit_cubes(slot_tiles, brick_edge_voxels, SCULPTED_BRICK_OCCUPIED)
}

/// The absolute CHUNK coordinate that owns an absolute world block (`floor_div` by
/// [`CHUNK_BLOCKS`]) — the partition the resident cache dirties on, so a record can be
/// tested for membership in an edit's dirty-chunk set.
fn chunk_coord_of_world_block(world_block: [i64; 3]) -> [i32; 3] {
    let blocks = CHUNK_BLOCKS as i64;
    [
        world_block[0].div_euclid(blocks) as i32,
        world_block[1].div_euclid(blocks) as i32,
        world_block[2].div_euclid(blocks) as i32,
    ]
}

/// What an [`IncrementalBrickField::apply_dirty_update`] touched — the per-edit "dirty
/// region" made observable so the GPU sink patches ONLY these atlas slots (never the
/// untouched ones) and the parity net can assert the cost is proportional to the edit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BrickFieldUpdate {
    /// Atlas slots (re)written this edit — newly allocated or overwritten sculpted
    /// bricks. When `atlas_grew` is false these are the ONLY slots the GPU patch writes.
    pub written_slots: Vec<u32>,
    /// Slots FREED this edit (their block became air/coarse or its chunk was removed);
    /// their tiles are now dead until reallocated. Free bytes are never uploaded.
    pub freed_slots: Vec<u32>,
    /// Whether the atlas tile geometry GREW (`bricks_per_axis` increased) — then every
    /// slot's 3D position moved, so the sink MUST re-pack + re-upload the whole atlas
    /// (the one legitimate wholesale re-pack, ADR 0011 pitfalls / ADR 0007 resize
    /// precedent). False ⇒ untouched slots keep their texels.
    pub atlas_grew: bool,
    /// MATERIAL SIDE ATLAS slots (re)written this edit — the cell-key tiles of the MIXED
    /// bricks the edit (re)emitted, in the second pool's OWN numbering. Empty for an edit that
    /// touched no mixed block (the common case: the pool is sparse). The GPU patch's cell-key
    /// work-list, exactly as `written_slots` is the occupancy atlas's.
    pub written_cell_key_slots: Vec<u32>,
    /// Cell-key slots FREED this edit: a block that stopped being mixed (or stopped existing)
    /// releases its material tile. Dead until reallocated; never uploaded.
    pub freed_cell_key_slots: Vec<u32>,
    /// Whether the MATERIAL SIDE ATLAS's tile geometry grew — its OWN signal, independent of
    /// [`atlas_grew`](Self::atlas_grew) (the pools size from their own slot counts, so either
    /// may grow without the other). True ⇒ the sink re-packs + re-uploads the whole side atlas.
    pub cell_key_atlas_grew: bool,
}

/// The PERSISTENT incremental brick field (ADR 0011 slice G3). Maintains the sorted
/// [`BrickRecord`] array + a slot-allocated atlas ACROSS edits so a per-edit update
/// re-evaluates only the DIRTY chunks' blocks and patches only their slots — the
/// "per-edit cost proportional to the dirty region, not the scene" win ADR 0009 promised.
///
/// Slots are managed by a **free-list** (allocate on a new sculpted brick, free when a
/// brick becomes air/coarse or its chunk is dirtied away), so slot numbers are STABLE
/// across edits and differ from the wholesale build's dense `0..count`. The invariant the
/// parity gate proves: after any edit, every LIVE record's slot bytes equal a from-scratch
/// [`build_brick_field`] of the same scene (free slots may hold garbage — they are
/// unreachable). The pyramid is REBUILT (not patched) from the merged record keys per
/// edit (a cheap pure function; incremental pyramid patching is deferred to G4).
#[derive(Debug, Clone)]
pub struct IncrementalBrickField {
    /// The brick edge in voxels (`voxels_per_block`, the ONE-BLOCK granule) — fixed for
    /// the field's life (a density change resets the field via a wholesale rebuild).
    brick_edge_voxels: u32,
    /// Records sorted strictly ascending by packed world-block key — the same order and
    /// content [`build_brick_field`] emits, only the sculpted records' slot NUMBERS differ.
    records: Vec<BrickRecord>,
    /// Per-slot occupancy tiles (bit-packed `edge²` X-row words each — see
    /// [`BrickOccupancyTile`]) indexed by atlas slot, WITH their free-list, delegated to
    /// substrate's [`SlotFreeList`]: a FREED slot's tile is retained (kept `edge²` words so
    /// the atlas packer never trips) but unreferenced — dead bits until the slot is
    /// reallocated. A new sculpted brick pops a freed slot (deterministic reuse — largest free
    /// index first) before growing; the reuse order is a test-readability nicety, not a
    /// correctness contract (parity tolerates slot renumbering — see the `records` doc above).
    slot_tiles: SlotFreeList<BrickOccupancyTile>,
    /// Per-cell-key-slot tiles of the MIXED bricks, in a **separate pool with its own
    /// free-list** — a mixed brick's `cell_key_slot` is not its `atlas_slot` and the two
    /// numberings never coincide (the material side atlas is sparse: only mixed bricks hold a
    /// slot, so tying it to the occupancy numbering would waste a tile per uniform brick).
    /// Same single-owner tile law as `slot_tiles`: tiles are MOVED in at emission, a freed
    /// slot's tile is dead until reallocated. A block flipping uniform↔mixed under an edit
    /// frees or allocates here exactly like any slot churn — the occupancy pool is untouched
    /// by that flip (its slot stays a sculpted slot either way).
    cell_key_tiles: SlotFreeList<BrickCellKeyTile>,
}

impl IncrementalBrickField {
    /// Seed the incremental field from a wholesale [`build_brick_field`] BY MOVE (the reset
    /// a scene load / density change / gate re-engagement performs), returning the mirror
    /// AND the [`SculptedAtlasPayload`] the install seam uploads. Consuming the build is the
    /// single-owner win (`docs/architecture/`, the brick-field display chapter): the record
    /// Vec moves straight into the mirror (no clone) and the flat atlas byte blob moves into
    /// the payload — the wholesale channel/inline reset now ships ONE copy of the field, not
    /// a build plus a mirror seeded from it. Slots are the build's dense `0..sculpted_count`;
    /// the free-list starts empty.
    pub fn from_wholesale(build: BrickFieldBuild) -> (Self, SculptedAtlasPayload) {
        // Re-derive the bit tiles from the build's flat atlas bytes (the one O(sculpted)
        // seeding cost) — the entry for callers that hold ONLY a `BrickFieldBuild` (the
        // golden `shot` tool, the parity/perf tests). The live worker/orchestrator wholesale
        // path instead calls [`from_wholesale_with_tiles`], MOVING the tiles the build just
        // rasterised straight in (no re-gather, no re-bit-pack).
        let sculpted_count = build.sculpted_brick_count();
        let slot_tiles: Vec<BrickOccupancyTile> = (0..sculpted_count as u32)
            .map(|slot| {
                BrickOccupancyTile::from_bytes(
                    build.brick_edge_voxels,
                    &build.sculpted_brick_occupancy(slot),
                )
            })
            .collect();
        Self::from_wholesale_with_tiles(build, slot_tiles)
    }

    /// Seed the incremental field from a wholesale build AND its already-rasterised per-slot
    /// occupancy tiles (dense slot order), MOVING both in — the zero-re-derive path for the
    /// live worker/orchestrator wholesale build. [`build_brick_field_with_tiles`] returns the
    /// build alongside the very tiles it rasterised; handing them here skips the
    /// `from_wholesale` re-gather (`sculpted_brick_occupancy` per slot) + re-bit-pack of
    /// bytes the packer just produced. The tiles MUST be the build's own sculpted tiles (one
    /// per sculpted record, slot order); a debug assert pins the count.
    pub fn from_wholesale_with_tiles(
        build: BrickFieldBuild,
        slot_tiles: Vec<BrickOccupancyTile>,
    ) -> (Self, SculptedAtlasPayload) {
        let sculpted_count = build.sculpted_brick_count();
        // Finding #5: a misordered / wrong-length tile vec would silently desync the mirror
        // from the build. The slot count + a representative brick-edge match are O(1) and
        // catch the structural mistakes, so they earn a RELEASE-mode assert. The exhaustive
        // per-slot byte-equality (O(sculpted·brick)) stays a debug-only check.
        assert_eq!(
            slot_tiles.len(),
            sculpted_count,
            "the carried tiles must be exactly the build's sculpted slots (dense 0..count)"
        );
        assert!(
            slot_tiles
                .first()
                .is_none_or(|tile| tile.edge() == build.brick_edge_voxels),
            "the carried tiles must share the build's brick edge"
        );
        debug_assert!(
            slot_tiles.iter().enumerate().all(|(slot, tile)| {
                tile.edge() == build.brick_edge_voxels
                    && tile.expand_to_bytes(SCULPTED_BRICK_OCCUPIED)
                        == build.sculpted_brick_occupancy(slot as u32)
            }),
            "each carried tile must byte-match the build's own sculpted slot in dense order"
        );
        let BrickFieldBuild {
            brick_records,
            sculpted_atlas_bytes,
            cell_key_tiles,
            brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        } = build;
        // The cell-key tiles ride in the build (the material side atlas has no byte-packed
        // GPU form yet), so they MOVE straight into the mirror's pool — one owner, no clone.
        debug_assert_eq!(
            cell_key_tiles.len(),
            brick_records
                .iter()
                .filter(|record| record.payload.cell_key_slot().is_some())
                .count(),
            "a wholesale build carries exactly one cell-key tile per MIXED record"
        );
        let payload = SculptedAtlasPayload {
            bytes: sculpted_atlas_bytes,
            geometry: SculptedAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels,
            },
            sculpted_slot_count: sculpted_count as u32,
        };
        let mirror = Self {
            brick_edge_voxels,
            records: brick_records,
            // Dense-seed the free-lists: every carried tile is a live slot `0..count`, no holes.
            slot_tiles: SlotFreeList::from_slots(slot_tiles),
            cell_key_tiles: SlotFreeList::from_slots(cell_key_tiles),
        };
        (mirror, payload)
    }

    /// One MIXED brick's per-voxel cell-key tile by its record's `cell_key_slot` — the CPU
    /// read of the material side atlas (the sink that samples it on the GPU is a later slice).
    /// A freed/dead slot yields its stale tile (unreachable from any live record).
    pub fn cell_key_tile(&self, cell_key_slot: u32) -> &BrickCellKeyTile {
        &self.cell_key_tiles[cell_key_slot]
    }

    /// The material side atlas's slot high-water mark (live + freed cell-key slots) — the
    /// pool's own growth signal, independent of the occupancy atlas's.
    pub fn cell_key_slot_high_water(&self) -> usize {
        self.cell_key_tiles.len()
    }

    /// The MATERIAL SIDE ATLAS's tile geometry, derived from ITS OWN slot high-water mark
    /// exactly as [`pack_cell_key_atlas`] would — the twin of
    /// [`atlas_geometry`](Self::atlas_geometry) for the second pool (the patch seam's
    /// slot-origin inputs, without materialising a build).
    pub fn cell_key_atlas_geometry(&self) -> SculptedCellKeyAtlasGeometry {
        let bricks_per_axis = CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len());
        SculptedCellKeyAtlasGeometry {
            bricks_per_axis,
            atlas_dim_voxels: bricks_per_axis * self.brick_edge_voxels,
            brick_edge_voxels: self.brick_edge_voxels,
        }
    }

    /// One cell-key slot's `2 · edge³` little-endian u16 texel bytes — the DIRTY-SLOT upload
    /// the incremental patch writes into the side atlas, straight from the owning tile (no
    /// whole-atlas re-pack). A freed/dead slot yields its stale bytes (unreachable, never
    /// uploaded).
    pub fn cell_key_slot_bytes(&self, cell_key_slot: u32) -> Vec<u8> {
        self.cell_key_tiles[cell_key_slot].to_le_bytes()
    }

    /// Materialise the full MATERIAL SIDE ATLAS as a [`SculptedCellKeyAtlasPayload`] — the
    /// second pool's wholesale re-pack, done only on a side-atlas GROW
    /// ([`BrickFieldUpdate::cell_key_atlas_grew`]) where every cell-key slot's 3D position
    /// moved. Reuses [`pack_cell_key_atlas`], so it stays byte-identical to
    /// [`to_build`](Self::to_build)'s + [`BrickFieldBuild::cell_key_atlas_payload`]'s.
    pub fn pack_cell_key_atlas_payload(&self) -> SculptedCellKeyAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_cell_key_atlas(self.cell_key_tiles.as_slice(), self.brick_edge_voxels);
        SculptedCellKeyAtlasPayload {
            bytes,
            geometry: SculptedCellKeyAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            cell_key_slot_count: self.mixed_brick_count() as u32,
        }
    }

    /// How many LIVE records are MIXED sculpted bricks (== live cell-key tiles).
    pub fn mixed_brick_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.payload.cell_key_slot().is_some())
            .count()
    }

    /// The live records — the sorted [`BrickRecord`] array the GPU record pack + the
    /// pyramid derive from. The mirror is the single CPU owner (item 9): the renderer's
    /// install/patch seams read records straight from here, never via [`to_build`](Self::to_build).
    pub fn records(&self) -> &[BrickRecord] {
        &self.records
    }

    /// How many live records are sculpted bricks — uniform AND mixed (mirror of
    /// [`BrickFieldBuild::sculpted_brick_count`]) — the wholesale install's slot count.
    pub fn sculpted_brick_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.payload.occupancy_atlas_slot().is_some())
            .count()
    }

    /// The sculpted atlas's tile geometry, derived from the slot high-water mark exactly as
    /// [`pack_sculpted_atlas`] would — the frame scalars + slot-origin inputs the patch seam
    /// needs without materialising a build.
    pub fn atlas_geometry(&self) -> SculptedAtlasGeometry {
        let bricks_per_axis = sculpted_atlas_bricks_per_axis(self.slot_tiles.len());
        SculptedAtlasGeometry {
            bricks_per_axis,
            atlas_dim_voxels: bricks_per_axis * self.brick_edge_voxels,
            brick_edge_voxels: self.brick_edge_voxels,
        }
    }

    /// One slot's `edge³` occupancy bytes (bit tile → R8 bytes, O(brick)) — the DIRTY-SLOT
    /// upload the incremental patch writes, straight from the owning tile (no whole-atlas
    /// re-pack). A freed/dead slot yields its stale bytes (unreachable, never uploaded).
    pub fn sculpted_slot_bytes(&self, slot: u32) -> Vec<u8> {
        self.slot_tiles[slot].expand_to_bytes(SCULPTED_BRICK_OCCUPIED)
    }

    /// Materialise the full atlas as a [`SculptedAtlasPayload`] — the ONE legitimate
    /// wholesale re-pack, done only on an atlas GROW (`BrickFieldUpdate::atlas_grew`) where
    /// every slot's 3D position moved. Reuses [`pack_sculpted_atlas`] so it stays
    /// byte-identical to [`to_build`](Self::to_build)'s atlas.
    pub fn pack_atlas_payload(&self) -> SculptedAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_sculpted_atlas(self.slot_tiles.as_slice(), self.brick_edge_voxels);
        SculptedAtlasPayload {
            bytes,
            geometry: SculptedAtlasGeometry {
                bricks_per_axis,
                atlas_dim_voxels,
                brick_edge_voxels: self.brick_edge_voxels,
            },
            sculpted_slot_count: self.sculpted_brick_count() as u32,
        }
    }

    /// The brick edge (voxels_per_block) the field is bound to.
    pub fn brick_edge_voxels(&self) -> u32 {
        self.brick_edge_voxels
    }

    /// The live record count (coarse + sculpted).
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// The atlas slot high-water mark (live + freed slots) — the tile count the atlas is
    /// sized to address. `>= ` the live sculpted count (holes from freed slots).
    pub fn slot_high_water(&self) -> usize {
        self.slot_tiles.len()
    }

    /// Re-evaluate ONLY the blocks of the dirty chunks (plus, for occlusion verdicts, their
    /// 26-neighbourhood ring) and merge them into the field.
    ///
    /// * `fresh_chunks` — the FULL current covering set (dirty chunks freshly resolved,
    ///   clean chunks reused verbatim). Only the dirty chunks + their ring are read.
    /// * `dirty_chunks` — the chunk coords the edit invalidated
    ///   ([`TwoLayerResidentCache::invalidate_aabb`](evaluation::two_layer_store::TwoLayerResidentCache::invalidate_aabb)
    ///   evicted). Every OCCUPANCY change lives in one of these; a block's record content
    ///   (key, material, seam flags, occupancy) is intrinsic to its own chunk.
    ///
    /// **The occlusion dilation (ADR 0011 interior elision — the tricky seam).** Under the
    /// surface-only record contract, whether a coarse block emits a record at all depends on
    /// its six FACE-NEIGHBOURS — which may live in an adjacent, NON-dirty chunk. An edit can
    /// therefore flip records in the 1-chunk dilation of the dirty set: carving a hole
    /// exposes previously-interior blocks of the neighbour chunk (their records must appear),
    /// and filling can occlude previously-surface blocks (their records must vanish). So the
    /// re-mask covers the dirty set DILATED by the 26-neighbourhood (the same dilation the
    /// mesh's cross-chunk seam culling uses; face-dilation would suffice for the 6-neighbour
    /// test, the 26-ring is the conservative shared convention):
    ///
    /// * **dirty chunks** — all records dropped (sculpted slots freed) and rebuilt from the
    ///   fresh data, exactly as before, with occlusion fused in.
    /// * **ring chunks** (dilated \ dirty) — their DATA is unchanged, only occlusion verdicts
    ///   of their COARSE blocks can flip: coarse records are dropped and re-derived against
    ///   the fresh oracle. Sculpted records (and their atlas slots) are KEPT untouched —
    ///   occupancy didn't change, so the per-edit atlas write-set stays ∝ the dirty region
    ///   (the `one_chunk_edit_writes_only_that_chunks_slots` guarantee). Coarse records carry
    ///   no slot, so the ring contributes zero atlas traffic.
    /// * **outside the dilation** — a block's verdict reads only its own chunk + face
    ///   neighbours, all unchanged, so its record is provably identical; kept verbatim.
    ///
    /// Byte-equality vs a from-scratch surface-only [`build_brick_field`] after every edit is
    /// the acceptance bar (`incremental_dirty_update_equals_wholesale_after_every_step`, the
    /// cross-chunk carve case, and the gpu_parity render gate). Returns the
    /// [`BrickFieldUpdate`] describing exactly which slots were touched (the GPU patch's
    /// work-list).
    pub fn apply_dirty_update(
        &mut self,
        fresh_chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
        dirty_chunks: &[[i32; 3]],
    ) -> BrickFieldUpdate {
        let edge = self.brick_edge_voxels;
        let dirty: std::collections::BTreeSet<[i32; 3]> = dirty_chunks.iter().copied().collect();
        // The occlusion ring: the dirty set's 26-neighbourhood minus the dirty set itself.
        let mut ring: std::collections::BTreeSet<[i32; 3]> = std::collections::BTreeSet::new();
        for coord in &dirty {
            for offset_z in -1i32..=1 {
                for offset_y in -1i32..=1 {
                    for offset_x in -1i32..=1 {
                        let neighbour =
                            [coord[0] + offset_x, coord[1] + offset_y, coord[2] + offset_z];
                        if !dirty.contains(&neighbour) {
                            ring.insert(neighbour);
                        }
                    }
                }
            }
        }
        let previous_bricks_per_axis = sculpted_atlas_bricks_per_axis(self.slot_tiles.len());
        let previous_cell_key_bricks_per_axis =
            CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len());

        // 1. Drop every previous record whose block is in a dirty chunk (freeing its slot),
        //    and every COARSE record of a ring chunk (its occlusion verdict may have flipped;
        //    ring SCULPTED records are kept — their chunk's data is unchanged, so record and
        //    slot are still exact, and the atlas is never touched for the ring).
        let mut freed_slots = Vec::new();
        // The MIXED bricks' cell-key slots freed alongside — the second, independent pool
        // (a block that stops being mixed — or stops existing — releases its material tile;
        // a block that BECOMES mixed allocates one below).
        let mut freed_cell_key_slots = Vec::new();
        self.records.retain(|record| {
            let chunk =
                chunk_coord_of_world_block(unpack_world_block_key(record.packed_world_block_key));
            if dirty.contains(&chunk) {
                if let Some(atlas_slot) = record.payload.occupancy_atlas_slot() {
                    freed_slots.push(atlas_slot);
                }
                if let Some(cell_key_slot) = record.payload.cell_key_slot() {
                    freed_cell_key_slots.push(cell_key_slot);
                }
                false
            } else if ring.contains(&chunk) {
                // Ring SCULPTED records (uniform or mixed) are kept verbatim — their chunk's
                // data is unchanged, so both their slots stay exact and neither pool is touched.
                record.payload.occupancy_atlas_slot().is_some()
            } else {
                true
            }
        });
        // Freed slots return to the pool; the free-list keeps them sorted/deduped so reuse is
        // deterministic (largest free index first). This is a nicety for test readability, not
        // correctness: incremental and wholesale agree only up to slot RENUMBERING (the parity
        // oracle compares atlas BYTES, not slot numbers — see `IncrementalBrickField`'s records
        // doc and `incremental_matches_wholesale`), so the reuse order never affects byte parity.
        self.slot_tiles.free(freed_slots.iter().copied());
        self.cell_key_tiles.free(freed_cell_key_slots.iter().copied());

        // 2. Rebuild the dirty chunks' records fully — and the ring chunks' COARSE records —
        //    from the FRESH data, with occlusion verdicts from the fresh oracle (the same
        //    fused elision `build_brick_field` performs, so incremental == wholesale stays
        //    structural).
        let oracle = BrickOcclusionOracle::new(fresh_chunks);
        let mut written_slots = Vec::new();
        // The side atlas's own write-list: the cell-key slots the (re)emitted MIXED bricks took.
        let mut written_cell_key_slots = Vec::new();
        for (chunk_coord, chunk) in fresh_chunks {
            let chunk_is_dirty = dirty.contains(chunk_coord);
            if !chunk_is_dirty && !ring.contains(chunk_coord) {
                continue;
            }
            // Interior-chunk fast path, exactly as the wholesale build: an all-interior
            // chunk emits nothing (it has no microblocks, so no sculpted record is skipped).
            if oracle.chunk_is_all_interior(*chunk_coord) {
                continue;
            }
            let occlusion = oracle.context_for_chunk(*chunk_coord, chunk.as_ref());
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        match classify_block_brick(chunk, block, world_block, edge) {
                            BlockBrick::Air => {}
                            BlockBrick::Coarse(record) => {
                                if !occlusion.coarse_block_occluded(block) {
                                    self.records.push(record);
                                }
                            }
                            BlockBrick::Sculpted {
                                material_id,
                                overlay,
                                seam_solidity,
                                tile,
                                cell_keys,
                            } => {
                                // Ring chunks keep their existing sculpted records (data
                                // unchanged); only a DIRTY chunk re-allocates and rewrites.
                                if !chunk_is_dirty {
                                    continue;
                                }
                                let slot = self.slot_tiles.allocate(tile);
                                written_slots.push(slot);
                                // A MIXED block allocates from the SEPARATE cell-key pool
                                // (its own free-list, its own high-water mark); a uniform
                                // block takes no material slot at all.
                                let payload = match cell_keys {
                                    None => BrickPayload::Sculpted { atlas_slot: slot },
                                    Some(cell_key_tile) => {
                                        let cell_key_slot =
                                            self.cell_key_tiles.allocate(cell_key_tile);
                                        written_cell_key_slots.push(cell_key_slot);
                                        BrickPayload::SculptedMixed {
                                            atlas_slot: slot,
                                            cell_key_slot,
                                        }
                                    }
                                };
                                self.records.push(BrickRecord {
                                    packed_world_block_key: pack_world_block_key(world_block),
                                    material_id,
                                    overlay,
                                    payload,
                                    seam_solidity,
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Re-sort (O(n log n) over records — trivially small next to atlas work).
        self.records
            .sort_unstable_by_key(|record| record.packed_world_block_key);
        debug_assert!(
            self.records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "brick keys must stay unique + sorted after an incremental merge"
        );

        let atlas_grew =
            sculpted_atlas_bricks_per_axis(self.slot_tiles.len()) != previous_bricks_per_axis;
        // The side atlas grows on ITS OWN slot count — a mixed brick appearing can move every
        // cell-key tile without the occupancy grid moving at all (and vice versa).
        let cell_key_atlas_grew = CubeTilePacking::tiles_per_axis(self.cell_key_tiles.len())
            != previous_cell_key_bricks_per_axis;
        BrickFieldUpdate {
            written_slots,
            freed_slots,
            atlas_grew,
            written_cell_key_slots,
            freed_cell_key_slots,
            cell_key_atlas_grew,
        }
    }

    /// Materialise the current field as a [`BrickFieldBuild`] (records + packed atlas).
    ///
    /// **Parity-oracle materialisation ONLY (item 9).** No production / per-frame path may
    /// call this: it clones ALL records and re-packs the ENTIRE flat atlas blob, the exact
    /// cost the single-owner rework removed from the per-edit patch path. The renderer's
    /// install/patch seams now read records / atlas geometry / dirty-slot bytes straight
    /// from the mirror ([`records`](Self::records), [`atlas_geometry`](Self::atlas_geometry),
    /// [`sculpted_slot_bytes`](Self::sculpted_slot_bytes), [`pack_atlas_payload`](Self::pack_atlas_payload)).
    /// This survives as the parity gate's witness — `to_build() == build_brick_field(...)`
    /// after every edit is the G3 acceptance bar. The atlas is sized to the slot high-water
    /// mark (live + freed holes), so a live record's slot bytes are always in range.
    pub fn to_build(&self) -> BrickFieldBuild {
        let (bricks_per_axis, atlas_dim_voxels, sculpted_atlas_bytes) =
            pack_sculpted_atlas(self.slot_tiles.as_slice(), self.brick_edge_voxels);
        BrickFieldBuild {
            brick_records: self.records.clone(),
            sculpted_atlas_bytes,
            // Cell-key tiles in SLOT order (freed holes included, exactly as the occupancy
            // atlas is packed over its high-water mark): a live record's `cell_key_slot`
            // indexes this vec, and a dead slot's tile is unreachable garbage.
            cell_key_tiles: self.cell_key_tiles.as_slice().to_vec(),
            brick_edge_voxels: self.brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        }
    }
}

/// The `bricks_per_axis` a slot-tile count packs to (`ceil(cbrt(count))`, 0 for empty) —
/// the atlas tile-grid edge, shared by the packer and the grow test. This IS substrate's
/// [`CubeTilePacking::tiles_per_axis`]; the wrapper keeps the domain name at the seam.
fn sculpted_atlas_bricks_per_axis(slot_count: usize) -> u32 {
    CubeTilePacking::tiles_per_axis(slot_count)
}

/// Land the sculpted-brick atlas bytes in an R8Unorm 3D texture — the shipped fog-atlas
/// upload mechanic (`upload_grid_per_chunk`'s `write_texture`, no row padding needed).
/// `COPY_SRC` is set so the parity net can read the texture back; a build with no
/// sculpted brick returns a 1³ placeholder (nothing samples it — every record is
/// coarse/air).
pub fn upload_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &SculptedAtlasPayload,
) -> wgpu::Texture {
    let atlas_dim = atlas.geometry.atlas_dim_voxels.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brick-field sculpted atlas"),
        size: wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    if atlas.geometry.atlas_dim_voxels > 0 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
    }
    texture
}

/// Land the MATERIAL SIDE ATLAS's cell-key bytes in an **R16Uint** 3D texture — the second,
/// independently pooled atlas beside [`upload_brick_atlas`]'s R8 occupancy one. `R16Uint`
/// because the texel IS the `u16` cell key verbatim (palette id + overlay bit): an integer
/// sampled with `textureLoad` and compared exactly, never filtered or normalised — a float
/// format would round the id. Two bytes per texel (little-endian, the packer's order), so a
/// row is `2 · edge` bytes. `COPY_SRC` is set for the parity net's readback; a field with no
/// MIXED brick returns a 1³ placeholder (nothing samples it — every record carries its one
/// cell key).
///
/// Known limit (inherited from the occupancy atlas, not introduced here): the app requests
/// `Limits::default()`, so `max_texture_dimension_3d` is 2048 and there is no pre-allocation
/// VRAM budget guard — see docs/design/vram-ceiling-probe.md. The side atlas is sparse (only
/// mixed bricks), so it reaches that ceiling far later than the occupancy pool does.
pub fn upload_brick_cell_key_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &SculptedCellKeyAtlasPayload,
) -> wgpu::Texture {
    let atlas_dim = atlas.geometry.atlas_dim_voxels.max(1);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("brick-field cell-key side atlas"),
        size: wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R16Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    if atlas.geometry.atlas_dim_voxels > 0 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim * CELL_KEY_TEXEL_BYTES),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
    }
    texture
}

/// Bytes per cell-key texel — the R16Uint stride (one little-endian `u16` per voxel). The ONE
/// name for the "2" every side-atlas row/extent arithmetic multiplies by.
pub const CELL_KEY_TEXEL_BYTES: u32 = 2;

/// Read an `atlas_dim³` R8 atlas texture back to row-unpadded bytes — the parity net's
/// A/B readback ONLY (mirrors `dispatch_atlas`; per ADR 0006 §4 nothing ever reads a
/// texture back as truth on a live path).
pub fn read_back_brick_atlas(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    atlas_dim: u32,
) -> Vec<u8> {
    if atlas_dim == 0 {
        return Vec::new();
    }
    // `copy_texture_to_buffer` rows must be 256-aligned (unlike `write_texture`).
    const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
    let padded_row = atlas_dim.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes = padded_row as u64 * atlas_dim as u64 * atlas_dim as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brick-field atlas readback"),
        size: padded_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(atlas_dim),
            },
        },
        wgpu::Extent3d {
            width: atlas_dim,
            height: atlas_dim,
            depth_or_array_layers: atlas_dim,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    receiver
        .recv()
        .expect("map_async channel dropped")
        .expect("buffer map failed");

    let mapped = slice.get_mapped_range();
    let atlas_dim_usize = atlas_dim as usize;
    let padded_row_usize = padded_row as usize;
    let mut atlas_bytes = vec![0u8; atlas_dim_usize.pow(3)];
    for atlas_z in 0..atlas_dim_usize {
        for atlas_y in 0..atlas_dim_usize {
            let source = (atlas_z * atlas_dim_usize + atlas_y) * padded_row_usize;
            let destination = (atlas_z * atlas_dim_usize + atlas_y) * atlas_dim_usize;
            atlas_bytes[destination..destination + atlas_dim_usize]
                .copy_from_slice(&mapped[source..source + atlas_dim_usize]);
        }
    }
    drop(mapped);
    readback.unmap();
    atlas_bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use document::scene::Scene;
    use evaluation::two_layer_store::TwoLayerStore;
    use voxel_core::voxel::{ShapeKind, Voxel};
    use document::voxel::{GeometryParams};

    // The bit-packed occupancy tile IS substrate's `BitCube`; its expand↔pack byte-parity and
    // full-word run-set-mask oracles moved with it (see `crates/substrate/src/bit_cube.rs`,
    // renamed to substrate vocabulary). The tests below exercise the DOMAIN mapping that
    // consumes it (record partition, atlas packing, incremental==wholesale parity).

    /// The interior-INCLUSIVE oracle build ([`build_brick_field_all_blocks`]) maps the
    /// two-layer partition one-to-one: coarse-solid → one kind-0 record (id carried, no
    /// slot), boundary → one kind-1 record (dense unique slots, seam flags carried
    /// unchanged), air → nothing; records sorted strictly ascending. This is the CPU half
    /// of the ADR 0011 gate clause (a) for the record/atlas PACKING mechanics (which the
    /// surface-only live build shares); the surface-only record CONTRACT itself is gated by
    /// `build_emits_only_surface_records_of_a_solid_box`. The `--features gpu` parity test
    /// re-asserts the bytes through the texture round-trip.
    #[test]
    fn brick_records_map_two_layer_partition_one_to_one() {
        // d4 deliberately (ADR 0011 Decision 1): the brick edge must follow the
        // density, not the number 16; odd voxel extents give partial boundary blocks.
        let voxels_per_block = 4;
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Sphere,
                size_voxels: [33, 33, 33],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field_all_blocks(&two_layer_chunks, voxels_per_block);

        assert_eq!(build.brick_edge_voxels, voxels_per_block);
        assert!(
            build
                .brick_records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "records must be sorted strictly ascending (unique keys)"
        );

        let mut expected_coarse = 0usize;
        let mut expected_sculpted = 0usize;
        let mut seen_slots = std::collections::BTreeSet::new();
        for (chunk_coord, chunk) in &two_layer_chunks {
            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        let record = build.find_record(world_block);
                        if let Some(block_id) = chunk.coarse_block(block) {
                            expected_coarse += 1;
                            let record = record.expect("coarse-solid block must have a record");
                            assert_eq!(record.payload.kind_discriminant(), 0);
                            assert_eq!(
                                record.payload,
                                BrickPayload::CoarseSolid { block_id },
                                "coarse record carries the block id, no atlas slot"
                            );
                            assert_eq!(record.seam_solidity.solid, [[true; 2]; 3]);
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            expected_sculpted += 1;
                            let record = record.expect("boundary block must have a record");
                            assert_eq!(record.payload.kind_discriminant(), 1);
                            let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                                panic!("boundary block must be a sculpted record");
                            };
                            assert!(
                                seen_slots.insert(atlas_slot),
                                "atlas slot {atlas_slot} assigned twice"
                            );
                            assert_eq!(
                                record.seam_solidity, geometry.seam_solidity,
                                "seam-solidity flags must carry across unchanged"
                            );
                        } else {
                            assert!(record.is_none(), "air block must emit nothing");
                        }
                    }
                }
            }
        }
        assert_eq!(build.brick_records.len(), expected_coarse + expected_sculpted);
        assert_eq!(build.sculpted_brick_count(), expected_sculpted);
        // Slots are dense 0..count — the atlas holds exactly the sculpted bricks.
        assert_eq!(
            seen_slots.iter().copied().collect::<Vec<_>>(),
            (0..expected_sculpted as u32).collect::<Vec<_>>()
        );
        // The scene must actually exercise both kinds, else the mapping is untested.
        assert!(expected_coarse > 0, "fixture must contain coarse-solid blocks");
        assert!(expected_sculpted > 0, "fixture must contain boundary blocks");
    }

    /// **The surface-only record contract (ADR 0011 interior elision, fused into the
    /// build).** [`build_brick_field`] over a SOLID box emits exactly the surface blocks (a
    /// block with ≥1 absent/air neighbour) of the interior-inclusive oracle build
    /// ([`build_brick_field_all_blocks`]) and omits the strictly-interior ones (all six
    /// neighbours present + solid) — checked against an independent neighbour-presence
    /// oracle over the FULL key set. The `--features gpu`
    /// `brick_surface_elision_hit_set_unchanged` proves the surface-only build renders the
    /// same hit set as the oracle build.
    #[test]
    fn build_emits_only_surface_records_of_a_solid_box() {
        let voxels_per_block = 4;
        // A solid box (ShapeKind::Box ignores wall_blocks — that is Tube-only), 6 blocks
        // per axis, so there is a genuine 4×4×4 fully-occluded interior to elide.
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [6 * voxels_per_block, 6 * voxels_per_block, 6 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let full_build = build_brick_field_all_blocks(&two_layer_chunks, voxels_per_block);
        let surface_build = build_brick_field(&two_layer_chunks, voxels_per_block);
        assert!(!full_build.brick_records.is_empty(), "fixture must build records");
        // Every block of a solid box is coarse-solid (all faces solid).
        assert!(
            full_build
                .brick_records
                .iter()
                .all(|r| r.seam_solidity.solid == [[true; 2]; 3]),
            "a solid box classifies every block coarse-solid"
        );

        // Independent oracle: with all blocks coarse-solid, a block is INTERIOR iff all six
        // of its neighbours are present in the FULL record set. (The tiny fixture never
        // nears the packed-key lane limit, so no range guard is needed.)
        let full_keys: std::collections::HashSet<u64> = full_build
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect();
        let expected_surface_keys: Vec<u64> = full_build
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .filter(|&key| {
                let block = unpack_world_block_key(key);
                let all_neighbours_present = [
                    [1i64, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                ]
                .iter()
                .all(|d| {
                    let nb = [block[0] + d[0], block[1] + d[1], block[2] + d[2]];
                    full_keys.contains(&pack_world_block_key(nb))
                });
                !all_neighbours_present
            })
            .collect();
        let surface_keys: Vec<u64> = surface_build
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect();
        assert_eq!(
            surface_keys, expected_surface_keys,
            "the surface-only build must emit exactly the oracle's surface blocks, in order"
        );
        // A solid box has a genuine interior to omit AND a surface to keep — the split is
        // non-trivial in both directions (else the elision would be vacuous or wrong).
        assert!(
            surface_build.brick_records.len() < full_build.brick_records.len(),
            "a solid box must have fully-occluded interior blocks to omit"
        );
        assert!(!surface_build.brick_records.is_empty(), "the surface blocks must be kept");
        // Both builds pack the identical sculpted atlas (the sculpted set is never elided).
        assert_eq!(surface_build.sculpted_atlas_bytes, full_build.sculpted_atlas_bytes);
        assert_eq!(surface_build.bricks_per_axis, full_build.bricks_per_axis);
    }

    /// The clip-map pyramid is CONSERVATIVE (ADR 0011 parity gate, coarse tier):
    /// each level's occupied-cell set is a SUPERSET of the true occupied cells
    /// (every record's cell present), sorted strictly ascending + unique, at ANY
    /// density (block-denominated cells — nothing hard-codes 16). A scattered
    /// multi-object scene so the levels actually span more than one cell.
    #[test]
    fn clipmap_pyramid_is_conservative_and_sorted() {
        use crate::{Node, NodeContent, NodeTransform};
        for &voxels_per_block in &[16u32, 4] {
            // A dozen small shapes far apart — the scattered scene the LOD targets.
            let mut nodes = Vec::new();
            for i in 0..12i64 {
                let shape = document::voxel::SdfShape::from_blocks(
                    ShapeKind::Sphere,
                    [3, 3, 3],
                    1,
                    voxels_per_block,
                );
                let mut node = Node::new(
                    format!("s{i}"),
                    NodeContent::Tool {
                        shape,
                        material: MaterialChoice::Stone,
                    },
                );
                // Spread them ~16 blocks apart on a lattice so cells are scattered.
                node.transform = NodeTransform::from_blocks(
                    [(i % 4) * 16, (i / 4) * 16, (i % 3) * 20],
                    voxels_per_block,
                );
                nodes.push(node);
            }
            let scene = Scene::from_nodes(nodes);
            let two_layer_chunks =
                TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
            let build = build_brick_field(&two_layer_chunks, voxels_per_block);
            assert!(!build.brick_records.is_empty());
            let pyramid = ClipmapPyramid::from_records(&build.brick_records);

            for (level, blocks_per_cell) in [
                (&pyramid.level_1, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
                (&pyramid.level_2, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
                (&pyramid.level_3, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
            ] {
                assert_eq!(level.blocks_per_cell, blocks_per_cell);
                assert!(
                    level.cell_keys.windows(2).all(|pair| pair[0] < pair[1]),
                    "level {blocks_per_cell} keys must be sorted strictly ascending + unique"
                );
                // Truth: the cell of every record must be present (superset ⇒ the
                // DDA never strides past a real surface).
                let level_set: std::collections::BTreeSet<u64> =
                    level.cell_keys.iter().copied().collect();
                let cell_size = blocks_per_cell as i64;
                let mut true_cells = std::collections::BTreeSet::new();
                for record in &build.brick_records {
                    let b = unpack_world_block_key(record.packed_world_block_key);
                    let cell = [
                        b[0].div_euclid(cell_size),
                        b[1].div_euclid(cell_size),
                        b[2].div_euclid(cell_size),
                    ];
                    true_cells.insert(pack_world_block_key(cell));
                }
                assert!(
                    true_cells.is_subset(&level_set),
                    "level {blocks_per_cell} must cover every occupied cell (conservative)"
                );
                // The min-mip carries no cell the records don't (exactness of the
                // derivation — a spurious occupied cell would only cost perf, but
                // proves the fold has no stray keys).
                assert_eq!(level_set, true_cells);
                assert!(!level.cell_keys.is_empty());
            }
            // Each coarser level must not be finer than the one below (monotone
            // cell counts as the cell size grows 8× per level).
            assert!(pyramid.level_2.cell_keys.len() <= pyramid.level_1.cell_keys.len());
            assert!(pyramid.level_3.cell_keys.len() <= pyramid.level_2.cell_keys.len());
        }
    }

    /// The **chunk-sourced** pyramid ([`ClipmapPyramid::from_chunks`]) is BYTE-IDENTICAL to the
    /// legacy record-sourced one ([`ClipmapPyramid::from_records`]) over the FULL, interior-
    /// inclusive record set — the direct oracle for the interior-elision pyramid rework (ADR
    /// 0011). `build_brick_field_all_blocks` is the interior-inclusive reference build (the live
    /// `build_brick_field` is surface-only, so its records would give a subset pyramid). Covers a
    /// solid box (heavy interior → the bulk fast path) and a scattered scene (partial chunks),
    /// at two densities.
    #[test]
    fn clipmap_from_chunks_equals_from_full_records() {
        use crate::{Node, NodeContent, NodeTransform};
        for &voxels_per_block in &[16u32, 4] {
            // (a) A solid box: every interior chunk is fully-solid → exercises the bulk path.
            let box_scene = Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [
                        7 * voxels_per_block,
                        7 * voxels_per_block,
                        7 * voxels_per_block,
                    ],
                    size_measurements: None,
                    voxels_per_block,
                    wall_blocks: 1,
                },
                MaterialChoice::Stone,
            );
            // (b) A scattered scene: many partial chunks → exercises the per-block path.
            let mut nodes = Vec::new();
            for i in 0..8i64 {
                let shape =
                    document::voxel::SdfShape::from_blocks(ShapeKind::Sphere, [3, 3, 3], 1, voxels_per_block);
                let mut node = Node::new(
                    format!("s{i}"),
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                node.transform = NodeTransform::from_blocks(
                    [(i % 3) * 14, (i / 3) * 14, (i % 2) * 18],
                    voxels_per_block,
                );
                nodes.push(node);
            }
            let scattered_scene = Scene::from_nodes(nodes);

            for scene in [box_scene, scattered_scene] {
                let chunks =
                    TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
                let full_build = build_brick_field_all_blocks(&chunks, voxels_per_block);
                let from_records = ClipmapPyramid::from_records(&full_build.brick_records);
                let from_chunks = ClipmapPyramid::from_chunks(&chunks);
                // Compare the SKIP levels only: `interior_masks` is a band-clip signal the
                // record-sourced oracle never carries (it is interior-inclusive), so it is
                // deliberately built empty there — it is not part of the min-mip identity claim.
                assert_eq!(
                    (&from_chunks.level_1, &from_chunks.level_2, &from_chunks.level_3),
                    (&from_records.level_1, &from_records.level_2, &from_records.level_3),
                    "chunk-sourced pyramid must equal the full-record oracle (density {voxels_per_block})"
                );
            }
        }
    }

    /// **The band-clip interior-occupancy map marks EXACTLY the full-record block set (this
    /// fix).** [`BlockOccupancyMasks::from_chunks`] must report a set bit for every block the
    /// interior-INCLUSIVE oracle build (`build_brick_field_all_blocks`) carries a record for —
    /// no more, no fewer — since that record set is what a band-clipped ray needs to resolve as
    /// coarse cubes where the surface-only build elided them. Covers a solid box (the bulk
    /// fully-solid path, heavy interior) and a scattered scene (the per-block partial path).
    #[test]
    fn block_occupancy_masks_mark_exactly_the_full_record_blocks() {
        use crate::{Node, NodeContent, NodeTransform};
        for &voxels_per_block in &[16u32, 4] {
            let box_scene = Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [
                        7 * voxels_per_block,
                        7 * voxels_per_block,
                        7 * voxels_per_block,
                    ],
                    size_measurements: None,
                    voxels_per_block,
                    wall_blocks: 1,
                },
                MaterialChoice::Stone,
            );
            let mut nodes = Vec::new();
            for i in 0..8i64 {
                let shape = document::voxel::SdfShape::from_blocks(
                    ShapeKind::Sphere,
                    [3, 3, 3],
                    1,
                    voxels_per_block,
                );
                let mut node = Node::new(
                    format!("s{i}"),
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                node.transform =
                    NodeTransform::from_blocks([(i % 3) * 14, (i / 3) * 14, (i % 2) * 18], voxels_per_block);
                nodes.push(node);
            }
            let scattered_scene = Scene::from_nodes(nodes);

            for scene in [box_scene, scattered_scene] {
                let chunks =
                    TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
                let full_build = build_brick_field_all_blocks(&chunks, voxels_per_block);
                let masks = BlockOccupancyMasks::from_chunks(&chunks);
                assert!(!masks.is_empty(), "the scene must occupy blocks");

                // Every full-record block reads as an occupied bit.
                let cell_size = BLOCK_OCCUPANCY_CELL_BLOCKS as i64;
                let bit_set = |world_block: [i64; 3]| -> bool {
                    let cell = [
                        world_block[0].div_euclid(cell_size),
                        world_block[1].div_euclid(cell_size),
                        world_block[2].div_euclid(cell_size),
                    ];
                    let local = [
                        world_block[0].rem_euclid(cell_size) as usize,
                        world_block[1].rem_euclid(cell_size) as usize,
                        world_block[2].rem_euclid(cell_size) as usize,
                    ];
                    let bit = (local[2] * cell_size as usize + local[1]) * cell_size as usize
                        + local[0];
                    masks.contains_bit(pack_world_block_key(cell), bit)
                };
                let mut expected_set: std::collections::BTreeSet<[i64; 3]> =
                    std::collections::BTreeSet::new();
                for record in &full_build.brick_records {
                    let block = unpack_world_block_key(record.packed_world_block_key);
                    assert!(bit_set(block), "full-record block {block:?} missing from the mask");
                    expected_set.insert(block);
                }
                // And no bit is set beyond the full-record set (the mask is not a superset).
                let mut mask_bits = 0u64;
                for mask in masks.cell_masks() {
                    for word in mask {
                        mask_bits += word.count_ones() as u64;
                    }
                }
                assert_eq!(
                    mask_bits,
                    expected_set.len() as u64,
                    "mask must set exactly the full-record blocks (density {voxels_per_block})"
                );
            }
        }
    }

    /// CPU byte-exactness at a non-16 density: every sculpted brick's atlas bytes equal
    /// the block occupancy the SHIPPED expansion (`expand_occupancy_into`, itself
    /// proven bit-exact vs the dense oracle) reports — rasterization from cuboids and
    /// expansion are independent paths over the same boundary set.
    #[test]
    fn sculpted_brick_bytes_match_expanded_occupancy_at_non_16_density() {
        let voxels_per_block = 4;
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Torus,
                size_voxels: [49, 13, 49],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&two_layer_chunks, voxels_per_block);

        let edge = voxels_per_block as usize;
        let mut compared_bricks = 0usize;
        for (chunk_coord, chunk) in &two_layer_chunks {
            // Chunk-local occupancy bitmap via the shipped expansion (offset zero).
            let mut expanded: Vec<Voxel> = Vec::new();
            chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
            let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as usize;
            let mut chunk_occupancy = vec![0u8; chunk_extent.pow(3)];
            for voxel in &expanded {
                let [x, y, z] = voxel.local_index;
                chunk_occupancy
                    [(z as usize * chunk_extent + y as usize) * chunk_extent + x as usize] =
                    SCULPTED_BRICK_OCCUPIED;
            }

            for block in chunk.microblocks.keys() {
                let world_block = [
                    chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block[0] as i64,
                    chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block[1] as i64,
                    chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block[2] as i64,
                ];
                let record = build.find_record(world_block).expect("sculpted record");
                let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                    panic!("boundary block must be sculpted");
                };
                let brick_bytes = build.sculpted_brick_occupancy(atlas_slot);
                let mut expected = vec![0u8; edge.pow(3)];
                for local_z in 0..edge {
                    for local_y in 0..edge {
                        for local_x in 0..edge {
                            let chunk_voxel = [
                                block[0] as usize * edge + local_x,
                                block[1] as usize * edge + local_y,
                                block[2] as usize * edge + local_z,
                            ];
                            expected[(local_z * edge + local_y) * edge + local_x] =
                                chunk_occupancy[(chunk_voxel[2] * chunk_extent
                                    + chunk_voxel[1])
                                    * chunk_extent
                                    + chunk_voxel[0]];
                        }
                    }
                }
                assert_eq!(
                    brick_bytes, expected,
                    "brick bytes must equal the expanded block occupancy at {world_block:?}"
                );
                compared_bricks += 1;
            }
        }
        assert!(compared_bricks > 0, "fixture must contain sculpted bricks");
    }
}

/// ADR 0011 slice G3 — the incremental dirty-brick atlas update net. The load-bearing
/// assertion: an [`IncrementalBrickField`] patched edit-by-edit (only dirty chunks
/// re-evaluated, slots free-listed) is byte-exact vs a from-scratch [`build_brick_field`]
/// of the SAME scene, after EVERY step, across explicit block-kind transitions
/// (air↔sculpted↔coarse) and add / move / recolour / delete edits.
#[cfg(test)]
mod incremental_tests {
    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use evaluation::cuboid::VoxelBox;
    use document::scene::{Node, NodeContent, NodeTransform, Scene};
    use evaluation::two_layer_store::{
        MicroblockGeometry, TwoLayerChunk, TwoLayerResidentCache, TwoLayerStore,
    };
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{GeometryParams, SdfShape};

    /// The owned covering set the shell feeds `apply_dirty_update` / `build_brick_field`
    /// (the resident cache borrows, so clone out — exactly as `AppCore::rebuild` does).
    fn covering_owned(
        cache: &mut TwoLayerResidentCache,
        scene: &Scene,
        density: u32,
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        cache.resident_two_layer_chunks(scene, density, 0)
    }

    /// A tool node (single material, so the scene stays brick-representable) of `blocks³`
    /// at a block offset — the small edited object.
    fn tool(kind: ShapeKind, offset_blocks: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        node
    }

    /// The set of atlas slots the live sculpted records reference, plus a check that no
    /// two live records share a slot (a "ghost brick" would show as a duplicate).
    fn live_slots(build: &BrickFieldBuild) -> std::collections::BTreeSet<u32> {
        let mut slots = std::collections::BTreeSet::new();
        for record in &build.brick_records {
            if let BrickPayload::Sculpted { atlas_slot } = record.payload {
                assert!(
                    slots.insert(atlas_slot),
                    "live slot {atlas_slot} referenced twice (ghost brick)"
                );
            }
        }
        slots
    }

    /// Assert the incremental field materialisation is byte-exact vs the wholesale build
    /// of the same scene: SAME record keys, kinds, materials, seam flags; each sculpted
    /// record's atlas bytes equal (slot NUMBERS differ — the free-list vs dense `0..count`
    /// — so compare the occupancy, not the slot). Free slots may hold garbage: they are
    /// asserted unreachable from live records (the `live_slots` uniqueness check).
    fn assert_incremental_matches_wholesale(
        incremental: &BrickFieldBuild,
        wholesale: &BrickFieldBuild,
        label: &str,
    ) {
        assert_eq!(
            incremental.brick_edge_voxels, wholesale.brick_edge_voxels,
            "[{label}] brick edge must match"
        );
        assert_eq!(
            incremental.brick_records.len(),
            wholesale.brick_records.len(),
            "[{label}] record count must match wholesale"
        );
        let _ = live_slots(incremental); // no ghost bricks (live slots unique)
        for whole_record in &wholesale.brick_records {
            let block = unpack_world_block_key(whole_record.packed_world_block_key);
            let inc_record = incremental
                .find_record(block)
                .unwrap_or_else(|| panic!("[{label}] incremental missing record at {block:?}"));
            assert_eq!(
                inc_record.packed_world_block_key, whole_record.packed_world_block_key,
                "[{label}] key mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.material_id, whole_record.material_id,
                "[{label}] material mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.seam_solidity, whole_record.seam_solidity,
                "[{label}] seam-solidity mismatch at {block:?}"
            );
            assert_eq!(
                inc_record.payload.kind_discriminant(),
                whole_record.payload.kind_discriminant(),
                "[{label}] kind mismatch at {block:?}"
            );
            match (inc_record.payload, whole_record.payload) {
                (
                    BrickPayload::CoarseSolid { block_id: a },
                    BrickPayload::CoarseSolid { block_id: b },
                ) => assert_eq!(a, b, "[{label}] coarse block id mismatch at {block:?}"),
                (
                    BrickPayload::Sculpted { atlas_slot: inc_slot },
                    BrickPayload::Sculpted { atlas_slot: whole_slot },
                ) => {
                    // Slot NUMBERS differ (free-list vs dense) — compare the bytes.
                    assert_eq!(
                        incremental.sculpted_brick_occupancy(inc_slot),
                        wholesale.sculpted_brick_occupancy(whole_slot),
                        "[{label}] sculpted occupancy bytes mismatch at {block:?}"
                    );
                }
                _ => panic!("[{label}] payload kind disagreement at {block:?}"),
            }
        }
    }

    /// THE parity gate for G3 (issue #69 acceptance): drive a scripted sequence of edits
    /// — recolour, move, shape-swap, delete, re-add — applying each INCREMENTALLY, and
    /// after every step assert the incremental field equals a from-scratch wholesale build
    /// of the same scene. Two fixed anchor tools at the extremes pin the covering set so an
    /// incremental edit never grows it (the app's reframe guard — a growth routes wholesale).
    /// A non-16 density exercises the block-denominated granule.
    #[test]
    fn incremental_dirty_update_equals_wholesale_after_every_step() {
        let density = 4u32;
        let material = MaterialChoice::Stone;
        // Two anchors far apart fix the covering chunk range; the middle tool is edited.
        let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], material, density);
        let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], material, density);
        let scene_with = |middle: Option<Node>| {
            let mut nodes = vec![anchor_lo.clone(), anchor_hi.clone()];
            if let Some(m) = middle {
                nodes.push(m);
            }
            Scene::from_nodes(nodes)
        };

        // The scripted edits (each keeps the anchors, edits the middle) — chosen to force
        // block-kind transitions: add (air→sculpted/coarse), move (sculpted↔air↔coarse),
        // recolour (sculpted/coarse material change), shape-swap (occupancy change), delete.
        let scenes = [
            ("initial", scene_with(None)),
            ("add-sphere", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density)))),
            ("recolour", scene_with(Some(tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density)))),
            ("move", scene_with(Some(tool(ShapeKind::Sphere, [2, 1, 0], MaterialChoice::Plain, density)))),
            ("shape-swap", scene_with(Some(tool(ShapeKind::Box, [2, 1, 0], MaterialChoice::Plain, density)))),
            ("delete", scene_with(None)),
            ("re-add", scene_with(Some(tool(ShapeKind::Torus, [0, 0, 0], MaterialChoice::Wood, density)))),
        ];

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let scene0 = &scenes[0].1;
        let mut previous_index = scene0.build_leaf_spatial_index(density);
        let fresh0 = covering_owned(&mut cache, scene0, density);
        let build0 = build_brick_field(&fresh0, density);
        let mut field = IncrementalBrickField::from_wholesale(build0.clone()).0;
        let mut covering: std::collections::BTreeSet<[i32; 3]> =
            fresh0.iter().map(|(coord, _)| *coord).collect();
        assert_incremental_matches_wholesale(&field.to_build(), &build0, scenes[0].0);

        let mut incremental_steps = 0usize;
        for (label, scene) in &scenes[1..] {
            let new_index = scene.build_leaf_spatial_index(density);
            let edit_aabb = new_index.edit_aabb_since(&previous_index);
            // Mirror `AppCore::rebuild`: localisable edit → invalidate its chunks; a `None`
            // (wholesale) edit clears. Build the fresh covering set AFTER invalidation.
            let dirty = match &edit_aabb {
                Some(aabb) => cache.invalidate_aabb(aabb, density),
                None => {
                    cache.clear();
                    Vec::new()
                }
            };
            let fresh = covering_owned(&mut cache, scene, density);
            let new_covering: std::collections::BTreeSet<[i32; 3]> =
                fresh.iter().map(|(coord, _)| *coord).collect();

            // Incremental applies only when localisable AND the covering set is invariant
            // (the app routes a growth/reframe wholesale). Otherwise reset from wholesale.
            if edit_aabb.is_some() && new_covering == covering {
                field.apply_dirty_update(&fresh, &dirty);
                incremental_steps += 1;
            } else {
                let build = build_brick_field(&fresh, density);
                field = IncrementalBrickField::from_wholesale(build).0;
            }
            covering = new_covering;

            let wholesale = build_brick_field(&fresh, density);
            assert_incremental_matches_wholesale(&field.to_build(), &wholesale, label);
            previous_index = new_index;
        }
        assert!(
            incremental_steps >= 4,
            "the script must exercise the INCREMENTAL path on most steps (was {incremental_steps})"
        );
    }

    /// Untouched-slot discipline (issue #69 acceptance): an edit confined to ONE chunk
    /// writes only that chunk's blocks' slots (+ frees), never the whole scene's — the
    /// "per-edit cost ∝ dirty region" claim made testable. A recolour keeps occupancy
    /// identical, so exactly the dirty chunk's sculpted blocks are freed + rewritten.
    #[test]
    fn one_chunk_edit_writes_only_that_chunks_slots() {
        let density = 4u32;
        // Anchors fix the covering set; a compact middle tool occupies its own chunks.
        let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
        ]);
        let scene_b = Scene::from_nodes(vec![
            anchor_lo,
            anchor_hi,
            // Same shape/placement, DIFFERENT material — a pure recolour (occupancy fixed).
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_a = scene_a.build_leaf_spatial_index(density);
        let fresh_a = covering_owned(&mut cache, &scene_a, density);
        let build_a = build_brick_field(&fresh_a, density);
        let mut field = IncrementalBrickField::from_wholesale(build_a.clone()).0;
        let total_sculpted = build_a.sculpted_brick_count();

        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("a recolour is a localisable edit");
        let dirty = cache.invalidate_aabb(&edit_aabb, density);
        let fresh_b = covering_owned(&mut cache, &scene_b, density);

        // Count the sculpted blocks living in the dirty chunks (the recolour re-writes
        // exactly these — occupancy is unchanged, only the record material differs).
        let dirty_set: std::collections::BTreeSet<[i32; 3]> = dirty.iter().copied().collect();
        let expected_written: usize = fresh_b
            .iter()
            .filter(|(coord, _)| dirty_set.contains(coord))
            .map(|(_, chunk)| chunk.microblocks.len())
            .sum();

        let update = field.apply_dirty_update(&fresh_b, &dirty);

        assert!(
            !dirty.is_empty() && dirty.len() < covering_owned(&mut cache, &scene_b, density).len(),
            "the edit must dirty SOME but not ALL chunks (dirtied {} of the covering set)",
            dirty.len()
        );
        assert_eq!(
            update.written_slots.len(),
            expected_written,
            "an edit must write exactly the dirty chunks' sculpted slots, no more"
        );
        assert!(
            update.written_slots.len() < total_sculpted,
            "a one-region edit must write FEWER than every scene slot ({} of {})",
            update.written_slots.len(),
            total_sculpted
        );
        // A pure recolour keeps occupancy, so freed == rewritten (slots recycled in place)
        // and the atlas does not grow.
        assert_eq!(update.freed_slots.len(), expected_written, "recolour frees what it rewrites");
        assert!(!update.atlas_grew, "a recolour does not grow the atlas");
        // And the result is still byte-exact vs wholesale.
        let wholesale = build_brick_field(&fresh_b, density);
        assert_incremental_matches_wholesale(&field.to_build(), &wholesale, "one-chunk-recolour");
    }

    /// **The patch-parity witness (item 9).** The renderer's patch path no longer materialises
    /// `to_build()` per edit — it reads each dirty slot's bytes and the atlas geometry straight
    /// from the mirror. This pins those owner-side accessors to what a `to_build()`
    /// materialisation would have produced: after an incremental edit, every written slot's
    /// `sculpted_slot_bytes` equals `to_build().sculpted_brick_occupancy` for that slot, and
    /// `atlas_geometry()` matches the build's tile geometry. If these ever drift, the GPU patch
    /// would upload the wrong texels while the parity gate (which still uses `to_build`) stayed
    /// green — so this is the guard the deleted per-edit `to_build` used to provide implicitly.
    #[test]
    fn patched_slot_bytes_and_geometry_match_to_build_materialisation() {
        let density = 4u32;
        let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
        ]);
        // A pure recolour (occupancy fixed) — writes the dirty chunk's slots without growing.
        let scene_b = Scene::from_nodes(vec![
            anchor_lo,
            anchor_hi,
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_a = scene_a.build_leaf_spatial_index(density);
        let fresh_a = covering_owned(&mut cache, &scene_a, density);
        let build_a = build_brick_field(&fresh_a, density);
        let mut field = IncrementalBrickField::from_wholesale(build_a).0;

        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("a recolour is a localisable edit");
        let dirty = cache.invalidate_aabb(&edit_aabb, density);
        let fresh_b = covering_owned(&mut cache, &scene_b, density);

        let update = field.apply_dirty_update(&fresh_b, &dirty);
        assert!(!update.written_slots.is_empty(), "the recolour must write some slots");

        // The materialisation the patch path used to build per edit — the witness.
        let materialised = field.to_build();
        let geometry = field.atlas_geometry();
        assert_eq!(
            geometry.brick_edge_voxels, materialised.brick_edge_voxels,
            "mirror edge matches the materialisation"
        );
        assert_eq!(
            geometry.bricks_per_axis, materialised.bricks_per_axis,
            "mirror tile-grid edge matches the materialisation"
        );
        assert_eq!(
            geometry.atlas_dim_voxels, materialised.atlas_dim_voxels,
            "mirror atlas dimension matches the materialisation"
        );
        for &slot in &update.written_slots {
            assert_eq!(
                field.sculpted_slot_bytes(slot),
                materialised.sculpted_brick_occupancy(slot),
                "written slot {slot} bytes must equal the to_build() materialisation"
            );
        }
        // The full re-pack payload equals the materialisation's atlas byte-for-byte.
        assert_eq!(
            field.pack_atlas_payload().bytes,
            materialised.sculpted_atlas_bytes,
            "the grow-path re-pack equals the materialised atlas"
        );
    }

    /// **The occlusion-dilation seam (ADR 0011 interior elision).** Under the surface-only
    /// record contract, an edit can flip records in NON-dirty neighbour chunks: carving away
    /// a block un-occludes the face-adjacent blocks across the chunk boundary (their records
    /// must APPEAR), and filling it back occludes them again (their records must VANISH).
    /// Two chunk-filling solid boxes abut across a chunk boundary; deleting the second is
    /// the carve, re-adding it the fill. After each step the incrementally-patched field
    /// must equal a from-scratch surface-only wholesale build byte-for-byte — this is what
    /// the 26-neighbourhood ring re-derivation in `apply_dirty_update` exists for. The test
    /// also asserts the scenario is REAL: the carve changes the record set of a chunk that
    /// was NOT in the dirty set (else the fixture is vacuous).
    #[test]
    fn incremental_carve_across_chunk_boundary_flips_neighbour_occlusion() {
        let density = 4u32;
        let material = MaterialChoice::Stone;
        let chunk_span = CHUNK_BLOCKS as i64;
        // A solid SKETCH-EXTRUDE cube of exactly CHUNK_BLOCKS³ blocks at a chunk-aligned
        // offset — the sketch producer classifies COARSE-solid blocks to the very face
        // (unlike an SDF Box tool, whose 1-block shell resolves as boundary microblocks and
        // would never exercise coarse-record occlusion flips at the interface).
        let chunk_filling_box = |offset_blocks: [i64; 3]| -> Node {
            let edge_voxels = chunk_span * density as i64;
            let producer = document::sketch::SketchSolid::extrude(
                document::sketch::Sketch::rectangle(
                    document::sketch::PlaneAxis::Z,
                    edge_voxels,
                    edge_voxels,
                ),
                edge_voxels as u32,
            );
            let mut node = Node::new(
                format!("box@{offset_blocks:?}"),
                NodeContent::SketchTool { producer, material },
            );
            node.transform = NodeTransform::from_blocks(offset_blocks, density);
            node
        };
        // Anchors pin the covering set so the delete / re-add stays an incremental edit.
        let anchor_lo = chunk_filling_box([-4 * chunk_span, 0, 0]);
        let anchor_hi = chunk_filling_box([4 * chunk_span, 0, 0]);
        // The resident pair: box A and box B abutting on +X across a chunk boundary. Box A's
        // +X-face blocks are occluded exactly while box B exists.
        let box_a = chunk_filling_box([0, 0, 0]);
        let box_b = chunk_filling_box([chunk_span, 0, 0]);
        let scene_with_b = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            box_a.clone(),
            box_b.clone(),
        ]);
        let scene_without_b =
            Scene::from_nodes(vec![anchor_lo.clone(), anchor_hi.clone(), box_a.clone()]);

        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_with_b = scene_with_b.build_leaf_spatial_index(density);
        let fresh_with_b = covering_owned(&mut cache, &scene_with_b, density);
        let build_with_b = build_brick_field(&fresh_with_b, density);
        let mut field = IncrementalBrickField::from_wholesale(build_with_b.clone()).0;

        // --- Step 1: CARVE (delete box B) — exposes box A's face blocks across the seam.
        let index_without_b = scene_without_b.build_leaf_spatial_index(density);
        let carve_aabb = index_without_b
            .edit_aabb_since(&index_with_b)
            .expect("a node delete is a localisable edit");
        let carve_dirty = cache.invalidate_aabb(&carve_aabb, density);
        let fresh_without_b = covering_owned(&mut cache, &scene_without_b, density);
        assert_eq!(
            fresh_with_b.len(),
            fresh_without_b.len(),
            "the anchors must pin the covering set (incremental precondition)"
        );
        field.apply_dirty_update(&fresh_without_b, &carve_dirty);
        let wholesale_without_b = build_brick_field(&fresh_without_b, density);
        assert_incremental_matches_wholesale(
            &field.to_build(),
            &wholesale_without_b,
            "carve-across-boundary",
        );

        // The scenario must be REAL: some chunk OUTSIDE the dirty set changed its record
        // set (box A's face blocks un-occluded) — else the ring re-derivation is untested.
        let dirty_set: std::collections::BTreeSet<[i32; 3]> =
            carve_dirty.iter().copied().collect();
        let records_by_chunk = |build: &BrickFieldBuild| {
            let mut by_chunk: std::collections::BTreeMap<[i32; 3], Vec<u64>> =
                std::collections::BTreeMap::new();
            for record in &build.brick_records {
                let chunk = {
                    let block = unpack_world_block_key(record.packed_world_block_key);
                    [
                        block[0].div_euclid(CHUNK_BLOCKS as i64) as i32,
                        block[1].div_euclid(CHUNK_BLOCKS as i64) as i32,
                        block[2].div_euclid(CHUNK_BLOCKS as i64) as i32,
                    ]
                };
                by_chunk.entry(chunk).or_default().push(record.packed_world_block_key);
            }
            by_chunk
        };
        let before_by_chunk = records_by_chunk(&build_with_b);
        let after_by_chunk = records_by_chunk(&wholesale_without_b);
        let non_dirty_chunk_changed = before_by_chunk
            .iter()
            .any(|(chunk, keys)| {
                !dirty_set.contains(chunk) && after_by_chunk.get(chunk) != Some(keys)
            });
        assert!(
            non_dirty_chunk_changed,
            "fixture must flip records in a NON-dirty chunk (the occlusion ring); \
             dirty set: {dirty_set:?}"
        );

        // --- Step 2: FILL (re-add box B) — re-occludes box A's face blocks.
        let fill_aabb = index_with_b
            .edit_aabb_since(&index_without_b)
            .expect("a node re-add is a localisable edit");
        let fill_dirty = cache.invalidate_aabb(&fill_aabb, density);
        let fresh_refilled = covering_owned(&mut cache, &scene_with_b, density);
        field.apply_dirty_update(&fresh_refilled, &fill_dirty);
        let wholesale_refilled = build_brick_field(&fresh_refilled, density);
        assert_incremental_matches_wholesale(
            &field.to_build(),
            &wholesale_refilled,
            "fill-across-boundary",
        );
        // Fill restores the original record keys (slot numbers may differ — free-listed).
        assert_eq!(
            wholesale_refilled
                .brick_records
                .iter()
                .map(|r| r.packed_world_block_key)
                .collect::<Vec<_>>(),
            build_with_b
                .brick_records
                .iter()
                .map(|r| r.packed_world_block_key)
                .collect::<Vec<_>>(),
            "re-adding box B must restore the original surface record keys"
        );
    }

    /// Perf probe (issue #69, `#[ignore]`d — run in release): a ~1–2k-block scene, a
    /// one-region recolour, incremental patch vs a full `build_brick_field`. The headless
    /// stand-in for the Tracy live latency measurement; numbers go in the commit message.
    /// Run: `cargo test --release incremental_vs_wholesale_perf_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn incremental_vs_wholesale_perf_probe() {
        use std::time::Instant;
        let density = 8u32;
        let anchor_lo = tool(ShapeKind::Box, [-20, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi = tool(ShapeKind::Box, [20, 0, 0], MaterialChoice::Stone, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo.clone(),
            anchor_hi.clone(),
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood, density),
        ]);
        let scene_b = Scene::from_nodes(vec![
            anchor_lo,
            anchor_hi,
            tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Plain, density),
        ]);
        let mut cache = TwoLayerResidentCache::enabled();
        cache.clear();
        let index_a = scene_a.build_leaf_spatial_index(density);
        let fresh_a = covering_owned(&mut cache, &scene_a, density);
        let build_a = build_brick_field(&fresh_a, density);
        let mut field = IncrementalBrickField::from_wholesale(build_a.clone()).0;

        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("localisable");
        let dirty = cache.invalidate_aabb(&edit_aabb, density);
        let fresh_b = covering_owned(&mut cache, &scene_b, density);

        let started = Instant::now();
        let update = field.apply_dirty_update(&fresh_b, &dirty);
        let _incremental_build = field.to_build();
        let incremental = started.elapsed();

        let started = Instant::now();
        let _ = build_brick_field(&fresh_b, density);
        let wholesale = started.elapsed();

        println!(
            "G3 perf probe: scene {} records, edit dirtied {} chunk(s) / {} slots — \
             incremental {:?} vs wholesale {:?} ({:.1}× )",
            build_a.brick_records.len(),
            dirty.len(),
            update.written_slots.len(),
            incremental,
            wholesale,
            wholesale.as_secs_f64() / incremental.as_secs_f64().max(1e-9),
        );
        assert!(update.written_slots.len() < build_a.sculpted_brick_count());
    }

    // ========================================================================
    // The per-voxel cell-key side atlas (the CPU half): emission classifies a
    // sculpted block uniform vs MIXED, and only a mixed block owns a cell-key tile.
    //
    // These fixtures drive the emission builder DIRECTLY with hand-built two-layer chunks —
    // the tightest test of the CPU classifier. (The representability gate is now deleted, so a
    // mixed scene DOES reach the brick path through the renderer; the rendering side is proven by
    // the mixed-material golden + parity test. This module remains the CPU mirror's own contract.)
    // ========================================================================

    /// The density the hand-built fixtures use — small enough to state a block's cuboids
    /// by hand, and deliberately NOT 16 (the brick edge follows the density).
    const HAND_DENSITY: u32 = 4;

    /// A hand-built covering set of ONE chunk at `[0, 0, 0]`: `coarse_blocks` are
    /// `(block, block_id, overlay)`, `sculpted_blocks` are `(block, cuboids)` whose cuboid
    /// labels are render-cell keys ([`CellKey::compose`]).
    fn hand_built_chunk(
        coarse_blocks: &[([u32; 3], u16, bool)],
        sculpted_blocks: &[([u32; 3], Vec<VoxelBox>)],
    ) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
        let block_count = (CHUNK_BLOCKS as usize).pow(3);
        let mut chunk = TwoLayerChunk {
            voxels_per_block: HAND_DENSITY,
            coarse: vec![None; block_count],
            coarse_overlay: vec![false; block_count],
            microblocks: std::collections::BTreeMap::new(),
        };
        for (block, block_id, overlay) in coarse_blocks {
            let flat = (block[2] as usize * CHUNK_BLOCKS as usize + block[1] as usize)
                * CHUNK_BLOCKS as usize
                + block[0] as usize;
            chunk.coarse[flat] = Some(BlockId(*block_id));
            chunk.coarse_overlay[flat] = *overlay;
        }
        for (block, cuboids) in sculpted_blocks {
            chunk.microblocks.insert(
                *block,
                MicroblockGeometry {
                    cuboids: cuboids.clone(),
                    seam_solidity: SeamSolidity::default(),
                },
            );
        }
        vec![([0, 0, 0], Arc::new(chunk))]
    }

    /// A block-local cuboid carrying the render-cell key `(block_id, overlay)`.
    fn cell_box(min: [u32; 3], max: [u32; 3], block_id: u16, overlay: bool) -> VoxelBox {
        VoxelBox {
            min,
            max,
            label: CellKey::compose(block_id, overlay).raw(),
        }
    }

    /// The independent oracle for one block's per-voxel cell keys: paint each cuboid's key
    /// into a dense `edge³` array in cuboid order (air stays [`AIR_CELL_KEY_DONT_CARE`]).
    fn expected_cell_keys(cuboids: &[VoxelBox]) -> Vec<u16> {
        let edge = HAND_DENSITY as usize;
        let mut keys = vec![AIR_CELL_KEY_DONT_CARE; edge.pow(3)];
        for cuboid in cuboids {
            for z in cuboid.min[2]..=cuboid.max[2] {
                for y in cuboid.min[1]..=cuboid.max[1] {
                    for x in cuboid.min[0]..=cuboid.max[0] {
                        keys[(z as usize * edge + y as usize) * edge + x as usize] = cuboid.label;
                    }
                }
            }
        }
        keys
    }

    /// The independent oracle for one block's occupancy bytes (the same walk, occupancy only).
    fn expected_occupancy_bytes(cuboids: &[VoxelBox]) -> Vec<u8> {
        let edge = HAND_DENSITY as usize;
        let mut bytes = vec![0u8; edge.pow(3)];
        for cuboid in cuboids {
            for z in cuboid.min[2]..=cuboid.max[2] {
                for y in cuboid.min[1]..=cuboid.max[1] {
                    for x in cuboid.min[0]..=cuboid.max[0] {
                        bytes[(z as usize * edge + y as usize) * edge + x as usize] =
                            SCULPTED_BRICK_OCCUPIED;
                    }
                }
            }
        }
        bytes
    }

    /// **Emission classifies uniform vs MIXED (the slice's core claim).** A block whose
    /// microblock cuboids all share one cell key is UNIFORM: its material + overlay ride on
    /// the record and it owns NO cell-key tile. A block whose cuboids disagree — on the
    /// material OR on the overlay bit alone — is MIXED: it additionally carries a per-voxel
    /// cell-key tile whose keys match its cuboids exactly, while its occupancy tile is
    /// unchanged (byte-identical to the occupancy-only rasterization). A coarse block carries
    /// its id + its chunk overlay marker and owns neither tile.
    #[test]
    fn emission_classifies_uniform_and_mixed_sculpted_blocks() {
        let uniform_block = [0u32, 0, 0];
        let uniform_cuboids = vec![
            cell_box([0, 0, 0], [1, 3, 3], 1, true),
            cell_box([2, 0, 0], [3, 1, 3], 1, true), // same cell key ⇒ still uniform
        ];
        let mixed_material_block = [1u32, 0, 0];
        let mixed_material_cuboids = vec![
            cell_box([0, 0, 0], [1, 3, 3], 1, false),
            cell_box([2, 0, 0], [3, 3, 3], 2, false), // different block id ⇒ MIXED
        ];
        let mixed_overlay_block = [2u32, 0, 0];
        let mixed_overlay_cuboids = vec![
            cell_box([0, 0, 0], [3, 3, 1], 1, false),
            cell_box([0, 0, 2], [3, 3, 3], 1, true), // same id, overlay differs ⇒ MIXED
        ];
        let coarse_block = [3u32, 0, 0];
        let chunks = hand_built_chunk(
            &[(coarse_block, 2, true)],
            &[
                (uniform_block, uniform_cuboids.clone()),
                (mixed_material_block, mixed_material_cuboids.clone()),
                (mixed_overlay_block, mixed_overlay_cuboids.clone()),
            ],
        );
        let build = build_brick_field(&chunks, HAND_DENSITY);

        // Exactly the two mixed blocks own a cell-key tile.
        assert_eq!(build.mixed_brick_count(), 2);
        assert_eq!(build.cell_key_tiles.len(), 2);
        assert_eq!(build.sculpted_brick_count(), 3, "all three boundary blocks are sculpted");

        // (a) The uniform block: one cell key on the record, NO cell-key tile.
        let record = build
            .find_record([uniform_block[0] as i64, uniform_block[1] as i64, uniform_block[2] as i64])
            .expect("uniform boundary block must have a record");
        assert_eq!(record.material_id, 1);
        assert!(record.overlay, "the uniform block's single cell key sets the overlay bit");
        assert_eq!(record.payload.cell_key_slot(), None, "a uniform brick owns no cell-key tile");
        assert!(matches!(record.payload, BrickPayload::Sculpted { .. }));
        assert_eq!(record.payload.kind_discriminant(), 1);

        // (b) + (c) The mixed blocks: a cell-key tile whose per-voxel keys are exactly the
        // cuboids', an occupancy tile unchanged by the classification.
        for (block, cuboids) in [
            (mixed_material_block, &mixed_material_cuboids),
            (mixed_overlay_block, &mixed_overlay_cuboids),
        ] {
            let record = build
                .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
                .expect("mixed boundary block must have a record");
            let BrickPayload::SculptedMixed {
                atlas_slot,
                cell_key_slot,
            } = record.payload
            else {
                panic!("a block whose cuboids disagree on their cell key must emit MIXED");
            };
            assert_eq!(
                record.payload.kind_discriminant(),
                2,
                "a MIXED brick is its own GPU record kind (it traverses like a sculpted one, \
                 but shades from its cell-key tile)"
            );
            assert_eq!(
                build.cell_key_tiles[cell_key_slot as usize].as_slice(),
                expected_cell_keys(cuboids).as_slice(),
                "the cell-key tile must carry each voxel's own cuboid key at {block:?}"
            );
            assert_eq!(
                build.sculpted_brick_occupancy(atlas_slot),
                expected_occupancy_bytes(cuboids),
                "the occupancy tile is unchanged by the material classification at {block:?}"
            );
        }

        // (d) The coarse block: id + the chunk's overlay marker, no slot of either pool.
        let record = build
            .find_record([coarse_block[0] as i64, coarse_block[1] as i64, coarse_block[2] as i64])
            .expect("coarse block must have a record");
        assert_eq!(record.material_id, 2);
        assert!(record.overlay, "a coarse block carries its chunk's per-block overlay marker");
        assert_eq!(record.payload.occupancy_atlas_slot(), None);
        assert_eq!(record.payload.cell_key_slot(), None);

        // The two mixed bricks' cell-key slots are DISTINCT and dense in the wholesale build.
        let mut cell_key_slots: Vec<u32> = build
            .brick_records
            .iter()
            .filter_map(|record| record.payload.cell_key_slot())
            .collect();
        cell_key_slots.sort_unstable();
        assert_eq!(cell_key_slots, vec![0, 1]);
    }

    /// Resolve every live record's cell-key tile through the mirror's own slot numbering and
    /// compare it against a from-scratch wholesale build of the same chunks (whose numbering
    /// is dense and unrelated) — the cell-key half of the incremental-vs-wholesale parity
    /// oracle: same kind, same material/overlay, same occupancy bytes, same per-voxel keys.
    fn assert_cell_key_parity(
        mirror: &IncrementalBrickField,
        chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
        label: &str,
    ) {
        let wholesale = build_brick_field(chunks, HAND_DENSITY);
        let incremental = mirror.to_build();
        assert_eq!(
            incremental.brick_records.len(),
            wholesale.brick_records.len(),
            "[{label}] record count must match wholesale"
        );
        assert_eq!(
            mirror.mixed_brick_count(),
            wholesale.mixed_brick_count(),
            "[{label}] live mixed-brick count must match wholesale"
        );
        for (mirrored, whole) in incremental
            .brick_records
            .iter()
            .zip(wholesale.brick_records.iter())
        {
            let block = unpack_world_block_key(whole.packed_world_block_key);
            assert_eq!(
                mirrored.packed_world_block_key, whole.packed_world_block_key,
                "[{label}] record order must match wholesale"
            );
            assert_eq!(
                (mirrored.material_id, mirrored.overlay),
                (whole.material_id, whole.overlay),
                "[{label}] record cell key at {block:?}"
            );
            assert_eq!(
                mirrored.payload.cell_key_slot().is_some(),
                whole.payload.cell_key_slot().is_some(),
                "[{label}] uniform/mixed verdict at {block:?}"
            );
            match (
                mirrored.payload.occupancy_atlas_slot(),
                whole.payload.occupancy_atlas_slot(),
            ) {
                (Some(mirror_slot), Some(whole_slot)) => assert_eq!(
                    incremental.sculpted_brick_occupancy(mirror_slot),
                    wholesale.sculpted_brick_occupancy(whole_slot),
                    "[{label}] occupancy bytes at {block:?} (slots renumber, bytes do not)"
                ),
                (None, None) => {}
                _ => panic!("[{label}] payload kind disagreement at {block:?}"),
            }
            if let (Some(mirror_slot), Some(whole_slot)) = (
                mirrored.payload.cell_key_slot(),
                whole.payload.cell_key_slot(),
            ) {
                assert_eq!(
                    mirror.cell_key_tile(mirror_slot).as_slice(),
                    wholesale.cell_key_tiles[whole_slot as usize].as_slice(),
                    "[{label}] cell-key tile at {block:?} (slots renumber, keys do not)"
                );
            }
        }
    }

    /// **A block flipping uniform↔mixed under an incremental edit allocates/frees its
    /// cell-key slot** — in the SEPARATE material pool, leaving the occupancy pool alone (the
    /// block stays a sculpted brick either way, so its occupancy slot is merely rewritten).
    /// After every step the mirror agrees with a from-scratch wholesale build, cell-key tiles
    /// included.
    #[test]
    fn incremental_uniform_mixed_flip_churns_only_the_cell_key_pool() {
        let block_a = [0u32, 0, 0];
        let block_b = [1u32, 0, 0];
        let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
        let mixed_a = vec![
            cell_box([0, 0, 0], [3, 3, 1], 1, false),
            cell_box([0, 0, 2], [3, 3, 3], 2, false),
        ];
        let mixed_b = vec![
            cell_box([0, 0, 0], [1, 3, 3], 2, false),
            cell_box([2, 0, 0], [3, 3, 3], 2, true), // overlay-only mix
        ];
        let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

        // Step 0 (wholesale seed): A uniform, B mixed — one cell-key slot in use.
        let step_0 = hand_built_chunk(
            &[],
            &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
        );
        let build = build_brick_field(&step_0, HAND_DENSITY);
        let (mut mirror, _atlas) = IncrementalBrickField::from_wholesale(build);
        assert_eq!(mirror.mixed_brick_count(), 1);
        assert_eq!(mirror.cell_key_slot_high_water(), 1);
        let occupancy_high_water = mirror.slot_high_water();
        assert_eq!(occupancy_high_water, 2, "both blocks are sculpted bricks");
        assert_cell_key_parity(&mirror, &step_0, "step 0 (wholesale seed)");

        // Step 1 (the FLIP): A becomes mixed, B becomes uniform. B's cell-key slot is freed
        // and A's allocation reuses it — the material pool churns, its high-water mark does
        // not grow, and the occupancy pool's does not move at all.
        let step_1 = hand_built_chunk(
            &[],
            &[(block_a, mixed_a.clone()), (block_b, uniform_b.clone())],
        );
        let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
        assert!(!update.atlas_grew, "the occupancy atlas must not grow on a material flip");
        assert_eq!(mirror.slot_high_water(), occupancy_high_water);
        assert_eq!(mirror.mixed_brick_count(), 1, "exactly one block is mixed after the flip");
        assert_eq!(
            mirror.cell_key_slot_high_water(),
            1,
            "the freed cell-key slot must be reused, not appended to"
        );
        let record_a = mirror
            .records()
            .iter()
            .find(|record| unpack_world_block_key(record.packed_world_block_key) == [0, 0, 0])
            .expect("block A must still have a record");
        let cell_key_slot = record_a
            .payload
            .cell_key_slot()
            .expect("block A is MIXED after the flip");
        assert_eq!(
            mirror.cell_key_tile(cell_key_slot).as_slice(),
            expected_cell_keys(&mixed_a).as_slice()
        );
        let record_b = mirror
            .records()
            .iter()
            .find(|record| unpack_world_block_key(record.packed_world_block_key) == [1, 0, 0])
            .expect("block B must still have a record");
        assert_eq!(record_b.payload.cell_key_slot(), None, "block B is UNIFORM after the flip");
        assert_eq!((record_b.material_id, record_b.overlay), (2, true));
        assert_cell_key_parity(&mirror, &step_1, "step 1 (uniform↔mixed flip)");

        // Step 2 (GROW): both blocks mixed — the second mixed brick appends a new cell-key
        // slot (the pool grows independently of the occupancy pool, which stays put).
        let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
        mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
        assert_eq!(mirror.mixed_brick_count(), 2);
        assert_eq!(mirror.cell_key_slot_high_water(), 2);
        assert_eq!(mirror.slot_high_water(), occupancy_high_water);
        assert_cell_key_parity(&mirror, &step_2, "step 2 (both mixed)");

        // Step 3 (FREE): both blocks uniform — every cell-key slot is freed (the mixed count
        // drops to zero; the high-water mark keeps the freed holes, as the occupancy pool does).
        let step_3 = hand_built_chunk(&[], &[(block_a, uniform_a), (block_b, uniform_b)]);
        mirror.apply_dirty_update(&step_3, &[[0, 0, 0]]);
        assert_eq!(mirror.mixed_brick_count(), 0, "no block is mixed any more");
        assert_eq!(
            mirror.cell_key_slot_high_water(),
            2,
            "freed slots keep their (dead) tiles until reallocated"
        );
        assert!(mirror
            .records()
            .iter()
            .all(|record| record.payload.cell_key_slot().is_none()));
        assert_cell_key_parity(&mirror, &step_3, "step 3 (both uniform again)");
    }

    /// A scene whose sculpted blocks are all UNIFORM (every scene the brick path renders
    /// today) emits NO cell-key tile at all — the sparse-side-atlas contract, and the reason
    /// the GPU bytes cannot move in this slice: `pack_gpu_records` reads the occupancy slot +
    /// the record material, both untouched.
    #[test]
    fn a_uniform_scene_emits_no_cell_key_tiles() {
        let voxels_per_block = 4;
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Sphere,
                size_voxels: [33, 33, 33],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&chunks, voxels_per_block);
        assert!(build.sculpted_brick_count() > 0, "the fixture must have sculpted bricks");
        assert!(
            build.cell_key_tiles.is_empty(),
            "a single-material scene must pay no per-voxel material cost"
        );
        assert_eq!(build.mixed_brick_count(), 0);
        assert!(
            build
                .brick_records
                .iter()
                .all(|record| !matches!(record.payload, BrickPayload::SculptedMixed { .. })),
            "no record may be mixed in a single-material scene"
        );

        // …and therefore packs a ZERO-LENGTH side atlas: the second pool costs such a scene
        // nothing at all (not even a tile grid).
        let side_atlas = build.cell_key_atlas_payload();
        assert!(side_atlas.bytes.is_empty(), "no mixed brick ⇒ no side-atlas bytes");
        assert_eq!(side_atlas.cell_key_slot_count, 0);
        assert_eq!(side_atlas.geometry.bricks_per_axis, 0);
        assert_eq!(side_atlas.geometry.atlas_dim_voxels, 0);
    }

    /// Read one cell-key slot's `edge³` keys back out of a PACKED side atlas — an INDEPENDENT
    /// re-derivation of the GPU's addressing (linear slot → 3D tile origin, x-fastest; texels
    /// little-endian u16, two bytes each), so a bug in the packer cannot hide behind the
    /// packer's own arithmetic.
    fn packed_cell_keys_at_slot(bytes: &[u8], bricks_per_axis: u32, slot: u32) -> Vec<u16> {
        let edge = HAND_DENSITY as usize;
        let tiles = bricks_per_axis.max(1) as usize;
        let atlas_dim = tiles * edge;
        let slot = slot as usize;
        let origin = [
            (slot % tiles) * edge,
            ((slot / tiles) % tiles) * edge,
            (slot / (tiles * tiles)) * edge,
        ];
        let mut keys = Vec::with_capacity(edge.pow(3));
        for local_z in 0..edge {
            for local_y in 0..edge {
                for local_x in 0..edge {
                    let texel = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                        * atlas_dim
                        + origin[0]
                        + local_x;
                    keys.push(u16::from_le_bytes([bytes[texel * 2], bytes[texel * 2 + 1]]));
                }
            }
        }
        keys
    }

    /// **The R16 side atlas packs each mixed brick's cell-key tile at its own slot origin.**
    /// The pool is sized from ITS OWN slot count (two mixed bricks ⇒ a 2-tile grid), holds two
    /// little-endian bytes per voxel, and every live slot reads back — through an independent
    /// addressing oracle — as exactly that block's per-voxel cuboid keys. Bricks that are
    /// uniform or coarse consume no texel here, whatever their occupancy slot.
    #[test]
    fn mixed_bricks_pack_the_r16_side_atlas_at_their_own_slot_origins() {
        let uniform_block = [0u32, 0, 0];
        let uniform_cuboids = vec![cell_box([0, 0, 0], [3, 3, 3], 1, true)];
        let mixed_material_block = [1u32, 0, 0];
        let mixed_material_cuboids = vec![
            cell_box([0, 0, 0], [1, 3, 3], 1, false),
            cell_box([2, 0, 0], [3, 3, 3], 2, false),
        ];
        let mixed_overlay_block = [2u32, 0, 0];
        let mixed_overlay_cuboids = vec![
            cell_box([0, 0, 0], [3, 3, 1], 1, false),
            cell_box([0, 0, 2], [3, 3, 3], 1, true),
        ];
        let chunks = hand_built_chunk(
            &[([3, 0, 0], 2, true)],
            &[
                (uniform_block, uniform_cuboids),
                (mixed_material_block, mixed_material_cuboids.clone()),
                (mixed_overlay_block, mixed_overlay_cuboids.clone()),
            ],
        );
        let build = build_brick_field(&chunks, HAND_DENSITY);
        let side_atlas = build.cell_key_atlas_payload();

        // The pool's OWN geometry: two mixed bricks ⇒ ceil(cbrt 2) = 2 tiles/axis, 8 voxels/axis
        // — while the occupancy pool holds THREE sculpted bricks (its own, larger, tile grid).
        assert_eq!(side_atlas.cell_key_slot_count, 2);
        assert_eq!(side_atlas.geometry.bricks_per_axis, 2);
        assert_eq!(side_atlas.geometry.atlas_dim_voxels, 2 * HAND_DENSITY);
        assert_eq!(side_atlas.geometry.brick_edge_voxels, HAND_DENSITY);
        assert_eq!(
            side_atlas.bytes.len(),
            2 * (2 * HAND_DENSITY as usize).pow(3),
            "two bytes per texel — the R16Uint stride"
        );
        assert_eq!(build.sculpted_brick_count(), 3);

        // Every live slot reads back as that block's own per-voxel keys.
        for (block, cuboids) in [
            (mixed_material_block, &mixed_material_cuboids),
            (mixed_overlay_block, &mixed_overlay_cuboids),
        ] {
            let record = build
                .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
                .expect("a mixed block must have a record");
            let slot = record
                .payload
                .cell_key_slot()
                .expect("a mixed block must own a cell-key slot");
            assert_eq!(
                packed_cell_keys_at_slot(
                    &side_atlas.bytes,
                    side_atlas.geometry.bricks_per_axis,
                    slot
                ),
                expected_cell_keys(cuboids),
                "the packed side atlas must carry {block:?}'s keys at its slot origin"
            );
        }

        // The two mixed slots occupy DISJOINT texel spans (the slot → origin map is injective):
        // exactly the two tiles' worth of texels are non-zero-keyed, the rest of the cube is
        // untouched fill.
        let occupied_texels = side_atlas
            .bytes
            .chunks_exact(2)
            .filter(|texel| u16::from_le_bytes([texel[0], texel[1]]) != AIR_CELL_KEY_DONT_CARE)
            .count();
        let expected_keyed: usize = [&mixed_material_cuboids, &mixed_overlay_cuboids]
            .iter()
            .map(|cuboids| {
                expected_cell_keys(cuboids)
                    .iter()
                    .filter(|key| **key != AIR_CELL_KEY_DONT_CARE)
                    .count()
            })
            .sum();
        assert_eq!(
            occupied_texels, expected_keyed,
            "no key may land outside its own slot's tile"
        );
    }

    /// **The incremental pool's GPU work-list.** A uniform↔mixed flip reports exactly the
    /// cell-key slots the sink must free and rewrite (the second pool's own lists, independent
    /// of the occupancy atlas's), and the bytes it packs are the bytes a from-scratch build
    /// packs — tile-for-tile at each live record's slot (the pools renumber across the two
    /// paths; the texels do not). The `to_build()` parity-oracle style, for the side atlas.
    #[test]
    fn incremental_cell_key_pool_reports_its_work_list_and_packs_like_wholesale() {
        let block_a = [0u32, 0, 0];
        let block_b = [1u32, 0, 0];
        let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
        let mixed_a = vec![
            cell_box([0, 0, 0], [3, 3, 1], 1, false),
            cell_box([0, 0, 2], [3, 3, 3], 2, false),
        ];
        let mixed_b = vec![
            cell_box([0, 0, 0], [1, 3, 3], 2, false),
            cell_box([2, 0, 0], [3, 3, 3], 2, true),
        ];
        let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

        // Seed: A uniform, B mixed — one cell-key slot, a 1-tile side atlas.
        let step_0 = hand_built_chunk(
            &[],
            &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
        );
        let (mut mirror, _atlas) =
            IncrementalBrickField::from_wholesale(build_brick_field(&step_0, HAND_DENSITY));
        assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 1);

        // The FLIP: A becomes mixed, B becomes uniform. B's slot is freed and A's allocation
        // reuses it — so the sink frees slot 0 and rewrites slot 0, and neither tile grid grows.
        let step_1 = hand_built_chunk(&[], &[(block_a, mixed_a.clone()), (block_b, uniform_b)]);
        let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
        assert_eq!(update.freed_cell_key_slots, vec![0]);
        assert_eq!(update.written_cell_key_slots, vec![0]);
        assert!(!update.cell_key_atlas_grew, "a reused slot cannot grow the side atlas");
        assert!(!update.atlas_grew, "the occupancy pool is untouched by a material flip");

        // The dirty-slot bytes the sink uploads ARE the tile's little-endian texels.
        let mut expected_bytes = Vec::new();
        for key in expected_cell_keys(&mixed_a) {
            expected_bytes.extend_from_slice(&key.to_le_bytes());
        }
        assert_eq!(mirror.cell_key_slot_bytes(0), expected_bytes);

        // GROW: B becomes mixed too — the side atlas's OWN grow signal fires (2 slots ⇒ a
        // 2-tile grid), while the occupancy pool's does not.
        let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
        let update = mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
        assert!(
            update.cell_key_atlas_grew,
            "the second mixed brick must grow the side atlas's tile grid"
        );
        assert!(!update.atlas_grew, "the occupancy pool's grid is unchanged");
        assert_eq!(update.written_cell_key_slots.len(), 2);
        assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 2);

        // The wholesale-parity bar: the mirror's packed side atlas carries, at every live mixed
        // record's slot, exactly the tile a from-scratch build packs at ITS slot.
        let packed = mirror.pack_cell_key_atlas_payload();
        assert_eq!(
            packed, mirror.to_build().cell_key_atlas_payload(),
            "the two materialisations of one mirror must be byte-identical"
        );
        let wholesale = build_brick_field(&step_2, HAND_DENSITY);
        let wholesale_atlas = wholesale.cell_key_atlas_payload();
        assert_eq!(packed.geometry, wholesale_atlas.geometry);
        assert_eq!(packed.cell_key_slot_count, wholesale_atlas.cell_key_slot_count);
        for record in mirror.records() {
            let Some(mirror_slot) = record.payload.cell_key_slot() else {
                continue;
            };
            let block = unpack_world_block_key(record.packed_world_block_key);
            let whole_slot = wholesale
                .find_record(block)
                .and_then(|whole| whole.payload.cell_key_slot())
                .expect("the wholesale build must call the same block mixed");
            assert_eq!(
                packed_cell_keys_at_slot(
                    &packed.bytes,
                    packed.geometry.bricks_per_axis,
                    mirror_slot
                ),
                packed_cell_keys_at_slot(
                    &wholesale_atlas.bytes,
                    wholesale_atlas.geometry.bricks_per_axis,
                    whole_slot
                ),
                "packed cell keys at {block:?} (slots renumber, texels do not)"
            );
        }
    }
}
