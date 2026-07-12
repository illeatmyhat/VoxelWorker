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

use crate::core_geom::{BlockId, CHUNK_BLOCKS};
use crate::cuboid_mesh::clean_block_id;
use crate::two_layer_store::{SeamSolidity, TwoLayerChunk};

/// Signed world-block coordinates are biased into this many bits per axis inside the
/// packed key: ±2^20 (~1M) blocks per axis, far beyond the anisotropic 10k+-block
/// target. Three 21-bit lanes fill bits 0..63 (z high), so the packed key's integer
/// order IS lexicographic (z, y, x) block order — sortable on the CPU and binary-
/// searchable as a `(hi, lo)` u32 pair in WGSL (no u64 there).
const WORLD_BLOCK_KEY_BITS_PER_AXIS: u32 = 21;
const WORLD_BLOCK_KEY_BIAS: i64 = 1 << (WORLD_BLOCK_KEY_BITS_PER_AXIS - 1);

/// Pack an absolute world-block coordinate into the sorted-record key (z-major
/// lexicographic order). Panics if a coordinate falls outside the ±2^20 biased lane —
/// a scene that large is out of every current target's range, and a silent wrap would
/// alias two blocks onto one brick.
pub fn pack_world_block_key(world_block: [i64; 3]) -> u64 {
    let mut packed = 0u64;
    // z fills the highest lane so integer order == (z, y, x) lexicographic order.
    for (lane, &coordinate) in [world_block[2], world_block[1], world_block[0]]
        .iter()
        .enumerate()
    {
        let biased = coordinate + WORLD_BLOCK_KEY_BIAS;
        assert!(
            (0..(1i64 << WORLD_BLOCK_KEY_BITS_PER_AXIS)).contains(&biased),
            "world-block coordinate {coordinate} exceeds the packed-key lane (±2^20 blocks)"
        );
        packed |= (biased as u64) << ((2 - lane) as u32 * WORLD_BLOCK_KEY_BITS_PER_AXIS);
    }
    packed
}

/// Unpack a [`pack_world_block_key`] key back to its world-block coordinate (the
/// parity harness's mismatch-location readout; the shader never needs it).
pub fn unpack_world_block_key(key: u64) -> [i64; 3] {
    let lane_mask = (1u64 << WORLD_BLOCK_KEY_BITS_PER_AXIS) - 1;
    let unpack_lane = |lane: u32| -> i64 {
        ((key >> (lane * WORLD_BLOCK_KEY_BITS_PER_AXIS)) & lane_mask) as i64
            - WORLD_BLOCK_KEY_BIAS
    };
    [unpack_lane(0), unpack_lane(1), unpack_lane(2)]
}

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

impl ClipmapLevel {
    /// An empty level (no occupied cells) — the "pyramid off" form the renderer
    /// installs to A/B the hierarchical skip (`record_count == 0` ⇒ the shader
    /// never skips, so the march is the flat G1 block-DDA).
    pub fn empty(blocks_per_cell: u32) -> Self {
        ClipmapLevel {
            blocks_per_cell: blocks_per_cell.max(1),
            cell_keys: Vec::new(),
        }
    }

    /// Fold a record set's block keys into this level's occupied-cell set: every
    /// record's block maps to exactly one cell; the deduplicated, sorted set is
    /// the min-mip. Pure function of the record keys (ADR 0011 4a).
    pub fn from_records(records: &[BrickRecord], blocks_per_cell: u32) -> Self {
        let blocks_per_cell = blocks_per_cell.max(1);
        let cell_size = blocks_per_cell as i64;
        let mut cell_keys: Vec<u64> = records
            .iter()
            .map(|record| {
                let block = unpack_world_block_key(record.packed_world_block_key);
                let cell = [
                    block[0].div_euclid(cell_size),
                    block[1].div_euclid(cell_size),
                    block[2].div_euclid(cell_size),
                ];
                pack_world_block_key(cell)
            })
            .collect();
        cell_keys.sort_unstable();
        cell_keys.dedup();
        ClipmapLevel {
            blocks_per_cell,
            cell_keys,
        }
    }

    /// Fold every non-air block of the two-layer chunk set into this level's occupied-cell
    /// set — the **chunk-sourced** min-mip that replaces [`from_records`](Self::from_records)
    /// now that the record set is SURFACE-ONLY (ADR 0011 interior elision, this epic). The
    /// pyramid must stay a conservative superset over EVERY occupied block (interior included,
    /// so the DDA never strides past an occupied cell), which the surface record set no longer
    /// enumerates — but the chunks do (their coarse layer holds the interior).
    ///
    /// **Solid-chunk bulk fast path (the interior-elision win carried to the pyramid):** a
    /// fully-solid chunk (all `CHUNK_BLOCKS³` coarse-solid, no microblocks) covers one aligned
    /// block box, so its occupied cells are the cell range that box spans — bulk-added WITHOUT
    /// visiting its 64 blocks. A boundary / partial chunk adds one cell per occupied (coarse or
    /// microblock) block. The resulting cell set is BYTE-IDENTICAL to [`from_records`] over the
    /// full, interior-inclusive record set (proven by
    /// `clipmap_from_chunks_equals_from_full_records`): every occupied block's cell is present,
    /// no others.
    pub fn from_chunks(chunks: &[([i32; 3], Arc<TwoLayerChunk>)], blocks_per_cell: u32) -> Self {
        let blocks_per_cell = blocks_per_cell.max(1);
        let cell_size = blocks_per_cell as i64;
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
                let cell_lo = [
                    chunk_block_low[0].div_euclid(cell_size),
                    chunk_block_low[1].div_euclid(cell_size),
                    chunk_block_low[2].div_euclid(cell_size),
                ];
                let cell_hi = [
                    (chunk_block_low[0] + chunk_blocks - 1).div_euclid(cell_size),
                    (chunk_block_low[1] + chunk_blocks - 1).div_euclid(cell_size),
                    (chunk_block_low[2] + chunk_blocks - 1).div_euclid(cell_size),
                ];
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
                            let cell = [
                                (chunk_block_low[0] + block_x as i64).div_euclid(cell_size),
                                (chunk_block_low[1] + block_y as i64).div_euclid(cell_size),
                                (chunk_block_low[2] + block_z as i64).div_euclid(cell_size),
                            ];
                            cell_keys.push(pack_world_block_key(cell));
                        }
                    }
                }
            }
        }
        cell_keys.par_sort_unstable();
        cell_keys.dedup();
        ClipmapLevel {
            blocks_per_cell,
            cell_keys,
        }
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
        ClipmapPyramid {
            level_1: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            level_2: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            level_3: ClipmapLevel::from_records(records, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
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
        .map(|&key| [(key >> 32) as u32, key as u32])
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
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockOccupancyMasks {
    /// Present 8-block cells' packed keys, sorted strictly ascending + deduplicated.
    pub cell_keys: Vec<u64>,
    /// Per-cell `512`-bit occupancy bitmask (`bit = (local_z*8 + local_y)*8 + local_x`,
    /// `local = block.rem_euclid(8)`), one `[u32; 16]` per key. Parallel to `cell_keys`.
    pub cell_masks: Vec<[u32; BLOCK_OCCUPANCY_MASK_WORDS]>,
    /// Per-cell fallback material colour index (the first occupied block's, in build order) —
    /// the coarse-cube's shade when the record-miss fallback fires. Exact for a uniform-material
    /// interior cell (every current band golden); best-effort where a cell mixes materials
    /// (the documented tolerance edge — the R8 atlas is occupancy-only, so per-interior-block
    /// material would re-introduce the O(volume) record set this contract deleted). Parallel to
    /// `cell_keys`.
    pub cell_materials: Vec<u32>,
}

impl BlockOccupancyMasks {
    /// The empty map (no occupied cells) — the "off" form for the record-sourced /
    /// pyramid-off constructors that never carry an interior signal (they run FULL-band only).
    pub fn empty() -> Self {
        BlockOccupancyMasks::default()
    }

    /// Set one block's bit (and, first-writer-wins, its cell material) in the cell map.
    fn insert_block(
        cells: &mut std::collections::BTreeMap<u64, ([u32; BLOCK_OCCUPANCY_MASK_WORDS], u32)>,
        world_block: [i64; 3],
        material: u32,
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
            .or_insert(([0u32; BLOCK_OCCUPANCY_MASK_WORDS], material));
        entry.0[bit / 32] |= 1u32 << (bit % 32);
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
                // block colour — no per-block visit beyond the constant bit-set (a 4-aligned
                // chunk box lands wholly inside one 8-block cell per axis).
                let material = chunk
                    .coarse_block([0, 0, 0])
                    .map(|block_id| block_id.color_index() as u32)
                    .unwrap_or(0);
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
                                material,
                            );
                        }
                    }
                }
            } else {
                for block_z in 0..CHUNK_BLOCKS {
                    for block_y in 0..CHUNK_BLOCKS {
                        for block_x in 0..CHUNK_BLOCKS {
                            let block = [block_x, block_y, block_z];
                            let material = if let Some(block_id) = chunk.coarse_block(block) {
                                block_id.color_index() as u32
                            } else if let Some(geometry) = chunk.microblocks.get(&block) {
                                geometry
                                    .cuboids
                                    .first()
                                    .map(|cuboid| clean_block_id(cuboid.material_id) as u32)
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
                                material,
                            );
                        }
                    }
                }
            }
        }
        let mut cell_keys = Vec::with_capacity(cells.len());
        let mut cell_masks = Vec::with_capacity(cells.len());
        let mut cell_materials = Vec::with_capacity(cells.len());
        for (key, (mask, material)) in cells {
            cell_keys.push(key);
            cell_masks.push(mask);
            cell_materials.push(material);
        }
        BlockOccupancyMasks {
            cell_keys,
            cell_masks,
            cell_materials,
        }
    }

    /// The present-cell count (== the shader's occupancy binary-search span; 0 ⇒ the
    /// band-clip interior fallback never fires).
    pub fn cell_count(&self) -> u32 {
        self.cell_keys.len() as u32
    }
}

/// What a brick holds — ADR 0011 Decision 2's two record kinds. The enum makes
/// "a coarse record consumes no atlas slot" structural, not a convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPayload {
    /// **Kind 0** — an analytic coarse brick: the whole block is solid at `block_id`,
    /// stored as this one record with no per-voxel data (interior elision on the GPU;
    /// also the residency-miss fallback form the G1 contract renders).
    CoarseSolid { block_id: BlockId },
    /// **Kind 1** — a sculpted brick: the block's voxel occupancy lives in atlas slot
    /// `atlas_slot` (an `edge³` R8 tile, edge = `voxels_per_block`).
    Sculpted { atlas_slot: u32 },
}

impl BrickPayload {
    /// The GPU-side record-kind discriminant (0 = coarse, 1 = sculpted). Pinned here —
    /// like `shape_kind_discriminant` — so a future enum reorder can't silently desync
    /// the G1 shader.
    pub fn kind_discriminant(&self) -> u32 {
        match self {
            BrickPayload::CoarseSolid { .. } => 0,
            BrickPayload::Sculpted { .. } => 1,
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
    /// `block_id`'s colour index for a coarse block, the (single) microblock material for
    /// a boundary block. The R8 atlas is occupancy-only (ADR 0011 G2), so this is the
    /// per-BLOCK material the raymarch shades with, packed into the GPU record's `kind`
    /// high bits by [`pack_gpu_records`]. A block that MIXES materials across its
    /// microblocks is not brick-representable (it never engages the sink), so this holds
    /// the first microblock's material there — unused, never shaded.
    pub material_id: u16,
    /// Coarse (kind 0) or sculpted (kind 1) — see [`BrickPayload`].
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
    /// `atlas_dim_voxels³` occupancy bytes (0 empty / 255 occupied), slot-packed — the
    /// bytes [`upload_brick_atlas`] lands in the R8 3D texture.
    pub bytes: Vec<u8>,
    /// The atlas texture dimension per axis (`bricks_per_axis * brick_edge_voxels`; 0 when
    /// the build has no sculpted brick).
    pub atlas_dim_voxels: u32,
    /// Sculpted-brick tile slots per atlas axis (`ceil(cbrt(slot_count))`) — the tile-grid
    /// edge the frame scalars carry.
    pub bricks_per_axis: u32,
    /// The brick edge in voxels (`voxels_per_block`, the ONE-BLOCK granule).
    pub brick_edge_voxels: u32,
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

/// The occupancy byte a solid voxel packs to — the fog atlas's 0/255 R8 convention.
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
            atlas_dim_voxels: self.atlas_dim_voxels,
            bricks_per_axis: self.bricks_per_axis,
            brick_edge_voxels: self.brick_edge_voxels,
            sculpted_slot_count: self.sculpted_brick_count() as u32,
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

    /// How many records are sculpted bricks (== atlas slots in use; slots are assigned
    /// densely `0..count`).
    pub fn sculpted_brick_count(&self) -> usize {
        self.brick_records
            .iter()
            .filter(|record| matches!(record.payload, BrickPayload::Sculpted { .. }))
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
    let brick_edge_voxels = voxels_per_block.max(1);
    let oracle = BrickOcclusionOracle::new(two_layer_chunks);
    let mut brick_records: Vec<BrickRecord> = Vec::new();
    // One bit-packed `edge²`-word tile per sculpted brick, in slot order; unpacked into
    // the atlas cube once the final count fixes the tile geometry.
    let mut sculpted_brick_tiles: Vec<BrickOccupancyTile> = Vec::new();

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
                            seam_solidity,
                            tile,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                payload: BrickPayload::Sculpted { atlas_slot },
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

    BrickFieldBuild {
        brick_records,
        sculpted_atlas_bytes,
        brick_edge_voxels,
        bricks_per_axis,
        atlas_dim_voxels,
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
                            seam_solidity,
                            tile,
                        } => {
                            let atlas_slot = sculpted_brick_tiles.len() as u32;
                            sculpted_brick_tiles.push(tile);
                            brick_records.push(BrickRecord {
                                packed_world_block_key: pack_world_block_key(world_block),
                                material_id,
                                payload: BrickPayload::Sculpted { atlas_slot },
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

/// A boundary block's occupancy tile, **bit-packed one voxel per bit**. Under the
/// document's density bound (1..=64 — see `docs/architecture/`, the "one voxel row = one
/// machine word" ruling) a whole X-row of a brick fits in a single `u64`, so a tile is
/// `edge²` X-row words (row index `local_z * edge + local_y`, bit `x = 1 << local_x` — the
/// same x-fastest bit order [`BlockOccupancyMasks`] uses one granule coarser). This is the
/// sculpt memory story: 8× smaller than the former `edge³` byte tile at density 64. The
/// GPU R8 atlas stays byte-per-voxel; the bits are unpacked to bytes at exactly one seam
/// ([`pack_sculpted_atlas`], the `BrickFieldBuild::sculpted_atlas_bytes` boundary), never
/// on the O(volume) path.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BrickOccupancyTile {
    /// The brick edge in voxels (`= density`, guaranteed 1..=64 so an X-row fits one word).
    edge: u32,
    /// `edge²` X-row occupancy words; row index `local_z * edge + local_y`, bit `local_x`.
    /// Bits at or above `edge` in every word are always clear (writers never set them), so
    /// `count_ones` over the words is an exact voxel popcount.
    row_words: Vec<u64>,
}

impl BrickOccupancyTile {
    /// An all-air tile of the given brick edge (1..=64).
    pub fn empty(edge: u32) -> Self {
        debug_assert!(
            (1..=64).contains(&edge),
            "brick edge (density) must be 1..=64 so an X-row fits one u64"
        );
        Self {
            edge,
            row_words: vec![0u64; (edge * edge) as usize],
        }
    }

    /// Mark the contiguous X-run `min_x..=max_x` of one row occupied (a mask-OR — the
    /// bit-packed form of the cuboid row `.fill(255)`). Overflow-safe for the full word:
    /// both bounds are `< edge <= 64` hence `<= 63`, so `63 - max_x` and the shifts never
    /// wrap even when `max_x == 63` (`u64::MAX >> 0`) or `min_x == 63`.
    pub fn set_x_run(&mut self, local_y: u32, local_z: u32, min_x: u32, max_x: u32) {
        debug_assert!(min_x <= max_x, "an X-run's min must not exceed its max");
        debug_assert!(max_x < self.edge, "an X-run must stay inside the brick edge");
        let low_bits_cleared = u64::MAX << min_x;
        let high_bits_cleared = u64::MAX >> (63 - max_x);
        let run_mask = low_bits_cleared & high_bits_cleared;
        let row = (local_z * self.edge + local_y) as usize;
        self.row_words[row] |= run_mask;
    }

    /// Whether the voxel `(local_x, local_y, local_z)` is occupied.
    pub fn is_occupied(&self, local_x: u32, local_y: u32, local_z: u32) -> bool {
        let row = (local_z * self.edge + local_y) as usize;
        (self.row_words[row] >> local_x) & 1 == 1
    }

    /// The occupied voxel count (a popcount sum over the row words).
    pub fn occupied_voxel_count(&self) -> u32 {
        self.row_words
            .iter()
            .map(|word| word.count_ones())
            .sum()
    }

    /// Expand back to `edge³` occupancy bytes (`0` / [`SCULPTED_BRICK_OCCUPIED`],
    /// block-local x-fastest — the exact layout of the former byte tile). The lone unpack
    /// used at the atlas seam; O(brick), never O(volume).
    pub fn unpack_to_bytes(&self) -> Vec<u8> {
        let edge = self.edge as usize;
        let mut brick_bytes = vec![0u8; edge * edge * edge];
        for local_z in 0..edge {
            for local_y in 0..edge {
                let word = self.row_words[local_z * edge + local_y];
                if word == 0 {
                    continue;
                }
                let row = (local_z * edge + local_y) * edge;
                for local_x in 0..edge {
                    if (word >> local_x) & 1 == 1 {
                        brick_bytes[row + local_x] = SCULPTED_BRICK_OCCUPIED;
                    }
                }
            }
        }
        brick_bytes
    }

    /// Pack `edge³` occupancy bytes into the bit tile (the inverse of
    /// [`unpack_to_bytes`](Self::unpack_to_bytes)) — any nonzero byte is occupied (today's
    /// bytes are only ever `0`/`SCULPTED_BRICK_OCCUPIED`). Used to seed a mirror tile from
    /// a wholesale build's per-slot atlas bytes.
    pub fn from_bytes(edge: u32, brick_bytes: &[u8]) -> Self {
        let edge_usize = edge as usize;
        debug_assert_eq!(
            brick_bytes.len(),
            edge_usize * edge_usize * edge_usize,
            "occupancy bytes must be edge³"
        );
        let mut tile = Self::empty(edge);
        for local_z in 0..edge_usize {
            for local_y in 0..edge_usize {
                let row = (local_z * edge_usize + local_y) * edge_usize;
                let mut word = 0u64;
                for local_x in 0..edge_usize {
                    if brick_bytes[row + local_x] != 0 {
                        word |= 1u64 << local_x;
                    }
                }
                tile.row_words[local_z * edge_usize + local_y] = word;
            }
        }
        tile
    }
}

/// Rasterize one boundary block's cuboids into an `edge²`-word occupancy tile (block-local
/// x-fastest). Occupancy only: the cuboid `material_id` render-cell key (id + overlay bit)
/// never enters the R8 payload — any voxel a cuboid covers is occupied. Each cuboid row is
/// a contiguous X-run, so the former `.fill(255)` becomes a [`BrickOccupancyTile::set_x_run`]
/// mask-OR.
fn rasterize_brick_occupancy(
    geometry: &crate::two_layer_store::MicroblockGeometry,
    brick_edge_voxels: u32,
) -> BrickOccupancyTile {
    let mut tile = BrickOccupancyTile::empty(brick_edge_voxels);
    for cuboid in &geometry.cuboids {
        for voxel_z in cuboid.min[2]..=cuboid.max[2] {
            for voxel_y in cuboid.min[1]..=cuboid.max[1] {
                tile.set_x_run(voxel_y, voxel_z, cuboid.min[0], cuboid.max[0]);
            }
        }
    }
    tile
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
    /// A boundary block: the record MINUS its atlas slot (the caller's allocator assigns
    /// it) plus the occupancy tile to land in that slot.
    Sculpted {
        material_id: u16,
        seam_solidity: SeamSolidity,
        tile: BrickOccupancyTile,
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
            payload: BrickPayload::CoarseSolid { block_id },
            // Fully solid through ⇒ every face is solid.
            seam_solidity: SeamSolidity {
                solid: [[true; 2]; 3],
            },
        })
    } else if let Some(geometry) = chunk.microblocks.get(&block) {
        // The block's material is the clean render-cell id of its microblocks; a
        // representable block is single-material, so the first cuboid's id is the
        // block's (a mixed block never engages the sink).
        let material_id = geometry
            .cuboids
            .first()
            .map(|cuboid| clean_block_id(cuboid.material_id))
            .unwrap_or(0);
        BlockBrick::Sculpted {
            material_id,
            seam_solidity: geometry.seam_solidity,
            tile: rasterize_brick_occupancy(geometry, brick_edge_voxels),
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
fn pack_sculpted_atlas(
    slot_tiles: &[BrickOccupancyTile],
    brick_edge_voxels: u32,
) -> (u32, u32, Vec<u8>) {
    let edge = brick_edge_voxels as usize;
    let slot_count = slot_tiles.len();
    let (bricks_per_axis, atlas_dim_voxels) = if slot_count == 0 {
        (0, 0)
    } else {
        let tiles = ((slot_count as f64).cbrt().ceil() as u32).max(1);
        (tiles, tiles * brick_edge_voxels)
    };
    let atlas_dim = atlas_dim_voxels as usize;
    let mut bytes = vec![0u8; atlas_dim * atlas_dim * atlas_dim];
    // The one unpack seam: the bit tiles expand to R8 bytes here (O(brick) per slot), so
    // everything GPU-facing keeps consuming `sculpted_atlas_bytes` unchanged.
    for (slot, tile) in slot_tiles.iter().enumerate() {
        debug_assert_eq!(
            tile.edge, brick_edge_voxels,
            "every slot tile shares the build's brick edge"
        );
        let tiles = bricks_per_axis;
        let s = slot as u32;
        let origin = [
            (s % tiles) as usize * edge,
            ((s / tiles) % tiles) as usize * edge,
            (s / (tiles * tiles)) as usize * edge,
        ];
        for local_z in 0..edge {
            for local_y in 0..edge {
                let word = tile.row_words[local_z * edge + local_y];
                if word == 0 {
                    continue;
                }
                let atlas_row = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0];
                for local_x in 0..edge {
                    if (word >> local_x) & 1 == 1 {
                        bytes[atlas_row + local_x] = SCULPTED_BRICK_OCCUPIED;
                    }
                }
            }
        }
    }
    (bricks_per_axis, atlas_dim_voxels, bytes)
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
    /// [`BrickOccupancyTile`]), indexed by atlas slot. A FREED slot's entry is retained
    /// (kept `edge²` words so the atlas packer never trips) but is unreferenced — dead
    /// bits until the slot is reallocated.
    slot_tiles: Vec<BrickOccupancyTile>,
    /// Reusable slot indices freed by removed / transitioned sculpted bricks — the
    /// free-list. A new sculpted brick pops from here before growing `slot_tiles`.
    free_slots: Vec<u32>,
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
        let sculpted_count = build.sculpted_brick_count();
        // Unpack the flat atlas bytes into the mirror's bit tiles BEFORE moving the blob into
        // the payload (the one O(sculpted) seeding cost, unchanged from before).
        let slot_tiles: Vec<BrickOccupancyTile> = (0..sculpted_count as u32)
            .map(|slot| {
                BrickOccupancyTile::from_bytes(
                    build.brick_edge_voxels,
                    &build.sculpted_brick_occupancy(slot),
                )
            })
            .collect();
        let BrickFieldBuild {
            brick_records,
            sculpted_atlas_bytes,
            brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        } = build;
        let payload = SculptedAtlasPayload {
            bytes: sculpted_atlas_bytes,
            atlas_dim_voxels,
            bricks_per_axis,
            brick_edge_voxels,
            sculpted_slot_count: sculpted_count as u32,
        };
        let mirror = Self {
            brick_edge_voxels,
            records: brick_records,
            slot_tiles,
            free_slots: Vec::new(),
        };
        (mirror, payload)
    }

    /// The live records — the sorted [`BrickRecord`] array the GPU record pack + the
    /// pyramid derive from. The mirror is the single CPU owner (item 9): the renderer's
    /// install/patch seams read records straight from here, never via [`to_build`](Self::to_build).
    pub fn records(&self) -> &[BrickRecord] {
        &self.records
    }

    /// How many live records are sculpted bricks (mirror of
    /// [`BrickFieldBuild::sculpted_brick_count`]) — the wholesale install's slot count.
    pub fn sculpted_brick_count(&self) -> usize {
        self.records
            .iter()
            .filter(|record| matches!(record.payload, BrickPayload::Sculpted { .. }))
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
        self.slot_tiles[slot as usize].unpack_to_bytes()
    }

    /// Materialise the full atlas as a [`SculptedAtlasPayload`] — the ONE legitimate
    /// wholesale re-pack, done only on an atlas GROW (`BrickFieldUpdate::atlas_grew`) where
    /// every slot's 3D position moved. Reuses [`pack_sculpted_atlas`] so it stays
    /// byte-identical to [`to_build`](Self::to_build)'s atlas.
    pub fn pack_atlas_payload(&self) -> SculptedAtlasPayload {
        let (bricks_per_axis, atlas_dim_voxels, bytes) =
            pack_sculpted_atlas(&self.slot_tiles, self.brick_edge_voxels);
        SculptedAtlasPayload {
            bytes,
            atlas_dim_voxels,
            bricks_per_axis,
            brick_edge_voxels: self.brick_edge_voxels,
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
    ///   ([`TwoLayerResidentCache::invalidate_aabb`](crate::two_layer_store::TwoLayerResidentCache::invalidate_aabb)
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

        // 1. Drop every previous record whose block is in a dirty chunk (freeing its slot),
        //    and every COARSE record of a ring chunk (its occlusion verdict may have flipped;
        //    ring SCULPTED records are kept — their chunk's data is unchanged, so record and
        //    slot are still exact, and the atlas is never touched for the ring).
        let mut freed_slots = Vec::new();
        self.records.retain(|record| {
            let chunk =
                chunk_coord_of_world_block(unpack_world_block_key(record.packed_world_block_key));
            if dirty.contains(&chunk) {
                if let BrickPayload::Sculpted { atlas_slot } = record.payload {
                    freed_slots.push(atlas_slot);
                }
                false
            } else if ring.contains(&chunk) {
                matches!(record.payload, BrickPayload::Sculpted { .. })
            } else {
                true
            }
        });
        // Freed slots return to the pool (ascending pop order keeps allocation
        // deterministic — a nicety for test readability, not correctness).
        self.free_slots.extend(freed_slots.iter().copied());
        self.free_slots.sort_unstable();
        self.free_slots.dedup();

        // 2. Rebuild the dirty chunks' records fully — and the ring chunks' COARSE records —
        //    from the FRESH data, with occlusion verdicts from the fresh oracle (the same
        //    fused elision `build_brick_field` performs, so incremental == wholesale stays
        //    structural).
        let oracle = BrickOcclusionOracle::new(fresh_chunks);
        let mut written_slots = Vec::new();
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
                                seam_solidity,
                                tile,
                            } => {
                                // Ring chunks keep their existing sculpted records (data
                                // unchanged); only a DIRTY chunk re-allocates and rewrites.
                                if !chunk_is_dirty {
                                    continue;
                                }
                                let slot = self.allocate_slot(tile);
                                written_slots.push(slot);
                                self.records.push(BrickRecord {
                                    packed_world_block_key: pack_world_block_key(world_block),
                                    material_id,
                                    payload: BrickPayload::Sculpted { atlas_slot: slot },
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
        BrickFieldUpdate {
            written_slots,
            freed_slots,
            atlas_grew,
        }
    }

    /// Allocate a slot for a fresh sculpted tile: reuse a freed slot if one is available
    /// (keeping the high-water mark — and thus the atlas — from growing needlessly),
    /// else append a new slot.
    fn allocate_slot(&mut self, tile: BrickOccupancyTile) -> u32 {
        match self.free_slots.pop() {
            Some(slot) => {
                self.slot_tiles[slot as usize] = tile;
                slot
            }
            None => {
                let slot = self.slot_tiles.len() as u32;
                self.slot_tiles.push(tile);
                slot
            }
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
            pack_sculpted_atlas(&self.slot_tiles, self.brick_edge_voxels);
        BrickFieldBuild {
            brick_records: self.records.clone(),
            sculpted_atlas_bytes,
            brick_edge_voxels: self.brick_edge_voxels,
            bricks_per_axis,
            atlas_dim_voxels,
        }
    }
}

/// The `bricks_per_axis` a slot-tile count packs to (`ceil(cbrt(count))`, 0 for empty) —
/// the atlas tile-grid edge, shared by the packer and the grow test.
fn sculpted_atlas_bricks_per_axis(slot_count: usize) -> u32 {
    if slot_count == 0 {
        0
    } else {
        ((slot_count as f64).cbrt().ceil() as u32).max(1)
    }
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
    let atlas_dim = atlas.atlas_dim_voxels.max(1);
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
    if atlas.atlas_dim_voxels > 0 {
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
    use crate::core_geom::MaterialChoice;
    use crate::scene::Scene;
    use crate::two_layer_store::TwoLayerStore;
    use crate::voxel::{GeometryParams, ShapeKind, Voxel};

    /// Byte↔bit parity gate (the density≤64 X-row packing oracle, `docs/architecture/`):
    /// for a spread of cuboid fixtures a bit-packed [`BrickOccupancyTile`] unpacks to
    /// EXACTLY the bytes a naive per-voxel dense rasterize produces, `is_occupied` agrees
    /// per voxel, `from_bytes` is the inverse, and the popcount matches. The edge cases
    /// (density 1, 32, 33 — spanning bit 32 — and 64 — the full word) exercise the
    /// mask math the GPU-atlas seam depends on.
    #[test]
    fn bit_tile_unpacks_byte_identical_to_dense_rasterize() {
        type Cuboid = ([u32; 3], [u32; 3]);
        let fixtures: &[(u32, &[Cuboid])] = &[
            (1, &[([0, 0, 0], [0, 0, 0])]),                 // density 1, single voxel
            (4, &[([1, 1, 1], [1, 1, 1])]),                 // single interior voxel
            (4, &[([0, 0, 0], [3, 3, 3])]),                 // full block
            (8, &[([0, 3, 2], [7, 3, 2])]),                 // full X-row slab
            (8, &[([0, 0, 0], [7, 2, 0]), ([0, 3, 3], [2, 7, 7])]), // L-shaped two cuboids
            (16, &[([2, 5, 9], [13, 10, 12])]),             // arbitrary interior box
            (32, &[([0, 0, 0], [31, 31, 31])]),             // full density-32 block
            (33, &[([0, 0, 0], [32, 0, 0]), ([32, 32, 32], [32, 32, 32])]), // spans bit 32
            (64, &[([0, 0, 0], [63, 0, 0])]),               // full 64-bit X-row
            (64, &[([0, 0, 0], [63, 63, 63])]),             // full density-64 block
        ];
        for (edge, cuboids) in fixtures {
            let edge = *edge;
            let e = edge as usize;
            // Reference: naive per-voxel dense byte fill (the pre-packing rasterize).
            let mut reference = vec![0u8; e * e * e];
            for (min, max) in cuboids.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        for x in min[0]..=max[0] {
                            reference[(z as usize * e + y as usize) * e + x as usize] =
                                SCULPTED_BRICK_OCCUPIED;
                        }
                    }
                }
            }
            // Bit path via `set_x_run` (the op `rasterize_brick_occupancy` now performs).
            let mut tile = BrickOccupancyTile::empty(edge);
            for (min, max) in cuboids.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        tile.set_x_run(y, z, min[0], max[0]);
                    }
                }
            }
            assert_eq!(tile.unpack_to_bytes(), reference, "edge {edge} unpack mismatch");
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let expected = reference[(z as usize * e + y as usize) * e + x as usize] != 0;
                        assert_eq!(
                            tile.is_occupied(x, y, z),
                            expected,
                            "edge {edge} is_occupied mismatch at ({x},{y},{z})"
                        );
                    }
                }
            }
            assert_eq!(
                BrickOccupancyTile::from_bytes(edge, &reference),
                tile,
                "edge {edge} from_bytes is not the inverse of unpack"
            );
            let occupied = reference.iter().filter(|byte| **byte != 0).count() as u32;
            assert_eq!(
                tile.occupied_voxel_count(),
                occupied,
                "edge {edge} popcount mismatch"
            );
        }
    }

    /// `set_x_run`'s mask math stays overflow-safe at the full 64-bit word (the case that
    /// would wrap a naive `(1 << (max+1)) - 1`): full-word, high-only, low-only, an
    /// interior run, and the density-1 degenerate all set exactly the intended bits.
    #[test]
    fn set_x_run_masks_are_overflow_safe_at_full_word() {
        let mut full = BrickOccupancyTile::empty(64);
        full.set_x_run(0, 0, 0, 63);
        assert_eq!(full.row_words[0], u64::MAX);
        assert_eq!(full.occupied_voxel_count(), 64);

        let mut high = BrickOccupancyTile::empty(64);
        high.set_x_run(0, 0, 63, 63);
        assert_eq!(high.row_words[0], 1u64 << 63);

        let mut low = BrickOccupancyTile::empty(64);
        low.set_x_run(0, 0, 0, 0);
        assert_eq!(low.row_words[0], 1u64);

        let mut interior = BrickOccupancyTile::empty(64);
        interior.set_x_run(0, 0, 5, 40);
        let expected_interior = (u64::MAX >> (63 - 40)) & (u64::MAX << 5);
        assert_eq!(interior.row_words[0], expected_interior);
        assert_eq!(interior.occupied_voxel_count(), 40 - 5 + 1);

        let mut degenerate = BrickOccupancyTile::empty(1);
        degenerate.set_x_run(0, 0, 0, 0);
        assert!(degenerate.is_occupied(0, 0, 0));
        assert_eq!(degenerate.occupied_voxel_count(), 1);
    }

    #[test]
    fn world_block_key_round_trips_and_orders_z_major() {
        let coordinates = [
            [0i64, 0, 0],
            [-1, -2, -3],
            [17, -300, 4096],
            [-(1 << 19), (1 << 19), 0],
        ];
        for &world_block in &coordinates {
            assert_eq!(
                unpack_world_block_key(pack_world_block_key(world_block)),
                world_block
            );
        }
        // Integer key order is (z, y, x) lexicographic — the sort the shader's
        // binary search relies on.
        assert!(pack_world_block_key([5, 0, 0]) < pack_world_block_key([0, 1, 0]));
        assert!(pack_world_block_key([0, 5, 0]) < pack_world_block_key([0, 0, 1]));
        assert!(pack_world_block_key([-1, 0, 0]) < pack_world_block_key([0, 0, 0]));
    }

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
                let shape = crate::voxel::SdfShape::from_blocks(
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
                    crate::voxel::SdfShape::from_blocks(ShapeKind::Sphere, [3, 3, 3], 1, voxels_per_block);
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
                let shape = crate::voxel::SdfShape::from_blocks(
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
                assert!(!masks.cell_keys.is_empty(), "the scene must occupy blocks");

                // Every full-record block reads as an occupied bit.
                let cell_size = BLOCK_OCCUPANCY_CELL_BLOCKS as i64;
                let bit_set = |world_block: [i64; 3]| -> bool {
                    let cell = [
                        world_block[0].div_euclid(cell_size),
                        world_block[1].div_euclid(cell_size),
                        world_block[2].div_euclid(cell_size),
                    ];
                    let Ok(index) = masks.cell_keys.binary_search(&pack_world_block_key(cell)) else {
                        return false;
                    };
                    let local = [
                        world_block[0].rem_euclid(cell_size) as usize,
                        world_block[1].rem_euclid(cell_size) as usize,
                        world_block[2].rem_euclid(cell_size) as usize,
                    ];
                    let bit = (local[2] * cell_size as usize + local[1]) * cell_size as usize
                        + local[0];
                    masks.cell_masks[index][bit / 32] & (1u32 << (bit % 32)) != 0
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
                for mask in &masks.cell_masks {
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
    use crate::core_geom::MaterialChoice;
    use crate::scene::{Node, NodeContent, NodeTransform, Scene};
    use crate::two_layer_store::{TwoLayerChunk, TwoLayerResidentCache};
    use crate::voxel::{ShapeKind, SdfShape};

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
            let producer = crate::sketch::SketchSolid::extrude(
                crate::sketch::Sketch::rectangle(
                    crate::sketch::PlaneAxis::Z,
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
}
