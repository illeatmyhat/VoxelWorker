use super::*;

// The block-occupancy masks' STORAGE is substrate's `SortedKeyBitmaskMap`: a sorted parallel-array
// map (keys ∥ fixed-width bitmasks ∥ per-key fallback scalar), binary-searchable, with the textbook
// word/bit indexing. The domain keeps the "occupancy masks" name and the CHUNK traversal
// (`BlockOccupancyMasks::from_chunks`, its solid-chunk bulk fast path, and its first-writer-wins
// fallback-material policy) at this seam; the parallel-array shape, the sort-by-key construction,
// the binary search, and the bit set/test live in the substrate module (fallback = a caller-defined
// `u32`, here the render-cell material colour index). See docs/architecture/03-display.md (the
// band-clip interior fallback) for how the packed cells feed the raymarch.
use substrate::occupancy::bitmask_map::{set_mask_bit, SortedKeyBitmaskMap};

/// The clip-map cell edge (in blocks) the [`BlockOccupancyMasks`] bitmask cells use —
/// the same 8-block granule as the pyramid's [`ClipmapLevel`] L1, so a `512`-block
/// interior-occupancy cell is one `u32[16]` bitmask.
pub const BLOCK_OCCUPANCY_CELL_BLOCKS: u32 = CLIPMAP_LEVEL_1_BLOCKS_PER_CELL;
/// Blocks per [`BlockOccupancyMasks`] cell (`8³ = 512`) — the bitmask's bit count.
pub(crate) const BLOCK_OCCUPANCY_BITS_PER_CELL: usize =
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
/// GPU seam ([`OccupancyCellPod`](crate::brick) reads them as two fields). With the
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
