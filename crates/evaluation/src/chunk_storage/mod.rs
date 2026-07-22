//! Lossless compressed storage for a single resolved chunk grid (issue #20 S6a).
//!
//! The out-of-core store (#20) needs a compact, serialisable on-disk form for a
//! resolved per-chunk [`VoxelGrid`] (the grids [`crate::chunk_cache`] resolves and
//! caches). A resolved chunk is almost always **mostly air** with only a handful of
//! distinct materials, so the dense `dimensions³ × (position + material)` form a
//! `VoxelGrid` carries in RAM is hugely wasteful as a storage shape. This module is
//! that storage shape — the data structure plus its lossless round-trip proof — and
//! it IS wired into the live resolve path: `Store`'s out-of-core spill (in
//! `crate::store::cache`, backed by `crate::disk_chunk_store`) compresses an evicted
//! resident chunk through `compress` before writing it to disk and restores it through
//! `decompress` on the next access. The round-trip is lossless, so no golden is
//! affected by a chunk having spilled and reloaded.
//!
//! ## Why this is lossless (ADR 0003 §3a — the payload is already integer)
//!
//! Since ADR 0003 §3a the per-voxel payload stores the voxel's INTEGER index
//! (`Voxel::local_index`) directly — the f32 centre is only ever reconstructed at
//! consumption as `index + 0.5` ([`voxel_core::voxel::Voxel::world_position`]). This codec
//! therefore consumes the stored integer DIRECTLY (it no longer reverse-engineers an
//! index out of an f32 via `floor()` + a uniform fractional-part debug-assert). We store
//! the integer index relative to the chunk's min corner (in i64 so a far-placed chunk
//! keeps full precision) and rebuild `local_index` as `min_corner + local`, the exact
//! inverse. The `centre_fraction` field is retained for on-disk-format stability and is
//! the constant `0.5` (a resolved voxel centre is always a half-integer).
//!
//! `block_local_coord` and the categorical `block_id` are stored directly (the former is
//! the producer's intra-block coordinate, which a rebase by a non-block-multiple origin
//! can decouple from the absolute index, so it is NOT reconstructed from the index; the
//! latter is the categorical block-palette id of §3a). The transient `grid_overlay`
//! render marker (§3c) is NOT serialized — it is a resolve→mesh render hint, not stored
//! truth (a chunk reloaded from a disk spill resolves its overlay afresh on the next
//! mesh).
//!
//! ## Encoding choice — sparse, with a dense bit-packed fallback (heuristic)
//!
//! A resolved chunk is typically a thin SDF shell inside a `64³`-capacity chunk, so
//! it is overwhelmingly empty. The default encoding is therefore **sparse**: a
//! material palette plus, for each occupied cell, its `(local_linear_index,
//! palette_index, block_local_coord)`. For the rare dense chunk (a near-solid box)
//! a sparse per-cell record costs more than one byte per cell would, so
//! [`compress`] also evaluates a **dense bit-packed** palette-index array (one
//! `ceil(log2(palette_len))`-bit index per cell, air = a reserved palette slot) and
//! keeps whichever is smaller, recording the winner in [`Occupancy`]. Both encodings
//! are exact inverses of [`decompress`]; the heuristic only ever changes the byte
//! cost, never the decoded grid.

use serde::{Deserialize, Serialize};

use voxel_core::voxel::{Voxel, VoxelGrid};

/// A single occupied cell in the **sparse** encoding: where it is and what it is.
///
/// `local_linear_index` is the cell's index into the chunk's occupied bounding box
/// (row-major over `[span_x, span_y, span_z]`), so it is reconstructed back to an
/// absolute voxel index via the chunk's `min_corner_voxels` + the box spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseCell {
    /// Row-major index into the occupied bounding box `[span_x, span_y, span_z]`.
    pub local_linear_index: u64,
    /// Index into [`CompressedChunk::material_palette`].
    pub palette_index: u32,
    /// The producer's intra-block coordinate `(i % d, j % d, k % d)`, preserved
    /// verbatim (a non-block-aligned rebase can decouple it from the absolute
    /// index, so it is NOT derived from the position).
    pub block_local_coord: [u8; 3],
}

/// The occupancy payload: either a sparse per-occupied-cell list (great for the
/// usual mostly-empty chunk) or a dense bit-packed palette-index array (smaller for
/// a near-solid chunk). [`compress`] picks whichever serialises smaller.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Occupancy {
    /// One record per occupied cell. Empty for an empty chunk.
    Sparse(Vec<SparseCell>),
    /// A dense palette-index per cell over the occupied bounding box, bit-packed at
    /// `bits_per_index` bits each (air uses the reserved palette index
    /// [`AIR_PALETTE_INDEX`]). `block_local_coords` carries one entry per occupied
    /// cell in row-major scan order (same order the packed indices are walked), so a
    /// dense chunk still preserves each voxel's intra-block coordinate.
    Dense {
        /// Bits used per cell index (`ceil(log2(palette_len + 1))`, min 1).
        bits_per_index: u8,
        /// Little-endian bit-packed palette indices, one per cell in the box.
        packed_indices: Vec<u8>,
        /// Intra-block coordinates for the occupied cells, in row-major scan order.
        block_local_coords: Vec<[u8; 3]>,
    },
}

/// The reserved palette index meaning "air" (empty cell) in the dense encoding. The
/// real material palette indices are offset by one in the dense packing so this slot
/// is always free; the sparse encoding never emits an air record so it does not use
/// it.
pub const AIR_PALETTE_INDEX: u32 = 0;

/// A compact, lossless, serialisable representation of one resolved chunk grid.
///
/// Holds (a) the chunk's full voxel dimensions and its occupied bounding box
/// (min-corner + spans, in absolute voxel-index space so a far chunk keeps
/// precision), (b) a small material palette (the distinct `material_id`s present, in
/// first-seen order, de-duplicated), and (c) the occupancy ([`Occupancy`], sparse or
/// dense). [`compress`] / [`decompress`] are exact inverses.
///
/// `Eq` is intentionally NOT derived: `centre_fraction` is `f32`. The stored
/// fractions are exact small constants (`0.0` / `0.5`), so `PartialEq` compares them
/// bit-for-bit in practice, which is all the serde round-trip assertion needs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressedChunk {
    /// The chunk grid's full voxel dimensions (`VoxelGrid::dimensions`), preserved
    /// so an empty chunk still round-trips its size.
    pub dimensions: [u32; 3],
    /// The voxel-index min corner of the occupied bounding box (in the chunk grid's
    /// carried frame). The `local_index` of a voxel at local box offset `o` is
    /// `min_corner_voxels + o`; its f32 centre is `that + 0.5`. `[0; 3]` for an empty
    /// chunk.
    pub min_corner_voxels: [i64; 3],
    /// The shared fractional part of every occupied voxel centre, per axis. Since the
    /// payload is integer (ADR 0003 §3a) every reconstructed centre is `index + 0.5`, so
    /// this is the constant `0.5` for a non-empty chunk (retained for on-disk-format
    /// stability). `[0.0; 3]` for an empty chunk.
    pub centre_fraction: [f32; 3],
    /// The occupied bounding box spans (per axis, in voxels). `[0; 3]` for an empty
    /// chunk. `local_linear_index` in the sparse encoding (and the dense cell count)
    /// is row-major over these spans.
    pub box_spans: [u32; 3],
    /// The distinct categorical `block_id`s present (as `u16`), in first-seen scan order,
    /// no duplicates (ADR 0003 §3a — the block palette, not a 3-value material).
    pub material_palette: Vec<u16>,
    /// The occupancy payload (sparse or dense — whichever is smaller).
    pub occupancy: Occupancy,
}

impl CompressedChunk {
    /// Number of occupied cells this compressed chunk encodes (matches the source
    /// grid's `occupied_count`). Cheap for either encoding.
    pub fn occupied_count(&self) -> usize {
        match &self.occupancy {
            Occupancy::Sparse(cells) => cells.len(),
            Occupancy::Dense { block_local_coords, .. } => block_local_coords.len(),
        }
    }
}

/// Compress a resolved chunk [`VoxelGrid`] into a [`CompressedChunk`] (lossless).
///
/// Builds the material palette (distinct ids in first-seen order), the occupied
/// bounding box, then both a sparse and (when it could win) a dense bit-packed
/// occupancy, keeping whichever serialises smaller. The result decompresses to a
/// grid equal to `grid` in dimensions, occupied set, per-voxel `material_id` and
/// per-voxel `block_local_coord` (order of the occupied vec may differ — the grid's
/// occupied set is order-independent, exactly as the resolve path treats it).
pub fn compress(grid: &VoxelGrid) -> CompressedChunk {
    // Empty chunk: nothing to box, palette empty, sparse-empty occupancy.
    if grid.occupied.is_empty() {
        return CompressedChunk {
            dimensions: grid.dimensions,
            min_corner_voxels: [0; 3],
            centre_fraction: [0.0; 3],
            box_spans: [0; 3],
            material_palette: Vec::new(),
            occupancy: Occupancy::Sparse(Vec::new()),
        };
    }

    // ADR 0003 §3a: every resolved voxel centre is `index + 0.5` (the payload now stores
    // the integer `local_index` directly), so the shared per-axis fractional offset is a
    // constant `0.5` — there is nothing to reverse-engineer out of an f32 any more. Kept
    // as a stored field so the on-disk format is stable and `decompress` rebuilds the
    // `world_position()`-equivalent centre.
    let centre_fraction = [0.5f32; 3];

    // 1) Occupied bounding box in the grid's integer index space (read DIRECTLY from the
    //    stored `local_index`, no f32 round-trip — ADR 0003 §3a: the codec consumes the
    //    integer rather than recovering it from a position).
    let mut min_corner = [i64::MAX; 3];
    let mut max_corner = [i64::MIN; 3];
    for voxel in &grid.occupied {
        for axis in 0..3 {
            let index = voxel.local_index[axis] as i64;
            min_corner[axis] = min_corner[axis].min(index);
            max_corner[axis] = max_corner[axis].max(index);
        }
    }
    let box_spans = [
        (max_corner[0] - min_corner[0] + 1) as u32,
        (max_corner[1] - min_corner[1] + 1) as u32,
        (max_corner[2] - min_corner[2] + 1) as u32,
    ];

    // 2) Block palette: distinct categorical block ids, first-seen order, de-duplicated
    //    (ADR 0003 §3a — the palette indexes the categorical cell, not a 3-value material).
    let mut material_palette: Vec<u16> = Vec::new();
    for voxel in &grid.occupied {
        if !material_palette.contains(&voxel.block_id.0) {
            material_palette.push(voxel.block_id.0);
        }
    }
    let palette_index_of = |block_id: u16| -> u32 {
        material_palette
            .iter()
            .position(|&id| id == block_id)
            .expect("every block id was inserted into the palette above") as u32
    };

    // 3) Sparse occupancy (always built — it is the default and the size baseline).
    let span_xy = box_spans[0] as u64 * box_spans[1] as u64;
    let local_linear_index = |voxel: &Voxel| -> u64 {
        let local = [
            (voxel.local_index[0] as i64 - min_corner[0]) as u64,
            (voxel.local_index[1] as i64 - min_corner[1]) as u64,
            (voxel.local_index[2] as i64 - min_corner[2]) as u64,
        ];
        local[2] * span_xy + local[1] * box_spans[0] as u64 + local[0]
    };
    let sparse_cells: Vec<SparseCell> = grid
        .occupied
        .iter()
        .map(|voxel| SparseCell {
            local_linear_index: local_linear_index(voxel),
            palette_index: palette_index_of(voxel.block_id.0),
            block_local_coord: voxel.block_local_coord,
        })
        .collect();
    let sparse = Occupancy::Sparse(sparse_cells);

    // 4) Dense bit-packed occupancy, but only when the box is small enough that a
    //    full per-cell array is even plausibly smaller — a fully-occupied box is the
    //    win case. `bits_per_index` reserves the air slot (palette_len + 1 values).
    let cell_count = box_spans[0] as u64 * box_spans[1] as u64 * box_spans[2] as u64;
    let occupancy = match build_dense_if_smaller(
        grid,
        cell_count,
        material_palette.len(),
        local_linear_index,
        palette_index_of,
        &sparse,
    ) {
        Some(dense) => dense,
        None => sparse,
    };

    CompressedChunk {
        dimensions: grid.dimensions,
        min_corner_voxels: min_corner,
        centre_fraction,
        box_spans,
        material_palette,
        occupancy,
    }
}

/// Build the dense bit-packed occupancy and return it **only if** it serialises
/// smaller than the already-built `sparse` occupancy; otherwise `None` (keep sparse).
///
/// The dense form is a `bits_per_index`-bit palette index per cell over the whole
/// occupied box (air = [`AIR_PALETTE_INDEX`], real materials offset by one), plus the
/// occupied cells' `block_local_coord`s in scan order. It wins for near-solid chunks.
fn build_dense_if_smaller(
    grid: &VoxelGrid,
    cell_count: u64,
    palette_len: usize,
    // The SAME row-major linear index + palette lookup the sparse pass built above,
    // passed in rather than re-derived — one formula, no drift.
    local_linear_index: impl Fn(&Voxel) -> u64,
    palette_index_of: impl Fn(u16) -> u32,
    sparse: &Occupancy,
) -> Option<Occupancy> {
    // Guard against a pathologically huge box (a sparse pair of far-apart voxels in
    // one chunk): a dense array over the whole box would be enormous, so never even
    // build it past a sane ceiling — sparse always wins there anyway.
    const MAX_DENSE_CELLS: u64 = 4 * 1024 * 1024; // 4M cells (a 64³ chunk is 262k).
    if cell_count == 0 || cell_count > MAX_DENSE_CELLS {
        return None;
    }

    // palette_len + 1 distinct values (air + each material) → bits per index.
    let value_count = palette_len as u64 + 1;
    let bits_per_index = bits_for_value_count(value_count);

    // Air-filled palette-index grid (air = AIR_PALETTE_INDEX = 0), then stamp each
    // occupied cell with (palette_index + 1) and record its block-local coord.
    let mut cell_values = vec![AIR_PALETTE_INDEX; cell_count as usize];
    let mut block_local_coords: Vec<[u8; 3]> = Vec::with_capacity(grid.occupied.len());
    // Walk occupied in row-major order so block_local_coords align with the packed
    // scan; collect (linear_index, voxel) then sort by index.
    let mut indexed: Vec<(u64, &Voxel)> = grid
        .occupied
        .iter()
        .map(|voxel| (local_linear_index(voxel), voxel))
        .collect();
    indexed.sort_by_key(|(linear, _)| *linear);
    for (linear, voxel) in &indexed {
        cell_values[*linear as usize] = palette_index_of(voxel.block_id.0) + 1;
        block_local_coords.push(voxel.block_local_coord);
    }

    let packed_indices = pack_indices(&cell_values, bits_per_index);
    let dense = Occupancy::Dense {
        bits_per_index,
        packed_indices,
        block_local_coords,
    };

    // Keep dense only if its binary layout is strictly smaller than sparse's.
    let dense_size = occupancy_binary_size(&dense);
    let sparse_size = occupancy_binary_size(sparse);
    if dense_size < sparse_size {
        Some(dense)
    } else {
        None
    }
}

/// Decompress a [`CompressedChunk`] back into a [`VoxelGrid`] (exact inverse of
/// [`compress`]).
///
/// Reconstructs each occupied voxel's f32 position from `min_corner_voxels` + its
/// local box offset (`(min + offset) as f32 + 0.5`, the producer's own arithmetic),
/// its `material_id` from the palette, and its `block_local_coord` verbatim. The
/// resulting grid equals the source grid in dimensions, occupied set, per-voxel
/// `material_id` and per-voxel `block_local_coord`.
pub fn decompress(compressed: &CompressedChunk) -> VoxelGrid {
    let mut grid = VoxelGrid::new(compressed.dimensions);
    let [span_x, span_y, _span_z] = compressed.box_spans;
    let span_xy = span_x as u64 * span_y as u64;
    let min_corner = compressed.min_corner_voxels;

    // Rebuild the INTEGER index of a cell at row-major local index `linear` (ADR 0003
    // §3a: the payload stores the integer directly, so the codec restores it directly —
    // `world_position()` reconstructs the `index + 0.5` centre at consumption). The
    // stored `centre_fraction` is the constant 0.5 and is asserted, not used to rebuild.
    debug_assert!(
        compressed.centre_fraction.iter().all(|&fraction| fraction == 0.5)
            || compressed.occupied_count() == 0,
        "a non-empty resolved chunk's voxel centres share the 0.5 fraction"
    );
    let index_of = |linear: u64| -> [i32; 3] {
        let local_x = if span_x == 0 { 0 } else { linear % span_x as u64 };
        let local_y = if span_y == 0 {
            0
        } else {
            (linear / span_x as u64) % span_y as u64
        };
        let local_z = linear.checked_div(span_xy).unwrap_or(0);
        [
            (min_corner[0] + local_x as i64) as i32,
            (min_corner[1] + local_y as i64) as i32,
            (min_corner[2] + local_z as i64) as i32,
        ]
    };

    match &compressed.occupancy {
        Occupancy::Sparse(cells) => {
            grid.occupied.reserve(cells.len());
            for cell in cells {
                grid.occupied.push(Voxel {
                    local_index: index_of(cell.local_linear_index),
                    block_local_coord: cell.block_local_coord,
                    block_id: voxel_core::core_geom::BlockId(
                        compressed.material_palette[cell.palette_index as usize],
                    ),
                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                    grid_overlay: false,
                });
            }
        }
        Occupancy::Dense {
            bits_per_index,
            packed_indices,
            block_local_coords,
        } => {
            let cell_count = compressed.box_spans[0] as u64
                * compressed.box_spans[1] as u64
                * compressed.box_spans[2] as u64;
            let mut next_coord = 0usize;
            for linear in 0..cell_count {
                let value = read_packed_index(packed_indices, *bits_per_index, linear as usize);
                if value == AIR_PALETTE_INDEX {
                    continue;
                }
                let palette_index = (value - 1) as usize;
                grid.occupied.push(Voxel {
                    local_index: index_of(linear),
                    block_local_coord: block_local_coords[next_coord],
                    block_id: voxel_core::core_geom::BlockId(
                        compressed.material_palette[palette_index],
                    ),
                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                    grid_overlay: false,
                });
                next_coord += 1;
            }
        }
    }
    grid
}

/// Bits needed to represent `value_count` distinct values (min 1, so a single-value
/// palette still packs at 1 bit rather than 0).
fn bits_for_value_count(value_count: u64) -> u8 {
    let mut bits = 1u8;
    while (1u64 << bits) < value_count {
        bits += 1;
    }
    bits
}

/// Bit-pack `values` (each `< 2^bits`) little-endian into a byte vector.
fn pack_indices(values: &[u32], bits: u8) -> Vec<u8> {
    let total_bits = values.len() as u64 * bits as u64;
    let byte_len = total_bits.div_ceil(8) as usize;
    let mut packed = vec![0u8; byte_len];
    let mut bit_cursor = 0u64;
    for &value in values {
        for bit in 0..bits {
            if (value >> bit) & 1 == 1 {
                let position = bit_cursor + bit as u64;
                packed[(position / 8) as usize] |= 1 << (position % 8);
            }
        }
        bit_cursor += bits as u64;
    }
    packed
}

/// Read the `index`-th `bits`-wide little-endian value from a packed byte vector.
fn read_packed_index(packed: &[u8], bits: u8, index: usize) -> u32 {
    let mut value = 0u32;
    let bit_cursor = index as u64 * bits as u64;
    for bit in 0..bits {
        let position = bit_cursor + bit as u64;
        let byte = packed[(position / 8) as usize];
        if (byte >> (position % 8)) & 1 == 1 {
            value |= 1 << bit;
        }
    }
    value
}

/// The **binary** on-disk byte size of an occupancy payload — the heuristic's size
/// metric and the honest figure the compression-ratio report uses.
///
/// This models the compact binary layout the future disk store will write (NOT a
/// verbose text format): a sparse cell is the packed index + palette index + 3
/// block-local bytes; the dense form is the bit-packed array plus 3 bytes per
/// occupied cell. JSON is deliberately NOT used as the size proxy — its text
/// encoding of numbers and byte arrays inflates the figure several-fold and would
/// make both the heuristic and the reported ratio meaningless.
fn occupancy_binary_size(occupancy: &Occupancy) -> usize {
    match occupancy {
        Occupancy::Sparse(cells) => {
            // Per cell: u32 local index (varint would be smaller, but u32 is the
            // honest fixed-width on-disk record) + u16 palette index + 3 bytes coord.
            cells.len() * (4 + 2 + 3)
        }
        Occupancy::Dense {
            packed_indices,
            block_local_coords,
            ..
        } => {
            // The bit-packed cell array + 1 byte (bits_per_index) + 3 bytes/coord.
            packed_indices.len() + 1 + block_local_coords.len() * 3
        }
    }
}

#[cfg(test)]
mod tests;
