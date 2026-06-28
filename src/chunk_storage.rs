//! Lossless compressed storage for a single resolved chunk grid (issue #20 S6a).
//!
//! The out-of-core store (#20) needs a compact, serialisable on-disk form for a
//! resolved per-chunk [`VoxelGrid`] (the grids [`crate::chunk_cache`] resolves and
//! caches). A resolved chunk is almost always **mostly air** with only a handful of
//! distinct materials, so the dense `dimensions³ × (position + material)` form a
//! `VoxelGrid` carries in RAM is hugely wasteful as a storage shape. This module is
//! that storage shape — and **only** the data structure plus its lossless
//! round-trip proof. It is NOT yet wired into the live resolve/render path (that
//! store integration is a later S6 step); goldens are untouched.
//!
//! ## Why this is lossless despite dropping the f32 positions
//!
//! Every resolved voxel centre sits at an integer-plus-half position: the producer
//! emits `i as f32 + 0.5 − half` and every translation/rebase applied afterwards is
//! an **integer voxel** count (`world_offset × density`, and the floating origin in
//! whole voxels). So each stored `world_position[axis]` is exactly
//! `voxel_index[axis] as f32 + 0.5` for an integer `voxel_index` — a fact this
//! module relies on and the round-trip tests assert byte-for-byte. We store the
//! integer `voxel_index` (relative to the chunk's min corner, in i64 so a far-placed
//! chunk keeps full precision) and rebuild the f32 position as
//! `(min_corner + local) as f32 + 0.5`, reproducing the producer's own arithmetic.
//!
//! `block_local_coord` and `material_id` are stored directly (the former is the
//! producer's intra-block coordinate, which a rebase by a non-block-multiple origin
//! can decouple from the absolute index, so it is NOT reconstructed from position).
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

use crate::voxel::{Voxel, VoxelGrid};

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
    /// The absolute voxel-index min corner of the occupied bounding box. The f32
    /// position of a voxel at local box offset `o` is `(min_corner_voxels + o) as f32
    /// + centre_fraction`. `[0; 3]` for an empty chunk.
    pub min_corner_voxels: [i64; 3],
    /// The shared fractional part of every occupied voxel centre, per axis (`pos −
    /// floor(pos)`). Within one resolved grid this is uniform per axis (every voxel
    /// comes from the same `i + 0.5 − half` formula plus integer translations), so
    /// one value per axis reproduces every centre exactly: an even-dimensioned grid
    /// centres at `n + 0.5`, an odd-dimensioned one at `n + 0.0`. `[0.0; 3]` for an
    /// empty chunk.
    pub centre_fraction: [f32; 3],
    /// The occupied bounding box spans (per axis, in voxels). `[0; 3]` for an empty
    /// chunk. `local_linear_index` in the sparse encoding (and the dense cell count)
    /// is row-major over these spans.
    pub box_spans: [u32; 3],
    /// The distinct `material_id`s present, in first-seen scan order, no duplicates.
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

/// The integer voxel index of a voxel centre: `floor(position)`.
///
/// Resolved voxel centres are always a half-integer (`n` for an odd-dimensioned
/// axis, `n + 0.5` for an even one) plus integer translations, so `floor` recovers a
/// unique integer index per cell; the sub-integer remainder is the shared
/// `centre_fraction`. Computed in f64 so a far-placed chunk's large magnitude does
/// not lose the integer part.
fn voxel_index_axis(position: f32) -> i64 {
    (position as f64).floor() as i64
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

    // The shared per-axis fractional offset of every voxel centre, taken from the
    // first occupied voxel. Within one resolved grid this is uniform per axis (same
    // `i + 0.5 − half` formula + integer translations) — debug-asserted below.
    let first = &grid.occupied[0];
    let centre_fraction = [
        (first.world_position[0] as f64 - (first.world_position[0] as f64).floor()) as f32,
        (first.world_position[1] as f64 - (first.world_position[1] as f64).floor()) as f32,
        (first.world_position[2] as f64 - (first.world_position[2] as f64).floor()) as f32,
    ];
    debug_assert!(
        grid.occupied.iter().all(|voxel| (0..3).all(|axis| {
            let frac = (voxel.world_position[axis] as f64
                - (voxel.world_position[axis] as f64).floor()) as f32;
            frac == centre_fraction[axis]
        })),
        "every voxel centre in a resolved grid must share the same per-axis fraction"
    );

    // 1) Occupied bounding box in absolute voxel-index space.
    let mut min_corner = [i64::MAX; 3];
    let mut max_corner = [i64::MIN; 3];
    for voxel in &grid.occupied {
        for axis in 0..3 {
            let index = voxel_index_axis(voxel.world_position[axis]);
            min_corner[axis] = min_corner[axis].min(index);
            max_corner[axis] = max_corner[axis].max(index);
        }
    }
    let box_spans = [
        (max_corner[0] - min_corner[0] + 1) as u32,
        (max_corner[1] - min_corner[1] + 1) as u32,
        (max_corner[2] - min_corner[2] + 1) as u32,
    ];

    // 2) Material palette: distinct ids, first-seen order, de-duplicated.
    let mut material_palette: Vec<u16> = Vec::new();
    for voxel in &grid.occupied {
        if !material_palette.contains(&voxel.material_id) {
            material_palette.push(voxel.material_id);
        }
    }
    let palette_index_of = |material_id: u16| -> u32 {
        material_palette
            .iter()
            .position(|&id| id == material_id)
            .expect("every material was inserted into the palette above") as u32
    };

    // 3) Sparse occupancy (always built — it is the default and the size baseline).
    let span_xy = box_spans[0] as u64 * box_spans[1] as u64;
    let local_linear_index = |voxel: &Voxel| -> u64 {
        let local = [
            (voxel_index_axis(voxel.world_position[0]) - min_corner[0]) as u64,
            (voxel_index_axis(voxel.world_position[1]) - min_corner[1]) as u64,
            (voxel_index_axis(voxel.world_position[2]) - min_corner[2]) as u64,
        ];
        local[2] * span_xy + local[1] * box_spans[0] as u64 + local[0]
    };
    let sparse_cells: Vec<SparseCell> = grid
        .occupied
        .iter()
        .map(|voxel| SparseCell {
            local_linear_index: local_linear_index(voxel),
            palette_index: palette_index_of(voxel.material_id),
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
        box_spans,
        min_corner,
        span_xy,
        &material_palette,
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
#[allow(clippy::too_many_arguments)]
fn build_dense_if_smaller(
    grid: &VoxelGrid,
    cell_count: u64,
    box_spans: [u32; 3],
    min_corner: [i64; 3],
    span_xy: u64,
    material_palette: &[u16],
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
    let value_count = material_palette.len() as u64 + 1;
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
        .map(|voxel| {
            let local = [
                (voxel_index_axis(voxel.world_position[0]) - min_corner[0]) as u64,
                (voxel_index_axis(voxel.world_position[1]) - min_corner[1]) as u64,
                (voxel_index_axis(voxel.world_position[2]) - min_corner[2]) as u64,
            ];
            let linear = local[2] * span_xy + local[1] * box_spans[0] as u64 + local[0];
            (linear, voxel)
        })
        .collect();
    indexed.sort_by_key(|(linear, _)| *linear);
    for (linear, voxel) in &indexed {
        let palette_index = material_palette
            .iter()
            .position(|&id| id == voxel.material_id)
            .expect("material in palette") as u32;
        cell_values[*linear as usize] = palette_index + 1;
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

    // Rebuild the f32 position of a cell at row-major local index `linear`, restoring
    // the per-axis fractional offset captured at compress time.
    let centre_fraction = compressed.centre_fraction;
    let position_of = |linear: u64| -> [f32; 3] {
        let local_x = if span_x == 0 { 0 } else { linear % span_x as u64 };
        let local_y = if span_y == 0 {
            0
        } else {
            (linear / span_x as u64) % span_y as u64
        };
        let local_z = if span_xy == 0 { 0 } else { linear / span_xy };
        [
            (min_corner[0] + local_x as i64) as f32 + centre_fraction[0],
            (min_corner[1] + local_y as i64) as f32 + centre_fraction[1],
            (min_corner[2] + local_z as i64) as f32 + centre_fraction[2],
        ]
    };

    match &compressed.occupancy {
        Occupancy::Sparse(cells) => {
            grid.occupied.reserve(cells.len());
            for cell in cells {
                grid.occupied.push(Voxel {
                    world_position: position_of(cell.local_linear_index),
                    block_local_coord: cell.block_local_coord,
                    material_id: compressed.material_palette[cell.palette_index as usize],
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
                    world_position: position_of(linear),
                    block_local_coord: block_local_coords[next_coord],
                    material_id: compressed.material_palette[palette_index],
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

/// The binary on-disk byte size of a whole [`CompressedChunk`] (header + palette +
/// occupancy), used by the ratio report.
#[cfg(test)]
fn compressed_binary_size(compressed: &CompressedChunk) -> usize {
    // Header: dimensions 3×u32, min_corner 3×i64, centre_fraction 3×f32, box_spans
    // 3×u32 = 12 + 24 + 12 + 12 = 60 bytes; palette: 2 bytes/entry.
    60 + compressed.material_palette.len() * 2 + occupancy_binary_size(&compressed.occupancy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::scene::{DefId, Node, NodeContent, Part, Scene};
    use crate::voxel::{GeometryParams, SdfShape, ShapeKind, Voxel, VoxelGrid, VoxelProducer};

    /// A pseudo-random generator (the same Numerical-Recipes LCG `cuboid.rs` uses),
    /// so the fuzz tests are deterministic without pulling in a `rand` dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    /// Canonicalise a grid's occupied set into a sorted multiset of
    /// `(bit_exact_position, block_local_coord, material_id)`. Keying on the raw f32
    /// bits (`to_bits`) makes the round-trip assertion **byte-for-byte** on every
    /// position (a sub-ULP shift fails), and includes `block_local_coord` so the
    /// intra-block coordinate is part of the losslessness guarantee. Order-independent
    /// (the resolve path treats the occupied vec as a set).
    fn occupied_multiset(
        grid: &VoxelGrid,
    ) -> std::collections::BTreeMap<([u32; 3], [u8; 3], u16), usize> {
        let mut multiset = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position_bits = [
                voxel.world_position[0].to_bits(),
                voxel.world_position[1].to_bits(),
                voxel.world_position[2].to_bits(),
            ];
            *multiset
                .entry((position_bits, voxel.block_local_coord, voxel.material_id))
                .or_insert(0) += 1;
        }
        multiset
    }

    /// Assert `decompress(compress(grid))` equals `grid` in dimensions and occupied
    /// set (position + block-local coord + material, byte-exact). Returns the
    /// `CompressedChunk` so callers can make follow-up assertions (palette, ratio).
    fn assert_lossless_round_trip(grid: &VoxelGrid, label: &str) -> CompressedChunk {
        let compressed = compress(grid);
        let restored = decompress(&compressed);
        assert_eq!(
            restored.dimensions, grid.dimensions,
            "[{label}] dimensions must round-trip"
        );
        assert_eq!(
            restored.occupied_count(),
            grid.occupied_count(),
            "[{label}] occupied count must round-trip"
        );
        assert_eq!(
            occupied_multiset(&restored),
            occupied_multiset(grid),
            "[{label}] occupied set (position + block-local + material) must be byte-identical"
        );
        // The compressed view's own occupied_count must agree with the grid's.
        assert_eq!(
            compressed.occupied_count(),
            grid.occupied_count(),
            "[{label}] CompressedChunk::occupied_count must match the source grid"
        );
        compressed
    }

    fn shape_grid(kind: ShapeKind, size: [u32; 3], voxels_per_block: u32) -> VoxelGrid {
        let shape = SdfShape {
            kind,
            size_blocks: size,
            wall_blocks: 1,
        };
        let mut grid = VoxelGrid::new(shape.grid_dimensions(voxels_per_block));
        shape.resolve(&mut grid, voxels_per_block);
        grid
    }

    #[test]
    fn round_trip_empty_chunk() {
        let grid = VoxelGrid::new([64, 64, 64]);
        let compressed = assert_lossless_round_trip(&grid, "empty");
        assert!(
            compressed.material_palette.is_empty(),
            "an empty chunk has an empty palette"
        );
        assert_eq!(compressed.occupied_count(), 0);
    }

    #[test]
    fn round_trip_full_single_material_chunk() {
        // A fully-occupied 8×8×8 box, single material — the dense win case.
        let dimensions = [8u32, 8, 8];
        let mut grid = VoxelGrid::new(dimensions);
        let half = [4.0f32; 3];
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    grid.occupied.push(Voxel {
                        world_position: [
                            x as f32 + 0.5 - half[0],
                            y as f32 + 0.5 - half[1],
                            z as f32 + 0.5 - half[2],
                        ],
                        block_local_coord: [(x % 4) as u8, (y % 4) as u8, (z % 4) as u8],
                        material_id: 7,
                    });
                }
            }
        }
        let compressed = assert_lossless_round_trip(&grid, "full-single-material");
        assert_eq!(
            compressed.material_palette,
            vec![7],
            "a single-material full chunk has a one-entry palette"
        );
        // A solid single-material box should pick the dense encoding (smaller).
        assert!(
            matches!(compressed.occupancy, Occupancy::Dense { .. }),
            "a fully-occupied single-material chunk should compress dense, got sparse"
        );
    }

    #[test]
    fn round_trip_multi_material_chunk() {
        // 4×4×2 quartered into four materials — distinct ids, no duplicates.
        let dimensions = [4u32, 4, 2];
        let mut grid = VoxelGrid::new(dimensions);
        let half = [2.0f32, 2.0, 1.0];
        for z in 0..2 {
            for y in 0..4 {
                for x in 0..4 {
                    let material = match (x < 2, y < 2) {
                        (true, true) => 11,
                        (false, true) => 22,
                        (true, false) => 33,
                        (false, false) => 44,
                    };
                    grid.occupied.push(Voxel {
                        world_position: [
                            x as f32 + 0.5 - half[0],
                            y as f32 + 0.5 - half[1],
                            z as f32 + 0.5 - half[2],
                        ],
                        block_local_coord: [x as u8, y as u8, z as u8],
                        material_id: material,
                    });
                }
            }
        }
        let compressed = assert_lossless_round_trip(&grid, "multi-material");
        let mut palette_sorted = compressed.material_palette.clone();
        palette_sorted.sort_unstable();
        assert_eq!(
            palette_sorted,
            vec![11, 22, 33, 44],
            "the palette must be exactly the distinct materials, no duplicates"
        );
        // No duplicate ids in the palette.
        let unique: std::collections::HashSet<u16> =
            compressed.material_palette.iter().copied().collect();
        assert_eq!(
            unique.len(),
            compressed.material_palette.len(),
            "palette must contain no duplicate materials"
        );
    }

    #[test]
    fn round_trip_real_resolved_chunks_across_shapes() {
        // Real resolved chunks via Scene::resolve_chunk across every SDF primitive.
        let voxels_per_block = 16u32;
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = Scene::from_geometry(
                GeometryParams {
                    shape: kind,
                    size_blocks: [5, 5, 5],
                    voxels_per_block,
                    wall_blocks: 1,
                },
                MaterialChoice::Stone,
            );
            let (min_chunk, max_chunk) = scene
                .covering_chunk_range(voxels_per_block)
                .expect("a placed shape has a covering chunk range");
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk = scene.resolve_chunk(
                            [chunk_x, chunk_y, chunk_z],
                            voxels_per_block,
                            0,
                        );
                        assert_lossless_round_trip(
                            &chunk,
                            &format!("{kind:?} chunk {chunk_x},{chunk_y},{chunk_z}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn round_trip_demo_scene_and_village_chunks() {
        let voxels_per_block = 16u32;

        // --demo-scene: three differently-materialled tools.
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let demo_scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);

        // --demo-village: an instanced house assembly.
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: size,
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut village = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        village.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );

        for (scene, label) in [(demo_scene, "demo-scene"), (village, "demo-village")] {
            let (min_chunk, max_chunk) = scene
                .covering_chunk_range(voxels_per_block)
                .expect("a placed scene has a covering chunk range");
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk = scene.resolve_chunk(
                            [chunk_x, chunk_y, chunk_z],
                            voxels_per_block,
                            0,
                        );
                        assert_lossless_round_trip(
                            &chunk,
                            &format!("{label} chunk {chunk_x},{chunk_y},{chunk_z}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn round_trip_part_only_debug_clouds_grid() {
        // A Part producer (debug clouds) fills a grid with material_id 0 voxels at a
        // pseudo-random fill — a different occupancy profile than the SDF shells.
        let scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(Part::DebugClouds { seed: 1 }),
        ));
        // Resolve over an explicit region (a Part-only scene has no chunk range).
        let grid = scene.resolve_region(
            crate::scene::RegionBlocks::new([4, 4, 4]),
            16,
            0,
        );
        if grid.occupied.is_empty() {
            return; // nothing to assert if the field produced no voxels.
        }
        assert_lossless_round_trip(&grid, "debug-clouds");
    }

    #[test]
    fn round_trip_randomized_fuzz_varied_fill_and_materials() {
        // Pseudo-random multi-material fills over varied extents / fill % / material
        // counts — the real safety net for both encodings (the heuristic flips
        // between sparse and dense across this matrix).
        let mut lcg = Lcg(0xc0ff_ee00_d15e_a5e5);
        let extents = [[1u32, 1, 1], [6, 4, 5], [9, 2, 7], [3, 8, 4], [7, 7, 7]];
        for &extent in &extents {
            for materials in [1u32, 2, 5] {
                for fill_percent in [5u32, 30, 75, 100] {
                    let half = [
                        extent[0] as f32 / 2.0,
                        extent[1] as f32 / 2.0,
                        extent[2] as f32 / 2.0,
                    ];
                    let mut grid = VoxelGrid::new(extent);
                    for z in 0..extent[2] {
                        for y in 0..extent[1] {
                            for x in 0..extent[0] {
                                if (lcg.next_u32() % 100) < fill_percent {
                                    let material = (lcg.next_u32() % materials) as u16;
                                    grid.occupied.push(Voxel {
                                        world_position: [
                                            x as f32 + 0.5 - half[0],
                                            y as f32 + 0.5 - half[1],
                                            z as f32 + 0.5 - half[2],
                                        ],
                                        block_local_coord: [
                                            (x % 4) as u8,
                                            (y % 4) as u8,
                                            (z % 4) as u8,
                                        ],
                                        material_id: material,
                                    });
                                }
                            }
                        }
                    }
                    assert_lossless_round_trip(
                        &grid,
                        &format!("fuzz {extent:?} m={materials} fill={fill_percent}"),
                    );
                }
            }
        }
    }

    #[test]
    fn palette_has_no_duplicates_and_covers_every_material() {
        // A grid whose materials repeat heavily across cells; the palette must still
        // be the DISTINCT set with no duplicates, and every cell's material must map
        // back through it.
        let dimensions = [6u32, 6, 1];
        let mut grid = VoxelGrid::new(dimensions);
        // Voxel centres sit at integer-plus-half (a resolved-grid invariant), so use
        // `n + 0.5` directly — NOT `n + 0.5 - half`, which would land a centre on an
        // integer (e.g. 0.0) that is not a valid voxel centre.
        let materials = [100u16, 200, 100, 300, 200, 100];
        for y in 0..6 {
            for x in 0..6 {
                grid.occupied.push(Voxel {
                    world_position: [x as f32 + 0.5, y as f32 + 0.5, 0.5],
                    block_local_coord: [0, 0, 0],
                    material_id: materials[x as usize],
                });
            }
        }
        let compressed = compress(&grid);
        let unique: std::collections::HashSet<u16> =
            compressed.material_palette.iter().copied().collect();
        assert_eq!(unique.len(), compressed.material_palette.len(), "no dup palette entries");
        assert_eq!(unique, [100, 200, 300].into_iter().collect(), "distinct materials only");
        assert_eq!(occupied_multiset(&decompress(&compressed)), occupied_multiset(&grid));
    }

    #[test]
    fn serde_round_trip_through_json_equals_original_grid() {
        // serialize → deserialize → decompress equals the original grid, proving the
        // CompressedChunk is serde-serialisable for the later disk store.
        let grid = shape_grid(ShapeKind::Sphere, [4, 4, 4], 8);
        assert!(!grid.occupied.is_empty(), "the sphere must resolve to voxels");
        let compressed = compress(&grid);

        let json = serde_json::to_string(&compressed).expect("CompressedChunk serialises");
        let restored: CompressedChunk =
            serde_json::from_str(&json).expect("CompressedChunk deserialises");
        assert_eq!(restored, compressed, "serde must round-trip the CompressedChunk exactly");

        let restored_grid = decompress(&restored);
        assert_eq!(
            occupied_multiset(&restored_grid),
            occupied_multiset(&grid),
            "serialize → deserialize → decompress must equal the original grid"
        );
    }

    /// Report measured compression ratios on representative real resolved chunks
    /// (sphere / torus / village). Asserts a meaningful win on the mostly-empty SDF
    /// shells and prints the numbers (run with `--nocapture` to read them).
    #[test]
    fn report_compression_ratios_on_real_chunks() {
        let voxels_per_block = 16u32;

        // Raw size of a VoxelGrid's occupied storage: each Voxel is
        // 3×f32 + 3×u8 + u16 = 12 + 3 + 2 = 17 bytes of payload (the Vec capacity is
        // ignored; this is the logical raw footprint of the occupied data).
        let raw_bytes = |grid: &VoxelGrid| -> usize { grid.occupied_count() * 17 };

        let report = |label: &str, grid: &VoxelGrid| {
            let compressed = compress(grid);
            // Compressed size via the same binary measure the heuristic uses.
            let compressed_bytes = compressed_binary_size(&compressed);
            let raw = raw_bytes(grid);
            let ratio = if compressed_bytes == 0 {
                0.0
            } else {
                raw as f64 / compressed_bytes as f64
            };
            let encoding = match compressed.occupancy {
                Occupancy::Sparse(_) => "sparse",
                Occupancy::Dense { .. } => "dense",
            };
            println!(
                "[ratio] {label}: {} voxels, raw {raw} B, compressed {compressed_bytes} B \
                 ({encoding}) → {ratio:.2}× smaller",
                grid.occupied_count()
            );
            ratio
        };

        // A sphere chunk (mostly-empty shell — the sparse win case).
        let sphere = shape_grid(ShapeKind::Sphere, [5, 5, 5], voxels_per_block);
        let sphere_ratio = report("sphere 5³@16 (whole grid)", &sphere);

        // A torus chunk.
        let torus = shape_grid(ShapeKind::Torus, [5, 5, 5], voxels_per_block);
        report("torus 5³@16 (whole grid)", &torus);

        // A solid box (the dense win case).
        let solid_box = shape_grid(ShapeKind::Box, [4, 4, 4], voxels_per_block);
        report("box 4³@16 (whole grid, solid)", &solid_box);

        // Real PER-CHUNK resolved pieces of a sphere: a solid SDF sphere is a filled
        // ellipsoid, so its chunks are dense-favourable (the dense bit-packed encoding
        // wins) — the honest figure for the common SDF-solid case. Report the
        // aggregate ratio and the single best chunk.
        let per_chunk_report = |label: &str, kind: ShapeKind, size: [u32; 3]| -> f64 {
            let scene = Scene::from_geometry(
                GeometryParams {
                    shape: kind,
                    size_blocks: size,
                    voxels_per_block,
                    wall_blocks: 1,
                },
                MaterialChoice::Stone,
            );
            let (lo, hi) = scene.covering_chunk_range(voxels_per_block).expect("placed");
            let mut total_raw = 0usize;
            let mut total_compressed = 0usize;
            let mut best_ratio = 0.0f64;
            let mut sparse_chunks = 0usize;
            let mut total_chunks = 0usize;
            for cz in lo[2]..=hi[2] {
                for cy in lo[1]..=hi[1] {
                    for cx in lo[0]..=hi[0] {
                        let chunk = scene.resolve_chunk([cx, cy, cz], voxels_per_block, 0);
                        if chunk.occupied.is_empty() {
                            continue;
                        }
                        total_chunks += 1;
                        let compressed = compress(&chunk);
                        if matches!(compressed.occupancy, Occupancy::Sparse(_)) {
                            sparse_chunks += 1;
                        }
                        let raw = raw_bytes(&chunk);
                        let comp = compressed_binary_size(&compressed);
                        total_raw += raw;
                        total_compressed += comp;
                        best_ratio = best_ratio.max(raw as f64 / comp.max(1) as f64);
                    }
                }
            }
            let ratio = total_raw as f64 / total_compressed.max(1) as f64;
            println!(
                "[ratio] {label}: {total_chunks} non-empty chunks ({sparse_chunks} sparse), \
                 raw {total_raw} B, compressed {total_compressed} B → {ratio:.2}× aggregate, \
                 best chunk {best_ratio:.2}×"
            );
            best_ratio
        };
        let sphere_best = per_chunk_report("sphere 5³@16 (per chunk)", ShapeKind::Sphere, [5, 5, 5]);
        per_chunk_report("torus 5³@16 (per chunk)", ShapeKind::Torus, [5, 5, 5]);

        // A genuinely sparse case (the sparse-encoding win): a very-low-fill grid.
        // The sparse vs dense crossover is ~`N < cells/48` (sparse 9 B/voxel vs dense
        // ~1 bit/cell), so a sub-1% fill over a big box lands firmly in sparse-land.
        let mut lcg = Lcg(0x5ade_5e00_1234_abcd_u64);
        let sparse_extent = [40u32, 40, 40];
        let sparse_half = [20.0f32, 20.0, 20.0];
        let mut sparse_grid = VoxelGrid::new(sparse_extent);
        for z in 0..40 {
            for y in 0..40 {
                for x in 0..40 {
                    // ~0.5% fill → ~320 voxels over 64000 cells, well under cells/48.
                    if lcg.next_u32() % 1000 < 5 {
                        sparse_grid.occupied.push(Voxel {
                            world_position: [
                                x as f32 + 0.5 - sparse_half[0],
                                y as f32 + 0.5 - sparse_half[1],
                                z as f32 + 0.5 - sparse_half[2],
                            ],
                            block_local_coord: [0, 0, 0],
                            material_id: 1,
                        });
                    }
                }
            }
        }
        let sparse_compressed = compress(&sparse_grid);
        assert!(
            matches!(sparse_compressed.occupancy, Occupancy::Sparse(_)),
            "a 3%-fill grid must pick the sparse encoding"
        );
        let sparse_ratio = report("random 0.5%-fill 40³ (sparse case)", &sparse_grid);

        // A real village chunk (resolved through the chunk path).
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape {
                kind,
                size_blocks: [5, 5, 5],
                wall_blocks: 1,
            };
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = crate::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let village = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        ]);
        let (min_chunk, max_chunk) =
            village.covering_chunk_range(voxels_per_block).expect("placed");
        let mut total_raw = 0usize;
        let mut total_compressed = 0usize;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk =
                        village.resolve_chunk([chunk_x, chunk_y, chunk_z], voxels_per_block, 0);
                    if chunk.occupied.is_empty() {
                        continue;
                    }
                    total_raw += raw_bytes(&chunk);
                    total_compressed += compressed_binary_size(&compress(&chunk));
                }
            }
        }
        let village_ratio = total_raw as f64 / total_compressed.max(1) as f64;
        println!(
            "[ratio] village (all non-empty chunks): raw {total_raw} B, compressed \
             {total_compressed} B → {village_ratio:.2}× smaller"
        );

        // The solid SDF shapes net a strong dense win (~5×); the genuinely sparse
        // grid nets an even bigger sparse win; the village chunks net a win.
        assert!(
            sphere_ratio > 3.0,
            "a solid sphere should compress strongly via dense (>3×), got {sphere_ratio:.2}×"
        );
        assert!(
            sphere_best > 3.0,
            "the best per-chunk sphere piece should compress strongly (>3×), got {sphere_best:.2}×"
        );
        assert!(
            sparse_ratio > 1.5,
            "a 3%-fill grid should compress via the sparse encoding (>1.5×), got {sparse_ratio:.2}×"
        );
        assert!(
            village_ratio > 1.0,
            "the village chunks should net a compression win, got {village_ratio:.2}×"
        );
    }
}
