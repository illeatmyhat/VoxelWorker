//! A cubic 3D grid of `Copy` payload values, one X-row at a time — the payload sibling of
//! [`BitCube`](crate::bit_cube::BitCube).
//!
//! `ValueCube<T>` is a fixed-size cube of `edge³` values with `edge <= 64`, stored X-row-major
//! in EXACTLY the [`BitCube`](crate::bit_cube::BitCube) row layout: row index `z * edge + y`,
//! element `x` inside the row, x-fastest. The two structures are therefore cell-for-cell
//! addressable by the same `(x, y, z)` arithmetic — a bit cube can gate a value cube of the
//! same edge (the bit says "this cell carries a value"; the value cube says which), and one
//! rasterizing walk can fill both without a second index derivation. This is the textbook
//! dense row-major array specialised to a cube whose edge matches the bitset's word bound.
//!
//! ## Why this is NOT [`CellGrid`](crate::greedy_cuboid_decomposition::CellGrid)
//!
//! `CellGrid<T>` is the *decomposition input*: an arbitrary-extent `[w, h, d]` grid of
//! `Option<T>` cells, where `None` IS the datum (an empty cell the greedy box decomposition
//! must not merge across). It shares the x-fastest linear order, and reusing it here would
//! have been tempting — but the two structures answer different questions:
//!
//! * **Occupancy is external here.** A value cube paired with a bit cube has no "empty" state
//!   of its own: the companion bitset already decides which cells carry a value, so the
//!   `Option` discriminant would encode a fact that is stored — authoritatively — elsewhere.
//!   Two sources for one predicate is exactly the desync a paired structure must avoid.
//! * **Storage.** `Option<u16>` is 4 bytes (niche-free: every `u16` is a valid payload), so a
//!   `CellGrid<u16>` doubles the memory of the same cube of raw `u16`s. For a per-cell payload
//!   tile that exists in the thousands, paying 2× for a redundant discriminant is the wrong
//!   trade.
//! * **Shape.** The edge is a cube bounded exactly as the bitset's is (`1..=64`), not a free
//!   `[w, h, d]`, so the row seam and the pairing are total rather than a runtime agreement.
//!
//! Cells the companion bitset marks empty hold whatever fill the constructor was given — a
//! **don't-care** value, never read. The structure names no such fill itself: the caller
//! injects it (exactly as `BitCube`'s row expansion injects its "set-bit" byte).
//!
//! Cite: Knuth, TAOCP vol. 1 §2.2.6 (sequential allocation of multidimensional arrays — the
//! row-major linear index); standard dense-array practice. Deviation from a general dense 3D
//! array: the extent is a cube bounded at 64 so the row layout coincides, index-for-index,
//! with the word-packed `BitCube` of the same edge.

/// A cubic grid of `edge³` values of `T`, X-row-major (row index `z * edge + y`, element `x`)
/// — the same row layout as [`BitCube`](crate::bit_cube::BitCube) of the same edge.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ValueCube<T: Copy> {
    /// The cube edge in cells (`1..=64`, matching the paired bitset's bound).
    edge: u32,
    /// `edge³` values, x-fastest: index `(z * edge + y) * edge + x`.
    values: Vec<T>,
}

impl<T: Copy> ValueCube<T> {
    /// A cube of the given edge (`1..=64`) with every cell set to `fill` — the "don't-care"
    /// value the caller injects for cells its companion occupancy marks empty.
    pub fn new_filled(edge: u32, fill: T) -> Self {
        debug_assert!(
            (1..=64).contains(&edge),
            "a ValueCube edge must be 1..=64 (it pairs with a BitCube of the same edge)"
        );
        let edge_usize = edge as usize;
        Self {
            edge,
            values: vec![fill; edge_usize * edge_usize * edge_usize],
        }
    }

    /// A cube seeded from `edge³` values already in the row layout (x-fastest, row index
    /// `z * edge + y`) — the inverse of [`as_slice`](Self::as_slice).
    pub fn from_values(edge: u32, values: Vec<T>) -> Self {
        let edge_usize = edge as usize;
        debug_assert!(
            (1..=64).contains(&edge),
            "a ValueCube edge must be 1..=64 (it pairs with a BitCube of the same edge)"
        );
        debug_assert_eq!(
            values.len(),
            edge_usize * edge_usize * edge_usize,
            "the value buffer must be edge³"
        );
        Self { edge, values }
    }

    /// The cube edge in cells.
    pub fn edge(&self) -> u32 {
        self.edge
    }

    /// The whole cube in row layout (x-fastest, row index `z * edge + y`) — `edge³` values.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }

    /// The flat index of `(x, y, z)` — the SAME `(row, element)` split
    /// [`BitCube`](crate::bit_cube::BitCube) uses (`row = z * edge + y`, element `x`).
    #[inline]
    fn flat_index(&self, x: u32, y: u32, z: u32) -> usize {
        let edge = self.edge as usize;
        (z as usize * edge + y as usize) * edge + x as usize
    }

    /// The value at `(x, y, z)`.
    pub fn get(&self, x: u32, y: u32, z: u32) -> T {
        self.values[self.flat_index(x, y, z)]
    }

    /// Write the value at `(x, y, z)`.
    pub fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let index = self.flat_index(x, y, z);
        self.values[index] = value;
    }

    /// Fill the contiguous X-run `min_x..=max_x` of the row at `(row_y, row_z)` with `value` —
    /// the payload twin of [`BitCube::set_x_run`](crate::bit_cube::BitCube::set_x_run) (which
    /// ORs the same run's mask), so one walk over a set of runs can fill an occupancy bitset
    /// and a value cube in lockstep.
    pub fn fill_x_run(&mut self, row_y: u32, row_z: u32, min_x: u32, max_x: u32, value: T) {
        debug_assert!(min_x <= max_x, "an X-run's min must not exceed its max");
        debug_assert!(max_x < self.edge, "an X-run must stay inside the cube edge");
        let row_start = (row_z * self.edge + row_y) as usize * self.edge as usize;
        let run = &mut self.values[row_start + min_x as usize..=row_start + max_x as usize];
        run.fill(value);
    }

    /// One X-row (`row_index = z * edge + y`) as a slice of `edge` values.
    pub fn row(&self, row_index: usize) -> &[T] {
        let edge = self.edge as usize;
        let start = row_index * edge;
        &self.values[start..start + edge]
    }

    /// Copy one X-row (`row_index = z * edge + y`) into `out_row`, the `edge`-long destination
    /// — the row seam mirroring
    /// [`BitCube::expand_row_into`](crate::bit_cube::BitCube::expand_row_into): a packer that
    /// scatters tiles into a larger destination cube differs only in how it slices `out_row`
    /// out of that cube, never in how a row is read.
    pub fn copy_row_into(&self, row_index: usize, out_row: &mut [T]) {
        debug_assert_eq!(
            out_row.len(),
            self.edge as usize,
            "the destination row must be `edge` long"
        );
        out_row.copy_from_slice(self.row(row_index));
    }
}

impl ValueCube<u16> {
    /// The whole cube as `2 · edge³` **little-endian** bytes in row layout — the byte string a
    /// 16-bit-per-texel consumer (a 16-bit image, a `u16` texture upload) reads, one value's low
    /// byte first. The LE choice is not free: it is the byte order every mainstream 16-bit texel
    /// format is defined in, and the same order
    /// [`CubeTilePacking::pack_u16_value_cubes`](crate::cube_packing::CubeTilePacking::pack_u16_value_cubes)
    /// scatters a set of these cubes in, so one cube's bytes and a packed cube-of-cubes'
    /// bytes agree value-for-value.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.values.len() * 2);
        for value in &self.values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bit_cube::BitCube;

    /// Round-trip: `set`/`get` per cell, `fill_x_run` over runs, and `as_slice` /
    /// `from_values` are inverses — checked against a naive dense reference at several edges
    /// including the bounds (1 and 64).
    #[test]
    fn set_get_and_run_fill_round_trip() {
        type Run = ([u32; 3], [u32; 3], u16);
        let fixtures: &[(u32, &[Run])] = &[
            (1, &[([0, 0, 0], [0, 0, 0], 7)]),        // edge 1 (lower bound)
            (4, &[([1, 1, 1], [2, 3, 1], 42)]),       // interior box
            (8, &[([0, 3, 2], [7, 3, 2], 5), ([2, 3, 2], [4, 3, 2], 9)]), // overlapping runs
            (16, &[([2, 5, 9], [13, 10, 12], 300)]),  // arbitrary interior box
            (64, &[([0, 0, 0], [63, 63, 63], 65535)]), // edge 64 (upper bound), full cube
        ];
        const AIR: u16 = 0;
        for (edge, runs) in fixtures {
            let edge = *edge;
            let edge_usize = edge as usize;
            // Reference: a naive per-cell dense fill, later runs overwriting earlier ones.
            let mut reference = vec![AIR; edge_usize.pow(3)];
            for (min, max, value) in runs.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        for x in min[0]..=max[0] {
                            reference[(z as usize * edge_usize + y as usize) * edge_usize
                                + x as usize] = *value;
                        }
                    }
                }
            }
            // The run-fill path.
            let mut cube = ValueCube::new_filled(edge, AIR);
            for (min, max, value) in runs.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        cube.fill_x_run(y, z, min[0], max[0], *value);
                    }
                }
            }
            assert_eq!(cube.as_slice(), reference.as_slice(), "edge {edge} run-fill mismatch");
            // Per-cell `get`, and the `set` path reproduces the same cube.
            let mut per_cell = ValueCube::new_filled(edge, AIR);
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let expected = reference
                            [(z as usize * edge_usize + y as usize) * edge_usize + x as usize];
                        assert_eq!(cube.get(x, y, z), expected, "edge {edge} get ({x},{y},{z})");
                        per_cell.set(x, y, z, expected);
                    }
                }
            }
            assert_eq!(per_cell, cube, "edge {edge}: per-cell `set` must rebuild the cube");
            // `from_values` is the inverse of `as_slice`.
            assert_eq!(
                ValueCube::from_values(edge, cube.as_slice().to_vec()),
                cube,
                "edge {edge}: from_values must invert as_slice"
            );
            assert_eq!(cube.edge(), edge);
        }
    }

    /// `to_le_bytes` emits each value low-byte-first, in the cube's row order — so byte pair
    /// `2i` of the output IS the value at flat index `i`, and the length is `2 · edge³`.
    #[test]
    fn to_le_bytes_is_row_order_low_byte_first() {
        let edge = 3u32;
        let values: Vec<u16> = (0..edge * edge * edge).map(|i| 0x0100 * i as u16 + 7).collect();
        let cube = ValueCube::from_values(edge, values.clone());
        let bytes = cube.to_le_bytes();
        assert_eq!(bytes.len(), 2 * (edge * edge * edge) as usize);
        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                [bytes[index * 2], bytes[index * 2 + 1]],
                value.to_le_bytes(),
                "value {index} must land low byte first"
            );
        }
    }

    /// The row layout IS [`BitCube`]'s: for the same runs, the rows a `ValueCube` reports
    /// (index `z * edge + y`) carry values at exactly the X positions the bit cube's row word
    /// has set — the index identity a paired occupancy/payload consumer depends on. Also pins
    /// the row seam (`copy_row_into` == `row`).
    #[test]
    fn row_layout_matches_bit_cube_indexing() {
        const AIR: u16 = 0;
        const SOLID: u16 = 0xBEEF;
        for edge in [1u32, 4, 33, 64] {
            let mut bits = BitCube::empty(edge);
            let mut values = ValueCube::new_filled(edge, AIR);
            // A run per row at a deterministic, edge-dependent span.
            for z in 0..edge {
                for y in 0..edge {
                    let min_x = (y + z) % edge;
                    let max_x = min_x + (edge - 1 - min_x) / 2;
                    bits.set_x_run(y, z, min_x, max_x);
                    values.fill_x_run(y, z, min_x, max_x, SOLID);
                }
            }
            for z in 0..edge {
                for y in 0..edge {
                    let row_index = (z * edge + y) as usize;
                    let row = values.row(row_index);
                    let mut copied = vec![AIR; edge as usize];
                    values.copy_row_into(row_index, &mut copied);
                    assert_eq!(copied.as_slice(), row, "edge {edge}: copy_row_into must equal row");
                    for x in 0..edge {
                        assert_eq!(
                            row[x as usize] == SOLID,
                            bits.is_set(x, y, z),
                            "edge {edge}: value row {row_index} disagrees with the bit cube at x={x}"
                        );
                        assert_eq!(values.get(x, y, z), row[x as usize]);
                    }
                }
            }
        }
    }
}
