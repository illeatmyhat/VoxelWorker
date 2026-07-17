//! A cubic 3D bitset whose edge is at most 64, stored one `u64` per X-row.
//!
//! `BitCube` is a fixed-size occupancy cube of `edge³` bits with `edge <= 64`. Because
//! the edge never exceeds the width of a machine word, a whole X-row (the `edge` bits
//! `x = 0..edge` at a fixed `(y, z)`) fits in a single `u64`, so the cube is exactly
//! `edge²` row words — row index `z * edge + y`, bit `x = 1 << x`, x-fastest. This is the
//! textbook word-packed bitset (one bit per element, popcount by `count_ones`, run-set by a
//! double-shift mask) specialised to a cube small enough that one dimension collapses into a
//! single word.
//!
//! The point of the packing is density: one bit per cell is 8× smaller than one byte per
//! cell, and the row-word layout lets a contiguous run along X be set with ONE masked OR
//! rather than a per-bit loop. Bits at or above `edge` in each word are never set (the
//! writers stay in range), so `count_ones` summed over the words is an exact population
//! count and the words compare equal iff the cubes are equal.
//!
//! ## The run-set mask (overflow-safe at the full word)
//!
//! Setting the inclusive X-run `[min_x, max_x]` is `(u64::MAX << min_x) & (u64::MAX >> (63 -
//! max_x))`: the low shift clears bits below `min_x`, the high shift clears bits above
//! `max_x`. Both bounds are `< edge <= 64` hence `<= 63`, so neither shift distance reaches
//! 64 — this avoids the shift-overflow the naive `(1 << (max_x + 1)) - 1` form hits when
//! `max_x == 63`.
//!
//! Cite: Warren, *Hacker's Delight* (2nd ed. 2013) — bit-run masks, population count, and
//! the shift-overflow avoidance; Knuth, TAOCP vol. 4A §7.1 (bitwise techniques, packed
//! subsets). Deviation from a general word-packed bitset: the edge is bounded at 64 so a
//! whole X-row is one word (the run-set becomes a single masked OR), and the structure is a
//! cube rather than a flat set — the caller addresses it by `(x, y, z)`.

/// A cubic bitset of edge `1..=64`, one `u64` per X-row. `PartialEq`/`Eq` are exact: two
/// cubes are equal iff every row word matches (unused high bits are always clear).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BitCube {
    /// The cube edge in cells (`1..=64` so a whole X-row of `edge` bits fits one `u64`).
    edge: u32,
    /// `edge²` X-row words; row index `z * edge + y`, bit `x`. Bits at or above `edge` in
    /// every word are always clear, so `count_ones` over the words is an exact popcount.
    row_words: Vec<u64>,
}

impl BitCube {
    /// An all-clear cube of the given edge (`1..=64`).
    pub fn empty(edge: u32) -> Self {
        debug_assert!(
            (1..=64).contains(&edge),
            "a BitCube edge must be 1..=64 so an X-row fits one u64"
        );
        Self {
            edge,
            row_words: vec![0u64; (edge * edge) as usize],
        }
    }

    /// The cube edge in cells.
    pub fn edge(&self) -> u32 {
        self.edge
    }

    /// Set the contiguous X-run `min_x..=max_x` of the row at `(row_y, row_z)` (a masked
    /// OR). Overflow-safe for the full word: both bounds are `< edge <= 64` hence `<= 63`,
    /// so `63 - max_x` and the shifts never wrap even at `max_x == 63` (`u64::MAX >> 0`) or
    /// `min_x == 63`.
    pub fn set_x_run(&mut self, row_y: u32, row_z: u32, min_x: u32, max_x: u32) {
        debug_assert!(min_x <= max_x, "an X-run's min must not exceed its max");
        debug_assert!(max_x < self.edge, "an X-run must stay inside the cube edge");
        let low_bits_cleared = u64::MAX << min_x;
        let high_bits_cleared = u64::MAX >> (63 - max_x);
        let run_mask = low_bits_cleared & high_bits_cleared;
        let row = (row_z * self.edge + row_y) as usize;
        self.row_words[row] |= run_mask;
    }

    /// Whether the cell `(x, y, z)` is set.
    pub fn is_set(&self, x: u32, y: u32, z: u32) -> bool {
        let row = (z * self.edge + y) as usize;
        (self.row_words[row] >> x) & 1 == 1
    }

    /// The set-cell count (a popcount sum over the row words).
    pub fn popcount(&self) -> u32 {
        self.row_words.iter().map(|word| word.count_ones()).sum()
    }

    /// Expand one X-row (`z * edge + y`) into `out_row`, the `edge`-long destination: every
    /// set bit becomes `set_byte`, clear bits are left untouched (`out_row` starts as its
    /// caller left it). The single word→bytes expansion, so the skip-empty-word guard + the
    /// per-bit test live in ONE place; callers differ only in how they slice `out_row` out
    /// of a larger buffer. The `set_byte` is the injected "a set bit reads as THIS byte"
    /// value — the structure names no such byte itself.
    pub fn expand_row_into(&self, row_index: usize, out_row: &mut [u8], set_byte: u8) {
        expand_row_word_into(self.row_words[row_index], out_row, set_byte);
    }

    /// Expand the whole cube to `edge³` bytes (`0` for a clear bit, `set_byte` for a set
    /// bit; x-fastest, row index `z * edge + y`). O(edge³), never larger.
    pub fn expand_to_bytes(&self, set_byte: u8) -> Vec<u8> {
        let edge = self.edge as usize;
        let mut bytes = vec![0u8; edge * edge * edge];
        for z in 0..edge {
            for y in 0..edge {
                let row_index = z * edge + y;
                let out_start = row_index * edge;
                self.expand_row_into(row_index, &mut bytes[out_start..out_start + edge], set_byte);
            }
        }
        bytes
    }

    /// Pack `edge³` bytes into the cube (the inverse of [`expand_to_bytes`](Self::expand_to_bytes)):
    /// any nonzero byte is a set bit. Used to seed a cube from a byte buffer produced by an
    /// earlier expand (the round-trip the equality/popcount oracles pin).
    pub fn from_bytes(edge: u32, bytes: &[u8]) -> Self {
        let edge_usize = edge as usize;
        debug_assert_eq!(
            bytes.len(),
            edge_usize * edge_usize * edge_usize,
            "the byte buffer must be edge³"
        );
        let mut cube = Self::empty(edge);
        for z in 0..edge_usize {
            for y in 0..edge_usize {
                let base = (z * edge_usize + y) * edge_usize;
                let mut word = 0u64;
                for x in 0..edge_usize {
                    if bytes[base + x] != 0 {
                        word |= 1u64 << x;
                    }
                }
                cube.row_words[z * edge_usize + y] = word;
            }
        }
        cube
    }
}

/// Expand one bit-packed X-row `word` (x-fastest, one cell per bit) into `out_row`: every
/// set bit becomes `set_byte`, clear bits are untouched. An all-clear word writes nothing
/// (the common-case fast path).
fn expand_row_word_into(word: u64, out_row: &mut [u8], set_byte: u8) {
    if word == 0 {
        return;
    }
    for (x, out) in out_row.iter_mut().enumerate() {
        if (word >> x) & 1 == 1 {
            *out = set_byte;
        }
    }
}

/// Kani bounded-model-checking proofs of [`BitCube`]'s two silent-corruption-prone kernels —
/// the **overflow-safe run mask** and **row isolation** — verified over the whole bounded input
/// space rather than the handful of fixtures the unit test names. The run mask is exactly the
/// place a naive `(1 << (max + 1)) - 1` form wraps at `max == 63`; a wrong mask sets stray
/// occupancy bits that no differential render reliably samples. The doctrine (ADR 0014 decision
/// 6 / `docs/architecture/05-proof.md`) assigns Kani to these finite bit kernels; the density
/// bound `1..=64` doubles as the verification bound. `#[cfg(kani)]` keeps them inactive in
/// ordinary builds. Run under WSL: `cargo kani -p substrate`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// **The run-set mask sets EXACTLY the inclusive run `[min_x, max_x]`, and no bit outside
    /// it.** Proved at edge 64 — the full-word case, the ONLY edge whose run can reach bit 63,
    /// where the naive mask overflows; a smaller edge sets a strict sub-range of the same mask,
    /// so edge 64 subsumes them. Sweeps every `min_x <= max_x < 64` and every query `x`, so the
    /// overflow-safety and the `is_set` shift are both pinned over the whole input space.
    #[kani::proof]
    fn set_x_run_mask_sets_exactly_the_inclusive_run_at_the_full_word() {
        let mut cube = BitCube::empty(64);
        let min_x: u32 = kani::any();
        let max_x: u32 = kani::any();
        kani::assume(min_x <= max_x && max_x < 64);
        cube.set_x_run(0, 0, min_x, max_x);
        let x: u32 = kani::any();
        kani::assume(x < 64);
        assert!(cube.is_set(x, 0, 0) == (min_x <= x && x <= max_x));
    }

    /// **Row isolation.** Setting a run in one row `(row_y, row_z)` leaves every cell of every
    /// OTHER row clear — the addressing `row = z·edge + y` never spills a run into a neighbour.
    /// (A concrete edge 8 keeps the row-word vector small; the row-index arithmetic is the same
    /// at every edge.)
    #[kani::proof]
    fn set_x_run_does_not_touch_other_rows() {
        let mut cube = BitCube::empty(8);
        let (row_y, row_z): (u32, u32) = (kani::any(), kani::any());
        let (min_x, max_x): (u32, u32) = (kani::any(), kani::any());
        kani::assume(row_y < 8 && row_z < 8);
        kani::assume(min_x <= max_x && max_x < 8);
        cube.set_x_run(row_y, row_z, min_x, max_x);
        let (x, y, z): (u32, u32, u32) = (kani::any(), kani::any(), kani::any());
        kani::assume(x < 8 && y < 8 && z < 8);
        kani::assume(y != row_y || z != row_z); // a cell in some other row
        assert!(!cube.is_set(x, y, z));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The byte the fixtures treat as "set" — an arbitrary nonzero value the injection
    /// leaves to the caller (here a distinctive `0xFF`, to catch a stray `1`).
    const SET_BYTE: u8 = 255;

    /// Expand↔pack parity: for a spread of run fixtures a bit-packed [`BitCube`] expands to
    /// EXACTLY the bytes a naive per-cell dense fill produces, `is_set` agrees per cell,
    /// `from_bytes` is the inverse of `expand_to_bytes`, and `popcount` matches. The edge
    /// cases (edge 1, 32, 33 — spanning bit 32 — and 64, the full word) exercise the mask
    /// math a consumer that unpacks these bits depends on.
    #[test]
    fn expand_is_byte_identical_to_a_dense_fill() {
        type Run = ([u32; 3], [u32; 3]);
        let fixtures: &[(u32, &[Run])] = &[
            (1, &[([0, 0, 0], [0, 0, 0])]),                 // edge 1, single cell
            (4, &[([1, 1, 1], [1, 1, 1])]),                 // single interior cell
            (4, &[([0, 0, 0], [3, 3, 3])]),                 // full cube
            (8, &[([0, 3, 2], [7, 3, 2])]),                 // full X-row
            (8, &[([0, 0, 0], [7, 2, 0]), ([0, 3, 3], [2, 7, 7])]), // L-shaped two runs
            (16, &[([2, 5, 9], [13, 10, 12])]),             // arbitrary interior box
            (32, &[([0, 0, 0], [31, 31, 31])]),             // full edge-32 cube
            (33, &[([0, 0, 0], [32, 0, 0]), ([32, 32, 32], [32, 32, 32])]), // spans bit 32
            (64, &[([0, 0, 0], [63, 0, 0])]),               // full 64-bit X-row
            (64, &[([0, 0, 0], [63, 63, 63])]),             // full edge-64 cube
        ];
        for (edge, runs) in fixtures {
            let edge = *edge;
            let e = edge as usize;
            // Reference: naive per-cell dense byte fill.
            let mut reference = vec![0u8; e * e * e];
            for (min, max) in runs.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        for x in min[0]..=max[0] {
                            reference[(z as usize * e + y as usize) * e + x as usize] = SET_BYTE;
                        }
                    }
                }
            }
            // Bit path via `set_x_run`.
            let mut cube = BitCube::empty(edge);
            for (min, max) in runs.iter() {
                for z in min[2]..=max[2] {
                    for y in min[1]..=max[1] {
                        cube.set_x_run(y, z, min[0], max[0]);
                    }
                }
            }
            assert_eq!(cube.expand_to_bytes(SET_BYTE), reference, "edge {edge} expand mismatch");
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let expected = reference[(z as usize * e + y as usize) * e + x as usize] != 0;
                        assert_eq!(
                            cube.is_set(x, y, z),
                            expected,
                            "edge {edge} is_set mismatch at ({x},{y},{z})"
                        );
                    }
                }
            }
            assert_eq!(
                BitCube::from_bytes(edge, &reference),
                cube,
                "edge {edge} from_bytes is not the inverse of expand"
            );
            let set = reference.iter().filter(|byte| **byte != 0).count() as u32;
            assert_eq!(cube.popcount(), set, "edge {edge} popcount mismatch");
        }
    }

    /// `set_x_run`'s mask math stays overflow-safe at the full 64-bit word (the case that
    /// would wrap a naive `(1 << (max + 1)) - 1`): full-word, high-only, low-only, an
    /// interior run, and the edge-1 degenerate all set exactly the intended bits.
    #[test]
    fn set_x_run_masks_are_overflow_safe_at_full_word() {
        let mut full = BitCube::empty(64);
        full.set_x_run(0, 0, 0, 63);
        assert_eq!(full.row_words[0], u64::MAX);
        assert_eq!(full.popcount(), 64);

        let mut high = BitCube::empty(64);
        high.set_x_run(0, 0, 63, 63);
        assert_eq!(high.row_words[0], 1u64 << 63);

        let mut low = BitCube::empty(64);
        low.set_x_run(0, 0, 0, 0);
        assert_eq!(low.row_words[0], 1u64);

        let mut interior = BitCube::empty(64);
        interior.set_x_run(0, 0, 5, 40);
        let expected_interior = (u64::MAX >> (63 - 40)) & (u64::MAX << 5);
        assert_eq!(interior.row_words[0], expected_interior);
        assert_eq!(interior.popcount(), 40 - 5 + 1);

        let mut degenerate = BitCube::empty(1);
        degenerate.set_x_run(0, 0, 0, 0);
        assert!(degenerate.is_set(0, 0, 0));
        assert_eq!(degenerate.popcount(), 1);
    }
}
