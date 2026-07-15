//! A sortable space-filling key: a signed integer 3-vector packed into one `u64`.
//!
//! [`pack_lattice_key`] maps a signed integer lattice coordinate `[x, y, z]` to a
//! single `u64` whose **integer order is the lexicographic (z, y, x) order of the
//! coordinates**. Each axis is biased into the non-negative range and laid into a
//! [`BITS_PER_AXIS`]-bit lane, with **z in the high lane** so that comparing two packed
//! keys as `u64` compares first by z, then y, then x. That property is what lets a set
//! of these keys be kept **sorted and binary-searched** — including on a GPU, where the
//! key is split into a `(hi, lo)` pair of `u32` ([`split_key_hi_lo`]) because 64-bit
//! integers are unavailable there.
//!
//! This is a **space-filling linearization** of a 3D lattice, in the lineage of Morton
//! (Z-order) codes. Cite: Morton 1966 (space-filling linearization of a multidimensional
//! index); Samet, *Foundations of Multidimensional and Metric Data Structures* (2006).
//! **Deviation from a Morton code:** the axes are laid out **z-major lexicographically**
//! (whole-axis lanes, z highest), NOT bit-interleaved. Interleaving would give locality
//! in every axis at once but destroys the plain lexicographic order; the whole-lane
//! layout keeps integer order == lexicographic cell order, which is exactly what a
//! sorted, binary-searchable key set (on CPU and GPU) needs.

/// Bits per axis lane in the packed key: ±2^20 per axis, three 21-bit lanes filling
/// bits 0..63 (z in the highest lane), so the packed key's integer order IS
/// lexicographic (z, y, x) order.
pub const BITS_PER_AXIS: u32 = 21;

/// The per-axis bias added before packing so a signed coordinate lands in the
/// non-negative `[0, 2^BITS_PER_AXIS)` lane range: `2^(BITS_PER_AXIS - 1)`.
pub const BIAS: i64 = 1 << (BITS_PER_AXIS - 1);

/// Pack a signed integer 3-vector into a single sortable `u64` (z-major lexicographic
/// order). Panics if a coordinate falls outside the ±2^(BITS_PER_AXIS - 1) biased lane —
/// a silent wrap would alias two distinct coordinates onto one key.
pub fn pack_lattice_key(coordinate: [i64; 3]) -> u64 {
    let mut packed = 0u64;
    // z fills the highest lane so integer order == (z, y, x) lexicographic order.
    for (lane, &axis_value) in [coordinate[2], coordinate[1], coordinate[0]]
        .iter()
        .enumerate()
    {
        let biased = axis_value + BIAS;
        assert!(
            (0..(1i64 << BITS_PER_AXIS)).contains(&biased),
            "lattice coordinate {axis_value} exceeds the {BITS_PER_AXIS}-bit biased lane"
        );
        packed |= (biased as u64) << ((2 - lane) as u32 * BITS_PER_AXIS);
    }
    packed
}

/// Unpack a [`pack_lattice_key`] key back to its signed 3-vector.
pub fn unpack_lattice_key(key: u64) -> [i64; 3] {
    let lane_mask = (1u64 << BITS_PER_AXIS) - 1;
    let unpack_lane =
        |lane: u32| -> i64 { ((key >> (lane * BITS_PER_AXIS)) & lane_mask) as i64 - BIAS };
    [unpack_lane(0), unpack_lane(1), unpack_lane(2)]
}

/// Split a `u64` key into its `[hi, lo]` pair of `u32` halves — the form a GPU (with no
/// native `u64`) binary-searches. `hi` is the high 32 bits, `lo` the low 32; comparing
/// `(hi, lo)` lexicographically reproduces the `u64` order.
pub fn split_key_hi_lo(key: u64) -> [u32; 2] {
    [(key >> 32) as u32, key as u32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_key_round_trips_and_orders_z_major() {
        let coordinates = [
            [0i64, 0, 0],
            [-1, -2, -3],
            [17, -300, 4096],
            [-(1 << 19), (1 << 19), 0],
        ];
        for &coordinate in &coordinates {
            assert_eq!(unpack_lattice_key(pack_lattice_key(coordinate)), coordinate);
        }
        // Integer key order is (z, y, x) lexicographic — the sort a binary search relies on.
        assert!(pack_lattice_key([5, 0, 0]) < pack_lattice_key([0, 1, 0]));
        assert!(pack_lattice_key([0, 5, 0]) < pack_lattice_key([0, 0, 1]));
        assert!(pack_lattice_key([-1, 0, 0]) < pack_lattice_key([0, 0, 0]));
    }

    #[test]
    fn hi_lo_split_reproduces_u64_order() {
        let key = pack_lattice_key([17, -300, 4096]);
        let [hi, lo] = split_key_hi_lo(key);
        assert_eq!(((hi as u64) << 32) | lo as u64, key);
    }
}
