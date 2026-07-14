//! A **sorted parallel-array associative map with fixed-width bitmask values**: a set of `u64`
//! keys kept sorted ascending, each carrying a fixed-width bitmask (a `[u32; MASK_WORDS]` word
//! array, one bit per cell-local position) and a caller-defined `u32` fallback scalar. Lookup is a
//! binary search on the key array; a bitmask bit is addressed by a cell-local *linear index* via
//! the standard word/bit split (`word = index / 32`, `bit = index % 32`).
//!
//! The three values of a key — its bitmask and its fallback scalar — live in three **parallel
//! arrays** ([`keys`](SortedKeyBitmaskMap::keys) ∥ [`masks`](SortedKeyBitmaskMap::masks) ∥
//! [`fallbacks`](SortedKeyBitmaskMap::fallbacks)), index-aligned, rather than an array of a single
//! record struct. This is the **structure-of-arrays** layout: a consumer that walks only the keys
//! (the binary search) touches one dense contiguous run, and a consumer that streams the map to a
//! packed GPU record — the reason this shape exists here — zips the three arrays back into whatever
//! interleaving its byte contract wants, at its own seam.
//!
//! Construction is from `(key, mask, fallback)` triples, **sorted by key**; the caller is
//! responsible for having already merged duplicates (bitmask union, fallback policy) before it
//! hands over triples, so the keys it supplies are unique — this kernel neither merges nor dedups,
//! it only sorts and splits into the parallel arrays. Bit accumulation *before* a map exists is
//! done directly on a bare `[u32; MASK_WORDS]` via the free [`set_mask_bit`] / [`mask_bit_is_set`]
//! helpers (a producer scatters bits into per-key masks, then emits the finished triples).
//!
//! ## Literature
//!
//! This is the classic **sorted table searched by binary search** (Knuth, *The Art of Computer
//! Programming* vol. 1 §2.2.6 on sequential-allocation tables, and vol. 3 §6.2.1 *Searching an
//! Ordered Table* — binary search over a sorted key array), stored in the **structure-of-arrays /
//! parallel-array** idiom (Knuth vol. 1 §2.2.2 on parallel arrays of linked data). The bitmask
//! value is a word-packed bitset addressed by the textbook word/bit split (Warren, *Hacker's
//! Delight* 2003, ch. 2 — the `>>5` / `&31` indexing of a bit within a 32-bit-word array). There is
//! **no single canonical name** for "a sorted-key map whose values are fixed-width bitmasks plus a
//! side scalar"; it is an association of these well-known pieces, so the type carries a descriptive
//! name rather than a borrowed one.
//!
//! ## Genericity
//!
//! `MASK_WORDS` (the bitmask's `u32`-word count, hence its bit capacity `MASK_WORDS * 32`) is a
//! const generic — a plain number, no domain meaning. Whether those bits index an 8³ cell, a 4²
//! tile, or anything else is the caller's cell geometry; this kernel only stores the words and
//! addresses a bit by its linear index. The key is an opaque sortable `u64`; the fallback is an
//! opaque `u32` payload. No key is a "block", no bit is a "voxel", no fallback is a "material" — the
//! domain names those at its own adapter seam.

/// Set the bit at cell-local `linear_index` in a fixed-width bitmask word array — the word/bit
/// split `mask[index / 32] |= 1 << (index % 32)` (Warren, *Hacker's Delight* ch. 2). Used to
/// scatter bits into a per-key mask *before* the triples are assembled into a
/// [`SortedKeyBitmaskMap`]. Panics (in debug) on an out-of-range index, exactly as an array index
/// would — the caller's cell geometry guarantees `linear_index < MASK_WORDS * 32`.
#[inline]
pub fn set_mask_bit<const MASK_WORDS: usize>(mask: &mut [u32; MASK_WORDS], linear_index: usize) {
    mask[linear_index / 32] |= 1u32 << (linear_index % 32);
}

/// Test the bit at cell-local `linear_index` in a fixed-width bitmask word array (the read
/// counterpart of [`set_mask_bit`]).
#[inline]
pub fn mask_bit_is_set<const MASK_WORDS: usize>(
    mask: &[u32; MASK_WORDS],
    linear_index: usize,
) -> bool {
    mask[linear_index / 32] & (1u32 << (linear_index % 32)) != 0
}

/// A sorted-key associative map with fixed-width [`MASK_WORDS`](Self::masks)-word bitmask values and
/// a per-key fallback scalar, in structure-of-arrays layout. [`keys`](Self::keys) is sorted strictly
/// ascending (the invariant the binary search relies on); [`masks`](Self::masks) and
/// [`fallbacks`](Self::fallbacks) are parallel to it (entry `i` of each describes key `i`). See the
/// module documentation for the layout rationale and the literature.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SortedKeyBitmaskMap<const MASK_WORDS: usize> {
    /// The keys, sorted strictly ascending — the order a binary search (and a GPU's split-key
    /// search) depends on.
    pub keys: Vec<u64>,
    /// One fixed-width bitmask per key, parallel to [`keys`](Self::keys). A bit is addressed by its
    /// cell-local linear index (see [`set_mask_bit`] / [`mask_bit_is_set`]).
    pub masks: Vec<[u32; MASK_WORDS]>,
    /// One caller-defined fallback scalar per key, parallel to [`keys`](Self::keys). The kernel
    /// stores it verbatim and assigns it no meaning.
    pub fallbacks: Vec<u32>,
}

impl<const MASK_WORDS: usize> SortedKeyBitmaskMap<MASK_WORDS> {
    /// The empty map (no keys). All three parallel arrays are empty.
    pub fn empty() -> Self {
        SortedKeyBitmaskMap::default()
    }

    /// Assemble a map from `(key, mask, fallback)` triples, **sorted by key** and split into the
    /// three parallel arrays. The keys MUST already be unique (the producer merges duplicate keys —
    /// mask union, fallback policy — before handing over triples); this constructor only sorts and
    /// splits, it does not merge or deduplicate. The sort is stable, so equal keys (a caller bug)
    /// keep their input order rather than the result being undefined.
    pub fn from_triples(mut triples: Vec<(u64, [u32; MASK_WORDS], u32)>) -> Self {
        triples.sort_by_key(|(key, _, _)| *key);
        let mut keys = Vec::with_capacity(triples.len());
        let mut masks = Vec::with_capacity(triples.len());
        let mut fallbacks = Vec::with_capacity(triples.len());
        for (key, mask, fallback) in triples {
            keys.push(key);
            masks.push(mask);
            fallbacks.push(fallback);
        }
        SortedKeyBitmaskMap {
            keys,
            masks,
            fallbacks,
        }
    }

    /// Assemble a map from `(key, mask, fallback)` triples the caller has **already** produced
    /// **sorted strictly ascending by key and deduplicated** — split into the parallel arrays
    /// with NO sort (unlike [`from_triples`](Self::from_triples)). The entry point for a producer
    /// that already accumulates in key order (e.g. drains an ordered-map / `BTreeMap`), so re-sorting
    /// would be redundant work. In debug builds the strict-ascending-unique precondition is checked
    /// with a [`debug_assert!`]; in release it is trusted, exactly as the binary search trusts the
    /// key order. Output is byte-identical to [`from_triples`](Self::from_triples) on the same
    /// already-sorted input.
    pub fn from_sorted_unique_triples(triples: Vec<(u64, [u32; MASK_WORDS], u32)>) -> Self {
        debug_assert!(
            triples.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "from_sorted_unique_triples requires keys strictly ascending and unique"
        );
        let mut keys = Vec::with_capacity(triples.len());
        let mut masks = Vec::with_capacity(triples.len());
        let mut fallbacks = Vec::with_capacity(triples.len());
        for (key, mask, fallback) in triples {
            keys.push(key);
            masks.push(mask);
            fallbacks.push(fallback);
        }
        SortedKeyBitmaskMap {
            keys,
            masks,
            fallbacks,
        }
    }

    /// The number of keys in the map (== the length of each parallel array).
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether the map holds no keys.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Binary-search the sorted key array for `key`, returning its parallel-array index, or `None`
    /// if absent. The result indexes both [`masks`](Self::masks) and [`fallbacks`](Self::fallbacks).
    pub fn find(&self, key: u64) -> Option<usize> {
        self.keys.binary_search(&key).ok()
    }

    /// Whether the bit at cell-local `linear_index` is set in the mask for `key` — a binary search
    /// for the key followed by a [`mask_bit_is_set`] on its mask. `false` if the key is absent.
    pub fn contains_bit(&self, key: u64, linear_index: usize) -> bool {
        match self.find(key) {
            Some(index) => mask_bit_is_set(&self.masks[index], linear_index),
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: usize = 16; // 512-bit masks — the domain's 8³-cell width, exercised as a plain number.

    /// `from_triples` produces a strictly-ascending key array and keeps the masks + fallbacks
    /// parallel to it, regardless of input order.
    #[test]
    fn construction_sorts_by_key_and_keeps_arrays_parallel() {
        let mut mask_a = [0u32; WORDS];
        set_mask_bit(&mut mask_a, 3);
        let mut mask_b = [0u32; WORDS];
        set_mask_bit(&mut mask_b, 100);
        let mut mask_c = [0u32; WORDS];
        set_mask_bit(&mut mask_c, 511);

        // Deliberately out of key order.
        let map = SortedKeyBitmaskMap::<WORDS>::from_triples(vec![
            (50u64, mask_b, 20u32),
            (10u64, mask_a, 10u32),
            (90u64, mask_c, 30u32),
        ]);

        assert_eq!(map.keys, vec![10, 50, 90]);
        assert!(map.keys.windows(2).all(|pair| pair[0] < pair[1]));
        // Each mask + fallback rode with its key through the sort.
        assert_eq!(map.masks[0], mask_a);
        assert_eq!(map.fallbacks[0], 10);
        assert_eq!(map.masks[1], mask_b);
        assert_eq!(map.fallbacks[1], 20);
        assert_eq!(map.masks[2], mask_c);
        assert_eq!(map.fallbacks[2], 30);
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    /// Binary search hits present keys (returning the right parallel index) and misses absent ones.
    #[test]
    fn binary_search_hit_and_miss() {
        let map = SortedKeyBitmaskMap::<WORDS>::from_triples(vec![
            (7u64, [0u32; WORDS], 1u32),
            (42u64, [0u32; WORDS], 2u32),
            (1000u64, [0u32; WORDS], 3u32),
        ]);
        assert_eq!(map.find(7), Some(0));
        assert_eq!(map.find(42), Some(1));
        assert_eq!(map.find(1000), Some(2));
        assert_eq!(map.find(0), None);
        assert_eq!(map.find(8), None);
        assert_eq!(map.find(u64::MAX), None);
        // The found index selects the matching fallback.
        assert_eq!(map.fallbacks[map.find(42).unwrap()], 2);
    }

    /// A bit set on a bare mask reads back through the map's `contains_bit`, and unset bits read
    /// false; an absent key reads false for every bit.
    #[test]
    fn bit_set_and_test_round_trip() {
        let mut mask = [0u32; WORDS];
        set_mask_bit(&mut mask, 5);
        set_mask_bit(&mut mask, 200);
        let map = SortedKeyBitmaskMap::<WORDS>::from_triples(vec![(77u64, mask, 9u32)]);

        assert!(map.contains_bit(77, 5));
        assert!(map.contains_bit(77, 200));
        assert!(!map.contains_bit(77, 6));
        assert!(!map.contains_bit(77, 199));
        // Absent key: no bit is set.
        assert!(!map.contains_bit(78, 5));
    }

    /// Word-boundary bits: the last bit of word 0 (31), the first bit of word 1 (32), and the very
    /// last bit of the fixed width (`MASK_WORDS * 32 - 1`) each land in exactly the expected word,
    /// with no bleed into a neighbour.
    #[test]
    fn word_boundary_bits_land_in_the_right_word() {
        let last_bit = WORDS * 32 - 1; // 511 for a 512-bit mask.
        for &index in &[0usize, 31, 32, 63, 64, last_bit] {
            let mut mask = [0u32; WORDS];
            set_mask_bit(&mut mask, index);
            // The bit reads back.
            assert!(mask_bit_is_set(&mask, index), "bit {index} must read set");
            // Exactly one bit is set across the whole mask (no bleed into a neighbouring word).
            let total: u32 = mask.iter().map(|word| word.count_ones()).sum();
            assert_eq!(total, 1, "setting bit {index} must set exactly one bit");
            // It lands in the arithmetic word / bit position.
            assert_eq!(mask[index / 32], 1u32 << (index % 32));
        }
    }

    /// The empty map: no keys, every lookup misses, and `from_triples(empty)` equals `empty()`.
    #[test]
    fn empty_map_finds_nothing() {
        let map = SortedKeyBitmaskMap::<WORDS>::empty();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.find(0), None);
        assert!(!map.contains_bit(0, 0));
        assert_eq!(map, SortedKeyBitmaskMap::<WORDS>::from_triples(vec![]));
    }
}
