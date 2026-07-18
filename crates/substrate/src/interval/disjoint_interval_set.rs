//! A sorted set of disjoint, non-touching integer intervals.
//!
//! [`DisjointIntervalSet`] maintains a set of half-open integer intervals `[lo, hi)`
//! kept **sorted, pairwise disjoint, and non-touching**: no two stored intervals
//! overlap OR abut (`a.hi == b.lo` is treated as adjacent and fused). This is the
//! canonical **interval set** (a.k.a. interval container / run-list) — the union of a
//! stream of intervals maintained in normalized form, so that the set is always minimal
//! and the widest contiguous run is simply `max(hi − lo)` over the stored intervals.
//!
//! [`insert`](DisjointIntervalSet::insert) unions one interval into the set, coalescing
//! with every interval it overlaps or abuts. It has an **O(1) ascending-append fast
//! path** for the dominant case where intervals arrive in increasing order (the run at
//! the end is simply pushed or extended), falling back to a linear splice-merge only
//! when an interval lands to the left of the last stored run.
//!
//! Cite: the union-of-intervals / "merge intervals" folklore (CLRS, *Introduction to
//! Algorithms*, interval material); interval-container libraries such as Boost ICL
//! generalize the same normalized-set idea. Deviation — kept deliberately simple: a
//! sorted `Vec` with in-place splice, NOT a balanced tree or skip list. The workload is
//! append-dominated (near-sorted arrivals hit the O(1) path and a fully-solid row stays
//! length one), so a tree's per-node overhead and allocation would cost more than the
//! rare linear splice it saves; abutment is fused because the intervals index a dense
//! lattice where `hi == lo` cells are physically contiguous.

/// A sorted set of disjoint, non-touching half-open integer intervals `[lo, hi)`.
///
/// Invariant after any [`insert`](Self::insert): the intervals are sorted ascending,
/// pairwise non-overlapping, and non-abutting (`intervals[i].1 < intervals[i+1].0`).
#[derive(Debug, Clone, Default)]
pub struct DisjointIntervalSet {
    /// The normalized intervals, each `(lo, hi)` with `lo < hi`, in ascending order.
    intervals: Vec<(i64, i64)>,
}

impl DisjointIntervalSet {
    /// An empty set.
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
        }
    }

    /// `true` when the set holds no intervals.
    pub fn is_empty(&self) -> bool {
        self.intervals.is_empty()
    }

    /// The stored intervals, in ascending order — each `(lo, hi)` disjoint and
    /// non-touching from its neighbours.
    pub fn intervals(&self) -> &[(i64, i64)] {
        &self.intervals
    }

    /// The width of the widest stored interval (`max(hi − lo)`), or `0` when empty.
    /// Because the set is kept minimal, this IS the widest contiguous run it covers.
    ///
    /// The width **saturates at `i64::MAX`**. `insert` accepts any `lo < hi`, so a caller may store
    /// an interval whose true width does not fit an `i64` — the span of `[i64::MIN, i64::MAX)` is
    /// `2^64 − 1` — and a plain `hi - lo` would overflow and panic there. Saturating keeps this
    /// total over every set the public API can build; in the voxel-coordinate workloads it serves,
    /// widths are bounded by the grid, so the saturating case is unreachable and the value exact.
    pub fn widest_span(&self) -> i64 {
        self.intervals
            .iter()
            .map(|&(lo, hi)| hi.saturating_sub(lo))
            .max()
            .unwrap_or(0)
    }

    /// Insert the half-open interval `[lo, hi)`, coalescing with every stored interval
    /// it overlaps OR abuts (`hi == other.lo` / `other.hi == lo` are adjacent and fuse
    /// into one run). Keeps the set sorted, disjoint, and non-touching. The dominant
    /// ascending-arrival case (intervals stream in increasing `lo`) hits the O(1) fast
    /// path; a solid row stays length one.
    pub fn insert(&mut self, lo: i64, hi: i64) {
        // Fast path: append after, or extend, the last interval. Intervals arriving in
        // ascending order coalesce here with no shifting.
        if let Some(&mut (last_lo, ref mut last_hi)) = self.intervals.last_mut() {
            if lo > *last_hi {
                self.intervals.push((lo, hi)); // strictly right of the last, with a gap
                return;
            }
            if lo >= last_lo {
                if hi > *last_hi {
                    *last_hi = hi; // overlaps / abuts the last, extends it right
                }
                return;
            }
        } else {
            self.intervals.push((lo, hi));
            return;
        }
        // General merge (rare: an out-of-order interval starting left of the last run).
        let mut start = 0;
        while start < self.intervals.len() && self.intervals[start].1 < lo {
            start += 1; // skip intervals strictly left of the run (a real gap)
        }
        let mut merged_lo = lo;
        let mut merged_hi = hi;
        let mut end = start;
        while end < self.intervals.len() && self.intervals[end].0 <= merged_hi {
            merged_lo = merged_lo.min(self.intervals[end].0);
            merged_hi = merged_hi.max(self.intervals[end].1);
            end += 1;
        }
        self.intervals
            .splice(start..end, std::iter::once((merged_lo, merged_hi)));
    }
}

// NOTE (verification): the `insert` normalization invariant (sorted ∧ disjoint ∧ non-touching
// after any insert) is NOT a Kani target. A `Vec::splice`-backed insert makes CBMC model the
// drain + reallocation machinery, which exploded to thousands of VCCs on a 3-interval set — a
// known bounded-model-checking pathology for heavy std-collection mutation, not a property
// failure. This is exactly why the extraction map (docs/design/substrate-extraction-map.md,
// decision-6) assigns this stateful invariant to a DEDUCTIVE prover (Creusot/Verus), not Kani;
// it waits for that tier to be stood up. The unit tests below pin the behaviour meanwhile.

/// Kani probes of the MACHINE-INTEGER edges of this type — the edges the Verus proof of `insert`
/// deliberately assumed away in its preconditions, and which the deductive tier therefore never
/// examined. (`verification/verus/disjoint_interval_set_insert.rs` proves the normalization
/// invariant; `verification/verus/widest_span.rs` proves the max-fold, but takes PRE-COMPUTED
/// non-negative widths, so the `hi - lo` subtraction below was never modelled at all.)
///
/// Note this does NOT contradict the NOTE above about `insert` not being a Kani target: these
/// harnesses insert into an EMPTY set, which takes the O(1) push fast path and never reaches the
/// `Vec::splice` that exploded BMC.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// `widest_span` computes `hi - lo` on `i64`. A maximally wide interval is legal input to
    /// `insert` (`lo < hi` is the only constraint), so this asks whether the subtraction can
    /// overflow on a set built entirely through the public API.
    #[kani::proof]
    fn widest_span_does_not_overflow_on_a_legal_interval() {
        let lo: i64 = kani::any();
        let hi: i64 = kani::any();
        kani::assume(lo < hi); // the type's own notion of a non-empty interval
        let mut set = DisjointIntervalSet::new();
        set.insert(lo, hi);
        let _ = set.widest_span();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascending_append_stays_disjoint_with_gaps() {
        let mut set = DisjointIntervalSet::new();
        set.insert(0, 2);
        set.insert(5, 7);
        set.insert(10, 12);
        assert_eq!(set.intervals(), &[(0, 2), (5, 7), (10, 12)]);
        assert_eq!(set.widest_span(), 2);
    }

    #[test]
    fn ascending_overlap_extends_the_last_run() {
        let mut set = DisjointIntervalSet::new();
        set.insert(0, 4);
        set.insert(2, 6); // overlaps the last ⇒ extends to [0, 6)
        assert_eq!(set.intervals(), &[(0, 6)]);
        // A contained interval leaves the run unchanged.
        set.insert(1, 3);
        assert_eq!(set.intervals(), &[(0, 6)]);
    }

    #[test]
    fn abutment_fuses_into_one_run() {
        let mut set = DisjointIntervalSet::new();
        set.insert(0, 3);
        set.insert(3, 6); // hi == lo abuts ⇒ single [0, 6)
        assert_eq!(set.intervals(), &[(0, 6)]);
        assert_eq!(set.widest_span(), 6);
    }

    #[test]
    fn mid_insert_bridges_two_runs() {
        let mut set = DisjointIntervalSet::new();
        set.insert(0, 2);
        set.insert(6, 8);
        // A run that spans the gap fuses BOTH neighbours into one interval.
        set.insert(2, 6);
        assert_eq!(set.intervals(), &[(0, 8)]);
        assert_eq!(set.widest_span(), 8);
    }

    #[test]
    fn out_of_order_left_insert_takes_the_splice_path() {
        let mut set = DisjointIntervalSet::new();
        set.insert(10, 12);
        set.insert(0, 2); // strictly left of the last run
        set.insert(4, 6); // between, disjoint
        assert_eq!(set.intervals(), &[(0, 2), (4, 6), (10, 12)]);
    }

    /// A maximally wide interval is legal input to `insert` (`lo < hi` is the only constraint), but
    /// its true width (`2^64 − 1`) does not fit an `i64`, so the plain `hi - lo` used to overflow
    /// and PANIC here. Found by the Kani harness
    /// `widest_span_does_not_overflow_on_a_legal_interval`; the width now saturates.
    #[test]
    fn widest_span_saturates_instead_of_overflowing() {
        let mut set = DisjointIntervalSet::new();
        set.insert(i64::MIN, i64::MAX);
        assert_eq!(set.widest_span(), i64::MAX, "saturates rather than panicking");

        // One step in from the boundary still overflows a plain subtraction, and still saturates.
        let mut nearly = DisjointIntervalSet::new();
        nearly.insert(i64::MIN + 1, i64::MAX);
        assert_eq!(nearly.widest_span(), i64::MAX);

        // Ordinary widths are unaffected — the value is exact wherever it fits.
        let mut ordinary = DisjointIntervalSet::new();
        ordinary.insert(-5, 7);
        assert_eq!(ordinary.widest_span(), 12);
    }

    #[test]
    fn widest_span_is_zero_when_empty() {
        let set = DisjointIntervalSet::new();
        assert!(set.is_empty());
        assert_eq!(set.widest_span(), 0);
    }
}
