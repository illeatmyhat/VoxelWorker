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
    pub fn widest_span(&self) -> i64 {
        self.intervals
            .iter()
            .map(|&(lo, hi)| hi - lo)
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

    #[test]
    fn widest_span_is_zero_when_empty() {
        let set = DisjointIntervalSet::new();
        assert!(set.is_empty());
        assert_eq!(set.widest_span(), 0);
    }
}
