// Verus deductive proof: DisjointIntervalSet::insert preserves the normalization invariant
// (sorted ∧ pairwise-disjoint ∧ non-touching ∧ non-empty). This is THE target the
// substrate-extraction-map (decision 6) reserved for a deductive prover: Kani's bounded model
// checker exploded on the `Vec::splice`-backed insert (thousands of VCCs on a 3-interval set — a
// std-collection-mutation BMC pathology, see the NOTE at the head of the source file), so here the
// stateful invariant rides LOOP INVARIANTS rather than a finite unrolling.
//
// The model reproduces the exec algorithm of
// crates/substrate/src/interval/disjoint_interval_set.rs faithfully — the O(1) ascending fast
// paths (empty / gap-append / extend-last) and the general skip-left + merge splice — replacing
// only `Vec::splice` (which Verus, like CBMC, has no tractable spec for) with an explicit
// prefix ++ [merged] ++ suffix rebuild that yields the identical result sequence. The invariant
// proved is exactly the one documented on `DisjointIntervalSet`: after any insert the intervals
// are non-empty and consecutive intervals leave a strict gap.
//
// Checked with:  <verus> verification/verus/disjoint_interval_set_insert.rs   (Verus 0.2026.07.12)

use vstd::prelude::*;

verus! {

/// The stored-set invariant: every interval is non-empty (`lo < hi`) and consecutive intervals
/// leave a strict gap (`s[i].hi < s[i+1].lo`). That single gap condition, together with
/// non-emptiness, forces sorted-ascending ∧ pairwise-disjoint ∧ non-abutting — the full contract.
spec fn valid(s: Seq<(i64, i64)>) -> bool {
    &&& (forall|i: int| 0 <= i < s.len() ==> (#[trigger] s[i]).0 < s[i].1)
    &&& (forall|i: int| 0 <= i < s.len() - 1 ==> (#[trigger] s[i]).1 < s[i + 1].0)
}

/// Across a valid set the interval starts are strictly increasing, hence non-strictly monotonic at
/// a distance. Used to carry the left-gap through the merge loop.
proof fn lemma_valid_mono(s: Seq<(i64, i64)>, i: int, j: int)
    requires
        valid(s),
        0 <= i <= j < s.len(),
    ensures
        s[i].0 <= s[j].0,
    decreases j - i,
{
    if i < j {
        lemma_valid_mono(s, i, j - 1);
        assert(s[j - 1].1 < s[j].0);   // gap at i = j-1
        assert(s[j - 1].0 < s[j - 1].1);
    }
}

/// Appending a non-empty interval that strictly clears the current last keeps the set valid.
proof fn lemma_valid_push(s: Seq<(i64, i64)>, x: (i64, i64))
    requires
        valid(s),
        x.0 < x.1,
        s.len() == 0 || s[s.len() - 1].1 < x.0,
    ensures
        valid(s.push(x)),
{
    let t = s.push(x);
    assert forall|i: int| 0 <= i < t.len() implies (#[trigger] t[i]).0 < t[i].1 by {
        if i < s.len() {
            assert(t[i] == s[i]);
        } else {
            assert(t[i] == x);
        }
    }
    assert forall|i: int| 0 <= i < t.len() - 1 implies (#[trigger] t[i]).1 < t[i + 1].0 by {
        assert(t[i] == s[i]);
        if i + 1 < s.len() {
            assert(t[i + 1] == s[i + 1]);
        } else {
            assert(t[i + 1] == x);
            assert(s[s.len() - 1].1 < x.0);
        }
    }
}

/// Replacing the last interval with a non-empty one whose `lo` still clears the predecessor keeps
/// the set valid (the extend-last fast path).
proof fn lemma_valid_update_last(s: Seq<(i64, i64)>, x: (i64, i64))
    requires
        valid(s),
        s.len() > 0,
        x.0 < x.1,
        s.len() == 1 || s[s.len() - 2].1 < x.0,
    ensures
        valid(s.update(s.len() - 1, x)),
{
    let k = s.len() - 1;
    let t = s.update(k, x);
    assert forall|i: int| 0 <= i < t.len() implies (#[trigger] t[i]).0 < t[i].1 by {
        if i == k {
            assert(t[i] == x);
        } else {
            assert(t[i] == s[i]);
        }
    }
    assert forall|i: int| 0 <= i < t.len() - 1 implies (#[trigger] t[i]).1 < t[i + 1].0 by {
        if i + 1 == k {
            assert(t[i] == s[i]);
            assert(t[i + 1] == x);
            assert(s[i].1 < x.0);   // predecessor gap: s[k-1].1 < x.0
        } else {
            assert(t[i] == s[i]);
            assert(t[i + 1] == s[i + 1]);
        }
    }
}

/// The model of `DisjointIntervalSet::insert`. `intervals` is the normalized store; `[lo, hi)` the
/// interval to union in. Precondition: the store is valid and the inserted interval is non-empty.
/// Postcondition: the store is still valid.
#[verifier::rlimit(100)]
fn insert(intervals: &mut Vec<(i64, i64)>, lo: i64, hi: i64)
    requires
        lo < hi,
        valid(old(intervals)@),
    ensures
        valid(final(intervals)@),
{
    let n = intervals.len();

    // Fast path — empty store: the single non-empty interval is trivially valid.
    if n == 0 {
        let ghost e = intervals@;
        assert(e.len() == 0);
        intervals.push((lo, hi));
        proof { lemma_valid_push(e, (lo, hi)); }
        return;
    }

    let last_lo = intervals[n - 1].0;
    let last_hi = intervals[n - 1].1;

    // Fast path — a real gap right of the last run: append.
    if lo > last_hi {
        let ghost prev = intervals@;
        assert(prev[prev.len() - 1].1 == last_hi);
        intervals.push((lo, hi));
        proof { lemma_valid_push(prev, (lo, hi)); }
        return;
    }

    // Fast path — overlaps/abuts the last run from the right: extend it (or leave it).
    if lo >= last_lo {
        if hi > last_hi {
            let ghost prev = intervals@;
            assert(prev[prev.len() - 1].0 == last_lo);
            intervals.set(n - 1, (last_lo, hi));
            proof { lemma_valid_update_last(prev, (last_lo, hi)); }
            return;
        }
        // A contained interval: the store is untouched and stays valid.
        assert(valid(intervals@));
        return;
    }

    // General path (rare): the interval starts left of the last run. Skip runs strictly left of it,
    // merge everything it touches, then rebuild prefix ++ [merged] ++ suffix.
    let mut start: usize = 0;
    while start < intervals.len() && intervals[start].1 < lo
        invariant
            0 <= start <= intervals.len(),
            valid(intervals@),
            start == 0 || intervals@[start - 1].1 < lo,
        decreases intervals.len() - start,
    {
        start += 1;
    }

    let mut merged_lo: i64 = lo;
    let mut merged_hi: i64 = hi;
    let mut end: usize = start;
    while end < intervals.len() && intervals[end].0 <= merged_hi
        invariant
            start <= end <= intervals.len(),
            valid(intervals@),
            merged_lo <= lo,
            merged_hi >= hi,
            start == 0 || intervals@[start - 1].1 < merged_lo,
        decreases intervals.len() - end,
    {
        proof { lemma_valid_mono(intervals@, start as int, end as int); }
        if intervals[end].0 < merged_lo {
            merged_lo = intervals[end].0;
        }
        if intervals[end].1 > merged_hi {
            merged_hi = intervals[end].1;
        }
        end += 1;
    }

    // Post-loop facts:
    //   merged_lo <= lo < hi <= merged_hi                        (merged is non-empty)
    //   start == 0  || intervals[start-1].1 < merged_lo          (left gap, loop invariant)
    //   end == len  || merged_hi < intervals[end].0              (right gap, negated guard)
    assert(merged_lo < merged_hi);
    assert(end == intervals.len() || merged_hi < intervals@[end as int].0);

    let ghost src = intervals@;

    let mut result: Vec<(i64, i64)> = Vec::new();

    // Copy the prefix intervals[0..start] — each already cleared by its predecessor.
    let mut i: usize = 0;
    while i < start
        invariant
            0 <= i <= start,
            start <= src.len(),
            src == intervals@,
            valid(src),
            valid(result@),
            result@.len() == i,
            i > 0 ==> result@[i as int - 1] == src[i as int - 1],
        decreases start - i,
    {
        proof {
            if i > 0 {
                assert(src[i as int - 1].1 < src[i as int].0);   // gap ⇒ push boundary
            }
            lemma_valid_push(result@, src[i as int]);
        }
        result.push(intervals[i]);
        i += 1;
    }

    // Append the merged run: its lo clears the prefix's last (left gap).
    proof {
        assert(start > 0 ==> result@[start as int - 1] == src[start as int - 1]);
        lemma_valid_push(result@, (merged_lo, merged_hi));
    }
    result.push((merged_lo, merged_hi));
    assert(result@[result@.len() - 1] == (merged_lo, merged_hi));

    // Copy the suffix intervals[end..] — the merged run clears the first (right gap), and the rest
    // keep their own gaps.
    let mut j: usize = end;
    while j < intervals.len()
        invariant
            end <= j <= intervals.len(),
            src == intervals@,
            valid(src),
            valid(result@),
            result@.len() >= 1,
            j < src.len() ==> result@[result@.len() - 1].1 < src[j as int].0,
        decreases intervals.len() - j,
    {
        proof {
            assert(src[j as int].0 < src[j as int].1);   // non-empty
            assert(j as int + 1 < src.len() ==> src[j as int].1 < src[j as int + 1].0);
            lemma_valid_push(result@, src[j as int]);
        }
        result.push(intervals[j]);
        j += 1;
    }

    *intervals = result;
    assert(valid(intervals@));
}

fn main() {}

} // verus!
