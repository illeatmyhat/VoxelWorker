// Verus deductive proof: the SlotFreeList allocator's safety contract — no double-allocation and
// always-valid (in-range) slot indices. The extraction-map (decision 6) assigns this stateful
// allocator invariant to the deductive tier: the guarantee is over an unbounded run of
// allocate/free operations, so it rides a data invariant on the free set rather than a bounded
// unrolling.
//
// Source: crates/substrate/src/occupancy/free_list.rs. The load-bearing contract there is that the
// free set is kept "sorted ascending and deduplicated" (a strictly-increasing sequence of slot
// indices, each addressing a real slot), and allocation pops its end. From THAT invariant two
// safety properties follow, proved here:
//   * allocate never returns a slot that is still in the free set  ⇒  no slot is handed out twice
//     while allocated (no double-allocation);
//   * every free index, and every allocated index, is < the backing store length  ⇒  the
//     `slots[slot]` indexing never goes out of bounds (stable, valid indices).
// `free` (the normalize step) is modelled as the sorted-unique insert of one index — the faithful
// model of "sort_unstable + dedup", proved to preserve the invariant; a batch free folds it.
//
// Checked with:  <verus> verification/verus/slot_free_list.rs   (Verus 0.2026.07.12)

use vstd::prelude::*;

verus! {

/// The free set is strictly increasing — sorted ascending AND deduplicated (no index appears
/// twice). This is the exact post-state the source keeps via `sort_unstable(); dedup()`.
spec fn sorted_uniq(s: Seq<u32>) -> bool {
    forall|i: int| 0 <= i < s.len() - 1 ==> (#[trigger] s[i]) < s[i + 1]
}

/// Every free index addresses a real slot: it is below the backing-store high-water mark `n`.
spec fn all_below(s: Seq<u32>, n: int) -> bool {
    forall|i: int| 0 <= i < s.len() ==> (#[trigger] s[i]) < n
}

/// Strictly-increasing adjacency lifts to a strict order at a distance: the free set has no
/// duplicates, and its last element strictly dominates every earlier one.
proof fn lemma_sorted_mono(s: Seq<u32>, i: int, j: int)
    requires
        sorted_uniq(s),
        0 <= i < j < s.len(),
    ensures
        s[i] < s[j],
    decreases j - i,
{
    if i + 1 < j {
        lemma_sorted_mono(s, i + 1, j);
        assert(s[i] < s[i + 1]);
    } else {
        assert(s[i] < s[i + 1]);   // j == i+1
    }
}

/// Model of the `free` normalize step for a single index: insert `idx` into the sorted-unique free
/// set, keeping it sorted-unique and in range (a duplicate free is idempotent — the dedup). A batch
/// free is this folded over the freed indices, so the invariant is preserved throughout.
fn free_one(free: &mut Vec<u32>, idx: u32, Ghost(n): Ghost<int>)
    requires
        sorted_uniq(old(free)@),
        all_below(old(free)@, n),
        (idx as int) < n,
    ensures
        sorted_uniq(final(free)@),
        all_below(final(free)@, n),
{
    // Scan to the first index >= idx; everything before it is strictly smaller.
    let mut p: usize = 0;
    while p < free.len() && free[p] < idx
        invariant
            0 <= p <= free.len(),
            sorted_uniq(free@),
            all_below(free@, n),
            forall|k: int| 0 <= k < p ==> free@[k] < idx,
        decreases free.len() - p,
    {
        p += 1;
    }

    // Already present (free[p] == idx) ⇒ dedup: leave the set untouched.
    if p < free.len() && free[p] == idx {
        return;
    }

    // Otherwise splice idx in at p: its left neighbour (if any) is < idx (scan), its right
    // neighbour (the old free[p], if any) is > idx (scan stopped, and it is not == idx).
    let ghost prev = free@;
    free.insert(p, idx);

    proof {
        let t = free@;
        assert(t == prev.insert(p as int, idx));
        assert forall|i: int| 0 <= i < t.len() - 1 implies (#[trigger] t[i]) < t[i + 1] by {
            if i + 1 < p {
                assert(t[i] == prev[i] && t[i + 1] == prev[i + 1]);
            } else if i + 1 == p {
                // left neighbour prev[p-1] < idx == t[p]
                assert(t[i] == prev[i]);
                assert(t[i + 1] == idx);
                assert(prev[i] < idx);
            } else if i == p {
                // idx < right neighbour prev[p]
                assert(t[i] == idx);
                assert(t[i + 1] == prev[p as int]);
                assert(prev[p as int] >= idx);
                assert(prev[p as int] != idx);
            } else {
                // both shifted from prev by one
                assert(t[i] == prev[i - 1] && t[i + 1] == prev[i]);
            }
        }
        assert forall|i: int| 0 <= i < t.len() implies (#[trigger] t[i]) < n by {
            if i < p {
                assert(t[i] == prev[i]);
            } else if i == p {
                assert(t[i] == idx);
            } else {
                assert(t[i] == prev[i - 1]);
            }
        }
    }
}

/// Model of `allocate`: pop the largest free index (reuse) or append at the high-water mark. Given
/// the free-set invariant, this proves the two safety properties. `slots_len` is the backing-store
/// length; the return is `(slot, new_len)`.
fn allocate(free: &mut Vec<u32>, slots_len: usize) -> (r: (u32, usize))
    requires
        sorted_uniq(old(free)@),
        all_below(old(free)@, slots_len as int),
        slots_len < 0xFFFF_FFFF,
    ensures
        // the invariant is preserved for the next operation ...
        sorted_uniq(final(free)@),
        all_below(final(free)@, r.1 as int),
        // ... the allocated slot addresses a real slot (indexing is in bounds) ...
        (r.0 as int) < (r.1 as int),
        r.1 >= slots_len,
        // ... and the allocated slot is NOT still in the free set: it can never be handed out again
        // until it is freed — no double-allocation.
        forall|i: int| 0 <= i < final(free)@.len() ==> final(free)@[i] != r.0,
{
    match free.pop() {
        Some(slot) => {
            // Reuse: free@ is the old set minus its last (largest) element.
            proof {
                let old_seq = old(free)@;
                assert(slot == old_seq[old_seq.len() - 1]);
                // the popped max differs from every survivor (strict monotonicity)
                assert forall|i: int| 0 <= i < free@.len() implies free@[i] != slot by {
                    assert(free@[i] == old_seq[i]);
                    lemma_sorted_mono(old_seq, i, old_seq.len() - 1);
                }
            }
            (slot, slots_len)
        }
        None => {
            // Append: the free set is untouched; the new index is the old high-water mark, which
            // is above every (in-range) free index, so it is not in the free set.
            let slot: u32 = slots_len as u32;
            (slot, slots_len + 1)
        }
    }
}

fn main() {}

} // verus!
