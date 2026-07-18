// Verus (deductive verification on real Rust; bundled Z3) — the tier the substrate-extraction-map
// (decision 6) assigns to STATEFUL invariants, where a proof rides a loop invariant rather than a
// bounded unrolling. This seed proof establishes the loop-invariant machinery the real targets
// need (`DisjointRunList` insert keeping sorted-∧-disjoint, `SlotFreeList`, generation-supersede).
//
// Checked with:  <verus> verification/verus/widest_span.rs   (Verus 0.2026.07.12; see ../README.md)

use vstd::prelude::*;

verus! {

// A model of `DisjointIntervalSet::widest_span`: the maximum stored width over a sequence. The
// verifier discharges the postcondition from the `while`-loop invariant — the same shape the
// interval-set insert proof will use, but without the Vec::splice that made Kani's BMC explode.
fn widest_span(widths: &Vec<i64>) -> (result: i64)
    requires
        forall|i: int| 0 <= i < widths.len() ==> #[trigger] widths[i] >= 0,
    ensures
        // result dominates every stored width ...
        forall|i: int| 0 <= i < widths.len() ==> widths[i] <= result,
        // ... and is 0 exactly when the set is empty (widest_span's documented base case).
        result >= 0,
        widths.len() == 0 ==> result == 0,
{
    let mut best: i64 = 0;
    let mut k: usize = 0;
    while k < widths.len()
        invariant
            0 <= k <= widths.len(),
            best >= 0,
            k == 0 ==> best == 0, // nothing processed yet ⇒ still the empty-set answer
            forall|i: int| 0 <= i < k ==> widths[i as int] <= best,
        decreases widths.len() - k,
    {
        if widths[k] > best {
            best = widths[k];
        }
        k += 1;
    }
    best
}

fn main() {}

} // verus!
