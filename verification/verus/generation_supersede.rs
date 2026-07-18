// Verus deductive proof: the generation-supersede accept side (`GenerationTracker`) — the
// monotonic-counter, newest-wins guarantee. The extraction-map (decision 6) assigns this to the
// deductive tier: the property holds over an UNBOUNDED stream of dispatches, so it is discharged
// from the monotonic-counter argument (a strictly increasing generation totally orders dispatches)
// rather than a bounded unrolling.
//
// Source: crates/substrate/src/supersede.rs — `GenerationTracker`. `next_generation` mints a
// strictly increasing generation and records it as newest; `accepts(g)` returns true iff `g` is
// non-zero and equals the newest dispatched. The theorems proved here are exactly the module's
// stated contract:
//   * generations strictly increase (a later dispatch always outranks an earlier in-flight one);
//   * at most one generation is ever accepted for a given state, and it is the newest;
//   * once a newer generation is dispatched, every previously-current generation is discarded as
//     stale (no stale result swaps in over a fresher state);
//   * before any dispatch nothing is accepted.
//
// Checked with:  <verus> verification/verus/generation_supersede.rs   (Verus 0.2026.07.12)

use vstd::prelude::*;

verus! {

/// The accept decision as a pure predicate: a result of generation `g` is accepted against a
/// tracker whose newest dispatched generation is `latest` iff `g` is non-zero and matches `latest`.
/// This is exactly `GenerationTracker::accepts`.
spec fn accepts_spec(latest: u64, g: u64) -> bool {
    g != 0 && g == latest
}

/// The monotonic-generation bookkeeping — a faithful model of `GenerationTracker`.
struct GenerationTracker {
    /// The generation of the most recent dispatch; `0` before any dispatch.
    latest_dispatched: u64,
}

impl GenerationTracker {
    /// A fresh tracker (nothing dispatched).
    fn new() -> (r: GenerationTracker)
        ensures
            r.latest_dispatched == 0,
    {
        GenerationTracker { latest_dispatched: 0 }
    }

    /// Mint the next generation and record it as newest. Strictly increasing (from 1), so a later
    /// dispatch always outranks an earlier one still in flight. The `< u64::MAX` precondition is
    /// the real counter's implicit domain — 2^64 dispatches is unreachable.
    fn next_generation(&mut self) -> (g: u64)
        requires
            old(self).latest_dispatched < u64::MAX,
        ensures
            final(self).latest_dispatched == old(self).latest_dispatched + 1,
            g == final(self).latest_dispatched,
            g > old(self).latest_dispatched,   // strictly increasing
            g >= 1,
    {
        self.latest_dispatched = self.latest_dispatched + 1;
        self.latest_dispatched
    }

    /// Accept `generation` iff it is the newest dispatched (and non-zero). Matches `accepts_spec`.
    fn accepts(&self, generation: u64) -> (b: bool)
        ensures
            b == accepts_spec(self.latest_dispatched, generation),
    {
        generation != 0 && generation == self.latest_dispatched
    }
}

// ---- The contract, as standalone theorems over the monotonic counter ------------------------

/// Nothing is accepted before any dispatch: a fresh tracker (`latest == 0`) discards every result.
proof fn theorem_nothing_accepted_before_dispatch(g: u64)
    ensures
        !accepts_spec(0, g),
{
}

/// At most one generation is accepted for a given state — acceptance identifies the newest
/// uniquely (there is no ambiguity about "which is current").
proof fn theorem_acceptance_is_unique(latest: u64, g1: u64, g2: u64)
    requires
        accepts_spec(latest, g1),
        accepts_spec(latest, g2),
    ensures
        g1 == g2,
{
}

/// Supersede discards the stale: if `g` was the current (accepted) generation and a strictly newer
/// generation has since been dispatched, `g` is no longer accepted — a stale in-flight result can
/// never swap in over the fresher state.
proof fn theorem_supersede_discards_stale(latest_old: u64, latest_new: u64, g: u64)
    requires
        latest_old < latest_new,
        accepts_spec(latest_old, g),
    ensures
        !accepts_spec(latest_new, g),
{
    // g == latest_old < latest_new  ⇒  g != latest_new
}

// ---- Tying the theorems to the real API over a burst of dispatches --------------------------

/// A burst of dispatches, then the newest-wins check on the resulting state — the machine-checked
/// form of the `only_final_generation_accepted_after_burst` unit test, but the stale-discard holds
/// for EVERY earlier generation, not the sampled few. Five dispatches from a fresh tracker leave
/// `latest == 5`; only generation 5 is accepted; generations 1..5 are all superseded.
fn burst_then_only_newest_wins() {
    let mut tracker = GenerationTracker::new();
    let mut k: u64 = 0;
    while k < 5
        invariant
            tracker.latest_dispatched == k,
            k <= 5,
        decreases 5 - k,
    {
        let _g = tracker.next_generation();
        k += 1;
    }
    assert(tracker.latest_dispatched == 5);

    // Only the newest generation is accepted ...
    let newest_accepted = tracker.accepts(5);
    assert(newest_accepted);
    // ... and every earlier generation is discarded as stale (unbounded over the whole range).
    assert(forall|stale: u64| 1 <= stale < 5 ==> !accepts_spec(tracker.latest_dispatched, stale));
    // ... including generation 0, which is never valid.
    let zero_accepted = tracker.accepts(0);
    assert(!zero_accepted);
}

fn main() {}

} // verus!
