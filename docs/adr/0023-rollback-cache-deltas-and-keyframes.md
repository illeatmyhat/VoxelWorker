# ADR 0023 — The rollback cache: deltas with periodic keyframes

- **Status:** Accepted (2026-07-20 — design grill; **implementation not started**). Builds on
  ADR 0022 (which placed the rollback cursor) and ADR 0017 (the ordered fold being scrubbed).
  Adds a **derived** category to ADR 0022's classification, which tightens that ADR's stated
  weak point.
- **Date:** 2026-07-20
- **Layer:** evaluation caching, with a persistence consequence.

## Context

ADR 0022 put a rollback cursor in each composition scope. This record decides what happens
when you author at one, and what it costs to move it.

**The insert semantics, corrected from Fusion 360 rather than guessed at.** Dropping a node at
the cursor inserts it there and **advances the cursor by exactly one**, so it sits just past the
new node and you see what you just made. The cursor does **not** jump to the end of the fold,
and it does **not** stay behind the new node. Downstream nodes are rebuilt on top of the
insertion — their results exist — but they remain rolled back as far as the *view* is
concerned until the cursor is moved forward deliberately.

So the downstream work is **cached, not displayed**. That distinction is the whole design: it
makes moving the cursor a lookup rather than a recompute.

**Why that matters here specifically.** The per-edit cost is measured
(`tests/edit_cost_probe.rs`, `tests/remesh_cost_probe.rs`): resolving an edit is flat in scene
size (1.7–4.0 ms even at 819 M voxels), but a **wholesale re-mesh is linear in resident
chunks** — ~395 ms at 3125 chunks, 307× the incremental path. A scrub that recomputed each step
would pay that per step. On a long fold that is not a slow feature, it is an unusable one.

## Decision

**The rollback cache is per-node sparse deltas with periodic keyframes.**

Three shapes were considered:

- **Every prefix state** — position *k* caches the whole accumulated body after *k* nodes.
  Scrubbing is a pure lookup. Rejected: that is N two-layer stores for an N-node fold, and the
  huge fixture is 3125 chunks *per state*. It is the "no dense grids anywhere" problem wearing
  a different hat, and it scales with scene size × fold length.
- **Per-node deltas alone** — each node caches only what it changed. Bounded by edit footprint
  rather than scene size, which is the property all three cost probes independently found
  (**cost tracks locality, not extent**). Rejected alone: scrubbing to position 400 replays 400
  deltas.
- **Deltas with periodic keyframes** — chosen. Storage stays proportional to the edits rather
  than to scene size × fold length, while seek time stays bounded by the keyframe interval. The
  interval is a tunable that trades memory against seek latency, and it should be **set from a
  measurement, not a preference** — the same standard the live-vs-ghost question was held to.

**The cache is derived state, and reaches neither the document nor the dump.**

This adds a category to ADR 0022 with a *checkable* admission test, rather than a second
escape hatch beside `transient`:

> **Derived** — reconstructible from classified state alone, by recomputation. A field may be
> classified derived only if dropping it changes how long something takes and nothing else.

That test is the point. ADR 0022 warned that `transient` would rot, because marking a field
transient will always be the cheapest way to silence the compiler. `derived` cannot rot the
same way: the claim it makes is falsifiable, and a reviewer can ask "reconstructible from
what?" and get a real answer or catch a real bug. A cache that cannot be rebuilt from
classified state is not derived — it is undeclared truth, and the compiler error is correct to
demand it be classified properly.

The dump's law from ADR 0022 — *a scene must be completely reproducible from it* — is satisfied
without storing the cache, because the cache is reproducible. Storing it would make a repro
load faster, never more faithful.

## Consequences

- **An upstream edit invalidates every downstream delta.** This is the classic parametric
  timeline cost and it is not avoidable, only bounded. Editing node 3 of 400 discards the cache
  from 3 onward. The saving grace is the measured one: rebuilding a delta is an *edit-footprint*
  cost, not a scene cost, so invalidation is proportional to what actually changed. That should
  be verified rather than assumed once the cache exists.

- **Deltas must be invertible to scrub backwards.** A subtract that removes material has to
  remember what it removed. This is the same "remember the prior state" obligation that the
  `enabled`-vs-rollback separation ran into in ADR 0022, at a different scale — and it is the
  reason keyframes earn their keep even where seek time would be acceptable: a keyframe is an
  escape from having to invert anything.

- **Keyframe placement is not obviously uniform.** A fixed interval is the simple answer, but
  the natural one may be *cost-weighted* — a keyframe after an expensive node rather than every
  N nodes. Unsettled, and worth measuring before choosing.

- **The delta representation is not decided here.** It should compose with the two-layer chunk
  store (ADR 0010) rather than inventing a parallel sparse format, but whether a delta is a set
  of dirty chunks, a chunk-granular diff, or something narrower is open.

## Open

- The keyframe interval, and whether it is uniform or cost-weighted.
- The delta representation, and how it composes with the two-layer store.
- Whether scrubbing backwards inverts deltas or always seeks to the previous keyframe and
  replays forward. The second is simpler and may be fast enough; it is a measurement.
