# ADR 0021 — Noise as a bounded field operation

- **Status:** Accepted (2026-07-18 — design grill; **implementation not started**). Builds on
  ADR 0019 (the field layer) and ADR 0020 (the field trait). **Corrects a premise of ADR 0020
  Decision 1**: `DebugCloudField` is not fieldless, and the `Option` on `as_field` rests on
  freehand sculpt rather than on this producer.
- **Date:** 2026-07-18
- **Layer:** evaluation semantics + one substrate theorem, with an authoring consequence.

## Context

`DebugCloudField` was treated throughout the ADR 0019/0020 design as the one producer
without a field — the motivating case for `as_field() -> Option<&dyn Field>`. Its own
docstring asserts the claim:

> The cloud field is displaced by FRACTAL PERLIN NOISE (fBm), whose value over a cell has no
> cheap conservative bracket — so this producer is UNBOUNDABLE and returns `None`.

**The claim is false, and it has been suppressing interior elision on this producer for no
reason.** The construction is already field arithmetic
(`crates/document/src/debug_clouds.rs`):

```
per puff   f(p) = radial + BILLOW · fbm(…)      radial = 1 − |p − centre| / R
solid iff  max over puffs of f(p) > 0
```

Bracketing the fBm *over a cell* is indeed hard. But it is never necessary: only the **global
range** of the noise is needed, after which the radial term does all the work.

## Decision

1. **`sup|fractal_noise| = sup|noise|`, exactly — and it is a one-line proof.** The fBm
   implementation divides the octave sum by the sum of amplitudes
   (`crates/substrate/src/noise/perlin.rs:96`), making it a **convex combination** of noise
   samples. A weighted average of values in `[−M, M]` lies in `[−M, M]`. **Octave count, gain
   and lacunarity are therefore irrelevant to the bound**, and the whole question reduces to
   the single-octave Perlin bound.

   The single-octave constant is a **substrate obligation** carrying a literature cite and a
   proof, per ADR 0014. `√3/2 ≈ 0.866` is the figure usually quoted for 3D improved Perlin
   but **must be verified against this implementation's actual gradient set** rather than
   taken on authority. `B = 1` is the safe working value and is sound provided that constant
   is below 1. The existing "roughly `[−1, 1]`" doc comments are replaced by the proven bound.

2. **A noise-displaced field is the base field's interval widened by the amplitude bound.**

   ```
   cell_interval(base ± A · noise)  =  cell_interval(base)  widened by  A · B
   ```

   This is ADR 0019 Decision 7's outset machinery reused verbatim — outset shifts an interval
   in one direction, displacement widens it in both. No new classification primitive.

3. **Noise displacement is a first-class field operation, not a debug special case.** Any
   body with a field may carry a bounded displacement. This is the field-layer construction
   for weathered stone, irregular rock faces and eroded edges — organic surfaces being the
   stated value proposition, and unreachable from parametric primitives alone.

4. **`DebugCloudField` becomes a full `Field` implementor**, classified from geometry alone
   with **no per-cell noise evaluation**:

   ```
   provably solid   d < R(1 − BILLOW·B)      [worst-case billow cannot retract past it]
   provably air     d > R(1 + BILLOW·B)      [best-case billow cannot reach it]
   ```

   Solid holds because the fold takes the max across puffs, so one puff claiming a point
   suffices; air requires the cell to miss **every** puff's bounding ball. At `B = 1` that is
   a `0.58 R` solid core and a `1.42 R` bounding ball. This is a real elision win precisely
   in the case the producer exists to exercise — many disjoint bodies in a large, mostly
   empty volume — where most cells are far from any puff and become provably air instead of
   per-voxel boundary.

5. **`as_field()` keeps its `Option`, on different grounds.** ADR 0020 Decision 1 justified
   it by `DebugCloudField`; that justification is withdrawn. The `Option` stands because
   **freehand sculpt is occupancy-native** — a sparse voxel delta has no analytic field — and
   ADR 0017 Decision 7 already requires that "nothing here may assume geometry is
   producer-analytic only." The trait shape is unchanged; only its rationale is corrected.

## Considered options

- **Bounding the fBm over a cell via a Lipschitz constant on the noise**: rejected as
  unnecessary. It is derivable — with `gain · lacunarity = 1` each octave contributes equally,
  giving `L_fbm ≈ octaves · L_noise · frequency` — but it yields a *shallower* field than the
  radial term does, since the cloud field saturates near `1` at a puff's core while the radial
  term grows without limit away from it. The range argument is both simpler and tighter, and
  needs no gradient theory.
- **Leaving `DebugCloudField` unboundable**: rejected. It costs elision on the exact workload
  the producer exists to represent, and rests on a false claim.
- **Deleting `DebugCloudField` and rebuilding its test content from many small primitives**:
  rejected. Scattered boxes cannot reproduce the irregular boundaries that stress the
  microblock layer, and the producer is load-bearing in the two-layer store, windowed-resolve,
  cell-interval-parity and scene tests. With Decision 4 it stops being an exception at all.
- **Tightening `B` to the exact single-octave supremum immediately**: not required. Any sound
  `B` yields a usable core and ball; a tighter constant only widens the elided regions. Prove
  `B = 1` first, tighten later if measurement justifies it.

## Consequences

- **`substrate`'s noise component gains a proven range bound** and loses its two "roughly
  `[−1, 1]`" hedges. The convex-combination argument is small enough to prove directly; the
  single-octave constant is the real work.
- **Soundness is gated as well as proved.** If `B` is wrong, `src/cell_interval_parity_tests.rs`
  catches it immediately — classification would disagree with per-voxel resolve.
- **`DebugCloudField`'s docstring must be corrected, not merely edited.** It currently asserts
  unboundability as a property of fBm. It is also stale elsewhere: five `pub const`s are
  documented as shared with the ADR 0007 GPU view-resolve, which is retired and deleted, and
  all five are used only within the module. They become private.
- **The onion skin remains a live consumer.** ADR 0012 deleted the volumetric fog *subsystem*;
  the onion-skin feature survives as ghost-shaded clip-slab passes, so the producer's stated
  purpose of exercising it still stands.
- **Displacement composes with everything already decided** — it is an interval widening, so
  it passes through the ordered fold, outset, and emboss without special cases. A displaced
  body may be a cutter.
- **Authoring gains an organic-surface route** that primitives cannot express, and it arrives
  as field arithmetic rather than as a new producer kind.
