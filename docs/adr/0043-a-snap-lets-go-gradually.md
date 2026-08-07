# ADR 0043 — A snap lets go gradually

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

The author asked a question rather than reporting a bug: "when dragging anything around, there
shouldn't be any situations where small changes in movement of the mouse result in massive swings
of movement back and forth like a spring. Do extant constraint solvers introduce dampeners, or is
there something different about the way they work that's inherently more stable?"

**They damp, and so do we already.** planegcs, the solver under FreeCAD, defaults to `GCS::DogLeg`
— a trust-region method — with Levenberg–Marquardt as its fallback.
[`nonlinear_least_squares`](../../crates/substrate/src/nonlinear_least_squares.rs) is Powell's Dog
Leg with an explicit trust region, LM as a singular-matrix repair, and a least-norm branch for the
rank-deficient systems a sketch is by construction. There was no damping to add. What the prior art
actually says about stability is about the QUESTION, not the answer: "start iteration from the
current geometry so the nearest solution is found", and "robust solvers detect ill-conditioning and
fall back to damped or least-squares methods rather than producing nonsense."

So the investigation measured the map instead. Every drag frame rebuilds the preview from the
pre-drag drawing and re-solves with the absolute cursor, which makes a drag a pure function of
cursor position — the same cursor always gives the same drawing. That is a better property than the
warm start planegcs uses, and it costs exactly one thing: no smoothness comes for free, so every
threshold anywhere in the pre-solve logic shows up undamped.

Sweeping a cursor finely and reporting gain — how far the drawing moved per unit of cursor motion —
found two, of very different sizes.

## Decision

**The snap fades out instead of being let go.** [`Problem::snapped`](../../crates/parametric/src/sketch/solve.rs)
tested `across > SNAP_CONE * travel` and dropped the whole correction the moment it was true. The
snapped and unsnapped answers differ by the entire correction exactly where the hand crosses
between them, so the crossing was a jump — and, the cone being a share of travel, a jump with no
bound: gestures of 3, 6 and 12 armed 1.90, 3.76 and 7.46.

In its place, a plateau and a smoothstep:

| where the hand is | what holds |
| --- | --- |
| within `SNAP_HOLD` (0.6) of the cone | the quantity, exactly, as before |
| across the band outside it | a smoothstep from that to nothing, zero SLOPE at both ends |
| past the rim | nothing, reached continuously rather than crossed |

A falloff without the plateau is not a snap. The first attempt faded from the moment the hand was
off the quantity, and a hand a fifteenth of a radius off it then landed a fifteenth off it too,
only slightly pulled in — the behaviour the snap exists to replace. Two slot tests said so.

**The fade is one map, applied to the lead and everything it carries alike.** The turn has to fade
with the radius. Fading only the radius left the set turning through the hand's full angular travel
however weakly the quantity was pulling, so the drawing was neither where a translation would put
it nor where a snap would, and the solve spent a real freedom reconciling the two. Because blending
the coefficients of two complex affine maps yields a third, the faded map is a translation at the
rim, the exact similarity of [ADR 0042](0042-a-gesture-states-its-own-rigid-set.md) on the quantity,
and a similarity — never a distortion — at every pull between.

## Consequences

**The spring the author described is gone where the snap caused it.** Rocking the cursor a fiftieth
of a unit across the rim:

| | before | after |
| --- | --- | --- |
| swing, every rock, forever | 3.79 | 0.067 |
| worst gain over the sweep | 189.57 | 84.39 |

Guarded by `rocking_the_cursor_where_a_snap_gives_up_does_not_rock_the_drawing` and
`a_hand_near_its_own_quantity_still_holds_it_exactly`.

**Two mechanisms that looked like separate bugs were the same threshold.** Crossing the cone also
flipped the walk between one frame and up to sixteen, and one frame against sixteen is not a
rounding difference — it collapses a slot's rails from 36/40/44 to 33.5/38.3/43.2 over eight
degrees. Stepping the walk by `pull * turn` — by how much rotation the snap is actually imposing,
rather than by how far the hand went — closes it without a second rule.

**The bigger instability is not the snap, and this does not fix it.** Measured after, and recorded
as `the_free_sweep_of_a_slot_is_still_spent_arbitrarily`: the far cap of a swept slot slides along
its own arc by up to **2.7** for a cursor step of 0.005, while every radial coordinate in the same
drawing stays smooth to five figures. It is the drawing's free sweep, spent differently for two
nearly identical questions.

Three measurements place it squarely outside this decision:

- Well outside any cone, where no snap fires at all, the gain is **39.25 — identical** with the
  falloff and with the hard threshold it replaced.
- Pinning the walk to sixteen frames always changes nothing.
- Capping the cone at a fixed length, on the theory that the correction's SIZE was the problem,
  made a long gesture **worse** (8.96 against 7.44). The hypothesis was wrong and is recorded so it
  is not tried twice.

So the honest ranking is the reverse of the order they were found in: the cone was a real
discontinuity worth removing, and the free degree of freedom is the one the author will still feel.
Its cure is the one [ADR 0042](0042-a-gesture-states-its-own-rigid-set.md) already named — least
motion emergent from the initial guess, the way planegcs and `SolveSpace` get it — and possibly the
warm start we deliberately gave up for path-independence. That is its own decision, with its own
measurements.

### Amendment, 2026-08-06 — where the free sweep actually is, and four ways it does not get fixed

The 2.7-per-0.005 slot sweep above is **closed** — now `2.8e-25`, and holding the radius is what
closed it, guarded by `the_free_sweep_of_a_slot_is_no_longer_spent_arbitrarily`. The text above is
left as written because it was true when written; this says what replaced it.

**The live instability is the UNSNAPPED drag, and it is far worse than the 39.25 recorded above.**
Walking a curved slot's end out from `[40, 0]` at 0.005 a step:

| heading | worst gain |
| --- | --- |
| 0° | 2.46 |
| 60° | **191** |
| 90° | 1.70 |
| 120° | 1.76 |
| 180° | **1318** |

**It is not a drift; it is isolated single frames.** The answer tracks smoothly — one coordinate
creeping `-0.414` to `-0.420` across the whole walk — and then one cursor value lands at `-1.418`,
or `+1.697`, or `-2.532`, and the next is back on the line. Frequency climbs with travel: sparse at
6.215, roughly every other step by 6.25. The coordinate that jumps is the far cap sliding along its
own arc. Guarded by `the_unsnapped_free_sweep_is_still_spent_arbitrarily`, which asserts the defect
so that fixing it fails the test.

Instrumenting all three solve passes says where it comes from. The preference pass stalls at a
stationary point after **46, 48, 50 or 54 iterations depending on the cursor**, at residual ~4.62 in
every case — a compromise, as designed. The exactness pass then crawls **48 iterations** from that
seed to reach `1.2e-9`, and where it ends up along the free direction is a property of the path, not
of the cursor. Both the clean answer and the outlier satisfy every authored constraint to `1.2e-9`.
Nothing downstream can recover the choice, because there is nothing left to choose on.

**Four approaches were measured and rejected. None of them is a tuning problem.**

- **A weak least-motion term for non-reshaping drags.** No effect at any weight, because the drag is
  already classed `reshaping = true` and the stay rows are already at full weight. The premise —
  that the preference is blind to this motion — is simply false.
- **Least-motion rows on the exactness passes, anchored at the pre-drag drawing.** Every step
  rejected: the penalty rows land in the report the acceptance test reads, so a satisfied drawing
  reports a residual of the penalty's size and the drag is refused.
- **The same, scoped to the middle pass only.** This WORKS on the number — 1318 → 2.45 and 191 →
  1.52, holding from `λ = 1e-6` to `1e-2`, with the smooth headings unchanged to three figures. It
  breaks **11 tests**, and they are not tests pinning an arbitrary answer: a group that must hold
  still moves 15, a slot's half-width goes 1 → 6.06, a point leaves the line it is constrained to,
  and a point the author FIXED moves from `[20, 0]` to `[22.4, -3.7]`.
- **Seeding with the penalty and following it with a clean exact pass.** Still 11. Re-anchoring the
  penalty at the seed rather than the pre-drag drawing: still 11–12.

**Why the magnitude of `λ` barely matters, which is the finding that rules the whole family out.**
Fourteen extra rows change `residual_count`, and the dog-leg chooses between the `JᵀJ` and the `JJᵀ`
branch on whether the system is over- or under-determined. Adding rows can flip that, switching the
entire solve strategy — so a penalty of `1e-6` breaks a `Fix` by 2.4 units. **A weighted penalty
row competes with the constraints and can lose to them.** That is structural, and no weight repairs
it.

**The null-space projection was then built and measured, and it is a fifth rejection.**
`solve_nearest` in [`nonlinear_least_squares`](../../crates/substrate/src/nonlinear_least_squares.rs):
solve as now, then walk the answer back toward its own starting guess along
`(I − Jᵀ(JJᵀ)⁻¹J)(x − anchor)`, each pass required to leave the residual no worse AND get strictly
closer, so it is a strict improvement or an exact no-op. It behaved exactly as designed — **529
tests green, no regression anywhere**, and the outlier at cursor 6.215 came down from `-1.418` to
`-0.064` against neighbours at `-0.033`, a factor of forty.

It is still not shippable. The worst pair on the walk is **unchanged to five decimals**, because a
single linear projection under-shoots a free direction that is CURVED, and the walk inches. Four
passes cost **3.7× the whole sketch suite** and did not reach it. Spending a multiple of every drag
frame to fix some outliers and not the worst one is not a trade worth making.

### And the coordinate that jumps is the SWEEP, which nothing prices

The measurement that should have been taken first. At the worst pair on the 180° walk:

| | radii | sweep |
| --- | --- | --- |
| out 6.290 | 37.7100 / 29.7100 / 33.7100 | **84.20°** |
| out 6.295 | 37.7050 / 29.7050 / 33.7050 | **90.64°** |

Every radius moves by 0.005 — exactly the cursor step, smooth to the last digit. What moves is the
slot's **sweep, 6.44° for a thousandth of cursor**. And the slot was drawn at 90°, so the good
answer is the one that KEPT its sweep and the bad one has spent 5.8° of it.

That names the gap in one line: **the rigidity preference holds spans and it holds radii, and an
arc's sweep is neither.** `EdgeSpan` prices a segment's displacement, `ScalarHold` prices a curve's
shape scalar — which for an arc is its radius alone. Sliding a cap around its hub keeps every radius
exactly, so the preference is indifferent to the one motion that is actually running away.

### The sweep row was then built too, and it RELOCATES the spike rather than removing it

A loosened arc has given up its chord — that is what loosening is, and it is why an end can sweep.
So the row it was missing is the chord's LENGTH, held while its direction stays free: the arc still
turns about its hub, but how far around it reaches is priced. With the radius already held, the
chord length IS the sweep (`chord = 2r sin(θ/2)`). One residual per loosened arc, in the preference
pass only — no change to the exactness passes' shape, so none of the dog-leg branch hazard above.

At full weight it does exactly what it was built to do, and that turns out not to be enough:

| heading | before | chord held |
| --- | --- | --- |
| 0° | 2.46 | **2724** |
| 60° | 191 | 1.53 |
| 90° | 1.70 | 1.73 |
| 120° | 1.76 | 1.79 |
| 180° | 1318 | 2.48 |

**Both spikes are gone and a bigger one has opened at a heading that was smooth.** Halving the
weight swaps them back — 0° returns to 2.45 and 180° goes to 2190 — and 0.2 does the same. At 0.05
it is essentially the unfixed drawing again. There is no weight at which the drag is smooth in every
direction, and the three sweeping tests (`an_arc_slot_end_follows_its_radius`,
`both_arc_slot_grammars_carry_by_the_middle_and_sweep_by_an_end`,
`pulling_a_slots_corner_along_its_rail_sweeps_the_near_end_and_leaves_the_far_one`) break at all of
them.

### What that retires

**"The free sweep is spent arbitrarily" is not the explanation, and this record has carried it since
it was written.** If an unpriced coordinate were the cause, pricing it would remove the spike.
Pricing it moves the spike. The instability attaches itself to whichever direction is softest at the
configuration the descent lands in, and closing one opens the next.

So the cause is the descent, not the model: at scattered cursor values the trust region walks into a
different place, and every coordinate the drawing leaves free is somewhere for that difference to
show. Six approaches have now been measured against it — four penalty variants, a null-space
projection, and this — and none of them touches the descent.

**What is left is the trade this record already named and did not price.** Every drag frame
re-solves from the pre-drag drawing, which buys path-independence: the same cursor always gives the
same drawing, and no history can accumulate. Its cost is stated in the Context above as "no
smoothness comes for free", and these measurements are what that costs — isolated frames at
thousands of times the gain of a smooth one. A warm start from the previous frame is the standard
cure and is what planegcs does; it would trade a property we chose deliberately for one the author
can feel. That is a decision about what a drag IS, not a numerical repair, and it belongs in its own
record with its own measurements.

**A generous fixed tolerance is still wanted, and is still not this.** Real systems anchor snap
tolerance in screen pixels — five by default in OCAD, an off-screen buffer in cloud CAD — because
model space is continuous and scale-free, so a tolerance measured in it has no natural size. Ours
is a share of travel and grows with the gesture. Converting it needs a length computed by the shell
from the camera and passed down, since `parametric` has no camera and the per-layer crate split
keeps the flow downward-only. Worth doing; it bounds the correction rather than smoothing it, and
the measurements above say it is not what the author is feeling.

## Amendment, 2026-08-06 — the ring is inked from the room left, not from the hold

The falloff shipped with the ring drawn at the hold's own strength, on the reasoning that a snap
which fades ought to be drawn fading. The author, on the shipped result: *"It feels fine but the
ghost fade is rather inconsistent and dies quickly."* Both halves are what the hold does, and
measuring it says so.

**The hold spends its whole range in the outer 40% of the cone.** It is exactly one over the
plateau, so the ring sat at full ink until the hand was already 60% of the way out and then
collapsed across `0.4 * cone` — for a radius, `0.3 * travel` of cursor. Early in a gesture that is a
handful of screen points.

**Its steepest slope is `3.75` per cone.** `1.5` from the smoothstep, over a band `0.4` wide. So a
one-point jitter of the mouse against a cone twenty points across swings the ink by nearly a fifth,
and at the start of a gesture, where the cone is a few points wide, it swings the whole range. The
ring strobes.

**And the ratio the hold reads is constant along a straight gesture.** `across` grows like
`travel * sin(heading)` while the cone grows like `share * travel`, so `across / cone` is
`sin(heading) / share` and does not depend on travel at all. Measured on a 90° arc of radius 40:
headings of 0°, 5°, 15° and 30° all sit pinned at a hold of `1.0` for the whole walk, and 60° never
snaps at all — the transition lives entirely inside a 22° window of *wrist angle*. So the fade is
not something the author moves through. It is a level their heading picks at the outset. Widening
the plateau could not have fixed that, which is why the plateau was left where it is.

**Decision: the ring reports how much cone is LEFT.** `KeptQuantity::ghost_ink` is
`1 - across / cone`, clamped, linear. Straight consequences, against the hold it replaces:

| | hold | room left |
|---|---|---|
| fade spans | `0.4 * cone` | `1.0 * cone` |
| steepest ink slope | `3.75 / cone` | `1 / cone` |
| dims from | 60% out | the first step off |

Linear rather than smoothed on purpose: a constant slope is the steadiest ink there is, and
smoothing it would put the peak back at `1.5 / cone`.

**What it costs is that a ring at 40% ink may still be holding its quantity exactly.** That is the
trade, taken deliberately. How much correction is being applied is not something the author can act
on; how much room is left is. A ring going grey while there is still room to act is a warning, and
a ring at full strength right up to the moment it collapses is not.

The physics is untouched — `pull` still scales the correction, the plateau is still `0.6`, and the
drag the author says feels fine answers exactly as it did. Only the ink changed. Bound end to end in
`the_snap_ring_is_inked_from_the_room_left_in_the_cone`, which walks a real drag radially out of its
cone and checks the closed form against every step of it.

## Amendment, 2026-08-06 — the explanation was wrong, and the fix was one level down

Everything above about the free sweep being "spent arbitrarily" described the symptom correctly and
attributed it to the wrong layer. It is not that the authoring vocabulary leaves the sweep unpriced.
It is that the solve formed `JᵀJ`, which squares the condition number past what a `f64` can
factorise, so every step came out of the damping repair — and a damped normal matrix is perturbed
hardest in exactly the directions the constraints pinned down least, which is where the sweep lives.

Two ten-minute experiments would have said so at any point: changing the trust region's initial
radius from 1.0 to 0.25 removes the 1318 spike outright, and raising the iteration budget makes the
191 one worse. Neither knob has any geometric meaning, so neither should have been able to move the
answer.

[ADR 0047](0047-a-free-direction-is-settled-by-a-gauge-not-by-damping.md) has the fix and the
measurements. The six rejections above keep their numbers; the four penalty-row ones lose their
stated cause, because the branch choice they were breaking no longer exists.
