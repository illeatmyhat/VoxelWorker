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
