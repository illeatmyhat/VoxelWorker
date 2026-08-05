# ADR 0041 — A gesture is read from where it started

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

[ADR 0040](0040-a-drag-snaps-to-the-quantity-it-moves-along.md) shipped a snap, a set of stays and
a walk, and measured them on a slot's boundary corner. The author tried it on the dot they actually
grab — a slot's spine end — and reported that nothing had changed: "the other end shakes around too
much, and there's no snapping at all."

Both halves of that were literally true, and every rule in 0040 was involved.

**The dot was not a point.** A slot reified a HANDLE on each spine end: an authored dot standing on
the derived center underneath, held there by a coincidence. It existed because a derived center
could not be dragged — dragging one authored the quantity behind it and did not settle.
[ADR 0038](0038-a-point-is-placed-never-computed.md) had already ended that: no point's coordinates
are anybody else's arithmetic, and every point moves the same way, an arc's center included. The
handle had nothing left to do but stand in the way. The author put the general rule plainly: there
are to be no points that are not fully fledged points, tangent handles excepted, and any extra
point minted on top of another is to be removed in favour of the canonical one.

**A reshape has more hands than the author has fingers.** Grabbing a spine end sends TWO hands: the
point, and the pivot it turns about, named as a hand standing exactly where it already stood so the
drawing is held there rather than sliding out from under the cursor. Every rule in 0040 that asked
"is one vertex being reshaped" counted the hands it was NAMED. So on the one gesture the author
cared about, the snap and the stays both switched themselves off and the far end was free.

**A pin does not stay put to the last bit.** Counting movers by "travelled at all" then failed for
the opposite reason: a settle answers to a tolerance, and a walked drag hands each step the drawing
the last one reached, so by step two the pin is a few ulps off its target and reads as a second
mover. The snap fired once in a nine-step sweep.

**The walk handed out a chord while the drag swept an arc.** Each step's target was a linear
interpolation of the raw cursor, measured against the previous step's landing. The gap between the
chord and the arc is the sagitta — a fixed piece of geometry — while a step's own travel shrinks as
the walk gets finer, and the snap cone is a fraction of that travel. Partway through, the cone loses.
It lost at step six and again at step nine, and step nine is the one that delivers the answer.

**A preference that follows the walk ratifies the walk's mistakes.** Every preference row — spans,
scalar holds, stays — was written against the drawing in front of the step. Re-aimed at each step's
answer, a preference can only say "stay where the last step left you", so whatever that step got
slightly wrong becomes the thing being preserved. The error ratchets and drains into whichever
quantity nothing else prices. On a slot that is its width, which grew 20% over a nine-step sweep —
and grew FURTHER when the steps were made finer, which is the signature of a per-step bias rather
than of curvature.

**A corner named no pivot.** The reshape policy asked whether the held point CENTERS one curve of a
shape that has a center of its own. A slot's outer corner centers nothing, so a corner drag had no
pivot and the hub it turns about drifted along behind the cursor.

## Decision

**A slot's spine is drawn between the boundary's own centers.** `slot_spine_points` returns them;
the coincidences and the minted dots are gone. A linear slot is six points instead of eight, and a
slot commits no `Coincident` at all.

**Every rule counts the hand that MOVED**, through one `Problem::moving_hand`. Stillness is relative
— a hand that has travelled less than a thousandth of the busiest hand's travel is a pin — because
a pin is exact only in exact arithmetic.

**Every step of a walk measures from where the GESTURE started**, not from where the last step
landed. The cone then grows with the drag and keeps, and the quantity being kept is the one the
author had rather than the one the last step settled at.

**A preference is written against the drawing the gesture opened with** — `Rigidity::Preferred`
carries it as `opening`. The problem's own points are not that drawing: the caller writes the hands
into the sketch and prepares the problem afterwards, so `was` is the only record of where the hands
stood, and the opening is the current positions with `was` written back over them.

**A corner names the pivot it turns about.** Where the held point centers nothing,
`pivot_a_reshape_turns_about` seeds from the curves it merely ENDS, which is the same gesture
arriving on the boundary instead of the spine.

## Consequences

Both arc-slot grammars now answer the author's three gestures identically, and answer them as
described. Measured on a curved slot of rails 4, 4, 36, 40 and 44:

| gesture | before | after |
| --- | --- | --- |
| hub pulled 8.6 | every point by 8.6, radii unchanged | unchanged |
| spine end swept 6 — far end | moved 7.99 | moved 0.00001 |
| spine end swept 6 — its radius | 40 → 40.45 | 40.00000 |
| corner pulled 6 — hub | moved 0.245 | moved 0.0002 |
| corner pulled 6 — far corner | moved 0.054 | moved 0.0002 |

The corner numbers are the ADR 0040 shipped row, so a corner naming its pivot is worth more than
every preference tuning tried against it. Weighting the stays was tried across three decades and is
not in the answer: lighter is worse, and heavier converges to the same place.

The walk is now frame-rate independent to seven figures rather than to half a voxel — the same
sweep at 9, 17 and 34 steps agrees exactly, where before it agreed approximately and drifted a
percent across a long one. That creep, recorded as inherent in ADR 0040, was the ratchet.

What remains: a slot sweeping by one end widens about 5% over a six-voxel pull, because the width is
the freedom a slot keeps on purpose and the least-norm answer would rather spend it than rotate a
far end that is 44 voxels out. This is now stable and frame-rate independent rather than
accumulating. A dimension on the width removes it, which is what a dimension is for.
