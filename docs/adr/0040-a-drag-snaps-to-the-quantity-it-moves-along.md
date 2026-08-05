# ADR 0040 — A drag snaps to the quantity it moves along

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

[ADR 0039](0039-a-preference-is-measured-before-the-hand.md) left one consequence standing open:
an arc endpoint drag TRANSLATED the drawing instead of sweeping the end around a hub that stays.
The reasoning there was honest and the conclusion was wrong-headed. A cursor is generally off the
circle the radius names, so "the end lands exactly under the cursor" and "the radius holds" cannot
both be true; the hand outranks the preference, so the radius gives. And because a translation
satisfies every row exactly, no strengthening of a preference can beat it — nothing outranks an
answer at zero residual.

Three further things were true and only showed up under measurement.

A preference prices an arc by its CHORD, and a chord is not something a sweep leaves alone: swing
one rail's end around and every concentric sibling's chord shortens with it. On a curved slot, one
rigid sibling outvoted the sweep by itself.

A span is TRANSLATION-INVARIANT, so the whole preference is indifferent to travel. That is
deliberate — it is what lets a shape be carried under one finger — but it means a lone vertex grab
had nothing at all saying the rest of the drawing should stay, and the arithmetic settled it by
spreading the correction over every coordinate it touched.

And a solve is LOCAL. A snapped drag is a rotation, which is the motion a linearization is worst
at, so the same gesture delivered in one frame and in twenty-four settled in different places.

## Decision

**A lone hand is pulled onto a quantity its own curve already had, when it is moving along one.**
Every curve names a distance — a segment its length, an arc end its radius — and the circle that
distance draws is a locus the hand can be snapped to. The disagreement is then settled BEFORE the
solve rather than inside it: with the cursor on the circle, the sweep is an exact answer too, and a
cheaper one, and the solve finds it without being told to.

The rule is the same for every curve, which is what keeps a slot nothing but its parts — at a
rail's cap the corner belongs to both an arc and a segment, and the cone below picks between them
by which one the hand is actually moving along.

**The cone is a quarter of the hand's travel**, about fifteen degrees, and nothing else. Moving
along a circle is second-order off it, so a sweep stays snapped however far it goes, while a
deliberate pull across leaves at once and the quantity is the author's to set again. A frame of a
real drag is small, so a gesture re-snaps every frame and the quantity survives the whole of it.
This is why it reads as a snap rather than as a mode.

**Concentric arcs are one rail family for CHORDS as well as radii**, extending the rule ADR 0039
already applied to the family's radii.

**A lone vertex grab prices place, not only shape.** Points the hand does not hold are held where
they stood, for that gesture alone; every other gesture — a hand on a center, the several hands of
a carry — leaves travel free, or a carry would deform instead of moving. This is the objective a
commercial parametric drag solves, every point weighted toward where it stood and the cursor
weighted above them, with the span rows added on top. It still only SEEDS: the second pass hands
the cursor full authority regardless.

**A drag walks its turn in steps of about a degree, at most sixteen.** Only a snapped drag pays,
and only when it turns far enough to have to. Each step is handed the drawing the walk has reached,
exactly as the document hands one frame of a real drag to the next — a problem carries its own
positions as the reference its preference is read against, so a step given the original drawing
re-asks for the original shape and the walk says nothing the first step did not.

## Consequences

Measured on a curved slot of rails 4, 4, 36, 40 and 44, its corner pulled six voxels sideways —
the far end and the hub are what should not move:

| | far end | hub | rails |
| --- | --- | --- | --- |
| before (ADR 0039) | 3.775 | 3.624 | 4.04 / 35.99 / 40.03 / 44.07 |
| snap alone | 2.729 | 2.729 | 4.00 / 36.00 / 40.00 / 44.00 |
| and the family's chords | 0.108 | 0.487 | 4.10 / 36.16 / 40.27 / 44.37 |
| and the walk (shipped) | 0.054 | 0.245 | 4.04 / 36.10 / 40.14 / 44.18 |

The last row is the answer at EVERY frame rate. Delivered in one frame, two, eight or twenty-four,
the rails agree within half a voxel; before the walk, one frame collapsed them to 33.5 / 38.3 /
43.2 while twenty-four held them.

Pulling the same corner straight out still meets the cursor exactly and grows the rail with it, so
a slot can be widened by its own corner. Dragging a hub still translates all ten points by exactly
the displacement with every radius unchanged, and dragging a rail's body still lands it exactly
where it was pulled.

A quantity creeps by about a percent across a long sweep, because each frame snaps to the value the
last one settled at and nothing dimensions it. Holding the quantity harder does not fix this: the
radius column was tried PINNED — taken out of the parameter vector for the seed pass, so it could
not be spent at all — and moved the outcome by 0.04 of a voxel, which is noise. The creep is in the
passes that follow, where the hand and then the drawing have their say and the radius is free
again by design. Dimensioning the radius removes it; that is what a dimension is for.
