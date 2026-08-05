# ADR 0039 — A preference is measured before the hand

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

A sketch drag runs three passes. The first carries the **rigidity preference** — one row per drawn
edge asking that its span come out of the solve as it went in — and its job is to rank the family
of answers a hand admits by how much the drawing has to CHANGE SHAPE, so a shape travels under one
finger instead of stretching. It only seeds; the later passes carry the hand at full authority with
the preference switched off, so a preference can never cost the author the cursor.

Two defects in that pass were reported as one symptom — dragging an arc slot was unstable, its
endpoints wandered, and dragging an arc's center did not translate the structure.

**The preference was read off the wrong drawing.** `point_move_attempt` writes its hands into the
sketch and *then* prepares the problem, deliberately, so a carried shape starts the settle already
standing rather than distorted around the point that led. But that means rigidity is built from the
post-hand drawing, and a preference read from there asks to keep the distortion. Spans survived it
by luck: a span whose end is the hand becomes its own answer, so it stops asking for anything
rather than asking for the wrong thing.

**An arc's radius had no such luck, because it had no name.** An arc carried no scalar parameter;
its shape lived in one equal-radius residual over four coordinates. Measured after a center drag,
its two ends disagree about how far away they are, and the mean of the two is a radius NEITHER end
has — an invented target the solve then rebuilt the whole arc around.

## Decision

**An arc names its radius as a solver column**, as a circle always has. Two rows replace the one:
each end stands its own radius from the center. Subtract them and the equal-radius condition comes
back exactly, so every reader that reaches an arc through its chord bisector still agrees.

This does not reopen ADR 0038. That ADR calls the radius a derived quantity and models the arc on
the circle; the column is solver-internal, and the document still derives an arc's radius from its
points on demand and never stores it beside them. `planegcs` reaches the same arrangement from the
other side, its `Arc` inheriting `rad` from its `Circle`.

**The column is the radius itself, not its logarithm.** A solve picks the shortest correction among
those satisfying its rows, and "shortest" is measured in whatever coordinates the columns are
written in. A transform is therefore a statement about relative cost, not a private convenience: a
coordinate whose derivative is large is one the solve will spend first. Held as `ln r`, a
forty-voxel arc's radius was forty times cheaper than moving any of its points. Positivity comes
instead from the clamp on the way out and from the geometry, every row that reads a radius equating
it to a distance.

**Every drag sends down where its hands stood**, and the whole hand set with it — for a caller that
moved nothing, where a hand stands is where it stood. Spans and radii are measured there.

**Two rules follow, both saying a preference must not price what the author is setting:**

1. A hand on a curve's **end** loosens that curve's span. Measured honestly, a rigid drawing under
   a pinned hand has exactly one answer — translate — whatever was grabbed and whatever else is
   asserted.
2. **Concentric arcs are one rail family**, and the gap between them is a width. Their radii go
   unheld once a hand has a whole rail. A lone arc — a slot's cap — keeps its hold.

## Consequences

Measured on a center-arc slot of radii 4, 4, 36, 40, 44:

| gesture | before | after |
| --- | --- | --- |
| drag the hub | points scattered, radii 4.0/31.4/35.4/39.4 | all ten points move by exactly the displacement, radii unchanged |
| drag an outer-rail end 6 across | radii reach 5.2/40.6/45.8/51.0 | 4.03/35.99/40.03/44.06 |
| pull the outer rail 2 out | inner rail flew to 33.1 | rail lands exactly on 46, inner holds 36.02, caps take up the width |

An endpoint drag translates the structure rather than sweeping the end around a fixed hub. That is
the honest consequence of the hand outranking the preference: a cursor is generally off the circle
the radius names, so meeting it exactly requires the hub or the radius to give, and a rigid
translation is the answer that satisfies every row. "Radius holds, the arc sweeps" would need the
end to land on the nearest point of the circle instead of under the cursor, which is a different
decision about who outranks whom and is not taken here.

The symmetry rank reading moves from `(7, 11)` to `(9, 9)`, which is the reading becoming right
rather than the freedom changing — a column and a row per arc net to nothing. An arc collapsed to
no sweep had an equal-radius row whose Jacobian degenerated with it, and the rank came back two
short. Named, the radius keeps the pair honest where the difference between the ends no longer is.

This ADR amends the solver consequence recorded in [ADR 0038](0038-a-point-is-placed-never-computed.md):
where that one says an arc's shape is priced by a single equal-radius row, it is now one column and
two rows.
