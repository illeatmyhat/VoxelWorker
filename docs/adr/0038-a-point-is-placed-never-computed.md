# ADR 0038 — A point is placed, never computed

- **Status:** Accepted. §1's solver arrangement **amended** by
  [ADR 0039](0039-a-preference-is-measured-before-the-hand.md) — an arc names its radius as a
  column, so the sixth coordinate buys one column and two rows, not one row. See the amendment
  below for the two names §1 gets wrong.
- **Date:** 2026-08-05

## Context

Two sketch entities reify a point whose coordinates the drawing owns rather than the author:

- `Arc::center`, recomputed from the two endpoints and the stored sweep.
- `Conic::shoulder`, recomputed from the two endpoints, the control point, and the stored rho.

Both are real `Point` entities with stable ids. They select, snap, hit-test, drag and take
constraints like any other point, and `Sketch::sync_derived_points` overwrites their coordinates
after every edit. `Sketch::is_derived_point` is the predicate that tells them apart, and eleven
call sites across four crates consult it to decide whether a point is a freedom, whether it may be
pruned, whether it may be dragged, whether it counts toward a degree of freedom, and whether it is
worth drawing a dot on.

The reified derived point was introduced to solve a real problem — the sweep and rho are the one
authored freedom each curve has with no other handle, so without a point to grab there is nothing
to drag. It has since cost more than it bought:

1. **Identity by position.** Three arcs of an arc slot are concentric, and each minted its own
   private center point at the same coordinates. Resizing the slot made them disagree and the
   author saw the extra dots. Nothing structural said they were one point, so sharing had to be
   decided by comparing coordinates — and identity decided by distance splits under a drag.
2. **First-writer-wins.** Once centers ARE shared, `sync_derived_points` has two curves claiming
   authority over one point's coordinates, and the tie is broken by iteration order. The sharer's
   own parameter then has to be back-derived so it agrees with the winner.
3. **A parallel model in the solver.** `parametric` cannot hold a point that is a function of other
   points in its free-parameter vector, so `CenterOf::{Point, Arc}` exists to say "this center is a
   point slot" or "this center is arithmetic over an arc's row", and every relation naming a center
   branches on it.
4. **It is a lie to the author.** The dot is draggable and it is not a freedom. Dragging it
   re-solves the curve's parameter, which is a different gesture wearing the same clothes.

The alternative was already in the repo, and it is what `Circle` and `Ellipse` do. A circle's
center is placed; its radius is authored beside it. An ellipse stores three placed points —
center, major endpoint, width point — and derives its radii and orientation from them. Neither has
a derived point, neither needs `is_derived_point`, and two concentric circles share a center by
naming the same id.

## Decision

**A point entity's coordinates are always authored. The drawing derives quantities, never points.**

A quantity the drawing derives is an authored-quantity measurement — an `Angle`, a `Length`, a ratio —
computed on demand from the points that determine it and never persisted beside them. What the
author placed is in the document; what follows from it is computed where it is needed.

### 1. An arc is two endpoints and a placed center, swept counter-clockwise

`Arc::bulge` is removed. `Arc::center` becomes an authored point like `Circle::center`.

The arc runs **counter-clockwise from `from` to `to` about `center`**, always. The direction is not
stored because the endpoint ORDER already carries it: an arc bent the other way is the same three
points with the ends swapped. The sweep is then unambiguous — the counter-clockwise angle from
`from` to `to`, in `(0, 360)` — and is derived as an `AngleMeasurement` wherever it is wanted. The
radius is derived as a `LengthMeasurement`.

This over-parameterizes: three placed points are six coordinates for a five-freedom arc. The sixth
is spent on an **equal-radius residual** the arc contributes to every solve — `‖center − from‖ =
‖center − to‖`. That row is the price, and it is the honest one: it says out loud the thing the
old design hid inside `sync_derived_points`.

This amends ADR 0037 for arcs. `ArcSweep` and the `CurveParameter` authority it instantiates are
retired; `CircleRadius`, the other instantiation, is unaffected and stays exactly as 0037 decided.

Retiring `ArcSweep` costs no authored quantity, because there was never a path to author one. Every
arc the tools make carries `ArcSweep::free`; the `Fixed` arm is reachable only by deserializing a
document that names one, and no gesture, inspector field or edit produces it. A sweep an author
wants HELD becomes an angle dimension when that relation lands — the same way a distance is held —
and until then a sweep is exactly as free as it already was.

### 2. A conic is two endpoints and a placed control point

`Conic::shoulder` is removed. `Conic::rho` stays authored, and stays the one freedom with no point
of its own.

Rho is re-authored by dragging the curve's BODY near its middle, through the same
`drag_curve_through` door every other curve uses — the grip lands on the curve and the curve's own
freedom takes up the motion. This is what dragging the shoulder did; it stops needing a point
standing there to do it.

### 3. `is_derived_point` is deleted

With nothing left to answer for, the predicate and every branch that consults it go. `Point` gains
no flag in its place: `PointLifetime` continues to say when a point may be swept, and `EntityRole`
continues to say whether it is real or construction, and neither is about authority any more
because nothing is.

### 4. `ABSENT_DERIVED_POINT` and the repair that materializes it are deleted

Both derived points used a sentinel id for documents written before they were reified, which
`Sketch::repair` filled in on load. There is nothing to materialize once the points are authored,
and per the standing no-back-compat position for configuration, a document that predates this
change is not migrated.

## Consequences

**Concentricity becomes structural.** Two arcs share a center by naming one id. The arc slot's
three arcs are one center by construction, and no coordinate comparison decides it. `CenterOf`,
the first-writer-wins tie-break, and the sweep back-derivation all go with it.

**The solver gets one more row per arc and loses a special case.** An arc's center becomes an
ordinary point in the free-parameter vector, so every relation that names one stops branching. The
equal-radius rows are cheap and well-conditioned; the branch they replace was neither.

**A three-point arc is the natural authoring gesture** — place two ends and a point the arc passes
through, and the center follows from the three. `connect_arc(from, to, angle)` becomes a
convenience that computes where the center has to be and places it there, which is what a slot
builder and a test want.

**The drawn arc always passes through both of its ends.** The stored center is projected onto the
chord's perpendicular bisector before the geometry is read, so an unsolved or mid-drag drawing
still shows a real arc between the two endpoints rather than one that misses them. The projection
throws away the center's motion ALONG THE CHORD, and that is exactly the freedom the equal-radius
residual removes — the two agree, and in a solved drawing the projection is the identity.

**Nothing here licenses a third case yet.** A fit-point spline's tangent arms are the remaining
structure that keeps a point and a derived twin — `sync_tangent_arms` and `carry_authored_handles`
maintain the mirrored arm. The same law would collapse them to one arm and a derived `Length`, and
that is a separate decision made when it is taken.

## Amendment, 2026-08-07 — §1 prices the arc wrongly, and names two types that do not exist

Three corrections, none of which touch the law. The law is that a point is placed; what follows is
housekeeping on how §1 said it.

**The sixth coordinate is one column and two rows, not one row.**
[ADR 0039](0039-a-preference-is-measured-before-the-hand.md) already decided this and says so in
its own closing paragraph; §1 was never pointed back at, so a reader landing there reads a retired
arrangement as current. What `ProblemBuilder::add_arc` actually does is mint an optional free
positive-radius column per arc, seeded by `arc_radius_seed` — the MEAN of what the two ends
currently say, so neither end is privileged mid-drag, which is exactly when the seed is read — and
spend it on two rows, `‖center − from‖ = r` and `‖center − to‖ = r`. Subtract them and §1's
equal-radius condition comes back exactly, so the freedom count is unchanged: six coordinates and
one row is five, and so is six coordinates, one column and two rows. An arc whose ends both sit on
its center gets no column and keeps §1's single row, because there is no positive radius to name.

The column is **solver-internal**. Nothing writes it back; the document still derives an arc's
radius from its three points on demand and never stores it beside them, which is what this record
means by a derived quantity. Naming a quantity for the duration of one solve does not persist one.

**`LengthMeasurement` names nothing.** §1 says an arc's radius "is derived as a `LengthMeasurement`".
The type is `parametric::units::Measurement`, which is always a length; a sketch wraps it as
`document::sketch::SketchLength`. §1's `AngleMeasurement` is right.

**The authored-quantity family is separate static types, not one type with a kind.** The Decision
paragraph above cited a since-consolidated record for "a `Measurement` — an `Angle`, a `Length`, a
ratio", which reads as one kinded value. It is not: `Measurement` is a length (an exact-rational
block term plus a whole-voxel term, evaluated at a density supplied per solve) and
`AngleMeasurement` is an angle (exact degrees, density-free), and they are distinct types on
purpose — they share retention semantics and none of the arithmetic, so a shared representation
would put a kind check at every length call site that could never fail. The one place a kind is not
known when the code is written is inside an expression, where `wall / gap` has no dimension until it
is evaluated; that single case is served by `parametric::quantity::Quantity`, an exact value
carrying a runtime `Dimension`. The dead link is rewritten to prose above, matching how the
2026-07-31 consolidation handled every other reference to that record.
