# ADR 0048 — A dimension states a quantity, at a place

- **Status:** Accepted. Extends [ADR 0035](0035-the-sketch-tool-suite.md) decision 5's relation
  catalogue with the members `Dimension` grew after that record shipped
- **Date:** 2026-08-07

## Context

[ADR 0035](0035-the-sketch-tool-suite.md) decision 5 lists the relations that ship as explicit
verbs — Coincident, Parallel, Perpendicular, Equal, Collinear, Midpoint, Horizontal, Vertical, Fix,
Distance. Distance became `ConstraintKind::Dimension(Dimension::Span)`, and three more members
joined it without a record: `Radius`, `Angle`, and `Diameter`. A fourth thing joined with them that
is not a member at all — `AngleArm`, the type that says WHERE on a curve an angle is read.

The catalogue is not the interesting part; the family is. Every other relation in decision 5 is a
claim with no number in it — two things are parallel, or they are not. A dimension is the other
kind: the author supplies the number, so the number is authored, and everything the repo already
decided about authored quantities applies to it. That is what makes these four one kind with four
members rather than four kinds.

The `Angle` member then ran into a question none of the others have. `Span` names two points and
`Radius` names one curve, and in both cases naming the entity is the whole question. An angle names
two DIRECTIONS, and a curve that turns has a different direction at every point on it, so naming
the arc does not ask anything yet.

## Decision

**A dimension is one constraint kind whose members differ only in what quantity is stated and what
the drawing must supply to state it. Where the quantity is not the same everywhere on what it names,
the member names the place.**

### 1. The members are statically typed, and the type is the quantity's kind

`Span`, `Radius` and `Diameter` each carry a `SketchLength`; `Angle` carries an
`AngleMeasurement`. This is [ADR 0035](0035-the-sketch-tool-suite.md) decision 12's "static above,
dynamic below" arriving at the constraint layer: a radius field takes a length and an angle field
takes an angle, and mixing them does not compile. Nothing here needs the runtime-tagged quantity,
because every one of these fields knows its kind when the code is written.

Each member is therefore an authored quantity in the full sense — the stored expression is the
truth, and a density re-target re-evaluates the three lengths and leaves the angle alone, because
an angle has no density. Restating one is release-and-assert rather than a value poked in place, so
the trial solve stays the one admission door and a number the drawing cannot reach costs the author
nothing.

### 2. A diameter is a member, not a display flag on the radius

The solver has one radius row and a diameter emits it halved. It would be less code to store a
radius and a `show_as_diameter: bool`.

**Rejected, because the number is authored.** An author who typed "one block across" wrote a
diameter. Halving it on the way in throws the expression away and hands back a halved one at the
next density re-target — the exact loss the authored-quantity rule exists to prevent. A flag also
makes `length` mean two different things depending on a neighbouring field, which is the failure the
repo's self-labelling rule forbids.

The two cannot both be asserted about one curve, and nothing checks for that. Both members answer
`subject()` with the curve's id twice, and `is_about_the_same_as` compares subject pairs, so the
second is refused as already asserted by the machinery that was already there. They say the same
thing; the refusal falls out of them saying it about the same thing.

Which one an author gets by default is read from the shape: a whole circle seeds a diameter and an
arc seeds a radius. A closed rim is a hole and is sized across; an open one is a fillet and is sized
out from its center. The rail switches either way, and switching keeps the size and drops the
expression, because half of `2 blocks` is not an expression anyone wrote.

### 3. An angle's arm is a type, and on a curve that turns it names an END

`AngleArm` is either a `Segment` or an `ArcEnd { arc, end }`.

A segment's direction is the same everywhere on it, so naming the segment is the whole question. An
arc's is not, so a place must come with it — and the place is an **end**, not a point on the curve.
A point on the curve would be a coincidence every later solve has to keep agreeing about, and a
solve that drifts would silently change what the angle was asked about. An end is on its own arc by
construction. Nothing has to be maintained for it to stay true.

**A whole circle cannot be an arm.** It has no end, so there is no place to name, and what an author
pointing at a circle wants is `Tangent`. The pick refuses it in those words rather than accepting the
circle and choosing a point on it.

**Which end is read comes from the click.** A pick already carries its locus, and `Tangent` already
uses one to choose its branch. The end nearer where the author clicked is a reading of the gesture,
not an inference about the drawing — which is the distinction [ADR 0041](0041-a-gesture-is-read-from-where-it-started.md)
and [ADR 0042](0042-a-gesture-states-its-own-rigid-set.md) draw, and it lands on the permitted side
of it. Pointing at part of an arc is the author saying which part.

**Neither arm carries a sense, and neither needs one.** The residual is `sin(turn − asked)`,
expanded as `cross · cos θ − dot · sin θ` so no arctangent picks a branch. A sine repeats every half
turn, so reading an arm end-for-end changes nothing it asserts. This also puts the row in the family
[ADR 0035](0035-the-sketch-tool-suite.md) decision 5 already describes: at θ = 0 it IS Parallel's
cross product, and at a quarter turn it is Perpendicular's dot product negated. The three are one
residual at three values.

### 4. Two angles at opposite ends of one arc are two different claims

`is_about_the_same_as` compares subject pairs for everything except Symmetry, whose stored values
participate because the branch is the claim. Angle joins it, for the same reason: an id pair holds
no room for the end, so an angle struck at an arc's start against a line and one struck at its
finish against the same line would collide, and the second would be refused as a restatement of the
first. They are statements about two different tangents.

The comparison is on a TOTAL key — `(entity, 0 | 1 | 2)` for segment, start, finish — sorted before
comparing, so the two ends of one arc never tie and get ordered by luck.

## Consequences

**`Parallel` and `Perpendicular` stay their own kinds.** An author asking for a right angle is not
authoring the number 90, and a badge says that where a dimension line cannot. The shared residual is
an implementation fact, not a reason to merge three verbs into one with two magic values.

**The solver gained an arm shape and an arc-liveness check.** `Relation::Angle` carries
`parametric::sketch::AngleArm`, and a relation naming an arc that has gone is `BuildError::UnknownArc`
rather than a segment-shaped absence — a different absence deserves a different name. An arc arm's
direction is read as the perpendicular of the radius standing at the named end, which is the tangent
there, and a zero-length radius reads as a zero direction rather than a normalized NaN.

**The gizmo synthesizes an arm's leg.** An arc arm has no two endpoints to strike a line through, so
the drawing builds `(end, end + tangent · radius)` — long enough to reach about as far as the curve
does. It is computed in PLANE coordinates and projected after, for the reason a virtual intersection
is: a fact about the drawing found after a perspective divide would be a fact about the camera.

**Nothing here licenses an angle at an arbitrary point on a curve.** If that is ever wanted, it is a
different decision, and the thing it has to answer first is what keeps the point on the curve across
a solve.
