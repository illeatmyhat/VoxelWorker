# ADR 0035 — The sketch tool suite: a constraint solver, a geometric arrangement, and a parametric crate

- **Status:** Accepted
- **Date:** 2026-07-30
- **Supersedes:** [ADR 0030 §2](0030-sketch-as-entity-collection.md)'s topological region
  ("a visual crossing with no shared point makes no region") and its §3 face identity (the
  boundary origin-set `FaceKey`); [ADR 0028 §4](0028-sketch-mode.md)'s nested session undo;
  [ADR 0030 §5](0030-sketch-as-entity-collection.md)'s "no solver in v1" and one-shot tangency.
- **Relates to:** [ADR 0029](0029-measurement-as-authored-quantity.md) (`Measurement` grows a
  dimension and moves to a crate), [ADR 0034](0034-curves-stay-curves.md) (the curve-native region
  this builds on), [ADR 0017](0017-csg-composition.md) (the no-operand-targeting law that cuts
  three tools), [ADR 0014](0014-substrate-crate.md) (where the continuous solver lives),
  [ADR 0022](0022-document-dump-and-state-classification.md) (what a solve writes).

## Context

The sketch layer ships five tools — Select, AddPoint, Polyline, Rectangle, ThreePointArc — against
a point-segment graph with no constraints. The target is Fusion's suite: ~25 creation tools, 12
constraints, 11 modifiers, 3 patterns, dimensions and a parameters panel. Roughly 53 commands.

Two things gate the other 51. **The solver**: the 12 constraints *are* the solver, and so are the
tangent-circle tools, every dimension that drives, and Fillet/Chamfer/Blend, which emit tangency.
**Curve–curve intersection**: Trim, Extend, Break, Fillet and the 3-point/3-tangent constructions
all need "where do these cross", and nothing in `substrate` computes it.

Below those, the existing model has three walls. A **circle has no endpoint points**, so it cannot
be an `Arc` (which references two point ids, and rejects `|sweep| >= 360°`) and cannot enter
`faces::derive`, which walks half-edges keyed on point ids. A **crossing with no shared point makes
no region**, so a circle crossing a line encloses nothing. And **snapping and constraints both want
to own a point's position** — `SketchPoint`'s sub-voxel remainder is the field a snap zeroes and a
solve writes.

Prior art splits on the second wall. SolveSpace is topological and ships a manual *Split Curves at
Intersection* whose whole job is to create the shared endpoint. Fusion computes the arrangement:
two overlapping circles are three profiles, with no constraints and no snapping ritual. Both model
a circle as centre + size with no on-curve vertex.

## Decision

### 1. Constraints own position; snapping is a birth-time assist

Snapping decides where a point is *born* and where a free drag lands. The moment a constraint
touches a point, the solver owns it, and the solution is continuous. An unconstrained snapped point
never moves, because nothing pulls it.

Snapping is **not** promoted to an implicit constraint. Auto-generating hundreds of unauthored
constraints produces an over-constrained system whose origin the author cannot see. An author who
wants lattice alignment *asserted* says so — with `Fix`, `Horizontal`/`Vertical`, or `Quantize`.

Sub-voxel sketch geometry is not a compromise: occupancy samples the exact field and quantizes at
resolve (ADR 0034). Rounding solver output to the lattice would make tangency unreachable, since a
circle tangent to two lines lands on the lattice essentially never.

### 2. The solver is two-tier: a continuous core, an integer loop above it

`substrate` gets a **pure continuous** geometric constraint solver — residuals, Jacobian, no
density and no lattice vocabulary, so it stays provable (ADR 0014) and free of domain knowledge.

The **integer outer loop** lives in `document`, where density and block pitch are known: solve
continuously, round the quantized degrees of freedom, fix them, re-solve the remainder, repeat until
stable. It converges in a few rounds or reports failure; it never hangs.

Algorithm: **DogLeg first, Levenberg–Marquardt behind it.** FreeCAD's planegcs ships three solvers
with a fallback chain and has filed cases where DogLeg fails and LM succeeds on the same sketch. The
chain is a requirement, not gold-plating.

### 3. Constraints are entities; a solve is an authored write

Constraints join points, segments and arcs in the stable-id space (ADR 0030 §1). They are
selectable, individually deletable, individually undoable, and delete-cascade reaches them when a
referenced entity dies. A side table without ids would reindex on every delete and break undo.

**Solved positions stay authored state, not `Derived`** (ADR 0022). The solver reads positions as
its initial guess and writes them back — they are both input and output. `Derived` is for what is
recomputed from nothing, and an under-constrained sketch has free degrees of freedom that only the
stored position remembers.

### 4. Reject conflicts at add; allow redundancy, flagged

Applying a constraint trial-solves. **Unsatisfiable** — refuse it, and name the constraint it
fights. **Redundant** (a solution exists but the Jacobian loses rank) — accept it and mark it,
because redundancy is sometimes the intent: symmetry asserted even though the geometry already
implies it is insurance against a later edit.

The system is therefore **always solvable**, which every downstream feature gets to assume rather
than defend against. The rank check that separates the two cases also yields the degree-of-freedom
count, so "fully constrained" is a real indicator rather than a guess.

### 5. Inference is Shift-gated and curated

Drawing infers nothing by default. **Holding Shift** during a gesture offers inference, sampled
live so candidates light up and go away as the key is held and released, committing on mouse-up.
Nothing is asserted unless the author asks — the same principle as Decision 1.

Inferable: **Tangent**, **Perpendicular**/**Parallel**, **`Quantize`** (assert the position the
snap just gave you), **Equal**/**Collinear**/**Midpoint**, and **rise:run**. Not inferable:
Horizontal/Vertical (snapping already delivers them, and asserting them silently is Decision 1's
mistake), Coincident (already free — shared point identity, not a constraint), Fix (asserting
immovability by accident is the worst failure mode), Symmetry and Curvature.

The inference tolerance is **in pixels**. It is a pick question, answered at the cursor, and it
never reaches the document — the one shape of tolerance ADR 0034 permits. In voxels it would be a
bug.

### 6. Project, Intersect and Spun Profile are cut

They define sketch geometry by another node's geometry, with a live dependency. That is exactly
ADR 0017's **no operand targeting, ever**. Honouring it as a live reference makes the fold a DAG
rather than an ordered DFS, leaks sealed scopes, and requires cycle detection — larger than the rest
of this epic combined.

A one-shot copy was rejected as worse than nothing: it *looks* associative, so the first time the
source moves and the copy does not, it reads as a bug rather than a documented limit.

### 7. A closed curve is its own loop, anchored by a centre

A circle stores a **centre point id and a radius**, with **no on-curve vertex** — as SolveSpace and
Fusion both do. `faces::derive` grows a second path: closed curves are loops immediately and skip
the half-edge walk. Ellipse and closed splines reuse it.

Splitting a circle into half-arcs, or closing an arc onto a single shared point, both invent an
on-curve point at an arbitrary angle that then appears as a draggable handle and as a seam in face
identity.

### 8. Regions come from the geometric arrangement

Every curve is cut at **every intersection with every other curve**; the pieces form a graph whose
bounded faces are the regions. A crossing needs no shared point. Two overlapping circles are three
regions.

This retires ADR 0030 §2's snap-a-point-at-the-crossing ritual, and it is what lets a circle
crossing a line close a profile. It shares the curve–curve intersection substrate that Trim, Extend,
Break and Fillet need regardless.

### 9. A region is identified by an interior sample point

The unpick identity is **one point strictly inside the face** — its deepest interior point, where
`signed_distance_to_region` is most negative. A re-derived face *is* that face when it still
contains the stored point.

Boundary-keyed identity is maximally fragile under an arrangement: decomposition is precisely what
changes when anything moves, and the three faces of two overlapping circles share one origin set.
A point in the middle of a face does not care how the boundary was cut.

It pays for itself three times with machinery that exists: `point_in_region` is the containment
test, and clicking a region and labelling it both need an interior point anyway.

Failure modes, accepted: a face that shrinks past its own sample point resets to picked (the
behaviour ADR 0030 §3 already documents for restructuring); a sample point that ends up in a
neighbouring face migrates the unpick there. `unpicked` stops being a `BTreeSet` — `f32` is not
`Ord` — and becomes a `Vec`.

### 10. Mirror and the patterns are associative and carry no freedom

A pattern stores a rule (count, spacing or angle, axis) plus source entity ids; generated entities
are `Derived` and regenerate when the source moves. **Constraints target the source only.**

The deciding argument is the solver, not convenience. Baked copies carry their own degrees of
freedom: a 12-instance pattern of a 3-DOF shape becomes 36 DOF the author never wanted to control,
and the sketch can never read fully-constrained without 33 constraints nobody wants to author.
Associative copies have **zero DOF**.

ADR 0017's law is not engaged: this is one entity deriving from another *inside a single sealed
scope* that already re-derives as a unit — no cross-node edge, no DAG in the fold. Break-link is
deferred; it would turn 12 zero-DOF entities into 36 live DOF in one click.

### 11. Both solver tiers run live during a drag

The dragged point becomes a temporary high-weight constraint pulling toward the cursor; the solver
runs; everything else redistributes; the point visibly lags when constraints will not let it follow.

The **integer loop runs during the drag too**, so a quantized point visibly steps cell to cell. The
entire reason those constraints exist is that the author cares where things land — deferring the
rounding to mouse-up hides the one thing they asked for until it is too late to adjust. Sketches are
small (tens to low hundreds of DOF), where a dense solve is well under a millisecond.

Accepted risk: a mid-drag round that oscillates at a cell boundary would show as jitter. Measure it
on a deliberately hostile sketch; the fallback is deferring only the integer tier to release.

### 12. Quantities become a crate, and carry a dimension

**`crates/parametric`** holds the quantity and expression layer: it absorbs today's
`voxel_core::units`, and depends on `substrate` only for `Rational`, keeping domain vocabulary out
of `substrate` (ADR 0014) and giving the units model somewhere to grow.

**Dimensions are an exponent pair `(length, angle)`.** Addition requires equal exponents;
multiplication adds them; division subtracts. So `wall + gap` is a Length, `wall / gap` is
dimensionless, `arc_length / radius` is an Angle — which is what a radian is — and `wall + angle`
is an error caught before it reaches the document. `voxel_density` needs no special case: it is
voxels-per-block, so Length ÷ Length, so **dimensionless**, and `3 blocks * voxel_density` types as
a Length by the ordinary rule.

**Static above, dynamic below.** Document fields are statically typed — a radius field takes a
`Length`, an angle dimension takes an `Angle`, and mixing them does not compile. The **expression
evaluator is dynamic**, on `Quantity { value, dimension }`, because `wall / gap` has a type only at
eval time. The boundary evaluates, checks the resulting exponents against the field's static type,
then stores or reports. ADR 0029's umbrella type is not reversed — it becomes the eval-layer
quantity, with static wrappers above it.

**Exactness is a storage and authoring invariant, not a solver invariant.** The expression language
is exact-rational (`+ - * /`), so the float-free rule survives everything the author writes. The
solver is floating-point by nature — Newton on transcendental residuals — and `SketchPoint` already
carries an `f32` remainder. `Quantize` is the mechanism that lands solver output back on an exactly
representable value. An inexact expression result routes through the existing
`MeasurementError::BlockTermNotWholeVoxels`, which already carries the nearest floor and ceil so the
UI can offer them rather than silently round.

An angle is exact in **two mutually inexact bases**: rational degrees (45°, 22.5°) and rational
rise:run (2:1). Converting between them is transcendental. rise:run is therefore not a nicety — it
is the representation that keeps a stair-clean slope exact where degrees cannot, since `atan(1/2)`
is 26.565051177…°, and a rounded entry gives a slope that is nearly-but-not 2:1.

### 13. Undo is one flat transient stack; the timeline is separate and persisted

ADR 0028 §4's **nested** session undo is retired. There is one flat stack, and **entering and
finishing a sketch are ordinary entries on it** — undoing past a Finish re-enters the mode and keeps
going, one operation at a time.

Nesting has a cliff: finish a sketch, do one more thing, press undo twice, and the second press
reverses an hour of work as a single step. "Undo always reverses exactly one thing I did" is worth
more than the tidiness of a group.

The persisted/transient split is unchanged and already correct — the op stack is the timeline
(ADR 0009), and the command stack is dropped on relaunch by accepted policy, because a dump replays
the scene rather than the edit history.

### 14. The voxel-native constraint set is one constraint

**`Quantize { pitch, phase }`** — this degree of freedom is a whole multiple of a pitch. On a
position coordinate it reads as *on the lattice*; on a distance, as *a whole number of blocks
thick*. `phase` carries parity: 0 is a voxel boundary, ½ a voxel centre, which is the difference
between an even-width and an odd-width mirror-symmetric shape and a whole class of off-by-one.

A quantization constraint earns its place **only where the value is an output of the solve**. If
the author knows the number, they type it. That test collapses the obvious longer list: a quantized
*dimension* is this constraint on a distance, a lattice *radius* is this constraint on a radius plus
this constraint on the centre, and parity symmetry is `phase`. A rational-slope constraint fails the
test outright — asserting 2:1 deliberately is an angle dimension, and it survives only as an
inference (Decision 5) and a representation (Decision 12).

Minimum feature size and "this arc is too small to survive quantization" are **lints, not
constraints**. They are properties to check after solving and report; as constraints they would give
the solver an objective it cannot converge on.

## Consequences

- **Text is deferred to its own epic.** It is a font subsystem — face loading, glyph outline
  extraction, hinting — and wildly out of proportion to its neighbours on the list.
- Two substrate subsystems are new: the continuous solver, and curve–curve intersection. The second
  is shared by the arrangement and by four modifiers, so it is built once.
- 16 of the 25 creation tools emit only points, segments and arcs — Line, Midpoint Line, the three
  Rectangles, the three Arcs, the three Polygons, the five Slots, Point. They are input state
  machines, not geometry. Only Ellipse and the two Splines need new `RegionEdge` variants, in the
  slot ADR 0034 already sized for Bézier.
- `Arc`'s `|sweep| < 360°` validity check relaxes for the closed case.
- Face identity, the face walk, and `unpicked`'s container all change together; unpick-survives-edit
  is shipped behaviour and needs its own tests before the arrangement lands.
- The parameters panel is the least entangled item in the epic — it touches `crates/parametric` and
  the panel and nothing else — so it lands last and blocks nothing.

## Alternatives rejected

- **Topological regions with a Split Curves command** (SolveSpace). Correct and cheaper, and it
  makes the author perform a snapping ritual at every crossing forever. Rejected knowing it breaks
  face identity and the face walk: those were architected before the feature set was defined.
- **Snapping as the constraint mechanism.** It cannot express tangency, and rounding solver output
  to the lattice makes most of the tool list unreachable.
- **A single solver that handles integers natively.** Entangles density arithmetic with the part
  worth proving, permanently.
- **Continuous-only, shipping the 12 Fusion constraints first.** Retrofitting the integer loop later
  means re-testing every constraint against it.
