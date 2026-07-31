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

**Being selectable, for a thing that draws no geometry, means being picked by its badge**
(Decision 16). The badge is the only place a constraint is on screen, so it is the hit target, and
it beats the geometry under it because it is drawn over it. Delete then reaches a constraint the
same way it reaches a segment. Two limits follow from a constraint having no position at all:

- **A transform skips it.** Translate, rotate and scale act on what has a place; moving a badge
  would change where a label sits and nothing about what is asserted. `SelectionTarget` answers
  this once, as a predicate, rather than each tool matching the variants itself.
- **A marquee sweeps it, on the point rule.** The badge is a small square mark: window takes it
  when its centre is inside the box, crossing when the box touches it. A constraint has no place
  of its own, but its badge does, and that badge is the whole of how it is on screen — so a box
  drawn around it names it as plainly as a box drawn around a vertex names that.

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

**"Unsatisfiable" is read off the residuals, never off the solver's own outcome flag.** The two
are different questions — the outcome says why the *search* stopped, the residual norm says
whether the *answer* is one — and confusing them shipped a bug that made the constraint tools look
dead (owner, 2026-07-30). The solver's residual tolerance is absolute while its step tolerance is
relative to the length of the whole parameter vector, so free geometry elsewhere in the drawing —
contributing nothing to the residual and everything to that length — makes the step test fire
first. It then reports `Stalled` with the constraint satisfied to about 1e-10 voxels, and
`Stalled` was being refused as a conflict. **Two** unrelated free points were enough to trigger
it. The trial now asks only whether the residuals are met, at the same scale a span has to close
to before the drawing calls it collapsed.

Two refusals sit alongside it, because convergence alone is not enough of a test:

**One constraint of a kind per entity set.** A literal second copy of a claim — `Horizontal` on a
segment already asserted horizontal — is refused, not flagged. Flagging is for redundancy that
carries intent, which a duplicate cannot: the two are indistinguishable, deleting either leaves the
drawing identically constrained, and two badges would stand on one anchor saying one thing. The
comparison is on kind and geometry, never on the stored value, so a second `Fix` on a fixed point
is refused whether or not it names the same place — "fix it here, and also there" is a re-fix,
which is a delete and an add.

**A converged solve that COLLAPSED geometry is unsatisfiable.** `Horizontal` and `Vertical` on one
segment have a solution: the zero-length segment, where both residuals are exactly zero. The
solver converges and reports success, and the drawing has been destroyed rather than constrained.
So the trial also asks whether any segment that had length lost it, and refuses if one did. Stated
as a property of the result rather than as a table of forbidden pairs, it covers every combination
meetable only by deleting what it names — including the ones the residual set does not have yet.

**Every refusal that has a culprit names it.** A diagnosis the author cannot act on is barely a
diagnosis: "it fights something" leaves them to find the something, and on a drawing carrying
twenty assertions that is the whole of the work. Constraints are selectable entities with badges,
so an id is all the shell needs — a refusal lands its culprits *in the selection*, lit, and Delete
is the next key rather than the next search. How the culprit is found differs by refusal, and the
difference is not incidental:

- *Already asserted* knows it directly — the check found the standing constraint.
- *Unsatisfiable* uses **leave-one-out**: re-trial with each standing constraint dropped in turn,
  and any drop that succeeds names a culprit. That is `n` solves of a system with a few dozen
  parameters, which at sketch scale is free, and it is an answer rather than an estimate. The
  alternative in the literature is a rank heuristic that blames whichever constraint appears in
  the most dependent groups, and it is known to blame the wrong one. An empty result means no
  single removal helps; saying nothing beats sending the author to delete something innocent.
- *Would collapse* is asked **structurally** — the constraints that act on the collapsing entity —
  because leave-one-out cannot answer it. An earlier solve has already moved the drawing, and
  releasing an assertion does not undo its effect, so dropping the `Horizontal` that levelled a
  segment leaves the segment level and `Vertical` still collapses it. "What else is holding this
  shape" is a question about the graph, and it always has an answer.

**Redundancy is read at the author's drawing, not at the solution.** Rank has to be read somewhere,
and the solution is the obvious place and the wrong one: rows of the Jacobian vanish at an
exactly-solved configuration, so a perfectly informative constraint can look redundant purely
because the solver did its job. This is a defect FreeCAD carries and documents (#5931). Reading
both ranks at the pre-solve drawing — a generic configuration, which is what the witness
configuration method means by a witness — avoids it. None of the four shipped residuals can vanish
this way; the point is that Tangent and Perpendicular can, and the reading is settled before they
arrive rather than after somebody reports it.

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

**The relations ship as explicit verbs first** (2026-07-30): Coincident, Parallel, Perpendicular,
Equal, Collinear and Midpoint join Horizontal/Vertical, Fix and Distance on the rail, each armed
and picked like every other constraint. Inference is the layer above them and is not built yet;
there is nothing for Shift to offer until the residuals exist, and now they do.

**Coincident is a constraint after all, not shared point identity.** The paragraph above put it in
the not-inferable list on the grounds that it is "already free" — one point instead of two. That
was the wrong call, and taking it seriously exposed why: a merge is destructive in a way the author
cannot see afterwards. The second id is gone, every segment that named it names the first instead,
and deleting the coincidence cannot put the drawing back, because there is no record of which point
was which. As an assertion it costs two residuals and deletes like anything else, and the two
points spring apart again — which is what removing a constraint should mean everywhere.

**Two residual-scaling calls worth stating.** The angle relations normalise: Parallel's residual is
the SINE of the angle between the two directions and Perpendicular's is the cosine, so both are
dimensionless and read the same on a 3-voxel segment and a 300-voxel one. Unnormalised, a long
segment's row would dominate the trust region and a short one would barely be heard. Collinear is
asked as **two distances** — how far each of the second segment's ends stands off the first's
infinite line — rather than as an angle plus an offset, so the solver never has to weigh a radian
against a voxel.

**Still unbacked:** Concentric, Tangent and Curvature, which need arcs and circles inside the
parameter vector (an arc's centre is derived from its ends and its sweep, so nothing can name it
until the sweep is a parameter too); Symmetry; and `Quantize`, which is Decision 14's integer tier.
Their glyphs are drawn on the design sheet and deliberately absent from the rail — an armable verb
that asserts nothing is worse than a cell that is not there.

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

**Shipped as a hard pin, not a weight, and the standing constraints win a fight** (2026-07-30). The
continuous tier is live: `Sketch::move_point` writes the coordinate and then re-solves with the
grabbed point held by an ephemeral `Fix` that is never stored — the hand is a constraint for exactly
as long as it is on the point. A weight was considered and dropped: with the pin the grabbed vertex
sits precisely under the cursor and the *rest* of the drawing absorbs the constraint, which is the
reading the author expects from a direct-manipulation handle. The lag the paragraph above describes
becomes a refusal instead — when the pinned system cannot be met (dragging a `Fix`ed point, or one
whose freedom is spent) the drawing is left exactly where it was and the vertex does not move at
all, rather than moving and being hauled back, which reads as the drag being broken.

Until this landed, a constraint survived only until it was tested: the level was asserted, the
author moved one of the line's own points, and the drawing tilted straight back off, because the
drag write path never re-solved. An assertion that does not hold through the next gesture is not an
assertion.

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

### 15. A constraint ARMS, then collects the entities it needs

A constraint arms like every other cell on the sketch rail: press it, then pick the geometry it is
about. It is not a verb applied to whatever happened to be selected first.

The two models differ in what the author must know **before** pressing anything. Selection-first
requires them to already hold each constraint's arity in their head and to have assembled a
matching selection; the button then either works or sits dead, and a dead button does not say what
it wanted. Arm-first lets the tool ask — one pick at a time, naming what it is waiting for, and
turning away a pick that does not fit. The knowledge moves from the author to the tool.

The typed slot is what makes that work. Each verb declares the entities it wants, in order:
Horizontal wants a line, Fix wants a point, and the relations will want a specific pair. A pick of
the wrong kind is a **refusal that leaves the gesture running** — the tool stays armed and keeps
waiting, because a mis-click is not a decision to abandon the command. The slot list IS the arity,
so adding a two-entity relation later is an entry in a table rather than a new branch in the
gesture.

Completion disarms. Once the last slot fills there is nothing left to ask, and holding the mode
open would give every constraint an explicit end the author has no reason to expect.

**A verb is not one-to-one with a constraint.** `Horizontal / Vertical` is one cell that asserts
either of two kinds, decided from the drawing — Fusion's arrangement, and the right one. The
author is saying *line this up with an axis*, and which axis is already visible in what they drew:
a line 5° off level wants to be level, and asserting plumb on it would swing it 85° and read as
the tool misfiring rather than as an instruction obeyed. So the rule is the nearer axis, with the
tie at exactly 45° going to horizontal — a coin toss resolved once and stated rather than left to
whichever comparison happened to be written, and one badge deletion away from the other answer.

This splits the cell's glyph from the badge's. The cell carries a mark of its own and the badge
carries a plain `Horizontal` or `Vertical`: **the cell asks the question, the badge reports the
answer.** A level line is marked level, not marked "level or plumb".

The cell's mark is the only one in the set that asks, and its ink says so. Under the two-tone rule
white is the reference and red is the entity that moves, so the cell draws the **two axes in white
as a corner** — each arm the exact length of the bar in `Horizontal` and `Vertical`, quoting the
two answers it stands for — with the author's own segment in **red, at exactly 45° across it**. The
angle is the point: the question is symmetric between the two axes, and a segment drawn tilted
toward either one would have answered it already. The two bars *superimposed* were tried first and
rejected — that is a plus with four nodes, which is what `Snap to voxel` already is, and it needed
its nodes shrunk below the pair's own to survive its own crossing.

**The waiting slot drives the hit-test, and does not merely check its result.** Every other click
in the sketch resolves by "the most specific thing under the cursor wins", which puts a vertex
ahead of the segments meeting at it. That is right for Select and wrong here: the vertex grab
radius is the wider of the two, so on a polyline of short edges most of a segment's length lies
inside one endpoint's circle. Resolving the general question and *then* refusing what came back
made `pick a line` refuse nearly every click on a line — the tool read as simply not working
(owner, 2026-07-30). A question that already knows the kind of answer it wants asks for that kind:
a slot waiting for a line looks only for lines, and there is no dead zone left to fall into.

**A miss reports.** A click that lands on nothing used to return in silence, on the reasoning that
it asked for nothing. But a tool that answers a miss with no sign at all cannot be told from a
tool that is broken, and that is how it was read. Every click with a constraint armed now leaves
either an assertion or a sentence.

Refusals draw **over the viewport**, in a floating box that takes the bottom-left slot the status
line otherwise holds. They lived in the top bar first, beside the passive readouts, where the
author — whose eyes are on the drawing they just clicked — never saw them. A refusal is the only
thing on screen that has to be acted on, and mode · dims · density is the least urgent; so the
urgent thing takes the established slot rather than opening a second one to compete for the same
glance. The readout returns the moment the refusal clears.

An armed constraint **overrides** the drawing tool for the duration of its gesture rather than
joining `SketchTool`'s enum. It hit-tests the same entities Select does but answers a different
question, and the two cannot run together without drawing geometry mid-assertion. It is also the
only sketch gesture that ends by itself, which no drawing tool does. Escape unwinds it on the
established two rungs: the first drops the picks and keeps the constraint armed, the second puts
the constraint down.

The gesture's picks ride in the ordinary selection, so they light up through the shipped highlight
path rather than a second one that could disagree with it. They are cleared on completion:
scaffolding for a question the author asked is not a selection they made, and leaving it lit would
aim the next Delete at geometry they only pointed at.

### 16. An assertion carries its own mark on the drawing

Every constraint draws a badge beside the geometry it names, in the constraint ink, using the same
glyph as the rail cell that made it.

This is not decoration. A solve moves the drawing until the assertion holds, after which the only
evidence of it is a line that *looks* level — and a line drawn nearly level looks exactly the same.
Without a mark there is no way to tell an asserted horizontal from a coincidence, which means no
way to predict what a later edit will and will not disturb.

The badge is anchored through the constraint's entity ids, resolved against the same projected
positions the handles and lines come from, so it cannot drift from its entity: it is placed **by**
the sketch entity graph, not merely beside it. A line's badge sits off the midpoint along the edge
normal, the one offset that reads as "about this line" at every angle; a point's sits up and to the
right, where a lock hangs in every other tool. Badges sharing an anchor step along that offset
instead of overprinting.

Glyph identity with the rail cell is the mechanism, not a saving: the mark the author pressed is
the mark they then see standing on the drawing, so the shelf teaches the notation once.

Dimensions are the exception — a `Distance` draws as a dimension gizmo, and the number is already
the mark. A glyph beside it would say the same thing twice.

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
