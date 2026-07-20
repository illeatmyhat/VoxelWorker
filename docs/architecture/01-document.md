# 01 — The Document

The document is the thing the user owns: their design, complete, editable, and
serializable. It is a **program, not a bitmap** — an ordered composition of authoring
operations whose evaluation *produces* voxels, rather than a store of voxels themselves.
This one decision shapes everything downstream: it is why undo is exact, why files stay
small at any scene size, why density can change after the fact, and why every cache in
the system is allowed to be ruthless about being a cache.

## The scene graph

A document is a tree of **nodes**. A node is one of:

- a **leaf producer** — an operation that can answer "is this point solid, and with what
  material?" for any point in its local frame;
- a **part** — a named composition of children, and the unit an assembly is made of;
- a **definition / instance** pair — a reusable part and its placements, so a repeated
  part is stored once and placed many times.

**The part is the assembly container, and the scene root is itself a part.** Primitives,
cutters and operators are a part's ingredients, never assembly citizens in their own
right. Making the root a concrete node rather than an implicit container means
whole-scene operations are expressed by selecting a thing, never by the absence of a
selection — a background misclick cannot silently retarget an operation from an object
to the world.

Every node carries a placement (an integer offset on the voxel lattice, plus a lattice
orientation), a material choice, and per-node display toggles (grids, visibility). The
graph is small — proportional to the user's design decisions, never to the voxels those
decisions imply.

**Reuse is lossless by construction.** An instance places its definition at an integer
lattice offset in one of the twenty-four axis-aligned orientations — the proper
rotations of the cube, the full symmetry the block lattice admits. Every such
orientation is an exact permutation of voxel indices, so an instance is
voxel-for-voxel its definition: rotating a carved gargoyle can never resample,
blur, or drift it. Transforms outside that group are not placements; they are
authoring operations (a producer's own parameters), where losslessness is defined
by the producer, not by the lattice.

## Composition

The tree composes by an **ordered fold**. Within a scope, children evaluate in document
order, each folding into the accumulated result under its combine operation — union,
subtract, intersect, or emboss. A boolean affects everything accumulated before it in its
scope, and nothing else; **placement order, never operand selection, decides what it
touches**.
There is no per-operation targeting, no "objects to cut" list, no feature scope. Geometry
is protected from a cut by being placed after it, or beside it in a sibling scope. The
tree therefore serves two masters — organization and boolean semantics — and that is a
deliberate trade: the structure the user reads is the structure that composes.

Union is the constructive operation and carries material (later writers win on overlap).
Subtract and intersect are **occupancy-only masks**: subtract clears cells, intersect
keeps only cells present on both sides, and neither ever stamps material — surviving
cells keep what they had. A cutter is not a special node kind; it is an ordinary part
placed under a subtract operation, reusable through the same definition/instance
machinery as any other part.

**Emboss moves a surface rather than adding or removing one.** Within its cutter's
footprint it raises or recesses whatever the fold has accumulated so far, by a signed
amount. It is a fold arm and not sugar over the other three, because the accumulated
body appears on both sides of its definition: a node that *referenced* the accumulated
result would be operand targeting wearing a different hat. Emboss stays lawful for the
same reason subtract does — it reads "everything accumulated before me in this scope"
and nothing else. The surface it moves is therefore whatever the ops before it produced;
reorder the node and the relief changes.

**Parts and definition bodies are sealed composition scopes.** A scope resolves its
children into one body, then folds that body into its parent under the scope node's own
operation. A boolean inside a scope can never affect geometry outside it — the seal is
what makes a subtree a *thing* rather than a region of influence, and it is why a
definition's internal cuts are fully spent before an instance places the finished body.
Every part is sealed; there is no unsealed flavour of container, and no setting that
makes one transparent. The consequence worth stating plainly is that a parent sees one
body where a part stands: cross-scope operand targeting is not forbidden by a rule, it
is unsayable, because the members have no name outside their scope.

One declaration pierces one level of that sealing: a definition may be flagged a
**fixture**. A fixture does not pre-compose; its children splice into the hosting
scope's fold at the instance's position, in order, under the instance's transform — so a
window definition (an opening cutter, then a frame) cuts its host wall and fills the
frame with a single placement. The host is **positional**: whatever accumulated before
the instance in its scope. It is never a stored reference, so there is no rehosting, no
orphaned voids, no host-tracking lifecycle — move the fixture into another wall's scope
and it cuts that wall. A fixture instance's own combine operation is inert (its children
carry their own), and the seal of every scope above the host remains absolute.

Where structures like walls meet at corners, the junction is **a part built to suit the
situation** — authored and placed like any other part, additive or fixture — never a
patch stored against the composed result. Site-specificity comes from being a distinct
part instanced at that spot, not from a world-frame override layer; no such layer exists.

## Producers

A leaf producer is the atom of geometry. Producers are **parametric and analytic**: a
box, a cylinder, a sphere-like solid are stored as their parameters; a drawn shape is
stored as its 2D sketch plus the operation that gives it volume (extrusion along an
axis, revolution around one). The sketch-to-volume path is the canonical authoring
motion — primitives are conveniences that could be expressed as sketches, not a separate
ontology.

Two properties are demanded of every producer:

1. **Point-exact evaluation.** Given a voxel center, the producer answers
   solid-or-empty deterministically. This is the ground truth every cache is measured
   against.
2. **A conservative interval bound.** Given an axis-aligned box, the producer must be
   able to answer "certainly all solid", "certainly all empty", or "cannot say" —
   erring only toward "cannot say". This is what lets evaluation skip the interior of
   large solids without ever being wrong (see [Evaluation](02-evaluation.md)).

A producer that cannot bound a region is still correct — it merely pays per-voxel
evaluation there. Boundedness is a performance contract, exactness is a correctness
contract, and the two are never traded against each other.

A drawn profile is a closed path of lines, arcs and curves in continuous coordinates,
never required to meet the lattice. It flattens to a polygon at a **fixed tolerance of
1/256 block**, and *that polygon is what the document means* — not an approximation of a
truer meaning hiding in the control points. The tolerance is deliberately
density-independent, which is what lets a density change re-voxelize without moving
geometry; because the polygon is the meaning, the flattening rule is versioned document
semantics rather than an implementation detail. The lifts are extrusion and revolution,
with sweep along a path reserved as the third. Control points survive as live editable
input, so parametric editability is kept without adopting a spline representation the
lattice would quantize away regardless.

A producer may be **unbounded** — a half-space is the simplest field there is, and it
replaces a whole trimming tool with "plane, subtracted". Unboundedness is legal only
where the accumulated result bounds the outcome: subtract and intersect yield results
contained in the accumulator, emboss a finite dilation of it. **Union is the one
operation that would be genuinely infinite, and an unbounded producer under it is
rejected.** Where a producer is unbounded, the region an edit dirties is computed from
the accumulator's bounds rather than the producer's.

## The field — what a node means

Beneath the fold sits a layer worth naming: a node's meaning is a **signed scalar field**,
negative inside the body. Composition is field algebra — union is a minimum, subtract a
maximum against negation, intersect a maximum — and occupancy is what you get by
classifying that field over cells. The pipeline reads **intent → field → occupancy →
display**, and the field is where new *geometric* affordances attach, exactly as intent
is where new *structural* ones do.

**Predicates classify; fields measure.** Both persist and neither replaces the other. An
exact predicate answers "is this point inside?" and owns occupancy; a field answers "how
far from the boundary?" and owns geometry. The distinction is not pedantry — it is load
bearing twice over. A field is often only conservative where a predicate is exact, so
replacing the predicate with the field would surrender real interior elision. And a
measurement is not entitled to decide occupancy on a measure-zero set: a sample landing
exactly on a boundary has distance zero, where only the sign carries the verdict.

A field also has a **metric**, and the metric follows the lift rather than the authored
edges. An extrusion is the product of a profile region with a slab, and the maximum-norm
of a product is the maximum of its factors — so an extruded polygon has an exact
square-metric field. A revolution introduces circular cross-sections, whose square-metric
distance has no closed form, so revolves and the curved primitives measure round. This
split follows a distinction the author chose and can predict; splitting instead on edge
angles would mean rotating one edge by a degree could silently change how a body dilates.

**Outset is a field combinator carried by any node** — a leaf, a part, or an instance —
and it attaches beside the node's combine operation rather than replacing it. It needs no
new machinery, because the fold yields a field and shifting a field is meaningful at every
level: a composed cutter can be given clearance as a whole without editing its internals.
Outset is a Measurement, not an integer voxel count. A negative outset insets, which is how
a deliberate gap between chiseled pieces is authored. **A part's metric is the weakest of
its members'** — a part mixing a box and a revolve dilates round — which is predictable but
is emphatically not something the interface may hide.

Not every producer has a field, and that is a property of the type rather than a runtime
surprise. A producer without one cannot be outset or embossed, and the system says so by
construction rather than discovering it later. Fabricating a field for a producer that
has none is precisely the mistake that sentinels once made here.

## Blocks, voxels, density

The world is measured in **blocks** — the game's coarse placement and material unit —
and each block subdivides into **voxels**, the chisel granularity. The subdivision
factor (`voxels_per_block`, the document **density**) is a document-level setting:
density is *fineness*, not *size*. A wall is authored "12 blocks long"; how many voxels
that is falls out of the density.

Density is **bounded: 1 through 64**. The bound is not a limitation of ambition but a
structural invariant: with at most 64 voxels along a block edge, one *row* of voxels
inside a block always fits in a single machine word (and, at 32 and below, in one word
native to the GPU). Occupancy structures throughout the system are entitled to assume
this — a row is a word, a row test is one bitwise operation, a row count is one
popcount — and that entitlement is worth more than unbounded fineness would be.

Consequently:

- Geometry parameters are canonically stored in voxels, but the **measurement the user
  typed is retained as authored** — "3 blocks" stays "3 blocks", and a later density
  change re-evaluates it rather than freezing a stale voxel count.
- **Materials are per voxel, and paid for only where they vary.** A voxel carries a cell
  key — a palette id plus an overlay bit — so a single block may hold several materials.
  Storage follows the boundary philosophy: a region of uniform material keeps one
  identity for the whole region, and only *mixed* regions pay per-voxel cost. Texturing
  stays block-face-anchored, which is what keeps it faithful to the host game: a voxel
  samples its material's texture at the position that voxel covers on the block face, so
  a carved surface reads as the block it was carved from rather than as a sticker.

## One door: the Intent

Every mutation of the document is expressed as an **Intent** — a small, serializable
description of one edit ("set this node's shape", "offset that node", "change density").
The panel UI produces intents; a transform gizmo produces intents; an agent produces
intents. One dispatcher applies them.

The payoff for this austerity:

- **Undo/redo is a single mechanism.** Applying an intent records its inverse on a
  command stack; undo is applying the inverse. There is no widget-specific undo code.
- **The document is replayable.** A sequence of intents is a test fixture, a repro
  case, and a collaboration substrate, for free.
- **Authority is unambiguous, and single-writer.** However many hands can edit —
  human, brush, agent — they contend at one door, and the door admits one at a time:
  an editing hand holds *presence* over the document, and another acquires it only
  when the first releases. This is turn-taking, not merge-based collaboration —
  conflicts are prevented at the door rather than resolved after the fact, and the
  command stack stays a single unambiguous history.

## Persistence

Every field of application state is **classified**, and a field that classifies itself as
nothing does not compile. There are four destinations, and only the first is a routing
decision — the rest all reach the debug dump, and are distinct because they answer
different questions at the field, where whoever adds the next field will be reading:

- **Document** — the scene graph and its operations. This is the user's work; its format
  is versioned and treated with the care of a file format.
- **Settings** — what the user *chose* and would want in every project: the projection,
  the window size, the Home view they pressed a button to keep.
- **Session** — how the workspace was *left*: the viewer mode, the folded panels, the
  diagnostic overlays. The browser's bargain — close it, open it, and your tabs come
  back. Restored across relaunch, never inside a shared file.
- **View** — where the author was *looking from*: the camera pose, the layer band.

The last three carry no design intent and no compatibility promise; a stale one is
deleted, not migrated. The discipline is to never let any of them leak into the document
format, and never to let document data hide in one of them.

Two artifacts consume those categories. **The document** carries what the model is.
**The dump** is the superset — every category — because its defining property is that a
scene must be completely reproducible from it; it needs no versioning, being read by the
build that wrote it. Both the exit save and the F9 repro write a dump, since restoring a
session needs the scene *and* the preferences *and* the camera pose.

The guarantee has two halves, deliberately not one mechanism. **Classification** is
recorded at the field, in review-visible form. **Completeness** comes from exhaustive
destructuring: every capture binds every field with no rest pattern, so adding a field
stops the build until somebody says where it goes. A category alone would classify a
type and say nothing about whether each field made the trip — which is exactly how a
camera pan target once went missing from a repro while sitting inside a camera that was
already "captured".
