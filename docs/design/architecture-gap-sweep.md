# What a sweep of real buildings said is missing

A sweep across archetypes — castles, temples, mills, arcades, ruins — asked what each would
need that the system cannot express. The headline was that **nothing broke the model**: the
document-is-a-program spine, the two-scale block/voxel seam, and the ordered fold all held
under every archetype. Everything the sweep surfaced is an *addition*.

Most of what it surfaced belongs to the agent-authoring kit, which sits above the intent door
and is out of the core's scope. What follows is only the residue that bears on layers this
system actually has, and that is still open.

## 1. Repetition has no representation

The single highest-leverage missing producer. Radial and path repetition of one sub-geometry —
eaves, arch rings, colonnades, apses, arcades, sails — can be authored today only by placing
each copy by hand, so the document holds N unrelated nodes with no record that they are one
family. Change the count and the work is manual; change the source and nothing follows.

What it wants is a **pattern producer**: one that voxelizes N copies of a sub-geometry at
arbitrary angles *from the field*, so nothing is resampled, while still exposing a stable
member index so a later edit can address one member. Two properties in one — a producer that
composes like a producer but is addressable like a set — which is what makes it the hard one
and also what makes it worth building once rather than three times.

Downstream of it: per-member suppression that survives a count change, a source that is itself
a sub-assembly, and repetition-with-variation.

## 2. Reflection is not a rotation

A parametric producer rotates continuously, so arbitrary angles are answered. Reflection is
not: a left-hand wing cannot be the mirror of a right-hand one. For a *baked body* this
matters and is cheap to fix — a reflection is an exact permutation of voxels, lossless and
byte-stable, so it is a one-bit widening of the stored orientation rather than the lossy
resample path. Bilateral symmetry is the most common symmetry in the subject matter and it is
currently outside the transform entirely.

The caveat is chirality: mirroring asymmetric ornament reverses it, which is sometimes wanted
and sometimes a defect. Nothing distinguishes the two yet.

## 3. Block state has no schema

A material is a block identifier plus an opaque attribute payload, with no typed schema and no
rule for how the payload transforms when the geometry rotates. Most of the target game's
interesting blocks are stateful — stair facing, log axis, door hinge, fence connection, slab
half — so a rotated assembly keeps stale facings and export is lossy by construction: a
functional gatehouse leaves as dumb stone.

What it needs: a typed per-identifier state schema, a stated algebra for how state composes
with orientation (facing must rotate with the geometry), and a neighbor-resolution pass for
the connective cases. Adjacent and equally undefined: a **world-origin export contract** — the
document's floating origin is decoupled from any world-meaningful origin, so nothing pins
which world coordinate a build anchors to, or asserts that voxel detail is in phase with the
destination's own sub-block grid.

## 4. Shared reference geometry, and instances that can differ

Two related absences, both cheap now and expensive after more is built on the current shape.

**Datums.** There is no named reference geometry that many nodes attach to — a level plane, a
grid line, a work axis — so "move this level up and everything on it follows" has no
expression. Relations are node-to-node only.

**Per-instance parameters.** A linked instance carries a transform and nothing else, so every
size variant of an otherwise identical part forces a separate definition. A parameter-override
bag on the instance, or a named type tier on the definition, would remove the fork.

## What the sweep did not look at

Recorded because the omissions were structural, not incidental: visibility and sightline
fields (which is what defensive architecture is actually designed around); merge, diffing and
provenance for a shared document; and the failure modes of the authoring loop itself —
contradictory constraints, unsatisfiable relations, requests for parts that do not exist.
