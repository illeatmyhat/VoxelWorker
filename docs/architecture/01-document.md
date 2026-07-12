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
- a **group** — a named composition of children;
- a **definition / instance** pair — a reusable sub-assembly and its placements, so a
  repeated part is stored once and placed many times.

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
- Materials are addressed **per block** (one texture spans a block face), matching the
  game's own texturing model. Voxels carve shape out of a block; they do not carry
  independent surface materials.

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

Two things persist, and they are not the same thing:

- **The document** — the scene graph and its operations. This is the user's work; its
  format is versioned and treated with the care of a file format.
- **Preferences** — camera, window, panel state, last-used paths. These are
  conveniences; they carry no design intent and no compatibility promise. A stale
  preference is deleted, not migrated.

The discipline is to never let a convenience leak into the document format, and never
to let document data hide in preferences.
