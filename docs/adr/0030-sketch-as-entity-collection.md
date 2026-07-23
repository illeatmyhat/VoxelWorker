# ADR 0030 — A sketch is an entity collection; the profile is derived from picked faces

- **Status:** Accepted
- **Date:** 2026-07-23
- **Supersedes:** the profile representation of [ADR 0028](0028-sketch-mode.md) (a sketch as a single
  hand-maintained closed polygon `Vec<SketchPoint>`). ADR 0028's mode shell, undo group, fused
  sketch-plus-operation, and snap-as-constraint-stand-in are all retained; only "the profile *is* an
  ordered vertex list" is replaced.
- **Relates to:** [ADR 0029](0029-measurement-as-authored-quantity.md) (every coordinate is a
  `Measurement`), [ADR 0017](0017-csg-composition.md) (the field algebra the flattened profile now
  reuses in 2D; the no-cross-node-operand-targeting law region-picking is checked against),
  [ADR 0022](0022-document-dump-and-state-classification.md) (faces are `Derived`; picks are
  document intent), [ADR 0019](0019-the-field-layer.md) (the flattened profile is the field's
  meaning), [ADR 0027](0027-placement-continuity.md) (the wandering-origin coordinate split a
  `SketchPoint` now carries).

## Context

ADR 0028 made a sketch a scene object you enter as a mode, but modelled its profile as one ordered
closed polygon (`Sketch { plane, profile: Vec<SketchPoint> }`). The owner reframed the authoring
model: a sketch is a **collection of first-class geometric entities** — points, line segments, arcs
(more later) — each independently add- and delete-able, and the extrudable **profile is derived**
from the closed regions those entities form (the Fusion / SolidWorks model). "Add a point anywhere,"
"delete any entity," and "draw a shape from loose segments" are not expressible over a single hand-
maintained vertex list. The shipped `Vec<SketchPoint>` (and #95's insert-on-edge / remove-vertex) is
the narrow special case of one closed loop.

## Decision

### 1. Entities are first-class with stable ids; segments/arcs reference points by id

A sketch stores `points`, `segments`, `arcs`, each carrying a **stable monotonic `EntityId`** (never
a `Vec` index — indices shift on delete and corrupt references). A `Segment` references its two
endpoint points by id; an `Arc` likewise. **Coincidence is identity, not a solved constraint** — two
segments meet because they share a point id, upholding ADR 0028's "lattice snapping stands in for a
solver." A point with no incident edge is a legal **free point**.

Entities carry a `role: Real | Construction` (default `Real`), reserved now for future **construction
lines** — reference geometry that never bounds a face. The toggle UI is deferred; the field is not.

A **sketch dimension** is reserved as a further entity kind: an annotation that makes a `Measurement`
visible on the canvas (an arc's radius, a point-to-element distance, an angle) and — with the solver —
*drives* it. **Display-only** dimensions (render the measured value) are solver-free and can ship
earlier; **driving** dimensions (edit the value → geometry moves) need the solver. Both are the UI
face of the ADR 0029 measurement substrate.

### 2. A region is a bounded face of the point-segment graph — crossings need a snapped point

Regions are the bounded faces of the **planar graph** whose nodes are points and edges are
segments/arcs — *not* the faces of the full geometric arrangement. Two segments that visually cross
without a shared point make **no region**; snapping a point at the intersection creates one
(coincidence = identity again). This keeps derivation a deterministic graph walk (DCEL
"next-edge-clockwise", ties broken by id) and avoids a continuous segment-intersection solver.

### 3. Faces default-picked; the user unpicks to carve holes; identity is the origin-set key

Every derived face is **included ("picked") by default**; the user explicitly **unpicks** faces to
carve holes. This is *not* even-odd fill — inclusion is explicit per-face state. Picking regions
*inside one's own sketch* is authoring one's own profile, **not** cross-node operand targeting, so it
does not violate ADR 0017.

Inclusion is user intent that must survive re-derivation (the topological-naming problem). A region's
identity is the **set of `origin` ids of its boundary edges**. Each segment/arc carries an `origin`
(a fresh segment's origin is itself; on split, both children inherit the parent's origin). The
document stores the **`unpicked` set of origin-set keys** — the exceptions, usually empty. On every
re-derivation a face whose origin-set matches a stored key stays unpicked; every other face is
picked. Consequences of this key:

- **Dragging a vertex** leaves segment ids and origins untouched → key unchanged → unpick preserved.
- **Inserting a vertex on a loop edge** splits a segment into two children of the same origin → the
  origin-*set* is unchanged → unpick preserved (an explicit owner requirement).
- **Adding/removing a distinct boundary edge, or merging two distinct-origin segments** changes the
  set → the face is genuinely different → resets to picked.

### 4. The flattened profile is `Fill`/`Hole` tagged loops resolved by 2D CSG — not even-odd

Flattening emits **simple boundary loops, each tagged `Fill` (from a picked face) or `Hole` (from an
unpicked pocket)**; same-classification adjacent faces dissolve their shared edge. The engine
evaluates the region as **2D field CSG** — union the `Fill` loops (`min`), subtract the `Hole` loops
(`max` against the negated field) — **reusing the field algebra ADR 0017 defines for 3D**. Each loop
is a single simple polygon (robust `point_in_polygon` / `signed_distance_to_polygon`); the
combination is an explicit boolean, not a global crossing-parity. Even-odd is rejected: its *global*
parity over a loop soup is fragile at touching loops, shared edges, and degeneracies.

**Both extrude and revolve lift this tagged-loop CSG.** Extrude maxes it with the slab; revolve
evaluates it in `(radius, axial)` — a holed revolve is a hollow vase. One `SketchSolid` may resolve
**disjoint** occupancy (two separate `Fill` faces → two prisms); the fold composes it as one body.

### 5. Arcs: one canonical representation, many creation methods as sugar

The canonical stored arc is **two endpoint points + one included-angle `Measurement` of kind
`Angle`** (ADR 0029) — unambiguous, compact, fully parametric; center and radius are derived.
Creation tools — **3-point**, **center-point**, **tangent** — all compute and store that canonical
form; their extra inputs (the through-point, the center) are consumed at creation, never persisted.
Tangency is a **one-shot at creation, not a maintained constraint** (no solver in v1); re-invoke the
tool to restore it after an edit. A future solver adds a maintained tangent constraint on top.

### 6. Faces are derived; the document is entities + picks; delete cascades on points

Faces and the flattened profile are **`Derived`** (ADR 0022) — recomputed each edit from the
entities, cached only in non-document sketch-mode session state. The **document stores only the
entities and the `unpicked` origin-set keys.** (`unpicked` references a *derived* key, which is fine:
the key is derived deterministically from document entities, and the pick is genuine intent.)

Delete is **one generic verb with integrity by construction:** deleting a **point** cascades to its
incident segments/arcs (no dangling reference ever); deleting a **segment/arc** removes only it,
leaving its endpoints as free points.

### 7. Coordinates: a `SketchPoint` mirrors a node position

A `SketchPoint` is a position, so it mirrors `NodeTransform` (ADR 0027 + ADR 0029):
`offset_voxels: [i64;2]` (integer wandering origin, voxel-at-density) + `offset_local_voxels: [f32;2]`
(sub-voxel remainder, written by `snap = None`) + `offset_measurements: Option<[Measurement;2]>`
(retained `Length` expression, `None` for a plain snapped point). `SetDensity` re-evaluates the
retained expression, so a profile does not warp under a density re-target.

## Considered options

- **Keep the single closed polygon (rejected).** Cannot express free points, standalone segments,
  arcs, or holes; "add a point anywhere / delete any entity" have no meaning over an ordered loop.
- **Even-odd fill over derived boundary loops (rejected, §4).** Global parity is fragile at
  coincidences; the explicit `Fill`/`Hole` CSG is simpler and reuses the 3D field algebra.
- **Geometric arrangement — auto-split at every crossing (rejected, §2).** Reintroduces the
  continuous intersection-robustness problems snapping-as-solver was chosen to avoid.
- **Per-region persistent ids tracked incrementally across edits (rejected, §3).** The full
  topological-naming machinery; the derived origin-set key gets the same behaviour with no id
  lifecycle.

## Consequences

- **Document schema changes structurally** (`profile: Vec<SketchPoint>` → `points`/`segments`/`arcs`/
  `unpicked`); `serde(default)` cannot absorb it. **No migration** (ADR 0028 sketches are an
  in-flight non-feature; box primitives are `SdfShape`, not `Sketch`). Load policy: **invalid objects
  are erased with a CLI warning**, never a hard failure — a segment referencing a missing point, a
  malformed entity, or an old-schema sketch is dropped + warned, and the load continues.
- **A new derivation pass** (graph → faces → picks → `Fill`/`Hole` loops), deterministic and run live
  per edit; cheap at sketch scale (tens of entities).
- **`resolve_extrude` / `resolve_revolve` generalize** from one loop to the tagged-loop 2D CSG; the
  revolve field's `signed_distance_to_polygon` becomes a multi-loop region distance.
- **The constraint-solver door is open** (ADR 0029): stable entity ids receive constraints,
  `Measurement`s receive driven dimensions, `role: Construction` receives construction lines — all
  deferred, none foreclosed.
- **New tools/verbs** (add point / add segment / add arc / delete-any / pick-unpick region) supersede
  #95's loop-specific insert/remove; the rail glyphs re-read as general verbs.
