# ADR 0028 — Sketch mode: a sketch is a scene object you enter, editing real entities in a sealed scope

- **Status:** Accepted
- **Date:** 2026-07-22
- **Relates to:** [ADR 0003 §3i](0003-shape-authoring.md) (the sketch→volume authoring atom this makes
  interactive), [ADR 0017](0017-csg-composition.md) (the ordered fold + sealed composition scope +
  the **no-operand-targeting** owner law this decision upholds), [ADR 0022](0022-document-dump-and-classified-state.md)
  (mode + undo state are non-document, classified as view/derived), [ADR 0018](0018-viewer-modes.md)
  (sketch mode is an *editing* mode, a different axis from the exclusive *viewer* render modes),
  [ADR 0019](0019-the-field-layer.md) (the flattened profile is the field's meaning), and
  [ADR 0027](0027-placement-continuity.md) (its position-snap `{None | Voxel | Block}` is reused for
  profile vertices).

## Context — the engine exists, the authoring does not

The sketch→volume engine is built: `SketchSolid { sketch, operation }` resolves an **Extrude** or a
**Revolve** of a 2D profile, with a field and a coarse bound, byte-identical to the primitives it
subsumes (a rectangle-extrude *is* a box, a rectangle-revolve *is* a cylinder). What is missing is all
of the **interactive authoring** on top: today a sketch's profile is only reachable through the
inspector (a rectangle's spans) or a hand-built `SetSketch`. There is no way to *draw* or *edit* a
profile directly. The value proposition — organic, complex shapes — lives entirely in that missing
authoring layer.

The owner's framing (grilled 2026-07-22): **a sketch is a scene object, and entering it is a mode.**
Entering swaps the left rail to sketch tools and disables operations irrelevant to sketching (so a 3D
op cannot be applied mid-sketch). Fusion 360 is the reference — editing a sketch is a distinct
environment with its own toolset; sketch entities are real, persistent geometry; the sketch is one
object in the timeline; features that reference its profile update on *Finish Sketch*.

## Decision

### 1. A sketch stays FUSED with its operation — no standalone-sketch + referencing-feature split

Fusion models a Sketch as a free-standing object that an Extrude **references**. That is **operand
targeting by reference**, which ADR 0017 forbids as a hard owner law (*"placement order — never
operand selection — decides what it touches… NO operand targeting ever"*). So the sketch scene object
**is** the `SketchSolid` node: the profile and its lifting operation live together, and the operation
lifts **its own** profile. The main-scene ordered fold sees one producer and never grows a cross-node
reference. The cost — one profile feeds exactly one operation, no reuse across ops — is accepted; the
value proposition is organic shapes, not parametric reuse.

### 2. Editing a sketch is a MODE with a sealed, self-contained scope

Entering a sketch is a mode: the left **rail swaps** to sketch tools, non-sketch **operations
disable**, and an unmistakable **editing indicator** frames the viewport (the Fusion-blue-environment
role, in the current design language). Inside is a **self-contained scope** — ADR 0017 already defines
a composition scope as "a Group or a definition body"; a sketch's interior is one more. The mode is
entered on a sketch node and exited by **Finish** (commit) or **Cancel** (discard).

The mode itself is **view/editing state**, never document state (ADR 0022) — like a viewer mode, it
follows what you are editing and is not saved in the shared file.

### 3. Sketch entities are REAL, directly-manipulated objects — the Profile → Flattened-profile seam

Inside the scope the author manipulates **real entities** — points, segments, rectangles, later
arcs/Béziers — not previews. This is the glossary's **Profile** ("the authored 2D outline… control
points, kept exact; what the author manipulates"). It **flattens** to the polygon the producer
resolves — the glossary's **Flattened profile** ("the profile reduced to a polygon… *this is the
profile's meaning, not an approximation of it*"). So a new curve kind is additive at the authoring
layer and invisible below it, and Fusion's "extrude consumes the profile" becomes "the operation lifts
the flattened profile of its own fused sketch" — consumption without a cross-node reference (§1).

Today only the flattened polygon exists (`Sketch { profile: Vec<SketchPoint> }`); the real-entity
Profile layer is new work this epic builds, flattening onto that polygon.

### 4. The mode owns an UNDO GROUP — live real edits, one atomic main-history entry

Direct manipulation on real objects would otherwise flood the flat command stack: 30 placed points and
their nudges = 30 top-level undo steps, and undoing "the sketch" becomes 30 undos. The fix is the
mechanism the owner's "transient, self-contained history" already named — an **undo group**:

- The mode **opens a group** on enter. Each in-mode edit is a **real, live intent** (the document
  updates, so the resolved volume gives live feedback), but the command routes into the **open group**
  instead of onto the top-level stack.
- **In-mode** undo/redo works *within* the group (fine-grained — undo the last point or drag).
- **Finish** closes the group as **one entry** on the main stack (undo-past-the-sketch = one step —
  Fusion's "sketch is one timeline node with contained editing").
- **Cancel** = undo the whole open group back to enter-state, then discard it. The discardable session
  for free, **no second stack** to keep coherent.
- Orthogonally: a **continuous drag coalesces to one command per gesture** (commit on mouse-up, not per
  frame), killing per-frame spam independent of grouping.

Undo history is itself non-document (ADR 0022), so grouping lives entirely inside that non-document
machinery.

### 5. Position snap is reused; lattice snapping stands in for a constraint solver

Profile vertices reuse ADR 0027's **position snap** `{None | Voxel | Block}` — the same quantization,
now in the plane's two in-plane axes. Following the glossary's **Lattice snapping**, the voxel grid
**stands in for a constraint solver**: snapping to grid, vertices, and axes delivers axis-alignment,
equal lengths, and coincidence as a by-product of quantization. The sketch layer therefore carries
**no constraint entities, no solver**, and none of the over-constrained / flipped-solution states a
solver brings. There are deliberately **no constraint tools** on the rail.

### 6. Minimal ghosts — real objects, not previews

There is **no volume ghost** (a translucent preview of the extruded solid as you draw) — noisy, and
the profile is the thing being manipulated, not the volume. The only ghost is the **pre-first-point
plane affordance** ("a point will land *here* on the plane"). The segment drawn between placed points
is a **real entity**, not a rubber-band preview.

## Considered options

- **Fusion-style split: a standalone Sketch that Extrude/Revolve reference (rejected).** The intuitive
  reading of "a sketch is a scene object," and how Fusion works — but it is operand targeting by
  reference, which ADR 0017 forbids. The sealed-scope + fused model (§1–§3) gives the same authoring
  feel with no cross-node reference.
- **A separate transient undo stack committing atomically on Finish (rejected).** Matches the
  "transient history" wording literally, but a parallel history system is machinery to keep coherent.
  An **open group** on the existing stack (§4) delivers the same in-mode-fine / main-coarse / Cancel
  behaviour with one concept and one stack.
- **All edits live on the flat main stack, no grouping (rejected).** Simplest, but the undo explosion
  the owner flagged: undoing a sketch means undoing every vertex one at a time.
- **A live volume ghost preview while drawing (rejected).** Redundant with editing the real profile,
  and noisy; direct manipulation on the real entities is the clearer feedback (§6).

## Consequences

- **New mode machinery:** enter/exit a sketch scope, a rail-swap, an ops-disable gate, and an editing
  indicator. The mode is view state (ADR 0022), not in the document.
- **New undo grouping:** the command stack gains "there is an open group; route commands into it until
  closed." Small and contained — no parallel history.
- **A new Profile entity layer** (points/segments/rectangles/…, later arcs/Béziers) flattening to the
  existing `Vec<SketchPoint>` polygon. New gizmos, cursors, and rail icons are logged in
  `docs/design/gizmos-and-cursors.md` (Sketch-mode section).
- **The `Sweep` operation arm** remains the reserved next producer op (unchanged by this ADR).

## Rollout — tracer-bullet slices (each feel-testable)

1. **Mode shell** — enter/exit on an existing sketch node; rail swap; non-sketch ops disabled; editing
   indicator. No editing yet.
2. **Vertex direct-editing** — move/add/delete profile vertices as real entities, position-snapped,
   live volume update, wrapped in the undo group + Finish/Cancel. *(First shippable slice folds 1+2:
   a mode that can only be entered is not feel-testable; vertex-drag is the smallest useful action.)*
3. **Rectangle + polyline tools** — drag a rect → 4-point profile; click-to-place polyline.
4. **Create-from-scratch** — pick a plane, start an empty sketch, draw.
5. **Arcs / circles**, then on-surface sketching, then primitives-reframed-as-sugar.
