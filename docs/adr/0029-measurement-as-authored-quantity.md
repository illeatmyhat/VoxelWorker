# ADR 0029 — `Measurement` is the authored-quantity type; occupancy is derived and carries none

- **Status:** Accepted
- **Date:** 2026-07-23
- **Relates to:** [ADR 0003 §3f(0)](0003-shape-authoring.md) (the voxel planning unit + blocks/voxels
  measurement retention this generalizes), [ADR 0027](0027-placement-continuity.md) (the
  wandering-origin `offset_voxels` + `offset_local_voxels` split a position's value rides on),
  [ADR 0022](0022-document-dump-and-state-classification.md) (the authored/derived state boundary
  this sharpens for quantities), and [ADR 0030](0030-sketch-as-entity-collection.md) (the first
  consumer to store *every* coordinate as a `Measurement`).

## Context

Today two document fields already retain an authored blocks+voxels expression alongside their
canonical voxel value — `NodeTransform::offset_measurements` (position) and `SdfShape::size_measurements`
(size) — so a density re-target re-evaluates losslessly (ADR 0003 §3f(0)). Everything else that
denotes a dimension is a raw number: extrude `height_voxels: u32`, revolve `turn_degrees: u32`,
tube `wall_blocks`, and — until ADR 0030 — sketch vertices.

Grilling the sketch entity model surfaced the underlying shape: a retained `Measurement` is not a
density-retarget cache, it is a **parametric expression**. Storing one per authored dimension is
what a future constraint solver and driven-dimension / expression system (Fusion-style parameters,
edge-length formulas) hang off. The owner's endgame explicitly includes that solver, construction
lines, and constraints on points and segments — so the data model should stop foreclosing it.

## Decision

### 1. `Measurement` is the umbrella type for every authored quantity, carrying a `kind`

A `Measurement` is an authored value carrying its parametric expression and a **kind**:
`Length { blocks, voxels }` and `Angle { degrees }` today, more later. It is **one type with a
kind**, not a family of sibling types — a position *is* three `Length` measurements from the origin,
a size is three, an extrude height is one, a revolve turn is one `Angle`. The canonical voxel/degree
number is an *evaluation* of the measurement; the expression is the truth. An expression evaluates
*to* a typed measurement, which is the substrate a solver drives.

### 2. Every authored dimension is a `Measurement`; the boundary is authored-vs-derived

**Authored intent carries measurements; resolved occupancy does not.** The voxels a field produces
are `Derived` (ADR 0022) — reconstructible, never authored, no measurement. The line is *authored
intent vs. resolved occupancy*, **not** geometry vs. number: a future **Array / pattern feature is
authoring**, so its count, spacing, and angle *are* measurements even though it emits geometry. Use
a `Measurement` for every authored length and angle; leave derived voxels raw.

### 3. Retention is optional per field — `None` for a plain literal

Mirroring `NodeTransform::offset_measurements`, a field stores `Option<Measurement>` (or an array of
them): `None` when the value is an ordinary literal with no parametric expression (the common case,
kept cheap), `Some` only when authored parametrically. A 30-vertex drawn sketch carries zero
`Measurement`s until dimensions are expressed on it.

## Considered options

- **`Measurement` stays "length"; add a sibling `Angle` type, unify later behind a trait (rejected).**
  More types, weaker unification, and it fights the "a position *is* a measurement" intuition. One
  umbrella with a `kind` is the cleaner substrate for an expression system.
- **Keep dimensions as raw numbers, add parameters later (rejected).** The parametric layer would
  then need a second document migration across every dimensional field. Retaining the expression now
  (optional, `None` by default) is nearly free and forecloses nothing.

## Consequences

- **The `Length`/`Angle` kinds are additive.** `Length` exists; `Angle` is added when the first angle
  goes parametric (revolve turn, arc bulge — ADR 0030). Counts (`voxels_per_block`) and enums
  (`RevolveAxis`) are not measurements and stay raw.
- **This is applied opportunistically, not as a sweep.** Each length/angle field becomes a
  `Measurement` as it is next touched; there is no repo-wide conversion pass.
- **The door to a constraint solver is now explicit, not accidental.** Construction lines, and
  constraints on points/segments, build on this expression substrate + stable entity ids (ADR 0030);
  they are deferred, but the model no longer forecloses them. This *amends* ADR 0028's "lattice
  snapping stands in for a solver; no constraint entities ever" to a **v1 stand-in, not a permanent
  no**.
