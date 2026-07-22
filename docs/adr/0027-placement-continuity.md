# ADR 0027 — Placement continuity: a continuous rotation and float position, with snap as quantization

- **Status:** Accepted
- **Date:** 2026-07-21
- **Supersedes:** [ADR 0026](0026-placement-orientation-on-the-transform.md) — its *discrete*
  `LatticeOrientation` is subsumed into a continuous rotation (the 24 axis-aligned turns are just the
  rotations that land on the exact path). ADR 0026's *home* decision stands: orientation lives on the
  `NodeTransform`, not the shape.
- **Relates to:** [ADR 0008](0008-voxel-frame-invariant.md) (the integer voxel frame is reopened —
  the *field* position becomes float, the *voxel* position stays integer, both carrying a wandering
  origin), [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) (the continuous transform is CPU/Intent
  truth; the GPU ghost is a display mirror of it), [ADR 0010](0010-boundary-residency-two-layer-store.md)
  (the classifier is the single application point, now an inverse *affine* rather than a permutation),
  [ADR 0015](0015-graphics-math-crates.md) (the new CPU surface-raycast lives in `crates/raycast`, and
  the GPU ghost sphere-trace binds to it), and [ADR 0019](0019-the-field-layer.md) (the surface normal
  is read from the composed field, degrading to a producer SDF, then to voxel occupancy).

## Context — a discrete turn cannot lie a tube along a cylinder

ADR 0026 wired the entered face's normal into the node's orientation as one of the 24 axis-aligned
cube rotations. That is exact for a **flat** axis-aligned face — a voxel face normal is an exact ±axis
— and wrong for a **curved** one. A tube laid on a cylinder's side must tilt to the *continuous radial
normal*; the 24 lattice turns can only approximate it to the nearest axis, so the tube reads as
"vertical / wrong" on the lower half of the curve. The same discreteness caps position at integer
voxels, so the armed ghost cannot slide sub-voxel.

The owner's ruling: **the field's position and orientation, and the resulting SDF, should be
completely continuous. Voxel-granular snapping is the default *quantization on top*, not the
representation.** This is the "rotation (with voxel resampling)" that ADR 0026 §Considered-options and
ADR 0001 decision 3 explicitly deferred behind the reserved word *rotation*. It is now the job.

## Decision

### 1. The transform carries a continuous rotation and a float position

`NodeTransform` gains a real **rotation** (`Quat`/`Mat3`, any angle) and its offset becomes a **float**.
`LatticeOrientation` is deleted from the transform; the 24 axis-aligned turns are simply the rotations
that hit the exact path (§4). Both reach the versioned document — this is the versioned state ADR 0026
deferred, chosen deliberately, `#[serde(default)]` = identity + zero so old documents load unchanged.

**Field position is float; voxel position is integer.** The continuous authoring position lives on the
transform as a float; the resolved occupancy it quantizes to stays integer-indexed. Both sides carry a
**wandering origin** for far-range precision — ADR 0008's i64 rebase is the integer side; the float side
rebases its local origin the same way. The wandering origin ships **now**, not deferred: a float offset
without it silently loses precision far from origin, and a deferred precision guard is a forgotten one.

### 2. Snap is a quantization over a surface-seated placement

A node dropped on a surface is **seated**: it contacts the surface with a consistent normal, so the
contact point and the rotation are two views of **one** degree of freedom (where the contact slides on
the surface). Snapping one re-solves the other.

- **Position snap** `{ None | Voxel(default) | Block }` quantizes the contact point.
- **Angle snap** quantizes the rotation, **each angular DOF independently** (tilt, azimuth, twist), to
  **15° increments** (24 per turn — the pleasant set 0/30/45/60/90 is its subset). "Upright" is the
  degenerate case: ignore the surface, align to world-Z.
- **Precedence is position-dominant.** When position is snapped the contact is fixed and the rotation
  takes the nearest reachable snapped angle; when position is `None` the angle snap **slides the
  contact** along the surface to where that angle occurs. The more-constrained axis wins; the freer one
  follows to keep the node seated.

### 3. The surface normal degrades gracefully; seams pick the dominant producer

The seated normal is read from the **most continuous surface available** at the hit: the composed
**field**'s gradient, else a producer's **SDF** normal, else the resolved **voxel** occupancy gradient.
At a **boolean seam** (where two producers meet and the composed gradient is discontinuous) the normal
is that of the **dominant producer** — the arg of the fold's min/max at the contact — never a blend.
The built-in **world planes** are never seated: they stand the node world-vertical (identity rotation).

### 4. Continuity does not break correctness — a rotation is an isometry

The classifier's absolute→producer-local map (`two_layer_store/classify.rs`, ADR 0026 §3) becomes an
inverse **affine** (rotate + translate) rather than a signed axis permutation. Because a rotation
preserves distances:

- **Per-voxel classification stays occupancy-identical to brute force.** The field's Lipschitz bound is
  untouched, so a point evaluated through the exact inverse affine classifies exactly. Continuity moves
  *nothing* about the final occupancy contract.
- **Only the block-cell interval bound loosens.** An absolute axis-aligned block cell maps to a
  *rotated* box in producer-local. Bounded by the **isometry rule** — a center sample plus the cell
  radius, the radius being rotation-invariant — the coarse-solid / air classification stays tight.
  Worst case it loosens and more blocks fall to exact per-voxel evaluation: **a performance cost, never
  a correctness one.**
- **The default path is byte-identical.** A voxel-snapped, axis-aligned placement resolves through an
  inverse affine that is exact integer arithmetic in float — the same sample coordinates as today — so
  the snapped-placement goldens stay green for free. Only genuinely off-axis / sub-voxel placements
  resample.

The two-layer store is the single source the mesh *and* the GPU brick raymarch read, so both display
paths inherit continuity with no further work. The parity twins in `sdf_shape.rs` and the WGSL probe do
not move — the producer stays unoriented; only the classifier's frame map changes.

### 5. One CPU surface-raycast, sliding on the composed SDF

The load-bearing new component: there is no CPU SDF raycast today, only a point-eval `signed_distance`
and a GPU sphere-trace. Continuous placement needs the **contact point + normal** and the **snap
slide**. It lives in `crates/raycast` (wgpu-free, ADR 0015), takes a field-eval closure, marches the
**composed** field (so placing on a boolean/Part result works), and the **GPU ghost sphere-trace
re-points at it** so ghost and committed geometry cannot diverge.

**One mechanism does everything: sliding on the composed SDF via gradient + damped Newton**
(`p -= field(p)·∇field / |∇field|²`). It serves the hit, the normal (central difference), the
contact projection, the voxel-snap re-projection, *and* the angle→position slide:

- On the primitives it is exactly as accurate as a closed-form inversion — for a true distance field the
  gradient *is* the unit normal, so Newton converges to the same point.
- Sliding on the **composed** field subsumes the "is the slid contact still exposed?" check for free:
  Newton converges to the exposed composed surface and physically cannot land in a carved-away region
  (the field is positive there). No per-producer analytic inversion + validity check to maintain — one
  predicate, no twin to drift.
- A non-true-distance field (L∞ boxes, post-outset/emboss) keeps a correct normal *direction*; the
  `/|∇|²` form absorbs the non-unit magnitude, and the 15° snap quantizes the result anyway, so accuracy
  is ample. A badly non-Lipschitz field (heavy displacement) is handled by **damped, iteration-capped**
  Newton.

Analytic per-primitive slides are added only *if* profiling ever shows the iteration is hot — it will
not be, at one contact-solve per drag frame (`[[measure-before-rejecting]]`).

## Considered options

- **Keep `LatticeOrientation` as a stored exact fast path (`enum { Lattice | Continuous }`) (rejected).**
  Hedges against the resample interval bound being too loose — but the isometry argument means the
  continuous path is *already* exact and the discrete path buys only interval *tightness*, which the
  isometry bound recovers without a second representation. Two orientation types and two classify paths
  to keep in sync forever, to avoid a cost we would rather measure than pre-optimize.
- **A single `[f64;3]` world offset instead of float + wandering origin (rejected).** Simpler to write,
  but collapses the carried integer frame ADR 0008 keeps and reintroduces the far-range downcast loss —
  precisely the (c)-re-derive failure ADR 0008 forbids.
- **Per-primitive analytic angle→position inversion, general composed fields deferred (rejected).**
  The free-position + angle-snap + boolean-surface case is a *main* use, not a rare one, so a deferral
  that excludes booleans is unacceptable. Sliding on the composed SDF covers it in v1 with less code.
- **Curated angle set {0,30,45,60,90} instead of uniform 15° (rejected).** A curated set's product
  across the multi-dimensional DOFs is lumpy; a regular 15° lattice composes cleanly per axis, matches
  the "24 turns", and contains the curated favorites as a subset.

## Consequences

- `NodeTransform` grows a `Quat` rotation and a float offset — **versioned document state**, a real
  migration surface, entered deliberately. The wandering-origin float precision guard ships with it.
- The classifier's frame map generalizes from a permutation to an inverse affine with an isometry
  interval bound. Occupancy is unchanged for every existing golden; performance may soften on heavily
  off-axis scenes (measure before optimizing the bound).
- `crates/raycast` gains a composed-SDF marcher + damped-Newton surface slide; the ghost WGSL
  (`placement_ghost.wgsl`) binds its `orientation_inverse` uniform to an arbitrary rotation and its
  trace to the same spec. Every *finished* display path (mesh, brick raymarch) needs no shader change.
- The glossary's reserved word **rotation** is now realized; **orientation** as a discrete term retires
  with `LatticeOrientation`.
- **Deferred:** placing on a **voxel/sculpt** surface with angle-snap slide (no analytic producer) falls
  back to the gradient walk or snap-orientation-keep-contact — acceptable because a sculpt surface has
  no crisp normal to snap to. Ancestor composition of a non-identity rotation remains guarded off (ADR
  0026 §4), landing with the general orient-any-node gizmo.

## Amendment — 2026-07-22 (owner rulings during implementation)

Three points diverge from or complete the decision text above; recorded here so §2 is not read
literally.

- **Precedence is combined-error, not position-dominant.** §2's "position-dominant" is superseded: when
  both position and angle are snapped the seat **minimizes the combined position + angle error**
  (a rim-weighted joint solve, `solve_seated_15deg`), not a strict precedence. The freer axis no longer
  automatically wins.
- **Seat and snap read the SDF, sampled corner-safe — never the rendered voxel geometry.** The seated
  normal is the composed field's gradient (§3), but sampled at the **entered face's interior**, not at
  the raw contact: at a box corner the raw gradient is the 45° diagonal, so the DDA-picked face chooses
  *where* to sample (its face normal) while the value stays the SDF's. A corner drop seats flush to the
  entered face — the face you approached disambiguates the three that meet there. (Voxel/Block *position*
  snap, by contrast, is a natural fit for the voxel surface — an open follow-up, not yet wired.)
- **NoSnap + Deg15 keeps the sub-voxel position (was "future work").** A flat face's normal is already a
  15° multiple, so quantizing it is a no-op; the angle snap must not move the NoSnap contact off the
  cursor. Shipped 2026-07-22 (`085a30a`): under NoSnap the drop seats at the continuous cursor contact
  and quantizes the corner-safe normal to 15°; the snapped-position joint solve is unchanged.
