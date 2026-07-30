# ADR 0034 — Curves stay curves; flattening is a consumer, not a stage

- **Status:** Accepted
- **Date:** 2026-07-29
- **Relates to:** [ADR 0019](0019-the-field-layer.md) (the substrate this amends — Decision 2
  made the flattened polygon the meaning; this delta withdraws that), [ADR 0030](0030-sketch-as-entity-collection.md)
  (the region whose loops now carry edges), [ADR 0014](0014-substrate-crate.md) (where the planar
  kernel lives), [ADR 0008](0008-voxel-frame-invariant.md) (frames unaffected).

## Context

ADR 0019 Decision 2 said a profile flattens to a polygon at a fixed tolerance and **that polygon
is what the document means**. It was a reasonable simplification when every consumer was a
polygon consumer, and it made a new curve kind purely additive at the authoring layer.

It stopped being reasonable once the region became a *field*. Flattening happened at derivation
time, so everything downstream — the extrude SDF, the revolve SDF, the exact occupancy sample,
the coarse cell classifier, the GPU wash, the crease catalogue — received chords, at a tolerance
chosen in **voxels**. A viewer that knows what a voxel is worth in pixels then has no way to ask
for anything better except to pass a tolerance back *up* into the query, which is what shipped in
`7b9052b` and did not work: a finer polygon is still a polygon, the wash still read as a fan of
triangles inside its own smooth outline, and the 3D SDF silently inherited a 1/16-voxel-faceted
profile.

The general shape of the mistake: **a tolerance parameter crossing a layer boundary means the
producer flattened and the consumer is haggling about it.** The 3D side never made this mistake —
a sphere in the volume is a sphere, and the only length scale is the occupancy sample lattice.

## Decision

**A region is a field over curve primitives. The flattened polygon is one lossy view of it, never
the meaning.**

1. **`substrate::geom2d::RegionEdge`** is the region's boundary unit: a straight span, or a
   circular arc carrying its centre, radius, start bearing and signed sweep. A loop is a list of
   edges, not a list of vertices. Distance to an arc is analytic; containment splits the arc at
   its own turning points into `axis1`-monotone pieces, each obeying the identical half-open
   crossing rule a segment does, so a vertex shared between a curve and a line counts exactly
   once.

2. **`faces::derive` takes no tolerance and has no variant that does.** The two places the walk
   touched chords are gone: the vertex-fan departure angle is the edge's analytic tangent, and the
   enclosed area integrates the curve by Green's theorem (`½[r²·sweep + cx·Δy − cy·Δx]` per arc),
   which is exact rather than inscribed.

3. **Flattening is a terminal adapter.** `ProfileLoop::flatten(tolerance)` /
   `sketch::flatten_edges` are called only where something discrete is genuinely produced: a
   crease polyline, a screen-space hit-test polygon, the exact-`f64` cell classifier's polygon.
   Nothing downstream of one of those inherits the tolerance, so
   `ARC_SAGITTA_TOLERANCE_VOXELS` is a tuning knob again rather than versioned document
   semantics.

4. **The coarse classifier keeps its polygon, plus a guard.** `rectangle_inside_region` is the
   exact-`f64` predicate half (ADR 0019's width split) and its soundness rests on `orient2d` over
   straight edges, so it still takes vertices. It additionally takes `curve_bounds` — the bounds
   of every edge that was approximated — and **declines any rectangle meeting one**. The
   chord/curve discrepancy lives strictly inside those bounds, so it can never sit inside a
   claimed cell. Without the guard, a chord cutting to the void side of a *concave* arc would let
   the classifier fill a cell that is not wholly solid, which is unsound rather than merely
   coarse.

5. **Extents are measured from the curve.** `RegionEdge::bounds()` / `ProfileEdge::bounds()`
   return the arc's own reach, not its chord's. One `filled_extent` feeds both the resolve anchor
   and the authoring anchor, so a bulge can no longer be clipped by a grid sized off a chord.

6. **The WGSL wash mirrors the edge field.** No tolerance, no screen-space chord budget. It is
   also cheaper: one arc primitive per pixel replaces the twenty-odd chords that stood in for it.

7. **ADR 0030 Decision 4's algebra stands; its primitives change.** `Fill`/`Hole` tagging,
   `min`/`max` composition, and the rejection of global even-odd are all unchanged. What is no
   longer true is "each loop is a single simple polygon": a loop is a simple *edge* loop, and the
   robust pair backing it is `point_in_edge_loop` / `nearest_boundary_distance` rather than
   `point_in_polygon` / `signed_distance_to_polygon`. The polygon pair survives, private, for the
   classifier.

## Consequences

- **This changes resolved occupancy**, by at most the old sagitta (1/16 voxel) — sub-voxel, but a
  real meaning change for any arc-bounded profile. ADR 0019 Decision 3's "changing the flattening
  constant is a document migration" no longer applies, because the constant no longer decides
  anything a document means. There is no migration: the new meaning is the one the author drew.
- ADR 0019 Decision 4's soundness obligation moves. There is no longer an occupancy claim resting
  on chord tolerance, because occupancy samples the exact field. The obligation that remains is
  the classifier guard in Decision 4 above.
- Bézier segments, when they arrive, are a third `RegionEdge` variant with an analytic (or
  bounded-iteration) distance — additive in the kernel, not in a flattener.
- `Metric::Chebyshev` needed an exact arc branch (the extrude SDF measures in it). It solves by
  candidate angles — the sweep ends, the four compass extremes, and the `|gx| = |gy|` swaps —
  the same convexity-free exactness argument `distance_point_to_segment`'s Chebyshev branch uses.
  CPU-only; the wash mirrors the Euclidean branch alone.

## Alternatives rejected

- **Finer tessellation, screen-adaptive** (shipped in `7b9052b`, withdrawn here). A polygon at any
  tolerance is still a polygon, it puts a rendering concern in a query signature, and it leaves
  the resolve faceted regardless of what the viewer asks for.
- **Arc-exact in display only, chords in the resolve.** Two definitions of the same boundary, which
  is the split that produced the original bug — the outline painter was screen-adaptive while the
  wash was not, and they visibly disagreed.
