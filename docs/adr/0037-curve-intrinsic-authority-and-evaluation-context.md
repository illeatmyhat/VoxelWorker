# ADR 0037 — Curve-intrinsic authority and evaluation context

- **Status:** Accepted
- **Date:** 2026-08-01

## Context

An arc sweep and a circle radius are intrinsic curve parameters: they belong to the curve, not to
one of its points. A solver may write a free value, while a fixed value is an authored
`Measurement` that must be resolved using the scene's voxel density. Persisting a second,
resolved radius beside that source would make density changes and undo observe stale geometry.

The profile region feeds bounds, voxel resolution, field sampling, handles, feature edges, and
render previews. Resolving a fixed source independently in each of those paths risks disagreement
and resolving it per sample makes a scalar lookup part of a hot loop.

## Decision

`CurveParameter<Free, Fixed>` records exactly one authority. `ArcSweep` is density-free; a
`CircleRadius` is either a solver-writable exact `ResolvedLength` or a fixed `Measurement`.
Fixed sources carry no cached voxel value.

Every curve-sensitive document operation takes `parametric::EvaluationContext`, constructed at
the density-bearing scene/producer boundary. The context is part of the region memo snapshot. A
memo miss resolves every circle once, builds the arrangement and its measurement-width field
edges, then hands that immutable derived view to bounds, resolve, field, coarse classification,
and edge generation. Dense field consumers call the field preparation seam once per operation;
their per-voxel and per-sample loops borrow the resolved curves only.

`SetDensity` rescales free `ResolvedLength` values by the exact `new / old` rational ratio; fixed
measurements remain untouched and resolve at the new context. Load repair receives the same
context and drops a circle whose resolved radius is invalid, together with dangling constraints.

## Consequences

There is no context-free curve-evaluation door that can silently omit a fixed circle. Direct API
callers construct an evaluation context explicitly. A fixed curve remains geometry, but it adds no
solver degree of freedom. Tangency remains a later relation migration; this decision adds no
Tangent constraint or UI.
