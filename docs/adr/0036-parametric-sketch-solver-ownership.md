# ADR 0036 — Parametric owns continuous sketch solving

- **Status:** Accepted
- **Date:** 2026-07-31

## Context

The document previously contained both persisted sketch entities and the continuous residual
solver. That mixed stable ids, density-aware storage, and document repair with local floating-point
parameter layout and generic numerical mechanics.

## Decision

`parametric::sketch` owns validated local planar problems, relation residuals, rigidity and drag
policy, two-pass solving, rank, and diagnostics. It exposes local handles, a validated builder,
explicit solve purposes, and resolved solutions. Its internals use generic NLLS facilities from
`substrate`.

`document::sketch` remains the adapter and persistence owner. It sorts stable ids before building a
local problem, maps persisted constraints once, performs document-specific refusal/cascade/repair
policy, and atomically writes accepted resolved positions. Scene identity, undo intents, density,
and voxel evaluation remain outside the kernel.

## Consequences

New continuous relation types, including future tangent relations, have one solver home. The
existing serialized document representation remains unchanged. Broader sketch authoring semantics
and driven scalar authority are separate migrations.
