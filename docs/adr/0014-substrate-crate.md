# ADR 0014 — The substrate crate: pure CS/math components split out of the domain, compile-enforced

- **Status:** **Accepted (2026-07-13), extraction in progress** — see
  `docs/design/substrate-extraction-map.md` for the dated inventory and slice order.
- **Date:** 2026-07-13
- **Layer:** repo shape / code organisation. No behaviour change; every slice is gated on the
  full baseline and moves the component's own oracles with it.

## Context

The owner's ruling: the most complex data structures should live as **discrete, named library
components** — easy to identify, read, and reason about in isolation (including performance) —
with a logical separation between *objects of computer science and pure math* (BVH, AABB,
bit-packed occupancy cubes, interval arithmetic, min-mip pyramids, allocators, key codecs,
rationals, supersede protocols — the Dreams-style machinery) and *objects of domain* (scenes,
producers, chunks, bricks-as-blocks). Today those structures are baked into their most relevant
domain files (`brick_field.rs`, `two_layer_store.rs`, `spatial_index.rs`, …), so their
algorithmic identity is discoverable only by reading domain code.

## Decision

1. **A cargo workspace with one internal library crate, `crates/substrate`.** Not intended for
   release; intended for reading. The app crate depends on substrate; substrate depends on no
   domain code — the dependency direction is **compile-enforced**, which is the reason a crate
   was chosen over an in-crate module cluster (a convention the compiler doesn't check would
   erode).
2. **The boundary law** (also in `CONTEXT.md` under **substrate**): a component belongs in the
   crate iff it is describable entirely in textbook CS/math vocabulary and is parameterized by
   plain numbers/generics — never by domain types. Domain adapters (chunk traversals, resolve
   plumbing, wgpu upload) stay in the app crate.
3. **Textbook naming inside the crate** (`MedianSplitBvh`, `BitCube`, `DisjointRunList`,
   `ExactRational`, …); domain vocabulary survives only at the adapter seams. Each component's
   module doc **cites the canonical scientific literature** for the structure and notes the
   local variant's deviation (anchors listed in the extraction map) — part of the
   definition of done for every extraction slice.
4. **Criterion microbenches for hot components only** (`crates/substrate/benches/`), run on
   demand — never part of the commit gates.
5. **Feature-free substrate**: no `gpu`/`oracle`/`tracy` features in the crate; std (+ rayon
   where a moved component already uses it).
6. **Machine-checked construction follows the extraction** (map item 10b): Kani harnesses for
   the finite bit/index kernels (the density bound 1..=64 doubles as the verification bound),
   Creusot/Verus for stateful invariants (run list, free list, supersede), a Lean model for the
   interval-conservatism and pyramid-superset theorems — sequenced after tiers 1–2 land, never
   blocking extraction, never replacing the oracle gates.

## Considered options

- **In-crate module cluster** (`src/substrate/`): same readability, no workspace churn —
  rejected because the no-domain-imports rule would be convention-only and a future edit could
  quietly reach back into domain types.
- **Gradual promotion** (module layer now, crate later): rejected as doing the migration twice.
- **Multiple narrow crates** (geometry / numeric / concurrency): rejected as taxonomy ahead of
  need; one crate, module-per-component, split only if a real pressure appears.

## Consequences

- Gates run at the workspace level; substrate's tests join the counted baseline as they move.
- The extraction map's restraint list is part of the decision: domain-shaped code
  (`classify_chunk_block`, the widest-run sweep body, `cuboid_mesh.rs`, `LeafSpatialIndex`)
  is deliberately NOT lifted — only their pure kernels.
- The ADR 0013 material-atlas epic builds on substrate primitives (`BitCube`, the
  free-list/cube-packing kit) instead of growing new private copies inside `brick_field.rs`;
  tiers 1–2 of the extraction are therefore sequenced before that epic's implementation.
- Map item 10b (machine-check the pure kernel) gets its natural target surface: the kernels to
  verify are, after this, the substrate components themselves.
