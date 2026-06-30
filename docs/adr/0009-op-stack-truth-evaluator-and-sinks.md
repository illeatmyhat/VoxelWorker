# ADR 0009 — Operation-stack is truth; one boundary evaluator, many sinks; drop the resident dense grid

- **Status:** Proposed
- **Date:** 2026-06-29
- **Layer:** BOUNDARY RULING + foundation refinement. **Extends and sharpens [ADR 0006](0006-authoring-truth-and-gpu-boundary.md)**
  (the CPU/GPU authoring-truth boundary), **generalizes [ADR 0007](0007-gpu-view-resolve.md)** (the GPU fog resolver is
  the first display *sink*), and **refines [ADR 0003](0003-foundation-rework.md) Phase D** (chunk residency becomes
  boundary-aware). It does not introduce a new product model — it ratifies the representation the foundation already
  implied and removes a structure (the resident dense voxel grid) that was scaffolding, not truth.

## Context

A performance investigation (per-edit fog latency) bottomed out in densifying solids into a dense `VoxelGrid`; an
800×800 revolve blows the `MAX_GRID_VOXELS` = 6M cap (`SketchSolid::resolve` silently empties over the cap). The
product owner reframed the problem: **we materialize every interior voxel of a solid that was never chiseled — the
opposite of the reference model.** Vintage Story keeps a block coarse until chiseled, only then giving it a sparse 16³
microblock representation; nothing densifies a world to voxels.

A deep-research pass (Media Molecule's Dreams, ESVO/SVDAG/NanoVDB, NAADF, binary greedy meshing, Aokana) confirmed the
direction at production scale: Dreams stores sculptures as an ordered list of 1–100k SDF edits (the op-stack *is* truth),
hit our exact dense-grid wall, and solved it with a hierarchical boundary-culling evaluator that evaluates the field only
near the surface (>99% culling). Two throwaway spikes (`experimental/sdf_voxel_torus/`) then retired the open risks:
direct-SDF + voxel-DDA quantization produces a **crisp blocky VS-block look** (toruses at 40/80/800 voxels; N=800 renders
in ~5–7 ms with O(1) memory — the size that blew the cap), and **sparse sculpt deltas composite into the implicit
occupancy test per-cell** (random added/removed voxels render as crisp bumps/pits at ~6.4 ms, no dense grid).

## Decisions (ruled with the product owner)

**1. The operation stack is the single source of truth.** SDF primitives + boolean CSG (authored from 2D sketches) +
sparse hand-sculpted voxel deltas, ordered. The **resolved voxel grid is a derived, on-demand cache, never truth, never
resident as a whole-scene dense buffer** — any code that materializes it densely must justify why a query against the
op-stack will not do. This *extends* ADR 0006 (whose truth was already `apply(overlay, evaluate(tree))`); it only demotes
the grid from "resident" to "lazily derived." ADR 0006's invariant (GPU never truth; CPU op-stack is) is untouched.

**2. One evaluator kernel, many sinks differentiated by policy.** Everything downstream is a *derivation* of the op-stack
through one evaluator, not a family of bespoke pipelines. Sinks differ only by policy (region, exactness, engine,
lifetime):
- **Display sink** — visible region, approximate (LOD / skin / aliasing OK), **GPU allowed**, transient.
- **Exporter sink** — bounded + complete region, **exact + deterministic**, CPU (or GPU + A/B), persistent file. A
  *pluggable family* over the region-occupancy query (generalizes ADR 0003 §S6c's `bound_region_occupied` →
  `VoxExport::from_region_voxels`). **`.vox` is demoted to one constrained backend** (256³/255-colour limit — tile or
  refuse beyond it); it does not shape the architecture.
- **Query sink** — point/region, exact, CPU, transient. Measurement/diameter readouts and the ADR 0004/0005
  agent-authoring + analysis stack bind here (they were always occupancy *query* consumers, not dense-grid iterators).
- **Project persistence is separate and simpler:** serialize the **op-stack itself** (the versioned document) — not a
  voxel export. The honest unifying word is *derivation/sink*, not *serializer*; display is a sibling derivation, not a
  serializer.

**3. Boundary-culling evaluation; boundary-aware chunk residency.** The evaluator evaluates the op-stack **only near the
surface** (Dreams hierarchical culling: coarse cells → shortened per-cell edit-lists → refine only boundary cells). The
chunk store classifies each covering chunk **empty / full / boundary** (cheap coarse op-stack eval) and **resolves +
stores only BOUNDARY chunks**; full and empty chunks carry an analytic flag, not voxels. This extends ADR 0003 Phase D
(store unification) with a classification step and generalizes the ADR 0007 covering-vs-non-empty distinction to a
boundary set.

**4. On the data seam, culling is occupancy-IDENTICAL to brute force; the display seam may approximate.** For exporters /
query / goldens (ADR 0006 + the golden spine demand bit-exact occupancy), boundary-culling is a **pure optimization**:
every op exposes a **conservative cell-interval bound** (min/max of its field over a cell) so empty/full classification
can never misclassify; an op that cannot bound a cell **falls back to per-voxel evaluation** (still exact). Enforced by a
**cull-vs-brute-force parity test** (mirroring `chunked_resolve_matches_monolithic_*` + the `gpu_parity` A/B net). On the
display seam, LOD / GPU-float / skin-only approximation is explicitly allowed.

**5. Two evaluator implementations, never one shared GPU evaluator.** CPU-deterministic (serves data + goldens) and GPU
(serves display, A/B-checked against the CPU one exactly as ADR 0007 established). The op-stack is the shared *input*;
boundary-culling is the shared *algorithm*; the implementations stay separate to preserve ADR 0006 determinism.

**6. Sculpt deltas are edits in the same evaluation.** Added = union, removed = subtract, composited per-cell with the
SDF/CSG result (`occupied = (sdf ≤ 0 && !removed) || added`). Deltas are stored **sparsely** (sorted-key list or hash
grid — O(#deltas), never a dense grid) and never materialized. The editable delta store stays edit-friendly; the static
compaction structures (ESVO / SVDAG / NanoVDB) are export/display-cache targets only, not the live store.

**7. Block vs voxel.** A **block** is the coarse placement + material/texture unit (one block-texture per face; coarse
until chiseled). A **voxel** is the chisel granularity (`voxels_per_block` per axis, VS = 16³; ADR 0003 document
density). Geometry/occupancy is per-voxel; texture/material is per-block (a brick spans a whole block face).

## Consequences

- Retire the monolithic `resolve_region` whole-grid assembly (ADR 0007 already flagged this); the per-chunk store stays
  and becomes boundary-aware. ADR 0006's "resolved into the chunked `VoxelGrid` … not negotiable" wording is **amended**:
  the read seam is an **occupancy query**, served lazily/region-scoped, not a resident dense buffer.
- The GPU fog resolver (ADR 0007) is reframed as the **first display sink**; the new evaluator generalizes it.
- The `.vox` 6M-cap "disappears" bug is dissolved (no dense interior is ever built); `.vox` becomes a constrained
  exporter behind the exporter-sink interface.
- Headless/agents/CI unaffected — they bind the CPU query sink (cleaner than iterating a dense grid).

## Prior art (external validation)

A solo-dev SDF game engine (Mike's engine, YouTube `il-TXbn5iMA`, 2026) independently builds this architecture — scene
as an ordered list of SDF edits (truth), boundary-only residency, dynamic real-time booleans against many objects, even
physics. It **rejects both pure analytic raymarch (the per-pixel "wall") and mesh (too slow to regenerate for a dynamic
world)** in favor of a **cached sparse brick field**: evaluate the op-stack once per edit into 8³ bricks (boundary cells
only, ~1 byte/distance, allocated from a texture atlas) and per-frame raymarch the *cache*, not the field. It uses a
**BVH (AABB tree) of edits** (shared CPU/GPU) for the broadphase + dirty-brick incremental recompute, and **geometry
clip-maps** (nested camera-centered grids) for LOD (2.5 km draw distance: 200 trillion dense cells → 20 million). Its
design rule — *optimize for recompute, not render, because the world is dynamic* — is our exact situation. **Our ADR-0007
GPU fog per-chunk atlas is already a brick map** (boundary-residency occupancy tiles in a 3D R8 texture atlas), so this
"third technique" is a short step from shipped code, not greenfield.

## Open (validation + deferred sub-decisions)

- **Object-count scaling (the running experiment):** how many objects each display technique handles on- and off-screen
  at a reasonable framerate (dynamic ramp to 20 FPS @ 4K). The benchmark races **three** display sinks (below); it is the
  first real prototype of the §3 evaluator + broadphase. Gating unknown for the display path.
- **Display path — DECIDED: C, the cached sparse brick field + clip-map LOD** (benchmarked RTX 4090, 4K, 20 FPS;
  `experimental/sdf_bench`). The gating experiment ran: adding a clip-map occupancy pyramid (coarse "any-brick-inside"
  levels + a hierarchical DDA that jumps empty space) lifted C's scattered ceiling **160 → 10240 (~64×)**, exactly as the
  prior art predicted; packed unaffected. Final scattered: **C 10240 @ 46 MB / 57 µs-edit vs A-mesh ≥16384 @ 185 MB /
  183 µs-edit** (B analytic raymarch ≈2.5–4k — the wall). C still trails mesh's *raw* scattered ceiling (the rasterizer
  gets frustum/Z cull free; C had only two LOD levels — a 3rd/4th closes most of the gap), but **C wins the currencies
  that matter for an edit-heavy, large, mostly-off-screen world: ~4× less memory, ~3× lower edit latency, render cost
  independent of scene complexity** — and it generalizes the shipped fog atlas (ADR 0007). Measure-don't-assume
  satisfied: the clip-map fix was *measured*, not assumed. **Remaining engineering** (not architecture): more LOD
  levels, GPU-side incremental atlas updates, production port into the chunk store. **Settled sub-result:**
  coarse-until-chiseled buys ~3× on the dominant cost (mesh instances / brick count) — not densifying plain blocks is a
  real win.
- **Broadphase + LOD:** uniform macrocell grid vs **BVH/AABB-tree** (prior art); **geometry clip-maps** for the
  >10k-block anisotropic + off-screen case (ties to ADR 0003's streaming/eviction).
- **Editable sparse delta structure:** sorted-key list (works now) vs hash grid / HashDAG as delta counts grow.
- **Minification aliasing** of the per-block texture at high N (mips/supersample) — display polish, not architectural.
