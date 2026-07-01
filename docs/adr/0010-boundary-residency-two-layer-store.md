# ADR 0010 — Boundary-aware residency: the two-layer (coarse + microblock) chunk store, one evaluator serving display + exact sinks

- **Status:** Accepted (E5 landed 2026-06-30 — the two-layer path is the SOLE runtime display path; the dense
  `resolve_region` / dense `VoxelGrid` fallback are retired to a test-only parity/golden oracle)
- **Date:** 2026-06-30
- **Layer:** PRODUCTION PORT of [ADR 0009](0009-op-stack-truth-evaluator-and-sinks.md) §3–§4 (boundary-culling
  evaluation + boundary-aware chunk residency) into the real chunk store. **Refines [ADR 0003](0003-foundation-rework.md)
  Phase D** (chunk residency becomes boundary-aware; this ADR is where Phase D §3a/§4 actually land) and **sequences on
  ADR 0003 §3a** (the chunk-local-integer + categorical block-palette payload, a prerequisite). It builds the **CPU,
  exact display + query/export sinks**; the GPU display *brick-field* sink ([ADR 0007](0007-gpu-view-resolve.md)
  generalized) is the NEXT port, not this one. No new product model — it implements representation ADR 0009 ruled.

## Context

ADR 0009 ruled the architecture (op-stack is truth; one evaluator → many sinks; display = cached brick field) and left
the production port as "engineering, not architecture." This ADR scopes that port for the **CPU exact seam**: the chunk
store today (`store.rs`) resolves **every covering chunk into a dense `VoxelGrid` occupied-voxel list** — it materializes
every interior voxel of a solid, which is exactly the waste ADR 0009 exists to remove (an 800×800 revolve densifies its
solid interior and blows `MAX_GRID_VOXELS` = 6M).

**Vintage Story's own storage is the reference model** (researched 2026-06-30, confirming "coarse until chiseled" is a
literal storage fact, not a slogan): a chunk is a **32³ block-ID array** (palette/protobuf-compressed) — a solid region,
even multi-material, is **just block ids with no voxel data** — plus a **sparse map of position → `BlockEntityMicroBlock`**
for chiseled blocks. Each microblock stores a small `BlockIds[]` material palette and its geometry **already
greedy-decomposed into cuboids** (one bit-packed `uint` each: min/max X,Y,Z in 0–15 + material index), with **per-face
solidity flags** (`sideAlmostSolid`/`sidecenterSolid`) for neighbour face culling. This dissolved an earlier mis-framing
(that a "full" chunk must be single-material): VS keeps material **per block** in the coarse array, so multi-material
solids need no voxels and no single-material constraint.

## Decisions (grilled with the product owner, 2026-06-30)

**1. The chunk is two layers, modelled on VS (CONTEXT.md "Boundary residency").** Per chunk:
- a **coarse layer** — a per-**block** block-ID grid (`CHUNK_BLOCKS³` ids, palette/RLE-compressed); a solid interior is
  block ids only, **no voxel data**, multi-material-free.
- a **microblock layer** — a **sparse** map of boundary blocks (blocks the producer surface passes through) → their
  sub-block geometry **stored already as cuboids** (our `cuboid::VoxelBox` list *is* VS's packed-cuboid form), never a
  dense 16³ grid.
- **per-face seam-solidity flags** per boundary block — the coarse-vs-microblock analogue of the dense-fog **apron**.

**2. One evaluator; conservative per-op interval bound classifies each block.** `VoxelProducer` gains a conservative
**cell field-interval bound** (`Option<(min,max)>` over a block cell): all-outside ⇒ **air**, all-inside ⇒ **coarse-solid**,
straddling ⇒ **boundary** (per-voxel field eval). SDF shapes bound analytically (**1-Lipschitz**: over a cell
`f ∈ [f(center)−r, f(center)+r]`, `r` = cell circumradius — never misclassifies); sketch via profile bbox; sculpt deltas
are exact and **any block a delta touches is forced to boundary**. **CSG composes by interval arithmetic** (union = min of
fields, subtract = `max(dA,−dB)`, intersect = `max(dA,dB)`). An op that **cannot bound** a cell (DebugClouds fBm noise)
returns `None` ⇒ that block is treated as boundary and resolved per-voxel — **still exact, just unelided**. The bound being
conservative makes classification **occupancy-identical to brute force** on the exact seam (ADR 0009 §4).

**3. The store is ONLY the display sink's cache; the exact sinks read the evaluator directly.** There is one evaluator
(decision 2); the sinks differ by policy (ADR 0009 §2):
- **Display sink** — runs the evaluator, **caches** the two-layer result + seam flags (the boundary-aware `store.rs`),
  incremental on edit.
- **Export / query / golden sinks** — run the **same evaluator region-scoped, cacheless, streaming**: a coarse-solid block
  is a fast `d³` fill (export) or an analytic run contribution (query: `run += d`, no expansion); a boundary block is
  per-voxel field eval. They **never read the elided display cache**. The monolithic dense `resolve_region` is **retired**
  (no vestigial dense path, no "expand the cache" seam) — this is why the `.vox` 6M-cap bug dissolves on the export path too.

**4. The mesher consumes the two layers; the apron generalizes to seam-solidity flags.** A coarse-solid block → a **one-box
fast path** (one `VoxelBox` spanning the block, no dense decompose); a boundary block → its stored cuboids (like VS
`GenShape`); inter-block/inter-chunk seam faces are culled via the **per-face seam-solidity flags** instead of a densified
apron. The **visible exposed-face set is identical** to today's dense mesher (the parity gate, below).

**5. Land the ADR 0003 §3a payload FIRST; the new layers are born integer.** Before the two-layer path, land §3a: the
per-voxel payload becomes **chunk-local integer coords + a categorical block-palette cell** (the absolute i64 lives only in
the chunk key), and `GRID_OVERLAY_BIT` moves to a **per-draw uniform** (§3c). The new coarse/microblock layers are then
built **chunk-local-integer from day one**; the legacy `Voxel { world_position: [f32;3], material_id }` path is **retired,
not migrated**. Gated by far-scene goldens first (ADR 0003 Phase D0).

**6. Coexist behind a capability with dense fallback; retire the dense path last.** The new evaluator/two-layer path engages
for the producers + scenes it supports; the dense `VoxelGrid` path stays as **fallback** (exactly how the ADR 0007 GPU fog
coexists with CPU fallback — unboundable clouds already fall back). Every commit stays green; **goldens cross-check
new-vs-old** each slice; the dense path is retired only once the new path covers everything.

## The exact-seam parity gate (non-negotiable, mirrors the existing nets)

A **cull-vs-brute-force parity test** mirroring `cache_region_matches_monolithic_*` (store.rs) + the cuboid apron parity +
the `gpu_parity` A/B net: for every gated scene, **(a)** the evaluator's streamed exact occupancy (coarse fast-fill +
boundary per-voxel) equals today's dense `resolve_region` occupied set bit-for-bit, and **(b)** the two-layer mesher's
exposed-face set equals the dense mesher's, with goldens pixel-identical. Boundary-culling is thereby a **pure
optimization** on the data seam, never an observable change.

## Slice plan (each independently green-gated; verification per the session gate)

- **D0 — far-scene goldens.** Guard the payload move with goldens at XZ~10k before touching the payload (ADR 0003 Phase D0).
- **D1 — §3a payload.** Chunk-local integer + categorical block-palette cell; `GRID_OVERLAY_BIT` → per-draw uniform.
  Coexist; near-scene goldens unchanged; far-scene goldens (D0) now exact.
- **E1 — interval-bound primitive.** `cell_interval` on every op + a **standalone exactness parity** (the bound never
  misclassifies a cell vs brute force), wired to nothing yet.
- **E2 — two-layer store + classifier.** The evaluator builds coarse + microblock + seam flags; behind the capability flag,
  dense fallback. Parity (a): streamed exact occupancy == dense `resolve_region`.
- **E3 — two-layer mesher.** One-box coarse, cuboid microblock, seam-flag culling. Parity (b): exposed-face set == today;
  goldens pixel-identical.
- **E4 — repoint export/query to the cacheless evaluator.** Coarse fast-fill / analytic run; boundary per-voxel. Parity:
  `.vox` + diameter == today. **Dissolves the 6M cap on the export path.**
- **E5 — retire (LANDED 2026-06-30).** The two-layer path is now the **sole runtime display path**: the live
  display cache is the `TwoLayerResidentCache` (chunk-granular incremental, #54), the shell meshes through
  `new_from_two_layer_chunks`, the fog grid is streamed from the evaluator (`expand_resident_chunks_into_grid` /
  `resolve_region_two_layer`), and export / diameter stream cacheless from the evaluator (E4) with the dense
  fallbacks removed. The runtime capability flag + the dense `Store` runtime branches are gone. The monolithic
  `Scene::resolve_region` / `Store::resolve_region` are **kept as the test-only parity + golden REFERENCE ORACLE**
  (the `cache_region_matches_*` / `*_matches_dense` nets, `gpu_parity`'s CPU reference, and `shot`'s dense golden
  cross-check all still bind them) — they are simply off every runtime path. An unboundable producer keeps the
  evaluator's per-voxel boundary path (not the old dense densify).

## Consequences

- The 800×800-revolve / `.vox` 6M-cap pattern is dissolved on **both** display and export (no dense interior is ever built).
- The live display cache is the boundary-aware `TwoLayerResidentCache`; the dense `Store` / `resolve_region` are
  **retired from every runtime path** (kept only as the test parity/golden oracle — see E5). ADR 0006's "resolved into
  the chunked `VoxelGrid` … not negotiable" wording is **amended** exactly as ADR 0009 §Consequences already foresaw —
  the read seam is an occupancy query / evaluator call (streaming exact sinks + a boundary-aware display cache), not a
  resident dense grid. **One whole-region `VoxelGrid` remains on the display path**: the onion-fog densify still
  consumes a `VoxelGrid`, so the evaluator STREAMS it (coarse fast-fill + boundary per-voxel) rather than caching a
  dense interior — this is not the retired dense `resolve_region`, and the fog will drop it when the GPU brick-field
  display sink (below) lands.
- The **GPU brick-field display sink** (ADR 0007 generalized: 8³ bricks, clip-map LOD) is the **next** port and gets its own
  ADR — it consumes this evaluator's boundary set; the shipped fog atlas is already a brick map, so it is a short step.
- **Incremental edit** reuses today's chunk-granular `invalidate_aabb` first (re-evaluate a dirty chunk's blocks);
  block-granular dirty-brick recompute (the prior-art incremental path) is a later optimization, not slice 1.
- **Rotated baked-voxel parts stay deferred** (ADR 0009: the brick/coarse lattice is world-axis-aligned; a rotated sculpted
  part would staircase on the shared lattice → the lossy-resample path, not this fast path).

## Considered alternatives (rejected)

- **Display brick-field first** (port the GPU fog atlas into the brick-field sink before the CPU exact seam). Fastest visible
  win and closest to shipped code, but defers the exact data seam every exporter/query/golden binds to and its correctness
  gate. Rejected: build the exact CPU seam first, then generalize it to the GPU display sink.
- **Single-material "full" chunk** (full = one solid bit + a majority material). Re-introduces the exact multi-material
  problem VS avoids by keeping material per block; rejected in favour of the coarse block-ID grid.
- **Expand the elided cache for exact sinks** (export/query read the display cache and re-materialize interiors). Keeps a
  dense `resolve_region`-shaped path alive and re-grows the 6M pattern on export; rejected in favour of cacheless evaluator
  reads.
- **Corner-sample block classification** (8 corners + center). Cheap and uniform but **not conservative** — a thin feature
  inside a cell is missed, violating the exact-on-data-seam rule. Rejected for the exact seam (it is an allowed *display*
  approximation only).
