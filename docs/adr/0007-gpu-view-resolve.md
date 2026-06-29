# ADR 0007 — GPU view-resolve: stream the compact tree, voxelize/mesh/fog on the GPU for display

- **Status:** Proposed
- **Date:** 2026-06-29
- **Extends:** [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) — names the "GPU view-resolve
  (mesh/fog generated on the GPU for display)" as a legitimate, *gated* display derivation; this ADR
  is its concrete contract. Builds on [ADR 0003](0003-foundation-rework.md) (the compact
  representation — producer tree + sculpt overrides + the chunk-windowed `resolve_into` seam) and
  [ADR 0002](0002-engine-streaming-meshing.md) (chunking, origin-rebased coordinates). Realises the
  GPU half of issue **#41**.
- **Does NOT supersede:** ADR 0006's authority rulings stand in full. The CPU stays the source of
  truth; this ADR only specifies the *display* derivation that runs alongside it.

## Context

A per-edit latency investigation bottomed out in the fog occupancy rebuild (`fog_upload` ≈ **592ms**
of the ~0.70s/edit, measured via Tracy on a large/high-density scene — the cuboid mesh after #40 is
~37–113ms by comparison). The shape of that cost exposed the real problem: **per edit, the CPU
densifies the compact scene into millions of voxels and uploads them to the GPU purely so the GPU can
display them.** The *input* to that work is already compact — a tree of SDF producers (kind + params
+ transform + combine op) plus, in future, sparse sculpt-override deltas. Expanding it to voxels on
the CPU and shipping the expansion is the waste.

ADR 0006 already ruled on the deeper question ("why is the SDF→voxel→fog pipeline on the CPU at
all?"): the CPU resolve stays **authoritative** (every non-render consumer — `.vox` export,
measurement readouts, agent `query`/`diagnostics`, undo, persistence, goldens — reads CPU occupancy),
but a **GPU view-resolve for display** is legitimate, gated on (i) being measured as the bottleneck,
(ii) the CPU foundation it derives from having stabilised, and (iii) a CPU↔GPU A/B equivalence net.
This ADR pins *how* that GPU view-resolve works.

**The reframe that makes it worth doing now:** the CPU authoritative resolve **does not have to run
every edit**. It must run only when a CPU *consumer* needs occupancy (export, a readout, an agent
query, undo, a golden). If the GPU owns the per-frame picture, then per edit the CPU only **updates
the compact tree, marks dirty chunks, and streams the changed chunks' compact descriptors to the
GPU** — kilobytes, no densify. The expensive CPU resolve becomes **on-demand and region-scoped**, off
the interactive hot path.

## Decision

A second, **display-only** resolver runs on the GPU, fed the **compact representation**, never
expanded voxels. It is chunked, derived, never authoritative, and never required.

### 1. The streamed representation is the compact per-chunk descriptor, not voxels

Per visible/resident chunk, the CPU streams a **compact descriptor**: the producers whose world-AABB
intersects the chunk (`ProducerKind` + params + the order-48 transform + `CombineOp` + `BlockId`),
and — once ADR 0003 §3e lands — that chunk's **sparse sculpt-override deltas** (integer force-on /
force-off addresses + block ids). This is the same data ADR 0003's `Layer` stack already holds; the
GPU receives the *recipe*, not the baked result. A chunk's descriptor is a few hundred bytes
regardless of how many voxels it resolves to.

### 2. The GPU view-resolve is CHUNKED, mirroring the CPU store residency

The GPU resolver is **not** "evaluate one SDF over the whole scene" — a >10k-block scene (ADR 0003)
cannot be one GPU volume (it exceeds `max_texture_dimension_3d`; the fog already self-disables past
that). It mirrors the CPU store: only **resident/visible chunks** get a GPU field, keyed by absolute
chunk coord, evaluated per chunk into a chunk-local field (the same `CHUNK_BLOCKS × density` extent +
1-voxel apron the CPU per-chunk fog and cuboid mesher already use). Residency, eviction, and the
dirty set are driven by the SAME `AppCore` bookkeeping that drives the CPU store (the
`invalidate_aabb` dirty coords #40 already surfaces). A GPU compute dispatch evaluates a chunk's
producer tree at each voxel sample point, composites the sculpt deltas (later-wins, ADR 0003 §3b),
and writes the chunk's display field.

### 3. The GPU field derives the display artifacts; the CPU never does so on the hot path

From each chunk's GPU-resident field the GPU produces, for display only:
- **(P1) Fog occupancy** — the R8 apron'd field the existing raymarch shader already samples (the
  atlas/meta layout of `OnionFogRenderer` is reused unchanged; only its *source* moves from CPU
  densify+upload to GPU evaluation).
- **(P2) The display mesh** — the surface, GPU-meshed (greedy/cuboid faces or an SDF raymarch; the
  exact mesh-vs-raymarch choice is a P2-time decision, below). This supersedes the *per-edit* CPU
  cuboid mesh **for display**; the CPU cuboid mesher (#20, #40) **remains** as the headless / no-GPU
  fallback and the A/B reference.
- **(P4) Instanced parts** — a referenced definition's chunk fields/meshes drawn N times under each
  instance transform.

### 4. The CPU authoritative resolve goes ON-DEMAND (off the per-edit display path)

`apply_intent` keeps doing exactly what ADR 0006 §7 mandates — mutate the tree, write the sparse
delta, **mark chunks dirty** — and additionally streams the dirty chunks' compact descriptors to the
GPU. It does **not** run the CPU densify for display. The CPU `resolve_into` / `resolve_region` runs
**lazily**, region/chunk-scoped, only when a CPU consumer pulls: `.vox` export, the diameter / layer
/ slice readouts, an agent `query`/`diagnostics`, an undo that needs occupancy, a golden. This is
sound because the CPU resolve is already cheap and region-scoped after `resolve_into` (~17ms for
`resolve_region`); the readouts are occasional, not per-frame.

### 5. Invariants (ADR 0006, restated concretely — non-negotiable)

- **CPU is truth; GPU is never authoritative and never required.** Headless `AppCore` (agents, CI,
  `shot`, goldens) resolves on the CPU with no GPU attached — the GPU view-resolve is a shell-only
  accelerator. Nothing reads the GPU field back as truth (no readback-as-delta — ADR 0006 §4).
- **A/B equivalence net.** Every GPU display field is checked against the CPU authoritative occupancy
  for the same chunk + sample points. **Parity is the target, not similarity** (see §6). This is a
  `--features gpu` test discipline (GPU field readback vs `resolve_into` / `build_per_chunk_fog_occupancy`)
  plus the existing byte-identical goldens; it gates every phase before live wiring (spike-first).
- **One `AppCore`, optional shell** (ADR 0006 §6) — the GPU resolver lives in the render shell, fed
  by `AppCore`'s dirty-chunk + compact-descriptor output; it is not a second authoring path.

### 6. The determinism contract for the A/B net

The CPU and GPU evaluate the **same SDF formula at the same integer voxel-sample points**, in a
**fixed combine-evaluation order** (the `Layer` stack order, ADR 0003 §3b). The occupancy decision is
a sign test (`inside ⇔ sdf ≤ 0`). For the **parametric tier the A/B net demands EXACT occupancy
parity** — same inputs + same op order + IEEE-754 should give the same sign at each sample. The
**known risk is Rust↔WGSL float divergence** (fma contraction, transcendental precision, rounding) at
voxels whose SDF sits within an ULP of zero — a sub-voxel boundary disagreement. The spike (P1)
**measures this first**: if exact parity holds across the shape/size matrix, exact is the contract; if
a thin boundary band diverges, the contract degrades to a **pinned, asserted tolerance** (e.g. "differ
only at voxels with `|sdf| < ε`, count bounded") — never silent. Sculpt deltas are **integer**
(force-on/off) and composite **exactly** on both sides, so the sculpt tier is exact by construction.

### 7. Producer portability — per-producer opt-in, CPU fallback for the rest

The SDF primitives (sphere / box / cylinder / torus, the order-48 transform, the `CombineOp` fold)
port to WGSL directly. The sketch **extrude/revolve** producer (`SketchSolid`) is in the **P1 GPU
set** too (owner decision — it covers the shapes actually authored, the VS-first sketch→volume atom):
its 2D profile polygon streams to the GPU and the eval is a crossing-number point-in-polygon test in
the cross-section plane (extrude) or in `(radius, height)` space (revolve). It is **harder to make
bit-matching** (more float ops, the polygon test) — so it is exactly where the §6 exact-parity spike
earns its keep. The remaining producer — `DebugClouds` (a static debug field, not an authoring
target) — does **not** port; chunks touching it **fall back to the CPU resolve-and-upload** display
path. So the GPU path is **per-producer opt-in**: a chunk whose producers are all GPU-portable is
GPU-resolved for display; a chunk touching `DebugClouds` falls back per chunk. Mixed scenes work
because residency + display are already per chunk. This keeps the port incremental and never blocks a
scene on an unported producer.

## Sequencing (each phase: spike behind the A/B net → wire → measure)

- **P1 — SDF tier (primitives + `SketchSolid`) → GPU per-chunk fog field.** The end-to-end proof of
  "stream params → GPU voxelizes → fog slices." GPU set = sphere/box/cylinder/torus + the order-48
  transform + `CombineOp` fold + extrude/revolve sketch solids (§7); `DebugClouds` chunks fall back to
  the CPU display path. Derives from the **shipped** `resolve_into` / producer-registry seam (ADR 0006
  gate (ii) satisfied — front-runs nothing). Spike = a `--features gpu` A/B test asserting the
  GPU-evaluated chunk field equals `build_per_chunk_fog_occupancy` **exactly** (§6 contract) across the
  shape/size matrix — including the sketch solids, where the float-divergence surface is largest. Kills
  the `fog_upload` CPU densify + per-edit texture re-create for GPU-resolved chunks.
- **P2 — GPU display mesh.** Mesh/raymarch the GPU field for the surface; CPU cuboid mesher stays as
  the headless fallback + A/B reference. Decision deferred to P2: GPU greedy-mesh (matches the cuboid
  silhouette, comparable to the existing goldens) vs SDF raymarch (no mesh, but a different visual
  contract). Removes the per-edit CPU mesh from the display path.
- **P3 — Sculpt-delta compositing (GATED on ADR 0003 §3e).** Stream the override layer's sparse
  integer deltas; composite on the GPU after SDF eval (later-wins). Waits for the sculpt foundation —
  this is the one phase ADR 0006's gate (ii) legitimately blocks until §3e lands.
- **P4 — Parts / GPU instancing.** Draw a definition's GPU chunk fields/meshes N times under instance
  transforms.
- **Throughout — CPU resolve goes on-demand.** As P1/P2 remove the per-edit display dependency on the
  CPU densify, the authoritative resolve is pulled only by CPU consumers (§4). The monolithic
  `resolve_region` whole-grid assembly (the fog's last per-edit consumer) is retired once P1 lands
  (ADR 0006 "Next (CPU)").

## What stays on the CPU (the on-demand authoritative path)

`.vox` export; the diameter / layer-band / slice readouts; the ADR 0004/0005 agent `query` /
`diagnostics`; undo / the command journal; chunk persistence + disk-spill; the golden + lib-test
spine. All read CPU resolved occupancy, resolved lazily and region-scoped. None depend on the GPU.

## Consequences

- **Positive:** per-edit display cost moves off the CPU densify entirely (the measured 592ms fog +
  the CPU mesh); per-edit CPU work collapses to "diff tree + stream compact descriptors"; bandwidth
  drops from "millions of voxels" to "a recipe per dirty chunk"; the chunked GPU resolver inherits the
  existing residency/scale story; the on-demand CPU resolve stays the single source of truth for
  everything that must be exact.
- **Negative / cost:** a **second resolver** — the SDF/compositor logic exists in Rust (CPU truth) and
  WGSL (GPU display), kept in lockstep ONLY by the A/B net (the maintenance burden the net exists to
  police); the **first compute pipeline** in the repo (new infra: compute pipelines, storage buffers,
  `copy_buffer_to_texture`); the float-determinism contract (§6) needs verification before it can be
  called exact; non-portable producers carry a CPU-fallback display path until ported (two display
  paths during the migration).
- **Risk if skipped / done wrong:** a GPU path that drifts from CPU truth silently is the failure mode
  ADR 0006 names — the A/B net + goldens are the mitigation, mandatory per phase, spike-first.

## Alternatives considered

- **Incremental CPU fog upload (re-upload only dirty chunks' tiles).** A real local win, but it still
  CPU-densifies + ships expanded voxels — it does not realise the compact-streaming goal and leaves the
  CPU on the per-edit display hot path. Rejected as the *target* (still viable as a stopgap if P1 slips).
- **GPU-scatter the occupied voxel list into the R8 texture.** Same flaw: the input is the *expanded*
  occupied list (for a dense scene, as large as the dense atlas), not the compact tree. A half-measure.
- **Full GPU-authoritative pipeline (GPU replaces the CPU resolve).** Rejected by ADR 0006: export /
  agents / determinism / scale / sculpt-authoring all require CPU-authoritative occupancy. The GPU is
  display, never truth.

## Decisions resolved (owner sign-off 2026-06-29)

1. **Determinism contract (§6): EXACT parity, proven in the spike.** Demand bit-exact CPU↔GPU
   occupancy parity; the P1 spike measures whether Rust↔WGSL float eval agrees at boundary voxels.
   Degrade to a pinned `|sdf| < ε` tolerance ONLY if measured to diverge — never silently.
2. **P1 scope: SDF primitives + `SketchSolid`.** The GPU set includes extrude/revolve sketch solids
   (the actually-authored shapes), not just primitives (§7); `DebugClouds` stays CPU-fallback. The
   sketch solids are where exact parity is hardest — the spike validates them explicitly.
3. **No sculpt front-run.** P1/P2 derive from the shipped `resolve_into` seam (ADR 0006 gate (ii)
   satisfied); P3 (sculpt-delta compositing) waits for ADR 0003 §3e. Confirmed.

## P1 spike result (2026-06-29 — the SDF-tier A/B net landed)

The P1 spike is **built and green** (`src/gpu_resolve.rs` — the repo's first compute pipeline —
`src/shaders/gpu_resolve.wgsl`, and the `tests/gpu_parity.rs` A/B net, `--features gpu`). It
GPU-evaluates each chunk's apron'd occupancy from the streamed compact descriptor (producer params +
profile) and asserts it **byte-identical** to `build_per_chunk_fog_occupancy` over a shape/size matrix
resolved through the REAL `Scene::resolve_region` path (the recentred frame the fog actually consumes).

**Measured finding: EXACT parity holds across the whole tested matrix — no tolerance needed (yet).**
Both the SDF tier (sphere/box/cylinder/tube/torus, even/odd parity, multi-chunk seams; f32 both sides)
AND the `SketchSolid` tier (extrude — rectangle/concave-L/triangle; revolve — cylinder/vase/bowl, a
partial 180° turn exercising the `atan2` gate, and a straddling profile) come out bit-exact, *despite*
the CPU running the polygon test in **f64** and the GPU in **f32** (no portable f64 in WGSL). The
§6 Rust↔WGSL boundary-divergence risk — which was expected to bite revolve's irrational-radius polygon
test hardest — **did not manifest on this adapter** (the RTX golden machine).

Caveat (kept honest, not silently assumed): "exact" here is *measured on this matrix + this adapter*,
not proven for all inputs. A denser/larger revolve or a different GPU could still surface a thin
boundary band; the §6 contract then degrades to the pinned tolerance below. The A/B net is the standing
guard that would catch it. The driver currently asserts a single-dimension workgroup-count limit (the
spike dispatches 1-D), so very large cases are split by keeping density vs chunk-count balanced.

**Atlas mechanic proven (2026-06-29).** The production texture-write path is built + green: a second
compute entry (`main_atlas`) packs the GPU-evaluated occupancy as bytes (`atomicOr`, 256-aligned rows)
straight into the `upload_grid_per_chunk` atlas layout; `copy_buffer_to_texture` lands it in the R8
atlas, and the A/B net asserts the texture (read back) is **byte-identical** to the CPU atlas packing —
including the empty-tile zero-fill and the row-alignment padding. So the GPU can now produce the exact
R8 atlas the per-chunk fog raymarch samples; what remains is the live call-site swap.

## Open (deferred to their phase)

- **Live call-site swap** — replace the per-edit CPU `build_per_chunk_fog_occupancy` + atlas pack at
  the `main.rs` / `shot.rs` fog-upload sites with the GPU atlas path. The remaining design question is
  the **chunk-set / residency** source: the GPU path enumerates COVERING chunks (from the producer
  AABB) rather than the CPU's occupied-bucketed non-empty set, so empty tiles appear (benign — no
  occupancy, no fog) but the atlas is no longer byte-identical to the CPU one; the goldens (pixel
  tolerance) are the guard there, and `DebugClouds` / multi-producer composites fall back to the CPU
  path per chunk. This is the architecturally-invasive step (touches `AppCore` dirty-set plumbing).
- **P2 mesh-vs-raymarch** for the GPU display surface (decided at P2, not now).
- **The pinned tolerance value**, only if a future case/adapter proves exact parity unattainable.
