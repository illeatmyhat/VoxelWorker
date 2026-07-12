# ADR 0011 — The GPU brick-field display sink: raymarch a cached 8³-brick atlas of the boundary set, clip-map LOD, BVH broadphase

- **Status:** **Accepted + SHIPPED (epic #64, slices G0→G5 all landed 2026-07-08; owner grill 2026-07-05/06).**
  Live brick raymarch (hit-set parity EXACT vs the evaluator, brick == mesh), L1–L3 clip-map pyramid, edit-BVH
  broadphase, incremental persistent atlas, and fog-from-bricks are all in the runtime. **G5 dense-grid retirement
  completed 2026-07-11**: the fog `VoxelGrid` stream is now GONE — the universal brick-fog law sources onion fog from
  the brick sink for EVERY chunkable scene on EVERY build (gpu or not, loaded VS material included), `AppCore::rebuild`
  assembles no dense grid, the shell carries no `VoxelGrid` husk, and `expand_resident_chunks_into_grid` /
  `Scene::resolve_region` survive only as `#[cfg(test)]` parity oracles + `bin/shot.rs`'s golden reference. The sole
  surviving dense resolve is the transient, non-chunkable-only Part-only fog densify (a degenerate empty region).
  Sub-decisions are tagged **[DECIDED-0009]** (settled by ADR 0009's benchmark) or **[DECIDED-grill]** (ruled in the
  owner grill; the former [OPEN-grill] positions, each resolved in §Open questions at the end). One draft position was
  overturned as a units error (Decision 1: the granule is one BLOCK, not a voxel count — amends ADR 0009's "8³").
  **Amended 2026-07-12:** two post-ship contract changes are live — brick records are
  **surface-only at emission** (interior elision fused into the build; the all-blocks
  build survives as the parity oracle), and the wholesale pipeline (record build +
  pyramid + classify) runs **asynchronously** on a dedicated worker with generation
  supersede (stale-while-rebuilding). The fog-from-bricks consumer named above was
  retired with the volumetric fog subsystem by ADR 0012. Living shape:
  `docs/architecture/03-display.md` / `04-work.md`.
- **Date:** 2026-07-01
- **Layer:** PRODUCTION PORT of the **GPU display sink** — the next port after [ADR 0010](0010-boundary-residency-two-layer-store.md).
  **Generalizes [ADR 0007](0007-gpu-view-resolve.md)** (the shipped per-chunk R8 fog atlas is already a brick map;
  this ADR turns each per-chunk occupancy tile into an 8³ brick allocated from a texture-atlas pool and adds the
  broadphase + LOD that make it scale). **Consumes [ADR 0010](0010-boundary-residency-two-layer-store.md)**'s
  two-layer evaluator (the `TwoLayerChunk` coarse/microblock/seam-flag boundary set is the sink's input).
  **Realises [ADR 0009](0009-op-stack-truth-evaluator-and-sinks.md) §Open "Display path — DECIDED: C, the cached
  sparse brick field + clip-map LOD."** Governed by [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) (CPU is
  truth; GPU is a display shell, never truth, never required) and [ADR 0008](0008-voxel-frame-invariant.md) (spatial
  values carry their frame; the brick lattice is world-axis-aligned). **No new product model** — it is the GPU display
  derivation ADR 0009 §2 already named and ADR 0010 §Consequences deferred to "its own ADR."

## Context

ADR 0010 landed the **CPU exact seam**: the live display path is now the `TwoLayerResidentCache` — a boundary-aware
two-layer store (coarse per-block `BlockId` grid + a sparse map of boundary blocks → cuboid microblocks + per-face
seam-solidity flags), incremental per edit, with the dense `resolve_region` / dense `VoxelGrid` retired to a test-only
parity/golden oracle. But ADR 0010 §Consequences named the debt it left standing:

> **One whole-region `VoxelGrid` remains on the display path**: the onion-fog densify still consumes a `VoxelGrid`, so
> the evaluator STREAMS it (coarse fast-fill + boundary per-voxel) rather than caching a dense interior … the fog will
> drop it when the GPU brick-field display sink lands.

So there are still **two per-edit densify-shaped costs on the display path**: the CPU-side fog `VoxelGrid` stream, and
the CPU cuboid **mesh** rebuild — both regenerated for display, both scaling with occupied-voxel count, not with edit
size. This is the exact waste ADR 0007 measured (fog rebuild ≈ 592 ms/edit via Tracy) and ADR 0009 diagnosed at the
architecture level ("we materialize every interior voxel of a solid that was never chiseled").

ADR 0009 already **decided the fix** and benchmarked it. Its `experimental/sdf_bench` raced three display techniques on
an RTX 4090 at 4K to a 20 FPS ceiling — **(A)** adaptive skin mesh, **(B)** pure analytic SDF raymarch, **(C)** a cached
sparse brick field + clip-map LOD — and ruled **C** the winner (see §Considered Alternatives for the numbers). This ADR
does the same thing ADR 0010 did for the CPU seam: it **scopes the production port of technique C into the real chunk
store** as the GPU display sink, takes a position on each sub-decision ADR 0009 left open, and pins the correctness net.

**The one-line design rule inherited from the prior art** (Mike's solo-dev SDF engine, ADR 0009's citation): *optimize
for recompute, not render, because the world is dynamic.* Evaluate the op-stack **once per edit** into cached bricks
(boundary blocks only), then **per frame raymarch the cache, not the field** — so per-frame cost is independent of
op-stack complexity, and per-edit cost is proportional to the dirty region, not the scene. That is the same
"recompute-cheap, render-from-cache" shape ADR 0007's incremental atlas already reaches for.

## Decisions (proposed)

**1. The brick-field is the GPU display sink; it GENERALIZES the shipped fog atlas. [DECIDED-0009 direction; port confirmed by grill 2026-07-05]**
ADR 0007 ships a per-chunk R8 occupancy tile packed into a 3D texture atlas via `copy_buffer_to_texture`, keyed by
world-origin. The brick-field is that same mechanic with a finer, sparser, boundary-only granule:
- A **brick** is one **block's** cube of voxels cached in one atlas slot. **[DECIDED-grill 2026-07-05] Brick granule =
  one block, denominated in BLOCK units: brick edge = `voxels_per_block`**, whatever the document's density is. The
  draft posed this as "16³ vs 8³", which the grill ruled a **units error** (the same law as the original density bug:
  density is fineness only, never a dimension). Nothing shipped hard-codes 16 — `TwoLayerChunk.voxels_per_block` is a
  runtime parameter and the fog atlas already sizes tiles from it at runtime; the benchmark's `BRICK =
  VOXELS_PER_BLOCK_I` was choosing *one block*, not the number 16. A fixed-voxel granule (ADR 0009's "8³" phrasing —
  hereby **amended**: it echoed the prior art, whose world has no block concept) would tile a block cleanly only when
  density is a multiple of the brick edge. Block-granule maps 1:1 onto ADR 0010's `microblocks` keys, the shipped fog
  tile, the material/texture unit, and the seam-flag unit — the brick is the quantum of atlas memory, dirty recompute,
  and fine ray traversal, and it is denominated in the same currency as every other unit in the store. **Sub-block
  bricks** (e.g. (density/2)³ halves, with their evenness constraint + sub-block indexing layer) are a measured
  follow-up with an explicit trigger: build them only if a real scene shows (a) atlas memory dominated by
  mostly-uniform sculpted bricks, or (b) raymarch time dominated by in-brick DDA over empty voxels.
- Bricks are allocated from a **texture-atlas pool** (the ADR 0007 atlas layout, `ATLAS_TILES_PER_AXIS³` slots; the
  benchmark used 32³ = 32768 slots of a 512³ R8 texture ≈ 128 MB). Only **boundary** bricks consume a slot.
- The GPU **raymarches the cache per frame** (texture fetches + a block-DDA + a residency lookup), **never the analytic
  SDF**. Per-frame cost ≈ O(occupied blocks the ray crosses + log LOD levels), independent of scene complexity — the
  currency ADR 0009's benchmark showed C wins.

**2. The sink reads ADR 0010's two-layer boundary set directly; it does not re-derive occupancy. [DECIDED-0009/0010]**
The evaluator/classifier already partitions every block of a covering chunk into **air / coarse-solid / boundary**
(`two_layer_store.rs`: `CoarseVerdict`, `TwoLayerChunk.coarse: Vec<Option<BlockId>>`, `microblocks: BTreeMap<[u32;3], MicroblockGeometry>`).
The sink maps that partition onto brick kinds one-to-one:
- **air block** → no brick (empty; the ray skips it via the clip-map, §4).
- **coarse-solid block** (`coarse[i] == Some(id)`) → an **analytic "coarse brick"**: a `BrickRecord{kind: 0}` marker, a
  solid block-cube, **no atlas slot, no per-voxel data** (the benchmark's coarse kind). This is the interior-elision win
  carried onto the GPU: a solid interior costs one record, not 16³ bytes.
- **boundary block** (a `microblocks` entry) → a **sculpted brick**: `BrickRecord{kind: 1, atlas_slot}` whose 16³ (or 8³)
  R8 occupancy is packed into an atlas slot. The microblock's cuboids rasterize into the brick exactly as ADR 0007's
  `main_atlas` packs a tile — the sculpted voxels ARE the brick payload.
- **per-face seam-solidity flags** (`SeamSolidity`) carry across unchanged: they let the block-DDA cull a face against a
  fully-solid neighbour without expanding it — the coarse-vs-microblock analogue of the fog apron (CONTEXT.md "Seam
  solidity"), and the brick-field's equivalent of ADR 0007's C′ apron-zeroing.

> **Amendment (2026-07, the 8000³-freeze fix): the record set is SURFACE-ONLY.** "One coarse record per solid block"
> still made the CPU pipeline O(all blocks): a 500³-block solid emitted 125M records, and every downstream stage
> (sort, elision mask, pyramid fold, GPU pack, incremental-mirror clone) hauled the ~123.5M fully-occluded interior
> records it would then discard — a measured 12.5s main-thread freeze + ~6 GB transient. The contract is now:
> a coarse-solid block whose six face-neighbours are all present and solid on the shared face **emits no record at
> all** (`build_brick_field` fuses the occlusion decision into emission, with an interior-CHUNK fast path that skips
> fully-solid chunks ringed by fully-solid face-neighbours without visiting their blocks). Boundary (sculpted) blocks
> are surface by definition and are never elided, so the atlas is unchanged. Interiors remain queryable through the
> two-layer chunks: the clip-map pyramid derives from the CHUNKS (`ClipmapPyramid::from_chunks`), and the fog fill
> box-fills coarse occupancy from the CHUNKS (`build_per_chunk_fog_occupancy_from_bricks`). Incremental edits
> re-derive occlusion verdicts over the dirty set dilated by the 26-neighbourhood
> (`IncrementalBrickField::apply_dirty_update` — a carve can expose previously-interior blocks in a NON-dirty
> neighbour chunk). The interior-inclusive build survives as the parity oracle (`build_brick_field_all_blocks`),
> gated hit-identical to the surface-only build in `brick_surface_elision_hit_set_unchanged`.

This is why ADR 0010 called it "a short step": the fog atlas is *already this shape* (boundary-residency R8 occupancy in
a 3D texture atlas); the brick-field adds (i) the coarse-brick analytic marker for solid interiors, (ii) a sparse resident
record set instead of a covering-box tile grid, and (iii) the broadphase + LOD below.

**3. Two evaluators stay separate; the GPU brick build is a display derivation, A/B-checked. [DECIDED-0006/0009]**
Per ADR 0009 §5 and ADR 0006, there is **no shared GPU evaluator that is truth**. The CPU two-layer evaluator (ADR 0010)
is authoritative; the GPU sink is a display derivation fed the **boundary set** (records + atlas bytes + clip-map keys),
kept honest by an A/B parity net (§Parity gate). Headless `AppCore` (agents, CI, `shot`, goldens) never needs the GPU
sink — it reads the CPU exact seam. The brick-field is a shell-only accelerator (ADR 0007 §5 invariants restated).

**4. Broadphase + LOD — a position on each ADR 0009 open sub-decision:**

- **4a. Empty-space skipping / LOD = a clip-map occupancy pyramid + hierarchical DDA. [DECIDED-0009 for technique;
  level count DECIDED-grill: 2 levels first (the benchmark's proven config), 3rd/4th at G4, measured]** Besides the fine resident bricks, the build emits **coarser "any-brick-inside" occupancy levels**
  (the benchmark: L1 = 8-block cells, L2 = 64-block cells, each a sorted set of packed cell keys). The shader's
  hierarchical DDA finds the **coarsest empty level** covering the current block and jumps the ray to that cell's exit —
  one big stride through empty space — descending to per-block brick work only where the finest level is occupied. This is
  the measured fix that lifted C's scattered ceiling **160 → 10240 (~64×)** in ADR 0009's gating experiment. **Position:
  port the benchmark's 2 coarse levels first, then add a 3rd/4th level** (ADR 0009 notes "C had only two LOD levels — a
  3rd/4th closes most of the gap" vs mesh's free frustum/Z cull). The distinction from *geometry* clip-maps
  (Losasso–Hoppe nested camera-centered vertex grids) is deliberate and worth the owner's eye: **[OPEN-grill]** the
  benchmark's pyramid is **world-fixed occupancy levels** (a min-mip of the brick set), NOT camera-recentred toroidal
  grids. **[DECIDED-grill 2026-07-05] World-fixed pyramid first; camera-centred residency rings deferred to a measured
  need — but the grill split the concern to answer "when does off-screen residency stop being engineering?":**
  - **The residency POLICY** (all boundary bricks resident vs finest-near-camera rings) is *engineering* — tunable,
    swappable, deferred until a real off-screen scene proves the resident budget bites (the benchmark's 512³ R8 atlas
    holds ~32k sculpted bricks in 128 MB at density 16; an anisotropic 10k+-block build can plausibly exceed it).
  - **The residency CONTRACT is architecture, and it is decided NOW, at G1:** the renderer must never assume every
    boundary block has a resident sculpted brick. **A residency miss is legal and renders degraded-but-correct: a
    non-resident boundary block falls back to its coarse form (solid block-cube at its block id)** — the coarse-brick
    record kind Decision 2 already defines, the same acceptable-transient visual as the #60 stale-while-rebuilding
    pattern. One branch in the shader, paid up front. Rings at G4 are thereby pre-classified as a pure eviction policy
    plugged into an existing hole, never a lookup-path redesign.

- **4b. Two broadphases: the clip-map pyramid for the RAY, one BVH of producer AABBs for EDITS. [DECIDED-grill 2026-07-05]**
  These are two different problems ADR 0009 collapsed into one bullet, and the grill confirmed the split:
  - **Render broadphase (per frame, in-shader):** the clip-map occupancy pyramid (4a) skips empty space hierarchically;
    a **sorted resident-record array + in-shader binary search** on a packed world-block key (`t3_brick.rs`:
    `pack_world_block`, WGSL binary-search) resolves residency — O(log #bricks) per block step, no tree. Ported as-is.
  - **Edit broadphase (per edit, CPU-side):** a **BVH of producer AABBs** is THE edit broadphase — it answers "which
    producers overlap this box" for both the wholesale two-layer build and G3's dirty-brick recompute (the prior art's
    exact use: dirty AABB → overlapping edits → bricks to re-evaluate). It **replaces** #63's
    `bucket_leaves_into_chunks` chunk-bucketing rather than sitting alongside it — the grill rejected carrying two
    structures for one query (divergence risk). The swap is gated by the existing two-layer parity tests: candidates
    are an exact superset either way, so classification is provably unchanged, exactly as #63's own swap was gated.
    **Held stateless: rebuilt per edit from scratch** (~1ms at the 10k-object target) — no invalidation obligation, no
    stale-cache bug class (the C1 lesson); a persistent/refitted BVH only if rebuild-per-edit ever measures hot. A BVH
    is also GPU-legible for per-tile candidate pruning if GPU-side brick fills want it later.

- **4c. Editable sparse delta structure = ADR 0010's `microblocks` BTreeMap remains truth; the brick set is a derived
  cache. [DECIDED-0009/0010]** ADR 0009 §6 keeps sculpt deltas in a sparse, edit-friendly store (sorted-key list / hash
  grid); ADR 0010 realised that as the per-chunk `microblocks: BTreeMap`. The brick records + atlas are a **derived,
  rebuildable cache** of the boundary set — never the edit store. ADR 0009's "HashDAG as delta counts grow" is a
  compaction target for the *export/display cache*, still deferred and not on this port's path.

**5. Correctness: the A/B parity net mirrors ADR 0007's `gpu_parity`; display may approximate, the net keeps it honest.
[DECIDED-0007 discipline]** ADR 0007 established GPU-vs-CPU byte-exact parity (`tests/gpu_parity.rs`, `--features gpu`)
and it held bit-exact across the whole matrix. The brick-field extends it: for a gated scene, the **GPU brick raymarch's
resolved occupancy** is checked against the **CPU exact evaluator's** occupancy (ADR 0010's streamed exact set, the
reference oracle). Two tiers, honestly separated:
- **The brick BUILD is exact.** Packing a boundary block's cuboids into an atlas brick, and marking a coarse-solid block,
  are integer operations — the atlas bytes must be **byte-identical** to the CPU boundary set (exactly ADR 0007's
  `main_atlas` parity, now over the two-layer boundary set instead of the covering box).
- **The brick RENDER may approximate.** LOD level selection, clip-map min-mip, and float ray/DDA math are display
  approximations ADR 0009 §4 explicitly allows on the display seam. The net asserts **the finest-LOD raymarch hits the
  same surface voxels** the CPU evaluator reports; coarser LOD is checked for *conservative* coverage (never drops a
  surface the finest level would show), not bit-equality. Minification/LOD aliasing is display polish (ADR 0009 §Open),
  not a parity failure.

**6. Coexist behind a capability; the CPU mesh is KEPT PERMANENTLY, pinned at exactness-not-feature-parity. [DECIDED-grill 2026-07-05]**
Exactly ADR 0010 §6 / ADR 0007's coexistence: the brick-field engages for the producers + scenes it supports (all ADR 0007
P1 producers already port); the CPU two-layer mesh (`new_from_two_layer_chunks`) stays as the **headless / no-GPU
fallback and the A/B reference** (ADR 0007 §3 kept the CPU cuboid mesher for exactly this). The grill noted this is
mostly *constitutionally forced*: ADR 0006 ("GPU never required") demands a CPU display path for headless `AppCore`
(agents, CI, `shot`, goldens), and the parity gate needs a CPU-rendered oracle forever — so "retire entirely" was never
live. **The ruling with teeth is the TERMS: the CPU mesh owes correct geometry at finest detail (the exactness the
parity gate + goldens check, and what a no-GPU user needs to plan a chisel job) and NOTHING else — no LOD, no clip-map,
no display polish.** A future display feature is implemented on both sides only if it changes *which voxels are shown*;
anything cosmetic is GPU-only by default. This kills the feature-parity treadmill (a second co-evolving renderer)
without a sunset date — the CPU mesh's two jobs (constitutional fallback, parity oracle) don't expire. Every commit
stays green; goldens cross-check GPU-vs-CPU each slice; the CPU **display** densify (the fog `VoxelGrid` stream ADR 0010
flagged) is retired only once the brick-field covers everything it did — and the CPU **exact** seam
(export/query/golden) is untouched (it never rendered).

## The parity gate (non-negotiable, mirrors the existing nets)

A **brick-vs-evaluator parity test** extending `tests/gpu_parity.rs`, gating every slice before live wiring (spike-first,
as ADR 0007 P1 was): for every gated scene, **(a)** each boundary block's packed atlas brick is byte-identical to the CPU
two-layer boundary set's occupancy for that block, and each coarse-solid block emits exactly one coarse record (no atlas
slot); **(b)** the finest-LOD GPU brick raymarch's hit-voxel set equals the CPU exact evaluator's surface set for the
gated views, and each coarser clip-map level is **conservative** (its "any-brick-inside" set is a superset of the true
occupied cells — never skips a ray past a real surface); **(c)** the existing pixel goldens (`onion-fog-perchunk`, the
sphere/cylinder/torus/sketch-revolve scenes, `debug-clouds`) render GPU-brick pixel-identical to the current
two-layer-mesh path where LOD is at its finest, so they auto-cover the sink with no new goldens. Brick caching is thereby
a **pure display derivation** on the exact data, and LOD is the only sanctioned approximation.

## Slice plan (each independently green-gated; verification per the session gate)

Modelled on ADR 0010's D0→E5 sequencing — minimal exact slice first, LOD/broadphase/incremental layered on, CPU display
path retired last.

- **G0 — brick-build parity harness (no render).** Extend `gpu_parity`: pack ADR 0010's `TwoLayerChunk` boundary set into
  a sorted `BrickRecord` array + R8 atlas (coarse records for coarse-solid, sculpted bricks for `microblocks`), assert
  **(a)** byte-identical to the CPU boundary occupancy. Wired to nothing yet (mirrors ADR 0007's atlas-mechanic-proven
  step and ADR 0010's E1 standalone parity). Reuses ADR 0007's `main_atlas` packing.
- **G1 — minimal brick raymarch, single ported producer, finest LOD only.** Port `t3_brick.wgsl`'s brick-DDA +
  resident-record binary search (no clip-map yet); a scene of one ADR 0007-ported producer sources display from the brick
  atlas. **Design requirement (4a ruling): the residency-miss contract is baked in here — a lookup miss on a boundary
  block renders its coarse form, never asserts/skips.** **Design requirement (grill Q5): the raymarch pass writes
  ray-hit DEPTH and composites correctly with the existing rasterized overlays (origin gizmo, fog, view cube, egui) —
  the one integration point the benchmark never exercised; verified by the existing goldens plus one overlay-inclusive
  shot.** Parity **(b)/(c)**: hit set == CPU surface; the sphere/revolve
  goldens render brick-path pixel-identical. This is the ADR 0007 live-swap analogue, now brick-granular. Kills the CPU
  fog `VoxelGrid` stream (ADR 0010's flagged debt) for single-producer scenes.
- **G2 — clip-map LOD (the scattered-ceiling fix).** Emit the L1/L2 occupancy pyramid + hierarchical DDA (4a);
  multi-producer + scattered scenes engage. Parity: each level conservative; finest-LOD goldens still pixel-identical.
  This is the port of ADR 0009's *measured* 160→10240 lift — the slice that makes the sink scale.
- **G3 — GPU-side incremental atlas updates (recompute-cheap edits).** The `AppCore` dirty-set (ADR 0007 §Open;
  `invalidate_aabb`, ADR 0010 §Consequences) re-evaluates **only dirty blocks** into their bricks in a **persistent** atlas
  (no full rebuild, no readback of occupancy — only the compact resize readback ADR 0007 proved unavoidable). This is
  where the **edit-broadphase BVH** (4b) lands: dirty AABB → overlapping producers → dirty bricks. Per 4b's ruling the
  BVH also **replaces** `bucket_leaves_into_chunks` on the wholesale build path in this slice (or a small precursor
  slice), gated by the existing two-layer parity tests — one edit broadphase, not two. The per-edit win ADR 0009
  promised ("~3× lower edit latency") lands here; Tracy-measured live, as ADR 0007 §6 established the incremental
  path can't be golden-tested headless.
- **G4 — more LOD levels + off-screen residency (scale polish).** Add a 3rd/4th clip-map level (ADR 0009: closes most of
  the mesh gap); if a real off-screen scene proves the resident brick budget bites, add camera-centred residency rings
  (4a) as an eviction policy tied to ADR 0003 streaming. Pure eviction policy by construction — G1's residency-miss
  contract already gave it its hole. Engineering, not architecture.
- **G5 — retire the CPU display densify.** Once the brick-field covers every producer/scene the two-layer mesh does, drop
  the CPU fog `VoxelGrid` stream from the runtime display path (ADR 0010's last dense-shaped display consumer). Keep the
  CPU two-layer mesh as the **headless/no-GPU fallback + A/B reference** (ADR 0007 §3 precedent) — the exact CPU
  export/query/golden seam is never touched. **[DECIDED-grill 2026-07-05] keep-both, permanently, on Decision 6's
  terms: the CPU mesh is pinned at exactness (correct geometry, finest detail), owes no feature parity, no sunset.**

## Consequences

- The last per-edit densify-shaped display cost (ADR 0010's flagged fog `VoxelGrid` stream + the CPU cuboid mesh) leaves
  the hot path; per-edit display work collapses to "re-evaluate dirty bricks → patch the persistent atlas," proportional
  to the dirty region, not the scene. Per-frame cost becomes independent of op-stack complexity (raymarch the cache).
- **A third GPU derivation to keep in lockstep with CPU truth.** After the ADR 0007 fog atlas, this is more WGSL that can
  silently drift — the parity net (§gate) is the mandatory, spike-first police, exactly as ADR 0006 demands. This is the
  standing cost the sink pays.
- **Two broadphases, made explicit:** the clip-map occupancy pyramid (render broadphase) and the edit BVH (dirty-brick
  broadphase). ADR 0009 listed one "broadphase + LOD" bullet; this ADR splits it because they solve different problems.
  The edit BVH is the SOLE edit broadphase — it supersedes #63's chunk-bucketing (one query, one structure) — and is
  held stateless (rebuilt per edit) to avoid the stale-cache bug class.
- **Interior elision reaches the GPU:** a solid interior costs one coarse `BrickRecord`, not 16³ atlas bytes — the
  coarse-until-chiseled win (ADR 0009's ~3× on the dominant cost) now holds on the display sink too, not just the CPU
  store.
- **The fog and the mesh converge onto one sink.** ADR 0007's fog atlas and the cuboid mesh were separate display
  artifacts; the brick-field is the single "raymarch the cached boundary set" derivation that subsumes both for display —
  ADR 0009's "one evaluator, many sinks" finally has its GPU sink singular.
- **Rotated baked-voxel parts stay deferred** (ADR 0009 §Consequences / ADR 0010): the brick lattice is world-axis-aligned
  (ADR 0008), so a rotated sculpted part would staircase on the shared lattice → the lossy-resample / local-lattice path,
  not this fast path. Off the critical path for this port.
- **LOD introduces sanctioned approximation** on the display seam (minification aliasing, coarse-level coverage) — allowed
  by ADR 0009 §4, bounded by the conservative-coverage clause of the parity gate, never leaking to the exact seam.

## Considered alternatives (rejected — ADR 0009's benchmark is the evidence)

- **Pure analytic SDF raymarch (technique B), no cache.** Raymarch the op-stack itself every frame. Rejected by ADR 0009's
  benchmark: B is "the wall" — a scattered ceiling of ≈2.5–4k objects because per-frame cost scales with op-stack
  complexity (every ray step evaluates every nearby edit). The prior art (Mike's engine) rejects it for the same reason
  and caches bricks instead; the "interpolating cached distances, not raymarching the field" note in the HN discussion is
  precisely this choice. The brick-field's whole point is to make per-frame cost independent of scene complexity.
- **Pure skin mesh (technique A), no brick cache.** Regenerate the surface mesh per edit and rasterize. ADR 0009's
  benchmark: A reaches a *higher raw* scattered ceiling (≥16384 — the rasterizer gets frustum/Z cull free) but at **185 MB
  / 183 µs-edit vs C's 46 MB / 57 µs-edit** — ~4× the memory and ~3× the edit latency. C wins the currencies that matter
  for an edit-heavy, large, mostly-off-screen world (memory, edit latency, render cost independent of complexity), and a
  3rd/4th LOD level closes most of A's raw-ceiling lead. The CPU two-layer mesh (ADR 0010) is *kept* as the headless
  fallback + A/B reference — this rejection is of mesh as the *GPU display sink at scale*, not of mesh existing.
- **Display brick-field before the CPU exact seam** (port the GPU fog atlas into a brick-field before ADR 0010's CPU
  seam). This was ADR 0010's own rejected alternative — build the exact CPU seam first, then generalize to the GPU sink.
  ADR 0010 landed; this ADR is the "then generalize" half, correctly sequenced.
- **Keep the ADR 0007 covering-box tile atlas as the display path** (don't go sparse/boundary-record). Covering tiles work
  for a single producer but re-grow the atlas budget at multi-producer scale (ADR 0007's own "covering set > MAX_FOG_CHUNKS
  → CPU fallback" finding); the sparse boundary-record + clip-map set is what scales past that, which is why ADR 0009
  benchmarked *sparse* bricks, not covering tiles.
- **Fixed-voxel brick granule (ADR 0009's "8³")**. Finer culling, smaller dirty units — but a granule denominated in
  voxels violates the units law (it tiles a block cleanly only when density is a multiple of the edge) and needs a
  sub-block indexing layer that doesn't map 1:1 onto ADR 0010's per-block `microblocks`. The grill ruled the granule is
  **one block** (Decision 1, amending ADR 0009's phrasing); *sub-block* bricks remain a measured follow-up behind the
  explicit trigger in Decision 1, not rejected outright.

## Prior art (external validation)

- **Mike's solo-dev SDF game engine** (YouTube `il-TXbn5iMA`; [HN discussion 46539478](https://news.ycombinator.com/item?id=46539478),
  2026) — ADR 0009's primary citation. Scene as an ordered list of SDF edits (truth); a **sparse grid storing only cells
  crossing the zero level set** (boundary residency); **cached distances interpolated per frame rather than raymarching
  the field** ("optimize for recompute, not render"); a **BVH of edits** (shared CPU/GPU) for broadphase + dirty-brick
  incremental recompute; **clip-map LOD** for draw distance. Our ADR 0007 fog atlas is already this brick map; this port
  is the short step ADR 0009 named. (Caveat, honest: the HN text confirms the sparse-boundary-grid + cache-not-field +
  BVH + Jolt-physics claims; the deeper brick/LOD specifics are in the video, which ADR 0009 already digested.)
- **Geometry clipmaps** — Losasso & Hoppe, *"Geometry clipmaps: terrain rendering using nested regular grids"* (ACM
  TOG 23(3), 2004, [hhoppe.com/proj/geomclipmap](https://hhoppe.com/proj/geomclipmap/)); GPU implementation, Asirvatham &
  Hoppe, *GPU Gems 2* Ch. 2, [NVIDIA](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry).
  The nested camera-centred grid + transition-region morph is the reference for Decision 4a's *camera-centred residency
  rings* variant; our first-slice pyramid is a world-fixed occupancy min-mip (a simpler cousin), and the ADR flags the
  distinction for the owner.
- **Sparse Brick Set (SBS) vs Sparse Voxel Set (SVS)** — the SDF acceleration-structure taxonomy (e.g. CrossRT,
  [arXiv:2409.12617](https://arxiv.org/pdf/2409.12617)): a small regular grid (2³/4³/8³) stored per brick reduces
  distance-value duplication vs per-voxel storage, at the cost of more per-brick intersection work. Our 16³/8³ R8 atlas
  brick **is** an SBS granule; this validates the "brick, not per-voxel" storage choice.
- **BVH vs uniform grid for GPU ray traversal** — BVH generally beats a pure uniform grid for many-object GPU raytracing
  (uniform grids over-assume uniformity and suffer traversal divergence; hierarchical traversal avoids it), with
  grid-of-BVH-node **hybrids** a known middle ground ([NVIDIA "Thinking Parallel, Part II"](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/);
  [Performance Comparison of BVHs and Kd-Trees for GPU Ray Tracing](https://www.researchgate.net/publication/284233414_Performance_Comparison_of_Bounding_Volume_Hierarchies_and_Kd-Trees_for_GPU_Ray_Tracing)).
  This supports Decision 4b's split: a hierarchical structure (our clip-map for rays, a BVH for edits) over a flat uniform
  macrocell grid.
- **Internal evidence — `experimental/sdf_bench` (untracked throwaway).** The three-technique benchmark ADR 0009 cites is
  the primary evidence for technique C. Its `t3_brick.rs`/`t3_brick.wgsl` already implement the coarse-vs-sculpted brick
  split, the 512³ R8 atlas, the L1/L2 clip-map occupancy pyramid, the hierarchical DDA, and the sorted-record binary-search
  broadphase this ADR ports. The numbers in §Considered Alternatives are its measured output. This ADR **synthesizes the
  existing benchmark** rather than rebuilding it; no new spike was added.

## Open questions for the owner grill — ALL RESOLVED (grill 2026-07-05/06)

1. ~~**Brick granule: 16³-first-then-measure-8³, or 8³ from day one?**~~ **RESOLVED (grill 2026-07-05): the question
   was a units error — the granule is ONE BLOCK (brick edge = `voxels_per_block`, block-denominated), for any density.
   Sub-block bricks are a measured follow-up behind Decision 1's explicit trigger. ADR 0009's "8³" phrasing amended.**
2. ~~**Two broadphases or one?**~~ **RESOLVED (grill 2026-07-05): two broadphases, split confirmed. The edit-side BVH
   of producer AABBs IS built (owner ruled effort doesn't count against it) and REPLACES #63's chunk-bucketing as the
   sole edit broadphase — never alongside it. Stateless (rebuilt per edit, ~1ms at 10k objects); persistent/refit only
   if measured hot. Swap parity-gated. Render broadphase stays pyramid + sorted-record binary search.**
3. ~~**World-fixed occupancy pyramid vs true camera-centred geometry clip-map rings?**~~ **RESOLVED (grill 2026-07-05):
   world-fixed first. The "when is it architecture?" split: residency POLICY is engineering (defer to G4, measured);
   the residency-miss CONTRACT is architecture, decided now — a miss renders the block's coarse form
   (degraded-but-correct), baked into G1's lookup design. Rings are thereby pre-classified as pure eviction policy.**
4. ~~**Retire-or-keep-both the CPU two-layer mesh?**~~ **RESOLVED (grill 2026-07-05): keep-both, permanent, no sunset —
   ADR 0006 constitutionally forces a CPU display path and the parity gate needs its oracle. The real ruling is the
   TERMS (Decision 6): pinned at exactness-not-feature-parity; cosmetic display features are GPU-only by default.**
5. ~~**Is a de-risking spike warranted before G0?**~~ **RESOLVED (grill 2026-07-06): no pre-G0 spike — the benchmark
   answered "does the technique work" and the atlas mechanic is shipped. The ONE integration point no prior work
   exercised — depth-compositing the raymarch with the rasterized overlays (gizmo, fog, view cube, egui) — is promoted
   to a named G1 requirement instead of a spike; `experimental/` throwaways remain the escape hatch if G1 hits genuine
   surprise there.**
