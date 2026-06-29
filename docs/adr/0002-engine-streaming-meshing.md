# ADR 0002 — Engine phase: streaming, meshing & coordinates

- **Status:** Accepted (O1–O7 signed off 2026-06-25). **O8** (atlas-UV'd vs per-material mesh geometry) — **resolved DONE 2026-06-25**: the cuboid mesher emits atlas-UV'd geometry (E3c-1), and the **cuboid path is now the DEFAULT** (E3c-2). The instanced path is retained behind `--mesher instanced` as a debug fallback. **Acceptance note:** the goldens were **rebaselined from the cuboid path**, NOT pixel-matched to the instanced path — the ~3–5% full-frame difference (merged-face triangulation, edge AA, procedural-noise phase, surface shading) is expected and acceptable per the matrix below; the switch is gated on feature/material parity (E3b) + a visually-reviewed golden baseline, not pixel-equivalence.
- **Date:** 2026-06-25
- **Sub-ADR of:** [ADR 0001](0001-scene-graph-parts-and-tools.md) ("Scale" section, build-sequence steps 5–7).
- **Issues:** Part of #14. Decomposes #18 (step 5), #19 (step 6), #20 (step 7). Leans on #24 (golden images).
- **Implementation note (2026-06-25):** E1 (coordinate retrofit / origin-rebasing) is **not testable in isolation** — no faraway geometry exists until chunking lands (a single resolved region can't span to a distant node without blowing the cap), so the 64-bit/rebased coordinate work is folded **into** E2 (chunking), not done before it. Revised order: goldens (E0 ✅) → **chunk the instanced renderer *with* the coordinate retrofit** → cuboid mesher (E3, A/B goldens) → region fog → out-of-core.

## Context

ADR 0001 committed the **scale** stack — chunked streaming, 64-bit addressing, origin-rebased
rendering, greedy meshing, GPU instancing, out-of-core store, palette/sparse compression, LOD
parked-but-seam-preserved — and flagged that step 6 "likely [needs] its own sub-ADR." **This is
that sub-ADR.** It covers steps 5–7, the **engine phase**, which ADR 0001 itself calls "by far the
largest piece" and "engine-level work."

> **[CROSS-REF 2026-06-29]** This ADR governs the renderer; the CPU↔GPU authority line it sits under
> is now pinned by [ADR 0006](0006-authoring-truth-and-gpu-boundary.md): the resolved grid is
> CPU-authoritative, the renderer (mesh + fog) is a DISPLAY derivation downstream of resolve, and the
> A/B golden-equivalence discipline this ADR used for the instanced→cuboid switch is the same net
> ADR 0006 requires for any future GPU view-resolve / GPU sculpt path.

The reason a sub-ADR is warranted is not the data model (that was ADR 0001's job) but a **risk**:
the current renderer (`src/renderer.rs`, `VoxelRenderer`) is not a thin draw of cubes — it is a
**pile of shipped features** welded onto the one-instanced-cube-per-voxel pipeline. A naive "rip it
out and greedy-mesh" rewrite would silently regress several of them. ADR 0001 said "greedy meshing
replaces the per-voxel-cube renderer"; before we cash that cheque this ADR (a) re-examines greedy
meshing itself against our actual medium (chiseled 16³ blocks), (b) enumerates every feature the
new path must keep, and (c) proposes an order that is **green at every step** and switches the
render path only behind a verified equivalence net.

The current pipeline draws one instanced unit cube per occupied voxel, capped at
`MAX_DRAWN_INSTANCES` (~450k) on the render side and `MAX_GRID_VOXELS` (6M) on the resolve side.
Both caps are scene-killers at canvas scale (ADR 0001: a 1024³ canvas at density 16 ≈ 4 trillion
voxels) and are the first thing the engine phase must retire.

## Feature-preservation matrix (the crux)

Every feature currently tied to the instanced-cube pipeline, and the constraint it places on the
new mesh/cuboid representation. **This table is the acceptance spec for the render-path switch** —
each row is a golden-image assertion (#24).

| Feature (today) | How it works on the instanced cube | What the cuboid/mesh path MUST preserve |
|---|---|---|
| **Per-face block textures** (6-layer `D2Array`, M7) | Shader picks the array layer from the face's outward normal (`face_layer`); a VS block supplies 6 per-face PNGs, a uniform material replicates one image ×6. | Each emitted **mesh face** must carry its **outward-normal → face-layer** mapping (the 6 VS face slots). A greedy quad or cuboid face spanning many voxels keeps one normal, so the layer is well-defined; the per-face material-array bind group and `MaterialSource::{Procedural,Loaded}` survive unchanged. |
| **Per-voxel texture slice** (BUG 1 fix) | Vertex stage offsets the face UV by `block_local_coord` and divides by `voxels_per_block`, so each voxel face shows its 1/density slice of the block texture (not the whole texture per cube). | This is the **single hardest row.** A merged face spanning N voxels must still tile the texture **once per voxel** across its extent. Emit per-vertex UVs in **block-local voxel units** (UV = `local_voxel_xy`, then `/voxels_per_block` in-shader) and let the sampler tile (`AddressMode::Repeat`) — the merged quad's UV runs `0..N` instead of `0..1`. Greedy/cuboid merges are therefore only legal **within a block boundary OR across whole-block multiples** so the per-voxel slice stays phase-aligned. (Constrains the meshing granularity — see Meshing.) |
| **Position-based grid overlay** (BUG 2 fix) | Fragment derives voxel/block boundary distance from the **absolute world voxel position** (`world_pos + grid_half_extent`), orientation-independent. | Carry the fragment's **absolute voxel position** as a varying on every mesh vertex (it interpolates correctly across a merged face). The overlay math is unchanged; it must read absolute position, never face UV. Origin-rebasing (below) means "absolute position" becomes **chunk-local + chunk world origin** — the overlay must add the chunk origin back, or run in chunk-local space with block phase preserved. |
| **Per-voxel material modulation** (ADR 0001 step 3) | Per-instance `material_id` indexes `material_base_colors`; fragment multiplies lit colour by the material's relative base colour. | `material_id` rides **per cuboid/quad** (a box is one material by construction — see Meshing). Carried as a flat per-vertex attribute (constant across the merged face). Modulation toggle + relative-base-colour uniform survive as-is. A merged face must **never straddle two materials** — the mesher splits on material change. |
| **Layer-range band clip** (#12 scrubber) | Fragment recovers the voxel's Y layer from the instance centre and discards outside `[band_min, band_max]`. | The band clip becomes **per-fragment from the absolute Y voxel layer** (already a varying candidate). A merged quad spanning the band edge must clip mid-quad → keep the **fragment discard** on interpolated absolute-Y (do NOT bake the band into the mesh, or scrubbing forces a re-mesh every frame). Cheaper alternative: clip whole chunks/cuboids by AABB, fragment-discard only the boundary chunk. |
| **`--debug-faces` mode** | Cull-off pipeline + shader colours by outward normal and stripes back-faces to expose winding bugs. | The mesher must emit **correctly wound, outward-normal faces**; debug-faces is the regression check that it does. Keep the cull-off debug pipeline variant; colour-by-normal needs the face normal as a varying (already present). This mode is also the **primary golden image** (#24, deterministic). |
| **Onion-skin fog occupancy** (#12, `OnionFogRenderer`) ✅ **REWORKED to per-chunk, now DEFAULT** (2026-06-25, #28 S5a/S5b) | Resolves the grid to a **3D R8 occupancy texture** and raymarches it as a cloud; sized to grid dims, capped by max 3D texture dimension. | **DONE.** Fog reads **occupancy, not the mesh** (decoupled from meshing). Per-chunk occupancy is now the **default** path: one apron'd `R8` volume per resident chunk, packed into a single small 3D **atlas** (atlas dimension bounded by chunk COUNT, not whole-grid extent), with a metadata uniform of per-chunk world origins; the raymarch (`onion_fog_perchunk.wgsl`) marches in recentred world space and candidate-samples the owning chunk's tile. The 1-voxel apron is filled from the GLOBAL occupancy so trilinear sampling across a chunk seam is **continuous (no banding/seam lines)** — A/B vs the legacy whole-grid path is **0.0000%** on normal scenes, and it still renders fog at scale where whole-grid exceeds `max_texture_dimension_3d` and disables itself. Legacy whole-grid kept as a fallback (`--fog=wholegrid`). Locked by the `onion-fog-perchunk` golden (#24). Region-scoping the resident set is deferred to #20. |
| **Diameter readout / layer scrubber range** (#12) | `widest_run_in_band` scans the sparse occupied list. | Reads occupancy, not the mesh; scope to the active region (ADR 0001). Unchanged in kind; bounded in extent. |
| **`.vox` export** (M8) | Iterates the sparse occupied grid. | Becomes a **streamed / region export** (ADR 0001). Reads occupancy, not the mesh. |

**Two highest-risk rows: the per-voxel texture slice (BUG 1) and the onion-fog occupancy texture.**
Both are why a blind greedy-mesh rewrite is dangerous and why the meshing granularity is
constrained below.

## Decision 1 — Meshing: cuboid (box) decomposition, primary

We adopt **cuboid / box decomposition** (Vintage Story style) as the **primary** meshing direction,
over classic 2D greedy-quad meshing. (ADR 0001 said "greedy meshing"; this sub-ADR refines *which*
greedy strategy, without changing ADR 0001's decision that the per-voxel-cube renderer is replaced
by merged geometry.)

**What it is.** VS renders chiseled blocks not as per-voxel cubes and not as Minecraft-style 2D
greedy quads, but by merging each chiseled block's 16³ voxels into a set of **axis-aligned
cuboids**, each bit-packed into a `uint` (X/Y/Z min+max as 4-bit fields + an 8-bit material index),
then tessellating the exposed faces (`BlockEntityMicroBlock.GenShape` → `ShapeElement`). Source:
deepwiki.com/anegostudios/vssurvivalmod storage systems. It is per-**block** (16³) granularity and
**doubles as compact storage**.

**Why cuboid over 2D greedy quads, for us specifically:**

1. **Domain match.** Our medium *is* chiseled 16³ blocks. VS proves cuboid decomposition on exactly
   this medium at exactly this granularity. 2D greedy meshing is the Minecraft answer to a
   different shape of problem (huge sparse terrain, mostly full blocks).
2. **Material rides for free.** A cuboid is **one material by construction** — it composes cleanly
   with our per-voxel `material_id` (matrix row 4): the box *carries* the id, no per-face material
   split needed within a box. 2D greedy quads must additionally not merge across a material seam;
   the box decomposition enforces single-material regions up front.
3. **Meshing + compression collapse toward ONE representation.** The cuboid packing **is** the
   step-7 palette/sparse compression (ADR 0001 decision 7, issue #20). Instead of "greedy-mesh for
   the GPU" *and* "palette/RLE for storage" as two encodings, the per-block cuboid set is both the
   stored form and the meshing input. This is the strongest reason: it removes a whole
   representation.

**The per-voxel-slice constraint forces the granularity knob** (matrix row 2): a merged face must
keep the texture tiling once-per-voxel and phase-aligned to block boundaries. Therefore:

- **Per-block cuboid merging (safe v1).** Merge only **within** a 16³ block. Every cuboid lies
  inside one block, so the per-voxel slice (UV in block-local voxel units, `Repeat` sampler) is
  trivially phase-correct, and the 6 per-face texture layers are well-defined. **This is the
  recommended v1.** It already collapses a solid 16³ block from 4096 cube instances to ~1–6 cuboid
  faces — the order-of-magnitude win that retires the instance caps.
- **Per-chunk (cross-block) cuboid merging (scale optimization, later).** Merge across block
  boundaries within a chunk for further reduction. **Only legal across whole-block multiples** (to
  keep the slice phase) and only across same-material runs. Deferred; revisit if per-block meshing
  is still GPU-bound at canvas scale.

**Recommendation: per-block cuboid decomposition for v1, per-chunk merge parked behind the same
LOD-style "not impossible" seam.** Material rides on each cuboid (one id per box); the cuboid set
is simultaneously the meshing input and the per-chunk compressed store (step 7).

**2D greedy quads — not rejected forever, just not primary.** If a specific surface (large flat
multi-block walls) proves cuboid decomposition leaves too many coplanar faces, a 2D greedy pass
*over the cuboid faces* is an additive optimization. It is not the foundation.

## Decision 2 — Coordinate model

Per ADR 0001 "Coordinate model (committed)", made concrete:

- **64-bit world addressing.** `i32`→`i64` **block** coordinates + a sub-block **voxel** offset.
  `Voxel.world_position` (today `[f32; 3]`, world-centred voxel-grid coords) is **not** the storage
  truth at scale — it stays the *chunk-local* render coordinate; the addressing truth is
  `(block_i64, voxel_u8_within_block)`.
- **f64 transform composition.** Node transforms (`NodeTransform`, today integer `offset_blocks`)
  compose down the tree in **f64** when affine (rotation/scale) lands; integer-translation v1
  composes exactly in i64. ADR 0001's `for_each_leaf` world-offset walk is the composition site.
- **Origin-rebased (camera-relative) f32 rendering.** Each chunk's model matrix is
  `chunk_world_origin − camera_floating_origin`, computed in i64/f64 and **downcast to f32 per
  frame**, so f32 precision is always high near the viewer (no far-lands jitter). Per-chunk
  meshing/resolution always works in small **chunk-local f32**. The grid-overlay varying (matrix
  row 3) must therefore carry chunk origin to recover absolute block phase.
- **`(chunk_coord, lod)` keying.** The chunk cache and render items are keyed by `(chunk_coord,
  lod)` exactly as ADR 0001's parked-LOD seam requires; `resolve_region(aabb, lod)` already carries
  the always-0 `lod`. Per-chunk resolution is not assumed globally uniform (a mixed-LOD scene stays
  representable). **This ADR does not build LOD** — it only honors the seam.

## Decision 3 — Streaming

Per ADR 0001 step 6 / issue #19:

- **Chunk size.** Recommend **a small whole number of blocks per chunk axis — start at 4 blocks
  (= 64 voxels/axis at density 16)**, i.e. one chunk = 4³ blocks. Rationale: a chunk must be a
  whole-block multiple (per-voxel-slice phase + per-block cuboid alignment); 4 blocks keeps a chunk
  mesh small enough to re-mesh on a single-block edit cheaply, while large enough that draw-call
  count stays sane. **Open question O3** parks the exact number for measurement.
- **On-demand resolve + cache/invalidate.** A chunk is resolved via `resolve_region(chunk_aabb,
  lod)`, cached by `(chunk_coord, lod)`, and **invalidated only when an edit's world-AABB
  intersects it** (ADR 0001). `MAX_GRID_VOXELS` becomes a **per-chunk** bound, not a scene guard.
- **Spatial index over node AABBs.** An AABB index over node world-bounds (updated on edits) gates
  the tree walk so resolving one chunk costs ~the nodes touching it, not the whole village graph.
- **Frustum culling.** Per-chunk render items, frustum-culled against the chunk world-AABB; resident
  chunks bounded by render distance → constant memory regardless of scene size.
- **Out-of-core store (step 7 / #20).** Authored/unique chunk data (the per-chunk cuboid sets)
  streams from **disk** with eviction, so scene size decouples from RAM. The cuboid set is the
  stored form (Decision 1).
- **Region-scoped seam consumers.** Onion fog, layer scrubber, diameter readout act on the **active
  region**, not the canvas; `.vox` export becomes a streamed/region export. (This is what makes the
  fog row tractable — matrix row 7.)

## Decision 4 — Safe incremental order (the big question)

**Principle: the app is green and visually identical at every checkpoint, and the render-path
switch happens only behind a verified pixel-equivalence net.** Two structural choices drive the
order:

1. **Land the golden-image net FIRST** (#24), before touching the renderer. Greedy/cuboid meshing
   *must not change pixels* (except where a feature row explicitly allows it). Without the net we
   are flying blind.
2. **Chunk the EXISTING instanced renderer before meshing.** Decouple "where geometry comes from"
   (chunks) from "how a chunk is drawn" (instanced vs meshed). This lets us retire the 450k/6M caps
   and prove streaming/culling/origin-rebasing **while still drawing known-good instanced cubes** —
   so if a streaming bug appears we know it is not the mesher. The mesher then drops in **behind a
   flag, per-chunk, A/B against the instanced draw**, and we flip the default only when goldens
   match.

### Checkpoints

| # | Checkpoint | Verifies | Issue |
|---|---|---|---|
| **E0** | **Golden-image net.** Commit reference PNGs from the headless `shot` tool (deterministic `--debug-faces` + a few shapes, the full feature matrix where capturable: overlay on/off, a band clip, two materials) + a re-render-and-compare-within-tolerance test. | The regression net exists *before* any renderer change. Every matrix row gets a golden. | **#24** (do first) |
| **E1** | **Coordinate model, single region.** Switch world addressing to i64 block + sub-voxel; f64/i64 transform composition; camera-relative (origin-rebased) f32 render matrices. **Still per-voxel-cube instanced, still one region.** Goldens unchanged. | The coordinate retrofit in isolation — no jitter, no visual change. The hardest-to-retrofit piece, proven before geometry changes. | #18 (part) |
| **E2** | **Chunk the instanced renderer; retire the caps.** Partition the single region into chunks; per-chunk instance buffers keyed `(chunk_coord, lod=0)`; on-demand resolve + cache/invalidate; spatial index; frustum cull. **Geometry is still instanced cubes** — just chunked. `MAX_DRAWN_INSTANCES`/`MAX_GRID_VOXELS` become per-chunk bounds. | Streaming, culling, invalidation, the spatial index — all on **known-good geometry**. Large scenes that previously hit the cap now render. | #19 (most of it) |
| **E3** | **Per-block cuboid mesher behind a flag, A/B.** Add the per-block cuboid decomposition as an **alternate per-chunk render item**, selected by a flag (`--mesh=cuboid` / runtime toggle), drawn alongside/instead of the instanced path. Re-implement the feature matrix on the mesh path (face-layer from normal, per-voxel slice via block-local UV + `Repeat`, absolute-position overlay varying, per-cuboid material, fragment band clip, debug-faces winding). | **Pixel-equivalence (goldens, E0) between instanced and cuboid paths**, row by row. This is where the matrix is cashed. Flag means a regression never ships — flip default only when green. | #18 (meshing) |
| **E4** | **Make cuboid the default** ✅ (2026-06-25, E3c-2; instanced KEPT as a `--mesher instanced` debug fallback, not removed). **Per-chunk onion-fog occupancy ✅ now the DEFAULT** (2026-06-25, #28 S5a behind `--fog=perchunk`, S5b flipped the default + added the `onion-fog-perchunk` golden; legacy whole-grid kept as `--fog=wholegrid`). Scrubber/diameter/`.vox` export region-scoping moved to #20 — *still to do.* | The fog/consumer rows at chunk scale; the instanced renderer is no longer load-bearing. | #18/#19/#28 (fog row) |
| **E5** | **Out-of-core store + the cuboid set AS the compressed form.** Disk-backed `(chunk_coord, lod)` store with eviction; the per-block cuboid/palette packing is both stored and meshed (Decision 1 §3). | Scene size decoupled from RAM; meshing+compression unified. | **#20** |
| **(later)** | Per-chunk cross-block cuboid merge (whole-block-multiple, same-material). Optional 2D greedy pass over cuboid faces. | Further draw-call reduction *if* E4 is GPU-bound. | new sub-issue |

### New sub-issues warranted

- **#24 must precede #18** (today it reads as parallel). Recommend re-scoping #24 as a **blocker**
  of #18/#19, and broadening its goldens to the full feature matrix, not just `--debug-faces`.
- **Split #18**: E1 (coordinate retrofit, instanced) and E3 (cuboid mesher behind a flag) are
  independently shippable green checkpoints. Suggest a new sub-issue "#18a coordinate model
  (instanced)" and "#18b cuboid mesher (flagged A/B)".
- **Split #19**: E2 (chunk the *instanced* renderer) is a green checkpoint *before* any meshing;
  the fog-region-scoping (E4) is a distinct piece. Suggest a sub-issue for "chunk the instanced
  renderer + retire caps" separate from "region-scope fog/export."
- **New issue: per-chunk onion-fog occupancy** (matrix row 7) — the one consumer that is not "read a
  resolved grid"; deserves its own ticket under #19.

## Current status — safe ordering recap (updated 2026-06-26)

This section was added to fully absorb **#22** ("draft a sub-ADR decomposing steps 5–7 into a SAFE
incremental order that preserves the renderer's features"). The decomposition itself lives above
(Decision 4 checkpoints E0–E5 + the feature-preservation matrix); this recap restates the order in
dependency sequence with **current DONE/REMAINING status**, and records the one piece of the original
framing that has since changed.

**Framing change — the instanced renderer is GONE (resolved context).** ADR 0001 and the E0–E4
checkpoints above were written around an *instanced-cube-first* strategy: chunk and origin-rebase the
**known-good instanced renderer** first (E1/E2), then introduce the cuboid mesher behind a flag and
A/B it against the instanced draw (E3), then flip the default (E4). That strategy did its job and is
now spent. The cuboid box-decomposition mesher reached full feature/material parity, became the
default (E3c-2), and the legacy one-cube-per-voxel instanced renderer was then **deleted outright**
(`31a9383`, 2026-06-26): `VoxelRenderer`, `VoxelInstance`, `MesherChoice`, the `--mesher`/`--instanced-via-chunks`
flags and `src/shaders/voxel.wgsl` are gone. **Consequence:** the **cuboid path is now the SOLE render
path**, so #22's original "preserve the *instanced* renderer's features" wording no longer applies as
written. The correct standing invariant is: **the cuboid path must RETAIN the full feature matrix
through the remaining streaming work** — there is no longer a second path to A/B against, so the
golden-image net (E0, #24) is the sole regression guard from here on.

**Safe order in dependency sequence, with status:**

1. **E0 — Golden-image net (#24).** ✅ **DONE.** Reference PNGs from the headless `shot` tool + a
   re-render-and-compare-within-tolerance test. Now the *only* regression net (no instanced A/B left).
2. **E1 — 64-bit coordinate model + origin-rebased rendering (Step 5 remainder, #18).** ✅ **DONE.**
   i64 block addressing (S4a) + camera-relative (origin-rebased) f32 render matrices (S4b): near-camera
   goldens pixel-identical, far geometry byte-identical.
3. **E2 — Chunked rendering + frustum cull, caps retired (Step 6, #19).** ✅ **DONE.** Per-chunk render
   items keyed `(chunk_coord, lod=0)`, on-demand resolve, frustum cull; the 450k `MAX_DRAWN_INSTANCES`
   cap is retired.
4. **Deep chunked resolve beyond the 6M cap (#27).** ✅ **DONE.** Lazy per-chunk resolve + cache +
   edit-AABB invalidation + spatial index; `MAX_GRID_VOXELS` is now a per-chunk bound.
5. **E3 — Cuboid (box-decomposition) mesher + texture atlas (#18 meshing, #26 tests, O8 atlas).**
   ✅ **DONE and now the SOLE path.** Full feature-matrix parity (per-voxel + loaded-block per-face
   textures, multi-material atlas, layer band clip, debug-faces winding, per-object grids); goldens
   rebaselined from the cuboid path; instanced path subsequently removed (`31a9383`).
6. **Per-chunk onion-fog occupancy (#28).** ✅ **DONE and now the DEFAULT.** Per-chunk occupancy
   volumes packed into one small 3D atlas + apron-from-global-occupancy raymarch; A/B-identical
   (0.0000%) to the legacy whole-grid fog on normal scenes; legacy kept as `--fog=wholegrid`. (Matrix
   row 7, formerly the second-highest risk, is closed.)
7. **E5 — Out-of-core store + palette/sparse (Step 7, #20).** 🟡 **IN PROGRESS — building blocks
   landed, final wiring remains.** Done: per-chunk material palette + sparse storage (S6a, lossless
   round-trip); standalone `DiskChunkStore` (S6b); region-scoped `widest_run_in_band` + `.vox` export,
   parity-proven vs whole-grid (S6d); consumers read `placed_region_dimensions` not the assembled grid
   (S6c-1); disk-spill of resident chunks wired into `ChunkResolveCache` behind an opt-in resident cap
   (Step 3, `534849b`). **Remaining (each preserves the cuboid feature matrix; gated by the E0 goldens
   + interactive smoke-test):**
   - **Wire the region-scoped consumers into the live app** ("Step 2"): route the interactive diameter
     readout and `.vox` export through the region-scoped / rebased path (`ChunkResolveCache::vox_export`)
     instead of the monolithic `resolve_scene`/`from_grid` path — far-offset scenes currently lose the
     voxel-centre `.5` on export. Parity-proven; interactive-only verification.
   - **Per-chunk GPU residency + remove the monolithic assembly** ("Step 4"): make the cuboid renderer
     rebuild only dirty chunks and consume per-chunk grids, then delete the monolithic
     `resolve_region`/`resolve_scene` whole-grid assembly. This is what actually decouples peak RAM from
     scene size (disk-spill or assembly-removal *alone* does not); the borrow-returning whole-region
     gather methods will need a reload-then-borrow pass under a real resident cap. Dominated by
     interactive paths the static goldens cannot exercise → needs the windowed app for smoke-testing.
     **[STATUS 2026-06-29] The dirty-chunk PLAN is computed but NOT yet consumed by the live renderer.**
     `renderer::incremental_rebuild_plan` exists (and is unit-tested via the `store.rs` CPU render-cache
     harness), but the windowed app still re-meshes WHOLESALE on every edit (`VoxelRenderer::rebuild_from_scene`,
     driven from `main.rs`); no `incremental_rebuild_from_chunks` consumer is wired up. Consuming this plan
     is the recorded next CPU perf step (see [ADR 0003](0003-foundation-rework.md) §4 / migration step 8 and
     [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) "Next (CPU)").
8. **(later) Per-chunk cross-block cuboid merge + optional 2D greedy pass over cuboid faces.** ⛔
   **PARKED.** Whole-block-multiple, same-material only; pursue only if E5 proves GPU-bound. **LOD
   stays parked** (seam preserved via the `(chunk_coord, lod)` key; O7).

**Net:** of #22's steps 5–7, everything except the final out-of-core wiring (#20 Step 2 + Step 4) is
DONE, and the safe-ordering risk that motivated #22 — switching the render path without regressing the
feature matrix — is fully retired (the switch happened, was golden-verified, and the old path is gone).

## Risks + open questions (need sign-off — this is Proposed)

- **O1 — Cuboid vs 2D greedy as primary.** This ADR recommends cuboid decomposition (domain-matched,
  VS-proven, unifies with compression). 2D greedy is the more conventional choice and has more
  off-the-shelf references. **Fork: confirm cuboid-primary, or prefer 2D greedy quads?** (Affects
  E3/E5 and whether meshing and storage unify.)
- **O2 — Meshing granularity v1.** Recommend **per-block** merging for v1 (keeps the per-voxel-slice
  phase trivially correct), per-chunk merge parked. **Fork: accept per-block v1, or push for
  per-chunk merge in the first cut** (more reduction, but complicates the slice + invalidation)?
- **O3 — Chunk size.** Recommend starting at **4 blocks/axis** (64 voxels at density 16), tuned by
  measurement. **Fork: agree to measure-and-tune, or fix a size now?**
- **O4 — Order: net + chunk-instanced-first.** Recommend E0 (goldens) → E1 (coords) → E2 (chunk the
  *instanced* renderer, retire caps) → E3 (cuboid behind a flag, A/B) → E4 (default + fog) → E5
  (out-of-core). The alternative is "mesh first, then chunk," which couples two big changes and
  loses the known-good A/B baseline. **Fork: confirm chunk-instanced-before-meshing.**
- **O5 — Per-voxel slice across merged faces.** The recommended approach (block-local UV in voxel
  units + `Repeat` sampler, runs `0..N`) is the load-bearing assumption that makes per-block cuboid
  meshing keep BUG 1's per-voxel texturing. It needs a spike at E3 to confirm pixel-equivalence
  with the current per-instance slice (it should be exact, but it is the highest-risk row). **Not a
  fork, a flagged risk** — E3 gates on the golden.
- **O6 — Onion fog at chunk scale. ✅ RESOLVED (2026-06-25, #28 S5a/S5b) — reworked to per-chunk,
  no fidelity reduction.** Per-chunk occupancy volumes packed into one small 3D atlas (dimension
  bounded by chunk COUNT, not whole-grid extent) + a candidate-sample raymarch
  (`onion_fog_perchunk.wgsl`). The fork ("accept possible fog-fidelity scope reduction at canvas
  scale?") was **NOT taken**: a 1-voxel apron filled from the GLOBAL occupancy makes trilinear
  sampling continuous across chunk seams, so per-chunk is **A/B-identical to the legacy whole-grid
  fog (0.0000%)** on normal scenes while still rendering fog past `max_texture_dimension_3d` (where
  whole-grid disables itself). Per-chunk is now the **default**; whole-grid kept as `--fog=wholegrid`.
  Locked by the `onion-fog-perchunk` golden. (Region-scoping the resident set → #20.)
- **O7 — LOD stays parked.** This ADR builds none; it only honors the `(chunk_coord, lod)` seam.
  Confirm LOD remains out of scope for the engine phase. (ADR 0001 decision 8 — restated for
  completeness.) Note even VS itself runs **per-frame frustum cull + distance LOD** ("draw or cull
  based on distance and the mesh's level of detail"), which reinforces that parking LOD *while
  keeping the `(chunk, lod)` seam* is the right call — and that minimal distance-based chunk
  dropping may be worth a small follow-up eventually (cheap, rides the same seam).

## Consequences

- **Positive:** the instance caps die; the renderer scales to the canvas; meshing and step-7
  compression collapse into **one** per-block cuboid representation (VS-proven); every shipped
  feature has an explicit preservation contract and a golden assertion; the render-path switch is
  reversible behind a flag, so a regression never ships.
- **Negative / cost:** this is the heaviest milestone in the project (ADR 0001's own assessment).
  Re-implementing the full feature matrix on the mesh path (especially the per-voxel slice and the
  per-chunk fog) is real work; the coordinate retrofit (E1) touches `Voxel`, the renderer, and the
  camera. The golden-image net (E0) is upfront cost that pays for itself the first time it catches a
  meshing regression.

## Performance techniques borrowed from Vintage Story

VS solves exactly our problem class (chunked, chiseled-voxel world at scale). These are the
performance techniques worth lifting, each mapped to our engine steps. Sources:
[Render_Stages](https://wiki.vintagestory.at/Modding:Render_Stages),
[Texture_Atlas](https://wiki.vintagestory.at/Modding:Texture_Atlas),
[storage systems](https://deepwiki.com/anegostudios/vssurvivalmod/5-storage-and-container-systems).

1. **Async meshing on a worker thread (request → complete queue).** VS tessellates chunks on a
   separate `tesselateterrain` thread: a request queue feeds tessellation, finished meshes land on a
   completion queue, and the main thread only does the GPU upload — rendering **never blocks on
   meshing**. → **Adopt at E2/E3 (issue #19):** mesh (cuboid-decompose) chunks **off the render
   thread**, so editing/orbiting never hitches during a chunk rebuild. Request→complete queue with
   main-thread upload; the `(chunk_coord, lod)` cache is the completion sink.

2. **Texture atlas → one mesh / one draw per chunk.** VS packs all block textures into one atlas
   (default 4096×2048); a whole chunk's blocks then combine into a **single `MeshData`** ≈ **one
   draw call per render pass**. → **Refines our material design.** Today we bind per-material
   (`MaterialSource`) and modulate by the step-3b `material_base_colors` array. At scale, instead
   pack material textures into an **atlas** and emit **atlas UVs per cuboid**, so a chunk of
   mixed-material boxes is **one draw**. This is the draw-call answer at canvas scale. **Interaction
   with per-voxel `material_id` (matrix row 4):** `material_id` would resolve to an atlas sub-rect at
   mesh time rather than indexing a uniform array at draw time — the id is consumed by the mesher,
   not the shader. The per-voxel slice (BUG 1) composes by slicing *within* the atlas sub-rect.
   **Open refinement O8 (flagged): does the cuboid mesher emit atlas-UV'd geometry from E3, or do we
   keep per-material binding for v1 and atlas later?** Atlas-from-E3 is more work but is the real
   draw-call win; per-material-first is the smaller green step.

3. **Dirty-whole-chunk invalidation + mesh cache.** VS: "if any block in the chunk is dirty, the
   **entire chunk** is re-tessellated." → **Validates our coarse invalidation granularity
   (Decision 3):** an edit dirties only the chunks its world-AABB intersects, and we re-mesh just
   those — whole-chunk, not per-voxel-diff. Simpler beats finer-grained; no incremental mesh
   patching. **[STATUS 2026-06-29] This is the TARGET, not yet the live behavior:** the dirty-chunk
   plan (`incremental_rebuild_plan`) is computed but the live renderer still re-meshes wholesale (see
   E5 Step 4 above). The resolve CACHE already invalidates only the dirty chunks; the GPU MESH rebuild
   does not yet honor that.

4. **Pooled GPU mesh buffers keyed by (render pass × atlas)** (VS `MeshDataPoolManager`s). → **Adopt
   at E2/E5:** reuse vertex/index allocations across frames and chunks via a pool keyed by
   (pass × atlas) instead of alloc/free churn at thousands of chunks. Complements the out-of-core
   store (E5): evicting a chunk returns its buffers to the pool.

5. **Per-vertex baked AO / light at mesh time.** Compute ambient-occlusion / light **once per
   re-tessellation** and store it **in the vertices**, not per-frame. → **Optional cheap quality win
   riding on the cuboid mesher (E3+):** bake AO into mesh vertices since we already re-mesh
   whole-chunk on edit. *(Not explicitly confirmed in the cited VS docs — standard for this engine
   class; noted as a borrowed pattern, not a VS citation.)* Out of scope for v1; a clean add later.

6. **Separate opaque vs OIT-transparent passes** (VS: Opaque → OIT). → **We already do this** — the
   onion-skin fog is its own translucent fullscreen pass (`OnionFogRenderer`), composited after the
   opaque resolve. VS validates keeping transparency in a **dedicated pass** rather than alpha-blended
   inline; our split stays as-is through the engine phase (matrix row 7).

**New open question:** **O8 — atlas-UV'd cuboid geometry vs per-material binding for v1** (technique
2). Recommend per-material binding through E3 (smaller green step, preserves the step-3b path),
atlas as the draw-call optimization once cuboid meshing is proven. **Fork: atlas from E3, or later?**
**RESOLVED (2026-06-25): atlas later** — per-material binding carried through E3a/E3b parity; the
atlas landed in E3c-1, after which the cuboid path was made the default (E3c-2) with goldens
rebaselined from the cuboid path (not pixel-matched).
