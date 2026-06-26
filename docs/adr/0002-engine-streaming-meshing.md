# ADR 0002 — Engine phase: streaming, meshing & coordinates

- **Status:** Proposed (awaiting sign-off — the forks in "Open questions" need a decision before implementation)
- **Date:** 2026-06-25
- **Sub-ADR of:** [ADR 0001](0001-scene-graph-parts-and-tools.md) ("Scale" section, build-sequence steps 5–7).
- **Issues:** Part of #14. Decomposes #18 (step 5), #19 (step 6), #20 (step 7). Leans on #24 (golden images).

## Context

ADR 0001 committed the **scale** stack — chunked streaming, 64-bit addressing, origin-rebased
rendering, greedy meshing, GPU instancing, out-of-core store, palette/sparse compression, LOD
parked-but-seam-preserved — and flagged that step 6 "likely [needs] its own sub-ADR." **This is
that sub-ADR.** It covers steps 5–7, the **engine phase**, which ADR 0001 itself calls "by far the
largest piece" and "engine-level work."

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
| **Onion-skin fog occupancy** (#12, `OnionFogRenderer`) | Resolves the grid to a **3D R8 occupancy texture** and raymarches it as a cloud; sized to grid dims, capped by max 3D texture dimension. | Fog reads **occupancy, not the mesh** — so it is decoupled from the meshing change, BUT it currently uploads **one whole-grid 3D texture**. At chunk scale it must become **per-chunk occupancy volumes** (or a sparse/region-scoped texture) and the raymarch must march across chunk volumes. Region-scoping (ADR 0001: "seam consumers scope to a region") makes this tractable: fog only needs the active region's chunks. **This is the second-hardest row** — fog is the one consumer that is NOT just "read a resolved grid." |
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
| **E4** | **Make cuboid the default; remove the instanced path** (or keep it as a debug fallback). Per-chunk **onion-fog occupancy** becomes region-scoped volumes; scrubber/diameter/`.vox` export region-scoped. | The fog/consumer rows at chunk scale; the instanced renderer is no longer load-bearing. | #18/#19 (fog row) |
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
- **O6 — Onion fog at chunk scale.** Per-chunk occupancy volumes + cross-volume raymarch is more
  involved than today's single 3D texture. If region-scoping proves insufficient, the fallback is a
  coarser/region-only fog. **Fork: accept possible fog-fidelity scope reduction at canvas scale?**
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
   patching.

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
