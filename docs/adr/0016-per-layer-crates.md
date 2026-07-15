# ADR 0016 — Per-layer crates: the five architecture layers become compile-enforced crate boundaries

- **Status:** **Proposed (2026-07-14)** — grilled and specified; extraction map + slice order in
  `docs/design/per-layer-crates-extraction-map.md`. Not yet executed.
- **Date:** 2026-07-14
- **Layer:** repo shape. **Supersedes, in part, ADR 0014's "keep the app fused" ruling** — see below.
- **Supersedes in part:** ADR 0014 (substrate) deliberately rejected splitting display/shell/
  workers/UI into crates, on the grounds that a crate must enforce *a dependency law worth
  compile-enforcing* and "taxonomy ahead of need" is not one. That reasoning stands; what changed
  is the recognition that the architecture's **downward-only data-flow law is itself exactly such a
  law**, and the owner reprioritized toward navigability. ADR 0014's bar is met, not lowered.

## Context

`docs/architecture/README.md` defines five layers — Document → Evaluation → Derivations → Work,
with the Shell on top — and law: *"Data flows downward only; nothing lower ever writes upward."*
That is a dependency law a crate boundary compile-enforces. The 53k-LOC `voxel_worker` app crate
holds all five layers in one flat module namespace, so the law lives in prose and code review, not
in the compiler.

A dependency survey (real `use` edges, comments and tests excluded) found the code is **already
~90% layered**: the true downward-only violations are four misplaced shared helpers, not deep
architectural knots —
1. `store → renderer` (`incremental_rebuild_plan`, a residency planner);
2. `two_layer_store`/`cuboid → cuboid_mesh` (the cell-key bit codec);
3. `block_palette → workers` (`decode_rgba`, an image util);
4. `intent → app_core` (**test-only**).

Each is a helper sitting one layer too high. Fix those four and the graph is a clean DAG matching
the five chapters. The owner also directed, during the grill, that **mega-files be broken into
folder-organized modules** as they move — several sinks exceed 3–4k lines (`cuboid_mesh` 4955,
`brick_field` 3901, `renderer` 3461, `two_layer_store` 3365).

## Decision

Decompose `src/` into one crate per layer, on top of the existing `substrate`/`camera`/`raycast`.
Each crate's `lib.rs` **states its law and cites its architecture chapter**; module docs carry the
rationale/citation voice established by substrate/camera/raycast — this documentation standard is
**definition-of-done**, not a later pass (owner direction 2026-07-14).

```
substrate · camera · raycast          CS/math + graphics math (done)
      ▲
voxel_core        core_geom (material vocabulary + the CellKey codec), Voxel/VoxelGrid value types,
                  RecentreVoxels, spatial_index (domain LeafSpatialIndex), units/Measurement, SDF math
      ▲
document          scene graph · producers · sketch · the VoxelProducer trait + SdfShape · debug_clouds ·
                  intents · command.   Law: truth imports no evaluation/display/wgpu.
      ▲
evaluation        the evaluator · two-layer store · residency/store · chunk cache/storage · cuboid
                  decomposition · incremental_rebuild_plan · measurement queries.
                  Law: consumes the document, produces the classified boundary set.
      ▲
display           renderer · cuboid mesh · brick field · raymarch · texture atlas · block palette ·
                  assets · engagement/routing.   Law: the ONLY crate that links wgpu.
interchange       vox_export (.vox).   Law: a headless sink — consumes the boundary set, never wgpu.
      ▲
work              workers · generations · staleness discipline.
      ▲
voxel_worker      app_core (composition root) · panel · settings · gpu · main + shot bins.
```

Grill rulings that shaped the seams (full rationale in the extraction map):

1. **`voxel.rs` splits.** Value types + SDF math → `voxel_core`; the `VoxelProducer` trait and the
   `SdfShape` producers → `document`. The foundation carries vocabulary, not a behavior contract.
2. **The cell-key codec is foundational vocabulary, not a mesh detail.** `compose_cell_key` /
   `clean_block_id` / the overlay bit are pure `u16` packing consumed by both evaluation and every
   display sink → `voxel_core`/`core_geom` as `CellKey`, renamed off the mesh. Too small/un-named
   for substrate.
3. **`cuboid` → evaluation** (the boundary-box decomposition; display consumes `VoxelBox` downward).
4. **`vox_export` → its own headless `interchange` crate**, not the `display` crate — a headless
   export must not link the GPU (law 4). This is the boundary law that earns interchange a crate.
5. **`display` is ONE crate with internal `brick/`/`mesh/`/`atlas/` folders**, not sub-crates: mesh
   and brick *interoperate by design* (the engagement orchestrator switches between them), so there
   is no downward law between them — mesh-vs-brick is a module boundary; wgpu-vs-below is the crate
   boundary.
6. **Measurement queries fold into `evaluation`** (its exact-query surface), not display/a crate.
7. **`incremental_rebuild_plan` → evaluation** (residency/targeted-dirtying). It is set-difference
   glue, not a substrate-worthy structure — the eviction semantics are the domain content.
8. **Materials:** vocabulary (`MaterialChoice`/`BlockId`) → `voxel_core`; `block_palette` (wgpu
   thumbnails/atlas) → `display`; `decode_rgba` → `assets` (inside display). `assets` stays a
   display folder until a second consumer appears.
9. **Proof (chapter 05) is not a crate:** parity tests travel as `#[cfg(test)]` with the code they
   guard; `shot` stays a `voxel_worker` bin (needs the whole stack + a GPU).

**No mega-files:** components break into folder-organized submodules within their crate as they land.

## Consequences

- **Untangle before cutting.** A first phase relocates the four misplaced helpers and splits
  `voxel.rs` — all inside the current crate, each gated — so the graph is a clean DAG before any
  crate is created.
- **Bottom-up slices:** untangle → `voxel_core` → `document` → `evaluation` → `display` (breaking its
  mega-files into folders) → `interchange` → `work` → the `voxel_worker` shell thins to the
  composition root. Each slice carries the full gate baseline + per-crate clippy/test CI gates, as
  substrate/camera/raycast do.
- **The eight laws become partly compiler-checked:** "CPU owns truth" (interchange/evaluation cannot
  import wgpu), "the document is a program" (document cannot import its own derivations), "one door"
  (intents live below the shell). A future upward edge fails to compile instead of passing review.
- This is a multi-slice epic larger than substrate; it is proposed, to be executed incrementally,
  and can be paused between slices with the tree green (each crate cut is independently valuable).

## Execution note — seam correction (2026-07-15, during Phase 4a)

Ruling #5 above described the engagement **orchestrator** as living inside the `display` crate ("the
engagement orchestrator switches between [mesh and brick]"). Cutting `display` revealed that is wrong:
`DisplayOrchestrator` (the old `src/display/orchestrator.rs`) **owns `GeometryWorker` + `BrickWorker`
as fields, spawns and dispatches them, and calls `build_brick_rebuild`** — all WORK-layer concerns. It
is a work-layer coordinator that drives the display sinks downward, not a display component. The
original dependency survey missed this upward edge because the worker types reach the orchestrator via
**flat crate-root re-exports** (`crate::BrickWorker`), which a `crate::workers::` grep does not catch —
a caution for the remaining cuts: audit crate-root re-exports, not just `crate::<module>::` paths.

Correction applied: the display crate holds only the seven GPU-sink modules + `assets/` + `shaders/`.
The orchestrator + routing stayed in the app crate, renamed `src/display/` → `src/engagement/` (the
extern `display` crate would otherwise collide with a local `mod display`), and are placed at the
Phase-6 **work**-crate cut (orchestrator → work; routing → work-or-display, decided then). Ruling #5's
"display is ONE crate, mesh/brick are folders not sub-crates" stands unchanged; only the orchestrator's
home moves. So "display is the only crate that links wgpu" is now compile-true for the sinks, while the
shell (gpu.rs/main/panel) and the engagement coordinator still link wgpu until the work/shell split.
