# Per-layer crates extraction map (2026-07-14)

**Provenance:** owner direction — "move most if not all of the files in `src/` out into crates,
if only to make the connections between components easier to understand," followed by a
grill-with-docs session that resolved every contested seam. Decision record: `docs/adr/0016`.
The boundary law per crate is the architecture chapter it implements (`docs/architecture/`).

**Status (2026-07-15): Phase 0 COMPLETE; Phases 1–6 LANDED — ALL 8 LAYERS CUT (voxel_core, document,
evaluation, display, interchange, work; on top of substrate/camera/raycast). Only Phase 7 remains:
carve the shell mega-files `app_core`(3090)/`panel`(2060) per the no-mega-files rule.** Grilled; ADR
written.

**Phase 6 landed** (`bffe329`): cut **work** = `workers/*` + `engagement/*` (deps display/interchange/
evaluation/document/voxel_core/substrate + wgpu/profiling; wgpu INTENTIONAL per owner — see the ADR law
revision). Enforced boundary: work imports no shell. `build_brick_rebuild`/`BrickDisplayInstall` were in
`workers/brick.rs` (not shell), so no upward edge. Recon 110+21+93+93+64+18+29=428. The shell
(`voxel_worker`) now = `app_core`, `panel`, `settings`, `gpu`, `main`+`shot` bins, lib.rs facade, 2
parity-test modules.

**Phase 5 landed** (`693d650`): cut **interchange** (`vox_export`, the headless `.vox` sink; deps
evaluation/document/voxel_core, ZERO wgpu — the no-wgpu law that earns it a crate holds). Recon
139+21+93+93+64+18=428.

**Phase 6 DESIGN FORK (open, needs owner):** the survey found `workers/geometry.rs` AND
`engagement/orchestrator.rs` both hold `wgpu::Device`/`Queue` directly (the geometry worker builds GPU
meshes on its thread; the orchestrator owns the device + the display renderers). So a `work` crate
CANNOT be wgpu-free — which revises the ADR's headline "display is the only crate that links wgpu."
Options: (a) cut a `work` crate that DOES link wgpu (workers + engagement); (b) fold work/engagement
into the shell (no separate work crate — the shell legitimately owns the device); (c) decouple the
workers from the device (pass it per-call) so the work layer purifies — a real refactor. No cycle
exists (workers reference engagement only in doc-comments; orchestrator→workers via flat re-exports).

**Phase 4b–4d landed** (`c0e4b9f`, `3f73f27`, `b7963c1`): the three display mega-files carved into
folders — `cuboid_mesh`(4907)→`mesh/` (geometry/builder/two_layer/emit/pipeline/tests, module renamed
`cuboid_mesh`→`mesh`); `brick_field`(3901)+`brick_raymarch`(2696) MERGED→`brick/` (clipmap/occupancy/
record/build/incremental/atlas/gpu_record/raymarch/cpu_march/tests, modules collapsed to one `brick`);
`renderer`(3389)→`renderer/` (materials/view_cube/chrome/lines/gizmo/grid/points/infinite_grid/targets/
onion/tests, name kept so zero consumer churn). No non-test submodule is a mega-file. Test counts
unchanged (=428) throughout; all gates incl. GPU golden/parity green.

**Phase 4a landed** (`04f0a37`): cut **display** — the ONLY crate that links wgpu — moving the seven
GPU-sink modules (`renderer`, `cuboid_mesh`, `brick_field`, `brick_raymarch`, `texture_atlas`,
`block_palette`), `assets/`, and `shaders/`. The mega-files moved WHOLE; carving them into folders is
Phases 4b–4d (decomposed because a cut + three ~4k-line carves exceeds one agent's ~200k budget). Recon
157+21+93+93+64=428.
**SEAM CORRECTION (discovered in 4a):** the old `src/display/` orchestrator+routing are NOT display —
`DisplayOrchestrator` OWNS `GeometryWorker`+`BrickWorker` and drives them, so it is WORK-layer. The
original ADR-0016 survey missed this upward edge because the worker types are imported via flat
crate-root re-exports (`crate::BrickWorker`, not `crate::workers::BrickWorker`). They were kept in the
app crate, renamed `src/display/` → `src/engagement/` (to avoid colliding with the new extern `display`
crate), and will be placed at the Phase-6 work-crate cut (orchestrator → work; routing → work-or-display,
decide then). So the file→crate table below is corrected: orchestrator/routing move OUT of the display row.

**Phases 1–3 landed** (all gated + pushed): `199ad8d` cut **voxel_core** (the foundational value
vocabulary); `90a69f2` cut **document** (the authored-TRUTH layer); this Phase 3 commit cut
**evaluation** (the one evaluator). Phase 3 also carved the two evaluation mega-files into folders:
`store.rs` (2299) → `store/` (`mod`/`key`/`cache`/`rebuild_plan`/`tests`) and `two_layer_store.rs`
(3366) → `two_layer_store/` (`mod`/`chunk`/`classify`/`builder`/`resident_cache`/`stream`/`tests`).
The one upward test edge — a brick-pipeline perf probe that packs GPU records from two-layer
chunks — was relocated up into the app crate's `brick_raymarch.rs` (it names DISPLAY types the
evaluator's law forbids), reaching the classifier via the public `evaluation::two_layer_store`
path. The dense `Store::resolve_region` / `resolve_region_two_layer` oracles gate behind
`evaluation/oracle`; the `expand_resident_chunks_into_grid` cross-crate test oracle behind
`evaluation/test-support` (the app's dev-dependency turns both on).

**Phase 0 landed 2026-07-14** (`b7d3c13`→`521c216`, all gated + pushed): the four untangle
relocations are done and the module graph is now a clean DAG.
1. `b7d3c13` — cell-key codec `cuboid_mesh` → `core_geom` as the `CellKey(u16)` newtype.
2. `8cb14b6` — `incremental_rebuild_plan` `renderer` → `store` (retired `store → renderer`).
3. `604ade0` — `decode_rgba`/`DecodedRgba` `workers/scan` → `assets::decode` (retired `block_palette → workers`).
4. `521c216` — `voxel.rs` split into `voxel/value.rs` (foundational) + `voxel/producer.rs`
   (document-bound), re-exported from `voxel/mod.rs` so call sites are unchanged; `mesh_cell_key`
   folded onto `Voxel::cell_key()` in the value half (retired the last `cuboid → cuboid_mesh` edge);
   the `AppCore`-importing dispatch test moved `intent` → `app_core` (retired the test-only
   `document → shell` edge). Call-site paths kept as `crate::voxel::*` via explicit re-exports — the
   crate cut rewrites them to the real `voxel_core::`/`document::` paths in one ast-grep pass, so the
   split isn't churned twice.

## The dependency law

Each crate boundary carries the architecture's downward-only flow law. A crate may import only
crates below it in this chain; an upward `use` fails to compile:

```
substrate · camera · raycast
      ▲
   voxel_core
      ▲
   document
      ▲
   evaluation
      ▲            ▲
   display      interchange        (parallel sinks over evaluation; display links wgpu, interchange never does)
      ▲            ▲
   work
      ▲
   voxel_worker (shell: composition root + bins)
```

## File → crate assignment

| Crate | Files (today) | Law / chapter |
|---|---|---|
| **voxel_core** | `core_geom` (+ the `CellKey` codec moved from cuboid_mesh), the value half of `voxel.rs` (`Voxel`, `VoxelGrid`, `RecentreVoxels`, constants, `signed_distance*`, `ShapeKind`), `spatial_index`, `units` | foundational vocabulary; no behavior contract |
| **document** | `scene/*`, `sketch`, the producer half of `voxel.rs` (`VoxelProducer` trait, `SdfShape`, `GeometryParams`), `debug_clouds`, `intent`, `command` | 01 — truth; imports no evaluation/display/wgpu |
| **evaluation** | `two_layer_store`, `store`, `chunk_cache`, `chunk_storage`, `disk_chunk_store`, `cuboid`, `incremental_rebuild_plan` (from renderer), measurement queries (`widest_run_in_band`, diameter) | 02 — one evaluator → boundary set |
| **display** | `renderer`, `cuboid_mesh`, `brick_field`, `brick_raymarch`, `texture_atlas`, `block_palette`, `assets/*` (+ `decode_rgba`); `gpu` handed in from the shell. (NOT orchestrator/routing — see seam correction) | 03 — the only crate that links wgpu |
| **interchange** | `vox_export` | 03 (export) — headless sink; never wgpu |
| **work** | `workers/*`, `engagement/orchestrator` (owns the workers), `engagement/routing` | 04 — tempos, generations, staleness |
| **voxel_worker** | `app_core`, `panel`, `settings`, `gpu`, `main` + `shot` bins | shell — composition root |

## Phase 0 — untangle (in the current crate, before any crate is cut)

Four relocations + one split make the graph a clean DAG. Each is a gated commit:

1. **CellKey codec** (`compose_cell_key`/`clean_block_id`/`cell_key_has_overlay`/`MESH_GRID_OVERLAY_BIT`/`mesh_cell_key`) `cuboid_mesh.rs` → `core_geom` as a `CellKey` type, renamed off the mesh. Retires `two_layer_store`/`cuboid`/`brick_*` → `cuboid_mesh`.
2. **`incremental_rebuild_plan`** (+ `IncrementalRebuildPlan`) `renderer.rs` → the residency module (evaluation-to-be). Retires `store → renderer`.
3. **`decode_rgba`/`DecodedRgba`** `workers/scan.rs` → `assets`. Retires `block_palette → workers`.
4. **Split `voxel.rs`** into the value half (→ voxel_core-to-be) and the producer half (`VoxelProducer` + `SdfShape` → document-to-be); relocate the `intent` test that imports `AppCore`.

After phase 0 the modules still live in one crate but respect the layer DAG; the crate cuts become mechanical.

## Phases 1–7 — cut crates bottom-up

Each slice: create the workspace crate, `git mv` its modules, add the `lib.rs` law-statement + chapter citation, add per-crate clippy/test CI gates, run the full gate baseline, push.

1. **voxel_core** — the foundation. (Substrate/camera/raycast already sit below it.)
2. **document** — truth. Law compile-checked: no evaluation/display import.
3. **evaluation** — the evaluator + residency + queries.
4. **display** — the wgpu sinks. **Break the mega-files into folders here**: `cuboid_mesh` (4955) → `mesh/`, `brick_field` (3901) → `brick/`, `renderer` (3461) → `renderer/` (device/pipelines/passes/rasterizer), `two_layer_store`'s display-facing pieces already left. Sub-structure `brick/`, `mesh/`, `atlas/` as module folders — NOT crates (they interoperate through the engagement orchestrator).
5. **interchange** — `vox_export`; the headless-sink law (no wgpu) compile-checked.
6. **work** — the worker pool.
7. **voxel_worker** — thins to `app_core` + `panel` + `settings` + `gpu` + bins.

## Mega-file split targets (as their crate lands)

`cuboid_mesh` 4955, `brick_field` 3901, `renderer` 3461, `app_core` 2400, `panel` 2059,
`sketch` 1783 — each breaks into a folder of cohesive submodules under its crate. Owner rule
(2026-07-14): no mega-files; folders for organization. **Done as their crate landed:**
`two_layer_store` (3366) and `store` (2299) carved into folders when evaluation was cut (Phase 3).

## Deliberately NOT crates (restraint, per ADR 0016 + 0014)

- **display sub-crates** (mesh vs brick) — they interoperate; module folders, not crates.
- **assets** — a display folder until a second (e.g. headless texture-baker) consumer appears.
- **proof / oracles** — parity tests travel `#[cfg(test)]` with their code; `shot` stays a bin.
- **measurement queries as a crate** — fold into evaluation's query surface.
- **`gpu` as its own crate** — device creation is shell-owned (born from the winit surface).

## Documentation standard (definition-of-done, owner 2026-07-14)

Every crate ships a `lib.rs` naming its law and citing its architecture chapter; module docs carry
the rationale/citation voice of substrate/camera/raycast. As each file moves, its docs come up to
that bar — the substrate/camera/raycast "readable spec" vibe, maintained throughout the project.
