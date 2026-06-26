# PROGRESS — VoxelWorker (Rust port)

Autonomous build log. Orchestrator updates this after each milestone. Newest at top.

## Status board

| # | Milestone | Issue | State |
|---|-----------|-------|-------|
| 0 | Repo + scaffolding + dev notes | — | ✅ done |
| 1 | Window + clear + empty egui panel + **headless `shot` binary** | #1 | ✅ done |
| 2 | Voxel core: SDF → instances → flat cubes + orbit cam (5×1×5 cylinder) | #2 | ✅ done |
| 3 | egui params + all shapes + ortho toggle | #3 | ✅ done |
| 4 | Shaders: per-voxel slice, then position-based grid overlay | #4 | ✅ done |
| 5 | View cube + origin gizmo + 2D slice map | #5 | ✅ done |
| 6 | VS folder auto-detect + scan + palette + thumbnails | #6 | ✅ done |
| 7 | Block-JSON per-face textures | #7 | ✅ done |
| 8 | Polish: `.vox` export, config persistence | #8 | ✅ done |
| + | Block lattice + fine floor grid (deferred from M5) | #10 | ✅ done |

## Environment (confirmed this session)

- GPUs: RTX 5070 Laptop, RTX 4090, AMD 890M. DX12 + Vulkan present. Headless render OK.
- Vintage Story **1.22.3 installed** at `%APPDATA%\Vintagestory\assets\survival` — m6/m7 testable.
- `gh` authed as `illeatmyhat` (repo scope). git user "Punleuk Oum".

## Architectural decisions

- **Resolved-grid seam (REPRESENTATION.md, adopted m2-onward).** The renderer, 2D slice, and
  `.vox` export consume a resolved `VoxelGrid`, never `sdf()` directly. The parametric SDF shape
  is the first `VoxelProducer` writing into that grid. v1 has exactly one producer; the seam lets
  future direct-sculpt / override producers plug in without touching anything downstream. Modes
  (bake-then-sculpt / sparse override) are deferred until sculptor users are real.

## Log

- **ADR 0002 — VS performance-techniques section — Part of #22** — appended "Performance techniques
  borrowed from Vintage Story" to `docs/adr/0002-…`: async meshing on a worker thread (off the
  render thread, E2/E3), texture-atlas → one draw per chunk (refines the material design; new open
  question O8: atlas-UV'd cuboid geometry vs per-material binding), dirty-whole-chunk invalidation
  (validates coarse granularity), pooled GPU mesh buffers, optional per-vertex baked AO, and the
  already-present opaque/OIT pass split. Plus a note under O7 that VS itself runs per-frame frustum
  cull + distance LOD, reinforcing the parked-LOD-with-seam call. Design only.

- **ADR 0002 (Proposed) — engine phase: streaming, meshing & coordinates — Part of #22** —
  `docs/adr/0002-engine-streaming-meshing.md`, pointer added to `docs/adr/0001-…` Scale section.
  Sub-ADR decomposing ADR 0001 steps 5–7 (issues #18/#19/#20). **Design only, no code.** Headlines:
  (1) **Cuboid (box) decomposition** as the primary meshing direction over 2D greedy quads —
  domain-matched to chiseled 16³ blocks, VS-proven (`BlockEntityMicroBlock.GenShape`), one material
  per box, and the cuboid packing *doubles as* the step-7 palette/sparse compression (meshing +
  storage collapse to one representation). **Per-block** merge for v1; per-chunk merge parked.
  (2) A **feature-preservation matrix** making every instanced-cube feature (per-face textures,
  per-voxel slice, position-based grid overlay, per-voxel material, layer-range clip, `--debug-faces`,
  onion-fog occupancy) an explicit golden-image contract; highest-risk rows = the **per-voxel
  texture slice** and the **onion-fog 3D occupancy texture**. (3) **Safe order:** land the
  golden-image net (#24) FIRST, then chunk the *existing instanced* renderer (retire the 450k/6M
  caps) BEFORE meshing, then drop the cuboid mesher in **behind a flag, A/B pixel-equivalence**
  against the instanced path, flip the default only when goldens match. Status **Proposed** — open
  forks (cuboid-vs-greedy, per-block-vs-per-chunk granularity, chunk size, order, fog fidelity at
  scale) await sign-off before implementation. Build sanity: `cargo build --bins` green.

- **CI + line-ending hygiene — Closes #23** — `.github/workflows/ci.yml`, `.gitattributes`,
  `Cargo.toml`, `tests/palette_click.rs`, `PROGRESS.md`. GitHub Actions workflow runs on push/PR to
  `main`: a `ubuntu-latest` job installs stable Rust + clippy, caches cargo (`Swatinem/rust-cache`),
  installs Linux GUI build deps, then runs `cargo build --bins`, `cargo clippy --all-targets -- -D
  warnings`, and `cargo test`; plus an optional `windows-latest` build-only job. **No-GPU
  constraint:** GitHub runners have no GPU. Of the 54 tests, exactly one needs a wgpu device —
  `windowed_palette_tile_click_reaches_apply_path` in `tests/palette_click.rs` (calls
  `GpuContext::new`). It is now gated behind a new off-by-default `gpu` cargo feature (file-level
  `#![cfg(feature = "gpu")]`), so default `cargo test` runs the 53 CPU tests green on a GPU-less
  runner while the GPU test stays runnable locally via `cargo test --features gpu`. The `shot`
  binary is built but never executed in CI. **Line endings:** added `.gitattributes` (`* text=auto`
  + explicit text/binary types) to stop the "LF will be replaced by CRLF" warnings; ran `git add
  --renormalize .` (index was already LF-normalized, so no existing files changed).

- **ADR 0001 step 4 (UI half): author groups, definitions & instances — COMPLETES step 4 / Closes
  #17** — `src/scene.rs`, `src/panel.rs`, `src/lib.rs`, `src/bin/shot.rs`, `PROGRESS.md`. Adds the
  UI to AUTHOR the recursion the resolve already supported (4a). **Tree node list:** the Scene
  section now renders the assembly as an INDENTED TREE (`Scene::tree_rows` → depth-first `(NodePath,
  depth)` rows; a Group's children nest one indent level under it), so Group children are visible +
  selectable at ANY depth, not just top-level nodes. Each row keeps its visibility checkbox + delete
  ✕; selecting a node (any depth) sets it active for the inspector. **Selection model:** `Scene.active`
  changed from `Option<usize>` to `Option<NodePath>` (a `Vec<usize>` of child indices through
  `nodes`/Group children); `active_node[_mut]`, `add_node`, `remove_node` are now path-based (remove
  falls back to parent/sibling/None), plus `node_at_path[_mut]`. **Group:** a **Group** button wraps
  the active node in a new `Group` (`group_active`); when a Group is active, **+ Add child** appends a
  Tool/Part into it (`add_child_to_group`). **Definitions + Instances (village workflow):** **Make
  definition** turns the active Group/node into an `AssemblyDef` in `scene.definitions` and replaces
  it with an `Instance` of it (`make_definition_from_active`); a new **Definitions** list shows each
  def with an **Add instance** button (`add_instance`, nudged +X so placements don't overlap) — one
  stored body placed by N instances. **Inspector** extended: Group/Instance active nodes show name (+
  the referenced def for an Instance) and the shared Offset editor; Tool/Part unchanged. **Scope:**
  in-memory only (tree persistence is step 8 — `// step 8` notes); no rotation/scale; resolve/model
  semantics from 4a untouched (only added mutation helpers). 5 new unit tests: `group_active` nests
  the active node under a new Group (active → the wrapped child `[0,0]`); `make_definition` puts a def
  in `scene.definitions` + replaces the node with an Instance (occupancy preserved); `add_instance`
  appends an Instance of that def and the scene resolves to 2× the def's occupancy from ONE stored
  body; `tree_rows` lists Group children indented depth-first; `node_at_path` reaches a Group child.
  Green: `cargo build --bins` clean, `cargo clippy --all-targets` clean, `cargo test` 53 lib + 1
  integration pass. New `shot --demo-groups` (top-level Group of 2 children + a sibling Tool + an
  Instance + a Definition): `shots/groups_tree.png` shows the INDENTED tree (Cluster·Group(2) with
  Core/Shell nested under it, Lone·Box, Widget instance) + the Definitions list + Group inspector;
  `shots/village_tree.png` (`--demo-village`) shows 4 Instance rows + `House (2 node)` def with Add
  instance. Honest note: interactive clicks (group/make-def/add-instance buttons) can't be exercised
  headlessly — they're covered by the helper unit tests; the shots prove the tree + Definitions list
  RENDER.

- **ADR 0001 step 4 (model + resolve half): recursion + instancing — Part of #17** —
  `src/scene.rs`, `src/bin/shot.rs`, `PROGRESS.md`. Makes `Group`/`Instance`/`AssemblyDef` WORK in
  `Scene::resolve_region` (they were typed no-ops). Added `Scene.definitions: Vec<AssemblyDef>` +
  `def_by_id(DefId)` lookup. Resolution + extent now both flow through one recursive tree-walk
  (`for_each_leaf` → `walk_nodes`): it descends `Group(children)` and `Instance(DefId)`, composing
  **world translation DOWN** the tree (`world = parent_offset + node.offset_blocks`, translation
  only — rotation/scale stay later), and visits every visible **leaf** (Tool/Part) with its
  accumulated world offset. An `Instance` resolves the referenced definition's children under the
  instance's transform, so ONE stored definition placed by N instances stamps at N locations (the
  village-of-reused-houses case; definitions stored once). **Cycle guard:** `walk_nodes` carries a
  `def_path` stack of the definition ids currently expanding; an `Instance` whose id is already on
  the path is skipped (logged) instead of recursing — a self-instancing def resolves finitely, never
  overflows. A dangling `Instance(id)` (no matching def) resolves to nothing. `full_extent_blocks`
  recurses too (gathers all leaf world-AABBs, so nested + instanced positions widen the composite).
  Existing flat-scene behaviour is bit-for-bit identical (the leaf walk over a flat list with zero
  parent offset reproduces the old per-node loop; same recentre). **No new UI** (the node list still
  shows only top-level nodes — that's the step-4b follow-up). 4 new unit tests: nested-Group
  transform composition (leaf at +B inside Group at +A lands at world A+B×density, matching a flat
  node at A+B); Instance-of-1-node-def at T == that node placed directly at T; 2-instance village ==
  2× the def's voxel count at two disjoint clusters; self-referential def resolves without overflow,
  contributing its leaves once. `shot` gains `--demo-village`: one small "house" `AssemblyDef` (a
  2³ stone Box body + a 1×2×1 wood Cylinder chimney composed as a Group, so the chimney offset is
  relative to the house) placed by FOUR `Instance` nodes in a row. Green: `cargo build --bins` clean,
  `cargo clippy --all-targets` clean, `cargo test` 48 pass; `shots/village.png` shows all four
  houses (stone body + wood chimney) at four separated locations from the single stored definition;
  `shots/sphere_check.png` confirms a normal `--shape sphere` shot is unchanged. NOTE (renderer cap,
  not a model issue): the renderer draws only the first `MAX_DRAWN_INSTANCES` (450k) voxels, so the
  demo house body is deliberately 2³ (4 houses ≈ 158k voxels) to keep all four under the draw cap;
  the model resolves all instances regardless (proven by the village unit test's 4× count).

- **ADR 0001 step 3 (per-voxel material half): COMPLETES step 3 / closes #16** —
  `src/panel.rs`, `src/scene.rs`, `src/renderer.rs`, `src/shaders/voxel.wgsl`, `src/main.rs`,
  `src/bin/shot.rs`. Distinct nodes now render in distinct materials, driven by `Voxel.material_id`.
  **Id mapping:** `MaterialChoice::material_id()` (Stone=0, Wood=1, Plain=2) + `from_material_id`;
  `scene.rs::material_id_for` now returns the Tool's real id (was always `Some(0)`), so each Tool
  stamps its one material; a Part (clouds) still emits its own (id 0 today). **Renderer:** added a
  `material_id: u32` field to `VoxelInstance` (new instance vertex attribute @location 5, `Uint32`)
  carried from the grid's `u16`. **Shader (bounded approach — NOT a per-material texture array):** a
  small uniform `array<vec4,3>` of per-material base colours, indexed by the per-instance
  `material_id`, MODULATES the lit/textured colour. The base colours are each material's average
  colour RELATIVE to the bound texture's average (computed in `update_uniforms`), so the bound
  material's own slot is ~neutral (its texture shows unchanged — single-material models look
  identical to before) and the others recolour the one shared texture toward their tint. **Scope
  kept:** modulation is gated OFF for `--debug-faces` (face-orientation colours intact) and for a
  loaded VS block (stays a single GLOBAL material, per the ADR) — `update_uniforms` now takes the
  bound `MaterialSource` and sets `material_modulation_enabled` accordingly. The per-face slice +
  grid overlay are unchanged. **Tests:** `wood_tool_stamps_wood_material_id` (every voxel a Wood
  Tool emits carries the Wood id) and `two_material_scene_has_both_material_ids` (a Stone+Wood scene
  composites both ids). **Headless visual:** `shot --demo-scene` (sphere=Stone grey, box=Wood brown,
  torus=Plain tan) shows DISTINCT per-node colours; `--material wood` still renders wood-tinted;
  `--debug-faces` still shows the R/G/B face palette. `cargo build --bins` clean, `cargo clippy
  --all-targets` clean, `cargo test` 44 + 1 pass.

- **ADR 0001 step 3 (translation half): per-node placement** (part of #16; per-voxel material is the
  remaining #16 sub-step) — `src/scene.rs`, `src/panel.rs`, `src/bin/shot.rs`. Nodes can now be
  PLACED in space, so two nodes occupy disjoint voxel regions instead of overlapping at the origin.
  **Inspector:** a new **Offset (blocks)** control (X/Y/Z integer `DragValue`s, may be negative)
  edits the active node's `transform.offset_blocks` for both Tools and Parts; editing it sets
  `scene_changed` so the caller re-resolves AND re-frames the composite. Offsets are in-memory only
  (persistence is step 8 — a `// step 8` note marks it). **Resolution:** `resolve_region` translates
  each node's producer voxels by `offset_blocks × voxels_per_block`, then subtracts a composite
  recentre (`((min+max)/2)×density` over all placed-node block AABBs) so the whole composite stays
  centred on the origin (what the renderer + camera auto-frame assume). A single zero-offset node
  recentres on itself → zero shift → bit-for-bit identical to step 2 (the identity guarantee holds).
  **Extent + framing:** `full_extent_blocks` now returns the per-axis size of the union AABB of every
  placed node (`max−min` of `offset ± half-size`), growing to encompass offset nodes; the camera
  auto-frames `grid.dimensions` (the whole composite) and the voxel cap is checked against the
  composited region. Rendering stays single-material (per-voxel material untouched — next #16 step).
  **Verification:** `cargo build --bins`, `cargo clippy --all-targets`, `cargo test` all clean (42
  lib tests + 1 integration test pass). Three new scene unit tests: (a) a node at `offset=[N,0,0]`
  shifts its voxels by exactly `N×density` in X vs offset 0; (b) two disjoint-offset nodes give
  `occupied_count == sum` of each alone (disjoint union); (c) `full_extent_blocks` grows from 2→6
  blocks in X when a node is offset. New TEMPORARY `shot --demo-scene` flag (kept + documented in
  `--help`): a hardcoded 3-node placed scene (sphere @origin + box +8 X + torus +6 Z) renders the
  solids clearly SEPARATED in space; a normal `--shape sphere` shot is unchanged (53776 voxels, one
  centred disc). **ADR deviation:** the demo's third node is a torus Tool, not the example's clouds
  Part — `DebugClouds` has no bounded size (it fills its whole region) so as fog it would occlude the
  separation; Part placement is covered by the unit tests + the inspector instead.

- **ADR 0001 step 2: flat node-list UI + add/select/delete/visibility; the node is the panel's
  source of truth** (closes #15) — `src/scene.rs`, `src/panel.rs`, `src/settings.rs`, `src/main.rs`,
  `src/bin/shot.rs`. `Scene` gained an `active: Option<usize>` selection plus `add_node` /
  `remove_node` (selection-preserving) helpers; `full_extent_blocks` now takes the per-axis MAX over
  all leaf nodes so several origin-centred nodes composite into one region (union). The panel holds
  the `Scene`: a new **Scene** node-list section lists each node as a selectable row (name +
  visibility checkbox + per-row ✕ delete) with an **+ Add** menu (a Tool of any shape, or a Clouds
  Part). The inspector **switches on the active node** — a Tool shows the existing
  Shape/Size/Density/Material controls (edits mirror back onto the active node); a Clouds Part shows
  its name + seed instead. `GeometryParams::debug_clouds` and the old "Clouds" chip are **removed** —
  the Part node content replaces the boolean. Resolution routes through `resolve_scene(&Scene, density)`
  in `main`/`shot`; the voxel cap is now evaluated against the composited region. Persistence
  (`AppConfig`) drops the `debug_clouds` field (old configs still load — serde ignores the unknown
  field — and migrate to a one-Tool-node scene); multi-node scene persistence is deferred to step 8
  (marked `// step 8`). New tests: a 2-node sphere+box `resolve_region` equals the set-union and is
  strictly larger than either alone; an old-config-with-`debug_clouds` migration test; updated
  round-trip. Verified: `cargo build --bins` / `clippy --all-targets` clean (no warnings), `cargo test`
  40 pass; `shots/step2-panel.png` shows the node list + Tool inspector over a rendered sphere, and
  `shots/step2-clouds.png` shows the Clouds node with the Part (name+seed) inspector. NOTE: interactive
  add/delete/select clicks can't be exercised headlessly — covered by the model unit tests + the two
  visual confirmations; transforms stay zero so multiple nodes overlap at origin (expected for step 2).

- **ADR 0001 step 1: scene model + region-addressable compositing (no UI)** — new `src/scene.rs`.
  Introduces the assembly model (`Scene`/`Node`/`NodeContent { Tool, Part, Group, Instance }`,
  `Part::DebugClouds`, `AssemblyDef`, `DefId`, `CombineOp::Union`, `NodeTransform`) and routes ALL
  voxel resolution through `Scene::resolve_region(region_blocks, voxels_per_block, lod)`. The producer
  trait is unchanged (producers still emit centred-at-origin content); the Scene now owns compositing
  — a union tree-walk that resolves each visible leaf and **stamps** it into the output grid. Only the
  two leaves that exist today (`Tool` = SDF + `MaterialChoice`; `Part::DebugClouds`) are resolved;
  `Group`/`Instance` are typed but no-ops (`// step 4`). `lod` is carried from day one (always 0) as
  the cheap forward-seam ADR 0001 requires. `resolve_active_producer` (main.rs) + the shot resolve now
  build a one-node scene; used in the constructor, `rebuild_geometry`, `export_vox`, and shot. **No
  user-visible change** — all 5 SDF shapes + Clouds render exactly as before. The `debug_clouds`
  boolean is intentionally KEPT as the selector (its deletion is step 2). Guarantee proven by a unit
  test: a one-node Tool scene's `resolve_region` yields the same dimensions + occupied count as
  `SdfShape::resolve` (and a Part scene vs `DebugCloudField::resolve`). Green: build/clippy clean,
  40 tests pass, sphere + clouds shots unchanged.
- **ADR 0001 accepted: scene graph (parts vs tools), streaming, scale** — `docs/adr/0001-scene-graph-
  parts-and-tools.md`. Replaces the `debug_clouds: bool` shortcut with a proper **assembly graph**:
  a recursive Scene of nodes, each a producer — **Tool** (parametric SDF, single material) or **Part**
  (static voxel body, multi-material, e.g. the cloud field / a saved chiseled block). Reuse by
  reference (definition + instances → a village of identical houses is cheap). Distinct from the
  per-Tool SDF *construction* tree (booleans/lathe) in REPRESENTATION.md — the scene sits above it.
  Decisions: union-only (CombineOp growth path), per-voxel materials, affine-target transform
  (translation first), density = app setting (16). **Scale committed:** explicit working canvas
  (~1024³ blocks, designed far beyond) → no monolithic grid; region-addressable `resolve_region(aabb,
  lod)`, chunked + spatial-indexed + lazily resolved, 64-bit addressing, origin-rebased rendering,
  greedy meshing, GPU instancing, out-of-core store, palette/sparse compression. **LOD parked** (may
  never build) but seam preserved: `lod` in the resolve signature + `(coord,lod)` cache key, renderer
  consumes opaque per-chunk items. 8-step build sequence; step 1 = model + `resolve_region` through a
  one-node scene, delete the boolean, no UI, behaves identically. Implementation pending step-1 go.
- **debug clouds in the interactive app (panel "Clouds" chip)** — Wired `DebugCloudField` into the app
  as a 6th shape option. `GeometryParams` gains `debug_clouds: bool`; the Shape section adds a "Clouds"
  chip (after a separator) that's mutually exclusive with the 5 SDF chips (an SDF chip highlights only
  when `!debug_clouds`). New `resolve_active_producer(grid, shape, geometry)` helper branches the
  producer; used in the constructor, `rebuild_geometry`, and `.vox export` so all three honour the
  toggle (grid dims still come from `shape.grid_dimensions()`, so the cap check / renderers / fog are
  unchanged). Persisted in `AppConfig` (serde `#[serde(default)]`, round-trip test updated). NOTE: the
  size sliders cap at 16 blocks, so for a large cloud volume in-app use density (e.g. 16 blocks @ 8 =
  128³); bigger sizes are reachable via `shot --shape debug-clouds`. Build (both bins) + clippy + 35
  tests green.
- **debug cloud field producer (`src/debug_clouds.rs`)** — A second `VoxelProducer` (besides
  `SdfShape`) that fills the grid with several visually distinct, billowy cloud blobs in a mostly-empty
  volume — richer test content than the 5 SDFs (many disjoint objects + space), exercising the renderer
  and onion fog. Recipe = the standard cloud one: each cloud is a soft RADIAL FALLOFF (bounded, separate
  puff) whose surface is displaced by FRACTAL PERLIN NOISE (fBm = summed octaves of gradient noise);
  `max` across clouds keeps them separate. 8 puffs on the octant centres, seed-jittered position/radius
  so none read alike. Self-contained improved-Perlin + fBm + small LCG (no new dependency); fully
  deterministic from `seed`. Recommended fBm over Worley/cellular (fBm = soft/fluffy; Worley =
  lumpier/cauliflower) but the code makes either easy. Exposed via `shot --shape debug-clouds`
  (grid dims still = size×density; honours the 6M cap). Verified at 128³ and 180³
  (`shots/debug-clouds*.png`): 8 fluffy puffs, clear gaps. +2 unit tests (non-empty/not-too-dense,
  determinism). Lib + shot clippy clean; 34 lib tests. NOTE: app exe was locked (app running) so only
  the `shot` bin + lib were rebuilt this turn; rebuild `voxel_worker` when free.
- **fog: AABB-clipped sampling + depth occlusion (Minecraft-cloud model)** — Two corrections after
  visual review. (1) **Sampling bug:** the march spread its fixed 96 steps over the FULL near→far ray
  (hundreds of units), so from a top/bottom view — where the onion band is thin in the view direction —
  almost no samples landed in the grid and the fog vanished. Fixed by clipping the ray to the grid's
  world-space AABB (slab test) and spending every step INSIDE the box. A torus now shows the expected
  cloud **donut from top/bottom**. (2) **Occlusion:** the earlier "x-ray, ignore depth" choice (option B
  from the handoff) was wrong — real voxel clouds (e.g. Minecraft's translucent cloud prisms) are
  depth-tested and occluded by solid geometry. Re-added the MSAA scene-depth binding and clamp the march
  at the nearest opaque surface, so the **displayed slice occludes the onion layers behind it**; only
  the near-side/beside layers show. Bonus: this also removes the view-angle "stripe" artifact (a
  single-layer sphere now reads as a coherent haze, not bright bands), because the bright far-side path
  is now occluded. NOTE: this reverses the handoff's recorded "option B = x-ray" fog decision, per the
  user's direct visual feedback. Tree green: build (both bins, no warnings) + clippy + 33 tests.
- **fog rearchitected: voxel-density cloud (replaces GPU SDF re-derivation)** — The fog raymarch now
  samples the **resolved VoxelGrid as a 3D R8 occupancy texture** (trilinear-filtered → smooth cloud
  density), instead of re-deriving the parametric SDF on the GPU. Why: the SDF approach only worked for
  the 5 built-in shapes, duplicated `src/voxel.rs`'s SDF library in WGSL (a sync hazard), and quietly
  bypassed the resolved-grid seam (REPRESENTATION.md — every other consumer reads the grid, never the
  SDF). The cloud approach works for ANY voxel set (future sculpt / override producers) and deletes the
  whole `scene_sdf` port. `OnionFogRenderer::upload_grid` densifies the sparse occupied list into a
  `D3` `R8Unorm` texture on each geometry rebuild (≤6 MB at the 6M voxel cap; grids past the device's
  3D-texture limit disable the fog instead of failing); world→grid coords are `world/semi_axes*0.5+0.5`
  (voxel centres land on texel centres). The shader rejects samples outside the grid box so clamp-to-edge
  can't smear border voxels along the ray. Option B carries over: x-ray full-ray march, edge inset via
  `smoothstep(0.35,0.85,density)`, strength 0.10. Depth is no longer sampled (removed the depth
  `TEXTURE_BINDING` + the fog's depth param). `shape_kind`/`wall_voxels` dropped from the fog uniform +
  params. **Validated** on a flat torus ring (`shots/onion-vox-ring-34.png`): the fog follows the ring
  and breaks across the central hole — proving it reads true occupancy, not a filled SDF disc. Tree
  green: build (both bins, no warnings) + clippy + 33 tests.
- **fog polish → option B (x-ray onion), implemented** — Applied the user-confirmed B decision to the
  volumetric fog (`onion_fog.wgsl` + `OnionFogRenderer`): (1) **x-ray** — the raymarch now ignores the
  opaque slab's depth and marches the FULL ray, so the neighbour onion bands show through the displayed
  slice on BOTH sides (`scene_depth` stays bound but unused, reserved for a future occluded mode);
  (2) **inset** — `FOG_EDGE_INSET = 0.75` voxels pushes the smooth SDF density edge inward so the haze
  stays inside the voxel slab's stair-stepped silhouette instead of haloing past it; (3) **lower
  density** — `ONION_FOG_STRENGTH 0.18 → 0.10` so it reads wispy/aerogel, not a frosted puck. Verified
  headless on a 6³ sphere single-equator slice + onion 3/5 (`shots/onion-B*.png`): bands show above AND
  below the slab, stop within the disc rim (no undercut), and the brown texture reads through. The
  horizontal-stripe look of a thin onion band is inherent to the already-chosen volumetric SDF method,
  not a regression. Tree green: build (both bins) + clippy + 33 tests.

- **volumetric onion fog + camera core #13a (green checkpoint)** — Onion skin evolved into a true
  **fullscreen SDF raymarch fog pass** (`onion_fog.wgsl`): ports the 5 shape SDFs, marches the view
  ray bounded by the 3D MSAA depth, Beer–Lambert haze in the onion Y-range; `OnionFogRenderer` in
  renderer.rs, wired via `FrameOverlays.onion_fog` in `render_frame`; per-voxel ghost machinery removed;
  3D depth store Discard→Store + TEXTURE_BINDING so the fog ray can read scene depth. Onion visual
  polish (occlusion vs x-ray / slab-edge inset / density) is an OPEN A/B/C decision for the user.
  **FOG VISUAL DECISION (user, confirmed):** go with **B — x-ray onion**: the fog pass must IGNORE
  the opaque slab's depth occlusion so neighbor fog shows on BOTH sides through the slice (the
  conventional cel-animation "onion skin" = see neighbors through the current layer), PLUS a slight
  inset so it doesn't undercut the voxel slab's stair-stepped edges, PLUS lower density. This is the
  next fog task (currently the fog is realistic/occluded = option A).
  Also folded in **#13a camera core**: Fusion pole fix (drag reaches the pole), drag-the-cube-to-orbit,
  and edge/corner snap views (all 26 orientations via the hot-zone model + unified snap-direction).
  Tree green: build + clippy + 32 tests. **Coordination lesson:** the user drives subagents directly
  (messages I don't see), and I ran #13a concurrently with the user's fog agent on shared files →
  collision. Going forward: serialize anything touching shared files; don't assume "fabrication" when
  a subagent cites direction I didn't relay. See [[multi-agent-coordination-lessons]].
- **layer scrubber (#12)** — Replaced the static 2D mid-Y slice map with a Y-layer range scrubber:
  two trim handles, block-boundary ticks, block snapping (toggle), band readout + measured-diameter
  stat. Bounds clip the 3D render to the slab (inclusive `[lower,upper]`; layer index recovered in
  shader from instance center). Single layer + TOP snap = the chisel stencil. Onion skin: ghost
  neighbor layers — initially an alpha-blended translucent fog (deviated from the spec'd screen-door
  dither). CORRECTION: an earlier note here wrongly said the subagent "fabricated user feedback" for
  that change — it did not. The user was steering that subagent **directly** (messages the orchestrator
  doesn't see), so the deviation was real user direction, not a hallucination. The onion later evolved
  into the volumetric SDF fog below. `VoxelUniforms` `shot` gains `--layer-lower/--layer-upper/--onion`.
  Persists snap/onion prefs. Clippy clean; --debug-faces OK.
- **fixes (post-v1, from first live run, #11)** — (1) Backface culling: `unit_cube_geometry` had
  mixed winding (+X/−X/+Y/−Y CW-from-outside) → standard Ccw/Back culled the visible faces; fixed to
  CCW-outward + winding tests. Invisible in static screenshots. (2) Removed the 90-block cap +
  label-dedup → **434 groups**; thumbnails built ≤8/frame to avoid startup hitch. (3) Palette click:
  full path verified correct + regression test; was likely masked by the backface bug. Added a
  **face-orientation debug mode** (`shot --debug-faces` / Display toggle) — colors faces by outward
  normal, stripe-marks back-faces (cull off); used it to CONFIRM the cull fix (default octant =
  red/green/blue, no marker). Window opens maximized. 27+1 tests pass.
- **m8** — Polish done & verified. (1) `.vox` export: hand-written chunked binary (VOX 150,
  MAIN/SIZE/XYZI/RGBA), Y-up→Z-up axis map, splits into ≤256 tiled models (no truncation), palette
  index 1 = active material avg color; "Export .vox" button (rfd) + `shot --export-vox`; round-trip
  validated with dot_vox (80,384 voxels, 322KB). (2) Config persistence: `%APPDATA%\VoxelWorker\
  config.json` (geometry/projection/material/toggles/applied-block/camera/window); load on start,
  save on close/exit; bad config → defaults, never panics; round-trip tested. (3) Block lattice +
  fine floor grid (closes #10) via M5 line pipeline (now RGBA/alpha); lattice default ON, floor OFF.
  (4) rayon parallel sampling: sphere 12³@16 **45.8ms → 19.8ms (2.3×)**, voxel set identical.
  24 tests pass; clippy clean. Future work: 24→8 instance packing, multi-material .vox palette.
- **m7** — Per-face block-JSON textures done & verified. `BlockSource::resolve_faces` + VS impl:
  cached `blocktypes/**.json` index (VS lenient-JSON normalized → serde_json), directory-keyed +
  scored matching, handles `all`/explicit faces/`sides`/`horizontals`/`verticals` + `texturesByType`
  + `{rock}`/`{wood}` placeholders + `domain:path` resolution; graceful uniform fallback. Renderer
  now binds a 6-layer `D2Array`; shader picks the layer from face normal (one pipeline serves uniform
  + per-face). Per-voxel slice + grid overlay preserved per face. Finding: **0/90 chiselable blocks
  have distinct faces** (vanilla rock all uniform) — mechanism proven on a log (end-grain top vs bark
  sides, m7-perface). `shot` gains `--apply-block/--list-perface/--force-demo-stem`. Deps serde +
  serde_json. Clippy clean; 19 tests pass.
- **m6** — VS auto-detect + scan + palette done & verified against the real install. Pluggable
  `BlockSource`/`SourceDetector` traits; `VintageStoryDetector` + `VintageStorySource` +
  `CustomFolderSource` + registry. Background thread (mpsc) does detect+walkdir+PNG-decode; main
  thread does GPU work (thumbnail render → `register_native_texture`). **Real scan: 90 groups**
  (Granite/Basalt/Sandstone/Slate/planks/marbles…). ALLOW/EXCLUDE tuned: added `metal/` + `painting/`
  excludes (the `chalk` substring was matching molybdochalkos + caveart). Dedup-by-label at the 90 cap
  → distinct materials. Palette dock with 45° cube thumbnails; click applies a variant as active
  material (`MaterialSource::Loaded`, per-voxel sliced — verified on m6-applied). "Connect folder…"
  rfd fallback. `shot` gains `--scan-vs`/`--apply-first-block`. Clippy clean; 13 tests pass.
- **m5** — View cube + gizmo + 2D slice done & verified. View cube: wgpu corner viewport (scissor),
  6 CPU bitmap-font face labels, mirrors main camera; click→ray-pick face→eased snap tween (8 unit
  tests for snap table / nearest-theta / easing). Gizmo: X/Y/Z lines + perpendicular squares,
  depth_compare=Always so it shows through the model; toggle off by default. 2D slice: mid-Y layer
  read from `VoxelGrid.occupied` → egui nearest image with teal block lines (circle/ring/square per
  shape). `shot` gains `--gizmo/--no-viewcube/--snap <face>`. Lattice/floor deferred → issue #10.
  Cosmetic: TOP label 180°-rotated head-on (picking correct). Clippy clean; tests pass.
- **m4** — Shaders done & verified; BOTH regression-guarded bugs confirmed fixed. Procedural
  Stone/Wood/Plain textures (CPU-gen, nearest/clamp). Bug 1: per-voxel `1/density` texture slice
  (one texture per block — wood top-down restarts grain per block cell). Bug 2: grid overlay from
  world position `vox_abs` (block lines align on vertical faces, no off-by-one). `VoxelUniforms`
  (view_proj + grid_half_extent + density + line colors/widths/alphas). 4× MSAA (resolve into
  surface/capture; egui at 1 sample on resolved view). Material selector now functional; "Voxel
  grid overlay" toggle live. `shot` gains `--material` + `--grid`. sRGB-correct (textures sRGB,
  lighting+lines in linear). Clippy clean. Goldens: m4-slice-wood-grid-topdown, m4-grid-box-zoom.
- **m3** — Params/shapes/ortho done & verified. Functional panel: shape chips (all 5), X/Y/Z
  block sliders, density, conditional Tube wall, projection toggle, inert material selector,
  Display placeholder. Params split into `GeometryParams` (drives dirty rebuild) vs display/camera
  (no rebuild). Auto-frame gated on size/density change only — **shape-switch keeps size & camera**;
  **density is fineness-only** (verified d8 vs d32 = same physical disc). Ortho branch in
  `OrbitCamera` (vh=dist*0.42). Voxel cap (6M grid / 450k instances) prevents freezes. `shot` gains
  `--shape/--size-*/--density/--wall/--proj`. Screenshots m3-{cylinder,sphere,sphere4,box,torus,
  tube,ortho,d8,d32}.png all correct. Clippy clean.
- **m2** — Voxel core done & verified. `VoxelGrid`/`VoxelProducer` seam in place (`SdfShape` is
  the sole producer; renderer builds instances from the grid, never calls the SDF). Full SDF set
  + dispatcher (descriptive names). Instanced unit cubes (per-face normals), flat directional+
  ambient shading, depth buffer, perspective orbit camera (drag-orbit + wheel-zoom windowed;
  `--theta/--phi/--dist` in `shot`). 5×1×5@16 cylinder = **80,384 voxels**; `shots/m2-persp.png`
  and `m2-top.png` show correct round disc + stair-stepped rim. Clippy clean. Sampling loop is
  order-independent (rayon-ready). egui pipeline given `Depth32Float` to share the depth pass.
- **m1** — Foundation done & verified. Crate `voxel_worker`: lib (render-target-agnostic
  `render_frame(&TextureView,...)`) + windowed bin (winit 0.30 / wgpu 29 / egui 0.34) + headless
  `shot` bin (offscreen → PNG, 256-byte row padding handled). `shots/m1.png` shows the panel as
  expected. Clippy clean. Minor forced API deviations logged in issue #1 (egui Panel API, wgpu 29
  surface/poll/instance changes) — all matched against registry source.
- **m0** — Scaffolding: `.gitignore`, `docs/DEV_NOTES.md` (verified API sigs), this file. Repo
  created and pushed. Issues #1–#8 + tracking issue opened.
