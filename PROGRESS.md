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

- **Chunk cache + lazy per-chunk resolve + per-chunk cap (S2) — Part of #27 (ADR 0002 streaming,
  Decision 3)** — turns S0's chunk-addressable resolve into the resolve MECHANISM, retiring the 6M
  whole-scene total cap. Render output stays BYTE-IDENTICAL (goldens green, not rebaselined). The
  recentre removal / camera-relative rebasing / renderer consuming per-chunk meshes are still S4.
  - **New `src/chunk_cache.rs` — `ChunkResolveCache`.** Key `ChunkCacheKey { chunk_coord: [i32;3],
    lod: u32 }` → resolved per-chunk `VoxelGrid` (absolute coords, from `Scene::resolve_chunk`). API:
    `chunk(coord, scene, density, lod)` returns the cached grid, resolving + storing on a miss and
    returning the cached one on a hit; `resolve_region(scene, density, lod)` reassembles the whole
    recentred monolithic grid from cached chunks; `clear()` / `invalidate_chunk(coord)` are the
    invalidation seam. Plain in-memory `HashMap`; a density change clears + re-binds (a chunk's voxel
    extent depends on density). Exported from `lib.rs`.
  - **S3 invalidation seam (TODO left, NOT implemented).** `clear()` and `invalidate_chunk()` carry
    explicit `TODO(#27 S3)` notes: S3 adds edit-world-AABB → dirty-whole-chunk invalidation on top of
    this seam. Until then, every scene edit must `clear()` wholesale (which `main.rs::rebuild_geometry`
    does each rebuild).
  - **Bit-identical assembly (how).** `resolve_region` is the render truth: it sizes the output to the
    composite extent and RECENTRES by subtracting `recentre_voxels` from every voxel. The cache pulls
    each covering chunk (absolute coords) and subtracts the SAME `recentre_voxels`
    (`Scene::recentre_voxels_for_resolve`, the exact value `resolve_region` inlines). For all
    near-origin scenes (every golden + every parity scene) the positions are exactly representable in
    f32, so the subtraction is exact → output is byte-for-byte identical. Parity tests key on raw
    `f32::to_bits()` (not rounded voxel indices), so they assert true byte-identity, not just the same
    voxel set.
  - **Latent S0 bug found + fixed (flat/odd shapes).** S0's `covering_chunk_range` derived the chunk
    range from the BLOCK-AABB (`placed_extent_blocks`, `floor(size/2)` per block), but producers emit
    voxels CENTRED on the origin — for an odd/flat axis (e.g. the default 5×1×5 cylinder, Y=1 block)
    the single layer straddles chunks Y=−1 and Y=0, and the block-AABB range covered only Y=0,
    **dropping half the voxels**. The S0 [5,5,5] tests never exercised a flat axis so it slipped
    through. Fixed by computing the covering range from a new `placed_extent_voxels` (the producer-true
    voxel frame, `grid/2 = size·d/2` half-extent), so `resolve_region_via_chunks` and the cache now
    cover every chunk a voxel can land in. The RECENTRE still uses the block frame (to match
    `resolve_region` exactly). New test `cache_region_matches_monolithic_for_flat_and_odd_shapes` pins
    it; the debug-clouds golden (which had silently resolved 0 voxels via the broken chunk path during
    development) is back to 147588.
  - **Per-chunk cap (`src/voxel.rs`).** Added `MAX_CHUNK_VOXELS = 6_000_000` (per-chunk bound) +
    `chunk_extent_exceeds_bound(density)` (true when one chunk's voxel CAPACITY,
    `(CHUNK_BLOCKS·density)³`, exceeds the bound). `MAX_GRID_VOXELS` is retained but documented as no
    longer a whole-scene total cap (still used by `SdfShape::exceeds_voxel_cap` for the single-shape
    `.vox`-export guard). **Call sites** (`main.rs::rebuild_geometry`, `shot.rs`): the old
    `region_voxel_count > MAX_GRID_VOXELS` rejection is gone — they now reject ONLY a pathological
    density via `chunk_extent_exceeds_bound`, with updated user-facing "one chunk is N.NM voxels …"
    messages. **Net effect:** a scene whose TOTAL exceeds 6M but whose chunks are small now resolves
    (proven by `scene_exceeding_old_total_cap_resolves_under_per_chunk_bound`: 64 boxes spread over an
    8.2M-voxel composite AABB, each chunk tiny). At density 16 one chunk is 64³ = 262k voxels (far
    under the bound); only a degenerate density (≥~63) trips the per-chunk guard.
  - **Routing.** `main.rs::rebuild_geometry` resolves via a persistent `ChunkResolveCache` field
    (cleared each rebuild pending S3); `shot.rs` resolves placed/single-shape scenes through a cache,
    and keeps the explicit-region monolithic path for a Part-only (`--shape debug-clouds`) scene (it
    has no composite AABB to chunk — `Scene::has_chunkable_extent` picks the path).
  - **Gate:** `cargo build --bins` ✅, `cargo clippy --all-targets` ✅ (no new warnings),
    `cargo test` ✅ (112 lib tests: 102 + 10 new), `cargo test --features gpu --test golden` ✅ GREEN
    (NOT rebaselined). Headless `shot --demo-scene` renders correctly (read the PNG: sphere + sphere +
    box, clean geometry, full panel).

- **Far-offset demo scene + CPU placement test + precision baseline (S1) — Part of #18 (ADR 0002
  streaming)** — creates the TEST CONDITION that makes 64-bit coords + origin-rebasing provable in
  S4. Mostly observational/test-only; **no render math, no recentre, no `offset_blocks` type change
  touched** (those are S4). Goldens stay green.
  - **New `shot` flags** (`src/bin/shot.rs`): `--demo-far-offset` builds a small 4³ stone box at
    `offset_blocks = [100_000, 0, 0]` (1.6M voxels out at density 16); `--demo-far-offset-near`
    builds the SAME box at the origin for A/B comparison. A `FAR_OFFSET_BLOCKS` const documents the
    offset. Both go through the single-node resolve path; `placed_scene` includes them so the region
    is the box's own 4³ extent.
  - **What the two PNGs show** (`shots/s1_far_offset.png` vs `shots/s1_far_offset_near.png`):
    geometrically IDENTICAL — a crisp grey box, clean edges, NO geometric jitter/wobble/distortion,
    nothing missing. The box silhouette is pixel-identical. This is because **`resolve_region`
    recentres the composite on its own centre**, so a lone far box is mapped straight back to the
    origin before meshing; the mesh vertices are tiny chunk-local f32 values → no far-lands jitter in
    geometry. **HOWEVER**, the PNGs are NOT byte-identical: ~0.207% of pixels differ (max channel
    delta 228) as a faint procedural-surface-noise SPECKLE. Cause: the per-voxel `world_position` is
    stored as f32 at ~1.6M (ULP ≈ 0.125) BEFORE the recentre subtraction, so the recentred near-zero
    position carries a tiny rounding error that shifts the position-keyed surface-noise phase on a
    scatter of surface pixels. So pre-rebasing f32 precision loss IS observable today, but only as
    subtle shading speckle, not as geometric jitter — the geometry recenter hides the structural
    jitter until S4 removes the recentre.
  - **Durable artifact — pure-CPU test** `scene::tests::far_offset_node_resolves_to_absolute_coords_near_100k`
    (in `src/scene.rs`): via the S0 absolute-coord chunk path (`resolve_region_via_chunks` /
    `covering_chunk_range`, which do NOT recentre), asserts the far box really lands far in absolute
    space — every voxel's absolute X is in `[99_998·d, 100_002·d)`, mean absolute X ≈ 1_600_000
    voxels (block 100_000 × 16), owning chunk-X > 1000 (chunk ≈ 25_000), and `recentre_voxels[0] ==
    100_000 × density` exactly. Cross-checks that the recentred render box and the absolute far box
    are the SAME shape differing ONLY by the recentre (= the far placement) — pinning that the
    recentre is exactly what S4 will remove.
  - **Absolute coords the far node resolves to** (density 16): voxels at absolute X ≈ 1_598_000 …
    1_602_000 (box centred on 1_600_000), Y/Z centred on 0; owning chunk-X ≈ 25_000
    (chunk_extent = 4·16 = 64 voxels). **For S4:** prove "no jitter at distance" by re-rendering
    `--demo-far-offset` after the recentre is replaced by 64-bit/camera-relative rebasing and
    asserting it stays geometrically crisp AND the procedural speckle disappears (the far and near
    PNGs become byte-identical once positions are rebased to small chunk-local values before f32
    downcast) — i.e. the S1 ~0.2% pixel diff regressing to 0 is the S4 success signal.

- **Chunk-addressable resolve (additive, behaviourally identical) (S0) — Part of #27** — the first,
  safest step toward deep chunked resolve (issue #27). **Purely additive: the live render path is
  untouched** (`Scene::resolve_region` still allocates one whole-region grid AND recentres the
  composite on the origin, exactly as before; goldens stay green). Two new functions in
  `src/scene.rs`:
  - `Scene::resolve_chunk(chunk_coord: [i32;3], voxels_per_block, lod) -> VoxelGrid` — resolves
    exactly ONE `CHUNK_BLOCKS³`-block chunk (`CHUNK_BLOCKS = 4`, reused from `renderer.rs`) in
    **absolute (non-recentred) composite voxel coordinates**. Reuses the same per-leaf
    resolve+stamp logic as `resolve_region` (via a new `stamp_producer_into_chunk` helper) but
    clips each voxel to the chunk's box. `lod` is the parked LOD seam — always `0`, asserted,
    present for forward-compat (ADR 0002 Decision 2).
  - `Scene::resolve_region_via_chunks(voxels_per_block, lod) -> VoxelGrid` — loops the covering
    chunk range, resolves each chunk, unions them into one full grid (absolute coords). Proves the
    decomposition reconstructs the whole scene; NOT wired into rendering.
  - **Chunk-AABB ownership (the rule the next step inherits):** a chunk covers the **half-open**
    absolute-voxel box `[chunk_coord·E, (chunk_coord+1)·E)` per axis, `E = CHUNK_BLOCKS·density`.
    A voxel belongs to `floor(world_position / E)`. Voxel centres sit at `n + 0.5` and boundaries
    at integer multiples of `E`, so `floor` is never ambiguous → **each voxel lands in exactly one
    chunk**. (`covering_chunk_range` uses `div_euclid` so negative chunk coords work; the high
    chunk owns `max_voxel − 1`.)
  - **Verification — pure-CPU unit tests** asserting the chunk-reassembled occupied set EXACTLY
    equals the monolithic `resolve_region`'s set (position + `material_id`), after normalising for
    the recentre: the two frames differ by exactly `recentre_voxels`, so the test adds it back to
    the monolithic frame (`chunked.world_position == resolve_region.world_position + recentre`).
    Also asserts no chunk emits a voxel outside its own AABB and that summed per-chunk counts equal
    the whole count (exactly-one-chunk). Coverage: all 5 SDF shapes, the `--demo-scene` and
    `--demo-village` (instanced) multi-node scenes, an **off-centre node (+8 blocks, non-zero
    recentre)**, an empty far-off chunk (no panic, empty grid), and density 8. **No discrepancy
    found — chunked and monolithic resolve are bit-identical (modulo recentre + order).**
  - Gates green: `cargo build --bins`, `cargo clippy --all-targets` (0 warnings), `cargo test`
    (**101 lib tests**, was 95 + 6 new), `cargo test --features gpu --test golden` (unchanged).

- **Make cuboid the DEFAULT mesher; rebaseline goldens from the cuboid path (E3c-2) — Part of #18**
  — with full feature + multi-material parity (E3b) and the texture atlas (E3c-1) done, the cuboid
  box-decomposition path is now the **default** render path. `MesherChoice`'s `#[default]` flipped
  `Instanced → Cuboid`, which cascades to the windowed app (`PanelState`/settings start cuboid) and
  the `shot` binary (`ShotOptions::default()` now uses `MesherChoice::default()`). The legacy
  instanced path is **kept fully working** behind `--mesher instanced` (and the panel checkbox,
  relabeled "Cuboid mesher (default; off = legacy instanced)") as a debug fallback — verified it
  still renders (sphere via `--mesher instanced` draws the per-voxel-cube disc, no cuboid log line).
  - **Golden rebaseline (deliberate, NOT pixel-match).** Per ADR 0002 the cuboid path is ~3–5%
    different full-frame from instanced (merged-face triangulation, edge AA, procedural-noise phase,
    surface shading) — expected and acceptable. Regenerated all 5 references via
    `UPDATE_GOLDENS=1 cargo test --features gpu --test golden`; each was produced by the cuboid path
    (confirmed by the `cuboid mesher: N boxes → M faces` log per case). The win is visible in the
    primitive counts: sphere 53 776 voxels → 545 boxes/2242 faces; cylinder 80 384 → 47 boxes;
    torus 242 984 → 3575 boxes; village 157 696 → 40 boxes; clouds 147 588 → 5953 boxes.
  - **Visual review (all 5 read + confirmed correct):** `sphere-debug-faces` — flattened disc with
    correct outward-normal debug colours (green top, R/B/C/M sides), correct winding, no back-face
    stripes; `cylinder` — solid stone disc, crisp silhouette; `torus` — clean stone ring with the
    central hole; `demo-village` — 4 separated houses, stone body + wood chimney each (distinct
    materials, no atlas bleed, reuse-by-reference); `debug-clouds` — several distinct stone blobs in
    a mostly-empty volume. No blanks/corruption/missing textures.
  - Files: `src/panel.rs` (`MesherChoice` default + checkbox label), `src/bin/shot.rs`
    (`ShotOptions::default()` + help text), `src/settings.rs` (comment), `docs/adr/0002…` (O8/switch
    marked done), `tests/golden/*.png` (5 rebaselined references).
  - Green checkpoint: `cargo build --bins` clean; `cargo clippy --all-targets` no new warnings;
    `cargo test` 95 lib tests pass; `cargo test --features gpu --test golden` GREEN against the new
    cuboid baseline; `--mesher instanced` sanity render confirmed.

- **Cuboid path reaches FULL parity: layer-range band clip + `--debug-faces` (E3b-3) — Part of #18**
  — added the last two features so the flag-gated cuboid mesher (`--mesher cuboid`) matches the instanced
  path on everything. The instanced path + goldens are untouched (default stays instanced; all 5 goldens
  pass unchanged).
  - **Layer-range band clip:** the instanced shader discards fragments per voxel-layer, but a fragment
    discard on the cuboid path's *merged* boxes leaves the displayed slab **open-topped** — a single tall
    column's only +Y face is at the model's true top, so it gets clipped away with no cap. So the cuboid
    path clips the band at **mesh-build time**: it masks the densified region to the band's absolute
    Y-layer range `[band_min, band_max]` (inclusive) *before* decomposition, so the greedy mesher caps the
    slab with real top/bottom faces exactly like the instanced slab's per-voxel faces. Region-local Y maps
    to the absolute layer by a constant `base_layer = floor(world_offset.y + 0.5 + half_y)`. The mesh
    re-builds only when the band changes (cached `current_band`), re-uploading the vertex/index buffers;
    the band uniforms are still carried for std140 parity but unused by the shader. Verified A/B
    (`sphere 6³ --layer-lower 48 --layer-upper 48`, plus a `box 4³` mid-band slab): the cuboid slice is
    **pixel-identical** to the instanced slice (foreground sym-diff = 0 over 183k px).
  - **`--debug-faces`:** ported the instanced cull-off debug pipeline + shader to the cuboid path —
    a second `cull_mode: None` pipeline (selected when the uploaded `debug_face_mode` is on) plus the
    identical `debug_face_color` normal→colour palette (+X red, -X cyan, +Y green, -Y magenta, +Z blue,
    -Z yellow) and the back-facing black-stripe marker; texture/material/overlay/band-clip are all bypassed
    in that mode (matching instanced). Verified A/B (`sphere --debug-faces`): same R/G/B outward faces, no
    back-face stripe (correct winding/cull), palette mismatch 0.07% (AA edges only).
  - **Result:** the cuboid path now supports texture slice + grid overlay + per-voxel material + layer
    clip + debug-faces — **full feature parity** with the instanced path (next step is the texture atlas).
  - Files: `src/cuboid_mesh.rs`, `src/shaders/cuboid.wgsl`, `src/renderer.rs` (`LayerBand: PartialEq`),
    `src/voxel.rs` (`VoxelGrid: Clone`), `src/bin/shot.rs`, `src/main.rs`. New unit tests:
    `band_clip_masks_region_and_caps_the_slab`, `band_clip_outside_occupied_layers_is_empty`.
  - Green checkpoint: `cargo build --bins` clean; `cargo clippy --all-targets` + `--features gpu --tests`
    clean; `cargo test` 83 pass; goldens (`--features gpu --test golden --lib`) all pass. (`cargo test`
    golden relink hit `voxel_worker.exe` "Access is denied" because the windowed app was running, so the
    goldens were run with `--lib` to skip the bin relink.)

- **Fix cuboid mesher partial-silhouette bug: shift-invariant densification — Part of #18**
  — the flag-gated cuboid path rendered the cylinder as ~1/4 of its disc (a wedge) while the instanced path
  drew the full disc; sphere/village slipped through because they happened to render. **Root cause:**
  `build_cuboid_mesh` densified the grid via `region_from_voxel_grid(grid, [0,0,0], dimensions)`, which uses
  the project-wide `round(world + dimensions/2 - 0.5)` index convention anchored at index 0. That assumes the
  voxel cloud is perfectly centred on `dimensions/2`. But `Scene::resolve_region` RECENTRES a composite by a
  non-zero offset for odd block sizes (a 5-block axis shifts the cloud by 8 voxels — its `(min+max)/2` block
  midpoint is +0.5 block off the node centre). The shifted cloud's convention indices ran negative / past
  `dimensions`, so the densifier silently dropped every out-of-bounds voxel: the cylinder kept only
  **36 032 of 80 384** voxels (~45%) → a wedge. The instanced path is immune because it draws raw
  `world_position`s; only the cuboid path, which rebuilds geometry from the dense region, was affected. The
  decomposition algorithm itself was never wrong (exact-cover holds on whatever region it is handed).
  **Fix:** new `region_from_voxel_cloud(grid)` densifies anchored on the cloud's OWN minimum voxel
  (`round(world - min_world_center)`, always ≥ 0) and returns the matching `world_offset = min_world − 0.5`
  so the mesh sits exactly where the instanced voxels are. Shift-invariant — a recentred composite lands
  fully in-bounds; a centred grid collapses to the old behaviour. **Regression test:**
  `cuboid_covers_every_voxel_for_all_shapes` asserts the box set covers EXACTLY `grid.occupied.len()` voxels
  for cylinder/sphere/torus/box/tube across three sizes AND a deliberately +8-shifted copy — directly
  prevents partial-coverage for any shape. **Verify:** 81 lib tests pass (80 + new); clippy clean
  (`--all-targets`, `--features gpu --tests`); all 5 goldens unchanged (instanced default untouched). Cuboid
  cylinder now renders the FULL disc; silhouette A/B vs instanced dropped from a ~1/4-disc wedge to a
  thin-rim 0.95% (just MSAA edge AA on merged-face triangulation). Regression-sweep silhouette mismatch
  (foreground mask, cuboid vs instanced): cylinder 0.95%, sphere 0.17%, torus 0.76%, box 1.26%, tube 1.40%,
  village 0.006%, clouds 0.003% — all full, correct silhouettes (residual is edge AA / surface texture
  phase, not geometry). NB the cuboid path is not pixel-identical to instanced even where there was never a
  bug (different triangulation + per-merged-face texture tiling), so full-frame A/B sits ~3–5%; the
  geometry/silhouette is what this fix targets and it now matches.

- **Cuboid faces: per-voxel texture slice + position-based grid overlay (ADR 0002 E3b-2) — Part of #18**
  — extends the flag-gated cuboid path (still default OFF; instanced path untouched) so a merged box face
  reads IDENTICAL to per-voxel cubes. **Per-voxel texture tiling:** each box-face fragment derives a UV in
  VOXEL units from its absolute voxel position (`world + grid_half_extent`) on the face's two in-plane axes;
  a new `Repeat` material sampler tiles the block texture once per voxel, so a face spanning N voxels shows
  N tiles phase-aligned to voxel/block boundaries. The per-face UV DIRECTION (and the block-local slice
  offset) replicate the instanced `unit_cube_geometry` face UVs exactly (`coord_component(a, sign)` mirrors
  the UV within each voxel for negative-direction faces), so even non-symmetric textures land texel-exact —
  box-stone A/B dropped from ~9% to 0.09%, box-wood to 0.005%. **Per-face texture:** the cuboid path now
  binds the SAME 6-layer `D2Array` material the instanced path uses (3 procedural Stone/Wood/Plain bind
  groups, selected by the bound material), with `face_layer(normal)` picking the layer (same mapping as
  instanced). **Grid overlay:** the per-voxel + per-block lines are drawn from the absolute voxel position
  (NOT face UVs — honors the project guard) with the EXACT instanced colours/half-widths/alphas (exposed
  via new `renderer::grid_overlay_params()`), respecting the `--grid` / Display toggle. **Material colour:**
  the E3b-1 per-box modulation still multiplies the lit texture (texture × material × lighting), same as
  instanced. **Files:** `src/cuboid_mesh.rs` (uniform grew to carry half-extent/density/overlay; new Repeat
  sampler + per-face material bind groups; `update_uniforms` + `new` take queue/dims/density/overlay),
  `src/shaders/cuboid.wgsl` (UV slice + layer + overlay), `src/renderer.rs` (3 new pub helpers: overlay
  params, procedural pixels, texture size — instanced path itself unchanged), `src/bin/shot.rs` +
  `src/main.rs` (call-site args). **Verification:** `cargo build --bins`, `cargo clippy --all-targets`
  (+`--features gpu --tests`), `cargo test --lib` (80 pass incl. 2 new: per-face UV span 0..N, normal→layer
  mapping) all clean; `cargo test --features gpu --test golden` — all 5 goldens UNCHANGED (4 bit-exact,
  debug-clouds 0.21%). **A/B (cuboid vs instanced golden, golden metric):** torus **0.32%** and demo-village
  **0.21%** now PASS the 0.5% tolerance; cylinder **3.35%** remains higher — but that residual is a
  PRE-EXISTING E3b-1 geometry defect (the cuboid cylinder renders a quarter-disc wedge; confirmed identical
  in the untouched-geometry baseline), NOT a texture/overlay gap (texture parity is exact). Read-back PNGs
  confirm the block texture tiles per-voxel and the grid lines appear on cuboid faces.

- **Cuboid mesh render path behind a flag (ADR 0002 E3b-1) — Part of #18**
  — new `src/cuboid_mesh.rs` + `src/shaders/cuboid.wgsl`; the experimental cuboid-mesher render path,
  default OFF, selected by a flag. The DEFAULT instanced path is byte-for-byte unchanged (all 5 goldens
  pass: 4 bit-exact, debug-clouds 0.21% < 0.5%). **Flag:** `MesherChoice::{Instanced,Cuboid}` (new in
  `panel.rs`, default `Instanced`); `shot --mesher <instanced|cuboid>`; a Display-section checkbox
  "Cuboid mesher (experimental)" in the app (session-only, not persisted). **Boxes→mesh:** when cuboid
  is selected, the whole grid is `region_from_voxel_grid` + `decompose_into_boxes`'d into single-material
  boxes (`src/cuboid.rs`), then `build_cuboid_mesh` emits a triangle quad per **exposed** box face only —
  a face is culled when every neighbour cell just beyond it (scanned across the face's two in-plane axes)
  is solid; box-internal faces and faces backed by adjacent solid voxels (even of a different material)
  are dropped, so the silhouette is the solid set's outer surface. Each face vertex carries the box's
  `material_id` + the face's outward normal, CCW-wound (matching the instanced cube), `cull_mode: Back`.
  Boxes are bucketed into the SAME `CHUNK_BLOCKS=4` chunk partition the instanced path uses (keyed by the
  box min-corner voxel), each chunk's world-AABB frustum-culled per frame (reusing `frustum.rs`). **Shader
  reuse:** `cuboid.wgsl` flat-shades with the SAME directional+ambient lighting constants as `voxel.wgsl`
  and the SAME step-3b per-material relative base-colour modulation (exposed via
  `renderer::relative_material_base_colors_public`) — NO texture slice / grid overlay / layer clip /
  debug-faces yet (later E3 sub-steps), so the cuboid render is flat colour + lighting, not golden-matching
  by design. **Selection in `render_frame`:** new `FrameOverlays::cuboid_mesh: Option<&CuboidMeshRenderer>`
  — when `Some` it draws INSTEAD of `voxel_renderer.draw` in the same MSAA pass; `None` (default) leaves the
  instanced draw untouched. **Verification:** `cargo build --bins`, `clippy --all-targets` (+ `--features gpu
  --tests`), `cargo test` (78 lib + 5 new `cuboid_mesh` unit tests) all clean; goldens pass UNCHANGED.
  Mesh sanity tests: a 1-voxel cube → 1 box → 6 faces / 12 tris / 36 indices / 24 verts; a 2-voxel run → 1
  merged box → still 6 faces; a solid 4³ block → 1 box → 6 faces; two adjacent boxes of different materials
  → 10 faces (the 2 shared faces culled). **Primitive reduction (the point):** sphere 6×6×6@16 = 463,400
  instanced voxels → **3,083 boxes / 12,350 exposed faces / 24,700 triangles** (~37× fewer); demo-village =
  157,696 voxels → **40 boxes / 172 faces / 344 triangles**. Shape-parity shots confirmed visually: the
  cuboid silhouette matches the instanced one for both sphere and village (4 houses, correct shapes +
  per-box stone/wood materials), flat-shaded with no texture/grid as expected
  (`shots/e3b1-{sphere,village}-{instanced,cuboid}.png`). Risk/subtlety: a merged box face is emitted
  whole if ANY neighbour cell beyond it is air (over-draw of at most one box face, never a hole — fine for
  shape parity); face winding reuses the instanced CCW-from-outside convention so `cull_mode: Back` keeps
  outward faces. A WGSL std140 gotcha was hit + fixed: a trailing `vec3` pad forces 16-byte alignment
  (144-byte struct) vs the Rust 128-byte buffer — replaced with three f32 scalars.

- **Greedy cuboid (box) decomposition — pure CPU algorithm (ADR 0002 Decision 1, E3) — Part of #18**
  — new `src/cuboid.rs` (registered in `lib.rs`). This is the box-decomposition algorithm ONLY (no
  rendering / GPU / flag — those remain in #18). `decompose_into_boxes(region: &VoxelRegion) ->
  Vec<VoxelBox>` turns a bounded region of solid, materialled voxels into a minimal-ish set of
  axis-aligned, single-material cuboids, VS-style (`BlockEntityMicroBlock.GenShape`). **Input** is a
  representation-agnostic dense `VoxelRegion { extent: [w,h,d], cells: Vec<Option<u16>> }` (row-major,
  X fastest — the same `(z*h+y)*w+x` order as `renderer::upload_grid`'s occupancy volume; `Some(id)` =
  solid, `None` = air). **`VoxelBox { min:[u32;3], max:[u32;3], material_id:u16 }` uses INCLUSIVE min
  AND INCLUSIVE max** (a single voxel is `min == max`; documented on the type). **Algorithm:** scan
  cells in fixed `(z, y, x)` order (deterministic); for each unconsumed solid seed, grow the run
  greedily +X (same material, unconsumed), then grow the whole X-run +Y, then the whole XY-slab +Z,
  mark all covered cells consumed, emit the box. **Adapter** `region_from_voxel_grid(grid, origin,
  extent)` builds a region from a `VoxelGrid` sub-box using the project-wide
  `round(world_position + dims/2 - 0.5)` index mapping, so the E3 rendering task can call it per
  render-chunk. Core stays pure (no GPU, no `VoxelGrid` dependency in the algorithm itself).
  **Verification:** `cargo build --bins`, `cargo clippy --all-targets`, `cargo clippy --features gpu
  --tests` all clean; `cargo test` 73 pass (was 62; +11 new). 11 exhaustive `cuboid.rs` tests assert
  the three invariants (exact cover / no overlap / single material) programmatically over every cell:
  single voxel → 1 box, full block → 1 box (not W·H·D), two-material split → 2 boxes, L-shape +
  5×5 ring (holes/concavity never covered), empty + zero-extent → 0 boxes, determinism (same input →
  identical output), a deterministic-LCG randomized safety net over 7 extents × {1,2,3,5} materials ×
  5 fill densities, and the box-count-reduction sanity (solid 4×4×4 → 1 box, not 64). Two adapter
  tests round-trip the `VoxelGrid` index mapping (whole grid + offset sub-region). Does NOT touch the
  renderer. Rendering the boxes + texture atlas remain open in #18.

- **Chunked instanced rendering + per-chunk frustum culling (ADR 0002 E2) — Part of #19** — the
  instanced voxel renderer now partitions the single resolved `VoxelGrid` into spatial chunks and
  frustum-culls them per frame, retiring the 450k draw-side truncation. **Chunk size** is
  `CHUNK_BLOCKS = 4` blocks/axis (= `4 * voxels_per_block` voxels/axis; ADR 0002 Decision 3).
  `bucket_instances_into_chunks(grid, voxels_per_block)` (new in `renderer.rs`) buckets every occupied
  voxel by `floor(world_position / chunk_extent)`, lays the single instance buffer out chunk-by-chunk
  (chunk keys sorted so the layout — and thus the goldens — is deterministic), and records each
  chunk's instance range + world AABB. `MAX_DRAWN_INSTANCES` (the 450k cap + `instances_from_grid`'s
  `.take()`) is **deleted** — every voxel in a visible chunk draws, so a scene up to the unchanged 6M
  `MAX_GRID_VOXELS` resolve cap renders fully instead of being cut at ~450k. New `src/frustum.rs`:
  Gribb–Hartmann plane extraction from the camera `view_projection` + a positive-vertex AABB test
  (never a false negative, so on-screen geometry is never wrongly culled); `update_uniforms` (now
  `&mut self`, already had the matrix) runs the cull and stores the visible-chunk list, and `draw`
  emits one `draw_indexed` per visible chunk over its instance range. All per-voxel attributes
  (position, block-local coord, material_id) and every shipped feature — per-face textures, per-voxel
  slice, grid overlay, layer band clip, `--debug-faces`, onion fog (still samples the untouched
  single-grid occupancy texture) — carry through per-chunk transparently. New `--debug-chunks` shot
  flag prints `chunks: drew X / Y`. **Verification:** `cargo build --bins`, `clippy --all-targets`
  (+ `--features gpu --tests`), `cargo test` (62 pass incl. new bucketing/AABB + frustum unit tests)
  all clean; **all 5 golden cases pixel-identical** (`cargo test --features gpu --test golden`);
  `sphere 6×6×6 @16` = 463,400 voxels (> old 450k cap) now renders a COMPLETE sphere across 8/8
  chunks; a zoomed-in `--demo-village` drew **8 / 16** chunks (off-screen half culled) with the
  on-screen geometry intact. SCOPE held: no chunked resolve, no coord rebasing, no meshing, no fog/UI
  changes. Files: `src/renderer.rs`, `src/frustum.rs` (new), `src/lib.rs`, `src/main.rs`,
  `src/bin/shot.rs`.

- **Scene persistence + migration (ADR 0001 step 8) — Closes #21** — the whole scene now persists,
  not just the single active-Tool geometry. Added `serde::{Serialize, Deserialize}` (+ `PartialEq`)
  to the scene model: `Scene`, `Node`, `NodeContent`, `Part`, `AssemblyDef`, `NodeTransform`,
  `CombineOp`, `DefId`, `NodePath`, and `SdfShape`. Every field is `#[serde(default)]` (with named
  default fns where the type has no `Default`, e.g. `SdfShape::kind`, `Node::visible`), honoring the
  flat-tolerant-mirror convention: a missing field falls back to default and loading never panics.
  `AppConfig` gained `scene: Option<Scene>`; `capture()` serializes `panel.scene`, and the legacy
  flat `shape/size_blocks/voxels_per_block/wall_blocks` fields are still written so a new config also
  opens in an older build. **Migration:** an OLD config with no `scene` field deserializes to `None`,
  which `to_panel_state()` routes through `seed_scene_from_geometry()` → a one-Tool-node scene from
  the flat params. A `Some(scene)` that resolves to no nodes (a malformed/empty persisted scene) is
  treated as absent → same seed, so load never yields an empty document. `voxels_per_block` stays an
  app-level field (ADR 0001 "Density"). **Deferred:** regional/streamed `.vox` export — meaningless
  until the chunking milestone; the current full-grid export already covers bounded scenes (noted in
  `settings.rs`). Tests: full non-trivial scene round-trip (Tool+Part+Group+AssemblyDef+Instance,
  structural equality + identical resolved occupancy), extended migration test (flat config →
  one-Tool-node matching the params), and a malformed-scene fallback test. `cargo build --bins` +
  `clippy --all-targets` + `clippy --features gpu --tests` clean; 55 tests pass; `shot --shape
  sphere` unchanged (53776 voxels).

- **Central 3D viewport — Closes #25** — the 3D pass used to render into the FULL window with the
  camera aspect computed from the whole window, so the egui right side panel + bottom palette dock
  (painted on top) covered part of the render and the model sat off-centre, partly hidden behind the
  side panel. Fix: after egui lays out its panels in `run_egui_frame`, capture the post-panel central
  area (`ui.available_rect_before_wrap()` × `pixels_per_point`, clamped into the target) and return it
  on `PreparedEguiFrame` as `viewport_px: [u32;4]` (x, y, w, h, physical px). Both callers now compute
  the camera aspect from `viewport_px` w/h (reordered so uniforms upload AFTER egui runs — `shot.rs`
  moved its whole camera/overlay/voxel uniform upload below `run_egui_frame`). In `render_frame` the
  voxel/gizmo/lattice MSAA pass, the onion-fog pass, and the view-cube pass all `set_viewport` +
  `set_scissor_rect` to the central rect (the target is still CLEARED full-screen to the workshop
  colour, so any uncovered sliver isn't garbage; only the 3D draws are confined). The view cube is
  positioned at the central rect's top-left + margin (not the window's), and the windowed view-cube
  hit-testing (`position_in_view_cube` / `pick_view_cube_element`) offsets by the cached
  `last_viewport_px` so clicking the cube still works. The origin gizmo follows automatically (centred
  via the camera). Files: `src/lib.rs`, `src/main.rs`, `src/bin/shot.rs`, `src/renderer.rs`,
  `PROGRESS.md`, and 5 regenerated `tests/golden/*.png` (layout changed intentionally — re-verified
  centred + golden test green). No CentralPanel background was added (the centre stays see-through to
  the 3D, per scope). build/clippy(`--all-targets`, `--features gpu --tests`)/test all clean; the
  sphere/village/gizmo shots confirm the model is centred LEFT of the panel with the view cube
  repositioned to the central top-left.

- **Golden-image regression harness — Closes #24 (E0 safety net for ADR 0002)** — added
  `tests/golden.rs`, a GPU-gated (`#![cfg(feature = "gpu")]`) integration test that renders 5
  canonical cases through the REAL `shot` binary (located via `CARGO_BIN_EXE_shot`) at a fixed
  `--width 1280 --height 720` and fixed orbit angles (`--theta 0.7 --phi 1.05`, auto-framed distance)
  into temp PNGs, then tolerance-compares each against a committed reference under `tests/golden/`.
  Cases: `sphere --debug-faces`, `cylinder`, `torus 8×2×8`, `--demo-village` (instanced scene graph),
  `debug-clouds 64³ @density 2`. Tolerance: a pixel "differs" when its max per-channel abs diff > 8/255;
  the test fails when > 0.5% of pixels differ (on this RTX machine all 5 cases render bit-exact run-to-run
  — observed 0.00000% across two repeated runs). `UPDATE_GOLDENS=1` rewrites the references instead of
  comparing (for intended visual changes). On failure it writes `<case>-actual.png` + `<case>-diff.png`
  to a temp dir and prints the mismatch fraction. The 5 reference PNGs are committed under `tests/golden/`
  (whitelisted in `.gitignore`). When the cuboid mesher replaces the renderer, these goldens prove the
  pixels did not change. Verified: `cargo build --bins` + `cargo clippy --all-targets` + `cargo clippy
  --features gpu --tests` all clean; default `cargo test` still 53 pass (golden correctly NOT compiled);
  `cargo test --features gpu --test golden` passes twice with 0.00000% mismatch per case. Run/regen
  documented in `docs/DEV_NOTES.md`.

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
