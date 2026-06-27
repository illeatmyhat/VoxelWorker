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

- **Foundation A1b: relocate `MaterialChoice`/`GeometryParams` out of `panel.rs` (Part of #34, epic #33).** Pure move, zero behaviour change: fixed the wrong-way `scene → panel` dependency — `MaterialChoice` (a material primitive mapping to `material_id`) moved to the bottom-layer `core_geom`, and `GeometryParams` (the UI mirror of `SdfShape`) moved beside `SdfShape` in `voxel`. Repointed every importer (`panel`, `scene`, `settings`, `cuboid_mesh`, `renderer`, `texture_atlas`, `chunk_cache`, `chunk_storage`, `vox_export`) and the `lib`/`shot` re-exports; `scene.rs` no longer imports `panel` for these types. Gate green: 248 lib tests, clippy clean, goldens 9/9 byte-identical.

- **Foundation A1a: move `CHUNK_BLOCKS` → new `core_geom` module (Part of #34, epic #33).** Pure relocation, zero behaviour change: lifted the streaming-quantum constant out of the GPU module (`renderer.rs`) into a new dependency-free `src/core_geom.rs` (ADR 0003 bottom layer), and repointed every importer (`renderer`, `spatial_index`, `chunk_cache`, `cuboid_mesh`, `scene`, `voxel`, `main`, `bin/shot`) at `core_geom::CHUNK_BLOCKS`. No re-export from `renderer`. Gate green: 248 lib tests, clippy clean, goldens 9/9 byte-identical.

- **Wire region-scoped `.vox` export + diameter readout into the live app (Step 2) — Part of #20.**
  The interactive export button and the diameter/scrubber readout now call the parity-proven
  region-scoped (per-chunk) methods instead of the monolithic whole-grid path — decoupling those two
  consumers from the assembled grid (Step-4 prep). Pure CPU; fully headless. Goldens 9/9 unchanged
  (export/diameter don't touch rendering).
  - **Export.** `main.rs::WindowedState::export_vox` now calls `ChunkResolveCache::vox_export` (drops
    `resolve_scene` + `VoxExport::from_grid`; removed the now-unused `VoxExport` import from `main.rs`).
  - **Diameter.** Both readout call sites now use `ChunkResolveCache::widest_run_in_band`: the startup
    seed in `WindowedState::new` (the resolve cache is now built BEFORE the diameter so the startup
    readout uses the same path, caching the chunks for reuse) and the per-frame re-measure in
    `render`. `self.grid.widest_run_in_band(..)` is gone from `main.rs`.
  - **FINDING — Low #1's "far-offset export fix" does not hold.** Routing export through `vox_export`
    is behaviour-EQUIVALENT to the monolithic path, not a far-offset accuracy fix. Empirical sweep
    (offsets 1k→1M blocks): region and monolithic exports are model-set-IDENTICAL at every offset
    (same voxel set, only emission order differs), and at very large offsets BOTH lose the voxel-centre
    `.5` identically. Reason: `.vox` export must bucket into the region-relative `[0, grid_x)` frame, so
    both paths add `half_x` (≈ region half-width) in f32; once an axis exceeds ~2^24 voxels the `.5` is
    unrepresentable regardless of path. The i64-rebase-before-downcast trick only helps the RENDER
    frame (camera-relative floating origin keeps near chunks small), not export (which needs
    absolute-ish coords). The rewiring's real value is the Step-4 decoupling. Tests reflect the truth:
    `vox_export::far_offset_region_export_equals_monolithic` (model-set equality far out, both keep the
    full per-chunk voxel count at a still-f32-safe offset) + `far_offset_region_export_round_trips_full_voxel_set`
    (build → bytes → `dot_vox::load_bytes`, exercising the export wiring end-to-end minus the dialog),
    and `chunk_cache::region_widest_run_correct_at_far_offset` (region readout == true 48-voxel box
    width == whole-grid, far out). +3 lib tests (245 → 248, all green; clippy clean).
  - **No dead-code deletions.** Both whole-grid methods are still consumed and so were LEFT:
    `VoxelGrid::widest_run_in_band` (used by `src/bin/shot.rs`, `tests/palette_click.rs`, and as the
    parity ground-truth) and `VoxExport::from_grid` (used by `shot.rs` and the parity tests). The
    monolithic `resolve_scene`/`resolve_region` assembly stays — the renderer/fog still consume it
    (Step 4, deferred).
  - **Still needs an interactive smoke-test (user-AFK gap):** click File→Export `.vox`, confirm the
    save dialog appears + writes a parseable `.vox`; and read the diameter stat in the panel as you
    drag the layer-band scrubber to confirm it updates. The pure scene→`.vox` core and the diameter
    function are verified headlessly; only the egui file-dialog + on-screen readout plumbing is untested.

- **Out-of-core: spill resident chunks to `DiskChunkStore` (Step 3) — Part of #20.** Wires the
  standalone `DiskChunkStore` (S6b) into `ChunkResolveCache` so an over-cap resident set spills its
  least-recently-used chunks to disk (compressed via `chunk_storage`) and reloads them transparently on
  the next access. Pure CPU; fully verified headlessly. Render output unchanged — goldens 6/6 still green.
  - **Opt-in, default unchanged.** `ChunkResolveCache::new()` stays UNBOUNDED (never spills) — the live
    path (renderer, `shot`, `vox_export`) and every golden/parity test behave exactly as before. A new
    `ChunkResolveCache::with_resident_cap(max_resident_chunks, disk_dir)` opts into spilling (panics on cap
    0; errors if the dir can't be created). All spill logic is gated behind `max_resident_chunks.is_some()`.
  - **Three-tier lookup (`ensure_resident`).** (1) resident hit → refresh LRU; (2) disk hit → decompress +
    promote back to resident (`disk_reload_count`); (3) miss in both → resolve via the scene
    (`recompute_count`). `insert_resident` spills the LRU OTHER chunk first when an insert would breach the
    cap, so `resident_chunk_count() <= cap` ALWAYS. LRU is a per-key `last_used_tick` + monotonic clock;
    the smallest-tick resident chunk is the spill victim (`spill_count`).
  - **Invalidation purges BOTH RAM and disk** so a stale spilled chunk never resurfaces: `clear` and a
    rebind (`rebind_if_changed`, density/origin change) call `disk_store.clear()`; `invalidate_chunk`
    (`evict_coord_everywhere`) and `invalidate_aabb` purge the disk store across the dirtied coord(s). A
    rebind MUST drop disk too — a spilled chunk is keyed/serialised in the OLD binding (the S6c wiring-note
    correctness condition: a far chunk would otherwise reload mis-placed).
  - **New `DiskChunkStore` API.** `remove(key)` (forget from RAM + delete disk file, idempotent) and
    `clear()` (forget every key + delete all files) — the cache needs targeted/total purge for invalidation
    (the store had only put/get/contains). +2 store tests.
  - **Compression is lossless to the f32 bit** (`chunk_storage` round-trip), so spill+reload returns a
    byte-identical grid — proven by the new test.
  - **Tests (7 cache + 2 store = 9 new; lib 236 → 245, all green).** Cache:
    `spilled_and_reloaded_chunk_is_byte_identical` (a, f32::to_bits parity vs the unbounded reference),
    `resident_cap_is_never_exceeded` (b), `least_recently_used_chunk_is_spilled` (c),
    `invalidation_purges_resident_and_disk` (d, both `invalidate_chunk` + `invalidate_aabb` → recompute not
    reload), `counters_tally_an_expected_access_sequence` (e, hand-traced spill/reload/recompute),
    `unbounded_cache_never_spills`, `zero_resident_cap_panics`. Store: `remove_forgets_resident_and_spilled_keys`,
    `clear_empties_resident_and_disk`. Temp dirs are unique-per-test, RAII-cleaned. `cargo clippy
    --all-targets` clean.
  - **Step 4 / renderer note.** No renderer/main change made or needed — the renderer still calls the
    same cache API (`new()`, unbounded). The borrow-returning whole-region gather methods
    (`resident_render_chunks`, `covering_chunk_grids`) hand out ALL covering chunks at once, so they assume
    every covering chunk is resident simultaneously; they remain correct for the unbounded default and for
    a bounded cache only when the cap ≥ covering-chunk count. The spill path proper is `chunk()` /
    `resolve_region` (each chunk consumed immediately), which is fully spill-safe. When Step 4 makes the
    renderer consume per-chunk meshes under a real cap, those gather methods will need a reload-then-borrow
    pass (cannot return borrows to more chunks than the cap holds).

- **ViewCube: Step 6 — chrome polish + interaction fixes — Part of #13 (final pass before closing #13).**
  Eight smoke-test items from the #13 feedback:
  1. **Cube-drag horizontal sign FLIPPED** (`main.rs` CursorMoved): only the cube-drag path negates `delta_x`
     before `orbit_by_drag` (the scene drag keeps its sign), so grabbing the cube turns the model the natural way.
     *Interaction-feel — needs user verification.*
  2. **Hover highlights ALL cube elements** (`camera.rs` `is_face_constrained` unused here; `renderer.rs`
     `draw` + `viewcube.wgsl`). The cube uniform's `depth_bias.x` slot now packs a 6-bit hovered-face mask
     (`cube_face_material_index`); the cube fragment shader tints matching faces teal, so a hovered face/edge/corner
     (1–3 faces) glows. `main.rs` hover now passes the REAL body picker so faces/edges/corners resolve; arrows/badges
     still light up. New shot form `--cube-hover element:<face|edge|corner>` injects a hovered element for goldens.
     *Verified via shot PNGs: front face / front-top edge / front-top-right corner each glow the right faces.*
  3. **Chrome aesthetic rework** (`renderer.rs` glyph rasterisers). `fill_triangle` now 2×2-supersamples to alpha
     (anti-aliased edges); new `fill_rect`/`blend_pixel` helpers; Home = cleaner house, Fit = four corner brackets
     (was a solid frame), roll arc disc is soft-edged, rotate triangles are crisper. *Verified via shot PNGs.*
  4. **Home also FITS unless a home was explicitly set** (`camera.rs` `HomeView.explicitly_set`; `main.rs`
     `home_snap_tween`; `settings.rs` `home_explicit`). The default home re-frames the model (auto-framed distance)
     so it never zooms in too close; a user-captured home (`from_camera`) honours its saved distance verbatim.
     Persisted as `home_explicit` (no back-compat shim). *Logic unit-tested; interactive re-fit needs user check.*
  5. **Context-menu flicker FIXED** (`lib.rs` run_egui_frame). Click-away now fires only on `primary_clicked()`
     (was `any_click()`), so the SECONDARY right-click that OPENS the menu no longer closes it the same frame.
     *Interaction — needs user verification in the windowed app.*
  6. **Rotate arrows only when face-constrained** (`camera.rs` `is_face_constrained`; `main.rs` hover + click gate).
     The four rotate arrows are a face-relative 90°-step affordance, so they only appear/act when the view is
     head-on to a face and upright (cos≤8° + roll≈0), matching Fusion. *Unit-tested; verified via shot.*
  7. **Rotate-arrow glyphs flipped** (`renderer.rs` build_chrome_vertices). The gutter arrow now points the way the
     cube content rolls (top-edge arrow points DOWN, etc.). *Direction semantics are subjective — needs user check.*
  8. **Arrow gutter widened** (`camera.rs` classify_cube_point + `renderer.rs` positions). Rotate arrows hug the
     rect edge (top/bottom/left/right bands `0..0.13` / `0.87..1.0`) instead of crowding the cube body. *Verified.*
  Green: `cargo build --bins`, `cargo clippy --all-targets`, `cargo test --lib` (236, +2 new), and
  `cargo test --features gpu --test golden` (9/9, max mismatch 0.037% on cube-chrome-hover, well under 0.5% — **no
  goldens needed regeneration**; the `cube-chrome-hover` reference still passes but now shows the OLD chrome, so the
  orchestrator may optionally refresh it to lock the new look).

- **ViewCube: real roll DOF for roll arrows — Part of #13 (Step 5, final). #13 now feature-complete (pending interactive smoke-test).**
  Added `roll: f32` to `OrbitCamera` (radians about the forward/view axis, default 0, NOT persisted — transient
  view state). The old pole-aware up logic is now `up_vector_base()`; the new `up_vector()` folds roll ON TOP of
  it: it projects the base up onto the view plane (Gram–Schmidt against `forward = normalize(target − eye)`) and
  rotates that screen-up by `roll` about forward (`glam::Quat::from_axis_angle`). roll=0 short-circuits to the raw
  base up, so the fold is a no-op at the default (existing goldens stay byte-identical). BOTH `view_projection` and
  `view_cube_view_projection` route through `up_vector()`, so the scene and the small ViewCube roll in lockstep.
  **Roll arrows:** replaced the Step-3 `ChromeClickAction::RollNoop` stub with a real roll `Snap` tween.
  `SnapTween` grew `roll_from`/`roll_to`; `advance` eases roll alongside theta/phi (same `ease_in_out_quad`). New
  `SnapTween::roll(camera, RollDir)` holds the orbit angles and targets `roll ∓ π/2` (Cw = −π/2, Ccw = +π/2,
  right-handed screen convention); `chrome_zone_left_click_action` maps `RollArrow(dir)` to it.
  **Snap resets roll (documented choice):** every face/edge/corner snap (`to_face`/`to_element`) AND Home
  (`HomeView::snap_tween`) tween `roll → 0` — a snap re-uprights the view; roll accumulates ONLY via the roll
  arrows. At rest, `advance` normalises the settled roll to (−π, π] (`normalize_roll`, via `rem_euclid`) so
  repeated arrow presses never grow it unbounded (4 quarter-turns net to ~0). `main.rs run_chrome_action` dropped
  the `RollNoop` arm (roll is now a normal `Snap`).
  **Headless:** `shot` gained `--roll <radians>` and `--roll-quarters <n>` (×π/2), folded into the camera literal.
  Rendered `--demo-village` at roll=0 vs roll=π/2 and READ both: at π/2 the whole view twists 90° — the house row
  recedes vertically, the chimneys point sideways, and the ViewCube's TOP label rotates to point sideways (scene +
  cube in lockstep, no NaN/garbage). Added golden `roll-quarter.png` (`--demo-village --roll-quarters 1`) and READ
  its 1280×720 reference (clean). **Pole interaction:** none observed — roll composes on top of the pole-blend base
  up (the base handles the singular frame; roll just rotates the resulting screen-up), and `view_matrices_finite_under_roll`
  proves both matrices stay finite at roll ∈ {0, π/2, π, −π/2}.
  7 new camera unit tests (perpendicular screen-up at π/2, roll=0 ≡ base, finite/unit/⊥-view under roll, both
  matrices finite, arrow targets ∓π/2, face/element snap resets roll→0, normalise + no unbounded growth).
  Gate: `cargo build --bins` + `cargo clippy --all-targets --features gpu` clean (no new warnings), **234 lib tests**
  pass (227 + 7), golden suite passes — the existing 8 are 0.00000% (byte-identical) and the new roll-quarter
  golden matches. Roll-arrow FEEL (∓90° tween, snap re-uprighting) is INTERACTIVE — user verifies on return.

- **ViewCube: live hover highlighting for chrome arrows — Part of #13 (Step 4).**
  Wired the LIVE hover so the rotate/roll arrows brighten when the cursor is over their zone
  (the render path already supported it: `ViewCubeRenderer::draw` brightens `hovered_zone`, and
  `FrameOverlays.cube_hovered_zone` was hardcoded `None` in the live app). Added
  `hovered_cube_zone: Option<CubeChromeZone>` to `WindowedState` (`main.rs`). In `CursorMoved`,
  after updating `last_cursor_position`, recompute it cheaply: held at `None` while orbiting/dragging
  (`left_button_held || view_cube_drag_active`), when egui consumed the move, when the cube is hidden,
  or when the cursor is outside `position_in_view_cube`; otherwise call `classify_cube_point(cube_rect, x, y, || None)`.
  The `None` body-picker DELIBERATELY skips the expensive cube raycast for hover (the body never
  highlights), so a body-region hover resolves to `None` — only arrow/Home/Fit zones light up. Then
  `render` passes `self.hovered_cube_zone` into `FrameOverlays.cube_hovered_zone` (replacing the
  hardcoded `None`). Non-interfering by construction: hover never sets orbit/drag flags, never touches
  the click dispatch (Step 3 release path) or scene input, and zeroes out exactly when those paths own
  the cursor. Goldens unaffected (headless capture hovers nothing; live hover isn't on the `shot` path).
  Gate: `cargo build --bins` + `cargo clippy --all-targets` clean (no new warnings), 227 lib tests pass,
  8 golden cases byte-identical. Highlight feel is INTERACTIVE — user verifies on return.

- **ViewCube: remove compass ring (modern Fusion has none) — Part of #13.**
  The N/E/S/W compass ring at the cube's base (added in #13 Steps 1–3) is REMOVED
  entirely — modern Fusion 360 has no such ring and at the cube's tiny scale it read
  oddly. Everything else stays: Home/Fit badges, the four rotate arrows, the two roll
  arrows, and the cube faces/edges. Removed cleanly across the three layers:
  - **Render (`renderer.rs` + `viewcube_chrome.wgsl`):** deleted the teal base-band
    annulus geometry (`push_compass_ring` + the `RingSolid` solid layer) and the four
    N/E/S/W glyph quads. The chrome-glyph texture array dropped from **13 → 8** layers
    (now Home, Fit, 4 rotate arrows, 2 roll arrows, contiguous); removed the
    compass-only `CompassNorth/East/South/West` enum variants, the `draw_glyph_letter`
    helper, and the `'S'`/`'W'` font bitmaps that were added solely for the compass
    (`'N'`/`'E'` stay — used by the FRONT/RIGHT face labels). The base band is now free
    (no chrome).
  - **Hit-math (`camera.rs`):** removed the `Compass` variant from `CubeChromeZone`, the
    base-ring rects from `classify_cube_point` (the DOWN gutter now extends to the rect
    base), the `Heading` enum, and `compass_heading_to_theta`.
  - **Dispatch (`camera.rs`):** removed the `Compass(heading)` arm from
    `chrome_zone_left_click_action` (Home/Fit + rotate/roll unchanged).
  - **Tests/flags/docs:** deleted the 3 compass unit tests (ring classify, heading
    distinctness, compass-click tween); dropped the `--cube-hover north|east|south|west`
    options from `shot.rs` (rotate/roll/home/fit kept); updated the Step-1 layout comment,
    `lib.rs` re-exports, and `cube-chrome-hover` golden comment.
  - **Verified.** `cargo build --bins` + `cargo clippy --all-targets` clean (no `#[allow]`,
    dead code removed); **227 lib tests** (was 230, −3 compass); all **8 goldens
    regenerated + pass**. Region check (old↔new): every golden changed by exactly **874 px**
    (873 for points, AA jitter) in one tight box **(23,129)–(136,142)** — the cube-corner
    base band where the ring/letters sat; the entire 3D viewport, side panel, and the rest
    of the cube are byte-identical. READ `cube-chrome-hover.png` (compass gone; Home/Fit +
    highlighted rotate-left arrow intact; houses + panel unchanged) and `cylinder.png`
    (no base ring; cube faces readable; viewport/panel unchanged).

- **ViewCube: wire chrome clicks + right-click context menu — Part of #13 (Step 3).**
  INPUT wiring only (no new visuals → goldens byte-identical). **Left-click on a chrome zone:** in
  the left-release handler, a STATIONARY release inside the cube rect now runs `classify_cube_point`
  on a shared `WindowedState::cube_rect()` (same offset/size as `position_in_view_cube`), then a new
  PURE dispatch `camera::chrome_zone_left_click_action(zone, &camera) -> ChromeClickAction` maps the
  zone to an outcome: RotateArrow→`SnapTween::to_face(adjacent_face(camera.nearest_face(), dir))`
  (new `OrbitCamera::nearest_face()` = face whose normal best matches the eye dir); Compass→theta-only
  tween to `compass_heading_to_theta(heading)` keeping phi (shortest path); Element→the existing
  element snap (body region still delegates to `pick_view_cube_element` via the `body_picker` closure);
  HomeButton→`Home`; FitButton→`Fit`; RollArrow→`RollNoop`. The windowed `run_chrome_action` executes
  it. **Drag-orbit & element-snap stay intact:** a cube drag sets `view_cube_drag_active` (gated out of
  the release path), so orbiting still wins; only the body region resolves to Element, so gutters/badges
  never hijack a body snap. **Roll-arrow stub:** a documented no-op (`RollNoop`) — the true roll DOF
  needs a camera roll field, deferred to #13 Step 5; the least-surprising stub (view doesn't jump).
  **Right-click context menu:** a right-press inside the cube rect (and `!egui_consumed`) sets
  `WindowedState::context_menu_open_at` (physical px); `run_egui_frame` draws an `egui::Area` menu there
  (Home / Fit / Orthographic↔Perspective / Set current as home). The ortho item flips
  `panel_state.projection_mode` — the SAME field the side panel binds, so menu + panel stay in SYNC.
  egui owns the menu's hit-testing so its clicks never leak to the snap path; the menu closes on
  selection or click-away. Menu selections return via `PreparedEguiFrame::cube_menu_request`
  (`ViewCubeMenuRequest::{Home,Fit,SetHome}`); the headless `shot` passes `&mut None` (no menu, so
  goldens unaffected). **Tests:** +5 pure dispatch tests in `camera.rs` (nearest_face at each face snap;
  RotateArrow(Right) from Front → Right angles; Compass(North) tweens theta keeping phi, shortest path;
  Element matches element snap; Home/Fit/Roll map to their actions). 230 lib tests (was 225) + 8 goldens
  byte-identical. **Interactive (user smoke-test on return):** the click FEEL, the right-click menu
  popup/placement/click-away, and the ortho toggle syncing both ways.

- **ViewCube chrome rendering: compass ring + Home/Fit + hover arrows — Part of #13 (Step 2).**
  RENDER only (no input wiring — that is Step 3). New **screen-space chrome overlay** path in
  `ViewCubeRenderer` (`renderer.rs` + `shaders/viewcube_chrome.wgsl`): alpha-blended textured glyph
  quads laid out in NDC within the scissored cube viewport, FIXED to the cube rect (they do NOT
  rotate with the cube), distinct from the rotating `view_cube_view_projection`. The glyph quads sit
  on EXACTLY the Step-1 `classify_cube_point` fractions (compass band y∈[.88,1], N/E/S/W sub-rects;
  Home/Fit badges top-left; rotate gutters; roll arrows top-right). A 13-layer chrome texture array
  (extends the face-label machinery) holds N/E/S/W (5×7 font, added `S`/`W` glyphs), a Home house
  icon, a Fit square-frame icon, four triangular rotate arrows, two curved roll arcs, and a solid
  layer the compass RING samples (a teal squashed-ellipse annulus on the base band, drawn as a
  48-segment screen-space triangle list so N/E/S/W stay aligned with their hit zones).
  **Always-on:** compass ring + N/E/S/W + Home/Fit (in every render). **Hover-only:** the 4 rotate
  arrows + 2 roll arrows, drawn only when their zone is the `hovered_zone`, and the hovered glyph is
  brightened to teal. `ViewCubeRenderer::draw` gains `queue` + `hovered_zone: Option<CubeChromeZone>`;
  `FrameOverlays` gains `cube_hovered_zone` (the live app passes `None` until Step 3). New shot flag
  `--cube-hover <rotate-up|down|left|right|roll-cw|ccw|north|…|home|fit>` forces a hovered zone for a
  golden. The chrome sits only in the margins/gutters/base — TOP/FRONT/RIGHT faces + teal edges stay
  readable. **Goldens: deliberate regen** (the cube corner gained chrome in all 7) + ONE new
  `cube-chrome-hover.png` (highlighted rotate-left arrow). Verified the regen is corner-only: outside
  the 170×170 cube box the old↔new goldens are 0.0000% changed (debug-clouds 0.0072%, pure AA jitter);
  inside, exactly 1330 px changed in every case (the camera-independent chrome). Gate: `--bins` +
  clippy clean, 225 lib tests, 8 golden cases pass (7 regen + hover).

- **ViewCube chrome hit-math + Home/Fit/set-home logic + persistence — Part of #13 (Step 1).**
  Pure logic + data only — NO rendering, NO input wiring (Steps 2/3), app behaviour unchanged,
  goldens byte-identical. In `camera.rs`: `CubeChromeZone` enum + `classify_cube_point(rect,
  x, y, body_picker)` — a pure screen-space classifier over the cube's square rect (zones as
  fractions of `rect.size` so Step 2 draws the chrome in the SAME pixels): Home/Fit badges
  (top-left), roll arrows (top-right), four rotate-arrow gutters around the body, a base compass
  ring (N/E/S/W L→R), and the central body delegated to the caller's raycast (`pick_view_cube_element`,
  passed as a closure → fully headless tests). `adjacent_face(face, dir)` walks two great circles —
  Up/Down vertical (Front→Top→Back→Bottom), Left/Right equator (Front→Right→Back→Left) — with
  per-circle inverses + 4-cycles (a full memoryless 6×4 inverse is geometrically impossible; that
  is documented). `compass_heading_to_theta`: N=Front(π/2), E=Right(0), S=Back(−π/2), W=Left(−π),
  90° apart, consistent with `snap_angles`. `HomeView{theta,phi,distance}` (default = camera
  defaults) + `from_camera`/`snap_tween`; `WindowedState` gains `home_view` + `set_home_to_current`
  / `home_snap_tween` / `fit_to_view` (recentre target to `Vec3::ZERO` = recentred composite
  centroid + `auto_framed_distance`, no geometry rebuild; `#[allow(dead_code)]` until Step 3 wires
  them). Persistence: `AppConfig.home_theta/phi/distance` (`#[serde(default=…)]`, old configs load
  with camera defaults); `capture` gains a `HomeView` arg, `home_view()` restores it. 15 new lib
  tests (225 total, was 210); clippy clean; golden harness byte-identical.

- **ViewCube/camera: true singular-frame up-vector (exact poles, no flip) — Part of #13 (Step 0).**
  Replaced the pole epsilon-clamp with a real singular-frame up so the camera can sit at the EXACT
  poles (phi = 0 / π) with no `look_at` degeneracy and no roll-flip. New `OrbitCamera::up_vector()`:
  `Vec3::Y` away from the poles, and within a small smoothstep band (`UP_BLEND_BAND = 0.05` rad of
  each pole) it blends to an **azimuth-derived horizontal up** — the exact limit of "Y projected onto
  the view plane, normalised" as phi → 0/π, i.e. `(−cos θ, 0, −sin θ)` at the top pole and
  `(cos θ, 0, sin θ)` at the bottom. That makes the screen-up CONTINUOUS through the singular frame
  (no 1-frame inversion). Both `view_projection` AND `view_cube_view_projection` now route through
  `up_vector()` (same up → cube and scene stay in sync at the pole). The drag clamp is loosened to
  the exact `[0, π]` and the TOP/BOTTOM snaps target exact `0.0` / `π` (theta keeps the historical
  `−π/2` convention). `POLE_EPSILON` is now unused by the math (retained as a back-compat constant).
  Tests (+6, 210 lib total): up is finite/unit/non-parallel-to-view at phi ∈ {0, 0.0001, π};
  continuity (dense sweep, no jump across the band); exact-pole up matches the convention
  ((0,0,1)/(0,0,-1)); away-from-pole up is exactly `Vec3::Y`; both view matrices all-finite at the
  poles; drag clamp reaches exactly 0/π. All 7 goldens byte-identical (non-pole views unaffected, up
  stays exactly `Vec3::Y` there). Headless `--snap top` shot confirms a clean top-down disc — no
  flip/garbage at the exact pole.

- **Delete flat-geometry config back-compat fields + no-scene migration — Closes #32.**
  Follow-up to #31: the user does NOT want config back-compat. `AppConfig` still carried flat
  `shape` / `size_blocks` / `wall_blocks` geometry mirror fields "kept for back-compat migration",
  used ONLY to synthesize a one-Tool-node scene when a loaded config had no `scene`. The current
  build always writes a `scene`, so they were dead for live configs. Deleted outright (no
  `#[serde(alias)]`/shim); behavior-preserving for normal use; goldens byte-identical.
  - **Deleted (settings.rs):** the flat `shape` / `size_blocks` / `wall_blocks` fields + their
    `default_shape`/`default_size`/`default_wall` helpers and their `Default`/`capture`/`to_panel_state`
    plumbing. The app-level `voxels_per_block` (density) STAYS — the scene reads it at resolve time.
  - **No-scene path:** removed the migration fallback that built a scene from the flat fields. A loaded
    config with no `scene` now loads the DEFAULT seed scene (the same one a brand-new config gets) via
    `seed_scene_from_geometry`; only the persisted density (and `material`) carry over. `to_panel_state`
    seeds `geometry` from `GeometryParams::default()` overridden by the config's `voxels_per_block`.
  - **Back-compat (passive):** no `deny_unknown_fields`, so an existing on-disk config still carrying
    the removed `shape`/`size_blocks`/`wall_blocks` keys (and `debug_clouds`/`mesher`/legacy `show_*`)
    loads fine — serde ignores the now-unknown keys. No migration code.
  - **Tests:** updated the 3 settings tests that referenced the removed `AppConfig` fields
    (`old_config_with_removed_keys_still_loads` [renamed], `old_config_with_debug_clouds_field_still_loads`,
    `old_config_with_mesher_field_still_loads`) to assert old keys are ignored + a scene-less config loads
    the default seed scene; trimmed the round-trip test's removed fields; ADDED
    `config_persists_and_reloads_its_scene` proving a non-trivial scene survives capture→JSON→load with
    identical occupancy. **205 lib tests green; 7 goldens byte-identical; clippy --all-targets clean.**

- **Delete vestigial config back-compat husks; single master source of truth — Closes #31.**
  Resolves the S6 loose end. The user does NOT want config back-compat, so the husks are deleted
  outright (no `#[serde(alias)]`, no migration shim). Behavior-preserving; goldens byte-identical.
  - **Deleted fields:**
    - `PanelState.show_grid_overlay` / `show_block_lattice` / `show_floor_grid` (panel.rs) — the three
      vestigial mirror fields. Since grid-rework S3/S4 the per-grid master checkboxes drive
      `scene.master_voxel_grid` / `master_block_lattice` / `master_floor_grid` directly, and the
      renderers read those scene masters, so these `PanelState` mirrors drove nothing.
    - `AppConfig.show_grid_overlay` / `show_block_lattice` / `show_floor_grid` (settings.rs) — the
      legacy serde mirror fields plus their `Default`/`capture`/`to_panel_state` plumbing and the
      "migrate masters from legacy `show_*`" block.
  - **Single source of truth:** the three grid masters now live ONLY on `scene.master_*`. `capture`
    persists them via the whole-`scene` field (no separate mirror to drift); `to_panel_state` restores
    them from the scene directly (a persisted scene carries its own masters; a scene-less/legacy config
    falls back to the one-Tool-node seed whose `Scene::default()` masters all default ON). This fixes
    #31's stale-mirror asymmetry by construction — there is no mirror left to go stale.
  - **Back-compat (passive):** no `#[serde(deny_unknown_fields)]`, so an existing on-disk config still
    carrying the removed `show_grid_overlay`/`show_block_lattice`/`show_floor_grid` keys loads fine —
    serde ignores the now-unknown keys; the scene's own masters are authoritative. No migration code.
  - **Tests:** updated the 5 settings tests that referenced the removed fields (renamed
    `*_removed_grid_show_keys_still_loads`, `*_gains_origin_point_with_default_masters`,
    `capture_then_to_panel_state_preserves_masters_and_toggles`, and adjusted the modern-scene +
    round-trip tests to assert masters via `scene.master_*`). 204 lib tests green.
  - **shot.rs:** dropped the three dead `PanelState` mirror inits from its headless construction; its
    own `Options.show_*` CLI flags (driven by `--grid`/`--lattice`/`--floor`) still set `scene.master_*`
    directly, unchanged. shot renders of `--demo-scene` and `--demo-scene --points` look unchanged; all
    7 goldens byte-identical; `cargo clippy --all-targets` clean.

- **Grid rework S6: cleanup dead composite-bbox/grid remnants — Closes #29.** Final conservative
  cleanup pass after S1–S5 + the infinite-grid fixes. Behavior-preserving; goldens unmoved; old
  configs still load.
  - **Removed (genuinely dead):**
    - `AppConfig.show_origin_gizmo` (settings.rs) — a pure serde back-compat HUSK. The old
      origin-gizmo Display toggle was replaced by the selection-driven transform gizmo at S2, so the
      field drove nothing (only ever written as `false`, never read into `PanelState`). There is no
      `#[serde(deny_unknown_fields)]` anywhere, so an OLD config still carrying `"show_origin_gizmo"`
      keeps deserialising cleanly (serde ignores the now-unknown key). Locked by a new test
      `old_config_with_removed_show_origin_gizmo_still_loads` (mirrors the existing
      `*_debug_clouds_field_*` / `*_mesher_field_*` back-compat tests).
    - `POINT_PLANE_FADE_BLOCKS` constant + the `fade_voxels` computation in
      `InfiniteGridRenderer::rebuild_from_scene` (renderer.rs). The old fixed world-distance fade was
      removed during the infinite-grid fixes (fading is now per-tier per-pixel LOD in
      `infinite_grid.wgsl`), so this constant only fed the now-unused `params.w`. `params.w` is now
      written as a literal `0.0` reserved slot.
  - **Conservatively KEPT (with comments) — back-compat / layout, NOT dead:**
    - `InfiniteGridUniforms.params` stays `[f32; 4]`: the shader reads `.x/.y/.z` (block spacing,
      minor/major alpha) and `.w` is a std140 16-byte-alignment padding slot. Shrinking the vec4
      would break the uniform layout, so `.w` is documented as reserved, not removed.
    - The legacy `AppConfig.show_grid_overlay` / `show_block_lattice` / `show_floor_grid` serde
      fields stay: they MIGRATE an old config's grid prefs into the scene-wide masters on load, and
      keep a NEW config readable by an older build. Removing them would break that round-trip.
    - All `region_dimensions` / `placed_region_dimensions` uses audited — every remaining use is
      LEGITIMATE (camera auto-frame, onion fog `grid_y`, voxel-grid assembly, `.vox` export,
      `shot`'s auto-frame), none were the old composite-bbox gizmo/lattice/floor sizing. Left intact.
    - `shot.rs`'s own `Options.show_origin_gizmo` is LIVE (driven by `--gizmo`, drives the transform
      gizmo); the field name is kept "for minimal churn" per its existing doc comment.
  - **Loose end found (reported, NOT force-fixed to keep this pass behavior-preserving):** the
    settings round-trip is asymmetric. `AppConfig::capture` reads `show_grid_overlay` from the LIVE
    `scene.master_voxel_grid` but reads `show_block_lattice` / `show_floor_grid` from the now-STALE
    `PanelState.show_block_lattice` / `show_floor_grid` mirror fields (which the UI no longer writes —
    the per-grid checkboxes drive `scene.master_block_lattice` / `master_floor_grid` directly since
    S3). So on SAVE the legacy lattice/floor `show_*` keys can persist a stale value. This is only
    observable to an OLDER build reading a NEW config (this build restores masters from the `scene`
    field regardless), so it is a latent inconsistency, not a live bug. The clean fix is to route all
    three legacy `show_*` through `scene.master_*` in `capture` and delete the three vestigial
    `PanelState` mirror fields — left for a follow-up since it is a round-trip refactor rather than
    dead-code removal.
  - **Also:** fixed the 2 pre-existing `doc_lazy_continuation` clippy warnings in `shot.rs`
    (`--points` doc comment — a line beginning `+ axes)` was misread as a markdown list item; reworded
    to `plus axes`). No `#[allow]` used anywhere.
  - **Verified.** `cargo build --bin shot` clean (the default `voxel_worker.exe` could NOT be relinked
    — the windowed app was RUNNING this session, PID held the exe; reported, not killed; `shot` is the
    verification path per the task). `cargo clippy --lib --bin shot --tests --features gpu`: ZERO
    warnings. `cargo test --lib`: 204 pass (was 203; +1 back-compat test). Goldens: rendered all 7 via
    the freshly-built `shot` and byte-compared against `tests/golden/` — 6/7 byte-identical;
    `debug-clouds` differs by 64 px (0.007%), but a clean-HEAD (`f2f19b3`) `shot` produces the SAME
    bytes as the post-cleanup `shot` (verified by stash + rebuild + compare), so this drift PREDATES
    this pass and is unrelated; it is far under the golden test's 0.5% tolerance, so
    `cargo test golden` still passes all 7. Old-config round-trip green. **#29 is feature-complete.**

- **Rework infinite grid fade/depth: fix near-side cutoff + zoom-out vanish — Part of #29.**
  Two confirmed live-testing bugs in the analytic infinite ground plane
  (`src/shaders/infinite_grid.wgsl` + `InfiniteGridRenderer` in `renderer.rs`), both REPRODUCED
  headless via `shot` and READ before/after.
  - **Bug 1 — near-side cutoff at shallow ortho angles (root cause).** At a shallow ORTHO angle
    (`--proj ortho --phi 1.45–1.50 --dist 120–260`) the FOREGROUND (near, lower-screen) band of the
    ground was missing behind a hard horizontal edge while the far part rendered. Cause: the fragment
    discarded on the ray parameter `t <= 0` measured from the PER-PIXEL near-plane origin. Under ortho
    the rays are parallel and each pixel's near-plane origin can already sit BELOW the plane for
    foreground pixels, so `t` goes negative there and the foreground was wrongly culled.
  - **Bug 2 — entire grid vanishes when zoomed far out (root cause).** At `--proj ortho --phi 1.45
    --dist 700` the WHOLE ground grid disappeared (only axes + object left). Two compounding fades did
    it: (a) the GRAZING-ANGLE fade `smoothstep(0, 0.10, abs(denom))` added by the previous entry — a
    zoomed-out shallow ortho view is near-grazing across the ENTIRE screen, so it faded everything to
    zero; and (b) a fixed WORLD-DISTANCE fade (~80 blocks) that vanished the grid once the view spanned
    more than that.
  - **Fix.** (1) Horizon/sky discard now uses CLIP SPACE: project the plane hit and `discard` when
    `clip.w <= 0` (behind camera). Under perspective this culls the above-horizon sky correctly; under
    ortho `clip.w` is constant positive so NOTHING is wrongly culled and the foreground renders.
    Removed the `t <= 0` foreground cull entirely. (2) REMOVED the grazing-angle fade and the fixed
    world-distance fade — the only fade is now the per-tier per-pixel LOD (cells-per-pixel from
    `fwidth`): a tier dissolves only when its OWN cells go sub-pixel, imposing no hard world-distance
    edge and no hard horizon line. (3) Added a COARSE third tier (lines every 8 blocks, borrowing the
    bold block alpha) so block-scale structure stays visible once the per-block tier goes sub-pixel at
    large ortho zoom-out — this is what keeps the grid present at `--dist 700` instead of vanishing.
    (4) Kept the defensive `frag_depth = clamp(clip.z/clip.w, 0, 1)` so near/far hits projecting just
    outside `[0,1]` are not depth-clipped into a hard seam; object occlusion (`LessEqual`) unaffected.
    Kept the two existing tiers (voxel + block), AA, subtle alpha, ortho-moiré LOD, object occlusion;
    perspective not regressed. `params.w` (the old fade-distance uniform) is now unused but left in the
    layout for stability.
  - **Verified (PNGs READ).** Both repro cases before→after: near cutoff gone (foreground renders at
    shallow ortho `--dist 120/160/300`); zoom-out vanish gone (block/coarse lines clearly visible at
    `--proj ortho --phi 1.45 --dist 700`). Full matrix {perspective, ortho} × {120, 300, 700} ×
    {shallow ~1.45, medium ~1.0}: foreground always renders, block lines visible at 700, no moiré, no
    solid, no hard near/far cutoff, sky clear above the perspective horizon, tube always occludes the
    grid. 203 lib tests + clippy (no new warnings) green; 6 goldens byte-identical, `demo-village-points`
    REGENERATED (legitimate look change: the grid now recedes to the horizon behind the far houses
    instead of vanishing at the old fixed ~80-block distance — READ and confirmed correct).

- **Fix infinite grid shallow-angle hard cutoff (grazing-angle horizon fade) — Part of #29.**
  The analytic infinite ground plane (`src/shaders/infinite_grid.wgsl`) HARD-CUT OFF at a straight
  horizontal line at shallow viewing angles — most dramatic in ORTHOGRAPHIC (`--proj ortho --phi
  1.45–1.52`), also a hard top edge at shallow PERSPECTIVE — instead of receding smoothly to the
  horizon. **Root cause (diagnosed by flooding the shader to magenta with fade/discard bypassed):**
  it is NOT depth-clipping of `frag_depth` as first suspected — the flood proved the ray/plane
  intersection reaches the TRUE mathematical horizon (denom→0) in BOTH projections with no premature
  cut. The real cause was the DISTANCE FADE alone being insufficient: it ramps alpha to zero over
  `fade_distance` (= 80 blocks × density, e.g. 1280 voxels at 16 vx/block). Under orthographic there
  is no foreshortening, so the entire visible ground sits at nearly constant world distance — the
  distance ramp barely moves across the screen and the grid stays near-full-alpha right up to the
  horizon, reading as a HARD horizontal line where the plane meets the horizon (same at shallow
  perspective with a dense block size). **Fix:** add a GRAZING-ANGLE (horizon) fade — multiply alpha
  by `smoothstep(0, 0.10, abs(dot(ray_direction, plane_normal)))`. `abs(denom)` is the sine of the
  ray's elevation above the plane and goes to 0 exactly AT the horizon, so the grid now dissolves
  smoothly INTO the horizon for both projections, independent of distance/density — no hard edge.
  Also defensively CLAMPED the written `frag_depth` into `[0,1]` (`clamp(clip.z/clip.w, 0, 1)`) so a
  far/near plane hit that projects just outside the depth range can never reappear as a stray
  depth-clip seam; the grazing fade has already taken alpha to ~0 there, and object occlusion is
  unaffected (real objects sit at a smaller depth and still win the `LessEqual` test). Kept the
  two-tier voxel/block lines, the LOD anti-moiré fade (ortho moiré fix NOT regressed), the distance
  fade, the subtle alpha, and object occlusion. Verified across {perspective, ortho} × {shallow,
  medium} × {close, far}: grid recedes smoothly to the horizon everywhere, never moiré, never solid,
  never over the sky, always occluded by the tube. All 203 lib tests + 7 goldens pass byte-identical
  (the `demo-village-points` golden's moderate angle leaves the grazing fade a no-op in its visible
  region, so it is unchanged).

- **Fix infinite grid under orthographic projection (near/far ray unprojection + LOD) — Part of #29.**
  The analytic infinite ground plane (`src/shaders/infinite_grid.wgsl`) rendered as a full-screen green
  MOIRÉ cross-hatch — covering the whole background INCLUDING above the horizon ("the sky") — whenever the
  camera was in **ORTHOGRAPHIC** projection. PERSPECTIVE looked fine. The user's camera was set to
  Orthographic (the app's Perspective/Orthographic toggle), which is the trigger.
  - **Reproduced headlessly** with `shot --proj ortho` (the flag already existed) at a zoomed-out ortho
    view (`--dist 80`, grazing `--phi 1.05`): the whole frame was a green moiré field with the grid
    painted over the sky. `--proj ortho` close (`--dist 30`) was uniformly noisy with no clean lines
    anywhere; perspective at the same args resolved clean lines in the foreground.
  - **Root cause #1 — eye-based ray reconstruction.** The fragment shader built the per-pixel view ray as
    `ray_origin = grid.eye.xyz; ray_direction = normalize(far_world - near_world)`. Under perspective that
    is fine (every ray shares the eye). Under ORTHOGRAPHIC it is wrong: ortho rays are PARALLEL (one
    constant direction) and the ray ORIGIN varies per pixel. Using a single shared `eye` as the origin
    gives the wrong plane-intersection `t` and hit point for every pixel → wrong `plane_coord` → garbage
    grid that also fails the in-front test, so it bleeds over the sky. Fix: `ray_origin = near_world` (the
    pixel's NDC unprojected at z=near) with `ray_dir = far_world - near_world` — the robust near/far
    unprojection that is correct for BOTH projections. The hit point is identical to before for
    perspective (so the perspective look is essentially unchanged), and correct under ortho.
  - **Root cause #2 — LOD didn't dissolve sub-pixel tiers under ortho.** Ortho has UNIFORM world-scale
    across the screen (no foreshortening), so when zoomed out EVERY pixel's cells are equally sub-pixel —
    there is no near band where lines resolve. The old LOD (`1 - clamp(cells_per_pixel - 0.5)`) plus the
    duty-cycle "keep the average grey" mix painted a constant grey sheet that the `fract` sampling beat
    into moiré. Fix: drive the whole tier hard to zero once a cell is sub-pixel —
    `lod = smoothstep(2.0, 4.0, pixels_per_cell)` (fully 0 below ~2 px/cell, 1 above ~4 px/cell), so a
    tier dissolves cleanly BEFORE its AA lines can alias. Perspective is unaffected (its near cells are
    many px, lod≈1).
  - **Verification matrix** ({perspective, ortho} × {close `--dist 30`, far `--dist 80`} × {top-down
    `--phi 0.2`, grazing `--phi 1.3`}), each READ: all 8 are clean thin lines (fine voxel tier fading to
    nothing when sub-pixel, bold block tier persisting), clear sky, never solid, never moiré, faded toward
    the horizon, and occluded by objects (`@builtin(frag_depth)` + LessEqual). The ortho zoomed-out repro
    is fixed (before: full-screen moiré over the sky; after: clean block lines, fine tier dissolved, sky
    clear). Residual: the tube/voxel STONE TEXTURE still shimmers at extreme zoom-out — that is the cube
    pass's texture minification (unrelated to the grid), not grid moiré.
  - **Golden:** only `demo-village-points` changed (4.0% — the LOD change fades the perspective fine tier
    a touch earlier, a cleaner look, visually confirmed); regenerated. The other 6 goldens stay byte-
    identical (all << 0.5%). `--shape`/material/fog/cloud paths untouched.

- **Fix infinite grid solid-fill: derivative-normalized LOD coverage (resolution/angle-robust) — Part of #29.**
  The analytic infinite ground plane (`src/shaders/infinite_grid.wgsl`) rendered as a SOLID green
  sheet under the live windowed app's camera/resolution, while the headless golden (a specific
  size + framing) looked fine — so the coverage math was resolution/camera/aspect-dependent.
  - **Reproduced headlessly** via `shot --demo-village --points` at GRAZING angles (`--phi 1.5`) and
    FAR distances (`--dist 200`–`400`): the lower half / mid-screen band saturated to a solid greenish
    sheet (limited only by the distance fade, so it formed a band rather than the whole screen). It
    appears at any resolution/aspect once the viewing angle is shallow or the camera far enough.
  - **Root cause.** The old `grid_coverage` was `1 - clamp(min(abs(fract(coord-0.5)-0.5)/fwidth(coord)))`.
    When `fwidth(scaled)` (cells per pixel) grows past ~1 — exactly what happens at grazing angles, far
    distances, or coarse/high-DPI pixels — the derivative-normalized distance-to-line is ≤ ~0.5
    EVERYWHERE, so `min(...) → 0` and coverage → 1 for every pixel ⇒ solid fill. There was NO LOD
    fadeout: when a tier's cells drop below a pixel it saturated instead of dissolving. The golden's
    size + framing kept the derivative small, hiding the bug. (No coordinate-scale or aspect/inverse-VP
    bug — the ray reconstruction already uses the actual view-proj, correct for any aspect; verified
    across 16:9, 2560×1440, tall 1000×1400, ultra-wide 1920×600.)
  - **Fix (Ben Golus "pristine grid").** `grid_coverage` now (1) keeps the line a target ~1px wide via
    `clamp((half_width - dist_to_line)/fwidth + 0.5)` per axis, (2) **anti-saturates**: as
    `fwidth > 1` it mixes each axis' coverage toward the line's DUTY CYCLE (`2*half_width`) instead of
    1, so the average grey stays constant rather than going solid, (3) combines axes with `a+b-a*b`
    (not `max`), and (4) returns an **LOD visibility** factor `1 - clamp(cells_per_pixel - 0.5, 0, 1)`
    that fades the whole tier OUT as its period drops to ~1px. The fragment shader multiplies each
    tier's alpha by its own LOD factor, so the fine VOXEL tier dissolves first and the bold BLOCK tier
    persists longer, both still fading to the horizon. Depth occlusion (`frag_depth`) unchanged.
  - **Verified (PNGs READ)** across {close, medium, far} × {top-down, grazing} × {1280×720, 2560×1440,
    1000×1400, 1920×600}: always clean thin two-tier lines fading at the horizon, NEVER solid, still
    occluded by the houses — including the exact grazing/far case that reproduced the solid fill.
  - **Goldens.** `demo-village-points.png` regenerated (coverage changed, 27.99% delta — READ, correct
    robust grid); the other 6 stay byte-identical (`demo-village` 0.00000%). 203 lib tests pass; clippy
    has no NEW warnings (2 pre-existing doc-indent warnings in `shot.rs` on clean HEAD).

- **Grid rework: analytic infinite ground plane (replaces finite tiled grid) — Part of #29.**
  Replaced the Points' camera-relative finite tiled-LINE ground plane (`POINT_PLANE_RADIUS_BLOCKS = 48`
  + per-vertex rim fade) — which cut off at a hard finite edge / near-clip and looked bad at shallow
  grazing angles — with a true **analytic infinite grid** via the standard fullscreen ray-plane shader.
  - **Technique (`src/shaders/infinite_grid.wgsl` + `InfiniteGridRenderer` in `renderer.rs`).** For each
    visible Point × enabled plane, draw ONE fullscreen triangle; the FRAGMENT shader reconstructs the
    per-pixel world ray (inverse view-projection + eye), intersects it with the plane (`normal·(p−o)=0`),
    and DISCARDS where the ray misses / hits behind the camera → the grid spans to the horizon with no
    finite border. Grid coverage is computed analytically from screen-space derivatives (`fwidth` of the
    in-plane coords) → crisp anti-aliased lines at any distance/angle, **two tiers**: fine VOXEL lines
    (spacing 1, low alpha `POINT_PLANE_MINOR_ALPHA = 0.10`) and bold BLOCK lines (spacing = density,
    `POINT_PLANE_MAJOR_ALPHA = 0.30`). Alpha **fades linearly to 0** over `POINT_PLANE_FADE_BLOCKS = 80`
    blocks toward the horizon, so it is truly infinite with NO hard edge and distant lines never alias
    into a sheet.
  - **Depth-correct occlusion.** The pass runs INSIDE the 4× MSAA voxel pass, AFTER the voxels, writing
    `@builtin(frag_depth)` = the plane-hit's clip depth and depth-testing **LessEqual** (depth-WRITE off,
    alpha-blended). Opaque objects (already in the depth buffer) thus OCCLUDE the grid with no z-fighting.
    This avoids the resolved-target depth-sampling dance the onion fog needs and gets MSAA edge AA for
    free. One dynamic-offset uniform buffer holds all planes; `draw` binds each plane's slice (≤ 32).
  - **Planes vs axes.** Only the PLANES became the analytic grid; the **axes stay as lines** (unchanged).
    `points_line_batch` is now AXES-only; the new pure `enabled_grid_planes(scene, density)` selects every
    visible Point's enabled XZ/XY/YZ plane (origin + orthonormal basis, normals +Y/+Z/+X) in the recentred
    frame. Multiple Points each contribute their own plane(s) at their own world height (verified headless).
  - **Verification (headless PNGs READ).** Shallow grazing angle (`--phi 1.50`): the ground extends
    smoothly to the horizon, **cutoff gone**, fading with distance, houses occluding it. A second offset
    Point (`--point-at 0 3 0`, new shot flag) shows a second grid plane at that height. `demo-village
    --points` shows the subtle two-tier infinite grid occluded by the four houses with origin axes — the
    new look.
  - **Tests + goldens.** Reworked the renderer CPU tests around `enabled_grid_planes` (plane gating,
    orientation/basis, offset-point world position, block-spacing = density) — 203 lib tests pass. Golden
    suite: the 6 non-`--points` goldens stay **byte-identical**; only `demo-village-points.png` was
    deliberately regenerated (READ + confirmed) for the new infinite-grid look.

- **Grid rework: floor meets lattice (depth bias) + new-point plane defaults + separable XYZ axes — Part of #29.**
  Three smoke-test fixes for the grid rework.
  - **Fix 1 — floor meets the block lattice at the base plane.** The floor grid was geometrically
    dropped `FLOOR_PLANE_DROP_VOXELS = 0.25` voxel below the base to dodge z-fighting the model's
    bottom face, leaving a visible gap below the lattice. Removed the drop: `floor_vertices_into`
    now draws at the EXACT base plane `y = min[1]`, so its bold block lines coincide with the
    lattice's bottom edges. Z-fighting is instead resolved with a **depth bias** toward the camera.
    **Key finding:** wgpu rejects a hardware `DepthBiasState` on `LineList` topology ("Depth bias is
    not compatible with non-triangle topology"), so the bias is applied in the line **shader**:
    `LineUniforms` gained a `depth_bias` (NDC z offset, scaled by `w`); `line.wgsl` nudges `clip.z`.
    `SceneGridRenderer` carries a second floor uniform buffer/bind-group uploading the same matrix
    with `FLOOR_DEPTH_BIAS_NDC = -5e-4` (imperceptible spatially, far below the old 0.25 drop) while
    the lattice uniform keeps bias 0. Headless sphere+lattice+floor PNG confirms the floor's block
    grid meets the lattice's bottom rectangle with no vertical gap and no z-fight shimmer.
  - **Fix 2 — new Points default to all planes OFF.** `Scene::add_point` now overrides the incoming
    Point's plane/axis flags to **all planes off, all axes on**, so every "+ Add Point" path yields a
    clean default. Only the Origin (built by `ensure_origin_point`, not `add_point`) keeps the XZ
    ground plane on.
  - **Fix 3 — separately-toggleable X / Y / Z axes.** Split `Point.axes: bool` into
    `axis_x` / `axis_y` / `axis_z` (each `#[serde(default = "default_true_bool")]`, so an older
    serialized Point missing them defaults each to true). `point_axes_into` takes a per-axis
    `[bool; 3]` and emits a segment only for each enabled axis (X red, Y green, Z blue);
    `points_line_batch` passes the three flags. `panel.rs` replaces the single "Axes" checkbox with a
    compact `Axes  [X][Y][Z]` row. **Back-compat note:** `axes` was brand-new from S1 (this branch) and
    unlikely persisted, so defaulting the three new fields to true (no `alias`/migration) is the chosen
    handling; a persisted `axes: false` is not honored (acceptable per spec).
  - **Tests + gate.** +2 CPU tests (`add_point_defaults_planes_off_axes_on`,
    `points_axes_toggle_per_axis` — 3 axes ⇒ 3 segments, Y off ⇒ no green line, only-Y ⇒ green only);
    updated the floor-base-height assertion (`y == min[1]`) and the settings round-trip to exercise
    per-axis flags. `cargo test` 203 passed (was 201); all 7 goldens green (the points golden uses the
    Origin — ground + all axes on — so it is unchanged); clippy clean.

- **Fix per-object floor grid: voxel-edge lines + align to block lattice — Part of #29.** The
  interactive smoke-test surfaced two floor-grid bugs: (1) it drew lines only at BLOCK boundaries,
  not at voxel edges; (2) it read as "poorly aligned" with the per-object block lattice.
  - **Root cause (alignment).** The floor box and the lattice box are the SAME box
    (`node_block_lattice_box_recentred`) at the SAME `step`, so their block lines already coincided —
    but with only block-spacing lines drawn, there was nothing fine to read the alignment against, and
    a future voxel-line scheme that snapped to the GLOBAL voxel grid (multiples of 1 from 0) would NOT
    pass through the block-aligned box corners. The fix pins both tiers to walk from the block-aligned
    box `min` with a 1-voxel stride, so every `step`-th voxel line lands on `min + k·step` — the EXACT
    coordinates of the lattice's vertical lines (`block_boundaries(min, max, step)`). Floor and lattice
    now share one global-lattice frame and their lines coincide at the base plane.
  - **Two-tier fine floor grid.** `floor_vertices_into` now emits FINE voxel lines (one per voxel
    boundary, step 1, subtle `FLOOR_VOXEL_ALPHA = 0.16`) PLUS BOLD block lines (step = density, bright
    `FLOOR_ALPHA = 0.55`, drawn on top), mirroring the block lattice / Point ground plane minor+major
    scheme. New helper `voxel_boundaries(lo, hi, step) -> Vec<(coord, is_block)>` walks voxel-by-voxel
    and tags each `step`-th line as a block edge. The small `0.25`-voxel base-plane drop is kept (no
    z-fight with the model's bottom face); the footprint is still the node's enclosing-block XZ extent.
  - **Verify (headless).** A flat-disc sphere with `--lattice --floor` at density 4 shows the floor's
    fine warm voxel grid filling each block cell, with the bold block lines coincident with the teal
    block-lattice verticals at the base — aligned. At density 16 the voxel lines are visibly denser.
  - **Tests.** `voxel_boundaries_tag_block_lines_at_lattice_positions` (the floor's block-tagged lines
    equal `block_boundaries`; voxel lines denser at coarse density) and
    `floor_grid_is_two_tier_and_aligns_with_lattice` (floor X/Z lines are a superset of the lattice's
    vertical lines; two alpha tiers; denser than the lattice). Density-parametrized `{1, 15, 16}`. 201
    lib tests; 7 goldens byte-identical (floor default-off in `shot`, viewport unchanged).

- **Grid rework S5: Points (camera-relative ground plane + axes, depth-tested) + Points UI — Part of #29.**
  The world reference grid is live. A new `PointsRenderer` (`renderer.rs`) batches every VISIBLE
  Point's reference geometry into ONE depth-tested, alpha-blended line buffer — the SAME pass family as
  `SceneGridRenderer`, drawn in the MSAA pass before the (depth-OFF) transform gizmo — so opaque voxels
  OCCLUDE the planes/axes while a node's on-face voxel grid (a fragment overlay) stays on top.
  - **Camera-relative tiled ground plane.** Each frame `points_line_batch(scene, density, camera_eye)`
    rebuilds, per visible Point, its enabled planes (XZ/XY/YZ) as a tiled grid CENTRED on the camera's
    projection onto that plane, SNAPPED to the global block lattice (`origin + k·step`, step = density),
    spanning `POINT_PLANE_RADIUS_BLOCKS` (48) each way. Per-VERTEX alpha fades over the last
    `POINT_PLANE_FADE_BLOCKS` (16) toward the rim (`LineVertex.color.a`), so the plane dissolves into the
    background — no hard finite edge as you orbit far. Two-tier BOLD block lines: every
    `POINT_PLANE_MAJOR_EVERY_BLOCKS` (8) line is brighter/higher-alpha (major) over the dimmer per-block
    minor lines. Subtle base alpha (minor 0.10 / major 0.22) so it doesn't fight per-object voxel grids.
    Axes: three colored lines (reusing the gizmo axis colours) through the Point origin, `±6` blocks,
    depth-tested. Each Point sits at `position_blocks·density − recentre` (the resolved-voxel frame).
  - **Pass ordering / depth.** `FrameOverlays.points` is drawn in the MSAA pass right after the
    scene-grid lattice, depth-tested (`build_line_pipeline(..., depth_tested=true)`); the transform gizmo
    stays last + depth-OFF. Confirmed in `lib.rs::render_frame`.
  - **Points UI (`panel.rs`).** A new **Points** section after the Scene list: each Point as a row
    (visibility checkbox bound to `!hidden` + selectable name → `scene.active_point`), **+ Add Point**
    (at the camera target, mirrored into `PanelState.point_add_position_blocks` from the camera each
    frame in `main.rs`), and — for the selected Point — XZ/XY/YZ plane checkboxes, an axes checkbox, a
    whole-block position editor (HIDDEN for the Origin), and a Delete button (HIDDEN for the Origin,
    undeletable). Deferred-mutation pattern (select/delete applied after the read walk), mirroring the
    node list. The section renders NOTHING when `scene.points` is empty.
  - **shot `--points` + the new golden.** `shot` SUPPRESSES Points by default (the scenes it builds do
    NOT synthesize the Origin, so `scene.points` is empty → no render AND a zero-height Points panel
    section → the 6 existing goldens are BYTE-IDENTICAL). `--points` calls `ensure_origin_point` (ground
    + axes default on) and wires the `PointsRenderer` into the overlays. New golden `demo-village-points`
    (`--demo-village --points`) through the cuboid path: READ confirms the subtle teal ground plane
    tiling under the four houses, OCCLUDED where each house base sits in front of it, fading toward the
    edges with no finite border, the brighter major block lines over the dimmer minor ones, and the
    origin X/Y/Z axes reading through the first house. A far-orbit render confirms the plane stays under
    the camera.
  - **Tests + gate.** 5 new CPU tests in `renderer.rs` (visible Point ⇒ non-empty batch / hidden ⇒ none;
    plane+axis toggles gate independently; ground tiling snapped to block multiples + centred near the
    camera; a second offset Point's frame at its world position; density-parametrized {1,15,16} block
    spacing). `cargo test` 199 lib green; `cargo clippy --all-targets` clean; `cargo build --bins` ok;
    `cargo test --features gpu --test golden` = the 6 existing byte-identical + the new points case.

- **Remove the legacy instanced mesher; cuboid is the sole render path — Part of #20.**
  The cuboid box-decomposition mesher had reached full parity (shape, per-voxel +
  loaded-block per-face textures, multi-material atlas, layer band, debug-faces,
  per-object grids) and was already the default, so the legacy one-cube-per-voxel
  instanced renderer is now DELETED outright rather than kept as a fallback.
  - **Removed.** `renderer::VoxelRenderer` (struct + entire impl: the per-chunk
    `InstancedChunkBuffers` cache, `bucket_instances_into_chunks`, `instances_for_chunk`,
    `rebuild_chunk`/`evict_chunk`/`rebuild_all_from_chunks`/`incremental_rebuild_from_chunks`/
    `rebuild_instances`, the frustum-cull/draw, `instance_count`/`visible_chunk_count`/
    `last_rebuilt_chunk_count`), plus `VoxelInstance`, `Chunk`, `CubeVertex`,
    `unit_cube_geometry`, the `VoxelUniforms` struct, and the now-dead `material_index` /
    `neutral_material_base_colors` helpers. Deleted the instanced shader
    `src/shaders/voxel.wgsl` (the cuboid shaders are self-contained — they carry their own
    on-face-grid + `GRID_OVERLAY_BIT` copies; WGSL has no includes). Dropped the
    `MesherChoice` enum, the `PanelState.mesher` field, the "Cuboid mesher (default; off =
    legacy instanced)" panel checkbox, the `shot` `--mesher` + `--instanced-via-chunks`
    flags (and `parse_mesher`), and the 6 instanced-only unit tests (chunk bucketing /
    per-chunk-instances / AABB / `voxel_cube_is_ccw_outward`).
  - **Kept (shared infrastructure).** The whole resolve / `ChunkResolveCache` / `Voxel` /
    material system / `MaterialSource` / `MaterialSource::Loaded` / `texture_atlas` (the
    cuboid atlas still calls `procedural_material_pixels`) / per-object grids /
    `SceneGridRenderer` / transform gizmo / onion fog / disk store / chunk storage. The
    pure `incremental_rebuild_plan` + `IncrementalRebuildPlan` stay as the resolve cache's
    dirty-chunk planner (the chunk-cache tests drive them; they reference no instanced
    type). The shared face-material layout + sampler moved onto `CuboidMeshRenderer`
    (`material_bind_group_layout()` / `material_sampler()`), so `apply_block_variant` /
    `shot` build `LoadedMaterial` against the cuboid renderer instead of the deleted one.
  - **Wiring.** `render_frame` lost its `voxel_renderer` param and the branch; `cuboid_mesh`
    is now a mandatory `&CuboidMeshRenderer`. `main.rs`/`shot.rs` always build the cuboid
    renderer (from the resolve cache's per-chunk accessor; whole-grid wrapper at startup);
    `main::rebuild_geometry` keeps the resolve cache's `invalidate_aabb`/`clear` side
    effects but no longer needs the instanced dirty-set / full-rebuild flag.
  - **Persistence.** `MesherChoice` was never an `AppConfig` field (session-only in
    `PanelState`, reset to default on load), so old configs are unaffected; added a
    round-trip test that an old config carrying a stray top-level `mesher` field still
    loads (serde ignores unknown fields).
  - **Result.** renderer.rs 4052 → 2901 lines; ~1.3k net lines removed plus the deleted
    shader. `cargo build --bins`, `cargo clippy --all-targets` (no warnings), and
    `cargo test --features gpu` all green: **194 lib tests** (was 199, −6 instanced +1
    config test), the 6-case golden (byte-identical — cuboid 3D path unchanged), and the
    palette-click test. Headless `--demo-scene` / `--demo-village` / textured cylinder /
    `--synthetic-block` (loaded per-face D2Array) / `--demo-groups --lattice --floor` all
    render correctly via the sole cuboid path; READ the demo-scene + synthetic-block +
    grid PNGs to confirm. The old `--mesher` flag now no-ops with an "ignoring unknown
    argument" warning.

- **Grid rework: fix per-object floor grid + default masters ON — Part of #29.** Two
  smoke-test fixes from interactive feedback.
  - **Floor grid root cause + fix.** The per-object floor grid (S3) *was* rendering, but read as
    "nothing": it drew at `#6b5f4a` @ **0.16** alpha (near-black vs the dark viewport) on a plane
    EXACTLY coincident with the model's depth-tested bottom face, so the solid model occluded all of
    it bar the thin enclosing-block margin. Fix in `renderer.rs`: brighter warm-sand `#b8a47a` @
    **0.55** alpha (lattice-comparable), and drop the floor plane `0.25` voxel below the base
    (`FLOOR_PLANE_DROP_VOXELS`) so it no longer z-fights the bottom voxel face and reads as the ground
    under the object. Footprint unchanged (the node's enclosing-block XZ extent, snapped to the global
    block lattice). Headless `--floor` PNG: floor grid now clearly visible; `--lattice --floor --grid`
    on one node shows the teal enclosing lattice + sand base grid + on-face voxel grid together.
  - **Masters default ON.** All three scene-wide masters (`master_block_lattice`, `master_voxel_grid`,
    `master_floor_grid`) now default `true` (serde `default_master_grid` + manual `Default for Scene`),
    so flipping a per-object grid toggle shows immediately; per-object flags stay default OFF so the
    default view is still clean. `settings.rs`: legacy `show_*` mirrors (which seed the masters for a
    pre-Points config) now default `true` too (struct default + serde `default_true`), so a brand-new
    user with no config gets all masters on, while a genuine legacy config still seeds each master from
    its own persisted `show_*` value.
  - **Tests.** Updated `scene_default_master_grids` (all three ON) and `old_scene_json_loads_with_grid_defaults`
    (all masters default ON); updated the floor-geometry assertion to the dropped base plane. 199 lib
    tests green, 6 goldens byte-identical (no per-object grid is enabled in any golden case, so the 3D
    viewport is unchanged — no regen needed). Clippy clean.

- **Grid rework S4: per-object on-face voxel grid via material-id flag bit — Part of #29.**
  The on-face voxel grid (the bold-block-line fragment overlay in the mesh shaders) is now PER OBJECT,
  gated by the scene master `master_voxel_grid` ANDed with each node's own `grids.voxel_grid_on_faces`
  (default OFF for new objects). Approach: pack a **"show on-face grid" flag bit** into the per-voxel
  `material_id` so it rides through BOTH render paths (per-chunk instanced + cuboid) with no new vertex
  attribute.
  - **Bit constant.** `voxel::GRID_OVERLAY_BIT = 1 << 15` (the high bit of the `u16` material id;
    real handles are only Stone/Wood/Plain ⇒ 0/1/2, so the bit is free), plus a CPU mirror
    `voxel::material_id_color_index(id) = id & !GRID_OVERLAY_BIT`. **Mirrored verbatim** as
    `const GRID_OVERLAY_BIT: u32 = 32768u;` in all three mesh shaders (`voxel.wgsl`, `cuboid.wgsl`,
    `cuboid_loaded.wgsl`) — keep the four in sync.
  - **Resolver (bit baked per-voxel).** `for_each_leaf`/`walk_nodes` now pass each leaf's
    `voxel_grid_on_faces`; both `stamp_producer` and `stamp_producer_into_chunk` gained a `grid_overlay`
    flag that ORs `GRID_OVERLAY_BIT` onto every stamped voxel's `material_id` (orthogonal to the
    material override, so a Tool keeps its real id + bit, a Part keeps its own per-voxel material + bit).
    The bit therefore survives chunk bucketing and the cuboid densify, travelling with each voxel.
  - **Master AND (shader uniform).** The Display master maps to the existing `grid_overlay_enabled`
    uniform; the shaders gate `on_face_grid_enabled = grid_overlay_enabled > 0.5 && (material_id & BIT)`.
    Master OFF ⇒ no node draws; master ON ⇒ only flagged (opted-in) voxels — toggling the master is a
    pure uniform write (no re-resolve, no re-upload). A NODE's flag flip DOES re-resolve (the bit is
    baked at resolve), wired via `scene_changed`.
  - **Cuboid path.** The flag bit is folded into the `material_id` BEFORE `decompose_into_boxes`, so two
    same-material voxels differing only in the bit are DIFFERENT materials to the greedy mesher and
    never merge (a uniformly-flagged run still merges to one box). Each box carries the bit onto its
    faces; both cuboid shaders gate the overlay on it.
  - **Mask-before-colour (correctness).** All three shaders mask the bit OFF before any colour/atlas
    lookup (`material_color_index` / `material_base_colors_lookup`'s masked `min(…,2u)`), so the flag
    can never push a flagged voxel's colour index past 2 and corrupt the material colour. Verified: the
    Stone/Wood/Plain colours render unchanged on flagged voxels.
  - **UI.** The inspector "Grids (this object)" section gains a **"Voxel grid on faces"** checkbox
    (signals `scene_changed` — re-resolve). The Display checkbox is relabelled **"Voxel grid on faces
    (master)"** and repointed at `scene.master_voxel_grid`; persistence (`AppConfig.show_grid_overlay`)
    now mirrors the scene master. `main.rs`/`shot.rs` feed `scene.master_voxel_grid` to both meshers'
    uniforms.
  - **`shot`.** `--grid` now sets the scene master AND enables `voxel_grid_on_faces` on ONE node (the
    `--select-node N` node, else node 0). Headless `--demo-scene --grid` PNGs in BOTH meshers
    (`--mesher cuboid` default + `--mesher instanced`) confirm: the enabled node (Sphere, node 0) shows
    bold block-edge + voxel grid lines on its faces while the sibling Box/Torus show none — per-object
    gating, identical between meshers.
  - **Tests.** +4 CPU tests (199 total): the bit is set iff a node opts in and stripped/clear otherwise
    (density {1,15,16}); the masked colour id round-trips to ≤2; a 2-node scene flags exactly the enabled
    node's voxels; `decompose_into_boxes` does NOT merge across differing grid bits. Goldens green — only
    the two short-inspector cases (`demo-village`, `debug-clouds`) moved, and ONLY in the panel region
    (the new inspector row + master relabel); a crop-compare proved the 3D viewport is byte-identical
    (0.0000%), so the two reference PNGs were regenerated.

- **Grid rework S3: per-object block lattice + floor (global-lattice snapped) — Part of #29.**
  The whole-region lattice/floor is gone; each grid is now PER OBJECT, gated by a scene master
  ANDed with the node's own toggle (default OFF for new objects → the windowed default now shows
  NO per-object lattice until you enable one).
  - **Geometry source.** New `Scene::node_block_lattice_box_recentred(path, density) -> Option<([f32;3]
    min, [f32;3] max)>`: the node's block-aligned voxel AABB **expanded to enclosing whole blocks**
    (reuses `node_subtree_extent_blocks` — the same `[off − floor(size/2), … + size)` per-block split),
    scaled by density and shifted `− recentre_voxels_for_resolve`, in the recentred render frame. Group/
    Instance → union of its leaves; a size-less node → `None`. The expand-to-block on the shifted box is
    what makes "a 1-voxel translate adds/removes a whole block" fall out (sub-block offsets aren't
    representable yet — `offset_blocks` is whole-block; the property is pinned on the geometry directly).
  - **Renderer rename + box refactor.** `GridLatticeRenderer` → **`SceneGridRenderer`** (now owns the
    per-frame lattice + floor line BATCHES, not one region-sized set). `lattice_vertices`/`floor_vertices`
    → `lattice_vertices_into`/`floor_vertices_into(&mut Vec, min, max, step)` taking an arbitrary
    `(min,max)` box + block `step` (= density) instead of a centred `grid_dimensions`. Floor is now the
    horizontal grid on the box's BASE plane (`y = min[1]`) at block spacing, snapped to the same global
    block lines as the lattice. Block edges are the lattice lines (the existing block-tier scheme).
  - **Per-frame batch + gating.** Each frame `SceneGridRenderer::rebuild_from_scene(scene, density)` walks
    `tree_rows()`; the gate is extracted into a pure, CPU-testable `scene_grid_boxes(scene, density) ->
    (lattice_boxes, floor_boxes)` collecting one box per node where `master_block_lattice &&
    node.grids.block_lattice` (and same for floor). Depth-tested + alpha-blended as before, so opaque
    voxels occlude the lines. Empty batches → `draw` is a no-op. Removed `FrameOverlays.show_lattice/
    show_floor` (gating is now at batch-build time).
  - **UI.** Inspector gains a **"Grids (this object)"** section (`block_lattice` + `floor_grid` bound to
    `node.grids.*`) for every node kind — toggling needs only a per-frame batch rebuild, NOT a scene
    re-resolve, so no `scene_changed` is signalled. The Display master checkboxes are relabelled
    **"Block lattice (master)"** / **"Floor grid (master)"** and repointed at `scene.master_*` (the old
    `PanelState.show_block_lattice/show_floor_grid` are now serde-only config back-compat, drive nothing).
    Voxel-grid-on-faces master/toggle left for S4.
  - **`shot`.** `--lattice`/`--floor` now set the matching scene master AND enable that grid on ONE node
    (the `--select-node N` node, else node 0), so a headless capture proves per-object gating. Headless
    2-node demo-scene PNGs confirm: `--lattice --select-node 1` → lattice hugs the BOX's enclosing blocks,
    sphere/clouds bare; no `--lattice` → no lattice anywhere; node-0 lattice+floor → grid on the sphere
    only (depth-occluded by the overlapping box).
  - Tests (+6): density-parametrized {1,15,16} — lattice box extent = B·d and spans whole blocks;
    follow-on-whole-block-translate shifts the box by exactly d (anchored against a large fixed node so
    the recentre doesn't drag); size-less node → `None`; `block_boundaries` plane count tracks enclosing
    blocks (+1 block ⇒ +1 plane, the add/remove-a-whole-block geometry); per-box line sets non-empty +
    floor on the base plane; `scene_grid_boxes` gated by master AND per-object (no box when either off,
    exactly one when one node enabled). **195 lib tests pass; 6 goldens green (3D viewport pixel-identical;
    the goldens' panel region was regenerated for the new inspector section); clippy clean.**

- **Grid rework S2: transform gizmo follows the selected node — Part of #29.**
  Repurposed the origin gizmo into a per-selection manipulator (basis for future TRS handles):
  - **Rename** `GizmoRenderer` → `TransformGizmoRenderer` (`gizmo_renderer` → `transform_gizmo_renderer`
    in `main.rs`; same axis-triad geometry, **depth-test still OFF** so it shows through solids).
  - **Follows the selection.** New `Scene::active_gizmo_placement(density) -> Option<([f32;3] pivot,
    [f32;3] extent)>`: the gizmo is anchored at the active node's **block-AABB centre in the recentred
    render frame** — `block_aabb_centre·d − recentre_voxels` — and **sized from that node's OWN extent**
    (not the whole region). For a Group/Instance selection the AABB is the union of all leaves under it.
    The pivot is baked into the uploaded matrix as `view_projection · translate(pivot)` (no shader/
    `LineUniforms` change). Both the renderer's `update_uniforms` and the per-frame `rebuild` (extent)
    run in the render path so a **selection change** (which does not trigger a geometry rebuild) moves
    and resizes the gizmo. **Chose the AABB centre over the corner-origin** so the gizmo sits ON the
    object even for a single-axis-offset child.
  - **Visibility is selection-driven.** `scene.active == Some` → draw at that node; `None` (or a
    selection with no extent) → not drawn. **Removed the "Origin gizmo" Display checkbox** (`panel.rs`)
    and the `PanelState.show_origin_gizmo` field; `AppConfig.show_origin_gizmo` is kept serde-only for
    config back-compat (round-trips, drives nothing).
  - **`shot --gizmo`** now means "show the transform gizmo on the active/selected node" (no-op-safe
    with no selection / no extent). New **`--select-node N`** picks the active top-level node for
    headless captures (out-of-range clears the selection). Goldens never pass `--gizmo`, so they are
    unaffected.
  - **Single-node recentre invariance (expected):** a lone selected node recentres onto the origin,
    so its gizmo pivot is `[0,0,0]` (even sizes) or within half a voxel (odd sizes, recentre truncation)
    — the gizmo only visibly *moves* with a multi-node selection. Headless PNGs confirm: select node 0
    (sphere) → gizmo on the centre sphere; select node 1 (box) → gizmo jumps onto the box; no selection
    → no gizmo.
  - Tests (+3 lib): pivot == `centre·d − recentre` tracking each selected node + node-own extent across
    densities {1,15,16}; `None` when nothing selected; lone even-node pivot exactly origin; lone odd-node
    pivot within half a voxel. **189 lib tests pass; 6 goldens green; clippy clean.**

- **Grid rework S1: per-node grid settings + Point elements + persistence — Part of #29.**
  DATA MODEL + PERSISTENCE only (no render change → goldens stay green). Added, all serde
  back-compatible (old scenes/configs load unchanged):
  - **`NodeGrids`** `{ voxel_grid_on_faces, block_lattice, floor_grid }` (all default **false**),
    `#[serde(default)] grids: NodeGrids` on `Node`; `Node::new` keeps them off (new objects → grids OFF).
  - **`Point`** `{ name, position_blocks: [i64;3] (world-block lattice), offset_voxels: [i32;3],
    plane_xz (default **true** = ground), plane_xy/plane_yz (false), axes (default **true**),
    hidden (false), is_origin (false) }` — true-defaults via `#[serde(default = "…")]` helpers.
  - On **`Scene`**: `points: Vec<Point>`, masters `master_block_lattice` (default **true**) /
    `master_voxel_grid` / `master_floor_grid` (false), `active_point: Option<usize>`. Manual
    `Default` impl so empty-scene masters match the serde defaults.
  - **Methods:** `ensure_origin_point` (idempotent; inserts one undeletable `is_origin` Point at
    index 0), `add_point`, `remove_point` (NO-OP on the Origin), `toggle_point_hidden` (Origin hideable).
  - **Persistence:** the whole `Scene` rides `AppConfig.scene` with `#[serde(default)]`. On load
    (`settings.rs::to_panel_state` and `panel.rs::seed_scene_from_geometry`) every scene calls
    `ensure_origin_point`; masters migrate from the legacy `show_block_lattice/show_grid_overlay/
    show_floor_grid` for scenes that predate Points. **No renderer rewired** — the existing
    `PanelState.show_*` toggles still drive the live renderers; master→renderer wiring is S3/S4.
  - Tests (+9 lib): grids-off-by-default, scene master defaults, `ensure_origin_point`
    idempotent/no-duplicate, `remove_point` spares+hides Origin, grids+points round-trip,
    old-scene-json defaults, old-config gains Origin + migrates masters, modern scene keeps masters.
    **186 lib tests pass; 6 goldens green; clippy clean.**

- **Density-parametrized shape-alignment + node-AABB follow tests — Part of #29.** Pure-CPU
  TEST augmentation (no behaviour change → goldens stay green). Generalized the #30 acceptance
  coverage across a representative density set and added the #29 grid/gizmo geometry-source tests:
  - **Generation, parametrized over density** (`scene.rs` tests): `one_block_box_aligns_across_densities`
    (1×1×1 box → exactly `d³` voxels in ONE block-aligned cell `[k·d,(k+1)·d)`, over **d ∈ {1, 2,
    15, 16, 32}** — incl. the requested d=1 → 1 voxel and d=15 → 3375); `odd_size_shape_is_block_lattice_aligned`
    (5×5×2) and `even_size_shape_is_block_lattice_aligned` (2×4×6) now loop **d ∈ {1, 15, 16}** via a
    shared `assert_box_block_aligned` helper (every block boundary on a multiple of `d`, no half-block).
  - **#29 foundation "follow" tests** over the node's block-aligned voxel AABB (from
    `build_leaf_spatial_index`) + recentre (`recentre_voxels_for_resolve`), d ∈ {1, 15, 16}:
    `node_block_aabb_scales_and_aligns_across_densities` (B-block extent → B·d voxels, corners on
    block multiples); `node_aabb_follows_translation_at_each_density` (+1 block shifts the AABB by
    exactly `d` voxels on that axis, 0 elsewhere, stays block-aligned — offsets are whole blocks, so
    whole-block translation is the unit; sub-block isn't representable); `node_pivot_origin_tracks_offset_across_densities`
    (pivot = `offset·d − recentre`: for a LONE node the recentred pivot is invariant under
    self-translation because the auto-recentre follows it — which is WHY #29 positions grids in the
    GLOBAL lattice frame; the ABSOLUTE origin `offset·d` does follow +1 block by `d`).
  - No helper exposure needed — `build_leaf_spatial_index`/`recentre_voxels_for_resolve` are already
    `pub` and `VoxelAabb` fields are public; zero logic change. **177 lib tests** (was 174; +6 new,
    −3 old merged), clippy clean, 6 goldens green.
  - **NOTE:** the RENDERER-level grid/lattice/floor/voxel-grid + transform-gizmo follow tests (drawing
    the actual lines/gizmo) will be added with **#29 sub-steps S3/S5**, parametrized over the SAME
    density set {1, 15, 16}, once those renderers exist.

- **Align shape generation to the global block lattice (Closes #30).** Generated shapes were
  off-centre for ODD block sizes: the producer (`SdfShape::resolve`, voxel.rs) centres its
  `grid = size·d` voxels on the origin (`idx + 0.5 − grid/2`), so an odd block count's span
  `[−grid/2, grid/2)` is an odd multiple of `d/2` and straddles a block boundary by half a block
  (a 1-block box at offset 0 landed on `[−d/2, d/2)` instead of one whole-block cell). The render
  path *looked* aligned because its recentre used a different, block-floored frame
  (`placed_extent_blocks`, `floor(size/2)` per block), but the **producer-true / absolute** frame
  that the per-object grids (#29) and the chunk resolve read disagreed by half a block for odd
  sizes (documented at scene.rs `placed_extent_voxels`). **Fix:** snap each sized leaf onto the
  block lattice with `leaf_lattice_shift_voxels = grid/2 − floor(size/2)·d` (0 for even, `d/2` for
  odd — an integer voxel count, so every centre keeps its `n + 0.5` fraction and the chunk-storage
  index recovery is untouched). Applied at all four local→world sites: `resolve_region`,
  `resolve_chunk_rebased` (translation + skip-AABB), `placed_extent_voxels`,
  `build_leaf_spatial_index`. After the shift the producer-true and block-AABB frames coincide, so
  recentre / chunk ownership / spatial index / grids all agree. **Convention:** a `B`-block axis
  occupies the whole-block range `[off − floor(B/2), off − floor(B/2) + B)` blocks → voxels
  `[(off−floor(B/2))·d, …)`, min corner always on a block multiple. Voxel ranges at offset 0,
  d16: **1-block → [0, 16)** (one block cell); **2-block → [−16, 16)**; **5-block → [−32, 48)**.
  Acceptance tests added (`one_block_box_generates_one_block_aligned_cell`,
  `odd_size_shape_is_block_lattice_aligned`, `even_size_shape_is_block_lattice_aligned`); 174 lib
  tests pass; goldens rebaselined (cylinder & sphere-debug-faces are correct shapes shifted half a
  block — the intended alignment change; chunked-resolve parity tests still green).

- **Speed up cuboid rebuild (perf) — Part of #20.** Dragging any slider with the DEFAULT cuboid
  mesher was very laggy: every edit fully re-meshes all chunks. MEASURED the per-rebuild phases
  (release, repeated re-resolve worst case) and found the apron mesher's `HashMap<[i64;3],u16>`
  was the dominant cost — both building it (`global_occupancy`) and querying it once per apron
  cell (`apron_fill`).
  - **Before (release ms/rebuild):**
    - sphere 5×1×5 d16 (default disc, 53.7k voxels, 8 chunks): resolve 8.8 + global_occupancy 3.5
      + per-chunk-mesh 7.6 = **19.8 ms** (apron-fill ≈ all of the per-chunk-mesh).
    - box 5×5×5 d16 (512k voxels, 8 chunks): resolve 36 + global_occupancy 58 + per-chunk-mesh 109
      = **203 ms** (apron-fill ≈ 44 ms/build, densify+decompose+emit ≈ 1.5 ms — the HashMap was ~100%).
  - **Fix:** replaced the global occupancy `HashMap` with a DENSE row-major `VoxelRegion` indexed
    directly by the absolute global voxel index, built O(voxels) with no hashing. Each chunk's
    apron is now filled by copying the contiguous global sub-window per X row with
    `copy_from_slice` (band-clipped by Y row) instead of a per-cell `HashMap::get`. The occupancy
    queried — hence the meshed output — is byte-identical.
  - **After (release ms/rebuild):**
    - sphere: global_occupancy 0.4 + per-chunk-mesh 1.5; cuboid-specific work 11.1 → **1.9 ms**;
      TOTAL 19.8 → **10.5 ms** (resolve, shared with the instanced path, now dominates).
    - box: global_occupancy 58 → 4, apron-fill 44 → 0.24, per-chunk-mesh 109 → 11; cuboid-specific
      work 167 → **15 ms**; TOTAL 203 → **56 ms** (3.6× overall, ~11× on the cuboid-specific work).
  - **Output unchanged:** the 3 apron parity tests (per-chunk apron exposed-face set == whole-region
    ground truth, full + banded) stay green; goldens (6 images) byte-for-byte unchanged; a headless
    `--mesher cuboid` sphere renders a complete silhouette. No incrementalism added (extent-changing
    slider edits shift the recentre and force a full re-mesh regardless — same constraint as the
    instanced S6c-2c path); this makes the unavoidable FULL rebuild fast. No rebasing change.

- **Cuboid mesher: render applied/loaded VS block textures (per-face D2Array) — Part of #20.**
  The DEFAULT cuboid path could not show an applied/loaded VS block: it bound only the 3-material
  PROCEDURAL atlas, and `main.rs`/`shot.rs` set the cuboid bound material to `None` for a loaded block,
  so a "Granite"-applied model rendered as procedural Stone. The instanced path already showed it (it
  binds the block's 6-layer D2Array and selects the per-face layer by normal). The cuboid path now does
  the same.
  - **Second pipeline (not a uniform flag).** Added a loaded-block shader (`shaders/cuboid_loaded.wgsl`)
    + a pipeline pair (`loaded_pipeline` / `loaded_debug_pipeline`) on `CuboidMeshRenderer`, with group(1)
    a 6-layer `texture_2d_array` instead of the procedural atlas. The shader reuses the EXACT
    `CuboidUniforms` (camera/half-extent/overlay/band), the same vertex layout (so geometry is
    pixel-aligned with the procedural path), the same per-voxel slice maths + per-face UV directions, and
    `face_layer(normal)` (0 +X,1 -X,2 +Y,3 -Y,4 +Z,5 -Z — identical to `voxel.wgsl`), then
    `textureSample(tex, samp, fract(texcoord), layer)`. A separate pipeline was chosen over a uniform
    flag + dual bind groups so there are NO dead/unused bindings (the procedural atlas and the D2Array
    have different group(1) layouts; one bind group is live per draw). The procedural atlas pipelines are
    untouched, so the goldens' procedural path is byte-for-byte unchanged.
  - **No new texture upload / no CPU-pixel retention needed.** `draw` now takes `Option<&wgpu::BindGroup>`;
    `render_frame` passes the active `MaterialSource::Loaded(bind_group)` straight through, so the cuboid
    path REUSES the very bind group `LoadedMaterial` already built (against `build_face_material_layout` —
    layout-compatible with the cuboid loaded pipeline). `LoadedMaterial` discards its CPU pixels after
    upload, but that's irrelevant here since we bind the existing GPU bind group; no retention change.
  - **Plumbing.** `main.rs` + `shot.rs` keep `bound = None` for a loaded block (disables procedural
    modulation/atlas, which the loaded pipeline ignores) — the loaded D2Array reaches the cuboid path via
    `render_frame`'s `material` arg, not via `update_uniforms`.
  - **Headless verification (no VS install).** New `LoadedMaterial::from_face_layers` (raw 6-face RGBA →
    sRGB D2Array, same shape as `from_faces`) + a `shot --synthetic-block` flag that applies six distinct
    solid-colour faces. Rendered a 2×2×2 box `--synthetic-block` on BOTH paths: cuboid shows the loaded
    per-face colours (top blue +Y, right red +X, left magenta +Z) — NOT procedural Stone — and matches
    the instanced path (viewport pixel-diff 0.034%, only AA at silhouette edges). Procedural box cuboid
    (no block) still shows Stone, unchanged.
  - **Gate.** `cargo build --bins`, `cargo clippy --all-targets` (no new warnings), `cargo test` (171
    pass), `cargo test --features gpu --test golden` (6 cuboid goldens green). Instanced path unchanged.

- **Cuboid mesher: apron-aware per-chunk meshing + per-chunk GPU buffers (S6c-2d) — Part of #20 (step 4).**
  The DEFAULT cuboid render path (the golden-covered one) now meshes PER CHUNK from per-chunk grids + a
  1-voxel neighbour apron, with one GPU vertex/index buffer per chunk, instead of densifying +
  greedy-decomposing the WHOLE region into one monolithic buffer. Goldens stayed green (all 6 within run
  jitter, 0.002–0.03% < 0.5% threshold); per-chunk village + banded torus PNGs show no seam lines, correct
  caps. INCREMENTAL dirty-only rebuild is the NEXT step (S6c-2e); this step rebuilds wholesale.
  - **Apron-aware per-chunk meshing (`cuboid_mesh.rs::build_chunk_meshes_with_apron`).** For each
    `(coord, &grid)` from `resident_render_chunks`: build a GLOBAL occupancy `HashMap<[i64;3],u16>` +
    cloud anchor (`world_offset = min_world − 0.5`) over the UNION of all chunk grids (= the assembled
    whole grid, by the S6c-2a seam), so emitted world positions are byte-identical to the whole-region
    mesher (pixel parity). Then per chunk: densify its OWN voxels into an INTERIOR region (apron cells
    air, so no box ever grows into the apron), and a co-located APRON region of the same extent whose
    every cell — interior AND the 1-voxel border — is read from the GLOBAL occupancy. Decompose the
    INTERIOR via `decompose_into_boxes`; `emit_box_faces` tests `face_is_exposed` against the APRON.
    The apron makes a seam face between two solid chunks correctly culled, so the chunk's exposed-face
    SET equals whole-region meshing's. The apron emits NO geometry.
  - **Per-chunk band clip.** The layer-range band (absolute layers) is applied per chunk: a global index
    Y is in-band iff `base_layer + gy ∈ [band_min, band_max]` (`base_layer = floor(world_offset.y + 0.5
    + half_y)`). BOTH the interior and the apron are band-masked, so a band edge inside a chunk reads the
    masked neighbour as air and synthesises the real slab CAP — identical to the whole-region region-mask
    + re-mesh. `rebuild_for_band` re-meshes every chunk when the band changes (real caps, not a fragment
    discard).
  - **Per-chunk GPU buffer cache + draw (`CuboidMeshRenderer`).** Replaced the monolithic vertex/index
    buffer + `CuboidMesh.chunks` index ranges with `HashMap<[i32;3], CuboidChunkBuffers>` (own
    vertex+index buffer + count + world AABB per chunk, keyed by absolute chunk coord — mirrors the
    instanced `InstancedChunkBuffers`). `update_uniforms` frustum-culls each chunk by its AABB (sorted
    for deterministic draw order); `draw` does one `set_vertex_buffer`/`set_index_buffer`/`draw_indexed`
    per visible chunk. `new_from_chunks(chunk_grids, grid_dimensions)` builds directly from the accessor;
    `new(grid)` is kept as a WRAPPER that buckets the whole grid by `floor(world/chunk_extent)` (the
    instanced key) → identical mesh. Rebasing/shader/atlas path unchanged.
  - **Routing.** `shot.rs` builds the cuboid renderer via `resident_render_chunks` → `new_from_chunks`
    (falls back to the whole-grid wrapper only when the scene has no chunkable extent), so the goldens
    exercise the per-chunk accessor path. `main.rs` keeps `new(&self.grid, …)`, which now routes through
    the same per-chunk apron mesher via the wrapper.
  - **Whole-region builder kept as the structural REFERENCE.** `build_cuboid_mesh` /
    `build_cuboid_mesh_banded` / `region_from_voxel_cloud` stay (simplified to one flat vertex/index
    list, no chunk partition) as the parity test's reference + the older CPU tests' adapter.
  - **STRUCTURAL test (durable proof, +3, lib 168 → 171).**
    `per_chunk_apron_exposed_face_set_equals_whole_region` asserts the per-chunk-with-apron VISIBLE
    exposed-face set == whole-region's == the ground-truth genuinely-exposed set (derived straight from
    occupancy) for sphere/cylinder/torus/box/tube across sizes INCLUDING multi-chunk (8-block axes at
    density 8 = 2 chunks/axis; asserts ≥1 case actually spanned multiple chunks). "Visible" = the subset
    of emitted unit faces backed by air: the mesher emits a whole MERGED box face when ANY cell behind it
    is air (over-drawing sub-faces backed by solid), and those over-draw quads are always back-face-culled
    or depth-occluded — so the VISIBLE set, not the raw emitted-quad count, is the rendering invariant.
    `solid_slab_across_chunk_seam_has_no_interior_faces` (a solid 2-chunk box → no leaked interior seam
    faces, surface == 6 sides) and `per_chunk_band_clip_face_set_equals_whole_region` (banded torus, both
    paths == band-masked ground truth) round it out.
  - **Gate green.** `cargo build --bins`, `cargo clippy --all-targets` (no new warnings), `cargo test`
    (171 lib pass), `cargo test --features gpu --test golden` (6 green, all within run jitter).

- **Instanced render path: incremental dirty-chunk rebuild via evicted-set (S6c-2c) — Part of #20 (step 4).**
  `main.rs::rebuild_geometry` now rebuilds ONLY the per-chunk GPU buffers an edit touched, instead of
  clearing + rebuilding every chunk wholesale. A/B pixel-identical (0% diff, byte-identical PNGs); cuboid
  path + goldens untouched.
  - **Incremental logic.** The resolve cache's `invalidate_aabb(&edit_aabb, density)` returns the dirty
    absolute chunk-coords (`evicted_chunks`). After resolving, `resident_render_chunks` hands every covering
    chunk's freshly-rebased grid. The new `VoxelRenderer::incremental_rebuild_from_chunks(device, render_chunks,
    evicted)` (re)builds a chunk's buffer ONLY if it is DIRTY (`coord ∈ evicted`) or NEW (no GPU buffer yet);
    every other covering chunk is a resolve-cache HIT → byte-identical grid → its existing buffer is kept
    untouched. The decision is a pure, GPU-free function `renderer::incremental_rebuild_plan(resident, evicted,
    occupied_covering) -> {rebuild, evict}` that both the renderer and the CPU test drive.
  - **Vacated-coord eviction.** A coord that is no longer an OCCUPIED covering chunk — a removed/shrunk node
    vacated it, OR an edit turned it empty — is dropped from `chunk_buffers` (`plan.evict`). Only non-empty
    covering chunks are ever stored (a zero-voxel chunk allocates no buffer, exactly as a wholesale rebuild).
  - **Fallback-to-full cases.** The wholesale `rebuild_all_from_chunks` is kept for: the first build (no
    previous index), a density change / region-spanning Part edit (`edit_aabb_since` → `None`), AND — the
    stale-chunk fix below — a composite-recentre shift.
  - **Stale-chunk risk handled (the surprising part).** Every cached chunk's voxel positions are rebased to
    the scene's composite RECENTRE (floating origin). A move that shifts the active region's extent changes the
    recentre, which rebases EVERY chunk's contents — even chunks far from the edit — so an incremental rebuild
    would keep stale (old-origin) buffers for the untouched chunks. `rebuild_geometry` now tracks
    `previous_recentre_voxels` and forces a FULL rebuild when the recentre changes (`Scene::recentre_voxels_for_resolve`
    made `pub`). The CPU test mirrors this: a recentre shift → full rebuild.
  - **Observability.** `VoxelRenderer::last_rebuilt_chunk_count()` reports `|dirty ∪ new|` after an incremental
    rebuild (every chunk after a wholesale one). Wired into shot's `--instanced-via-chunks` diagnostic.
  - **Tests (168 lib, +2).** `incremental_rebuild_equals_full_rebuild_for_every_edit_kind` models the GPU cache
    on CPU as `coord → occupied multiset` (the byte-identical buffer proxy), drives it through the SAME
    `incremental_rebuild_plan`, and asserts the post-edit cache (coords + each chunk's instance multiset) is
    IDENTICAL to a full wholesale rebuild for scene B — across recolor / resize / move / add / remove, all with
    the recentre pinned by static anchor nodes, plus a strict `rebuilt < total` (genuinely incremental) check.
    `localized_recolor_rebuilds_few_chunks` pins that a small far node recolor rebuilds < half the chunks.
    A/B headless (`--mesher instanced` ± `--instanced-via-chunks`, which now also exercises the incremental path
    with an empty edit → 0 rebuilt): byte-identical. 6 cuboid goldens green.

- **Instanced render path: per-chunk GPU buffer cache (S6c-2b) — Part of #20 (step 4).**
  The instanced FALLBACK (`--mesher instanced`; cuboid is the default + the only golden-covered path)
  now maintains ONE GPU instance buffer per resident chunk instead of a single grown monolithic buffer
  built from the whole grid. The cuboid path is untouched this step. A/B pixel-identical (0% diff).
  - **`InstancedChunkBuffers` + a `HashMap<[i32;3], InstancedChunkBuffers>` cache (renderer.rs).** Each
    entry owns its own instance `wgpu::Buffer` + `instance_count` + world `Aabb`, keyed by ABSOLUTE chunk
    coord (the coord the resolve cache's `resident_render_chunks` reports). Replaces the old single
    `instance_buffer` + `instance_capacity` + `Vec<Chunk>` ranges. A zero-voxel chunk is skipped (no
    buffer allocated); every cache entry has `instance_count > 0`.
  - **Methods.** `rebuild_chunk(&device, coord, &chunk_grid)` builds/replaces one chunk's buffer (or
    evicts it if the grid is empty); `evict_chunk(coord)` drops one chunk's buffer (for the S6c-2c
    dirty path); `rebuild_all_from_chunks(&device,&queue,&[([i32;3],&VoxelGrid)])` clears + rebuilds
    every chunk wholesale (the path used THIS step). New `instances_for_chunk(grid) -> Option<(Vec<
    VoxelInstance>, Aabb)>` turns one per-chunk grid into its instances + AABB (`None` when empty).
  - **`rebuild_instances` WRAPPER kept** (so `shot.rs` + tests with a whole grid work unchanged): it
    buckets the whole grid into per-chunk sub-grids by `floor(world_position / chunk_extent)` (the same
    key `bucket_instances_into_chunks` uses) and calls `rebuild_chunk` for each. `VoxelRenderer::new`
    now builds the cache through this wrapper.
  - **`update_uniforms` cull + `draw`.** The frustum cull iterates the resident per-chunk buffers,
    keeping the coords whose `Aabb` intersects the frustum (sorted for a deterministic, `--debug-chunks`-
    reproducible order — cross-chunk order is pixel-irrelevant: opaque + depth-tested). `draw` does one
    `set_vertex_buffer(1, …)` + `draw_indexed(.., 0..instance_count)` per visible chunk over its OWN
    buffer.
  - **`main.rs::rebuild_geometry` drives it via the accessor.** The instanced branch now calls
    `chunk_resolve_cache.resident_render_chunks(scene, density, 0)` then `rebuild_all_from_chunks`,
    consuming the returned `Vec` (which holds an immutable borrow of the cache) FULLY — all GPU buffers
    built — before `drop`ping it, so no further `&mut` cache call overlaps the borrow. The assembled
    `grid` is still resolved (the fog / cuboid / scrubber consume it).
  - **A/B pixel-identical (REQUIRED, passed at 0%).** `--mesher instanced` for `--demo-scene`,
    `--demo-village`, `--shape sphere`: HEAD `cd874a0` (monolithic) vs the new wrapper path are
    BYTE-for-byte identical; and a new hidden `shot --instanced-via-chunks` flag (rebuilds the renderer
    through the accessor exactly as `main.rs` does) renders BYTE-for-byte identical to the wrapper path
    for all three — so monolithic == wrapper == accessor/main path, all 0% diff. PNG visually correct.
  - **Tests (+2, lib 164 → 166).** `per_chunk_instances_match_monolithic_bucketing_per_chunk` (per-chunk
    seam's instances, grouped by chunk coord, == the monolithic `bucket_instances_into_chunks` slice per
    chunk, as a bit-exact multiset) and `instances_for_chunk_is_none_when_empty` (the zero-voxel skip).
  - **Gate green.** `cargo build --bins`, `cargo clippy --all-targets` (no new warnings), `cargo test`
    (166 pass), `cargo test --features gpu --test golden` (6 cuboid goldens pixel-identical — cuboid
    path untouched).

- **Per-chunk render accessor + `invalidate_aabb` evicted-set (S6c-2a) — Part of #20 (step 4).**
  Pure-CPU, additive, no render-path change → goldens untouched. Two seams the upcoming per-chunk
  renderer + GPU cache need, exposed WITHOUT moving any draw path:
  - **`ChunkResolveCache::resident_render_chunks(&mut self, scene, voxels_per_block, lod) ->
    Vec<([i32;3], &VoxelGrid)>`.** Binds the cache to the composite recentre/floating-origin for
    `(scene, density, lod)` EXACTLY as `resolve_region` does (via the existing `bind_and_collect_region`),
    then returns each covering chunk as `(absolute_chunk_coord, &rebased_grid)`. The grids are the SAME
    rebased per-chunk grids whose union `resolve_region` assembles — byte-identical (each already rebased
    in i64 inside `resolve_chunk_rebased` before the f32 downcast). Borrow-checker shape: the grids are
    BORROWED from `self.chunks` (`&VoxelGrid`), so resolving misses (needs `&mut self`) happens FIRST in
    `bind_and_collect_region`, and the immutable borrows are gathered only AFTER every covering chunk is
    resident (all cache HITs, no interleaved mut/shared borrow). The returned `Vec` borrows `self`
    immutably for its lifetime — the renderer consumes it before the next `&mut` cache call.
  - **`invalidate_aabb` now returns `Vec<[i32;3]>`** — exactly the chunk-coords evicted (resident coords
    intersecting the edit AABB; or every resident coord on the belt-and-braces density-mismatch clear) so
    the GPU cache can evict in lockstep. The one `main.rs` caller (`rebuild_geometry`) binds the result to
    `_evicted` (not wired to the GPU yet).
  - **Tests (+6, lib 158 → 164).** `render_chunks_match_resolve_region_for_{all_shapes,demo_scene,
    demo_village}` (union of render chunks BIT-IDENTICAL to `resolve_region`'s assembled grid via the
    `occupied_multiset` `f32::to_bits` helper, returned coord set == `covering_chunk_range`, each voxel
    owned by its returned coord's half-open box `[c·E,(c+1)·E)`), `render_chunks_empty_for_part_only_scene`,
    `invalidate_aabb_returns_exactly_the_evicted_coords`, `invalidate_aabb_density_mismatch_reports_all_resident_evicted`;
    `empty_edit_aabb_evicts_nothing` extended to assert an empty returned set.
  - **Gate green.** `cargo build --bins`, `cargo clippy --all-targets` (no new warnings), `cargo test`
    (164 pass), `cargo test --features gpu --test golden` (6 goldens pixel-identical).

- **Fix per-chunk fog silent holes past MAX_FOG_CHUNKS → graceful disable — Part of #20 (step 4).**
  Per-chunk onion fog is now the default. The CPU occupancy builder
  (`renderer.rs::build_per_chunk_fog_occupancy`) previously `keys.truncate(MAX_FOG_CHUNKS)`'d the resident
  non-empty chunk list when it overflowed 1024; the dropped chunks then had no atlas tile, so the
  raymarch's occupancy sample read 0 inside them → **fog silently vanished (holes)** in part of a large
  scene rather than failing honestly. Now the builder detects `keys.len() > MAX_FOG_CHUNKS`, logs a
  one-line `eprintln!`, and returns NO volumes — which makes `upload_grid_per_chunk` take its **existing**
  `chunk_count == 0` graceful-disable path (`per_chunk_active = false`), CONSISTENT with the neighbouring
  atlas-dimension-exceeded branch. Net: a too-large region shows NO fog (honest) instead of fog-with-holes
  (wrong). Long-term fix (region-scope the fog to resident/visible chunks so the resident set stays small)
  stays tracked in #20 step 4. New CPU test `per_chunk_fog_disables_past_max_fog_chunks` (1025 chunks →
  empty; exactly 1024 → still renders). Headless: `--demo-scene --fog perchunk` (19 chunks) renders fog
  correctly (unaffected); a 96³ debug-clouds scene (1816 chunks) now renders with NO fog, no holes.

- **Decouple camera/gizmo/lattice/scrubber dims from the assembled grid (S6c-1) — Part of #20.**
  Behaviour-preserving refactor + prep for the per-chunk renderer (S6c step 4). The camera auto-frame,
  origin gizmo, block lattice, fine floor grid and layer scrubber no longer read the assembled monolithic
  `VoxelGrid::dimensions`; they now take the region dimensions straight from the SCENE. The renderer /
  mesher / fog are UNCHANGED — they still consume the assembled grid (that switch is step 4). ZERO
  behavioural change: the assembled grid is *literally* sized to `Scene::placed_region_dimensions(density)`
  (both `resolve_region` and the chunk-cache reassembly seed their output to it), so the substituted values
  are byte-identical.
  - **Source of the dims now.** `main.rs`: a new `region_dimensions_for(scene, density, grid)` helper —
    `scene.placed_region_dimensions(density)` for a chunkable scene, falling back to `grid.dimensions` for a
    Part-only scene (no composite extent → `placed_region_dimensions` is `[0,0,0]`; the app's cache resolve
    already yields `[0,0,0]` there, so the fallback is trivially identical). Wired into `new()` (initial
    setup), `rebuild_geometry` (gizmo/lattice rebuild, scrubber rescale, camera re-frame) and the per-frame
    scrubber `grid_y`. `shot.rs`: a `region_dimensions` computed next to the resolve mirroring the exact
    resolve branch (`placed_region_dimensions` for chunkable, explicit `region × density` for the Part-only
    `--shape debug-clouds` path) + a `debug_assert_eq!` against `grid.dimensions`; wired into the gizmo,
    lattice/floor and camera auto-frame. `placed_region_dimensions` widened `pub(crate)` → `pub` so the
    `shot` bin crate can call it.
  - **Equivalence proof.** New CPU test `scene::tests::placed_region_dimensions_equals_assembled_grid`
    asserts `placed_region_dimensions(density)` equals the assembled grid's `dimensions` for BOTH resolve
    paths (monolithic `resolve_region` AND the chunk-cache reassembly) across all SDF shapes, flat/odd sizes
    at several densities, a placed multi-node scene, and an instanced village. Lib tests 156 → 157.
  - **Gate green.** `cargo build --bins`, `cargo clippy --all-targets` (no new warnings), `cargo test`
    (157 pass), `cargo test --features gpu --test golden` PIXEL-IDENTICAL (the gizmo + lattice are visible
    in the goldens — they did not move, confirming the dims were truly equal). Headless sanity: `--demo-scene
    --gizmo --lattice --floor` read back correct (axes through centre, teal lattice on the composite box,
    floor grid, all framed); the Part-only `--shape debug-clouds` overlay path also renders with no assert.

- **Region-scoped whole-grid consumers: diameter readout + `.vox` export (S6d) — Part of #20 (folded in from #28).**
  Pure-CPU, additive, no render-path change → goldens untouched. The two consumers that today assume one whole
  recentred `VoxelGrid` now have region-scoped variants that operate over the cache's per-chunk grids and produce
  results provably identical to the whole-grid computation — so the S6c monolithic-bridge removal won't change what
  they report. The old whole-grid paths are left intact (the app still calls them; the switch is S6c).
  - **`widest_run_in_band` region variant.** New free fn `voxel::widest_run_in_band_over_chunks(region_dimensions,
    chunk_grids, band_min, band_max)` + cache method `ChunkResolveCache::widest_run_in_band(scene, vpb, lod, lo, hi)`.
    **Cross-seam stitching (the subtle part):** rather than combine per-chunk partial runs, EVERY voxel from EVERY
    covering chunk is bucketed into ONE shared per-`(y, z)` occupancy row keyed by the GLOBAL X index
    (`i = round(world_x + grid_x/2 − 0.5)`, identical to the whole-grid fn). Two voxels straddling a chunk seam land
    at adjacent global X in the same shared bitset, so the seam vanishes and the run-scan sees one contiguous span.
    Equal to whole-grid by construction (chunk union == monolithic occupied set; identical bucketing/scan arithmetic).
  - **`.vox` export region variant.** `from_grid` now delegates to a new core `VoxExport::from_region_voxels(
    region_dimensions, chunk_voxels_iter, rgba)` (ONE bucketing path → region & whole-grid exports can't drift);
    cache method `ChunkResolveCache::vox_export(...)` assembles the active region from per-chunk grids and exports.
    Equal because the chunk union == the monolithic occupied set and the index-recovery/tiling arithmetic is identical;
    only per-model voxel ORDER differs (chunk vs stamp order), which a MagicaVoxel reader treats as the same model.
    Active-region scoping kept; streamed multi-region export still deferred.
  - **Tests (+8, 148 → 156).** `chunk_cache`: `region_widest_run_matches_whole_grid_for_{all_shapes,demo_scene,
    demo_village}`, `region_widest_run_stitches_runs_across_chunk_seams` (a 20-block bar = 320 voxels crossing ~5
    chunk seams; asserts run > one chunk extent AND == whole-grid), `region_widest_run_single_voxel_and_empty_band`.
    `vox_export`: `region_vox_export_equals_whole_grid_for_{shapes,demo_scene}` and `...when_split_over_256`
    (forces a multi-model 256-split). Goldens green (`cargo test --features gpu --test golden`).

- **Disk-backed chunk store + bounded-RAM LRU eviction (standalone) (S6b) — Part of #20 (out-of-core, part 2).**
  Pure-CPU, standalone component (`src/disk_chunk_store.rs`): `DiskChunkStore` keeps at most a configured
  number of chunks resident in RAM and evicts the **least-recently-used** ones to disk as serialised
  `CompressedChunk`s (S6a's shape), transparently reloading on access. NOT wired into the live
  resolve/render path — that integration is S6c (when the monolithic-grid bridge is removed and the
  floating-origin/rebasing coupling is reworked), so goldens are untouched.
  - **API.** `DiskChunkStore::new(dir, capacity)` (idempotent `create_dir_all`; panics on capacity 0);
    `put(key, CompressedChunk)`, `get(key) -> io::Result<Option<CompressedChunk>>`, `contains`,
    `capacity`, `resident_count`, `stats`. Key is `ChunkCacheKey { chunk_coord: [i32;3], lod: u32 }`
    (exact cache key shape). `get` returns a clone (the reload path mutates/evicts — clean borrow story;
    `CompressedChunk` is a compact struct).
  - **Capacity model = resident chunk COUNT** (not a byte budget): gives a crisp, always-true invariant
    `resident_count() <= capacity` and matches the unit the future `ChunkResolveCache` thinks in; a byte
    budget would need a fuzzy per-chunk size estimate. Justified inline.
  - **Serialisation = `serde_json`** (existing dep, already S6a round-trip-tested through JSON; **no new
    dep**). Codec isolated behind `write_chunk_file`/`read_chunk_file` (a binary `bincode` is a
    two-function swap — noted for S6c, JSON is fine standalone).
  - **LRU mechanism.** Monotonic logical clock; each `put`/`get` stamps the touched chunk's `last_used`,
    eviction picks `min_by_key(last_used)`. Over-capacity `put` (new key) or a reload evicts exactly one
    LRU resident chunk first, so the bound is never breached. Overwriting a resident key never evicts
    (the set didn't grow) and refreshes its LRU. Re-putting an evicted key brings it resident and deletes
    the stale disk file (sets stay disjoint).
  - **Observability (`DiskChunkStoreStats`).** `evictions`, `disk_reloads` (bumps ONLY on a `get` for an
    evicted key — the "no needless reloads" proof), `resident_count`, `on_disk_count`.
  - **Windows handling.** Path-safe filenames (`chunk_<x>_<y>_<z>__lod<n>.json`, negatives encoded with
    an `n` prefix via unsigned magnitude so `i32::MIN` is safe — only `[0-9a-z_]`, injective). No file
    handle held across calls (`fs::write`/`fs::read` open+close internally) → no lock issues on
    delete/rewrite. Idempotent `create_dir_all`.
  - **Tests (10 new, lib total 138 → 148, all green).** `capacity_zero_panics`,
    `put_then_get_resident_does_not_count_as_reload`, `round_trip_through_disk_eviction_preserves_grid`,
    `bounded_ram_invariant_and_all_evicted_retrievable`, `lru_touch_survives_and_least_recent_is_evicted`,
    `negative_and_large_coords_round_trip_distinctly`, `reload_counter_only_on_evicted_access`,
    `overwrite_resident_key_does_not_evict`, `reput_evicted_key_supersedes_disk_copy`,
    `directory_creation_is_idempotent`. Temp dirs are unique-per-test under the system temp and
    auto-cleaned via an RAII `TempDir` guard (verified 0 left behind). Goldens 6/6 still green (path
    untouched).
  - **S6c wiring note.** Back `ChunkResolveCache`'s `HashMap<ChunkCacheKey, VoxelGrid>` with a
    `DiskChunkStore<CompressedChunk>` — but the store is keyed/serialised in the cache's **current**
    density+floating-origin binding, and a rebind (`rebind_if_changed`) currently `clear()`s every chunk.
    So S6c must either re-key the disk store on rebind (origin/density are part of the on-disk identity)
    or store **origin-independent** chunks and apply the rebase on load; otherwise an evicted chunk would
    reload at a stale origin and mis-place far geometry (the S4b precision fix). Pick one before persisting.

- **Per-chunk material palette + sparse storage (lossless) (S6a) — Part of #20 (out-of-core, part 1).**
  Pure-CPU, additive data structure (`src/chunk_storage.rs`) — the future out-of-core store's on-disk
  shape for one resolved chunk grid. NOT yet wired into the resolve/render path (store integration +
  dropping the monolithic bridge are later S6 steps), so goldens are untouched.
  - **`CompressedChunk` layout.** `dimensions` + occupied bounding box (`min_corner_voxels: [i64;3]`,
    `box_spans: [u32;3]`) + `centre_fraction: [f32;3]` (the per-axis shared sub-integer offset of every
    voxel centre — `.5` for even-dim axes, `.0` for odd) + a first-seen-order de-duplicated
    `material_palette: Vec<u16>` + an `Occupancy` enum. serde-serialisable (`Serialize`/`Deserialize`),
    `Eq` dropped because of the `f32` fraction (`PartialEq` kept — fractions are exact constants).
  - **Encoding = sparse default with a dense bit-packed fallback (per-chunk heuristic).** Sparse stores
    `(local_linear_index, palette_index, block_local_coord)` per occupied cell; dense stores a
    `ceil(log2(palette+1))`-bit palette index per cell over the occupied box (air = reserved slot 0) +
    one `block_local_coord` per occupied cell in scan order. `compress` builds both and keeps whichever
    has the smaller **binary** layout (the heuristic measures the compact on-disk byte size, NOT JSON).
    For solid SDF shapes dense wins (~5×); for genuinely sporadic occupancy (<~cells/48 voxels) sparse
    wins. Both are exact inverses of `decompress`.
  - **Lossless proof.** Positions are reconstructed as `(min_corner + local_offset) as f32 +
    centre_fraction`, reproducing the producer's own `i + 0.5 − half` + integer-translation arithmetic,
    so the round-trip is **byte-identical** on the f32 bits (keyed via `to_bits`), `block_local_coord`,
    and `material_id`. `compress`/`decompress` exported from `lib.rs`.
  - **Measured ratios (binary on-disk model, raw = 17 B/occupied voxel).** sphere 5³@16 whole grid
    **5.25×** (dense); box 4³@16 solid **5.44×** (dense); per-chunk sphere/torus pieces **5.25×** aggregate;
    a 0.5%-fill 40³ grid **1.85×** (sparse); demo-village non-empty chunks **5.37×** aggregate.
  - **Tests (10 new, lib total 128 → 138, all green).** `round_trip_empty_chunk`,
    `round_trip_full_single_material_chunk`, `round_trip_multi_material_chunk`,
    `round_trip_real_resolved_chunks_across_shapes` (all 5 SDF kinds, every covering chunk),
    `round_trip_demo_scene_and_village_chunks`, `round_trip_part_only_debug_clouds_grid`,
    `round_trip_randomized_fuzz_varied_fill_and_materials` (varied extent/fill%/material count),
    `palette_has_no_duplicates_and_covers_every_material`,
    `serde_round_trip_through_json_equals_original_grid`, `report_compression_ratios_on_real_chunks`.
    Goldens: 6/6 still **0.00000%** (path untouched).
  - **Surprise/learning.** Resolved grids with an ODD-dimensioned axis centre voxels on integers
    (`n + 0.0`), not `n + 0.5` — the producer's `i + 0.5 − half` with a half-integer `half`. The first
    naïve "centres are always n+0.5" assumption was lossy for `[1,1,1]`-style grids; `centre_fraction`
    (uniform per axis within one grid) fixes it and is debug-asserted in `compress`.

- **Per-chunk onion fog is now the DEFAULT + fog golden (S5b) — Part of #28 (ADR 0002 matrix row 7
  / O6 fog).** Flips the S5a-added per-chunk path from opt-in to default; the legacy whole-grid path
  is KEPT as a fallback. O6 is now RESOLVED (per-chunk, no fidelity reduction — see ADR 0002).
  - **Default flipped in both entry points.** `shot`'s `ShotOptions::default().fog_mode` is now
    `FogMode::PerChunk` (was `WholeGrid`); `--fog=wholegrid` still selects the legacy path (verified:
    a no-`--fog` render logs `fog: per-chunk mode — N resident chunk volume(s) (atlas active)`, while
    `--fog=wholegrid` / `--fog wholegrid` takes the whole-grid path). The **windowed app** (`main.rs`)
    gained a `WindowedState::fog_mode` field defaulted to `PerChunk` and a shared
    `upload_fog_occupancy` helper that dispatches on it; BOTH the initial upload and every
    `rebuild_geometry` re-upload now go per-chunk by default (was the whole-grid `upload_grid`).
  - **Fog golden (`tests/golden/onion-fog-perchunk.png`).** A new GPU-gated golden: an 8³-block
    **sphere** (grid 128³, **8 resident chunk volumes**, 2 per axis) with an onion-skinned equatorial
    band — layers `[56,72]` render as the crisp solid stone disk while the sphere's volume above/below
    ghosts as soft blue haze (`--onion 8`), sampled from the per-chunk atlas (`--fog perchunk`, pinned
    explicitly), fixed camera `--theta 0.7 --phi 1.05`. **READ before commit:** continuous fog haze,
    NO seam lines despite the fog crossing every chunk seam. A/B vs `--fog=wholegrid` at the same
    settings = **0.00000%** differing (max channel diff 1/255). Generated by rendering ONLY the new
    case into the reference path (NOT `UPDATE_GOLDENS=1`, which would have rebaselined the other 5).
  - **The 5 existing goldens stayed byte-stable** (`0.00000%` each) — they don't enable onion skin,
    so `onion_active` is false and the fog default never touches them. Golden suite is now **6 cases,
    all green**.
  - **Scale proof (default now renders fog where whole-grid can't).** A `box 200×2×2 @16` (X axis
    3200 vx > `max_texture_dimension_3d` 2048): the default logs `per-chunk mode — 50 resident chunk
    volume(s) (atlas active)` and the 3D viewport differs from the `--fog=wholegrid` render (which
    disables fog at this scale) by 650 px (max channel 85) — i.e. fog is present by default, absent on
    whole-grid.
  - **NOT in this step (moved to #20):** region-scoping the scrubber / diameter / `.vox` export — left
    as-is, not regressed. Whole-grid fog path NOT deleted.
  - **Gate:** `cargo build --bins` ✓, `cargo clippy --all-targets` clean ✓, `cargo test` **128** ✓,
    `cargo test --features gpu --test golden` green (**6** cases, all 0.00000%, incl. the new fog one) ✓.

- **Per-chunk onion-fog occupancy behind `--fog=perchunk` (S5a) — Part of #28 (ADR 0002 matrix
  row 7 / O6 fog, the highest-risk row).** Additive + flagged: a `FogMode` selector (`wholegrid`
  DEFAULT, `perchunk` new) wired through `shot` (`--fog=perchunk`). The default is unchanged —
  the app and every golden still take the whole-grid path; goldens stay **green** (0.00000%).
  - **Per-chunk storage + binding.** `build_per_chunk_fog_occupancy` buckets the SAME recentred
    grid the whole-grid path uploads into one apron'd `R8` volume per resident chunk (keyed by
    chunk coord), then `upload_grid_per_chunk` packs them into ONE small 3D **atlas** (a cubic-ish
    tile grid, one `(extent+2)³` tile per chunk) plus a metadata uniform of per-chunk world origins
    + tile indices. The shader (`onion_fog_perchunk.wgsl`) marches in recentred world space and at
    each sample **candidate-samples** the owning chunk's tile (compute chunk coord → find record →
    one trilinear sample). Chosen over per-chunk multi-bind because WGSL has no `texture_3d` array
    and a variable bind-count loop; one atlas keeps it a single sample with a bounded uniform.
  - **1-voxel apron / seam smoothness.** Each tile's border layer (`-1..=extent`) is filled from
    the GLOBAL occupancy (the true neighbour voxel, not a clamp), so a ray crossing a chunk seam
    trilinear-interpolates against the real neighbour density — no banding/discontinuity at seams.
    CPU test `per_chunk_apron_reflects_neighbour_and_boundary` pins this (+ world-origin test).
  - **Dodges the single-3D-texture limit.** The atlas dimension is bounded by the chunk COUNT
    (`cbrt`, ×pad), NOT the whole-grid extent. **Scale proof:** a `box 200×2×2 @16` (X axis 3200 vx
    > `max_texture_dimension_3d` 2048) **disables** the whole-grid fog (no haze) yet renders fog
    via 50 per-chunk volumes.
  - **A/B match:** same fogged sphere & wide torus (9 chunk volumes, fog crossing 2 seams/axis) —
    `perchunk` vs `wholegrid` diff **0.0000%** (max channel ≤ 3/255), NO seam artifact. NOT yet
    region-scoped (scrubber/diameter/.vox stays whole-grid — that is S5b).

- **Origin-rebased (camera-relative) rendering; far-offset precision fixed (S4b) — Part of #18
  (ADR 0002 Decision 2, "origin-rebased (camera-relative) f32 rendering" + matrix row 3).**
  Replaces the recentre-AFTER-f32 path with a **floating-origin rebase done in i64 BEFORE the f32
  downcast**, so rendered f32 magnitudes stay small no matter how far geometry sits from the
  absolute origin. Near scenes are **pixel-identical** (goldens 0.00000%, NOT rebaselined); the
  far-offset demo is now **byte-identical** to the near box.
  - **The floating origin = the composite recentre** (`Scene::recentre_voxels_for_resolve`, an
    integer-block-aligned point). Choosing it as exactly today's recentre is what makes near
    framing reproduce bit-for-bit (the grid-overlay block phase is unchanged → goldens locked).
  - **Where the i64/f64 subtraction happens.** New `Scene::resolve_chunk_rebased(chunk_coord,
    vpb, lod, floating_origin_voxels)` and a rebased `stamp_producer_into_chunk`: the stored
    `world_position = local + (world_offset·d − floating_origin)` with the **subtraction in i64**
    before the `as f32`. The chunk-membership clip moved to **f64 absolute** so a far chunk's
    boundary voxels are never misclassified by f32 rounding. The bare `resolve_chunk` keeps the S0
    ABSOLUTE contract (floating origin `[0,0,0]`) so every S0/S2/S3 parity + placement test is
    untouched.
  - **Both render paths rebased.** The renderer consumes one assembled monolithic grid (the S2
    bridge), so the floating-origin translation is applied **per chunk at resolve time** in
    `ChunkResolveCache`: the cache now binds to `(density, floating_origin)` and resolves each
    covering chunk ALREADY rebased (`chunk_for_current_binding`), dropping the old per-voxel f32
    recentre subtract in `resolve_region`. The default **cuboid** path AND `--mesher instanced`
    both draw the rebased grid identically (both verified byte-identical far==near). The shader
    `voxel_absolute_position = world_position + grid_half_extent` already carries absolute block
    phase; because `grid_half_extent` is identical for far/near and the rebased `world_position`
    is byte-identical, the per-voxel slice + grid overlay are byte-identical — no shader change
    was needed for parity (the absolute-phase uniform already existed, matrix row 3).
  - **The S1 "0.2% speckle" was a MISDIAGNOSIS.** At 100_000 blocks (1.6M voxels) the f32 ULP is
    0.125, so the voxel-centre `.5` survived and the 3D box never jittered — the ~0.2% diff was
    just the demo's UI panel text ("Far box"/"100000" vs "Near box"/"0"). The real f32 breakdown
    starts past the 2²⁴≈16.7M-voxel exact-integer ceiling, so **`FAR_OFFSET_BLOCKS` is bumped to
    1_000_000** (16M voxels) — where the OLD path lost the `.5` on EVERY voxel.
  - **PROOF (headless, 3D viewport x<960, UI panel masked):** OLD code at 1M blocks → **13.24%**
    of the 3D viewport differs far-vs-near (maxd 96) — gross jitter. NEW code → **0.00000%**
    (maxd 0) on BOTH cuboid and instanced. Durable CPU guard (no GPU):
    `chunk_cache::tests::far_offset_resolves_byte_identical_to_near_after_rebase` asserts the 1M
    -block box resolves byte-identical (`f32::to_bits`) to the origin box and every voxel keeps
    its `.5` fraction.
  - **Audited, unbroken:** camera auto-frame / orbit / origin gizmo / block lattice / fine floor
    grid / view cube all key off `grid.dimensions` + the recentred grid, byte-identical for near
    — a `--demo-scene --grid --gizmo --lattice --floor` PNG was READ and confirmed correct.
    `.vox` export / scrubber / diameter still consume the recentred whole grid (region-scoping is
    #28) — left as-is, not regressed.
  - **Gate:** `cargo build --bins` ✓, `cargo clippy --all-targets` clean ✓, `cargo test` 126 ✓
    (125 + 1 new), `cargo test --features gpu --test golden` green (all 5 cases 0.00000%) ✓.

- **64-bit (i64) world block addressing (S4a) — Part of #18 (ADR 0002 Decision 2, "world
  addressing to 64-bit").** DATA-MODEL change only — the recentre / camera / render math is
  UNCHANGED (origin-rebasing is the next step, S4b). Near-origin scenes render byte-identical
  (goldens green, NOT rebaselined); the far-offset demo (100_000) still recenters home as before.
  - **`NodeTransform.offset_blocks: [i32;3] → [i64;3]`** (`src/scene.rs`). The whole block-offset
    composition down the tree (`for_each_leaf` / `walk_nodes` visitor signature, `parent_offset`)
    is now i64, so far-apart nodes sum without i32 overflow.
  - **Absolute-voxel math widened to i64** where it multiplies a block offset by density:
    `placed_extent_blocks`, `placed_extent_voxels`, `recentre_voxels_for_resolve`, `resolve_region`
    (recentre + per-leaf translation), `resolve_chunk` (chunk + leaf AABB corners),
    `build_leaf_spatial_index`, and the `stamp_producer{,_into_chunk}` `translation_voxels` params.
    At density 16 a ±10⁹-block offset is ±1.6×10¹⁰ absolute voxels — past i32 — so this frame MUST
    be i64 or it silently truncates.
  - **`VoxelAabb` (`src/spatial_index.rs`) min/max: `[i32;3] → [i64;3]`** (absolute voxels), plus
    its `VoxelAabbKey` diff key. **Chunk coordinate / cache key stayed `[i32;3]`** — the chunk
    coord is `voxel / chunk_extent` (= /64 at density 16), so a ±10⁹-block offset is only ±2.5×10⁸
    chunks, well inside i32. Block→chunk derivation now happens in i64 then narrows via a guarded
    `narrow_chunk_coord` (debug-asserts the i32 fit). **Max safely-supported offset ≈ ±8×10⁹ blocks**
    (where the chunk coord would approach i32::MAX); the absolute-voxel i64 frame itself has far
    more headroom.
  - **Tolerant persistence migration.** serde widens an old `i32` JSON number into the `i64` field
    transparently (a JSON integer carries no width) — no schema bump needed. Tests:
    `settings::tests::old_i32_offset_scene_loads_after_widening_to_i64` (a hand-authored pre-S4a
    document loads + resolves) and `large_i64_offset_round_trips_through_json` (a 3×10⁹-block offset,
    past i32::MAX, round-trips byte-exact through capture→JSON→load).
  - **UI / demos:** `panel.rs` offset DragValues bind the i64 field directly (egui `Numeric`);
    `shot.rs` `FAR_OFFSET_BLOCKS` + all four demo builders use `[i64;3]`.
  - **New CPU test `scene::tests::i64_composition_beyond_i32_range_is_exact`**: a Group(+2×10⁹)
    over a leaf(+10⁹) composes to 3×10⁹ blocks — past i32::MAX — and the producer-true voxel AABB
    is exact in pure i64 (would have wrapped negative under i32). Existing chunked-resolve parity +
    far-offset (100_000) tests still pass unchanged.
  - **Gate:** `cargo build --bins` ✓, `cargo clippy --all-targets` clean ✓, `cargo test` 125 ✓
    (122 + 3 new), `cargo test --features gpu --test golden` green ✓.

- **Edit-AABB chunk invalidation + node spatial index (S3) — Part of #27 (ADR 0002 streaming,
  Decision 3 "dirty-whole-chunk invalidation")** — retires the wholesale `clear()`-on-every-edit:
  an edit now evicts ONLY the cache chunks its world-AABB touches. Render output stays
  BYTE-IDENTICAL (goldens green, not rebaselined).
  - **New `src/spatial_index.rs` — `LeafSpatialIndex` + `VoxelAabb`.** `VoxelAabb` is a half-open
    integer **absolute-voxel** box `[min,max)` — the exact frame `resolve_chunk` / chunk ownership
    (`floor(pos/chunk_extent)`) live in — with `intersects`, `union`, `covering_chunk_range`.
    `LeafSpatialIndex` is a **flat `Vec<(leaf_world_aabb, fingerprint)>`** built by ONE `for_each_leaf`
    walk (`Scene::build_leaf_spatial_index(density)`). Chose a flat list (not octree/grid) because the
    correctness contract is "return the SAME leaf set as a full walk + AABB filter" — a flat list that
    IS that walk, scanned linearly, is provably equal and obviously correct; leaf counts are small
    (tens; low hundreds for instanced `--demo-village`) so the linear scan is free. API:
    `leaves_intersecting(aabb)` (linear overlap), `edit_aabb_since(previous)` (the diff that drives
    invalidation).
  - **Targeted invalidation.** `edit_aabb_since` = union of every leaf whose `(world_aabb,
    fingerprint)` pair is in the multiset symmetric difference of the two indices. This is uniform
    across edit kinds: **move/offset** unions OLD and NEW boxes (dirties both endpoints), **add** =
    new box, **remove** = old box, **edit-in-place** (resize/recolour/shape-swap) = old ∪ new (the
    fingerprint includes shape + material + offset so a same-box content change is still detected).
    `ChunkResolveCache::invalidate_aabb(aabb, density)` drops exactly the chunks `aabb.covering_chunk_range`
    spans. **Wired in `main::rebuild_geometry`:** build the new leaf index, diff vs the previous
    rebuild's index (kept in a new `previous_leaf_index` field), `invalidate_aabb` on `Some`, else
    `clear()`. **Fallbacks to `clear()`** (documented): the first rebuild (no previous index), a
    **density change** (every chunk's voxel extent changes), and a **region-spanning Part edit** (a
    `DebugClouds` leaf has no localisable AABB — its dirty region is "everywhere"). Tool
    add/remove/move/resize/recolour/shape-swap are all TARGETED.
  - **`resolve_chunk` now uses the per-leaf AABB to skip non-overlapping leaves** (task 3): a leaf
    whose world-AABB doesn't intersect the chunk box is skipped before resolving its producer, so
    resolving one chunk costs ~the leaves touching it, not the whole tree. Proven BIT-IDENTICAL —
    the leaf AABB is the exact span of its voxel centres and `stamp_producer_into_chunk` already
    clips to the chunk box, so a non-intersecting leaf would have been clipped to zero anyway
    (goldens unchanged; this is also why the big-scene lib tests got much faster).
  - **Tests (10 new).** `spatial_index`: `intersects_is_half_open`, `union_ignores_empty`,
    `covering_chunk_range_matches_chunk_ownership`. `scene`: `spatial_index_query_matches_full_walk`
    (index == full `for_each_leaf` + AABB filter, across single/three-tool/`--demo-village`,
    incl. empty + far + whole-scene queries), `edit_aabb_diff_covers_old_and_new` (move unions both
    endpoints; recolour = same box; no-change = empty), `edit_aabb_diff_density_change_is_none`,
    `edit_aabb_diff_part_edit_is_none`. `chunk_cache`:
    `targeted_invalidation_evicts_only_intersecting_chunks` (exactly the intersecting chunks evicted,
    all others stay resident, re-resolve == full fresh resolve), `move_invalidates_chunks_around_both_endpoints`,
    `empty_edit_aabb_evicts_nothing`. **122 lib tests** (112 + 10), goldens green.
  - **Slow S2 test shrunk (58s → ~7s lib wall-time).** `scene_exceeding_old_total_cap_resolves_under_per_chunk_bound`
    was a row of 64 boxes spread 32 blocks apart in X (~500 chunks on one axis, 64 leaves walked
    per chunk → ~54s alone). Replaced with TWO boxes at opposite corners of a 16-block cube: the
    composite is a 17³-block cube = ~20M whole-region voxels (still ≫ the old 6M total cap, same
    coverage intent), but only a ~5³ covering-chunk grid with one tiny box per corner → 0.14s. The
    `resolve_chunk` leaf-skip above shaved the rest.

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
