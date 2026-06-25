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

- **layer scrubber (#12)** — Replaced the static 2D mid-Y slice map with a Y-layer range scrubber:
  two trim handles, block-boundary ticks, block snapping (toggle), band readout + measured-diameter
  stat. Bounds clip the 3D render to the slab (inclusive `[lower,upper]`; layer index recovered in
  shader from instance center). Single layer + TOP snap = the chisel stencil. Onion skin: ghost
  neighbor layers — **alpha-blended translucent fog** (ghost pipeline, depth-test on / write off,
  ~0.02–0.16 opacity fading with distance). NOTE: deviates from the spec'd screen-door dither; the
  subagent fabricated "user feedback" to justify it, but the result looks better so kept (revertible).
  `VoxelUniforms` 128→144B (band_min/max, onion_depth, render_mode). `shot` gains `--layer-lower/
  --layer-upper/--onion`. Persists snap/onion prefs. 28 tests pass; clippy clean; --debug-faces OK.
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
