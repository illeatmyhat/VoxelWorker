# PROGRESS — VoxelWorker (Rust port)

Autonomous build log. Orchestrator updates this after each milestone. Newest at top.

## Status board

| # | Milestone | Issue | State |
|---|-----------|-------|-------|
| 0 | Repo + scaffolding + dev notes | — | ✅ done |
| 1 | Window + clear + empty egui panel + **headless `shot` binary** | #1 | ✅ done |
| 2 | Voxel core: SDF → instances → flat cubes + orbit cam (5×1×5 cylinder) | #2 | ✅ done |
| 3 | egui params + all shapes + ortho toggle | #3 | ⏳ pending |
| 4 | Shaders: per-voxel slice, then position-based grid overlay | #4 | ⏳ pending |
| 5 | View cube + origin gizmo + 2D slice map | #5 | ⏳ pending |
| 6 | VS folder auto-detect + scan + palette + thumbnails | #6 | ⏳ pending |
| 7 | Block-JSON per-face textures | #7 | ⏳ pending |
| 8 | Polish: `.vox` export, config persistence | #8 | ⏳ pending |

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
