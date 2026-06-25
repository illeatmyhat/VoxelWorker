# PROGRESS — VoxelWorker (Rust port)

Autonomous build log. Orchestrator updates this after each milestone. Newest at top.

## Status board

| # | Milestone | Issue | State |
|---|-----------|-------|-------|
| 0 | Repo + scaffolding + dev notes | — | ✅ done |
| 1 | Window + clear + empty egui panel + **headless `shot` binary** | #1 | ⏳ pending |
| 2 | Voxel core: SDF → instances → flat cubes + orbit cam (5×1×5 cylinder) | #2 | ⏳ pending |
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

## Log

- **m0** — Scaffolding: `.gitignore`, `docs/DEV_NOTES.md` (verified API sigs), this file. Repo
  created and pushed. Issues #1–#8 + tracking issue opened.
