# CPU frame profiling (Tracy)

VoxelWorker is instrumented with the [`profiling`](https://crates.io/crates/profiling)
facade. By default the facade compiles to **no-ops** — a normal `cargo build` /
`cargo run` pulls in NO Tracy code, no `tracy-client`, and no C/C++ toolchain, and
has zero runtime overhead. Profiling is opt-in behind the `tracy` cargo feature.

## How to build / run

```
cargo run --features tracy
```

Then launch the external **Tracy profiler app** and connect to the running
process (it auto-discovers localhost clients, or use *Connect*). You will see the
per-frame timeline, a flamegraph of the CPU zones, and frame marks.

## Exact matching Tracy app version

Tracy's wire protocol is version-locked: a mismatched profiler app **silently
fails to connect**, so the versions must line up exactly.

- `profiling = "1"` resolves to `profiling 1.0.18`, whose `profile-with-tracy`
  backend requires `tracy-client = "0.18"`.
- We pin `tracy-client = "=0.18.4"` (exact) in `Cargo.toml`. That maps, via
  `tracy-client-sys 0.28.0`, to the **Tracy profiler app v0.13.1**.

**Use Tracy profiler app v0.13.1.**

Download / build it from the releases page of the Tracy repo:
<https://github.com/wolfpld/tracy/releases/tag/v0.13.1>
(Windows: grab the prebuilt `tracy-profiler` from the v0.13.1 release assets, or
build `profiler/` from the v0.13.1 source tag.)

If you bump `tracy-client`, re-check the version table at
<https://github.com/nagisa/rust_tracy_client> and update the app version above —
the crate's SemVer does NOT track protocol breaks.

## What is instrumented (CPU only)

One `profiling::finish_frame!()` is emitted per rendered frame (end of the winit
redraw handler in `src/main.rs`). Zones (`profiling::scope!`):

| Scope name | Location |
| --- | --- |
| `render` | `WindowedState::render` (`src/main.rs`) — the whole frame |
| `egui_frame` | `run_egui_frame` call (`src/main.rs`) — UI build |
| `render_submit` | `render_frame` + `present()` (`src/main.rs`) — GPU submit |
| `rebuild_geometry` | `WindowedState::rebuild_geometry` (`src/main.rs`) |
| `app_core_rebuild` | `AppCore::rebuild` (`src/app_core.rs`) |
| `invalidate_aabb` / `invalidate_clear` | the invalidation arm in `AppCore::rebuild` (incremental vs full clear) |
| `resolve_region` | `store.resolve_region(...)` (the SDF resolve) in `AppCore::rebuild` |
| `resident_render_chunks` | `store.resident_render_chunks(...)` in `AppCore::rebuild` |
| `cuboid_mesh_build` | `CuboidMeshRenderer::new_from_chunks` (`src/cuboid_mesh.rs`) |
| `sdf_resolve` | `SdfShape::resolve` (`src/voxel.rs`) — per-producer SDF sampling |
| `sketch_resolve` | `SketchSolid::resolve` (`src/sketch.rs`) — per-producer sketch sampling |

The flamegraph distinguishes per-frame UI (`egui_frame`) from geometry rebuilds
(`rebuild_geometry` → `app_core_rebuild` → resolve / mesh), and shows which
invalidation path was taken and its cost.

## Not yet wired: GPU zones (TODO)

This is **CPU-scope profiling only**. GPU timestamp zones (Tracy GpuContext /
wgpu timestamp queries) are a documented follow-up and are NOT instrumented yet.
