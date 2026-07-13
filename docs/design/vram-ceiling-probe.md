# VRAM-ceiling probe results (2026-07-13)

**Provenance:** the "sculpt VRAM-ceiling graceful-degradation" pre-work (user-present session,
RTX 4090 24 GB, driver 591.86, wgpu 29, 95 GB system RAM). A standalone harness replicated the
app's exact device setup (`gpu.rs`: HighPerformance adapter, `Limits::default()` +
adapter-sized buffer limits) and probed the sculpted-atlas allocation path
(`upload_brick_atlas`-shaped D3 R8 textures) to failure on both live backends. Dated snapshot;
the timeless conclusions belong to the sculpt design when it lands.

## The two ceilings

1. **Dimension ceiling (self-imposed).** The app requests `Limits::default()` →
   `max_texture_dimension_3d = 2048`, while the adapter supports **16384**. The atlas is a cube
   (`atlas_dim = bricks_per_axis × edge`), so at density 16 the cap is 128³ ≈ **2.1M sculpted
   bricks ≈ 8 GB** — and `create_texture` past it is a validation error. Raising the requested
   limit moves this ceiling far past VRAM (16384³ = 4 TB), i.e. the *real* ceiling becomes
   memory, not validation.
2. **Memory ceiling (backend-divergent, the dangerous one).** Repeated 2 GiB R8 slabs, held
   live, fully written:

   | | Vulkan (RTX 4090) | DX12 (same GPU) |
   | --- | --- | --- |
   | Oversubscription | **None** — allocation fails at the dedicated-VRAM edge (22 GiB held, 23.7/24 GB used) | **Silent, 3× dedicated** — dedicated pins at ~21.8 GB while held total climbs 22→62 GiB paged into shared system RAM |
   | Perf on approach | Flat ~1.3 s/slab; one 3 s outlier at the edge | Degrades 1.3 s → ~2.7 s/slab while paging (display-frame cost would degrade likewise) |
   | Failure mode | **Clean scoped `OutOfMemory` error**; `create_texture` fails instantly | `ID3D12Device::CreateHeap` → `DXGI_ERROR_DEVICE_REMOVED` (0x887A0005) → **DEVICE LOST** at ~64 GiB |
   | Error scopes | Catch the OOM | **Catch nothing** — the OOM scope stayed empty; failure surfaces as device loss |
   | Aftermath | Device fully usable (16 MiB alloc + write OK) | Device dead — every subsequent operation invalid; recovery = recreate the whole `GpuContext` (device, pipelines, resources) |
   | System | No display disturbance | No display disturbance; other processes' devices unaffected |

## What the app does today

No error scopes, no `on_uncaptured_error` handler, no device-lost callback anywhere in the
runtime. Both ceilings route into wgpu's *"Handling wgpu errors as fatal by default"* →
**process panic** (verified with an unscoped over-limit create, exit 101). On DX12 the panic
would come *after* a stretch of silently degrading, paging frames.

## Conclusions for the sculpt design

- **A reactive catch is not a strategy.** On DX12 the first catchable signal is device loss —
  too late. The guard must be **an app-side byte budget checked BEFORE allocating**: the app
  knows every atlas's exact size (`atlas_dim³`, and ADR 0013's material pool likewise), so a
  running GPU-bytes ledger against a budget is cheap and exact. (The deleted fog subsystem's
  `MAX_FOG_ATLAS_BYTES` was this pattern; the brick pipeline currently has no equivalent.)
- **Budget size is not queryable through wgpu 29** (no VRAM size / memory-budget API), so the
  budget is configuration with a conservative default; the probe machine's law of "dedicated
  VRAM minus working headroom" cannot be auto-derived portably.
- **Error scopes + a device-lost callback are still worth installing**: scopes make Vulkan's
  ceiling fully graceful, and the lost-callback turns DX12's worst case from undefined
  behaviour into a reportable, deliberate shutdown (or a future device-rebuild path).
- **The 2048 dimension cap is not a safety net.** It caps the atlas at 8 GB, which already
  exceeds the 8 GB RTX 5070 Laptop GPU in this very machine's pool — DX12 would oversubscribe
  and eventually device-remove well under the cap. Whether to raise
  `max_texture_dimension_3d` toward the adapter limit is a sculpt-design decision that only
  makes sense together with the byte budget.
- **What degradation *does* (refuse the edit? stale display? coarser LOD-only residency?) is
  the open design question** — deliberately deferred to the sculpt grill, alongside ADR 0013's
  note that the material pool shares whatever policy is chosen.

Probe source: session scratchpad `vram-probe/` (standalone crate, not part of the repo build);
re-run recipe: `vram-probe [validate|pressure|panic]`, `WGPU_BACKEND=dx12` for the DX12 arm.
