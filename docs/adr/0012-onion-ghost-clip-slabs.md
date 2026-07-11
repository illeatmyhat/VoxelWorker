# ADR 0012 — Onion skin as ghost-shaded clip slabs on the display passes; retire the volumetric fog subsystem

- **Status:** **Accepted (owner decision 2026-07-11)** — implementation in two slices: H1 (ghost passes live on both
  display paths, fog machinery dark but compiling), H2 (delete the volumetric fog subsystem + retire ADR 0007's
  `gpu_resolve` evaluator, whose last live consumer dies with it).
- **Date:** 2026-07-11
- **Layer:** DISPLAY simplification. **Supersedes the volumetric onion-fog pipeline** (issue #28 per-chunk atlas,
  issue #59 band slabs, ADR 0007's fog resolve, ADR 0011 G5's fog-from-bricks). Governed by
  [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) (GPU is a display shell, never truth) and
  [ADR 0008](0008-voxel-frame-invariant.md) (the slab uniforms carry the recentred-Z frame the band already uses).
  Finishes the "no dense state anywhere" retirement of ADR 0010/0011: the per-chunk fog occupancy tiles are the
  last dense-shaped per-chunk volumes in the runtime.

## Context

Onion skin (issue #12) ghosts the voxels OUTSIDE the current layer band (band ± `onion_depth` layers) so the user
keeps spatial context while scrubbing. Today it is implemented as a **second, parallel display pipeline**: a dense
R8 occupancy tile per resident chunk in the band's Z-slab (`(chunk_extent+2)³` bytes, 1-voxel apron for seam-smooth
trilinear sampling), packed into a 3D atlas, raymarched by `OnionFogRenderer` into a soft accumulating haze.

That design predates the brick field — the fog atlas was the FIRST GPU occupancy sink (ADR 0007 built the GPU
resolver to feed it), and ADR 0011 later observed "the shipped per-chunk R8 fog atlas is already a brick map."
After ADR 0011 the GPU already holds the scene's full occupancy (surface brick records + sculpted atlas + clip-map
pyramid + the band-clip block-occupancy masks). The fog atlas is now a dense re-encoding of information the display
pipeline already has, and its costs are structural, not incidental:

- **Tile bytes scale with the slab's XY footprint.** On an 8000³-voxel scene, ONE chunk-row of slab is 125×125
  chunks ≈ 4.5 GB of tiles — the `MAX_FOG_ATLAS_BYTES` budget (512 MiB) therefore disables onion skin outright on
  exactly the large scenes this codebase now targets.
- **Every band scrub that crosses a chunk row rebuilds + re-uploads the slab.** The fill is O(tiles) since
  `b51b8b5`, but the tiles themselves are dense per-chunk volumes.
- The subsystem carries its own budget caps, byte-parity oracles, demand-driven dirty tracking, slab-range reuse
  logic, and a transient Part-only densify — all maintenance surface for a haze aesthetic.

The owner's framing: *"Isn't it simply two boolean intersections against the selected mesh from plane-to-plane?"*
— correct in substance. Both display paths (brick raymarch and cuboid mesh) already clip to the band via a
per-frame uniform; the onion region is the same geometry under a different clip and a ghost shade.

## Decision

1. **Onion skin = a ghost pass on the EXISTING display pipelines.** Each frame with onion active, the engaged
   display path (brick raymarch, or the cuboid mesh when the raymarch is disengaged) draws a SECOND pass whose
   shader-side clip test selects the onion slabs — recentred-Z in `[onion_z_min, band_z_min) ∪ (band_z_max,
   onion_z_max]`, the SAME frame math `AppCore::onion_fog_params` derives today (floored half, Z-up, depth clamped
   1..8) — and shades hits as a translucent ghost (tint matching the current haze hue). No occupancy is built,
   uploaded, or cached: a band scrub is a pure uniform update, O(1) at any scene size.
2. **Pass ordering + blending:** the ghost pass draws after the solid band pass with depth test ON (read-only,
   no depth write) and alpha blending, so solid geometry occludes the ghost correctly and ghosts never occlude
   anything.
3. **Aesthetic trade, accepted by the owner:** crisp translucent voxels replace the trilinear thickness-weighted
   haze. This is judged BETTER for the actual purpose (layer context while chiseling) and is what makes the
   deletion possible.
4. **Delete the volumetric fog subsystem (H2):** `OnionFogRenderer`'s raymarch/atlas machinery (per-chunk AND the
   shot-only whole-grid mode), `build_per_chunk_fog_occupancy` + `build_per_chunk_fog_occupancy_from_bricks` +
   `FogBrickSource`, the `FogZSlab` band-slab machinery (issue #59), `MAX_FOG_ATLAS_BYTES` / `MAX_FOG_CHUNKS`,
   `fog_brick_field` + its startup seeding, the fog dirty/covering-range tracking in the shell, the transient
   Part-only fog densify, and `FogMode`. Their parity oracles retire with them (the ghost pass is gated by
   display-path goldens instead).
5. **ADR 0007's GPU resolver retires with it.** `try_install_gpu_per_chunk_fog` / `resolve_single_producer_fog_atlas`
   is the LAST live consumer of `gpu_resolve.rs` / `gpu_resolve.wgsl`; with the fog gone, the whole producer-mirror
   evaluator and its parity suite are deleted. This leaves exactly ONE GPU-side producer surface (the brick
   pipeline consuming the CPU-classified boundary set) — the consolidation the 2026-07-11 architecture audit
   recommended. The CPU evaluator remains the sole truth (ADR 0006 unchanged).
6. **Goldens re-baseline:** the onion-fog goldens are replaced by ghost-pass goldens on both display paths
   (raymarch ghost; mesh ghost incl. a loaded-VS-material scene, which ghosts flat translucent — no texture in the
   ghost). `shot --fog wholegrid` dies with the debug mode.

## Alternatives considered

- **Keep the volumetric haze, sample the brick atlas directly** (no fog tiles): removes the duplication but keeps
  a second raymarch pipeline + its goldens alive purely for an aesthetic; the crisp ghost is preferred anyway.
- **Screen-space post-process haze:** wrong tool — needs depth peeling to know onion-region thickness, more
  complexity than the thing it replaces.

## Consequences

- Onion skin works on EVERY scene size (the 512 MiB disable and the atlas-VRAM ceiling vanish); scrubbing is
  instant at any scale.
- Hundreds of MB of transient CPU+GPU fog state per band change → zero.
- The last dense-shaped per-chunk volumes leave the runtime (completes the ADR 0009/0010/0011 retirement, per the
  owner's law: no dense grids anywhere; oracles/goldens excepted).
- The `gpu_resolve` deletion removes the second producer-evaluator implementation — adding a future producer
  (e.g. the reserved Sweep arm) touches ONE evaluator + its parity tests, not two.
- Risk: the ghost pass inherits the display paths' own gaps — a scene that displays nothing (degenerate Part-only)
  ghosts nothing; this matches the fog's current behaviour there (empty region ⇒ no fog) and is accepted.
