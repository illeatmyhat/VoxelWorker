# ADR 0013 — Per-voxel materials as a sparse R16 cell-key side atlas; the representability gate is deleted

- **Status:** **Accepted (2026-07-13), not yet built** — the design ruling for the "mixed-material mesh cliff"
  (slow-paths item 1). Architecture chapters get their timeless update when the epic lands.
- **Date:** 2026-07-13
- **Layer:** DISPLAY (the ADR 0011 brick-field sink). Governed by
  [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) (GPU is a display shell — authoring truth already
  stores per-cuboid materials; this ADR only teaches the sink to display them) and
  [ADR 0011](0011-gpu-brick-field-display-sink.md) (whose G2 "R8 atlas is occupancy-only, material is
  per-record" limit this supersedes).

## Context

Authoring truth already mixes materials within a block: every microblock cuboid carries a **cell key**
(`u16` = block-palette id + on-face-grid overlay flag), and Vintage Story chiseling itself produces
multi-material blocks. Only the brick display sink cannot show them: its atlas is occupancy-only and
material/overlay live per-record and per-scene, so `brick_representable_overlay` rejects any scene
containing (a) a block that mixes cell keys internally, or (b) blocks that disagree on the scene-wide
overlay uniform — dumping the **whole scene** onto the cuboid-mesh path (~92k draws/frame at 8000³,
crisp ghost instead of haze, the parked top-cap-seam defect).

Ruling (grill 2026-07-13): a **mixed-material block is first-class** — no authoring rule may resolve the
mix away, and no encoding may cap it (a cap re-creates the cliff at the cap).

The countervailing force is the sculpt VRAM plan: density is bounded 1..=64 precisely so occupancy can go
bit-per-voxel (8× cut; ~1M chiseled blocks: 4 GB → 512 MB). Per-voxel material data must not drag every
brick back to byte-per-voxel.

## Decision

1. **Sparse side atlas.** The occupancy atlas stays on its bit-per-voxel trajectory. Only **mixed**
   bricks get a slot in a second, separately-pooled 3D material atlas; uniform sculpted bricks keep
   their cell identity on the record. Per-voxel material cost is paid only where mixing exists — the
   boundary-residency philosophy applied to materials.
2. **Texel = the cell key, verbatim (R16Uint).** The `u16` cell key (palette id + overlay bit) is stored
   raw. No indirection tables, no per-brick palettes, no material-count caps at any scope. Air texels are
   don't-care (occupancy gates the sample).
3. **Both gate arms die.** Because the texel carries the overlay bit and uniform-brick records gain an
   overlay bit beside their material, `brick_representable_overlay` is **deleted**: the brick path
   engages unconditionally on gpu builds. The mesh path remains only for non-gpu builds and debug-face.
4. **CPU mirror follows the single-owner tile law.** The mirror stores one cell-key tile (`edge³` u16s)
   per mixed brick beside its occupancy tile — same emission scatter, same ownership-by-move, same
   parity-oracle materialisation. Uniform bricks store no material tile.
5. **Pool mechanics mirror the occupancy atlas.** Own cube-root tile grid, own free-list, own (much
   smaller) high-water mark; incremental edits never grow a pool — growth routes wholesale. A brick
   flipping uniform↔mixed under an edit allocates/frees its material slot like any slot churn.
6. **Correctness bar = CPU-march parity, not mesh-path mimicry.** The CPU reference march samples the
   material tile exactly as the shader does; gpu_parity locks shader == reference, and a mixed-scene
   golden locks appearance. The two renderers stay deliberately divergent otherwise. UV law is
   block-face-anchored: a voxel's material samples *its* texture at the position the voxel covers on the
   block face — the existing UV math, only the id source changes.

## Considered options

- **Occupancy byte becomes the material index** (one R8 atlas, 0 = air): zero VRAM growth versus today
  and the simplest shader, but it permanently forfeits the 8× bit-packing cut for *every* sculpted brick
  and caps a scene at ~255 materials. Rejected: pays the per-voxel price everywhere to serve the mixed
  minority.
- **R8 texel + scene-wide active-material table**: half the VRAM of R16 on mixed bricks, but adds an
  indirection table to maintain incrementally and a 255-distinct-materials-per-scene bound that a large
  scene could plausibly hit. Rejected: a cap plus machinery, to halve the cost of a sparse minority.
- **4-bit texel + per-record ≤16-entry palette**: quarter the VRAM, but a 17-material block becomes
  non-representable again — the cliff returns — and records grow 32 bytes while pack/unpack seams
  multiply. Rejected: contradicts the first-class (uncapped) ruling outright.

## Consequences

- VRAM: a mixed brick pays `2 × edge³` bytes (d=16: 8 KB) *in addition to* its occupancy slot; uniform
  bricks pay nothing new. The untested VRAM-ceiling behaviour (slow-paths item 3) now covers two pools —
  the planned graceful-degradation test must exercise both.
- The parked mesh-path ghost top-cap seam (#17) becomes unreachable outside non-gpu builds, as does the
  1–2-frame blank on the non-representable display handover; both retire with the gate.
- The GPU bit-per-voxel occupancy switch (the remaining half of the 8× cut; CPU tiles already bit-packed)
  is **deliberately a separate later slice**, not part of this epic — different texture, independently
  gateable. The material pool must not assume the occupancy atlas stays R8.
- Glossary: **mixed-material block**, **mixed brick**, **cell key** are now canonical (`CONTEXT.md`).
