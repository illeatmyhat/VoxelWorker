# Coordinate limits: how far from the origin can you build (2026-07-24)

**Provenance:** the wide-baseline precision work (commit `a06d215`). That commit fixed the
*rendering* failures that appear when the camera and geometry sit far apart but WITHIN the
representable envelope. This note records the envelope itself — the three nested ceilings that
bound where geometry can live at all — and the two ways to move them out if the product ever has
to. The fix is dated; the envelope and the scaling options are the durable content.

## The envelope: three nested ceilings

The document can name a coordinate long after the renderer can draw it. The binding limits, from
tightest to loosest:

1. **~±1,048,576 blocks, ABSOLUTE — the brick record key.** The brick path packs the *absolute*
   block coordinate into a sortable key: `+2^20` bias into three 21-bit lanes filling a `u64`
   (`cpu_pack_key_split` in `crates/display/src/brick/cpu_march.rs`, mirroring
   `substrate::spatial::lattice_key::pack_lattice_key` and the WGSL). `BIAS = 1 << 20`, so a block
   coordinate outside `[-2^20, 2^20)` overflows its lane. Substrate's `pack_lattice_key` *panics*
   on overflow (`assert!` in `lattice_key.rs`); the display mirror wraps, and the key silently
   aliases — **the object stops rendering on the brick path, even alone with the recentre sitting
   on it.** This is the wall hit in practice at 10M blocks.

2. **~±2^24 voxels from the recentre — f32 loses the integers.** Beyond ~16.7M voxels from the
   floating render origin, f32 can no longer represent adjacent integers, so the render frame
   cannot name an individual voxel (ULP is ~8 voxels at 5M blocks, density 16). At density 16 that
   is ≈±1M blocks *measured from the recentre*, not from the world origin. It bites when the
   **composite SPAN** exceeds ~2M blocks: the floating origin is the composite midpoint, and a
   midpoint cannot be near both ends of a span wider than twice its own reach.

3. **i64 document coordinates — effectively unlimited.** `offset_voxels: [i64; 3]`
   (`crates/document`, ADR 0008 carried frame) reaches ~9×10^18. The document is never the
   constraint; the display representations are.

Ceiling 1 is absolute-position bound, ceiling 2 is span-and-distance-from-recentre bound. A single
object 5M blocks out dies on (1) first. A pair of objects 3M blocks apart, each near the origin,
dies on (2).

## What the wide-baseline fix did (inside the envelope)

Commit `a06d215` fixed the *rendering* breakdown that occurs at large but representable baselines;
it did not move any ceiling. Two parts. First, a per-mode ray frame (`camera::SceneMatrices`,
`crates/camera/src/projection.rs`): perspective gets an eye-anchored `ray_unprojection` with a
camera-sized unprojection bracket, so the per-fragment `/w` divide stays precise instead of
unprojecting a distant near/far slab into two huge nearly-equal points; orthographic is left in the
plain view-projection frame to preserve CPU/GPU bit-parity. Second, grid plane-origin coarse-cell
anchoring, so the infinite grid stops shimmering at distance. Details live in the commit message
and the cited code.

## Interim guard: reject rather than vanish

Being implemented in parallel: an authoring-time bound of `COORDINATE_LIMIT_BLOCKS = 1_000_000`
blocks per axis (`crates/document`), rejecting placements and geometry that cross it, surfaced as
an inspector warning. This converts ceiling 1's failure mode — silent disappearance on the brick
path — into an explicit, legible error at the moment of authoring. It is a guard rail, not a
scaling change; the envelope is unchanged.

## If the envelope must grow: two options

Recorded here so they are not re-derived under pressure. Neither is built; each targets a different
ceiling.

### Option A — field-local brick keys (cheap; fixes the single far object)

Key the brick records **relative to the field's minimum block** instead of absolute zero, carrying
the i64 base alongside the records (the ADR 0008 carried-frame pattern). The 21-bit lanes then span
the field's own extent, not its distance from the world origin, so a lone field placed at any i64
position renders as long as **one field spans < ~2M blocks**. The mesh path is already
recentre-relative, so it follows for free. This lifts ceiling 1 to i64 range for the
single-object case; it does nothing for ceiling 2, because a wide *composite* still exceeds f32's
reach from any single recentre.

### Option B — per-chunk camera-relative render frames (the full fix; needed for wide composites)

Rebase each chunk's origin against the eye in exact i64 on the CPU (`chunk_origin − eye`), hand the
GPU only the small f32 residual, and render each chunk in its own camera-relative frame. This
removes ceiling 2 entirely: precision is bounded by chunk size, not by distance from any global
origin, so a composite of any span stays sharp. It also forces **sparse covering-set handling** in
the resolve — a 10M-block composite covers millions of mostly-empty chunks, and the resolve cannot
afford to walk them densely. This is the larger change and is only warranted once multi-object
composites spanning >~2M blocks are a real requirement.

## Summary

| ceiling | limit | bound by | failure |
| --- | --- | --- | --- |
| brick key | ~±1M blocks absolute | key lane overflow (`+2^20` bias, 21-bit) | object stops rendering (silent on brick path) |
| f32 recentre | ~±2^24 voxels from recentre; span < ~2M blocks | f32 integer precision | voxels unnameable, raymarch + grid melt |
| document | ~9×10^18 | i64 | never the problem |

Options: A (field-local keys) lifts the brick ceiling for a single object cheaply; B (per-chunk
camera-relative frames + sparse covering set) removes the f32 ceiling for wide composites at higher
cost. The interim guard makes the current envelope's edge an explicit error rather than a
disappearance.
