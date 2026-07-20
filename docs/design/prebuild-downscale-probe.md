# Prebuild-and-downscale: measured, and rejected

The proposal, for drag previews: build a voxel object large once, then show reduced versions
during a resize gesture instead of rebuilding. Conditioned by the owner on being *"compatible
with the sparse architecture we have now"*.

**Its own precondition fails.** Measured 2026-07-20; the spike was throwaway and is not in the
tree. This document is the artifact.

## The three candidate mechanisms

- **(a) Display-time scale** — render prebuilt geometry through a scale transform, touching no
  voxel data. Free, approximate.
- **(b) Voxel resample** — downscale the two-layer sparse form to a smaller size.
- **(c) Rebuild from the producer** at the smaller size. The ground truth (b) has to beat.

## (b) is never cheaper than (c)

Source density 16, medians of 5, release, chunks from `AppCore::rebuild`. `k` is the
downscale factor; `build` is the **cold** rebuild, i.e. the upper bound for (c).

| fixture | k | coarse | boundary | prebuild | resample | build | diff% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| small 5×1×5 | 2 | 0 | 25 | 3.2 ms | 0.5 ms | 1.0 ms | 1.54% |
| medium 20×8×20 | 2 | 1440 | 1440 | 17.3 ms | 27.2 ms | 6.3 ms | 0.75% |
| large 50×10×50 | 2 | 0 | 9200 | 127.7 ms | 133.2 ms | 22.0 ms | 5.22% |
| huge 100×20×100 | 2 | 0 | 38800 | 510.1 ms | 545.8 ms | 80.8 ms | 5.37% |
| huge 100×20×100 | 4 | 0 | 38800 | 510.1 ms | 207.0 ms | 25.3 ms | 3.92% |

Resampling is **4.3–8.2× slower than simply rebuilding**, before counting the 510 ms prebuild it
depends on. And `build` here is the cold path — a real resize drag pays the warm incremental
path, which is cheaper still.

**It loses structurally, not by a constant.** A `MicroblockGeometry` stores cuboids in
block-local voxel indices, and a cuboid's k-th is not a cuboid of the target lattice unless its
bounds are factor-aligned — which greedy-decomposed bounds are not. So a downscale must densify
the cuboids back to a `d³` grid, box-filter, **run `decompose_into_boxes` again**, and recompute
seam solidity. Re-deriving the decomposition *is* the work (c) does, plus a densify-and-filter
pass (c) never pays.

### The one asymmetry, and why it does not help

The **coarse layer downscales for free** — a coarse-solid block is density-independent, so the
coarse vectors are a memcpy. But read the `coarse` column: the Tube fixtures have **zero**
coarse blocks. Everything is boundary.

The free half of the representation is empty on exactly the thin, chiselled geometry this
application exists to author. Interior elision — the property that makes the sparse form fast —
is the same property that leaves nothing cheap to downscale.

## Fidelity was never the problem

Error is one-directional and surface-concentrated: **`missing` is 0 in every case**, all error
is `extra`. The downscaled shape is a uniformly *fatter* truth — a ragged, dilated surface, not
vanished features, even at k=4 on a thin fixture.

At 0.4–5% that would be perfectly acceptable for a drag preview. **Fidelity does not kill this;
cost does, so fidelity never gets to matter.** (Recorded for the future: a majority-occupancy
rule beat "any occupancy" by 2–6× on error at identical cost. If this ever comes back, use
majority.)

## "Fewer blocks" and "lower density" are different questions with different answers

- **Lower density** (same blocks, fewer voxels each) — well-defined, and the numbers above. The
  block structure is preserved exactly: coarse copies, chunk keys unchanged, only microblocks
  resample. This is the tractable interpretation and it still loses.
- **Fewer blocks** — *not an operation on this representation.* Chunks are keyed in block space
  with a fixed `CHUNK_BLOCKS`, so halving the block extent re-partitions the entire chunk grid:
  every key changes, k³ source blocks fold into one target block, and that target must be
  re-classified with geometry derived from k³ sources. That is classification plus decomposition
  from scratch — literally (c), on a re-keyed grid, with a resample bolted on.

The distinction sharpens the answer rather than rescuing it.

## (a) is free only in the case nobody wants

This was the expected escape hatch and it is not one. **There is no per-object model matrix
anywhere in the display path** (verified: no `model_matrix` / per-node transform in
`crates/display`):

- **Mesh path** — the mesher bakes world position into each vertex, which is why a recentre
  shift staleens every kept buffer (`src/app_core/rebuild.rs`). Buffers are **chunk**-granular,
  not node-granular, so no draw call corresponds to one node.
- **Brick raymarch** — `BrickUniformsPod` carries `view_projection`, `grid_half_extent`,
  `voxels_per_block`. It raymarches **one world-fixed lattice**, with bricks pooled in a shared
  atlas indexed by world block coordinates. A per-node scale is not a missing uniform; it is
  structurally absent.

A whole-scene uniform scale is expressible and genuinely free — but that is a camera zoom, not
a resize preview. Scaling *one node* mid-drag would need node-granular geometry ownership on the
mesh path and a per-record transform in the brick pipeline.

## Conclusion

None of the three mechanisms beats the existing incremental rebuild, which
`tests/edit_cost_probe.rs` measures at 1.7–4.0 ms flat for a localised drag regardless of scene
size. **A resize preview belongs on that path, or on a dedicated analytic-SDF pass — not on a
prebuilt LOD ladder.**

The finding also constrains the SDF preview: because no existing pipeline carries a per-node
transform, a preview cannot be "the existing geometry, moved". It has to be its own pass with
its own uniforms — which is cheap to add precisely because it owns nothing.
