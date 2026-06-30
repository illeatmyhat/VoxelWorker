# Context glossary

Canonical terms for VoxelWorker. This file is a **glossary only** — no implementation detail,
no decisions (those live in `docs/adr/`). Define a term here the first time an ambiguity bites.

## Fog residency (the per-chunk onion fog)

- **Chunk** — a fixed cube tile of voxel space (`CHUNK_BLOCKS` blocks per axis). Space is sliced
  into chunks so a scene too large for one 3D texture can still be stored/rendered tile-by-tile.

- **Apron** — the 1-voxel border each chunk tile carries, copied from its neighbours, so trilinear
  fog sampling blends smoothly across chunk seams instead of showing a hard line at every tile edge.
  A tile is stored at `(extent + 2)³`; apron index `0` is the neighbour voxel at chunk-local `−1`.

- **Resident / non-empty chunk set** — the chunks the **CPU** fog path stores: a chunk is included
  **iff it has ≥1 occupied voxel in its interior** (`[0, extent)`, apron excluded). Empty space
  stores nothing; a ray through an unstored chunk reads "no fog."

- **Covering chunk set** — the chunks the **GPU** fog path enumerates: **every** chunk whose box
  overlaps the producer's bounding AABB (`Scene::covering_chunk_range`). Chosen because it needs no
  occupancy knowledge (no densify). It is a **superset** of the non-empty set.

- **Sliver** — the rendering difference between the two sets above. A covering chunk can have an
  empty interior but a non-zero apron (its border touches the shape's surface). The non-empty set
  drops such a chunk; the covering set keeps it, drawing a 1-voxel band of extra fog at the surface.
  Eliminated by zeroing interior-empty tiles' aprons (ADR 0007 residency option C′).

## Block vs voxel

- **Block** — the coarse placement + **material/texture unit**: one block-texture per face (e.g. the
  Vintage Story andesite-ashlar brick). A block stays coarse until chiseled; only then is it
  subdivided into voxels. Texture/material is addressed per **block**, so a brick spans a whole
  block face (`voxels_per_block` voxels across), not a single voxel.

- **Voxel** — the **chisel granularity**: `voxels_per_block` per axis within a block (document-level
  density, `docs/adr/0003`; VS = 16³). Geometry/occupancy is addressed per voxel; a chiseled block's
  surface steps on the voxel lattice while its faces still carry the per-block texture.

## Boundary residency (the two-layer chunk)

The chunk representation a boundary-aware store keeps, modelled on Vintage Story's
own split (un-chiseled blocks live in the bulk block array; chiseled blocks are
separate microblock entities). A solid interior is never voxelized.

- **Coarse layer** — a per-**block** block-ID grid over the chunk (one id per block,
  palette/RLE-compressed). A solid region — even a multi-material one — lives here
  as block ids only, with **no voxel data**. The "coarse until chiseled" storage
  fact: an un-chiseled block carries texture/material, not occupancy voxels.

- **Microblock layer** — a **sparse** map of the chunk's **boundary blocks** (the
  blocks the producer surface passes through, i.e. chiseled blocks) to their
  sub-block voxel geometry, stored **already decomposed to cuboids** (not a dense
  16³ grid). Only surface blocks appear here.

- **Block classification** — each block of a chunk is **air**, **coarse-solid**
  (in the coarse layer), or **boundary** (in the microblock layer). Decided by a
  conservative per-op field-interval bound over the block cell (see the op-stack
  ADR / ADR 0009 §3–§4): all-outside ⇒ air, all-inside ⇒ coarse-solid, straddling
  (or unboundable) ⇒ boundary. The bound is conservative so classification is
  **occupancy-identical to brute force** on the exact seam (exporter/query/golden);
  an op that cannot bound a cell falls back to per-voxel evaluation (still exact).

- **Seam solidity** — per-boundary-block, per-face solidity flags (VS
  `sideAlmostSolid`/`sidecenterSolid`) used to **cull faces across a chunk/block
  seam** without expanding a neighbour's voxels. The coarse-vs-microblock analogue
  of the dense-fog **apron**.

## Authoring truth

- **Operation stack** — the ordered list of authoring operations for a part's geometry: parametric
  SDF primitives, boolean CSG ops (authored from 2D sketches), and sparse hand-sculpted voxel
  deltas. **This is the single source of truth.** (See `docs/adr/0006`, and the op-stack ADR.)

- **Resolved grid** — voxel occupancy obtained by evaluating the operation stack
  (`apply(overlay, evaluate(tree))`). A **derived cache, never truth**; materialized lazily and
  region-scoped, never kept resident as a whole-scene dense buffer. Any code that materializes it
  densely must justify why a query against the operation stack will not do.

## Authoring frame

- **Recentre** — the integer voxel offset a producer's grid was placed at. A placed Tool is
  recentred onto the origin by `floor(dim/2)`; a corner-anchored Part (e.g. `DebugClouds`) has
  recentre `[0,0,0]`. **Carried on the grid, never re-derived** (ADR 0008).
