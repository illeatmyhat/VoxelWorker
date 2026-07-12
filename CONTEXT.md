# Context glossary

Canonical terms for VoxelWorker. This file is a **glossary only** — no implementation detail,
no decisions (those live in `docs/adr/`). Define a term here the first time an ambiguity bites.

## Chunks

- **Chunk** — a fixed cube tile of voxel space (`CHUNK_BLOCKS` blocks per axis). Space is sliced
  into chunks so residency, invalidation, meshing, and display caching can all operate
  tile-by-tile at any scene size.

- **Covering chunk set** — **every** chunk whose box overlaps the scene's bounding AABB
  (`Scene::covering_chunk_range`), enumerated without any occupancy knowledge. The unit of
  wholesale evaluation: a resolve classifies each covering chunk's blocks and empty chunks
  simply store nothing.

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
  seam** without expanding a neighbour's voxels: a tiny pre-digested summary of
  exactly what a neighbour is entitled to know.

## GPU brick-field display sink (the cached-brick raymarch)

The GPU display derivation that raymarches a **cached** copy of the boundary set instead of the
op-stack field (see `docs/adr/0011`; generalizes the ADR 0007 fog atlas).

- **Brick** — **one block's** cube of voxels cached in one **atlas slot** (a slice of an R8 3D
  texture pool). The granule is denominated in **blocks**, never a fixed voxel count: brick edge =
  `voxels_per_block`, whatever the document's density is (the units law — density is fineness
  only). A boundary block's voxels are packed into a **sculpted brick**; a coarse-solid block is a
  **coarse brick** (a solid-block marker record, no atlas slot, no per-voxel data). Empty space
  gets no brick. "Raymarch the cache, not the field": per-frame cost is independent of op-stack
  complexity because rays sample cached bricks, never the analytic SDF.

- **Clip-map occupancy pyramid** — coarser "any-brick-inside" occupancy levels above the fine brick
  set (e.g. cells of 8 blocks, then 64 blocks). A ray's **hierarchical DDA** jumps straight to the
  exit of the coarsest EMPTY level covering its position — one big stride through empty space —
  descending to per-block brick work only where the finest level is occupied. World-fixed levels (a
  min-mip of the brick set) are distinct from a **geometry clip-map** (nested camera-centred grids,
  Losasso–Hoppe); the latter is the residency-ring variant for the off-screen case.

- **Render broadphase vs edit broadphase** — two different "what's near this box?" problems, never
  one structure. The **render broadphase** answers per-frame ray queries (the clip-map pyramid +
  resident-brick lookup). The **edit broadphase** answers per-edit dirty queries — which producers
  overlap a region — via a single BVH of producer bounds (stateless, rebuilt per edit). One query,
  one structure: the edit broadphase is shared by the wholesale build and incremental re-evaluation.

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
