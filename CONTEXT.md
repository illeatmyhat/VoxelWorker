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

## Authoring frame

- **Recentre** — the integer voxel offset a producer's grid was placed at. A placed Tool is
  recentred onto the origin by `floor(dim/2)`; a corner-anchored Part (e.g. `DebugClouds`) has
  recentre `[0,0,0]`. **Carried on the grid, never re-derived** (ADR 0008).
