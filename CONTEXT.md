# Context glossary

Canonical terms for VoxelWorker. This file is a **glossary only** — no implementation detail,
no decisions (those live in `docs/adr/`). Define a term here the first time an ambiguity bites.

## Substrate vs domain

- **Substrate** — the objects of computer science and pure math: components describable
  entirely in textbook CS/math vocabulary (BVH, AABB, bit cube, interval list, min-mip pyramid,
  rational, free-list, key codec, supersede protocol), parameterized by plain numbers/generics,
  never by domain types. They live apart from the domain (`docs/adr/0014`) so they can be
  identified, read, and performance-reasoned in isolation.

- **Domain** — the objects of VoxelWorker's subject matter (scene, producer, chunk, brick,
  chisel, resolve). Domain code *uses* substrate components through adapter seams; substrate
  never names a domain concept.

## Chunks

- **Chunk** — a fixed cube tile of voxel space (`CHUNK_BLOCKS` blocks per axis). Space is sliced
  into chunks so residency, invalidation, meshing, and display caching can all operate
  tile-by-tile at any scene size.

- **Covering chunk set** — **every** chunk whose box overlaps the scene's bounding AABB
  (`Scene::covering_chunk_range`), enumerated without any occupancy knowledge. The unit of
  wholesale evaluation: a resolve classifies each covering chunk's blocks and empty chunks
  simply store nothing.

## Block vs voxel

- **Block** — the coarse placement + **texture-addressing granule**: one block-texture spans a
  whole block face (`voxels_per_block` voxels across, e.g. the Vintage Story andesite-ashlar
  brick), never a single voxel. A block stays coarse until chiseled; only then is it subdivided
  into voxels. Texture *addressing* is per block, but material *identity* is per voxel cell once
  chiseled — see **mixed-material block**.

- **Mixed-material block** — a chiseled block whose microblocks carry more than one material.
  **First-class**, not an authoring accident: Vintage Story chiseling mixes materials within one
  block, and sculpt painting will create them deliberately. Every display sink must render them
  faithfully; no authoring rule may resolve the mix away.

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

- **Brick** — **one block's** cube of voxels cached in one **atlas slot** (a slice of a 3D
  texture pool). The granule is denominated in **blocks**, never a fixed voxel count: brick edge =
  `voxels_per_block`, whatever the document's density is (the units law — density is fineness
  only). A boundary block's voxels are packed into a **sculpted brick**; a coarse-solid block is a
  **coarse brick** (a solid-block marker record, no atlas slot, no per-voxel data). Empty space
  gets no brick. "Raymarch the cache, not the field": per-frame cost is independent of op-stack
  complexity because rays sample cached bricks, never the analytic SDF.

- **Mixed brick** — the sculpted brick of a **mixed-material block**. Besides its occupancy slot
  it holds a slot in a second, **sparse material atlas** carrying one **cell key** per voxel;
  a uniform sculpted brick carries its single cell identity on its record alone and pays no
  per-voxel material storage. Per-voxel cost is paid only where mixing exists (`docs/adr/0013`).

- **Cell key** — the per-voxel-cell display identity: the block-palette id together with the
  on-face-grid overlay flag, packed as one value. Carried per microblock cuboid in the boundary
  set, and per voxel in a mixed brick's material-atlas slot.

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

## Composition

- **Part** — the fundamental assembly object: a **container** that composes primitives,
  tools, operators, and nested parts into one body under the ordered fold. A part is a
  sealed composition scope — its internal booleans are spent inside it. A **one-off part**
  is authored in place (the scene-tree grouping node); a **reusable part** is a definition
  placed by instances. **The scene root is itself a concrete part** (the Fusion 360 root-
  component model): primitives placed "at root" are ingredients of the root part, and
  anything that applies to a part — per-part display state included — applies to the root
  part too. Primitives and tools are never assembly citizens on their own — they are a
  part's ingredients.

- **Ordered fold** — the composition semantics (`docs/adr/0017`): a scope's children
  evaluate in document order, each folding into the accumulated result under its combine
  operation (union / subtract / intersect). A boolean affects everything accumulated
  before it; **placement order — never operand selection — decides what it touches**.
  Geometry is protected from a cut by being placed after it, or in a sibling scope.

- **Composition scope** — the boundary an ordered fold runs inside: a Group or a
  definition body. Children compose into one body within the scope; a boolean inside a
  scope can never affect geometry outside it.

- **Sealed part** — the default definition behaviour: the definition pre-composes its
  children into one finished body, and an instance places that body under a single
  combine operation. Internal booleans are fully spent inside the definition.

- **Fixture** — a definition flagged so its children **splice into the hosting scope's
  ordered fold at the instance's position** instead of pre-composing. How a window both
  cuts its opening and adds its frame with one placement. The "host" is positional —
  whatever accumulated before the instance in its scope — never a stored reference; a
  fixture moved into a different wall's scope cuts that wall.

- **Cutter** — a part placed under a subtract operation; it carves the accumulated
  result. Cutters are ordinary parts (reusable via definitions), not a special node kind.

- **Junction** — a corner/meeting piece where instanced parts adjoin (walls at a bastion,
  roads at an intersection). A junction is **a part built to suit the situation** —
  authored and placed like any other (possibly as a fixture) — never a world-frame patch
  on the composed result.

## Sketching

- **Profile** — the authored 2D outline a body is lifted from: a closed path of lines, arcs,
  Bézier segments and whatever curve kinds arrive later, positioned in continuous
  coordinates and **never required to align to the voxel lattice**. Editable input with
  control points, kept exact; it is what the author manipulates, not what the document
  means.

- **Flattened profile** — the profile reduced to a polygon at sub-voxel chord tolerance.
  **This is the profile's meaning, not an approximation of it**: field, classification,
  resolve and outset see only the polygon, so a new curve kind is additive at the authoring
  layer and invisible below it. Because the polygon *is* the meaning, the flattening is
  deterministic and versioned — changing it changes existing documents.

- **Lattice snapping** — the voxel grid standing in for a constraint solver. Snapping to
  grid, edges and axes delivers axis-alignment, equal lengths and coincidence as a
  by-product of quantization, so the profile layer carries no constraint entities, no
  solver, and none of the over-constrained or flipped-solution states those bring.

## Field

- **Field** — the signed scalar meaning of a node: negative inside the body, positive
  outside, zero on the surface. Every producer has one, and composition is field algebra
  (union is the minimum, subtract and intersect are maxima). The field is the seam between
  **Intent** and **Occupancy**: authoring says what a body *is* in field terms, and
  occupancy is derived by asking where the field is at or below the isolevel. It is a
  *meaning*, not a stored artifact — nothing keeps a field resident.

- **Field metric** — the distance notion a field is exact in. **Euclidean** (L2) measures
  straight-line distance and rounds offsets; **Chebyshev** (L∞) measures largest-axis
  distance and keeps offsets square, matching the rectilinear grain of block and voxel
  work. Exactness is a property of the *data*, not of a producer kind — a rectilinear
  profile admits a cheap exact L∞ field, the same producer over a diagonal edge does not —
  so a body's metric is **derived on demand, never authored and never saved**.

- **Field lift** — how a lower-dimensional field becomes a body's field: an operation takes
  the sketch's 2D profile field into 3D. Extrude combines it with a slab, revolve evaluates
  it in radius-and-axial terms, sweep carries it along a path. The sketch→volume authoring
  atom and the lift are the same joint seen from two sides.

- **Lipschitz bound** — the guarantee that a field never changes faster than distance does
  (moving a step can change it by at most that step's length, in the field's own metric).
  What makes a field usable over a whole cell rather than point by point: a single sample
  plus the cell's radius brackets every value inside it. A field claiming a metric it does
  not satisfy makes every classification built on it unsound.

- **Outset** — a body dilated uniformly outward before it composes; equivalently, the field
  shifted by a constant. Applied to a **cutter** it is clearance: cut this, but leave a gap.
  A negative outset (**inset**) shrinks instead. Uniform across producers because it is a
  property of the field, not of any particular shape's parameters. Its shape follows the
  body's category, never its edge angles: **boxes and every profile-lifted body outset
  square** (they are polygonal once flattened), **curved primitives outset round** (no
  closed-form L∞ distance exists for them). A measurement, not an integer voxel count.

## Viewing

- **View mode** — the scene viewer's exclusive rendering mode, one of exactly three:
  **Normal** (the finished look — no ghosts, no clipping), **Onion fog** (the selected
  object clips to the layer band with ghost haze for the layers outside it, inside the
  object's own region; everything else renders finished), **Show booleans** (every
  subtract/intersect operand in the selected subtree x-rays over the finished scene).
  A property of the **viewer, never the document**: it follows the active selection,
  is not saved with the scene, and never enters undo history. Selecting the root part
  applies the mode scene-wide; with nothing selected a mode has no target and the scene
  renders finished.

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
