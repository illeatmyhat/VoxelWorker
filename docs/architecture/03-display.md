# 03 — Display

Display is a *derivation*: a cache of the evaluator's output arranged for the GPU, never
an authority on anything. The design goal, like evaluation's, is a cost envelope:

> **Per-frame cost independent of scene complexity. Per-edit display cost proportional
> to the edit. The model never blanks and the interface never stutters while a display
> rebuilds.**

Two display paths exist, with a strict seniority between them.

## What kind of renderer this is

The renderer sits at a deliberate point in a well-explored design space, and naming
its neighbours is the fastest way to say what it is:

- **From the edit-list school** (the lineage of Dreams and Claybook — systems whose
  truth is a list of sculpting operations, not a mesh) it takes its one structural
  idea: **march the cache, not the field**. The renderer never evaluates the
  document's analytic fields; it walks a derived occupancy cache, so a frame's cost
  is independent of how many operations compose the document, and an edit re-bakes
  only its dirty region.
- **From that same school it pointedly refuses the renderer.** Those systems draw an
  *impression* — splatted point clouds, temporally accumulated softness — because
  their medium tolerates approximation. This medium does not: a chiseled block in the
  target game *is* a lattice of boolean microvoxels with a per-block material, and
  the planner's promise is that the screen shows exactly what will be built. The
  display is therefore **exact occupancy, always** — approximation is not a style
  this system declined but a correctness violation it cannot express.
- **Signed-distance mathematics lives only in evaluation.** Producers are analytic
  fields and compose by boolean operators, but what rendering consumes is their
  *discretization*: rays never sphere-trace a distance, they step the block lattice
  with a grid DDA (hierarchical over the occupancy pyramid). Distance-like queries
  answer per-edit questions; per-frame work is lattice walking over cached bits.
- **Against the deep-octree school** (sparse voxel octrees) it chooses a **shallow
  brick map**: a flat sorted record array, a pooled atlas, and a few coarse
  occupancy levels. The brick granule is one *block* — the unit the document, the
  game, and the material system already share — and flat structures are trivially
  patchable per edit (swap records, recycle atlas slots), where deep hierarchies
  make incremental update their hardest problem. The pyramid's few levels buy all
  the empty-space skipping the scenes require; depth beyond that would purchase
  asymptotics the workload never asks for.
- **From the host game itself** comes the storage ontology: bulk blocks kept coarse,
  chiseled blocks kept as separate micro-geometry — the split the two-layer chunk
  mirrors and the display inherits.

One sentence, then: the document layer thinks like a sculpting edit list, the storage
layer thinks like the target game, and the display layer is a classical
exact-occupancy brick-map raymarcher with a rasterized-mesh understudy.

## The brick field — the primary display

The primary display raymarches a **cached brick field**: "march the cache, not the
field". Rays never touch the operation stack; per-frame cost is a function of what is
on screen, not of how the document was authored.

Its vocabulary:

- A **brick** is one block's worth of voxels, the same granule the document is
  denominated in. A boundary block becomes a **sculpted brick** — its voxel occupancy
  packed into one slot of a pooled 3D texture (the **atlas**). A coarse-solid block
  becomes a **coarse brick** — a marker record with no per-voxel data at all.
- **Records are surface-only.** A block completely enclosed by solid neighbours can
  never be the first thing a ray hits, so it never becomes a record; the interior is
  represented by the chunks it came from, not by display records. Upload cost per
  wholesale rebuild is therefore proportional to the scene's *skin*.
- The record list is sorted by a packed world-block key; a ray resolves "which brick am
  I in?" by binary search. Above the records sits a **clip-map occupancy pyramid** —
  coarser any-brick-inside levels — so a ray's traversal strides through empty space at
  the coarsest empty level and descends only where occupancy exists.
- Solid hits are shaded per-face from the block's material. The face texture is a pure
  function of the block lattice — the same rule the mesh path uses — so the two display
  paths are pixel-comparable by construction and no per-brick shading state exists.

Per-edit, the brick field is **incrementally patched**: dirty chunks re-emit their
records, freed sculpted slots return to a free list, and only touched atlas slots are
re-uploaded. A wholesale rebuild (density change, region-spanning edit) rebuilds the
whole field — off the main thread, per [Work](04-work.md).

## The cuboid mesh — the understudy and the oracle

The second path is a classical mesh: each chunk's coarse solids emitted as merged
boxes, each boundary block's cuboids emitted directly, faces across chunk seams culled
by the seam-solidity flags — all read from the same two-layer chunks as everything
else.

The mesh is kept for two permanent reasons:

1. **It is the display of last resort.** Some scenes the brick field cannot represent —
   the field's atlas is occupancy-only, so a block whose *interior* mixes materials, or
   a scene whose blocks disagree on a scene-wide surface treatment, falls back to the
   mesh. Debug views that need per-face vertex attributes are mesh-only by nature.
2. **It is the pixel oracle.** A raymarched display is only trustworthy next to an
   independently derived rendering of the same chunks. Exactness gates compare the two.

The mesh is an understudy, not an equal: when the brick field is on stage, the mesh is
**not built at all** — building a display nobody draws is pure waste — and it is marked
stale so nothing dares patch it later.

## Engagement and handover

Exactly one path draws a given frame. **Engagement** — "the brick field is the
display" — is a single predicate evaluated identically at every decision point: a live
field exists, and no mesh-only mode (a debug view) is active. Everything that follows
from engagement follows from that one predicate; no seam is allowed its own private
notion of who is drawing.

When engagement flips, a **handover** occurs, governed by one rule: *the model never
blanks*. If the replacement display is already current, the old one is dropped
immediately. If the replacement is still building, the old display — stale though it
is — keeps drawing until the replacement lands, and is dropped at the moment of
installation. A stale display kept for this purpose is a *placeholder*: it may be
looked at, never patched (see the staleness law in [Work](04-work.md)).

## The onion skin

Layer inspection — scrubbing through horizontal slices of the model — is a display
*clip*, not a geometry operation. The active layer band renders normally; material
outside the band renders as a ghosted slab pass driven entirely by per-frame uniforms.
Scrubbing therefore costs nothing proportional to the scene: no geometry is rebuilt, no
occupancy is re-derived, and the same mechanism serves both display paths.

## The camera and the shell

The camera orbits; explicit actions (home, fit, focus) frame the model. An *edit never
moves the camera*: the document recentres itself as it grows, and the shell compensates
the camera target by the recentre shift so the view stays planted. The shell draws
whatever is current every frame and owns nothing the document or evaluator would
recognize as state.
