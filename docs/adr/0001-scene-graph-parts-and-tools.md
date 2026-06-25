# ADR 0001 — Scene graph: parts vs tools (assembly layer)

- **Status:** Accepted (design signed off; implementation begins at sequence step 1)
- **Date:** 2026-06-25
- **Supersedes / extends:** the single-producer assumption baked into `GeometryParams`;
  builds on the resolved-grid seam in `REPRESENTATION.md`.

## Context

Today the app has exactly one producer, and it is conflated with its own parameters:
`GeometryParams { shape, size_blocks, voxels_per_block, wall_blocks }` *is* "the object."
When a second producer (the debug cloud field) was added, it was smuggled in as a
`debug_clouds: bool` flag on `GeometryParams` — i.e. a producer expressed as a *mode of the
first producer's parameters*. That is backwards and does not scale: a third producer would be
a second boolean, producers can't coexist, nothing can be positioned, and there is no notion of
"the document contains these things."

The resolved-grid seam was deliberately built for more than one producer
(`VoxelProducer::resolve(&self, grid)` — *"multiple producers can target one grid"*), and
`REPRESENTATION.md` defers the composition story until "sculptor users are real." This ADR pulls
the **assembly** half of that forward, because we already have two producers and want more
(saved chiseled "parts").

### Two graphs, not one (important)

Fusion has two distinct hierarchies, and we should not conflate them:

1. **Assembly graph** — instances of components positioned in space. *This ADR.* The scene is a
   list of placed nodes.
2. **Feature / construction tree** — the parametric history *inside* one part (sketch → extrude →
   fillet …). In our world that is the future **SDF construction tree** (booleans / lathe / array)
   described in `REPRESENTATION.md`. That lives *inside* a single Tool node.

So: a Tool node may eventually own an SDF construction tree; the sculpt overlay
(`REPRESENTATION.md` mode 2) is an overlay producer on a node or the scene. The scene graph sits
**above** both. Keeping these separate is what stops us painting into a corner.

## Decision

Introduce a **Scene** (assembly) of **nodes**, each wrapping a **producer** plus a placement.
Producers come in the two kinds the user named:

- **Tool** — a *parametric* producer, re-resolved when its parameters change. v1: `SdfShape`.
- **Part** — a *static* voxel body with no meaningful generation parameters; dropped in as-is.
  v1: the debug cloud field; future: a saved chiseled block (a "corner piece"), an imported `.vox`.

The graph is **recursive** (assemblies contain assemblies — a village is an assembly of building
assemblies of parts), and reuse is **by reference**: a repeated house is one *definition* placed by
many *instances*, not many deep copies.

```rust
struct Scene {
    extent_blocks: [u32; 3],          // working volume / canvas — expect up to ~1024³ (see "Scale")
    definitions: Vec<AssemblyDef>,    // reusable sub-assemblies (e.g. "house"), placed by Instance
    root: AssemblyDef,                // the top-level assembly
    active: Option<NodeId>,
    // voxels_per_block is NOT here — it is an application setting (default 16), see "Density".
}

struct AssemblyDef { id: DefId, name: String, children: Vec<Node> }

struct Node {
    id: NodeId,
    name: String,
    transform: NodeTransform,    // LOCAL transform; composes with ancestors' (world = parent ∘ local)
    operation: CombineOp,        // v1: always Union
    visible: bool,
    content: NodeContent,
}

enum NodeContent {
    Tool { shape: SdfShape, material: MaterialId },  // leaf: parametric, single material
    Part(Part),                                      // leaf: static, carries per-voxel materials
    Group(Vec<Node>),                                // owned one-off sub-assembly
    Instance(DefId),                                 // reuse a definition (village house ×N)
}

enum Part {
    DebugClouds { seed: u32 },   // a "part with one trivial knob"
    // future: SavedBody(VoxelBlob), ImportedVox(...)  — each with baked per-voxel materials
}

// v1 only constructs `Union`. The enum exists so subtract/intersect/override is a
// data change on the node, not a re-architecture.
enum CombineOp { Union /*, Subtract, Intersect, … (later) */ }

// v1 exposes integer translation only, but the type targets a full AFFINE
// (translation + rotation + scale) so rotation/scale (with voxel resampling) slot
// in later without a rewrite. Cycle guard: an Instance may not reference an
// ancestor definition.
struct NodeTransform {
    offset_blocks: [i32; 3],     // v1: translation
    // future: rotation, scale → a general affine
}
```

No flat per-node `material`: material is per-VOXEL (`Voxel.material_id`); a Tool assigns one, a Part
carries its own (see Materials).

### Resolution / composition

Resolution is **region-addressable**, not whole-scene (this is what lets the canvas be huge — see
Scale): `Scene::resolve_region(&self, region_blocks: Aabb) -> VoxelGrid`.

1. Size the output grid to the requested region × `voxels_per_block` (density from the app setting).
2. Walk the node tree, composing transforms down (`world = parent ∘ local`). A **spatial index**
   over node world-AABBs skips the whole subtree of anything that doesn't intersect the region — so
   a 1000-node village costs ~the nodes actually touching the region, not all of them.
3. For each **visible** leaf whose world-AABB hits the region: resolve its producer into a local
   grid (producers still emit centred-at-origin content — the trait is unchanged), then **stamp**
   it into the output under the node's world transform. Each written voxel carries a `material_id`:
   a **Tool** stamps its one material; a **Part** copies its own per-voxel materials.
4. **v1 composition = union only** (additive): the voxel set is the OR of the contributing nodes; on
   overlap the later node wins the material. The per-node `operation` is `Union` for now — subtract
   / intersect / the sparse force-off override (`REPRESENTATION.md` mode 2) are the growth path,
   out of scope for v1.

The producer trait does **not** change; compositing (tree walk + transform + stamp) is the new step
the Scene owns. For a small scene the requested region is the whole extent and this is one grid —
identical behaviour to today.

### Materials (per-voxel)

Material is **per-voxel** — `Voxel.material_id` already exists for exactly this. The node kinds
differ in how they fill it: a **Tool** is single-material by nature (it assigns one `material_id`
to every voxel it emits, picked in its inspector), whereas a **Part** can be multi-material and
carries its own per-voxel `material_id`s in its stored data (a saved chiseled block, a `.vox`
import). So there is no single per-node material override.

`material_id` indexes a **material table** (procedural Stone/Wood/Plain today; VS palette blocks
later). The table is a scene-level concern; fleshing it out (and the per-voxel material picker for
multi-material parts) tracks with when real multi-material parts arrive. v1 keeps the existing
procedural set and one material per Tool.

### Density

`voxels_per_block` is an **application-level setting**, default **16**, that users almost never
change — not a per-node and not really a per-document property. It stays a global app setting; the
scene reads it at resolve time. A Part stored at a different resolution would need resampling, which
v1 sidesteps by assuming the app default.

### Scale — the canvas is huge (target ~1024³ blocks, designed to go far beyond)

This is the constraint that reshapes everything downstream of the seam. A 1024³-block canvas at
density 16 is ~16384³ ≈ **4 trillion** voxels — there is **no monolithic resolved grid**, dense or
sparse. The "resolved grid is the one truth" seam still holds, but the truth becomes **chunked and
lazily materialised**, exactly like a voxel-world engine (Minecraft / VS). We commit to the full
streaming stack below; **LOD is the one piece deliberately parked** (see "Deferred: LOD").

**Streaming model (committed):**

- **Chunked resolution.** The world is partitioned into fixed regions (chunk size TBD, ~8–32 blocks).
  A chunk is resolved on demand via `resolve_region(aabb, lod)`, cached, and invalidated only when an
  edit's world-AABB intersects it. `MAX_GRID_VOXELS` stops being a scene guard and becomes a
  **per-chunk bound**.
- **Spatial index for the tree.** With thousands of village nodes, resolving a chunk must not walk
  the whole graph; an AABB index over node world-bounds (updated on edits) gates the walk.
- **Frustum culling + out-of-core store.** Per-chunk render items, frustum-culled; resident chunks
  bounded by render distance (constant memory regardless of scene size). Authored/unique chunk data
  streams from **disk**, not just RAM, so scene size is decoupled from RAM entirely.
- **Seam consumers scope to a region.** Onion fog, layer scrubber, diameter readout act on the
  active region, not the canvas. `.vox` export becomes a **streamed / region export**.

**Coordinate model (committed, to avoid a retrofit):**

- World addressing is **64-bit**: `i32` (→ later `i64`) **block** coordinates plus a sub-block voxel
  offset; node transforms compose in **f64**. This pushes the addressing ceiling to billions of
  blocks/axis and beyond.
- Rendering is **camera-relative (origin-rebased)**: each chunk's model matrix is
  `chunk_world_origin − camera_floating_origin`, computed in i64/f64 and downcast to **f32 per
  frame**, so f32 precision is always high near the viewer (no "far lands" jitter). Per-chunk
  resolution always works in small **chunk-local f32** coordinates.

**Geometry at scale (committed):**

- **Greedy meshing** replaces the current one-instanced-cube-per-voxel renderer (which cannot scale):
  merge coplanar voxel faces into quads per chunk.
- **GPU instancing** for repeated parts: an `Instance`d definition draws one shared mesh N times.
- **Palette + sparse/RLE** per chunk (material palette + indices, skip air) for the stored and
  resident representation.

### Deferred: LOD (seam preserved, may never be built)

LOD (drawing distant chunks at reduced resolution / as impostors) is **out of scope** and may never
be done. The architecture only has to make it *not impossible*, which costs us almost nothing now:

- `resolve_region(aabb, lod)` **carries an `lod` parameter from day one** — always `0` (full
  resolution) for now. A future LOD level just downsamples the chunk before meshing.
- The chunk cache and render items are **keyed by `(chunk_coord, lod)`**, and the renderer consumes
  **opaque per-chunk render items** it does not introspect. A future LOD system substitutes a coarse
  item for a far chunk and nothing upstream changes.
- Per-chunk resolution is **not assumed globally uniform** in the mesh/buffer types — each chunk
  carries its own resolution — so mixed-LOD scenes are representable even though we only ever emit
  one level.

That is the entire LOD tax we pay up front: an unused `lod` field in the cache key and resolve
signature, and not hard-coding a single global resolution into the renderer.

Honest scoping note: even without LOD this is **engine-level work** (chunked, culled, origin-rebased,
greedy-meshed voxel renderer) and is the heaviest part of the effort — far bigger than the
scene-graph data model. It is sequenced as its own milestone(s) so the early steps stay shippable on
small scenes.

### What this costs each subsystem

| Subsystem | Change |
|---|---|
| Renderer instances, onion fog occupancy texture, layer scrubber, `.vox` export, view cube, gizmo, diameter readout | **None at small-scene scale** — they already read the resolved grid (the seam paying off). **At canvas scale** the renderer/fog/export must operate per-chunk with frustum culling + greedy meshing (the scale milestones), but they still read *a resolved grid* — just a regional one. |
| Camera auto-frame | Frame `extent_blocks` instead of the single shape's dims. |
| `GeometryParams` | Demoted: it becomes the *inspector state of a Tool node* (now also carrying the Tool's one material). `debug_clouds: bool` is **deleted** (a Part node replaces it). |
| `MaterialChoice` | Moves from a global panel control to a **Tool-node property**. Parts ignore it (they bring their own per-voxel materials). |
| Panel UI | **The bulk of the work.** A node list (add / select / delete / visibility) + an inspector that shows the active node's editor (Tool → shape/size/wall + material as today; Part → name + seed + transform). |
| Persistence (`AppConfig`) | Serialize the node list + scene extent/density. Keep reading the old single-geometry config → migrate to a one-Tool-node scene. |

### Migration

- Old config (single `geometry`) loads as a **one-node scene** (a Tool node). No user-visible loss.
- The `debug_clouds` boolean and its three branch sites collapse into "the scene contains a
  DebugClouds Part node."
- `rebuild_geometry` / `resolve_active_producer` become `scene.resolve_region(region)`.

## Proposed build sequence (each a green checkpoint)

1. **Model + region-addressable compositing, no UI.** Add `Scene`/`Node`/`NodeContent` and
   `resolve_region`; route the current object through a one-node scene resolved as a single region;
   delete the `debug_clouds` boolean. App behaves identically. Proves resolution + the regional API.
2. **Flat node list + add/delete/select + visibility.** Several leaf nodes compositing into one
   region; inspector switches on the active node. Drop the clouds in as a Part beside an SDF Tool.
3. **Per-node translation + per-voxel material.** Place nodes; `material_id` wired through the
   renderer (Tool = one material, Part = its own).
4. **Recursion + instancing.** `Group` and `Instance(DefId)` + a definitions list; nested transform
   composition; the village-of-reused-houses case. Still small-scene (single region).
5. **Coordinate model + greedy meshing.** Switch world addressing to 64-bit block+sub-voxel and f64
   transform composition; camera-relative (origin-rebased) rendering; replace per-voxel-cube
   instancing with greedy-meshed chunks. Still a single region — proves the render path that scale
   needs. `resolve_region` gains its (always-0) `lod` parameter; render items keyed by `(coord,lod)`.
6. **Chunked streaming.** Spatial index over node bounds; on-demand chunk resolve + cache/invalidate;
   frustum cull; fog/scrubber/export scoped to a region. Memory bounded by render distance. *(Likely
   its own sub-ADR.)*
7. **Out-of-core + compression.** Disk-backed chunk store with eviction; per-chunk palette + sparse
   storage. Decouples scene size from RAM. *(Likely folded into the sub-ADR.)*
8. **Persistence** of the scene (tree + definitions); migrate old single-geometry configs; regional
   `.vox` export.

**Parked (may never happen, but the seam is preserved):** LOD / impostors.
**Explicitly later:** rotation/scale transforms, subtract/intersect/override ops, the saved-part
library, and the in-Tool SDF construction tree.

## Decisions (resolved)

1. **Composition** — **union only** for v1; the per-node `CombineOp` enum exists so subtract /
   intersect / override expand as a data change, not a rewrite.
2. **Materials** — **per-voxel**. Tools are single-material (one `material_id` they assign); Parts
   are potentially multi-material and carry their own per-voxel materials.
3. **Transforms** — **translation only** in v1, but `NodeTransform` is a full-affine target
   (translation + rotation + scale) so rotation/scale (with resampling) slot in later.
4. **Density** — **application setting, default 16**; users rarely change it. Not per-node, not
   per-document.

5. **Scene extent** — **explicit user-set working volume** (a "stock" / build canvas), **expected to
   get very large (target ~1024³ blocks, designed to go far beyond)**.
6. **Nesting & reuse** — the graph is **recursive** (assemblies of assemblies); reuse is **by
   reference** (definition + instances) so a village of identical houses is cheap.
7. **Scale model** — no monolithic grid: **chunked + lazily-resolved + spatial-indexed**, with
   **64-bit addressing**, **origin-rebased rendering**, **greedy meshing**, **GPU instancing**,
   **out-of-core store**, and **per-chunk palette/sparse** compression. Committed.
8. **LOD** — **deferred (may never be built)**, but architected-for: `resolve_region(aabb, lod)` and
   render items keyed by `(coord, lod)`, renderer consumes opaque per-chunk items, per-chunk
   resolution not assumed uniform. Cheap seam, no implementation now.

### Still open

9. **First slice** — start at sequence step 1 (model + region-addressable compositing through a
   one-node scene, no UI, boolean deleted)? Assuming yes unless you say otherwise.

## Consequences

- **Positive:** the producer seam finally carries its intended weight; new producers are nodes, not
  flags; per-node material/placement become possible; the door to assemblies of chiseled blocks is
  open; the assembly vs construction-tree split keeps the parametric story (booleans/lathe) and the
  sculpt overlay composable later.
- **Negative / cost:** real UI work (node list + inspector), a persistence format change, and —
  dominating everything — the **scale milestones** (steps 5–7): turning the single-grid pipeline into
  a chunked, frustum-culled, origin-rebased, greedy-meshed, out-of-core voxel-world renderer. That is
  engine-level and by far the largest piece; the model refactor (steps 1–4) is comparatively small.
  Sequenced so small scenes ship long before the canvas does. The current `MAX_GRID_VOXELS`
  single-object cap is retired in favour of per-chunk bounds. **LOD is intentionally not built** — we
  pay only a tiny seam (an unused `lod` key) to keep it possible.

## Alternatives considered

- **Keep adding booleans** (status quo). Rejected: doesn't compose, can't position, no document model.
- **A general entity-component system.** Rejected as over-built for a fixed, small set of producers;
  the enum model is legible and matches the part/tool distinction directly.
