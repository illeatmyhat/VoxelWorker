# ADR 0003 — Foundation rework: parts, sculpt, command journal & streaming store

- **Status:** Proposed
- **Date:** 2026-06-27
- **Supersedes / extends:** consolidates and supersedes the open growth-path clauses of
  [ADR 0001](0001-scene-graph-parts-and-tools.md) (composition beyond union; transforms beyond
  translation; the sculpt overlay; persistence) and [ADR 0002](0002-engine-streaming-meshing.md)
  (E5 out-of-core wiring, #20 Steps 2/4). ADR 0001's scene-graph/producer decisions and ADR 0002's
  cuboid-mesher/coordinate/streaming decisions **stand**; this ADR is the layer that makes them
  *editable, sculptable, undoable, share-serializable, and horizontally streamable* without a
  big-bang rewrite.

## Context

The app is v1 feature-complete: a Vintage Story chiseling planner (wgpu 29 / egui 0.34 / winit
0.30) that composes parametric shapes into a 3D model and exports `.vox`. We are doing a
**foundation rework before hundreds of features stack on top**. Correctness and extensibility
dominate effort/token minimization.

### Locked trajectory constraints (decided by the product owner — not relitigated here)

1. **Direct voxel sculpting is coming**, modeled as a per-part **sparse override layer**
   (force-on / force-off deltas) composited on top of parametric producers. Composition becomes
   **ordered add/subtract layers**, not union-only. The override layer reuses the *concept* of
   sparse, chunk-keyed storage — but, per §3/§5 below, with a **dedicated integer-delta codec**, not
   a binary reuse of the existing f32-grid `chunk_storage::compress`.
2. **Part / assembly separation (Fusion 360 model).** Geometry/sculpt edits belong to a part
   **definition**, anchored in the definition's **local** frame. An **instance** in an assembly is
   reference + transform only — position/orient/pattern, never edit-geometry. To sculpt you "open
   the definition"; editing a definition propagates to all instances. The code already has
   `AssemblyDef`/`DefId` + `NodeContent::Instance(DefId)` + `Group` recursion to build on.
3. **No real-time collaboration, ever.** A simple **linear command stack with inverse commands**
   for undo/redo. No event-sourcing, no CRDTs.
4. **Users save & share PROJECT FILES.** The document format is a **published interface**: version
   tag + forward migration from v1. (This is the *opposite* of config back-compat, which the owner
   explicitly does not care about — configs may break freely; shared documents may not.)
5. **Anisotropic large scenes:** routinely >10,000 blocks horizontal (XZ), but vertical (Y) bounded
   to ~1,000–2,000 blocks. **Out-of-core horizontal streaming is foundational** — tile by XZ
   distance, keep full-Y columns resident. At this extent f32 precision dies, so the existing
   **i64-subtract-before-f32-downcast** rebasing (ADR 0002 Decision 2) is mandatory; a far-edge edit
   must NOT re-resolve the world.

### The convergent backbone to deliver

```
command → sparse delta → per-chunk invalidation → incremental re-mesh
```
reversible, scaling horizontally. Concretely: a per-part, chunked, sparse, **command-journaled**
voxel store, with **parametric producers + sculpt overrides composited per-chunk** and **rebased at
consumption**.

### What we KEEP (design around, do not redesign)

- **The resolved-grid READ seam** — every consumer reads a resolved `VoxelGrid`; nothing reads the
  SDF directly. The single best asset; it is what lets us insert the override-compositor and the
  per-chunk store *behind* an unchanged read interface.
- **`panel.rs` intent/effect split** (`PanelResponse` carries expensive/IO intents; cheap UI mutates
  in place). Becomes the command-emission seam.
- **Pure, well-tested cores:** camera orbit/snap/roll math, `vox_export`, `cuboid::decompose_into_boxes`
  (greedy box mesher over a representation-agnostic `VoxelRegion`), `SnapTween`, `classify_cube_point`.
- **The i64 rebasing precision trick** (`scene.rs::resolve_chunk_rebased`, where the
  `absolute − floating_origin` subtraction is done in i64 *before* the f32 downcast); the per-chunk
  cuboid mesher (chunk-resident `HashMap<chunk_coord, buffers>`, frustum-culled); palette/sparse
  chunk compression + standalone `DiskChunkStore` (RAM-LRU + disk spill); the golden-PNG headless
  harness **concept**.

### The debt this foundation must fix or supersede

- **God-objects.** `Scene` (document + selection + resolve/chunk/frame-math/gizmo engine);
  `WindowedState` (124 fields, 321-line `render()`, no controller/intent layer); `ChunkResolveCache`
  (a cache that accreted export/scrubber/renderer methods + a **second LRU** on top of
  `DiskChunkStore`'s).
- **`shot.rs` is a ~1700-line parallel re-implementation of the windowed render path** — the golden
  harness tests a *copy*, not the real app; every interactive bug escaped it. A shared headless
  **`AppCore`** used by BOTH the window and the screenshot tool is the keystone.
- **Compositor is union-only; transforms translation-only.** Adding a producer edits 3 hard-coded
  `match` arms — no producer registry/trait. (`scene.rs::walk_nodes` composes placement by **pure
  i64 addition** of `offset_blocks`; `NodeTransform` carries translation only — `voxel.rs`'s
  `VoxelProducer::resolve(&mut grid)` has no chunk window at all.)
- **Selection is a positional `NodePath`** that invalidates on structural edits — no stable id.
  Blocks undo/multi-select/references.
- **Wrong-way deps:** `scene.rs` imports `panel.rs` (domain types stranded in UI) and `renderer.rs`
  (`CHUNK_BLOCKS` — the streaming quantum — lives in the GPU module); `chunk_cache` imports
  `vox_export`.
- **Leaks:** a `GRID_OVERLAY_BIT` (`1 << 15`) packed into the `material_id` u16 (a render flag in a
  data field, mirrored in 2 shaders — see `voxel.rs::GRID_OVERLAY_BIT`); 3 coordinate frames
  reconciled by comment-arithmetic; "extent-changing edit nukes the whole cache" (the
  `ChunkResolveCache` rebinds on a *floating-origin / density* change — `chunk_cache.rs::rebind_if_changed`
  — and chunks are stored pre-rebased against the composite recentre); `incremental_rebuild_plan` is
  **computed but not consumed** (renderer re-meshes wholesale on edit).
- **Per-voxel payload is f32-absolute.** `voxel.rs::Voxel` stores `world_position: [f32; 3]` (the
  voxel *centre* in world-grid space). At XZ~10k that f32 has ~1 mantissa bit below the voxel and the
  position becomes ambiguous; the current code only stays exact because the i64 rebase in
  `resolve_chunk_rebased` keeps the *rendered* magnitude small — the **stored** payload is still f32
  and is the wrong representation for an absolute-i64 store (see §3 G1).

## Decision

One coherent foundation across all areas. Each subsection states the choice and how it resolves the
red-team blockers.

### 0. Module layering (the dependency-direction fix)

Strict acyclic layering; dependencies point **down** only. Domain types stop living in UI/GPU
modules.

```
core_geom    : VoxelGrid/VoxelRegion, Voxel, SdfShape, cuboid decompose, classify_cube_point,
               material table, AND the streaming quantum CHUNK_BLOCKS (moved out of renderer.rs).
               Pure; no egui/wgpu/winit.
doc          : the document model — stable ids (NodeId/DefId), scene graph, part definitions,
               sculpt override layers, producer registry, serialization (versioned). Pure.
edit         : the command stack — Command trait + inverse, the linear journal, edit-mode/open-def
               state machine. Depends on doc + core_geom. Pure.
store        : per-part chunked sparse store, DiskChunkStore, residency manager, override+producer
               compositor, resolve_region/resolve_chunk. Depends on doc + core_geom. Pure
               (no GPU); this is where ChunkResolveCache's data role goes.
app_core     : headless orchestrator (AppCore) — owns doc + edit + store + camera; applies commands,
               drives invalidation→remesh, produces opaque per-chunk render items + occupancy.
               Pure of windowing; depends on all above.  <-- THE KEYSTONE
gpu          : wgpu mesh upload/draw, fog, atlas. Consumes AppCore's opaque render items.
ui           : egui panels/gizmo. Emits intents → commands. Depends on app_core (NOT vice-versa).
bin/main     : windowed shell (winit) = AppCore + gpu + ui.
bin/shot     : headless shell = AppCore + gpu (offscreen). SAME AppCore as main.
```

Resolves: god-objects (`Scene`/`WindowedState`/`ChunkResolveCache` decompose into doc + edit +
store + app_core); wrong-way deps (`scene.rs→panel.rs` becomes `ui→app_core`; `CHUNK_BLOCKS`
relocates to `core_geom`; `chunk_cache→vox_export` becomes `store`/`vox_export` both under
`core_geom`). The shared `AppCore` retires `shot.rs`'s parallel render path.

### 1. Identity, scene graph & part/assembly modes

**Stable `NodeId` replaces positional `NodePath` as the identity of record.** `NodePath` was a
positional path that invalidated on every structural edit; it cannot survive undo, multi-select, or
references.

```rust
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(u64);   // process-stable, monotonic from a document-owned counter
// DefId stays as-is (already stable u32; widen to u64-from-counter for symmetry).
```

- Nodes/definitions are stored in **id-keyed arenas** (`SlotMap<NodeId, Node>` per `AssemblyDef`),
  with children expressed as `Vec<NodeId>` for ordering. Selection, command targets, and gizmo
  anchoring all reference `NodeId`, never an index path.
- `NodePath` is **demoted to an ephemeral UI projection** for the tree widget (compute on render
  from the arena); it is never stored in the document, in commands, or in selection. This keeps the
  existing tree-row rendering while removing the invalidation footgun.

**Part definitions vs assembly instances (the Fusion rule), enforced by an edit-mode state machine
in `edit`:**

- `AssemblyDef` gains the per-part editable content: its parametric producer node(s) **and** its
  sculpt override layer (below), all in the **definition-local frame**.
- An `Instance(DefId)` node carries only `NodeTransform` (promoted to the 24-orientation lattice
  rotation + i64 translation — §3, milestone-scoped). It has **no** editable geometry surface.
- `EditTarget` is either `Assembly` (place/move/pattern instances; geometry tools disabled) or
  `OpenDefinition(DefId)` (sculpt/producer tools enabled; instance placement disabled). The UI's
  "active context" is this enum, not a node.
- **Attempting a geometry edit on an instance is rejected at command construction** (the command
  factory returns `Err(EditError::InstanceNotEditable { def: DefId })`), and the UI offers "Open
  definition" (which switches `EditTarget` to `OpenDefinition(def)`). This is **S3**.

### 2. Command / undo: linear journal with inverse commands

Per locked constraint 3 — a **linear stack**, no event-sourcing/CRDT.

```rust
pub trait Command {
    /// Apply to the document + store; return the inverse needed to undo exactly.
    fn apply(&self, doc: &mut Doc, store: &mut Store) -> Result<Box<dyn Command>, EditError>;
    fn label(&self) -> &str;            // for the undo menu
    fn coalesce_key(&self) -> Option<CoalesceKey>;  // e.g. one drag = one undo
}

pub struct CommandStack {
    done: Vec<Box<dyn Command>>,        // each entry stores the inverse captured at apply time
    undone: Vec<Box<dyn Command>>,
}
```

- **All document mutation flows through commands.** `ui` emits an *intent*; `app_core` turns it into
  a `Command`, applies it, pushes the captured inverse. This is the single consistency point for
  threading (§10).
- **Sculpt strokes capture a sparse inverse delta, not a snapshot.** A `SculptStroke` command stores
  only the `(voxel_addr → prior_override_state)` pairs it overwrote (typically a few thousand), so
  undo is **O(stroke)**, touching only the chunks the stroke intersected. This is **S1**.
  - **O(stroke) is only achievable once the producer trait gains a chunk window (§3 G5).** Today
    `VoxelProducer::resolve(&mut grid)` re-evaluates a *whole region* per call (`voxel.rs`), so a
    naive "re-resolve the stroke's dirty chunks" is O(part_volume × dirty_chunks), not O(stroke).
    The `resolve_into(chunk_box, out)` precondition is what makes the chunked re-resolve proportional
    to the stroke, not the part.
- **Coalescing:** a continuous drag/scrub shares a `CoalesceKey` so it collapses to one undo entry.
- **Editing a definition is one command** whose `apply` mutates the `AssemblyDef`; instance
  propagation is automatic because instances *reference* the def (no per-instance edits to journal).
  Invalidation fans out to all instance placements (§4). This is **S2**.

### 3. Compositor: chunk-local payload, ordered override layers, producer registry, rotation, orphan policy

#### 3a. Per-voxel payload becomes chunk-local integer (load-bearing representation change — G1)

This is a **prerequisite** of the absolute-i64 store (§4) and of the override codec (§5), not a
detail. Today `voxel.rs::Voxel` stores `world_position: [f32; 3]` (the absolute voxel centre);
`chunk_storage::compress` already has to *reverse-engineer* the integer index out of that f32 (it
debug-asserts a uniform per-axis `centre_fraction` and `floor()`s the position — `chunk_storage.rs`).
At XZ~10k the f32 itself is lossy, so absolute storage in f32 is unsound.

**Decision:** the per-voxel payload is **chunk-local integer coordinates + material**:

```rust
pub struct Voxel {
    pub local: [u16; 3],          // voxel index WITHIN the chunk (u8 suffices at the app default
                                  // density 16 → chunk extent 64; u16 leaves headroom for denser)
    pub block_local_coord: [u8; 3],
    pub material_id: u16,         // real material only — the GRID_OVERLAY_BIT leak is gone (3c)
}
```

- The **absolute i64 origin lives ONLY in the chunk key** (`chunk_coord: [i64; 3]`), never in the
  per-voxel record. A chunk's stored content is position-independent of where the chunk sits.
- f32 is **produced only at consumption**, exactly as today's rebase does, as
  `(chunk_origin_i64 + local) − camera_floating_origin_i64`, the subtraction in i64, then a single
  f32 downcast (the existing `resolve_chunk_rebased` arithmetic, but now driven by the stored
  integer instead of recovered from an f32).
- This is what makes "absolute-i64 storage + i64-rebase-before-downcast" **exact** rather than merely
  "exact for near scenes". The region-scoped consumers (diameter / scrubber / `.vox` export) recover
  their integer indices directly from `local + chunk_coord` instead of `round(world_position + …)`.

This change is sequenced as a **prerequisite of the absolute-storage phase** (Phase D), gated by the
far-scene goldens added first (§G3 / Phase D0).

#### 3b. Composition stops being union-only

A part definition resolves as an **ordered layer stack**:

```rust
pub enum Layer {
    Producer { producer: Box<dyn VoxelProducer>, op: CombineOp, material: MaterialId },
    Sculpt(SculptOverride),   // sparse force-on / force-off deltas, part-local (see 3e / §5)
}
pub enum CombineOp { Union, Subtract, Intersect }   // the existing enum, finally exercised
```

Resolution of a chunk = fold the layers in order: producers stamp under their `CombineOp`; the
sculpt override is applied **last** (force-on sets occupancy+material; force-off clears).

#### 3c. The `GRID_OVERLAY_BIT` leak is removed — overlay flag becomes a per-DRAW uniform

The existing per-voxel `material_id` becomes the carrier of the **real material only**; the
`GRID_OVERLAY_BIT` (`voxel.rs`, mirrored in `cuboid.wgsl` + `cuboid_loaded.wgsl`) is removed from the
data field. The on-face-grid overlay is a **per-node render concern**, so it is carried as a
**per-draw uniform** (one bool per chunk-mesh draw, set from the owning node's
`grids.voxel_grid_on_faces`), **not** as a per-vertex attribute. Keeping it a per-draw uniform means
`cuboid::decompose_into_boxes` stays **representation-agnostic** (it never sees the flag), the high
bit is returned to `material_id`, and the 2-shader mirror is killed. (A per-vertex attribute would
re-pollute the mesh vertex format with a render flag — the same category of leak we are removing.)

#### 3d. Producer registry + chunk-windowed resolve (kills the 3-match-arm footgun; G5 precondition)

Producers become trait objects discovered through a registry, not hard-coded arms — **and the trait
gains a chunk window and a world AABB**, which is a *load-bearing precondition*, not a routine
signature change:

```rust
pub trait VoxelProducer: Send + Sync {
    fn kind(&self) -> ProducerKind;
    /// Resolve ONLY the voxels inside `chunk_box` into `out` (chunk-local integer coords).
    /// This replaces today's whole-grid `resolve(&mut grid)` (voxel.rs), which has no window.
    fn resolve_into(&self, chunk_box: ChunkBox, out: &mut VoxelRegion, frame: LocalFrame);
    /// The producer's bounded extent in world blocks, or None for an unbounded producer.
    /// A bounded producer lets invalidation/residency touch only intersecting chunks; an
    /// unbounded producer (e.g. DebugClouds) returns None and falls back to a wholesale clear.
    fn world_aabb_blocks(&self, xf: &NodeTransform) -> Option<Aabb64>;
    fn serialize(&self) -> ProducerData;                                  // versioned (§5)
}
```

- **Why this is a precondition, not routine.** The current trait
  `VoxelProducer::resolve(&mut grid)` (`voxel.rs`) re-evaluates the entire region every call. The
  per-chunk incremental / O(stroke) claims (**S1**, **S2**, **S8**) all depend on resolving *only*
  the dirty chunk window; without `resolve_into(chunk_box, …)` the "re-resolve dirty chunks" path is
  O(part_volume × dirty_chunks). This trait redesign therefore **sequences before** the sculpt and
  residency phases (it is its own early milestone — Phase F0).
- `world_aabb_blocks()` returns `Option<Aabb>`: a **bounded** producer (`SdfShape`) yields a finite
  AABB so the spatial index / invalidation only touch intersecting chunks; an **unbounded** producer
  (`DebugClouds`) returns `None` and keeps the **wholesale-clear fallback** (its edits invalidate the
  whole resident region — acceptable because it is a debug/static field, not a sculpt target).
- Adding a producer = register one impl. The existing `SdfShape` and `DebugClouds` become the first
  two registrants.

#### 3e. Sculpt overrides — part-local, address-anchored (codec specified in §5)

Sculpt overrides are part-local and anchored to **integer voxel addresses in the definition frame**.
They are stored as a **sparse, chunk-keyed** layer (the *concept* reused from the existing store), but
encoded with a **dedicated integer-delta codec** (§5 G2) — explicitly **not** a binary reuse of
`chunk_storage::compress`, which consumes an f32 `VoxelGrid` and debug-asserts a uniform
`centre_fraction` that integer force-on/force-off deltas do not satisfy.

#### 3f. Rotation is its own milestone (G4)

Today `NodeTransform` is **translation-only** (`scene.rs`: `offset_blocks: [i64; 3]`, with a
`// future: rotation, scale` marker) and the resolver composes placement by **pure i64 addition**
(`scene.rs::walk_nodes` sums `offset_blocks` down the tree). Promoting to affine is **not** a
one-substep change; it is a **dedicated milestone** (Phase F-rot) with three golden-gated parts:

- **(a) 24-orientation rotation as a TYPE-level enum** — `NodeTransform` gains
  `rotation: LatticeOrientation` (one of the 24 axis-aligned rotations), **not** a general affine in
  the stored type. Keeping it an enum means voxel resampling stays an **exact index permutation**
  (no interpolation), positions stay lattice-exact under i64 rebase, and the **serialization stays
  byte-stable** (an enum discriminant, not a float matrix).
- **(b) Rotated-chunk conservative-cover fan-out** — a rotated part's chunks no longer map 1:1 to
  output chunks; invalidation/residency must fan out over the **conservative cover** of the rotated
  AABB.
- **(c) Rotation-aware AABB-skip + spatial-index fingerprint** — the world-AABB used for skip tests
  and the `LeafFingerprint` (spatial index) must incorporate the orientation so a pure rotation
  re-keys correctly.
- Because overrides live in the **part-local** frame, moving/rotating an instance moves its overrides
  for free — the transform composes down the tree in i64, and rebasing happens at consumption (§4).
  This is **S4**.
- **Free / non-lattice affine + voxel resampling stays explicitly deferred** (see Deferred).

#### 3g. Orphan-override policy (S5) — PRESERVE-AND-FLAG, never silently drop

When a part's base producer changes (cylinder → box) so that a force-on override now sits where the
producer already fills, or a force-off sits where the producer no longer fills:

- Overrides are **address-anchored, not occupancy-anchored**, so they are *kept verbatim* across a
  producer change — the layer fold simply re-evaluates. A force-off over now-absent producer voxels
  becomes a **no-op but is retained** (so re-adding the producer restores intent); a force-on over
  now-present producer voxels is **redundant but retained**.
- Overrides that fall **outside the new producer's AABB entirely** are flagged `orphaned` (surfaced
  in the UI as "N detached sculpt voxels") and remain force-on (they still render — a sculpted spur
  beyond the base shape is legitimate). The user gets an explicit "prune orphaned overrides" command
  (itself undoable). **Policy: never auto-delete; preserve, flag, offer one-click prune.** This is
  **S5**.

### 4. Chunked sparse streaming store: anisotropic tiling, rebase-at-consume, unified residency

**One residency layer, not two.** `ChunkResolveCache`'s second LRU (stacked on `DiskChunkStore`'s
LRU) is removed. There is a single `ResidencyManager` owning the resident set; `DiskChunkStore`
becomes its spill backend, not an independent cache. Eviction returns GPU buffers to a pool (ADR
0002 borrowed technique 4).

**Anisotropic tiling (constraint 5).** Chunks remain `CHUNK_BLOCKS`-cubed for *meshing/storage*, but
**residency is governed by XZ distance with full-Y columns kept resident**:

```rust
struct ResidencyPolicy {
    xz_radius_blocks: i64,   // stream/evict horizontally by camera XZ distance
    keep_full_y: bool,       // a resident XZ column keeps ALL its Y chunks (Y is bounded ~1-2k)
}
```
The resident set is the set of XZ columns within `xz_radius_blocks` of the camera focus, each
holding its full (bounded) Y stack. This matches the >10k XZ / ~1–2k Y profile and makes horizontal
out-of-core the foundation, not an afterthought.

**Absolute-i64 addressing; rebase at CONSUMPTION, store rebase-free (fixes the cache-nuke).** Today
chunks are stored **pre-rebased against the composite recentre** and the cache rebinds (clears) on a
floating-origin / density change (`chunk_cache.rs::rebind_if_changed`), so any extent-changing edit
that moves the recentre invalidates *everything*. The fix:

- Chunks are stored and keyed by **absolute i64 chunk coordinates** — never against a moving
  recentre/origin. A chunk's stored content (now chunk-local integers per §3a) is independent of the
  scene's current extent.
- Rebasing to f32 happens **only at consumption** (mesh upload / render), as
  `chunk_world_origin_i64 − camera_floating_origin_i64`, downcast per frame (ADR 0002). The store
  never re-rebases.
- **The floating origin is sticky/quantized and decoupled from composite extent.** It is NOT
  recomputed per edit from the composite AABB (which is what makes the current recentre move on every
  extent change and re-rebase all resident meshes). It snaps to a quantized grid near the camera
  focus and only re-bases when the camera has moved far enough that f32 precision demands it — so an
  edit never invalidates resident meshes merely by growing the bbox.
- **Density stays a cache-clearing key.** Removing the *origin* half of the rebind trigger does not
  remove the *density* half: a chunk's voxel extent is density-dependent, so a density change still
  clears + re-binds the store (`chunk_cache.rs::rebind_if_changed` keeps its density guard).
- **Consequence:** a sculpt stroke that grows the bounding box adds/edits only the chunks it touches;
  no other chunk's stored coordinate changes, so **the cache is not nuked**. This is **S8** (and is
  what makes **S1**'s far edit O(stroke)).

**Incremental re-mesh is finally CONSUMED.** `incremental_rebuild_plan` (computed-but-ignored today)
becomes the load-bearing path: an edit's world-AABB → set of dirty `(chunk_coord)` → those chunks
re-resolve (fold layers via `resolve_into`, §3d) and re-mesh; **all other chunks keep their cached
buffers**. The renderer stops re-meshing wholesale on edit. Region-scoped consumers (diameter, layer
scrubber, `.vox` export) read the per-chunk store over the active region, never an assembled
whole-grid. Scrubbing the layer-band is a **per-fragment clip on absolute-Y** (ADR 0002 matrix row) —
no re-mesh — so a 10k-XZ scrub is interactive. This is **S7**.

### 5. Serialization, versioning & migration (the published document interface)

Project files are a **versioned, additive document format**; configs remain free to break (memory:
no-config-back-compat).

```rust
#[derive(Serialize, Deserialize)]
struct ProjectFile {
    format: &'static str,      // "voxelworker.project"
    version: u32,              // 2 (this foundation). v1 had no tag → detected & migrated.
    doc: DocV2,                // arena of defs/nodes (NodeId), layer stacks, sculpt overrides
}
```

- **Format:** a top-level header (magic + `version`) + the document body.

- **Sculpt overrides use a DEDICATED sparse override codec (G2), not `chunk_storage::compress`.**
  The existing `chunk_storage::compress`/`decompress` consume an **f32 `VoxelGrid`** and
  **debug-assert a uniform per-axis `centre_fraction`** (`chunk_storage.rs`) — an invariant that
  producer-resolved grids satisfy but that **integer force-on / force-off deltas do NOT** (a delta is
  not a centred resolved grid). So overrides keep the *conceptual* "sparse, chunk-keyed storage"
  reuse but get their own codec:
  - **force-on:** a per-chunk set of **sorted integer voxel keys** (chunk-local `[u16; 3]` packed to
    a single sorted `u64` or delta-varint key list) + a **material palette** (palette index per
    force-on key).
  - **force-off:** a **separate sorted key set** — force-off MUST be its own set, **NEVER a reserved
    palette slot / sentinel material**. A sentinel material would re-pollute the very `material_id`
    field §3c is cleaning, re-introducing a "meaning packed into a data field" leak.
  - **byte order is deterministic** (keys ascending, palette in first-seen order) so the encoding is
    canonical, and the codec has its **own round-trip byte-identity test** — that test **is** the
    **S9** golden (encode → bytes → decode → byte-identical override layer + bytes stable across
    runs).
  - Large overrides stay compact (sorted keys + delta-varint + palette) and reload byte-identical.
  This is **S9**.

- **Migration is a forward chain** `migrate_v1_to_v2(...) → migrate_v2_to_v3(...)`. v1 (no version
  tag, single-geometry / `NodePath`-positional scene) loads via a v1 reader that builds a v2 doc: one
  `AssemblyDef` with one `Producer` layer, fresh `NodeId`s minted. The loader **refuses to silently
  reinterpret** an unknown future version (hard error with a clear message) rather than corrupt. This
  is **S6**.
- Each migration step has a **fixture test**: a checked-in v1 file → load → assert the resulting v2
  doc. The published-interface contract is thus regression-guarded.

### 6. Headless `AppCore` + testing strategy (the keystone)

**The single biggest correctness win: one `AppCore`, two shells.** `shot.rs`'s ~1700-line parallel
render path is **deleted**; `bin/shot` and `bin/main` both construct the same `AppCore` and the same
`gpu` render-item consumer, differing only in window vs offscreen surface.

```rust
pub struct AppCore {
    doc: Doc, edits: CommandStack, store: Store, camera: OrbitCamera,
}
impl AppCore {
    pub fn apply(&mut self, intent: Intent) -> Result<(), EditError>; // builds+applies a Command
    pub fn undo(&mut self); pub fn redo(&mut self);
    pub fn render_items(&mut self, frustum: &Frustum) -> Vec<ChunkRenderItem>; // opaque to gpu
    pub fn occupancy_region(&mut self, region: Aabb64) -> Occupancy;           // fog/scrubber/export
}
```

- **The golden net becomes real:** because the screenshot tool now drives the *actual* `AppCore`,
  the golden PNGs test the real interactive path. Interactive bugs can no longer escape into the gap
  between two implementations.
- **Far-scene goldens are added BEFORE the precision refactor (G3).** The current golden harness
  (`tests/golden.rs`) covers only **near-origin** scenes — `--demo-village`, single small shapes,
  default placements — and `resolve_chunk_rebased` is documented bit-identical **only for near
  scenes** (`scene.rs`: "for a near scene the result is bit-identical … while a far-placed scene
  renders with no f32 jitter"). Before the keystone precision refactor (the §3a chunk-local payload +
  the §4 rebase-free store), we **add XZ~10k far-scene golden fixtures first** (e.g. a
  `--demo-village` placed at `offset_blocks ≈ [10000, 0, 10000]`), so the precision refactor ships
  **guarded** rather than unverified. This is **Phase D0**, a gate before the store/payload phase.
- **Test layers:** (a) pure unit tests on `core_geom`/`doc`/`edit`/`store` (command inverse
  round-trips, layer-fold correctness, the override-codec byte-identity round-trip = S9, migration
  fixtures); (b) the golden-PNG harness over `AppCore` for the render feature matrix (ADR 0002),
  **now including the far-scene fixtures**; (c) **stress-case integration tests S1–S10** as named
  tests against `AppCore` headless (no GPU needed for S1/S2/S5/S6/S8/S9/S10; S4/S7 add a golden).

### 7. Threading / ownership model (S10)

- **`AppCore` owns the document and the command stack on the main thread.** It is the **sole writer**
  of `doc`/`edits`. No background thread mutates the document.
- **Background work is read-only or produces detached artifacts handed back via a completion queue**
  (ADR 0002 borrowed technique 1): asset scanning (`scan_worker`) and chunk meshing run on workers,
  consuming an **immutable snapshot** (Arc) of the relevant def/chunk inputs, returning
  `(chunk_coord, MeshData)` to the main thread for GPU upload.
- **Edit-during-mesh consistency uses a per-scope monotonic revision stamped into the cache ENTRY,
  not a global doc version.** A global doc version would mark *every* resident chunk stale on any
  edit, starving far-chunk re-meshing (a far edit would force near chunks to re-mesh and vice-versa).
  Instead each cache **entry** carries a per-scope (per-chunk / per-def-scope) monotonic revision; a
  completion is stamped with the revision it was computed against; a stale completion (revision older
  than the entry's current revision) is **discarded** and the chunk re-queued. A chunk is marked
  **clean only when an epoch-matched result is INTEGRATED** (not when the job is dispatched). The
  command stack is therefore never blocked by, and never inconsistent with, background
  meshing/scanning. This is **S10**.

## Acceptance criteria — stress-case walkthrough (S1–S10)

| # | Stress case | How the design satisfies it |
|---|---|---|
| **S1** | Undo a single sculpt stroke on a FAR chunk (XZ ~10k) | `SculptStroke` stored a **sparse inverse delta** (only overwritten `(addr→prior)` pairs); undo applies it, dirtying only the stroke's chunks → `incremental_rebuild_plan` re-meshes just those via `resolve_into(chunk_box)` (§3d). Chunk-local integer payload (§3a) + absolute-i64 store means the far edit is exact and never touched other chunks. **O(stroke), exact restore.** (§2, §3a, §3d, §4) |
| **S2** | Edit a def with ~50 instances | One `Command` mutates the `AssemblyDef`. Instances *reference* the def, so propagation is free; invalidation fans out to the 50 instance placements' world-AABBs (via `world_aabb_blocks`, §3d) → only **intersected chunks** re-mesh. **One undoable command.** (§1, §2, §3d, §4) |
| **S3** | Sculpt an INSTANCE | `EditTarget::Assembly` disables geometry tools; the command factory returns `EditError::InstanceNotEditable`; UI offers **"Open definition"** → `EditTarget::OpenDefinition(def)`. **Disallowed, redirected.** (§1) |
| **S4** | Move + 90° rotate a SCULPTED part | Overrides are **part-local**; the `NodeTransform` gains a **24-orientation lattice rotation enum** (§3f, its own milestone) that composes in i64 and rebases at consumption; resampling is an exact index permutation, serialization stays byte-stable. **Overrides move with the part; positions exact.** (§3f, §4) |
| **S5** | Change base producer with anchored overrides | Overrides are **address-anchored, not occupancy-anchored**: kept verbatim, redundant/no-op ones retained, out-of-AABB ones **flagged `orphaned`** and rendered; explicit undoable **prune** offered. **Never silently dropped.** (§3g) |
| **S6** | Load v1 file into v2 build | Header `version` detected (v1 = absent tag); `migrate_v1_to_v2` builds a v2 doc (one def, one producer layer, fresh `NodeId`s); unknown future version = hard error, never silent corruption. **Migrates cleanly.** (§5) |
| **S7** | Scrub layer-band on 10k-XZ scene | Band clip is a **per-fragment discard on absolute-Y** — no re-mesh; consumers read the **region-scoped** store, not a whole-grid. **Interactive, no full re-mesh.** (§4) |
| **S8** | Sculpt stroke that GROWS the bbox | Chunks keyed by **absolute i64** coords with chunk-local integer payload (§3a); store is rebase-free; the floating origin is **sticky/quantized** (§4) so growing the extent does not move it; only touched chunks change. **Cache not nuked.** (§3a, §4) |
| **S9** | Save/share large sculpt overrides | Overrides use the **dedicated integer-delta codec** (§5): sorted force-on keys + palette, a **separate** sorted force-off set (no sentinel material), deterministic byte order, **own byte-identity round-trip test**. **Compact, exact.** (§5) |
| **S10** | Scan assets + mesh while editing | `AppCore` is sole document writer (main thread); workers consume **Arc snapshots**, return completions stamped with a **per-scope cache-entry revision**; a chunk is clean only when an **epoch-matched result is integrated**; stale completions discarded. **Command stack stays consistent; far chunks not starved.** (§7) |

## Migration sequence (incremental, behind the golden net)

Each phase is a **green checkpoint** — the app is built, golden-verified, and shippable. The
golden-PNG net (already DONE, ADR 0002 E0) is the guard throughout; **nothing changes pixels except
where a feature row explicitly allows it.** Reuse is maximized; the only outright *replacements* are
the `shot.rs` parallel path and the second LRU.

**Phase A — Layering + `AppCore` extraction (no behavior change). [ships first; gates everything]**
1. Move `CHUNK_BLOCKS` → `core_geom`; move domain types out of `panel.rs` into `doc`; break
   `chunk_cache→vox_export` by relocating both under the new layering. (Pure moves; goldens prove
   no change.)
2. Extract `AppCore` from `WindowedState`/`Scene`: pull the resolve/chunk/frame-math out of the
   god-objects into `app_core`/`store`. `WindowedState` becomes a thin shell.
3. **Re-point `bin/shot` at `AppCore`; delete its parallel render path.** *(This makes the golden net
   real — the single highest-leverage step; do it as early as possible so every later phase is
   guarded by the actual path.)* **Reuse:** cuboid mesher, fog, camera math, `DiskChunkStore`.
   **Replace:** `shot.rs` render copy.

**Phase B — Identity: `NodeId` arena (no user-visible change).**
4. Introduce `NodeId` + id-keyed arenas; selection/commands key on `NodeId`; demote `NodePath` to an
   ephemeral tree-render projection. **Parallelizable** with C once A lands. **Reuse:** existing tree
   widget (recompute paths on render).

**Phase C — Command stack + intent→command wiring.**
5. Add the `Command` trait + `CommandStack`; route the existing `PanelResponse` intents through
   `AppCore::apply`/`undo`/`redo`. Convert today's in-place mutations to commands incrementally
   (move/add/delete/rename first — each a small command with an obvious inverse).

**Phase D — Store unification + chunk-local payload + rebase-free + incremental remesh consumed.**
- **D0 (GATE — G3): add XZ~10k far-scene golden fixtures FIRST.** Before any payload/store change,
  add far-scene goldens (e.g. `--demo-village` at `offset ≈ [10000, 0, 10000]`) so the keystone
  precision refactor ships **guarded**. The current goldens are near-only (`tests/golden.rs`).
6. **Chunk-local integer payload (G1, prerequisite):** change `Voxel` from f32 `world_position` to
   chunk-local integer `local` + material; the absolute i64 origin lives only in the chunk key; f32
   is produced only at consumption. (Unblocks exact absolute storage and the §5 override codec.)
7. Collapse the two LRUs into one `ResidencyManager` over `DiskChunkStore`; key chunks by **absolute
   i64**; stop pre-rebasing on store, rebase at consumption only; make the floating origin
   **sticky/quantized** and decoupled from composite extent; **keep density as a cache-clearing
   key**. **(Fixes S8/cache-nuke.)**
8. **Consume `incremental_rebuild_plan`:** renderer re-meshes only dirty chunks. (This completes ADR
   0002 #20 Step 4, the deferred per-chunk GPU residency.) **Reuse:** `incremental_rebuild_plan`
   (finally wired), `DiskChunkStore`, the per-chunk mesher.

**Phase E — Anisotropic residency.**
9. Add `ResidencyPolicy` (XZ-radius streaming, full-Y columns). Stress-test >10k XZ. **(D2/S7.)**

**Phase F — Compositor: producer registry (windowed) + layers + rotation milestone.**
- **F0 (PRECONDITION — G5): producer trait redesign.** Replace `VoxelProducer::resolve(&mut grid)`
  with `resolve_into(chunk_box, out, frame)` + `world_aabb_blocks() -> Option<Aabb>`; build the
  registry; register `SdfShape` (bounded) and `DebugClouds` (unbounded → wholesale-clear fallback).
  **This precedes the sculpt and residency-dependent incremental claims** (S1/S2/S8) — without the
  chunk window, "re-resolve dirty chunks" is O(part_volume × dirty_chunks). **Parallelizable** with
  D/E (different module) but must land **before** Phase G.
10. Exercise `CombineOp::{Subtract,Intersect}` in the layer fold; **remove `GRID_OVERLAY_BIT` from
    `material_id`** — move the overlay flag to a **per-draw uniform** (not a per-vertex attribute),
    keeping `decompose_into_boxes` representation-agnostic; update both shaders. Golden-gated.
11. **Rotation milestone (G4 — its own milestone, golden-gated):** promote `NodeTransform` with a
    **24-orientation `LatticeOrientation` enum** (type-level, not general affine); add
    (a) exact index-permutation resampling, (b) rotated-chunk conservative-cover fan-out,
    (c) rotation-aware AABB-skip + spatial-index fingerprint. **(S4.)**

**Phase G — Sculpt override layer.**
12. Add `SculptOverride` (sparse, chunk-keyed) as the last layer, using the **dedicated integer-delta
    codec** (sorted force-on keys + palette, separate sorted force-off set — §5), **not**
    `chunk_storage::compress`; `SculptStroke` command with sparse inverse delta. **(S1/S2.)** Add the
    orphan policy + prune command. **(S5.)** (Depends on F0's `resolve_into` for O(stroke).)

**Phase H — Part/assembly edit modes.**
13. `EditTarget` state machine; instance-edit rejection + "Open definition". **(S3.)**

**Phase I — Serialization v2 + migration.**
14. Versioned `ProjectFile`; the override codec's **byte-identity round-trip test (= S9)**;
    `migrate_v1_to_v2` + fixture tests. **(S6/S9.)**

**Phase J — Threading hardening.**
15. **Per-scope cache-entry revision** stamping (not a global doc version); Arc snapshots; chunk
    clean only on **integrated, epoch-matched** result; stale-discard. **(S10.)** (Much of the worker
    scaffolding exists in `scan_worker`/ADR 0002 async meshing — this formalizes the invariant.)

**Parallelizable:** B∥(start of C); F0 + registry (step 9/F0) ∥ D/E; the migration fixtures (I) can be
authored anytime after B. **Sequential bottlenecks:** A3 (real golden net) gates everything;
**D0 (far-scene goldens) gates D6/D7 (the payload + rebase-free store)**; D (rebase-free store) +
**F0 (producer `resolve_into`)** both gate G (sculpt); D also gates E (residency); the rotation
milestone (F11) gates S4.

## Consequences

**Better**
- The convergent backbone exists end-to-end: command → sparse delta → per-chunk invalidation →
  incremental re-mesh, reversible and horizontally scalable.
- Sculpting, ordered add/subtract layers, lattice rotations, and references all become **data
  changes**, not re-architectures — the foundation's whole point.
- **The golden net tests the real app** (one `AppCore`) **and now covers far scenes**, closing both
  the parallel-path gap and the never-tested far-precision case.
- The per-voxel payload is **chunk-local integer**, so absolute-i64 storage is exact at XZ~10k, not
  merely "exact near the origin".
- God-objects decompose into a strictly layered, AI-navigable codebase (D5); adding a producer/
  command/tool is a registration, not a 3-arm edit.
- Far-edge edits, big scenes, and bbox growth no longer nuke the cache (sticky/quantized origin); undo
  is O(stroke) (chunk-windowed `resolve_into`).
- Shared project files have a real versioned contract with a byte-identity-tested override codec and
  tested migration.

**Costs more**
- Up-front extraction (Phase A), the `NodeId` arena (Phase B), the **chunk-local payload migration
  (D6)**, and the **producer-trait redesign (F0)** are pure-foundation work paid before the payoff.
- Every mutation now goes through a command (more boilerplate per edit); each command must define a
  correct inverse (tested).
- Maintaining the versioned format + the dedicated override codec + migration chain is ongoing tax on
  the document interface.
- The single-`AppCore` constraint means the windowed and headless shells must stay genuinely thin.
- Rotation is a whole milestone (conservative-cover fan-out + rotation-aware indexing), not a
  one-liner.

**Explicitly deferred**
- **LOD / impostors** — seam preserved via the `(chunk_coord, lod)` key (ADR 0002 O7); not built.
- **Per-chunk cross-block cuboid merge** and an optional 2D greedy pass over cuboid faces (ADR 0002,
  pursue only if GPU-bound).
- **Free (non-lattice) rotation/scale with voxel resampling** — the rotation milestone (§3f) ships
  only the 24 axis-aligned lattice rotations as a type-level enum (exact index permutation,
  byte-stable serialization); arbitrary affine + interpolating resampling is a later layer.
- **Multi-material picker for parts / VS palette block table** beyond the current procedural set.
- **In-part parametric construction tree** (booleans/lathe/array history) — the ordered layer stack
  is the seam it will slot into.

## Alternatives considered

**Non-winning emphases (folded in rather than chosen wholesale):**
- *"Compositor-first / producer-registry-first."* Strong on extensibility but does not by itself fix
  the cache-nuke, undo, or the golden-net gap. **Grafted** as Phase F, with its `resolve_into`
  precondition (F0) sequenced before sculpt, after the store and `AppCore` fixes that unblock it.
- *"Store-first / streaming-first."* Correctly identifies the rebase-free + single-residency fix as
  load-bearing (Phase D, the gate for sculpt/residency), but shipping it before `AppCore`/`NodeId`
  would build on god-objects. **Grafted** as Phase D, after A/B — and itself gated by the far-scene
  goldens (D0) and the chunk-local payload change (D6).
- *"Identity/command-first."* Right that `NodeId` + the command stack are prerequisites for undo and
  references (Phases B/C), but alone they don't scale or sculpt. **Taken as the early backbone**, then
  extended by D–G.

**Ruled out:**
- **ECS (entity-component system).** Rejected (consistent with ADR 0001): over-built for a fixed,
  small set of producers/parts; an id-keyed arena + producer-registry trait gives the
  extensibility we need with far better navigability for a solo dev + AI agents (D5). ECS would also
  fight the part/assembly *definition vs reference* model, which is a graph, not a flat entity soup.
- **Event-sourcing / CRDT for undo or document state.** Explicitly forbidden by locked constraint 3
  (no real-time collaboration, ever). A linear command stack with inverse commands is simpler,
  O(stroke) for sculpt undo, and trivially serializable. Event logs would bloat shared files and
  complicate migration for zero benefit here.
- **World-space sculpting / world-anchored overrides.** Rejected: it breaks the part/assembly rule
  (overrides must live in the definition's local frame to propagate to instances — S2) and breaks S4
  (overrides would not follow a moved/rotated part). Part-local, address-anchored overrides are
  mandatory for both fidelity dimensions (D3) and the orphan policy (S5).
- **Reusing `chunk_storage::compress` for sculpt overrides.** Rejected: `compress` consumes an f32
  `VoxelGrid` and debug-asserts a uniform per-axis `centre_fraction` (`chunk_storage.rs`), which
  integer force-on / force-off deltas do not satisfy. A dedicated integer-delta codec (sorted keys +
  palette + separate force-off set) is required; only the *concept* of sparse chunk-keyed storage is
  reused.
- **Force-off as a reserved palette slot / sentinel material.** Rejected: it re-pollutes the
  `material_id` field with a non-material meaning — the same class of leak as `GRID_OVERLAY_BIT` that
  §3c removes. Force-off is a **separate sorted key set**.
- **f32-absolute per-voxel payload in an absolute-i64 store.** Rejected: at XZ~10k the f32 centre is
  lossy, so absolute storage in f32 is unsound. The payload must be chunk-local integer (§3a) with
  the absolute origin in the chunk key only.
- **Keep two LRUs / keep the recentre-pre-rebased store.** Rejected: the second LRU is redundant
  bookkeeping over `DiskChunkStore`, and recentre-pre-rebased storage with an extent-coupled floating
  origin is the direct cause of the extent-change cache-nuke (S8). Absolute-i64 + sticky/quantized
  origin + rebase-at-consume removes both.
- **Global doc version for worker-staleness.** Rejected: it marks every resident chunk stale on any
  edit, starving far-chunk re-meshing. A per-scope cache-entry revision (integrated, epoch-matched)
  is used instead (§7).
- **Keep `shot.rs` as a parallel render path.** Rejected outright: it is the reason the golden net
  tested a copy; a shared `AppCore` is the keystone of this entire ADR.
