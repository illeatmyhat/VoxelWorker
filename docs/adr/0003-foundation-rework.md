# ADR 0003 — Foundation rework: parts, sculpt, command journal & streaming store

- **Status:** Accepted (2026-06-29 — the keystone decisions are accepted and committed-to; the
  shipped portions (units §3f(0), the Sketch system §3i, chunk-windowed `resolve_into` §3d, the
  `Intent` door §6a / command stack / `shot --replay`) prove the direction. Remaining unbuilt pieces
  — sculpt, the absolute-i64 store, rotation (§3f G4), async (§7) — are tracked as issues, not as an
  open proposal.)
- **Date:** 2026-06-27
- **Supersedes / extends:** consolidates and supersedes the open growth-path clauses of
  [ADR 0001](0001-scene-graph-parts-and-tools.md) (composition beyond union; transforms beyond
  translation; the sculpt overlay; persistence) and [ADR 0002](0002-engine-streaming-meshing.md)
  (E5 out-of-core wiring, #20 Steps 2/4). ADR 0001's scene-graph/producer decisions and ADR 0002's
  cuboid-mesher/coordinate/streaming decisions **stand**; this ADR is the layer that makes them
  *editable, sculptable, undoable, share-serializable, and horizontally streamable* without a
  big-bang rewrite.

## Context

The app is feature-complete for its first workflow: a Vintage Story chiseling planner (wgpu 29 /
egui 0.34 / winit 0.30) that composes parametric shapes into a 3D model and exports `.vox`. The
project is **V0 pre-alpha** — nothing here is a shipped "v1"/"v2", and **breaking changes are
allowed until further notice**. We are doing a **foundation rework before hundreds of features
stack on top**, the largest of which is **AI agents composing arbitrarily complex buildings**
through the `Intent` boundary (§6a). Correctness and extensibility dominate effort/token
minimization.

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
4. **Users save & share PROJECT FILES.** The document format carries a **magic header + a
   version/epoch tag from day one** (cheap insurance) and the loader **hard-errors on an
   unrecognized tag** — it never silently misreads. But the project is **V0 pre-alpha: no migration
   code is written yet, and pre-alpha project files may break freely** (the same posture as config —
   memory: no-config-back-compat) until we declare a stable format. The tag is the seam that lets us
   *introduce* migration the day we leave pre-alpha; it is not a promise of cross-version
   compatibility now.
5. **Anisotropic large scenes:** routinely >10,000 blocks horizontal (XZ), but vertical (Y) bounded
   to ~1,000–2,000 blocks. **Out-of-core horizontal streaming is foundational** — tile by XZ
   distance, keep full-Y columns resident. At this extent f32 precision dies, so the existing
   **i64-subtract-before-f32-downcast** rebasing (ADR 0002 Decision 2) is mandatory; a far-edge edit
   must NOT re-resolve the world.

### Foundation seam rulings surfaced by the architecture gap sweep (F1–F6)

A 9-lens **architecture gap sweep** ([`docs/design/architecture-gap-sweep.md`](../design/architecture-gap-sweep.md))
was run against ADR 0003/0004 and surfaced six **foundation data-model seams + scope rulings** that
are cheapest to pin **while 0003 is still Proposed** — the same way the joint-arity fix in §1 was
driven by ADR 0004's stress-test. They are decided with the product owner and woven into the relevant
sections below; this is the index:

- **F1 — order-48 instance transform (mirror symmetry).** Widen the stored orientation from the 24
  proper rotations to the **full order-48 signed-permutation group** (add a handedness/reflection
  bit), unlocking bilateral mirror symmetry. Still an exact index permutation. (§3f.)
- **F2 — typed `BlockAttrs` + rotation algebra + block-entity side-table + world-origin export.**
  Define `BlockAttrs` as a typed per-`block_id` state schema, specify how it composes with the F1
  transform, add a neighbor connection-resolve pass, a sparse block-entity side-table, and a
  world-origin export contract. (§3a/§3a-bis, §5.)
- **F3 — terrain is a MUTABLE layer with controlled coupling.** The product owner chose **writable
  terrain**: terrain is a first-class mutable layer, and a **controlled producer↔terrain coupling**
  (sampling `GroundHeightAt` to tie in to live grade) is permitted; no general producer↔producer
  free-for-all. (See ruling below + Consequences/Alternatives.)
- **F4 — Datums / levels / grids + `HostedOnDatum`.** Scene-owned named reference geometry, may be
  terrain-relative. (§1.)
- **F5 — instance param overrides + a Def TYPE tier.** Relax "instance = transform-only". (§1.)
- **F6 — STATIC-function scope ruling.** Movable/stateful mechanisms are inert annotations for export
  fidelity; live kinematics is out of scope. (See ruling below.)

**F3 ruling — writable terrain + controlled coupling.** Terrain is a **first-class mutable layer**,
not a frozen import. A producer **may** sample a `GroundHeightAt`-style terrain query to blend/tie-in
to live grade (cut/fill/terrace/berm/excavate semantics). This is the **only** sanctioned
producer↔producer-class coupling — a *controlled* terrain query, **not** a general producer→producer
coupling free-for-all. This **supersedes the earlier "terrain read-only + no producer↔producer
coupling" stance** (see Alternatives/Consequences). The specific cut/fill/terrace/berm/excavate
**PRODUCERS are ADR 0005**; here we only establish that terrain is mutable and the coupling is
allowed. The terrain *import format* remains the open research item (unchanged).

**F6 ruling — the planner is STATIC.** Movable/stateful mechanisms (doors that open, drawbridges,
lifts, portcullises, windmill cap yaw, furnaces/gears) are modeled as **inert ANNOTATIONS** — a
`block_id` + the F2 `BlockAttrs` (e.g. `hinge-left/closed`) + the F2 block-entity side-table —
carried purely for **VS export fidelity** so that **VS supplies the actual behavior**. A static
*pose* (a windmill cap at a fixed angle) is just the F1 transform. The concrete form of an inert
annotation for an interactable that has **no clean static-voxel representation** is a
**placeholder / proxy entity** with two parts: (1) **proxy geometry** — a recognizable static
stand-in so a human reading the plan grasps the *feel* ("a drawbridge goes here"); and (2) a
**substitution payload** carried in the F2 side-table (`BlockEntity::Substitution` — the real
`target` block/entity + attrs + orientation + params) so that **on export the proxy is substituted**
for the real VS block/entity, or, failing that, a human has the exact data to place it manually. The
planner itself stays static: it stores the proxy and the substitution data; **VS (or the human on
substitution) supplies the real behavior**. **Live kinematics, state-machines, tick simulation, and
signal networks are explicitly OUT of scope** — a categorically different subsystem, not bolted into
the three-tier static model. This is the boundary; F2 carries the annotation, nothing here runs it.
The placeholder PRODUCER library, the distinct placeholder RENDER treatment, substitute-on-export,
and the `PlaceholderUnsubstituted` / `PlaceholderUnmapped` diagnostics are **ADR 0005**.

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

- **God-objects.** The **legacy `Scene` god-object** (today's monolithic `scene.rs::Scene`:
  document + selection + resolve/chunk/frame-math/gizmo engine — everything in one type);
  `WindowedState` (124 fields, 321-line `render()`, no controller/intent layer); `ChunkResolveCache`
  (a cache that accreted export/scrubber/renderer methods + a **second LRU** on top of
  `DiskChunkStore`'s). This ADR **dissolves the legacy `Scene`** and **reclaims the name**: the new
  clean, logic-free data layer is `scene::Scene` (§0/§1), and the legacy Scene decomposes INTO it
  alongside `store` + `app_core`. Throughout this ADR, "legacy `Scene`" / "today's monolithic
  `Scene`" means the god-object being dissolved; an unqualified `scene::Scene` means the new data
  layer.
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
  reconciled by comment-arithmetic (the half-block `leaf_lattice_shift` correction an *implicit-center*
  shape needs for odd sizes — **now resolved-by-design** once shapes are **corner/face-anchored point
  shapes** that make the shift identically zero by construction, see §3i); "extent-changing edit nukes
  the whole cache" (the
  `ChunkResolveCache` rebinds on a *floating-origin / density* change — `chunk_cache.rs::rebind_if_changed`
  — and chunks are stored pre-rebased against the composite recentre); ~~`incremental_rebuild_plan` is
  computed but not consumed (renderer re-meshes wholesale on edit)~~ **[RESOLVED 2026-06-29, #40 `9ff63c3`]:
  the live cuboid renderer now re-meshes only the dirty chunks (apron-dilated via `cuboid_incremental_plan`),
  wholesale only on a floating-origin shift / density change — see §4.**
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
               block-palette/material table, AND the streaming quantum CHUNK_BLOCKS (moved out of
               renderer.rs). Pure; no egui/wgpu/winit.
scene        : the document DATA layer — stable ids (NodeId/DefId), scene-graph node tree PLUS a
               relationship/constraint graph (nodes reference joints to other NodeIds, §1), part
               definitions, sculpt override layers, producer registry, serialization (tagged). The
               root type is `scene::Scene`. Clean & logic-free; reclaims the name from the dissolved
               legacy `Scene` god-object. Pure.
edit         : the command stack — Command trait + inverse, the linear journal, edit-mode/open-def
               state machine. Depends on scene + core_geom. Pure.
store        : per-part chunked sparse store, ChunkStore backend trait (DiskChunkStore is one impl),
               residency manager, override+producer compositor, resolve_region/resolve_chunk.
               Depends on scene + core_geom. Pure (no GPU); this is where ChunkResolveCache's data
               role goes.
app_core     : headless orchestrator (AppCore) — owns scene + edit + store + camera; applies
               intents→commands, answers spatial queries + diagnostics, drives invalidation→remesh,
               produces opaque per-chunk render items + occupancy. Pure of windowing; depends on all
               above.  <-- THE KEYSTONE
gpu          : wgpu mesh upload/draw, fog, atlas. Consumes AppCore's opaque render items.
ui/gizmos    : 3D-overlay interactive handles that hit-test in world/screen space — transform
               gizmo, the ViewCube widget, point/axis manipulators. They share a uniform
               hover/drag → Intent contract. Depends on app_core.
ui/panels    : egui 2D surfaces — inspector, tree, export, layers. Emit intents. Depends on
               app_core.
bin/main     : windowed shell (winit) = AppCore + gpu + ui/{gizmos,panels} + live control socket.
bin/shot     : headless shell = AppCore + gpu (offscreen), replays an Intent script. SAME AppCore.
```

Resolves: god-objects (the legacy `Scene` / `WindowedState` / `ChunkResolveCache` decompose into
`scene` + edit + store + app_core); wrong-way deps (`scene.rs→panel.rs` becomes `ui→app_core`;
`CHUNK_BLOCKS` relocates to `core_geom`; `chunk_cache→vox_export` becomes `store`/`vox_export` both
under `core_geom`). The shared `AppCore` retires `shot.rs`'s parallel render path. Note `ui` is two
sibling sub-layers — `ui/gizmos` (3D, world/screen hit-testing) and `ui/panels` (2D egui) — that
both depend only on `app_core` and both speak the same `Intent` boundary downward.

### 1. Identity, scene graph (+ relationship graph) & part/assembly modes

The `scene` layer is the clean data layer the legacy `Scene` god-object dissolves into. The root
type is `scene::Scene`. It is a **node tree PLUS a relationship/constraint graph** (see "Constraint
/ joint data seam" below).

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
- An `Instance(DefId)` node carries `NodeTransform` (promoted to the **order-48 signed-permutation
  orientation** — 24 rotations + 24 reflections, F1 — plus i64 translation — §3f, milestone-scoped),
  **and may carry a `KitParams` override bag (F5).** It still has **no** editable *geometry* surface
  (no per-instance sculpt); a `KitParams` override only re-parameterizes the referenced def's
  producers, it does not edit voxels on the instance.

**Instance param overrides + a Def TYPE tier (F5 — relaxing "instance = transform-only").** Today an
instance is transform-only, which forces a **def fork per size variant** (a 3-wide window and a
4-wide window become two unrelated defs). That is CAD/BIM table stakes to avoid, so the rule is
relaxed two ways, either or both usable:

```rust
pub struct Instance {
    pub def: DefId,
    pub transform: NodeTransform,            // order-48 orientation + i64 translation (F1)
    pub param_overrides: Option<KitParams>,  // F5: per-instance parametric override bag (None = use def defaults/type)
}
// A Def may additionally expose named parametric TYPE tiers (the "family" tier):
pub struct DefType { pub name: String, pub params: KitParams }   // e.g. Window::"3-wide", "4-wide"
// AssemblyDef carries `types: Vec<DefType>` (empty = a plain, non-parametric def).
```

- A `param_overrides` bag re-evaluates the def's **producers** with overridden `KitParams` at resolve
  (the parametric tier, §3b/§3d) — it is **not** a sculpt/override layer and does **not** touch the
  voxel-edit purity rule (sculpt still lives only in defs, §3e). A named `DefType` is a stored,
  shareable parametric variant; an instance may select a type and/or further override it.
- This **bends the "instance = transform-only" purity** of the Fusion rule (constraint 2)
  deliberately — exactly as `assembly_overrides` (below) already bent the "geometry lives only in
  definitions" purity. **Precedent already set in this same ADR.** Edited here so the seam exists; the
  KIT producers that *consume* `KitParams` are designed in ADR 0005, not here.
- `EditTarget` is either `Assembly` (place/move/pattern instances; geometry tools disabled) or
  `OpenDefinition(DefId)` (sculpt/producer tools enabled; instance placement disabled). The UI's
  "active context" is this enum, not a node.
- **Attempting a geometry edit on an instance is rejected at command construction** (the command
  factory returns `Err(EditError::InstanceNotEditable { def: DefId })`), and the UI offers "Open
  definition" (which switches `EditTarget` to `OpenDefinition(def)`). This is **S3**.

**Constraint / joint data seam (reserved now; solver is a future feature).** The scene carries
**joints — N-ary relationships referencing other nodes by `NodeId`** — the foundation already mints
stable ids, so the references are durable across undo/structural edits. We **reserve only the DATA
seam** now (it serializes with the document); the SOLVER is a future feature (see ADR 0004).
**Refined by ADR 0004's stress-test:** joints are **scene-owned and id-keyed** (not a per-node
`Vec`) and **n-ary** — a curtain wall must `Span` ≥2 bastions, and an ADR 0004 `Issue`/`SpatialQuery`
must be able to name an individual joint durably across undo; neither is expressible with a unary,
positionally-addressed per-node joint.

```rust
pub struct JointId(u64);          // stable, minted from a document-owned counter (like NodeId)
pub struct Node { /* …content/transform as above… */ }
// A joint is a relationship, not a property of one node → scene-owned, id-keyed, and N-ary.
pub struct Joint { pub id: JointId, pub refs: Vec<NodeId>, pub kind: JointKind, /* params */ }
// scene owns `joints: SlotMap<JointId, Joint>` alongside the node arena.
```

This makes `scene` a node tree **plus a relationship graph**, which is the common enabler for both
**human assembly-constraints** (Fusion-style mate/joint) and **agent-driven building** (an agent
expresses "this wall meets that floor" structurally, not by absolute coordinates). The
constraint/joint **SOLVER and the parametric architectural kit are features-on-top** and belong in
their own future design doc; we do **not** build the solver here — only the data seam so that
adding it later is not a schema cascade.

**Datums — scene-owned named reference geometry + `HostedOnDatum` (F4).** Joints relate part to part;
there is no **shared reference datum** that many parts attach to so that "move Level 3 up → everything
hosted on it follows" or "columns on grid A-3" works. The scene gains a **`Datum`** primitive — a
**level plane, grid line, or work axis** — as **scene-owned, named reference geometry** that **reuses
the `NodeId` arena** and **serializes with the document** (§5):

```rust
pub struct Datum {
    pub id: NodeId,                 // reuses the existing stable-id arena (durable across undo)
    pub name: String,              // "Level 3", "Grid A", "ridge axis"
    pub kind: DatumKind,           // LevelPlane { y } | GridLine { … } | WorkAxis { … }
    pub anchor: DatumAnchor,       // absolute, OR terrain-relative (a named site level vs imported grade — see F3)
}
// HostedOnDatum reuses the joint/hosting machinery: parts attach to a datum, not to each other.
// Expressed as a JointKind so it rides the existing scene-owned, id-keyed joint graph above.
pub enum JointKind { /* …existing kinds…, */ HostedOnDatum { datum: NodeId } }
```

- `HostedOnDatum` **reuses the joint/hosting machinery** (it is a `JointKind`, so it inherits the
  durable id-keyed, n-ary graph above) — moving a datum propagates to everything hosted on it through
  the same relationship fan-out, exactly like the future solver's other joints.
- A datum's anchor may be **terrain-relative** (a named site level defined relative to imported grade),
  which is the seam that ties datums to **writable terrain (F3)**. The terrain *import format* remains
  the open research item (unchanged).
- **Point-anchored shapes host on datums through this same seam.** A `ShapePoint::HostedOnDatum`
  reference (§3i) lets a shape's corner/face anchor name a datum, so the relational tier can drive
  sub-block carving geometry off a level/grid/axis exactly like it drives part placement.
- This is a **data seam only** — datums serialize and reference durably; the editing/solver UX that
  drives them is features-on-top.

**Assembly-scoped (scene-root) override layers — a world-frame, this-site-only patch.** Surfaced by
ADR 0004's intersecting-wall junctions (an intersection is between two *instances in the assembly*,
but the §3e sculpt layer is **part-local** and propagates to ALL instances of a definition — a
junction is placement-specific and must NOT change other instances), the scene gains override layers
that live **on the scene root, in the world/assembly frame**, not on a part definition:

```rust
// scene root carries assembly-scoped override layers, distinct from def-local sculpt:
pub struct Scene {
    // …defs/nodes arenas, joints SlotMap as above…
    pub assembly_overrides: Vec<Layer>,   // world-frame, this-site-only; does NOT propagate to defs
}
```

These **assembly patches** are address-anchored in the **world frame** (not a definition's local
frame), serialize with the document (§5), and are composited at assembly resolve (§4) — they patch a
specific spot in *this* scene (e.g. a wall-meets-bastion junction) and **never touch the part
definition**, so other instances of the same wall are unaffected. This **deliberately bends the
"geometry lives only in definitions" purity** of the Fusion rule (constraint 2) and is **clearly
distinguished from def-local sculpt**: def-local sculpt belongs to a definition and propagates to its
instances; an assembly override belongs to the scene root and propagates to nothing. The precedent is
**Fusion's assembly-context features** (edits made in the assembly that are scoped to that assembly,
not pushed back into the component). This is the third of the three Thread-4 seam items ADR 0004
requires (see §3b layer-stack semantics and §4 ordered composition).

### 2. Command / undo: linear journal with inverse commands

Per locked constraint 3 — a **linear stack**, no event-sourcing/CRDT.

```rust
pub trait Command {
    /// Apply to the document + store; return the inverse needed to undo exactly.
    fn apply(&self, scene: &mut Scene, store: &mut Store) -> Result<Box<dyn Command>, EditError>;
    fn label(&self) -> &str;            // for the undo menu
    fn coalesce_key(&self) -> Option<CoalesceKey>;  // e.g. one drag = one undo
}

pub struct CommandStack {
    done: Vec<Box<dyn Command>>,        // each entry stores the inverse captured at apply time
    undone: Vec<Box<dyn Command>>,
}
```

- **All document mutation flows through commands.** `ui/{gizmos,panels}` emit a serializable
  *`Intent`* (§6a); `app_core` turns it into a `Command`, applies it, pushes the captured inverse.
  This is the single consistency point for threading (§7) and the single automation/test/agent
  surface (§6a).
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

These compositor seams are the foundation's slot in a **three-tier authoring model** spelled out in
[ADR 0004 Decision](0004-agent-authoring-stack.md): **parametric** (producers — arbitrary geometry,
angles, curves, SDF-exact and re-voxelized at resolve; §3b/§3d/§3f case 1), **relational**
(joints/solver — how parts place and connect; §1), and **corrective** (override/patch layers — ordered
local block replacement, later-wins; §3b/§3e and the assembly-scoped layers added below). The
foundation provides the parametric and corrective tiers and reserves the relational data seam (§1);
ADR 0004 builds the relational solver and the parametric kit on top.

#### 3a. Per-voxel payload becomes chunk-local integer + categorical block-palette cell (load-bearing — G1 + materials FOUNDATIONAL)

This is a **prerequisite** of the absolute-i64 store (§4) and of the override codec (§5), not a
detail. Today `voxel.rs::Voxel` stores `world_position: [f32; 3]` (the absolute voxel centre, see
`voxel.rs:99-107`); `chunk_storage::compress` already has to *reverse-engineer* the integer index
out of that f32 (it debug-asserts a uniform per-axis `centre_fraction` and `floor()`s the position —
`chunk_storage.rs`). At XZ~10k the f32 itself is lossy, so absolute storage in f32 is unsound.

**The categorical material/attribute model is ELEVATED from deferred to FOUNDATIONAL.** Today a
voxel carries `material_id: u16` (`voxel.rs:106`) that is really a **3-value enum** (Stone/Wood/Plain
⇒ 0/1/2, clamped ≤2 — `voxel.rs:71-72`) **plus the `GRID_OVERLAY_BIT` (`1 << 15`) render flag jammed
into the same field** (`voxel.rs:82`, mirrored in two shaders). This is the wrong representation for
agent-composed buildings, and it cannot be retrofitted later without touching the payload, the
override codec, serialization, and meshing all at once — so it is foundational. The per-voxel cell
carries a **block-palette ID + attributes**, replacing both the 3-material enum and the
`GRID_OVERLAY_BIT`-in-`material_id` hack (the flag moves to a per-draw uniform, §3c).

**Decision:** the per-voxel payload is **chunk-local integer coordinates + a categorical
block-palette cell**:

```rust
pub struct Voxel {
    pub local: [u16; 3],          // voxel index WITHIN the chunk (u8 suffices at the app default
                                  // density 16 → chunk extent 64; u16 leaves headroom for denser)
    pub block_local_coord: [u8; 3],
    pub block_id: BlockId,        // categorical block-palette id (replaces the 3-material enum;
                                  // foundational — the rich VS palette CONTENT is the deferred part)
    pub attrs: BlockAttrs,        // TYPED per-block-id state schema (F2): orientation + variant +
                                  // neighbor-connection bits; rotates/reflects WITH the geometry
}
```

- The per-voxel categorical **capability** (a palette id + attributes, not a 3-value enum) is
  **foundational**; only the *full VS block-palette table / picker UI* (the rich palette content)
  stays a deferred feature. The `GRID_OVERLAY_BIT` is **not** in this payload at all — it is a
  per-draw uniform (§3c).

- The **absolute i64 origin lives ONLY in the chunk key** (`chunk_coord: [i64; 3]`), never in the
  per-voxel record. A chunk's stored content is position-independent of where the chunk sits.
- f32 is **produced only at consumption**, exactly as today's rebase does, as
  `(chunk_origin_i64 + local) − camera_floating_origin_i64`, the subtraction in i64, then a single
  f32 downcast (the existing `resolve_chunk_rebased` arithmetic, but now driven by the stored
  integer instead of recovered from an f32).
- This is what makes "absolute-i64 storage + i64-rebase-before-downcast" **exact** rather than merely
  "exact for near scenes". The region-scoped consumers (diameter / scrubber / `.vox` export) recover
  their integer indices directly from `local + chunk_coord` instead of `round(world_position + …)`.
- The categorical `block_id` rides through the store, override codec (§5), and meshing as an opaque
  palette index; `.vox` export maps it through the active block palette. This is exactly the per-voxel
  CAPABILITY the agent-composition work consumes (it builds with named blocks, not 0/1/2).

This change is sequenced as a **prerequisite of the absolute-storage phase** (Phase D), gated by the
far-scene goldens added first (§G3 / Phase D0).

#### 3a-bis. `BlockAttrs` is a TYPED per-`block_id` state schema + rotation/reflection algebra (F2)

`BlockAttrs` (§3a) must **not** stay an opaque "rotation/variant flags" payload: most real VS blocks
are **stateful** (stair facing, log axis, door hinge/open, fence/wall connectivity, slab half). An
opaque payload means a rotated instance keeps **stale facings**, neighbor connections are uncomputed,
and VS schematic export is **lossy by construction** — a functional gatehouse exports as dumb stone.
So `BlockAttrs` is pinned now as a **typed, per-`block_id` state schema**:

```rust
pub struct BlockAttrs {
    pub orientation: Option<LatticeOrientation>,  // facing/axis, in the SAME order-48 group as the transform (F1)
    pub variant: VariantFlags,                    // slab-half, stair-shape (inner/outer corner), etc. (per block_id)
    pub connections: ConnectionBits,              // neighbor-connection bits (fence/wall/stairs/glass-pane)
}
// The schema is keyed by block_id: a given block_id declares WHICH of these fields are meaningful.
```

- **Attrs compose with the instance transform (the load-bearing rule):** a block's `orientation`
  **rotates AND reflects WITH the geometry** under the F1 order-48 transform. When an instance is
  rotated 90° or mirrored, every stateful block's facing is **re-composed through the same
  signed-permutation**, so a rotated stair faces the rotated way and a mirrored door hinges the
  mirrored way. **Stale facings are the bug this prevents.** (`orientation` lives in the same order-48
  group as the transform precisely so the composition is one exact permutation, not a special case.)
- **Connection-resolve pass (the state analogue of WFC):** a **neighbor-aware** pass computes
  `connections` from adjacency (fences/walls/stairs/panes that connect to neighbors). It runs over
  resolved occupancy + attrs, deterministic and seed-free; it is the *state* counterpart to the
  material-fill WFC (which stays deferred, §Consequences). Specified here as a seam; the producers/UX
  that drive it are ADR 0005.
- **Sparse, address-keyed block-entity side-table (optional):** for VS **block entities / contents**
  (chests, signs, mechanisms — F6), the scene carries an optional **sparse side-table keyed by voxel
  address**, anchored **part-local** (same frame as the sculpt overrides, §3e) so it moves/rotates
  with the part, and **serialized ALONGSIDE the occupancy codec** (§5). It is sparse because almost no
  voxels carry an entity. An entry is one of:

  ```rust
  pub enum BlockEntity {
      Contents(EntityBlob),                  // a real VS block entity / contents (chest, sign, …)
      // F6: a PLACEHOLDER / PROXY for an interactable with no clean static-voxel form
      // (door that opens, drawbridge, lift, windmill, furnace/gears). The voxel occupancy at this
      // address is recognizable PROXY GEOMETRY (the "feel" — a human reads "a drawbridge goes here");
      // this payload is the SUBSTITUTION DATA — the real thing the proxy stands for, applied on export.
      Substitution {
          target: TargetKind,                       // the real VS block_id OR entity-type to place
          attrs: BlockAttrs,                        // F2 attrs of the real target (facing/variant/…)
          orientation: Option<LatticeOrientation>,  // order-48 (F1); composes with the instance transform
          params: EntityParams,                     // any extra params the real block/entity needs
      },
  }
  // A `Substitution` entry IS the "placeholder, not final" marker by construction: the variant itself
  // says this is a proxy, so export can substitute it and diagnostics can enumerate unsubstituted ones.
  ```

  On export (the world-origin export contract below) a `Substitution` proxy is **replaced** by its
  `target` + `attrs` + `orientation` + `params`, and the proxy's `orientation` composes through the
  instance transform exactly like any other stateful block (a mirrored drawbridge proxy substitutes a
  mirror-oriented real block, F1). If the exporter cannot place the target automatically, the
  substitution data is still the exact record a human needs to place the real thing manually. The
  placeholder PRODUCER library (recognizable proxy shapes per interactable), the distinct
  "this-is-a-placeholder" RENDER treatment (so proxies read as proxies — e.g. a per-draw proxy flag),
  the substitute-on-export logic, and the `PlaceholderUnsubstituted` / `PlaceholderUnmapped`
  diagnostics are **ADR 0005 consumers** — here we pin only the side-table `Substitution` schema.
- **World-origin export contract:** export defines a **build-anchor block coord → target-game world
  coord** mapping, and **asserts sub-block detail is phased to the document's `d`-grid**
  (`voxels_per_block`³ voxels/block — e.g. 16³ for Vintage Story, §3f(0)) so a round-trip lands on the
  target game's micro-block boundaries. The store's quantized floating origin (§4) is
  decoupled from any world-meaningful origin, so this anchor is what gives export a real, agreed
  world registration. (Detailed serialization in §5; the VS-native schematic *exporter* itself is a
  consumer feature in ADR 0005 — here we pin the schema, the rotation algebra, the side-table seam,
  and the export contract.)

#### 3b. Composition stops being union-only

A part definition resolves as an **ordered layer stack** — a `Vec<Layer>`, explicitly permitting
**N ordered override/patch layers, not just a single `Sculpt`**:

```rust
pub enum Layer {                                       // ENUM, deliberately (see traits-vs-enums)
    Producer { producer: Box<dyn VoxelProducer>, op: CombineOp, block: BlockId },
    Sculpt(SculptOverride),   // sparse force-on / force-off deltas, part-local (see 3e / §5)
}
pub enum CombineOp { Union, Subtract, Intersect }   // the existing enum, finally exercised
```

Resolution of a chunk = fold the layers in order: producers stamp under their `CombineOp`; each
override/`Sculpt` layer is applied in stack order (force-on sets occupancy + categorical `block_id`;
force-off clears).

**Layer-stack semantics are explicit: the stack is a `Vec<Layer>` of arbitrary length, applied in
order, and LATER LAYERS WIN.** The part-local sculpt layer and **multiple agent patch layers coexist
deterministically** — they are simply additional override layers later in the same fold, each
overwriting the cells it touches. (The `Vec<Layer>` representation already permits this; this clause
makes the *semantics* — N layers, ordered, later-wins — load-bearing rather than implicit, because
ADR 0004's junction/patch work stacks several corrective layers on one definition and on the assembly
root, §1. Surfaced by ADR 0004's requirements — see the joint-arity cross-reference.)

**Sub-block boolean composition IS the kit-authoring workflow (a supported authoring path).** Within a
part definition, child shapes (the point-anchored primitives of §3i) compose through these same ordered
`CombineOp` layers (`Union`/`Subtract`/`Intersect`) at **sub-voxel placement** (the within-block voxel
remainder, §3f(0)) — e.g. **subtract a slot a few voxels deep out of a corner**. This yields
**parametric sub-block features that are exact at the document's voxel resolution and land on the target
game's grid at export** — the **no-code alternative to writing a bespoke `VoxelProducer` per sub-block
part**. It is the within-part detail tier (§3i), and it is exactly what the kit-of-parts authoring
composes with: a part definition is a boolean stack of point-anchored primitives placed at sub-voxel
offsets, not a hand-written producer.

The **`Layer` kind** (`Producer` vs `Sculpt`) and **`CombineOp`** are kept as ENUMs **on purpose**
(see "Traits vs enums", §3h): both are closed, small, serialized, and exhaustively matched in the
fold — a trait object here would only add `dyn` indirection + `erased_serde` ceremony with no
extension benefit. The *open* extension point (producers) is the trait `dyn VoxelProducer` nested
inside the `Producer` arm.

#### 3c. The `GRID_OVERLAY_BIT` leak is removed — overlay flag becomes a per-DRAW uniform

The per-voxel cell carries the **categorical `block_id` only** (§3a); the `GRID_OVERLAY_BIT`
(`1 << 15`, `voxel.rs:82`, mirrored in `cuboid.wgsl` + `cuboid_loaded.wgsl`) is removed from the
data field entirely. The on-face-grid overlay is a **per-node render concern**, so it is carried as a
**per-draw uniform** (one bool per chunk-mesh draw, set from the owning node's
`grids.voxel_grid_on_faces`), **not** as a per-vertex attribute. Keeping it a per-draw uniform means
`cuboid::decompose_into_boxes` stays **representation-agnostic** (it never sees the flag), the data
field is freed for a real categorical id (not a 3-value enum sharing space with a render flag), and
the 2-shader mirror is killed. (A per-vertex attribute would re-pollute the mesh vertex format with
a render flag — the same category of leak we are removing.)

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
    fn serialize(&self) -> ProducerData;                                  // tagged format (§5)
}
```

- **Why this is a precondition, not routine.** The current trait
  `VoxelProducer::resolve(&mut grid)` (`voxel.rs`) re-evaluates the entire region every call. The
  per-chunk incremental / O(stroke) claims (**S1**, **S2**, **S8**) all depend on resolving *only*
  the dirty chunk window; without `resolve_into(chunk_box, …)` the "re-resolve dirty chunks" path is
  O(part_volume × dirty_chunks). This trait redesign therefore **sequences before** the sculpt and
  residency phases (it is its own early milestone — Phase F0).
  **[SHIPPED 2026-06-29] The chunk-windowed `resolve_into` is now IMPLEMENTED** (`af661cd` added it
  with no behavior change; `d2d4d96` switched per-chunk resolution to it, killing the ~11x per-chunk
  full-grid redundancy and restoring the per-chunk memory bound). Producers now resolve only a chunk
  window instead of the full grid per overlapping chunk. (Producer resolve is also parallelised via
  rayon, e.g. `dd08c81` for `SketchSolid`.) This is the seam ADR 0004/0005 consume and the one a future
  GPU view-resolve hooks behind — see [ADR 0006](0006-authoring-truth-and-gpu-boundary.md).
- `world_aabb_blocks()` returns `Option<Aabb>`: a **bounded** producer (`SdfShape`) yields a finite
  AABB so the spatial index / invalidation only touch intersecting chunks; an **unbounded** producer
  (`DebugClouds`) returns `None` and keeps the **wholesale-clear fallback** (its edits invalidate the
  whole resident region — acceptable because it is a debug/static field, not a sculpt target).
- Adding a producer = register one impl. The existing `SdfShape` and `DebugClouds` become the first
  two registrants. `VoxelProducer` is a **trait** (not an enum) precisely because it is an open,
  extensible set with uniform behavior and real polymorphism (§3h).

#### 3e. Sculpt overrides — part-local, address-anchored (codec specified in §5)

Sculpt overrides are part-local and anchored to **integer voxel addresses in the definition frame**.
They are stored as a **sparse, chunk-keyed** layer (the *concept* reused from the existing store), but
encoded with a **dedicated integer-delta codec** (§5 G2) — explicitly **not** a binary reuse of
`chunk_storage::compress`, which consumes an f32 `VoxelGrid` and debug-asserts a uniform
`centre_fraction` that integer force-on/force-off deltas do not satisfy.

**A sculpt/override layer is VOXEL-GRANULAR at the document's density (pinning the
previously-unspecified density-change case).** A sculpt layer's keys are integer voxel addresses at the
document's density `d` (§3f(0)). Since `d` is a **document-level attribute, fixed for the document's
life** (§3f(0)), a sculpt layer simply lives at that one density — there is no per-layer density
mismatch to reconcile during normal editing. The only density change is the **explicit, warned,
DESTRUCTIVE re-target** (§3f(0)): re-targeting the document to a different game/grid **reinterprets
every voxel-granular sculpt/override layer against a different grid**, which they do **not** survive
cleanly, so it is surfaced as a destructive operation ("this breaks voxel-granular detail"), never
silent and never a casual toggle. (An integer-multiple convenience rescale is a possible future
affordance, but the spec posture is **warn — this breaks the document**.) **Both placement (§3f(0)) and
resolved sculpt are voxel-granular at the document's own `d`** — the grid travels with the file, so
there is no density trap; the destructive re-target is the single, explicit exception.

#### 3f. Rotation is its own milestone (G4) — three-way angle model (geometry is NOT 24-limited)

**Arbitrary angle is a PRODUCER-PARAMETER concern, not a transform limit.** A common
misreading is that the 24-orientation lattice below limits *geometry* to axis-aligned shapes —
that an angled star-fort bastion or a curved wall is impossible. It is not. There are three
distinct cases, and only one of them is the 24-orientation case:

1. **Producer/parametric geometry can be ANY angle, exact.** An `SdfShape` (or any
   `VoxelProducer`) rotated by an arbitrary angle and **voxelized fresh at resolve** is
   exact-from-the-field: the SDF is sampled at the chunk lattice, so a 45° wedge, a 30° bastion
   flank, or a curve is produced exactly — just *staircased* on the voxel grid, which is inherent
   to voxels and is exactly what VS chiseling looks like. Arbitrary angles, curves, and tapers are
   therefore **producer parameters** inside a part definition, fully supported, with no transform
   change at all.
2. **Instance-transform ROTATION/REFLECTION of already-baked voxel data stays the order-48
   signed-permutation lattice group** (part (a) below). This was **never a geometry limit** — only a
   "losslessly rotate/mirror *already-voxelized* data" limit: an axis-aligned rotation **or
   reflection** is an **exact index permutation** (lossless, byte-stable, reversible), which is what
   keeps the sparse sculpt-override layer intact and the future joint solver integer-exact. The
   reflection half (the other 24 of the order-48 group) is what makes **bilateral mirror symmetry**
   — a left wing as a mirrored instance of a right-wing def — expressible at all (see (a)).
3. **Rotating a *sculpted* (non-parametric) part to a non-lattice angle** would require **lossy
   resampling** of the sparse override layer onto the rotated lattice. This is a **deferred,
   flagged opt-in** ("this resamples/bakes the sculpt layer" warning) — not the default and not a
   silent operation (see Deferred). Parametric parts have no such cost (case 1 re-voxelizes from
   the field).

So: angle/curve = producer parameters (exact from the field); lossless instance rotation/reflection
of baked data = the order-48 signed-permutation enum; non-lattice rotation of a *sculpted* part =
deferred lossy-resample opt-in. The `LatticeOrientation` enum below governs **only** case 2.

Today `NodeTransform` is **translation-only** (`scene.rs`: `offset_blocks: [i64; 3]`, with a
`// future: rotation, scale` marker) and the resolver composes placement by **pure i64 addition**
(`scene.rs::walk_nodes` sums `offset_blocks` down the tree). Two extensions ride here — a
non-gating sub-block translation term (item (0), an early data change) and ROTATION, which (unlike (0))
is **not** a one-substep change but a **dedicated milestone** (Phase F-rot) with three golden-gated parts
((a)–(c)):

- **(0) Sub-block placement is VOXEL-GRANULAR at the DOCUMENT'S density (NOT a milestone gate;
  the kit-authoring primitive — driven by sub-block boolean kit-of-parts authoring, ADR 0004 §A/§G9).**
  **THE PLANNING UNIT IS THE VOXEL.** "Blocks" are a **DERIVED overlay** — a grid line every `d` voxels,
  shown to display the target game's block grid and to drive block-aligned mating — **not** a hardcoded
  fundamental. So `NodeTransform`'s placement becomes a single voxel-granular field, replacing the
  block-only `offset_blocks: [i64; 3]`:

  ```rust
  pub offset_voxels: [i64; 3],   // placement in VOXELS at the document's density d (§3f(0))
  // "Blocks" are a derived overlay, exposed via accessors — NOT a stored field:
  //   .blocks(d)        = offset_voxels / d          (the whole-block view for UI/mating)
  //   .block_aligned(d) = offset_voxels % d == 0     (the connector/joint mating predicate, §3i)
  ```

  where `d = voxels_per_block` is the **document's density** (below). Placement is stored **directly in
  voxels** — a single field, voxel-granular, period. At resolve it enters the i64 placement sum as-is
  (`translation_voxels = offset_voxels + lattice_shift − recentre`), with **no rounding and no
  integer-multiple caveat** (the resolved grid *is* `d`). It composes by pure i64 addition, serializes
  byte-stably as plain integers, and leaves the store/chunking unaffected. It follows the
  `Point.offset_voxels` precedent already in the code (`scene.rs:381`), which is itself single-field
  voxels.

  - **One field, not two — the decided representation (amended 2026-06-28).** An earlier draft split
    this into `offset_blocks: [i64;3]` + a within-block `offset_subvoxels: [i32;3]` remainder. That was
    **rejected**: two fields summing to one position admit **redundant representations**
    (`blocks=1,sub=0` ≡ `blocks=0,sub=d`) with **no canonical form**, which breaks `NodeTransform`'s
    derived `PartialEq` (geometrically equal transforms compare unequal → broken no-op / drag-coalescing
    guards) and makes the block-aligned mating test `offset_subvoxels == 0` **unreliable** (it
    mis-judges the equivalent `blocks=0,sub=d`). The single `offset_voxels` field has a canonical form
    by construction, a geometric `PartialEq`, and a **canonical-form-independent** mating predicate
    `offset_voxels % d == 0`. The integer-exactness of between-part joints (§3i) comes from doing the
    placement sum in **i64** plus the connector tier's lattice constraint — **not** from a two-field
    representation — so nothing is lost. "Blocks" stay a *derived overlay* exactly as this section
    argues, via `.blocks(d)` accessors.
  - **No magic number — the denominator IS the document's density.** A **Minecraft** document
    (`d = 1`) has placement that is naturally block-granular (every voxel is a block — correct for MC);
    a **Vintage Story** document (`d = 16`) has 16 sub-positions per block per axis; a **Hytale** (or
    any other game) document has whatever `d` that game uses. **Same mechanism, every game, zero
    hardcoded constant.** (Sub-block placement is *within-part* detail authoring; inter-part mating
    stays block-aligned via `offset_voxels % d == 0` — see §3i "within-part vs between-part".)
  - **Units are an INPUT/DISPLAY layer over the canonical voxel storage (Fusion-style; added 2026-06-28).**
    Because placement (and sizes/radii) are stored as **canonical voxels**, a user-facing measurement is a
    *unit expression* parsed onto that canonical store and formatted back — exactly Fusion's model (it stores
    one canonical unit internally and lets you type measurements in any unit). The two interconvertible units
    are **blocks** and **voxels**, related by the document's `d` (`blocks · d = voxels`); a measurement like
    "3 blocks", "56 voxels", or a mixed "3b 8v" parses to `3·d + 8` voxels and round-trips. This is **why
    canonical single-field `offset_voxels` matters** — the rejected two-field blocks+subvoxels split would
    have baked one unit decomposition into storage, fighting the units layer instead of feeding it. The
    parser/formatter + a per-document default display unit are a **Slice-2 input/display concern (no
    foundation change)**. **[SHIPPED 2026-06-29]** This units layer is now WIRED into the app: the
    exact-rational blocks/voxels Measurement parser core landed (`b844d17`) and is wired into the inspector
    **Offset** fields (`fb2ad2a`) and the voxel-granular shape **Size** with parametric Measurement
    retention (`f3841b0`); `SetDensity` re-evaluates retained measurements per the rule below. (The earlier
    `size_blocks` field was renamed `size_voxels`, no migration — pre-alpha, per constraint 4.) DECIDED
    (2026-06-28): **(i) units are blocks + voxels only** (the two grid-native
    units, interconvertible via `d`); real-world units (m/cm) are out of scope. **(ii) a measurement RETAINS
    its authored expression (parametric)** — the canonical `offset_voxels` (and size/radius) is the DERIVED
    value, with the authored block/voxel expression stored alongside, Fusion-parameter-style. A measurement
    is a sum of block-terms and voxel-terms (`"3 blocks + 8 voxels" → 3·d + 8` voxels). This retention
    **refines the "density change is always destructive" posture**: on a density re-target the expression
    re-evaluates — **block-terms scale in block-space (lossless) and, for an INTEGER-multiple re-target,
    voxel-terms scale by the ratio and also land exactly (a lossless refine of the whole document)**; only a
    NON-integer-ratio re-target can push voxel-terms off the lattice (the warned/lossy case). Consequences are
    **additive and deferred to Slice 2+ / the ADR 0004 parametric-relations tier — the §3f(0) canonical
    single-field storage is unchanged**: the placement `Intent`s (`SetOffset`/`SetAnchor`/`SetCorner`/
    `SetShapePoints`, §3i) carry the **expression**, not just the derived voxels, so replay/undo preserve
    authored intent; and this parametric measurement model is the same machinery the datum/relations tier
    (ADR 0004) uses for hosted/derived points.
  - **Parser policy — STRICT (2026-06-28; revisit on demand).** Measurements evaluate as **exact rationals**
    (no floats), and the result MUST land on a whole voxel (the atomic unit). Fractions/decimals are allowed
    on **block-terms only** (`3.5 blocks`, `8/16 blocks`, `3 8/16 blocks`); a **voxel-term must be an
    integer** — sub-voxel input (`8/16 voxels`, `8.5 voxels`) is **rejected** (nothing is finer than a voxel;
    nudge to a block-fraction or a denser document). A block-fraction that does NOT land on a whole voxel at
    the current `d` (e.g. `3.5 blocks` at an odd `d=15` = 52.5 voxels) is **rejected at entry with the nearest
    representable values shown** — never silently rounded-and-stored as if exact. (At even densities like VS's
    `d=16`, halves/quarters/eighths/sixteenths all land, so this case is rare.) The looser
    *approximate-and-retain* alternative — store the expression, snap the derived voxel value, flag
    "approximate at this density" — is deferred unless users ask for it.
  - **Density `voxels_per_block` is a DOCUMENT-LEVEL attribute (the crux).** `d` is a
    **document/project property** — "which game/grid this plan targets" — **saved with the document,
    uniform across it**, and is **NOT** a per-node field, **NOT** a casual render knob, and **NOT**
    redundantly stored per shape. Different games are different documents with different `d`. (This
    also resolves the latent inconsistency where density was carried per-`Tool`-shape: density belongs
    on the **document**, not on each shape.)
  - **The old "density trap" is INVERTED, not avoided.** The earlier worry — *don't store placement in
    voxels because voxels are density-dependent* — assumed density was a mutable render setting. It is
    not: density is **fixed per document and travels with the file**, so a voxel-granular placement at
    the document's own `d` is unambiguous and portable — the grid moves with the document and there is
    **no trap**. The only density change is the explicit destructive re-target below.
  - **Changing the density is a DESTRUCTIVE RE-TARGET that breaks the document and MUST warn.** `d` is
    fixed for a document's life. Changing it **reinterprets every voxel-granular placement and every
    voxel-granular sculpt/override layer against a different grid**, which they do **not** survive
    cleanly — so it is an **explicit, warned, destructive operation** ("this re-targets the plan to a
    different game/grid and breaks voxel-granular detail"), never silent and never a casual toggle. (A
    convenience integer-multiple rescale — re-targeting VS `d=16` → a `d=32` document by doubling every
    voxel key — is a **possible future affordance**, but the spec posture is **warn: this breaks the
    document**.)

- **(a) order-48 signed-permutation orientation as a TYPE-level enum (F1 — widened from 24 to 48)**
  — `NodeTransform` gains `rotation: LatticeOrientation`, **plus a handedness/reflection bit** so the
  stored orientation covers the **full order-48 signed-permutation group** (the 24 proper rotations +
  the 24 reflections), **not** a general affine in the stored type. Keeping it an enum (+ one bit)
  means voxel resampling — rotation **or mirror** — stays an **exact index permutation** (no
  interpolation), positions stay lattice-exact under i64 rebase, and the **serialization stays
  byte-stable** (an enum discriminant + a bit, not a float matrix). The reflection half is the
  cheapest possible unlock of **bilateral mirror symmetry** (a left wing = a mirrored instance of a
  right-wing def), the most common architectural symmetry and previously **inexpressible** in the
  stored transform. **Caveat:** an *asymmetric* sculpt ornament inside a def is **chiral** — a
  reflected instance mirrors that ornament too; reflection is exact-and-lossless, but it does mirror
  any handed detail (acceptable and expected; flagged here so it is not a surprise).
- **(b) Rotated-chunk conservative-cover fan-out** — a rotated part's chunks no longer map 1:1 to
  output chunks; invalidation/residency must fan out over the **conservative cover** of the rotated
  AABB.
- **(c) Rotation-aware AABB-skip + spatial-index fingerprint** — the world-AABB used for skip tests
  and the `LeafFingerprint` (spatial index) must incorporate the orientation so a pure rotation
  re-keys correctly.
- Because overrides live in the **part-local** frame, moving/rotating an instance moves its overrides
  for free — the transform composes down the tree in i64, and rebasing happens at consumption (§4).
  This is **S4**.
- **Non-lattice rotation of a *sculpted* part = a deferred, flagged lossy-resample opt-in** (case 3
  above): it bakes/resamples the sparse override layer onto the rotated lattice and is surfaced with a
  "this resamples the sculpt layer" warning, never silent, never the default. A *parametric* part needs
  no such opt-in — it is simply a producer-angle parameter re-voxelized from the field (case 1). General
  free/non-lattice affine of baked voxel data + interpolating resampling stays explicitly deferred
  (see Deferred).

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

#### 3h. Traits vs enums — the dispatch principle, applied

A single rule decides every "trait or enum?" call in this foundation:

> **Open, extensible set + uniform behavior + real polymorphism → trait. Closed, small,
> serialized/matched set → enum.** Over-using `dyn` has real cost: pointer indirection,
> non-inlineable calls, and (for anything serialized) `erased_serde`/type-registry ceremony — so a
> trait object must earn its keep with genuine open-ended extension.

**Traits** (open extension points):
- **`VoxelProducer`** (§3d) — anyone can add a producer kind; uniform `resolve_into`.
- **`Command`** (§2) — already a trait; an open, growing family of edits with a uniform apply/inverse.
- **`Tool`** — interactive tools (sculpt / select / move / measure) sharing
  `activate / handle_intent / preview / commit → Command`. **Many future features are tools**, so a
  `Tool` trait + a tool registry is the deliberate extension point (the same shape as the producer
  registry). A tool consumes `Intent`s and emits `Command`s; it never mutates the document directly.
- **`ChunkStore` backend trait** — the residency layer talks to storage through a trait;
  `DiskChunkStore` (RAM-LRU + disk spill) becomes **one impl**, leaving room for a future packed /
  network-backed store without touching the residency manager.

**Enums** (closed, serialized, exhaustively matched — kept as enums *on purpose*):
- **`Intent`** (§6a) — it is the **replay/control boundary**; a trait object would wreck scripting
  (no clean serialize, no exhaustive match, no stable on-disk form). Enum is mandatory here.
- **The `Layer` kind** (`Producer` vs `Sculpt`, §3b) — a fixed two-arm fold.
- **`CombineOp`** (`Union`/`Subtract`/`Intersect`) — a fixed, matched, serialized set.

#### 3i. SKETCH → VOLUME authoring (the atom) + point-defined substrate — retires the 3-frame leak; within-/between-part granularity

**EVOLVED 2026-06-28 — the authoring atom is 2D SKETCH → VOLUME; primitives + box-drag are sugar.**
A design conversation + sourced tool research (HardCuts — a standalone **voxel/SDF** app, not a Blender
plugin — plus SketchUp push/pull, Revit massing, Fusion "every model starts as a 2D sketch": the
buildings-focused consensus) reframed the shape model. The point-defined work below is RIGHT and is the
**substrate**, but its *target* generalizes from "2-point primitives" to **sketches**:

- **The atom = a `Sketch` (a grid-aligned plane + an ordered point *profile*, voxel-granular at `d`,
  §3f(0)) + an `Operation` (`Extrude` / `Revolve` / `Sweep`)**, producing a volume. This is a new
  `VoxelProducer` family (§3d; producers are the open trait, §3h), composed through the existing
  `CombineOp` boolean `Layer` stack (§3b) — "anything that bounds a volume is a boolean operand on the
  voxel field" (the HardCuts insight). Buildings are overwhelmingly footprint-extrude + profile-sweep +
  revolve, so this natively covers walls/floors/roofs/arches/columns/cornices; primitive boxes
  under-cover them and are the trivial by-hand case. **The product's value is organic / complex shapes**
  (incl. VS sub-block chiseled surfaces like fake brick), not blocky massing.
- **Primitives are SUGAR over sketches, not a parallel engine.** `box` = a rectangle profile extruded;
  `cylinder` = a circle profile extruded; `sphere`/`dome` = a circle revolved; `torus` = a circle
  revolved off-axis — each desugars to (profile + operation). The existing `SdfShape` SDF code is **kept
  and reused as the rasterizer for curved profiles** (a circle profile *is* the SDF circle), not
  discarded; the new producer is added **alongside** `SdfShape`, keeping goldens green during migration,
  and primitives are reframed as sugar incrementally.
- **Even box-drag is sugar.** Dragging a rectangle across the ground plane — or onto the *surface of an
  existing object* (SketchUp push/pull-style on-surface sketching) — creates a rectangle sketch that
  extrudes. So the zero-ceremony entry is *also* sketch-underneath: **one operand pipeline** fed by
  box-drag, named primitives, and drawn/swept/lathed profiles. This dissolves the floor-vs-ceiling
  tension (see the `authoring-atom-sketch-to-volume` memory).
- **`ShapeAnchors` generalizes to a sketch.** The 2-point `Box{anchor, corner}` below is the degenerate
  rectangle-extrude; a polyline footprint is the same machinery with more `ShapePoint`s. `ShapePoint` is
  exactly a sketch vertex; the **leak-retirement and within-/between-part split below still hold verbatim**
  (sketches are point/corner-defined → leak-free by construction). The "honest scope" caveats below are
  reframed: a free-axis cylinder is a circle profile on a *rotated sketch plane* (still gated on the
  plane-orientation milestone, §3f(a)); an "arc" is just a curved profile segment, not a new `ShapeKind`.

**Build arc (Slice 2):** **2a** = sketch→`Extrude` engine, headless `shot`-verified · **2b** =
box-drag-on-plane sugar (first interactive authoring) · **2c** = free polyline sketches +
`Revolve`/`Sweep` + primitives-reframed-as-sugar + on-surface sketching. The parametric blocks/voxels
units layer (§3f(0)) feeds sketch dimensions. Minecraft (`d=1`) collapses sub-voxel profile precision to
whole blocks (sub-block features off), no special-casing. **Slice 1 (the §3f(0) coordinate substrate —
document density + single `offset_voxels`) is SHIPPED.**

**[SHIPPED 2026-06-29]** The sketch→volume pipeline is now USER-REACHABLE, and `Revolve` shipped early:
the sketch→extrude engine (`f62f031`), then sketch nodes creatable + editable via intents in the headless
core (`NodeSpec::Sketch` + `Intent::SetSketch`, `18f0abd`) and the interactive authoring UI (Add-menu chip
+ inspector editor, `93552e2`). The `Operation` enum (`Extrude | Revolve`) and the unified
`SketchSolid { sketch, operation }` producer landed (`d568b05`), the `Revolve` solid-of-revolution
operation (full + partial turn, `1c9eccb`), and its UI (Operation picker + axis/angle fields + demo
golden, `85bba95`). **STILL DEFERRED per the arc:** box-drag-on-plane (2b), free-polyline editing,
`Sweep`, primitives-reframed-as-sugar, and on-surface sketching (2c).

*The original point-defined rationale (still valid as the substrate) follows.*

Driven by the kit-of-parts authoring need (composing primitives with boolean ops at sub-block
resolution, ADR 0004 §A/§G9), shapes stop being **`size_blocks` + an implicit center anchor** and
become **parameterized by a small set of TYPED REFERENCE POINTS**. This is a **parameterization layer
ABOVE the `VoxelProducer` trait (§3d), not a payload change** — the points DERIVE the `(size, origin)`
the producer already consumes, so the producer's resolved output is unchanged:

```rust
pub enum ShapeAnchors {                          // derives (size, origin) for the existing producer
    Box      { anchor: ShapePoint, corner: ShapePoint },              // 2-point: anchor + opposite corner
    Cylinder { end_a: ShapePoint, end_b: ShapePoint, radius: SubVoxelLen }, // 2-point axis + radius (also Tube)
    Arc      { a: ShapePoint, b: ShapePoint, c: ShapePoint },         // 3-point arc / plane / wedge
}
// A reference point carries voxel-granular coordinates at the document's density d (§3f(0)),
// stored as a single voxel vector (same representation as NodeTransform.offset_voxels):
pub enum ShapePoint {
    Inline { offset_voxels: [i64; 3] },   // a typed inline point, voxel-granular at d
    HostedOnDatum(NodeId),                // names a Datum/anchor (F4 seam, §1)
}
```

- **Corner/face anchoring makes `leaf_lattice_shift` identically ZERO by construction.** The
  implicit-center model forced a half-block shift for odd `size_blocks` (the comment-arithmetic that
  reconciled the 3 coordinate frames — the flagged Leak). A point-anchored shape places its corner/face
  exactly on the lattice (or on a sub-voxel position via `offset_voxels`, §3f(0)), so **there is no
  half-block correction to carry** — this is the **intended RESOLUTION of the 3-frame "Leak"**, and that
  leak can be retired once shapes are corner-anchored (recorded in the Leaks note above). *Scope note:*
  `leaf_lattice_shift` is computed from `size_blocks` **regardless of kind**, so corner-anchoring zeroes
  the **block-lattice** shift for every shape (Box and radius shapes alike). The stronger "surface lands
  exactly on the lattice" is a **Box** property; a radius shape with an **odd voxel diameter** still
  centres on a half-voxel — but that is a *producer* sampling property (the SDF inscribes the shape in
  its AABB), not a placement shift, and is independent of this anchoring change.
- **Points are inline OR `HostedOnDatum` references (the F4 Datums seam, §1)** so the relational/solver
  tier can **NAME a shape's anchor** — a shape corner can be hosted on a datum/level/grid exactly like a
  part, riding the same id-keyed durable graph.
- **The Phase-C `Intent` enum gains additive serializable variants** — `SetShapePoints` /
  `SetAnchor` / `SetCorner` — and `SetOffset` becomes **derived/anchor-based** (moving the anchor moves
  the shape), all additive growth on the §6a serializable boundary.

- **Honest scope of the three variants (amended 2026-06-28).** Only **Box** is purely "a
  parameterization above the existing producer"; the other two are NOT free parameterizations of today's
  producers and must not be planned as such:
  - **`Box{anchor, corner}`** corner-anchoring derives `(size, origin)` the axis-aligned `SdfShape`
    already consumes and zeroes the block-lattice shift (above). Genuinely "payload unchanged."
  - **`Cylinder{end_a, end_b, radius}` with two *free* endpoints implies an arbitrary axis**, but the
    only cylinder producer today (`signed_distance_elliptical_cylinder`) is **hard-coded axis-along-Y**.
    So for v1 the two endpoints are **constrained to a single axis** (they differ on exactly one
    coordinate); a genuinely diagonal cylinder needs the **deferred Phase F-rot rotation milestone**
    (§3f (a)), not this section. "Producer payload unchanged" is honest only for the axis-aligned case.
  - **`Arc{a, b, c}` has NO producer today** — the closed `ShapeKind` set is
    `Cylinder/Tube/Sphere/Torus/Box`. It is **new producer work** (a new `ShapeKind` + a new
    `signed_distance` arc/wedge function), tracked as such, **not** folded into the anchoring slice.
  - **Derived size must be non-degenerate.** `Box{anchor, corner}` derives `size = |corner − anchor|`
    per axis under a **min/max normalization** (inverted corners are canonicalized, never underflowed
    into the `u32` `size_blocks`), and `SetAnchor`/`SetCorner` **reject a zero-extent box** rather than
    silently resolving an empty grid. This invariant lives on the anchor-editing intents.

**Within-part vs between-part granularity (load-bearing — state it clearly).** Sub-block placement
(an `offset_voxels` not divisible by `d`, §3f(0)) applies to **child-shape placement WITHIN a part
DEFINITION** (carving/detail authoring — the sub-voxel anchoring of the points above). **CONNECTORS** (a
part's mating interface, ADR 0004 §A) and **INTER-PART JOINTS** (§1) stay on the **integer BLOCK
lattice** (`offset_voxels % d == 0`), so the ADR 0004 joint solver remains **integer-exact**. The rule: **sub-block
where you carve, block-aligned where you mate.** This reconciles ADR 0004's "connectors stay block-clean"
(still true — between-part mating is unchanged) with sub-block kit authoring (a new within-part freedom):
the two granularities live in different tiers and never meet.

### 4. Chunked sparse streaming store: anisotropic tiling, rebase-at-consume, unified residency

**One residency layer, not two.** `ChunkResolveCache`'s second LRU (stacked on `DiskChunkStore`'s
LRU) is removed. There is a single `ResidencyManager` owning the resident set; it talks to storage
through the **`ChunkStore` backend trait** (§3h), with `DiskChunkStore` (RAM-LRU + disk spill) as the
first impl — its spill backend, not an independent cache. Eviction returns GPU buffers to a pool (ADR
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

**Assembly composition is ORDERED / PRIORITIZED, not a plain union.** Today the resolve **unions all
leaves** (`scene.rs::walk_nodes` composes placement by pure i64 addition with no precedence). The
refinement: the assembly resolve in `store` composes leaves **in a defined order with later-wins
precedence**, and the scene-root **assembly override layers (§1) composite LAST** over the unioned
parts in their region. This is what lets a **junction module or an assembly patch WIN over the walls
it sits across** — without it, an intersecting wall and a tower-at-the-crossing module would merely
union, and a bespoke world-frame patch could not override the parts beneath it. (Surfaced by ADR
0004's overlap-as-junction handling; it slots into the migration sequence near the compositor/layer
work, below.) Per-part layer folding (§3b) stays as-is; this clause governs **inter-part / assembly**
precedence, the level above it.

**Incremental re-mesh is finally CONSUMED.** **[SHIPPED 2026-06-29, #40 `9ff63c3`]** An edit's
world-AABB → set of dirty `(chunk_coord)` (from `invalidate_aabb`) → those chunks re-resolve and
re-mesh; **all other chunks keep their cached GPU buffers**. The renderer stops re-meshing wholesale
on edit (falls back to wholesale only on a floating-origin shift / density change). NB the *live*
plan is `cuboid_mesh::cuboid_incremental_plan`, NOT the instanced-era `renderer::incremental_rebuild_plan`:
the cuboid mesher culls each chunk's boundary faces against a 1-voxel apron from global occupancy, so
the dirty set is DILATED by the 26-neighbourhood (a neighbour's occupancy change alters this chunk's
seam faces) — the instanced planner had no such inter-chunk dependency. Region-scoped consumers
(diameter, layer
scrubber, `.vox` export) read the per-chunk store over the active region, never an assembled
whole-grid. Scrubbing the layer-band is a **per-fragment clip on absolute-Y** (ADR 0002 matrix row) —
no re-mesh — so a 10k-XZ scrub is interactive. This is **S7**.

**The per-chunk `ChunkRevision` is EXPOSED on the read seam (reconciled from the ADR 0005 gap sweep — FLAG 1).**
The per-chunk revision the store already tracks internally to drive incremental re-mesh (the dirty-mark
→ stale-discard machinery, §7) is **carried out on the region read API**: `resolve_region` (and the
per-chunk read) **returns / carries the `ChunkRevision` of each resolved chunk** alongside the
occupancy, so a derived-analysis consumer (the ADR 0005 space/nav/topo graphs and other
occupancy-reading passes) can **cache-key on exactly the revisions of the chunks it read** and reuse
the foundation's per-chunk invalidation rather than maintaining a parallel dirty-tracking scheme. This
is a **read-side addition only** — no per-voxel payload field, no schema cascade; it surfaces state the
store already owns. (The gap-sweep / ADR 0005 surfaced this need, the same posture as the ADR 0004
DATA-seam reconciliations and the §1 joint-arity fix crediting 0004's stress-test.)

### 5. Serialization & format tagging (V0 pre-alpha — tag now, migrate later)

The project is **V0 pre-alpha**: project files carry a **magic header + a version/epoch tag from day
one** (cheap), the loader **hard-errors on an unrecognized tag** (never silently misreads), but **no
migration code is written yet** and **pre-alpha files may break freely** — the same posture as config
(memory: no-config-back-compat) — until we declare a stable format. There is no "v1→v2" anything; the
tag exists so we can *introduce* migration the day we leave pre-alpha, not before.

```rust
#[derive(Serialize, Deserialize)]
struct ProjectFile {
    magic: [u8; 4],            // b"VXWP" — reject anything else immediately
    epoch: u32,               // format/epoch tag; the loader hard-errors on an unrecognized value
    voxels_per_block: u32,    // the document's DENSITY d (§3f(0)) — "which game/grid this plan targets"
                              // (Minecraft 1, VS 16, …); document-level, uniform, fixed for the doc's life.
                              // Changing it is the explicit destructive re-target (§3e/§3f(0)), not a load knob.
    scene: Scene,             // arena of defs/nodes (NodeId), per-def layer stacks, sculpt overrides,
                              // joints, datums (F4), instance KitParams/Def types (F5), the
                              // block-entity side-table (F2), AND the scene-root assembly-scoped
                              // override layers (§1)
}
```

- **Format:** a top-level header (magic + `epoch` tag) + the document body. The loader matches the
  `magic` and the `epoch` and **returns a clear error on any unrecognized tag** rather than
  attempting to reinterpret bytes. **No migration is written now**; cross-version migration is
  deferred until we leave pre-alpha.

- **Sculpt overrides use a DEDICATED sparse override codec (G2), not `chunk_storage::compress`.**
  The existing `chunk_storage::compress`/`decompress` consume an **f32 `VoxelGrid`** and
  **debug-assert a uniform per-axis `centre_fraction`** (`chunk_storage.rs`) — an invariant that
  producer-resolved grids satisfy but that **integer force-on / force-off deltas do NOT** (a delta is
  not a centred resolved grid). So overrides keep the *conceptual* "sparse, chunk-keyed storage"
  reuse but get their own codec:
  - **force-on:** a per-chunk set of **sorted integer voxel keys** (chunk-local `[u16; 3]` packed to
    a single sorted `u64` or delta-varint key list) + a **block palette** (a `block_id` + attrs
    palette index per force-on key — the categorical cell of §3a, not a 3-value material).
  - **force-off:** a **separate sorted key set** — force-off MUST be its own set, **NEVER a reserved
    palette slot / sentinel block**. A sentinel block would re-pollute the very categorical `block_id`
    field §3c is cleaning, re-introducing a "meaning packed into a data field" leak.
  - **byte order is deterministic** (keys ascending, palette in first-seen order) so the encoding is
    canonical, and the codec has its **own round-trip byte-identity test** — that test **is** the
    **S9** golden (encode → bytes → decode → byte-identical override layer + bytes stable across
    runs).
  - Large overrides stay compact (sorted keys + delta-varint + palette) and reload byte-identical.
  - **Override keys are voxel-granular at the DOCUMENT's density `d`** (§3e/§3f(0)): `d` is a
    document-level attribute (stored once on the document, §3f(0)), so an override layer needs **no
    per-layer density tag** — it is authored and resolved at the one document density. A density change
    is the **explicit destructive re-target** (§3f(0)), which reinterprets these voxel keys against a
    different grid and is warned, not a silent codec migration.
  - The **same codec serializes both the part-local sculpt overrides and the scene-root
    assembly-scoped override layers (§1)** — they differ only in anchoring frame (definition-local
    vs world), not in encoding — so assembly patches reload byte-identical too.
  This is **S9**.

- **Block-entity side-table + world-origin export contract (F2).** Alongside the occupancy/override
  codec the document serializes:
  - the **sparse, address-keyed block-entity side-table** (§3a-bis) — a `(voxel_addr → BlockEntity)`
    map, **part-local-anchored** (same frame as the sculpt overrides, so it rotates/reflects with the
    part under F1) and written **next to** the override codec for that scope. Sparse and deterministic
    (keys ascending) so it reloads byte-identical; it is the carrier for VS block entities / contents
    **and the F6 placeholder/proxy annotations** — including the `BlockEntity::Substitution` variant
    (proxy + real `target`/attrs/orientation/params) that the export contract below substitutes.
  - the **world-origin export contract**: a stored **build-anchor block coord** that maps to a
    target-game world coord on export, with the loader/exporter **asserting sub-block detail is phased
    to the document's `d`-grid** (`voxels_per_block`³ voxels/block — e.g. 16³ for Vintage Story,
    §3f(0)). This is the agreed world registration the store's
    decoupled quantized floating origin (§4) otherwise lacks. (The VS-native schematic *exporter* —
    micro-block + block-state + entity round-trip — is a consumer feature in ADR 0005; the **contract
    and the side-table format** are pinned here.)

- **No migration code is written yet (V0 pre-alpha).** The loader's whole contract right now is:
  match `magic`, match `epoch`, and **hard-error on an unrecognized tag** with a clear message rather
  than silently misread. Pre-alpha project files may break freely between epochs — we are not
  promising cross-version compatibility until we leave pre-alpha. This is **S6**.
- The seam that makes future migration cheap is exactly the tag: when we *do* declare a stable
  format, an unrecognized older `epoch` becomes a dispatch into a forward-migration step instead of an
  error. We pay nothing for that now beyond writing the four magic bytes and the `epoch` u32.

### 6. Headless `AppCore` + testing strategy (the keystone)

**The single biggest correctness win: one `AppCore`, two shells.** `shot.rs`'s ~1700-line parallel
render path is **deleted**; `bin/shot` and `bin/main` both construct the same `AppCore` and the same
`gpu` render-item consumer, differing only in window vs offscreen surface.

```rust
pub struct AppCore {
    scene: Scene, edits: CommandStack, store: Store, camera: OrbitCamera,
}
impl AppCore {
    // --- the control surface: intents + queries + diagnostics + render ---
    pub fn apply_intent(&mut self, intent: Intent) -> Result<(), EditError>; // builds+applies a Command
    pub fn undo(&mut self); pub fn redo(&mut self);
    pub fn get_state(&self) -> StateSnapshot;                  // serializable doc/selection snapshot
    pub fn query(&self, q: SpatialQuery) -> Answer;           // structured, machine-readable (G3)
    pub fn diagnostics(&self) -> Vec<Issue>;                  // machine- AND human-readable (G3)
    pub fn render_items(&mut self, frustum: &Frustum) -> Vec<ChunkRenderItem>; // opaque to gpu
    pub fn render_png(&mut self, view: &ViewParams) -> Png;    // multi-view gestalt channel
    pub fn occupancy_region(&mut self, region: Aabb64) -> Occupancy;           // fog/scrubber/export
}
```

- **`query(SpatialQuery) -> Answer`** returns structured, machine-readable geometry/relationship
  facts over the **resolved grid + scene** — contact / gap / overhang / connectivity / bounds — built
  on the same region-scoped read path the existing diameter / scrubber / export consumers use (§4).
- **`diagnostics() -> Vec<Issue>`** returns constraint + structural checks as a list that is BOTH
  machine-readable (an agent's correct-step input) and human-readable (the same list surfaces in the
  inspector for a person). Orphaned-override flags (§3g) and unsatisfied joints (§1) report here.

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
- **Test layers:** (a) pure unit tests on `core_geom`/`scene`/`edit`/`store` (command inverse
  round-trips, layer-fold correctness, the override-codec byte-identity round-trip = S9, the
  tag/hard-error loader check); (b) the golden-PNG harness over `AppCore` for the render feature
  matrix (ADR 0002), **now including the far-scene fixtures**; (c) **stress-case integration tests
  S1–S10 as Intent scripts** against `AppCore` headless (the §6a scripted mode — no GPU needed for
  S1/S2/S5/S6/S8/S9/S10; S4/S7 add a golden).

### 6a. The `Intent` boundary — the automation / test / agent-control surface

> **[CROSS-REF 2026-06-29]** [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) ratifies this as
> THE one mutation door for ALL sources (human gizmo, future GPU sculpt brush, agent/LLM/solver): a
> GPU edit must **lower to an integer-addressed `Intent`** recorded CPU-side — there is no
> raycast/voxel-coordinate variant and must not be one. It also keys the human↔agent presence-lock off
> the `IntentEffect` split and enforces it AT this door.

`Intent` is a **serializable enum** and it is **THE single boundary through which all mutation
flows**: `ui/gizmos` and `ui/panels` both emit `Intent`s, `AppCore::apply_intent` is the only door,
and `Command` (the inverse-bearing journal entry) is built *from* an `Intent` (§2). New edit kinds are
**additive serializable variants** on this enum — the point-anchored shapes of §3i add
`SetShapePoints` / `SetAnchor` / `SetCorner`, and the legacy `SetOffset` becomes **derived/anchor-based**
(it moves the anchor rather than carrying a raw center offset) — additive growth, no boundary rewrite.
Because there is exactly one serializable door, several capabilities **fall out with no new
mechanisms**:

1. **`shot` = replay an intent script → PNG.** Today's screenshot is just the **empty-script** case;
   the general tool loads an `Intent` script, replays it through the real `AppCore`, and renders. No
   parallel path (the §6 keystone), no new harness.
2. **Scripted interaction tests.** The **S1–S10 cases become intent scripts**, so the golden harness
   gains a **scripted mode**: interaction bugs (drag, select, open-definition, sculpt) are caught
   **headlessly**, not just static-scene renders. This is the same net §6 describes, expressed as the
   §6a scripts.
3. **A live control socket (loopback).** The windowed app exposes a loopback socket that **pumps
   received `Intent`s into its own event loop** — so an external driver controls the running app
   through the identical door a human's gizmo drag uses.
4. **An MCP server thin-wrapping that socket** — `apply_intent` / `undo` / `get_state` /
   `screenshot` (and the §6 `query` / `diagnostics`). It is a *thin* wrapper: all logic lives in
   `AppCore`; MCP only marshals.

**Phasing:** (i) serializable `Intent` + headless replay first — this **doubles as the interaction
test net**; (ii) then the live loopback socket; (iii) then the MCP wrapper.

This makes the **`AppCore` keystone pull triple duty**: the real app, the test net, and the agent
driver, all behind one door. The hard requirement that buys all three is that the surface be
**deterministic and replayable** — an `Intent` script must reproduce the same scene (and, with a
fixed view, the same pixels) every run. (This is why `Intent` is an enum, not a trait object — §3h.)

### 7. Async / responsiveness model (S10)

> **[CROSS-REF 2026-06-29]** This §7 invariant — `apply` mutates the tree + marks chunks dirty but
> does NOT resolve/mesh inline; the GPU sits DOWNSTREAM of resolve, never upstream of the journal — is
> the load-bearing data-flow [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) pins. ADR 0006
> explicitly REJECTS the inverse ("stream voxel diffs back from the GPU and treat them as the delta")
> because it would invert this flow and break determinism/headless/journal-as-truth.

The whole point is that **edits, undo/redo, and camera always feel instant**, even mid-rebuild on a
10k-XZ scene. The model splits work across thread classes by cost and bounds every main-thread touch.

**Main thread — only cheap, bounded work:**
- input → `Intent` → `Command::apply`, where `apply` **ONLY** mutates the node tree, writes the
  sparse sculpt delta, and **MARKS chunks dirty** — it does **NOT** resolve or mesh inline. (That is
  what makes undo/redo and edits feel instant: an edit is a tree mutation + a dirty mark, never a
  re-resolve.)
- run egui UI; issue GPU draws;
- drain a **budgeted** slice of completed worker results;
- submit a **bounded** amount of GPU upload.

**Worker pool — heavy, embarrassingly parallel per chunk (rayon-style):**
- **resolve** (producers via `resolve_into` + override fold, §3b/§3d) and **mesh** (cuboid
  decomposition, ADR 0002) as **per-chunk jobs**, each consuming an **immutable `Arc` snapshot** of
  the relevant def/chunk inputs (no background thread ever mutates the document — `AppCore` is the
  sole writer of `scene`/`edits`).
- **prioritized:** in-frustum / nearest-camera / just-edited chunks first.
- **cancellable via a revision stamp:** a job is computed against a revision; a **stale-epoch result
  is discarded** and the chunk re-queued. A chunk is **clean only when an epoch-matched result is
  INTEGRATED**, never at dispatch.

**I/O threads — never on the main thread:**
- chunk spill/load, project save, and `.vox` export run off-thread.
- for **>10k-XZ scenes, prefetch chunks ahead of horizontal camera motion** and **evict behind** (the
  anisotropic XZ residency policy, §4).

**Integration discipline (this is what kills hitches):**
- drain completions under a **per-frame time budget (~2 ms)** plus a **bounded GPU-upload budget**, so
  a big rebuild **amortizes across frames** instead of stalling one.
- **never `device.poll(Wait)` on the main thread;** use **async readback** for `shot` / export.

**Optimistic sculpt feedback (the one feel special-case):** on a sculpt stroke, render the touched
stroke region **immediately** — a tiny bounded local remesh or a lightweight overlay — while the full
per-chunk re-resolve completes off-thread and then **replaces** the optimistic patch.

**Tie back to S10 (the per-scope revision invariant — unchanged, this is its model):** `AppCore` as
the **sole document writer** + **`Arc` snapshots** to workers + the **revision-stamped completion
queue** IS exactly this model. Edit-during-mesh consistency uses a **per-scope monotonic revision
stamped into the cache ENTRY, not a global doc version** — a global version would mark *every*
resident chunk stale on any edit, starving far-chunk re-meshing (a far edit forcing near chunks to
re-mesh and vice-versa). Each cache **entry** carries a per-scope (per-chunk / per-def-scope)
revision; a completion is stamped with the revision it was computed against; a stale completion is
**discarded** and the chunk re-queued; a chunk is **clean only when an epoch-matched result is
INTEGRATED**. The command stack is therefore never blocked by, and never inconsistent with,
background meshing/scanning. This is **S10**.

**This same per-chunk revision is the `ChunkRevision` exposed on the read seam (§4, FLAG 1).** The
per-chunk revision stamp the worker pipeline already maintains is precisely the value the region read
API now carries out (§4), so ADR 0005's derived-analysis caches key on the same monotonic stamp the
re-mesh pipeline uses — one revision model, two consumers (internal re-mesh + external derived
graphs), no parallel dirty-tracking.

## Acceptance criteria — stress-case walkthrough (S1–S10)

| # | Stress case | How the design satisfies it |
|---|---|---|
| **S1** | Undo a single sculpt stroke on a FAR chunk (XZ ~10k) | `SculptStroke` stored a **sparse inverse delta** (only overwritten `(addr→prior cell)` pairs, the cell being the categorical `block_id`+attrs of §3a); undo applies it, dirtying only the stroke's chunks → `incremental_rebuild_plan` re-meshes just those via `resolve_into(chunk_box)` (§3d). Chunk-local integer payload (§3a) + absolute-i64 store means the far edit is exact and never touched other chunks. **O(stroke), exact restore.** (§2, §3a, §3d, §4) |
| **S2** | Edit a def with ~50 instances | One `Command` mutates the `AssemblyDef`. Instances *reference* the def, so propagation is free; invalidation fans out to the 50 instance placements' world-AABBs (via `world_aabb_blocks`, §3d) → only **intersected chunks** re-mesh. **One undoable command.** (§1, §2, §3d, §4) |
| **S3** | Sculpt an INSTANCE | `EditTarget::Assembly` disables geometry tools; the command factory returns `EditError::InstanceNotEditable`; UI offers **"Open definition"** → `EditTarget::OpenDefinition(def)`. **Disallowed, redirected.** (§1) |
| **S4** | Move + 90° rotate a SCULPTED part | Overrides are **part-local**; the `NodeTransform` gains a **24-orientation lattice rotation enum** (§3f, its own milestone) that composes in i64 and rebases at consumption; resampling is an exact index permutation, serialization stays byte-stable. **Overrides move with the part; positions exact.** (§3f, §4) |
| **S5** | Change base producer with anchored overrides | Overrides are **address-anchored, not occupancy-anchored**: kept verbatim, redundant/no-op ones retained, out-of-AABB ones **flagged `orphaned`** and rendered; explicit undoable **prune** offered. **Never silently dropped.** (§3g) |
| **S6** | Load a file written by a different format epoch | The format is **tagged (magic + `epoch`)** and the loader **refuses to silently misread**: a recognized tag loads, an unrecognized tag is a **hard error** with a clear message — never silent corruption. **Cross-version migration is deferred until we leave pre-alpha** (V0: pre-alpha files may break freely). **Tagged, never misread.** (§5) |
| **S7** | Scrub layer-band on 10k-XZ scene | Band clip is a **per-fragment discard on absolute-Y** — no re-mesh; consumers read the **region-scoped** store, not a whole-grid. **Interactive, no full re-mesh.** (§4) |
| **S8** | Sculpt stroke that GROWS the bbox | Chunks keyed by **absolute i64** coords with chunk-local integer payload (§3a); store is rebase-free; the floating origin is **sticky/quantized** (§4) so growing the extent does not move it; only touched chunks change. **Cache not nuked.** (§3a, §4) |
| **S9** | Save/share large sculpt overrides | Overrides use the **dedicated integer-delta codec** (§5): sorted force-on keys + a **block palette** (categorical `block_id`+attrs), a **separate** sorted force-off set (no sentinel block), deterministic byte order, **own byte-identity round-trip test**. **Compact, exact.** (§5) |
| **S10** | Scan assets + mesh while editing | `Command::apply` only mutates the tree + marks chunks dirty (no inline resolve/mesh), so edits feel instant; resolve + mesh run as **prioritized, cancellable per-chunk worker jobs** on **Arc snapshots**; completions drain under a **~2 ms + bounded-GPU-upload per-frame budget** (never `poll(Wait)` on main); results are stamped with a **per-scope cache-entry revision**, a chunk is clean only when an **epoch-matched result is integrated**, stale completions discarded. **Always responsive; command stack consistent; far chunks not starved.** (§7) |

## Migration sequence (incremental, behind the golden net)

Each phase is a **green checkpoint** — the app is built, golden-verified, and shippable. The
golden-PNG net (already DONE, ADR 0002 E0) is the guard throughout; **nothing changes pixels except
where a feature row explicitly allows it.** Reuse is maximized; the only outright *replacements* are
the `shot.rs` parallel path and the second LRU.

**Phase A — Layering + `AppCore` extraction (no behavior change). [ships first; gates everything]**
1. Move `CHUNK_BLOCKS` → `core_geom`; move domain types out of `panel.rs` into the new `scene` layer;
   break `chunk_cache→vox_export` by relocating both under the new layering. **Dissolve the legacy
   `Scene` god-object and reclaim the name:** the clean data layer is `scene::Scene`; split `ui` into
   `ui/gizmos` (3D) + `ui/panels` (2D). (Pure moves; goldens prove no change.)
2. Extract `AppCore` from `WindowedState` / the legacy `Scene`: pull the resolve/chunk/frame-math out
   of the god-objects into `app_core`/`store`. `WindowedState` becomes a thin shell.
3. **Re-point `bin/shot` at `AppCore`; delete its parallel render path.** *(This makes the golden net
   real — the single highest-leverage step; do it as early as possible so every later phase is
   guarded by the actual path.)* **Reuse:** cuboid mesher, fog, camera math, `DiskChunkStore`.
   **Replace:** `shot.rs` render copy.

**Phase B — Identity: `NodeId` arena (no user-visible change).**
4. Introduce `NodeId` + id-keyed arenas; selection/commands key on `NodeId`; demote `NodePath` to an
   ephemeral tree-render projection. **Add the F4 `Datum` primitive** (scene-owned named reference
   geometry, reusing the `NodeId` arena; `HostedOnDatum` as a `JointKind` over the joint graph) and
   the **F5 instance param-override bag + Def TYPE tier** (`Instance.param_overrides: Option<KitParams>`,
   `AssemblyDef.types`). These are pure scene-schema additions that ride the identity work; the
   producers/UX that consume `KitParams`/datums are ADR 0005. **Parallelizable** with C once A lands.
   **Reuse:** existing tree widget (recompute paths on render), the joint graph (HostedOnDatum).

**Phase C — Serializable `Intent` boundary + command stack + the headless replay/test net.**
5. Define the **serializable `Intent` enum** (§6a) and route the existing `PanelResponse` intents (now
   from both `ui/gizmos` and `ui/panels`) through `AppCore::apply_intent`/`undo`/`redo`; add the
   `Command` trait + `CommandStack`. Convert today's in-place mutations to commands incrementally
   (move/add/delete/rename first — each a small command with an obvious inverse).
5b. **Headless `Intent`-script replay + scripted interaction tests (§6a, phase i).** Make `shot` a
   general intent-script replay (the old screenshot = empty script), and recast **S1–S10 as intent
   scripts** so the golden harness gains a scripted mode. This is the interaction test net; it lands
   **early** so every later phase is guarded by scripted-interaction goldens, not just static renders.
   *(The live loopback socket and the MCP wrapper, §6a phases ii/iii, come after the foundation is in
   place — they are pure thin wrappers over this same door.)*

**Phase D — Store unification + chunk-local payload + rebase-free + incremental remesh consumed.**
- **D0 (GATE — G3): add XZ~10k far-scene golden fixtures FIRST.** Before any payload/store change,
  add far-scene goldens (e.g. `--demo-village` at `offset ≈ [10000, 0, 10000]`) so the keystone
  precision refactor ships **guarded**. The current goldens are near-only (`tests/golden.rs`).
6. **Chunk-local integer payload + categorical cell (G1, prerequisite; materials FOUNDATIONAL):**
   change `Voxel` from f32 `world_position` to chunk-local integer `local`, and replace the 3-value
   `material_id` (+ the `GRID_OVERLAY_BIT` hack) with a categorical **`block_id` + `attrs`** cell,
   **`attrs` being the F2 typed per-`block_id` `BlockAttrs` schema** (orientation in the order-48
   group, variant flags, connection bits) — not an opaque payload; the absolute i64 origin lives only
   in the chunk key; f32 is produced only at consumption. (Unblocks exact absolute storage, the §5
   override codec, and agent-composed buildings. The rich VS palette table / picker UI stays the
   deferred feature; the F2 connection-resolve pass and VS exporter are ADR 0005 consumers.)
7. Collapse the two LRUs into one `ResidencyManager` over the **`ChunkStore` backend trait**
   (`DiskChunkStore` as the first impl); key chunks by **absolute i64**; stop pre-rebasing on store,
   rebase at consumption only; make the floating origin **sticky/quantized** and decoupled from
   composite extent; **keep density as a cache-clearing key**. **(Fixes S8/cache-nuke.)**
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
10. Exercise `CombineOp::{Subtract,Intersect}` in the layer fold; **make the per-def layer stack an
    explicit ordered N-layer fold (later-wins, §3b)**; **make assembly composition ordered/prioritized
    (later-wins, not plain union, §4)** so a junction module / assembly patch can win over the walls in
    its region; **move the overlay flag to a per-draw uniform** (not a per-vertex attribute) and
    **delete `GRID_OVERLAY_BIT` from both shaders** (`cuboid.wgsl` + `cuboid_loaded.wgsl`) — the payload
    already dropped it in D6 — keeping `decompose_into_boxes` representation-agnostic. Golden-gated.
11. **Rotation milestone (G4 — its own milestone, golden-gated):** promote `NodeTransform` with the
    **order-48 signed-permutation `LatticeOrientation` enum (F1 — 24 rotations + 24 reflections, a
    type-level enum + handedness bit, not general affine)**; add (a) exact index-permutation
    resampling for **both rotation and reflection**, (b) rotated/reflected-chunk conservative-cover
    fan-out, (c) orientation-aware AABB-skip + spatial-index fingerprint. **Also compose F2
    `BlockAttrs.orientation` through the same order-48 permutation** so rotated/mirrored stateful
    blocks (stairs, logs, doors) re-face correctly rather than keeping stale facings. **(S4, F1, F2.)**

**Phase G — Sculpt override layer.**
12. Add `SculptOverride` (sparse, chunk-keyed) as an override layer, using the **dedicated integer-delta
    codec** (sorted force-on keys + palette, separate sorted force-off set — §5), **not**
    `chunk_storage::compress`; `SculptStroke` command with sparse inverse delta. **(S1/S2.)** Add the
    orphan policy + prune command. **(S5.)** (Depends on F0's `resolve_into` for O(stroke).)
12b. **Extend the override-layer phase to ASSEMBLY scope (§1):** add the scene-root
    `assembly_overrides: Vec<Layer>` (world-frame, this-site-only, does NOT propagate to defs), composite
    them LAST at assembly resolve (§4), serialize them with the document via the same override codec (§5).
    This is the third Thread-4 seam that ADR 0004's placement-specific junction patches consume; it is a
    deliberate, clearly-distinguished bend of the def-only-geometry rule (Fusion assembly-context-feature
    precedent). (Depends on step 10's ordered assembly composition.)

**Phase H — Part/assembly edit modes + `Tool` registry + control-surface queries/diagnostics.**
13. `EditTarget` state machine; instance-edit rejection + "Open definition". **(S3.)** Introduce the
    **`Tool` trait + registry** (§3h) and port the existing interactive tools (select/move/measure,
    then sculpt) onto it. Add the **control-surface `query(SpatialQuery)` + `diagnostics()`**
    (§6/G3) — contact/gap/overhang/connectivity/bounds over the region-scoped read path, plus the
    orphaned-override / unsatisfied-joint issue list. **Reserve the constraint/joint DATA seam**
    (the scene-owned, id-keyed, n-ary joint graph referencing `NodeId`s, §1) — **no solver**. *(These
    are what the agent feedback loop and the future architectural-kit / constraint-solver features
    consume; the solver + kit are a separate future design doc.)*

**Phase I — Serialization: tag + magic + hard-error (V0 pre-alpha; migrations deferred).**
14. Add the `ProjectFile` **magic + `epoch` tag**; loader **hard-errors on an unrecognized tag**
    (the S6 check); add the override codec's **byte-identity round-trip test (= S9)**. **Serialize the
    F2 sparse block-entity side-table** (address-keyed, part-local-anchored, deterministic byte order)
    alongside the occupancy/override codec, and **record the F2 world-origin export contract**
    (build-anchor block coord → target-game world coord, sub-block detail asserted phased to the
    document's `d`-grid — `voxels_per_block`³ voxels/block, e.g. 16³ for Vintage Story). **No migration
    code is written** — pre-alpha files may break freely until we leave
    pre-alpha. **(S6/S9, F2.)**

**Phase J — Async/responsiveness hardening (the full §7 model).**
15. Move resolve+mesh fully onto **prioritized, cancellable per-chunk worker jobs** over **Arc
    snapshots**; enforce the main-thread **integration budget (~2 ms drain + bounded GPU upload)** and
    **never `poll(Wait)`** (async readback for `shot`/export); add I/O-thread **prefetch-ahead /
    evict-behind** for >10k-XZ horizontal motion; add **optimistic sculpt feedback** (immediate
    bounded local remesh/overlay replaced by the off-thread re-resolve). Land the **per-scope
    cache-entry revision** invariant (not a global doc version): chunk clean only on **integrated,
    epoch-matched** result; stale-discard. **(S10.)** (Much of the worker scaffolding exists in
    `scan_worker`/ADR 0002 async meshing — this formalizes and completes the model.)

**Phase K — Agent control: live socket + MCP wrapper (§6a phases ii/iii).**
16. Add the **loopback control socket** the windowed app pumps into its event loop, then the **MCP
    server** thin-wrapping it (`apply_intent`/`undo`/`get_state`/`query`/`diagnostics`/`screenshot`).
    Pure thin wrappers over the Phase C door — all logic stays in `AppCore`. *(WFC / architectural-kit
    / constraint-SOLVER features build ON these seams; they are out of scope here, see Consequences.)*

**Parallelizable:** B∥(start of C); F0 + registry (step 9/F0) ∥ D/E; the serialization tag (I) can be
authored anytime after B. **Sequential bottlenecks:** A3 (real golden net) **and C5b (the Intent
replay/test net)** gate everything after them; **D0 (far-scene goldens) gates D6/D7 (the payload +
rebase-free store)**; D (rebase-free store) + **F0 (producer `resolve_into`)** both gate G (sculpt);
D also gates E (residency); the rotation milestone (F11) gates S4; the Phase K socket/MCP requires
the Phase C `Intent` door + the §6 query/diagnostics surface (H13).

## Consequences

**Better**
- The convergent backbone exists end-to-end: command → sparse delta → per-chunk invalidation →
  incremental re-mesh, reversible and horizontally scalable.
- Sculpting, ordered add/subtract layers, lattice rotations, and references all become **data
  changes**, not re-architectures — the foundation's whole point.
- **The golden net tests the real app** (one `AppCore`) **and now covers far scenes**, closing both
  the parallel-path gap and the never-tested far-precision case.
- The per-voxel payload is **chunk-local integer** carrying a **categorical block-palette cell**
  (`block_id`+attrs), so absolute-i64 storage is exact at XZ~10k *and* voxels carry real named blocks
  (not a 3-value enum sharing a field with a render flag) — the capability agent-composed buildings
  need, foundational rather than a retrofit.
- God-objects decompose into a strictly layered, AI-navigable codebase (D5); adding a producer /
  command / **tool** / **chunk-store backend** is a registration against a trait, not a 3-arm edit;
  closed sets (`Intent`, `Layer`, `CombineOp`) stay serializable enums (§3h).
- **One serializable `Intent` door makes `AppCore` pull triple duty** — real app, headless
  interaction-test net (S1–S10 as scripts), and agent driver (live socket + MCP) — with no new
  mechanisms; the agent feedback loop is **data-primary** (`query`/`diagnostics` over exact
  geometry + relationships) with multi-view render as the secondary gestalt channel, and the same
  diagnostics serve human inspection.
- **The async model keeps the app responsive under load:** edits/undo are a tree-mutate + dirty-mark
  (never inline resolve/mesh), heavy work is prioritized cancellable per-chunk jobs, and a per-frame
  integration budget amortizes big rebuilds — far-edge edits, big scenes, and bbox growth no longer
  nuke the cache (sticky/quantized origin); undo is O(stroke) (chunk-windowed `resolve_into`).
- Shared project files have a **magic + epoch tag** and a byte-identity-tested override codec; the tag
  is the cheap seam that makes future migration a dispatch rather than a rewrite (none written yet,
  V0 pre-alpha).
- **The `scene` layer is a node tree + a relationship/constraint graph** (nodes reference joints to
  other `NodeId`s), reserving the seam for human assembly-constraints and agent-driven building
  without committing to a solver now. It also carries **F4 datums** (scene-owned named reference
  geometry — levels/grids/axes, optionally terrain-relative — with `HostedOnDatum` riding the joint
  graph so "move Level 3 → everything hosted follows") and the **F5 instance param-override bag + Def
  TYPE tier** (a window family is one parametric def, not a fork per size), both pinned now while the
  schema is cheap to widen.
- **The instance transform is the full order-48 signed-permutation group (F1 — 24 rotations + 24
  reflections), not just the 24 proper rotations.** Reflection stays an exact, lossless, byte-stable
  index permutation, so **bilateral mirror symmetry** (a left wing as a mirrored instance of a
  right-wing def — the most common architectural symmetry) is expressible for one extra bit. Caveat:
  reflecting an instance mirrors any *asymmetric* sculpt ornament inside the def (chirality), exact
  but handed.
- **`BlockAttrs` is a typed per-`block_id` state schema (F2)** — orientation (in the order-48 group,
  so it rotates/reflects WITH the geometry), variant flags, neighbor-connection bits — with a
  connection-resolve pass, a sparse part-local block-entity side-table, and a world-origin export
  contract (build-anchor → target-game world coord, sub-block detail phased to the document's `d`-grid —
  `voxels_per_block`³ voxels/block, e.g. 16³ for Vintage Story). This is
  what stops rotated stairs/doors from keeping stale facings and stops VS export from being lossy by
  construction (a functional gatehouse no longer exports as dumb stone). The connection-resolve
  producers + VS exporter are ADR 0005 consumers.
- **Terrain is a first-class MUTABLE layer with a controlled producer↔terrain coupling (F3)** — a
  producer may sample live grade (`GroundHeightAt`) to tie in — superseding the earlier terrain
  read-only stance and unblocking cut/fill/terrace/berm/excavate work (those producers are ADR 0005).
  Only the terrain query is coupled; no general producer↔producer free-for-all.
- **The planner is explicitly STATIC (F6):** movable mechanisms are inert annotations (block_id +
  attrs + entity side-table) carried for VS export fidelity; interactables with no clean static-voxel
  form become **placeholder/proxy entities** — recognizable proxy geometry (the "feel") plus a
  `BlockEntity::Substitution` payload (the real target + attrs + orientation + params) that export
  substitutes, or a human places by hand. Live kinematics/state-machines/tick simulation are out of
  scope by ruling, keeping the three-tier static model uncluttered. (Placeholder producers, render
  treatment, substitute-on-export, and unsubstituted/unmapped diagnostics are ADR 0005.)
- **Composition is the foundation's parametric + corrective tiers of a three-tier authoring model**
  (parametric producers / relational joints / corrective override layers — ADR 0004): the layer stack
  is explicitly **N ordered layers, later-wins** (§3b), assembly composition is **ordered/prioritized,
  not plain union** (§4), and override layers exist at **two scopes** — part-local sculpt (propagates to
  instances) and **scene-root assembly-scoped patches** (world-frame, this-site-only, do NOT propagate;
  a deliberate, clearly-bounded bend of the def-only-geometry rule, Fusion assembly-context precedent).
  These let placement-specific junctions/patches win locally without disturbing other instances.

**Costs more**
- Up-front extraction (Phase A), the `NodeId` arena (Phase B), the **chunk-local payload migration
  (D6)**, and the **producer-trait redesign (F0)** are pure-foundation work paid before the payoff.
- Every mutation now goes through a command (more boilerplate per edit); each command must define a
  correct inverse (tested).
- Maintaining the dedicated override codec + the tagged format is ongoing tax on the document
  interface (the migration chain itself is deferred — V0 pre-alpha).
- Elevating materials to a foundational categorical block-palette cell means the payload, override
  codec, serialization, and meshing all carry `block_id`+attrs from day one (paid before the rich
  palette feature exists).
- Reserving the constraint/joint data seam + the `query`/`diagnostics` surface is foundation work
  whose payoff (assembly constraints, agent loops) lands only when the deferred features are built.
- The single-`AppCore` constraint means the windowed and headless shells must stay genuinely thin.
- Rotation is a whole milestone (conservative-cover fan-out + rotation-aware indexing), not a
  one-liner.

**Explicitly deferred**
- **LOD / impostors** — seam preserved via the `(chunk_coord, lod)` key (ADR 0002 O7); not built.
- **Per-chunk cross-block cuboid merge** and an optional 2D greedy pass over cuboid faces (ADR 0002,
  pursue only if GPU-bound).
- **Free (non-lattice) rotation/scale of a *sculpted* part with voxel resampling** — the rotation
  milestone (§3f) ships only the 24 axis-aligned lattice rotations as a type-level enum for the
  lossless instance-rotation case (exact index permutation, byte-stable serialization). Rotating a
  **sculpted** (non-parametric) part to a non-lattice angle is a **deferred, flagged lossy-resample
  opt-in** ("this resamples/bakes the sculpt layer" warning), never silent and never the default
  (§3f case 3); arbitrary affine + interpolating resampling of baked voxel data is a later layer.
  **Note:** arbitrary-angle *parametric* geometry (angled bastions, curves) is **NOT deferred** — it is
  a producer parameter, SDF-exact and re-voxelized at resolve (§3f case 1), available with no transform
  change.
- **The full VS block-palette table / picker UI** (the rich palette *content*). The per-voxel
  categorical CAPABILITY (`block_id`+attrs) is **foundational** (§3a) and **not** deferred — only the
  populated palette table + picker UI is.
- **In-part parametric construction tree** (booleans/lathe/array history) — the ordered layer stack
  is the seam it will slot into.
- **The constraint/joint SOLVER and the parametric architectural kit** — `scene` reserves only the
  joint DATA seam (§1); the solver + kit are features-on-top with their own future design doc.
- **WFC (wave-function-collapse)** — deferred, and scoped narrowly as a **regular-detail / material
  FILL producer** (deterministic, seeded), **not** a composition mechanism. It consumes the reserved
  seams (the `VoxelProducer` registry for fill, the categorical block-palette cell) as a future
  feature; it is not part of this foundation.

**Forward-looking note (the seams these features consume):** WFC, the architectural kit, and the
constraint SOLVER are all **deferred features that build on seams reserved here** — the `Intent`
control surface (§6a) + `query`/`diagnostics` (§6/G3) for agent-driven composition, the
categorical block-palette cell (§3a) and `VoxelProducer` registry (§3d) for WFC fill, and the
`scene` joint graph (§1) for assembly constraints. Reserving the seams now (cheap) avoids the
retrofit cascade later; building the features is separate design-doc work.

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
  O(stroke) for sculpt undo, and trivially serializable. Event logs would bloat shared files for zero
  benefit here.
- **Writing a versioned migration chain now (`migrate_v1_to_v2`, …).** Rejected: the project is V0
  pre-alpha (locked constraint 4) — pre-alpha files may break freely. We pay only for the cheap
  insurance (magic + `epoch` tag + hard-error on an unrecognized tag), and defer all migration code
  until we declare a stable format. The tag is the seam that makes adding migration later a dispatch,
  not a rewrite.
- **World-space sculpting as the DEFAULT / only override scope.** Rejected as the default: a
  *definition's* sculpt must live in its local frame to propagate to instances (S2) and to follow a
  moved/rotated part (S4). Part-local, address-anchored overrides are mandatory for both fidelity
  dimensions (D3) and the orphan policy (S5). **However**, a *second, explicitly-scoped* world-frame
  override layer is **deliberately allowed at ASSEMBLY scope** (scene-root `assembly_overrides`, §1):
  this-site-only patches that **do NOT propagate** to definitions, distinct from def-local sculpt and
  required for placement-specific junctions (ADR 0004 F2/F3). It is not a contradiction — it is a
  second, clearly-distinguished tier (Fusion assembly-context-feature precedent), not a replacement of
  the part-local rule.
- **Terrain READ-ONLY + no producer↔producer coupling.** **Superseded (F3).** The product owner
  chose **writable terrain**: terrain is a first-class mutable layer and a *controlled* producer↔terrain
  coupling (sampling `GroundHeightAt` to tie in to live grade) is permitted. The read-only/no-coupling
  posture forecloses all cut/fill/terrace/berm/excavate/terrain-relative-datum work, which is the bulk
  of site design. The coupling is kept narrow — **only** the terrain query, not a general
  producer→producer free-for-all. (The producers themselves are ADR 0005; the terrain *import format*
  stays an open research item.)
- **Restricting the instance transform to the 24 PROPER rotations only.** **Superseded (F1).** The
  24-rotation `LatticeOrientation` leaves the other 24 of the order-48 signed-permutation group (the
  reflections) inexpressible, so **bilateral mirror symmetry** — the most common architectural
  symmetry — has no stored transform and a left wing cannot be a mirrored instance of a right-wing
  def. Widening to the full order-48 group is **one handedness bit**, stays an exact lossless
  byte-stable index permutation (NOT the deferred lossy-resample path), and is cheapest now while
  0003 is Proposed. (Caveat: reflection mirrors any asymmetric sculpt ornament — chiral but exact.)
- **Leaving `BlockAttrs` an opaque payload (define it later).** **Rejected (F2).** An opaque
  rotation/variant blob with no rotation algebra means a rotated/mirrored instance keeps **stale
  facings**, neighbor connections are uncomputed, and VS schematic export is **lossy by
  construction** (a functional gatehouse exports as dumb stone). The typed schema + the rule that
  orientation composes through the F1 order-48 transform touches the payload, override codec,
  serialization, and meshing together — retrofitting it after those ship is a cascade, so it is
  pinned now (the schema + algebra + side-table + export contract; the exporter/connection-resolve
  producers are ADR 0005).
- **Live kinematics / state-machine simulation of mechanisms in the planner.** **Ruled out of scope
  (F6).** Doors/drawbridges/lifts/windmill-yaw are modeled as **inert annotations** (block_id + attrs
  + entity side-table) for VS export fidelity; VS supplies the behavior. A tick/signal simulation is a
  categorically different subsystem and is not bolted into the static three-tier model.
- **Reusing `chunk_storage::compress` for sculpt overrides.** Rejected: `compress` consumes an f32
  `VoxelGrid` and debug-asserts a uniform per-axis `centre_fraction` (`chunk_storage.rs`), which
  integer force-on / force-off deltas do not satisfy. A dedicated integer-delta codec (sorted keys +
  block palette + separate force-off set) is required; only the *concept* of sparse chunk-keyed
  storage is reused.
- **Force-off as a reserved palette slot / sentinel block.** Rejected: it re-pollutes the categorical
  `block_id` field with a non-block meaning — the same class of leak as `GRID_OVERLAY_BIT` that §3c
  removes. Force-off is a **separate sorted key set**.
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
- **`Intent` as a trait object (`Box<dyn Intent>`).** Rejected: `Intent` is the replay/control
  boundary (§6a) — it must serialize cleanly, round-trip byte-stably, and be exhaustively matched.
  A trait object would wreck scripting (no clean serialize without `erased_serde`/registry ceremony,
  no exhaustive match, no stable on-disk form). It stays a serializable **enum** (§3h). Conversely,
  the *open* sets (`VoxelProducer`, `Command`, `Tool`, `ChunkStore` backend) are traits — the
  traits-vs-enums rule (§3h) decides each by openness, not by reflex.
- **Deferring the categorical material model (keep the 3-value enum + `GRID_OVERLAY_BIT` for now).**
  Rejected: the categorical block-palette cell touches the payload, override codec, serialization, and
  meshing simultaneously (§3a) — retrofitting it after sculpt/serialization ship is a cascade across
  every one of those. The per-voxel CAPABILITY is therefore foundational; only the rich palette
  content/picker is deferred. (The `GRID_OVERLAY_BIT`-in-`material_id` hack at `voxel.rs:82` is the
  exact "meaning in a data field" leak this removes.)
