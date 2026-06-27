# ADR 0004 — Agent-authoring & generative building stack

- **Status:** Proposed
- **Date:** 2026-06-27
- **Layer:** FEATURES-ON-TOP of [ADR 0003](0003-foundation-rework.md). **Not built until the ADR 0003
  foundation phases are underway** (concretely: the Phase C `Intent` door + Phase H `query`/`diagnostics`
  surface + the `scene` joint DATA seam must exist first — this ADR is the SOLVER and the producers and
  the agent loop that those reserved seams were reserved *for*). This ADR consumes ADR 0003 seams and
  mostly drops onto them additively. It needed a handful of cheap ADR 0003 DATA-seam / resolve-ordering
  edits (joint arity + a stable `JointId`; and three Thread-4 override/composition seams — N ordered
  override layers, ordered assembly composition, scene-root assembly-scoped overrides) — **all now
  reconciled INTO ADR 0003 in this same amendment, so Foundation-fit shows ZERO outstanding flags.**
  Everywhere else, additions are additive enum variants or new downward-depending crates.

## Context

ADR 0003 locked a foundation whose single door for all mutation is a serializable `Intent` enum
(`AppCore::apply_intent`), with a structured read surface (`query(SpatialQuery) -> Answer`), a
machine-and-human-readable check surface (`diagnostics() -> Vec<Issue>`), multi-view `render_png`, a
`scene` that is a node tree **plus a relationship/constraint graph** (scene-owned, id-keyed, **N-ary**
`Joint`s referencing other nodes by stable `NodeId`/`JointId` — ADR 0003 §1), a Fusion-360 part/assembly
model (definitions own producer geometry + sculpt overrides in a local frame; **arbitrary-angle geometry
is a producer parameter, SDF-exact at resolve, NOT a transform limit**; instances are reference +
24-orientation lattice rotation *of baked data* + i64 translation; plus scene-root assembly-scoped
override layers for this-site-only patches), a `VoxelProducer` registry with chunk-windowed
`resolve_into`, an ordered add/subtract layer compositor, a categorical per-voxel `block_id` + `attrs`
cell, and a streaming async store. ADR 0003 **deliberately reserved** the joint graph DATA seam but built
**no solver** (§1, lines 206–210), and deferred WFC narrowly to "a seeded detail/material FILL producer,
not a composition mechanism." This ADR designs exactly those deferred features as one coherent stack: the
systems that let an agent (or a human) say "build me a star fort" and converge through a short perceive →
diagnose → correct loop.

### The macro/micro thesis (the spine of this design)

There are two kinds of validity, and they need different machinery — conflating them is the central
design error this ADR avoids:

- **MACRO / relational** ("does the tower connect to the parapet?") — part-to-part. Handled by a
  **Fusion-style assembly constraint/JOINT SOLVER**: declarative relationships the system *solves* to
  place parts and *continuously validates*, surfaced through `query`/`diagnostics`. This is the spine
  of agent-driven building.
- **MICRO / local** ("is this merlon supported? do the stone bands follow height?") — block-to-block
  adjacency *within one surface of one part*. Handled by **WFC / constraint-FILL as a deterministic,
  seeded `VoxelProducer`** over a bounded part-local region — ONE tool among many, **not** the spine.

WFC is poor at the macro problem (it is a global stochastic solver; macro intent washes out under
propagation). So the macro intent is carried by the joint solver and a **parametric kit-of-parts**;
WFC only fills regular surface detail/material *after* the macro structure is committed and *inside*
boundaries the kit has already fixed.

### The two-phase invariant (SOLVE is sync; VALIDATE is async — ADR 0003 §7)

A second conflation this ADR is careful to avoid is between **placing parts** and **reading resolved
voxels**. ADR 0003 §7 (line 627) makes `apply` ONLY mutate the node tree, write the sparse delta, and
**mark chunks dirty — it does NOT resolve or mesh inline**. The stack honors that invariant by splitting
the solver's two jobs across the sync/async boundary:

- **Joint SOLVE — synchronous, transform-only, no voxels.** Pure transform/parameter math over the
  discrete lattice (placing parts: orientation + i64 offset + a solved module parameter). It reads only
  the joint/connector graph and anchor frames — *cheap metadata, never resolved occupancy* — and writes
  its results as `Intent::SetTransform`/`Intent::SetParam` through the command journal. Because it touches
  no voxels, it runs synchronously inside the agent loop with no resolve.
- **Geometry VALIDATE — asynchronous, post-integration.** Anything that reads *resolved voxels* (gaps,
  overlaps, surface occupancy, connector-on-surface checks) runs as a post-integration diagnostics pass:
  after the dirtied chunks have been resolved/meshed off-thread (ADR 0003 §7), `diagnostics()` reads the
  region-scoped occupancy and reports `Issue`s. It is *never* run inline inside `apply_intent`.

This is why the `SpatialQuery` taxonomy (C) is labeled **cheap-synchronous** (connector/joint-graph
math — `AnchorFrame`, `JointResidual`, `Connectors`, `ConnectorGap`, `NearestConnector`, `Connectivity`
over the joint graph) vs **async-resolved** (anything reading voxels — `Occupancy`, `Contact`, `Gap`,
`Overhang`, and the broad-phase `PartsOverlap`/connector-validity passes). The solver loop steers on the
synchronous half; the resolved-geometry half feeds `diagnostics()` after integration.

### Prior-art reuse map (reuse > build)

| Prior art | Use here | Build vs reuse |
|---|---|---|
| 4-layer model: massing → kit-of-parts → block-grid solve → manual sculpt | massing = existing SDF producers + booleans (ADR 0003 §3b) **+ the LLM/massing layer's rough macro layout**; kit = parametric modules (this ADR); block-grid = WFC fill producer; sculpt = existing override layer | reuse layers 1 & 4; build 2 & 3 |
| WFC unit = architectural MODULES, not raw blocks | WFC operates on a module/tile alphabet, never raw `block_id`s (avoids "voxel soup") | **build (informed)** — rules out raw-block WFC |
| parametric-prototype → module-extraction → WFC | kit modules double as WFC tiles; adjacency learned from authored prototypes | build, pattern reused |
| outer-optimizer-around-WFC | the optional VLM-critic slot wraps the loop, never the inner solver | reuse as optional outer loop |
| LLM-drives-documented-library (3D-GPT / Infinigen, RAG over API docs) | LLM emits `Intent`s against a documented kit/joint vocabulary **and does the rough macro LAYOUT** (e.g. bastions equidistant on a polygon); never voxels | **reuse pattern** |
| generate → validate → repair (Minecraft gen, Word2Minecraft retry-≤5) | the perceive → diagnose → correct loop, with a bounded retry budget | **reuse pattern** |
| off-the-shelf 3D voxel WFC w/ architectural modules | the seeded fill producer's inner engine | **reuse a crate where viable**, wrap behind `VoxelProducer` |
| typed/oriented connector frames as first-class geometry | the kit's anchors are promoted to queryable **connector** frames the agent steers (G2/G6) | **build (informed)** — connector-substrate pattern |

**Genuinely novel (no prior art — the load-bearing research bets):** (1) LLM massing bound to a
constraint solve in *categorical block-ID* space; (2) the block↔chisel two-scale seam where the ARCH
is the canonical cross-scale element (opening spans full blocks; the curve is sub-block chisel detail).
The cheapest-first research risk — *does discrete integer constraint propagation actually converge over
a coupled joint graph* — is probed before any kit or any AppCore plumbing is built (see Build plan, P0).

## Decision

One coherent stack across five areas. Every type below is **new code in new modules that depend
downward on `app_core`/`scene`/`store`/`core_geom`** (ADR 0003 §0 layering); the foundation-type touches
are the flagged DATA-seam edits in Foundation-fit (joint arity + `JointId`, plus the three Thread-4
override/composition seams now reconciled into ADR 0003 in this same amendment).

### The three-tier authoring model (the spine that unifies everything)

Every feature in this stack — and every seam it consumes from ADR 0003 — falls into exactly one of
three authoring tiers. Naming them makes the rest of this ADR (and ADR 0003 §3, which cross-references
this section) coherent:

- **Parametric (producers).** Arbitrary geometry — **angles, curves, terrain-following sweeps** —
  generated SDF-exact and **re-voxelized at resolve** (staircased on the voxel grid, inherent to
  voxels). Lives in part definitions / massing producers (ADR 0003 §3b/§3d producer layer stack). This
  is where *any-angle* geometry lives: an angled bastion or a curved wall is a producer **parameter**,
  not a transform — there is no 24-orientation limit on geometry (see F).
- **Relational (joints / solver).** How parts **place and connect** — the assembly graph, integer-
  lattice transforms (24-orientation `LatticeOrientation` + i64), solve + validate, overlap detection.
  Lives in the joint graph (ADR 0003 §1) + the solver (B). This tier is integer-exact precisely because
  it transforms *already-baked* parts on the lattice, never geometry-at-an-angle.
- **Corrective (override / patch layers).** Local **block replacement** — ordered layers, **later
  wins** (ADR 0003 §3b). Two scopes: **part-local sculpt** (belongs to a definition, propagates to all
  its instances) AND **assembly-scoped patches** (scene-root, world-frame, this-site-only, does NOT
  propagate — ADR 0003 §1). This is where bespoke junctions and agent block-replacement live (F).

The three tiers map one-to-one onto the foundation: parametric ↔ §3b/§3d producers; relational ↔ §1
joint graph; corrective ↔ §3b/§3e layer stack + the §1 assembly-scoped override layers. Each tier has
the *right* machinery for its job, and conflating them is the recurring design error this ADR avoids
(it is the same shape as the macro/micro split below, viewed across authoring rather than validity).

### A. The architectural kit + the cross-scale ARCH seam

A **kit** is a registry of **parametric modules** — wall segments (incl. the path-swept `Wall`, below),
tower rings, parapet/merlon courses, gatehouses, bastion wedges, arch pieces, and **junction modules**
(corner bond, T-junction, X-crossing, tower-at-crossing — F2) whose connectors mate two crossing walls. Architecturally each module is **either** a part
*definition generator* (it builds an `AssemblyDef` from parameters: producer layer-stack + part-local
sculpt + named connector frames) **or** a *composite generator* (it instantiates and joints other
modules). It is **driven entirely through `Intent`** — a module never touches `scene` directly; it
expands into a deterministic `Vec<Intent>` (create-def, add-instance, add-joint, set-param) that flow
through the one door.

```rust
// new crate `kit`, depends on `scene` + `app_core` (downward only)
pub trait KitModule: Send + Sync {
    fn id(&self) -> KitModuleId;
    fn params_schema(&self) -> ParamSchema;          // documented, RAG-able by the LLM (3D-GPT pattern)
    /// Deterministic expansion into the ONE door. No direct scene mutation.
    fn expand(&self, params: &KitParams, ctx: &ExpandCtx) -> Result<Vec<Intent>, KitError>;
    /// First-class connection frames in the module's local lattice frame (block-relative).
    fn connectors(&self) -> Vec<Connector>;          // e.g. "base_ring", "wall_north", "arch_springline"
}
pub struct Connector {
    pub name: ConnectorName,
    pub frame: LocalFrame,                            // lattice-aligned offset + LatticeOrientation
    pub kind: ConnectorKind,                          // Face | Edge | Springline | Course
    pub block_span: [u32; 3],                         // how many full blocks the connector face spans
    pub sub_block_snap: SubBlockSnap,                 // the chisel-grid snap for sub-block detail (G9)
}
```

**Connectors are FIRST-CLASS, queryable, block-snapped geometry (H6).** A connector is the kit's
contribution to a **joint**: when two modules are jointed, the solver mates their connector frames.
Connectors are typed, oriented, and block-snapped, expressed in the **definition-local lattice frame**
so a moved or rotated instance carries them (ADR 0003 part-local rule — G3). They are exposed to the
agent through the `Connectors` / `ConnectorGap` / `NearestConnector` `SpatialQuery` variants (C), so the
agent steers named sockets ("mate `wall_north` to `bastion.left_flank`") and G2 gap diagnosis is exact
*frame-math*, not a voxel scan.

**Connector-validity self-check (H6).** Because the whole loop is data-primary, a mis-authored connector
that reports `gap == 0` while the resolved geometry is actually wrong would silently poison every
downstream decision. So the async VALIDATE pass includes a **connector-validity check**: for each
declared connector frame, sample the producer's resolved occupancy at the frame and assert the connector
actually lies on the part's resolved surface. A connector floating in air (or buried inside solid)
emits `Issue::ConnectorOffSurface { node, connector }`. This catches authoring errors at the source
rather than letting them propagate as phantom "satisfied" joints.

- **The cross-scale ARCH seam (G9).** The arch is the canonical two-scale element: its OPENING spans
  several full blocks (a macro/kit concern — the springline connectors snap to the block lattice), while
  its CURVE is sub-block chisel detail (a part-local sculpt override on the arch definition, snapped to
  the chisel sub-grid via `SubBlockSnap`). The kit places the arch *block-relative* (connector frames on
  the block lattice); the curve lives as part-local sculpt overrides inside the arch `AssemblyDef`.
  Because overrides are part-local and address-anchored (ADR 0003 §3e/§3g), the curve follows the arch
  under move/rotate for free. **No new payload, no new transform type** — block placement uses the
  existing `LatticeOrientation` + i64 translation; sub-block curve uses the existing sculpt override
  layer; `SubBlockSnap` is a kit-side quantization helper, not a foundation field.

- **Walls that climb terrain — the path-swept wall producer (parametric tier).** A wall is **not** a
  box + length; it is a **cross-section `profile` swept along a 3D `path`**:

  ```rust
  pub struct Wall { pub path: Vec<[i64; 3]>, pub profile: WallProfile }  // a VoxelProducer
  ```

  A wall that climbs a hill is just a **path with rising Y**. All path complexity (the rise, bends,
  taper) lives **INSIDE the producer** as an SDF sweep, so it is arbitrary-shaped and re-voxelized
  exactly at resolve (the parametric tier — consistent with F: geometry is a producer parameter, never
  a transform angle). Critically, the producer's instance transform stays a clean lattice placement and
  its **endpoint connectors stay lattice-clean** (block-aligned springs at the path ends), so the joint
  solver (B) still mates walls by **integer connector position** — the climbing geometry never leaks
  into the relational tier.

- **Terrain is IMPORTED, not authored.** Terrain is modeled as an imported `VoxelProducer` (e.g. a
  heightmap producer registered like any other, ADR 0003 §3d). **OPEN RESEARCH ITEM — flagged, NOT
  decided:** Vintage Story has **no known terrain-export path**, so the import format must be
  **searched / invented / adapted**. Candidate formats: a heightmap (PNG / RAW / GeoTIFF) or a voxel
  format (`.vox` / `.schematic`). **This is a flagged research spike (TBD)** — we are *not* pretending
  the import format is settled; the spike must establish whether VS data can be extracted at all and in
  which format before the heightmap producer is specified.

- **Ground query — `GroundHeightAt` (drape on terrain at the massing/agent layer).** Add a surface-
  sample query to the **extensible** `SpatialQuery` taxonomy (ADR 0003 §6 / C below):
  `GroundHeightAt { x: i64, z: i64 }` (or a surface-sample variant) returns the terrain surface Y at an
  XZ column. The **massing / agent layer** uses it to compute a wall's 3D `path` by **draping** the wall
  across the terrain (sampling ground height along the run) **BEFORE** handing the finished path to the
  `Wall` producer. This keeps producers **pure** — there is **no producer→producer coupling** (the wall
  producer never reads the terrain producer); the agent layer does the draping and passes a concrete
  path down. If terrain isn't present, the agent supplies the 3D `path` directly. `GroundHeightAt` rides
  the async-resolved half (it reads the imported terrain's resolved occupancy).

- **Footing / skirt sub-producer + diagnostic.** Over uneven terrain a swept wall can leave voids
  beneath it (the flat-bottomed profile bridges a dip). An **optional footing/skirt sub-producer** fills
  the voids under a wall down to the ground surface. The matching diagnostic — **"wall base doesn't meet
  ground / floating segment"** — is added to the `Issue` taxonomy (C) as `Issue::WallBaseGap` so the
  agent can detect an under-wall void and add the footing (or adjust the path).

### B. The constraint / joint SOLVER — solve + validate + conflict-detection

This is the spine. It consumes the reserved `Node.joints` seam (ADR 0003 §1) and adds solve/validate
logic in a new `solve` crate (depends on `scene` + `app_core`). The `Joint`/`JointKind` DATA serialize
with the document; the solver reads them and *emits placement `Intent`s* (or `Issue`s), never mutating
`scene` directly. **Note:** the spine required two cheap edits to the §1 joint DATA seam — n-ary joint
refs and a stable `JointId` — now **reconciled into ADR 0003 §1** in this same amendment (Foundation-fit
FLAG 1/2). They are pure data-shape changes with no solver/behavior impact on the foundation.

**Joint vocabulary** (the agent's macro grammar — `JointKind` values; closed serialized enum per the
ADR 0003 traits-vs-enums rule):

```rust
pub enum JointKind {
    Mate     { a: ConnectorRef, b: ConnectorRef },          // coincident frames (tower base ⇔ ground ring)
    Flush    { a: ConnectorRef, b: ConnectorRef, axis: Axis },// coplanar faces (parapet top ⇔ wall top)
    Adjacent { a: ConnectorRef, b: ConnectorRef, gap: i64 },// separated by exactly `gap` blocks
    Span     { wall: ConnectorRef, ends: [ConnectorRef; 2] },// N-ARY: a curtain bridging TWO fixed bastions
    Concentric { a: ConnectorRef, b: ConnectorRef },        // ring/tower coaxial
    Fixed,                                                   // pin an instance's transform (anchor to world)
}
pub struct ConnectorRef { pub node: NodeId, pub connector: ConnectorName }
```

`Span` is **n-ary by construction** — one joint referencing ≥2 other nodes (the curtain plus its two
fixed bastions). The reserved unary `Joint { other: NodeId, … }` cannot encode this; that is exactly
the §1 arity flag in Foundation-fit.

**The solver is a discrete propagation over the lattice, NOT a continuous numeric IK:**

- All joints reduce to constraints on each instance's `NodeTransform` (a 24-orientation
  `LatticeOrientation` + i64 translation — a *discrete* domain). `Mate`/`Concentric`/`Flush` pin
  relative orientation and offset exactly; `Adjacent` pins offset along an axis; `Span` is closed
  ONE-SHOT (below). This is exact integer arithmetic — no floating-point drift, deterministic,
  replayable (G7).
- **Solve order:** build a joint graph over `NodeId`s; pick `Fixed`/world-anchored nodes as roots;
  propagate transforms outward (a topological pass for the acyclic core, bounded fixpoint iteration for
  cycles — closed rings of curtain walls between towers are cycles and must converge or report).
- **Two-ended `Span` is solved ONE-SHOT, not by stretch-after-place (H4, G3).** A curtain wall between
  two already-fixed bastions is a **2-point boundary problem**: single-pass propagation cannot close it
  by placing one end then stretching, because both ends are pinned simultaneously. Instead the solver
  derives the span's length and orientation *up front* from `|A − B|` of the two fixed anchor frames
  (exact i64 vector between the two `ends` connectors) and emits the span's `SetTransform` + a
  `SetParam(length)` in one shot. The wall is built to fit the gap, never placed-then-stretched.
- **Macro LAYOUT is the LLM/massing layer's job, not the solver's (H4).** Putting four bastions
  *roughly* equidistant on a polygon is initial massing — the LLM/massing layer emits that rough layout.
  The solver does **relational closure and validation** between roughly-placed parts (snap the curtain
  exactly between the two bastions wherever the LLM put them); it is not a global layout optimizer.
- **Output is `Intent`s (synchronous, transform-only):** the solver computes target transforms/params
  and emits `Intent::SetTransform`/`Intent::SetParam` through `apply_intent`, so every solve step is a
  journaled, undoable command (G7, G3). Per the two-phase invariant, this SOLVE step touches no voxels
  and runs synchronously; it does not resolve or mesh (ADR 0003 §7).

**Validate (async, post-integration):** after a solve pass has been integrated (chunks resolved/meshed
off-thread per §7), the async `diagnostics()` pass re-checks each joint's *residual* — the actual
geometric relationship at the resolved grid vs the joint's intent — and emits `Issue`s for any
unsatisfied joint (the named G2 failure: tower not connecting to parapet). The cheap joint-graph residual
(`JointResidual`, frame-math only) is available synchronously; the resolved-geometry residual is the
async half.

**Conflict / over-constraint detection (G4) — the hard requirement: detect + report, never loop or
silently produce garbage. A graph-only transform solver is blind to two real G4 failures, so both get
explicit checks:**

- **Fan-in over-constraint (synchronous, in the solver — H3).** A node mated to *two* anchors is **not a
  cycle**, so cycle detection misses it. When solving a node, the solver computes the transform implied
  by **every incoming joint independently** and asserts they agree **i64-exact**. Any mismatch emits
  `Issue::OverConstrained { node, joints, demanded: Vec<Transform> }` instead of silently picking one
  demand. (Plain incompatible cyclic demands report the same way.)
- **Volumetric overlap — conflict vs JUNCTION (async broad-phase — H3, F).** The transform solver is
  blind to a part *occupying the same space as another*. So standing `diagnostics()` runs a
  **broad-phase AABB overlap pass** over all parts' world-AABBs; any overlapping pair triggers
  `Issue::PartsOverlap`. **The overlap is not assumed to be a conflict** — geometric union of the two
  parts is automatic, but a *good* junction is not, so the solver **classifies** the overlap (`kind:
  OverlapKind`) and, for a resolvable crossing, **proposes a resolution** (a kit junction module or a
  patch — `proposed: Some(Intent)`). See F for the full junction model. (Broad-phase AABB is cheap
  metadata; a narrow-phase voxel check is only run on the flagged pair if disambiguation is needed.)
  This is the check that makes "detects conflicts/junctions, never silent garbage" actually true — a
  graph solver alone cannot see interpenetration.
- **Termination (H5 — see the loop, C).** The propagation runs a **bounded retry budget**; convergence
  is budgeted best-effort, governed by a residual-magnitude potential plus an oscillation guard, **not**
  claimed as a proof. A group that exhausts the budget reports `Issue::SolveDidNotConverge` rather than
  spinning.

### C. The agent perceive → diagnose → correct loop (data-primary)

The loop is **DATA-PRIMARY**: the agent reasons from structured `query`/`diagnostics` answers; the
multi-view `render_png` is a *secondary gestalt channel* for sanity/vibe, never the primary signal
(rules out pixel-primary feedback). The query set (G6) is small and split by cost: the connector/joint-
graph queries are **cheap-synchronous** (frame-math, no resolve); the resolved-occupancy queries ride
the **region-scoped** read path (ADR 0003 §4) and are part of the async VALIDATE half.

**The `SpatialQuery` taxonomy** (extends the ADR 0003 `query(SpatialQuery) -> Answer` enum, §6 lines
557/565 — these are *new variants of an enum the foundation already exposes*, the canonical "consume the
reserved control surface" move; **not** a new method, **not** a foundation type change):

```rust
pub enum SpatialQuery {
    // --- cheap-synchronous: connector / joint-graph math, no resolved voxels ---
    AnchorFrame  { node: NodeId, connector: ConnectorName }, // resolved world frame of a connector
    Connectors   { node: NodeId },                           // a node's connector frames (typed/oriented)
    ConnectorGap { a: ConnectorRef, b: ConnectorRef },       // exact frame-math gap (G2, no voxel scan)
    NearestConnector { node: NodeId, to: NodeId },           // hint: which connectors SHOULD mate (no joint yet)
    JointResidual(JointId),                                  // actual vs intended relationship for one joint
    Connectivity { from: NodeId, to: NodeId },               // zero-residual joint-PATH A→B (graph, not flood-fill)
    // --- async-resolved: reads occupancy over the region-scoped path (§4) ---
    Bounds(NodeId),                                          // part/assembly AABB (world blocks)
    Contact   { a: NodeId, b: NodeId },                      // do they touch? shared face area
    Gap       { a: NodeId, b: NodeId },                      // nearest resolved-surface distance + face pair
    Overhang  { node: NodeId },                              // unsupported voxels (no block below)
    Occupancy { region: Aabb64 },                            // categorical block fill of a region
    GroundHeightAt { x: i64, z: i64 },                       // imported-terrain surface Y at an XZ column (A) —
                                                            // agent drapes a wall's 3D path on terrain before the producer
}
```

**Connectivity is computed over the connector/joint graph, NOT a voxel flood-fill (H6).** "Is the fort
one piece?" is answered by asking whether there is a path of *zero-residual joints* from A to B in the
connector graph — cheap, scene-independent, precise, and synchronous. A voxel flood-fill would be
scene-scale and would conflate incidental block adjacency with intended structural connection.

**`NearestConnector` is the no-joint-yet hint (H6).** For the G2 case where the agent hasn't declared a
joint yet (it sees a gap but doesn't know which sockets should mate), `NearestConnector { node, to }`
returns the best candidate connector pair so the agent learns which connectors SHOULD mate, then emits
the `Mate`/`Adjacent` joint.

**The `Issue` taxonomy** (extends the ADR 0003 `diagnostics() -> Vec<Issue>`, §6 lines 558/568 — again
*new enum variants on an existing surface*; ADR 0003 already routes orphaned-override + unsatisfied-joint
flags here). **Every `Issue` is self-describing and the list is priority-ordered (H7):**

```rust
pub enum Issue {
    // foundation-era (ADR 0003 §3g/§1):
    OrphanedOverrides { def: DefId, count: u32 },
    // this ADR — each variant carries enough inline geometry to act with ZERO follow-up query:
    UnsatisfiedJoint  {
        joint: JointId, connectors: [ConnectorRef; 2],   // the exact face/connector pair
        gap_blocks: i64,                                  // residual magnitude (feeds the potential, H5)
        intended: JointKind,
        fix_hint: Option<Intent>,                        // e.g. SetParam(length) / SetTransform nudge
    },
    OverConstrained   { node: NodeId, joints: Vec<JointId>, demanded: Vec<Transform> },
    SolveDidNotConverge { scc: Vec<NodeId>, iterations: u32, residual_blocks: i64 },
    PartsOverlap      {                                   // a "junction-to-resolve", NOT pure conflict (F)
        a: NodeId, b: NodeId, overlap_aabb: Aabb64,
        kind: OverlapKind,                               // Conflict (over-constrained) | Junction (resolvable)
        proposed: Option<Intent>,                        // solver's proposed resolution (kit module / patch)
    },
    ConnectorOffSurface { node: NodeId, connector: ConnectorName }, // connector-validity (H6)
    Disconnected      { components: Vec<Vec<NodeId>> },             // fort is in N pieces (graph)
    Unsupported       { node: NodeId, floating_voxels: u64 },
    WallBaseGap       { node: NodeId, gap_aabb: Aabb64 },          // wall base doesn't meet ground / floating segment (A)
    WfcContradiction  { region: Aabb64, cell: [i64; 3] },          // see D
}
pub enum OverlapKind { Conflict, Junction }   // see F: union is automatic; a GOOD junction is not
```

**Self-describing + well-ordered (H7).** Each `Issue` carries its own inline geometry — connector/face
refs, `gap_blocks`, the intended joint, and an optional `fix_hint: Option<Intent>` — so the agent acts
straight off `diagnostics()` with **zero follow-up query**. `diagnostics()` returns the list ordered
**anchors → structural → micro** (world-anchor/`Fixed` problems first, then joint/overlap/connectivity,
then WFC/support detail), so the descent is well-ordered: fixing the high-priority issues first prevents
the agent from chasing micro symptoms of a macro cause. The same `Vec<Issue>` renders in the human
inspector (ADR 0003 §6), so the human and the agent diagnose from identical data.

**The loop:** `apply_intent` (massing + kit instantiation + joints) → `solve` emits placement intents
(SYNC, transform-only) → chunks resolve/mesh off-thread (§7) → async `diagnostics()` (incl. broad-phase
overlap + connector-validity) → if empty, done; else the agent reads precise, ordered, self-describing
issues, emits local corrective intents (often the `fix_hint` verbatim), re-solves. **Bounded retry
budget** (Word2Minecraft-style ≤N) before escalating to the human.

**Termination — budgeted best-effort, NOT a proven guarantee (H5).** Convergence of a coupled discrete
constraint graph is **not provable here**, so this ADR does not claim a guarantee. Instead:

- **Potential over residual MAGNITUDE.** Progress is measured by a potential function = the **sum of
  `gap_blocks`** over all open issues (total residual magnitude), not the issue *count*. (Count can stay
  flat while magnitude shrinks, or one fix can split one issue into two smaller ones — magnitude is the
  honest progress signal.)
- **Oscillation guard.** Each pass hashes the **diagnostics multiset**; if a hash repeats, the loop is
  oscillating between equivalent states and **aborts** rather than spinning.
- **Bounded retry budget.** A max-pass ceiling caps the generate→validate→repair loop regardless; on
  exhaustion it escalates to the human with the standing `Vec<Issue>`.

So the loop is **budgeted best-effort, empirically measured** (P0 measures it), and "converges or
reports" is true by the budget + oscillation guard, not by a convergence proof.

### D. WFC as the deterministic seeded detail/material FILL producer

WFC is **a `VoxelProducer`** (ADR 0003 §3d registry) — one producer among many, the last detail layer,
**not** a composition mechanism. It is registered like `SdfShape`/`DebugClouds`:

```rust
pub struct WfcFill {
    pub region: Aabb64,                 // BOUNDED, part-local — never the whole scene, never across a joint
    pub tileset: TilesetId,            // architectural MATERIAL tiles (bands, variants), NOT raw blocks
    pub seed: u64,                     // determinism (G7)
    pub occupancy: FixedOccupancy,     // the kit-fixed solid/void mask WFC fills WITHIN (never alters)
}
impl VoxelProducer for WfcFill {
    fn resolve_into(&self, chunk_box: ChunkBox, out: &mut VoxelRegion, frame: LocalFrame) { /* … */ }
    fn world_aabb_blocks(&self, xf: &NodeTransform) -> Option<Aabb64> { Some(self.region /*…*/) }
    // …kind/serialize…
}
```

**Scope resolution — rhythm is KIT, material is WFC, within FIXED occupancy (H9):**

1. **Merlon / crenellation RHYTHM is KIT geometry, not WFC occupancy.** Where the solid/void of a
   crenellated parapet lives is a kit concern: a `PatternAlong`/`Parapet` parameter on the parapet module
   (period, merlon width, crenel width). Because it is a kit parameter, the rhythm **survives macro
   intent by construction** — it is part of the committed occupancy, not something WFC can wash out. This
   resolves the "merlon-is-occupancy" contradiction: occupancy (incl. crenellation) is fixed by the kit
   *before* WFC runs.
2. **WFC does ONLY material banding / variant fill within FIXED occupancy.** WFC never decides solid vs
   void; the kit + solver fix the occupancy mask first, and WFC chooses *which `block_id`/variant* fills
   each already-solid cell (stone base → brick → weathered crenellation). Material *bands* are
   height-keyed tile-adjacency rules, so "material follows height" is structural, not stochastic.
3. **Cross-part band continuity uses a shared world-Y-keyed deterministic rule, NOT a WFC region
   spanning a joint (H9).** The corner-seam washout (bands not lining up where a wall meets a bastion)
   is avoided by *not* running one WFC region across the joint. Instead each part's WFC is keyed to a
   **shared, deterministic, world-Y-indexed band rule** (the band at world-Y *k* is the same material on
   both parts by construction). Continuity is enforced by the shared key, not by propagation across a
   boundary.
4. **Determinism:** a fixed `seed` + a fixed occupancy → byte-identical fill every run (G7), so the
   agent can reason, undo, and retry. Output is categorical `block_id`+`attrs` cells through the same
   store/codec as any producer.
5. **Contradiction = a precise issue, never garbage.** If propagation hits a contradiction (the
   material tileset cannot satisfy the fixed occupancy + band rule), WFC does **not** backtrack into soup
   or fail silently — it emits `Issue::WfcContradiction { region, cell }` so the agent loosens the
   tileset locally. This is the macro-survives-propagation failure made *observable and local*.

**Off-the-shelf:** wrap an existing 3D voxel/tile WFC engine behind the `VoxelProducer` trait rather
than writing a solver from scratch; only the fixed-occupancy plumbing, the world-Y band key, and the
material-tileset authoring are bespoke.

### E. The LLM / VLM role

- **The LLM emits PARAMETERS / INTENTS against the documented kit + joint vocabulary, NEVER voxels**
  (well-supported by prior art; 3D-GPT/Infinigen RAG-over-docs). It also owns the **rough macro LAYOUT**
  (placing bastions roughly equidistant on a polygon, H4) — the solver then does relational closure
  between those roughly-placed parts. Its entire output surface is the `Intent` enum (kit instantiation,
  joints, params, seeds, rough transforms) — the same door a human gizmo drag uses, so every LLM action
  is journaled, undoable, replayable. The `params_schema()` of each `KitModule` and the
  `JointKind`/`SpatialQuery`/`Issue` taxonomies ARE the documented library the LLM is grounded on (RAG
  over these schemas).
- **The VLM critic is OPTIONAL and OUTER** (the outer-optimizer-around-WFC slot): it consumes
  `render_png` multi-view and emits *new* `Intent`s / param nudges to wrap the loop for "vibe it from a
  photo." It is never in the inner solve/validate loop and never the primary feedback (which stays
  data-primary, C). Image conditioning is a flavor knob, not the spine.

### F. Angled bastions (a producer parameter, NOT a limit) + intersecting-wall junctions

#### F1. Angled bastions are a producer-parameter concern — there is no geometry limit

A star fort's defining feature is the **angled bastion** (e.g. a 45° wedge). This is **fully supported
and not a limitation**: an angled bastion is **parametric geometry** (the parametric tier of the
three-tier model). The bastion module's `AssemblyDef` produces the wedge as an **SDF rotated by the
desired angle and voxelized fresh at resolve** — exact-from-the-field, just staircased on the voxel grid
(inherent to voxels, and exactly what VS chiseling looks like). Curves, tapers, and any angle are the
same kind of producer parameter. **The angle is geometry, expressed in the producer; it is never a
transform rotation**, so the discrete `LatticeOrientation` transform space is **never asked to express
45°** and there is no contradiction to confront.

The reason the 24-orientation lattice does **not** limit this: `LatticeOrientation` constrains only the
**instance-transform rotation of already-baked voxel data** (the relational tier — an exact index
permutation, lossless/byte-stable, which is what keeps the sculpt-override layer intact and the solver
integer-exact; ADR 0003 §3f). It was **never** a geometry limit. The bastion's connector frames stay
**block-aligned** (its flanks expose axis-aligned connector faces even though the wedge surface between
them is angled), so the solver only ever mates block-aligned connectors and joints stay axis-aligned —
while the *geometry* between those connectors is any angle the producer wants.

**Deferred opt-in (the only place angle meets the transform):** rotating a **sculpted** (non-parametric)
part to a **non-lattice** angle would require **lossy resampling** of its sparse override layer; that is
a deferred, flagged opt-in ("this resamples/bakes the sculpt layer" warning), never the default (ADR
0003 §3f case 3 / Deferred). A *parametric* angled part has no such cost — it re-voxelizes from the field
for free. Enriching `LatticeOrientation` beyond 24 for inter-part angled mates of *baked* data remains
the noted alternative (a foundation change, deliberately avoided), but it is **not needed** for angled
bastions, which are producer geometry.

#### F2. Intersecting walls & junctions — overlap is a "junction-to-resolve", not pure conflict

When two walls cross, the **geometric union is automatic** (the compositor unions the parts). A **GOOD
junction**, however — proper bonding/coursing, matched material, or inserting a tower at the crossing —
**is not automatic**. The honest model:

- **`PartsOverlap` is reframed from pure-conflict to "junction-to-resolve."** The broad-phase AABB
  overlap pass (B / G4) **detects** the crossing; the solver then **classifies** it
  (`OverlapKind::{Conflict, Junction}`) and, for a junction, **PROPOSES a resolution**
  (`PartsOverlap.proposed: Some(Intent)`). A `Conflict` is a real over-constraint (a part forced into
  space that cannot host a sensible junction); a `Junction` is a resolvable crossing.
- **Common junctions are parametric kit MODULES.** Corner bond, T-junction, X-crossing, and
  tower-at-crossing are `KitModule`s whose connectors **mate both crossing walls** (the relational tier
  resolves them via joints). The solver's proposed `Intent` instantiates the appropriate junction module
  and joints it to both walls.
- **Bespoke junctions are handled by override / patch layers (corrective tier).** Where no kit module
  fits, the case is resolved by an **assembly-scoped override layer** (ADR 0003 §1 — world-frame,
  this-site-only). This is the same wrinkle as agent block-replacement (F3) and is placement-specific,
  so it must NOT change other instances of either wall — exactly what the assembly-scoped (not
  part-local) override layer provides.
- **Honest scope:** overlaps are **always detected**; **common cases are auto-handled by kit modules**;
  **bespoke cases are handled by agent/human patches**; it is **never silently wrong.** Full-generality
  *automatic aesthetic* junction resolution (a system that always produces a beautiful bond for any
  crossing) is a **hard, unsolved problem** — this ADR does **not** claim it; it claims detect +
  common-case-kit + bespoke-patch + never-silent.

#### F3. Agent block-replacement = ordered override / patch layers (corrective tier)

"Layer in replacement blocks in a section" is exactly the **override layer at REGION granularity**:
replace the `block_id`s over a region, composited **LAST so it wins** (ADR 0003 §3b later-wins +
§4 ordered assembly composition). For a placement-specific junction this is an **assembly-scoped**
override (ADR 0003 §1), so it does not propagate to other instances. It runs through the **same
Intent / loop** as everything else: a `PartsOverlap`/junction diagnostic → the agent places a patch
(or the proposed junction module) → re-solve → re-diagnose, under the bounded retry budget (C).

This is the wrinkle that drove the three ADR 0003 seam additions (now reconciled — see Foundation-fit):
an intersection is between two **instances in the assembly**, but the §3e sculpt layer is **part-local**
(propagates to all instances). A junction is **placement-specific** and must not change other instances,
which is why ADR 0003 now makes explicit: (1) **N ordered override/patch layers, later-wins** (§3b);
(2) **ordered/prioritized assembly composition** so a junction module / patch wins over the walls in its
region (§4); (3) **scene-root assembly-scoped override layers** that do not propagate to definitions (§1).

## Acceptance criteria — G1–G10 walkthrough

| # | Stress case | How the design satisfies it |
|---|---|---|
| **G1** | "build me a star fort" end-to-end; does the loop TERMINATE? | LLM emits rough macro layout + kit-instantiation + joint `Intent`s (E/A) → solver propagates transforms over the discrete lattice (B, SYNC) → chunks resolve off-thread (§7) → async `diagnostics()` → local corrections → re-solve. Termination is **budgeted best-effort (H5)**, NOT a proof: a **residual-MAGNITUDE potential** (sum of `gap_blocks`) + an **oscillation guard** (diagnostics-multiset hash) + a **bounded retry budget**. **Converges or reports — empirically measured by P0, never claimed as guaranteed.** |
| **G2** | The named failure: tower doesn't connect to parapet | `diagnostics()` emits `Issue::UnsatisfiedJoint` carrying the **exact connector pair + `gap_blocks` + intended joint + `fix_hint`** (C/H7) — and if no joint is declared yet, `NearestConnector` (H6) tells the agent which connectors SHOULD mate. Gap diagnosis is exact **connector frame-math** (`ConnectorGap`), not a voxel scan. Agent fixes with ONE local `Intent` (often the `fix_hint` verbatim), not a re-roll. **Precise + local + self-describing.** |
| **G3** | Move/rotate a whole bastion | The bastion is an assembly instance; its joints, connectors, sculpt, and materials are **part-local** (ADR 0003 §3f) so they follow under the `LatticeOrientation` + i64 transform for free; **dependent joints re-solve** (B emits new placement intents) or, if now conflicting, **report drift** via `UnsatisfiedJoint`/`OverConstrained`. A curtain `Span` between two moved bastions re-closes **ONE-SHOT** from the new `|A − B|` (H4), never stretch-after-place. |
| **G4** | Over-constrained / conflicting joints AND intersecting-wall junctions | Two real blind-spots are both checked: **fan-in over-constraint** — solving a node, the solver asserts EVERY incoming joint's implied transform agrees **i64-exact**; mismatch → `Issue::OverConstrained` (H3, catches the not-a-cycle case cycle-detection misses); **volumetric overlap** — a **broad-phase AABB overlap pass** in standing `diagnostics()` → `Issue::PartsOverlap`, **classified `OverlapKind::{Conflict, Junction}`** (F2): a **conflict** is a real over-constraint; a **junction** (two walls crossing) is *resolvable* — the geometric union is automatic, a good junction is not, so the solver **proposes** a kit junction module (corner/T/X/tower-at-crossing) or, for bespoke cases, an **assembly-scoped override patch** (corrective tier). Always detected; common cases auto-handled; bespoke handled by agent/human patches; **never silently wrong** (full-generality automatic aesthetic junctioning is a hard problem, not claimed). Plus the bounded budget + oscillation guard (H5) so it **never loops forever**. |
| **G5** | WFC fill within a curtain-wall surface (bands + merlons) | Merlon **RHYTHM is KIT geometry** (`Parapet`/`PatternAlong` param — survives macro intent by construction, H9); **WFC does ONLY material banding within FIXED occupancy** (never solid/void); cross-part band continuity uses a **shared world-Y-keyed deterministic rule, NOT a WFC region across the joint** (kills the corner-seam washout, H9). Bounded part-local region, seeded (deterministic); a contradiction is `Issue::WfcContradiction`, not soup. (D) |
| **G6** | Feedback bandwidth: minimal sufficient query set, cheap? | The `SpatialQuery` set (C) is split by cost: **cheap-synchronous** connector/joint-graph math (`Connectors`/`ConnectorGap`/`NearestConnector`/`JointResidual`/`Connectivity`/`AnchorFrame`) and **async-resolved** occupancy reads (`Bounds`/`Contact`/`Gap`/`Overhang`/`Occupancy`) over the region-scoped path. **`Connectivity` is a zero-residual joint-PATH over the graph, NOT a voxel flood-fill** (H6) — cheap and precise. Data-primary; `render_png` secondary. |
| **G7** | Determinism / replay | Everything flows through the serializable `Intent` door (ADR 0003 §6a); solver math is exact integer lattice arithmetic; WFC is `seed`+fixed-occupancy deterministic. **Same intent script → same building**, undoable/retryable. |
| **G8** | Scale: fort over large XZ; loop stays responsive | Queries/diagnostics split sync vs async (§7); the SYNC solve works on transforms/connectors (cheap metadata), the async VALIDATE uses the **region-scoped** store read (ADR 0003 §4) and never forces a full resolve; broad-phase overlap is AABB-only; WFC fill is a per-chunk producer job under the existing **per-frame budget** (ADR 0003 §7). The loop **does not fight the budget**. |
| **G9** | The macro/micro ARCH seam + arbitrary-angle geometry | Opening spans full blocks (kit connectors on the block lattice, `LatticeOrientation`+i64 placement); curve is sub-block chisel detail (part-local sculpt override, `SubBlockSnap` to the chisel sub-grid). The kit handles cross-scale placement block-relative; **no new payload/transform** (A). **Arbitrary-angle geometry (45° bastion wedge, curves) is NOT a limit** — it is **parametric producer geometry**, SDF-rotated and re-voxelized exact-from-the-field at resolve (staircased, as voxels are); the 24-orientation `LatticeOrientation` constrains only **instance rotation of already-baked data** (relational tier, lossless index permutation), never geometry. Connectors stay block-aligned so joints stay axis-aligned while the geometry between them is any angle (F1). The only transform-meets-angle case — non-lattice rotation of a *sculpted* part — is a deferred flagged lossy-resample opt-in (ADR 0003 §3f). |
| **G10** | The cheap load-bearing probe FIRST | Yes — and it is now **foundation-free** (H10): a pure `solve(&Scene) -> Vec<TransformResult>` unit test on **hand-built structs**, zero AppCore/Intent/diagnostics plumbing, isolating the one novel spine risk (does discrete integer propagation converge). The full Intent-script probe is gated behind ADR 0003 Phases C/F/H. (P0/P1 below.) |

## Foundation-fit

**Foundation-fit: ZERO outstanding flags — all required ADR 0003 seam changes are reconciled into 0003
in this same amendment; everything else consumes existing seams.** The stack is overwhelmingly additive
— additive enum variants on growth surfaces + new downward crates. The five DATA-seam edits it needs
back to ADR 0003 — the **joint arity** + stable **`JointId`** (§1), and the three Thread-4 items
(**N ordered override layers** §3b, **ordered/prioritized assembly composition** §4, **scene-root
assembly-scoped override layers** §1) — are **all now present in ADR 0003** (added by this same pass,
while 0003 is still Proposed). All are pure data-shape / resolve-ordering changes with no solver impact;
**none remains outstanding**. The two original flags are retained below for the record, now marked
**RECONCILED**.

### ADR 0003 seam changes — RECONCILED (no longer outstanding)

- **FLAG 1 — Joint arity (RECONCILED).** The reserved `Joint` was UNARY (`pub other: NodeId`) and could
  not encode an n-ary `Span` (a curtain wall bridging **two** fixed bastions is **one joint referencing
  ≥2 nodes**). **ADR 0003 §1 now defines `pub struct Joint { pub id: JointId, pub refs: Vec<NodeId>, pub
  kind: JointKind, … }` — scene-owned, id-keyed, and N-ary** (see §1 "Constraint / joint data seam"). The
  n-ary `Span` is expressible; this flag is closed. (Pure serialization shape — ADR 0003 builds no
  solver.)

- **FLAG 2 — Stable `JointId` (RECONCILED).** Joints were positionally addressed in a per-node
  `Vec<Joint>` with no stable id, so `Issue::UnsatisfiedJoint { joint: JointId, … }` /
  `SpatialQuery::JointResidual(JointId)` could not durably name a joint across undo. **ADR 0003 §1 now
  mints a stable `JointId` (like `NodeId`, from a document-owned counter) and stores joints in a
  scene-owned `joints: SlotMap<JointId, Joint>`.** This ADR's `Issue`/`SpatialQuery` surface names joints
  stably; this flag is closed.

- **FLAG 4 — N ordered override/patch layers, later-wins (RECONCILED — Thread-4 item 1).** A junction
  needs the part-local sculpt layer and **multiple agent patch layers** to coexist deterministically.
  **ADR 0003 §3b now makes the layer-stack semantics explicit: a `Vec<Layer>` of arbitrary length,
  applied in order, LATER LAYERS WIN** — sculpt + N agent patches compose deterministically. Closed.

- **FLAG 5 — Ordered/prioritized assembly composition (RECONCILED — Thread-4 item 2).** The resolve
  previously **unioned all leaves** with no precedence, so a junction module / patch could not win over
  the walls it crosses. **ADR 0003 §4 now specifies the assembly resolve composes leaves in defined
  order, later-wins, with scene-root assembly overrides composited LAST** over the parts in their region.
  Closed.

- **FLAG 6 — Scene-root assembly-scoped override layers (RECONCILED — Thread-4 item 3).** An
  intersection is between two **instances**, but §3e sculpt is **part-local** (propagates to all
  instances); a placement-specific junction must NOT change other instances. **ADR 0003 §1 now adds
  scene-root `assembly_overrides: Vec<Layer>` — world-frame, this-site-only, does NOT propagate to
  definitions** (a deliberate, clearly-distinguished bend of the def-only-geometry rule; Fusion
  assembly-context-feature precedent), serialized via the same override codec (§5) and composited last
  (§4). Closed.

All six are **cheap DATA-seam / resolve-ordering changes with no solver/behavior impact on the
foundation**, and all were made **now while 0003 is Proposed**, avoiding a schema cascade later.
Everything else in this stack consumes existing seams with no foundation change. **No flag remains
outstanding.**

**ADR 0003 seams consumed (no change required):**

- **`Intent` door** (§6a) — kit expansion, joint creation, solver output, LLM/VLM all emit `Intent`s
  through `apply_intent`. New `Intent` *variants* are added (kit-instantiate, add-joint, set-param,
  set-transform) — additive enum growth, the foundation's intended extension mode.
- **`query(SpatialQuery) -> Answer`** (§6, lines 557/565) — the C taxonomy adds *variants* to the
  existing enum (the §6 surface is designated extensible: "contact / gap / overhang / connectivity /
  bounds", line 566).
- **`diagnostics() -> Vec<Issue>`** (§6, lines 558/568) — the C `Issue` taxonomy adds *variants*; ADR
  0003 already declared this surface carries "unsatisfied joints" and orphaned overrides (line 570).
- **`scene` joint graph** (§1) — the scene-owned, id-keyed, **N-ary** `Joint` / `JointKind` are consumed
  as the solver's input; ADR 0003 reserved the DATA seam and explicitly left the SOLVER to "their own
  future design doc" — this is that doc. (Consumed *as amended/reconciled by FLAG 1 + FLAG 2*.)
- **Part/assembly model + override layers** (§1, §3b, §3e, §3f) — kit modules generate `AssemblyDef`s;
  placement is the existing `LatticeOrientation` + i64 translation; **arbitrary-angle geometry is a
  producer parameter, not a transform** (F1, §3f case 1); part-local sculpt is the existing layer;
  **junctions/agent block-replacement consume the now-explicit N ordered layers (§3b), ordered assembly
  composition (§4), and scene-root assembly-scoped override layers (§1)** — reconciled as FLAG 4/5/6.
- **`VoxelProducer` registry + `resolve_into`** (§3d) — `WfcFill` is one registrant.
- **Categorical `block_id` + `attrs` cell** (§3a) — WFC material output and kit material bands write
  through it (exactly the "named blocks, not 0/1/2" capability §3a foundationalized for this).
- **`Tool` trait + registry** (§3h) — interactive kit-placement / joint-draw tools register here.
- **Region-scoped read path + per-frame budget + async store** (§4, §7) — the async VALIDATE half and
  the WFC fill job ride these; nothing forces a full resolve; the SYNC solve touches no voxels at all.

**The sync/async split is enforced, not assumed (§7).** Per ADR 0003 §7 (line 627), `apply` ONLY mutates
the tree + marks chunks dirty; it does **not** resolve or mesh inline. The solver's SOLVE step (B) emits
transform-only intents and reads no voxels, so it composes with that invariant directly; everything that
reads resolved voxels (geometry residuals, broad-phase overlap, connector-validity, `Gap`/`Contact`/
`Occupancy`) is the **async post-integration** VALIDATE pass feeding `diagnostics()`, never inline.

**FLAG 3 (watch-item, not a required change):** the joint solver emits `Intent::SetTransform`/`SetParam`
*programmatically and in bulk*. ADR 0003's command stack has per-drag `coalesce_key` for human input;
the solver should wrap one solve pass as **one coalesced undo entry** so a re-solve is a single undo.
This is satisfiable with the existing `CoalesceKey` mechanism (no foundation change) — recorded here so
the implementer uses it. If a future solve genuinely needed a transactional multi-intent atom that
`CoalesceKey` cannot express, *that* would be a further flag — it does not today.

## Incremental build plan (cheap load-bearing probe FIRST)

Each step is a green checkpoint. **No kit and no WFC are built until the cheap probes pass** — they
de-risk the novel bets before any expensive authoring. The genuinely cheapest first slice is
**foundation-free** (H10): ADR 0003 is only ~5% built (no AppCore/Intent/query/diagnostics/joints yet),
so the very first probe must not depend on any of that plumbing.

- **P0 — THE cheapest, foundation-free spine probe (G10/G1/H10), the literal first thing:** a pure
  `solve(&Scene) -> Vec<TransformResult>` **unit test on hand-built structs** — zero AppCore, zero
  `Intent`, zero `diagnostics()` plumbing, zero store. Hand-construct a tiny scene (4 parts, a couple of
  `Mate`/`Adjacent` + one two-ended `Span`) and assert the solver returns the right transforms AND that
  the **residual-magnitude potential + oscillation guard (H5)** make the loop converge or report on a
  deliberately over-constrained case. This isolates the **one novel spine risk: does discrete integer
  propagation actually converge** — measured empirically, since H5 makes no proof. *If P0 does not
  converge cleanly, the spine is wrong — stop and fix before anything else, with nothing else built.*
- **P1 — the WFC-intent-survival probe (the second probe, H9-aimed):** fill ONE bounded surface
  **at a JUNCTION** — a wall meeting an angled/curved bastion, or a corner — with a small height-banded
  **material** tileset over **fixed occupancy**, with the shared world-Y band key across the seam, fixed
  seed. Measure **pattern-intent-survival**: course alignment / band-period divisibility **across the
  pins** at the junction (not mere boundary-equality of two abutting cells). Does the band intent survive
  at the seam, or wash out / contradict? **Reuse:** an off-the-shelf 3D WFC crate behind `VoxelProducer`;
  **build:** fixed-occupancy pinning + the world-Y band key + the contradiction → `Issue::WfcContradiction`
  path. *This answers the KEY open WFC question at the hardest spot (the junction) before any kit
  investment.*
- **P2 — the full P1-equivalent Intent-script probe (GATED behind ADR 0003 Phases C/F/H):** once the
  `Intent` door (Phase C), rotation (Phase F), and `query`/`diagnostics` (Phase H) exist, re-run the P0
  scenario as an **Intent script** end-to-end: place 4 part instances under joints, `query(ConnectorGap)`,
  read a self-describing `Issue::UnsatisfiedJoint`, apply its `fix_hint` `Intent`, re-solve to empty
  diagnostics — exercising the joint DATA seam → solver → query/issue → corrective-intent loop through the
  real door. **Build:** the additive `SpatialQuery`/`Issue`/`Intent`/`JointKind` variants; the broad-phase
  overlap + connector-validity passes.
- **P3 — conflict, junction & convergence hardening (G4):** add `Concentric`/`Flush`/n-ary `Span`, cyclic
  curtain rings (bounded fixpoint), the **fan-in over-constraint** check (i64-exact agreement of all
  incoming joints), the **broad-phase AABB overlap** pass with **`OverlapKind` classification +
  `proposed` resolution** (F2), and `OverConstrained`/`SolveDidNotConverge`/`PartsOverlap`/
  `ConnectorOffSurface`. Stress with a deliberately over-constrained gatehouse-vs-tower AND a fan-in node
  mated to two anchors AND **two crossing walls (junction-to-resolve, not conflict)**. (Consumes ADR 0003
  §3b N-layers + §4 ordered composition + §1 assembly overrides for bespoke-junction patches.)
- **P3-terrain (research spike, FLAGGED — Thread 3):** the **VS terrain-import format spike** — establish
  whether/how VS terrain can be extracted (heightmap PNG/RAW/GeoTIFF vs `.vox`/`.schematic`), then a
  minimal **imported heightmap `VoxelProducer`** + the `GroundHeightAt` query + a path-swept `Wall`
  draped on it + the footing/skirt sub-producer and `Issue::WallBaseGap`. **Format is TBD until the spike
  lands** — do not specify the heightmap producer before it.
- **P4 — the architectural kit:** author the first modules (wall segment, **path-swept `Wall`**, tower
  ring, parapet course with the `Parapet`/`PatternAlong` merlon rhythm, gatehouse, **angled bastion wedge
  as parametric producer geometry + block-aligned connectors** (F1), **junction modules — corner / T / X
  / tower-at-crossing — mating two walls** (F2), ARCH with cross-scale springline connectors + sub-block
  curve). **Reuse:** parametric-prototype → module-extraction pattern (kit modules double as P1's WFC
  tiles).
- **P5 — the full agent loop:** wire the LLM to the documented kit/joint/query/issue vocabulary (RAG over
  `params_schema()` + the taxonomies), with the LLM owning rough macro layout; run G1 "star fort"
  end-to-end with the bounded retry budget + oscillation guard. **Reuse:** generate → validate → repair
  pattern; the ADR 0003 Phase K MCP/socket as the agent transport.
- **P6 — the optional VLM critic** (outer loop, `render_png`-conditioned) — last, optional, a flavor knob.

**Reuse-vs-build summary:** *reuse* — the entire ADR 0003 control surface, joint DATA seam (as amended by
FLAG 1/2), producer registry, categorical cell, async/region-scoped read path, an off-the-shelf WFC
engine, and the documented-library-LLM + generate-validate-repair patterns. *Build* — the discrete-lattice
joint solver with fan-in + broad-phase overlap (incl. **junction classification + proposed resolution**)
checks (B/F2), the kit modules incl. **path-swept walls + junction modules** (A/F2), the WFC
fixed-occupancy + world-Y band key + material tileset (D), the **imported-terrain heightmap producer +
`GroundHeightAt` + footing/skirt** (A — gated on the flagged VS-export research spike), and the additive
query/issue/intent/joint enum variants. *Research (TBD)* — the **VS terrain-import format** (no known
export path; spike before specifying).

## Consequences

**Better**
- The macro/micro split gives each problem the right machinery: a sound discrete joint solver for
  relational validity, WFC strictly subordinate for **material within fixed occupancy**. Macro intent
  survives propagation because merlon RHYTHM is kit geometry and WFC only bands material — intent is
  structural, not soft bias.
- The agent loop is **data-primary, precise, self-describing, and budget-bounded**: ordered,
  self-describing issues (with `fix_hint`) → local fixes → budgeted best-effort convergence with an
  oscillation guard; the same diagnostics serve the human inspector.
- The SOLVE/VALIDATE split honors ADR 0003 §7: synchronous transform-only solve, asynchronous
  resolved-geometry validation — edits stay instant, no inline resolve.
- Determinism end-to-end (integer solver + seeded WFC + the `Intent` journal) makes the whole stack
  replayable, undoable, and testable as `Intent` scripts on the headless `AppCore`.
- The cheapest probe (P0) is **foundation-free**, so the single novel spine risk is de-risked before any
  AppCore/Intent plumbing or kit/WFC authoring exists.

**Costs more**
- Several cheap ADR 0003 DATA-seam / resolve-ordering edits (joint arity + `JointId`; the three Thread-4
  override/composition seams) were reconciled into the foundation in this same amendment while 0003 is
  Proposed — so the solver and the junction/patch work land with **zero outstanding foundation flags**.
- A discrete constraint solver with **fan-in over-constraint detection, broad-phase volumetric overlap,
  and cyclic-graph convergence** is real engineering (P0/P3), even bounded to the lattice — and its
  convergence is empirical, not proven.
- Authoring the kit modules (incl. parametric angled bastions, path-swept walls, and junction modules)
  and the WFC material tilesets/adjacency is ongoing content work, not just code.
- **Terrain import is an OPEN research item:** Vintage Story has no known terrain-export, so the import
  format (heightmap PNG/RAW/GeoTIFF vs `.vox`/`.schematic`) must be searched/invented/adapted by a spike
  before the heightmap producer can be specified — it is **not decided**.
- New enum taxonomies (`SpatialQuery`/`Issue`/`JointKind`) plus their query implementations split across
  the cheap-synchronous and async-resolved read paths, plus connector-validity sampling.
- The LLM grounding (RAG over schemas) and the optional VLM critic are external-model integration with
  their own cost/latency, gated behind the ADR 0003 Phase K transport.

**Deferred**
- Free (non-lattice) joint placement / continuous IK — the solver stays discrete-lattice exact;
  arbitrary-angle *inter-part* mates of *baked* data are out. **(Arbitrary-angle *geometry* is NOT
  deferred — it is a producer parameter, SDF-exact at resolve; the angled bastion is producer geometry
  with block-aligned connectors, F1.)** Enriching the `LatticeOrientation` set beyond 24 for inter-part
  angled mates of baked voxel data is the noted alternative if a future need forces it. **Non-lattice
  rotation of a *sculpted* part** is a deferred flagged lossy-resample opt-in (ADR 0003 §3f case 3).
- Learned adjacency (auto-extracting WFC rules from example builds) — start with authored tilesets.
- Multi-objective outer optimization beyond the single VLM critic slot.
- Cross-part WFC (WFC spanning a whole assembly, or one WFC region across a joint) — explicitly out;
  WFC stays per-surface/per-part, with cross-part band continuity carried by the shared world-Y key, not
  by propagation across a boundary (H9).
- **Full-generality automatic aesthetic junction resolution** (a system that always produces a correct,
  beautiful bond for any arbitrary wall crossing) — a hard, unsolved problem, explicitly NOT claimed.
  The scope is: overlaps always detected, common junctions auto-handled by kit modules, bespoke handled
  by agent/human override patches, never silently wrong (F2).
- **Terrain authoring** (modeling terrain inside the app) — out; terrain is **imported** as a
  `VoxelProducer`, and the **import format is an open research spike** (no known VS export; F / build
  plan P3-terrain).

## Alternatives considered

- **WFC as the SPINE (global composition by WFC).** Rejected: WFC is a global stochastic solver; the
  macro "does the tower connect to the parapet?" problem either over-constrains into contradictions or
  washes into generic soup. The spine is the declarative joint solver; WFC is one subordinate fill
  producer doing material-within-fixed-occupancy only. (The P1 junction probe exists to confirm even the
  *subordinate* use survives propagation at the hardest spot before we invest.)
- **Generate-then-repair over voxel soup** (LLM/WFC emits raw voxels, a repair pass fixes it).
  Rejected: there is no precise relational signal to repair against, repairs are non-local, and it fights
  the part/assembly model. The LLM emits `Intent`s against a documented kit, never voxels, and repair is
  *local corrective intents* driven by precise, self-describing `Issue`s.
- **Pixel-primary agent feedback** (agent reasons mainly from `render_png`). Rejected: imprecise,
  expensive, non-deterministic to reason over, and unable to give exact gap vectors / connector pairs.
  Feedback is data-primary (`query`/`diagnostics`); render is a secondary gestalt channel only.
- **Raw-block WFC** (WFC alphabet = individual `block_id`s). Rejected: the empirical prior-art result is
  voxel soup; the alphabet is architectural MATERIAL tiles, with material bands as world-Y-keyed
  deterministic rules, and merlon rhythm lifted out of WFC into kit geometry entirely.
- **Stretch-after-place for the two-ended span.** Rejected: a curtain between two fixed bastions is a
  2-point boundary problem single-pass propagation cannot close; the span length/orientation is derived
  one-shot from `|A − B|` before placement (H4).
- **Strict monotone issue-COUNT termination guarantee.** Rejected as a false guarantee: a coupled
  discrete constraint graph's convergence is not provable here, and issue *count* is not a sound progress
  signal. Replaced by a **residual-magnitude potential + oscillation guard + bounded budget** (H5),
  framed as budgeted best-effort and empirically measured (P0).
- **Continuous numeric constraint solver / general IK.** Rejected: introduces float drift, breaks
  determinism/replay, and is incompatible with the lattice transform domain (`LatticeOrientation` + i64).
  The discrete propagation solver is exact and bounded.
- **Inter-part angled (45°) mates via an enriched rotation set.** Noted as the alternative but not
  chosen: it would change the foundation's `LatticeOrientation` (the lossless rotation of *baked* voxel
  data). It is **not needed for angled bastions** — those are **parametric producer geometry** (any
  angle, SDF-exact at resolve, block-aligned connectors so joints stay axis-aligned; F1). Enriching the
  rotation set is reserved only for a future need to losslessly mate *baked* parts at a non-lattice
  angle, which the angled-bastion case does not require.
- **A new foundation field for anchors/sub-block snap.** Rejected as unnecessary: connectors live in the
  part-local frame and the ARCH sub-block curve reuses the existing part-local sculpt override layer. (The
  two genuinely required foundation edits are the joint *arity* and *`JointId`* DATA-seam changes, flagged
  in Foundation-fit — not a new anchor/snap field.)
