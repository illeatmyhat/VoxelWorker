# ADR 0004 — Agent-authoring & generative building stack

- **Status:** Proposed
- **Date:** 2026-06-27
- **Layer:** FEATURES-ON-TOP of [ADR 0003](0003-foundation-rework.md). **Not built until the ADR 0003
  foundation phases are underway** (concretely: the Phase C `Intent` door + Phase H `query`/`diagnostics`
  surface + the `scene` joint DATA seam must exist first — this ADR is the SOLVER and the producers and
  the agent loop that those reserved seams were reserved *for*). This ADR consumes ADR 0003 seams and
  mostly drops onto them additively — **with two crisp exceptions: it requires two cheap DATA-seam edits
  back to ADR 0003 (joint arity + a stable `JointId`), flagged explicitly in Foundation-fit.** Everywhere
  else, additions are additive enum variants or new downward-depending crates.

## Context

ADR 0003 locked a foundation whose single door for all mutation is a serializable `Intent` enum
(`AppCore::apply_intent`), with a structured read surface (`query(SpatialQuery) -> Answer`), a
machine-and-human-readable check surface (`diagnostics() -> Vec<Issue>`), multi-view `render_png`, a
`scene` that is a node tree **plus a relationship/constraint graph** (`Node.joints: Vec<Joint>`
referencing other nodes by stable `NodeId` — ADR 0003 §1, lines 197–203), a Fusion-360 part/assembly
model (definitions own geometry + sculpt overrides in a local frame; instances are reference +
24-orientation lattice rotation + i64 translation), a `VoxelProducer` registry with chunk-windowed
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
downward on `app_core`/`scene`/`store`/`core_geom`** (ADR 0003 §0 layering); the only foundation-type
touches are the two flagged DATA-seam edits in Foundation-fit (joint arity + `JointId`).

### A. The architectural kit + the cross-scale ARCH seam

A **kit** is a registry of **parametric modules** — wall segments, tower rings, parapet/merlon
courses, gatehouses, bastion wedges, arch pieces. Architecturally each module is **either** a part
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

### B. The constraint / joint SOLVER — solve + validate + conflict-detection

This is the spine. It consumes the reserved `Node.joints` seam (ADR 0003 §1) and adds solve/validate
logic in a new `solve` crate (depends on `scene` + `app_core`). The `Joint`/`JointKind` DATA serialize
with the document; the solver reads them and *emits placement `Intent`s* (or `Issue`s), never mutating
`scene` directly. **Note:** the spine requires two cheap edits to the §1 joint DATA seam — n-ary joint
refs and a stable `JointId` — flagged in Foundation-fit. They are pure data-shape changes with no
solver/behavior impact on the foundation, and are best made now while ADR 0003 is still Proposed.

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
- **Volumetric interpenetration (async broad-phase — H3).** The transform solver is blind to a part
  *forced where another already occupies space*. So standing `diagnostics()` runs a **broad-phase AABB
  overlap pass** over all parts' world-AABBs; any overlapping pair triggers `Issue::PartsOverlap { a, b,
  overlap_aabb }`. (Broad-phase AABB is cheap metadata; a narrow-phase voxel check is only run on the
  flagged pair if disambiguation is needed.) This is the check that makes "detects conflicts, never
  silent garbage" actually true — a graph solver alone cannot see interpenetration.
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
    PartsOverlap      { a: NodeId, b: NodeId, overlap_aabb: Aabb64 },
    ConnectorOffSurface { node: NodeId, connector: ConnectorName }, // connector-validity (H6)
    Disconnected      { components: Vec<Vec<NodeId>> },             // fort is in N pieces (graph)
    Unsupported       { node: NodeId, floating_voxels: u64 },
    WfcContradiction  { region: Aabb64, cell: [i64; 3] },          // see D
}
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

### F. The 24-orientation limit vs angled bastions (confronted — H8)

A star fort's defining feature is the **angled bastion** (e.g. a 45° wedge), but ADR 0003 reserves only
24 orthogonal `LatticeOrientation`s — the discrete transform space cannot express 45°. This is confronted
head-on rather than left as a latent contradiction:

**Chosen approach — bake the bastion angle INTO the part definition.** The angled geometry is treated as
**intra-part sub-lattice detail**: the bastion module's `AssemblyDef` contains the 45° wedge as producer
geometry + part-local sculpt, with its connector frames still **block-aligned** (the flanks expose
axis-aligned connector faces even though the wedge surface between them is angled). Joints therefore stay
**axis-aligned** — the solver only ever mates block-aligned connectors, and the discrete transform space
is **never asked to express 45°**. The angle lives entirely inside one part, below the joint layer.

**Alternative (noted, not chosen now):** enrich the reserved rotation set beyond 24 orthogonal
orientations if a future need genuinely requires inter-part angled mates. That would be a foundation
change to `LatticeOrientation` and is deliberately avoided here — baking the angle into the part keeps the
foundation's discrete rotation space intact.

## Acceptance criteria — G1–G10 walkthrough

| # | Stress case | How the design satisfies it |
|---|---|---|
| **G1** | "build me a star fort" end-to-end; does the loop TERMINATE? | LLM emits rough macro layout + kit-instantiation + joint `Intent`s (E/A) → solver propagates transforms over the discrete lattice (B, SYNC) → chunks resolve off-thread (§7) → async `diagnostics()` → local corrections → re-solve. Termination is **budgeted best-effort (H5)**, NOT a proof: a **residual-MAGNITUDE potential** (sum of `gap_blocks`) + an **oscillation guard** (diagnostics-multiset hash) + a **bounded retry budget**. **Converges or reports — empirically measured by P0, never claimed as guaranteed.** |
| **G2** | The named failure: tower doesn't connect to parapet | `diagnostics()` emits `Issue::UnsatisfiedJoint` carrying the **exact connector pair + `gap_blocks` + intended joint + `fix_hint`** (C/H7) — and if no joint is declared yet, `NearestConnector` (H6) tells the agent which connectors SHOULD mate. Gap diagnosis is exact **connector frame-math** (`ConnectorGap`), not a voxel scan. Agent fixes with ONE local `Intent` (often the `fix_hint` verbatim), not a re-roll. **Precise + local + self-describing.** |
| **G3** | Move/rotate a whole bastion | The bastion is an assembly instance; its joints, connectors, sculpt, and materials are **part-local** (ADR 0003 §3f) so they follow under the `LatticeOrientation` + i64 transform for free; **dependent joints re-solve** (B emits new placement intents) or, if now conflicting, **report drift** via `UnsatisfiedJoint`/`OverConstrained`. A curtain `Span` between two moved bastions re-closes **ONE-SHOT** from the new `|A − B|` (H4), never stretch-after-place. |
| **G4** | Over-constrained / conflicting joints | Two real blind-spots are both checked: **fan-in over-constraint** — solving a node, the solver asserts EVERY incoming joint's implied transform agrees **i64-exact**; mismatch → `Issue::OverConstrained` (H3, catches the not-a-cycle case cycle-detection misses); **volumetric interpenetration** — a **broad-phase AABB overlap pass** in standing `diagnostics()` → `Issue::PartsOverlap` (H3, catches a part forced into occupied space — invisible to a graph-only solver). Plus the bounded budget + oscillation guard (H5) so it **never loops forever**. **"Detects conflicts, never silent garbage" is now actually true.** |
| **G5** | WFC fill within a curtain-wall surface (bands + merlons) | Merlon **RHYTHM is KIT geometry** (`Parapet`/`PatternAlong` param — survives macro intent by construction, H9); **WFC does ONLY material banding within FIXED occupancy** (never solid/void); cross-part band continuity uses a **shared world-Y-keyed deterministic rule, NOT a WFC region across the joint** (kills the corner-seam washout, H9). Bounded part-local region, seeded (deterministic); a contradiction is `Issue::WfcContradiction`, not soup. (D) |
| **G6** | Feedback bandwidth: minimal sufficient query set, cheap? | The `SpatialQuery` set (C) is split by cost: **cheap-synchronous** connector/joint-graph math (`Connectors`/`ConnectorGap`/`NearestConnector`/`JointResidual`/`Connectivity`/`AnchorFrame`) and **async-resolved** occupancy reads (`Bounds`/`Contact`/`Gap`/`Overhang`/`Occupancy`) over the region-scoped path. **`Connectivity` is a zero-residual joint-PATH over the graph, NOT a voxel flood-fill** (H6) — cheap and precise. Data-primary; `render_png` secondary. |
| **G7** | Determinism / replay | Everything flows through the serializable `Intent` door (ADR 0003 §6a); solver math is exact integer lattice arithmetic; WFC is `seed`+fixed-occupancy deterministic. **Same intent script → same building**, undoable/retryable. |
| **G8** | Scale: fort over large XZ; loop stays responsive | Queries/diagnostics split sync vs async (§7); the SYNC solve works on transforms/connectors (cheap metadata), the async VALIDATE uses the **region-scoped** store read (ADR 0003 §4) and never forces a full resolve; broad-phase overlap is AABB-only; WFC fill is a per-chunk producer job under the existing **per-frame budget** (ADR 0003 §7). The loop **does not fight the budget**. |
| **G9** | The macro/micro ARCH seam | Opening spans full blocks (kit connectors on the block lattice, `LatticeOrientation`+i64 placement); curve is sub-block chisel detail (part-local sculpt override, `SubBlockSnap` to the chisel sub-grid). The kit handles cross-scale placement block-relative; **no new payload/transform** (A). The angled-bastion case is resolved the same way — angle baked into the part, connectors block-aligned (F/H8). |
| **G10** | The cheap load-bearing probe FIRST | Yes — and it is now **foundation-free** (H10): a pure `solve(&Scene) -> Vec<TransformResult>` unit test on **hand-built structs**, zero AppCore/Intent/diagnostics plumbing, isolating the one novel spine risk (does discrete integer propagation converge). The full Intent-script probe is gated behind ADR 0003 Phases C/F/H. (P0/P1 below.) |

## Foundation-fit

**Foundation-fit: two required ADR 0003 seam changes (flagged), otherwise consumes existing seams.**
The stack is overwhelmingly additive — additive enum variants on growth surfaces + new downward crates —
**except for two cheap DATA-seam edits back to ADR 0003 §1**, both flagged below. Both are pure
data-shape changes (no solver, no behavior, no resolve-path impact in the foundation), and both are best
made **now while ADR 0003 is still Proposed**, before the joint serialization format is frozen.

### REQUIRED ADR 0003 seam changes (flagged)

- **FLAG 1 — Joint arity: the reserved `Joint` is UNARY and cannot encode an n-ary `Span`.** ADR 0003
  §1 (line 202) reserves `pub struct Joint { pub other: NodeId, pub kind: JointKind, /* params */ }` —
  a **single** `other: NodeId`. A curtain wall bridging **two** fixed bastions is **one joint referencing
  ≥2 other nodes** (the wall + its two end anchors). The unary shape cannot express it. **Required
  change:** widen the joint to n-ary references, e.g. `pub struct Joint { pub refs: Vec<NodeId>, pub
  kind: JointKind, … }`, or split `JointKind` into unary and n-ary families. This is a data-seam edit
  only — ADR 0003 builds no solver (§1 lines 206–210), so nothing in the foundation reads or acts on the
  arity; it is pure serialization shape.

- **FLAG 2 — Stable `JointId`: joints are positionally addressed in `Vec<Joint>` with no stable id.**
  ADR 0003 §1 (line 201) stores joints as `pub joints: Vec<Joint>` on `Node` — addressed **positionally
  by index**, with no stable identity (in contrast to `NodeId`, which §1 mints precisely so references
  survive undo/structural edits, lines 160–176). An `Issue::UnsatisfiedJoint { joint: JointId, … }` or a
  `SpatialQuery::JointResidual(JointId)` cannot **durably reference an individual joint** across undo or
  structural edits if the joint's only identity is its `Vec` index (which shifts when a joint is
  inserted/removed). **Required change:** mint a stable `JointId` (like `NodeId`, from a document-owned
  counter) and key joints by it. Again a pure data-seam edit — the foundation does not consume `JointId`
  behaviorally; it just needs to exist and serialize so this ADR's `Issue`/`SpatialQuery` surface can
  name a joint stably.

Both are **cheap DATA-seam changes with no solver/behavior impact on the foundation**, and making them
now (while 0003 is Proposed) avoids a schema cascade later. Everything else in this stack consumes
existing seams with no foundation change.

**ADR 0003 seams consumed (no change required):**

- **`Intent` door** (§6a) — kit expansion, joint creation, solver output, LLM/VLM all emit `Intent`s
  through `apply_intent`. New `Intent` *variants* are added (kit-instantiate, add-joint, set-param,
  set-transform) — additive enum growth, the foundation's intended extension mode.
- **`query(SpatialQuery) -> Answer`** (§6, lines 557/565) — the C taxonomy adds *variants* to the
  existing enum (the §6 surface is designated extensible: "contact / gap / overhang / connectivity /
  bounds", line 566).
- **`diagnostics() -> Vec<Issue>`** (§6, lines 558/568) — the C `Issue` taxonomy adds *variants*; ADR
  0003 already declared this surface carries "unsatisfied joints" and orphaned overrides (line 570).
- **`scene` joint graph** (§1) — `Node.joints` / `Joint` / `JointKind` are consumed as the solver's
  input; ADR 0003 reserved the DATA seam and explicitly left the SOLVER to "their own future design doc"
  (lines 206–210) — this is that doc. (Consumed *as amended by FLAG 1 + FLAG 2*.)
- **Part/assembly model** (§1, §3f) — kit modules generate `AssemblyDef`s; placement is the existing
  `LatticeOrientation` + i64 translation; sculpt overrides are the existing part-local layer.
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
- **P3 — conflict & convergence hardening (G4):** add `Concentric`/`Flush`/n-ary `Span`, cyclic curtain
  rings (bounded fixpoint), the **fan-in over-constraint** check (i64-exact agreement of all incoming
  joints), the **broad-phase AABB overlap** pass, and `OverConstrained`/`SolveDidNotConverge`/
  `PartsOverlap`/`ConnectorOffSurface`. Stress with a deliberately over-constrained gatehouse-vs-tower
  AND a fan-in node mated to two anchors.
- **P4 — the architectural kit:** author the first modules (wall segment, tower ring, parapet course with
  the `Parapet`/`PatternAlong` merlon rhythm, gatehouse, **angled bastion wedge with the angle baked in +
  block-aligned connectors**, ARCH with cross-scale springline connectors + sub-block curve). **Reuse:**
  parametric-prototype → module-extraction pattern (kit modules double as P1's WFC tiles).
- **P5 — the full agent loop:** wire the LLM to the documented kit/joint/query/issue vocabulary (RAG over
  `params_schema()` + the taxonomies), with the LLM owning rough macro layout; run G1 "star fort"
  end-to-end with the bounded retry budget + oscillation guard. **Reuse:** generate → validate → repair
  pattern; the ADR 0003 Phase K MCP/socket as the agent transport.
- **P6 — the optional VLM critic** (outer loop, `render_png`-conditioned) — last, optional, a flavor knob.

**Reuse-vs-build summary:** *reuse* — the entire ADR 0003 control surface, joint DATA seam (as amended by
FLAG 1/2), producer registry, categorical cell, async/region-scoped read path, an off-the-shelf WFC
engine, and the documented-library-LLM + generate-validate-repair patterns. *Build* — the discrete-lattice
joint solver with fan-in + broad-phase overlap checks (B), the kit modules (A), the WFC fixed-occupancy +
world-Y band key + material tileset (D), and the additive query/issue/intent/joint enum variants.

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
- Two required (cheap) ADR 0003 seam edits (joint arity + `JointId`) must be reconciled into the
  foundation before the solver lands — best done now while 0003 is Proposed.
- A discrete constraint solver with **fan-in over-constraint detection, broad-phase volumetric overlap,
  and cyclic-graph convergence** is real engineering (P0/P3), even bounded to the lattice — and its
  convergence is empirical, not proven.
- Authoring the kit modules (incl. angle-baked bastions) and the WFC material tilesets/adjacency is
  ongoing content work, not just code.
- New enum taxonomies (`SpatialQuery`/`Issue`/`JointKind`) plus their query implementations split across
  the cheap-synchronous and async-resolved read paths, plus connector-validity sampling.
- The LLM grounding (RAG over schemas) and the optional VLM critic are external-model integration with
  their own cost/latency, gated behind the ADR 0003 Phase K transport.

**Deferred**
- Free (non-lattice) joint placement / continuous IK — the solver stays discrete-lattice exact;
  arbitrary-angle *inter-part* mates are out (the angled bastion is handled by baking the angle into the
  part, F/H8). Enriching the `LatticeOrientation` set beyond 24 is the noted alternative if a future need
  forces it.
- Learned adjacency (auto-extracting WFC rules from example builds) — start with authored tilesets.
- Multi-objective outer optimization beyond the single VLM critic slot.
- Cross-part WFC (WFC spanning a whole assembly, or one WFC region across a joint) — explicitly out;
  WFC stays per-surface/per-part, with cross-part band continuity carried by the shared world-Y key, not
  by propagation across a boundary (H9).

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
- **Inter-part angled (45°) mates via an enriched rotation set.** Noted as the alternative to H8 but not
  chosen: it would change the foundation's `LatticeOrientation`. Instead the angle is baked into the part
  (block-aligned connectors, axis-aligned joints), keeping the discrete rotation space intact.
- **A new foundation field for anchors/sub-block snap.** Rejected as unnecessary: connectors live in the
  part-local frame and the ARCH sub-block curve reuses the existing part-local sculpt override layer. (The
  two genuinely required foundation edits are the joint *arity* and *`JointId`* DATA-seam changes, flagged
  in Foundation-fit — not a new anchor/snap field.)
