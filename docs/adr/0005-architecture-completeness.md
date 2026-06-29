# ADR 0005 — Architecture-completeness additions: pattern producer, space/nav graphs, construction systems, terrain & site, placeholders, decay, diagnostics hardening, and the analysis-perf budget

- **Status:** Proposed
- **Date:** 2026-06-27
- **Layer:** FEATURES / SUBSYSTEMS-ON-TOP of [ADR 0003](0003-foundation-rework.md) (the foundation) and
  [ADR 0004](0004-agent-authoring-stack.md) (the agent-authoring stack). **Not built until the ADR 0003
  foundation phases are underway** — concretely the same gate ADR 0004 sits behind (the Phase C `Intent`
  door + Phase F producer-registry/rotation + Phase H `query`/`diagnostics` surface + the `scene`
  joint/datum DATA seams must exist first). This ADR is the set of additions those seams were reserved
  *for*. It consumes ADR 0003 seams **F1–F6** and ADR 0004 seams and drops onto them additively. It is
  designed to require **NO new foundation change**: the [architecture gap sweep](../design/architecture-gap-sweep.md)
  found **no model-breaker** — the producer / voxel / three-tier spine holds — and every gap below is an
  **addition** that consumes an existing seam. One genuinely new foundation need surfaced — a
  resolved-occupancy *revision* read seam for incremental cache invalidation (FLAG 1) — and it has been
  **reconciled into ADR 0003 §4/§7** (a small read-side API addition, not a model change). After that
  reconciliation, Foundation-fit shows **zero outstanding foundation flags**.
- **Cross-ref (added 2026-06-29):** the ~6 analysis subsystems here all read the **CPU resolved-occupancy
  seam** ([§J](#j-the-perf-budget-discipline-load-bearing)), which
  [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) pins as CPU-authoritative precisely because every
  one of them is a CPU consumer of authoritative occupancy (export/analysis/persistence) that a
  GPU-authoritative volume would break. ADR 0006 names this ADR's analysis-perf budget (§J) as the real
  agent-side bottleneck.

## Context

ADR 0003 locked the foundation (three-tier authoring: parametric PRODUCERS / relational JOINTS /
corrective OVERRIDE-PATCH layers; part/assembly; categorical per-voxel `block_id` + typed `BlockAttrs`;
chunked sparse absolute-i64 streaming store with rebase-at-consume and per-chunk incremental re-mesh; the
F1–F6 seams). ADR 0004 built the agent-authoring stack on top (the joint solver, the parametric kit, the
WFC material-fill producer, and the data-primary perceive → diagnose → correct loop over
`query(SpatialQuery)` + `diagnostics() -> Vec<Issue>` + multi-view `render_png`).

> **[STATUS 2026-06-29]** "Per-chunk incremental re-mesh" above is now LIVE (#40, commit `9ff63c3`):
> the cuboid renderer re-meshes only the dirty chunks (apron-dilated via `cuboid_incremental_plan`),
> wholesale only on a floating-origin shift / density change (see ADR 0002 E5 Step 4 / ADR 0003 §4). The
> §J caching this ADR builds on the per-chunk `ChunkRevision` was always unaffected by this — it depends
> on the resolve cache's per-chunk invalidation, which was already live, not on the GPU mesh rebuild.

A 9-lens **architecture gap sweep** ([`docs/design/architecture-gap-sweep.md`](../design/architecture-gap-sweep.md))
was then run across the whole stack. Its headline finding: **the spine is sound — there is no
model-breaker.** The producer/voxel/three-tier architecture absorbs everything the sweep threw at it. But
it surfaced **22 gaps**, all of which are *additions*: features and analysis subsystems that **consume the
existing seams** rather than reshape them. This ADR designs those 22 gaps as **one coherent additions
stack**. The discipline of the design is: every addition names the 0003 F-seam or 0004 seam it consumes,
and the only thing that may go back to 0003 is flagged loudly (one item does — see Foundation-fit).

### Owner scope rulings (decided with the product owner — honored, not relitigated)

- **The planner is STATIC (F6).** A mechanism's *function* is a **placeholder annotation for export
  fidelity**, never live behavior. Doors that open, drawbridges, portcullises, lifts, windmill yaw,
  furnaces/gears are **proxy entities + a `BlockEntity::Substitution` payload** (F2) substituted on
  export. **Live kinematics, tick simulation, state machines, and signal networks are OUT of scope** — a
  categorically different subsystem, not bolted into the three-tier static model.
- **Terrain is WRITABLE (F3).** Terrain is a first-class mutable layer; a producer **may** sample
  `GroundHeightAt` to tie into live grade. This ADR designs the cut/fill/site PRODUCERS that F3 reserved
  the coupling for (ADR 0004 left terrain *import* as a flagged research spike; that spike is unchanged
  and still upstream of the site producers here).
- **Structural validity is OUT.** This ADR does **not** add statics, thrust-line analysis, or a
  structural solver. It does **one** structural-adjacent thing: it **fixes the misleading
  `Unsupported`/overhang stub** (which false-fires on every arch/vault/dome/corbel/cantilever) by adding a
  part-level structural-ROLE flag + load-path-aware support tracing. That is a *diagnostics correctness*
  fix, not a statics subsystem.
- **The PERF BUDGET is a load-bearing constraint, not a footnote.** The ~6 new analysis subsystems below
  (space graph, traversability, terrain-coupling, decay, topological naming, plus pattern voxelization)
  all read **resolved occupancy** at the foundation's anisotropic **>10k-XZ** extent. They **cannot all be
  live.** The design must specify the always-on vs computed-on-query split, the **LOCAL (region-scoped) vs
  GLOBAL-TOPOLOGY (seed-and-grow + budget cap)** query split, caching, and incremental invalidation — or
  it does not fit the streaming/worker budget at all. This is treated as a first-class section (§J), not
  an afterthought.

### The convergent additions thesis

The sweep's deepest structural insight: **the foundation gives us a relationship graph over SOLIDS
(joints, §1) and a derived occupancy field — but almost every missing subsystem is really a derived graph
over something the foundation does not yet name.** Specifically:

- a graph over **VOIDS** (the space graph — rooms, portals, enclosed volume);
- a graph over **walkable surface** (the nav graph — derived from occupancy under an agent profile);
- a graph over **derived geometry features** (topological naming — `roof.ridge`, `wall.top_edge`,
  `arch[k].springline` surviving regeneration).

These three derived-graph subsystems are the **shared substrate** the rest of the additions stand on
(vaults host on space cells; egress queries run on the nav graph; ornament rides named edges). So the
build plan promotes the **space graph FIRST** and treats topological naming and the nav graph as the next
substrate layer — exactly the "incrementality / probe-first / build-the-substrate-first" discipline
(GD6). Everything else (construction systems, terrain producers, placeholders, decay) is generative
breadth that consumes the substrate.

## Decision

One coherent additions stack across all ten areas. Each subsection states the choice, the 0003 F-seam /
0004 seam it consumes, and how it resolves the red-team's BLOCKER/major findings (called out as **[RT-…]**
where a red-team finding is being closed). The recurring rule from 0003 §3h — *open extensible set with
uniform behavior → trait; closed small serialized/matched set → enum* — governs every dispatch choice
below, and the recurring rule from ADR 0004 — *the SOLVE/transform half is synchronous and reads no
voxels; anything reading resolved voxels is the async, region-scoped VALIDATE half* — governs every
analysis subsystem's cost placement.

### A. Pattern / revolve producer — the highest-leverage primitive

The single highest-leverage addition. A **`PatternProducer` family** that voxelizes **N copies of a
sub-geometry at arbitrary angles FROM THE FIELD** (the lossless parametric tier — ADR 0004 F1: angle is a
producer parameter, SDF-exact and re-voxelized at resolve, never a transform limit) **while exposing
stable per-member connectors and a member index**, so joints and diagnostics can address
`arch[k].springline`. It is a `VoxelProducer` (0003 §3d) — one registrant among many, with
`resolve_into(chunk_box, …)` and a bounded `world_aabb_blocks()` (the conservative cover of all members),
so it rides the chunk-windowed incremental path for free.

```rust
// new crate `pattern`, depends on `scene` + `core_geom` (downward only); registers as a VoxelProducer.
pub struct PatternProducer {
    pub source: PatternSource,          // a sub-geometry: an SDF producer, OR a referenced sub-AssemblyDef
                                        // (so a member can be a COMPOSITE sub-assembly, not just a shape)
    pub layout: PatternLayout,          // Linear | Radial | Path (closed serialized enum)
    pub count: u32,                     // edit-by-count: change N and members re-derive
    pub per_member: MemberTable,        // sparse per-index param table + override / SUPPRESSION (see below)
}
pub enum PatternLayout {
    Linear { step: [i64; 3] },                              // axis-aligned or skewed lattice run
    Radial { center: [i64; 3], axis: Axis, sweep_deg: f32, // ARBITRARY angle — 360/N from the FIELD
             radius: i64, pitch_y: i64 },                  // pitch_y != 0 ⇒ HELICAL (spiral stair / screw)
    Path   { path: Vec<[i64; 3]>, align: PathAlign },      // members distributed along a 3D path
}
```

**Member identity is a stable, durable index (the load-bearing property — [RT: "patterns lose identity on
re-count"]).** Each member has a `MemberId(u32)` minted **per layout slot by a deterministic rule keyed to
the layout, not by array position**, so per-member data survives a count/pitch change wherever the slot
still exists. This is what makes `arch[k].springline`, per-index capital variation, and a *suppressed*
member that stays suppressed across a re-count all expressible.

- **Identity semantics under a count change are ORDINAL-SLOT (explicit choice).** Member `k` is **the
  k-th slot**; its *physical angle re-derives* at `360/N` when the count changes. So per-member data
  keyed to slot `k` stays on the k-th member across a re-count (a `Radial` colonnade's "member 3's
  capital variation" stays on member 3, even as every member's angle shifts) — the durable thing is the
  **ordinal slot**, not the physical angle. The alternative (**angular-slot identity**, where a member
  keeps its *angle* and a count change re-buckets) is **not** chosen; where a count drop leaves a keyed
  member with no surviving ordinal slot, it surfaces as an **override-orphan via `Issue::PatternBroken`**
  (the same PRESERVE-AND-FLAG posture as 0003 §3g) rather than silently re-binding. This pins the
  previously-ambiguous "survives a re-count" to a single rule.

- **Per-member override / suppression surviving count change.** `MemberTable` is a sparse
  `HashMap<MemberId, MemberEntry>` where an entry is `Override(KitParams) | Suppress | HostedFeatures(…)`.
  Editing the count adds/removes only the *unkeyed* slots; explicitly-keyed members keep their entry. A
  suppressed member (rotunda colonnade with one column removed, P5) is a `Suppress` entry — it survives a
  re-count because it is keyed, not positional.
- **Stable per-member CONNECTORS + member index addressing.** The pattern re-exposes the source's
  connectors **namespaced by member** (`arch[3].springline`, `column[k].capital`). These are ordinary
  ADR 0004 `Connector` frames in the part-local lattice frame, so the joint solver mates them and
  `SpatialQuery::ConnectorGap`/`AnchorFrame` address them exactly — the pattern is fully first-class to
  the relational tier and the diagnostics loop.
- **HostedOn features ride the array.** A feature hosted on the source (a finial on a column capital) is
  expanded per surviving member via the same member-index namespacing; suppression removes the member's
  hosted features with it.
- **Composite source.** `PatternSource::SubAssembly(DefId)` lets a member be a whole sub-assembly (an arch
  *with* its imposts and keystone), re-voxelized per slot — reusing the existing def/instance machinery
  rather than inventing a second composition path.
- **Edits are parametric and lossless.** Edit-by-count, per-member override, and arbitrary radial/helical
  angle all re-voxelize from the field (0004 F1), so there is no lossy resample (the deferred sculpt-resample
  caveat, 0003 §3f case 3, does not apply to a parametric pattern).
- **`Issue::PatternBroken`** is added to the 0004 `Issue` taxonomy: emitted when a keyed `MemberId` no
  longer maps to any layout slot (e.g. count dropped below an explicitly-overridden index), so a dangling
  per-member override/suppression is surfaced and pruneable rather than silently lost (the same
  PRESERVE-AND-FLAG posture as 0003 §3g orphaned overrides).

This resolves radial/curved/helical repetition of **reusable** parts (arcade columns, rose-window
tracery, spiral stairs, rotunda colonnades) as one primitive, addressable by the relational and
diagnostics tiers.

### B. Space graph — the shared substrate (promoted FIRST)

Joints relate **solids**; nothing relates **voids**. The space graph segments **resolved VOID** into
**named space cells + portals** and is the substrate for traversability, vaults, and site reasoning. It is
a derived graph computed on-demand over a region (never a stored authoritative structure — §J), cached and
incrementally invalidated.

```rust
// new crate `space`, depends on `store` (reads resolved occupancy) + `scene`.
pub struct SpaceGraph {
    pub cells:   SlotMap<SpaceCellId, SpaceCell>,   // a connected void region (a "room")
    pub portals: SlotMap<PortalId, Portal>,         // an opening between two cells (door/arch/window gap)
}
pub struct SpaceCell  { pub volume_blocks: u64, pub aabb: Aabb64, pub enclosure: Enclosure }
pub enum   Enclosure  { Bounded, Leaks, Indeterminate }  // Indeterminate = the cell touches a grow-frontier (§B global-topology rule)
pub struct Portal     { pub between: [SpaceCellId; 2], pub aperture: Aabb64 }
```

**LOCAL vs GLOBAL-TOPOLOGY queries — the load-bearing distinction (region-scoping is NOT uniform).**
The space graph answers two categorically different kinds of question, and **only one of them may be
naively region-clipped**:

- **LOCAL queries** — the volume of a *known* cell, adjacency, a `MaterialTally` over a window — depend
  only on what is inside the region. They are genuinely region-clipped and cheap (§J region-scoped
  case).
- **GLOBAL-TOPOLOGY queries** — *is this cell enclosed?* (`NotEnclosed`), *how many egress portals
  reach outside?* (`EgressCount`), *is A reachable from B?* (`PortalConnectivity`/nav `Reachable`),
  *is this basin watertight?* (`BasinNotWatertight`) — are **global properties of the connected
  void/surface/basin**. A residency-clipped flood-fill gives a **WRONG answer at the clip boundary**: a
  void that simply continues past the region window reads as `NotEnclosed`/leaking, and a path that
  exits and re-enters the window reads as `Unreachable`, when in truth the cell is enclosed / the path
  exists. These queries **cannot be naively clipped to a fixed region window.**

**Global-topology derivation — SEED-AND-GROW across residency boundaries (NOT a fixed window).** A
global-topology query seeds at the known cell / surface / basin and **grows the flood-fill / path
along the connected void (or walkable surface, or basin), loading chunks on demand along the frontier**
— bounded by the **connected extent itself** plus an explicit **budget cap**, never by a fixed region
`Aabb64`. It costs more than a local query (the working set grows along the frontier, not a pre-sized
window) and so runs on the **async VALIDATE half over several frames** (§J's seed-and-grow case). When
the grow hits the **budget cap before the connected extent closes**, the query returns
**indeterminate** (`Issue::AnalysisBudgetExceeded`) — it reports "could not determine," **never a false
negative**. A cell whose flood reaches the live grow-frontier is marked `Enclosure::Indeterminate`, so
it is **never falsely reported enclosed or leaking**. *(Optional alternative for cheap global topology:
a coarse, always-resident low-res occupancy summary — a downsampled enclosure/connectivity digest — can
answer is-enclosed approximately without a full grow; the seed-and-grow path is the exact fallback.)*

- **Derivation:** a flood-fill / connected-components pass over resolved void occupancy (the
  foundation's resolved-grid read seam, 0003 §4). **Local-query** passes are region-clipped; the
  **portal-graph topology** behind a global-topology query is grown seed-and-grow across residency
  boundaries per the rule above. Portals are detected where a thin void aperture joins two cells.
  Deterministic, seed-free.
- **Named cells.** Cells get **durable names** the way derived geometry does (§D's naming substrate): a
  cell is identified by a stable signature (its seed-voxel address class + a content fingerprint) so
  "the nave" keeps its name across an edit that does not destroy it. Naming voids reuses **exactly** the
  topological-naming machinery (§D) rather than a parallel scheme.
- **SpatialQuery / diagnostics over voids** (additive variants on the 0004 enums — the canonical
  "consume the reserved control surface" move):
  - `SpatialQuery::EnclosedVolume { cell }` (LOCAL: the cell's volume), `SpaceAdjacency { cell }`
    (LOCAL). **Global-topology** (seed-and-grow): `PortalConnectivity { from, to }` (a portal-path over
    the space graph, the void analogue of 0004's joint-graph `Connectivity`) and `EgressCount { cell }`
    (number of portals reaching an outside cell) — both follow the connected void across residency
    boundaries and may return *indeterminate* under the budget cap.
  - `Issue::DeadEndSpace { cell }`, `Issue::UnreachableSpace { cell }`, `Issue::NotEnclosed { cell }`
    (a "room" that leaks to outside when it should not). **These global-topology issues fire only on a
    cell whose enclosure is `Bounded`/`Leaks` — never on `Indeterminate`** (a cell touching the
    grow-frontier is reported indeterminate, never falsely `NotEnclosed`/`UnreachableSpace`).
  - `Issue::AnalysisBudgetExceeded { query }` — a global-topology grow hit its budget cap before the
    connected extent closed; the answer is **indeterminate, never a false negative** (§J).
- **Substrate role.** Vaults host on a space cell (§E); traversability (§C) and egress run portal
  connectivity; site datums can be cell-relative. Building this first is what makes the rest cheap.

### C. Traversability / nav subsystem

Derives a **nav graph from RESOLVED occupancy** under an `AgentProfile`. **This deliberately overturns
ADR 0004's implicit "connectivity = the joint graph"** — joint connectivity answers "is the fort one
*structural* piece?"; it cannot answer "can you *walk* from the gate to the keep top floor?" That is a
**derived nav/space graph** question, named here explicitly as a new derived graph (the red-team
[RT: "connectivity conflates structural and traversable"] is closed by separating the two graphs).

```rust
// new crate `nav`, depends on `space` + `store`.
pub struct AgentProfile { pub height: u32, pub step_up: u32, pub stride: u32, pub width: u32 } // in voxels
pub struct NavGraph { pub nodes: Vec<NavNode>, pub edges: Vec<NavEdge> }  // walkable cells + step/stride edges
```

- **Walkable derivation:** a voxel column is a nav node if it is **empty + solid-directly-below +
  headroom ≥ profile.height**; an edge exists between adjacent nodes within **step-up / stride / width
  tolerance** under the profile. Derived region-scoped, cached, invalidated per §J.
- **Queries / diagnostics** (additive variants):
  - `SpatialQuery::Reachable { from: [i64;3], to: [i64;3], profile }` (a nav-path, the literal P6 query)
    — a **GLOBAL-TOPOLOGY query** (§B): A→B reachability is a property of the connected walkable
    surface, so the path is grown **seed-and-grow along the walkable frontier, loading chunks on
    demand**, NOT clipped to a fixed region window (a route that exits and re-enters a window would
    otherwise read as `Unreachable`). It is bounded by the connected extent plus a budget cap and runs
    on the async VALIDATE half; on cap it returns **indeterminate (`AnalysisBudgetExceeded`), never a
    false `Unreachable`**.
  - `Issue::Unreachable { from, to }` (fires only when the seed-and-grow closed the connected surface
    without reaching the target — never on a budget-truncated grow); gait diagnostics
    `Issue::StairTooSteep { node, rise, run }`,
    `Issue::HeadroomBelowMin { at, have, need }`, `Issue::RunTooNarrow { at, have, need }`.
  - `Issue::UnguardedEdge { at }` — a walkable node bordering a fall with no parapet/guard (the safety
    diagnostic, derived from nav + occupancy).
- **Coupling to the space graph:** nav edges through a portal are how "up the spiral stair, through the
  floor opening, to the keep top floor" is answered — the stair (an §A helical pattern) generates nav
  edges, the floor opening is a portal (§B), and `Reachable` traverses both (P6).
- **Cost placement:** nav is **never always-on** — it is computed on a `Reachable`/gait query over the
  query's region, the most expensive analysis subsystem and the strongest reason §J exists.

### D. Topological naming substrate — named DERIVED geometry surviving regeneration

The substrate for everything that "follows an edge" or "hosts on a derived feature." Producers regenerate
from parameters, so a feature pinned to a raw voxel address breaks on every edit. The naming substrate
gives **named derived geometry** (`roof.ridge`, `wall.top_edge`, `skeleton.face[k]`, `arch[k].springline`)
a **durable identity that survives regeneration**, the way `NodeId` gives nodes durable identity.

```rust
// in `scene` as a DERIVED, non-authoritative index (recomputed, never the source of truth):
pub struct TopoName { pub of: NodeId, pub selector: TopoSelector }   // stable, regen-surviving handle
pub enum TopoSelector {                                              // a deterministic rule, not an address
    Ridge, TopEdge, SkeletonFace(u32), Springline(MemberId), EdgeLoop(EdgeKey), /* … */
}
```

- **Stability rule:** a `TopoName` resolves through a **deterministic selector against the producer's
  current output**, not a stored voxel address — so it re-resolves correctly after the producer regenerates
  (the same principle that makes §A member ids and §B cell names durable; one mechanism, three consumers).
- **Queries:** `SpatialQuery::EdgeCurveOf(TopoName)` (returns a 1D edge path — a poly-line of lattice
  points) and `SurfaceFrameAlong { edge: TopoName, t: f32 }` (an oriented frame at parameter *t* along
  the edge, for sweeping ornament). These feed **1D edge-path connectors** — a `ConnectorKind::EdgePath`
  added to the 0004 `Connector` so a dressing can mate to an edge, not just a face.
- **In-part feature tree — a def-level side-registry, NOT a `Layer` enum change.** A part def gains an
  **ordered feature list** (a small DAG) so a later feature can consume an earlier feature's topology (a
  string course consumes `wall.top_edge`; a cornice consumes `roof.eave`). The feature-output naming is a
  **def-level side-registry** — a `name → TopoSelector` map per def — **populated during the existing
  `Vec<Layer>` fold (0003 §3b) and resolved against producer output**: as each layer resolves it may
  *publish* `TopoName`s into the registry, and a later layer *references* them by name. This adds **NO
  variant and NO new data to the frozen `Layer` enum** — 0003 §3h keeps `Layer` as a **closed, two-arm
  fold (`Producer` | `Sculpt`) deliberately kept as an enum** (designated-CLOSED, unlike the
  designated-extensible `Issue`/`SpatialQuery`). The feature tree is therefore **consume-only against
  `Layer`**: a side-registry beside the fold, not an arm or field on it — consistent with §3h, no
  foundation change. No new tier.
- **Sub-block miter / return solver.** Where an edge-following ornament turns a corner (a raking cornice
  meeting a gable, P8), a **miter/return solver** computes the sub-block joint geometry at the corner from
  the two incident edge frames. It is a kit-side helper operating in the chisel sub-grid (the 0004
  `SubBlockSnap` seam), emitting part-local sculpt overrides — **no new foundation field** (same posture
  as the 0004 ARCH cross-scale seam).

This is what hosted features, dressings, and ornament-following-edges quietly assumed and the foundation
did not name. **[RT: "dressings host on regen-fragile addresses"]** is closed: they host on `TopoName`s.

### E. Construction systems

Generative breadth, all built as producers / kit composites / override layers consuming the existing
tiers. Each is a producer family, not a foundation change.

- **Layered cross-section walls.** The 0004 path-swept `Wall` gains a **laminated profile**: an ordered
  `Vec<Lamella { material: BlockId, thickness: u32 }>` swept along the path → multi-skin core + cavity +
  weather-skin. It is the same SDF sweep producer with a richer cross-section; output is categorical
  `block_id` per lamella (0003 §3a). An **optional per-FACE material channel** is a `BlockAttrs.variant`
  selection keyed by which swept face a voxel belongs to.
- **Coursing / bond producer.** A producer that tessellates header/stretcher courses + mortar across a
  wall surface, parameterized by bond pattern and course height. Output is categorical `block_id`/variant
  — **this is occupancy+material geometry the KIT fixes, not WFC** (consistent with 0004 H9: rhythm is kit,
  material-within-fixed-occupancy is WFC).
- **HostedOn DRESSINGS.** Quoins / jambs / sills / voussoirs are **hosted features** (§D feature tree) that
  **re-materialize with openings**: a jamb hosts on a window opening's edge (`TopoName`), so when the
  opening moves/resizes the jamb redrapes. Penetrations auto-recut via the existing 0004 HostedOn rule.
- **Frame-and-infill.** A **structural-MEMBER producer family** (beam / post / brace) = path-sweeps along
  a member network (an §A `Path` pattern of members or an explicit member graph) + an **infill-panel
  producer** that fills bays between members. A **framed-wall composite** (a kit composite generator, 0004
  A) instantiates members + panels sharing **one bay rhythm** (a shared §I subdivision). This is the
  half-timber longhouse (P2): the void rhythm *between* members is authored by the member network +
  panel producer, which material-only WFC cannot do.
- **Vaults.** A **vault producer family** (barrel / groin / rib / fan / pendentive) **hosted on a BAY or
  SPACE CELL** (§B), **not** on the outer footprint — the vault springs from the cell it covers. Each is a
  parametric SDF/sweep producer; rib vaults expose rib `TopoName`s (§D) for ornament. **[RT: "vaults can't
  host on the right thing"]** is closed by hosting on the space graph (§B), which is exactly why §B is
  promoted first.
- **Recursive shape-grammar facade subdivision.** A CGA-style **split grammar** (split → floors → bays →
  window + pier), composed onto faces, with **bounded recursion** (a depth cap, surfaced as
  `Issue::GrammarDepthExceeded` rather than runaway). Built on a **shared integer-subdivision utility**
  (§I) so facade splits and frame bays use one rule.
- **Coursing / merlon PHASE registration via a shared WORLD-SPACE phase field across ALL axes** (not
  Y-only). A single deterministic `phase(world_xyz)` function (the generalization of 0004 H9's world-Y
  band key from one axis to three) keys every coursing/merlon/banding producer, so courses line up across
  a wall junction **and** wrap continuously around a round tower (P12). **[RT: "world-Y key only handles
  vertical bands; horizontal coursing around a tower still seams"]** is closed by promoting the key to a
  full 3D world-space phase field.

### F. Terrain & site (consuming F3)

The cut/fill/site producers F3 reserved the coupling for. All sample `GroundHeightAt` (the 0004 query) to
tie into live grade — the only sanctioned producer↔terrain coupling (F3), still *no* general
producer↔producer coupling.

- **`GradeTo` / `Berm` / `Terrace` / `Excavate` producers.** Each is a `VoxelProducer` that **mutates the
  writable terrain layer** (F3) by sampling `GroundHeightAt` along its footprint and blending/tying-in
  (cut where terrain is above target, fill where below). `Terrace` produces stepped levels; `Excavate`
  removes terrain within a volume; `Berm`/`GradeTo` add/grade to a target surface. Walls drape over the
  result via the existing 0004 drape-then-sweep path; foundations tie into grade by the same coupling
  (P3).
- **Terrain-relative datums** are already an F4 seam (`DatumAnchor` may be terrain-relative); the site
  producers and terrace levels anchor to them, so "raise the terrace → everything hosted on it follows"
  works through the existing datum fan-out.
- **WATER as a filled volume.** Water is a **flood-fill-to-level-Y within a bounded basin** producer:
  given a basin (a bounded void, an §B space cell qualifies) and a target Y, fill void below Y with a
  water `block_id`. Recomputed when the basin changes (incremental, §J). **`Issue::BasinNotWatertight`
  is a GLOBAL-TOPOLOGY query (§B), not a region-clipped one:** watertightness is a property of the whole
  connected basin, so the leak-check **grows the below-water void seed-and-grow across residency
  boundaries** (a portal/aperture below water level reaching an *outside* cell), bounded by the
  connected basin extent plus a budget cap — a naive region clip would falsely report a basin
  continuing past the window as leaking (or falsely watertight). On cap it returns **indeterminate
  (`AnalysisBudgetExceeded`), never a false `BasinNotWatertight`**. This is the moat (P3).
- **DRAINAGE gradient diagnostic.** `Issue::DrainageNoOutfall { region }` / a monotonic-descent check:
  trace surface descent from a region to an outfall; flag if no monotonic path to drainage exists. A
  cheap surface-graph query over the terrain layer (reuses the nav-derivation machinery on the terrain
  surface).

### G. Placeholders & export (consuming F6 / F2)

F6/F2 reserved the `BlockEntity::Substitution` schema; this ADR builds the consumers.

- **Placeholder PROXY producer library.** A library of recognizable static stand-ins per interactable
  (door / drawbridge / portcullis / lift / windmill / furnace / gear) — each a small `VoxelProducer` that
  emits proxy geometry so a human reads "a drawbridge goes here" (the F6 "feel"), paired with a
  `BlockEntity::Substitution` payload (F2) carrying the real `target` + `attrs` + `orientation` + `params`.
- **Distinct "this is a placeholder, not final" RENDER treatment.** A **per-draw proxy flag** (the same
  per-draw-uniform mechanism that 0003 §3c used to evict `GRID_OVERLAY_BIT` from the data field — reuse,
  not a new payload field) tints/hatches proxy geometry so proxies read as proxies in the viewport.
- **Substitute-on-export.** On export the proxy occupancy is **replaced** by its `Substitution.target` +
  attrs + orientation + params (orientation composing through the instance transform, F1/F2), through the
  existing 0003 world-origin export contract.
- **Diagnostics:** `Issue::PlaceholderUnsubstituted { at }` (a proxy with no Substitution payload) and
  `Issue::PlaceholderUnmapped { at, target }` (a Substitution whose `target` has no known VS mapping) — so
  a forgotten or unmappable placeholder is surfaced before export rather than silently shipping proxy
  geometry (P7).
- **BOM / material tally.** `SpatialQuery::MaterialTally { region }` returns a categorical
  `block_id → count` histogram over a region (region-scoped occupancy read, §J).
- **World-grid registration + VS-native schematic exporter.** The exporter (a consumer feature; the 0003
  world-origin export contract is the seam) writes a VS-native schematic carrying **micro-block + block-state
  + entity** (the F2 attrs + side-table), with the world-grid anchor mapping build-anchor block coord → VS
  world coord (0003 §3a-bis). `.vox` export remains the existing path; the VS schematic exporter is the new
  full-fidelity path.

### H. Diagnostics & loop hardening (incl. the Unsupported-stub fix)

- **Fix the misleading `Unsupported`/overhang stub — the structural-correctness fix (NO statics).** The
  0004 `Issue::Unsupported` ("no block directly below") **false-fires on every arch, vault, dome, corbel,
  cantilever, and ornament-overhang** — it is the diagnostic most likely to make the agent loop thrash
  against legitimate geometry. The fix has two parts, neither of which is a statics subsystem:
  1. **A structural-ROLE annotation — a 0005-OWNED `DefId`-keyed side table, NOT a foundation field.**
     `enum StructuralRole { LoadBearing, Decorative, Veneer, Tie }` lives in a
     `HashMap<DefId, StructuralRole>` **side table owned by the diagnostics subsystem**, keyed by the
     def's stable `DefId` (0003 §1). It is deliberately **NOT a field on the foundation's `AssemblyDef`**
     (that would be a 0003 schema change in the same category as F5) — the role is a *diagnostics
     annotation*, not part-defining data, so it rides a side table the diagnostics layer owns and
     **requires no foundation change**. `Decorative`/`Veneer` parts are **exempt** from support tracing
     (an ornament overhang is supposed to overhang); `Tie` members are recognized as tension elements.
     A def with no entry defaults to `LoadBearing`.
  2. **Load-path-aware support tracing** (NOT statics, and BOUNDED): support is traced along the
     **load-bearing structure** — a voxel is "supported" if a path of `LoadBearing` occupancy connects
     it to ground **within a bounded search radius**, **including lateral/arch paths** (a voussoir is
     supported by its neighbors thrusting to the springline, not by a block directly below). This is a
     **reachability-to-support over load-bearing solid cells inside a bounded search radius** — a
     **buildability / intent check, NOT engineering**: it asks "did the author leave this load-bearing
     mass floating?", not "will it stand under load." The bound is explicit so the trace cannot become a
     slow unbounded flood (which would also defang the diagnostic). This is a connectivity trace over
     load-bearing occupancy, **not** a thrust/force computation. `Issue::Unsupported` only fires when no
     load-path to ground exists within the radius for a `LoadBearing` voxel. **[RT: "Unsupported is
     useless on any real building"]** is closed: a cathedral's arches/vaults/buttresses/corbels no
     longer false-fire (P9), with no statics added.
- **Agent-loop failure-mode taxonomy** (the authoring channel's own failure modes — detect + report,
  never thrash, on top of 0004's convergence budget):
  - `Issue::ContradictoryConstraints { constraints }` — a constraint set with no satisfying assignment
    (distinct from 0004's `OverConstrained`, which is i64-mismatch among *placed* joints; this is a
    declared set that is unsatisfiable *before* placement). Detected and reported; the loop does not
    thrash trying to satisfy the impossible.
  - `Issue::HallucinatedModule { id }` / `Issue::HallucinatedConnector { ref }` — the LLM referenced a kit
    module or connector that does not exist in the registry. Caught at `Intent` validation (the one door),
    reported as a precise issue rather than a panic or silent no-op (P11). This is the authoring channel's
    own taxonomy, complementing 0004's geometric `Issue`s.
- **Column ORDERS / cross-param expressions.** A **param-relation / expression layer**: a def may declare
  cross-parameter relations (a column's capital height = f(shaft diameter); the classical orders as
  parameter tables) **evaluated into `SetParam` intents** before the producers run. It is an evaluation
  pass over the `KitParams` (0003 F5) emitting ordinary `SetParam` `Intent`s through the one door — no new
  tier, no new mutation path. Cyclic relations report `Issue::ParamCycle`.

### I. Shared integer-subdivision utility (cross-cutting)

A single **integer-subdivision utility** used by the facade grammar (§E), frame bays (§E), and any
repeat-with-remainder layout: `subdivide(total, rule) -> Vec<span>` with
`rule ∈ { Fixed(n), Proportional(weights), RepeatWithRemainder(unit), Centered(unit) }`. Pure integer
arithmetic over the lattice, deterministic. Sharing one utility is what keeps the facade splits and the
frame bays *commensurate* (a window pier lines up with a frame post) instead of two drifting schemes.

### J. The PERF-BUDGET discipline (load-bearing)

The six analysis subsystems — space graph (§B), nav (§C), terrain-coupling (§F), decay (§K),
topological naming (§D), and pattern voxelization (§A) — all read resolved occupancy at >10k-XZ. **They
cannot all be live.** The discipline:

- **Always-on vs computed-on-query split.**
  - *Always-on (cheap, metadata-only):* the synchronous half — joint/connector graph math, `TopoName`
    resolution against producer output, the §I subdivision, param-expression evaluation. These read no
    resolved voxels (the 0004 SOLVE-is-sync rule), so they stay live in the loop.
  - *Computed-on-query (async):* **everything that reads resolved occupancy** — space graph, nav graph,
    terrain gradient/drainage, decay sampling, pattern *validation* (the voxelization itself rides the
    per-chunk producer pool). None is a standing structure; each is computed on demand on the async
    VALIDATE half (0004 §7), never inline in `apply_intent`. **A LOCAL query is region-scoped; a
    GLOBAL-TOPOLOGY query is seed-and-grow under a budget cap** (the two cases below) — the cost split
    that follows is the load-bearing part of this section.
- **Region-scoped (LOCAL queries).** A **local** analysis pass — the volume of a known cell,
  adjacency, `MaterialTally`, a region voxelization — takes an `Aabb64` region and reads only the
  resident chunks intersecting it (the 0003 §4 region-scoped read path). No analysis ever forces a
  full-world resolve.
- **Seed-and-grow + budget cap (GLOBAL-TOPOLOGY queries).** A **global-topology** query — is-enclosed,
  egress-count, A→B reachable, basin-watertight (§B/§C/§F) — **cannot** be region-clipped: the property
  spans the whole connected void / walkable surface / basin, so a fixed window gives wrong answers at
  the clip boundary. These run a **seed-and-grow** flood/path that **follows the connected extent across
  residency boundaries, loading chunks on demand along the frontier**, bounded by the connected extent
  itself plus an explicit **budget cap** (a working-set / frontier-size / chunk-load ceiling). They cost
  more than local passes (the set grows along the frontier), so they run on the async VALIDATE half over
  several frames. Hitting the cap before the extent closes returns **indeterminate** via
  `Issue::AnalysisBudgetExceeded` — the discipline is **never report a false negative**; report "could
  not determine" and let the agent loop decide. *(A coarse always-resident low-res occupancy summary is
  the optional cheap-approximation alternative; seed-and-grow is the exact path.)* No global-topology
  query forces a full-world resolve either — the cap bounds it.
- **Cached + incrementally invalidated via the per-chunk revision model.** Each derived graph caches its
  result keyed by the **revisions of the chunks it read**. When a chunk's revision bumps (an edit dirtied
  it), only caches that read that chunk are invalidated and recomputed (over their region), reusing the
  foundation's existing per-chunk incremental invalidation rather than a parallel dirty-tracking scheme.
  (A seed-and-grow query cache-keys on the set of chunk revisions its grow actually touched — including
  the frontier chunks — so a far edit outside the grown extent does not invalidate it.) **This requires
  one small foundation read-seam — FLAG 1, now reconciled into ADR 0003 §4/§7 (a resolved-occupancy
  revision read).**
- **Prioritized + budgeted.** Analysis passes run on the existing per-chunk worker pool under the existing
  per-frame budget (0003 §7), prioritized **anchors → structural → space → nav → decay** (cheapest /
  most-depended-on first), so the loop stays responsive: a `Reachable` query may take several frames to
  seed-and-grow its connected surface (or hit the cap and return indeterminate), but it never blocks the
  edit loop. (P10.)

### K. Decay / ruins

A **seeded DECAY / EROSION subsystem** — a producer (or an override-generating tool) taking an
**exposure / age field + a material filter**, emitting two effects:

1. **Stochastic force-OFF** (occupancy removal — crumbling), and
2. **Material reclassification** (stone → mossy-stone — a `block_id`/`variant` rewrite).

```rust
pub struct Decay {
    pub age: AgeField,            // a scalar WORLD-FRAME exposure/age field (e.g. distance-to-exterior),
                                  // residency-independent so per-SITE decay is deterministic
    pub filter: MaterialFilter,   // which block_ids decay (stone yes, bedrock no)
    pub seed: u64,                // deterministic (replayable, undoable)
}
```

- **It deliberately steps OUTSIDE the WFC invariant** ("never moves geometry / occupancy fixed before
  fill"). Decay **removes occupancy after the fact** — it is the one subsystem that mutates committed
  occupancy. **This boundary is named explicitly** [RT: "decay violates the occupancy-fixed-before-fill
  invariant"]: decay is modeled as a **corrective OVERRIDE LAYER** (0003 §3b force-off + block-rewrite
  deltas) composited **LAST** (later-wins), not as a fill producer running inside fixed occupancy. So it
  does not break WFC's invariant — it operates in the *corrective* tier, which is *defined* as the tier
  that overrides committed occupancy. Seeded + an override layer ⇒ deterministic, undoable, and
  removable (re-enable to "un-ruin").
- **Decay is pinned to the ASSEMBLY-SCOPED override tier (world-frame, this-site, non-propagating) —
  0003 `assembly_overrides`, NOT part-local sculpt.** This scope choice is load-bearing: a *part-local*
  force-off lives in the def frame and propagates to **all instances of that def identically**, so all
  12 identical columns would crumble the same way and the decay would bleed into *sheltered* instances
  that should be pristine. The assembly-scoped tier (0003 §1, the world-frame this-site-only patch that
  propagates to nothing) gives **per-site, per-instance** decay: each column crumbles by its own
  world-frame exposure. The **age/exposure field is world-frame and residency-independent** (it keys off
  absolute world position / distance-to-exterior, not the resident window), so the same site decays
  identically regardless of which chunks happen to be resident — deterministic per-site decay.
- **Cost:** it reads resolved occupancy to compute the age field, so it is a **computed-on-query**
  analysis pass (§J), region-scoped and cached.

## P1–P12 acceptance walkthrough

| # | Stress case | How the additions satisfy it |
|---|---|---|
| **P1** | Gothic cathedral: nave + aisles, ribbed VAULTS over bays, arcade COLUMNS, rose-window TRACERY, recursive facade, spire — composes + converges? | Arcade = **§A pattern** (columns, member-addressable); vaults = **§E vault producers hosted on §B space cells / bays**; rose tracery = **§A radial pattern** addressed via **§D topological naming on the window edge** (`EdgeCurveOf`); facade = **§E shape grammar** on the §I subdivision; spire = SDF producer. Converges via the 0004 budgeted loop; the **§H Unsupported fix** keeps arches/vaults from false-firing. |
| **P2** | Half-timber longhouse: frame-and-infill + thatch roof; the void RHYTHM between members | **§E frame-and-infill** (member network producer + infill-panel producer + framed-wall composite sharing one **§I bay rhythm**) authors the void rhythm directly — exactly what material-only WFC can't. Thatch roof = skeleton producer (0004). |
| **P3** | Terraced hillside fort + moat: terrain cut/fill/terrace + moat + walls draping + foundations into grade + terrain-relative datums | **§F** `Terrace`/`GradeTo`/`Excavate` sample `GroundHeightAt` (F3 coupling); walls drape (0004); foundations tie into grade; moat = **§F water-as-volume** with `BasinNotWatertight`; **F4 terrain-relative datums** anchor the terraces. |
| **P4** | Mirror a SCULPTED wing (left = reflected instance of right) | **F1 order-48** reflection — a left wing is a reflected instance of the right-wing def. The **chirality caveat** (0003 §3f): an asymmetric sculpt ornament mirrors too — exact and lossless, but handed detail flips (flagged, expected). |
| **P5** | Rotunda colonnade: N columns at 360/N, per-column capital VARIATION, one column SUPPRESSED, identity surviving a count change | **§A radial pattern**: arbitrary-angle members from the field; per-member `Override` (capital variation) + `Suppress` (the removed column), both **keyed to stable `MemberId`** so they survive a re-count; `Issue::PatternBroken` if a keyed member is dropped. |
| **P6** | "Can you walk from the gate, up the spiral stair, to the keep top floor?" | **§C `SpatialQuery::Reachable`** over the **§C nav graph** coupled to the **§B space graph**: spiral stair = **§A helical pattern** generating nav edges; floor opening = a **§B portal**; gait diagnostics (`StairTooSteep`/`HeadroomBelowMin`/`RunTooNarrow`) report if the climb is blocked. |
| **P7** | Drawbridge + portcullis + furnace — PLACEHOLDER entities | **§G proxy producer library** (reads as the thing) + **F2 `Substitution`** payload exporting to real VS blocks; `Issue::PlaceholderUnsubstituted` fires if unmapped; distinct proxy render treatment (per-draw flag). |
| **P8** | Ornament following a RAKING cornice up a gable + a string course wrapping a ROUND tower | **§D topological naming** (`wall.top_edge`, `roof.rake` as `TopoName`) + **1D edge-path connectors** + `SurfaceFrameAlong` to sweep ornament; **sub-block miter/return solver** at the gable corner; redrape-on-regen because the dressing hosts on the `TopoName`, not an address. The round-tower string course wraps continuously via the **§E 3D world-space phase field**. |
| **P9** | The Unsupported diagnostic must NOT false-fire on arches/vaults/buttresses/corbels | **§H stub fix**: `StructuralRole` (a 0005-owned `DefId`-keyed side table, Decorative/Veneer exempt; no foundation field) + **bounded load-path-aware tracing** (reachability-to-support over load-bearing cells in a bounded radius — a buildability check, not statics; lateral/arch load paths count). Only a `LoadBearing` voxel with no load-path to ground within the radius fires. |
| **P10** | PERF: all of the above on a 10k-XZ cathedral-town, loop stays responsive | **§J discipline**: cheap synchronous metadata always-on; LOCAL occupancy reads **region-scoped**, GLOBAL-TOPOLOGY queries (is-enclosed / reachable / watertight) **seed-and-grow under a budget cap** (indeterminate, never false-negative, on cap); all computed-on-query, **cached + incrementally invalidated via per-chunk `ChunkRevision`s** (FLAG 1, reconciled into 0003), prioritized under the per-frame worker budget. No analysis forces a full-world resolve. |
| **P11** | AGENT-LOOP FAILURE: contradictory constraints / unsatisfiable joint set / hallucinated kit module | **§H failure taxonomy**: `Issue::ContradictoryConstraints` (unsatisfiable declared set), 0004's `OverConstrained`/`SolveDidNotConverge` (placed-joint conflict + budget), `Issue::HallucinatedModule`/`HallucinatedConnector` (caught at `Intent` validation) — detected + reported, never thrash/crash/silent-garbage. |
| **P12** | Coursing/merlon PHASE continuity across a wall junction AND around a round tower | **§E shared 3D world-space phase field** — one `phase(world_xyz)` keys every coursing/banding/merlon producer, so courses align across a junction and wrap a round tower without a seam (the generalization of 0004's world-Y key to all axes). |

## Foundation-fit

The additions are **overwhelmingly consume-only**: new downward-depending crates (`pattern`, `space`,
`nav`, plus producer families in existing crates) + additive enum *variants* on the 0004 growth surfaces
(`SpatialQuery` / `Issue` / `VoxelProducer` registrants / `JointKind` is untouched). The gap sweep found
**no model-breaker**. **One** genuinely new foundation need surfaced and is flagged.

### ADR 0003 F-seams consumed (no change required)

- **F1 (order-48 transform):** §A pattern members are placed by lattice orientation; §G proxy orientation
  composes through it; P4 mirror is a reflected instance.
- **F2 (typed `BlockAttrs` + rotation algebra + block-entity side-table + `Substitution` + world-origin
  export):** §G placeholders ARE the `Substitution` consumer; §E per-face material is a `BlockAttrs.variant`
  channel; the VS schematic exporter consumes the world-origin contract + side-table.
- **F3 (writable terrain + `GroundHeightAt` coupling):** §F cut/fill/terrace/excavate/water producers are
  the F3 coupling's reserved consumers.
- **F4 (datums + `HostedOnDatum`, terrain-relative):** §F terraces/foundations and §B/site reasoning
  anchor to datums; terrace fan-out reuses the datum-hosting graph.
- **F5 (instance param-overrides + Def TYPE tier):** §A per-member `Override` is a `KitParams` override;
  §H column orders / param expressions evaluate into `KitParams`/`SetParam`. (Note: §H's `StructuralRole`
  is **NOT** an F5-style schema add — it is a 0005-owned `DefId`-keyed diagnostics side table, no
  foundation touch; see below.)
- **F6 (STATIC planner / placeholder ruling):** §G is the F6 placeholder consumer in full; live kinematics
  stays out.

### ADR 0004 seams consumed (no change required)

- The `Intent` door (every producer/feature expands to intents); `query(SpatialQuery)` and
  `diagnostics() -> Vec<Issue>` (additive variants throughout); the `VoxelProducer` registry +
  chunk-windowed `resolve_into` (§A/§E/§F/§G/§K producers register here); the part/assembly + ordered
  layer-stack / assembly-scoped override tiers (§K decay is a corrective override layer; §E dressings are
  features); the region-scoped read path + per-frame worker budget + async VALIDATE half (§J is built on
  it); the kit composite-generator pattern (§E framed-wall, junction modules); `SubBlockSnap` (§D miter
  solver, §E coursing). `Connector` gains a `ConnectorKind::EdgePath` (additive variant, §D).

### FLAG 1 (RECONCILED INTO ADR 0003 §4/§7): resolved-occupancy revision READ seam

**The one genuinely new foundation requirement — now reconciled into ADR 0003.** §J's incremental cache
invalidation needs to know **which resolved chunks a derived-graph computation read, and whether any has
changed since** — i.e. a **per-chunk resolved-occupancy revision/epoch counter exposed on the read
seam**, bumped when a chunk is re-resolved. ADR 0003 §4/§7 already track per-chunk dirtiness internally to
drive incremental re-mesh; this flag asked that the **revision be EXPOSED on the read API** so a consumer
can cache-key on it. **It is a read-side addition, not a data-model change** — no new payload field, no
schema cascade. **This has been reconciled into ADR 0003:** §4 (read path) now states that
`resolve_region` / the per-chunk read **returns/carries each resolved chunk's `ChunkRevision`**, and §7
ties that exposed value to the per-scope revision the re-mesh pipeline already maintains (one revision
model, two consumers). It was reconciled the same way the §1 joint-arity fix credited ADR 0004's
stress-test — cheapest while 0003 is Proposed. Without it, every analysis subsystem in §J would have to
recompute wholesale or maintain a parallel dirty-tracking scheme that duplicates the foundation's,
defeating §J's budget.

**After this reconciliation, there are NO outstanding foundation flags.** Every other addition in this
ADR consumes an existing 0003 F-seam or 0004 seam (§H's `StructuralRole` is a 0005-owned `DefId`-keyed
diagnostics side table, not a foundation field, H2; §D's in-part feature tree is a def-level side-registry
populated during the existing `Layer` fold, adding no variant/field to the frozen `Layer` enum, H3; §K's
decay rides the existing assembly-scoped override tier, H4) — none requires a foundation change.

(Watch-item, not a required change: §A pattern voxelization at large N stresses the per-chunk producer
pool's conservative-cover fan-out (0003 §3f-b); it is satisfiable with the existing rotated-chunk fan-out,
but a pathological N×huge-source pattern should be budget-capped — an implementer note, not a foundation
change.)

## Incremental build plan (lying-diagnostic fix + space-graph substrate FIRST; cheap foundation-light probes before expensive authoring)

Each step is a green checkpoint. The discipline (GD6): **build the shared substrate first, probe the
novel/risky bets cheaply before authoring breadth, reuse off-the-shelf where viable.** All steps sit
behind the same ADR 0003 phase gate as ADR 0004 (Intent door + producer registry/rotation +
query/diagnostics + joint/datum seams).

- **Q0 — the FLAG-1 revision read seam (foundation-light, FIRST).** Expose the per-chunk
  resolved-occupancy `ChunkRevision` on the read seam (reconcile into ADR 0003). Tiny, but it gates §J's
  caching and therefore every analysis subsystem — do it first so the substrates are cacheable from day
  one. *If this is not in place, the analysis subsystems cannot fit the budget.*
- **Q1 — DIAGNOSTICS hardening (cheapest, foundation-light, highest immediate value — MOVED EARLY).**
  The §H Unsupported-stub fix (`StructuralRole` 0005-owned side table + bounded load-path-aware tracing)
  + the agent-loop failure taxonomy (`ContradictoryConstraints`/`HallucinatedModule`/
  `HallucinatedConnector`) + param-expression eval. **Why first (right after Q0, before/alongside the
  space graph):** these are the cheapest and most foundation-light slices and they **stop a lying
  diagnostic from poisoning every run** — the `Unsupported` stub false-fires on every arch/vault/corbel
  and makes the agent loop thrash against legitimate geometry, so fixing it pays for itself before any
  substrate or breadth stresses the loop. No new foundation touch (the role flag is a 0005-owned side
  table, §H/H2). Probe P9 + P11.
- **Q2 — SPACE GRAPH (the shared substrate, GD6).** A pure
  `space_graph(&ResolvedRegion) -> SpaceGraph` over hand-built / fixture occupancy — flood-fill cells +
  portal detection, region-scoped, cache-keyed on `ChunkRevision`. Foundation-light (reads resolved
  occupancy via the seam; no kit, no LLM). Add `EnclosedVolume`/`SpaceAdjacency`/`PortalConnectivity`
  queries — and the **seed-and-grow global-topology path** (§B/§J) for is-enclosed / egress with the
  budget-cap + `AnalysisBudgetExceeded` escape (so a boundary-clipped flood never false-fires). **Reuse:**
  standard connected-components; build the portal-detection + naming + the cross-residency grow.
- **Q3 — TOPOLOGICAL NAMING substrate (the second substrate).** `TopoName` resolution against producer
  output + `EdgeCurveOf`/`SurfaceFrameAlong` + the `ConnectorKind::EdgePath`. Probe on a roof producer
  (`roof.ridge`/`eave`) and a wall (`top_edge`). Foundation-light; no breadth yet. This unblocks §B cell
  naming (shared mechanism), §E dressings, and §D ornament.
- **Q4 — PATTERN producer (high priority, the highest-leverage primitive).** `LinearArray` first
  (cheapest), then `Radial` (arbitrary angle from the field) and `Path`; stable **ordinal-slot**
  `MemberId` + per-member override/suppression + member-namespaced connectors + `Issue::PatternBroken`.
  Probe P5 (rotunda colonnade, suppress-one, re-count) as the acceptance test — **member identity
  surviving a count change is the load-bearing correctness bet, probed here before any breadth consumes
  it.** **Build** (no off-the-shelf fits the connector/member-id requirement). Helical (spiral stair)
  lands here too, feeding Q6.
- **Q5 — NAV / traversability (consumes Q2 space graph).** `nav_graph(&ResolvedRegion, AgentProfile)` +
  `Reachable` + gait/`UnguardedEdge` diagnostics, cached. `Reachable` is the **seed-and-grow
  global-topology** path (§C/§J) — grows along the walkable surface across residency boundaries under the
  budget cap, indeterminate on cap, never a false `Unreachable`. Probe P6 (gate → spiral stair → keep
  top) using Q4's helical stair + Q2's portals. **Reuse:** grid pathfinding (A*); build the
  walkable-derivation + profile tolerances + the cross-residency grow. Most expensive subsystem →
  strongest §J test.
- **Q6 — CONSTRUCTION systems (breadth, consumes Q3/Q4).** Layered walls, coursing/bond + the **3D
  world-space phase field** (P12), HostedOn dressings (on `TopoName`), frame-and-infill + the **§I
  subdivision utility**, vaults (hosted on Q2 space cells), shape-grammar facades. **Reuse:** CGA split-grammar
  pattern (prior art) for facades; straight-skeleton (already in 0003 producers) for roofs/vaults ridges.
  Probe P1 (cathedral), P2 (longhouse), P8 (ornament on edges).
- **Q7 — TERRAIN & site (consumes F3 + the upstream 0004 terrain-import spike).** GradeTo/Berm/Terrace/
  Excavate + water-as-volume + drainage diagnostic. `BasinNotWatertight` uses the seed-and-grow
  global-topology path (§F/§J). **Gated on the 0004 P3-terrain import-format spike** (no known VS export
  path — that spike is unchanged and upstream). Probe P3.
- **Q8 — PLACEHOLDERS & export (consumes F6/F2).** Proxy producer library + per-draw proxy render flag +
  substitute-on-export + `PlaceholderUnsubstituted`/`Unmapped` + `MaterialTally` + the VS-native schematic
  exporter. Probe P7.
- **Q9 — DECAY / ruins (last; consumes the ASSEMBLY-SCOPED corrective override tier).** Seeded decay as
  an **assembly-scoped** (world-frame, this-site, non-propagating — 0003 `assembly_overrides`) force-off
  + block-rewrite override layer composited last; world-frame age-field derivation as a
  computed-on-query pass. Probe a ruined wing (and confirm sheltered instances stay pristine).
- **Q10 — PERF pass over the whole stack (§J validation).** Run P10: the 10k-XZ cathedral-town with all
  six analysis subsystems region-scoped + cached + incrementally invalidated under the worker budget,
  and the global-topology queries seed-and-grow under their budget caps; confirm the loop stays
  responsive and no pass forces a full-world resolve.

**Reuse-vs-build summary:** *reuse* — the entire ADR 0003/0004 control surface, producer registry,
region-scoped read path + worker budget, connected-components (space graph), A* (nav), CGA split-grammar
pattern (facades, prior art), straight-skeleton (already in 0003) for roof/vault ridges, an off-the-shelf
WFC crate (0004, unchanged) for material fill. *Build* — the pattern producer with stable member identity
+ connectors (no off-the-shelf fits), the space/nav/topo-naming derived-graph subsystems + their §J
caching, the construction-system producers (layered walls, coursing, dressings, frame-and-infill, vaults,
facade grammar, 3D phase field, §I subdivision), the terrain/site producers + water/drainage, the
placeholder library + VS schematic exporter, the decay override subsystem, and the diagnostics
hardening (Unsupported fix + failure taxonomy + param expressions). *Research (upstream, unchanged)* — the
0004 VS terrain-import format spike gates Q7.

## Consequences

**Better**
- **The spine held under a 9-lens sweep** — 22 gaps, all additions, all consuming existing seams; the
  producer/voxel/three-tier model needed no rework. The one new foundation need is a small read-side
  revision seam, not a model change.
- **Three new derived-graph substrates** (space / nav / topological-naming) give voids, walkability, and
  derived geometry the same first-class, durable, queryable identity solids already had — and the breadth
  (vaults, egress, ornament, decay) falls out cheaply on top.
- **The pattern producer** turns radial/curved/helical repetition of reusable parts into one
  member-addressable primitive, with identity surviving count/pitch edits — arcades, tracery, spiral
  stairs, colonnades for free.
- **The Unsupported diagnostic becomes trustworthy** without any statics, so the agent loop stops
  thrashing against legitimate arches/vaults; the failure taxonomy makes contradictory/unsatisfiable/
  hallucinated inputs observable and local rather than crashes or silent garbage.
- **The perf budget is designed, not assumed:** a clear always-on/computed-on-query split, region-scoping,
  and per-chunk-revision incremental caching keep ~6 occupancy-reading subsystems inside the streaming
  worker budget at 10k-XZ.
- Determinism end-to-end (seeded patterns/decay/WFC + integer subdivision + the `Intent` journal) keeps the
  whole additions stack replayable, undoable, and testable as `Intent` scripts on the headless `AppCore`.

**Costs more**
- **One foundation read-seam addition** (FLAG 1, the per-chunk resolved-occupancy revision exposed on the
  read API) — small, but a foundation touch; **now reconciled into ADR 0003 §4/§7**.
- Three derived-graph subsystems + six occupancy-reading analysis passes are real engineering, each with
  its own region-scoping + cache-invalidation correctness to get right; the cache-keying on chunk revisions
  is subtle (a missed dependency = a stale graph).
- Heavy ongoing *content* work: the construction-system producers, the placeholder proxy library, the kit
  tilesets/orders tables, and the decay material maps are authored content, not just code.
- The pattern producer's stable-member-identity rule and the topological-naming selectors are novel
  correctness bets (probed early at Q3/Q2, but novel).
- Terrain/site (Q7) is gated on the unresolved upstream VS terrain-import spike (0004) — it cannot land
  until that research settles.

**Deferred (explicitly out of scope)**
- **Full statics / thrust-line / structural solver** — out; only the Unsupported-stub fix + role flag +
  load-path tracing (a connectivity trace, not force computation).
- **Live kinematics / tick simulation / state machines / signal networks** — out (F6); mechanisms are
  static proxy + Substitution annotations.
- **Sightlines / fields-of-fire / military line-of-sight analysis** — out (a derived-visibility graph,
  not built; could later reuse the §J discipline if needed).
- **Collaborative / multi-user merge** — out (0003 constraint 3: linear undo, no CRDT).
- **Units / dimensioning / measured-drawing output** — out; the planner is voxel-lattice native, export is
  block/micro-block, no metric dimensioning subsystem.
- **Terrain *authoring from scratch* inside the app** — out; terrain is imported (0004) then *mutated* by
  the §F site producers (F3), but not modeled freehand.
- **Learned decay / learned facade grammars** — out; start with seeded/authored rules.
- **Cross-part WFC / WFC as anything but per-surface material fill** — out (0004, unchanged); construction
  rhythm is kit/producer geometry, not WFC.
- **Full-generality automatic aesthetic junction resolution** — out (0004, unchanged); detect + common-case
  kit + bespoke patch + never-silent.

## Alternatives considered

- **Make the analysis subsystems always-on (live space/nav/decay graphs).** Rejected: six subsystems each
  reading resolved occupancy at 10k-XZ cannot all be live within the streaming worker budget (§J). They are
  computed-on-query, and incrementally cached instead — LOCAL queries region-scoped, GLOBAL-TOPOLOGY
  queries seed-and-grow.
- **Region-clip ALL analysis (including is-enclosed / reachable / watertight) to a fixed window.**
  Rejected as **incorrect**: enclosure, egress, A→B reachability, and basin-watertightness are **GLOBAL**
  properties of the connected void / walkable surface / basin. A residency-clipped flood-fill or path
  gives a **false negative at the clip boundary** — a cell or route that simply continues past the window
  reads as `NotEnclosed`/`Unreachable`/`BasinNotWatertight` when it is not. Global-topology queries instead
  **seed-and-grow across residency boundaries under a budget cap** and report **indeterminate**
  (`AnalysisBudgetExceeded`) on cap, never a false negative (§B/§C/§F/§J). Only genuinely LOCAL queries
  (known-cell volume, adjacency, `MaterialTally`) are region-clipped.
- **Reuse the joint-graph `Connectivity` for "can you walk there".** Rejected: structural connectivity
  (one rigid piece?) is a different question from traversable connectivity (can an agent walk it?). The nav
  graph (§C) is a distinct derived graph; conflating them was the original under-spec the sweep caught.
- **A standing/authoritative stored space graph.** Rejected: voids change on every edit; a stored
  authoritative graph would need the same invalidation as a derived one but also a persistence/consistency
  burden. It is a derived, recomputed, cached structure (§B/§J), never the source of truth.
- **Pin dressings/ornament to voxel addresses (no topological naming).** Rejected: producers regenerate, so
  address-pinned features break on every edit. `TopoName` selectors re-resolve against current output (§D).
- **Per-member pattern data keyed by array position.** Rejected: a re-count shifts positions and silently
  reassigns every per-member override/suppression. Members are keyed by stable `MemberId` derived from the
  layout slot (§A), so identity survives count/pitch changes.
- **Decay as a fill producer inside fixed occupancy (WFC-style), or as part-local sculpt.** Rejected on
  both counts: decay *removes* committed occupancy, which the fixed-occupancy invariant forbids; and a
  *part-local* override would propagate to **all instances of the def identically** (all 12 columns
  crumble the same, and decay bleeds into sheltered instances). Modeled instead as an
  **assembly-scoped** corrective override layer (world-frame, this-site, non-propagating — 0003
  `assembly_overrides`; force-off + rewrite, later-wins) with a **world-frame, residency-independent**
  exposure field, in the tier *defined* to override committed occupancy (§K) — so the invariant is
  honored and each site decays per-instance deterministically.
- **A statics / thrust subsystem to fix Unsupported.** Rejected (owner ruling: structural validity out).
  The role flag + load-path-aware connectivity trace fixes the false-firing without computing forces (§H).
- **Y-only band key for coursing (the 0004 mechanism, unchanged).** Rejected as insufficient: it handles
  vertical bands but seams on horizontal coursing wrapping a round tower or crossing a junction. Promoted to
  a full 3D world-space phase field (§E/P12).
- **A new foundation payload field for revisions / topo names / member ids.** Rejected as unnecessary:
  member ids and topo names are derived (selector rules, not stored payload), and the revision need is a
  read-API exposure of state the foundation already tracks internally (FLAG 1) — no new per-voxel payload,
  no schema cascade.
