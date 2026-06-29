# ADR 0006 — Authoring truth is CPU/`Intent`-authoritative; the GPU is a display + optional input shell

- **Status:** Proposed
- **Date:** 2026-06-29
- **Layer:** BOUNDARY RULING. Not a feature-on-top like [ADR 0004](0004-agent-authoring-stack.md) /
  [ADR 0005](0005-architecture-completeness.md); it sits beside [ADR 0003](0003-foundation-rework.md) (the
  foundation) and constrains [ADR 0002](0002-engine-streaming-meshing.md) (the renderer) and **any future
  GPU-pipeline work**. It pins the CPU↔GPU authority line so the recurring "shouldn't the SDF→voxel→fog
  pipeline be on the GPU?" question is answered once, with its rationale, rather than re-litigated into a
  rewrite that quietly breaks export / analysis / persistence / sculpt / agent-authoring. It introduces
  **no new foundation model** — it ratifies and names boundaries that ADR 0003/0004 already imply, and
  records the decision gate for the GPU work that is genuinely worthwhile.

## Context

A performance investigation (per-edit latency on a large/high-density scene) bottomed out in the fog
occupancy rebuild, and the fix work (parallelised producer resolve; chunk-windowed `resolve_into`,
commits `af661cd`/`d2d4d96`; fog scatter build, `83a715b`) raised a deeper question from the product
owner: **why is the entire SDF boolean → voxel → fog-slicing pipeline on the CPU at all, when that is
textbook GPU-parallel work?** Four grounded investigations followed, stress-testing a "do edits on the
GPU" proposal from three angles — display, sculpt, and agent-authoring. Their convergent finding is the
substance of this ADR.

The headline: **the CPU "one consumed truth" design is deliberate and load-bearing, and it is correct for
what this product is** — a planner/editor with export, measurement, persistence, undo, a coming sculpt
overlay, and an entire agent-authoring + analysis stack, *every one of which is a CPU consumer of
authoritative occupancy*. The SDF runs on the CPU not because nobody moved it, but because the *output* of
the resolve (the grid) must be CPU-authoritative for all the non-render features. (`REPRESENTATION.md`:
"the voxel grid is the one consumed truth"; `final = apply(overlay, evaluate(tree))`. ADR 0003: the
resolved-grid READ seam is "the single best asset … insert the compositor *behind* an unchanged read
interface.")

Two hard blockers make a *wholesale* GPU-authoritative pipeline wrong here, not merely hard:
1. **Scale.** Scenes are routinely >10k blocks XZ (ADR 0003); a whole-scene GPU volume cannot fit VRAM and
   exceeds `max_texture_dimension_3d` (the fog already self-disables past that limit). A GPU pipeline would
   still need the CPU+disk chunk store — so you would maintain *both* representations, not replace one.
2. **Determinism / authoring semantics.** The sculpt/override layer holds hand-authored voxels with no
   generating formula; re-evaluating the field would erase them (the Fusion parametric-vs-direct split,
   `REPRESENTATION.md`). Tests and goldens assert *exact* CPU occupancy; GPU float/ordering is not
   bit-reproducible. And ADR 0004's agent authors **never emit voxels** — only `Intent`s — perceives
   **data-primary** (`query`/`diagnostics` over the CPU resolved-occupancy seam; render is "a secondary
   gestalt channel, never the primary signal"), runs **headless** (MCP is a thin marshaller over the
   headless `AppCore`), and requires **deterministic replay** ("same intent script → same building").

So the honest framing is not "CPU vs GPU" wholesale. It is: **the GPU is the right place for the *display*
derivation; the CPU is the right place for the *authoritative* one.** A GPU sculpt brush is welcome — but
as a *human input transducer that synthesises an `Intent`*, recorded CPU-side, never as the source of
truth. That reconciliation is what preserves ADR 0003 §7 (`apply_intent` writes the sparse delta and marks
chunks dirty; the GPU sits *downstream* of resolve, never upstream of the journal).

## Owner rulings (decided with the product owner — honored, not relitigated)

**1. Resolve-to-truth is CPU-authoritative.** The single source of truth is the CPU model:
`final = apply(overlay, evaluate(tree))` resolved into the chunked `VoxelGrid`, plus the `Intent`/command
journal and the sparse-delta backbone. **A GPU-resident volume is never the truth.** Every non-render
consumer (`.vox` export, the diameter/layer/slice readouts, chunk persistence + disk-spill, undo, the
ADR 0004/0005 agent-authoring & analysis stack, and the lib-test/golden spine) reads the CPU resolved
occupancy. This is the ADR 0003 read seam; it is not negotiable here.

**2. One `Intent` door, many sources.** All mutation flows through the serializable `Intent` enum into
`AppCore`. Human gizmo drags, a GPU sculpt brush, and the agent/LLM/solver **all synthesise `Intent`s**
recorded on one journal. There is no `Raycast`/`Brush`/voxel-coordinate `Intent` variant and there must not
be one: a GPU edit **lowers to an integer-addressed `Intent`** (e.g. an override-layer region), recorded
CPU-side. The recorded artifact is always the integer `Intent`, so replay is deterministic regardless of
whether a GPU computed the brush region. The GPU is an author-time accelerator, never the recording
surface — this preserves ADR 0003 §7.

**3. GPU = display + optional input/compute accelerator; never required, never truth.** Headless (no-GPU)
authoring is first-class (agents, CI, dev smoke-tests). A GPU **view-resolve** (mesh/fog generated on the
GPU for display) and a GPU **sculpt brush** are both legitimate, but they are derived display/input
conveniences layered on the CPU truth — gated on being *measured* as the bottleneck, built **after** the
sculpt/override foundation lands, and kept honest behind a **CPU↔GPU A/B equivalence net** (the same
discipline ADR 0002 used for the instanced→cuboid switch). Never authoritative.

**4. Agent authoring is CPU/`Intent`/data-primary and headless-capable; the GPU edit path must never gate
it.** Per ADR 0004 the agent authors only via `Intent`, perceives **data-primary**, and runs headless.
(Status note: the `query(SpatialQuery)` / `diagnostics()` read surface is the ADR 0004 *planned* perception
API — not yet built; today only the `Intent` door and the resolved-occupancy read seam exist. This ruling
describes the intended design and does **not** depend on that surface being implemented — it depends only
on the agent authoring through `Intent` and perceiving CPU-side data rather than the GPU/rendered image.)
A GPU sculpt mode is a *human-only* accelerator. Crucially, the proposed "stream voxel diffs back
from the GPU and treat them as the delta" is **rejected** — it inverts the §7 data-flow and would break
determinism, headless agents, and the journal-as-truth model. The reconciled flow is the only one allowed:
**GPU edit → lower to `Intent` → CPU records → CPU resolves → GPU re-renders.**

**5. Human↔agent concurrency is a single-writer lock, NOT collaboration.** `AppCore` applies `Intent`s
serially and is the natural serialization point; the "lock" is an authoring-ownership gate over it. This is
explicitly *not* multi-user concurrent merge (no CRDT/OT — consistent with the "no collab" trajectory);
it is turn-taking. While the agent owns the lock, **scene-mutating** human input is **disabled (not the app
frozen)** — view/camera/selection stay live so the human can watch and inspect — and a prominent
**revoke / take-control** affordance is *always* live; revoking aborts the agent's loop cleanly at the last
fully-applied `Intent`. The gate is keyed off the existing `IntentEffect` split (scene-mutating vs
selection/view) and **enforced at the `Intent` door** (refuse scene-mutating human `Intent`s when not
owned) — UI disabling is the *affordance*, never the correctness boundary. The lock is **human-presence
arbitration**: with no human attached, the agent is unblocked by default, so unattended dev/CI runs are
never gated on a human.

**6. One `AppCore`, an optional shell — anti-divergence.** `AppCore` is *the* orchestrator and the only
authoring path: it owns the document + journal, applies `Intent`s, marks chunks dirty, runs the async
resolve, and produces the CPU resolved occupancy + diagnostics — GPU-independent. The window/render shell
is a **thin optional layer** that does exactly two things: upload `AppCore`'s resolved output to the GPU and
render it, and translate human input (gizmo, GPU brush) into `Intent`s. **Headless is not a fork — it is
`AppCore` with no shell attached.** Attended (human spectating the agent) and unattended (CI/dev smoke-test)
are the *same* `AppCore` and the *same* lock; the agent's experience (Intents in → resolve + diagnostics
out) is identical either way. This generalises the existing `bin/shot` "replay an `Intent` script through
the same `AppCore`" harness from "replay a script" to "an agent drives it live."

**7. History is a Fusion-style scrubbable operation timeline; a sculpt session is ONE operation.** The
command journal *is* the operation list — source-agnostic, so human, GPU-brush, and agent `Intent`s
interleave in one timeline. Scrubbing the marker to an earlier position is bulk undo/redo (apply
inverse-bearing entries / rebuild from the nearest checkpoint); "undo across a mixed human/agent stream" is
just a position in the one list. **A GPU sculpt-mode session is coalesced into a single operation** (one
undo unit / one accumulated override `Intent`), not thousands of per-dab entries — so the timeline stays
legible and the revoke→scrub path lets a human discard an agent's whole batch in one move.

## What we KEEP (design around, do not redesign)

- **The CPU resolved-grid READ seam** (ADR 0003) — every consumer reads a resolved `VoxelGrid`; nothing
  reads the SDF directly. Pinned.
- **The one `Intent` door + the §7 sync/async invariant** — `apply_intent` mutates the tree, writes the
  sparse delta, marks chunks dirty; it does NOT resolve/mesh inline. The GPU is downstream of resolve.
- **The `IntentEffect` classification** (scene-mutating vs selection/view) — reused as the lock gate axis
  and the disable-affordance driver. No new classification needed.
- **Chunk-windowed `VoxelProducer::resolve_into`** (ADR 0004 named it; shipped `af661cd`/`d2d4d96`) — the
  seam a future GPU view-resolve hooks behind, unchanged.
- **`AppCore` as the headless keystone** and the MCP/`shot` thin-shell pattern.

## Decision gate for the GPU work (when, not whether)

GPU display-side work is pursued **only** when (i) it is *measured* as the current bottleneck, AND (ii) the
relevant CPU foundation it derives from has stabilised (for sculpt: the override/compositor + producer
registry; ADR 0003 §3d/§3e), AND (iii) it lands behind a CPU↔GPU A/B equivalence net + byte-identical
goldens. Recorded near-term items, in order, that do NOT require any GPU-authority change:

- **Done:** parallelise producer resolve; chunk-windowed `resolve_into` (kills the per-chunk full-grid
  redundancy + restores the per-chunk memory bound); fog scatter build (kills the global-HashSet gather);
  incremental cuboid re-mesh (#40, `9ff63c3`) — only re-mesh the dirty chunks (apron-dilated via
  `cuboid_incremental_plan`), wholesale only on a floating-origin shift / density change.
- **Next (CPU, on the existing trajectory):** the fog occupancy upload + the monolithic `resolve_region`
  grid assembly still run wholesale every edit (the cuboid path no longer needs the monolithic grid) —
  Tracy-measure the post-#40 per-edit profile first; then per-chunk async resolve on a worker.
- **Then (GPU display derivation, gated as above) — now specified concretely in
  [ADR 0007](0007-gpu-view-resolve.md) (the GPU view-resolve):** stream the **compact tree** (producers +
  later sculpt deltas), not expanded voxels, to a **chunked** GPU resolver that voxelizes → fog-slices →
  meshes → instances for display, while the CPU authoritative resolve goes **on-demand**. (This supersedes
  the earlier sketch here of "compute-scatter the occupied voxel list into the R8 texture" — that still
  ships *expanded* voxels, the very cost the view-resolve removes; see ADR 0007 Alternatives.) P1 = the
  SDF tier (primitives + `SketchSolid`) → GPU per-chunk fog field; P3 (sculpt compositing) waits for the
  ADR 0003 §3e foundation; the GPU sculpt brush as a human input transducer is later still.

## Consequences

**Enables:** a coherent story where humans *and* agents author through one deterministic, replayable,
headless-capable CPU core; a human can watch an agent build live and reclaim control instantly; the GPU can
make display (and human sculpt) fast without ever endangering export/analysis/persistence/determinism;
scenes scale past VRAM because only working regions are GPU-resident.

**Forbids:** a GPU-authoritative volume; any GPU edit that bypasses the `Intent` journal ("readback = delta");
a GPU requirement on the agent-authoring or headless path; concurrent multi-writer collab/merge.

**Costs / risks to manage when the GPU work is built:** CPU↔GPU divergence (mitigated by the A/B net — the
exact `shot.rs` parallel-reimplementation debt ADR 0003 is killing, so it must not be reintroduced casually);
the GPU sculpt brush's per-session readback (bounded to the session region, coalesced to one operation per
ruling 7, validated by a spike before committing the sculpt foundation to it).

## Open / deferred sub-decisions

- **Mid-history parametric edit** (Fusion "edit an op in the timeline and replay forward") — natural since
  the journal is parametric `Intent`s, but beyond scrub-to-undo; deferred.
- **Trait vs. flag for the disable affordance** — a `Lockable`-style trait gives compile-time completeness
  (a new authoring control can't forget to be lock-aware) but egui is immediate-mode, so the idiomatic
  disabling is `ui.add_enabled_ui(human_owns_authoring, …)` per authoring section driven by the
  `IntentEffect` classification. Implementation detail; not pinned here.
- **Lock-handoff polish** — queue vs. drop of human input during agent ownership; the abort/clean-stop
  semantics on revoke mid-batch; how the spectate render cadence is throttled.
- **A spike** of the GPU-edit → bounded-readback → coalesced-`Intent` loop to measure per-session latency
  before any sculpt-foundation commitment.

## References

- `REPRESENTATION.md` — "the one consumed truth"; `final = apply(overlay, evaluate(tree))`.
- [ADR 0002](0002-engine-streaming-meshing.md) — chunked/lazy/cached resolve; the A/B golden discipline.
- [ADR 0003](0003-foundation-rework.md) — the one `Intent` door; §7 sync/async invariant; `AppCore` headless
  keystone; `IntentEffect`; chunk-windowed `resolve_into`; §3d producer registry / §3e override+sculpt.
- [ADR 0004](0004-agent-authoring-stack.md) — agent authors via `Intent` only; data-primary perception;
  pixel-primary feedback rejected; headless MCP transport; deterministic replay.
- [ADR 0005](0005-architecture-completeness.md) — analysis subsystems read the resolved-occupancy seam; the
  analysis-perf budget is the real agent-side bottleneck.
- Source: `src/intent.rs` (the `Intent` door — no raycast/voxel variant), `src/app_core.rs` (headless
  orchestrator; `IntentEffect`), `src/store.rs` (chunked store; `invalidate_aabb` returns the dirty-chunk
  set), `src/voxel.rs` (`VoxelProducer`/`resolve_into`), `src/renderer.rs` (fog occupancy → 3D texture, the
  GPU-voxelization target). Perf work: `af661cd`, `d2d4d96` (`resolve_into`), `83a715b` (fog scatter).
