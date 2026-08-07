# Architecture Decision Records

This directory is **append-only decision history**: each record captures a decision at
the moment it was made — the context, the alternatives weighed, the evidence, and the
ruling. Records are never rewritten to match later reality; when reality moves, the
record's **Status** line is amended (Superseded / Retired / Amended-by / Shipped) and a
new record carries the new decision.

The **current shape of the system** is not described here. It lives in
[`docs/architecture/`](../architecture/README.md), which is edited freely and kept
timeless. The division of labor:

| Place | Role | Editing rule |
| --- | --- | --- |
| `CONTEXT.md` (repo root) | Terms and their meanings | Prune freely; terms only |
| `docs/adr/` | Decisions and their reasoning | Append-only; amend Status lines only |
| `docs/design/` | What is still to do, and what was measured to learn it | Pruned as its content folds into architecture; may reference architecture |
| `docs/architecture/` | The living shape of the system | Edit in place; no history, no roadmap, no reference to anything outside itself |

When writing a new ADR, describe the **delta** against `docs/architecture/` rather
than restating it, and update the architecture set in the same change that ships the
decision.

Read an ADR for *why* and *what was rejected* — its Status line tells you whether the
*what* still stands.

## How to use this directory

It is nine thousand lines and growing, and that is the point: this is the reasoning, not
the reference. Three ways in, in order of how often each is the right one:

1. **You want to know how the system works.** Do not read anything here — read
   [`docs/architecture/`](../architecture/README.md).
2. **You want to know why a law is the way it is, or what was rejected on the way to
   it.** Find the record in the index and read that one.
3. **You are about to overturn something.** Read the record that decided it *and* every
   record its Status line names. Supersession here is partial far more often than total,
   so "superseded" rarely means the whole record is dead.

The index is the load-bearing part. A record marked *Retired*, or superseded in full,
can be skipped entirely; one superseded *in part* cannot.

## The record index

| # | What it decided | Status |
| --- | --- | --- |
| [0001](0001-scene-graph-parts-and-tools.md) | Scene graph: parts versus tools, the assembly layer | Accepted; shipped in part — the graph, groups and definition/instance reuse are live, connectors are not |
| [0002](0002-engine-streaming-meshing.md) | The first engine phase: streaming, meshing, coordinates | **Largely superseded** by 0003 and its successors 0009/0010/0011 |
| [0003](0003-foundation-rework.md) | The foundation rework: parts, sculpt, the command journal, the streaming store | Accepted; the keystone record. Shipped in large part; several seams retired unbuilt by 0017 |
| [0004](0004-agent-authoring-stack.md) | The agent-authoring and generative building stack | **Proposed, unimplemented.** Sits above the intent door, deliberately outside the core architecture set |
| [0005](0005-architecture-completeness.md) | Pattern producer, space and nav graphs, terrain, decay, diagnostics | **Proposed, unimplemented.** A feature backlog, not a foundation decision |
| [0006](0006-authoring-truth-and-gpu-boundary.md) | The CPU owns truth; the GPU is a display and optional input shell | Accepted. This is Law 4 |
| [0008](0008-voxel-frame-invariant.md) | A spatial value carries its frame; nothing re-derives one | Accepted — the carry half binds; the decode authority is retired |
| [0009](0009-op-stack-truth-evaluator-and-sinks.md) | The operation stack is truth; one evaluator, many sinks, no resident dense grid | Accepted and implemented |
| [0010](0010-boundary-residency-two-layer-store.md) | Boundary-aware residency: the two-layer chunk store | Accepted and shipped — the sole runtime display path |
| [0011](0011-gpu-brick-field-display-sink.md) | The GPU brick field: raymarch a cached brick atlas under a clip-map pyramid | Accepted and shipped |
| [0012](0012-onion-ghost-clip-slabs.md) | Onion skin as ghost-shaded clip slabs; delete the volumetric fog | Accepted and shipped |
| [0017](0017-composition-beyond-union.md) | Composition beyond union: the ordered fold, sealed scopes, fixtures | Accepted and shipped. Supersedes parts of 0003 |
| [0018](0018-viewer-modes-and-the-root-part.md) | Exclusive viewer modes and the reified root part | Accepted and shipped; decision 3's persistence half superseded by 0024 |
| [0022](0022-document-dump-and-state-classification.md) | The document, the dump, and classified state | Accepted, **partially implemented**; decision 2 superseded by 0025 |
| [0023](0023-rollback-cache-deltas-and-keyframes.md) | The rollback cache: deltas with periodic keyframes | Accepted, **unimplemented** |
| [0024](0024-session-state.md) | Session state: the workspace comes back | Accepted and implemented; supersedes 0018 decision 3's persistence half |
| [0025](0025-embedded-session-on-save-as.md) | The author's view travels in the document, opt in on Save As | Accepted, **unimplemented**; supersedes 0022 decision 2 |
| [0028](0028-sketch-mode.md) | A sketch is a scene object you enter, editing real entities in a sealed scope | Accepted; §4's nested undo **superseded** by 0035 decision 13 |
| [0030](0030-sketch-as-entity-collection.md) | A sketch is an entity collection; the profile is derived from picked faces | Accepted; **three decisions superseded** by 0035 |
| [0031](0031-frame-phases-and-scene-draw.md) | The viewport render is ordered frame phases of one scene draw | Accepted |
| [0032](0032-selection-as-workspace-state.md) | Selection is workspace state, unified across target kinds | Accepted |
| [0033](0033-selection-is-not-undo-state.md) | Selection is view state, not undo state | Accepted |
| [0035](0035-the-sketch-tool-suite.md) | The sketch tool suite: a constraint solver, a geometric arrangement, a parametric library | Accepted; being built. Decision 3's badge pick **amended** by 0046 |
| [0036](0036-parametric-sketch-solver-ownership.md) | Parametric owns continuous sketch solving | Accepted |
| [0037](0037-curve-intrinsic-authority-and-evaluation-context.md) | Curve-intrinsic authority and density-aware evaluation | Accepted; the `ArcSweep` half **amended** by 0038 |
| [0038](0038-a-point-is-placed-never-computed.md) | A point is placed, never computed | Accepted; being built. Amends 0037 for arcs |
| [0039](0039-a-preference-is-measured-before-the-hand.md) | A preference is measured before the hand | Accepted. Amends 0038's solver consequence: an arc names its radius |
| [0040](0040-a-drag-snaps-to-the-quantity-it-moves-along.md) | A drag snaps to the quantity it moves along | Accepted. Closes 0039's open consequence: an arc endpoint sweeps |
| [0041](0041-a-gesture-is-read-from-where-it-started.md) | A gesture is read from where it started | Accepted. Slot spine handles deleted; count the hand that MOVED; a walk and its preference both measure from the opening |
| [0042](0042-a-gesture-states-its-own-rigid-set.md) | A gesture states its own rigid set | Accepted. Hands carry a Lead/Carried/Pin role instead of being told apart by their travel; a center is rigid with the curves it centers, so an arc drags whole |
| [0043](0043-a-snap-lets-go-gradually.md) | A snap lets go gradually | Accepted. The cone becomes a plateau and a smoothstep, so nothing switches at its rim; the bigger instability is measured to be the drawing's free sweep, not the snap. Amended 2026-08-06: the ring is inked from the room left in the cone, not from the hold |
| [0044](0044-an-end-of-a-round-curve-holds-its-radius.md) | An end of a round curve holds its radius | Accepted. The cone belongs to the quantity, not the drawing: a radius is held three times harder than a span because the body drag already authors it, which closes 0043's free-sweep wander |
| [0045](0045-a-snap-reaches-only-as-far-as-the-shell-allows.md) | A snap reaches only as far as the shell allows | Accepted. `SnapReach` caps the cone at a length the shell converts from 90 screen points, because an angle cannot say how far from the cursor the drawing may end up; a ceiling, never an invention |
| [0046](0046-a-badge-takes-a-click-never-a-drag.md) | A badge takes a click, never a drag | Accepted. Amends 0035 decision 3 twice: refusing the whole press over a badge made the geometry under it undraggable, and an unpicked badge draws only a glyph, so a dot inside its box outranks it. Order is dot, badge, lever, edge |
| [0047](0047-a-free-direction-is-settled-by-a-gauge-not-by-damping.md) | A free direction is settled by a gauge, not by damping | Accepted. Closes the instability 0043 ranked first and could not reach: the solve formed `JᵀJ`, which squares the condition number past what a `f64` holds, so every step came out of the damping repair and landed in the free sweep. Minimum-norm least squares on `J` itself, by complete orthogonal decomposition. 191 and 1318 become 1.5 and 2.5 |

## Records that quote documents no longer in the repo

Two rounds of pruning removed files the early records quote by path. **Those quotations
are left in place** — this directory is append-only, and a quote with attribution is
still readable as provenance. Nothing was lost; every substantive claim had already been
absorbed, usually in more depth.

**The four root design documents** (`ARCHITECTURE.md`, `DATA.md`, `REPRESENTATION.md`
and `HANDOFF.md`, deleted 2026-07-19 along with the `PROGRESS.md` milestone log)
described the project as a single-shape parametric tool ported from a browser prototype,
and had drifted into being actively misleading — they still claimed the renderer does no
raymarching, that the isolevel was a slider, and that the instance and voxel caps were
live. All three were false.

| Retired document | Where its content lives now |
| --- | --- |
| `REPRESENTATION.md` — "the voxel grid is the one consumed truth" | 0006 (quoted verbatim); the sparse-override layer in 0003 §3g |
| `ARCHITECTURE.md` §3 — the two shader-bug regression guards | 0002 (per-voxel texture slice; position-based grid overlay) |
| `ARCHITECTURE.md` §4/§5/§8 — camera rig, gizmo, palette | 0015 (the camera library), 0018, `docs/architecture/06-authoring.md` |
| `ARCHITECTURE.md` §7 — the instance and voxel caps | 0002 retires them explicitly; 0009 and 0010 dissolve the need |
| `DATA.md` — units model, install paths, chiselable block list | `docs/architecture/01-document.md` for units; the assets library is the source of truth for the rest |
| `HANDOFF.md` — tech-choice rationale, build order | `docs/DEV_NOTES.md` for pinned versions; the build order is complete and historical |

**The prior-art studies** (deleted 2026-07-30) surveyed how other tools solve placement,
storage, composition and chrome. They were removed because the design set is
self-contained by rule now, and because a survey of other products ages into an
impression. What each study concluded had already been absorbed into the record that
cited it — the reasoning survives in the ADR that acted on it, which is where it was
load-bearing in the first place.
