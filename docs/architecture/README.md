# VoxelWorker Architecture

VoxelWorker is a planner for chiseled voxel builds: a native application in which a
person (or an agent acting for them) designs block-and-voxel structures — walls, arches,
carved ornament — at the fidelity of the game that will ultimately host them, on scenes
far larger than any game client would willingly load. The design must stay editable
forever, render instantly at any size, and answer questions ("how wide is this at layer
40?", "export layers 12–18") exactly.

Those three demands — permanence of intent, indifference to scale, and exactness — are
in tension. This document set describes the architecture that resolves the tension. It
is not a history and not a roadmap; it describes the shape of the system, the laws that
hold it together, and why each law earns its place.

## The eight laws

Everything else in this set is elaboration of these.

1. **The document is a program.** A design is an ordered stack of authoring operations —
   parametric solids, sketches swept into volume, boolean composition, sculpted
   deltas — not a bitmap of voxels. Voxels are what the program *evaluates to*. Every
   voxel array in the system is a derived cache that may be discarded and rebuilt from
   the stack at any moment. → [The Document](01-document.md)

2. **Memory follows the surface.** No runtime state is ever proportional to volume.
   Solids are stored as their boundary plus a rule for the interior; a scene of a
   trillion voxels costs what its skin costs. Dense volumetric buffers exist only inside
   test oracles, where their honesty is the point and their cost is quarantined.
   → [Evaluation](02-evaluation.md)

3. **One door for change.** Every mutation of the document — from the panel, from a
   gizmo drag, from an agent — passes through a single serializable intent boundary,
   and every intent is undoable through one inverse-command stack. There is no second
   way to edit, so there is no second thing to test, replay, or synchronize.
   → [The Document](01-document.md)

4. **The CPU owns truth; the GPU owns the frame.** Evaluation, classification, export,
   and measurement are CPU code, correct without any GPU present. The GPU receives
   derived display caches and is free to be fast, approximate in cost but never in
   occupancy, and absent (a headless build renders the same voxels). → [Display](03-display.md)

5. **A value carries its frame.** Every spatial quantity — a placement, a recentre, an
   offset — travels with the coordinate frame it was authored in and is consumed in that
   frame. Nothing downstream re-derives a frame from context; re-derivation is how two
   halves of a system come to disagree by half a voxel. → [Evaluation](02-evaluation.md)

6. **Classified once, consumed everywhere.** One evaluator turns the operation stack
   into one classified chunk set per edit; the mesh, the brick field, the exporter, and
   the measurement queries all read that same set. Sinks never re-evaluate the stack,
   so they can never disagree with each other. → [Evaluation](02-evaluation.md)

7. **The interface never waits.** The frame loop never blocks on work proportional to
   the scene. Small work happens inline; large work happens on workers while the
   previous result keeps drawing; results that arrive late for a world that has moved
   on are discarded, and a stale artifact is never patched — only replaced. → [Work](04-work.md)

8. **Fast paths are proven, not trusted.** Every optimization — a conservative bound, an
   elision, an incremental patch, a hierarchy — ships with an oracle it must match
   byte-for-byte, and a gate that runs the comparison forever. An optimization that
   cannot state its oracle is not an optimization; it is a second implementation.
   → [Proof](05-proof.md)

## The layers

```
┌──────────────────────────────────────────────────────────────┐
│  Shell           window, input, panel UI, camera, persistence │
│                  emits Intents; draws whatever is current     │
├──────────────────────────────────────────────────────────────┤
│  Document        scene graph · producers · materials · units  │
│                  the operation stack — the only truth         │
├──────────────────────────────────────────────────────────────┤
│  Field           the signed meaning of a node; fold algebra   │
│                  metrics, outset — where geometry attaches    │
├──────────────────────────────────────────────────────────────┤
│  Evaluation      one evaluator: interval bounds → two-layer   │
│                  chunks; resident cache; targeted dirtying    │
├──────────────────────────────────────────────────────────────┤
│  Derivations     brick field (primary display) · cuboid mesh  │
│                  (understudy + oracle) · export · measures    │
├──────────────────────────────────────────────────────────────┤
│  Work            workers, generations, staleness discipline   │
│                  — the tempo that keeps the shell at 60 Hz    │
└──────────────────────────────────────────────────────────────┘
```

Data flows downward only: the shell writes intents into the document; the evaluator
derives chunks from the document; displays derive from chunks. Nothing lower ever
writes upward. When a lower layer needs to influence the document (a GPU-side brush,
an agent proposal), it does so by emitting an intent at the top like everyone else —
that is what "one door" means.

## Scope

This set describes the **core planner**: document, evaluation, display, work, proof.
One companion layer is deliberately outside it: the agent-authoring kit — the
vocabulary an autonomous builder needs *above* the intent door (connectors and joints
between parts, spatial queries, design diagnostics, generative patterns). That layer
is specified in its own documents and rests entirely on what this set describes;
nothing in the core is specific to who is editing, and an agent enters as one more
client of the same intent door.

## Reading order

| Document | What it owns |
| --- | --- |
| [01 — The Document](01-document.md) | Scene graph, parts, producers, sketches, the field layer, materials, units, intents, undo, persistence |
| [02 — Evaluation](02-evaluation.md) | The evaluator, block classification, two-layer chunks, residency, invalidation, frames |
| [03 — Display](03-display.md) | The brick field, the pyramid, the mesh understudy, engagement and handover, viewer modes |
| [04 — Work](04-work.md) | Tempos, the worker pattern, generations, the staleness law |
| [05 — Proof](05-proof.md) | Oracles, parity gates, goldens, probes — how exactness is kept |
| [Data structures](data-structures.md) | The load-bearing structures and the quality each one buys |

Terminology is defined once, in the root `CONTEXT.md` glossary; these documents use it
without redefining it.
