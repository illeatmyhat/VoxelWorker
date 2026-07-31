# ADR 0032 — Selection is workspace state, unified across target kinds

- **Status:** Accepted
- **Date:** 2026-07-26
- **Relates to:** [ADR 0001](0001-scene-graph-parts-and-tools.md) (selection reaches any depth —
  kept), [ADR 0017](0017-composition-beyond-union.md) (sealed instances — a viewport pick never
  addresses inside a definition), [ADR 0022](0022-document-dump-and-state-classification.md)
  (the state categories this decision files selection under),
  [ADR 0024](0024-session-state.md) (the session route selection rides),
  [ADR 0030](0030-sketch-as-entity-collection.md) (sketch entities as selection targets).
  Design map: `docs/design/tool-modes-and-navigation.md`. Glossary: `CONTEXT.md` "Selection",
  "Selection target", "Picked node".

## Context

The interaction reframe (2026-07-23) makes left-click = Select app-wide and puts the Q/W/E/R
tool modes over "whatever is selected" — so selection becomes the shared substrate every mode
acts on. But the codebase held **three incompatible selection systems** stating opposite laws:

| what | representation | lives in | multi | undoable |
| --- | --- | --- | --- | --- |
| scene nodes | `Scene::active: Option<NodeId>` | **document** (serde-persisted) | no | **yes** — every command captured `selection_before` and undo restored it |
| Points | `Scene::active_point: Option<usize>` | document, a separate field | no | — |
| sketch entities | `SketchSelection { points, segments }` | session | yes | no ("selecting is not an edit") |

A fourth system (viewport node picking) was about to be built, and it had no representation to
land in: `pick_voxel` returns a voxel, and `NodeId` appears nowhere in `crates/evaluation` —
composed occupancy carries no node attribution.

## Decision

### 1. One `Selection`: a mixed-kind set of tagged targets

One set holding **selection targets** tagged by kind — scene node / reference Point / sketch
entity (point, segment) — not one set per kind. Which kinds *can* enter is a property of the
current editing mode (sketch mode admits sketch entities; normal mode admits nodes and Points),
a filter on admission, never a second data structure. `Scene::active` and `Scene::active_point`
are deleted. Sketch entities remain a genuinely different target kind (sealed scope, only
addressable from inside it) but are targets of the same substrate.

### 2. Selection is workspace state — never document, never undoable

The sketch side's law wins and the node side's law is repealed. Selection is classified like a
view mode or the rollback cursor: it never travels in a shared file (the same argument that
keeps Settings out of the document — "whatever I had selected" imposed on the next person), it
never enters undo history, and it **rides the dump** (full repro).

Undo/redo and structural edits still **steer** selection as a workspace *effect*: a created
node arrives selected, undoing a delete re-selects what came back. `selection_before` stays on
the command for exactly that effect — it just stops being document truth restored by undo.

**Given up, deliberately:** selection surviving save/load of a shared document (a tested
behavior). A future session-restore can bring back same-machine continuity via the ADR 0024
route without re-admitting selection into the document.

### 3. The picked node: leaf producer, additive owner, no stored attribution

A viewport click on composed geometry resolves to the **leaf producer** that made the clicked
surface — any depth, so the viewport agrees with the browser (ADR 0001) — except an
**instance**, which picks as itself (ADR 0017: nothing addresses inside a definition from the
hosting scene). Ownership follows the ordered fold: a surviving voxel belongs to the
**additive** node it survived from — clicking the wall of a carved hole picks the carved body,
never the cutter; where unioned bodies overlap, the later node in fold order wins.

The rule is **evaluated against authoring truth at the picked voxel** (leaf fields in reverse
fold order), never stored: the chunk store, bricks, and meshes stay attribution-free, honoring
"no dense grids" and truth-is-the-op-stack (ADR 0006/0009).

## Considered options

- **Per-kind selection sets** (nodes here, sketch there): rejected — marquee over a viewport
  containing a box and a Point must return both; mode exclusivity is an admission filter, not a
  reason for parallel structures.
- **Selection stays in the document, undo keeps restoring it:** rejected — the shared-file
  argument, and the sketch side had already stated the correct law.
- **Fusion-style top-level pick with drill-down** (click = component, double-click descends):
  rejected for now — bakes click-count state into the first slice, and the inspector is the
  numeric mirror of the *leaf's* params; drill-down can be added later as sugar.
- **Node-ID attribution in the chunk store / bricks:** rejected — pays memory and invalidation
  cost on every voxel for a per-click question the op stack answers exactly.

## Consequences

- Every consumer of `scene.active` / `active_node()` (~24 call sites: inspector, nodes panel,
  render seams) reads the workspace selection instead — mechanical.
- The document schema loses two fields; a loaded scene's stale `active` is ignored (no
  migration, per the no-back-compat rule for pre-alpha saves).
- Multi-select exists the moment shift-click does; the inspector shows a count summary for
  N > 1 (common-field editing deferred to the W/E/R epic).
- The settings tests asserting "the active selection survives" save/load are retired with the
  behavior.
