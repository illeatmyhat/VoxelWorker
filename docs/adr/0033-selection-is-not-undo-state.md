# ADR 0033 — Selection is view state, not undo state

- **Status:** Accepted
- **Date:** 2026-07-28
- **Relates to:** [ADR 0032](0032-selection-as-workspace-state.md) (the substrate this amends —
  §2 kept `selection_before` on the command "for exactly that effect"; this delta removes it),
  [ADR 0022](0022-document-dump-and-state-classification.md) (the classification selection now
  fully honors). Design map: `docs/design/tool-modes-and-navigation.md`.

## Context

ADR 0032 declared selection never-undoable but let each recorded command keep two flat scalars
(`selection_before: Option<NodeId>`, `point_selection_before: Option<usize>`) so undo could put
you back on the node you were editing. With multi-select real end to end, those scalars are a
lie: undoing any edit collapses a multi-selection to one node, silently. The choices were to
widen the record into a full `Selection` snapshot per command, or to remove selection from the
undo stack entirely.

## Decision

Follow Fusion: **the undo stack carries no selection at all.** The two fields are deleted.
Undo/redo touch only the document; what replaces the restore is a **validity prune** after any
document mutation (apply, undo, redo) — a target that no longer resolves (a `NodeId` not in the
scene, an `EntityId` not in its sketch, a dead Point id) is dropped from the set. Undoing an add
leaves nothing selected, exactly like Fusion.

Forward **steers** survive unchanged — they are dispatch effects, not undo effects: a minted
node arrives selected (replacing the node targets, as in every DCC — a new object is the sole
selection), `RemoveNode` selects a survivor.

For the prune to check identity rather than mere existence, **Reference Points get a stable
`PointId`** minted like `NodeId`; `SelectionTarget::ReferencePoint(usize)` dies. The positional
index could silently re-point at a *different* Point after edits shuffled `Scene::points` — an
existence check cannot catch that, and a positional array with per-index meaning was already
against the codebase's own law.

## Considered options

- **Snapshot the full `Selection` into each command:** rejected on prior art — Fusion does not
  capture selection into the undo stack, and the restore-on-undo behavior being protected was
  itself inherited from the pre-0032 `Scene::active` days, not designed.
- **Keep the scalars, accept the collapse:** rejected — a multi-selection that survives every
  gesture except undo is a trap.

## Consequences

- `RecordedCommand` shrinks to document deltas only; the sketch-group cancel path loses its
  selection restore with it.
- The prune is the one place selection learns about document mutations — no per-intent
  selection bookkeeping anywhere else.
- Multi-node Delete removes the **selection roots** (a node whose ancestor is also selected is
  skipped) in one undo step, mirroring the sketch multi-delete.
