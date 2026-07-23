# Sketch selection — the Fusion-style select/delete model

How a user selects and deletes entities inside a sketch (ADR 0030 — a sketch is a collection of
points, segments, later arcs and derived faces). Decided with the owner 2026-07-23; supersedes the
tool grammar ADR 0028 shipped for sketch editing (the three-tool Select / Add-point / **Delete**
rail). This is the living spec; it graduates to an ADR 0030 amendment once the slices land.

The premise, in the owner's words: *Fusion's selection is the model.* Select is the one place you
touch geometry; delete is something you **do to a selection**, not a mode you enter.

## The correction this replaces

ADR 0028 (#95) made **Delete a mode** — a rail tool you arm, then click an entity to remove it,
with a warn-`✕` hover to show what a click would take. That was shipped, including a segment
delete-hover (2026-07-23). Fusion has no delete mode: delete lives only on the selection, reached
by the **Delete key** or the **right-click context menu**. Keeping delete as a mode fights muscle
memory and burns a rail slot. So Delete stops being a tool; its warn-`✕` visual survives as the
"armed to remove" cue, but its *trigger* moves from tool-hover to the selection.

## What is selectable

A **selection set** of mixed entities held on the sketch editing session:

- **Points** and **segments** — first-class, directly selectable (ADR 0030 entities with stable ids).
- **Faces** — a *derived* entity (graph → faces, ADR 0030 §region). First-class **pickable**, so the
  same selection machinery drives face pick/unpick for extrusion (#100). Deferred, but the model
  treats a face as a selectable from the start so #100 is a data addition, not a rewrite.

Selecting a point and selecting the segment between two points are different: a segment in the set
carries its own id; deleting it removes only the line, while deleting a point cascades its incident
segments (ADR 0030 delete semantics, already built).

## Building the selection

The Select tool, no mode switch:

1. **Click an entity** → the selection becomes exactly that entity.
2. **Shift-click** → toggle that entity in/out of the set (accumulate).
3. **Click empty space** → clear the set.
4. **Marquee drag from empty space** → box-select (below).

Vertices keep priority over segments in every hit-test (a click or box near a shared endpoint
resolves to the point), matching the existing Select-grab and delete hit order.

## The directional marquee (window vs crossing)

Fusion's two-direction box, so the user picks the semantic by drag direction and reads it by style:

| Drag direction | Mode | Selects | Outline | Fill |
| --- | --- | --- | --- | --- |
| **left → right** | **Window** | entities **fully enclosed** by the box | **solid** | faint **accent** |
| **right → left** | **Crossing** | entities the box **intersects** (any overlap) | **dashed** | lighter |

- **Window** (drag right): points inside the box; segments incident to those inside points (later
  also segments tied by constraint logic); and any other entity fully enclosed. The "I meant this
  whole thing" box.
- **Crossing** (drag left): any entity the box touches, so you can grab *part* of a face or a run of
  edges without enclosing all of it. The "reach across" box.
- **Two distinct styles are required**, not decorative: the user must tell window from crossing at a
  glance mid-drag. Solid-outline/filled = window; dashed-outline/lighter = crossing. Dashed already
  means "looser / uncommitted" in the gizmo family (`dashed_segment`, `dashed_rect`), so it reads as
  the reaching box without a new idiom. Colours are Signal tokens (`ACCENT`), never Fusion's literal
  blue/green.

**Open — window-select segment predicate:** is a segment with *one* endpoint inside selected
(because it is "associated with an inside point"), or only when *both* endpoints are inside (strictly
"fully enclosed")? The owner's phrasing allows both; resolve before the marquee slice.

## Delete as an action

- **Delete / Backspace key** with a non-empty selection → delete every selected entity (points
  cascade their segments, ADR 0030), one undo step.
- **Right-click → Delete** in the context menu, same effect.
- The Delete **tool is removed** from the sketch rail. Rail becomes Select + Add-point (Add-point's
  eventual fold into Select is a separate, later question).

## The context menu

No viewport context menu exists today (only the ViewCube's own right-click menu). Build a
**general-purpose viewport right-click menu** for all modes, its contents **overridden per mode**:

- Base (any mode): the shared actions (TBD — the point is the *infrastructure* is general).
- **Sketch mode override:** Delete (on a selection) now; face pick/unpick and add-arc etc. later.

This is the surface #100 (region pick/unpick) and future sketch verbs hang off, so it is built as a
mode-dispatched menu, not a sketch-only widget.

## Slices (tracer-bullet order)

1. **Selection set + click / shift-click / clear** — selection state on the session; a `Selected`
   visual for points *and* segments (define the token, distinct from Idle/Hover); Select-tool click
   wiring; vertices-over-segments hit priority.
2. **Delete as action** — Delete key + a general context menu (infra) with a sketch Delete override;
   remove the Delete tool from the rail; retire the now-dead delete-hover trigger.
3. **Directional marquee** — window/crossing predicates + the two-style rubber band; resolve the
   window-segment predicate first.
4. **Faces as derived selectables** — later, feeds #100 extrusion pick/unpick.

## What this reuses / retires

- **Keeps:** segment-line rendering (an open sketch's edges must show); the Select-hover highlight
  (it *is* the selection hover); the `marked_segment` / `Marked` gizmos (repurposed to "armed to
  delete" for a selection); the entity-id delete ops (`with_point_deleted` / `with_segment_deleted`).
- **Retires:** the Delete rail tool and its hover trigger; the tool-armed delete press path.
