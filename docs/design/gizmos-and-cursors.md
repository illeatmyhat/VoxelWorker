# Gizmos and cursors — a running tally

A **deferred-design register**, not a design. As the authoring workflow is built, every gizmo (an
on-canvas manipulator) and every cursor (the pointer's feedback state) it implies gets logged here
with *what it is*, *why it is needed*, and *when it arises* — so a future agent designing them has
the whole list in one place and the rationale that motivated each. Nothing here is a spec; entries
graduate to their own design doc when picked up.

Keep it current: when a workflow decision creates a need for a manipulator or a distinct pointer
state, add a row **here** in the same change, even if the visual is months away.

Related: `direct-manipulation.md` (the authoring grammar these serve), `placement-prior-art.md`
(where the cursor states below come from), `viewport-chrome-signal.md` (the visual language they
must speak).

---

## Cursors — the pointer's feedback while an armed tool tracks

These come straight out of [`PlacementTarget`](../../crates/raycast/src/placement.rs). Placement
resolves to exactly one of four answers every frame the cursor moves with a tool armed, and each
owes a *distinct* pointer state — collapsing any two loses the corrective action the user needs.

| state | means | the pointer must say | corrective action |
| --- | --- | --- | --- |
| `OnSurface` | ray hit geometry | preview at the face, oriented to the face normal | — (place) |
| `OnWorldPlane` | ray hit a built-in plane in empty space | preview on the plane, **upright**; ideally show *which* plane (ground vs a vertical) so a fallback is not mistaken for the ground | — (place) |
| `NoSurface` | pointing at the sky — nothing in front | **"point toward the ground"** — placement is unavailable *because there is nothing there*, and this must not read as a dead app | aim at the ground or geometry |
| `TooFar` | resolved depth is sub-pixel | **"zoom in"** — placement is unavailable *because it is too small to author* | zoom in |

**The hard requirement, from the prior-art review (Vermeulen CHI 2013, NN/g):** `NoSurface` and
`TooFar` must not share an affordance. "Nothing" is strictly weaker than "nothing, because it is
too far" — only the second names the fix. A single greyed-out cursor for both is a regression to
Minecraft's one-bit vocabulary.

Open sub-questions for whoever designs these:
- Does `OnWorldPlane` visibly distinguish the ground from a vertical fallback, or is upright-preview
  enough? (The fallback is rare; over-signalling it may be noise.)
- Is the grazing hand-off (ground → vertical) animated/continuous, or a hard switch? A hard switch
  at the threshold could read as a jump.

## The armed-tool preview itself

Owner ruling (ADR 0022 context): the drag/armed preview is a **coloured transparent SDF** where the
voxels will land — nothing recomposes during the gesture. This is the body of every cursor row
above; the pointer state decorates it. Logged here because it is the shared visual all four cursors
sit inside.

---

## Manipulator gizmos — per tool, once a node is placed/selected

From `direct-manipulation.md`'s tool table. These are the on-canvas handles a *selected* node
exposes. None are designed yet.

| tool | gizmos it owes |
| --- | --- |
| Box / Sphere / Cylinder / Tube / Torus | position (3 axis handles) · the shape's own dimensions · **continuous** rotation (it is a field) |
| Sketch | the plane's anchor · its orientation · the profile itself |
| Sculpt (add / carve) | brush radius · metric · flow |
| Measure | the two anchors |

Cross-cutting needs already visible:
- **Axis-handle translation** snapped to the lattice (the position gizmo, shared by every solid).
- **Dimension handles** that read/write the node's `Measurement`-retained size (units UX).
- **Continuous rotation** — rotation is a field, so the gizmo is a free dial, not 90° steps.
- A **reference-plane** manipulator, if/when users create their own planes (the custom-plane tier
  of placement): position + orientation, distinct chrome from the built-in origin planes so the two
  are never confused.

---

## How to use this list

When you pick an entry up: pull it into its own design doc, link back here, and mark the row taken.
Do not design in this file — it is the index, kept deliberately shallow so it stays a complete map
rather than a stale half-spec.
