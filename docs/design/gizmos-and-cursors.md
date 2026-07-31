# Gizmos and cursors — a running tally

A **deferred-design register**, not a design. Every on-canvas manipulator and every distinct
pointer state the authoring workflow implies is logged here with *what it is*, *why it is
needed*, and *the trap it must avoid* — so whoever designs them has the whole list in one place
and the reasoning that motivated each. Nothing here is a spec; an entry graduates to its own
design when it is picked up.

Keep it current: when a decision creates a need for a manipulator or a distinct pointer state,
add a row here in the same change, even if the visual is months away.

The visual language these must speak is in `docs/architecture/06-authoring.md`; the grammar
they serve is there too. Iconography is no longer tracked here — the marks are drawn and the
icon set is its own register.

---

## Cursors — the pointer while an armed tool tracks

Placement resolves to exactly one of four answers every frame the cursor moves with a tool
armed, and each owes a *distinct* pointer state. Collapsing any two loses the corrective action
the person needs.

| state | means | the pointer must say | corrective action |
| --- | --- | --- | --- |
| on a surface | the ray hit geometry | preview at the face, **seated to the surface normal** — the pivot on the contact, the object's own up along the normal | — (place) |
| on a world plane | the ray hit a built-in plane in empty space | preview on the plane, **seated to the plane normal facing the approach side** — upright on the ground seen from above, flipped when placing on the underside; ideally *which* plane, so a fallback is not mistaken for the ground | — (place) |
| nothing ahead | pointing at the sky | **"point toward the ground"** — unavailable *because there is nothing there*, and it must not read as a dead application | aim at ground or geometry |
| too far | the resolved depth is sub-pixel | **"zoom in"** — unavailable *because it is too small to author* | zoom in |

**The hard requirement:** the last two must not share an affordance. "Nothing" is strictly
weaker than "nothing, because it is too far" — only the second names the fix, and a single
grayed-out cursor for both is a regression to a one-bit vocabulary.

Open sub-questions:

- Does the world-plane state visibly distinguish the ground from a vertical fallback, or is the
  seated preview enough? The fallback is rare, and over-signaling it may be noise.
- Is the grazing hand-off from ground to vertical animated, or a hard switch? A hard switch at
  the threshold could read as a jump.
- Placing on the ground's **underside** flips the preview. Does the pointer signal which side
  it is about to seat on, or is the flipped preview self-evident?

The body inside all four is the same: a colored transparent preview of where the voxels will
land. The pointer state decorates it.

---

## Manipulator gizmos — what a selected node exposes

None of these exist. The only on-canvas manipulators today are the sketch vertex handles and
the camera's axis guide, so every transform mode is currently a mode with nothing in it.

| gizmo | what it manipulates | why, and the trap |
| --- | --- | --- |
| **position — three axis handles** | the node's offset, dragged along one world axis | the shared translate handle of every solid; it must snap to the same lattice choice as placement, or drag and drop disagree |
| **dimension handles** | the shape's own size, per axis | they read and write the *authored* units, not raw voxels — a handle that quantized to voxels would discard the retained measurement |
| **continuous rotation** | the node's orientation, at any angle | a free dial, not quarter turns, because a parametric shape is a field; its center is the **pivot**, not the bounding-box center, or a tilted object swims off its contact |
| **pivot handle** | the authoring pivot itself — base, center, or a custom point | it is both the placement anchor and the rotation center, so it must be movable; distinct chrome from the position handles, since this moves the *handle* within the object rather than the object. The data origin stays fixed and is never a grab point |
| **reference-plane manipulator** | a user-created plane's position and orientation | needed only if people create their own planes; distinct chrome from the built-in planes so the two are never confused |
| **brush radius, metric, flow** | a sculpt stroke's footprint, distance metric, and rate | deferred with sculpt; logged so the register stays complete. A brush with no visible radius is the classic invisible-cursor-size complaint |
| **measurement anchors** | the two endpoints, each lattice-snappable | they must snap like a profile vertex, or a measurement cannot be taken exactly |

---

## Inside a sketch

The entity model, the constraints and their marks, and the tool set all shipped; what follows
is only the on-canvas behavior still owed.

| gizmo | what it is | why it is needed |
| --- | --- | --- |
| **close-loop affordance** | the start point highlights when the cursor is near enough to close the path | closing is the profile's completion, and it needs an unmistakable "click here" |
| **snap indicator** | feedback when a point engages the lattice, another point, or an axis | the author must see *why* a point locked, not just that it moved |
| **working-plane manipulator** | the plane's anchor and orientation, for creating a sketch from scratch | the create-from-scratch entry; distinct chrome from the built-in planes |

| cursor state | means | the pointer must say |
| --- | --- | --- |
| **place a point** | over the working plane, ready to drop | "a point lands *here*" — where the snap will actually put it |
| **grab a point** | hovering an existing point | "this is draggable" — distinct from empty-plane hover |
| **close the loop** | near the start point with an open path | "clicking closes the profile" |
| **snap engaged** | a candidate snap is active under the cursor | "you are locked to *this*" — pairs with the snap indicator |

---

## How to use this list

When you pick an entry up, pull it into its own design file and mark the row taken. Do not
design in this file — it is the index, kept deliberately shallow so it stays a complete map
rather than a stale half-spec.
