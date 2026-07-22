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
| `OnSurface` | ray hit geometry | preview at the face, **seated to the surface normal** (the object's pivot on the contact, its local +Z along the normal) | — (place) |
| `OnWorldPlane` | ray hit a built-in plane in empty space | preview on the plane, **seated to the plane normal facing the approach side** — upright on the ground seen from above, **flipped upside-down when placing on the underside** (uprightness retired, owner ruling 2026-07-21); ideally show *which* plane (ground vs a vertical) so a fallback is not mistaken for the ground | — (place) |
| `NoSurface` | pointing at the sky — nothing in front | **"point toward the ground"** — placement is unavailable *because there is nothing there*, and this must not read as a dead app | aim at the ground or geometry |
| `TooFar` | resolved depth is sub-pixel | **"zoom in"** — placement is unavailable *because it is too small to author* | zoom in |

**The hard requirement, from the prior-art review (Vermeulen CHI 2013, NN/g):** `NoSurface` and
`TooFar` must not share an affordance. "Nothing" is strictly weaker than "nothing, because it is
too far" — only the second names the fix. A single greyed-out cursor for both is a regression to
Minecraft's one-bit vocabulary.

Open sub-questions for whoever designs these:
- Does `OnWorldPlane` visibly distinguish the ground from a vertical fallback, or is the seated
  preview enough? (The fallback is rare; over-signalling it may be noise.)
- Is the grazing hand-off (ground → vertical) animated/continuous, or a hard switch? A hard switch
  at the threshold could read as a jump.
- Placing on the ground's **underside** flips the preview upside-down. Does the pointer signal the
  side it is about to seat on (above vs below), or is the flipped preview self-evident?

## The armed-tool preview itself

Owner ruling (ADR 0022 context): the drag/armed preview is a **coloured transparent SDF** where the
voxels will land — nothing recomposes during the gesture. This is the body of every cursor row
above; the pointer state decorates it. Logged here because it is the shared visual all four cursors
sit inside.

---

## Placement-option icons — the Add-shape dialog toggles

Small glyphs (not on-canvas gizmos, not pointer states) that label the session-durable placement
settings in the armed-tool dialog. Logged here because they are the visual vocabulary of the
placement grammar and must read as one family. All arise from ADR 0027 (placement continuity) and
its pivot ruling (this session, 2026-07-21). The **origin vs pivot** distinction underlies the
first pair: the data origin is the lattice corner (never authored directly); the pivot is the
continuous handle the user grabs, and these icons choose *which* point on the object it is.

| icon | means | why it is needed | when it arose |
| --- | --- | --- | --- |
| pivot: **base** | authoring pivot = the object's **bottom-centre** | default surface-drop anchor — the object rests its base on the contact and grows along the normal (the convergent industry default for standing/round primitives) | ADR 0027 pivot ruling |
| pivot: **center** | authoring pivot = the **volumetric centre** (centroid) | the Fusion-style alternative — the centroid lands on the contact (object half-embedded); needed the moment the dialog offers a choice | ADR 0027 pivot ruling |
| pivot: **custom** (deferred) | pivot = a user-placed point, anywhere | the general case (Blender Set-Origin / Maya `D`); off-lattice-continuous | deferred pivot tier |
| angle snap: **continuous** | the seat rotation is used exactly (any angle) | the freest orientation — a tube tilts to the true gradient normal | slice 6 (angle snap) |
| angle snap: **15°** | the seat rotation's angle quantized to 15° steps | clean mating angles without hand-alignment (position-dominant, ADR 0027 §2) | slice 6 |
| position snap: **no snap** | drop keeps the pivot exactly under the cursor (sub-voxel) | continuous placement — the fraction rides `offset_local`; the finest freedom | ADR 0027 |
| position snap: **voxel** | pivot's data corner snaps to the voxel lattice | the default — clean whole-voxel placement | ADR 0027 |
| position snap: **block** | pivot's data corner snaps to block boundaries | clean inter-part mating (offset a multiple of density) | ADR 0027 |

Family note: **base** and **center** must be instantly distinguishable at glyph size (e.g. a dot at
the base face vs a dot at the body centre of the same silhouette) — the OpenSCAD cube-vs-cylinder
anchor inconsistency (issue #3128) is the failure mode of a muddy anchor vocabulary.

## Manipulator gizmos — per tool, once a node is placed/selected

From `direct-manipulation.md`'s tool table. These are the on-canvas handles a *selected* node
exposes. None are designed yet. Per tool, what each handle is, why it is owed, and the trap it must
avoid — enough for a designer to act, still short of the spec each graduates to.

| tool | gizmo | what it manipulates | why / the trap it avoids |
| --- | --- | --- | --- |
| Box / Sphere / Cylinder / Tube / Torus | **position (3 axis handles)** | the node's `offset_voxels`, dragged along one world axis, lattice-snapped | the shared translate gizmo of every solid; must snap to the same `{None/Voxel/Block}` lattice as placement so drag and drop agree |
| ″ | **dimension handles** | the shape's own size (`Measurement`-retained), per axis | reads/writes the authored units (blocks/voxels), not raw voxels — a handle that quantized to voxels would lose the retained measurement (units UX) |
| ″ | **continuous rotation** | the node's `Quat` (any angle — it is a field) | a **free dial, not 90° steps** (ADR 0027); its centre is the **pivot**, not the bbox centre, or a tilted object swims off its contact |
| Sketch | — | the plane anchor · orientation · the profile | **see the Sketch-mode section below** (ADR 0028) — it is a whole mode, not a handful of handles |
| Sculpt (add / carve) | **brush radius · metric · flow** | the stroke's footprint, its distance metric (L2/L∞), and its rate | deferred with the sculpt epic; logged so the register stays complete — a brush with no visible radius/metric is the MagicaVoxel "invisible cursor size" complaint |
| Measure | **two anchors** | the endpoints of a measurement, each lattice-snappable | the anchors must snap to voxel/vertex/face like a profile vertex, or a measurement cannot be taken exactly |

Cross-cutting needs already visible (shared chrome the per-tool handles above reuse):
- **Axis-handle translation** snapped to the lattice (the position gizmo, shared by every solid).
- **Dimension handles** that read/write the node's `Measurement`-retained size (units UX).
- **Continuous rotation** — rotation is a field, so the gizmo is a free dial, not 90° steps. Its
  centre of rotation is the **pivot** (below), not the bbox centre — or a tilted object swims off
  its contact.
- **Pivot handle** — the authoring pivot made manipulable: a relocatable dot (base · centre ·
  custom point) that is BOTH the placement anchor and the rotation centre (origin-vs-pivot, this
  session). Distinct chrome from the position axis-handles (which move the whole object) — this
  moves the *handle* within the object (Blender Set-Origin / Maya `D` / Max Affect-Pivot-Only). The
  data origin (lattice corner) stays fixed and is never shown as a grab point; only the pivot is.
- A **reference-plane** manipulator, if/when users create their own planes (the custom-plane tier
  of placement): position + orientation, distinct chrome from the built-in origin planes so the two
  are never confused.

---

## Sketch mode — the epic's chrome, tools, gizmos, and cursors (ADR 0028)

Entering a sketch is a **mode** (ADR 0028): the left rail swaps to sketch tools, non-sketch
operations disable, and everything below is scoped to that mode. Sketch entities are **real,
directly-manipulated objects** (not previews), edited inside the sketch's self-contained undo group.
The only "ghost" kept is the pre-first-point plane affordance. Grouped by kind, most-load-bearing
first; nothing here is designed yet.

### Mode chrome — "you are editing this sketch"

**RESOLVED 2026-07-22 (owner review of the design mocks — base = the minimal "C" prototype).** No
banner, no "editing sketch / sealed scope" prose, no held-fold ribbon, no right-side entity panel. The
mode is signalled by two things and nothing more: **the rail changes** and a **CANCEL | FINISH SKETCH**
button appears. Everything the mode needs sits on the **left rail** plus a **contextual floating
tool-settings box** on the canvas.

| element | means | why it is needed | when it arose |
| --- | --- | --- | --- |
| **rail "SKETCH" label** | a highlighted `SKETCH` label at the top of the swapped left rail — the whole mode indicator | owner ruling: a full explainer banner is "too much"; the rail change + the Cancel/Finish button already tell you where you are | ADR 0028, owner review 2026-07-22 |
| **rail swap** | the left rail shows the sketch tools **and the set-operation picker** (Extrude / Revolve / Sweep) in place of the normal tool grammar | the mode's tools are different; the operation the fused sketch feeds lives here too (owner moved it off the deleted right panel) | ADR 0028 |
| **CANCEL \| FINISH SKETCH** | commit the undo group as one main-history entry / roll it back — **and** a primary mode signal | the mode's exit; its mere presence tells you you're in a mode (owner: "people will know once the rail changes and this button appears") | ADR 0028 |
| **contextual tool-settings box** | a small floating box on the canvas showing the **active tool's** settings — for the vertex/line/rectangle tools that includes **position snap** (none / voxel / block) | owner reworked C's floating readout ("useless… nowhere to change the snap") into the place tool settings live; replaces the deleted right panel as snap's home | ADR 0028, owner review 2026-07-22 |
| **disabled non-sketch ops** | non-sketch operations **withdrawn** while in the mode (C's treatment, not dim+hatch) | avoids applying a 3D op mid-sketch; the minimal read the owner preferred — gone, not greyed | ADR 0028, owner review 2026-07-22 |
| **working-plane display** | the sketch plane shown as a bounded grid, with C's subtle environment tint | the author needs to see the surface they draw on; also carries the position-snap lattice visually | ADR 0028 |
| *deleted by review* | ~~explainer banner~~ · ~~"sealed scope" prose~~ · ~~held-fold ribbon~~ · ~~right-side vertex/edge/object list~~ · ~~"live solid result" readout box~~ · ~~left browser / scene tree~~ | owner: none earn their space in sketch mode — sketch mode is **rail + canvas only** ("Fusion works fine off almost none") | owner review 2026-07-22 |
| **live solid (faint)** | the resolved solid shown as a faint translucent hint behind the profile, no readout | the ADR §4 live-volume feedback, kept subdued so the 2D profile stays primary and is never occluded | owner review 2026-07-22 (confirmed) |

**Reference mock: `sketch-c2-converged.html`** (claude.ai/design project "VoxelWorker — Future UX", group
"K · Sketch mode"). This is the locked visual target for the slice-1/2 implementation.

### Sketch rail tools (icons) — one family, must read as a set

| icon | means | why / when |
| --- | --- | --- |
| **select / move vertex** | the default arrow — pick and drag a profile vertex | the core direct-manipulation action; ADR 0028 first slice |
| **line / polyline** | click to place connected profile points | the organic value prop — arbitrary profiles; ADR 0028 slice 3 |
| **rectangle** | drag a box → a 4-point profile | the box-drag sugar, now inside the mode; ADR 0028 slice 3 |
| **delete vertex** | remove a profile point | inverse of place; ADR 0028 slice 2 |
| **position snap: none / voxel / block** | reuse the placement snap glyphs (above) — the profile vertex's lattice quantization; **lives in the contextual tool-settings box, not on the rail** (owner review 2026-07-22) | ADR 0027 snap reused for 2D (ADR 0028); lattice snapping stands in for a constraint solver |
| **operation: extrude / revolve / (sweep)** | the lift the fused sketch feeds — **the set-operation picker, on the left rail** (a property of the same node, not a separate feature) | Extrude + Revolve ship; Sweep is the reserved arm |
| *circle / arc / Bézier (deferred)* | curved profile segments | the glossary's Profile admits arcs/Béziers → flatten to the polygon; ADR 0028 slice 5 |
| *NO constraint tools* | — | deliberate absence: lattice snapping delivers axis-alignment / equal-length / coincidence as a by-product of quantization (glossary *Lattice snapping*), so there is no constraint entity to draw |

### On-canvas gizmos (while in the mode)

| gizmo | what it is | why it is needed | when it arose |
| --- | --- | --- | --- |
| **profile vertex handle** | a draggable point, position-snapped, on the working plane | the load-bearing manipulator — distinct chrome from the 3D position axis-handles (those move a whole node; this moves one profile vertex) | ADR 0028 slice 2 |
| **active / open segment** | the real segment from the last committed vertex to the cursor (and the closing segment) — a *real* entity mid-placement, not a volume ghost | you draw blind without it; reframed from "rubber-band preview" to "a real segment you're placing" (owner ruling) | ADR 0028 slice 2 |
| **close-loop affordance** | the start vertex highlights when the cursor is near enough to close the polygon | closing the loop is the profile's completion; needs an unmistakable "click here to close" | ADR 0028 slice 3 |
| **snap indicator** | feedback when a vertex engages the lattice / another vertex / an axis | this IS the constraint vocabulary (snapping stands in for a solver) — the author must see *why* a point locked | ADR 0028 slice 2 |
| **working-plane manipulator (deferred)** | plane anchor + orientation, for creating a sketch from scratch | the create-from-scratch entry (slice 4); distinct chrome from the built-in origin planes so the two never confuse | ADR 0028 slice 4 |

### Cursors — the pointer's feedback in the mode

| state | means | the pointer must say | when it arose |
| --- | --- | --- | --- |
| **on-plane / place-point** | cursor is over the working plane, ready to drop a vertex (the pre-first-point plane affordance — the one kept ghost) | "a point lands *here* on the plane" — where the lattice snap will put it | ADR 0028 slice 2 |
| **grab-vertex** | hovering an existing profile vertex | "this is draggable" — distinct from empty-plane hover | ADR 0028 slice 2 |
| **close-loop** | cursor is near the start vertex with an open polyline | "clicking closes the profile" | ADR 0028 slice 3 |
| **snap-engaged** | a candidate snap (grid / vertex / axis) is active under the cursor | "you are locked to *this*" — pairs with the snap-indicator gizmo | ADR 0028 slice 2 |

---

## How to use this list

When you pick an entry up: pull it into its own design doc, link back here, and mark the row taken.
Do not design in this file — it is the index, kept deliberately shallow so it stays a complete map
rather than a stale half-spec.
