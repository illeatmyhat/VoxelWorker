# Direct manipulation — the tool grammar

How a user creates and shapes geometry in the viewport, rather than by typing into the
inspector. Decided 2026-07-19 with the owner.

The premise, in the owner's words: *"Users should be able to prototype at the speed of
thought, not at the speed of calculation."* Direct manipulation is the primary path. The
inspector is the numeric mirror and the precision fallback — never the only way to touch an
object.

This document exists because the model spans every tool. Written per-tool it would become
per-tool improvisation, and the tools would disagree about what a click means.

## The correction this replaces

An earlier plan made the inspector the authoring surface, arguing that typing retained
`Measurement`s is more precise than dragging because a drag yields a float you then quantise.
**That argument is wrong**, and it is worth recording why, because it is easy to re-derive.

A gizmo that snaps to the lattice produces exact blocks-and-voxels *by construction* — it is
precise **and** spatial. The apparent primacy of the inspector was never a conclusion; it was
an artifact of the only surface that happened to be wired. `Intent::SetOffset` already
anticipates the other caller in its own doc: *"the inspector guarantees each axis lands on a
whole voxel before emitting."* A snapping manipulator guarantees the same thing.

## Two properties of a voxel lattice that make this cheap

Both are worth stating because they remove problems that make direct manipulation hard in a
continuous modeller.

**A surface normal is always one of six axis directions.** The march returns
`MarchHit::face_normal`, "an exact ±1 axis vector". So *snapping something to the surface under
the cursor* has an exact, finite answer — the on-surface case needs no tolerance and no
fallback.

## Rotation: the distinction that decides the widget

It is tempting — and wrong — to conclude that rotation must therefore be limited to the 24
lattice orientations. That constraint belongs to **baked voxel bodies**, not to the document
as a whole, and generalising it would cripple the tool.

* **A parametric producer is a FIELD.** An `SdfShape` box, a cylinder, a sketch→extrude: these
  are evaluated, not stored. Rotating one is exact at any density, because the rotation is
  applied to the field and the voxelisation happens *after* — for an SDF, you inverse-rotate
  the sample point. A long flat box turned 10° is a perfectly ordinary thing to author, and
  the result is not an approximation of a rotation; it is the rotated shape, sampled.
* **A baked body is an ARRAY.** A `VoxelBody`, or the cached body a linked instance shares,
  has no field behind it. Rotating one means sampling between voxels — a resample, and lossy.
  That is where lattice-preserving-only comes from, and it stays there.

So the manipulator follows the selection, which is the rule the grammar already runs on: a
Tool node offers continuous rotation; an instance of a baked body offers quarter turns. Same
rule, different affordance, no special case.

**`NodeTransform` is translation-only today**, and its own doc says the type targets a full
affine "so rotation / scale (with voxel resampling) slot in later" — which encodes exactly the
conflation above, from a time when rotation was assumed to mean resampling. Continuous
rotation of a parametric producer is therefore a NEW decision, not one ADR 0001 already made.

The real implementation cost is not the transform field; it is the evaluator. The two-layer
classifier bounds a node's field over a cell interval, and interval arithmetic under rotation
is harder than under translation: a rotated shape's axis-aligned bound is looser, so the
boundary set grows and the coarse/boundary classification has to stay conservative to remain
correct. That is the piece to cost before promising the widget.

**An uncommitted preview is free to move.** A preview participates in no boolean, so nothing
recomposes when it follows the cursor — moving it is a display transform. The cost question
below applies only to nodes that are already in the fold.

## The grammar

Every tool is the same three-part shape. Only the middle part differs.

```
  ARMED          a tool is picked on the rail
    │            a preview follows the cursor, landing at the PICKED POINT
    │
  DROP           left click: the node is created and selected
    │
  SELECTED       the node's manipulators are live, and STAY live
```

**The picked point** is the nearer of the ray's hit on existing geometry and its hit on the
ground plane. Hitting geometry also yields the face normal, which tools may use for
orientation. This is one primitive, shared by every tool, and it is the same march the display
path already runs — driven by the cursor ray instead of a pixel ray.

**Manipulators belong to the selection, not to placement.** There is no "adjust phase" to
enter or leave: a freshly dropped cylinder and one selected three days later show the same
handles. Placement is not a mode with its own editing rules; it is a way of bringing a node
into existence already selected.

## Per tool

| Tool | Preview while armed | What the drop creates | Manipulators when selected |
| --- | --- | --- | --- |
| Box / Sphere / Cylinder / Tube / Torus | the solid at default size, at the picked point | a Tool node | position (3 axis handles) · the shape's own dimensions · **continuous** rotation (it is a field) |
| Sketch | the sketch plane, **aligned to the picked face normal**, falling back to the ground plane | a sketch node on that plane | the plane's anchor · its orientation · the profile itself |
| Sculpt · add / carve | the brush core inside its reach, at the picked point | a stroke in the live session | brush radius · metric · flow |
| Measure | the measurement's first anchor | nothing in the document | the two anchors |

The sketch row is the one that shows the grammar earning its keep: "align the plane to the
surface I clicked" is the same picked-point primitive every other tool uses, plus the normal
it already returns.

## What a manipulator may do

* **Snap to the lattice, always.** The gesture is continuous; the result is an integer voxel
  count. There is no float left for a caller to commit by accident. Block-granularity snapping
  is the coarse default where it reads better; voxel granularity is the fine step.
* **Emit exactly one intent per gesture.** A drag is one undoable act. Undoing a move one
  voxel at a time would be a worse tool than no undo.
* **Carry the frame.** Manipulators work in the recentred render frame; the document takes
  blocks-and-voxels. The recentre travels with the value rather than being re-derived at the
  far end (ADR 0008).
* **Never offer what the document cannot say.** No operand picking, and no transform on a
  BAKED body that would silently resample it — but a parametric producer's rotation and scale
  are field parameters and are fair game.

## Open: what a drag costs on a committed node

A preview is free, but a *committed* node's move changes the composed geometry — other nodes
boolean against it — so something recomposes. That leaves two shapes, and the choice should be
made with a measurement rather than a preference:

* **Live per snap step.** The object itself moves under the cursor. Truest to the premise, but
  it recomposes on every step, and one gesture must still collapse to one undo entry.
* **Ghost during the drag, commit on release.** One clean intent, no recompose storm, but the
  object does not move until release — a preview does.

The number that decides it is the cost of one `SetOffset` apply plus its rebuild on a
representative scene, post-async-brick. The last figure on record (~592 ms per edit) predates
the deletion of the subsystem that dominated it, so it is not evidence about today.
