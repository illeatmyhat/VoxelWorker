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

## What a drag costs on a committed node — measured

A preview is free, but a *committed* node's move changes the composed geometry, so something
recomposes. The choice was between moving **live per snap step** (truest to the premise, but it
recomposes on every step) and **ghosting during the drag** (one clean intent, but the object
does not move until release). It was left to a measurement rather than a preference.

`tests/edit_cost_probe.rs` measures both. The answer is that **the drag can be live**, and the
reason is not that rebuilds are fast — it is that *the cost tracks the dirtied volume, not the
scene*.

| backdrop | voxels | drag a small node inside it | drag the backdrop itself |
| --- | --- | --- | --- |
| 5×1×5 | 102 K | 3.5 ms · 2 chunks | 3.0 ms |
| 20×8×20 | 13 M | 4.0 ms · 2 chunks | 23.7 ms |
| 50×10×50 | 102 M | 2.5 ms · 1 chunk | 121.9 ms |
| 100×20×100 | 819 M | **1.7 ms** · 2 chunks | 501.9 ms |

Dragging a small node is **flat in scene size** — about 2–4 ms whether the scene holds a
hundred thousand voxels or eight hundred million. Targeted invalidation evicts one or two
chunks and every other resident chunk survives as a refcount bump, so an 819 M-voxel scene
costs *less* than a small one whose single node is proportionally larger.

The right-hand column is the earlier probe's case, and it is the one that grows: there the
moved node **is** the whole scene, so its dirty AABB covers everything. That is a real case
(a first node in an empty document) but it is not what a manipulator usually drags.

So the honest design is **adaptive**, not one behaviour for every scene: move live while the
last rebuild was cheap, fall back to a ghost when it was not. The gesture still collapses to
one intent on release regardless — undoing a move one voxel at a time would be a worse tool
than no undo.

**This is the rebuild only.** The brick sink rebuilds on its own worker and does not block the
frame, so this is what the user waits for — but see the seam caveat below, which is sharper
than it first looked.

### The preview is its own pass, because nothing else can carry it

Prebuild-and-downscale — build large once, show reduced versions while dragging — was measured
and rejected; `docs/design/prebuild-downscale-probe.md` has the numbers. Resampling the sparse
form costs 4–8× more than simply rebuilding, and it loses structurally: re-deriving the cuboid
decomposition *is* the work a rebuild does.

The part that constrains this document is the incidental finding. **No pipeline carries a
per-object transform** — the mesher bakes world position into each vertex and its buffers are
chunk-granular, and the brick raymarch walks one world-fixed lattice. So a preview can never be
"the committed geometry, moved or scaled". There is no seam to move it by.

That is an argument *for* the analytic-SDF preview rather than against it: a dedicated pass with
its own uniforms is cheap precisely because it owns nothing and reuses nothing. **How cheap was
then measured** — `docs/design/wgsl-sdf-spike.md`. 61 lines of WGSL cover all five shapes, zero
voxels disagree with the CPU resolve, and the objection that it would duplicate every shape's
definition is wrong: the `Field` trait already unifies them, so the GPU side is one dispatcher.
The real cost is the frame work and the ongoing per-producer obligation, not the field math.

**There is no fieldless-producer gap** — a concern raised here in an earlier draft and withdrawn.
`SdfShape` and `SketchSolid` both carry fields, a `Composite` carries one when its members do,
and `Outset` delegates. **Every producer a tool can place has a field to render**, so the
preview covers the grammar.

Two producers legitimately answer `None`, and neither is a counterexample:

* **Freehand sculpt** is occupancy-native — a sparse voxel delta has no analytic field. It is
  also not previewable in the first place: a stroke is either represented or it is not, so
  there is nothing for a preview to show. ADR 0021 Decision 5 rests the `Option` on exactly
  this case.
* **The cloud** answers `None` because its geometry *is not a distance*. ADR 0021 established
  that it is **boundable** — `cell_field_interval` classifies a cell from puff geometry with no
  noise evaluation, and that is implemented — but boundable and metric are different claims.
  `radial + BILLOW·fbm` has the right zero set and the wrong magnitude everywhere else, so
  exposing it through `Field::signed_distance` would make outset and emboss lie. The trait's own
  doc already states the test: `None` is the honest answer for a producer whose occupancy is
  real but whose geometry is not a distance.

### Noted for later: an SDF viewer mode

If the drag preview renders the parametric field directly, the machinery for *seeing the SDF
instead of the voxels* exists as a side effect. That is worth having as a viewer mode in its
own right — a way to look at what the document actually means, before voxelisation, with the
lattice out of the way. It would join the exclusive viewer modes of `docs/adr/0018` rather than
being a toggle bolted onto one tool. Not scoped, not scheduled; recorded so the preview work
does not accidentally foreclose it.

### Leaving the extent costs nothing to resolve — and that is not the reassurance it sounds like

A node dragged past the composite's current bound grows the region, which moves the floating
origin, which makes `rebuild` withhold its incremental hint — so every baked vertex buffer has
to be re-meshed. That looked like a second, costlier path worth measuring, and the third probe
measured it.

The result is that **the origin shift costs `rebuild` nothing measurable.** Holding region
growth constant and splitting a single outward drag on whether the extent midpoint actually
moved, the two regimes are indistinguishable (2.4/2.4, 1.9/1.8, 4.8/5.1, 2.0/2.0 ms). The
reason is mechanical: withholding the hint is a *branch*, not work. `invalidate_aabb` has
already run and already localised — the resident cache is frame-independent — so a reframing
rebuild re-classifies exactly the same handful of chunks and then sets a flag.

The honest conclusion is therefore **not** "extent-growing drags are free". It is that *the
cost is not at this seam*. The wholesale re-mesh is real, and it lands entirely downstream in
the shell, which `AppCore::rebuild` returns before ever reaching. Every number in this document
is a lower bound on what a drag step costs the user, and the extent-growing figures are the
loosest of them. **Measuring the re-mesh requires a probe where the mesher runs, not here.**
That is the open number now, and it is the one that decides whether the adaptive rule needs a
second trigger for "this drag left the extent".

A smaller surprise from the same table: outward steps are often *cheaper* than inside ones
(1.9 ms vs 5.3 ms on the medium scene). A node dragged out into empty space has a dirty AABB
that touches fewer occupied chunks than the same node nudged through dense geometry. At this
layer, cost tracks **locality**, not extent — the same conclusion the locality probe reached,
arrived at from the opposite direction.
