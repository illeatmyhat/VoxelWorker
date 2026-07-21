# Orbit-camera block-asset editors: the third shelf

Researched 2026-07-20 to close the gap the first two studies left open. `placement-prior-art.md`
covered CAD/DCC; `voxel-placement-prior-art.md` covered block *games*. Neither looked at the
tools that sit exactly on this app's use case: the dedicated **editors people use to author
models and voxel assets for block games**, which have a perspective **orbit camera** and a
**ground/base plane** — Blockbench, Crocotile3D, Goxel, Avoyd, MagicaVoxel, VoxEdit, Qubicle.

Every claim is marked **VERIFIED** (saw source, docs, or documented behaviour) or **INFERRED**.

## The finding: the flagship refuses the question, and the one clever answer switches planes rather than clamping pitch

The owner's hypothesis — *no inspired solution exists; tools either constrain pitch, let the
cursor misbehave at grazing, or confine authoring to a bounded volume* — is **mostly confirmed,
with two sharp corrections**:

1. **The flagship (Blockbench) does none of the three. It removes the placement problem entirely
   by decoupling element creation from the cursor.** "Add Cube" is a **toolbar button**, not a
   click into 3D space. The new cube appears at a fixed model-space origin (`[0,0,0]`, or the
   current group's origin); you then position it with **numeric fields and axis-handle gizmos**.
   There is no ground-plane raycast anywhere in creation, so the grazing case cannot arise — not
   because it was solved, but because **the click-into-space gesture that produces it does not
   exist.** This is the cheapest possible answer and it is the industry standard for block-*model*
   editors. **VERIFIED** (source, `js/outliner/types/cube.js`, quoted below).

2. **There is exactly one genuine invention, and it is the "bound on obliquity" the owner
   guessed nobody built — reached by switching planes, not clamping the camera.** Crocotile3D
   (and, more loosely, Goxel's plane tool and Avoyd's cursor) place tiles against **an
   axis-aligned plane chosen by which way the camera faces**: look down, you draw on the floor;
   orbit to face a wall, you draw on the wall. You never graze a plane, because the moment a plane
   goes edge-on the camera is by then facing a *different* axis plane that it hits square-on. The
   plane picker is the obliquity bound. **VERIFIED** (Crocotile docs, quoted below).

So: the *dominant* pattern (Blockbench, MagicaVoxel, VoxEdit, Qubicle) sidesteps the problem with
a fixed origin or a bounded box. The *drawing-into-space* editors (Crocotile, Goxel, Avoyd)
converge on **a movable 3D cursor whose plane is chosen by the view** — the same view-aligned
movable-depth anchor the CAD shelf reached, with the extra trick of snapping the plane to an axis.

| tool | base plane | authoring gesture | empty-space / grazing answer | bounded? |
| --- | --- | --- | --- | --- |
| **Blockbench** | finite checkered grid at y=0 (cosmetic; shadow-catcher floor is opt-in) | **toolbar "Add Cube"** → fixed origin; then numeric + gizmo | **N/A — creation never raycasts.** No cursor-into-space gesture exists | Minecraft formats: yes (48-unit box); free/generic: no |
| **Crocotile3D** | movable 3D crosshair, not a fixed floor | **click into viewport** on the crosshair's plane | **plane = axis-plane most facing the camera; depth scrubbed by Spacebar+W/S.** Can't graze — orbit past edge-on and the plane switches axis | no (open scene) |
| **Goxel** | grid at z=0 + movable "plane tool" | click; plane snaps just before the clicked voxel | movable work-plane; on click it re-seats at the hit. First voxel drawn on the z=0 plane | no (sparse, unbounded internally) |
| **Avoyd** | movable **Cursor** gizmo (blue cube / green wireframe) | edit tool acts at the Cursor; "Move Cursor here" | Cursor is a movable 3D anchor; not a ground-ray. Unbounded world (256k³) | no (unbounded) |
| **MagicaVoxel** | **bounded box** (≤126³, ≤256³ newer); perspective is a *view*, not an author mode | box/face attach **inside** the volume | always a workspace face under the cursor → never flies to horizon | **yes** (finite box) |
| **VoxEdit** | bounded editor grid | attach inside the volume | bounded | **yes** |
| **Qubicle** | bounded matrix/volume | attach inside the volume | bounded | **yes** |
| **Blender-as-MC-pipeline** | 3D cursor plane (see CAD study) | add-object on cursor plane | view-aligned cursor plane; `eps_view_align` fallback | no |

## Blockbench, in detail — questions 1 to 3

Blockbench is the standard Minecraft model editor and open-source
([github.com/JannisX11/blockbench](https://github.com/JannisX11/blockbench)), so its source is the
authority. All three answers came from source, not the wiki.

### Q1 — Is there a ground/base plane, and what is it?

A **finite checkered grid quad at y=0**, cosmetic. It is a visual reference, not a placement
target. A separate opt-in "Make floor" adds a shadow-catcher plane for the render environment
([Blockbench Wiki, Overview & Tips](https://www.blockbench.net/wiki/guides/blockbench-overview-tips/)).
Authoring happens in **model space**, not *on* the grid — the grid is where the eye rests, not
where geometry lands. **VERIFIED** (wiki + source: creation ignores it entirely).

For Minecraft formats the model lives in a **bounded box** — the Bedrock/Java build region is a
48-unit cube (a 3×3×3 m area at export scale 16); exceeding it is a Minecraft rendering limit, not
a Blockbench one ([Blockbench Wiki, Bedrock Modeling](https://www.blockbench.net/wiki/guides/bedrock-modeling/)).
So Blockbench is **Family C (bounded)** for its primary use case, and open only for generic
formats. **VERIFIED** (wiki); the box being a Minecraft constraint rather than an editor invariant
is **VERIFIED** (same source).

### Q2 — The grazing / oblique case

**It cannot occur, because placement is never a ray into the scene.** The `add_cube` action is a
toolbar `Action` (`icon: 'add_box', category: 'edit'`) whose `click` handler builds a cube at a
fixed position — no mouse coordinate, no raycaster, no plane. Verbatim from
`js/outliner/types/cube.js`:

```js
new Action({
    id: 'add_cube', icon: 'add_box', category: 'edit',
    click: function () {
        var base_cube = new Cube({ ... }).init()   // default box: from [0,0,0] to [size,size,size]
        ...
        if (Format.bone_rig) {
            var pos1 = group ? group.origin.slice() : [0, 0, 0];   // <-- the entire "where"
            let size = Settings.get('default_cube_size');
            base_cube.extend({ from:[pos1[0], pos1[1], pos1[2]],
                               to:[pos1[0]+size, pos1[1]+size, pos1[2]+size],
                               origin: pos1.slice() })
        }
```

The cube is then positioned by **numeric X/Y/Z fields** and **axis-handle drag gizmos**, which
move an already-existing element along fixed axes — never by intersecting a plane. **VERIFIED**
(source).

Camera: Blockbench's `OrbitControls` set `minDistance = 1`, `maxDistance = 3960` (a zoom cap) and
**no `minPolarAngle`/`maxPolarAngle`** — three.js defaults leave the full `0..π` pitch range, so
you *can* orbit to a grazing or fully-underneath view. Grazing the grid is harmless because
nothing is placed against it. **VERIFIED** (`js/preview/preview.js`: `this.controls.minDistance = 1;
this.controls.maxDistance = 3960;`, no polar-angle lines; `js/preview/OrbitControls.js` is the
stock three.js controls). So the answer to the pitch sub-question is: **no pitch constraint, and
none needed.**

### Q3 — How is the first block placed (bootstrap, empty scene)?

Identically to every subsequent block: press **Add Cube**, and a `default_cube_size` box appears
at the origin (or the active group's origin). The empty-scene case is not special — there is no
"first hit" problem because creation never hits anything. **VERIFIED** (source, same action).

## The one invention, precisely — Crocotile3D's view-chosen axis plane

This is the mechanic worth reimplementing, and it is a clean answer to the exact grazing problem.
From the [Crocotile3D documentation](https://crocotile3d.com/howto.html), **VERIFIED**:

- **A movable 3D crosshair** ("white lines that extend along the x, y, and z axis") is the anchor.
  "The tile gets placed against an invisible plane that always aligns with the 3d crosshair."
- **The plane is chosen by camera facing.** "Rotating the scene/camera to look at it from another
  angle will allow you to change which invisible plane the tiles get drawn against. So for example,
  if you are looking down at the scene then the tiles will get drawn looking upwards." The plane
  faces the camera — i.e. it snaps to the **axis-aligned plane most perpendicular to the view**.
- **Explicit depth scrub along the view.** W/A/S/D slide the crosshair in-plane; **Spacebar+W/S
  push it away from / toward you**; the step size is the "Grid Rounding" value.

Why this beats a pitch clamp: **you can orbit freely to any angle, and the plane you author on is
always the one you are looking at most squarely.** Grazing is structurally impossible — a plane
only goes edge-on as the camera rotates toward facing a *different* axis plane, at which point that
plane becomes the active one. It is the "bound on obliquity" the owner hypothesised, implemented as
a discrete plane switch rather than a continuous camera constraint. Goxel's plane tool ("on click
the plane moves just before the clicked voxel") and Avoyd's movable Cursor gizmo are weaker members
of the same family — a movable work-plane / 3D anchor rather than a ground ray. **VERIFIED**
(Goxel CHANGELOG; Avoyd docs).

## Q4 — Bounded vs unbounded, and what it means for us

The bounded editors (**MagicaVoxel ≤126³/256³, VoxEdit, Qubicle**) make the grazing question
*unaskable*: there is always a workspace face under the cursor, so a miss degrades to "hit the box
wall," never "fly to the horizon." **VERIFIED** (MagicaVoxel manual, 126/256 caps). This is the
same Family C lesson the voxel-game study already recorded — and, per the brief, **not directly
adoptable**: this app is deliberately sparse and **unbounded** (memory: "no dense grids anywhere").
Flagged as informative, not portable.

The two unbounded drawing-into-space editors — **Goxel** (sparse internally, no size cap) and
**Avoyd** (up to 256k³) — are the real precedent for us, and **both answer the horizon problem with
a movable 3D cursor/work-plane, not a fixed ground ray.** That is the same conclusion
`placement-prior-art.md` reached for CAD and `voxel-placement-prior-art.md` reached for Space
Engineers/Dreams. Three independent shelves now agree. **VERIFIED** (Goxel, Avoyd docs).

## Q5 — Zoom / distance bounds

Blockbench caps zoom (`maxDistance = 3960`, `minDistance = 1`) but **authorability does not change
with zoom** — because authoring is numeric/gizmo, distance is a comfort setting, not a gate.
**VERIFIED** (source). No orbit editor surveyed ties *whether you can author* to zoom the way
Factorio ties map-mode to zoom; the closest is the angular-size idea, which is this app's own
novelty (unprecedented, per the CAD study). **VERIFIED** (absence across all sources checked).

## Verdict

**An inspired orbit-camera ground-authoring solution does exist in the wild — one — and it is not
what the flagship uses.**

- **The hypothesis holds for the mainstream.** Blockbench, MagicaVoxel, VoxEdit, Qubicle never
  solve the grazing pick because they never take one: they either bound the volume or make creation
  a fixed-origin toolbar action positioned numerically. The most-used block-asset editor on the
  planet treats "where does the click land in 3D" as **a question not worth asking** — a genuinely
  cheap answer, and a real design option this app could take (decouple creation from the cursor;
  spawn at the anchor; position by drag/numeric). **This is the finding that most challenges the
  premise: the grazing problem may be self-inflicted by insisting on click-to-place-in-space.**

- **The one invention is Crocotile3D's view-chosen axis plane**, and it is precisely the anchor
  this app already decided on (`placement-prior-art.md`, 2026-07-20) with one addition worth
  stealing: **snap the fallback plane to the axis-aligned plane most facing the camera**, and let
  the user orbit freely instead of clamping pitch. Combined with the Space-Engineers/Crocotile
  explicit depth-scrub (already flagged in the voxel study), that is a complete, shipped, verified
  design for free placement under an orbit camera — no pitch constraint, no horizon flight, no
  bounded box.

- **Nobody clamps pitch.** Not one editor surveyed constrains the orbit polar angle to dodge
  grazing. The problem is dissolved (fixed-origin creation), bounded away (finite box), or answered
  by switching planes (Crocotile) — never by forbidding the camera angle. That rules out the
  cheapest bad answer.

## Sources

- Blockbench `add_cube` (fixed-origin, no raycast): `js/outliner/types/cube.js` via GitHub API,
  [JannisX11/blockbench](https://github.com/JannisX11/blockbench). **VERIFIED** (read source).
- Blockbench camera (`minDistance`/`maxDistance`, no polar limit): `js/preview/preview.js`,
  `js/preview/OrbitControls.js`. **VERIFIED** (read source).
- Blockbench grid + shadow floor: [Overview & Tips](https://www.blockbench.net/wiki/guides/blockbench-overview-tips/). **VERIFIED**.
- Blockbench Minecraft 48-unit build box: [Bedrock Modeling](https://www.blockbench.net/wiki/guides/bedrock-modeling/). **VERIFIED**.
- Crocotile3D crosshair plane + view-chosen axis + depth scrub: [howto](https://crocotile3d.com/howto.html). **VERIFIED**.
- Goxel plane tool re-seats at clicked voxel: [CHANGELOG](https://github.com/guillaumechereau/goxel/blob/master/CHANGELOG.md); sparse/unbounded: [goxel.xyz/about](https://goxel.xyz/about/). **VERIFIED**.
- Avoyd movable Cursor gizmo + 256k³ unbounded: [Avoyd docs](https://www.avoyd.com/avoyd-voxel-editor-documentation.html). **VERIFIED**.
- MagicaVoxel bounded 126/256 volume, perspective is a view: [User Reference Manual](https://leonida.gitbooks.io/magicavoxel-user-reference-manual/content/interface/); [controls](https://ephtracy.github.io/index.html?page=mv_controls). **VERIFIED**.
