# CSG composition prior-art study (2026-07-15)

**Provenance:** two rounds of parallel web research (7 subagents) run during the ADR 0017
design grill, answering two questions the sculpt/CSG epic turned on. Dated snapshot; the
decisions it fed live in `docs/adr/0017-composition-beyond-union.md`. Sources are cited
per finding in the round summaries below; confidence flags are preserved from the agents'
reports.

## Round 1 — how do mature systems handle the "reusable segment vs site-specific
## junction" problem?

Systems studied: ASAM OpenDRIVE, MathWorks RoadRunner, Cities: Skylines (+ Node
Controller), Esri CityEngine, Parish–Müller 2001, Chen et al. 2008 tensor-field streets,
StreetGen (arXiv 1801.05741), Houdini road workflows (Labs Road Generator, Intersection
Stitch), EasyRoads3D, Unreal landscape splines, Revit wall joins, Fusion 360 assembly
context, SolidWorks in-context features.

**Unanimous verdict — a hierarchy, not a choice.** Across road standards, road tools,
procedural city generators, and BIM/CAD:

1. **The junction is a first-class entity distinct from the segment** (OpenDRIVE
   `<junction>` + connecting roads; RoadRunner Junction object with polygon geometry;
   CityEngine Junction/Crossing shape types with their own CGA rules; EasyRoads3D
   crossing prefab; StreetGen "transition part"). Overlap only *triggers* junction
   creation; it is never the mechanism that produces final geometry.
2. **Segments are trimmed back to a junction boundary; the junction owns the fill**
   (StreetGen cuts section surfaces at intersection border lines and builds the
   intersection polygon separately; Houdini removes curve points within the intersection
   radius and PolyCuts the junction group out; EasyRoads3D ends the road at the
   crossing's connection socket; Revit miters/butts wall layers).
3. **Pure "boolean the overlapping instances" is the fragile path everyone abandoned**
   (Unreal has no junction generator — overlapping spline meshes artifact exactly at the
   crossing, and shipped titles patch it with hand-placed meshes + decals) — **but
   bounded boolean/blend survives as the junction's internal fill strategy** (Houdini
   VDB-combine; StreetGen largest-surface-from-polylines).
4. **Site-specific edits are stored on the junction, scoped to the instance,
   structurally unable to touch the template** (Skylines Node Controller per-node
   overrides; CityEngine Convert-to-Static bake; Revit instance-end conditions vs wall
   type; Fusion Assembly Context features "updated within the context of the parent
   design only").
5. **Lifecycle: regenerate freely until hand-edited, then lock/pin; on input movement go
   *stale* and offer re-eval** — never silent recompute, never silent orphaning
   (RoadRunner automatic-vs-locked junctions; Fusion "out of sync"; SolidWorks "out of
   context"; Revit's Disallow-Join latch).

**What VoxelWorker took from round 1 (owner ruling):** a junction is **a part built to
suit the situation** — authored and placed into the assembly like any other part — not a
pre-existing thing that gets patched. This dissolves the ADR 0003 §1 "assembly-scoped
override layer" seam entirely (nothing to patch ⇒ no world-frame patch mechanism), and
it re-derives what the mature systems do (OpenDRIVE junctions *contain ordinary roads*;
EasyRoads3D custom crossings are user-authored prefabs). The reference-tracking /
stale-pinned machinery those systems carry exists to support *auto-generated* junctions;
a human-driven chiseling planner hand-authors them and accepts manual fixup, trading
automation for radical model simplicity.

## Round 2 — boolean targeting: pure ordered accumulator vs explicit operands?

Systems studied: SolidWorks (Feature Scope), Fusion 360 (Objects to Cut / Combine),
Creo/Pro-ENGINEER (30-year single-body history → 2020 multibody), Onshape (merge scope),
OpenSCAD, Blender (Boolean modifier + Geometry Nodes), Houdini Boolean SOP, 3ds Max
ProBoolean, Maya booleans, Media Molecule Dreams, Inigo Quilez SDF operators, Claybook,
Reinder Nijhoff's WebGPU SDF editor, MagicaCSG.

**Findings, by lineage:**

- **Dreams (closest intent-relative alongside MagicaCSG / normalMagic):** a sculpt is a
  flat **ordered edit list** — "each edit only affects the edits made before it"; you
  protect geometry by placing it after the subtract. Pure ordered-accumulator, shipped
  at platform scale. Crucially, the scope is the **sculpt object**: a subtract in sculpt
  A can never touch sculpt B. Order within an object; the object is the group.
  Documented pain: moving one edit does not move other edits' consequences (cuts don't
  travel with their target).
- **Contemporary SDF editors re-introduce exactly one construct: the scoping group.**
  Nijhoff's WebGPU SDF editor stores primitives as a depth-first ordered list evaluated
  with an accumulator + a group push/pop stack — "when you subtract geometry inside a
  group, it only affects other geometry within that same group." MagicaCSG organizes
  strokes into groups with "Subgroup Boolean" as its roadmap headline. Nobody ships one
  global flat list.
- **CSG-tree / node tools use explicit operands, but the tree does the scoping.**
  OpenSCAD `difference()` = first child minus the union of the rest, and cannot touch
  anything outside its braces; Blender's modifier names a target but the stack is an
  ordered local accumulator; Houdini Boolean SOP is explicit binary A/B chained. 3ds Max
  ProBoolean is the hybrid: an ordered, reorderable operation list where each entry
  still names its operand.
- **Heavyweight CAD is the cautionary tale:** Creo ran the pure ordered single-body
  accumulator for ~30 years (Pro/E v1 → Creo 6.0), then added explicit body targeting
  (Creo 7.0, 2020). SolidWorks added Feature Scope with multibody (2003); Onshape's
  Remove *errors* if no merge scope is chosen. The recurring pain: one cut slicing
  sibling bodies at the same history depth, where no reordering can separate them. All
  keep accumulator behaviour as the *default* (auto-select-intersecting / all-visible /
  merge-with-all).
- **SDF math note (IQ):** union of exact SDFs is exact; **subtraction and intersection
  produce only conservative bounds**, not true SDFs — the interval-bound evaluator must
  treat them conservatively (ours already errs to "cannot say").

**What VoxelWorker took from round 2 (owner ruling):** **ordered-accumulator semantics
scoped by the scene tree** — the exact convergent shape of the sculpting lineage
(Dreams-object / WebGPU-editor-group / OpenSCAD-braces). No per-operation operand
targeting; Groups and definition bodies are the sealed composition scopes. The
known limitation (two sibling bodies overlapping one cutter's path cannot be separated
by order alone) is answered by grouping; if that ever proves insufficient at scale, the
compatible retrofit is ProBoolean's shape (ordered list whose entries name operands) —
recorded as a rejected-for-now option in ADR 0017, not a seam.

**The fixture ruling (owner, closing the compound-fixture gap):** the one scenario the
sealed tree cannot express is a reusable part that must both **cut and fill its host**
(window = opening + frame). Revit answers with hosted families whose voids cut a
*referenced* host; the owner's ruling keeps it automatic instead: a definition may be
flagged a **fixture**, whose children splice into the hosting scope's fold at the
instance's position rather than pre-composing. The host is positional (whatever
accumulated before the instance), never a stored reference — no rehosting, no orphaned
voids, no operand UI. Sealed remains the default; the flag lives on the definition
(being a fixture is what the part *is*); a fixture instance's own combine operation is
inert.
