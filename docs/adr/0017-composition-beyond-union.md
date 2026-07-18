# ADR 0017 — Composition beyond union: the ordered fold, sealed scopes, and fixtures

- **Status:** Accepted (2026-07-15 — design grill + two prior-art study rounds,
  `docs/design/csg-prior-art-study.md`; **implementation not started** — epic #72,
  slices #73–#77). Supersedes, in
  part, ADR 0003: the §1 *assembly-scoped override layers* seam is retired unbuilt
  (junctions are parts, nothing patches the composed result), and the §3b per-part
  `Vec<Layer>` stack is not adopted — composition stays node-granular on the scene graph,
  extended by the scope/fixture rules below. ADR 0003's §3e *def-local sculpt override*
  remains reserved for the follow-on freehand-sculpt ADR.
- **Date:** 2026-07-15
- **Layer:** document model + evaluation semantics. The first epic of the
  sculpt/chiseling arc; freehand sculpt sequences after it.

## Context

The sculpt epic's design grill opened on "where does a hand-sculpt override attach?" and
redirected twice, both times on owner rulings backed by prior-art studies:

1. **Junctions are parts.** The fortress-corner scenario (two wall instances meeting at a
   bastion needing a bespoke, non-propagating patch) was studied against 13 road/CAD/BIM
   systems. All reify the junction as a first-class entity whose geometry is authored
   like any other part's, with segments trimmed back to its boundary — never a patch on
   emergent overlap. The owner's ruling sharpened it: *a junction is a part built to
   satisfy the needs of every connecting object* — created and added to the assembly to
   suit the situation. There is no pre-existing junction to modify, so there is no
   assembly-level sculpt/override mechanism to build. ADR 0003 §1's `assembly_overrides`
   seam dissolves.
2. **CSG before freehand sculpt.** Doors, windows, stairs, and penetrations — and the
   subtractive side of junctions — are parametric boolean cutters, not hand-chiseled
   voxels. Building the reserved `CombineOp::Subtract`/`Intersect` arms is smaller, more
   general, and more immediately valuable than the sparse-delta sculpt tier, and the
   sculpt tier composes on top of the same fold later. This ADR is that epic.

What exists today: composition is node-granular (`Node.operation: CombineOp`) and
union-only; `Group` is a transparent pass-through in the resolver; a definition's body
resolves inline under its instances' transforms. The `CombineOp` enum has carried
`// future: Subtract, Intersect` since ADR 0001.

The targeting question — *what does a subtract remove from?* — was studied against three
lineages (heavyweight CAD, CSG-tree/node tools, SDF edit-list sculptors; see the study
doc). The sculpting lineage (Dreams, MagicaCSG, normalMagic, Nijhoff's WebGPU editor —
our intent-relatives) converges on **ordered-accumulator semantics within a scope
boundary that subtraction cannot cross**, with no per-operation operand selection. The
owner's law: *booleans apply in a predictable order of operations — no exceptions, no
escape hatches; if you don't want things cut, put them after the subtract.*

## Decision

1. **`CombineOp` gains `Subtract` and `Intersect`.** `Union` keeps later-wins material
   on overlap. `Subtract` and `Intersect` are **occupancy-only masks**: subtract clears
   cells, intersect keeps only cells present in both the accumulated result and the
   node's body (surviving cells keep their accumulated material). Boolean nodes never
   stamp material.
2. **Composition is an ordered depth-first fold with NO operand targeting — this is a
   law, not a default.** Within a scope, children evaluate in spine order; each folds
   into the accumulated result under its `CombineOp`. A boolean affects everything
   accumulated before it in its scope. Geometry is protected by *placement* — after the
   cutter, or in a sibling scope — never by per-operation target selection. No Feature
   Scope, no merge scope, no "objects to cut" UI, ever.
3. **Groups and definition bodies are sealed composition scopes.** A `Group` resolves
   its children into one body, then folds that body into its parent under the Group's
   own `CombineOp`; a boolean inside a scope can never affect geometry outside it. A
   definition pre-composes the same way, and an instance places the finished body under
   the instance's `CombineOp` (so a *reusable cutter* is just a definition instanced
   with `Subtract` — no new node kind).
4. **A definition may be flagged a `fixture`.** A fixture does **not** pre-compose: its
   children splice into the hosting scope's fold at the instance's spine position, in
   order, under the instance's transform. A window def = [opening cutter `Subtract`,
   frame `Union`] cuts its host wall and fills the frame with one placement. The host is
   **positional** — whatever accumulated before the instance in its scope — never a
   stored reference: no rehosting, no orphaned voids, no host-tracking lifecycle. The
   flag lives on the definition (being a fixture is what the part *is*), instances stay
   pure reference+transform, and a fixture instance's own `CombineOp` is inert (the UI
   hides it). A fixture pierces exactly one level of pre-composition — its host scope's
   seal above it remains absolute.
5. **Junctions are ordinary parts** — additive corner pieces placed with `Union`, or
   fixtures when they must carve their neighbors. No assembly-level override layer, no
   world-frame patch storage, no junction entity kind.
6. **Evaluator contract:** union of exact occupancy-interval bounds stays exact-ish, but
   **subtraction and intersection propagate only conservative bounds** (IQ's SDF result
   holds for interval classification too). The two-layer classifier's "cannot say ⇒
   per-voxel evaluation" posture (ADR 0009 §3–§4) already absorbs this; the fold's bound
   algebra must err toward boundary, never toward air/solid.
7. **Freehand sculpt is the follow-on ADR**, not this one. A sculpt body will enter the
   same fold as one more step (ADR 0003 §3e's def-local sparse override, on this ADR's
   scope rules); nothing here may assume geometry is producer-analytic only.

## Considered options

- **Explicit operand targeting** (SolidWorks Feature Scope, Onshape merge scope, Fusion
  "Objects to Cut"): rejected. That machinery exists because flat multibody part files
  have no tree to scope with; our scene tree is the scope container. The known
  limitation — two sibling bodies overlapping one cutter's path can't be separated by
  order alone — is answered by grouping. If it ever bites at scale, the compatible
  retrofit is 3ds Max ProBoolean's shape (ordered list whose entries also name
  operands); recorded here so it isn't re-derived, deliberately not built.
- **Transparent definitions (macro-splice by default)**: rejected — editing a def could
  silently start cutting host geometry in every assembly that places it. Sealing is the
  default precisely to prevent that spooky action; the fixture flag makes piercing a
  deliberate, per-definition declaration.
- **Revit-style hosted voids (host as stored reference)**: rejected in favor of the
  positional host. References buy auto-update at the cost of a rehost/orphan lifecycle;
  a human-driven planner prefers the rule "the host is what you placed it after."
- **Reviving ADR 0003 §3b's per-part `Vec<Layer>`**: rejected — the node graph already
  is the ordered composition structure; a second intra-part layer list would duplicate
  it. (The sculpt follow-on attaches its override to this fold instead.)
- **Assembly-scoped override layers (ADR 0003 §1)**: retired unbuilt, per Decision 5.

## Consequences

- **`Group` becomes a composition boundary — a resolver behavior change.** For
  pure-`Union` scenes (all existing documents) the result is provably identical (with
  later-wins material, the winning writer at a voxel is the last DFS writer whether or
  not groups pre-compose), so current goldens stand; the boundary only becomes
  observable once `Subtract`/`Intersect` exist. New goldens must cover: cutter in
  scope / out of scope, fixture splice, cutter-before-target no-op.
- **The tree serves two masters** — organization and boolean semantics. Users may need
  to restructure groups to get the composition they want. Accepted deliberately: "it
  depends entirely on how the assembly is organized" is the model, and it is how the
  intent-relative tools (OpenSCAD, MagicaCSG) already think.
- **The deferred box-drag-on-plane primitive becomes the cutter entry** — "draw a box,
  pull it through the wall, set Subtract" is the zero-ceremony penetration workflow this
  fold enables.
- **Incremental invalidation is unchanged in kind**: a cutter/fixture dirties chunks
  within its AABB via the existing edit-broadphase BVH. One new fact: a `Subtract` can
  turn coarse-solid blocks into boundary or air, so invalidation must re-classify, not
  merely re-mesh (the store already re-classifies dirtied chunks).
- **`docs/architecture/01-document.md` gains a Composition section when the fold
  ships** (the architecture set describes the living system; this ADR is the delta).
- The fixture splice is the one place instance identity and fold position interact;
  the implementation must keep ADR 0008's carried-frame discipline when the fixture's
  children enter the host fold under the instance transform.

## Amendment 2026-07-18 — one cited intent-relative was mis-identified

The Context section above names the sculpting lineage as "Dreams, MagicaCSG, normalMagic,
Nijhoff's WebGPU editor". **normalMagic does not belong in that list.** It is a Blender
add-on for mesh normal control (SpaghetMeNot, `normalmagic/2.0/`), not an SDF edit-list
sculptor; its Boolean Pro replaces Blender's Boolean modifier to keep booleans from
wrecking mesh shading. It was never among the systems the study actually examined — it
was name-dropped into the lineage parenthetical and carried here.

**The decision is unaffected.** The ordered-accumulator-within-a-scope convergence is
established by Dreams, MagicaCSG, Nijhoff's editor and OpenSCAD independently; removing
the fourth name changes no premise of Decisions 1–7. This amendment corrects the
attribution rather than editing the record above.

The study doc is corrected in place and now carries what normalMagic *is* worth reading
for — per-cut **outset** as a borrowable authoring affordance, **slice mode** as an open
question against the *junctions are parts* law. Neither is decided here.
