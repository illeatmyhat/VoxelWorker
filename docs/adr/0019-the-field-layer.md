# ADR 0019 — The field layer: profiles, metrics, and outset

- **Status:** Accepted (2026-07-18 — design grill; **implementation not started**).
  **Extended by ADR 0020**, which settles the boolean sugar Decision 10 left open and fixes
  the field trait shape. Extends ADR 0017's ordered fold with a named seam beneath it. Does not alter any ADR 0017
  law: the fold stays ordered DFS with no operand targeting, Groups and definition bodies
  stay sealed scopes, fixtures stay def-level flags with positional hosts, and
  `Subtract`/`Intersect` stay occupancy-only masks. Confirms and depends on ADR 0011's
  display boundary. Supplies the field contract ADR 0009/0010's classifier already assumed.
- **Date:** 2026-07-18
- **Layer:** document model + evaluation semantics, with one substrate extraction. The
  foundation the remaining boolean affordances attach to.

## Context

The session opened on borrowing boolean authoring affordances from normalMagic 2.0 —
per-cut **outset**, **slice mode**, and named sugar (Boolean Extrude / Trim / Cut Groove).
Designing outset immediately exposed that the layer it belongs to has no name.

VoxelWorker names three layers and documents each: **Intent** (the op stack — nodes, order,
scopes, fixtures), **Occupancy** (the two-layer boundary store), and **Display** (bricks,
mesh, ghosts). Between Intent and Occupancy sits a fourth that is real but undeclared: the
**field** — the signed scalar meaning of a node. Every producer has one. The classifier
consumes one. The fold's `min`/`max` algebra *is* field algebra. But because it was never
declared it has no contract, no uniform representation, and nowhere for a feature to attach.

Two symptoms of the missing declaration, both found in the code during the grill:

1. **`SketchSolid` smuggles a three-valued enum through `FieldInterval`.** It returns the
   sentinels `(1,2)` for air, `(-2,-1)` for solid and `(-1,1)` for boundary
   (`crates/document/src/sketch/produce.rs:125`). These carry no distance. The sentinel
   algebra happens to be sound under the fold's min/max because it is sign-monotone, but it
   is fragile and undocumented — and it arms a specific trap: the natural implementation of
   outset is to shift the interval, which turns sketch *air* `(1,2)` into `(1−N, 2−N)` and
   classifies it **solid** for N ≥ 2. Silently wrong, no type error.
2. **`substrate::geom2d` is entirely predicates** — `orient2d`, `point_in_polygon`,
   `rectangle_inside_polygon`, exact by construction and citing Shewchuk. There is no
   distance function anywhere in the sketch stack. The whole profile path is boolean, not
   metric, which is why no field-level operation had a home.

The grill also raised whether the foundation should be **NURBS**, as production CAD chooses.
That question is settled here because the answer determines what a profile *is*.

## Decision

1. **Declare the field layer.** The pipeline is **Intent → Field → Occupancy → Display**.
   A node's meaning is a signed scalar field, negative inside; composition is field algebra;
   occupancy is derived by classifying that field over cells. **Display never reads the
   field** — ADR 0011's cached-brick raymarch stands unchanged, and per-frame cost stays
   independent of op-stack complexity. New geometric affordances attach at Field; new
   *structural* affordances attach at Intent.

2. **A profile is authored curves; its flattened polygon is its meaning.** A profile is a
   closed path of lines, arcs, Bézier segments and whatever curve kinds arrive later, in
   continuous coordinates, never required to align to the voxel lattice. It flattens to a
   polygon at a **fixed tolerance of 1/256 block**, and that polygon is what the document
   *means* — not an approximation of a truer meaning living in the control points. Field,
   classification, resolve and outset see only the polygon, so a new curve kind is purely
   additive at the authoring layer.

3. **The flattening tolerance is density-independent, and bounded density is what makes
   that possible.** Half a voxel at the maximum density of 64 is 1/128 block; the fixed
   1/256 is finer at every legal density, with one octave of headroom against a future
   bound raise. Consequently `SetDensity` re-voxelizes and never changes geometry — the
   units law ("density is fineness only") holds end to end. Because the polygon is the
   meaning, the flattening algorithm and its constant are **versioned document semantics**:
   changing either is a document migration, not a bug fix.

4. **Flattening is a substrate component with a soundness proof obligation.** Adaptive
   curve flattening is pure math with a literature (de Casteljau subdivision, flatness
   tests) and by ADR 0014's boundary law belongs in `crates/substrate`, not in `document`.

   **The theorem to prove is soundness, not fidelity.** Recorded because it was got wrong
   once during the grill and is easy to get wrong again: *chord tolerance below half a voxel
   does NOT imply occupancy identical to the exact curve.* Occupancy comes from an even-odd
   test at voxel centres, and a centre can lie arbitrarily close to the true boundary, so
   any positive tolerance can flip a bit. No tolerance fixes this. What is true, and what
   the architecture actually needs, is that **`cell_field_interval` never disagrees with
   `resolve_into` over the same flattening** — they consume one normal form, so they agree
   by construction, for any tolerance. Consistency is the obligation; absolute accuracy is
   not available and is not required.

5. **Every producer exposes a true field. The sentinels are retired.** `SketchSolid` gains
   a real signed distance — exact for extrude and full revolve, 1-Lipschitz-but-conservative
   for partial revolve, where the wedge clip is a `max` (the same conservative posture ADR
   0017 Decision 6 already takes). **It keeps its exact predicate.** `cell_field_interval`
   tries `rectangle_inside_polygon` first and returns a tight interval when it fires,
   falling back to a Lipschitz ball otherwise.

   This is deliberate and generalizes: **predicates are exact and answer classification;
   fields are approximate and answer geometry.** Both persist; neither replaces the other.
   Replacing the predicate with a Lipschitz bound would have cost real interior elision — a
   ball must cover a cube cell's corners, so it decides only when `|d(center)| > h√3`,
   whereas the predicate decides for any cell inside the profile however close to an edge.
   At 16³ that is roughly a six-voxel shell of interior that would have stopped eliding.

6. **A body's metric is derived on demand, never authored and never saved.** Exactness is a
   property of the data, not of a producer kind. **Outset's shape follows the body's
   category, never its edge angles:** boxes and every profile-lifted body outset **square**
   (all are polygonal once flattened, and every polygon admits an exact L∞ field as the
   minimum over edges of L∞ distance-to-segment); curved primitives — sphere, cylinder,
   torus — outset **round**, having no closed-form L∞ distance. Classification may
   opportunistically use a *tighter* metric than the authored geometry does: over a cube
   cell an L∞ ball is the cell itself, so the bound becomes `h` rather than `h√3` — a 42%
   thinner undecided shell wherever an exact L∞ field exists.

7. **Outset is a field combinator carried by any node.** It attaches beside the node's
   combine operation, so a leaf, a Group or an instance may all carry one and a composed
   cutter dilates as a whole. This needs no new machinery: the fold yields a field, so
   `d − N` is meaningful at every level. A negative outset (inset) shrinks. Outset is a
   **Measurement**, not an integer voxel count. A group's metric is the weakest of its
   members', so a group mixing a box and a sphere outsets round.

8. **No NURBS, and the reason is the deliverable.** The output is a hard-quantized lattice;
   exactness beyond that resolution is unobservable in the shipped artifact. NURBS' entire
   value is metric exactness at arbitrary scale — CAD needs it because a non-rational spline
   cannot represent a circle exactly, because a model is a contract with a machine shop
   inspected to microns, and because STEP/IGES interchange demands it. None of those apply
   here, and the final voxelization destroys the benefit regardless. The boolean argument
   is decisive on its own: **NURBS booleans are exact surface-surface intersection and are
   the famously failure-prone part of CAD, while field booleans are `min`/`max` on a scalar
   and cannot fail.** Adopting NURBS would trade a boolean model that never fails for one
   that does — while pursuing a boolean feature set. Control points survive as live editable
   input (Decision 2), which keeps CAD's real virtue, parametric editability, without its
   representation, its topological naming problem, or its brittleness.

9. **The lattice replaces the constraint solver.** Sketching is direct manipulation plus
   snapping to grid, edges and axes. Quantization supplies axis-alignment, equal lengths and
   coincidence as a by-product, so the profile layer carries no constraint entities, no
   solver, and none of the over-constrained or flipped-solution failure states. **Snapping
   is an input aid only:** what is stored is a Measurement in block-relative units, never a
   voxel index, so a density change cannot move an authored point (ADR 0008's carried-frame
   discipline; the units law).

10. **Slice mode is rejected pending a concrete use case; the boolean sugar is deferred.**

    Slice yields two bodies, and a field is a single scalar function — so slice is not a
    field operation at all but an **Intent-layer topology-and-identity** question. It
    collides with *junctions are parts* and no-operand-targeting not by coincidence but
    because it operates at the wrong layer; any version of it would first have to answer who
    owns the manufactured second body, and no field math will answer that.

    The owner's ruling goes further than deferral: **there is no evident value in it here.**
    Slice earns its keep in mesh modeling because generating the split geometry by hand is
    tedious — the boolean's value is that it *produces* geometry. In a parametric planner
    authoring two parts costs about what configuring a slice would, and lands in the model
    the fold already understands, with both bodies named and owned. The feature solves a
    problem this representation does not have. **Reopen only when a user surfaces a
    well-reasoned argument from a concrete case**, not on grounds of prior-art parity.

    Note that slice's genuinely useful half is already captured: its **Inset** parameter —
    the deliberate gap between the resulting pieces — is a negative outset under Decision 7.

    Boolean Extrude / Trim / Cut Groove remain **deferred, not rejected**: they are
    Intent-layer sugar expanding into field combinators — the primitives-as-sugar pattern,
    now with a precise meaning for "sugar over what."

## Considered options

- **NURBS / B-rep foundation**: rejected per Decision 8. Recorded rather than omitted
  because it is the obvious question to re-raise and the answer is structural.
- **Replacing `SketchSolid`'s exact predicate with the Lipschitz bound**: rejected per
  Decision 5. It would have unified the classifier at the cost of levelling the better
  producer down to the weaker one, losing measurable interior elision.
- **Density-derived flattening tolerance**: rejected. Combined with a definitional normal
  form it would make `SetDensity` mutate geometry, violating the units law. The fixed
  constant is only viable because density is bounded `1..=64`; the bound was adopted for
  bit-packing reasons and pays a second dividend here.
- **The true curve as the meaning, polygon as implementation detail**: rejected. It leaves
  every downstream exactness claim carrying an ε-band caveat and makes "exact occupancy"
  permanently unachievable, in exchange for freedom to change flattening — freedom that
  versioning already provides at a schedule we control.
- **Per-body metric split on rectilinear-vs-diagonal**: rejected. Filleting one corner, or
  rotating one edge by 1°, would flip a cutter's outset from square to round — an
  imperceptible data change producing a visible authoring change. The category split of
  Decision 6 follows a distinction the author can see and predict.
- **Exact predicates on curves instead of flattening**: not adopted. Even-odd via exact
  ray-curve crossing counts is tractable for Béziers even though distance is not, and it
  would make occupancy genuinely exact. Rejected for now as more machinery than the
  quantized deliverable justifies; recorded as the compatible retrofit if flattening's
  versioning burden ever bites.
- **A CAD-style constraint solver in sketches**: rejected per Decision 9.
- **Leaf-only outset**: rejected. A reusable composed cutter could not be given clearance
  without editing its internals — exactly the case a definition-based cutter exists for.

## Consequences

- **`substrate::geom2d` grows a metric vocabulary beside its exact predicates** — L2 and
  L∞ distance-to-segment, and adaptive curve flattening. This changes the module's
  numerical character: its existing predicates are exact by construction, and distance
  cannot be. Decision 5's "predicates classify, fields measure" split is what keeps the two
  from contaminating each other.
- **`src/cell_interval_parity_tests.rs` is the gate for all of it.** Every change here —
  the sketch SDF, the tightened L∞ bound, outset — alters cell classification, and the
  fuzzer already asserts classification never disagrees with per-voxel resolve. The
  Decision 4 soundness theorem is the proof-tier version of what that gate tests.
- **Interior elision improves wherever an exact L∞ field exists**, via the `h` rather than
  `h√3` bound. Boxes and rectilinear profile-lifted bodies are the common case in
  block work, so this should be measurable on real scenes.
- **Documents gain a flattening version.** Config needs no back-compat, but shared project
  documents do; the flattening algorithm and its 1/256 constant are part of what a saved
  document means.
- **`Node` gains an outset alongside `operation`**, and the fold must apply it before the
  node folds. Incremental invalidation must account for the dilated AABB: an outset cutter
  dirties a larger region than its undilated bounds, so the edit-broadphase BVH must be fed
  the outset bounds, not the producer bounds.
- **A group's outset shape depends on its contents** (Decision 7). Predictable — "a group
  is as round as its roundest member" — but it is a rule the UI must not hide.
- **`docs/architecture/` gains the four-layer picture when this ships.** The architecture
  set describes the living shape; this ADR is the delta.
- **The sketch→volume atom and the field lift are the same joint.** Extrude combines the
  profile field with a slab, revolve evaluates it in radius-and-axial terms, and the
  reserved Sweep arm carries it along a path. That the authoring atom and the field
  construction coincide is evidence the layering carves somewhere real. An open polyline
  has no inside and so no field — a path is the *parameter* of a sweep, never a body.
- **Self-intersecting and cusped profiles need no special handling.** Magnitude is
  distance-to-boundary and is 1-Lipschitz everywhere; the even-odd sign can only flip where
  that distance is zero, so the signed field stays continuous. The field framing demands no
  more well-formedness than today's even-odd test does. Recorded so the worry is not
  re-derived.
- **The ε-band error survives outset**, non-obviously: offsetting a flattened polygon is not
  the offset of the true curve, but Minkowski addition preserves Hausdorff distance, so the
  bound composes through the operation that motivated it.
