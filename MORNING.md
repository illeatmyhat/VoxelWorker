# Morning report — ADR 0035, slices A–E

All five slices shipped and pushed. Four gates green on every commit.

**Look at this first** (10 seconds, no build):

```
shots/morning/slice-d-two-circles-three-regions.png
```

Two circles that share no point, the lens of overlap carved out. Two crescents with a gap
between them. That picture is impossible without the arrangement.

---

## Slice A — `crates/parametric` · SHIPPED · `9f66497`

Dimension/quantity algebra, an exact-rational expression AST + evaluator, and a symbol table
with `voxel_density` built in. `voxel_core::units` moved in wholesale.

**Check it** (~20 s):

```
cargo test -p parametric
```

Expect **47 passed**. Look for `adding_across_dimensions_is_refused` and
`a_cycle_is_caught_not_looped_on`.

**Decisions you might disagree with**

1. Kept `pub mod units` inside the crate, so paths read `parametric::units::Measurement`, not
   `parametric::Measurement`.
2. **Deferred the text parser.** Shipped the AST + evaluator + symbol table only. Typing
   `2*width + 3mm` into a box is the parameters-panel slice; nothing here parses a string.
3. Added `minus` / `negated` / `divided_by` to substrate's `Rational` rather than reimplementing
   them in `parametric`.
4. Added `Measurement::to_voxels_exact` alongside the rounding `to_voxels`.

---

## Slice B — closed curves · SHIPPED · `376c5a7`

`Circle { center, radius }` as a real entity. A closed curve is its own loop, anchored by a
centre, with no on-curve vertex (Decision 7).

**Check it** (no build — already rendered):

```
shots/morning/slice-b-circle-is-a-disc.png
shots/morning/slice-b-circle-in-a-square-is-a-donut.png
```

The second is a square with a **round** hole through it: one region, two kinds of boundary, no
conversion between them.

Or (~40 s, after a document build): `cargo test -p document circles` — expect **15 passed**.

**Decisions you might disagree with**

1. **I did not relax `arc_sweep_is_valid`**, which is what the goal's wording asked for. A 360°
   bulge is still refused for the endpoint-plus-bulge `Arc` form, because its chord shrinks to
   nothing and takes the circle it was meant to determine with it. Decision 7 says a closed curve
   is a `Circle`, so admitting an unsolvable arc would have been sparing a tool one branch at the
   store's expense. The real blocker was `ProfileEdge::interior_points` re-deriving the circle
   from that zero-length chord; that is what I fixed.
2. A circle's minted centre is `EntityRole::Construction`, so deleting the circle takes the
   centre with it — unless something else has since drawn to it.
3. A radius is a `SketchLength`, not a bare `f64`, so an authored "1 block" survives a density
   re-target.

---

## Slice C — curve–curve intersection · SHIPPED · `c05a72d`

`substrate::curve_intersection`: segment×segment, segment×arc, arc×arc, each crossing located by
**parameter on both curves** so the arrangement can cut at it. Coincident stretches report their
two ends rather than a bogus point.

**Check it** (~15 s): `cargo test -p substrate curve_intersection` — expect **25 passed**.

---

## Slice D — the arrangement · SHIPPED · `6193d5c` (cut) + `bd8316e` (faces + identity)

`faces::derive` no longer walks the graph of drawn entities. Every curve is cut at every crossing
with every other, the pieces are welded into arrangement vertices, and the DCEL walks *that*.

`FaceKey` stopped being the boundary's origin set and became **one point strictly inside the
face** — its deepest, from a new `geom2d::deepest_interior_point` (pole of inaccessibility,
measured to the curves, not to a flattened polygon).

**Check it** — the picture at the top of this file, then (~60 s):

```
cargo test -p document regions
```

Expect **13 passed**, including two tests whose meaning I inverted:

- `a_crossing_bounds_faces_with_no_snapped_point` — was
  `a_crossing_needs_a_snapped_point_to_bound_anything`. The bowtie is now two triangles.
- `cutting_an_unpicked_face_in_two_migrates_the_unpick` — was
  `restructuring_the_boundary_resets_the_face_to_picked`.

**Decisions you might disagree with**

1. **The unpick MIGRATES rather than resetting.** Cut an unpicked pocket in two and the carve
   follows whichever half still holds the stored point, instead of both halves reverting to
   picked. Decision 9 lists this as an accepted failure mode; I have now pinned it as *behaviour*
   in a test, which makes it a promise. If you want the old reset semantics this is the place to
   say so.
2. **A saved unpick from before tonight is lost.** `unpicked` (an origin-set list) is renamed to
   `unpicked_points`, so an older document loads with every face **picked** rather than failing.
   That is document state, not config — the loss is silent. Given no shipped users I took the
   no-migration route, but it is your call, and reverting it later is much harder than deciding
   now.
3. A face's identity point is searched against its own **ground**, not its outer loop — a face
   with others nested inside it is re-keyed with those as holes. Without it, the middle of three
   nested squares would name the innermost one. This costs a second pole search for nested faces
   only.
4. The identity precision is **half a voxel**, not a tenth. It is an identity, not a measurement;
   the search is deterministic either way, and it sits on the per-voxel resolve path.

---

## Slice E — the continuous solver core · SHIPPED · `955e217`

`substrate::nonlinear_least_squares`: Powell's Dog Leg with a trust region in parameter units,
Levenberg–Marquardt as the repair for the singular case, and a rank report.

**Check it** (~15 s): `cargo test -p substrate nonlinear` — expect **10 passed**, including
Rosenbrock's valley from `(-1.2, 1)`, the start plain Gauss-Newton diverges from.

The half you will actually see in the UI is `SolveReport`: `degrees_of_freedom` (how many ways
the drawing can still move) and `redundant_residuals` (how many constraints repeat themselves).
Both ways a sketch is wrong converge, so "solved" says nothing on its own.

**Decisions you might disagree with**

1. LM is a **fallback inside** Dog Leg, not an outer alternative. A rank-deficient Jacobian is the
   normal case for a sketch, not an exotic one.
2. The Jacobian is by **central** finite differences, not analytic derivatives and not forward
   differences. Analytic Jacobians are ~2× faster and are what SolveSpace does; they also mean
   every future constraint type has to hand-write and hand-verify its derivatives. Reversible
   later — the trait can grow an optional analytic hook.
3. A contradictory system settles on the **least-squares compromise** (the midpoint) rather than
   refusing to move. That is what makes "these constraints conflict" a diagnosis instead of a
   freeze.

---

## What I skipped, and why

1. **A region cache.** `SketchSolid::signed_distance` calls `sketch.region()` — and therefore
   `faces::derive()` — **once per voxel sample**. That was already true before tonight; the
   arrangement made each derive more expensive, so the `document` sketch suite went from **17.9 s
   to 28 s** (it was 78 s before I heap-ordered the pole search and coarsened its precision).
   Fixing it properly means interior mutability inside `Sketch` (a `OnceLock` region, cleared by
   every mutator) or a `Field` adapter that precomputes the region the way `RevolveField` already
   does. Both are real architecture in a document type and I was not going to add one while you
   were asleep. **This is the decision I need from you.**
2. **Kani harnesses** for `curve_intersection`, `deepest_interior_point`, and the solver. The
   first two are plausible targets; a float solver is not.
3. **Nothing is wired to Slice E.** There are no constraint entities yet, so the solver core has
   no caller. That is the next slice, not an omission from this one.
4. **The inspector still reads "Custom profile (N points)"** for a circle — it counts document
   points, and a circle has one (its centre). Cosmetic; belongs to the tool-suite UI slice.
5. `shots/morning/` is **gitignored**, so the three PNGs are on disk only and not in the commits.

---

## The one decision I need

**Do I add a region cache to `Sketch`?** The honest options:

- **A.** `#[serde(skip)] OnceLock<Vec<ProfileLoop>>` on `Sketch`, cleared by every `&mut` method.
  Fastest, and needs a hand-written `PartialEq` (a cache is not identity). Risk: one mutator that
  forgets to clear it is a stale-geometry bug that will not look like a cache bug.
- **B.** An `ExtrudeField` precomputed once per resolve, mirroring the `RevolveField` that already
  exists for the other operation. Symmetric with what is there, no interior mutability, and it
  only fixes the resolve path — the picking and overlay paths keep re-deriving.
- **C.** Leave it. 28 s of test suite and an unmeasured cost in the live app, paid down later when
  sculpt forces the issue anyway.

I lean **B**: it is the shape the codebase already chose for revolve, and it buys the hot path
without putting a cache inside a serialisable document type.
