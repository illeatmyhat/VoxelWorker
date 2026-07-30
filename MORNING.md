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

**Check it** (~15 s): `cargo test -p substrate nonlinear` — expect **11 passed**, including
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

1. **Kani harnesses** for `curve_intersection`, `deepest_interior_point`, and the solver. The
   first two are plausible targets; a float solver is not.
2. **Nothing is wired to Slice E.** There are no constraint entities yet, so the solver core has
   no caller. That is the next slice, not an omission from this one.
3. **The inspector still reads "Custom profile (N points)"** for a circle — it counts document
   points, and a circle has one (its centre). Cosmetic; belongs to the tool-suite UI slice.
4. `shots/morning/` is **gitignored**, so the three PNGs are on disk only and not in the commits.

---

## The lag you reported · FIXED · `e11aafd`

You said adding lines had gone laggy. It had, and not where this file first guessed.

Measured, on a 128×96×16 sketch field (a rectangle, a pocket, a circle, extruded):

| path | before | after |
| --- | --- | --- |
| the solid's own resolve | 2 ms | 2 ms |
| **one `signed_distance` per voxel** — what a composite/boolean fold does | **1315 ms** | **46 ms** |

`SketchSolid::signed_distance` ran the whole arrangement on every call, twice — once directly and
once through `profile_bounds`. Rasterising a sketch on its own never touched that path, which is
why the resolve looked fine and anything folded with it did not.

`Sketch` now carries a `RegionMemo`. It validates by **comparing the entity store the region was
derived from**, not by a flag every mutator has to remember to clear — the failure mode of option
A below is a stale-geometry bug that does not look like a cache bug, and this makes that
unreachable rather than merely unlikely. The cell is skipped by serde, clones empty, compares
equal, and is boxed, so `Sketch` grows by two words and the document is unchanged.
`crates/document/src/sketch/tests/region_memo.rs` pins one test per way the store can move.

**Check it** (~40 s): `cargo test -p document region_memo` — expect **6 passed**.

**Decisions you might disagree with**

1. This is **option A**, which I had leant away from, with the discipline it needed replaced by a
   comparison. I took it over B because B fixes the resolve path only, and the measurement says
   the resolve path was never the slow one.
2. The cache compares entities rather than hashing them. `f32` is not `Hash`, and a comparison
   that a `NaN` fails is a miss — slow and correct, where a hash would have to invent an answer.
3. **`RevolveField` now borrows its region.** Owning a copy was the whole per-sample cost once
   the derive was cached, so the type grew a lifetime.

---

## The drawing moved when you drew past its fill · FIXED · `985ffe6`

Separate bug, same session. The overlay frame anchored on the bbox over the sketch's real
**points**; the resolve anchors the solid on the **filled region's** bbox-min. Equal only while
every point sits on the filled boundary — so a line reaching past the fill moved one anchor and
not the other, and the whole drawing walked. From your dump: points-min `-42 → -72` while the
resolve's anchor held at `-42`. Thirty voxels.

Nothing was cancelling it: `anchor_preserving_offset` runs on every profile edit, but it corrects
for a change in the *resolve's* anchor, which had not moved.

The overlay now anchors where the resolve does — one anchor, so a handle is on the solid by
construction. Both new tests fail under the old anchor.

**Check it** (~40 s): `cargo test -p document sketch_handles` — expect **9 passed**.

---

## Decisions settled

- **The unpick MIGRATES** when its face is cut in two (your call, 2026-07-30). Already the shipped
  behaviour and pinned by `cutting_an_unpicked_face_in_two_migrates_the_unpick`; nothing to change.

## What ADR 0035 still owes

1. **Slice E has no caller.** The solver core is real and tested and drives nothing. The smallest
   useful slice: constraint entities (coincident, horizontal/vertical, distance), a residual system
   built from them, and `SolveReport`'s DOF surfaced in the UI. This is the ADR's whole point.
2. **The expression text parser.** The AST and evaluator ship; nothing parses `2*width + 3mm`.
   Belongs to the parameters panel.
3. **Kani harnesses** for `curve_intersection` and `deepest_interior_point`.
4. **The inspector reads "Custom profile (1 points)"** for a circle. Cosmetic.
