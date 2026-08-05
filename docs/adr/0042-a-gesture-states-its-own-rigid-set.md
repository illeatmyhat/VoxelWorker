# ADR 0042 — A gesture states its own rigid set

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

The author asked four times, in the same words, for the same thing: dragging an arc's center point
should translate all three points. Each time the answer was a new rule inside the solve, and each
time the arc's center still moved alone. The fourth time came with the question that mattered —
"Is something preventing you from doing this in the solver? Would it be easier if you had another
derived property? Or does this require a design change?" — and then with the verdict on the method:
"These functions you've been writing to tune the solver seem like the wrong approach. What do
Fusion and other CAD software do?"

They do not tune. Three findings, all of them load-bearing:

**Nobody weighs points against constraints.** The question "how do other solvers weigh regular
points against open points against control points against segment types against actual
constraints?" has a flat answer: they do not weigh them at all. planegcs, the solver under FreeCAD,
answers a drag by adding a *temporary* point-to-point constraint at `initMove()` and dropping it at
the end — "temporary constraints are only enforced so much that they don't conflict with other
constraints", and they do not reduce degrees of freedom. Least motion is not a term in the
objective; it is emergent, from starting the iteration at the drawing the author is looking at.

**Rigidity is declared per entity, not inferred per drag.** D-Cubed 2D DCM, the kernel under Fusion
and SolidWorks and NX, takes RIGID SETS: "collections of geometries which 2D DCM solves as if they
are constrained relative to each other without requiring the use of individual dimensions and
constraints." A spline is declared rigid, scalable or flexible. The caller says what moves as one
piece; the solver never works it out from the numbers.

**The gesture is named, not measured.** SolveSpace's interface distinguishes two things the author
can do with the mouse: "drag individual points (which leaves other points stationary and changes
the size) or drag the entire entity (which moves all points together while maintaining size and
rotation)." Two gestures. Not one gesture with an inferred meaning. And on the simplest case Fusion
is explicit: "if you drag the center point you will change the position of the arc like in a
circle."

Against that, what this codebase had was a fifth mechanism nobody else has: a preference pass whose
span rows, scalar holds and stays all competed with the constraint rows at weight 1, and a
`moving_hand` rule that told a pin from a lead by measuring how far each had travelled, against a
relative tolerance, because a pin is exact only in exact arithmetic. Everything downstream of that
tolerance was a guess about what the author meant.

The measurements had been saying so for a while. Sweeping the stay weight across three decades
changed nothing — lighter was worse, heavier converged to the same place. Both real wins of
[ADR 0041](0041-a-gesture-is-read-from-where-it-started.md) were topological, not numerical.

## Decision

**A drag states its hands and what each one is doing.** `Hand { point, to, role }` with
`HandRole::{Lead, Carried, Pin}`, carried from the document's producers through to the solver:

- **Lead** — the point the author has hold of. At most one. It is what the snap measures from,
  because the quantity being kept is the lead's.
- **Carried** — the rest of a rigid set, riding the same motion.
- **Pin** — held still for the duration, which is how a reshape names what it turns about.

`Problem::moving_hand` and the relative stillness tolerance are gone. `reshaping` is now `any hand
is a Pin`, which survives a settle, a snap and a walked step without a tolerance to protect it.

**A center is rigid with the curves it centers.** That is the whole of `rest_of_the_shape_held_by`,
and how far it reaches is a question the drawing answers:

| the center names | its rigid set |
| --- | --- |
| a whole shape (a slot's hub — both rails and the spine turn about it) | the shape, walked out through the relations holding it together |
| one curve of a bigger thing (a slot's cap) | that curve and its own points |
| a curve standing on its own (a lone arc) | that curve and its own points |

The rule used to be "a whole shape or nothing", and the *nothing* was doing all the damage. A bare
arc's center centers one curve and is concentric with nothing, so it was classified as an end cap
and carried nothing at all.

**The snap turns the whole rigid set.** Pulling only the lead back onto its circle and leaving what
it carries where a straight cursor delta put them tears the set apart on every step. The snap is
written as a similarity about the pivot — the same complex multiply that takes the lead's old
position to its snapped one takes every carried point with it — so the set keeps its shape exactly.
Pins are handed back untouched.

## Consequences

**A bare arc's center translates all three points**, by exactly the displacement asked for, which
is what the author asked for four times. Measured on a 90° arc, center pulled 5 voxels: all three
points move 5.000, and the sweep is unchanged to seven figures.

**An arc slot's sweep is exact rather than merely stable.** Measured on the curved slot of rails 4,
4, 36, 40 and 44, one spine end swept a full radian in twenty successive drags:

| | ADR 0041 | now |
| --- | --- | --- |
| swept end's radius | 40.00000 | 40.0000 every step |
| far end and hub | 1e-4 | exactly still, every step |
| rails after a full radian | ~5% wider per six voxels | 4, 4, 36, 40, 44 exactly |

The widening ADR 0041 recorded as "the freedom a slot keeps on purpose, which least-norm would
rather spend" was not that. It was the cap center running ahead of its own two corners while the
cap stretched to stay attached — a rigid set that was not being carried as one. Carrying it fixed
it, and the freedom is still there for a deliberate widening to spend.

**Four tests changed sides.** `dragging_a_center_changes_the_radius_and_nothing_else`,
`a_center_drag_projects_onto_the_bisector`, `a_center_dragged_into_the_bulge_makes_the_major_arc`
and `dragging_an_arc_center_moves_only_that_point` all asserted the derived-center model, where a
center had one freedom — how far out along the chord's perpendicular bisector it stood — and a drag
of it authored the sweep. [ADR 0038](0038-a-point-is-placed-never-computed.md) ended that model;
these were what survived it. A center drag now lands where it is put, both freedoms of it, and the
sweep is authored by dragging an END or the rim.

**The snap is now visible.** A drag reports the quantity it kept — `KeptQuantity { about, radius }`,
riding home on `Settled` and out through `Sketch::move_point_reporting_its_snap` — and the overlay
draws that circle dashed at the guide weight, the linetype already reserved for the thing a shape is
being derived from. A snap puts the point a little off the cursor, which from the outside is
indistinguishable from a solve that could not reach; the author said so plainly — "I can't really
tell if it's snapping." The cone is unchanged at a quarter of the gesture's travel. It was never
measured to be wrong, only invisible, and the ghost is the instrument for measuring it.

**What is not done.** The preference pass survives. It is still a mechanism no other solver has, and
the honest next step is to try deleting it and letting least motion be emergent from the initial
guess the way planegcs and SolveSpace do — but that is a separate change with its own measurements,
and this one is already load-bearing.
