# Sketch constraint solving — what moves when you assert something

What the author sees a constraint *do* to their drawing, and the measurements behind each
ruling. Worked out with the owner 2026-07-30 and 2026-07-31. The timeless shape lives in
`docs/architecture/01-document.md`; this file is the dated input behind it — the reports, the
numbers, and the alternatives that were measured and dropped.

The premise, in the owner's words: *"Applying constraints should avoid having a large 'blast
radius'. Pure translation of large groups of sketch entities is better than any changes that
result in changes in length, orientation, and area."*

## The four reports this answers

Each began as an owner bug report against a shipped behavior, and each turned out to be the
same question in a different costume: **when a constraint cannot be met without moving
something, what should move?**

| Report | What was happening | What answers it |
| --- | --- | --- |
| "I can't coincident a point to an arc's center" | The center is derived; the solver wrote a coordinate `sync_derived_points` overwrote on the next edit | Read the derived point as a function of the arc's ends |
| "I can't move the other end of the line any more" | The drag was a hard pin; a cursor off the one allowed line made the system unsatisfiable | The hand is a pull, resolved in two passes |
| "It ended up translating both of them towards a midpoint" | Rigidity made each piece move as one, but least squares split the gap by mass | Anchor the heavier piece out of the parameter vector |
| Blast radius (above) | Least travel drags the named point alone and deforms everything attached | Span-preserving rows per edge and axis |

## The measurements

**Mass alone is not "one translates to the other".** With rigidity but no anchor, a
four-corner quad joined to a two-point stick by `Coincident` splits the gap in inverse
proportion to point count — measured at quad 16.67, stick 33.33, gap 50, which is exactly
2:4. Directionally right, and not what was asked for.

**A soft anchor does not close it.** Holding the heavy piece with one unit-weight row per
point takes its travel from `d·n_B/(n_A+n_B)` to `d·n_B/(2n_A+n_B)` — for the quad and stick,
from a third of the way down to a fifth. Every value that would actually reach zero is a
number someone has to defend. Which piece is the reference is not a quantity, so it was made
a structural fact instead: the heavy piece's coordinates are dropped from the parameter
vector for the preference pass.

**Weight 1 needs no tuning once there are two passes.** Where a rigid motion satisfies the
constraint, both residual blocks reach zero simultaneously and the weight is irrelevant.
Where they genuinely conflict — leveling one edge of a closed quad cannot leave the other
three spans alone — the exactness pass re-solves the constraints alone, so rigidity can only
ever rank answers that satisfy them equally well. Verified: `Horizontal` on a quad edge still
levels to within 1e-9.

## The rulings, and what each rejected

**Per-axis spans, not per-edge lengths.** A length-only row leaves a connected piece free to
rotate about anything the constraints do not pin, at zero cost. A drawing that spins to meet
a constraint has moved much further than one that slides. The accepted cost is that rotation
across two pieces — `Parallel` between separate pieces — leans on the exactness pass rather
than being expressed as a preference.

**Only a strict winner anchors.** Pick order and id order were both available as tie-breaks
and both rejected: they are rules the author has to be taught rather than rules they can see
in the drawing. Two loose points still meet in the middle, as they always have.

**A fixed piece outranks any point count.** A piece something has already `Fix`ed is not
going to travel whatever its size. Without this, a four-corner quad would outweigh a single
pinned target point and the anchor would try to move the thing that cannot move.

**Rigidity during a drag is measured BEFORE the hand.** It answers "which configuration
would the author have chosen?", and a drag answers that too — the hand — so the two only
agree if they are asked about the same drawing. `move_point` writes the grabbed point to the
cursor *before* the settle runs, so a reference read from the live points has every span
through that point already stretched, and rigidity measured against it resists the author's
own gesture. Four drag tests caught that on the first wiring, and the first fix was to switch
rigidity off during a drag entirely. [ADR 0039](../adr/0039-a-preference-is-measured-before-the-hand.md)
replaced that with the reference the preference actually wanted: the problem carries `was`,
the pre-drag positions, and every span is measured against those.

**Deleting an edge takes the ends nothing else draws.** A line removed from a drawing used to
leave two dots the author never placed. A constraint is not a reason for a point to outlive
the geometry it was drawn for, so the cascade takes the constraint too. The point-delete
cascade is deliberately unchanged, which leaves a known asymmetry.

## Curve-intrinsic evaluation context

Arc sweep and circle radius are curve-owned scalars. A free radius is exact solver state; a fixed
radius is a `Measurement` source, not a cached voxel number. Geometry consumers therefore take an
explicit evaluation context carrying scene density. The region memo resolves fixed radii once per
logical derivation and keys the result by that context, so faces, bounds, field sampling, voxel
resolve, handles and feature edges borrow one resolved curve set. Dense field walks prepare that
view once, so sample callbacks do not re-enter the memo. `SetDensity` rescales free
radii by an exact integer ratio and leaves fixed sources untouched. This is Phase 0 authority
plumbing only; Tangent remains out of scope.

## Open

- **Rotation preference across pieces.** `Parallel` between two separate pieces must rotate
  one of them, and per-axis rigidity resists exactly that. The exactness pass produces a
  correct answer; whether it produces a *tidy* one is unmeasured.
- **Arcs are rigid by chord only.** The chord span is preserved and the sweep is stored, so an
  arc translates rigidly, but nothing expresses a preference against a sweep change.
