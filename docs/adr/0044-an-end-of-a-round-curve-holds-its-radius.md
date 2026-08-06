# ADR 0044 — An end of a round curve holds its radius

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

[ADR 0043](0043-a-snap-lets-go-gradually.md) removed a threshold and then said plainly what it had
not fixed: the far cap of a swept slot slid along its own arc by up to **2.7** for a cursor step of
0.005, while every radial coordinate in the same drawing stayed smooth to five figures. It named
that the bigger of the two instabilities and left it open.

The author tried the result and agreed — "it's still quite unstable" — and prescribed the fix in
one line: *"try making an arc slot endpoint roughly follow its radius."*

That turned out to be the cure for the wander, not merely a separate nicety, and the reason is
worth writing down. The wander was never noise in the arithmetic. When the radius is held exactly,
the whole rigid set travels as ONE similarity: nothing about the drawing has to change shape, so
the solve has nothing to reconcile and never reaches for a freedom to pay with. When the radius is
only partly held, the drawing must genuinely deform, the sweep is the cheapest thing to spend, and
which point along it the solve settles on depends on details of the iteration that two neighbouring
cursor positions do not share. Holding the quantity does not damp the wander. It removes the
question that produced it.

Two things were tried first and did not work, recorded so they are not tried again:

- **A slack measured as a share of the QUANTITY** (hold the radius while within 15% of it). It held
  beautifully — exact to a three-unit pull — and broke
  `pulling_a_slots_corner_across_its_rail_lets_the_snap_go`, because a purely radial pull is the
  author setting the radius and must be met exactly. A quantity-relative slack has thrown away the
  one thing that tells "along" from "across".
- **Widening the one cone for everything**, to 0.75. That broke
  `an_achievable_drag_lands_exactly_on_the_cursor` and
  `a_level_segment_stays_level_when_an_end_is_dragged`.

## Decision

**The cone is a property of the quantity being kept, not one number for the drawing.** Each
candidate the snap considers carries its own:

| the hand is keeping | cone | why |
| --- | --- | --- |
| the SPAN it stands at the end of | `SNAP_CONE_KEEPING_A_SPAN`, 0.25 | dragging an end IS how a segment's length is authored, so the end has to give it up readily — there is no other door |
| the RADIUS it stands at | `SNAP_CONE_KEEPING_A_RADIUS`, 0.75 | an arc's radius has its own door: dragging the arc's BODY offsets each end along its own outward direction, which is a change of radius and nothing else |

The distinction is the authoring grammar, not the geometry. An end of a round curve may hold its
radius hard precisely because the only thing left for that gesture to mean is a sweep. An end of a
segment may not, because the gesture means the length.

Direction is kept as the discriminator, which is what makes both true at once: the cone is a share
of the hand's TRAVEL, so `across / travel` is the sine of the angle between the gesture and the
circle it would keep. A tangential hand is deep in the plateau however far it goes; a radial hand
is outside every cone at any distance, and a corner pulled straight out still sets the radius
exactly.

## Consequences

**An arc slot's end follows its radius, and the rest of the slot stops moving.** Pulled outward
from a rail at radius 40, six units of travel:

| cursor pulled out to | end's radius | far cap |
| --- | --- | --- |
| 40.0 → 42.5 | 40.0000, exactly | 44.0000 at x = 0.0000, exactly |
| 43.0 | 40.3469 | 44.3466 |
| 44.0 | 42.4446 | 46.4460 |
| 46.0 | 46.3829 | 50.3895 |

Exact across a two-and-a-half unit pull, then given up smoothly through ADR 0043's falloff. Guarded
by `an_arc_slot_end_follows_its_radius`.

**The wander ADR 0043 left open is closed.** The far cap's slide across a cursor step of 0.005:

| | ADR 0043 | now |
| --- | --- | --- |
| far cap wander | 2.7 | **2.8e-25** |

`the_free_sweep_of_a_slot_is_still_spent_arbitrarily` was written to assert the wander was STILL
above 1.0, so that a fix would be recognized by watching it fail. It failed on the first run of
this change and has been replaced by `the_free_sweep_of_a_slot_is_no_longer_spent_arbitrarily`,
which asserts the opposite.

**The three-pass structure is untouched, and so is the preference pass.** It was tempting to read
the wander as a preference that needed strengthening — the radius already has a `ScalarHold` row,
and it holds only in pass one, which seeds. That reading would have led to weighting, which
[ADR 0042](0042-a-gesture-states-its-own-rigid-set.md) established is a mechanism no other solver
has. The fix instead makes the question easier, which is what planegcs does by starting from the
current geometry and what D-Cubed does by taking rigid sets. Least motion stays emergent.

**Still not done.** A fixed snap tolerance in screen pixels — the author asked for a generous one —
remains unbuilt. It needs a length computed by the shell from the camera and passed down, since
`parametric` has no camera. ADR 0043's measurements say it bounds the correction rather than curing
anything, and the cure has now arrived by another route, so it is a comfort feature rather than a
fix. The `SNAP_CONE_KEEPING_A_SPAN` case has had no equivalent attention: nothing has measured
whether a segment's end wants a wider cone than 0.25, only that 0.75 is too wide.
