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

**Reaching every arc-like end, which this decision at first did not.** The author used the result
and reported the gap: "the circle ghost and snapping should apply to any arc-like endpoint". A drag
whose reachable part has NO standing relation short-circuits in `settle_under_the_hands` — nothing
to trade the pull against means the hands are the answer — and the snap was skipped along with the
solve. So the simplest arc there is, drawn on an empty plane, was the one place an end followed the
cursor freely and no ghost ever appeared: radius 40.4, 41.4, 42.4, 43.4 as the hand went out.

A snap is geometry, not a relation. `Problem::snap_the_hands` answers it without solving anything,
and the short-circuit asks for it before writing the hands down. A bare arc's end now holds ~40
across the same pull and reports its quantity, so the ghost draws. It holds *roughly* rather than
exactly because that arc's center is derived from its two ends, so it shifts with them and the
radius is measured against a center that has itself moved — which is what the author asked for in
the first place: **roughly** follow its radius.

**Still not done.** A fixed snap tolerance in screen pixels — the author asked for a generous one —
remains unbuilt. It needs a length computed by the shell from the camera and passed down, since
`parametric` has no camera. ADR 0043's measurements say it bounds the correction rather than curing
anything, and the cure has now arrived by another route, so it is a comfort feature rather than a
fix. The `SNAP_CONE_KEEPING_A_SPAN` case has had no equivalent attention: nothing has measured
whether a segment's end wants a wider cone than 0.25, only that 0.75 is too wide.

**The ghost named a circle the arc was never on, and this decision's own explanation of why was
wrong.** The section above closed with "it holds *roughly* rather than exactly because that arc's
center is derived from its two ends, so it shifts with them". That is not what an arc's center is.
[ADR 0038](0038-a-point-is-placed-never-computed.md) made it AUTHORED, and
`Sketch::seat_arc_centers` only takes away the component running along the chord — the center keeps
its one real freedom, how far out along the bisector it stands.

The actual fault was in when the quantity is measured. A snap measures to a PIVOT, and a pivot is
rarely a hand: `was`, the record of where the gesture started, carried only the points under the
hand, so the kernel fell back to where the pivot stands *now*. By then the caller has written the
raw cursor into the drawing and re-seated the arc centers on top of it. The radius was therefore
measured to a center the gesture had already dragged — and the ghost drew that.

The author saw it: *"the ghost circle doesn't correctly follow the arc. it's the same center point
and radius as the arc so I'm confused."* Measured on a bare arc, pulling the end out:

| pulled out | the ghost said | where the arc actually was | apart by |
| --- | --- | --- | --- |
| 0.5 | 39.45 about `[0.55, -0.46]` | 39.98 about `[0.00, 0.00]` | 0.72 |
| 1.5 | 38.87 about `[1.14, -0.93]` | 39.89 about `[-0.06, 0.11]` | 1.47 |
| 2.5 | 38.29 about `[1.74, -1.39]` | 39.87 about `[0.00, 0.13]` | 2.21 |

The ghost drifted further every frame while the arc stayed put, which is precisely the drawing the
author was looking at.

`was` now carries the WHOLE pre-drag drawing rather than the hands alone.
`Sketch::point_move_attempt` already held it — `before`, cloned for
[`carry_authored_handles`](../../crates/document/src/sketch/mod.rs) — and was narrowing it on the
way down. The prepared problem is scoped to the part of the plane a drag can reach, so a point it
does not carry is dropped rather than refused; refusing would have failed every snap on a plane
with a second shape on it.

| | before | after |
| --- | --- | --- |
| the ghost, over a two-and-a-half unit pull | drifted 39.45 → 38.29, center wandering to `[1.74, -1.39]` | **40.0000 about `[0, 0]`, every frame** |
| the arc's own radius at the far end of that pull | 39.87 | 39.93 |
| ghost against arc | 2.21 apart | **0.09 apart** |

Guarded by `the_ghost_names_the_circle_the_arc_is_on`. It also fixes something nothing was watching:
`Rigidity::Preferred`'s `opening` is documented as "the drawing as the gesture FOUND it" and was
being handed the same narrowed record, so every non-hand point it priced was priced off the bent
drawing.

**An arc deformed near a whole turn, because its center was seated on the raw cursor.** With the
ghost fixed, the author swept an end the long way round: *"towards the end of the full 360, it tends
to deform and the radius won't stay consistent; the center point ends up moving."*

The hand was landing on its radius correctly the whole time. It was the arc that ran away from it.
`Sketch::point_move_attempt` writes the raw cursor into the drawing before the settle and then calls
`sync_derived_points`, which seats every arc center back onto its chord's bisector. The seat is a
projection, and a projection is lossy in exactly the case that matters here: as an arc's two ends
close up the chord shortens until the bisector is nearly parallel to the correction, so a small
cursor error throws the center a long way, and projecting THAT onto the corrected bisector after the
snap cannot recover where it started.

| stage, at a chord of 10 with the cursor three units proud | the end | the center |
| --- | --- | --- |
| as the gesture found it | `[40, 0]` | `[0, 0]` |
| the raw cursor written down | `[-11.13, 41.53]` | `[0, 0]` |
| seated on the raw cursor | `[-11.13, 41.53]` | **`[-10.98, 1.51]`** |
| the snap lands the end on its radius | `[-10.35, 38.64]` | `[-0.38, 2.91]` |

Three units of cursor threw the center eleven, and the arc came out at radius 37.09 instead of 40.

**The cursor is scaffolding, and nothing should be derived from it.** The pre-settle sync now carries
the tangent arms only; the arc centers are seated once, at the end of the settle, from the position
the author actually gave them. Left alone the authored center is already on the bisector as soon as
the snap has held the radius, so the seat moves it not at all. Swept a whole turn with three units
of cursor error at every step:

| chord between the ends | radius before | radius now |
| --- | --- | --- |
| 30.6 | 39.687 | **40.0000** |
| 20.7 | 39.259 | **40.0000** |
| 10.4 | 37.093 | **40.0000** |
| 0.0 | 1.500 | **40.0000** |

The center holds the origin to 1e-6 throughout. Guarded by
`an_arc_keeps_its_circle_around_a_whole_turn`, which starts one step in: a hand that has not swept at
all is pulling straight out, and that IS the author setting the radius, so the center must move.

## Amendment, 2026-08-06 — what the two shares come to in degrees

This record shipped `SNAP_CONE_KEEPING_A_SPAN = 0.25` against `SNAP_CONE_KEEPING_A_RADIUS = 0.75`
with a reason for the ORDERING and nothing behind either number. The ordering still stands and the
reason is unchanged. The numbers now have a measurement, and it says two things worth recording.

**A cone is not a fixed number of degrees.** `across` is measured to the LOCUS, and a straight line
tangent to a circle of radius R leaves it by about `travel² / 2R`, while the cone grows only
linearly in travel. So the angle a gesture may be struck at and still be read as moving ALONG the
quantity narrows the longer the gesture commits — and a hand that FOLLOWS the locus keeps `across`
at zero and is held however far it goes. On a radius of 40, grabbing the same point either way:

| travel | radius held exactly within | span held exactly within |
| --- | --- | --- |
| 2 | 25.5° | 7.2° |
| 6 | 23.0° | 4.4° |
| 10 | 20.5° | 1.6° |
| 15 | 17.5° | — |
| 30 | 8.7° | — |

The span column is the finding. Past a travel of about 15 on a span of 40 there is no angle at all
that holds the length exactly — a segment cannot be rotated about its far end by any real gesture
without changing its length. That is the intended DIRECTION of the decision taken further than it
was ever measured to go. The doc for the constant said "about fifteen degrees", which was never
true of any gesture.

**0.25 is conservative, not forced.** Sweeping the share against the whole suite:

| share | what breaks |
| --- | --- |
| 0.25, 0.35, 0.40 | nothing |
| 0.45, 0.48 | `mirror_regenerates_after_source_moves_and_adds_no_authored_geometry` |
| 0.60, 0.75 | that, plus `a_level_segment_stays_level_when_an_end_is_dragged` and `an_achievable_drag_lands_exactly_on_the_cursor` |

At 0.40 a span holds exactly within 12.5° on a small nudge and still 3.6° at a travel of 15. The
value is left at 0.25 because widening it is a question about FEEL and the author has not had their
hands on it; the measurement is recorded so the choice is a minute's work rather than a study.
Guarded by `the_two_snap_cones_are_the_angles_they_are_measured_to_be`, whose bounds are loose on
purpose — it exists so a change to either share cannot pass unnoticed, not to claim these are the
right angles.

## Amendment, 2026-08-16 — only a curve that draws the circle offers it

**`SNAP_CONE_KEEPING_A_SPAN` is deleted, and with it the whole span candidate.** The row above that
gives a segment's end a cone of 0.25 is reversed. A hand standing at a segment's end is now offered
nothing; only a round curve's end offers anything, and the surviving constant is renamed to say so
in the singular.

The reason is the one this record already gave for the ordering, followed to its end. A cone was
justified by "the author is moving ALONG the quantity", and along WHAT is the question. An arc's
end slides along the circle its own arc draws: the locus is on the screen, the author can see it,
and holding it is holding something the drawing has. A segment draws a line. The circle about its
far end is drawn by nothing, exists nowhere in the drawing, and a hand pulled onto it is a hand
pulled onto geometry that is not there. [ADR 0040](0040-a-drag-snaps-to-the-quantity-it-moves-along.md)
says the hand snaps onto the circle its CURVE draws; the span candidate was manufacturing a circle
no curve drew.

**What forced it was a rectangle.** The author: *"I'm seeing arc radius snapping trigger even when
there is no arc or circle to snap on. For example, in the repro dump there is a rectangle. Selecting
one of its points causes it to treat another point on the rectangle as an arc with a radius."*

The amendment above argued the span cone self-extinguishes: past a travel of about 15 there is no
angle that holds a length exactly, because a straight gesture leaves the locus quadratically. That
argument assumes the gesture is free. A rectangle's horizontals and verticals hold a dragged corner
*exactly* tangential to the circle about its neighbour, so `across` never grows at all and the cone
that lets go of every free gesture engages permanently. On the author's own drawing, dragging a
corner 20 units in each of 24 directions:

| direction | corner missed the cursor by | ring drawn about |
| --- | --- | --- |
| 0°, 180° | 2.42, 2.39 | the corner above it |
| 90°, 270° | 1.72, 1.71 | the corner beside it |
| the other 20 | 0.00000 | nothing |

The four that miss are the four that run along an edge. So the branch was not rare on the most
common shape in the application, and where it fired it fought the very constraints that made it
fire — the miss is the solve reconciling the snap against Horizontal and Vertical.

**The suite never defended it, in either direction.** With the branch switched off, 588 of 589
document tests pass; the one failure is
`the_two_snap_cones_are_the_angles_they_are_measured_to_be`, which exists to measure the constants
against each other. Every other test named around the span — `an_achievable_drag_lands_exactly_on_the_cursor`,
`a_level_segment_stays_level_when_an_end_is_dragged`, the mirror test in the 0.45 row above —
constrains the cone from ABOVE. Nothing anywhere asserted that a span snap should ever fire, and
the "Still not done" note admitted no benefit had ever been measured for one.

The reversal is a two-line change with a longer tail:

- `Problem::quantities_a_hand_could_keep` offers arc centers only.
- `the_two_snap_cones_are_the_angles_they_are_measured_to_be` becomes
  `a_radius_is_held_within_the_angle_it_is_measured_to_be`, and the span becomes its CONTROL: a
  segment of the same length, grabbed at the same place, keeping the same circle as a locus, must
  NOT hold — at travels 2, 10 and 30, struck straight along the locus, which is the friendliest
  gesture there is.
- `Segment::scaffolding` and `Problem::add_scaffolding_segment` are deleted. They existed only to
  withhold a scaffold's span from this snap — an earlier report of the same fault, one size
  smaller, answered by narrowing the branch instead of removing it. `a_scaffold_span_offers_no_quantity_to_hold`
  survives as a guard and now says why.
- `dragging_a_rectangles_corner_resizes_it_rather_than_moving_it` gains two axis-aligned cursors and
  an assertion that no quantity is kept. Its three oblique cursors never triggered the branch, which
  is why the test was green while the application was not.

The gesture a span-keep might have served — rotating a fixed-length link about its far end — is
served by authoring a length dimension, after which the length is a hard row and no snap is
involved.

## Amendment, 2026-08-16 — a pivot is read from the opening, again

The section above titled "The ghost named a circle the arc was never on" recorded this fault and
fixed it once: a pivot is not a hand, `was` carries only hands, so `Problem::stood_of` fell through
to where the pivot stands NOW. That fix landed at the caller that prepares a drag. It did not hold,
because a WALKED drag re-enters the kernel once per substep with the drawing the walk has reached.

`Problem::drag_together` states the law four lines above where it hands `origin` to every step:
*"Every step measures from where the GESTURE started, not from where the last step landed… A step
that measures from the last one snaps to whatever that step settled at, so the quantity it is meant
to be keeping drifts a little each time."* The hand obeyed it. The pivot did not.

It ratchets. Dragging a rectangle corner 20 units, over the sixteen substeps of one gesture:

| substep | pivot read at | the span being kept |
| --- | --- | --- |
| 1 | −90.053 | 72.5185 |
| 2 | −88.800 | 72.5458 |
| 8 | −80.017 | 73.1943 |
| 16 | −71.218 | **74.5786** |

The ring was drawn about a corner that had slid nineteen units, at a radius nobody authored.

`drag_one_frame` already receives `opening` — "the drawing as the gesture FOUND it" — for exactly
this purpose, and the preference pass beside it already reads spans and radii out of it. The snap
kernel was the one read that did not. It now takes `opening` in place of the walked positions, and
its third parameter is named for what it is.

This survives the deletion above, because an arc's center is a pivot and is measured through the
same fallback. Measured on the same rectangle before the span candidate was removed, the fix alone
made the ghost name a real corner at a stable radius — and made the branch fire on MORE directions
rather than fewer, because an honest quantity engages where a ratcheting one had drifted out of its
own cone. A correctness fix is not a behaviour fix; both were needed.
