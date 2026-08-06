# ADR 0045 — A snap reaches only as far as the shell allows

- **Status:** Accepted
- **Date:** 2026-08-05

## Context

The snap's cone is a share of how far the hand TRAVELLED
([ADR 0043](0043-a-snap-lets-go-gradually.md), [ADR 0044](0044-an-end-of-a-round-curve-holds-its-radius.md)).
Read as an angle that is exactly right: it asks whether the gesture is going ALONG the quantity or
across it, and the answer should not depend on how far the author dragged.

It is also, quietly, already screen-relative. Travel is read from the cursor, so a hundred-pixel
gesture is a hundred-pixel gesture at every zoom, and the cone it opens covers the same patch of
screen whatever the drawing's scale. What the cone does NOT have is a length. An angle cannot say
how far from the cursor the drawing may end up, so a long sweep opens a wide cone and a wide cone
can hold the drawing a long way from where the author is pointing, with nothing to stop it.

Every CAD tool that snaps states a tolerance in screen pixels for this reason — OCAD defaults to
five, cloud CAD computes its own against an off-screen buffer — because pixels are the unit the
author's patience is actually measured in. The author asked for the same thing, twice: *"well
screen pixel limits sound good. is there a reason not to do it?"* and *"I'd like a fairly generous
limit."*

The honest caveat, recorded here because it was measured before the fact and should not be
forgotten: **this is a comfort feature, not the cure.** ADR 0043 tried capping the cone at a fixed
length as a stability fix and found it made a long gesture WORSE — 8.96 against 7.44 — and the
instability the author was reporting was closed by another route entirely, by holding the quantity
so the rigid set travels as one similarity. A ceiling bounds how far a snap may take the drawing.
It does not make the drag smooth; that is already done.

## Decision

**`SnapReach` — a ceiling on the cone, in the drawing's own units, set by the shell.**

`crates/parametric` has no camera and must not acquire one; the per-layer crate split keeps the
flow downward-only. So the layer that knows how big a pixel is converts, and the kernel is handed a
length:

| layer | what it holds |
| --- | --- |
| `ui::chrome::SKETCH_SNAP_REACH` | **90 egui points**, beside the other on-screen sizes |
| `src/windowed/render.rs` | points → drawing units, and the `SnapReach` |
| `document::Sketch::move_point_reporting_its_snap` | takes it and passes it down |
| `parametric::sketch::Problem::holding_a_snap_within` | applies it to the cone |

Three things make it safe to hand a number to:

- **It is a ceiling and only a ceiling.** `SnapReach::UNBOUNDED` is the kernel's own behaviour, and
  everything the kernel's own tests measure. A caller that sets one can narrow a snap; it can never
  invent one.
- **It narrows the cone rather than switching the snap off.** `min(share * travel, reach)` is still
  a cone, just a shorter one, so ADR 0043's falloff still arrives at the rim already faded to
  nothing — only sooner. This is the whole reason a threshold is admissible here at all.
- **A camera that degenerates loses the ceiling, not the snap.** `SnapReach::of_length` answers
  `UNBOUNDED` for anything that is not a positive finite length.

The shell measures the conversion by asking the SAME cursor-to-plane map one pixel to the right and
one pixel down, rather than deriving a scale from the camera a second time. That is exact under
perspective and on a tilted plane, and it cannot drift out of step with the map the drag itself
used — the two are one function. It takes the LARGER of the two steps, so a foreshortened plane
errs toward letting the snap hold.

Ninety points is deliberately generous. Three fifths of it is the plateau where the quantity holds
exactly and the rest is the falloff, so it is the whole band and not the yank; and at any ordinary
zoom the gesture's own cone is the narrower of the two, which means the ceiling normally does
nothing at all. That is the intent. It is there for the sweep long enough to be surprising.

## Consequences

**Tightening the ceiling gives the radius up on a slope, not in a step.** A slot's end at radius 40,
pulled to `[41.5, 6.0]` — 1.9 units off its own circle:

| ceiling allowed | radius the end lands on |
| --- | --- |
| unbounded | 40.000000 |
| 4.0 | 40.000000 |
| 3.0 | 40.064156 |
| 2.5 | 40.764706 |
| 2.0 | 41.883066 |
| 1.5 and below | 41.923001 — the raw cursor |

Guarded by `a_snap_reaches_no_further_than_the_shell_allows`.

**The ceiling does not bring the spring back.** Rocking the cursor a fiftieth of a unit at the width
where a ceiling of 2.0 is the thing letting go:

| | swing |
| --- | --- |
| the hard cone this file was written to catch | 3.79 |
| the bar the falloff has to clear | 0.25 |
| under a biting ceiling | **0.084** |

Guarded by `a_ceiling_does_not_bring_the_spring_back`.

**Only the door the shell drives carries one.** `move_point_reporting_its_snap` takes a `SnapReach`;
every other drag verb passes `UNBOUNDED`, because a body drag names no lead hand and so has no snap
to bound. `move_point` — the same gesture for a caller that does not want to draw the ghost — passes
`UNBOUNDED` too: a caller that cannot show the author what was kept has no business narrowing it.

**Still not measured.** `SNAP_CONE_KEEPING_A_SPAN` remains at 0.25 with nothing behind it but the
knowledge that 0.75 is too wide for a segment. And ninety points is a judgement, not a measurement:
what it wants is the author dragging something at a few zoom levels and saying whether it ever
bites when it should not.

## Amendment, 2026-08-06 — what ninety points decides, and that zoom cancels

This record closed with "ninety points is a judgement, not a measurement: what it wants is the
author dragging something at a few zoom levels." Half of that turns out to be answerable without
anyone's hands, and answering it changes what the remaining question is.

**Zoom cancels out.** The cone is `share × travel`, travel is read from the cursor, and the ceiling
is `90 × units-per-point`. Divide both by units-per-point and the comparison is
`share × travel_in_points` against `90` — no drawing scale left in it. So the ceiling engages at a
fixed gesture LENGTH:

| quantity | share | the drag that first reaches the ceiling |
| --- | --- | --- |
| a radius | 0.75 | **120 points** |
| a span | 0.25 | **360 points** |

Measured rather than only derived: `a_ceiling_in_screen_points_means_the_same_at_every_zoom` scales
a slot, its gesture and its ceiling together by four, at five ceilings spanning the whole slope from
"does nothing" to "gives the radius up entirely", and the answers agree to about a part in a
million — the solve's own convergence, not anything about scale.

The one place it is approximate is the shell's conversion, not the kernel's arithmetic. Under
perspective on a tilted plane, drawing-units-per-pixel varies across the screen and the shell
measures it once, at the cursor. That is the same one-sample assumption the drag itself makes, and
deliberately so — the two must not drift apart — but it does mean a long gesture across a steeply
foreshortened plane is bounded by the reach as measured where it started, not where it ended.

**So the open question is narrower than it was.** Not "is ninety points right at every zoom" — it is
the same everywhere. It is: *should a drag be allowed to travel 120 points before the radius snap
starts being reined in?* That is one number about one gesture, and it still wants the author's hands.
