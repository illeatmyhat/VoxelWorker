# ADR 0046 — A badge takes a click, never a drag

- **Status:** Accepted. Amends [ADR 0035](0035-the-sketch-tool-suite.md) decision 3
- **Date:** 2026-08-06

## Context

[ADR 0035](0035-the-sketch-tool-suite.md) decision 3 settled two things about a constraint badge,
both of them right: it **beats the geometry under it** for a pick, because it is drawn over it; and
a **transform skips it**, because a constraint has no position and moving a label asserts nothing.

The shell read the second one one step too far. `begin_sketch_vertex_drag` asked
`sketch_entity_target_at` what was under the press and refused the whole gesture whenever the answer
was not `is_positional()` — that is, whenever it was a badge. The reasoning was sound in isolation:
a drag on a badge cannot mean anything, so do not start one. What it missed is that the badge is not
the only thing under the cursor. Refusing the *gesture* is not the same as refusing to *move the
badge*; it also refuses to move the vertex, the tangent lever and the curve the badge happens to be
covering.

The geometry makes this constant rather than occasional. A badge is a **32-point square** floating
**30 points** off the geometry it labels, and successive badges on one anchor step further along
that same offset. A vertex is grabbable within **10 points** and a curve body within **7**. So a
badge is a large opaque patch that by construction does *not* sit on its own anchor — it sits over
whatever else is thirty points away. On a drawing carrying many relations, which is the normal
state of a constrained drawing and exactly the state a slot is in, those patches tile the area the
author is trying to grab. The author's report: *"Dragging the arc slot is annoying because hit
detection keeps getting interrupted by the constraints."*

## Decision

**A badge takes a click. It never takes a drag.**

The refusal is deleted. `begin_sketch_vertex_drag` no longer consults the entity hit-test at all; it
goes straight to the grabs, and the grabs are positional by construction — a vertex, a tangent
lever, a curve body. There is nothing among them that a badge could have protected the author from
starting.

Deleting it rather than qualifying it is the point. The obvious repair was to keep the guard and
only apply it when nothing grabbable was underneath, so a badge over empty space still refused. That
is a guard that cannot ever change an answer: with nothing grabbable underneath, every grab already
misses and the function already returns `None`. A conditional that reads as a rule but decides
nothing is worse than no conditional, because the next reader has to prove it inert.

**The badge loses nothing.** Its click is armed on a separate path: the press sets
`sketch_select_press` before and independently of the drag, and a release that never left the press
by the drag threshold calls `resolve_sketch_selection_click`, which hit-tests afresh and still puts
the badge first. A click on a badge selects the badge whether or not a drag armed under it.

**A click is not a tiny drag**, which is what makes the two coexist. A press that has not crossed
the threshold has moved nothing: every arm of the drag preview is gated on `began`, and a commit
whose final producer equals its original records no command. So an armed-but-unmoved drag over a
badge is inert, and the click resolves on top of it.

## Consequences

**Hover and click still go to the badge; only the drag goes through it.** This is a deliberate
asymmetry, not an oversight. The badge is drawn on top, so it should answer for the pixel — but
"answer for the pixel" is a question about what the author is *pointing at*, and a drag is not
pointing, it is grabbing. The two questions have different right answers over the same pixel.

**The pick order in `sketch_entity_target_at` is untouched**, and so is the marquee. The marquee
anchor is gated on `sketch_drag.is_none()`, and the only presses that now arm a drag where they did
not before are presses with grabbable geometry under them — presses that never armed a marquee
anyway, because the same geometry answers `nearest_sketch_edge`.

**`SelectionTarget::is_positional` is left in place** with no production caller. It is the predicate
ADR 0035 decision 3 promised — the one place a future entity kind answers "does this have a place?"
— and the transforms it was written for are what should be asking it. That it was being asked by the
drag hit-test was the bug.

**Not covered by a test.** The hit-test lives in the shell, above the seam anything in the workspace
can construct: it needs a live camera, a laid-out badge set from the last overlay refresh, and a
window scale factor. What is checkable — that a badge is not positional, and that everything else is
— is already guarded in `crates/ui/src/panel/selection.rs`. The change itself was verified by
reading the two paths it depends on: that the click arms independently, and that an unmoved drag
commits nothing.
