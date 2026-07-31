# Growing the coordinate envelope — two options, neither built

The envelope itself — the three nested ceilings that bound where geometry can live — is
folded into `docs/architecture/data-structures.md` § *Wide-integer frames*. This file holds
only what is **not** built: the two ways to move those ceilings out, recorded so they are
not re-derived under pressure.

Neither is warranted yet. Each targets a different ceiling, and they are not alternatives —
B is strictly the larger change and does not subsume A's cheapness.

## Option A — field-local brick keys

*Fixes the single far object. Cheap.*

Key the brick records relative to the field's own minimum block instead of absolute zero,
carrying the wide integer base alongside the records the way every other artifact carries
its reference point. The 21-bit lanes then span the field's own extent rather than its
distance from the world origin, so a lone field placed anywhere in the document's range
renders — as long as **one field** spans under ~2M blocks. The mesh path is already
recenter-relative, so it follows for free.

Lifts the brick-key ceiling to document range for the single-object case. Does nothing for
the f32 ceiling: a wide *composite* still exceeds f32's reach from any single recenter.

## Option B — per-chunk camera-relative render frames

*The full fix. Needed only for wide composites.*

Rebase each chunk's origin against the eye in exact integers on the CPU, hand the GPU only
the small residual, and render each chunk in its own camera-relative frame. Precision is
then bounded by chunk size rather than by distance from any global origin, so a composite of
any span stays sharp — the f32 ceiling disappears rather than moving.

The hidden cost is not the rebase. It forces **sparse covering-set handling** in the
resolve: a 10M-block composite covers millions of mostly-empty chunks, and the resolve
cannot afford to walk them densely. That, not the frame arithmetic, is what makes this the
larger change.

Warranted once multi-object composites spanning more than ~2M blocks are a real
requirement. Until then the authoring bound makes the current edge an explicit error.
