# Substrate — what was left behind, and what is left to prove

The extraction is finished: every structure whose identity is purely algorithmic now lives in
the substrate library, grouped by category so the taxonomy is visible at the call site. What
substrate is, the boundary law that admits a structure, and the naming rule are in
`docs/architecture/data-structures.md`. The tiers of machine-checked proof and how a property
is assigned to one are in `docs/architecture/05-proof.md`.

This file holds only the residue.

## The component hunt is closed

Three scans ran — the original survey, a second pass for well-known structures hiding as
private functions, and a third over the display, shell and interface files that the first two
never touched. The third yielded one extraction. **The yield is down to formulas and
deliberately-restrained kernels, so a fourth scan is not warranted**; a new candidate should
arrive from writing new code, not from re-reading old code.

## Cold, and waiting on a trigger

Two components are genuinely substrate but are not on any live path, so they stay where they
are until something uses them:

- The **palette minimum-bit codec** inside chunk storage — extract when the disk path goes
  live.
- The **spill cache** in the disk chunk store, which is a textbook least-recently-used cache —
  same trigger.

Extracting either now would move code no one runs into a library that claims to hold what the
system depends on.

## Surveyed and deliberately left in the domain

A register of restraint, so each is not re-proposed. Each was recognized as a real textbook
structure and left anyway:

- **The leaf spatial index.** Its identity is "must agree with the producer walk", which is a
  domain obligation, not an algorithmic one.
- **The chrome glyph rasterizer** — an edge-function rasterizer and standard compositing, both
  genuinely textbook, but they are the interior of a component that was already restrained.
- **One shape's exact distance function.** Clean on its own, but extracting it alone would
  split the family whose other members are approximate and belong to the domain.
- **The neighborhood dilation** in the mesh rebuild plan — a real name attached to a twelve-line
  loop whose function is the rebuild plan itself.
- **Asset-side helpers** — the relaxed-format normalizer, the block-type inverted index, the
  small image resample, the export framer. Single consumer each, or domain-baked scoring.

None of these introduces a dependency law, which is the only test that justifies a boundary.

## Still to prove

Two targets remain from the verification pass, both small:

- **The voxel-frame algebra.** The last algebraic-tier target: that carrying a frame with a
  value and consuming it in that frame composes correctly across the transforms the system
  performs. Same integer-and-order shape as the folds already proved, so it should not need a
  library of mathematics — attempt it core-only first.
- **The whole-cube expand-and-pack round trip.** The row-word kernel is proved; the cube loop
  needs a fixed edge to unwind, so the harness has to be anchored at concrete edges the way
  the others were.

Benches for the hot structures live beside the library and run on demand. They are never
commit gates — a bench that gates a commit becomes a flaky test on a machine with other work
on it.
