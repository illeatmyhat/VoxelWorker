# Rendering SDFs in WGSL: measured, and cheaper than argued

Drag previews (`docs/design/direct-manipulation.md`) and a possible "see the SDF instead of the
voxels" viewer mode both want the same thing — parametric fields rendered analytically on the
GPU. Spiked 2026-07-20 in a worktree; the code is throwaway, this is the artifact.

**It was rejected on reasoning first, and the reasoning was wrong.** The argument was that
option A duplicates every shape's definition in WGSL and creates a drift risk this codebase has
been bitten by before. The duplication claim is the part that does not survive measurement, and
it is recorded here because it is exactly the kind of plausible objection that would otherwise
have quietly killed a viable approach.

## The duplication is one function, not one per shape

**61 lines of WGSL cover all five `ShapeKind` variants**, because the `Field` trait already
unifies them (ADR 0019) — the GPU side is one dispatcher over a kind tag plus a parameter block,
not a mirror per shape. All five compiled and rendered correctly on the **first attempt with
zero iteration on the math**: `glam::Vec3` → `vec3<f32>`, `.max()` → `max()`, `.length()` →
`length()`, essentially character for character.

| Piece | Lines |
| --- | --- |
| WGSL field math, all five shapes | **61** |
| WGSL pass scaffolding (fullscreen tri, camera ray, sphere-trace, shade, parity probe) | 126 |
| Rust renderer (≈270 pipeline cloned from `infinite_grid.rs`, ≈140 parity readback) | 408 |
| Wiring into the frame | 26 |

Only one shape needed tracer care: **Tube**, because `max(outer, −inner)` is not a true distance
field and a full sphere-trace step can tunnel the wall. Fixed with a 0.7 step relaxation — one
constant.

## Fidelity: zero voxels disagree, and the check is not vacuous

Two different questions, easily conflated:

**Value drift, WGSL against Rust**, over the producer's own sampling lattice: worst
`|gpu − cpu|` is 1.3e−5 at 2.1M samples, and **Box is bit-exact**. Pure f32 rounding-order noise.

**Zero-set placement** — does the analytic surface land on the voxel boundary?

> **0 voxels disagree. Every shape, every fixture, 2.1M samples on the largest.**

No half-voxel convention error, no density-scaling error, no frame error. The sensitivity was
measured rather than assumed, by perturbing the sample frame:

| Shape | +0.25 vx | +0.5 vx | +1.0 vx |
| --- | --- | --- | --- |
| Torus | 4.20% flip | 10.13% | 15.97% |
| Tube | 2.39% | 4.77% | 10.28% |
| Cylinder | 1.56% | 3.58% | 9.27% |
| Sphere | 2.20% | 4.24% | 8.03% |

A quarter-voxel frame error flips 1.5–4% of voxels; the check reports 0. It is a real assertion.
(A Box that fills its own bounding box is insensitive to sub-voxel shifts — a bad fixture. The
curved kinds carry the signal.)

## `SketchSolid`: the f64 objection does not survive numbers

The polygon producer was expected to be the wall. **Port size is ~150–200 lines of WGSL** —
`point_in_polygon`, `distance_point_to_segment` in both metrics, the polygon SDF, and two
operation wrappers. The coarse-cell machinery (`cell_field_interval` and friends) is not needed
for a point query.

The cited blocker is that `geom2d.rs` is f64 and WGSL has none. Measured, counting verdict flips
over the sampling lattice:

| profile | radius | offset | samples | flips |
| --- | --- | --- | --- | --- |
| n=256 | 400 | 0 | 648,025 | **0 (0.0000%)** |
| n=64 | 40 | 1,000,000 | 7,225 | **0 (0.0000%)** |
| n=64 | 40 | 100,000,000 | 7,225 | 854 (11.82%) |

**f32 is verdict-identical to f64 up to ~10⁶ voxels of offset**, collapsing only past f32's 2²⁴
exact-integer ceiling — the far-lands regime this codebase already solves everywhere else by
rebasing to a local frame in i64 before the downcast (ADR 0008). A preview uploads profile
vertices rebased to the shape's own origin, where f32 is exact.

The genuine `SketchSolid` risks are elsewhere, and two of them are real:

* **The `-0.0` boundary convention.** Integer vertices plus half-integer sample centres mean
  samples land *exactly* on edges, so the verdict lives in the **sign bit** — callers must use
  `is_sign_negative`, not `< 0.0`. A GPU evaluator must port the even-odd predicate as the
  authority and never infer occupancy from `d <= 0`. `solid.rs` documents a **shipped**
  platform-dependent bug from exactly this class (`atan2` vs `cos/sin` disagreeing by an ULP,
  dropping voxels on Linux but not Windows).
* **Unbounded vertex count.** Nothing caps `profile.len()`, and the polygon test is O(n)
  brute-force crossing-number with no acceleration structure. A 256-vertex profile at 1600×1000
  is 400M edge tests per frame. This is the one place `SketchSolid` is meaningfully harder than
  a primitive, and it is a *performance* problem, unmeasured.

## Drift is policeable, which was the real crux

A parity probe was built as a **second fragment entry point in the same shader module**,
rendering to `Rgba32Float` and read back with the existing `copy_texture_to_buffer` idiom — so it
shares the shader with the display pass **by construction and cannot test a stale copy**. No
compute pipeline is needed, which matters: there are zero compute pipelines in the repo.

Three clauses, and the first is the one that gets forgotten:

0. **The probe read the point it claims to read.** The shader returns its sample point alongside
   the value; the test rebuilds that point independently and requires agreement. Without this,
   the value diff silently compares two different questions and passes.
1. **Mirror drift** below 1e−4 at every lattice point.
2. **Zero-set placement** exact, zero mismatches, with the sensitivity figure recorded so the
   assertion cannot go vacuous.

## The frame is where the cost actually is

Both bugs hit during the spike were frame/wiring bugs; **neither was math**.

```
absolute         = shading_absolute + (recentre − half)     [the display frame law]
shading_absolute = world + half                             [the camera ray]
⟹ sample = world_point − (world_offset + grid/2 − recentre)
```

* **The half-voxel term is not zero.** `grid/2` is the *exact* half (half-integer on odd axes);
  `recentre` is the *floored* half. Assuming "the shape is at the origin" is correct only for
  even grids and silently half-voxel-shifts every odd one — which the sensitivity table prices
  at 4–10% of voxels.
* **A resolved region grid's indices are recentred.** `resolve_region` stores voxel `v` at index
  `v − recentre`, so a 32³ box spans `[−16, 15]`, not `[0, 32)`. Comparing producer-local
  indices against a recentred grid reported 87.5% disagreement for a *Box*.

The lesson for costing any future work here: **budget the frame work, not the SDF work.**

## What survives as a concern

* **Ongoing cost is real.** Every future producer is written twice, and `Sweep` is already a
  reserved arm in `SketchSolid::Operation`. The parity suite only covers shapes someone
  remembered to add a fixture for, so it needs a **`ShapeKind` exhaustiveness guard** that fails
  loudly when a variant lands without a WGSL arm — without it the suite silently tests four of
  six shapes and still looks green.
* **The CSG fold on GPU is unbuilt and unpriced.** The spike renders one leaf. A real preview
  composes Union/Subtract/Intersect — `min`/`max`/`max(-)`, but needing the whole scope walk
  uploaded.
* **Throughput is unmeasured.** Correctness was measured; frames per second were not.

## Estimate

**About a week to build properly** — the five primitives are the *day*; the week is
`SketchSolid`, multi-leaf frame plumbing, the CSG fold, the parity suite and the viewer-mode UI.
**~1 day per new producer thereafter**, most of it the parity fixture.

## Open

Whether the WGSL can be **generated from the Rust** rather than transliterated by hand. The
mapping is mechanical, and this repo already owns a precedent for the trick (the icon-parity work
transpiles Rust painters to SVG and diffs segment sets). If it works, drift becomes
*structurally impossible* rather than merely tested and the ongoing cost collapses. The decisive
question is not whether a generator can be written but **what it does when it meets Rust it
cannot handle** — silent mistranslation would be worse than hand-duplication, which at least
makes someone look at both sides.
