# ADR 0008 — The voxel-frame invariant: carry or enforce the frame, never re-derive it

- **Status:** Accepted — the CARRY half still binds; the decode authority is retired (see
  "Amendment 2026-07-18" at the foot)
- **Date:** 2026-06-29
- **Relates to:** [ADR 0006](0006-authoring-truth-and-gpu-boundary.md) (the single Intent door is the
  enforcement point), [ADR 0007](0007-gpu-view-resolve.md) (the GPU resolve already carries its frame
  as an explicit `local_offset` — this ADR pulls the same discipline back onto the CPU side), and
  [ADR 0003](0003-foundation-rework.md) (the integer-indexed compact representation is the structural
  end-state where the invariant holds by construction).

## Context — a bug that was really a missing invariant

The per-chunk onion fog drew only ~1 of the 8 cloud puffs for a `DebugClouds` scene, while the cuboid
mesh of the **same** `VoxelGrid` drew all 8. The two consumers read the identical occupied-voxel list
and disagreed.

Root cause: a resolved voxel stores a floating-point `world_position`, but **nothing records what
frame that position is in.** Every consumer re-derives the integer voxel index from the float with its
own hard-coded assumption:

- the cuboid mesh / chunk store use `floor(world_position)` — a frame-*agnostic* absolute cell (the
  frame is carried *in* the position), so it is always self-consistent;
- the per-chunk fog uses `round(world_position + floor(dim/2) − 0.5)` and then **drops** any index
  outside `[0, dim)` — which bakes in the assumption *"this grid was recentred onto the origin by
  `floor(dim/2)`."*

That assumption holds for the five SDF shapes and the sketch solids (they resolve through
`Scene::resolve_region`, which recentres a placed composite by `(min+max)/2 = floor(dim/2)` for a lone
producer). It is **false** for a `DebugClouds` Part: a Part-only scene has no composite extent, so
`recentre_voxels_for_resolve` returns `[0,0,0]` and the grid stays corner-anchored at `[0, dim)`. The
fog's decode then yields `idx + floor(dim/2)`, and its `[0, dim)` bounds-drop discards everything with
any axis `idx ≥ ⌈dim/2⌉` — the whole field except the one corner octant. The `debug-clouds` golden
simply captured that broken half-empty fog, so the regression net never flagged it.

This is not a cloud bug. It is the general failure mode of **(c): producers legitimately differ in
their frame, the difference is not carried, and each consumer re-guesses it.**

## Decision — the invariant

Any value an Intent or producer emits that has a **frame, unit, or anchor** must resolve that context
in exactly one of two ways:

- **(a) One enforced convention** — there is no legitimate reason to differ, so a single convention is
  fixed and everyone obeys it (e.g. Z-up; "occupancy indices live in `[0, size)`").
- **(b) Carried explicitly** — producers legitimately differ, so the difference travels *with the
  data*, and consumers read it (never assume it).

The forbidden option is **(c): differ-but-don't-carry → re-derive.** Every spatial/quantitative Intent
must be auditable as (a) or (b); a (c) is a latent fog-bug.

This is already the rule behind the codebase's good decisions — it just was not named:

- **Offset / Size** carry the authored `Measurement` alongside the canonical voxels (b); that is why a
  density re-target is lossless. (The historical "density bug" was a (c): the authored frame was
  re-derived instead of retained.)
- **Far placement** rebases to the floating origin in `i64` *before* the `f32` downcast (ADR 0002 S4b)
  — carrying the origin rather than letting `f32` re-derive and lose it (b).
- **The GPU view-resolve** takes `local_offset` as an explicit uniform, never assuming `floor(dim/2)`
  (b). The CPU fog was the lone (c).

### Application to the resolve frame (this ADR's concrete fix)

Producers here *legitimately* differ: a placed Tool is recentred onto the origin (the renderer +
camera auto-frame want it centred), while a `DebugClouds` Part is **intentionally corner-anchored** at
`[0, region)` — a shipped decision with tests (`part_only_cloud_at_odd_density_drops_no_voxels`,
`mixed_tool_and_cloud_resolve_in_one_frame`). So this is a **(b): carry the frame**, not (a).

`VoxelGrid` now carries the integer `recentre_voxels` it was resolved with —
`(min+max)/2 = floor(dim/2)` for a placed composite, `[0,0,0]` for a corner-anchored Part-only grid.
`Scene::resolve_region` and `Store::resolve_region` record it; a bare `producer.resolve` leaves the
`[0,0,0]` default, which is correct (it emits corner-anchored). The fog — the lone (c) — decodes with
that carried value instead of a hard-coded `floor(dim/2)`. Because a centred grid's carried recentre
*is* `floor(dim/2)`, the decode and `world_origin` reduce to the historical formulas exactly, so the
five SDF shapes, the sketch solids and every non-cloud golden are **byte-identical**; only the cloud
fog changes (it stops dropping ~7/8 of the field — measured: 63 → 679 resident fog chunks).

### The structural guard — one decode authority ("the trait")

To stop (c) from re-appearing, the world→index decode lives in **one place** — `VoxelGrid::voxel_index_of`,
which reads the grid's own `recentre_voxels` — and consumers call it instead of re-inlining
`round(wp + floor(dim/2) − 0.5)`. The frame the decode needs now travels *on the grid it decodes*, so a
consumer cannot use the wrong one. Today there is a single grid type, so a method is the right
granularity; if more grid-like types appear it graduates to a trait. (The cuboid mesh's frame-agnostic
`floor(world)` absolute-cell decode is a *different, also-valid* view — it carries the frame implicitly
in the position — and is left as is; the invariant is satisfied either way, never by re-deriving.)

## Consequences

- **Positive:** the cloud fog renders all 8 puffs; the fog and mesh decode the same grid identically by
  construction; the `floor(dim/2)` formula has one home; the rule gives a concrete audit for every
  future spatial Intent.
- **Cost / change:** the `debug-clouds` golden is re-baselined once (it now shows the *correct* fully
  fogged field — a bug fix, not a rebaseline of working output). The five SDF shapes and the sketch
  solids are **byte-identical** (their recentre already equals `floor(dim/2)`), so every other golden
  is unchanged.
- **Where it pays off next:** the **sculpt-delta Intent** (ADR 0003 §3e, not yet built) carries sparse
  integer force-on/off addresses. Designing them as (a) "always producer-local integer" or (b)
  "carry their chunk/origin" — rather than letting the CPU apply and the GPU compositor (ADR 0007 P3)
  each re-guess — avoids re-running exactly this debugging session in a harder, sparser setting.
- **End-state:** once occupancy is the ADR 0003 integer-indexed compact representation rather than a
  float `world_position` everyone round-trips through a guessed decode, option (c) becomes
  *unrepresentable* — which is the real goal; this ADR is the convention that holds until then.

## Amendment 2026-07-18 — the end-state arrived, and it retired the decode authority

The end-state above is **reached**. `Voxel` stores an integer `local_index` (ADR 0003 §3a) and f32
is produced only at consumption, so no consumer holds a world position that needs decoding back to
an index. Combined with ADR 0012 deleting the volumetric fog — the decode's only caller — a survey
found `VoxelGrid::voxel_index_of` had **zero callers in the tree, not even a test.** It has been
deleted; restore from git history if a consumer reappears.

What this changes about the decision:

- **The (a)/(b)/never-(c) rule stands unchanged.** It is the durable part and applies to every
  future frame-bearing Intent, including the sculpt delta.
- **The carry half still binds:** `recentre_voxels` travels on the grid and through the worker
  channels as `RecentreVoxels`, and the two-layer expansion applies it at one arithmetic site.
- **The structural guard is gone with its consumer.** "One decode authority" was the mechanism that
  stopped (c) re-appearing; there is now no decode to be the authority over. The guard against
  re-introducing (c) is that a world position is no longer available to re-derive from — a stronger
  guarantee than the method was, which is why deleting it is not a regression.

### The audit gap this surfaced

Retiring the decode focused attention on the surviving arithmetic, which turned out to carry an
unstated bound. The expansion rebases in `i64` and stamps into an **`i32`** `local_index` with an
unchecked `as` (`two_layer_store::chunk::stamped_voxel`). The S4a audit on `narrow_chunk_coord`
(ADR 0002 Decision 2) states a supported range of ±~8×10⁹ blocks — but that audit bounds the CHUNK
COORDINATE, which fits `i32` precisely because it is a voxel index *divided* by the chunk extent.
The expansion then multiplies back by that same extent. **Bounding a quotient says nothing about
the product it came from**: at the stated range the voxel index overruns `i32` by 4× at density 1
and 238× at density 64, and the cast wraps rather than saturating, so two chunks a multiple of 2³²
voxels apart would alias onto one `local_index`.

Not reachable through the UI (the true bound still allows ±3.3×10⁷ blocks at density 64, against
scenes of tens to thousands), so it is **named and proved rather than fixed** — the cast is in the
innermost expansion loop and the useful output is an honest bound, not a branch. The bound is
`voxel_core::core_geom::max_supported_block_offset`, with Kani harnesses establishing that the cast
is lossless within it and aliases beyond it.
