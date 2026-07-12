# Refactoring map — where the code should be simplified, and why it matters later

**Provenance:** an opinionated survey of the current codebase against the architecture
described in `docs/architecture/`. Unlike that set, this document is deliberately
*dated*: it names files, line counts, and duplications as they stand today, and ties
each proposed action to the work the project is heading into — the sculpt epic, the
per-voxel material atlas, cheap background workers for export-class jobs, 10k+ scenes,
and versioned shared documents. Items are ordered by leverage, not effort.

---

## 1. One generic worker, three deletions

`geometry_worker.rs`, `diameter_worker.rs`, and `brick_worker.rs` are three verbatim
copies of the same machine: spawn a named thread, channel requests in, drain-to-latest,
build under panic containment, channel results out, poll without blocking. Only the
build function and the request/result types differ.

**Action.** Extract `Worker<Request, Result>` (generic over a build closure), keeping
the three domain modules for their request/result types and pure routing functions.
Drain-to-latest, panic containment, and thread lifetime become one implementation with
one set of tests.

**Why it ties into the plan.** The architecture's answer to "this blocks the UI" is
always "make it a worker" — the `.vox` exporter is already queued for exactly this
treatment, and the sculpt epic will want bake/compress workers. Each future worker
should cost a request type and a build function, not another copy of channel plumbing
that can drift (the drain and panic policies have already needed fixes; today those
fixes must be applied three times).

## 2. Extract the display orchestrator out of `main.rs`

`main.rs` (~2,700 lines) is a winit shell that has quietly become the owner of the
display-state machine: two renderers, `mesh_stale`, `brick_display_pending_clear`,
`brick_fallback_reported`, two generation trackers, two outstanding flags, two install
seams, engagement predicates, and the startup replica of all of the above built from
locals before `Self` exists. The pure *decisions* are already extracted and tested
(`route_mesh_build`, `route_brick_rebuild`, `brick_display_handover`,
`brick_patch_in_place`) — but the *state they act on* is a constellation of fields on
the window struct, and every seam that mutates it is hand-wired.

**Action.** Introduce a `DisplayOrchestrator` owning both renderers and all
display-state fields, with the install seams, poll handlers, and the
startup-first-build as methods. The winit shell keeps input, surface, egui, and camera,
and calls the orchestrator at its (few) integration points. The orchestrator is
constructible without a window, which makes the state machine — not just its pure
fragments — unit-testable.

**Why it ties into the plan.** The recent review history is the evidence: the last
change to this state machine shipped with three real transition bugs that only a
multi-agent review caught, because the state lives too diffusely to reason about
locally. The per-voxel material atlas will *add* display states (a third representable
regime), and sculpt will add patch sources. The constellation is at the edge of
hand-verifiability now; it should become a type before it grows again.

## 3. One routing policy for all derived artifacts

`route_geometry_rebuild`, `route_mesh_build`, and `route_brick_rebuild` are three
dialects of a single rule: *patch inline iff the resident artifact is current and the
edit is localised; otherwise rebuild wholesale, inline below a chunk threshold, async
above it; while a rebuild is outstanding, never patch.* The dialects differ only in
which staleness inputs they fold in (mesh staleness, mirror residency, engagement).

**Action.** One `route_derived_artifact(artifact_state, edit_shape, threshold)` policy
plus a small per-artifact state struct (`current`, `outstanding`, `patchable`). The
three existing functions become thin, named wrappers or disappear into the
orchestrator of item 2. The decision table is then tested once, exhaustively.

**Why it ties into the plan.** Every future derived artifact — the material atlas, an
export snapshot, a nav/occupancy summary for agents — needs exactly this routing. The
cost of a fourth hand-written dialect is not writing it; it is that four dialects can
disagree about the interlock, which is the one rule that must never be dialectal.

## 4. Give the display decisions their own module

The pure display-state functions (`brick_display_handover`, `brick_patch_in_place`,
`route_mesh_build`, and friends) live in `geometry_worker.rs` for historical reasons.
That file's name says "background mesh builder"; its contents say "display policy".

**Action.** A `display_routing.rs` (or the orchestrator module of item 2) owning every
pure display decision and its tests; `geometry_worker.rs` shrinks to the worker itself
(or dissolves into item 1's generic worker plus a build function).

**Why it ties into the plan.** Pure decision functions are this codebase's best habit —
they are where the review effort concentrates and where regressions get caught cheaply.
They deserve an address that tells contributors (and future reviews) where policy
lives.

## 5. Bit-per-voxel occupancy under the density bound

The document now guarantees density ∈ 1..=64, which entitles occupancy storage to
word-aligned row bitmasks (one row = one `u64`; one row = one native `u32` at density
≤ 32). Today the sculpted-brick atlas stores one *byte* per voxel (R8), and CPU-side
occupancy work touches voxels individually.

**Action.** Migrate carved-block occupancy to bit-per-voxel with word rows, in two
independent steps: (a) the CPU side — the incremental brick mirror's tiles and the
occupancy consumed by measurement queries (widest-run becomes shifts, masks, and
popcounts); (b) the GPU side — the atlas becomes a storage buffer of row words (or a
packed R32Uint texture), and the in-brick ray step tests bits. Each step carries its
parity oracle: byte-atlas vs bit-atlas must be hit-identical.

**Why it ties into the plan.** This is the sculpt epic's memory story. At density 16 a
carved block costs 4 KB in R8; a million chiseled blocks — a realistic sculpted build —
is ~4 GB of VRAM, which is at or past the ceiling of the target hardware. Bit-packing
is an 8× cut (~512 MB) *before* any eviction scheme is needed, and it makes the
measurement path faster on exactly the anisotropic 10k+ scenes the project aims at.
Do (a) before sculpt ships; do (b) when atlas pressure is first measured, and only
then decide whether eviction rings are still worth building.

## 6. Split `scene.rs`

At ~6,200 lines, `scene.rs` holds four separable concerns: the node graph and
selection; the leaf producers and their interval bounds; units/measurement; and the
spatial-index / covering-range queries.

**Action.** Four modules under a `scene/` directory along exactly those seams. No
behaviour change; the seams already exist as comment banners.

**Why it ties into the plan.** Sculpt adds a producer arm (the sparse voxel delta) and
agent authoring adds intent surface; both land in the producer seam. A file this size
taxes every future change with navigation and merge friction — and it is the file the
versioned-document format work will live in, which is reason enough to make its
structure legible first.

## 7. Quarantine the dense oracles by visibility

The dense-grid code (`store.rs` and friends) survives correctly as test oracles and in
the headless `shot` binary. Nothing *structurally* prevents a production path from
reaching for a dense resolve again; the law "memory follows the surface" is enforced
by review.

**Action.** Move oracle-only entry points behind `#[cfg(any(test, feature = "oracle"))]`
(the `shot` binary opting into the feature), so a production call to a dense path is a
compile error, not a review catch.

**Why it ties into the plan.** The law has been re-broken before by well-meaning
features (fog, startup, measurement all grew dense paths at some point and each cost a
session to retire). Scale (10k+ anisotropic scenes) makes any regression here an OOM,
not a slowdown. Compile-time enforcement is cheap and permanent.

## 8. Decide the fate of the display-less brick mirror

On builds without the GPU feature, the incremental brick mirror is still maintained —
built wholesale on a worker, patched per edit — and *nothing consumes it*. It is kept
warm on the expectation that sculpt's delta pipeline will read it.

**Action.** Decide at sculpt-design time, not before: if sculpt consumes the mirror on
all builds, keep it and document the consumer; if not, gate the mirror to the GPU
feature and let non-GPU builds skip the work entirely. Do not leave it consumer-less
past that decision — the project's no-husk rule exists because unused-but-maintained
machinery is where staleness bugs breed.

## 9. Ship one copy of the field across the worker channel

A finished wholesale brick build sends both the field (`BrickFieldBuild`) and the
incremental mirror seeded from it — two deep copies of the same records and tiles
crossing the channel and coexisting until install.

**Action.** Make the mirror the single owner and let the install borrow what it needs
from it (or construct the mirror from the field by move). Transient peak memory halves;
one equality invariant ("mirror round-trips the field") stops needing maintenance.

**Why it ties into the plan.** Minor today (~tens of MB transient at the largest
scenes); it becomes real when sculpted tiles dominate the payload, which is exactly
what sculpt does.

## 10. Type-enforce the frame law, then machine-check the pure kernel

Two escalations of the proof doctrine (`docs/architecture/05-proof.md`), in order of
return on effort:

**(a) Frame newtypes.** The frame law — a spatial value carries the frame it was
authored in — is today enforced by doc-comments and review. Rust can enforce it:
wrap lattice coordinates in newtypes tagged by frame (`WorldVoxel`, `ChunkLocal`,
`Recentred`), with explicit, named conversions that require the recentre value. The
half-voxel-drift class of bug becomes a compile error. This is mechanical, incremental
(one seam at a time, starting with the recentre-bearing worker requests), and pays for
itself the first time a new producer or intent carries a coordinate.

**(b) Verify the kernel.** The pure kernel is small and stable enough to
machine-check with Rust-native tools — Kani (bounded model checking) for the packed
world-key encode/decode round-trip and the row-bitmask operations of item 5; Creusot
or Verus (deductive verification) for the generation-tracker supersede protocol and
the routing decision tables. The interval-bound *conservatism* theorem and the
patch-equals-rebuild algebra are better suited to a small Lean (or similar) model of
the evaluator, proven once and kept as the mathematical spec the Rust implementation
mirrors — with the existing parity gates serving as the bridge between model and
implementation. What this deliberately does **not** attempt: proving the GPU side.
Shader compilers and drivers sit below any source-level proof (the observed
nondeterministic shader-compile flake is a *driver-toolchain* defect no theorem would
have touched), which is why the oracle gates remain permanent regardless of how far
verification goes.

## 11. A common fixture crate for integration tests

The box-scene builders, threshold-sized fixtures, and bounded poll loops are duplicated
across `tests/geometry_worker_async.rs`, `tests/brick_worker_async.rs`, and unit tests.

**Action.** A `tests/common/` module (or a `testsupport` crate feature) with the
scene-fixture builders and a generic `poll_until_result`. Worth doing opportunistically
with item 1, whose generic worker will want a generic test harness anyway.

## 12. Keep the documentation contract honest

With `docs/architecture/` as the living description, the documentation roles are:
`CONTEXT.md` defines terms; `docs/adr/` records *decisions and their reasoning* at the
moment they were made (append-only, never retconned); `docs/architecture/` describes
the *current shape* and is edited freely; `docs/design/` holds analysis inputs like
this one. Two immediate alignments: `CONTEXT.md` still opens with a glossary section
for a subsystem that no longer exists (the per-chunk volumetric fog terms — apron,
sliver, fog residency), which should be pruned to the terms the system still uses; and
new ADRs should describe deltas against the architecture set rather than restating it.

---

## What *not* to refactor

Restraint is part of the map:

- **Do not merge the edit and render broadphases.** They answer different questions at
  different tempos; a unified spatial index would couple per-edit statelessness to
  per-frame residency and inherit both invalidation problems.
- **Do not sunset the cuboid mesh.** It is the understudy display and the pixel
  oracle; both roles are permanent, and "one display path" is a false economy that
  costs the proof doctrine its independent witness.
- **Do not build atlas eviction (residency rings) ahead of measured pressure.**
  Item 5's bit-packing moves the ceiling 8× for a fraction of the complexity; measure
  again on the far side before adding cache policy.
- **Do not generalize the intent system for collaboration.** Shared documents need
  versioning at the *file* level; the single-writer intent door is a feature, not a
  bottleneck.
