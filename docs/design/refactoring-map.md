# Refactoring map — where the code should be simplified, and why it matters later

**Provenance:** an opinionated survey of the codebase against the architecture described
in `docs/architecture/`. Unlike that set, this document is deliberately *dated*: it names
files and duplications as they stand, and ties each proposed action to the work the
project is heading into — the sculpt epic, the per-voxel material atlas, cheap background
workers for export-class jobs, 10k+ scenes, and versioned shared documents. Items are
ordered by leverage, not effort.

**Status (2026-07-13):** the structural tier (items 1–4, 6, 7, 11) and the deferred tier
(5a-tiles, 9, 10a first increment, 12) have been executed; each item below carries its
outcome. Still open: 5a-measurement (rejected with reasons — see *What not to refactor*),
5b (waits on measured atlas pressure), 8 (sculpt-design-time), 10a further increments,
10b.

---

## 1. One generic worker, three deletions — **DONE** (`6f895d8`)

`geometry_worker.rs`, `diameter_worker.rs`, and `brick_worker.rs` were three verbatim
copies of the same machine: spawn a named thread, channel requests in, drain-to-latest,
build under panic containment, channel results out, poll without blocking.

**Outcome.** `Worker<Request, Response>` lives in `src/workers/mod.rs` (generic over a
build closure); the domain workers are its submodules (`src/workers/{geometry, diameter,
brick, scan}.rs`). Drain-to-latest, panic containment, and thread lifetime are one
implementation with one set of tests. The prediction held immediately: the `.vox`
exporter (`src/workers/export.rs`, `67bd305`) cost exactly a request type and a build
closure — plus one deliberate divergence worth knowing: an export is a *user-chosen
file*, so it opts OUT of supersede semantics (the shell serialises with a single-flight
flag instead; a drain-to-latest drop would silently discard a file the user asked for).

## 2. Extract the display orchestrator out of `main.rs` — **DONE** (`6ffc553`)

**Outcome.** `DisplayOrchestrator` (`src/display/orchestrator.rs`) owns both renderers
and every display-state field; the winit shell keeps input, surface, egui, and camera.
The orchestrator is constructible without a window, and the state machine has its own
tests (`tests/display_orchestrator.rs`). The review history that motivated this item
remains the operating rule: every slice touching the orchestrator gets a high-effort
multi-agent review afterwards — both post-extraction review rounds (2026-07-12/13) still
found real defects in freshly written shell/export code, so the rule has not aged out.

## 3. One routing policy for all derived artifacts — **DONE** (`c13ace7`)

**Outcome.** One `route_derived_artifact` policy + a per-artifact `DerivedArtifactState`;
the three former dialects are thin named wrappers. The one real divergence — bricks may
install a small wholesale rebuild inline mid-flight — survives explicitly as the
`inline_install_supersedes_in_flight` capability flag rather than as a dialect. The
decision table is tested once, exhaustively.

## 4. Give the display decisions their own module — **DONE** (`aa3e1af`)

**Outcome.** `src/display/routing.rs` owns every pure display decision and its tests
(`route_*`, `brick_display_handover`, `brick_patch_in_place`, `GenerationTracker`);
`src/workers/geometry.rs` is the worker itself.

## 5. Bit-per-voxel occupancy under the density bound — **(a-tiles) DONE, (a-measurement) REJECTED, (b) DEFERRED**

The document guarantees density ∈ 1..=64, which entitles occupancy storage to
word-aligned row bitmasks (one X-row = one `u64`; X is the fastest-varying axis in every
occupancy layout the codebase has).

**Outcome (a-tiles, `d3c6bb3`).** The incremental brick mirror's tiles are
`BrickOccupancyTile` — `edge²` u64 X-row words (8× at density 64, 2× at 16, break-even
at 8). The GPU atlas stays byte R8; the ONE unpack seam is `pack_sculpted_atlas` /
`SculptedAtlasPayload`. A byte↔bit parity oracle pins the packing.

**Outcome (a-measurement): rejected after reading the code** — see *What not to
refactor* below. The survey's "widest-run becomes shifts, masks, and popcounts" was
written against an imagined per-voxel path that no longer exists.

**(b) GPU side — deferred, unchanged:** the atlas becomes row words only when atlas
pressure is first *measured*; only then decide whether eviction rings are still worth
building. This is the sculpt epic's VRAM story (bit-packing is an 8× ceiling move before
any cache policy).

## 6. Split `scene.rs` — **DONE** (`c767f35`)

**Outcome.** `scene/` holds `mod` (facade), `graph`, `extent`, `producers`, `spatial`,
`tests`; the facade re-exports kept all consumers at zero edits. Sculpt's producer arm
and agent authoring land in the `producers` seam as planned.

## 7. Quarantine the dense oracles by visibility — **DONE** (`72d155c`)

**Outcome.** Dense entry points sit behind `#[cfg(any(test, feature = "oracle"))]`; the
`shot` binary opts in via `required-features = ["oracle"]`; a production call to a dense
path is a compile error. Dead `app_core::resolve_scene` deleted in the same slice.

## 8. Decide the fate of the display-less brick mirror — **OPEN (by design)**

On builds without the GPU feature, the incremental brick mirror is still maintained and
nothing consumes it. **Decide at sculpt-design time, not before:** if sculpt consumes the
mirror on all builds, keep it and document the consumer; if not, gate it to the GPU
feature. Do not leave it consumer-less past that decision (the no-husk rule).

## 9. Ship one copy of the field across the worker channel — **DONE, WIDENED** (`d2a0c37`)

**Outcome.** The mirror (`IncrementalBrickField`) is the single CPU owner of records +
tiles. `from_wholesale` consumes the wholesale build **by move** (tiles carried by move
too — no byte round-trip); `BrickDisplayInstall` ships `{atlas payload, gpu_records,
pyramid, overlay, mirror}` with no duplicate `build`; the renderer seams take
`&[BrickRecord]` + `SculptedAtlasPayload` instead of a materialised `BrickFieldBuild`.
The widening: executing the item exposed that every INLINE incremental edit was paying a
full `to_build()` — all records cloned + the whole atlas re-packed on the event-loop
thread, per edit — which the map had not seen. That call is deleted; `to_build()`
survives only as the parity-oracle materialisation (every remaining caller is a test).

## 10. Type-enforce the frame law, then machine-check the pure kernel — **(a) BEGUN, (b) FUTURE**

**(a) Frame newtypes — first increment done (`26cfd81`, `de3da33`).** `RecentreVoxels`
(spatial-primitive layer, `src/voxel.rs`; no arithmetic, `new()` in / `voxels()` out) is
minted at the ONE origin — `Scene::recentre_voxels_for_resolve` — and carried through
the orchestrator, both worker channels, and the renderer install seams; it unwraps once
at uniform packing. Deliberately still raw: `recentre_shift_voxels` (a frame *delta*)
and `previous_recentre_voxels` (a comparison cache) — positional arithmetic, not
transport — and the dense-oracle grid. **Next increments:** `cuboid_mesh.rs` and
`two_layer_store.rs` consume the newtype instead of unwrapping at their boundaries; then
`scene/` internals; then the next frame-bearing value (the sculpt-delta Intent's
addresses, per ADR 0008, when sculpt lands).

**(b) Verify the kernel — future, unchanged in shape:** Kani for the packed world-key
round-trip and the row-bitmask operations (now real code, from 5a); Creusot/Verus for
the generation-tracker supersede protocol and the routing table (now ONE table, from
item 3); a small Lean model for interval-bound conservatism. Still deliberately not
attempted: proving the GPU side. That stance got fresh evidence this cycle — the
long-standing nondeterministic shader-compile flake was diagnosed as legacy FXC
nondeterministically rejecting byte-identical HLSL (fixed at the WGSL layer, `d3ea9cf`);
no source-level theorem would have touched it, which is why the oracle gates remain
permanent regardless of how far verification goes.

## 11. A common fixture crate for integration tests — **DONE** (`6f895d8`, with item 1)

**Outcome.** `tests/common/` owns the scene-fixture builders and the bounded
`poll_until`; the async worker suites and the orchestrator tests share it.

## 12. Keep the documentation contract honest — **DONE / STANDING**

The roles stand: `CONTEXT.md` defines terms; `docs/adr/` records decisions append-only;
`docs/architecture/` describes the current shape and is edited freely; `docs/design/`
holds dated analysis inputs like this one. The specific alignment this item named —
pruning the dead volumetric-fog glossary from `CONTEXT.md` — was already done when
checked (one legitimate historical mention remains inside an ADR reference). The
standing rules: new ADRs describe deltas against the architecture set; new doc comments
reference architecture chapters, not ADR numbers.

---

## What *not* to refactor

Restraint is part of the map:

- **Do not bit-pack the measurement path** (the rejected half of item 5a). The streamed
  widest-run is rayon-parallel across bands; coarse blocks contribute one *analytic*
  X-span per block-row (no per-voxel work at all); boundary cuboids expand to X-spans
  per row into sorted, coalescing interval lists. X-runs cross block and chunk
  boundaries, which interval lists merge naturally and per-block u64 rows cannot —
  word rows would ADD cross-word run-merging machinery for no measured win. The
  representation to beat is intervals, not bytes.
- **Do not merge the edit and render broadphases.** They answer different questions at
  different tempos; a unified spatial index would couple per-edit statelessness to
  per-frame residency and inherit both invalidation problems.
- **Do not sunset the cuboid mesh.** It is the understudy display and the pixel oracle;
  both roles are permanent, and "one display path" is a false economy that costs the
  proof doctrine its independent witness.
- **Do not build atlas eviction (residency rings) ahead of measured pressure.** Item 5's
  bit-packing moved the CPU ceiling 8×; measure the GPU side before adding cache policy.
- **Do not generalize the intent system for collaboration.** Shared documents need
  versioning at the *file* level; the single-writer intent door is a feature, not a
  bottleneck.
- **Do not write through dynamically indexed vector/array components in WGSL.** Learned
  from the X3500 diagnosis (`d3ea9cf`): naga lowers such stores to HLSL l-values that
  legacy FXC nondeterministically rejects. Dynamic reads are fine; stores use masked
  `select`. This is a shader-authoring rule, not a refactor target — new shader code
  must be born conforming.
