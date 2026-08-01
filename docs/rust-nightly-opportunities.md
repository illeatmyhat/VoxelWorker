# Rust nightly opportunities

> Checked: 2026-07-31
>
> Stable Rust checked: **1.97.1**
>
> Nightly documentation checked: **1.99.0-nightly**

This is a forward-looking note, not a recommendation to move the project to
nightly. Revisit these areas if they become stable in a future Rust release.

## `portable_simd`

Potentially useful for CPU-side hot loops in:

- SDF evaluation and composition
- voxel occupancy sampling
- raycast/DDA stepping
- brick decoding and mesh preprocessing
- palette or vertex-buffer transformations

This is the most promising candidate, but only if profiling shows CPU math is
the bottleneck. Branching, irregular memory access, and boundary handling may
limit the benefit; much of the rendering parallelism already runs on the GPU.

## `allocator_api`

A per-worker arena or bump allocator could reduce allocation churn in temporary
structures created during wholesale geometry rebuilds, scans, and exports. This
would require an ownership/lifetime refactor, so it is worth considering only
if allocation cost is visible in profiles.

## `generic_const_exprs`

Could encode chunk dimensions, brick sizes, face counts, and fixed lookup-table
shapes in types rather than relying on runtime checks. The likely benefit is
stronger invariants and clearer APIs, not runtime performance. Most scene and
window dimensions are intentionally runtime-sized, so applicability is limited.

## Async iterators and async trait features

Could express scans, exports, or geometry results as async streams. The current
design already uses dedicated worker threads, `mpsc` channels, polling, and
generation-based superseding, so this would primarily change the programming
model rather than improve throughput.

## Coroutines / generators

Could represent incremental scanning, meshing, or export as resumable state
machines. This might simplify some streaming code, but the existing worker loops
are explicit, testable, and already provide the required non-blocking behavior.

## Nightly compiler and Cargo options

Nightly-only profiling, code-generation, custom-target, and build experiments
may be useful for investigation. They should remain tooling experiments rather
than production requirements unless a concrete measured benefit emerges.

## Suggested revisit order

1. Profile CPU geometry and allocation costs.
2. Prototype `portable_simd` in one isolated hot loop if CPU math dominates.
3. Consider an arena allocator if temporary allocation churn is significant.
4. Reconsider the remaining features mainly for API ergonomics or invariant
   enforcement.
