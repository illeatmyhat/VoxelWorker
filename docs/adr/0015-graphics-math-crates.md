# ADR 0015 — Graphics-math crates (`camera`, `raycast`): wgpu-free implementations of the well-known concepts; the crate is the shader's readable specification

- **Status:** **Accepted & shipped (2026-07-14)** — slices G0–G2 all landed
  (`b5cb208`..`1dcce29`); extraction map in `docs/design/graphics-crates-extraction-map.md`.
  **Amended during execution:** the owner unified the two AABB types by co-location in
  substrate (`LatticeAabb` half-open integer / `RealAabb` closed f32, side-by-side docs) —
  generics and traits were considered and rejected (a bound-policy parameter hurts the
  accessibility goal; a `BoundingVolume` trait waits for a second BVH consumer). **Supersedes, in part, the
  future-crates ruling recorded in the substrate extraction map** (2026-07-13, same day),
  which listed "display/shell" as deliberately-not-a-crate: that stands for wgpu *plumbing*,
  but the owner's follow-on direction — graphics crates in the substrate vein — is satisfied
  by the finer distinction this ADR draws.
- **Date:** 2026-07-13
- **Layer:** repo shape. Extends ADR 0014's crate test to the display side.

## Context

ADR 0014 proved the pattern: well-known structures extracted under their literature names,
compile-enforced boundaries, tests and citations traveling with them. The owner asked what the
same treatment yields for graphics. The survey found the split that matters is not
"display vs the rest" but **graphics mathematics vs wgpu plumbing**: the mathematics
(projection, orbit control, frustum culling, ray–volume traversal) has literature identity and
is already nearly pure in this codebase (`camera.rs` imports only glam; the CPU reference
march's DDA kernel is separable — `cpu_march_exact_occupancy` already takes an injected
occupancy closure), while the plumbing (pipelines, bind groups, uploads) has neither a
literature name nor isolation value.

This codebase adds a binding law no ordinary extraction has: **gpu_parity**. The WGSL shaders
are maintained mirrors of CPU reference implementations. Extracting the references into
literature-named crates makes each crate *the readable specification of a shader*, with the
parity suite as the mechanical link.

## Decision

1. **Two crates**: `crates/camera` (viewing + projective geometry: orbit control, projection,
   pole-continuous up, ViewCube model, framing fit, Gribb–Hartmann frustum + p-vertex culling,
   screen→ray unprojection) and `crates/raycast` (ray–volume traversal: slab test,
   Amanatides–Woo DDA at block and voxel scale, hierarchical empty-space skip, entry-face
   normals, band-clip — generic over injected occupancy closures).
2. **The graphics-crate law**: glam + substrate only; never wgpu; never each other. Shared
   vocabulary (`Ray`) lives in substrate beside `Aabb`. Domain adapters (the ADR 0008 carried
   frame, record/atlas byte fetch, occupancy policies, winit input handling) stay in the app.
3. **WGSL stays app**, documented as each crate's GPU mirror; gpu_parity + goldens are the
   proof obligation for every extraction slice, exactly as the parity oracles were for
   substrate.
4. **`shading` is deliberately NOT a crate** — the survey found ~4 small CPU-side functions;
   all substantive shading is WGSL-only. The sRGB↔linear codec and the shelf rect packer go to
   **substrate** (textbook pure math / pure CS), not to a graphics crate.
5. ADR 0014's rules carry over verbatim: literature names ARE the component names, citations
   in module docs are definition-of-done, benches only for hot components (none initially —
   the CPU march is an oracle, not a hot path).

## Considered options

- **One `graphics` crate** holding both: rejected — camera math and traversal have disjoint
  consumers (UI vs parity oracle/picking) and no shared code beyond `Ray`; one crate would be
  a taxonomy bucket, not a law.
- **`shading` as a third crate**: rejected on survey evidence (too thin; no dependency law).
- **Ray owned by `raycast` with a camera→raycast edge**: rejected to keep the graphics crates
  mutually independent; a ray is textbook geometry and substrate already owns the geometric
  primitives.

## Consequences

- The ADR 0013 material-atlas epic sequences AFTER slice G2: its per-voxel material sampling
  extends the `raycast` kernel rather than being re-extracted later.
- The dependency picture gains two spokes: `substrate ← camera / raycast ← voxel_worker`;
  the future `document`/`evaluation` chain is unchanged.
- Machine-checked construction (ADR 0014 decision 6) gains natural targets here too — the DDA
  loop's termination/coverage and the frustum test's no-false-negatives claim are
  Kani/Creusot-shaped; sequenced with the rest of 10b, never blocking extraction.
