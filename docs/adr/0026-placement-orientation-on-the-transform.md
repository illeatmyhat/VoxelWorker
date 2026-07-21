# ADR 0026 — Placement orientation: a lattice-exact turn on the node transform

- **Status:** Accepted
- **Date:** 2026-07-21
- **Relates to:** [ADR 0001](0001-scene-graph-parts-and-tools.md) (decision 3 — `NodeTransform`
  targets a full affine; this fills the *rotation* slot with its discrete, lattice-preserving
  first half), [ADR 0008](0008-voxel-frame-invariant.md) (orientation is one more carried frame
  fact, never re-derived), [ADR 0010](0010-boundary-residency-two-layer-store.md) (the classifier
  is the single application point), and the placement model in
  `crates/raycast/src/placement.rs` (a geometry face sets orientation; the world planes do not).

## Context — the resolver already knows the facing; nothing downstream turns the node

`resolve_placement` reports `OnSurface { point, face_normal }` — the entered face's outward normal
is an exact ±1 axis vector — but `Intent::PlaceNode` carried only `offset_voxels`. A cylinder
dropped on a wall stayed upright instead of lying on its side poking out along the wall's normal.
The facing was computed and thrown away. This ADR wires it through.

## Decision

**A placed node carries an orientation: one of the 24 axis-aligned cube rotations, a signed axis
permutation, stored on `NodeTransform` beside the offset.** It is named **orientation** and typed
`LatticeOrientation`; the word **rotation** is reserved for a later *continuous* affine turn that
would resample voxels. See the glossary (`CONTEXT.md`, "Authoring frame").

Four things follow:

1. **It lives on the transform, not the shape.** Orientation is a placement fact — "which way is
   this node turned in its parent" — the same category as the offset, and it applies to *any* node
   kind (a Tool today; an instance or sketch-solid the day they become arm-and-droppable), not only
   an `SdfShape`. Putting it on the shape (the smaller change) would have scoped it to one producer
   kind and split a future general-orient gizmo across two homes. This is the home ADR 0001
   decision 3 already reserved.

2. **It is discrete and lattice-exact — never a matrix.** A voxel face normal is an exact ±axis, so
   the whole turn is a signed axis permutation. It relabels and flips axes: no resampling, it
   preserves a field's Lipschitz bound, and classification stays occupancy-identical to brute force
   (ADR 0010). This is the house style already set by the sketch subsystem's `PlaneAxis` /
   `RevolveAxis` — orientation as a discrete enum, not a float rotation.

3. **The producer stays unoriented; the classifier applies the inverse permutation.** The transform
   is applied where the evaluator maps an absolute voxel into a producer's local frame — today a
   subtract of `world_offset_voxels` in `two_layer_store/classify.rs`; now that map also applies the
   leaf's inverse orientation (and its forward orientation when emitting local voxel indices back to
   absolute). Because the permutation is exact and invertible, a box maps to a box and the
   producer's field / interval / resolve are called unchanged. **The SDF parity surface — the three
   hand-synced twins in `sdf_shape.rs` and the WGSL `value_main` probe — does not move**, so the
   parity gate stays green for free. The two-layer store is the single source the mesh *and* the GPU
   brick raymarch read, so both display paths inherit orientation with no further work.

4. **Orientation composes down the tree as a rigid transform.** `world_orientation = parent ∘ node`,
   and a child's offset turns by its parent's world-orientation before it sums —
   `world_offset = parent_offset + parent_orientation · node_offset`. This is pivot-free integer
   work (an offset is a vector, not a point), so it is written fully correct rather than
   leaf-restricted, and nesting an oriented parent is sound even though only placement sets a
   non-identity orientation today.

**Derivation at placement:** a geometry face turns the node's local **+Z to the face normal** by
the shortest-arc swing (zero twist; the −Z face takes a canonical flip). The curved primitives are
symmetric about Z, so the twist freedom is invisible for them; a box is symmetric under 90° twists,
so it is invisible there too at v1 sizes. The built-in **world planes keep identity orientation** —
they position only, standing the node world-vertical — which is consistent with a +Z geometry face
(a box on the ground or on another box's top stays upright, exactly as before).

## Considered options

- **Orientation on `SdfShape` (rejected).** Smaller — composition at `producers.rs` stays
  translation-only and only the shape and its ghost change. Rejected because it scopes orientation
  to one producer kind, conflates "what the shape is" with "how it is placed", and would force a
  document migration when a general orient gizmo arrives. The producer must see the orientation for
  its interval bound *either way*, so the shape home bought no parity simplification the transform
  home doesn't also get from the classifier-remap.
- **A general (continuous) rotation now (rejected).** Would demand voxel resampling, break the
  lattice-exact invariants every classifier and the GPU brick sink rely on, and vastly overshoot a
  placement that only ever needs an axis turn. Deferred behind the reserved word "rotation".

## Consequences

- `NodeTransform` gains an orientation field (`#[serde(default)]` = identity, so old documents load
  unchanged). It reaches the document, so it is versioned state — a later move of this field would
  be a real migration, which is exactly why the home was chosen deliberately here rather than for
  minimum diff.
- One GPU shader must learn orientation: the **ghost preview** (`placement_ghost.wgsl`), which
  samples the analytic SDF directly rather than reading the two-layer store. Every *finished*
  display path (mesh, brick raymarch) needs no shader change.
- `face_anchored_offset` must seat against the **oriented** extent (the permuted grid dimensions),
  reconciling the flush-anchor with the turn.
