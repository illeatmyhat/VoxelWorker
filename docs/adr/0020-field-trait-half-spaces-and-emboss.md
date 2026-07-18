# ADR 0020 — The field trait, half-space cutters, and emboss

- **Status:** Accepted (2026-07-18 — design grill; **implementation not started**).
  **Decision 1's premise corrected by ADR 0021**: `DebugCloudField` is *not* fieldless, and
  the `Option` on `as_field` rests on freehand sculpt instead. The trait shape is unchanged.
  Builds on ADR 0019 (the field layer) and extends ADR 0017's `CombineOp`. Alters no ADR 0017 law:
  the new arm reads the accumulator and nothing else, so ordered-DFS-with-no-operand-
  targeting stands.
- **Date:** 2026-07-18
- **Layer:** document model + evaluation semantics. Settles the boolean sugar ADR 0019
  Decision 10 left open.

## Context

ADR 0019 deferred normalMagic's named boolean operations as "Intent-layer sugar expanding
into field combinators." Reading their actual documentation changes that verdict: **three
of the four are workarounds for mesh limitations a field world does not have.**

- **Boolean Trim** cuts with a single-sided plane. Its documented mechanism extrudes the
  plane "into solid forms in the opposite direction" before differencing — machinery that
  exists solely because a mesh cannot represent an unbounded solid. A field can; a half-space
  is the simplest SDF there is.
- **Boolean Extrude / Panel** is "perform an Intersect Boolean and extrude the result into a
  solid panel." The extrusion compensates for a surface∩solid yielding a surface. Voxels are
  already solid, so `Intersect` alone is the whole operation.
- **Boolean Extrude / Emboss** — "slice mesh and extrude intersection faces in or out" — is
  the one with real content: raise or recess the target's surface within a cutter's
  footprint.
- **Cut Groove** is a swept profile subtracted along a path.

ADR 0019 also left the trait shape open. It requires field access and a derived metric, and
the half-space requires something `full_dimensions() -> [u32; 3]` structurally cannot say.

## Decision

1. **The field is a separate trait, optionally provided.**

   ```rust
   trait Field: Send + Sync {
       fn signed_distance(&self, point_local: [f32; 3]) -> f32;
       fn metric(&self) -> FieldMetric;
       fn cell_interval(&self, cell_local_voxels: VoxelAabb) -> Option<FieldInterval>;
   }

   trait VoxelProducer: Send + Sync {
       // ...resolve, resolve_into, full_dimensions as today...
       fn as_field(&self) -> Option<&dyn Field> { None }
   }
   ```

   This makes ADR 0019's "predicates classify, fields measure" split structural rather than
   conventional. A producer without a field **cannot** be outset or embossed, enforced by the
   type rather than by a runtime `None` — `DebugCloudField` (fBm-displaced, genuinely
   fieldless) returns `None` and is excluded by construction. Requiring a field of every
   producer was rejected: it would force `DebugCloudField` to fabricate one, which is
   precisely the sentinel mistake ADR 0019 retired.

2. **A half-space is a primitive, and Boolean Trim dissolves into it.** A plane producer
   plus the existing `Subtract` replaces the entire Trim tool. Its `signed_distance` is exact
   and 1-Lipschitz in both metrics, and its `cell_interval` is exact for any cell — the field
   path handles unboundedness without special cases.

3. **Unbounded producers are legal only under operations whose result the accumulator
   bounds.** `full_dimensions` feeds the ADR 0010 E2 leaf bound and the edit-broadphase BVH;
   an unbounded producer has no AABB and would defeat both. But an unbounded cutter's
   *effect* is bounded: `Subtract` and `Intersect` yield results contained in the
   accumulator, and `Emboss` in a finite dilation of it. **`Union` is the sole operation that
   would be genuinely infinite, and an unbounded producer under it is rejected.** Dirty-region
   computation uses the accumulator's bounds, not the producer's, wherever the producer is
   unbounded.

4. **`CombineOp` gains an `Emboss` arm with a signed amount.** Verified against a
   set-theoretic ground truth over 64,000 samples spanning overlapping, nested, disjoint,
   anisotropic and cutter-exceeds-target configurations at four amounts:

   ```
   outward (N > 0)   A' = min(A, max(A − N, C))        ≡  A ∪ (dilate(A,N) ∩ C)
   inward  (N < 0)   A' = max(A, min(A − N, −C))       ≡  A \ (dilate(¬A,|N|) ∩ C)
   ```

   The composition is exactly 1-Lipschitz (worst observed ratio 1.000000), so the cell
   classifier's bound stays sound.

   **Emboss cannot be sugar, and that is why it is an arm.** The accumulator `A` appears
   twice in both formulas. No node references the accumulated result, and adding one would
   *be* operand targeting — so this cannot decompose into a sequence of existing fold steps.
   It remains law-compatible because, exactly like `Subtract`, it reads "everything
   accumulated before it in this scope" and nothing else.

5. **Boolean Panel is already shipped.** `Intersect` is the whole operation. Recorded so it
   is not built twice or mistaken for missing capability.

6. **Cut Groove is sugar over Sweep + Subtract**, and is therefore blocked on the **Sweep**
   lift arm — the third alongside Extrude and Revolve (ADR 0019 Decision 2). Adopting Cut
   Groove is in practice a decision to prioritise Sweep.

7. **Every new `CombineOp` arm must land in both fold implementations.** The fold exists
   twice: over voxel sets (`crates/document/src/scene/producers.rs`, `fold_closed_scope_into`
   and the inline leaf masks) and over intervals (`crates/substrate/src/solids/
   cell_classification.rs`, `CellClassification::classify`). These are two evaluations of one
   semantics and they diverge silently if only one is updated;
   `src/cell_interval_parity_tests.rs` is the gate that catches it. Any arm added without a
   parity case is unlanded work.

## Considered options

- **Field methods on `VoxelProducer` directly**: rejected per Decision 1. Smaller diff and
  consistent with how `cell_field_interval` arrived, but it grows a trait whose methods are
  inapplicable to some implementors and keeps "has a field" a runtime property.
- **Mandatory `Field` on every producer**: rejected — see Decision 1.
- **Porting Boolean Trim as a named operation**: rejected per Decision 2. Porting the
  *mechanism* would import an extrude-to-solid workaround for a constraint we do not have;
  the affordance survives as primitive + `Subtract`.
- **Emboss via a node that references the accumulated result**: rejected. It is operand
  targeting wearing a different hat, and ADR 0017's law admits no exceptions. The fold arm
  achieves the same authoring result while reading only what `Subtract` already reads.
- **Bounding the half-space to the scene AABB at authoring time**: rejected. It makes the
  cutter's meaning depend on scene extent at the moment of placement, so later growth of the
  scene would silently leave the plane too small. Decision 3's accumulator-bounded rule gets
  the same finite dirty region without baking a stale extent into the document.
- **Outset via expanded-box occupancy query instead of a field**: not adopted as the primary
  path. It works for any producer with no field at all — a point is in the dilated body iff
  the body is non-empty within N of it — but costs O(N³) per voxel against O(1) for a field
  shift. Recorded as the fallback if a fieldless producer ever needs dilation.

## Consequences

- **`SdfShape`, `SketchSolid` and the new half-space implement `Field`; `DebugCloudField`
  does not.** Outset and emboss are therefore unavailable on the cloud field, which is a
  debug producer — acceptable, and now visible in the type rather than discovered at runtime.
- **`full_dimensions` acquires an unbounded case**, and every caller that assumes a finite
  AABB must be audited: the E2 leaf bound, the edit-broadphase BVH, chunk-window clipping,
  and `resolve_into`'s `clamp_window_to_grid`. Decision 3 keeps the blast radius finite but
  does not remove the audit.
- **The parity gate grows cases for `Emboss` in both fold implementations**, plus half-space
  cells (where the exact interval makes classification cheap and total).
- **Emboss depends on outset** (ADR 0019 Decision 7) since its formulas are stated in terms
  of `A − N`; it cannot land first.
- **Sweep becomes the gating item for Cut Groove**, raising its priority relative to the
  other deferred authoring work.
- **Three of normalMagic's four boolean operations cost far less than a port would suggest**
  — one is already shipped, one becomes a primitive, one is a fold arm. Only Cut Groove needs
  genuinely new authoring machinery, and that machinery (Sweep) was already planned.
