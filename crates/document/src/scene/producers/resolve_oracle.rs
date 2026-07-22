//! The cfg-gated dense oracle: the whole-region [`Scene::resolve_region`] and its
//! chunk-decomposition twin [`Scene::resolve_region_via_chunks`] — the dense,
//! O(volume) measuring sticks the sparse runtime path is held against, excluded from
//! production builds behind the `oracle` feature (tests reach them via `cfg(test)`) —
//! plus their per-leaf stamp / mask helpers ([`stamp_producer`] / [`mask_producer`]).
//! See the proof chapter's "Oracles" section (`docs/architecture/05-proof.md`).

use voxel_core::voxel::VoxelGrid;
use crate::voxel::VoxelProducer;

use super::*;
use super::gather::{dense_leaf_placement, gather_placed_field_into_grid, leaf_is_out_of_phase};
use super::scope_fold::sync_grid_scope_stack;
use crate::scene::*;

impl Scene {
    /// Resolve `region` into a fresh [`VoxelGrid`] by a union tree-walk: each
    /// enabled leaf producer is resolved into its own local grid and **stamped**
    /// into the output under the node's transform.
    ///
    /// `voxels_per_block` is the application density (ADR 0001 "Density": a global
    /// setting, default 16, that the scene reads at resolve time).
    ///
    /// `lod` is the level-of-detail seam required by ADR 0001 ("Deferred: LOD").
    /// It is **always `0`** (full resolution) for now; the parameter exists from
    /// day one so a future LOD level (which would downsample a chunk before
    /// meshing) is a possible change rather than a signature break. Step 1
    /// asserts it is `0`.
    ///
    /// **Identical-behaviour guarantee:** for a one-node scene whose `region`
    /// equals the node's full extent with a zero offset, the stamp is the
    /// identity, so the result equals what the bare producer emits today.
    ///
    /// **Oracle — compile-gated.** This is a dense, O(volume) whole-region resolver:
    /// the measuring stick the sparse runtime path is held against, never a runtime
    /// path itself. It is excluded from production builds behind the `oracle` feature
    /// (tests reach it via `cfg(test)`), so "memory follows the surface" is enforced by
    /// the compiler, not by review — see the proof chapter's "Oracles" section
    /// (`docs/architecture/05-proof.md`).
    #[cfg(any(test, feature = "oracle"))]
    pub fn resolve_region(
        &self,
        region: RegionBlocks,
        voxels_per_block: u32,
        lod: u32,
    ) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "step 1 only resolves full resolution (lod 0)");

        // The region grid is sized in the PRODUCER VOXEL FRAME (corner-anchoring):
        // the recentred composite occupies exactly `[region_low, region_low + D)` with
        // `D = max_v − min_v` (`placed_extent_voxels`) and `region_low = min_v −
        // recentre`, so a block-framed region (`size·d`) would clip a parity-mismatched
        // multi-leaf composite. For a chunkable scene we IGNORE the passed-in block
        // `region` for sizing and use the voxel span; the explicit `region` argument
        // still sizes a VoxelBody-only scene (which has no composite voxel extent).
        let region_dimensions = match self.placed_extent_voxels(voxels_per_block) {
            Some(_) => self.placed_region_dimensions(voxels_per_block),
            None => [
                region.size_blocks[0] * voxels_per_block,
                region.size_blocks[1] * voxels_per_block,
                region.size_blocks[2] * voxels_per_block,
            ],
        };
        let mut output = VoxelGrid::new(region_dimensions);

        // Recentre the composite so its world positions sit symmetrically about
        // the origin (what the renderer + camera auto-frame assume). Each producer
        // CORNER-ANCHORS its grid (local span `[0, grid)`); a leaf's low corner in the
        // composite's voxel space is `offset_voxels`, and the whole composite's centre
        // is `(min + max).div_euclid(2)` (producer-true voxel frame). Subtracting that
        // centre from every node's translation lands the composite centred in `output`.
        // A VoxelBody-only scene (e.g. `DebugClouds`) has no composite extent, so this is
        // `[0,0,0]` and the field stays CORNER-anchored at `[0, region)` — the shipped
        // convention (see `part_only_cloud_at_odd_density_drops_no_voxels` /
        // `mixed_tool_and_cloud_resolve_in_one_frame`). ADR 0008: the recentre is CARRIED on
        // the grid (below), so every consumer decodes correctly without re-deriving the
        // frame as `floor(dim/2)` (the assumption that dropped the corner-anchored cloud fog).
        let recentre_voxels = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        output.recentre_voxels = recentre_voxels;

        // Walk the whole tree (groups + instances recurse, composing world
        // translation down — ADR 0001 step 4). Each visited leaf is stamped under
        // its WORLD voxel offset minus the composite recentre. The offset is
        // already voxels at the document density (ADR 0003 §3f(0)), so it enters
        // the sum as-is. All of this is in i64 (S4a) so a far-placed node composes
        // without overflow; the result is downcast to f32 inside the stamp (the
        // render frame stays f32 — S4b makes the far case byte-identical via origin
        // rebasing).
        // ADR 0017 Decision 3 (issue #74): the walk is evaluated as a SCOPED depth-first
        // fold — each open Group / definition-body scope composes its leaves into its own
        // scratch grid, and a closing scope folds that composed body into its parent under
        // the SCOPE's operation (`sync_grid_scope_stack`), so a boolean inside a scope can
        // never affect geometry outside it.
        let mut scope_stack: Vec<(ScopeFrame, VoxelGrid)> = Vec::new();
        // ADR 0026: the discrete lattice `orientation` is still not applied here (an oriented
        // leaf is checked through the two-layer classifier against a hand-derived expectation, not
        // against this oracle) — every parity-gate scene is lattice-identity. ADR 0027 "Step 2":
        // the CONTINUOUS `rotation` quaternion and the fractional `offset_local_voxels` ARE now
        // applied, by routing a genuinely out-of-phase FIELD leaf through the shared inverse-gather
        // ([`gather_placed_field_into_grid`], substrate's ONE placement affine) so the dense
        // reference agrees with the live path on rotated / sub-voxel seats. A whole-phase leaf
        // (integer offset, axis-aligned rotation) keeps the exact translate-and-stamp path below.
        self.for_each_leaf(&mut |world_offset_voxels, offset_local_voxels, rotation, body, grid_on_faces, operation, outset, scope_path| {
            sync_grid_scope_stack(&mut scope_stack, &mut output, scope_path, region_dimensions);
            let target: &mut VoxelGrid = match scope_stack.last_mut() {
                Some((_, scratch)) => scratch,
                None => &mut output,
            };
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            // Every producer corner-anchors its grid at its world voxel offset (the low
            // corner); the recentre (from the producer-true voxel frame) symmetrises the
            // composite about the origin for ALL size·d parities, so no per-leaf lattice
            // shift is needed — a leaf simply sits at its world voxel offset.
            //
            // An outset body grows on every side, so its low corner moves DOWN by the outset
            // (ADR 0008 — the frame is carried, never re-derived).
            let translation_voxels = [
                world_offset_voxels[0] - recentre_voxels[0] - outset_voxels,
                world_offset_voxels[1] - recentre_voxels[1] - outset_voxels,
                world_offset_voxels[2] - recentre_voxels[2] - outset_voxels,
            ];
            // ADR 0017: Subtract and Intersect leaves are occupancy-only masks — they
            // never stamp material, so they take a mask path instead of a stamp. A
            // Subtract CARVES its body out of everything stamped before it (document
            // order, within its scope); an Intersect (issue #75) keeps ONLY the cells
            // its body covers, killing accumulated cells anywhere OUTSIDE its body —
            // including an empty result when nothing accumulated yet (fold start).
            // ONE producer serves both the mask and the stamp paths, so the outset wrapper
            // applies at a single point (ADR 0019 Decision 7 — the outset dilates the body
            // before it folds, whatever the fold role).
            let Some((material, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };

            // ADR 0027 "Step 2": a genuinely out-of-phase FIELD leaf (a continuous rotation or a
            // fractional sub-voxel seat) cannot be emitted one-cell-per-abs-cell by the integer
            // translation below, so resample it by inverse gather through substrate's shared
            // placement affine — the SAME map (and the same per-cell field test) the two-layer
            // classifier folds through, so the dense oracle agrees with the live path. The output
            // grid index `oi` denotes absolute cell `oi + recentre_voxels`, and the leaf's low
            // corner in the absolute frame is `world_offset_voxels − outset` (matching the
            // two-layer leaf's `world_offset_voxels`).
            if leaf_is_out_of_phase(rotation, offset_local_voxels) && producer.as_field().is_some() {
                let leaf_abs_low: [i64; 3] =
                    std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
                let placement = dense_leaf_placement(
                    rotation,
                    offset_local_voxels,
                    leaf_abs_low,
                    producer.as_ref(),
                    voxels_per_block,
                );
                gather_placed_field_into_grid(
                    target,
                    &placement,
                    producer.as_ref(),
                    material,
                    grid_on_faces,
                    operation,
                    recentre_voxels,
                    None,
                    voxels_per_block,
                );
                return;
            }

            // ADR 0017: Subtract and Intersect leaves are occupancy-only masks — they
            // never stamp material, so they take a mask path instead of a stamp.
            match operation {
                CombineOp::Subtract => mask_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    false,
                ),
                CombineOp::Intersect => mask_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    true,
                ),
                // Unreachable in practice: a scope containing an Emboss node is pre-composed
                // into a CompositeProducer (`CombineOp::needs_accumulated_field`), which
                // evaluates the formulas on the accumulated FIELD — the only representation
                // the voxel-set fold and the interval fold can agree on. A voxel-set
                // accumulator has no `A − N` to read. Skipping rather than falling back to
                // Union keeps an unevaluable emboss VISIBLE as a missing feature instead of
                // silently resolving as the wrong operation.
                CombineOp::Emboss { .. } => {
                    eprintln!(
                        "scene: skipping an Emboss node whose scope could not be composed                          (an un-composable scope has no accumulated field to emboss)"
                    );
                }
                CombineOp::Union => stamp_producer(
                    target,
                    region_dimensions,
                    translation_voxels,
                    material,
                    // Issue #29 S4: OR the on-face-grid flag bit onto every
                    // stamped voxel iff this node opted in, so the bit travels
                    // with each voxel (and survives chunk bucketing).
                    grid_on_faces,
                    producer.as_ref(),
                    voxels_per_block,
                ),
            }
        });
        // Close every scope still open after the last leaf (folding each composed
        // body down into `output` under its scope's operation).
        sync_grid_scope_stack(&mut scope_stack, &mut output, &[], region_dimensions);

        output
    }

    /// Resolve the scene's whole region by **decomposing it into chunks** and
    /// merging them back into one grid, in **absolute (non-recentred) coordinates**.
    ///
    /// This loops over every chunk coordinate covering the composite AABB, calls
    /// [`resolve_chunk`](Self::resolve_chunk) for each, and unions the results. It
    /// proves the chunk decomposition reconstructs the whole scene; it is **not**
    /// wired into rendering (the render path stays on `resolve_region`, which
    /// recentres — see issue #27 S0). The returned grid is sized to the full
    /// composite extent and its voxels keep their absolute composite positions;
    /// compared against `resolve_region`'s output it differs only by the
    /// recentre offset.
    ///
    /// **Oracle — compile-gated.** A dense whole-region resolver kept only to prove the
    /// chunk decomposition reconstructs the scene; it is excluded from production builds
    /// behind the `oracle` feature (tests reach it via `cfg(test)`) so a dense path is a
    /// compile error, not a review catch — see the proof chapter's "Oracles" section
    /// (`docs/architecture/05-proof.md`).
    #[cfg(any(test, feature = "oracle"))]
    pub fn resolve_region_via_chunks(&self, voxels_per_block: u32, lod: u32) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let mut output = VoxelGrid::new(region_dimensions);

        let Some(chunk_range) = self.covering_chunk_range(voxels_per_block) else {
            // No leaf has an intrinsic size (a VoxelBody-only scene with no Tools): no
            // composite AABB, so there are no chunks to resolve.
            return output;
        };
        let (min_chunk, max_chunk) = chunk_range;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk =
                        self.resolve_chunk([chunk_x, chunk_y, chunk_z], voxels_per_block, lod);
                    output.occupied.extend(chunk.occupied);
                }
            }
        }
        output
    }
}

/// Resolve `producer` into its own local grid (centred at the origin, as the
/// trait guarantees) and **stamp** it into `output`, translated by
/// `translation_voxels` (the node's placement minus the composite recentre, in
/// voxels).
///
/// When `translation_voxels` is zero and no material override applies, the stamp
/// is the identity: the producer's occupied set is moved into `output` unchanged
/// (the one-node, zero-offset path — guarantees a bit-for-bit match with the bare
/// producer). When `material_override` is `Some(id)`, every stamped voxel takes
/// that id (a Tool's single material); when `None`, each voxel keeps the material
/// the producer emitted (a VoxelBody's own per-voxel materials).
///
/// Private helper of the dense [`Scene::resolve_region`] oracle only (the per-chunk
/// path uses [`stamp_producer_into_chunk`]), so it carries the same `oracle` compile
/// gate — see the proof chapter's "Oracles" section (`docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
fn stamp_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    material_override: Option<voxel_core::core_geom::BlockId>,
    grid_overlay: bool,
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
) {
    // The producer sizes its own grid (`SdfShape::resolve` overwrites
    // `dimensions` to its own canonical `size_voxels`, centred at the origin), so
    // the local grid need only seed the dimensions; the cloud field, which has no
    // intrinsic size, fills the region it is handed.
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local, voxels_per_block);

    let zero_offset = translation_voxels == [0, 0, 0];

    if zero_offset && material_override.is_none() && !grid_overlay {
        // Fast path / exact identity: no translation, no material rewrite and no
        // on-face-grid flag bit, so the local occupied set IS the output.
        if output.occupied.is_empty() {
            output.occupied = local.occupied;
            return;
        }
        output.occupied.extend(local.occupied);
        return;
    }

    // General stamp: translate each voxel into the composite (the producer's
    // origin-centred position plus the node's recentred placement), overwrite its
    // material id for a Tool, then OR the on-face-grid flag bit (issue #29 S4) when
    // this node opted in so it travels with each voxel.
    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        if !zero_offset {
            // ADR 0003 §3a / ADR 0008: translate the INTEGER index in the grid's frame
            // (the absolute origin lives on the grid), never an f32 position. The add is
            // i64 then downcast, so the placement is exact for any magnitude.
            voxel.local_index[0] = (voxel.local_index[0] as i64 + translation_voxels[0]) as i32;
            voxel.local_index[1] = (voxel.local_index[1] as i64 + translation_voxels[1]) as i32;
            voxel.local_index[2] = (voxel.local_index[2] as i64 + translation_voxels[2]) as i32;
        }
        if let Some(id) = material_override {
            voxel.block_id = id;
        }
        // ADR 0003 §3c: the on-face-grid flag is a transient render marker on the cell,
        // NOT the categorical `block_id` — the cuboid mesher reads it (splitting boxes on
        // it) and the draw enables the overlay; it never enters the categorical id.
        voxel.grid_overlay = grid_overlay;
        output.occupied.push(voxel);
    }
}

/// Resolve `producer` into its own local grid and **occupancy-mask** `output` with it
/// (ADR 0017 Decision 1 — `Subtract`/`Intersect` are occupancy-only, never stamping
/// material). Each output voxel whose index coincides with one of the producer's
/// occupied cells (translated by `translation_voxels`) is *covered*; whether covered
/// voxels are the ones KEPT or the ones REMOVED is the single varying bit:
///
/// * `keep_if_covered = false` → **Subtract** (carve): covered voxels are removed.
/// * `keep_if_covered = true`  → **Intersect** (issue #75): only covered voxels
///   survive, so every accumulated voxel outside the mask's body dies — however far
///   from its AABB.
///
/// Surviving voxels keep their material and overlay; the cutter/mask's own material
/// never enters the output. The mask sibling of [`stamp_producer`], and like it a
/// private helper of the dense [`Scene::resolve_region`] oracle only, so it carries
/// the same `oracle` compile gate (see the proof chapter's "Oracles" section,
/// `docs/architecture/05-proof.md`).
#[cfg(any(test, feature = "oracle"))]
fn mask_producer(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    keep_if_covered: bool,
) {
    let mut local = VoxelGrid::new(region_dimensions);
    producer.resolve(&mut local, voxels_per_block);

    // The mask's occupied INTEGER indices in the output's frame (the same
    // i64-then-downcast translation the stamp applies, so a covered cell coincides
    // bit-exactly with the stamped cell it keeps or removes).
    let covered: std::collections::HashSet<[i32; 3]> = local
        .occupied
        .iter()
        .map(|voxel| {
            [
                (voxel.local_index[0] as i64 + translation_voxels[0]) as i32,
                (voxel.local_index[1] as i64 + translation_voxels[1]) as i32,
                (voxel.local_index[2] as i64 + translation_voxels[2]) as i32,
            ]
        })
        .collect();
    output
        .occupied
        .retain(|voxel| covered.contains(&voxel.local_index) == keep_if_covered);
}
