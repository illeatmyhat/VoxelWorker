//! Runtime chunk resolve: the sole live per-chunk resolver
//! ([`Scene::resolve_chunk`] / [`Scene::resolve_chunk_rebased`], ADR 0002 / ADR
//! 0010) the two-layer store calls, plus its per-leaf chunk-clipped stamp / mask
//! helpers ([`stamp_producer_into_chunk`] / [`mask_producer_in_chunk`]).

use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::VoxelGrid;
use crate::voxel::VoxelProducer;

use super::*;
use super::gather::{dense_leaf_placement, gather_placed_field_into_grid, leaf_is_out_of_phase};
use super::scope_fold::sync_grid_scope_stack;
use crate::scene::*;

impl Scene {
    /// Resolve exactly **one chunk** of the scene into a fresh [`VoxelGrid`], in
    /// **absolute (non-recentred) composite voxel coordinates**.
    ///
    /// This is the chunk-addressable counterpart to `resolve_region` required by
    /// issue #27 (deep chunked resolve). `resolve_region` is now the test/oracle-only
    /// dense measuring stick (ADR 0010 boundary residency retired it from the live
    /// render path; it is compile-gated behind `cfg(test)`/`oracle`) — the two-layer
    /// store (`evaluation::two_layer_store`) is the sole runtime path, and it calls
    /// THIS resolver per chunk. `resolve_region` recentres the composite on the
    /// origin; this path does **not** recentre, so its voxel positions are the
    /// scene's true composite coordinates. The two frames differ by exactly the
    /// recentre offset `resolve_region` subtracts (see
    /// `recentre_voxels`).
    ///
    /// A chunk is a `CHUNK_BLOCKS³`-block cell (`CHUNK_BLOCKS = 4`,
    /// [`voxel_core::core_geom::CHUNK_BLOCKS`]); one chunk therefore spans
    /// `CHUNK_BLOCKS * voxels_per_block` voxels per axis. `chunk_coord` is that
    /// cell's integer coordinate, so the chunk covers the **half-open** absolute
    /// voxel box
    /// `[chunk_coord * chunk_extent_voxels, (chunk_coord + 1) * chunk_extent_voxels)`
    /// per axis. Boundary ownership is `floor(world_position / chunk_extent_voxels)`:
    /// because every resolved voxel centre sits at an `n + 0.5` position and chunk
    /// boundaries fall on integer multiples of `chunk_extent_voxels`, the `floor`
    /// is never ambiguous and every voxel lands in **exactly one** chunk.
    ///
    /// The returned grid's `dimensions` are one chunk's voxel extent
    /// (`chunk_extent_voxels³`); the occupied voxels keep their **absolute**
    /// composite `world_position` (they are NOT rebased to the chunk's local origin
    /// — that, like the recentre removal, is a later step). An empty chunk (no leaf
    /// overlaps it) returns an empty grid; it never panics.
    ///
    /// `voxels_per_block` is the application density (ADR 0001). `lod` is the parked
    /// level-of-detail seam (ADR 0002 Decision 2): it is **always `0`** for now and
    /// is asserted so; it exists from day one so a future down-sampling LOD level is
    /// a behavioural change, not a signature break.
    pub fn resolve_chunk(
        &self,
        chunk_coord: [i32; 3],
        voxels_per_block: u32,
        lod: u32,
    ) -> VoxelGrid {
        // The bare `resolve_chunk` keeps the S0 contract: ABSOLUTE composite
        // positions (floating origin `[0, 0, 0]`). The live render path uses
        // `resolve_chunk_rebased` with the floating origin = the composite recentre.
        self.resolve_chunk_rebased(chunk_coord, voxels_per_block, lod, [0, 0, 0])
    }

    /// Resolve one chunk like [`resolve_chunk`](Self::resolve_chunk), but store each
    /// voxel's position **rebased to `floating_origin_voxels`** (ADR 0002 Decision 2,
    /// camera-relative / origin-rebased rendering — S4b).
    ///
    /// The stored `world_position` is `absolute_composite_position −
    /// floating_origin_voxels`, with the subtraction performed in **i64 before the
    /// f32 downcast**, so the rendered f32 magnitude stays small no matter how far the
    /// chunk sits from the absolute origin. The chunk-membership clip is still decided
    /// in **absolute** space (f64), so a far chunk's boundary voxels are never
    /// misclassified by f32 rounding.
    ///
    /// `floating_origin_voxels = [0, 0, 0]` reproduces `resolve_chunk` exactly. The
    /// live render passes [`recentre_voxels_for_resolve`](Self::recentre_voxels_for_resolve)
    /// (the composite recentre, an integer-block-aligned point), so for a near scene
    /// the result is bit-identical to today's recentred `resolve_region` while a
    /// far-placed scene renders with no f32 jitter (the S1 speckle fix).
    pub fn resolve_chunk_rebased(
        &self,
        chunk_coord: [i32; 3],
        voxels_per_block: u32,
        lod: u32,
        floating_origin_voxels: [i64; 3],
    ) -> VoxelGrid {
        debug_assert_eq!(lod, 0, "S0 only resolves full resolution (lod 0)");

        // Chunk extent fits i64 trivially; the chunk's absolute-voxel corners can be
        // large (a far-placed chunk), so they are computed in i64 (S4a).
        let chunk_extent_voxels = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;

        // The chunk's half-open absolute-voxel box `[min, max)` per axis.
        let chunk_min_voxels = [
            chunk_coord[0] as i64 * chunk_extent_voxels,
            chunk_coord[1] as i64 * chunk_extent_voxels,
            chunk_coord[2] as i64 * chunk_extent_voxels,
        ];
        let chunk_max_voxels = [
            chunk_min_voxels[0] + chunk_extent_voxels,
            chunk_min_voxels[1] + chunk_extent_voxels,
            chunk_min_voxels[2] + chunk_extent_voxels,
        ];

        // The chunk grid is one chunk's voxel extent. (The voxels keep ABSOLUTE
        // positions inside it; `dimensions` describes the chunk's size, not the
        // window of absolute space the positions live in — the consumers that need
        // chunk-local coordinates rebase later, S4.)
        let chunk_dimensions = [
            chunk_extent_voxels as u32,
            chunk_extent_voxels as u32,
            chunk_extent_voxels as u32,
        ];
        let mut output = VoxelGrid::new(chunk_dimensions);

        // Each leaf is resolved into its own origin-centred local grid (exactly as
        // `resolve_region` does), translated by its WORLD offset × density — but
        // WITHOUT the composite recentre, so positions are absolute. We then keep
        // only the voxels whose absolute centre falls in this chunk's box.
        let region_dimensions = self.placed_region_dimensions(voxels_per_block);
        let chunk_box = VoxelAabb::new(chunk_min_voxels, chunk_max_voxels);
        // ADR 0017 Decision 3 (issue #74): the same scoped depth-first fold as
        // `resolve_region`, restricted to this chunk. Composition is cell-local (a
        // union appends a cell, a subtract removes a cell), so restricting every
        // stamp / carve / scope-close to the chunk's cells commutes with the fold —
        // the reassembled chunks equal the monolithic scoped resolve exactly. A leaf
        // whose AABB misses the chunk is skipped WITHOUT syncing the stack: it
        // contributes no cells here, and a scope none of whose leaves touch the
        // chunk simply never opens (an empty scope folds to nothing under Union /
        // Subtract). EXCEPTION (ADR 0017 #75): an Intersect-influence leaf is never
        // skipped — its mask applies precisely where its body has no cells, and an
        // Intersect-closing scope must open even here so its ∅-in-chunk body
        // annihilates the parent on close (see the skip guard below).
        let mut scope_stack: Vec<(ScopeFrame, VoxelGrid)> = Vec::new();
        // ADR 0026: the discrete lattice `orientation` is still not applied here (identity for
        // every gate scene). ADR 0027 "Step 2": the CONTINUOUS `rotation` and the fractional
        // `offset_local_voxels` ARE applied — a genuinely out-of-phase FIELD leaf is resampled by
        // the shared inverse-gather ([`gather_placed_field_into_grid`]) AND its chunk-skip AABB is
        // taken from the ROTATED world box (the placement affine), so a tilted body is neither
        // truncated by the upright skip nor stamped upright.
        self.for_each_leaf(&mut |world_offset_voxels, offset_local_voxels, rotation, body, grid_on_faces, operation, outset, scope_path| {
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            // An outset body grows on every side, so its low corner moves DOWN by the
            // outset — and the skip AABB below must use the DILATED span, or a cutter whose
            // dilation reaches into this chunk would be skipped and its mask silently lost.
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            // Issue #27 S3 optimisation: skip a leaf whose world-AABB doesn't touch
            // this chunk, so resolving one chunk costs ~the leaves that overlap it
            // (not the whole tree). This is BIT-IDENTICAL to stamping-then-clipping:
            // the leaf's AABB `[off·d − grid/2, off·d + grid/2)` is the exact span of
            // its voxel centres, and `stamp_producer_into_chunk` keeps only centres
            // inside `[chunk_min, chunk_max)`; if those two half-open boxes don't
            // intersect, the stamp would have clipped EVERY voxel anyway. A
            // region-spanning leaf (a VoxelBody, `leaf_size_blocks` → `None`) has no
            // localisable AABB, so it is never skipped (it may emit anywhere).
            //
            // ADR 0017 (#75): an Intersect-INFLUENCE leaf (its own operation is
            // Intersect, or any enclosing scope closes under Intersect) is NEVER
            // skipped either: its mask kills accumulated cells anywhere OUTSIDE its
            // body, so a chunk its AABB misses is exactly where the mask must still
            // apply (its body has no cells here ⇒ everything accumulated in this
            // chunk within its scope dies). Keeping it also guarantees every
            // Intersect-closing scope OPENS in this chunk's fold (its leaves all
            // carry the Intersect frame), so the ∅-body scope close annihilates the
            // parent here exactly as the monolithic fold does.
            if !operation_masks_beyond_bounds(operation, scope_path) {
                if let Some(grid_voxels) = body.grid_voxels(voxels_per_block, outset_voxels) {
                    // The leaf's true footprint in the absolute frame. For a whole-phase leaf
                    // (axis-aligned rotation, integer offset — every gate scene) the producer
                    // corner-anchors its grid, so this is `[off, off + grid)`, bit-identical to
                    // stamping-then-clipping. ADR 0027: a genuinely rotated / sub-voxel-seated
                    // leaf's footprint is the ROTATED box, so it is taken from the SAME placement
                    // affine the gather stamps through — otherwise the upright box would skip the
                    // chunks the tilted body occupies and TRUNCATE it (the tubes-render-upright bug).
                    let leaf_box = if leaf_is_out_of_phase(rotation, offset_local_voxels) {
                        let full = glam::Vec3::new(
                            grid_voxels[0] as f32,
                            grid_voxels[1] as f32,
                            grid_voxels[2] as f32,
                        );
                        let world_offset = glam::Vec3::new(
                            world_offset_voxels[0] as f32,
                            world_offset_voxels[1] as f32,
                            world_offset_voxels[2] as f32,
                        ) + glam::Vec3::from_array(offset_local_voxels);
                        let (min, max) = substrate::spatial::LeafPlacement::new(
                            rotation,
                            full,
                            substrate::spatial::TrueWorldVoxelPoint::from_voxels(world_offset),
                        )
                        .world_aabb();
                        VoxelAabb::new(min, max)
                    } else {
                        let leaf_min = world_offset_voxels;
                        let leaf_max: [i64; 3] =
                            std::array::from_fn(|axis| leaf_min[axis] + grid_voxels[axis]);
                        VoxelAabb::new(leaf_min, leaf_max)
                    };
                    if !leaf_box.intersects(&chunk_box) {
                        return;
                    }
                }
            }
            let translation_voxels = world_offset_voxels;
            // ADR 0019 Decision 7: dilate before folding, exactly as the dense path does.
            let Some((material_override, producer)) =
                body.into_producer(region_dimensions, voxels_per_block, outset_voxels)
            else {
                return;
            };
            // The leaf overlaps the chunk: sync the scope stack to its path (closing /
            // opening scopes exactly where the depth-first fold does) and compose into
            // the innermost open scope's scratch grid — or `output` at root level.
            sync_grid_scope_stack(&mut scope_stack, &mut output, scope_path, chunk_dimensions);
            let target: &mut VoxelGrid = match scope_stack.last_mut() {
                Some((_, scratch)) => scratch,
                None => &mut output,
            };
            // ADR 0027 "Step 2": a genuinely out-of-phase FIELD leaf is resampled by the shared
            // inverse-gather through substrate's placement affine — the SAME map (and per-cell
            // field test) the two-layer classifier folds through, so the dense chunk oracle agrees
            // with the live path on rotated / sub-voxel seats. Here the output grid holds ABSOLUTE
            // positions (floating origin `[0,0,0]` for the bare `resolve_chunk`, the recentre for
            // the rebased render path), so `oi` denotes absolute cell `oi + floating_origin_voxels`,
            // and the chunk membership clip keeps only cells in `[chunk_min, chunk_max)`.
            if leaf_is_out_of_phase(rotation, offset_local_voxels) && producer.as_field().is_some() {
                let placement = dense_leaf_placement(
                    rotation,
                    offset_local_voxels,
                    world_offset_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                );
                gather_placed_field_into_grid(
                    target,
                    &placement,
                    producer.as_ref(),
                    material_override,
                    grid_on_faces,
                    operation,
                    floating_origin_voxels,
                    Some(chunk_box),
                    voxels_per_block,
                );
                return;
            }
            // ADR 0017: a Subtract leaf carves its body's cells OUT of the voxels
            // stamped so far in this chunk WITHIN ITS SCOPE (occupancy-only — no
            // material, no stamp). A leaf whose AABB missed the chunk was already
            // skipped above (it carves nothing here), so this sees only
            // genuinely-overlapping cutters.
            if operation == CombineOp::Subtract {
                mask_producer_in_chunk(
                    target,
                    region_dimensions,
                    translation_voxels,
                    floating_origin_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    chunk_min_voxels,
                    chunk_max_voxels,
                    false,
                );
                return;
            }
            // ADR 0017 (#75): an Intersect leaf keeps ONLY the cells its body covers
            // in this chunk within its scope (occupancy-only). It is never skipped by
            // the AABB guard, so a mask whose box misses the chunk resolves an EMPTY
            // window here and correctly kills everything accumulated so far — the
            // restriction to this chunk's cells still commutes with the fold, because
            // a cell survives iff the mask occupies THAT cell.
            if operation == CombineOp::Intersect {
                mask_producer_in_chunk(
                    target,
                    region_dimensions,
                    translation_voxels,
                    floating_origin_voxels,
                    producer.as_ref(),
                    voxels_per_block,
                    chunk_min_voxels,
                    chunk_max_voxels,
                    true,
                );
                return;
            }
            stamp_producer_into_chunk(
                target,
                region_dimensions,
                translation_voxels,
                floating_origin_voxels,
                material_override,
                // Issue #29 S4: OR the on-face-grid flag bit onto each kept voxel
                // iff this node opted in, so the bit travels through the chunked
                // render path exactly as it does through `resolve_region`.
                grid_on_faces,
                producer.as_ref(),
                voxels_per_block,
                chunk_min_voxels,
                chunk_max_voxels,
            );
        });
        // Close every scope still open after the last overlapping leaf.
        sync_grid_scope_stack(&mut scope_stack, &mut output, &[], chunk_dimensions);

        output
    }
}

/// Resolve `producer` into its own origin-centred local grid, translate it by
/// `translation_voxels` (the node's WORLD placement × density — **no recentre**),
/// and stamp only the voxels whose absolute centre falls in the half-open chunk
/// box `[chunk_min_voxels, chunk_max_voxels)` into `output`.
///
/// This is the chunk-scoped sibling of [`stamp_producer`]: same per-leaf
/// resolution, same material-override rule (a Tool overwrites every voxel's id;
/// `None` keeps the producer's own ids), but it (a) never recentres and (b)
/// clips each voxel to one chunk. Ownership is `floor(world_position /
/// chunk_extent_voxels)` per axis; since centres sit at `n + 0.5` and boundaries
/// at integer multiples of the chunk extent, each voxel lands in exactly one
/// chunk.
/// `floating_origin_voxels` is the **render floating origin** (ADR 0002 Decision 2,
/// camera-relative / origin-rebased rendering — S4b): the integer-voxel point the
/// rendered f32 frame is rebased around. The stored `world_position` is the voxel's
/// absolute composite position **minus the floating origin**, with the subtraction
/// done in **i64 BEFORE the f32 downcast** so the rendered f32 magnitude stays small
/// regardless of how far the chunk sits from the absolute origin (no far-lands
/// jitter). Pass `[0, 0, 0]` to store true absolute positions (the chunk-cache
/// parity tests / `.vox`-style consumers). The chunk-membership clip is computed in
/// **f64 absolute** space (independent of the rebase) so a far chunk's boundary
/// voxels are never misclassified by f32 rounding.
#[allow(clippy::too_many_arguments)]
fn stamp_producer_into_chunk(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    floating_origin_voxels: [i64; 3],
    material_override: Option<voxel_core::core_geom::BlockId>,
    grid_overlay: bool,
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    chunk_min_voxels: [i64; 3],
    chunk_max_voxels: [i64; 3],
) {
    // Resolve ONLY the cells this chunk owns, in the producer's LOCAL voxel-index
    // frame `[0, full_dim)`. A producer's local cell `idx` has absolute centre
    // `translation_voxels[axis] + idx + 0.5`; the historical chunk-membership clip
    // kept `chunk_min ≤ translation + idx + 0.5 < chunk_max`. The `+ 0.5` cancels on
    // half-open INTEGER chunk edges:
    //   idx + 0.5 ≥ chunk_min  ⟺  idx ≥ chunk_min − translation
    //   idx + 0.5 <  chunk_max  ⟺  idx <  chunk_max − translation
    // so the chunk window in the local frame is the integer half-open box below.
    // `resolve_into` clamps it to `[0, full_dim)` internally, so an out-of-range
    // window is safe, and it returns EXACTLY the cells the old per-voxel clip kept —
    // a producer spanning N chunks now resolves each chunk's cells once instead of
    // re-resolving its full extent N×.
    let mut local = VoxelGrid::new(region_dimensions);
    let window_local = voxel_core::spatial_index::VoxelAabb::new(
        [
            chunk_min_voxels[0] - translation_voxels[0],
            chunk_min_voxels[1] - translation_voxels[1],
            chunk_min_voxels[2] - translation_voxels[2],
        ],
        [
            chunk_max_voxels[0] - translation_voxels[0],
            chunk_max_voxels[1] - translation_voxels[1],
            chunk_max_voxels[2] - translation_voxels[2],
        ],
    );
    producer.resolve_into(&mut local, voxels_per_block, window_local);

    // The voxel's chunk-local placement, rebased to the floating origin in i64
    // FIRST so the f32 add never sees a large magnitude. For the live render the
    // floating origin equals the composite recentre, so for a near scene this is
    // EXACTLY the small `world_offset·d − recentre` translation `resolve_region`
    // adds in f32 today — bit-identical framing — while a far chunk no longer loses
    // the voxel-centre `.5` to f32 rounding at ~1e6 magnitude (the S1 speckle).
    let rebased_translation = [
        translation_voxels[0] - floating_origin_voxels[0],
        translation_voxels[1] - floating_origin_voxels[1],
        translation_voxels[2] - floating_origin_voxels[2],
    ];

    output.occupied.reserve(local.occupied.len());
    for mut voxel in local.occupied {
        // Store the rebased (origin-relative) INTEGER index (ADR 0003 §3a). The rebase
        // is a pure i64 subtraction done here BEFORE the downcast, so the far chunk's
        // index keeps full precision — the f32 magnitude loss the old f32 payload took
        // at ~1e6 (the S1 speckle) is gone, and `world_position()` (= index + 0.5)
        // reproduces the small rebased centre exactly for a near scene.
        voxel.local_index[0] = (voxel.local_index[0] as i64 + rebased_translation[0]) as i32;
        voxel.local_index[1] = (voxel.local_index[1] as i64 + rebased_translation[1]) as i32;
        voxel.local_index[2] = (voxel.local_index[2] as i64 + rebased_translation[2]) as i32;

        if let Some(id) = material_override {
            voxel.block_id = id;
        }
        // ADR 0003 §3c: transient render marker, not the categorical id (see stamp_producer).
        voxel.grid_overlay = grid_overlay;
        output.occupied.push(voxel);
    }
}

/// Resolve `producer`'s cells inside the chunk window and **occupancy-mask** `output`
/// with them (ADR 0017 Decision 1). Each already-stamped voxel whose (rebased) index
/// coincides with one of the mask's cells is *covered*; `keep_if_covered` picks which
/// side of the mask survives — the chunk-scoped sibling of [`mask_producer`]:
///
/// * `keep_if_covered = false` → **Subtract** (carve): covered voxels are removed.
/// * `keep_if_covered = true`  → **Intersect** (issue #75): only covered voxels
///   survive. Restricting the mask to the chunk window is EXACT (not merely
///   conservative): a cell survives iff the mask occupies that very cell, and every
///   output voxel here lies inside the chunk — a mask cell in another chunk can only
///   affect that other chunk. A mask whose box misses this chunk entirely resolves an
///   EMPTY window and thus clears everything accumulated so far, exactly
///   `accumulated ∩ ∅ = ∅` restricted here.
///
/// Like [`stamp_producer_into_chunk`], uses the same local resolve window
/// (`[chunk_min, chunk_max)` mapped into the producer's local frame — a mask cell
/// outside this chunk can only affect OTHER chunks) and the same
/// i64-before-f32-downcast rebase to `floating_origin_voxels`, so the covered index
/// coincides bit-exactly with the stamped index it keeps or removes.
#[allow(clippy::too_many_arguments)]
fn mask_producer_in_chunk(
    output: &mut VoxelGrid,
    region_dimensions: [u32; 3],
    translation_voxels: [i64; 3],
    floating_origin_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
    chunk_min_voxels: [i64; 3],
    chunk_max_voxels: [i64; 3],
    keep_if_covered: bool,
) {
    // Resolve ONLY the mask cells this chunk owns, in the producer's LOCAL
    // voxel-index frame — the identical window arithmetic as the stamp (see
    // `stamp_producer_into_chunk` for the half-open-edge derivation).
    let mut local = VoxelGrid::new(region_dimensions);
    let window_local = voxel_core::spatial_index::VoxelAabb::new(
        [
            chunk_min_voxels[0] - translation_voxels[0],
            chunk_min_voxels[1] - translation_voxels[1],
            chunk_min_voxels[2] - translation_voxels[2],
        ],
        [
            chunk_max_voxels[0] - translation_voxels[0],
            chunk_max_voxels[1] - translation_voxels[1],
            chunk_max_voxels[2] - translation_voxels[2],
        ],
    );
    producer.resolve_into(&mut local, voxels_per_block, window_local);

    // Rebase the mask's indices exactly as the stamp rebases stamped ones (pure
    // i64 subtraction BEFORE the downcast), so mask and stamp agree bit-exactly.
    let rebased_translation = [
        translation_voxels[0] - floating_origin_voxels[0],
        translation_voxels[1] - floating_origin_voxels[1],
        translation_voxels[2] - floating_origin_voxels[2],
    ];
    let covered: std::collections::HashSet<[i32; 3]> = local
        .occupied
        .iter()
        .map(|voxel| {
            [
                (voxel.local_index[0] as i64 + rebased_translation[0]) as i32,
                (voxel.local_index[1] as i64 + rebased_translation[1]) as i32,
                (voxel.local_index[2] as i64 + rebased_translation[2]) as i32,
            ]
        })
        .collect();
    output
        .occupied
        .retain(|voxel| covered.contains(&voxel.local_index) == keep_if_covered);
}
