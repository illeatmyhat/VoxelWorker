//! ADR 0027 rotated / sub-voxel placement for the dense oracle: the continuous
//! [`substrate::spatial::LeafPlacement`] construction ([`dense_leaf_placement`]), the
//! out-of-phase predicate ([`leaf_is_out_of_phase`]), and the inverse-resample gather
//! that writes a genuinely rotated / sub-voxel-seated field into a [`VoxelGrid`]
//! ([`gather_placed_field_into_grid`]). Shared by both dense resolve paths.

use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::VoxelGrid;
use crate::voxel::VoxelProducer;

use crate::scene::*;

/// Build the ADR 0027 continuous placement [`substrate::spatial::LeafPlacement`] for a leaf the
/// dense oracle is stamping — the SAME corner-anchored world↔producer-local affine the
/// two-layer classifier folds through (its evaluation-layer `leaf_affine` constructs an
/// identical `LeafPlacement`). Sharing substrate's ONE map — rather than the dense path's old
/// translation-only copy — is what stops the reference oracle silently disagreeing with the
/// live path on where a rotated / sub-voxel-seated producer's cells land (the deferred "Step 2").
///
/// `leaf_abs_low_voxels` is the OUTSET producer's low corner in the scene's ABSOLUTE voxel frame
/// (the visitor's `world_offset_voxels` minus the outset), matching the two-layer leaf's
/// `world_offset_voxels`; `offset_local_voxels` is the ADR 0027 continuous sub-voxel slide added
/// on top. `producer` is the same boxed producer the stamp resolves, so `full_dimensions` matches.
pub(super) fn dense_leaf_placement(
    rotation: glam::Quat,
    offset_local_voxels: [f32; 3],
    leaf_abs_low_voxels: [i64; 3],
    producer: &dyn VoxelProducer,
    voxels_per_block: u32,
) -> substrate::spatial::LeafPlacement {
    let full_dimensions = producer.full_dimensions(voxels_per_block);
    let full = glam::Vec3::new(
        full_dimensions[0] as f32,
        full_dimensions[1] as f32,
        full_dimensions[2] as f32,
    );
    let world_offset = glam::Vec3::new(
        leaf_abs_low_voxels[0] as f32,
        leaf_abs_low_voxels[1] as f32,
        leaf_abs_low_voxels[2] as f32,
    ) + glam::Vec3::from_array(offset_local_voxels);
    substrate::spatial::LeafPlacement::new(
        rotation,
        full,
        substrate::spatial::TrueWorldVoxelPoint::from_voxels(world_offset),
    )
}

/// Whether a leaf is OUT OF PHASE with the absolute voxel lattice (ADR 0027): a genuine
/// (non-axis-aligned) rotation, OR a fractional `offset_local_voxels` sub-voxel seat. An
/// out-of-phase FIELD leaf cannot be emitted one-cell-per-abs-cell by a translation, so the dense
/// oracle resamples it by inverse gather ([`gather_placed_field_into_grid`]) — mirroring the
/// two-layer classifier's `gather_rotated_leaf_into_region`. A whole-phase leaf (integer offset,
/// axis-aligned rotation — every gate scene) keeps the exact translate-and-stamp path, so the
/// existing goldens stay byte-identical.
pub(super) fn leaf_is_out_of_phase(rotation: glam::Quat, offset_local_voxels: [f32; 3]) -> bool {
    let axis_aligned = substrate::spatial::is_axis_aligned(rotation);
    let integer_offset = offset_local_voxels.iter().all(|slide| slide.fract() == 0.0);
    !(axis_aligned && integer_offset)
}

/// The ADR 0027 **inverse-resample gather** for a genuinely out-of-phase (rotated or sub-voxel-
/// seated) FIELD leaf, writing into the dense oracle's output [`VoxelGrid`]. The single-leaf
/// occupancy definition BOTH dense paths ([`Scene::resolve_region`] and
/// [`Scene::resolve_chunk_rebased`]) share, and the exact `VoxelGrid` mirror of the two-layer
/// classifier's `gather_rotated_leaf_into_region` — both fold through substrate's ONE
/// [`substrate::spatial::LeafPlacement`], so the dense reference can no longer drop the rotation
/// the live path applies.
///
/// For every output cell in the placed box, its centre is inverse-mapped into the producer-local
/// frame and the field is sampled: inside-or-on-surface cells are covered. The leaf's `operation`
/// is then applied to `output` exactly as the forward stamp path does — `Union` stamps the covered
/// cells (later document-order write wins on overlap), `Subtract` clears every covered cell, and
/// `Intersect` keeps ONLY the covered cells (killing accumulated cells anywhere outside the body,
/// including the whole grid when the body covers nothing — `A ∩ ∅ = ∅`).
///
/// `output_origin_abs` is the absolute voxel the output grid's index `[0,0,0]` denotes (the
/// recentre for `resolve_region`; the floating origin for `resolve_chunk_rebased`), so output
/// index `oi` denotes absolute cell `oi + output_origin_abs`. `clip_abs`, when `Some`, keeps only
/// cells whose absolute index lies in the half-open box (the chunk membership clip — the voxel
/// centre `+0.5` cancels on integer chunk edges exactly as the forward chunk stamp derives).
#[allow(clippy::too_many_arguments)]
pub(super) fn gather_placed_field_into_grid(
    output: &mut VoxelGrid,
    placement: &substrate::spatial::LeafPlacement,
    producer: &dyn VoxelProducer,
    material_override: Option<voxel_core::core_geom::BlockId>,
    grid_overlay: bool,
    operation: CombineOp,
    output_origin_abs: [i64; 3],
    clip_abs: Option<VoxelAabb>,
    voxels_per_block: u32,
) {
    use voxel_core::voxel::{BlockAttrs, Voxel, SURFACE_ISOLEVEL};

    let field = producer
        .as_field()
        .expect("the dense gather is only reached for field producers (ADR 0027)");
    let (world_min, world_max) = placement.world_aabb();

    // The output-index box the leaf can touch: its absolute world AABB rebased to the output
    // frame (`abs − output_origin_abs`), intersected with the optional absolute clip box. Both the
    // world box and the clip are half-open, so the per-axis min/max of their rebased edges is the
    // exact overlap. The result is NOT clamped to the grid dimensions: a recentred dense grid
    // stores `i32` indices whose origin sits at a negative position (see `Voxel::local_index`), so
    // the stamp path never bounds the index to `[0, dimensions)`, and neither may the gather.
    let mut lo = [0i64; 3];
    let mut hi = [0i64; 3];
    for axis in 0..3 {
        let mut min_index = world_min[axis] - output_origin_abs[axis];
        let mut max_index = world_max[axis] - output_origin_abs[axis];
        if let Some(clip) = clip_abs {
            min_index = min_index.max(clip.min[axis] - output_origin_abs[axis]);
            max_index = max_index.min(clip.max[axis] - output_origin_abs[axis]);
        }
        lo[axis] = min_index;
        hi[axis] = max_index.max(min_index);
    }

    // Sample the field at every candidate cell centre, collecting the covered output cells and
    // the material each takes (the leaf's single-material override, else the producer's per-voxel
    // material, else the default id — the same precedence the forward stamp uses).
    let mut covered: Vec<([i32; 3], voxel_core::core_geom::BlockId)> = Vec::new();
    for z in lo[2]..hi[2] {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                let output_index = [x, y, z];
                let abs_centre = glam::Vec3::new(
                    (output_index[0] + output_origin_abs[0]) as f32 + 0.5,
                    (output_index[1] + output_origin_abs[1]) as f32 + 0.5,
                    (output_index[2] + output_origin_abs[2]) as f32 + 0.5,
                );
                let local = placement
                    .local_of(substrate::spatial::TrueWorldVoxelPoint::from_voxels(abs_centre))
                    .voxels()
                    .to_array();
                if field.signed_distance(local, voxels_per_block) <= SURFACE_ISOLEVEL {
                    let block_id = material_override
                        .or_else(|| producer.material_at(local, voxels_per_block))
                        .unwrap_or(voxel_core::core_geom::BlockId::DEFAULT);
                    // The recentred dense grid stores i32 indices (ADR 0008): the rebased output
                    // index fits i32 for every representable scene, as the stamp path assumes.
                    covered.push((
                        [output_index[0] as i32, output_index[1] as i32, output_index[2] as i32],
                        block_id,
                    ));
                }
            }
        }
    }

    match operation {
        // Later document-order leaf wins on overlap: appending the covered voxels reproduces the
        // dense Union (the resolved occupancy set keeps the last writer at each cell).
        CombineOp::Union => {
            output.occupied.reserve(covered.len());
            for (output_index, block_id) in covered {
                output.occupied.push(Voxel {
                    local_index: output_index,
                    block_local_coord: std::array::from_fn(|axis| {
                        (output_index[axis] as i64 + output_origin_abs[axis])
                            .rem_euclid(voxels_per_block.max(1) as i64) as u8
                    }),
                    block_id,
                    attrs: BlockAttrs::DEFAULT,
                    grid_overlay,
                });
            }
        }
        // Occupancy-only masks (ADR 0017 Decision 1): the covered cells are removed (Subtract) or
        // are the ONLY survivors (Intersect); surviving voxels keep their own material/overlay.
        CombineOp::Subtract => {
            let carved: std::collections::HashSet<[i32; 3]> =
                covered.iter().map(|(index, _)| *index).collect();
            output.occupied.retain(|voxel| !carved.contains(&voxel.local_index));
        }
        CombineOp::Intersect => {
            let kept: std::collections::HashSet<[i32; 3]> =
                covered.iter().map(|(index, _)| *index).collect();
            output.occupied.retain(|voxel| kept.contains(&voxel.local_index));
        }
        // Unreachable: an Emboss scope is pre-composed into a CompositeProducer before it reaches
        // a visitor, and a composed root sits at identity rotation / integer offset (in phase), so
        // it never routes to this gather.
        CombineOp::Emboss { .. } => {}
    }
}
