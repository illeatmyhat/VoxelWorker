//! Which node owns a voxel (ADR 0032): the picked-node resolver
//! ([`Scene::picked_node_at_voxel`]), the single-cell scoped fold it runs
//! ([`fold_owner_into`]), and the per-leaf coverage test that fold asks
//! ([`leaf_covers_cell`]).

use super::gather::{dense_leaf_placement, leaf_is_out_of_phase};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{VoxelGrid, SURFACE_ISOLEVEL};

use crate::scene::*;

impl Scene {
    /// The node a viewport pick on `absolute_voxel` selects (ADR 0032), or `None` when the
    /// cell resolves to nothing.
    ///
    /// **The pick follows the material.** This runs ADR 0017's ordered scoped fold over the
    /// ONE cell, carrying "who owns this cell" where the resolvers carry occupancy — so the
    /// node it names is exactly the node whose material the resolve stamped there. Any other
    /// rule would select a node whose colour the user cannot see at the point they clicked.
    ///
    /// Running the real fold rather than scanning for a covering body is what makes the
    /// booleans come out right, and they are the whole difficulty:
    /// - a `Subtract` cutter's surface IS the wall of the hole it carved, so every cavity
    ///   wall cell lies on the cutter's boundary — a nearest-hit rule hands back the cutter;
    /// - a cutter INSIDE a scope carves only its own scope's body, so the cell falls through
    ///   to whatever the scope was folded over — which is a different node, not none;
    /// - a boolean never owns a cell in the first place, because it never stamps material
    ///   (ADR 0017 Decision 1). It only takes ownership away.
    ///
    /// A pre-composed scope is ONE leaf to the fold (ADR 0019 Decision 7, ADR 0020 Decision
    /// 4 — and a single top-level `Emboss` pre-composes the whole scene), so ownership of a
    /// cell inside one descends into its members via [`VoxelProducer::origin_at`] and names
    /// the innermost authored leaf.
    ///
    /// `absolute_voxel` is an integer cell index in the scene's ABSOLUTE voxel frame (ADR
    /// 0008 — the same frame `resolve_chunk` clips against), NOT a recentred display index.
    ///
    /// [`VoxelProducer::origin_at`]: crate::voxel::VoxelProducer::origin_at
    pub fn picked_node_at_voxel(
        &self,
        absolute_voxel: [i64; 3],
        voxels_per_block: u32,
    ) -> Option<NodeId> {
        let leaves = self.leaf_producers(voxels_per_block);
        // One `Option<LeafOrigin>` per open scope, exactly where `sync_grid_scope_stack`
        // keeps one scratch `VoxelGrid` — the same stack-evaluated depth-first fold, asked
        // about a single cell.
        let mut open_scopes: Vec<(ScopeFrame, Option<LeafOrigin>)> = Vec::new();
        let mut root_owner: Option<LeafOrigin> = None;

        for leaf in &leaves {
            sync_owner_scope_stack(&mut open_scopes, &mut root_owner, &leaf.scope_path);
            let placement = leaf_placement(leaf, voxels_per_block);
            let local = placement
                .local_of_abs_cell_centre(absolute_voxel)
                .voxels()
                .to_array();
            let covered =
                leaf_covers_cell(leaf, &placement, absolute_voxel, local, voxels_per_block);
            let owner = match open_scopes.last_mut() {
                Some((_, scope_owner)) => scope_owner,
                None => &mut root_owner,
            };
            match leaf.operation {
                // Later-wins: a covering Union takes the cell from whoever held it.
                CombineOp::Union => {
                    if covered {
                        *owner = Some(leaf_origin_at_local_point(leaf, local, voxels_per_block));
                    }
                }
                // Occupancy-only masks (ADR 0017 Decision 1) never own — they only unown.
                CombineOp::Subtract => {
                    if covered {
                        *owner = None;
                    }
                }
                CombineOp::Intersect => {
                    if !covered {
                        *owner = None;
                    }
                }
                // A scope holding an Emboss node normally pre-composes into one
                // CompositeProducer before it reaches a visitor. One that could NOT compose
                // still arrives here — the dense fold's matching arm prints and skips it — so
                // this skips too, and both agree that an unevaluable emboss stays visible as
                // missing geometry instead of silently resolving as some other operation.
                CombineOp::Emboss { .. } => {}
            }
        }
        // Close everything still open, innermost first.
        sync_owner_scope_stack(&mut open_scopes, &mut root_owner, &[]);
        root_owner.map(LeafOrigin::picked_node)
    }
}

/// Sync the single-cell fold's scope stack to `target_path`, closing and opening exactly
/// where [`sync_grid_scope_stack`](super::scope_fold) does for the dense resolvers — scopes
/// are contiguous in the depth-first walk, so comparing the open stack against the next
/// leaf's carried path recovers the marker sequence.
fn sync_owner_scope_stack(
    open_scopes: &mut Vec<(ScopeFrame, Option<LeafOrigin>)>,
    root_owner: &mut Option<LeafOrigin>,
    target_path: &[ScopeFrame],
) {
    let mut common = 0;
    while common < open_scopes.len()
        && common < target_path.len()
        && open_scopes[common].0 == target_path[common]
    {
        common += 1;
    }
    while open_scopes.len() > common {
        let (frame, closed) = open_scopes
            .pop()
            .expect("len checked by the loop condition");
        let parent = match open_scopes.last_mut() {
            Some((_, scope_owner)) => scope_owner,
            None => &mut *root_owner,
        };
        fold_owner_into(parent, frame.operation, closed);
    }
    for frame in &target_path[common..] {
        open_scopes.push((*frame, None));
    }
}

/// Fold one CLOSED scope's owner into its parent under the scope's own [`CombineOp`] — the
/// ownership mirror of `fold_closed_scope_into`'s occupancy rules (ADR 0017 Decision 3).
/// "The scope's body covers this cell" is exactly "the scope has an owner for it".
fn fold_owner_into(
    parent: &mut Option<LeafOrigin>,
    operation: CombineOp,
    closed: Option<LeafOrigin>,
) {
    match operation {
        // The scope's body wins the cell, carrying the member that authored it.
        CombineOp::Union => {
            if closed.is_some() {
                *parent = closed;
            }
        }
        CombineOp::Subtract => {
            if closed.is_some() {
                *parent = None;
            }
        }
        CombineOp::Intersect => {
            if closed.is_none() {
                *parent = None;
            }
        }
        // Skipped rather than folded, exactly as the dense resolvers skip a scope close whose
        // siblings could not be composed.
        CombineOp::Emboss { .. } => {}
    }
}

/// The origin a hit at `local` (the cell centre in the leaf's own frame) resolves to: the
/// composite member that authored it when the leaf is a pre-composed scope, else the leaf's
/// own origin.
fn leaf_origin_at_local_point(
    leaf: &LeafProducer,
    local: [f32; 3],
    voxels_per_block: u32,
) -> LeafOrigin {
    leaf.producer
        .origin_at(local, voxels_per_block)
        .unwrap_or(leaf.origin)
}

/// Whether `leaf` occupies `absolute_voxel`, whose centre maps to `local` in the leaf's own
/// frame. Each branch below answers exactly as the emit path that owns that case does — and
/// crucially, each takes ITS OWN bound, because the two paths are bounded differently.
///
/// **Out of phase** (a genuine rotation or a sub-voxel seat): the field is sampled at the
/// mapped point, bounded by the placed WORLD box — the same cells
/// `gather_placed_field_into_grid` iterates, and the live classifier's
/// `gather_rotated_leaf_into_region` applies no bound of its own beyond the blocks it is
/// handed. The local `[0, full_dim)` box is NOT the same bound: a slid or turned body's
/// outermost cell centre can map exactly onto `full_dim` and still be on the surface, so
/// testing the local box would declare a stamped cell empty — dead clicks along one face of
/// every sub-voxel-seated body, and a cutter that fails to carve ownership at its own rim.
///
/// **In phase**: a lattice bijection, so the mapped point is an exact cell centre and a
/// one-cell `resolve_into` window answers as the forward emit does. Here the producer's own
/// `[0, full_dim)` grid IS the bound, because that is what the forward stamp resolves and
/// translates. A fieldless out-of-phase body falls through to this rather than being declined,
/// matching the resolvers, which have no guard either.
fn leaf_covers_cell(
    leaf: &LeafProducer,
    placement: &substrate::spatial::LeafPlacement,
    absolute_voxel: [i64; 3],
    local: [f32; 3],
    voxels_per_block: u32,
) -> bool {
    if leaf_is_out_of_phase(leaf.rotation, leaf.offset_local_voxels) {
        if let Some(field) = leaf.producer.as_field() {
            let (world_min, world_max) = placement.world_aabb();
            if (0..3).any(|axis| {
                absolute_voxel[axis] < world_min[axis] || absolute_voxel[axis] >= world_max[axis]
            }) {
                return false;
            }
            return field.signed_distance(local, voxels_per_block) <= SURFACE_ISOLEVEL;
        }
    }
    let full_dimensions = leaf.producer.full_dimensions(voxels_per_block);
    if (0..3).any(|axis| local[axis] < 0.0 || local[axis] >= full_dimensions[axis] as f32) {
        return false;
    }
    let local_index: [i64; 3] = std::array::from_fn(|axis| local[axis].floor() as i64);
    let mut cell = VoxelGrid::new([0, 0, 0]);
    leaf.producer.resolve_into(
        &mut cell,
        voxels_per_block,
        VoxelAabb::new(
            local_index,
            std::array::from_fn(|axis| local_index[axis] + 1),
        ),
    );
    !cell.occupied.is_empty()
}

/// The leaf's continuous world↔local affine, built from the already-outset-adjusted low
/// corner so no dilation is re-applied here. The rebase happens in i64 before any f32
/// rotation, so a leaf placed millions of voxels out keeps full sub-voxel precision (ADR 0008).
fn leaf_placement(leaf: &LeafProducer, voxels_per_block: u32) -> substrate::spatial::LeafPlacement {
    dense_leaf_placement(
        leaf.rotation,
        leaf.offset_local_voxels,
        leaf.world_offset_voxels,
        leaf.producer.as_ref(),
        voxels_per_block,
    )
}
