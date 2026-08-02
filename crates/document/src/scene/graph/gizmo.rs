#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::wildcard_imports
)]

//! The per-node transform gizmo's recentered placement: the pivot (center of the node
//! subtree's block-aligned AABB) and its extent.

use super::*;

impl Scene {
    /// The transform gizmo's placement for `node_id`, in the SAME recentered render
    /// frame the resolved voxels live in. Backs both the selection
    /// manipulator and the camera "Focus" view action (right-click a tree row → frame
    /// that node). `None` when the id no longer resolves or the node has no intrinsic
    /// extent (e.g. a lone VoxelBody with no size).
    ///
    /// Returns `(pivot_voxels, extent_voxels)`:
    /// * `pivot_voxels` — the **center** of the node's block-aligned AABB in the
    ///   recentered frame: `block_aabb_center · density − recenter_voxels`. The
    ///   gizmo is anchored here so it sits ON the object rather than at the
    ///   composite origin. (We chose the AABB center over the node's corner-origin
    ///   so a single-axis-offset child still reads as "on the object".)
    /// * `extent_voxels` — the node's own AABB size in voxels, so the gizmo is
    ///   sized from that node's extent (not the whole region).
    ///
    /// For a Group / Instance the AABB is the union of all leaves under it (the same
    /// union `placed_extent_blocks` forms scene-wide, but rooted at the node).
    /// Single-node scenes recenter that node onto the origin, so its pivot is
    /// `[0, 0, 0]` — the gizmo only visibly *moves* in a multi-node scene.
    #[must_use]
    pub fn gizmo_placement_for_id(
        &self,
        node_id: NodeId,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let path = self.path_of(node_id)?;
        self.gizmo_placement_at_path(&path, voxels_per_block)
    }

    /// Body of [`gizmo_placement_for_id`](Self::gizmo_placement_for_id): the recentered pivot
    /// (center of the node subtree's block-aligned AABB) + its extent, in voxels.
    fn gizmo_placement_at_path(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        // The gizmo PIVOT is the center of the node's PRODUCER-TRUE voxel AABB — the
        // exact frame the resolved voxels (and the composite recenter) live in. This
        // makes a lone node of ANY size (even or odd) recenter onto the origin: its
        // producer center coincides with the composite recenter. Mixing the
        // block-floored AABB center with the voxel recenter instead would leave odd
        // sizes half a block off.
        let (min_voxels, max_voxels) = self.node_subtree_extent_voxels(path, voxels_per_block)?;
        // The gizmo SIZE is the node's enclosing-whole-block extent (the visible box
        // snaps to whole blocks), taken from the block-AABB.
        let (min_blocks, max_blocks) = self.node_subtree_extent_blocks(path, voxels_per_block)?;
        let density = voxels_per_block.max(1) as i64;
        let mut pivot = [0.0f32; 3];
        let mut extent = [0.0f32; 3];
        // Unwrap the carried frame at the recentered pivot arithmetic.
        let recenter = self.recenter_voxels_for_resolve(voxels_per_block).voxels();
        for axis in 0..3 {
            // Producer-true voxel-AABB center minus the composite recenter — same
            // frame the resolved voxels sit in. `* 1` then `/ 2.0` last avoids a
            // half-voxel rounding bias on an odd voxel span.
            let center_voxels = min_voxels[axis] + max_voxels[axis];
            let pivot_voxels = center_voxels - 2 * recenter[axis];
            pivot[axis] = pivot_voxels as f32 / 2.0;
            extent[axis] = ((max_blocks[axis] - min_blocks[axis]) * density) as f32;
        }
        Some((pivot, extent))
    }
}
