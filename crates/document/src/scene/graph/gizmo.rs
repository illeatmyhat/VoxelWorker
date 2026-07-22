//! The per-selection transform gizmo's recentred placement (issue #29 S2): the
//! pivot (centre of the node subtree's block-aligned AABB) and extent, for the
//! active selection or an arbitrary node.

use super::*;

impl Scene {
    /// The transform gizmo's placement for the **active/selected** node, in the
    /// SAME recentred render frame the resolved voxels live in (issue #29 S2).
    /// `None` when nothing is selected (the gizmo is hidden) or the selection has
    /// no intrinsic extent (e.g. a lone VoxelBody with no size).
    ///
    /// Returns `(pivot_voxels, extent_voxels)`:
    /// * `pivot_voxels` — the **centre** of the node's block-aligned AABB in the
    ///   recentred frame: `block_aabb_centre · density − recentre_voxels`. The
    ///   gizmo is anchored here so it sits ON the object rather than at the
    ///   composite origin. (We chose the AABB centre over the node's corner-origin
    ///   so a single-axis-offset child still reads as "on the object".)
    /// * `extent_voxels` — the node's own AABB size in voxels, so the gizmo is
    ///   sized from the SELECTED node's extent (not the whole region).
    ///
    /// For a Group / Instance selection the AABB is the union of all leaves under
    /// it (the same union `placed_extent_blocks` forms scene-wide, but rooted at
    /// the selected node). Single-node scenes recentre that node onto the origin,
    /// so its pivot is `[0, 0, 0]` — the gizmo only visibly *moves* with a
    /// multi-node selection (which is the point of a per-selection manipulator).
    pub fn active_gizmo_placement(
        &self,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let path = self.active_path()?;
        self.gizmo_placement_at_path(&path, voxels_per_block)
    }

    /// The recentred `(pivot_voxels, extent_voxels)` for the node identified by
    /// `node_id` — the SAME computation as [`active_gizmo_placement`](Self::active_gizmo_placement)
    /// but scoped to an arbitrary node rather than the active selection. Used by the
    /// camera "Focus" view action (right-click a tree row → frame that node): the
    /// camera target is set to `pivot` and the distance fitted from `extent`.
    /// `None` when the id no longer resolves or the node's subtree has no extent.
    pub fn gizmo_placement_for_id(
        &self,
        node_id: NodeId,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        let path = self.path_of(node_id)?;
        self.gizmo_placement_at_path(&path, voxels_per_block)
    }

    /// Shared body of [`active_gizmo_placement`](Self::active_gizmo_placement) and
    /// [`gizmo_placement_for_id`](Self::gizmo_placement_for_id): the recentred pivot
    /// (centre of the node subtree's block-aligned AABB) + its extent, in voxels.
    fn gizmo_placement_at_path(
        &self,
        path: &NodePath,
        voxels_per_block: u32,
    ) -> Option<([f32; 3], [f32; 3])> {
        // The gizmo PIVOT is the centre of the node's PRODUCER-TRUE voxel AABB — the
        // exact frame the resolved voxels (and the composite recentre) live in. This
        // makes a lone node of ANY size (even or odd) recentre onto the origin: its
        // producer centre coincides with the composite recentre. (Center-anchoring
        // retirement: we no longer mix the block-floored AABB centre with the voxel
        // recentre, which left odd sizes half a block off.)
        let (min_voxels, max_voxels) = self.node_subtree_extent_voxels(path, voxels_per_block)?;
        // The gizmo SIZE is the node's enclosing-whole-block extent (the visible box
        // snaps to whole blocks), taken from the block-AABB.
        let (min_blocks, max_blocks) = self.node_subtree_extent_blocks(path, voxels_per_block)?;
        let density = voxels_per_block.max(1) as i64;
        let mut pivot = [0.0f32; 3];
        let mut extent = [0.0f32; 3];
        // Unwrap the carried frame at the recentred pivot arithmetic.
        let recentre = self.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for axis in 0..3 {
            // Producer-true voxel-AABB centre minus the composite recentre — same
            // frame the resolved voxels sit in. `* 1` then `/ 2.0` last avoids a
            // half-voxel rounding bias on an odd voxel span.
            let centre_voxels = min_voxels[axis] + max_voxels[axis];
            let pivot_voxels = centre_voxels - 2 * recentre[axis];
            pivot[axis] = pivot_voxels as f32 / 2.0;
            extent[axis] = ((max_blocks[axis] - min_blocks[axis]) * density) as f32;
        }
        Some((pivot, extent))
    }
}
