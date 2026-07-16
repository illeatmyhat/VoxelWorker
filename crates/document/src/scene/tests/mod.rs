use super::*;
use voxel_core::voxel::VoxelGrid;

mod graph;
mod resolve;
mod placement;
mod grids;
mod subtract;

/// Mint stable [`NodeId`]s for a freshly-built test scene and select the
/// top-level node at `index` by id (ADR 0003 Phase B3: selection is keyed by
/// [`NodeId`], so a fixture built with positional intent must resolve "select
/// node `index`" to that node's id after minting). Returns the scene with its
/// ids minted and the chosen node active — the id-era equivalent of the old
/// `active: Some(NodePath::root_index(index))` struct-literal fixtures.
pub(super) fn scene_with_top_level_selected(mut scene: Scene, index: usize) -> Scene {
    scene.ensure_node_ids();
    scene.active = scene
        .id_at_path(&NodePath::root_index(index));
    scene
}

/// Canonicalise an occupied set into a multiset of
/// `(absolute_voxel_index, material_id)` so two resolves can be compared as
/// the same shape regardless of voxel emission ORDER.
///
/// `recentre_voxels` translates the frame into ABSOLUTE composite space: pass
/// `[0,0,0]` for the chunked (already-absolute) frame, and the scene's
/// recentre for the monolithic frame (whose positions are `absolute −
/// recentre`). A voxel centre sits at an `n + 0.5` position, so `(p − 0.5)`
/// recovers the integer voxel index exactly.
pub(super) fn occupied_multiset(
    grid: &VoxelGrid,
    recentre_voxels: [i64; 3],
) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
    let mut multiset = std::collections::BTreeMap::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let key = [
            (position[0] - 0.5).round() as i64 + recentre_voxels[0],
            (position[1] - 0.5).round() as i64 + recentre_voxels[1],
            (position[2] - 0.5).round() as i64 + recentre_voxels[2],
        ];
        *multiset.entry((key, voxel.color_index())).or_insert(0) += 1;
    }
    multiset
}
