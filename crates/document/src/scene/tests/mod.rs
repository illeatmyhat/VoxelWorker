use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::{ShapeKind, VoxelGrid};
use crate::voxel::SdfShape;

mod graph;
mod resolve;
mod placement;
mod grids;
mod subtract;
mod sealed_scopes;
mod intersect;
mod cutter_definitions;
mod fixture_definitions;
mod operand_body;

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

/// The default authoring density the CSG-fixture scenes are built at (whole-block
/// boxes at density 8). Shared by the fixtures below; a child test module may still
/// declare its own `DENSITY` for scenes it builds at a different granularity.
pub(super) const DENSITY: u32 = 8;

/// A whole-block Box Tool of `size_blocks` at `offset_blocks` carrying `material` and
/// `operation` — the shared CSG fixture (axis-aligned boxes, so the expected occupied
/// set is exact). Was copy-pasted verbatim across the subtract / intersect / sealed /
/// cutter / fixture test modules; one definition now.
pub(super) fn box_tool(
    size_blocks: [u32; 3],
    offset_blocks: [i64; 3],
    material: MaterialChoice,
    operation: CombineOp,
) -> Node {
    let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, DENSITY);
    let mut node = Node::new("Box", NodeContent::Tool { shape, material });
    node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
    node.operation = operation;
    node
}

/// An [`NodeContent::Instance`] of `def_id` named `name`, at `offset_blocks` carrying
/// `operation` — the shared instance fixture (was duplicated in the cutter / fixture
/// definition test modules).
pub(super) fn instance_node(
    def_id: DefId,
    offset_blocks: [i64; 3],
    operation: CombineOp,
    name: &str,
) -> Node {
    let mut node = Node::new(name, NodeContent::Instance(def_id));
    node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
    node.operation = operation;
    node
}

/// Resolve `scene` through the dense oracle and return its occupancy multiset in
/// ABSOLUTE voxel space (recentre-normalised), keyed `(index, material)`. The shared
/// resolve-and-canonicalise fixture (was duplicated across the CSG test modules).
pub(super) fn resolved_absolute_multiset(
    scene: &Scene,
) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
    let grid = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
    occupied_multiset(&grid, scene.recentre_voxels(DENSITY))
}
