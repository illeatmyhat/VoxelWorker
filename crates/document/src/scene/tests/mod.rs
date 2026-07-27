use super::*;
use crate::voxel::SdfShape;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::{ShapeKind, VoxelGrid};

mod cutter_definitions;
mod fixture_definitions;
mod graph;
mod grids;
mod intersect;
mod leaf_origin;
mod operand_body;
mod pick;
mod placement;
mod resolve;
mod sealed_scopes;
mod subtract;

/// Mint stable [`NodeId`]s for a freshly-built test scene (ADR 0003 Phase B3), so a
/// fixture can name its nodes by id. ADR 0032 dropped the selection this used to set:
/// the document has none, and no document op reads one.
pub(super) fn with_minted_ids(mut scene: Scene) -> Scene {
    scene.ensure_node_ids();
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

/// The `--demo-scene` shape: a Sphere + an offset Box + an offset Torus, three
/// materials, top-level node 0 selected. Was rebuilt inline in resolve.rs and named in
/// placement.rs — one definition now.
pub(super) fn demo_three_tool_scene(voxels_per_block: u32) -> Scene {
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = with_minted_ids(Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
    ]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// The `--demo-village` scene: four `Instance`s of one `House` definition (a Box body +
/// a Cylinder chimney), top-level node 0 selected — proves instance/group transform
/// composition (reuse-by-reference). One definition, shared by resolve.rs + placement.rs.
pub(super) fn demo_village_scene(voxels_per_block: u32) -> Scene {
    let house_def_id = DefId(1);
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let instance = |name: &str, offset: [i64; 3]| {
        let mut node = Node::new(name, NodeContent::Instance(house_def_id));
        node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = Scene::from_nodes(vec![
        instance("House 1", [0, 0, 0]),
        instance("House 2", [6, 0, 0]),
        instance("House 3", [12, 0, 0]),
        instance("House 4", [18, 0, 0]),
    ]);
    scene.add_definition(
        house_def_id,
        "House".to_string(),
        vec![
            tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
            tool(
                ShapeKind::Cylinder,
                [1, 2, 1],
                [0, 2, 0],
                MaterialChoice::Wood,
            ),
        ],
    );
    scene.voxels_per_block = voxels_per_block;
    with_minted_ids(scene)
}
