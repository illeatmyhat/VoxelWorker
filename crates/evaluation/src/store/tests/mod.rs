//! Store residency, resolve, invalidation, spill, and rebuild-plan tests.

use document::scene::Scene;
use voxel_core::spatial_index::ChunkCoverage;

use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use document::voxel::GeometryParams;
    use document::scene::{
        DefId, Node, NodeContent, RegionBlocks,
    };
    use voxel_core::voxel::{ShapeKind, VoxelGrid};
    use document::voxel::{SdfShape};

mod core;
mod invalidation;
mod consumers;
mod incremental;
mod spill;

    /// Canonicalise an occupied set into a sorted multiset of
    /// `(bit_exact_voxel_position, material_id)`, so two resolves compare equal
    /// regardless of voxel emission ORDER but **byte-for-byte** on each `f32`
    /// position. Keying on the raw `f32` bits (`to_bits`) — not a rounded integer —
    /// means this asserts the bytes the renderer/mesher/fog consume are IDENTICAL,
    /// the S2 bit-identical-output guarantee (not merely the same rounded voxel
    /// set). A sub-ULP shift in any position fails the comparison.
    pub(super) fn occupied_multiset(grid: &VoxelGrid) -> std::collections::BTreeMap<([u32; 3], u16), usize> {
        let mut multiset = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
            ];
            *multiset.entry((key, voxel.color_index())).or_insert(0) += 1;
        }
        multiset
    }

    pub(super) fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    pub(super) fn three_tool_scene(voxels_per_block: u32, box_offset_x: i64) -> Scene {
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [box_offset_x, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        scene.voxels_per_block = voxels_per_block;
        scene
    }
