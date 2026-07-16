//! Two-layer chunk classifier parity, residency, incremental-edit, and stream tests.

use std::collections::BTreeMap;
use voxel_core::core_geom::CHUNK_BLOCKS;
use document::scene::{LeafProducer, Scene};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::VoxelGrid;

    use super::*;
    // The submodules the mod-level `pub(crate) use` glob does not re-export (their items are
    // reached only by the tests, not by non-test sibling code): the resident cache internals
    // and the stream / oracle functions.
    #[allow(unused_imports)]
    use super::resident_cache::*;
    #[allow(unused_imports)]
    use super::stream::*;
    use voxel_core::core_geom::MaterialChoice;
    use document::scene::{DefId, Node, NodeContent, NodeTransform};
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{GeometryParams, SdfShape};

mod core;
mod streaming;
mod incremental;
mod subtract;
mod sealed_scopes;
mod intersect;
mod fixture_definitions;

    pub(super) fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    pub(super) fn make_tool(kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    pub(super) fn make_tool_density(
        kind: ShapeKind,
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
        size_blocks: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, [size_blocks; 3], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }
