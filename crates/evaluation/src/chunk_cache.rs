//! Relocated to [`crate::store`] in slice A2b. This module is now a thin
//! re-export shim so existing `chunk_cache::*` call sites keep compiling
//! until later slices migrate them. New code should use `crate::store`.
pub use crate::store::{ChunkCacheKey, ChunkResolveCache, Store};

// ADR 0016 Phase 2: these two scene↔cache equivalence proofs were relocated here from
// the document crate's `scene::tests` module. Each compares the dense
// `Scene::resolve_region` oracle (which lives DOWN in the document/truth layer) against
// the `ChunkResolveCache` reassembly (which lives HERE, in the evaluation layer) — so it
// straddles the crate boundary and cannot compile inside the truth crate, whose law
// forbids naming any evaluation type. The `Scene::resolve_region` oracle is reached via
// the app's dev-dependency on `document`'s `oracle` feature (test builds only).
#[cfg(test)]
mod scene_cache_equivalence_tests {
    use crate::store::ChunkResolveCache;
    use document::scene::{
        DefId, Node, NodeContent, NodePath, NodeTransform, Scene,
    };
    use document::voxel::{GeometryParams, SdfShape};
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::{ShapeKind, VoxelGrid};

    /// Select the top-level node at `index` (mirrors the document-crate test helper).
    fn scene_with_top_level_selected(mut scene: Scene, index: usize) -> Scene {
        scene.ensure_node_ids();
        scene.active = scene.id_at_path(&NodePath::root_index(index));
        scene
    }

    /// Build the review's parity-mismatched composite: Tool A `size [1,1,1] @ offset
    /// 0` + Tool B `size [2,1,1] @ offset +1 block` at density `vpb` — the exact
    /// X-axis parity mismatch (odd 1 vs even 2) the adversarial review caught.
    fn parity_mismatch_scene(vpb: u32) -> Scene {
        let mut node_a = Node::new(
            "A",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, vpb),
                material: MaterialChoice::Stone,
            },
        );
        node_a.transform = NodeTransform::from_blocks([0, 0, 0], vpb);
        let mut node_b = Node::new(
            "B",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 1, 1], 1, vpb),
                material: MaterialChoice::Wood,
            },
        );
        node_b.transform = NodeTransform::from_blocks([1, 0, 0], vpb);
        scene_with_top_level_selected(Scene::from_nodes(vec![node_a, node_b]), 0)
    }

    /// **Issue #20 S6c-1 equivalence proof.** `placed_region_dimensions(density)`
    /// is exactly the size the assembled render grid takes — both the monolithic
    /// `resolve_region` and the chunk-cache reassembly seed their output to it. So
    /// the camera / gizmo / lattice / floor-grid / layer-scrubber may read the
    /// region dimensions from the SCENE rather than from the assembled `VoxelGrid`,
    /// with zero behavioural change. This pins that substitution across every
    /// representative scene (all SDF shapes, flat/odd sizes, a placed multi-node
    /// scene, and an instanced village) for BOTH resolve paths.
    #[test]
    fn placed_region_dimensions_equals_assembled_grid() {
        let assert_equal = |scene: &Scene, vpb: u32, label: &str| {
            let from_scene = scene.placed_region_dimensions(vpb);

            // (1) The monolithic resolve_region (the initial-resolve path).
            let region = scene.full_extent_blocks(vpb);
            let monolithic = scene.resolve_region(region, vpb, 0);
            assert_eq!(
                from_scene, monolithic.dimensions,
                "[{label}] placed_region_dimensions must equal the monolithic assembled grid"
            );

            // (2) The chunk-cache reassembly (the live rebuild path).
            let mut cache = ChunkResolveCache::new();
            let assembled = cache.resolve_region(scene, vpb, 0);
            assert_eq!(
                from_scene, assembled.dimensions,
                "[{label}] placed_region_dimensions must equal the cache-assembled grid"
            );
        };

        // All SDF shapes at the app default density.
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = Scene::from_geometry(
                GeometryParams { shape: kind, size_voxels: [5 * 16, 5 * 16, 5 * 16], size_measurements: None, voxels_per_block: 16, wall_blocks: 1 },
                MaterialChoice::Stone,
            );
            assert_equal(&scene, 16, &format!("{kind:?}"));
        }

        // Flat / odd sizes (the 5×1×5 app default and friends), several densities.
        for vpb in [1u32, 8, 16] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams { shape: ShapeKind::Cylinder, size_voxels: [size[0] * vpb, size[1] * vpb, size[2] * vpb], size_measurements: None, voxels_per_block: vpb, wall_blocks: 1 },
                    MaterialChoice::Stone,
                );
                assert_equal(&scene, vpb, &format!("cylinder {size:?}@{vpb}"));
            }
        }

        // A placed multi-node scene (sphere at origin + box +8X + torus +6Z).
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, 16);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let demo_scene = scene_with_top_level_selected(Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]), 0);
        assert_equal(&demo_scene, 16, "demo-scene");

        // An instanced village (one house definition placed by four instances).
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, 16);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, 16);
            node
        };
        let mut village = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        village.add_definition(
            house_def_id,
            "House",
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        let village = scene_with_top_level_selected(village, 0);
        assert_equal(&village, 16, "demo-village");
    }

    /// THE BUG-CLASS MATRIX (corner-anchoring): across size ∈ {1,2,3,5,6} ×
    /// density ∈ {1,2,5,15,16}, for BOTH a single shape AND a 2-leaf mixed-parity
    /// composite, assert the four invariants that the old center-emit broke:
    ///
    /// (a) every occupied voxel CENTRE is a HALF-INTEGER (`fract()==0.5`) — on the
    ///     voxel lattice, inside a cell, for ANY size·d parity (the win: odd grids no
    ///     longer land on integers and straddle cell boundaries);
    /// (b) ZERO voxels dropped — occupied count == the expected filled-cell count;
    /// (c) every DECODED index is in `[0, dim)` (no clipped slab, none at `== dim`),
    ///     using the production decode `round(world + floor(dim/2) − 0.5)`;
    /// (d) the monolithic and chunk paths emit the IDENTICAL voxel set.
    ///
    /// Crucially this passes at ODD density (d ∈ {1,5,15}) and MIXED parity — the
    /// cases the center-emit convention could not represent.
    #[test]
    fn corner_anchoring_parity_matrix() {
        // Decode an occupied set to integer cell indices with the production rule.
        let decode_cells = |grid: &VoxelGrid| -> std::collections::BTreeSet<[i64; 3]> {
            let [dx, dy, dz] = grid.dimensions;
            let half = [(dx / 2) as f32, (dy / 2) as f32, (dz / 2) as f32];
            grid.occupied
                .iter()
                .map(|voxel| {
                    let position = voxel.world_position();
                    [
                        (position[0] + half[0] - 0.5).round() as i64,
                        (position[1] + half[1] - 0.5).round() as i64,
                        (position[2] + half[2] - 0.5).round() as i64,
                    ]
                })
                .collect()
        };
        // The exact f32-bit + material multiset (order-independent path comparison).
        let multiset = |grid: &VoxelGrid| {
            let mut set = std::collections::BTreeMap::<([u32; 3], u16), usize>::new();
            for voxel in &grid.occupied {
                let position = voxel.world_position();
                let key = (
                    [
                        position[0].to_bits(),
                        position[1].to_bits(),
                        position[2].to_bits(),
                    ],
                    voxel.color_index(),
                );
                *set.entry(key).or_insert(0) += 1;
            }
            set
        };

        // Run the four-invariant battery on one scene, returning its decoded cell set.
        let check = |scene: &Scene, vpb: u32, label: &str| -> std::collections::BTreeSet<[i64; 3]> {
            let dims = scene.placed_region_dimensions(vpb);
            let monolithic = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
            let mut cache = ChunkResolveCache::new();
            let assembled = cache.resolve_region(scene, vpb, 0);

            assert_eq!(monolithic.dimensions, dims, "[{label}] monolithic dims voxel-framed");
            assert_eq!(assembled.dimensions, dims, "[{label}] assembled dims voxel-framed");

            // (a) every centre is a half-integer.
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for axis in 0..3 {
                    assert_eq!(
                        position[axis].fract().abs(),
                        0.5,
                        "[{label}] centre {:?} axis {axis} must be a half-integer (on the lattice)",
                        position
                    );
                }
            }
            // (c) every decoded index is in [0, dim).
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for (axis, &dim) in dims.iter().enumerate() {
                    let half = (dim / 2) as f32;
                    let index = (position[axis] + half - 0.5).round() as i64;
                    assert!(
                        index >= 0 && index < dim as i64,
                        "[{label}] voxel {:?} axis {axis} decodes to {index} OUTSIDE [0, {dim})",
                        position
                    );
                }
            }
            // (d) the two paths emit the identical voxel set.
            assert_eq!(
                multiset(&monolithic),
                multiset(&assembled),
                "[{label}] monolithic and chunk paths must emit the identical voxel set"
            );
            assert!(!monolithic.occupied.is_empty(), "[{label}] non-empty");
            decode_cells(&monolithic)
        };

        for vpb in [1u32, 2, 5, 15, 16] {
            // --- single shape: a Box fully fills `size·d`³ cells, zero dropped (b). ---
            for size in [1u32, 2, 3, 5, 6] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: ShapeKind::Box,
                        size_voxels: [size * vpb, size * vpb, size * vpb],
                        size_measurements: None,
                        voxels_per_block: vpb,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                let label = format!("box {size}³ @ d{vpb}");
                let cells = check(&scene, vpb, &label);
                let expected = (size * vpb).pow(3) as usize;
                assert_eq!(
                    cells.len(), expected,
                    "[{label}] (b) zero dropped: distinct cells {} must equal size·d cubed {expected}",
                    cells.len()
                );
                let monolithic = scene.resolve_region(scene.full_extent_blocks(vpb), vpb, 0);
                assert_eq!(
                    monolithic.occupied_count(), expected,
                    "[{label}] (b) occupied count must equal the filled-cell count"
                );
            }

            // --- 2-leaf mixed-parity composite: A [1,1,1]@0 + B [2,1,1]@+1 block. ---
            let scene = parity_mismatch_scene(vpb);
            let label = format!("parity-composite @ d{vpb}");
            let cells = check(&scene, vpb, &label);
            // (b) distinct cells = |A| + |B| − overlap. A spans X[0,d), B spans
            // X[d, 3d) (off=1 block=d voxels, grid 2d) → DISJOINT on X (no overlap),
            // both full d×d in Y,Z. So distinct = d³ + 2d³ = 3d³.
            let d = vpb as i64;
            let expected_distinct = d * d * d + 2 * d * d * d;
            assert_eq!(
                cells.len() as i64, expected_distinct,
                "[{label}] (b) distinct occupied cells {} must equal |A|+|B| (disjoint) {expected_distinct}",
                cells.len()
            );
        }
    }
}
