    use super::producers::leaf_producer_grid_voxels;
    use super::*;
    use crate::core_geom::MaterialChoice;
    use crate::debug_clouds::DebugCloudField;
    use crate::sketch::SketchSolid;
    use crate::spatial_index::VoxelAabb;
    use crate::units::{ExactRational, Measurement};
    use crate::voxel::{GeometryParams, SdfShape, ShapeKind, VoxelGrid, VoxelProducer};

    /// `from_measurements` derives the canonical voxel offset from the per-axis
    /// authored expression and retains the expression (ADR 0003 §3f(0)). A `3.5
    /// blocks` axis lands on `3.5 · d` voxels (56 at d16, 112 at d32 — the lossless
    /// parametric refine).
    #[test]
    fn transform_from_measurements_derives_voxels_and_retains_expression() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(-4),                           // -4 voxels
            Measurement::new(ExactRational::from_integer(2), 8),    // 2 blocks 8 voxels
        ];
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels, [56, -4, 40]);
        // The expression is retained verbatim.
        assert_eq!(at_sixteen.offset_measurements(), measurements);
        // The SAME measurements re-evaluate at a denser document (lossless refine).
        let at_thirty_two = NodeTransform::from_measurements(measurements, 32);
        assert_eq!(at_thirty_two.offset_voxels, [112, -4, 72]);
    }

    /// A retained NON-block-multiple offset (`3.5 blocks` on X = 56 vx at d16)
    /// re-evaluated at a NON-dividing density (d15, where 3.5·15 = 52.5) must not
    /// panic, floors X to a whole voxel, and keeps the retained measurement
    /// CONSISTENT with `offset_voxels` (the seam bug: they used to disagree). This
    /// is the lossy density-retarget path inside `from_measurements`.
    #[test]
    fn from_measurements_non_dividing_density_stays_self_consistent() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        // 3.5 blocks lands cleanly at d16 (= 56 voxels).
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels[0], 56);

        // Re-evaluate the SAME authored expression at the non-dividing d15.
        let at_fifteen = NodeTransform::from_measurements(at_sixteen.offset_measurements(), 15);
        // 3.5·15 = 52.5 → floored to 52 voxels (no panic).
        assert_eq!(at_fifteen.offset_voxels[0], 52);
        // The retained measurement now AGREES with the floored voxels: re-evaluating
        // it at d15 yields exactly offset_voxels[0] (no silent disagreement).
        let retained = at_fifteen.offset_measurements();
        assert_eq!(
            retained[0].to_voxels(15).unwrap(),
            at_fifteen.offset_voxels[0],
            "retained measurement must be consistent with the floored canonical voxels"
        );
    }

    /// A retained `3 blocks 8 voxels` (= 56 vx at d16) re-evaluated at the
    /// integer-multiple d32 keeps the VOXEL TERM EXACT: 3·32 + 8 = 104, NOT the
    /// integer rescale's 56·2 = 112. The authored expression is preserved.
    #[test]
    fn from_measurements_integer_multiple_density_keeps_voxel_term_exact() {
        let measurements = [
            Measurement::new(ExactRational::from_integer(3), 8), // 3 blocks 8 voxels
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        let at_sixteen = NodeTransform::from_measurements(measurements, 16);
        assert_eq!(at_sixteen.offset_voxels[0], 56);

        let at_thirty_two =
            NodeTransform::from_measurements(at_sixteen.offset_measurements(), 32);
        assert_eq!(
            at_thirty_two.offset_voxels[0], 104,
            "voxel term stays exact (3*32 + 8), NOT the integer rescale 112"
        );
        // The authored expression is preserved verbatim.
        assert_eq!(at_thirty_two.offset_measurements()[0], measurements[0]);
    }

    /// An OLD `NodeTransform` JSON that predates `offset_measurements` still
    /// deserialises (serde default → `None`), and the accessor SYNTHESISES a
    /// pure-voxel measurement equal to `offset_voxels` per axis — which
    /// re-evaluates back to exactly those voxels at any density (versioning:
    /// shared documents must load forward, ADR 0003 §3f(0)).
    #[test]
    fn transform_serde_back_compat_synthesises_measurements_from_voxels() {
        let old_json = r#"{ "offset_voxels": [48, -16, 7] }"#;
        let restored: NodeTransform =
            serde_json::from_str(old_json).expect("old transform without measurements must load");
        assert_eq!(restored.offset_voxels, [48, -16, 7]);
        let synthesised = restored.offset_measurements();
        for (axis, &voxels) in restored.offset_voxels.iter().enumerate() {
            assert_eq!(synthesised[axis], Measurement::from_voxels(voxels));
            assert_eq!(synthesised[axis].to_voxels(16).unwrap(), voxels);
            assert_eq!(synthesised[axis].to_voxels(32).unwrap(), voxels);
        }
    }

    /// A `NodeTransform` carrying retained measurements round-trips through serde
    /// unchanged (the new field persists for a forward-saved document).
    #[test]
    fn transform_serde_round_trips_with_retained_measurements() {
        let transform = NodeTransform::from_measurements(
            [
                Measurement::new(ExactRational::new(7, 2).unwrap(), 0),
                Measurement::from_voxels(-4),
                Measurement::new(ExactRational::from_integer(2), 8),
            ],
            16,
        );
        let json = serde_json::to_string(&transform).expect("serialises");
        let restored: NodeTransform = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(restored, transform);
        assert_eq!(restored.offset_measurements(), transform.offset_measurements());
        assert_eq!(restored.offset_voxels, transform.offset_voxels);
    }

    /// The identical-behaviour guarantee (ADR 0001 step 1): a one-node Tool scene
    /// resolved over the node's full extent yields the SAME occupied count as
    /// calling `SdfShape::resolve` directly — and the same grid dimensions.
    #[test]
    fn tool_scene_matches_bare_producer() {
        let geometry = GeometryParams {
            shape: ShapeKind::Sphere,
            size_voxels: [6 * 16, 6 * 16, 6 * 16],
            size_measurements: None,
            voxels_per_block: 16,
            wall_blocks: 1,
        };

        // Bare producer (today's path).
        let shape = SdfShape::from_geometry(geometry.clone());
        let mut bare = VoxelGrid::new(shape.grid_dimensions(geometry.voxels_per_block));
        shape.resolve(&mut bare, geometry.voxels_per_block);

        // Through the scene.
        let scene = Scene::from_geometry(geometry.clone(), MaterialChoice::Stone);
        let region = scene.full_extent_blocks(geometry.voxels_per_block);
        let resolved = scene.resolve_region(region, geometry.voxels_per_block, 0);

        assert_eq!(
            resolved.dimensions, bare.dimensions,
            "scene grid dimensions must match the bare producer"
        );
        assert_eq!(
            resolved.occupied_count(),
            bare.occupied_count(),
            "scene occupied count must match the bare producer"
        );
    }

    /// **Issue #20 S6c-1 equivalence proof.** `placed_region_dimensions(density)`
    /// is exactly the size the assembled render grid takes — both the monolithic
    /// [`resolve_region`] and the chunk-cache reassembly seed their output to it. So
    /// the camera / gizmo / lattice / floor-grid / layer-scrubber may read the
    /// region dimensions from the SCENE rather than from the assembled `VoxelGrid`,
    /// with zero behavioural change. This pins that substitution across every
    /// representative scene (all SDF shapes, flat/odd sizes, a placed multi-node
    /// scene, and an instanced village) for BOTH resolve paths.
    #[test]
    fn placed_region_dimensions_equals_assembled_grid() {
        use crate::chunk_cache::ChunkResolveCache;

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
        use crate::chunk_cache::ChunkResolveCache;

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

    /// The same guarantee for a Part (the debug cloud field): a one-node Part
    /// scene matches `DebugCloudField::resolve` at the same dimensions. Step 2
    /// builds the Part node directly (the `debug_clouds` selector is gone).
    #[test]
    fn part_scene_matches_bare_cloud_field() {
        let size_blocks = [4u32, 4, 4];
        let voxels_per_block = 16u32;
        let dimensions = [
            size_blocks[0] * voxels_per_block,
            size_blocks[1] * voxels_per_block,
            size_blocks[2] * voxels_per_block,
        ];
        let bare_field = DebugCloudField {
            dimensions,
            seed: 0,
        };
        let mut bare = VoxelGrid::new(dimensions);
        bare_field.resolve(&mut bare, voxels_per_block);

        let scene =
            Scene::single_node(Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 0 })));
        let region = RegionBlocks::new(size_blocks);
        let resolved = scene.resolve_region(region, voxels_per_block, 0);

        assert_eq!(resolved.dimensions, bare.dimensions);
        assert_eq!(resolved.occupied_count(), bare.occupied_count());
    }

    /// CORNER-ANCHORING (cloud producer): a PART-ONLY cloud at ODD density drops ZERO
    /// voxels — every occupied centre is a HALF-INTEGER (on the voxel lattice) and
    /// every decoded index ∈ [0, dim). This is the case center-emit broke: at an odd
    /// region dim the centred bottom voxel decoded to index −1 and was dropped.
    /// Tested at d=1 and d=5 (odd densities → odd region dims for an odd block size).
    #[test]
    fn part_only_cloud_at_odd_density_drops_no_voxels() {
        // 5×5×5 blocks at odd density → region dims 5·d (odd). A 64-vx field has plenty
        // of solid voxels so the boundary cells are genuinely exercised.
        for (size_blocks, vpb) in [([5u32, 5, 5], 1u32), ([5, 5, 5], 5)] {
            let scene = Scene::single_node(Node::new(
                "Clouds",
                NodeContent::Part(Part::DebugClouds { seed: 7 }),
            ));
            let region = RegionBlocks::new(size_blocks);
            let dims = [
                size_blocks[0] * vpb,
                size_blocks[1] * vpb,
                size_blocks[2] * vpb,
            ];
            let label = format!("part-only cloud {size_blocks:?}@d{vpb}");

            let monolithic = scene.resolve_region(region, vpb, 0);
            assert_eq!(monolithic.dimensions, dims, "[{label}] dims = region·d");
            assert!(!monolithic.occupied.is_empty(), "[{label}] non-empty cloud");

            // (a) every centre is a half-integer; (c) every decoded index ∈ [0, dim).
            // A Part-only cloud is corner-anchored at the explicit region (low corner 0,
            // recentre 0), so the decode is `floor(world)`.
            let mut decoded = 0usize;
            for voxel in &monolithic.occupied {
                let position = voxel.world_position();
                for (axis, &dim) in dims.iter().enumerate() {
                    let pos = position[axis];
                    assert_eq!(
                        pos.fract().abs(), 0.5,
                        "[{label}] centre {pos} axis {axis} must be a half-integer"
                    );
                    let index = pos.floor() as i64;
                    assert!(
                        index >= 0 && index < dim as i64,
                        "[{label}] voxel {pos} axis {axis} decodes to {index} OUTSIDE [0, {dim})"
                    );
                }
                decoded += 1;
            }
            assert_eq!(
                decoded, monolithic.occupied_count(),
                "[{label}] every emitted voxel decodes in-range (no silent drop)"
            );

            // A Part-only scene has no chunkable extent, so the monolithic path above
            // IS the resolve path (the chunk reassembly is for Tool-bearing scenes).
            assert!(
                !scene.has_chunkable_extent(vpb),
                "[{label}] a Part-only cloud has no chunkable extent"
            );
        }
    }

    /// CORNER-ANCHORING (mixed frame): a Tool and a Cloud in the SAME scene resolve in
    /// ONE frame — the cloud's voxels are NOT offset by `region_dim/2` from the Tool.
    /// Center-emit broke this: the Tool corner-anchored but the cloud center-emitted,
    /// so they sat in different frames. Now BOTH corner-anchor at `[0, region_dim)`, so
    /// a Tool placed at offset 0 and the region-filling cloud share the same low corner.
    #[test]
    fn mixed_tool_and_cloud_resolve_in_one_frame() {
        // A Box Tool at offset 0 (size 3³) plus a Cloud. The Tool's voxel span and the
        // cloud's region span share the SAME low corner in the resolved frame.
        let vpb = 4u32;
        let mut tool = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb),
                material: MaterialChoice::Stone,
            },
        );
        tool.transform = NodeTransform::from_blocks([0, 0, 0], vpb);
        let cloud = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 3 }));
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![tool, cloud]),
            0,
        );

        // The Tool gives the scene a chunkable extent; the region is its voxel span.
        let region = scene.full_extent_blocks(vpb);
        let dims = scene.placed_region_dimensions(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        assert_eq!(grid.dimensions, dims, "region is the Tool's voxel-framed span");

        // Decode in the recentred frame (low corner −floor(dim/2)). EVERY voxel —
        // whether from the Tool or the Cloud — must decode to an index in [0, dim) with
        // a half-integer centre. If the cloud were still center-emitting it would be
        // offset by ~region_dim/2 and a slab would decode out of range.
        let recentre = scene.recentre_voxels_for_resolve(vpb).voxels();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            for (axis, &dim) in dims.iter().enumerate() {
                let pos = position[axis];
                assert_eq!(
                    pos.fract().abs(), 0.5,
                    "mixed scene: centre {pos} axis {axis} must be a half-integer (same frame)"
                );
                let half = (dim / 2) as f32;
                let index = (pos + half - 0.5).round() as i64;
                assert!(
                    index >= 0 && index < dim as i64,
                    "mixed scene: voxel {pos} axis {axis} decodes to {index} OUTSIDE [0, {dim}) \
                     — a cloud offset by region_dim/2 would land here"
                );
            }
        }

        // The Tool's voxels land EXACTLY where corner-anchored math says: a 3³ box at
        // offset 0 occupies absolute `[0, 3d)`; recentred, its low corner is
        // `0 − recentre`. At least one voxel sits at that low corner (the box fully
        // fills its AABB). This pins the cloud sharing the Tool's frame, not an offset.
        let expected_low = [
            (0 - recentre[0]) as f32 + 0.5,
            (0 - recentre[1]) as f32 + 0.5,
            (0 - recentre[2]) as f32 + 0.5,
        ];
        let has_box_low_corner = grid.occupied.iter().any(|v| {
            let position = v.world_position();
            (position[0] - expected_low[0]).abs() < 1e-3
                && (position[1] - expected_low[1]).abs() < 1e-3
                && (position[2] - expected_low[2]).abs() < 1e-3
        });
        assert!(
            has_box_low_corner,
            "the corner-anchored Box must place a voxel at its recentred low corner {expected_low:?}"
        );
    }

    /// ADR 0001 step 2: several leaf nodes composite into one region under union.
    /// A 2-node scene (a sphere Tool + a box Tool, both centred at origin) yields
    /// the SET-UNION of their occupied voxels: the union count is at least each
    /// node alone, and exactly equals the union of the two single-node sets.
    #[test]
    fn two_node_scene_resolves_to_union() {
        let voxels_per_block = 12u32;
        let region = RegionBlocks::new([6, 6, 6]);

        let sphere = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [6, 6, 6], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        // A full-extent box: its corners poke outside the inscribed sphere, so the
        // union is strictly larger than the sphere alone (a real composite).
        let cube = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [6, 6, 6], 1, voxels_per_block),
                material: MaterialChoice::Wood,
            },
        );

        // Each node resolved alone.
        let sphere_only = Scene::single_node(sphere.clone())
            .resolve_region(region, voxels_per_block, 0);
        let cube_only =
            Scene::single_node(cube.clone()).resolve_region(region, voxels_per_block, 0);

        // Both nodes composited.
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![sphere, cube]), 0);
        let union = scene.resolve_region(region, voxels_per_block, 0);

        // The expected set-union of the two single-node occupied sets, keyed by
        // integer voxel position (the producers emit voxel-centre world positions).
        use std::collections::HashSet;
        let key = |grid: &VoxelGrid| -> HashSet<[i64; 3]> {
            grid.occupied
                .iter()
                .map(|voxel| {
                    let position = voxel.world_position();
                    [
                        position[0].round() as i64,
                        position[1].round() as i64,
                        position[2].round() as i64,
                    ]
                })
                .collect()
        };
        let sphere_set = key(&sphere_only);
        let cube_set = key(&cube_only);
        let union_set = key(&union);
        let expected: HashSet<[i64; 3]> = sphere_set.union(&cube_set).copied().collect();

        // Union is at least as occupied as either node alone …
        assert!(union_set.len() >= sphere_set.len());
        assert!(union_set.len() >= cube_set.len());
        // … and equals the set-union exactly (the box pokes outside the sphere, so
        // the union is strictly larger than the sphere alone — a real composite).
        assert_eq!(union_set, expected);
        assert!(union_set.len() > sphere_set.len());
    }

    /// ADR 0001 step 3 (per-voxel material): a Tool with `MaterialChoice::Wood`
    /// stamps voxels whose `material_id` equals the Wood id (1) — every voxel it
    /// emits carries the Tool's single material, so distinct nodes are distinct.
    #[test]
    fn wood_tool_stamps_wood_material_id() {
        let voxels_per_block = 8u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
        let scene = Scene::single_node(Node::new(
            "Wood box",
            NodeContent::Tool { shape, material: MaterialChoice::Wood },
        ));
        let grid = scene.resolve_region(RegionBlocks::new([2, 2, 2]), voxels_per_block, 0);
        let wood_id = MaterialChoice::Wood.material_id();
        assert!(grid.occupied_count() > 0, "the box must emit voxels");
        assert!(
            grid.occupied.iter().all(|voxel| voxel.color_index() == wood_id),
            "every voxel a Wood Tool stamps must carry the Wood material id"
        );
    }

    /// ADR 0001 step 3 (per-voxel material): a 2-Tool scene (Stone + Wood, placed
    /// disjointly) yields BOTH material ids present — proving the per-voxel id
    /// travels through compositing so the two nodes render in distinct materials.
    #[test]
    fn two_material_scene_has_both_material_ids() {
        let voxels_per_block = 8u32;
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut stone = Node::new("Stone", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        stone.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut wood = Node::new("Wood", NodeContent::Tool { shape: base, material: MaterialChoice::Wood });
        wood.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![stone, wood]), 0);
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        assert_ne!(stone_id, wood_id, "Stone and Wood must map to distinct ids");
        assert!(
            grid.occupied.iter().any(|voxel| voxel.color_index() == stone_id),
            "the Stone node's voxels must carry the Stone id"
        );
        assert!(
            grid.occupied.iter().any(|voxel| voxel.color_index() == wood_id),
            "the Wood node's voxels must carry the Wood id"
        );
    }

    /// Issue #29 S4 (per-object on-face grid): the resolver ORs
    /// [`crate::voxel::GRID_OVERLAY_BIT`] into a node's stamped `material_id`
    /// **iff** that node's `grids.voxel_grid_on_faces` is set — and the masked
    /// material id still round-trips to the real handle (≤2). Parametrized over
    /// density {1, 15, 16} so the bit survives every density's chunk bucketing.
    #[test]
    fn voxel_grid_flag_bit_set_iff_node_opts_in() {
        for &voxels_per_block in &[1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
            let wood_id = MaterialChoice::Wood.material_id();

            // Node with the on-face grid ON → every voxel carries the flag bit, and
            // the masked id is still the real Wood handle (the bit never corrupts it).
            let mut on = Node::new(
                "On",
                NodeContent::Tool { shape: shape.clone(), material: MaterialChoice::Wood },
            );
            on.grids.voxel_grid_on_faces = true;
            let scene = Scene::single_node(on);
            let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
            assert!(grid.occupied_count() > 0);
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| v.grid_overlay),
                "density {voxels_per_block}: a node with voxel_grid_on_faces must flag every voxel"
            );
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| v.color_index() == wood_id),
                "density {voxels_per_block}: the colour index must round-trip to Wood (≤2)"
            );

            // Same node with the flag OFF → no voxel carries the bit (the default).
            let off = Node::new(
                "Off",
                NodeContent::Tool { shape, material: MaterialChoice::Wood },
            );
            let scene = Scene::single_node(off);
            let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
            assert!(grid.occupied_count() > 0);
            assert!(
                grid.occupied
                    .iter()
                    .all(|v| !v.grid_overlay),
                "density {voxels_per_block}: a node WITHOUT the flag must leave the bit clear"
            );
        }
    }

    /// Issue #29 S4: in a 2-node scene with the on-face grid enabled on ONE node
    /// only, exactly that node's voxels carry the flag bit; the other node's don't —
    /// the per-object gating the headless capture verifies. Also confirms the bit
    /// travels through the chunked resolve path (`resolve_chunk`) identically.
    #[test]
    fn voxel_grid_flag_bit_is_per_object() {
        let voxels_per_block = 8u32;
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        // Stone node opts IN; Wood node opts OUT, placed disjointly.
        let mut stone = Node::new(
            "Stone",
            NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone },
        );
        stone.grids.voxel_grid_on_faces = true;
        let wood = Node::new(
            "Wood",
            NodeContent::Tool { shape: base, material: MaterialChoice::Wood },
        );
        let mut wood = wood;
        wood.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![stone, wood]), 0);
        let region = scene.full_extent_blocks(voxels_per_block);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        let stone_id = MaterialChoice::Stone.material_id();
        let wood_id = MaterialChoice::Wood.material_id();
        // Every flagged voxel is a Stone voxel; every Wood voxel is unflagged.
        let stone_flagged = grid
            .occupied
            .iter()
            .filter(|v| v.color_index() == stone_id)
            .all(|v| v.grid_overlay);
        let wood_unflagged = grid
            .occupied
            .iter()
            .filter(|v| v.color_index() == wood_id)
            .all(|v| !v.grid_overlay);
        assert!(stone_flagged, "the enabled (Stone) node's voxels must all be flagged");
        assert!(wood_unflagged, "the disabled (Wood) node's voxels must all be unflagged");
        assert!(
            grid.occupied.iter().any(|v| v.grid_overlay),
            "at least one voxel (the Stone node's) must carry the flag"
        );
    }

    /// A hidden node contributes nothing.
    #[test]
    fn hidden_node_stamps_nothing() {
        let mut node = Node::new(
            "Shape",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 8),
                material: MaterialChoice::Stone,
            },
        );
        node.visible = false;
        let scene = Scene::single_node(node);
        let resolved = scene.resolve_region(RegionBlocks::new([2, 2, 2]), 8, 0);
        assert_eq!(resolved.occupied_count(), 0);
    }

    /// A box Tool sized to fill a single block (so the whole block of voxels is
    /// occupied), at the given block offset along X, in a wide region. Returns the
    /// set of occupied voxel positions keyed to integer coordinates.
    fn boxed_block_positions(offset_x: i64, voxels_per_block: u32) -> std::collections::HashSet<[i64; 3]> {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks([offset_x, 0, 0], voxels_per_block);
        // A region wide enough to hold the offset box without clipping.
        let region = RegionBlocks::new([8, 1, 1]);
        let grid = Scene::single_node(node).resolve_region(region, voxels_per_block, 0);
        grid.occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    position[0].round() as i64,
                    position[1].round() as i64,
                    position[2].round() as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 3 (a): a node placed at a whole-block offset `[N, 0, 0]` places
    /// its voxels shifted by exactly `N × voxels_per_block` in X versus offset 0.
    ///
    /// A two-node scene (a 1-block box at offset 0 and an identical box at offset
    /// N, far enough apart to be disjoint) shares ONE composite recentre, so the
    /// only difference between the two boxes' positions is the N-block placement.
    /// The occupied set splits into two equal clusters whose X-spans are exactly
    /// `N × voxels_per_block` apart; shifting one cluster by that amount reproduces
    /// the other.
    #[test]
    fn offset_node_shifts_voxels_by_blocks_times_density() {
        let voxels_per_block = 8u32;
        let n = 5i64; // 5 blocks apart: a 1-block box leaves a 4-block gap (disjoint).
        let region = RegionBlocks::new([8, 1, 1]);
        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut at_zero = Node::new("A", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        at_zero.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut at_n = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        at_n.transform = NodeTransform::from_blocks([n, 0, 0], voxels_per_block);

        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![at_zero, at_n]), 0);
        let grid = scene.resolve_region(region, voxels_per_block, 0);

        // Key each voxel by its EXACT world position (the producers emit voxel-
        // centre positions; the placement is an exact integer-voxel translation, so
        // float comparison is safe and exact — no rounding). The boxes are disjoint
        // in X (5 blocks apart, 1 block wide), so the occupied set splits cleanly at
        // the gap between box A's X-run and box B's X-run.
        let shift = (n * voxels_per_block as i64) as f32; // N blocks → N×density voxels.
        let key = |position: [f32; 3]| -> [i64; 3] {
            // Bit-exact key: positions are k+0.5 half-integers, so ×2 is an exact
            // integer and avoids any float-equality fragility in the HashSet.
            [
                (position[0] * 2.0) as i64,
                (position[1] * 2.0) as i64,
                (position[2] * 2.0) as i64,
            ]
        };

        // The composite centre lies between the two boxes; split there.
        let mut xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position()[0]).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let split_x = (xs.first().unwrap() + xs.last().unwrap()) / 2.0;

        let cluster_low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] < split_x)
            .map(|v| key(v.world_position()))
            .collect();
        let cluster_high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] >= split_x)
            .map(|v| key(v.world_position()))
            .collect();

        assert!(!cluster_low.is_empty() && !cluster_high.is_empty(), "both boxes present");
        assert_eq!(cluster_low.len(), cluster_high.len(), "both boxes fill one block");
        // Shifting the low box by exactly N×density in X reproduces the high box.
        let shifted: std::collections::HashSet<[i64; 3]> = cluster_low
            .iter()
            .map(|c| [c[0] + (shift * 2.0) as i64, c[1], c[2]])
            .collect();
        assert_eq!(shifted, cluster_high, "offset N blocks shifts voxels by exactly N×density");
    }

    /// ADR 0001 step 3 (b): two nodes at non-overlapping offsets give an occupied
    /// count equal to the SUM of each alone (a disjoint union — the placement
    /// genuinely separates them in space, no longer overlapping at the origin).
    #[test]
    fn disjoint_offsets_give_summed_occupancy() {
        let voxels_per_block = 8u32;
        // Two 1-block boxes 5 blocks apart in X — far enough that their voxel sets
        // never touch (each is 1 block = 8 voxels wide, gap is 4 empty blocks).
        let a_alone = boxed_block_positions(0, voxels_per_block).len();
        let b_alone = boxed_block_positions(5, voxels_per_block).len();
        assert!(a_alone > 0 && b_alone > 0);

        let base = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut a = Node::new("A", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        a.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut b = Node::new("B", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        b.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);

        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![a, b]), 0);
        // Region spans the full composite (offset 0..5, each 1 block) → 6 blocks X.
        let region = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(region.size_blocks, [6, 1, 1], "composite extent encompasses both offsets");
        let grid = scene.resolve_region(region, voxels_per_block, 0);
        assert_eq!(
            grid.occupied_count(),
            a_alone + b_alone,
            "disjoint placement → occupied count is the sum (no overlap)"
        );
    }

    /// ADR 0001 step 3 (c): `full_extent_blocks` grows to encompass an offset node.
    /// A single 2-block box pushed +4 blocks in X spans blocks `[3, 5]` in X (centre
    /// 4, ±1), so the composite X extent is 6 blocks (`0..6` once recentred), while
    /// Y/Z stay at the box's 2 blocks. (A zero-offset single node would be just the
    /// box's own 2×2×2.)
    #[test]
    fn full_extent_encompasses_offset_node() {
        let voxels_per_block = 4u32;
        let base = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
        let mut node = Node::new("Box", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks([4, 0, 0], voxels_per_block);
        let scene = Scene::single_node(node);

        // The box centred at block 4 with half-size 1 spans X blocks [3, 5] → its
        // own size (2) is unchanged but its placement means the bounding box from
        // the origin is wider. `full_extent_blocks` returns the box SIZE of the
        // composite: for a single node that is just the node's own size in every
        // axis (the offset moves it but doesn't enlarge a single box). To prove the
        // extent ACCOUNTS for the offset, compare against a two-node scene where the
        // offset opens a real gap.
        let single = scene.full_extent_blocks(voxels_per_block);
        assert_eq!(single.size_blocks, [2, 2, 2], "a lone offset box keeps its own size");

        // Add a second box at the origin: now the composite must span from the
        // origin box (blocks [-1, 1]) to the offset box (blocks [3, 5]) → X width 6.
        let mut origin_box =
            Node::new("Origin", NodeContent::Tool { shape: base.clone(), material: MaterialChoice::Stone });
        origin_box.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut offset_box =
            Node::new("Offset", NodeContent::Tool { shape: base, material: MaterialChoice::Stone });
        offset_box.transform = NodeTransform::from_blocks([4, 0, 0], voxels_per_block);
        let two = scene_with_top_level_selected(Scene::from_nodes(vec![origin_box, offset_box]), 0);
        let extent = two.full_extent_blocks(voxels_per_block);
        assert_eq!(
            extent.size_blocks,
            [6, 2, 2],
            "the offset node widens the composite extent in X from 2 to 6 blocks"
        );
    }

    /// A 1×1×1 box Tool shape, used as a leaf in the step-4 recursion/instancing
    /// tests (the node carries the material; the shape does not).
    fn unit_box_shape() -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 8)
    }

    /// Key a grid's occupied voxels by exact half-integer voxel position (×2 → an
    /// exact integer, no float-equality fragility). Used to compare voxel SETS.
    fn position_keys(grid: &VoxelGrid) -> std::collections::HashSet<[i64; 3]> {
        grid.occupied
            .iter()
            .map(|v| {
                let position = v.world_position();
                [
                    (position[0] * 2.0) as i64,
                    (position[1] * 2.0) as i64,
                    (position[2] * 2.0) as i64,
                ]
            })
            .collect()
    }

    /// ADR 0001 step 4 (nested transform composition): a leaf inside a `Group`
    /// offset by `+A` blocks, with the leaf itself offset `+B`, lands at world
    /// `A + B` (× density). We compare the grouped scene against a FLAT scene whose
    /// single node sits directly at `A + B` — same composite, so the recentre is
    /// identical and the voxel sets must match exactly.
    #[test]
    fn nested_group_composes_transforms_down() {
        let voxels_per_block = 8u32;
        let region = RegionBlocks::new([10, 1, 1]);
        let a = 3i64; // group offset
        let b = 2i64; // leaf offset within the group

        // Grouped: a Group at +A containing a box at +B.
        let mut leaf = Node::new(
            "Leaf",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        );
        leaf.transform = NodeTransform::from_blocks([b, 0, 0], voxels_per_block);
        let grouped = scene_with_top_level_selected(
            Scene::from_nodes(vec![NodeBuilder::group_at("Group", [a, 0, 0], voxels_per_block, vec![leaf.into()])]),
            0,
        );
        let grouped_grid = grouped.resolve_region(region, voxels_per_block, 0);

        // Flat reference: the same box placed directly at A + B.
        let mut flat_leaf = Node::new(
            "Flat",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        );
        flat_leaf.transform = NodeTransform::from_blocks([a + b, 0, 0], voxels_per_block);
        let flat = Scene::single_node(flat_leaf);
        let flat_grid = flat.resolve_region(region, voxels_per_block, 0);

        assert!(grouped_grid.occupied_count() > 0, "the grouped leaf must emit voxels");
        assert_eq!(
            position_keys(&grouped_grid),
            position_keys(&flat_grid),
            "a leaf at +B inside a Group at +A must land at world A+B (× density)"
        );
    }

    /// ADR 0001 step 4 (instancing): an `Instance` of a 1-node definition placed at
    /// offset `T` resolves to the SAME voxels as that node placed directly at `T`.
    #[test]
    fn instance_matches_direct_placement() {
        let voxels_per_block = 8u32;
        let region = RegionBlocks::new([10, 1, 1]);
        let t = 4i64;
        let def_id = DefId(7);

        let mut instance = Node::new("I", NodeContent::Instance(def_id));
        instance.transform = NodeTransform::from_blocks([t, 0, 0], voxels_per_block);
        let mut instanced_scene = Scene::from_nodes(vec![instance]);
        // Definition: a single box at the origin (within the def).
        instanced_scene.add_definition(
            def_id,
            "Body".to_string(),
            vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
            )],
        );
        let instanced = scene_with_top_level_selected(instanced_scene, 0);
        let instanced_grid = instanced.resolve_region(region, voxels_per_block, 0);

        // Direct: the same box placed directly at T.
        let mut direct = Node::new(
            "Direct",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
        );
        direct.transform = NodeTransform::from_blocks([t, 0, 0], voxels_per_block);
        let direct_grid = Scene::single_node(direct).resolve_region(region, voxels_per_block, 0);

        assert!(instanced_grid.occupied_count() > 0, "the instance must emit voxels");
        assert_eq!(
            position_keys(&instanced_grid),
            position_keys(&direct_grid),
            "an Instance of a 1-node def at T equals that node placed directly at T"
        );
    }

    /// ADR 0001 step 4 (village): a 2-instance scene (the SAME def placed at two
    /// different offsets) yields `occupied_count == 2 × the def's own count`, at two
    /// DISJOINT locations (the two voxel clusters never overlap).
    #[test]
    fn two_instance_village_doubles_occupancy_disjointly() {
        let voxels_per_block = 8u32;
        let def_id = DefId(2);

        // The "house": a single 1-block box (so its count is easy to reason about).
        let house_body = || {
            vec![Node::new(
                "Box",
                NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
            )]
        };

        // The def's own occupied count (resolved alone at the origin).
        let mut def_only_scene =
            Scene::from_nodes(vec![Node::new("I", NodeContent::Instance(def_id))]);
        def_only_scene.add_definition(def_id, "House".to_string(), house_body());
        let def_only = scene_with_top_level_selected(def_only_scene, 0);
        let def_count = def_only
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(def_count > 0);

        // Two instances 6 blocks apart in X (a 1-block house → 5-block gap: disjoint).
        let mut house_a = Node::new("A", NodeContent::Instance(def_id));
        house_a.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let mut house_b = Node::new("B", NodeContent::Instance(def_id));
        house_b.transform = NodeTransform::from_blocks([6, 0, 0], voxels_per_block);
        let mut village_scene = Scene::from_nodes(vec![house_a, house_b]);
        village_scene.add_definition(def_id, "House".to_string(), house_body());
        let village = scene_with_top_level_selected(village_scene, 0);
        let region = village.full_extent_blocks(voxels_per_block);
        let grid = village.resolve_region(region, voxels_per_block, 0);

        assert_eq!(
            grid.occupied_count(),
            2 * def_count,
            "two disjoint instances of one def → 2× the def's voxel count"
        );

        // Disjoint: split the occupied set at the composite centre; each half is a
        // full house, and the two halves share no voxel position.
        let xs: Vec<f32> = grid.occupied.iter().map(|v| v.world_position()[0]).collect();
        let split_x = (xs.iter().cloned().fold(f32::MAX, f32::min)
            + xs.iter().cloned().fold(f32::MIN, f32::max))
            / 2.0;
        let low: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] < split_x)
            .map(|v| { let p = v.world_position(); [(p[0] * 2.0) as i64, (p[1] * 2.0) as i64, (p[2] * 2.0) as i64] })
            .collect();
        let high: std::collections::HashSet<[i64; 3]> = grid
            .occupied
            .iter()
            .filter(|v| v.world_position()[0] >= split_x)
            .map(|v| { let p = v.world_position(); [(p[0] * 2.0) as i64, (p[1] * 2.0) as i64, (p[2] * 2.0) as i64] })
            .collect();
        assert_eq!(low.len(), def_count, "the low cluster is one full house");
        assert_eq!(high.len(), def_count, "the high cluster is one full house");
        assert!(low.is_disjoint(&high), "the two houses occupy disjoint locations");
    }

    /// ADR 0001 step 4 (cycle guard): a definition that instances ITSELF resolves
    /// without stack overflow. The self-instance is skipped on re-entry, so the def
    /// contributes only its non-cyclic leaves finitely (here: one box) — never
    /// infinitely.
    #[test]
    fn self_referential_definition_does_not_overflow() {
        let voxels_per_block = 8u32;
        let def_id = DefId(1);

        let mut scene_build =
            Scene::from_nodes(vec![Node::new("Root", NodeContent::Instance(def_id))]);
        // A definition whose children are (a) a real box leaf and (b) an Instance of
        // ITSELF — the cycle the guard must break.
        scene_build.add_definition(
            def_id,
            "Recursive".to_string(),
            vec![
                Node::new(
                    "Box",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
                ),
                Node::new("Self", NodeContent::Instance(def_id)),
            ],
        );
        let scene = scene_with_top_level_selected(scene_build, 0);

        // Resolves (no overflow) and contributes the single box ONCE — the self-
        // instance is skipped, so the count is finite and equals one box's voxels.
        let grid = scene.resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0);
        let one_box = Scene::single_node(Node::new(
            "Box",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        ))
        .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
        .occupied_count();
        assert_eq!(
            grid.occupied_count(),
            one_box,
            "a self-instancing def contributes its leaves finitely (cycle skipped)"
        );
    }

    /// Mint stable [`NodeId`]s for a freshly-built test scene and select the
    /// top-level node at `index` by id (ADR 0003 Phase B3: selection is keyed by
    /// [`NodeId`], so a fixture built with positional intent must resolve "select
    /// node `index`" to that node's id after minting). Returns the scene with its
    /// ids minted and the chosen node active — the id-era equivalent of the old
    /// `active: Some(NodePath::root_index(index))` struct-literal fixtures.
    fn scene_with_top_level_selected(mut scene: Scene, index: usize) -> Scene {
        scene.ensure_node_ids();
        scene.active = scene
            .id_at_path(&NodePath::root_index(index));
        scene
    }

    /// A small flat scene of two box Tools, the first selected — the fixture the
    /// tree-mutation UI helper tests build on. ADR 0003 Phase B3: ids are minted so
    /// the selection (and the `group_active` it drives) resolves by identity.
    fn two_box_scene(voxels_per_block: u32) -> Scene {
        let mut scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                Node::new(
                    "A",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
                ),
                Node::new(
                    "B",
                    NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Wood },
                ),
            ]),
            0,
        );
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    /// ADR 0001 step 4 (UI helper): `group_active` wraps the active node in a new
    /// Group, so the active node becomes a CHILD of that Group. After grouping, the
    /// top-level node at the old slot is a `Group` whose sole child is the original
    /// node, and the active selection points at that child (path `[0, 0]`).
    #[test]
    fn group_active_nests_node_under_new_group() {
        let mut scene = two_box_scene(8);
        // Node "A" (top-level 0) is the active selection; remember its stable id so
        // we can confirm the wrap keeps that SAME node selected by identity.
        let node_a_id = scene.id_at_path(&NodePath::root_index(0)).expect("A has an id");
        assert_eq!(scene.active, Some(node_a_id));

        let group_id = scene.group_active().expect("there is an active node to group");
        // B4: `group_active` now returns the new Group's stable id; it resolves to
        // the old top-level slot the Group took (path [0]).
        assert_eq!(
            scene.path_of(group_id),
            Some(NodePath::root_index(0)),
            "the Group takes the old slot"
        );

        // The top-level node is now a Group with exactly one child (the old "A").
        match &scene.root_node(0).content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1, "the Group holds exactly the wrapped node");
                assert_eq!(
                    scene.arena[&children[0]].name, "A",
                    "the wrapped child is the original node"
                );
            }
            other => panic!("expected a Group at slot 0, got {other:?}"),
        }
        // The wrapped child is still the active selection — by identity it is the
        // SAME node "A", now living at path [0, 0] inside the new Group.
        assert_eq!(scene.active, Some(node_a_id), "the wrapped node stays selected by id");
        assert_eq!(
            scene.active_path(),
            Some(NodePath::from_indices(vec![0, 0])),
            "the selection now resolves to the child slot inside the Group"
        );
        // The second node is untouched.
        assert_eq!(scene.roots.len(), 2);
        assert!(matches!(scene.root_node(1).content, NodeContent::Tool { .. }));
    }

    /// ADR 0001 step 4 (UI helper): `make_definition_from_active` creates an
    /// `AssemblyDef` in `scene.definitions` and replaces the active node with an
    /// `Instance` of it. The resolved occupancy is unchanged (one stored body
    /// resolved via one instance == the original single node).
    #[test]
    fn make_definition_creates_def_and_instance() {
        let voxels_per_block = 8u32;
        // The fixture already selects top-level node 0 (by id).
        let mut scene = two_box_scene(voxels_per_block);

        // Occupancy of just the active node before the change (resolved alone).
        let before = Scene::single_node(scene.root_node(0).clone())
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(before > 0);

        let def_id = scene
            .make_definition_from_active("House")
            .expect("there is an active node to define");

        // A definition now exists, named, with the node's body as its children.
        assert_eq!(scene.definitions.len(), 1, "a definition appears in scene.definitions");
        let def = scene.def_by_id(def_id).expect("the new def is looked up by id");
        assert_eq!(def.name, "House");
        assert_eq!(def.children.len(), 1, "a single leaf becomes a one-node body");

        // The former node is now an Instance of that def.
        assert!(matches!(scene.root_node(0).content, NodeContent::Instance(id) if id == def_id));

        // Resolving the (now-instanced) node reproduces the original occupancy.
        // Reuse the live scene's arena + definitions, keeping only the single root.
        let mut after_scene = scene.clone();
        let kept_root = after_scene.roots[0];
        after_scene.roots = vec![kept_root];
        after_scene.active = Some(kept_root);
        let after = after_scene
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert_eq!(after, before, "an instance of the def equals the original node");
    }

    /// ADR 0001 step 4 (UI helper, the village): after `make_definition_from_active`,
    /// `add_instance` appends another `Instance` node referencing the SAME def, and
    /// the scene resolves with the EXPECTED MULTIPLIED occupancy — two disjoint
    /// instances of a one-box def give 2× the box's voxel count.
    #[test]
    fn add_instance_multiplies_occupancy_via_one_definition() {
        let voxels_per_block = 8u32;
        // Start from a single box node, make it a definition (→ one instance), then
        // add a second instance.
        let mut scene = Scene::single_node(Node::new(
            "House",
            NodeContent::Tool { shape: unit_box_shape(), material: MaterialChoice::Stone },
        ));
        let def_id = scene.make_definition_from_active("House").expect("active node");
        assert_eq!(scene.definitions.len(), 1);
        assert_eq!(scene.roots.len(), 1, "the original node became one instance");

        // The def's own voxel count (one box).
        let one = scene
            .resolve_region(RegionBlocks::new([1, 1, 1]), voxels_per_block, 0)
            .occupied_count();
        assert!(one > 0);

        // Add a second instance — an Instance node referencing the same def appears.
        // B4: `add_instance` now returns the new node's stable id; resolve it by id.
        let instance_id = scene.add_instance(def_id).expect("the def exists");
        assert_eq!(scene.roots.len(), 2, "an Instance node referencing the def appears");
        assert!(matches!(
            scene.node_by_id(instance_id).map(|n| &n.content),
            Some(NodeContent::Instance(id)) if *id == def_id
        ));
        // Still exactly ONE stored definition (reuse by reference).
        assert_eq!(scene.definitions.len(), 1, "the body is stored once, not copied");

        // The two instances are placed disjointly (add_instance nudges +X), so the
        // scene resolves to 2× the def's occupancy.
        let region = scene.full_extent_blocks(voxels_per_block);
        let total = scene.resolve_region(region, voxels_per_block, 0).occupied_count();
        assert_eq!(total, 2 * one, "two instances of one def → 2× the def's voxel count");
    }

    /// ADR 0001 step 4 (UI helper): `tree_rows` flattens the assembly depth-first,
    /// a parent immediately preceding its Group children at increasing depth, so the
    /// tree UI can render an indented list with selectable child nodes.
    #[test]
    fn tree_rows_lists_group_children_indented() {
        let mut scene = two_box_scene(8);
        // Group node A, then add a child into the Group, so the tree is:
        //   [0]          Group           depth 0
        //   [0, 0]         A (wrapped)    depth 1
        //   [0, 1]         child          depth 1
        //   [1]          B                depth 0
        // Node 0 ("A") is already the active selection (the fixture selects it).
        let group_id = scene.group_active().expect("active node");
        let added = scene.add_child_to_group(
            group_id,
            Node::new("child", NodeContent::Part(Part::DebugClouds { seed: 0 })),
        );
        assert!(added, "the wrapped node is a Group so a child can be added");

        let rows = scene.tree_rows();
        let paths: Vec<(Vec<usize>, usize)> =
            rows.iter().map(|(p, _id, d)| (p.indices.clone(), *d)).collect();
        assert_eq!(
            paths,
            vec![
                (vec![0], 0),    // Group
                (vec![0, 0], 1), // wrapped A
                (vec![0, 1], 1), // added child
                (vec![1], 0),    // B
            ],
            "tree_rows is depth-first with Group children indented under their parent"
        );
    }

    /// Selecting a node by path reaches a Group child (not just top-level nodes) —
    /// the inspector can therefore edit a node at any depth.
    #[test]
    fn node_at_path_reaches_group_child() {
        // Node 0 ("A") is already the active selection (the fixture selects it).
        let mut scene = two_box_scene(8);
        scene.group_active();
        // The active selection now resolves to the wrapped child at path [0, 0].
        let active_path = scene
            .active_path()
            .expect("a child is selected after grouping");
        assert_eq!(active_path, NodePath::from_indices(vec![0, 0]));
        let node = scene.node_at_path(&active_path).expect("the child resolves by path");
        assert_eq!(node.name, "A", "the path reaches the wrapped child node");
    }

    // ---- S0: chunk-addressable resolve (issue #27) ---------------------------
    //
    // These tests prove the ADDITIVE chunked resolve path reconstructs EXACTLY
    // what the monolithic `resolve_region` produces, after normalising for the
    // recentre offset that `resolve_region` applies and the chunk path does not.
    // The render path (`resolve_region`) is untouched; only these new functions
    // are exercised.

    /// Canonicalise an occupied set into a multiset of
    /// `(absolute_voxel_index, material_id)` so two resolves can be compared as
    /// the same shape regardless of voxel emission ORDER.
    ///
    /// `recentre_voxels` translates the frame into ABSOLUTE composite space: pass
    /// `[0,0,0]` for the chunked (already-absolute) frame, and the scene's
    /// recentre for the monolithic frame (whose positions are `absolute −
    /// recentre`). A voxel centre sits at an `n + 0.5` position, so `(p − 0.5)`
    /// recovers the integer voxel index exactly.
    fn occupied_multiset(
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

    /// Assert the chunk-reassembled occupied set EXACTLY equals the monolithic
    /// `resolve_region`'s set (position + material), after recentre normalisation,
    /// AND that no chunk emits a voxel outside its own chunk AABB.
    fn assert_chunked_matches_monolithic(scene: &Scene, voxels_per_block: u32, label: &str) {
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let chunked = scene.resolve_region_via_chunks(voxels_per_block, 0);

        let recentre = scene.recentre_voxels(voxels_per_block);
        let monolithic_set = occupied_multiset(&monolithic, recentre);
        let chunked_set = occupied_multiset(&chunked, [0, 0, 0]);

        assert_eq!(
            chunked_set, monolithic_set,
            "[{label}] chunked occupied set must equal monolithic resolve (recentre-normalised)"
        );
        // Cross-check the count too (a multiset equality already implies it, but
        // this pins the failure message to the simplest symptom first).
        assert_eq!(
            chunked.occupied_count(),
            monolithic.occupied_count(),
            "[{label}] chunked occupied count must equal monolithic"
        );

        // Each per-chunk resolve must keep every voxel inside its OWN chunk AABB
        // (exactly-one-chunk ownership). Walk the covering range and re-resolve.
        let chunk_extent_voxels =
            (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as i32;
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            let mut total_from_chunks = 0usize;
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        let chunk_coord = [chunk_x, chunk_y, chunk_z];
                        let chunk = scene.resolve_chunk(chunk_coord, voxels_per_block, 0);
                        total_from_chunks += chunk.occupied_count();
                        for voxel in &chunk.occupied {
                            let world_position = voxel.world_position();
                            for axis in 0..3 {
                                let lo = (chunk_coord[axis] * chunk_extent_voxels) as f32;
                                let hi = lo + chunk_extent_voxels as f32;
                                let position = world_position[axis];
                                assert!(
                                    position >= lo && position < hi,
                                    "[{label}] voxel {position} on axis {axis} escaped chunk \
                                     {chunk_coord:?} box [{lo}, {hi})"
                                );
                            }
                        }
                    }
                }
            }
            // Every monolithic voxel is accounted for by exactly one chunk (no
            // double-counting, no drops): the chunk total equals the whole count.
            assert_eq!(
                total_from_chunks,
                monolithic.occupied_count(),
                "[{label}] summed per-chunk counts must equal the monolithic count \
                 (each voxel in exactly one chunk)"
            );
        }
    }

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
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

    /// Single-shape parity, all five SDF kinds — mirrors the all-shapes coverage
    /// style. (Single-node zero-offset scenes also exercise the recentre
    /// normalisation, since `resolve_region` recentres even a lone node.)
    #[test]
    fn chunked_resolve_matches_monolithic_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_chunked_matches_monolithic(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// A multi-node placed scene (the `--demo-scene` shape: a Sphere + an offset
    /// Box + an offset Torus, three materials) — proves the chunked path composes
    /// several leaves at distinct offsets and materials.
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_scene() {
        let voxels_per_block = 16;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ]),
            0,
        );
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    /// The `--demo-village` scene: four `Instance`s of one `House` definition (a
    /// Box body + a Cylinder chimney `Group`) — proves the chunked path follows
    /// instance + group transform composition (reuse-by-reference).
    #[test]
    fn chunked_resolve_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
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
        let mut scene_build = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene_build.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        let scene = scene_with_top_level_selected(scene_build, 0);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// ADR 0003 §3i Slice 2a: the new sketch→extrude producer composes through the
    /// chunked resolve identically to the monolithic one — mirrors the SDF parity
    /// harness for a SketchTool leaf. Two cases: a plain rectangle extrude (the box
    /// sugar) and a concave L-shape extrude (the added-value path), both at the app
    /// density and at an off-origin placement so the recentre/cover math is real.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_extrude() {
        use crate::sketch::{PlaneAxis, Sketch, SketchPoint};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // (a) Rectangle extrude (box sugar), placed off-origin on X. Z-up:
        // footprint-extrude-up uses PlaneAxis::Z (profile in XY, extruded along +Z).
        let rect = SketchSolid::extrude(
            Sketch::rectangle(PlaneAxis::Z, 3 * density, 2 * density),
            2 * density as u32,
        );
        let mut rect_node = Node::new(
            "Sketch rect",
            NodeContent::SketchTool {
                producer: rect,
                material: MaterialChoice::Stone,
            },
        );
        rect_node.transform = NodeTransform::from_blocks([5, 0, 0], voxels_per_block);
        let rect_scene = Scene::single_node(rect_node);
        assert_chunked_matches_monolithic(&rect_scene, voxels_per_block, "sketch-rect");

        // (b) Concave L-shape extrude (the added value a box can't make).
        let two = 2 * density;
        let four = 4 * density;
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(four, 0),
            SketchPoint::new(four, two),
            SketchPoint::new(two, two),
            SketchPoint::new(two, four),
            SketchPoint::new(0, four),
        ];
        let l_extrude =
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 3 * density as u32);
        let mut l_node = Node::new(
            "Sketch L",
            NodeContent::SketchTool {
                producer: l_extrude,
                material: MaterialChoice::Wood,
            },
        );
        // Off-origin (crossing chunk boundaries on both in-plane axes X and Y) so the
        // off-origin chunked path is proven on the concave/reflex shape, not just the
        // convex rectangle above. (Z-up: the L footprint lives in the XY ground plane.)
        l_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let l_scene = Scene::single_node(l_node);
        assert_chunked_matches_monolithic(&l_scene, voxels_per_block, "sketch-L");
    }

    /// ADR 0003 §3i: the revolve operation composes through the chunked resolve
    /// identically to the monolithic one — mirrors the extrude parity harness for a
    /// solid of revolution. A rectangle revolved 360° about Z (a cylinder) placed
    /// off-origin on X+Y so the recentre/cover math is real and the disc crosses
    /// chunk boundaries on both radial axes.
    #[test]
    fn chunked_resolve_matches_monolithic_for_sketch_revolve() {
        use crate::sketch::{PlaneAxis, RevolveAxis, Sketch};
        let voxels_per_block = 16;
        let density = voxels_per_block as i64;

        // PlaneAxis::X + RevolveAxis::InPlane1 ⇒ axial = Z (vertical), radial = {X, Y}.
        // (a) Profile (radial, axial) = rectangle(radial = 2 blocks, axial = 3 blocks)
        // ⇒ a 4-block-diameter, 3-block-tall cylinder. EVEN radial + whole-block axial.
        let revolve = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 2 * density, 3 * density),
            RevolveAxis::InPlane1,
            360,
        );
        let mut node = Node::new(
            "Sketch revolve",
            NodeContent::SketchTool {
                producer: revolve,
                material: MaterialChoice::Stone,
            },
        );
        // Off-origin so the covering chunk range and recentre offset are non-trivial.
        node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let scene = Scene::single_node(node);
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "sketch-revolve");

        // (b) ODD axial extent (NOT a whole number of blocks) with an even radial, so
        // the even-radial diameter + odd-axial block-rounding combo is exercised through
        // the chunked path. Radial 30 voxels (diameter 60), axial 2·16 + 5 = 37 voxels.
        let revolve_odd_axial = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 30, 2 * density + 5),
            RevolveAxis::InPlane1,
            360,
        );
        let mut odd_node = Node::new(
            "Sketch revolve odd axial",
            NodeContent::SketchTool {
                producer: revolve_odd_axial,
                material: MaterialChoice::Wood,
            },
        );
        odd_node.transform = NodeTransform::from_blocks([5, 5, 0], voxels_per_block);
        let odd_scene = Scene::single_node(odd_node);
        assert_chunked_matches_monolithic(&odd_scene, voxels_per_block, "sketch-revolve-odd-axial");
    }

    /// A scene with a single node shifted well OFF the origin (+8 blocks on X) —
    /// proves the chunked path handles off-centre placement (the AABB does not
    /// start at the origin, so the covering chunk range is non-trivial and the
    /// recentre offset is non-zero).
    #[test]
    fn chunked_resolve_matches_monolithic_for_offset_node() {
        let voxels_per_block = 16;
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Offset sphere",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        node.transform = NodeTransform::from_blocks([8, 0, 0], voxels_per_block);
        let scene = Scene::single_node(node);

        // Sanity: the recentre is genuinely non-zero for this off-centre scene, so
        // the normalisation is actually exercised (a zero recentre would make the
        // test vacuous on that axis).
        let recentre = scene.recentre_voxels(voxels_per_block);
        assert_ne!(
            recentre, [0, 0, 0],
            "an off-centre node must produce a non-zero recentre (else the \
             normalisation is untested)"
        );
        assert_chunked_matches_monolithic(&scene, voxels_per_block, "offset-node");
    }

    /// A chunk that no leaf overlaps resolves to an EMPTY grid (no panic), and its
    /// dimensions are still one chunk's extent.
    #[test]
    fn empty_chunk_resolves_to_empty_grid() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        // A chunk far outside the (origin-area) composite AABB.
        let chunk = scene.resolve_chunk([1000, 1000, 1000], 16, 0);
        assert_eq!(chunk.occupied_count(), 0, "a far-off chunk must be empty");
        let chunk_extent = crate::core_geom::CHUNK_BLOCKS * 16;
        assert_eq!(
            chunk.dimensions,
            [chunk_extent, chunk_extent, chunk_extent],
            "an empty chunk still reports one chunk's voxel extent"
        );
    }

    /// Parity holds at a non-default density too (16 is the app default; this pins
    /// that the chunk-extent / ownership math is density-correct).
    #[test]
    fn chunked_resolve_matches_monolithic_at_density_8() {
        let scene = shape_scene(ShapeKind::Torus, 8);
        assert_chunked_matches_monolithic(&scene, 8, "torus@8");
    }

    // ---- S1: far-offset placement (ADR 0002 streaming, part of #18) -----------
    //
    // The durable artifact for streaming S1: a node placed at a LARGE block offset
    // (matching `shot --demo-far-offset`'s 100_000 blocks) really lands far away in
    // ABSOLUTE composite space, independent of the live render recentre. This is
    // proved via the S0 absolute-coordinate chunk path (`resolve_chunk` /
    // `resolve_region_via_chunks`), which — unlike `resolve_region` — does NOT
    // recentre, so its voxel positions ARE the scene's true composite coordinates.
    //
    // A node's whole-block offset is `[i64; 3]` (widened in S4a); 100_000 blocks is comfortably
    // in i32 range too, and at density 16 lands the box ~1.6M voxels out. The
    // BEYOND-i32 composition (offsets past ±2.1×10⁹) is proven separately in
    // `i64_composition_beyond_i32_range_is_exact` (pure integer, no f32 precision
    // loss).

    /// A far-offset node resolves to absolute voxel/chunk coordinates around
    /// 100_000 blocks: the box's voxels sit at absolute X ≈ 100_000 × density, the
    /// owning chunks are around `100_000 × density / chunk_extent`, and the box is
    /// genuinely placed far away (the absolute coords are NOT near the origin —
    /// only the recentred render path maps it home). Independent of any render math.
    #[test]
    fn far_offset_node_resolves_to_absolute_coords_near_100k() {
        let voxels_per_block = 16u32;
        let offset_blocks = 100_000i64;
        // A 4³ box — the same recognizable shape `shot --demo-far-offset` builds.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks([offset_blocks, 0, 0], voxels_per_block);
        let scene = Scene::single_node(node);

        // The ABSOLUTE-coordinate chunk path (no recentre): these positions are the
        // scene's TRUE composite coordinates, so they reveal the far placement that
        // the render recentre hides.
        let absolute = scene.resolve_region_via_chunks(voxels_per_block, 0);
        assert!(
            absolute.occupied_count() > 0,
            "the far box must resolve to voxels"
        );

        // CORNER-ANCHORING: every voxel's absolute X centre lands in the far block's
        // voxel span. The 4-block box CORNER-ANCHORED at block 100_000 spans blocks
        // [100_000, 100_004), i.e. absolute voxels [100_000·d, 100_004·d). (Y/Z start
        // at 0.) The box's geometric centre is `off·d + 2·d`.
        let density = voxels_per_block as f32;
        let span_lo = offset_blocks as f32 * density;
        let span_hi = (offset_blocks + 4) as f32 * density;
        let expected_centre_voxels = (offset_blocks as f32 + 2.0) * density; // 1_600_032
        for voxel in &absolute.occupied {
            let x = voxel.world_position()[0];
            assert!(
                x >= span_lo && x < span_hi,
                "far-box voxel X={x} must lie in the absolute span [{span_lo}, {span_hi}) \
                 around 100_000 blocks — NOT near the origin"
            );
        }
        // The box is genuinely ~1.6M voxels out (sanity: not collapsed to origin).
        assert!(
            expected_centre_voxels > 1_000_000.0,
            "at density {voxels_per_block}, 100_000 blocks is >1M voxels from the origin"
        );

        // Mean absolute X is within half a block of the far centre (the box is
        // symmetric about block 100_000), confirming the placement, not the recentre.
        let mean_x: f64 = absolute
            .occupied
            .iter()
            .map(|v| v.world_position()[0] as f64)
            .sum::<f64>()
            / absolute.occupied_count() as f64;
        assert!(
            (mean_x - expected_centre_voxels as f64).abs() <= (density / 2.0) as f64,
            "the far box's mean absolute X ({mean_x}) must sit at ~{expected_centre_voxels} \
             voxels (block 100_000 × density), proving far placement in absolute space"
        );

        // The owning chunk coordinates are around 100_000 × density / chunk_extent,
        // i.e. far from chunk 0 — the chunk addressing places it far away too.
        let chunk_extent_voxels =
            (crate::core_geom::CHUNK_BLOCKS * voxels_per_block) as i64;
        let expected_chunk_x = ((offset_blocks * voxels_per_block as i64) / chunk_extent_voxels) as i32;
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("the far box has an intrinsic size → a covering chunk range");
        assert!(
            min_chunk[0] <= expected_chunk_x && expected_chunk_x <= max_chunk[0],
            "the far box's owning chunk-X range [{}, {}] must bracket chunk {expected_chunk_x} \
             (≈100_000 blocks out), not chunk 0",
            min_chunk[0],
            max_chunk[0]
        );
        assert!(
            min_chunk[0] > 1000,
            "the far box must be owned by a high chunk coordinate (>1000), proving it is \
             far from the origin in chunk space (got {})",
            min_chunk[0]
        );

        // Cross-check: the ABSOLUTE chunk path and the RECENTRED render path agree
        // on the box's SHAPE — they differ ONLY by the recentre offset, which is
        // exactly the far placement. This pins that the render recentre is what maps
        // the far box home (and is the exact thing S4 will remove), while the
        // absolute path keeps it far.
        let recentre = scene.recentre_voxels(voxels_per_block);
        assert_eq!(
            recentre[0],
            offset_blocks * voxels_per_block as i64 + 2 * voxels_per_block as i64,
            "CORNER-ANCHORING: the recentre is the box's geometric CENTRE `off·d + 2·d` \
             (corner `off·d` + half the 4-block extent) — it is what hides the far \
             offset from the live render today (S4 removes it)"
        );
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        assert_eq!(
            occupied_multiset(&monolithic, recentre),
            occupied_multiset(&absolute, [0, 0, 0]),
            "the recentred render box and the absolute far box are the SAME shape, \
             offset by exactly the recentre (the far placement)"
        );
    }

    /// S4a (64-bit world addressing): nested transforms compose down the tree in
    /// **i64**, so a leaf whose accumulated block offset exceeds the `i32` range
    /// lands at the EXACT absolute coordinate — no overflow, no truncation. This is
    /// the load-bearing data-model guarantee of S4a, proven in PURE INTEGER space
    /// (the producer-true voxel AABB from `build_leaf_spatial_index`) so there is no
    /// f32 precision loss to muddy the result.
    ///
    /// A Group offset `+2_000_000_000` blocks contains a leaf offset `+1_000_000_000`
    /// blocks; their sum `3_000_000_000` is past `i32::MAX` (2_147_483_647). The
    /// composed absolute-voxel centre must be `3_000_000_000 × density` — a value
    /// that would have wrapped to a negative number under the old i32 composition.
    #[test]
    fn i64_composition_beyond_i32_range_is_exact() {
        let voxels_per_block = 16u32;
        let density = voxels_per_block as i64;
        let group_offset: i64 = 2_000_000_000; // ~i32::MAX on its own
        let leaf_offset: i64 = 1_000_000_000;
        let composed_blocks = group_offset + leaf_offset; // 3e9 — past i32::MAX
        assert!(
            composed_blocks > i32::MAX as i64,
            "the composed offset must exceed i32 range to exercise 64-bit addressing"
        );

        // A 1-block box leaf inside a Group; the leaf carries +leaf_offset, the Group
        // +group_offset, so the leaf's world offset composes to their sum.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let mut leaf = Node::new(
            "Leaf",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        leaf.transform = NodeTransform::from_blocks([leaf_offset, 0, 0], voxels_per_block);
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![NodeBuilder::group_at(
                "Group",
                [group_offset, 0, 0],
                voxels_per_block,
                vec![leaf.into()],
            )]),
            0,
        );

        // CORNER-ANCHORING: the producer-true voxel AABB (pure i64) is `[off·d,
        // off·d + grid)` — the composed offset IS the low corner (block-aligned for a
        // whole-block offset). The point of THIS test is the exact i64 composition
        // (no i32 overflow).
        let index = scene.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(index.entries.len(), 1, "exactly one leaf is indexed");
        let aabb = index.entries[0].world_aabb;
        let composed_voxels = composed_blocks * density; // 48_000_000_000 — past i32 too
        assert_eq!(
            aabb.min[0], composed_voxels,
            "the composed leaf min-X must equal (group+leaf)·d (corner-anchored), \
             with NO i32 overflow (got {}, want {})",
            aabb.min[0], composed_voxels
        );
        assert_eq!(
            aabb.max[0], composed_voxels + density,
            "the composed leaf max-X must be exact in i64"
        );
        // Sanity: this absolute voxel coordinate genuinely exceeds the i32 range, so
        // the test would have FAILED (wrapped negative) under i32 composition.
        assert!(
            composed_voxels > i32::MAX as i64,
            "the absolute voxel coordinate ({composed_voxels}) is past i32::MAX — the \
             exact case 64-bit addressing exists to handle"
        );

        // The covering chunk range also derives correctly (chunk coord narrows to i32
        // safely): chunk-X = composed_voxels / chunk_extent, well inside i32.
        let chunk_extent = (crate::core_geom::CHUNK_BLOCKS as i64) * density;
        let expected_chunk_x = composed_voxels.div_euclid(chunk_extent);
        assert!(
            expected_chunk_x <= i32::MAX as i64,
            "the derived chunk coordinate stays inside i32 even for a 3e9-block offset"
        );
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("the far leaf has an intrinsic size");
        assert!(
            (min_chunk[0] as i64) <= expected_chunk_x && expected_chunk_x <= (max_chunk[0] as i64),
            "the covering chunk-X range must bracket the composed chunk {expected_chunk_x}"
        );
    }

    // ===== Issue #27 S3: leaf spatial index =====================================

    /// The ground-truth leaf set a query AABB selects: a FULL `for_each_leaf` walk,
    /// recomputing each leaf's producer-true voxel AABB inline (the same maths
    /// `build_leaf_spatial_index` uses), filtered by overlap with `query`. The
    /// spatial index must return exactly this set; that equality is the S3
    /// correctness contract.
    fn walk_leaf_aabbs_intersecting(
        scene: &Scene,
        voxels_per_block: u32,
        query: &crate::spatial_index::VoxelAabb,
    ) -> Vec<crate::spatial_index::VoxelAabb> {
        let mut matched = Vec::new();
        scene.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces| {
            let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block) else {
                return; // region-spanning leaf — not an AABB match.
            };
            let mut min = [0i64; 3];
            let mut max = [0i64; 3];
            for axis in 0..3 {
                // Corner-anchored span `[off, off + grid)` (offset is the low corner),
                // the same maths `build_leaf_spatial_index` now uses.
                let grid = grid_voxels[axis];
                min[axis] = world_offset_voxels[axis];
                max[axis] = min[axis] + grid;
            }
            let aabb = crate::spatial_index::VoxelAabb::new(min, max);
            if aabb.intersects(query) {
                matched.push(aabb);
            }
        });
        matched
    }

    fn sorted_aabbs(
        mut boxes: Vec<crate::spatial_index::VoxelAabb>,
    ) -> Vec<([i64; 3], [i64; 3])> {
        boxes.sort_by_key(|b| (b.min, b.max));
        boxes.into_iter().map(|b| (b.min, b.max)).collect()
    }

    fn demo_three_tool_scene(voxels_per_block: u32) -> Scene {
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let mut scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
            ]),
            0,
        );
        scene.voxels_per_block = voxels_per_block;
        scene
    }

    fn demo_village_scene(voxels_per_block: u32) -> Scene {
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
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
            ],
        );
        scene.voxels_per_block = voxels_per_block;
        scene_with_top_level_selected(scene, 0)
    }

    /// The index query returns EXACTLY the leaves a full walk + AABB filter returns,
    /// across several query boxes and several scenes (incl. instanced/recursive
    /// `--demo-village`). This is the S3 spatial-index correctness proof.
    #[test]
    fn spatial_index_query_matches_full_walk() {
        use crate::spatial_index::VoxelAabb;
        let voxels_per_block = 16;
        let scenes = [
            ("single", Scene::from_geometry(
                GeometryParams { shape: ShapeKind::Sphere, size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block], size_measurements: None, voxels_per_block, wall_blocks: 1 },
                MaterialChoice::Stone,
            )),
            ("three-tool", demo_three_tool_scene(voxels_per_block)),
            ("village", demo_village_scene(voxels_per_block)),
        ];
        // A spread of query boxes: empty, tiny near origin, a slab, the whole scene,
        // and a far-away box that should match nothing.
        let queries = [
            VoxelAabb::new([0, 0, 0], [0, 0, 0]),
            VoxelAabb::new([-8, -8, -8], [8, 8, 8]),
            VoxelAabb::new([0, -200, -200], [64, 200, 200]),
            VoxelAabb::new([-5000, -5000, -5000], [5000, 5000, 5000]),
            VoxelAabb::new([100_000, 0, 0], [100_064, 64, 64]),
        ];
        for (label, scene) in &scenes {
            let index = scene.build_leaf_spatial_index(voxels_per_block);
            for query in &queries {
                let from_index: Vec<VoxelAabb> = index
                    .leaves_intersecting(query)
                    .into_iter()
                    .map(|entry| entry.world_aabb)
                    .collect();
                let from_walk = walk_leaf_aabbs_intersecting(scene, voxels_per_block, query);
                assert_eq!(
                    sorted_aabbs(from_index),
                    sorted_aabbs(from_walk),
                    "[{label}] index query {query:?} must match the full walk + AABB filter"
                );
            }
        }
    }

    /// The diff that drives invalidation: an edit's AABB is the union of the old and
    /// new boxes of whatever changed.
    #[test]
    fn edit_aabb_diff_covers_old_and_new() {
        let voxels_per_block = 16;
        let scene_a = demo_three_tool_scene(voxels_per_block);
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);

        // No change: empty edit AABB.
        let index_a2 = scene_a.build_leaf_spatial_index(voxels_per_block);
        let no_edit = index_a2.edit_aabb_since(&index_a).expect("same density");
        assert!(no_edit.is_empty(), "an identical scene dirties nothing");

        // Move the Box (node 1) from +8X to +40X: the edit AABB must span BOTH the
        // old (+8) and new (+40) boxes.
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = NodeTransform::from_blocks([40, 0, 0], voxels_per_block);
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        let moved = index_b.edit_aabb_since(&index_a).expect("same density");
        assert!(!moved.is_empty());
        // CORNER-ANCHORING: a 5-block box spans `[off·d, off·d + 5d)` = `[off·16,
        // off·16 + 80)`. Old at +8: [128, 208). New at +40: [640, 720). The union
        // must contain both.
        assert!(moved.min[0] <= 8 * 16, "edit AABB must cover the OLD location");
        assert!(moved.max[0] >= 40 * 16 + 80, "edit AABB must cover the NEW location");

        // Recolour the Sphere (node 0, same box): edit AABB is just that box.
        let mut scene_c = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_c.root_node_mut(0).content {
            *material = MaterialChoice::Wood;
        }
        let index_c = scene_c.build_leaf_spatial_index(voxels_per_block);
        let recoloured = index_c.edit_aabb_since(&index_a).expect("same density");
        assert!(!recoloured.is_empty(), "a same-box content change is still dirty");
        // CORNER-ANCHORING: Sphere at origin, 5 blocks → span [0, 5·16) = [0, 80).
        assert_eq!(recoloured, crate::spatial_index::VoxelAabb::new([0, 0, 0], [80, 80, 80]));
    }

    /// A density change can't be localised: the diff returns `None` (clear).
    #[test]
    fn edit_aabb_diff_density_change_is_none() {
        let scene = demo_three_tool_scene(16);
        let index_16 = scene.build_leaf_spatial_index(16);
        let index_8 = scene.build_leaf_spatial_index(8);
        assert_eq!(
            index_8.edit_aabb_since(&index_16),
            None,
            "a density change forces a wholesale clear"
        );
    }

    /// A region-spanning Part edit can't be localised: the diff returns `None`.
    #[test]
    fn edit_aabb_diff_part_edit_is_none() {
        let voxels_per_block = 16;
        // A scene with a Tool plus a debug-cloud Part.
        let mut tool = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [5, 5, 5], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        tool.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let part = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 1 }));
        let scene_a = scene_with_top_level_selected(Scene::from_nodes(vec![tool.clone(), part]), 0);
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);
        assert!(index_a.has_region_spanning_leaf);

        // Change the Part's seed (a region-spanning content change).
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).content = NodeContent::Part(Part::DebugClouds { seed: 2 });
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(
            index_b.edit_aabb_since(&index_a),
            None,
            "editing a region-spanning Part forces a wholesale clear"
        );
    }

    // ===== Issue #30: shape generation aligns to the global block lattice ========

    /// Resolve a single Box leaf of `size_blocks` at the origin and return its
    /// occupied voxels' **absolute** (producer-true, non-recentred) integer-index
    /// bounding box `(min_corner, max_corner_exclusive)` plus the occupied count. A
    /// Box fully fills its bounding box, so the count is `prod(size·d)` and the box
    /// is the exact placed extent — letting the lattice-alignment tests read where
    /// generation actually lands relative to block multiples (multiples of `d`).
    fn absolute_box_extent(
        size_blocks: [u32; 3],
        voxels_per_block: u32,
    ) -> ([i64; 3], [i64; 3], usize) {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [size_blocks[0] * voxels_per_block, size_blocks[1] * voxels_per_block, size_blocks[2] * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        // `resolve_region_via_chunks` keeps ABSOLUTE (non-recentred) positions, so
        // its voxels are in the producer-true frame the per-object grids (#29) read.
        let grid = scene.resolve_region_via_chunks(voxels_per_block, 0);
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in &grid.occupied {
            let world_position = voxel.world_position();
            for axis in 0..3 {
                // Voxel centres sit at `n + 0.5`; `floor` recovers the cell index.
                let index = world_position[axis].floor() as i64;
                min[axis] = min[axis].min(index);
                max[axis] = max[axis].max(index + 1); // half-open upper bound
            }
        }
        (min, max, grid.occupied.len())
    }

    /// Assert that a box of `size_blocks` resolved at `density` (at world offset 0) is
    /// CORNER-ANCHORED at the origin in the ABSOLUTE frame: it generates exactly
    /// `prod(size·d)` voxels occupying the span `[0, size·d)` per axis, with every
    /// voxel centre on a half-integer (the global voxel lattice).
    ///
    /// CHANGED (corner-anchoring): the producer corner-emits, so its world offset is
    /// the LOW CORNER. In the ABSOLUTE (non-recentred) frame a zero-offset box spans
    /// `[0, size·d)`, NOT the old centred `[−size·d/2, size·d/2)`. The recentre then
    /// symmetrises it for the render frame (see the recentred-frame tests).
    fn assert_box_corner_at_origin(size: [u32; 3], density: u32) {
        let (min, max, count) = absolute_box_extent(size, density);
        let expected_count = (size[0] * density) as usize
            * (size[1] * density) as usize
            * (size[2] * density) as usize;
        assert_eq!(
            count, expected_count,
            "a {size:?}-block box at density {density} must generate prod(size·d) voxels"
        );
        for (axis, &size_axis) in size.iter().enumerate() {
            let grid = (size_axis * density) as i64;
            assert_eq!(
                min[axis], 0,
                "axis {axis}: corner-anchored min is 0 (size {size:?} @ {density})"
            );
            assert_eq!(
                max[axis], grid,
                "axis {axis}: corner-anchored max is size·d (size {size:?} @ {density})"
            );
        }
    }

    /// A 1×1×1-block box (size 1, ODD) at density `d` and offset 0 generates exactly
    /// `d³` voxels CORNER-ANCHORED at the origin: the absolute span is `[0, d)` per
    /// axis. Across the representative density set — d=1 (→ 1 voxel), d=2, d=15
    /// (→ 15³ = 3375), d=16 (default → 4096), d=32.
    ///
    /// CHANGED (corner-anchoring): the absolute span is `[0, d)` (offset = low corner),
    /// not the old centred `[−d/2, d/2)`.
    #[test]
    fn one_block_box_corner_anchored_across_densities() {
        for density in [1u32, 2, 15, 16, 32] {
            assert_box_corner_at_origin([1, 1, 1], density);
        }
    }

    /// An odd-sized shape (5×5×2) is CORNER-ANCHORED at the origin across densities:
    /// it generates `(5d)×(5d)×(2d)` voxels spanning `[0, size·d)`.
    ///
    /// CHANGED (corner-anchoring): the absolute span is `[0, size·d)` (offset = low
    /// corner), at d ∈ {1, 15, 16}. ODD `size·d` (d=15) no longer straddles voxel
    /// cells — every centre is a half-integer.
    #[test]
    fn odd_size_shape_corner_anchored_at_origin() {
        for density in [1u32, 15, 16] {
            assert_box_corner_at_origin([5, 5, 2], density);
        }
    }

    /// An even-sized shape (2×4×6) corner-anchored at the origin spans `[0, size·d)`,
    /// at d ∈ {1, 15, 16}.
    #[test]
    fn even_size_shape_corner_anchored_at_origin() {
        for density in [1u32, 15, 16] {
            assert_box_corner_at_origin([2, 4, 6], density);
            // Corner-anchored: the absolute min corner is 0 (offset = low corner).
            let size = [2u32, 4, 6];
            let (min, _max, _count) = absolute_box_extent(size, density);
            for (axis, &min_axis) in min.iter().enumerate() {
                assert_eq!(
                    min_axis, 0,
                    "axis {axis} @ d{density}: a corner-anchored box starts at index 0"
                );
            }
        }
    }

    /// The bounding box of the OCCUPIED VOXEL CENTRES for a single `shape` of
    /// `size_blocks` placed at world offset `[0, 0, 0]`, resolved at `density` in the
    /// **recentred render frame** ([`resolve_region`] — the frame the camera, gizmo
    /// and renderer use, which centres the composite on the origin). Returns
    /// `(min_centre, max_centre)` per axis (centres sit at `n + 0.5`). A shape is
    /// centred on the origin iff `min_centre + max_centre == 0` per axis.
    ///
    /// We assert on voxel CENTRES, not corners. CORNER-ANCHORING: the producer
    /// corner-emits (`[0, grid)`) and the recentre `floor(grid/2)` lands the composite
    /// in the render frame. For an EVEN voxel span the centre bbox is exactly symmetric
    /// (`min + max == 0`); for an ODD span the floor-recentre leaves it off by exactly
    /// one voxel (`min + max == 1`), since an odd extent has no voxel-centred origin.
    fn occupied_voxel_centre_bbox(
        shape: ShapeKind,
        size_blocks: [u32; 3],
        density: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape,
                size_voxels: [size_blocks[0] * density, size_blocks[1] * density, size_blocks[2] * density],
                size_measurements: None,
                voxels_per_block: density,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        // The recentred frame (what the renderer/camera/gizmo see): the composite's
        // block AABB is centred on the origin.
        let region = scene.full_extent_blocks(density);
        let grid = scene.resolve_region(region, density, 0);
        assert!(!grid.occupied.is_empty(), "shape {shape:?} {size_blocks:?}@{density} resolved empty");
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for voxel in &grid.occupied {
            let world_position = voxel.world_position();
            for axis in 0..3 {
                min[axis] = min[axis].min(world_position[axis]);
                max[axis] = max[axis].max(world_position[axis]);
            }
        }
        (min, max)
    }

    /// PERMANENT GUARD: a shape placed at world offset `[0, 0, 0]` is centred (to
    /// sub-voxel precision) on the origin in the **recentred monolithic
    /// `resolve_region` frame** — its occupied-voxel-CENTRE bounding box is symmetric
    /// about 0 within half a voxel on every axis. CORNER-ANCHORING: the centre-sum is
    /// `grid mod 2` — exactly 0 for an EVEN voxel span, exactly 1 for an ODD one (an
    /// odd extent has no voxel-centred origin; the floor-recentre biases it one voxel).
    ///
    /// This is the MONOLITHIC resolve frame; it is BIT-IDENTICAL to the windowed-app
    /// render path for a near scene (the per-chunk store applies the same recentre).
    /// Covers a 5×5×5 sphere (odd) and a 5×1×5 box (odd-X/Z, 1-block-Y).
    #[test]
    fn shape_centered_within_half_voxel_in_resolve_region_frame() {
        let cases: [(ShapeKind, [u32; 3]); 2] =
            [(ShapeKind::Sphere, [5, 5, 5]), (ShapeKind::Box, [5, 1, 5])];
        for density in [1u32, 8, 16] {
            for (shape, size) in cases {
                let (min, max) = occupied_voxel_centre_bbox(shape, size, density);
                for axis in 0..3 {
                    let grid = size[axis] * density;
                    let expected = (grid % 2) as f32; // 0 even, 1 odd
                    let centre_sum = min[axis] + max[axis];
                    assert_eq!(
                        centre_sum, expected,
                        "{shape:?} {size:?}@d{density} axis {axis}: voxel-centre bbox \
                         [{}, {}] sum must be grid%2 = {expected} (corner-anchored recentre)",
                        min[axis], max[axis]
                    );
                }
            }
        }
    }

    /// HEADLINE WIN (corner-anchoring): an ODD extent at ODD DENSITY (d=1) lands on the
    /// voxel lattice — every centre is a HALF-INTEGER, sitting strictly INSIDE its
    /// voxel cell `[k, k+1)`. This is the exact case the old centred-emit got wrong:
    /// at odd grid the centred convention put centres on INTEGERS (`idx + 0.5 − grid/2`
    /// = whole numbers), straddling cell boundaries — visibly off the global voxel
    /// grid. Corner-emit (`idx + 0.5`) makes every centre a half-integer for ANY parity.
    ///
    /// A 3×1×3 box @ d=1, recentred (recentre = floor(grid/2) = 1 on X/Z): X/Z centres
    /// are `idx + 0.5 − 1` = {−0.5, 0.5, 1.5}; Y centre = 0.5. Nine voxels, every centre
    /// a half-integer.
    #[test]
    fn odd_extent_at_odd_density_lands_on_voxel_lattice() {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 3], 1, 1);
        let scene = scene_with_top_level_selected(
            Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            )]),
            0,
        );
        let density = 1u32;
        let region = scene.full_extent_blocks(density);
        let grid = scene.resolve_region(region, density, 0);

        assert_eq!(grid.occupied.len(), 9, "3×1×3 box @ d=1 is a full 9-cell prism (3·1·3)");

        // THE WIN: every voxel centre is a half-integer (frac == 0.5) — on the lattice,
        // inside a cell — NOT an integer straddling a boundary (the old odd-grid bug).
        for voxel in &grid.occupied {
            for (axis, pos) in voxel.world_position().into_iter().enumerate() {
                assert_eq!(
                    pos.fract().abs(), 0.5,
                    "axis {axis} centre {pos} must be a HALF-INTEGER (on the voxel lattice) at d=1 odd extent"
                );
            }
        }

        // The recovered cells (floor of the recentred centre) are the symmetric set
        // {−1, 0, 1} on X/Z and {0} on Y.
        use std::collections::BTreeSet;
        let cells: BTreeSet<[i64; 3]> = grid
            .occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    position[0].floor() as i64,
                    position[1].floor() as i64,
                    position[2].floor() as i64,
                ]
            })
            .collect();
        let mut expected: BTreeSet<[i64; 3]> = BTreeSet::new();
        for x in [-1i64, 0, 1] {
            for z in [-1i64, 0, 1] {
                expected.insert([x, 0, z]);
            }
        }
        assert_eq!(cells, expected, "odd 3×1×3 @ d=1 cells are {{−1,0,1}}²×{{0}}");
    }

    // ===== Issue #29 foundation: per-object block-aligned voxel AABB + pivot =====
    //
    // The grid rework (#29) positions each object's block lattice / floor / voxel
    // grid and the transform gizmo from the node's BLOCK-ALIGNED VOXEL AABB and its
    // pivot/origin, in the recentred frame, across densities. The renderers don't
    // exist yet, but the geometry SOURCE does — `build_leaf_spatial_index` (the
    // per-leaf world AABB) and `recentre_voxels_for_resolve` (the recentre). These
    // tests pin that source. The RENDERER-level grid/lattice/gizmo-follow tests
    // (drawing the actual lines and the gizmo) will be added with #29 sub-steps
    // S3/S5, parametrized over the SAME density set {1, 15, 16}, once those
    // renderers exist.

    /// The single leaf's block-aligned voxel AABB, as `build_leaf_spatial_index`
    /// records it (the #29 grids' geometry source).
    fn single_leaf_aabb(size_blocks: [u32; 3], offset_blocks: [i64; 3], density: u32) -> VoxelAabb {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
        let index = scene.build_leaf_spatial_index(density);
        assert_eq!(index.entries.len(), 1, "one Tool leaf → one index entry");
        index.entries[0].world_aabb
    }

    /// The `NodeTransform` block/voxel accessors round-trip (incl. negatives), the
    /// mating predicate distinguishes block-aligned from sub-block offsets, and a
    /// 0-density document cannot panic (density clamped to ≥1 like the resolve
    /// sites). ADR 0003 §3f(0).
    #[test]
    fn node_transform_accessors_round_trip_and_guard() {
        // Round-trip through canonical voxels, including negative components.
        let transform = NodeTransform::from_blocks([-3, 2, -7], 16);
        assert_eq!(transform.offset_voxels, [-48, 32, -112], "blocks·d = voxels");
        assert_eq!(transform.blocks(16), [-3, 2, -7], "blocks(d) inverts from_blocks");
        assert!(transform.block_aligned(16), "a whole-block offset is on the lattice");

        // A hand-set SUB-block offset is NOT block-aligned (the mating predicate).
        let sub_block = NodeTransform { offset_voxels: [1, 0, 0], ..Default::default() };
        assert!(
            !sub_block.block_aligned(16),
            "an offset of 1 voxel at d=16 is off the block lattice"
        );

        // A 0-density document must not panic: density is clamped to ≥1.
        let _ = NodeTransform::from_blocks([2, 0, 0], 0);
        let zero_density = NodeTransform { offset_voxels: [2, 0, 0], ..Default::default() };
        let _ = zero_density.blocks(0);
        let _ = zero_density.block_aligned(0);
    }

    /// A `B`-block extent → a `B·d`-voxel AABB CORNER-ANCHORED at the node's world
    /// offset, at each density. This is the geometry the per-object block lattice /
    /// floor / voxel grid (#29) will span.
    ///
    /// CHANGED (corner-anchoring): the AABB is the producer-true span
    /// `[off·d, off·d + size·d)` — the offset IS the low corner. For a whole-block
    /// offset the corner is a block multiple of `d` at ANY size parity (no more
    /// half-block straddle for odd sizes).
    #[test]
    fn node_block_aabb_scales_and_corner_anchors_across_densities() {
        let size = [5u32, 5, 2]; // a representative mixed (odd X/Y, even Z) extent
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let aabb = single_leaf_aabb(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                let grid = (size_axis * density) as i64;
                let off_voxels = offset[axis] * density as i64;
                // Scales with density: a B-block extent → B·d voxels.
                assert_eq!(
                    aabb.max[axis] - aabb.min[axis],
                    grid,
                    "axis {axis} @ d{density}: AABB extent must be size·d voxels"
                );
                // Corner-anchored: the offset is the LOW corner.
                assert_eq!(
                    aabb.min[axis], off_voxels,
                    "axis {axis} @ d{density}: AABB min corner is off·d (corner-anchored)"
                );
                assert_eq!(
                    aabb.max[axis], off_voxels + grid,
                    "axis {axis} @ d{density}: AABB max corner is off·d + size·d"
                );
                // A whole-block offset → block-aligned corner at ANY size parity.
                assert_eq!(
                    aabb.min[axis].rem_euclid(density as i64), 0,
                    "axis {axis} @ d{density}: a whole-block offset is block-aligned"
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its AABB by
    /// exactly `d` voxels per axis (the grids/gizmo follow it), and the AABB stays
    /// block-aligned, at each density. A node's placement here is a whole-block
    /// offset (`[i64; 3]` blocks), so sub-block translation is not representable —
    /// whole-block translation is the unit tested.
    #[test]
    fn node_aabb_follows_translation_at_each_density() {
        let size = [5u32, 5, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let before = single_leaf_aabb(size, base, density);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let after = single_leaf_aabb(size, shifted, density);
                for axis in 0..3 {
                    let expected = if axis == moved_axis { density as i64 } else { 0 };
                    assert_eq!(
                        after.min[axis] - before.min[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block on axis {moved_axis} must shift \
                         the AABB min by exactly d on that axis (0 elsewhere)"
                    );
                    assert_eq!(
                        after.max[axis] - before.max[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block must shift the AABB max by d"
                    );
                    // The corner's lattice RESIDUE is preserved by a whole-block move
                    // (a +d shift can't change `min mod d`). We no longer require it to
                    // be 0 — an odd extent is centred on the offset, half a block off
                    // the lattice (center-anchoring retirement) — only that the move
                    // doesn't perturb it.
                    assert_eq!(
                        after.min[axis].rem_euclid(density as i64),
                        before.min[axis].rem_euclid(density as i64),
                        "axis {axis} @ d{density}: a whole-block translate preserves the corner's lattice residue"
                    );
                }
            }
        }
    }

    /// The node pivot/origin the selection transform gizmo (#29) will track: the
    /// node's world origin = `offset_in_blocks·d − recentre`, in the recentred frame.
    /// Pinned across densities for two facets:
    ///
    /// 1. **Recentred-frame value.** For a SINGLE-node scene the recentre always
    ///    re-centres that one node, so its pivot in the recentred frame is the
    ///    node's own centre offset from the recentre — INVARIANT under translation
    ///    (translating the lone node drags the auto-recentre with it). We pin the
    ///    concrete value `offset·d − recentre` and assert it does NOT move when the
    ///    node is translated alone. (This is why #29 positions grids in the GLOBAL
    ///    lattice frame, not this auto-recentred composite — only a fixed frame
    ///    makes "the gizmo follows the object" observable.)
    /// 2. **Absolute-frame follow.** In the producer-true ABSOLUTE frame the node
    ///    origin is `offset_in_blocks·d`; this DOES follow a `+1 block` translate by
    ///    exactly `d` voxels per axis (the property the global-frame gizmo inherits).
    #[test]
    fn node_pivot_origin_tracks_offset_across_densities() {
        let size = [5u32, 5, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let recentre_of = |offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
                let mut node =
                    Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
                node.transform = NodeTransform::from_blocks(offset, density);
                let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
                scene.recentre_voxels_for_resolve(density).voxels()
            };
            // Pivot in the recentred frame = offset·d − recentre.
            let recentred_pivot = |offset: [i64; 3]| {
                let recentre = recentre_of(offset);
                [
                    offset[0] * density as i64 - recentre[0],
                    offset[1] * density as i64 - recentre[1],
                    offset[2] * density as i64 - recentre[2],
                ]
            };
            // Absolute-frame node origin = offset·d (no recentre).
            let absolute_origin =
                |offset: [i64; 3]| [offset[0] * density as i64, offset[1] * density as i64, offset[2] * density as i64];

            let base_recentred = recentred_pivot(base);
            let base_absolute = absolute_origin(base);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let moved_recentred = recentred_pivot(shifted);
                let moved_absolute = absolute_origin(shifted);
                for axis in 0..3 {
                    // (1) Single-node recentred pivot is invariant under self-translation.
                    assert_eq!(
                        moved_recentred[axis], base_recentred[axis],
                        "axis {axis} @ d{density}: a lone node's recentred pivot is invariant \
                         under self-translation (the auto-recentre follows it)"
                    );
                    // (2) Absolute origin follows +1 block by exactly d on that axis.
                    let expected = if axis == moved_axis { density as i64 } else { 0 };
                    assert_eq!(
                        moved_absolute[axis] - base_absolute[axis],
                        expected,
                        "axis {axis} @ d{density}: absolute node origin must follow a +1-block \
                         translate on axis {moved_axis} by exactly d voxels (0 elsewhere)"
                    );
                }
            }
        }
    }

    // ---- issue #29 (grid rework S3): per-object block lattice box (renderer-follow) ----

    /// Build a single-Box-node scene at `offset`, return its
    /// `node_block_lattice_box_recentred` for node 0 at `density`.
    fn single_node_lattice_box(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        density: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
        scene
            .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
            .expect("a sized Box node has a lattice box")
    }

    /// The per-object lattice box spans the node's enclosing-block AABB and SCALES
    /// with density: a `B`-block extent → a `B·d`-voxel box, at each density
    /// {1, 15, 16} (the explicit user ask).
    ///
    /// The producer-true corner geometry is asserted in
    /// `node_block_aabb_scales_and_centres_across_densities` — in the RECENTRED frame
    /// the box is shifted by the composite recentre, so the recentred corners need not
    /// be block multiples; the block-aligned STRUCTURE (extent = B·d, planes step d)
    /// is what survives the recentre, and that is what this asserts.
    #[test]
    fn lattice_box_spans_enclosing_blocks_and_scales_with_density() {
        let size = [5u32, 3, 2];
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let (min, max) = single_node_lattice_box(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                // Box extent = size · density voxels (B-block extent → B·d voxels).
                assert_eq!(
                    (max[axis] - min[axis]) as i64,
                    (size_axis * density) as i64,
                    "axis {axis} @ d{density}: lattice box extent must be size·d voxels"
                );
                // The extent is an exact multiple of a block, so the box encloses
                // exactly `size_axis` whole blocks along each axis.
                assert_eq!(
                    ((max[axis] - min[axis]) as i64).rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: box extent spans whole blocks"
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its lattice box
    /// by exactly `density` voxels per axis (the lattice follows the object), at each
    /// density {1, 15, 16}. Because the node offset is whole-block, a SUB-block
    /// (1-voxel) translate is NOT representable at the node level, so the
    /// "add/remove a whole block on a sub-block move" requirement cannot be
    /// constructed here; the whole-block follow IS the unit tested. (The
    /// expand-to-block that WOULD turn a sub-block shift into a whole-block box
    /// change is exercised directly on `block_boundaries`/`*_vertices_into` in the
    /// renderer tests.)
    #[test]
    fn lattice_box_follows_whole_block_translate_at_each_density() {
        let size = [5u32, 3, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            // A SECOND, LARGE anchor node (centred at the origin, ±100 blocks on
            // every axis) dominates the composite AABB on all axes, so the small
            // moving node never touches a composite corner and the recentre stays
            // FIXED. Observed in that fixed frame, moving the node by +1 block shifts
            // its box by exactly d — the "lattice follows the object in the global
            // lattice frame" property. (A lone node would drag its own recentre, so
            // the box would NOT appear to move — see `node_pivot_origin_*`.)
            let make_scene = |offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
                let mut moving = Node::new(
                    "Moving",
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                moving.transform = NodeTransform::from_blocks(offset, density);
                let anchor_shape = SdfShape::from_blocks(ShapeKind::Box, [200, 200, 200], 1, density);
                let mut anchor = Node::new(
                    "Anchor",
                    NodeContent::Tool { shape: anchor_shape, material: MaterialChoice::Stone },
                );
                // CORNER-ANCHORING: a leaf spans `[off, off+size)` blocks, so to make
                // the 200³ anchor BRACKET the small moving node on every axis (and so
                // dominate the composite AABB, fixing the recentre) it must be offset to
                // `[−100, 100)` blocks, not corner-anchored at the origin.
                anchor.transform = NodeTransform::from_blocks([-100, -100, -100], density);
                scene_with_top_level_selected(Scene::from_nodes(vec![moving, anchor]), 0)
            };
            let box_of = |offset: [i64; 3]| {
                make_scene(offset)
                    .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
                    .expect("moving node has a lattice box")
            };
            let before = box_of(base);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let after = box_of(shifted);
                for axis in 0..3 {
                    let expected = if axis == moved_axis { density as f32 } else { 0.0 };
                    assert_eq!(
                        after.0[axis] - before.0[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block on axis {moved_axis} must shift the \
                         lattice box min by exactly d (0 elsewhere)"
                    );
                    assert_eq!(
                        after.1[axis] - before.1[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block must shift the lattice box max by d"
                    );
                }
            }
        }
    }

    /// A size-less node (a Part with no intrinsic extent — `DebugClouds`) has NO
    /// lattice box: `node_block_lattice_box_recentred` returns `None` (nothing to
    /// draw), at each density.
    #[test]
    fn sizeless_node_has_no_lattice_box() {
        for density in [1u32, 15, 16] {
            let scene = Scene::single_node(Node::new(
                "Clouds",
                NodeContent::Part(Part::DebugClouds { seed: 0 }),
            ));
            assert_eq!(
                scene.node_block_lattice_box_recentred(&NodePath::root_index(0), density),
                None,
                "@ d{density}: a size-less node yields no lattice box"
            );
        }
    }

    // ---- issue #29 (grid rework S1): per-node grids, Points, masters ----

    /// A freshly-built node carries NO grids (issue #29: grids default OFF for new
    /// objects). `NodeGrids::default()` is all-false, and `Node::new` adopts it.
    #[test]
    fn new_node_has_all_grids_off() {
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        assert!(!node.grids.voxel_grid_on_faces);
        assert!(!node.grids.block_lattice);
        assert!(!node.grids.floor_grid);
        assert_eq!(node.grids, NodeGrids::default());
    }

    /// An empty `Scene::default()` has the issue-#29 grid-rework master defaults:
    /// ALL THREE masters ON (per-object flags stay OFF), and no Points yet.
    #[test]
    fn scene_default_master_grids() {
        let scene = Scene::default();
        assert!(scene.master_block_lattice, "block lattice master defaults ON");
        assert!(scene.master_voxel_grid, "voxel grid master defaults ON");
        assert!(scene.master_floor_grid, "floor grid master defaults ON");
        assert!(scene.points.is_empty(), "no Points until ensure_origin_point");
        assert_eq!(scene.active_point, None);
    }

    /// `ensure_origin_point` is idempotent and creates EXACTLY one Origin at index 0
    /// with the spec defaults (ground plane + axes on); a second call (or a scene
    /// that already has an Origin) does not duplicate it.
    #[test]
    fn ensure_origin_point_is_idempotent_and_creates_one_origin() {
        let mut scene = Scene::default();
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "exactly one Point after first call");
        let origin = &scene.points[0];
        assert!(origin.is_origin, "the synthesized Point is the Origin");
        assert_eq!(origin.name, "Origin");
        assert_eq!(origin.position_blocks, [0, 0, 0]);
        // Z-up: the ground plane is XY (`plane_xy`).
        assert!(origin.plane_xy, "ground plane (XY) on by default");
        assert!(origin.axis_x && origin.axis_y && origin.axis_z, "all axes on by default");
        assert!(!origin.plane_xz && !origin.plane_yz);
        assert!(!origin.hidden);

        // Idempotent: a second call does not add another Origin.
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "second call adds nothing");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// ADR 0003 Phase B: `ensure_node_ids` mints a unique non-zero id for every
    /// node — top-level, Group children, and definition nodes — and is idempotent.
    #[test]
    fn ensure_node_ids_mints_unique_stable_ids() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::Part(Part::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group("G", vec![clouds("B").into(), clouds("C").into()]),
        ]);
        scene.add_definition(DefId(1), "Def".to_string(), vec![clouds("D")]);

        scene.ensure_node_ids();

        // Collect every id (top-level + Group children + definition nodes). Every node
        // lives in the arena keyed by its id, so the arena keys ARE the full id set.
        let ids: Vec<NodeId> = scene.arena.keys().copied().collect();
        assert_eq!(ids.len(), 5, "A, G, B, C, D all visited");
        assert!(ids.iter().all(|&id| id != NodeId(0)), "no node keeps the 0 sentinel");
        let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "every minted id is unique");

        // Idempotent: a second pass mints nothing and changes no id.
        let before = scene.clone();
        scene.ensure_node_ids();
        assert_eq!(scene, before, "second call is a no-op");
    }

    /// A loaded scene that already carries an id keeps it, and the counter advances
    /// past it so a newly-minted node never collides.
    #[test]
    fn ensure_node_ids_preserves_existing_and_advances_counter() {
        // A loaded scene: the arena is keyed by id, so a node that already carries a
        // minted id (the "preset", id 5) lives under key NodeId(5), while a still-
        // unminted node sits under the NodeId(0) sentinel. `next_node_id` starts at 0,
        // as it would for a freshly-deserialized scene before normalization.
        let mut preset = Node::new("preset", NodeContent::Part(Part::DebugClouds { seed: 0 }));
        preset.id = NodeId(5);
        let mut fresh = Node::new("fresh", NodeContent::Part(Part::DebugClouds { seed: 0 }));
        fresh.id = NodeId(0);
        let mut scene = Scene::default();
        scene.arena.insert(NodeId(5), preset);
        scene.arena.insert(NodeId(0), fresh);
        scene.roots = vec![NodeId(5), NodeId(0)];

        scene.ensure_node_ids();

        // The preset id is preserved verbatim.
        assert!(scene.arena.contains_key(&NodeId(5)), "existing id preserved");
        assert_eq!(scene.arena[&NodeId(5)].name, "preset");
        // The unminted node was re-keyed out of the 0 sentinel into a fresh, distinct id.
        assert!(!scene.arena.contains_key(&NodeId(0)), "the 0 sentinel is gone");
        let fresh_id = scene
            .arena
            .iter()
            .find(|(_, node)| node.name == "fresh")
            .map(|(id, _)| *id)
            .expect("the fresh node still exists under a minted id");
        assert_ne!(fresh_id, NodeId(0), "fresh node minted");
        assert_ne!(fresh_id, NodeId(5), "fresh id does not collide with the existing one");
        assert!(scene.next_node_id > 5, "counter advanced past the loaded id");
        // Re-keying must repoint the SPINE, not just move the arena entry: the root slot
        // that referenced the sentinel now names the fresh id, so the node is still
        // reachable through `roots` (a stale NodeId(0) here would silently orphan it).
        assert_eq!(scene.roots[1], fresh_id, "the root spine slot was repointed off the sentinel");
        assert_eq!(
            scene.node_at_path(&NodePath::root_index(1)).map(|node| node.name.as_str()),
            Some("fresh"),
            "the re-keyed node still resolves through the spine, not orphaned",
        );
    }

    /// ADR 0003 Phase B2: `id_at_path` / `path_of` / `node_by_id` agree with the
    /// positional `node_at_path` for EVERY node in the tree (the ⇄ equivalence the
    /// later selection/command migration relies on).
    #[test]
    fn node_id_and_path_resolution_round_trip() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::Part(Part::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group(
                "G",
                vec![
                    clouds("B").into(),
                    NodeBuilder::group("H", vec![clouds("C").into()]),
                ],
            ),
            NodeBuilder::Leaf(clouds("D")),
        ]);
        scene.ensure_node_ids();

        // Every tree row resolves both ways, consistently.
        for (path, row_id, _depth) in scene.tree_rows() {
            let id = scene.id_at_path(&path).expect("path resolves to an id");
            assert_eq!(id, row_id, "the row's carried id matches id_at_path");
            assert_ne!(id, NodeId(0), "a minted node never has the 0 sentinel");
            assert_eq!(
                scene.path_of(id),
                Some(path.clone()),
                "path_of inverts id_at_path"
            );
            // node_by_id and node_at_path reach the SAME node.
            let by_id = scene.node_by_id(id).expect("id resolves to a node");
            let by_path = scene.node_at_path(&path).expect("path resolves to a node");
            assert_eq!(by_id.id, by_path.id);
            assert_eq!(by_id.name, by_path.name);
        }

        // Sentinel + unknown ids resolve to nothing.
        assert!(scene.node_by_id(NodeId(0)).is_none());
        assert!(scene.path_of(NodeId(0)).is_none());
        assert!(scene.node_by_id(NodeId(9_999)).is_none());
        assert!(scene.path_of(NodeId(9_999)).is_none());

        // Mutable lookup reaches the same node.
        let first_id = scene.id_at_path(&NodePath::root_index(0)).unwrap();
        scene.node_by_id_mut(first_id).unwrap().name = "renamed".to_string();
        assert_eq!(scene.node_at_path(&NodePath::root_index(0)).unwrap().name, "renamed");
    }

    /// An existing Origin (anywhere in the list) is NOT duplicated by
    /// `ensure_origin_point`; a scene that already carries one is left untouched.
    #[test]
    fn ensure_origin_point_does_not_duplicate_existing_origin() {
        let mut scene = Scene::default();
        // Seed a non-origin Point first, then an Origin at index 1.
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });
        scene.add_point(Point { name: "Origin".to_string(), is_origin: true, ..Point::default() });
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 2, "no Origin inserted when one exists");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// `add_point` gives a newly-added user Point the clean default (issue #29 fix):
    /// **all planes OFF** with **all three axes ON** — even if the caller passes a
    /// Point with planes enabled. Only the Origin (built by `ensure_origin_point`,
    /// not `add_point`) keeps the ground (XY, Z-up) plane on.
    #[test]
    fn add_point_defaults_planes_off_axes_on() {
        let mut scene = Scene::default();
        // Pass a Point with EVERY plane on; add_point must override them off.
        scene.add_point(Point {
            name: "User".to_string(),
            plane_xz: true,
            plane_xy: true,
            plane_yz: true,
            axis_x: false,
            axis_y: false,
            axis_z: false,
            ..Point::default()
        });
        let point = &scene.points[0];
        assert!(!point.plane_xz && !point.plane_xy && !point.plane_yz, "new point: all planes OFF");
        assert!(point.axis_x && point.axis_y && point.axis_z, "new point: all axes ON");

        // The Origin (via ensure_origin_point) still keeps the ground plane on
        // (Z-up: ground = XY = `plane_xy`).
        let mut origin_scene = Scene::default();
        origin_scene.ensure_origin_point();
        assert!(origin_scene.points[0].plane_xy, "Origin keeps the ground plane (XY)");
    }

    /// `remove_point` deletes a normal Point but NO-OPS on the Origin (undeletable),
    /// and `toggle_point_hidden` hides the Origin (hideable).
    #[test]
    fn remove_point_spares_origin_which_is_hideable() {
        let mut scene = Scene::default();
        scene.ensure_origin_point(); // Origin at index 0
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() }); // index 1

        // Removing the Origin is a no-op.
        scene.remove_point(0);
        assert_eq!(scene.points.len(), 2, "the Origin is undeletable");
        assert!(scene.points[0].is_origin);

        // Removing a normal Point works.
        scene.remove_point(1);
        assert_eq!(scene.points.len(), 1, "a normal Point is removable");
        assert!(scene.points[0].is_origin);

        // Out-of-range removal is a no-op (never panics).
        scene.remove_point(99);
        assert_eq!(scene.points.len(), 1);

        // The Origin is hideable: toggling its hidden flag works.
        assert!(!scene.points[0].hidden);
        scene.toggle_point_hidden(0);
        assert!(scene.points[0].hidden, "the Origin can be hidden");
        scene.toggle_point_hidden(0);
        assert!(!scene.points[0].hidden, "and un-hidden");
    }

    /// Serde round-trip: a Scene whose node carries non-default `NodeGrids` plus a
    /// custom Point round-trips through JSON byte-equal (structurally).
    #[test]
    fn scene_with_grids_and_points_round_trips() {
        let mut node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        node.grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let mut built = Scene::from_nodes(vec![node]);
        built.master_block_lattice = false;
        built.master_voxel_grid = true;
        built.master_floor_grid = true;
        built.active_point = Some(1);
        let mut scene = scene_with_top_level_selected(built, 0);
        scene.ensure_origin_point();
        // Push directly (not via `add_point`, which overrides plane/axis flags to the
        // new-point default) so the round-trip exercises non-default per-axis flags.
        scene.points.push(Point {
            name: "Corner".to_string(),
            position_blocks: [3, 4, 5],
            plane_xz: false,
            plane_xy: true,
            plane_yz: true,
            axis_x: true,
            axis_y: false,
            axis_z: true,
            hidden: true,
            ..Point::default()
        });

        let json = serde_json::to_string_pretty(&scene).expect("serialise");
        let restored: Scene = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(scene, restored, "scene with grids + points round-trips");
        assert!(restored.root_node(0).grids.voxel_grid_on_faces);
        assert!(restored.root_node(0).grids.floor_grid);
        assert!(!restored.master_block_lattice);
        assert!(restored.master_voxel_grid);
        assert_eq!(restored.points.len(), 2);
        assert_eq!(restored.points[1].position_blocks, [3, 4, 5]);
        // Per-axis flags survive the round-trip (issue #29 fix: split axes).
        assert!(restored.points[1].axis_x && !restored.points[1].axis_y && restored.points[1].axis_z);
    }

    /// Back-compat: an OLD serialized scene (no `grids`, no `points`, no masters)
    /// deserialises with the correct defaults — node grids all-off, all three
    /// masters at their struct default (ON, issue #29 grid-rework fix), empty points.
    #[test]
    fn old_scene_json_loads_with_grid_defaults() {
        // Build a one-Box scene, serialize it, then STRIP the optional fields that an
        // old document would not carry (the per-node `grids`, the scene-wide masters,
        // `points`, `active_point`). Deserializing the trimmed JSON must fill every
        // missing field with its struct default.
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let mut value = serde_json::to_value(&scene).expect("serialise");
        let object = value.as_object_mut().expect("scene serializes to an object");
        // Drop the optional/defaulted fields so the load path must synthesize them.
        object.remove("master_block_lattice");
        object.remove("master_voxel_grid");
        object.remove("master_floor_grid");
        object.remove("points");
        object.remove("active_point");
        // Strip every node's `grids` so the per-node default (#29 all-off) is exercised.
        if let Some(arena) = object.get_mut("arena").and_then(|a| a.as_object_mut()) {
            for stored in arena.values_mut() {
                if let Some(node_obj) = stored.as_object_mut() {
                    node_obj.remove("grids");
                }
            }
        }
        let old_json = serde_json::to_string(&value).expect("re-serialise trimmed doc");

        let scene: Scene = serde_json::from_str(&old_json).expect("old scene parses");
        assert_eq!(scene.roots.len(), 1);
        assert_eq!(scene.root_node(0).grids, NodeGrids::default(), "grids default off");
        assert!(scene.master_block_lattice, "lattice master default on");
        assert!(scene.master_voxel_grid && scene.master_floor_grid, "all masters default on");
        assert!(scene.points.is_empty(), "no points in the old document");
        assert_eq!(scene.active_point, None);
    }

    /// Issue #29 S2: the transform gizmo's pivot is the SELECTED node's block-AABB
    /// centre in the recentred render frame — `block_aabb_centre·d − recentre` —
    /// `None` when nothing is selected, across densities.
    #[test]
    fn active_gizmo_placement_follows_selected_node() {
        for vpb in [1u32, 15, 16] {
            // Bake each node's whole-block offset at the resolve density `vpb` so the
            // stored voxel offset divides back to the same block offset under this
            // resolution (the gizmo reads `offset_voxels / vpb` → blocks).
            let make_tool = |kind, size: [u32; 3], offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(kind, size, 1, vpb);
                let mut node = Node::new(
                    format!("{kind:?}"),
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                node.transform = NodeTransform::from_blocks(offset, vpb);
                node
            };
            // Three even-sized boxes; box B sits +8X, box C sits +6Z. CORNER-ANCHORING:
            // a 4-block box at offset `off` spans `[off, off+4]` blocks, centre `off+2`.
            let mut scene = Scene::from_nodes(vec![
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [8, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 6]),
            ]);
            scene.active = None;
            // ADR 0003 Phase B3: mint ids so selecting a node by id resolves.
            scene.ensure_node_ids();

            // Nothing selected → no gizmo.
            assert_eq!(
                scene.active_gizmo_placement(vpb),
                None,
                "no selection hides the gizmo (vpb={vpb})"
            );

            let recentre = scene.recentre_voxels_for_resolve(vpb).voxels();
            let density = vpb as i64;

            // Expected pivot for a 4-block box at block OFFSET `off`: its geometric
            // centre is `(off + 2)·d` voxels (corner-anchored), minus the recentre.
            let half_extent_voxels = 2 * density; // half of the 4-block extent
            let expected_pivot = |off_blocks: [i64; 3]| {
                [
                    (off_blocks[0] * density + half_extent_voxels - recentre[0]) as f32,
                    (off_blocks[1] * density + half_extent_voxels - recentre[1]) as f32,
                    (off_blocks[2] * density + half_extent_voxels - recentre[2]) as f32,
                ]
            };

            // Select each node in turn; the gizmo pivot tracks it.
            for (index, centre) in [([0, 0, 0]), ([8, 0, 0]), ([0, 0, 6])].into_iter().enumerate() {
                scene.active = scene.id_at_path(&NodePath::root_index(index));
                let (pivot, extent) =
                    scene.active_gizmo_placement(vpb).expect("selection shows the gizmo");
                assert_eq!(
                    pivot,
                    expected_pivot(centre),
                    "pivot == centre·d − recentre for node {index} (vpb={vpb})"
                );
                // Extent is the node's OWN 4-block AABB (not the whole region).
                assert_eq!(
                    extent,
                    [(4 * density) as f32; 3],
                    "gizmo sized from the node's own extent (vpb={vpb})"
                );
            }
        }
    }

    /// Issue #29 S2: a SINGLE selected node recentres onto the origin, so its gizmo
    /// pivot is exactly `[0, 0, 0]` (for an EVEN-sized node, whose block-AABB centre
    /// lands on an integer voxel). The gizmo only visibly moves with a multi-node
    /// selection. Guards against reading the pivot from absolute (un-recentred) space.
    #[test]
    fn single_even_selected_node_gizmo_sits_at_origin() {
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 2, 6], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            assert_eq!(
                pivot,
                [0.0, 0.0, 0.0],
                "the lone even-sized selected node recentres onto the origin (vpb={vpb})"
            );
        }
    }

    /// CHANGED (center-anchoring retirement): for an ODD-sized lone node the gizmo
    /// pivot now sits at WITHIN HALF A VOXEL of the origin for ALL densities —
    /// including the odd-size/odd-density case the old block-lattice shift got wrong
    /// (it left the pivot half a BLOCK off). The gizmo pivot and the composite
    /// recentre are now BOTH derived from the producer-true voxel frame, so a lone
    /// node's centre coincides with the recentre: pivot is exactly 0 for an even voxel
    /// span and ±0.5 voxel for an odd one (the truncation of a half-voxel centre).
    #[test]
    fn single_odd_selected_node_gizmo_is_at_most_half_voxel_off_origin() {
        // Sizes (3, 1, 5) are all odd. The lone node's pivot stays WITHIN half a voxel
        // of origin (NOT half a block, as the retired #30 shift produced) — exactly 0
        // when the voxel span size·d is even, ±0.5 voxel when odd.
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 5], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            for (axis, &component) in pivot.iter().enumerate() {
                assert!(
                    component.abs() <= 0.5,
                    "lone odd-sized node pivot within half a voxel of origin \
                     (axis {axis}, vpb={vpb}, got {component})"
                );
            }
            if vpb % 2 == 0 {
                assert_eq!(
                    pivot, [0.0, 0.0, 0.0],
                    "even density makes the lone-node recentre exact (vpb={vpb})"
                );
            }
        }
    }
