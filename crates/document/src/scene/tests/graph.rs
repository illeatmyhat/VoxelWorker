use super::*;
use voxel_core::core_geom::MaterialChoice;
use crate::debug_clouds::DebugCloudField;
use voxel_core::units::ExactRational;
use voxel_core::units::Measurement;
use voxel_core::voxel::ShapeKind;
use voxel_core::voxel::VoxelGrid;
use crate::voxel::GeometryParams;
use crate::voxel::SdfShape;
use crate::voxel::VoxelProducer;


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

    /// The same guarantee for a VoxelBody (the debug cloud field): a one-node VoxelBody
    /// scene matches `DebugCloudField::resolve` at the same dimensions. Step 2
    /// builds the VoxelBody node directly (the `debug_clouds` selector is gone).
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
            Scene::single_node(Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 })));
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
                NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }),
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
            // A VoxelBody-only cloud is corner-anchored at the explicit region (low corner 0,
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

            // A VoxelBody-only scene has no chunkable extent, so the monolithic path above
            // IS the resolve path (the chunk reassembly is for Tool-bearing scenes).
            assert!(
                !scene.has_chunkable_extent(vpb),
                "[{label}] a VoxelBody-only cloud has no chunkable extent"
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
        let cloud = Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 3 }));
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
            Node::new("child", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 })),
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

