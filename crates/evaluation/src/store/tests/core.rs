use super::*;


    /// (a) A cache HIT returns a grid identical to a fresh `Scene::resolve_chunk`.
    #[test]
    fn cache_hit_matches_fresh_resolve_chunk() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let mut cache = Store::new();
        let chunk_coord = [0, 0, 0];

        let fresh = scene.resolve_chunk(chunk_coord, 16, 0);

        // First call: a miss (resolves + stores).
        assert_eq!(cache.resident_chunk_count(), 0);
        let first = cache.chunk(chunk_coord, &scene, 16, 0).clone();
        assert_eq!(cache.resident_chunk_count(), 1);
        // Second call: a hit (no new resident chunk).
        let second = cache.chunk(chunk_coord, &scene, 16, 0).clone();
        assert_eq!(cache.resident_chunk_count(), 1, "a hit must not add a chunk");

        assert_eq!(first.dimensions, fresh.dimensions);
        assert_eq!(
            occupied_multiset(&first),
            occupied_multiset(&fresh),
            "a cached chunk must equal a fresh resolve_chunk"
        );
        assert_eq!(
            occupied_multiset(&second),
            occupied_multiset(&fresh),
            "a cache HIT must return the same grid as the miss"
        );
    }

    /// (b) The cache-assembled `resolve_region` output is IDENTICAL (occupied set +
    /// material_id, same recentre) to the monolithic `Scene::resolve_region` — for
    /// every required scene: all SDF shapes, demo-scene, demo-village.
    fn assert_cache_region_matches_monolithic(scene: &Scene, voxels_per_block: u32, label: &str) {
        let monolithic = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let mut cache = Store::new();
        let assembled = cache.resolve_region(scene, voxels_per_block, 0);

        assert_eq!(
            assembled.dimensions, monolithic.dimensions,
            "[{label}] cache-assembled dimensions must match monolithic"
        );
        assert_eq!(
            assembled.occupied_count(),
            monolithic.occupied_count(),
            "[{label}] cache-assembled occupied count must match monolithic"
        );
        assert_eq!(
            occupied_multiset(&assembled),
            occupied_multiset(&monolithic),
            "[{label}] cache-assembled occupied set (position + material) must be \
             BIT-IDENTICAL to monolithic resolve_region (same recentre)"
        );
    }

    #[test]
    fn cache_region_matches_monolithic_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_cache_region_matches_monolithic(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// FLAT / odd-sized shapes (e.g. a 5×1×5 cylinder — the app default) are the
    /// regression case for the S0 covering-range bug S2 fixed: the producer centres
    /// its grid on the origin, so a 1-block (odd) axis straddles two chunks, but the
    /// old block-AABB covering range (`floor(size/2)` per block) missed one of them
    /// and dropped half the voxels. This pins that the cache covers the
    /// producer-true voxel extent and reassembles bit-identically.
    #[test]
    fn cache_region_matches_monolithic_for_flat_and_odd_shapes() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = Scene::from_geometry(
                    GeometryParams {
                        shape: kind,
                        size_voxels: [size[0] * 16, size[1] * 16, size[2] * 16],
                        size_measurements: None,
                        voxels_per_block: 16,
                        wall_blocks: 1,
                    },
                    MaterialChoice::Stone,
                );
                assert_cache_region_matches_monolithic(&scene, 16, &format!("{kind:?} {size:?}"));
            }
        }
    }

    #[test]
    fn cache_region_matches_monolithic_for_demo_scene() {
        let voxels_per_block = 16;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-scene");
    }

    #[test]
    fn cache_region_matches_monolithic_for_demo_village() {
        let voxels_per_block = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
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
        assert_cache_region_matches_monolithic(&scene, voxels_per_block, "demo-village");
    }

    /// A density change clears + re-binds the cache (a chunk's voxel extent depends
    /// on density), and the re-resolve still matches the monolithic at the new
    /// density.
    #[test]
    fn density_change_rebinds_cache() {
        let scene = shape_scene(ShapeKind::Torus, 16);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, 16, 0);
        assert!(cache.resident_chunk_count() > 0);

        let scene_8 = shape_scene(ShapeKind::Torus, 8);
        let assembled_8 = cache.resolve_region(&scene_8, 8, 0);
        let monolithic_8 =
            scene_8.resolve_region(scene_8.full_extent_blocks(8), 8, 0);
        assert_eq!(
            occupied_multiset(&assembled_8),
            occupied_multiset(&monolithic_8),
            "after a density change the cache re-resolves correctly at the new density"
        );
    }

    /// `clear` empties the cache (the S3 invalidation seam).
    #[test]
    fn clear_empties_cache() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let mut cache = Store::new();
        let _ = cache.chunk([0, 0, 0], &scene, 16, 0);
        assert!(cache.resident_chunk_count() > 0);
        cache.clear();
        assert_eq!(cache.resident_chunk_count(), 0, "clear drops every chunk");
    }

    /// (c) A synthetic scene whose TOTAL voxel count exceeds the old 6M whole-region
    /// cap, but whose individual chunks are each small, resolves successfully under
    /// the new PER-CHUNK bound — proving total scene size is no longer capped at 6M.
    ///
    /// The scene is two small boxes pushed to opposite corners of a cube spaced 16
    /// blocks apart on EVERY axis. The composite AABB is a 17³-block cube → at
    /// density 16 that is `(17·16)³ ≈ 20M` whole-region voxels (well past the old 6M
    /// total cap), yet only the two corner chunks hold any voxels and each holds one
    /// tiny box — far under the per-chunk bound.
    ///
    /// (Spreading the boxes DIAGONALLY rather than in a long row keeps the same
    /// "total ≫ 6M, chunks tiny" coverage while the covering-chunk grid stays a small
    /// ~5³ cube — the row-of-64 form this replaced spanned ~500 chunks on one axis
    /// and dominated the lib-test wall-time. See issue #27 S3.)
    #[test]
    fn scene_exceeding_old_total_cap_resolves_under_per_chunk_bound() {
        let voxels_per_block = 16u32;
        // Two 1-block stone cubes at opposite corners of a 16-block cube, so the
        // composite spans a huge cubic extent while each chunk holds at most one box.
        let spacing_blocks = 16i64;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, voxels_per_block);
        let corner = |label: &str, offset: [i64; 3]| {
            let mut node = Node::new(
                label,
                NodeContent::Tool { shape: shape.clone(), material: MaterialChoice::Stone },
            );
            node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
            node
        };
        let scene = Scene::from_nodes(vec![
            corner("Box lo", [0, 0, 0]),
            corner("Box hi", [spacing_blocks, spacing_blocks, spacing_blocks]),
        ]);

        // The OLD whole-region cap would reject this: the composite AABB voxel count
        // is far beyond 6M.
        let region = scene.full_extent_blocks(voxels_per_block);
        let whole_region_voxels = region.size_blocks[0] as u64
            * region.size_blocks[1] as u64
            * region.size_blocks[2] as u64
            * (voxels_per_block as u64).pow(3);
        assert!(
            whole_region_voxels > voxel_core::voxel::MAX_GRID_VOXELS,
            "the synthetic scene's whole-region voxel count ({whole_region_voxels}) must \
             exceed the OLD 6M total cap to prove the point"
        );

        // Every individual chunk is small (one small box at most) — under the new
        // per-chunk bound, so the lazy per-chunk resolve succeeds.
        let mut cache = Store::new();
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("a placed scene has a covering chunk range");
        let mut total_resolved = 0usize;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = cache.chunk([chunk_x, chunk_y, chunk_z], &scene, voxels_per_block, 0);
                    assert!(
                        (chunk.occupied_count() as u64) <= voxel_core::voxel::MAX_CHUNK_VOXELS,
                        "every chunk must stay under the per-chunk bound"
                    );
                    total_resolved += chunk.occupied_count();
                }
            }
        }
        assert!(
            total_resolved > 0,
            "the lazy per-chunk resolve must produce voxels for a scene the old total \
             cap would have rejected outright"
        );
    }

    /// One whose SINGLE chunk exceeds the per-chunk bound is still rejected (the cap
    /// did not simply vanish — it moved to per-chunk granularity).
    #[test]
    fn single_chunk_exceeding_per_chunk_bound_is_rejected() {
        // The per-chunk bound is the chunk's voxel CAPACITY (one chunk's voxel
        // extent cubed). A density large enough that one chunk's capacity exceeds
        // the bound must be rejected by the guard helper.
        let chunk_capacity_at = |voxels_per_block: u32| -> u64 {
            let extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block) as u64;
            extent * extent * extent
        };
        // Density 16: chunk extent = 64 voxels → 64³ = 262_144 voxels/chunk, well
        // under the bound — NOT rejected.
        assert!(chunk_capacity_at(16) <= voxel_core::voxel::MAX_CHUNK_VOXELS);
        assert!(
            !voxel_core::voxel::chunk_extent_exceeds_bound(16),
            "a normal density-16 chunk is under the per-chunk bound"
        );

        // A density whose single chunk capacity exceeds the bound IS rejected.
        // chunk extent = CHUNK_BLOCKS × density; pick a density making 64³·k > bound.
        let huge_density = 64u32; // extent = 256 → 256³ = 16_777_216 voxels/chunk.
        assert!(
            chunk_capacity_at(huge_density) > voxel_core::voxel::MAX_CHUNK_VOXELS,
            "the chosen huge density must make one chunk exceed the per-chunk bound"
        );
        assert!(
            voxel_core::voxel::chunk_extent_exceeds_bound(huge_density),
            "a chunk whose voxel capacity exceeds the per-chunk bound must be rejected"
        );
    }

