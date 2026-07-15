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

    /// Canonicalise an occupied set into a sorted multiset of
    /// `(bit_exact_voxel_position, material_id)`, so two resolves compare equal
    /// regardless of voxel emission ORDER but **byte-for-byte** on each `f32`
    /// position. Keying on the raw `f32` bits (`to_bits`) — not a rounded integer —
    /// means this asserts the bytes the renderer/mesher/fog consume are IDENTICAL,
    /// the S2 bit-identical-output guarantee (not merely the same rounded voxel
    /// set). A sub-ULP shift in any position fails the comparison.
    fn occupied_multiset(grid: &VoxelGrid) -> std::collections::BTreeMap<([u32; 3], u16), usize> {
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

    // ===== Issue #27 S3: targeted edit-AABB invalidation ========================

    fn three_tool_scene(voxels_per_block: u32, box_offset_x: i64) -> Scene {
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

    /// The set of chunk coords currently resident in the cache (for assertions).
    fn resident_coords(cache: &Store) -> std::collections::BTreeSet<[i32; 3]> {
        cache.chunks.keys().map(|key| key.chunk_coord).collect()
    }

    /// After an edit at offset X, exactly the chunks intersecting the edit AABB are
    /// evicted; every other chunk stays resident; and a re-resolve after the
    /// targeted invalidation yields a grid IDENTICAL to a full fresh resolve.
    #[test]
    fn targeted_invalidation_evicts_only_intersecting_chunks() {
        let density = 16u32;
        // A scene spread far enough in X that the Box occupies chunks no other leaf
        // touches (so moving it is a clean, localised edit).
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);
        let all_resident = resident_coords(&cache);
        assert!(!all_resident.is_empty());

        // Move the Box from +40X to +80X. Compute the edit AABB via the spatial-index
        // diff, exactly as `main::rebuild_geometry` does.
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = document::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The chunks the edit AABB intersects — the EXPECTED evicted set (those that
        // were resident).
        let (min_chunk, max_chunk) = edit_aabb
            .covering_chunk_range(density)
            .expect("a non-empty edit AABB has a chunk range");
        let mut expected_evicted = std::collections::BTreeSet::new();
        for &coord in &all_resident {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                expected_evicted.insert(coord);
            }
        }
        assert!(!expected_evicted.is_empty(), "the move must dirty at least one resident chunk");

        cache.invalidate_aabb(&edit_aabb, density);
        let after = resident_coords(&cache);

        // Every expected-evicted chunk is gone; every other chunk is still resident.
        for coord in &expected_evicted {
            assert!(!after.contains(coord), "chunk {coord:?} intersecting the edit must be evicted");
        }
        for coord in &all_resident {
            if !expected_evicted.contains(coord) {
                assert!(after.contains(coord), "chunk {coord:?} outside the edit must stay resident");
            }
        }

        // A re-resolve after targeted invalidation == a full fresh resolve of B.
        let reresolved = cache.resolve_region(&scene_b, density, 0);
        let mut fresh_cache = Store::new();
        let fresh = fresh_cache.resolve_region(&scene_b, density, 0);
        assert_eq!(
            occupied_multiset(&reresolved),
            occupied_multiset(&fresh),
            "re-resolve after targeted invalidation must equal a full fresh resolve"
        );
    }

    /// Moving a node from A to B invalidates chunks around BOTH A and B (the diff
    /// unions the old and new boxes).
    #[test]
    fn move_invalidates_chunks_around_both_endpoints() {
        let density = 16u32;
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);

        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = document::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The chunk owning the OLD Box centre (40·16 = 640 voxels) and the chunk
        // owning the NEW centre (80·16 = 1280 voxels) must BOTH be in the edit range.
        let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * density) as i32;
        let old_chunk_x = (640i32).div_euclid(chunk_extent);
        let new_chunk_x = (1280i32).div_euclid(chunk_extent);
        let (min_chunk, max_chunk) = edit_aabb.covering_chunk_range(density).unwrap();
        assert!(min_chunk[0] <= old_chunk_x && old_chunk_x <= max_chunk[0], "edit range must cover OLD chunk");
        assert!(min_chunk[0] <= new_chunk_x && new_chunk_x <= max_chunk[0], "edit range must cover NEW chunk");
    }

    /// An empty edit AABB (nothing changed) evicts nothing.
    #[test]
    fn empty_edit_aabb_evicts_nothing() {
        let density = 16u32;
        let scene = three_tool_scene(density, 8);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, density, 0);
        let before = resident_coords(&cache);
        let empty = voxel_core::spatial_index::VoxelAabb::new([0, 0, 0], [0, 0, 0]);
        let evicted = cache.invalidate_aabb(&empty, density);
        assert!(evicted.is_empty(), "an empty edit AABB reports an empty evicted set");
        assert_eq!(resident_coords(&cache), before, "an empty edit AABB evicts nothing");
    }

    /// **S6c-2a: the evicted-set return.** `invalidate_aabb` returns exactly the
    /// coords spanned by the edit AABB's `covering_chunk_range` that were resident —
    /// the same set the cache actually drops — so the GPU cache can evict in lockstep.
    #[test]
    fn invalidate_aabb_returns_exactly_the_evicted_coords() {
        let density = 16u32;
        let scene_a = three_tool_scene(density, 40);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene_a, density, 0);
        let all_resident = resident_coords(&cache);

        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = document::scene::NodeTransform::from_blocks([80, 0, 0], density);
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b.edit_aabb_since(&index_a).expect("same density");

        // The expected evicted set: resident coords inside the edit's chunk range.
        let (min_chunk, max_chunk) = edit_aabb
            .covering_chunk_range(density)
            .expect("a non-empty edit AABB has a chunk range");
        let mut expected: std::collections::BTreeSet<[i32; 3]> = std::collections::BTreeSet::new();
        for &coord in &all_resident {
            let inside = (0..3).all(|axis| coord[axis] >= min_chunk[axis] && coord[axis] <= max_chunk[axis]);
            if inside {
                expected.insert(coord);
            }
        }
        assert!(!expected.is_empty(), "the move must dirty at least one resident chunk");

        let returned: std::collections::BTreeSet<[i32; 3]> =
            cache.invalidate_aabb(&edit_aabb, density).into_iter().collect();
        assert_eq!(
            returned, expected,
            "the returned evicted set must equal exactly the resident coords inside \
             the edit AABB's covering_chunk_range"
        );
        // And the returned set is exactly what was dropped.
        let after = resident_coords(&cache);
        for coord in &returned {
            assert!(!after.contains(coord), "a returned coord must no longer be resident");
        }
        assert_eq!(
            after.len() + returned.len(),
            all_resident.len(),
            "evicted + remaining must partition the originally-resident set"
        );
    }

    /// A density mismatch (the belt-and-braces clear path) reports EVERY resident
    /// coord as evicted.
    #[test]
    fn invalidate_aabb_density_mismatch_reports_all_resident_evicted() {
        let scene = three_tool_scene(16, 8);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, 16, 0);
        let before = resident_coords(&cache);
        assert!(!before.is_empty());

        // Invalidate at a DIFFERENT density than the cache is bound to → clear path.
        let aabb = voxel_core::spatial_index::VoxelAabb::new([0, 0, 0], [16, 16, 16]);
        let returned: std::collections::BTreeSet<[i32; 3]> =
            cache.invalidate_aabb(&aabb, 8).into_iter().collect();
        assert_eq!(returned, before, "a density mismatch evicts (and reports) every resident coord");
        assert_eq!(cache.resident_chunk_count(), 0, "the cache is cleared");
    }

    // ===== Issue #20 S6c step 4: per-chunk render accessor ========================

    /// (S6c-2a parity) The union of `resident_render_chunks` (occupied cells +
    /// material_id, in each chunk's rebased frame) equals `resolve_region`'s
    /// assembled grid BYTE-FOR-BYTE, AND each returned coord is the absolute chunk
    /// coord that owns its grid's voxels, AND the coord set equals the scene's
    /// `covering_chunk_range`.
    fn assert_render_chunks_match_resolve_region(scene: &Scene, voxels_per_block: u32, label: &str) {
        // The truth: the assembled monolithic grid the renderer consumes today.
        let mut region_cache = Store::new();
        let assembled = region_cache.resolve_region(scene, voxels_per_block, 0);

        let mut render_cache = Store::new();
        let chunks = render_cache.resident_render_chunks(scene, voxels_per_block, 0);

        // Parity: the union of the per-chunk grids' occupied sets (already rebased,
        // same frame as the assembled grid) is bit-identical to the assembled grid.
        let mut union: std::collections::BTreeMap<([u32; 3], u16), usize> =
            std::collections::BTreeMap::new();
        for (_coord, grid) in &chunks {
            for (key, count) in occupied_multiset(grid) {
                *union.entry(key).or_insert(0) += count;
            }
        }
        assert_eq!(
            union,
            occupied_multiset(&assembled),
            "[{label}] union of resident_render_chunks must be BIT-IDENTICAL to \
             resolve_region's assembled grid (same rebased frame)"
        );

        // Coord set equals the scene's covering_chunk_range.
        let returned_coords: std::collections::BTreeSet<[i32; 3]> =
            chunks.iter().map(|(coord, _)| *coord).collect();
        let mut expected_coords: std::collections::BTreeSet<[i32; 3]> =
            std::collections::BTreeSet::new();
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(voxels_per_block) {
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        expected_coords.insert([chunk_x, chunk_y, chunk_z]);
                    }
                }
            }
        }
        assert_eq!(
            returned_coords, expected_coords,
            "[{label}] returned coord set must equal the scene's covering_chunk_range"
        );

        // Coord correctness: each returned coord is the absolute chunk coord that
        // owns its grid's voxels. The accessor binds to the recentre, so a chunk
        // coord `c` owns rebased voxels in `[c·E - recentre, (c+1)·E - recentre)`.
        let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block) as i64;
        let recentre = scene.recentre_voxels_for_resolve(voxels_per_block).voxels();
        for (coord, grid) in &chunks {
            for voxel in &grid.occupied {
                let position = voxel.world_position();
                for axis in 0..3 {
                    // Rebased absolute voxel index = floor(position) + recentre.
                    let absolute = position[axis].floor() as i64 + recentre[axis];
                    let owner = absolute.div_euclid(chunk_extent) as i32;
                    assert_eq!(
                        owner, coord[axis],
                        "[{label}] voxel at {:?} (axis {axis}) must be owned by chunk \
                         coord {coord:?}, not {owner}",
                        position
                    );
                }
            }
        }
    }

    #[test]
    fn render_chunks_match_resolve_region_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_render_chunks_match_resolve_region(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn render_chunks_match_resolve_region_for_demo_scene() {
        let scene = three_tool_scene(16, 8);
        assert_render_chunks_match_resolve_region(&scene, 16, "demo-scene");
    }

    #[test]
    fn render_chunks_match_resolve_region_for_demo_village() {
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
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
        assert_render_chunks_match_resolve_region(&scene, vpb, "demo-village");
    }

    /// A Part-only scene (no composite extent) yields an empty render-chunk set,
    /// matching `resolve_region`'s empty grid.
    #[test]
    fn render_chunks_empty_for_part_only_scene() {
        let scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(document::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache = Store::new();
        let chunks = cache.resident_render_chunks(&scene, 16, 0);
        assert!(chunks.is_empty(), "a Part-only scene has no covering chunks");
    }

    /// **ADR 0002 S4b — origin-rebased rendering, far-offset precision.** A box
    /// placed a HUGE distance from the origin must resolve to a grid whose voxel
    /// positions are **byte-identical** to the SAME box at the origin — because the
    /// render frame is rebased to the floating origin (= the composite recentre) in
    /// i64 BEFORE the f32 downcast, so the absolute distance never reaches the f32
    /// data.
    ///
    /// The offset is **1_000_000 blocks** = 16_000_000 voxels at density 16, PAST the
    /// f32 exact-integer ceiling (2²⁴ ≈ 16.7M). Under the OLD recentre-AFTER-f32-add
    /// path the absolute position `local + 1.6e7` lost the voxel-centre `.5` on EVERY
    /// voxel (the S1 far-lands jitter — verified at ~13% of the 3D viewport in the
    /// headless render). This test is the durable CPU regression guard that the
    /// rebased path keeps far == near to the LAST BIT (replacing S1's degraded
    /// far-offset behaviour). The bit-exact key (`f32::to_bits`) fails on any sub-ULP
    /// shift, so it would catch a regression that a rounded-voxel comparison misses.
    #[test]
    fn far_offset_resolves_byte_identical_to_near_after_rebase() {
        let vpb = 16u32;
        let box_scene = |offset_x: i64| -> Scene {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, vpb);
            let mut node = Node::new(
                "box",
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform = document::scene::NodeTransform::from_blocks([offset_x, 0, 0], vpb);
            Scene::single_node(node)
        };

        let mut near_cache = Store::new();
        let near = near_cache.resolve_region(&box_scene(0), vpb, 0);
        // 1_000_000 blocks → 16M voxels, past the f32 exact-integer ceiling.
        let mut far_cache = Store::new();
        let far = far_cache.resolve_region(&box_scene(1_000_000), vpb, 0);

        assert_eq!(near.occupied_count(), far.occupied_count(), "same shape");
        assert!(near.occupied_count() > 0, "the box must resolve to voxels");
        // Every voxel-centre `.5` fraction must survive the rebase (would be lost to
        // f32 rounding at 1.6e7 under the old subtract-AFTER-f32 path).
        for voxel in &far.occupied {
            let position = voxel.world_position();
            for axis in 0..3 {
                let frac = position[axis].fract().abs();
                assert!(
                    (frac - 0.5).abs() < 1e-4,
                    "far voxel centre lost its .5 fraction (f32 jitter): {:?}",
                    position
                );
            }
        }
        assert_eq!(
            occupied_multiset(&far),
            occupied_multiset(&near),
            "the far box must resolve BYTE-IDENTICAL to the near box — the rebase \
             subtracts the floating origin in i64 before the f32 downcast, so the \
             absolute distance never degrades the rendered f32 positions (S4b)"
        );
    }

    /// A Part-only scene (no intrinsic-size leaf) resolves to an empty recentred
    /// grid through the cache, exactly as monolithic `resolve_region` does.
    #[test]
    fn part_only_scene_resolves_empty_through_cache() {
        let scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(document::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache = Store::new();
        let assembled = cache.resolve_region(&scene, 16, 0);
        // A Part-only scene has no composite AABB → resolve_region returns a
        // zero-sized empty grid; the cache path matches.
        let monolithic = scene.resolve_region(RegionBlocks::new([0, 0, 0]), 16, 0);
        assert_eq!(assembled.occupied_count(), monolithic.occupied_count());
        assert_eq!(assembled.occupied_count(), 0);
    }

    // ===== Issue #20 S6d: region-scoped consumers =================================

    /// The whole-grid diameter readout for a scene's full active region — the
    /// reference value the region-scoped variants must reproduce.
    fn whole_grid_widest_run(scene: &Scene, vpb: u32, band: (u32, u32)) -> u32 {
        let region = scene.full_extent_blocks(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        grid.widest_run_in_band(band.0, band.1)
    }

    /// The cache's region-scoped `widest_run_in_band` returns the SAME value as the
    /// whole-grid `VoxelGrid::widest_run_in_band` for every required scene, across
    /// several layer bands.
    fn assert_region_widest_run_matches_whole_grid(scene: &Scene, vpb: u32, label: &str) {
        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z dimension (index 2).
        let grid_z = dims[2];
        // A spread of bands: the whole stack, the bottom layer, the top layer, the
        // exact mid-Z layer (the old slice), a thin interior band, and an
        // out-of-range band (above the grid → empty).
        let mid = grid_z.saturating_sub(1) / 2;
        let bands = [
            (0, grid_z.saturating_sub(1)),
            (0, 0),
            (grid_z.saturating_sub(1), grid_z.saturating_sub(1)),
            (mid, mid),
            (mid, (mid + 2).min(grid_z.saturating_sub(1))),
            (grid_z + 10, grid_z + 20),
        ];
        for band in bands {
            let expected = whole_grid_widest_run(scene, vpb, band);
            let mut cache = Store::new();
            let actual = cache.widest_run_in_band(scene, vpb, 0, band.0, band.1);
            assert_eq!(
                actual, expected,
                "[{label}] region widest_run_in_band band {band:?} must equal whole-grid"
            );
        }
    }

    /// **Far-offset diameter (issue #20 Step 2).** Two 3-block boxes 20,000 blocks
    /// apart on X: the composite is centred ~10,000 blocks out, so each box sits
    /// ~160,000 voxels from the recentred origin — far beyond any object the camera
    /// frames, while keeping the whole-grid reference cheap. The live diameter readout
    /// now routes through
    /// the region-scoped `widest_run_in_band`; it must report the box's TRUE width (a
    /// 48-voxel face row), confirming the rewired readout is correct far from the
    /// origin. It also equals the whole-grid value (the parity reference) — the two
    /// stay in lockstep until the region grid exceeds ~2^24 voxels on an axis, beyond
    /// which f32 collapses both identically (see the export test's NOTE).
    #[test]
    fn region_widest_run_correct_at_far_offset() {
        let vpb = 16u32;
        let make_box = |offset: [i64; 3]| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb);
            let mut node = Node::new("box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        // 20,000-block separation → composite centred ~10,000 blocks out → each box
        // ~160,000 voxels from the origin (far beyond any normal scene), while the
        // whole-grid reference (an O(grid_x)-per-row bitset) stays cheap to assemble.
        let scene = Scene::from_nodes(vec![make_box([0, 0, 0]), make_box([20_000, 0, 0])]);

        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z stack (both boxes at z=0).
        let band = (0, dims[2].saturating_sub(1));
        let true_box_width = 3 * vpb; // each box spans a full 48-voxel face row.

        let mut cache = Store::new();
        let region = cache.widest_run_in_band(&scene, vpb, 0, band.0, band.1);
        assert_eq!(
            region, true_box_width,
            "region widest_run must report the box's true 48-voxel width at far offset"
        );
        // And it equals the whole-grid reference at this (still f32-safe) far offset.
        assert_eq!(
            region,
            whole_grid_widest_run(&scene, vpb, band),
            "region widest_run must equal whole-grid at far offset"
        );
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_region_widest_run_matches_whole_grid(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_demo_scene() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        assert_region_widest_run_matches_whole_grid(&scene, vpb, "demo-scene");
    }

    #[test]
    fn region_widest_run_matches_whole_grid_for_demo_village() {
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
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
        assert_region_widest_run_matches_whole_grid(&scene, vpb, "demo-village");
    }

    /// **The cross-seam stitching case (the one that catches a stitching bug).** A
    /// long thin horizontal bar that deliberately spans MANY chunks on X: a box of
    /// 20 blocks × density 16 = 320 voxels wide, while a chunk is `CHUNK_BLOCKS=4 ×
    /// 16 = 64` voxels wide — so the bar crosses ~5 chunk seams. The widest run in a
    /// band through the bar must be the FULL bar width (one contiguous run), not the
    /// per-chunk fragment width. A naive per-chunk-max-then-combine implementation
    /// would report ~64 (one chunk's worth); the correct stitched answer equals the
    /// whole-grid run. We assert both: region == whole-grid AND the run is wider than
    /// a single chunk's voxel extent (proving the seam was actually crossed).
    #[test]
    fn region_widest_run_stitches_runs_across_chunk_seams() {
        let vpb = 16u32;
        let bar_blocks_x = 20u32; // 20 × 16 = 320 voxels wide.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [bar_blocks_x, 1, 1], 1, vpb);
        let scene = Scene::from_nodes(vec![Node::new(
            "bar",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        )]);

        let dims = scene.placed_region_dimensions(vpb);
        // Z-up: layers are Z-slices, so the band spans the Z dimension (index 2).
        let band = (0, dims[2].saturating_sub(1));

        let expected = whole_grid_widest_run(&scene, vpb, band);
        let mut cache = Store::new();
        let actual = cache.widest_run_in_band(&scene, vpb, 0, band.0, band.1);

        let chunk_extent_voxels = voxel_core::core_geom::CHUNK_BLOCKS * vpb; // 64
        assert!(
            expected > chunk_extent_voxels,
            "the bar's widest run ({expected}) must exceed one chunk's voxel extent \
             ({chunk_extent_voxels}) so the run genuinely crosses chunk seams"
        );
        assert_eq!(
            actual, expected,
            "region widest_run must stitch the run across chunk seams to equal the \
             whole-grid full-bar width"
        );
        // And the bar is the full grid width (a solid 320-voxel box row).
        assert_eq!(actual, dims[0], "the bar fills the whole X extent");
    }

    /// Single-voxel and empty bands: a 1×1×1 box (one voxel) reports a widest run of
    /// 1 in its band and 0 outside it; the region variant matches the whole grid.
    #[test]
    fn region_widest_run_single_voxel_and_empty_band() {
        let vpb = 16u32;
        // A 1-block box at density 16 is a 16³ solid; pick density 1 for a true
        // single voxel so the run is exactly 1.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 1);
        let _ = vpb;
        let scene = Scene::from_nodes(vec![Node::new(
            "dot",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        )]);
        let dims = scene.placed_region_dimensions(1);
        assert_eq!(dims, [1, 1, 1], "a 1×1×1@1 box is a single voxel");

        // In-band: widest run 1.
        let expected_in = whole_grid_widest_run(&scene, 1, (0, 0));
        let mut cache = Store::new();
        let actual_in = cache.widest_run_in_band(&scene, 1, 0, 0, 0);
        assert_eq!(expected_in, 1);
        assert_eq!(actual_in, expected_in);

        // Out-of-range band: empty → 0.
        let expected_out = whole_grid_widest_run(&scene, 1, (5, 9));
        let mut cache2 = Store::new();
        let actual_out = cache2.widest_run_in_band(&scene, 1, 0, 5, 9);
        assert_eq!(expected_out, 0);
        assert_eq!(actual_out, expected_out);

        // A wholly empty scene (Part-only, no occupied voxels): region run is 0.
        let empty_scene = Scene::single_node(Node::new(
            "Clouds",
            NodeContent::Part(document::scene::Part::DebugClouds { seed: 0 }),
        ));
        let mut cache3 = Store::new();
        assert_eq!(cache3.widest_run_in_band(&empty_scene, 16, 0, 0, 100), 0);
    }

    // ===== Issue #20 S6c-2c: incremental dirty-chunk rebuild ======================

    /// A per-chunk GPU instance cache, MODELLED on CPU as `coord → that chunk's
    /// occupied multiset` (the multiset is the byte-identical proxy for the GPU
    /// buffer's contents — `renderer::instances_for_chunk` builds one VoxelInstance
    /// per occupied voxel, so two chunks with equal occupied multisets produce
    /// byte-identical instance buffers). This lets the incremental-rebuild decision
    /// logic ([`incremental_rebuild_plan`], the EXACT function the GPU path
    /// uses) be exercised without a wgpu device, while still proving the post-edit
    /// cache CONTENTS match a full rebuild.
    type RenderCache = std::collections::BTreeMap<[i32; 3], ChunkMultiset>;
    type ChunkMultiset = std::collections::BTreeMap<([u32; 3], u16), usize>;

    /// Build the render cache a WHOLESALE rebuild produces for `scene`: every
    /// covering chunk's grid as a multiset (skipping zero-voxel chunks, exactly as
    /// `renderer::rebuild_chunk` drops them — no buffer is allocated for an empty
    /// chunk).
    fn full_render_cache(scene: &Scene, density: u32) -> RenderCache {
        let mut cache = Store::new();
        let chunks = cache.resident_render_chunks(scene, density, 0);
        chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `render_cache` IN PLACE,
    /// driving the GPU-cache decisions through [`incremental_rebuild_plan`]
    /// — the same plan `VoxelRenderer::incremental_rebuild_from_chunks` applies.
    /// Returns the number of chunks rebuilt (the observability count). The resolve
    /// cache (`resolve_cache`) carries state across edits exactly as the live app's
    /// does, so a HIT chunk is reused verbatim.
    fn apply_incremental_edit(
        render_cache: &mut RenderCache,
        resolve_cache: &mut Store,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> usize {
        // 1. Edit AABB → evicted (dirty) coords, exactly as main::rebuild_geometry.
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("same-density localisable edit");
        let evicted = resolve_cache.invalidate_aabb(&edit_aabb, density);

        // A recentre shift rebases EVERY chunk's contents, so the incremental path is
        // invalid — main::rebuild_geometry falls back to a full rebuild. Model that.
        let recentre_changed = scene_a.recentre_voxels_for_resolve(density)
            != scene_b.recentre_voxels_for_resolve(density);

        // 2. Freshly-resolved covering chunks for scene B (resolves the dirty/new
        //    chunks, reuses HITs).
        let render_chunks = resolve_cache.resident_render_chunks(scene_b, density, 0);

        if recentre_changed {
            // Full rebuild: clear + restore every non-empty covering chunk.
            render_cache.clear();
            for (coord, grid) in &render_chunks {
                if !grid.occupied.is_empty() {
                    render_cache.insert(*coord, occupied_multiset(grid));
                }
            }
            return render_chunks.len();
        }

        let resident: Vec<[i32; 3]> = render_cache.keys().copied().collect();
        // Only NON-EMPTY covering chunks deserve a buffer (matching the renderer).
        let occupied_covering: Vec<[i32; 3]> = render_chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();

        // 3. The plan — the SAME pure function the renderer drives the GPU from.
        let plan = incremental_rebuild_plan(&resident, &evicted, &occupied_covering);

        // 4. Rebuild only the planned coords (dirty ∪ new); evict the vacated ones.
        let rebuild_set: std::collections::BTreeSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        for (coord, grid) in &render_chunks {
            if rebuild_set.contains(coord) {
                render_cache.insert(*coord, occupied_multiset(grid));
            }
        }
        for coord in &plan.evict {
            render_cache.remove(coord);
        }
        plan.rebuild.len()
    }

    /// A tool node at the given offset, for building edit scenes.
    fn tool_node(kind: ShapeKind, size: [u32; 3], offset: [i64; 3], material: MaterialChoice) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, 16);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, 16);
        node
    }

    /// **The key S6c-2c correctness test.** For a sequence of localised edits, the
    /// render cache built INCREMENTALLY (rebuild only dirty/new chunks, evict
    /// vacated) is IDENTICAL — coord set AND every chunk's instance multiset — to a
    /// full wholesale rebuild of the post-edit scene. Proves no stale chunk survives
    /// and no fresh chunk is missed. Also asserts the dirty-chunk count is STRICTLY
    /// LESS than the total resident count for a localised edit (so it is genuinely
    /// incremental, not a disguised full rebuild).
    #[test]
    fn incremental_rebuild_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // The base scene: three tools spread far apart in X so each occupies chunks
        // the others don't touch (clean localised edits). Start the render + resolve
        // caches as a wholesale build of scene A.
        // Two STATIC anchor nodes at the X extremes pin the composite extent (hence
        // the recentre / floating origin) so the interior edits below keep it FIXED —
        // that is the regime where the incremental dirty-only path is valid (a
        // recentre shift rebases every chunk and forces a full rebuild instead; see
        // `apply_incremental_edit`). The interior "subject" box sits between them.
        let anchor_lo = || tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone);
        let anchor_hi = || tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood),
            anchor_hi(),
        ]);

        // Each case mutates scene_a → scene_b by ONE edit kind, all keeping the
        // composite extent (recentre) fixed via the anchors, so all are genuinely
        // incremental. Each is checked independently from a fresh wholesale build of A.
        let recolor = {
            let mut b = scene_a.clone();
            // In-place recolor of the interior Box (material change, same geometry).
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            // In-place resize of the interior Box (few dirty chunks around it).
            // Replace content + transform in place so the node keeps its arena id.
            let replacement = tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            // Move the interior Box from +60X to +70X (still interior → recentre
            // fixed; dirty around BOTH endpoints).
            b.root_node_mut(1).transform = document::scene::NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            // ADD a new INTERIOR tool (brand-new covering chunks; extent unchanged).
            b.add_node(tool_node(ShapeKind::Box, [3, 3, 3], [90, 0, 0], MaterialChoice::Stone));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            // REMOVE the interior Box (its chunks must be evicted/vacated; the
            // anchors keep the extent so the recentre is unchanged).
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node] {
            // Precondition: every edit keeps the recentre fixed (so the incremental
            // path applies — a recentre shift would force a full rebuild and the
            // dirty-count assertion below would not hold).
            assert_eq!(
                scene_a.recentre_voxels_for_resolve(density),
                scene_b.recentre_voxels_for_resolve(density),
                "[{label}] this edit must keep the composite recentre fixed"
            );

            // Incremental: wholesale-build A, then apply the single edit to B.
            let mut resolve_cache = Store::new();
            let mut render_cache: RenderCache = {
                let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
                chunks
                    .iter()
                    .filter(|(_, grid)| !grid.occupied.is_empty())
                    .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                    .collect()
            };
            let total_before = render_cache.len();
            let rebuilt = apply_incremental_edit(
                &mut render_cache,
                &mut resolve_cache,
                &scene_a,
                &scene_b,
                density,
            );

            // The full wholesale rebuild for the post-edit scene B (the truth).
            let full = full_render_cache(&scene_b, density);

            assert_eq!(
                render_cache, full,
                "[{label}] incremental render cache (coords + each chunk's instance \
                 multiset) MUST equal a full wholesale rebuild of scene B — a stale \
                 chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: a localised edit rebuilds strictly fewer chunks
            // than the scene's total resident chunks (proving it is incremental, not
            // a disguised full rebuild). `total_before` and `full.len()` are both the
            // scene's full per-chunk count (A and B differ by one localised node), so
            // a genuine incremental edit touches a strict subset.
            let scene_chunks = total_before.max(full.len());
            assert!(
                rebuilt < scene_chunks,
                "[{label}] a localised edit must rebuild strictly FEWER chunks \
                 ({rebuilt}) than the scene's total ({scene_chunks}) — else it is a \
                 disguised full rebuild"
            );
        }
    }

    /// A focused dirty-count assertion: an in-place recolor of ONE SMALL far-flung
    /// node dirties only the handful of chunks that node occupies, NOT the whole
    /// scene — so a localised edit rebuilds far fewer than half the resident chunks.
    #[test]
    fn localized_recolor_rebuilds_few_chunks() {
        let density = 16u32;
        // A wide sphere (many chunks) plus a tiny 1-block box pushed far out in X,
        // so the box owns only ~1 chunk no other leaf touches.
        let scene_a = Scene::from_nodes(vec![
            tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone),
            tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut resolve_cache = Store::new();
        let mut render_cache: RenderCache = {
            let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
            chunks
                .iter()
                .filter(|(_, grid)| !grid.occupied.is_empty())
                .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                .collect()
        };
        let total = render_cache.len();
        let rebuilt =
            apply_incremental_edit(&mut render_cache, &mut resolve_cache, &scene_a, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            rebuilt * 2 < total,
            "a localised recolor of a small node must rebuild far fewer than half the \
             chunks: rebuilt {rebuilt} of {total}"
        );
        // And the result still matches a full rebuild.
        assert_eq!(render_cache, full_render_cache(&scene_b, density));
    }

    // ===== Issue #20 Step 3: out-of-core spill to DiskChunkStore ==================

    /// A unique temp directory under the system temp dir, removed on drop so no spill
    /// test leaves disk litter (mirrors the disk-store tests' RAII guard).
    struct TempDir {
        path: std::path::PathBuf,
    }
    impl TempDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "voxelworker_chunk_cache_spill_test_{label}_{}_{unique}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            Self { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// The covering chunk coords of `scene` at `density`, in chunk order.
    fn covering_coords(scene: &Scene, density: u32) -> Vec<[i32; 3]> {
        let mut coords = Vec::new();
        if let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(density) {
            for chunk_z in min_chunk[2]..=max_chunk[2] {
                for chunk_y in min_chunk[1]..=max_chunk[1] {
                    for chunk_x in min_chunk[0]..=max_chunk[0] {
                        coords.push([chunk_x, chunk_y, chunk_z]);
                    }
                }
            }
        }
        coords
    }

    /// (a) A chunk fetched AFTER being spilled-and-reloaded is BYTE-IDENTICAL (every
    /// f32 position bit + material) to the resident result — the spill/reload round-trip
    /// is transparent. A capacity of 1 forces every-other access to spill the prior.
    #[test]
    fn spilled_and_reloaded_chunk_is_byte_identical() {
        let density = 16u32;
        // A scene spread across several chunks in X so we have >1 covering chunk.
        let scene = three_tool_scene(density, 40);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2, "need at least two covering chunks to force a spill");

        // The reference: every covering chunk's grid from an UNBOUNDED cache (no spill).
        let mut reference = Store::new();
        let expected: std::collections::HashMap<[i32; 3], _> = coords
            .iter()
            .map(|&coord| (coord, occupied_multiset(reference.chunk(coord, &scene, density, 0))))
            .collect();

        // A capacity-1 spilling cache: fetch every coord once (filling + spilling), then
        // re-fetch every coord — each re-fetch reloads from disk (or recomputes) and
        // must equal the unbounded reference byte-for-byte.
        let temp = TempDir::new("byte_identical");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();
        for &coord in &coords {
            let _ = cache.chunk(coord, &scene, density, 0);
            assert!(cache.resident_chunk_count() <= 1, "capacity 1 keeps at most one resident");
        }
        assert!(cache.spill_count() >= 1, "filling past capacity 1 must spill");

        for &coord in &coords {
            let got = occupied_multiset(cache.chunk(coord, &scene, density, 0));
            assert_eq!(
                got, expected[&coord],
                "chunk {coord:?} after spill+reload must be byte-identical to the resident result"
            );
        }
    }

    /// (b) The resident cap is honored: under sustained load over many chunks the
    /// resident count NEVER exceeds the cap, and every chunk remains correct.
    #[test]
    fn resident_cap_is_never_exceeded() {
        let density = 16u32;
        let cap = 3usize;
        let scene = three_tool_scene(density, 80); // a wide spread → many chunks.
        let coords = covering_coords(&scene, density);
        assert!(coords.len() > cap, "the scene must have more chunks than the cap to exercise spill");

        let temp = TempDir::new("cap_honored");
        let mut cache = Store::with_resident_cap(cap, &temp.path).unwrap();
        // Repeat the sweep twice so reloads (which also insert) are stress-tested.
        for _ in 0..2 {
            for &coord in &coords {
                let _ = cache.chunk(coord, &scene, density, 0);
                assert!(
                    cache.resident_chunk_count() <= cap,
                    "resident count {} exceeded cap {cap}",
                    cache.resident_chunk_count()
                );
            }
        }
        assert_eq!(cache.resident_chunk_count(), cap.min(coords.len()), "fills to the cap");
    }

    /// (c) LRU order: the LEAST-recently-used chunk is the one spilled. Touch A, then B,
    /// then fetch a third over a cap of 2 — A (the LRU) is the spill victim, not B.
    #[test]
    fn least_recently_used_chunk_is_spilled() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 3, "need at least three covering chunks");
        let (a, b, c) = (coords[0], coords[1], coords[2]);

        let temp = TempDir::new("lru_order");
        let mut cache = Store::with_resident_cap(2, &temp.path).unwrap();

        // Fetch A then B (both resident, cap 2); A is now the LRU.
        let _ = cache.chunk(a, &scene, density, 0);
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!(cache.resident_chunk_count(), 2);
        assert_eq!(cache.spill_count(), 0, "two chunks fit the cap of 2");

        // Fetch C over capacity → A (the LRU) is spilled, B stays resident.
        let _ = cache.chunk(c, &scene, density, 0);
        assert_eq!(cache.spill_count(), 1, "exactly one chunk spilled");
        assert_eq!(cache.resident_chunk_count(), 2);

        // Re-fetch B: resident → NO reload. Re-fetch A: spilled → exactly one reload.
        let reloads_before = cache.disk_reload_count();
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before,
            "B stayed resident (A was the LRU victim) — no reload"
        );
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before + 1,
            "A was the spilled LRU — fetching it reloads exactly once"
        );
    }

    /// (d) Invalidation purges BOTH resident and disk: a spilled chunk that an edit
    /// dirties must NOT resurface (a later fetch recomputes it, it does not reload the
    /// stale disk copy). Verified through both `invalidate_chunk` and `invalidate_aabb`.
    #[test]
    fn invalidation_purges_resident_and_disk() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2);

        // --- invalidate_chunk path ---
        let temp = TempDir::new("invalidate_chunk");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();
        // Fill so coords[0] gets spilled to disk (cap 1, fetch a second coord after).
        let _ = cache.chunk(coords[0], &scene, density, 0);
        let _ = cache.chunk(coords[1], &scene, density, 0);
        assert!(cache.spill_count() >= 1, "coords[0] must be spilled to disk");
        let reloads_before = cache.disk_reload_count();

        // Invalidate the spilled coord, then fetch it: it must RECOMPUTE, not reload.
        cache.invalidate_chunk(coords[0]);
        let recomputes_before = cache.recompute_count();
        let _ = cache.chunk(coords[0], &scene, density, 0);
        assert_eq!(
            cache.disk_reload_count(), reloads_before,
            "an invalidated spilled chunk must NOT reload the stale disk copy"
        );
        assert_eq!(
            cache.recompute_count(), recomputes_before + 1,
            "the invalidated chunk is recomputed from the scene"
        );

        // --- invalidate_aabb path ---
        let temp2 = TempDir::new("invalidate_aabb");
        let mut cache2 = Store::with_resident_cap(1, &temp2.path).unwrap();
        let _ = cache2.resolve_region(&scene, density, 0); // resolves + spills all but one.
        assert!(cache2.spill_count() >= 1, "resolve_region over cap 1 must spill");

        // An edit AABB spanning the whole covering chunk grid purges every chunk
        // (resident + disk). The AABB is in absolute (producer-true) voxels, the frame
        // `invalidate_aabb` expects.
        let region_aabb = {
            let (lo, hi) = scene.covering_chunk_range(density).unwrap();
            let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * density) as i64;
            let min_v = [
                lo[0] as i64 * chunk_extent,
                lo[1] as i64 * chunk_extent,
                lo[2] as i64 * chunk_extent,
            ];
            let max_v = [
                (hi[0] as i64 + 1) * chunk_extent,
                (hi[1] as i64 + 1) * chunk_extent,
                (hi[2] as i64 + 1) * chunk_extent,
            ];
            voxel_core::spatial_index::VoxelAabb::new(min_v, max_v)
        };
        let _ = cache2.invalidate_aabb(&region_aabb, density);
        let reloads_before2 = cache2.disk_reload_count();
        // Re-resolve: every chunk must recompute, none reload a purged disk copy.
        let _ = cache2.resolve_region(&scene, density, 0);
        assert_eq!(
            cache2.disk_reload_count(), reloads_before2,
            "after invalidate_aabb no chunk reloads a stale spilled copy"
        );
    }

    /// (e) Counters tally an expected access sequence: spill / reload / recompute counts
    /// match a hand-traced sequence over a capacity-1 cache and two distinct chunks.
    #[test]
    fn counters_tally_an_expected_access_sequence() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let coords = covering_coords(&scene, density);
        assert!(coords.len() >= 2);
        let (a, b) = (coords[0], coords[1]);

        let temp = TempDir::new("counters");
        let mut cache = Store::with_resident_cap(1, &temp.path).unwrap();

        // 1. Fetch A (miss in both → recompute 1; nothing to spill yet).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (1, 0, 0));

        // 2. Fetch A again (resident hit → no counter moves).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (1, 0, 0));

        // 3. Fetch B (recompute 2; inserting over cap 1 spills A → spill 1).
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 1, 0));

        // 4. Fetch A (spilled → reload 1; inserting over cap 1 spills B → spill 2).
        let _ = cache.chunk(a, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 2, 1));

        // 5. Fetch B (spilled → reload 2; spills A → spill 3). No recompute (both exist).
        let _ = cache.chunk(b, &scene, density, 0);
        assert_eq!((cache.recompute_count(), cache.spill_count(), cache.disk_reload_count()), (2, 3, 2));
    }

    /// An unbounded cache (the default `new()`) NEVER spills, reloads or tracks LRU —
    /// proving the live path / goldens are untouched by Step 3.
    #[test]
    fn unbounded_cache_never_spills() {
        let density = 16u32;
        let scene = three_tool_scene(density, 80);
        let mut cache = Store::new();
        let _ = cache.resolve_region(&scene, density, 0);
        assert!(cache.resident_chunk_count() > 1, "an unbounded cache keeps every chunk resident");
        assert_eq!(cache.spill_count(), 0);
        assert_eq!(cache.disk_reload_count(), 0);
        assert!(cache.recompute_count() > 0, "recompute count tracks first-time resolves");
    }

    /// A zero resident cap is rejected at construction (a cache that holds nothing
    /// resident is a misconfiguration).
    #[test]
    fn zero_resident_cap_panics() {
        let temp = TempDir::new("zero_cap");
        let result = std::panic::catch_unwind(|| {
            Store::with_resident_cap(0, &temp.path)
        });
        assert!(result.is_err(), "a zero resident cap must panic");
    }
