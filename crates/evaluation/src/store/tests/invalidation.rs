use super::*;

    // ===== Issue #27 S3: targeted edit-AABB invalidation ========================


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

