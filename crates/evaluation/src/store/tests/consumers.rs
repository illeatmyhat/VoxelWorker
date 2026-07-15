use super::*;

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

