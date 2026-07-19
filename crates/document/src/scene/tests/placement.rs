use super::*;
use crate::scene::producers::{leaf_producer_grid_voxels, outset_voxels_at};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::ShapeKind;
use crate::voxel::GeometryParams;
use crate::voxel::SdfShape;

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
            (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block) as i64;
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
        let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS as i64) * density;
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
        query: &voxel_core::spatial_index::VoxelAabb,
    ) -> Vec<voxel_core::spatial_index::VoxelAabb> {
        let mut matched = Vec::new();
        scene.for_each_leaf(&mut |world_offset_voxels, content, _grid_on_faces, _operation, outset, _scope_path| {
            let outset_voxels = outset_voxels_at(outset, voxels_per_block);
            let world_offset_voxels: [i64; 3] =
                std::array::from_fn(|axis| world_offset_voxels[axis] - outset_voxels);
            let Some(grid_voxels) = leaf_producer_grid_voxels(content, voxels_per_block, outset_voxels) else {
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
            let aabb = voxel_core::spatial_index::VoxelAabb::new(min, max);
            if aabb.intersects(query) {
                matched.push(aabb);
            }
        });
        matched
    }

    fn sorted_aabbs(
        mut boxes: Vec<voxel_core::spatial_index::VoxelAabb>,
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
        use voxel_core::spatial_index::VoxelAabb;
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
        assert_eq!(recoloured, voxel_core::spatial_index::VoxelAabb::new([0, 0, 0], [80, 80, 80]));
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

    /// A region-spanning VoxelBody edit can't be localised: the diff returns `None`.
    #[test]
    fn edit_aabb_diff_part_edit_is_none() {
        let voxels_per_block = 16;
        // A scene with a Tool plus a debug-cloud VoxelBody.
        let mut tool = Node::new(
            "Sphere",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Sphere, [5, 5, 5], 1, voxels_per_block),
                material: MaterialChoice::Stone,
            },
        );
        tool.transform = NodeTransform::from_blocks([0, 0, 0], voxels_per_block);
        let voxel_body = Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 1 }));
        let scene_a = scene_with_top_level_selected(Scene::from_nodes(vec![tool.clone(), voxel_body]), 0);
        let index_a = scene_a.build_leaf_spatial_index(voxels_per_block);
        assert!(index_a.has_region_spanning_leaf);

        // Change the VoxelBody's seed (a region-spanning content change).
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).content = NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 2 });
        let index_b = scene_b.build_leaf_spatial_index(voxels_per_block);
        assert_eq!(
            index_b.edit_aabb_since(&index_a),
            None,
            "editing a region-spanning VoxelBody forces a wholesale clear"
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

