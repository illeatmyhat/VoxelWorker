use super::*;

    // ===== ADR 0010 E4: cacheless STREAMING diameter / widest-run query ===========

    /// The whole-grid diameter readout — today's reference value the streamed query
    /// must reproduce (same as `store.rs::whole_grid_widest_run`).
    fn whole_grid_widest_run(scene: &Scene, vpb: u32, band: (u32, u32)) -> u32 {
        let region = scene.full_extent_blocks(vpb);
        let grid = scene.resolve_region(region, vpb, 0);
        grid.widest_run_in_band(band.0, band.1)
    }

    /// **THE E4 diameter PARITY GATE:** the STREAMED widest-run (coarse blocks accounted
    /// ANALYTICALLY, boundary per-voxel) equals today's dense
    /// `VoxelGrid::widest_run_in_band` for the gated scene, across a spread of bands.
    /// Mirrors `store.rs::assert_region_widest_run_matches_whole_grid`.
    fn assert_streamed_widest_run_matches_dense(scene: &Scene, vpb: u32, label: &str) {
        let dims = scene.placed_region_dimensions(vpb);
        let grid_z = dims[2];
        let mid = grid_z.saturating_sub(1) / 2;
        let bands = [
            (0, grid_z.saturating_sub(1)),
            (0, 0),
            (grid_z.saturating_sub(1), grid_z.saturating_sub(1)),
            (mid, mid),
            (mid, (mid + 2).min(grid_z.saturating_sub(1))),
            (grid_z + 10, grid_z + 20),
        ];
        let store = TwoLayerStore::enabled();
        for band in bands {
            let expected = whole_grid_widest_run(scene, vpb, band);
            let actual = streamed_widest_run_in_band(&store, scene, vpb, band.0, band.1)
                .expect("the two-layer capability is enabled");
            assert_eq!(
                actual, expected,
                "[{label}] streamed widest_run band {band:?} must equal the dense readout"
            );
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_all_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            assert_streamed_widest_run_matches_dense(&scene, 16, &format!("{kind:?}"));
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_flat_and_odd_shapes() {
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
                assert_streamed_widest_run_matches_dense(&scene, 16, &format!("{kind:?} {size:?}"));
            }
        }
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_demo_scene() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, density),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "demo-scene");
    }

    #[test]
    fn streamed_widest_run_matches_dense_for_demo_village() {
        let density = 16;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, density);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, density);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = NodeTransform::from_blocks(offset, density);
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
        assert_streamed_widest_run_matches_dense(&scene, density, "demo-village");
    }

    /// A sketch-revolve solid (boundary-only) — its diameter streams identically.
    #[test]
    fn streamed_widest_run_matches_dense_for_sketch_solid() {
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let density = 16;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_streamed_widest_run_matches_dense(&scene, density, "sketch-revolve");
    }

    /// An OVERLAP multi-material scene (overlap blocks classify boundary) streams the
    /// same widest-run as the dense readout — a run crossing a coarse↔boundary seam is
    /// one contiguous span.
    #[test]
    fn streamed_widest_run_matches_dense_for_overlap_multi_material() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 4),
            make_tool_density(ShapeKind::Box, [2, 0, 0], MaterialChoice::Wood, density, 4),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "overlap-multi-material");
    }

    /// **Band-at-a-time interval fold parity (the OOM-fix guard):** two solid boxes
    /// separated along X give every covering row TWO disjoint occupied runs (a coalescing
    /// bug would merge them across the gap and report a doubled diameter); a torus adds
    /// boundary blocks that seam with the coarse interiors; and the helper's single-Z-slice
    /// bands clip blocks mid-row. The streamed interval fold must still match the dense
    /// oracle exactly across every band.
    #[test]
    fn streamed_widest_run_matches_dense_for_disjoint_runs_and_mixed_blocks() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [10, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 8, 0], MaterialChoice::Plain, density),
        ]);
        assert_streamed_widest_run_matches_dense(&scene, density, "disjoint-runs-mixed-blocks");
    }

    /// **COARSE + BOUNDARY in the SAME block-row, band slicing mid-block (the block-row
    /// dedup guard).** A small SOLID box has, along an interior `(block_y, block_z)`
    /// block-row, boundary FACE blocks at both X extremes flanking coarse INTERIOR blocks —
    /// so the dedup must both (a) count the block-row's coarse run once and (b) refine the
    /// boundary voxel rows per-voxel and MERGE them across the coarse↔boundary seam. The
    /// helper's single-Z-slice bands `(0,0)` / `(mid,mid)` cut a 16-voxel-tall block mid-height
    /// (a partial block layer), exercising the block-row's band clip. Must match the dense
    /// oracle exactly across every band.
    #[test]
    fn streamed_widest_run_matches_dense_for_coarse_and_boundary_in_same_block_row() {
        let density = 16;
        // A 4-block solid cube: interior blocks classify coarse, the six faces boundary, so an
        // interior block-row is boundary-coarse-…-coarse-boundary along X.
        let scene = Scene::from_nodes(vec![make_tool_density(
            ShapeKind::Box,
            [0, 0, 0],
            MaterialChoice::Stone,
            density,
            4,
        )]);
        assert_streamed_widest_run_matches_dense(&scene, density, "coarse+boundary-same-block-row");
    }

    /// **6M-CAP DISSOLUTION (query side):** the streamed diameter of an
    /// 800×800-revolve-class solid is accounted with coarse blocks ANALYTICALLY (no
    /// per-voxel expansion), so the whole-region densify the dense path needs never
    /// happens. We assert the streamed widest run equals the box's true 800-voxel face
    /// width and quantify the analytic saving (coarse cells vs per-voxel cells avoided).
    #[test]
    fn streamed_widest_run_dissolves_6m_cap_with_analytic_coarse() {
        let density = 16u32;
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, density);
        assert!(
            shape.exceeds_voxel_cap(density),
            "the large solid must exceed the dense 6M cap to prove the point"
        );
        let node = Node::new("BigBox", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        let scene = Scene::from_nodes(vec![node]);
        let dims = scene.placed_region_dimensions(density);
        let band = (0, dims[2].saturating_sub(1));
        let true_width = blocks * density; // 800-voxel face row.

        let store = TwoLayerStore::enabled();
        let widest = streamed_widest_run_in_band(&store, &scene, density, band.0, band.1)
            .expect("the two-layer capability is enabled");
        assert_eq!(
            widest, true_width,
            "the streamed diameter must report the solid box's true 800-voxel width"
        );

        // Quantify the analytic saving: count coarse-solid blocks (accounted by run +=
        // d, NO per-voxel expansion) vs boundary blocks (per-voxel). Each coarse block
        // elides d³ per-voxel cells from the scan.
        let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
        let mut coarse_blocks = 0u64;
        let mut boundary_blocks = 0u64;
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = store
                        .build_chunk([chunk_x, chunk_y, chunk_z], &scene, density, 0)
                        .unwrap();
                    for bz in 0..CHUNK_BLOCKS {
                        for by in 0..CHUNK_BLOCKS {
                            for bx in 0..CHUNK_BLOCKS {
                                let block = [bx, by, bz];
                                if chunk.coarse_block(block).is_some() {
                                    coarse_blocks += 1;
                                } else if chunk.microblocks.contains_key(&block) {
                                    boundary_blocks += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        let cells_per_block = (density as u64).pow(3);
        let analytic_cells_elided = coarse_blocks * cells_per_block;
        assert!(
            coarse_blocks > boundary_blocks,
            "a large solid box must be mostly coarse blocks (coarse {coarse_blocks} > \
             boundary {boundary_blocks})"
        );
        eprintln!(
            "E4 analytic diameter: {coarse_blocks} coarse blocks (accounted run += d, \
             {analytic_cells_elided} per-voxel cells ELIDED) vs {boundary_blocks} boundary \
             blocks (per-voxel); dense path would densify all {} region voxels",
            (blocks as u64 * density as u64).pow(3)
        );
    }

