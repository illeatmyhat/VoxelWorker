use super::*;

    /// Canonicalise an occupied set into the **resolved occupancy SET**: a map from each
    /// bit-exact voxel position to the block id of the LAST (document-order) writer at that
    /// position. This is the ADR 0010 parity-gate canonical form (the resolved occupancy
    /// SET keyed by position+block_id) — it differs from the dense store's
    /// `cache_region_matches_monolithic_*` MULTISET only at positions where leaves overlap.
    ///
    /// The two-layer store is a one-id-per-cell representation (a boundary block resolves
    /// to a dense region where the later leaf overwrites the earlier — Union "later wins"),
    /// so it never carries the dense path's DUPLICATE Vec entries at a shared position. The
    /// dense `Scene::resolve_region` emits leaves in document order, so the LAST entry at a
    /// position is the winner there too — taking the last writer on BOTH sides compares the
    /// true resolved occupancy. For every non-overlapping scene (all the SDF-shape /
    /// flat-odd cases) each position has exactly one writer, so this is byte-identical to
    /// the dense multiset; only genuinely-overlapping leaves (cloud-over-box) differ, and
    /// there the resolved-set is the correct comparison.
    ///
    /// Keying on the raw `f32` bits (`to_bits`) asserts the BYTES a consumer reads are
    /// identical, not merely the rounded voxel set.
    fn resolved_occupancy_set(
        grid: &VoxelGrid,
    ) -> std::collections::BTreeMap<[u32; 3], u16> {
        let mut set = std::collections::BTreeMap::new();
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
            ];
            // Last document-order writer wins (Union later-wins material).
            set.insert(key, voxel.color_index());
        }
        set
    }

    /// THE GATE (parity (a)): the two-layer round-trip occupancy (coarse fast-fill +
    /// boundary per-voxel) is BIT-IDENTICAL (position + block id) to the dense
    /// `Scene::resolve_region`, for the gated scene. Mirrors
    /// `store.rs::cache_region_matches_monolithic_*`. Returns the chunk + cell counts the
    /// build classified (so the harness can report coverage).
    pub(super) fn assert_two_layer_round_trip_matches_dense(
        scene: &Scene,
        voxels_per_block: u32,
        label: &str,
    ) -> (usize, u64) {
        let dense = scene.resolve_region(
            scene.full_extent_blocks(voxels_per_block),
            voxels_per_block,
            0,
        );
        let store = TwoLayerStore::enabled();
        let assembled = resolve_region_two_layer(&store, scene, voxels_per_block, 0)
            .expect("the capability is enabled");

        assert_eq!(
            assembled.dimensions, dense.dimensions,
            "[{label}] two-layer round-trip dimensions must match dense resolve_region"
        );
        assert_eq!(
            assembled.recentre_voxels, dense.recentre_voxels,
            "[{label}] two-layer round-trip must carry the SAME recentre as dense"
        );
        let dense_set = resolved_occupancy_set(&dense);
        let assembled_set = resolved_occupancy_set(&assembled);
        assert_eq!(
            assembled_set.len(),
            dense_set.len(),
            "[{label}] two-layer resolved occupancy count must match dense (the dense Vec \
             may hold duplicate entries at overlap positions; the resolved SET must agree)"
        );
        assert_eq!(
            assembled_set, dense_set,
            "[{label}] two-layer round-trip resolved occupancy SET (position + block id, \
             last-writer-wins) must be BIT-IDENTICAL to the dense resolve_region"
        );

        // Coverage accounting: count chunks + blocks the build classified.
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .unwrap_or(([0; 3], [-1; 3]));
        let chunks = if max_chunk[0] < min_chunk[0] {
            0
        } else {
            ((max_chunk[0] - min_chunk[0] + 1)
                * (max_chunk[1] - min_chunk[1] + 1)
                * (max_chunk[2] - min_chunk[2] + 1)) as usize
        };
        let cells = chunks as u64 * (CHUNK_BLOCKS as u64).pow(3);
        (chunks, cells)
    }

    #[test]
    fn round_trip_matches_dense_for_all_shapes() {
        let mut total_chunks = 0usize;
        let mut total_cells = 0u64;
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16);
            let (chunks, cells) =
                assert_two_layer_round_trip_matches_dense(&scene, 16, &format!("{kind:?}"));
            total_chunks += chunks;
            total_cells += cells;
        }
        eprintln!(
            "two-layer parity (all shapes): {total_chunks} chunks, {total_cells} block cells"
        );
    }

    /// FLAT / odd-sized shapes — the S0 covering-range regression case (a 1-block axis
    /// straddles two chunks). The classifier must cover the producer-true voxel extent
    /// and round-trip bit-identically, just as the dense net pins.
    #[test]
    fn round_trip_matches_dense_for_flat_and_odd_shapes() {
        let mut total_cells = 0u64;
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
                let (_chunks, cells) = assert_two_layer_round_trip_matches_dense(
                    &scene,
                    16,
                    &format!("{kind:?} {size:?}"),
                );
                total_cells += cells;
            }
        }
        eprintln!("two-layer parity (flat/odd): {total_cells} block cells");
    }

    #[test]
    fn round_trip_matches_dense_for_demo_scene() {
        let density = 16;
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, density),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, density),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, density),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-scene");
    }

    #[test]
    fn round_trip_matches_dense_for_demo_village() {
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
        assert_two_layer_round_trip_matches_dense(&scene, density, "demo-village");
    }

    /// A sketch-revolve solid (the 800×800-revolve CLASS that stressed the dense cap): the
    /// interior now ELIDES to coarse-solid blocks (ADR 0010 rollout) while the round-trip
    /// stays bit-identical to the dense store — pinning the coarse + boundary composition
    /// exact.
    #[test]
    fn round_trip_matches_dense_for_sketch_revolve() {
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
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve");
    }

    /// A PARTIAL-turn revolve with an AXIS-STRADDLING profile (radial spans negative→positive,
    /// so the resolve's mirrored `−radius` union is live) — the ADR 0010 partial-sweep coarse
    /// test must round-trip bit-identically to the dense oracle: interior blocks inside the
    /// swept arc elide to coarse, the excluded wedge stays boundary/air, and the mirrored
    /// occupancy is reproduced exactly.
    #[test]
    fn round_trip_matches_dense_for_partial_revolve() {
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
        let density = 16;
        // Radial (c1) straddles the axis: [-20, 20]; axial (c0) [8, 56].
        let profile = Sketch::new(
            PlaneAxis::Z,
            vec![
                SketchPoint::new(8, -20),
                SketchPoint::new(56, -20),
                SketchPoint::new(56, 20),
                SketchPoint::new(8, 20),
            ],
        );
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 135);
        let node = Node::new(
            "PartialRevolve",
            NodeContent::SketchTool {
                producer,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-partial-revolve");
    }

    /// A region-spanning UNBOUNDABLE producer (the fBm cloud field) forces every covering
    /// block BOUNDARY (its `cell_field_interval` is `None`) and STILL round-trips
    /// bit-identically — the "unboundable ops fall back, still exact" acceptance criterion.
    /// (Mixed with a Tool so the scene has a composite chunk extent.)
    #[test]
    fn round_trip_matches_dense_with_unboundable_cloud() {
        use document::scene::VoxelBody;
        let density = 16;
        let mut cloud = Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }));
        cloud.transform = NodeTransform::from_blocks([0, 0, 0], density);
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density),
            cloud,
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, density, "tool+cloud");
    }

    /// INTERIOR ELISION (the whole point): a LARGE solid box stores ZERO interior voxels
    /// under the two-layer path — only its surface shell lives in the microblock layer,
    /// while its interior is coarse block ids. The dense path would densify all
    /// ~`(size·d)³` interior voxels (and a revolve-class size blows the 6M cap); the
    /// two-layer stored count is surface-only.
    #[test]
    fn large_solid_box_stores_zero_interior_voxels() {
        let density = 16;
        // 50×50×50 BLOCKS @ d16 = 800×800×800 voxels — the revolve-class size the ADR
        // calls out. Dense interior would be 800³ ≈ 5.1e8 voxels (far past the 6M cap);
        // the two-layer interior holds NONE.
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, density);
        let node = Node::new(
            "BigBox",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let store = TwoLayerStore::enabled();

        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(density)
            .expect("a placed box has a covering chunk range");

        let mut total_stored = 0u64;
        let mut interior_chunks = 0u64;
        let mut total_chunks = 0u64;
        // An interior chunk (no block of it touches a face of the box) is entirely
        // coarse-solid, so it must store ZERO voxels.
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = store
                        .build_chunk([chunk_x, chunk_y, chunk_z], &scene, density, 0)
                        .unwrap();
                    let stored = chunk.stored_voxel_count();
                    total_stored += stored;
                    total_chunks += 1;
                    let is_interior_chunk = chunk_x > min_chunk[0]
                        && chunk_x < max_chunk[0]
                        && chunk_y > min_chunk[1]
                        && chunk_y < max_chunk[1]
                        && chunk_z > min_chunk[2]
                        && chunk_z < max_chunk[2];
                    if is_interior_chunk {
                        interior_chunks += 1;
                        assert_eq!(
                            stored, 0,
                            "interior chunk ({chunk_x},{chunk_y},{chunk_z}) of a solid box \
                             must store ZERO voxels (interior elision), got {stored}"
                        );
                    }
                }
            }
        }
        let dense_interior_voxels = (blocks as u64 * density as u64).pow(3);
        assert!(
            interior_chunks > 0,
            "the box must be large enough to have fully-interior chunks"
        );
        // The stored voxels are the 1-block-thick SURFACE SHELL only (each surface block
        // is d³ = 4096 voxels at d16, so a 50²-face shell is legitimately ~12% of the
        // volume) — a fraction of the dense interior, and every FULLY-interior chunk
        // (asserted above) holds ZERO. The dense path would densify the whole volume and
        // blow the 6M cap; the two-layer path never builds the interior.
        assert!(
            total_stored < dense_interior_voxels / 4,
            "two-layer stored voxels ({total_stored}) must be well below the dense interior \
             volume ({dense_interior_voxels}) — surface-shell-only residency"
        );
        eprintln!(
            "interior elision: {total_chunks} chunks ({interior_chunks} fully interior); \
             two-layer stored {total_stored} voxels vs dense interior {dense_interior_voxels}"
        );
    }

    /// INTERIOR ELISION for the SKETCH producer — the completion of the ADR 0010 rollout.
    /// A SOLID extrude box and a full 360° revolve now classify their interiors
    /// COARSE-SOLID (dominating the surface-only boundary shell), and a CONCAVE L extrude
    /// elides its interior while keeping the reflex-corner block BOUNDARY and the removed
    /// quadrant AIR (proving the polygon test, not just axis-aligned rectangles). Every
    /// case also round-trips bit-identically to the dense oracle (the over-claim police).
    #[test]
    fn sketch_interior_elides_to_coarse_solid() {
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
        let density = 8u32;

        // Count (coarse, boundary) blocks across a producer's covering chunk range by
        // classifying every block directly (no per-voxel resolve → fast).
        let classify_scene = |scene: &Scene| -> (u64, u64) {
            let leaves = scene.leaf_producers(density);
            let leaves: Vec<&LeafProducer> = leaves.iter().collect();
            let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
            let chunk_extent = (CHUNK_BLOCKS * density) as i64;
            let block = density as i64;
            let (mut coarse, mut boundary) = (0u64, 0u64);
            for cz in min_chunk[2]..=max_chunk[2] {
                for cy in min_chunk[1]..=max_chunk[1] {
                    for cx in min_chunk[0]..=max_chunk[0] {
                        for bz in 0..CHUNK_BLOCKS {
                            for by in 0..CHUNK_BLOCKS {
                                for bx in 0..CHUNK_BLOCKS {
                                    let low = [
                                        cx as i64 * chunk_extent + bx as i64 * block,
                                        cy as i64 * chunk_extent + by as i64 * block,
                                        cz as i64 * chunk_extent + bz as i64 * block,
                                    ];
                                    let cell = VoxelAabb::new(
                                        low,
                                        [low[0] + block, low[1] + block, low[2] + block],
                                    );
                                    match classify_chunk_block(&leaves, cell, density) {
                                        BlockClassification::CoarseSolid(_) => coarse += 1,
                                        BlockClassification::Boundary => boundary += 1,
                                        BlockClassification::Air => {}
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (coarse, boundary)
        };

        // (1) SOLID extrude box, 8 blocks per axis (64³ voxels), BLOCK-ALIGNED: every block
        // is fully solid (the axis-aligned wall blocks too — their face lattice line is
        // collinear with the profile edge but every voxel centre is inside), so the whole
        // box is COARSE with ZERO boundary blocks (the sample-centre rectangle win).
        let edge = 8 * density as i64;
        let extrude =
            SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
        )]);
        let (coarse, boundary) = classify_scene(&scene);
        assert_eq!(
            boundary, 0,
            "a block-aligned solid box has NO boundary blocks (walls are fully solid ⇒ coarse)"
        );
        assert_eq!(
            coarse,
            (CHUNK_BLOCKS as u64 * 2).pow(3),
            "every block of the 8-block-per-axis box must be coarse-solid, got {coarse}"
        );
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-extrude-box");

        // (2) FULL 360° revolve (a solid cylinder, radial 3 blocks × axial 4 blocks):
        // interior near the axis elides to coarse.
        let revolve = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 3 * density as i64, 4 * density as i64),
            RevolveAxis::InPlane1,
            360,
        );
        let scene = Scene::from_nodes(vec![Node::new(
            "Cyl",
            NodeContent::SketchTool { producer: revolve, material: MaterialChoice::Stone },
        )]);
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "full 360 revolve must elide interior blocks to coarse-solid");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve-cyl");

        // (3) CONCAVE L extrude (notch corner at voxel 20 = mid-block at d8, so the reflex
        // edges CUT a block): interior elides, the reflex-corner block stays boundary, and
        // the removed quadrant is NOT coarse (a plain rectangle would over-claim it solid).
        let l_profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(32, 0),
            SketchPoint::new(32, 20),
            SketchPoint::new(20, 20), // reflex vertex, mid-block
            SketchPoint::new(20, 32),
            SketchPoint::new(0, 32),
        ];
        let l = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 24);
        let scene = Scene::from_nodes(vec![Node::new(
            "L",
            NodeContent::SketchTool { producer: l, material: MaterialChoice::Wood },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        // Deep inside the bottom bar (not touching any face) ⇒ coarse.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([8, 8, 8], [16, 16, 16]), density),
            BlockClassification::CoarseSolid(MaterialChoice::Wood.block_id()),
            "an interior L block must be coarse-solid"
        );
        // The block the reflex edges cut through ([16,24)² in-plane, spanning y=20 & x=20)
        // ⇒ boundary (a coarse claim would over-fill the notch).
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([16, 16, 8], [24, 24, 16]), density),
            BlockClassification::Boundary,
            "the L reflex-corner block must stay boundary"
        );
        // The removed top-right quadrant ([24,32)² in-plane) is EMPTY, and the metric cell
        // bracket proves it outright: the block overlaps the producer AABB, but its distance
        // to the L polygon is positive throughout, so the whole block elides to AIR without a
        // per-voxel resolve. It was BOUNDARY while the bracket carried sentinels rather than
        // distances — air could then only be claimed for a cell wholly outside the AABB, so a
        // notch inside the AABB had to fall back to resolving. Crucially it is still NOT
        // coarse-solid, which is what a naive bbox-solid claim would have wrongly returned.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([24, 24, 8], [32, 32, 16]), density),
            BlockClassification::Air,
            "the removed L quadrant must be AIR (the polygon excludes it), never coarse-solid"
        );
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "the L extrude must still elide its solid interior");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-L-extrude");

        // (4) NON-BLOCK-ALIGNED interior edge: a right-triangle profile whose hypotenuse
        // (x + y = 24) cuts through block INTERIORS at d8. A block the hypotenuse crosses
        // stays BOUNDARY; a block fully below it goes coarse — proving the sample-centre
        // test still distinguishes true-boundary blocks from fully-solid axis-aligned walls.
        let triangle = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(24, 0),
            SketchPoint::new(0, 24),
        ];
        let tri = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, triangle), 24);
        let scene = Scene::from_nodes(vec![Node::new(
            "Tri",
            NodeContent::SketchTool { producer: tri, material: MaterialChoice::Stone },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        // Fully below the hypotenuse (max x+y = 15 < 24) ⇒ coarse.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([0, 0, 8], [8, 8, 16]), density),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "a block fully below the triangle hypotenuse must be coarse-solid"
        );
        // The hypotenuse passes through this block's interior ⇒ boundary (not coarse).
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([8, 8, 8], [16, 16, 16]), density),
            BlockClassification::Boundary,
            "a block the hypotenuse cuts through the interior of must stay boundary"
        );
        let (coarse, _) = classify_scene(&scene);
        assert!(coarse > 0, "the triangle extrude must still elide its solid interior");
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-triangle-extrude");

        // (5) PARTIAL 270° revolve (a fat cylinder WEDGE, radial 6 blocks × axial 4 blocks):
        // closing the ADR 0010 deferral — a partial sweep now elides its interior via the
        // angular-containment coarse test. Before this fix `revolve_cell_is_solid` returned
        // false for every partial-turn cell, so a wedge densified its WHOLE interior (0 coarse
        // blocks); now interior blocks fully inside the [0°, 270°] arc AND the radial/axial
        // profile classify coarse-solid, while the excluded fourth quadrant (270°–360°) stays
        // boundary/air. The round-trip stays bit-identical to the dense oracle.
        let wedge = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 6 * density as i64, 4 * density as i64),
            RevolveAxis::InPlane1,
            270,
        );
        let scene = Scene::from_nodes(vec![Node::new(
            "Wedge",
            NodeContent::SketchTool { producer: wedge, material: MaterialChoice::Stone },
        )]);
        let (coarse, boundary) = classify_scene(&scene);
        assert!(
            coarse > 0,
            "a PARTIAL 270° revolve wedge must now elide interior blocks to coarse-solid \
             (the ADR 0010 partial-sweep deferral is closed), got {coarse} coarse / {boundary} boundary"
        );
        assert_two_layer_round_trip_matches_dense(&scene, density, "sketch-revolve-wedge");
    }

    /// A fully-interior block of a solid box classifies COARSE-SOLID (no voxels); a block
    /// straddling the box face classifies BOUNDARY; a block well outside classifies AIR.
    #[test]
    fn classifier_sorts_air_coarse_and_boundary() {
        let density = 8u32;
        // A 5×5×5-block box at the origin → voxel extent [0, 40) per axis.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = density as i64;

        // A deep-interior block ([16,24) on each axis, well inside [0,40)) is coarse-solid.
        let interior = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, interior, density),
            BlockClassification::CoarseSolid(MaterialChoice::Wood.block_id()),
            "a deep-interior block of a solid box must be coarse-solid at its material"
        );

        // A block straddling the +X face (the box ends at voxel 40; block [40−4,40+4)).
        let straddle = VoxelAabb::new([36, 16, 16], [36 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, straddle, density),
            BlockClassification::Boundary,
            "a block straddling the box surface must be boundary"
        );

        // A block far outside the box ([200, 208) on X) is air.
        let outside = VoxelAabb::new([200, 16, 16], [200 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, outside, density),
            BlockClassification::Air,
            "a block well outside every leaf must be air"
        );
    }

    /// Sculpt-touched / multi-leaf-overlap conservatism: a block where TWO Tools overlap
    /// is forced BOUNDARY (the Union's per-voxel later-wins material is not coarsely
    /// decidable), even if geometrically solid — still exact after per-voxel.
    #[test]
    fn overlapping_leaves_force_boundary() {
        let density = 8u32;
        // Two boxes overlapping at the origin region, different materials.
        let scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 5),
            make_tool_density(ShapeKind::Box, [1, 0, 0], MaterialChoice::Wood, density, 5),
        ]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = density as i64;
        // A block in the overlap region of both boxes.
        let overlap = VoxelAabb::new([16, 16, 16], [16 + block, 16 + block, 16 + block]);
        assert_eq!(
            classify_chunk_block(&leaves, overlap, density),
            BlockClassification::Boundary,
            "a block two leaves both fill must be boundary (per-voxel material resolution)"
        );
    }

    /// **CHUNK-GRANULAR FAST-PATH BYTE-IDENTITY (ADR 0010 Decision 2).** Over every covering
    /// chunk of a battery of mixed scenes, the whole-chunk interval fast path
    /// ([`build_two_layer_chunk_from_leaves`]) produces a `TwoLayerChunk` BYTE-IDENTICAL to
    /// the forced per-block sweep ([`build_two_layer_chunk_per_block`]) — coarse layer +
    /// overlay + microblock maps + seam flags. This pins the fast path's
    /// CONSERVATIVE-NEVER-NARROW contract directly (the round-trip-vs-dense gates check
    /// occupancy; this checks the exact two-layer STRUCTURE the fast path claims).
    ///
    /// The scenes exercise every fast-path arm: solid interiors (whole-chunk COARSE), the
    /// surface shell + concave/diagonal profiles (whole-chunk BOUNDARY → per-block),
    /// multi-leaf overlaps with DIFFERENT materials (uniformity guard forces per-block),
    /// `DebugClouds` (unboundable → per-block), and a partial revolve (angular ambiguity).
    #[test]
    fn whole_chunk_fast_path_matches_per_block_sweep() {
        use document::scene::VoxelBody;
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};

        // Assert fast-path == per-block over EVERY covering chunk of `scene`.
        fn assert_identical(scene: &Scene, density: u32, label: &str) {
            let leaves = scene.leaf_producers(density);
            let leaves: Vec<&LeafProducer> = leaves.iter().collect();
            let Some((min_chunk, max_chunk)) = scene.covering_chunk_range(density) else {
                return;
            };
            let chunk_extent = (CHUNK_BLOCKS * density) as i64;
            let mut coarse_chunks = 0u64;
            for cz in min_chunk[2]..=max_chunk[2] {
                for cy in min_chunk[1]..=max_chunk[1] {
                    for cx in min_chunk[0]..=max_chunk[0] {
                        let coord = [cx, cy, cz];
                        let fast = build_two_layer_chunk_from_leaves(coord, &leaves, density);
                        let chunk_min = [
                            cx as i64 * chunk_extent,
                            cy as i64 * chunk_extent,
                            cz as i64 * chunk_extent,
                        ];
                        let per_block =
                            build_two_layer_chunk_per_block(chunk_min, &leaves, density, density);
                        assert_eq!(
                            fast, per_block,
                            "[{label}] chunk {coord:?}: fast-path classification must be \
                             BYTE-IDENTICAL to the per-block sweep"
                        );
                        if fast.coarse.iter().all(Option::is_some) && !fast.coarse.is_empty() {
                            coarse_chunks += 1;
                        }
                    }
                }
            }
            eprintln!("[{label}] fast-path==per-block over all chunks ({coarse_chunks} all-coarse)");
        }

        let density = 8u32;

        // (a) SOLID sketch-extrude box — the whole-CHUNK-COARSE perf target (interior chunks
        // resolve in ONE interval call). Block-aligned so interiors AND walls are coarse.
        let edge = 8 * density as i64;
        let box_scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32),
                material: MaterialChoice::Stone,
            },
        )]);
        assert_identical(&box_scene, density, "sketch-extrude-box");

        // (b) SDF shapes — curved surfaces give a real boundary shell + coarse interiors,
        // and exercise the Lipschitz-centre interval's inclusion-monotonicity.
        for kind in [ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Cylinder, ShapeKind::Torus] {
            let scene = Scene::from_nodes(vec![make_tool_density(
                kind,
                [0, 0, 0],
                MaterialChoice::Stone,
                density,
                6,
            )]);
            assert_identical(&scene, density, &format!("sdf-{kind:?}"));
        }

        // (c) MULTI-LEAF overlap, DIFFERENT materials — the uniformity guard must force the
        // overlap chunks to per-block (Union later-wins material is not coarsely decidable).
        let overlap_scene = Scene::from_nodes(vec![
            make_tool_density(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone, density, 6),
            make_tool_density(ShapeKind::Box, [3, 0, 0], MaterialChoice::Wood, density, 6),
        ]);
        assert_identical(&overlap_scene, density, "multi-leaf-materials");

        // (d) DebugClouds — unboundable (`cell_field_interval == None`) ⇒ the whole chunk
        // falls back to per-block; every chunk must still match.
        let cloud_scene = Scene::from_nodes(vec![Node::new(
            "Clouds",
            NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }),
        )]);
        assert_identical(&cloud_scene, density, "debug-clouds");

        // (e) CONCAVE L extrude — reflex-corner + removed quadrant keep boundary/air chunks
        // adjacent to coarse interiors.
        let l_scene = Scene::from_nodes(vec![Node::new(
            "L",
            NodeContent::SketchTool {
                producer: SketchSolid::extrude(
                    Sketch::new(
                        PlaneAxis::Z,
                        vec![
                            SketchPoint::new(0, 0),
                            SketchPoint::new(32, 0),
                            SketchPoint::new(32, 20),
                            SketchPoint::new(20, 20),
                            SketchPoint::new(20, 32),
                            SketchPoint::new(0, 32),
                        ],
                    ),
                    24,
                ),
                material: MaterialChoice::Wood,
            },
        )]);
        assert_identical(&l_scene, density, "sketch-L-extrude");

        // (f) PARTIAL 270° revolve — angular ambiguity keeps the excluded wedge boundary/air
        // while the swept interior elides to coarse.
        let wedge_scene = Scene::from_nodes(vec![Node::new(
            "Wedge",
            NodeContent::SketchTool {
                producer: SketchSolid::revolve(
                    Sketch::rectangle(PlaneAxis::X, 6 * density as i64, 4 * density as i64),
                    RevolveAxis::InPlane1,
                    270,
                ),
                material: MaterialChoice::Stone,
            },
        )]);
        assert_identical(&wedge_scene, density, "sketch-revolve-wedge");
    }

    /// **#66 edit-broadphase exactness (belt-and-braces, the #63 gate carried over).** The
    /// per-chunk candidate set the BVH ([`leaf_edit_broadphase`]) hands each chunk MUST
    /// equal the naive "all leaves filtered by AABB-overlaps-chunk" set — leaf-index-
    /// identical, in document order. If they ever diverge, a chunk could be classified
    /// against the wrong candidate set and the two-layer output would drift from the dense
    /// path (which the parity gate would catch, but this pins the invariant directly at the
    /// broadphase boundary).
    #[test]
    fn broadphase_candidate_set_equals_naive_filter() {
        let density = 8u32;
        // A 4×4×4 grid of small boxes spaced 3 blocks apart — leaves land in many chunks,
        // some sharing a chunk (adjacency), so the candidate sets are non-trivial.
        let mut nodes = Vec::new();
        for grid_z in 0..4i64 {
            for grid_y in 0..4i64 {
                for grid_x in 0..4i64 {
                    nodes.push(make_tool_density(
                        ShapeKind::Box,
                        [grid_x * 3, grid_y * 3, grid_z * 3],
                        MaterialChoice::Stone,
                        density,
                        2,
                    ));
                }
            }
        }
        let scene = Scene::from_nodes(nodes);
        let leaves = scene.leaf_producers(density);
        let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
        let broadphase = leaf_edit_broadphase(&leaves, density);

        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let coord = [chunk_x, chunk_y, chunk_z];
                    let chunk_box = chunk_world_voxel_aabb(coord, density);
                    // Naive: every leaf whose world AABB overlaps this chunk's box, in
                    // document order (a filter — never a reorder).
                    let naive: Vec<usize> = leaves
                        .iter()
                        .enumerate()
                        .filter(|(_, leaf)| {
                            leaf_world_aabb(leaf, density).intersects(&chunk_box)
                        })
                        .map(|(index, _)| index)
                        .collect();
                    assert_eq!(
                        broadphase.overlapping_input_indices(&chunk_box),
                        naive,
                        "edit-broadphase candidates for chunk {coord:?} must equal the \
                         naive all-leaves-filtered set, in document order"
                    );
                }
            }
        }
    }

    /// SEAM-SOLIDITY flags: a boundary block's per-face flag matches its ACTUAL face
    /// occupancy. We resolve a boundary block of a solid box and assert the face that lies
    /// INSIDE the box is solid while the face that pokes OUT of the box is not.
    #[test]
    fn seam_solidity_flags_match_face_occupancy() {
        let density = 8u32;
        // A solid box [0,40) per axis. Take the block straddling the +X face: block
        // [32,40) on X (the last fully-inside block column is [32,40); the face at X=39
        // is the box's last solid layer, and X=40+ is air). To get a STRADDLING block on
        // a different axis we instead take a block at the +X edge whose low-X face is
        // solid (inside the box) and whose geometry is the surface shell.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 5, 5], 1, density);
        let scene = Scene::from_nodes(vec![Node::new(
            "Box",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        )]);
        let leaves = scene.leaf_producers(density);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();

        // The block [32,40) on X, interior on Y/Z ([16,24)). The box fills X∈[0,40), so
        // this whole block is solid — BUT it touches the +X face of the box, so its
        // classification depends on the conservative bound. Resolve it directly to read
        // its seam flags regardless of the coarse verdict.
        let block_min = [32i64, 16, 16];
        let geometry = resolve_boundary_block(&leaves, block_min, density, density);
        // The low-X face (X=32, the 0th local layer) is deep inside the box ⇒ fully solid.
        assert!(
            geometry.seam_solidity.face_is_solid(0, 0),
            "the low-X face of an interior-touching block must be solid"
        );
        // Every face of this all-solid block is solid (the box fully covers [32,40)³ here,
        // since Y,Z ∈ [16,24) ⊂ [0,40) and X ∈ [32,40) ⊂ [0,40)).
        for axis in 0..3 {
            for side in 0..2 {
                assert!(
                    geometry.seam_solidity.face_is_solid(axis, side),
                    "a fully-solid block must report every face solid (axis {axis}, side {side})"
                );
            }
        }

        // A block straddling the +X surface (X∈[36,44), so X∈[40,44) is OUTSIDE the box):
        // its low-X face (X=36, inside) is solid; its high-X face (X=43, outside) is NOT.
        let straddle_min = [36i64, 16, 16];
        let straddle = resolve_boundary_block(&leaves, straddle_min, density, density);
        assert!(
            straddle.seam_solidity.face_is_solid(0, 0),
            "the inside (low-X) face of a +X-straddling block must be solid"
        );
        assert!(
            !straddle.seam_solidity.face_is_solid(0, 1),
            "the outside (high-X) face of a +X-straddling block must NOT be solid"
        );
    }

    /// The capability is OFF by default: `build_chunk` / `resolve_region_two_layer` return
    /// `None` so the caller falls back to the dense path (the coexistence contract).
    #[test]
    fn capability_off_by_default_returns_none() {
        let scene = shape_scene(ShapeKind::Sphere, 16);
        let store = TwoLayerStore::default();
        assert!(!store.is_enabled());
        assert!(store.build_chunk([0, 0, 0], &scene, 16, 0).is_none());
        assert!(resolve_region_two_layer(&store, &scene, 16, 0).is_none());
        // E4 exact sinks also return None when the capability is OFF (dense fallback).
        assert!(streamed_widest_run_in_band(&store, &scene, 16, 0, 0).is_none());
        assert!(stream_vox_occupancy(&store, &scene, 16, |_| {}).is_none());
    }


    /// **The scope-outset parity gate**: the two-layer classifier must agree with the dense
    /// resolve for a Part carrying an outset (ADR 0019 Decision 7, ADR 0020 Decision 7).
    ///
    /// This is the composed-scope analogue of the per-leaf outset gate. The two paths reach
    /// the dilated Part by different routes — the dense one resolves the composed field's
    /// sign per voxel, the two-layer one classifies whole blocks from the composed field's
    /// Lipschitz bracket and only resolves the straddling ones. A bracket that claimed AIR
    /// or COARSE-SOLID anywhere the composed body disagrees would show up here as a
    /// mismatch, and nowhere else until it reached a user's screen as a hole.
    #[test]
    fn an_outset_part_classifies_as_the_dense_resolve_does() {
        use document::scene::{CombineOp, NodeBuilder, NodePath};
        let density = 8u32;
        let member = |size: [u32; 3], offset: [i64; 3], material, operation| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
            let mut node = Node::new("M", NodeContent::Tool { shape, material });
            node.transform = NodeTransform::from_blocks(offset, density);
            node.operation = operation;
            NodeBuilder::Leaf(node)
        };

        for outset_voxels in [3i64, -2] {
            // A Part with an INTERNAL cut, so the composed field exercises `max(a, −b)` and
            // not just the exact `min` of a union.
            let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
                "Part",
                vec![
                    member([4, 3, 3], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
                    member([3, 3, 3], [3, 1, 0], MaterialChoice::Wood, CombineOp::Union),
                    member([2, 2, 2], [1, 1, 1], MaterialChoice::Plain, CombineOp::Subtract),
                ],
            )]);
            scene
                .node_at_path_mut(&NodePath::from_indices(vec![0]))
                .expect("the Part resolves at path [0]")
                .outset = voxel_core::units::Measurement::from_voxels(outset_voxels);
            assert_two_layer_round_trip_matches_dense(
                &scene,
                density,
                &format!("outset-part-{outset_voxels}"),
            );
        }
    }

    /// **The emboss parity gate** (ADR 0020 Decision 7): the two-layer classifier must agree
    /// with the dense resolve for a scene containing an `Emboss` node.
    ///
    /// The ADR requires every new `CombineOp` arm to land in BOTH folds or they diverge
    /// silently. Emboss satisfies that by being absorbed BEFORE either fold: `A − N` is only
    /// meaningful on a field, and the voxel-set fold's accumulator is a set, so a scope
    /// containing an emboss is pre-composed into one `CompositeProducer` and both folds see a
    /// single ordinary leaf. This test is what proves the absorption is real — if an emboss
    /// ever leaked into either fold, the two would part company here.
    #[test]
    fn an_emboss_classifies_as_the_dense_resolve_does() {
        use document::scene::{CombineOp, NodeBuilder};
        let density = 8u32;
        for amount in [3i64, -3] {
            let slab = {
                let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 2, 5], 1, density);
                Node::new("Slab", NodeContent::Tool { shape, material: MaterialChoice::Stone })
            };
            let stamp = {
                let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 3, 2], 1, density);
                let mut node =
                    Node::new("Stamp", NodeContent::Tool { shape, material: MaterialChoice::Wood });
                node.transform = NodeTransform::from_blocks([2, 1, 2], density);
                node.operation = CombineOp::Emboss {
                    amount: voxel_core::units::Measurement::from_voxels(amount),
                };
                node
            };
            let scene = Scene::from_nodes(vec![NodeBuilder::group(
                "Part",
                vec![NodeBuilder::Leaf(slab), NodeBuilder::Leaf(stamp)],
            )]);
            assert_two_layer_round_trip_matches_dense(&scene, density, &format!("emboss-{amount}"));
        }
    }
