//! Two-layer chunk classifier parity, residency, incremental-edit, and stream tests.

use std::collections::BTreeMap;
use voxel_core::core_geom::CHUNK_BLOCKS;
use document::scene::{LeafProducer, Scene};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::VoxelGrid;

    use super::*;
    // The submodules the mod-level `pub(crate) use` glob does not re-export (their items are
    // reached only by the tests, not by non-test sibling code): the resident cache internals
    // and the stream / oracle functions.
    #[allow(unused_imports)]
    use super::resident_cache::*;
    #[allow(unused_imports)]
    use super::stream::*;
    use voxel_core::core_geom::MaterialChoice;
    use document::scene::{DefId, Node, NodeContent, NodeTransform};
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{GeometryParams, SdfShape};

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

    fn shape_scene(kind: ShapeKind, voxels_per_block: u32) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                    5 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// THE GATE (parity (a)): the two-layer round-trip occupancy (coarse fast-fill +
    /// boundary per-voxel) is BIT-IDENTICAL (position + block id) to the dense
    /// `Scene::resolve_region`, for the gated scene. Mirrors
    /// `store.rs::cache_region_matches_monolithic_*`. Returns the chunk + cell counts the
    /// build classified (so the harness can report coverage).
    fn assert_two_layer_round_trip_matches_dense(
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

    fn make_tool(kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32) -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
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
        use document::scene::Part;
        let density = 16;
        let mut cloud = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed: 7 }));
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
        // The removed top-right quadrant ([24,32)² in-plane) overlaps the producer AABB so
        // it classifies BOUNDARY (resolves per-voxel to EMPTY) — crucially NOT coarse-solid,
        // which is exactly what a naive bbox-solid claim would have wrongly returned.
        assert_eq!(
            classify_chunk_block(&leaves, VoxelAabb::new([24, 24, 8], [32, 32, 16]), density),
            BlockClassification::Boundary,
            "the removed L quadrant must NOT be coarse-solid (the polygon excludes it)"
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
        use document::scene::Part;
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
            NodeContent::Part(Part::DebugClouds { seed: 7 }),
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

    fn make_tool_density(
        kind: ShapeKind,
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
        size_blocks: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, [size_blocks; 3], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
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

    // ===== ADR 0010 #54: chunk-granular INCREMENTAL edits on the two-layer path ======
    //
    // Mirrors `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`:
    // for every edit kind, the two-layer resident cache after an INCREMENTAL edit
    // (invalidate the dirty AABB's chunks, re-derive only those) is IDENTICAL — the
    // coarse layer + overlay + microblock maps + seam flags, via the derived
    // `TwoLayerChunk: PartialEq` — to a full from-scratch two-layer rebuild of scene B.

    /// A tool node for the incremental edit scenes (mirrors `store.rs::tool_node`).
    fn incr_tool_node(
        kind: ShapeKind,
        size: [u32; 3],
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// The full resident map a WHOLESALE two-layer rebuild produces for `scene`: every
    /// covering chunk built from scratch, keyed by absolute coord. This is the parity
    /// gate's ground truth — the "full rebuild" every incremental edit must equal.
    fn full_two_layer_resident(
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        let mut cache = TwoLayerResidentCache::enabled();
        let chunks = cache.resident_two_layer_chunks(scene, density, 0);
        chunks
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Snapshot a resident cache's covering chunks (post-edit) as an owned coord→chunk
    /// map, for the `== full` comparison.
    fn resident_snapshot(
        cache: &mut TwoLayerResidentCache,
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        cache
            .resident_two_layer_chunks(scene, density, 0)
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `cache` in place, driving the
    /// dirty set exactly as `app_core::rebuild`: build the leaf spatial index for both
    /// scenes, diff for the edit AABB, and `invalidate_aabb` the dirty chunks (or
    /// `clear()` for the non-localisable fallback). Returns `(evicted_count, took_aabb_path)`
    /// so the harness can assert the localisable edits touch a strict subset.
    fn apply_two_layer_incremental_edit(
        cache: &mut TwoLayerResidentCache,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> (usize, bool) {
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        match index_b.edit_aabb_since(&index_a) {
            Some(edit_aabb) => {
                let evicted = cache.invalidate_aabb(&edit_aabb, density);
                (evicted.len(), true)
            }
            None => {
                // The wholesale fallback: a density change or a region-spanning Part edit
                // has no localisable box (mirrors `app_core::rebuild`'s `clear()` arm).
                cache.clear();
                (0, false)
            }
        }
    }

    /// **THE #54 GATE — incremental == full for every LOCALISABLE edit kind.** For each of
    /// add / remove / move / resize / recolor, the two-layer resident cache after the
    /// incremental edit is IDENTICAL (coarse layer + overlay + microblock maps + seam
    /// flags) to a full from-scratch two-layer rebuild of scene B, AND the edit touched a
    /// strict SUBSET of the scene's chunks (proving it is genuinely incremental, not a
    /// disguised full rebuild). Mirrors
    /// `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`.
    #[test]
    fn incremental_two_layer_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // Three tools spread far apart in X so each occupies chunks the others don't
        // touch (clean localised edits). The interior "subject" box sits between two
        // static anchors that pin the composite extent (as in the dense net) — though
        // note a recentre shift does NOT invalidate the two-layer cache (chunk-local
        // frame), the anchors keep the setup parallel to the dense parity net.
        let anchor_lo =
            || incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi =
            || incr_tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            incr_tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood, density),
            anchor_hi(),
        ]);

        let recolor = {
            let mut b = scene_a.clone();
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            let replacement =
                incr_tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood, density);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            b.root_node_mut(1).transform = NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            b.add_node(incr_tool_node(
                ShapeKind::Box,
                [3, 3, 3],
                [90, 0, 0],
                MaterialChoice::Stone,
                density,
            ));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node] {
            // Incremental: wholesale-build A, then apply the single edit and re-fill.
            let mut cache = TwoLayerResidentCache::enabled();
            let total_before = {
                let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
                cache.resident_len()
            };
            let (evicted, took_aabb_path) =
                apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
            assert!(
                took_aabb_path,
                "[{label}] this edit kind must be localisable (the AABB path, not clear())"
            );
            let incremental = resident_snapshot(&mut cache, &scene_b, density);

            // The full from-scratch rebuild for scene B (the truth).
            let full = full_two_layer_resident(&scene_b, density);

            assert_eq!(
                incremental, full,
                "[{label}] incremental two-layer cache (coarse layer + overlay + microblock \
                 maps + seam flags per covering chunk) MUST equal a full from-scratch rebuild \
                 of scene B — a stale chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: the edit evicted strictly fewer chunks than the scene's
            // total resident count (so it is genuinely incremental, not a full rebuild).
            let scene_chunks = total_before.max(full.len());
            assert!(
                evicted < scene_chunks,
                "[{label}] a localised edit must evict strictly FEWER chunks ({evicted}) than \
                 the scene's total ({scene_chunks}) — else it is a disguised full rebuild"
            );
        }
    }

    /// Perf probe (block-row-dedup regression guard): the full-band diameter re-measure —
    /// the query that fires when the layer band or grid changes. Before the ADR 0010 E5
    /// block-row dedup this was O(volume) (a coarse block stamped all `d²` of its voxel rows):
    /// 130ms @800³ → 127s @8000³, freezing the main thread. After, it is O(total blocks) and
    /// runs on the background diameter worker (never the UI thread). Reports wall-clock across
    /// four solid-cube edge lengths. Run:
    /// `cargo test --release widest_run_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn widest_run_scaling_probe() {
        use document::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [50i64, 125, 250, 500] {
            let edge = blocks * density as i64;
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let widest = streamed_widest_run_in_band(
                &TwoLayerStore::enabled(),
                &scene,
                density,
                0,
                edge as u32,
            );
            let elapsed = start.elapsed();
            println!("widest-run {edge}^3 vx full band: {widest:?} in {elapsed:?}");
        }
    }

    /// Perf probe (interior-elision win): time the LIVE two-layer build for a large
    /// SOLID sketch-extrude box — the path the app actually runs (NOT shot's dense
    /// `resolve_region` golden oracle). Before elision every interior block resolved
    /// per-voxel (O(volume)); after, interiors classify coarse (O(surface)). Reports the
    /// coarse/sculpted split + wall-clock. Run:
    /// `cargo test --release two_layer_sketch_box_build_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn two_layer_sketch_box_build_probe() {
        use document::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [25i64, 50] {
            let edge = blocks * density as i64; // 400, then 800 voxels/axis (block-aligned)
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let elapsed = start.elapsed();
            let coarse: u64 = chunks
                .iter()
                .map(|(_, chunk)| chunk.coarse.iter().filter(|id| id.is_some()).count() as u64)
                .sum();
            let sculpted: u64 = chunks.iter().map(|(_, chunk)| chunk.microblocks.len() as u64).sum();
            println!(
                "sketch box {edge}³ voxels ({blocks} blocks/axis): two-layer build {:?} — \
                 {coarse} coarse + {sculpted} sculpted blocks over {} chunks",
                elapsed,
                chunks.len()
            );
        }
    }

    /// A localised recolor of one small far-flung node dirties only the handful of chunks
    /// that node occupies, NOT the whole scene — the two-layer analogue of
    /// `store.rs::localized_recolor_rebuilds_few_chunks`.
    #[test]
    fn incremental_two_layer_localized_recolor_evicts_few_chunks() {
        let density = 16u32;
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "an in-place recolor must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            evicted * 2 < total,
            "a localised recolor of a small node must evict far fewer than half the chunks: \
             evicted {evicted} of {total}"
        );
        assert_eq!(incremental, full_two_layer_resident(&scene_b, density));
    }

    /// **Localisable move re-derives BOTH endpoints.** A moved node's dirty AABB spans its
    /// source AND destination (the `edit_aabb_since` union), so the two-layer cache vacates
    /// the source chunks and rebuilds the destination — and the result equals a full
    /// rebuild (no stale geometry left at the old location).
    #[test]
    fn incremental_two_layer_move_clears_source_and_fills_destination() {
        let density = 16u32;
        // A wide anchor keeps many chunks resident that the moved box never touches, so a
        // move touching a strict subset is meaningful.
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [2, 2, 2], [70, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = NodeTransform::from_blocks([85, 0, 0], density);

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "a move must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "a move must leave no stale geometry at the source and match a full rebuild"
        );
        assert!(evicted < total, "a move touches a strict subset ({evicted} of {total})");
    }

    /// **WHOLESALE FALLBACK — a density change re-derives everything.** A density change
    /// resizes every chunk's voxel extent, so `edit_aabb_since` returns `None` and the
    /// cache clears (belt-and-braces: `invalidate_aabb` also clears on a density mismatch).
    /// After the fallback the cache still equals a full rebuild at the NEW density.
    #[test]
    fn incremental_two_layer_density_change_falls_back_to_wholesale() {
        let density_a = 16u32;
        let density_b = 8u32;
        let scene = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density_a),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene, density_a, 0);
        // The density-change diff: the same scene rebuilt at a different density has no
        // localisable AABB (the indices differ in density), so `edit_aabb_since` is None.
        let index_a = scene.build_leaf_spatial_index(density_a);
        let index_b = scene.build_leaf_spatial_index(density_b);
        assert!(
            index_b.edit_aabb_since(&index_a).is_none(),
            "a density change must have no localisable edit AABB (the wholesale fallback)"
        );
        cache.clear();
        let incremental = resident_snapshot(&mut cache, &scene, density_b);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene, density_b),
            "after the density-change wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// **WHOLESALE FALLBACK — editing an unbounded (region-spanning) producer.** Editing a
    /// `DebugClouds` Part (its dirty region is "everywhere", `edit_aabb_since` returns
    /// `None`) forces a wholesale clear; the rebuilt cache still equals a full rebuild.
    /// This is the "unboundable-producer edit falls back to wholesale" acceptance case.
    #[test]
    fn incremental_two_layer_cloud_edit_falls_back_to_wholesale() {
        use document::scene::Part;
        let density = 16u32;
        let cloud = |seed: u32| {
            let mut node = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed }));
            node.transform = NodeTransform::from_blocks([0, 0, 0], density);
            node
        };
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Box, [3, 3, 3], [0, 0, 0], MaterialChoice::Stone, density),
            cloud(7),
        ]);
        // Edit the cloud's seed (a region-spanning content change; root index 1).
        let mut scene_b = scene_a.clone();
        if let NodeContent::Part(Part::DebugClouds { seed }) =
            &mut scene_b.root_node_mut(1).content
        {
            *seed = 42;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
        let (_evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(
            !took_aabb_path,
            "editing a region-spanning Part must take the wholesale fallback, not the AABB path"
        );
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "after the cloud-edit wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// Wholesale-build timing probe across a WIDE object-count range (#66; the #63 lesson —
    /// a small N hides a super-linear asymptote). Not a correctness gate: run manually with
    /// `cargo test --release --lib wholesale_build_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe, run manually with --release --ignored --nocapture"]
    fn wholesale_build_scaling_probe() {
        let density = 16u32;
        for boxes_per_axis in [5i64, 12, 22] {
            let mut nodes = Vec::new();
            for grid_z in 0..boxes_per_axis {
                for grid_y in 0..boxes_per_axis {
                    for grid_x in 0..boxes_per_axis {
                        nodes.push(make_tool_density(
                            ShapeKind::Box,
                            [grid_x * 4, grid_y * 4, grid_z * 4],
                            MaterialChoice::Stone,
                            density,
                            2,
                        ));
                    }
                }
            }
            let object_count = boxes_per_axis.pow(3);
            let scene = Scene::from_nodes(nodes);
            let leaves_started = std::time::Instant::now();
            let leaves = scene.leaf_producers(density);
            let leaves_elapsed = leaves_started.elapsed();
            let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
            let chunk_count = (0..3)
                .map(|axis| (max_chunk[axis] - min_chunk[axis] + 1) as i64)
                .product::<i64>();
            let broadphase_started = std::time::Instant::now();
            let broadphase = leaf_edit_broadphase(&leaves, density);
            let broadphase_elapsed = broadphase_started.elapsed();
            std::hint::black_box(&broadphase);
            let build_started = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let build_elapsed = build_started.elapsed();
            eprintln!(
                "N={object_count} objects, {chunk_count} covering chunks: leaf hoist \
                 {leaves_elapsed:?}, edit-broadphase BVH rebuild {broadphase_elapsed:?}, \
                 wholesale build {build_elapsed:?} ({} chunks emitted)",
                chunks.len()
            );
        }
    }

    /// The capability OFF (the default): the resident cache is a no-op — it never fills and
    /// `resident_two_layer_chunks` returns empty, so a caller falls back to the dense path.
    #[test]
    fn incremental_two_layer_capability_off_is_noop() {
        let density = 16u32;
        let scene = shape_scene(ShapeKind::Sphere, density);
        let mut cache = TwoLayerResidentCache::default();
        assert!(!cache.is_enabled());
        assert!(cache.resident_two_layer_chunks(&scene, density, 0).is_empty());
        assert_eq!(cache.resident_len(), 0);
    }
