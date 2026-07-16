use super::*;
use super::core::assert_two_layer_round_trip_matches_dense;
use document::scene::{CombineOp, Node, NodeTransform};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use document::voxel::SdfShape;

    // ---- ADR 0017 (#73): the ordered fold's Subtract role through the CLASSIFIER ----
    //
    // The dense oracle's subtract semantics are pinned in the document crate
    // (`scene::tests::subtract`); these tests hold the two-layer interval
    // classification + boundary resolve against that oracle (the ADR 0010 parity
    // pattern), and pin the conservative re-classification a cutter forces: a
    // coarse-solid block under a Subtract must degrade to boundary or air, never
    // over-claim.

    const DENSITY: u32 = 8;

    /// A whole-block Box Tool at `offset_blocks` carrying `operation`.
    fn box_tool(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        material: MaterialChoice,
        operation: CombineOp,
    ) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, DENSITY);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// A whole-block Sphere Tool at `offset_blocks` carrying `operation` — the
    /// curved-surface cutter/body (boundary blocks everywhere the shell passes).
    fn sphere_tool(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        material: MaterialChoice,
        operation: CombineOp,
    ) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, size_blocks, 1, DENSITY);
        let mut node = Node::new("Sphere", NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// THE GATE for the subtract slice (issue #73 acceptance): the two-layer
    /// classification + boundary resolve after a subtract is occupancy-IDENTICAL to
    /// the dense brute-force oracle, across the carve geometries that stress each
    /// classifier path (coarse→boundary degradation, whole-block carve to air, a
    /// curved cutter, a cutter that carves everything, and the ordering no-op).
    #[test]
    fn round_trip_matches_dense_for_subtract_scenes() {
        // (1) Corner carve — the golden scene's shape: coarse corner blocks of the
        // body become boundary (notch faces) or air (fully carved).
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "subtract-corner");

        // (2) Interior cavity — the cutter is wholly INSIDE the body, so previously
        // coarse-solid interior blocks must re-classify (a carve the mesh alone
        // could never show without re-classification).
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "subtract-cavity");

        // (3) Curved cutter through a box: the carve boundary is non-axis-aligned,
        // so the notch blocks are genuine per-voxel boundary blocks.
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            sphere_tool([3, 3, 3], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "subtract-sphere");

        // (4) Total carve — the cutter covers the whole body, so EVERY chunk must
        // come out empty (the AllAir fast path under a chunk-containing cutter).
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "subtract-total");

        // (5) The ordering law: a cutter BEFORE its target subtracts from nothing —
        // the classification must equal the body's alone (occupancy-identical to the
        // dense oracle, which pins the no-op on its side).
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "subtract-before-noop");
    }

    /// Conservative RE-classification (ADR 0017 Decision 6): under a cutter, a block
    /// that classified coarse-solid must degrade toward boundary/air — and a block
    /// only a cutter overlaps is provably air (nothing accumulated to carve).
    #[test]
    fn subtract_degrades_coarse_blocks_to_boundary_or_air() {
        // Body: an 8³-block Stone box (blocks [0,8)³ in absolute space). Cutter: a
        // 4³-block box at blocks [1,5)³ — wholly interior, so before the cutter that
        // whole region is deep-interior coarse-solid.
        let body_only = Scene::from_nodes(vec![box_tool(
            [8, 8, 8],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        let carved = Scene::from_nodes(vec![
            box_tool([8, 8, 8], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let block = DENSITY as i64;
        // The block at blocks [2,3) per axis — DEEP inside the cutter (a whole block
        // from every cutter face, farther than the block's circumradius, so the
        // conservative interval PROVES the carve).
        let deep_carved_block = VoxelAabb::new(
            [2 * block, 2 * block, 2 * block],
            [3 * block, 3 * block, 3 * block],
        );
        // The block at blocks [4,5) per axis — inside the cutter but ADJACENT to its
        // +face, where the Lipschitz-centre bound cannot prove the whole block is
        // deeply inside the cutter: the conservative verdict degrades to BOUNDARY
        // (never coarse-solid), and the per-voxel resolve finds it empty — exact,
        // just unelided.
        let grazed_block = VoxelAabb::new(
            [4 * block, 4 * block, 4 * block],
            [5 * block, 5 * block, 5 * block],
        );

        let body_leaves = body_only.leaf_producers(DENSITY);
        let body_leaves: Vec<&LeafProducer> = body_leaves.iter().collect();
        for interior in [deep_carved_block, grazed_block] {
            assert_eq!(
                classify_chunk_block(&body_leaves, interior, DENSITY),
                BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
                "without the cutter the interior block is coarse-solid"
            );
        }

        let carved_leaves = carved.leaf_producers(DENSITY);
        let carved_leaves: Vec<&LeafProducer> = carved_leaves.iter().collect();
        assert_eq!(
            classify_chunk_block(&carved_leaves, deep_carved_block, DENSITY),
            BlockClassification::Air,
            "a coarse-solid block the cutter provably covers must RE-classify to air"
        );
        assert_ne!(
            classify_chunk_block(&carved_leaves, grazed_block, DENSITY),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "a coarse-solid block the cutter reaches must DEGRADE (boundary or air) — \
             never keep claiming coarse-solid"
        );
        // A block far from the cutter stays coarse-solid at the body's material —
        // the cutter's interval must not degrade blocks it provably cannot reach.
        let far_block = VoxelAabb::new(
            [6 * block, 6 * block, 6 * block],
            [7 * block, 7 * block, 7 * block],
        );
        assert_eq!(
            classify_chunk_block(&carved_leaves, far_block, DENSITY),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "an interior block outside the cutter stays coarse-solid"
        );
        // A block only the CUTTER overlaps (outside the body) is provably air: a
        // Subtract can only remove occupancy, never add it.
        let cutter_only = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [8, 8, 8], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let cutter_only_leaves = cutter_only.leaf_producers(DENSITY);
        let cutter_only_leaves: Vec<&LeafProducer> = cutter_only_leaves.iter().collect();
        let cutter_block = VoxelAabb::new(
            [8 * block, 8 * block, 8 * block],
            [9 * block, 9 * block, 9 * block],
        );
        assert_eq!(
            classify_chunk_block(&cutter_only_leaves, cutter_block, DENSITY),
            BlockClassification::Air,
            "a block only a cutter overlaps is provably air (nothing to carve)"
        );
    }

    /// The surviving cells of a boundary block the cutter grazes keep the BODY's
    /// material end-to-end through the two-layer boundary resolve — the parity gate
    /// already proves position+id identity to the dense oracle, so this simply pins
    /// the human-readable claim: no cell anywhere carries the cutter's material.
    #[test]
    fn subtract_never_stamps_the_cutter_material() {
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let store = TwoLayerStore::enabled();
        let assembled = resolve_region_two_layer(&store, &scene, DENSITY, 0)
            .expect("the capability is enabled");
        assert!(
            !assembled.occupied.is_empty(),
            "the carved body must still have surviving voxels"
        );
        for voxel in &assembled.occupied {
            assert_eq!(
                voxel.block_id,
                MaterialChoice::Stone.block_id(),
                "a surviving voxel must keep the body's material — the cutter never stamps"
            );
        }
    }
