use super::*;
use super::core::assert_two_layer_round_trip_matches_dense;
use document::scene::{CombineOp, Node, NodeBuilder, NodeTransform};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use document::voxel::SdfShape;

    // ---- ADR 0017 (#75): the ordered fold's Intersect role through the CLASSIFIER ----
    //
    // The dense oracle's intersect semantics are pinned in the document crate
    // (`scene::tests::intersect`); these tests hold the two-layer interval
    // classification + boundary resolve against that oracle (the ADR 0010 parity
    // pattern), and pin the conservative bound algebra of ADR 0017 Decision 6: a mask
    // may keep a coarse-solid verdict only where solidity is PROVEN, degrades grazed
    // blocks to boundary, and kills blocks outside its body — including blocks its own
    // AABB never touches (the never-dropped-mask rule).

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
    /// curved-surface mask (boundary blocks everywhere the shell passes).
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

    /// THE GATE for the intersect slice (issue #75 acceptance): the two-layer
    /// classification + boundary resolve after an intersect is occupancy-IDENTICAL to
    /// the dense brute-force oracle, across the mask geometries that stress each
    /// classifier path (the plain overlap, a curved mask, a mask whose AABB misses
    /// most of the body's chunks, the fold-start no-op, and the scoped closes).
    #[test]
    fn round_trip_matches_dense_for_intersect_scenes() {
        // (1) The golden scene's shape: only the corner-octant overlap survives.
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [2, 2, 2], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "intersect-overlap");

        // (2) Curved mask through a box: the surviving boundary is non-axis-aligned,
        // so the rim blocks are genuine per-voxel boundary blocks.
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            sphere_tool([3, 3, 3], [1, 1, 1], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "intersect-sphere");

        // (3) The Subtract↔Intersect ASYMMETRY: a small mask deep inside a large body
        // kills every chunk the mask's AABB never touches. This is the case a
        // dropped-because-non-overlapping mask candidate would get WRONG (erring
        // toward solid), so it pins the never-dropped-mask broadphase rule.
        let scene = Scene::from_nodes(vec![
            box_tool([8, 8, 8], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "intersect-far-kill");

        // (4) A mask DISJOINT from the body annihilates everything (their
        // intersection is empty) — every chunk must come out empty.
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [8, 8, 8], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "intersect-disjoint");

        // (5) The ordering law / fold start: a mask BEFORE its target intersects the
        // empty accumulator (∅) and the body that follows stands alone.
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Intersect),
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "intersect-before-noop");

        // (6) A GROUP placed under Intersect: the group's composed occupancy masks
        // the parent accumulator (the scope-close-under-Intersect path, incl. the
        // ∅-in-chunk annihilation for chunks the group's body never reaches).
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [8, 8, 8],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Mask scope",
                vec![box_tool([2, 2, 2], [5, 5, 5], MaterialChoice::Wood, CombineOp::Union)
                    .into()],
            ),
        ]);
        let group_id = scene.roots[1];
        scene
            .node_by_id_mut(group_id)
            .expect("the group resolves")
            .operation = CombineOp::Intersect;
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "group-under-intersect");

        // (7) An Intersect leaf SEALED inside a group trims the group's body only;
        // an outside bystander must survive whole.
        let scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [2, 2, 2],
                [6, 6, 6],
                MaterialChoice::Wood,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Masked body",
                vec![
                    box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union).into(),
                    box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Plain, CombineOp::Intersect)
                        .into(),
                ],
            ),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "sealed-intersect");
    }

    /// Conservative bound algebra (ADR 0017 Decision 6): under a mask, a coarse-solid
    /// block may STAY coarse only where the fold PROVES both bodies deeply solid; a
    /// block the mask's surface grazes must degrade (boundary — never over-claim), and
    /// a block outside the mask's body is provably air — even when the mask's own AABB
    /// never touches it (the per-block never-dropped-mask rule).
    #[test]
    fn intersect_degrades_coarse_blocks_conservatively() {
        // Body: an 8³-block Stone box. Mask: a 4³-block box at blocks [1,5)³ —
        // wholly interior, so before the mask that whole region is coarse-solid.
        let body_only = Scene::from_nodes(vec![box_tool(
            [8, 8, 8],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        let masked = Scene::from_nodes(vec![
            box_tool([8, 8, 8], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [1, 1, 1], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        let block = DENSITY as i64;
        // The block at blocks [2,3) per axis — DEEP inside the mask (a whole block
        // from every mask face, farther than the block's circumradius, so the
        // conservative interval PROVES the mask covers it).
        let deep_inside_mask = VoxelAabb::new(
            [2 * block, 2 * block, 2 * block],
            [3 * block, 3 * block, 3 * block],
        );
        // The block at blocks [4,5) per axis — inside the mask but ADJACENT to its
        // +face, where the Lipschitz-centre bound cannot prove the whole block deeply
        // covered: the conservative verdict degrades to BOUNDARY (never coarse-solid),
        // and the per-voxel resolve finds it full — exact, just unelided.
        let grazed_block = VoxelAabb::new(
            [4 * block, 4 * block, 4 * block],
            [5 * block, 5 * block, 5 * block],
        );

        let body_leaves = body_only.leaf_producers(DENSITY);
        let body_leaves: Vec<&LeafProducer> = body_leaves.iter().collect();
        for interior in [deep_inside_mask, grazed_block] {
            assert_eq!(
                classify_chunk_block(&body_leaves, interior, DENSITY),
                BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
                "without the mask the interior block is coarse-solid"
            );
        }

        let masked_leaves = masked.leaf_producers(DENSITY);
        let masked_leaves: Vec<&LeafProducer> = masked_leaves.iter().collect();
        assert_eq!(
            classify_chunk_block(&masked_leaves, deep_inside_mask, DENSITY),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "a block PROVABLY deep inside both bodies may stay coarse — at the \
             BODY's material (the mask never stamps)"
        );
        assert_eq!(
            classify_chunk_block(&masked_leaves, grazed_block, DENSITY),
            BlockClassification::Boundary,
            "a coarse-solid block the mask's surface grazes must DEGRADE to \
             boundary — never keep claiming coarse-solid"
        );
        // A body block OUTSIDE the mask's box (blocks [6,7)³ — the mask's AABB does
        // not even touch it) is provably air: the mask kills everything outside its
        // body, and its interval must reach this block despite the AABB miss.
        let outside_mask = VoxelAabb::new(
            [6 * block, 6 * block, 6 * block],
            [7 * block, 7 * block, 7 * block],
        );
        assert_eq!(
            classify_chunk_block(&masked_leaves, outside_mask, DENSITY),
            BlockClassification::Air,
            "a coarse-solid block outside the mask's body must RE-classify to air \
             (the mask is never dropped from a block's fold)"
        );
    }

    /// The fold-start edge case through the classifier: a block only a MASK overlaps
    /// (no additive leaf anywhere near) is provably air — intersecting the empty
    /// accumulator yields empty.
    #[test]
    fn intersect_with_empty_accumulator_classifies_air() {
        let mask_only = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Wood,
            CombineOp::Intersect,
        )]);
        let leaves = mask_only.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = DENSITY as i64;
        assert_eq!(
            classify_chunk_block(
                &leaves,
                VoxelAabb::new([0, 0, 0], [block, block, block]),
                DENSITY
            ),
            BlockClassification::Air,
            "a block only a mask overlaps is provably air (nothing accumulated to keep)"
        );
    }

    /// The surviving cells keep their ACCUMULATED material end-to-end through the
    /// two-layer boundary resolve: a two-material body masked by a third material
    /// keeps both survivor materials and shows the mask's nowhere — the parity gate
    /// already proves position+id identity to the dense oracle; this pins the
    /// human-readable claim.
    #[test]
    fn intersect_never_stamps_the_mask_material() {
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 0, 0], MaterialChoice::Wood, CombineOp::Union),
            // Keeps the lower-Z half of the two-material body.
            box_tool([3, 2, 1], [0, 0, 0], MaterialChoice::Plain, CombineOp::Intersect),
        ]);
        let store = TwoLayerStore::enabled();
        let assembled = resolve_region_two_layer(&store, &scene, DENSITY, 0)
            .expect("the capability is enabled");
        assert!(
            !assembled.occupied.is_empty(),
            "the masked body must still have surviving voxels"
        );
        let mut survivor_materials = std::collections::BTreeSet::new();
        for voxel in &assembled.occupied {
            assert_ne!(
                voxel.block_id,
                MaterialChoice::Plain.block_id(),
                "no surviving voxel may carry the MASK's material — an Intersect never stamps"
            );
            survivor_materials.insert(voxel.block_id);
        }
        assert!(
            survivor_materials.contains(&MaterialChoice::Stone.block_id())
                && survivor_materials.contains(&MaterialChoice::Wood.block_id()),
            "both ACCUMULATED materials must survive inside the mask \
             (got {survivor_materials:?})"
        );
    }
