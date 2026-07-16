use super::*;
use super::core::assert_two_layer_round_trip_matches_dense;
use document::scene::{CombineOp, Node, NodeBuilder, NodeTransform};
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use document::voxel::SdfShape;

    // ---- ADR 0017 Decision 3 (#74): sealed scopes through the CLASSIFIER ----
    //
    // The dense oracle's sealed-scope semantics are pinned in the document crate
    // (`scene::tests::sealed_scopes`); these tests hold the two-layer SCOPED interval
    // classification + scoped boundary resolve against that oracle (the ADR 0010
    // parity pattern), and pin the sealing at the interval level: a cutter inside a
    // scope must never degrade — let alone carve — a block outside its scope.

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

    /// THE GATE for the sealed-scope slice (issue #74 acceptance): the two-layer
    /// classification + boundary resolve for scoped scenes is occupancy-IDENTICAL to
    /// the dense brute-force oracle, across the scope shapes the resolver must honour:
    /// a cutter sealed inside a group (with an overlapping outside bystander), a root
    /// cutter carving a group's composed body, nested scopes, a definition's internal
    /// cutter under multiple instances, and a whole group placed under Subtract.
    #[test]
    fn round_trip_matches_dense_for_scoped_scenes() {
        // (1) The golden scene's shape: a bystander placed BEFORE a group whose
        // internal cutter overlaps it — only the scope seal protects the bystander.
        let scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [2, 2, 2],
                [3, 3, 3],
                MaterialChoice::Wood,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Carved body",
                vec![
                    box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union).into(),
                    box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Plain, CombineOp::Subtract)
                        .into(),
                ],
            ),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "scoped-corner");

        // (2) A root cutter AFTER a group carves the group's composed body.
        let scene = Scene::from_nodes(vec![
            NodeBuilder::group(
                "Body",
                vec![box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union)
                    .into()],
            ),
            NodeBuilder::Leaf(box_tool(
                [2, 2, 2],
                [2, 2, 2],
                MaterialChoice::Wood,
                CombineOp::Subtract,
            )),
        ]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "cutter-after-group");

        // (3) Nested scopes: the inner group's cutter overlaps the outer group's body
        // but may carve only its own (inner) sibling.
        let scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Outer",
            vec![
                box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union).into(),
                NodeBuilder::group(
                    "Inner",
                    vec![
                        box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Union)
                            .into(),
                        box_tool([2, 2, 2], [3, 3, 3], MaterialChoice::Plain, CombineOp::Subtract)
                            .into(),
                    ],
                ),
            ],
        )]);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "nested-scopes");

        // (4) A definition's internal cutter is fully spent inside the definition;
        // two instances place the finished body, a bystander overlaps the first
        // instance's cutter volume.
        let def_id = DefId(1);
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [2, 2, 2],
                [3, 3, 3],
                MaterialChoice::Wood,
                CombineOp::Union,
            )),
            NodeBuilder::Leaf({
                let mut instance = Node::new("Notched 1", NodeContent::Instance(def_id));
                instance.transform = NodeTransform::from_blocks([0, 0, 0], DENSITY);
                instance
            }),
            NodeBuilder::Leaf({
                let mut instance = Node::new("Notched 2", NodeContent::Instance(def_id));
                instance.transform = NodeTransform::from_blocks([8, 0, 0], DENSITY);
                instance
            }),
        ]);
        scene.add_definition(
            def_id,
            "Notched",
            vec![
                box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
                box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Plain, CombineOp::Subtract),
            ],
        );
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "sealed-definition");

        // (5) A whole GROUP placed under Subtract: the group's composed occupancy
        // carves the parent accumulator (occupancy-only — the survivors keep their
        // material; the resolver's Group-under-Subtract close path). NodeBuilder has
        // no operation knob, so the GROUP node's own operation is flipped by id after
        // construction (roots[1] is the group).
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [4, 4, 4],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Group cutter",
                vec![box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Union)
                    .into()],
            ),
        ]);
        let group_id = scene.roots[1];
        scene
            .node_by_id_mut(group_id)
            .expect("the group resolves")
            .operation = CombineOp::Subtract;
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "group-under-subtract");
    }

    /// Sealing at the INTERVAL level: a block deep inside a scoped cutter's box but
    /// outside the cutter's scope must keep its coarse-solid verdict — the sealed
    /// cutter can neither carve it nor degrade its elision (its interval folds inside
    /// the inner scope only).
    #[test]
    fn sealed_cutter_does_not_degrade_blocks_outside_its_scope() {
        // Outer body: an 8³-block Stone box. Inner group: a small Wood body at blocks
        // [1,2)³ plus a 4³ cutter at blocks [1,5)³ — the cutter covers deep-interior
        // Stone blocks, but is sealed inside the inner group.
        let scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [8, 8, 8],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Inner",
                vec![
                    box_tool([1, 1, 1], [1, 1, 1], MaterialChoice::Wood, CombineOp::Union).into(),
                    box_tool([4, 4, 4], [1, 1, 1], MaterialChoice::Plain, CombineOp::Subtract)
                        .into(),
                ],
            ),
        ]);
        let leaves = scene.leaf_producers(DENSITY);
        let leaves: Vec<&LeafProducer> = leaves.iter().collect();
        let block = DENSITY as i64;
        // The block at blocks [3,4)³ — deep inside the cutter's box, outside the
        // inner Wood body: only the outer Stone leaf ADDS here, and the cutter is
        // sealed away from it, so the verdict stays coarse-solid Stone. (In the #73
        // flat fold this exact block re-classified to Air — the sealing is the delta.)
        let deep_block = VoxelAabb::new(
            [3 * block, 3 * block, 3 * block],
            [4 * block, 4 * block, 4 * block],
        );
        assert_eq!(
            classify_chunk_block(&leaves, deep_block, DENSITY),
            BlockClassification::CoarseSolid(MaterialChoice::Stone.block_id()),
            "a sealed cutter must not carve or degrade a block outside its scope"
        );
    }
