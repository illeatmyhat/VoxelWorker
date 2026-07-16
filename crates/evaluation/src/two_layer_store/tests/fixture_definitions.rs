use super::*;
use super::core::assert_two_layer_round_trip_matches_dense;
use document::scene::{CombineOp, NodeBuilder};

    // ---- ADR 0017 Decision 4 (#77): fixture definitions through the CLASSIFIER ----
    //
    // The dense oracle's fixture-splice semantics are pinned in the document crate
    // (`scene::tests::fixture_definitions`); these tests hold the two-layer interval
    // classification + boundary resolve against that oracle (the ADR 0010 parity
    // pattern). No classifier machinery is fixture-specific: a fixture's leaves
    // arrive carrying the HOSTING scope's path plus their own operations (the walk
    // pushes no frame for a fixture expansion), so its Subtract child is exactly a
    // scoped-or-root cutter to every conservative fast path — these tests prove
    // that identification holds occupancy-identically end to end.

    const DENSITY: u32 = 8;

    /// The window definition's id in every fixture scene below.
    const WINDOW_DEF: DefId = DefId(1);

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

    /// An `Instance(def_id)` node at `offset_blocks` (its operation is inert on a
    /// fixture instance — left at the Union default).
    fn window_instance(offset_blocks: [i64; 3]) -> Node {
        let mut node = Node::new("Window", NodeContent::Instance(WINDOW_DEF));
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node
    }

    /// Register the WINDOW fixture on `scene`: [opening cutter `Subtract` (3×1×3),
    /// Wood frame `Union` (3×1×1)], flagged `fixture` so it splices.
    fn add_window_fixture(scene: &mut Scene) {
        scene.add_definition(
            WINDOW_DEF,
            "Window",
            vec![
                box_tool([3, 1, 3], [0, 0, 0], MaterialChoice::Plain, CombineOp::Subtract),
                box_tool([3, 1, 1], [0, 0, 0], MaterialChoice::Wood, CombineOp::Union),
            ],
        );
        assert!(scene.set_definition_fixture(WINDOW_DEF, true));
    }

    /// A Stone wall standing in the XZ plane (Z-up): 8 blocks wide, 1 thick, 6 tall.
    fn wall(offset_blocks: [i64; 3]) -> Node {
        box_tool([8, 1, 6], offset_blocks, MaterialChoice::Stone, CombineOp::Union)
    }

    /// THE GATE for the fixture slice (issue #77 acceptance): the two-layer
    /// classification + boundary resolve for fixture scenes is occupancy-IDENTICAL
    /// to the dense brute-force oracle, across the splice shapes the resolver must
    /// honour: the window golden (a root-scope splice — the spliced cutter is a
    /// plain root cutter to the fast paths), a fixture sealed inside a Group with an
    /// overlapping outside bystander (the spliced children carry the GROUP frame),
    /// a fixture placed before its would-be host (subtract-from-nothing + a buried
    /// Union), and two placements over separated walls (per-instance transforms).
    #[test]
    fn round_trip_matches_dense_for_fixture_scenes() {
        // (1) The window golden's shape: one wall, one fixture placement after it.
        let mut scene = Scene::from_nodes(vec![wall([0, 0, 0]), window_instance([2, 0, 2])]);
        add_window_fixture(&mut scene);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "window-fixture");

        // (2) One-level piercing: the fixture splices into its host GROUP's fold;
        // the bystander before the group (filling the opening's volume) survives.
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [3, 1, 3],
                [2, 0, 2],
                MaterialChoice::Wood,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Walled",
                vec![wall([0, 0, 0]).into(), window_instance([2, 0, 2]).into()],
            ),
        ]);
        add_window_fixture(&mut scene);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "fixture-in-group");

        // (3) The ordering law: a fixture before the wall cuts nothing; its frame
        // is buried inside the later wall (later-wins).
        let mut scene = Scene::from_nodes(vec![window_instance([2, 0, 2]), wall([0, 0, 0])]);
        add_window_fixture(&mut scene);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "fixture-before-wall");

        // (4) Two placements over separated walls: each splice enters the root fold
        // under its own instance transform (ADR 0008 carried frames).
        let mut scene = Scene::from_nodes(vec![
            wall([0, 0, 0]),
            wall([16, 0, 0]),
            window_instance([2, 0, 2]),
            window_instance([18, 0, 2]),
        ]);
        add_window_fixture(&mut scene);
        assert_two_layer_round_trip_matches_dense(&scene, DENSITY, "two-window-placements");
    }
