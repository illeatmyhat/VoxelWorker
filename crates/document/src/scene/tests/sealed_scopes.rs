use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- ADR 0017 Decision 3 (#74): sealed composition scopes — the dense oracle ----
    //
    // Groups and definition bodies PRE-COMPOSE: a scope resolves its children into one
    // body via the ordered fold, then that body folds into the parent under the SCOPE
    // node's own `CombineOp`. A boolean inside a scope can never affect geometry
    // outside it. These tests pin the dense-oracle semantics; the two-layer classifier
    // is held against this oracle in the evaluation crate's parity tests
    // (`two_layer_store::tests::subtract`).

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

    /// Resolve `scene` through the dense oracle and return its occupancy multiset in
    /// ABSOLUTE voxel space (recentre-normalised), keyed `(index, material)`.
    fn resolved_absolute_multiset(
        scene: &Scene,
    ) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
        let grid = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        occupied_multiset(&grid, scene.recentre_voxels(DENSITY))
    }

    /// The golden scene's shape (acceptance #1): a cutter INSIDE a Group carves only
    /// within the Group; a sibling OUTSIDE the group — placed BEFORE it in document
    /// order, so only the scope seal (never the ordering law) protects it — is
    /// untouched even though the cutter's box overlaps it. The result is exactly the
    /// disjoint union of "the group's notched body" and "the bystander".
    #[test]
    fn group_cutter_carves_only_inside_its_group() {
        // Bystander [3,5)³ blocks; group body [0,4)³; cutter [2,4)³ (overlapping the
        // bystander's [3,4)³ corner). The group comes AFTER the bystander.
        let sealed = Scene::from_nodes(vec![
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
        // Expected: the notched body (the #73 sibling-level carve, resolved WITHOUT
        // the bystander) plus the whole bystander — the two parts are cell-disjoint
        // (the cutter removed every body cell the bystander touches).
        let notched_body_alone = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Plain, CombineOp::Subtract),
        ]);
        let bystander_alone = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [3, 3, 3],
            MaterialChoice::Wood,
            CombineOp::Union,
        )]);
        let mut expected = resolved_absolute_multiset(&notched_body_alone);
        for (key, count) in resolved_absolute_multiset(&bystander_alone) {
            assert!(
                !expected.contains_key(&key),
                "the notched body and the bystander must be cell-disjoint at {key:?}"
            );
            expected.insert(key, count);
        }
        assert_eq!(
            resolved_absolute_multiset(&sealed),
            expected,
            "the group's cutter must carve the group's body and NOTHING outside the group"
        );
    }

    /// Acceptance #2: a cutter placed AFTER a Group (as its sibling) carves the
    /// group's COMPOSED body — a root-level boolean sees the finished body exactly as
    /// it would see a flat sibling.
    #[test]
    fn cutter_after_group_carves_the_groups_composed_body() {
        let grouped = Scene::from_nodes(vec![
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
        let flat = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        assert_eq!(
            resolved_absolute_multiset(&grouped),
            resolved_absolute_multiset(&flat),
            "a root cutter after a group must carve the group's composed body"
        );
    }

    /// Acceptance #3: a definition's internal Subtract is fully SPENT inside the
    /// definition (the sealed part) — every instance places the finished (notched)
    /// body, and a bystander before the instances stays whole even where an
    /// instance's internal cutter volume overlaps it.
    #[test]
    fn definition_internal_subtract_is_spent_inside_the_definition() {
        let def_id = DefId(1);
        let mut scene = Scene::from_nodes(vec![
            // Bystander overlapping the FIRST instance's internal cutter box.
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

        // Expected: the bystander plus one notched body at each instance offset —
        // all three parts cell-disjoint.
        let notched_at = |offset_blocks: [i64; 3]| {
            Scene::from_nodes(vec![
                box_tool([4, 4, 4], offset_blocks, MaterialChoice::Stone, CombineOp::Union),
                box_tool(
                    [2, 2, 2],
                    [offset_blocks[0] + 2, offset_blocks[1] + 2, offset_blocks[2] + 2],
                    MaterialChoice::Plain,
                    CombineOp::Subtract,
                ),
            ])
        };
        let bystander_alone = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [3, 3, 3],
            MaterialChoice::Wood,
            CombineOp::Union,
        )]);
        let mut expected = resolved_absolute_multiset(&bystander_alone);
        for offset in [[0, 0, 0], [8, 0, 0]] {
            for (key, count) in resolved_absolute_multiset(&notched_at(offset)) {
                assert!(
                    !expected.contains_key(&key),
                    "the expected parts must be cell-disjoint at {key:?}"
                );
                expected.insert(key, count);
            }
        }
        assert_eq!(
            resolved_absolute_multiset(&scene),
            expected,
            "each instance must place the pre-composed notched body; the internal \
             cutter must be fully spent inside the definition"
        );
    }

    /// Acceptance #4: nested scopes — an INNER group's cutter cannot escape to the
    /// OUTER group's body, while still acting on its own (inner) siblings. The inner
    /// cutter's box overlaps both the inner Wood body and the outer Stone body; only
    /// the inner body loses cells.
    #[test]
    fn nested_group_cutter_cannot_escape_to_the_outer_group() {
        let density = DENSITY as i64;
        let scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Outer",
            vec![
                // Outer body: blocks [0,4)³.
                box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union).into(),
                NodeBuilder::group(
                    "Inner",
                    vec![
                        // Inner body: blocks [2,4)³ (overlapping the outer body).
                        box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Union)
                            .into(),
                        // Inner cutter: blocks [3,5)³ — overlaps BOTH bodies, may
                        // carve only the inner one.
                        box_tool([2, 2, 2], [3, 3, 3], MaterialChoice::Plain, CombineOp::Subtract)
                            .into(),
                    ],
                ),
            ],
        )]);
        let resolved = resolved_absolute_multiset(&scene);

        // (a) Inside the cutter's box, every cell of the outer body's span survives —
        // as STONE (the inner body's cells there were carved, so Wood never wins the
        // overlap; the cutter never escaped to the Stone).
        let cutter_low = 3 * density;
        let body_high = 4 * density; // the outer body ends at block 4.
        for x in cutter_low..body_high {
            for y in cutter_low..body_high {
                for z in cutter_low..body_high {
                    assert_eq!(
                        resolved.get(&([x, y, z], MaterialChoice::Stone.block_id().0)),
                        Some(&1),
                        "outer-body cell [{x},{y},{z}] inside the inner cutter's box \
                         must survive as Stone (the cutter is sealed inside the inner group)"
                    );
                }
            }
        }
        // (b) The cutter DID act inside its own scope: no Wood cell anywhere in the
        // cutter's box…
        for ((index, material), _count) in &resolved {
            let inside_cutter =
                (0..3).all(|axis| index[axis] >= cutter_low && index[axis] < cutter_low + 2 * density);
            assert!(
                !(inside_cutter && *material == MaterialChoice::Wood.block_id().0),
                "inner-body cell {index:?} inside the inner cutter's box must be carved"
            );
        }
        // …while the inner body's cells OUTSIDE the cutter's box survive as Wood.
        assert!(
            resolved
                .keys()
                .any(|(_, material)| *material == MaterialChoice::Wood.block_id().0),
            "the inner body must survive outside the cutter's box"
        );
    }

    /// The provable-equivalence regression (the pure-Union goldens' law): for a scene
    /// with NO booleans, pre-composing groups is occupancy- and material-identical to
    /// the flat walk (later-wins material under depth-first order is preserved whether
    /// or not groups pre-compose).
    #[test]
    fn pure_union_groups_pre_compose_identically_to_the_flat_walk() {
        // Three OVERLAPPING boxes so later-wins material is actually exercised, with
        // the middle one wrapped in a group in one scene and flat in the other.
        let grouped = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool([3, 3, 3], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union)),
            NodeBuilder::group(
                "Middle",
                vec![box_tool([3, 3, 3], [1, 1, 0], MaterialChoice::Wood, CombineOp::Union).into()],
            ),
            NodeBuilder::Leaf(box_tool([3, 3, 3], [2, 2, 0], MaterialChoice::Plain, CombineOp::Union)),
        ]);
        let flat = Scene::from_nodes(vec![
            box_tool([3, 3, 3], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([3, 3, 3], [1, 1, 0], MaterialChoice::Wood, CombineOp::Union),
            box_tool([3, 3, 3], [2, 2, 0], MaterialChoice::Plain, CombineOp::Union),
        ]);
        assert_eq!(
            resolved_absolute_multiset(&grouped),
            resolved_absolute_multiset(&flat),
            "pure-Union scenes must resolve identically whether or not groups pre-compose"
        );
    }

    /// The chunk-addressable resolve applies the SAME sealed-scope semantics as the
    /// monolithic oracle: reassembling every covering chunk reproduces the scoped
    /// composition exactly (composition is cell-local, so restriction to a chunk
    /// commutes with the fold).
    #[test]
    fn chunked_resolve_matches_monolithic_for_scoped_scene() {
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
        let monolithic = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        let chunked = scene.resolve_region_via_chunks(DENSITY, 0);
        assert_eq!(
            occupied_multiset(&chunked, [0, 0, 0]),
            occupied_multiset(&monolithic, scene.recentre_voxels(DENSITY)),
            "chunked scoped resolve must equal the monolithic oracle (recentre-normalised)"
        );
    }

    /// Invalidation (#74): a GROUP's operation flip must dirty the group's subtree
    /// AABB — it changes the fingerprint of every leaf INSIDE the scope (the scope
    /// path, ops included, is part of each leaf's fingerprint) and of no leaf outside,
    /// so the edit diff dirties exactly the enclosed leaves' AABBs (whose union is the
    /// subtree AABB) and the store RE-CLASSIFIES those chunks.
    #[test]
    fn group_operation_flip_changes_only_the_enclosed_leaf_fingerprints() {
        let build = |group_operation: CombineOp| {
            let mut scene = Scene::from_nodes(vec![
                NodeBuilder::Leaf(box_tool(
                    [2, 2, 2],
                    [8, 0, 0],
                    MaterialChoice::Wood,
                    CombineOp::Union,
                )),
                NodeBuilder::group(
                    "Scope",
                    vec![
                        box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union)
                            .into(),
                        box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Plain, CombineOp::Subtract)
                            .into(),
                    ],
                ),
            ]);
            // Flip the GROUP node's own operation (roots[1] is the group).
            let group_id = scene.roots[1];
            scene
                .node_by_id_mut(group_id)
                .expect("the group resolves")
                .operation = group_operation;
            scene
        };
        let union_index = build(CombineOp::Union).build_leaf_spatial_index(DENSITY);
        let subtract_index = build(CombineOp::Subtract).build_leaf_spatial_index(DENSITY);
        assert_eq!(union_index.entries.len(), 3);
        assert_eq!(subtract_index.entries.len(), 3);
        // The outside leaf (entry 0, walk order) is untouched by the flip…
        assert_eq!(
            union_index.entries[0].fingerprint, subtract_index.entries[0].fingerprint,
            "a leaf OUTSIDE the flipped group must keep its fingerprint"
        );
        // …while both enclosed leaves' fingerprints change (their scope path carries
        // the group's operation).
        for entry in 1..=2 {
            assert_ne!(
                union_index.entries[entry].fingerprint,
                subtract_index.entries[entry].fingerprint,
                "a leaf INSIDE the flipped group must change its fingerprint (entry {entry})"
            );
        }
    }
