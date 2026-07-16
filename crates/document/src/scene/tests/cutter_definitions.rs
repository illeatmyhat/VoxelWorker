use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- ADR 0017 / #76: reusable cutter definitions — the dense oracle ----
    //
    // A CUTTER as a reusable part: a definition instanced with `operation:
    // Subtract`. The sealed definition body pre-composes (issue #74), and the
    // instance folds the finished body into the host scope as a carve — no new
    // node kind, just the sealed-scope rule meeting the instance's node-level
    // `CombineOp`. These tests pin the dense-oracle semantics: each placement
    // carves its own host at its own transform, and editing the ONE definition
    // re-carves EVERY placement (reuse by reference).

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

    /// An `Instance(def_id)` node at `offset_blocks` carrying `operation`.
    fn instance_node(
        def_id: DefId,
        offset_blocks: [i64; 3],
        operation: CombineOp,
        name: &str,
    ) -> Node {
        let mut node = Node::new(name, NodeContent::Instance(def_id));
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// The two-hosts fixture (the golden's shape): two separated Stone host boxes,
    /// then ONE `cutter_blocks`-sized cutter definition instanced twice under
    /// Subtract, each placement overlapping its own host's top corner octant.
    fn two_hosts_with_instanced_cutters(cutter_blocks: u32) -> Scene {
        let cutter_def_id = DefId(1);
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [8, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            instance_node(cutter_def_id, [2, 2, 2], CombineOp::Subtract, "Cut 1"),
            instance_node(cutter_def_id, [10, 2, 2], CombineOp::Subtract, "Cut 2"),
        ]);
        scene.add_definition(
            cutter_def_id,
            "Corner cutter",
            vec![box_tool(
                [cutter_blocks; 3],
                [0, 0, 0],
                MaterialChoice::Wood,
                CombineOp::Union,
            )],
        );
        scene
    }

    /// The flat scene that must resolve identically to
    /// [`two_hosts_with_instanced_cutters`]: the same hosts carved by two PLAIN
    /// Subtract leaves at the instances' placements.
    fn two_hosts_with_flat_cutters(cutter_blocks: u32) -> Scene {
        Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([4, 4, 4], [8, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([cutter_blocks; 3], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
            box_tool([cutter_blocks; 3], [10, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
        ])
    }

    /// Acceptance #1 (the golden's oracle): ONE cutter definition instanced twice
    /// carves TWO separate hosts at their placements — each placement's carve equals
    /// a plain Subtract leaf at the instance transform (the instance's operation
    /// applies to the definition's pre-composed body under the instance's frame,
    /// ADR 0008 carried-frame discipline), and no cell anywhere carries the cutter's
    /// material.
    #[test]
    fn one_cutter_definition_instanced_twice_carves_both_hosts() {
        let instanced = two_hosts_with_instanced_cutters(2);
        let flat = two_hosts_with_flat_cutters(2);
        let carved = resolved_absolute_multiset(&instanced);
        assert_eq!(
            carved,
            resolved_absolute_multiset(&flat),
            "each instanced-cutter placement must carve exactly like a plain Subtract \
             leaf at the instance's transform"
        );
        // Both hosts lost exactly one 2³-block corner octant; nothing is Wood.
        let host_voxels = (4 * DENSITY as usize).pow(3);
        let notch_voxels = (2 * DENSITY as usize).pow(3);
        assert_eq!(
            carved.len(),
            2 * (host_voxels - notch_voxels),
            "each host must lose exactly its own cutter-box of voxels"
        );
        assert!(
            carved
                .keys()
                .all(|(_index, material)| *material == MaterialChoice::Stone.block_id().0),
            "a Subtract instance never stamps — every survivor keeps the host's Stone"
        );
    }

    /// Acceptance #2 (reuse by reference): editing the ONE definition's geometry
    /// re-carves EVERY placement — growing the def's box from 1³ to 2³ blocks makes
    /// both hosts' notches grow identically, matching a from-scratch scene authored
    /// at the new size.
    #[test]
    fn editing_the_definition_recarves_every_placement() {
        let mut scene = two_hosts_with_instanced_cutters(1);
        assert_eq!(
            resolved_absolute_multiset(&scene),
            resolved_absolute_multiset(&two_hosts_with_flat_cutters(1)),
            "the pre-edit carve matches the 1-block flat oracle"
        );

        // Edit the shared definition body IN PLACE (the def's child node lives in
        // the scene arena; both instances reference it), growing the cutter to 2³.
        let def_body_id = scene
            .def_by_id(DefId(1))
            .expect("the cutter definition exists")
            .children[0];
        let grown = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, DENSITY);
        match &mut scene
            .node_by_id_mut(def_body_id)
            .expect("the definition body node resolves")
            .content
        {
            NodeContent::Tool { shape, .. } => *shape = grown,
            other => panic!("the cutter def body is a Tool, got {other:?}"),
        }

        assert_eq!(
            resolved_absolute_multiset(&scene),
            resolved_absolute_multiset(&two_hosts_with_flat_cutters(2)),
            "one definition edit must re-carve BOTH placements (reuse by reference)"
        );
    }

    /// The ordering law holds for instanced cutters exactly as for leaf cutters: a
    /// Subtract instance placed BEFORE its host carves nothing.
    #[test]
    fn subtract_instance_before_its_host_is_a_no_op() {
        let cutter_def_id = DefId(1);
        let mut cutter_first = Scene::from_nodes(vec![
            instance_node(cutter_def_id, [2, 2, 2], CombineOp::Subtract, "Cut"),
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
        ]);
        cutter_first.add_definition(
            cutter_def_id,
            "Corner cutter",
            vec![box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Wood, CombineOp::Union)],
        );
        let host_alone = Scene::from_nodes(vec![box_tool(
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        assert_eq!(
            resolved_absolute_multiset(&cutter_first),
            resolved_absolute_multiset(&host_alone),
            "a Subtract instance preceding its host must remove nothing (the ordered fold)"
        );
    }

    /// A leaf's enclosing-scope fingerprint reflects the INSTANCE's operation, so
    /// flipping an instance Union↔Subtract dirties its expanded leaves' AABBs and
    /// the store re-classifies them (the same invalidation contract as #73's leaf
    /// flip and #74's group flip).
    #[test]
    fn instance_operation_flip_changes_the_expanded_leaf_fingerprints() {
        let union_scene = {
            let mut scene = two_hosts_with_instanced_cutters(2);
            // Same graph, but the first instance folds under Union instead.
            let first_instance_id = scene.roots[2];
            scene
                .node_by_id_mut(first_instance_id)
                .expect("the instance resolves")
                .operation = CombineOp::Union;
            scene
        };
        let subtract_scene = two_hosts_with_instanced_cutters(2);
        let union_index = union_scene.build_leaf_spatial_index(DENSITY);
        let subtract_index = subtract_scene.build_leaf_spatial_index(DENSITY);
        // Leaves: two hosts + one expanded def body per instance = 4 entries.
        assert_eq!(union_index.entries.len(), 4);
        // The FIRST instance's expanded leaf differs; the second instance's and the
        // hosts' fingerprints are untouched (the flip dirties only its own subtree).
        assert_ne!(
            union_index.entries[2].fingerprint, subtract_index.entries[2].fingerprint,
            "an instance operation flip must dirty its expanded leaves"
        );
        assert_eq!(
            union_index.entries[3].fingerprint, subtract_index.entries[3].fingerprint,
            "the sibling instance's expansion is untouched by the flip"
        );
    }
