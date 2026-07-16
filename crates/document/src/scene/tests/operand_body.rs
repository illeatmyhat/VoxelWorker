use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- Issue #78: the selected-operand ghost's document-side derivation ----
    //
    // `active_operand_body_slices` re-roots the document on the ACTIVE node so the
    // ghost can resolve the node's OWN body standalone: absolute placement is kept
    // (ancestor Group offsets bake into the slice root — ADR 0008 carried frames),
    // the root's own operation is neutralised to Union (a Subtract root at fold
    // start would yield nothing), and a fixture instance yields one slice per
    // spliced child under the CHILD's operation (ADR 0017 Decision 4: the
    // instance's own operation is inert).

    const DENSITY: u32 = 8;

    /// A whole-block Box Tool at `offset_blocks` carrying `operation`.
    fn box_tool(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        operation: CombineOp,
        name: &str,
    ) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, DENSITY);
        let mut node = Node::new(
            name,
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// A host box carved by a smaller Subtract cutter (the demo-subtract shape).
    fn host_and_cutter_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host"),
            box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene
    }

    #[test]
    fn no_selection_yields_no_slices() {
        let mut scene = host_and_cutter_scene();
        scene.active = None;
        assert!(scene.active_operand_body_slices().is_empty());
    }

    #[test]
    fn hidden_selection_yields_no_slices() {
        let mut scene = host_and_cutter_scene();
        let cutter_id = scene.roots[1];
        scene.active = Some(cutter_id);
        scene
            .node_by_id_mut(cutter_id)
            .expect("cutter resolves")
            .visible = false;
        assert!(
            scene.active_operand_body_slices().is_empty(),
            "a hidden node contributes no body — no ghost"
        );
    }

    /// The core carve case: selecting a Subtract cutter yields ONE slice carrying the
    /// Subtract role, and resolving that slice standalone produces the cutter's FULL
    /// own body (not the carved remainder, and not nothing) at its absolute placement.
    #[test]
    fn subtract_cutter_slice_resolves_its_full_body_in_place() {
        let mut scene = host_and_cutter_scene();
        scene.active = Some(scene.roots[1]);

        let slices = scene.active_operand_body_slices();
        assert_eq!(slices.len(), 1);
        let (operation, slice) = &slices[0];
        assert_eq!(*operation, CombineOp::Subtract, "the ghost styles by the node's own role");

        // The slice resolves the cutter's own 2³-block body — the ghost's mesh source.
        let resolved = slice.resolve_region(RegionBlocks::new([8, 8, 8]), DENSITY, 0);
        let expected = {
            let mut lone = Scene::from_nodes(vec![box_tool(
                [2, 2, 2],
                [1, 1, 1],
                CombineOp::Union,
                "Cutter body",
            )]);
            lone.voxels_per_block = DENSITY;
            lone.ensure_node_ids();
            lone.resolve_region(RegionBlocks::new([8, 8, 8]), DENSITY, 0)
        };
        assert_eq!(
            occupied_multiset(&resolved, resolved.recentre_voxels),
            occupied_multiset(&expected, expected.recentre_voxels),
            "the slice body must be the cutter's own full body at its absolute placement"
        );
    }

    /// The derivation bound: a slice's covering chunk range is the SELECTED SUBTREE's
    /// extent, never the whole scene's — re-deriving the ghost on a selection change
    /// resolves only the selected body.
    #[test]
    fn slice_covers_only_the_selected_subtree_extent() {
        // A small cutter beside a FAR-away large host: the whole-scene chunk range is
        // wide, the cutter slice's must stay the cutter's own.
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [40, 0, 0], CombineOp::Union, "Far host"),
            box_tool([2, 2, 2], [0, 0, 0], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.active = Some(scene.roots[1]);

        let slices = scene.active_operand_body_slices();
        let (_, slice) = &slices[0];
        let (slice_min, slice_max) = slice
            .covering_chunk_range(DENSITY)
            .expect("the cutter slice has an extent");
        let (scene_min, scene_max) = scene
            .covering_chunk_range(DENSITY)
            .expect("the scene has an extent");
        assert!(
            slice_max[0] < scene_max[0],
            "the slice's chunk range must not span to the far host \
             (slice {slice_min:?}..={slice_max:?}, scene {scene_min:?}..={scene_max:?})"
        );
        // The 2-block cutter at the origin covers exactly one 4-block chunk.
        assert_eq!((slice_min, slice_max), ([0, 0, 0], [0, 0, 0]));
    }

    /// A Group child's slice keeps its WORLD placement: the ancestor Group offsets are
    /// baked into the slice root's transform (ADR 0008 — carry the frame).
    #[test]
    fn group_child_slice_keeps_its_world_placement() {
        let child = box_tool([2, 2, 2], [2, 0, 0], CombineOp::Subtract, "Nested cutter");
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group_at(
            "Assembly",
            [3, 0, 5],
            DENSITY,
            vec![child.into()],
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        // Select the nested child (path [0, 0]).
        scene.active = scene.id_at_path(&NodePath::from_indices(vec![0, 0]));

        let slices = scene.active_operand_body_slices();
        assert_eq!(slices.len(), 1);
        let (_, slice) = &slices[0];
        let root = slice
            .node_by_id(slice.roots[0])
            .expect("slice root resolves");
        let density = DENSITY as i64;
        assert_eq!(
            root.transform.offset_voxels,
            [(3 + 2) * density, 0, 5 * density],
            "group offset + child offset must bake into the slice root"
        );
        assert_eq!(root.operation, CombineOp::Union, "the root's own role is neutralised");
    }

    /// A FIXTURE instance (inert own operation) yields one slice PER spliced child,
    /// each under the CHILD's operation and the instance's transform.
    #[test]
    fn fixture_instance_slices_one_body_per_spliced_child() {
        let window_def = DefId(1);
        let wall = box_tool([8, 1, 6], [0, 0, 0], CombineOp::Union, "Wall");
        let mut window = Node::new("Window", NodeContent::Instance(window_def));
        window.transform = NodeTransform::from_blocks([2, 0, 2], DENSITY);
        let mut scene = Scene::from_nodes(vec![wall, window]);
        scene.add_definition(
            window_def,
            "Window",
            vec![
                box_tool([3, 1, 3], [0, 0, 0], CombineOp::Subtract, "Opening"),
                box_tool([3, 1, 1], [0, 0, 0], CombineOp::Union, "Frame"),
            ],
        );
        scene.set_definition_fixture(window_def, true);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.active = Some(scene.roots[1]);

        let slices = scene.active_operand_body_slices();
        let operations: Vec<CombineOp> = slices.iter().map(|(op, _)| *op).collect();
        assert_eq!(
            operations,
            vec![CombineOp::Subtract, CombineOp::Union],
            "each spliced child ghosts under its OWN operation (the inert instance op is \
             never consulted)"
        );
        let density = DENSITY as i64;
        for (_, slice) in &slices {
            let root = slice.node_by_id(slice.roots[0]).expect("slice root resolves");
            assert_eq!(
                root.transform.offset_voxels,
                [2 * density, 0, 2 * density],
                "each spliced child is placed under the instance's transform"
            );
            assert_eq!(root.operation, CombineOp::Union);
        }
    }

    // ---- Issue #79: the PERSISTENT child-boolean ghost's derivation ----
    //
    // `shown_child_boolean_body_slices` collects, for every subtree covered by a
    // node with "Show child booleans" checked, the standalone body slice of EVERY
    // visible Subtract/Intersect operand — the same slice mechanics as the #78
    // selection derivation above, keyed by the per-node flag instead of the
    // selection, and restricted to the boolean masks (Union bodies are already
    // visible, so they never join the set).

    /// The flag defaults OFF everywhere: a scene with booleans but no checkbox
    /// derives NOTHING (all existing goldens keep the finished look).
    #[test]
    fn without_the_flag_no_persistent_slices_derive() {
        let scene = host_and_cutter_scene();
        assert!(scene.shown_child_boolean_body_slices().is_empty());
    }

    /// The core case: a checked Group yields one slice per boolean operand in its
    /// subtree — Subtract AND Intersect, each under its own operation with the
    /// ancestor offset baked in — and NEVER a Union body.
    #[test]
    fn checked_group_collects_every_subtree_boolean_and_never_a_union() {
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group_at(
            "Assembly",
            [3, 0, 5],
            DENSITY,
            vec![
                box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Body").into(),
                box_tool([2, 2, 2], [2, 2, 2], CombineOp::Subtract, "Cutter").into(),
                box_tool([3, 3, 3], [0, 0, 0], CombineOp::Intersect, "Mask").into(),
            ],
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene
            .node_by_id_mut(scene.roots[0])
            .expect("group resolves")
            .show_child_booleans = true;

        let slices = scene.shown_child_boolean_body_slices();
        let operations: Vec<CombineOp> = slices.iter().map(|(op, _)| *op).collect();
        assert_eq!(
            operations,
            vec![CombineOp::Subtract, CombineOp::Intersect],
            "both boolean masks join the set (document order); the Union body never does"
        );
        // The cutter slice keeps its WORLD placement: group offset + own offset.
        let density = DENSITY as i64;
        let (_, cutter_slice) = &slices[0];
        let root = cutter_slice
            .node_by_id(cutter_slice.roots[0])
            .expect("slice root resolves");
        assert_eq!(root.transform.offset_voxels, [(3 + 2) * density, 2 * density, (5 + 2) * density]);
        assert_eq!(root.operation, CombineOp::Union, "the slice root's own role is neutralised");
    }

    /// "The node itself included if it is a boolean": checking a boolean LEAF (or any
    /// boolean node) ghosts that node's own body.
    #[test]
    fn checked_boolean_leaf_ghosts_itself() {
        let mut scene = host_and_cutter_scene();
        let cutter_id = scene.roots[1];
        scene
            .node_by_id_mut(cutter_id)
            .expect("cutter resolves")
            .show_child_booleans = true;
        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].0, CombineOp::Subtract);
    }

    /// The nesting/dedupe rule: an OUTER checked node covers inner subtrees, and a
    /// redundantly-checked inner node must not emit its operands twice (a body drawn
    /// twice would read as doubled ghost alpha — the style bug the derivation guards).
    #[test]
    fn nested_checked_scopes_never_emit_a_body_twice() {
        let inner = NodeBuilder::group(
            "Inner",
            vec![box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter").into()],
        );
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group("Outer", vec![inner])]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        // Check BOTH the outer and the inner group.
        let outer_id = scene.roots[0];
        scene.node_by_id_mut(outer_id).expect("outer resolves").show_child_booleans = true;
        let inner_id = scene
            .id_at_path(&NodePath::from_indices(vec![0, 0]))
            .expect("inner group resolves");
        scene.node_by_id_mut(inner_id).expect("inner resolves").show_child_booleans = true;

        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1, "one cutter, one body — the set is deduped");
    }

    /// An inner checkbox ALONE scopes to the inner subtree: a boolean beside the inner
    /// group (inside the unchecked outer) stays hidden.
    #[test]
    fn inner_checkbox_scopes_to_the_inner_subtree() {
        let inner = NodeBuilder::group(
            "Inner",
            vec![box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Inner cutter").into()],
        );
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Outer",
            vec![
                box_tool([2, 2, 2], [4, 4, 4], CombineOp::Subtract, "Outer cutter").into(),
                inner,
            ],
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        let inner_id = scene
            .id_at_path(&NodePath::from_indices(vec![0, 1]))
            .expect("inner group resolves");
        scene.node_by_id_mut(inner_id).expect("inner resolves").show_child_booleans = true;

        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1, "only the inner subtree's cutter is covered");
        let (_, slice) = &slices[0];
        let root = slice.node_by_id(slice.roots[0]).expect("slice root resolves");
        assert_eq!(
            root.transform.offset_voxels,
            [DENSITY as i64, DENSITY as i64, DENSITY as i64],
            "the emitted body is the INNER cutter, not the outer one"
        );
    }

    /// A hidden operand (or a whole hidden subtree) contributes nothing: a hidden node
    /// stamps nothing into the composition, so there is no invisible-by-success body
    /// to reveal.
    #[test]
    fn hidden_operands_contribute_no_persistent_slices() {
        let mut scene = host_and_cutter_scene();
        let host_id = scene.roots[0];
        let cutter_id = scene.roots[1];
        scene.node_by_id_mut(host_id).expect("host resolves").show_child_booleans = true;
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").show_child_booleans = true;
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").visible = false;
        assert!(scene.shown_child_boolean_body_slices().is_empty());
    }

    /// The cross-overlay dedupe rule: the ACTIVE node's body is the #78 selection
    /// ghost's (same style for a boolean), so the persistent set skips it — the two
    /// overlays never double a body's alpha. Deselecting restores it to the set.
    #[test]
    fn active_selection_is_excluded_from_the_persistent_set() {
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Assembly",
            vec![
                box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Body").into(),
                box_tool([2, 2, 2], [2, 2, 2], CombineOp::Subtract, "Cutter A").into(),
                box_tool([1, 1, 1], [1, 1, 1], CombineOp::Subtract, "Cutter B").into(),
            ],
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.node_by_id_mut(scene.roots[0]).expect("group resolves").show_child_booleans = true;
        assert_eq!(scene.shown_child_boolean_body_slices().len(), 2);

        // Select Cutter A: it leaves the persistent set (the selection ghost draws it).
        scene.active = scene.id_at_path(&NodePath::from_indices(vec![0, 1]));
        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1, "the active cutter is the selection ghost's body");

        scene.active = None;
        assert_eq!(scene.shown_child_boolean_body_slices().len(), 2, "deselect restores it");
    }

    /// A checked FIXTURE instance (inert own operation, issue #77) splices its
    /// definition children into the host fold — so its BOOLEAN children join the
    /// persistent set under the instance's transform, and its Union children do not.
    #[test]
    fn checked_fixture_instance_splices_its_boolean_children() {
        let window_def = DefId(1);
        let mut window = Node::new("Window", NodeContent::Instance(window_def));
        window.transform = NodeTransform::from_blocks([2, 0, 2], DENSITY);
        let mut scene = Scene::from_nodes(vec![window]);
        scene.add_definition(
            window_def,
            "Window",
            vec![
                box_tool([3, 1, 3], [0, 0, 0], CombineOp::Subtract, "Opening"),
                box_tool([3, 1, 1], [0, 0, 0], CombineOp::Union, "Frame"),
            ],
        );
        scene.set_definition_fixture(window_def, true);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.node_by_id_mut(scene.roots[0]).expect("instance resolves").show_child_booleans =
            true;

        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1, "the Opening joins; the Union Frame does not");
        let (operation, slice) = &slices[0];
        assert_eq!(*operation, CombineOp::Subtract);
        let density = DENSITY as i64;
        let root = slice.node_by_id(slice.roots[0]).expect("slice root resolves");
        assert_eq!(root.transform.offset_voxels, [2 * density, 0, 2 * density]);
    }

    /// A SEALED-definition Instance is a leaf operand: instanced under `Subtract` it is
    /// the reusable cutter (issue #76), and its FINISHED definition body is the ghost.
    #[test]
    fn sealed_cutter_instance_emits_its_finished_body() {
        let cutter_def = DefId(1);
        let mut cutter_instance = Node::new("Notch", NodeContent::Instance(cutter_def));
        cutter_instance.operation = CombineOp::Subtract;
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Assembly",
            vec![
                box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host").into(),
                cutter_instance.into(),
            ],
        )]);
        scene.add_definition(
            cutter_def,
            "Corner cutter",
            vec![box_tool([2, 2, 2], [2, 2, 2], CombineOp::Union, "Cutter body")],
        );
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.node_by_id_mut(scene.roots[0]).expect("group resolves").show_child_booleans = true;

        let slices = scene.shown_child_boolean_body_slices();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].0, CombineOp::Subtract);
    }

    /// Persistence (issue #79 acceptance): the flag survives the document's JSON
    /// round-trip, and an OLDER document without the field deserialises to OFF.
    #[test]
    fn show_child_booleans_persists_and_defaults_off() {
        let mut scene = host_and_cutter_scene();
        scene.node_by_id_mut(scene.roots[0]).expect("host resolves").show_child_booleans = true;
        let json = serde_json::to_string(&scene).expect("serialize");
        let back: Scene = serde_json::from_str(&json).expect("deserialize");
        assert!(back.node_by_id(back.roots[0]).expect("host resolves").show_child_booleans);
        assert!(!back.node_by_id(back.roots[1]).expect("cutter resolves").show_child_booleans);

        // An older document (no field) → serde default OFF.
        let stripped = json.replace("\"show_child_booleans\":true,", "");
        assert_ne!(stripped, json, "the field must have been present to strip");
        let old: Scene = serde_json::from_str(&stripped).expect("older document deserialises");
        assert!(!old.node_by_id(old.roots[0]).expect("host resolves").show_child_booleans);
    }
