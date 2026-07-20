use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- ADR 0018 Decision 6: the "Show booleans" mode's document-side derivation ----
    //
    // `boolean_operand_body_slices` walks the ACTIVE selection's subtree and collects the
    // standalone body slice of EVERY enabled Subtract/Intersect operand inside it (the
    // selected node included when it is a boolean): absolute placement is kept (ancestor
    // Group offsets bake into the slice root — ADR 0008 carried frames), each emitted
    // body's own operation is neutralised to Union (a Subtract root at fold start would
    // yield nothing), and the Union bodies (already visible) never join the set. Selecting
    // the root part covers the whole scene; a fixture instance splices its children.

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
        assert!(scene.boolean_operand_body_slices().is_empty());
    }

    #[test]
    fn hidden_selection_yields_no_slices() {
        let mut scene = host_and_cutter_scene();
        let cutter_id = scene.roots[1];
        scene.active = Some(cutter_id);
        scene
            .node_by_id_mut(cutter_id)
            .expect("cutter resolves")
            .enabled = false;
        assert!(
            scene.boolean_operand_body_slices().is_empty(),
            "a disabled node contributes no body — no ghost"
        );
    }

    /// The core carve case: selecting a Subtract cutter yields ONE slice carrying the
    /// Subtract role, and resolving that slice standalone produces the cutter's FULL
    /// own body (not the carved remainder, and not nothing) at its absolute placement.
    #[test]
    fn selected_subtract_cutter_resolves_its_full_body_in_place() {
        let mut scene = host_and_cutter_scene();
        scene.active = Some(scene.roots[1]);

        let slices = scene.boolean_operand_body_slices();
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

    /// A non-boolean leaf selection is degenerate but consistent: selecting the Union
    /// host scopes the walk to that ingredient — nothing to reveal.
    #[test]
    fn selected_union_leaf_yields_no_slices() {
        let mut scene = host_and_cutter_scene();
        scene.active = Some(scene.roots[0]);
        assert!(scene.boolean_operand_body_slices().is_empty());
    }

    /// The derivation bound: a slice's covering chunk range is the SELECTED operand's
    /// extent, never the whole scene's — re-deriving on a selection change resolves only
    /// the ghosted body.
    #[test]
    fn slice_covers_only_the_selected_operand_extent() {
        // A small cutter beside a FAR-away large host: the whole-scene chunk range is
        // wide, the cutter slice's must stay the cutter's own.
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [40, 0, 0], CombineOp::Union, "Far host"),
            box_tool([2, 2, 2], [0, 0, 0], CombineOp::Subtract, "Cutter"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.active = Some(scene.roots[1]);

        let slices = scene.boolean_operand_body_slices();
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

        let slices = scene.boolean_operand_body_slices();
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

    /// Selecting a Group collects one slice per boolean operand in its subtree —
    /// Subtract AND Intersect, each under its own operation with the ancestor offset
    /// baked in — and NEVER a Union body.
    #[test]
    fn selected_group_collects_every_subtree_boolean_and_never_a_union() {
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
        scene.active = Some(scene.roots[0]);

        let slices = scene.boolean_operand_body_slices();
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

    /// "The node itself included if it is a boolean": selecting a boolean LEAF ghosts
    /// that node's own body.
    #[test]
    fn selected_boolean_leaf_ghosts_itself() {
        let mut scene = host_and_cutter_scene();
        let cutter_id = scene.roots[1];
        scene.active = Some(cutter_id);
        let slices = scene.boolean_operand_body_slices();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].0, CombineOp::Subtract);
    }

    /// Selecting the ROOT PART covers the WHOLE scene: every boolean operand across all
    /// top-level subtrees ghosts (the scene-wide master).
    #[test]
    fn selecting_the_root_part_covers_the_whole_scene() {
        let mut scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Host A"),
            box_tool([2, 2, 2], [1, 1, 1], CombineOp::Subtract, "Cutter A"),
            box_tool([4, 4, 4], [20, 0, 0], CombineOp::Union, "Host B"),
            box_tool([2, 2, 2], [21, 1, 1], CombineOp::Intersect, "Mask B"),
        ]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        scene.active = Some(ROOT_NODE_ID);

        let slices = scene.boolean_operand_body_slices();
        let operations: Vec<CombineOp> = slices.iter().map(|(op, _)| *op).collect();
        assert_eq!(
            operations,
            vec![CombineOp::Subtract, CombineOp::Intersect],
            "every boolean in the scene ghosts when the root part is selected"
        );
    }

    /// A disabled operand (or a whole disabled subtree) contributes nothing: a disabled
    /// node stamps nothing into the composition, so there is no invisible-by-success body
    /// to reveal.
    #[test]
    fn disabled_operands_contribute_nothing() {
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "Assembly",
            vec![
                box_tool([4, 4, 4], [0, 0, 0], CombineOp::Union, "Body").into(),
                box_tool([2, 2, 2], [2, 2, 2], CombineOp::Subtract, "Cutter").into(),
            ],
        )]);
        scene.voxels_per_block = DENSITY;
        scene.ensure_node_ids();
        let cutter_id = scene
            .id_at_path(&NodePath::from_indices(vec![0, 1]))
            .expect("cutter resolves");
        scene.node_by_id_mut(cutter_id).expect("cutter resolves").enabled = false;
        scene.active = Some(scene.roots[0]);
        assert!(scene.boolean_operand_body_slices().is_empty());
    }

    /// A FIXTURE instance (inert own operation, issue #77) selected: its definition
    /// children splice into the host fold, so its BOOLEAN children join the set under the
    /// instance's transform, and its Union children do not.
    #[test]
    fn selected_fixture_instance_splices_its_boolean_children() {
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
        scene.active = Some(scene.roots[0]);

        let slices = scene.boolean_operand_body_slices();
        assert_eq!(slices.len(), 1, "the Opening joins; the Union Frame does not");
        let (operation, slice) = &slices[0];
        assert_eq!(*operation, CombineOp::Subtract);
        let density = DENSITY as i64;
        let root = slice.node_by_id(slice.roots[0]).expect("slice root resolves");
        assert_eq!(root.transform.offset_voxels, [2 * density, 0, 2 * density]);
    }

    /// A SEALED-definition Instance is a leaf operand: instanced under `Subtract` it is
    /// the reusable cutter (issue #76), and its FINISHED definition body is the ghost.
    /// Selecting the enclosing Group reaches it.
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
        scene.active = Some(scene.roots[0]);

        let slices = scene.boolean_operand_body_slices();
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].0, CombineOp::Subtract);
    }
