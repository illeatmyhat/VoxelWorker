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
