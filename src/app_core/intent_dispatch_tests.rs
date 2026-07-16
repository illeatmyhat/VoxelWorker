    use crate::app_core::AppCore;
    use voxel_core::core_geom::MaterialChoice;
    use camera::OrbitCamera;
    use document::intent::{whole_block_offset, Intent, IntentEffect, NodeSpec};
    use document::scene::{
        DefId, Node, NodeBuilder, NodeContent, NodeGrids, NodeId, NodeTransform, Part, Point, Scene,
    };
    use document::sketch::{PlaneAxis, Sketch, SketchSolid};
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape};

    /// A headless [`AppCore`] for the dispatch tests. `apply_intent` reads no AppCore
    /// state (it borrows the scene), so a default camera suffices — no GPU.
    fn test_core() -> AppCore {
        AppCore::new(OrbitCamera::default())
    }

    /// A box Tool shape at the given BLOCK size (the default-ish fixture shape),
    /// built at the default density 16 (`size_voxels = blocks · 16`).
    fn box_shape(size: [u32; 3]) -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, size, 1, 16)
    }

    /// A rectangle-footprint sketch→extrude producer at the given BLOCK size, built
    /// at the default density 16 (the "box footprint" sketch — `PlaneAxis::Z` is the
    /// footprint-extrude-up default: profile on the XY ground, extruded up along +Z).
    fn box_sketch(size_blocks: [u32; 3]) -> SketchSolid {
        let density = 16u32;
        let grid_x = (size_blocks[0] * density) as i64;
        let grid_y = (size_blocks[1] * density) as i64;
        let grid_z = size_blocks[2] * density;
        SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, grid_x, grid_y), grid_z)
    }

    /// A Sketch node named `"Sketch"` (matching [`NodeSpec::into_node`]).
    fn sketch_node(producer: SketchSolid, material: MaterialChoice) -> Node {
        Node::new("Sketch", NodeContent::SketchTool { producer, material })
    }

    /// A Tool node named after its kind (matching [`NodeSpec::into_node`]).
    fn tool_node(shape: SdfShape, material: MaterialChoice) -> Node {
        Node::new(format!("{:?}", shape.kind), NodeContent::Tool { shape, material })
    }

    /// A normalized two-Tool scene with stable ids minted, the first node active.
    fn two_tool_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone),
            tool_node(box_shape([3, 1, 4]), MaterialChoice::Wood),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        scene
    }

    /// The stable id of the top-level node at `index`.
    fn root_id(scene: &Scene, index: usize) -> NodeId {
        scene.roots[index]
    }

    /// Assert `apply_intent(intent)` produces the SAME scene as `direct` applied to a
    /// clone — the core `apply_intent ≡ direct op` invariant. Both sides start from
    /// the SAME scene state (so id-minting counters match), so the scenes compare
    /// equal (Scene derives PartialEq).
    fn assert_dispatch_matches(scene: &Scene, intent: Intent, direct: impl FnOnce(&mut Scene)) {
        let mut core = test_core();
        let mut applied = scene.clone();
        core.apply_intent(&mut applied, intent);
        let mut expected = scene.clone();
        direct(&mut expected);
        assert_eq!(applied, expected);
    }

    // === Structural ===

    #[test]
    fn add_node_dispatches_to_add_node() {
        let scene = two_tool_scene();
        let spec = NodeSpec::Tool {
            shape: box_shape([5, 5, 5]),
            material: MaterialChoice::Plain,
        };
        assert_dispatch_matches(
            &scene,
            Intent::AddNode { content: spec.clone() },
            |s| {
                s.add_node(spec.into_node());
            },
        );
    }

    #[test]
    fn add_node_clouds_part_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::AddNode { content: NodeSpec::CloudsPart },
            |s| {
                s.add_node(NodeSpec::CloudsPart.into_node());
            },
        );
    }

    #[test]
    fn add_node_sketch_dispatches() {
        let scene = two_tool_scene();
        let spec = NodeSpec::Sketch {
            producer: box_sketch([3, 2, 4]),
            material: MaterialChoice::Wood,
        };
        assert_dispatch_matches(
            &scene,
            Intent::AddNode { content: spec.clone() },
            |s| {
                s.add_node(spec.into_node());
            },
        );
    }

    #[test]
    fn add_child_dispatches_to_add_child_to_group() {
        // A scene with a Group so the child has somewhere to land.
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into()],
        )]);
        scene.ensure_node_ids();
        let group_id = root_id(&scene, 0);
        let spec = NodeSpec::Tool {
            shape: box_shape([4, 4, 4]),
            material: MaterialChoice::Wood,
        };
        assert_dispatch_matches(
            &scene,
            Intent::AddChild { group: group_id, content: spec.clone() },
            |s| {
                s.add_child_to_group(group_id, spec.into_node());
            },
        );
    }

    #[test]
    fn group_node_dispatches_via_active() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(&scene, Intent::GroupNode { target }, |s| {
            s.active = Some(target);
            s.group_active();
        });
    }

    #[test]
    fn make_definition_dispatches_via_active() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::MakeDefinition { target, name: "House".to_string() },
            |s| {
                s.active = Some(target);
                s.make_definition_from_active("House".to_string());
            },
        );
    }

    #[test]
    fn add_instance_dispatches() {
        // Build a scene that already has a definition to instance.
        let mut scene = two_tool_scene();
        let target = root_id(&scene, 0);
        scene.active = Some(target);
        let def_id = scene.make_definition_from_active("Body").expect("definition made");
        assert_dispatch_matches(&scene, Intent::AddInstance { def: def_id }, |s| {
            s.add_instance(def_id);
        });
    }

    #[test]
    fn remove_node_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(&scene, Intent::RemoveNode { target }, |s| {
            s.remove_node(target);
        });
    }

    // === Node field writes ===

    #[test]
    fn set_visible_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetVisible { target, visible: false },
            |s| {
                s.set_node_visible(target, false);
            },
        );
    }

    #[test]
    fn set_shape_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let shape = box_shape([7, 7, 7]);
        assert_dispatch_matches(&scene, Intent::SetShape { target, shape: shape.clone() }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                if let NodeContent::Tool { shape: node_shape, .. } = &mut node.content {
                    *node_shape = shape.clone();
                }
            }
        });
    }

    #[test]
    fn set_sketch_dispatches() {
        // A scene whose first node is a Sketch (the target for the producer edit).
        let mut scene = Scene::from_nodes(vec![
            sketch_node(box_sketch([2, 2, 2]), MaterialChoice::Stone),
            tool_node(box_shape([3, 1, 4]), MaterialChoice::Wood),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let target = root_id(&scene, 0);
        let producer = box_sketch([7, 5, 3]);
        assert_dispatch_matches(
            &scene,
            Intent::SetSketch { target, producer: producer.clone() },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    if let NodeContent::SketchTool { producer: node_producer, .. } = &mut node.content {
                        *node_producer = producer.clone();
                    }
                }
            },
        );
    }

    #[test]
    fn set_sketch_on_non_sketch_is_noop() {
        // The target is a Tool node, not a Sketch — SetSketch must no-op.
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(
            &mut applied,
            Intent::SetSketch { target, producer: box_sketch([2, 2, 2]) },
        );
        assert_eq!(applied, scene);
        assert_eq!(effect, IntentEffect::none());
    }

    #[test]
    fn set_material_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetMaterial { target, material: MaterialChoice::Plain },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    if let NodeContent::Tool { material, .. } = &mut node.content {
                        *material = MaterialChoice::Plain;
                    }
                }
            },
        );
    }

    #[test]
    fn set_operation_dispatches() {
        // ADR 0017 (#73): SetOperation writes the leaf node's combine operation.
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetOperation {
                target,
                operation: document::scene::CombineOp::Subtract,
            },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    node.operation = document::scene::CombineOp::Subtract;
                }
            },
        );
    }

    #[test]
    fn set_operation_intersect_dispatches() {
        // ADR 0017 (#75): SetOperation carries the Intersect arm through the same
        // field write as Subtract — no structural change, just the third value.
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetOperation {
                target,
                operation: document::scene::CombineOp::Intersect,
            },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    node.operation = document::scene::CombineOp::Intersect;
                }
            },
        );
    }

    #[test]
    fn set_operation_on_group_dispatches() {
        // ADR 0017 Decision 3 (#74): a Group is a sealed composition scope whose
        // composed body folds under the GROUP's own operation, so SetOperation on a
        // Group target must apply (it was a no-op in the #73 sibling-level slice).
        let mut scene = two_tool_scene();
        scene.active = Some(root_id(&scene, 0));
        let group = scene.group_active().expect("grouping the active node succeeds");
        assert_dispatch_matches(
            &scene,
            Intent::SetOperation {
                target: group,
                operation: document::scene::CombineOp::Subtract,
            },
            |s| {
                if let Some(node) = s.node_by_id_mut(group) {
                    node.operation = document::scene::CombineOp::Subtract;
                }
            },
        );
    }

    #[test]
    fn set_operation_on_instance_dispatches() {
        // ADR 0017 / issue #76: an Instance folds the referenced definition's
        // finished body under the INSTANCE's own operation — a definition instanced
        // with Subtract is the reusable cutter — so SetOperation on an Instance
        // target must apply (it was a deliberate no-op until this slice).
        let mut scene = two_tool_scene();
        scene.active = Some(root_id(&scene, 0));
        scene
            .make_definition_from_active("Part def")
            .expect("definition from the active node succeeds");
        let instance = root_id(&scene, 0); // the active node became the Instance.
        assert_dispatch_matches(
            &scene,
            Intent::SetOperation {
                target: instance,
                operation: document::scene::CombineOp::Subtract,
            },
            |s| {
                if let Some(node) = s.node_by_id_mut(instance) {
                    node.operation = document::scene::CombineOp::Subtract;
                }
            },
        );
    }

    #[test]
    fn set_definition_fixture_dispatches() {
        // ADR 0017 Decision 4 (#77): SetDefinitionFixture is a DEFINITION field
        // write — the flag lives on the AssemblyDef (being a fixture is what the
        // part IS), so the dispatch mutates the definition, not any node.
        let mut scene = two_tool_scene();
        scene.active = Some(root_id(&scene, 0));
        let def = scene
            .make_definition_from_active("Window")
            .expect("definition from the active node succeeds");
        assert_dispatch_matches(
            &scene,
            Intent::SetDefinitionFixture { def, fixture: true },
            |s| {
                s.set_definition_fixture(def, true);
            },
        );
    }

    #[test]
    fn set_offset_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        assert_dispatch_matches(
            &scene,
            Intent::SetOffset { target, offset_measurements: whole_block_offset([3, -2, 5]) },
            |s| {
                // `apply` derives canonical voxels from the per-axis measurement at
                // the document density (ADR 0003 §3f(0)); mirror that here.
                let density = s.voxels_per_block;
                if let Some(node) = s.node_by_id_mut(target) {
                    node.transform =
                        NodeTransform::from_measurements(whole_block_offset([3, -2, 5]), density);
                }
            },
        );
    }

    #[test]
    fn set_name_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetName { target, name: "Renamed".to_string() },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    node.name = "Renamed".to_string();
                }
            },
        );
    }

    #[test]
    fn set_cloud_seed_dispatches() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(&scene, Intent::SetCloudSeed { target, seed: 42 }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                if let NodeContent::Part(Part::DebugClouds { seed }) = &mut node.content {
                    *seed = 42;
                }
            }
        });
    }

    #[test]
    fn set_node_grids_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: true,
            floor_grid: false,
        };
        assert_dispatch_matches(&scene, Intent::SetNodeGrids { target, grids }, |s| {
            if let Some(node) = s.node_by_id_mut(target) {
                node.grids = grids;
            }
        });
    }

    #[test]
    fn set_show_child_booleans_dispatches() {
        // Issue #79: a plain per-node field write.
        let scene = two_tool_scene();
        let target = root_id(&scene, 0);
        assert_dispatch_matches(
            &scene,
            Intent::SetShowChildBooleans { target, show: true },
            |s| {
                if let Some(node) = s.node_by_id_mut(target) {
                    node.show_child_booleans = true;
                }
            },
        );
    }

    /// Issue #79 acceptance: toggling the child-boolean checkbox re-derives the ghost
    /// overlay ONLY — its effect is `operand_ghosts()`, never `scene_changed`, so no
    /// whole-scene re-resolve fires at the shell's rebuild seam.
    #[test]
    fn set_show_child_booleans_effect_is_ghosts_only_never_a_re_resolve() {
        let mut scene = two_tool_scene();
        let target = root_id(&scene, 0);
        let mut core = test_core();
        let effect = core.apply_intent(&mut scene, Intent::SetShowChildBooleans { target, show: true });
        assert_eq!(effect, IntentEffect::operand_ghosts());
        assert!(!effect.scene_changed, "the display toggle must not re-resolve the scene");
        // A write to a missing id stays a no-op (the field-write law).
        let effect =
            core.apply_intent(&mut scene, Intent::SetShowChildBooleans { target: NodeId(9999), show: true });
        assert_eq!(effect, IntentEffect::none());
    }

    #[test]
    fn field_write_to_missing_id_is_noop() {
        let scene = two_tool_scene();
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(
            &mut applied,
            Intent::SetName { target: NodeId(9999), name: "ghost".to_string() },
        );
        assert_eq!(applied, scene);
        assert_eq!(effect, IntentEffect::none());
    }

    #[test]
    fn set_shape_on_non_tool_is_noop() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        let target = root_id(&scene, 0);
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect =
            core.apply_intent(&mut applied, Intent::SetShape { target, shape: box_shape([2, 2, 2]) });
        assert_eq!(applied, scene);
        assert_eq!(effect, IntentEffect::none());
    }

    // === Global ===

    #[test]
    fn set_density_sets_document_field() {
        // Density is a single document-level field (ADR 0003 §3f(0)): the dispatch sets
        // `scene.voxels_per_block`, not a per-Tool fan-out.
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
            NodeBuilder::group(
                "G",
                vec![tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into()],
            ),
            NodeSpec::CloudsPart.into_node().into(),
        ]);
        scene.ensure_node_ids();
        assert_dispatch_matches(&scene, Intent::SetDensity { voxels_per_block: 20 }, |s| {
            // `apply` rescales every node's voxel offset old→new density to preserve
            // block placement, AND re-targets every Tool's voxel-granular size (ADR
            // 0003 §3f(0)); mirror both here. (Every node has a zero offset, so the
            // offset rescale is a no-op; the sizes DO re-target — a retained whole-block
            // size re-evaluates at the new density.)
            let old_density = s.voxels_per_block.max(1) as i64;
            for node in s.arena.values_mut() {
                for axis in 0..3 {
                    node.transform.offset_voxels[axis] =
                        node.transform.offset_voxels[axis] * 20 / old_density;
                }
                if let NodeContent::Tool { shape, .. } = &mut node.content {
                    *shape = document::voxel::SdfShape::from_measurements(
                        shape.kind,
                        shape.size_measurements(),
                        shape.wall_blocks,
                        20,
                    );
                }
            }
            s.voxels_per_block = 20;
        });
    }

    #[test]
    fn set_grid_masters_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
            |s| {
                s.master_voxel_grid = false;
                s.master_block_lattice = true;
                s.master_floor_grid = false;
            },
        );
    }

    // === Selection ===

    #[test]
    fn select_node_dispatches() {
        let scene = two_tool_scene();
        let target = root_id(&scene, 1);
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(&mut applied, Intent::SelectNode { target: Some(target) });
        let mut expected = scene.clone();
        expected.active = Some(target);
        assert_eq!(applied, expected);
        assert_eq!(effect, IntentEffect::selection());
    }

    #[test]
    fn select_point_dispatches() {
        let scene = two_tool_scene();
        let mut core = test_core();
        let mut applied = scene.clone();
        let effect = core.apply_intent(&mut applied, Intent::SelectPoint { target: Some(0) });
        let mut expected = scene.clone();
        expected.active_point = Some(0);
        assert_eq!(applied, expected);
        assert_eq!(effect, IntentEffect::selection());
    }

    // === Points ===

    #[test]
    fn add_point_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::AddPoint { position_blocks: [4, 0, -3], name: "Anchor".to_string() },
            |s| {
                let point = Point {
                    name: "Anchor".to_string(),
                    position_blocks: [4, 0, -3],
                    ..Point::default()
                };
                s.add_point(point);
            },
        );
    }

    #[test]
    fn remove_point_dispatches() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [1, 2, 3],
            ..Point::default()
        });
        assert_dispatch_matches(&scene, Intent::RemovePoint { index: 1 }, |s| {
            s.remove_point(1);
        });
    }

    #[test]
    fn set_point_hidden_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointHidden { index: 0, hidden: true },
            |s| {
                s.points[0].hidden = true;
            },
        );
    }

    #[test]
    fn set_point_planes_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointPlanes { index: 0, xz: false, xy: true, yz: true },
            |s| {
                s.points[0].plane_xz = false;
                s.points[0].plane_xy = true;
                s.points[0].plane_yz = true;
            },
        );
    }

    #[test]
    fn set_point_axes_dispatches() {
        let scene = two_tool_scene();
        assert_dispatch_matches(
            &scene,
            Intent::SetPointAxes { index: 0, x: false, y: true, z: false },
            |s| {
                s.points[0].axis_x = false;
                s.points[0].axis_y = true;
                s.points[0].axis_z = false;
            },
        );
    }

    #[test]
    fn set_point_position_dispatches() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [0, 0, 0],
            ..Point::default()
        });
        assert_dispatch_matches(
            &scene,
            Intent::SetPointPosition { index: 1, position_blocks: [9, -1, 2] },
            |s| {
                s.points[1].position_blocks = [9, -1, 2];
            },
        );
    }

    // === serde round-trip: every variant serializes → deserializes to itself ===

    #[test]
    fn every_intent_variant_round_trips_through_json() {
        let shape = box_shape([2, 3, 4]);
        let grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let variants = vec![
            Intent::AddNode {
                content: NodeSpec::Tool { shape: shape.clone(), material: MaterialChoice::Wood },
            },
            Intent::AddNode { content: NodeSpec::CloudsPart },
            Intent::AddNode {
                content: NodeSpec::Sketch {
                    producer: box_sketch([2, 3, 4]),
                    material: MaterialChoice::Stone,
                },
            },
            Intent::AddChild {
                group: NodeId(7),
                content: NodeSpec::Tool { shape: shape.clone(), material: MaterialChoice::Plain },
            },
            Intent::GroupNode { target: NodeId(3) },
            Intent::MakeDefinition { target: NodeId(3), name: "House".to_string() },
            Intent::AddInstance { def: DefId(2) },
            Intent::RemoveNode { target: NodeId(5) },
            Intent::SetVisible { target: NodeId(1), visible: false },
            Intent::SetShape { target: NodeId(1), shape },
            Intent::SetSketch { target: NodeId(1), producer: box_sketch([2, 3, 4]) },
            Intent::SetMaterial { target: NodeId(1), material: MaterialChoice::Stone },
            Intent::SetOffset { target: NodeId(1), offset_measurements: whole_block_offset([-1, 2, -3]) },
            Intent::SetName { target: NodeId(1), name: "Foo".to_string() },
            Intent::SetCloudSeed { target: NodeId(1), seed: 9 },
            Intent::SetNodeGrids { target: NodeId(1), grids },
            Intent::SetShowChildBooleans { target: NodeId(1), show: true },
            Intent::SetDensity { voxels_per_block: 16 },
            Intent::SetGridMasters { voxel: true, lattice: false, floor: true },
            Intent::SelectNode { target: Some(NodeId(4)) },
            Intent::SelectNode { target: None },
            Intent::SelectPoint { target: Some(2) },
            Intent::SelectPoint { target: None },
            Intent::AddPoint { position_blocks: [1, 2, 3], name: "P".to_string() },
            Intent::RemovePoint { index: 1 },
            Intent::SetPointHidden { index: 0, hidden: true },
            Intent::SetPointPlanes { index: 0, xz: true, xy: false, yz: true },
            Intent::SetPointAxes { index: 0, x: true, y: false, z: true },
            Intent::SetPointPosition { index: 0, position_blocks: [4, 5, 6] },
        ];
        for intent in variants {
            let json = serde_json::to_string(&intent).expect("serialize");
            let back: Intent = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(intent, back, "round-trip mismatch for {intent:?}");
        }
    }

    #[test]
    fn node_spec_into_node_matches_panel_tool_naming() {
        // A Tool spec yields a node named after its kind (the panel's chip label),
        // wrapping the same shape + material.
        let shape = box_shape([2, 2, 2]);
        let node = NodeSpec::Tool { shape: shape.clone(), material: MaterialChoice::Wood }.into_node();
        assert_eq!(node.name, "Box");
        assert_eq!(node.transform, NodeTransform::identity());
        match node.content {
            NodeContent::Tool { shape: s, material } => {
                assert_eq!(s, shape);
                assert_eq!(material, MaterialChoice::Wood);
            }
            other => panic!("expected Tool, got {other:?}"),
        }
    }

    #[test]
    fn node_spec_sketch_into_node() {
        // A Sketch spec yields a node named "Sketch" at the identity transform,
        // wrapping the SAME producer + material.
        let producer = box_sketch([3, 2, 4]);
        let node =
            NodeSpec::Sketch { producer: producer.clone(), material: MaterialChoice::Wood }.into_node();
        assert_eq!(node.name, "Sketch");
        assert_eq!(node.transform, NodeTransform::identity());
        match node.content {
            NodeContent::SketchTool { producer: p, material } => {
                assert_eq!(p, producer);
                assert_eq!(material, MaterialChoice::Wood);
            }
            other => panic!("expected SketchTool, got {other:?}"),
        }
    }

    /// LOAD-BEARING equivalence: a default rectangle-footprint sketch
    /// (`SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, w, d), h)`) resolves to
    /// EXACTLY the same occupied voxel set as the matching `SdfShape` `Box` of the
    /// same voxel size + density. This locks "a default sketch == a box" so a future
    /// UI default (a freshly-added sketch) can never silently drift from the box it
    /// is meant to mirror. (Mirrors the `rectangle_extrude_equals_box` proof in
    /// sketch.rs, here pinned on the footprint-extrude-up `PlaneAxis::Z` default.)
    #[test]
    fn default_sketch_spec_equals_box() {
        use voxel_core::voxel::{Voxel, VoxelGrid};
        use document::voxel::{VoxelProducer};
        use std::collections::BTreeSet;

        // Resolve a producer to a SET of (world_position_bits, block_local, material)
        // so two producers compare independent of emission order (mirrors sketch.rs's
        // occupancy_set helper; world positions are integer+0.5, so f32 bits are exact).
        fn occupancy_set(
            producer: &dyn VoxelProducer,
            density: u32,
        ) -> BTreeSet<([i32; 3], [u8; 3], u16)> {
            let mut grid = VoxelGrid::default();
            producer.resolve(&mut grid, density);
            grid.occupied
                .iter()
                .map(|voxel: &Voxel| {
                    let position = voxel.world_position();
                    (
                        [
                            (position[0] * 2.0).round() as i32,
                            (position[1] * 2.0).round() as i32,
                            (position[2] * 2.0).round() as i32,
                        ],
                        voxel.block_local_coord,
                        voxel.color_index(),
                    )
                })
                .collect()
        }

        // A block-sized box at the default density 16. PlaneAxis::Z ⇒ in-plane axes
        // X, Y; normal Z — so rectangle(Z, grid_x, grid_y) extruded grid_z matches a
        // Box of [grid_x, grid_y, grid_z] voxels exactly.
        let size_blocks = [3u32, 2, 4];
        let density = 16u32;
        let box_shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let sketch = box_sketch(size_blocks);

        assert_eq!(
            sketch.grid_dimensions(),
            box_shape.grid_dimensions(density),
            "default sketch AABB must match the box AABB"
        );
        assert_eq!(
            occupancy_set(&sketch, density),
            occupancy_set(&box_shape, density),
            "a default rectangle sketch must resolve to exactly the matching Box"
        );
    }
