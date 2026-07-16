    use super::*;
    use camera::OrbitCamera;
    use voxel_core::core_geom::MaterialChoice;
    use document::intent::{whole_block_offset, Intent, IntentEffect, NodeSpec};
    use document::scene::{Node, NodeBuilder, NodeContent, NodeGrids, NodeTransform, Point, Scene};
    use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
    use voxel_core::units::Measurement;
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape};

    /// A headless [`AppCore`] for the undo tests (no GPU — `apply_intent`/`undo`/`redo`
    /// only touch the borrowed scene + the owned command stack).
    fn test_core() -> AppCore {
        AppCore::new(OrbitCamera::default())
    }

    /// A rectangle-footprint sketch→extrude producer of the given BLOCK size at the
    /// default density 16 (`PlaneAxis::Z` = footprint-extrude-up: profile in XY,
    /// extruded along +Z).
    fn box_sketch(size_blocks: [u32; 3]) -> SketchSolid {
        let density = 16u32;
        let grid_x = (size_blocks[0] * density) as i64;
        let grid_y = (size_blocks[1] * density) as i64;
        let grid_z = size_blocks[2] * density;
        SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, grid_x, grid_y), grid_z)
    }

    /// A revolve producer: a rectangle radial×axial profile revolved a full 360° about
    /// an in-plane axis (a cylinder of the given block radius / axial height at d16).
    fn revolve_sketch(radius_blocks: u32, axial_blocks: u32) -> SketchSolid {
        let density = 16u32;
        let radial = (radius_blocks * density) as i64;
        let axial = (axial_blocks * density) as i64;
        SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, radial, axial),
            RevolveAxis::InPlane1,
            360,
        )
    }

    /// A Sketch node named `"Sketch"` (matching [`NodeSpec::into_node`]).
    fn sketch_node(producer: SketchSolid, material: MaterialChoice) -> Node {
        Node::new("Sketch", NodeContent::SketchTool { producer, material })
    }

    /// A box Tool shape of the given BLOCK size, built at the default density 16
    /// (canonical `size_voxels = blocks · 16`). The undo / recenter fixtures key on
    /// structure + offsets, not the exact voxel size, and `two_tool_scene` runs at
    /// the default density 16.
    fn box_shape(size: [u32; 3]) -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, size, 1, 16)
    }

    /// A Tool node named after its kind (matching [`NodeSpec::into_node`]).
    fn tool_node(shape: SdfShape, material: MaterialChoice) -> Node {
        Node::new(format!("{:?}", shape.kind), NodeContent::Tool { shape, material })
    }

    /// A normalized two-Tool scene with stable ids minted + an Origin point, the first
    /// node active.
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

    /// Apply `intent`, asserting the round-trip invariant: `undo()` restores the
    /// scene byte-for-byte to `before`, and `redo()` restores it byte-for-byte to the
    /// post-apply `after`. Returns the core so the caller can inspect the stacks.
    fn assert_round_trips(scene: &mut Scene, intent: Intent) {
        let mut core = test_core();
        let before = scene.clone();
        core.apply_intent(scene, intent);
        let after = scene.clone();
        assert_ne!(*scene, before, "the forward op must change the scene");
        assert_eq!(core.undo_depth(), 1, "one command pushed");

        core.undo(scene);
        assert_eq!(*scene, before, "undo must restore the scene byte-for-byte");
        assert_eq!(core.undo_depth(), 0);
        assert_eq!(core.redo_depth(), 1);

        core.redo(scene);
        assert_eq!(*scene, after, "redo must restore the post-apply scene byte-for-byte");
        assert_eq!(core.undo_depth(), 1);
        assert_eq!(core.redo_depth(), 0);
    }

    // === Structural inverses (the correctness-critical arms) ===

    #[test]
    fn add_node_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::AddNode {
                content: NodeSpec::Tool {
                    shape: box_shape([5, 5, 5]),
                    material: MaterialChoice::Plain,
                },
            },
        );
    }

    #[test]
    fn add_node_sketch_round_trips() {
        // Proves `Inverse::RemoveAdded` (which keys on the add intent KIND, not the
        // NodeSpec payload) covers a Sketch add too.
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::AddNode {
                content: NodeSpec::Sketch {
                    producer: box_sketch([5, 5, 5]),
                    material: MaterialChoice::Plain,
                },
            },
        );
    }

    #[test]
    fn add_child_round_trips() {
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into()],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let group = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::AddChild {
                group,
                content: NodeSpec::Tool {
                    shape: box_shape([4, 4, 4]),
                    material: MaterialChoice::Wood,
                },
            },
        );
    }

    #[test]
    fn group_node_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(&mut scene, Intent::GroupNode { target });
    }

    #[test]
    fn group_node_nested_round_trips() {
        // Group a node that already lives inside a Group — exercises the parent-spine
        // (not roots) slot restore.
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![
                tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
                tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
            ],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        // The second child of the Group.
        let group = scene.roots[0];
        let child = match &scene.arena[&group].content {
            NodeContent::Group(children) => children[1],
            _ => unreachable!(),
        };
        assert_round_trips(&mut scene, Intent::GroupNode { target: child });
    }

    #[test]
    fn make_definition_from_leaf_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::MakeDefinition { target, name: "House".to_string() },
        );
    }

    #[test]
    fn make_definition_from_group_round_trips() {
        // A Group active node DONATES its children to the def — the harder inverse
        // (restore the donated spine into the node's content, pop the def, no body).
        let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
            "G",
            vec![
                tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
                tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
            ],
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::MakeDefinition { target, name: "Body".to_string() },
        );
    }

    #[test]
    fn add_instance_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        scene.active = Some(target);
        let def = scene.make_definition_from_active("Body").expect("def made");
        assert_round_trips(&mut scene, Intent::AddInstance { def });
    }

    #[test]
    fn remove_leaf_node_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(&mut scene, Intent::RemoveNode { target });
    }

    #[test]
    fn remove_group_with_children_round_trips() {
        // The critical case: removing a Group detaches a whole subtree; the inverse
        // must re-insert every descendant under its original id at the original slot.
        let mut scene = Scene::from_nodes(vec![
            tool_node(box_shape([2, 2, 2]), MaterialChoice::Stone).into(),
            NodeBuilder::group(
                "G",
                vec![
                    tool_node(box_shape([3, 3, 3]), MaterialChoice::Wood).into(),
                    NodeBuilder::group(
                        "Inner",
                        vec![tool_node(box_shape([1, 1, 1]), MaterialChoice::Plain).into()],
                    ),
                ],
            ),
            tool_node(box_shape([4, 4, 4]), MaterialChoice::Plain).into(),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let group = scene.roots[1];
        assert_round_trips(&mut scene, Intent::RemoveNode { target: group });
    }

    // === Field-set inverses ===

    #[test]
    fn set_visible_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(&mut scene, Intent::SetVisible { target, visible: false });
    }

    #[test]
    fn set_shape_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetShape { target, shape: box_shape([9, 9, 9]) },
        );
    }

    #[test]
    fn set_material_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetMaterial { target, material: MaterialChoice::Plain },
        );
    }

    #[test]
    fn set_operation_round_trips() {
        // ADR 0017 (#73): flipping a leaf's combine operation to Subtract reverses
        // via the field-inverse pattern — undo restores the prior Union, redo
        // re-applies the Subtract.
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetOperation { target, operation: document::scene::CombineOp::Subtract },
        );
    }

    #[test]
    fn set_operation_on_group_round_trips() {
        // ADR 0017 Decision 3 (#74): a GROUP's operation is meaningful (its composed
        // body folds under it), so the flip must capture the same field inverse —
        // undo restores the group's prior Union, redo re-applies the Subtract.
        let mut scene = two_tool_scene();
        scene.active = Some(scene.roots[0]);
        let group = scene.group_active().expect("grouping the active node succeeds");
        assert_round_trips(
            &mut scene,
            Intent::SetOperation { target: group, operation: document::scene::CombineOp::Subtract },
        );
    }

    #[test]
    fn set_operation_on_instance_round_trips() {
        // ADR 0017 / issue #76: an INSTANCE's operation is meaningful (the referenced
        // definition's finished body folds under it — the reusable cutter), so the
        // flip captures the same field inverse as leaves and Groups.
        let mut scene = two_tool_scene();
        scene.active = Some(scene.roots[0]);
        scene.make_definition_from_active("Body").expect("def made");
        let instance = scene.roots[0]; // the active node became the Instance.
        assert_round_trips(
            &mut scene,
            Intent::SetOperation {
                target: instance,
                operation: document::scene::CombineOp::Subtract,
            },
        );
    }

    #[test]
    fn set_definition_fixture_round_trips() {
        // ADR 0017 Decision 4 (#77): the fixture flag is a DEFINITION field write,
        // so the flip captures a definition-targeted field inverse — undo restores
        // the sealed default, redo re-applies the splice.
        let mut scene = two_tool_scene();
        scene.active = Some(scene.roots[0]);
        let def = scene.make_definition_from_active("Window").expect("def made");
        assert_round_trips(&mut scene, Intent::SetDefinitionFixture { def, fixture: true });
    }

    #[test]
    fn placing_a_subtract_instance_undoes_to_intact_hosts() {
        // Issue #76 acceptance: the reusable-cutter placement gesture — AddInstance
        // then SetOperation(Subtract) on the minted node — undoes cleanly: two undos
        // restore the pre-placement scene byte-for-byte (both hosts intact, the
        // instance gone), and two redos re-apply the carve placement.
        let mut scene = two_tool_scene();
        scene.active = Some(scene.roots[0]);
        let def = scene.make_definition_from_active("Cutter body").expect("def made");
        let mut core = test_core();
        let before = scene.clone();

        core.apply_intent(&mut scene, Intent::AddInstance { def });
        let minted = scene.active.expect("AddInstance selects the minted instance");
        core.apply_intent(
            &mut scene,
            Intent::SetOperation {
                target: minted,
                operation: document::scene::CombineOp::Subtract,
            },
        );
        let after = scene.clone();
        assert_ne!(scene, before, "the placement must change the scene");
        assert_eq!(core.undo_depth(), 2, "two commands pushed");

        core.undo(&mut scene);
        core.undo(&mut scene);
        assert_eq!(
            scene, before,
            "undoing the Subtract-instance placement must restore the hosts intact"
        );

        core.redo(&mut scene);
        core.redo(&mut scene);
        assert_eq!(scene, after, "redo must re-apply the carve placement byte-for-byte");
    }

    #[test]
    fn set_operation_intersect_round_trips() {
        // ADR 0017 (#75): the Intersect arm rides the SAME field-inverse pattern —
        // undo restores the prior Union, redo re-applies the Intersect — on a leaf
        // and on a Group (the scope's composed body folds under Intersect).
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetOperation { target, operation: document::scene::CombineOp::Intersect },
        );
        let mut scene = two_tool_scene();
        scene.active = Some(scene.roots[0]);
        let group = scene.group_active().expect("grouping the active node succeeds");
        assert_round_trips(
            &mut scene,
            Intent::SetOperation { target: group, operation: document::scene::CombineOp::Intersect },
        );
    }

    /// A normalized scene whose first node is a Sketch and whose second is a Tool,
    /// ids minted + Origin point, first node active — the sketch-edit fixture.
    fn sketch_then_tool_scene() -> Scene {
        let mut scene = Scene::from_nodes(vec![
            sketch_node(box_sketch([2, 2, 2]), MaterialChoice::Stone),
            tool_node(box_shape([3, 1, 4]), MaterialChoice::Wood),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        scene
    }

    #[test]
    fn set_sketch_round_trips() {
        // Undo restores the prior producer byte-for-byte; redo re-applies the new one.
        let mut scene = sketch_then_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetSketch { target, producer: box_sketch([9, 7, 3]) },
        );
    }

    #[test]
    fn set_sketch_revolve_round_trips() {
        // A SetSketch carrying a REVOLVE producer round-trips: undo restores the prior
        // producer byte-for-byte, redo re-applies the revolve. Proves the dispatch /
        // capture_inverse path is operation-agnostic (the inspector's revolve rebuild
        // flows through the same SetSketch intent as extrude).
        let mut scene = sketch_then_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetSketch { target, producer: revolve_sketch(4, 6) },
        );
    }

    #[test]
    fn set_material_on_sketch_node() {
        // The shared material edit applies to a SketchTool node, and undo restores the
        // prior material (proves the extended SetMaterial dispatch + capture_inverse
        // arms cover sketch nodes).
        let mut scene = sketch_then_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetMaterial { target, material: MaterialChoice::Plain },
        );
    }

    #[test]
    fn set_offset_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[1];
        assert_round_trips(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: whole_block_offset([3, -2, 5]) },
        );
    }

    /// Applying a `SetOffset` with a blocks+voxels expression derives the canonical
    /// voxel offset at the document density, and the same expression refines
    /// losslessly at a denser document (ADR 0003 §3f(0)). `3.5 blocks` → 56 voxels
    /// at d16, 112 at d32; a signed `-2 blocks 4 voxels` axis derives signed.
    #[test]
    fn set_offset_apply_derives_voxels_at_density() {
        let expression = [
            Measurement::new(voxel_core::units::ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::new(voxel_core::units::ExactRational::from_integer(-2), 4),   // -2 blocks 4 voxels
            Measurement::from_voxels(7),                                          // 7 voxels
        ];

        let mut scene = two_tool_scene();
        scene.voxels_per_block = 16;
        let target = scene.roots[1];
        let mut core = test_core();
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: expression },
        );
        assert_eq!(
            scene.node_by_id(target).unwrap().transform.offset_voxels,
            [56, -28, 7],
            "blocks·d + voxels derived per axis at density 16"
        );

        let mut dense = two_tool_scene();
        dense.voxels_per_block = 32;
        let dense_target = dense.roots[1];
        core.apply_intent(
            &mut dense,
            Intent::SetOffset { target: dense_target, offset_measurements: expression },
        );
        assert_eq!(
            dense.node_by_id(dense_target).unwrap().transform.offset_voxels,
            [112, -60, 7],
            "the SAME expression refines losslessly at density 32"
        );
    }

    /// Undo of a `SetOffset` replays the node's prior RETAINED measurement exactly
    /// — voxel-granular and parametric, not the floored block view (ADR 0003
    /// §3f(0)). A prior `2 blocks 8 voxels` axis is restored verbatim, not flattened
    /// to whole blocks.
    #[test]
    fn set_offset_undo_restores_retained_measurement() {
        let mut scene = two_tool_scene();
        scene.voxels_per_block = 16;
        let target = scene.roots[1];
        let mut core = test_core();

        let first = [
            Measurement::new(voxel_core::units::ExactRational::from_integer(2), 8), // 2 blocks 8 voxels
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: first },
        );
        assert_eq!(scene.node_by_id(target).unwrap().transform.offset_voxels[0], 40);

        // A second SetOffset, then undo it → the FIRST expression is restored.
        let second = whole_block_offset([5, 0, 0]);
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: second },
        );
        assert_eq!(scene.node_by_id(target).unwrap().transform.offset_voxels[0], 80);

        core.undo(&mut scene);
        let restored = scene.node_by_id(target).unwrap().transform.offset_measurements();
        assert_eq!(
            restored, first,
            "undo restored the exact authored expression (2 blocks 8 voxels), not a block-floored view"
        );
        assert_eq!(scene.node_by_id(target).unwrap().transform.offset_voxels[0], 40);
    }

    #[test]
    fn set_name_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetName { target, name: "Renamed".to_string() },
        );
    }

    #[test]
    fn set_cloud_seed_round_trips() {
        let mut scene = Scene::from_nodes(vec![NodeSpec::CloudsPart.into_node()]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let target = scene.roots[0];
        assert_round_trips(&mut scene, Intent::SetCloudSeed { target, seed: 42 });
    }

    #[test]
    fn set_node_grids_round_trips() {
        let mut scene = two_tool_scene();
        let target = scene.roots[0];
        assert_round_trips(
            &mut scene,
            Intent::SetNodeGrids {
                target,
                grids: NodeGrids {
                    voxel_grid_on_faces: true,
                    block_lattice: true,
                    floor_grid: false,
                },
            },
        );
    }

    #[test]
    fn set_density_round_trips() {
        // Density is a single document-level field now (ADR 0003 §3f(0)); start from a
        // non-default prior so the inverse must restore the exact prior value, not 16.
        // Size is now voxel-granular and SetDensity RE-TARGETS each Tool's size at the
        // new density (ADR 0003 §3f(0)), so the fixture's shapes must be built at the
        // SAME density the scene runs at (5) — a `2 blocks` shape is 10 voxels at d5,
        // not the d16 default's 32 — otherwise the density round-trip would normalise
        // the inconsistency and undo could not restore it byte-for-byte.
        let mut scene = Scene::from_nodes(vec![
            tool_node(SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 5), MaterialChoice::Stone)
                .into(),
            NodeBuilder::group(
                "G",
                vec![tool_node(
                    SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, 5),
                    MaterialChoice::Wood,
                )
                .into()],
            ),
        ]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.voxels_per_block = 5;
        scene.active = scene.roots.first().copied();
        assert_round_trips(&mut scene, Intent::SetDensity { voxels_per_block: 20 });
    }

    /// A density change must PRESERVE each node's block placement (ADR 0003 §3f(0)):
    /// the casual density control is fineness-only, so a node at block 5 stays at
    /// block 5 — its canonical voxel offset rescales old→new density. (The explicit
    /// destructive game-re-target is a separate future op.) Undo rescales back exactly
    /// for block-multiple offsets.
    #[test]
    fn set_density_preserves_block_position() {
        let mut node = tool_node(box_shape([1, 1, 1]), MaterialChoice::Stone);
        node.transform = NodeTransform::from_blocks([5, 0, 0], 8); // block 5 @ d=8 → 40 voxels
        let mut scene = Scene::single_node(node);
        scene.voxels_per_block = 8;
        let node_id = scene.roots[0];

        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SetDensity { voxels_per_block: 16 });

        let after = scene.node_by_id(node_id).expect("node survives");
        assert_eq!(after.transform.blocks(16), [5, 0, 0], "block 5 preserved across d 8→16");
        assert_eq!(after.transform.offset_voxels, [80, 0, 0], "5 blocks @ d=16 = 80 voxels");
        assert!(after.transform.block_aligned(16), "still on the mating lattice");

        core.undo(&mut scene);
        let restored = scene.node_by_id(node_id).expect("node survives undo");
        assert_eq!(restored.transform.blocks(8), [5, 0, 0], "block 5 preserved on undo");
        assert_eq!(restored.transform.offset_voxels, [40, 0, 0], "back to 40 voxels @ d=8");
    }

    /// A `SetOffset` undo across an interleaved density change still restores the
    /// node's prior placement: the inverse captures the prior RETAINED measurement
    /// (`5 blocks`), which re-evaluates at the new density to the same block 5, so
    /// the density between apply and undo does not corrupt it (ADR 0003 §3f(0)).
    #[test]
    fn set_offset_undo_across_density_change() {
        let mut node = tool_node(box_shape([1, 1, 1]), MaterialChoice::Stone);
        node.transform = NodeTransform::from_blocks([5, 0, 0], 8);
        let mut scene = Scene::single_node(node);
        scene.voxels_per_block = 8;
        let node_id = scene.roots[0];

        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SetDensity { voxels_per_block: 16 });
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target: node_id, offset_measurements: whole_block_offset([3, 0, 0]) },
        );
        assert_eq!(
            scene.node_by_id(node_id).unwrap().transform.blocks(16),
            [3, 0, 0],
            "SetOffset moved the node to block 3 at the current density"
        );

        // Undo only the SetOffset → back to the pre-offset block placement (block 5).
        core.undo(&mut scene);
        assert_eq!(
            scene.node_by_id(node_id).unwrap().transform.blocks(16),
            [5, 0, 0],
            "undo restores the prior block placement across the density change"
        );
    }

    /// `SetDensity` RE-EVALUATES a node's RETAINED expression at the new density
    /// (the seam fix): `3 blocks 8 voxels` (56 vx at d16) becomes 3*32 + 8 = 104 at
    /// d32 — the voxel term stays exact, NOT the legacy integer rescale's 112 — and
    /// the retained measurement and canonical voxels stay consistent.
    #[test]
    fn set_density_re_evaluates_retained_measurement_exactly() {
        let mut scene = two_tool_scene();
        scene.voxels_per_block = 16;
        let target = scene.roots[1];
        let mut core = test_core();
        let expression = [
            Measurement::new(voxel_core::units::ExactRational::from_integer(3), 8), // 3 blocks 8 voxels
            Measurement::from_voxels(0),
            Measurement::from_voxels(0),
        ];
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: expression },
        );
        assert_eq!(scene.node_by_id(target).unwrap().transform.offset_voxels[0], 56);

        core.apply_intent(&mut scene, Intent::SetDensity { voxels_per_block: 32 });
        let transform = &scene.node_by_id(target).unwrap().transform;
        assert_eq!(
            transform.offset_voxels[0], 104,
            "voxel term exact across density re-target (3*32 + 8), NOT the rescale 112"
        );
        assert_eq!(
            transform.offset_measurements()[0],
            expression[0],
            "the authored expression is preserved across the re-target"
        );
    }

    /// `SetDensity` on a node with NO retained measurement (a `None` transform, the
    /// legacy/drag path) KEEPS the integer rescale, preserving the physical block
    /// position, and leaves the field `None` (existing behavior untouched).
    #[test]
    fn set_density_integer_rescales_non_retained_offset() {
        let mut node = tool_node(box_shape([1, 1, 1]), MaterialChoice::Stone);
        // A hand-set sub-block voxel offset with NO authored expression: start from
        // the identity (retained field None) and set only the canonical voxels.
        node.transform = NodeTransform::identity();
        node.transform.offset_voxels = [40, 0, 0];
        assert!(!node.transform.has_retained_measurements());
        let mut scene = Scene::single_node(node);
        scene.voxels_per_block = 16;
        let node_id = scene.roots[0];

        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SetDensity { voxels_per_block: 32 });
        let transform = &scene.node_by_id(node_id).unwrap().transform;
        assert_eq!(
            transform.offset_voxels[0], 80,
            "non-retained offset integer-rescales (40 * 32 / 16 = 80), preserving position"
        );
        assert!(
            !transform.has_retained_measurements(),
            "the legacy rescale leaves the retained field None"
        );
    }

    #[test]
    fn set_grid_masters_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
        );
    }

    // === Point inverses ===

    #[test]
    fn add_point_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::AddPoint { position_blocks: [4, 0, -3], name: "Anchor".to_string() },
        );
    }

    #[test]
    fn remove_point_round_trips() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [1, 2, 3],
            ..Point::default()
        });
        assert_round_trips(&mut scene, Intent::RemovePoint { index: 1 });
    }

    #[test]
    fn set_point_hidden_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(&mut scene, Intent::SetPointHidden { index: 0, hidden: true });
    }

    #[test]
    fn set_point_planes_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetPointPlanes { index: 0, xz: false, xy: true, yz: true },
        );
    }

    #[test]
    fn set_point_axes_round_trips() {
        let mut scene = two_tool_scene();
        assert_round_trips(
            &mut scene,
            Intent::SetPointAxes { index: 0, x: false, y: true, z: false },
        );
    }

    #[test]
    fn set_point_position_round_trips() {
        let mut scene = two_tool_scene();
        scene.add_point(Point {
            name: "P".to_string(),
            position_blocks: [0, 0, 0],
            ..Point::default()
        });
        assert_round_trips(
            &mut scene,
            Intent::SetPointPosition { index: 1, position_blocks: [9, -1, 2] },
        );
    }

    // === Selection intents push NOTHING ===

    #[test]
    fn select_node_pushes_no_command() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[1];
        core.apply_intent(&mut scene, Intent::SelectNode { target: Some(target) });
        assert_eq!(core.undo_depth(), 0, "selection is not an undoable step");
        assert_eq!(scene.active, Some(target));
    }

    #[test]
    fn select_point_pushes_no_command() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SelectPoint { target: Some(0) });
        assert_eq!(core.undo_depth(), 0, "point selection is not an undoable step");
        assert_eq!(scene.active_point, Some(0));
    }

    // === No-op forward → no-op inverse (still pushes a command, undo restores nothing) ===

    #[test]
    fn field_write_to_missing_id_undo_is_noop() {
        let mut scene = two_tool_scene();
        let before = scene.clone();
        let mut core = test_core();
        core.apply_intent(
            &mut scene,
            Intent::SetName { target: document::scene::NodeId(9999), name: "ghost".to_string() },
        );
        assert_eq!(scene, before, "a no-op forward changes nothing");
        core.undo(&mut scene);
        assert_eq!(scene, before, "undo of a no-op restores nothing");
    }

    // === Scripted realistic sequence ===

    #[test]
    fn scripted_sequence_undo_redo_round_trips() {
        let mut scene = two_tool_scene();
        let seed = scene.clone();
        let mut core = test_core();

        // A realistic authoring sequence.
        core.apply_intent(
            &mut scene,
            Intent::AddNode {
                content: NodeSpec::Tool {
                    shape: box_shape([2, 2, 2]),
                    material: MaterialChoice::Plain,
                },
            },
        );
        let added = scene.active.expect("added node selected");
        core.apply_intent(&mut scene, Intent::GroupNode { target: added });
        // The wrapped child is now active; group IT into a definition.
        let active = scene.active.expect("active after group");
        core.apply_intent(
            &mut scene,
            Intent::MakeDefinition { target: active, name: "Kit".to_string() },
        );
        let def = scene.definitions.last().expect("def made").id;
        core.apply_intent(&mut scene, Intent::AddInstance { def });
        let instance = scene.active.expect("instance selected");
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target: instance, offset_measurements: whole_block_offset([7, 0, 0]) },
        );
        core.apply_intent(&mut scene, Intent::RemoveNode { target: instance });

        let final_scene = scene.clone();
        assert_eq!(core.undo_depth(), 6, "six undoable steps");

        // Undo all the way back to the seed.
        for _ in 0..6 {
            core.undo(&mut scene);
        }
        assert_eq!(scene, seed, "undo all the way restores the seed byte-for-byte");

        // Redo all the way back to the final scene.
        for _ in 0..6 {
            core.redo(&mut scene);
        }
        assert_eq!(scene, final_scene, "redo all the way restores the final scene");
    }

    #[test]
    fn redo_cleared_after_apply() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetName { target, name: "First".to_string() });
        core.undo(&mut scene);
        assert_eq!(core.redo_depth(), 1, "undo populated redo");
        // A new, different apply must clear the redo future.
        core.apply_intent(&mut scene, Intent::SetName { target, name: "Second".to_string() });
        assert_eq!(core.redo_depth(), 0, "a fresh edit clears the redo stack");
    }

    // === effect_of routing: undo/redo return the per-intent effect, not blanket-true ===

    #[test]
    fn undo_of_field_edit_reports_scene_not_points() {
        // A trivial rename re-resolves the scene but must NOT force a points rebuild
        // (the per-edit cost ADR 0003 optimizes against at 10k nodes).
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetName { target, name: "Renamed".to_string() });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.scene_changed, "rename re-resolves the scene");
        assert!(!undo_effect.points_changed, "rename does not touch the points overlay");
        assert!(undo_effect.selection_changed, "undo always re-syncs the selection mirror");
        // And it is not the old blanket-true effect.
        assert_ne!(
            undo_effect,
            IntentEffect { scene_changed: true, points_changed: true, selection_changed: true },
            "undo no longer returns blanket-true",
        );
        let redo_effect = core.redo(&mut scene);
        assert!(redo_effect.scene_changed);
        assert!(!redo_effect.points_changed, "redo of a rename does not touch points");
    }

    #[test]
    fn undo_of_shape_edit_reports_scene_not_points() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        let target = scene.roots[0];
        core.apply_intent(&mut scene, Intent::SetShape { target, shape: box_shape([9, 9, 9]) });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.scene_changed);
        assert!(!undo_effect.points_changed);
    }

    #[test]
    fn undo_of_point_edit_reports_points_not_scene() {
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(&mut scene, Intent::SetPointHidden { index: 0, hidden: true });
        let undo_effect = core.undo(&mut scene);
        assert!(undo_effect.points_changed, "a point edit is overlay-only");
        assert!(!undo_effect.scene_changed, "a point edit triggers no voxel re-resolve");
        assert!(undo_effect.selection_changed);
    }

    #[test]
    fn undo_of_grid_masters_does_not_claim_scene_changed() {
        // The forward SetGridMasters effect is `none()` (masters are read live); undo
        // must match — claiming scene_changed would wrongly force a re-resolve.
        let mut scene = two_tool_scene();
        let mut core = test_core();
        core.apply_intent(
            &mut scene,
            Intent::SetGridMasters { voxel: false, lattice: true, floor: false },
        );
        let undo_effect = core.undo(&mut scene);
        assert!(!undo_effect.scene_changed, "grid masters need no re-resolve");
        assert!(!undo_effect.points_changed, "grid masters do not touch points");
        // Selection is still re-synced (undo restores selection_before).
        assert!(undo_effect.selection_changed);
        let redo_effect = core.redo(&mut scene);
        assert!(!redo_effect.scene_changed, "redo of grid masters needs no re-resolve");
    }

    /// Count the on-face-grid-flagged voxels (ADR 0003 §3c `grid_overlay` marker) in a
    /// fresh `rebuild` of `scene` at `density`. `rebuild` routes through the per-chunk
    /// store (the chunk cache), so this exercises the SAME invalidation path the live app
    /// uses — not the always-full `resolve_region`.
    fn rebuild_grid_overlay_count(core: &mut AppCore, scene: &Scene, density: u32) -> usize {
        match core.rebuild(scene, density) {
            RebuildOutcome::Built(output) => {
                // ADR 0011 G5: `rebuild` no longer returns a dense grid. Expand the resident
                // two-layer chunks it DID return (the cache's output, so this still exercises
                // the S3 invalidation path) through the test-oracle expander, then count the
                // flagged voxels — the property under test is unchanged.
                let grid = evaluation::two_layer_store::expand_resident_chunks_into_grid(
                    &output.two_layer_chunks,
                    output.region_dimensions,
                    output.recentre_voxels,
                    density,
                );
                grid.occupied.iter().filter(|voxel| voxel.grid_overlay).count()
            }
            RebuildOutcome::DensityRejected { .. } => {
                panic!("density {density} unexpectedly rejected")
            }
        }
    }

    /// Read the recentre shift a single `rebuild` of `scene` at `density` reports.
    fn rebuild_recentre_shift(core: &mut AppCore, scene: &Scene, density: u32) -> [i64; 3] {
        match core.rebuild(scene, density) {
            RebuildOutcome::Built(output) => output.recentre_shift_voxels,
            RebuildOutcome::DensityRejected { .. } => {
                panic!("density {density} unexpectedly rejected")
            }
        }
    }

    /// The camera-stability wiring (the windowed re-frame bug): `rebuild` must report
    /// the floating-origin SHIFT so the shell can compensate `camera.target` and keep
    /// the view put across an edit. The first build shifts nothing; an offset that
    /// moves the composite extent shifts the recentre by exactly the change in
    /// `recentre_voxels_for_resolve` — the delta the camera subtracts.
    #[test]
    fn rebuild_reports_recentre_shift_across_extent_change() {
        let density = 8;
        let mut scene = two_tool_scene();
        let mut core = test_core();

        // First rebuild: no previous recentre, so the shift is zero (the camera is
        // framed explicitly at startup, never compensated on the first build).
        let first_shift = rebuild_recentre_shift(&mut core, &scene, density);
        assert_eq!(first_shift, [0; 3], "the first rebuild must not move the camera");

        // Move a node so the composite extent (hence its recentre) shifts.
        let recentre_before = scene.recentre_voxels_for_resolve(density).voxels();
        let target = scene.roots[0];
        core.apply_intent(
            &mut scene,
            Intent::SetOffset { target, offset_measurements: whole_block_offset([10, -4, 6]) },
        );
        let recentre_after = scene.recentre_voxels_for_resolve(density).voxels();
        let expected_shift = [
            recentre_after[0] - recentre_before[0],
            recentre_after[1] - recentre_before[1],
            recentre_after[2] - recentre_before[2],
        ];
        assert_ne!(expected_shift, [0; 3], "the offset must actually move the origin");

        let reported_shift = rebuild_recentre_shift(&mut core, &scene, density);
        assert_eq!(
            reported_shift, expected_shift,
            "rebuild must report the exact recentre delta the camera compensates",
        );

        // A re-resolve with no further extent change reports zero — a no-op edit (or a
        // pure selection change) must not nudge the view.
        let steady_shift = rebuild_recentre_shift(&mut core, &scene, density);
        assert_eq!(steady_shift, [0; 3], "an unchanged extent must not move the camera");
    }

    /// ADR 0011 G5 startup door (the OOM-hang regression guard): the startup door builds NO
    /// `VoxelGrid` at all — it returns only the region dimensions + resolve recentre. The
    /// persisted 8000×800×800 scene can therefore no longer build a dense ~5.1-billion-cell
    /// grid at startup, on EITHER binary (the door is `gpu`-feature-agnostic). The dims match
    /// the placed region and the recentre matches the resolve frame the camera + fog consume.
    #[test]
    fn startup_region_returns_dims_and_recentre_no_grid() {
        let density = 16u32;
        let scene = default_replay_seed_scene();
        assert!(scene.has_chunkable_extent(density), "the seed scene is chunkable");
        let (dimensions, recentre) = AppCore::startup_region(&scene, density);
        assert_eq!(
            dimensions,
            scene.placed_region_dimensions(density),
            "startup dimensions must match the placed region"
        );
        assert_eq!(
            recentre,
            scene.recentre_voxels_for_resolve(density).voxels(),
            "startup recentre must match the resolve frame (the camera consumes it)"
        );
    }

    /// ADR 0011 G5 retirement assertion (load-bearing): a rebuild yields ONLY the sparse
    /// two-layer covering chunks + scalar metadata — there is NO dense `VoxelGrid` in the
    /// output type at all (the field is gone, compile-enforced). This pins the retirement at
    /// runtime: even the multi-producer scene that streamed a whole-region grid before G5 now
    /// produces the sparse set the mesher + brick sink consume, and the region dimensions
    /// still match the scene's placed region (the camera / scrubber consumer contract).
    #[test]
    fn rebuild_yields_sparse_two_layer_output_no_dense_grid() {
        let density = 16u32;
        let scene = two_tool_scene();
        assert!(scene.has_chunkable_extent(density), "the two-tool fixture is chunkable");
        let mut core = test_core();
        let RebuildOutcome::Built(output) = core.rebuild(&scene, density) else {
            panic!("density {density} unexpectedly rejected");
        };
        // The sole display truth is the sparse resident set — a chunkable scene always covers
        // at least one chunk. (The absence of a dense grid is enforced by `RebuildOutput`'s
        // shape; this asserts the surviving sparse output is well-formed.)
        assert!(
            !output.two_layer_chunks.is_empty(),
            "a chunkable rebuild must return its sparse covering chunks"
        );
        assert_eq!(
            output.region_dimensions,
            scene.placed_region_dimensions(density),
            "the region dimensions must match the placed region (camera / scrubber contract)"
        );
    }

    /// The occupied-voxel CORNER bounding box of a single `shape` of `size_blocks` at
    /// offset `[0, 0, 0]`, resolved at `density` through **`AppCore::rebuild`** — the
    /// per-chunk store path the WINDOWED APP actually renders. Returns
    /// `(min_corner, max_corner)` per axis in absolute voxel units (the half-open box
    /// `[min, max)`; voxel centres sit at `n + 0.5`, so the corner is `floor(centre)`
    /// for the min and `floor(centre) + 1` for the max).
    fn rebuild_frame_corner_bbox(
        shape: SdfShape,
        density: u32,
    ) -> ([i64; 3], [i64; 3]) {
        let mut scene = Scene::from_nodes(vec![tool_node(shape, MaterialChoice::Stone)]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.voxels_per_block = density;
        scene.active = scene.roots.first().copied();
        let mut core = test_core();
        let RebuildOutcome::Built(output) = core.rebuild(&scene, density) else {
            panic!("density {density} unexpectedly rejected");
        };
        // ADR 0011 G5: `rebuild` returns no dense grid. Expand its OWN resident two-layer
        // chunks (the exact windowed-app path) through the test-oracle expander — bit-identical
        // to the retired rebuild grid, so the pinned render-frame coordinates are unchanged.
        let grid = evaluation::two_layer_store::expand_resident_chunks_into_grid(
            &output.two_layer_chunks,
            output.region_dimensions,
            output.recentre_voxels,
            density,
        );
        assert!(!grid.occupied.is_empty(), "shape resolved empty");
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            for axis in 0..3 {
                let corner = position[axis].floor() as i64;
                min[axis] = min[axis].min(corner);
                max[axis] = max[axis].max(corner + 1); // half-open upper bound
            }
        }
        (min, max)
    }

    /// PERMANENT GUARD (corrects the coordinator's mistaken premise). A shape placed
    /// at world offset `[0, 0, 0]` is rendered CENTRED ON THE WORLD ORIGIN through
    /// the `AppCore::rebuild` / per-chunk store path — the exact path the windowed app
    /// renders. This pins the EMPIRICAL render-frame coordinates so the convention can
    /// never be misdescribed again.
    ///
    /// The per-chunk store applies the composite recentre (`Store::bind_region`
    /// rebases every chunk to the composite's recentre / floating origin), so the
    /// rebuild grid is in the SAME centred frame as the monolithic `resolve_region`
    /// (bit-identical for a near scene — proven by the goldens). The #30 lattice shift
    /// snaps the producer grid onto the block lattice in the PRODUCER-TRUE
    /// (pre-recentre) frame, but the recentre then re-symmetrises the composite about
    /// the origin — so the shape the user sees is centred, NOT corner-at-origin.
    ///
    /// Measured coordinates (this test pins them):
    ///   * 1×1×1 box  @ d16 → `[−8, 8)`  per axis  (d8 → `[−4, 4)`)  — centred, NOT `[0, 16)`.
    ///   * 5×5×5 sphere @ d16 → `[−40, 40)` per axis (d8 → `[−20, 20)`).
    ///   * 5×1×5 box  @ d16 → X/Z `[−40, 40)`, Y `[−8, 8)` (d8 → `[−20, 20)`, `[−4, 4)`).
    ///
    /// We assert the CORNER bbox is symmetric (`min + max == 0`): an odd voxel span
    /// (`size·d` is even here, so the span is even-in-voxels) makes the corner bbox
    /// exactly symmetric, with a voxel BOUNDARY on the origin.
    #[test]
    fn shapes_render_centered_on_origin_in_rebuild_frame() {
        use voxel_core::voxel::ShapeKind;
        let cases: [(ShapeKind, [u32; 3]); 3] = [
            (ShapeKind::Box, [1, 1, 1]),
            (ShapeKind::Sphere, [5, 5, 5]),
            (ShapeKind::Box, [5, 1, 5]),
        ];
        for density in [8u32, 16] {
            for (kind, size) in cases {
                let shape = SdfShape::from_blocks(kind, size, 1, density);
                let (min, max) = rebuild_frame_corner_bbox(shape, density);
                for axis in 0..3 {
                    // Centred: the half-open corner box is symmetric about 0.
                    assert_eq!(
                        min[axis] + max[axis],
                        0,
                        "{kind:?} {size:?}@d{density} axis {axis}: rebuild-frame corner bbox \
                         [{}, {}) must be centred on the origin (min + max == 0)",
                        min[axis], max[axis]
                    );
                    // …and spans exactly size·d voxels (no clipping / no half-block leak).
                    assert_eq!(
                        max[axis] - min[axis],
                        (size[axis] * density) as i64,
                        "{kind:?} {size:?}@d{density} axis {axis}: span must be size·d voxels"
                    );
                }
            }
        }
        // Pin the exact 1×1×1 @ d16 box so the convention is unambiguous: it occupies
        // [−8, 8) per axis (centred), NOT [0, 16) (corner-at-origin).
        let one_block = SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16);
        let (min, max) = rebuild_frame_corner_bbox(one_block, 16);
        assert_eq!(min, [-8, -8, -8], "1×1×1 box @ d16 min corner is centred at −8, not 0");
        assert_eq!(max, [8, 8, 8], "1×1×1 box @ d16 max corner is centred at +8, not 16");
    }

    /// Regression (FIX 1): toggling ONLY `voxel_grid_on_faces` must make the on-face
    /// grid appear on the FIRST rebuild — no unrelated edit needed to evict the
    /// stale cached chunks.
    ///
    /// The flag is baked into the resolved voxels as `GRID_OVERLAY_BIT`, but it had
    /// been OMITTED from the leaf content fingerprint. So a lone toggle produced an
    /// identical fingerprint → `edit_aabb_since` found nothing dirty → `rebuild`
    /// evicted no chunks → the cached (grid-less) chunks were reused, and the grid
    /// only "caught up" when a later move/resize/etc. happened to evict them. Folding
    /// the flag into the fingerprint dirties the leaf's AABB on the toggle itself.
    #[test]
    fn toggling_voxel_grid_on_faces_appears_on_first_rebuild() {
        let mut scene = Scene::from_nodes(vec![tool_node(box_shape([3, 3, 3]), MaterialChoice::Stone)]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        scene.active = scene.roots.first().copied();
        let target = scene.roots[0];
        let density = 8;

        let mut core = test_core();

        // Seed the chunk cache with a rebuild while the flag is OFF: zero flagged
        // voxels, and (critically) this populates the store + `previous_leaf_index`
        // so the NEXT rebuild diffs against it.
        let before = rebuild_grid_overlay_count(&mut core, &scene, density);
        assert_eq!(before, 0, "with the flag OFF no voxel may carry the grid_overlay marker");

        // Flip ONLY voxel_grid_on_faces ON via the intent door (no other edit).
        core.apply_intent(
            &mut scene,
            Intent::SetNodeGrids {
                target,
                grids: NodeGrids { voxel_grid_on_faces: true, ..NodeGrids::default() },
            },
        );

        // Rebuild AGAIN. Before the fix the fingerprint was unchanged → no chunk
        // evicted → this stayed 0. With the flag in the fingerprint the leaf's AABB
        // reports dirty, its chunks re-resolve WITH the bit, and the grid appears now.
        let after = rebuild_grid_overlay_count(&mut core, &scene, density);
        assert!(
            after > 0,
            "after toggling voxel_grid_on_faces ON, the FIRST rebuild must flag voxels \
             (was {before}, now {after}) — no unrelated edit should be needed"
        );
    }
