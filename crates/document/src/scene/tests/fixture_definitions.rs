use super::*;
use voxel_core::core_geom::MaterialChoice;

    // ---- ADR 0017 Decision 4 / #77: fixture definitions — the dense oracle ----
    //
    // A FIXTURE definition does not pre-compose: its children splice into the
    // HOSTING scope's ordered fold at the instance's spine position, in order,
    // under the instance's transform. The host is POSITIONAL — whatever
    // accumulated before the instance in its scope — never a stored reference.
    // These tests pin the dense-oracle semantics: the window (cut + fill in one
    // placement), the positional host, the ordering law, the one-level piercing
    // limit, reuse-by-reference def edits, the inert instance operation, and the
    // fingerprint contract a fixture flip must honour.

    const DENSITY: u32 = 8;

    /// The window definition's id in every fixture scene below.
    const WINDOW_DEF: DefId = DefId(1);

    // `box_tool` / `instance_node` / `resolved_absolute_multiset` are the shared CSG
    // fixtures in `super` (tests/mod.rs), reached via `use super::*`.

    /// Register the WINDOW fixture on `scene`: [opening cutter `Subtract`
    /// (`opening_blocks`³ footprint through the 1-block wall thickness), Wood frame
    /// `Union` (the opening's bottom row)] — the ADR's window shape — and flag it
    /// `fixture` so it splices instead of pre-composing.
    fn add_window_fixture(scene: &mut Scene, opening_blocks: u32) {
        scene.add_definition(
            WINDOW_DEF,
            "Window",
            vec![
                box_tool(
                    [opening_blocks, 1, opening_blocks],
                    [0, 0, 0],
                    MaterialChoice::Plain,
                    CombineOp::Subtract,
                ),
                box_tool(
                    [opening_blocks, 1, 1],
                    [0, 0, 0],
                    MaterialChoice::Wood,
                    CombineOp::Union,
                ),
            ],
        );
        assert!(
            scene.set_definition_fixture(WINDOW_DEF, true),
            "the window definition exists to be flagged"
        );
    }

    /// The FLAT children the window fixture must splice as, at the instance
    /// placement `offset_blocks`: the same cutter + frame authored in place.
    fn flat_window_children(opening_blocks: u32, offset_blocks: [i64; 3]) -> Vec<Node> {
        vec![
            box_tool(
                [opening_blocks, 1, opening_blocks],
                offset_blocks,
                MaterialChoice::Plain,
                CombineOp::Subtract,
            ),
            box_tool(
                [opening_blocks, 1, 1],
                offset_blocks,
                MaterialChoice::Wood,
                CombineOp::Union,
            ),
        ]
    }

    /// A Stone wall standing in the XZ plane (Z-up): 8 blocks wide, 1 thick, 6 tall.
    fn wall(offset_blocks: [i64; 3]) -> Node {
        box_tool([8, 1, 6], offset_blocks, MaterialChoice::Stone, CombineOp::Union)
    }

    /// Acceptance #1 (the golden's oracle): ONE window-fixture placement into the
    /// wall's scope both CUTS the opening and FILLS the frame — exactly as if the
    /// def's children were authored flat at the instance's transform (ADR 0008: the
    /// splice enters the host fold under the carried instance frame), with the
    /// instance's own operation never consulted.
    #[test]
    fn window_fixture_cuts_the_wall_and_fills_the_frame_in_one_placement() {
        let mut spliced = Scene::from_nodes(vec![
            wall([0, 0, 0]),
            instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window"),
        ]);
        add_window_fixture(&mut spliced, 3);

        let mut flat_children = vec![wall([0, 0, 0])];
        flat_children.extend(flat_window_children(3, [2, 0, 2]));
        let flat = Scene::from_nodes(flat_children);

        let composed = resolved_absolute_multiset(&spliced);
        assert_eq!(
            composed,
            resolved_absolute_multiset(&flat),
            "a fixture's children must splice into the host fold exactly like flat \
             leaves at the instance's transform"
        );
        // The wall lost the opening but regained the frame's bottom row; the
        // cutter's Plain material appears nowhere and the frame's Wood does.
        let d = DENSITY as usize;
        // Voxels in an x*y*z BLOCK box at this density. Named so the three extents
        // (including the single-block Y thickness) stay visible as dimensions rather
        // than collapsing into a bare product.
        let block_box_voxels = |x: usize, y: usize, z: usize| x * y * z * d.pow(3);
        let wall_voxels = block_box_voxels(8, 1, 6);
        let opening_voxels = block_box_voxels(3, 1, 3);
        let frame_voxels = block_box_voxels(3, 1, 1);
        assert_eq!(composed.len(), wall_voxels - opening_voxels + frame_voxels);
        assert!(
            composed
                .keys()
                .any(|(_index, material)| *material == MaterialChoice::Wood.block_id().0),
            "the frame child must STAMP Wood into the host (a spliced Union adds)"
        );
        assert!(
            composed
                .keys()
                .all(|(_index, material)| *material != MaterialChoice::Plain.block_id().0),
            "the opening child never stamps (a spliced Subtract is occupancy-only)"
        );
    }

    /// Acceptance #2 (the positional host): the host is whatever accumulated before
    /// the instance IN ITS SCOPE — moving the same fixture from one wall's Group into
    /// another's cuts THAT wall and restores the first (no stored host reference, no
    /// rehosting).
    #[test]
    fn moving_the_fixture_into_a_different_walls_scope_cuts_that_wall() {
        let scene_with_window_in = |carved_wall: usize| {
            let group = |name: &str, offset: [i64; 3], with_window: bool| {
                let mut children: Vec<NodeBuilder> = vec![wall(offset).into()];
                if with_window {
                    children.push(
                        instance_node(WINDOW_DEF, [offset[0] + 2, 0, 2], CombineOp::Union, "Window")
                            .into(),
                    );
                }
                NodeBuilder::group(name, children)
            };
            let mut scene = Scene::from_nodes(vec![
                group("Wall A", [0, 0, 0], carved_wall == 0),
                group("Wall B", [16, 0, 0], carved_wall == 1),
            ]);
            add_window_fixture(&mut scene, 3);
            scene
        };
        let flat_oracle = |carved_wall: usize| {
            let group = |name: &str, offset: [i64; 3], with_window: bool| {
                let mut children: Vec<NodeBuilder> = vec![wall(offset).into()];
                if with_window {
                    for child in flat_window_children(3, [offset[0] + 2, 0, 2]) {
                        children.push(child.into());
                    }
                }
                NodeBuilder::group(name, children)
            };
            Scene::from_nodes(vec![
                group("Wall A", [0, 0, 0], carved_wall == 0),
                group("Wall B", [16, 0, 0], carved_wall == 1),
            ])
        };
        for carved_wall in [0, 1] {
            assert_eq!(
                resolved_absolute_multiset(&scene_with_window_in(carved_wall)),
                resolved_absolute_multiset(&flat_oracle(carved_wall)),
                "the fixture must carve exactly the wall whose scope hosts it \
                 (carved_wall = {carved_wall})"
            );
        }
    }

    /// Acceptance #3 (the ordering law): a fixture placed BEFORE the wall cuts
    /// nothing — its cutter precedes everything in the scope (subtract-from-nothing)
    /// and splices exactly like flat leaves authored first; the wall's occupancy
    /// survives whole (the frame — inside the later wall's body — loses later-wins,
    /// so no cell renders anything but the wall).
    #[test]
    fn fixture_placed_before_the_wall_cuts_nothing() {
        let mut window_first = Scene::from_nodes(vec![
            instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window"),
            wall([0, 0, 0]),
        ]);
        add_window_fixture(&mut window_first, 3);
        let mut flat_children = flat_window_children(3, [2, 0, 2]);
        flat_children.push(wall([0, 0, 0]));
        let flat = Scene::from_nodes(flat_children);
        let composed = resolved_absolute_multiset(&window_first);
        assert_eq!(
            composed,
            resolved_absolute_multiset(&flat),
            "a fixture preceding its would-be host must splice exactly like flat \
             leaves authored first"
        );
        // Nothing was carved and nothing added outside the wall: the occupied CELL
        // set equals the wall alone (the frame's cells lie inside the wall's body,
        // where the later wall stamp wins the render).
        let wall_alone = Scene::from_nodes(vec![wall([0, 0, 0])]);
        let occupied_cells = |multiset: &std::collections::BTreeMap<([i64; 3], u16), usize>| {
            multiset
                .keys()
                .map(|(index, _material)| *index)
                .collect::<std::collections::BTreeSet<[i64; 3]>>()
        };
        assert_eq!(
            occupied_cells(&composed),
            occupied_cells(&resolved_absolute_multiset(&wall_alone)),
            "a fixture preceding the wall must neither carve it nor add cells \
             outside it (the ordering law)"
        );
    }

    /// Acceptance #4 (one-level piercing): a fixture pierces exactly ONE level of
    /// pre-composition — its host scope's seal above it stays absolute. A fixture
    /// inside a sealed Group splices into the GROUP's fold (carving the group's
    /// wall), but cannot carve the bystander outside the Group even though the
    /// bystander overlaps the cutter's volume.
    #[test]
    fn fixture_inside_a_sealed_group_cannot_carve_outside_it() {
        // A Wood bystander filling exactly the opening's volume, placed BEFORE the
        // group: under an unsealed fold the spliced cutter — later in depth-first
        // order — would carve it.
        let bystander =
            box_tool([3, 1, 3], [2, 0, 2], MaterialChoice::Wood, CombineOp::Union);
        let mut spliced = Scene::from_nodes(vec![
            NodeBuilder::Leaf(bystander.clone()),
            NodeBuilder::group(
                "Walled",
                vec![
                    wall([0, 0, 0]).into(),
                    instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window").into(),
                ],
            ),
        ]);
        add_window_fixture(&mut spliced, 3);

        // The oracle: the same children authored flat INSIDE the group — the splice
        // lands in the group's fold, one level up and no further.
        let flat = {
            let mut group_children: Vec<NodeBuilder> = vec![wall([0, 0, 0]).into()];
            for child in flat_window_children(3, [2, 0, 2]) {
                group_children.push(child.into());
            }
            Scene::from_nodes(vec![
                NodeBuilder::Leaf(bystander),
                NodeBuilder::group("Walled", group_children),
            ])
        };
        let composed = resolved_absolute_multiset(&spliced);
        assert_eq!(
            composed,
            resolved_absolute_multiset(&flat),
            "the fixture must splice into its host GROUP's fold, never past its seal"
        );
        // The bystander survives where the group's carved body left the opening
        // empty (above the frame row): Wood cells remain there, proving the cutter
        // never escaped the group.
        assert!(
            composed.keys().any(|(index, material)| {
                *material == MaterialChoice::Wood.block_id().0
                    && index[2] >= (3 * DENSITY) as i64
            }),
            "the bystander must survive in the opening above the frame row"
        );
    }

    /// Acceptance #5 (reuse by reference): editing the ONE fixture definition
    /// updates EVERY placement — hole and frame both — matching a from-scratch flat
    /// scene authored at the new size.
    #[test]
    fn editing_the_fixture_definition_updates_every_placement() {
        let two_walls_with_windows = |opening_blocks: u32| {
            let mut scene = Scene::from_nodes(vec![
                wall([0, 0, 0]),
                wall([16, 0, 0]),
                instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window 1"),
                instance_node(WINDOW_DEF, [18, 0, 2], CombineOp::Union, "Window 2"),
            ]);
            add_window_fixture(&mut scene, opening_blocks);
            scene
        };
        let flat_two_walls = |opening_blocks: u32| {
            let mut children = vec![wall([0, 0, 0]), wall([16, 0, 0])];
            children.extend(flat_window_children(opening_blocks, [2, 0, 2]));
            children.extend(flat_window_children(opening_blocks, [18, 0, 2]));
            Scene::from_nodes(children)
        };

        let mut scene = two_walls_with_windows(2);
        assert_eq!(
            resolved_absolute_multiset(&scene),
            resolved_absolute_multiset(&flat_two_walls(2)),
            "the pre-edit windows match the 2-block flat oracle at both placements"
        );

        // Edit the shared definition IN PLACE (both instances reference it): grow
        // the opening to 3×3 and the frame bar to match.
        let def_children = scene
            .def_by_id(WINDOW_DEF)
            .expect("the window definition exists")
            .children
            .clone();
        let grown_shapes = [
            SdfShape::from_blocks(ShapeKind::Box, [3, 1, 3], 1, DENSITY),
            SdfShape::from_blocks(ShapeKind::Box, [3, 1, 1], 1, DENSITY),
        ];
        for (child_id, grown) in def_children.into_iter().zip(grown_shapes) {
            match &mut scene
                .node_by_id_mut(child_id)
                .expect("the definition child resolves")
                .content
            {
                NodeContent::Tool { shape, .. } => *shape = grown,
                other => panic!("the window def children are Tools, got {other:?}"),
            }
        }

        assert_eq!(
            resolved_absolute_multiset(&scene),
            resolved_absolute_multiset(&flat_two_walls(3)),
            "one definition edit must update hole AND frame at BOTH placements"
        );
    }

    /// A fixture instance's own `CombineOp` is INERT (ADR 0017 Decision 4): the
    /// resolver never consults it, so flipping it changes nothing — the spliced
    /// children fold under their OWN operations.
    #[test]
    fn fixture_instance_operation_is_inert() {
        let scene_with_instance_op = |operation: CombineOp| {
            let mut scene = Scene::from_nodes(vec![
                wall([0, 0, 0]),
                instance_node(WINDOW_DEF, [2, 0, 2], operation, "Window"),
            ]);
            add_window_fixture(&mut scene, 3);
            scene
        };
        let union_placed = resolved_absolute_multiset(&scene_with_instance_op(CombineOp::Union));
        for operation in [CombineOp::Subtract, CombineOp::Intersect] {
            assert_eq!(
                union_placed,
                resolved_absolute_multiset(&scene_with_instance_op(operation)),
                "a fixture instance's operation must be inert ({operation:?})"
            );
        }
    }

    /// The inspector-hiding predicate ([`Scene::node_operation_is_inert`]): true
    /// exactly for an Instance of a FIXTURE definition — false for a sealed
    /// instance, for a leaf, and after the flag is cleared again.
    #[test]
    fn operation_inertness_predicate_tracks_the_fixture_flag() {
        let mut scene = Scene::from_nodes(vec![
            wall([0, 0, 0]),
            instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window"),
        ]);
        add_window_fixture(&mut scene, 3);
        let wall_id = scene.roots[0];
        let instance_id = scene.roots[1];
        let node = |scene: &Scene, id: NodeId| scene.node_by_id(id).expect("node resolves").clone();

        assert!(
            scene.node_operation_is_inert(&node(&scene, instance_id)),
            "a fixture instance's operation is inert — the inspector hides the selector"
        );
        assert!(
            !scene.node_operation_is_inert(&node(&scene, wall_id)),
            "a leaf's operation is never inert"
        );
        assert!(scene.set_definition_fixture(WINDOW_DEF, false));
        assert!(
            !scene.node_operation_is_inert(&node(&scene, instance_id)),
            "a SEALED definition's instance folds under its own operation again"
        );
        assert!(
            !scene.set_definition_fixture(DefId(99), true),
            "a dangling definition id is a no-op field write"
        );
    }

    /// The invalidation contract (the fingerprint's scope-path clause): flipping a
    /// definition's fixture flag changes EVERY expanded leaf's carried scope path
    /// (the instance frame appears/disappears), so every placement's leaves
    /// re-fingerprint — and their AABBs, which bound every cell the splice can
    /// differ in, are dirtied. Untouched siblings keep their fingerprints.
    #[test]
    fn fixture_flip_changes_every_expanded_leaf_fingerprint() {
        let scene_flagged = |fixture: bool| {
            let mut scene = Scene::from_nodes(vec![
                wall([0, 0, 0]),
                instance_node(WINDOW_DEF, [2, 0, 2], CombineOp::Union, "Window 1"),
                instance_node(WINDOW_DEF, [18, 0, 2], CombineOp::Union, "Window 2"),
            ]);
            add_window_fixture(&mut scene, 3);
            assert!(scene.set_definition_fixture(WINDOW_DEF, fixture));
            scene
        };
        let sealed_index = scene_flagged(false).build_leaf_spatial_index(DENSITY);
        let spliced_index = scene_flagged(true).build_leaf_spatial_index(DENSITY);
        // Leaves: the wall + (opening, frame) per instance = 5 entries either way
        // (the splice changes composition, not the leaf list).
        assert_eq!(sealed_index.entries.len(), 5);
        assert_eq!(spliced_index.entries.len(), 5);
        assert_eq!(
            sealed_index.entries[0].fingerprint, spliced_index.entries[0].fingerprint,
            "the wall is outside every expansion — its fingerprint is untouched"
        );
        for leaf in 1..5 {
            assert_ne!(
                sealed_index.entries[leaf].fingerprint,
                spliced_index.entries[leaf].fingerprint,
                "expanded leaf {leaf} must re-fingerprint on a fixture flip \
                 (sealed↔spliced changes its carried scope path)"
            );
        }
    }
