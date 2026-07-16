use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- issue #29 (grid rework S3): per-object block lattice box (renderer-follow) ----

    /// Build a single-Box-node scene at `offset`, return its
    /// `node_block_lattice_box_recentred` for node 0 at `density`.
    fn single_node_lattice_box(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        density: u32,
    ) -> ([f32; 3], [f32; 3]) {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
        node.transform = NodeTransform::from_blocks(offset_blocks, density);
        let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
        scene
            .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
            .expect("a sized Box node has a lattice box")
    }

    /// The per-object lattice box spans the node's enclosing-block AABB and SCALES
    /// with density: a `B`-block extent → a `B·d`-voxel box, at each density
    /// {1, 15, 16} (the explicit user ask).
    ///
    /// The producer-true corner geometry is asserted in
    /// `node_block_aabb_scales_and_centres_across_densities` — in the RECENTRED frame
    /// the box is shifted by the composite recentre, so the recentred corners need not
    /// be block multiples; the block-aligned STRUCTURE (extent = B·d, planes step d)
    /// is what survives the recentre, and that is what this asserts.
    #[test]
    fn lattice_box_spans_enclosing_blocks_and_scales_with_density() {
        let size = [5u32, 3, 2];
        let offset = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            let (min, max) = single_node_lattice_box(size, offset, density);
            for (axis, &size_axis) in size.iter().enumerate() {
                // Box extent = size · density voxels (B-block extent → B·d voxels).
                assert_eq!(
                    (max[axis] - min[axis]) as i64,
                    (size_axis * density) as i64,
                    "axis {axis} @ d{density}: lattice box extent must be size·d voxels"
                );
                // The extent is an exact multiple of a block, so the box encloses
                // exactly `size_axis` whole blocks along each axis.
                assert_eq!(
                    ((max[axis] - min[axis]) as i64).rem_euclid(density as i64),
                    0,
                    "axis {axis} @ d{density}: box extent spans whole blocks"
                );
            }
        }
    }

    /// Follow-on-translate: translating the node by `+1 block` shifts its lattice box
    /// by exactly `density` voxels per axis (the lattice follows the object), at each
    /// density {1, 15, 16}. Because the node offset is whole-block, a SUB-block
    /// (1-voxel) translate is NOT representable at the node level, so the
    /// "add/remove a whole block on a sub-block move" requirement cannot be
    /// constructed here; the whole-block follow IS the unit tested. (The
    /// expand-to-block that WOULD turn a sub-block shift into a whole-block box
    /// change is exercised directly on `block_boundaries`/`*_vertices_into` in the
    /// renderer tests.)
    #[test]
    fn lattice_box_follows_whole_block_translate_at_each_density() {
        let size = [5u32, 3, 2];
        let base = [3i64, -2, 4];
        for density in [1u32, 15, 16] {
            // A SECOND, LARGE anchor node (centred at the origin, ±100 blocks on
            // every axis) dominates the composite AABB on all axes, so the small
            // moving node never touches a composite corner and the recentre stays
            // FIXED. Observed in that fixed frame, moving the node by +1 block shifts
            // its box by exactly d — the "lattice follows the object in the global
            // lattice frame" property. (A lone node would drag its own recentre, so
            // the box would NOT appear to move — see `node_pivot_origin_*`.)
            let make_scene = |offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, density);
                let mut moving = Node::new(
                    "Moving",
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                moving.transform = NodeTransform::from_blocks(offset, density);
                let anchor_shape = SdfShape::from_blocks(ShapeKind::Box, [200, 200, 200], 1, density);
                let mut anchor = Node::new(
                    "Anchor",
                    NodeContent::Tool { shape: anchor_shape, material: MaterialChoice::Stone },
                );
                // CORNER-ANCHORING: a leaf spans `[off, off+size)` blocks, so to make
                // the 200³ anchor BRACKET the small moving node on every axis (and so
                // dominate the composite AABB, fixing the recentre) it must be offset to
                // `[−100, 100)` blocks, not corner-anchored at the origin.
                anchor.transform = NodeTransform::from_blocks([-100, -100, -100], density);
                scene_with_top_level_selected(Scene::from_nodes(vec![moving, anchor]), 0)
            };
            let box_of = |offset: [i64; 3]| {
                make_scene(offset)
                    .node_block_lattice_box_recentred(&NodePath::root_index(0), density)
                    .expect("moving node has a lattice box")
            };
            let before = box_of(base);
            for moved_axis in 0..3 {
                let mut shifted = base;
                shifted[moved_axis] += 1; // +1 block
                let after = box_of(shifted);
                for axis in 0..3 {
                    let expected = if axis == moved_axis { density as f32 } else { 0.0 };
                    assert_eq!(
                        after.0[axis] - before.0[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block on axis {moved_axis} must shift the \
                         lattice box min by exactly d (0 elsewhere)"
                    );
                    assert_eq!(
                        after.1[axis] - before.1[axis],
                        expected,
                        "axis {axis} @ d{density}: +1 block must shift the lattice box max by d"
                    );
                }
            }
        }
    }

    /// A size-less node (a VoxelBody with no intrinsic extent — `DebugClouds`) has NO
    /// lattice box: `node_block_lattice_box_recentred` returns `None` (nothing to
    /// draw), at each density.
    #[test]
    fn sizeless_node_has_no_lattice_box() {
        for density in [1u32, 15, 16] {
            let scene = Scene::single_node(Node::new(
                "Clouds",
                NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }),
            ));
            assert_eq!(
                scene.node_block_lattice_box_recentred(&NodePath::root_index(0), density),
                None,
                "@ d{density}: a size-less node yields no lattice box"
            );
        }
    }

    // ---- issue #29 (grid rework S1): per-node grids, Points, masters ----

    /// A freshly-built node carries NO grids (issue #29: grids default OFF for new
    /// objects). `NodeGrids::default()` is all-false, and `Node::new` adopts it.
    #[test]
    fn new_node_has_all_grids_off() {
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        assert!(!node.grids.voxel_grid_on_faces);
        assert!(!node.grids.block_lattice);
        assert!(!node.grids.floor_grid);
        assert_eq!(node.grids, NodeGrids::default());
    }

    /// An empty `Scene::default()` has the issue-#29 grid-rework master defaults:
    /// ALL THREE masters ON (per-object flags stay OFF), and no Points yet.
    #[test]
    fn scene_default_master_grids() {
        let scene = Scene::default();
        assert!(scene.master_block_lattice, "block lattice master defaults ON");
        assert!(scene.master_voxel_grid, "voxel grid master defaults ON");
        assert!(scene.master_floor_grid, "floor grid master defaults ON");
        assert!(scene.points.is_empty(), "no Points until ensure_origin_point");
        assert_eq!(scene.active_point, None);
    }

    /// `ensure_origin_point` is idempotent and creates EXACTLY one Origin at index 0
    /// with the spec defaults (ground plane + axes on); a second call (or a scene
    /// that already has an Origin) does not duplicate it.
    #[test]
    fn ensure_origin_point_is_idempotent_and_creates_one_origin() {
        let mut scene = Scene::default();
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "exactly one Point after first call");
        let origin = &scene.points[0];
        assert!(origin.is_origin, "the synthesized Point is the Origin");
        assert_eq!(origin.name, "Origin");
        assert_eq!(origin.position_blocks, [0, 0, 0]);
        // Z-up: the ground plane is XY (`plane_xy`).
        assert!(origin.plane_xy, "ground plane (XY) on by default");
        assert!(origin.axis_x && origin.axis_y && origin.axis_z, "all axes on by default");
        assert!(!origin.plane_xz && !origin.plane_yz);
        assert!(!origin.hidden);

        // Idempotent: a second call does not add another Origin.
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 1, "second call adds nothing");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// ADR 0003 Phase B: `ensure_node_ids` mints a unique non-zero id for every
    /// node — top-level, Group children, and definition nodes — and is idempotent.
    #[test]
    fn ensure_node_ids_mints_unique_stable_ids() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group("G", vec![clouds("B").into(), clouds("C").into()]),
        ]);
        scene.add_definition(DefId(1), "Def".to_string(), vec![clouds("D")]);

        scene.ensure_node_ids();

        // Collect every id (top-level + Group children + definition nodes). Every node
        // lives in the arena keyed by its id, so the arena keys ARE the full id set.
        let ids: Vec<NodeId> = scene.arena.keys().copied().collect();
        assert_eq!(ids.len(), 5, "A, G, B, C, D all visited");
        assert!(ids.iter().all(|&id| id != NodeId(0)), "no node keeps the 0 sentinel");
        let unique: std::collections::HashSet<_> = ids.iter().copied().collect();
        assert_eq!(unique.len(), ids.len(), "every minted id is unique");

        // Idempotent: a second pass mints nothing and changes no id.
        let before = scene.clone();
        scene.ensure_node_ids();
        assert_eq!(scene, before, "second call is a no-op");
    }

    /// A loaded scene that already carries an id keeps it, and the counter advances
    /// past it so a newly-minted node never collides.
    #[test]
    fn ensure_node_ids_preserves_existing_and_advances_counter() {
        // A loaded scene: the arena is keyed by id, so a node that already carries a
        // minted id (the "preset", id 5) lives under key NodeId(5), while a still-
        // unminted node sits under the NodeId(0) sentinel. `next_node_id` starts at 0,
        // as it would for a freshly-deserialized scene before normalization.
        let mut preset = Node::new("preset", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }));
        preset.id = NodeId(5);
        let mut fresh = Node::new("fresh", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }));
        fresh.id = NodeId(0);
        let mut scene = Scene::default();
        scene.arena.insert(NodeId(5), preset);
        scene.arena.insert(NodeId(0), fresh);
        scene.roots = vec![NodeId(5), NodeId(0)];

        scene.ensure_node_ids();

        // The preset id is preserved verbatim.
        assert!(scene.arena.contains_key(&NodeId(5)), "existing id preserved");
        assert_eq!(scene.arena[&NodeId(5)].name, "preset");
        // The unminted node was re-keyed out of the 0 sentinel into a fresh, distinct id.
        assert!(!scene.arena.contains_key(&NodeId(0)), "the 0 sentinel is gone");
        let fresh_id = scene
            .arena
            .iter()
            .find(|(_, node)| node.name == "fresh")
            .map(|(id, _)| *id)
            .expect("the fresh node still exists under a minted id");
        assert_ne!(fresh_id, NodeId(0), "fresh node minted");
        assert_ne!(fresh_id, NodeId(5), "fresh id does not collide with the existing one");
        assert!(scene.next_node_id > 5, "counter advanced past the loaded id");
        // Re-keying must repoint the SPINE, not just move the arena entry: the root slot
        // that referenced the sentinel now names the fresh id, so the node is still
        // reachable through `roots` (a stale NodeId(0) here would silently orphan it).
        assert_eq!(scene.roots[1], fresh_id, "the root spine slot was repointed off the sentinel");
        assert_eq!(
            scene.node_at_path(&NodePath::root_index(1)).map(|node| node.name.as_str()),
            Some("fresh"),
            "the re-keyed node still resolves through the spine, not orphaned",
        );
    }

    /// ADR 0003 Phase B2: `id_at_path` / `path_of` / `node_by_id` agree with the
    /// positional `node_at_path` for EVERY node in the tree (the ⇄ equivalence the
    /// later selection/command migration relies on).
    #[test]
    fn node_id_and_path_resolution_round_trip() {
        fn clouds(name: &str) -> Node {
            Node::new(name, NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }))
        }
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(clouds("A")),
            NodeBuilder::group(
                "G",
                vec![
                    clouds("B").into(),
                    NodeBuilder::group("H", vec![clouds("C").into()]),
                ],
            ),
            NodeBuilder::Leaf(clouds("D")),
        ]);
        scene.ensure_node_ids();

        // Every tree row resolves both ways, consistently.
        for (path, row_id, _depth) in scene.tree_rows() {
            let id = scene.id_at_path(&path).expect("path resolves to an id");
            assert_eq!(id, row_id, "the row's carried id matches id_at_path");
            assert_ne!(id, NodeId(0), "a minted node never has the 0 sentinel");
            assert_eq!(
                scene.path_of(id),
                Some(path.clone()),
                "path_of inverts id_at_path"
            );
            // node_by_id and node_at_path reach the SAME node.
            let by_id = scene.node_by_id(id).expect("id resolves to a node");
            let by_path = scene.node_at_path(&path).expect("path resolves to a node");
            assert_eq!(by_id.id, by_path.id);
            assert_eq!(by_id.name, by_path.name);
        }

        // Sentinel + unknown ids resolve to nothing.
        assert!(scene.node_by_id(NodeId(0)).is_none());
        assert!(scene.path_of(NodeId(0)).is_none());
        assert!(scene.node_by_id(NodeId(9_999)).is_none());
        assert!(scene.path_of(NodeId(9_999)).is_none());

        // Mutable lookup reaches the same node.
        let first_id = scene.id_at_path(&NodePath::root_index(0)).unwrap();
        scene.node_by_id_mut(first_id).unwrap().name = "renamed".to_string();
        assert_eq!(scene.node_at_path(&NodePath::root_index(0)).unwrap().name, "renamed");
    }

    /// An existing Origin (anywhere in the list) is NOT duplicated by
    /// `ensure_origin_point`; a scene that already carries one is left untouched.
    #[test]
    fn ensure_origin_point_does_not_duplicate_existing_origin() {
        let mut scene = Scene::default();
        // Seed a non-origin Point first, then an Origin at index 1.
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });
        scene.add_point(Point { name: "Origin".to_string(), is_origin: true, ..Point::default() });
        scene.ensure_origin_point();
        assert_eq!(scene.points.len(), 2, "no Origin inserted when one exists");
        assert_eq!(scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// `add_point` gives a newly-added user Point the clean default (issue #29 fix):
    /// **all planes OFF** with **all three axes ON** — even if the caller passes a
    /// Point with planes enabled. Only the Origin (built by `ensure_origin_point`,
    /// not `add_point`) keeps the ground (XY, Z-up) plane on.
    #[test]
    fn add_point_defaults_planes_off_axes_on() {
        let mut scene = Scene::default();
        // Pass a Point with EVERY plane on; add_point must override them off.
        scene.add_point(Point {
            name: "User".to_string(),
            plane_xz: true,
            plane_xy: true,
            plane_yz: true,
            axis_x: false,
            axis_y: false,
            axis_z: false,
            ..Point::default()
        });
        let point = &scene.points[0];
        assert!(!point.plane_xz && !point.plane_xy && !point.plane_yz, "new point: all planes OFF");
        assert!(point.axis_x && point.axis_y && point.axis_z, "new point: all axes ON");

        // The Origin (via ensure_origin_point) still keeps the ground plane on
        // (Z-up: ground = XY = `plane_xy`).
        let mut origin_scene = Scene::default();
        origin_scene.ensure_origin_point();
        assert!(origin_scene.points[0].plane_xy, "Origin keeps the ground plane (XY)");
    }

    /// `remove_point` deletes a normal Point but NO-OPS on the Origin (undeletable),
    /// and `toggle_point_hidden` hides the Origin (hideable).
    #[test]
    fn remove_point_spares_origin_which_is_hideable() {
        let mut scene = Scene::default();
        scene.ensure_origin_point(); // Origin at index 0
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() }); // index 1

        // Removing the Origin is a no-op.
        scene.remove_point(0);
        assert_eq!(scene.points.len(), 2, "the Origin is undeletable");
        assert!(scene.points[0].is_origin);

        // Removing a normal Point works.
        scene.remove_point(1);
        assert_eq!(scene.points.len(), 1, "a normal Point is removable");
        assert!(scene.points[0].is_origin);

        // Out-of-range removal is a no-op (never panics).
        scene.remove_point(99);
        assert_eq!(scene.points.len(), 1);

        // The Origin is hideable: toggling its hidden flag works.
        assert!(!scene.points[0].hidden);
        scene.toggle_point_hidden(0);
        assert!(scene.points[0].hidden, "the Origin can be hidden");
        scene.toggle_point_hidden(0);
        assert!(!scene.points[0].hidden, "and un-hidden");
    }

    /// Serde round-trip: a Scene whose node carries non-default `NodeGrids` plus a
    /// custom Point round-trips through JSON byte-equal (structurally).
    #[test]
    fn scene_with_grids_and_points_round_trips() {
        let mut node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [1, 1, 1], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        node.grids = NodeGrids {
            voxel_grid_on_faces: true,
            block_lattice: false,
            floor_grid: true,
        };
        let mut built = Scene::from_nodes(vec![node]);
        built.master_block_lattice = false;
        built.master_voxel_grid = true;
        built.master_floor_grid = true;
        built.active_point = Some(1);
        let mut scene = scene_with_top_level_selected(built, 0);
        scene.ensure_origin_point();
        // Push directly (not via `add_point`, which overrides plane/axis flags to the
        // new-point default) so the round-trip exercises non-default per-axis flags.
        scene.points.push(Point {
            name: "Corner".to_string(),
            position_blocks: [3, 4, 5],
            plane_xz: false,
            plane_xy: true,
            plane_yz: true,
            axis_x: true,
            axis_y: false,
            axis_z: true,
            hidden: true,
            ..Point::default()
        });

        let json = serde_json::to_string_pretty(&scene).expect("serialise");
        let restored: Scene = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(scene, restored, "scene with grids + points round-trips");
        assert!(restored.root_node(0).grids.voxel_grid_on_faces);
        assert!(restored.root_node(0).grids.floor_grid);
        assert!(!restored.master_block_lattice);
        assert!(restored.master_voxel_grid);
        assert_eq!(restored.points.len(), 2);
        assert_eq!(restored.points[1].position_blocks, [3, 4, 5]);
        // Per-axis flags survive the round-trip (issue #29 fix: split axes).
        assert!(restored.points[1].axis_x && !restored.points[1].axis_y && restored.points[1].axis_z);
    }

    /// Back-compat: an OLD serialized scene (no `grids`, no `points`, no masters)
    /// deserialises with the correct defaults — node grids all-off, all three
    /// masters at their struct default (ON, issue #29 grid-rework fix), empty points.
    #[test]
    fn old_scene_json_loads_with_grid_defaults() {
        // Build a one-Box scene, serialize it, then STRIP the optional fields that an
        // old document would not carry (the per-node `grids`, the scene-wide masters,
        // `points`, `active_point`). Deserializing the trimmed JSON must fill every
        // missing field with its struct default.
        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        let mut value = serde_json::to_value(&scene).expect("serialise");
        let object = value.as_object_mut().expect("scene serializes to an object");
        // Drop the optional/defaulted fields so the load path must synthesize them.
        object.remove("master_block_lattice");
        object.remove("master_voxel_grid");
        object.remove("master_floor_grid");
        object.remove("points");
        object.remove("active_point");
        // Strip every node's `grids` so the per-node default (#29 all-off) is exercised.
        if let Some(arena) = object.get_mut("arena").and_then(|a| a.as_object_mut()) {
            for stored in arena.values_mut() {
                if let Some(node_obj) = stored.as_object_mut() {
                    node_obj.remove("grids");
                }
            }
        }
        let old_json = serde_json::to_string(&value).expect("re-serialise trimmed doc");

        let scene: Scene = serde_json::from_str(&old_json).expect("old scene parses");
        assert_eq!(scene.roots.len(), 1);
        assert_eq!(scene.root_node(0).grids, NodeGrids::default(), "grids default off");
        assert!(scene.master_block_lattice, "lattice master default on");
        assert!(scene.master_voxel_grid && scene.master_floor_grid, "all masters default on");
        assert!(scene.points.is_empty(), "no points in the old document");
        assert_eq!(scene.active_point, None);
    }

    /// Issue #29 S2: the transform gizmo's pivot is the SELECTED node's block-AABB
    /// centre in the recentred render frame — `block_aabb_centre·d − recentre` —
    /// `None` when nothing is selected, across densities.
    #[test]
    fn active_gizmo_placement_follows_selected_node() {
        for vpb in [1u32, 15, 16] {
            // Bake each node's whole-block offset at the resolve density `vpb` so the
            // stored voxel offset divides back to the same block offset under this
            // resolution (the gizmo reads `offset_voxels / vpb` → blocks).
            let make_tool = |kind, size: [u32; 3], offset: [i64; 3]| {
                let shape = SdfShape::from_blocks(kind, size, 1, vpb);
                let mut node = Node::new(
                    format!("{kind:?}"),
                    NodeContent::Tool { shape, material: MaterialChoice::Stone },
                );
                node.transform = NodeTransform::from_blocks(offset, vpb);
                node
            };
            // Three even-sized boxes; box B sits +8X, box C sits +6Z. CORNER-ANCHORING:
            // a 4-block box at offset `off` spans `[off, off+4]` blocks, centre `off+2`.
            let mut scene = Scene::from_nodes(vec![
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [8, 0, 0]),
                make_tool(ShapeKind::Box, [4, 4, 4], [0, 0, 6]),
            ]);
            scene.active = None;
            // ADR 0003 Phase B3: mint ids so selecting a node by id resolves.
            scene.ensure_node_ids();

            // Nothing selected → no gizmo.
            assert_eq!(
                scene.active_gizmo_placement(vpb),
                None,
                "no selection hides the gizmo (vpb={vpb})"
            );

            let recentre = scene.recentre_voxels_for_resolve(vpb).voxels();
            let density = vpb as i64;

            // Expected pivot for a 4-block box at block OFFSET `off`: its geometric
            // centre is `(off + 2)·d` voxels (corner-anchored), minus the recentre.
            let half_extent_voxels = 2 * density; // half of the 4-block extent
            let expected_pivot = |off_blocks: [i64; 3]| {
                [
                    (off_blocks[0] * density + half_extent_voxels - recentre[0]) as f32,
                    (off_blocks[1] * density + half_extent_voxels - recentre[1]) as f32,
                    (off_blocks[2] * density + half_extent_voxels - recentre[2]) as f32,
                ]
            };

            // Select each node in turn; the gizmo pivot tracks it.
            for (index, centre) in [([0, 0, 0]), ([8, 0, 0]), ([0, 0, 6])].into_iter().enumerate() {
                scene.active = scene.id_at_path(&NodePath::root_index(index));
                let (pivot, extent) =
                    scene.active_gizmo_placement(vpb).expect("selection shows the gizmo");
                assert_eq!(
                    pivot,
                    expected_pivot(centre),
                    "pivot == centre·d − recentre for node {index} (vpb={vpb})"
                );
                // Extent is the node's OWN 4-block AABB (not the whole region).
                assert_eq!(
                    extent,
                    [(4 * density) as f32; 3],
                    "gizmo sized from the node's own extent (vpb={vpb})"
                );
            }
        }
    }

    /// Issue #29 S2: a SINGLE selected node recentres onto the origin, so its gizmo
    /// pivot is exactly `[0, 0, 0]` (for an EVEN-sized node, whose block-AABB centre
    /// lands on an integer voxel). The gizmo only visibly moves with a multi-node
    /// selection. Guards against reading the pivot from absolute (un-recentred) space.
    #[test]
    fn single_even_selected_node_gizmo_sits_at_origin() {
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 2, 6], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            assert_eq!(
                pivot,
                [0.0, 0.0, 0.0],
                "the lone even-sized selected node recentres onto the origin (vpb={vpb})"
            );
        }
    }

    /// CHANGED (center-anchoring retirement): for an ODD-sized lone node the gizmo
    /// pivot now sits at WITHIN HALF A VOXEL of the origin for ALL densities —
    /// including the odd-size/odd-density case the old block-lattice shift got wrong
    /// (it left the pivot half a BLOCK off). The gizmo pivot and the composite
    /// recentre are now BOTH derived from the producer-true voxel frame, so a lone
    /// node's centre coincides with the recentre: pivot is exactly 0 for an even voxel
    /// span and ±0.5 voxel for an odd one (the truncation of a half-voxel centre).
    #[test]
    fn single_odd_selected_node_gizmo_is_at_most_half_voxel_off_origin() {
        // Sizes (3, 1, 5) are all odd. The lone node's pivot stays WITHIN half a voxel
        // of origin (NOT half a block, as the retired #30 shift produced) — exactly 0
        // when the voxel span size·d is even, ±0.5 voxel when odd.
        for vpb in [1u32, 15, 16] {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 5], 1, vpb);
            let mut node =
                Node::new("Box", NodeContent::Tool { shape, material: MaterialChoice::Stone });
            node.transform = NodeTransform::from_blocks([123, -45, 67], vpb);
            let scene = scene_with_top_level_selected(Scene::from_nodes(vec![node]), 0);
            let (pivot, _) = scene.active_gizmo_placement(vpb).expect("gizmo shown");
            for (axis, &component) in pivot.iter().enumerate() {
                assert!(
                    component.abs() <= 0.5,
                    "lone odd-sized node pivot within half a voxel of origin \
                     (axis {axis}, vpb={vpb}, got {component})"
                );
            }
            if vpb % 2 == 0 {
                assert_eq!(
                    pivot, [0.0, 0.0, 0.0],
                    "even density makes the lone-node recentre exact (vpb={vpb})"
                );
            }
        }
    }
