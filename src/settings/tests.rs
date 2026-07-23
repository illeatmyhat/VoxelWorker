    use super::*;
    use voxel_core::voxel::ShapeKind;

    /// Save and reload through the on-disk artifact, which since the ADR 0022 split is
    /// the dump rather than this struct. Most tests below care about what survives a
    /// save/load, not about which type spells the JSON, so they go through here.
    fn save_and_reload(config: &AppConfig) -> AppConfig {
        let json = config.to_dump_json().expect("serialise");
        AppConfig::from_dump_json(&json).expect("deserialise")
    }

    #[test]
    fn config_round_trips_through_json() {
        let config = AppConfig {
            scene: None,
            voxels_per_block: 24,
            projection_mode: ProjectionMode::Orthographic,
            material: MaterialChoice::Wood,
            show_view_cube: false,
            applied_block_label: Some("Granite".to_string()),
            snap_to_blocks: false,
            onion_skin: true,
            onion_depth: 5,
            orbit_theta: 1.23,
            orbit_phi: 0.95,
            orbit_distance: 42.0,
            orbit_target: [3.0, -7.5, 11.0],
            home_theta: 2.34,
            home_phi: 1.11,
            home_distance: 18.0,
            home_explicit: true,
            window_size: [1600, 900],
            view_mode: ViewMode::OnionFog,
            stack: SignalStackState {
                folded: true,
                viewport_open: false,
                onion_open: true,
                grids_open: false,
            },
            debug_face_orientation: true,
            debug_brick_faces: true,
            placement_ghost: Some(PlacementGhostConfig {
                shape_kind: ShapeKind::Sphere,
                size_voxels: [24, 16, 32],
                wall_blocks: 2,
                offset_voxels: [40, -8, 12],
            }),
            // Non-default so the round-trip actually exercises persistence.
            placement_snap: PlacementSnap {
                position: ui::panel::PositionSnap::Block,
                angle: ui::panel::AngleSnap::Deg15,
                pivot: ui::panel::PlacementPivot::VolumetricCenter,
            },
            // Non-default (Some) so the round-trip exercises sketch-mode persistence (ADR 0028).
            sketch_mode: Some(document::scene::NodeId(9)),
            // Non-default (not Select) so the round-trip exercises the armed sketch tool (#95).
            sketch_tool: ui::panel::SketchTool::AddPoint,
        };

        let restored = save_and_reload(&config);
        assert_eq!(config, restored);
    }

    /// ADR 0024: the session state survives the full live round trip —
    /// `PanelState → capture → JSON → load → to_panel_state` — which is the leg that was
    /// broken. Each of these four was classified as reaching the dump and was hard-coded
    /// to a default on the way back, so a test asserting only that `AppConfig` round-trips
    /// would have passed throughout. The assertion has to start and end at the panel.
    #[test]
    fn the_session_survives_a_relaunch_through_the_panel_and_back() {
        let mut panel = PanelState::with_view_cube_default();
        panel.view_mode = ViewMode::OnionFog;
        panel.stack = SignalStackState {
            folded: true,
            viewport_open: false,
            onion_open: true,
            grids_open: false,
        };
        panel.debug_face_orientation = true;
        panel.debug_brick_faces = true;
        panel.placement_snap = PlacementSnap {
            position: ui::panel::PositionSnap::Block,
            angle: ui::panel::AngleSnap::Deg15,
            pivot: ui::panel::PlacementPivot::VolumetricCenter,
        };

        let config = AppConfig::capture(
            &panel,
            &OrbitCamera::default(),
            HomeView::default(),
            [1280, 800],
        );
        let restored = save_and_reload(&config).to_panel_state();

        assert_eq!(restored.view_mode, ViewMode::OnionFog);
        assert_eq!(restored.stack, panel.stack);
        assert!(restored.debug_face_orientation);
        assert!(restored.debug_brick_faces);
        assert_eq!(restored.placement_snap, panel.placement_snap, "snap settings survive relaunch");
    }

    /// The other direction of the same promise: a dump written before the session category
    /// existed carries none of these keys, and must load as the finished look rather than
    /// failing. Every session field has a serde default, so an old repro still replays —
    /// which is the reason the dump tolerates missing keys at all.
    #[test]
    fn a_dump_without_session_keys_loads_the_finished_look() {
        let panel = AppConfig::from_dump_json(r#"{"voxels_per_block": 8}"#)
            .expect("a pre-session dump parses")
            .to_panel_state();
        assert_eq!(panel.view_mode, ViewMode::Normal);
        assert_eq!(panel.stack, SignalStackState::default());
        assert!(!panel.debug_face_orientation);
        assert!(!panel.debug_brick_faces);
    }

    /// #13: the home-view fields persist through capture→JSON→load, and an OLD
    /// config WITHOUT them loads with the camera defaults (serde fills each
    /// missing key from its `#[serde(default = ...)]` fn).
    #[test]
    fn home_view_persists_and_old_config_defaults() {
        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        let camera = OrbitCamera::default();
        let home = HomeView { theta: 2.5, phi: 0.6, distance: 33.0, explicitly_set: true };
        let config = AppConfig::capture(&panel, &camera, home, [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_home = restored.home_view();
        assert!((restored_home.theta - 2.5).abs() < 1e-5);
        assert!((restored_home.phi - 0.6).abs() < 1e-5);
        assert!((restored_home.distance - 33.0).abs() < 1e-5);

        // An old config with no home_* keys loads with the camera defaults.
        let old_json = r#"{ "voxels_per_block": 8 }"#;
        let old = AppConfig::from_dump_json(old_json).expect("old config without home_* parses");
        let old_home = old.home_view();
        let defaults = HomeView::default();
        assert!((old_home.theta - defaults.theta).abs() < 1e-5);
        assert!((old_home.phi - defaults.phi).abs() < 1e-5);
        assert!((old_home.distance - defaults.distance).abs() < 1e-5);
    }

    /// issue #32: a config persists and reloads its `scene` correctly. A non-trivial
    /// scene (two offset Tool nodes with distinct materials) survives
    /// `capture → JSON → deserialize → to_panel_state` with the same node count,
    /// active selection, and resolved occupancy — the `scene` field is the single
    /// source of truth now that the flat geometry mirror fields are gone.
    #[test]
    fn config_persists_and_reloads_its_scene() {
        use document::scene::{Node, NodeContent, NodePath, Scene};
        use document::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape::from_blocks(kind, [1, 1, 1], 1, voxels_per_block);
        let stone = Node::new(
            "Stone",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Stone },
        );
        let mut wood = Node::new(
            "Wood",
            NodeContent::Tool { shape: unit_box(ShapeKind::Box), material: MaterialChoice::Wood },
        );
        wood.transform = document::scene::NodeTransform::from_blocks([3, 0, 0], voxels_per_block);
        // ADR 0003 Phase B3: selection is keyed by NodeId, so mint ids and select
        // the second node (top-level index 1) by its stable id.
        let mut scene = Scene::from_nodes(vec![stone, wood]);
        scene.active = scene.id_at_path(&NodePath::root_index(1));

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the scene");

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        assert_eq!(restored_panel.scene.roots.len(), 2, "both nodes survive the reload");
        assert_eq!(restored_panel.scene.active, scene.active, "the active selection survives");
        assert_eq!(
            restored_panel.scene.root_node(1).transform.blocks(voxels_per_block),
            [3, 0, 0]
        );

        let region = scene.full_extent_blocks(voxels_per_block);
        let before = scene.resolve_region(region, voxels_per_block, 0).occupied_count();
        let after_region = restored_panel.scene.full_extent_blocks(voxels_per_block);
        let after = restored_panel
            .scene
            .resolve_region(after_region, voxels_per_block, 0)
            .occupied_count();
        assert_eq!(before, after, "the restored scene resolves identically");
    }

    #[test]
    fn bad_json_falls_back_without_panicking() {
        // An empty object still parses thanks to the per-field defaults.
        let restored = AppConfig::from_dump_json("{}").expect("empty object parses");
        assert_eq!(restored, AppConfig::default());

        // Outright invalid JSON must be a clean Err (the caller turns it into a
        // defaults fallback), never a panic.
        assert!(AppConfig::from_dump_json("not json at all}{").is_err());
    }

    /// issue #31 + #32: the legacy grid `show_*` mirror fields (`show_grid_overlay` /
    /// `show_block_lattice` / `show_floor_grid`), the older `show_origin_gizmo`, AND
    /// the flat geometry mirror fields (`shape` / `size_blocks` / `wall_blocks`) were
    /// all removed from `AppConfig`. There is no `deny_unknown_fields`, so an OLD
    /// config still carrying those keys must keep deserializing cleanly — serde
    /// ignores the now-unknown keys. The masters no longer migrate from the grid
    /// keys, and a scene-less config simply loads the default seed scene whose
    /// `Scene::default()` masters all default ON.
    #[test]
    fn old_config_with_removed_keys_still_loads() {
        let old_json = r#"{
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "show_grid_overlay": true,
            "show_block_lattice": false,
            "show_floor_grid": true,
            "show_origin_gizmo": true
        }"#;
        let config = AppConfig::from_dump_json(old_json)
            .expect("old config with removed keys still parses");
        assert!(config.scene.is_none());
        // The app-level density key is the one flat field still read.
        assert_eq!(config.voxels_per_block, 8);

        let panel = config.to_panel_state();
        // The removed keys are simply ignored — they no longer seed the masters or
        // the geometry. A scene-less config loads the default seed scene whose
        // masters all default ON.
        assert!(panel.scene.master_block_lattice, "fresh scene masters default ON");
        assert!(panel.scene.master_voxel_grid, "fresh scene masters default ON");
        assert!(panel.scene.master_floor_grid, "fresh scene masters default ON");
        // Exactly one Origin Point, as on any load path.
        assert_eq!(panel.scene.points.iter().filter(|p| p.is_origin).count(), 1);
    }

    /// issue #32: an OLD config carrying the dropped `debug_clouds` boolean AND the
    /// removed flat geometry mirror fields (`shape` / `size_blocks` / `wall_blocks`)
    /// must load gracefully — serde ignores the now-unknown keys. The persisted
    /// app-level density (`voxels_per_block`) and `material` still round-trip, and a
    /// scene-less config loads the DEFAULT seed scene (no longer a scene built from
    /// the removed flat params).
    #[test]
    fn old_config_with_debug_clouds_field_still_loads() {
        let old_json = r#"{
            "shape": "Sphere",
            "size_blocks": [3, 4, 5],
            "voxels_per_block": 20,
            "wall_blocks": 2,
            "debug_clouds": true,
            "material": "Wood"
        }"#;
        let restored = AppConfig::from_dump_json(old_json).expect("old config (with debug_clouds) must still parse");
        // The flat geometry keys are ignored; only density + material survive.
        assert_eq!(restored.voxels_per_block, 20);
        assert_eq!(restored.material, MaterialChoice::Wood);
        // An old config has NO `scene` field, so it deserialises to `None`, which now
        // loads the default seed scene (the same one a brand-new config gets).
        assert!(restored.scene.is_none(), "an old flat config carries no scene");

        // It loads the DEFAULT seed scene (a one-Tool-node Cylinder, NOT a scene built
        // from the removed flat `shape`/`size_blocks`/`wall_blocks`). Only the density
        // carries over from the config.
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1);
        // Density DID carry over from the config and now lives on the document
        // (ADR 0003 §3f(0)), not the shape.
        assert_eq!(panel.scene.voxels_per_block, 20);
        match panel.scene.active_node().map(|node| &node.content) {
            Some(document::scene::NodeContent::Tool { shape, material }) => {
                // The default seed geometry, NOT the persisted flat params.
                assert_eq!(shape.kind, ShapeKind::Cylinder);
                // Size is voxel-canonical now (ADR 0003 §3f(0)): the 5×1×5-block seed
                // built at the persisted density 20 = [100, 20, 100] voxels.
                assert_eq!(shape.size_voxels, [100, 20, 100]);
                // The persisted `material` rides the seed (it is still an AppConfig field).
                assert_eq!(*material, MaterialChoice::Wood);
            }
            other => panic!("the seed must build a one Tool node, got {other:?}"),
        }
    }

    /// Part of #20: the legacy instanced mesher was removed along with the
    /// `MesherChoice` toggle. The choice was never a persisted `AppConfig` field
    /// (it lived only in the session-only `PanelState`), but defend the migration
    /// regardless: an OLD config JSON that carried a stray top-level `mesher` field
    /// (e.g. hand-edited) must STILL load — serde ignores the now-unknown field —
    /// and every real field round-trips.
    #[test]
    fn old_config_with_mesher_field_still_loads() {
        let old_json = r#"{
            "shape": "Cylinder",
            "size_blocks": [5, 1, 5],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "mesher": "Instanced",
            "material": "Stone"
        }"#;
        let restored = AppConfig::from_dump_json(old_json)
            .expect("old config (with mesher) must still parse");
        // The flat geometry keys are ignored (issue #32); density + material survive.
        assert_eq!(restored.voxels_per_block, 8);
        assert_eq!(restored.material, MaterialChoice::Stone);
        // It loads cleanly to the default one-Tool-node seed scene (the stray `mesher`
        // and the removed flat geometry keys are all ignored).
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1);
    }

    /// step 8 round-trip: a NON-TRIVIAL scene (top-level Tool + VoxelBody nodes with
    /// non-zero offsets and distinct materials, a Group with children, an
    /// `AssemblyDef`, and an `Instance` of it) survives
    /// `capture → JSON → deserialize → to_panel_state` structurally intact and
    /// resolves to the SAME occupied count.
    #[test]
    fn full_scene_round_trips_through_json() {
        use document::scene::{
            DefId, Node, NodeBuilder, NodeContent, NodePath, VoxelBody, Scene,
        };
        use document::voxel::SdfShape;

        let voxels_per_block = 8u32;
        let unit_box = |kind| SdfShape::from_blocks(kind, [1, 1, 1], 1, voxels_per_block);

        // A definition (the reusable "house" body): a single Wood box.
        let def_id = DefId(3);

        // Top-level node 0: a Stone Tool at the origin.
        let stone = Node::new(
            "Stone",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Box),
                material: MaterialChoice::Stone,
            },
        );
        // Top-level node 1: a Clouds VoxelBody, offset.
        let mut clouds = Node::new("Clouds", NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 7 }));
        clouds.transform = document::scene::NodeTransform::from_blocks([3, 0, 0], voxels_per_block);
        // Top-level node 2: a Group containing a Plain Tool offset within it.
        let mut grouped_leaf = Node::new(
            "Leaf",
            NodeContent::Tool {
                shape: unit_box(ShapeKind::Sphere),
                material: MaterialChoice::Plain,
            },
        );
        grouped_leaf.transform = document::scene::NodeTransform::from_blocks([1, 0, 0], voxels_per_block);
        // Top-level node 2: a Group at +6X containing the Plain Tool offset within it
        // (`CombineOp::Union` is the default operation a built Group carries).
        let group = NodeBuilder::group_at("Group", [6, 0, 0], voxels_per_block, vec![grouped_leaf.into()]);
        // Top-level node 3: an Instance of the def, offset disjointly.
        let mut instance = Node::new("House instance", NodeContent::Instance(def_id));
        instance.transform = document::scene::NodeTransform::from_blocks([-6, 0, 0], voxels_per_block);

        // ADR 0003 Phase B3: selection is keyed by NodeId, so mint ids and select
        // the Group's child (path [2, 0]) by its stable id.
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(stone),
            NodeBuilder::Leaf(clouds),
            group,
            NodeBuilder::Leaf(instance),
        ]);
        scene.add_definition(
            def_id,
            "House".to_string(),
            vec![Node::new(
                "Body",
                NodeContent::Tool {
                    shape: unit_box(ShapeKind::Box),
                    material: MaterialChoice::Wood,
                },
            )],
        );
        scene.active = scene.id_at_path(&NodePath::from_indices(vec![2, 0]));

        // Build a panel carrying this scene and capture → JSON → restore.
        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = voxels_per_block;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        assert!(config.scene.is_some(), "capture persists the full scene");

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        // Structural equality: same node tree, definitions, and active selection.
        assert_eq!(
            restored_panel.scene.roots.len(),
            scene.roots.len(),
            "all top-level nodes survive"
        );
        assert_eq!(restored_panel.scene.definitions.len(), 1, "the def survives");
        assert_eq!(
            restored_panel.scene.active,
            scene.active,
            "the active selection survives"
        );
        // The Group's child and the def's body survive with their offsets/materials.
        match &restored_panel.scene.root_node(2).content {
            NodeContent::Group(children) => {
                assert_eq!(children.len(), 1);
                assert_eq!(
                    restored_panel.scene.arena[&children[0]]
                        .transform
                        .blocks(voxels_per_block),
                    [1, 0, 0]
                );
            }
            other => panic!("node 2 must stay a Group, got {other:?}"),
        }
        assert!(matches!(
            restored_panel.scene.root_node(3).content,
            NodeContent::Instance(id) if id == def_id
        ));

        // Same resolved occupancy (the document means the same thing on reload).
        let region = scene.full_extent_blocks(voxels_per_block);
        let before = scene
            .resolve_region(region, voxels_per_block, 0)
            .occupied_count();
        let after_region = restored_panel.scene.full_extent_blocks(voxels_per_block);
        let after = restored_panel
            .scene
            .resolve_region(after_region, voxels_per_block, 0)
            .occupied_count();
        assert_eq!(before, after, "the restored scene resolves identically");
    }

    /// step 8 (never panic on load): a config whose `scene` value is broken/partial
    /// still loads. A scene object missing its inner fields deserialises to an
    /// empty-node scene (every scene field is `#[serde(default)]`), which
    /// `to_panel_state` treats as absent → falls back to the one-Tool-node seed.
    #[test]
    fn malformed_scene_falls_back_to_default_without_panicking() {
        // A `scene` present but EMPTY (no nodes) — a partial/degenerate persisted
        // value. It parses (defaults fill the missing fields) and migrates.
        let partial = r#"{
            "scene": {},
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 12,
            "wall_blocks": 1
        }"#;
        let restored = AppConfig::from_dump_json(partial).expect("a partial scene object still parses");
        let panel = restored.to_panel_state();
        assert_eq!(
            panel.scene.roots.len(),
            1,
            "an empty persisted scene falls back to the one-Tool-node seed"
        );

        // A `scene` whose arena holds a node with a content variant that doesn't exist
        // is a clean parse error wholesale → `load()` would return None → caller uses
        // defaults. We assert it never panics: the deserialize is an Err, not an unwind.
        // (The id-keyed arena is the real node storage, so the broken node must live
        // there — a stray legacy `"nodes"` key would simply be ignored by serde.)
        let broken = r#"{ "scene": { "roots": [1], "arena": { "1": { "content": "NotAVariant" } } } }"#;
        assert!(
            AppConfig::from_dump_json(broken).is_err(),
            "a structurally broken scene is a clean Err (load → defaults), never a panic"
        );
    }

    /// S4a back-compat: a small i32-range `offset_voxels` value carried in a JSON
    /// document widens into the now-`[i64; 3]` field unchanged. A JSON integer carries
    /// no width, so serde reads it straight into `i64` — the "tolerant persistence
    /// migration" S4a requires. The document must load, keep its offsets, and resolve
    /// to a non-empty grid. (Placement is canonical voxels at the document density now,
    /// ADR 0003 §3f(0); authored here as a whole-block offset via `from_blocks`.)
    ///
    /// **ADR 0003 Phase B5 note:** the original version of this test hand-authored a
    /// `"scene": { "nodes": [ … ] }` document in the OLD positional-`Vec<Node>` on-disk
    /// shape. Phase B5 flipped scene storage to an id-keyed `arena` + `roots` spine, so
    /// that legacy array shape no longer deserializes (the field is gone). Per project
    /// policy (pre-alpha; old saves may break — see no-config-back-compat memory) the
    /// test is REWRITTEN to author the scene via the API and round-trip it through the
    /// NEW on-disk shape, still exercising i64-offset WIDENING (its real purpose, which
    /// is orthogonal to the storage layout). The small i32-range offset is what an old
    /// save held; the assertion that it lands as the same `i64` value is unchanged.
    #[test]
    fn old_i32_offset_scene_loads_after_widening_to_i64() {
        use document::scene::{Node, NodeContent, Scene};
        use document::voxel::SdfShape;

        // A single Box Tool offset +5 blocks in X — a small i32-range offset, exactly
        // what a pre-S4a `[i32; 3]` save held. Authored via the API, then serialized to
        // the current on-disk format and reloaded; the offset is a plain JSON integer
        // in that document, so reloading proves it reads into the `i64` field intact.
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16);
        let mut node = Node::new(
            "Box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        // +5 blocks in X at density 8 → canonical offset_voxels = [40, 0, 0].
        node.transform = document::scene::NodeTransform::from_blocks([5, 0, 0], 8);
        let scene = Scene::single_node(node);

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        let json = config.to_dump_json().expect("serialise");

        // Sanity: the persisted offset really is a bare JSON integer (no width), the
        // exact condition the widening relies on. Checked against the parsed value rather
        // than the text, because a dump is written pretty and the whitespace between the
        // array elements is not the property under test.
        let written: serde_json::Value = serde_json::from_str(&json).expect("re-parse");
        assert_eq!(
            written["scene"]["arena"]
                .as_object()
                .and_then(|arena| arena.values().next())
                .map(|node| &node["transform"]["offset_voxels"]),
            Some(&serde_json::json!([40, 0, 0])),
            "the offset persists as plain JSON integers (no width): {json}"
        );

        let restored = AppConfig::from_dump_json(&json).expect("an i32-range-offset scene must parse");
        let panel = restored.to_panel_state();
        assert_eq!(panel.scene.roots.len(), 1, "the node survives the widening");
        // The i32-range offset widened into the i64 field intact.
        assert_eq!(
            panel.scene.root_node(0).transform.offset_voxels,
            [40i64, 0, 0],
            "the old i32 offset must widen to the same i64 value"
        );
        assert!(matches!(
            panel.scene.root_node(0).content,
            NodeContent::Tool { .. }
        ));
        // The migrated document still resolves to a non-empty grid.
        let region = panel.scene.full_extent_blocks(8);
        assert!(
            panel.scene.resolve_region(region, 8, 0).occupied_count() > 0,
            "the migrated old-offset scene resolves to voxels"
        );
    }

    /// S4a: a scene whose `offset_voxels` is a LARGE i64 (well beyond the old
    /// `i32` range, ±2.1×10⁹) round-trips through `capture → JSON → load` byte-exact.
    /// This proves the widened field both serializes and deserializes the full
    /// 64-bit range — far-apart village nodes survive a save/load. (Placement is
    /// canonical voxels now, ADR 0003 §3f(0); the large value is set directly on the
    /// voxel field to exercise the full i64 range it persists.)
    #[test]
    fn large_i64_offset_round_trips_through_json() {
        use document::scene::{Node, NodeContent, Scene};
        use document::voxel::SdfShape;

        // Beyond i32::MAX (2_147_483_647): a node placed ~3 billion blocks out. An
        // i32 field could never have held this; the i64 field must persist it exactly.
        let far_offset: i64 = 3_000_000_000;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16);
        let mut node = Node::new(
            "Far box",
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform.offset_voxels = [far_offset, -far_offset, far_offset / 2];
        let scene = Scene::single_node(node);

        let mut panel = PanelState::with_view_cube_default();
        panel.geometry.voxels_per_block = 8;
        panel.scene = scene.clone();
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        assert_eq!(
            restored_panel.scene.roots.len(),
            1,
            "the far node survives the round-trip"
        );
        assert_eq!(
            restored_panel.scene.root_node(0).transform.offset_voxels,
            [far_offset, -far_offset, far_offset / 2],
            "a >i32-range i64 offset must round-trip byte-exact through save/load"
        );
        // ADR 0003 Phase B3: selection is keyed by NodeId; `single_node` minted the
        // lone node an id and selected it, and that id round-trips intact.
        assert_eq!(
            restored_panel.scene.active,
            scene.active,
            "the active selection survives"
        );
        assert!(scene.active.is_some(), "the lone node is selected by id");
    }

    /// issue #31: the grid masters are the single source of truth on `scene.master_*`
    /// and round-trip through `capture → JSON → to_panel_state` directly (no legacy
    /// `show_*` mirror). Non-default master values must survive the round-trip.
    #[test]
    fn capture_then_to_panel_state_preserves_masters_and_toggles() {
        let mut panel = PanelState::with_view_cube_default();
        // Drive non-default master values directly on the scene (the UI checkboxes do
        // the same). Mixed values prove each master persists independently.
        panel.scene.master_block_lattice = false;
        panel.scene.master_voxel_grid = true;
        panel.scene.master_floor_grid = false;
        panel.material = MaterialChoice::Plain;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1024, 768]);
        let restored = config.to_panel_state();
        // The masters round-trip via `scene.master_*` — the single source of truth.
        assert_eq!(restored.scene.master_block_lattice, panel.scene.master_block_lattice);
        assert_eq!(restored.scene.master_voxel_grid, panel.scene.master_voxel_grid);
        assert_eq!(restored.scene.master_floor_grid, panel.scene.master_floor_grid);
        assert_eq!(restored.material, panel.material);
        assert_eq!(restored.geometry, panel.geometry);
    }

    /// issue #29 (grid rework S1) + issue #31: loading an OLD config (no `scene`
    /// field — the legacy flat geometry) gains exactly one Origin Point on the load
    /// path. The grid masters no longer migrate from legacy `show_*` keys (deleted in
    /// #31); the scene-less config seeds a fresh scene whose masters all default ON.
    #[test]
    fn old_config_gains_origin_point_with_default_masters() {
        let old_json = r#"{
            "shape": "Box",
            "size_blocks": [2, 2, 2],
            "voxels_per_block": 8,
            "wall_blocks": 1,
            "show_grid_overlay": true,
            "show_block_lattice": false,
            "show_floor_grid": true
        }"#;
        let config = AppConfig::from_dump_json(old_json).expect("old config parses");
        assert!(config.scene.is_none(), "an old flat config carries no scene");

        let panel = config.to_panel_state();
        // Exactly one Origin Point synthesized on load.
        assert_eq!(
            panel.scene.points.iter().filter(|p| p.is_origin).count(),
            1,
            "the load path synthesizes exactly one Origin Point"
        );
        assert_eq!(panel.scene.points.len(), 1);
        assert!(panel.scene.points[0].is_origin);
        assert_eq!(panel.scene.points[0].name, "Origin");

        // The removed legacy `show_*` keys are ignored — masters default ON from
        // `Scene::default()` (NOT migrated from the stale `show_block_lattice=false`).
        assert!(panel.scene.master_block_lattice, "fresh scene masters default ON");
        assert!(panel.scene.master_voxel_grid, "fresh scene masters default ON");
        assert!(panel.scene.master_floor_grid, "fresh scene masters default ON");
    }

    /// issue #29 + #31: a scene carrying its own masters keeps them on reload — the
    /// masters persist directly on the `scene` field (the single source of truth),
    /// not via any legacy `show_*` mirror. The Origin is not duplicated.
    #[test]
    fn modern_scene_keeps_its_masters_and_single_origin() {
        use document::scene::{Node, NodeContent, NodePath, Point, Scene};
        use document::voxel::SdfShape;

        let node = Node::new(
            "Box",
            NodeContent::Tool {
                shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                material: MaterialChoice::Stone,
            },
        );
        let mut scene = Scene::from_nodes(vec![node]);
        scene.master_block_lattice = false;
        scene.master_voxel_grid = true;
        scene.master_floor_grid = false;
        // ADR 0003 Phase B3: select the lone node by its stable id (from_nodes minted it).
        scene.active = scene.id_at_path(&NodePath::root_index(0));
        scene.ensure_origin_point();
        scene.add_point(Point { name: "Marker".to_string(), ..Point::default() });

        let mut panel = PanelState::with_view_cube_default();
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);

        let restored = save_and_reload(&config);
        let restored_panel = restored.to_panel_state();

        // The scene's own masters survive (NOT overwritten by the legacy show_*).
        assert!(!restored_panel.scene.master_block_lattice);
        assert!(restored_panel.scene.master_voxel_grid);
        assert!(!restored_panel.scene.master_floor_grid);
        // Still exactly one Origin (not duplicated on reload).
        assert_eq!(
            restored_panel.scene.points.iter().filter(|p| p.is_origin).count(),
            1
        );
        assert_eq!(restored_panel.scene.points.len(), 2, "Origin + Marker survive");
    }

    /// ADR 0003 Phase B3 regression: a persisted scene whose nodes carry the
    /// `NodeId(0)` sentinel and a stale `next_node_id` (an unminted save) must be
    /// minted on the load path, not left selection-dead. Without the
    /// `ensure_node_ids` call in `to_panel_state`, `id_at_path` would resolve a
    /// clicked node to `NodeId(0)`, which `node_by_id`/`path_of` reject — so the
    /// node would be silently unselectable and the next edit op would mint a
    /// colliding id.
    #[test]
    fn unminted_persisted_scene_gets_ids_minted_on_load() {
        use document::scene::{Node, NodeContent, NodePath, NodeId, Scene};
        use document::voxel::SdfShape;

        let make_box = |name: &str| {
            Node::new(
                name,
                NodeContent::Tool {
                    shape: SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, 16),
                    material: MaterialChoice::Stone,
                },
            )
        };
        // REWRITTEN for the id-keyed arena (ADR 0003 B5): the old fixture built two
        // `NodeId(0)` nodes, but the arena is keyed BY id, so it cannot hold two
        // sentinel nodes, and `ensure_node_ids` re-keys a lone 0-node in the arena
        // WITHOUT rewriting the `roots`/Group spines that reference it — so a
        // roots-references-sentinel save is not representable/positionally reachable.
        // The surviving, load-path-exercised guarantee is the STALE-COUNTER half: a
        // persisted scene whose nodes already carry ids but whose `next_node_id` was
        // never advanced past them must be normalised on load so a later edit op mints
        // a non-colliding id and every row stays selectable. We forge exactly that
        // persisted shape by resetting the counter in the serialized JSON.
        let scene = Scene::from_nodes(vec![make_box("First"), make_box("Second")]);

        let mut panel = PanelState::with_view_cube_default();
        panel.scene = scene;
        let camera = OrbitCamera::default();
        let config = AppConfig::capture(&panel, &camera, HomeView::default(), [1280, 800]);
        let dump_json = config.to_dump_json().expect("serialise");
        let mut config_value: serde_json::Value =
            serde_json::from_str(&dump_json).expect("re-parse the dump");
        // Forge a stale counter: the nodes carry real ids, but `next_node_id` sits at 0
        // (as a save written before the counter was persisted/advanced would).
        *config_value
            .get_mut("scene")
            .and_then(|s| s.get_mut("next_node_id"))
            .expect("the persisted scene carries a counter") = serde_json::json!(0);

        let json = serde_json::to_string_pretty(&config_value).expect("re-serialise");
        let restored = AppConfig::from_dump_json(&json).expect("deserialise");
        let loaded = restored.to_panel_state();

        // Every node carries a real id, and the counter now sits past all of them.
        assert!(
            loaded.scene.arena.values().all(|node| node.id != NodeId(0)),
            "every loaded node carries a real (non-sentinel) id"
        );
        let max_id = loaded.scene.arena.keys().map(|id| id.0).max().unwrap();
        assert!(loaded.scene.next_node_id > max_id, "counter advanced past every live id");

        // A clicked top-level row resolves to a selectable node (not the sentinel).
        let clicked_id = loaded.scene.id_at_path(&NodePath::root_index(0)).expect("path resolves to an id");
        assert_ne!(clicked_id, NodeId(0));
        assert!(loaded.scene.node_by_id(clicked_id).is_some(), "the resolved id is selectable");
    }
