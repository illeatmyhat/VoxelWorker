    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use document::intent::{Intent, NodeSpec};
    use document::scene::{NodeContent, NodeId};
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape};

    /// The `RecentreVoxels` frame newtype round-trips through its one door: whatever
    /// triple it carries is exactly what `voxels()` hands back (the accessor contract the
    /// mesh / two-layer / uniform boundaries unwrap with), and equal triples compare equal.
    #[test]
    fn recentre_voxels_round_trips_through_voxels() {
        for triple in [[0, 0, 0], [7, -3, 11], [i64::MIN, 0, i64::MAX]] {
            assert_eq!(RecentreVoxels::new(triple).voxels(), triple);
        }
        assert_eq!(RecentreVoxels::new([1, 2, 3]), RecentreVoxels::new([1, 2, 3]));
        assert_ne!(RecentreVoxels::new([1, 2, 3]), RecentreVoxels::new([1, 2, 4]));
    }

    /// A small box Tool shape for the script fixtures (3 blocks at the default
    /// density 16 → 48 voxels per axis).
    fn box_shape() -> SdfShape {
        SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, 16)
    }

    /// The replay seed base is the windowed default: exactly one top-level node (a
    /// Tool from the default geometry) and exactly one Point (the Origin), with ids
    /// minted. Scripts are written against this known starting point.
    #[test]
    fn default_seed_scene_matches_windowed_base() {
        let seed = default_replay_seed_scene();
        assert_eq!(seed.roots.len(), 1, "seed has one top-level Tool node");
        assert_eq!(seed.points.len(), 1, "seed carries exactly the Origin Point");
        assert!(
            seed.roots.iter().all(|id| id.0 != 0),
            "every seed node has a minted (non-zero) NodeId"
        );
    }

    /// Plumbing proof: an `AddNode` then a `SetOffset` (targeting the just-added node
    /// by its minted id) replays to a scene whose NEW node sits at the requested
    /// offset — i.e. the script parsed, dispatched through `apply_intent`, and the
    /// mutations landed in order.
    #[test]
    fn replay_add_then_offset_places_new_node() {
        let seed = default_replay_seed_scene();
        let roots_before = seed.roots.len();
        // The add mints the next id past the seed's counter. `apply_intent`'s add op
        // assigns `next_node_id`, which after the seed equals `roots_before + 1` given
        // the seed mints ids 1..=roots_before. Derive it from the seed to stay robust.
        let new_node_id = NodeId(seed.next_node_id);

        let add = Intent::AddNode {
            content: NodeSpec::Tool {
                shape: box_shape(),
                material: MaterialChoice::Wood,
            },
        };
        let set_offset = Intent::SetOffset {
            target: new_node_id,
            offset_measurements: document::intent::whole_block_offset([7, -2, 4]),
        };
        let script = format!(
            "{}\n\n{}\n",
            serde_json::to_string(&add).unwrap(),
            serde_json::to_string(&set_offset).unwrap(),
        );

        let scene = replay_intent_script(&script).expect("replay succeeds");
        assert_eq!(
            scene.roots.len(),
            roots_before + 1,
            "AddNode added exactly one top-level node"
        );
        let added = scene
            .node_by_id(new_node_id)
            .expect("the added node exists at its minted id");
        assert_eq!(
            // The block-granular intent is stored as canonical voxels; the derived
            // block view round-trips it exactly (block-aligned, ADR 0003 §3f(0)).
            added.transform.blocks(scene.voxels_per_block),
            [7, -2, 4],
            "SetOffset moved the just-added node to the requested offset"
        );
        assert!(
            matches!(added.content, NodeContent::Tool { .. }),
            "the added node is a Tool"
        );
    }

    /// A malformed line is reported as an `Err` (naming the line number), NOT a panic.
    #[test]
    fn replay_malformed_line_is_reported_not_panicked() {
        let good = Intent::AddNode {
            content: NodeSpec::CloudsPart,
        };
        let script = format!(
            "{}\nthis is not json\n",
            serde_json::to_string(&good).unwrap()
        );
        let error = replay_intent_script(&script).expect_err("malformed line must error");
        assert!(
            error.contains("line 2"),
            "error names the offending 1-based line number, got: {error}"
        );
    }

    /// Blank / whitespace-only lines are skipped (not parse errors).
    #[test]
    fn replay_skips_blank_lines() {
        let add = Intent::AddNode {
            content: NodeSpec::CloudsPart,
        };
        let script = format!("\n   \n{}\n\n", serde_json::to_string(&add).unwrap());
        let scene = replay_intent_script(&script).expect("blank lines skipped");
        // Seed (1 Tool) + 1 Clouds Part = 2 top-level nodes.
        assert_eq!(scene.roots.len(), 2);
    }
