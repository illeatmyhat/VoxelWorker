use super::*;
use voxel_core::core_geom::MaterialChoice;

    // ---- ADR 0017 (#73): CombineOp::Subtract — the ordered document-order fold ----
    //
    // These tests pin the DENSE ORACLE semantics of the sibling-level subtract slice:
    // a leaf under `Subtract` removes occupancy from everything accumulated before it
    // (document order), never stamps material, and is a no-op when nothing precedes
    // it. The two-layer classifier is held against this oracle in the evaluation
    // crate's parity tests.

    const DENSITY: u32 = 8;

    // `box_tool` / `resolved_absolute_multiset` are the shared CSG fixtures in
    // `super` (tests/mod.rs), reached via `use super::*`.

    /// A cutter placed AFTER a solid carves it: the resolved occupancy is exactly
    /// the body's voxels MINUS the cutter's box, and no voxel carries the cutter's
    /// material (a Subtract never stamps).
    #[test]
    fn subtract_after_body_carves_its_box() {
        // Body: 2³ blocks of Stone at the origin. Cutter: 1³ block at [1,1,1] —
        // the top corner octant — placed AFTER the body under Subtract.
        let scene = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([1, 1, 1], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let carved = resolved_absolute_multiset(&scene);

        let body_voxels = (2 * DENSITY as usize).pow(3);
        let cutter_voxels = (DENSITY as usize).pow(3);
        assert_eq!(
            carved.len(),
            body_voxels - cutter_voxels,
            "the cutter must remove exactly its overlapping box of voxels"
        );
        let cutter_low = DENSITY as i64; // block [1,1,1] at density 8 → voxel 8.
        let cutter_high = 2 * DENSITY as i64;
        for (index, material) in carved.keys() {
            let inside_cutter = (0..3)
                .all(|axis| index[axis] >= cutter_low && index[axis] < cutter_high);
            assert!(
                !inside_cutter,
                "voxel {index:?} inside the cutter's box must be carved away"
            );
            assert_eq!(
                *material,
                MaterialChoice::Stone.block_id().0,
                "a surviving voxel must keep the BODY's material — the cutter never stamps"
            );
        }
    }

    /// The ordering law (ADR 0017 Decision 2): a cutter placed BEFORE its target
    /// subtracts from NOTHING — the resolved scene is identical to the body alone.
    #[test]
    fn subtract_before_body_is_a_no_op() {
        let cutter_first = Scene::from_nodes(vec![
            box_tool([1, 1, 1], [1, 1, 1], MaterialChoice::Wood, CombineOp::Subtract),
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
        ]);
        let body_alone = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        assert_eq!(
            resolved_absolute_multiset(&cutter_first),
            resolved_absolute_multiset(&body_alone),
            "a Subtract preceding its target must remove nothing (the ordered fold)"
        );
    }

    /// Subtract never changes the material of surviving cells: every voxel that
    /// survives the carve carries EXACTLY the (index, material) it has in the same
    /// scene without the cutter — across a two-material overlapping body.
    #[test]
    fn subtract_preserves_surviving_materials() {
        // Two OVERLAPPING Union boxes (Stone then Wood — later wins on the overlap)
        // so the pre-carve material field is non-uniform, then a cutter through the
        // middle of both.
        let body = vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 0, 0], MaterialChoice::Wood, CombineOp::Union),
        ];
        let mut with_cutter = body.clone();
        with_cutter.push(box_tool([1, 2, 1], [1, 0, 1], MaterialChoice::Plain, CombineOp::Subtract));

        let uncarved = resolved_absolute_multiset(&Scene::from_nodes(body));
        let carved = resolved_absolute_multiset(&Scene::from_nodes(with_cutter));

        assert!(
            carved.len() < uncarved.len(),
            "the cutter must remove at least one voxel"
        );
        for (key, count) in &carved {
            assert_eq!(
                Some(count),
                uncarved.get(key),
                "surviving voxel {key:?} must keep its pre-carve material and multiplicity"
            );
        }
    }

    /// The chunk-addressable resolve applies the SAME subtract semantics as the
    /// monolithic oracle: reassembling every covering chunk reproduces the carved
    /// occupancy exactly (recentre-normalised) — so the runtime per-chunk paths and
    /// the dense oracle can never disagree about a carve.
    #[test]
    fn chunked_resolve_matches_monolithic_for_subtract_scene() {
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract),
        ]);
        let monolithic = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        let chunked = scene.resolve_region_via_chunks(DENSITY, 0);
        assert_eq!(
            occupied_multiset(&chunked, [0, 0, 0]),
            occupied_multiset(&monolithic, scene.recentre_voxels(DENSITY)),
            "chunked subtract resolve must equal the monolithic oracle (recentre-normalised)"
        );
    }

    /// A leaf's `CombineOp` is part of its spatial-index fingerprint: flipping
    /// Union↔Subtract must change the fingerprint so the edit diff dirties the
    /// leaf's AABB (the store then RE-CLASSIFIES those chunks — a Subtract can turn
    /// coarse-solid blocks into boundary or air).
    #[test]
    fn operation_flip_changes_the_leaf_fingerprint() {
        let union_scene = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        let subtract_scene = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Subtract,
        )]);
        let union_index = union_scene.build_leaf_spatial_index(DENSITY);
        let subtract_index = subtract_scene.build_leaf_spatial_index(DENSITY);
        assert_eq!(union_index.entries.len(), 1);
        assert_eq!(subtract_index.entries.len(), 1);
        assert_ne!(
            union_index.entries[0].fingerprint, subtract_index.entries[0].fingerprint,
            "a Union↔Subtract flip must dirty the leaf (fingerprints must differ)"
        );
    }
