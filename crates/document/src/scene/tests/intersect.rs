use super::*;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::spatial_index::LeafFingerprint;
use voxel_core::voxel::ShapeKind;
use crate::voxel::SdfShape;

    // ---- ADR 0017 (#75): CombineOp::Intersect — the ordered document-order fold ----
    //
    // These tests pin the DENSE ORACLE semantics of the intersect slice: a leaf under
    // `Intersect` keeps ONLY the cells present in both the accumulated result and its
    // own body (an occupancy-only mask — surviving cells keep their ACCUMULATED
    // material, the mask never stamps), intersecting the EMPTY accumulator yields
    // empty (the fold-start edge case of the ordering law), and — unlike Subtract —
    // the mask kills accumulated cells anywhere OUTSIDE its own AABB. The two-layer
    // classifier is held against this oracle in the evaluation crate's parity tests.

    const DENSITY: u32 = 8;

    /// A whole-block Box Tool at `offset_blocks` carrying `operation` — the intersect
    /// fixtures are all axis-aligned boxes so the expected surviving set is exact.
    fn box_tool(
        size_blocks: [u32; 3],
        offset_blocks: [i64; 3],
        material: MaterialChoice,
        operation: CombineOp,
    ) -> Node {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, DENSITY);
        let mut node = Node::new("Box", NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset_blocks, DENSITY);
        node.operation = operation;
        node
    }

    /// Resolve `scene` through the dense oracle and return its occupancy multiset in
    /// ABSOLUTE voxel space (recentre-normalised), keyed `(index, material)`.
    fn resolved_absolute_multiset(
        scene: &Scene,
    ) -> std::collections::BTreeMap<([i64; 3], u16), usize> {
        let grid = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        occupied_multiset(&grid, scene.recentre_voxels(DENSITY))
    }

    /// A mask placed AFTER a solid keeps exactly the overlap: the resolved occupancy
    /// is the body's voxels INSIDE the mask's box — every body voxel outside the mask
    /// dies (including voxels far from the mask's AABB), and every survivor keeps the
    /// BODY's material (an Intersect never stamps).
    #[test]
    fn intersect_after_body_keeps_only_the_overlap() {
        // Body: 4³ blocks of Stone at the origin. Mask: 2³ blocks at [3,3,3] — its
        // lower corner octant [3,4)³ overlaps the body's top corner octant.
        let scene = Scene::from_nodes(vec![
            box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [3, 3, 3], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        let survivors = resolved_absolute_multiset(&scene);

        // The overlap is blocks [3,4)³ → voxels [24,32)³ at density 8.
        let overlap_low = 3 * DENSITY as i64;
        let overlap_high = 4 * DENSITY as i64;
        assert_eq!(
            survivors.len(),
            (DENSITY as usize).pow(3),
            "exactly the overlap block of voxels must survive the mask"
        );
        for ((index, material), _count) in &survivors {
            let inside_overlap = (0..3)
                .all(|axis| index[axis] >= overlap_low && index[axis] < overlap_high);
            assert!(
                inside_overlap,
                "surviving voxel {index:?} must lie in the body∩mask overlap"
            );
            assert_eq!(
                *material,
                MaterialChoice::Stone.block_id().0,
                "a surviving voxel must keep the BODY's material — the mask never stamps"
            );
        }
    }

    /// The fold-start edge case (ADR 0017 Decision 2): a mask placed BEFORE anything
    /// accumulated intersects the EMPTY accumulator — yielding empty — and a body
    /// placed AFTER it unions into the (still empty) accumulator untouched, so the
    /// scene resolves identical to the body alone.
    #[test]
    fn intersect_with_empty_accumulator_yields_empty() {
        // A lone Intersect leaf: nothing accumulated ⇒ nothing survives.
        let mask_alone = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Wood,
            CombineOp::Intersect,
        )]);
        assert!(
            resolved_absolute_multiset(&mask_alone).is_empty(),
            "a mask over the empty accumulator must resolve to nothing"
        );

        // Mask BEFORE the body: the mask spends itself on ∅; the body then stands alone.
        let mask_first = Scene::from_nodes(vec![
            box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Wood, CombineOp::Intersect),
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
        ]);
        let body_alone = Scene::from_nodes(vec![box_tool(
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
        )]);
        assert_eq!(
            resolved_absolute_multiset(&mask_first),
            resolved_absolute_multiset(&body_alone),
            "a mask preceding its target must remove nothing from what follows (the ordered fold)"
        );
    }

    /// Surviving cells keep their ACCUMULATED material across a two-material body: the
    /// survivors of a third-material mask are EXACTLY the uncarved cells inside the
    /// mask's box — same indices, same materials, same multiplicities — and the mask's
    /// material appears nowhere (later-wins applies to additive ops only).
    #[test]
    fn intersect_preserves_surviving_materials() {
        // Two OVERLAPPING Union boxes (Stone then Wood — later wins on the overlap)
        // so the pre-mask material field is non-uniform, then a Plain mask covering
        // the lower half in Z: survivors = every accumulated voxel with z < 8.
        let body = vec![
            box_tool([2, 2, 2], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 0, 0], MaterialChoice::Wood, CombineOp::Union),
        ];
        let mut with_mask = body.clone();
        with_mask.push(box_tool([3, 2, 1], [0, 0, 0], MaterialChoice::Plain, CombineOp::Intersect));

        let unmasked = resolved_absolute_multiset(&Scene::from_nodes(body));
        let masked = resolved_absolute_multiset(&Scene::from_nodes(with_mask));

        let mask_z_high = DENSITY as i64; // block [0,1) in Z → voxels [0,8).
        let expected: std::collections::BTreeMap<([i64; 3], u16), usize> = unmasked
            .iter()
            .filter(|((index, _), _)| index[2] < mask_z_high)
            .map(|(key, count)| (*key, *count))
            .collect();
        assert!(!expected.is_empty(), "the mask must keep a non-empty slab");
        assert_eq!(
            masked, expected,
            "survivors must be exactly the accumulated voxels inside the mask, \
             with their pre-mask materials and multiplicities"
        );
        assert!(
            masked
                .keys()
                .any(|(_, material)| *material == MaterialChoice::Stone.block_id().0)
                && masked
                    .keys()
                    .any(|(_, material)| *material == MaterialChoice::Wood.block_id().0),
            "both accumulated materials must survive inside the mask"
        );
        assert!(
            masked
                .keys()
                .all(|(_, material)| *material != MaterialChoice::Plain.block_id().0),
            "the mask's own material must appear nowhere — an Intersect never stamps"
        );
    }

    /// Sealed scopes (ADR 0017 Decision 3) compose with Intersect on both sides:
    /// a GROUP placed under Intersect masks the parent accumulator with the group's
    /// composed occupancy, and an Intersect leaf INSIDE a group masks only within its
    /// scope (an outside bystander survives untouched).
    #[test]
    fn intersect_respects_sealed_scopes() {
        // (a) Group-under-Intersect: the Stone body is masked by the group's Wood box
        // — survivors are the Stone cells inside blocks [2,4)³, still Stone.
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [4, 4, 4],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Mask scope",
                vec![box_tool([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Union)
                    .into()],
            ),
        ]);
        let group_id = scene.roots[1];
        scene
            .node_by_id_mut(group_id)
            .expect("the group resolves")
            .operation = CombineOp::Intersect;
        let survivors = resolved_absolute_multiset(&scene);
        let low = 2 * DENSITY as i64;
        let high = 4 * DENSITY as i64;
        assert_eq!(
            survivors.len(),
            (2 * DENSITY as usize).pow(3),
            "exactly the group-covered cells must survive"
        );
        for ((index, material), _count) in &survivors {
            assert!(
                (0..3).all(|axis| index[axis] >= low && index[axis] < high),
                "survivor {index:?} must lie inside the group's composed body"
            );
            assert_eq!(*material, MaterialChoice::Stone.block_id().0);
        }

        // (b) Intersect INSIDE a group: the mask trims the group's own body only; the
        // bystander placed before the group — overlapping nothing of the mask's scope
        // — survives whole.
        let scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [2, 2, 2],
                [6, 6, 6],
                MaterialChoice::Wood,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Masked body",
                vec![
                    box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union).into(),
                    box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Plain, CombineOp::Intersect)
                        .into(),
                ],
            ),
        ]);
        let survivors = resolved_absolute_multiset(&scene);
        // Group body [0,4)³ ∩ mask [1,3)³ = blocks [1,3)³ Stone; bystander [6,8)³ Wood.
        let stone_cells = (2 * DENSITY as usize).pow(3);
        let wood_cells = (2 * DENSITY as usize).pow(3);
        assert_eq!(survivors.len(), stone_cells + wood_cells);
        for ((index, material), _count) in &survivors {
            if *material == MaterialChoice::Wood.block_id().0 {
                assert!(
                    (0..3).all(|axis| {
                        index[axis] >= 6 * DENSITY as i64 && index[axis] < 8 * DENSITY as i64
                    }),
                    "the bystander must survive UNTOUCHED — a mask sealed in a scope \
                     cannot reach outside it (voxel {index:?})"
                );
            } else {
                assert_eq!(*material, MaterialChoice::Stone.block_id().0);
                assert!(
                    (0..3).all(|axis| {
                        index[axis] >= DENSITY as i64 && index[axis] < 3 * DENSITY as i64
                    }),
                    "the group's body must be trimmed to its internal mask (voxel {index:?})"
                );
            }
        }
    }

    /// The chunk-addressable resolve applies the SAME intersect semantics as the
    /// monolithic oracle — including the never-skip rule for masks (a chunk the mask's
    /// AABB misses must still be emptied by it) and the scoped close — so the runtime
    /// per-chunk paths and the dense oracle can never disagree about a mask.
    #[test]
    fn chunked_resolve_matches_monolithic_for_intersect_scenes() {
        // Flat: a big body whose far chunks lie OUTSIDE the mask's AABB (the
        // asymmetry that distinguishes Intersect from Subtract in the chunk skip).
        let scene = Scene::from_nodes(vec![
            box_tool([8, 8, 8], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union),
            box_tool([2, 2, 2], [1, 1, 1], MaterialChoice::Wood, CombineOp::Intersect),
        ]);
        let monolithic = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        let chunked = scene.resolve_region_via_chunks(DENSITY, 0);
        assert_eq!(
            occupied_multiset(&chunked, [0, 0, 0]),
            occupied_multiset(&monolithic, scene.recentre_voxels(DENSITY)),
            "chunked intersect resolve must equal the monolithic oracle (recentre-normalised)"
        );

        // Scoped: a group closing under Intersect whose body misses whole chunks of
        // the parent body — the ∅-in-chunk scope close must still annihilate there.
        let mut scene = Scene::from_nodes(vec![
            NodeBuilder::Leaf(box_tool(
                [8, 8, 8],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
            )),
            NodeBuilder::group(
                "Mask scope",
                vec![box_tool([2, 2, 2], [5, 5, 5], MaterialChoice::Wood, CombineOp::Union)
                    .into()],
            ),
        ]);
        let group_id = scene.roots[1];
        scene
            .node_by_id_mut(group_id)
            .expect("the group resolves")
            .operation = CombineOp::Intersect;
        let monolithic = scene.resolve_region(scene.full_extent_blocks(DENSITY), DENSITY, 0);
        let chunked = scene.resolve_region_via_chunks(DENSITY, 0);
        assert_eq!(
            occupied_multiset(&chunked, [0, 0, 0]),
            occupied_multiset(&monolithic, scene.recentre_voxels(DENSITY)),
            "chunked SCOPED intersect resolve must equal the monolithic oracle"
        );
    }

    /// Invalidation conservatism (ADR 0017 #75): an Intersect-influence leaf's edits
    /// cannot be localised to its box (the mask kills cells anywhere outside its
    /// body), so (a) a Union↔Intersect flip changes the fingerprint, (b) such a leaf
    /// carries the `MasksBeyondItsBox` fingerprint kind, and (c) an edit diff
    /// involving it degrades to `None` — the wholesale-clear fallback — never a
    /// too-small box union.
    #[test]
    fn intersect_edits_force_the_wholesale_invalidation_fallback() {
        let body = || box_tool([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union);
        let mask_at = |offset_blocks: [i64; 3]| {
            box_tool([2, 2, 2], offset_blocks, MaterialChoice::Wood, CombineOp::Intersect)
        };

        let union_scene = Scene::from_nodes(vec![body()]);
        let masked_scene = Scene::from_nodes(vec![body(), mask_at([1, 1, 1])]);
        let moved_mask_scene = Scene::from_nodes(vec![body(), mask_at([2, 2, 2])]);

        let masked_index = masked_scene.build_leaf_spatial_index(DENSITY);
        assert!(
            matches!(
                masked_index.entries[1].fingerprint,
                LeafFingerprint::MasksBeyondItsBox(_)
            ),
            "an Intersect leaf must carry the beyond-its-box fingerprint kind"
        );
        assert!(
            matches!(masked_index.entries[0].fingerprint, LeafFingerprint::Bounded(_)),
            "a plain Union leaf keeps the localisable fingerprint kind"
        );

        // Adding the mask, and moving it, both involve a MasksBeyondItsBox entry in
        // the diff ⇒ the conservative wholesale fallback.
        let union_index = union_scene.build_leaf_spatial_index(DENSITY);
        let moved_index = moved_mask_scene.build_leaf_spatial_index(DENSITY);
        assert_eq!(
            masked_index.edit_aabb_since(&union_index),
            None,
            "introducing a mask must force the wholesale-clear fallback"
        );
        assert_eq!(
            moved_index.edit_aabb_since(&masked_index),
            None,
            "moving a mask must force the wholesale-clear fallback (its effect \
             reaches outside both boxes)"
        );
        // An unchanged masked scene still diffs to the empty AABB (no false clears).
        let masked_again = masked_scene.build_leaf_spatial_index(DENSITY);
        assert!(
            masked_again
                .edit_aabb_since(&masked_index)
                .expect("identical scenes must not force a clear")
                .is_empty(),
            "an identical masked scene must diff to the empty dirty AABB"
        );
    }
