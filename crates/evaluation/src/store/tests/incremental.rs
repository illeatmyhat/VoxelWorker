use super::*;

    // ===== Issue #20 S6c-2c: incremental dirty-chunk rebuild ======================

    /// A per-chunk GPU instance cache, MODELLED on CPU as `coord → that chunk's
    /// occupied multiset` (the multiset is the byte-identical proxy for the GPU
    /// buffer's contents — `renderer::instances_for_chunk` builds one VoxelInstance
    /// per occupied voxel, so two chunks with equal occupied multisets produce
    /// byte-identical instance buffers). This lets the incremental-rebuild decision
    /// logic ([`incremental_rebuild_plan`], the EXACT function the GPU path
    /// uses) be exercised without a wgpu device, while still proving the post-edit
    /// cache CONTENTS match a full rebuild.
    type RenderCache = std::collections::BTreeMap<[i32; 3], ChunkMultiset>;
    type ChunkMultiset = std::collections::BTreeMap<([u32; 3], u16), usize>;

    /// Build the render cache a WHOLESALE rebuild produces for `scene`: every
    /// covering chunk's grid as a multiset (skipping zero-voxel chunks, exactly as
    /// `renderer::rebuild_chunk` drops them — no buffer is allocated for an empty
    /// chunk).
    fn full_render_cache(scene: &Scene, density: u32) -> RenderCache {
        let mut cache = Store::new();
        let chunks = cache.resident_render_chunks(scene, density, 0);
        chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `render_cache` IN PLACE,
    /// driving the GPU-cache decisions through [`incremental_rebuild_plan`]
    /// — the same plan `VoxelRenderer::incremental_rebuild_from_chunks` applies.
    /// Returns the number of chunks rebuilt (the observability count). The resolve
    /// cache (`resolve_cache`) carries state across edits exactly as the live app's
    /// does, so a HIT chunk is reused verbatim.
    fn apply_incremental_edit(
        render_cache: &mut RenderCache,
        resolve_cache: &mut Store,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> usize {
        // 1. Edit AABB → evicted (dirty) coords, exactly as main::rebuild_geometry.
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        let edit_aabb = index_b
            .edit_aabb_since(&index_a)
            .expect("same-density localisable edit");
        let evicted = resolve_cache.invalidate_aabb(&edit_aabb, density);

        // A recentre shift rebases EVERY chunk's contents, so the incremental path is
        // invalid — main::rebuild_geometry falls back to a full rebuild. Model that.
        let recentre_changed = scene_a.recentre_voxels_for_resolve(density)
            != scene_b.recentre_voxels_for_resolve(density);

        // 2. Freshly-resolved covering chunks for scene B (resolves the dirty/new
        //    chunks, reuses HITs).
        let render_chunks = resolve_cache.resident_render_chunks(scene_b, density, 0);

        if recentre_changed {
            // Full rebuild: clear + restore every non-empty covering chunk.
            render_cache.clear();
            for (coord, grid) in &render_chunks {
                if !grid.occupied.is_empty() {
                    render_cache.insert(*coord, occupied_multiset(grid));
                }
            }
            return render_chunks.len();
        }

        let resident: Vec<[i32; 3]> = render_cache.keys().copied().collect();
        // Only NON-EMPTY covering chunks deserve a buffer (matching the renderer).
        let occupied_covering: Vec<[i32; 3]> = render_chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();

        // 3. The plan — the SAME pure function the renderer drives the GPU from.
        let plan = incremental_rebuild_plan(&resident, &evicted, &occupied_covering);

        // 4. Rebuild only the planned coords (dirty ∪ new); evict the vacated ones.
        let rebuild_set: std::collections::BTreeSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        for (coord, grid) in &render_chunks {
            if rebuild_set.contains(coord) {
                render_cache.insert(*coord, occupied_multiset(grid));
            }
        }
        for coord in &plan.evict {
            render_cache.remove(coord);
        }
        plan.rebuild.len()
    }

    /// A tool node at the given offset, for building edit scenes.
    fn tool_node(kind: ShapeKind, size: [u32; 3], offset: [i64; 3], material: MaterialChoice) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, 16);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, 16);
        node
    }

    /// **The key S6c-2c correctness test.** For a sequence of localised edits, the
    /// render cache built INCREMENTALLY (rebuild only dirty/new chunks, evict
    /// vacated) is IDENTICAL — coord set AND every chunk's instance multiset — to a
    /// full wholesale rebuild of the post-edit scene. Proves no stale chunk survives
    /// and no fresh chunk is missed. Also asserts the dirty-chunk count is STRICTLY
    /// LESS than the total resident count for a localised edit (so it is genuinely
    /// incremental, not a disguised full rebuild).
    #[test]
    fn incremental_rebuild_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // The base scene: three tools spread far apart in X so each occupies chunks
        // the others don't touch (clean localised edits). Start the render + resolve
        // caches as a wholesale build of scene A.
        // Two STATIC anchor nodes at the X extremes pin the composite extent (hence
        // the recentre / floating origin) so the interior edits below keep it FIXED —
        // that is the regime where the incremental dirty-only path is valid (a
        // recentre shift rebases every chunk and forces a full rebuild instead; see
        // `apply_incremental_edit`). The interior "subject" box sits between them.
        let anchor_lo = || tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone);
        let anchor_hi = || tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood),
            anchor_hi(),
        ]);

        // Each case mutates scene_a → scene_b by ONE edit kind, all keeping the
        // composite extent (recentre) fixed via the anchors, so all are genuinely
        // incremental. Each is checked independently from a fresh wholesale build of A.
        let recolor = {
            let mut b = scene_a.clone();
            // In-place recolor of the interior Box (material change, same geometry).
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            // In-place resize of the interior Box (few dirty chunks around it).
            // Replace content + transform in place so the node keeps its arena id.
            let replacement = tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            // Move the interior Box from +60X to +70X (still interior → recentre
            // fixed; dirty around BOTH endpoints).
            b.root_node_mut(1).transform = document::scene::NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            // ADD a new INTERIOR tool (brand-new covering chunks; extent unchanged).
            b.add_node(tool_node(ShapeKind::Box, [3, 3, 3], [90, 0, 0], MaterialChoice::Stone));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            // REMOVE the interior Box (its chunks must be evicted/vacated; the
            // anchors keep the extent so the recentre is unchanged).
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node] {
            // Precondition: every edit keeps the recentre fixed (so the incremental
            // path applies — a recentre shift would force a full rebuild and the
            // dirty-count assertion below would not hold).
            assert_eq!(
                scene_a.recentre_voxels_for_resolve(density),
                scene_b.recentre_voxels_for_resolve(density),
                "[{label}] this edit must keep the composite recentre fixed"
            );

            // Incremental: wholesale-build A, then apply the single edit to B.
            let mut resolve_cache = Store::new();
            let mut render_cache: RenderCache = {
                let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
                chunks
                    .iter()
                    .filter(|(_, grid)| !grid.occupied.is_empty())
                    .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                    .collect()
            };
            let total_before = render_cache.len();
            let rebuilt = apply_incremental_edit(
                &mut render_cache,
                &mut resolve_cache,
                &scene_a,
                &scene_b,
                density,
            );

            // The full wholesale rebuild for the post-edit scene B (the truth).
            let full = full_render_cache(&scene_b, density);

            assert_eq!(
                render_cache, full,
                "[{label}] incremental render cache (coords + each chunk's instance \
                 multiset) MUST equal a full wholesale rebuild of scene B — a stale \
                 chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: a localised edit rebuilds strictly fewer chunks
            // than the scene's total resident chunks (proving it is incremental, not
            // a disguised full rebuild). `total_before` and `full.len()` are both the
            // scene's full per-chunk count (A and B differ by one localised node), so
            // a genuine incremental edit touches a strict subset.
            let scene_chunks = total_before.max(full.len());
            assert!(
                rebuilt < scene_chunks,
                "[{label}] a localised edit must rebuild strictly FEWER chunks \
                 ({rebuilt}) than the scene's total ({scene_chunks}) — else it is a \
                 disguised full rebuild"
            );
        }
    }

    /// A focused dirty-count assertion: an in-place recolor of ONE SMALL far-flung
    /// node dirties only the handful of chunks that node occupies, NOT the whole
    /// scene — so a localised edit rebuilds far fewer than half the resident chunks.
    #[test]
    fn localized_recolor_rebuilds_few_chunks() {
        let density = 16u32;
        // A wide sphere (many chunks) plus a tiny 1-block box pushed far out in X,
        // so the box owns only ~1 chunk no other leaf touches.
        let scene_a = Scene::from_nodes(vec![
            tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone),
            tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut resolve_cache = Store::new();
        let mut render_cache: RenderCache = {
            let chunks = resolve_cache.resident_render_chunks(&scene_a, density, 0);
            chunks
                .iter()
                .filter(|(_, grid)| !grid.occupied.is_empty())
                .map(|(coord, grid)| (*coord, occupied_multiset(grid)))
                .collect()
        };
        let total = render_cache.len();
        let rebuilt =
            apply_incremental_edit(&mut render_cache, &mut resolve_cache, &scene_a, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            rebuilt * 2 < total,
            "a localised recolor of a small node must rebuild far fewer than half the \
             chunks: rebuilt {rebuilt} of {total}"
        );
        // And the result still matches a full rebuild.
        assert_eq!(render_cache, full_render_cache(&scene_b, density));
    }

