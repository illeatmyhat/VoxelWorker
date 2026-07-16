use super::*;

    // ===== ADR 0010 #54: chunk-granular INCREMENTAL edits on the two-layer path ======
    //
    // Mirrors `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`:
    // for every edit kind, the two-layer resident cache after an INCREMENTAL edit
    // (invalidate the dirty AABB's chunks, re-derive only those) is IDENTICAL — the
    // coarse layer + overlay + microblock maps + seam flags, via the derived
    // `TwoLayerChunk: PartialEq` — to a full from-scratch two-layer rebuild of scene B.

    /// A tool node for the incremental edit scenes (mirrors `store.rs::tool_node`).
    fn incr_tool_node(
        kind: ShapeKind,
        size: [u32; 3],
        offset: [i64; 3],
        material: MaterialChoice,
        density: u32,
    ) -> Node {
        let shape = SdfShape::from_blocks(kind, size, 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    }

    /// The full resident map a WHOLESALE two-layer rebuild produces for `scene`: every
    /// covering chunk built from scratch, keyed by absolute coord. This is the parity
    /// gate's ground truth — the "full rebuild" every incremental edit must equal.
    fn full_two_layer_resident(
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        let mut cache = TwoLayerResidentCache::enabled();
        let chunks = cache.resident_two_layer_chunks(scene, density, 0);
        chunks
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Snapshot a resident cache's covering chunks (post-edit) as an owned coord→chunk
    /// map, for the `== full` comparison.
    fn resident_snapshot(
        cache: &mut TwoLayerResidentCache,
        scene: &Scene,
        density: u32,
    ) -> BTreeMap<[i32; 3], TwoLayerChunk> {
        cache
            .resident_two_layer_chunks(scene, density, 0)
            .into_iter()
            .map(|(coord, chunk)| (coord, (*chunk).clone()))
            .collect()
    }

    /// Apply ONE incremental edit (scene_a → scene_b) to `cache` in place, driving the
    /// dirty set exactly as `app_core::rebuild`: build the leaf spatial index for both
    /// scenes, diff for the edit AABB, and `invalidate_aabb` the dirty chunks (or
    /// `clear()` for the non-localisable fallback). Returns `(evicted_count, took_aabb_path)`
    /// so the harness can assert the localisable edits touch a strict subset.
    fn apply_two_layer_incremental_edit(
        cache: &mut TwoLayerResidentCache,
        scene_a: &Scene,
        scene_b: &Scene,
        density: u32,
    ) -> (usize, bool) {
        let index_a = scene_a.build_leaf_spatial_index(density);
        let index_b = scene_b.build_leaf_spatial_index(density);
        match index_b.edit_aabb_since(&index_a) {
            Some(edit_aabb) => {
                let evicted = cache.invalidate_aabb(&edit_aabb, density);
                (evicted.len(), true)
            }
            None => {
                // The wholesale fallback: a density change or a region-spanning Part edit
                // has no localisable box (mirrors `app_core::rebuild`'s `clear()` arm).
                cache.clear();
                (0, false)
            }
        }
    }

    /// **THE #54 GATE — incremental == full for every LOCALISABLE edit kind.** For each of
    /// add / remove / move / resize / recolor, the two-layer resident cache after the
    /// incremental edit is IDENTICAL (coarse layer + overlay + microblock maps + seam
    /// flags) to a full from-scratch two-layer rebuild of scene B, AND the edit touched a
    /// strict SUBSET of the scene's chunks (proving it is genuinely incremental, not a
    /// disguised full rebuild). Mirrors
    /// `store.rs::incremental_rebuild_equals_full_rebuild_for_every_edit_kind`.
    #[test]
    fn incremental_two_layer_equals_full_rebuild_for_every_edit_kind() {
        let density = 16u32;

        // Three tools spread far apart in X so each occupies chunks the others don't
        // touch (clean localised edits). The interior "subject" box sits between two
        // static anchors that pin the composite extent (as in the dense net) — though
        // note a recentre shift does NOT invalidate the two-layer cache (chunk-local
        // frame), the anchors keep the setup parallel to the dense parity net.
        let anchor_lo =
            || incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density);
        let anchor_hi =
            || incr_tool_node(ShapeKind::Torus, [5, 5, 5], [120, 0, 0], MaterialChoice::Plain, density);
        let scene_a = Scene::from_nodes(vec![
            anchor_lo(),
            incr_tool_node(ShapeKind::Box, [5, 5, 5], [60, 0, 0], MaterialChoice::Wood, density),
            anchor_hi(),
        ]);

        let recolor = {
            let mut b = scene_a.clone();
            if let NodeContent::Tool { material, .. } = &mut b.root_node_mut(1).content {
                *material = MaterialChoice::Stone;
            }
            ("recolor", b)
        };
        let resize = {
            let mut b = scene_a.clone();
            let replacement =
                incr_tool_node(ShapeKind::Box, [3, 3, 3], [60, 0, 0], MaterialChoice::Wood, density);
            let slot = b.root_node_mut(1);
            slot.content = replacement.content;
            slot.transform = replacement.transform;
            ("resize", b)
        };
        let move_node = {
            let mut b = scene_a.clone();
            b.root_node_mut(1).transform = NodeTransform::from_blocks([70, 0, 0], density);
            ("move", b)
        };
        let add_node = {
            let mut b = scene_a.clone();
            b.add_node(incr_tool_node(
                ShapeKind::Box,
                [3, 3, 3],
                [90, 0, 0],
                MaterialChoice::Stone,
                density,
            ));
            ("add", b)
        };
        let remove_node = {
            let mut b = scene_a.clone();
            let interior_id = b.roots[1];
            b.remove_node(interior_id);
            ("remove", b)
        };
        let operation_flip = {
            // ADR 0017 (#73): flip the subject Union→Subtract — it becomes a cutter
            // (here carving nothing, so its own chunks empty out). The flip must be
            // localisable (the operation is part of the leaf fingerprint, so the diff
            // dirties exactly the leaf's AABB) and the dirtied chunks must
            // RE-CLASSIFY, not merely re-mesh: solid blocks become air.
            let mut b = scene_a.clone();
            b.root_node_mut(1).operation = document::scene::CombineOp::Subtract;
            ("operation-flip", b)
        };

        for (label, scene_b) in [recolor, resize, move_node, add_node, remove_node, operation_flip]
        {
            // Incremental: wholesale-build A, then apply the single edit and re-fill.
            let mut cache = TwoLayerResidentCache::enabled();
            let total_before = {
                let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
                cache.resident_len()
            };
            let (evicted, took_aabb_path) =
                apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
            assert!(
                took_aabb_path,
                "[{label}] this edit kind must be localisable (the AABB path, not clear())"
            );
            let incremental = resident_snapshot(&mut cache, &scene_b, density);

            // The full from-scratch rebuild for scene B (the truth).
            let full = full_two_layer_resident(&scene_b, density);

            assert_eq!(
                incremental, full,
                "[{label}] incremental two-layer cache (coarse layer + overlay + microblock \
                 maps + seam flags per covering chunk) MUST equal a full from-scratch rebuild \
                 of scene B — a stale chunk or a missed fresh chunk would differ here"
            );

            // Dirty-count-is-less: the edit evicted strictly fewer chunks than the scene's
            // total resident count (so it is genuinely incremental, not a full rebuild).
            let scene_chunks = total_before.max(full.len());
            assert!(
                evicted < scene_chunks,
                "[{label}] a localised edit must evict strictly FEWER chunks ({evicted}) than \
                 the scene's total ({scene_chunks}) — else it is a disguised full rebuild"
            );
        }
    }

    /// Perf probe (block-row-dedup regression guard): the full-band diameter re-measure —
    /// the query that fires when the layer band or grid changes. Before the ADR 0010 E5
    /// block-row dedup this was O(volume) (a coarse block stamped all `d²` of its voxel rows):
    /// 130ms @800³ → 127s @8000³, freezing the main thread. After, it is O(total blocks) and
    /// runs on the background diameter worker (never the UI thread). Reports wall-clock across
    /// four solid-cube edge lengths. Run:
    /// `cargo test --release widest_run_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn widest_run_scaling_probe() {
        use document::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [50i64, 125, 250, 500] {
            let edge = blocks * density as i64;
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let widest = streamed_widest_run_in_band(
                &TwoLayerStore::enabled(),
                &scene,
                density,
                0,
                edge as u32,
            );
            let elapsed = start.elapsed();
            println!("widest-run {edge}^3 vx full band: {widest:?} in {elapsed:?}");
        }
    }

    /// Perf probe (interior-elision win): time the LIVE two-layer build for a large
    /// SOLID sketch-extrude box — the path the app actually runs (NOT shot's dense
    /// `resolve_region` golden oracle). Before elision every interior block resolved
    /// per-voxel (O(volume)); after, interiors classify coarse (O(surface)). Reports the
    /// coarse/sculpted split + wall-clock. Run:
    /// `cargo test --release two_layer_sketch_box_build_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "perf probe — run in release with --nocapture"]
    fn two_layer_sketch_box_build_probe() {
        use document::sketch::{PlaneAxis, Sketch, SketchSolid};
        let density = 16u32;
        for blocks in [25i64, 50] {
            let edge = blocks * density as i64; // 400, then 800 voxels/axis (block-aligned)
            let extrude =
                SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, edge, edge), edge as u32);
            let scene = Scene::from_nodes(vec![Node::new(
                "Box",
                NodeContent::SketchTool { producer: extrude, material: MaterialChoice::Stone },
            )]);
            let start = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let elapsed = start.elapsed();
            let coarse: u64 = chunks
                .iter()
                .map(|(_, chunk)| chunk.coarse.iter().filter(|id| id.is_some()).count() as u64)
                .sum();
            let sculpted: u64 = chunks.iter().map(|(_, chunk)| chunk.microblocks.len() as u64).sum();
            println!(
                "sketch box {edge}³ voxels ({blocks} blocks/axis): two-layer build {:?} — \
                 {coarse} coarse + {sculpted} sculpted blocks over {} chunks",
                elapsed,
                chunks.len()
            );
        }
    }

    /// A localised recolor of one small far-flung node dirties only the handful of chunks
    /// that node occupies, NOT the whole scene — the two-layer analogue of
    /// `store.rs::localized_recolor_rebuilds_few_chunks`.
    #[test]
    fn incremental_two_layer_localized_recolor_evicts_few_chunks() {
        let density = 16u32;
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [1, 1, 1], [80, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        if let NodeContent::Tool { material, .. } = &mut scene_b.root_node_mut(1).content {
            *material = MaterialChoice::Stone;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "an in-place recolor must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);

        assert!(total >= 8, "the spread scene has many resident chunks ({total})");
        assert!(
            evicted * 2 < total,
            "a localised recolor of a small node must evict far fewer than half the chunks: \
             evicted {evicted} of {total}"
        );
        assert_eq!(incremental, full_two_layer_resident(&scene_b, density));
    }

    /// **Localisable move re-derives BOTH endpoints.** A moved node's dirty AABB spans its
    /// source AND destination (the `edit_aabb_since` union), so the two-layer cache vacates
    /// the source chunks and rebuilds the destination — and the result equals a full
    /// rebuild (no stale geometry left at the old location).
    #[test]
    fn incremental_two_layer_move_clears_source_and_fills_destination() {
        let density = 16u32;
        // A wide anchor keeps many chunks resident that the moved box never touches, so a
        // move touching a strict subset is meaningful.
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [9, 9, 9], [0, 0, 0], MaterialChoice::Stone, density),
            incr_tool_node(ShapeKind::Box, [2, 2, 2], [70, 0, 0], MaterialChoice::Wood, density),
        ]);
        let mut scene_b = scene_a.clone();
        scene_b.root_node_mut(1).transform = NodeTransform::from_blocks([85, 0, 0], density);

        let mut cache = TwoLayerResidentCache::enabled();
        let total = {
            let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
            cache.resident_len()
        };
        let (evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(took_aabb_path, "a move must be localisable");
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "a move must leave no stale geometry at the source and match a full rebuild"
        );
        assert!(evicted < total, "a move touches a strict subset ({evicted} of {total})");
    }

    /// **WHOLESALE FALLBACK — a density change re-derives everything.** A density change
    /// resizes every chunk's voxel extent, so `edit_aabb_since` returns `None` and the
    /// cache clears (belt-and-braces: `invalidate_aabb` also clears on a density mismatch).
    /// After the fallback the cache still equals a full rebuild at the NEW density.
    #[test]
    fn incremental_two_layer_density_change_falls_back_to_wholesale() {
        let density_a = 16u32;
        let density_b = 8u32;
        let scene = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Sphere, [5, 5, 5], [0, 0, 0], MaterialChoice::Stone, density_a),
        ]);

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene, density_a, 0);
        // The density-change diff: the same scene rebuilt at a different density has no
        // localisable AABB (the indices differ in density), so `edit_aabb_since` is None.
        let index_a = scene.build_leaf_spatial_index(density_a);
        let index_b = scene.build_leaf_spatial_index(density_b);
        assert!(
            index_b.edit_aabb_since(&index_a).is_none(),
            "a density change must have no localisable edit AABB (the wholesale fallback)"
        );
        cache.clear();
        let incremental = resident_snapshot(&mut cache, &scene, density_b);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene, density_b),
            "after the density-change wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// **WHOLESALE FALLBACK — editing an unbounded (region-spanning) producer.** Editing a
    /// `DebugClouds` Part (its dirty region is "everywhere", `edit_aabb_since` returns
    /// `None`) forces a wholesale clear; the rebuilt cache still equals a full rebuild.
    /// This is the "unboundable-producer edit falls back to wholesale" acceptance case.
    #[test]
    fn incremental_two_layer_cloud_edit_falls_back_to_wholesale() {
        use document::scene::Part;
        let density = 16u32;
        let cloud = |seed: u32| {
            let mut node = Node::new("Clouds", NodeContent::Part(Part::DebugClouds { seed }));
            node.transform = NodeTransform::from_blocks([0, 0, 0], density);
            node
        };
        let scene_a = Scene::from_nodes(vec![
            incr_tool_node(ShapeKind::Box, [3, 3, 3], [0, 0, 0], MaterialChoice::Stone, density),
            cloud(7),
        ]);
        // Edit the cloud's seed (a region-spanning content change; root index 1).
        let mut scene_b = scene_a.clone();
        if let NodeContent::Part(Part::DebugClouds { seed }) =
            &mut scene_b.root_node_mut(1).content
        {
            *seed = 42;
        }

        let mut cache = TwoLayerResidentCache::enabled();
        let _ = cache.resident_two_layer_chunks(&scene_a, density, 0);
        let (_evicted, took_aabb_path) =
            apply_two_layer_incremental_edit(&mut cache, &scene_a, &scene_b, density);
        assert!(
            !took_aabb_path,
            "editing a region-spanning Part must take the wholesale fallback, not the AABB path"
        );
        let incremental = resident_snapshot(&mut cache, &scene_b, density);
        assert_eq!(
            incremental,
            full_two_layer_resident(&scene_b, density),
            "after the cloud-edit wholesale rebuild the cache must equal a full rebuild"
        );
    }

    /// Wholesale-build timing probe across a WIDE object-count range (#66; the #63 lesson —
    /// a small N hides a super-linear asymptote). Not a correctness gate: run manually with
    /// `cargo test --release --lib wholesale_build_scaling_probe -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing probe, run manually with --release --ignored --nocapture"]
    fn wholesale_build_scaling_probe() {
        let density = 16u32;
        for boxes_per_axis in [5i64, 12, 22] {
            let mut nodes = Vec::new();
            for grid_z in 0..boxes_per_axis {
                for grid_y in 0..boxes_per_axis {
                    for grid_x in 0..boxes_per_axis {
                        nodes.push(make_tool_density(
                            ShapeKind::Box,
                            [grid_x * 4, grid_y * 4, grid_z * 4],
                            MaterialChoice::Stone,
                            density,
                            2,
                        ));
                    }
                }
            }
            let object_count = boxes_per_axis.pow(3);
            let scene = Scene::from_nodes(nodes);
            let leaves_started = std::time::Instant::now();
            let leaves = scene.leaf_producers(density);
            let leaves_elapsed = leaves_started.elapsed();
            let (min_chunk, max_chunk) = scene.covering_chunk_range(density).unwrap();
            let chunk_count = (0..3)
                .map(|axis| (max_chunk[axis] - min_chunk[axis] + 1) as i64)
                .product::<i64>();
            let broadphase_started = std::time::Instant::now();
            let broadphase = leaf_edit_broadphase(&leaves, density);
            let broadphase_elapsed = broadphase_started.elapsed();
            std::hint::black_box(&broadphase);
            let build_started = std::time::Instant::now();
            let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, density, 0);
            let build_elapsed = build_started.elapsed();
            eprintln!(
                "N={object_count} objects, {chunk_count} covering chunks: leaf hoist \
                 {leaves_elapsed:?}, edit-broadphase BVH rebuild {broadphase_elapsed:?}, \
                 wholesale build {build_elapsed:?} ({} chunks emitted)",
                chunks.len()
            );
        }
    }

    /// The capability OFF (the default): the resident cache is a no-op — it never fills and
    /// `resident_two_layer_chunks` returns empty, so a caller falls back to the dense path.
    #[test]
    fn incremental_two_layer_capability_off_is_noop() {
        let density = 16u32;
        let scene = shape_scene(ShapeKind::Sphere, density);
        let mut cache = TwoLayerResidentCache::default();
        assert!(!cache.is_enabled());
        assert!(cache.resident_two_layer_chunks(&scene, density, 0).is_empty());
        assert_eq!(cache.resident_len(), 0);
    }
