    use super::*;
    use voxel_core::core_geom::MaterialChoice;
    use voxel_core::voxel::{ShapeKind};
    use document::voxel::{SdfShape};

    /// Resolve a small cylinder and round-trip it through `.vox`, asserting the
    /// voxel count and dimensions survive (Z-up, no axis swap).
    ///
    /// Corner-anchoring: `from_grid`'s decode (`round(world + floor(dim/2) − 0.5)`)
    /// expects the grid in the RECENTRED frame (low corner `−floor(dim/2)`), which is
    /// what production produces. So resolve through a one-node scene (recentred), NOT
    /// the bare producer grid (whose low corner is 0).
    #[test]
    fn vox_round_trip_matches_grid() {
        let scene = document::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [80, 16, 80],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);
        assert!(grid.occupied_count() > 0, "expected a non-empty grid");

        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]));
        assert_eq!(export.model_count(), 1, "small grid is a single model");
        assert_eq!(export.voxel_count(), grid.occupied_count());

        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse our file");

        assert_eq!(parsed.models.len(), 1);
        let model = &parsed.models[0];
        // Z-up: vox size = (our X, our Y, our Z) — no swap. Grid 80×16×80 → vox 80×16×80.
        let [gx, gy, gz] = grid.dimensions;
        assert_eq!(model.size.x, gx);
        assert_eq!(model.size.y, gy);
        assert_eq!(model.size.z, gz);
        // Every occupied voxel was written exactly once.
        assert_eq!(model.voxels.len(), grid.occupied_count());
        // All coordinates are within the model's declared size.
        for voxel in &model.voxels {
            assert!((voxel.x as u32) < model.size.x);
            assert!((voxel.y as u32) < model.size.y);
            assert!((voxel.z as u32) < model.size.z);
        }
    }

    /// The atomic write's post-conditions (findings 2/3/4): a successful `write` leaves
    /// the final file present and NO stray temp behind, and the unique temp name it picks
    /// is a dot-prefixed sibling in the SAME directory (so the rename stays on one
    /// filesystem). The Windows rename→copy fallback on a share-violating destination is
    /// not portably simulatable, so it is reasoned in `write`'s doc comment rather than
    /// exercised here.
    #[test]
    fn atomic_write_leaves_no_temp_and_temp_is_a_dir_sibling() {
        let scene = document::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [16, 16, 16],
                size_measurements: None,
                voxels_per_block: 8,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(8), 8, 0);
        let export = VoxExport::from_grid(
            &grid,
            VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]),
        );

        let dir = std::env::temp_dir()
            .join(format!("voxel_worker_write_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create the test dir");
        let final_path = dir.join("model.vox");

        // The unique temp name is a dot-prefixed sibling in the same directory.
        let temp = VoxExport::unique_temp_path(&final_path);
        assert_eq!(temp.parent(), Some(dir.as_path()), "temp is a sibling of the final file");
        assert!(
            temp.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with('.') && n.ends_with(".tmp")),
            "temp name is dot-prefixed and .tmp-suffixed: {temp:?}"
        );

        // A successful write leaves the final file and NO temp behind.
        let bytes = export.write(&final_path).expect("write succeeds");
        assert!(bytes > 0, "wrote a non-empty file");
        assert!(final_path.exists(), "the final file is present");
        let leftover_temps: Vec<_> = std::fs::read_dir(&dir)
            .expect("read the test dir")
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".tmp"))
            })
            .collect();
        assert!(
            leftover_temps.is_empty(),
            "a successful write leaves no temp file behind, found: {leftover_temps:?}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Z-up convention pin: an ASYMMETRIC shape (tall in Z) puts its vertical extent
    /// on vox-Z with NO swap. A cylinder 2×2×5 blocks (5 blocks tall along the +Z
    /// axis) must export to a vox model whose Z size is the largest — proving the
    /// vertical axis lands on vox-Z directly, not relocated to vox-Y.
    #[test]
    fn vox_export_puts_vertical_on_vox_z_no_swap() {
        let scene = document::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Cylinder,
                size_voxels: [2 * 16, 2 * 16, 5 * 16],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);
        let [gx, gy, gz] = grid.dimensions; // 32 × 32 × 80
        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]));
        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse our file");
        let model = &parsed.models[0];
        // No swap: vox size matches our (X, Y, Z) exactly, with Z the tallest axis.
        assert_eq!([model.size.x, model.size.y, model.size.z], [gx, gy, gz]);
        assert!(
            model.size.z > model.size.x && model.size.z > model.size.y,
            "the tall (Z) axis must land on vox-Z (got {:?})",
            (model.size.x, model.size.y, model.size.z)
        );
    }

    /// A grid wider than 256 voxels on an axis must split into multiple models,
    /// never silently truncate.
    #[test]
    fn vox_splits_models_over_256() {
        // 17 blocks × 16 vx = 272 > 256 on X. Resolve through a scene (recentred
        // frame) so `from_grid`'s corner-anchored decode lands every voxel in range.
        let scene = document::scene::Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [272, 16, 16],
                size_measurements: None,
                voxels_per_block: 16,
                wall_blocks: 1,
            },
            voxel_core::core_geom::MaterialChoice::Stone,
        );
        let grid = scene.resolve_region(scene.full_extent_blocks(16), 16, 0);

        let export = VoxExport::from_grid(&grid, VoxExport::block_palette_from_active(MaterialChoice::Stone, [200, 200, 200, 255]));
        assert!(export.model_count() >= 2, "272-wide grid should split");
        // No voxels lost across the split.
        assert_eq!(export.voxel_count(), grid.occupied_count());

        let bytes = export.to_bytes();
        let parsed = dot_vox::load_bytes(&bytes).expect("dot_vox should parse split file");
        let total: usize = parsed.models.iter().map(|m| m.voxels.len()).sum();
        assert_eq!(total, grid.occupied_count());
        for model in &parsed.models {
            assert!(model.size.x <= 256 && model.size.y <= 256 && model.size.z <= 256);
        }
    }

    // ===== Issue #20 S6d: region-scoped `.vox` export ============================

    use evaluation::chunk_cache::ChunkResolveCache;
    use document::scene::{Node, NodeContent, Scene};
    use document::voxel::GeometryParams;

    /// Parse a `.vox` byte stream into a per-model SORTED multiset of
    /// `(size, voxel (x, y, z, color))`, so two exports compare equal regardless of
    /// per-model voxel emission ORDER (chunk-iteration order vs monolithic stamp
    /// order) — a MagicaVoxel reader treats reordered voxels as the same model.
    type ModelVoxelSet = std::collections::BTreeSet<(u8, u8, u8, u8)>;
    type ModelSets = std::collections::BTreeSet<([u32; 3], ModelVoxelSet)>;
    fn parsed_model_sets(bytes: &[u8]) -> ModelSets {
        let parsed = dot_vox::load_bytes(bytes).expect("dot_vox should parse our file");
        parsed
            .models
            .iter()
            .map(|model| {
                let size = [model.size.x, model.size.y, model.size.z];
                let voxels = model
                    .voxels
                    .iter()
                    .map(|v| (v.x, v.y, v.z, v.i))
                    .collect::<std::collections::BTreeSet<_>>();
                (size, voxels)
            })
            .collect()
    }

    /// Parse a `.vox` byte stream into a per-model **last-writer-wins** map
    /// `(x, y, z) -> colour` — the occupancy a MagicaVoxel reader actually renders. The
    /// dense-path export writes DUPLICATE voxels at positions where leaves overlap (the
    /// dense occupied Vec keeps both leaves' entries; the LATER one in document order is
    /// the resolved winner a reader shows); the streamed two-layer export is one-id-per-
    /// cell (Union later-wins resolved). Reducing both to last-writer-per-coord compares
    /// the TRUE resolved file: for every non-overlapping scene each coord has one writer,
    /// so this is identical to [`parsed_model_sets`]; only genuine overlap differs, and
    /// there the last-writer map is the correct comparison (ADR 0010 parity-gate canonical
    /// form, mirroring `two_layer_store.rs::resolved_occupancy_set`).
    type ModelLastWriter = std::collections::BTreeMap<(u8, u8, u8), u8>;
    type ModelLastWriterSets = std::collections::BTreeSet<([u32; 3], Vec<((u8, u8, u8), u8)>)>;
    fn parsed_model_last_writer_sets(bytes: &[u8]) -> ModelLastWriterSets {
        let parsed = dot_vox::load_bytes(bytes).expect("dot_vox should parse our file");
        parsed
            .models
            .iter()
            .map(|model| {
                let size = [model.size.x, model.size.y, model.size.z];
                // Voxels are in write order; the LAST entry at a coord wins (insert
                // overwrites), reproducing the MagicaVoxel reader's resolved occupancy.
                let mut last: ModelLastWriter = std::collections::BTreeMap::new();
                for v in &model.voxels {
                    last.insert((v.x, v.y, v.z), v.i);
                }
                (size, last.into_iter().collect::<Vec<_>>())
            })
            .collect()
    }

    fn assert_region_vox_export_equals_whole_grid(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        // Whole-grid export: assemble the monolithic region grid, export via the
        // existing `from_grid` path.
        let region = scene.full_extent_blocks(vpb);
        let whole = scene.resolve_region(region, vpb, 0);
        let whole_export = VoxExport::from_grid(&whole, rgba);

        // Region export: from the per-chunk grids, no monolithic grid assembled.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        assert_eq!(
            region_export.voxel_count(),
            whole_export.voxel_count(),
            "[{label}] region export voxel count must equal whole-grid"
        );
        assert_eq!(
            region_export.model_count(),
            whole_export.model_count(),
            "[{label}] region export model count must equal whole-grid"
        );
        assert_eq!(
            parsed_model_sets(&region_export.to_bytes()),
            parsed_model_sets(&whole_export.to_bytes()),
            "[{label}] region export model-set (sizes + voxels) must equal whole-grid"
        );
    }

    fn shape_scene(kind: ShapeKind, vpb: u32, size: [u32; 3]) -> Scene {
        Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [size[0] * vpb, size[1] * vpb, size[2] * vpb],
                size_measurements: None,
                voxels_per_block: vpb,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        )
    }

    /// The region-scoped `.vox` export equals the whole-grid export for the bounded
    /// SDF shapes (single-model cases).
    #[test]
    fn region_vox_export_equals_whole_grid_for_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16, [5, 5, 5]);
            assert_region_vox_export_equals_whole_grid(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// The region-scoped `.vox` export equals the whole-grid export even when the
    /// region is wide enough to FORCE a 256-split into multiple models — proving the
    /// per-chunk bucketing tiles identically to the monolithic path.
    #[test]
    fn region_vox_export_equals_whole_grid_when_split_over_256() {
        // 20 blocks × 16 = 320 voxels > 256 on X → splits into 2 models.
        let scene = shape_scene(ShapeKind::Box, 16, [20, 1, 1]);
        assert_region_vox_export_equals_whole_grid(&scene, 16, "wide-box-split");
    }

    /// A multi-leaf demo scene (spans several chunks across leaves) exports
    /// identically through the region path.
    #[test]
    fn region_vox_export_equals_whole_grid_for_demo_scene() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
        ]);
        assert_region_vox_export_equals_whole_grid(&scene, vpb, "demo-scene");
    }

    // ===== Issue #20 Step 2: far-offset export ===================================

    /// Build a two-node scene whose composite is centred FAR from the world origin:
    /// one node at the origin and one node `offset_blocks` away on X. The composite
    /// centre lands at the midpoint, so each node sits ~`offset/2 × vpb` voxels from
    /// the recentred frame's origin. The second node is placed `offset_blocks` blocks
    /// away on X.
    fn far_offset_two_box_scene(vpb: u32, offset_blocks: i64) -> Scene {
        let make_box = |offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, vpb);
            let mut node = Node::new("Box", NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            make_box([0, 0, 0], MaterialChoice::Stone),
            make_box([offset_blocks, 0, 0], MaterialChoice::Wood),
        ]);
        scene.voxels_per_block = vpb;
        scene
    }

    /// **The rewired export is behaviour-equivalent to the old monolithic export, far
    /// from the origin (issue #20 Step 2).** The live export button now routes through
    /// `ChunkResolveCache::vox_export` instead of a dense whole-region resolve + `from_grid`. This
    /// proves the rewiring is safe at far offset: for a scene whose composite is
    /// centred ~250,000 blocks out (4e6 voxels — well into the f32 large-magnitude
    /// regime), the region-scoped export's model SET (sizes + per-model voxels) equals
    /// the old whole-grid export's, AND both keep the full voxel count (the per-chunk
    /// ground truth). So the wiring change is a true no-op on the written file.
    ///
    /// NOTE (finding, issue #20 Low #1): routing through `vox_export` does NOT make a
    /// genuinely region-WIDE far scene more accurate than the monolithic path. Both
    /// bucket into the region-relative `[0, grid_x)` frame, so both add `half_x` (≈ the
    /// region half-width) in f32; once the region exceeds ~2^24 voxels on an axis the
    /// voxel-centre `.5` is unrepresentable and BOTH paths collapse identically (the
    /// exports stay model-set-equal). The f32-`.5` loss is inherent to the f32
    /// `world_position` at large magnitude, not to which assembly path is used. The
    /// rewiring's value is the Step-4 decoupling from the monolithic grid, not a
    /// far-offset accuracy gain.
    #[test]
    fn far_offset_region_export_equals_monolithic() {
        let vpb = 16u32;
        // 500,000-block separation → composite centred ~250,000 blocks out → each box
        // ~4e6 voxels from origin. Region grid stays under 2^24 voxels wide so the full
        // voxel set survives (the per-chunk ground truth is matched exactly).
        let scene = far_offset_two_box_scene(vpb, 500_000);
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        // Ground-truth voxel count (frame-independent): the per-chunk assembly rebases
        // each chunk in i64, so its occupied count is the TRUE distinct-voxel count.
        let expected_voxels = scene.resolve_region_via_chunks(vpb, 0).occupied_count();

        // New (region-scoped) path — what the export button now calls.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(&scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);
        assert_eq!(
            region_export.voxel_count(),
            expected_voxels,
            "region export must keep every voxel at this far offset"
        );

        // Old (monolithic) path the button used before.
        let region = scene.full_extent_blocks(vpb);
        let whole = scene.resolve_region(region, vpb, 0);
        let monolithic_export = VoxExport::from_grid(&whole, rgba);

        // The rewiring is a no-op on the written file: same model set (sizes + voxels),
        // same counts. (Per-model voxel ORDER may differ — chunk-iteration vs
        // monolithic stamp order — which a MagicaVoxel reader treats as the same model.)
        assert_eq!(
            region_export.voxel_count(),
            monolithic_export.voxel_count(),
            "rewired export voxel count must equal the old monolithic export far out"
        );
        assert_eq!(
            parsed_model_sets(&region_export.to_bytes()),
            parsed_model_sets(&monolithic_export.to_bytes()),
            "rewired export model-set must equal the old monolithic export far out"
        );
    }

    /// The far-offset region export, once parsed and re-read, round-trips to the same
    /// total voxel count the per-chunk ground truth holds — exercising the full
    /// build → serialise → `dot_vox::load_bytes` path the export button drives (minus
    /// the file dialog), so the wiring is verified end to end headlessly.
    #[test]
    fn far_offset_region_export_round_trips_full_voxel_set() {
        let vpb = 16u32;
        let scene = far_offset_two_box_scene(vpb, 500_000);
        let rgba = VoxExport::block_palette_from_active(MaterialChoice::Stone, [132, 126, 118, 255]);

        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(&scene, vpb, 0);
        let region_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        let parsed = dot_vox::load_bytes(&region_export.to_bytes())
            .expect("dot_vox should parse the far-offset export");

        let expected_voxels = scene.resolve_region_via_chunks(vpb, 0).occupied_count();
        let total: usize = parsed.models.iter().map(|model| model.voxels.len()).sum();
        assert_eq!(
            total, expected_voxels,
            "far-offset export must round-trip every voxel"
        );

        // The two far-separated boxes occupy different 256-tiles on X, so the parsed
        // file must contain at least two non-empty models.
        let nonempty_models = parsed
            .models
            .iter()
            .filter(|model| !model.voxels.is_empty())
            .count();
        assert!(
            nonempty_models >= 2,
            "the two far-separated boxes must land in >=2 distinct tiles (got {nonempty_models})"
        );
    }

    // ===== ADR 0010 E4: cacheless STREAMING `.vox` export ========================

    use voxel_core::core_geom::MaterialChoice as Mat;
    use evaluation::two_layer_store::{stream_vox_occupancy, TwoLayerStore};

    /// Build the `.vox` export by STREAMING the cacheless two-layer evaluator (coarse
    /// `d³` fast-fill + boundary per-voxel) — the E4 path the export button drives.
    fn streamed_vox_export(scene: &Scene, vpb: u32, rgba: BlockPaletteColors) -> VoxExport {
        let store = TwoLayerStore::enabled();
        let mut chunks: Vec<Vec<voxel_core::voxel::Voxel>> = Vec::new();
        let dims = stream_vox_occupancy(&store, scene, vpb, |chunk| chunks.push(chunk))
            .expect("the two-layer capability is enabled");
        VoxExport::from_region_voxel_chunks(dims, chunks, rgba)
    }

    /// **THE E4 `.vox` PARITY GATE:** the streamed export's written `.vox` (model set =
    /// sizes + per-voxel `(x, y, z, colour)`) is IDENTICAL to today's dense-path region
    /// export, for the gated scene. Mirrors
    /// `assert_region_vox_export_equals_whole_grid` on the streaming path.
    fn assert_streamed_vox_export_equals_dense(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);

        // Dense path (today's export): per-chunk `bound_region_occupied` → `from_region_voxels`.
        let mut cache = ChunkResolveCache::new();
        let (dims, occupied) = cache.bound_region_occupied(scene, vpb, 0);
        let dense_export = VoxExport::from_region_voxels(dims, occupied, rgba);

        // Streamed path (E4): the cacheless two-layer evaluator.
        let streamed_export = streamed_vox_export(scene, vpb, rgba);

        assert_eq!(
            streamed_export.model_count(),
            dense_export.model_count(),
            "[{label}] streamed export model count must equal the dense-path export"
        );
        // The faithful parity comparison is the RESOLVED occupancy a MagicaVoxel reader
        // renders — last-writer-per-coord (position + palette colour). For every
        // non-overlapping scene each coord has one writer, so this is bit-identical to
        // the raw per-voxel set; only genuine leaf overlap differs (the dense file keeps
        // duplicate entries there, the streamed file is resolved), and the last-writer
        // map is the correct comparison (ADR 0010 parity-gate canonical form).
        let streamed_bytes = streamed_export.to_bytes();
        let dense_bytes = dense_export.to_bytes();
        assert_eq!(
            parsed_model_last_writer_sets(&streamed_bytes),
            parsed_model_last_writer_sets(&dense_bytes),
            "[{label}] streamed export resolved occupancy (last-writer position + palette \
             colour) must be IDENTICAL to the dense-path `.vox` export"
        );
        // The streamed export is one-id-per-cell: it writes NO duplicate voxels, so its
        // raw voxel count equals its resolved count (the dense path over-counts at
        // overlaps; the streamed path never does — that is the elision win).
        let streamed_resolved_count: usize = parsed_model_last_writer_sets(&streamed_bytes)
            .iter()
            .map(|(_, last)| last.len())
            .sum();
        assert_eq!(
            streamed_export.voxel_count(),
            streamed_resolved_count,
            "[{label}] the streamed export must write one voxel per resolved cell (no \
             duplicate-at-overlap entries)"
        );
    }

    /// **THE STREAMED-SINK PEAK-MEMORY PROOF (ADR 0010 E4).** The live export button now
    /// buckets each streamed chunk DIRECTLY into a [`VoxExportBuilder`] then drops it
    /// (peak = O(one chunk + output buffers)), instead of accumulating every chunk into a
    /// `Vec<Vec<Voxel>>` before one `from_region_voxel_chunks` conversion (peak =
    /// O(all voxels)). This asserts the two produce a BYTE-IDENTICAL `.vox` for a
    /// multi-chunk scene: both drive the SAME `stream_vox_occupancy` (identical chunk
    /// order), and the incremental builder IS the core `from_region_voxel_chunks` flattens
    /// into — so the memory fix is a pure no-op on the written file, down to voxel emission
    /// order and palette bytes. The accumulate-then-convert path is kept here as the oracle.
    fn assert_streamed_builder_matches_accumulated(scene: &Scene, vpb: u32, label: &str) {
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);
        let store = TwoLayerStore::enabled();

        // Incremental streaming sink (the live button): bucket each chunk, then drop it —
        // only one chunk's voxels are ever resident.
        let region_dimensions = scene.placed_region_dimensions(vpb);
        let mut builder = VoxExportBuilder::new(region_dimensions, rgba);
        let dims_stream =
            stream_vox_occupancy(&store, scene, vpb, |chunk| builder.ingest_chunk(&chunk))
                .expect("the two-layer capability is enabled");
        assert_eq!(
            dims_stream, region_dimensions,
            "[{label}] the builder must be pre-created with the SAME dims the stream emits"
        );
        let streamed = builder.finish();

        // Accumulate-then-convert ORACLE (the retired path): push every chunk into a
        // Vec<Vec<Voxel>> before converting — O(all voxels) peak.
        let mut accumulated_chunks: Vec<Vec<voxel_core::voxel::Voxel>> = Vec::new();
        stream_vox_occupancy(&store, scene, vpb, |chunk| accumulated_chunks.push(chunk))
            .expect("the two-layer capability is enabled");
        let accumulated =
            VoxExport::from_region_voxel_chunks(region_dimensions, accumulated_chunks, rgba);

        assert_eq!(
            streamed.voxel_count(),
            accumulated.voxel_count(),
            "[{label}] streamed-sink voxel count must equal the accumulate-then-convert path"
        );
        assert_eq!(
            streamed.model_count(),
            accumulated.model_count(),
            "[{label}] streamed-sink model count must equal the accumulate-then-convert path"
        );
        assert_eq!(
            streamed.to_bytes(),
            accumulated.to_bytes(),
            "[{label}] the streamed-sink `.vox` bytes must be IDENTICAL to the \
             accumulate-then-convert export (the peak-memory fix is a no-op on the file)"
        );
    }

    #[test]
    fn streamed_builder_matches_accumulated_for_multi_chunk_scenes() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        // A multi-leaf scene spanning several chunks across leaves.
        let demo = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], Mat::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], Mat::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], Mat::Plain),
        ]);
        assert_streamed_builder_matches_accumulated(&demo, vpb, "demo-scene");

        // A wide box forcing a 256-split (multiple models across many chunks).
        let wide = shape_scene(ShapeKind::Box, vpb, [20, 1, 1]);
        assert_streamed_builder_matches_accumulated(&wide, vpb, "wide-box-split");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_shapes() {
        for kind in [
            ShapeKind::Sphere,
            ShapeKind::Cylinder,
            ShapeKind::Tube,
            ShapeKind::Torus,
            ShapeKind::Box,
        ] {
            let scene = shape_scene(kind, 16, [5, 5, 5]);
            assert_streamed_vox_export_equals_dense(&scene, 16, &format!("{kind:?}"));
        }
    }

    /// FLAT / odd-sized shapes (a 1-block axis straddling two chunks) stream identically.
    #[test]
    fn streamed_vox_export_equals_dense_for_flat_and_odd_shapes() {
        for kind in [ShapeKind::Cylinder, ShapeKind::Sphere, ShapeKind::Torus] {
            for size in [[5u32, 1, 5], [3, 1, 3], [5, 3, 5], [1, 1, 1]] {
                let scene = shape_scene(kind, 16, size);
                assert_streamed_vox_export_equals_dense(
                    &scene,
                    16,
                    &format!("{kind:?} {size:?}"),
                );
            }
        }
    }

    /// A wide box forcing a 256-split streams the same multi-model set as the dense path.
    #[test]
    fn streamed_vox_export_equals_dense_when_split_over_256() {
        let scene = shape_scene(ShapeKind::Box, 16, [20, 1, 1]);
        assert_streamed_vox_export_equals_dense(&scene, 16, "wide-box-split");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_demo_scene() {
        let vpb = 16u32;
        let make_tool = |kind, offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool(ShapeKind::Sphere, [0, 0, 0], Mat::Stone),
            make_tool(ShapeKind::Box, [8, 0, 0], Mat::Wood),
            make_tool(ShapeKind::Torus, [0, 0, 6], Mat::Plain),
        ]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "demo-scene");
    }

    #[test]
    fn streamed_vox_export_equals_dense_for_demo_village() {
        use document::scene::DefId;
        let vpb = 16u32;
        let house_def_id = DefId(1);
        let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(kind, size, 1, vpb);
            let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let instance = |name: &str, offset: [i64; 3]| {
            let mut node = Node::new(name, NodeContent::Instance(house_def_id));
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let mut scene = Scene::from_nodes(vec![
            instance("House 1", [0, 0, 0]),
            instance("House 2", [6, 0, 0]),
            instance("House 3", [12, 0, 0]),
            instance("House 4", [18, 0, 0]),
        ]);
        scene.add_definition(
            house_def_id,
            "House".to_string(),
            vec![
                tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], Mat::Stone),
                tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], Mat::Wood),
            ],
        );
        assert_streamed_vox_export_equals_dense(&scene, vpb, "demo-village");
    }

    /// A sketch-revolve solid (always classifies BOUNDARY — its polygon fill is not a
    /// coarse box) streams its per-voxel boundary path identically to the dense export.
    #[test]
    fn streamed_vox_export_equals_dense_for_sketch_solid() {
        use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
        let vpb = 16u32;
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, 360);
        let node = Node::new(
            "Revolve",
            NodeContent::SketchTool {
                producer,
                material: Mat::Stone,
            },
        );
        let scene = Scene::from_nodes(vec![node]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "sketch-revolve");
    }

    /// An OVERLAP multi-material scene (two boxes of different materials overlapping):
    /// the overlap blocks classify BOUNDARY (Union later-wins material is per-voxel), so
    /// each voxel's `.vox` palette colour must match the dense export through the palette.
    #[test]
    fn streamed_vox_export_equals_dense_for_overlap_multi_material() {
        let vpb = 16u32;
        let make_tool = |offset: [i64; 3], material| {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, vpb);
            let mut node = Node::new("Box", NodeContent::Tool { shape, material });
            node.transform = document::scene::NodeTransform::from_blocks(offset, vpb);
            node
        };
        let scene = Scene::from_nodes(vec![
            make_tool([0, 0, 0], Mat::Stone),
            make_tool([2, 0, 0], Mat::Wood),
        ]);
        assert_streamed_vox_export_equals_dense(&scene, vpb, "overlap-multi-material");
    }

    /// **6M-CAP DISSOLUTION (the E4 headline):** an 800×800-revolve-class solid box —
    /// 50³ blocks @ d16 = 800³ voxels, whose dense whole-region count (~5.1e8) blows the
    /// 6M `MAX_GRID_VOXELS` cap — EXPORTS SUCCESSFULLY via the streaming path. We assert
    /// (a) the dense single-shape guard WOULD reject it, and (b) the streamed export
    /// produces the full surface+interior occupancy without a whole-region densify.
    #[test]
    fn streamed_vox_export_dissolves_6m_cap_on_large_solid() {
        let vpb = 16u32;
        let blocks = 50u32;
        let shape = SdfShape::from_blocks(ShapeKind::Box, [blocks, blocks, blocks], 1, vpb);
        // (a) The dense single-shape cap WOULD reject this scene outright.
        assert!(
            shape.exceeds_voxel_cap(vpb),
            "the large solid must exceed the dense 6M cap to prove the point"
        );
        let node = Node::new("BigBox", NodeContent::Tool { shape, material: Mat::Stone });
        let scene = Scene::from_nodes(vec![node]);
        let rgba = VoxExport::block_palette_from_active(Mat::Stone, [132, 126, 118, 255]);

        // (b) The streamed export succeeds — no whole-region grid is ever built.
        let export = streamed_vox_export(&scene, vpb, rgba);

        // The export holds the FULL occupancy (surface shell + coarse interior fast-fill).
        // 800³ voxels = 5.1e8; the dense path could never assemble it. (The `.vox` 256
        // cap tiles the 800-axis into ceil(800/256)=4 models per axis.)
        let region_voxels = (blocks as u64 * vpb as u64).pow(3);
        assert_eq!(
            export.voxel_count() as u64,
            region_voxels,
            "the streamed export must hold the FULL solid occupancy (surface + interior \
             coarse fast-fill), far past the dense 6M cap"
        );
        // The file parses and tiles correctly (800 > 256 on every axis → 4³ = 64 models).
        let parsed = dot_vox::load_bytes(&export.to_bytes())
            .expect("dot_vox should parse the large streamed export");
        let total: u64 = parsed.models.iter().map(|m| m.voxels.len() as u64).sum();
        assert_eq!(total, region_voxels, "every voxel must survive the 256-split tiling");
        for model in &parsed.models {
            assert!(model.size.x <= 256 && model.size.y <= 256 && model.size.z <= 256);
        }
    }
