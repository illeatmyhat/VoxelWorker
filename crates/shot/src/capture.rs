//! The capture driver: `--replay` scene assembly plus the big `run_capture`
//! GPU-setup → render → PNG save / export routine.

use display::block_texture::LoadedMaterial;
use voxel_worker::block_palette::PaletteHost;
use work::workers::scan::{run_auto_scan_blocking, FaceResolver};
use voxel_worker::{
    create_depth_view, create_msaa_color_view, procedural_material_average_color, render_frame,
    run_egui_frame, AppCore, CuboidMeshRenderer, EguiPaintBridge, FrameOverlays, GpuContext,
    InfiniteGridRenderer, LayerBand, LayerRange, MaterialSource, Node, NodeContent, NodePath,
    OrbitCamera, PanelState, PlacementGhost, PlacementGhostRenderer, VoxelBody, Point,
    PointsRenderer, RegionBlocks, Scene, SceneGridRenderer, SdfShape, SelectedOperandGhostRenderer,
    TransformGizmoRenderer, ViewCubeRenderer, ViewMode, VoxExport, VoxelGrid,
    COLOR_TARGET_FORMAT, PLACEMENT_GHOST_TINT,
};

use crate::demos::{
    build_demo_groups, build_demo_mixed_material, build_demo_overlap, build_demo_scene,
    build_demo_sketch_box, build_demo_sketch_extrude, build_demo_sketch_revolve,
    build_demo_buried_cutter, build_demo_child_booleans,
    build_demo_cutter_def, build_demo_group_subtract, build_demo_intersect, build_demo_subtract,
    build_demo_window_fixture,
    build_demo_two_material,
    build_demo_village,
    build_demo_village_far,
    build_far_offset_scene, file_stem_of, resolve_demo_stem, FAR_OFFSET_BLOCKS,
    FAR_SCENE_BASE_BLOCKS,
};
use crate::options::ShotOptions;

/// `--replay` (ADR 0003 Phase C, slice C3): **replay-script -> Scene**.
///
/// The script at `replay_path` is **newline-delimited JSON**: one
/// [`voxel_worker::Intent`] per non-empty line. Each line is parsed with
/// `serde_json::from_str::<Intent>` and applied IN ORDER, via
/// [`AppCore::apply_intent`], to the default seed scene (the same base the windowed
/// app starts from — `voxel_worker::default_replay_seed_scene`). Blank /
/// whitespace-only lines are skipped. The returned [`Scene`] is the post-replay
/// document; the caller flows it into the SAME render path (resolve -> offscreen
/// render -> write PNG to `--out`).
///
/// File IO lives here; the parse + apply core is `voxel_worker::replay_intent_script`
/// (lib-tested without a GPU). On a read error, or a JSON parse error on any line,
/// returns `Err` with a clear message naming the offending line — `run_capture`
/// prints it and exits non-zero (no panic).
fn build_scene_from_replay(replay_path: &std::path::Path) -> Result<Scene, String> {
    let script = std::fs::read_to_string(replay_path).map_err(|error| {
        format!("--replay: failed to read '{}': {error}", replay_path.display())
    })?;
    voxel_worker::replay_intent_script(&script)
        .map_err(|error| format!("--replay: '{}': {error}", replay_path.display()))
}

pub(crate) async fn run_capture(options: ShotOptions) {
    assert!(options.width > 0 && options.height > 0, "capture size must be non-zero");

    // Fully headless: no surface, no window.
    let gpu = GpuContext::new(None).await;

    // Offscreen colour target. Same sRGB format as the windowed surface so the
    // screenshot is identical to the window; COPY_SRC so we can read it back.
    let capture_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless capture color"),
        size: wgpu::Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // 4× MSAA depth + colour at the offscreen size. The 3D pass renders into the
    // multisampled colour texture and resolves into `capture_texture` (the single
    // -sample COPY_SRC target read back below).
    let depth_view = create_depth_view(&gpu.device, options.width, options.height);
    let msaa_color_view =
        create_msaa_color_view(&gpu.device, options.width, options.height, COLOR_TARGET_FORMAT);

    // Resolve the requested geometry into the grid, then build the renderer's
    // instance buffer FROM the grid (the resolved-grid seam, `docs/adr/0006`). The voxel cap
    // (the stability cap) guards against an enormous CLI request.
    let shape = SdfShape::from_geometry(options.geometry.clone());
    // Z-up: layers are Z-slices, so the layer track spans the Z dimension (index 2).
    let grid_z = shape.grid_dimensions(options.geometry.voxels_per_block)[2];
    // Issue #12: build the layer-range band from the raw CLI voxel indices (no
    // snapping — flags take raw indices). Defaults to the full range.
    let layer_range = LayerRange {
        lower: options.layer_lower.unwrap_or(0).min(grid_z),
        upper: options.layer_upper.unwrap_or(grid_z).min(grid_z),
        snap_to_blocks: true,
        onion_skin: options.onion_depth > 0,
        onion_depth: options.onion_depth.clamp(1, 8),
    };
    let mut panel_state = PanelState {
        geometry: options.geometry.clone(),
        projection_mode: options.projection_mode,
        material: options.material,
        show_view_cube: options.show_view_cube,
        // Issue #31: the grid masters are no longer mirrored onto PanelState. The CLI
        // `--grid`/`--lattice`/`--floor` flags drive `scene.master_*` directly below
        // (the single source of truth); the scene's masters otherwise default ON.
        debug_face_orientation: options.debug_face_orientation,
        // ADR 0018 Decision 3: the viewer mode (`--view-mode`). Only Show-booleans
        // populates the boolean-operand ghost this slice; Normal / Onion-fog leave it empty.
        view_mode: options.view_mode,
        // Issue #88: pin the folded/expanded state of the floating Signal display stack for
        // the golden (no pointer input exists on the single shot frame to fold it live).
        stack: voxel_worker::SignalStackState {
            folded: options.stack_folded,
            ..voxel_worker::SignalStackState::default()
        },
        layer_range,
        ..PanelState::default()
    };
    // ADR 0001 step 2/3: resolve through a scene. `--demo-scene` builds a
    // hardcoded multi-node PLACED scene (sphere at origin + box offset +8 in X +
    // clouds offset in Z) to verify separated placement; otherwise a one-node
    // scene — a Tool, or a DebugClouds VoxelBody when `--shape debug-clouds`. Seed the
    // panel's scene so the node-list section renders the nodes in the captured
    // panel.
    // `--replay` (ADR 0003 Phase C, slice C3) is the highest-precedence scene SOURCE:
    // when present it REPLACES the demo/shape sources entirely — the scene is built by
    // replaying the JSONL Intent script against the default seed via
    // `AppCore::apply_intent`. A parse/read error is reported (line number + bad line)
    // and the process exits non-zero, rather than panicking. The camera/projection
    // flags below still apply to the replay render.
    // `--from-config` (repro flow): the app's persisted `config.json` / an F9 repro dump is the
    // HIGHEST-precedence scene+camera source. Its scene REPLACES the shape/demo build and its
    // camera is stashed to override the CLI theta/phi/dist/proj below. Loaded loud (a bad path
    // exits) so a headless repro never silently renders a different scene.
    let from_config = options.from_config.as_ref().map(|path| {
        voxel_worker::AppConfig::load_from(path).unwrap_or_else(|error| {
            eprintln!("shot: --from-config {error}");
            std::process::exit(2);
        })
    });
    if let Some(config) = &from_config {
        // Adopt the persisted scene + density + material + projection so the render matches the
        // app frame-for-frame. `to_panel_state` reconstructs the full node tree (with ids + the
        // origin point); we take its scene and the app-level display attributes.
        let restored = config.to_panel_state();
        panel_state.scene = restored.scene;
        panel_state.geometry = restored.geometry;
        panel_state.material = restored.material;
        panel_state.projection_mode = config.projection_mode;
        panel_state.applied_block_label = restored.applied_block_label;
        // ADR 0024: the session state a dump now carries. A repro that replays the scene
        // and the camera but resets the viewer mode renders a different picture than the
        // one the fault was reported in, which is the whole failure this category was
        // added to end. The CLI still wins where it spoke: `--view-mode` when actually
        // passed, and the two set-only bool flags, which can express `true` and have no
        // way to say `false` — so an explicit flag ORs on top rather than being erased.
        if !options.view_mode_explicit {
            panel_state.view_mode = restored.view_mode;
        }
        panel_state.stack = restored.stack;
        if options.stack_folded {
            panel_state.stack.folded = true;
        }
        panel_state.debug_face_orientation =
            restored.debug_face_orientation || options.debug_face_orientation;
        panel_state.debug_brick_faces = restored.debug_brick_faces;
        // ADR 0022: adopt the armed placement ghost the dump carried (session state), so a
        // mid-gesture F9 repro renders the pending drop.
        panel_state.placement_ghost = restored.placement_ghost;
    }

    let mut scene = if from_config.is_some() {
        // The scene was already adopted into `panel_state.scene` from the loaded config above.
        panel_state.scene.clone()
    } else if let Some(replay_path) = &options.replay_path {
        match build_scene_from_replay(replay_path) {
            Ok(replayed_scene) => replayed_scene,
            Err(message) => {
                eprintln!("{message}");
                std::process::exit(1);
            }
        }
    } else if options.far_offset || options.far_offset_near {
        build_far_offset_scene(options.geometry.voxels_per_block, options.far_offset)
    } else if options.demo_groups {
        build_demo_groups(options.geometry.voxels_per_block)
    } else if let Some(edge_voxels) = options.demo_sketch_box {
        build_demo_sketch_box(edge_voxels, options.geometry.voxels_per_block)
    } else if options.demo_sketch_extrude {
        build_demo_sketch_extrude(options.geometry.voxels_per_block)
    } else if options.demo_sketch_revolve {
        build_demo_sketch_revolve(options.geometry.voxels_per_block)
    } else if options.demo_village_far {
        build_demo_village_far(options.geometry.voxels_per_block)
    } else if options.demo_village {
        build_demo_village(options.geometry.voxels_per_block)
    } else if options.demo_overlap {
        build_demo_overlap(options.geometry.voxels_per_block)
    } else if options.demo_subtract {
        build_demo_subtract(options.geometry.voxels_per_block)
    } else if options.demo_group_subtract {
        build_demo_group_subtract(options.geometry.voxels_per_block)
    } else if options.demo_intersect {
        build_demo_intersect(options.geometry.voxels_per_block)
    } else if options.demo_cutter_def {
        build_demo_cutter_def(options.geometry.voxels_per_block)
    } else if options.demo_window_fixture {
        build_demo_window_fixture(options.geometry.voxels_per_block)
    } else if options.demo_buried_cutter {
        build_demo_buried_cutter(options.geometry.voxels_per_block)
    } else if options.demo_child_booleans {
        build_demo_child_booleans(options.geometry.voxels_per_block)
    } else if options.demo_two_material {
        build_demo_two_material(options.geometry.voxels_per_block)
    } else if options.demo_mixed_material {
        build_demo_mixed_material(options.geometry.voxels_per_block)
    } else if options.demo_scene {
        build_demo_scene(options.geometry.voxels_per_block)
    } else if options.debug_clouds {
        Scene::single_node(Node::new(
            "Clouds",
            NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 0 }),
        ))
    } else {
        Scene::from_geometry(options.geometry.clone(), options.material)
    };
    // Issue #29 S5: Points are SUPPRESSED unless `--points`. The headless scenes do
    // NOT synthesize an Origin Point (that runs on the windowed load/seed path), so by
    // default `scene.points` is empty → nothing renders and the panel's Points section
    // is zero-height (the 6 existing goldens stay byte-identical). `--points` adds the
    // Origin (ground + axes on by default) so the deliberate Points golden shows the
    // world reference grid.
    if options.show_points {
        scene.ensure_origin_point();
        // An optional extra Point (issue #29 Points fast-follow) at a chosen world
        // block position with its XY ground plane (Z-up) + axes on, so a headless
        // capture can verify a second analytic grid plane at a different height/offset.
        if let Some(position_blocks) = options.extra_point_blocks {
            scene.points.push(Point {
                name: "Extra".to_string(),
                position_blocks,
                plane_xy: true,
                plane_xz: false,
                plane_yz: false,
                ..Point::default()
            });
        }
    }
    // ADR 0003 Phase B: mint a stable NodeId for every node before the scene is
    // consumed (idempotent; nothing reads the id yet in B1).
    scene.ensure_node_ids();
    panel_state.scene = scene.clone();
    // Issue #29 S2: `--select-node N` overrides the active selection so a headless
    // capture can place the transform gizmo on a chosen (non-origin) node and prove
    // it follows the selection. An out-of-range index clears the selection.
    if let Some(index) = options.select_node {
        // ADR 0003 Phase B3: selection is keyed by NodeId. Parse the same top-level
        // index as before, then resolve it to that node's stable id (ids were minted
        // by `ensure_node_ids` above), so the SAME `--select-node N` argument selects
        // the SAME node. An out-of-range index resolves to None → clears selection.
        panel_state.scene.active = panel_state
            .scene
            .id_at_path(&NodePath::root_index(index));
    }
    // ADR 0018 Decision 2: `--select-root` selects the ROOT PART, so a headless capture
    // can prove a view mode applies scene-wide (Show-booleans x-rays every boolean).
    // Takes precedence over `--select-node`.
    if options.select_root {
        panel_state.scene.active = Some(voxel_worker::ROOT_NODE_ID);
    }
    // Issue #29 S3: the per-object block lattice + floor grid are now gated by a
    // scene master ANDed with each NODE's own toggle (default OFF), so a headless
    // capture must enable them explicitly. `--lattice`/`--floor` set the matching
    // scene master AND turn the per-object flag on for ONE node — the
    // `--select-node N` node (else the top-level node 0). This proves the grid
    // hugs that node's enclosing blocks while a sibling shows none.
    if options.show_block_lattice || options.show_floor_grid {
        panel_state.scene.master_block_lattice = options.show_block_lattice;
        panel_state.scene.master_floor_grid = options.show_floor_grid;
        let grid_node = options.select_node.unwrap_or(0);
        // B5: address the top-level node by position via the public path helper.
        if let Some(node) = panel_state
            .scene
            .node_at_path_mut(&NodePath::root_index(grid_node))
        {
            node.grids.block_lattice = options.show_block_lattice;
            node.grids.floor_grid = options.show_floor_grid;
        }
    }
    // Issue #29 S4: the on-face voxel grid is likewise per-object now (master AND a
    // node's own `voxel_grid_on_faces`). `--grid` sets the scene master AND turns the
    // per-object flag on for ONE node — the `--select-node N` node (else node 0) — so
    // a 2-node capture shows the enabled node's faces bearing bold block-edge grid
    // lines while the sibling's faces show none. The bit is baked at resolve, so this
    // must run BEFORE the resolve below.
    if options.show_grid_overlay {
        // Mutate the LOCAL `scene` — the resolve below reads it (not
        // `panel_state.scene`), so the flag must be baked here for the bit to land
        // on each voxel's `material_id`. Re-sync the panel copy so the inspector
        // and per-frame uniforms agree.
        scene.master_voxel_grid = true;
        let grid_node = options.select_node.unwrap_or(0);
        // B5: address the top-level node by position via the public path helper.
        if let Some(node) = scene.node_at_path_mut(&NodePath::root_index(grid_node)) {
            node.grids.voxel_grid_on_faces = true;
        }
        panel_state.scene = scene.clone();
    }
    // ADR 0022: `--placement-ghost` arms the translucent SDF ghost of the current
    // `--shape`/`--size`/`--density` geometry at `--ghost-offset` (default the origin) —
    // the headless verification path for the placement ghost (does it render, and does it
    // COINCIDE with an equivalent solid node at the same offset?). Overrides any ghost a
    // `--from-config` dump adopted above.
    if options.placement_ghost {
        panel_state.placement_ghost = Some(PlacementGhost {
            shape: SdfShape::from_geometry(options.geometry.clone()),
            offset_voxels: options.ghost_offset,
            // The headless verification places on whole-voxel `--ghost-offset`, so no sub-voxel
            // remainder (ADR 0027 `NoSnap`); the coincidence check runs at integer offsets.
            offset_local: [0.0, 0.0, 0.0],
            // ADR 0027: `--ghost-face N` tilts the ghost against that face (local +Z → N) as the
            // continuous rotation the classifier resolves; absent, the upright identity of a
            // world-plane / +Z-face drop. The lattice turn is bridged to a `Quat` by
            // `quat_from_lattice` — the same discrete→continuous map the leaf composes.
            rotation: options
                .ghost_face
                .map(|face| {
                    document::scene::quat_from_lattice(
                        substrate::spatial::LatticeOrientation::from_face_normal(face),
                    )
                })
                .unwrap_or(glam::Quat::IDENTITY),
        });
    }
    // The resolve region: for a placed multi-node scene this is the whole
    // composite extent (per-axis box over all node offsets ± sizes); for a single
    // node it equals the node's own size (the step-2 region).
    // A placed/instanced scene (demo-scene or demo-village) resolves its whole
    // composite extent; a single-node scene uses its own block size (step-2 region).
    // The far-offset demo also resolves its full composite extent (a single 4³
    // box). `full_extent_blocks` returns the box's own size (4³) for a lone node,
    // and the resolve rebases it to the floating origin (= the composite recentre)
    // in i64 BEFORE the f32 downcast (S4b), so even at a 1_000_000-block offset
    // (16M voxels, past the f32 exact-integer ceiling) the grid is BYTE-IDENTICAL
    // to the near box at the origin — the far-lands jitter is gone (S4b proof).
    let placed_scene = options.demo_scene
        || options.demo_overlap
        || options.demo_subtract
        || options.demo_group_subtract
        || options.demo_intersect
        || options.demo_cutter_def
        || options.demo_window_fixture
        || options.demo_buried_cutter
        || options.demo_child_booleans
        || options.demo_two_material
        || options.demo_mixed_material
        || options.demo_village
        || options.demo_village_far
        || options.demo_groups
        || options.far_offset
        || options.far_offset_near;
    let region = if placed_scene {
        scene.full_extent_blocks(options.geometry.voxels_per_block)
    } else {
        {
            // The geometry mirror is voxel-canonical (ADR 0003 §3f(0)); the explicit
            // single-shape region is whole blocks, so round the voxel size UP to
            // whole blocks (a whole-block size divides cleanly).
            let density = options.geometry.voxels_per_block.max(1);
            let voxels = options.geometry.size_voxels;
            RegionBlocks::new([
                voxels[0].div_ceil(density),
                voxels[1].div_ceil(density),
                voxels[2].div_ceil(density),
            ])
        }
    };
    // Issue #27 S2: the old whole-region `MAX_GRID_VOXELS` total cap is now a
    // PER-CHUNK bound — a scene whose TOTAL voxel count is far beyond 6M resolves
    // fine as long as each chunk is small. Only a pathological density (one chunk's
    // voxel capacity alone exceeds the bound) is rejected.
    let density = options.geometry.voxels_per_block;
    // The headless core: the resolve store + the CLI camera (assigned once the
    // region dimensions are known, below). The CHUNKABLE path — every shape + demo
    // scene the goldens test — resolves through `AppCore::rebuild`, the SAME
    // store-backed resolve + per-chunk path the windowed app drives, so the golden
    // net now exercises the real core instead of a parallel copy (A3 keystone). The
    // density-cap and VoxelBody-only branches stay shot-specific (the windowed app never
    // produces them). `render_chunks_for_mesh` carries the per-chunk accessor
    // (chunkable path only) to the cuboid mesh build below; it borrows the store, so
    // `app_core` is left untouched until it is consumed + dropped there.
    let mut app_core = AppCore::new(OrbitCamera::default());
    // Issue #20 S6c-1: `region_dimensions` (what the camera auto-frame, origin gizmo,
    // block lattice and fine floor grid are sized from) is read from the SCENE, not
    // by reaching into the assembled grid object. For a chunkable scene it equals
    // `grid.dimensions` exactly (the resolve sizes the grid to
    // `placed_region_dimensions`, proven in
    // `scene::tests::placed_region_dimensions_equals_assembled_grid`); for the
    // chunkable path it comes straight off `AppCore::rebuild`'s output (no recompute).
    // A VoxelBody-only scene (`--shape debug-clouds`) has no composite extent, so it is
    // resolved through the explicit-region path and sized `region × density` (rather
    // than `placed_region_dimensions`, which is `[0,0,0]` for it).
    // The dense reference `Store` (ADR 0010 E5) owns the per-chunk grids the default
    // (dense) mesh path borrows via `render_chunks_for_mesh`; it must outlive that
    // borrow, so it lives here at the `main` scope. `None` on the density-cap /
    // VoxelBody-only branches (which build no dense per-chunk accessor).
    let mut reference_store: Option<voxel_worker::Store> = None;
    let (grid, region_dimensions, mut render_chunks_for_mesh) =
        if voxel_core::voxel::chunk_extent_exceeds_bound(density) {
            let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * density.max(1)) as u64;
            let chunk_voxels = chunk_extent * chunk_extent * chunk_extent;
            panel_state.voxel_cap_warning_millions = Some(chunk_voxels as f32 / 1_000_000.0);
            eprintln!(
                "3D paused — one chunk is {:.1}M voxels, exceeding the per-chunk bound; \
                 rendering empty grid",
                chunk_voxels as f32 / 1_000_000.0
            );
            // Render an EMPTY grid + (below) an empty mesh — NO resolve at the
            // pathological density. This matches the windowed app, where
            // `AppCore::rebuild` returns `DensityRejected` and resolves nothing; it
            // also makes shot's mesh consistent with the empty `grid` above + the
            // "rendering empty grid" message (the pre-A3 cuboid path keyed on
            // `has_chunkable_extent` and resolved per-chunk anyway, risking the very
            // huge-allocation/hang the cap exists to prevent).
            let grid = VoxelGrid::new([
                region.size_blocks[0] * density,
                region.size_blocks[1] * density,
                region.size_blocks[2] * density,
            ]);
            let region_dimensions = if scene.has_chunkable_extent(density) {
                scene.placed_region_dimensions(density)
            } else {
                grid.dimensions
            };
            (grid, region_dimensions, None)
        } else if scene.has_chunkable_extent(density) {
            // ADR 0010 E5: `shot` is the golden **DENSE REFERENCE ORACLE** — it resolves
            // through the dense `Store` (the retired-from-runtime `resolve_region`, kept
            // as the parity/golden reference) so the committed reference PNGs are the
            // dense-path truth the two-layer live path is cross-checked against
            // (`--two-layer` renders the E5 runtime path; the golden test asserts they
            // are pixel-identical). `render_chunks_for_mesh` carries the per-chunk dense
            // accessor to the cuboid mesh build below; it borrows the store, so the store
            // must outlive the mesh build (owned here, dropped after).
            let store = reference_store.insert(voxel_worker::Store::new());
            let grid = store.resolve_region(&scene, density, 0);
            let region_dimensions = AppCore::region_dimensions_for(&scene, density);
            let render_chunks = store.resident_render_chunks(&scene, density, 0);
            (grid, region_dimensions, Some(render_chunks))
        } else {
            // A VoxelBody-only scene (e.g. `--shape debug-clouds`) has no intrinsic-size
            // leaf, so there is no composite AABB to chunk — the cloud field sizes
            // itself to the EXPLICIT region. Resolve it directly through the monolithic
            // path, exactly as before (unchanged output). The windowed app never
            // produces a VoxelBody-only scene.
            let grid = scene.resolve_region(region, density, 0);
            let region_dimensions = [
                region.size_blocks[0] * density,
                region.size_blocks[1] * density,
                region.size_blocks[2] * density,
            ];
            (grid, region_dimensions, None)
        };
    // The voxel-space grid dimensions actually resolved (the composite region for
    // a placed scene), used for the layer track and the uniforms / fog.
    let grid_dimensions = grid.dimensions;
    debug_assert_eq!(
        region_dimensions, grid_dimensions,
        "S6c-1: scene region dimensions must equal the assembled grid the consumers used"
    );
    if options.replay_path.is_some() {
        println!(
            "resolved {} voxels for --replay ({} top-level node(s), {} definition(s), {} point(s))",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            scene.points.len(),
        );
    } else if options.far_offset || options.far_offset_near {
        println!(
            "resolved {} voxels for demo-far-offset ({}, offset {:?} blocks, region {:?} blocks) \
             — S4b: the resolve rebases to the floating origin in i64 before the f32 downcast, so \
             the far box renders BYTE-IDENTICAL to the near box (no far-lands jitter)",
            grid.occupied_count(),
            if options.far_offset { "far" } else { "near" },
            if options.far_offset { FAR_OFFSET_BLOCKS } else { [0, 0, 0] },
            region.size_blocks
        );
    } else if options.demo_groups {
        println!(
            "resolved {} voxels for demo-groups ({} top-level nodes, {} definition(s), region {:?} blocks)",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            region.size_blocks
        );
    } else if options.demo_village_far {
        println!(
            "resolved {} voxels for demo-village-far ({} instances of {} definition(s), base offset \
             {:?} blocks, region {:?} blocks) — ADR 0010 D0: the composite is rebased to its \
             floating origin in i64 before the f32 downcast, so the far village renders crisp \
             (the §3a payload-move baseline)",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            FAR_SCENE_BASE_BLOCKS,
            region.size_blocks
        );
    } else if options.demo_village {
        println!(
            "resolved {} voxels for demo-village ({} instances of {} definition(s), region {:?} blocks)",
            grid.occupied_count(),
            scene.roots.len(),
            scene.definitions.len(),
            region.size_blocks
        );
    } else if options.demo_scene {
        println!(
            "resolved {} voxels for demo-scene (region {:?} blocks)",
            grid.occupied_count(),
            region.size_blocks
        );
    } else if options.debug_clouds {
        println!(
            "resolved {} voxels for DebugClouds {:?}@{}",
            grid.occupied_count(),
            shape.size_voxels,
            options.geometry.voxels_per_block
        );
    } else {
        println!(
            "resolved {} voxels for {:?} {:?}@{}",
            grid.occupied_count(),
            shape.kind,
            shape.size_voxels,
            options.geometry.voxels_per_block
        );
    }

    // M8: `--export-vox` writes the resolved grid as a MagicaVoxel .vox and then
    // exits (no render needed — this is the headless verification path).
    if let Some(vox_path) = &options.export_vox_path {
        // ADR 0003 §3a: map each categorical `block_id` to its colour via the procedural
        // block palette (slot `material_id` = that material's average), so a multi-
        // material grid exports each block in its own colour. The active material's slot
        // keeps its representative colour, so a single-material grid is unchanged.
        let representative = procedural_material_average_color(options.material);
        let mut palette_colors = VoxExport::block_palette_from_active(options.material, representative);
        for (slot, color) in palette_colors.iter_mut().enumerate() {
            *color = procedural_material_average_color(
                voxel_core::core_geom::MaterialChoice::from_material_id(slot as u16),
            );
        }
        palette_colors[options.material.material_id() as usize] = representative;
        let export = VoxExport::from_grid(&grid, palette_colors);
        match export.write(vox_path) {
            Ok(bytes) => println!(
                "wrote {} ({} voxels, {} model(s), {} bytes)",
                vox_path.display(),
                export.voxel_count(),
                export.model_count(),
                bytes
            ),
            Err(error) => {
                eprintln!("export .vox failed: {error}");
                std::process::exit(1);
            }
        }
        return;
    }

    // ADR 0011 G2: `--brick` sources the voxel display from the brick raymarch. The
    // gate mirrors the live app's: a chunkable procedural scene
    // (SDF / SketchSolid — the ADR 0007-ported set; DebugClouds is VoxelBody-only, so the
    // two-layer store has no boundary set for it), brick-representable (every rendered
    // block single-material + uniform overlay — per-record ids carry per-block materials,
    // so multi-producer distinct-material scenes engage; the R8 atlas is occupancy-only,
    // so a block mixing materials stays on the mesh), and none of the mesh-only modes
    // (debug-faces, `--scan-vs` loaded VS textures).
    let mut brick_raymarch_renderer: Option<voxel_worker::BrickRaymarchRenderer> = None;
    if options.brick {
        if !scene.has_chunkable_extent(options.geometry.voxels_per_block) || options.scan_vs {
            // NOTE: `--debug-faces` is NO LONGER a brick disqualifier — with `--brick` it
            // drives the brick raymarch's OWN diagnostic mode (`set_debug_mode` below), the
            // grazing-rim geometry-vs-shading discriminator. Without `--brick` it still means
            // the cuboid-mesh face-orientation debug.
            println!(
                "brick: scene not gated (needs a chunkable procedural scene, \
                 no loaded VS material) — falling back to the mesh path"
            );
        } else {
            let density = options.geometry.voxels_per_block;
            let two_layer_chunks = evaluation::two_layer_store::TwoLayerStore::enabled()
                .build_covering_chunks(&scene, density, 0);
            // shot's --brick goldens verify the brick display against the dense reference
            // INCLUDING onion-band cut planes. This installs the LIVE surface-only build
            // (ADR 0011 interior elision) — the exact record set the running app uploads —
            // so the goldens pin the live path, not a test-only oracle. A band cut plane can
            // start a ray inside the solid where the surface-only set holds no record; the
            // raymarch's block-occupancy fallback (the pyramid rides the same chunks and
            // carries the block-granular interior mask) resolves those elided interiors to
            // their coarse cubes, so the cross-section renders correctly.
            let build = voxel_worker::build_brick_field(&two_layer_chunks, density);
            // The representability gate is deleted (material atlas): every NON-EMPTY scene
            // engages the brick path, mixed-material included — a mixed brick's per-voxel cell
            // keys upload alongside the occupancy atlas via `install_brick_field_with_cell_keys`.
            if !build.brick_records.is_empty() {
                let gpu_records = voxel_worker::pack_gpu_records(&build.brick_records, |_| {
                    options.brick_force_miss
                });
                let sculpted = build.sculpted_brick_count();
                println!(
                    "brick raymarch: {} records ({} coarse + {} sculpted), atlas {}³, \
                     {}display=bricks",
                    build.brick_records.len(),
                    build.brick_records.len() - sculpted,
                    sculpted,
                    build.atlas_dim_voxels,
                    if options.brick_force_miss {
                        "ALL sculpted forced non-resident (residency-miss), "
                    } else {
                        ""
                    },
                );
                let mut renderer = voxel_worker::BrickRaymarchRenderer::new(
                    &gpu.device,
                    &gpu.queue,
                    COLOR_TARGET_FORMAT,
                );
                let pyramid = voxel_worker::ClipmapPyramid::from_chunks(&two_layer_chunks);
                let atlas = build.atlas_payload();
                let cell_key_atlas = build.cell_key_atlas_payload();
                renderer.install_brick_field_with_cell_keys(
                    &gpu.device,
                    &gpu.queue,
                    &build.brick_records,
                    &atlas,
                    &cell_key_atlas,
                    &gpu_records,
                    &pyramid,
                    voxel_worker::RecentreVoxels::new(grid.recentre_voxels),
                );
                // Grazing-rim DIAGNOSTIC: `--brick --debug-faces` shades every hit by its
                // face axis + a per-voxel UV checkerboard (geometry-staircase vs shading-
                // precision discriminator). Default off leaves the brick goldens byte-identical.
                if options.debug_face_orientation {
                    renderer.set_debug_mode(1);
                }
                brick_raymarch_renderer = Some(renderer);
            } else {
                println!("brick: boundary set is empty — falling back to the mesh path");
            }
        }
    }

    // Part of #20: the cuboid mesh path is the sole voxel renderer. Since issue #20
    // S6c-2d it meshes PER CHUNK with a 1-voxel neighbour apron: built from the
    // resolve cache's per-chunk accessor (`resident_render_chunks`) so the goldens
    // exercise the per-chunk path, falling back to the whole-grid wrapper when the
    // scene has no chunkable extent (the wrapper buckets internally → identical mesh).
    let mut cuboid_mesh_renderer = if brick_raymarch_renderer.is_some() {
        // ADR 0011 G1: bricks own the frame — build the mesh renderer EMPTY (the
        // borrow of the dense store is released first) so the capture provably
        // renders from the brick atlas, not the mesh.
        if let Some(render_chunks) = render_chunks_for_mesh.take() {
            drop(render_chunks);
        }
        CuboidMeshRenderer::new(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &VoxelGrid::new(grid_dimensions),
            options.geometry.voxels_per_block,
        )
    } else if !options.dense && scene.has_chunkable_extent(options.geometry.voxels_per_block) {
        // ADR 0010 E3 / #50: mesh THROUGH the two-layer path — now the DEFAULT (the live-app
        // path), so a headless render matches the window, INCLUDING the ADR 0027 continuous
        // rotation the dense oracle drops. `--dense` opts back to the parity oracle below. Build
        // each covering chunk's [`evaluation::two_layer_store::TwoLayerChunk`]
        // `TwoLayerChunk` (coarse one-box + microblock cuboids + seam flags) and mesh from
        // it. PROVES pixel-identity to the dense path (the E3 golden gate). The render-chunk
        // borrow is dropped first (the two-layer build reads the SCENE evaluator, not the
        // dense store), freeing `app_core` for the camera assignment below.
        if let Some(render_chunks) = render_chunks_for_mesh.take() {
            drop(render_chunks);
        }
        let density = options.geometry.voxels_per_block;
        let store = evaluation::two_layer_store::TwoLayerStore::enabled();
        let two_layer_chunks = store.build_covering_chunks(&scene, density, 0);
        println!(
            "two-layer mesher: {} covering chunks, {} stored boundary voxels (interior elided)",
            two_layer_chunks.len(),
            two_layer_chunks
                .iter()
                .map(|(_, c)| c.stored_voxel_count())
                .sum::<u64>(),
        );
        CuboidMeshRenderer::new_from_two_layer_chunks(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &two_layer_chunks,
            grid_dimensions,
            // The dense-oracle grid carries its recentre as a raw triple; mint the frame
            // newtype at this `shot` boundary (the builder now speaks `RecentreVoxels`).
            voxel_worker::RecentreVoxels::new(grid.recentre_voxels),
            density,
        )
    } else if let Some(render_chunks) = render_chunks_for_mesh.take() {
        // Chunkable path: mesh the per-chunk accessor from `AppCore::rebuild` above
        // (1-voxel neighbour apron per chunk), so the goldens exercise the real
        // per-chunk mesh path. `render_chunks` holds an immutable borrow of the store;
        // consume + drop it here, freeing `app_core` for the camera assignment below.
        let renderer = CuboidMeshRenderer::new_from_chunks(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &render_chunks,
            grid_dimensions,
        );
        drop(render_chunks);
        renderer
    } else {
        // Density-cap (empty grid) or VoxelBody-only scene: mesh the whole grid (the
        // wrapper buckets it into per-chunk sub-grids internally → identical mesh).
        CuboidMeshRenderer::new(
            &gpu.device,
            &gpu.queue,
            COLOR_TARGET_FORMAT,
            &grid,
            options.geometry.voxels_per_block,
        )
    };
    // ADR 0018 Decision 6: the boolean-operand ghost — every Subtract/Intersect operand
    // body in the selected subtree, as an operation-coded x-ray (quiet where directly
    // visible, loud where buried). Populated only in Show-booleans mode (`--view-mode
    // booleans`); derived from the SAME `panel_state.scene` the gizmo reads (so
    // `--select-node` / `--select-root` steer it), bounded by the ghosted operands'
    // covering chunks; meshed against the COMPOSED scene's recentre so it lands
    // voxel-exact on each operand's place (ADR 0008). Per-frame uniforms upload below.
    let mut selected_operand_ghost_renderer =
        SelectedOperandGhostRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    if options.view_mode == ViewMode::ShowBooleans {
        if let Some(ghost) = AppCore::boolean_operand_ghost(
            &panel_state.scene,
            options.geometry.voxels_per_block,
        ) {
            selected_operand_ghost_renderer.rebuild(
                &gpu.device,
                &ghost.bodies,
                ghost.grid_dimensions,
                ghost.recentre,
                ghost.density,
            );
            println!("boolean-operand ghost: {} body(ies)", ghost.bodies.len());
        }
    }

    // Transform gizmo (issue #29 S2): when `--gizmo` is passed, place it ON the
    // active/selected node at its recentred pivot, screen-stable-sized via the model
    // matrix below. `None` (no selection / no extent) keeps `--gizmo` a no-op, and the
    // goldens (which never pass `--gizmo`) are unaffected.
    let gizmo_placement = if options.show_origin_gizmo {
        AppCore::gizmo_placement(&panel_state.scene, options.geometry.voxels_per_block)
    } else {
        None
    };
    let transform_gizmo_renderer = TransformGizmoRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // Per-object block lattice + floor grid (issue #29 S3): its line batch is built
    // from the scene's grid-enabled nodes below (after the camera is known).
    let mut scene_grid_renderer = SceneGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // The world reference grid (issue #29 S5): SUPPRESSED by default (so the existing
    // goldens are byte-identical); `--points` enables it. Its batch is built below
    // from `scene.points` + the camera once the view matrix is known.
    let mut points_renderer = PointsRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // The analytic infinite reference grid (issue #29 Points fast-follow): the Points'
    // enabled planes. SUPPRESSED by default with the rest of Points; `--points` enables
    // it. Built below from `scene.points` + the camera once the view matrix is known.
    let mut infinite_grid_renderer = InfiniteGridRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // ADR 0022: the armed-tool placement ghost. Held disarmed; armed below (after the
    // camera matrix is known) from `panel_state.placement_ghost` in the grid's recentre —
    // the EXACT frame the solid voxels above were resolved in (ADR 0008).
    let mut placement_ghost_renderer =
        PlacementGhostRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    let view_cube_renderer = ViewCubeRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    // Issue #91 (item 1): the Signal viewport background gradient (always on).
    let background_gradient_renderer =
        display::renderer::BackgroundGradientRenderer::new(&gpu.device, COLOR_TARGET_FORMAT);
    // Z-up: the voxel-space grid_z of the ACTUALLY resolved grid (the composite for a
    // placed scene), used for the band clip + uniforms so a demo scene that grew
    // past the single-shape `grid_z` is not clipped or mis-sized. Layers are Z-slices.
    let render_grid_z = grid_dimensions[2];
    // ADR 0018 Decisions 4–5: the region-scoped layer clip (band + onion-fog region),
    // derived by the SAME `AppCore::mesh_clip` the windowed shell uses. The band bites only
    // in Onion-fog mode with a selection (`--view-mode onion` + `--select-node`/`--select-root`);
    // Normal / Show-booleans (and a placed/demo scene with no onion selection) render finished.
    // In Onion-fog the `--layer-lower/--layer-upper` handles are OBJECT-RELATIVE over the
    // selected object's Z extent; `mesh_clip` offsets them into the scene-absolute `band` and
    // confines it to the object's placed AABB (`region`). BOTH display paths honour the region:
    // the cuboid mesh path (geometry) and the brick raymarch (per-frame uniforms, #85).
    let clip = AppCore::mesh_clip(
        &panel_state.scene,
        density,
        options.view_mode,
        layer_range,
        render_grid_z,
        options.debug_face_orientation,
    );
    let band = clip.band;
    // The measured-diameter readout: the widest occupied run in the effective (scene-absolute)
    // band, or the whole scene when the band is FULL.
    let (diam_lower, diam_upper) = if band == LayerBand::FULL {
        (0, render_grid_z)
    } else {
        (band.band_min, band.band_max)
    };
    let measured_diameter = grid.widest_run_in_band(diam_lower, diam_upper);

    // Build the orbit camera from the CLI flags. `--snap` overrides theta/phi
    // with the face's snapped angles directly (no tween in the headless path).
    let (theta, phi) = match (&from_config, options.snap_element) {
        // `--from-config` reproduces the app's EXACT live view: its orbit angles win over the
        // CLI theta/phi (and over --snap — the whole point is the persisted pose).
        (Some(config), _) => (config.orbit_theta, config.orbit_phi),
        (None, Some(element)) => element.snap_angles(),
        (None, None) => (options.theta, options.phi),
    };
    // The render-chunk borrow from `AppCore::rebuild` was consumed + dropped at the
    // cuboid mesh build above, so `app_core` is free again — install the CLI camera
    // so the headless render sources `view_projection` from the same `AppCore` the
    // windowed shell does.
    app_core.camera = OrbitCamera {
        // `--from-config` reproduces the app's PAN too: the persisted orbit target (the world
        // point the camera looks at). Without it a panned view reframes on the origin and misses
        // the artifact. A non-config render keeps the origin-centred target (the scene recentres
        // there).
        target: match &from_config {
            Some(config) => glam::Vec3::from_array(config.orbit_target),
            None => glam::Vec3::ZERO,
        },
        orbit_theta: theta,
        orbit_phi: phi,
        // `--from-config` uses the persisted orbit distance (the exact live zoom); otherwise the
        // CLI `--dist`, or the auto-frame. The scene resolves recentred on the origin on both the
        // app and shot paths, so the app's target≈origin and the distance transfers directly.
        orbit_distance: match &from_config {
            Some(config) => config.orbit_distance,
            None => options
                .distance
                .unwrap_or_else(|| OrbitCamera::auto_framed_distance(region_dimensions)),
        },
        // #13 Step 5: `--roll <radians>` twists the whole view about the view axis.
        roll: options.roll,
        projection_mode: match &from_config {
            Some(config) => config.projection_mode,
            None => options.projection_mode,
        },
    };
    // Issue #25: ALL uniform uploads (camera matrix → gizmo/lattice/view-cube
    // and the voxel pass) are deferred to AFTER `run_egui_frame`, because the
    // camera aspect must come from the CENTRAL 3D viewport rect (window minus the
    // side panel + bottom dock), which egui only reports once its panels are laid
    // out. The view-cube matrix is aspect-independent but uploaded alongside for
    // simplicity.

    // egui driven WITHOUT winit: build RawInput by hand.
    let mut egui_bridge = EguiPaintBridge::new(&gpu.device, COLOR_TARGET_FORMAT);
    let pixels_per_point = 1.0;
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::pos2(0.0, 0.0),
            egui::vec2(options.width as f32, options.height as f32),
        )),
        ..Default::default()
    };

    // M6: synchronously scan the VS install and populate the palette so the
    // screenshot shows real block thumbnails. Optionally apply the first block.
    let mut palette = PaletteHost::new(&gpu.device, &gpu.queue, String::new());
    let mut loaded_material: Option<LoadedMaterial> = None;
    if options.scan_vs {
        let (groups, source_name) = run_auto_scan_blocking();
        println!(
            "scan: {} groups from {}",
            groups.len(),
            source_name.as_deref().unwrap_or("(no install found)")
        );
        palette.ui.status = match source_name {
            Some(name) => format!("{} blocks loaded — {}", groups.len(), name),
            None => "No VS install found — use Connect folder".to_string(),
        };

        // M7: per-face resolver, kept alive so each group can resolve its
        // blocktype JSON → per-face PNGs.
        let resolver = FaceResolver::auto();

        // `--list-perface`: report which chiselable blocks have a distinct top vs
        // side face, then exit before any rendering.
        if options.list_per_face {
            let mut distinct = 0usize;
            println!("--- per-face scan (top != side) ---");
            for (group, _) in &groups {
                let variant = group.variants.first().cloned().unwrap_or_default();
                let faces = resolver.resolve(group, &variant);
                if faces.top_differs_from_side() {
                    distinct += 1;
                    println!(
                        "DISTINCT  {:<22} key={}  up={}  side={}  [{:?}]",
                        group.label,
                        group.key,
                        file_stem_of(&faces.paths[2]),
                        file_stem_of(&faces.paths[0]),
                        faces.provenance,
                    );
                }
            }
            println!(
                "--- {distinct}/{} chiselable blocks resolve to distinct per-face textures ---",
                groups.len()
            );
            return;
        }

        // Choose which block to apply (if any) and resolve its per-face textures.
        let chosen: Option<(String, voxel_worker::FaceTextures)> =
            if let Some(stem) = &options.force_demo_stem {
                // Demo escape hatch: resolve an arbitrary texture stem directly via
                // the JSON index, even outside the chiselable allow-list, to prove
                // the per-face mechanism on a known block (e.g. wood/treetrunk/oak).
                resolve_demo_stem(stem).map(|faces| (stem.clone(), faces))
            } else {
                let target = if let Some(substring) = &options.apply_block_substring {
                    let lower = substring.to_ascii_lowercase();
                    groups.iter().find(|(group, _)| {
                        group.label.to_ascii_lowercase().contains(&lower)
                            || group.key.to_ascii_lowercase().contains(&lower)
                    })
                } else if options.apply_first_block {
                    groups.first()
                } else {
                    None
                };
                target.map(|(group, _)| {
                    let variant = group.variants.first().cloned().unwrap_or_default();
                    let faces = resolver.resolve(group, &variant);
                    (group.label.clone(), faces)
                })
            };

        if let Some((label, faces)) = chosen {
            let material = LoadedMaterial::from_faces(
                &gpu.device,
                &gpu.queue,
                cuboid_mesh_renderer.material_bind_group_layout(),
                cuboid_mesh_renderer.material_sampler(),
                &faces,
                label.clone(),
            );
            println!(
                "applied block: {label} (per_face={}, provenance={:?})",
                material.is_per_face, faces.provenance
            );
            panel_state.applied_block_label = Some(label);
            loaded_material = Some(material);
        }

        for (group, decoded) in groups {
            palette.add_group(
                &gpu.device,
                &gpu.queue,
                &mut egui_bridge.renderer,
                group,
                &decoded,
            );
        }
    }

    // Part of #20: synthetic loaded block — six distinct solid-colour faces built
    // in-process (no VS install). Proves the cuboid path now renders a loaded
    // per-face D2Array (layer selected by normal) and matches the instanced path
    // per face. CubeFaceSlot order: 0 +X red, 1 -X green, 2 +Y blue, 3 -Y yellow,
    // 4 +Z magenta, 5 -Z cyan.
    if options.synthetic_block {
        const FACE_SIZE: u32 = 16;
        let face_colors: [[u8; 4]; 6] = [
            [220, 40, 40, 255],   // +X red
            [40, 200, 40, 255],   // -X green
            [40, 80, 220, 255],   // +Y blue
            [230, 210, 40, 255],  // -Y yellow
            [210, 40, 210, 255],  // +Z magenta
            [40, 210, 210, 255],  // -Z cyan
        ];
        let layer_bufs: Vec<Vec<u8>> = face_colors
            .iter()
            .map(|c| c.iter().copied().cycle().take((FACE_SIZE * FACE_SIZE * 4) as usize).collect())
            .collect();
        let layers: [&[u8]; 6] = [
            &layer_bufs[0], &layer_bufs[1], &layer_bufs[2],
            &layer_bufs[3], &layer_bufs[4], &layer_bufs[5],
        ];
        let material = LoadedMaterial::from_face_layers(
            &gpu.device,
            &gpu.queue,
            cuboid_mesh_renderer.material_bind_group_layout(),
            cuboid_mesh_renderer.material_sampler(),
            FACE_SIZE,
            FACE_SIZE,
            &layers,
            "synthetic".to_string(),
        );
        println!("applied synthetic 6-face block (per_face=true)");
        panel_state.applied_block_label = Some("synthetic".to_string());
        loaded_material = Some(material);
    }

    // The armed "Add <shape>" dialog shows when a ghost is armed; read the kind before the
    // call borrows `panel_state` mutably.
    let armed_shape = panel_state.placement_ghost.as_ref().map(|ghost| ghost.shape.kind);
    let prepared = run_egui_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &mut panel_state,
        // ADR 0018 Decision 5: the layer scrubber's track spans the selected object's Z
        // extent in Onion-fog mode (else the whole scene).
        clip.track_len,
        measured_diameter,
        // The headless capture never runs an export; the section renders idle.
        voxel_worker::ExportPanelState::default(),
        &palette.ui,
        raw_input,
        [options.width, options.height],
        pixels_per_point,
        // #13 Step 3: the headless path never opens the ViewCube context menu.
        &mut None,
        // ADR 0030: nor the general viewport context menu (windowed-only interaction).
        &mut None,
        // Signal (#86): no zone-name readout in the goldens — the highlight lives in
        // the cube itself; the readout is a windowed-only overlay. Keeps every golden
        // diff to the two cube corners.
        None,
        // The "Add <shape>" dialog shows when a ghost is armed (the `--placement-ghost`
        // headless verification path); otherwise off, so the goldens are unchanged.
        armed_shape,
        // ADR 0028 (#94): the headless capture computes no live vertex handles (sketch
        // authoring is a windowed-only interaction); the goldens stay handle-free.
        &[],
        // ADR 0030: no sketch segment lines either — windowed-only overlay.
        &[],
        // ADR 0028 (#95): likewise no add-point insert preview in the headless goldens.
        None,
    );

    // Issue #25: now that egui has laid out its panels, derive the camera aspect
    // from the CENTRAL 3D viewport rect (window minus side panel + bottom dock) so
    // the model is centred in the visible 3D area instead of partly hidden behind
    // the side panel. Then upload every uniform that depends on the camera matrix.
    let [_, _, viewport_width, viewport_height] = prepared.viewport_px;
    let aspect_ratio = viewport_width as f32 / viewport_height.max(1) as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let gizmo_pivot = gizmo_placement
        .map(|(pivot, _)| glam::Vec3::from_array(pivot))
        .unwrap_or(glam::Vec3::ZERO);
    let gizmo_fraction = display::renderer::GIZMO_SCREEN_FRACTION;
    let gizmo_model = app_core.camera.screen_stable_model(gizmo_pivot, gizmo_fraction);
    // The gizmo gets its OWN near/far (screen-stable size grows with zoom); depth-OFF draw.
    let gizmo_vp =
        app_core
            .camera
            .screen_stable_view_projection(aspect_ratio, gizmo_pivot, gizmo_fraction);
    transform_gizmo_renderer.update_uniforms(&gpu.queue, gizmo_vp, gizmo_model);
    // Build this capture's per-object grid batch from the scene's grid-enabled nodes
    // (issue #29 S3), then upload the camera matrix.
    scene_grid_renderer.rebuild_from_scene(
        &gpu.device,
        &gpu.queue,
        &panel_state.scene,
        options.geometry.voxels_per_block,
    );
    scene_grid_renderer.update_uniforms(&gpu.queue, view_projection);
    // World reference grid (issue #29 S5): build the visible Points' tiled planes +
    // axes, centred on the camera's projection onto each plane. Only wired into the
    // overlays when `--points` is passed (default OFF keeps the goldens unchanged).
    if options.show_points {
        points_renderer.rebuild_from_scene(
            &gpu.device,
            &gpu.queue,
            &panel_state.scene,
            options.geometry.voxels_per_block,
        );
        points_renderer.update_uniforms(&gpu.queue, view_projection);
        // The analytic infinite grid (issue #29 Points fast-follow): build the visible
        // Points' planes with the camera matrices (recentred frame) so the fullscreen
        // ray-plane shader can intersect each pixel's view ray with the plane.
        infinite_grid_renderer.rebuild_from_scene(
            &gpu.queue,
            &panel_state.scene,
            options.geometry.voxels_per_block,
            view_projection,
            app_core.camera.eye().to_array(),
        );
    }
    // ADR 0022: arm the placement ghost from `panel_state.placement_ghost`. The
    // render-frame field centre is resolved from THIS grid's recentre (`grid.recentre_voxels`)
    // via the frame law — the same recentre the solid voxels were resolved in — so a ghost
    // at offset P coincides with a solid node at P (the frame-error guard the shot verifies).
    if let Some(ghost) = &panel_state.placement_ghost {
        let voxels_per_block = options.geometry.voxels_per_block;
        let recentre = grid.recentre_voxels;
        let center_world = ghost.center_world(recentre, voxels_per_block);
        let semi_axes = ghost.semi_axes(voxels_per_block);
        println!(
            "placement ghost: {:?} offset={:?} centre_world={:?} semi_axes={:?} recentre={:?}",
            ghost.shape.kind, ghost.offset_voxels, center_world, semi_axes, recentre
        );
        placement_ghost_renderer.update_uniforms(
            &gpu.queue,
            view_projection,
            view_projection.inverse(),
            prepared.viewport_px,
            glam::Vec3::from_array(center_world),
            ghost.shape.kind,
            glam::Vec3::from_array(semi_axes),
            ghost.wall_voxels(voxels_per_block),
            PLACEMENT_GHOST_TINT,
            ghost.rotation_inverse_columns(),
        );
    }
    // ADR 0018 Decision 6: the boolean-operand ghost's camera + tint upload (mesh was
    // built at derivation above).
    selected_operand_ghost_renderer.update_uniforms(&gpu.queue, view_projection);
    view_cube_renderer.update_uniforms(&gpu.queue, app_core.camera.view_cube_view_projection());

    // Part of #20: upload the cuboid path's uniforms (camera + per-material base
    // colours + band clip) and frustum-cull its mesh chunks. A loaded VS block
    // textures the cuboid path per-face (its 6-layer D2Array is bound at draw time in
    // `render_frame`, selecting the loaded pipeline); `bound = None` here just
    // disables the procedural per-box modulation the loaded pipeline ignores.
    let bound = match &loaded_material {
        Some(_) => None,
        None => Some(options.material),
    };
    cuboid_mesh_renderer.update_uniforms(
        &gpu.device,
        &gpu.queue,
        view_projection,
        grid_dimensions,
        options.geometry.voxels_per_block,
        options.show_grid_overlay,
        bound,
        band,
        clip.region,
        options.debug_face_orientation,
    );
    println!(
        "cuboid mesher: {} boxes → {} exposed faces ({} triangles), {} chunks",
        cuboid_mesh_renderer.box_count(),
        cuboid_mesh_renderer.face_count(),
        cuboid_mesh_renderer.triangle_count(),
        cuboid_mesh_renderer.chunk_count(),
    );

    // ADR 0011 G1: the brick pass's per-frame uniforms mirror the cuboid path's
    // shading inputs (camera, viewport, band clip, overlay master, bound material)
    // so the two paths render pixel-comparable.
    if let Some(brick_raymarch) = &mut brick_raymarch_renderer {
        brick_raymarch.update_uniforms(
            &gpu.queue,
            view_projection,
            prepared.viewport_px,
            grid_dimensions,
            band,
            // ADR 0018 Decision 5 (S5): the brick path honours the region clip too.
            clip.region,
            options.show_grid_overlay,
            bound,
        );
        // ADR 0012 (H1): prepare the onion GHOST slabs (self-gates on `band.onion_depth`).
        brick_raymarch.update_ghost_uniforms(
            &gpu.queue,
            view_projection,
            prepared.viewport_px,
            grid_dimensions,
            band,
            clip.region,
        );
    }

    // ADR 0002 E2 (#19): the frustum cull ran inside `update_uniforms`. Report the
    // drawn/total chunk counts so the chunking + culling are verifiable headlessly.
    if options.debug_chunks {
        println!(
            "chunks: drew {} / {} ({} boxes total)",
            cuboid_mesh_renderer.visible_chunk_count(),
            cuboid_mesh_renderer.chunk_count(),
            cuboid_mesh_renderer.box_count(),
        );
    }

    // M6: the active material is a loaded VS block when one was applied,
    // otherwise the procedural choice.
    let material = match &loaded_material {
        Some(loaded) => MaterialSource::Loaded(&loaded.bind_group),
        None => MaterialSource::Procedural(options.material),
    };

    let overlays = FrameOverlays {
        background_gradient: &background_gradient_renderer,
        gizmo: gizmo_placement
            .is_some()
            .then_some(&transform_gizmo_renderer),
        view_cube: if options.show_view_cube {
            Some(&view_cube_renderer)
        } else {
            None
        },
        cube_hovered_zone: options.cube_hover,
        // #13 Step 6 follow-up: the four rotate arrows draw persistently when the view
        // is face-constrained. A forced `--cube-hover rotate-*` also enables them so a
        // golden can pin the arrow render even from a non-face camera.
        cube_rotate_arrows_visible: app_core.camera.is_face_constrained()
            || matches!(
                options.cube_hover,
                Some(camera::CubeChromeZone::RotateArrow(_))
            ),
        scene_grid: Some(&scene_grid_renderer),
        // Issue #29 S5: Points SUPPRESSED unless `--points` (keeps the 6 goldens
        // byte-identical); the new `demo-village --points` golden enables them.
        points: options.show_points.then_some(&points_renderer),
        // Issue #29 Points fast-follow: the analytic infinite grid (Points' planes),
        // suppressed with the rest of Points unless `--points`.
        infinite_grid: options.show_points.then_some(&infinite_grid_renderer),
        // ADR 0012: draw the onion ghost pass when the band is a real onion slab
        // (`band.onion_depth > 0` ⇔ onion skin on, non-full-range, not debug-faces).
        // The display ghosts the onion slabs (prepared in the cuboid/brick
        // `update_uniforms` above); the volumetric fog is retired.
        onion_ghost_active: band.onion_depth > 0,
        // ADR 0018 Decision 6: the boolean-operand ghost draws over BOTH display paths
        // (mesh + brick). Suppressed in debug-faces mode (a diagnostic render — every
        // ghost is off there); self-gates on an empty ghost (only Show-booleans populates
        // it).
        selected_operand_ghost: (!options.debug_face_orientation)
            .then_some(&selected_operand_ghost_renderer),
        // ADR 0022: the armed-tool placement ghost (self-gates on being armed).
        placement_ghost: panel_state
            .placement_ghost
            .as_ref()
            .map(|_| &placement_ghost_renderer),
        cuboid_mesh: &cuboid_mesh_renderer,
        // ADR 0011 G1: when engaged, the brick raymarch takes the voxel-model draw
        // (the mesh renderer above was built empty); everything else is unchanged.
        brick_raymarch: brick_raymarch_renderer.as_ref(),
        target_width: options.width,
        target_height: options.height,
        // Signal (issue #88): slide the cube left of the floating display stack.
        view_cube_right_inset_px: prepared.view_cube_right_inset_px,
    };

    // Paint via the exact same render-target-agnostic core the window uses.
    render_frame(
        &mut egui_bridge,
        &gpu.device,
        &gpu.queue,
        &capture_view,
        &msaa_color_view,
        &depth_view,
        material,
        &overlays,
        &prepared,
    );

    // --- Read back the texture into a PNG ---
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = options.width * bytes_per_pixel;
    let row_alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row =
        unpadded_bytes_per_row.div_ceil(row_alignment) * row_alignment;

    let readback_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("headless readback buffer"),
        size: (padded_bytes_per_row * options.height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut copy_encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("headless copy encoder"),
    });
    copy_encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &capture_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(options.height),
            },
        },
        wgpu::Extent3d {
            width: options.width,
            height: options.height,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(std::iter::once(copy_encoder.finish()));

    // Map and wait.
    let buffer_slice = readback_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |result| {
        result.expect("failed to map readback buffer");
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");

    // Strip the per-row padding into a tight RGBA8 image.
    let mut tightly_packed = Vec::with_capacity((unpadded_bytes_per_row * options.height) as usize);
    {
        let mapped = buffer_slice.get_mapped_range();
        for row_index in 0..options.height {
            let row_start = (row_index * padded_bytes_per_row) as usize;
            let row_end = row_start + unpadded_bytes_per_row as usize;
            tightly_packed.extend_from_slice(&mapped[row_start..row_end]);
        }
    }
    readback_buffer.unmap();

    if let Some(parent_dir) = options.output_path.parent() {
        if !parent_dir.as_os_str().is_empty() {
            std::fs::create_dir_all(parent_dir).expect("failed to create output directory");
        }
    }

    image::save_buffer(
        &options.output_path,
        &tightly_packed,
        options.width,
        options.height,
        image::ColorType::Rgba8,
    )
    .expect("failed to write PNG");

    println!(
        "wrote {} ({}x{})",
        options.output_path.display(),
        options.width,
        options.height
    );
}
