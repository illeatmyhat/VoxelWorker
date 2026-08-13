//! The hardcoded demo scene builders (`--demo-*`, `--demo-far-offset`) and the
//! texture-stem file helpers they and the `--force-demo-stem` path share.

use voxel_worker::{
    CombineOp, DefId, MaterialChoice, Node, NodeBuilder, NodeContent, PlaneAxis, RevolveAxis,
    Scene, SdfShape, ShapeKind, Sketch, SketchLength, SketchPoint, SketchSolid,
};

/// The block offset of the far-offset demo box — a large offset resolved through the
/// `i64` voxel offset. At density 16 this is **16 million voxels** from the origin,
/// past the f32 exact-integer ceiling (2²⁴ ≈ 16.7M), where a recenter-AFTER-f32-add
/// would lose the voxel-center `.5` fraction on EVERY voxel. The camera-relative rebase
/// (subtract the floating origin in i64 BEFORE the f32 downcast) renders this
/// byte-identical to the near box. An offset of 100_000 would not exercise it: the f32
/// ULP at 1.6M is 0.125, so `.5` survives and the box never jitters.
pub(crate) const FAR_OFFSET_BLOCKS: [i64; 3] = [1_000_000, 0, 0];

/// The block offset that places the `--demo-village-far` composite at the FAR end of
/// the anisotropic horizontal extent: ~10,000 blocks on
/// both horizontal axes (X and Z), with the VERTICAL axis (Z-up → index 2 is vertical;
/// the horizontal ground plane is X/Y) bounded. Per the project's Z-up convention the
/// two HORIZONTAL axes are X (index 0) and Y (index 1), and the VERTICAL axis is Z
/// (index 2) — so the far horizontal offset goes on X and Y and the vertical Z stays at
/// 0. At density 16 this sits 160,000 voxels from the origin per horizontal axis, where
/// an absolute f32 voxel center has barely a fractional bit left (the precision loss the
/// a chunk-local-integer payload removes). The composite SPAN stays small (a
/// ~20-block row of houses), so only the OFFSET is far — the resolved grid is the same
/// size as the near `--demo-village`.
pub(crate) const FAR_SCENE_BASE_BLOCKS: [i64; 3] = [10_000, 10_000, 0];

/// The file stem (no dir, no extension) of a path, for compact log output.
pub(crate) fn file_stem_of(path: &std::path::Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Resolve an arbitrary texture stem (e.g. `wood/treetrunk/oak`) to per-face
/// textures via the VS JSON index, bypassing the chiselable allow-list. Used by
/// `--force-demo-stem` to demonstrate per-face rendering on a known block when no
/// chiselable block has distinct faces. Returns `None` if no install is found or
/// the stem can't be located on disk.
pub(crate) fn resolve_demo_stem(stem: &str) -> Option<voxel_worker::FaceTextures> {
    use assets::registry::detect_all_sources;

    // Find the actual variant PNG on disk for this stem, under each install's
    // textures dir, trying both domains. The detectors give us the block dirs.
    let chosen_variant = locate_stem_png(stem)?;
    // Build a synthetic group keyed by the stem so the resolver can look up the
    // matching blocktype (its `base` entries reference this stem's directory).
    let group = assets::BlockGroup {
        label: stem.rsplit('/').next().unwrap_or(stem).to_string(),
        key: stem.to_string(),
        variants: vec![chosen_variant.clone()],
    };
    let sources = detect_all_sources();
    let mut fallback: Option<voxel_worker::FaceTextures> = None;
    for source in &sources {
        let faces = source.resolve_faces(&group, &chosen_variant);
        if !faces.is_uniform() {
            return Some(faces);
        }
        if fallback.is_none() {
            fallback = Some(faces);
        }
    }
    fallback
}

/// Locate the PNG for a bare texture stem on disk, scanning the same install
/// roots the detectors use, trying the `game` then `survival` domains.
fn locate_stem_png(stem: &str) -> Option<std::path::PathBuf> {
    let appdata = std::env::var("APPDATA").ok()?;
    let assets_root = std::path::Path::new(&appdata)
        .join("Vintagestory")
        .join("assets");
    for domain in ["game", "survival", "creative"] {
        let candidate = assets_root
            .join(domain)
            .join("textures")
            .join("block")
            .join(format!("{stem}.png"));
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Mint stable [`NodeId`](document::scene::NodeId)s for a freshly-built demo scene, so the
/// fixture can name its arriving selection by id. Selection is keyed by
/// [`NodeId`](document::scene::NodeId), so a demo built with positional intent — "select node
/// 0" — resolves that to an id here. The later `ensure_node_ids` on the load path is
/// idempotent.
fn with_node_ids(mut scene: Scene) -> Scene {
    scene.ensure_node_ids();
    scene
}

/// A demo fixture: the scene AND how the workspace arrives — which node is selected.
/// Selection lives outside the document, so a demo cannot smuggle it on the scene.
pub(crate) struct DemoScene {
    /// The built scene.
    pub scene: Scene,
    /// The node the capture arrives with selected, `None` for nothing picked.
    pub selection: Option<document::scene::NodeId>,
}

impl DemoScene {
    /// The usual demo: its first top-level node arrives selected.
    pub(crate) fn first_node(scene: Scene) -> Self {
        let scene = with_node_ids(scene);
        let selection = scene.roots.first().copied();
        Self { scene, selection }
    }

    /// A demo (or a non-demo capture path) that names its own arriving selection.
    pub(crate) fn selecting(scene: Scene, selection: Option<document::scene::NodeId>) -> Self {
        Self { scene, selection }
    }
}

/// A zero-density request has no voxel lattice. Keep demo construction non-geometric instead of
/// manufacturing a one-voxel block size for a sketch's fixed sources to evaluate against.
fn invalid_density_demo(voxels_per_block: u32) -> DemoScene {
    let scene = Scene {
        voxels_per_block,
        ..Scene::default()
    };
    DemoScene::selecting(scene, None)
}

/// Build the `--demo-scene`: a hardcoded multi-node PLACED
/// scene proving disjoint placement. A sphere at the origin, a box offset +8
/// blocks in X, and a torus offset +6 blocks in Z. Each Tool is 5 blocks, so the
/// offsets open clear gaps and the three solids sit visibly apart (no overlap at
/// the origin) — the headless check the demo exists to confirm.
///
/// The third node is an SDF Tool rather than a `DebugClouds` VoxelBody: that body has no
/// intrinsic bounded size — it fills whatever region it is handed — so as a region-filling
/// fog it would densely OCCLUDE the sphere and box and defeat the very separation the demo
/// verifies. A torus is a crisp, bounded solid that makes the disjoint placement
/// unambiguous in the PNG. VoxelBody placement
/// itself is covered by the scene.rs unit tests (a VoxelBody stamps under its offset),
/// and the in-app inspector offsets both Tools and Parts.
pub(crate) fn build_demo_scene(voxels_per_block: u32) -> DemoScene {
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
    ]));
    // Density is document-level.
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-overlap`: two solid boxes of DIFFERENT materials
/// placed so they OVERLAP, exercising the multi-material overlap case.
/// The overlap region resolves last-writer-wins by document order (the Wood box is second,
/// so it wins where they overlap), and the golden pins that the dense and two-layer paths
/// render this IDENTICALLY. The boxes are 4 blocks each, offset 2 blocks in X+Y so a corner
/// volume overlaps; their union is a recognizable two-tone L-ish solid.
pub(crate) fn build_demo_overlap(voxels_per_block: u32) -> DemoScene {
    let make = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone),
        make(ShapeKind::Box, [2, 2, 0], MaterialChoice::Wood),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-subtract`: a solid Stone box CARVED by a smaller
/// box placed AFTER it under [`CombineOp::Subtract`] — the ordered-fold tracer golden. The
/// cutter is a 2³-block box overlapping the Stone box's top +X/+Y corner octant, so the
/// render shows a crisp cubic NOTCH bitten out of the corner. The cutter deliberately
/// carries the WOOD material: a Subtract is an occupancy-only mask that never stamps, so
/// every newly-exposed face inside the notch must render STONE — visible proof that
/// surviving cells keep their material.
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_subtract(voxels_per_block: u32) -> DemoScene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make(
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
            "Body",
        ),
        // Placed AFTER the body ⇒ it carves it (document-order fold). Spans blocks
        // [2, 4)³ inside the body plus empty space beyond — the corner octant notch.
        make(
            [2, 2, 2],
            [2, 2, 2],
            MaterialChoice::Wood,
            CombineOp::Subtract,
            "Cutter",
        ),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-cylinder-subtract`: a solid Stone box
/// DRILLED by a Subtract cylinder standing through its top face — a blind bore. The bore
/// mouth is a CIRCLE on the top face: the curved junction the selection cel must trace
/// (no straight catalog edge could stand in for it), while the cylinder's own rim
/// circles sit above the box, off the composed surface.
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_cylinder_subtract(voxels_per_block: u32) -> DemoScene {
    let make = |kind, size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make(
            ShapeKind::Box,
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
            "Body",
        ),
        // The 2-block-wide bore stands on blocks [1,3)² and spans z ∈ [1, 6): its wall
        // crosses the body's top face z=4 on a circle, its floor stays inside — blind.
        make(
            ShapeKind::Cylinder,
            [2, 2, 5],
            [1, 1, 1],
            MaterialChoice::Wood,
            CombineOp::Subtract,
            "Bore",
        ),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-buried-cutter`: a solid 4³-block Stone host carrying a
/// 2³-block Subtract cutter placed ENTIRELY inside it (blocks `[1,3)³` within `[0,4)³`) —
/// an internal void that is invisible by success: the composed render is just the host's
/// unbroken outer surface. The CUTTER is the active selection (not the host), so the
/// selected-operand ghost must render the cutter's whole body in the LOUD occluded style
/// — the buried-cutter golden, deliberately more obvious than leaving an internal void
/// invisible. The cutter carries the Wood material, which must appear nowhere (a Subtract is
/// an occupancy-only mask).
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_buried_cutter(voxels_per_block: u32) -> DemoScene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = Scene::from_nodes(vec![
        make(
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
            "Host",
        ),
        // Placed AFTER the host ⇒ it carves (document-order fold). Spans blocks [1,3)³
        // — strictly interior to the host's [0,4)³, so the void never reaches a face.
        make(
            [2, 2, 2],
            [1, 1, 1],
            MaterialChoice::Wood,
            CombineOp::Subtract,
            "Buried cutter",
        ),
    ]);
    scene.ensure_node_ids();
    scene.voxels_per_block = voxels_per_block;
    // Arrive with the CUTTER selected (the demo's whole point): the operand ghost x-rays it.
    let cutter = scene.roots.get(1).copied();
    DemoScene::selecting(scene, cutter)
}

/// Build the `--demo-child-booleans` scene: a Group whose 4³-block
/// Stone body carries TWO Subtract cutters — a corner cutter whose carve faces are exposed
/// (ghosting QUIET where camera-visible) and a 1³-block cutter buried STRICTLY inside the
/// body (an invisible-by-success void, ghosting wholly LOUD). The scene itself is finished
/// (no selection, no mode baked in): the golden pins the two viewer modes by flag —
/// `--select-root --view-mode booleans` x-rays both cutters, `--view-mode normal` shows
/// the finished carved look with zero ghosts.
pub(crate) fn build_demo_child_booleans(voxels_per_block: u32) -> DemoScene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = Scene::from_nodes(vec![NodeBuilder::group(
        "Carved assembly",
        vec![
            make(
                [4, 4, 4],
                [0, 0, 0],
                MaterialChoice::Stone,
                CombineOp::Union,
                "Body",
            )
            .into(),
            // Spans blocks [2,4)³ — the body's top corner octant. Its carve faces are
            // exposed on the notch (quiet); the rest sits behind body walls (loud).
            make(
                [2, 2, 2],
                [2, 2, 2],
                MaterialChoice::Plain,
                CombineOp::Subtract,
                "Corner cutter",
            )
            .into(),
            // Spans blocks [1,2)³ — strictly interior: a void that never reaches a
            // face, so its ghost renders WHOLLY loud (the buried-portions-loud case).
            make(
                [1, 1, 1],
                [1, 1, 1],
                MaterialChoice::Wood,
                CombineOp::Subtract,
                "Buried cutter",
            )
            .into(),
        ],
    )]);
    scene.ensure_node_ids();
    scene.voxels_per_block = voxels_per_block;
    // Arrives with NOTHING selected: the golden proves the child booleans render without
    // any selection-driven ghost.
    DemoScene::selecting(scene, None)
}

/// Build the `--demo-intersect`: a solid Stone body box INTERSECTED by an
/// overlapping box placed AFTER it under [`CombineOp::Intersect`]. Only the cells present in
/// BOTH bodies survive, so the render shows exactly the overlap volume — a 2³-block cube at
/// blocks `[2,4)³` — floating where the two boxes met. The mask deliberately carries the WOOD
/// material: an Intersect is an occupancy-only mask that never stamps, so the surviving cube
/// must render STONE — visible proof that surviving cells keep their ACCUMULATED material.
///
/// [`CombineOp::Intersect`]: voxel_worker::CombineOp
pub(crate) fn build_demo_intersect(voxels_per_block: u32) -> DemoScene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make(
            [4, 4, 4],
            [0, 0, 0],
            MaterialChoice::Stone,
            CombineOp::Union,
            "Body",
        ),
        // Placed AFTER the body ⇒ it masks it (document-order fold). Spans blocks
        // [2, 6)³, overlapping the body's top corner octant [2, 4)³ — the survivor.
        make(
            [4, 4, 4],
            [2, 2, 2],
            MaterialChoice::Wood,
            CombineOp::Intersect,
            "Mask",
        ),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-group-subtract`: the SEALED-SCOPE golden.
/// A Group holds a Stone body plus a cutter placed AFTER it under [`CombineOp::Subtract`],
/// so the cutter bites a corner-octant notch out of the body — INSIDE the group. A sibling
/// Wood "bystander" box sits BEFORE the group in document order and overlaps the cutter's
/// volume: under a flat (unsealed) fold the cutter — later in depth-first order — would
/// carve the bystander too, so the bystander rendering INTACT, nestled into the notch, is
/// the visible proof that a boolean inside a scope can never affect geometry outside it.
/// The cutter carries the Plain material, which must appear nowhere (a Subtract never
/// stamps — the notch faces render Stone).
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_group_subtract(voxels_per_block: u32) -> DemoScene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        // The bystander spans blocks [3,5)³ — its lower corner octant [3,4)³ lies INSIDE
        // the cutter's box. Placed BEFORE the group, so only the scope seal protects it.
        NodeBuilder::leaf(make(
            [2, 2, 2],
            [3, 3, 3],
            MaterialChoice::Wood,
            CombineOp::Union,
            "Bystander",
        )),
        NodeBuilder::group(
            "Carved body",
            vec![
                make(
                    [4, 4, 4],
                    [0, 0, 0],
                    MaterialChoice::Stone,
                    CombineOp::Union,
                    "Body",
                )
                .into(),
                // Spans blocks [2,4)³ — the body's top corner octant — placed AFTER the
                // body inside the group, so it carves the body and nothing else.
                make(
                    [2, 2, 2],
                    [2, 2, 2],
                    MaterialChoice::Plain,
                    CombineOp::Subtract,
                    "Cutter",
                )
                .into(),
            ],
        ),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-cutter-def`: the REUSABLE CUTTER golden. ONE
/// "corner cutter" definition (a 2³-block Wood box) placed by TWO `Instance` nodes under
/// [`CombineOp::Subtract`], each overlapping its own separated 4³-block Stone host's top
/// corner octant — so the render shows two hosts wearing IDENTICAL notches carved from a
/// single stored definition (reuse by reference: editing the def re-carves every
/// placement). The def body deliberately carries the WOOD material, which must appear
/// nowhere: a Subtract instance folds the def's pre-composed body as an occupancy-only
/// mask, so every notch face renders STONE.
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_cutter_def(voxels_per_block: u32) -> DemoScene {
    let cutter_def_id = DefId(1);
    let host = |offset: [i64; 3], name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            name,
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        );
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let cut = |offset: [i64; 3], name: &str| {
        let mut node = Node::new(name, NodeContent::Instance(cutter_def_id));
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = CombineOp::Subtract;
        node
    };
    let mut scene = Scene::from_nodes(vec![
        host([0, 0, 0], "Host 1"),
        host([8, 0, 0], "Host 2"),
        // Placed AFTER the hosts ⇒ each carves (document-order fold), each at its own
        // transform over its own host's top +X/+Y corner octant (blocks [2,4)³ resp.
        // [10,12)×[2,4)²).
        cut([2, 2, 2], "Cut 1"),
        cut([10, 2, 2], "Cut 2"),
    ]);
    scene.add_definition(
        cutter_def_id,
        "Corner cutter",
        vec![{
            let shape = SdfShape::from_blocks(ShapeKind::Box, [2, 2, 2], 1, voxels_per_block);
            Node::new(
                "Cutter body",
                NodeContent::Tool {
                    shape,
                    material: MaterialChoice::Wood,
                },
            )
        }],
    );
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-window-fixture`: THE WINDOW golden.
/// A Stone wall (8×1×6 blocks, standing in the XZ plane — Z-up), then ONE `Instance`
/// of a FIXTURE definition "Window" = [opening cutter `Subtract` (3×1×3, Plain),
/// frame `Union` (3×1×1, Wood)] placed AFTER the wall at the opening's low corner.
/// Because the definition is flagged `fixture`, it does NOT pre-compose: its children
/// splice into the wall's (root) scope at the instance's spine position, in order,
/// under the instance's transform — so ONE placement both CUTS the 3×3-block hole
/// through the wall and FILLS the Wood frame bar along the hole's bottom. The render
/// shows daylight through the opening above a Wood sill; the cutter's Plain material
/// appears nowhere (a Subtract never stamps), and the instance's own operation is
/// left at the default (it is INERT on a fixture instance — the spliced children fold
/// under their own operations).
pub(crate) fn build_demo_window_fixture(voxels_per_block: u32) -> DemoScene {
    let window_def_id = DefId(1);
    let wall = {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [8, 1, 6], 1, voxels_per_block);
        Node::new(
            "Wall",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        )
    };
    let window = {
        // Placed AFTER the wall ⇒ the spliced cutter carves it (document-order
        // fold); the opening spans blocks [2,5)×[0,1)×[2,5) of the wall.
        let mut node = Node::new("Window", NodeContent::Instance(window_def_id));
        node.transform = document::scene::NodeTransform::from_blocks([2, 0, 2], voxels_per_block);
        node
    };
    let mut scene = Scene::from_nodes(vec![wall, window]);
    scene.add_definition(
        window_def_id,
        "Window",
        vec![
            {
                // The opening: cuts the full wall thickness. Its Plain material must
                // never render (an occupancy-only mask).
                let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 3], 1, voxels_per_block);
                let mut node = Node::new(
                    "Opening",
                    NodeContent::Tool {
                        shape,
                        material: MaterialChoice::Plain,
                    },
                );
                node.operation = CombineOp::Subtract;
                node
            },
            {
                // The frame: a Wood bar refilling the opening's bottom row — the
                // visible proof the SAME placement also ADDS geometry to the host.
                let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 1, 1], 1, voxels_per_block);
                Node::new(
                    "Frame",
                    NodeContent::Tool {
                        shape,
                        material: MaterialChoice::Wood,
                    },
                )
            },
        ],
    );
    scene.set_definition_fixture(window_def_id, true);
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-two-material`: two solid boxes of DISTINCT materials
/// placed SEPARATED (a whole chunk of air between them) so NO block is shared — every
/// rendered block is single-material. This is the brick-representable multi-producer scene
/// (per-record material ids shade each block from its own record); the golden locks its
/// brick render == its mesh render. The 4-block boxes sit 8 blocks apart in X (`CHUNK_
/// BLOCKS` is 4, so they land in disjoint chunks with an empty chunk between).
pub(crate) fn build_demo_two_material(voxels_per_block: u32) -> DemoScene {
    let make = |offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            format!("{material:?}"),
            NodeContent::Tool { shape, material },
        );
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![
        make([0, 0, 0], MaterialChoice::Stone),
        make([8, 0, 0], MaterialChoice::Wood),
    ]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-mixed-material`: two solid boxes of DISTINCT
/// materials whose second box is offset by a SUB-BLOCK voxel amount, so a block STRADDLES the
/// boundary and its microblocks MIX both materials — the genuinely-non-representable case that
/// a representability gate would have routed to the mesh. There is no such gate: this engages
/// the brick sink and shades each voxel from its cell-key side atlas (last-writer-wins gives the
/// Wood box the overlap voxels; the Stone voxels the offset leaves uncovered stay Stone in the
/// same block). The golden pins its brick render == its mesh render — the proof the mixed-material
/// mesh cliff is closed. The 2-voxel X offset lands mid-block for any `voxels_per_block >= 3`.
pub(crate) fn build_demo_mixed_material(voxels_per_block: u32) -> DemoScene {
    use parametric::units::Measurement;
    let stone = {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        Node::new(
            "Stone",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Stone,
            },
        )
    };
    let wood = {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(
            "Wood",
            NodeContent::Tool {
                shape,
                material: MaterialChoice::Wood,
            },
        );
        // A 2-VOXEL X offset (not a whole block), so the boundary cuts THROUGH a block —
        // that block's voxels are part Stone, part Wood: a mixed brick.
        node.transform = document::scene::NodeTransform::from_measurements(
            [
                Measurement::from_voxels(2),
                Measurement::from_voxels(0),
                Measurement::from_voxels(0),
            ],
            voxels_per_block,
        );
        node
    };
    let mut scene = with_node_ids(Scene::from_nodes(vec![stone, wood]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-village`: an **instanced** scene that
/// proves reuse-by-reference. One small "house" [`AssemblyDef`](document::scene::AssemblyDef) (a Box body Tool
/// with a Cylinder "chimney" Tool offset on top, as a `Group`) is stored ONCE in
/// `definitions`; the top-level scene places it by FOUR [`NodeContent::Instance`]
/// nodes at four different X/Z offsets. The four houses appear at four separated
/// locations from a single definition — the village-of-reused-houses case. The
/// headless capture confirms the repeated assembly shows up at multiple disjoint
/// locations.
pub(crate) fn build_demo_village(voxels_per_block: u32) -> DemoScene {
    // The default village sits at the origin; the far-scene golden
    // reuses the SAME builder with a far base offset.
    build_demo_village_at(voxels_per_block, [0, 0, 0])
}

/// Build the `--demo-village-far`: the SAME
/// instanced village as [`build_demo_village`], but with its whole composite shifted
/// to [`FAR_SCENE_BASE_BLOCKS`] (~XZ 10,000 blocks, vertical bounded). The composite
/// SPAN is unchanged (the row of four houses), so only the OFFSET is far — the
/// resolved grid is the same size as the near village, but every absolute voxel center
/// now lives ~160k voxels out, where the f32 payload is lossy. The render is still
/// crisp because the resolve rebases to the composite floating-origin in i64 before the
/// f32 downcast; this golden is the baseline a chunk-local payload move must preserve.
pub(crate) fn build_demo_village_far(voxels_per_block: u32) -> DemoScene {
    build_demo_village_at(voxels_per_block, FAR_SCENE_BASE_BLOCKS)
}

/// Shared village builder used by both [`build_demo_village`] (origin) and
/// [`build_demo_village_far`] (far). `base_offset_blocks` is added to every instance's
/// placement, shifting the WHOLE composite without changing its internal layout or
/// span. With `[0, 0, 0]` the output is byte-identical to the historical
/// `--demo-village`.
fn build_demo_village_at(voxels_per_block: u32, base_offset_blocks: [i64; 3]) -> DemoScene {
    let house_def_id = DefId(1);
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };

    // The house: a 2³ stone body with a 1×2×1 wood "chimney" sitting on top, so the
    // chimney's local offset is RELATIVE to the house (it composes down through the
    // instance + group transforms). The body is kept small (2 blocks) so that four
    // instances stay well under the renderer's drawn-instance cap and all four draw.
    // Four instances of the SAME definition in a straight row, 8 blocks apart in X
    // (a 4-block house → 4-block gap between neighbors). A row (not a 2×2 grid, in
    // which diagonal pairs self-occlude from an isometric angle) keeps all four
    // houses non-overlapping in screen space when viewed perpendicular to the row,
    // so the headless PNG unambiguously shows the repeated assembly at four
    // separated locations from a single stored definition. The shared `base_offset`
    // shifts every house equally so the far-scene variant keeps this exact layout.
    let instance = |name: &str, offset: [i64; 3]| {
        let placement = [
            offset[0] + base_offset_blocks[0],
            offset[1] + base_offset_blocks[1],
            offset[2] + base_offset_blocks[2],
        ];
        let mut node = Node::new(name, NodeContent::Instance(house_def_id));
        node.transform = document::scene::NodeTransform::from_blocks(placement, voxels_per_block);
        node
    };
    let mut scene = Scene::from_nodes(vec![
        instance("House 1", [0, 0, 0]),
        instance("House 2", [6, 0, 0]),
        instance("House 3", [12, 0, 0]),
        instance("House 4", [18, 0, 0]),
    ]);
    // The house: a 2³ stone body with a 1×2×1 wood "chimney" sitting on top, so the
    // chimney's local offset is RELATIVE to the house (it composes down through the
    // instance + group transforms). The body is kept small (2 blocks) so that four
    // instances stay well under the renderer's drawn-instance cap and all four draw.
    scene.add_definition(
        house_def_id,
        "House",
        vec![
            tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
            tool(
                ShapeKind::Cylinder,
                [1, 2, 1],
                [0, 2, 0],
                MaterialChoice::Wood,
            ),
        ],
    );
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-extrude`: a single
/// **sketch → extrude → volume** producer with a RECOGNIZABLE non-box footprint —
/// an L-shaped (plus a notch) profile on the GROUND plane, extruded UP several
/// blocks. A box obviously cannot make this footprint, so the headless capture
/// proves the new producer resolves + renders through the SAME pipeline as `SdfShape`.
///
/// The profile is an L: a `4×4`-block square with its top-right `2×2`-block
/// quadrant removed (a reflex vertex), at the document density `d`, extruded
/// `3` blocks (`3·d` voxels) along +Z (Z-up: "up"). The whole footprint is a whole
/// multiple of blocks so it sits cleanly on the lattice in the recentered render frame.
/// Build the `--demo-sketch-box <edge_voxels>` fixture: a solid cube of `edge_voxels`
/// per axis, produced through the SKETCH EXTRUDE path — a square profile of
/// `edge_voxels` per side extruded `edge_voxels` along +Z. Used to exercise the
/// two-layer / brick display + per-chunk fog at large scale (e.g. an 800³ cube) at a
/// fixed density. Profile coords are absolute voxels, so the cube's block size is
/// `edge_voxels / voxels_per_block`.
pub(crate) fn build_demo_sketch_box(edge_voxels: i64, voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let edge = edge_voxels.max(1);
    let profile = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(edge, 0),
        SketchPoint::new(edge, edge),
        SketchPoint::new(0, edge),
    ];
    let producer = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), edge as u32);
    let node = Node::new(
        "Sketch box",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Stone,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

pub(crate) fn build_demo_sketch_extrude(voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let density = i64::from(voxels_per_block);
    let two = 2 * density;
    let four = 4 * density;
    // L footprint (CCW), in voxels on the XY ground plane (PlaneAxis::Z in-plane axes
    // are X,Y): outer 0..4×0..2 blocks plus the left 0..2×2..4 block column, leaving
    // the top-right quadrant empty. Extruded UP along +Z.
    let profile = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(four, 0),
        SketchPoint::new(four, two),
        SketchPoint::new(two, two), // reflex vertex (the inside corner of the L)
        SketchPoint::new(two, four),
        SketchPoint::new(0, four),
    ];
    let producer = SketchSolid::extrude(Sketch::new(PlaneAxis::Z, profile), (3 * density) as u32);
    let node = Node::new(
        "Sketch L",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Wood,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-revolve`: a single **sketch → revolve →
/// volume** producer that is visibly a SOLID OF REVOLUTION — a stepped, vase-like
/// silhouette revolved a full 360° about the vertical Z axis. A box / extrude cannot
/// make a round, axially-symmetric, varying-radius body, so the headless capture
/// proves the revolve producer resolves + renders through the SAME pipeline as
/// `SdfShape`.
///
/// Orientation: `PlaneAxis::X` + `RevolveAxis::InPlane1` puts the AXIAL world axis on
/// Z (the vase stands up, Z-up) and the two RADIAL world axes on X and Y (the round
/// cross-section). The profile coords `(c0, c1) = (radial, axial)`, so each vertex is
/// `(radius, height)` in voxels. The silhouette: a wide foot, a pinched waist, and a
/// flared lip — a stepped vase. All extents are whole blocks so the body sits cleanly
/// on the lattice in the recentered render frame.
pub(crate) fn build_demo_sketch_revolve(voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let block = i64::from(voxels_per_block);
    // Radial profile (radius, height) in voxels, walked up one side of the silhouette
    // from the bottom of the axis, then back DOWN the axis (radius 0) to close — a
    // stepped vase: foot (r=4b) → waist (r=2b) → shoulder (r=4b) → lip (r=3b), 8 blocks
    // tall. Revolving this 360° about Z sweeps the silhouette into a round vase.
    let radial = |blocks: i64| blocks * block;
    let axial = |blocks: i64| blocks * block;
    let profile = vec![
        SketchPoint::new(0, axial(0)),         // bottom center, on the axis
        SketchPoint::new(radial(4), axial(0)), // foot outer edge
        SketchPoint::new(radial(4), axial(1)), // foot top
        SketchPoint::new(radial(2), axial(3)), // pinch in to the waist
        SketchPoint::new(radial(2), axial(5)), // waist
        SketchPoint::new(radial(4), axial(6)), // flare out to the shoulder
        SketchPoint::new(radial(3), axial(8)), // lip
        SketchPoint::new(0, axial(8)),         // top center, back on the axis
    ];
    let producer = SketchSolid::revolve(
        Sketch::new(PlaneAxis::X, profile),
        RevolveAxis::InPlane1,
        360,
    );
    let node = Node::new(
        "Sketch vase",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Stone,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-circle`: ONE whole-circle entity, extruded.
///
/// A circle has no on-curve vertex to hang a loop from, so nothing about the graph walk that
/// derives faces from segments and arcs applies to it — it closes on itself and IS a face. A round
/// disc in the render is the proof that path derives, resolves and displays; an octagon would mean
/// something flattened it, and nothing at all would mean the closed curve never reached a face.
pub(crate) fn build_demo_sketch_circle(voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let block = i64::from(voxels_per_block);
    let sketch = Sketch::circle(PlaneAxis::Z, SketchPoint::new(0, 0), 3 * block);
    let producer = SketchSolid::extrude(sketch, block as u32);
    let node = Node::new(
        "Sketch circle",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Stone,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-donut`: a square with a circle inside it, the
/// circle UNPICKED.
///
/// Two faces, one of them closed-curve, folded smallest-area-first — so the disc carves the square
/// it sits in and the render has a round hole through it. That the hole is round and the outside is
/// square is the point: one region, two kinds of boundary, no conversion between them.
pub(crate) fn build_demo_sketch_donut(voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let block = i64::from(voxels_per_block);
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let span = 8 * block;
    let corners = [
        SketchPoint::new(0, 0),
        SketchPoint::new(span, 0),
        SketchPoint::new(span, span),
        SketchPoint::new(0, span),
    ]
    .map(|at| sketch.add_free_point(at));
    for index in 0..4 {
        sketch.connect(corners[index], corners[(index + 1) % 4]);
    }
    sketch.add_circle(
        SketchPoint::new(span / 2, span / 2),
        SketchLength::new(3 * block),
    );
    // The disc is the smaller face; unpicking it is what turns the square into a ring.
    let context = document::sketch::evaluation_context_from_density(voxels_per_block)
        .expect("non-zero density returned above");
    let disc = sketch
        .identified_faces(context)
        .into_iter()
        .min_by(|a, b| a.0.area.total_cmp(&b.0.area))
        .expect("the square and the disc")
        .1;
    sketch.set_face_picked(disc, false, context);
    let producer = SketchSolid::extrude(sketch, block as u32);
    let node = Node::new(
        "Sketch donut",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Wood,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-lens`: two overlapping circles, whose LENS —
/// the almond of overlap — is unpicked.
///
/// The headline of the arrangement. Nothing here shares a point: the circles simply cross, and the
/// arrangement cuts each of them at the two crossings, so the drawing is three separately pickable
/// regions rather than two. Carving the middle one is what makes that visible — the render is two
/// crescents with a gap between them, which no pair of whole circles could produce.
pub(crate) fn build_demo_sketch_lens(voxels_per_block: u32) -> DemoScene {
    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let block = i64::from(voxels_per_block);
    let radius = 4 * block;
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    // Centers 1.5 radii apart, so the lens is a substantial third of the drawing.
    sketch.add_circle(SketchPoint::new(0, 0), SketchLength::new(radius));
    sketch.add_circle(
        SketchPoint::new(radius + radius / 2, 0),
        SketchLength::new(radius),
    );
    // The lens is the smallest of the three faces; unpicking it splits the pair into crescents.
    let context = document::sketch::evaluation_context_from_density(voxels_per_block)
        .expect("non-zero density returned above");
    let lens = sketch
        .identified_faces(context)
        .into_iter()
        .min_by(|a, b| a.0.area.total_cmp(&b.0.area))
        .expect("two crescents and a lens")
        .1;
    sketch.set_face_picked(lens, false, context);
    let producer = SketchSolid::extrude(sketch, block as u32);
    let node = Node::new(
        "Sketch lens",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Wood,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-sketch-constraints`: a rectangle carrying one badge of every ANCHORING
/// species, for capture with `--enter-sketch`.
///
/// The species are what the drawing is for, because a badge's seat is chosen differently for each:
/// `Horizontal` and `Vertical` stand beside a SEGMENT, `Perpendicular` and `Equal` stand in the
/// corner between a PAIR, and `Fix` stands on a POINT. Ten successive reports said these marks were
/// not coplanar with the sketch plane; every one was closed against a number, because no capture
/// could draw them. This is the picture.
pub(crate) fn build_demo_sketch_constraints(voxels_per_block: u32) -> DemoScene {
    use document::sketch::ConstraintKind;

    if voxels_per_block == 0 {
        return invalid_density_demo(voxels_per_block);
    }
    let block = i64::from(voxels_per_block);
    let (run, rise) = (8 * block, 5 * block);
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corners = [
        SketchPoint::new(0, 0),
        SketchPoint::new(run, 0),
        SketchPoint::new(run, rise),
        SketchPoint::new(0, rise),
    ]
    .map(|at| sketch.add_free_point(at));
    let edges: Vec<_> = (0..4)
        .map(|index| {
            sketch
                .connect(corners[index], corners[(index + 1) % 4])
                .expect("four distinct corners")
        })
        .collect();
    let (bottom, right, top, left) = (edges[0], edges[1], edges[2], edges[3]);

    let context = document::sketch::evaluation_context_from_density(voxels_per_block)
        .expect("non-zero density returned above");
    let mut producer = SketchSolid::extrude(sketch, block as u32);
    // Already true of the rectangle as authored, every one of them — so the badges appear without
    // the solve moving a single point, and the render pins the MARKS rather than a re-solve.
    for kind in [
        ConstraintKind::Horizontal { segment: bottom },
        ConstraintKind::Vertical { segment: left },
        ConstraintKind::Perpendicular {
            first: bottom,
            second: right,
        },
        ConstraintKind::Equal {
            first: bottom,
            second: top,
        },
        ConstraintKind::Fix {
            point: corners[0],
            at: SketchPoint::new(0, 0),
        },
    ] {
        producer = producer
            .with_constraint(kind, context)
            .expect("a relation the rectangle already honours")
            .0;
    }

    let node = Node::new(
        "Sketch constraints",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Wood,
        },
    );
    let mut scene = with_node_ids(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-groups`: a scene that
/// exercises the indented TREE in the panel. A top-level `Group` ("Cluster") holds
/// two child Tools (a Sphere + a Box at a small offset); a sibling top-level Box
/// Tool sits beside it; and one `Instance` of a small "Widget" definition sits
/// beyond. So the captured panel node list shows: the Group with its two children
/// nested+indented under it, a top-level Tool, and an Instance row, plus the
/// Definitions list — the whole authoring surface this step adds.
pub(crate) fn build_demo_groups(voxels_per_block: u32) -> DemoScene {
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material, name: &str| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };

    let widget_def_id = DefId(1);

    // A Group with two children, placed at the origin; the children carry their own
    // local offsets relative to the Group.
    let cluster = NodeBuilder::group_at(
        "Cluster",
        [0, 0, 0],
        voxels_per_block,
        vec![
            tool(
                ShapeKind::Sphere,
                [2, 2, 2],
                [0, 0, 0],
                MaterialChoice::Stone,
                "Core",
            )
            .into(),
            tool(
                ShapeKind::Box,
                [2, 2, 2],
                [3, 0, 0],
                MaterialChoice::Wood,
                "Shell",
            )
            .into(),
        ],
    );

    let lone = tool(
        ShapeKind::Box,
        [2, 2, 2],
        [8, 0, 0],
        MaterialChoice::Wood,
        "Lone",
    );
    let mut widget_instance = Node::new("Widget instance", NodeContent::Instance(widget_def_id));
    widget_instance.transform =
        document::scene::NodeTransform::from_blocks([12, 0, 0], voxels_per_block);

    let mut scene = Scene::from_nodes(vec![
        cluster,
        NodeBuilder::leaf(lone),
        NodeBuilder::leaf(widget_instance),
    ]);
    scene.add_definition(
        widget_def_id,
        "Widget",
        vec![tool(
            ShapeKind::Sphere,
            [2, 2, 2],
            [0, 0, 0],
            MaterialChoice::Plain,
            "Ball",
        )],
    );
    scene.voxels_per_block = voxels_per_block;
    DemoScene::first_node(scene)
}

/// Build the `--demo-far-offset` / `--demo-far-offset-near` scene: a single small
/// recognizable box Tool placed either
/// at the FAR offset ([`FAR_OFFSET_BLOCKS`], i.e. 100_000 blocks in X) or at the
/// ORIGIN, for an A/B precision baseline.
///
/// The box is a 4³-block solid (a crisp, unambiguous shape that frames cleanly).
/// At density 16 the far placement sits 1.6M voxels from the origin in ABSOLUTE
/// composite space — which the CPU placement test in `scene.rs` asserts directly.
///
/// `Scene::resolve_region` recenters the composite on its OWN center, so a lone far
/// box is recentered straight back to the origin before rendering. The far and near
/// renders therefore look identical, and f32 jitter from the large offset cannot show
/// up in the live render while the recenter stands. The flag is the visual regression
/// target for origin-rebasing work: it must STAY jitter-free.
pub(crate) fn build_far_offset_scene(voxels_per_block: u32, far: bool) -> Scene {
    let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
    let mut node = Node::new(
        if far { "Far box" } else { "Near box" },
        NodeContent::Tool {
            shape,
            material: MaterialChoice::Stone,
        },
    );
    node.transform = document::scene::NodeTransform::from_blocks(
        if far { FAR_OFFSET_BLOCKS } else { [0, 0, 0] },
        voxels_per_block,
    );
    let mut scene = Scene::single_node(node);
    scene.voxels_per_block = voxels_per_block;
    scene
}
