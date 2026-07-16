//! The hardcoded demo scene builders (`--demo-*`, `--demo-far-offset`) and the
//! texture-stem file helpers they and the `--force-demo-stem` path share.

use voxel_worker::{
    CombineOp, DefId, MaterialChoice, Node, NodeBuilder, NodeContent, PlaneAxis, RevolveAxis,
    Scene, SdfShape, ShapeKind, Sketch, SketchPoint, SketchSolid,
};

/// The block offset of the far-offset demo box (ADR 0002 streaming S1; S4b makes it
/// jitter-free). A large block offset, resolved through the now-`i64` voxel offset
/// (widened in S4a). At
/// density 16 this is **16 million voxels** from the origin — past the f32
/// exact-integer ceiling (2²⁴ ≈ 16.7M), where the old recentre-AFTER-f32-add path
/// lost the voxel-centre `.5` fraction on EVERY voxel (the real precision breakdown
/// the S1 flag exists to expose). The S4b camera-relative rebase (subtract the
/// floating origin in i64 BEFORE the f32 downcast) renders this byte-identical to the
/// near box. (At the previous 100_000 the f32 ULP at 1.6M is 0.125, so `.5` survived
/// and the box never actually jittered — only the demo's UI text differed.)
pub(crate) const FAR_OFFSET_BLOCKS: [i64; 3] = [1_000_000, 0, 0];

/// The block offset that places the `--demo-village-far` composite at the FAR end of
/// the anisotropic horizontal extent (ADR 0010 D0 / ADR 0003 §G3): ~10,000 blocks on
/// both horizontal axes (X and Z), with the VERTICAL axis (Z-up → index 2 is vertical;
/// the horizontal ground plane is X/Y) bounded. Per the project's Z-up convention the
/// two HORIZONTAL axes are X (index 0) and Y (index 1), and the VERTICAL axis is Z
/// (index 2) — so the far horizontal offset goes on X and Y and the vertical Z stays at
/// 0. At density 16 this sits 160,000 voxels from the origin per horizontal axis, where
/// an absolute f32 voxel centre has barely a fractional bit left (the precision loss the
/// §3a chunk-local-integer payload exists to remove). The composite SPAN stays small (a
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

/// Mint stable [`NodeId`]s for a freshly-built demo scene and select its first
/// top-level node by id (ADR 0003 Phase B3: selection is keyed by [`NodeId`], so a
/// demo built with positional intent ("select node 0") must resolve that to an id
/// after minting). The later `ensure_node_ids` on the load path is idempotent.
fn selecting_first_node(mut scene: Scene) -> Scene {
    scene.ensure_node_ids();
    scene.active = scene.roots.first().copied();
    scene
}

/// Build the `--demo-scene` (ADR 0001 step 3): a hardcoded multi-node PLACED
/// scene proving disjoint placement. A sphere at the origin, a box offset +8
/// blocks in X, and a torus offset +6 blocks in Z. Each Tool is 5 blocks, so the
/// offsets open clear gaps and the three solids sit visibly apart (no overlap at
/// the origin) — the headless check the demo exists to confirm.
///
/// NOTE (ADR deviation): the task example named a clouds Part as the third node.
/// The `DebugClouds` Part has no intrinsic bounded size — it fills whatever region
/// it is handed (a bounded stored body is a later Part variant), so as a region-
/// filling fog it would densely OCCLUDE the sphere and box and defeat the very
/// separation the demo verifies. A third SDF Tool (torus) is a crisp, bounded
/// solid that makes the disjoint placement unambiguous in the PNG. Part placement
/// itself is covered by the scene.rs unit tests (a Part stamps under its offset),
/// and the in-app inspector offsets both Tools and Parts.
pub(crate) fn build_demo_scene(voxels_per_block: u32) -> Scene {
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(
            format!("{kind:?}"),
            NodeContent::Tool { shape, material },
        );
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
    ]));
    // Density is document-level (ADR 0003 §3f(0)).
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-overlap` (ADR 0010 E3 / #50): two solid boxes of DIFFERENT materials
/// placed so they OVERLAP, exercising the multi-material overlap case (the E2 carry-over).
/// The overlap region resolves last-writer-wins by document order (the Wood box is second,
/// so it wins where they overlap), and the golden pins that the dense and two-layer paths
/// render this IDENTICALLY. The boxes are 4 blocks each, offset 2 blocks in X+Y so a corner
/// volume overlaps; their union is a recognizable two-tone L-ish solid.
pub(crate) fn build_demo_overlap(voxels_per_block: u32) -> Scene {
    let make = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        make(ShapeKind::Box, [0, 0, 0], MaterialChoice::Stone),
        make(ShapeKind::Box, [2, 2, 0], MaterialChoice::Wood),
    ]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-subtract` (ADR 0017 / #73): a solid Stone box CARVED by a smaller
/// box placed AFTER it under [`CombineOp::Subtract`] — the ordered-fold tracer golden. The
/// cutter is a 2³-block box overlapping the Stone box's top +X/+Y corner octant, so the
/// render shows a crisp cubic NOTCH bitten out of the corner. The cutter deliberately
/// carries the WOOD material: a Subtract is an occupancy-only mask that never stamps, so
/// every newly-exposed face inside the notch must render STONE — visible proof that
/// surviving cells keep their material.
///
/// [`CombineOp::Subtract`]: voxel_worker::CombineOp
pub(crate) fn build_demo_subtract(voxels_per_block: u32) -> Scene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        make([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union, "Body"),
        // Placed AFTER the body ⇒ it carves it (document-order fold). Spans blocks
        // [2, 4)³ inside the body plus empty space beyond — the corner octant notch.
        make([2, 2, 2], [2, 2, 2], MaterialChoice::Wood, CombineOp::Subtract, "Cutter"),
    ]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-group-subtract` (ADR 0017 Decision 3 / #74): the SEALED-SCOPE golden.
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
pub(crate) fn build_demo_group_subtract(voxels_per_block: u32) -> Scene {
    let make = |size: [u32; 3], offset: [i64; 3], material, operation, name: &str| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, size, 1, voxels_per_block);
        let mut node = Node::new(name, NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node.operation = operation;
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        // The bystander spans blocks [3,5)³ — its lower corner octant [3,4)³ lies INSIDE
        // the cutter's box. Placed BEFORE the group, so only the scope seal protects it.
        NodeBuilder::Leaf(make(
            [2, 2, 2],
            [3, 3, 3],
            MaterialChoice::Wood,
            CombineOp::Union,
            "Bystander",
        )),
        NodeBuilder::group(
            "Carved body",
            vec![
                make([4, 4, 4], [0, 0, 0], MaterialChoice::Stone, CombineOp::Union, "Body").into(),
                // Spans blocks [2,4)³ — the body's top corner octant — placed AFTER the
                // body inside the group, so it carves the body and nothing else.
                make([2, 2, 2], [2, 2, 2], MaterialChoice::Plain, CombineOp::Subtract, "Cutter")
                    .into(),
            ],
        ),
    ]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-two-material` (ADR 0011 G2): two solid boxes of DISTINCT materials
/// placed SEPARATED (a whole chunk of air between them) so NO block is shared — every
/// rendered block is single-material. This is the brick-representable multi-producer scene
/// (per-record material ids shade each block from its own record); the golden locks its
/// brick render == its mesh render. The 4-block boxes sit 8 blocks apart in X (`CHUNK_
/// BLOCKS` is 4, so they land in disjoint chunks with an empty chunk between).
pub(crate) fn build_demo_two_material(voxels_per_block: u32) -> Scene {
    let make = |offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node = Node::new(format!("{material:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut scene = selecting_first_node(Scene::from_nodes(vec![
        make([0, 0, 0], MaterialChoice::Stone),
        make([8, 0, 0], MaterialChoice::Wood),
    ]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-mixed-material` (material atlas / ADR 0013): two solid boxes of DISTINCT
/// materials whose second box is offset by a SUB-BLOCK voxel amount, so a block STRADDLES the
/// boundary and its microblocks MIX both materials — the genuinely-non-representable case that
/// the deleted representability gate used to route to the mesh. With the gate gone this engages
/// the brick sink and shades each voxel from its cell-key side atlas (last-writer-wins gives the
/// Wood box the overlap voxels; the Stone voxels the offset leaves uncovered stay Stone in the
/// same block). The golden pins its brick render == its mesh render — the proof the mixed-material
/// mesh cliff is closed. The 2-voxel X offset lands mid-block for any `voxels_per_block >= 3`.
pub(crate) fn build_demo_mixed_material(voxels_per_block: u32) -> Scene {
    use voxel_core::units::Measurement;
    let stone = {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        Node::new("Stone", NodeContent::Tool { shape, material: MaterialChoice::Stone })
    };
    let wood = {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [4, 4, 4], 1, voxels_per_block);
        let mut node =
            Node::new("Wood", NodeContent::Tool { shape, material: MaterialChoice::Wood });
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
    let mut scene = selecting_first_node(Scene::from_nodes(vec![stone, wood]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-village` (ADR 0001 step 4): an **instanced** scene that
/// proves reuse-by-reference. One small "house" [`AssemblyDef`] (a Box body Tool
/// with a Cylinder "chimney" Tool offset on top, as a `Group`) is stored ONCE in
/// `definitions`; the top-level scene places it by FOUR [`NodeContent::Instance`]
/// nodes at four different X/Z offsets. The four houses appear at four separated
/// locations from a single definition — the village-of-reused-houses case. The
/// headless capture confirms the repeated assembly shows up at multiple disjoint
/// locations.
pub(crate) fn build_demo_village(voxels_per_block: u32) -> Scene {
    // The default village sits at the origin; the far-scene golden (ADR 0010 D0)
    // reuses the SAME builder with a far base offset.
    build_demo_village_at(voxels_per_block, [0, 0, 0])
}

/// Build the `--demo-village-far` (ADR 0010 D0 / ADR 0003 §G3, Phase D0): the SAME
/// instanced village as [`build_demo_village`], but with its whole composite shifted
/// to [`FAR_SCENE_BASE_BLOCKS`] (~XZ 10,000 blocks, vertical bounded). The composite
/// SPAN is unchanged (the row of four houses), so only the OFFSET is far — the
/// resolved grid is the same size as the near village, but every absolute voxel centre
/// now lives ~160k voxels out, where the f32 payload is lossy. The render is still
/// crisp today because the resolve rebases to the composite floating-origin in i64
/// before the f32 downcast (S4b); this golden is the baseline the §3a chunk-local
/// payload move (#48) must preserve.
pub(crate) fn build_demo_village_far(voxels_per_block: u32) -> Scene {
    build_demo_village_at(voxels_per_block, FAR_SCENE_BASE_BLOCKS)
}

/// Shared village builder used by both [`build_demo_village`] (origin) and
/// [`build_demo_village_far`] (far). `base_offset_blocks` is added to every instance's
/// placement, shifting the WHOLE composite without changing its internal layout or
/// span. With `[0, 0, 0]` the output is byte-identical to the historical
/// `--demo-village`.
fn build_demo_village_at(voxels_per_block: u32, base_offset_blocks: [i64; 3]) -> Scene {
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
    // (a 4-block house → 4-block gap between neighbours). A row (not a 2×2 grid, in
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
            tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
        ],
    );
    scene.voxels_per_block = voxels_per_block;
    selecting_first_node(scene)
}

/// Build the `--demo-sketch-extrude` (ADR 0003 §3i Slice 2a): a single
/// **sketch → extrude → volume** producer with a RECOGNIZABLE non-box footprint —
/// an L-shaped (plus a notch) profile on the GROUND plane, extruded UP several
/// blocks. A box obviously cannot make this footprint, so the headless capture
/// proves the new producer resolves + renders through the SAME pipeline as `SdfShape`.
///
/// The profile is an L: a `4×4`-block square with its top-right `2×2`-block
/// quadrant removed (a reflex vertex), at the document density `d`, extruded
/// `3` blocks (`3·d` voxels) along +Z (Z-up: "up"). The whole footprint is a whole
/// multiple of blocks so it sits cleanly on the lattice in the recentred render frame.
/// Build the `--demo-sketch-box <edge_voxels>` fixture: a solid cube of `edge_voxels`
/// per axis, produced through the SKETCH EXTRUDE path — a square profile of
/// `edge_voxels` per side extruded `edge_voxels` along +Z. Used to exercise the
/// two-layer / brick display + per-chunk fog at large scale (e.g. an 800³ cube) at a
/// fixed density. Profile coords are absolute voxels, so the cube's block size is
/// `edge_voxels / voxels_per_block`.
pub(crate) fn build_demo_sketch_box(edge_voxels: i64, voxels_per_block: u32) -> Scene {
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
    let mut scene = selecting_first_node(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

pub(crate) fn build_demo_sketch_extrude(voxels_per_block: u32) -> Scene {
    let density = voxels_per_block.max(1) as i64;
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
    let mut scene = selecting_first_node(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-sketch-revolve` (ADR 0003 §3i): a single **sketch → revolve →
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
/// on the lattice in the recentred render frame.
pub(crate) fn build_demo_sketch_revolve(voxels_per_block: u32) -> Scene {
    let block = voxels_per_block.max(1) as i64;
    // Radial profile (radius, height) in voxels, walked up one side of the silhouette
    // from the bottom of the axis, then back DOWN the axis (radius 0) to close — a
    // stepped vase: foot (r=4b) → waist (r=2b) → shoulder (r=4b) → lip (r=3b), 8 blocks
    // tall. Revolving this 360° about Z sweeps the silhouette into a round vase.
    let radial = |blocks: i64| blocks * block;
    let axial = |blocks: i64| blocks * block;
    let profile = vec![
        SketchPoint::new(0, axial(0)),          // bottom centre, on the axis
        SketchPoint::new(radial(4), axial(0)),  // foot outer edge
        SketchPoint::new(radial(4), axial(1)),  // foot top
        SketchPoint::new(radial(2), axial(3)),  // pinch in to the waist
        SketchPoint::new(radial(2), axial(5)),  // waist
        SketchPoint::new(radial(4), axial(6)),  // flare out to the shoulder
        SketchPoint::new(radial(3), axial(8)),  // lip
        SketchPoint::new(0, axial(8)),          // top centre, back on the axis
    ];
    let producer = SketchSolid::revolve(Sketch::new(PlaneAxis::X, profile), RevolveAxis::InPlane1, 360);
    let node = Node::new(
        "Sketch vase",
        NodeContent::SketchTool {
            producer,
            material: MaterialChoice::Stone,
        },
    );
    let mut scene = selecting_first_node(Scene::from_nodes(vec![node]));
    scene.voxels_per_block = voxels_per_block;
    scene
}

/// Build the `--demo-groups` (ADR 0001 step 4, UI verification): a scene that
/// exercises the indented TREE in the panel. A top-level `Group` ("Cluster") holds
/// two child Tools (a Sphere + a Box at a small offset); a sibling top-level Box
/// Tool sits beside it; and one `Instance` of a small "Widget" definition sits
/// beyond. So the captured panel node list shows: the Group with its two children
/// nested+indented under it, a top-level Tool, and an Instance row, plus the
/// Definitions list — the whole authoring surface this step adds.
pub(crate) fn build_demo_groups(voxels_per_block: u32) -> Scene {
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
            tool(ShapeKind::Sphere, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone, "Core").into(),
            tool(ShapeKind::Box, [2, 2, 2], [3, 0, 0], MaterialChoice::Wood, "Shell").into(),
        ],
    );

    let lone = tool(ShapeKind::Box, [2, 2, 2], [8, 0, 0], MaterialChoice::Wood, "Lone");
    let mut widget_instance = Node::new("Widget instance", NodeContent::Instance(widget_def_id));
    widget_instance.transform = document::scene::NodeTransform::from_blocks([12, 0, 0], voxels_per_block);

    let mut scene = Scene::from_nodes(vec![
        cluster,
        NodeBuilder::Leaf(lone),
        NodeBuilder::Leaf(widget_instance),
    ]);
    scene.add_definition(
        widget_def_id,
        "Widget",
        vec![tool(ShapeKind::Sphere, [2, 2, 2], [0, 0, 0], MaterialChoice::Plain, "Ball")],
    );
    scene.voxels_per_block = voxels_per_block;
    selecting_first_node(scene)
}

/// Build the `--demo-far-offset` / `--demo-far-offset-near` scene (ADR 0002
/// streaming S1, part of #18): a single small recognizable box Tool placed either
/// at the FAR offset ([`FAR_OFFSET_BLOCKS`], i.e. 100_000 blocks in X) or at the
/// ORIGIN, for an A/B precision baseline.
///
/// The box is a 4³-block solid (a crisp, unambiguous shape that frames cleanly).
/// At density 16 the far placement sits 1.6M voxels from the origin in ABSOLUTE
/// composite space — which the CPU placement test in `scene.rs` asserts directly.
///
/// IMPORTANT (today's render math): `Scene::resolve_region` recentres the
/// composite on its OWN centre, so a lone far box is recentred straight back to
/// the origin before rendering. The far and near renders therefore look identical
/// today — f32 jitter from the large offset cannot show up in the live render
/// until S4 removes the recentre / adds origin-rebasing. This flag exists to be
/// the visual regression target for S4 (it must STAY jitter-free once S4 lands).
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
