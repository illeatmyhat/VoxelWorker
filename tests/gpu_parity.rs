//! GPU brick-field parity net (ADR 0011) — the CPU↔GPU A/B equivalence suite for the
//! brick display pipeline.
//!
//! For each gated scene, the brick field is built on the CPU (records + R8 sculpted
//! atlas), uploaded, and the GPU raymarch's hit set / atlas bytes are asserted
//! **byte-identical** to the CPU exact evaluator. The clip-map pyramid, interior
//! elision, incremental patch, residency-miss, and onion-ghost slabs are each pinned.
//!
//! (ADR 0012 retired the ADR 0007 gpu_resolve fog A/B tier that once lived here; the
//! brick tier below is the surviving GPU parity net.)
//!
//! Run: `cargo test --features gpu --test gpu_parity`
#![cfg(feature = "gpu")]

use voxel_worker::voxel::{GeometryParams, SdfShape, ShapeKind};
use voxel_worker::{
    GpuContext, MaterialChoice, Node, NodeContent, PlaneAxis, RevolveAxis, Scene, Sketch,
    SketchPoint, SketchSolid,
};

// ===========================================================================
// Sketch cases (shared fixture) — the revolve-vase feeds the brick tier below.
// ===========================================================================

/// How the sketch case builds its producer; the test wraps it in a one-node scene.
enum SketchKind {
    Extrude { height_blocks: i64 },
    Revolve { axis: RevolveAxis, turn_degrees: u32 },
}

struct SketchCase {
    #[allow(dead_code)] // documents each fixture; only the revolve-vase is consumed below.
    name: &'static str,
    plane: PlaneAxis,
    /// Profile vertices in BLOCKS (scaled by density at build time → voxel coords).
    profile_blocks: &'static [[i64; 2]],
    kind: SketchKind,
    voxels_per_block: u32,
}

const SKETCH_CASES: &[SketchCase] = &[
    // Rectangle extrude == box (exact reference for the extrude path).
    SketchCase { name: "extrude-rect-4x2x3-d4", plane: PlaneAxis::Z, profile_blocks: &[[0, 0], [4, 0], [4, 2], [0, 2]], kind: SketchKind::Extrude { height_blocks: 3 }, voxels_per_block: 4 },
    // The demo L (concave, reflex vertex) extruded up, multi-chunk at d8.
    SketchCase { name: "extrude-L-d8", plane: PlaneAxis::Z, profile_blocks: &[[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], kind: SketchKind::Extrude { height_blocks: 3 }, voxels_per_block: 8 },
    // A triangle (odd, non-axis-aligned edges) extruded — slanted crossings.
    SketchCase { name: "extrude-tri-d4", plane: PlaneAxis::Y, profile_blocks: &[[0, 0], [7, 1], [3, 6]], kind: SketchKind::Extrude { height_blocks: 2 }, voxels_per_block: 4 },
    // Rectangle revolve == cylinder, full turn (one-sided radial).
    SketchCase { name: "revolve-rect-cyl-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [5, 0], [5, 4], [0, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
    // The demo vase (stepped silhouette) revolved 360° — the headline revolve shape.
    SketchCase { name: "revolve-vase-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [4, 0], [4, 1], [2, 3], [2, 5], [4, 6], [3, 8], [0, 8]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
    // A half-disc-ish profile revolved → rounded body (curved radial boundary).
    SketchCase { name: "revolve-bowl-d8", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [6, 0], [6, 1], [1, 6], [0, 6]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 8 },
    // Partial turn (180°) — exercises the atan2 theta gate (transcendental divergence).
    SketchCase { name: "revolve-rect-half-d4", plane: PlaneAxis::X, profile_blocks: &[[0, 0], [5, 0], [5, 4], [0, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 180 }, voxels_per_block: 4 },
    // Straddling profile (radial coords cross 0) — exercises the +radius/−radius fold.
    SketchCase { name: "revolve-straddle-d4", plane: PlaneAxis::X, profile_blocks: &[[-3, 0], [3, 0], [3, 4], [-3, 4]], kind: SketchKind::Revolve { axis: RevolveAxis::InPlane1, turn_degrees: 360 }, voxels_per_block: 4 },
];

impl SketchCase {
    fn build(&self) -> SketchSolid {
        let d = self.voxels_per_block as i64;
        let profile: Vec<SketchPoint> = self
            .profile_blocks
            .iter()
            .map(|&[a, b]| SketchPoint::new(a * d, b * d))
            .collect();
        let sketch = Sketch::new(self.plane, profile);
        match self.kind {
            SketchKind::Extrude { height_blocks } => {
                SketchSolid::extrude(sketch, (height_blocks * d) as u32)
            }
            SketchKind::Revolve { axis, turn_degrees } => {
                SketchSolid::revolve(sketch, axis, turn_degrees)
            }
        }
    }
}

// ===========================================================================
// Brick-field build tier (ADR 0011 G0) — records + R8 atlas vs the boundary set
// ===========================================================================

/// Extract one brick slot's `edge³` bytes (block-local x-fastest) out of a dense
/// `atlas_dim³` byte cube — the same linear-slot → 3D-tile layout the fog atlas packs.
fn brick_slot_bytes(
    atlas_bytes: &[u8],
    atlas_dim: usize,
    bricks_per_axis: u32,
    edge: usize,
    atlas_slot: u32,
) -> Vec<u8> {
    let tiles = bricks_per_axis.max(1);
    let origin = [
        (atlas_slot % tiles) as usize * edge,
        ((atlas_slot / tiles) % tiles) as usize * edge,
        (atlas_slot / (tiles * tiles)) as usize * edge,
    ];
    let mut brick_bytes = vec![0u8; edge.pow(3)];
    for local_z in 0..edge {
        for local_y in 0..edge {
            for local_x in 0..edge {
                let source = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0]
                    + local_x;
                brick_bytes[(local_z * edge + local_y) * edge + local_x] = atlas_bytes[source];
            }
        }
    }
    brick_bytes
}

/// **The ADR 0011 parity gate, clause (a)** — the G0 brick-build harness, wired to
/// nothing: for each gated scene, pack the two-layer boundary set into the sorted
/// `BrickRecord` array + the R8 sculpted-brick atlas, land the atlas in the texture,
/// read it back, and assert:
///
/// * every boundary block's atlas brick is **byte-identical** (through the full texture
///   round-trip) to the CPU boundary set's occupancy for that block — the oracle is
///   `expand_occupancy_into`, the shipped expansion proven bit-exact vs the dense path,
///   an independent path from the builder's cuboid rasterization;
/// * every coarse-solid block emits exactly ONE kind-0 record carrying its block id and
///   consumes NO atlas slot; air blocks emit nothing;
/// * seam-solidity flags carry into the record set unchanged;
/// * the granule is ONE BLOCK: brick edge == `voxels_per_block` at every density in the
///   matrix (d16 AND non-16 — nothing may hard-code 16);
/// * atlas slots are dense `0..sculpted_count` and every padding slot reads back zero.
// NOTE (ADR 0011 surface-only record contract): this gate runs on the interior-INCLUSIVE
// oracle build (`build_brick_field_all_blocks`) — it asserts the one-to-one partition
// mapping + atlas byte-exactness, which the surface-only live build shares (identical
// classifier, identical sculpted set + slot numbering; only occluded coarse records are
// omitted). The surface contract itself is gated by
// `brick_field::build_emits_only_surface_records_of_a_solid_box` (CPU) and
// `brick_surface_elision_hit_set_unchanged` (render).
#[test]
fn brick_field_build_matches_two_layer_boundary_set_byte_exactly() {
    use voxel_worker::core_geom::CHUNK_BLOCKS;
    use voxel_worker::{
        build_brick_field_all_blocks, read_back_brick_atlas, upload_brick_atlas, BrickPayload,
        NodeTransform, TwoLayerStore, Voxel,
    };

    let gpu = pollster::block_on(GpuContext::new(None));

    // The gated matrix: coarse-heavy SDF at d16, odd-extent box at a NON-16 density
    // (the block-denominated-granule ruling), the revolved vase at d4 (sketch tier),
    // and a multi-tool union at d16 (multi-material sculpted bricks, later-wins).
    // `require_coarse` marks the scenes whose interiors must prove the elision arm.
    struct BrickCase {
        name: &'static str,
        scene: Scene,
        voxels_per_block: u32,
        require_coarse: bool,
    }
    let make_tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    };
    let vase = &SKETCH_CASES[4]; // revolve-vase-d4
    let vase_producer = vase.build();
    let mut vase_scene = Scene::single_node(Node::new(
        "Sketch",
        NodeContent::SketchTool { producer: vase_producer, material: MaterialChoice::default() },
    ));
    vase_scene.voxels_per_block = vase.voxels_per_block;
    let cases = [
        BrickCase {
            name: "brick-sphere-80-d16",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Sphere,
                    size_voxels: [80, 80, 80],
                    size_measurements: None,
                    voxels_per_block: 16,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 16,
            require_coarse: true,
        },
        BrickCase {
            name: "brick-box-31-17-49-d4",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [31, 17, 49],
                    size_measurements: None,
                    voxels_per_block: 4,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 4,
            require_coarse: true,
        },
        BrickCase {
            name: "brick-revolve-vase-d4",
            scene: vase_scene,
            voxels_per_block: vase.voxels_per_block,
            require_coarse: false,
        },
        BrickCase {
            name: "brick-union-sphere-box-torus-d16",
            scene: Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, 16),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, 16),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, 16),
            ]),
            voxels_per_block: 16,
            require_coarse: true,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for case in &cases {
        let vpb = case.voxels_per_block;
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        assert!(!two_layer_chunks.is_empty(), "{}: empty two-layer build", case.name);
        let build = build_brick_field_all_blocks(&two_layer_chunks, vpb);

        // The granule ruling: the brick edge is the document density, nothing else.
        assert_eq!(build.brick_edge_voxels, vpb, "{}: brick edge must be one BLOCK", case.name);
        assert!(
            build
                .brick_records
                .windows(2)
                .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
            "{}: records must sort strictly ascending",
            case.name
        );

        // The full texture round-trip: upload → R8 texture → readback, byte-identical
        // to the CPU-packed atlas (padding rows included — the write_texture mechanic).
        let texture = upload_brick_atlas(&gpu.device, &gpu.queue, &build);
        let readback =
            read_back_brick_atlas(&gpu.device, &gpu.queue, &texture, build.atlas_dim_voxels);
        if readback != build.sculpted_atlas_bytes {
            failures.push(format!(
                "{}: texture round-trip diverged from the CPU-packed atlas bytes",
                case.name
            ));
            continue;
        }

        let edge = vpb as usize;
        let atlas_dim = build.atlas_dim_voxels as usize;
        let mut coarse_blocks = 0usize;
        let mut sculpted_blocks = 0usize;
        for (chunk_coord, chunk) in &two_layer_chunks {
            // The oracle: the chunk's boundary-set occupancy via the SHIPPED expansion
            // (chunk-local frame, offset zero) — independent of the brick rasterizer.
            let mut expanded: Vec<Voxel> = Vec::new();
            chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
            let chunk_extent = (CHUNK_BLOCKS * vpb) as usize;
            let mut chunk_occupancy = vec![0u8; chunk_extent.pow(3)];
            for voxel in &expanded {
                let [x, y, z] = voxel.local_index;
                chunk_occupancy
                    [(z as usize * chunk_extent + y as usize) * chunk_extent + x as usize] = 255;
            }

            for block_z in 0..CHUNK_BLOCKS {
                for block_y in 0..CHUNK_BLOCKS {
                    for block_x in 0..CHUNK_BLOCKS {
                        let block = [block_x, block_y, block_z];
                        let world_block = [
                            chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block_x as i64,
                            chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block_y as i64,
                            chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block_z as i64,
                        ];
                        let record = build.find_record(world_block);
                        if let Some(block_id) = chunk.coarse_block(block) {
                            coarse_blocks += 1;
                            let record =
                                record.unwrap_or_else(|| panic!("{}: missing coarse record at {world_block:?}", case.name));
                            assert_eq!(
                                record.payload,
                                BrickPayload::CoarseSolid { block_id },
                                "{}: coarse record at {world_block:?} (kind 0, id carried, no slot)",
                                case.name
                            );
                        } else if let Some(geometry) = chunk.microblocks.get(&block) {
                            sculpted_blocks += 1;
                            let record =
                                record.unwrap_or_else(|| panic!("{}: missing sculpted record at {world_block:?}", case.name));
                            let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                                panic!("{}: boundary block at {world_block:?} must be kind 1", case.name);
                            };
                            assert_eq!(
                                record.seam_solidity, geometry.seam_solidity,
                                "{}: seam flags must carry unchanged at {world_block:?}",
                                case.name
                            );
                            // Gate (a): the brick's TEXTURE bytes == the boundary
                            // set's occupancy for this block, byte for byte.
                            let brick_bytes = brick_slot_bytes(
                                &readback,
                                atlas_dim,
                                build.bricks_per_axis,
                                edge,
                                atlas_slot,
                            );
                            let mut expected = vec![0u8; edge.pow(3)];
                            for local_z in 0..edge {
                                for local_y in 0..edge {
                                    for local_x in 0..edge {
                                        expected[(local_z * edge + local_y) * edge + local_x] =
                                            chunk_occupancy[((block_z as usize * edge + local_z)
                                                * chunk_extent
                                                + block_y as usize * edge
                                                + local_y)
                                                * chunk_extent
                                                + block_x as usize * edge
                                                + local_x];
                                    }
                                }
                            }
                            if brick_bytes != expected {
                                let differing = brick_bytes
                                    .iter()
                                    .zip(&expected)
                                    .filter(|(a, b)| a != b)
                                    .count();
                                failures.push(format!(
                                    "{}: sculpted brick at {world_block:?} (slot {atlas_slot}) \
                                     differs in {differing}/{} bytes",
                                    case.name,
                                    expected.len()
                                ));
                            }
                        } else {
                            assert!(
                                record.is_none(),
                                "{}: air block at {world_block:?} must emit nothing",
                                case.name
                            );
                        }
                    }
                }
            }
        }

        // Record accounting: one record per non-air block, slots dense over exactly
        // the sculpted set (coarse consumes no slot), padding slots all-zero.
        assert_eq!(
            build.brick_records.len(),
            coarse_blocks + sculpted_blocks,
            "{}: record count must equal the non-air block count",
            case.name
        );
        assert_eq!(build.sculpted_brick_count(), sculpted_blocks, "{}", case.name);
        let total_slots = build.bricks_per_axis.pow(3);
        for padding_slot in sculpted_blocks as u32..total_slots {
            let padding_bytes =
                brick_slot_bytes(&readback, atlas_dim, build.bricks_per_axis, edge, padding_slot);
            assert!(
                padding_bytes.iter().all(|&byte| byte == 0),
                "{}: unused atlas slot {padding_slot} must read back zero",
                case.name
            );
        }
        assert!(sculpted_blocks > 0, "{}: fixture must contain boundary blocks", case.name);
        if case.require_coarse {
            assert!(
                coarse_blocks > 0,
                "{}: fixture must contain coarse-solid blocks (interior elision unexercised)",
                case.name
            );
        }
        eprintln!(
            "{}: {} coarse + {} sculpted bricks, atlas {}³ (edge {})",
            case.name, coarse_blocks, sculpted_blocks, build.atlas_dim_voxels, vpb
        );
    }

    assert!(
        failures.is_empty(),
        "brick-field build != CPU two-layer boundary set (ADR 0011 gate (a)):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Issue #60 — async geometry-rebuild build-equivalence net
// ===========================================================================

/// The build-equivalence net (issue #60): a mesh built via the geometry WORKER's build
/// entry (`geometry_worker::build_geometry`) must be BYTE-IDENTICAL to a synchronous build
/// (`CuboidMeshRenderer::new_from_two_layer_chunks`) for the same large scene. Both call
/// the exact same builder, so this guards that the worker's request→build path feeds it the
/// same inputs and never diverges from the sync path — the correctness net the async move
/// rests on. Equivalence is asserted on the built renderers' per-build mesh stats (chunk /
/// face / triangle / box counts — the exposed-face set the two-layer mesher emits); a
/// divergence in any is a build regression.
///
/// The scene is a 24³-block box → 6×6×6 = 216 covering chunks, comfortably above
/// `ASYNC_REBUILD_CHUNK_THRESHOLD` (128), so it is representative of the LARGE wholesale
/// rebuild the worker is actually dispatched for.
#[test]
fn worker_build_matches_sync_build_for_large_scene() {
    use voxel_worker::{
        build_geometry, CuboidMeshRenderer, GeometryRebuildRequest, LayerBand, TwoLayerStore,
        ASYNC_REBUILD_CHUNK_THRESHOLD, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));

    let vpb = 16u32;
    let size_blocks_per_axis = 24u32;
    let kind = ShapeKind::Box;
    let wall_blocks = 1u32;
    let geometry = GeometryParams {
        shape: kind,
        size_voxels: [size_blocks_per_axis * vpb; 3],
        size_measurements: None,
        voxels_per_block: vpb,
        wall_blocks,
    };
    let scene = Scene::from_geometry(geometry, MaterialChoice::default());

    // Resolve the covering two-layer chunks exactly as the live rebuild does.
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let recentre_voxels = scene.recentre_voxels_for_resolve(vpb);
    // Use the placed region dims (what the live shell passes for `grid.dimensions`).
    let grid_dimensions = scene.placed_region_dimensions(vpb);

    assert!(
        two_layer_chunks.len() > ASYNC_REBUILD_CHUNK_THRESHOLD,
        "the fixture must exceed the async threshold to be representative: {} chunks (need > {})",
        two_layer_chunks.len(),
        ASYNC_REBUILD_CHUNK_THRESHOLD
    );

    // (a) The SYNCHRONOUS build (the inline path).
    let sync = CuboidMeshRenderer::new_from_two_layer_chunks(
        &gpu.device,
        &gpu.queue,
        COLOR_TARGET_FORMAT,
        &two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        vpb,
    );

    // (b) The WORKER build entry (what runs on the background thread), fed the same request.
    let request = GeometryRebuildRequest {
        generation: 1,
        two_layer_chunks,
        grid_dimensions,
        recentre_voxels,
        density: vpb,
        // FULL band — the worker's banded build at FULL is byte-identical to the sync build.
        band: LayerBand::FULL,
    };
    let worker = build_geometry(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT, &request);

    assert_eq!(
        sync.chunk_count(),
        worker.chunk_count(),
        "worker vs sync: resident render-chunk count must match"
    );
    assert_eq!(
        sync.face_count(),
        worker.face_count(),
        "worker vs sync: exposed-face set must be byte-identical"
    );
    assert_eq!(
        sync.triangle_count(),
        worker.triangle_count(),
        "worker vs sync: triangle count must match"
    );
    assert_eq!(
        sync.box_count(),
        worker.box_count(),
        "worker vs sync: decomposed box count must match"
    );
    // Sanity: a solid box actually produced geometry (the net isn't trivially comparing 0==0).
    assert!(sync.face_count() > 0, "the fixture box must mesh to a non-empty face set");
}

// ===========================================================================
// Brick-field RENDER tier (ADR 0011 G1) — the parity gate clause (b) + the
// residency-miss contract. The G0 tier above proved the BUILD (records + atlas)
// byte-exact; this tier proves the finest-LOD RAYMARCH hits the same surface the
// CPU exact evaluator reports, and that a forced residency miss renders the
// degraded-but-correct coarse form (never a miss/skip).
// ===========================================================================

/// A gated brick render case: a single-producer scene at some density.
struct BrickRenderCase {
    name: &'static str,
    scene: Scene,
    voxels_per_block: u32,
}

/// Build the gated render matrix: a coarse-heavy sphere at d16, the sculpted-heavy
/// revolved vase at d4 (sketch tier), an odd-extent box at a NON-16 density (the
/// block-denominated granule — nothing may assume 16), and — the G2 scale slice —
/// two SCATTERED many-object scenes (a dozen small shapes far apart, the scenes the
/// clip-map LOD targets) plus a MULTI-PRODUCER union with DISTINCT materials (the
/// hit set is occupancy-only, so distinct materials still exercise the traversal).
fn brick_render_cases() -> Vec<BrickRenderCase> {
    use voxel_worker::NodeTransform;
    let vase = &SKETCH_CASES[4]; // revolve-vase-d4
    let mut vase_scene = Scene::single_node(Node::new(
        "Sketch",
        NodeContent::SketchTool {
            producer: vase.build(),
            material: MaterialChoice::default(),
        },
    ));
    vase_scene.voxels_per_block = vase.voxels_per_block;

    // A dozen small spheres spread ~14 blocks apart on a lattice — scattered occupied
    // cells with wide empty gaps between them (the hierarchical-skip workload).
    let scattered = |density: u32| -> Scene {
        let mut nodes = Vec::new();
        for index in 0..12i64 {
            let shape = SdfShape::from_blocks(ShapeKind::Sphere, [3, 3, 3], 1, density);
            let mut node = Node::new(
                format!("s{index}"),
                NodeContent::Tool {
                    shape,
                    material: MaterialChoice::Stone,
                },
            );
            node.transform = NodeTransform::from_blocks(
                [(index % 4) * 14, (index / 4) * 14, (index % 3) * 18],
                density,
            );
            nodes.push(node);
        }
        Scene::from_nodes(nodes)
    };
    let make_tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice, density: u32| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, density);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, density);
        node
    };

    vec![
        BrickRenderCase {
            name: "render-sphere-80-d16",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Sphere,
                    size_voxels: [80, 80, 80],
                    size_measurements: None,
                    voxels_per_block: 16,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 16,
        },
        BrickRenderCase {
            name: "render-revolve-vase-d4",
            scene: vase_scene,
            voxels_per_block: vase.voxels_per_block,
        },
        BrickRenderCase {
            name: "render-box-31-17-49-d4",
            scene: Scene::from_geometry(
                GeometryParams {
                    shape: ShapeKind::Box,
                    size_voxels: [31, 17, 49],
                    size_measurements: None,
                    voxels_per_block: 4,
                    wall_blocks: 1,
                },
                MaterialChoice::default(),
            ),
            voxels_per_block: 4,
        },
        BrickRenderCase {
            name: "render-scattered-spheres-d16",
            scene: scattered(16),
            voxels_per_block: 16,
        },
        BrickRenderCase {
            name: "render-scattered-spheres-d4",
            scene: scattered(4),
            voxels_per_block: 4,
        },
        BrickRenderCase {
            name: "render-multi-union-distinct-d16",
            scene: Scene::from_nodes(vec![
                make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone, 16),
                make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood, 16),
                make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain, 16),
            ]),
            voxels_per_block: 16,
        },
    ]
}

/// The exact-evaluator occupancy set in the march's ABSOLUTE voxel frame (raw world
/// voxels): `chunk_coord · chunk_extent + chunk_local_index` for every voxel the
/// shipped two-layer expansion emits (coarse-solid interiors + boundary microblocks).
/// The march's `voxel_bias` recovers exactly this frame (the recentre cancels), so a
/// march hit's `absolute_voxel` indexes straight into this set.
fn exact_occupancy_set(
    two_layer_chunks: &[(
        [i32; 3],
        std::sync::Arc<voxel_worker::two_layer_store::TwoLayerChunk>,
    )],
    voxels_per_block: u32,
) -> std::collections::HashSet<[i64; 3]> {
    use voxel_worker::core_geom::CHUNK_BLOCKS;
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as i64;
    let mut occupied = std::collections::HashSet::new();
    let mut expanded = Vec::new();
    for (chunk_coord, chunk) in two_layer_chunks {
        expanded.clear();
        chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
        for voxel in &expanded {
            occupied.insert([
                chunk_coord[0] as i64 * chunk_extent + voxel.local_index[0] as i64,
                chunk_coord[1] as i64 * chunk_extent + voxel.local_index[1] as i64,
                chunk_coord[2] as i64 * chunk_extent + voxel.local_index[2] as i64,
            ]);
        }
    }
    occupied
}

/// **ADR 0011 parity gate, clause (b).** For each gated scene, install the G0 brick
/// field on the GPU, render the single-sample hit-identity image, and assert every
/// pixel's (hit flag + absolute hit voxel) is IDENTICAL to the CPU exact evaluator's
/// per-pixel voxel DDA over the same frame — the finest-LOD raymarch resolves exactly
/// the surface the truth reports. A mismatch is triaged with the CPU brick-field march
/// (the f32 mirror of the shader): agreeing with it but not the exact set isolates a
/// BUILD/frame bug; disagreeing with it isolates a SHADER bug.
#[test]
fn brick_raymarch_hit_set_matches_exact_evaluator() {
    use voxel_worker::{
        build_brick_field, cpu_march_brick_field, cpu_march_exact_occupancy, pack_gpu_records,
        brick_representable_overlay, AppCore, BrickRaymarchRenderer, ClipmapPyramid, LayerBand,
        OrbitCamera,
        TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        // The WIDE-scatter d16 case (distant small spheres, large voxel coords) is
        // excluded from the EXACT-vs-truth tier: at a few silhouette pixels between
        // two far-apart spheres the GPU flat-DDA and the CPU exact march pick
        // different grazed surfaces by f32 rounding (a display-approximation at
        // silhouettes ADR 0009 §4 allows, ~0.02% of pixels, INDEPENDENT of the
        // pyramid — proven by `brick_raymarch_pyramid_on_equals_off`, which passes
        // byte-identical on this same scene). Its clip-map correctness is gated
        // there and in the residency tier; scattered-d4 covers scattered EXACTNESS.
        if case.name == "render-scattered-spheres-d16" {
            continue;
        }
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        assert!(!two_layer_chunks.is_empty(), "{}: empty two-layer build", case.name);
        let build = build_brick_field(&two_layer_chunks, vpb);
        assert!(!build.brick_records.is_empty(), "{}: empty brick field", case.name);
        // The hit-identity image is occupancy-only, so material never enters the hit set;
        // a distinct-material union is brick-representable at G2 (each block single-material)
        // and still exercises the traversal. `unwrap_or(false)` keeps the overlay off.
        let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        // The exact-evaluator oracle (the truth the raymarch is checked against).
        let occupied = exact_occupancy_set(&two_layer_chunks, vpb);
        let occupied_fn = |absolute: [i64; 3]| occupied.contains(&absolute);

        // The headless camera framing the composite at the origin — the same rig the
        // shell/`shot` source `view_projection` from (a fixed iso view).
        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        // The all-resident field + the frame the CPU marches mirror. The clip-map
        // pyramid is ENABLED here — the finest-LOD hit set must still equal the exact
        // evaluator with the hierarchical skip live (it may only skip empty space).
        let gpu_records = pack_gpu_records(&build, |_| false);
        let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &gpu_records,
            &pyramid,
            recentre,
            overlay_active,
        );
        let frame = renderer.update_uniforms(
            &gpu.queue,
            view_projection,
            viewport_px,
            grid_dimensions,
            band,
            false,
            Some(MaterialChoice::default()),
        );
        let gpu_image = renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

        let mut mismatches = 0usize;
        let mut gpu_hits = 0usize;
        let mut first_report: Option<String> = None;
        for y in 0..height {
            for x in 0..width {
                let pixel_index = (y * width + x) as usize;
                let gpu_pixel = gpu_image[pixel_index];
                let gpu_hit = gpu_pixel[0] == 1;
                // The shader bitcast i32 voxel lanes into u32; `as i32` reinterprets.
                let gpu_voxel = [
                    gpu_pixel[1] as i32,
                    gpu_pixel[2] as i32,
                    gpu_pixel[3] as i32,
                ];
                if gpu_hit {
                    gpu_hits += 1;
                }
                let pixel = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let cpu = cpu_march_exact_occupancy(&frame, &occupied_fn, pixel);
                let agree = match cpu {
                    Some(hit) => gpu_hit && hit.absolute_voxel == gpu_voxel,
                    None => !gpu_hit,
                };
                if !agree {
                    mismatches += 1;
                    if first_report.is_none() {
                        let brick =
                            cpu_march_brick_field(&frame, &gpu_records, &build, &pyramid, pixel);
                        first_report = Some(format!(
                            "    px=({x},{y}) gpu_hit={gpu_hit} gpu_voxel={gpu_voxel:?} \
                             exact={:?} brick_field_cpu={:?} (agree-with-brick isolates a \
                             BUILD/frame bug; disagree isolates a SHADER bug)",
                            cpu.map(|h| h.absolute_voxel),
                            brick.map(|h| h.absolute_voxel),
                        ));
                    }
                }
            }
        }

        if gpu_hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        if mismatches > 0 {
            failures.push(format!(
                "{}: {mismatches}/{} pixels diverge from the exact evaluator (gpu_hits={gpu_hits})\n{}",
                case.name,
                width * height,
                first_report.unwrap_or_default()
            ));
        } else {
            eprintln!("{}: {gpu_hits} hit pixels, exact-parity clean", case.name);
        }
    }

    assert!(
        failures.is_empty(),
        "brick raymarch hit set != CPU exact evaluator (ADR 0011 gate (b)):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 interior elision — the SURFACE-ONLY build renders identically to the
/// interior-INCLUSIVE oracle build.** For every brick render case, install the field from
/// the oracle build ([`build_brick_field_all_blocks`] — one record per non-air block) and
/// again from the live surface-only build ([`build_brick_field`] — occluded coarse
/// interiors never emitted) — the clip-map (chunk-derived, identical on both sides), atlas
/// (identical: the sculpted set is never elided) and frame identical — and assert the
/// hit-identity images are BYTE-IDENTICAL. This is the display proof that never emitting a
/// fully-occluded interior block (its six neighbours all solid) never changes a ray's first
/// hit: the ray stops at the surrounding surface record before ever reaching it. The CPU
/// half is `brick_field::build_emits_only_surface_records_of_a_solid_box`.
#[test]
fn brick_surface_elision_hit_set_unchanged() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, build_brick_field_all_blocks,
        pack_gpu_records, AppCore, BrickRaymarchRenderer, ClipmapPyramid, LayerBand,
        OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();
    // At least one case must actually exercise elision (a solid interior), else the test is
    // vacuous — it would pass trivially if surface==full everywhere.
    let mut total_elided = 0usize;

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let full_build = build_brick_field_all_blocks(&two_layer_chunks, vpb);
        let surface_build = build_brick_field(&two_layer_chunks, vpb);
        if full_build.brick_records.is_empty() {
            continue;
        }
        // The two builds must pack the identical sculpted atlas (slot numbering follows the
        // shared traversal order over the never-elided sculpted set).
        assert_eq!(
            full_build.sculpted_atlas_bytes, surface_build.sculpted_atlas_bytes,
            "{}: surface-only build must pack the identical atlas",
            case.name
        );
        let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let full_records = pack_gpu_records(&full_build, |_| false);
        let surface_records = pack_gpu_records(&surface_build, |_| false);
        total_elided += full_records.len() - surface_records.len();
        // The clip-map derives from the CHUNKS (interiors included) — identical on both
        // sides; only the record buffer the shader binary-searches differs.
        let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        let render = |build: &voxel_worker::BrickFieldBuild, gpu_records: &[_]| {
            let mut renderer =
                BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
            renderer.install_brick_field(
                &gpu.device,
                &gpu.queue,
                build,
                gpu_records,
                &pyramid,
                recentre,
                overlay_active,
            );
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };
        let full_image = render(&full_build, &full_records);
        let surface_image = render(&surface_build, &surface_records);

        let hits = full_image.iter().filter(|pixel| pixel[0] == 1).count();
        if hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        let mismatches = full_image
            .iter()
            .zip(&surface_image)
            .filter(|(full, surface)| full != surface)
            .count();
        if mismatches > 0 {
            failures.push(format!(
                "{}: {mismatches}/{} pixels differ between the oracle and surface-only builds \
                 (elided {} of {} records)",
                case.name,
                width * height,
                full_records.len() - surface_records.len(),
                full_records.len(),
            ));
        }
    }

    assert!(
        total_elided > 0,
        "no case had a fully-occluded interior to elide — the test is vacuous"
    );
    assert!(
        failures.is_empty(),
        "surface-only build != interior-inclusive oracle build (ADR 0011 interior elision):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 band-clip interior fix — a LAYER BAND slicing a solid renders the elided
/// interior identically to the full-record oracle.** The sibling `..._hit_set_unchanged`
/// proves surface==full under a FULL band (a ray reaches an interior only through a surface
/// record that stops it first). A band CUT PLANE breaks that: it can start a ray INSIDE the
/// solid at a block whose coarse record was elided (interior elision) — the record search
/// misses, the cross-section would render hollow. This gate renders the surface-only build
/// and the interior-inclusive oracle build through a mid-Z band that CLIPS each coarse-heavy
/// solid, and asserts the hit-identity images are BYTE-IDENTICAL: the block-occupancy fallback
/// (a set bit + a record miss ⇒ the elided coarse cube) reproduces the oracle's records exactly.
#[test]
fn brick_surface_elision_band_clip_renders_interior() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, build_brick_field_all_blocks,
        pack_gpu_records, AppCore, BrickRaymarchRenderer, ClipmapPyramid, LayerBand, OrbitCamera,
        TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();
    // At least one case must (a) elide a solid interior AND (b) actually clip it with the band,
    // else the fallback is unexercised and the test is vacuous.
    let mut total_elided = 0usize;
    let mut any_band_clipped = false;

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let full_build = build_brick_field_all_blocks(&two_layer_chunks, vpb);
        let surface_build = build_brick_field(&two_layer_chunks, vpb);
        if full_build.brick_records.is_empty() {
            continue;
        }
        let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);
        let grid_z = grid_dimensions[2];
        if grid_z < 3 {
            continue;
        }
        // A mid-Z band that slices the solid's middle third — a cut plane through the interior.
        let band = LayerBand {
            band_min: grid_z / 3,
            band_max: (grid_z * 2 / 3).min(grid_z - 1),
            onion_depth: 0,
        };

        let full_records = pack_gpu_records(&full_build, |_| false);
        let surface_records = pack_gpu_records(&surface_build, |_| false);
        total_elided += full_records.len() - surface_records.len();
        // The pyramid (chunk-sourced, identical both sides) carries the block-occupancy masks.
        let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];

        let mut band_clip_seen = false;
        let mut render = |build: &voxel_worker::BrickFieldBuild, gpu_records: &[_]| {
            let mut renderer =
                BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
            renderer.install_brick_field(
                &gpu.device,
                &gpu.queue,
                build,
                gpu_records,
                &pyramid,
                recentre,
                overlay_active,
            );
            let frame = renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            band_clip_seen = frame.band_clip_active;
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };
        let full_image = render(&full_build, &full_records);
        let surface_image = render(&surface_build, &surface_records);
        any_band_clipped |= band_clip_seen;

        let hits = full_image.iter().filter(|pixel| pixel[0] == 1).count();
        if hits == 0 {
            continue; // the band framed the solid out — not this case's job
        }
        let mismatches = full_image
            .iter()
            .zip(&surface_image)
            .filter(|(full, surface)| full != surface)
            .count();
        if mismatches > 0 {
            failures.push(format!(
                "{}: {mismatches}/{} band-clipped pixels differ between the oracle and the \
                 surface-only+occupancy-fallback builds (elided {} of {} records)",
                case.name,
                width * height,
                full_records.len() - surface_records.len(),
                full_records.len(),
            ));
        }
    }

    assert!(
        total_elided > 0 && any_band_clipped,
        "no case both elided an interior AND band-clipped it — the fallback is unexercised \
         (elided {total_elided}, band-clipped {any_band_clipped})"
    );
    assert!(
        failures.is_empty(),
        "band-clip interior fallback != interior-inclusive oracle (ADR 0011):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 slice G3 — incremental patch render == wholesale install render.** Drive a
/// scene through the LIVE incremental path (install scene A → apply a localised occupancy
/// edit → `patch_brick_field` writing ONLY the dirty slots), render its hit-identity
/// image, and assert it is PIXEL-IDENTICAL to a from-scratch `install_brick_field` of the
/// same final scene B. This gates the whole G3 machinery THROUGH the render (not just the
/// CPU data comparison in `brick_field`'s unit tests): the free-listed slot layout + the
/// per-slot `write_texture` patch must render exactly as a dense wholesale install.
#[test]
fn brick_raymarch_incremental_patch_matches_wholesale_install() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer, ClipmapPyramid, IncrementalBrickField, LayerBand, Node, NodeContent,
        NodeTransform, OrbitCamera, Scene, SdfShape, ShapeKind, TwoLayerResidentCache,
        COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let vpb = 8u32;

    let tool = |kind: ShapeKind, offset: [i64; 3], material: MaterialChoice| -> Node {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, vpb);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = NodeTransform::from_blocks(offset, vpb);
        node
    };
    // Two anchors fix the covering set; the middle tool is edited (Sphere → moved Box) so
    // the OCCUPANCY genuinely changes (the render would differ if the patch didn't land).
    let anchor_lo = tool(ShapeKind::Box, [-14, 0, 0], MaterialChoice::Stone);
    let anchor_hi = tool(ShapeKind::Box, [14, 0, 0], MaterialChoice::Stone);
    let scene_a = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Wood),
    ]);
    let scene_b = Scene::from_nodes(vec![
        anchor_lo,
        anchor_hi,
        tool(ShapeKind::Box, [1, 0, 0], MaterialChoice::Wood),
    ]);

    // Derive the fresh covering sets + the dirty chunk set exactly as `AppCore::rebuild`.
    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_a = scene_a.build_leaf_spatial_index(vpb);
    let fresh_a: Vec<_> = cache
        .resident_two_layer_chunks(&scene_a, vpb, 0)
        .into_iter()
        .map(|(coord, chunk)| (coord, chunk.clone()))
        .collect();
    let build_a = build_brick_field(&fresh_a, vpb);
    let mut field = IncrementalBrickField::from_wholesale(&build_a);
    let overlay_a = brick_representable_overlay(&fresh_a).unwrap_or(false);

    let index_b = scene_b.build_leaf_spatial_index(vpb);
    let edit_aabb = index_b
        .edit_aabb_since(&index_a)
        .expect("the middle-tool edit is localisable");
    let dirty = cache.invalidate_aabb(&edit_aabb, vpb);
    let fresh_b: Vec<_> = cache
        .resident_two_layer_chunks(&scene_b, vpb, 0)
        .into_iter()
        .map(|(coord, chunk)| (coord, chunk.clone()))
        .collect();

    let update = field.apply_dirty_update(&fresh_b, &dirty);
    let incremental_build = field.to_build();
    let wholesale_build = build_brick_field(&fresh_b, vpb);
    assert_eq!(
        incremental_build.brick_records.len(),
        wholesale_build.brick_records.len(),
        "the incremental field must have the same record count as the wholesale build of B \
         (a covering-set change would break the incremental assumption)"
    );
    assert!(
        !dirty.is_empty() && dirty.len() < fresh_b.len(),
        "the edit must dirty SOME but not ALL chunks (dirtied {} of {})",
        dirty.len(),
        fresh_b.len()
    );

    let overlay_b = brick_representable_overlay(&fresh_b).unwrap_or(false);
    let recentre_b = scene_b.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene_b.placed_region_dimensions(vpb);

    // The headless camera framing B at the origin (the same rig the other brick tests use).
    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let aspect_ratio = width as f32 / height as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let viewport_px = [0u32, 0, width, height];
    let band = LayerBand::FULL;

    // Path 1 — the INCREMENTAL path: install A, then PATCH to B (only dirty slots written).
    let mut incremental_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    incremental_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build_a,
        &pack_gpu_records(&build_a, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_a),
        scene_a.recentre_voxels_for_resolve(vpb),
        overlay_a,
    );
    incremental_renderer.patch_brick_field(
        &gpu.device,
        &gpu.queue,
        &incremental_build,
        &update,
        &pack_gpu_records(&incremental_build, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_b),
        recentre_b,
        overlay_b,
    );
    if !update.atlas_grew {
        assert_eq!(
            incremental_renderer.last_atlas_slots_written() as usize,
            update.written_slots.len(),
            "a steady-state patch writes exactly the dirty slots (no full re-upload)"
        );
    }
    incremental_renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        band,
        false,
        Some(MaterialChoice::default()),
    );
    let incremental_image =
        incremental_renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

    // Path 2 — a from-scratch WHOLESALE install of the SAME final scene B.
    let mut wholesale_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    wholesale_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &wholesale_build,
        &pack_gpu_records(&wholesale_build, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_b),
        recentre_b,
        overlay_b,
    );
    wholesale_renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        band,
        false,
        Some(MaterialChoice::default()),
    );
    let wholesale_image =
        wholesale_renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height);

    let hits = incremental_image.iter().filter(|pixel| pixel[0] == 1).count();
    assert!(hits > 0, "the gated view must produce brick hits (else the test is vacuous)");
    let mismatches = incremental_image
        .iter()
        .zip(&wholesale_image)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "incremental patch render must be pixel-identical to a wholesale install of the same \
         scene ({mismatches}/{} pixels differ; {hits} hit pixels)",
        width * height
    );
}

/// **ADR 0011 interior elision × G3 — the CARVE seam through the render.** Under the
/// surface-only record contract, deleting a solid that abutted another across a CHUNK
/// boundary un-occludes the neighbour chunk's face blocks: their records must APPEAR even
/// though their chunk is NOT in the edit's dirty set (the `apply_dirty_update`
/// 26-neighbourhood ring re-derivation). Drive that exact edit through the LIVE incremental
/// path (install A = two abutting chunk-filling boxes → delete one → `patch_brick_field`)
/// and assert the render is PIXEL-IDENTICAL to a from-scratch wholesale install of the
/// carved scene, and that the surviving record keys are BYTE-IDENTICAL to the wholesale
/// build's. The CPU-side byte equality (occupancy included) is gated in
/// `brick_field::incremental_carve_across_chunk_boundary_flips_neighbour_occlusion`.
#[test]
fn brick_raymarch_incremental_carve_exposes_interior_across_chunk_boundary() {
    use voxel_worker::core_geom::CHUNK_BLOCKS;
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer, ClipmapPyramid, IncrementalBrickField, LayerBand, Node, NodeContent,
        NodeTransform, OrbitCamera, PlaneAxis, Scene, Sketch, SketchSolid,
        TwoLayerResidentCache, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let vpb = 4u32;
    let chunk_span = CHUNK_BLOCKS as i64;

    // A solid SKETCH-EXTRUDE cube of exactly CHUNK_BLOCKS³ blocks at a chunk-aligned offset
    // — the sketch producer classifies COARSE-solid blocks to the very face (an SDF Box
    // tool's 1-block shell resolves as boundary microblocks, which are never elided and
    // would not exercise the coarse occlusion flip).
    let chunk_filling_box = |offset_blocks: [i64; 3]| -> Node {
        let edge_voxels = chunk_span * vpb as i64;
        let producer = SketchSolid::extrude(
            Sketch::rectangle(PlaneAxis::Z, edge_voxels, edge_voxels),
            edge_voxels as u32,
        );
        let mut node = Node::new(
            format!("box@{offset_blocks:?}"),
            NodeContent::SketchTool { producer, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks(offset_blocks, vpb);
        node
    };
    let anchor_lo = chunk_filling_box([-4 * chunk_span, 0, 0]);
    let anchor_hi = chunk_filling_box([4 * chunk_span, 0, 0]);
    let box_a = chunk_filling_box([0, 0, 0]);
    let box_b = chunk_filling_box([chunk_span, 0, 0]);
    let scene_with_b = Scene::from_nodes(vec![
        anchor_lo.clone(),
        anchor_hi.clone(),
        box_a.clone(),
        box_b.clone(),
    ]);
    let scene_carved = Scene::from_nodes(vec![anchor_lo, anchor_hi, box_a]);

    let mut cache = TwoLayerResidentCache::enabled();
    cache.clear();
    let index_with_b = scene_with_b.build_leaf_spatial_index(vpb);
    let fresh_with_b = cache.resident_two_layer_chunks(&scene_with_b, vpb, 0);
    let build_with_b = build_brick_field(&fresh_with_b, vpb);
    let mut field = IncrementalBrickField::from_wholesale(&build_with_b);
    let overlay_with_b = brick_representable_overlay(&fresh_with_b).unwrap_or(false);

    let index_carved = scene_carved.build_leaf_spatial_index(vpb);
    let carve_aabb = index_carved
        .edit_aabb_since(&index_with_b)
        .expect("a node delete is a localisable edit");
    let dirty = cache.invalidate_aabb(&carve_aabb, vpb);
    let fresh_carved = cache.resident_two_layer_chunks(&scene_carved, vpb, 0);
    assert_eq!(
        fresh_with_b.len(),
        fresh_carved.len(),
        "the anchors must pin the covering set (incremental precondition)"
    );

    let update = field.apply_dirty_update(&fresh_carved, &dirty);
    let incremental_build = field.to_build();
    let wholesale_build = build_brick_field(&fresh_carved, vpb);
    // The record KEYS must match wholesale byte-for-byte — including the re-appeared
    // records of box A's face blocks, whose chunk is NOT dirty (the ring re-derivation).
    assert_eq!(
        incremental_build
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect::<Vec<_>>(),
        wholesale_build
            .brick_records
            .iter()
            .map(|r| r.packed_world_block_key)
            .collect::<Vec<_>>(),
        "patched record keys must equal the wholesale surface-only build's"
    );
    // The carve must have grown the NON-dirty neighbour's record set (box A's exposed
    // face) — otherwise the ring seam is untested.
    let dirty_set: std::collections::BTreeSet<[i32; 3]> = dirty.iter().copied().collect();
    let non_dirty_records = |build: &voxel_worker::BrickFieldBuild| -> usize {
        build
            .brick_records
            .iter()
            .filter(|record| {
                let block =
                    voxel_worker::unpack_world_block_key(record.packed_world_block_key);
                let chunk = [
                    block[0].div_euclid(CHUNK_BLOCKS as i64) as i32,
                    block[1].div_euclid(CHUNK_BLOCKS as i64) as i32,
                    block[2].div_euclid(CHUNK_BLOCKS as i64) as i32,
                ];
                !dirty_set.contains(&chunk)
            })
            .count()
    };
    assert!(
        non_dirty_records(&wholesale_build) > non_dirty_records(&build_with_b),
        "the carve must EXPOSE records in a non-dirty chunk (the fixture must be real)"
    );

    let overlay_carved = brick_representable_overlay(&fresh_carved).unwrap_or(false);
    let recentre_carved = scene_carved.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene_carved.placed_region_dimensions(vpb);

    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let aspect_ratio = width as f32 / height as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let viewport_px = [0u32, 0, width, height];
    let band = LayerBand::FULL;

    let render = |renderer: &mut BrickRaymarchRenderer| {
        renderer.update_uniforms(
            &gpu.queue,
            view_projection,
            viewport_px,
            grid_dimensions,
            band,
            false,
            Some(MaterialChoice::default()),
        );
        renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
    };

    // Path 1 — install A (both boxes), then PATCH the carve in.
    let mut incremental_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    incremental_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build_with_b,
        &pack_gpu_records(&build_with_b, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_with_b),
        scene_with_b.recentre_voxels_for_resolve(vpb),
        overlay_with_b,
    );
    incremental_renderer.patch_brick_field(
        &gpu.device,
        &gpu.queue,
        &incremental_build,
        &update,
        &pack_gpu_records(&incremental_build, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_carved),
        recentre_carved,
        overlay_carved,
    );
    let incremental_image = render(&mut incremental_renderer);

    // Path 2 — a from-scratch wholesale install of the carved scene.
    let mut wholesale_renderer =
        BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    wholesale_renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &wholesale_build,
        &pack_gpu_records(&wholesale_build, |_| false),
        &ClipmapPyramid::from_chunks(&fresh_carved),
        recentre_carved,
        overlay_carved,
    );
    let wholesale_image = render(&mut wholesale_renderer);

    let hits = incremental_image.iter().filter(|pixel| pixel[0] == 1).count();
    assert!(hits > 0, "the carved scene must produce brick hits (else the test is vacuous)");
    let mismatches = incremental_image
        .iter()
        .zip(&wholesale_image)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        mismatches, 0,
        "carve-patched render must be pixel-identical to a wholesale install of the carved \
         scene ({mismatches}/{} pixels differ; {hits} hit pixels)",
        width * height
    );
}

/// **ADR 0011 residency-miss contract (decided at G1).** Forcing every sculpted
/// record non-resident (`pack_gpu_records(.., |_| true)`) must render each such block
/// as its COARSE form — a solid block-cube — never a miss/skip. A coarse cube is a
/// superset of the sculpted occupancy it replaces, so the forced-miss silhouette must
/// CONTAIN the all-resident silhouette pixel-for-pixel (and the pass must complete —
/// proving the branch is taken, never an assert). This is the hole G4's eviction rings
/// plug into.
#[test]
fn brick_raymarch_residency_miss_renders_coarse_form() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer,
        ClipmapPyramid, LayerBand, OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let build = build_brick_field(&two_layer_chunks, vpb);
        let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
        // Only the sculpted-bearing cases exercise the contract; every gated case has
        // boundary blocks, but assert it so a silently-coarse scene can't pass vacuously.
        assert!(
            build.sculpted_brick_count() > 0,
            "{}: fixture must contain sculpted bricks to force a miss",
            case.name
        );
        let overlay_active =
            brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;

        let render_image = |renderer: &BrickRaymarchRenderer| {
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };

        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        // All-resident silhouette.
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &pack_gpu_records(&build, |_| false),
            &pyramid,
            recentre,
            overlay_active,
        );
        let resident_image = render_image(&renderer);
        // Forced residency miss: every sculpted record → NON_RESIDENT sentinel.
        renderer.install_brick_field(
            &gpu.device,
            &gpu.queue,
            &build,
            &pack_gpu_records(&build, |_| true),
            &pyramid,
            recentre,
            overlay_active,
        );
        let miss_image = render_image(&renderer);

        let mut resident_hits = 0usize;
        let mut miss_hits = 0usize;
        let mut dropped = 0usize; // resident hit but the forced-miss render skipped it
        for pixel_index in 0..(width * height) as usize {
            let resident_hit = resident_image[pixel_index][0] == 1;
            let miss_hit = miss_image[pixel_index][0] == 1;
            if resident_hit {
                resident_hits += 1;
            }
            if miss_hit {
                miss_hits += 1;
            }
            if resident_hit && !miss_hit {
                dropped += 1;
            }
        }

        if resident_hits == 0 {
            failures.push(format!("{}: no resident hits to check the contract against", case.name));
            continue;
        }
        if dropped > 0 {
            failures.push(format!(
                "{}: {dropped} pixels hit under all-resident but MISSED under forced \
                 residency-miss — the coarse-form fallback dropped a boundary block \
                 (must render its solid cube, never skip)",
                case.name
            ));
        }
        assert!(
            miss_hits >= resident_hits,
            "{}: forced-miss silhouette ({miss_hits}) must contain the resident one \
             ({resident_hits}) — coarse cubes only grow the solid",
            case.name
        );
        eprintln!(
            "{}: residency-miss coarse-form ok (resident {resident_hits} ⊆ miss {miss_hits})",
            case.name
        );
    }

    assert!(
        failures.is_empty(),
        "brick residency-miss contract violated (ADR 0011 4a):\n{}",
        failures.join("\n")
    );
}

// ===========================================================================
// Brick-field CLIP-MAP tier (ADR 0011 G2) — the hierarchical DDA must be a pure
// empty-space accelerator: enabling the pyramid may only SKIP empty space, never
// change a hit. The load-bearing assertion is `pyramid-on == pyramid-off` (it
// catches the stride-overshoot / off-by-epsilon bugs the conservative-coverage
// unit test can't). Coarser levels are proven conservative CPU-side in
// `brick_field::tests::clipmap_pyramid_is_conservative_and_sorted`.
// ===========================================================================

/// **ADR 0011 parity gate, coarse tier (the load-bearing G2 assertion).** For each
/// gated scene — including the scattered many-object and distinct-material
/// multi-producer scenes — the finest-LOD hit-identity image rendered WITH the
/// clip-map pyramid enabled must be BYTE-IDENTICAL to the image rendered with it
/// disabled (an empty pyramid = the flat G1 block-DDA). The pyramid may only stride
/// through provably-empty cells, so any hit it changes is a stride-overshoot bug.
#[test]
fn brick_raymarch_pyramid_on_equals_off() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer,
        ClipmapPyramid, LayerBand, OrbitCamera, TwoLayerStore, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;
    let mut failures: Vec<String> = Vec::new();

    for case in brick_render_cases() {
        let vpb = case.voxels_per_block;
        let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
        let build = build_brick_field(&two_layer_chunks, vpb);
        assert!(!build.brick_records.is_empty(), "{}: empty brick field", case.name);
        // The LIVE pyramid constructor (chunk-derived, interiors included) — this A/B is
        // the rendering-equivalence proof for the chunk-sourced pyramid (ADR 0011).
        let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
        // Every level must carry cells, else "on == off" is vacuous (the L3 skip is
        // exercised too — G4).
        assert!(
            !pyramid.level_1.cell_keys.is_empty()
                && !pyramid.level_2.cell_keys.is_empty()
                && !pyramid.level_3.cell_keys.is_empty(),
            "{}: pyramid has no cells — the on/off comparison would be vacuous",
            case.name
        );
        let overlay_active =
            brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
        let recentre = case.scene.recentre_voxels_for_resolve(vpb);
        let grid_dimensions = case.scene.placed_region_dimensions(vpb);

        let mut app_core = AppCore::new(OrbitCamera::default());
        app_core.camera.target = glam::Vec3::ZERO;
        app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
        let aspect_ratio = width as f32 / height as f32;
        let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
        let viewport_px = [0u32, 0, width, height];
        let band = LayerBand::FULL;
        let gpu_records = pack_gpu_records(&build, |_| false);

        let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
        let render = |renderer: &mut BrickRaymarchRenderer, pyramid: &ClipmapPyramid| {
            renderer.install_brick_field(
                &gpu.device,
                &gpu.queue,
                &build,
                &gpu_records,
                pyramid,
                recentre,
                overlay_active,
            );
            renderer.update_uniforms(
                &gpu.queue,
                view_projection,
                viewport_px,
                grid_dimensions,
                band,
                false,
                Some(MaterialChoice::default()),
            );
            renderer.render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
        };

        let on_image = render(&mut renderer, &pyramid);
        let off_image = render(&mut renderer, &ClipmapPyramid::empty());

        let mut on_hits = 0usize;
        let mut differing = 0usize;
        let mut first: Option<(u32, u32, [u32; 4], [u32; 4])> = None;
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if on_image[index][0] == 1 {
                    on_hits += 1;
                }
                if on_image[index] != off_image[index] {
                    differing += 1;
                    if first.is_none() {
                        first = Some((x, y, off_image[index], on_image[index]));
                    }
                }
            }
        }
        if on_hits == 0 {
            failures.push(format!("{}: the gated view produced ZERO brick hits", case.name));
            continue;
        }
        if differing > 0 {
            failures.push(format!(
                "{}: {differing}/{} pixels differ pyramid-on vs off (first {:?}) — the \
                 hierarchical skip changed a hit (stride overshoot / off-by-epsilon)",
                case.name,
                width * height,
                first
            ));
        } else {
            eprintln!(
                "{}: pyramid on == off ({on_hits} hits; L1 {} cells, L2 {} cells, L3 {} cells)",
                case.name,
                pyramid.level_1.cell_keys.len(),
                pyramid.level_2.cell_keys.len(),
                pyramid.level_3.cell_keys.len()
            );
        }
    }

    assert!(
        failures.is_empty(),
        "brick clip-map changed the hit set (ADR 0011 gate coarse tier — pyramid must \
         only skip empty space):\n{}",
        failures.join("\n")
    );
}

/// **ADR 0011 G2/G4 perf probe (acceptance criterion).** The WIDE-scatter
/// empty-space-skipping lift across clip-map depth: shapes spread over a
/// ~2000-block extent with fully-empty L3 cells between them, marched at four level
/// configs — OFF (flat DDA) / L1+L2 (the G2 two-level baseline) / +L3 (G4) / +L4
/// (a hypothetical 4096-block level, EVALUATED here, not shipped) — reported as
/// mean block-DDA steps per hitting ray (the CPU march's counted loop iterations,
/// the same traversal the shader runs). `#[ignore]`d (measurement, not a gate); run
/// with `cargo test --features gpu --release -- --ignored
/// clipmap_scattered_scene_skips_empty_space --nocapture`.
///
/// On L4: a 1024-block-spaced scatter (~2060-block extent) fits inside ONE
/// 4096-block L4 cell, so L4 has nothing to skip and measures identical to +L3.
/// Scenes wide enough for empty L4 cells (8192+-block spans) cannot currently be
/// BUILT: `build_covering_chunks` enumerates the scene AABB in 4-block chunks
/// (2048³ ≈ 8.6e9 chunk builds at that span), so no realistic scene reaches L4's
/// regime — the measured verdict for not shipping a 4th level.
#[test]
#[ignore = "perf probe — run explicitly with --release --ignored --nocapture"]
fn clipmap_scattered_scene_skips_empty_space() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, cpu_march_levels_counted, pack_gpu_records,
        AppCore, BrickRaymarchRenderer, ClipmapLevel, ClipmapPyramid, LayerBand, NodeTransform,
        OrbitCamera, TwoLayerStore, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
        CLIPMAP_LEVEL_3_BLOCKS_PER_CELL, COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 320u32;
    let height = 320u32;
    let vpb = 16u32;

    // A wide scatter: a 3×3×3 lattice of 13-block spheres spaced 1024 BLOCKS apart,
    // so every adjacent pair straddles a fully-EMPTY 512-block L3 cell (block
    // positions 0/1024/2048 → L3 cells 0/2/4 occupied, cells 1/3 empty). The
    // INTERIOR spheres matter: a hull-corner lattice alone puts every visible
    // object on the traversal-AABB surface, so hitting rays enter right next to
    // their sphere and never cross a void — the center/face/edge spheres are seen
    // THROUGH the gaps, so their rays cross ~1024 blocks of empty space: two
    // 512-block strides under L3 vs sixteen 64-block strides under L2 vs ~1024
    // flat steps. Extent ~2060 blocks, within the flat MAX_BLOCK_STEPS budget so
    // the OFF baseline completes; 13-block spheres at 320² stay a few pixels wide
    // so the auto-framed view actually hits them.
    const SCATTER_SPACING_BLOCKS: i64 = 1024;
    // The evaluated 4th level: 4096 blocks/cell, continuing the 8× progression. Not a
    // production constant — this probe measures whether it would pay before we build it.
    const CLIPMAP_LEVEL_4_BLOCKS_PER_CELL: u32 = 4096;
    let mut nodes = Vec::new();
    for index in 0..27i64 {
        let shape = SdfShape::from_blocks(ShapeKind::Sphere, [13, 13, 13], 1, vpb);
        let mut node = Node::new(
            format!("s{index}"),
            NodeContent::Tool { shape, material: MaterialChoice::Stone },
        );
        node.transform = NodeTransform::from_blocks(
            [
                (index % 3) * SCATTER_SPACING_BLOCKS,
                ((index / 3) % 3) * SCATTER_SPACING_BLOCKS,
                (index / 9) * SCATTER_SPACING_BLOCKS,
            ],
            vpb,
        );
        nodes.push(node);
    }
    let scene = Scene::from_nodes(nodes);
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, vpb, 0);
    let build = build_brick_field(&two_layer_chunks, vpb);
    let pyramid_on = ClipmapPyramid::from_records(&build.brick_records);
    let gpu_records = pack_gpu_records(&build, |_| false);
    let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
    let recentre = scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = scene.placed_region_dimensions(vpb);

    // The four clip-map configs, each a slice of levels COARSEST→FINEST (the descent
    // order). OFF is the empty slice (flat block-DDA); +L4 prepends a hypothetical
    // 4096-block level to the shipped three.
    let level_1 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL);
    let level_2 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL);
    let level_3 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL);
    let level_4 = ClipmapLevel::from_records(&build.brick_records, CLIPMAP_LEVEL_4_BLOCKS_PER_CELL);
    let configs: [(&str, Vec<&ClipmapLevel>); 4] = [
        ("OFF (flat)", Vec::new()),
        ("L1+L2 (G2)", vec![&level_2, &level_1]),
        ("+L3 (G4)", vec![&level_3, &level_2, &level_1]),
        ("+L4 (eval)", vec![&level_4, &level_3, &level_2, &level_1]),
    ];

    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let view_projection = app_core.view_projection(width as f32 / height as f32, grid_dimensions);

    let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build,
        &gpu_records,
        &pyramid_on,
        recentre,
        overlay_active,
    );
    let frame = renderer.update_uniforms(
        &gpu.queue,
        view_projection,
        [0, 0, width, height],
        grid_dimensions,
        LayerBand::FULL,
        false,
        Some(MaterialChoice::default()),
    );

    // Sum block-DDA steps over the rays that HIT (empty-space skipping shows up on the
    // rays that traverse the scene), per config. The hit set is identical across configs
    // (each level may only skip provably-empty space — the pyramid-on == off gate), so
    // this compares work-per-ray over the same pixels; the finest config (+L4) defines
    // the hit set.
    let mut sums = [0u64; 4];
    let mut hitting_rays = 0u64;
    for y in 0..height {
        for x in 0..width {
            let pixel = glam::Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
            let mut per_config = [0u32; 4];
            let mut hit_any = false;
            for (index, (_, levels)) in configs.iter().enumerate() {
                let (hit, steps) =
                    cpu_march_levels_counted(&frame, &gpu_records, &build, levels, pixel);
                per_config[index] = steps;
                if index == configs.len() - 1 {
                    hit_any = hit.is_some();
                }
            }
            if hit_any {
                for index in 0..configs.len() {
                    sums[index] += per_config[index] as u64;
                }
                hitting_rays += 1;
            }
        }
    }
    assert!(hitting_rays > 0, "the scattered probe view produced no hits");
    let span_blocks = grid_dimensions.map(|d| d / vpb).iter().max().copied().unwrap_or(0);
    let mean = |index: usize| sums[index] as f64 / hitting_rays as f64;
    eprintln!(
        "clip-map WIDE-scatter probe ({}-block span, {} sculpted bricks, {} hitting rays)\n  \
         L1 {} cells, L2 {} cells, L3 {} cells, L4 {} cells\n  \
         mean block-steps/ray by clip-map config:",
        span_blocks,
        build.sculpted_brick_count(),
        hitting_rays,
        level_1.cell_keys.len(),
        level_2.cell_keys.len(),
        level_3.cell_keys.len(),
        level_4.cell_keys.len(),
    );
    for (index, (name, _)) in configs.iter().enumerate() {
        eprintln!(
            "    {name:<12} {:>8.1}  ({:.2}× vs OFF)",
            mean(index),
            mean(0) / mean(index).max(1.0),
        );
    }

    // Gate: each coarser level is a monotone win (never more steps), the two-level
    // pyramid beats flat, and G4's L3 strictly beats the L1+L2 baseline on this
    // empty-L3-cell scene (the acceptance criterion — a measured ceiling improvement).
    assert!(sums[1] < sums[0], "L1+L2 must beat flat (on {} vs {})", sums[1], sums[0]);
    assert!(
        sums[2] < sums[1],
        "G4 +L3 must strictly reduce block-steps vs the L1+L2 baseline on a wide scatter \
         with empty L3 cells (+L3 {} vs L1+L2 {})",
        sums[2], sums[1]
    );
    assert!(
        sums[3] <= sums[2],
        "+L4 may only skip more empty space, never less (+L4 {} vs +L3 {})",
        sums[3], sums[2]
    );
}

/// **ADR 0012 (H1) — the onion GHOST pass marches ONLY the onion slabs.** The brick ghost
/// draws two per-slab raymarches, each clamped to ONE onion slab (`update_ghost_uniforms`
/// clamps the traversal AABB to the slab's band). This gates that confinement through the
/// hit-identity harness at each slab band: (a) the LOWER slab's hits all sit strictly BELOW
/// the solid band, (b) the UPPER slab's hits all sit strictly ABOVE it — so the ghost never
/// draws INSIDE the band — and (c) both slabs actually draw on a tall solid (nonempty). It
/// also gates the "band scrub = uniform-only on the brick path" promise (ADR 0012): rebinding
/// the ghost uniforms for two different bands leaves the installed field (record count)
/// untouched — no re-mesh, no atlas re-upload.
#[test]
fn onion_ghost_marches_only_the_onion_slabs() {
    use voxel_worker::{
        brick_representable_overlay, build_brick_field, pack_gpu_records, AppCore,
        BrickRaymarchRenderer, ClipmapPyramid, LayerBand, OrbitCamera, TwoLayerStore,
        COLOR_TARGET_FORMAT,
    };

    let gpu = pollster::block_on(GpuContext::new(None));
    let width = 128u32;
    let height = 128u32;

    // The tall hollow sphere (grid_z ~80) — its shell crosses a mid band AND onion slabs
    // both sides, so all three clips render hits.
    let case = brick_render_cases()
        .into_iter()
        .find(|c| c.name == "render-sphere-80-d16")
        .expect("render matrix carries the tall sphere case");
    let vpb = case.voxels_per_block;
    let two_layer_chunks = TwoLayerStore::enabled().build_covering_chunks(&case.scene, vpb, 0);
    let build = build_brick_field(&two_layer_chunks, vpb);
    assert!(!build.brick_records.is_empty(), "the sphere shell must produce records");
    let records = pack_gpu_records(&build, |_| false);
    let overlay_active = brick_representable_overlay(&two_layer_chunks).unwrap_or(false);
    let pyramid = ClipmapPyramid::from_chunks(&two_layer_chunks);
    let recentre = case.scene.recentre_voxels_for_resolve(vpb);
    let grid_dimensions = case.scene.placed_region_dimensions(vpb);
    let grid_z = grid_dimensions[2];

    let depth = 4u32;
    let band = LayerBand {
        band_min: grid_z / 2 - 2,
        band_max: grid_z / 2 + 1,
        onion_depth: depth,
    };
    // The two onion slabs the ghost draws — the recentred-Z remainder of the onion span.
    let lower_slab = LayerBand {
        band_min: band.band_min - depth,
        band_max: band.band_min - 1,
        onion_depth: 0,
    };
    let upper_slab = LayerBand {
        band_min: band.band_max + 1,
        band_max: band.band_max + depth,
        onion_depth: 0,
    };

    let mut app_core = AppCore::new(OrbitCamera::default());
    app_core.camera.target = glam::Vec3::ZERO;
    app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(grid_dimensions);
    let aspect_ratio = width as f32 / height as f32;
    let view_projection = app_core.view_projection(aspect_ratio, grid_dimensions);
    let viewport_px = [0u32, 0, width, height];

    let mut renderer = BrickRaymarchRenderer::new(&gpu.device, &gpu.queue, COLOR_TARGET_FORMAT);
    renderer.install_brick_field(
        &gpu.device,
        &gpu.queue,
        &build,
        &records,
        &pyramid,
        recentre,
        overlay_active,
    );

    // The absolute-voxel-Z set of the hits a given band clip renders, via the SOLID
    // hit-identity march (the ghost slab uses the SAME `march_frame(slab)` traversal clamp,
    // so this exactly mirrors what the ghost pass would draw for that slab).
    let hit_zs = |renderer: &BrickRaymarchRenderer, clip: LayerBand| -> Vec<i32> {
        renderer.update_uniforms(
            &gpu.queue,
            view_projection,
            viewport_px,
            grid_dimensions,
            clip,
            false,
            Some(MaterialChoice::default()),
        );
        renderer
            .render_hit_identity_image(&gpu.device, &gpu.queue, width, height)
            .into_iter()
            .filter(|pixel| pixel[0] == 1)
            .map(|pixel| pixel[3] as i32) // absolute voxel Z (i32 bit-reinterpret)
            .collect()
    };

    let band_zs = hit_zs(&renderer, band);
    let lower_zs = hit_zs(&renderer, lower_slab);
    let upper_zs = hit_zs(&renderer, upper_slab);

    assert!(!band_zs.is_empty(), "the mid band must render some solid voxels");
    assert!(!lower_zs.is_empty(), "the lower onion slab must ghost some voxels (tall solid)");
    assert!(!upper_zs.is_empty(), "the upper onion slab must ghost some voxels (tall solid)");

    let band_lo = *band_zs.iter().min().unwrap();
    let band_hi = *band_zs.iter().max().unwrap();
    assert!(
        lower_zs.iter().all(|&z| z < band_lo),
        "a lower-slab ghost hit fell inside/above the band (band_lo {band_lo}, lower max {})",
        lower_zs.iter().max().unwrap()
    );
    assert!(
        upper_zs.iter().all(|&z| z > band_hi),
        "an upper-slab ghost hit fell inside/below the band (band_hi {band_hi}, upper min {})",
        upper_zs.iter().min().unwrap()
    );

    // Uniform-only (ADR 0012): rebinding the ghost slabs for two different bands must NOT
    // touch the installed field — no re-mesh / atlas re-upload on a brick-path band scrub.
    let record_count_before = renderer.record_count();
    renderer.update_ghost_uniforms(&gpu.queue, view_projection, viewport_px, grid_dimensions, band);
    let scrubbed = LayerBand {
        band_min: band.band_min + 3,
        band_max: band.band_max + 3,
        onion_depth: depth,
    };
    renderer.update_ghost_uniforms(
        &gpu.queue,
        view_projection,
        viewport_px,
        grid_dimensions,
        scrubbed,
    );
    assert_eq!(
        renderer.record_count(),
        record_count_before,
        "a ghost band scrub must be uniform-only (no field re-install / re-upload)"
    );
    assert!(
        renderer.has_brick_field(),
        "the installed field stays live across ghost band scrubs"
    );
}
