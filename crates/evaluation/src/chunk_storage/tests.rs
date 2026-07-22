use super::*;
use voxel_core::core_geom::MaterialChoice;
use document::scene::{DefId, Node, NodeContent, VoxelBody, Scene};
use voxel_core::voxel::{ShapeKind, Voxel, VoxelGrid};
use document::voxel::{GeometryParams, SdfShape, VoxelProducer};

/// The binary on-disk byte size of a whole [`CompressedChunk`] (header + palette +
/// occupancy), used by the ratio report.
fn compressed_binary_size(compressed: &CompressedChunk) -> usize {
    // Header: dimensions 3×u32, min_corner 3×i64, centre_fraction 3×f32, box_spans
    // 3×u32 = 12 + 24 + 12 + 12 = 60 bytes; palette: 2 bytes/entry.
    60 + compressed.material_palette.len() * 2 + occupancy_binary_size(&compressed.occupancy)
}

/// A pseudo-random generator (the same Numerical-Recipes LCG `cuboid.rs` uses),
/// so the fuzz tests are deterministic without pulling in a `rand` dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Canonicalise a grid's occupied set into a sorted multiset of
/// `(local_index, block_local_coord, block_id)`. The index is now stored EXACTLY
/// (ADR 0003 §3a), so the round-trip assertion is byte-for-byte on the integer index
/// (no f32 ULP comparison needed); `block_local_coord` + the categorical `block_id`
/// keep the intra-block coordinate and the block in the losslessness guarantee.
/// Order-independent (the resolve path treats the occupied vec as a set).
fn occupied_multiset(
    grid: &VoxelGrid,
) -> std::collections::BTreeMap<([i32; 3], [u8; 3], u16), usize> {
    let mut multiset = std::collections::BTreeMap::new();
    for voxel in &grid.occupied {
        *multiset
            .entry((voxel.local_index, voxel.block_local_coord, voxel.block_id.0))
            .or_insert(0) += 1;
    }
    multiset
}

/// Assert `decompress(compress(grid))` equals `grid` in dimensions and occupied
/// set (position + block-local coord + material, byte-exact). Returns the
/// `CompressedChunk` so callers can make follow-up assertions (palette, ratio).
fn assert_lossless_round_trip(grid: &VoxelGrid, label: &str) -> CompressedChunk {
    let compressed = compress(grid);
    let restored = decompress(&compressed);
    assert_eq!(
        restored.dimensions, grid.dimensions,
        "[{label}] dimensions must round-trip"
    );
    assert_eq!(
        restored.occupied_count(),
        grid.occupied_count(),
        "[{label}] occupied count must round-trip"
    );
    assert_eq!(
        occupied_multiset(&restored),
        occupied_multiset(grid),
        "[{label}] occupied set (position + block-local + material) must be byte-identical"
    );
    // The compressed view's own occupied_count must agree with the grid's.
    assert_eq!(
        compressed.occupied_count(),
        grid.occupied_count(),
        "[{label}] CompressedChunk::occupied_count must match the source grid"
    );
    compressed
}

fn shape_grid(kind: ShapeKind, size: [u32; 3], voxels_per_block: u32) -> VoxelGrid {
    let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
    let mut grid = VoxelGrid::new(shape.grid_dimensions(voxels_per_block));
    shape.resolve(&mut grid, voxels_per_block);
    grid
}

#[test]
fn round_trip_empty_chunk() {
    let grid = VoxelGrid::new([64, 64, 64]);
    let compressed = assert_lossless_round_trip(&grid, "empty");
    assert!(
        compressed.material_palette.is_empty(),
        "an empty chunk has an empty palette"
    );
    assert_eq!(compressed.occupied_count(), 0);
}

#[test]
fn round_trip_full_single_material_chunk() {
    // A fully-occupied 8×8×8 box, single material — the dense win case.
    let dimensions = [8u32, 8, 8];
    let mut grid = VoxelGrid::new(dimensions);
    let half = [4.0f32; 3];
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                grid.occupied.push(Voxel {
                    local_index: [
                        (x as f32 + 0.5 - half[0]).floor() as i32,
                        (y as f32 + 0.5 - half[1]).floor() as i32,
                        (z as f32 + 0.5 - half[2]).floor() as i32,
                    ],
                    block_local_coord: [(x % 4) as u8, (y % 4) as u8, (z % 4) as u8],
                    block_id: voxel_core::core_geom::BlockId(7),
                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                    grid_overlay: false,
                });
            }
        }
    }
    let compressed = assert_lossless_round_trip(&grid, "full-single-material");
    assert_eq!(
        compressed.material_palette,
        vec![7],
        "a single-material full chunk has a one-entry palette"
    );
    // A solid single-material box should pick the dense encoding (smaller).
    assert!(
        matches!(compressed.occupancy, Occupancy::Dense { .. }),
        "a fully-occupied single-material chunk should compress dense, got sparse"
    );
}

#[test]
fn round_trip_multi_material_chunk() {
    // 4×4×2 quartered into four materials — distinct ids, no duplicates.
    let dimensions = [4u32, 4, 2];
    let mut grid = VoxelGrid::new(dimensions);
    let half = [2.0f32, 2.0, 1.0];
    for z in 0..2 {
        for y in 0..4 {
            for x in 0..4 {
                let material = match (x < 2, y < 2) {
                    (true, true) => 11,
                    (false, true) => 22,
                    (true, false) => 33,
                    (false, false) => 44,
                };
                grid.occupied.push(Voxel {
                    local_index: [
                        (x as f32 + 0.5 - half[0]).floor() as i32,
                        (y as f32 + 0.5 - half[1]).floor() as i32,
                        (z as f32 + 0.5 - half[2]).floor() as i32,
                    ],
                    block_local_coord: [x as u8, y as u8, z as u8],
                    block_id: voxel_core::core_geom::BlockId(material),
                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                    grid_overlay: false,
                });
            }
        }
    }
    let compressed = assert_lossless_round_trip(&grid, "multi-material");
    let mut palette_sorted = compressed.material_palette.clone();
    palette_sorted.sort_unstable();
    assert_eq!(
        palette_sorted,
        vec![11, 22, 33, 44],
        "the palette must be exactly the distinct materials, no duplicates"
    );
    // No duplicate ids in the palette.
    let unique: std::collections::HashSet<u16> =
        compressed.material_palette.iter().copied().collect();
    assert_eq!(
        unique.len(),
        compressed.material_palette.len(),
        "palette must contain no duplicate materials"
    );
}

#[test]
fn round_trip_real_resolved_chunks_across_shapes() {
    // Real resolved chunks via Scene::resolve_chunk across every SDF primitive.
    let voxels_per_block = 16u32;
    for kind in [
        ShapeKind::Sphere,
        ShapeKind::Cylinder,
        ShapeKind::Tube,
        ShapeKind::Torus,
        ShapeKind::Box,
    ] {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [5 * voxels_per_block, 5 * voxels_per_block, 5 * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("a placed shape has a covering chunk range");
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = scene.resolve_chunk(
                        [chunk_x, chunk_y, chunk_z],
                        voxels_per_block,
                        0,
                    );
                    assert_lossless_round_trip(
                        &chunk,
                        &format!("{kind:?} chunk {chunk_x},{chunk_y},{chunk_z}"),
                    );
                }
            }
        }
    }
}

#[test]
fn round_trip_demo_scene_and_village_chunks() {
    let voxels_per_block = 16u32;

    // --demo-scene: three differently-materialled tools.
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let demo_scene = Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
        make_tool(ShapeKind::Torus, [0, 0, 6], MaterialChoice::Plain),
    ]);

    // --demo-village: an instanced house assembly.
    let house_def_id = DefId(1);
    let tool = |kind, size: [u32; 3], offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, size, 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let instance = |name: &str, offset: [i64; 3]| {
        let mut node = Node::new(name, NodeContent::Instance(house_def_id));
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let mut village = Scene::from_nodes(vec![
        instance("House 1", [0, 0, 0]),
        instance("House 2", [6, 0, 0]),
        instance("House 3", [12, 0, 0]),
        instance("House 4", [18, 0, 0]),
    ]);
    village.add_definition(
        house_def_id,
        "House".to_string(),
        vec![
            tool(ShapeKind::Box, [2, 2, 2], [0, 0, 0], MaterialChoice::Stone),
            tool(ShapeKind::Cylinder, [1, 2, 1], [0, 2, 0], MaterialChoice::Wood),
        ],
    );

    for (scene, label) in [(demo_scene, "demo-scene"), (village, "demo-village")] {
        let (min_chunk, max_chunk) = scene
            .covering_chunk_range(voxels_per_block)
            .expect("a placed scene has a covering chunk range");
        for chunk_z in min_chunk[2]..=max_chunk[2] {
            for chunk_y in min_chunk[1]..=max_chunk[1] {
                for chunk_x in min_chunk[0]..=max_chunk[0] {
                    let chunk = scene.resolve_chunk(
                        [chunk_x, chunk_y, chunk_z],
                        voxels_per_block,
                        0,
                    );
                    assert_lossless_round_trip(
                        &chunk,
                        &format!("{label} chunk {chunk_x},{chunk_y},{chunk_z}"),
                    );
                }
            }
        }
    }
}

#[test]
fn round_trip_part_only_debug_clouds_grid() {
    // A VoxelBody producer (debug clouds) fills a grid with material_id 0 voxels at a
    // pseudo-random fill — a different occupancy profile than the SDF shells.
    let scene = Scene::single_node(Node::new(
        "Clouds",
        NodeContent::VoxelBody(VoxelBody::DebugClouds { seed: 1 }),
    ));
    // Resolve over an explicit region (a VoxelBody-only scene has no chunk range).
    let grid = scene.resolve_region(
        document::scene::RegionBlocks::new([4, 4, 4]),
        16,
        0,
    );
    if grid.occupied.is_empty() {
        return; // nothing to assert if the field produced no voxels.
    }
    assert_lossless_round_trip(&grid, "debug-clouds");
}

#[test]
fn round_trip_randomized_fuzz_varied_fill_and_materials() {
    // Pseudo-random multi-material fills over varied extents / fill % / material
    // counts — the real safety net for both encodings (the heuristic flips
    // between sparse and dense across this matrix).
    let mut lcg = Lcg(0xc0ff_ee00_d15e_a5e5);
    let extents = [[1u32, 1, 1], [6, 4, 5], [9, 2, 7], [3, 8, 4], [7, 7, 7]];
    for &extent in &extents {
        for materials in [1u32, 2, 5] {
            for fill_percent in [5u32, 30, 75, 100] {
                let half = [
                    extent[0] as f32 / 2.0,
                    extent[1] as f32 / 2.0,
                    extent[2] as f32 / 2.0,
                ];
                let mut grid = VoxelGrid::new(extent);
                for z in 0..extent[2] {
                    for y in 0..extent[1] {
                        for x in 0..extent[0] {
                            if (lcg.next_u32() % 100) < fill_percent {
                                let material = (lcg.next_u32() % materials) as u16;
                                grid.occupied.push(Voxel {
                                    local_index: [
                                        (x as f32 + 0.5 - half[0]).floor() as i32,
                                        (y as f32 + 0.5 - half[1]).floor() as i32,
                                        (z as f32 + 0.5 - half[2]).floor() as i32,
                                    ],
                                    block_local_coord: [
                                        (x % 4) as u8,
                                        (y % 4) as u8,
                                        (z % 4) as u8,
                                    ],
                                    block_id: voxel_core::core_geom::BlockId(material),
                                    attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                                    grid_overlay: false,
                                });
                            }
                        }
                    }
                }
                assert_lossless_round_trip(
                    &grid,
                    &format!("fuzz {extent:?} m={materials} fill={fill_percent}"),
                );
            }
        }
    }
}

#[test]
fn palette_has_no_duplicates_and_covers_every_material() {
    // A grid whose materials repeat heavily across cells; the palette must still
    // be the DISTINCT set with no duplicates, and every cell's material must map
    // back through it.
    let dimensions = [6u32, 6, 1];
    let mut grid = VoxelGrid::new(dimensions);
    // Voxel centres sit at integer-plus-half (a resolved-grid invariant), so use
    // `n + 0.5` directly — NOT `n + 0.5 - half`, which would land a centre on an
    // integer (e.g. 0.0) that is not a valid voxel centre.
    let materials = [100u16, 200, 100, 300, 200, 100];
    for y in 0..6 {
        for x in 0..6 {
            grid.occupied.push(Voxel {
                local_index: [x, y, 0],
                block_local_coord: [0, 0, 0],
                block_id: voxel_core::core_geom::BlockId(materials[x as usize]),
                attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                grid_overlay: false,
            });
        }
    }
    let compressed = compress(&grid);
    let unique: std::collections::HashSet<u16> =
        compressed.material_palette.iter().copied().collect();
    assert_eq!(unique.len(), compressed.material_palette.len(), "no dup palette entries");
    assert_eq!(unique, [100, 200, 300].into_iter().collect(), "distinct materials only");
    assert_eq!(occupied_multiset(&decompress(&compressed)), occupied_multiset(&grid));
}

#[test]
fn serde_round_trip_through_json_equals_original_grid() {
    // serialize → deserialize → decompress equals the original grid, proving the
    // CompressedChunk is serde-serialisable for the later disk store.
    let grid = shape_grid(ShapeKind::Sphere, [4, 4, 4], 8);
    assert!(!grid.occupied.is_empty(), "the sphere must resolve to voxels");
    let compressed = compress(&grid);

    let json = serde_json::to_string(&compressed).expect("CompressedChunk serialises");
    let restored: CompressedChunk =
        serde_json::from_str(&json).expect("CompressedChunk deserialises");
    assert_eq!(restored, compressed, "serde must round-trip the CompressedChunk exactly");

    let restored_grid = decompress(&restored);
    assert_eq!(
        occupied_multiset(&restored_grid),
        occupied_multiset(&grid),
        "serialize → deserialize → decompress must equal the original grid"
    );
}

/// Report measured compression ratios on representative real resolved chunks
/// (sphere / torus / village). Asserts a meaningful win on the mostly-empty SDF
/// shells and prints the numbers (run with `--nocapture` to read them).
#[test]
fn report_compression_ratios_on_real_chunks() {
    let voxels_per_block = 16u32;

    // Raw size of a VoxelGrid's occupied storage: each Voxel is
    // 3×f32 + 3×u8 + u16 = 12 + 3 + 2 = 17 bytes of payload (the Vec capacity is
    // ignored; this is the logical raw footprint of the occupied data).
    let raw_bytes = |grid: &VoxelGrid| -> usize { grid.occupied_count() * 17 };

    let report = |label: &str, grid: &VoxelGrid| {
        let compressed = compress(grid);
        // Compressed size via the same binary measure the heuristic uses.
        let compressed_bytes = compressed_binary_size(&compressed);
        let raw = raw_bytes(grid);
        let ratio = if compressed_bytes == 0 {
            0.0
        } else {
            raw as f64 / compressed_bytes as f64
        };
        let encoding = match compressed.occupancy {
            Occupancy::Sparse(_) => "sparse",
            Occupancy::Dense { .. } => "dense",
        };
        println!(
            "[ratio] {label}: {} voxels, raw {raw} B, compressed {compressed_bytes} B \
             ({encoding}) → {ratio:.2}× smaller",
            grid.occupied_count()
        );
        ratio
    };

    // A sphere chunk (mostly-empty shell — the sparse win case).
    let sphere = shape_grid(ShapeKind::Sphere, [5, 5, 5], voxels_per_block);
    let sphere_ratio = report("sphere 5³@16 (whole grid)", &sphere);

    // A torus chunk.
    let torus = shape_grid(ShapeKind::Torus, [5, 5, 5], voxels_per_block);
    report("torus 5³@16 (whole grid)", &torus);

    // A solid box (the dense win case).
    let solid_box = shape_grid(ShapeKind::Box, [4, 4, 4], voxels_per_block);
    report("box 4³@16 (whole grid, solid)", &solid_box);

    // Real PER-CHUNK resolved pieces of a sphere: a solid SDF sphere is a filled
    // ellipsoid, so its chunks are dense-favourable (the dense bit-packed encoding
    // wins) — the honest figure for the common SDF-solid case. Report the
    // aggregate ratio and the single best chunk.
    let per_chunk_report = |label: &str, kind: ShapeKind, size: [u32; 3]| -> f64 {
        let scene = Scene::from_geometry(
            GeometryParams {
                shape: kind,
                size_voxels: [size[0] * voxels_per_block, size[1] * voxels_per_block, size[2] * voxels_per_block],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let (lo, hi) = scene.covering_chunk_range(voxels_per_block).expect("placed");
        let mut total_raw = 0usize;
        let mut total_compressed = 0usize;
        let mut best_ratio = 0.0f64;
        let mut sparse_chunks = 0usize;
        let mut total_chunks = 0usize;
        for cz in lo[2]..=hi[2] {
            for cy in lo[1]..=hi[1] {
                for cx in lo[0]..=hi[0] {
                    let chunk = scene.resolve_chunk([cx, cy, cz], voxels_per_block, 0);
                    if chunk.occupied.is_empty() {
                        continue;
                    }
                    total_chunks += 1;
                    let compressed = compress(&chunk);
                    if matches!(compressed.occupancy, Occupancy::Sparse(_)) {
                        sparse_chunks += 1;
                    }
                    let raw = raw_bytes(&chunk);
                    let comp = compressed_binary_size(&compressed);
                    total_raw += raw;
                    total_compressed += comp;
                    best_ratio = best_ratio.max(raw as f64 / comp.max(1) as f64);
                }
            }
        }
        let ratio = total_raw as f64 / total_compressed.max(1) as f64;
        println!(
            "[ratio] {label}: {total_chunks} non-empty chunks ({sparse_chunks} sparse), \
             raw {total_raw} B, compressed {total_compressed} B → {ratio:.2}× aggregate, \
             best chunk {best_ratio:.2}×"
        );
        best_ratio
    };
    let sphere_best = per_chunk_report("sphere 5³@16 (per chunk)", ShapeKind::Sphere, [5, 5, 5]);
    per_chunk_report("torus 5³@16 (per chunk)", ShapeKind::Torus, [5, 5, 5]);

    // A genuinely sparse case (the sparse-encoding win): a very-low-fill grid.
    // The sparse vs dense crossover is ~`N < cells/48` (sparse 9 B/voxel vs dense
    // ~1 bit/cell), so a sub-1% fill over a big box lands firmly in sparse-land.
    let mut lcg = Lcg(0x5ade_5e00_1234_abcd_u64);
    let sparse_extent = [40u32, 40, 40];
    let sparse_half = [20.0f32, 20.0, 20.0];
    let mut sparse_grid = VoxelGrid::new(sparse_extent);
    for z in 0..40 {
        for y in 0..40 {
            for x in 0..40 {
                // ~0.5% fill → ~320 voxels over 64000 cells, well under cells/48.
                if lcg.next_u32() % 1000 < 5 {
                    sparse_grid.occupied.push(Voxel {
                        local_index: [
                            (x as f32 + 0.5 - sparse_half[0]).floor() as i32,
                            (y as f32 + 0.5 - sparse_half[1]).floor() as i32,
                            (z as f32 + 0.5 - sparse_half[2]).floor() as i32,
                        ],
                        block_local_coord: [0, 0, 0],
                        block_id: voxel_core::core_geom::BlockId(1),
                        attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                        grid_overlay: false,
                    });
                }
            }
        }
    }
    let sparse_compressed = compress(&sparse_grid);
    assert!(
        matches!(sparse_compressed.occupancy, Occupancy::Sparse(_)),
        "a 3%-fill grid must pick the sparse encoding"
    );
    let sparse_ratio = report("random 0.5%-fill 40³ (sparse case)", &sparse_grid);

    // A real village chunk (resolved through the chunk path).
    let make_tool = |kind, offset: [i64; 3], material| {
        let shape = SdfShape::from_blocks(kind, [5, 5, 5], 1, voxels_per_block);
        let mut node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material });
        node.transform = document::scene::NodeTransform::from_blocks(offset, voxels_per_block);
        node
    };
    let village = Scene::from_nodes(vec![
        make_tool(ShapeKind::Sphere, [0, 0, 0], MaterialChoice::Stone),
        make_tool(ShapeKind::Box, [8, 0, 0], MaterialChoice::Wood),
    ]);
    let (min_chunk, max_chunk) =
        village.covering_chunk_range(voxels_per_block).expect("placed");
    let mut total_raw = 0usize;
    let mut total_compressed = 0usize;
    for chunk_z in min_chunk[2]..=max_chunk[2] {
        for chunk_y in min_chunk[1]..=max_chunk[1] {
            for chunk_x in min_chunk[0]..=max_chunk[0] {
                let chunk =
                    village.resolve_chunk([chunk_x, chunk_y, chunk_z], voxels_per_block, 0);
                if chunk.occupied.is_empty() {
                    continue;
                }
                total_raw += raw_bytes(&chunk);
                total_compressed += compressed_binary_size(&compress(&chunk));
            }
        }
    }
    let village_ratio = total_raw as f64 / total_compressed.max(1) as f64;
    println!(
        "[ratio] village (all non-empty chunks): raw {total_raw} B, compressed \
         {total_compressed} B → {village_ratio:.2}× smaller"
    );

    // The solid SDF shapes net a strong dense win (~5×); the genuinely sparse
    // grid nets an even bigger sparse win; the village chunks net a win.
    assert!(
        sphere_ratio > 3.0,
        "a solid sphere should compress strongly via dense (>3×), got {sphere_ratio:.2}×"
    );
    assert!(
        sphere_best > 3.0,
        "the best per-chunk sphere piece should compress strongly (>3×), got {sphere_best:.2}×"
    );
    assert!(
        sparse_ratio > 1.5,
        "a 3%-fill grid should compress via the sparse encoding (>1.5×), got {sparse_ratio:.2}×"
    );
    assert!(
        village_ratio > 1.0,
        "the village chunks should net a compression win, got {village_ratio:.2}×"
    );
}
