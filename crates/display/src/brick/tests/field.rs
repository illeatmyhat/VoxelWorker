use crate::brick::*;
use voxel_core::core_geom::MaterialChoice;
use document::scene::Scene;
use evaluation::two_layer_store::TwoLayerStore;
use voxel_core::voxel::{ShapeKind, Voxel};
use document::voxel::{GeometryParams};

// The bit-packed occupancy tile IS substrate's `BitCube`; its expand↔pack byte-parity and
// full-word run-set-mask oracles moved with it (see `crates/substrate/src/occupancy/bit_cube.rs`,
// renamed to substrate vocabulary). The tests below exercise the DOMAIN mapping that
// consumes it (record partition, atlas packing, incremental==wholesale parity).

/// The interior-INCLUSIVE oracle build ([`build_brick_field_all_blocks`]) maps the
/// two-layer partition one-to-one: coarse-solid → one kind-0 record (id carried, no
/// slot), boundary → one kind-1 record (dense unique slots, seam flags carried
/// unchanged), air → nothing; records sorted strictly ascending. This is the CPU half
/// of the ADR 0011 gate clause (a) for the record/atlas PACKING mechanics (which the
/// surface-only live build shares); the surface-only record CONTRACT itself is gated by
/// `build_emits_only_surface_records_of_a_solid_box`. The GPU parity test
/// re-asserts the bytes through the texture round-trip.
#[test]
fn brick_records_map_two_layer_partition_one_to_one() {
    // d4 deliberately (ADR 0011 Decision 1): the brick edge must follow the
    // density, not the number 16; odd voxel extents give partial boundary blocks.
    let voxels_per_block = 4;
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Sphere,
            size_voxels: [33, 33, 33],
            size_measurements: None,
            voxels_per_block,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    let two_layer_chunks =
        TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
    let build = build_brick_field_all_blocks(&two_layer_chunks, voxels_per_block);

    assert_eq!(build.brick_edge_voxels, voxels_per_block);
    assert!(
        build
            .brick_records
            .windows(2)
            .all(|pair| pair[0].packed_world_block_key < pair[1].packed_world_block_key),
        "records must be sorted strictly ascending (unique keys)"
    );

    let mut expected_coarse = 0usize;
    let mut expected_sculpted = 0usize;
    let mut seen_slots = std::collections::BTreeSet::new();
    for (chunk_coord, chunk) in &two_layer_chunks {
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
                        expected_coarse += 1;
                        let record = record.expect("coarse-solid block must have a record");
                        assert_eq!(record.payload.kind_discriminant(), 0);
                        assert_eq!(
                            record.payload,
                            BrickPayload::CoarseSolid { block_id },
                            "coarse record carries the block id, no atlas slot"
                        );
                        assert_eq!(record.seam_solidity.solid, [[true; 2]; 3]);
                    } else if let Some(geometry) = chunk.microblocks.get(&block) {
                        expected_sculpted += 1;
                        let record = record.expect("boundary block must have a record");
                        assert_eq!(record.payload.kind_discriminant(), 1);
                        let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                            panic!("boundary block must be a sculpted record");
                        };
                        assert!(
                            seen_slots.insert(atlas_slot),
                            "atlas slot {atlas_slot} assigned twice"
                        );
                        assert_eq!(
                            record.seam_solidity, geometry.seam_solidity,
                            "seam-solidity flags must carry across unchanged"
                        );
                    } else {
                        assert!(record.is_none(), "air block must emit nothing");
                    }
                }
            }
        }
    }
    assert_eq!(build.brick_records.len(), expected_coarse + expected_sculpted);
    assert_eq!(build.sculpted_brick_count(), expected_sculpted);
    // Slots are dense 0..count — the atlas holds exactly the sculpted bricks.
    assert_eq!(
        seen_slots.iter().copied().collect::<Vec<_>>(),
        (0..expected_sculpted as u32).collect::<Vec<_>>()
    );
    // The scene must actually exercise both kinds, else the mapping is untested.
    assert!(expected_coarse > 0, "fixture must contain coarse-solid blocks");
    assert!(expected_sculpted > 0, "fixture must contain boundary blocks");
}

/// **The surface-only record contract (ADR 0011 interior elision, fused into the
/// build).** [`build_brick_field`] over a SOLID box emits exactly the surface blocks (a
/// block with ≥1 absent/air neighbour) of the interior-inclusive oracle build
/// ([`build_brick_field_all_blocks`]) and omits the strictly-interior ones (all six
/// neighbours present + solid) — checked against an independent neighbour-presence
/// oracle over the FULL key set. The GPU
/// `brick_surface_elision_hit_set_unchanged` proves the surface-only build renders the
/// same hit set as the oracle build.
#[test]
fn build_emits_only_surface_records_of_a_solid_box() {
    let voxels_per_block = 4;
    // A solid box (ShapeKind::Box ignores wall_blocks — that is Tube-only), 6 blocks
    // per axis, so there is a genuine 4×4×4 fully-occluded interior to elide.
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Box,
            size_voxels: [6 * voxels_per_block, 6 * voxels_per_block, 6 * voxels_per_block],
            size_measurements: None,
            voxels_per_block,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    let two_layer_chunks =
        TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
    let full_build = build_brick_field_all_blocks(&two_layer_chunks, voxels_per_block);
    let surface_build = build_brick_field(&two_layer_chunks, voxels_per_block);
    assert!(!full_build.brick_records.is_empty(), "fixture must build records");
    // Every block of a solid box is coarse-solid (all faces solid).
    assert!(
        full_build
            .brick_records
            .iter()
            .all(|r| r.seam_solidity.solid == [[true; 2]; 3]),
        "a solid box classifies every block coarse-solid"
    );

    // Independent oracle: with all blocks coarse-solid, a block is INTERIOR iff all six
    // of its neighbours are present in the FULL record set. (The tiny fixture never
    // nears the packed-key lane limit, so no range guard is needed.)
    let full_keys: std::collections::HashSet<u64> = full_build
        .brick_records
        .iter()
        .map(|r| r.packed_world_block_key)
        .collect();
    let expected_surface_keys: Vec<u64> = full_build
        .brick_records
        .iter()
        .map(|r| r.packed_world_block_key)
        .filter(|&key| {
            let block = unpack_world_block_key(key);
            let all_neighbours_present = [
                [1i64, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            ]
            .iter()
            .all(|d| {
                let nb = [block[0] + d[0], block[1] + d[1], block[2] + d[2]];
                full_keys.contains(&pack_world_block_key(nb))
            });
            !all_neighbours_present
        })
        .collect();
    let surface_keys: Vec<u64> = surface_build
        .brick_records
        .iter()
        .map(|r| r.packed_world_block_key)
        .collect();
    assert_eq!(
        surface_keys, expected_surface_keys,
        "the surface-only build must emit exactly the oracle's surface blocks, in order"
    );
    // A solid box has a genuine interior to omit AND a surface to keep — the split is
    // non-trivial in both directions (else the elision would be vacuous or wrong).
    assert!(
        surface_build.brick_records.len() < full_build.brick_records.len(),
        "a solid box must have fully-occluded interior blocks to omit"
    );
    assert!(!surface_build.brick_records.is_empty(), "the surface blocks must be kept");
    // Both builds pack the identical sculpted atlas (the sculpted set is never elided).
    assert_eq!(surface_build.sculpted_atlas_bytes, full_build.sculpted_atlas_bytes);
    assert_eq!(surface_build.bricks_per_axis, full_build.bricks_per_axis);
}

/// The clip-map pyramid is CONSERVATIVE (ADR 0011 parity gate, coarse tier):
/// each level's occupied-cell set is a SUPERSET of the true occupied cells
/// (every record's cell present), sorted strictly ascending + unique, at ANY
/// density (block-denominated cells — nothing hard-codes 16). A scattered
/// multi-object scene so the levels actually span more than one cell.
#[test]
fn clipmap_pyramid_is_conservative_and_sorted() {
    use document::scene::{Node, NodeContent, NodeTransform};
    for &voxels_per_block in &[16u32, 4] {
        // A dozen small shapes far apart — the scattered scene the LOD targets.
        let mut nodes = Vec::new();
        for i in 0..12i64 {
            let shape = document::voxel::SdfShape::from_blocks(
                ShapeKind::Sphere,
                [3, 3, 3],
                1,
                voxels_per_block,
            );
            let mut node = Node::new(
                format!("s{i}"),
                NodeContent::Tool {
                    shape,
                    material: MaterialChoice::Stone,
                },
            );
            // Spread them ~16 blocks apart on a lattice so cells are scattered.
            node.transform = NodeTransform::from_blocks(
                [(i % 4) * 16, (i / 4) * 16, (i % 3) * 20],
                voxels_per_block,
            );
            nodes.push(node);
        }
        let scene = Scene::from_nodes(nodes);
        let two_layer_chunks =
            TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
        let build = build_brick_field(&two_layer_chunks, voxels_per_block);
        assert!(!build.brick_records.is_empty());
        let pyramid = ClipmapPyramid::from_records(&build.brick_records);

        for (level, blocks_per_cell) in [
            (&pyramid.level_1, CLIPMAP_LEVEL_1_BLOCKS_PER_CELL),
            (&pyramid.level_2, CLIPMAP_LEVEL_2_BLOCKS_PER_CELL),
            (&pyramid.level_3, CLIPMAP_LEVEL_3_BLOCKS_PER_CELL),
        ] {
            assert_eq!(level.blocks_per_cell, blocks_per_cell);
            assert!(
                level.cell_keys.windows(2).all(|pair| pair[0] < pair[1]),
                "level {blocks_per_cell} keys must be sorted strictly ascending + unique"
            );
            // Truth: the cell of every record must be present (superset ⇒ the
            // DDA never strides past a real surface).
            let level_set: std::collections::BTreeSet<u64> =
                level.cell_keys.iter().copied().collect();
            let cell_size = blocks_per_cell as i64;
            let mut true_cells = std::collections::BTreeSet::new();
            for record in &build.brick_records {
                let b = unpack_world_block_key(record.packed_world_block_key);
                let cell = [
                    b[0].div_euclid(cell_size),
                    b[1].div_euclid(cell_size),
                    b[2].div_euclid(cell_size),
                ];
                true_cells.insert(pack_world_block_key(cell));
            }
            assert!(
                true_cells.is_subset(&level_set),
                "level {blocks_per_cell} must cover every occupied cell (conservative)"
            );
            // The min-mip carries no cell the records don't (exactness of the
            // derivation — a spurious occupied cell would only cost perf, but
            // proves the fold has no stray keys).
            assert_eq!(level_set, true_cells);
            assert!(!level.cell_keys.is_empty());
        }
        // Each coarser level must not be finer than the one below (monotone
        // cell counts as the cell size grows 8× per level).
        assert!(pyramid.level_2.cell_keys.len() <= pyramid.level_1.cell_keys.len());
        assert!(pyramid.level_3.cell_keys.len() <= pyramid.level_2.cell_keys.len());
    }
}

/// The **chunk-sourced** pyramid ([`ClipmapPyramid::from_chunks`]) is BYTE-IDENTICAL to the
/// legacy record-sourced one ([`ClipmapPyramid::from_records`]) over the FULL, interior-
/// inclusive record set — the direct oracle for the interior-elision pyramid rework (ADR
/// 0011). `build_brick_field_all_blocks` is the interior-inclusive reference build (the live
/// `build_brick_field` is surface-only, so its records would give a subset pyramid). Covers a
/// solid box (heavy interior → the bulk fast path) and a scattered scene (partial chunks),
/// at two densities.
#[test]
fn clipmap_from_chunks_equals_from_full_records() {
    use document::scene::{Node, NodeContent, NodeTransform};
    for &voxels_per_block in &[16u32, 4] {
        // (a) A solid box: every interior chunk is fully-solid → exercises the bulk path.
        let box_scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [
                    7 * voxels_per_block,
                    7 * voxels_per_block,
                    7 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        // (b) A scattered scene: many partial chunks → exercises the per-block path.
        let mut nodes = Vec::new();
        for i in 0..8i64 {
            let shape =
                document::voxel::SdfShape::from_blocks(ShapeKind::Sphere, [3, 3, 3], 1, voxels_per_block);
            let mut node = Node::new(
                format!("s{i}"),
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform = NodeTransform::from_blocks(
                [(i % 3) * 14, (i / 3) * 14, (i % 2) * 18],
                voxels_per_block,
            );
            nodes.push(node);
        }
        let scattered_scene = Scene::from_nodes(nodes);

        for scene in [box_scene, scattered_scene] {
            let chunks =
                TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
            let full_build = build_brick_field_all_blocks(&chunks, voxels_per_block);
            let from_records = ClipmapPyramid::from_records(&full_build.brick_records);
            let from_chunks = ClipmapPyramid::from_chunks(&chunks);
            // Compare the SKIP levels only: `interior_masks` is a band-clip signal the
            // record-sourced oracle never carries (it is interior-inclusive), so it is
            // deliberately built empty there — it is not part of the min-mip identity claim.
            assert_eq!(
                (&from_chunks.level_1, &from_chunks.level_2, &from_chunks.level_3),
                (&from_records.level_1, &from_records.level_2, &from_records.level_3),
                "chunk-sourced pyramid must equal the full-record oracle (density {voxels_per_block})"
            );
        }
    }
}

/// **The band-clip interior-occupancy map marks EXACTLY the full-record block set (this
/// fix).** [`BlockOccupancyMasks::from_chunks`] must report a set bit for every block the
/// interior-INCLUSIVE oracle build (`build_brick_field_all_blocks`) carries a record for —
/// no more, no fewer — since that record set is what a band-clipped ray needs to resolve as
/// coarse cubes where the surface-only build elided them. Covers a solid box (the bulk
/// fully-solid path, heavy interior) and a scattered scene (the per-block partial path).
#[test]
fn block_occupancy_masks_mark_exactly_the_full_record_blocks() {
    use document::scene::{Node, NodeContent, NodeTransform};
    for &voxels_per_block in &[16u32, 4] {
        let box_scene = Scene::from_geometry(
            GeometryParams {
                shape: ShapeKind::Box,
                size_voxels: [
                    7 * voxels_per_block,
                    7 * voxels_per_block,
                    7 * voxels_per_block,
                ],
                size_measurements: None,
                voxels_per_block,
                wall_blocks: 1,
            },
            MaterialChoice::Stone,
        );
        let mut nodes = Vec::new();
        for i in 0..8i64 {
            let shape = document::voxel::SdfShape::from_blocks(
                ShapeKind::Sphere,
                [3, 3, 3],
                1,
                voxels_per_block,
            );
            let mut node = Node::new(
                format!("s{i}"),
                NodeContent::Tool { shape, material: MaterialChoice::Stone },
            );
            node.transform =
                NodeTransform::from_blocks([(i % 3) * 14, (i / 3) * 14, (i % 2) * 18], voxels_per_block);
            nodes.push(node);
        }
        let scattered_scene = Scene::from_nodes(nodes);

        for scene in [box_scene, scattered_scene] {
            let chunks =
                TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
            let full_build = build_brick_field_all_blocks(&chunks, voxels_per_block);
            let masks = BlockOccupancyMasks::from_chunks(&chunks);
            assert!(!masks.is_empty(), "the scene must occupy blocks");

            // Every full-record block reads as an occupied bit.
            let cell_size = BLOCK_OCCUPANCY_CELL_BLOCKS as i64;
            let bit_set = |world_block: [i64; 3]| -> bool {
                let cell = [
                    world_block[0].div_euclid(cell_size),
                    world_block[1].div_euclid(cell_size),
                    world_block[2].div_euclid(cell_size),
                ];
                let local = [
                    world_block[0].rem_euclid(cell_size) as usize,
                    world_block[1].rem_euclid(cell_size) as usize,
                    world_block[2].rem_euclid(cell_size) as usize,
                ];
                let bit = (local[2] * cell_size as usize + local[1]) * cell_size as usize
                    + local[0];
                masks.contains_bit(pack_world_block_key(cell), bit)
            };
            let mut expected_set: std::collections::BTreeSet<[i64; 3]> =
                std::collections::BTreeSet::new();
            for record in &full_build.brick_records {
                let block = unpack_world_block_key(record.packed_world_block_key);
                assert!(bit_set(block), "full-record block {block:?} missing from the mask");
                expected_set.insert(block);
            }
            // And no bit is set beyond the full-record set (the mask is not a superset).
            let mut mask_bits = 0u64;
            for mask in masks.cell_masks() {
                for word in mask {
                    mask_bits += word.count_ones() as u64;
                }
            }
            assert_eq!(
                mask_bits,
                expected_set.len() as u64,
                "mask must set exactly the full-record blocks (density {voxels_per_block})"
            );
        }
    }
}

/// CPU byte-exactness at a non-16 density: every sculpted brick's atlas bytes equal
/// the block occupancy the SHIPPED expansion (`expand_occupancy_into`, itself
/// proven bit-exact vs the dense oracle) reports — rasterization from cuboids and
/// expansion are independent paths over the same boundary set.
#[test]
fn sculpted_brick_bytes_match_expanded_occupancy_at_non_16_density() {
    let voxels_per_block = 4;
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: ShapeKind::Torus,
            size_voxels: [49, 13, 49],
            size_measurements: None,
            voxels_per_block,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    let two_layer_chunks =
        TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
    let build = build_brick_field(&two_layer_chunks, voxels_per_block);

    let edge = voxels_per_block as usize;
    let mut compared_bricks = 0usize;
    for (chunk_coord, chunk) in &two_layer_chunks {
        // Chunk-local occupancy bitmap via the shipped expansion (offset zero).
        let mut expanded: Vec<Voxel> = Vec::new();
        chunk.expand_occupancy_into(&mut expanded, [0, 0, 0]);
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block) as usize;
        let mut chunk_occupancy = vec![0u8; chunk_extent.pow(3)];
        for voxel in &expanded {
            let [x, y, z] = voxel.local_index;
            chunk_occupancy
                [(z as usize * chunk_extent + y as usize) * chunk_extent + x as usize] =
                SCULPTED_BRICK_OCCUPIED;
        }

        for block in chunk.microblocks.keys() {
            let world_block = [
                chunk_coord[0] as i64 * CHUNK_BLOCKS as i64 + block[0] as i64,
                chunk_coord[1] as i64 * CHUNK_BLOCKS as i64 + block[1] as i64,
                chunk_coord[2] as i64 * CHUNK_BLOCKS as i64 + block[2] as i64,
            ];
            let record = build.find_record(world_block).expect("sculpted record");
            let BrickPayload::Sculpted { atlas_slot } = record.payload else {
                panic!("boundary block must be sculpted");
            };
            let brick_bytes = build.sculpted_brick_occupancy(atlas_slot);
            let mut expected = vec![0u8; edge.pow(3)];
            for local_z in 0..edge {
                for local_y in 0..edge {
                    for local_x in 0..edge {
                        let chunk_voxel = [
                            block[0] as usize * edge + local_x,
                            block[1] as usize * edge + local_y,
                            block[2] as usize * edge + local_z,
                        ];
                        expected[(local_z * edge + local_y) * edge + local_x] =
                            chunk_occupancy[(chunk_voxel[2] * chunk_extent
                                + chunk_voxel[1])
                                * chunk_extent
                                + chunk_voxel[0]];
                    }
                }
            }
            assert_eq!(
                brick_bytes, expected,
                "brick bytes must equal the expanded block occupancy at {world_block:?}"
            );
            compared_bricks += 1;
        }
    }
    assert!(compared_bricks > 0, "fixture must contain sculpted bricks");
}
