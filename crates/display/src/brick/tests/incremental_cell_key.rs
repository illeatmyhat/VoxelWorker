//! The per-voxel cell-key side atlas (the CPU half): emission classifies a
//! sculpted block uniform vs MIXED, and only a mixed block owns a cell-key tile.
//!
//! These fixtures drive the emission builder DIRECTLY with hand-built two-layer chunks —
//! the tightest test of the CPU classifier. (The representability gate is now deleted, so a
//! mixed scene DOES reach the brick path through the renderer; the rendering side is proven by
//! the mixed-material golden + parity test. This module remains the CPU mirror's own contract.)
use crate::brick::*;
use voxel_core::core_geom::MaterialChoice;
use evaluation::cuboid::VoxelBox;
use document::scene::Scene;
use evaluation::two_layer_store::{MicroblockGeometry, TwoLayerChunk, TwoLayerStore};
use voxel_core::voxel::ShapeKind;
use document::voxel::GeometryParams;

/// The density the hand-built fixtures use — small enough to state a block's cuboids
/// by hand, and deliberately NOT 16 (the brick edge follows the density).
const HAND_DENSITY: u32 = 4;

/// A hand-built covering set of ONE chunk at `[0, 0, 0]`: `coarse_blocks` are
/// `(block, block_id, overlay)`, `sculpted_blocks` are `(block, cuboids)` whose cuboid
/// labels are render-cell keys ([`CellKey::compose`]).
fn hand_built_chunk(
    coarse_blocks: &[([u32; 3], u16, bool)],
    sculpted_blocks: &[([u32; 3], Vec<VoxelBox>)],
) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
    let block_count = (CHUNK_BLOCKS as usize).pow(3);
    let mut chunk = TwoLayerChunk {
        voxels_per_block: HAND_DENSITY,
        coarse: vec![None; block_count],
        coarse_overlay: vec![false; block_count],
        microblocks: std::collections::BTreeMap::new(),
    };
    for (block, block_id, overlay) in coarse_blocks {
        let flat = (block[2] as usize * CHUNK_BLOCKS as usize + block[1] as usize)
            * CHUNK_BLOCKS as usize
            + block[0] as usize;
        chunk.coarse[flat] = Some(BlockId(*block_id));
        chunk.coarse_overlay[flat] = *overlay;
    }
    for (block, cuboids) in sculpted_blocks {
        chunk.microblocks.insert(
            *block,
            MicroblockGeometry {
                cuboids: cuboids.clone(),
                seam_solidity: SeamSolidity::default(),
            },
        );
    }
    vec![([0, 0, 0], Arc::new(chunk))]
}

/// A block-local cuboid carrying the render-cell key `(block_id, overlay)`.
fn cell_box(min: [u32; 3], max: [u32; 3], block_id: u16, overlay: bool) -> VoxelBox {
    VoxelBox {
        min,
        max,
        label: CellKey::compose(block_id, overlay).raw(),
    }
}

/// The independent oracle for one block's per-voxel cell keys: paint each cuboid's key
/// into a dense `edge³` array in cuboid order (air stays [`AIR_CELL_KEY_DONT_CARE`]).
fn expected_cell_keys(cuboids: &[VoxelBox]) -> Vec<u16> {
    let edge = HAND_DENSITY as usize;
    let mut keys = vec![AIR_CELL_KEY_DONT_CARE; edge.pow(3)];
    for cuboid in cuboids {
        for z in cuboid.min[2]..=cuboid.max[2] {
            for y in cuboid.min[1]..=cuboid.max[1] {
                for x in cuboid.min[0]..=cuboid.max[0] {
                    keys[(z as usize * edge + y as usize) * edge + x as usize] = cuboid.label;
                }
            }
        }
    }
    keys
}

/// The independent oracle for one block's occupancy bytes (the same walk, occupancy only).
fn expected_occupancy_bytes(cuboids: &[VoxelBox]) -> Vec<u8> {
    let edge = HAND_DENSITY as usize;
    let mut bytes = vec![0u8; edge.pow(3)];
    for cuboid in cuboids {
        for z in cuboid.min[2]..=cuboid.max[2] {
            for y in cuboid.min[1]..=cuboid.max[1] {
                for x in cuboid.min[0]..=cuboid.max[0] {
                    bytes[(z as usize * edge + y as usize) * edge + x as usize] =
                        SCULPTED_BRICK_OCCUPIED;
                }
            }
        }
    }
    bytes
}

/// **Emission classifies uniform vs MIXED (the slice's core claim).** A block whose
/// microblock cuboids all share one cell key is UNIFORM: its material + overlay ride on
/// the record and it owns NO cell-key tile. A block whose cuboids disagree — on the
/// material OR on the overlay bit alone — is MIXED: it additionally carries a per-voxel
/// cell-key tile whose keys match its cuboids exactly, while its occupancy tile is
/// unchanged (byte-identical to the occupancy-only rasterization). A coarse block carries
/// its id + its chunk overlay marker and owns neither tile.
#[test]
fn emission_classifies_uniform_and_mixed_sculpted_blocks() {
    let uniform_block = [0u32, 0, 0];
    let uniform_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, true),
        cell_box([2, 0, 0], [3, 1, 3], 1, true), // same cell key ⇒ still uniform
    ];
    let mixed_material_block = [1u32, 0, 0];
    let mixed_material_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, false), // different block id ⇒ MIXED
    ];
    let mixed_overlay_block = [2u32, 0, 0];
    let mixed_overlay_cuboids = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 1, true), // same id, overlay differs ⇒ MIXED
    ];
    let coarse_block = [3u32, 0, 0];
    let chunks = hand_built_chunk(
        &[(coarse_block, 2, true)],
        &[
            (uniform_block, uniform_cuboids.clone()),
            (mixed_material_block, mixed_material_cuboids.clone()),
            (mixed_overlay_block, mixed_overlay_cuboids.clone()),
        ],
    );
    let build = build_brick_field(&chunks, HAND_DENSITY);

    // Exactly the two mixed blocks own a cell-key tile.
    assert_eq!(build.mixed_brick_count(), 2);
    assert_eq!(build.cell_key_tiles.len(), 2);
    assert_eq!(build.sculpted_brick_count(), 3, "all three boundary blocks are sculpted");

    // (a) The uniform block: one cell key on the record, NO cell-key tile.
    let record = build
        .find_record([uniform_block[0] as i64, uniform_block[1] as i64, uniform_block[2] as i64])
        .expect("uniform boundary block must have a record");
    assert_eq!(record.material_id, 1);
    assert!(record.overlay, "the uniform block's single cell key sets the overlay bit");
    assert_eq!(record.payload.cell_key_slot(), None, "a uniform brick owns no cell-key tile");
    assert!(matches!(record.payload, BrickPayload::Sculpted { .. }));
    assert_eq!(record.payload.kind_discriminant(), 1);

    // (b) + (c) The mixed blocks: a cell-key tile whose per-voxel keys are exactly the
    // cuboids', an occupancy tile unchanged by the classification.
    for (block, cuboids) in [
        (mixed_material_block, &mixed_material_cuboids),
        (mixed_overlay_block, &mixed_overlay_cuboids),
    ] {
        let record = build
            .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
            .expect("mixed boundary block must have a record");
        let BrickPayload::SculptedMixed {
            atlas_slot,
            cell_key_slot,
        } = record.payload
        else {
            panic!("a block whose cuboids disagree on their cell key must emit MIXED");
        };
        assert_eq!(
            record.payload.kind_discriminant(),
            2,
            "a MIXED brick is its own GPU record kind (it traverses like a sculpted one, \
             but shades from its cell-key tile)"
        );
        assert_eq!(
            build.cell_key_tiles[cell_key_slot as usize].as_slice(),
            expected_cell_keys(cuboids).as_slice(),
            "the cell-key tile must carry each voxel's own cuboid key at {block:?}"
        );
        assert_eq!(
            build.sculpted_brick_occupancy(atlas_slot),
            expected_occupancy_bytes(cuboids),
            "the occupancy tile is unchanged by the material classification at {block:?}"
        );
    }

    // (d) The coarse block: id + the chunk's overlay marker, no slot of either pool.
    let record = build
        .find_record([coarse_block[0] as i64, coarse_block[1] as i64, coarse_block[2] as i64])
        .expect("coarse block must have a record");
    assert_eq!(record.material_id, 2);
    assert!(record.overlay, "a coarse block carries its chunk's per-block overlay marker");
    assert_eq!(record.payload.occupancy_atlas_slot(), None);
    assert_eq!(record.payload.cell_key_slot(), None);

    // The two mixed bricks' cell-key slots are DISTINCT and dense in the wholesale build.
    let mut cell_key_slots: Vec<u32> = build
        .brick_records
        .iter()
        .filter_map(|record| record.payload.cell_key_slot())
        .collect();
    cell_key_slots.sort_unstable();
    assert_eq!(cell_key_slots, vec![0, 1]);
}

/// Resolve every live record's cell-key tile through the mirror's own slot numbering and
/// compare it against a from-scratch wholesale build of the same chunks (whose numbering
/// is dense and unrelated) — the cell-key half of the incremental-vs-wholesale parity
/// oracle: same kind, same material/overlay, same occupancy bytes, same per-voxel keys.
fn assert_cell_key_parity(
    mirror: &IncrementalBrickField,
    chunks: &[([i32; 3], Arc<TwoLayerChunk>)],
    label: &str,
) {
    let wholesale = build_brick_field(chunks, HAND_DENSITY);
    let incremental = mirror.to_build();
    assert_eq!(
        incremental.brick_records.len(),
        wholesale.brick_records.len(),
        "[{label}] record count must match wholesale"
    );
    assert_eq!(
        mirror.mixed_brick_count(),
        wholesale.mixed_brick_count(),
        "[{label}] live mixed-brick count must match wholesale"
    );
    for (mirrored, whole) in incremental
        .brick_records
        .iter()
        .zip(wholesale.brick_records.iter())
    {
        let block = unpack_world_block_key(whole.packed_world_block_key);
        assert_eq!(
            mirrored.packed_world_block_key, whole.packed_world_block_key,
            "[{label}] record order must match wholesale"
        );
        assert_eq!(
            (mirrored.material_id, mirrored.overlay),
            (whole.material_id, whole.overlay),
            "[{label}] record cell key at {block:?}"
        );
        assert_eq!(
            mirrored.payload.cell_key_slot().is_some(),
            whole.payload.cell_key_slot().is_some(),
            "[{label}] uniform/mixed verdict at {block:?}"
        );
        match (
            mirrored.payload.occupancy_atlas_slot(),
            whole.payload.occupancy_atlas_slot(),
        ) {
            (Some(mirror_slot), Some(whole_slot)) => assert_eq!(
                incremental.sculpted_brick_occupancy(mirror_slot),
                wholesale.sculpted_brick_occupancy(whole_slot),
                "[{label}] occupancy bytes at {block:?} (slots renumber, bytes do not)"
            ),
            (None, None) => {}
            _ => panic!("[{label}] payload kind disagreement at {block:?}"),
        }
        if let (Some(mirror_slot), Some(whole_slot)) = (
            mirrored.payload.cell_key_slot(),
            whole.payload.cell_key_slot(),
        ) {
            assert_eq!(
                mirror.cell_key_tile(mirror_slot).as_slice(),
                wholesale.cell_key_tiles[whole_slot as usize].as_slice(),
                "[{label}] cell-key tile at {block:?} (slots renumber, keys do not)"
            );
        }
    }
}

/// **A block flipping uniform↔mixed under an incremental edit allocates/frees its
/// cell-key slot** — in the SEPARATE material pool, leaving the occupancy pool alone (the
/// block stays a sculpted brick either way, so its occupancy slot is merely rewritten).
/// After every step the mirror agrees with a from-scratch wholesale build, cell-key tiles
/// included.
#[test]
fn incremental_uniform_mixed_flip_churns_only_the_cell_key_pool() {
    let block_a = [0u32, 0, 0];
    let block_b = [1u32, 0, 0];
    let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
    let mixed_a = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 2, false),
    ];
    let mixed_b = vec![
        cell_box([0, 0, 0], [1, 3, 3], 2, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, true), // overlay-only mix
    ];
    let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

    // Step 0 (wholesale seed): A uniform, B mixed — one cell-key slot in use.
    let step_0 = hand_built_chunk(
        &[],
        &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
    );
    let build = build_brick_field(&step_0, HAND_DENSITY);
    let (mut mirror, _atlas) = IncrementalBrickField::from_wholesale(build);
    assert_eq!(mirror.mixed_brick_count(), 1);
    assert_eq!(mirror.cell_key_slot_high_water(), 1);
    let occupancy_high_water = mirror.slot_high_water();
    assert_eq!(occupancy_high_water, 2, "both blocks are sculpted bricks");
    assert_cell_key_parity(&mirror, &step_0, "step 0 (wholesale seed)");

    // Step 1 (the FLIP): A becomes mixed, B becomes uniform. B's cell-key slot is freed
    // and A's allocation reuses it — the material pool churns, its high-water mark does
    // not grow, and the occupancy pool's does not move at all.
    let step_1 = hand_built_chunk(
        &[],
        &[(block_a, mixed_a.clone()), (block_b, uniform_b.clone())],
    );
    let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
    assert!(!update.atlas_grew, "the occupancy atlas must not grow on a material flip");
    assert_eq!(mirror.slot_high_water(), occupancy_high_water);
    assert_eq!(mirror.mixed_brick_count(), 1, "exactly one block is mixed after the flip");
    assert_eq!(
        mirror.cell_key_slot_high_water(),
        1,
        "the freed cell-key slot must be reused, not appended to"
    );
    let record_a = mirror
        .records()
        .iter()
        .find(|record| unpack_world_block_key(record.packed_world_block_key) == [0, 0, 0])
        .expect("block A must still have a record");
    let cell_key_slot = record_a
        .payload
        .cell_key_slot()
        .expect("block A is MIXED after the flip");
    assert_eq!(
        mirror.cell_key_tile(cell_key_slot).as_slice(),
        expected_cell_keys(&mixed_a).as_slice()
    );
    let record_b = mirror
        .records()
        .iter()
        .find(|record| unpack_world_block_key(record.packed_world_block_key) == [1, 0, 0])
        .expect("block B must still have a record");
    assert_eq!(record_b.payload.cell_key_slot(), None, "block B is UNIFORM after the flip");
    assert_eq!((record_b.material_id, record_b.overlay), (2, true));
    assert_cell_key_parity(&mirror, &step_1, "step 1 (uniform↔mixed flip)");

    // Step 2 (GROW): both blocks mixed — the second mixed brick appends a new cell-key
    // slot (the pool grows independently of the occupancy pool, which stays put).
    let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
    mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
    assert_eq!(mirror.mixed_brick_count(), 2);
    assert_eq!(mirror.cell_key_slot_high_water(), 2);
    assert_eq!(mirror.slot_high_water(), occupancy_high_water);
    assert_cell_key_parity(&mirror, &step_2, "step 2 (both mixed)");

    // Step 3 (FREE): both blocks uniform — every cell-key slot is freed (the mixed count
    // drops to zero; the high-water mark keeps the freed holes, as the occupancy pool does).
    let step_3 = hand_built_chunk(&[], &[(block_a, uniform_a), (block_b, uniform_b)]);
    mirror.apply_dirty_update(&step_3, &[[0, 0, 0]]);
    assert_eq!(mirror.mixed_brick_count(), 0, "no block is mixed any more");
    assert_eq!(
        mirror.cell_key_slot_high_water(),
        2,
        "freed slots keep their (dead) tiles until reallocated"
    );
    assert!(mirror
        .records()
        .iter()
        .all(|record| record.payload.cell_key_slot().is_none()));
    assert_cell_key_parity(&mirror, &step_3, "step 3 (both uniform again)");
}

/// A scene whose sculpted blocks are all UNIFORM (every scene the brick path renders
/// today) emits NO cell-key tile at all — the sparse-side-atlas contract, and the reason
/// the GPU bytes cannot move in this slice: `pack_gpu_records` reads the occupancy slot +
/// the record material, both untouched.
#[test]
fn a_uniform_scene_emits_no_cell_key_tiles() {
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
    let chunks = TwoLayerStore::enabled().build_covering_chunks(&scene, voxels_per_block, 0);
    let build = build_brick_field(&chunks, voxels_per_block);
    assert!(build.sculpted_brick_count() > 0, "the fixture must have sculpted bricks");
    assert!(
        build.cell_key_tiles.is_empty(),
        "a single-material scene must pay no per-voxel material cost"
    );
    assert_eq!(build.mixed_brick_count(), 0);
    assert!(
        build
            .brick_records
            .iter()
            .all(|record| !matches!(record.payload, BrickPayload::SculptedMixed { .. })),
        "no record may be mixed in a single-material scene"
    );

    // …and therefore packs a ZERO-LENGTH side atlas: the second pool costs such a scene
    // nothing at all (not even a tile grid).
    let side_atlas = build.cell_key_atlas_payload();
    assert!(side_atlas.bytes.is_empty(), "no mixed brick ⇒ no side-atlas bytes");
    assert_eq!(side_atlas.cell_key_slot_count, 0);
    assert_eq!(side_atlas.geometry.bricks_per_axis, 0);
    assert_eq!(side_atlas.geometry.atlas_dim_voxels, 0);
}

/// Read one cell-key slot's `edge³` keys back out of a PACKED side atlas — an INDEPENDENT
/// re-derivation of the GPU's addressing (linear slot → 3D tile origin, x-fastest; texels
/// little-endian u16, two bytes each), so a bug in the packer cannot hide behind the
/// packer's own arithmetic.
fn packed_cell_keys_at_slot(bytes: &[u8], bricks_per_axis: u32, slot: u32) -> Vec<u16> {
    let edge = HAND_DENSITY as usize;
    let tiles = bricks_per_axis.max(1) as usize;
    let atlas_dim = tiles * edge;
    let slot = slot as usize;
    let origin = [
        (slot % tiles) * edge,
        ((slot / tiles) % tiles) * edge,
        (slot / (tiles * tiles)) * edge,
    ];
    let mut keys = Vec::with_capacity(edge.pow(3));
    for local_z in 0..edge {
        for local_y in 0..edge {
            for local_x in 0..edge {
                let texel = ((origin[2] + local_z) * atlas_dim + origin[1] + local_y)
                    * atlas_dim
                    + origin[0]
                    + local_x;
                keys.push(u16::from_le_bytes([bytes[texel * 2], bytes[texel * 2 + 1]]));
            }
        }
    }
    keys
}

/// **The R16 side atlas packs each mixed brick's cell-key tile at its own slot origin.**
/// The pool is sized from ITS OWN slot count (two mixed bricks ⇒ a 2-tile grid), holds two
/// little-endian bytes per voxel, and every live slot reads back — through an independent
/// addressing oracle — as exactly that block's per-voxel cuboid keys. Bricks that are
/// uniform or coarse consume no texel here, whatever their occupancy slot.
#[test]
fn mixed_bricks_pack_the_r16_side_atlas_at_their_own_slot_origins() {
    let uniform_block = [0u32, 0, 0];
    let uniform_cuboids = vec![cell_box([0, 0, 0], [3, 3, 3], 1, true)];
    let mixed_material_block = [1u32, 0, 0];
    let mixed_material_cuboids = vec![
        cell_box([0, 0, 0], [1, 3, 3], 1, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, false),
    ];
    let mixed_overlay_block = [2u32, 0, 0];
    let mixed_overlay_cuboids = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 1, true),
    ];
    let chunks = hand_built_chunk(
        &[([3, 0, 0], 2, true)],
        &[
            (uniform_block, uniform_cuboids),
            (mixed_material_block, mixed_material_cuboids.clone()),
            (mixed_overlay_block, mixed_overlay_cuboids.clone()),
        ],
    );
    let build = build_brick_field(&chunks, HAND_DENSITY);
    let side_atlas = build.cell_key_atlas_payload();

    // The pool's OWN geometry: two mixed bricks ⇒ ceil(cbrt 2) = 2 tiles/axis, 8 voxels/axis
    // — while the occupancy pool holds THREE sculpted bricks (its own, larger, tile grid).
    assert_eq!(side_atlas.cell_key_slot_count, 2);
    assert_eq!(side_atlas.geometry.bricks_per_axis, 2);
    assert_eq!(side_atlas.geometry.atlas_dim_voxels, 2 * HAND_DENSITY);
    assert_eq!(side_atlas.geometry.brick_edge_voxels, HAND_DENSITY);
    assert_eq!(
        side_atlas.bytes.len(),
        2 * (2 * HAND_DENSITY as usize).pow(3),
        "two bytes per texel — the R16Uint stride"
    );
    assert_eq!(build.sculpted_brick_count(), 3);

    // Every live slot reads back as that block's own per-voxel keys.
    for (block, cuboids) in [
        (mixed_material_block, &mixed_material_cuboids),
        (mixed_overlay_block, &mixed_overlay_cuboids),
    ] {
        let record = build
            .find_record([block[0] as i64, block[1] as i64, block[2] as i64])
            .expect("a mixed block must have a record");
        let slot = record
            .payload
            .cell_key_slot()
            .expect("a mixed block must own a cell-key slot");
        assert_eq!(
            packed_cell_keys_at_slot(
                &side_atlas.bytes,
                side_atlas.geometry.bricks_per_axis,
                slot
            ),
            expected_cell_keys(cuboids),
            "the packed side atlas must carry {block:?}'s keys at its slot origin"
        );
    }

    // The two mixed slots occupy DISJOINT texel spans (the slot → origin map is injective):
    // exactly the two tiles' worth of texels are non-zero-keyed, the rest of the cube is
    // untouched fill.
    let occupied_texels = side_atlas
        .bytes
        .chunks_exact(2)
        .filter(|texel| u16::from_le_bytes([texel[0], texel[1]]) != AIR_CELL_KEY_DONT_CARE)
        .count();
    let expected_keyed: usize = [&mixed_material_cuboids, &mixed_overlay_cuboids]
        .iter()
        .map(|cuboids| {
            expected_cell_keys(cuboids)
                .iter()
                .filter(|key| **key != AIR_CELL_KEY_DONT_CARE)
                .count()
        })
        .sum();
    assert_eq!(
        occupied_texels, expected_keyed,
        "no key may land outside its own slot's tile"
    );
}

/// **The incremental pool's GPU work-list.** A uniform↔mixed flip reports exactly the
/// cell-key slots the sink must free and rewrite (the second pool's own lists, independent
/// of the occupancy atlas's), and the bytes it packs are the bytes a from-scratch build
/// packs — tile-for-tile at each live record's slot (the pools renumber across the two
/// paths; the texels do not). The `to_build()` parity-oracle style, for the side atlas.
#[test]
fn incremental_cell_key_pool_reports_its_work_list_and_packs_like_wholesale() {
    let block_a = [0u32, 0, 0];
    let block_b = [1u32, 0, 0];
    let uniform_a = vec![cell_box([0, 0, 0], [3, 3, 3], 1, false)];
    let mixed_a = vec![
        cell_box([0, 0, 0], [3, 3, 1], 1, false),
        cell_box([0, 0, 2], [3, 3, 3], 2, false),
    ];
    let mixed_b = vec![
        cell_box([0, 0, 0], [1, 3, 3], 2, false),
        cell_box([2, 0, 0], [3, 3, 3], 2, true),
    ];
    let uniform_b = vec![cell_box([0, 0, 0], [3, 3, 3], 2, true)];

    // Seed: A uniform, B mixed — one cell-key slot, a 1-tile side atlas.
    let step_0 = hand_built_chunk(
        &[],
        &[(block_a, uniform_a.clone()), (block_b, mixed_b.clone())],
    );
    let (mut mirror, _atlas) =
        IncrementalBrickField::from_wholesale(build_brick_field(&step_0, HAND_DENSITY));
    assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 1);

    // The FLIP: A becomes mixed, B becomes uniform. B's slot is freed and A's allocation
    // reuses it — so the sink frees slot 0 and rewrites slot 0, and neither tile grid grows.
    let step_1 = hand_built_chunk(&[], &[(block_a, mixed_a.clone()), (block_b, uniform_b)]);
    let update = mirror.apply_dirty_update(&step_1, &[[0, 0, 0]]);
    assert_eq!(update.freed_cell_key_slots, vec![0]);
    assert_eq!(update.written_cell_key_slots, vec![0]);
    assert!(!update.cell_key_atlas_grew, "a reused slot cannot grow the side atlas");
    assert!(!update.atlas_grew, "the occupancy pool is untouched by a material flip");

    // The dirty-slot bytes the sink uploads ARE the tile's little-endian texels.
    let mut expected_bytes = Vec::new();
    for key in expected_cell_keys(&mixed_a) {
        expected_bytes.extend_from_slice(&key.to_le_bytes());
    }
    assert_eq!(mirror.cell_key_slot_bytes(0), expected_bytes);

    // GROW: B becomes mixed too — the side atlas's OWN grow signal fires (2 slots ⇒ a
    // 2-tile grid), while the occupancy pool's does not.
    let step_2 = hand_built_chunk(&[], &[(block_a, mixed_a), (block_b, mixed_b)]);
    let update = mirror.apply_dirty_update(&step_2, &[[0, 0, 0]]);
    assert!(
        update.cell_key_atlas_grew,
        "the second mixed brick must grow the side atlas's tile grid"
    );
    assert!(!update.atlas_grew, "the occupancy pool's grid is unchanged");
    assert_eq!(update.written_cell_key_slots.len(), 2);
    assert_eq!(mirror.cell_key_atlas_geometry().bricks_per_axis, 2);

    // The wholesale-parity bar: the mirror's packed side atlas carries, at every live mixed
    // record's slot, exactly the tile a from-scratch build packs at ITS slot.
    let packed = mirror.pack_cell_key_atlas_payload();
    assert_eq!(
        packed, mirror.to_build().cell_key_atlas_payload(),
        "the two materialisations of one mirror must be byte-identical"
    );
    let wholesale = build_brick_field(&step_2, HAND_DENSITY);
    let wholesale_atlas = wholesale.cell_key_atlas_payload();
    assert_eq!(packed.geometry, wholesale_atlas.geometry);
    assert_eq!(packed.cell_key_slot_count, wholesale_atlas.cell_key_slot_count);
    for record in mirror.records() {
        let Some(mirror_slot) = record.payload.cell_key_slot() else {
            continue;
        };
        let block = unpack_world_block_key(record.packed_world_block_key);
        let whole_slot = wholesale
            .find_record(block)
            .and_then(|whole| whole.payload.cell_key_slot())
            .expect("the wholesale build must call the same block mixed");
        assert_eq!(
            packed_cell_keys_at_slot(
                &packed.bytes,
                packed.geometry.bricks_per_axis,
                mirror_slot
            ),
            packed_cell_keys_at_slot(
                &wholesale_atlas.bytes,
                wholesale_atlas.geometry.bricks_per_axis,
                whole_slot
            ),
            "packed cell keys at {block:?} (slots renumber, texels do not)"
        );
    }
}
