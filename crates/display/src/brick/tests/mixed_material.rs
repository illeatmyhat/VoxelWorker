//! ADR 0013 — the CPU-march material reference ([`cpu_brick_hit_material`]) resolves a MIXED
//! brick's per-voxel materials from its cell-key tile (the same tile + voxel the shader's
//! `mixed_voxel_material` samples), masking the overlay bit off to the clean id; a UNIFORM hit
//! resolves the per-record material. This is the CPU half of the shader == reference bar; the
//! GPU half is `tests/gpu_parity.rs::brick_mixed_material_matches_cpu_reference`.
use crate::brick::*;
use crate::brick::build_brick_field;
use voxel_core::core_geom::CHUNK_BLOCKS;
use evaluation::cuboid::VoxelBox;
use evaluation::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
use std::collections::BTreeMap;
use std::sync::Arc;

const EDGE: u32 = 4;

/// A chunk holding ONE fully-solid boundary block at chunk-local `[0,0,0]` (world-block
/// `[0,0,0]`, so absolute voxel == brick-local voxel): its left X-half carries cell key
/// `left`, its right X-half `right`. Distinct keys ⇒ `classify_block_brick` sees disagreeing
/// cuboids and emits a MIXED brick; equal keys ⇒ a uniform brick.
fn one_block_chunk(left: u16, right: u16) -> Vec<([i32; 3], Arc<TwoLayerChunk>)> {
    let half = EDGE / 2;
    let mut microblocks = BTreeMap::new();
    microblocks.insert(
        [0, 0, 0],
        MicroblockGeometry {
            cuboids: vec![
                VoxelBox { min: [0, 0, 0], max: [half - 1, EDGE - 1, EDGE - 1], label: left },
                VoxelBox { min: [half, 0, 0], max: [EDGE - 1, EDGE - 1, EDGE - 1], label: right },
            ],
            seam_solidity: SeamSolidity { solid: [[true; 2]; 3] },
        },
    );
    let block_count = (CHUNK_BLOCKS * CHUNK_BLOCKS * CHUNK_BLOCKS) as usize;
    vec![(
        [0, 0, 0],
        Arc::new(TwoLayerChunk {
            voxels_per_block: EDGE,
            coarse: vec![None; block_count],
            coarse_overlay: vec![false; block_count],
            microblocks,
        }),
    )]
}

#[test]
fn reference_resolves_per_voxel_mixed_material() {
    let left = CellKey::compose(0, false).raw(); // clean id 0
    let right = CellKey::compose(1, true).raw(); // clean id 1, overlay bit set — must be masked off
    let build = build_brick_field(&one_block_chunk(left, right), EDGE);
    assert_eq!(build.mixed_brick_count(), 1, "the fixture must produce exactly one mixed brick");
    let records = pack_gpu_records(&build.brick_records, |_| false);

    for z in 0..EDGE as i32 {
        for y in 0..EDGE as i32 {
            for x in 0..EDGE as i32 {
                let material = cpu_brick_hit_material(
                    &records,
                    &build,
                    EDGE as i32,
                    CpuMarchHit { absolute_voxel: [x, y, z], face_normal: [-1, 0, 0] },
                );
                let expected = if (x as u32) < EDGE / 2 { 0 } else { 1 };
                assert_eq!(
                    material, expected,
                    "voxel ({x},{y},{z}) must resolve its authored clean material id \
                     (overlay bit masked)"
                );
            }
        }
    }
}

#[test]
fn reference_uniform_block_uses_record_material() {
    // Both halves share one key ⇒ a UNIFORM brick: no cell-key tile, material on the record.
    let key = CellKey::compose(2, false).raw();
    let build = build_brick_field(&one_block_chunk(key, key), EDGE);
    assert_eq!(build.mixed_brick_count(), 0, "a single-material block is not mixed");
    let records = pack_gpu_records(&build.brick_records, |_| false);
    let material = cpu_brick_hit_material(
        &records,
        &build,
        EDGE as i32,
        CpuMarchHit { absolute_voxel: [1, 1, 1], face_normal: [-1, 0, 0] },
    );
    assert_eq!(material, 2, "a uniform hit resolves the per-record material id");
}
