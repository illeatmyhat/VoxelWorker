//! The GPU record format — the byte-level contract `shaders/brick_raymarch.wgsl` decodes.
//! Pinned here because the shader cannot assert: a silent desync (a kind discriminant, the
//! material mask, the overlay bit, the field order) shows up only as wrong pixels.
use crate::brick::*;
use crate::brick::{pack_world_block_key, BrickPayload, BrickRecord};
use voxel_core::core_geom::BlockId;
use evaluation::two_layer_store::SeamSolidity;

fn record(material_id: u16, overlay: bool, payload: BrickPayload) -> BrickRecord {
    BrickRecord {
        packed_world_block_key: pack_world_block_key([1, 2, 3]),
        material_id,
        overlay,
        payload,
        seam_solidity: SeamSolidity {
            solid: [[true; 2]; 3],
        },
    }
}

/// The `kind` word packs three independent facts — discriminant, material id, overlay bit —
/// in disjoint bit ranges, and the widened record carries the cell-key slot beside the
/// occupancy one. A coarse or UNIFORM record's cell-key slot is the non-resident sentinel
/// (it owns no tile); only a MIXED record names a slot of the material side atlas.
#[test]
fn the_packed_kind_word_splits_into_discriminant_material_and_overlay() {
    let records = [
        record(3, false, BrickPayload::CoarseSolid { block_id: BlockId(3) }),
        record(5, true, BrickPayload::Sculpted { atlas_slot: 7 }),
        record(
            9,
            true,
            BrickPayload::SculptedMixed {
                atlas_slot: 7,
                cell_key_slot: 2,
            },
        ),
    ];
    let packed = pack_gpu_records(&records, |_| false);

    // Coarse: kind 0, no occupancy tile, no cell-key tile — overlay + material still ride
    // on the record (a coarse block-cube shades from them).
    assert_eq!(record_kind_discriminant(packed[0].kind), 0);
    assert_eq!(packed[0].kind >> BRICK_RECORD_MATERIAL_ID_SHIFT & 0xffff, 3);
    assert_eq!(packed[0].kind >> BRICK_RECORD_OVERLAY_SHIFT, 0);
    assert_eq!(packed[0].cell_key_slot, NON_RESIDENT_ATLAS_SLOT);
    assert!(record_is_coarse_form(&packed[0]));

    // Sculpted-uniform: kind 1, the occupancy slot, the overlay bit set, still no tile.
    assert_eq!(record_kind_discriminant(packed[1].kind), 1);
    assert_eq!(packed[1].kind >> BRICK_RECORD_MATERIAL_ID_SHIFT & 0xffff, 5);
    assert_eq!(packed[1].kind >> BRICK_RECORD_OVERLAY_SHIFT, 1);
    assert_eq!(packed[1].atlas_slot, 7);
    assert_eq!(packed[1].cell_key_slot, NON_RESIDENT_ATLAS_SLOT);
    assert!(!record_is_coarse_form(&packed[1]));

    // Sculpted-MIXED: its own kind, the SAME occupancy slot discipline, plus a slot in the
    // (independently numbered) material side atlas. It traverses as a sculpted brick.
    assert_eq!(record_kind_discriminant(packed[2].kind), 2);
    assert_eq!(packed[2].atlas_slot, 7);
    assert_eq!(packed[2].cell_key_slot, 2);
    assert!(!record_is_coarse_form(&packed[2]));

    // The overlay bit must not bleed into the material id (the mask the WGSL applies).
    assert_eq!(packed[1].kind >> BRICK_RECORD_MATERIAL_ID_SHIFT & 0xffff, 5);
    // A non-resident OCCUPANCY slot still renders the coarse form, mixed or not.
    let forced = pack_gpu_records(&records, |_| true);
    assert!(forced.iter().all(record_is_coarse_form));
    assert_eq!(forced[2].cell_key_slot, 2, "residency is per-pool");
}

/// The record is five tightly-packed `u32`s — the std430 array stride the WGSL struct must
/// agree on (any padding here would shift every record the shader binary-searches).
#[test]
fn the_gpu_record_is_five_tightly_packed_words() {
    assert_eq!(std::mem::size_of::<BrickGpuRecord>(), 5 * 4);
    assert_eq!(std::mem::align_of::<BrickGpuRecord>(), 4);
}
