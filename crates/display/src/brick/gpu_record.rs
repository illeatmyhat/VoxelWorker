use super::*;

/// The sentinel marking a sculpted record whose atlas payload is NOT resident (the
/// residency-miss contract). Must match `NON_RESIDENT_ATLAS_SLOT` in the WGSL.
pub const NON_RESIDENT_ATLAS_SLOT: u32 = u32::MAX;

/// `BrickGpuRecord.kind` is a bit-packed field, LOW to HIGH — MUST match the decode in
/// `shaders/brick_raymarch.wgsl`:
///
/// * bits `[0, BRICK_RECORD_MATERIAL_ID_SHIFT)` — the **kind discriminant**:
///   `0` coarse, `1` sculpted-uniform, `2` sculpted-**mixed** (a block whose microblocks
///   disagree on their cell key: its per-voxel keys live in the material side atlas at
///   `BrickGpuRecord.cell_key_slot`, and the record's own material/overlay are don't-care).
///   Kinds 1 and 2 traverse identically — both descend into the occupancy atlas slot; only
///   the SHADE source differs.
/// * bits `[SHIFT, SHIFT + BRICK_RECORD_MATERIAL_ID_BITS)` — the block's **material-colour
///   index** (per-record shading: a multi-producer scene of distinct per-block materials
///   shades each hit from its own record).
/// * bit `BRICK_RECORD_OVERLAY_SHIFT` — the block's **on-face-grid overlay bit**, the other
///   half of its cell key. Per-RECORD (not a scene-wide uniform), so blocks that disagree on
///   the overlay are still one brick field. Meaningful for coarse + uniform records; a MIXED
///   record's overlay rides per-voxel in its cell-key texel instead.
///
/// One `u32`; the record struct grows only by the cell-key slot.
pub const BRICK_RECORD_MATERIAL_ID_SHIFT: u32 = 8;

/// Width of the material-id field in `BrickGpuRecord.kind` — the full `u16` a
/// [`BrickRecord::material_id`](crate::brick::BrickRecord::material_id) can hold, so the
/// packing caps nothing.
pub const BRICK_RECORD_MATERIAL_ID_BITS: u32 = 16;

/// The bit of `BrickGpuRecord.kind` carrying the record's overlay flag (immediately above the
/// material-id field).
pub const BRICK_RECORD_OVERLAY_SHIFT: u32 =
    BRICK_RECORD_MATERIAL_ID_SHIFT + BRICK_RECORD_MATERIAL_ID_BITS;

/// Mask isolating the kind discriminant below [`BRICK_RECORD_MATERIAL_ID_SHIFT`].
pub(crate) const BRICK_RECORD_KIND_MASK: u32 = (1 << BRICK_RECORD_MATERIAL_ID_SHIFT) - 1;

/// The `kind` discriminant of a COARSE record (a solid block-cube, no atlas slot) — the ONE
/// value this side decodes against; the discriminants themselves are pinned by
/// [`BrickPayload::kind_discriminant`](crate::brick::BrickPayload::kind_discriminant),
/// which [`gpu_record_of`] packs verbatim.
pub(crate) const BRICK_KIND_COARSE: u32 = 0;

/// ADR 0012 (H1) — the dynamic-offset uniform slots the field bind group indexes. The
/// SINGLE uniform buffer holds three `BrickUniformsPod` slots (each aligned up to the
/// device's `min_uniform_buffer_offset_alignment`): the SOLID band draw, plus the LOWER
/// and UPPER onion GHOST slabs. One bind group, records/atlas/clip-map shared; only the
/// bound dynamic offset (and the shading uniforms it selects) differ per draw.
pub(crate) const BRICK_UNIFORM_SLOT_SOLID: u32 = 0;
pub(crate) const BRICK_UNIFORM_SLOT_GHOST_LOWER: u32 = 1;
pub(crate) const BRICK_UNIFORM_SLOT_GHOST_UPPER: u32 = 2;
pub(crate) const BRICK_UNIFORM_SLOT_COUNT: u64 = 3;

/// The kind discriminant (0 coarse / 1 sculpted-uniform / 2 sculpted-mixed) of a packed
/// `BrickGpuRecord.kind` — the mirror of the WGSL `record_kind(kind)`. The material id and the
/// overlay bit live above it.
pub(crate) fn record_kind_discriminant(kind: u32) -> u32 {
    kind & BRICK_RECORD_KIND_MASK
}

/// The block MATERIAL colour index packed above the kind discriminant — the mirror of the WGSL
/// `record_material_id(kind)` (masked to the material field; the overlay bit rides above it). The
/// per-record shade of a coarse or sculpted-UNIFORM hit.
pub(crate) fn record_material_id(kind: u32) -> u32 {
    (kind >> BRICK_RECORD_MATERIAL_ID_SHIFT) & ((1 << BRICK_RECORD_MATERIAL_ID_BITS) - 1)
}

/// Does this packed record render as a solid block-cube — i.e. is it COARSE, or a sculpted
/// brick whose occupancy tile is not resident (the residency-miss contract)? The ONE reader of
/// "no voxel DDA for this block", mirroring the WGSL's `is_coarse` test; a MIXED record is a
/// sculpted one here (kinds 1 and 2 traverse identically — only the shade source differs).
pub(crate) fn record_is_coarse_form(record: &BrickGpuRecord) -> bool {
    record_kind_discriminant(record.kind) == BRICK_KIND_COARSE
        || record.atlas_slot == NON_RESIDENT_ATLAS_SLOT
}

/// One resident brick as the shader consumes it: the packed world-block key split
/// into a `(hi, lo)` u32 pair (sorted ascending — the in-shader binary search's
/// order), the packed `kind` field (kind discriminant + material id + overlay bit — see
/// [`BRICK_RECORD_MATERIAL_ID_SHIFT`]), the occupancy atlas slot (or
/// [`NON_RESIDENT_ATLAS_SLOT`]), and the MATERIAL SIDE ATLAS slot holding the block's
/// per-voxel cell-key tile.
///
/// `cell_key_slot` is [`NON_RESIDENT_ATLAS_SLOT`] for every non-MIXED record (coarse or
/// sculpted-uniform: they own no cell-key tile — their one cell key rides in `kind`), and it
/// carries the same sentinel meaning for a mixed record whose side-atlas tile is not resident:
/// shade from the record, degraded but correct.
///
/// Five `u32`s, tightly packed (std430 stride 20 — no padding on either side of the seam).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct BrickGpuRecord {
    pub key_hi: u32,
    pub key_lo: u32,
    pub kind: u32,
    pub atlas_slot: u32,
    pub cell_key_slot: u32,
}

/// Pack the build's records for the GPU. The record set is already **surface-only** (ADR
/// 0011 interior elision, fused into
/// [`build_brick_field`](crate::brick::build_brick_field): a fully-occluded interior
/// block never emits a record — no second mask pass exists), so this is a plain 1:1 mapping
/// and the uploaded buffer is ∝ surface, not volume, for a large solid. Hit-identity of the
/// surface-only set vs the interior-inclusive oracle build is gated in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`; interiors stay
/// queryable through the two-layer chunks (the clip-map derives from the chunks, and the
/// block-occupancy map box-fills coarse occupancy from the chunks).
///
/// `non_resident` marks sculpted slots to upload as [`NON_RESIDENT_ATLAS_SLOT`] — the
/// residency-miss test's forced-miss hook (and G4's future eviction seam); pass
/// `|_| false` for the all-resident set.
pub fn pack_gpu_records(
    records: &[BrickRecord],
    mut non_resident: impl FnMut(u32) -> bool,
) -> Vec<BrickGpuRecord> {
    records
        .iter()
        .map(|record| gpu_record_of(record, &mut non_resident))
        .collect()
}

/// Pack one brick record into its GPU form — the per-record body of [`pack_gpu_records`].
pub(crate) fn gpu_record_of(
    record: &crate::brick::BrickRecord,
    non_resident: &mut impl FnMut(u32) -> bool,
) -> BrickGpuRecord {
    let key = record.packed_world_block_key;
    // A MIXED brick is its OWN kind (2) and carries a second slot — its per-voxel cell-key
    // tile in the material side atlas. A coarse/uniform record's `cell_key_slot` is the
    // non-resident sentinel (it owns no tile: its one cell key is the material + overlay
    // packed into `kind` below). The discriminant is the payload's own — one source, no
    // parallel match to drift.
    let kind_discriminant = record.payload.kind_discriminant();
    let atlas_slot = match record.payload.occupancy_atlas_slot() {
        None => 0u32,
        Some(atlas_slot) if non_resident(atlas_slot) => NON_RESIDENT_ATLAS_SLOT,
        Some(atlas_slot) => atlas_slot,
    };
    let cell_key_slot = record
        .payload
        .cell_key_slot()
        .unwrap_or(NON_RESIDENT_ATLAS_SLOT);
    // Pack the block material above the kind discriminant, and the block's overlay bit above
    // that: the shader shades the hit from its OWN record, not a scene-wide uniform.
    let kind = kind_discriminant
        | ((record.material_id as u32) << BRICK_RECORD_MATERIAL_ID_SHIFT)
        | ((record.overlay as u32) << BRICK_RECORD_OVERLAY_SHIFT);
    let [key_hi, key_lo] = substrate::spatial::lattice_key::split_key_hi_lo(key);
    BrickGpuRecord {
        key_hi,
        key_lo,
        kind,
        atlas_slot,
        cell_key_slot,
    }
}

/// Write ONE sculpted brick's `edge³` occupancy tile into the persistent atlas texture
/// at its slot's tile origin (ADR 0011 G3 per-slot patch). `write_texture` needs no
/// 256-byte row alignment (unlike `copy_texture_to_buffer`), so a `bytes_per_row = edge`
/// sub-region upload lands exactly the slot's cube — untouched slots are never rewritten.
pub(crate) fn write_atlas_slot(
    queue: &wgpu::Queue,
    atlas_texture: &wgpu::Texture,
    tile_bytes: &[u8],
    brick_edge_voxels: u32,
    bricks_per_axis: u32,
    slot: u32,
) {
    let edge = brick_edge_voxels.max(1);
    let tiles = bricks_per_axis.max(1);
    let origin = wgpu::Origin3d {
        x: (slot % tiles) * edge,
        y: ((slot / tiles) % tiles) * edge,
        z: (slot / (tiles * tiles)) * edge,
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: atlas_texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        tile_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(edge),
            rows_per_image: Some(edge),
        },
        wgpu::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: edge,
        },
    );
}

/// Write ONE mixed brick's `edge³` cell-key tile into the persistent MATERIAL SIDE ATLAS at
/// its slot's tile origin — the twin of [`write_atlas_slot`] for the second pool, differing
/// only in the texel stride (2 bytes per R16Uint texel) and in `bricks_per_axis` (the side
/// atlas sizes from its OWN slot count).
pub(crate) fn write_cell_key_atlas_slot(
    queue: &wgpu::Queue,
    cell_key_texture: &wgpu::Texture,
    tile_bytes: &[u8],
    brick_edge_voxels: u32,
    bricks_per_axis: u32,
    slot: u32,
) {
    let edge = brick_edge_voxels.max(1);
    let tiles = bricks_per_axis.max(1);
    let origin = wgpu::Origin3d {
        x: (slot % tiles) * edge,
        y: ((slot / tiles) % tiles) * edge,
        z: (slot / (tiles * tiles)) * edge,
    };
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: cell_key_texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        tile_bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(edge * CELL_KEY_TEXEL_BYTES),
            rows_per_image: Some(edge),
        },
        wgpu::Extent3d {
            width: edge,
            height: edge,
            depth_or_array_layers: edge,
        },
    );
}
