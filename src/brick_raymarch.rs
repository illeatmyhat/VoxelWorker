//! ADR 0011 G1 — the **minimal brick raymarch display sink**: a fullscreen pass
//! that walks a block-space DDA per pixel over the G0 [`BrickFieldBuild`] (sorted
//! records + R8 sculpted-brick atlas), finest LOD only (no clip-map — that is G2).
//!
//! * **Kind 0 (coarse)** records hit as a solid block-cube (interior elision on
//!   the display path); **kind 1 (sculpted)** records descend to a voxel DDA over
//!   the brick's atlas slot; a lookup miss steps on (air).
//! * **Residency-miss contract (ADR 0011 4a, decided at G1):** a sculpted record
//!   whose `atlas_slot` is [`NON_RESIDENT_ATLAS_SLOT`] renders its COARSE form —
//!   degraded-but-correct, never asserted/skipped. G4's residency rings plug into
//!   this hole as a pure eviction policy.
//! * **Depth compositing:** the pass runs INSIDE the shared 4× MSAA voxel pass and
//!   writes per-sample ray-hit depth via `frag_depth`, so the rasterized overlays
//!   (scene grid, infinite grid, points, gizmo, onion fog's depth-stop, view cube,
//!   egui) composite exactly as over the mesh.
//! * **Shading** transcribes `cuboid.wgsl` (per-voxel texture slice, lighting,
//!   material modulation, position-based grid overlay) and binds an identical
//!   procedural material atlas, so a brick-path pixel samples the same texel the
//!   mesh path would (parity gate clause (c)).
//!
//! Per ADR 0006 the sink is a **display derivation**: the records + atlas are
//! built from CPU truth (the two-layer boundary set) and nothing is ever read
//! back as truth. The CPU two-layer mesh stays the headless/no-GPU fallback and
//! the A/B reference (ADR 0011 Decision 6).
//!
//! The module also hosts the **CPU reference march** ([`cpu_march_brick_field`],
//! [`cpu_march_exact_occupancy`]) — a f32 mirror of the WGSL traversal used by
//! `tests/gpu_parity.rs` to gate the hit-voxel set against the exact evaluator.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::voxel::RecentreVoxels;
use crate::brick_field::{
    pack_clipmap_level_keys, unpack_world_block_key, upload_brick_atlas,
    upload_brick_cell_key_atlas, BlockOccupancyMasks, BrickFieldBuild, BrickFieldUpdate,
    BrickRecord, ClipmapLevel, ClipmapPyramid, IncrementalBrickField, SculptedAtlasPayload,
    SculptedCellKeyAtlasPayload, BLOCK_OCCUPANCY_MASK_WORDS, CELL_KEY_TEXEL_BYTES,
};
use crate::core_geom::{CellKey, MaterialChoice};
use crate::renderer::{LayerBand, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};

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
/// [`BrickRecord::material_id`](crate::brick_field::BrickRecord::material_id) can hold, so the
/// packing caps nothing.
pub const BRICK_RECORD_MATERIAL_ID_BITS: u32 = 16;

/// The bit of `BrickGpuRecord.kind` carrying the record's overlay flag (immediately above the
/// material-id field).
pub const BRICK_RECORD_OVERLAY_SHIFT: u32 =
    BRICK_RECORD_MATERIAL_ID_SHIFT + BRICK_RECORD_MATERIAL_ID_BITS;

/// Mask isolating the kind discriminant below [`BRICK_RECORD_MATERIAL_ID_SHIFT`].
const BRICK_RECORD_KIND_MASK: u32 = (1 << BRICK_RECORD_MATERIAL_ID_SHIFT) - 1;

/// The `kind` discriminant of a COARSE record (a solid block-cube, no atlas slot) — the ONE
/// value this side decodes against; the discriminants themselves are pinned by
/// [`BrickPayload::kind_discriminant`](crate::brick_field::BrickPayload::kind_discriminant),
/// which [`gpu_record_of`] packs verbatim.
const BRICK_KIND_COARSE: u32 = 0;

/// ADR 0012 (H1) — the dynamic-offset uniform slots the field bind group indexes. The
/// SINGLE uniform buffer holds three `BrickUniformsPod` slots (each aligned up to the
/// device's `min_uniform_buffer_offset_alignment`): the SOLID band draw, plus the LOWER
/// and UPPER onion GHOST slabs. One bind group, records/atlas/clip-map shared; only the
/// bound dynamic offset (and the shading uniforms it selects) differ per draw.
const BRICK_UNIFORM_SLOT_SOLID: u32 = 0;
const BRICK_UNIFORM_SLOT_GHOST_LOWER: u32 = 1;
const BRICK_UNIFORM_SLOT_GHOST_UPPER: u32 = 2;
const BRICK_UNIFORM_SLOT_COUNT: u64 = 3;

/// The kind discriminant (0 coarse / 1 sculpted-uniform / 2 sculpted-mixed) of a packed
/// `BrickGpuRecord.kind` — the mirror of the WGSL `record_kind(kind)`. The material id and the
/// overlay bit live above it.
fn record_kind_discriminant(kind: u32) -> u32 {
    kind & BRICK_RECORD_KIND_MASK
}

/// The block MATERIAL colour index packed above the kind discriminant — the mirror of the WGSL
/// `record_material_id(kind)` (masked to the material field; the overlay bit rides above it). The
/// per-record shade of a coarse or sculpted-UNIFORM hit.
fn record_material_id(kind: u32) -> u32 {
    (kind >> BRICK_RECORD_MATERIAL_ID_SHIFT) & ((1 << BRICK_RECORD_MATERIAL_ID_BITS) - 1)
}

/// Does this packed record render as a solid block-cube — i.e. is it COARSE, or a sculpted
/// brick whose occupancy tile is not resident (the residency-miss contract)? The ONE reader of
/// "no voxel DDA for this block", mirroring the WGSL's `is_coarse` test; a MIXED record is a
/// sculpted one here (kinds 1 and 2 traverse identically — only the shade source differs).
fn record_is_coarse_form(record: &BrickGpuRecord) -> bool {
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
/// [`build_brick_field`](crate::brick_field::build_brick_field): a fully-occluded interior
/// block never emits a record — no second mask pass exists), so this is a plain 1:1 mapping
/// and the uploaded buffer is ∝ surface, not volume, for a large solid. Hit-identity of the
/// surface-only set vs the interior-inclusive oracle build is gated in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`; interiors stay
/// queryable through the two-layer chunks (the clip-map derives from the chunks, the fog
/// box-fills coarse occupancy from the chunks).
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
fn gpu_record_of(
    record: &crate::brick_field::BrickRecord,
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
fn write_atlas_slot(
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
fn write_cell_key_atlas_slot(
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

/// The exact frame the march runs in — every value the shader's uniforms carry,
/// mirrored so the CPU reference march ([`cpu_march_brick_field`]) computes with
/// IDENTICAL parameters (ADR 0008: the frame is carried, never re-derived).
#[derive(Debug, Clone, Copy)]
pub struct BrickMarchFrame {
    pub view_projection: glam::Mat4,
    pub inverse_view_projection: glam::Mat4,
    /// x, y, width, height in physical pixels.
    pub viewport: [f32; 4],
    /// `floor(grid_dimensions / 2)` — the cuboid path's corner-anchoring half.
    pub grid_half_extent: glam::Vec3,
    /// `(recentre − half) mod edge` per axis — re-aligns block boundaries onto
    /// multiples of the brick edge in the shifted march frame.
    pub lattice_shift: [i32; 3],
    /// absolute block = sv block cell + this.
    pub block_bias: [i32; 3],
    /// absolute voxel = sv voxel cell + this.
    pub voxel_bias: [i32; 3],
    /// `[first_in_band, one_past_last]` voxel-Z in the shifted frame (band clip).
    pub band_voxel_sv: [i32; 2],
    /// Whether the band actually clips the resident solid's Z-extent — the gate for the
    /// block-occupancy interior fallback (a cut plane can enter an elided coarse interior).
    /// False under a full/non-clipping band, where the surface-only set is already hit-identical.
    pub band_clip_active: bool,
    /// The traversal AABB (resident-brick bounds ∩ band slab), shifted frame.
    pub traversal_lo: glam::Vec3,
    pub traversal_hi: glam::Vec3,
    pub brick_edge_voxels: i32,
    pub bricks_per_axis: u32,
}

/// One block-occupancy cell as the shader consumes it (ADR 0011 band-clip interior fallback):
/// the split `(hi, lo)` cell key, the fallback material, and the `512`-bit block bitmask. Field
/// order + packing MUST match `OccupancyCell` in `shaders/brick_raymarch.wgsl` (std430: all
/// `u32`, so a flat 80-byte record, `mask` stride 4). Sorted ascending by key — the shader's
/// binary-search order.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OccupancyCellPod {
    key_hi: u32,
    key_lo: u32,
    material: u32,
    // The fallback block's on-face-grid overlay bit (0/1), split out of the fallback word's
    // `OCCUPANCY_FALLBACK_OVERLAY_BIT`. Occupies the former pad slot (stride unchanged). With
    // the scene-wide overlay bool gone, an interior-elision coarse hit sources its overlay here.
    overlay: u32,
    mask: [u32; BLOCK_OCCUPANCY_MASK_WORDS],
}

/// Pack the block-occupancy map into the shader's sorted cell records (the parallel SoA
/// `cell_keys`/`cell_masks`/`cell_materials` → AoS). Empty ⇒ a single zeroed placeholder (its
/// count is 0, so the shader never binary-searches it).
fn pack_occupancy_cells(masks: &BlockOccupancyMasks) -> Vec<OccupancyCellPod> {
    masks
        .cell_keys()
        .iter()
        .zip(masks.cell_masks())
        .zip(masks.cell_materials())
        .map(|((&key, &mask), &fallback)| {
            let [key_hi, key_lo] = substrate::spatial::lattice_key::split_key_hi_lo(key);
            // The fallback word packs the overlay bit above the material colour index (the map
            // stores one u32 per cell); split it into the pod's two fields the shader reads.
            let overlay = u32::from(
                fallback & crate::brick_field::OCCUPANCY_FALLBACK_OVERLAY_BIT != 0,
            );
            let material = fallback & (crate::brick_field::OCCUPANCY_FALLBACK_OVERLAY_BIT - 1);
            OccupancyCellPod {
                key_hi,
                key_lo,
                material,
                overlay,
                mask,
            }
        })
        .collect()
}

/// The GPU-side uniform block; field order and 16-byte packing MUST match
/// `BrickUniforms` in `shaders/brick_raymarch.wgsl`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct BrickUniformsPod {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    viewport: [f32; 4],
    grid_half_extent: [f32; 3],
    voxels_per_block: f32,
    voxel_line_color: [f32; 3],
    grid_overlay_enabled: f32,
    block_line_color: [f32; 3],
    material_modulation_enabled: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    // Material is per-record (packed into `BrickGpuRecord.kind`, ADR 0011 G2), so no
    // scene-wide material id rides here — `record_count` plus the band-clip fields fill the slot.
    record_count: u32,
    // ADR 0011 band-clip interior fallback: 1 when the band clips the solid's Z-extent, so a
    // record MISS consults the block-occupancy map (elided coarse interiors the band exposes).
    band_clip_active: u32,
    // The block-occupancy cell count (`occupancy_cells` binary-search span); 0 ⇒ off.
    occupancy_cell_count: u32,
    // ADR 0012 (H1): the onion GHOST flag (0 = solid shade, 1 = flat translucent tint).
    // Occupies the former `_render_cell_pad2` slot.
    ghost_mode: u32,
    lattice_shift_and_edge: [i32; 4],
    block_bias_and_tiles: [i32; 4],
    voxel_bias: [i32; 4],
    band_voxel_sv: [i32; 4],
    // ADR 0011 G2 clip-map pyramid: [L1 blocks/cell, L1 cell count, L2 blocks/cell,
    // L2 cell count]. A zero count disables that level's hierarchical skip (the
    // flat G1 block-DDA), which is how the pyramid-on == off parity is A/B'd.
    clipmap_blocks_and_counts: [u32; 4],
    // ADR 0011 G4 third clip-map level: [L3 blocks/cell, L3 cell count, reserved,
    // reserved]. A fourth level was measured not to pay (G4 report), so zw stay 0.
    clipmap_blocks_and_counts_hi: [u32; 4],
    traversal_lo: [f32; 4],
    traversal_hi: [f32; 4],
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    material_atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    // ADR 0012 (H1): the onion ghost tint (linear RGB + src alpha), read only when
    // `ghost_mode != 0`. Appended so the solid draw's uniform layout is unchanged.
    ghost_tint: [f32; 4],
}

/// The G1 brick raymarch renderer: owns the record buffer, the sculpted atlas
/// texture, its own copy of the procedural material atlas (identical texels +
/// sub-rects to the cuboid path's), and the two pipelines (the MSAA render pass
/// entry + the single-sample hit-identity entry the parity net reads back).
pub struct BrickRaymarchRenderer {
    render_pipeline: wgpu::RenderPipeline,
    /// ADR 0012 (H1): the onion GHOST pipeline — same shader + layout as
    /// `render_pipeline`, but alpha-blends the flat-tinted ghost over the solid with the
    /// depth test `Less` with depth WRITE ON (so the nearest ghost surface wins — the
    /// render is builder-independent), while the solid (drawn first) occludes the ghost.
    ghost_render_pipeline: wgpu::RenderPipeline,
    hit_identity_pipeline: wgpu::RenderPipeline,
    /// ADR 0011 G2 — the single-sample COLOUR entry (`fragment_color_identity`) the
    /// colour-parity test reads back: shades each hit exactly as the MSAA render pass'
    /// centre-ray evaluation would, into a plain `Rgba8Unorm` target. Same pipeline
    /// layout (group 2 = loaded material) as the render pipeline.
    color_identity_pipeline: wgpu::RenderPipeline,
    /// ADR 0013 — the single-sample MATERIAL-identity entry (`fragment_material_identity`)
    /// the mixed-brick parity test reads back: reports each hit's RESOLVED per-voxel material
    /// id (the clean cell-key id for a mixed brick, else the per-record material) into an
    /// `Rgba32Uint` target. The direct "shader material == CPU-march reference" gate, with no
    /// shading to reproduce. Same pipeline layout as `hit_identity_pipeline`.
    material_identity_pipeline: wgpu::RenderPipeline,
    /// The uniform buffer: [`BRICK_UNIFORM_SLOT_COUNT`] `BrickUniformsPod` slots
    /// (solid + two ghost slabs), each `uniform_slot_stride` bytes, indexed by dynamic
    /// offset (ADR 0012 H1).
    uniform_buffer: wgpu::Buffer,
    /// The per-slot byte stride (`size_of::<BrickUniformsPod>` rounded up to the device's
    /// `min_uniform_buffer_offset_alignment`) — the dynamic offset multiplier.
    uniform_slot_stride: u32,
    /// (ADR 0012 H1) Whether each onion GHOST slab has a valid non-empty Z-range this
    /// frame (its uniform slot was written), so [`draw_ghost`](Self::draw_ghost) skips a
    /// degenerate slab (e.g. no layers below a band anchored at layer 0).
    ghost_lower_active: bool,
    ghost_upper_active: bool,
    field_bind_group_layout: wgpu::BindGroupLayout,
    field_bind_group: wgpu::BindGroup,
    material_bind_group: wgpu::BindGroup,
    /// ADR 0011 G2 — the group(2) LOADED-material bind group bound when NO VS block is
    /// applied: a dummy 1×1×6 D2Array (the shader ignores it while `voxel_bias.w == 0`).
    /// When a block is applied the app binds `LoadedMaterial::bind_group` at group(2)
    /// instead (built against the SAME `renderer::build_face_material_layout`), so the
    /// raymarch textures per-face by the owner's lattice rule. Kept alive here so the
    /// hit-identity / colour / ghost passes (which never sample it) can still satisfy
    /// the 3-group pipeline layout.
    dummy_loaded_material_bind_group: wgpu::BindGroup,
    /// Whether a VS block is applied this frame — mirrored into `voxel_bias.w` so the
    /// shader shades solid hits from the loaded D2Array (`true`) or the procedural
    /// atlas (`false`). Set by [`set_loaded_material_active`](Self::set_loaded_material_active).
    loaded_material_active: bool,
    /// The PERSISTENT sculpted-brick atlas texture (ADR 0011 G3). Kept across edits so an
    /// incremental patch ([`patch_brick_field`](Self::patch_brick_field)) writes only the
    /// dirty slots' texels via `write_texture` — untouched slots keep their bytes. A
    /// wholesale install or an atlas GROW recreates it.
    atlas_texture: wgpu::Texture,
    /// The persistent atlas texture's per-axis dimension in voxels (`>= 1`; the 1³
    /// placeholder when no field is installed). A patch whose build dim differs must
    /// recreate the texture (grow/shrink), not `write_texture` into a stale-sized one.
    atlas_texture_dim: u32,
    /// The PERSISTENT **material side atlas** (R16Uint): the MIXED bricks' per-voxel cell-key
    /// tiles, a SECOND independently-pooled texture beside `atlas_texture`. Patched per dirty
    /// cell-key slot exactly as the occupancy atlas is; a 1³ placeholder when no brick is mixed.
    /// The shader samples it per-voxel for kind-2 records (`mixed_voxel_cell_key`), so a mixed
    /// scene shades each voxel from its own clean id + overlay bit.
    cell_key_texture: wgpu::Texture,
    /// The side atlas's per-axis dimension in voxels (`>= 1`; 1 for the placeholder) — the
    /// grow/shrink test of the second pool, independent of `atlas_texture_dim`.
    cell_key_texture_dim: u32,
    /// The number of atlas slots the LAST update wrote (ADR 0011 G3 "per-edit cost ∝ dirty
    /// region" instrument): a wholesale install writes every sculpted slot; an incremental
    /// patch writes only the dirty chunks' slots (unless the atlas grew — then every slot).
    last_atlas_slots_written: u32,
    record_count: u32,
    /// The composite recentre the boundary set was resolved under (ADR 0008 —
    /// carried from the install as [`RecentreVoxels`], the same value the two-layer mesher
    /// bakes; unwrapped with `.voxels()` only where `march_frame` packs the uniform).
    recentre_voxels: RecentreVoxels,
    brick_edge_voxels: u32,
    bricks_per_axis: u32,
    /// Inclusive absolute world-block bounds of the resident record set (the
    /// traversal AABB's source); `None` when no field is installed.
    absolute_block_bounds: Option<([i64; 3], [i64; 3])>,
    /// ADR 0011 G2 clip-map pyramid: cells/blocks per level + the installed cell
    /// counts (0 ⇒ that level's hierarchical skip is off). Uploaded to the shader
    /// as `clipmap_blocks_and_counts`.
    clipmap_level_1_blocks: u32,
    clipmap_level_1_count: u32,
    clipmap_level_2_blocks: u32,
    clipmap_level_2_count: u32,
    clipmap_level_3_blocks: u32,
    clipmap_level_3_count: u32,
    /// ADR 0011 band-clip interior fallback: the present block-occupancy cell count uploaded
    /// last install (0 ⇒ the shader's record-miss fallback never fires). The occupancy buffer is
    /// rebuilt with the records/pyramid in [`rebuild_field_state`](Self::rebuild_field_state).
    occupancy_cell_count: u32,
}

impl BrickRaymarchRenderer {
    /// Build the renderer's PERSISTENT half — pipelines, material atlas, uniform
    /// buffer — with an EMPTY brick field (`draw` no-ops until a field is
    /// installed). The per-edit half is
    /// [`install_brick_field`](Self::install_brick_field): records + atlas swap in
    /// WITHOUT recompiling pipelines, so a live edit never pays a pipeline build.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        // ADR 0012 (H1): ONE uniform buffer of three dynamic-offset slots (solid + two
        // onion ghost slabs). Each slot is padded up to the device's uniform-offset
        // alignment so a dynamic offset lands slot `n` exactly.
        let uniform_size = std::mem::size_of::<BrickUniformsPod>() as u64;
        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let uniform_slot_stride = uniform_size.div_ceil(uniform_alignment) * uniform_alignment;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick raymarch uniforms"),
            size: uniform_slot_stride * BRICK_UNIFORM_SLOT_COUNT,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_slot_stride = uniform_slot_stride as u32;

        // Placeholder field: one zeroed record + a 1³ atlas (record_count 0 means
        // the binary search never reads either).
        let placeholder = [BrickGpuRecord::zeroed()];
        let record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch records"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let empty_atlas = SculptedAtlasPayload {
            bytes: Vec::new(),
            geometry: crate::brick_field::SculptedAtlasGeometry {
                bricks_per_axis: 0,
                atlas_dim_voxels: 0,
                brick_edge_voxels: 1,
            },
            sculpted_slot_count: 0,
        };
        let atlas_texture = upload_brick_atlas(device, queue, &empty_atlas);
        let atlas_texture_dim = empty_atlas.geometry.atlas_dim_voxels.max(1);
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // The material side atlas starts EMPTY (a 1³ R16Uint placeholder): no mixed brick, no
        // cell-key tile. An install/patch that carries mixed bricks replaces/patches it.
        let empty_cell_key_atlas = SculptedCellKeyAtlasPayload::empty(1);
        let cell_key_texture = upload_brick_cell_key_atlas(device, queue, &empty_cell_key_atlas);
        let cell_key_texture_dim = empty_cell_key_atlas.geometry.atlas_dim_voxels.max(1);
        let cell_key_view = cell_key_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Placeholder clip-map key buffers (count 0 ⇒ the shader never reads them).
        let placeholder_keys = [[0u32, 0u32]];
        let level_1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L1 keys"),
            contents: bytemuck::cast_slice(&placeholder_keys),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let level_2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L2 keys"),
            contents: bytemuck::cast_slice(&placeholder_keys),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let level_3_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L3 keys"),
            contents: bytemuck::cast_slice(&placeholder_keys),
            usage: wgpu::BufferUsages::STORAGE,
        });
        // Placeholder occupancy buffer (count 0 ⇒ the shader never binary-searches it).
        let placeholder_occupancy = [OccupancyCellPod::zeroed()];
        let occupancy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch block-occupancy cells"),
            contents: bytemuck::cast_slice(&placeholder_occupancy),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("brick raymarch field layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            // ADR 0012 (H1): dynamic offset selects the solid / ghost-lower /
                            // ghost-upper slot from the one 3-slot uniform buffer.
                            has_dynamic_offset: true,
                            min_binding_size: std::num::NonZeroU64::new(
                                std::mem::size_of::<BrickUniformsPod>() as u64,
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // ADR 0011 G2: the two clip-map occupancy levels (sorted cell keys).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // ADR 0011 G4: the third clip-map level (512-block cell keys).
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // ADR 0011 band-clip interior fallback: the block-occupancy cells.
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The MATERIAL SIDE ATLAS (the mixed bricks' per-voxel cell keys). Its
                    // sample type is UINT, not float: the texel IS the u16 cell key, read
                    // exactly with `textureLoad` (a filterable-float binding would both
                    // validation-error against an R16Uint view and round the id).
                    wgpu::BindGroupLayoutEntry {
                        binding: 7,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });
        let field_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick raymarch field bind group"),
            layout: &field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // Sized to ONE slot (dynamic offset selects which) — not the whole
                    // 3-slot buffer, so `offset + size` stays in bounds at every slot.
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(uniform_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: record_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: level_1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: level_2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: level_3_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&cell_key_view),
                },
            ],
        });

        // The material atlas: the SAME procedural packing + nearest/clamp sampler
        // the cuboid path builds, so both paths sample identical texels.
        let material_atlas = crate::texture_atlas::MaterialAtlas::from_procedural_materials();
        let material_bind_group_layout = crate::cuboid_mesh::build_atlas_bind_group_layout(device);
        let material_texture =
            crate::cuboid_mesh::upload_atlas_texture(device, queue, &material_atlas);
        let material_view = material_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("brick raymarch material sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick raymarch material bind group"),
            layout: &material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&material_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&material_sampler),
                },
            ],
        });

        // ADR 0011 G2 — the group(2) LOADED-material slot. Its layout is the SAME
        // `renderer::build_face_material_layout` the mesh path (and `LoadedMaterial`)
        // uses, so an applied block's bind group binds here directly. A dummy 1×1×6
        // sRGB D2Array binds when no block is applied (the shader ignores it while
        // `voxel_bias.w == 0`); the same nearest/clamp sampler slices it like the mesh.
        let loaded_material_layout = crate::renderer::build_face_material_layout(device);
        let dummy_loaded_texture = crate::renderer::upload_face_material_texture(
            device,
            queue,
            1,
            1,
            &[&[0u8, 0, 0, 255]; 6],
        );
        let dummy_loaded_view = dummy_loaded_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let dummy_loaded_material_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("brick raymarch dummy loaded material bind group"),
                layout: &loaded_material_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&dummy_loaded_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&material_sampler),
                    },
                ],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brick raymarch shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brick_raymarch.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brick raymarch pipeline layout"),
            bind_group_layouts: &[
                Some(&field_bind_group_layout),
                Some(&material_bind_group_layout),
                Some(&loaded_material_layout),
            ],
            immediate_size: 0,
        });

        // The live pass: fullscreen triangle INSIDE the 4× MSAA voxel pass, writing
        // colour + per-sample ray-hit depth (Less, exactly the mesh pipeline's
        // depth state) so everything after composites unchanged.
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brick raymarch render pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_render"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // ADR 0012 H1.5 (spike) — the onion GHOST pipeline is the Beer–Lambert HAZE
        // variant: `fragment_ghost_haze` accumulates the ray's in-solid path length across
        // the slab and outputs the tint at `1 − exp(−k·thickness)` — the retired volumetric
        // fog's aerogel look, sourced from the brick field alone. Alpha-blended, depth test
        // `Less` with depth WRITE OFF: the haze march produces exactly ONE fragment per slab
        // per pixel (all in-slab thickness is folded in-shader), so there is no intra-slab
        // overlap for a depth write to disambiguate (the crisp ghost's reason for write-ON),
        // and with the SAME tint RGB on both slabs the two-slab alpha composite is
        // order-independent. The solid (drawn first, depth written) still occludes each slab
        // via the haze's first-in-solid `frag_depth` — exact per slab, since z(t) is
        // monotonic so a slab's t-interval lies entirely on one side of any solid-band hit.
        let ghost_render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("brick raymarch onion ghost haze pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_ghost_haze"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // The parity-harness pass: single sample, no depth, hit voxel identity into
        // an Rgba32Uint target (read back by tests/gpu_parity.rs only).
        let hit_identity_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("brick raymarch hit-identity pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_hit_identity"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        // ADR 0011 G2 — the colour-parity pass: single sample, no depth, the SHADED
        // colour into a plain `Rgba8Unorm` target (read back by tests/gpu_parity.rs).
        let color_identity_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("brick raymarch colour-identity pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_color_identity"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        // ADR 0013 — the material-identity pass: single sample, no depth, the resolved
        // per-voxel material id per hit into an `Rgba32Uint` target (read back by
        // tests/gpu_parity.rs). Same layout as the hit-identity pass.
        let material_identity_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("brick raymarch material-identity pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_material_identity"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba32Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });

        Self {
            render_pipeline,
            ghost_render_pipeline,
            hit_identity_pipeline,
            color_identity_pipeline,
            material_identity_pipeline,
            uniform_buffer,
            uniform_slot_stride,
            ghost_lower_active: false,
            ghost_upper_active: false,
            field_bind_group_layout,
            field_bind_group,
            material_bind_group,
            dummy_loaded_material_bind_group,
            loaded_material_active: false,
            atlas_texture,
            atlas_texture_dim,
            cell_key_texture,
            cell_key_texture_dim,
            last_atlas_slots_written: 0,
            record_count: 0,
            recentre_voxels: RecentreVoxels::new([0, 0, 0]),
            brick_edge_voxels: 1,
            bricks_per_axis: 0,
            absolute_block_bounds: None,
            clipmap_level_1_blocks: crate::brick_field::CLIPMAP_LEVEL_1_BLOCKS_PER_CELL,
            clipmap_level_1_count: 0,
            clipmap_level_2_blocks: crate::brick_field::CLIPMAP_LEVEL_2_BLOCKS_PER_CELL,
            clipmap_level_2_count: 0,
            clipmap_level_3_blocks: crate::brick_field::CLIPMAP_LEVEL_3_BLOCKS_PER_CELL,
            clipmap_level_3_count: 0,
            occupancy_cell_count: 0,
        }
    }

    /// Install (or replace) the brick field: upload the packed records + the
    /// sculpted atlas and rebuild the field bind group — the per-edit swap, no
    /// pipeline work. `gpu_records` is [`pack_gpu_records`]' output (possibly with
    /// forced non-resident slots); `recentre_voxels` the resolve's carried
    /// recentre. Material AND the on-face-grid overlay are per-record (packed in
    /// `gpu_records`, ADR 0011 G2 / material atlas) — no scene-wide overlay rides here.
    #[allow(clippy::too_many_arguments)]
    pub fn install_brick_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[BrickRecord],
        atlas: &SculptedAtlasPayload,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: RecentreVoxels,
    ) {
        // The occupancy-only install: the MATERIAL SIDE ATLAS installs EMPTY — the honest default
        // for a caller that holds no cell-key payload (a scene with no MIXED brick). A field WITH
        // mixed bricks installs through
        // [`install_brick_field_with_cell_keys`](Self::install_brick_field_with_cell_keys).
        let empty_cell_keys =
            SculptedCellKeyAtlasPayload::empty(atlas.geometry.brick_edge_voxels);
        debug_assert!(
            gpu_records
                .iter()
                .all(|record| record.cell_key_slot == NON_RESIDENT_ATLAS_SLOT),
            "a field with MIXED bricks must install its cell-key side atlas \
             (install_brick_field_with_cell_keys), not the empty one"
        );
        self.install_brick_field_with_cell_keys(
            device,
            queue,
            records,
            atlas,
            &empty_cell_keys,
            gpu_records,
            pyramid,
            recentre_voxels,
        );
    }

    /// Install (or replace) the brick field INCLUDING its material side atlas — the full
    /// wholesale seam: both pools' textures are (re)created from scratch and every slot of each
    /// uploaded. The pools are independent (own slot numbering, own tile grid), so a field with
    /// 10k sculpted bricks and 3 mixed ones uploads a 22-tile occupancy cube and a 2-tile
    /// cell-key cube.
    #[allow(clippy::too_many_arguments)]
    pub fn install_brick_field_with_cell_keys(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[BrickRecord],
        atlas: &SculptedAtlasPayload,
        cell_key_atlas: &SculptedCellKeyAtlasPayload,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: RecentreVoxels,
    ) {
        // A wholesale install (re)creates the atlas texture from scratch and uploads
        // every sculpted slot — the from-scratch / scene-load / gate-re-engage path.
        let atlas_texture = upload_brick_atlas(device, queue, atlas);
        self.atlas_texture = atlas_texture;
        self.atlas_texture_dim = atlas.geometry.atlas_dim_voxels.max(1);
        self.last_atlas_slots_written = atlas.sculpted_slot_count;
        self.cell_key_texture = upload_brick_cell_key_atlas(device, queue, cell_key_atlas);
        self.cell_key_texture_dim = cell_key_atlas.geometry.atlas_dim_voxels.max(1);
        let atlas_view = self
            .atlas_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.rebuild_field_state(
            device,
            &atlas_view,
            records,
            atlas.geometry.brick_edge_voxels,
            atlas.geometry.bricks_per_axis,
            gpu_records,
            pyramid,
            recentre_voxels,
        );
    }

    /// **ADR 0011 G3 — incremental dirty-brick patch.** Patch ONLY the dirty slots of the
    /// PERSISTENT atlas from an [`IncrementalBrickField`]
    /// update, then swap in the merged records + rebuilt pyramid — no wholesale atlas
    /// re-upload, no occupancy readback. `update.written_slots` are the only texels
    /// touched (untouched slots keep their bytes) UNLESS `update.atlas_grew`, where the
    /// tile grid moved and the whole atlas is re-packed (the one legitimate wholesale
    /// re-pack, ADR 0007 resize precedent). Records, atlas geometry, and each dirty slot's
    /// bytes are read straight from `mirror` (the single CPU owner — item 9), so the
    /// per-edit path never materialises a `BrickFieldBuild`.
    ///
    /// Preconditions the live shell (`WindowedState::rebuild_geometry`) upholds: a field is
    /// already installed AND its density/frame match the mirror (an incremental edit never
    /// changes density — that routes wholesale). Records + pyramid re-upload whole (they
    /// are small — the traffic G3 kills is the atlas texels + the re-evaluation).
    #[allow(clippy::too_many_arguments)]
    pub fn patch_brick_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mirror: &IncrementalBrickField,
        update: &BrickFieldUpdate,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: RecentreVoxels,
    ) {
        // Read the atlas geometry + dirty-slot bytes straight from the single-owner mirror —
        // no `to_build()` (item 9: the per-edit full records clone + whole-atlas re-pack is gone).
        let geometry = mirror.atlas_geometry();
        let target_dim = geometry.atlas_dim_voxels.max(1);
        if update.atlas_grew || target_dim != self.atlas_texture_dim {
            // The tile grid grew/shrank: every slot's 3D position moved, so recreate the
            // texture and re-upload wholesale (ADR 0011 pitfalls — the resize is the one
            // place a full re-pack is legitimate, logged by the caller).
            let atlas = mirror.pack_atlas_payload();
            self.atlas_texture = upload_brick_atlas(device, queue, &atlas);
            self.atlas_texture_dim = target_dim;
            self.last_atlas_slots_written = atlas.sculpted_slot_count;
        } else {
            // Steady state: write ONLY the dirty slots' tiles into the persistent texture.
            // Untouched slots — and freed (dead) slots — keep their texels untouched.
            for &slot in &update.written_slots {
                let tile_bytes = mirror.sculpted_slot_bytes(slot);
                write_atlas_slot(
                    queue,
                    &self.atlas_texture,
                    &tile_bytes,
                    geometry.brick_edge_voxels,
                    geometry.bricks_per_axis,
                    slot,
                );
            }
            self.last_atlas_slots_written = update.written_slots.len() as u32;
        }
        // The MATERIAL SIDE ATLAS patches by the SAME discipline over its OWN pool: its own
        // grow test (its tile grid sizes from its own slot count), its own dirty-slot list.
        // A no-mixed-brick field leaves both empty, so this is a no-op there.
        let cell_key_geometry = mirror.cell_key_atlas_geometry();
        let cell_key_target_dim = cell_key_geometry.atlas_dim_voxels.max(1);
        if update.cell_key_atlas_grew || cell_key_target_dim != self.cell_key_texture_dim {
            let cell_key_atlas = mirror.pack_cell_key_atlas_payload();
            self.cell_key_texture = upload_brick_cell_key_atlas(device, queue, &cell_key_atlas);
            self.cell_key_texture_dim = cell_key_target_dim;
        } else {
            for &slot in &update.written_cell_key_slots {
                let tile_bytes = mirror.cell_key_slot_bytes(slot);
                write_cell_key_atlas_slot(
                    queue,
                    &self.cell_key_texture,
                    &tile_bytes,
                    cell_key_geometry.brick_edge_voxels,
                    cell_key_geometry.bricks_per_axis,
                    slot,
                );
            }
        }
        let atlas_view = self
            .atlas_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.rebuild_field_state(
            device,
            &atlas_view,
            mirror.records(),
            geometry.brick_edge_voxels,
            geometry.bricks_per_axis,
            gpu_records,
            pyramid,
            recentre_voxels,
        );
    }

    /// The number of atlas slots the last install / patch wrote (ADR 0011 G3 instrument):
    /// a wholesale install writes every sculpted slot; an incremental patch writes only
    /// the dirty region's slots (or, on a grow, every slot). The "per-edit cost ∝ dirty
    /// region" claim, made observable.
    pub fn last_atlas_slots_written(&self) -> u32 {
        self.last_atlas_slots_written
    }

    /// Re-upload the records + clip-map levels and rebuild the field bind group over
    /// `atlas_view`, then set the frame scalars — the shared tail of
    /// [`install_brick_field`](Self::install_brick_field) and
    /// [`patch_brick_field`](Self::patch_brick_field). Atlas texture management is the
    /// caller's (wholesale re-create vs per-slot patch); everything else is identical.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_field_state(
        &mut self,
        device: &wgpu::Device,
        atlas_view: &wgpu::TextureView,
        records: &[BrickRecord],
        brick_edge_voxels: u32,
        bricks_per_axis: u32,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: RecentreVoxels,
    ) {
        // Inclusive absolute block bounds over the record set (the sort is z-major,
        // so x/y still need the full scan; records are few — thousands).
        let mut absolute_block_bounds: Option<([i64; 3], [i64; 3])> = None;
        for record in records {
            let block = unpack_world_block_key(record.packed_world_block_key);
            absolute_block_bounds = Some(match absolute_block_bounds {
                None => (block, block),
                Some((lo, hi)) => (
                    [lo[0].min(block[0]), lo[1].min(block[1]), lo[2].min(block[2])],
                    [hi[0].max(block[0]), hi[1].max(block[1]), hi[2].max(block[2])],
                ),
            });
        }

        let placeholder = [BrickGpuRecord::zeroed()];
        let record_bytes: &[u8] = if gpu_records.is_empty() {
            bytemuck::cast_slice(&placeholder)
        } else {
            bytemuck::cast_slice(gpu_records)
        };
        let record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch records"),
            contents: record_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // The clip-map levels: split cell keys → (hi, lo) storage buffers. An empty
        // level uploads a single zeroed placeholder (its count is 0, so the shader
        // never reads it — that is the "pyramid off" install the A/B parity uses).
        let placeholder_keys = [[0u32, 0u32]];
        let level_1_keys = pack_clipmap_level_keys(&pyramid.level_1);
        let level_2_keys = pack_clipmap_level_keys(&pyramid.level_2);
        let level_3_keys = pack_clipmap_level_keys(&pyramid.level_3);
        let level_1_bytes: &[u8] = if level_1_keys.is_empty() {
            bytemuck::cast_slice(&placeholder_keys)
        } else {
            bytemuck::cast_slice(&level_1_keys)
        };
        let level_2_bytes: &[u8] = if level_2_keys.is_empty() {
            bytemuck::cast_slice(&placeholder_keys)
        } else {
            bytemuck::cast_slice(&level_2_keys)
        };
        let level_3_bytes: &[u8] = if level_3_keys.is_empty() {
            bytemuck::cast_slice(&placeholder_keys)
        } else {
            bytemuck::cast_slice(&level_3_keys)
        };
        let level_1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L1 keys"),
            contents: level_1_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let level_2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L2 keys"),
            contents: level_2_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let level_3_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch clip-map L3 keys"),
            contents: level_3_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ADR 0011 band-clip interior fallback: the block-occupancy cells (empty ⇒ a single
        // zeroed placeholder; its count is 0, so the shader never binary-searches it).
        let placeholder_occupancy = [OccupancyCellPod::zeroed()];
        let occupancy_cells = pack_occupancy_cells(&pyramid.interior_masks);
        let occupancy_bytes: &[u8] = if occupancy_cells.is_empty() {
            bytemuck::cast_slice(&placeholder_occupancy)
        } else {
            bytemuck::cast_slice(&occupancy_cells)
        };
        let occupancy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch block-occupancy cells"),
            contents: occupancy_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // The side atlas's view: its texture is managed by the caller (wholesale re-create vs
        // per-slot patch), exactly as the occupancy atlas's is — this only re-views it so the
        // rebuilt bind group points at the current texture.
        let cell_key_view = self
            .cell_key_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.field_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick raymarch field bind group"),
            layout: &self.field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    // ADR 0012 (H1): sized to ONE slot (dynamic offset selects solid /
                    // ghost-lower / ghost-upper), so `offset + size` is valid at every slot.
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform_buffer,
                        offset: 0,
                        size: std::num::NonZeroU64::new(
                            std::mem::size_of::<BrickUniformsPod>() as u64,
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: record_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: level_1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: level_2_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: level_3_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&cell_key_view),
                },
            ],
        });
        self.occupancy_cell_count = occupancy_cells.len() as u32;
        self.clipmap_level_1_blocks = pyramid.level_1.blocks_per_cell;
        self.clipmap_level_1_count = level_1_keys.len() as u32;
        self.clipmap_level_2_blocks = pyramid.level_2.blocks_per_cell;
        self.clipmap_level_2_count = level_2_keys.len() as u32;
        self.clipmap_level_3_blocks = pyramid.level_3.blocks_per_cell;
        self.clipmap_level_3_count = level_3_keys.len() as u32;
        self.record_count = gpu_records.len() as u32;
        self.recentre_voxels = recentre_voxels;
        self.brick_edge_voxels = brick_edge_voxels;
        self.bricks_per_axis = bricks_per_axis;
        self.absolute_block_bounds = absolute_block_bounds;
    }

    /// Drop the installed brick field (disengage — `draw` no-ops again). The
    /// pipelines and material atlas stay; the next install re-engages.
    pub fn clear_brick_field(&mut self) {
        self.record_count = 0;
        self.absolute_block_bounds = None;
        self.clipmap_level_1_count = 0;
        self.clipmap_level_2_count = 0;
        self.clipmap_level_3_count = 0;
        self.occupancy_cell_count = 0;
    }

    /// Whether a non-empty brick field is installed (the draw would show bricks).
    pub fn has_brick_field(&self) -> bool {
        self.record_count > 0
    }

    /// The resident record count (0 = nothing to march; `draw` is then a no-op).
    pub fn record_count(&self) -> u32 {
        self.record_count
    }

    /// Compute this frame's march frame (the uniform values) WITHOUT uploading —
    /// the shared math behind [`update_uniforms`](Self::update_uniforms) and the
    /// CPU reference march.
    pub fn march_frame(
        &self,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        grid_dimensions: [u32; 3],
        band: LayerBand,
    ) -> BrickMarchFrame {
        let edge = self.brick_edge_voxels.max(1) as i64;
        // Corner-anchoring: the cuboid path recovers the shading-absolute frame
        // with the FLOORED half (integer-valued), so mirror it exactly.
        let half = [
            (grid_dimensions[0] / 2) as i64,
            (grid_dimensions[1] / 2) as i64,
            (grid_dimensions[2] / 2) as i64,
        ];
        // absolute voxel = shading-absolute p + S, with S = recentre − half. Unwrap the
        // carried frame to its raw triple exactly here — the one uniform-packing consumption.
        let recentre = self.recentre_voxels.voxels();
        let shading_to_absolute = [
            recentre[0] - half[0],
            recentre[1] - half[1],
            recentre[2] - half[2],
        ];
        let mut lattice_shift = [0i32; 3];
        let mut voxel_bias = [0i32; 3];
        let mut block_bias = [0i32; 3];
        for axis in 0..3 {
            let shift = shading_to_absolute[axis].rem_euclid(edge);
            let bias = shading_to_absolute[axis] - shift;
            lattice_shift[axis] = i32::try_from(shift).expect("lattice shift fits i32");
            voxel_bias[axis] = i32::try_from(bias).expect("voxel bias fits i32");
            block_bias[axis] = i32::try_from(bias / edge).expect("block bias fits i32");
        }

        // The traversal AABB: the resident blocks' bounds in the shifted frame
        // (sv voxel = absolute voxel − voxel_bias), intersected with the band slab.
        let (mut traversal_lo, mut traversal_hi) = match self.absolute_block_bounds {
            Some((lo, hi)) => (
                glam::Vec3::new(
                    (lo[0] * edge - voxel_bias[0] as i64) as f32,
                    (lo[1] * edge - voxel_bias[1] as i64) as f32,
                    (lo[2] * edge - voxel_bias[2] as i64) as f32,
                ),
                glam::Vec3::new(
                    ((hi[0] + 1) * edge - voxel_bias[0] as i64) as f32,
                    ((hi[1] + 1) * edge - voxel_bias[1] as i64) as f32,
                    ((hi[2] + 1) * edge - voxel_bias[2] as i64) as f32,
                ),
            ),
            // No records: an empty AABB — every ray misses.
            None => (glam::Vec3::ZERO, glam::Vec3::ZERO),
        };
        // The band, converted voxel-Z layer indices → shifted-frame Z. A layer
        // index b is shading-absolute p ∈ [b, b+1), so sv ∈ [b + shift, b+1+shift).
        // Clamp the i64 math into i32 (LayerBand::FULL uses band_max = u32::MAX).
        let clamp_i32 = |value: i64| value.clamp(i32::MIN as i64 + 1, i32::MAX as i64 - 1) as i32;
        let band_lo_sv = clamp_i32(band.band_min as i64 + lattice_shift[2] as i64);
        let band_hi_sv = clamp_i32(band.band_max as i64 + 1 + lattice_shift[2] as i64);
        // The band ACTUALLY clips the solid when it narrows the resident Z-extent — only then
        // can a cut plane enter an elided coarse interior, so only then does the record-miss
        // block-occupancy fallback fire (ADR 0011 band-clip interior fix). A full/loose band
        // leaves the surface-only set hit-identical, so the fallback stays off (common path).
        let pre_band_lo_z = traversal_lo.z;
        let pre_band_hi_z = traversal_hi.z;
        traversal_lo.z = traversal_lo.z.max(band_lo_sv as f32);
        traversal_hi.z = traversal_hi.z.min(band_hi_sv as f32);
        let band_clip_active =
            traversal_lo.z > pre_band_lo_z || traversal_hi.z < pre_band_hi_z;

        BrickMarchFrame {
            view_projection,
            inverse_view_projection: view_projection.inverse(),
            viewport: [
                viewport_px[0] as f32,
                viewport_px[1] as f32,
                viewport_px[2] as f32,
                viewport_px[3] as f32,
            ],
            grid_half_extent: glam::Vec3::new(half[0] as f32, half[1] as f32, half[2] as f32),
            lattice_shift,
            block_bias,
            voxel_bias,
            band_voxel_sv: [band_lo_sv, band_hi_sv],
            band_clip_active,
            traversal_lo,
            traversal_hi,
            brick_edge_voxels: self.brick_edge_voxels.max(1) as i32,
            bricks_per_axis: self.bricks_per_axis.max(1),
        }
    }

    /// Upload this frame's uniforms (camera, viewport, band, overlay + material
    /// shading), mirroring `CuboidMeshRenderer::update_uniforms`' shading inputs so
    /// the two paths render pixel-comparable. Returns the frame for the CPU
    /// reference march (the parity harness).
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        grid_dimensions: [u32; 3],
        band: LayerBand,
        grid_overlay_master: bool,
        bound: Option<MaterialChoice>,
    ) -> BrickMarchFrame {
        let frame = self.march_frame(view_projection, viewport_px, grid_dimensions, band);
        // The bound procedural material drives modulation exactly as the cuboid
        // path: `Some` enables the relative base-colour array, `None` (a loaded VS
        // block — the brick path disengages for those, but mirror anyway) is neutral.
        let (modulation_enabled, base_colors) = match bound {
            Some(material) => (
                1.0,
                crate::renderer::relative_material_base_colors_public(material),
            ),
            None => (0.0, [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT]),
        };
        // The uniform is now the MASTER toggle only (the user's grid-overlay switch). Whether a
        // given hit draws the grid is `master AND the hit's own per-record/per-voxel overlay bit`,
        // resolved in the shader — the scene-wide overlay bool the representability gate carried
        // is deleted (blocks may disagree on the overlay and still be one brick field).
        let grid_overlay_enabled = if grid_overlay_master { 1.0 } else { 0.0 };
        // The SOLID draw's uniform: slot 0, `ghost_mode = 0` (its zeroed tint is unread).
        // Dynamic offset 0 selects it, so this is byte-identical to the pre-0012 single-slot
        // buffer (parity + non-onion goldens unaffected).
        let uniforms = self.build_uniforms_pod(
            &frame,
            grid_overlay_enabled,
            modulation_enabled,
            base_colors,
            0,
            [0.0; 4],
        );
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        frame
    }

    /// Assemble a [`BrickUniformsPod`] for one draw (ADR 0012 H1: shared by the solid draw
    /// and the two ghost-slab draws). `ghost_mode`/`ghost_tint` select the flat translucent
    /// ghost shade; every other field is the frame + shading the shader consumes.
    #[allow(clippy::too_many_arguments)]
    fn build_uniforms_pod(
        &self,
        frame: &BrickMarchFrame,
        grid_overlay_enabled: f32,
        modulation_enabled: f32,
        base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
        ghost_mode: u32,
        ghost_tint: [f32; 4],
    ) -> BrickUniformsPod {
        let material_atlas = crate::texture_atlas::MaterialAtlas::from_procedural_materials();
        let overlay = crate::renderer::grid_overlay_params();
        BrickUniformsPod {
            view_projection: frame.view_projection.to_cols_array_2d(),
            inverse_view_projection: frame.inverse_view_projection.to_cols_array_2d(),
            viewport: frame.viewport,
            grid_half_extent: frame.grid_half_extent.to_array(),
            voxels_per_block: self.brick_edge_voxels.max(1) as f32,
            voxel_line_color: overlay.voxel_line_color,
            grid_overlay_enabled,
            block_line_color: overlay.block_line_color,
            material_modulation_enabled: modulation_enabled,
            voxel_line_half_width: overlay.voxel_line_half_width,
            block_line_half_width: overlay.block_line_half_width,
            voxel_line_alpha: overlay.voxel_line_alpha,
            block_line_alpha: overlay.block_line_alpha,
            record_count: self.record_count,
            band_clip_active: u32::from(frame.band_clip_active),
            occupancy_cell_count: self.occupancy_cell_count,
            ghost_mode,
            lattice_shift_and_edge: [
                frame.lattice_shift[0],
                frame.lattice_shift[1],
                frame.lattice_shift[2],
                frame.brick_edge_voxels,
            ],
            block_bias_and_tiles: [
                frame.block_bias[0],
                frame.block_bias[1],
                frame.block_bias[2],
                frame.bricks_per_axis as i32,
            ],
            voxel_bias: [
                frame.voxel_bias[0],
                frame.voxel_bias[1],
                frame.voxel_bias[2],
                // w = loaded_material_active (ADR 0011 G2): shade solid hits from the
                // loaded 6-layer D2Array by the lattice rule instead of the procedural
                // atlas. The ghost draws pass this too but never shade (ghost_mode short-
                // circuits before `shade_cuboid_surface`), so it is inert for them.
                i32::from(self.loaded_material_active),
            ],
            band_voxel_sv: [
                frame.band_voxel_sv[0],
                frame.band_voxel_sv[1],
                // ADR 0013: the MATERIAL SIDE ATLAS's tiles-per-axis (its own pool sizes from
                // its mixed-brick slot count = dim / edge), so `mixed_voxel_material` addresses
                // the cell-key cube — never the occupancy atlas's `block_bias_and_tiles.w`. 1 for
                // the placeholder / a no-mixed-brick field (its cell-key sample never fires).
                (self.cell_key_texture_dim / self.brick_edge_voxels.max(1)).max(1) as i32,
                0,
            ],
            clipmap_blocks_and_counts: [
                self.clipmap_level_1_blocks.max(1),
                self.clipmap_level_1_count,
                self.clipmap_level_2_blocks.max(1),
                self.clipmap_level_2_count,
            ],
            clipmap_blocks_and_counts_hi: [
                self.clipmap_level_3_blocks.max(1),
                self.clipmap_level_3_count,
                0,
                0,
            ],
            traversal_lo: frame.traversal_lo.extend(0.0).to_array(),
            traversal_hi: frame.traversal_hi.extend(0.0).to_array(),
            material_base_colors: base_colors,
            material_atlas_rects: crate::cuboid_mesh::atlas_rects_from(&material_atlas),
            ghost_tint,
        }
    }

    /// (ADR 0012 H1) Upload the two onion GHOST slab uniforms (slots 1 + 2) for `band`.
    /// Each slab is the SAME march as the solid but with its band clamped to ONE onion
    /// slab — `[band_min − depth, band_min)` (lower) and `(band_max, band_max + depth]`
    /// (upper), the recentred-Z remainder of `AppCore::onion_fog_params`' onion span — plus
    /// `ghost_mode = 1` + the flat tint. The traversal-AABB clamp `march_frame` applies for
    /// the slab band IS the onion clip (so a slab draw hits only its slab's voxels, capped at
    /// the slab edges exactly as the mesh ghost's per-slab geometry, and `band_clip_active`
    /// re-fires the elided-interior occupancy fallback). Records the per-slab active flags
    /// [`draw_ghost`](Self::draw_ghost) reads; a degenerate slab (no layers that side of the
    /// band, or no field installed) is left inactive. Call AFTER
    /// [`update_uniforms`](Self::update_uniforms) each frame onion skin is on.
    pub fn update_ghost_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        viewport_px: [u32; 4],
        grid_dimensions: [u32; 3],
        band: LayerBand,
    ) {
        self.ghost_lower_active = false;
        self.ghost_upper_active = false;
        if self.record_count == 0 || band.onion_depth == 0 {
            return;
        }
        let tint = crate::renderer::onion_ghost_tint();
        let neutral = [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT];
        let depth = band.onion_depth;
        let last_layer = grid_dimensions[2].saturating_sub(1);
        // Lower slab: layers [band_min − depth, band_min − 1]; skipped when the band bottom
        // is already layer 0 (nothing below to ghost).
        if band.band_min > 0 {
            let slab = LayerBand {
                band_min: band.band_min.saturating_sub(depth),
                band_max: band.band_min - 1,
                onion_depth: 0,
            };
            let frame = self.march_frame(view_projection, viewport_px, grid_dimensions, slab);
            let pod = self.build_uniforms_pod(&frame, 0.0, 0.0, neutral, 1, tint);
            queue.write_buffer(
                &self.uniform_buffer,
                self.slot_offset(BRICK_UNIFORM_SLOT_GHOST_LOWER),
                bytemuck::bytes_of(&pod),
            );
            self.ghost_lower_active = true;
        }
        // Upper slab: layers [band_max + 1, band_max + depth]; skipped when the band top is
        // already the last layer (nothing above to ghost).
        if band.band_max < last_layer {
            let slab = LayerBand {
                band_min: band.band_max + 1,
                band_max: (band.band_max + depth).min(last_layer),
                onion_depth: 0,
            };
            let frame = self.march_frame(view_projection, viewport_px, grid_dimensions, slab);
            let pod = self.build_uniforms_pod(&frame, 0.0, 0.0, neutral, 1, tint);
            queue.write_buffer(
                &self.uniform_buffer,
                self.slot_offset(BRICK_UNIFORM_SLOT_GHOST_UPPER),
                bytemuck::bytes_of(&pod),
            );
            self.ghost_upper_active = true;
        }
    }

    /// The byte offset of dynamic-offset uniform `slot` (ADR 0012 H1).
    fn slot_offset(&self, slot: u32) -> u64 {
        slot as u64 * self.uniform_slot_stride as u64
    }

    /// Set whether a VS block is applied this frame — mirrored into `voxel_bias.w` by the
    /// next [`update_uniforms`](Self::update_uniforms) so the shader shades solid hits from
    /// the loaded 6-layer D2Array (the owner's lattice rule) instead of the procedural
    /// atlas (ADR 0011 G2). Call BEFORE `update_uniforms`; pass the SAME block's bind group
    /// to [`draw`](Self::draw). A no-op state change when it matches the current value.
    pub fn set_loaded_material_active(&mut self, active: bool) {
        self.loaded_material_active = active;
    }

    /// Draw the brick raymarch INSIDE the shared MSAA voxel pass (viewport +
    /// scissor already set by `render_frame`). Uniforms must be uploaded first.
    /// `loaded_material` is the applied VS block's group(2) bind group (built against
    /// `renderer::build_face_material_layout`, ADR 0011 G2); `None` binds the dummy —
    /// pass `Some(..)` exactly when [`set_loaded_material_active(true)`](Self::set_loaded_material_active)
    /// was set this frame so the sampled texture matches the shading branch.
    pub fn draw<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        loaded_material: Option<&'a wgpu::BindGroup>,
    ) {
        if self.record_count == 0 {
            return;
        }
        pass.set_pipeline(&self.render_pipeline);
        // ADR 0012 (H1): dynamic offset selects the SOLID uniform slot.
        pass.set_bind_group(
            0,
            &self.field_bind_group,
            &[self.slot_offset(BRICK_UNIFORM_SLOT_SOLID) as u32],
        );
        pass.set_bind_group(1, &self.material_bind_group, &[]);
        pass.set_bind_group(
            2,
            loaded_material.unwrap_or(&self.dummy_loaded_material_bind_group),
            &[],
        );
        pass.draw(0..3, 0..1);
    }

    /// (ADR 0012 H1) Draw the onion GHOST pass: one fullscreen raymarch per ACTIVE onion
    /// slab (lower then upper — the same order the cuboid mesh ghost draws), each selecting
    /// its ghost uniform slot by dynamic offset. Flat-tinted + alpha-blended, depth test
    /// `Less` with depth WRITE ON (nearest ghost surface wins). MUST run AFTER [`draw`](Self::draw)
    /// inside the same MSAA pass; `update_ghost_uniforms` must have prepared the slots. A
    /// no-op when no field is installed or neither slab is active.
    pub fn draw_ghost<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.record_count == 0 {
            return;
        }
        pass.set_pipeline(&self.ghost_render_pipeline);
        pass.set_bind_group(1, &self.material_bind_group, &[]);
        // The ghost is flat-tinted (never samples a material) but the 3-group pipeline
        // layout still requires group(2) bound — the dummy loaded material suffices.
        pass.set_bind_group(2, &self.dummy_loaded_material_bind_group, &[]);
        if self.ghost_lower_active {
            pass.set_bind_group(
                0,
                &self.field_bind_group,
                &[self.slot_offset(BRICK_UNIFORM_SLOT_GHOST_LOWER) as u32],
            );
            pass.draw(0..3, 0..1);
        }
        if self.ghost_upper_active {
            pass.set_bind_group(
                0,
                &self.field_bind_group,
                &[self.slot_offset(BRICK_UNIFORM_SLOT_GHOST_UPPER) as u32],
            );
            pass.draw(0..3, 0..1);
        }
    }

    /// Render the hit-identity image (the parity harness): one `[hit, x, y, z]`
    /// u32 quad per pixel, hit voxel coordinates in ABSOLUTE voxels (i32 bitcast).
    /// Uses the CURRENT uniforms — call [`update_uniforms`](Self::update_uniforms)
    /// with `viewport_px = [0, 0, width, height]` first.
    pub fn render_hit_identity_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Vec<[u32; 4]> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brick hit-identity target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let bytes_per_pixel = 16u32;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick hit-identity readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brick hit-identity pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.hit_identity_pipeline);
            // ADR 0012 (H1): the parity harness reads the SOLID slot.
            pass.set_bind_group(
                0,
                &self.field_bind_group,
                &[self.slot_offset(BRICK_UNIFORM_SLOT_SOLID) as u32],
            );
            pass.set_bind_group(1, &self.material_bind_group, &[]);
            // Hit-identity never samples a material; the dummy satisfies group(2).
            pass.set_bind_group(2, &self.dummy_loaded_material_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        receiver
            .recv()
            .expect("map_async channel dropped")
            .expect("buffer map failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let row_start = (row * padded_row) as usize;
            let row_words: &[u32] =
                bytemuck::cast_slice(&mapped[row_start..row_start + unpadded_row as usize]);
            for pixel in row_words.chunks_exact(4) {
                pixels.push([pixel[0], pixel[1], pixel[2], pixel[3]]);
            }
        }
        drop(mapped);
        readback.unmap();
        pixels
    }

    /// ADR 0013 — render the MATERIAL-identity image (the mixed-brick parity harness): one
    /// `[hit, material_id, 0, 0]` u32 quad per pixel, where `material_id` is the RESOLVED
    /// per-voxel material (a mixed brick's clean cell-key id, else the per-record material).
    /// Uses the CURRENT uniforms — call [`update_uniforms`](Self::update_uniforms) with
    /// `viewport_px = [0, 0, width, height]` first. The direct "shader == CPU-march reference"
    /// material gate (ADR 0013): no shading is reproduced, only the resolved id compared.
    pub fn render_material_identity_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Vec<[u32; 4]> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brick material-identity target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let bytes_per_pixel = 16u32;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick material-identity readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brick material-identity pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.material_identity_pipeline);
            pass.set_bind_group(
                0,
                &self.field_bind_group,
                &[self.slot_offset(BRICK_UNIFORM_SLOT_SOLID) as u32],
            );
            pass.set_bind_group(1, &self.material_bind_group, &[]);
            // Material-identity never samples a material texture; the dummy satisfies group(2).
            pass.set_bind_group(2, &self.dummy_loaded_material_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        receiver
            .recv()
            .expect("map_async channel dropped")
            .expect("buffer map failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let row_start = (row * padded_row) as usize;
            let row_words: &[u32] =
                bytemuck::cast_slice(&mapped[row_start..row_start + unpadded_row as usize]);
            for pixel in row_words.chunks_exact(4) {
                pixels.push([pixel[0], pixel[1], pixel[2], pixel[3]]);
            }
        }
        drop(mapped);
        readback.unmap();
        pixels
    }

    /// ADR 0011 G2 — render the SHADED colour image (the colour-parity harness): one
    /// `Rgba8Unorm` pixel per hit, shaded exactly as the MSAA render pass' centre-ray
    /// evaluation. `loaded_material` binds the applied block's group(2) D2Array (call
    /// [`set_loaded_material_active(true)`](Self::set_loaded_material_active) +
    /// `update_uniforms` first so the shading branch matches); `None` binds the dummy.
    /// Non-hit pixels are the cleared background. Used ONLY by tests/gpu_parity.rs.
    pub fn render_color_image(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        loaded_material: Option<&wgpu::BindGroup>,
    ) -> Vec<[u8; 4]> {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("brick colour-identity target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let bytes_per_pixel = 4u32;
        let unpadded_row = width * bytes_per_pixel;
        let padded_row = unpadded_row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick colour-identity readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("brick colour-identity pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.color_identity_pipeline);
            pass.set_bind_group(
                0,
                &self.field_bind_group,
                &[self.slot_offset(BRICK_UNIFORM_SLOT_SOLID) as u32],
            );
            pass.set_bind_group(1, &self.material_bind_group, &[]);
            pass.set_bind_group(
                2,
                loaded_material.unwrap_or(&self.dummy_loaded_material_bind_group),
                &[],
            );
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll failed");
        receiver
            .recv()
            .expect("map_async channel dropped")
            .expect("buffer map failed");

        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height) as usize);
        for row in 0..height {
            let row_start = (row * padded_row) as usize;
            let row_bytes = &mapped[row_start..row_start + unpadded_row as usize];
            for pixel in row_bytes.chunks_exact(4) {
                pixels.push([pixel[0], pixel[1], pixel[2], pixel[3]]);
            }
        }
        drop(mapped);
        readback.unmap();
        pixels
    }
}

// ============================================================================
// CPU reference march — the f32 mirror of the WGSL traversal (the parity net's
// oracle side; never on a live path).
// ============================================================================

/// A CPU march hit: the hit voxel in ABSOLUTE voxel coordinates (the exact
/// evaluator's frame), plus the entered face's outward normal as an exact ±1 axis
/// vector (`[i32; 3]`, so `Eq` still derives). The normal drives the loaded-material
/// shading rule (`face_layer`) the colour-parity test cross-checks (ADR 0011 G2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMarchHit {
    pub absolute_voxel: [i32; 3],
    pub face_normal: [i32; 3],
}

/// The pixel-centre camera ray in the shifted march frame — mirrors `camera_ray`.
fn cpu_camera_ray(frame: &BrickMarchFrame, pixel: glam::Vec2) -> (glam::Vec3, glam::Vec3) {
    let ndc_x = (pixel.x - frame.viewport[0]) / frame.viewport[2] * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - frame.viewport[1]) / frame.viewport[3] * 2.0;
    let near_h = frame.inverse_view_projection * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = frame.inverse_view_projection * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_h.truncate() / near_h.w;
    let far_world = far_h.truncate() / far_h.w;
    let direction = (far_world - near_world).normalize();
    let shift = glam::Vec3::new(
        frame.lattice_shift[0] as f32,
        frame.lattice_shift[1] as f32,
        frame.lattice_shift[2] as f32,
    );
    (near_world + frame.grid_half_extent + shift, direction)
}

/// Is a sculpted brick's block-local voxel occupied in the build's atlas bytes?
fn cpu_sculpted_voxel_occupied(
    build: &BrickFieldBuild,
    atlas_slot: u32,
    brick_local: [i32; 3],
) -> bool {
    let tiles = build.bricks_per_axis.max(1);
    let edge = build.brick_edge_voxels.max(1) as usize;
    let atlas_dim = build.atlas_dim_voxels as usize;
    let tile = [
        (atlas_slot % tiles) as usize,
        ((atlas_slot / tiles) % tiles) as usize,
        (atlas_slot / (tiles * tiles)) as usize,
    ];
    let coord = [
        tile[0] * edge + brick_local[0] as usize,
        tile[1] * edge + brick_local[1] as usize,
        tile[2] * edge + brick_local[2] as usize,
    ];
    build.sculpted_atlas_bytes[(coord[2] * atlas_dim + coord[1]) * atlas_dim + coord[0]] > 127
}

/// Binary-search the packed GPU records for a split key — mirrors the shader.
fn cpu_find_brick_record(records: &[BrickGpuRecord], key_hi: u32, key_lo: u32) -> Option<usize> {
    let key = ((key_hi as u64) << 32) | key_lo as u64;
    records
        .binary_search_by_key(&key, |record| {
            ((record.key_hi as u64) << 32) | record.key_lo as u64
        })
        .ok()
}

/// The split (hi, lo) key of an absolute block — mirrors the shader's packing.
fn cpu_pack_key_split(absolute_block: [i32; 3]) -> (u32, u32) {
    const BIAS: i32 = 1 << 20;
    let biased_x = (absolute_block[0] + BIAS) as u32;
    let biased_y = (absolute_block[1] + BIAS) as u32;
    let biased_z = (absolute_block[2] + BIAS) as u32;
    (
        (biased_z << 10) | (biased_y >> 11),
        ((biased_y & 0x7ff) << 21) | biased_x,
    )
}

/// Is the clip-map cell containing `absolute_block` occupied — or the level OFF
/// (empty ⇒ no hierarchical skip, the flat G1 DDA)? Mirrors the shader's
/// `clipmap_cell_occupied`: floor-div the absolute block into the cell lattice,
/// pack the cell key, binary-search the sorted level.
fn cpu_clipmap_cell_occupied(level: &ClipmapLevel, absolute_block: glam::IVec3) -> bool {
    // Domain policy: a level with NO keys is "off" — never skip, so report every cell occupied
    // (the flat G1 DDA). This "empty ⇒ occupied" reading is the domain's, not the kernel's; the
    // pure fold+binary-search below is substrate's `sorted_cell_keys_contain`.
    if level.cell_keys.is_empty() {
        return true;
    }
    substrate::spatial::min_mip_pyramid::sorted_cell_keys_contain(
        &level.cell_keys,
        [
            absolute_block.x as i64,
            absolute_block.y as i64,
            absolute_block.z as i64,
        ],
        level.blocks_per_cell,
    )
}

/// March one pixel-centre ray through the brick field on the CPU — a step-for-step
/// f32 mirror of the WGSL `march_brick_field` (same op order, same tie-breaks, same
/// clamped boxes, residency-miss branch, and G2 hierarchical clip-map skip),
/// returning the hit voxel in absolute coordinates. The parity net asserts the GPU
/// hit-identity image equals this. `pyramid` with empty levels is the "pyramid off"
/// form (the flat block-DDA) — the A/B baseline the pyramid-on == off parity uses.
pub fn cpu_march_brick_field(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    pyramid: &ClipmapPyramid,
    pixel: glam::Vec2,
) -> Option<CpuMarchHit> {
    cpu_march_brick_field_counted(frame, records, build, pyramid, pixel).0
}

/// [`cpu_march_brick_field`] plus the number of block-DDA loop iterations the ray
/// took (each iteration is one hierarchical jump OR one per-block step) — the
/// empty-space-skip metric the scattered-scene perf probe reports pyramid on vs off.
pub fn cpu_march_brick_field_counted(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    pyramid: &ClipmapPyramid,
    pixel: glam::Vec2,
) -> (Option<CpuMarchHit>, u32) {
    cpu_march_levels_counted(
        frame,
        records,
        build,
        &pyramid.levels_coarse_to_fine(),
        pixel,
    )
}

/// The core hierarchical-DDA CPU march, generalized over an arbitrary set of
/// clip-map levels ordered COARSEST → FINEST (the shader's else-if descent, as a
/// loop). `cpu_march_brick_field_counted` passes the production pyramid's three
/// levels; the perf probe passes custom level sets (L2-only, +L3, +L4) to measure
/// each configuration's block-steps/ray honestly. An empty level (off) is skipped
/// over. Returns the hit voxel (absolute) plus the block-DDA iteration count.
pub fn cpu_march_levels_counted(
    frame: &BrickMarchFrame,
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    levels_coarse_to_fine: &[&ClipmapLevel],
    pixel: glam::Vec2,
) -> (Option<CpuMarchHit>, u32) {
    // The pure hierarchical march lives in `raycast::march_brick_hierarchy` (the WGSL's
    // GPU-mirror specification). This function is the domain ADAPTER (ADR 0008 carried
    // frame, docs/architecture/03-display.md): it derives the ray from the shifted frame,
    // packs the frame's plain numerics into the kernel's params, and builds the three
    // injected occupancy closures from the records/atlas/clip-map. The kernel's `MarchHit`
    // maps 1:1 onto `CpuMarchHit`.
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let params = raycast::HierarchicalMarchParams {
        traversal_lo: frame.traversal_lo,
        traversal_hi: frame.traversal_hi,
        brick_edge_voxels: frame.brick_edge_voxels,
        block_bias: glam::IVec3::from_array(frame.block_bias),
        voxel_bias: frame.voxel_bias,
        band_voxel_sv: frame.band_voxel_sv,
        level_blocks_per_cell: levels_coarse_to_fine
            .iter()
            .map(|level| level.blocks_per_cell as i32)
            .collect(),
    };
    let (hit, steps) = raycast::march_brick_hierarchy(
        substrate::spatial::Ray::new(origin, direction),
        &params,
        // Level-occupancy: the domain's "empty level ⇒ occupied (skip disabled)" policy
        // over substrate's sorted cell-key search.
        |level_index, absolute_block| {
            cpu_clipmap_cell_occupied(levels_coarse_to_fine[level_index], absolute_block)
        },
        // Per-block classification: the record binary search + the WGSL kind decode. A
        // sculpted block carries a closure over its atlas slot for the inner voxel DDA.
        |absolute_block| {
            let (key_hi, key_lo) = cpu_pack_key_split(absolute_block);
            match cpu_find_brick_record(records, key_hi, key_lo) {
                None => raycast::BlockContents::Empty,
                Some(record_index) => {
                    let record = records[record_index];
                    if record_is_coarse_form(&record) {
                        raycast::BlockContents::CoarseSolid
                    } else {
                        let atlas_slot = record.atlas_slot;
                        raycast::BlockContents::Sculpted(move |brick_local| {
                            cpu_sculpted_voxel_occupied(build, atlas_slot, brick_local)
                        })
                    }
                }
            }
        },
    );
    (
        hit.map(|hit| CpuMarchHit {
            absolute_voxel: hit.absolute_voxel,
            face_normal: hit.face_normal,
        }),
        steps,
    )
}

/// March one pixel-centre ray over the EXACT evaluator's occupancy — a plain
/// voxel-level DDA (no bricks, no records) inside the same frame/band, querying
/// `occupied(absolute_voxel)`. This is the parity net's INDEPENDENT content
/// oracle: the brick march's hit-voxel set must equal this march's hit-voxel set
/// (ADR 0011 parity gate clause (b)).
pub fn cpu_march_exact_occupancy(
    frame: &BrickMarchFrame,
    occupied: &dyn Fn([i64; 3]) -> bool,
    pixel: glam::Vec2,
) -> Option<CpuMarchHit> {
    // Domain adapter over `raycast::march_exact_occupancy` (the flat reference kernel):
    // derive the ray from the shifted frame, pass the band + biases, and forward the
    // absolute-voxel occupancy predicate unchanged. See docs/architecture/03-display.md.
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let params = raycast::ExactMarchParams {
        traversal_lo: frame.traversal_lo,
        traversal_hi: frame.traversal_hi,
        band_voxel_sv: frame.band_voxel_sv,
        voxel_bias: frame.voxel_bias,
    };
    raycast::march_exact_occupancy(substrate::spatial::Ray::new(origin, direction), &params, |absolute| {
        occupied(absolute)
    })
    .map(|hit| CpuMarchHit {
        absolute_voxel: hit.absolute_voxel,
        face_normal: hit.face_normal,
    })
}

/// The MATERIAL a brick hit shades from — the CPU-march reference for ADR 0013's per-voxel
/// mixed shading (`docs/architecture/03-display.md`, the brick-field atlas). For a MIXED brick
/// (kind 2 with a resident cell-key slot) it samples the SAME cell-key tile at the SAME hit
/// voxel the shader's `mixed_voxel_material` reads and returns its clean block id; for a coarse
/// or sculpted-UNIFORM block it returns the per-record material. `tests/gpu_parity.rs` asserts
/// [`BrickRaymarchRenderer::render_material_identity_image`] equals this at every agreeing pixel.
///
/// The material is a DOMAIN fact (a cell key, a palette id, an overlay bit); the `raycast` kernel
/// stays material-free, so this resolves off the returned [`CpuMarchHit::absolute_voxel`] — the
/// hit voxel and the carried march frame's `brick_edge_voxels` recover the block and the
/// brick-local voxel exactly (`voxel_bias` is a multiple of the brick edge, so absolute-voxel
/// `div`/`rem` edge give the absolute block and brick-local coordinate the record search + tile
/// sample need).
pub fn cpu_brick_hit_material(
    records: &[BrickGpuRecord],
    build: &BrickFieldBuild,
    brick_edge_voxels: i32,
    hit: CpuMarchHit,
) -> u32 {
    let edge = brick_edge_voxels.max(1);
    let absolute_block = [
        hit.absolute_voxel[0].div_euclid(edge),
        hit.absolute_voxel[1].div_euclid(edge),
        hit.absolute_voxel[2].div_euclid(edge),
    ];
    let brick_local = [
        hit.absolute_voxel[0].rem_euclid(edge) as u32,
        hit.absolute_voxel[1].rem_euclid(edge) as u32,
        hit.absolute_voxel[2].rem_euclid(edge) as u32,
    ];
    let (key_hi, key_lo) = cpu_pack_key_split(absolute_block);
    match cpu_find_brick_record(records, key_hi, key_lo) {
        None => 0,
        Some(index) => {
            let record = records[index];
            if record_kind_discriminant(record.kind) == 2
                && record.cell_key_slot != NON_RESIDENT_ATLAS_SLOT
            {
                // The mixed brick's per-voxel cell key, masked to its clean block id — the CPU
                // twin of the shader's `mixed_voxel_material` (same tile, same voxel, same mask).
                let cell_key = build.cell_key_tiles[record.cell_key_slot as usize].get(
                    brick_local[0],
                    brick_local[1],
                    brick_local[2],
                );
                CellKey::from_raw(cell_key).block_id() as u32
            } else {
                record_material_id(record.kind)
            }
        }
    }
}

#[cfg(test)]
mod record_packing_tests {
    //! The GPU record format — the byte-level contract `shaders/brick_raymarch.wgsl` decodes.
    //! Pinned here because the shader cannot assert: a silent desync (a kind discriminant, the
    //! material mask, the overlay bit, the field order) shows up only as wrong pixels.
    use super::*;
    use crate::brick_field::{pack_world_block_key, BrickPayload, BrickRecord};
    use crate::core_geom::BlockId;
    use crate::two_layer_store::SeamSolidity;

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
}

#[cfg(test)]
mod mixed_material_reference_tests {
    //! ADR 0013 — the CPU-march material reference ([`cpu_brick_hit_material`]) resolves a MIXED
    //! brick's per-voxel materials from its cell-key tile (the same tile + voxel the shader's
    //! `mixed_voxel_material` samples), masking the overlay bit off to the clean id; a UNIFORM hit
    //! resolves the per-record material. This is the CPU half of the shader == reference bar; the
    //! GPU half is `tests/gpu_parity.rs::brick_mixed_material_matches_cpu_reference`.
    use super::*;
    use crate::brick_field::build_brick_field;
    use crate::core_geom::CHUNK_BLOCKS;
    use crate::cuboid::VoxelBox;
    use crate::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
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
}
