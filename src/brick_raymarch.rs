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

use crate::brick_field::{
    pack_clipmap_level_keys, pack_world_block_key, unpack_world_block_key, upload_brick_atlas,
    BlockOccupancyMasks, BrickFieldBuild, BrickFieldUpdate, BrickPayload, BrickRecord,
    ClipmapLevel, ClipmapPyramid, IncrementalBrickField, SculptedAtlasPayload,
    BLOCK_OCCUPANCY_MASK_WORDS,
};
use crate::core_geom::MaterialChoice;
use crate::cuboid_mesh::{cell_key_has_overlay, clean_block_id};
use crate::renderer::{LayerBand, DEPTH_FORMAT, MSAA_SAMPLE_COUNT};
use crate::two_layer_store::TwoLayerChunk;

/// The sentinel marking a sculpted record whose atlas payload is NOT resident (the
/// residency-miss contract). Must match `NON_RESIDENT_ATLAS_SLOT` in the WGSL.
pub const NON_RESIDENT_ATLAS_SLOT: u32 = u32::MAX;

/// `BrickGpuRecord.kind` packs the record's block material-colour index in the bits
/// ABOVE the kind discriminant (ADR 0011 G2 per-record shading): bits `[0, SHIFT)`
/// hold the kind (0 coarse / 1 sculpted), bits `[SHIFT, 32)` the material id. One
/// `u32`, no struct-layout change — a multi-producer scene of distinct per-block
/// materials shades each hit from its own record. MUST match the decode in
/// `shaders/brick_raymarch.wgsl`.
pub const BRICK_RECORD_MATERIAL_ID_SHIFT: u32 = 8;

/// Mask isolating the kind discriminant below [`BRICK_RECORD_MATERIAL_ID_SHIFT`].
const BRICK_RECORD_KIND_MASK: u32 = (1 << BRICK_RECORD_MATERIAL_ID_SHIFT) - 1;

/// ADR 0012 (H1) — the dynamic-offset uniform slots the field bind group indexes. The
/// SINGLE uniform buffer holds three `BrickUniformsPod` slots (each aligned up to the
/// device's `min_uniform_buffer_offset_alignment`): the SOLID band draw, plus the LOWER
/// and UPPER onion GHOST slabs. One bind group, records/atlas/clip-map shared; only the
/// bound dynamic offset (and the shading uniforms it selects) differ per draw.
const BRICK_UNIFORM_SLOT_SOLID: u32 = 0;
const BRICK_UNIFORM_SLOT_GHOST_LOWER: u32 = 1;
const BRICK_UNIFORM_SLOT_GHOST_UPPER: u32 = 2;
const BRICK_UNIFORM_SLOT_COUNT: u64 = 3;

/// The kind discriminant (0 coarse / 1 sculpted) of a packed `BrickGpuRecord.kind` —
/// the mirror of the WGSL `record_kind(kind)`. The material id lives above it.
fn record_kind_discriminant(kind: u32) -> u32 {
    kind & BRICK_RECORD_KIND_MASK
}

/// One resident brick as the shader consumes it: the packed world-block key split
/// into a `(hi, lo)` u32 pair (sorted ascending — the in-shader binary search's
/// order), the record kind (0 coarse / 1 sculpted) and the atlas slot (or
/// [`NON_RESIDENT_ATLAS_SLOT`]).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Pod, Zeroable)]
pub struct BrickGpuRecord {
    pub key_hi: u32,
    pub key_lo: u32,
    pub kind: u32,
    pub atlas_slot: u32,
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
    let (kind_discriminant, atlas_slot) = match record.payload {
        BrickPayload::CoarseSolid { .. } => (0u32, 0u32),
        BrickPayload::Sculpted { atlas_slot } => (
            1u32,
            if non_resident(atlas_slot) {
                NON_RESIDENT_ATLAS_SLOT
            } else {
                atlas_slot
            },
        ),
    };
    // Pack the block material above the kind discriminant (ADR 0011 G2): the
    // shader shades the hit from its own record, not a scene-wide uniform.
    let kind = kind_discriminant | ((record.material_id as u32) << BRICK_RECORD_MATERIAL_ID_SHIFT);
    BrickGpuRecord {
        key_hi: (key >> 32) as u32,
        key_lo: key as u32,
        kind,
        atlas_slot,
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

/// Whether the boundary set is **brick-representable** (ADR 0011 G2), and, if so, the
/// scene-wide on-face-grid overlay state the shader binds. `Some(overlay)` engages the
/// brick path; `None` keeps the scene on the mesh path.
///
/// Representable ⇔ every non-air block is INTERNALLY single-cell (all its microblocks
/// share one clean material id + overlay state) AND the whole scene shares ONE overlay
/// state. Per-BLOCK materials may differ across blocks — [`pack_gpu_records`] packs each
/// block's material into its record (G2). The two limits are structural: the R8 atlas is
/// occupancy-only, so a block that MIXES materials across its microblocks can't be one
/// occupancy brick; and the overlay is a scene-wide uniform (not per-record), so a scene
/// whose blocks disagree on it can't be represented. Both fall back to the mesh path.
///
/// A single ported producer is trivially representable (uniform by construction) — the
/// G1 gate — so widening to this predicate keeps every G1 scene engaged and adds the
/// distinct-material multi-producer scenes.
pub fn brick_representable_overlay(
    two_layer_chunks: &[([i32; 3], std::sync::Arc<TwoLayerChunk>)],
) -> Option<bool> {
    // The scene-wide overlay: every rendered block must agree on it.
    let mut scene_overlay: Option<bool> = None;
    let mut fold_scene_overlay = |overlay: bool| -> bool {
        match scene_overlay {
            None => {
                scene_overlay = Some(overlay);
                true
            }
            Some(existing) => existing == overlay,
        }
    };
    for (_, chunk) in two_layer_chunks {
        // A coarse-solid block is single-material by construction; only its overlay
        // participates in the scene-wide agreement.
        for (index, coarse) in chunk.coarse.iter().enumerate() {
            if coarse.is_some() && !fold_scene_overlay(chunk.coarse_overlay[index]) {
                return None;
            }
        }
        // A boundary block must be internally single-cell (one material + overlay across
        // its microblocks), then its overlay folds into the scene-wide agreement.
        for geometry in chunk.microblocks.values() {
            let mut block_cell: Option<(u16, bool)> = None;
            for cuboid in &geometry.cuboids {
                let key = cuboid.material_id;
                let cell = (clean_block_id(key), cell_key_has_overlay(key));
                match block_cell {
                    None => block_cell = Some(cell),
                    Some(existing) if existing != cell => return None, // mixed within a block
                    Some(_) => {}
                }
            }
            if let Some((_, overlay)) = block_cell {
                if !fold_scene_overlay(overlay) {
                    return None;
                }
            }
        }
    }
    Some(scene_overlay.unwrap_or(false))
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
    _pad: u32,
    mask: [u32; BLOCK_OCCUPANCY_MASK_WORDS],
}

/// Pack the block-occupancy map into the shader's sorted cell records (the parallel SoA
/// `cell_keys`/`cell_masks`/`cell_materials` → AoS). Empty ⇒ a single zeroed placeholder (its
/// count is 0, so the shader never binary-searches it).
fn pack_occupancy_cells(masks: &BlockOccupancyMasks) -> Vec<OccupancyCellPod> {
    masks
        .cell_keys
        .iter()
        .zip(&masks.cell_masks)
        .zip(&masks.cell_materials)
        .map(|((&key, &mask), &material)| OccupancyCellPod {
            key_hi: (key >> 32) as u32,
            key_lo: key as u32,
            material,
            _pad: 0,
            mask,
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
    /// The number of atlas slots the LAST update wrote (ADR 0011 G3 "per-edit cost ∝ dirty
    /// region" instrument): a wholesale install writes every sculpted slot; an incremental
    /// patch writes only the dirty chunks' slots (unless the atlas grew — then every slot).
    last_atlas_slots_written: u32,
    record_count: u32,
    /// The scene-wide on-face-grid overlay state, derived from the boundary set at
    /// install (`brick_representable_overlay`). Material is per-record (ADR 0011 G2).
    overlay_active: bool,
    /// The composite recentre the boundary set was resolved under (ADR 0008 —
    /// carried from the install, the same value the two-layer mesher bakes).
    recentre_voxels: [i64; 3],
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

        Self {
            render_pipeline,
            ghost_render_pipeline,
            hit_identity_pipeline,
            color_identity_pipeline,
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
            last_atlas_slots_written: 0,
            record_count: 0,
            overlay_active: false,
            recentre_voxels: [0, 0, 0],
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
    /// recentre; `overlay_active` the scene-wide overlay state
    /// ([`brick_representable_overlay`]). Material is per-record (packed in
    /// `gpu_records`, ADR 0011 G2).
    #[allow(clippy::too_many_arguments)]
    pub fn install_brick_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        records: &[BrickRecord],
        atlas: &SculptedAtlasPayload,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: [i64; 3],
        overlay_active: bool,
    ) {
        // A wholesale install (re)creates the atlas texture from scratch and uploads
        // every sculpted slot — the from-scratch / scene-load / gate-re-engage path.
        let atlas_texture = upload_brick_atlas(device, queue, atlas);
        self.atlas_texture = atlas_texture;
        self.atlas_texture_dim = atlas.geometry.atlas_dim_voxels.max(1);
        self.last_atlas_slots_written = atlas.sculpted_slot_count;
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
            overlay_active,
        );
    }

    /// **ADR 0011 G3 — incremental dirty-brick patch.** Patch ONLY the dirty slots of the
    /// PERSISTENT atlas from an [`IncrementalBrickField`](crate::brick_field::IncrementalBrickField)
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
        recentre_voxels: [i64; 3],
        overlay_active: bool,
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
            overlay_active,
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
        recentre_voxels: [i64; 3],
        overlay_active: bool,
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
        self.overlay_active = overlay_active;
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
        // absolute voxel = shading-absolute p + S, with S = recentre − half.
        let shading_to_absolute = [
            self.recentre_voxels[0] - half[0],
            self.recentre_voxels[1] - half[1],
            self.recentre_voxels[2] - half[2],
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
        let grid_overlay_enabled = if grid_overlay_master && self.overlay_active {
            1.0
        } else {
            0.0
        };
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
            band_voxel_sv: [frame.band_voxel_sv[0], frame.band_voxel_sv[1], 0, 0],
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

/// The outward face normal (an exact ±1 axis vector) for a march that ENTERED a box
/// through face `axis`: the normal opposes the ray's motion on that axis (mirrors the
/// shader's `hit.normal_sign = -sign(ray.direction[axis])`). Feeds `face_layer` for the
/// loaded-material colour-parity check (ADR 0011 G2).
fn axis_normal(axis: usize, direction: glam::Vec3) -> [i32; 3] {
    let mut normal = [0i32; 3];
    normal[axis] = if direction[axis] > 0.0 { -1 } else { 1 };
    normal
}

fn safe_direction(direction: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::new(
        if direction.x.abs() < 1e-20 { 1e-20 } else { direction.x },
        if direction.y.abs() < 1e-20 { 1e-20 } else { direction.y },
        if direction.z.abs() < 1e-20 { 1e-20 } else { direction.z },
    )
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

/// The hair the hierarchical DDA steps PAST a coarse-cell exit face before
/// re-deriving the block cell — larger than the per-block `1e-4` so the jump
/// reliably lands in the next cell. MUST match `CLIPMAP_JUMP_EPSILON` in the WGSL.
const CLIPMAP_JUMP_EPSILON: f32 = 1e-3;

/// Block-DDA step budget — the CPU mirror of the shader's `MAX_BLOCK_STEPS`. The
/// pyramid collapses empty space to a handful of strides; this ceiling only bounds
/// the flat fallback (pyramid off) crossing a wide traversal AABB. MUST match the
/// WGSL constant so the two marches cap identically.
const MAX_BLOCK_STEPS: u32 = 4096;

/// Is the clip-map cell containing `absolute_block` occupied — or the level OFF
/// (empty ⇒ no hierarchical skip, the flat G1 DDA)? Mirrors the shader's
/// `clipmap_cell_occupied`: floor-div the absolute block into the cell lattice,
/// pack the cell key, binary-search the sorted level.
fn cpu_clipmap_cell_occupied(level: &ClipmapLevel, absolute_block: glam::IVec3) -> bool {
    if level.cell_keys.is_empty() {
        return true;
    }
    let blocks = level.blocks_per_cell.max(1) as i32;
    let cell = [
        absolute_block.x.div_euclid(blocks) as i64,
        absolute_block.y.div_euclid(blocks) as i64,
        absolute_block.z.div_euclid(blocks) as i64,
    ];
    let key = pack_world_block_key(cell);
    level.cell_keys.binary_search(&key).is_ok()
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
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let safe = safe_direction(direction);
    let edge = frame.brick_edge_voxels as f32;
    let edge_i = frame.brick_edge_voxels;
    let bounds_lo = frame.traversal_lo;
    let bounds_hi = frame.traversal_hi;
    let block_bias = glam::IVec3::from_array(frame.block_bias);

    let inverse = 1.0 / safe;
    let t_a = (bounds_lo - origin) * inverse;
    let t_b = (bounds_hi - origin) * inverse;
    let t_near = t_a.min(t_b);
    let t_far = t_a.max(t_b);
    let t_enter = t_near.x.max(t_near.y).max(t_near.z).max(0.0);
    let t_exit = t_far.x.min(t_far.y).min(t_far.z);
    if t_exit < t_enter {
        return (None, 0);
    }

    let entry_position = origin + direction * (t_enter + 1e-4);
    let mut block_cell = (entry_position / edge).floor().as_ivec3();
    let block_step = glam::IVec3::new(
        direction.x.signum() as i32,
        direction.y.signum() as i32,
        direction.z.signum() as i32,
    );
    let t_delta = (glam::Vec3::splat(edge) / safe).abs();
    let seed_axis = |cell: i32, step: i32, entry: f32, safe_axis: f32| -> f32 {
        if step > 0 {
            ((cell + 1) as f32 * edge - entry) / safe_axis
        } else {
            (cell as f32 * edge - entry) / safe_axis
        }
    };
    let mut t_max = glam::Vec3::new(
        seed_axis(block_cell.x, block_step.x, entry_position.x, safe.x) + t_enter,
        seed_axis(block_cell.y, block_step.y, entry_position.y, safe.y) + t_enter,
        seed_axis(block_cell.z, block_step.z, entry_position.z, safe.z) + t_enter,
    );
    let mut t_block_enter = t_enter;

    // Re-seed the block DDA at the exit of the clip-map cell of `absolute_block`
    // (cells `blocks` blocks/axis) — the CPU mirror of the shader `clipmap_try_skip`.
    // Returns `(new_block_cell, new_t_max, jump_t)`; the caller compares `new_block`
    // to the current cell to decide advancement (no capture of the mutated cell).
    let cell_exit_and_reseed =
        |absolute_block: glam::IVec3, blocks: i32| -> (glam::IVec3, glam::Vec3, f32) {
            let cell = glam::IVec3::new(
                absolute_block.x.div_euclid(blocks),
                absolute_block.y.div_euclid(blocks),
                absolute_block.z.div_euclid(blocks),
            );
            let sv_block_lo = cell * blocks - block_bias;
            let cell_lo = sv_block_lo.as_vec3() * edge;
            let cell_hi = (sv_block_lo + glam::IVec3::splat(blocks)).as_vec3() * edge;
            let ta = (cell_lo - origin) * inverse;
            let tb = (cell_hi - origin) * inverse;
            let tfar = ta.max(tb);
            let cell_exit = tfar.x.min(tfar.y).min(tfar.z);
            let jump_t = cell_exit + CLIPMAP_JUMP_EPSILON;
            let jump_pos = origin + direction * jump_t;
            let new_block = (jump_pos / edge).floor().as_ivec3();
            let new_t_max = glam::Vec3::new(
                seed_axis(new_block.x, block_step.x, jump_pos.x, safe.x) + jump_t,
                seed_axis(new_block.y, block_step.y, jump_pos.y, safe.y) + jump_t,
                seed_axis(new_block.z, block_step.z, jump_pos.z, safe.z) + jump_t,
            );
            (new_block, new_t_max, jump_t)
        };

    let mut steps = 0u32;
    'march: for _ in 0..MAX_BLOCK_STEPS {
        steps += 1;
        let absolute_block_v = block_cell + block_bias;
        // G2/G4 hierarchical DDA: descend the levels coarsest→finest and skip by
        // the coarsest level whose cell is EMPTY — an empty cell jumps the ray to
        // that cell's exit in ONE stride (L3 → L2 → L1 → per-block). A jump that
        // wouldn't advance the block cell falls through to a per-block step
        // (guaranteed progress). A step-for-step mirror of the shader's else-if
        // chain: only the coarsest empty level is attempted each step.
        let mut jumped = false;
        for level in levels_coarse_to_fine {
            if cpu_clipmap_cell_occupied(level, absolute_block_v) {
                continue; // occupied (or level off) — try the next finer level
            }
            let (new_block, new_t_max, jump_t) =
                cell_exit_and_reseed(absolute_block_v, level.blocks_per_cell.max(1) as i32);
            if new_block != block_cell {
                if jump_t > t_exit {
                    break 'march;
                }
                block_cell = new_block;
                t_max = new_t_max;
                t_block_enter = jump_t;
                jumped = true;
            }
            break; // only the coarsest empty level is attempted this step
        }
        if jumped {
            continue 'march;
        }
        let absolute_block = [absolute_block_v.x, absolute_block_v.y, absolute_block_v.z];
        let (key_hi, key_lo) = cpu_pack_key_split(absolute_block);
        if let Some(record_index) = cpu_find_brick_record(records, key_hi, key_lo) {
            let record = records[record_index];
            let block_lo = block_cell.as_vec3() * edge;
            let block_hi = block_lo + glam::Vec3::splat(edge);
            let clamped_lo = block_lo.max(bounds_lo);
            let clamped_hi = block_hi.min(bounds_hi);
            if clamped_lo.x < clamped_hi.x && clamped_lo.y < clamped_hi.y && clamped_lo.z < clamped_hi.z
            {
                // Clamped-box entry — mirrors `clamped_box_entry` (x → y → z ties).
                let box_t_a = (clamped_lo - origin) * inverse;
                let box_t_b = (clamped_hi - origin) * inverse;
                let box_near = box_t_a.min(box_t_b);
                let box_far = box_t_a.max(box_t_b);
                let box_exit = box_far.x.min(box_far.y).min(box_far.z);
                let (entry_axis, mut box_enter) =
                    if box_near.x >= box_near.y && box_near.x >= box_near.z {
                        (0usize, box_near.x)
                    } else if box_near.y >= box_near.z {
                        (1usize, box_near.y)
                    } else {
                        (2usize, box_near.z)
                    };
                box_enter = box_enter.max(0.0);
                if box_exit >= box_enter {
                    // Mirror the WGSL kind decode: the discriminant is the low bits of
                    // `kind` (the material id rides above it, ADR 0011 G2).
                    let coarse_form = record_kind_discriminant(record.kind) == 0
                        || record.atlas_slot == NON_RESIDENT_ATLAS_SLOT;
                    if coarse_form {
                        let hit_position = origin + direction * (box_enter + 1e-4);
                        let block_min_voxel = block_cell * edge_i;
                        let voxel_cell = hit_position
                            .floor()
                            .as_ivec3()
                            .clamp(block_min_voxel, block_min_voxel + glam::IVec3::splat(edge_i - 1));
                        return (
                            Some(CpuMarchHit {
                                absolute_voxel: [
                                    voxel_cell.x + frame.voxel_bias[0],
                                    voxel_cell.y + frame.voxel_bias[1],
                                    voxel_cell.z + frame.voxel_bias[2],
                                ],
                                face_normal: axis_normal(entry_axis, direction),
                            }),
                            steps,
                        );
                    }
                    // Sculpted brick voxel DDA — mirrors the shader loop (tracking the
                    // per-voxel entry axis for the hit face's normal).
                    let mut voxel_entry_axis = entry_axis;
                    let voxel_entry = origin + direction * (box_enter + 1e-4);
                    let mut voxel_cell = voxel_entry.floor().as_ivec3();
                    let voxel_step = block_step;
                    let voxel_t_delta = (1.0 / safe).abs();
                    let seed_voxel = |cell: i32, step: i32, entry: f32, safe_axis: f32| -> f32 {
                        if step > 0 {
                            ((cell + 1) as f32 - entry) / safe_axis
                        } else {
                            (cell as f32 - entry) / safe_axis
                        }
                    };
                    let mut voxel_t_max = glam::Vec3::new(
                        seed_voxel(voxel_cell.x, voxel_step.x, voxel_entry.x, safe.x) + box_enter,
                        seed_voxel(voxel_cell.y, voxel_step.y, voxel_entry.y, safe.y) + box_enter,
                        seed_voxel(voxel_cell.z, voxel_step.z, voxel_entry.z, safe.z) + box_enter,
                    );
                    let block_min_voxel = block_cell * edge_i;
                    let block_max_voxel = block_min_voxel + glam::IVec3::splat(edge_i);
                    let band_z_lo = block_min_voxel.z.max(frame.band_voxel_sv[0]);
                    let band_z_hi = block_max_voxel.z.min(frame.band_voxel_sv[1]);
                    for _ in 0..256 {
                        if voxel_cell.x < block_min_voxel.x
                            || voxel_cell.y < block_min_voxel.y
                            || voxel_cell.z < band_z_lo
                            || voxel_cell.x >= block_max_voxel.x
                            || voxel_cell.y >= block_max_voxel.y
                            || voxel_cell.z >= band_z_hi
                        {
                            break;
                        }
                        let brick_local = voxel_cell - block_min_voxel;
                        if cpu_sculpted_voxel_occupied(
                            build,
                            record.atlas_slot,
                            [brick_local.x, brick_local.y, brick_local.z],
                        ) {
                            return (
                                Some(CpuMarchHit {
                                    absolute_voxel: [
                                        voxel_cell.x + frame.voxel_bias[0],
                                        voxel_cell.y + frame.voxel_bias[1],
                                        voxel_cell.z + frame.voxel_bias[2],
                                    ],
                                    face_normal: axis_normal(voxel_entry_axis, direction),
                                }),
                                steps,
                            );
                        }
                        if voxel_t_max.x <= voxel_t_max.y && voxel_t_max.x <= voxel_t_max.z {
                            voxel_cell.x += voxel_step.x;
                            voxel_t_max.x += voxel_t_delta.x;
                            voxel_entry_axis = 0;
                        } else if voxel_t_max.y <= voxel_t_max.z {
                            voxel_cell.y += voxel_step.y;
                            voxel_t_max.y += voxel_t_delta.y;
                            voxel_entry_axis = 1;
                        } else {
                            voxel_cell.z += voxel_step.z;
                            voxel_t_max.z += voxel_t_delta.z;
                            voxel_entry_axis = 2;
                        }
                    }
                }
            }
        }

        if t_block_enter > t_exit {
            break;
        }
        if t_max.x <= t_max.y && t_max.x <= t_max.z {
            block_cell.x += block_step.x;
            t_block_enter = t_max.x;
            t_max.x += t_delta.x;
        } else if t_max.y <= t_max.z {
            block_cell.y += block_step.y;
            t_block_enter = t_max.y;
            t_max.y += t_delta.y;
        } else {
            block_cell.z += block_step.z;
            t_block_enter = t_max.z;
            t_max.z += t_delta.z;
        }
    }

    (None, steps)
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
    let (origin, direction) = cpu_camera_ray(frame, pixel);
    let safe = safe_direction(direction);
    let bounds_lo = frame.traversal_lo;
    let bounds_hi = frame.traversal_hi;

    let inverse = 1.0 / safe;
    let t_a = (bounds_lo - origin) * inverse;
    let t_b = (bounds_hi - origin) * inverse;
    let t_near = t_a.min(t_b);
    let t_far = t_a.max(t_b);
    let t_enter = t_near.x.max(t_near.y).max(t_near.z).max(0.0);
    let t_exit = t_far.x.min(t_far.y).min(t_far.z);
    if t_exit < t_enter {
        return None;
    }

    let entry_position = origin + direction * (t_enter + 1e-4);
    let mut voxel_cell = entry_position.floor().as_ivec3();
    let step = glam::IVec3::new(
        direction.x.signum() as i32,
        direction.y.signum() as i32,
        direction.z.signum() as i32,
    );
    let t_delta = (1.0 / safe).abs();
    let seed_voxel = |cell: i32, step: i32, entry: f32, safe_axis: f32| -> f32 {
        if step > 0 {
            ((cell + 1) as f32 - entry) / safe_axis
        } else {
            (cell as f32 - entry) / safe_axis
        }
    };
    let mut t_max = glam::Vec3::new(
        seed_voxel(voxel_cell.x, step.x, entry_position.x, safe.x) + t_enter,
        seed_voxel(voxel_cell.y, step.y, entry_position.y, safe.y) + t_enter,
        seed_voxel(voxel_cell.z, step.z, entry_position.z, safe.z) + t_enter,
    );
    let mut t_voxel_enter = t_enter;
    // The entered face's axis — the AABB entry (x→y→z ties), updated each DDA step.
    let mut entry_axis = if t_near.x >= t_near.y && t_near.x >= t_near.z {
        0usize
    } else if t_near.y >= t_near.z {
        1
    } else {
        2
    };

    // Generous budget: the traversal AABB's voxel diagonal for every gated scene.
    for _ in 0..4096 {
        // Band clip per voxel (the traversal AABB already bounds Z; the integer
        // check keeps float-edge voxels honest, mirroring the brick march's bound).
        if voxel_cell.z >= frame.band_voxel_sv[0] && voxel_cell.z < frame.band_voxel_sv[1] {
            let absolute = [
                (voxel_cell.x + frame.voxel_bias[0]) as i64,
                (voxel_cell.y + frame.voxel_bias[1]) as i64,
                (voxel_cell.z + frame.voxel_bias[2]) as i64,
            ];
            if occupied(absolute) {
                return Some(CpuMarchHit {
                    absolute_voxel: [
                        voxel_cell.x + frame.voxel_bias[0],
                        voxel_cell.y + frame.voxel_bias[1],
                        voxel_cell.z + frame.voxel_bias[2],
                    ],
                    face_normal: axis_normal(entry_axis, direction),
                });
            }
        }
        if t_voxel_enter > t_exit {
            break;
        }
        if t_max.x <= t_max.y && t_max.x <= t_max.z {
            voxel_cell.x += step.x;
            t_voxel_enter = t_max.x;
            t_max.x += t_delta.x;
            entry_axis = 0;
        } else if t_max.y <= t_max.z {
            voxel_cell.y += step.y;
            t_voxel_enter = t_max.y;
            t_max.y += t_delta.y;
            entry_axis = 1;
        } else {
            voxel_cell.z += step.z;
            t_voxel_enter = t_max.z;
            t_max.z += t_delta.z;
            entry_axis = 2;
        }
    }

    None
}

#[cfg(test)]
mod representability_tests {
    //! ADR 0011 G2 — `brick_representable_overlay` decides the widened live gate. The
    //! genuinely-non-representable cases (a block mixing materials; blocks disagreeing on
    //! the overlay) are built directly here, so the fallback is gated without a
    //! sub-block-offset demo scene (every whole-block-offset demo is single-material per
    //! block and thus representable — see the golden's note on `--demo-overlap`).
    use super::brick_representable_overlay;
    use crate::cuboid::VoxelBox;
    use crate::cuboid_mesh::compose_cell_key;
    use crate::two_layer_store::{MicroblockGeometry, SeamSolidity, TwoLayerChunk};
    use std::collections::BTreeMap;

    fn geom(material_keys: &[u16]) -> MicroblockGeometry {
        MicroblockGeometry {
            cuboids: material_keys
                .iter()
                .map(|&material_id| VoxelBox {
                    min: [0, 0, 0],
                    max: [0, 0, 0],
                    material_id,
                })
                .collect(),
            seam_solidity: SeamSolidity {
                solid: [[true; 2]; 3],
            },
        }
    }

    fn chunk_with(
        microblocks: Vec<([u32; 3], MicroblockGeometry)>,
    ) -> Vec<([i32; 3], std::sync::Arc<TwoLayerChunk>)> {
        vec![(
            [0, 0, 0],
            std::sync::Arc::new(TwoLayerChunk {
                voxels_per_block: 4,
                coarse: Vec::new(),
                coarse_overlay: Vec::new(),
                microblocks: microblocks.into_iter().collect::<BTreeMap<_, _>>(),
            }),
        )]
    }

    #[test]
    fn representable_across_distinct_single_material_blocks() {
        // Two boundary blocks, each internally single-material but DIFFERENT materials —
        // the G2 multi-producer case (per-record ids). Uniform overlay off ⇒ Some(false).
        let chunks = chunk_with(vec![
            ([0, 0, 0], geom(&[compose_cell_key(0, false)])),
            ([1, 0, 0], geom(&[compose_cell_key(1, false)])),
        ]);
        assert_eq!(brick_representable_overlay(&chunks), Some(false));
    }

    #[test]
    fn representable_with_uniform_overlay_on() {
        let chunks = chunk_with(vec![
            ([0, 0, 0], geom(&[compose_cell_key(0, true)])),
            ([1, 0, 0], geom(&[compose_cell_key(1, true)])),
        ]);
        assert_eq!(brick_representable_overlay(&chunks), Some(true));
    }

    #[test]
    fn brick_representable_overlay_rejects_mixed_block() {
        // ONE block whose microblocks mix two materials — the R8 atlas is occupancy-only,
        // so this block can't be a single brick ⇒ not representable.
        let chunks = chunk_with(vec![(
            [0, 0, 0],
            geom(&[compose_cell_key(0, false), compose_cell_key(1, false)]),
        )]);
        assert_eq!(brick_representable_overlay(&chunks), None);
    }

    #[test]
    fn rejects_overlay_disagreement_across_blocks() {
        // Two single-material blocks that DISAGREE on the on-face grid — overlay is a
        // scene-wide uniform (not per-record), so the set can't be represented.
        let chunks = chunk_with(vec![
            ([0, 0, 0], geom(&[compose_cell_key(0, false)])),
            ([1, 0, 0], geom(&[compose_cell_key(0, true)])),
        ]);
        assert_eq!(brick_representable_overlay(&chunks), None);
    }
}
