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
    BrickFieldBuild, BrickFieldUpdate, BrickPayload, ClipmapLevel, ClipmapPyramid,
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

/// Pack the G0 build's records for the GPU. `non_resident` marks sculpted slots to
/// upload as [`NON_RESIDENT_ATLAS_SLOT`] — the residency-miss test's forced-miss
/// hook (and G4's future eviction seam); pass `|_| false` for the all-resident set.
pub fn pack_gpu_records(
    build: &BrickFieldBuild,
    mut non_resident: impl FnMut(u32) -> bool,
) -> Vec<BrickGpuRecord> {
    build
        .brick_records
        .iter()
        .map(|record| gpu_record_of(record, &mut non_resident))
        .collect()
}

/// As [`pack_gpu_records`], but ELIDES fully-occluded interior blocks (ADR 0011 interior
/// elision — the brick display sink's analogue of the mesh's interior-face culling). A
/// block whose six neighbours are all solid on the shared face can never be a ray's first
/// hit (the ray stops at the surrounding solid), so dropping it from the record buffer the
/// shader binary-searches is **hit-identical** — gated in
/// `tests/gpu_parity.rs::brick_surface_elision_hit_set_unchanged`. The clip-map, atlas and
/// fog keep the FULL set (built from `build`); only the per-edit record buffer shrinks —
/// ∝ surface, not volume, for a large solid (a 500×50×50-block box drops ~1.14M of 1.25M
/// records). Ordering is preserved (a filter over the sorted set stays sorted), so the
/// in-shader binary search is unaffected.
pub fn pack_surface_gpu_records(
    build: &BrickFieldBuild,
    mut non_resident: impl FnMut(u32) -> bool,
) -> Vec<BrickGpuRecord> {
    let keep = crate::brick_field::surface_record_mask(&build.brick_records);
    build
        .brick_records
        .iter()
        .zip(keep)
        .filter(|&(_, keep)| keep)
        .map(|(record, _)| gpu_record_of(record, &mut non_resident))
        .collect()
}

/// Pack one brick record into its GPU form — the shared per-record body of
/// [`pack_gpu_records`] and [`pack_surface_gpu_records`].
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
    build: &BrickFieldBuild,
    slot: u32,
) {
    let edge = build.brick_edge_voxels.max(1);
    let tiles = build.bricks_per_axis.max(1);
    let origin = wgpu::Origin3d {
        x: (slot % tiles) * edge,
        y: ((slot / tiles) % tiles) * edge,
        z: (slot / (tiles * tiles)) * edge,
    };
    let tile_bytes = build.sculpted_brick_occupancy(slot);
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: atlas_texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        &tile_bytes,
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
    two_layer_chunks: &[([i32; 3], TwoLayerChunk)],
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
    /// The traversal AABB (resident-brick bounds ∩ band slab), shifted frame.
    pub traversal_lo: glam::Vec3,
    pub traversal_hi: glam::Vec3,
    pub brick_edge_voxels: i32,
    pub bricks_per_axis: u32,
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
    // scene-wide material id rides here — `record_count` plus std140 padding fills the slot.
    record_count: u32,
    _render_cell_pad0: u32,
    _render_cell_pad1: u32,
    _render_cell_pad2: u32,
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
}

/// The G1 brick raymarch renderer: owns the record buffer, the sculpted atlas
/// texture, its own copy of the procedural material atlas (identical texels +
/// sub-rects to the cuboid path's), and the two pipelines (the MSAA render pass
/// entry + the single-sample hit-identity entry the parity net reads back).
pub struct BrickRaymarchRenderer {
    render_pipeline: wgpu::RenderPipeline,
    hit_identity_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    field_bind_group_layout: wgpu::BindGroupLayout,
    field_bind_group: wgpu::BindGroup,
    material_bind_group: wgpu::BindGroup,
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
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick raymarch uniforms"),
            size: std::mem::size_of::<BrickUniformsPod>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Placeholder field: one zeroed record + a 1³ atlas (record_count 0 means
        // the binary search never reads either).
        let placeholder = [BrickGpuRecord::zeroed()];
        let record_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brick raymarch records"),
            contents: bytemuck::cast_slice(&placeholder),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let empty_build = BrickFieldBuild {
            brick_records: Vec::new(),
            sculpted_atlas_bytes: Vec::new(),
            brick_edge_voxels: 1,
            bricks_per_axis: 0,
            atlas_dim_voxels: 0,
        };
        let atlas_texture = upload_brick_atlas(device, queue, &empty_build);
        let atlas_texture_dim = empty_build.atlas_dim_voxels.max(1);
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

        let field_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("brick raymarch field layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
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
                ],
            });
        let field_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick raymarch field bind group"),
            layout: &field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brick raymarch shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/brick_raymarch.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brick raymarch pipeline layout"),
            bind_group_layouts: &[Some(&field_bind_group_layout), Some(&material_bind_group_layout)],
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

        Self {
            render_pipeline,
            hit_identity_pipeline,
            uniform_buffer,
            field_bind_group_layout,
            field_bind_group,
            material_bind_group,
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
        build: &BrickFieldBuild,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: [i64; 3],
        overlay_active: bool,
    ) {
        // A wholesale install (re)creates the atlas texture from scratch and uploads
        // every sculpted slot — the from-scratch / scene-load / gate-re-engage path.
        let atlas_texture = upload_brick_atlas(device, queue, build);
        self.atlas_texture = atlas_texture;
        self.atlas_texture_dim = build.atlas_dim_voxels.max(1);
        self.last_atlas_slots_written = build.sculpted_brick_count() as u32;
        let atlas_view = self
            .atlas_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.rebuild_field_state(
            device,
            &atlas_view,
            build,
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
    /// re-pack, ADR 0007 resize precedent). `build` is the field's current
    /// [`to_build`](crate::brick_field::IncrementalBrickField::to_build) materialisation.
    ///
    /// Preconditions the live shell (`WindowedState::rebuild_geometry`) upholds: a field is
    /// already installed AND its density/frame match `build` (an incremental edit never
    /// changes density — that routes wholesale). Records + pyramid re-upload whole (they
    /// are small — the traffic G3 kills is the atlas texels + the re-evaluation).
    #[allow(clippy::too_many_arguments)]
    pub fn patch_brick_field(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        build: &BrickFieldBuild,
        update: &BrickFieldUpdate,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: [i64; 3],
        overlay_active: bool,
    ) {
        let target_dim = build.atlas_dim_voxels.max(1);
        if update.atlas_grew || target_dim != self.atlas_texture_dim {
            // The tile grid grew/shrank: every slot's 3D position moved, so recreate the
            // texture and re-upload wholesale (ADR 0011 pitfalls — the resize is the one
            // place a full re-pack is legitimate, logged by the caller).
            self.atlas_texture = upload_brick_atlas(device, queue, build);
            self.atlas_texture_dim = target_dim;
            self.last_atlas_slots_written = build.sculpted_brick_count() as u32;
        } else {
            // Steady state: write ONLY the dirty slots' tiles into the persistent texture.
            // Untouched slots — and freed (dead) slots — keep their texels untouched.
            for &slot in &update.written_slots {
                write_atlas_slot(queue, &self.atlas_texture, build, slot);
            }
            self.last_atlas_slots_written = update.written_slots.len() as u32;
        }
        let atlas_view = self
            .atlas_texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.rebuild_field_state(
            device,
            &atlas_view,
            build,
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
        build: &BrickFieldBuild,
        gpu_records: &[BrickGpuRecord],
        pyramid: &ClipmapPyramid,
        recentre_voxels: [i64; 3],
        overlay_active: bool,
    ) {
        // Inclusive absolute block bounds over the record set (the sort is z-major,
        // so x/y still need the full scan; records are few — thousands).
        let mut absolute_block_bounds: Option<([i64; 3], [i64; 3])> = None;
        for record in &build.brick_records {
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

        self.field_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brick raymarch field bind group"),
            layout: &self.field_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
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
            ],
        });
        self.clipmap_level_1_blocks = pyramid.level_1.blocks_per_cell;
        self.clipmap_level_1_count = level_1_keys.len() as u32;
        self.clipmap_level_2_blocks = pyramid.level_2.blocks_per_cell;
        self.clipmap_level_2_count = level_2_keys.len() as u32;
        self.clipmap_level_3_blocks = pyramid.level_3.blocks_per_cell;
        self.clipmap_level_3_count = level_3_keys.len() as u32;
        self.record_count = gpu_records.len() as u32;
        self.overlay_active = overlay_active;
        self.recentre_voxels = recentre_voxels;
        self.brick_edge_voxels = build.brick_edge_voxels;
        self.bricks_per_axis = build.bricks_per_axis;
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
        traversal_lo.z = traversal_lo.z.max(band_lo_sv as f32);
        traversal_hi.z = traversal_hi.z.min(band_hi_sv as f32);

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
        let material_atlas = crate::texture_atlas::MaterialAtlas::from_procedural_materials();
        let overlay = crate::renderer::grid_overlay_params();
        let uniforms = BrickUniformsPod {
            view_projection: frame.view_projection.to_cols_array_2d(),
            inverse_view_projection: frame.inverse_view_projection.to_cols_array_2d(),
            viewport: frame.viewport,
            grid_half_extent: frame.grid_half_extent.to_array(),
            voxels_per_block: self.brick_edge_voxels.max(1) as f32,
            voxel_line_color: overlay.voxel_line_color,
            grid_overlay_enabled: if grid_overlay_master && self.overlay_active {
                1.0
            } else {
                0.0
            },
            block_line_color: overlay.block_line_color,
            material_modulation_enabled: modulation_enabled,
            voxel_line_half_width: overlay.voxel_line_half_width,
            block_line_half_width: overlay.block_line_half_width,
            voxel_line_alpha: overlay.voxel_line_alpha,
            block_line_alpha: overlay.block_line_alpha,
            record_count: self.record_count,
            _render_cell_pad0: 0,
            _render_cell_pad1: 0,
            _render_cell_pad2: 0,
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
                0,
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
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        frame
    }

    /// Draw the brick raymarch INSIDE the shared MSAA voxel pass (viewport +
    /// scissor already set by `render_frame`). Uniforms must be uploaded first.
    pub fn draw<'a>(&'a self, pass: &mut wgpu::RenderPass<'a>) {
        if self.record_count == 0 {
            return;
        }
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.field_bind_group, &[]);
        pass.set_bind_group(1, &self.material_bind_group, &[]);
        pass.draw(0..3, 0..1);
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
            pass.set_bind_group(0, &self.field_bind_group, &[]);
            pass.set_bind_group(1, &self.material_bind_group, &[]);
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
}

// ============================================================================
// CPU reference march — the f32 mirror of the WGSL traversal (the parity net's
// oracle side; never on a live path).
// ============================================================================

/// A CPU march hit: the hit voxel in ABSOLUTE voxel coordinates (the exact
/// evaluator's frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuMarchHit {
    pub absolute_voxel: [i32; 3],
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
                            }),
                            steps,
                        );
                    }
                    // Sculpted brick voxel DDA — mirrors the shader loop.
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
                                }),
                                steps,
                            );
                        }
                        if voxel_t_max.x <= voxel_t_max.y && voxel_t_max.x <= voxel_t_max.z {
                            voxel_cell.x += voxel_step.x;
                            voxel_t_max.x += voxel_t_delta.x;
                        } else if voxel_t_max.y <= voxel_t_max.z {
                            voxel_cell.y += voxel_step.y;
                            voxel_t_max.y += voxel_t_delta.y;
                        } else {
                            voxel_cell.z += voxel_step.z;
                            voxel_t_max.z += voxel_t_delta.z;
                        }
                    }
                    let _ = entry_axis; // entry axis feeds shading, not identity
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
        } else if t_max.y <= t_max.z {
            voxel_cell.y += step.y;
            t_voxel_enter = t_max.y;
            t_max.y += t_delta.y;
        } else {
            voxel_cell.z += step.z;
            t_voxel_enter = t_max.z;
            t_max.z += t_delta.z;
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
    ) -> Vec<([i32; 3], TwoLayerChunk)> {
        vec![(
            [0, 0, 0],
            TwoLayerChunk {
                voxels_per_block: 4,
                coarse: Vec::new(),
                coarse_overlay: Vec::new(),
                microblocks: microblocks.into_iter().collect::<BTreeMap<_, _>>(),
            },
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
