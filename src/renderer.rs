//! The instanced voxel renderer (Milestone 4).
//!
//! Owns the GPU resources that turn a resolved [`VoxelGrid`](crate::voxel::VoxelGrid)
//! into textured instanced cubes: one shared unit-cube vertex/index buffer
//! (24 verts / 36 indices, per-face normals + per-face base UVs), an instance
//! buffer built FROM the grid, the [`VoxelUniforms`] uniform, the three
//! procedural material textures (Stone/Wood/Plain), and the render pipeline.
//!
//! Milestone 4 adds:
//!   * Procedural CPU-generated material textures, selected by [`MaterialChoice`].
//!   * Per-voxel texture slicing (vertex shader; BUG 1 fix).
//!   * A position-based grid overlay (fragment shader; BUG 2 fix).
//!   * 4× MSAA for the 3D pass, resolved into the single-sample target.
//!
//! It is render-target-agnostic: [`VoxelRenderer::draw`] records into a render
//! pass the caller has already begun against any colour view + depth view, so the
//! window and the headless capture paint identically.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::frustum::{Aabb, Frustum};
use crate::panel::MaterialChoice;
use crate::scene::Scene;
use crate::voxel::VoxelGrid;

/// Depth format used by the voxel pass and the depth texture.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sample count for the 3D voxel pass (4× MSAA). The depth texture, the
/// multisampled colour texture and the pipeline all share this count; egui still
/// renders at 1 sample onto the resolved target.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Edge length of every procedural material texture (square, no mipmaps).
const MATERIAL_TEXTURE_SIZE: u32 = 32;

/// Edge length of a render chunk, in BLOCKS (ADR 0002 Decision 3, part of #19).
/// A chunk therefore spans `CHUNK_BLOCKS * voxels_per_block` voxels per axis
/// (e.g. 4 blocks × density 16 = 64 voxels/axis). Chosen as a small whole-block
/// multiple so a chunk stays a phase-aligned, frustum-cullable unit while the
/// draw-call count stays sane. The resolved grid's occupied voxels are bucketed
/// into these chunks at rebuild time; each frame only the chunks whose world
/// AABB intersects the camera frustum are drawn.
pub const CHUNK_BLOCKS: u32 = 4;

/// One cube vertex: position on the unit cube, its face normal, and the base
/// (0..1) UV for that face.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    face_uv: [f32; 2],
}

/// Per-voxel instance data (28-byte stride).
///
/// `material_id` (ADR 0001 step 3) is the per-voxel material handle carried from
/// the resolved grid: a Tool stamps its single id, a Part its own per-voxel ids.
/// It is uploaded as a `u32` (the GPU has no 16-bit vertex format) and indexes the
/// shader's `material_base_colors` uniform array so distinct nodes modulate to
/// distinct colours.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VoxelInstance {
    pub world_position: [f32; 3],
    pub block_local_coord: [f32; 3],
    pub material_id: u32,
}

/// One spatial render chunk (ADR 0002 E2, part of #19): a contiguous
/// `[instance_start, instance_start + instance_count)` slice of the single
/// instance buffer holding every voxel whose centre falls in this chunk's
/// `CHUNK_BLOCKS³`-block cell, plus that cell's world-space AABB for frustum
/// culling. The instance buffer is laid out chunk-by-chunk so each chunk draws
/// as one `draw_indexed` over its own instance range.
#[derive(Debug, Clone, Copy)]
pub struct Chunk {
    /// First instance index of this chunk's slice in the instance buffer.
    pub instance_start: u32,
    /// Number of instances (voxels) in this chunk.
    pub instance_count: u32,
    /// World-space AABB of the chunk's voxel cubes (centres ±0.5 per axis).
    pub aabb: Aabb,
}

/// Bucket a grid's occupied voxels into spatial chunks (ADR 0002 E2).
///
/// Returns the instance list REORDERED so every chunk's voxels are contiguous,
/// plus the per-chunk ranges + world AABBs. Each voxel lands in exactly one
/// chunk, keyed by `floor(world_position / chunk_extent_voxels)` where
/// `chunk_extent_voxels = CHUNK_BLOCKS * voxels_per_block` (one voxel = one world
/// unit, voxel CENTRES at half-integer world coords). The union of all chunk
/// instance ranges is the whole occupied set, with no truncation — the old 450k
/// draw cap is gone, so a scene up to the 6M RESOLVE cap renders fully.
pub fn bucket_instances_into_chunks(
    grid: &VoxelGrid,
    voxels_per_block: u32,
) -> (Vec<VoxelInstance>, Vec<Chunk>) {
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;

    // Group voxel indices by integer chunk coordinate.
    let mut buckets: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (index, voxel) in grid.occupied.iter().enumerate() {
        let key = [
            (voxel.world_position[0] / chunk_extent).floor() as i32,
            (voxel.world_position[1] / chunk_extent).floor() as i32,
            (voxel.world_position[2] / chunk_extent).floor() as i32,
        ];
        buckets.entry(key).or_default().push(index);
    }

    // Sort the chunk keys so the instance-buffer layout (and thus the goldens)
    // is deterministic regardless of HashMap iteration order.
    let mut keys: Vec<[i32; 3]> = buckets.keys().copied().collect();
    keys.sort_unstable();

    let mut instances = Vec::with_capacity(grid.occupied.len());
    let mut chunks = Vec::with_capacity(keys.len());
    for key in keys {
        let indices = &buckets[&key];
        let instance_start = instances.len() as u32;
        let mut aabb = Aabb::empty();
        for &index in indices {
            let voxel = &grid.occupied[index];
            // A voxel cube spans ±0.5 around its centre world position.
            let center = glam::Vec3::from(voxel.world_position);
            aabb.expand(center - glam::Vec3::splat(0.5));
            aabb.expand(center + glam::Vec3::splat(0.5));
            instances.push(VoxelInstance {
                world_position: voxel.world_position,
                block_local_coord: [
                    voxel.block_local_coord[0] as f32,
                    voxel.block_local_coord[1] as f32,
                    voxel.block_local_coord[2] as f32,
                ],
                material_id: voxel.material_id as u32,
            });
        }
        chunks.push(Chunk {
            instance_start,
            instance_count: indices.len() as u32,
            aabb,
        });
    }
    (instances, chunks)
}

/// Build the instance list + world-space AABB for ONE chunk's voxels (issue #20
/// S6c-2b). The per-chunk render accessor (`ChunkResolveCache::resident_render_chunks`)
/// already hands each chunk as its own [`VoxelGrid`] in render (recentred) coords,
/// so unlike [`bucket_instances_into_chunks`] there is no spatial grouping to do:
/// every occupied voxel becomes one [`VoxelInstance`] and the AABB is the union of
/// their cubes (centres ±0.5 per axis).
///
/// The produced bytes are identical to the slice `bucket_instances_into_chunks`
/// produces for this chunk from the assembled whole grid — bucketing a single
/// per-chunk grid yields exactly that chunk's instances (proven in S6c-2a:
/// per-chunk grids are byte-identical to the corresponding slices of the assembled
/// grid). Returns `None` when the chunk has zero voxels (the caller skips it — no
/// buffer is allocated).
fn instances_for_chunk(grid: &VoxelGrid) -> Option<(Vec<VoxelInstance>, Aabb)> {
    if grid.occupied.is_empty() {
        return None;
    }
    let mut instances = Vec::with_capacity(grid.occupied.len());
    let mut aabb = Aabb::empty();
    for voxel in &grid.occupied {
        // A voxel cube spans ±0.5 around its centre world position.
        let center = glam::Vec3::from(voxel.world_position);
        aabb.expand(center - glam::Vec3::splat(0.5));
        aabb.expand(center + glam::Vec3::splat(0.5));
        instances.push(VoxelInstance {
            world_position: voxel.world_position,
            block_local_coord: [
                voxel.block_local_coord[0] as f32,
                voxel.block_local_coord[1] as f32,
                voxel.block_local_coord[2] as f32,
            ],
            material_id: voxel.material_id as u32,
        });
    }
    Some((instances, aabb))
}

/// The dirty-chunk rebuild plan (issue #20 S6c-2c): which per-chunk GPU buffers an
/// incremental edit must (re)build, and which it must evict.
///
/// Computed purely from coord SETS — the GPU-cache resident coords, the edit's
/// evicted (dirty) coords, and the post-edit covering coords — so it is unit-tested
/// without a GPU device, yet [`VoxelRenderer::incremental_rebuild_from_chunks`]
/// drives the real cache from exactly this plan. Applying it makes the resident set
/// equal the covering set and every rebuilt chunk's contents match a fresh resolve,
/// so the post-edit GPU cache is identical to a wholesale rebuild.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct IncrementalRebuildPlan {
    /// Covering coords whose buffer must be (re)built: DIRTY (evicted by this edit)
    /// or NEW (no resident buffer yet). Their grids are the only resolve-cache
    /// MISSES; every other covering chunk is a HIT (byte-identical → keep).
    pub rebuild: Vec<[i32; 3]>,
    /// Resident coords the post-edit scene no longer covers (a removed/shrunk node
    /// vacated them) — their buffers must be dropped.
    pub evict: Vec<[i32; 3]>,
}

/// Compute the incremental dirty-chunk rebuild plan (issue #20 S6c-2c) from coord
/// sets alone (no GPU).
///
/// `resident` is the GPU cache's current coord set (only NON-empty chunks ever have
/// a buffer — [`VoxelRenderer::rebuild_chunk`] allocates nothing for a zero-voxel
/// chunk). `occupied_covering` is the set of post-edit covering coords that resolve
/// to a NON-EMPTY grid (so deserve a buffer); empty covering chunks are excluded
/// here so they are never treated as "new" work nor kept resident. `evicted` is the
/// edit's dirty coords from the resolve cache.
///
/// A coord is REBUILT iff it is occupied-covering AND (dirty OR not currently
/// resident). A resident coord is EVICTED iff it is no longer occupied-covering —
/// which captures BOTH a vacated chunk (a removed/shrunk node) AND a chunk that an
/// edit turned empty (dirty + now zero voxels). Occupied coords that are
/// resident-and-not-dirty are kept untouched (resolve-cache hits → byte-identical →
/// buffers already correct).
///
/// Applying this plan and making every rebuilt entry equal its fresh grid yields
/// EXACTLY the occupied-covering coord set with fresh contents — identical to a
/// wholesale rebuild (which also stores only non-empty chunks). The returned vectors
/// are sorted so the plan is deterministic and the rebuild count is order-independent.
pub fn incremental_rebuild_plan(
    resident: &[[i32; 3]],
    evicted: &[[i32; 3]],
    occupied_covering: &[[i32; 3]],
) -> IncrementalRebuildPlan {
    let resident_set: std::collections::HashSet<[i32; 3]> = resident.iter().copied().collect();
    let evicted_set: std::collections::HashSet<[i32; 3]> = evicted.iter().copied().collect();
    let covering_set: std::collections::HashSet<[i32; 3]> =
        occupied_covering.iter().copied().collect();

    let mut rebuild: Vec<[i32; 3]> = occupied_covering
        .iter()
        .copied()
        .filter(|coord| evicted_set.contains(coord) || !resident_set.contains(coord))
        .collect();
    rebuild.sort_unstable();
    rebuild.dedup();

    let mut evict: Vec<[i32; 3]> = resident
        .iter()
        .copied()
        .filter(|coord| !covering_set.contains(coord))
        .collect();
    evict.sort_unstable();
    evict.dedup();

    IncrementalRebuildPlan { rebuild, evict }
}

/// The uniform block uploaded to the shader.
///
/// std140-safe: every `vec3` (`[f32; 3]`) is immediately followed by a scalar so
/// the vec3 never straddles a 16-byte boundary. Field order matches the WGSL
/// `VoxelUniforms` struct exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VoxelUniforms {
    view_projection: [[f32; 4]; 4],
    grid_half_extent: [f32; 3],
    voxels_per_block: f32,
    voxel_line_color: [f32; 3],
    grid_overlay_enabled: f32,
    block_line_color: [f32; 3],
    /// Face-orientation debug flag (0 = normal, 1 = colour-by-normal debug).
    /// Reuses the std140 scalar slot that pads the preceding vec3 to 16 bytes.
    debug_face_mode: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
    /// Layer-range scrubber band (issue #12), in voxel Y-layer indices. A voxel
    /// is drawn solid when `band_min <= layer <= band_max` (both ends INCLUSIVE).
    /// Full range = `band_min 0`, `band_max >= grid_y - 1` (nothing clipped). The
    /// onion skin is a separate volumetric fog pass, so the voxel pass only needs
    /// the band; the two pads keep the trailing std140 16-byte slot.
    band_min: f32,
    band_max: f32,
    /// Per-voxel material modulation toggle (ADR 0001 step 3): `1` = modulate the
    /// lit/textured colour by `material_base_colors[material_id]`, `0` = leave it
    /// (the bound texture wins globally). Off for debug-faces and for a loaded VS
    /// block (which stays a single global material). Reuses a former band pad.
    material_modulation_enabled: f32,
    _band_pad1: f32,
    /// Per-material base colours (ADR 0001 step 3), one `vec4` per
    /// [`MaterialChoice`] (`[r, g, b, _pad]`, linear). Indexed by the per-instance
    /// `material_id`; the fragment shader MODULATES the lit/textured colour by this
    /// base so distinct nodes render in distinct materials cheaply. Each entry is
    /// the material's average colour ([`procedural_material_average_color`])
    /// RELATIVE to the bound texture's own average (i.e. divided by it), so the
    /// bound material's own slot is ~neutral white and the others recolour the
    /// shared texture toward their own tint (see `update_uniforms`). Padded to
    /// `vec4` for std140 array stride.
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
}

/// Grid overlay tuning, transcribed from the prototype `GRID` uniforms
/// (chisel-bench-reference.html). Half-widths are in voxel units (the overlay is
/// computed from absolute voxel position), alphas are blend strengths, and the
/// colours are the sRGB hex line colours (ARCHITECTURE.md §8).
const VOXEL_LINE_HALF_WIDTH: f32 = 0.05;
const BLOCK_LINE_HALF_WIDTH: f32 = 0.11;
const VOXEL_LINE_ALPHA: f32 = 0.40;
const BLOCK_LINE_ALPHA: f32 = 0.92;
/// Voxel grid line colour `#17120b` (sRGB hex → linear).
const VOXEL_LINE_COLOR_HEX: u32 = 0x17_12_0b;
/// Block grid line colour `#080605` (sRGB hex → linear, darker/bolder).
const BLOCK_LINE_COLOR_HEX: u32 = 0x08_06_05;

/// Build the 24 vertices / 36 indices of a unit cube spanning `[-1, 1]` per axis
/// with one outward normal AND one 0..1 base UV per face. The shader scales the
/// position by 0.5, giving a unit cube centred on each voxel.
fn unit_cube_geometry() -> (Vec<CubeVertex>, Vec<u16>) {
    // Base UVs for the four corners of every face, in winding order.
    const FACE_UVS: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    // (normal, the four corner offsets in the plane of that face). Every corner
    // list is wound counter-clockwise WHEN VIEWED FROM OUTSIDE the cube so that
    // `front_face: Ccw` + `cull_mode: Back` keeps the outward faces. (The +X/-X/
    // +Y/-Y lists were previously wound clockwise-from-outside, which culled the
    // four side/top/bottom faces and rendered only the inner +Z/-Z faces — the
    // "backfaces only" bug.)
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        // -Z
        ([0.0, 0.0, -1.0], [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u16;
        for (corner_index, corner) in corners.iter().enumerate() {
            vertices.push(CubeVertex {
                position: *corner,
                normal,
                face_uv: FACE_UVS[corner_index],
            });
        }
        // Two triangles per face. Every face's corner list above is wound
        // counter-clockwise WHEN VIEWED FROM OUTSIDE the cube (verified by
        // `voxel_cube_is_ccw_outward`), so with `front_face: Ccw` +
        // `cull_mode: Back` the OUTWARD faces are kept and the inner ones culled.
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// Convert one sRGB 8-bit component to a linear float (matches the sRGB texture
/// decode the GPU applies to material samples, so the grid line colours mix in
/// the same colour space as the textured surface).
fn srgb_component_to_linear(byte: u8) -> f32 {
    let value = byte as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert a packed `0xRRGGBB` sRGB hex colour to a linear `[f32; 3]`.
fn srgb_hex_to_linear(hex: u32) -> [f32; 3] {
    [
        srgb_component_to_linear(((hex >> 16) & 0xff) as u8),
        srgb_component_to_linear(((hex >> 8) & 0xff) as u8),
        srgb_component_to_linear((hex & 0xff) as u8),
    ]
}

/// Append an alpha channel to a linear RGB colour, producing the `[f32; 4]` the
/// line pipeline's vertices carry (M8: lattice/floor draw at low opacity).
fn with_alpha(rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], alpha]
}

/// The visible layer band (issue #12), in voxel Y-layer indices, passed to the
/// voxel shader. The band is INCLUSIVE on both ends: layers `[band_min, band_max]`
/// render solid. `onion_depth` is the number of layers OUTSIDE the band that
/// render ghosted (screen-door dither); `0` means a hard clip at the band.
///
/// Pass [`LayerBand::FULL`] (or any band whose `band_max >= grid_y - 1` and
/// `band_min == 0`) to draw the whole model unclipped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerBand {
    pub band_min: u32,
    pub band_max: u32,
    pub onion_depth: u32,
}

impl LayerBand {
    /// An effectively-unbounded band (the whole grid, no onion skin). `band_max`
    /// is huge so no layer is ever clipped regardless of `grid_y`.
    pub const FULL: LayerBand = LayerBand {
        band_min: 0,
        band_max: u32::MAX,
        onion_depth: 0,
    };
}

/// One render chunk's own GPU buffers (issue #20 S6c-2b): an instance
/// `wgpu::Buffer` holding exactly this chunk's voxels, the instance count, and
/// the chunk's world-space AABB for frustum culling. Each resident chunk owns its
/// buffer independently, so a per-chunk dirty rebuild replaces one entry without
/// touching the rest (the incremental path lands in S6c-2c; this step rebuilds
/// every chunk wholesale via [`VoxelRenderer::rebuild_all_from_chunks`]).
///
/// A chunk that resolves to zero voxels is never stored (no buffer is allocated),
/// so every entry in the cache has `instance_count > 0`.
pub struct InstancedChunkBuffers {
    /// This chunk's instance buffer (one [`VoxelInstance`] per voxel).
    instance_buffer: wgpu::Buffer,
    /// Number of instances (voxels) in this chunk's buffer.
    instance_count: u32,
    /// World-space AABB of the chunk's voxel cubes (centres ±0.5 per axis).
    aabb: Aabb,
}

/// All GPU resources for drawing the voxel grid as textured instanced cubes.
pub struct VoxelRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Face-orientation debug pipeline: identical to `pipeline` except
    /// `cull_mode: None`, so a back face that is the nearest surface (a winding
    /// bug) still DRAWS and gets flagged by the shader's `front_facing` marker.
    /// Depth testing stays on so the nearest face still wins.
    debug_pipeline: wgpu::RenderPipeline,
    cube_vertex_buffer: wgpu::Buffer,
    cube_index_buffer: wgpu::Buffer,
    cube_index_count: u32,
    /// The per-chunk GPU buffer cache (issue #20 S6c-2b): one
    /// [`InstancedChunkBuffers`] per resident chunk, keyed by ABSOLUTE chunk
    /// coordinate (the coord the resolve cache's accessor reports). Replaces the
    /// single grown monolithic instance buffer + `Vec<Chunk>` ranges: each chunk
    /// now owns its own instance buffer, built from its own per-chunk
    /// [`VoxelGrid`]. Frustum-culled per frame in `update_uniforms`; one
    /// `draw_indexed` per visible chunk over its own buffer in `draw`.
    chunk_buffers: HashMap<[i32; 3], InstancedChunkBuffers>,
    /// The chunk coords (keys into `chunk_buffers`) that survived the last frustum
    /// cull (computed in `update_uniforms`, consumed in `draw`). Reset to "every
    /// resident chunk visible" on every rebuild so a first draw before any uniform
    /// upload still shows everything.
    visible_chunks: Vec<[i32; 3]>,
    /// Observability counter (issue #20 S6c-2c): how many per-chunk GPU buffers the
    /// LAST rebuild actually (re)built. After an incremental dirty-chunk rebuild
    /// this is the number of chunks rebuilt (dirty ∪ new); after a wholesale
    /// `rebuild_all_from_chunks` / `rebuild_instances` it is every chunk built. A
    /// smoke-test reads it via [`VoxelRenderer::last_rebuilt_chunk_count`] to confirm
    /// a localised edit rebuilt only N chunks, not the whole scene.
    last_rebuilt_chunk_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// One bind group per material (Stone/Wood/Plain), indexed by
    /// [`MaterialChoice`] order.
    material_bind_groups: [wgpu::BindGroup; 3],
    /// The material bind-group layout (texture + sampler). Exposed so a
    /// runtime-loaded VS block (M6) can build a bind group of the SAME shape and
    /// be drawn interchangeably with the procedural materials.
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// The shared material sampler (nearest, clamp-to-edge) — reused by loaded
    /// materials so they slice/filter exactly like the procedural ones.
    material_sampler: wgpu::Sampler,
}

impl VoxelRenderer {
    /// Create the renderer for a given colour target format. The instance buffer
    /// is built from `grid` immediately.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> Self {
        let (vertices, indices) = unit_cube_geometry();
        let cube_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel cube vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let cube_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel cube indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel uniforms"),
            size: std::mem::size_of::<VoxelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel uniform bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Procedural material textures (Stone/Wood/Plain) ---
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("voxel material sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        // M7: the material is a 6-layer texture array (one layer per cube face).
        // A uniform material (the procedural Stone/Wood/Plain, or a VS block with
        // a single `all` texture) puts the same image on all six layers, so the
        // SAME pipeline draws both uniform and genuinely per-face materials.
        let material_bind_group_layout = build_face_material_layout(device);
        let material_bind_groups = [
            generate_stone_texture(),
            generate_wood_texture(),
            generate_plain_texture(),
        ]
        .iter()
        .map(|pixels| {
            // Replicate the single procedural image across all six face layers.
            let layers: [&[u8]; 6] = [pixels, pixels, pixels, pixels, pixels, pixels];
            let texture = upload_face_material_texture(
                device,
                queue,
                MATERIAL_TEXTURE_SIZE,
                MATERIAL_TEXTURE_SIZE,
                &layers,
            );
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("voxel material bind group"),
                layout: &material_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&material_sampler),
                    },
                ],
            })
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("exactly three material textures");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/voxel.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&material_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as u64,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Per-voxel material id (ADR 0001 step 3), indexes the shader's
                // `material_base_colors` array. `u32`: the GPU has no 16-bit
                // vertex format, so the grid's `u16` is widened on upload.
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as u64,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        // The opaque + debug pipelines share everything except the cull mode. The
        // debug pipeline disables culling (`cull_mode: None`) so that if a back
        // face is the nearest surface to the camera (a winding bug), it draws and
        // the shader's `front_facing` marker flags it — culling would otherwise
        // hide the evidence. Depth testing stays on in both, so the nearest face
        // wins. The ghost pipeline (issue #12) alpha-blends with depth writes OFF
        // for the translucent onion-skin fog (see its dedicated builder below).
        let build_pipeline = |label: &str,
                              cull_mode: Option<wgpu::Face>,
                              blend: wgpu::BlendState,
                              depth_write: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[vertex_layout.clone(), instance_layout.clone()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fragment_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth_write),
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
            })
        };
        let pipeline =
            build_pipeline("voxel pipeline", Some(wgpu::Face::Back), wgpu::BlendState::REPLACE, true);
        let debug_pipeline =
            build_pipeline("voxel debug pipeline", None, wgpu::BlendState::REPLACE, true);
        // (Onion skin is a separate volumetric fog pass — see `OnionFogRenderer`
        // — so the voxel renderer no longer needs a translucent ghost pipeline.)

        let mut renderer = Self {
            pipeline,
            debug_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            cube_index_count: indices.len() as u32,
            chunk_buffers: HashMap::new(),
            visible_chunks: Vec::new(),
            last_rebuilt_chunk_count: 0,
            uniform_buffer,
            uniform_bind_group,
            material_bind_groups,
            material_bind_group_layout,
            material_sampler,
        };
        // Build the per-chunk GPU buffer cache from the initial whole grid via the
        // wrapper (it buckets the grid into per-chunk groups and builds one buffer
        // per chunk — issue #20 S6c-2b).
        renderer.rebuild_instances(device, queue, grid, voxels_per_block);
        renderer
    }

    /// The material bind-group layout (texture @ binding 0, sampler @ binding 1).
    /// A loaded VS block builds a bind group against this so it can be bound
    /// exactly like Stone/Wood/Plain (M6).
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_bind_group_layout
    }

    /// The shared material sampler, reused by loaded materials (M6).
    pub fn material_sampler(&self) -> &wgpu::Sampler {
        &self.material_sampler
    }

    /// Number of voxel instances currently drawn across all resident chunk
    /// buffers (issue #20 S6c-2b): the sum of every chunk's instance count.
    pub fn instance_count(&self) -> u32 {
        self.chunk_buffers
            .values()
            .map(|buffers| buffers.instance_count)
            .sum()
    }

    /// Build (or replace) ONE chunk's GPU buffers from its per-chunk
    /// [`VoxelGrid`] (issue #20 S6c-2b), keyed by its ABSOLUTE chunk `coord`. A
    /// chunk that resolves to zero voxels is EVICTED (no buffer allocated); any
    /// previous buffer for `coord` is dropped + replaced. `voxels_per_block` is
    /// unused here (the per-chunk grid is already in render coords) but kept on the
    /// signature for symmetry with the resolve seam and the dirty-rebuild step.
    pub fn rebuild_chunk(
        &mut self,
        device: &wgpu::Device,
        coord: [i32; 3],
        chunk_grid: &VoxelGrid,
    ) {
        match instances_for_chunk(chunk_grid) {
            Some((instances, aabb)) => {
                let instance_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("voxel chunk instances"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    });
                self.chunk_buffers.insert(
                    coord,
                    InstancedChunkBuffers {
                        instance_buffer,
                        instance_count: instances.len() as u32,
                        aabb,
                    },
                );
            }
            // Zero voxels: don't allocate, and drop any stale buffer for this coord.
            None => {
                self.chunk_buffers.remove(&coord);
            }
        }
    }

    /// Evict one chunk's GPU buffers (issue #20 S6c-2b), dropping its instance
    /// buffer. A no-op if the chunk was not resident. Used by the dirty-rebuild
    /// path (S6c-2c) to evict exactly the coords the resolve cache evicted.
    pub fn evict_chunk(&mut self, coord: [i32; 3]) {
        self.chunk_buffers.remove(&coord);
    }

    /// Clear every resident chunk buffer and rebuild from the per-chunk grids the
    /// resolve cache's accessor hands out (issue #20 S6c-2b). This step rebuilds
    /// the WHOLE cache wholesale; the incremental dirty-only rebuild is the next
    /// step (S6c-2c). `chunk_grids` is exactly
    /// `ChunkResolveCache::resident_render_chunks`'s output: `(absolute_chunk_coord,
    /// &rebased_grid)` per covering chunk. Zero-voxel chunks are skipped by
    /// [`rebuild_chunk`].
    pub fn rebuild_all_from_chunks(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        chunk_grids: &[([i32; 3], &VoxelGrid)],
    ) {
        self.chunk_buffers.clear();
        for (coord, grid) in chunk_grids {
            self.rebuild_chunk(device, *coord, grid);
        }
        // A wholesale rebuild touches every covering chunk (the input slice).
        self.last_rebuilt_chunk_count = chunk_grids.len() as u32;
        self.reset_visible_to_all();
    }

    /// **Incremental dirty-chunk rebuild (issue #20 S6c-2c).** Rebuild ONLY the
    /// chunks an edit touched, instead of clearing + rebuilding every per-chunk
    /// buffer. `render_chunks` is the freshly-resolved per-chunk accessor output
    /// (`(absolute_chunk_coord, &rebased_grid)` for every covering chunk, exactly
    /// `ChunkResolveCache::resident_render_chunks`); `evicted` is the set of
    /// absolute chunk-coords the resolve cache evicted for this edit
    /// (`ChunkResolveCache::invalidate_aabb`'s return).
    ///
    /// For each covering chunk this rebuilds its GPU buffer ONLY if the chunk is
    /// DIRTY (`coord ∈ evicted`, so the resolve cache re-resolved it) or NEW (the
    /// GPU cache has no buffer for `coord` yet). A covering chunk that is neither is
    /// a resolve-cache HIT — its rebased grid is byte-identical to what produced the
    /// existing buffer, so the buffer is already correct and is kept untouched.
    /// Then any GPU-cache entry for a coord NOT in `render_chunks` is evicted (a
    /// node removed / shrunk the region, vacating chunks the post-edit scene no
    /// longer covers).
    ///
    /// The result is IDENTICAL to [`rebuild_all_from_chunks`] for the post-edit
    /// scene — same resident coord set, same per-chunk instance contents — but it
    /// only re-uploads the dirty/new chunks. [`last_rebuilt_chunk_count`] records
    /// how many chunks were actually (re)built (dirty ∪ new), for the smoke-test /
    /// `--debug-chunks` readout.
    ///
    /// `render_chunks` is consumed fully (every needed buffer built) BEFORE the
    /// eviction pass, honouring the S6c-2a borrow rule (the slice borrows the cache
    /// immutably for its lifetime).
    ///
    /// [`last_rebuilt_chunk_count`]: VoxelRenderer::last_rebuilt_chunk_count
    pub fn incremental_rebuild_from_chunks(
        &mut self,
        device: &wgpu::Device,
        render_chunks: &[([i32; 3], &VoxelGrid)],
        evicted: &[[i32; 3]],
    ) {
        let resident: Vec<[i32; 3]> = self.chunk_buffers.keys().copied().collect();
        // Only NON-EMPTY covering chunks deserve a buffer; an empty covering chunk is
        // never "new work" and never kept resident (a wholesale rebuild stores none
        // either — `rebuild_chunk` allocates nothing for zero voxels).
        let occupied_covering: Vec<[i32; 3]> = render_chunks
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = incremental_rebuild_plan(&resident, evicted, &occupied_covering);

        // 1. Rebuild only DIRTY (evicted) or NEW (no GPU buffer yet) occupied chunks.
        //    `render_chunks` is consumed fully here, before the eviction pass.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        for (coord, grid) in render_chunks {
            if rebuild_set.contains(coord) {
                self.rebuild_chunk(device, *coord, grid);
            }
        }

        // 2. Evict any GPU-cache entry for a coord that is no longer an occupied
        //    covering chunk — a removed/shrunk node VACATED it, or an edit turned it
        //    EMPTY (both must drop their stale buffer).
        for coord in &plan.evict {
            self.chunk_buffers.remove(coord);
        }

        self.last_rebuilt_chunk_count = plan.rebuild.len() as u32;
        self.reset_visible_to_all();
    }

    /// Rebuild the per-chunk GPU buffer cache FROM a freshly-resolved WHOLE grid
    /// (the wrapper kept for `shot.rs` and tests that have a monolithic grid —
    /// issue #20 S6c-2b). Buckets the whole grid into `CHUNK_BLOCKS³`-block chunks
    /// (ADR 0002 E2) and builds one buffer per chunk via [`rebuild_chunk`], so the
    /// per-chunk buffer contents equal today's per-chunk slices of the monolithic
    /// buffer. The chunk coord key is `floor(world_position / chunk_extent)` — the
    /// same key `bucket_instances_into_chunks` uses.
    pub fn rebuild_instances(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) {
        let chunk_extent = (CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
        // Group the whole grid's voxels into per-chunk sub-grids keyed by integer
        // chunk coord, then build one buffer per chunk. A sub-grid carries only the
        // occupied voxels — `instances_for_chunk` ignores `dimensions`.
        let mut buckets: HashMap<[i32; 3], VoxelGrid> = HashMap::new();
        for voxel in &grid.occupied {
            let key = [
                (voxel.world_position[0] / chunk_extent).floor() as i32,
                (voxel.world_position[1] / chunk_extent).floor() as i32,
                (voxel.world_position[2] / chunk_extent).floor() as i32,
            ];
            buckets
                .entry(key)
                .or_insert_with(|| VoxelGrid::new([0, 0, 0]))
                .occupied
                .push(*voxel);
        }
        self.chunk_buffers.clear();
        for (coord, sub_grid) in &buckets {
            self.rebuild_chunk(device, *coord, sub_grid);
        }
        self.last_rebuilt_chunk_count = self.chunk_buffers.len() as u32;
        self.reset_visible_to_all();
    }

    /// Reset the visible-chunk set to "every resident chunk", so a draw between a
    /// rebuild and the next `update_uniforms` frustum cull shows everything.
    fn reset_visible_to_all(&mut self) {
        self.visible_chunks = self.chunk_buffers.keys().copied().collect();
    }

    /// Total number of resident render chunks (issue #20 S6c-2b).
    pub fn chunk_count(&self) -> u32 {
        self.chunk_buffers.len() as u32
    }

    /// How many per-chunk GPU buffers the LAST rebuild actually (re)built (issue
    /// #20 S6c-2c). After an incremental dirty-chunk rebuild this is `|dirty ∪ new|`
    /// (so a localised edit reads a small number well under
    /// [`chunk_count`](Self::chunk_count)); after a wholesale rebuild it is every
    /// chunk. The smoke-test / `--debug-chunks` readout uses it to confirm an edit
    /// rebuilt only the chunks it touched.
    pub fn last_rebuilt_chunk_count(&self) -> u32 {
        self.last_rebuilt_chunk_count
    }

    /// Number of chunks that survived the last frustum cull (i.e. will be drawn).
    /// Paired with [`VoxelRenderer::chunk_count`] this is the `drew X / Y chunks`
    /// stat for the `--debug-chunks` diagnostic.
    pub fn visible_chunk_count(&self) -> u32 {
        self.visible_chunks.len() as u32
    }

    /// Upload the per-frame uniforms: the camera matrix, the grid half-extent and
    /// density (for the per-voxel slice + overlay), and the grid-overlay toggle.
    ///
    /// `grid_dimensions` are the voxel-space dims of the current grid; the half
    /// extent is `dimensions / 2` so a fragment's `world_pos + half_extent` makes
    /// voxel boundaries fall on integers (BUG 2 fix). `voxels_per_block` is the
    /// current density. `grid_overlay_enabled` reflects the Display toggle.
    /// `debug_face_mode` enables the face-orientation debug shader path (colour by
    /// outward normal + back-facing marker); it must match the pipeline chosen in
    /// [`VoxelRenderer::draw`].
    ///
    /// `material` is the material that will be BOUND in [`VoxelRenderer::draw`]
    /// (it must match). It drives the per-voxel material modulation (ADR 0001 step
    /// 3): for a procedural bound material, each voxel's `material_id` recolours the
    /// shared texture toward that material's tint, so distinct nodes look distinct.
    /// Modulation is OFF in debug-faces mode and for a loaded VS block (which stays
    /// a single global material, per the ADR scope note).
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
        grid_overlay_enabled: bool,
        debug_face_mode: bool,
        band: LayerBand,
        material: MaterialSource<'_>,
    ) {
        // Per-voxel material modulation (ADR 0001 step 3). Only meaningful when a
        // PROCEDURAL material is bound and we are not in debug-faces mode. A loaded
        // VS block stays global, so modulation is off and the bound texture wins.
        let (modulation_enabled, base_colors) = match material {
            MaterialSource::Procedural(bound) if !debug_face_mode => {
                (true, relative_material_base_colors(bound))
            }
            _ => (false, neutral_material_base_colors()),
        };
        let uniforms = VoxelUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            grid_half_extent: [
                grid_dimensions[0] as f32 / 2.0,
                grid_dimensions[1] as f32 / 2.0,
                grid_dimensions[2] as f32 / 2.0,
            ],
            voxels_per_block: voxels_per_block.max(1) as f32,
            voxel_line_color: srgb_hex_to_linear(VOXEL_LINE_COLOR_HEX),
            grid_overlay_enabled: if grid_overlay_enabled { 1.0 } else { 0.0 },
            block_line_color: srgb_hex_to_linear(BLOCK_LINE_COLOR_HEX),
            debug_face_mode: if debug_face_mode { 1.0 } else { 0.0 },
            voxel_line_half_width: VOXEL_LINE_HALF_WIDTH,
            block_line_half_width: BLOCK_LINE_HALF_WIDTH,
            voxel_line_alpha: VOXEL_LINE_ALPHA,
            block_line_alpha: BLOCK_LINE_ALPHA,
            band_min: band.band_min as f32,
            band_max: band.band_max as f32,
            material_modulation_enabled: if modulation_enabled { 1.0 } else { 0.0 },
            _band_pad1: 0.0,
            material_base_colors: base_colors,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // Frustum-cull the resident per-chunk buffers for this frame (ADR 0002 E2):
        // extract the six camera planes from `view_projection` and keep only chunks
        // whose world AABB intersects the frustum. At small scale every chunk is
        // visible → identical output to the old single un-culled draw. The
        // positive-vertex test never produces a false NEGATIVE, so on-screen
        // geometry is never wrongly dropped. The kept coords are sorted so the draw
        // order is deterministic (it is pixel-irrelevant — the pass is opaque +
        // depth-tested — but a stable order keeps `--debug-chunks` reproducible).
        let frustum = Frustum::from_view_projection(view_projection);
        self.visible_chunks.clear();
        for (coord, buffers) in &self.chunk_buffers {
            if frustum.intersects_aabb(&buffers.aabb) {
                self.visible_chunks.push(*coord);
            }
        }
        self.visible_chunks.sort_unstable();
    }

    /// Record the voxel draw into an already-begun render pass.
    ///
    /// The active material is a [`MaterialSource`]: either one of the procedural
    /// textures (Stone/Wood/Plain) or a runtime-loaded VS block bind group (M6).
    /// In both cases the SAME pipeline + per-voxel slice shader run — only the
    /// bound texture differs — so a loaded block textures the model with correct
    /// 1/density slicing, identically to the procedural materials.
    ///
    /// When `debug_face_mode` is true the cull-off debug pipeline is selected (it
    /// must match the `debug_face_mode` flag passed to
    /// [`VoxelRenderer::update_uniforms`]); otherwise the normal back-culled
    /// pipeline runs, leaving the lit/textured output unchanged.
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        material: MaterialSource<'_>,
        debug_face_mode: bool,
    ) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        let material_bind_group = match material {
            MaterialSource::Procedural(choice) => &self.material_bind_groups[material_index(choice)],
            MaterialSource::Loaded(bind_group) => bind_group,
        };
        let pipeline = if debug_face_mode {
            &self.debug_pipeline
        } else {
            &self.pipeline
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, material_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);

        // Draw each frustum-visible chunk from its OWN instance buffer (issue #20
        // S6c-2b): one `draw_indexed` per visible chunk over its full instance
        // range. Every voxel in a visible chunk is drawn — no draw-side truncation
        // — so a scene up to the per-chunk resolve cap renders fully. The instance
        // attributes (position, block-local coord, material_id) and every shader
        // feature carry through per-chunk unchanged; the per-chunk buffer contents
        // equal today's per-chunk slices of the old monolithic buffer, and
        // cross-chunk draw order is pixel-irrelevant (opaque, depth-tested). If
        // `update_uniforms` has not run yet, `visible_chunks` still lists every
        // resident chunk, so nothing is dropped.
        for coord in &self.visible_chunks {
            let Some(buffers) = self.chunk_buffers.get(coord) else {
                continue;
            };
            if buffers.instance_count == 0 {
                continue;
            }
            render_pass.set_vertex_buffer(1, buffers.instance_buffer.slice(..));
            render_pass.draw_indexed(0..self.cube_index_count, 0, 0..buffers.instance_count);
        }
    }
}

/// Which texture the voxel pass binds for the active material.
///
/// `Procedural` selects one of the built-in Stone/Wood/Plain textures;
/// `Loaded` overrides with a runtime-loaded VS block's bind group (M6). Both use
/// the identical pipeline + per-voxel slice shader.
#[derive(Clone, Copy)]
pub enum MaterialSource<'a> {
    Procedural(MaterialChoice),
    Loaded(&'a wgpu::BindGroup),
}

/// Index a [`MaterialChoice`] into the `material_bind_groups` array.
fn material_index(material: MaterialChoice) -> usize {
    match material {
        MaterialChoice::Stone => 0,
        MaterialChoice::Wood => 1,
        MaterialChoice::Plain => 2,
    }
}

/// Build the 6-layer face-material bind-group layout (M7): a `D2Array` texture
/// (binding 0, one layer per cube face) + a sampler (binding 1). Both the
/// procedural materials and a loaded VS block build a bind group of this shape,
/// so the single voxel pipeline draws uniform and per-face materials alike.
pub fn build_face_material_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxel face material bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

/// Upload six RGBA8 sRGB layers (one per cube face) as a single `D2Array`
/// texture (nearest filter, clamp-to-edge, no mipmaps). Every layer must be the
/// same `width`×`height`; callers that have per-face PNGs of differing sizes
/// rescale to a common size first (see `block_palette::upload_face_layers`).
pub fn upload_face_material_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    layers: &[&[u8]; 6],
) -> wgpu::Texture {
    let width = width.max(1);
    let height = height.max(1);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 6,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel face material texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB so the GPU decodes samples to linear; lighting + the grid overlay
        // then run in linear space and the sRGB target re-encodes on write.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer_index, layer_pixels) in layers.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer_index as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            layer_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture
}

/// A small deterministic value-noise generator so the procedural textures are
/// stable across runs (the prototype used `Math.random`; we want reproducible
/// screenshots). Returns a float in `[0, 1)`.
struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_unit(&mut self) -> f32 {
        // Numerical Recipes LCG constants.
        self.state = self.state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.state >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// Pack three components into an opaque RGBA8 pixel (alpha = 255).
fn rgba(r: f32, g: f32, b: f32) -> [u8; 4] {
    [
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
        255,
    ]
}

/// Stone: 32×32 grey ~rgb(132,126,118) with ±20 per-pixel noise + darker speckles.
/// Port of `makeStone` (chisel-bench-reference.html).
fn generate_stone_texture() -> Vec<u8> {
    let mut rng = Lcg::new(0x5701_3a9f);
    let count = (MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE) as usize;
    let mut pixels = vec![0u8; count * 4];
    // The prototype iterates i (x) outer, j (y) inner, filling column-major; the
    // exact per-pixel correspondence is cosmetic noise, so we fill row-major.
    for pixel in pixels.chunks_exact_mut(4) {
        let noise = 132.0 + (rng.next_unit() * 40.0 - 20.0).floor();
        pixel.copy_from_slice(&rgba(noise, noise - 6.0, noise - 14.0));
    }
    // ~22 darker speckles.
    for _ in 0..22 {
        let x = (rng.next_unit() * MATERIAL_TEXTURE_SIZE as f32) as u32;
        let y = (rng.next_unit() * MATERIAL_TEXTURE_SIZE as f32) as u32;
        let dark = 90.0 + (rng.next_unit() * 30.0).floor();
        let index = ((y.min(MATERIAL_TEXTURE_SIZE - 1) * MATERIAL_TEXTURE_SIZE
            + x.min(MATERIAL_TEXTURE_SIZE - 1))
            * 4) as usize;
        pixels[index..index + 4].copy_from_slice(&rgba(dark, dark - 8.0, dark - 16.0));
    }
    pixels
}

/// Wood: 32×32 brown base with a horizontal sine grain + per-pixel noise.
/// Port of `makeWood` (chisel-bench-reference.html).
fn generate_wood_texture() -> Vec<u8> {
    let mut rng = Lcg::new(0x00c0_ffee);
    let mut pixels = Vec::with_capacity((MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE * 4) as usize);
    for row in 0..MATERIAL_TEXTURE_SIZE {
        let grain = (row as f32 * 0.9).sin() * 10.0 + (rng.next_unit() * 10.0 - 5.0);
        for _ in 0..MATERIAL_TEXTURE_SIZE {
            let red = 120.0 + grain + (rng.next_unit() * 8.0 - 4.0);
            pixels.extend_from_slice(&rgba(red.floor(), (red * 0.62).floor(), (red * 0.34).floor()));
        }
    }
    pixels
}

/// Plain: flat warm grey `#b6a079`. Port of `makePlain`.
fn generate_plain_texture() -> Vec<u8> {
    let count = (MATERIAL_TEXTURE_SIZE * MATERIAL_TEXTURE_SIZE) as usize;
    let mut pixels = Vec::with_capacity(count * 4);
    for _ in 0..count {
        pixels.extend_from_slice(&[0xb6, 0xa0, 0x79, 0xff]);
    }
    pixels
}

/// The average RGBA colour of a procedural material's texture — the
/// representative palette colour used by the `.vox` export (M8). A loaded VS
/// block can supply its own average instead; this covers the procedural case.
pub fn procedural_material_average_color(material: MaterialChoice) -> [u8; 4] {
    let pixels = match material {
        MaterialChoice::Stone => generate_stone_texture(),
        MaterialChoice::Wood => generate_wood_texture(),
        MaterialChoice::Plain => generate_plain_texture(),
    };
    let mut sums = [0u64; 3];
    let count = (pixels.len() / 4) as u64;
    for pixel in pixels.chunks_exact(4) {
        sums[0] += pixel[0] as u64;
        sums[1] += pixel[1] as u64;
        sums[2] += pixel[2] as u64;
    }
    let count = count.max(1);
    [
        (sums[0] / count) as u8,
        (sums[1] / count) as u8,
        (sums[2] / count) as u8,
        255,
    ]
}

/// The average colour of a material's procedural texture as a LINEAR `[r, g, b]`
/// (the space the shader lights/blends in). Indexed by `material_id` order
/// (Stone/Wood/Plain) via [`MaterialChoice::from_material_id`].
fn material_average_linear(id: u16) -> [f32; 3] {
    let srgb = procedural_material_average_color(MaterialChoice::from_material_id(id));
    [
        srgb_component_to_linear(srgb[0]),
        srgb_component_to_linear(srgb[1]),
        srgb_component_to_linear(srgb[2]),
    ]
}

/// The neutral (identity) per-material base-colour array: every slot is white, so
/// modulation by it is a no-op. Used when modulation is OFF (debug-faces or a
/// loaded VS block), so the uniform is always well-defined.
fn neutral_material_base_colors() -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT]
}

/// The per-voxel material base colours (ADR 0001 step 3) RELATIVE to the bound
/// texture's own average colour. Slot `id` holds `avg(id) / avg(bound)`, so:
///   * the bound material's own slot is ~`[1,1,1]` (neutral — its texture is
///     shown unchanged, preserving the existing look for a single-material model);
///   * every other material's slot recolours the shared bound texture toward that
///     material's tint, so a Wood node and a Stone node drawn from one bound
///     texture render in visibly distinct colours.
///
/// This is the cheap base-colour-modulation the ADR/task call for, NOT a
/// per-material texture array.
fn relative_material_base_colors(
    bound: MaterialChoice,
) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let bound_avg = material_average_linear(bound.material_id());
    let mut colors = [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT];
    for (id, slot) in colors.iter_mut().enumerate() {
        let avg = material_average_linear(id as u16);
        // Guard against a near-zero bound channel (a flat black texture); fall back
        // to a neutral 1.0 so a divide can't explode.
        for axis in 0..3 {
            slot[axis] = if bound_avg[axis] > 1e-4 {
                avg[axis] / bound_avg[axis]
            } else {
                1.0
            };
        }
    }
    colors
}

/// Public access to the per-material relative base colours (step 3b) for the
/// flag-gated cuboid mesh path (ADR 0002 E3b-1), so it modulates per-box material
/// colour with the EXACT same array the instanced path uses. Returns each
/// material's average colour relative to `bound`'s average (the bound material's
/// own slot is ~neutral white).
pub fn relative_material_base_colors_public(
    bound: MaterialChoice,
) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    relative_material_base_colors(bound)
}

/// The grid-overlay tuning the instanced voxel pass uses, exposed so the
/// flag-gated cuboid mesh path (ADR 0002 E3b-2) draws the position-based grid
/// overlay with the EXACT same colours/half-widths/alphas — keeping the merged
/// box faces phase-aligned to the same per-voxel/per-block lines.
#[derive(Debug, Clone, Copy)]
pub struct GridOverlayParams {
    pub voxel_line_color: [f32; 3],
    pub block_line_color: [f32; 3],
    pub voxel_line_half_width: f32,
    pub block_line_half_width: f32,
    pub voxel_line_alpha: f32,
    pub block_line_alpha: f32,
}

/// The instanced path's grid-overlay parameters (colours in LINEAR space, the
/// same the voxel shader receives), for the cuboid path to reuse verbatim.
pub fn grid_overlay_params() -> GridOverlayParams {
    GridOverlayParams {
        voxel_line_color: srgb_hex_to_linear(VOXEL_LINE_COLOR_HEX),
        block_line_color: srgb_hex_to_linear(BLOCK_LINE_COLOR_HEX),
        voxel_line_half_width: VOXEL_LINE_HALF_WIDTH,
        block_line_half_width: BLOCK_LINE_HALF_WIDTH,
        voxel_line_alpha: VOXEL_LINE_ALPHA,
        block_line_alpha: BLOCK_LINE_ALPHA,
    }
}

/// Generate the three procedural material textures (Stone/Wood/Plain) as RGBA8
/// sRGB pixel buffers, in `MaterialChoice` order, so the cuboid path (E3b-2) can
/// upload the SAME procedural textures the instanced path binds.
pub fn procedural_material_pixels() -> [Vec<u8>; 3] {
    [
        generate_stone_texture(),
        generate_wood_texture(),
        generate_plain_texture(),
    ]
}

/// The edge length of every procedural material texture (square), exposed so the
/// cuboid path uploads them at the matching size.
pub fn procedural_material_texture_size() -> u32 {
    MATERIAL_TEXTURE_SIZE
}

// ============================================================================
// View cube (Milestone 5) — ARCHITECTURE.md §4.
// ============================================================================

/// Edge length (pixels) of the corner view-cube viewport (top-left).
pub const VIEW_CUBE_VIEWPORT_PIXELS: u32 = 128;
/// Margin (pixels) from the top-left corner to the viewport.
pub const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;
/// Edge length of each square face-label texture.
const FACE_LABEL_TEXTURE_SIZE: u32 = 128;

/// One view-cube vertex: position, face normal, face UV, and the texture-array
/// layer (face index in +X,-X,+Y,-Y,+Z,-Z order).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeLabelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    layer: u32,
}

/// The corner view cube: a labelled cube mirroring the main camera, plus a teal
/// edge wireframe (ARCHITECTURE.md §4). Rendered into a scissored top-left
/// viewport in its own pass (depth cleared there first).
pub struct ViewCubeRenderer {
    face_pipeline: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    edge_buffer: wgpu::Buffer,
    edge_vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    label_bind_group: wgpu::BindGroup,
}

impl ViewCubeRenderer {
    /// Create the view-cube renderer for a colour target format.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, color_format: wgpu::TextureFormat) -> Self {
        let (vertices, indices) = view_cube_geometry();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let edges = view_cube_edges();
        let edge_vertex_count = edges.len() as u32;
        let edge_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view cube uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            cube_uniform_bind_group(device, &uniform_buffer);

        // --- 6-layer face-label texture array ---
        let label_pixels = generate_face_label_textures();
        let label_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view cube label textures"),
            size: wgpu::Extent3d {
                width: FACE_LABEL_TEXTURE_SIZE,
                height: FACE_LABEL_TEXTURE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &label_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &label_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * FACE_LABEL_TEXTURE_SIZE),
                rows_per_image: Some(FACE_LABEL_TEXTURE_SIZE),
            },
            wgpu::Extent3d {
                width: FACE_LABEL_TEXTURE_SIZE,
                height: FACE_LABEL_TEXTURE_SIZE,
                depth_or_array_layers: 6,
            },
        );
        let label_view = label_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let label_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("view cube label sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let label_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view cube label layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let label_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view cube label bind group"),
            layout: &label_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&label_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&label_sampler),
                },
            ],
        });

        // --- Face pipeline (textured cube) ---
        let cube_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("view cube shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/viewcube.wgsl").into()),
        });
        let face_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("view cube face pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&label_bind_group_layout)],
            immediate_size: 0,
        });
        let cube_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CubeLabelVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Uint32 },
            ],
        };
        // The view cube renders at 1 sample into the resolved target (after the
        // 3D MSAA resolve, before egui), so its pipelines use sample_count 1.
        let face_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("view cube face pipeline"),
            layout: Some(&face_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cube_shader,
                entry_point: Some("vertex_main"),
                buffers: &[cube_vertex_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cube_shader,
                entry_point: Some("fragment_main"),
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
                cull_mode: Some(wgpu::Face::Back),
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
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });

        // --- Edge pipeline (teal wireframe, 1 sample, depth-tested) ---
        let edge_pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "view cube edge",
            true,
            1,
        );

        Self {
            face_pipeline,
            edge_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            edge_buffer,
            edge_vertex_count,
            uniform_buffer,
            uniform_bind_group,
            label_bind_group,
        }
    }

    /// Upload the view-cube camera matrix (`OrbitCamera::view_cube_view_projection`).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms { view_projection: view_projection.to_cols_array_2d() };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the cube into a scissored corner of `target_view` (its own render pass,
    /// with a freshly-cleared private depth texture). The colour attachment loads
    /// the already-resolved scene so only the corner is touched.
    ///
    /// Issue #25: the corner is the top-left of the CENTRAL 3D viewport rect
    /// (`viewport_x/y/w/h`, physical pixels), NOT the whole window — so the cube
    /// lines up with the visible 3D area instead of hiding behind the side panel.
    /// `target_width/height` are the full target dims (the colour + depth
    /// attachments span the whole target; the scissor confines the draw).
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
        viewport: [u32; 4],
    ) {
        let [viewport_x, viewport_y, viewport_width, viewport_height] = viewport;
        let margin = VIEW_CUBE_VIEWPORT_MARGIN;
        let size = VIEW_CUBE_VIEWPORT_PIXELS;
        // Bail if the central viewport is too small to host the corner cube.
        if viewport_width < margin + size || viewport_height < margin + size {
            return;
        }
        // The cube's top-left corner, offset into the central viewport.
        let corner_x = viewport_x + margin;
        let corner_y = viewport_y + margin;
        // Bail if the cube would fall outside the actual target (defensive).
        if corner_x + size > target_width || corner_y + size > target_height {
            return;
        }
        // The depth attachment must match the colour attachment's size, so this
        // transient single-sample depth texture spans the whole target; the
        // scissor/viewport still confine the cube to the top-left corner.
        let depth_texture =
            create_single_sample_depth_view(device, target_width, target_height);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("view cube pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // Load the resolved scene; the scissor confines our writes to
                    // the corner so the rest of the frame is untouched.
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_viewport(corner_x as f32, corner_y as f32, size as f32, size as f32, 0.0, 1.0);
        pass.set_scissor_rect(corner_x, corner_y, size, size);

        pass.set_pipeline(&self.face_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, &self.label_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);

        pass.set_pipeline(&self.edge_pipeline);
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_vertex_buffer(0, self.edge_buffer.slice(..));
        pass.draw(0..self.edge_vertex_count, 0..1);
    }
}

/// Uniform bind group for the view cube (binding 0 = view-projection).
fn cube_uniform_bind_group(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("view cube uniform layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view cube uniform bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    (layout, bind_group)
}

/// Build the labelled-cube geometry (side 1.4, centred on origin). Face order +X,
/// -X, +Y, -Y, +Z, -Z (matches `materialIndex` / `CubeFace`).
fn view_cube_geometry() -> (Vec<CubeLabelVertex>, Vec<u16>) {
    const HALF: f32 = 0.7; // side 1.4
    const UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [[HALF, -HALF, HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [HALF, HALF, HALF]]),
        ([-1.0, 0.0, 0.0], [[-HALF, -HALF, -HALF], [-HALF, -HALF, HALF], [-HALF, HALF, HALF], [-HALF, HALF, -HALF]]),
        ([0.0, 1.0, 0.0], [[-HALF, HALF, HALF], [HALF, HALF, HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF]]),
        ([0.0, -1.0, 0.0], [[-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, -HALF, HALF], [-HALF, -HALF, HALF]]),
        ([0.0, 0.0, 1.0], [[-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF]]),
        ([0.0, 0.0, -1.0], [[HALF, -HALF, -HALF], [-HALF, -HALF, -HALF], [-HALF, HALF, -HALF], [HALF, HALF, -HALF]]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (layer, (normal, corners)) in faces.iter().enumerate() {
        let base = vertices.len() as u16;
        for (corner_index, corner) in corners.iter().enumerate() {
            vertices.push(CubeLabelVertex {
                position: *corner,
                normal: *normal,
                uv: UVS[corner_index],
                layer: layer as u32,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// Teal wireframe edges (12 cube edges) for the view cube.
fn view_cube_edges() -> Vec<LineVertex> {
    const HALF: f32 = 0.705; // a hair outside the faces so the edges read crisply
    let color = with_alpha(srgb_hex_to_linear(0x5f_b8_a4), 1.0);
    let corners = [
        [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF],
        [-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF],
    ];
    let edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), // back face
        (4, 5), (5, 6), (6, 7), (7, 4), // front face
        (0, 4), (1, 5), (2, 6), (3, 7), // connecting
    ];
    let mut vertices = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        vertices.push(LineVertex { position: corners[a], color });
        vertices.push(LineVertex { position: corners[b], color });
    }
    vertices
}

/// Render the six face-label textures (RIGHT/LEFT/TOP/BOTTOM/FRONT/BACK) into one
/// stacked RGBA8 buffer (6 layers, in `materialIndex` order). Each is a dark
/// warm panel `#241d15` with a teal `#5fb8a4` border and parchment `#e9e1d1`
/// text, transcribed from the prototype `faceTex`.
fn generate_face_label_textures() -> Vec<u8> {
    const LABELS: [&str; 6] = ["RIGHT", "LEFT", "TOP", "BOTTOM", "FRONT", "BACK"];
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let mut all = Vec::with_capacity(size * size * 4 * 6);
    for label in LABELS {
        all.extend_from_slice(&render_face_label(label));
    }
    all
}

/// Render one face-label texture (RGBA8, `FACE_LABEL_TEXTURE_SIZE` square).
fn render_face_label(label: &str) -> Vec<u8> {
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    const BACKGROUND: [u8; 4] = [0x24, 0x1d, 0x15, 0xff];
    const BORDER: [u8; 4] = [0x5f, 0xb8, 0xa4, 0xff];
    const TEXT: [u8; 4] = [0xe9, 0xe1, 0xd1, 0xff];

    let mut pixels = vec![0u8; size * size * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&BACKGROUND);
    }
    // Teal border (7px, inset 4px) like the prototype `strokeRect(4,4,120,120)`.
    let border_inset = 4usize;
    let border_thickness = 7usize;
    let put = |pixels: &mut [u8], x: usize, y: usize, color: [u8; 4]| {
        if x < size && y < size {
            let index = (y * size + x) * 4;
            pixels[index..index + 4].copy_from_slice(&color);
        }
    };
    for offset in 0..border_thickness {
        let lo = border_inset + offset;
        let hi = size - 1 - border_inset - offset;
        for c in border_inset..(size - border_inset) {
            put(&mut pixels, c, lo, BORDER);
            put(&mut pixels, c, hi, BORDER);
            put(&mut pixels, lo, c, BORDER);
            put(&mut pixels, hi, c, BORDER);
        }
    }

    // Centred bitmap text.
    draw_centered_label(&mut pixels, size, label, TEXT);
    pixels
}

/// Draw `label` centred using the built-in 5×7 bitmap font, scaled to fill the
/// face, into the RGBA8 `pixels` buffer.
fn draw_centered_label(pixels: &mut [u8], size: usize, label: &str, color: [u8; 4]) {
    let glyph_width = 5usize;
    let glyph_height = 7usize;
    let spacing = 1usize;
    let count = label.chars().count().max(1);
    let text_cells_wide = count * glyph_width + (count - 1) * spacing;
    // Choose an integer scale that fits within ~80% of the face width/height.
    let max_scale_w = (size * 8 / 10) / text_cells_wide.max(1);
    let max_scale_h = (size * 5 / 10) / glyph_height;
    let scale = max_scale_w.min(max_scale_h).max(1);

    let text_pixel_width = text_cells_wide * scale;
    let text_pixel_height = glyph_height * scale;
    let origin_x = (size.saturating_sub(text_pixel_width)) / 2;
    let origin_y = (size.saturating_sub(text_pixel_height)) / 2;

    let mut cursor_x = origin_x;
    for ch in label.chars() {
        let glyph = glyph_bitmap(ch);
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..glyph_width {
                if (bits >> (glyph_width - 1 - col)) & 1 == 1 {
                    // Filled cell → scale×scale block.
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let x = cursor_x + col * scale + dx;
                            let y = origin_y + row * scale + dy;
                            if x < size && y < size {
                                let index = (y * size + x) * 4;
                                pixels[index..index + 4].copy_from_slice(&color);
                            }
                        }
                    }
                }
            }
        }
        cursor_x += (glyph_width + spacing) * scale;
    }
}

/// A 5×7 bitmap (7 rows of 5-bit masks) for the uppercase letters used by the
/// face labels. Unknown characters render blank.
fn glyph_bitmap(ch: char) -> [u8; 7] {
    match ch {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        _ => [0; 7],
    }
}

/// Create a single-sample depth texture view (used by the view-cube pass).
fn create_single_sample_depth_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("view cube depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create a 4-sample (MSAA) colour texture view for the 3D pass, sized to a
/// render target. Recreated on window resize / created at the offscreen size for
/// the headless capture. `format` matches the resolve target.
pub fn create_msaa_color_view(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel msaa color texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

// ============================================================================
// Transform gizmo (Milestone 5 origin gizmo, repurposed in issue #29 S2) —
// ARCHITECTURE.md §5.
// ============================================================================

/// X axis colour `#d9603f` (sRGB hex → linear).
const GIZMO_AXIS_X_HEX: u32 = 0xd9_60_3f;
/// Y axis colour `#6fcf5f`.
const GIZMO_AXIS_Y_HEX: u32 = 0x6f_cf_5f;
/// Z axis colour `#5a8cff`.
const GIZMO_AXIS_Z_HEX: u32 = 0x5a_8c_ff;
/// Right-angle square colour `#bdb39a`.
const GIZMO_SQUARE_HEX: u32 = 0xbd_b3_9a;

/// One coloured line-segment vertex (position + linear RGBA colour). The alpha
/// lets the M8 block lattice / floor grid draw at low opacity through the same
/// alpha-blending line pipeline the gizmo / view-cube edges use (those pass 1.0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// Camera uniform for the line passes (gizmo + view-cube edges): just the
/// view-projection matrix.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct LineUniforms {
    view_projection: [[f32; 4]; 4],
}

/// The transform gizmo (issue #29 S2): three coloured axis lines and three
/// perpendicular square line-loops, drawn with **depth-test disabled** so it
/// shows through a solid model (correct manipulator behavior — ARCHITECTURE.md
/// §5). Drawn in the MSAA pass, after the voxels. Unlike the old origin gizmo it
/// FOLLOWS the selected node: its pivot translation is baked into the uploaded
/// view-projection (`view_projection · translate(pivot)`) so it sits ON the
/// object, and it is sized from the selected node's own extent. The axis-triad
/// geometry is kept for now; full TRS handles are future work.
pub struct TransformGizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl TransformGizmoRenderer {
    /// Create the transform gizmo renderer for a colour target format.
    /// `grid_dimensions` sizes the gizmo (`L = max(dims) * 0.62`); the caller
    /// rebuilds it to the SELECTED node's extent each frame.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid_dimensions: [u32; 3],
    ) -> Self {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        let vertex_capacity = vertex_count.max(1);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(vertices, vertex_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gizmo uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "gizmo");

        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "gizmo",
            // Depth-test OFF (Always, no write) so the gizmo shows through solids.
            false,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            vertex_buffer,
            vertex_count,
            vertex_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Resize the gizmo to a freshly-resolved grid (matches the voxel rebuild).
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid_dimensions: [u32; 3]) {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        if vertex_count <= self.vertex_capacity {
            if vertex_count > 0 {
                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        } else {
            self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gizmo line vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.vertex_capacity = vertex_count;
        }
        self.vertex_count = vertex_count;
    }

    /// Upload the camera matrix with the selected node's `pivot` translation baked
    /// in (issue #29 S2): the shader does `view_projection · position`, so feeding
    /// `view_projection · translate(pivot)` here moves the whole gizmo onto the
    /// selected node WITHOUT touching the shared `LineUniforms` layout. `pivot` is
    /// in the SAME recentred frame as the voxels, so the gizmo sits on the object.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        pivot: glam::Vec3,
    ) {
        let model = glam::Mat4::from_translation(pivot);
        let uniforms = LineUniforms {
            view_projection: (view_projection * model).to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the gizmo draw into an already-begun (MSAA) render pass.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}

/// Build the gizmo line vertices (axes + perpendicular squares), in world space.
fn gizmo_vertices(grid_dimensions: [u32; 3]) -> Vec<LineVertex> {
    let longest = grid_dimensions[0]
        .max(grid_dimensions[1])
        .max(grid_dimensions[2]) as f32;
    let axis_length = (longest * 0.62).max(1.0);
    let square_side = axis_length * 0.28;

    let x_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_X_HEX), 1.0);
    let y_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), 1.0);
    let z_color = with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Z_HEX), 1.0);
    let square_color = with_alpha(srgb_hex_to_linear(GIZMO_SQUARE_HEX), 1.0);

    let mut vertices = Vec::new();
    let mut line = |from: [f32; 3], to: [f32; 3], color: [f32; 4]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Three axes from the origin.
    line([0.0, 0.0, 0.0], [axis_length, 0.0, 0.0], x_color);
    line([0.0, 0.0, 0.0], [0.0, axis_length, 0.0], y_color);
    line([0.0, 0.0, 0.0], [0.0, 0.0, axis_length], z_color);

    let s = square_side;
    // Square line-loops (closed) in the XY, YZ and ZX planes (prototype `sq`).
    let loop_segments = |points: &[[f32; 3]], color: [f32; 4], out: &mut Vec<LineVertex>| {
        for pair in points.windows(2) {
            out.push(LineVertex { position: pair[0], color });
            out.push(LineVertex { position: pair[1], color });
        }
    };
    loop_segments(
        &[[0.0, 0.0, 0.0], [s, 0.0, 0.0], [s, s, 0.0], [0.0, s, 0.0], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    loop_segments(
        &[[0.0, 0.0, 0.0], [0.0, s, 0.0], [0.0, s, s], [0.0, 0.0, s], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    loop_segments(
        &[[0.0, 0.0, 0.0], [0.0, 0.0, s], [s, 0.0, s], [s, 0.0, 0.0], [0.0, 0.0, 0.0]],
        square_color,
        &mut vertices,
    );
    vertices
}

/// Pad a line-vertex list to `capacity` with zeroed (degenerate) vertices.
fn pad_lines(mut vertices: Vec<LineVertex>, capacity: u32) -> Vec<LineVertex> {
    if (vertices.len() as u32) < capacity {
        vertices.resize(
            capacity as usize,
            LineVertex { position: [0.0; 3], color: [0.0; 4] },
        );
    }
    vertices
}

/// Build the shared uniform bind group (binding 0 = `LineUniforms`) for a line pass.
fn line_uniform_bind_group(
    device: &wgpu::Device,
    uniform_buffer: &wgpu::Buffer,
    label: &str,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label} line uniform layout")),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label} line uniform bind group")),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });
    (layout, bind_group)
}

/// Build a `LineList` render pipeline (shared shader `line.wgsl`). `depth_tested`
/// selects whether the pass writes/tests depth; the gizmo passes `false`
/// (depth-test off so it shows through solids).
fn build_line_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    uniform_bind_group_layout: &wgpu::BindGroupLayout,
    label: &str,
    depth_tested: bool,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/line.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} line pipeline layout")),
        bind_group_layouts: &[Some(uniform_bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<LineVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: std::mem::size_of::<[f32; 3]>() as u64,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label} line pipeline")),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex_main"),
            buffers: &[vertex_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            // Depth-test off (Always + no write) makes the gizmo show through the
            // model; depth-test on uses standard Less for the in-cube edges.
            depth_write_enabled: Some(depth_tested),
            depth_compare: Some(if depth_tested {
                wgpu::CompareFunction::Less
            } else {
                wgpu::CompareFunction::Always
            }),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}

// ============================================================================
// Block lattice + fine floor grid (Milestone 8) — prototype `buildGrids`.
// ============================================================================

/// Block lattice colour `#5fb8a4` (teal patina) at ~0.28 alpha.
const LATTICE_COLOR_HEX: u32 = 0x5f_b8_a4;
const LATTICE_ALPHA: f32 = 0.28;
/// Floor grid colour `#b8a47a` (warm sand) at 0.55 alpha. Issue #29 fix: the
/// floor grid was previously a very dim `#6b5f4a` at 0.16 alpha — coincident with
/// the model's depth-tested base plane and near-black against the background, so
/// it read as "nothing" when toggled on. A brighter colour at a lattice-comparable
/// opacity makes the base-plane grid clearly visible (it still hugs the node's
/// enclosing-block XZ footprint, snapped to the global block lattice).
const FLOOR_COLOR_HEX: u32 = 0xb8_a4_7a;
const FLOOR_ALPHA: f32 = 0.55;

/// The per-object block lattice and floor grid (ARCHITECTURE.md §6 / prototype
/// `buildGrids`), drawn through the shared alpha-blended, depth-tested line
/// pipeline in the MSAA pass.
///
/// Issue #29 S3: this is no longer ONE whole-region lattice. Each frame the caller
/// walks the scene and, for every node whose grids are enabled (the scene master
/// ANDed with the node's own toggle), appends that node's block lattice and/or
/// floor lines into the renderer's per-frame batch via [`Self::set_batch`]. A
/// lattice box is a 3D box lattice with lines at every BLOCK boundary (spacing =
/// density) spanning the node's enclosing-block AABB; the floor is the horizontal
/// grid at the node's base plane, snapped to the same global block lines.
pub struct SceneGridRenderer {
    pipeline: wgpu::RenderPipeline,
    lattice_buffer: wgpu::Buffer,
    lattice_vertex_count: u32,
    lattice_capacity: u32,
    floor_buffer: wgpu::Buffer,
    floor_vertex_count: u32,
    floor_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl SceneGridRenderer {
    /// Create the renderer for a colour target. The line batches start empty —
    /// the caller fills them each frame via [`Self::set_batch`] from the visible
    /// nodes' enabled grids.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let lattice_capacity = 1u32;
        let floor_capacity = 1u32;

        let lattice_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lattice line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), lattice_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let floor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("floor line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), floor_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lattice uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "lattice");

        // Depth-tested (true) so the lattice/floor are occluded by the solid model
        // — they read as a scaffold around/under it, not an overlay on top.
        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "lattice",
            true,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            lattice_buffer,
            lattice_vertex_count: 0,
            lattice_capacity,
            floor_buffer,
            floor_vertex_count: 0,
            floor_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Rebuild this frame's lattice + floor line batches by walking `scene` (issue
    /// #29 S3). For every visible node whose grids are enabled — the scene-wide
    /// master ANDed with that node's own per-object toggle — the node's
    /// enclosing-block lattice box ([`Scene::node_block_lattice_box_recentred`]) is
    /// appended to the corresponding batch:
    ///
    /// * `master_block_lattice && node.grids.block_lattice` → block lattice lines.
    /// * `master_floor_grid && node.grids.floor_grid` → base-plane floor lines.
    ///
    /// A node with no intrinsic extent (size-less Part / empty subtree) yields no
    /// box and is skipped. When NOTHING is enabled both batches are empty and
    /// [`Self::draw`] becomes a no-op — the new default, where per-object grids are
    /// off until the user turns them on.
    pub fn rebuild_from_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
    ) {
        let step = voxels_per_block.max(1);
        let (lattice_boxes, floor_boxes) = scene_grid_boxes(scene, voxels_per_block);
        let mut lattice: Vec<LineVertex> = Vec::new();
        let mut floor: Vec<LineVertex> = Vec::new();
        for (min, max) in lattice_boxes {
            lattice_vertices_into(&mut lattice, min, max, step);
        }
        for (min, max) in floor_boxes {
            floor_vertices_into(&mut floor, min, max, step);
        }
        self.lattice_vertex_count = upload_lines(
            device,
            queue,
            &mut self.lattice_buffer,
            &mut self.lattice_capacity,
            lattice,
            "lattice line vertices",
        );
        self.floor_vertex_count = upload_lines(
            device,
            queue,
            &mut self.floor_buffer,
            &mut self.floor_capacity,
            floor,
            "floor line vertices",
        );
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the lattice + floor draws into an already-begun (MSAA) pass. Gating
    /// is done at batch-build time (issue #29 S3): only grid-enabled nodes
    /// contributed lines, so empty batches simply draw nothing here.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.lattice_vertex_count == 0 && self.floor_vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        if self.lattice_vertex_count > 0 {
            render_pass.set_vertex_buffer(0, self.lattice_buffer.slice(..));
            render_pass.draw(0..self.lattice_vertex_count, 0..1);
        }
        if self.floor_vertex_count > 0 {
            render_pass.set_vertex_buffer(0, self.floor_buffer.slice(..));
            render_pass.draw(0..self.floor_vertex_count, 0..1);
        }
    }
}

/// Write a line-vertex list to `buffer`, growing it if needed; returns the count.
fn upload_lines(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut wgpu::Buffer,
    capacity: &mut u32,
    vertices: Vec<LineVertex>,
    label: &str,
) -> u32 {
    let count = vertices.len() as u32;
    if count <= *capacity {
        if count > 0 {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
        }
    } else {
        *buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        *capacity = count;
    }
    count
}

/// The per-object grid boxes for a scene (issue #29 S3), gated CPU-side so the walk
/// is unit-testable without a GPU. Returns `(lattice_boxes, floor_boxes)` where each
/// box is the `(min, max)` enclosing-block AABB (recentred voxels) of a node whose
/// grid is enabled — the scene-wide master ANDed with the node's own per-object
/// toggle. A node with no intrinsic extent contributes no box. When a master is off,
/// or a node's flag is off, that node contributes nothing to that batch (gating).
#[allow(clippy::type_complexity)]
pub(crate) fn scene_grid_boxes(
    scene: &Scene,
    voxels_per_block: u32,
) -> (Vec<([f32; 3], [f32; 3])>, Vec<([f32; 3], [f32; 3])>) {
    let mut lattice_boxes = Vec::new();
    let mut floor_boxes = Vec::new();
    let want_lattice_master = scene.master_block_lattice;
    let want_floor_master = scene.master_floor_grid;
    if !want_lattice_master && !want_floor_master {
        return (lattice_boxes, floor_boxes);
    }
    for (path, _depth) in scene.tree_rows() {
        let Some(node) = scene.node_at_path(&path) else {
            continue;
        };
        let want_lattice = want_lattice_master && node.grids.block_lattice;
        let want_floor = want_floor_master && node.grids.floor_grid;
        if !want_lattice && !want_floor {
            continue;
        }
        let Some(node_box) = scene.node_block_lattice_box_recentred(&path, voxels_per_block) else {
            continue;
        };
        if want_lattice {
            lattice_boxes.push(node_box);
        }
        if want_floor {
            floor_boxes.push(node_box);
        }
    }
    (lattice_boxes, floor_boxes)
}

/// Block-boundary coordinates `[lo, lo+step, …, hi]` along one axis. The corners
/// `lo`/`hi` are block-aligned (the caller supplies an enclosing-block box), so the
/// `step`-stride walk lands exactly on `hi`; a final clamp guards float drift so the
/// closing block plane is always present.
fn block_boundaries(lo: f32, hi: f32, step: u32) -> Vec<f32> {
    let step = step.max(1) as f32;
    let mut values = Vec::new();
    let mut g = lo;
    // `+ step * 0.5` tolerance: include the plane at (or fractionally past) `hi`.
    while g <= hi + step * 0.5 {
        values.push(g.min(hi));
        g += step;
    }
    if values.last().copied() != Some(hi) {
        values.push(hi);
    }
    values
}

/// Append a 3D block lattice for the box `[min, max]` (voxels) — grid lines at every
/// BLOCK boundary (spacing = `step`) — into `vertices` (issue #29 S3, per-object).
/// Port of the prototype `buildGrids` lattice loop, now spanning an arbitrary box.
fn lattice_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let color = with_alpha(srgb_hex_to_linear(LATTICE_COLOR_HEX), LATTICE_ALPHA);
    let xs = block_boundaries(min[0], max[0], step);
    let ys = block_boundaries(min[1], max[1], step);
    let zs = block_boundaries(min[2], max[2], step);

    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Lines along Y at every (x, z) lattice node.
    for &x in &xs {
        for &z in &zs {
            add([x, min[1], z], [x, max[1], z]);
        }
    }
    // Lines along X at every (y, z) lattice node.
    for &y in &ys {
        for &z in &zs {
            add([min[0], y, z], [max[0], y, z]);
        }
    }
    // Lines along Z at every (x, y) lattice node.
    for &x in &xs {
        for &y in &ys {
            add([x, y, min[2]], [x, y, max[2]]);
        }
    }
}

/// How far BELOW the node's base plane the floor grid sits, in voxels (issue #29
/// fix). The enclosing-block box bottom is coincident with the model's lowest
/// voxel face; drawing the depth-tested floor exactly there z-fights that face
/// (the floor flickers / vanishes under the model). Dropping it a fraction of a
/// voxel makes it read as the ground UNDER the object and removes the fight, while
/// staying visually on the base plane.
const FLOOR_PLANE_DROP_VOXELS: f32 = 0.25;

/// Append a floor grid for the box `[min, max]` (voxels) on its BASE plane
/// (just below `y = min[1]`) — lines at every BLOCK boundary (spacing = `step`),
/// snapped to the same global block lines as the lattice (issue #29 S3). The base
/// plane is the node's bottom; the grid reads as the ground under the object.
fn floor_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_ALPHA);
    let y = min[1] - FLOOR_PLANE_DROP_VOXELS;
    let xs = block_boundaries(min[0], max[0], step);
    let zs = block_boundaries(min[2], max[2], step);

    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Lines parallel to Z, at every X block boundary.
    for &x in &xs {
        add([x, y, min[2]], [x, y, max[2]]);
    }
    // Lines parallel to X, at every Z block boundary.
    for &z in &zs {
        add([min[0], y, z], [max[0], y, z]);
    }
}

// ============================================================================
// Onion-skin volumetric fog (issue #12) — fullscreen SDF raymarch.
// ============================================================================

/// Parameters for one frame of the onion-skin fog pass. The fog raymarches the
/// RESOLVED voxel grid (uploaded via [`OnionFogRenderer::upload_grid`]) as a 3D
/// cloud density field and integrates a faint haze in the onion-band Y range
/// OUTSIDE the displayed (solid) band. Option B (x-ray onion): the march ignores
/// opaque depth so neighbour layers show through the slice on both sides.
#[derive(Debug, Clone, Copy)]
pub struct OnionFogParams {
    /// Inverse camera view-projection (to unproject screen → world rays).
    pub inverse_view_projection: glam::Mat4,
    /// Inscribed semi-axes (= grid_dimensions / 2); maps world → normalised grid.
    pub semi_axes: [f32; 3],
    /// World-space Y extent of the onion band (the layers to fog).
    pub onion_y_min: f32,
    pub onion_y_max: f32,
    /// World-space Y extent of the displayed solid band (excluded from the fog —
    /// the opaque voxel pass already drew it).
    pub band_y_min: f32,
    pub band_y_max: f32,
}

/// std140-safe uniform block; field order matches `FogUniforms` in onion_fog.wgsl.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OnionFogUniforms {
    inverse_view_projection: [[f32; 4]; 4],
    semi_axes: [f32; 3],
    fog_strength: f32,
    fog_color: [f32; 3],
    _pad0: f32,
    onion_y_min: f32,
    onion_y_max: f32,
    band_y_min: f32,
    band_y_max: f32,
}

/// Fog tint (cool blue-grey) and Beer–Lambert strength. Strength is low so the
/// haze is aerogel-faint and the solid band clearly shows through. Option B
/// (x-ray onion) wants it wispier still, so the band reads as a faint ghost rather
/// than a frosted puck — lowered from the original 0.18.
const ONION_FOG_COLOR_HEX: u32 = 0x9c_b4_d8;
const ONION_FOG_STRENGTH: f32 = 0.10;

/// Which occupancy source the onion fog raymarches (issue #28 S5a).
///
/// * [`WholeGrid`](FogMode::WholeGrid) (DEFAULT) — the original path: ONE whole-grid
///   `D3 R8` occupancy texture densified from the entire sparse list, disabled when
///   any axis exceeds `max_texture_dimension_3d`.
/// * [`PerChunk`](FogMode::PerChunk) — one apron'd `R8` occupancy volume per resident
///   chunk, packed into a small 3D atlas scoped to the active region, so a scene too
///   large for a single whole-grid 3D texture still renders fog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FogMode {
    #[default]
    WholeGrid,
    PerChunk,
}

/// Cap on the number of resident chunk volumes the per-chunk fog tracks in one frame
/// (issue #28 S5a). Each chunk contributes a `[u32; 4]` record to the metadata uniform
/// (1024 × 16 B = 16 KiB, well under the 64 KiB uniform limit) and one apron'd tile in
/// the atlas. A region scene stays far under this; the scrubber region-scoping (S5b)
/// keeps it that way once the default flips.
pub const MAX_FOG_CHUNKS: usize = 1024;

/// One resident chunk's apron'd occupancy plus where it lives, in the per-chunk fog
/// path (issue #28 S5a). The occupancy is stored at `(extent + 2)³` so a **1-voxel
/// apron** on every face replicates the neighbour occupancy and trilinear sampling
/// stays smooth across chunk seams (no banding at the boundary).
#[derive(Debug, Clone)]
pub struct ChunkFogVolume {
    /// The chunk's integer coordinate in `CHUNK_BLOCKS`-cell space.
    pub chunk_coord: [i32; 3],
    /// The world-space (recentred) coordinate of this chunk's `[0,0,0]` voxel CORNER
    /// (i.e. the apron's interior origin), so the shader maps a world sample into the
    /// chunk's local `[0, extent)` voxel space.
    pub world_origin: [f32; 3],
    /// The apron'd occupancy, `(extent + 2)³` bytes in `(k*pad + j)*pad + i` order
    /// where local apron index `0` is the apron voxel at chunk-local `-1`.
    pub occupancy: Vec<u8>,
}

/// The CPU result of bucketing a recentred whole grid into apron'd per-chunk fog
/// volumes (issue #28 S5a): the per-chunk volumes plus the shared chunk voxel extent.
#[derive(Debug, Clone, Default)]
pub struct PerChunkFogOccupancy {
    /// `CHUNK_BLOCKS * voxels_per_block` — the voxel extent of one chunk per axis.
    pub chunk_extent: u32,
    /// The apron'd volumes, one per non-empty resident chunk. Empty when the resident
    /// non-empty chunk count exceeds [`MAX_FOG_CHUNKS`] (per-chunk fog disables itself
    /// for that build rather than dropping chunks and rendering with holes).
    pub volumes: Vec<ChunkFogVolume>,
}

/// Bucket a recentred [`VoxelGrid`] into one apron'd `R8` occupancy volume per
/// non-empty chunk (issue #28 S5a, the per-chunk fog path).
///
/// This reads the SAME recentred grid the whole-grid path uploads and uses the SAME
/// `world → voxel` mapping (`round(world + half - 0.5)`), so the per-chunk occupancy
/// is voxel-for-voxel identical to the whole-grid volume — the A/B match is exact by
/// construction. Each chunk's volume carries a **1-voxel apron**: the border layer is
/// filled from the global occupancy (the true neighbour voxel, NOT a clamp), so a ray
/// crossing a chunk seam trilinear-interpolates against the real neighbour density and
/// shows no discontinuity.
///
/// `chunk_coord = floor(voxel_index / chunk_extent)`; the chunk's interior origin in
/// recentred world space is `chunk_coord * chunk_extent - half_grid` (voxel CORNER),
/// so a world sample maps to chunk-local voxel space by `world - world_origin`.
pub fn build_per_chunk_fog_occupancy(
    grid: &VoxelGrid,
    voxels_per_block: u32,
) -> PerChunkFogOccupancy {
    let chunk_extent = (CHUNK_BLOCKS * voxels_per_block.max(1)) as i64;
    let [grid_x, grid_y, grid_z] = grid.dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return PerChunkFogOccupancy {
            chunk_extent: chunk_extent as u32,
            volumes: Vec::new(),
        };
    }
    let half = [grid_x as f32 / 2.0, grid_y as f32 / 2.0, grid_z as f32 / 2.0];

    // First pass: integer voxel coords of every occupied voxel (the SAME mapping the
    // whole-grid upload uses), bucketed by chunk coordinate. We keep a per-chunk set of
    // local voxel coords so the apron can be filled exactly (a neighbour voxel that
    // belongs to an adjacent chunk still lands in THIS chunk's apron layer).
    use std::collections::{HashMap, HashSet};
    let mut occupied_voxels: HashSet<[i64; 3]> = HashSet::new();
    for voxel in &grid.occupied {
        let i = (voxel.world_position[0] + half[0] - 0.5).round() as i64;
        let j = (voxel.world_position[1] + half[1] - 0.5).round() as i64;
        let k = (voxel.world_position[2] + half[2] - 0.5).round() as i64;
        if i < 0 || j < 0 || k < 0 || i >= grid_x as i64 || j >= grid_y as i64 || k >= grid_z as i64
        {
            continue;
        }
        occupied_voxels.insert([i, j, k]);
    }

    // Which chunks contain at least one occupied voxel.
    let mut chunk_coords: HashMap<[i32; 3], ()> = HashMap::new();
    for &[i, j, k] in &occupied_voxels {
        let coord = [
            narrow_chunk_coord_local(i.div_euclid(chunk_extent)),
            narrow_chunk_coord_local(j.div_euclid(chunk_extent)),
            narrow_chunk_coord_local(k.div_euclid(chunk_extent)),
        ];
        chunk_coords.insert(coord, ());
    }
    let mut keys: Vec<[i32; 3]> = chunk_coords.keys().copied().collect();
    keys.sort_unstable();
    // Too many resident non-empty chunks for the per-chunk atlas to hold. Degrade
    // gracefully and CONSISTENTLY with `upload_grid_per_chunk`'s other overflow branch
    // (atlas-dimension-exceeded): return NO volumes so the upload takes its existing
    // `chunk_count == 0` disable path (per_chunk_active = false) → the region shows NO
    // fog (honest) rather than fog-with-holes (wrong: a previous `keys.truncate` dropped
    // the overflow chunks, whose raymarch occupancy then read 0 → silent fog holes).
    // The proper long-term fix (region-scope the fog to resident/visible chunks so the
    // resident set stays small) is tracked in #20 step 4.
    if keys.len() > MAX_FOG_CHUNKS {
        eprintln!(
            "per-chunk fog: {} non-empty chunks exceeds MAX_FOG_CHUNKS ({}); disabling \
             per-chunk fog for this build (no fog) rather than rendering with holes",
            keys.len(),
            MAX_FOG_CHUNKS,
        );
        return PerChunkFogOccupancy {
            chunk_extent: chunk_extent as u32,
            volumes: Vec::new(),
        };
    }

    let pad = (chunk_extent + 2) as usize; // apron: -1 .. extent (inclusive)
    let mut volumes = Vec::with_capacity(keys.len());
    for coord in keys {
        let chunk_min = [
            coord[0] as i64 * chunk_extent,
            coord[1] as i64 * chunk_extent,
            coord[2] as i64 * chunk_extent,
        ];
        let mut occupancy = vec![0u8; pad * pad * pad];
        // Fill the apron'd box `[-1, extent]` per axis from the GLOBAL occupancy, so the
        // border layer carries the true neighbour voxel (seam-smooth trilinear).
        for local_k in -1..=chunk_extent {
            for local_j in -1..=chunk_extent {
                for local_i in -1..=chunk_extent {
                    let global = [
                        chunk_min[0] + local_i,
                        chunk_min[1] + local_j,
                        chunk_min[2] + local_k,
                    ];
                    if occupied_voxels.contains(&global) {
                        let ai = (local_i + 1) as usize;
                        let aj = (local_j + 1) as usize;
                        let ak = (local_k + 1) as usize;
                        occupancy[(ak * pad + aj) * pad + ai] = 255;
                    }
                }
            }
        }
        volumes.push(ChunkFogVolume {
            chunk_coord: coord,
            // Interior origin (voxel CORNER of local [0,0,0]) in recentred world space.
            world_origin: [
                chunk_min[0] as f32 - half[0],
                chunk_min[1] as f32 - half[1],
                chunk_min[2] as f32 - half[2],
            ],
            occupancy,
        });
    }

    PerChunkFogOccupancy {
        chunk_extent: chunk_extent as u32,
        volumes,
    }
}

/// Narrow an i64 chunk-coordinate quotient to i32 (saturating). Chunk coords stay tiny
/// in practice; this mirrors `scene::narrow_chunk_coord` without exposing it.
fn narrow_chunk_coord_local(value: i64) -> i32 {
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Fullscreen volumetric-fog renderer for the onion skin (issue #12). Raymarches
/// the resolved voxel grid (uploaded as a 3D occupancy texture) as a cloud.
pub struct OnionFogRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Trilinear sampler for the occupancy grid (the cloud density read).
    sampler: wgpu::Sampler,
    /// Current grid as a 3D R8 occupancy texture view; replaced on `upload_grid`.
    grid_view: wgpu::TextureView,
    /// Largest 3D texture dimension the device allows (grids past this skip fog).
    max_grid_dimension: u32,
    /// Whether the current grid uploaded successfully (else `draw` is a no-op).
    active: bool,
    /// Which occupancy source the next `draw` raymarches (issue #28 S5a). Set per
    /// upload (`upload_grid` → `WholeGrid`, `upload_grid_per_chunk` → `PerChunk`).
    mode: FogMode,
    // --- Per-chunk path (issue #28 S5a) ---
    /// Pipeline that raymarches the per-chunk atlas (separate WGSL entry point).
    per_chunk_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the per-chunk path: shared camera uniform, atlas D3
    /// texture, sampler, scene depth, plus the per-chunk metadata uniform.
    per_chunk_bind_group_layout: wgpu::BindGroupLayout,
    /// The packed apron'd per-chunk occupancy atlas (one tile per resident chunk).
    per_chunk_atlas_view: wgpu::TextureView,
    /// Per-chunk metadata uniform (atlas tiling + per-chunk world origin / tile coord).
    per_chunk_meta_buffer: wgpu::Buffer,
    /// Whether the last per-chunk upload produced a renderable atlas.
    per_chunk_active: bool,
}

/// std140 per-chunk fog metadata (issue #28 S5a). The shader walks the ray, and at
/// each sample point computes the chunk coord, looks up that chunk's atlas tile from
/// `chunks[]`, and samples the apron'd tile. Field order matches the WGSL
/// `PerChunkMeta` struct exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PerChunkFogMeta {
    /// Number of resident chunk records in `chunks` (≤ [`MAX_FOG_CHUNKS`]).
    chunk_count: u32,
    /// Voxel extent of one chunk per axis (`CHUNK_BLOCKS * voxels_per_block`).
    chunk_extent: f32,
    /// Padded interior tile extent in the atlas (`chunk_extent + 2`, the apron).
    pad_extent: f32,
    /// Number of tiles per axis in the (cubic-ish) atlas tile grid.
    tiles_per_axis: u32,
    /// Atlas dimension in texels per axis (`tiles_per_axis * pad_extent`).
    atlas_dim: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    /// One record per resident chunk: `[world_origin.xyz, packed_tile_index]`. The
    /// world origin is the chunk's interior `[0,0,0]` voxel CORNER in recentred world
    /// space; `packed_tile_index` is the linear atlas tile index (decode to a 3D tile
    /// coord in the shader). Unused entries are zeroed.
    chunks: [[f32; 4]; MAX_FOG_CHUNKS],
}

impl OnionFogRenderer {
    /// Create the fog renderer for a colour target format.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("onion fog shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/onion_fog.wgsl").into()),
        });

        // Binding 0: uniform; binding 1: the resolved voxel grid as a 3D occupancy
        // texture (R8, trilinear-filtered); binding 2: its sampler.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("onion fog bind group layout"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 3: the MSAA scene depth, so the fog is occluded by the
                // displayed opaque slice (depth-tested like Minecraft's clouds).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("onion fog pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("onion fog pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    // Straight alpha-over: fog colour composited onto the resolved
                    // scene by its `coverage` alpha.
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
            // The fog runs at 1 sample onto the resolved target (after the 3D MSAA
            // resolve, before egui), so no depth attachment / no MSAA here.
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog uniforms"),
            size: std::mem::size_of::<OnionFogUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Trilinear sampler: linear filtering turns the binary occupancy grid into
        // a smooth cloud density. Clamp-to-edge (the shader also rejects samples
        // outside the grid box, so the border value never smears along the ray).
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("onion fog occupancy sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Start with a 1×1×1 empty grid so the bind group is valid before the first
        // `upload_grid`. `active` stays false until a real grid lands.
        let grid_view = create_empty_occupancy_view(device);

        // --- Per-chunk path (issue #28 S5a) ---
        let per_chunk_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("onion fog per-chunk shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/onion_fog_perchunk.wgsl").into(),
            ),
        });
        let per_chunk_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("onion fog per-chunk bind group layout"),
                entries: &[
                    // 0: shared camera/band uniform (same OnionFogUniforms).
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
                    // 1: the packed apron'd per-chunk occupancy atlas (R8, trilinear).
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 2: occupancy sampler (trilinear).
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 3: MSAA scene depth (depth-tested like the whole-grid path).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // 4: per-chunk metadata uniform (atlas tiling + chunk records).
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let per_chunk_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("onion fog per-chunk pipeline layout"),
                bind_group_layouts: &[Some(&per_chunk_bind_group_layout)],
                immediate_size: 0,
            });
        let per_chunk_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("onion fog per-chunk pipeline"),
                layout: Some(&per_chunk_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &per_chunk_shader,
                    entry_point: Some("vertex_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &per_chunk_shader,
                    entry_point: Some("fragment_main"),
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
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            });
        let per_chunk_atlas_view = create_empty_occupancy_view(device);
        let per_chunk_meta_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("onion fog per-chunk meta"),
            size: std::mem::size_of::<PerChunkFogMeta>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            sampler,
            grid_view,
            max_grid_dimension: device.limits().max_texture_dimension_3d,
            active: false,
            mode: FogMode::WholeGrid,
            per_chunk_pipeline,
            per_chunk_bind_group_layout,
            per_chunk_atlas_view,
            per_chunk_meta_buffer,
            per_chunk_active: false,
        }
    }

    /// Upload the resolved voxel grid as a 3D occupancy texture (the cloud density
    /// the fog raymarches). Call whenever the grid changes (geometry rebuild). A
    /// grid whose dimensions exceed the device's 3D-texture limit, or that is
    /// empty, disables the fog (`draw` becomes a no-op) rather than failing.
    pub fn upload_grid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid: &VoxelGrid) {
        let [grid_x, grid_y, grid_z] = grid.dimensions;
        let limit = self.max_grid_dimension;
        if grid_x == 0
            || grid_y == 0
            || grid_z == 0
            || grid_x > limit
            || grid_y > limit
            || grid_z > limit
        {
            self.active = false;
            return;
        }

        // Densify the sparse occupied list into an R8 volume. Texel order matches a
        // 3D texture: index = (k * height + j) * width + i, with width=x, height=y,
        // depth=z. Voxel (i, j, k) ← round(world + half - 0.5), the same mapping the
        // grid uses elsewhere (voxel.rs::widest_run_in_band).
        let (width, height, depth) = (grid_x as usize, grid_y as usize, grid_z as usize);
        let mut occupancy = vec![0u8; width * height * depth];
        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;
        for voxel in &grid.occupied {
            let i = (voxel.world_position[0] + half_x - 0.5).round() as i64;
            let j = (voxel.world_position[1] + half_y - 0.5).round() as i64;
            let k = (voxel.world_position[2] + half_z - 0.5).round() as i64;
            if i < 0
                || j < 0
                || k < 0
                || i >= grid_x as i64
                || j >= grid_y as i64
                || k >= grid_z as i64
            {
                continue;
            }
            let index = (k as usize * height + j as usize) * width + i as usize;
            occupancy[index] = 255;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("onion fog occupancy grid"),
            size: wgpu::Extent3d {
                width: grid_x,
                height: grid_y,
                depth_or_array_layers: grid_z,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &occupancy,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(grid_x),
                rows_per_image: Some(grid_y),
            },
            wgpu::Extent3d {
                width: grid_x,
                height: grid_y,
                depth_or_array_layers: grid_z,
            },
        );
        self.grid_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.active = true;
        self.mode = FogMode::WholeGrid;
    }

    /// Upload the resolved grid as PER-CHUNK apron'd occupancy volumes packed into a
    /// small 3D atlas (issue #28 S5a, `--fog=perchunk`). Unlike [`upload_grid`], the
    /// atlas size is bounded by the number of resident chunks, NOT the whole-grid
    /// extent, so a scene whose whole-grid axis would exceed `max_texture_dimension_3d`
    /// (and thus disable the whole-grid fog) still renders fog here.
    ///
    /// Each chunk's tile is `(chunk_extent + 2)³` (a 1-voxel apron filled from the
    /// global occupancy), so trilinear sampling is seam-smooth across chunk boundaries.
    /// The shader marches in recentred world space and, at each sample, maps the world
    /// point into the owning chunk's tile via the metadata records.
    pub fn upload_grid_per_chunk(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) {
        let occupancy = build_per_chunk_fog_occupancy(grid, voxels_per_block);
        let pad = occupancy.chunk_extent as usize + 2;
        let chunk_count = occupancy.volumes.len();
        if chunk_count == 0 || pad == 0 {
            self.per_chunk_active = false;
            self.mode = FogMode::PerChunk;
            return;
        }

        // Arrange the resident chunk tiles into a cubic-ish 3D tile grid, so the atlas
        // dimension per axis (`tiles_per_axis * pad`) stays small — bounded by the chunk
        // COUNT, not the whole-grid extent. This is the core of why per-chunk dodges the
        // single-3D-texture limit.
        let tiles_per_axis = (chunk_count as f64).cbrt().ceil() as u32;
        let tiles_per_axis = tiles_per_axis.max(1);
        let atlas_dim = tiles_per_axis * pad as u32;
        if atlas_dim > self.max_grid_dimension {
            // The active region has too many chunks for the atlas to fit the 3D limit;
            // fall back to disabled fog rather than failing. (S5b's region scoping keeps
            // the resident set small; a region this large is out of S5a scope.)
            self.per_chunk_active = false;
            self.mode = FogMode::PerChunk;
            return;
        }

        // Pack every chunk's apron'd occupancy into the atlas at its tile slot, and
        // record each chunk's world origin + linear tile index in the metadata.
        let atlas_texels = (atlas_dim as usize).pow(3);
        let mut atlas = vec![0u8; atlas_texels];
        let mut meta = PerChunkFogMeta {
            chunk_count: chunk_count as u32,
            chunk_extent: occupancy.chunk_extent as f32,
            pad_extent: pad as f32,
            tiles_per_axis,
            atlas_dim: atlas_dim as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
            chunks: [[0.0; 4]; MAX_FOG_CHUNKS],
        };
        for (tile_index, volume) in occupancy.volumes.iter().enumerate() {
            // Linear tile index → 3D tile coord in the atlas.
            let tx = (tile_index as u32) % tiles_per_axis;
            let ty = ((tile_index as u32) / tiles_per_axis) % tiles_per_axis;
            let tz = (tile_index as u32) / (tiles_per_axis * tiles_per_axis);
            let base = [tx as usize * pad, ty as usize * pad, tz as usize * pad];
            for local_z in 0..pad {
                for local_y in 0..pad {
                    for local_x in 0..pad {
                        let src = (local_z * pad + local_y) * pad + local_x;
                        let ax = base[0] + local_x;
                        let ay = base[1] + local_y;
                        let az = base[2] + local_z;
                        let dst = (az * atlas_dim as usize + ay) * atlas_dim as usize + ax;
                        atlas[dst] = volume.occupancy[src];
                    }
                }
            }
            meta.chunks[tile_index] = [
                volume.world_origin[0],
                volume.world_origin[1],
                volume.world_origin[2],
                tile_index as f32,
            ];
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("onion fog per-chunk atlas"),
            size: wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_dim),
                rows_per_image: Some(atlas_dim),
            },
            wgpu::Extent3d {
                width: atlas_dim,
                height: atlas_dim,
                depth_or_array_layers: atlas_dim,
            },
        );
        self.per_chunk_atlas_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        queue.write_buffer(&self.per_chunk_meta_buffer, 0, bytemuck::bytes_of(&meta));
        self.per_chunk_active = true;
        self.mode = FogMode::PerChunk;
    }

    /// The fog mode the last upload selected (issue #28 S5a).
    pub fn mode(&self) -> FogMode {
        self.mode
    }

    /// Whether the per-chunk path has a renderable atlas (diagnostic / tests).
    pub fn per_chunk_active(&self) -> bool {
        self.per_chunk_active
    }

    /// Upload this frame's fog parameters.
    pub fn update(&self, queue: &wgpu::Queue, params: OnionFogParams) {
        let uniforms = OnionFogUniforms {
            inverse_view_projection: params.inverse_view_projection.to_cols_array_2d(),
            semi_axes: params.semi_axes,
            fog_strength: ONION_FOG_STRENGTH,
            fog_color: srgb_hex_to_linear(ONION_FOG_COLOR_HEX),
            _pad0: 0.0,
            onion_y_min: params.onion_y_min,
            onion_y_max: params.onion_y_max,
            band_y_min: params.band_y_min,
            band_y_max: params.band_y_max,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the fog into `target_view` (the resolved scene), raymarching the
    /// uploaded occupancy grid and depth-testing against `depth_view` (the 3D pass's
    /// MSAA depth) so the displayed opaque slice occludes the onion layers behind
    /// it. A no-op until a grid has been uploaded (`upload_grid`). Its own render
    /// pass loads the existing colour and composites the haze over it.
    /// Issue #25: `viewport` (`[x, y, w, h]`, physical pixels) confines the
    /// fullscreen raymarch to the central 3D viewport rect. The fog reconstructs
    /// world rays from the central-aspect `inverse_view_projection`, so it is only
    /// valid inside that rect; the scissor keeps it off the panels.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        viewport: [u32; 4],
    ) {
        // Build the bind group + pick the pipeline for the active mode. Both modes share
        // the camera uniform (binding 0), occupancy texture (1), sampler (2) and depth
        // (3); the per-chunk path adds the metadata uniform (4).
        let (pipeline, bind_group) = match self.mode {
            FogMode::WholeGrid => {
                if !self.active {
                    return;
                }
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("onion fog bind group"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.grid_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                    ],
                });
                (&self.pipeline, bind_group)
            }
            FogMode::PerChunk => {
                if !self.per_chunk_active {
                    return;
                }
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("onion fog per-chunk bind group"),
                    layout: &self.per_chunk_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.per_chunk_atlas_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(depth_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.per_chunk_meta_buffer.as_entire_binding(),
                        },
                    ],
                });
                (&self.per_chunk_pipeline, bind_group)
            }
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("onion fog pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let [vx, vy, vw, vh] = viewport;
        pass.set_viewport(vx as f32, vy as f32, vw as f32, vh as f32, 0.0, 1.0);
        pass.set_scissor_rect(vx, vy, vw, vh);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// A 1×1×1 empty (zero) R8 occupancy texture view, used to keep the fog bind group
/// valid before/without a real grid upload.
fn create_empty_occupancy_view(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("onion fog occupancy (empty)"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Create a 4-sample (MSAA) depth texture view sized to a render target.
pub fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel depth texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: MSAA_SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        // TEXTURE_BINDING so the onion fog pass can sample this MSAA depth (sample 0)
        // to occlude the haze behind the displayed opaque slice.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For a triangle wound CCW *as seen from outside*, the geometric face normal
    /// (edge0 × edge1) points in the SAME direction as the stored outward normal,
    /// so their dot product is positive. A negative dot means the winding is
    /// inside-out (BUG 1) and back-face culling would hide the visible face.
    fn assert_ccw_outward(positions: &[[f32; 3]], normals: &[[f32; 3]], indices: &[u16]) {
        assert_eq!(indices.len() % 3, 0, "indices must form whole triangles");
        for tri in indices.chunks_exact(3) {
            let a = positions[tri[0] as usize];
            let b = positions[tri[1] as usize];
            let c = positions[tri[2] as usize];
            let edge0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let edge1 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            // edge0 × edge1
            let geometric_normal = [
                edge0[1] * edge1[2] - edge0[2] * edge1[1],
                edge0[2] * edge1[0] - edge0[0] * edge1[2],
                edge0[0] * edge1[1] - edge0[1] * edge1[0],
            ];
            let outward = normals[tri[0] as usize];
            let dot = geometric_normal[0] * outward[0]
                + geometric_normal[1] * outward[1]
                + geometric_normal[2] * outward[2];
            assert!(
                dot > 0.0,
                "triangle {tri:?} is wound inside-out (dot={dot}); outward faces would be culled",
            );
        }
    }

    #[test]
    fn voxel_cube_is_ccw_outward() {
        let (vertices, indices) = unit_cube_geometry();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
        assert_ccw_outward(&positions, &normals, &indices);
    }

    // ===== Issue #28 S5a: per-chunk fog apron generation ========================

    /// Build a recentred [`VoxelGrid`] of `dims` voxels with the given integer voxel
    /// coords occupied, using the SAME `voxel ↔ world` mapping the fog upload uses
    /// (world = i + 0.5 - dim/2). So `build_per_chunk_fog_occupancy` reads them back at
    /// the exact integer coords here.
    fn grid_with_voxels(dims: [u32; 3], coords: &[[u32; 3]]) -> VoxelGrid {
        let mut grid = VoxelGrid::new(dims);
        let half = [dims[0] as f32 / 2.0, dims[1] as f32 / 2.0, dims[2] as f32 / 2.0];
        for &[i, j, k] in coords {
            grid.occupied.push(Voxel {
                world_position: [
                    i as f32 + 0.5 - half[0],
                    j as f32 + 0.5 - half[1],
                    k as f32 + 0.5 - half[2],
                ],
                block_local_coord: [0, 0, 0],
                material_id: 0,
            });
        }
        grid
    }

    /// Read the apron'd occupancy of `volume` at chunk-LOCAL coord `(li, lj, lk)`
    /// (`-1 ..= extent`), where `0` is the chunk's interior `[0,0,0]` voxel.
    fn apron_at(volume: &ChunkFogVolume, extent: i64, local: [i64; 3]) -> u8 {
        let pad = (extent + 2) as usize;
        let a = [
            (local[0] + 1) as usize,
            (local[1] + 1) as usize,
            (local[2] + 1) as usize,
        ];
        volume.occupancy[(a[2] * pad + a[1]) * pad + a[0]]
    }

    /// The apron of a chunk reflects a NEIGHBOUR chunk's boundary occupancy (seam
    /// smoothness), and an interior/edge voxel of the chunk shows up in its own volume.
    #[test]
    fn per_chunk_apron_reflects_neighbour_and_boundary() {
        // density 1 → CHUNK_BLOCKS * 1 = 4 voxels/chunk. A 2-chunk-wide grid in X so a
        // voxel in chunk 1 sits in chunk 0's +X apron.
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4
        let dims = [(extent * 2) as u32, extent as u32, extent as u32]; // 8x4x4
        // Occupy: chunk-0 boundary voxel at x=3 (its own +X edge), and chunk-1's first
        // voxel at x=4 (the neighbour that must appear in chunk-0's apron).
        let grid = grid_with_voxels(dims, &[[3, 0, 0], [4, 0, 0]]);

        let occ = build_per_chunk_fog_occupancy(&grid, density);
        assert_eq!(occ.chunk_extent, extent as u32);
        // Two chunks are occupied (x=3 in chunk 0, x=4 in chunk 1).
        assert_eq!(occ.volumes.len(), 2, "two chunks hold voxels");

        let chunk0 = occ
            .volumes
            .iter()
            .find(|v| v.chunk_coord == [0, 0, 0])
            .expect("chunk 0 resident");
        // Its own edge voxel (local x=3) is occupied.
        assert_eq!(apron_at(chunk0, extent, [3, 0, 0]), 255, "chunk-0 own edge voxel");
        // The neighbour voxel (chunk-1 x=4 → chunk-0 local x=extent) sits in the +X
        // apron and is filled from the global occupancy → seam-smooth trilinear.
        assert_eq!(
            apron_at(chunk0, extent, [extent, 0, 0]),
            255,
            "chunk-0 +X apron carries the neighbour chunk's boundary voxel"
        );
        // An empty apron cell stays 0 (e.g. -1 in X, outside everything).
        assert_eq!(apron_at(chunk0, extent, [-1, 0, 0]), 0, "empty apron stays empty");
    }

    /// An empty grid yields no volumes (fog disables itself), and the world origin of a
    /// chunk is its interior `[0,0,0]` voxel corner in recentred world space.
    #[test]
    fn per_chunk_world_origin_is_recentred_corner() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4
        let dims = [(extent * 2) as u32, extent as u32, extent as u32]; // 8x4x4
        let half = dims[0] as f32 / 2.0; // 4
        let grid = grid_with_voxels(dims, &[[5, 0, 0]]); // chunk 1 in X
        let occ = build_per_chunk_fog_occupancy(&grid, density);
        let chunk1 = occ
            .volumes
            .iter()
            .find(|v| v.chunk_coord == [1, 0, 0])
            .expect("chunk 1 resident");
        // Chunk 1's interior origin = chunk_coord*extent - half = 4 - 4 = 0 in X.
        assert!((chunk1.world_origin[0] - (extent as f32 - half)).abs() < 1e-6);

        // Empty grid → no volumes.
        let empty = VoxelGrid::new(dims);
        assert!(build_per_chunk_fog_occupancy(&empty, density).volumes.is_empty());
    }

    /// When the resident non-empty chunk count exceeds `MAX_FOG_CHUNKS`, the builder
    /// disables per-chunk fog for that build (returns NO volumes) instead of dropping the
    /// overflow chunks — which would render fog with silent holes. The empty result makes
    /// `upload_grid_per_chunk` take its `chunk_count == 0` graceful-disable path. (#20 s4
    /// region-scoping is the proper long-term fix that keeps the resident set small.)
    #[test]
    fn per_chunk_fog_disables_past_max_fog_chunks() {
        let density = 1u32;
        let extent = (CHUNK_BLOCKS * density) as i64; // 4 voxels per chunk per axis
        // One occupied voxel in each of (MAX_FOG_CHUNKS + 1) distinct chunks along X.
        let chunk_count = MAX_FOG_CHUNKS + 1;
        let dims = [(extent as usize * chunk_count) as u32, extent as u32, extent as u32];
        let coords: Vec<[u32; 3]> = (0..chunk_count)
            .map(|chunk_index| [(chunk_index as i64 * extent) as u32, 0, 0])
            .collect();
        let grid = grid_with_voxels(dims, &coords);

        let occ = build_per_chunk_fog_occupancy(&grid, density);
        assert!(
            occ.volumes.is_empty(),
            "over MAX_FOG_CHUNKS resident chunks must disable fog (no volumes), not truncate"
        );

        // The common case (≤ MAX_FOG_CHUNKS) still produces volumes — exactly at the cap.
        let coords_at_cap: Vec<[u32; 3]> = (0..MAX_FOG_CHUNKS)
            .map(|chunk_index| [(chunk_index as i64 * extent) as u32, 0, 0])
            .collect();
        let dims_at_cap =
            [(extent as usize * MAX_FOG_CHUNKS) as u32, extent as u32, extent as u32];
        let grid_at_cap = grid_with_voxels(dims_at_cap, &coords_at_cap);
        let occ_at_cap = build_per_chunk_fog_occupancy(&grid_at_cap, density);
        assert_eq!(
            occ_at_cap.volumes.len(),
            MAX_FOG_CHUNKS,
            "exactly MAX_FOG_CHUNKS resident chunks still renders (boundary is inclusive)"
        );
    }

    #[test]
    fn view_cube_is_ccw_outward() {
        let (vertices, indices) = view_cube_geometry();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
        assert_ccw_outward(&positions, &normals, &indices);
    }

    use crate::voxel::{Voxel, VoxelGrid};

    /// Build a small solid box grid of `dims` voxels (every cell occupied),
    /// world-centred so voxel centres sit at half-integer coords (matching the
    /// SDF producer's convention).
    fn solid_grid(dims: [u32; 3]) -> VoxelGrid {
        let [nx, ny, nz] = dims;
        let half = [nx as f32 / 2.0, ny as f32 / 2.0, nz as f32 / 2.0];
        let mut occupied = Vec::new();
        for j in 0..ny {
            for k in 0..nz {
                for i in 0..nx {
                    occupied.push(Voxel {
                        world_position: [
                            i as f32 + 0.5 - half[0],
                            j as f32 + 0.5 - half[1],
                            k as f32 + 0.5 - half[2],
                        ],
                        block_local_coord: [0, 0, 0],
                        material_id: 0,
                    });
                }
            }
        }
        VoxelGrid { dimensions: dims, occupied }
    }

    /// Every occupied voxel must land in exactly one chunk, and the union of all
    /// chunk instance ranges must reproduce the whole occupied set with no
    /// truncation (the old 450k draw cap is gone). We verify by partitioning the
    /// instance list along the chunk ranges and checking the multiset of
    /// world_positions equals the grid's.
    #[test]
    fn chunk_bucketing_partitions_every_voxel_exactly_once() {
        // 8 blocks/axis at density 4 = 32 voxels/axis, so CHUNK_BLOCKS=4 yields a
        // 2×2×2 = 8-chunk grid with several voxels per chunk.
        let density = 4u32;
        let dims = [8 * density, 8 * density, 8 * density];
        let grid = solid_grid(dims);
        let (instances, chunks) = bucket_instances_into_chunks(&grid, density);

        // No truncation: every occupied voxel is present as an instance.
        assert_eq!(instances.len(), grid.occupied.len());

        // The chunk ranges tile [0, instances.len()) contiguously with no gaps or
        // overlaps (each instance is covered by exactly one chunk range).
        let mut covered = vec![0u32; instances.len()];
        let mut total_from_chunks = 0u32;
        for chunk in &chunks {
            total_from_chunks += chunk.instance_count;
            for index in chunk.instance_start..chunk.instance_start + chunk.instance_count {
                covered[index as usize] += 1;
            }
        }
        assert_eq!(total_from_chunks as usize, instances.len());
        assert!(covered.iter().all(|&c| c == 1), "every instance in exactly one chunk");

        // The set of world positions across chunks equals the grid's occupied set.
        let key = |p: [f32; 3]| {
            (
                (p[0] * 2.0).round() as i64,
                (p[1] * 2.0).round() as i64,
                (p[2] * 2.0).round() as i64,
            )
        };
        let mut from_chunks: Vec<_> = instances.iter().map(|i| key(i.world_position)).collect();
        let mut from_grid: Vec<_> = grid.occupied.iter().map(|v| key(v.world_position)).collect();
        from_chunks.sort_unstable();
        from_grid.sort_unstable();
        assert_eq!(from_chunks, from_grid);

        // Each voxel's chunk key must match the chunk whose range it falls in.
        let chunk_extent = (CHUNK_BLOCKS * density) as f32;
        for chunk in &chunks {
            for index in chunk.instance_start..chunk.instance_start + chunk.instance_count {
                let p = instances[index as usize].world_position;
                let voxel_key = [
                    (p[0] / chunk_extent).floor() as i32,
                    (p[1] / chunk_extent).floor() as i32,
                    (p[2] / chunk_extent).floor() as i32,
                ];
                // Every voxel sharing a chunk has the same chunk key.
                let first = instances[chunk.instance_start as usize].world_position;
                let first_key = [
                    (first[0] / chunk_extent).floor() as i32,
                    (first[1] / chunk_extent).floor() as i32,
                    (first[2] / chunk_extent).floor() as i32,
                ];
                assert_eq!(voxel_key, first_key, "voxel in wrong chunk");
            }
        }

        // A 32-voxel/axis grid with 64-voxel chunks = 1 chunk/axis... wait: 8
        // blocks / 4 blocks-per-chunk = 2 chunks/axis → 8 chunks. Confirm.
        assert_eq!(chunks.len(), 8);
    }

    /// An empty grid produces no chunks and no instances (no panic, no padding
    /// drawn).
    #[test]
    fn empty_grid_has_no_chunks() {
        let grid = VoxelGrid::new([16, 16, 16]);
        let (instances, chunks) = bucket_instances_into_chunks(&grid, 4);
        assert!(instances.is_empty());
        assert!(chunks.is_empty());
    }

    /// Issue #20 S6c-2b: the per-chunk buffer build (`instances_for_chunk` over a
    /// single chunk's grid — the seam the per-chunk GPU cache uses) must produce
    /// EXACTLY the instances the old monolithic `bucket_instances_into_chunks`
    /// produced for that chunk. We verify the strong invariant: grouping the whole
    /// grid into per-chunk sub-grids by the SAME chunk key the wrapper uses, then
    /// running `instances_for_chunk` per group, yields a per-chunk-coord →
    /// instance-multiset map IDENTICAL to slicing `bucket_instances_into_chunks`'s
    /// reordered instance list along its chunk ranges. (Order within a chunk is
    /// HashMap-iteration-dependent for the sub-grid build, so we compare multisets;
    /// within-chunk order is pixel-irrelevant — same depth-tested cubes.)
    #[test]
    fn per_chunk_instances_match_monolithic_bucketing_per_chunk() {
        let density = 4u32;
        let dims = [8 * density, 2 * density, 4 * density];
        let grid = solid_grid(dims);
        let chunk_extent = (CHUNK_BLOCKS * density) as f32;

        let chunk_key = |p: [f32; 3]| {
            [
                (p[0] / chunk_extent).floor() as i32,
                (p[1] / chunk_extent).floor() as i32,
                (p[2] / chunk_extent).floor() as i32,
            ]
        };
        // An order-independent fingerprint of one instance (bit-exact position +
        // block-local coord + material id).
        let fingerprint = |inst: &VoxelInstance| {
            (
                [
                    inst.world_position[0].to_bits(),
                    inst.world_position[1].to_bits(),
                    inst.world_position[2].to_bits(),
                ],
                [
                    inst.block_local_coord[0].to_bits(),
                    inst.block_local_coord[1].to_bits(),
                    inst.block_local_coord[2].to_bits(),
                ],
                inst.material_id,
            )
        };
        type Fp = ([u32; 3], [u32; 3], u32);
        type Multiset = std::collections::BTreeMap<Fp, usize>;

        // (1) The TRUTH: the monolithic bucketing, sliced into per-chunk multisets
        // keyed by chunk coord.
        let (mono_instances, mono_chunks) = bucket_instances_into_chunks(&grid, density);
        let mut truth: HashMap<[i32; 3], Multiset> = HashMap::new();
        for chunk in &mono_chunks {
            let coord = chunk_key(mono_instances[chunk.instance_start as usize].world_position);
            let entry = truth.entry(coord).or_default();
            for index in chunk.instance_start..chunk.instance_start + chunk.instance_count {
                *entry.entry(fingerprint(&mono_instances[index as usize])).or_insert(0) += 1;
            }
        }

        // (2) The per-chunk seam: group the whole grid into per-chunk sub-grids
        // (exactly as the `rebuild_instances` wrapper does), then build each chunk's
        // instances via `instances_for_chunk`.
        let mut sub_grids: HashMap<[i32; 3], VoxelGrid> = HashMap::new();
        for voxel in &grid.occupied {
            sub_grids
                .entry(chunk_key(voxel.world_position))
                .or_insert_with(|| VoxelGrid::new([0, 0, 0]))
                .occupied
                .push(*voxel);
        }
        let mut from_chunks: HashMap<[i32; 3], Multiset> = HashMap::new();
        for (coord, sub_grid) in &sub_grids {
            let (instances, _aabb) =
                instances_for_chunk(sub_grid).expect("a non-empty sub-grid yields instances");
            let entry = from_chunks.entry(*coord).or_default();
            for inst in &instances {
                *entry.entry(fingerprint(inst)).or_insert(0) += 1;
            }
        }

        // Same set of chunk coords, and each chunk's instance multiset is identical.
        assert_eq!(
            from_chunks.keys().copied().collect::<std::collections::BTreeSet<_>>(),
            truth.keys().copied().collect::<std::collections::BTreeSet<_>>(),
            "per-chunk seam must cover exactly the monolithic bucketing's chunk coords"
        );
        for (coord, truth_multiset) in &truth {
            assert_eq!(
                from_chunks.get(coord),
                Some(truth_multiset),
                "chunk {coord:?}: per-chunk instances must equal the monolithic slice"
            );
        }
    }

    /// `instances_for_chunk` returns `None` (no buffer allocated) for an empty
    /// chunk grid — the zero-voxel skip the per-chunk GPU cache relies on.
    #[test]
    fn instances_for_chunk_is_none_when_empty() {
        let empty = VoxelGrid::new([0, 0, 0]);
        assert!(instances_for_chunk(&empty).is_none());
    }

    /// Each chunk's AABB must contain all its voxel cubes (centre ±0.5) and no
    /// voxel cube from another chunk.
    #[test]
    fn chunk_aabb_bounds_its_voxels() {
        let density = 4u32;
        let dims = [8 * density, density, density];
        let grid = solid_grid(dims);
        let (instances, chunks) = bucket_instances_into_chunks(&grid, density);
        for chunk in &chunks {
            for index in chunk.instance_start..chunk.instance_start + chunk.instance_count {
                let c = glam::Vec3::from(instances[index as usize].world_position);
                let lo = c - glam::Vec3::splat(0.5);
                let hi = c + glam::Vec3::splat(0.5);
                assert!(lo.cmpge(chunk.aabb.min).all(), "voxel below chunk AABB min");
                assert!(hi.cmple(chunk.aabb.max).all(), "voxel above chunk AABB max");
            }
        }
    }

    // ---- issue #29 S3: per-object grid line geometry + gating ----

    use crate::panel::MaterialChoice as Mc;
    use crate::scene::{Node, NodeContent, NodePath};
    use crate::voxel::ShapeKind;
    use crate::voxel::SdfShape;

    /// `block_boundaries` returns the closing plane at `hi` (the box is enclosed in
    /// whole blocks), so a `B`-block box yields `B + 1` planes — and EXPANDING the
    /// box by one block on an axis adds exactly one boundary plane there. This is the
    /// geometry that makes "add/remove a whole block" fall out: a box grown by one
    /// enclosing block gains one lattice plane; shrunk by one, it loses one.
    #[test]
    fn block_boundaries_count_tracks_enclosing_blocks() {
        for step in [1u32, 15, 16] {
            let s = step as f32;
            // A 3-block box [0, 3·step] → planes at 0, step, 2·step, 3·step = 4.
            let three = block_boundaries(0.0, 3.0 * s, step);
            assert_eq!(three.len(), 4, "@step{step}: a 3-block box has 4 boundary planes");
            assert_eq!(*three.first().unwrap(), 0.0);
            assert_eq!(*three.last().unwrap(), 3.0 * s, "closing plane lands exactly on hi");
            // ADD a whole block (expand by +step): exactly one more plane.
            let four = block_boundaries(0.0, 4.0 * s, step);
            assert_eq!(four.len(), 5, "@step{step}: +1 enclosing block ⇒ +1 lattice plane");
            // REMOVE a whole block (shrink by step): exactly one fewer plane.
            let two = block_boundaries(0.0, 2.0 * s, step);
            assert_eq!(two.len(), 3, "@step{step}: -1 enclosing block ⇒ -1 lattice plane");
        }
    }

    /// One node's lattice/floor box → a non-empty line set at every density; the
    /// vertex count is a multiple of 2 (whole segments).
    #[test]
    fn lattice_and_floor_vertices_nonempty_per_box() {
        for step in [1u32, 15, 16] {
            let s = step as f32;
            let (min, max) = ([0.0, 0.0, 0.0], [2.0 * s, s, 3.0 * s]);
            let mut lattice = Vec::new();
            lattice_vertices_into(&mut lattice, min, max, step);
            assert!(!lattice.is_empty(), "@step{step}: a sized box has lattice lines");
            assert_eq!(lattice.len() % 2, 0, "lattice lines are whole segments");
            let mut floor = Vec::new();
            floor_vertices_into(&mut floor, min, max, step);
            assert!(!floor.is_empty(), "@step{step}: a sized box has floor lines");
            // Floor sits a fixed small drop below the base plane (issue #29 fix:
            // dropped off the model's coincident bottom face to avoid z-fighting),
            // flat in Y, and is uniform across every vertex.
            let floor_y = min[1] - FLOOR_PLANE_DROP_VOXELS;
            assert!(floor.iter().all(|v| v.position[1] == floor_y), "floor on dropped base plane");
        }
    }

    fn box_node(name: &str, offset: [i64; 3], density: u32) -> Node {
        let shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks: [2, 2, 2],
            voxels_per_block: density,
            wall_blocks: 1,
        };
        let mut node = Node::new(name, NodeContent::Tool { shape, material: Mc::Stone });
        node.transform.offset_blocks = offset;
        node
    }

    /// Gating (issue #29 S3): a node's lattice box appears in the batch ONLY when the
    /// master AND the node's per-object toggle are both ON; turning EITHER off drops
    /// it. A two-node scene with the grid enabled on ONE node yields exactly ONE
    /// lattice box (the other node contributes none).
    #[test]
    fn scene_grid_boxes_gated_by_master_and_per_object() {
        for density in [1u32, 15, 16] {
            let mut scene = Scene {
                nodes: vec![
                    box_node("A", [0, 0, 0], density),
                    box_node("B", [8, 0, 0], density),
                ],
                active: Some(NodePath::root_index(0)),
                ..Scene::default()
            };
            scene.master_block_lattice = true;
            scene.master_floor_grid = true;

            // Both per-object toggles OFF → no boxes regardless of masters.
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty() && flr.is_empty(), "@d{density}: per-object OFF ⇒ no boxes");

            // Enable block lattice on node A ONLY.
            scene.nodes[0].grids.block_lattice = true;
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert_eq!(lat.len(), 1, "@d{density}: one node enabled ⇒ exactly one lattice box");
            assert!(flr.is_empty(), "@d{density}: floor still off");

            // Master OFF cancels it even though the node's flag is on.
            scene.master_block_lattice = false;
            let (lat, _flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty(), "@d{density}: master OFF ⇒ no lattice box (AND gating)");

            // Floor: node B's flag on + master on → one floor box, no lattice.
            scene.master_floor_grid = true;
            scene.nodes[1].grids.floor_grid = true;
            let (lat, flr) = scene_grid_boxes(&scene, density);
            assert!(lat.is_empty(), "@d{density}: lattice master still off");
            assert_eq!(flr.len(), 1, "@d{density}: one floor box from node B");
        }
    }
}
