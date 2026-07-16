use super::*;

/// std140-safe uniform block for the cuboid pass (ADR 0002 E3b-2). Carries the
/// camera matrix, the grid half-extent and density (driving the per-voxel texture
/// slice and the position-based grid overlay), the grid-overlay parameters, and
/// the per-material base colours (reused from the instanced step-3b modulation).
/// Every `vec3` is followed by a scalar so it never straddles a 16-byte boundary;
/// the four grid-line scalars then fill the slot before the `vec4` array (which
/// must be 16-aligned). Field order matches the WGSL `CuboidUniforms` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CuboidUniforms {
    view_projection: [[f32; 4]; 4],
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
    // Layer-range band clip (issue #12 parity) + debug-faces flag. The two band
    // bounds plus the debug flag plus a pad fill one 16-byte slot, so the colour
    // array below stays 16-aligned (matching the WGSL `CuboidUniforms`).
    band_min: f32,
    band_max: f32,
    debug_face_mode: f32,
    /// ADR 0012 (H1): the onion GHOST flag (0 = normal solid render, 1 = flat
    /// translucent ghost tint). Occupies the former `_band_pad` slot; `0.0` for the
    /// solid draw keeps the solid uniform bytes identical (non-onion goldens byte-green).
    ghost_mode: f32,
    material_base_colors: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Per-material atlas sub-rect (ADR 0002 E3c-1 / O8), indexed by `material_id`:
    /// `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]`. The shader maps the
    /// per-voxel slice's `fract`-tiled UV into this window of the single atlas, so a
    /// chunk of mixed materials is ONE mesh = ONE draw (no per-material texture
    /// bind). Each `vec4` is naturally 16-aligned.
    material_atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// ADR 0012 (H1): the onion ghost tint (linear RGB + src alpha), read only when
    /// `ghost_mode > 0.5`. Appended so the solid draw's uniform layout is unchanged.
    ghost_tint: [f32; 4],
}

/// Convert a packed [`MaterialAtlas`]'s per-material sub-rects into the uniform
/// array layout `[inset_min_u, inset_min_v, inset_size_u, inset_size_v]` the shader
/// indexes by `material_id`. Materials without a packed sub-rect (should not happen
/// for the procedural set) fall back to the WHOLE atlas (`[0,0,1,1]`), so a missing
/// id degrades to "sample the atlas" rather than panicking.
pub(crate) fn atlas_rects_from(atlas: &MaterialAtlas) -> [[f32; 4]; MaterialChoice::MATERIAL_COUNT] {
    let mut rects = [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT];
    for (slot, sub_rect) in rects.iter_mut().zip(atlas.sub_rects.iter()) {
        let [size_u, size_v] = sub_rect.inset_size();
        *slot = [sub_rect.inset_min_u, sub_rect.inset_min_v, size_u, size_v];
    }
    rects
}

/// Build a ghost-only [`CuboidUniforms`] block (issue #78 — the selected-operand ghost
/// passes; ADR 0012 H1 is the ghost-branch precedent): `ghost_mode = 1` + `ghost_tint`,
/// with the camera + frame scalars the vertex stage reads. The `cuboid.wgsl` ghost branch
/// returns the flat tint before any texture / material / overlay / band read, so every
/// other field is filled with inert values (overlay + modulation off, band FULL).
pub(crate) fn flat_ghost_uniforms(
    view_projection: glam::Mat4,
    grid_dimensions: [u32; 3],
    voxels_per_block: u32,
    ghost_tint: [f32; 4],
) -> CuboidUniforms {
    let overlay = crate::renderer::grid_overlay_params();
    CuboidUniforms {
        view_projection: view_projection.to_cols_array_2d(),
        // FLOORED half, matching the solid draw's corner-anchoring (an odd dim's
        // `dim/2.0` would sit half a voxel off — see `update_uniforms`).
        grid_half_extent: [
            (grid_dimensions[0] / 2) as f32,
            (grid_dimensions[1] / 2) as f32,
            (grid_dimensions[2] / 2) as f32,
        ],
        voxels_per_block: voxels_per_block.max(1) as f32,
        voxel_line_color: overlay.voxel_line_color,
        grid_overlay_enabled: 0.0,
        block_line_color: overlay.block_line_color,
        material_modulation_enabled: 0.0,
        voxel_line_half_width: overlay.voxel_line_half_width,
        block_line_half_width: overlay.block_line_half_width,
        voxel_line_alpha: overlay.voxel_line_alpha,
        block_line_alpha: overlay.block_line_alpha,
        band_min: 0.0,
        band_max: u32::MAX as f32,
        debug_face_mode: 0.0,
        ghost_mode: 1.0,
        material_base_colors: [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
        material_atlas_rects: [[0.0, 0.0, 1.0, 1.0]; MaterialChoice::MATERIAL_COUNT],
        ghost_tint,
    }
}

/// The group(0) camera/frame uniform bind-group layout every cuboid-shader pipeline binds
/// (the solid/ghost draws in [`CuboidMeshRenderer::assemble`] and the selected-operand
/// ghost passes, issue #78). ONE builder so the layouts stay bind-compatible.
pub(crate) fn cuboid_uniform_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cuboid uniform bind group layout"),
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
    })
}

/// The per-draw on-face-grid overlay-active bind-group layout (group 2, ADR 0003 §3c / ADR
/// 0010 E3): one `u32` uniform read with a DYNAMIC OFFSET, so the overlay-off and overlay-on
/// draws of a chunk select `0` / `1` from a two-entry buffer without a per-vertex flag.
pub(crate) fn overlay_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cuboid overlay-active bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<u32>() as u64),
            },
            count: None,
        }],
    })
}

/// Build the two-entry per-draw overlay-active uniform buffer + its dynamic-offset bind
/// group (ADR 0003 §3c). Entry 0 = `0` (overlay off), entry 1 (at the device's
/// `min_uniform_buffer_offset_alignment`) = `1` (overlay on). Returns the bind group and
/// the stride to pass as the dynamic offset for the overlay-on draw.
pub(crate) fn build_overlay_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, u32) {
    let stride = device
        .limits()
        .min_uniform_buffer_offset_alignment
        .max(std::mem::size_of::<u32>() as u32);
    // Two `u32` entries, each at a `stride`-aligned offset (the rest is padding).
    let mut bytes = vec![0u8; (stride as usize) + std::mem::size_of::<u32>()];
    bytes[0..4].copy_from_slice(&0u32.to_ne_bytes()); // entry 0: overlay OFF
    bytes[stride as usize..stride as usize + 4].copy_from_slice(&1u32.to_ne_bytes()); // entry 1: overlay ON
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cuboid overlay-active uniform"),
        contents: &bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cuboid overlay-active bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: std::num::NonZeroU64::new(std::mem::size_of::<u32>() as u64),
            }),
        }],
    });
    (bind_group, stride)
}

/// The cuboid atlas bind-group layout: a single 2D texture (binding 0) + sampler
/// (binding 1). One atlas for ALL materials replaces the former per-material
/// D2Array binds (ADR 0002 O8).
pub(crate) fn build_atlas_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cuboid atlas bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
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

/// Upload a packed [`MaterialAtlas`] image as a single RGBA8 sRGB 2D texture
/// (Nearest, no mipmaps), matching the instanced path's sRGB decode so lighting +
/// overlay run in linear space and the sRGB target re-encodes on write.
pub(crate) fn upload_atlas_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    atlas: &MaterialAtlas,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: atlas.width.max(1),
        height: atlas.height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cuboid material atlas"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        &atlas.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * atlas.width.max(1)),
            rows_per_image: Some(atlas.height.max(1)),
        },
        size,
    );
    texture
}

/// One render chunk's GPU buffers for the cuboid path (issue #20 S6c-2d): its own
/// vertex + index buffer, the index count, and the world AABB for frustum culling.
/// Mirrors the instanced [`crate::renderer::InstancedChunkBuffers`]. A chunk that
/// meshes to zero faces is never stored (no buffer allocated).
pub(crate) struct CuboidChunkBuffers {
    vertex_buffer: wgpu::Buffer,
    /// One index buffer holding the overlay-OFF run followed by the overlay-ON run (ADR
    /// 0003 §3c). `index_count` is the overlay-off run length (drawn with the per-draw
    /// overlay-active uniform = 0); `index_count_overlay` is the overlay-on run, drawn at
    /// byte offset `index_count * 4` with the uniform = 1. Splitting by overlay state into
    /// two draws keeps the render flag out of the vertex format while preserving the
    /// per-object overlay behaviour.
    index_buffer: wgpu::Buffer,
    index_count: u32,
    index_count_overlay: u32,
    aabb: Aabb,
    /// Boxes this chunk decomposed into (diagnostic). Retained per chunk so the
    /// renderer's `total_box_count` can be recomputed exactly after an INCREMENTAL
    /// rebuild touches only a subset of chunks (an incremental update can't sum from
    /// the freshly-built meshes alone — the untouched chunks' buffers carry it).
    box_count: u32,
}

impl CuboidChunkBuffers {
    /// Record one indexed draw over the chunk's WHOLE index buffer (the overlay-off run
    /// followed by the overlay-on run together) into an already-begun pass. The draw
    /// style of a ghost pass: the flat-tinting ghost branch ignores the on-face grid
    /// overlay, so the ADR 0003 §3c two-draw split is unnecessary — one draw suffices
    /// (the onion ghost in [`CuboidMeshRenderer::draw_ghost`] and the selected-operand
    /// ghost passes, issue #78, both draw this way). A no-op for an empty chunk.
    pub(crate) fn draw_all_runs(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        let total = self.index_count + self.index_count_overlay;
        if total == 0 {
            return;
        }
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..total, 0, 0..1);
    }
}

/// All GPU resources for drawing the cuboid mesh (DEFAULT render path; per-chunk
/// buffers since issue #20 S6c-2d).
pub struct CuboidMeshRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Face-orientation debug pipeline: identical to `pipeline` except
    /// `cull_mode: None`, so a back face that is the nearest surface (a winding
    /// bug) still draws and is flagged by the shader's `front_facing` marker.
    /// Selected in `draw` when `debug_face_mode` is on — mirroring the instanced
    /// path's cull-off debug pipeline.
    debug_pipeline: wgpu::RenderPipeline,
    /// Loaded-VS-block pipelines (part of #20): same vertex layout + uniform group,
    /// but group(1) is a 6-layer D2Array (the block's per-face textures) instead of
    /// the procedural atlas, and the shader (`cuboid_loaded.wgsl`) selects the face
    /// layer FROM THE FACE NORMAL — exactly like the instanced loaded path. Selected
    /// in `draw` when a loaded material's bind group is supplied (else the procedural
    /// atlas pipelines above run, unchanged). The debug variant is cull-off.
    loaded_pipeline: wgpu::RenderPipeline,
    loaded_debug_pipeline: wgpu::RenderPipeline,
    /// Whether the last `update_uniforms` requested debug-faces mode (selects the
    /// cull-off pipeline in `draw`, matching the uploaded `debug_face_mode` flag).
    debug_face_mode: bool,
    /// Per-chunk GPU buffers (issue #20 S6c-2d), keyed by absolute chunk coord (the
    /// coord `resident_render_chunks` reports). Replaces the single monolithic
    /// vertex/index buffer + `CuboidMesh.chunks` index ranges: each chunk owns its
    /// own buffers, meshed from its own per-chunk grid + a 1-voxel neighbour apron.
    chunk_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    /// Chunk coords (keys into `chunk_buffers`) that survived the last frustum cull;
    /// computed in `update_uniforms`, consumed in `draw`. Sorted for a deterministic
    /// draw order (cross-chunk order is pixel-irrelevant: opaque + depth-tested).
    visible_chunks: Vec<[i32; 3]>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Per-draw on-face-grid overlay-active bind group (group 2, ADR 0003 §3c / ADR 0010
    /// E3): a single tiny `u32` uniform read with a DYNAMIC OFFSET. The backing buffer
    /// holds the value `0` at offset 0 and `1` at offset `overlay_dynamic_stride`, so the
    /// overlay-off draw binds offset 0 and the overlay-on draw binds the stride — the
    /// per-draw uniform that replaced the per-vertex overlay flag (one bool per draw, §3c).
    overlay_bind_group: wgpu::BindGroup,
    /// The dynamic-offset stride between the two overlay-active uniform entries (the
    /// device's `min_uniform_buffer_offset_alignment`, rounded up from the `u32` value).
    overlay_dynamic_stride: u32,
    /// ONE atlas bind group (ADR 0002 E3c-1 / O8): all material textures packed
    /// into a single 2D atlas texture + sampler. Replaces the former per-material
    /// D2Array binds — a chunk of mixed materials is now one mesh = one draw, with
    /// the shader mapping each face's `material_id` to its atlas sub-rect (carried
    /// in the uniforms). Clamp-to-edge sampler: the shader tiles the per-voxel slice
    /// itself via `fract` mapped into the sub-rect (a Repeat sampler would wrap into
    /// a neighbouring material's cell).
    atlas_bind_group: wgpu::BindGroup,
    /// The packed atlas's per-material sub-rects (inset sampling window), uploaded
    /// in the per-frame uniforms so the shader maps `material_id` → atlas window.
    atlas_rects: [[f32; 4]; MaterialChoice::MATERIAL_COUNT],
    /// Which procedural material the per-frame modulation was bound to.
    /// `update_uniforms` records it (drives the per-box base-colour modulation only;
    /// the atlas is bound once regardless of material).
    bound_material: MaterialChoice,
    /// The per-chunk grids the mesh was last built from (OWNED copies), retained so
    /// the mesh can be re-built CLIPPED to a new layer-range band (issue #12 parity)
    /// without the caller re-supplying them. The cuboid band clip masks each chunk's
    /// region before decomposition (real cap faces), so a band change re-meshes; we
    /// cache the last band and rebuild only when it differs.
    source_chunk_grids: Vec<([i32; 3], VoxelGrid)>,
    /// The two-layer chunks the mesh was last built from (ADR 0010 #53), retained so a band
    /// reclip (the layer scrubber) can re-mesh DIRECTLY from the two-layer store — no dense
    /// source grids. Empty on the dense path; populated only by [`new_from_two_layer_chunks`].
    /// `recentre`/`density` are the frame + density the two-layer mesher needs to re-emit in
    /// the SAME world frame on every band change.
    source_two_layer_chunks: Vec<([i32; 3], Arc<evaluation::two_layer_store::TwoLayerChunk>)>,
    source_two_layer_recentre: RecentreVoxels,
    source_two_layer_density: u32,
    /// The whole composite grid's voxel dims (the band clip maps an absolute layer to
    /// the global region-local Z; only the Z half is used).
    source_grid_dimensions: [u32; 3],
    /// Total boxes across all chunks the last build produced (diagnostic).
    total_box_count: u32,
    current_band: LayerBand,
    /// The loaded-VS-block material bind-group layout (a 6-layer D2Array + sampler,
    /// from [`crate::renderer::build_face_material_layout`]). Retained so a
    /// runtime-loaded block (M6/M7) can build a bind group of the SAME shape via
    /// [`Self::material_bind_group_layout`] and be drawn by the loaded pipeline.
    loaded_material_layout: wgpu::BindGroupLayout,
    /// The shared material sampler (nearest, clamp-to-edge) reused by loaded
    /// materials so they slice/filter exactly like the procedural atlas. Exposed via
    /// [`Self::material_sampler`].
    loaded_material_sampler: wgpu::Sampler,
    // --- ADR 0012 (H1): the onion GHOST pass ---
    /// The ghost pipeline: the SAME procedural `cuboid.wgsl` vertex/fragment (its
    /// `ghost_mode` branch flat-tints), but alpha-blended over the solid with the depth
    /// test ON (`Less`) and depth WRITE OFF, so solid geometry occludes the ghost and
    /// the ghost occludes nothing. Used for BOTH procedural and loaded-material scenes
    /// (the ghost never textures — flat tint even over `cuboid_loaded`).
    ghost_pipeline: wgpu::RenderPipeline,
    /// The ghost draw's uniform buffer (`ghost_mode = 1` + tint), separate from the
    /// solid `uniform_buffer` so the same frame carries both states.
    ghost_uniform_buffer: wgpu::Buffer,
    ghost_uniform_bind_group: wgpu::BindGroup,
    /// The GHOST geometry: two thin per-slab meshes clipped to the onion slabs below /
    /// above the band (`[band_min − depth, band_min)` and `(band_max, band_max + depth]`,
    /// ADR 0012). Built via the SAME banded mesher the solid uses (so the two paths — and
    /// the dense vs two-layer builds — ghost identically), just at the slab bands. Empty
    /// when onion is off. Kept as two maps because a tall chunk can straddle both slabs.
    ghost_lower_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    ghost_upper_buffers: std::collections::HashMap<[i32; 3], CuboidChunkBuffers>,
    /// The band the ghost slabs were last built for (`None` = never built / cleared), so
    /// a same-band frame skips the slab re-mesh and a band change (or the first frame
    /// after an async swap that built only the solid) rebuilds them.
    ghost_built_band: Option<LayerBand>,
}

impl CuboidMeshRenderer {
    /// Build the cuboid renderer from a WHOLE grid (the wrapper kept for `shot.rs`
    /// and tests that have a monolithic grid). Buckets the grid into per-chunk
    /// sub-grids by `floor(world_position / chunk_extent)` — the SAME key the
    /// instanced `crate::renderer::VoxelRenderer::rebuild_instances` wrapper uses —
    /// then meshes per chunk with an apron via [`Self::new_from_chunks`]. So a build
    /// from the whole grid is byte-identical to a build from the resolve cache's
    /// per-chunk accessor.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
        voxels_per_block: u32,
    ) -> Self {
        let buckets = bucket_grid_into_chunk_grids(grid, voxels_per_block);
        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> =
            buckets.iter().map(|(coord, g)| (*coord, g)).collect();
        Self::new_from_chunks(
            device,
            queue,
            color_format,
            &chunk_refs,
            grid.dimensions,
        )
    }

    /// Build the cuboid renderer DIRECTLY from the resolve cache's per-chunk grids
    /// (issue #20 S6c-2d). `chunk_grids` is `resident_render_chunks`'s output
    /// (`(absolute_chunk_coord, &rebased_grid)` per covering chunk); `grid_dimensions`
    /// is the whole composite grid's voxel dims (the band-clip layer mapping). Meshes
    /// every chunk with a 1-voxel neighbour apron (see [`build_chunk_meshes_with_apron`])
    /// and stores one [`CuboidChunkBuffers`] per non-empty chunk.
    pub fn new_from_chunks(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunk_grids: &[([i32; 3], &VoxelGrid)],
        grid_dimensions: [u32; 3],
    ) -> Self {
        profiling::scope!("cuboid_mesh_build");
        let source_chunk_grids: Vec<([i32; 3], VoxelGrid)> = chunk_grids
            .iter()
            .map(|(coord, grid)| (*coord, (*grid).clone()))
            .collect();
        let chunk_meshes =
            build_chunk_meshes_with_apron(chunk_grids, grid_dimensions, LayerBand::FULL);
        Self::assemble(
            device,
            queue,
            color_format,
            chunk_meshes,
            source_chunk_grids,
            grid_dimensions,
        )
    }

    /// Build the cuboid renderer from a [`TwoLayerChunk`] per covering chunk (ADR 0010 E3):
    /// a coarse-solid block becomes a ONE-BOX fast path, a boundary block its stored
    /// microblock cuboids, and inter-block / inter-chunk seam faces are culled via the
    /// per-face seam-solidity flags (plus the neighbour coarse layer) — NOT a densified
    /// apron. The emitted exposed-face set is proven identical to the dense
    /// `new_from_chunks` path (the E3 parity gate), so it renders pixel-identical.
    ///
    /// `chunks` is `(absolute_chunk_coord, TwoLayerChunk)` per covering chunk;
    /// `grid_dimensions` is the whole composite voxel dims; `recentre_voxels` is the
    /// resolve's carried recentre (ADR 0008) so the two-layer mesh lands in the SAME world
    /// frame the dense path assembles. The INITIAL build is FULL-band (the E3 fast paths);
    /// the two-layer chunks are RETAINED so a later band reclip (the layer scrubber, ADR
    /// 0010 #53) re-meshes DIRECTLY from the store — no dense source grids needed.
    pub fn new_from_two_layer_chunks(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunks: &[([i32; 3], Arc<evaluation::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
    ) -> Self {
        // The synchronous path builds the FULL model (no band clip). Delegates to the banded
        // builder with `LayerBand::FULL` so its output is byte-identical to before (goldens
        // + gpu_parity stay pixel-exact).
        Self::new_from_two_layer_chunks_banded(
            device,
            queue,
            color_format,
            chunks,
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            LayerBand::FULL,
        )
    }

    /// As `new_from_two_layer_chunks`, but builds the mesh already CLIPPED to `band`
    /// (issue #60 M2). The async worker uses this so the swapped-in renderer already matches
    /// the active `effective_band` — the swap frame then does NOT trigger a full synchronous
    /// `rebuild_for_band` re-mesh on the main thread (the multi-second hitch #60 removed,
    /// which would fire on EVERY async swap during onion-skin scrubbing). Sets `current_band`
    /// so the per-frame `update_uniforms` treats the band as already applied. `LayerBand::FULL`
    /// is identical to the plain builder.
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_two_layer_chunks_banded(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunks: &[([i32; 3], Arc<evaluation::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
        band: LayerBand,
    ) -> Self {
        profiling::scope!("cuboid_mesh_build_two_layer");
        let chunk_meshes = build_two_layer_chunk_meshes(
            chunks,
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            band,
        );
        let mut renderer = Self::assemble(
            device,
            queue,
            color_format,
            chunk_meshes,
            Vec::new(),
            grid_dimensions,
        );
        // Retain the two-layer chunks + frame so `rebuild_for_band` re-meshes the band
        // slab from the store (ADR 0010 #53) — the layer scrubber on the two-layer path.
        renderer.source_two_layer_chunks = chunks.to_vec();
        renderer.source_two_layer_recentre = recentre_voxels;
        renderer.source_two_layer_density = voxels_per_block.max(1);
        // The mesh was built AT `band`, so record it — a same-band `update_uniforms` is then
        // a no-op instead of a full re-mesh (M2). A later band change still re-clips.
        renderer.current_band = band;
        renderer
    }

    /// Shared GPU-resource assembly for both the dense ([`new_from_chunks`]) and two-layer
    /// ([`new_from_two_layer_chunks`]) builders: upload the per-chunk meshes, build the
    /// uniform / per-draw-overlay / atlas / loaded bind groups + pipelines, and assemble
    /// the renderer. `source_chunk_grids` is retained for the band reclip (empty on the
    /// two-layer path, which stays FULL-band until E5).
    fn assemble(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        chunk_meshes: Vec<CuboidChunkMesh>,
        source_chunk_grids: Vec<([i32; 3], VoxelGrid)>,
        grid_dimensions: [u32; 3],
    ) -> Self {
        let total_box_count = chunk_meshes.iter().map(|m| m.box_count).sum();
        let chunk_buffers = upload_chunk_meshes(device, &chunk_meshes);

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cuboid uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group_layout = cuboid_uniform_bind_group_layout(device);
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // ADR 0012 (H1): the onion ghost draw's own uniform buffer + bind group (same
        // layout as the solid, a separate buffer so one frame carries both states).
        let ghost_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cuboid ghost uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ghost_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid ghost uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ghost_uniform_buffer.as_entire_binding(),
            }],
        });

        // --- Per-draw on-face-grid overlay-active uniform (group 2, ADR 0003 §3c) ---
        // The overlay flag is no longer a vertex attribute (ADR 0010 E3): a chunk mesh is
        // split into an overlay-off and an overlay-on draw, each selecting this per-draw
        // `u32` via a DYNAMIC OFFSET. Two entries — `0` then `1` — packed one
        // `min_uniform_buffer_offset_alignment` apart, so the off-draw binds offset 0 and
        // the on-draw binds the stride.
        let (overlay_bind_group, overlay_dynamic_stride) =
            build_overlay_bind_group(device, &overlay_bind_group_layout(device));

        // --- Material texture ATLAS (E3c-1 / ADR 0002 O8) ---
        // Pack ALL material textures (Stone/Wood/Plain) into ONE atlas image and
        // bind it as a SINGLE 2D texture, so a chunk of mixed materials is one mesh
        // = one draw (the Vintage Story approach) — no per-material texture bind.
        // Each face's `material_id` maps to its atlas sub-rect (uploaded in the
        // uniforms); the shader tiles the per-voxel slice INTO that sub-rect.
        //
        // Sampler is CLAMP-to-edge + Nearest (matching the instanced texel grid).
        // The per-voxel tiling can NOT use a Repeat sampler here — Repeat would wrap
        // to the WHOLE atlas, i.e. into a neighbour material — so the shader does the
        // `fract`-tiling into the sub-rect itself, and the atlas's replicated-edge
        // gutter + half-texel inset (see `texture_atlas`) defend the cell borders.
        let atlas = MaterialAtlas::from_procedural_materials();
        let atlas_rects = atlas_rects_from(&atlas);
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cuboid atlas sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let atlas_bind_group_layout = build_atlas_bind_group_layout(device);
        let atlas_texture = upload_atlas_texture(device, queue, &atlas);
        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cuboid atlas bind group"),
            layout: &atlas_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuboid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cuboid.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cuboid pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&atlas_bind_group_layout),
                // group(2): the per-draw overlay-active uniform (ADR 0003 §3c).
                Some(&overlay_bind_group_layout(device)),
            ],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CuboidVertex>() as u64,
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
                    format: wgpu::VertexFormat::Uint32,
                },
                // ADR 0003 §3c / ADR 0010 E3: the on-face-grid flag is NO LONGER a vertex
                // attribute — the chunk mesh is split into overlay-off / overlay-on draws,
                // each selecting a per-draw `grid_overlay_active` uniform (group 2).
            ],
        };

        // Build the render pipeline, parameterized by cull mode: the normal pass
        // back-culls; the debug-faces pass disables culling so a back face that is
        // the nearest surface (a winding bug) still draws and is flagged by the
        // shader's `front_facing` marker — exactly like the instanced path's
        // cull-on / cull-off pipeline pair.
        let build_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
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
                    cull_mode,
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
            })
        };
        let pipeline = build_pipeline("cuboid pipeline", Some(wgpu::Face::Back));
        let debug_pipeline = build_pipeline("cuboid debug pipeline", None);

        // ADR 0012 (H1): the onion GHOST pipeline. Same shader + layout as the solid, but
        // alpha-blends the flat-tinted ghost OVER the solid, depth-tested `Less`. Depth WRITE
        // is ON (not off): each pixel then shows only the NEAREST ghost surface, blended once
        // — NOT an order-dependent accumulation of every overlapping translucent face. This
        // makes the ghost render a pure function of the visible surface, so it is IDENTICAL
        // across the display paths whose greedy decomposition / raymarch differ face-for-face
        // (dense vs two-layer mesh, and the brick raymarch) exactly as the OPAQUE solid render
        // already matches — the `brick_golden_matches_dense` / two-layer cross-checks depend
        // on it. Solid geometry (drawn first) still occludes the ghost via the same depth
        // buffer; the ghost may occlude the depth-tested overlays drawn after it, which for a
        // translucent context slab is acceptable. Back-face culled like the solid.
        let ghost_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cuboid onion ghost pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: std::slice::from_ref(&vertex_layout),
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
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        // --- Loaded-VS-block pipelines (part of #20) ---
        // A second shader + pipeline pair that binds the applied block's 6-layer
        // D2Array at group(1) (built externally by `LoadedMaterial`, against the
        // SAME `build_face_material_layout` descriptor used here, so the bind group
        // is layout-compatible) and selects the per-face layer by the face normal.
        // It shares the uniform group(0) and the same vertex layout, so a loaded
        // block renders pixel-aligned with the procedural geometry — only the
        // texture source differs. The procedural atlas pipelines stay the default.
        let loaded_material_layout = crate::renderer::build_face_material_layout(device);
        // The shared material sampler (nearest, clamp-to-edge) — reused by loaded VS
        // blocks so they slice/filter exactly like the procedural atlas. Retained on
        // the renderer and exposed so the app can build a `LoadedMaterial` against it.
        let loaded_material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cuboid loaded material sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let loaded_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cuboid loaded-block shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cuboid_loaded.wgsl").into()),
        });
        let loaded_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cuboid loaded pipeline layout"),
                bind_group_layouts: &[
                    Some(&uniform_bind_group_layout),
                    Some(&loaded_material_layout),
                    // group(2): the per-draw overlay-active uniform (ADR 0003 §3c).
                    Some(&overlay_bind_group_layout(device)),
                ],
                immediate_size: 0,
            });
        let build_loaded_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&loaded_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &loaded_shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &loaded_shader,
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
                    cull_mode,
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
            })
        };
        let loaded_pipeline = build_loaded_pipeline("cuboid loaded pipeline", Some(wgpu::Face::Back));
        let loaded_debug_pipeline = build_loaded_pipeline("cuboid loaded debug pipeline", None);

        // Every resident chunk visible until the next frustum cull in `update_uniforms`.
        let mut visible_chunks: Vec<[i32; 3]> = chunk_buffers.keys().copied().collect();
        visible_chunks.sort_unstable();

        Self {
            pipeline,
            debug_pipeline,
            loaded_pipeline,
            loaded_debug_pipeline,
            debug_face_mode: false,
            chunk_buffers,
            visible_chunks,
            uniform_buffer,
            uniform_bind_group,
            overlay_bind_group,
            overlay_dynamic_stride,
            atlas_bind_group,
            atlas_rects,
            bound_material: MaterialChoice::Plain,
            source_chunk_grids,
            // The dense builders retain no two-layer chunks; `new_from_two_layer_chunks`
            // overrides these after `assemble` so its band reclip re-meshes from the store.
            source_two_layer_chunks: Vec::new(),
            source_two_layer_recentre: RecentreVoxels::new([0; 3]),
            source_two_layer_density: 1,
            source_grid_dimensions: grid_dimensions,
            total_box_count,
            current_band: LayerBand::FULL,
            loaded_material_layout,
            loaded_material_sampler,
            ghost_pipeline,
            ghost_uniform_buffer,
            ghost_uniform_bind_group,
            ghost_lower_buffers: std::collections::HashMap::new(),
            ghost_upper_buffers: std::collections::HashMap::new(),
            ghost_built_band: None,
        }
    }

    /// Incrementally update the per-chunk buffers for a geometry edit (issue #40):
    /// re-mesh + re-upload ONLY the chunks the edit (and its apron neighbours) touched,
    /// drop vacated chunks, and KEEP every other chunk's existing buffers — instead of
    /// the wholesale `new_from_chunks` recreate (the measured ~600ms/edit GPU cost).
    ///
    /// `chunk_grids` is the FULL post-edit covering set (`resident_render_chunks`),
    /// needed IN FULL so the re-meshed chunks' aprons see every neighbour; `grid_dimensions`
    /// is the whole composite's voxel dims (band-clip mapping); `evicted_dirty` is the
    /// resolve cache's evicted coords for this edit (from `invalidate_aabb`).
    ///
    /// PRECONDITION: the floating origin did NOT shift since the last rebuild. Chunk
    /// grids are stored pre-rebased against the composite recentre, so a recentre shift
    /// staleens EVERY buffer — the caller must fall back to `new_from_chunks` then. The
    /// active layer band is preserved (re-meshes at `self.current_band`).
    pub fn incremental_rebuild_from_chunks(
        &mut self,
        device: &wgpu::Device,
        chunk_grids: &[([i32; 3], &VoxelGrid)],
        grid_dimensions: [u32; 3],
        evicted_dirty: &[[i32; 3]],
    ) {
        profiling::scope!("cuboid_mesh_incremental");
        self.source_grid_dimensions = grid_dimensions;

        // The renderer's KNOWN set is its source grids' coords (includes occupied-but-
        // fully-occluded chunks that carry no buffer), so occluded chunks stay stable
        // instead of being treated as "new" and re-meshed every edit.
        let resident: Vec<[i32; 3]> = self.source_chunk_grids.iter().map(|(c, _)| *c).collect();
        let occupied: Vec<[i32; 3]> = chunk_grids
            .iter()
            .filter(|(_, grid)| !grid.occupied.is_empty())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = cuboid_incremental_plan(&resident, evicted_dirty, &occupied);

        // Re-mesh only the dirty-dilated subset (aprons from the full set) at the
        // active band, then upload those chunks' buffers.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let meshes = build_chunk_meshes_with_apron_filtered(
            chunk_grids,
            Some(&rebuild_set),
            grid_dimensions,
            self.current_band,
        );
        let rebuilt_buffers = upload_chunk_meshes(device, &meshes);

        // Apply. Drop evicted buffers, then drop EVERY rebuild coord's old buffer (a
        // rebuild coord that now meshes to EMPTY — e.g. fully occluded by new neighbour
        // occupancy — produces no buffer and must lose its stale one), then insert the
        // freshly built buffers. Net result == wholesale rebuild's buffer set.
        let grids_by_coord: std::collections::HashMap<[i32; 3], &VoxelGrid> =
            chunk_grids.iter().map(|(coord, grid)| (*coord, *grid)).collect();
        for coord in &plan.evict {
            self.chunk_buffers.remove(coord);
        }
        for coord in &plan.rebuild {
            self.chunk_buffers.remove(coord);
        }
        self.chunk_buffers.extend(rebuilt_buffers);

        // Keep `source_chunk_grids` the COMPLETE current covering set (a later band
        // re-clip reads it for global occupancy): drop evicted, upsert each rebuilt
        // coord's grid. Untouched chunks are resolve-cache hits → already correct.
        let evict_set: std::collections::HashSet<[i32; 3]> = plan.evict.iter().copied().collect();
        self.source_chunk_grids
            .retain(|(coord, _)| !evict_set.contains(coord));
        for coord in &plan.rebuild {
            if let Some(grid) = grids_by_coord.get(coord) {
                match self.source_chunk_grids.iter_mut().find(|(c, _)| c == coord) {
                    Some(entry) => entry.1 = (*grid).clone(),
                    None => self.source_chunk_grids.push((*coord, (*grid).clone())),
                }
            }
        }

        // Recompute the diagnostics from the (now-correct) full buffer set. All chunks
        // visible until the next frustum cull in `update_uniforms`.
        self.total_box_count = self.chunk_buffers.values().map(|c| c.box_count).sum();
        self.visible_chunks = self.chunk_buffers.keys().copied().collect();
        self.visible_chunks.sort_unstable();
    }

    /// Incrementally update the per-chunk buffers for a geometry edit on the **two-layer**
    /// path (issue #55 — the two-layer analogue of `incremental_rebuild_from_chunks`):
    /// re-mesh + re-upload ONLY the chunks the edit (and its 26-neighbourhood seam footprint)
    /// touched, drop vacated chunks, and KEEP every other chunk's existing buffers — instead
    /// of the wholesale `new_from_two_layer_chunks` recreate that re-meshes + re-uploads the
    /// WHOLE resident set every edit (the exact per-edit latency #40 fixed for the dense path,
    /// regressed onto the two-layer live renderer after E5).
    ///
    /// `chunks` is the FULL post-edit covering set (the `TwoLayerResidentCache`'s resident
    /// chunks), needed IN FULL so the re-meshed chunks' seam-flag culling consults every
    /// neighbour; `recentre_voxels` / `voxels_per_block` are the resolve's carried frame
    /// (ADR 0008); `grid_dimensions` the whole composite's voxel dims (band-clip mapping);
    /// `evicted_dirty` the resident cache's evicted coords for this edit (from
    /// [`TwoLayerResidentCache::invalidate_aabb`](evaluation::two_layer_store::TwoLayerResidentCache::invalidate_aabb)).
    ///
    /// The dirty set is dilated by the 26-neighbourhood via the SAME
    /// [`cuboid_incremental_plan`] the dense path uses — the seam-solidity dependency footprint
    /// is that same 26-neighbourhood (a neighbour's coarse / microblock face occupancy can cull
    /// this chunk's seam faces). Applying the plan — re-mesh `rebuild`, drop `evict`, keep the
    /// rest — yields a per-chunk buffer set IDENTICAL to a wholesale two-layer rebuild (proven
    /// by `incremental_two_layer_gpu_buffer_rebuild_equals_wholesale`).
    ///
    /// PRECONDITION: this must be the two-layer path (built via
    /// `new_from_two_layer_chunks`). A two-layer chunk is chunk-local-integer (ADR 0008), so
    /// — unlike the dense path — a floating-origin recentre SHIFT does NOT staleen the resident
    /// buffers (the recentre is a pure index offset re-applied here as `recentre_voxels`); the
    /// caller need not fall back on a recentre shift, only on a DENSITY change (which resizes
    /// every chunk's voxel extent and re-keys the whole buffer set). The active layer band is
    /// preserved (re-meshes at `self.current_band`).
    pub fn incremental_rebuild_from_two_layer_chunks(
        &mut self,
        device: &wgpu::Device,
        chunks: &[([i32; 3], Arc<evaluation::two_layer_store::TwoLayerChunk>)],
        grid_dimensions: [u32; 3],
        recentre_voxels: RecentreVoxels,
        voxels_per_block: u32,
        evicted_dirty: &[[i32; 3]],
    ) {
        profiling::scope!("cuboid_mesh_incremental_two_layer");
        self.source_grid_dimensions = grid_dimensions;
        self.source_two_layer_recentre = recentre_voxels;
        self.source_two_layer_density = voxels_per_block.max(1);

        // The renderer's KNOWN set is its retained two-layer chunks' coords (includes
        // occupied-but-fully-occluded chunks that carry no buffer), so occluded chunks stay
        // stable instead of being treated as "new" and re-meshed every edit.
        let resident: Vec<[i32; 3]> =
            self.source_two_layer_chunks.iter().map(|(c, _)| *c).collect();
        let occupied: Vec<[i32; 3]> = chunks
            .iter()
            .filter(|(_, chunk)| chunk.has_geometry())
            .map(|(coord, _)| *coord)
            .collect();
        let plan = cuboid_incremental_plan(&resident, evicted_dirty, &occupied);

        // Re-mesh only the dirty-dilated subset (seam culling from the full set) at the
        // active band, then upload those chunks' buffers.
        let rebuild_set: std::collections::HashSet<[i32; 3]> =
            plan.rebuild.iter().copied().collect();
        let meshes = build_two_layer_chunk_meshes_filtered(
            chunks,
            Some(&rebuild_set),
            grid_dimensions,
            recentre_voxels,
            voxels_per_block,
            self.current_band,
        );
        let rebuilt_buffers = upload_chunk_meshes(device, &meshes);

        // Apply. Drop evicted buffers, then drop EVERY rebuild coord's old buffer (a rebuild
        // coord that now meshes to EMPTY — e.g. fully occluded by new neighbour occupancy —
        // produces no buffer and must lose its stale one), then insert the freshly built
        // buffers. Net result == wholesale two-layer rebuild's buffer set.
        for coord in &plan.evict {
            self.chunk_buffers.remove(coord);
        }
        for coord in &plan.rebuild {
            self.chunk_buffers.remove(coord);
        }
        self.chunk_buffers.extend(rebuilt_buffers);

        // Keep `source_two_layer_chunks` the COMPLETE current covering set (a later band
        // reclip re-meshes from it): drop evicted, upsert each rebuilt coord's chunk.
        // Untouched chunks are resident-cache hits → already correct. Rebuilding a chunk that
        // went all-air still upserts its (empty) chunk so the retained set matches `chunks`.
        let chunks_by_coord: std::collections::HashMap<[i32; 3], &Arc<evaluation::two_layer_store::TwoLayerChunk>> =
            chunks.iter().map(|(coord, chunk)| (*coord, chunk)).collect();
        let evict_set: std::collections::HashSet<[i32; 3]> = plan.evict.iter().copied().collect();
        self.source_two_layer_chunks
            .retain(|(coord, _)| !evict_set.contains(coord));
        for coord in &plan.rebuild {
            if let Some(&chunk) = chunks_by_coord.get(coord) {
                // `Arc::clone` (O(1)) — the retained source set shares the resident chunk,
                // never deep-copies it.
                match self
                    .source_two_layer_chunks
                    .iter_mut()
                    .find(|(c, _)| c == coord)
                {
                    Some(entry) => entry.1 = Arc::clone(chunk),
                    None => self.source_two_layer_chunks.push((*coord, Arc::clone(chunk))),
                }
            }
        }

        // Recompute the diagnostics from the (now-correct) full buffer set. All chunks
        // visible until the next frustum cull in `update_uniforms`.
        self.total_box_count = self.chunk_buffers.values().map(|c| c.box_count).sum();
        self.visible_chunks = self.chunk_buffers.keys().copied().collect();
        self.visible_chunks.sort_unstable();
    }

    /// The loaded-VS-block material bind-group layout (6-layer D2Array texture +
    /// sampler). Exposed so a runtime-loaded block (M6) can build a bind group of the
    /// SAME shape (via `LoadedMaterial`) and be drawn by the loaded pipeline.
    pub fn material_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.loaded_material_layout
    }

    /// The shared material sampler (nearest, clamp-to-edge) — reused by loaded
    /// materials so they slice/filter exactly like the procedural atlas.
    pub fn material_sampler(&self) -> &wgpu::Sampler {
        &self.loaded_material_sampler
    }

    /// Re-mesh the stored per-chunk grids CLIPPED to `band` (issue #12 parity) and
    /// re-upload every chunk's buffers, when `band` differs from the last build. The
    /// cuboid band clip masks each chunk's region before decomposition so the band
    /// edges get real cap faces, so it must rebuild geometry (a fragment discard
    /// would leave a merged column's slab open-topped). No-op when the band is
    /// unchanged.
    fn rebuild_for_band(&mut self, device: &wgpu::Device, band: LayerBand) {
        // --- SOLID geometry (clipped to the exact [band_min, band_max]; `onion_depth` is
        // NOT a solid input, so the solid band is unchanged by ADR 0012). Skipped when the
        // band is unchanged (the M2 no-swap-rehitch property). ---
        if band != self.current_band {
            self.current_band = band;
            if let Some(chunk_meshes) = self.build_band_meshes(band) {
                self.total_box_count = chunk_meshes.iter().map(|m| m.box_count).sum();
                self.chunk_buffers = upload_chunk_meshes(device, &chunk_meshes);
                // All chunks visible until the next frustum cull in `update_uniforms`.
                self.visible_chunks = self.chunk_buffers.keys().copied().collect();
                self.visible_chunks.sort_unstable();
            }
            // A source-less (empty) build leaves the geometry in place (matches pre-0012).
        }

        // --- GHOST geometry (ADR 0012 H1): the thin per-slab onion meshes. Rebuilt on a
        // band change OR when never built for this band (the first frame after an async
        // swap that pre-built only the solid — the slabs are cheap, so this is not the
        // multi-second re-mesh #60 removed). ---
        if self.ghost_built_band != Some(band) {
            self.rebuild_ghost_slabs(device, band);
            self.ghost_built_band = Some(band);
        }
    }

    /// Build the per-chunk SOLID meshes clipped to `band` from whichever source the
    /// renderer retains (the two-layer store, else the dense per-chunk grids). `None`
    /// when the renderer has neither source (an empty build). The two-layer analogue of
    /// the dense apron mesher, kept as ONE helper so [`rebuild_for_band`] and the ghost
    /// slab build share the exact same clip semantics (ADR 0012: the two ghost slabs are
    /// just this build at the slab bands).
    fn build_band_meshes(&self, band: LayerBand) -> Option<Vec<CuboidChunkMesh>> {
        if !self.source_two_layer_chunks.is_empty() {
            return Some(build_two_layer_chunk_meshes(
                &self.source_two_layer_chunks,
                self.source_grid_dimensions,
                self.source_two_layer_recentre,
                self.source_two_layer_density,
                band,
            ));
        }
        if self.source_chunk_grids.is_empty() {
            return None;
        }
        let chunk_refs: Vec<([i32; 3], &VoxelGrid)> = self
            .source_chunk_grids
            .iter()
            .map(|(coord, g)| (*coord, g))
            .collect();
        Some(build_chunk_meshes_with_apron(
            &chunk_refs,
            self.source_grid_dimensions,
            band,
        ))
    }

    /// (ADR 0012 H1) Rebuild the two onion GHOST slab meshes for `band`: the layers
    /// `[band_min − depth, band_min)` (lower slab) and `(band_max, band_max + depth]`
    /// (upper slab), the recentred-Z remainder of the onion span `AppCore::onion_fog_params`
    /// derives (floored half, Z-up, depth clamped 1..8). Each slab is meshed by the SAME
    /// banded builder the solid uses, so it carries real cap faces at the slab edges — the
    /// brick raymarch ghost's per-slab traversal clamp produces the same caps, which is what
    /// keeps `brick_golden_matches_dense` green. Empty (both maps cleared) when onion is off
    /// (`onion_depth == 0`) or a slab falls outside the grid.
    fn rebuild_ghost_slabs(&mut self, device: &wgpu::Device, band: LayerBand) {
        self.ghost_lower_buffers.clear();
        self.ghost_upper_buffers.clear();
        if band.onion_depth == 0 {
            return;
        }
        let depth = band.onion_depth;
        let grid_z = self.source_grid_dimensions[2];
        let last_layer = grid_z.saturating_sub(1);
        // Lower slab: layers [band_min − depth, band_min − 1]. Skipped when the band bottom
        // is already layer 0 (nothing below to ghost).
        if band.band_min > 0 {
            let slab = LayerBand {
                band_min: band.band_min.saturating_sub(depth),
                band_max: band.band_min - 1,
                onion_depth: 0,
            };
            if let Some(meshes) = self.build_band_meshes(slab) {
                self.ghost_lower_buffers = upload_chunk_meshes(device, &meshes);
            }
        }
        // Upper slab: layers [band_max + 1, band_max + depth]. Skipped when the band top is
        // already the last layer (nothing above to ghost).
        if band.band_max < last_layer {
            let slab = LayerBand {
                band_min: band.band_max + 1,
                band_max: (band.band_max + depth).min(last_layer),
                onion_depth: 0,
            };
            if let Some(meshes) = self.build_band_meshes(slab) {
                self.ghost_upper_buffers = upload_chunk_meshes(device, &meshes);
            }
        }
    }

    /// Total exposed quad faces across all resident chunks (diagnostic, both overlay runs).
    pub fn face_count(&self) -> u32 {
        self.chunk_buffers
            .values()
            .map(|c| (c.index_count + c.index_count_overlay) / 6)
            .sum()
    }

    /// Total triangles across all resident chunks (diagnostic, both overlay runs).
    pub fn triangle_count(&self) -> u32 {
        self.chunk_buffers
            .values()
            .map(|c| (c.index_count + c.index_count_overlay) / 3)
            .sum()
    }

    /// Total boxes the last build decomposed into across all chunks (diagnostic).
    pub fn box_count(&self) -> u32 {
        self.total_box_count
    }

    /// Number of resident render chunks (non-empty cuboid meshes).
    pub fn chunk_count(&self) -> u32 {
        self.chunk_buffers.len() as u32
    }

    /// Number of chunks that survived the last frustum cull (will be drawn).
    pub fn visible_chunk_count(&self) -> u32 {
        self.visible_chunks.len() as u32
    }

    /// Upload the per-frame uniforms (camera matrix, grid half-extent + density
    /// for the per-voxel texture slice + grid overlay, grid-overlay params +
    /// toggle, per-material base colours) and frustum-cull the mesh chunks.
    ///
    /// `grid_dimensions` give the half-extent so `world + half` is the absolute
    /// voxel position the UV slice + overlay key off. `voxels_per_block` is the
    /// density (slice size + block-line period). `grid_overlay_enabled` reflects
    /// the Display toggle. `bound` is the active procedural material: it selects
    /// the bound texture (E3b-2) AND drives the relative base-colour modulation
    /// (exactly like the instanced step-3b). `None` means a loaded VS block is
    /// active: modulation is disabled here, and the loaded-block pipeline selected in
    /// `draw` (when its 6-layer D2Array bind group is supplied) ignores the
    /// procedural atlas/modulation uniforms entirely (part of #20).
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
        grid_overlay_enabled: bool,
        bound: Option<MaterialChoice>,
        band: LayerBand,
        debug_face_mode: bool,
    ) {
        // Layer-range band clip (issue #12 parity): re-mesh the grid clipped to the
        // band (real cap faces at the band edges) when it changed. Debug-faces mode
        // bypasses the band (the instanced check sees the whole model), so force the
        // full band while it is on.
        let effective_band = if debug_face_mode {
            LayerBand::FULL
        } else {
            band
        };
        self.rebuild_for_band(device, effective_band);
        // The bound procedural material drives BOTH the texture binding (selected
        // in `draw`) and the per-box modulation. A `None` (loaded VS block) falls
        // back to Plain's texture + neutral modulation for now (the cuboid path
        // renders a loaded block as a single global material this sub-step).
        // Debug-faces mode forces modulation off (the shader bypasses it anyway),
        // matching the instanced path.
        let (modulation_enabled, base_colors, material) = match bound {
            Some(material) if !debug_face_mode => (
                true,
                crate::renderer::relative_material_base_colors_public(material),
                material,
            ),
            Some(material) => (
                false,
                [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
                material,
            ),
            None => (
                false,
                [[1.0, 1.0, 1.0, 0.0]; MaterialChoice::MATERIAL_COUNT],
                MaterialChoice::Plain,
            ),
        };
        self.bound_material = material;
        // Record the debug flag so `draw` selects the matching cull-off pipeline.
        self.debug_face_mode = debug_face_mode;

        let overlay = crate::renderer::grid_overlay_params();
        let uniforms = CuboidUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            // Corner-anchoring: the grid's low corner is `−floor(dim/2)`, so the GPU
            // recovers the absolute voxel frame with `world_position + floor(dim/2)`
            // (integer-valued). Using `dim/2.0` would be half a voxel off for an ODD
            // dim, mis-snapping the voxel/block grid overlay and the Z-band clip.
            grid_half_extent: [
                (grid_dimensions[0] / 2) as f32,
                (grid_dimensions[1] / 2) as f32,
                (grid_dimensions[2] / 2) as f32,
            ],
            voxels_per_block: voxels_per_block.max(1) as f32,
            voxel_line_color: overlay.voxel_line_color,
            grid_overlay_enabled: if grid_overlay_enabled { 1.0 } else { 0.0 },
            block_line_color: overlay.block_line_color,
            material_modulation_enabled: if modulation_enabled { 1.0 } else { 0.0 },
            voxel_line_half_width: overlay.voxel_line_half_width,
            block_line_half_width: overlay.block_line_half_width,
            voxel_line_alpha: overlay.voxel_line_alpha,
            block_line_alpha: overlay.block_line_alpha,
            // Layer-range band clip (issue #12 parity): the shader keeps fragments
            // whose voxel layer is in [band_min, band_max] (both INCLUSIVE),
            // matching the instanced voxel pass. `LayerBand::FULL` uses band_max =
            // u32::MAX, so `as f32` (≈ 4.29e9) leaves every layer unclipped.
            band_min: band.band_min as f32,
            band_max: band.band_max as f32,
            debug_face_mode: if debug_face_mode { 1.0 } else { 0.0 },
            // ADR 0012 (H1): the SOLID draw is never the ghost — 0 here keeps the solid
            // uniform bytes identical to pre-onion-ghost (non-onion goldens byte-green).
            ghost_mode: 0.0,
            material_base_colors: base_colors,
            material_atlas_rects: self.atlas_rects,
            ghost_tint: [0.0, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        // ADR 0012 (H1) — the onion GHOST uniform. Identical camera/frame to the solid,
        // but `ghost_mode = 1` (flat translucent tint) + the tint colour. Both onion
        // slabs share this ONE uniform (the slab distinction lives in the per-slab GHOST
        // geometry, not the uniform), so a band scrub only re-meshes the thin slabs and
        // never touches this buffer's shape. The tint is the SAME constant the brick
        // ghost binds — `brick_golden_matches_dense` depends on the two matching.
        let ghost_uniforms = CuboidUniforms {
            ghost_mode: 1.0,
            ghost_tint: crate::renderer::onion_ghost_tint(),
            ..uniforms
        };
        queue.write_buffer(&self.ghost_uniform_buffer, 0, bytemuck::bytes_of(&ghost_uniforms));

        // Frustum-cull the per-chunk buffers by their world AABBs (sorted for a
        // deterministic draw order; cross-chunk order is pixel-irrelevant — opaque +
        // depth-tested).
        let frustum = Frustum::from_view_projection(view_projection);
        self.visible_chunks.clear();
        for (coord, chunk) in &self.chunk_buffers {
            if frustum.intersects_aabb(&chunk.aabb) {
                self.visible_chunks.push(*coord);
            }
        }
        self.visible_chunks.sort_unstable();
    }

    /// Record the cuboid draw into an already-begun render pass. Iterates the
    /// frustum-visible per-chunk buffers, one indexed draw per chunk over its own
    /// vertex/index buffer.
    ///
    /// `loaded_material` (part of #20): when an applied/loaded VS block is active,
    /// the caller passes the block's 6-layer D2Array bind group (`LoadedMaterial::
    /// bind_group`); the cuboid path then selects the loaded-block pipeline + shader,
    /// binding that D2Array at group(1) and selecting the per-face layer by the face
    /// normal — so the cuboid path shows the SAME texture the instanced path shows.
    /// `None` (no block applied) keeps the procedural-atlas path, unchanged.
    pub fn draw(
        &self,
        render_pass: &mut wgpu::RenderPass<'_>,
        loaded_material: Option<&wgpu::BindGroup>,
    ) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        // Debug-faces mode selects the cull-off pipeline (matching the uploaded
        // `debug_face_mode` flag) so back faces surviving a winding bug still draw
        // and get the shader's stripe marker — same as the instanced path. The
        // pipeline pair is the loaded-block pair when a block is applied (binds its
        // D2Array at group 1), else the procedural atlas pair.
        let (pipeline, material_bind_group) = match loaded_material {
            Some(loaded_bind_group) => (
                if self.debug_face_mode {
                    &self.loaded_debug_pipeline
                } else {
                    &self.loaded_pipeline
                },
                loaded_bind_group,
            ),
            None => (
                if self.debug_face_mode {
                    &self.debug_pipeline
                } else {
                    &self.pipeline
                },
                &self.atlas_bind_group,
            ),
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        // group(1) is either the procedural ATLAS (per-face `material_id` → atlas
        // sub-rect in the shader, one bind for a mixed-material chunk) or the loaded
        // block's D2Array (per-face layer selected by normal). One bind, one draw/chunk.
        render_pass.set_bind_group(1, material_bind_group, &[]);
        for coord in &self.visible_chunks {
            let Some(chunk) = self.chunk_buffers.get(coord) else {
                continue;
            };
            if chunk.index_count == 0 && chunk.index_count_overlay == 0 {
                continue;
            }
            render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
            render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            // ADR 0003 §3c: two draws per chunk — the overlay-OFF run (group(2) dynamic
            // offset 0 → overlay-active uniform = 0) then the overlay-ON run (dynamic
            // offset `overlay_dynamic_stride` → uniform = 1). The on-run is the second
            // half of the single index buffer (byte offset `index_count * 4`).
            if chunk.index_count > 0 {
                render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
                render_pass.draw_indexed(0..chunk.index_count, 0, 0..1);
            }
            if chunk.index_count_overlay > 0 {
                render_pass.set_bind_group(2, &self.overlay_bind_group, &[self.overlay_dynamic_stride]);
                let start = chunk.index_count;
                render_pass
                    .draw_indexed(start..start + chunk.index_count_overlay, 0, 0..1);
            }
        }
    }

    /// (ADR 0012 H1) Draw the onion GHOST pass: the two thin per-slab meshes flat-tinted
    /// translucent, alpha-blended over the solid with the depth test `Less` + depth WRITE ON
    /// (nearest ghost surface wins, builder-independent). MUST be called AFTER `draw`, inside
    /// the same MSAA pass (the solid's depth is what occludes the ghost). A no-op when onion is off (both slab
    /// maps empty). Group(1) binds the procedural atlas even for loaded-material scenes —
    /// the ghost shader flat-tints and never samples it (flat tint even over `cuboid_loaded`).
    /// Both slabs are drawn with the whole index buffer per chunk (overlay-off + overlay-on
    /// runs together): the ghost ignores the on-face grid overlay, so one draw suffices.
    pub fn draw_ghost(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.ghost_lower_buffers.is_empty() && self.ghost_upper_buffers.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.ghost_pipeline);
        render_pass.set_bind_group(0, &self.ghost_uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.atlas_bind_group, &[]);
        // Overlay disabled for the ghost (the shader flat-tints before any overlay); bind
        // the off-slot (0) so group(2) is satisfied.
        render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
        // Lower slab THEN upper slab — the same order the brick raymarch ghost draws its
        // two slabs, so any screen overlap of the two blends identically across paths.
        // Within a slab, iterate in SORTED coord order: the ghost writes no depth, so a
        // stable draw order keeps the alpha-blend result deterministic across runs AND
        // identical between the dense and two-layer builds (the two_layer golden gate).
        for buffers in [&self.ghost_lower_buffers, &self.ghost_upper_buffers] {
            let mut coords: Vec<[i32; 3]> = buffers.keys().copied().collect();
            coords.sort_unstable();
            for coord in coords {
                let Some(chunk) = buffers.get(&coord) else {
                    continue;
                };
                let total = chunk.index_count + chunk.index_count_overlay;
                if total == 0 {
                    continue;
                }
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..total, 0, 0..1);
            }
        }
    }
}

/// Upload built per-chunk meshes into GPU buffers, one [`CuboidChunkBuffers`] per
/// non-empty chunk (issue #20 S6c-2d).
pub(crate) fn upload_chunk_meshes(
    device: &wgpu::Device,
    chunk_meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], CuboidChunkBuffers> {
    let mut buffers = std::collections::HashMap::new();
    for mesh in chunk_meshes {
        if mesh.indices.is_empty() && mesh.indices_overlay.is_empty() {
            continue;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // One index buffer = overlay-OFF run then overlay-ON run (ADR 0003 §3c); the two
        // draws slice it by count + offset.
        let mut all_indices = mesh.indices.clone();
        all_indices.extend_from_slice(&mesh.indices_overlay);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk indices"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        buffers.insert(
            mesh.coord,
            CuboidChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                index_count_overlay: mesh.indices_overlay.len() as u32,
                aabb: mesh.aabb,
                box_count: mesh.box_count,
            },
        );
    }
    buffers
}

/// Bucket a whole [`VoxelGrid`] into per-chunk sub-grids keyed by integer chunk
/// coord `floor(world_position / chunk_extent)` (issue #20 S6c-2d) — the SAME key
/// [`crate::renderer::VoxelRenderer::rebuild_instances`] uses, so the cuboid `new`
/// wrapper's chunk partition matches the instanced one and the resolve cache's
/// per-chunk accessor. A sub-grid carries only the occupied voxels (its `dimensions`
/// is unused by the apron mesher, which keys off `world_position`).
pub(crate) fn bucket_grid_into_chunk_grids(
    grid: &VoxelGrid,
    voxels_per_block: u32,
) -> Vec<([i32; 3], VoxelGrid)> {
    use std::collections::HashMap;
    let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
    let mut buckets: HashMap<[i32; 3], VoxelGrid> = HashMap::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let key = [
            (position[0] / chunk_extent).floor() as i32,
            (position[1] / chunk_extent).floor() as i32,
            (position[2] / chunk_extent).floor() as i32,
        ];
        buckets
            .entry(key)
            .or_insert_with(|| VoxelGrid::new([0, 0, 0]))
            .occupied
            .push(*voxel);
    }
    let mut out: Vec<([i32; 3], VoxelGrid)> = buckets.into_iter().collect();
    out.sort_unstable_by_key(|(coord, _)| *coord);
    out
}
