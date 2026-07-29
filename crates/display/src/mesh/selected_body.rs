//! Selection-cel renderer (ADR 0032) — the selected nodes' derived bodies, cel-shaded.
//!
//! Viewport selection feedback in EVERY view mode: each selected node's own standalone
//! body (derived by the same bounded per-node slice machinery as the boolean-operand
//! ghost — `AppCore::selected_body_cel`) is meshed through the shared two-layer cuboid
//! mesher and drawn ONCE over the solid with the `cuboid.wgsl` cel branch
//! (`ghost_mode = 2`): the Signal accent in quantised Lambert bands, a hard near-opaque
//! band at the screen-space silhouette (outline-emphasis).
//!
//! Depth-tested `LessEqual`, depth write OFF — feedback shades only the surface the
//! composed model actually shows (the owner ruling: cel over the geometry, NOT a
//! translucent x-ray; the buried remainder of a cutter stays invisible here — the
//! Show-booleans operand ghost is the x-ray voice). The same toward-the-viewer depth
//! bias as the operand ghost keeps a body face COINCIDENT with the scene surface (the
//! usual case — a selected node's visible surface IS the scene's) from being dropped to
//! a depth-tie ULP.
//!
//! Mesh rebuilt only on selection/geometry change (`rebuild`); `update_uniforms` per
//! frame writes camera + tint only.

use super::selected_operand::build_unsampled_atlas_bind_group;
use super::*;
use crate::renderer::selection_cel_tint;

/// One selected node's body: its covering chunks in the composed scene's absolute chunk
/// coords (ADR 0008 — frame carried in, never re-derived).
pub type SelectedBodyChunks = Vec<([i32; 3], Arc<TwoLayerChunk>)>;

/// Toward-the-viewer depth bias — the operand ghost's exact constants (see
/// `selected_operand.rs`): a body face coincident with the scene surface must pass
/// `LessEqual` robustly even when another mesher's triangulation of the same plane
/// interpolates depth a ULP apart.
const SELECTION_CEL_DEPTH_BIAS_CONSTANT: i32 = -2;
const SELECTION_CEL_DEPTH_BIAS_SLOPE_SCALE: f32 = -2.0;

/// GPU resources for the selection-cel overlay. Owned by the shell beside the other
/// overlay renderers; self-gating (`draw` is a no-op with no bodies).
pub struct SelectedBodyCelRenderer {
    /// The cel pipeline: `cuboid.wgsl` cel branch, alpha-blended, depth test `LessEqual`,
    /// depth write OFF.
    pipeline: wgpu::RenderPipeline,
    /// The ONE cel uniform buffer — every body shares the accent tint, so one bind
    /// group serves all chunk draws.
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Group(1): the never-sampled 1×1 atlas placeholder (the cel branch returns its
    /// banded tint before any sample).
    unsampled_atlas_bind_group: wgpu::BindGroup,
    /// Group(2): the per-draw overlay-active uniform at the OFF slot — the cel ignores
    /// the on-face grid.
    overlay_bind_group: wgpu::BindGroup,
    /// The uploaded chunk buffers of every selected body (empty = nothing selected /
    /// the selection has no body), sorted by coord within each body for a
    /// deterministic draw order.
    chunk_buffers: Vec<CuboidChunkBuffers>,
    /// The composed scene's voxel dims + density the meshes were built against, echoed
    /// into the per-frame uniforms (the vertex stage's corner-anchoring scalars).
    grid_dimensions: [u32; 3],
    voxels_per_block: u32,
}

impl SelectedBodyCelRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_bind_group_layout = cuboid_uniform_bind_group_layout(device);
        let atlas_bind_group_layout = build_atlas_bind_group_layout(device);
        let unsampled_atlas_bind_group =
            build_unsampled_atlas_bind_group(device, queue, &atlas_bind_group_layout);
        let (overlay_bind_group, _overlay_stride) =
            build_overlay_bind_group(device, &overlay_bind_group_layout(device));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection-cel shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shaders::with_shared_shading(include_str!("../shaders/cuboid.wgsl")).into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection-cel pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&atlas_bind_group_layout),
                Some(&overlay_bind_group_layout(device)),
            ],
            immediate_size: 0,
        });

        // The cuboid vertex layout, matching `build_two_layer_chunk_meshes` output —
        // same as the solid + operand-ghost pipelines.
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
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection-cel pipeline"),
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
                // No depth write: the cel occludes nothing; the visible/occluded split
                // comes purely from the SOLID's depth already in the attachment.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: SELECTION_CEL_DEPTH_BIAS_CONSTANT,
                    slope_scale: SELECTION_CEL_DEPTH_BIAS_SLOPE_SCALE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selection-cel uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("selection-cel uniforms"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            uniform_buffer,
            uniform_bind_group,
            unsampled_atlas_bind_group,
            overlay_bind_group,
            chunk_buffers: Vec::new(),
            grid_dimensions: [0; 3],
            voxels_per_block: 1,
        }
    }

    /// Drop every body (the selection cleared / resolves to no geometry).
    pub fn clear(&mut self) {
        self.chunk_buffers.clear();
    }

    /// (Re)build the cel meshes for a fresh selection derivation. Called ONLY on
    /// selection/geometry change, never per frame. Each body is meshed by the SAME
    /// two-layer cuboid mesher the solid path uses, at the FULL band, against the
    /// COMPOSED scene's `recentre` — so the cel lands voxel-exact on the selected
    /// node's place in the render frame (ADR 0008).
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        bodies: &[SelectedBodyChunks],
        grid_dimensions: [u32; 3],
        recentre: RecentreVoxels,
        voxels_per_block: u32,
    ) {
        self.chunk_buffers.clear();
        self.grid_dimensions = grid_dimensions;
        self.voxels_per_block = voxels_per_block.max(1);
        for chunks in bodies {
            let meshes = build_two_layer_chunk_meshes(
                chunks,
                grid_dimensions,
                recentre,
                voxels_per_block,
                LayerBand::FULL,
                None,
            );
            let mut buffers_by_coord = upload_chunk_meshes(device, &meshes);
            // Sorted coord order per body: no depth write, so a stable draw order keeps
            // the alpha-blend result deterministic across runs.
            let mut coords: Vec<[i32; 3]> = buffers_by_coord.keys().copied().collect();
            coords.sort_unstable();
            self.chunk_buffers.extend(
                coords
                    .into_iter()
                    .filter_map(|coord| buffers_by_coord.remove(&coord)),
            );
        }
    }

    /// Upload the per-frame camera + tint. `camera_position` (the eye, render frame)
    /// drives the shader's silhouette facing term.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        camera_position: glam::Vec3,
    ) {
        let uniforms = cel_selection_uniforms(
            view_projection,
            self.grid_dimensions,
            self.voxels_per_block,
            selection_cel_tint(),
            camera_position.to_array(),
        );
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the cel pass into an already-begun MSAA pass. MUST run AFTER the frame's
    /// solid voxel draw (mesh or brick — both leave their depth in the shared
    /// attachment, which is what confines the cel to the visible surface). A no-op with
    /// no bodies.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.unsampled_atlas_bind_group, &[]);
        render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
        for chunk in &self.chunk_buffers {
            chunk.draw_all_runs(render_pass);
        }
    }
}
