//! Analytic infinite reference grid (issue #29 Points fast-follow) — replaces the
//! finite tiled-line ground plane with a fullscreen ray-plane shader.

use super::*;

/// std140 uniform for one analytic-grid plane; field order matches `GridUniforms`
/// in `infinite_grid.wgsl` exactly. One instance per visible Point × enabled plane.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct InfiniteGridUniforms {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    /// Camera eye (recentred frame); `.w` unused.
    eye: [f32; 4],
    /// Plane origin (the Point's recentred position); `.w` unused.
    plane_origin: [f32; 4],
    /// In-plane unit axes spanning the plane, and the plane normal (`.w` unused).
    u_axis: [f32; 4],
    v_axis: [f32; 4],
    normal_axis: [f32; 4],
    /// Line colour (linear RGB); `.w` = voxel spacing (1.0).
    line_color: [f32; 4],
    /// `[block_spacing(=density), minor_alpha, major_alpha, reserved]`. The shader
    /// reads only `.x/.y/.z`; `.w` is a reserved padding slot (the old fixed
    /// world-distance fade was removed — fading is now per-tier LOD in the shader).
    /// Kept as `vec4` for the std140 16-byte uniform alignment.
    params: [f32; 4],
}

/// Maximum number of analytic-grid planes drawn in one frame (3 planes × a handful
/// of Points). Bounds the dynamic-offset uniform buffer; extra planes are dropped.
const MAX_GRID_PLANES: usize = 32;

/// The analytic infinite reference grid (issue #29 Points fast-follow): for each
/// visible [`Point`]'s enabled plane it draws a fullscreen triangle whose fragment
/// shader intersects the per-pixel view ray with that plane, computes a two-tier
/// (voxel + block) anti-aliased grid via screen-space derivatives, fades with
/// distance, and writes `@builtin(frag_depth)` so opaque voxels (drawn earlier in
/// the SAME MSAA pass) occlude it. This replaces the old finite tiled LINE quad,
/// whose hard edge / near-clip cutoff looked bad at shallow angles.
///
/// One dynamic-offset uniform buffer holds all planes' uniforms; [`Self::draw`]
/// binds each plane's slice and issues one 3-vertex draw. With no enabled plane the
/// draw is a no-op.
pub struct InfiniteGridRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Aligned stride (bytes) between consecutive plane uniforms in the buffer.
    aligned_stride: u32,
    /// Number of planes uploaded this frame (≤ [`MAX_GRID_PLANES`]).
    plane_count: u32,
}

impl InfiniteGridRenderer {
    /// Create the analytic-grid renderer for a colour target. The plane batch starts
    /// empty — the caller fills it each frame via [`Self::rebuild_from_scene`].
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("infinite grid shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/infinite_grid.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("infinite grid bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<InfiniteGridUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        // Each plane's uniform must start at a `min_uniform_buffer_offset_alignment`
        // boundary for the dynamic offset; pad the stride up to it.
        let uniform_size = std::mem::size_of::<InfiniteGridUniforms>() as u32;
        let alignment = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let aligned_stride = uniform_size.div_ceil(alignment) * alignment;
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infinite grid uniforms"),
            size: (aligned_stride as u64) * MAX_GRID_PLANES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("infinite grid bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(uniform_size as u64),
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("infinite grid pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("infinite grid pipeline"),
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
            // Drawn INSIDE the MSAA pass: depth-tested LessEqual against the voxels'
            // depth (written via `frag_depth`) so opaque objects occlude the grid.
            // Depth WRITE is off so the (alpha-blended, transparent) grid never
            // occludes a later transparent draw or itself.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
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

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            aligned_stride,
            plane_count: 0,
        }
    }

    /// Rebuild this frame's analytic-grid planes by walking `scene.points` (issue #29
    /// Points fast-follow), uploading one plane uniform per visible Point × enabled
    /// plane. `view_projection` and its inverse + `camera_eye` are all in the
    /// recentred render frame the voxels live in. With no enabled plane this uploads
    /// nothing and [`Self::draw`] becomes a no-op.
    pub fn rebuild_from_scene(
        &mut self,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
        view_projection: glam::Mat4,
        camera_eye: [f32; 3],
    ) {
        let planes = enabled_grid_planes(scene, voxels_per_block);
        let density = voxels_per_block.max(1) as f32;
        let inverse_view_projection = view_projection.inverse();
        let line_color = srgb_hex_to_linear(POINT_PLANE_COLOR_HEX);

        let count = planes.len().min(MAX_GRID_PLANES);
        for (index, plane) in planes.iter().take(count).enumerate() {
            let uniforms = InfiniteGridUniforms {
                view_projection: view_projection.to_cols_array_2d(),
                inverse_view_projection: inverse_view_projection.to_cols_array_2d(),
                eye: [camera_eye[0], camera_eye[1], camera_eye[2], 0.0],
                plane_origin: [plane.origin[0], plane.origin[1], plane.origin[2], 0.0],
                u_axis: [plane.u_axis[0], plane.u_axis[1], plane.u_axis[2], 0.0],
                v_axis: [plane.v_axis[0], plane.v_axis[1], plane.v_axis[2], 0.0],
                normal_axis: [plane.normal[0], plane.normal[1], plane.normal[2], 0.0],
                line_color: [line_color[0], line_color[1], line_color[2], 1.0],
                // `.w` is a reserved padding slot (the shader reads only x/y/z); the
                // old world-distance fade was removed in favour of per-tier LOD fade.
                params: [
                    density,
                    POINT_PLANE_MINOR_ALPHA,
                    POINT_PLANE_MAJOR_ALPHA,
                    0.0,
                ],
            };
            let offset = (index as u32 * self.aligned_stride) as u64;
            queue.write_buffer(&self.uniform_buffer, offset, bytemuck::bytes_of(&uniforms));
        }
        self.plane_count = count as u32;
    }

    /// Record the analytic-grid draws into an already-begun (MSAA) pass: one
    /// fullscreen triangle per plane, each binding its dynamic-offset uniform slice.
    /// Self-gating: no enabled plane → nothing drawn.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.plane_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        for index in 0..self.plane_count {
            let offset = index * self.aligned_stride;
            render_pass.set_bind_group(0, &self.bind_group, &[offset]);
            render_pass.draw(0..3, 0..1);
        }
    }
}
