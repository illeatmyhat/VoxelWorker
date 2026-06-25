//! The instanced voxel renderer (Milestone 2).
//!
//! Owns the GPU resources that turn a resolved [`VoxelGrid`](crate::voxel::VoxelGrid)
//! into flat-shaded instanced cubes: one shared unit-cube vertex/index buffer
//! (24 verts / 36 indices, per-face normals), an instance buffer built FROM the
//! grid, the camera uniform, and the render pipeline.
//!
//! It is render-target-agnostic: [`VoxelRenderer::draw`] records into a render
//! pass the caller has already begun against any colour view + depth view, so the
//! window and the headless capture paint identically.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::voxel::VoxelGrid;

/// Depth format used by the voxel pass and the depth texture.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Stability cap on the number of cube instances actually uploaded
/// (ARCHITECTURE.md §7). The CPU may resolve more occupied voxels than this; we
/// only ever draw the first `MAX_DRAWN_INSTANCES` so dragging a sphere to a huge
/// size/density can't blow up GPU memory or stall the draw. The separate 6M
/// voxel cap in `voxel.rs` usually fires first; this is the belt-and-braces
/// limit on the render side.
pub const MAX_DRAWN_INSTANCES: usize = 450_000;

/// One cube vertex: position on the unit cube plus its face normal.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Per-voxel instance data (24-byte stride).
///
/// `block_local_coord` is unused by the M2 shader but populated now so M4 only
/// adds shader logic, not wiring.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VoxelInstance {
    pub world_position: [f32; 3],
    pub block_local_coord: [f32; 3],
}

/// The camera matrix uploaded to the shader uniform.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CameraUniform {
    view_projection: [[f32; 4]; 4],
}

/// Build the 24 vertices / 36 indices of a unit cube spanning `[-1, 1]` per axis
/// with one outward normal per face (so faces shade independently). The shader
/// scales by 0.5, giving a unit cube centred on each voxel.
fn unit_cube_geometry() -> (Vec<CubeVertex>, Vec<u16>) {
    // (normal, the four corner offsets in the plane of that face).
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X
        ([1.0, 0.0, 0.0], [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0]]),
        // -X
        ([-1.0, 0.0, 0.0], [[-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]]),
        // +Y
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        // -Y
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]),
        // +Z
        ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        // -Z
        ([0.0, 0.0, -1.0], [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u16;
        for corner in corners {
            vertices.push(CubeVertex {
                position: corner,
                normal,
            });
        }
        // Two CCW triangles (counter-clockwise wound so the default front-face /
        // back-face culling keeps outward faces).
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// All GPU resources for drawing the voxel grid as instanced cubes.
pub struct VoxelRenderer {
    pipeline: wgpu::RenderPipeline,
    cube_vertex_buffer: wgpu::Buffer,
    cube_index_buffer: wgpu::Buffer,
    cube_index_count: u32,
    instance_buffer: wgpu::Buffer,
    instance_count: u32,
    /// Number of instances the current `instance_buffer` can hold without a
    /// reallocation. `rebuild_instances` grows the buffer only when exceeded.
    instance_capacity: u32,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
}

impl VoxelRenderer {
    /// Create the renderer for a given colour target format. The instance buffer
    /// is built from `grid` immediately.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid: &VoxelGrid,
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

        let instances = instances_from_grid(grid);
        let instance_count = instances.len() as u32;
        let instance_capacity = instance_count.max(1);
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel instances"),
            // Always allocate room for at least one instance so an initially empty
            // grid still has a valid (zero-drawn) buffer to grow from.
            contents: bytemuck::cast_slice(&pad_to_capacity(instances, instance_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel camera uniform"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel camera bind group layout"),
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
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/voxel.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel pipeline layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout)],
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
            ],
        };
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<VoxelInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as u64,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: &[vertex_layout, instance_layout],
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
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            cube_index_count: indices.len() as u32,
            instance_buffer,
            instance_count,
            instance_capacity,
            camera_buffer,
            camera_bind_group,
        }
    }

    /// Number of voxel instances currently drawn from the buffer.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Rebuild the instance buffer FROM a freshly-resolved grid (M3 live edit).
    ///
    /// Reuses the existing `COPY_DST` buffer with `queue.write_buffer` when the
    /// new instance count fits the current capacity; otherwise reallocates a
    /// larger buffer. The instance count is capped at [`MAX_DRAWN_INSTANCES`]
    /// (ARCHITECTURE.md §7) so an enormous grid can't stall the draw.
    pub fn rebuild_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid: &VoxelGrid) {
        let instances = instances_from_grid(grid);
        let instance_count = instances.len() as u32;

        if instance_count <= self.instance_capacity {
            if instance_count > 0 {
                queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
            }
        } else {
            // Grow: allocate exactly the new count. A `create_buffer_init` keeps
            // the COPY_DST usage so subsequent rebuilds can reuse it again.
            self.instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxel instances"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.instance_capacity = instance_count;
        }
        self.instance_count = instance_count;
    }

    /// Upload a new `view_projection` matrix to the camera uniform.
    pub fn update_camera(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniform = CameraUniform {
            view_projection: view_projection.to_cols_array_2d(),
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }

    /// Record the voxel draw into an already-begun render pass.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.instance_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.cube_index_count, 0, 0..self.instance_count);
    }
}

/// Build the instance list FROM the resolved grid (REPRESENTATION.md seam: the
/// instance buffer is built from the [`VoxelGrid`], not from the sampling loop).
///
/// Capped at [`MAX_DRAWN_INSTANCES`] (ARCHITECTURE.md §7): if the grid resolved
/// more occupied voxels than the cap, only the first `MAX_DRAWN_INSTANCES` are
/// uploaded.
fn instances_from_grid(grid: &VoxelGrid) -> Vec<VoxelInstance> {
    grid.occupied
        .iter()
        .take(MAX_DRAWN_INSTANCES)
        .map(|voxel| VoxelInstance {
            world_position: voxel.world_position,
            block_local_coord: [
                voxel.block_local_coord[0] as f32,
                voxel.block_local_coord[1] as f32,
                voxel.block_local_coord[2] as f32,
            ],
        })
        .collect()
}

/// Grow `instances` to at least `capacity` entries with zeroed padding so the
/// initial buffer allocation reserves room (degenerate zero-size cubes at the
/// origin are never drawn because `instance_count` < capacity).
fn pad_to_capacity(mut instances: Vec<VoxelInstance>, capacity: u32) -> Vec<VoxelInstance> {
    if (instances.len() as u32) < capacity {
        instances.resize(
            capacity as usize,
            VoxelInstance {
                world_position: [0.0; 3],
                block_local_coord: [0.0; 3],
            },
        );
    }
    instances
}

/// Create a depth texture view sized to a render target. Recreated on window
/// resize and created at the offscreen size for the headless capture.
pub fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel depth texture"),
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
