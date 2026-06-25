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

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::panel::MaterialChoice;
use crate::voxel::VoxelGrid;

/// Depth format used by the voxel pass and the depth texture.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Sample count for the 3D voxel pass (4× MSAA). The depth texture, the
/// multisampled colour texture and the pipeline all share this count; egui still
/// renders at 1 sample onto the resolved target.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Edge length of every procedural material texture (square, no mipmaps).
const MATERIAL_TEXTURE_SIZE: u32 = 32;

/// Stability cap on the number of cube instances actually uploaded
/// (ARCHITECTURE.md §7). The CPU may resolve more occupied voxels than this; we
/// only ever draw the first `MAX_DRAWN_INSTANCES` so dragging a sphere to a huge
/// size/density can't blow up GPU memory or stall the draw. The separate 6M
/// voxel cap in `voxel.rs` usually fires first; this is the belt-and-braces
/// limit on the render side.
pub const MAX_DRAWN_INSTANCES: usize = 450_000;

/// One cube vertex: position on the unit cube, its face normal, and the base
/// (0..1) UV for that face.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    face_uv: [f32; 2],
}

/// Per-voxel instance data (24-byte stride).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct VoxelInstance {
    pub world_position: [f32; 3],
    pub block_local_coord: [f32; 3],
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
    _pad: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
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
        for (corner_index, corner) in corners.iter().enumerate() {
            vertices.push(CubeVertex {
                position: *corner,
                normal,
                face_uv: FACE_UVS[corner_index],
            });
        }
        // Two CCW triangles (counter-clockwise wound so the default front-face /
        // back-face culling keeps outward faces).
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

/// All GPU resources for drawing the voxel grid as textured instanced cubes.
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
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// One bind group per material (Stone/Wood/Plain), indexed by
    /// [`MaterialChoice`] order.
    material_bind_groups: [wgpu::BindGroup; 3],
}

impl VoxelRenderer {
    /// Create the renderer for a given colour target format. The instance buffer
    /// is built from `grid` immediately.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("voxel material bind group layout"),
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
            });
        let material_bind_groups = [
            generate_stone_texture(),
            generate_wood_texture(),
            generate_plain_texture(),
        ]
        .iter()
        .map(|pixels| {
            let texture = upload_material_texture(device, queue, pixels);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
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
            cube_vertex_buffer,
            cube_index_buffer,
            cube_index_count: indices.len() as u32,
            instance_buffer,
            instance_count,
            instance_capacity,
            uniform_buffer,
            uniform_bind_group,
            material_bind_groups,
        }
    }

    /// Number of voxel instances currently drawn from the buffer.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }

    /// Rebuild the instance buffer FROM a freshly-resolved grid (M3 live edit).
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

    /// Upload the per-frame uniforms: the camera matrix, the grid half-extent and
    /// density (for the per-voxel slice + overlay), and the grid-overlay toggle.
    ///
    /// `grid_dimensions` are the voxel-space dims of the current grid; the half
    /// extent is `dimensions / 2` so a fragment's `world_pos + half_extent` makes
    /// voxel boundaries fall on integers (BUG 2 fix). `voxels_per_block` is the
    /// current density. `grid_overlay_enabled` reflects the Display toggle.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
        grid_overlay_enabled: bool,
    ) {
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
            _pad: 0.0,
            voxel_line_half_width: VOXEL_LINE_HALF_WIDTH,
            block_line_half_width: BLOCK_LINE_HALF_WIDTH,
            voxel_line_alpha: VOXEL_LINE_ALPHA,
            block_line_alpha: BLOCK_LINE_ALPHA,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the voxel draw into an already-begun render pass, binding the
    /// texture for `material`.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>, material: MaterialChoice) {
        if self.instance_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_bind_group(1, &self.material_bind_groups[material_index(material)], &[]);
        render_pass.set_vertex_buffer(0, self.cube_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.cube_index_count, 0, 0..self.instance_count);
    }
}

/// Index a [`MaterialChoice`] into the `material_bind_groups` array.
fn material_index(material: MaterialChoice) -> usize {
    match material {
        MaterialChoice::Stone => 0,
        MaterialChoice::Wood => 1,
        MaterialChoice::Plain => 2,
    }
}

/// Upload a 32×32 RGBA8 sRGB texture (nearest filter, clamp-to-edge, no mipmaps).
fn upload_material_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pixels: &[u8],
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: MATERIAL_TEXTURE_SIZE,
        height: MATERIAL_TEXTURE_SIZE,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("voxel material texture"),
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
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * MATERIAL_TEXTURE_SIZE),
            rows_per_image: Some(MATERIAL_TEXTURE_SIZE),
        },
        size,
    );
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

/// Build the instance list FROM the resolved grid (REPRESENTATION.md seam).
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

/// Grow `instances` to at least `capacity` entries with zeroed padding.
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
