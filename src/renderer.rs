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
    /// Face-orientation debug flag (0 = normal, 1 = colour-by-normal debug).
    /// Reuses the std140 scalar slot that pads the preceding vec3 to 16 bytes.
    debug_face_mode: f32,
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
            ],
        };

        // Both pipelines share everything except the cull mode. The debug
        // pipeline disables culling (`cull_mode: None`) so that if a back face is
        // the nearest surface to the camera (a winding bug), it draws and the
        // shader's `front_facing` marker flags it — culling would otherwise hide
        // the evidence. Depth testing stays on in both, so the nearest face wins.
        let build_pipeline = |label: &str, cull_mode: Option<wgpu::Face>| {
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
        let pipeline = build_pipeline("voxel pipeline", Some(wgpu::Face::Back));
        let debug_pipeline = build_pipeline("voxel debug pipeline", None);

        Self {
            pipeline,
            debug_pipeline,
            cube_vertex_buffer,
            cube_index_buffer,
            cube_index_count: indices.len() as u32,
            instance_buffer,
            instance_count,
            instance_capacity,
            uniform_buffer,
            uniform_bind_group,
            material_bind_groups,
            material_bind_group_layout,
            material_sampler,
        }
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
    /// `debug_face_mode` enables the face-orientation debug shader path (colour by
    /// outward normal + back-facing marker); it must match the pipeline chosen in
    /// [`VoxelRenderer::draw`].
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
        grid_overlay_enabled: bool,
        debug_face_mode: bool,
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
            debug_face_mode: if debug_face_mode { 1.0 } else { 0.0 },
            voxel_line_half_width: VOXEL_LINE_HALF_WIDTH,
            block_line_half_width: BLOCK_LINE_HALF_WIDTH,
            voxel_line_alpha: VOXEL_LINE_ALPHA,
            block_line_alpha: BLOCK_LINE_ALPHA,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
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
        if self.instance_count == 0 {
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
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.cube_index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.cube_index_count, 0, 0..self.instance_count);
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

    /// Draw the cube into a scissored top-left corner of `target_view` (its own
    /// render pass, with a freshly-cleared private depth texture). The colour
    /// attachment loads the already-resolved scene so only the corner is touched.
    pub fn draw(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
    ) {
        let margin = VIEW_CUBE_VIEWPORT_MARGIN;
        let size = VIEW_CUBE_VIEWPORT_PIXELS;
        // Bail if the target is too small to host the corner viewport.
        if target_width < margin + size || target_height < margin + size {
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

        pass.set_viewport(margin as f32, margin as f32, size as f32, size as f32, 0.0, 1.0);
        pass.set_scissor_rect(margin, margin, size, size);

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
// Origin gizmo (Milestone 5) — ARCHITECTURE.md §5.
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

/// The origin gizmo: three coloured axis lines and three perpendicular square
/// line-loops, drawn with **depth-test disabled** so it shows through a solid
/// model (ARCHITECTURE.md §5). Drawn in the MSAA pass, after the voxels.
pub struct GizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl GizmoRenderer {
    /// Create the gizmo renderer for a colour target format. `grid_dimensions`
    /// sizes the gizmo (`L = max(dims) * 0.62`).
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

    /// Upload the camera matrix (same `view_projection` as the voxel pass).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
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
/// Fine floor grid colour `#6b5f4a` (dim warm) at ~0.16 alpha.
const FLOOR_COLOR_HEX: u32 = 0x6b_5f_4a;
const FLOOR_ALPHA: f32 = 0.16;

/// The block lattice and fine floor grid (ARCHITECTURE.md §6 / prototype
/// `buildGrids`), drawn through the shared alpha-blended, depth-tested line
/// pipeline in the MSAA pass. The lattice is a 3D box lattice with lines at every
/// BLOCK boundary (spacing = `voxels_per_block`); the floor is a flat grid on the
/// bottom plane. Each is toggled independently by the caller.
pub struct GridLatticeRenderer {
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

impl GridLatticeRenderer {
    /// Create the renderer for a colour target, sized to the current grid.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
    ) -> Self {
        let lattice = lattice_vertices(grid_dimensions, voxels_per_block);
        let floor = floor_vertices(grid_dimensions);
        let lattice_vertex_count = lattice.len() as u32;
        let floor_vertex_count = floor.len() as u32;
        let lattice_capacity = lattice_vertex_count.max(1);
        let floor_capacity = floor_vertex_count.max(1);

        let lattice_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lattice line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(lattice, lattice_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let floor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("floor line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(floor, floor_capacity)),
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
            lattice_vertex_count,
            lattice_capacity,
            floor_buffer,
            floor_vertex_count,
            floor_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Rebuild both line sets for a freshly-resolved grid (matches the voxel
    /// rebuild; `buildGrids` keyed on the same dims/density).
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_dimensions: [u32; 3],
        voxels_per_block: u32,
    ) {
        let lattice = lattice_vertices(grid_dimensions, voxels_per_block);
        let floor = floor_vertices(grid_dimensions);

        self.lattice_vertex_count =
            upload_lines(device, queue, &mut self.lattice_buffer, &mut self.lattice_capacity, lattice, "lattice line vertices");
        self.floor_vertex_count =
            upload_lines(device, queue, &mut self.floor_buffer, &mut self.floor_capacity, floor, "floor line vertices");
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the lattice and/or floor draws into an already-begun (MSAA) pass.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>, show_lattice: bool, show_floor: bool) {
        if !show_lattice && !show_floor {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        if show_lattice && self.lattice_vertex_count > 0 {
            render_pass.set_vertex_buffer(0, self.lattice_buffer.slice(..));
            render_pass.draw(0..self.lattice_vertex_count, 0..1);
        }
        if show_floor && self.floor_vertex_count > 0 {
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

/// Block lattice line segments: a 3D box lattice with grid lines at every BLOCK
/// boundary (spacing = `voxels_per_block`). Port of the prototype `buildGrids`
/// lattice loop. World-centred so it aligns with the voxel grid.
fn lattice_vertices(grid_dimensions: [u32; 3], voxels_per_block: u32) -> Vec<LineVertex> {
    let [grid_x, grid_y, grid_z] = grid_dimensions;
    let half_x = grid_x as f32 / 2.0;
    let half_y = grid_y as f32 / 2.0;
    let half_z = grid_z as f32 / 2.0;
    let step = voxels_per_block.max(1);
    let color = with_alpha(srgb_hex_to_linear(LATTICE_COLOR_HEX), LATTICE_ALPHA);

    // Block-boundary planes along each axis (0, D, 2D, …, N), world-centred.
    let boundaries = |extent: u32, half: f32| -> Vec<f32> {
        let mut values = Vec::new();
        let mut g = 0u32;
        while g <= extent {
            values.push(g as f32 - half);
            g += step;
        }
        values
    };
    let xs = boundaries(grid_x, half_x);
    let ys = boundaries(grid_y, half_y);
    let zs = boundaries(grid_z, half_z);

    let mut vertices = Vec::new();
    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Lines along Y at every (x, z) lattice node.
    for &x in &xs {
        for &z in &zs {
            add([x, -half_y, z], [x, half_y, z]);
        }
    }
    // Lines along X at every (y, z) lattice node.
    for &y in &ys {
        for &z in &zs {
            add([-half_x, y, z], [half_x, y, z]);
        }
    }
    // Lines along Z at every (x, y) lattice node.
    for &x in &xs {
        for &y in &ys {
            add([x, y, -half_z], [x, y, half_z]);
        }
    }
    vertices
}

/// Fine floor grid: a flat grid on the bottom plane (`y = -grid_y/2`) at 1-VOXEL
/// spacing (prototype `buildGrids` floor loop steps `g += 1`; "fine" = per-voxel).
fn floor_vertices(grid_dimensions: [u32; 3]) -> Vec<LineVertex> {
    let [grid_x, grid_y, grid_z] = grid_dimensions;
    let half_x = grid_x as f32 / 2.0;
    let half_z = grid_z as f32 / 2.0;
    let y = -(grid_y as f32 / 2.0);
    let color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_ALPHA);

    let mut vertices = Vec::new();
    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Lines parallel to Z, stepping along X.
    for g in 0..=grid_x {
        let c = g as f32 - half_x;
        add([c, y, -half_z], [c, y, half_z]);
    }
    // Lines parallel to X, stepping along Z.
    for g in 0..=grid_z {
        let c = g as f32 - half_z;
        add([-half_x, y, c], [half_x, y, c]);
    }
    vertices
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

    #[test]
    fn view_cube_is_ccw_outward() {
        let (vertices, indices) = view_cube_geometry();
        let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
        let normals: Vec<[f32; 3]> = vertices.iter().map(|v| v.normal).collect();
        assert_ccw_outward(&positions, &normals, &indices);
    }
}
