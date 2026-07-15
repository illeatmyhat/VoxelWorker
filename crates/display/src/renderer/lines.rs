//! Shared line-drawing infrastructure (the `LineList` pipeline + its vertex/uniform
//! formats) used by the gizmo, the view-cube edge wireframe, the lattice/floor scene
//! grid, and the Points reference axes.

use super::*;

/// Grid overlay tuning, transcribed from the prototype `GRID` uniforms
/// (chisel-bench-reference.html). Half-widths are in voxel units (the overlay is
/// computed from absolute voxel position), alphas are blend strengths, and the
/// colours are the sRGB hex line colours (ARCHITECTURE.md §8).
pub(crate) const VOXEL_LINE_HALF_WIDTH: f32 = 0.05;
pub(crate) const BLOCK_LINE_HALF_WIDTH: f32 = 0.11;
pub(crate) const VOXEL_LINE_ALPHA: f32 = 0.40;
pub(crate) const BLOCK_LINE_ALPHA: f32 = 0.92;
/// Voxel grid line colour `#17120b` (sRGB hex → linear).
pub(crate) const VOXEL_LINE_COLOR_HEX: u32 = 0x17_12_0b;
/// Block grid line colour `#080605` (sRGB hex → linear, darker/bolder).
pub(crate) const BLOCK_LINE_COLOR_HEX: u32 = 0x08_06_05;

/// One coloured line-segment vertex (position + linear RGBA colour). The alpha
/// lets the M8 block lattice / floor grid draw at low opacity through the same
/// alpha-blending line pipeline the gizmo / view-cube edges use (those pass 1.0).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineVertex {
    pub(crate) position: [f32; 3],
    pub(crate) color: [f32; 4],
}

/// Camera uniform for the line passes (gizmo + view-cube edges + lattice/floor +
/// Points): the view-projection matrix plus a small NDC `depth_bias` (issue #29
/// floor fix). The bias is zero for every pass except the floor grid, which uses a
/// negative value to win the depth test against the model's coincident bottom face
/// without a geometric drop (wgpu forbids a hardware depth bias on `LineList`).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct LineUniforms {
    pub(crate) view_projection: [[f32; 4]; 4],
    /// `[bias, 0, 0, 0]` — only `.x` is read; the rest pad to 16-byte alignment.
    pub(crate) depth_bias: [f32; 4],
}

/// Pad a line-vertex list to `capacity` with zeroed (degenerate) vertices.
pub(crate) fn pad_lines(mut vertices: Vec<LineVertex>, capacity: u32) -> Vec<LineVertex> {
    if (vertices.len() as u32) < capacity {
        vertices.resize(
            capacity as usize,
            LineVertex { position: [0.0; 3], color: [0.0; 4] },
        );
    }
    vertices
}

/// Build the shared uniform bind group (binding 0 = `LineUniforms`) for a line pass.
pub(crate) fn line_uniform_bind_group(
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
/// (depth-test off so it shows through solids). Depth bias is applied in the SHADER
/// (via [`LineUniforms::depth_bias`]) rather than the pipeline, because wgpu rejects
/// a hardware `DepthBiasState` on `LineList` topology — the floor grid uses this to
/// win coincident depth against the model's base face without a geometric drop.
pub(crate) fn build_line_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    uniform_bind_group_layout: &wgpu::BindGroupLayout,
    label: &str,
    depth_tested: bool,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("line shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/line.wgsl").into()),
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

/// Write a line-vertex list to `buffer`, growing it if needed; returns the count.
pub(crate) fn upload_lines(
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
