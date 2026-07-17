//! Viewport background gradient (issue #91, item 1; `docs/design/viewport-chrome-signal.md`).
//!
//! A dependency-free fullscreen pass that paints the Signal "field" — a cool near-black
//! radial gradient biased above-left of centre — as the viewport's background. It draws
//! FIRST in the shared 3D MSAA pass (before the voxels, depth-test off), so both display
//! paths (cuboid mesh + brick raymarch) and the headless `shot` composite the scene over
//! an identical background. See `shaders/background_gradient.wgsl` for the stops + the
//! sRGB-correct evaluation.

use super::*;

/// The fullscreen radial-gradient background. Holds only its pipeline (the gradient is
/// entirely analytic in the shader — no uniforms, no vertex buffer, no bind groups).
pub struct BackgroundGradientRenderer {
    pipeline: wgpu::RenderPipeline,
}

impl BackgroundGradientRenderer {
    /// Build the background-gradient renderer for a colour target format. The pipeline is
    /// 4× MSAA (it draws inside the shared 3D MSAA pass) and does not touch depth.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("background gradient shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/background_gradient.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("background gradient pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("background gradient pipeline"),
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
                    // Opaque background — it replaces the cleared field; no blending.
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
            // A depth attachment is present in the pass, so the pipeline must declare a
            // matching depth state — but the background writes no depth and always passes,
            // so it never occludes the voxels drawn over it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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
        Self { pipeline }
    }

    /// Paint the gradient across the current viewport/scissor rect (a fullscreen
    /// triangle). Call FIRST in the 3D MSAA pass, before the voxel draw.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..3, 0..1);
    }
}
