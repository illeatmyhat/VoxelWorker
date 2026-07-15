//! The palette PREVIEW thumbnail renderer — a shell-side GPU sink, NOT a scene view.
//!
//! [`ThumbnailRenderer`] is a tiny offscreen pipeline that draws a textured unit
//! cube at a fixed 45° orthographic view (prototype `thumbCam`: azimuth π/4,
//! elevation 0.62) into a ~96×96 `Rgba8Unorm` texture. The rendered texture is
//! returned to the caller ([`crate::block_palette::PaletteHost`]), which registers
//! it with the UI layer as a palette tile's thumbnail.
//!
//! This is the UI PREVIEW half of the block palette's GPU work. It lives in the
//! shell (not the `display` scene-view crate) because it renders a palette widget,
//! not the scene: it reaches DOWN into `display` only for the shared block-texture
//! bind-group layout ([`display::block_texture::block_texture_bind_group_layout`])
//! and into `assets` for the decoded RGBA. The applied (loaded) SCENE material
//! stays in `display::block_texture`.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use assets::DecodedRgba;
use display::block_texture::block_texture_bind_group_layout;

/// Edge length (pixels) of each square thumbnail texture (prototype 96×96).
pub const THUMBNAIL_SIZE: u32 = 96;

/// Thumbnail offscreen format. MUST be `Rgba8Unorm` (NOT sRGB): the UI layer's
/// `register_native_texture` requires it. We therefore sample the block texture
/// as raw bytes (also `Rgba8Unorm`) and apply lighting in that same space, so the
/// thumbnail reads like the prototype's preview without a double sRGB encode.
const THUMBNAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One thumbnail-cube vertex (the textured cube used only for the preview render).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ThumbnailVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ThumbnailUniforms {
    view_projection: [[f32; 4]; 4],
}

/// Offscreen renderer for 45° cube thumbnails (one per palette tile).
pub struct ThumbnailRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl ThumbnailRenderer {
    /// Build the thumbnail pipeline. The view-projection is fixed at construction
    /// (the camera never moves), uploaded once into the uniform buffer.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let (vertices, indices) = textured_cube_geometry();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thumbnail cube vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thumbnail cube indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("thumbnail uniforms"),
            contents: bytemuck::bytes_of(&ThumbnailUniforms {
                view_projection: thumbnail_view_projection().to_cols_array_2d(),
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("thumbnail uniform layout"),
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
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thumbnail uniform bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let texture_bind_group_layout = block_texture_bind_group_layout(device, "thumbnail");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("thumbnail sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("thumbnail shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/thumbnail.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("thumbnail pipeline layout"),
            bind_group_layouts: &[Some(&uniform_bind_group_layout), Some(&texture_bind_group_layout)],
            immediate_size: 0,
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ThumbnailVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
            ],
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("thumbnail pipeline"),
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
                    format: THUMBNAIL_FORMAT,
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });

        let _ = queue; // uniform uploaded via buffer init; queue kept for symmetry.

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            uniform_bind_group,
            texture_bind_group_layout,
            sampler,
        }
    }

    /// Render one decoded block texture into a fresh offscreen thumbnail texture
    /// and return it (caller registers it with the UI layer). The texture stays alive as
    /// long as the returned handle does.
    pub fn render_thumbnail(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        decoded: &DecodedRgba,
    ) -> wgpu::Texture {
        let block_texture = upload_block_texture(device, queue, decoded, THUMBNAIL_FORMAT);
        let block_view = block_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("thumbnail block texture bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&block_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumbnail target"),
            size: wgpu::Extent3d {
                width: THUMBNAIL_SIZE,
                height: THUMBNAIL_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: THUMBNAIL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("thumbnail encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("thumbnail pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Transparent clear so the tile shows the cube on the panel
                        // background (the UI layer composites the alpha).
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &texture_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        target
    }
}

/// The fixed thumbnail camera view-projection (prototype `thumbCam`): an
/// orthographic box `[-0.92, 0.92]` looking from azimuth π/4, elevation 0.62.
fn thumbnail_view_projection() -> glam::Mat4 {
    let azimuth = std::f32::consts::FRAC_PI_4;
    let elevation = 0.62f32;
    let radius = 4.0f32;
    let eye = glam::Vec3::new(
        elevation.cos() * azimuth.sin() * radius,
        elevation.sin() * radius,
        elevation.cos() * azimuth.cos() * radius,
    );
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
    // Orthographic frustum half-extent 0.92 (prototype), near/far around the cube.
    let projection = glam::Mat4::orthographic_rh(-0.92, 0.92, -0.92, 0.92, 0.1, 10.0);
    projection * view
}

/// Upload a decoded RGBA buffer as a 2D texture in `format` (no mipmaps). The
/// thumbnail preview's private uploader (the applied scene material uses the
/// renderer's own D2Array uploader instead).
fn upload_block_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    decoded: &DecodedRgba,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let (width, height, ref pixels) = *decoded;
    let size = wgpu::Extent3d {
        width: width.max(1),
        height: height.max(1),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("loaded block texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
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
            bytes_per_row: Some(4 * width.max(1)),
            rows_per_image: Some(height.max(1)),
        },
        size,
    );
    texture
}

/// Build a textured unit cube spanning `[-1, 1]` with one outward normal and a
/// full 0..1 UV per face (the WHOLE block texture on every face — a preview, not
/// a per-voxel slice).
fn textured_cube_geometry() -> (Vec<ThumbnailVertex>, Vec<u16>) {
    const FACE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [[1.0, -1.0, 1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]),
        ([-1.0, 0.0, 0.0], [[-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]]),
        ([0.0, 1.0, 0.0], [[-1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]]),
        ([0.0, -1.0, 0.0], [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]),
        ([0.0, 0.0, 1.0], [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
        ([0.0, 0.0, -1.0], [[1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, -1.0]]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = vertices.len() as u16;
        for (corner_index, corner) in corners.iter().enumerate() {
            vertices.push(ThumbnailVertex {
                position: *corner,
                normal,
                uv: FACE_UVS[corner_index],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}
