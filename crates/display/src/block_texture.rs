//! Block-texture GPU work: thumbnail rendering + the applied (loaded) material.
//!
//! This is the pure-wgpu half of the block palette (all GPU work, no UI toolkit). It owns:
//!
//!   * [`ThumbnailRenderer`] — a tiny offscreen pipeline that draws a textured
//!     unit cube at a fixed 45° orthographic view (prototype `thumbCam`:
//!     azimuth π/4, elevation 0.62) into a ~96×96 `Rgba8Unorm` texture. The
//!     rendered texture is returned to the caller, which registers it with the UI layer.
//!   * [`LoadedMaterial`] — a runtime-loaded RGBA block texture uploaded as a
//!     bind group laid out exactly like the procedural Stone/Wood/Plain ones, so
//!     the per-voxel slice shader textures the model with a real VS block.
//!
//! The UI-facing palette state (`BlockPalette` / `PaletteTile`) lives in the shell
//! crate's `block_palette` module; this module is the GPU backing behind it.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::assets::DecodedRgba;

/// Edge length (pixels) of each square thumbnail texture (prototype 96×96).
pub const THUMBNAIL_SIZE: u32 = 96;

/// Thumbnail offscreen format. MUST be `Rgba8Unorm` (NOT sRGB): the UI layer's
/// `register_native_texture` requires it. We therefore sample the block texture
/// as raw bytes (also `Rgba8Unorm`) and apply lighting in that same space, so the
/// thumbnail reads like the prototype's preview without a double sRGB encode.
const THUMBNAIL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One thumbnail-cube vertex (shared with the loaded material has no use; this is
/// the textured cube used only for the preview render).
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

/// A runtime-loaded block material: a 6-layer texture array (one layer per cube
/// face), bound exactly like the procedural materials so the per-voxel slice
/// shader treats it identically. A uniform block puts the same image on all six
/// layers; a per-face block (M7) puts each face's PNG on its own layer.
pub struct LoadedMaterial {
    pub bind_group: wgpu::BindGroup,
    /// The label of the applied block (for the panel readout).
    pub label: String,
    /// Whether the resolved faces are genuinely per-face (top != side); used for
    /// logging / verification (`--list-perface`, `--apply-block`).
    pub is_per_face: bool,
    /// Average RGBA colour of the side face — the representative palette colour
    /// used by the `.vox` export (M8).
    pub average_color: [u8; 4],
}

impl LoadedMaterial {
    /// Build a 6-layer material DIRECTLY from six raw RGBA8 face buffers (part of
    /// #20 verification / synthetic blocks). Each `layers[i]` is a tightly-packed
    /// `width*height*4` RGBA8 buffer in CubeFaceSlot order (0 +X, 1 -X, 2 +Y, 3 -Y,
    /// 4 +Z, 5 -Z); the texture is uploaded as the SAME sRGB D2Array + bind-group
    /// shape `from_faces` produces, so it is interchangeable on both render paths
    /// without needing a real VS install. Used by the headless harness to apply six
    /// distinct solid-colour faces and prove the cuboid path textures per-face.
    #[allow(clippy::too_many_arguments)]
    pub fn from_face_layers(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
        layers: &[&[u8]; 6],
        label: String,
    ) -> Self {
        let average_color = average_rgba(layers[0]);
        let texture = crate::renderer::upload_face_material_texture(
            device, queue, width, height, layers,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("synthetic loaded block material bind group"),
            layout: material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            bind_group,
            label,
            is_per_face: true,
            average_color,
        }
    }

    /// Build a 6-layer material from resolved per-face PNG paths (M7).
    ///
    /// Each face PNG is decoded and, if face sizes differ, rescaled to the
    /// largest common size so all six layers share one `Extent3d`. The texture is
    /// sRGB so the main shader's lighting + grid overlay mix in linear space,
    /// matching Stone/Wood/Plain. Falls back to a uniform layer set from any face
    /// that fails to decode.
    pub fn from_faces(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_bind_group_layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        faces: &crate::assets::FaceTextures,
        label: String,
    ) -> Self {
        let is_per_face = !faces.is_uniform();

        // Decode each face PNG (in CubeFaceSlot layer order). A face that fails
        // to decode is left None and filled from a sibling below.
        let decoded_faces: Vec<Option<DecodedRgba>> = faces
            .paths
            .iter()
            .map(|path| crate::assets::decode_rgba(path))
            .collect();

        // Pick a representative decoded face to size the array + fill gaps.
        let representative = decoded_faces
            .iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or((1, 1, vec![0u8, 0u8, 0u8, 255u8]));
        let (target_width, target_height, _) = representative;

        // Normalise every layer to (target_width, target_height) so the array's
        // layers share one size. Missing faces use the representative image.
        let layers_rgba: Vec<Vec<u8>> = decoded_faces
            .into_iter()
            .map(|face| {
                let decoded = face.unwrap_or_else(|| representative.clone());
                resize_rgba_nearest(&decoded, target_width, target_height)
            })
            .collect();

        // Representative palette colour for the .vox export: the average of the
        // side face (layer 0), which is the most representative of the block.
        let average_color = average_rgba(&layers_rgba[0]);

        let layer_slices: [&[u8]; 6] = [
            &layers_rgba[0],
            &layers_rgba[1],
            &layers_rgba[2],
            &layers_rgba[3],
            &layers_rgba[4],
            &layers_rgba[5],
        ];
        let texture = crate::renderer::upload_face_material_texture(
            device,
            queue,
            target_width,
            target_height,
            &layer_slices,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("loaded block material bind group"),
            layout: material_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        Self {
            bind_group,
            label,
            is_per_face,
            average_color,
        }
    }
}

/// Average RGBA of a tightly-packed RGBA8 buffer (alpha forced opaque). Used for
/// the `.vox` export's representative palette colour (M8).
fn average_rgba(pixels: &[u8]) -> [u8; 4] {
    if pixels.len() < 4 {
        return [0x80, 0x80, 0x80, 0xff];
    }
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

/// Nearest-neighbour rescale of a decoded RGBA image to `target_width`×
/// `target_height`. A no-op (clone) when the size already matches. Nearest keeps
/// VS block textures crisp and matches the material sampler's filter.
fn resize_rgba_nearest(decoded: &DecodedRgba, target_width: u32, target_height: u32) -> Vec<u8> {
    let (width, height, ref pixels) = *decoded;
    if width == target_width && height == target_height {
        return pixels.clone();
    }
    let width = width.max(1);
    let height = height.max(1);
    let mut out = vec![0u8; (target_width * target_height * 4) as usize];
    for y in 0..target_height {
        let source_y = (y * height / target_height.max(1)).min(height - 1);
        for x in 0..target_width {
            let source_x = (x * width / target_width.max(1)).min(width - 1);
            let source_index = ((source_y * width + source_x) * 4) as usize;
            let dest_index = ((y * target_width + x) * 4) as usize;
            if source_index + 4 <= pixels.len() {
                out[dest_index..dest_index + 4]
                    .copy_from_slice(&pixels[source_index..source_index + 4]);
            }
        }
    }
    out
}

/// Build the standard block-texture bind group layout (binding 0 = texture,
/// binding 1 = sampler) — the SAME shape the voxel renderer's material layout
/// uses, so a loaded texture is interchangeable with the procedural ones.
pub fn block_texture_bind_group_layout(
    device: &wgpu::Device,
    label: &str,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label} block texture layout")),
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

/// Upload a decoded RGBA buffer as a 2D texture in `format` (no mipmaps).
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
