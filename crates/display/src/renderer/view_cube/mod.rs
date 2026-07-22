//! View cube (Milestone 5; restyled to the "Signal" language, issue #86 / ADR 0018
//! Decision 8) — see `docs/design/viewport-chrome-signal.md`.
//!
//! The cube is a near-black instrument-panel widget in the **top-right** of the 3D
//! viewport (industry norm): translucent flat face fills within `#10141a`–`#1b2126`,
//! hairline `#2b3238` slice lines partitioning each face 3×3 at the **68 %** centre
//! patch (so the drawn partition IS the pick partition — see
//! [`raycast::VIEW_CUBE_ZONE_THRESHOLD`]), a `#59636d` silhouette, projected FRONT /
//! TOP / RIGHT labels, and the three axis-coloured cube edges emanating from the
//! front-bottom-right corner. Hovering a zone lights every across-the-fold facet of
//! the 26-element in the `#9cb4d8` accent (computed geometrically in the shader from
//! the element's [`camera::ViewCubeElement::axis_selectors`]).
//!
//! This module owns the GPU renderer + its pipelines; the pure-CPU asset generation
//! lives in two coupling-free sinks: [`geometry`] (mesh + wireframe → `Vec`s) and
//! [`labels`] (face-label textures + bitmap font → RGBA8 bytes).

use super::*;

mod geometry;
mod labels;

// The CPU asset generators used by `ViewCubeRenderer::new`. `view_cube_geometry` is
// re-exported `pub(crate)` so the renderer-module glob still hands it to the unit
// tests (`renderer::tests`); the other two are internal to the GPU half.
pub(crate) use geometry::view_cube_geometry;
use geometry::{expand_thick_lines, view_cube_edges};
use labels::generate_face_label_textures;

/// Edge length (pixels) of the corner view-cube viewport (top-right). Bumped 128 → 144
/// (issue #91 item 3) for a modestly larger cube; the 16 px margin is unchanged and the
/// rail anchor + shell hit-testing derive from this constant, so they track it.
pub const VIEW_CUBE_VIEWPORT_PIXELS: u32 = 144;
/// Margin (pixels) from the viewport's top-right corner to the cube.
pub const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;

/// The cube's top-left origin (physical pixels) within the central 3D `viewport`
/// (`[x, y, w, h]`), placed in the **top-right** just left of the side-panel edge.
/// Both the renderer and the shell's hit-testing derive the cube rect from this ONE
/// function so the drawn cube and the pick rect always coincide. Returns `None` when
/// the viewport is smaller than the cube + margin on either axis — the **minimum
/// on-screen size** rule that keeps the 68 %-centre slice lines' 16 % edge strips
/// (≈ `0.16 · 144 ≈ 23 px`) comfortably hittable; below it the cube is not drawn.
pub fn view_cube_corner(viewport: [u32; 4], right_inset_px: u32) -> Option<(u32, u32)> {
    let [viewport_x, viewport_y, viewport_width, viewport_height] = viewport;
    let margin = VIEW_CUBE_VIEWPORT_MARGIN;
    let size = VIEW_CUBE_VIEWPORT_PIXELS;
    // Issue #88: the cube's right inset is the floating display stack's current width (the
    // cube slides left of it), replacing the old bare `margin`. It must still clear the
    // cube + a vertical margin — below that the cube isn't drawn (the min on-screen rule).
    if viewport_width < right_inset_px + size || viewport_height < margin + size {
        return None;
    }
    // Top-RIGHT: hug the right edge (just left of the display stack), inset for the stack
    // horizontally and `margin` down from the top.
    let corner_x = viewport_x + viewport_width - right_inset_px - size;
    let corner_y = viewport_y + margin;
    Some((corner_x, corner_y))
}

/// Edge length of each square face-label texture.
const FACE_LABEL_TEXTURE_SIZE: u32 = 128;

// --- Signal tokens (docs/design/viewport-chrome-signal.md §Tokens) ---
// The face-fill (now opaque, issue #91 item 6), the `#2b3238` slice lines (now an SDF)
// and the `#9cb4d8` hover accent all live in `viewcube.wgsl`; the tokens baked on the
// CPU (silhouette + axis edges + labels) are below.
/// Cube silhouette colour `#59636d` (the 9 non-axis edges).
const SILHOUETTE_HEX: u32 = 0x59_63_6d;
/// Face-label lettering colour `#aeb9c4` (Signal "text — secondary"), monospace.
const FACE_LABEL_HEX: u32 = 0xae_b9_c4;
/// Axis colours (Signal): X `#d9603f`, Y `#7dba6a`, Z `#9cb4d8`.
const AXIS_X_HEX: u32 = 0xd9_60_3f;
const AXIS_Y_HEX: u32 = 0x7d_ba_6a;
const AXIS_Z_HEX: u32 = 0x9c_b4_d8;

/// One view-cube vertex: position, face normal, face UV, and the texture-array
/// layer (face index in +X,-X,+Y,-Y,+Z,-Z order).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CubeLabelVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    uv: [f32; 2],
    layer: u32,
}

/// One expanded thick-line vertex (issue #91 item 3): the segment's two endpoints (so
/// the vertex shader can compute the screen-space direction), the line colour, and a
/// `[side, end]` selector (`side` ∈ {-1,+1} across the width, `end` ∈ {0,1} picks the
/// endpoint). Six per source segment → a screen-space quad of constant pixel width.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ThickLineVertex {
    position_a: [f32; 3],
    position_b: [f32; 3],
    color: [f32; 4],
    side_end: [f32; 2],
}

/// Uniforms for the anti-aliased cube-line pipeline: the cube VP matrix, the cube's
/// square on-screen pixel size (to convert the pixel width into an NDC offset), and the
/// line's core half-width + feather in pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CubeLineUniforms {
    view_projection: [[f32; 4]; 4],
    viewport_px: [f32; 2],
    half_width_px: f32,
    feather_px: f32,
}

/// Core half-width (px) of the cube linework → a ~1.4 px line, plus a 1 px feather.
const CUBE_LINE_HALF_WIDTH_PX: f32 = 0.7;
const CUBE_LINE_FEATHER_PX: f32 = 1.0;

/// The corner view cube: a labelled cube mirroring the main camera's orientation, plus
/// a silhouette + axis-coloured edge wireframe (Signal style, see the module doc).
/// Rendered into a scissored top-right viewport in its own pass (depth cleared there
/// first).
pub struct ViewCubeRenderer {
    face_pipeline: wgpu::RenderPipeline,
    /// The anti-aliased screen-space thick-line pipeline (issue #91 item 3): silhouette,
    /// axis edges, and X/Y/Z letters at constant ~1.4 px width.
    line_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    line_buffer: wgpu::Buffer,
    line_vertex_count: u32,
    line_uniform_buffer: wgpu::Buffer,
    line_uniform_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    label_bind_group: wgpu::BindGroup,
    // --- Issue #91 item 3: composite the resolved MSAA cube over the scene ---
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    composite_sampler: wgpu::Sampler,
    /// The colour target format, for building the transient offscreen MSAA + resolve
    /// textures the cube renders into.
    color_format: wgpu::TextureFormat,
    // --- #13 Step 2: screen-space chrome overlay (rotate + roll arrows) ---
    chrome_pipeline: wgpu::RenderPipeline,
    chrome_bind_group: wgpu::BindGroup,
    chrome_vertex_buffer: wgpu::Buffer,
    /// Capacity (in vertices) of `chrome_vertex_buffer`; the per-frame glyph quads
    /// fit within this fixed cap (4 glyphs × 6 verts, generous).
    chrome_vertex_capacity: u32,
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

        let line_vertices = expand_thick_lines(&view_cube_edges());
        let line_vertex_count = line_vertices.len() as u32;
        let line_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("view cube edge lines"),
            contents: bytemuck::cast_slice(&line_vertices),
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
        // Allocated with the shared padded layer count, not 6: this texture is square and
        // single-sampled, so a 6-layer allocation trips wgpu's GL cubemap heuristic and
        // samples black. See `FACE_MATERIAL_ARRAY_LAYERS`. Only the first 6 are written.
        let label_pixels = generate_face_label_textures();
        let label_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view cube label textures"),
            size: wgpu::Extent3d {
                width: FACE_LABEL_TEXTURE_SIZE,
                height: FACE_LABEL_TEXTURE_SIZE,
                depth_or_array_layers: crate::renderer::FACE_MATERIAL_ARRAY_LAYERS,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/viewcube.wgsl").into()),
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
        // 3D MSAA resolve, before the shell's UI pass), so its pipelines use sample_count 1.
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
                // Signal: the faces are FULLY OPAQUE flat fills (issue #91 item 6), reading
                // solid over the scene; the pipeline still alpha-blends, but only for the
                // AA slice-line feathering. Back faces are culled and the three visible
                // faces never overlap in screen space, so no per-face depth sorting is
                // needed.
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
            // Issue #91 (item 3): the cube renders into its own 4× MSAA offscreen target
            // (resolved + composited in `draw`) so the face silhouettes + linework are
            // coverage-anti-aliased, not just the SDF/thick-line feathering.
            multisample: wgpu::MultisampleState { count: MSAA_SAMPLE_COUNT, mask: !0, alpha_to_coverage_enabled: false },
            multiview_mask: None,
            cache: None,
        });

        // --- Edge lines: constant screen-space width, anti-aliased (issue #91 item 3) ---
        let line_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view cube line uniforms"),
            size: std::mem::size_of::<CubeLineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let line_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view cube line uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    // The fragment stage reads the width/feather for the AA edge too.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let line_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view cube line uniform bind group"),
            layout: &line_uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: line_uniform_buffer.as_entire_binding(),
            }],
        });
        let line_pipeline = build_cube_line_pipeline(
            device,
            color_format,
            &line_uniform_bind_group_layout,
        );

        // --- #13 Step 2: screen-space chrome overlay pipeline + glyph textures ---
        let (chrome_pipeline, chrome_bind_group) =
            build_chrome_overlay(device, queue, color_format, MSAA_SAMPLE_COUNT);

        // --- Composite pipeline (issue #91 item 3): blend the resolved MSAA cube on top ---
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("view cube composite layout"),
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
        let composite_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("view cube composite sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let composite_pipeline =
            build_cube_composite_pipeline(device, color_format, &composite_bind_group_layout);
        // Cap: the four persistent rotate arrows + one hovered roll arrow on screen at
        // once; size generously for all glyph quads (6 verts each).
        let chrome_vertex_capacity = 12 * 6;
        let chrome_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view cube chrome vertices"),
            size: (chrome_vertex_capacity as usize * std::mem::size_of::<ChromeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            face_pipeline,
            line_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            line_buffer,
            line_vertex_count,
            line_uniform_buffer,
            line_uniform_bind_group,
            uniform_buffer,
            uniform_bind_group,
            label_bind_group,
            composite_pipeline,
            composite_bind_group_layout,
            composite_sampler,
            color_format,
            chrome_pipeline,
            chrome_bind_group,
            chrome_vertex_buffer,
            chrome_vertex_capacity: chrome_vertex_capacity as u32,
        }
    }

    /// Upload the view-cube camera matrix (`OrbitCamera::view_cube_view_projection`).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            depth_bias: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        // The anti-aliased line pipeline (issue #91 item 3) needs the same VP plus the
        // cube's square on-screen pixel size + line width to expand its screen-space quads.
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f32;
        let line_uniforms = CubeLineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            viewport_px: [size, size],
            half_width_px: CUBE_LINE_HALF_WIDTH_PX,
            feather_px: CUBE_LINE_FEATHER_PX,
        };
        queue.write_buffer(&self.line_uniform_buffer, 0, bytemuck::bytes_of(&line_uniforms));
    }

    /// Draw the cube into a scissored corner of `target_view` (its own render pass,
    /// with a freshly-cleared private depth texture). The colour attachment loads
    /// the already-resolved scene so only the corner is touched.
    ///
    /// Issue #25: the corner is the top-left of the CENTRAL 3D viewport rect
    /// (`viewport_x/y/w/h`, physical pixels), NOT the whole window — so the cube
    /// lines up with the visible 3D area instead of hiding behind the side panel.
    /// `target_width/height` are the full target dims (the colour + depth
    /// attachments span the whole target; the scissor confines the draw).
    ///
    /// #13 Step 2: `hovered_zone` is the chrome zone currently under the cursor
    /// (from `classify_cube_point`). The roll arrows are drawn ONLY when their zone is
    /// hovered. #13 Step 6 follow-up: the four rotate arrows are drawn PERSISTENTLY
    /// whenever `rotate_arrows_visible` (the view is face-constrained), with the hovered
    /// one brightened. (Home/Fit left the cube for the Signal icon rail — ADR 0018
    /// Decision 8.) The chrome is a screen-space overlay FIXED to the cube rect (it does
    /// NOT rotate with the cube), laid out in the same `rect.size` fractions Step 1
    /// hit-tests against.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        target_width: u32,
        target_height: u32,
        viewport: [u32; 4],
        right_inset_px: u32,
        hovered_zone: Option<camera::CubeChromeZone>,
        rotate_arrows_visible: bool,
    ) {
        // Signal hover: when a face/edge/corner ELEMENT is hovered, upload its per-axis
        // sign selector `[sx, sy, sz, active]` into the cube uniform's `depth_bias`
        // slot (byte offset 64). The shader lights a face fragment at cube position `p`
        // iff, on every axis, `p` is on the selector's side of the 68 % centre patch —
        // which highlights exactly the 1/2/3 across-the-fold facets of the element.
        // `active = 0` for a non-Element hover (arrow/badge) or no hover clears it.
        let highlight = match hovered_zone {
            Some(camera::CubeChromeZone::Element(element)) => {
                let [sx, sy, sz] = element.axis_selectors();
                [sx, sy, sz, 1.0f32]
            }
            _ => [0.0f32; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 64, bytemuck::bytes_of(&highlight));

        let size = VIEW_CUBE_VIEWPORT_PIXELS;
        // Signal: top-RIGHT placement (shared with the shell's hit-testing). The helper
        // also enforces the minimum on-screen size (bails when the viewport is too small).
        let Some((corner_x, corner_y)) = view_cube_corner(viewport, right_inset_px) else {
            return;
        };
        // Bail if the cube would fall outside the actual target (defensive).
        if corner_x + size > target_width || corner_y + size > target_height {
            return;
        }
        // Issue #91 (item 3): render the cube into its OWN small 4× MSAA offscreen target
        // (cleared transparent), resolve it to a single-sample texture with coverage-AA'd
        // silhouettes, then composite that over the scene in the corner. This anti-aliases
        // the opaque FACE silhouettes as well as the linework — the whole cube reads clean.
        let msaa_color = create_msaa_color_view(device, size, size, self.color_format);
        let msaa_depth = create_depth_view(device, size, size);
        let resolve_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("view cube resolve texture"),
            size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.color_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let resolve_view = resolve_texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("view cube msaa pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_color,
                    resolve_target: Some(&resolve_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Transparent clear: the corners of the square around the cube stay
                        // empty so the composite lets the scene background show through.
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &msaa_depth,
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
            // Full offscreen viewport (the cube square); the composite places it at the corner.
            pass.set_viewport(0.0, 0.0, size as f32, size as f32, 0.0, 1.0);
            pass.set_scissor_rect(0, 0, size, size);

            pass.set_pipeline(&self.face_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_bind_group(1, &self.label_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..self.index_count, 0, 0..1);

            pass.set_pipeline(&self.line_pipeline);
            pass.set_bind_group(0, &self.line_uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.line_buffer.slice(..));
            pass.draw(0..self.line_vertex_count, 0..1);

            // --- #13 Step 2: screen-space chrome overlay, fixed to the cube rect. ---
            let chrome = build_chrome_vertices(hovered_zone, rotate_arrows_visible);
            if !chrome.is_empty() {
                let count = chrome.len().min(self.chrome_vertex_capacity as usize);
                queue.write_buffer(
                    &self.chrome_vertex_buffer,
                    0,
                    bytemuck::cast_slice(&chrome[..count]),
                );
                pass.set_pipeline(&self.chrome_pipeline);
                pass.set_bind_group(0, &self.chrome_bind_group, &[]);
                pass.set_vertex_buffer(0, self.chrome_vertex_buffer.slice(..));
                pass.draw(0..count as u32, 0..1);
            }
        }

        // Composite the resolved cube over the scene at the corner (premultiplied OVER).
        let composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("view cube composite bind group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&resolve_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.composite_sampler),
                },
            ],
        });
        let mut composite_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("view cube composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        composite_pass.set_viewport(corner_x as f32, corner_y as f32, size as f32, size as f32, 0.0, 1.0);
        composite_pass.set_scissor_rect(corner_x, corner_y, size, size);
        composite_pass.set_pipeline(&self.composite_pipeline);
        composite_pass.set_bind_group(0, &composite_bind_group, &[]);
        composite_pass.draw(0..3, 0..1);
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
            // #13 Step 6.2: the cube fragment shader now reads `highlight` from this
            // uniform too, so it must be visible to BOTH stages.
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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

/// Build the anti-aliased cube-line pipeline (issue #91 item 3): screen-space thick-line
/// quads (`viewcube_lines.wgsl`), alpha-blended over the opaque faces, depth-tested `Less`
/// against the cube's private depth, 1 sample (the cube pass resolves at 1 sample).
fn build_cube_line_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    uniform_bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view cube line shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/viewcube_lines.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view cube line pipeline layout"),
        bind_group_layouts: &[Some(uniform_bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ThickLineVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 12, shader_location: 1, format: wgpu::VertexFormat::Float32x3 },
            wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 40, shader_location: 3, format: wgpu::VertexFormat::Float32x2 },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("view cube line pipeline"),
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
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // The expanded quads can wind either way depending on the projected edge
            // direction, so no back-face culling.
            cull_mode: None,
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
        multisample: wgpu::MultisampleState { count: MSAA_SAMPLE_COUNT, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    })
}

/// Build the composite pipeline (issue #91 item 3): a fullscreen textured quad that blends
/// the resolved MSAA cube over the scene in the corner with premultiplied-alpha blending.
fn build_cube_composite_pipeline(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view cube composite shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/viewcube_composite.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view cube composite pipeline layout"),
        bind_group_layouts: &[Some(bind_group_layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("view cube composite pipeline"),
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
                // Premultiplied-alpha OVER: dst = src + dst·(1 − src.a).
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    })
}
