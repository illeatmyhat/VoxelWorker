//! View cube (Milestone 5; restyled to the "Signal" language, issue #86 / ADR 0018
//! Decision 8) — ARCHITECTURE.md §4 + `docs/design/viewport-chrome-signal.md`.
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

use super::*;

/// Edge length (pixels) of the corner view-cube viewport (top-right).
pub const VIEW_CUBE_VIEWPORT_PIXELS: u32 = 128;
/// Margin (pixels) from the viewport's top-right corner to the cube.
pub const VIEW_CUBE_VIEWPORT_MARGIN: u32 = 16;

/// The cube's top-left origin (physical pixels) within the central 3D `viewport`
/// (`[x, y, w, h]`), placed in the **top-right** just left of the side-panel edge.
/// Both the renderer and the shell's hit-testing derive the cube rect from this ONE
/// function so the drawn cube and the pick rect always coincide. Returns `None` when
/// the viewport is smaller than the cube + margin on either axis — the **minimum
/// on-screen size** rule that keeps the 68 %-centre slice lines' 16 % edge strips
/// (≈ `0.16 · 128 ≈ 20 px`) comfortably hittable; below it the cube is not drawn.
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
// The translucent face-fill alpha (~0.82) and the `#9cb4d8` hover accent live in
// `viewcube.wgsl` (fragment output + `mix`); the tokens baked on the CPU are below.
/// Hairline slice-line colour `#2b3238` (the 3×3 face partition).
const SLICE_LINE_HEX: u32 = 0x2b_32_38;
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/viewcube.wgsl").into()),
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
                // Signal: the faces are TRANSLUCENT flat fills (~80 %) over the
                // resolved scene, so the pipeline alpha-blends. Back faces are culled
                // and the three visible faces never overlap in screen space, so no
                // per-face depth sorting is needed.
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

        // --- #13 Step 2: screen-space chrome overlay pipeline + glyph textures ---
        let (chrome_pipeline, chrome_bind_group) =
            build_chrome_overlay(device, queue, color_format);
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
            edge_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            edge_buffer,
            edge_vertex_count,
            uniform_buffer,
            uniform_bind_group,
            label_bind_group,
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

        pass.set_viewport(corner_x as f32, corner_y as f32, size as f32, size as f32, 0.0, 1.0);
        pass.set_scissor_rect(corner_x, corner_y, size, size);

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

/// Build the labelled-cube geometry (side 1.4, centred on origin). Face order +X,
/// -X, +Y, -Y, +Z, -Z (matches `materialIndex` / `CubeFace`).
pub(crate) fn view_cube_geometry() -> (Vec<CubeLabelVertex>, Vec<u16>) {
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
        // Z-up: the BACK (+Y, layer 2) and BOTTOM (−Z, layer 5) faces wind such that
        // the shared UV table maps their label upside-down. Rotate just those two
        // faces' UVs 180° (corner_index + 2) so every label reads upright — the fix
        // lives in the unwrap, keeping the label textures themselves canonical.
        let uv_rotated = layer == 2 || layer == 5;
        for (corner_index, corner) in corners.iter().enumerate() {
            let uv_index = if uv_rotated { (corner_index + 2) % 4 } else { corner_index };
            vertices.push(CubeLabelVertex {
                position: *corner,
                normal: *normal,
                uv: UVS[uv_index],
                layer: layer as u32,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// The Signal cube wireframe: the 12 silhouette edges (`#59636d`), the three
/// axis-coloured edges emanating from the front-bottom-right corner
/// (`(+HALF, −HALF, −HALF)` = right/front/bottom), and small projected X/Y/Z letter
/// glyphs at the far ends of those edges. All drawn by the shared line pipeline
/// (per-vertex colour, cube VP transform), so the axis triad foreshortens WITH the
/// cube — never a screen-space approximation.
///
/// Z-up world mapping (front = −Y): from the shared corner, X (`#d9603f`) runs along
/// the bottom-front edge toward −X, Y (`#7dba6a`) up the receding right-bottom edge
/// toward +Y, and Z (`#9cb4d8`) up the front-right vertical toward +Z.
fn view_cube_edges() -> Vec<LineVertex> {
    const HALF: f32 = 0.705; // a hair outside the faces so the edges read crisply
    let silhouette = with_alpha(srgb_hex_to_linear(SILHOUETTE_HEX), 1.0);
    let axis_x = with_alpha(srgb_hex_to_linear(AXIS_X_HEX), 1.0);
    let axis_y = with_alpha(srgb_hex_to_linear(AXIS_Y_HEX), 1.0);
    let axis_z = with_alpha(srgb_hex_to_linear(AXIS_Z_HEX), 1.0);

    // The three axis edges share the front-bottom-right corner `(+HALF, −HALF, −HALF)`;
    // these are their FAR endpoints (where the letter glyphs sit).
    let x_far = [-HALF, -HALF, -HALF]; // along the bottom-front edge (−X)
    let y_far = [HALF, HALF, -HALF]; //  up the receding right-bottom edge (+Y)
    let z_far = [HALF, -HALF, HALF]; //  up the front-right vertical (+Z)

    let corners = [
        [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF],
        [-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF],
    ];
    // The 12 edges as index pairs; the three axis edges are tagged with their colour.
    let edges: [((usize, usize), [f32; 4]); 12] = [
        ((0, 1), axis_x),       // bottom-front (varies X): the X axis edge
        ((1, 2), axis_y),       // right-bottom (varies Y): the Y axis edge
        ((2, 3), silhouette),
        ((3, 0), silhouette),
        ((4, 5), silhouette),
        ((5, 6), silhouette),
        ((6, 7), silhouette),
        ((7, 4), silhouette),
        ((0, 4), silhouette),
        ((1, 5), axis_z),       // front-right vertical (varies Z): the Z axis edge
        ((2, 6), silhouette),
        ((3, 7), silhouette),
    ];
    let mut vertices = Vec::with_capacity(edges.len() * 2 + 24);
    for ((a, b), color) in edges {
        vertices.push(LineVertex { position: corners[a], color });
        vertices.push(LineVertex { position: corners[b], color });
    }

    // Axis letter glyphs at the FAR ends (offset a touch outward from the corner so
    // they read past the silhouette). Each glyph is drawn in a cube-space plane
    // (right/up unit vectors) and projected with the cube.
    const GLYPH: f32 = 0.20; // glyph box side, cube units
    const OUT: f32 = 0.10; // outward nudge from the endpoint
    // X: on the bottom-front edge; stand it in the XZ plane (right = +X, up = +Z),
    // nudged down/forward so it sits just outside the front-bottom edge.
    push_line_letter(
        &mut vertices,
        'X',
        [x_far[0] - OUT, x_far[1] - OUT, x_far[2] - OUT],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_x,
    );
    // Y: on the receding right-bottom edge; plane (right = +Y, up = +Z).
    push_line_letter(
        &mut vertices,
        'Y',
        [y_far[0] + OUT, y_far[1] + OUT, y_far[2] - OUT],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_y,
    );
    // Z: on the front-right vertical; plane (right = +X, up = +Z).
    push_line_letter(
        &mut vertices,
        'Z',
        [z_far[0] + OUT, z_far[1] - OUT, z_far[2] + OUT],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_z,
    );
    vertices
}

/// One letter stroke in the unit `[-0.5, 0.5]²` glyph cell: a `(u, v)` start/end pair.
type LetterStroke = ((f32, f32), (f32, f32));

/// Append the line segments of a single axis letter (`X`/`Y`/`Z`) to `vertices`,
/// centred at `center` in the cube-space plane spanned by unit vectors `right` and
/// `up`, with box side `scale` and colour `color`. Strokes are defined in a unit
/// `[-0.5, 0.5]²` cell (u along `right`, v along `up`) and mapped into cube space, so
/// the glyph foreshortens with the cube under the shared VP.
fn push_line_letter(
    vertices: &mut Vec<LineVertex>,
    letter: char,
    center: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    scale: f32,
    color: [f32; 4],
) {
    // Unit-cell strokes (u, v) → (u, v) endpoint pairs per letter.
    let strokes: &[LetterStroke] = match letter {
        'X' => &[((-0.5, -0.5), (0.5, 0.5)), ((-0.5, 0.5), (0.5, -0.5))],
        'Y' => &[
            ((-0.5, 0.5), (0.0, 0.0)),
            ((0.5, 0.5), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, -0.5)),
        ],
        'Z' => &[
            ((-0.5, 0.5), (0.5, 0.5)),
            ((0.5, 0.5), (-0.5, -0.5)),
            ((-0.5, -0.5), (0.5, -0.5)),
        ],
        _ => &[],
    };
    let map = |u: f32, v: f32| {
        [
            center[0] + (right[0] * u + up[0] * v) * scale,
            center[1] + (right[1] * u + up[1] * v) * scale,
            center[2] + (right[2] * u + up[2] * v) * scale,
        ]
    };
    for ((u0, v0), (u1, v1)) in strokes {
        vertices.push(LineVertex { position: map(*u0, *v0), color });
        vertices.push(LineVertex { position: map(*u1, *v1), color });
    }
}

/// A sRGB hex → RGBA8 (opaque) texel; the label textures are `Rgba8UnormSrgb`, so the
/// sRGB byte values are written straight through.
fn hex_texel(hex: u32) -> [u8; 4] {
    [((hex >> 16) & 0xff) as u8, ((hex >> 8) & 0xff) as u8, (hex & 0xff) as u8, 0xff]
}

/// The flat Signal fill of face `layer` (GEOMETRIC order +X,-X,+Y,-Y,+Z,-Z =
/// Right,Left,Back,Front,Top,Bottom) — distinct near-black values within
/// `#10141a`–`#1b2126` so the three visible faces (TOP lightest, FRONT mid, RIGHT
/// darker) read apart under the flat (unlit) shading.
fn face_fill_hex(layer: usize) -> u32 {
    match layer {
        0 => 0x13_18_20, // Right
        1 => 0x0f_13_18, // Left
        2 => 0x12_16_1b, // Back
        3 => 0x16_1c_22, // Front
        4 => 0x1b_21_26, // Top
        _ => 0x10_14_19, // Bottom
    }
}

/// Render the six face-label textures into one stacked RGBA8 buffer (6 layers, in
/// GEOMETRIC `materialIndex` order +X,-X,+Y,-Y,+Z,-Z). Z-up labels each geometric
/// face: +Y = BACK, −Y = FRONT, +Z = TOP, −Z = BOTTOM. Signal style: a flat
/// near-black fill (per-face, [`face_fill_hex`]), hairline `#2b3238` slice lines at
/// the 68 %-centre partition, and a monospace `#aeb9c4` label in the centre patch.
fn generate_face_label_textures() -> Vec<u8> {
    const LABELS: [&str; 6] = ["RIGHT", "LEFT", "BACK", "FRONT", "TOP", "BOTTOM"];
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let mut all = Vec::with_capacity(size * size * 4 * 6);
    for (layer, label) in LABELS.iter().enumerate() {
        all.extend_from_slice(&render_face_label(label, face_fill_hex(layer)));
    }
    all
}

/// Render one Signal face-label texture (RGBA8, `FACE_LABEL_TEXTURE_SIZE` square):
/// flat `fill_hex` background, hairline slice lines at the 68 %-centre partition, and
/// the monospace label.
fn render_face_label(label: &str, fill_hex: u32) -> Vec<u8> {
    let size = FACE_LABEL_TEXTURE_SIZE as usize;
    let background = hex_texel(fill_hex);
    let slice = hex_texel(SLICE_LINE_HEX);
    let text = hex_texel(FACE_LABEL_HEX);

    let mut pixels = vec![0u8; size * size * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&background);
    }
    let put = |pixels: &mut [u8], x: usize, y: usize, color: [u8; 4]| {
        if x < size && y < size {
            let index = (y * size + x) * 4;
            pixels[index..index + 4].copy_from_slice(&color);
        }
    };

    // Hairline 3×3 slice lines at the 68 %-centre boundaries (16 % / 84 % of the face).
    // These coincide with the geometric hover-highlight threshold (±0.68·half in cube
    // units → UV 0.16 / 0.84), and — since the pattern is symmetric — with the pick
    // partition regardless of the per-face UV winding.
    let low = ((1.0 - raycast::VIEW_CUBE_CENTRE_PATCH_FRACTION) * 0.5 * size as f32) as usize;
    let high = size - 1 - low;
    for c in 0..size {
        put(&mut pixels, low, c, slice); //  vertical, left boundary
        put(&mut pixels, high, c, slice); // vertical, right boundary
        put(&mut pixels, c, low, slice); //  horizontal, top boundary
        put(&mut pixels, c, high, slice); // horizontal, bottom boundary
    }

    // Monospace label, sized to sit inside the 68 % centre patch.
    draw_centered_label(&mut pixels, size, label, text);
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
    // Choose an integer scale that keeps the label inside the 68 % centre patch
    // (~60% of the face width, ~34% of its height), clear of the slice lines.
    let max_scale_w = (size * 6 / 10) / text_cells_wide.max(1);
    let max_scale_h = (size * 34 / 100) / glyph_height;
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
