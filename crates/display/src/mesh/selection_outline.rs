//! Selection outline + wash (ADR 0032, owner-reworked 2026-07-29) — screen-space
//! selection feedback from a depth map of the selected bodies.
//!
//! The shipped first take re-drew each selected body as a depth-biased cel mesh
//! (`ghost_mode = 2`); the owner rejected it — the bias popped where the standalone
//! body disagreed with the composed surface, and the "rim" flipped whole voxel faces.
//! This replacement never re-shades geometry. Two passes:
//!
//! 1. **G-buffer** — the selected bodies' meshes (same two-layer cuboid mesher, same
//!    `SceneMatrices::view_projection` as the solid pass) rasterised into TWO private
//!    non-MSAA hull map pairs, each a `Depth32Float` depth + an `Rgba16Float` normal
//!    (outward face normal, coverage in alpha): the FRONT hull (Less, back faces
//!    culled) and the BACK hull (Greater, front faces culled). Together the depths
//!    bound the selection's depth interval per pixel; each normal map then folds
//!    down a full mip chain of coverage-weighted MEAN normals (the crease pass's
//!    block-relative field). One map pair for the whole selection: bodies union in
//!    the depth tests, so multi-select is one pass pair, never per-body.
//! 2. **Composite** — a full-screen pass on the RESOLVED colour target (after the
//!    voxel MSAA pass, before the view cube) tests the scene's MSAA depth against
//!    that interval. A pixel is WASHED where ANY scene sample lies inside
//!    `[front − ε, back + ε]` — the visible surface is ON or IN the selected body,
//!    which is what makes a carved concavity wash under both the cutter that carved
//!    it (its inside wall — all back faces, invisible to a front hull alone) and the
//!    host that owns it, while an occluded body washes nothing (owner law: never an
//!    x-ray). All four samples are tested because sample positions are intra-pixel
//!    offsets: on a steeply sloped face the sample-0 depth alone drifts past ε and
//!    speckles. ε is a half-voxel converted to NDC at the sampled depth
//!    (`NdcDepthMapping` — the compare stays in hardware-z space; linearising would
//!    amplify quantisation noise by `view_depth²/near`). A 1px OUTLINE lands just
//!    outside the washed region's boundary — one rule that hugs both the body's
//!    silhouette and the edge where it slips behind other geometry. Inside the wash,
//!    CREASE lines mark edges sharp RELATIVE TO A BLOCK (the wash alpha lifts to the
//!    outline's): the angle between normalised mean normals across a block-projected
//!    footprint gates a fine per-pixel angle test — see `selection_outline.wgsl`'s
//!    header for why the mean-normal ANGLE (not Toksvig |mean| alone) is what keeps
//!    voxelised rounding blunt.
//!
//! Mesh rebuilt only on selection/geometry change (`rebuild`); `update_uniforms` per
//! frame writes camera + mapping only; `prepare` re-sizes the map with the target.

use super::*;
use crate::renderer::selection_cel_tint;

/// One selected node's body: its covering chunks in the composed scene's absolute chunk
/// coords (ADR 0008 — frame carried in, never re-derived).
pub type SelectedBodyChunks = Vec<([i32; 3], Arc<TwoLayerChunk>)>;

/// Outline (silhouette ring) src alpha; the wash alpha is `selection_cel_tint()`'s.
const SELECTION_OUTLINE_ALPHA: f32 = 0.92;

/// Half a voxel in render-frame units (the render frame IS voxel units — the cuboid
/// vertex stage recovers absolute voxel coords by adding `grid_half_extent` directly).
/// The wash's world-space coincidence tolerance: the selected body's mesh and the
/// composed scene's surface describe the same voxel face, so they can only disagree
/// by strictly less than a voxel where they genuinely coincide.
const SELECTION_COINCIDENCE_HALF_WIDTH: f32 = 0.5;

/// A few `Depth32Float` ULPs: the epsilon's floor, so two triangulations of the same
/// plane (interpolated a bit apart) still read as coincident even where the analytic
/// slope epsilon collapses toward zero.
const SELECTION_EPSILON_FLOOR: f32 = 1e-6;

/// The hull normal maps' format: filterable + renderable, signed components for
/// the raw outward normals the mip chain averages.
const NORMAL_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// std140-safe composite uniforms; field order matches the WGSL
/// `SelectionOutlineUniforms` exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SelectionOutlineUniforms {
    /// Wash colour: the Signal accent (linear) + the resting wash alpha.
    tint: [f32; 4],
    outline_alpha: f32,
    half_voxel: f32,
    /// `NdcDepthMapping` — the projection z-row (see the camera crate's doc).
    depth_scale: f32,
    depth_offset: f32,
    orthographic: f32,
    epsilon_floor: f32,
    /// The projection's `y_axis.y` + the viewport height + the density — together
    /// the projected-block-size the crease pass sizes its footprint with.
    vertical_scale: f32,
    viewport_height_px: f32,
    voxels_per_block: f32,
    _pad: [f32; 3],
}

/// GPU resources for the selection outline + wash. Owned by the shell beside the
/// other overlay renderers; both draws self-gate (no selection body → no-op).
pub struct SelectionOutlineRenderer {
    /// Hull pipelines: `cuboid.wgsl`'s `vertex_main` + the `hull_normal_main`
    /// fragment (normal + coverage into the normal map), 1 sample, depth write on.
    /// Front = Less + back-cull; back = Greater + front-cull.
    front_hull_pipeline: wgpu::RenderPipeline,
    back_hull_pipeline: wgpu::RenderPipeline,
    /// Full-screen composite: `selection_outline.wgsl`, alpha-blended onto the
    /// resolved target, no depth attachment.
    composite_pipeline: wgpu::RenderPipeline,
    composite_bind_group_layout: wgpu::BindGroupLayout,
    /// One-level downsample of a normal map (`selection_normal_mip.wgsl`), run
    /// level-by-level after the hull passes to build the mean-normal mip chain.
    normal_mip_pipeline: wgpu::RenderPipeline,
    normal_mip_bind_group_layout: wgpu::BindGroupLayout,
    normal_sampler: wgpu::Sampler,
    gbuffer_uniform_buffer: wgpu::Buffer,
    gbuffer_uniform_bind_group: wgpu::BindGroup,
    composite_uniform_buffer: wgpu::Buffer,
    /// Rebuilt by `prepare` on target resize (it references the depth views).
    composite_bind_group: Option<wgpu::BindGroup>,
    /// The private hull depth maps, target-sized, recreated on resize.
    front_hull_view: Option<wgpu::TextureView>,
    back_hull_view: Option<wgpu::TextureView>,
    /// Per-mip-level views of the two normal maps (index = level), for the hull
    /// pass's level-0 attachment and the downsample chain's source/target pairs.
    front_normal_level_views: Vec<wgpu::TextureView>,
    back_normal_level_views: Vec<wgpu::TextureView>,
    /// Downsample source bind groups: index `i` binds level `i` as the source
    /// when rendering into level `i + 1`.
    front_normal_mip_bind_groups: Vec<wgpu::BindGroup>,
    back_normal_mip_bind_groups: Vec<wgpu::BindGroup>,
    target_size: (u32, u32),
    /// The uploaded chunk buffers of every selected body (empty = nothing selected /
    /// the selection has no body), sorted by coord within each body.
    chunk_buffers: Vec<CuboidChunkBuffers>,
    /// The composed scene's voxel dims + density the meshes were built against, echoed
    /// into the per-frame uniforms (the vertex stage's corner-anchoring scalars).
    grid_dimensions: [u32; 3],
    voxels_per_block: u32,
}

impl SelectionOutlineRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let uniform_bind_group_layout = cuboid_uniform_bind_group_layout(device);

        let cuboid_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection-outline gbuffer shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shaders::with_shared_shading(include_str!("../shaders/cuboid.wgsl")).into(),
            ),
        });
        let gbuffer_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection-outline gbuffer pipeline layout"),
            // `vertex_main` statically uses only group(0); with no fragment stage the
            // atlas/overlay groups never bind.
            bind_group_layouts: &[Some(&uniform_bind_group_layout)],
            immediate_size: 0,
        });

        // The cuboid vertex layout, matching `build_two_layer_chunk_meshes` output.
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CuboidVertex>() as u64,
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
                    format: wgpu::VertexFormat::Uint32,
                },
            ],
        };

        // The two hull rasterisations differ only in which faces survive and which
        // depth wins: the front hull keeps the NEAREST front face, the back hull the
        // FARTHEST back face — together the selection's depth interval per pixel.
        // Alongside each depth the winning fragment's outward normal lands in the
        // hull's normal map (coverage in alpha) for the crease pass's mip chain.
        let hull_pipeline = |label: &str, cull: wgpu::Face, compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&gbuffer_layout),
                vertex: wgpu::VertexState {
                    module: &cuboid_shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &cuboid_shader,
                    entry_point: Some("hull_normal_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: NORMAL_MAP_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(cull),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let front_hull_pipeline = hull_pipeline(
            "selection-outline front-hull pipeline",
            wgpu::Face::Back,
            wgpu::CompareFunction::Less,
        );
        let back_hull_pipeline = hull_pipeline(
            "selection-outline back-hull pipeline",
            wgpu::Face::Front,
            wgpu::CompareFunction::Greater,
        );

        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection-outline composite shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/selection_outline.wgsl").into(),
            ),
        });
        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("selection-outline composite bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // The private front-hull depth map.
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // The private back-hull depth map.
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // The scene's MSAA depth attachment (all four samples tested).
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: true,
                        },
                        count: None,
                    },
                    // The two hull normal maps (full mip chains) + their sampler.
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });
        let composite_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection-outline composite pipeline layout"),
            bind_group_layouts: &[Some(&composite_bind_group_layout)],
            immediate_size: 0,
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection-outline composite pipeline"),
            layout: Some(&composite_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // The mean-normal mip chain: one trilinear clamp sampler shared by the
        // downsample and the composite's block-relative taps.
        let normal_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("selection-outline normal sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let normal_mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selection-outline normal mip shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/selection_normal_mip.wgsl").into(),
            ),
        });
        let normal_mip_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("selection-outline normal mip bind group layout"),
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
        let normal_mip_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selection-outline normal mip pipeline layout"),
            bind_group_layouts: &[Some(&normal_mip_bind_group_layout)],
            immediate_size: 0,
        });
        let normal_mip_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("selection-outline normal mip pipeline"),
            layout: Some(&normal_mip_layout),
            vertex: wgpu::VertexState {
                module: &normal_mip_shader,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &normal_mip_shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: NORMAL_MAP_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let gbuffer_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selection-outline gbuffer uniforms"),
            size: std::mem::size_of::<CuboidUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let gbuffer_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("selection-outline gbuffer uniforms"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gbuffer_uniform_buffer.as_entire_binding(),
            }],
        });
        let composite_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("selection-outline composite uniforms"),
            size: std::mem::size_of::<SelectionOutlineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            front_hull_pipeline,
            back_hull_pipeline,
            composite_pipeline,
            composite_bind_group_layout,
            normal_mip_pipeline,
            normal_mip_bind_group_layout,
            normal_sampler,
            gbuffer_uniform_buffer,
            gbuffer_uniform_bind_group,
            composite_uniform_buffer,
            composite_bind_group: None,
            front_hull_view: None,
            back_hull_view: None,
            front_normal_level_views: Vec::new(),
            back_normal_level_views: Vec::new(),
            front_normal_mip_bind_groups: Vec::new(),
            back_normal_mip_bind_groups: Vec::new(),
            target_size: (0, 0),
            chunk_buffers: Vec::new(),
            grid_dimensions: [0; 3],
            voxels_per_block: 1,
        }
    }

    /// (Re)create the target-sized hull depth maps + composite bind group. Keyed
    /// on the target size, matching the callers' own depth-view lifetime (both shells
    /// recreate `scene_depth_view` only on resize, so same-size staleness cannot
    /// occur). Cheap no-op when the size is unchanged.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        scene_depth_view: &wgpu::TextureView,
    ) {
        if self.target_size == (width, height) && self.composite_bind_group.is_some() {
            return;
        }
        self.target_size = (width, height);
        let hull_map = |label: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: width.max(1),
                    height: height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let front_hull_view = hull_map("selection-outline front-hull map");
        let back_hull_view = hull_map("selection-outline back-hull map");

        // The two mean-normal maps: full mip chains down to 1×1, one level view per
        // level (hull pass attaches level 0; the downsample renders level i from a
        // bind group over level i−1). The full-chain view feeds the composite.
        let extent_width = width.max(1);
        let extent_height = height.max(1);
        let mip_level_count = 32 - extent_width.max(extent_height).leading_zeros();
        let normal_map = |label: &str| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: extent_width,
                    height: extent_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: NORMAL_MAP_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let level_views: Vec<wgpu::TextureView> = (0..mip_level_count)
                .map(|level| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            let mip_bind_groups: Vec<wgpu::BindGroup> = level_views
                .iter()
                .take(mip_level_count as usize - 1)
                .map(|source_view| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("selection-outline normal mip bind group"),
                        layout: &self.normal_mip_bind_group_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(source_view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&self.normal_sampler),
                            },
                        ],
                    })
                })
                .collect();
            let full_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (level_views, mip_bind_groups, full_view)
        };
        let (front_normal_level_views, front_normal_mip_bind_groups, front_normal_full_view) =
            normal_map("selection-outline front normal map");
        let (back_normal_level_views, back_normal_mip_bind_groups, back_normal_full_view) =
            normal_map("selection-outline back normal map");

        self.composite_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("selection-outline composite bind group"),
            layout: &self.composite_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.composite_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&front_hull_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&back_hull_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(scene_depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&front_normal_full_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&back_normal_full_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.normal_sampler),
                },
            ],
        }));
        self.front_hull_view = Some(front_hull_view);
        self.back_hull_view = Some(back_hull_view);
        self.front_normal_level_views = front_normal_level_views;
        self.back_normal_level_views = back_normal_level_views;
        self.front_normal_mip_bind_groups = front_normal_mip_bind_groups;
        self.back_normal_mip_bind_groups = back_normal_mip_bind_groups;
    }

    /// Drop every body (the selection cleared / resolves to no geometry).
    pub fn clear(&mut self) {
        self.chunk_buffers.clear();
    }

    /// (Re)build the selection meshes for a fresh derivation. Called ONLY on
    /// selection/geometry change, never per frame. Each body is meshed by the SAME
    /// two-layer cuboid mesher the solid path uses, at the FULL band, against the
    /// COMPOSED scene's `recentre` — so the depth map lands voxel-exact on the
    /// selected node's place in the render frame (ADR 0008).
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        bodies: &[SelectedBodyChunks],
        grid_dimensions: [u32; 3],
        recentre: RecentreVoxels,
        voxels_per_block: u32,
    ) {
        self.chunk_buffers.clear();
        self.grid_dimensions = grid_dimensions;
        self.voxels_per_block = voxels_per_block.max(1);
        for chunks in bodies {
            let meshes = build_two_layer_chunk_meshes(
                chunks,
                grid_dimensions,
                recentre,
                voxels_per_block,
                LayerBand::FULL,
                None,
            );
            let mut buffers_by_coord = upload_chunk_meshes(device, &meshes);
            let mut coords: Vec<[i32; 3]> = buffers_by_coord.keys().copied().collect();
            coords.sort_unstable();
            self.chunk_buffers.extend(
                coords
                    .into_iter()
                    .filter_map(|coord| buffers_by_coord.remove(&coord)),
            );
        }
    }

    /// Upload the per-frame camera + depth mapping. `view_projection` MUST be the
    /// scene pass's own matrix ([`camera::SceneMatrices::view_projection`]) — the
    /// wash compares the two hardware depths directly, so the G-buffer has to record
    /// the identical projection; `ndc_depth` is that same frame's mapping.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        ndc_depth: camera::NdcDepthMapping,
        viewport_height_px: u32,
    ) {
        // The G-buffer runs `cuboid.wgsl`'s vertex stage alone, which reads only the
        // camera + corner-anchoring scalars from this block; the ghost fields are inert.
        let gbuffer_uniforms = flat_ghost_uniforms(
            view_projection,
            self.grid_dimensions,
            self.voxels_per_block,
            [0.0; 4],
        );
        queue.write_buffer(
            &self.gbuffer_uniform_buffer,
            0,
            bytemuck::bytes_of(&gbuffer_uniforms),
        );
        let tint = selection_cel_tint();
        let composite_uniforms = SelectionOutlineUniforms {
            tint,
            outline_alpha: SELECTION_OUTLINE_ALPHA,
            half_voxel: SELECTION_COINCIDENCE_HALF_WIDTH,
            depth_scale: ndc_depth.depth_scale,
            depth_offset: ndc_depth.depth_offset,
            orthographic: if ndc_depth.orthographic { 1.0 } else { 0.0 },
            epsilon_floor: SELECTION_EPSILON_FLOOR,
            vertical_scale: ndc_depth.vertical_scale,
            viewport_height_px: viewport_height_px.max(1) as f32,
            voxels_per_block: self.voxels_per_block as f32,
            _pad: [0.0; 3],
        };
        queue.write_buffer(
            &self.composite_uniform_buffer,
            0,
            bytemuck::bytes_of(&composite_uniforms),
        );
    }

    /// Record the two depth-only hull passes (their own passes, before the scene's
    /// MSAA pass). Each clear covers its WHOLE map (load ops ignore the scissor), so
    /// out-of-viewport texels read "no body" in the composite's neighbour taps.
    /// A no-op with no bodies — the composite is gated on the same condition, so a
    /// stale map is never read.
    pub fn draw_gbuffer(&self, encoder: &mut wgpu::CommandEncoder, viewport_px: [u32; 4]) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        let (Some(front_hull_view), Some(back_hull_view)) =
            (&self.front_hull_view, &self.back_hull_view)
        else {
            return;
        };
        // Front hull keeps the nearest surface toward the clear-at-far 1.0; back hull
        // keeps the farthest away from a clear-at-near 0.0 under Greater. The winning
        // fragment's normal lands in the hull's normal map level 0 alongside (cleared
        // to zero coverage over the whole attachment).
        let hulls = [
            (
                front_hull_view,
                &self.front_hull_pipeline,
                1.0f32,
                self.front_normal_level_views.first(),
            ),
            (
                back_hull_view,
                &self.back_hull_pipeline,
                0.0f32,
                self.back_normal_level_views.first(),
            ),
        ];
        for (view, pipeline, clear, normal_view) in hulls {
            let Some(normal_view) = normal_view else {
                return;
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("selection-outline hull pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: normal_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let [x, y, width, height] = viewport_px;
            pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, width, height);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.gbuffer_uniform_bind_group, &[]);
            for chunk in &self.chunk_buffers {
                chunk.draw_all_runs(&mut pass);
            }
        }
        // Build each normal map's mean-normal mip chain: level i renders from a
        // single bilinear tap over level i−1 (coverage-weighted — normals ride
        // premultiplied by alpha, so the chain averages surface normals without
        // bleeding the cleared background in).
        let chains = [
            (
                &self.front_normal_level_views,
                &self.front_normal_mip_bind_groups,
            ),
            (
                &self.back_normal_level_views,
                &self.back_normal_mip_bind_groups,
            ),
        ];
        for (level_views, mip_bind_groups) in chains {
            for (target_view, source_bind_group) in level_views.iter().skip(1).zip(mip_bind_groups)
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("selection-outline normal mip pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(&self.normal_mip_pipeline);
                pass.set_bind_group(0, source_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Record the composite pass onto the RESOLVED colour target (after the scene's
    /// MSAA pass has stored its depth, before the view cube). A no-op with no bodies.
    pub fn draw_composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        viewport_px: [u32; 4],
    ) {
        if self.chunk_buffers.is_empty() {
            return;
        }
        let Some(composite_bind_group) = &self.composite_bind_group else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("selection-outline composite pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let [x, y, width, height] = viewport_px;
        pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, width, height);
        pass.set_pipeline(&self.composite_pipeline);
        pass.set_bind_group(0, composite_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
