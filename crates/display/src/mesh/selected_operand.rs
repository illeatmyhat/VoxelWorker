//! Boolean-operand ghost renderer (ADR 0018 Decision 6) — the x-ray of the selected
//! subtree's boolean operands.
//!
//! In Show-booleans mode the shell derives each boolean operand body in the selected
//! subtree (resolved standalone, `AppCore::boolean_operand_ghost`) and hands it here as
//! two-layer chunks + an operation style. This renderer meshes each body through the SAME
//! two-layer
//! cuboid mesher the solid path uses, then draws the mesh TWICE per frame with the
//! `cuboid.wgsl` ghost branch (the ADR 0012 H1 onion-ghost precedent) — the owner-decided
//! **two-pass depth split**:
//!
//! * **QUIET pass** — depth test `LessEqual`, the fragments ON or IN FRONT of the scene's
//!   rendered surface (the directly visible operand surface, including a cutter's exposed
//!   carve faces, which coincide exactly with the scene's cut surface).
//! * **LOUD pass** — depth test `Greater`, the fragments BEHIND the scene's surface (the
//!   operand body buried inside solid geometry), at noticeably higher opacity. An
//!   entirely-internal cutter therefore renders wholly loud — deliberately more obvious
//!   than Fusion's invisible internal voids.
//!
//! Neither pass writes depth (the ghost occludes nothing; unlike the onion ghost, whose
//! nearest-surface-wins rendering needs the write, the two passes here partition fragments
//! by the SOLID's depth, so self-overdraw within a pass is the accepted translucent
//! accumulation). Both passes carry a small toward-the-viewer depth bias so a ghost face
//! COINCIDENT with the scene surface (the carve-face case, where another mesher's
//! triangulation of the same plane may interpolate depth a ULP apart) classifies robustly
//! as quiet, never as loud and never dropped.
//!
//! ADR 0018 Decision 6: in "Show booleans" mode the shell feeds this renderer the
//! boolean-operand bodies of the selected subtree (`AppCore::boolean_operand_ghost`) —
//! one renderer instance, drawn over both display paths.
//!
//! The mesh is rebuilt only on selection/geometry change (`rebuild`), never per frame;
//! `update_uniforms` per frame writes only the camera + tints. Drawn as a raster overlay
//! inside the shared MSAA pass AFTER the solid draw, so it composes over BOTH display
//! paths (the cuboid mesh and the brick raymarch — the raymarch writes ray-hit depth into
//! the same attachment).

use super::*;
use crate::renderer::{operand_ghost_loud_tint, operand_ghost_quiet_tint, OperandGhostStyle};

/// One ghost body: an operation style plus the body's two-layer covering chunks, ALREADY
/// in the composed scene's absolute chunk coords (the app_core derivation resolves the
/// selected subtree standalone but keeps its absolute placement, so meshing with the
/// COMPOSED scene's recentre lands the ghost exactly on the node's voxels — ADR 0008,
/// carry the frame, never re-derive it). A plain selection is one body; a fixture-
/// instance selection is one body per spliced child (each under its own operation).
pub struct SelectedOperandGhostBody {
    /// How the body folds — picks the ghost hue (red / amber / subtle).
    pub style: OperandGhostStyle,
    /// The body's covering chunks, from the two-layer evaluator over the selection slice.
    pub chunks: Vec<([i32; 3], Arc<TwoLayerChunk>)>,
}

/// One uploaded ghost body: its per-chunk buffers (sorted by coord for a deterministic
/// draw order) plus the two per-pass uniform buffers (quiet / loud tints).
struct OperandGhostBodyBuffers {
    style: OperandGhostStyle,
    chunk_buffers: Vec<CuboidChunkBuffers>,
    quiet_uniform_buffer: wgpu::Buffer,
    quiet_bind_group: wgpu::BindGroup,
    loud_uniform_buffer: wgpu::Buffer,
    loud_bind_group: wgpu::BindGroup,
}

/// Toward-the-viewer depth bias for BOTH ghost passes (constant + slope-scaled, in the
/// standard negative-is-nearer convention — the depth buffer clears to 1.0 and the solid
/// tests `Less`). The two passes share ONE bias so they compare the SAME biased depth:
/// `LessEqual` and `Greater` then partition every fragment exactly (coincident surfaces
/// land in the quiet pass), and no fragment is ever double-shaded or dropped.
const OPERAND_GHOST_DEPTH_BIAS_CONSTANT: i32 = -2;
const OPERAND_GHOST_DEPTH_BIAS_SLOPE_SCALE: f32 = -2.0;

/// GPU resources for the selected-operand ghost overlay (issue #78). Owned by the shell
/// beside the other overlay renderers; self-gating (`draw` is a no-op with no selection).
pub struct SelectedOperandGhostRenderer {
    /// The QUIET pass pipeline: `cuboid.wgsl` ghost branch, alpha-blended, depth test
    /// `LessEqual`, depth write OFF.
    quiet_pipeline: wgpu::RenderPipeline,
    /// The LOUD pass pipeline: identical but depth test `Greater` (the x-ray half).
    loud_pipeline: wgpu::RenderPipeline,
    /// Group(0) layout, retained so `rebuild` can build per-body uniform bind groups.
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    /// Group(1): the pipeline layout wants the atlas slot bound, but the ghost branch
    /// returns its flat tint before any sample — a 1×1 placeholder texture satisfies it.
    unsampled_atlas_bind_group: wgpu::BindGroup,
    /// Group(2): the per-draw overlay-active uniform, bound at the OFF slot (offset 0) —
    /// the ghost ignores the on-face grid.
    overlay_bind_group: wgpu::BindGroup,
    /// The uploaded ghost bodies (empty = no selection / selection has no body).
    bodies: Vec<OperandGhostBodyBuffers>,
    /// The composed scene's voxel dims + density the meshes were built against, echoed
    /// into the per-frame uniforms (the vertex stage's corner-anchoring scalars).
    grid_dimensions: [u32; 3],
    voxels_per_block: u32,
}

impl SelectedOperandGhostRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let uniform_bind_group_layout = cuboid_uniform_bind_group_layout(device);
        let atlas_bind_group_layout = build_atlas_bind_group_layout(device);
        let unsampled_atlas_bind_group =
            build_unsampled_atlas_bind_group(device, queue, &atlas_bind_group_layout);
        let (overlay_bind_group, _overlay_stride) =
            build_overlay_bind_group(device, &overlay_bind_group_layout(device));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("selected-operand ghost shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shaders::with_grid_overlay(include_str!("../shaders/cuboid.wgsl")).into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("selected-operand ghost pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&atlas_bind_group_layout),
                Some(&overlay_bind_group_layout(device)),
            ],
            immediate_size: 0,
        });

        // The cuboid vertex layout (world position + face normal + material id), matching
        // the meshes `build_two_layer_chunk_meshes` emits — same as the solid pipelines.
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

        // The two depth-split pipelines differ ONLY in the depth compare (issue #78).
        let build_ghost_pipeline = |label: &str, depth_compare: wgpu::CompareFunction| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex_main"),
                    buffers: std::slice::from_ref(&vertex_layout),
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
                    cull_mode: Some(wgpu::Face::Back),
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // NO depth write on either pass (unlike the onion ghost): the ghost
                    // occludes nothing, and the quiet/loud partition comes purely from
                    // the SOLID's depth already in the attachment.
                    depth_write_enabled: Some(false),
                    depth_compare: Some(depth_compare),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState {
                        constant: OPERAND_GHOST_DEPTH_BIAS_CONSTANT,
                        slope_scale: OPERAND_GHOST_DEPTH_BIAS_SLOPE_SCALE,
                        clamp: 0.0,
                    },
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
        let quiet_pipeline = build_ghost_pipeline(
            "selected-operand ghost quiet pipeline",
            wgpu::CompareFunction::LessEqual,
        );
        let loud_pipeline = build_ghost_pipeline(
            "selected-operand ghost loud pipeline",
            wgpu::CompareFunction::Greater,
        );

        Self {
            quiet_pipeline,
            loud_pipeline,
            uniform_bind_group_layout,
            unsampled_atlas_bind_group,
            overlay_bind_group,
            bodies: Vec::new(),
            grid_dimensions: [0; 3],
            voxels_per_block: 1,
        }
    }

    // `has_bodies` was DELETED 2026-07-18 with zero callers — a residency probe that was
    // never wired into the shell. `draw` already no-ops on an empty body list, so a caller
    // has nothing to gate on.

    /// Drop every ghost body (the selection cleared / resolves to no geometry).
    pub fn clear(&mut self) {
        self.bodies.clear();
    }

    /// (Re)build the ghost meshes for a fresh selection derivation. Called ONLY on
    /// selection/geometry change, never per frame. Each body is meshed by the SAME
    /// two-layer cuboid mesher the solid path uses, at the FULL band, against the
    /// COMPOSED scene's `recentre` — so the ghost lands voxel-exact on the selected
    /// node's place in the render frame (ADR 0008: the frame is carried in, never
    /// re-derived from the slice). `grid_dimensions` is the composed scene's voxel
    /// extent (the corner-anchoring scalar the shader echoes).
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        bodies: &[SelectedOperandGhostBody],
        grid_dimensions: [u32; 3],
        recentre: RecentreVoxels,
        voxels_per_block: u32,
    ) {
        self.bodies.clear();
        self.grid_dimensions = grid_dimensions;
        self.voxels_per_block = voxels_per_block.max(1);
        for body in bodies {
            let meshes = build_two_layer_chunk_meshes(
                &body.chunks,
                grid_dimensions,
                recentre,
                voxels_per_block,
                LayerBand::FULL,
                None,
            );
            let mut buffers_by_coord = upload_chunk_meshes(device, &meshes);
            if buffers_by_coord.is_empty() {
                continue;
            }
            // Sorted coord order: the ghost writes no depth, so a stable draw order keeps
            // the alpha-blend result deterministic across runs (the golden gate).
            let mut coords: Vec<[i32; 3]> = buffers_by_coord.keys().copied().collect();
            coords.sort_unstable();
            let chunk_buffers = coords
                .into_iter()
                .filter_map(|coord| buffers_by_coord.remove(&coord))
                .collect();

            let make_pass_uniforms = |label: &str| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: std::mem::size_of::<CuboidUniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &self.uniform_bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                (buffer, bind_group)
            };
            let (quiet_uniform_buffer, quiet_bind_group) =
                make_pass_uniforms("selected-operand ghost quiet uniforms");
            let (loud_uniform_buffer, loud_bind_group) =
                make_pass_uniforms("selected-operand ghost loud uniforms");

            self.bodies.push(OperandGhostBodyBuffers {
                style: body.style,
                chunk_buffers,
                quiet_uniform_buffer,
                quiet_bind_group,
                loud_uniform_buffer,
                loud_bind_group,
            });
        }
    }

    /// Upload the per-frame camera matrix + per-pass tints into every body's quiet/loud
    /// uniform buffers. Cheap (two small writes per body); the meshes are untouched.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        for body in &self.bodies {
            let quiet = flat_ghost_uniforms(
                view_projection,
                self.grid_dimensions,
                self.voxels_per_block,
                operand_ghost_quiet_tint(body.style),
            );
            queue.write_buffer(&body.quiet_uniform_buffer, 0, bytemuck::bytes_of(&quiet));
            let loud = flat_ghost_uniforms(
                view_projection,
                self.grid_dimensions,
                self.voxels_per_block,
                operand_ghost_loud_tint(body.style),
            );
            queue.write_buffer(&body.loud_uniform_buffer, 0, bytemuck::bytes_of(&loud));
        }
    }

    /// Record the two ghost passes into an already-begun MSAA pass. MUST run AFTER the
    /// frame's solid voxel draw (mesh or brick — both leave their depth in the shared
    /// attachment, which is what splits quiet from loud). A no-op with no bodies.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.bodies.is_empty() {
            return;
        }
        // Group(1)/(2) are pass-invariant: the never-sampled atlas slot and the
        // overlay-OFF slot (dynamic offset 0).
        render_pass.set_bind_group(1, &self.unsampled_atlas_bind_group, &[]);
        render_pass.set_bind_group(2, &self.overlay_bind_group, &[0]);
        for body in &self.bodies {
            // Quiet then loud; the depth split makes the two passes touch DISJOINT
            // fragments, so the order between them is pixel-irrelevant — kept fixed
            // for deterministic command streams.
            for (pipeline, bind_group) in [
                (&self.quiet_pipeline, &body.quiet_bind_group),
                (&self.loud_pipeline, &body.loud_bind_group),
            ] {
                render_pass.set_pipeline(pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                for chunk in &body.chunk_buffers {
                    chunk.draw_all_runs(render_pass);
                }
            }
        }
    }
}

/// A 1×1 white RGBA texture + nearest sampler filling the atlas bind-group slot the
/// pipeline layout requires. The ghost branch never samples it (`cuboid.wgsl` returns the
/// flat tint first), so no real material atlas needs to be packed or resident here.
fn build_unsampled_atlas_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("selected-operand ghost placeholder atlas"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
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
        &[255u8, 255, 255, 255],
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("selected-operand ghost placeholder sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("selected-operand ghost placeholder atlas bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}
