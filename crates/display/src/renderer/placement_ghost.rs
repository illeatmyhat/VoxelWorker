//! The placement ghost renderer (ADR 0022): a translucent analytic SDF drawn where an
//! armed primitive's voxels WILL land — "nothing recomposes during the gesture, render a
//! coloured transparent SDF where the voxels will be".
//!
//! A fullscreen sphere-trace of the parametric field (the `InfiniteGridRenderer`
//! precedent: fullscreen triangle, one uniform, no vertex buffers), drawn INSIDE the
//! existing MSAA voxel pass so it composites over whichever voxel display path took the
//! frame and writes `@builtin(frag_depth)` so the solid voxels occlude it where they are
//! in front. The shader is a hand-written mirror of `voxel_core::voxel::signed_distance`,
//! promoted verbatim from the parity-proven spike (`docs/design/wgsl-sdf-spike.md`).

use super::*;

/// std140 uniform for the placement ghost; field order matches `PlacementGhostUniforms`
/// in `placement_ghost.wgsl` **byte-for-byte** (the mirror the parity probe checks).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct PlacementGhostUniforms {
    view_projection: [[f32; 4]; 4],
    /// The RAY-FRAME unprojection matrix (`SceneMatrices::ray_unprojection`), inverted — the
    /// wide-baseline-precise unproject the shader casts through (a06d215). `view_projection`
    /// still reprojects the hit for depth (forward projection is immune).
    ray_inverse_unprojection: [[f32; 4]; 4],
    /// The ray frame's render-frame origin (`SceneMatrices::ray_eye`): the eye under perspective,
    /// zero under ortho. xyz; `w` padding. Added back to the eye-relative unprojected ray.
    ray_eye: [f32; 4],
    /// The central 3D viewport rect in physical pixels (x, y, width, height).
    viewport: [f32; 4],
    /// xyz: the shape's field centre in the world/render frame. w: the `ShapeKind`
    /// discriminant.
    center_and_kind: [f32; 4],
    /// xyz: the inscribed semi-axes in voxels (`grid/2` per axis). w: `wall_blocks *
    /// density` in voxels (Tube only).
    semi_axes_and_wall: [f32; 4],
    /// Linear RGB tint + source alpha for the translucent shell.
    tint: [f32; 4],
    /// x: iso level (`SURFACE_ISOLEVEL`). y: shade flag (1 = display). z/w: value-probe
    /// only, unused by the display pass.
    params: [f32; 4],
    /// The INVERSE lattice orientation (ADR 0026), column-major, as three `vec4` columns
    /// (std140 `mat3x3` stride; the `w` lane is padding). The shader maps a world sample back
    /// into the shape's un-turned SDF frame with this, so a side-placed cylinder's ghost lies
    /// on its side. Identity for an upright drop.
    orientation_inverse: [[f32; 4]; 3],
}

/// The `ShapeKind` discriminant the shader switches on. **MUST match `ShapeKind`'s
/// declaration order** in `voxel_core::voxel` (0 Cylinder, 1 Tube, 2 Sphere, 3 Torus,
/// 4 Box) — the one place a hand-written mirror drifts without any distance ever being
/// wrong. The exhaustive `match` makes a new variant a compile error here.
pub fn placement_ghost_shape_discriminant(kind: voxel_core::voxel::ShapeKind) -> u32 {
    use voxel_core::voxel::ShapeKind;
    match kind {
        ShapeKind::Cylinder => 0,
        ShapeKind::Tube => 1,
        ShapeKind::Sphere => 2,
        ShapeKind::Torus => 3,
        ShapeKind::Box => 4,
    }
}

/// The default translucent tint of the placement ghost — a cyan distinct from every
/// procedural material, so "this is a preview, not committed geometry" reads at a glance.
/// Linear RGB + source alpha.
pub const PLACEMENT_GHOST_TINT: [f32; 4] = [0.32, 0.78, 0.92, 0.55];

/// The analytic placement-ghost overlay: it draws ONE fullscreen triangle whose fragment
/// sphere-traces the armed primitive's field and writes `@builtin(frag_depth)`, so the
/// voxels drawn earlier in the SAME MSAA pass occlude it wherever they are in front.
///
/// The renderer is deliberately dumb: the frame math (`center_world = world_offset +
/// grid/2 - recentre`, ADR 0008) lives in the CALLER, which passes a resolved
/// `center_world` — passing the shape's raw parameters and letting the shader re-derive
/// its placement is exactly the silent frame-error mode this split prevents.
///
/// Self-gating: [`draw`](Self::draw) is a no-op until [`update_uniforms`](Self::update_uniforms)
/// arms it, so the frame path can hold the renderer permanently and gate visibility by
/// whether the caller uploaded a ghost this frame.
pub struct PlacementGhostRenderer {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Whether a ghost was uploaded this frame. `false` after `new`; set by
    /// `update_uniforms`, cleared by `disarm`.
    armed: bool,
}

impl PlacementGhostRenderer {
    /// Create the placement-ghost renderer for a colour target. It starts DISARMED —
    /// the caller arms it each frame via [`Self::update_uniforms`] when a tool is armed.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("placement ghost shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/placement_ghost.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("placement ghost bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<PlacementGhostUniforms>() as u64,
                    ),
                },
                count: None,
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("placement ghost uniforms"),
            size: std::mem::size_of::<PlacementGhostUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("placement ghost bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("placement ghost pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("placement ghost pipeline"),
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
                    // The shader outputs PREMULTIPLIED colour (`tint.rgb * lit * alpha`).
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
            // Drawn INSIDE the MSAA pass. Depth compare `Always` — the ghost is a PLACEMENT
            // AFFORDANCE, not scene geometry: it previews what will drop under the cursor,
            // so it must show in full even where it overlaps the object it is being placed
            // ON. Depth-testing it `LessEqual` against the voxels let that object's silhouette
            // clip the ghost (the owner's "cut off at an upper corner over another object"),
            // hiding exactly the contact footprint the user is judging. Depth WRITE stays off
            // — a translucent shell must never occlude real geometry, and with no write the
            // ghost still draws over whatever is behind it without corrupting the depth buffer.
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

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            armed: false,
        }
    }

    /// Arm and upload this frame's ghost. `view_projection` / `ray_inverse_unprojection` /
    /// `ray_eye` are the SAME ray-frame values the voxel pass used (`SceneMatrices`), so the
    /// analytic ray and the voxel ray are the same ray — and the unprojection stays precise at a
    /// wide-baseline recentre (a06d215).
    ///
    /// `center_world` is the field centre in the display's render frame — the caller
    /// resolves it via the frame law (`world_offset + grid/2 - recentre`, ADR 0008);
    /// `semi_axes` are the inscribed half-extents in voxels (`grid/2` per axis, EXACT
    /// half); `wall_voxels` is `wall_blocks * density` (Tube only); `tint` is linear RGB +
    /// source alpha.
    #[allow(clippy::too_many_arguments)]
    pub fn update_uniforms(
        &mut self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        ray_inverse_unprojection: glam::Mat4,
        ray_eye: glam::Vec3,
        viewport_px: [u32; 4],
        center_world: glam::Vec3,
        shape_kind: voxel_core::voxel::ShapeKind,
        semi_axes: glam::Vec3,
        wall_voxels: f32,
        tint: [f32; 4],
        orientation_inverse: [[f32; 4]; 3],
    ) {
        let uniforms = PlacementGhostUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            ray_inverse_unprojection: ray_inverse_unprojection.to_cols_array_2d(),
            ray_eye: [ray_eye.x, ray_eye.y, ray_eye.z, 0.0],
            viewport: [
                viewport_px[0] as f32,
                viewport_px[1] as f32,
                viewport_px[2] as f32,
                viewport_px[3] as f32,
            ],
            center_and_kind: [
                center_world.x,
                center_world.y,
                center_world.z,
                placement_ghost_shape_discriminant(shape_kind) as f32,
            ],
            semi_axes_and_wall: [semi_axes.x, semi_axes.y, semi_axes.z, wall_voxels],
            tint,
            params: [voxel_core::voxel::SURFACE_ISOLEVEL, 1.0, 0.0, 0.0],
            orientation_inverse,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        self.armed = true;
    }

    /// Disarm the ghost (a later frame with no armed tool). [`draw`](Self::draw) becomes a
    /// no-op again until the next [`update_uniforms`](Self::update_uniforms).
    pub fn disarm(&mut self) {
        self.armed = false;
    }

    /// Record the ghost draw into an already-begun (MSAA) pass: one fullscreen triangle.
    /// Self-gating — nothing is drawn until [`update_uniforms`](Self::update_uniforms)
    /// arms it.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if !self.armed {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::voxel::ShapeKind;

    /// The discriminant order the WGSL mirror switches on MUST match `ShapeKind`'s
    /// declaration order — the one place this hand-written mirror drifts without any
    /// distance ever being wrong (the spike's discriminant-order guard, lifted).
    #[test]
    fn discriminant_order_matches_shape_kind_declaration() {
        assert_eq!(placement_ghost_shape_discriminant(ShapeKind::Cylinder), 0);
        assert_eq!(placement_ghost_shape_discriminant(ShapeKind::Tube), 1);
        assert_eq!(placement_ghost_shape_discriminant(ShapeKind::Sphere), 2);
        assert_eq!(placement_ghost_shape_discriminant(ShapeKind::Torus), 3);
        assert_eq!(placement_ghost_shape_discriminant(ShapeKind::Box), 4);
    }

    /// The Rust twin's size is a multiple of 16 bytes (std140 uniform alignment) and
    /// matches the blocks the WGSL struct declares: two mat4 (128) + six vec4 (ray_eye,
    /// viewport, center_and_kind, semi_axes_and_wall, tint, params = 96) + a mat3x3 (three padded
    /// vec4 columns = 48, the ADR 0026 inverse orientation) = 272 bytes.
    #[test]
    fn uniform_layout_is_std140_sized() {
        assert_eq!(std::mem::size_of::<PlacementGhostUniforms>(), 272);
        assert_eq!(std::mem::size_of::<PlacementGhostUniforms>() % 16, 0);
    }
}
