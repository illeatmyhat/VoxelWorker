//! Transform gizmo (Milestone 5 origin gizmo, repurposed in issue #29 S2).

use super::*;

/// X axis colour `#d9603f` (sRGB hex → linear).
pub(crate) const GIZMO_AXIS_X_HEX: u32 = 0xd9_60_3f;
/// Y axis colour `#6fcf5f`.
pub(crate) const GIZMO_AXIS_Y_HEX: u32 = 0x6f_cf_5f;
/// Z axis colour `#5a8cff`.
pub(crate) const GIZMO_AXIS_Z_HEX: u32 = 0x5a_8c_ff;
/// Right-angle square colour `#bdb39a`.
const GIZMO_SQUARE_HEX: u32 = 0xbd_b3_9a;

/// The fraction of the viewport height a manipulator gizmo's axis spans — the one place the
/// screen-stable size of the transform gizmos (and every future W/E/R manipulator) is set.
/// Fed to [`OrbitCamera::screen_stable_model`](camera::OrbitCamera::screen_stable_model).
pub const GIZMO_SCREEN_FRACTION: f32 = 0.16;

/// The transform gizmo (issue #29 S2): three coloured axis lines and three
/// perpendicular square line-loops, drawn with **depth-test disabled** so it
/// shows through a solid model (correct manipulator behavior). Drawn in the MSAA
/// pass, after the voxels. The geometry is a fixed **unit** gizmo; it FOLLOWS the
/// selected node and holds a **fixed screen size** at any zoom, because the caller
/// bakes `translate(pivot) · scale(screen_stable_size)` into the model matrix
/// passed to [`update_uniforms`](Self::update_uniforms) (see
/// [`OrbitCamera::screen_stable_model`](camera::OrbitCamera::screen_stable_model)).
/// The axis-triad geometry is kept for now; full TRS handles are future work.
pub struct TransformGizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl TransformGizmoRenderer {
    /// Create the transform gizmo renderer for a colour target format. The geometry is a
    /// **unit** gizmo (unit axis length); its on-screen size is set per-frame by the `scale`
    /// [`update_uniforms`](Self::update_uniforms) bakes into the model matrix, so the gizmo
    /// holds a fixed screen size at any zoom.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let vertices = gizmo_vertices();
        let vertex_count = vertices.len() as u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo line vertices"),
            contents: bytemuck::cast_slice(&vertices),
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
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Upload the camera matrix with the gizmo's `model` baked in: the shader does
    /// `view_projection · position`, so feeding `view_projection · model` here places and
    /// sizes the unit gizmo WITHOUT touching the shared `LineUniforms` layout. `model` is the
    /// screen-stable `translate(pivot) · scale(size)` the caller builds from
    /// [`OrbitCamera::screen_stable_model`](camera::OrbitCamera::screen_stable_model); `pivot`
    /// is in the SAME recentred frame as the voxels, so the gizmo sits on the object.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        model: glam::Mat4,
    ) {
        let uniforms = LineUniforms {
            view_projection: (view_projection * model).to_cols_array_2d(),
            depth_bias: [0.0; 4],
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

/// Build the gizmo line vertices (axes + perpendicular squares) in **unit** space: the axes
/// run one unit from the origin, and the caller's model matrix scales the whole gizmo to a
/// screen-stable world size.
fn gizmo_vertices() -> Vec<LineVertex> {
    let axis_length = 1.0;
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
