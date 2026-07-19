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

/// The transform gizmo (issue #29 S2): three coloured axis lines and three
/// perpendicular square line-loops, drawn with **depth-test disabled** so it
/// shows through a solid model (correct manipulator behavior). Drawn in the MSAA pass, after the voxels. Unlike the old origin gizmo it
/// FOLLOWS the selected node: its pivot translation is baked into the uploaded
/// view-projection (`view_projection · translate(pivot)`) so it sits ON the
/// object, and it is sized from the selected node's own extent. The axis-triad
/// geometry is kept for now; full TRS handles are future work.
pub struct TransformGizmoRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl TransformGizmoRenderer {
    /// Create the transform gizmo renderer for a colour target format.
    /// `grid_dimensions` sizes the gizmo (`L = max(dims) * 0.62`); the caller
    /// rebuilds it to the SELECTED node's extent each frame.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        grid_dimensions: [u32; 3],
    ) -> Self {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        let vertex_capacity = vertex_count.max(1);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gizmo line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(vertices, vertex_capacity)),
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
            vertex_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Resize the gizmo to a freshly-resolved grid (matches the voxel rebuild).
    pub fn rebuild(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, grid_dimensions: [u32; 3]) {
        let vertices = gizmo_vertices(grid_dimensions);
        let vertex_count = vertices.len() as u32;
        if vertex_count <= self.vertex_capacity {
            if vertex_count > 0 {
                queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
            }
        } else {
            self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gizmo line vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
            self.vertex_capacity = vertex_count;
        }
        self.vertex_count = vertex_count;
    }

    /// Upload the camera matrix with the selected node's `pivot` translation baked
    /// in (issue #29 S2): the shader does `view_projection · position`, so feeding
    /// `view_projection · translate(pivot)` here moves the whole gizmo onto the
    /// selected node WITHOUT touching the shared `LineUniforms` layout. `pivot` is
    /// in the SAME recentred frame as the voxels, so the gizmo sits on the object.
    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        view_projection: glam::Mat4,
        pivot: glam::Vec3,
    ) {
        let model = glam::Mat4::from_translation(pivot);
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

/// Build the gizmo line vertices (axes + perpendicular squares), in world space.
fn gizmo_vertices(grid_dimensions: [u32; 3]) -> Vec<LineVertex> {
    let longest = grid_dimensions[0]
        .max(grid_dimensions[1])
        .max(grid_dimensions[2]) as f32;
    let axis_length = (longest * 0.62).max(1.0);
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
