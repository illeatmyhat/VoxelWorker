//! Points — the world reference grid (issue #29 S5).

use super::*;

/// Reference-plane line colour `#39414a` (issue #91 item 4): a desaturated near-neutral
/// slate from the Signal token family (the mock's faint `#2a2e33` ground strokes are the
/// visual target), replacing the old bright teal `#5fb8a4` that buried the bottom-left
/// status text. Used by the analytic infinite-grid shader.
pub(crate) const POINT_PLANE_COLOR_HEX: u32 = 0x39_41_4a;
/// Base alpha of a MINOR (per-VOXEL, spacing 1) analytic-grid line. Kept low so the
/// ground stays a quiet scaffold; the shader's per-tier LOD fade scales it toward the rim.
pub(crate) const POINT_PLANE_MINOR_ALPHA: f32 = 0.08;
/// Base alpha of a MAJOR (per-BLOCK, spacing = density) analytic-grid line — bolder than
/// the voxel lines so block boundaries still read, but the two-tier contrast is COMPRESSED
/// (issue #91 item 4) from the old 3× ratio so the field stays calm over the gradient.
pub(crate) const POINT_PLANE_MAJOR_ALPHA: f32 = 0.18;

/// Fraction of the viewport height each half-axis spans — the Point axes are a screen-stable
/// nav marker (ADR 0031), so they hold a constant on-screen size at any zoom instead of a fixed
/// world length that clips against the scene's near/far. Fed to
/// [`OrbitCamera::screen_stable_size`](camera::OrbitCamera::screen_stable_size).
const POINT_AXIS_SCREEN_FRACTION: f32 = 0.09;
/// Base alpha of a Point's axis lines.
pub(crate) const POINT_AXIS_ALPHA: f32 = 0.85;

/// Which reference plane a tiled grid lies in (issue #29 S5). The plane is spanned
/// by its two in-plane axes; the third (constant) axis is pinned at the Point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferencePlane {
    /// The FRONT plane (Z-up): spanned by X and Z; constant Y (normal +Y).
    Xz,
    /// The GROUND plane (Z-up): spanned by X and Y; constant Z (normal +Z).
    Xy,
    /// The side plane: spanned by Y and Z; constant X (normal +X).
    Yz,
}

/// Append a Point's coloured axis lines (issue #29 S5; per-axis fix) through
/// `origin_voxels` (the recentred render-frame position), reusing the gizmo axis
/// colours. `enabled[axis]` gates each axis independently (X = red +X, Y = green
/// +Y, Z = blue +Z), so e.g. turning Y off drops the green line and emits only the
/// X and Z segments. Each enabled axis spans `±half` world units — a screen-stable
/// length the caller derives per Point from the camera.
fn point_axes_into(
    vertices: &mut Vec<LineVertex>,
    origin_voxels: [f32; 3],
    half: f32,
    enabled: [bool; 3],
) {
    let colors = [
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_X_HEX), POINT_AXIS_ALPHA),
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Y_HEX), POINT_AXIS_ALPHA),
        with_alpha(srgb_hex_to_linear(GIZMO_AXIS_Z_HEX), POINT_AXIS_ALPHA),
    ];
    for axis in 0..3 {
        if !enabled[axis] {
            continue;
        }
        let mut from = origin_voxels;
        let mut to = origin_voxels;
        from[axis] = origin_voxels[axis] - half;
        to[axis] = origin_voxels[axis] + half;
        vertices.push(LineVertex { position: from, color: colors[axis] });
        vertices.push(LineVertex { position: to, color: colors[axis] });
    }
}

/// The recentred render-frame position (voxels) of a Point's origin (issue #29 S5):
/// `position_blocks·density − recentre`, the SAME frame the resolved voxels and the
/// per-object grids live in.
fn point_origin_voxels(point: &Point, recentre: RecentreVoxels, density: i64) -> [f32; 3] {
    // Unwrap the carried frame at this positional arithmetic (the recentre subtraction).
    let recentre = recentre.voxels();
    let mut origin = [0.0f32; 3];
    for axis in 0..3 {
        origin[axis] = (point.position_blocks[axis] * density - recentre[axis]) as f32;
    }
    origin
}

/// Build the AXIS line batch for every VISIBLE Point in `scene` (issue #29 S5),
/// gated CPU-side so it is unit-testable without a GPU. For each non-hidden Point
/// its enabled axes (X = red +X, Y = green +Y, Z = blue +Z) are emitted as three
/// coloured line segments through the Point's origin, in the recentred render frame.
///
/// Issue #29 Points fast-follow: the reference PLANES no longer live here — they are
/// drawn by [`InfiniteGridRenderer`] as an ANALYTIC infinite grid (a fullscreen
/// ray-plane shader), which fixes the old finite tiled quad's hard edge / near-clip
/// cutoff at shallow angles. This batch is now AXES-only (the axes were fine as
/// lines and stay unchanged). A hidden Point contributes nothing.
pub(crate) fn points_line_batch(
    scene: &Scene,
    voxels_per_block: u32,
    camera: &camera::OrbitCamera,
) -> Vec<LineVertex> {
    let mut vertices = Vec::new();
    let step = voxels_per_block.max(1);
    let density = step as i64;
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    for point in &scene.points {
        if point.hidden {
            continue;
        }
        let origin = point_origin_voxels(point, recentre, density);
        if point.axis_x || point.axis_y || point.axis_z {
            // Screen-stable half-length at this Point's depth (ADR 0031).
            let half = camera
                .screen_stable_size(glam::Vec3::from_array(origin), POINT_AXIS_SCREEN_FRACTION);
            point_axes_into(
                &mut vertices,
                origin,
                half,
                [point.axis_x, point.axis_y, point.axis_z],
            );
        }
    }
    vertices
}

/// One enabled reference PLANE of a visible Point (issue #29 Points fast-follow),
/// resolved into the recentred render frame for the analytic infinite-grid shader.
/// Computed CPU-side from the scene so the plane selection is unit-testable without
/// a GPU; [`InfiniteGridRenderer`] turns each into one fullscreen draw.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridPlaneInstance {
    /// The Point origin in the recentred render frame (voxels).
    pub origin: [f32; 3],
    /// The two in-plane unit axes spanning the plane (`u`, `v`).
    pub u_axis: [f32; 3],
    pub v_axis: [f32; 3],
    /// The plane normal (the pinned/constant world axis).
    pub normal: [f32; 3],
}

/// The unit basis (`u`, `v`, `normal`) for a [`ReferencePlane`]: the two in-plane
/// axes and the plane normal, in world coordinates.
fn reference_plane_basis(plane: ReferencePlane) -> ([f32; 3], [f32; 3], [f32; 3]) {
    match plane {
        // Front (Z-up): spanned by X and Z, normal +Y.
        ReferencePlane::Xz => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        // GROUND (Z-up): spanned by X and Y, normal +Z.
        ReferencePlane::Xy => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        // Side: spanned by Y and Z, normal +X.
        ReferencePlane::Yz => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
    }
}

/// Collect every enabled reference PLANE of every VISIBLE Point (issue #29 Points
/// fast-follow), in the recentred render frame, for the analytic infinite-grid pass.
/// Hidden Points and disabled planes contribute nothing; the common case (the
/// Origin Point's XY ground plane, Z-up) yields exactly one instance. Pure + GPU-free
/// so the plane selection/orientation is unit-tested.
pub fn enabled_grid_planes(scene: &Scene, voxels_per_block: u32) -> Vec<GridPlaneInstance> {
    let step = voxels_per_block.max(1);
    let density = step as i64;
    let recentre = scene.recentre_voxels_for_resolve(voxels_per_block);
    let mut planes = Vec::new();
    for point in &scene.points {
        if point.hidden {
            continue;
        }
        let origin = point_origin_voxels(point, recentre, density);
        let mut push = |plane: ReferencePlane| {
            let (u_axis, v_axis, normal) = reference_plane_basis(plane);
            planes.push(GridPlaneInstance { origin, u_axis, v_axis, normal });
        };
        if point.plane_xz {
            push(ReferencePlane::Xz);
        }
        if point.plane_xy {
            push(ReferencePlane::Xy);
        }
        if point.plane_yz {
            push(ReferencePlane::Yz);
        }
    }
    planes
}

/// The world reference AXES (issue #29 S5): every visible [`Point`]'s axis lines, batched into
/// one alpha-blended line buffer. Since ADR 0031 the axes are a **screen-stable nav marker** —
/// each half-axis spans a fixed fraction of the viewport ([`POINT_AXIS_SCREEN_FRACTION`]) at any
/// zoom — drawn ON TOP by default (depth off, through the model) with the option to occlude
/// (depth-tested), selected per frame by [`rebuild_from_scene`](Self::rebuild_from_scene).
///
/// Issue #29 Points fast-follow: the reference PLANES moved to [`InfiniteGridRenderer`] (an
/// analytic infinite grid); this renderer draws AXES only. Each frame the caller rebuilds the
/// batch from `scene.points` via [`Self::rebuild_from_scene`], then uploads the camera matrix.
/// With no visible Point (all hidden / axes off) the batch is empty and [`Self::draw`] is a no-op.
pub struct PointsRenderer {
    /// Depth-tested pipeline — the near occluded case: opaque geometry occludes the axes by depth
    /// (ADR 0031, scaffold phase).
    pipeline: wgpu::RenderPipeline,
    /// Depth-OFF pipeline — the on-top nav marker AND the far-distance paint-order fallback (both
    /// draw without depth-testing; the phase they sit in decides whether geometry paints over).
    pipeline_off: wgpu::RenderPipeline,
    /// Whether this frame depth-tests (near occluded) or not (on-top / far fallback). Set by
    /// [`rebuild_from_scene`](Self::rebuild_from_scene).
    depth_tested: bool,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    vertex_capacity: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl PointsRenderer {
    /// Create the Points renderer for a colour target. The batch starts empty — the
    /// caller fills it each frame from the visible Points via [`Self::rebuild_from_scene`].
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let vertex_capacity = 1u32;
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("points line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), vertex_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("points uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "points");

        // Two pipelines: depth-tested (occluded scaffold) and depth-OFF (on-top nav marker,
        // the default). The caller picks per frame via `on_top` (ADR 0031).
        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "points",
            true,
            MSAA_SAMPLE_COUNT,
        );
        let pipeline_off = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "points depth-off",
            false,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            pipeline_off,
            depth_tested: false,
            vertex_buffer,
            vertex_count: 0,
            vertex_capacity,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Rebuild this frame's Point AXIS line batch by walking `scene.points` (issue
    /// #29 S5). Hidden Points and disabled axes contribute nothing; an all-off scene
    /// yields an empty batch (the draw becomes a no-op). The reference planes are
    /// drawn separately by [`InfiniteGridRenderer`].
    pub fn rebuild_from_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
        camera: &camera::OrbitCamera,
        depth_tested: bool,
    ) {
        self.depth_tested = depth_tested;
        let vertices = points_line_batch(scene, voxels_per_block, camera);
        self.vertex_count = upload_lines(
            device,
            queue,
            &mut self.vertex_buffer,
            &mut self.vertex_capacity,
            vertices,
            "points line vertices",
        );
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass). Points
    /// use no depth bias (only the floor grid does — issue #29 fix).
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let uniforms = LineUniforms {
            view_projection: view_projection.to_cols_array_2d(),
            depth_bias: [0.0; 4],
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Record the Points draw into an already-begun (MSAA) pass. Self-gating: an
    /// empty batch (no visible Point) draws nothing.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.vertex_count == 0 {
            return;
        }
        let pipeline = if self.depth_tested {
            &self.pipeline
        } else {
            &self.pipeline_off
        };
        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertex_count, 0..1);
    }
}
