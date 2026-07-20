//! Block lattice + fine floor grid (Milestone 8) — prototype `buildGrids`.

use super::*;

/// Block lattice colour `#5fb8a4` (teal patina) at ~0.28 alpha.
const LATTICE_COLOR_HEX: u32 = 0x5f_b8_a4;
const LATTICE_ALPHA: f32 = 0.28;
/// Floor grid colour `#b8a47a` (warm sand) at 0.55 alpha. Issue #29 fix: the
/// floor grid was previously a very dim `#6b5f4a` at 0.16 alpha — coincident with
/// the model's depth-tested base plane and near-black against the background, so
/// it read as "nothing" when toggled on. A brighter colour at a lattice-comparable
/// opacity makes the base-plane grid clearly visible (it still hugs the node's
/// enclosing-block XZ footprint, snapped to the global block lattice).
const FLOOR_COLOR_HEX: u32 = 0xb8_a4_7a;
/// Alpha of a BOLD (block-edge) floor line — the major tier of the two-tier fine
/// floor grid (issue #29 fix). These lines sit at every block boundary and so
/// coincide exactly with the block lattice's vertical lines at the base plane.
const FLOOR_ALPHA: f32 = 0.55;
/// Alpha of a fine VOXEL-edge floor line — the minor tier (issue #29 fix). One
/// line per voxel boundary (step = 1) at a deliberately low opacity, so the floor
/// reads as a dense fine grid under the object without drowning the bold block
/// lines or the model. Mirrors the Point ground plane's minor/major two-tier
/// scheme (`POINT_PLANE_MINOR_ALPHA` vs `POINT_PLANE_MAJOR_ALPHA`).
const FLOOR_VOXEL_ALPHA: f32 = 0.16;

/// The per-object block lattice and floor grid (the prototype's `buildGrids`), drawn through the shared alpha-blended, depth-tested line
/// pipeline in the MSAA pass.
///
/// Issue #29 S3: this is no longer ONE whole-region lattice. Each frame the caller
/// walks the scene and, for every node whose grids are enabled (the scene master
/// ANDed with the node's own toggle), appends that node's block lattice and/or
/// floor lines into the renderer's per-frame batch via `Self::set_batch`. A
/// lattice box is a 3D box lattice with lines at every BLOCK boundary (spacing =
/// density) spanning the node's enclosing-block AABB; the floor is the horizontal
/// grid at the node's base plane, snapped to the same global block lines.
pub struct SceneGridRenderer {
    pipeline: wgpu::RenderPipeline,
    lattice_buffer: wgpu::Buffer,
    lattice_vertex_count: u32,
    lattice_capacity: u32,
    floor_buffer: wgpu::Buffer,
    floor_vertex_count: u32,
    floor_capacity: u32,
    /// Uniforms for the lattice draw — view-projection with ZERO depth bias.
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    /// Separate uniforms for the floor draw (issue #29 fix): the SAME
    /// view-projection but a NEGATIVE [`LineUniforms::depth_bias`], so the floor
    /// draws at the EXACT base plane `z = min[2]` (Z-up; meeting the lattice's bottom
    /// edges) yet wins the `Less` depth test against the model's coincident bottom
    /// face — no z-fight shimmer, no geometric vertical drop. (A hardware
    /// `DepthBiasState` is rejected by wgpu on `LineList`, so the bias is applied
    /// in the line shader via this uniform.)
    floor_uniform_buffer: wgpu::Buffer,
    floor_uniform_bind_group: wgpu::BindGroup,
}

/// The NDC depth bias (issue #29 fix) the floor grid uploads in its
/// [`LineUniforms::depth_bias`]: a small NEGATIVE offset pulls the floor lines a
/// hair toward the camera so they win the `Less` depth test against the model's
/// coincident bottom face. ~5e-4 in NDC is imperceptible spatially (far below the
/// old 0.25-voxel geometric drop) yet reliably resolves coincident depth on the
/// `Depth32Float` target.
const FLOOR_DEPTH_BIAS_NDC: f32 = -5.0e-4;

impl SceneGridRenderer {
    /// Create the renderer for a colour target. The line batches start empty —
    /// the caller fills them each frame via `Self::set_batch` from the visible
    /// nodes' enabled grids.
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let lattice_capacity = 1u32;
        let floor_capacity = 1u32;

        let lattice_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lattice line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), lattice_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let floor_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("floor line vertices"),
            contents: bytemuck::cast_slice(&pad_lines(Vec::new(), floor_capacity)),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lattice uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (uniform_bind_group_layout, uniform_bind_group) =
            line_uniform_bind_group(device, &uniform_buffer, "lattice");

        // A SECOND uniform buffer for the floor draw, carrying the same matrix with a
        // negative NDC depth bias (issue #29 fix) — wgpu rejects a hardware depth bias
        // on LineList, so the floor biases its depth in the line shader via this buffer.
        let floor_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("floor uniforms"),
            size: std::mem::size_of::<LineUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (_floor_layout, floor_uniform_bind_group) =
            line_uniform_bind_group(device, &floor_uniform_buffer, "floor");

        // Depth-tested (true) so the lattice/floor are occluded by the solid model
        // — they read as a scaffold around/under it, not an overlay on top. The floor
        // shares this pipeline; its depth bias comes from its uniform, not the pipeline.
        let pipeline = build_line_pipeline(
            device,
            color_format,
            &uniform_bind_group_layout,
            "lattice",
            true,
            MSAA_SAMPLE_COUNT,
        );

        Self {
            pipeline,
            lattice_buffer,
            lattice_vertex_count: 0,
            lattice_capacity,
            floor_buffer,
            floor_vertex_count: 0,
            floor_capacity,
            uniform_buffer,
            uniform_bind_group,
            floor_uniform_buffer,
            floor_uniform_bind_group,
        }
    }

    /// Rebuild this frame's lattice + floor line batches by walking `scene` (issue
    /// #29 S3). For every visible node whose grids are enabled — the scene-wide
    /// master ANDed with that node's own per-object toggle — the node's
    /// enclosing-block lattice box ([`Scene::node_block_lattice_box_recentred`]) is
    /// appended to the corresponding batch:
    ///
    /// * `master_block_lattice && node.grids.block_lattice` → block lattice lines.
    /// * `master_floor_grid && node.grids.floor_grid` → base-plane floor lines.
    ///
    /// A node with no intrinsic extent (size-less VoxelBody / empty subtree) yields no
    /// box and is skipped. When NOTHING is enabled both batches are empty and
    /// [`Self::draw`] becomes a no-op — the new default, where per-object grids are
    /// off until the user turns them on.
    pub fn rebuild_from_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        voxels_per_block: u32,
    ) {
        let step = voxels_per_block.max(1);
        let (lattice_boxes, floor_boxes) = scene_grid_boxes(scene, voxels_per_block);
        let mut lattice: Vec<LineVertex> = Vec::new();
        let mut floor: Vec<LineVertex> = Vec::new();
        for (min, max) in lattice_boxes {
            lattice_vertices_into(&mut lattice, min, max, step);
        }
        for (min, max) in floor_boxes {
            floor_vertices_into(&mut floor, min, max, step);
        }
        self.lattice_vertex_count = upload_lines(
            device,
            queue,
            &mut self.lattice_buffer,
            &mut self.lattice_capacity,
            lattice,
            "lattice line vertices",
        );
        self.floor_vertex_count = upload_lines(
            device,
            queue,
            &mut self.floor_buffer,
            &mut self.floor_capacity,
            floor,
            "floor line vertices",
        );
    }

    /// Upload the camera matrix (same `view_projection` as the voxel pass) to BOTH
    /// the lattice uniform (zero depth bias) and the floor uniform (a negative NDC
    /// [`FLOOR_DEPTH_BIAS_NDC`] depth bias — issue #29 fix), so the floor wins
    /// coincident depth against the model's base face without a geometric drop.
    pub fn update_uniforms(&self, queue: &wgpu::Queue, view_projection: glam::Mat4) {
        let view_projection = view_projection.to_cols_array_2d();
        let lattice = LineUniforms { view_projection, depth_bias: [0.0; 4] };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&lattice));
        let floor = LineUniforms {
            view_projection,
            depth_bias: [FLOOR_DEPTH_BIAS_NDC, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.floor_uniform_buffer, 0, bytemuck::bytes_of(&floor));
    }

    /// Record the lattice + floor draws into an already-begun (MSAA) pass. Gating
    /// is done at batch-build time (issue #29 S3): only grid-enabled nodes
    /// contributed lines, so empty batches simply draw nothing here. Both draws use
    /// the same line pipeline; the floor binds its own (depth-biased) uniform.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        if self.lattice_vertex_count == 0 && self.floor_vertex_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        if self.lattice_vertex_count > 0 {
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.lattice_buffer.slice(..));
            render_pass.draw(0..self.lattice_vertex_count, 0..1);
        }
        if self.floor_vertex_count > 0 {
            // Floor's own uniform carries the negative depth bias (issue #29 fix) so
            // the base-plane floor wins coincident depth against the model's bottom face.
            render_pass.set_bind_group(0, &self.floor_uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.floor_buffer.slice(..));
            render_pass.draw(0..self.floor_vertex_count, 0..1);
        }
    }
}

/// The per-object grid boxes for a scene (issue #29 S3), gated CPU-side so the walk
/// is unit-testable without a GPU. Returns `(lattice_boxes, floor_boxes)` where each
/// box is the `(min, max)` enclosing-block AABB (recentred voxels) of a node whose
/// grid is enabled — the scene-wide master ANDed with the node's own per-object
/// toggle. A node with no intrinsic extent contributes no box. When a master is off,
/// or a node's flag is off, that node contributes nothing to that batch (gating).
#[allow(clippy::type_complexity)]
pub(crate) fn scene_grid_boxes(
    scene: &Scene,
    voxels_per_block: u32,
) -> (Vec<([f32; 3], [f32; 3])>, Vec<([f32; 3], [f32; 3])>) {
    let mut lattice_boxes = Vec::new();
    let mut floor_boxes = Vec::new();
    let want_lattice_master = scene.master_block_lattice;
    let want_floor_master = scene.master_floor_grid;
    if !want_lattice_master && !want_floor_master {
        return (lattice_boxes, floor_boxes);
    }
    for (path, _id, _depth) in scene.tree_rows() {
        let Some(node) = scene.node_at_path(&path) else {
            continue;
        };
        let want_lattice = want_lattice_master && node.grids.block_lattice;
        let want_floor = want_floor_master && node.grids.floor_grid;
        if !want_lattice && !want_floor {
            continue;
        }
        let Some(node_box) = scene.node_block_lattice_box_recentred(&path, voxels_per_block) else {
            continue;
        };
        if want_lattice {
            lattice_boxes.push(node_box);
        }
        if want_floor {
            floor_boxes.push(node_box);
        }
    }
    (lattice_boxes, floor_boxes)
}

/// Block-boundary coordinates `[lo, lo+step, …, hi]` along one axis. The corners
/// `lo`/`hi` are block-aligned (the caller supplies an enclosing-block box), so the
/// `step`-stride walk lands exactly on `hi`; a final clamp guards float drift so the
/// closing block plane is always present.
pub(crate) fn block_boundaries(lo: f32, hi: f32, step: u32) -> Vec<f32> {
    let step = step.max(1) as f32;
    let mut values = Vec::new();
    let mut g = lo;
    // `+ step * 0.5` tolerance: include the plane at (or fractionally past) `hi`.
    while g <= hi + step * 0.5 {
        values.push(g.min(hi));
        g += step;
    }
    if values.last().copied() != Some(hi) {
        values.push(hi);
    }
    values
}

/// VOXEL-boundary coordinates `[lo, lo+1, …, hi]` along one axis, each tagged with
/// whether it is also a BLOCK boundary (`is_block`). The walk steps one voxel at a
/// time from the block-aligned `lo`, so every `step`-th line is flagged as a block
/// edge — meaning the bold (block) floor lines land on EXACTLY the same coordinates
/// as the block lattice's vertical lines (which `block_boundaries(lo, hi, step)`
/// places at `lo + k·step`). This is what makes the fine floor grid align with the
/// block lattice: the two share the block-aligned `lo` origin and the same stride.
pub(crate) fn voxel_boundaries(lo: f32, hi: f32, step: u32) -> Vec<(f32, bool)> {
    let step = step.max(1);
    let mut values = Vec::new();
    let mut index = 0i64;
    loop {
        let coord = lo + index as f32;
        // Closing guard: never overshoot `hi`; the final line is the block-aligned `hi`.
        if coord >= hi - 0.5 {
            values.push((hi, true));
            break;
        }
        values.push((coord, index.rem_euclid(step as i64) == 0));
        index += 1;
    }
    values
}

/// Append a 3D block lattice for the box `[min, max]` (voxels) — grid lines at every
/// BLOCK boundary (spacing = `step`) — into `vertices` (issue #29 S3, per-object).
/// Port of the prototype `buildGrids` lattice loop, now spanning an arbitrary box.
pub(crate) fn lattice_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let color = with_alpha(srgb_hex_to_linear(LATTICE_COLOR_HEX), LATTICE_ALPHA);
    let xs = block_boundaries(min[0], max[0], step);
    let ys = block_boundaries(min[1], max[1], step);
    let zs = block_boundaries(min[2], max[2], step);

    let mut add = |from: [f32; 3], to: [f32; 3]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // The full 3D block lattice draws line families along all three axes. Z-up: the
    // VERTICAL pillars are the Z-along family below (between the XY ground nodes); the
    // X- and Y-along families are the horizontal grid lines.
    // Lines along Y at every (x, z) lattice node.
    for &x in &xs {
        for &z in &zs {
            add([x, min[1], z], [x, max[1], z]);
        }
    }
    // Lines along X at every (y, z) lattice node.
    for &y in &ys {
        for &z in &zs {
            add([min[0], y, z], [max[0], y, z]);
        }
    }
    // Lines along Z (the VERTICAL pillars, Z-up) at every (x, y) lattice node.
    for &x in &xs {
        for &y in &ys {
            add([x, y, min[2]], [x, y, max[2]]);
        }
    }
}

/// Append a FINE floor grid for the box `[min, max]` (voxels) on its BASE plane
/// (Z-up: exactly at `z = min[2]`, an XY grid) into `vertices` (issue #29 fix).
/// Two-tier, mirroring the block lattice and the Point ground plane:
///
/// * **Fine voxel lines** — one per voxel boundary (step 1), at the subtle
///   [`FLOOR_VOXEL_ALPHA`].
/// * **Bold block lines** — at every block boundary (step = `step`), at the
///   brighter [`FLOOR_ALPHA`], drawn ON TOP so block edges read clearly.
///
/// Both tiers walk from the BLOCK-ALIGNED `min` corner with a 1-voxel stride
/// ([`voxel_boundaries`]), so the bold block lines land on `min + k·step` — the
/// EXACT coordinates of the block lattice's vertical lines
/// ([`block_boundaries`]). The floor grid therefore shares the lattice's global
/// frame and their lines coincide at the base plane. Z-up: the base plane is the
/// node's bottom EXACTLY (`z = min[2]`), an XY-plane grid, so the floor's block
/// lines meet the block lattice's bottom edges with no vertical gap; z-fighting
/// against the model's coincident bottom face is avoided by the floor's own
/// depth-biased uniform buffer (the SAME line pipeline as the lattice draw, not a
/// separate one) rather than a geometric drop.
pub(crate) fn floor_vertices_into(vertices: &mut Vec<LineVertex>, min: [f32; 3], max: [f32; 3], step: u32) {
    let voxel_color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_VOXEL_ALPHA);
    let block_color = with_alpha(srgb_hex_to_linear(FLOOR_COLOR_HEX), FLOOR_ALPHA);
    // Z-up: the floor is an XY grid at the node's bottom (`z = min[2]`).
    let z = min[2];
    let xs = voxel_boundaries(min[0], max[0], step);
    let ys = voxel_boundaries(min[1], max[1], step);

    let mut add = |from: [f32; 3], to: [f32; 3], color: [f32; 4]| {
        vertices.push(LineVertex { position: from, color });
        vertices.push(LineVertex { position: to, color });
    };

    // Minor pass: fine voxel lines (one per voxel boundary), subtle.
    // Lines parallel to Y, at every X voxel boundary.
    for &(x, _) in &xs {
        add([x, min[1], z], [x, max[1], z], voxel_color);
    }
    // Lines parallel to X, at every Y voxel boundary.
    for &(y, _) in &ys {
        add([min[0], y, z], [max[0], y, z], voxel_color);
    }
    // Major pass: bold block lines, on top, coincident with the block lattice.
    for &(x, is_block) in &xs {
        if is_block {
            add([x, min[1], z], [x, max[1], z], block_color);
        }
    }
    for &(y, is_block) in &ys {
        if is_block {
            add([min[0], y, z], [max[0], y, z], block_color);
        }
    }
}
