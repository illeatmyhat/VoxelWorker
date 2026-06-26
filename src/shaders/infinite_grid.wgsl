// Analytic infinite reference grid (issue #29 Points fast-follow).
//
// Replaces the old finite, camera-relative TILED LINE quad (48-block radius) that
// cut off at a hard edge / near-clip at shallow viewing angles. This is the
// standard fullscreen ray-plane technique:
//
//   * The vertex stage emits ONE oversized triangle covering the viewport.
//   * The fragment stage reconstructs the world-space view ray for the pixel
//     (inverse view-projection + eye) and intersects it with the Point's plane.
//     Pixels whose ray does not hit the plane in front of the camera are discarded
//     → the grid is truly infinite (it spans to the horizon) with NO finite border.
//   * Grid coverage is computed analytically from screen-space derivatives
//     (`fwidth`) of the plane's in-plane coordinates, so the lines are crisply
//     anti-aliased at any distance/angle — two tiers: a fine VOXEL grid (spacing 1)
//     and a bold BLOCK grid (spacing = density).
//   * Alpha fades to 0 with distance from the camera (toward the horizon) so the
//     plane dissolves into the background — infinite, no hard rim.
//   * `@builtin(frag_depth)` is written to the plane-hit point's clip depth and the
//     pipeline depth-tests LessEqual, so opaque objects (drawn earlier in the SAME
//     MSAA pass) OCCLUDE the grid. The grid reads as subtle world scaffold behind /
//     under the model, never an overlay on top of it.

struct GridUniforms {
    // Camera matrices, both in the RECENTRED render frame the voxels live in.
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    // Camera eye (recentred frame), xyz; .w unused.
    eye: vec4<f32>,
    // Plane origin (the Point's recentred position), xyz; .w unused.
    plane_origin: vec4<f32>,
    // Plane orientation: which world axes are the two IN-PLANE axes (u, v) and the
    // constant (normal) axis, encoded as basis vectors. u_axis/v_axis span the
    // plane; normal_axis is the plane normal. Packed as vec4 (xyz used).
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
    normal_axis: vec4<f32>,
    // Line colour (linear RGB) in .xyz; .w = voxel spacing (always 1.0, kept for
    // clarity / future tuning).
    line_color: vec4<f32>,
    // Grid tuning: x = block spacing (= density, voxels per block), y = minor (per
    // voxel) base alpha, z = major (per block) base alpha, w = fade distance (in
    // voxels) over which alpha ramps to zero.
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> grid: GridUniforms;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    // A single oversized triangle covering the viewport (same trick as the fog).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let p = positions[vertex_index];
    var out: VsOut;
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

// Unproject an NDC point (z in [0,1]) to world space (recentred frame).
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let world = grid.inverse_view_projection * vec4<f32>(ndc, 1.0);
    return world.xyz / world.w;
}

// Analytic anti-aliased grid coverage for one tier at the given spacing. Returns a
// value in [0,1] that is ~1 ON a line and 0 between lines, with the line width
// driven by the screen-space derivative of the in-plane coordinate so it stays one
// pixel wide at any distance/angle (the standard `fwidth` grid AA from
// "Best Darn Grid" / Inigo Quilez `filteredGrid`).
fn grid_coverage(coord: vec2<f32>, spacing: f32) -> f32 {
    let scaled = coord / spacing;
    // Derivative of the scaled coordinate → cells per pixel; the line half-width is
    // tied to this so distant cells don't alias into solid fill.
    let derivative = fwidth(scaled);
    // Distance to the nearest grid line on each axis, normalised by the derivative.
    let grid_dist = abs(fract(scaled - 0.5) - 0.5) / max(derivative, vec2<f32>(1e-6));
    let line = min(grid_dist.x, grid_dist.y);
    // 1 on the line, fading to 0 one (derivative-scaled) pixel away.
    return 1.0 - clamp(line, 0.0, 1.0);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment_main(input: VsOut) -> FsOut {
    var out: FsOut;

    let ndc_xy = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);

    // Reconstruct the world-space view ray for this pixel.
    let near_world = unproject(vec3<f32>(ndc_xy, 0.0));
    let far_world = unproject(vec3<f32>(ndc_xy, 1.0));
    let ray_origin = grid.eye.xyz;
    let ray_direction = normalize(far_world - near_world);

    // Intersect the ray with the Point's plane: normal · (p - origin) = 0.
    let normal = grid.normal_axis.xyz;
    let denom = dot(ray_direction, normal);
    // Ray parallel to the plane (grazing exactly edge-on): nothing to draw.
    if (abs(denom) < 1e-6) {
        discard;
    }
    let t = dot(grid.plane_origin.xyz - ray_origin, normal) / denom;
    // Plane is behind the camera for this pixel: no hit in front → infinite sky.
    if (t <= 0.0) {
        discard;
    }
    let hit = ray_origin + ray_direction * t;

    // In-plane coordinates relative to the Point origin (so grid lines land on the
    // GLOBAL lattice: origin + k·spacing). The two basis axes are unit world axes.
    let rel = hit - grid.plane_origin.xyz;
    let plane_coord = vec2<f32>(dot(rel, grid.u_axis.xyz), dot(rel, grid.v_axis.xyz));

    let block_spacing = grid.params.x;
    let minor_alpha = grid.params.y;
    let major_alpha = grid.params.z;
    let fade_distance = grid.params.w;

    // Two-tier coverage: fine per-VOXEL lines (spacing 1) and bold per-BLOCK lines.
    let minor = grid_coverage(plane_coord, 1.0);
    let major = grid_coverage(plane_coord, block_spacing);

    // Combine: the block lines are bolder (higher base alpha); the voxel lines are
    // subtle. Take the stronger contribution so a block line (which is also a voxel
    // line) reads at the bold alpha rather than summing past it.
    let alpha = max(minor * minor_alpha, major * major_alpha);
    if (alpha < 0.002) {
        discard;
    }

    // Distance fade toward the horizon: the farther the hit is from the camera, the
    // more the grid dissolves into the background. Linear ramp to zero at
    // `fade_distance`, so the plane is truly infinite with no hard edge.
    let hit_distance = length(hit - ray_origin);
    let fade = clamp(1.0 - hit_distance / max(fade_distance, 1.0), 0.0, 1.0);
    let final_alpha = alpha * fade;
    if (final_alpha < 0.002) {
        discard;
    }

    // Depth-correct occlusion: write the plane-hit point's clip depth so the
    // pipeline's LessEqual test lets opaque voxels (already in the depth buffer)
    // occlude the grid, with no z-fighting against itself.
    let clip = grid.view_projection * vec4<f32>(hit, 1.0);
    out.depth = clip.z / clip.w;

    out.color = vec4<f32>(grid.line_color.xyz, final_alpha);
    return out;
}
