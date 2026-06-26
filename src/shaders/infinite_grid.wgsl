// Analytic infinite reference grid (issue #29 Points fast-follow).
//
// Replaces the old finite, camera-relative TILED LINE quad (48-block radius) that
// cut off at a hard edge / near-clip at shallow viewing angles. This is the
// standard fullscreen ray-plane technique:
//
//   * The vertex stage emits ONE oversized triangle covering the viewport.
//   * The fragment stage reconstructs the world-space view ray for the pixel by
//     UNPROJECTING the pixel's NDC at z=near AND z=far through the inverse view-
//     projection (perspective-dividing each), then ray_origin = near point,
//     ray_dir = far - near. This is correct for BOTH perspective AND orthographic
//     projection: under perspective every ray shares the eye; under ortho the rays
//     are PARALLEL (constant direction) and the ORIGIN varies per pixel — an
//     eye-based ray (`normalize(world - eye)`) is wrong under ortho and produces a
//     full-screen moiré that even covers the sky. It intersects the ray with the
//     Point's plane; pixels whose ray does not hit the plane in front of the camera
//     are discarded → the grid is truly infinite (spans to the horizon) with NO
//     finite border, and the sky stays clear under ortho too.
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

// Grazing-angle (horizon) fade threshold: the grid's alpha ramps from 0 to full as
// `abs(dot(ray_direction, plane_normal))` (the sine of the ray's elevation above the
// plane) rises from 0 to this value. Below it the view ray is grazing the plane —
// approaching the horizon — so the grid dissolves smoothly into the background
// instead of cutting off at a hard horizontal line. ~0.10 ≈ within ~5.7° of edge-on;
// large enough to kill the hard ortho/shallow cutoff, small enough that the grid
// still reads as receding far toward the horizon before it fades.
const GRAZING_FADE_START: f32 = 0.10;

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

// Robust anti-aliased "pristine grid" coverage for one tier at the given spacing
// (Ben Golus, "The Best Darn Grid Shader (Yet)"). Returns a value in [0,1] that is
// ~1 ON a line and 0 between lines, with the line kept ~`line_pixels` wide via the
// screen-space derivative of the in-plane coordinate. The critical property over a
// naive `1 - dist/fwidth` grid: when a tier's cells shrink below ~1 pixel (grazing,
// far, or coarse pixels) the per-axis coverage FADES toward the line's duty cycle
// instead of saturating to 1, so the grid NEVER fills solid. `line_pixels` is the
// target on-screen line half-extent in pixels (≈1 for crisp AA lines).
//
// Returns vec2(coverage, lod_visibility): `.x` is the line coverage in [0,1]; `.y`
// is an LOD visibility factor in [0,1] that goes to 0 as the cells drop below a
// pixel, so the caller can fade the tier OUT (rather than letting it become a sheet)
// when its period is sub-pixel.
fn grid_coverage(coord: vec2<f32>, spacing: f32, line_pixels: f32) -> vec2<f32> {
    let scaled = coord / spacing;
    // Per-axis derivative of the scaled coordinate → cells per pixel.
    let derivative = fwidth(scaled);
    let inv_derivative = 1.0 / max(derivative, vec2<f32>(1e-8));
    // Target line half-width in CELL units per axis (line_pixels worth of pixels).
    let half_width = derivative * line_pixels;
    // Triangle-wave distance to the nearest grid line, in [0, 0.5] cell units.
    let dist_to_line = abs(fract(scaled - 0.5) - 0.5);
    // Antialiased line: 1 inside the half-width, ramping to 0 over one pixel. This is
    // the standard `smoothstep`-free analytic AA: coverage = clamp((hw - d)/fw + 0.5).
    var line2 = clamp((half_width - dist_to_line) * inv_derivative + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    // KEY anti-saturation step: as cells approach sub-pixel (derivative → large), the
    // line can no longer be resolved; fade each axis' coverage toward its DUTY CYCLE
    // (2*half_width, the fraction of the cell the line covers) rather than letting it
    // clamp to 1 everywhere. This keeps the average grey constant instead of solid.
    line2 = mix(line2, clamp(half_width * 2.0, vec2<f32>(0.0), vec2<f32>(1.0)), clamp(derivative - 1.0, vec2<f32>(0.0), vec2<f32>(1.0)));
    // Combine the two axes the pristine-grid way: a + b - a*b (a pixel on EITHER line
    // is lit), which avoids the over-bright corner of a naive max.
    let coverage = line2.x + line2.y - line2.x * line2.y;
    // LOD visibility: 1 while a cell still spans several pixels, fading to 0 as the
    // cell period drops toward a pixel. `cells_per_pixel` = derivative (cells per
    // screen pixel); its RECIPROCAL is the cell's on-screen PIXEL size. We fade the
    // whole tier out over the window where a cell shrinks from ~4 px down to ~2 px
    // and force it fully to 0 once a cell is < ~2 px, BEFORE the AA line can alias.
    //
    // This is the core of the ortho moiré fix: under orthographic the world scale is
    // UNIFORM across the screen (no foreshortening), so when zoomed out EVERY pixel's
    // cells are equally sub-pixel — there is no near band where the lines resolve.
    // A duty-cycle "keep the average grey" trick then paints a constant sheet that
    // the `fract` sampling turns into a beat/moiré pattern. Driving the tier hard to
    // zero once it is sub-pixel dissolves it cleanly instead (perspective is
    // unaffected: its near cells are many px and keep lod≈1).
    let cells_per_pixel = max(derivative.x, derivative.y);
    let pixels_per_cell = 1.0 / max(cells_per_pixel, 1e-8);
    let lod = smoothstep(2.0, 4.0, pixels_per_cell);
    return vec2<f32>(clamp(coverage, 0.0, 1.0), lod);
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment_main(input: VsOut) -> FsOut {
    var out: FsOut;

    let ndc_xy = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);

    // Reconstruct the world-space view ray for this pixel by UNPROJECTING the
    // pixel's NDC at the near AND far planes through the inverse view-projection.
    // The ray ORIGIN is the per-pixel near point and the DIRECTION is far - near.
    // This is correct for BOTH projections: under perspective every ray passes
    // through the shared eye; under ORTHOGRAPHIC the rays are parallel (one
    // constant direction) and the origin varies per pixel — using `grid.eye` as a
    // single shared origin (the old eye-based ray) is WRONG under ortho and yields
    // a wrong plane hit per pixel → full-screen moiré covering the sky. `t` is then
    // a parameter along this NORMALIZED ray; the near point already sits on the
    // near plane so `t > 0` is "in front of the camera" for both projections.
    let near_world = unproject(vec3<f32>(ndc_xy, 0.0));
    let far_world = unproject(vec3<f32>(ndc_xy, 1.0));
    let ray_origin = near_world;
    let ray_direction = normalize(far_world - near_world);

    // Intersect the ray with the Point's plane: normal · (p - origin) = 0.
    let normal = grid.normal_axis.xyz;
    let denom = dot(ray_direction, normal);
    // Ray parallel to the plane (grazing exactly edge-on): nothing to draw.
    if (abs(denom) < 1e-6) {
        discard;
    }
    let t = dot(grid.plane_origin.xyz - ray_origin, normal) / denom;
    // Plane is behind the camera (or behind the near plane) for this pixel: no hit
    // in front → infinite sky. Discarding here is what keeps the grid off the sky
    // under ortho (the parallel rays that miss the plane forward are dropped).
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
    // Each returns (coverage, lod_visibility); the lod factor fades the tier OUT as
    // its cells drop below ~1 pixel so a tier NEVER saturates into a solid sheet.
    let minor = grid_coverage(plane_coord, 1.0, 1.0);
    let major = grid_coverage(plane_coord, block_spacing, 1.0);

    // Apply each tier's LOD fade to its own alpha: the fine voxel tier dissolves
    // first (its cells go sub-pixel sooner), the bold block tier persists longer and
    // also fades toward the horizon. This is the core fix for the solid-fill bug.
    let minor_contribution = minor.x * minor_alpha * minor.y;
    let major_contribution = major.x * major_alpha * major.y;

    // Combine: the block lines are bolder (higher base alpha); the voxel lines are
    // subtle. Take the stronger contribution so a block line (which is also a voxel
    // line) reads at the bold alpha rather than summing past it.
    let alpha = max(minor_contribution, major_contribution);
    if (alpha < 0.002) {
        discard;
    }

    // Distance fade toward the horizon: the farther the hit is from the camera, the
    // more the grid dissolves into the background. Linear ramp to zero at
    // `fade_distance`, so the plane is truly infinite with no hard edge.
    let hit_distance = length(hit - ray_origin);
    let distance_fade = clamp(1.0 - hit_distance / max(fade_distance, 1.0), 0.0, 1.0);

    // HORIZON (grazing-angle) fade — the real fix for the shallow-angle hard cutoff
    // (issue #29). At the horizon the view ray becomes PARALLEL to the plane, so it
    // hits at an unboundedly large `t` and the lattice compresses into a single screen
    // row. `denom = dot(ray_direction, normal)` is the sine of the ray's elevation
    // above the plane; it goes to 0 exactly AT the horizon. The pure distance fade is
    // not enough on its own: under ORTHOGRAPHIC the whole visible ground sits at nearly
    // constant world distance (no foreshortening), so the distance ramp barely moves
    // across the screen and the grid stays near-full-alpha right up to the horizon —
    // reading as a HARD horizontal cutoff line (and, with a dense block size, the same
    // happens at shallow perspective). Fading alpha out as the ray grazes (abs(denom)
    // → 0) dissolves the grid smoothly INTO the horizon for BOTH projections, with no
    // hard edge and independent of distance / density — the grid truly recedes.
    let grazing_fade = smoothstep(0.0, GRAZING_FADE_START, abs(denom));

    let fade = distance_fade * grazing_fade;
    let final_alpha = alpha * fade;
    if (final_alpha < 0.002) {
        discard;
    }

    // Depth-correct occlusion: write the plane-hit point's clip depth so the
    // pipeline's LessEqual test lets opaque voxels (already in the depth buffer)
    // occlude the grid, with no z-fighting against itself.
    //
    // The written depth is CLAMPED into [0,1] as a defensive guard: at grazing /
    // orthographic angles the far band of the infinite plane can project to a clip
    // depth just OUTSIDE [0,1] (beyond the far plane, or, very close to the camera, in
    // front of the near plane), and with `unclipped_depth: false` the hardware would
    // DISCARD those fragments. The grazing fade above has already dissolved the grid's
    // alpha to ~0 by then, so this clamp only affects an otherwise-invisible tail — but
    // it guarantees no stray hard depth-clip seam can ever reappear at the horizon.
    // Occlusion still holds: real objects sit at a SMALLER depth than the clamped far
    // value, so they still win the LessEqual test and occlude the grid.
    let clip = grid.view_projection * vec4<f32>(hit, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);

    out.color = vec4<f32>(grid.line_color.xyz, final_alpha);
    return out;
}
