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
//   * Each tier fades out via its OWN per-pixel LOD (cells-per-pixel from `fwidth`)
//     once its cells go sub-pixel — finer tiers fading before coarser ones — so the
//     plane dissolves into the background at the horizon with no hard world-distance
//     rim and no hard horizon line. There is NO grazing-angle fade and NO fixed
//     world-distance fade (both were removed: they caused the zoom-out vanish).
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
    // voxel) base alpha, z = major (per block) base alpha. w = legacy fade distance,
    // now UNUSED (the fixed world-distance fade was removed; fading is per-tier LOD
    // only) but kept in the layout for uniform-buffer stability.
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
    // Issue #91 (item 4): fade each tier out EARLIER (a cell must span ~3→7 px to be fully
    // lit, vs the old 2→4) so the plane dissolves harder toward the rim/distance and stays
    // a calm scaffold rather than a noisy sheet that buries the bottom-left status text.
    let lod = smoothstep(3.0, 7.0, pixels_per_cell);
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
    let hit = ray_origin + ray_direction * t;

    // HORIZON / SKY discard in CLIP SPACE. Project the plane-hit point through the
    // view-projection and reject the fragment when it lands BEHIND the camera, i.e.
    // `clip.w <= 0`:
    //   * Under PERSPECTIVE, clip.w is the camera-space depth in front of the eye;
    //     it is positive for points the eye can see and goes negative for plane hits
    //     that lie above the horizon (the ray would have to travel BACKWARD to reach
    //     them). This is exactly the sky region, so it is correctly culled.
    //   * Under ORTHOGRAPHIC, clip.w is a CONSTANT positive value (the projection has
    //     no perspective divide), so NOTHING is wrongly culled — every plane hit,
    //     foreground or far, is kept. This is what makes the near/foreground band of
    //     the ortho ground render instead of being cut off.
    // Note: we deliberately do NOT cull on the ray parameter `t` (a `t <= 0` test
    // from the per-pixel near-plane origin) — under ortho the near-plane origin can
    // already sit below the plane for foreground pixels, so a `t` test wrongly culls
    // the foreground and produces the hard near-side cutoff. `clip.w` is the correct,
    // projection-aware behind-camera test.
    let clip = grid.view_projection * vec4<f32>(hit, 1.0);
    if (clip.w <= 0.0) {
        discard;
    }

    // In-plane coordinates relative to the Point origin (so grid lines land on the
    // GLOBAL lattice: origin + k·spacing). The two basis axes are unit world axes.
    let rel = hit - grid.plane_origin.xyz;
    let plane_coord = vec2<f32>(dot(rel, grid.u_axis.xyz), dot(rel, grid.v_axis.xyz));

    let block_spacing = grid.params.x;
    let minor_alpha = grid.params.y;
    let major_alpha = grid.params.z;

    // THREE-tier coverage, each at a coarser spacing than the last:
    //   * minor  — fine per-VOXEL lines (spacing 1),
    //   * major  — bold per-BLOCK lines (spacing = density),
    //   * coarse — a per-8-BLOCK lattice that only carries weight once the finer tiers
    //     have gone sub-pixel.
    // Each `grid_coverage` returns (coverage, lod_visibility); the lod factor fades a
    // tier OUT as its cells drop below ~1 pixel so a tier NEVER saturates into a solid
    // sheet (the ortho moiré fix) AND so each tier dissolves on its OWN schedule.
    //
    // The per-tier LOD fade is now the WHOLE fade story — the old fixed world-DISTANCE
    // fade and the grazing-angle fade were both removed (they caused the dist-700
    // zoom-out vanish). A tier stays fully visible until its own cells are genuinely
    // sub-pixel, then dissolves; this imposes NO hard world-distance edge and NO hard
    // horizon line. The grid simply recedes — finer tiers fading before coarser ones,
    // all the way to the perspective horizon. When zoomed very far out in ortho, the
    // coarse 8-block tier keeps block-scale structure visible after the per-block tier
    // has gone sub-pixel.
    let coarse_spacing = block_spacing * 8.0;
    let minor = grid_coverage(plane_coord, 1.0, 1.0);
    let major = grid_coverage(plane_coord, block_spacing, 1.0);
    let coarse = grid_coverage(plane_coord, coarse_spacing, 1.0);

    // Apply each tier's LOD fade to its own alpha. The coarse tier borrows the bold
    // (major) base alpha so that, once the block tier fades out when zoomed far, the
    // coarse 8-block lattice still reads at a comparable weight rather than vanishing.
    let minor_contribution = minor.x * minor_alpha * minor.y;
    let major_contribution = major.x * major_alpha * major.y;
    let coarse_contribution = coarse.x * major_alpha * coarse.y;

    // Take the strongest contribution so a line shared by several tiers reads at the
    // bold alpha rather than summing past it.
    let final_alpha = max(max(minor_contribution, major_contribution), coarse_contribution);
    if (final_alpha < 0.002) {
        discard;
    }

    // Depth-correct occlusion: write the plane-hit point's clip depth so the pipeline's
    // LessEqual test lets opaque voxels (already in the depth buffer) occlude the grid,
    // with no z-fighting against itself.
    //
    // The written depth is CLAMPED into [0,1] as a defensive guard: at shallow /
    // orthographic angles the far (or very near) band of the infinite plane can project
    // to a clip depth just OUTSIDE [0,1]. With `unclipped_depth: false` the hardware
    // would otherwise DISCARD those fragments, reintroducing exactly the kind of hard
    // near/far depth-clip seam this rework removes. Clamping keeps those hits drawn.
    // Occlusion still holds: real objects sit at a SMALLER depth than the clamped far
    // value, so they win the LessEqual test and occlude the grid. (`clip` was computed
    // above for the behind-camera / horizon discard.)
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);

    out.color = vec4<f32>(grid.line_color.xyz, final_alpha);
    return out;
}
