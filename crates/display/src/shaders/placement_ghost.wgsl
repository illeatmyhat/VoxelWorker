// Placement ghost — a translucent analytic SDF drawn where an armed primitive's voxels
// WILL land (ADR 0022: "nothing recomposes during the gesture — render a coloured
// transparent SDF where the voxels will be"). The five `ShapeKind` primitives are rendered
// as parametric fields, sphere-traced on the GPU, over the composed voxel display.
//
// **This is a hand-written WGSL MIRROR of `voxel_core::voxel::signed_distance`**
// (crates/voxel_core/src/voxel.rs) and its three helpers, promoted verbatim from the
// parity-proven spike (`docs/design/wgsl-sdf-spike.md`: 0 voxels disagree with the CPU
// resolve). Every line below marked MIRROR has a named Rust counterpart; the `value_main`
// entry point exists so a parity test can read these functions' output back and diff it
// against the Rust.
//
// ## Frames (ADR 0008)
//
// The producer samples its SDF at `local_voxel_index + 0.5 - grid/2`, i.e. in a frame
// CENTRED on the producer's own grid. The display's world frame relates to absolute
// voxels by `absolute = world + recentre` (derivable from the display frame law
// `absolute = shading_absolute + (recentre - half)` together with
// `shading_absolute = world + half`). A leaf's producer-local voxel is
// `absolute - world_offset`. Composing:
//
//     sample = world_point + recentre - world_offset - grid/2
//
// so the CPU packs `center_world = world_offset + grid/2 - recentre` and the shader
// evaluates the field at `world_point - center_world`. `grid/2` is the EXACT
// half (a half-integer for an odd grid); `recentre` is the FLOORED half. The
// difference is the half-voxel term that a naive "the shape is at the origin"
// assumption silently drops.

struct PlacementGhostUniforms {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    // The central 3D viewport rect in physical pixels (x, y, width, height).
    viewport: vec4<f32>,
    // xyz: the shape's field centre in the world/render frame (see the frame note
    // above). w: the ShapeKind discriminant (0 Cylinder, 1 Tube, 2 Sphere, 3 Torus,
    // 4 Box) — matching `ShapeKind`'s declaration order in voxel_core.
    center_and_kind: vec4<f32>,
    // xyz: the inscribed semi-axes in voxels (`grid/2` per axis). w: `wall_blocks *
    // density` in voxels (Tube only).
    semi_axes_and_wall: vec4<f32>,
    // Linear RGB tint + source alpha for the translucent shell.
    tint: vec4<f32>,
    // x: the iso level (SURFACE_ISOLEVEL, 0.0). y: 1 when the pass should shade a lit
    // surface, 0 when it is the value probe. z: the value-probe plane's constant
    // coordinate. w: the value-probe world extent per axis.
    params: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: PlacementGhostUniforms;

const KIND_CYLINDER: u32 = 0u;
const KIND_TUBE: u32 = 1u;
const KIND_SPHERE: u32 = 2u;
const KIND_TORUS: u32 = 3u;
const KIND_BOX: u32 = 4u;

// ---------------------------------------------------------------------------
// The field. MIRROR of voxel_core::voxel.
// ---------------------------------------------------------------------------

// MIRROR of `signed_distance_box` (voxel.rs).
fn signed_distance_box(point: vec3<f32>, box_half: vec3<f32>) -> f32 {
    let q = abs(point) - box_half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// MIRROR of `signed_distance_ellipsoid` (voxel.rs) — the IQ approximation,
// including its exact zero-length early-out (which is NOT a numerical guard: the Rust
// returns the negative minor semi-axis there, and dropping it would return NaN).
fn signed_distance_ellipsoid(point: vec3<f32>, semi_axes: vec3<f32>) -> f32 {
    let scaled = point / semi_axes;
    let distance_to_unit = length(scaled);
    if (distance_to_unit == 0.0) {
        return -min(semi_axes.x, min(semi_axes.y, semi_axes.z));
    }
    let scaled_squared = point / (semi_axes * semi_axes);
    let gradient = length(scaled_squared);
    return distance_to_unit * (distance_to_unit - 1.0) / gradient;
}

// MIRROR of `signed_distance_elliptical_cylinder` (voxel.rs). Z-up: the circular
// cross-section lies in XY, `half_height` is the vertical (Z) half-extent.
fn signed_distance_elliptical_cylinder(
    point: vec3<f32>,
    semi_axis_x: f32,
    semi_axis_y: f32,
    half_height: f32,
) -> f32 {
    let radial =
        (length(vec2<f32>(point.x / semi_axis_x, point.y / semi_axis_y)) - 1.0)
        * min(semi_axis_x, semi_axis_y);
    let vertical = abs(point.z) - half_height;
    return min(max(radial, vertical), 0.0)
        + length(vec2<f32>(max(radial, 0.0), max(vertical, 0.0)));
}

// MIRROR of the `signed_distance` dispatcher (voxel.rs). `semi_axes` are the
// inscribed half-extents; `wall_voxels` is `wall_blocks * density` (Tube only).
fn signed_distance(kind: u32, point: vec3<f32>, semi_axes: vec3<f32>, wall_voxels: f32) -> f32 {
    let semi_axis_x = semi_axes.x;
    let semi_axis_y = semi_axes.y;
    let semi_axis_z = semi_axes.z;

    if (kind == KIND_CYLINDER) {
        return signed_distance_elliptical_cylinder(
            point, semi_axis_x, semi_axis_y, semi_axis_z);
    }
    if (kind == KIND_TUBE) {
        let outer = signed_distance_elliptical_cylinder(
            point, semi_axis_x, semi_axis_y, semi_axis_z);
        let inner = signed_distance_elliptical_cylinder(
            point,
            max(semi_axis_x - wall_voxels, 0.01),
            max(semi_axis_y - wall_voxels, 0.01),
            semi_axis_z + 1.0,
        );
        return max(outer, -inner);
    }
    if (kind == KIND_SPHERE) {
        return signed_distance_ellipsoid(point, semi_axes);
    }
    if (kind == KIND_TORUS) {
        // Z-up: the ring lies in XY, swept about +Z; the minor radius is the Z extent.
        let tube_radius = semi_axis_z;
        let ring_radius = max(min(semi_axis_x, semi_axis_y) - tube_radius, 0.0);
        let radial = length(vec2<f32>(point.x, point.y)) - ring_radius;
        return length(vec2<f32>(radial, point.z)) - tube_radius;
    }
    // KIND_BOX
    return signed_distance_box(point, semi_axes);
}

// The field in the shader's own sample frame: world point -> producer sample point.
fn field_at_world(world_point: vec3<f32>) -> f32 {
    let kind = u32(uniforms.center_and_kind.w);
    let sample = world_point - uniforms.center_and_kind.xyz;
    return signed_distance(
        kind, sample, uniforms.semi_axes_and_wall.xyz, uniforms.semi_axes_and_wall.w);
}

// ---------------------------------------------------------------------------
// The pass.
// ---------------------------------------------------------------------------

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOutput;
    output.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return output;
}

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

// Unproject a framebuffer pixel through the inverse view-projection into a WORLD ray.
// Near/far unprojection handles perspective AND orthographic. Same construction as
// `camera_ray` in brick_raymarch.wgsl, minus the sv-frame shift (this pass works in
// world coordinates and moves the SHAPE into them instead).
fn camera_ray(pixel: vec2<f32>) -> Ray {
    let ndc_x = (pixel.x - uniforms.viewport.x) / uniforms.viewport.z * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - uniforms.viewport.y) / uniforms.viewport.w * 2.0;
    let near_h = uniforms.inverse_view_projection * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = uniforms.inverse_view_projection * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_world = near_h.xyz / near_h.w;
    let far_world = far_h.xyz / far_h.w;
    var ray: Ray;
    ray.origin = near_world;
    ray.direction = normalize(far_world - near_world);
    return ray;
}

// Slab test against the shape's bounding box (the semi-axes, padded), in the sample
// frame. Returns (t_enter, t_exit); a miss has t_enter > t_exit.
fn bounds_interval(ray: Ray) -> vec2<f32> {
    let half = uniforms.semi_axes_and_wall.xyz + vec3<f32>(2.0);
    let origin = ray.origin - uniforms.center_and_kind.xyz;
    // Guard zero components without changing sign (the direction-guard law).
    let guard = 1e-8;
    var safe = ray.direction;
    safe.x = select(safe.x, guard * sign(safe.x + 1e-30), abs(safe.x) < guard);
    safe.y = select(safe.y, guard * sign(safe.y + 1e-30), abs(safe.y) < guard);
    safe.z = select(safe.z, guard * sign(safe.z + 1e-30), abs(safe.z) < guard);
    let inverse = 1.0 / safe;
    let t0 = (-half - origin) * inverse;
    let t1 = (half - origin) * inverse;
    let t_low = min(t0, t1);
    let t_high = max(t0, t1);
    // Enter at the AABB slab, NOT clamped to the near plane (no `max(..., 0.0)`). The
    // near plane is sized around the scene at the world origin, so once the camera pans
    // away the ghost sits BEHIND it; clamping the entry to 0 then made the march start
    // past the shape and miss, so the ghost only drew in the band where it fell in front
    // of the near plane. A placement ghost is an overlay affordance, not scene geometry —
    // it must march its own bounding box wherever it lands, so a negative entry (the shape
    // behind the near-plane ray origin) is allowed. `t_exit > 0` still rejects a shape
    // fully behind the ray.
    let t_enter = max(t_low.x, max(t_low.y, t_low.z));
    let t_exit = min(t_high.x, min(t_high.y, t_high.z));
    return vec2<f32>(t_enter, t_exit);
}

const MAX_TRACE_STEPS: u32 = 192u;
// The step relaxation. The IQ ellipsoid and the tube's `max(outer, -inner)` are not
// exact distance fields, so a full step can overshoot a thin feature; 0.7 is the
// factor at which every fixture traces cleanly.
const STEP_RELAXATION: f32 = 0.7;

// A miss sentinel that CANNOT collide with a real hit `t`. The entry `t` may now be
// NEGATIVE (the shape can sit behind the near-plane ray origin once the camera pans away
// from the origin-sized near plane — see `bounds_interval`), so `-1.0` is no longer a safe
// "no hit" marker: a genuine hit can be negative. A huge positive value never overlaps any
// real interval `t`.
const TRACE_MISS: f32 = 1.0e30;

// Sphere-trace the field. Returns `t` of the surface hit (possibly NEGATIVE), or
// [`TRACE_MISS`] on a miss.
fn trace(ray: Ray) -> f32 {
    let interval = bounds_interval(ray);
    if (interval.x > interval.y) {
        return TRACE_MISS;
    }
    // Scale the hit tolerance with the shape so a 1280-voxel shape is not traced to
    // sub-micron precision (and a 4-voxel one still resolves).
    let scale = max(max(uniforms.semi_axes_and_wall.x, uniforms.semi_axes_and_wall.y),
                    uniforms.semi_axes_and_wall.z);
    let tolerance = max(scale * 1e-4, 1e-3);
    var t = interval.x;
    for (var step = 0u; step < MAX_TRACE_STEPS; step = step + 1u) {
        if (t > interval.y) {
            return TRACE_MISS;
        }
        let distance = field_at_world(ray.origin + ray.direction * t) - uniforms.params.x;
        if (distance < tolerance) {
            return t;
        }
        t = t + max(distance * STEP_RELAXATION, tolerance);
    }
    return TRACE_MISS;
}

// The field's gradient by central differences at a CHOSEN epsilon. NOT `fwidth` — this
// pass runs in non-uniform control flow, the same constraint the brick shader carries.
// The epsilon is a parameter because the outline reads the surface at two scales: a fine
// probe for the true shading normal, and a coarse one that averages across a hard edge (a
// crease shows as the two disagreeing).
fn field_normal_eps(world_point: vec3<f32>, epsilon: f32) -> vec3<f32> {
    let dx = vec3<f32>(epsilon, 0.0, 0.0);
    let dy = vec3<f32>(0.0, epsilon, 0.0);
    let dz = vec3<f32>(0.0, 0.0, epsilon);
    let gradient = vec3<f32>(
        field_at_world(world_point + dx) - field_at_world(world_point - dx),
        field_at_world(world_point + dy) - field_at_world(world_point - dy),
        field_at_world(world_point + dz) - field_at_world(world_point - dz),
    );
    let magnitude = length(gradient);
    if (magnitude < 1e-12) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return gradient / magnitude;
}

// The shape's overall scale (largest semi-axis) — sizes the epsilons and tolerances.
fn shape_scale() -> f32 {
    return max(max(uniforms.semi_axes_and_wall.x, uniforms.semi_axes_and_wall.y),
               uniforms.semi_axes_and_wall.z);
}

// The fine shading normal.
fn field_normal(world_point: vec3<f32>) -> vec3<f32> {
    return field_normal_eps(world_point, max(shape_scale() * 1e-3, 1e-3));
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> FragmentOutput {
    let ray = camera_ray(position.xy);
    let t = trace(ray);
    // A real hit `t` may be negative (the shape behind the near-plane ray origin after a
    // pan), so miss is the sentinel, NOT `t < 0`.
    if (t >= TRACE_MISS) {
        discard;
    }
    let hit_world = ray.origin + ray.direction * t;
    let normal = field_normal(hit_world);

    // The same headlight-ish shade the mesh path uses, kept deliberately flat so the
    // analytic shell reads as a preview rather than competing with the voxels.
    let to_eye = -ray.direction;
    let facing = abs(dot(normal, to_eye));
    // Fresnel-ish rim: the shell is most opaque at grazing angles, so the silhouette —
    // exactly where the ghost declares where the node will land — is the strongest edge.
    let rim = pow(1.0 - facing, 2.0);
    let lit = 0.35 + 0.65 * facing;
    let alpha = clamp(uniforms.tint.w * (0.45 + 0.85 * rim), 0.0, 1.0);

    let clip = uniforms.view_projection * vec4<f32>(hit_world, 1.0);

    var output: FragmentOutput;
    output.color = vec4<f32>(uniforms.tint.rgb * lit * alpha, alpha);
    // Clamp into the depth range. The scene's near/far are sized around the geometry at
    // the world origin, so once the camera pans away the ghost's true clip depth falls
    // OUTSIDE [0, 1] and the fragment is depth-clipped — the ghost vanished across most of
    // the screen. It is an overlay affordance (depth compare Always, depth write off), so
    // clamping keeps it in range and always visible without disturbing the depth buffer.
    output.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return output;
}

// ---------------------------------------------------------------------------
// The parity probe (drift policing).
// ---------------------------------------------------------------------------
//
// Not a display pass: each pixel is a SAMPLE POINT, and the fragment writes the raw
// f32 field value there into an Rgba32Float target. A CPU test renders this, reads it
// back, and diffs against `voxel_core::voxel::signed_distance` at the identical points
// — which is what makes the mirror above mechanically checkable rather than a promise.
//
// The sample lattice: pixel (x, y) of an N x N target maps to the producer's own local
// voxel-centre grid, `(x + 0.5, y + 0.5, plane + 0.5)` less `grid/2` — i.e. exactly the
// points `SdfShape::resolve_into` evaluates. `params.z` is the plane index; `params.w`
// is the axis span in voxels.
@fragment
fn value_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let semi = uniforms.semi_axes_and_wall.xyz;
    let sample = vec3<f32>(
        floor(position.x) + 0.5 - semi.x,
        floor(position.y) + 0.5 - semi.y,
        uniforms.params.z + 0.5 - semi.z,
    );
    let kind = u32(uniforms.center_and_kind.w);
    let value = signed_distance(kind, sample, semi, uniforms.semi_axes_and_wall.w);
    return vec4<f32>(value, sample.x, sample.y, sample.z);
}
