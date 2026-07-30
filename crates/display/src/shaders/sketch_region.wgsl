// The sketch region wash — the RESOLVED 2D material region, shaded ON the sketch plane
// (ADR 0030 §3). What is picked is what will resolve as material, so the plane itself carries
// the signal at low alpha and a void carries none.
//
// **This is a hand-written WGSL MIRROR of `substrate::geom2d::signed_distance_to_region`**
// and the three functions it folds — `signed_distance_to_polygon`, `distance_point_to_segment`
// (the `Metric::Euclidean` branch) and `point_in_polygon` for the sign. Every function marked
// MIRROR has a named Rust counterpart, and the CPU and the shader are handed the SAME
// `(LoopRole, Vec<[f32; 2]>)` value the resolve consumes — that is what the geom2d module doc's
// f32 measurement half exists for, since WGSL has no f64.
//
// ## Why a field and not a mesh
//
// The overlay used to triangulate each face and fill it with an `egui::Mesh`. Two faces that
// nest have overlapping polygons, so the alpha composited twice; a void had to be bridged out of
// its contour by hand. Evaluated as a field none of that arises — `point_in_region`'s rule
// ("inside a Fill, inside no Hole") IS the nesting, per pixel — and the edge gets antialiasing
// from the distance for free.
//
// ## Frames (ADR 0008)
//
// The CPU packs the plane as the render-frame position of profile coordinate `(0, 0)` plus the
// render-frame displacement of `+1` voxel along each of the profile's two in-plane axes — all
// three from the ONE forward map, `SketchHandles::profile_to_render`. The axes come off a lattice
// rotation, so they are orthonormal and a dot product recovers the coordinate. Re-deriving the
// plane from the sketch's own numbers here is exactly the silent frame-error mode that split
// prevents.

struct SketchRegionUniforms {
    // The RAY-FRAME unprojection matrix (camera::SceneMatrices::ray_unprojection), inverted:
    // eye-anchored + camera-bracketed under perspective, the plain frame under ortho. Unproject
    // through it for an EYE-RELATIVE ray; `ray_eye` carries the render-frame origin added back
    // outside the matrix math (a06d215).
    ray_inverse_unprojection: mat4x4<f32>,
    // The ray frame's origin in the render frame (SceneMatrices::ray_eye): the eye under
    // perspective, zero under ortho. xyz; .w unused.
    ray_eye: vec4<f32>,
    // The central 3D viewport rect in physical pixels (x, y, width, height).
    viewport: vec4<f32>,
    // xyz: the render-frame position of profile coordinate (0, 0). w unused.
    plane_origin: vec4<f32>,
    // xyz: the render-frame displacement of +1 voxel along the profile's first in-plane axis.
    // w unused.
    plane_axis0: vec4<f32>,
    // The same for the second in-plane axis.
    plane_axis1: vec4<f32>,
    // xyz: the plane's unit normal in the render frame. w unused.
    plane_normal: vec4<f32>,
    // LINEAR RGB + source alpha of the wash. The shell converts the theme token, so the colour
    // has one definition and it is the one the 2D chrome uses.
    tint: vec4<f32>,
    // The profile's bounding box, padded: xy = minimum, zw = maximum, in profile voxels. The
    // early-out that keeps a whole-screen pass from evaluating every edge at every pixel.
    bounds: vec4<f32>,
    // x: the number of loops. y/z/w unused.
    counts: vec4<u32>,
};

// One boundary loop's slice of `region_points`.
struct RegionLoop {
    // 0 Fill, 1 Hole — matching `LoopRole`'s declaration order in substrate::geom2d
    // (`sketch_region_loop_role_discriminant` is the guarded conversion).
    role: u32,
    start: u32,
    count: u32,
    padding: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: SketchRegionUniforms;
@group(0) @binding(1)
var<storage, read> region_loops: array<RegionLoop>;
@group(0) @binding(2)
var<storage, read> region_points: array<vec2<f32>>;

const ROLE_FILL: u32 = 0u;
const ROLE_HOLE: u32 = 1u;

// WGSL has no infinity literal. Every distance here is in profile VOXELS, so a value this large
// folds through `min`/`max` exactly as `f32::INFINITY` does in the Rust.
const FAR: f32 = 1.0e30;

// The widest antialiasing band, in profile voxels. At a grazing view the per-pixel footprint
// diverges, and an unclamped band would smear the whole region into a flat haze.
const MAX_EDGE_FOOTPRINT: f32 = 4.0;

// ---------------------------------------------------------------------------
// The field. MIRROR of substrate::geom2d.
// ---------------------------------------------------------------------------

// MIRROR of `distance_point_to_segment` (geom2d.rs), the `Metric::Euclidean` branch. The
// Chebyshev branch is not mirrored: it is the lattice metric an outset measures in, and a wash
// wants the round one.
fn distance_point_to_segment(a: vec2<f32>, b: vec2<f32>, point: vec2<f32>) -> f32 {
    let delta = b - a;
    let offset = point - a;
    if (delta.x == 0.0 && delta.y == 0.0) {
        return length(offset);
    }
    let along = clamp(dot(offset, delta) / dot(delta, delta), 0.0, 1.0);
    return length(offset - along * delta);
}

// MIRROR of `point_in_polygon` (geom2d.rs): the crossing number of a ray in the +axis1
// direction. The loop is implicitly closed (last vertex → first).
fn point_in_polygon(start: u32, count: u32, sample: vec2<f32>) -> bool {
    if (count == 0u) {
        return false;
    }
    var inside = false;
    var previous = count - 1u;
    for (var current = 0u; current < count; current = current + 1u) {
        let here = region_points[start + current];
        let last = region_points[start + previous];
        if ((here.y > sample.y) != (last.y > sample.y)) {
            let crossing = (last.x - here.x) * (sample.y - here.y) / (last.y - here.y) + here.x;
            if (sample.x < crossing) {
                inside = !inside;
            }
        }
        previous = current;
    }
    return inside;
}

// MIRROR of `signed_distance_to_polygon` (geom2d.rs): nearest edge, signed by containment.
fn signed_distance_to_polygon(start: u32, count: u32, point: vec2<f32>) -> f32 {
    if (count < 2u) {
        return FAR;
    }
    var nearest = FAR;
    var previous = region_points[start + count - 1u];
    for (var index = 0u; index < count; index = index + 1u) {
        let current = region_points[start + index];
        nearest = min(nearest, distance_point_to_segment(previous, current, point));
        previous = current;
    }
    if (point_in_polygon(start, count, point)) {
        return -nearest;
    }
    return nearest;
}

// MIRROR of `signed_distance_to_region` (geom2d.rs): `min` over the Fill loops (union), then
// `max` against each negated Hole loop (subtraction). Two passes, in that order — the same 2D
// reading of the 3D composite algebra, so a hole in a profile behaves like a subtracted body.
fn signed_distance_to_region(point: vec2<f32>) -> f32 {
    var distance = FAR;
    for (var index = 0u; index < uniforms.counts.x; index = index + 1u) {
        let region = region_loops[index];
        if (region.role == ROLE_FILL) {
            distance = min(
                distance,
                signed_distance_to_polygon(region.start, region.count, point)
            );
        }
    }
    for (var index = 0u; index < uniforms.counts.x; index = index + 1u) {
        let region = region_loops[index];
        if (region.role == ROLE_HOLE) {
            distance = max(
                distance,
                -signed_distance_to_polygon(region.start, region.count, point)
            );
        }
    }
    return distance;
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

// Unproject a framebuffer pixel through the RAY-FRAME inverse into a render-frame ray — the same
// construction `placement_ghost.wgsl` and the brick shader's `camera_ray` use, so the wash's ray
// and the voxel ray are the same ray.
fn camera_ray(pixel: vec2<f32>) -> Ray {
    let ndc_x = (pixel.x - uniforms.viewport.x) / uniforms.viewport.z * 2.0 - 1.0;
    let ndc_y = 1.0 - (pixel.y - uniforms.viewport.y) / uniforms.viewport.w * 2.0;
    let near_h = uniforms.ray_inverse_unprojection * vec4<f32>(ndc_x, ndc_y, 0.0, 1.0);
    let far_h = uniforms.ray_inverse_unprojection * vec4<f32>(ndc_x, ndc_y, 1.0, 1.0);
    let near_eye_relative = near_h.xyz / near_h.w;
    let far_eye_relative = far_h.xyz / far_h.w;
    var ray: Ray;
    ray.origin = uniforms.ray_eye.xyz + near_eye_relative;
    ray.direction = normalize(far_eye_relative - near_eye_relative);
    return ray;
}

// The profile coordinate of a render-frame point on the plane. The in-plane axes are orthonormal
// (a lattice rotation), so the projection is two dot products.
fn profile_of(hit: vec3<f32>) -> vec2<f32> {
    let offset = hit - uniforms.plane_origin.xyz;
    return vec2<f32>(
        dot(offset, uniforms.plane_axis0.xyz),
        dot(offset, uniforms.plane_axis1.xyz)
    );
}

@fragment
fn fragment_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let ray = camera_ray(position.xy);
    let normal = uniforms.plane_normal.xyz;
    // Guard the grazing case without changing sign (the direction-guard law).
    let denominator = dot(normal, ray.direction);
    let guard = 1.0e-8;
    let safe = select(
        denominator,
        guard * sign(denominator + 1.0e-30),
        abs(denominator) < guard
    );
    let along = dot(normal, uniforms.plane_origin.xyz - ray.origin) / safe;
    let profile = profile_of(ray.origin + ray.direction * along);
    // The screen footprint of one profile voxel, taken BEFORE any branch: `dpdx`/`dpdy` are only
    // legal in uniform control flow.
    let footprint = min(
        max(length(dpdx(profile)), length(dpdy(profile))),
        MAX_EDGE_FOOTPRINT
    );
    // Behind the ray origin: the plane is behind the viewer, so there is nothing to wash.
    if (along <= 0.0) {
        discard;
    }
    // Outside the padded profile bounds: the early-out that keeps the whole-screen pass from
    // folding every edge at every pixel.
    if (any(profile < uniforms.bounds.xy) || any(profile > uniforms.bounds.zw)) {
        discard;
    }
    let distance = signed_distance_to_region(profile);
    let coverage = 1.0 - smoothstep(-footprint * 0.5, footprint * 0.5, distance);
    if (coverage <= 0.0) {
        discard;
    }
    let alpha = uniforms.tint.a * coverage;
    return vec4<f32>(uniforms.tint.rgb * alpha, alpha);
}
