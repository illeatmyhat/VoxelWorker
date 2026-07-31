// The sketch region wash — the RESOLVED 2D material region, shaded ON the sketch plane
// What is picked is what will resolve as material, so the plane itself carries
// the signal at low alpha and a void carries none.
//
// **This is a hand-written WGSL MIRROR of `substrate::geom2d::signed_distance_to_region`**
// and the functions it folds — `nearest_boundary_distance`, `RegionEdge::distance` (the
// `Metric::Euclidean` branch) and `point_in_edge_loop` for the sign. Every function marked MIRROR
// has a named Rust counterpart, and the CPU and the shader are handed the SAME
// `(LoopRole, Vec<RegionEdge>)` value the resolve consumes — that is what the geom2d module doc's
// f32 measurement half exists for, since WGSL has no f64.
//
// The boundary arrives as EDGES, arcs included, so a curve is shaded as a curve. There is no
// tolerance in this pass and no screen-space chord budget to tune: a circle is one arc primitive
// however far the view has zoomed in, which is both exact and cheaper than the twenty-odd chords
// that replaces the old fill-based overlay.
//
// The loop ORDER is part of that value: innermost-first (smallest enclosed area first), which is
// what makes a loop govern its own area and nothing nested inside it.
//
// ## Why a field and not a mesh
//
// The overlay evaluates the region directly instead of triangulating each face with an `egui::Mesh`. Two faces that
// nest have overlapping polygons, so the alpha composited twice; a void had to be bridged out of
// its contour by hand. Evaluated as a field none of that arises — `point_in_region`'s rule
// ("the innermost loop containing this point decides") IS the nesting, per pixel — and the edge
// gets antialiasing from the distance for free.
//
// ## Frames
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
    // LINEAR RGB + source alpha of the wash. The shell converts the theme token, so the color
    // has one definition and it is the one the 2D chrome uses.
    tint: vec4<f32>,
    // The profile's bounding box, padded: xy = minimum, zw = maximum, in profile voxels. The
    // early-out that keeps a whole-screen pass from evaluating every edge at every pixel.
    bounds: vec4<f32>,
    // x: the number of loops, ordered innermost-first. y/z/w unused.
    counts: vec4<u32>,
};

// One boundary loop's slice of `region_edges`. Innermost-first in `region_loops`.
struct RegionLoop {
    // 0 Fill, 1 Hole — matching `LoopRole`'s declaration order in substrate::geom2d
    // (`sketch_region_loop_role_discriminant` is the guarded conversion).
    role: u32,
    start: u32,
    count: u32,
    padding: u32,
};

// MIRROR of `substrate::geom2d::RegionEdge` — a straight span, or an arc that stays an arc.
// `center`/`radius`/`start_radians`/`sweep_radians` are unused when `kind` is EDGE_SEGMENT.
struct RegionEdge {
    // `from`/`to` would be nicer, but `from` is a WGSL reserved keyword.
    start_point: vec2<f32>,
    end_point: vec2<f32>,
    center: vec2<f32>,
    radius: f32,
    start_radians: f32,
    sweep_radians: f32,
    kind: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: SketchRegionUniforms;
@group(0) @binding(1)
var<storage, read> region_loops: array<RegionLoop>;
@group(0) @binding(2)
var<storage, read> region_edges: array<RegionEdge>;

const ROLE_FILL: u32 = 0u;
const ROLE_HOLE: u32 = 1u;

const EDGE_SEGMENT: u32 = 0u;
const EDGE_ARC: u32 = 1u;

const TAU: f32 = 6.2831853;
const HALF_PI: f32 = 1.5707963;

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

// MIRROR of `travel_to_bearing` (geom2d.rs): how far along the sweep a bearing sits, or a negative
// sentinel when it is off the arc. WGSL has no Option, and travel is never negative on the arc.
fn travel_to_bearing(start_radians: f32, sweep_radians: f32, bearing: f32) -> f32 {
    var travelled: f32;
    if (sweep_radians < 0.0) {
        travelled = start_radians - bearing;
    } else {
        travelled = bearing - start_radians;
    }
    // `rem_euclid(TAU)`: the non-negative remainder, which `%` alone does not give for a negative.
    travelled = travelled - TAU * floor(travelled / TAU);
    if (travelled > abs(sweep_radians)) {
        return -1.0;
    }
    return travelled;
}

// MIRROR of `RegionEdge::distance` (geom2d.rs), the `Metric::Euclidean` branch. The Chebyshev
// branch is not mirrored: it is the lattice metric an outset measures in, and a wash wants the
// round one.
//
// The arc is measured as a CURVE — the distance to the circle where the point's bearing falls
// inside the sweep, the nearer endpoint outside it. No chords exist to be seen.
fn distance_to_edge(edge: RegionEdge, point: vec2<f32>) -> f32 {
    if (edge.kind == EDGE_SEGMENT) {
        return distance_point_to_segment(edge.start_point, edge.end_point, point);
    }
    let offset = point - edge.center;
    let bearing = atan2(offset.y, offset.x);
    if (travel_to_bearing(edge.start_radians, edge.sweep_radians, bearing) >= 0.0) {
        return abs(length(offset) - edge.radius);
    }
    return min(distance(edge.start_point, point), distance(edge.end_point, point));
}

// MIRROR of `segment_crossings` (geom2d.rs).
fn segment_crossings(a: vec2<f32>, b: vec2<f32>, sample: vec2<f32>) -> u32 {
    if ((b.y > sample.y) == (a.y > sample.y)) {
        return 0u;
    }
    let crossing_0 = (a.x - b.x) * (sample.y - b.y) / (a.y - b.y) + b.x;
    if (sample.x < crossing_0) {
        return 1u;
    }
    return 0u;
}

// MIRROR of `RegionEdge::crossings` (geom2d.rs): how many times a ray cast from `sample` in the
// +axis0 direction crosses this edge.
//
// An arc can cross the ray's line twice, so it is first cut at its own top and bottom — the only
// places its tangent turns horizontal — leaving pieces that are axis1-monotone and obey the same
// half-open rule a segment does. The cuts are the arc's own, so the parity does not depend on
// where the ray sits.
fn edge_crossings(edge: RegionEdge, sample: vec2<f32>) -> u32 {
    if (edge.kind == EDGE_SEGMENT) {
        return segment_crossings(edge.start_point, edge.end_point, sample);
    }
    let span = abs(edge.sweep_radians);
    var cuts = array<f32, 4>(0.0, span, span, span);
    var count = 2u;
    for (var index = 0u; index < 2u; index = index + 1u) {
        var extreme = HALF_PI;
        if (index == 1u) {
            extreme = -HALF_PI;
        }
        let travel = travel_to_bearing(edge.start_radians, edge.sweep_radians, extreme);
        if (travel > 0.0 && travel < span) {
            cuts[count] = travel;
            count = count + 1u;
        }
    }
    // Insertion sort over at most four entries — the two extremes arrive out of order.
    for (var index = 1u; index < count; index = index + 1u) {
        let held = cuts[index];
        var slot = index;
        while (slot > 0u && cuts[slot - 1u] > held) {
            cuts[slot] = cuts[slot - 1u];
            slot = slot - 1u;
        }
        cuts[slot] = held;
    }
    var direction = 1.0;
    if (edge.sweep_radians < 0.0) {
        direction = -1.0;
    }
    var crossings = 0u;
    for (var index = 0u; index + 1u < count; index = index + 1u) {
        let entry = cuts[index];
        let exit = cuts[index + 1u];
        if (exit <= entry) {
            continue;
        }
        // The outer ends are the STORED endpoints, so a vertex shared with the next edge is the
        // same value on both sides of the join.
        var low = edge.start_point;
        if (entry != 0.0) {
            let bearing = edge.start_radians + direction * entry;
            low = edge.center + edge.radius * vec2<f32>(cos(bearing), sin(bearing));
        }
        var high = edge.end_point;
        if (exit != span) {
            let bearing = edge.start_radians + direction * exit;
            high = edge.center + edge.radius * vec2<f32>(cos(bearing), sin(bearing));
        }
        if ((low.y > sample.y) == (high.y > sample.y)) {
            continue;
        }
        let rise = sample.y - edge.center.y;
        let half_chord = sqrt(max(edge.radius * edge.radius - rise * rise, 0.0));
        let middle = edge.start_radians + direction * (entry + exit) * 0.5;
        var crossing_0 = edge.center.x - half_chord;
        if (cos(middle) >= 0.0) {
            crossing_0 = edge.center.x + half_chord;
        }
        if (sample.x < crossing_0) {
            crossings = crossings + 1u;
        }
    }
    return crossings;
}

// MIRROR of `point_in_edge_loop` (geom2d.rs): the crossing-number test over a closed loop of
// edges, straight or curved alike.
fn point_in_edge_loop(start: u32, count: u32, sample: vec2<f32>) -> bool {
    var crossings = 0u;
    for (var index = 0u; index < count; index = index + 1u) {
        crossings = crossings + edge_crossings(region_edges[start + index], sample);
    }
    return (crossings % 2u) == 1u;
}

// MIRROR of `nearest_boundary_distance` (geom2d.rs): the UNSIGNED distance to the nearest edge.
// The region decides the sign for itself, so the edges are walked once.
fn nearest_boundary_distance(start: u32, count: u32, point: vec2<f32>) -> f32 {
    var nearest = FAR;
    for (var index = 0u; index < count; index = index + 1u) {
        nearest = min(nearest, distance_to_edge(region_edges[start + index], point));
    }
    return nearest;
}

// MIRROR of `signed_distance_to_region` (geom2d.rs): the magnitude is the distance to the nearest
// loop boundary of any role (every boundary of the region is one of those, so the field is zero
// wherever the sign flips), and the sign comes from the FIRST loop containing the point.
//
// The loops arrive innermost-first (smallest area first), so each one governs its own area and
// nothing nested inside it — which is what leaves a picked region standing inside a carved one.
fn signed_distance_to_region(point: vec2<f32>) -> f32 {
    var nearest = FAR;
    var decided = false;
    var inside = false;
    for (var index = 0u; index < uniforms.counts.x; index = index + 1u) {
        let region = region_loops[index];
        nearest = min(nearest, nearest_boundary_distance(region.start, region.count, point));
        if (!decided && point_in_edge_loop(region.start, region.count, point)) {
            decided = true;
            inside = region.role == ROLE_FILL;
        }
    }
    if (inside) {
        return -nearest;
    }
    return nearest;
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
