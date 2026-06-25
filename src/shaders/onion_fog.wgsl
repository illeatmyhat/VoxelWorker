// Volumetric onion-skin fog (issue #12) — fullscreen SDF raymarch.
//
// The onion skin is no longer drawn as per-voxel translucent cubes (which showed
// cube faces/edges and wrongly treated a rough/concave volume as one solid
// shell). Instead this fullscreen pass raymarches the SAME parametric SDF the
// CPU producer uses, integrating a smooth fog density wherever the shape is
// SOLID *and* the world-Y falls in the onion-skin layer band OUTSIDE the
// displayed band. The result is true optical-thickness fog: a ray that crosses
// material → gap → material (e.g. a torus side-on, or the near + far walls of a
// hollow shape) accumulates correctly, with NO voxel quantization and no edges.
//
// Putting the SDF on the GPU is fine here (the user lifted the CPU-only
// assumption): the shapes are cheap closed-form SDFs and there is no fidelity
// loss versus the resolved grid for a smooth haze.
//
// The pass:
//   * Draws a single full-screen triangle (no vertex buffer).
//   * Reconstructs each pixel's world-space view ray from the inverse
//     view-projection.
//   * Marches the ray from the near plane up to the SCENE depth (sampled from the
//     resolved opaque depth) so the solid band correctly occludes fog behind it.
//   * Integrates fog density (Beer–Lambert) and composites OVER the scene colour.

// std140-safe: each vec3 is followed by a scalar; field order matches the Rust
// `OnionFogUniforms` struct in renderer.rs exactly (128 bytes).
struct FogUniforms {
    // Inverse of the camera view-projection, to unproject screen → world rays.
    inverse_view_projection: mat4x4<f32>,
    // Inscribed semi-axes (voxel-space half-extents) of the shape.
    semi_axes: vec3<f32>,
    // Shape selector: 0 cylinder, 1 tube, 2 sphere, 3 torus, 4 box.
    shape_kind: f32,
    // Fog tint (linear RGB) and tube wall thickness in voxels (tube only).
    fog_color: vec3<f32>,
    wall_voxels: f32,
    // The world-space Y range of the ONION band (layers OUTSIDE the displayed
    // band that should fog), in world units (voxel-centred). Fog is integrated
    // only where world_y is within [onion_y_min, onion_y_max] AND outside
    // [band_y_min, band_y_max] (the solid band the opaque pass already drew).
    onion_y_min: f32,
    onion_y_max: f32,
    band_y_min: f32,
    band_y_max: f32,
    // Overall fog strength (Beer–Lambert coefficient) + padding to 16 bytes.
    fog_strength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> fog: FogUniforms;
// The opaque 3D pass's 4× MSAA depth buffer, read directly (sample 0) via
// `textureLoad` — no resolve pass needed. The march stops at this nearest opaque
// surface so the solid band correctly occludes fog behind it.
@group(0) @binding(1) var scene_depth: texture_depth_multisampled_2d;

// Voxels of inset applied to the fog's soft edge (option B): keeps the smooth haze
// inside the voxel slab's quantised silhouette rather than poking past its edges.
const FOG_EDGE_INSET: f32 = 0.75;

// ---- SDF library (ported 1:1 from src/voxel.rs) ----

fn sd_box(point: vec3<f32>, box_half: vec3<f32>) -> f32 {
    let q = abs(point) - box_half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_ellipsoid(point: vec3<f32>, semi_axes: vec3<f32>) -> f32 {
    let scaled = point / semi_axes;
    let k0 = length(scaled);
    if (k0 == 0.0) {
        return -min(semi_axes.x, min(semi_axes.y, semi_axes.z));
    }
    let scaled_sq = point / (semi_axes * semi_axes);
    let k1 = length(scaled_sq);
    return k0 * (k0 - 1.0) / k1;
}

fn sd_elliptical_cylinder(point: vec3<f32>, semi_axis_x: f32, semi_axis_z: f32, half_height: f32) -> f32 {
    let radial = (length(vec2<f32>(point.x / semi_axis_x, point.z / semi_axis_z)) - 1.0)
        * min(semi_axis_x, semi_axis_z);
    let vertical = abs(point.y) - half_height;
    return min(max(radial, vertical), 0.0)
        + length(vec2<f32>(max(radial, 0.0), max(vertical, 0.0)));
}

// Dispatch matching src/voxel.rs `signed_distance`.
fn scene_sdf(point: vec3<f32>) -> f32 {
    let ax = fog.semi_axes.x;
    let ay = fog.semi_axes.y;
    let az = fog.semi_axes.z;
    let kind = i32(fog.shape_kind + 0.5);
    if (kind == 0) {
        // Cylinder.
        return sd_elliptical_cylinder(point, ax, az, ay);
    } else if (kind == 1) {
        // Tube: outer cylinder minus inner cylinder.
        let outer = sd_elliptical_cylinder(point, ax, az, ay);
        let inner = sd_elliptical_cylinder(
            point,
            max(ax - fog.wall_voxels, 0.01),
            max(az - fog.wall_voxels, 0.01),
            ay + 1.0,
        );
        return max(outer, -inner);
    } else if (kind == 2) {
        // Sphere (ellipsoid).
        return sd_ellipsoid(point, fog.semi_axes);
    } else if (kind == 3) {
        // Torus.
        let tube_radius = ay;
        let ring_radius = max(min(ax, az) - tube_radius, 0.0);
        let radial = length(vec2<f32>(point.x, point.z)) - ring_radius;
        return length(vec2<f32>(radial, point.y)) - tube_radius;
    } else {
        // Box.
        return sd_box(point, fog.semi_axes);
    }
}

// ---- Fullscreen triangle ----

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    // A single oversized triangle covering the viewport.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let p = positions[vertex_index];
    var out: VsOut;
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    // UV in [0,1], y flipped so (0,0) is top-left like the depth texture.
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

// Unproject an NDC point (z in [0,1]) to world space.
fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let world = fog.inverse_view_projection * vec4<f32>(ndc, 1.0);
    return world.xyz / world.w;
}

@fragment
fn fragment_main(input: VsOut) -> @location(0) vec4<f32> {
    let ndc_xy = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);

    // Reconstruct the world-space view ray for this pixel.
    let near_world = unproject(vec3<f32>(ndc_xy, 0.0));
    let far_world = unproject(vec3<f32>(ndc_xy, 1.0));
    let ray_origin = near_world;
    let ray_full = far_world - near_world;
    let ray_length_total = length(ray_full);
    let ray_direction = ray_full / max(ray_length_total, 1e-6);

    // X-ray onion (option B, user-confirmed): march the FULL ray and deliberately
    // ignore the opaque slab's depth occlusion. The conventional cel-animation
    // "onion skin" lets you see the neighbour layers *through* the current frame, so
    // the ghost layers must show on BOTH sides of the displayed slice — including
    // the band that sits behind the solid slab from the camera's view. (`scene_depth`
    // stays bound for a possible future occluded mode but is unused here.)
    let march_far = ray_length_total;

    // Fixed-step integration of fog density along the ray. The step count trades
    // quality for cost; the band is thin so a modest count suffices. (Avoid the
    // name `step` for the local — it shadows the WGSL builtin `step()`.)
    let step_count = 96;
    let step_size = march_far / f32(step_count);
    var optical_thickness = 0.0;
    var t = step_size * 0.5;
    for (var i = 0; i < step_count; i = i + 1) {
        let sample_point = ray_origin + ray_direction * t;
        // Inside the shape?
        let distance = scene_sdf(sample_point);
        // Smooth density: 1 well inside the surface, fading to 0 at/just outside,
        // so the fog edge is soft (no hard SDF iso-cliff). The FOG_EDGE_INSET pushes
        // the soft edge INWARD from the ideal SDF surface so the smooth haze stays
        // inside the voxel slab's stair-stepped silhouette instead of undercutting /
        // haloing past its quantised edges (option B inset).
        let inside = smoothstep(0.5, -0.5, distance + FOG_EDGE_INSET);

        // Vertical onion weight: 0 inside the displayed band (the opaque pass owns
        // it) and BELOW/ABOVE the onion reach, ramping to its peak just outside the
        // band edge and fading smoothly to 0 at the onion extent. This per-layer
        // falloff keeps the fog wispy — the nearest ghost layers read strongest and
        // it never stacks into a solid puck, even for a near-flat slice whose
        // neighbour layers are nearly full cross-sections.
        let y = sample_point.y;
        var vertical = 0.0;
        if (y < fog.band_y_min) {
            // Below the band: distance below the bottom edge, normalised by reach.
            let reach = max(fog.band_y_min - fog.onion_y_min, 1e-4);
            let d = (fog.band_y_min - y) / reach; // 0 at edge → 1 at onion_y_min
            vertical = clamp(1.0 - d, 0.0, 1.0);
        } else if (y > fog.band_y_max) {
            let reach = max(fog.onion_y_max - fog.band_y_max, 1e-4);
            let d = (y - fog.band_y_max) / reach;
            vertical = clamp(1.0 - d, 0.0, 1.0);
        }
        optical_thickness = optical_thickness + inside * vertical * step_size;
        t = t + step_size;
    }

    // Beer–Lambert: convert accumulated thickness to coverage. Faint by design so
    // the band shows through (aerogel-like).
    let coverage = 1.0 - exp(-optical_thickness * fog.fog_strength);
    if (coverage < 0.002) {
        discard;
    }
    // Premultiplied OVER composite: blend state is src.a-driven alpha-over, so
    // return straight (non-premultiplied) colour with `coverage` as alpha.
    return vec4<f32>(fog.fog_color, coverage);
}
