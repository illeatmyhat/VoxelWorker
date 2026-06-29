// Volumetric onion-skin fog (issue #12) — fullscreen raymarch of the RESOLVED
// voxel grid as a cloud density field.
//
// The onion skin is just the set of voxels in the layers neighbouring the
// displayed band. Rather than re-deriving the parametric shape on the GPU (which
// only works for the 5 built-in SDFs and forces a 1:1 duplicate of the SDF
// library from src/voxel.rs), this pass treats the resolved VoxelGrid as a 3D
// density field — exactly how volumetric clouds are rendered — and raymarches it.
// Trilinear filtering of the binary occupancy yields a smooth density for free,
// so the haze is soft (no cube faces, no SDF iso-cliff) AND it works for ANY
// voxel set, including future sculpt / override producers. This honours the
// resolved-grid seam in REPRESENTATION.md (every consumer reads the grid, never
// the SDF).
//
// Depth-tested like Minecraft's translucent clouds: the displayed solid slice
// occludes the onion layers behind it (the march stops at opaque depth), while the
// neighbour layers in front of and beside the slice still show as haze.

// std140-safe: field order matches the Rust `OnionFogUniforms` struct in
// renderer.rs exactly (112 bytes).
struct FogUniforms {
    // Inverse of the camera view-projection, to unproject screen → world rays.
    inverse_view_projection: mat4x4<f32>,
    // Inscribed semi-axes (= grid_dimensions / 2). The shape is centred at the
    // origin, so world → normalised grid coords is `world / semi_axes * 0.5 + 0.5`.
    semi_axes: vec3<f32>,
    // Overall fog strength (Beer–Lambert coefficient).
    fog_strength: f32,
    // Fog tint (linear RGB).
    fog_color: vec3<f32>,
    _pad0: f32,
    // World-space Z range (Z-up: layers are Z-slices) of the ONION band (layers
    // OUTSIDE the displayed band that should fog) and of the displayed solid band
    // (excluded — the opaque pass already drew it). Fog integrates only where world_z
    // is within [onion_z_min, onion_z_max] AND outside [band_z_min, band_z_max].
    onion_z_min: f32,
    onion_z_max: f32,
    band_z_min: f32,
    band_z_max: f32,
};

@group(0) @binding(0) var<uniform> fog: FogUniforms;
// The resolved voxel grid as an R8 occupancy field (1 = solid, 0 = empty),
// trilinear-sampled so the binary grid reads as a smooth cloud density.
@group(0) @binding(1) var occupancy: texture_3d<f32>;
@group(0) @binding(2) var occupancy_sampler: sampler;
// The opaque 3D pass's MSAA depth (read at sample 0). Like Minecraft's clouds,
// the fog is depth-tested: the march stops at the nearest opaque surface so the
// displayed solid slice OCCLUDES the onion layers behind it.
@group(0) @binding(3) var scene_depth: texture_depth_multisampled_2d;

// Trilinear interpolation reads ~0.5 at a solid/empty voxel boundary, so mapping
// occupancy through this soft window insets the haze edge INWARD — keeping it
// inside the voxel slab's stair-stepped silhouette instead of bleeding past the
// quantised edges (option B inset). Values below FOG_EDGE_LOW read as empty.
const FOG_EDGE_LOW: f32 = 0.35;
const FOG_EDGE_HIGH: f32 = 0.85;

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
    // UV in [0,1], y flipped so (0,0) is top-left.
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

    // The grid occupies only a small slab of the (potentially hundreds of units
    // long) near→far ray, so spending the fixed step budget over the FULL ray would
    // land only a few samples inside the grid — from a top/bottom view the onion
    // band is thin in the view direction and the fog all but vanishes. So clip the
    // ray to the grid's world-space AABB ([-semi_axes, +semi_axes]) and spend every
    // step INSIDE the box, where all the density lives.
    let box_min = -fog.semi_axes;
    let box_max = fog.semi_axes;
    // Robust slab test. A zero ray_direction component yields ±inf t's; the min/max
    // collapse leaves that axis unconstrained iff the origin is within the slab.
    let inv_dir = 1.0 / ray_direction;
    let t_lo = (box_min - ray_origin) * inv_dir;
    let t_hi = (box_max - ray_origin) * inv_dir;
    let t_small = min(t_lo, t_hi);
    let t_big = max(t_lo, t_hi);
    let t_enter = max(max(t_small.x, t_small.y), t_small.z);
    let t_exit = min(min(t_big.x, t_big.y), t_big.z);
    // Miss (or the box is entirely behind the camera): nothing to fog.
    if (t_exit < t_enter || t_exit <= 0.0) {
        discard;
    }
    let march_near = max(t_enter, 0.0);
    var march_far = min(t_exit, ray_length_total);

    // Depth-test like Minecraft's clouds: stop the march at the nearest opaque
    // surface so the displayed solid slice occludes the onion layers behind it.
    // (Fog in FRONT of the slice still shows — only the far side is blocked.)
    let depth_texel = vec2<i32>(input.clip_position.xy);
    let sampled_depth = textureLoad(scene_depth, depth_texel, 0);
    if (sampled_depth < 1.0) {
        let hit_world = unproject(vec3<f32>(ndc_xy, sampled_depth));
        march_far = min(march_far, length(hit_world - ray_origin));
    }
    // Fully behind the opaque surface (or zero-length segment): nothing to fog.
    if (march_far <= march_near) {
        discard;
    }

    // Fixed-step integration of fog density across the in-box segment. (Avoid the
    // name `step` for the local — it shadows the WGSL builtin `step()`.)
    let step_count = 96;
    let step_size = (march_far - march_near) / f32(step_count);
    var optical_thickness = 0.0;
    var t = march_near + step_size * 0.5;
    for (var i = 0; i < step_count; i = i + 1) {
        let sample_point = ray_origin + ray_direction * t;

        // World → normalised grid coords. The shape is centred at the origin and
        // semi_axes = grid_dimensions / 2, so this lands [0,1] across the grid box,
        // with voxel centres at texel centres (trilinear-aligned).
        let grid_uvw = sample_point / fog.semi_axes * 0.5 + vec3<f32>(0.5);
        var density = 0.0;
        // Only sample inside the grid box. Clamp-to-edge would otherwise smear the
        // border voxels (e.g. a box that fills the grid) along the entire ray.
        let inside_box = all(grid_uvw >= vec3<f32>(0.0)) && all(grid_uvw <= vec3<f32>(1.0));
        if (inside_box) {
            density = textureSampleLevel(occupancy, occupancy_sampler, grid_uvw, 0.0).r;
        }
        // Soft, inset density from the trilinear occupancy.
        let inside = smoothstep(FOG_EDGE_LOW, FOG_EDGE_HIGH, density);

        // Vertical onion weight: 0 inside the displayed band (the opaque pass owns
        // it) and BELOW/ABOVE the onion reach, ramping to its peak just outside the
        // band edge and fading smoothly to 0 at the onion extent. This per-layer
        // falloff keeps the fog wispy — the nearest ghost layers read strongest and
        // it never stacks into a solid puck, even for a near-flat slice whose
        // neighbour layers are nearly full cross-sections.
        let z = sample_point.z;
        var vertical = 0.0;
        if (z < fog.band_z_min) {
            // Below the band: distance below the bottom edge, normalised by reach.
            let reach = max(fog.band_z_min - fog.onion_z_min, 1e-4);
            let d = (fog.band_z_min - z) / reach; // 0 at edge → 1 at onion_z_min
            vertical = clamp(1.0 - d, 0.0, 1.0);
        } else if (z > fog.band_z_max) {
            let reach = max(fog.onion_z_max - fog.band_z_max, 1e-4);
            let d = (z - fog.band_z_max) / reach;
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
    // Straight (non-premultiplied) colour with `coverage` as alpha; blend state is
    // alpha-over.
    return vec4<f32>(fog.fog_color, coverage);
}
