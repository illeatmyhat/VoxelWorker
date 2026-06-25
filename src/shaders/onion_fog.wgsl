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
// X-ray onion (option B, user-confirmed): the march ignores opaque depth and
// integrates the FULL ray, so the neighbour onion layers show through the
// displayed slice on BOTH sides (the conventional cel-animation "onion skin" =
// see the neighbouring frames through the current one).

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
    // World-space Y range of the ONION band (layers OUTSIDE the displayed band that
    // should fog) and of the displayed solid band (excluded — the opaque pass
    // already drew it). Fog integrates only where world_y is within
    // [onion_y_min, onion_y_max] AND outside [band_y_min, band_y_max].
    onion_y_min: f32,
    onion_y_max: f32,
    band_y_min: f32,
    band_y_max: f32,
};

@group(0) @binding(0) var<uniform> fog: FogUniforms;
// The resolved voxel grid as an R8 occupancy field (1 = solid, 0 = empty),
// trilinear-sampled so the binary grid reads as a smooth cloud density.
@group(0) @binding(1) var occupancy: texture_3d<f32>;
@group(0) @binding(2) var occupancy_sampler: sampler;

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

    // X-ray onion (option B): march the FULL ray and ignore opaque occlusion, so
    // the onion bands show through the displayed slice on both sides.
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
    // Straight (non-premultiplied) colour with `coverage` as alpha; blend state is
    // alpha-over.
    return vec4<f32>(fog.fog_color, coverage);
}
