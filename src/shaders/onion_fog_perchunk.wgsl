// Per-chunk volumetric onion-skin fog (issue #28 S5a) — the per-chunk variant of
// onion_fog.wgsl.
//
// Identical raymarch, density window, vertical onion weight and Beer–Lambert
// integration as the whole-grid path; the ONLY difference is the occupancy source.
// Instead of one whole-grid 3D texture, the occupancy lives in a small 3D ATLAS:
// one apron'd tile per resident chunk. At each ray sample the shader finds the
// owning chunk, looks up its atlas tile, and trilinear-samples the tile (the
// 1-voxel apron makes the sample seam-smooth across chunk boundaries).
//
// Why this dodges the single-3D-texture limit: the atlas dimension is bounded by
// the chunk COUNT (cbrt of it, times the small per-chunk pad extent), NOT the
// whole-grid voxel extent — so a scene whose whole-grid axis exceeds
// max_texture_dimension_3d (which disables the whole-grid fog) still renders fog.

// Shared camera/band uniform — byte-identical to the whole-grid FogUniforms so the
// SAME OnionFogUniforms buffer binds to both pipelines.
struct FogUniforms {
    inverse_view_projection: mat4x4<f32>,
    semi_axes: vec3<f32>,
    fog_strength: f32,
    fog_color: vec3<f32>,
    _pad0: f32,
    // World-space Z range (Z-up: layers are Z-slices). Matches `OnionFogParams`.
    onion_z_min: f32,
    onion_z_max: f32,
    band_z_min: f32,
    band_z_max: f32,
};

// Per-chunk metadata HEADER (atlas tiling). Matches the Rust `PerChunkFogMeta`
// struct (renderer.rs) exactly. The per-chunk records used to live here as a fixed
// `array<vec4, 1024>`, capping the scene at 1024 non-empty fog chunks (the 64 KiB
// uniform limit); they now live in the `chunk_records` STORAGE buffer (binding 5),
// which is runtime-sized, so the real ceiling is the atlas 3D-texture dimension.
struct PerChunkMeta {
    chunk_count: u32,
    chunk_extent: f32,
    pad_extent: f32,
    tiles_per_axis: u32,
    atlas_dim: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> fog: FogUniforms;
@group(0) @binding(1) var occupancy_atlas: texture_3d<f32>;
@group(0) @binding(2) var occupancy_sampler: sampler;
@group(0) @binding(3) var scene_depth: texture_depth_multisampled_2d;
@group(0) @binding(4) var<uniform> chunk_meta: PerChunkMeta;
// One record per resident chunk: `[world_origin.xyz, tile_index]`. Runtime-sized
// (was a fixed 1024-entry uniform array) → no artificial fog-chunk cap.
@group(0) @binding(5) var<storage, read> chunk_records: array<vec4<f32>>;

const FOG_EDGE_LOW: f32 = 0.35;
const FOG_EDGE_HIGH: f32 = 0.85;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
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

fn unproject(ndc: vec3<f32>) -> vec3<f32> {
    let world = fog.inverse_view_projection * vec4<f32>(ndc, 1.0);
    return world.xyz / world.w;
}

// Sample the per-chunk occupancy at a recentred-world point. Finds the owning chunk
// among the resident records, maps the world point into that chunk's apron'd atlas
// tile, and trilinear-samples it. Returns 0 if the point is in no resident chunk.
fn sample_occupancy(world: vec3<f32>) -> f32 {
    let extent = chunk_meta.chunk_extent;
    // Recentred world → absolute-voxel coords (voxel centres at n+0.5). The whole
    // grid is centred at origin spanning [-semi_axes, +semi_axes].
    let voxel_pos = world + fog.semi_axes; // [0, grid_dim) in voxels
    let chunk_coord = floor(voxel_pos / extent);

    // Find the resident record whose world_origin matches this chunk. world_origin is
    // chunk_coord*extent - semi_axes, so reconstruct and compare. Linear scan over the
    // resident set (bounded by chunk_count); records live in the storage buffer.
    let want_origin = chunk_coord * extent - fog.semi_axes;
    var found = false;
    var tile_index = 0u;
    for (var c = 0u; c < chunk_meta.chunk_count; c = c + 1u) {
        let rec = chunk_records[c];
        let d = abs(rec.xyz - want_origin);
        if (d.x < 0.25 && d.y < 0.25 && d.z < 0.25) {
            tile_index = u32(rec.w + 0.5);
            found = true;
            break;
        }
    }
    if (!found) {
        return 0.0;
    }

    // Local continuous voxel coordinate within the chunk's interior [0, extent).
    let local = world - want_origin; // == voxel_pos - chunk_coord*extent
    // Tile's 3D coordinate in the atlas (linear tile_index → tx,ty,tz).
    let tpa = chunk_meta.tiles_per_axis;
    let tx = tile_index % tpa;
    let ty = (tile_index / tpa) % tpa;
    let tz = tile_index / (tpa * tpa);
    let tile_base = vec3<f32>(f32(tx), f32(ty), f32(tz)) * chunk_meta.pad_extent;
    // The apron occupies atlas texel 0 of the tile; interior voxel i sits at texel
    // (1 + i). A continuous local coord maps to texel (tile_base + 1 + local), then
    // normalise by atlas_dim for the [0,1] sampler coordinate. Trilinear then blends
    // against the apron texel at the boundary → seam-smooth.
    let atlas_texel = tile_base + vec3<f32>(1.0) + local;
    let uvw = atlas_texel / chunk_meta.atlas_dim;
    return textureSampleLevel(occupancy_atlas, occupancy_sampler, uvw, 0.0).r;
}

@fragment
fn fragment_main(input: VsOut) -> @location(0) vec4<f32> {
    let ndc_xy = vec2<f32>(input.uv.x * 2.0 - 1.0, (1.0 - input.uv.y) * 2.0 - 1.0);

    let near_world = unproject(vec3<f32>(ndc_xy, 0.0));
    let far_world = unproject(vec3<f32>(ndc_xy, 1.0));
    let ray_origin = near_world;
    let ray_full = far_world - near_world;
    let ray_length_total = length(ray_full);
    let ray_direction = ray_full / max(ray_length_total, 1e-6);

    // Clip to the whole grid's world AABB (same as the whole-grid path): all the
    // density lives inside [-semi_axes, +semi_axes], so spend the step budget there.
    let box_min = -fog.semi_axes;
    let box_max = fog.semi_axes;
    let inv_dir = 1.0 / ray_direction;
    let t_lo = (box_min - ray_origin) * inv_dir;
    let t_hi = (box_max - ray_origin) * inv_dir;
    let t_small = min(t_lo, t_hi);
    let t_big = max(t_lo, t_hi);
    let t_enter = max(max(t_small.x, t_small.y), t_small.z);
    let t_exit = min(min(t_big.x, t_big.y), t_big.z);
    if (t_exit < t_enter || t_exit <= 0.0) {
        discard;
    }
    let march_near = max(t_enter, 0.0);
    var march_far = min(t_exit, ray_length_total);

    let depth_texel = vec2<i32>(input.clip_position.xy);
    let sampled_depth = textureLoad(scene_depth, depth_texel, 0);
    if (sampled_depth < 1.0) {
        let hit_world = unproject(vec3<f32>(ndc_xy, sampled_depth));
        march_far = min(march_far, length(hit_world - ray_origin));
    }
    if (march_far <= march_near) {
        discard;
    }

    let step_count = 96;
    let step_size = (march_far - march_near) / f32(step_count);
    var optical_thickness = 0.0;
    var t = march_near + step_size * 0.5;
    for (var i = 0; i < step_count; i = i + 1) {
        let sample_point = ray_origin + ray_direction * t;
        let grid_uvw = sample_point / fog.semi_axes * 0.5 + vec3<f32>(0.5);
        var density = 0.0;
        let inside_box = all(grid_uvw >= vec3<f32>(0.0)) && all(grid_uvw <= vec3<f32>(1.0));
        if (inside_box) {
            density = sample_occupancy(sample_point);
        }
        let inside = smoothstep(FOG_EDGE_LOW, FOG_EDGE_HIGH, density);

        let z = sample_point.z;
        var vertical = 0.0;
        if (z < fog.band_z_min) {
            let reach = max(fog.band_z_min - fog.onion_z_min, 1e-4);
            let d = (fog.band_z_min - z) / reach;
            vertical = clamp(1.0 - d, 0.0, 1.0);
        } else if (z > fog.band_z_max) {
            let reach = max(fog.onion_z_max - fog.band_z_max, 1e-4);
            let d = (z - fog.band_z_max) / reach;
            vertical = clamp(1.0 - d, 0.0, 1.0);
        }
        optical_thickness = optical_thickness + inside * vertical * step_size;
        t = t + step_size;
    }

    let coverage = 1.0 - exp(-optical_thickness * fog.fog_strength);
    if (coverage < 0.002) {
        discard;
    }
    return vec4<f32>(fog.fog_color, coverage);
}
