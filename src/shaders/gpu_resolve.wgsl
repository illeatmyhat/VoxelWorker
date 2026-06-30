// GPU view-resolve P1 spike (ADR 0007): evaluate a producer's voxel occupancy on
// the GPU, per chunk, over the same apron'd field the CPU per-chunk fog builds.
//
// This shader is the GPU side of the CPU↔GPU A/B equivalence net (ADR 0007 §5/§6).
// It does NOT replace the CPU authoritative resolve; it re-evaluates the SAME
// occupancy predicate at the SAME integer voxel sample points so the result can be
// asserted byte-identical to `build_per_chunk_fog_occupancy`. The math below mirrors
// `signed_distance*` (src/voxel.rs) and `resolve_extrude` / `resolve_revolve` /
// `point_in_polygon` (src/sketch.rs) op-for-op to keep the occupancy decision in
// lockstep with the CPU. CPU uses f64 in the polygon test; WGSL core has only f32,
// so the sketch tiers are exactly where §6's exact-parity question is measured.
//
// Layout (must match the Rust `Descriptor` in `src/gpu_resolve.rs`): vec3 fields are
// padded to vec4 for uniform-buffer alignment; unused lanes are zero.

struct Descriptor {
    // Producer sampling-grid dimensions in voxels (x, y, z).
    grid: vec4<i32>,
    // The integer subtracted from a fog-global voxel coord to recover the producer's
    // [0, grid) local voxel index. In the RECENTRED frame `build_per_chunk_fog_occupancy`
    // consumes, the composite recentre and the fog's floor(grid/2) decode cancel for a
    // lone untranslated producer, so this is [0,0,0]; kept as a uniform so a translated /
    // multi-leaf producer can carry its own offset later.
    local_offset: vec4<i32>,
    // grid as f32 / 2.0 per axis — the inscribed SDF semi-axes AND the revolve centring
    // half (`idx + 0.5 - grid/2.0`).
    semi_axes: vec4<f32>,
    // Sketch integer params packed: [extrude_min0, extrude_min1, revolve_axial_min,
    // revolve_radial_max]. Unused lanes are zero per producer type.
    profile_ints: vec4<i32>,
    // Sketch world-axis params: [in_plane_0, in_plane_1, normal, revolve_axial_world_axis].
    sketch_axes: vec4<u32>,
    // Revolve params: [radial_a, radial_b, revolve_is_inplane0, profile_straddles_axis].
    revolve_axes: vec4<u32>,
    // DebugClouds fBm constants: [edge_billow, wavelength_fraction, lacunarity, gain].
    cloud_params: vec4<f32>,
    // 0 = SDF primitive, 1 = sketch extrude, 2 = sketch revolve, 3 = debug clouds.
    producer_type: u32,
    // ShapeKind discriminant (producer_type 0): 0=Cyl 1=Tube 2=Sphere 3=Torus 4=Box.
    kind: u32,
    // wall_blocks * voxels_per_block (Tube only).
    wall_voxels: f32,
    // Revolve sweep angle in degrees (producer_type 2).
    turn: f32,
    // Number of profile vertices (sketch producers).
    profile_count: u32,
    // 1 if the revolve is a partial turn (< 360°), else 0.
    is_partial: u32,
    // CHUNK_BLOCKS * voxels_per_block — one chunk's voxel extent per axis.
    chunk_extent: i32,
    // chunk_extent + 2 — the apron'd per-axis span (apron index 0 == local -1).
    pad: u32,
    // Number of chunk volumes to evaluate.
    num_chunks: u32,
    // Atlas packing (the `main_atlas` entry only): tiles per axis, the atlas cube
    // dimension (tiles_per_axis * pad), and the 256-aligned bytes-per-row stride of the
    // packed-byte buffer (a copy_buffer_to_texture requirement).
    tiles_per_axis: u32,
    atlas_dim: u32,
    padded_row: u32,
    // DebugClouds: fBm octave count + number of cloud puffs in `cloud_puffs`.
    cloud_octaves: u32,
    num_puffs: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> desc: Descriptor;
@group(0) @binding(1) var<storage, read> chunk_coords: array<vec4<i32>>;
// Read-write occupancy. The `main` (A/B) entry writes one u32 (0/255) per apron cell;
// the `main_atlas` entry packs occupancy BYTES into the atlas via `atomicOr` (4 cells
// share a u32 word, written concurrently). Typed atomic so both entries share the layout.
@group(0) @binding(2) var<storage, read_write> occupancy: array<atomic<u32>>;
// Sketch profile vertices (in-plane voxel coords); one dummy element for SDF cases.
@group(0) @binding(3) var<storage, read> profile: array<vec2<i32>>;
// DebugClouds puffs: 2 vec4 per puff — [center.xyz, radius], [noise_offset.xyz, _].
// One dummy element for non-cloud producers.
@group(0) @binding(4) var<storage, read> cloud_puffs: array<vec4<f32>>;
// DebugClouds Perlin permutation table (512 seed-shuffled entries, 0..255). One dummy
// element for non-cloud producers.
@group(0) @binding(5) var<storage, read> cloud_perm: array<u32>;
// Per-chunk "has ≥1 INTERIOR occupied voxel" flag (ADR 0007 residency option C′). Used
// by the atlas path ONLY: `main_flags` sets it, `main_atlas` reads it to suppress an
// interior-empty chunk's apron writes so a COVERING tile renders identically to "no tile"
// (the CPU non-empty-set residency). Bound only on the atlas/flags bind group layout.
@group(0) @binding(6) var<storage, read_write> chunk_flags: array<atomic<u32>>;

// --- SDF primitives (mirror src/voxel.rs `signed_distance_*`, glam f32) ---

fn sd_box(point: vec3<f32>, box_half: vec3<f32>) -> f32 {
    let q = abs(point) - box_half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sd_ellipsoid(point: vec3<f32>, semi_axes: vec3<f32>) -> f32 {
    let scaled = point / semi_axes;
    let distance_to_unit = length(scaled);
    if (distance_to_unit == 0.0) {
        return -min(semi_axes.x, min(semi_axes.y, semi_axes.z));
    }
    let scaled_squared = point / (semi_axes * semi_axes);
    let gradient = length(scaled_squared);
    return distance_to_unit * (distance_to_unit - 1.0) / gradient;
}

fn sd_elliptical_cylinder(
    point: vec3<f32>,
    semi_axis_x: f32,
    semi_axis_y: f32,
    half_height: f32,
) -> f32 {
    let radial = (length(vec2<f32>(point.x / semi_axis_x, point.y / semi_axis_y)) - 1.0)
        * min(semi_axis_x, semi_axis_y);
    let vertical = abs(point.z) - half_height;
    return min(max(radial, vertical), 0.0)
        + length(vec2<f32>(max(radial, 0.0), max(vertical, 0.0)));
}

fn signed_distance(point: vec3<f32>, semi_axes: vec3<f32>, wall_voxels: f32) -> f32 {
    let semi_axis_x = semi_axes.x;
    let semi_axis_y = semi_axes.y;
    let semi_axis_z = semi_axes.z;

    switch (desc.kind) {
        case 0u: { // Cylinder
            return sd_elliptical_cylinder(point, semi_axis_x, semi_axis_y, semi_axis_z);
        }
        case 1u: { // Tube
            let outer = sd_elliptical_cylinder(point, semi_axis_x, semi_axis_y, semi_axis_z);
            let inner = sd_elliptical_cylinder(
                point,
                max(semi_axis_x - wall_voxels, 0.01),
                max(semi_axis_y - wall_voxels, 0.01),
                semi_axis_z + 1.0,
            );
            return max(outer, -inner);
        }
        case 2u: { // Sphere
            return sd_ellipsoid(point, semi_axes);
        }
        case 3u: { // Torus
            let tube_radius = semi_axis_z;
            let ring_radius = max(min(semi_axis_x, semi_axis_y) - tube_radius, 0.0);
            let radial = length(vec2<f32>(point.x, point.y)) - ring_radius;
            return length(vec2<f32>(radial, point.z)) - tube_radius;
        }
        case 4u: { // Box
            return sd_box(point, semi_axes);
        }
        default: {
            return 1.0;
        }
    }
}

// --- Sketch profile test (mirror src/sketch.rs `point_in_polygon`, CPU is f64) ---

// Even-odd crossing-number test in the profile's own coordinate space. CPU computes
// this in f64; here in f32 (no portable f64 in WGSL) — the measured divergence surface.
fn point_in_polygon(sample_0: f32, sample_1: f32) -> bool {
    var inside = false;
    let count = desc.profile_count;
    var previous = count - 1u;
    for (var current = 0u; current < count; current = current + 1u) {
        let current_point = vec2<f32>(profile[current]);
        let previous_point = vec2<f32>(profile[previous]);
        let current_0 = current_point.x;
        let current_1 = current_point.y;
        let previous_0 = previous_point.x;
        let previous_1 = previous_point.y;
        let straddles = (current_1 > sample_1) != (previous_1 > sample_1);
        if (straddles) {
            let crossing_0 = (previous_0 - current_0) * (sample_1 - current_1)
                / (previous_1 - current_1)
                + current_0;
            if (sample_0 < crossing_0) {
                inside = !inside;
            }
        }
        previous = current;
    }
    return inside;
}

// Revolve: reconstruct the profile point for a signed radius and test it.
fn revolve_inside(signed_radius: f32, profile_axial: f32, is_inplane0: bool) -> bool {
    var sample_0: f32;
    var sample_1: f32;
    if (is_inplane0) {
        sample_0 = profile_axial;
        sample_1 = signed_radius;
    } else {
        sample_0 = signed_radius;
        sample_1 = profile_axial;
    }
    return point_in_polygon(sample_0, sample_1);
}

// --- DebugClouds: Perlin fBm + per-puff radial falloff (mirror src/debug_clouds.rs) ---

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn nlerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

// Perlin's 12-direction gradient (mirror `grad`, hash low bits select the edge).
fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    var u: f32;
    if (h < 8u) { u = x; } else { u = y; }
    var v: f32;
    if (h < 4u) { v = y; } else if (h == 12u || h == 14u) { v = x; } else { v = z; }
    var u_term: f32;
    if ((h & 1u) == 0u) { u_term = u; } else { u_term = -u; }
    var v_term: f32;
    if ((h & 2u) == 0u) { v_term = v; } else { v_term = -v; }
    return u_term + v_term;
}

fn perm(index: u32) -> u32 {
    return cloud_perm[index];
}

// Improved-Perlin 3D noise (mirror `PerlinNoise::noise`).
fn perlin_noise(point: vec3<f32>) -> f32 {
    let xi = floor(point.x);
    let yi = floor(point.y);
    let zi = floor(point.z);
    let cube_x = u32(i32(xi) & 255);
    let cube_y = u32(i32(yi) & 255);
    let cube_z = u32(i32(zi) & 255);

    let fx = point.x - xi;
    let fy = point.y - yi;
    let fz = point.z - zi;

    let u = fade(fx);
    let v = fade(fy);
    let w = fade(fz);

    let a = perm(cube_x) + cube_y;
    let aa = perm(a) + cube_z;
    let ab = perm(a + 1u) + cube_z;
    let b = perm(cube_x + 1u) + cube_y;
    let ba = perm(b) + cube_z;
    let bb = perm(b + 1u) + cube_z;

    let x1 = nlerp(grad(perm(aa), fx, fy, fz), grad(perm(ba), fx - 1.0, fy, fz), u);
    let x2 = nlerp(grad(perm(ab), fx, fy - 1.0, fz), grad(perm(bb), fx - 1.0, fy - 1.0, fz), u);
    let y1 = nlerp(x1, x2, v);

    let x3 = nlerp(grad(perm(aa + 1u), fx, fy, fz - 1.0), grad(perm(ba + 1u), fx - 1.0, fy, fz - 1.0), u);
    let x4 = nlerp(grad(perm(ab + 1u), fx, fy - 1.0, fz - 1.0), grad(perm(bb + 1u), fx - 1.0, fy - 1.0, fz - 1.0), u);
    let y2 = nlerp(x3, x4, v);

    return nlerp(y1, y2, w);
}

// fBm: summed octaves, normalised (mirror `PerlinNoise::fractal_noise`).
fn fractal_noise(point: vec3<f32>) -> f32 {
    let octaves = desc.cloud_octaves;
    let lacunarity = desc.cloud_params.z;
    let gain = desc.cloud_params.w;
    var frequency = 1.0;
    var amplitude = 1.0;
    var sum = 0.0;
    var normalization = 0.0;
    for (var i = 0u; i < octaves; i = i + 1u) {
        sum = sum + amplitude * perlin_noise(point * frequency);
        normalization = normalization + amplitude;
        amplitude = amplitude * gain;
        frequency = frequency * lacunarity;
    }
    if (normalization == 0.0) {
        return 0.0;
    }
    return sum / normalization;
}

// Whether the centred sample lands in any puff (mirror `cloud_field_is_solid`).
fn cloud_field_is_solid(point: vec3<f32>) -> bool {
    let edge_billow = desc.cloud_params.x;
    let wavelength_fraction = desc.cloud_params.y;
    for (var p = 0u; p < desc.num_puffs; p = p + 1u) {
        let center = cloud_puffs[2u * p].xyz;
        let radius = cloud_puffs[2u * p].w;
        let noise_offset = cloud_puffs[2u * p + 1u].xyz;
        let dist = length(point - center);
        let radial = 1.0 - dist / radius;
        if (radial < -edge_billow) {
            continue;
        }
        let wavelength = radius * wavelength_fraction;
        let frequency = 1.0 / max(wavelength, 1.0);
        let billow = fractal_noise((point + noise_offset) * frequency);
        if (radial + edge_billow * billow > 0.0) {
            return true;
        }
    }
    return false;
}

// Occupancy of a single producer-local voxel index. Returns 255u (inside) or 0u.
fn evaluate(voxel_index: vec3<i32>) -> u32 {
    var vi = array<i32, 3>(voxel_index.x, voxel_index.y, voxel_index.z);

    switch (desc.producer_type) {
        // SDF primitive
        case 0u: {
            let sample = vec3<f32>(voxel_index) + vec3<f32>(0.5) - desc.semi_axes.xyz;
            if (signed_distance(sample, desc.semi_axes.xyz, desc.wall_voxels) <= 0.0) {
                return 255u;
            }
            return 0u;
        }
        // Sketch extrude: the in-plane cell tested against the profile, swept on the normal.
        case 1u: {
            let in_plane_0 = desc.sketch_axes.x;
            let in_plane_1 = desc.sketch_axes.y;
            let sample_0 = f32(desc.profile_ints.x + vi[in_plane_0]) + 0.5;
            let sample_1 = f32(desc.profile_ints.y + vi[in_plane_1]) + 0.5;
            if (point_in_polygon(sample_0, sample_1)) {
                return 255u;
            }
            return 0u;
        }
        // Sketch revolve.
        case 2u: {
            let half = desc.semi_axes.xyz;
            var centred = array<f32, 3>(
                f32(vi[0]) + 0.5 - half.x,
                f32(vi[1]) + 0.5 - half.y,
                f32(vi[2]) + 0.5 - half.z,
            );
            let radial_a = desc.revolve_axes.x;
            let radial_b = desc.revolve_axes.y;
            let ca = centred[radial_a];
            let cb = centred[radial_b];
            let radial = sqrt(ca * ca + cb * cb);

            if (desc.is_partial != 0u) {
                var theta = degrees(atan2(cb, ca));
                if (theta < 0.0) {
                    theta = theta + 360.0;
                }
                if (theta > desc.turn) {
                    return 0u;
                }
            }

            // Radial early-out: a cell past the farthest profile vertex can't be inside.
            if (radial > f32(desc.profile_ints.w)) {
                return 0u;
            }

            let axial_world = desc.sketch_axes.w;
            let profile_axial = f32(desc.profile_ints.z + vi[axial_world]) + 0.5;
            let is_inplane0 = desc.revolve_axes.z != 0u;
            let straddles = desc.revolve_axes.w != 0u;

            var is_inside = revolve_inside(radial, profile_axial, is_inplane0);
            if (straddles && !is_inside) {
                is_inside = revolve_inside(-radial, profile_axial, is_inplane0);
            }
            if (is_inside) {
                return 255u;
            }
            return 0u;
        }
        // DebugClouds: Perlin fBm cloud field at the centred sample.
        case 3u: {
            let sample = vec3<f32>(voxel_index) + vec3<f32>(0.5) - desc.semi_axes.xyz;
            if (cloud_field_is_solid(sample)) {
                return 255u;
            }
            return 0u;
        }
        default: {
            return 0u;
        }
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let pad = desc.pad;
    let cells_per_chunk = pad * pad * pad;
    let total = desc.num_chunks * cells_per_chunk;
    if (linear >= total) {
        return;
    }

    // Decode linear index → (chunk, ak, aj, ai) apron coordinate.
    let chunk = linear / cells_per_chunk;
    var rem = linear % cells_per_chunk;
    let ak = rem / (pad * pad);
    rem = rem % (pad * pad);
    let aj = rem / pad;
    let ai = rem % pad;

    // Apron index a maps to chunk-local voxel coord (a - 1) ∈ [-1, chunk_extent].
    let coord = chunk_coords[chunk].xyz;
    let chunk_min = coord * desc.chunk_extent;
    let g = chunk_min + vec3<i32>(i32(ai), i32(aj), i32(ak)) - vec3<i32>(1, 1, 1);
    let voxel_index = g - desc.local_offset.xyz;

    var occupied = 0u;
    if (all(voxel_index >= vec3<i32>(0, 0, 0)) && all(voxel_index < desc.grid.xyz)) {
        occupied = evaluate(voxel_index);
    }

    atomicStore(&occupancy[linear], occupied);
}

// Phase 1 of the atlas path (ADR 0007 C′): per chunk, OR a 1 into `chunk_flags[chunk]`
// iff this cell is an INTERIOR voxel (apron index ∈ [1, chunk_extent], i.e. local ∈
// [0, chunk_extent)) AND occupied. `main_atlas` then gates its writes on this flag so a
// covering chunk with no interior occupancy (only an apron grazing the surface) stays an
// all-zero tile — reproducing the CPU non-empty-set render without any densify/readback.
@compute @workgroup_size(64)
fn main_flags(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let pad = desc.pad;
    let cells_per_chunk = pad * pad * pad;
    let total = desc.num_chunks * cells_per_chunk;
    if (linear >= total) {
        return;
    }

    let chunk = linear / cells_per_chunk;
    var rem = linear % cells_per_chunk;
    let ak = rem / (pad * pad);
    rem = rem % (pad * pad);
    let aj = rem / pad;
    let ai = rem % pad;

    // Interior iff every apron index is in [1, chunk_extent] (local voxel ∈ [0, extent)).
    let ext = u32(desc.chunk_extent);
    let interior = ai >= 1u && ai <= ext && aj >= 1u && aj <= ext && ak >= 1u && ak <= ext;
    if (!interior) {
        return;
    }

    let coord = chunk_coords[chunk].xyz;
    let chunk_min = coord * desc.chunk_extent;
    let g = chunk_min + vec3<i32>(i32(ai), i32(aj), i32(ak)) - vec3<i32>(1, 1, 1);
    let voxel_index = g - desc.local_offset.xyz;
    if (all(voxel_index >= vec3<i32>(0, 0, 0)) && all(voxel_index < desc.grid.xyz)) {
        if (evaluate(voxel_index) != 0u) {
            atomicOr(&chunk_flags[chunk], 1u);
        }
    }
}

// Pack the same per-chunk apron occupancy into the OnionFogRenderer atlas byte layout
// (mirrors `upload_grid_per_chunk`'s tile placement), as PACKED BYTES in a 256-padded-row
// buffer ready for `copy_buffer_to_texture` into the R8 atlas. One invocation per apron
// cell; the byte is OR'd into its u32 word (the buffer is cleared to 0 first).
@compute @workgroup_size(64)
fn main_atlas(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear = global_id.x;
    let pad = desc.pad;
    let cells_per_chunk = pad * pad * pad;
    let total = desc.num_chunks * cells_per_chunk;
    if (linear >= total) {
        return;
    }

    let chunk = linear / cells_per_chunk;
    var rem = linear % cells_per_chunk;
    let ak = rem / (pad * pad);
    rem = rem % (pad * pad);
    let aj = rem / pad;
    let ai = rem % pad;

    let coord = chunk_coords[chunk].xyz;
    let chunk_min = coord * desc.chunk_extent;
    let g = chunk_min + vec3<i32>(i32(ai), i32(aj), i32(ak)) - vec3<i32>(1, 1, 1);
    let voxel_index = g - desc.local_offset.xyz;

    var occupied = 0u;
    if (all(voxel_index >= vec3<i32>(0, 0, 0)) && all(voxel_index < desc.grid.xyz)) {
        occupied = evaluate(voxel_index);
    }
    if (occupied == 0u) {
        return; // byte stays 0; nothing to OR
    }
    // ADR 0007 C′: suppress an interior-empty covering chunk's apron writes (its tile
    // stays all-zero → renders as "no tile", matching the CPU non-empty-set residency).
    // `main_flags` populated `chunk_flags` in a prior pass; an interior occupied cell
    // always set its own chunk's flag, so it is never wrongly suppressed here.
    if (atomicLoad(&chunk_flags[chunk]) == 0u) {
        return;
    }

    // Tile slot of this chunk in the cubic atlas (linear tile index → 3D tile coord),
    // matching `upload_grid_per_chunk`.
    let tpa = desc.tiles_per_axis;
    let tx = chunk % tpa;
    let ty = (chunk / tpa) % tpa;
    let tz = chunk / (tpa * tpa);
    let ax = tx * pad + ai;
    let ay = ty * pad + aj;
    let az = tz * pad + ak;

    // Byte offset in the 256-padded-row buffer; OR the 0xFF byte into its u32 word.
    let byte_index = (az * desc.atlas_dim + ay) * desc.padded_row + ax;
    let word = byte_index / 4u;
    let shift = (byte_index % 4u) * 8u;
    atomicOr(&occupancy[word], occupied << shift);
}
