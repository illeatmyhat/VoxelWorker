// Instanced unit-cube voxel shader (Milestone 4).
//
// Each instance is one occupied voxel. The unit cube has per-face normals and
// per-face base UVs (24 vertices / 36 indices) so faces shade and texture
// independently.
//
// M4 adds three things on top of the M2 flat shading:
//   * BUG 1 fix — per-voxel texture slice. One procedural texture belongs to the
//     whole BLOCK; the vertex stage offsets the base face UV by the voxel's
//     `block_local_coord` and divides by `voxels_per_block`, so each voxel face
//     shows only its 1/density slice (NOT the whole texture repeated per cube).
//   * BUG 2 fix — grid overlay derived from the fragment's ABSOLUTE voxel
//     position (world position + grid half-extent), not from face UVs. Cube faces
//     flip UV direction, so a UV-derived block boundary lands one voxel off on
//     vertical faces; the world-position form is orientation-independent.
//   * Directional + ambient lighting applied to the sampled texture colour.

// std140-safe uniform block. Each vec3 is followed by a scalar so the vec3 never
// straddles a 16-byte boundary. Field order matches `VoxelUniforms` in renderer.rs.
struct VoxelUniforms {
    view_projection: mat4x4<f32>,
    grid_half_extent: vec3<f32>,
    voxels_per_block: f32,
    voxel_line_color: vec3<f32>,
    grid_overlay_enabled: f32,
    block_line_color: vec3<f32>,
    _pad: f32,
    voxel_line_half_width: f32,
    block_line_half_width: f32,
    voxel_line_alpha: f32,
    block_line_alpha: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: VoxelUniforms;

// M7: the material is a 6-layer array, one layer per cube face. Layer order
// matches the renderer's CubeFaceSlot: 0 +X(east), 1 -X(west), 2 +Y(up),
// 3 -Y(down), 4 +Z(south), 5 -Z(north). A uniform material has the same image
// on all six layers, so the per-voxel slice + grid overlay below are unchanged.
@group(1) @binding(0)
var material_texture: texture_2d_array<f32>;
@group(1) @binding(1)
var material_sampler: sampler;

// Pick the texture-array layer for a cube face from its outward normal.
fn face_layer(face_normal: vec3<f32>) -> i32 {
    let axis_magnitude = abs(face_normal);
    if (axis_magnitude.x > 0.5) {
        // +X → east (0), -X → west (1).
        return select(1, 0, face_normal.x > 0.0);
    } else if (axis_magnitude.y > 0.5) {
        // +Y → up (2), -Y → down (3).
        return select(3, 2, face_normal.y > 0.0);
    } else {
        // +Z → south (4), -Z → north (5).
        return select(5, 4, face_normal.z > 0.0);
    }
}

struct VertexInput {
    @location(0) vertex_position: vec3<f32>,
    @location(1) face_normal: vec3<f32>,
    @location(2) face_uv: vec2<f32>,
};

struct InstanceInput {
    @location(3) world_position: vec3<f32>,
    @location(4) block_local_coord: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) texture_coord: vec2<f32>,
    // Absolute voxel position: world position of this fragment shifted so that
    // voxel boundaries fall on integers (BUG 2 fix input).
    @location(2) voxel_absolute_position: vec3<f32>,
    // M7: the texture-array layer for this face (flat — constant per face).
    @location(3) @interpolate(flat) face_texture_layer: i32,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Unit cube centred on the voxel centre (half-extent 0.5 in each axis).
    let world_point = instance.world_position + vertex.vertex_position * 0.5;

    // --- BUG 1 fix: per-voxel texture slice within the block ---
    // Pick the two in-plane axes from the face normal, offset the base face UV by
    // the voxel's block-local coordinate, then divide by the density so one
    // block's texture spans all its voxels (each voxel = a 1/density slice).
    let axis_magnitude = abs(vertex.face_normal);
    var voxel_offset: vec2<f32>;
    if (axis_magnitude.x > 0.5) {
        voxel_offset = vec2<f32>(instance.block_local_coord.z, instance.block_local_coord.y);
    } else if (axis_magnitude.y > 0.5) {
        voxel_offset = vec2<f32>(instance.block_local_coord.x, instance.block_local_coord.z);
    } else {
        voxel_offset = vec2<f32>(instance.block_local_coord.x, instance.block_local_coord.y);
    }

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(world_point, 1.0);
    output.world_normal = vertex.face_normal;
    output.texture_coord = (vertex.face_uv + voxel_offset) / uniforms.voxels_per_block;
    output.voxel_absolute_position = world_point + uniforms.grid_half_extent;
    output.face_texture_layer = face_layer(vertex.face_normal);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sampled material colour for this face's layer (sRGB texture → linear via
    // the format). The per-voxel slice in `texture_coord` is unchanged: each face
    // samples its own layer, sliced by block_local_coord exactly as before.
    let sampled = textureSample(
        material_texture,
        material_sampler,
        input.texture_coord,
        input.face_texture_layer,
    ).rgb;

    // Directional + ambient lighting (kept from M2), applied to the texture.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.45;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    var color = sampled * lighting;

    // --- BUG 2 fix: grid overlay from absolute voxel position ---
    if (uniforms.grid_overlay_enabled > 0.5) {
        let absolute = input.voxel_absolute_position;
        // 1 on the two in-plane axes, 0 on the face-normal axis.
        let in_plane = step(abs(input.world_normal), vec3<f32>(0.5));
        // Distance to the nearest voxel boundary (per axis).
        let voxel_distance = abs(absolute - floor(absolute + 0.5));
        // Distance to the nearest block boundary (per axis).
        let density = uniforms.voxels_per_block;
        let block_distance =
            abs(absolute / density - floor(absolute / density + 0.5)) * density;

        let antialias = 0.012;
        let voxel_half_width = uniforms.voxel_line_half_width;
        let block_half_width = uniforms.block_line_half_width;
        let voxel_line = (vec3<f32>(1.0)
            - smoothstep(vec3<f32>(voxel_half_width), vec3<f32>(voxel_half_width + antialias), voxel_distance))
            * in_plane;
        let block_line = (vec3<f32>(1.0)
            - smoothstep(vec3<f32>(block_half_width), vec3<f32>(block_half_width + antialias), block_distance))
            * in_plane;
        let voxel_strength = max(max(voxel_line.x, voxel_line.y), voxel_line.z);
        let block_strength = max(max(block_line.x, block_line.y), block_line.z);

        // The block line is bolder/darker and wins where it is present.
        var blend = voxel_strength * uniforms.voxel_line_alpha;
        var line_color = uniforms.voxel_line_color;
        let block_blend = block_strength * uniforms.block_line_alpha;
        if (block_blend > blend) {
            blend = block_blend;
            line_color = uniforms.block_line_color;
        }
        color = mix(color, line_color, blend);
    }

    return vec4<f32>(color, 1.0);
}
