// Instanced unit-cube voxel shader (Milestone 2).
//
// Each instance is one occupied voxel. The unit cube has per-face normals (24
// vertices / 36 indices) so faces are flat-shaded independently, making the
// stair-stepped quantization of the curved rim read clearly.
//
// M4 will grow this file: the vertex stage will pick a texture slice from
// `block_local_coord` and the fragment stage will composite the per-block grid
// overlay. Those inputs are already plumbed (`block_local_coord` is passed per
// instance) so M4 only adds logic, not wiring.

struct CameraUniform {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) vertex_position: vec3<f32>,
    @location(1) face_normal: vec3<f32>,
};

struct InstanceInput {
    @location(2) world_position: vec3<f32>,
    // Unused by the M2 fragment shader; populated now for M4.
    @location(3) block_local_coord: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput, instance: InstanceInput) -> VertexOutput {
    // Unit cube centred on the voxel centre (half-extent 0.5 in each axis).
    let world_point = instance.world_position + vertex.vertex_position * 0.5;

    var output: VertexOutput;
    output.clip_position = camera.view_projection * vec4<f32>(world_point, 1.0);
    output.world_normal = vertex.face_normal;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Warm stone-ish base colour (ARCHITECTURE.md §8 plain warm grey).
    let base_color = vec3<f32>(0.71, 0.63, 0.47);

    // Simple directional + ambient flat lighting from the face normal.
    let light_direction = normalize(vec3<f32>(0.4, 0.9, 0.5));
    let normal = normalize(input.world_normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.35;
    let lighting = ambient + (1.0 - ambient) * diffuse;

    return vec4<f32>(base_color * lighting, 1.0);
}
