// Block thumbnail shader (Milestone 6).
//
// Draws a textured unit cube at a fixed 45° orthographic view into a small
// offscreen texture, with simple hemisphere-ish lighting (ambient + one
// directional term). Unlike the main voxel shader there is NO per-voxel slice:
// the WHOLE block texture is shown on each face (this is just a material preview
// tile). The view-projection is supplied as a uniform so the CPU controls the
// azimuth/elevation (prototype `thumbCam`).

struct ThumbnailUniforms {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: ThumbnailUniforms;

@group(1) @binding(0)
var block_texture: texture_2d<f32>;
@group(1) @binding(1)
var block_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    // The cube vertices span [-1, 1]; scale to a unit cube (half-extent 0.5).
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.position * 0.5, 1.0);
    output.normal = vertex.normal;
    output.uv = vertex.uv;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(block_texture, block_sampler, input.uv);
    let light_direction = normalize(vec3<f32>(0.5, 0.85, 0.6));
    let normal = normalize(input.normal);
    let diffuse = max(dot(normal, light_direction), 0.0);
    let ambient = 0.55;
    let lighting = ambient + (1.0 - ambient) * diffuse;
    return vec4<f32>(sampled.rgb * lighting, sampled.a);
}
