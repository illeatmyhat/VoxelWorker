// View-cube shader (Milestone 5).
//
// Draws a small labelled cube in a corner viewport that mirrors the main
// camera's orientation. Each face samples its own label texture from a 6-layer
// 2D texture array (layer = materialIndex order +X,-X,+Y,-Y,+Z,-Z). A simple
// hemispheric light keeps the faces readable.

struct CubeUniforms {
    view_projection: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: CubeUniforms;

@group(1) @binding(0)
var label_textures: texture_2d_array<f32>;
@group(1) @binding(1)
var label_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) layer: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) layer: u32,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.position, 1.0);
    output.normal = vertex.normal;
    output.uv = vertex.uv;
    output.layer = vertex.layer;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(label_textures, label_sampler, input.uv, input.layer).rgb;
    // Soft hemispheric lighting so each face stays legible but shaded.
    let light_direction = normalize(vec3<f32>(0.4, 0.7, 0.6));
    let lit = 0.6 + 0.4 * max(dot(normalize(input.normal), light_direction), 0.0);
    return vec4<f32>(texel * lit, 1.0);
}
