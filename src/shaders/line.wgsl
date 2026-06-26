// Flat coloured-line shader (Milestone 5).
//
// Shared by the origin gizmo and the view-cube edge wireframe. Each vertex
// carries a world-space position and a linear RGB colour; the only uniform is
// the view-projection matrix.

struct LineUniforms {
    view_projection: mat4x4<f32>,
    // A small NDC depth offset applied to every vertex (issue #29 floor fix).
    // wgpu forbids a hardware DepthBiasState on LineList topology, so the floor
    // grid biases its depth here instead: a NEGATIVE value pulls the line a hair
    // toward the camera (smaller NDC z) so it wins the `Less` depth test against
    // the model's coincident bottom face — letting the floor draw at the EXACT
    // base plane with no z-fight and no geometric vertical drop. Zero for every
    // other line pass (gizmo, lattice, view-cube edges, Points). `.yzw` pad to
    // keep the 16-byte std140 alignment after the mat4.
    depth_bias: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: LineUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    // Linear RGBA. Alpha is 1.0 for the gizmo / view-cube edges and < 1.0 for the
    // M8 block lattice / fine floor grid (alpha-blended at low opacity).
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    var clip = uniforms.view_projection * vec4<f32>(vertex.position, 1.0);
    // Bias depth in NDC (post-perspective): scale by w so the offset is applied
    // after the perspective divide. Negative `depth_bias` ⇒ closer to the camera.
    clip.z = clip.z + uniforms.depth_bias.x * clip.w;
    output.clip_position = clip;
    output.color = vertex.color;
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.color;
}
