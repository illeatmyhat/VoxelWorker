// View-cube CHROME overlay shader (#13 Step 2).
//
// Draws screen-space textured glyph quads (Home/Fit button badges and the hover
// rotate/roll arrows) that are FIXED to the cube rect — they do NOT rotate with
// the cube. Positions arrive already in NDC
// (computed on the CPU from the Step-1 layout fractions, within the scissored
// cube viewport), so the vertex stage is a pass-through. Each glyph samples its
// own layer from a 2D texture array and is tinted by a per-vertex colour (used
// to highlight a hovered arrow). Alpha-blended over the already-drawn scene/cube.

struct ChromeVertexInput {
    @location(0) position: vec2<f32>,   // NDC x,y
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,      // tint (rgb) * alpha
    @location(3) layer: u32,
};

struct ChromeVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) layer: u32,
};

@group(0) @binding(0)
var chrome_textures: texture_2d_array<f32>;
@group(0) @binding(1)
var chrome_sampler: sampler;

@vertex
fn vertex_main(vertex: ChromeVertexInput) -> ChromeVertexOutput {
    var output: ChromeVertexOutput;
    output.clip_position = vec4<f32>(vertex.position, 0.0, 1.0);
    output.uv = vertex.uv;
    output.color = vertex.color;
    output.layer = vertex.layer;
    return output;
}

@fragment
fn fragment_main(input: ChromeVertexOutput) -> @location(0) vec4<f32> {
    let texel = textureSample(chrome_textures, chrome_sampler, input.uv, input.layer);
    // The glyph textures carry their own alpha (transparent background, opaque
    // glyph). Multiply by the per-vertex tint so a hovered arrow can brighten.
    return vec4<f32>(texel.rgb * input.color.rgb, texel.a * input.color.a);
}
