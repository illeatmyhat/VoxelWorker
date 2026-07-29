// Selection normal-map mip downsample (selection_outline.rs, slice 2).
//
// One level per pass: the source binding is a view restricted to level N−1, the
// target is level N. A single bilinear tap at the destination texel's centre
// averages the 2×2 source block (exact for even extents; the odd-edge texel's
// clamp bias is irrelevant at the crease pass's tolerances). Normals ride
// premultiplied by coverage (alpha), so the average stays a coverage-weighted
// MEAN NORMAL — the field the crease pass reads at a block-relative level.

@group(0) @binding(0)
var source_level: texture_2d<f32>;
@group(0) @binding(1)
var source_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = select(-1.0, 3.0, index == 1u);
    let y = select(-1.0, 3.0, index == 2u);
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Address the SOURCE directly: target texel i averages source texels 2i and
    // 2i+1, whose shared corner sits at source coordinate 2i+1. Dividing by the
    // target size instead would stretch by source/(2·target) at every ODD level
    // (wgpu floors mip extents), and the compounding skew displaces the coarse
    // mean-normal field by whole blocks near the far screen edges.
    let source_size = vec2<f32>(textureDimensions(source_level));
    let corner = 2.0 * floor(input.clip_position.xy) + vec2<f32>(1.0, 1.0);
    return textureSampleLevel(source_level, source_sampler, corner / source_size, 0.0);
}
