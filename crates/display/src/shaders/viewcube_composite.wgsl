// View-cube composite (issue #91 item 3) — blend the resolved MSAA cube onto the scene.
//
// The view cube renders into its own small 4× MSAA target (cleared transparent), which is
// resolved to a single-sample texture with coverage-anti-aliased silhouette edges. This
// pass samples that resolved texture and composites it over the already-drawn scene in the
// cube's corner, using PREMULTIPLIED-alpha blending (the offscreen pass accumulated
// premultiplied values over a zeroed clear), so the cube's edges anti-alias cleanly
// against the viewport background.

struct VsOut {
    @builtin(position) clip: vec4<f32>,
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
    out.clip = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@group(0) @binding(0) var cube_tex: texture_2d<f32>;
@group(0) @binding(1) var cube_sampler: sampler;

@fragment
fn fragment_main(input: VsOut) -> @location(0) vec4<f32> {
    // Premultiplied RGBA from the resolved cube; the pipeline blends (One, 1-SrcAlpha).
    return textureSample(cube_tex, cube_sampler, input.uv);
}
