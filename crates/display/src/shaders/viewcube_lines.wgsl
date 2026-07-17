// View-cube wireframe lines — constant screen-space width, anti-aliased (issue #91 item 3).
//
// Replaces the hardware `LineList` pipeline (1 px, aliased) for the cube's silhouette,
// the three axis-coloured edges, and the projected X/Y/Z letter glyphs. Each source
// segment is expanded on the GPU into a screen-space quad of CONSTANT pixel width with a
// 1 px feathered alpha edge, so the linework stays ~1.4 px wide and crisp at ANY orbit
// angle (no perspective-minification thinning at glancing angles). The face fills stay
// flat/textured; only the linework moved here.

struct LineUniforms {
    view_projection: mat4x4<f32>,
    // The cube's on-screen viewport size in pixels (square), used to convert the
    // pixel-space width into an NDC offset.
    viewport_px: vec2<f32>,
    // Core half-width (px) and the additional feather (px) at each edge.
    half_width_px: f32,
    feather_px: f32,
};

@group(0) @binding(0) var<uniform> u: LineUniforms;

struct VsIn {
    @location(0) position_a: vec3<f32>,
    @location(1) position_b: vec3<f32>,
    @location(2) color: vec4<f32>,
    // .x = side (-1/+1 across the width), .y = end (0 → a, 1 → b).
    @location(3) side_end: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    // Signed distance (px) from the segment centre-line, for the fragment AA.
    @location(1) v_offset: f32,
};

@vertex
fn vertex_main(in: VsIn) -> VsOut {
    let clip_a = u.view_projection * vec4<f32>(in.position_a, 1.0);
    let clip_b = u.view_projection * vec4<f32>(in.position_b, 1.0);
    let ndc_a = clip_a.xy / clip_a.w;
    let ndc_b = clip_b.xy / clip_b.w;

    let half_vp = u.viewport_px * 0.5;
    let px_a = ndc_a * half_vp;
    let px_b = ndc_b * half_vp;
    var dir = px_b - px_a;
    let len = max(length(dir), 1e-6);
    dir = dir / len;
    let normal = vec2<f32>(-dir.y, dir.x);

    let is_b = in.side_end.y > 0.5;
    let this_clip = select(clip_a, clip_b, is_b);
    let this_ndc = select(ndc_a, ndc_b, is_b);

    let total_half = u.half_width_px + u.feather_px;
    let offset_px = normal * in.side_end.x * total_half;
    let offset_ndc = offset_px / half_vp;

    var out: VsOut;
    // Re-apply the perspective w so the offset is a true screen-space displacement.
    out.clip = vec4<f32>((this_ndc + offset_ndc) * this_clip.w, this_clip.z, this_clip.w);
    out.color = in.color;
    out.v_offset = in.side_end.x * total_half;
    return out;
}

@fragment
fn fragment_main(in: VsOut) -> @location(0) vec4<f32> {
    let dist = abs(in.v_offset);
    // Solid within the core half-width, feathering to 0 over `feather_px` to the edge.
    let alpha = 1.0 - smoothstep(u.half_width_px, u.half_width_px + u.feather_px, dist);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
