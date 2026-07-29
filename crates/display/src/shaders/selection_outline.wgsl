// Selection outline + wash composite (ADR 0032, reworked 2026-07-29).
//
// Full-screen pass over the RESOLVED colour target. Inputs: the selection's FRONT
// and BACK hull depth maps (the selected bodies rasterised depth-only under the
// SAME view_projection as the scene pass — nearest front face / farthest back
// face) and the scene's MSAA depth. Per pixel:
//
//   * WASH where ANY scene sample lies inside `[front − ε, back + ε]` — the
//     visible surface is ON or IN the selected body. The interval (not front-face
//     equality) is what lets a carved concavity wash: its visible wall is the
//     cutter's INSIDE — back faces a front hull alone can never see — and it is
//     interior volume of the host that owns it. A body buried behind other
//     geometry washes nothing (scene sample nearer than its front hull — owner
//     law: selection feedback never x-rays). All four samples are tested because
//     sample positions are intra-pixel offsets: on a steep face the sample-0 depth
//     alone drifts past ε and the wash speckles. ε is a half-voxel converted to
//     NDC at the sampled depth via the projection z-row (`ndc_epsilon`), never by
//     linearising the hardware z: the inverse map amplifies depth-buffer
//     quantisation by view_depth²/near, while the forward slope degrades
//     gracefully at any baseline.
//   * OUTLINE (1px, near-opaque) just OUTSIDE the washed region's boundary — one
//     rule that traces both the body's screen silhouette and the edge where it
//     slips behind other geometry.
//
// Uniform control flow throughout (no textureSample), so the neighbour taps are
// plain loads; out-of-viewport texels were cleared to "no body" by the hull
// passes' full-attachment clears.

struct SelectionOutlineUniforms {
    tint: vec4<f32>,
    outline_alpha: f32,
    half_voxel: f32,
    // NdcDepthMapping (camera crate): the projection z-row. With d = view depth,
    // perspective ndc(d) = depth_offset/d − depth_scale; ortho ndc(d) =
    // −depth_scale·d + depth_offset.
    depth_scale: f32,
    depth_offset: f32,
    orthographic: f32,
    epsilon_floor: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: SelectionOutlineUniforms;
@group(0) @binding(1)
var front_hull_depth: texture_depth_2d;
@group(0) @binding(2)
var back_hull_depth: texture_depth_2d;
@group(0) @binding(3)
var scene_depth: texture_depth_multisampled_2d;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// One over-sized triangle covering the viewport (the scissor clips the overhang).
@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    var output: VertexOutput;
    let x = select(-1.0, 3.0, index == 1u);
    let y = select(-1.0, 3.0, index == 2u);
    output.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return output;
}

// The half-voxel coincidence tolerance IN NDC at a sampled selection depth:
// half_voxel · |∂ndc/∂d|, floored by a few depth-buffer ULPs so two triangulations
// of the same plane still read coincident where the slope collapses.
fn ndc_epsilon(selection_ndc: f32) -> f32 {
    var epsilon: f32;
    if (uniforms.orthographic > 0.5) {
        epsilon = uniforms.half_voxel * abs(uniforms.depth_scale);
    } else {
        // Recover the view depth from the sample itself; depth_scale < −1 and
        // ndc ∈ [0, 1), so the denominator is strictly negative — never zero.
        let view_depth = uniforms.depth_offset / (selection_ndc + uniforms.depth_scale);
        epsilon = uniforms.half_voxel * abs(uniforms.depth_offset) / (view_depth * view_depth);
    }
    return max(epsilon, uniforms.epsilon_floor);
}

// Whether the selected body is the visible surface at a pixel: covered by the
// front hull (< the 1.0 clear) AND some scene sample inside the padded interval.
fn selection_visible_at(pixel: vec2<i32>) -> bool {
    let front = textureLoad(front_hull_depth, pixel, 0);
    if (front >= 1.0) {
        return false;
    }
    let back = textureLoad(back_hull_depth, pixel, 0);
    let near_bound = front - ndc_epsilon(front);
    let far_bound = back + ndc_epsilon(back);
    for (var sample_index = 0; sample_index < 4; sample_index = sample_index + 1) {
        let scene = textureLoad(scene_depth, pixel, sample_index);
        if (scene >= near_bound && scene <= far_bound) {
            return true;
        }
    }
    return false;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.clip_position.xy);
    if (selection_visible_at(pixel)) {
        return uniforms.tint;
    }
    // 1px outer ring: a non-washed pixel with a washed 4-neighbour. Clamped taps:
    // at the map's edge the clamped neighbour is the pixel itself (not visible
    // here), so the border can never phantom-outline.
    let edge = vec2<i32>(textureDimensions(front_hull_depth)) - vec2<i32>(1, 1);
    let zero = vec2<i32>(0, 0);
    let outline = selection_visible_at(clamp(pixel + vec2<i32>(1, 0), zero, edge))
        || selection_visible_at(clamp(pixel + vec2<i32>(-1, 0), zero, edge))
        || selection_visible_at(clamp(pixel + vec2<i32>(0, 1), zero, edge))
        || selection_visible_at(clamp(pixel + vec2<i32>(0, -1), zero, edge));
    if (outline) {
        return vec4<f32>(uniforms.tint.rgb, uniforms.outline_alpha);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
