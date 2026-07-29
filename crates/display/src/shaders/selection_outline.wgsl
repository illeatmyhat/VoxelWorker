// Selection outline + wash composite (ADR 0032, reworked 2026-07-29; crease slice
// 2026-07-30).
//
// Full-screen pass over the RESOLVED colour target. Inputs: the selection's FRONT
// and BACK hull depth maps (the selected bodies rasterised depth-only under the
// SAME view_projection as the scene pass — nearest front face / farthest back
// face), the matching hull NORMAL maps (outward face normal, coverage in alpha,
// mip chain of coverage-weighted means), and the scene's MSAA depth. Per pixel:
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
//   * CREASE lines inside the wash where the surface bends sharply RELATIVE TO A
//     BLOCK (`crease_strength`) — the wash alpha lifts toward the outline alpha,
//     so the line speaks the same single accent. Two scales multiply:
//       - FINE: the angle between the normalised mean normals a couple of pixels
//         either side of the pixel — a thin, precise line that fires at every
//         edge, including per-voxel steps.
//       - GATE: the same angle test at a block-projected footprint (taps half a
//         block apart, mean normals from the matching mip). A voxel staircase is
//         a CONSTANT diagonal at this scale (no response — that is what makes
//         "voxelised rounding stays blunt" true), a real block edge flips the
//         field across the line, and a block-radius fillet turns it slowly enough
//         to stay under threshold. Toksvig |mean| alone cannot make this cut: the
//         staircase and the true edge both average to |mean| ≈ 0.7.
//     The map (front vs back) is chosen per pixel by which hull the visible scene
//     sample sits on — a selected cutter's visible bowl is its BACK faces.
//
// Uniform control flow throughout (no derivative-driven sampling: explicit-LOD
// `textureSampleLevel` only), so all taps are legal anywhere; out-of-viewport
// texels were cleared to "no body" by the hull passes' full-attachment clears.

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
    // NdcDepthMapping's lateral companion (projection y_axis.y): pixels per world
    // unit at view depth d = vertical_scale · viewport_height_px / (2·d), d = 1
    // under ortho. Sizes the crease pass's block-relative footprint.
    vertical_scale: f32,
    viewport_height_px: f32,
    voxels_per_block: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: SelectionOutlineUniforms;
@group(0) @binding(1)
var front_hull_depth: texture_depth_2d;
@group(0) @binding(2)
var back_hull_depth: texture_depth_2d;
@group(0) @binding(3)
var scene_depth: texture_depth_multisampled_2d;
@group(0) @binding(4)
var front_normal_map: texture_2d<f32>;
@group(0) @binding(5)
var back_normal_map: texture_2d<f32>;
@group(0) @binding(6)
var normal_sampler: sampler;

// Crease tuning. Footprint = this fraction of a projected block; the gate's taps
// sit a half-footprint out, its mip matches the footprint. Angle thresholds are
// on the dot of the normalised mean normals: full line at ≤ 60° (dot 0.5) — a
// true 90° edge reads ~0 — fading out by ~32° (dot 0.85), just above what a
// block-radius fillet turns across the footprint (~28°).
const CREASE_FOOTPRINT_BLOCK_FRACTION: f32 = 0.5;
const CREASE_FOOTPRINT_MIN_PX: f32 = 4.0;
const CREASE_FOOTPRINT_MAX_PX: f32 = 128.0;
const CREASE_FINE_TAP_PX: f32 = 1.5;
const CREASE_DOT_SHARP: f32 = 0.5;
const CREASE_DOT_BLUNT: f32 = 0.85;
// Below this coverage a tap is silhouette spill, not surface — no crease claim.
const CREASE_MIN_COVERAGE: f32 = 0.25;
// |mean| confidence ramp: below LOW the direction estimate is noise (opposed
// faces cancelled), above HIGH it is trustworthy.
const CREASE_CONFIDENCE_MEAN_LOW: f32 = 0.3;
const CREASE_CONFIDENCE_MEAN_HIGH: f32 = 0.6;

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

// The visible scene depth ON the selected body at a pixel, or −1.0: covered by
// the front hull (< the 1.0 clear) AND some scene sample inside the padded
// interval — that sample's depth is returned (any in-interval sample works; the
// crease pass only asks which HULL it is nearer).
fn selection_scene_depth(pixel: vec2<i32>) -> f32 {
    let front = textureLoad(front_hull_depth, pixel, 0);
    if (front >= 1.0) {
        return -1.0;
    }
    let back = textureLoad(back_hull_depth, pixel, 0);
    let near_bound = front - ndc_epsilon(front);
    let far_bound = back + ndc_epsilon(back);
    for (var sample_index = 0; sample_index < 4; sample_index = sample_index + 1) {
        let scene = textureLoad(scene_depth, pixel, sample_index);
        if (scene >= near_bound && scene <= far_bound) {
            return scene;
        }
    }
    return -1.0;
}

fn mean_normal_sample(use_back: bool, uv: vec2<f32>, mip: f32) -> vec4<f32> {
    if (use_back) {
        return textureSampleLevel(back_normal_map, normal_sampler, uv, mip);
    }
    return textureSampleLevel(front_normal_map, normal_sampler, uv, mip);
}

// The dot of the normalised mean normals at two taps — 1.0 (no crease evidence)
// where either tap lacks coverage. A SHORT mean (|mean| = |rgb|/alpha toward 0 —
// opposed faces mixed inside one footprint, e.g. both bowl walls) is an
// UNRELIABLE direction, not crease evidence: normalising it amplifies filter
// noise into direction flips that strobe under subpixel motion, so confidence
// fades the claim out instead (Toksvig |mean|'s surviving job — scoring the
// direction estimate, never classifying the crease).
fn crease_pair_dot(use_back: bool, uv_a: vec2<f32>, uv_b: vec2<f32>, mip: f32) -> f32 {
    let a = mean_normal_sample(use_back, uv_a, mip);
    let b = mean_normal_sample(use_back, uv_b, mip);
    if (min(a.a, b.a) < CREASE_MIN_COVERAGE) {
        return 1.0;
    }
    let length_a = length(a.rgb);
    let length_b = length(b.rgb);
    if (min(length_a, length_b) < 1e-4) {
        return 1.0;
    }
    let unit_dot = dot(a.rgb / length_a, b.rgb / length_b);
    let confidence = smoothstep(
        CREASE_CONFIDENCE_MEAN_LOW,
        CREASE_CONFIDENCE_MEAN_HIGH,
        min(length_a / a.a, length_b / b.a),
    );
    return mix(1.0, unit_dot, confidence);
}

// Min-over-axes pair dot at one scale: taps ±delta_px along each screen axis.
fn crease_scale_dot(use_back: bool, pixel: vec2<i32>, delta_px: f32, mip: f32) -> f32 {
    let full_size = vec2<f32>(textureDimensions(front_normal_map));
    let centre = vec2<f32>(pixel) + vec2<f32>(0.5, 0.5);
    let dx = vec2<f32>(delta_px, 0.0);
    let dy = vec2<f32>(0.0, delta_px);
    let dot_x = crease_pair_dot(use_back, (centre + dx) / full_size, (centre - dx) / full_size, mip);
    let dot_y = crease_pair_dot(use_back, (centre + dy) / full_size, (centre - dy) / full_size, mip);
    return min(dot_x, dot_y);
}

// Block-relative crease strength at a washed pixel (see the header): fine line
// response × block-footprint gate, each smoothstepped on the mean-normal angle.
fn crease_strength(pixel: vec2<i32>, scene_ndc: f32) -> f32 {
    let front = textureLoad(front_hull_depth, pixel, 0);
    let back = textureLoad(back_hull_depth, pixel, 0);
    let use_back = abs(scene_ndc - back) < abs(scene_ndc - front);
    let hull_ndc = select(front, back, use_back);
    var view_depth = 1.0;
    if (uniforms.orthographic < 0.5) {
        view_depth = uniforms.depth_offset / (hull_ndc + uniforms.depth_scale);
    }
    let pixels_per_voxel =
        uniforms.vertical_scale * uniforms.viewport_height_px * 0.5 / max(view_depth, 1e-3);
    let block_px = pixels_per_voxel * uniforms.voxels_per_block;
    let footprint_px = clamp(
        block_px * CREASE_FOOTPRINT_BLOCK_FRACTION,
        CREASE_FOOTPRINT_MIN_PX,
        CREASE_FOOTPRINT_MAX_PX,
    );
    let max_mip = f32(textureNumLevels(front_normal_map) - 1u);
    let gate_delta = footprint_px * 0.5;
    let gate_mip = clamp(log2(gate_delta), 0.0, max_mip);
    // Fine tap grows with the projected voxel (capped by the gate's own radius):
    // zoomed in, a per-voxel step would otherwise read as a full-90° fine response
    // and dash the whole gate band; at ~1.5 voxels the steps average out of the
    // fine field while a block-scale crease still flips it.
    let fine_delta = clamp(1.5 * pixels_per_voxel, CREASE_FINE_TAP_PX, gate_delta);
    let fine_mip = clamp(log2(fine_delta), 0.0, max_mip);
    let gate_dot = crease_scale_dot(use_back, pixel, gate_delta, gate_mip);
    let fine_dot = crease_scale_dot(use_back, pixel, fine_delta, fine_mip);
    let gate = 1.0 - smoothstep(CREASE_DOT_SHARP, CREASE_DOT_BLUNT, gate_dot);
    let fine = 1.0 - smoothstep(CREASE_DOT_SHARP, CREASE_DOT_BLUNT, fine_dot);
    return fine * gate;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.clip_position.xy);
    let scene_ndc = selection_scene_depth(pixel);
    if (scene_ndc >= 0.0) {
        let crease = crease_strength(pixel, scene_ndc);
        return vec4<f32>(uniforms.tint.rgb, mix(uniforms.tint.a, uniforms.outline_alpha, crease));
    }
    // 1px outer ring: a non-washed pixel with a washed 4-neighbour. Clamped taps:
    // at the map's edge the clamped neighbour is the pixel itself (not visible
    // here), so the border can never phantom-outline.
    let edge = vec2<i32>(textureDimensions(front_hull_depth)) - vec2<i32>(1, 1);
    let zero = vec2<i32>(0, 0);
    let outline = selection_scene_depth(clamp(pixel + vec2<i32>(1, 0), zero, edge)) >= 0.0
        || selection_scene_depth(clamp(pixel + vec2<i32>(-1, 0), zero, edge)) >= 0.0
        || selection_scene_depth(clamp(pixel + vec2<i32>(0, 1), zero, edge)) >= 0.0
        || selection_scene_depth(clamp(pixel + vec2<i32>(0, -1), zero, edge)) >= 0.0;
    if (outline) {
        return vec4<f32>(uniforms.tint.rgb, uniforms.outline_alpha);
    }
    return vec4<f32>(0.0, 0.0, 0.0, 0.0);
}
