// View-cube shader (Milestone 5; restyled to the "Signal" language, #86).
//
// Draws the small orientation cube in the top-right corner viewport. Each face is a
// FLAT, FULLY OPAQUE near-black fill (issue #91 item 6) sampled from its own label
// texture (a 6-layer 2D array, layer = materialIndex order +X,-X,+Y,-Y,+Z,-Z) — no
// lighting, per the Signal "flat fills" rule. On hover, an element's across-the-fold
// facets are tinted with the onion-haze accent, decided GEOMETRICALLY from the
// fragment's cube-space position and the hovered element's per-axis sign selector
// (so an edge lights a thin strip cell on each of its two faces, a corner a corner
// cell on each of three).

struct CubeUniforms {
    view_projection: mat4x4<f32>,
    // Signal hover: `highlight = [sel_x, sel_y, sel_z, active]`. Each `sel` ∈ {-1,0,+1}
    // is the hovered element's face-normal sign along that axis; `active` (0/1) gates
    // the highlight. Reuses the Rust `LineUniforms.depth_bias` vec4 slot (offset 64).
    highlight: vec4<f32>,
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
    // Cube-space surface position (in [-HALF, HALF]³) for the geometric hover test.
    @location(3) local_pos: vec3<f32>,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * vec4<f32>(vertex.position, 1.0);
    output.normal = vertex.normal;
    output.uv = vertex.uv;
    output.layer = vertex.layer;
    output.local_pos = vertex.position;
    return output;
}

// The half-width of the 68 %-centre patch in cube units: 0.68 · HALF (HALF = 0.7).
// MUST track `raycast::VIEW_CUBE_ZONE_THRESHOLD` so the highlight lands on the drawn
// slice lines and the pick zones.
const CENTRE_HALF: f32 = 0.476;

// The 68 %-centre patch fraction (issue #91 item 3): the 3×3 partition boundaries sit
// at these face-UV coordinates. MUST track `raycast::VIEW_CUBE_CENTRE_PATCH_FRACTION`.
const PATCH_FRACTION: f32 = 0.68;

// sRGB (0..1) → linear, for baking the hairline slice-line colour in shader.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(low, high, c > cutoff);
}

// Issue #91 (item 3): the 3×3 slice lines rendered as a screen-space SDF in face-UV
// space (NOT baked into the face texture), so `fwidth` keeps them a CONSTANT ~1.4 px
// wide with 1 px anti-aliased edges at ANY orbit angle — no perspective-minification
// thinning at glancing angles. Returns line coverage in [0,1].
fn slice_coverage(uv: vec2<f32>) -> f32 {
    let low = (1.0 - PATCH_FRACTION) * 0.5; // 0.16
    let high = 1.0 - low;                   // 0.84
    // Distance (in pixels) to the nearest of the two boundaries on each axis.
    let du = min(abs(uv.x - low), abs(uv.x - high)) / max(fwidth(uv.x), 1e-6);
    let dv = min(abs(uv.y - low), abs(uv.y - high)) / max(fwidth(uv.y), 1e-6);
    let d = min(du, dv);
    let half_px = 0.7; // → a 1.4 px line
    return 1.0 - smoothstep(half_px - 0.5, half_px + 0.5, d);
}

// Is coordinate `p` on the selector's side of the centre patch on one axis?
//   sel > 0  → high strip  (p ≥ +CENTRE_HALF)
//   sel < 0  → low strip   (p ≤ -CENTRE_HALF)
//   sel = 0  → centre band (|p| ≤ CENTRE_HALF)
fn axis_ok(sel: f32, p: f32) -> bool {
    if (sel > 0.5) { return p >= CENTRE_HALF; }
    if (sel < -0.5) { return p <= -CENTRE_HALF; }
    return abs(p) <= CENTRE_HALF;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Flat fill + projected FRONT/TOP/RIGHT label from the face texture. No lighting.
    var color = textureSample(label_textures, label_sampler, input.uv, input.layer).rgb;

    // Issue #91 (item 3): composite the hairline `#2b3238` 3×3 slice lines over the fill
    // as a constant-width anti-aliased SDF (screen-space, so glancing angles don't thin
    // them). The drawn partition still coincides with the pick partition (PATCH_FRACTION).
    let slice = srgb_to_linear(vec3<f32>(f32(0x2b), f32(0x32), f32(0x38)) / 255.0);
    color = mix(color, slice, slice_coverage(input.uv));

    // Geometric hover highlight: tint the fragment toward the accent iff it lies in the
    // hovered element's cell on every axis.
    if (uniforms.highlight.w > 0.5) {
        let sel = uniforms.highlight.xyz;
        let p = input.local_pos;
        if (axis_ok(sel.x, p.x) && axis_ok(sel.y, p.y) && axis_ok(sel.z, p.z)) {
            // #9cb4d8 in linear space; ~45 % accent fill (Signal's sole accent).
            let accent = vec3<f32>(0.333, 0.457, 0.687);
            color = mix(color, accent, 0.45);
        }
    }

    // Issue #91 (item 6): the instrument-panel faces are FULLY OPAQUE (they must read
    // solid over a textured voxel scene, matching the approved screenshots), so no scene
    // bleeds through. The face pipeline still alpha-blends for the AA-line feathering.
    return vec4<f32>(color, 1.0);
}
