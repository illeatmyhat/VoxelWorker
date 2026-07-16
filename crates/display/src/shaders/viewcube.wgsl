// View-cube shader (Milestone 5; restyled to the "Signal" language, #86).
//
// Draws the small orientation cube in the top-right corner viewport. Each face is a
// FLAT translucent near-black fill sampled from its own label texture (a 6-layer 2D
// array, layer = materialIndex order +X,-X,+Y,-Y,+Z,-Z) — no lighting, per the Signal
// "flat fills" rule. On hover, an element's across-the-fold facets are tinted with the
// onion-haze accent, decided GEOMETRICALLY from the fragment's cube-space position and
// the hovered element's per-axis sign selector (so an edge lights a thin strip cell on
// each of its two faces, a corner a corner cell on each of three).

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
    // Flat fill (translucent near-black + baked slice lines + label). No lighting.
    var color = textureSample(label_textures, label_sampler, input.uv, input.layer).rgb;

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

    // Translucent instrument-panel face (~80 % alpha over the resolved scene).
    return vec4<f32>(color, 0.82);
}
