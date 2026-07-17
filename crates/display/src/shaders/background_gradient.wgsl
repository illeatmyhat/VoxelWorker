// Viewport background gradient (issue #91, item 1) — the Signal field.
//
// A fullscreen radial gradient painted FIRST in the 3D MSAA pass (before the voxels,
// depth-test off / no write) so both display paths (cuboid mesh + brick raymarch) and
// the headless `shot` composite the scene over the SAME background. It reproduces the
// approved mock's viewport field:
//
//   radial-gradient(120% 90% at 45% 38%, #1d2023 0%, #141618 62%, #0e0f11 100%)
//
// a cool near-black radial biased above-left of centre. The gradient is evaluated in
// sRGB (gamma) space exactly like the CSS mock, then converted to LINEAR for the
// Rgba8UnormSrgb target (the GPU re-encodes to sRGB on write), so the rendered pixels
// match the mock's, not merely the hex stops.

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    // Normalised viewport coordinate in [0,1]² (0,0 = top-left of the 3D viewport).
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VsOut {
    // One oversized triangle covering the viewport; the pass scissor clips it to the
    // central 3D rect, and `uv` interpolates 0..1 across that rect (the set_viewport
    // transform maps NDC ±1 onto the rect, so uv is rect-relative regardless of size).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(3.0, 1.0),
    );
    let p = positions[vertex_index];
    var out: VsOut;
    out.clip_position = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

// sRGB (0..1) → linear (0..1), the standard IEC 61966-2-1 transfer.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(low, high, c > cutoff);
}

@fragment
fn fragment_main(input: VsOut) -> @location(0) vec4<f32> {
    // The three CSS stops in sRGB [0,1].
    let stop0 = vec3<f32>(0x1d, 0x20, 0x23) / 255.0; // #1d2023 @ 0%
    let stop1 = vec3<f32>(0x14, 0x16, 0x18) / 255.0; // #141618 @ 62%
    let stop2 = vec3<f32>(0x0e, 0x0f, 0x11) / 255.0; // #0e0f11 @ 100%

    // Ellipse: centre 45%/38%, radii 120%/90% of the viewport (CSS `at 45% 38%`,
    // `120% 90%`). The gradient parameter is the normalised elliptical distance.
    let centre = vec2<f32>(0.45, 0.38);
    let radii = vec2<f32>(1.20, 0.90);
    let d = (input.uv - centre) / radii;
    let t = clamp(length(d), 0.0, 1.0);

    // Piecewise-linear interpolation in sRGB space (as CSS does), then to linear.
    var srgb: vec3<f32>;
    if (t < 0.62) {
        srgb = mix(stop0, stop1, t / 0.62);
    } else {
        srgb = mix(stop1, stop2, (t - 0.62) / 0.38);
    }
    return vec4<f32>(srgb_to_linear(srgb), 1.0);
}
