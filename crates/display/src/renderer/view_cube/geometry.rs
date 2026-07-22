//! Pure-CPU view-cube mesh + wireframe generation (no wgpu): the labelled-cube
//! geometry, the Signal silhouette/axis-edge line list, and the thick-line quad
//! expansion the GPU renderer uploads. Returns plain `Vec`s so the geometry is
//! testable without a device.

use super::*;

/// Expand a `LineList` vertex stream (consecutive `a, b` pairs) into thick-line quad
/// vertices (6 per segment). Both endpoints of a cube edge share a colour, so the quad
/// takes the pair's first colour.
pub(super) fn expand_thick_lines(segments: &[LineVertex]) -> Vec<ThickLineVertex> {
    // (side, end) for the two triangles of the quad.
    const CORNERS: [(f32, f32); 6] = [
        (-1.0, 0.0),
        (-1.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (-1.0, 1.0),
        (1.0, 1.0),
    ];
    let mut out = Vec::with_capacity(segments.len() / 2 * 6);
    for pair in segments.chunks_exact(2) {
        let (a, b) = (pair[0], pair[1]);
        for (side, end) in CORNERS {
            out.push(ThickLineVertex {
                position_a: a.position,
                position_b: b.position,
                color: a.color,
                side_end: [side, end],
            });
        }
    }
    out
}

/// Build the labelled-cube geometry (side 1.4, centred on origin). Face order +X,
/// -X, +Y, -Y, +Z, -Z (matches `materialIndex` / `CubeFace`).
pub(crate) fn view_cube_geometry() -> (Vec<CubeLabelVertex>, Vec<u16>) {
    const HALF: f32 = 0.7; // side 1.4
    const UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [[HALF, -HALF, HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [HALF, HALF, HALF]]),
        ([-1.0, 0.0, 0.0], [[-HALF, -HALF, -HALF], [-HALF, -HALF, HALF], [-HALF, HALF, HALF], [-HALF, HALF, -HALF]]),
        ([0.0, 1.0, 0.0], [[-HALF, HALF, HALF], [HALF, HALF, HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF]]),
        ([0.0, -1.0, 0.0], [[-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, -HALF, HALF], [-HALF, -HALF, HALF]]),
        ([0.0, 0.0, 1.0], [[-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF]]),
        ([0.0, 0.0, -1.0], [[HALF, -HALF, -HALF], [-HALF, -HALF, -HALF], [-HALF, HALF, -HALF], [HALF, HALF, -HALF]]),
    ];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (layer, (normal, corners)) in faces.iter().enumerate() {
        let base = vertices.len() as u16;
        // Z-up: the BACK (+Y, layer 2) and BOTTOM (−Z, layer 5) faces wind such that
        // the shared UV table maps their label upside-down. Rotate just those two
        // faces' UVs 180° (corner_index + 2) so every label reads upright — the fix
        // lives in the unwrap, keeping the label textures themselves canonical.
        let uv_rotated = layer == 2 || layer == 5;
        for (corner_index, corner) in corners.iter().enumerate() {
            let uv_index = if uv_rotated { (corner_index + 2) % 4 } else { corner_index };
            vertices.push(CubeLabelVertex {
                position: *corner,
                normal: *normal,
                uv: UVS[uv_index],
                layer: layer as u32,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// The Signal cube wireframe: the 12 silhouette edges (`#59636d`), the three
/// axis-coloured edges emanating from the front-bottom-right corner
/// (`(+HALF, −HALF, −HALF)` = right/front/bottom), and small projected X/Y/Z letter
/// glyphs at the far ends of those edges. All drawn by the shared line pipeline
/// (per-vertex colour, cube VP transform), so the axis triad foreshortens WITH the
/// cube — never a screen-space approximation.
///
/// Z-up world mapping (front = −Y): from the shared corner, X (`#d9603f`) runs along
/// the bottom-front edge toward −X, Y (`#7dba6a`) up the receding right-bottom edge
/// toward +Y, and Z (`#9cb4d8`) up the front-right vertical toward +Z.
pub(super) fn view_cube_edges() -> Vec<LineVertex> {
    const HALF: f32 = 0.705; // a hair outside the faces so the edges read crisply
    let silhouette = with_alpha(srgb_hex_to_linear(SILHOUETTE_HEX), 1.0);
    let axis_x = with_alpha(srgb_hex_to_linear(AXIS_X_HEX), 1.0);
    let axis_y = with_alpha(srgb_hex_to_linear(AXIS_Y_HEX), 1.0);
    let axis_z = with_alpha(srgb_hex_to_linear(AXIS_Z_HEX), 1.0);

    // The three axis edges share the front-bottom-right corner `(+HALF, −HALF, −HALF)`;
    // these are their FAR endpoints (where the letter glyphs sit).
    let x_far = [-HALF, -HALF, -HALF]; // along the bottom-front edge (−X)
    let y_far = [HALF, HALF, -HALF]; //  up the receding right-bottom edge (+Y)
    let z_far = [HALF, -HALF, HALF]; //  up the front-right vertical (+Z)

    let corners = [
        [-HALF, -HALF, -HALF], [HALF, -HALF, -HALF], [HALF, HALF, -HALF], [-HALF, HALF, -HALF],
        [-HALF, -HALF, HALF], [HALF, -HALF, HALF], [HALF, HALF, HALF], [-HALF, HALF, HALF],
    ];
    // The 12 edges as index pairs; the three axis edges are tagged with their colour.
    let edges: [((usize, usize), [f32; 4]); 12] = [
        ((0, 1), axis_x),       // bottom-front (varies X): the X axis edge
        ((1, 2), axis_y),       // right-bottom (varies Y): the Y axis edge
        ((2, 3), silhouette),
        ((3, 0), silhouette),
        ((4, 5), silhouette),
        ((5, 6), silhouette),
        ((6, 7), silhouette),
        ((7, 4), silhouette),
        ((0, 4), silhouette),
        ((1, 5), axis_z),       // front-right vertical (varies Z): the Z axis edge
        ((2, 6), silhouette),
        ((3, 7), silhouette),
    ];
    let mut vertices = Vec::with_capacity(edges.len() * 2 + 24);
    for ((a, b), color) in edges {
        vertices.push(LineVertex { position: corners[a], color });
        vertices.push(LineVertex { position: corners[b], color });
    }

    // Axis letter glyphs at the FAR ends (offset a touch outward from the corner so
    // they read past the silhouette). Each glyph is drawn in a cube-space plane
    // (right/up unit vectors) and projected with the cube.
    const GLYPH: f32 = 0.20; // glyph box side, cube units
    const OUT: f32 = 0.10; // outward nudge from the endpoint
    // X: on the bottom-front edge; stand it in the XZ plane (right = +X, up = +Z),
    // nudged down/forward so it sits just outside the front-bottom edge.
    push_line_letter(
        &mut vertices,
        'X',
        [x_far[0] - OUT, x_far[1] - OUT, x_far[2] - OUT],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_x,
    );
    // Y: on the receding right-bottom edge; plane (right = +Y, up = +Z).
    push_line_letter(
        &mut vertices,
        'Y',
        [y_far[0] + OUT, y_far[1] + OUT, y_far[2] - OUT],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_y,
    );
    // Z: on the front-right vertical; plane (right = +X, up = +Z).
    push_line_letter(
        &mut vertices,
        'Z',
        [z_far[0] + OUT, z_far[1] - OUT, z_far[2] + OUT],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        GLYPH,
        axis_z,
    );
    vertices
}

/// One letter stroke in the unit `[-0.5, 0.5]²` glyph cell: a `(u, v)` start/end pair.
type LetterStroke = ((f32, f32), (f32, f32));

/// Append the line segments of a single axis letter (`X`/`Y`/`Z`) to `vertices`,
/// centred at `center` in the cube-space plane spanned by unit vectors `right` and
/// `up`, with box side `scale` and colour `color`. Strokes are defined in a unit
/// `[-0.5, 0.5]²` cell (u along `right`, v along `up`) and mapped into cube space, so
/// the glyph foreshortens with the cube under the shared VP.
fn push_line_letter(
    vertices: &mut Vec<LineVertex>,
    letter: char,
    center: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    scale: f32,
    color: [f32; 4],
) {
    // Unit-cell strokes (u, v) → (u, v) endpoint pairs per letter.
    let strokes: &[LetterStroke] = match letter {
        'X' => &[((-0.5, -0.5), (0.5, 0.5)), ((-0.5, 0.5), (0.5, -0.5))],
        'Y' => &[
            ((-0.5, 0.5), (0.0, 0.0)),
            ((0.5, 0.5), (0.0, 0.0)),
            ((0.0, 0.0), (0.0, -0.5)),
        ],
        'Z' => &[
            ((-0.5, 0.5), (0.5, 0.5)),
            ((0.5, 0.5), (-0.5, -0.5)),
            ((-0.5, -0.5), (0.5, -0.5)),
        ],
        _ => &[],
    };
    let map = |u: f32, v: f32| {
        [
            center[0] + (right[0] * u + up[0] * v) * scale,
            center[1] + (right[1] * u + up[1] * v) * scale,
            center[2] + (right[2] * u + up[2] * v) * scale,
        ]
    };
    for ((u0, v0), (u1, v1)) in strokes {
        vertices.push(LineVertex { position: map(*u0, *v0), color });
        vertices.push(LineVertex { position: map(*u1, *v1), color });
    }
}
