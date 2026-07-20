use super::*;

/// One mesh vertex of a cuboid face: world position, the face's outward normal, and the
/// box's `block_id` (the clean colour index, constant across the face).
///
/// ADR 0010 E3 / ADR 0003 §3c: the on-face-grid overlay flag is **no longer a vertex
/// attribute**. A chunk mesh is SPLIT into an overlay-off and an overlay-on index run over
/// this one shared vertex list (a box never spans both — the overlay bit is part of the
/// decomposition key), and the draw selects the per-draw overlay-active uniform per run. So
/// the render flag is entirely out of the per-vertex format while the per-object behaviour
/// (the `voxel_grid_flag_bit_is_per_object` invariant) is preserved by the split.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct CuboidVertex {
    pub(crate) position: [f32; 3],
    pub(crate) normal: [f32; 3],
    pub(crate) material_id: u32,
}

/// The six cube-face directions, each with its outward normal and the four
/// corner offsets (in voxel units, relative to the box's min corner, scaled by
/// the box's extent) wound COUNTER-CLOCKWISE when viewed from OUTSIDE — so
/// `front_face: Ccw` + `cull_mode: Back` keeps the outward faces (this matched the
/// winding convention of the instanced per-voxel-cube renderer, since removed with
/// the legacy mesher — #20).
///
/// Each corner is `[x, y, z]` in {0,1}: 0 = the box's min-corner plane on that
/// axis, 1 = its max-corner plane. The mesh builder maps 0→`min` and
/// 1→`max+1` (inclusive box → exclusive far plane) to get the world corner.
pub(crate) struct FaceTemplate {
    /// `+1`/`-1` direction along the axis this face faces; used both for the
    /// outward normal and to find the neighbour cell to test for exposure.
    pub(crate) neighbor_delta: [i32; 3],
    pub(crate) normal: [f32; 3],
    /// Four corners as {0,1} per axis, CCW from outside.
    pub(crate) corners: [[u32; 3]; 4],
}

pub(crate) const FACE_TEMPLATES: [FaceTemplate; 6] = [
    // +X
    FaceTemplate {
        neighbor_delta: [1, 0, 0],
        normal: [1.0, 0.0, 0.0],
        corners: [[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0]],
    },
    // -X
    FaceTemplate {
        neighbor_delta: [-1, 0, 0],
        normal: [-1.0, 0.0, 0.0],
        corners: [[0, 1, 1], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
    },
    // +Y
    FaceTemplate {
        neighbor_delta: [0, 1, 0],
        normal: [0.0, 1.0, 0.0],
        corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
    },
    // -Y
    FaceTemplate {
        neighbor_delta: [0, -1, 0],
        normal: [0.0, -1.0, 0.0],
        corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    },
    // +Z
    FaceTemplate {
        neighbor_delta: [0, 0, 1],
        normal: [0.0, 0.0, 1.0],
        corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    },
    // -Z
    FaceTemplate {
        neighbor_delta: [0, 0, -1],
        normal: [0.0, 0.0, -1.0],
        corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
    },
];
