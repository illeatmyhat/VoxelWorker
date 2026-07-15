//! The **culled box meshing** of the voxel-meshing literature: given a set of disjoint
//! axis-aligned integer boxes and a neighbour-solidity oracle, decide which of each box's
//! six faces are *exposed* — part of the solid set's outer surface — and which are *hidden*
//! by a solid neighbour and may be culled.
//!
//! This is the pure **kernel** of a culled mesher: it never sees vertices, UVs, atlas
//! layers, materials, chunk seams, or wgpu. Its input is one [`Cuboid<T>`] (the box whose
//! faces are being tested — reusing the box type of [`GreedyCuboidDecomposition`](crate::solids::GreedyCuboidDecomposition)),
//! a unit axis step naming the face direction, and a caller-supplied predicate answering
//! *"is this integer lattice cell solid?"*. Its output is a single boolean per face: exposed
//! or not. Face geometry (the world-space rectangle, its winding, normals, and per-vertex
//! attributes) is the caller's to build from the box and the exposed direction — none of that
//! is a computer-science concern and none of it lives here.
//!
//! ## The culling decision
//!
//! A box is a solid, so a face can only be hidden from *outside*: the face at direction
//! `d` is **hidden** exactly when every cell of the one-cell-thick neighbour slab immediately
//! beyond that face is solid, and **exposed** the instant any such neighbour cell is not
//! solid (air, or outside the caller's populated domain). The slab is the box's face pushed
//! one cell along `d`: the face-normal axis collapses to the single layer just past the box,
//! while the two in-plane axes scan the box's full extent. A merged box therefore keeps
//! **one quad per face** — if a face is even partially exposed the whole face is reported
//! exposed (an over-draw of at most the box's own face, never a hole). Exactness of the
//! *silhouette* is preserved; the interior over-draw is depth-buried / back-face-culled by
//! the renderer downstream. This is the deliberate culled-meshing trade: cheap merged quads
//! at the cost of some hidden over-draw, versus per-voxel faces.
//!
//! ## The oracle
//!
//! The neighbour-solidity predicate is how domain concerns stay out of the kernel. It is
//! queried on **signed** integer lattice coordinates (a face on the low side of a box at the
//! domain origin queries a negative coordinate), and it must answer `false` for any cell
//! outside the populated region — an unpopulated cell is air, and air exposes the face. A
//! dense occupancy grid, a per-face seam-solidity flag replicated across the slab, or any
//! mixture of the two collapses into this one predicate at the caller's seam; the kernel
//! cannot tell which, and does not care.
//!
//! ## Literature
//!
//! This is the **culled meshing** of Lysenko, *Meshing in a Minecraft Game* (0fps, 2012) —
//! the culled-vs-greedy voxel-meshing taxonomy — and the older hidden-surface face-culling
//! folklore: emit a face only when the cell across it is empty. **Deviation:** the canonical
//! culled mesher culls the six faces of each individual *voxel*; here we cull the six faces
//! of pre-decomposed disjoint *boxes* — the output of the greedy box cover
//! ([`GreedyCuboidDecomposition`](crate::solids::GreedyCuboidDecomposition), the same Lysenko greedy
//! lineage). Culling merged-box faces is why the slab is scanned across the box's full
//! in-plane extent and why a partially-exposed face is reported whole, rather than the
//! single-cell test a per-voxel mesher uses.

use crate::solids::greedy_cuboid_decomposition::Cuboid;

/// The culled box mesher — a namespace for the [`CulledBoxMeshing::face_is_exposed`] culling
/// decision. Zero-sized: it carries no state, only the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CulledBoxMeshing;

impl CulledBoxMeshing {
    /// Is the `direction`-face of `box_` exposed under the neighbour-solidity oracle?
    ///
    /// Returns `true` when **any** cell of the one-cell-thick neighbour slab immediately
    /// beyond the face is not solid (so the face is part of the outer surface and must be
    /// emitted), and `false` when every neighbour cell is solid (the face is hidden and may
    /// be culled).
    ///
    /// * `box_` — the (inclusive–inclusive) integer box whose face is under test; only its
    ///   `min`/`max` extent is read, so the label type `T` is irrelevant here.
    /// * `direction` — a **unit axis step**: exactly one component is `+1` or `-1` and names
    ///   which of the six faces to test; the sign picks the low or high face on that axis.
    /// * `neighbour_is_solid` — the caller's oracle over **signed** lattice cells: `true`
    ///   iff the cell is backed by solid. Cells outside the populated domain must answer
    ///   `false` (air ⇒ exposed).
    ///
    /// The face-normal axis (the non-zero component of `direction`) collapses to the single
    /// neighbour layer one cell past the box; the other two axes scan the box's full extent.
    pub fn face_is_exposed<T>(
        box_: &Cuboid<T>,
        direction: [i32; 3],
        neighbour_is_solid: impl Fn([i64; 3]) -> bool,
    ) -> bool {
        // The box's inclusive extent per axis, as signed lattice bounds.
        let box_min = [box_.min[0] as i64, box_.min[1] as i64, box_.min[2] as i64];
        let box_max = [box_.max[0] as i64, box_.max[1] as i64, box_.max[2] as i64];

        // For the axis the face faces along, the neighbour plane is the single layer at the
        // box edge + direction; the other two axes scan the box's full extent.
        let scan_range = |axis: usize| -> (i64, i64) {
            if direction[axis] != 0 {
                let plane = if direction[axis] > 0 {
                    box_max[axis] + 1
                } else {
                    box_min[axis] - 1
                };
                (plane, plane)
            } else {
                (box_min[axis], box_max[axis])
            }
        };
        let (nx0, nx1) = scan_range(0);
        let (ny0, ny1) = scan_range(1);
        let (nz0, nz1) = scan_range(2);

        for nz in nz0..=nz1 {
            for ny in ny0..=ny1 {
                for nx in nx0..=nx1 {
                    // A single not-solid neighbour cell exposes the whole merged face.
                    if !neighbour_is_solid([nx, ny, nz]) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The six unit axis steps naming the cube faces (±X, ±Y, ±Z).
    const FACE_DIRECTIONS: [[i32; 3]; 6] = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ];

    fn unit_box() -> Cuboid<u16> {
        Cuboid {
            min: [1, 1, 1],
            max: [1, 1, 1],
            label: 7,
        }
    }

    #[test]
    fn isolated_box_exposes_all_six_faces() {
        // A box floating in air: every neighbour cell is air ⇒ every face exposed.
        let box_ = unit_box();
        for direction in FACE_DIRECTIONS {
            assert!(
                CulledBoxMeshing::face_is_exposed(&box_, direction, |_| false),
                "direction {direction:?} should be exposed against all-air",
            );
        }
    }

    #[test]
    fn fully_enclosed_box_culls_all_six_faces() {
        // Every neighbour cell is solid ⇒ no face exposed (fully interior box).
        let box_ = unit_box();
        for direction in FACE_DIRECTIONS {
            assert!(
                !CulledBoxMeshing::face_is_exposed(&box_, direction, |_| true),
                "direction {direction:?} should be culled against all-solid",
            );
        }
    }

    #[test]
    fn two_abutting_boxes_cull_their_shared_face() {
        // Box A = [0,0,0]; box B = [1,0,0] abut across the +X / -X seam. With B modelled as
        // solid by the oracle, A's +X face is hidden and B's -X face is hidden; the other ten
        // faces (against air) stay exposed.
        let a = Cuboid { min: [0, 0, 0], max: [0, 0, 0], label: 1u16 };
        let b = Cuboid { min: [1, 0, 0], max: [1, 0, 0], label: 2u16 };
        let cell_is_in_b = |cell: [i64; 3]| cell == [1, 0, 0];
        let cell_is_in_a = |cell: [i64; 3]| cell == [0, 0, 0];

        // A's +X face is backed by B ⇒ culled; its other five faces are air ⇒ exposed.
        assert!(!CulledBoxMeshing::face_is_exposed(&a, [1, 0, 0], cell_is_in_b));
        for direction in FACE_DIRECTIONS.into_iter().filter(|d| *d != [1, 0, 0]) {
            assert!(CulledBoxMeshing::face_is_exposed(&a, direction, cell_is_in_b));
        }
        // B's -X face is backed by A ⇒ culled; its other five faces are air ⇒ exposed.
        assert!(!CulledBoxMeshing::face_is_exposed(&b, [-1, 0, 0], cell_is_in_a));
        for direction in FACE_DIRECTIONS.into_iter().filter(|d| *d != [-1, 0, 0]) {
            assert!(CulledBoxMeshing::face_is_exposed(&b, direction, cell_is_in_a));
        }
    }

    #[test]
    fn oracle_solid_neighbour_culls_and_air_neighbour_exposes_the_same_face() {
        // The single boundary face at +X: a solid oracle culls it, an air oracle exposes it.
        let box_ = unit_box();
        assert!(!CulledBoxMeshing::face_is_exposed(&box_, [1, 0, 0], |cell| cell == [2, 1, 1]));
        assert!(CulledBoxMeshing::face_is_exposed(&box_, [1, 0, 0], |cell| cell != [2, 1, 1]));
    }

    #[test]
    fn partially_backed_merged_face_is_reported_exposed() {
        // A merged face over a 3-wide slab: back only two of its three neighbour cells solid;
        // the one air cell must expose the WHOLE merged quad (the culled-merged over-draw rule).
        let box_ = Cuboid { min: [0, 0, 0], max: [2, 0, 0], label: 5u16 };
        // +Y neighbour slab is the cells (0,1,0),(1,1,0),(2,1,0); leave (2,1,0) air.
        let all_but_one_solid = |cell: [i64; 3]| cell[1] == 1 && cell[0] < 2;
        assert!(CulledBoxMeshing::face_is_exposed(&box_, [0, 1, 0], all_but_one_solid));
        // Back all three ⇒ the merged face is fully hidden ⇒ culled.
        let full_slab_solid = |cell: [i64; 3]| cell[1] == 1;
        assert!(!CulledBoxMeshing::face_is_exposed(&box_, [0, 1, 0], full_slab_solid));
    }

    #[test]
    fn low_side_face_queries_negative_lattice_cell() {
        // A box at the domain origin: its -X face queries x = -1, which the oracle answers air.
        let box_ = Cuboid { min: [0, 0, 0], max: [0, 0, 0], label: 3u16 };
        let queried_negative = std::cell::Cell::new(false);
        let exposed = CulledBoxMeshing::face_is_exposed(&box_, [-1, 0, 0], |cell| {
            if cell[0] < 0 {
                queried_negative.set(true);
            }
            false
        });
        assert!(exposed);
        assert!(queried_negative.get(), "the low face must probe the negative neighbour cell");
    }
}
