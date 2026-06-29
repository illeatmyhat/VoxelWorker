//! 2D **sketch → extrude → volume** — the sketch-to-volume authoring atom
//! (ADR 0003 §3i, Slice 2a).
//!
//! This is a SECOND [`VoxelProducer`](crate::voxel::VoxelProducer), added
//! **alongside** [`SdfShape`](crate::voxel::SdfShape) (NOT replacing it). It takes
//! a grid-aligned plane plus a closed polygon *profile* of voxel-granular points
//! and extrudes that profile a whole number of voxels along the plane normal,
//! producing a prism. It is the engine the §3i build arc reframes primitives as
//! sugar over — a rectangle profile extruded *is* a box, a circle profile extruded
//! *is* a cylinder — so it resolves through the SAME stamp / `CombineOp` / chunk
//! path the SDF producer already uses.
//!
//! **Leak-free by construction (§3i leak-retirement).** The profile points and the
//! extrude span are integer voxels on the lattice/sub-lattice — there is no
//! implicit centre anchor and so no half-block correction. The producer emits its
//! voxels centred on its own origin-centred grid exactly the way `SdfShape` does
//! (centres at `idx + 0.5 − grid/2`), but its placement does NOT route through
//! `leaf_lattice_shift_voxels`: a sketch's footprint is corner-anchored, so the
//! block-lattice shift the implicit-centre model needed is identically zero. (The
//! resolve path treats a sketch leaf like a Part — no intrinsic block size, no
//! lattice snap — see `Scene::resolve_*`.)
//!
//! 2a SCOPE: AXIS-ALIGNED planes only (the normal is one of ±X / ±Y / ±Z). A
//! free-angle sketch plane is the deferred plane-orientation milestone (§3f(a)).
//! The profile is a closed simple polygon (≥3 points); a degenerate profile
//! (fewer than 3 points, or zero area) resolves to nothing rather than panicking.

use crate::voxel::{Voxel, VoxelGrid, VoxelProducer, MAX_GRID_VOXELS};

/// Which axis the sketch plane's normal points along — i.e. the axis the profile
/// is EXTRUDED along (ADR 0003 §3i, 2a axis-aligned scope).
///
/// The two in-plane axes (the ones the 2D profile lives in) are the OTHER two
/// world axes, taken in ascending order so the mapping is unambiguous:
///
/// | normal | in-plane axis 0 | in-plane axis 1 |
/// |--------|-----------------|-----------------|
/// | `X`    | Y               | Z               |
/// | `Y`    | X               | Z               |
/// | `Z`    | X               | Y               |
///
/// Sign of the normal does not change the resolved occupancy (an axis-aligned
/// prism is symmetric about its own grid), so 2a stores the bare axis; a signed
/// normal is only meaningful once on-surface sketching (§3i, Slice 2b) needs a
/// facing direction, which is a later concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlaneAxis {
    /// Profile in the YZ plane, extruded along X.
    X,
    /// Profile in the XZ plane, extruded along Y.
    Y,
    /// Profile in the XY plane, extruded along Z (Z-up: the footprint-extrude-up
    /// default — profile on the XY ground, extruded up along +Z).
    Z,
}

impl PlaneAxis {
    /// The two WORLD axes the 2D profile lives in, in ascending order
    /// (`in_plane_axes()[0]` is profile coordinate 0, `[1]` is profile
    /// coordinate 1). The remaining axis is the extrude/normal axis.
    pub fn in_plane_axes(self) -> [usize; 2] {
        match self {
            PlaneAxis::X => [1, 2], // Y, Z
            PlaneAxis::Y => [0, 2], // X, Z
            PlaneAxis::Z => [0, 1], // X, Y
        }
    }

    /// The WORLD axis the profile is extruded along (the plane normal).
    pub fn normal_axis(self) -> usize {
        match self {
            PlaneAxis::X => 0,
            PlaneAxis::Y => 1,
            PlaneAxis::Z => 2,
        }
    }
}

/// One vertex of a sketch profile — a 2D point, voxel-granular at the document's
/// density `d` (ADR 0003 §3f(0) `offset_voxels` integer-voxel convention, the same
/// representation as `ShapePoint::Inline` and `NodeTransform.offset_voxels`).
///
/// The two coordinates are in the plane's in-plane axes (see
/// [`PlaneAxis::in_plane_axes`]). They may be negative; the producer normalizes the
/// profile's bounding box to the local grid origin at resolve, so absolute values
/// only matter relative to the other points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SketchPoint {
    /// In-plane voxel coordinates `[axis0, axis1]` at the document density `d`.
    pub offset_voxels: [i64; 2],
}

impl SketchPoint {
    /// A profile vertex at the given in-plane voxel coordinates.
    pub fn new(axis0: i64, axis1: i64) -> Self {
        Self {
            offset_voxels: [axis0, axis1],
        }
    }
}

/// A grid-aligned PLANE plus a closed POLYGON PROFILE of ordered points (ADR 0003
/// §3i). The profile is a closed simple polygon: the last vertex connects back to
/// the first, so the points list does NOT repeat the start vertex.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Sketch {
    /// Which axis the plane normal points along (2a: axis-aligned only).
    pub plane: PlaneAxis,
    /// The ordered profile vertices (≥3 for a non-degenerate polygon).
    pub profile: Vec<SketchPoint>,
}

impl Sketch {
    /// A sketch on `plane` with the given ordered profile.
    pub fn new(plane: PlaneAxis, profile: Vec<SketchPoint>) -> Self {
        Self { plane, profile }
    }

    /// A rectangle profile spanning `[0, width] × [0, height]` voxels on `plane`
    /// (the degenerate "box footprint" — proves box = rectangle-extrude sugar,
    /// §3i). The four corners are wound counter-clockwise; winding does not affect
    /// the even-odd rasterizer.
    pub fn rectangle(plane: PlaneAxis, width_voxels: i64, height_voxels: i64) -> Self {
        Self::new(
            plane,
            vec![
                SketchPoint::new(0, 0),
                SketchPoint::new(width_voxels, 0),
                SketchPoint::new(width_voxels, height_voxels),
                SketchPoint::new(0, height_voxels),
            ],
        )
    }
}

/// A [`Sketch`] extruded a whole number of voxels along its plane normal,
/// producing a prism — the 2a sketch→volume producer (ADR 0003 §3i). Added
/// **alongside** `SdfShape`; both implement [`VoxelProducer`] and resolve through
/// the same stamp / `CombineOp` / chunk path.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SketchExtrude {
    /// The closed 2D profile + its plane.
    pub sketch: Sketch,
    /// Extrude span in voxels along the plane normal (≥1 for a non-empty prism).
    pub height_voxels: u32,
}

impl SketchExtrude {
    /// A sketch extruded `height_voxels` along its plane normal.
    pub fn new(sketch: Sketch, height_voxels: u32) -> Self {
        Self {
            sketch,
            height_voxels,
        }
    }

    /// The profile's 2D bounding box in voxels as `(min, max)` half-open per
    /// in-plane axis, or `None` for a degenerate profile (fewer than 3 points or a
    /// zero-extent span on either in-plane axis). The local in-plane grid is sized
    /// `max − min`; cells are addressed from `min`.
    fn profile_bounds(&self) -> Option<([i64; 2], [i64; 2])> {
        if self.sketch.profile.len() < 3 || self.height_voxels == 0 {
            return None;
        }
        let first = self.sketch.profile[0].offset_voxels;
        let mut min = first;
        let mut max = first;
        for point in &self.sketch.profile {
            for axis in 0..2 {
                min[axis] = min[axis].min(point.offset_voxels[axis]);
                max[axis] = max[axis].max(point.offset_voxels[axis]);
            }
        }
        // A zero-extent span on either in-plane axis is a degenerate (collinear /
        // zero-area) profile: no cell can be inside it.
        if max[0] <= min[0] || max[1] <= min[1] {
            return None;
        }
        Some((min, max))
    }

    /// The resolved grid's voxel dimensions `[x, y, z]` (the prism's AABB), or
    /// `[0, 0, 0]` for a degenerate profile. The two in-plane axes get the
    /// profile's bounding-box span; the normal axis gets `height_voxels`.
    pub fn grid_dimensions(&self) -> [u32; 3] {
        let Some((min, max)) = self.profile_bounds() else {
            return [0, 0, 0];
        };
        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        let mut dimensions = [0u32; 3];
        // Saturating downcast: a profile span exceeding u32::MAX must clamp to a huge
        // dimension (rejected by downstream bounds), never silently wrap to a small one.
        dimensions[in_plane_0] = u32::try_from(max[0] - min[0]).unwrap_or(u32::MAX);
        dimensions[in_plane_1] = u32::try_from(max[1] - min[1]).unwrap_or(u32::MAX);
        dimensions[normal] = self.height_voxels;
        dimensions
    }

    /// Total sampling-grid voxel count (`x · y · z`) as `u64` so it can't overflow.
    pub fn grid_voxel_count(&self) -> u64 {
        let [x, y, z] = self.grid_dimensions();
        x as u64 * y as u64 * z as u64
    }

    /// Whether the prism's AABB exceeds [`MAX_GRID_VOXELS`] — the same single-shape
    /// sanity cap `SdfShape::exceeds_voxel_cap` applies, so a pathological
    /// profile/height can't blow memory on a lone resolve.
    pub fn exceeds_voxel_cap(&self) -> bool {
        self.grid_voxel_count() > MAX_GRID_VOXELS
    }
}

/// Even-odd (ray-crossing) point-in-polygon test for the cell whose centre sits at
/// `(sample_0, sample_1)` in the profile's own (un-normalized) coordinate space.
///
/// The classic crossing-number test: count how many polygon edges a ray cast in
/// the +axis1 direction from the sample point crosses. An odd count is inside.
/// Edges are taken between consecutive vertices, closing the last → first. Cell
/// centres sit at half-integer positions (`min + i + 0.5`), which never coincide
/// with the integer-coordinate vertices/edges, so there are no on-boundary
/// ambiguities — the rasterization is exact and deterministic.
fn point_in_polygon(profile: &[SketchPoint], sample_0: f64, sample_1: f64) -> bool {
    let mut inside = false;
    let count = profile.len();
    let mut previous = count - 1;
    for current in 0..count {
        let current_point = profile[current].offset_voxels;
        let previous_point = profile[previous].offset_voxels;
        let current_0 = current_point[0] as f64;
        let current_1 = current_point[1] as f64;
        let previous_0 = previous_point[0] as f64;
        let previous_1 = previous_point[1] as f64;
        // Does a horizontal-in-axis1 ray from the sample cross this edge?
        let straddles = (current_1 > sample_1) != (previous_1 > sample_1);
        if straddles {
            // X (axis0) of the edge at the sample's axis1 height.
            let crossing_0 = (previous_0 - current_0) * (sample_1 - current_1)
                / (previous_1 - current_1)
                + current_0;
            if sample_0 < crossing_0 {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

impl VoxelProducer for SketchExtrude {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let dimensions = self.grid_dimensions();
        grid.dimensions = dimensions;
        grid.occupied.clear();

        let Some((min, _max)) = self.profile_bounds() else {
            // Degenerate profile: empty occupancy, no panic (§3i edge case).
            return;
        };

        let [in_plane_0, in_plane_1] = self.sketch.plane.in_plane_axes();
        let normal = self.sketch.plane.normal_axis();
        let in_plane_span_0 = dimensions[in_plane_0];
        let in_plane_span_1 = dimensions[in_plane_1];
        let density = voxels_per_block.max(1);

        // Rasterize the 2D profile ONCE (axis-aligned extrusion ⇒ the same fill on
        // every layer along the normal — §3i, cheap + predictable), then sweep it
        // across the `height_voxels` layers. A cell `(cell_0, cell_1)` at local
        // origin `min` is occupied iff its centre `(min + cell + 0.5)` is inside the
        // polygon (even-odd test at the cell centre — §3i).
        let mut filled_in_plane: Vec<[u32; 2]> =
            Vec::with_capacity((in_plane_span_0 as usize) * (in_plane_span_1 as usize));
        for cell_1 in 0..in_plane_span_1 {
            let sample_1 = min[1] as f64 + cell_1 as f64 + 0.5;
            for cell_0 in 0..in_plane_span_0 {
                let sample_0 = min[0] as f64 + cell_0 as f64 + 0.5;
                if point_in_polygon(&self.sketch.profile, sample_0, sample_1) {
                    filled_in_plane.push([cell_0, cell_1]);
                }
            }
        }

        grid.occupied.reserve(filled_in_plane.len() * self.height_voxels as usize);
        // The voxel's grid index per world axis, assembled from the in-plane cell
        // and the normal layer, then CORNER-ANCHORED (centre = idx + 0.5) exactly the
        // way `SdfShape::resolve` does, so a rectangle extrude is byte-identical to the
        // matching `Box`. The centre is a half-integer for any grid size → always on
        // the global voxel lattice.
        for layer in 0..self.height_voxels {
            for &[cell_0, cell_1] in &filled_in_plane {
                let mut index = [0u32; 3];
                index[in_plane_0] = cell_0;
                index[in_plane_1] = cell_1;
                index[normal] = layer;
                let world_position = [
                    index[0] as f32 + 0.5,
                    index[1] as f32 + 0.5,
                    index[2] as f32 + 0.5,
                ];
                grid.occupied.push(Voxel {
                    world_position,
                    block_local_coord: [
                        (index[0] % density) as u8,
                        (index[1] % density) as u8,
                        (index[2] % density) as u8,
                    ],
                    material_id: 0,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{SdfShape, ShapeKind, VoxelProducer};
    use std::collections::BTreeSet;

    /// Collect a producer's occupied voxels as a sorted set of
    /// `(world_position_bits, block_local_coord, material_id)` so two producers can
    /// be compared for SET equality independent of emission order. World positions
    /// are integer + 0.5, so the f32 bit pattern is exact and hashable.
    fn occupancy_set(
        producer: &dyn VoxelProducer,
        density: u32,
    ) -> BTreeSet<([i32; 3], [u8; 3], u16)> {
        let mut grid = VoxelGrid::default();
        producer.resolve(&mut grid, density);
        grid.occupied
            .iter()
            .map(|voxel| {
                (
                    [
                        (voxel.world_position[0] * 2.0).round() as i32,
                        (voxel.world_position[1] * 2.0).round() as i32,
                        (voxel.world_position[2] * 2.0).round() as i32,
                    ],
                    voxel.block_local_coord,
                    voxel.material_id,
                )
            })
            .collect()
    }

    /// LOAD-BEARING: a rectangle-profile extrude produces EXACTLY the same occupied
    /// voxel set (positions, block-local coords, materials) as the axis-aligned
    /// `Box` `SdfShape` of the same size/placement/density. This is the "box =
    /// rectangle-extrude sugar" proof (§3i). Covered for several sizes including an
    /// odd size and density 16.
    #[test]
    fn rectangle_extrude_equals_box() {
        // (size_blocks, density). An odd size (3) and density 16 are included.
        let cases = [
            ([2u32, 2, 2], 4u32),
            ([3, 1, 5], 4),
            ([3, 3, 3], 16),
            ([4, 2, 6], 16),
        ];
        for (size_blocks, density) in cases {
            let box_shape = SdfShape {
                kind: ShapeKind::Box,
                size_blocks,
                wall_blocks: 1,
            };
            let grid_x = (size_blocks[0] * density) as i64;
            let grid_y = (size_blocks[1] * density) as i64;
            let grid_z = (size_blocks[2] * density) as i64;
            // Plane Y: profile in XZ (width = X span, height = Z span), extruded
            // grid_y voxels along Y — matches the box's [x, y, z] grid exactly.
            let extrude = SketchExtrude::new(
                Sketch::rectangle(PlaneAxis::Y, grid_x, grid_z),
                grid_y as u32,
            );
            assert_eq!(
                extrude.grid_dimensions(),
                box_shape.grid_dimensions(density),
                "grid dims must match for size {size_blocks:?} @ d{density}"
            );
            assert_eq!(
                occupancy_set(&extrude, density),
                occupancy_set(&box_shape, density),
                "rectangle extrude must equal Box for size {size_blocks:?} @ d{density}"
            );
        }
    }

    /// A rectangle extrude on EACH of the three axis-aligned plane orientations
    /// equals the matching `Box` — proves the plane mapping is correct for X, Y, Z.
    #[test]
    fn rectangle_extrude_each_plane_equals_box() {
        let density = 4u32;
        let size_blocks = [2u32, 3, 4];
        let box_shape = SdfShape {
            kind: ShapeKind::Box,
            size_blocks,
            wall_blocks: 1,
        };
        let dims = box_shape.grid_dimensions(density);
        let box_set = occupancy_set(&box_shape, density);
        for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
            let [in_plane_0, in_plane_1] = plane.in_plane_axes();
            let normal = plane.normal_axis();
            let extrude = SketchExtrude::new(
                Sketch::rectangle(plane, dims[in_plane_0] as i64, dims[in_plane_1] as i64),
                dims[normal],
            );
            assert_eq!(
                extrude.grid_dimensions(),
                dims,
                "plane {plane:?} grid dims must match the box AABB"
            );
            assert_eq!(
                occupancy_set(&extrude, density),
                box_set,
                "plane {plane:?} rectangle extrude must equal the same Box"
            );
        }
    }

    /// CONCAVITY / added value: an L-shaped (non-convex) hexagon profile extrudes to
    /// the correct occupancy. A box CANNOT make this; the reflex vertex exercises the
    /// rasterizer. The L is a 4×4 square with its top-right 2×2 quadrant removed:
    ///
    /// ```text
    /// axis1
    ///  3 | X X . .
    ///  2 | X X . .
    ///  1 | X X X X
    ///  0 | X X X X
    ///      0 1 2 3  axis0
    /// ```
    #[test]
    fn l_shape_extrude_occupancy() {
        // Profile (CCW) of the L: outer rectangle 0..4 × 0..2, plus left column
        // 0..2 × 2..4. Six vertices, one reflex corner at (2, 2).
        let profile = vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(4, 2),
            SketchPoint::new(2, 2), // reflex vertex
            SketchPoint::new(2, 4),
            SketchPoint::new(0, 4),
        ];
        let extrude = SketchExtrude::new(Sketch::new(PlaneAxis::Y, profile), 1);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, 4);
        assert_eq!(grid.dimensions, [4, 1, 4], "L AABB is 4×1×4");

        // Recover the in-plane cell of each voxel (plane Y ⇒ axes X, Z). Corner-
        // anchored: centres are `idx + 0.5`, so the cell index is `world − 0.5`.
        let mut cells: BTreeSet<(i64, i64)> = BTreeSet::new();
        for voxel in &grid.occupied {
            let cell_x = (voxel.world_position[0] - 0.5).round() as i64;
            let cell_z = (voxel.world_position[2] - 0.5).round() as i64;
            cells.insert((cell_x, cell_z));
        }
        // The L occupies the full bottom two rows (z 0..2, all x) and the left two
        // columns of the top two rows (x 0..2, z 2..4) — 8 + 4 = 12 cells.
        let mut expected: BTreeSet<(i64, i64)> = BTreeSet::new();
        for x in 0..4 {
            for z in 0..2 {
                expected.insert((x, z));
            }
        }
        for x in 0..2 {
            for z in 2..4 {
                expected.insert((x, z));
            }
        }
        assert_eq!(cells, expected, "L footprint occupancy is wrong");
        // Spot-check specific in/out cells: filled bottom-right corner, EMPTY
        // top-right quadrant (the removed 2×2 a box could not exclude).
        assert!(cells.contains(&(3, 0)), "(3,0) inside the L");
        assert!(cells.contains(&(0, 3)), "(0,3) inside the L left column");
        assert!(!cells.contains(&(3, 3)), "(3,3) is the removed quadrant");
        assert!(!cells.contains(&(2, 2)), "(2,2) is outside the reflex corner");
    }

    /// EDGE CASE: degenerate profiles resolve to empty occupancy without panicking —
    /// fewer than 3 points, collinear (zero-area) points, and a zero height.
    #[test]
    fn degenerate_profiles_are_empty() {
        let empty = |producer: &SketchExtrude| {
            let mut grid = VoxelGrid::default();
            producer.resolve(&mut grid, 4);
            assert!(grid.occupied.is_empty());
            assert_eq!(grid.dimensions, [0, 0, 0]);
        };
        // < 3 points.
        empty(&SketchExtrude::new(
            Sketch::new(PlaneAxis::Y, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]),
            2,
        ));
        // Collinear (zero-area) — three points on one line.
        empty(&SketchExtrude::new(
            Sketch::new(
                PlaneAxis::Y,
                vec![
                    SketchPoint::new(0, 0),
                    SketchPoint::new(2, 0),
                    SketchPoint::new(4, 0),
                ],
            ),
            2,
        ));
        // Zero height.
        empty(&SketchExtrude::new(Sketch::rectangle(PlaneAxis::Y, 4, 4), 0));
    }

    /// EDGE CASE: a sub-block-precise profile at d=16 (a vertex NOT on a block
    /// boundary) rasterizes correctly. The profile is a 20×20-voxel square (1.25
    /// blocks per side at d16) whose extent is not a whole number of blocks; the fill
    /// is exactly the 20×20 cell set on every layer.
    #[test]
    fn sub_block_precise_profile_at_d16() {
        let density = 16u32;
        // 20 voxels = 1 block + 4 voxels — a sub-block extent on a non-block boundary.
        let extrude = SketchExtrude::new(Sketch::rectangle(PlaneAxis::Y, 20, 20), 3);
        assert_eq!(extrude.grid_dimensions(), [20, 3, 20]);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, density);
        // A full 20×3×20 rectangular prism.
        assert_eq!(grid.occupied.len(), 20 * 3 * 20);
        // block_local_coord wraps at the density: a cell at in-plane index 17 has
        // block-local X = 17 % 16 = 1 (proves the sub-block fraction is carried).
        let has_local_one = grid.occupied.iter().any(|voxel| {
            // Corner-anchored: cell index = world − 0.5.
            let cell_x = (voxel.world_position[0] - 0.5).round() as i64;
            cell_x == 17 && voxel.block_local_coord[0] == 1
        });
        assert!(has_local_one, "sub-block block_local_coord must wrap at d=16");
    }

    /// A non-rectangular extrude still matches between `grid_dimensions` and the
    /// resolved grid's `dimensions`, and respects the voxel cap predicate.
    #[test]
    fn grid_dimensions_consistent_and_cap() {
        let extrude = SketchExtrude::new(Sketch::rectangle(PlaneAxis::Z, 6, 4), 5);
        let mut grid = VoxelGrid::default();
        extrude.resolve(&mut grid, 16);
        assert_eq!(grid.dimensions, extrude.grid_dimensions());
        assert!(!extrude.exceeds_voxel_cap());
    }
}
