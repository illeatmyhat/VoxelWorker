use super::*;
use crate::voxel::{SdfShape, VoxelProducer};
use voxel_core::voxel::{ShapeKind, VoxelGrid};
use std::collections::BTreeSet;

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
        let box_shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
        let grid_x = (size_blocks[0] * density) as i64;
        let grid_y = (size_blocks[1] * density) as i64;
        let grid_z = (size_blocks[2] * density) as i64;
        // Plane Y: profile in XZ (width = X span, height = Z span), extruded
        // grid_y voxels along Y — matches the box's [x, y, z] grid exactly.
        let extrude = SketchSolid::extrude(
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
    let box_shape = SdfShape::from_blocks(ShapeKind::Box, size_blocks, 1, density);
    let dims = box_shape.grid_dimensions(density);
    let box_set = occupancy_set(&box_shape, density);
    for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
        let [in_plane_0, in_plane_1] = plane.in_plane_axes();
        let normal = plane.normal_axis();
        let extrude = SketchSolid::extrude(
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
    let extrude = SketchSolid::extrude(Sketch::new(PlaneAxis::Y, profile), 1);
    let mut grid = VoxelGrid::default();
    extrude.resolve(&mut grid, 4);
    assert_eq!(grid.dimensions, [4, 1, 4], "L AABB is 4×1×4");

    // Recover the in-plane cell of each voxel (plane Y ⇒ axes X, Z). Corner-
    // anchored: centres are `idx + 0.5`, so the cell index is `world − 0.5`.
    let mut cells: BTreeSet<(i64, i64)> = BTreeSet::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let cell_x = (position[0] - 0.5).round() as i64;
        let cell_z = (position[2] - 0.5).round() as i64;
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
    let empty = |producer: &SketchSolid| {
        let mut grid = VoxelGrid::default();
        producer.resolve(&mut grid, 4);
        assert!(grid.occupied.is_empty());
        assert_eq!(grid.dimensions, [0, 0, 0]);
    };
    // < 3 points.
    empty(&SketchSolid::extrude(
        Sketch::new(PlaneAxis::Y, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]),
        2,
    ));
    // Collinear (zero-area) — three points on one line.
    empty(&SketchSolid::extrude(
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
    empty(&SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Y, 4, 4), 0));
}

/// EDGE CASE: a sub-block-precise profile at d=16 (a vertex NOT on a block
/// boundary) rasterizes correctly. The profile is a 20×20-voxel square (1.25
/// blocks per side at d16) whose extent is not a whole number of blocks; the fill
/// is exactly the 20×20 cell set on every layer.
#[test]
fn sub_block_precise_profile_at_d16() {
    let density = 16u32;
    // 20 voxels = 1 block + 4 voxels — a sub-block extent on a non-block boundary.
    let extrude = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Y, 20, 20), 3);
    assert_eq!(extrude.grid_dimensions(), [20, 3, 20]);
    let mut grid = VoxelGrid::default();
    extrude.resolve(&mut grid, density);
    // A full 20×3×20 rectangular prism.
    assert_eq!(grid.occupied.len(), 20 * 3 * 20);
    // block_local_coord wraps at the density: a cell at in-plane index 17 has
    // block-local X = 17 % 16 = 1 (proves the sub-block fraction is carried).
    let has_local_one = grid.occupied.iter().any(|voxel| {
        // Corner-anchored: cell index = world − 0.5.
        let cell_x = (voxel.world_position()[0] - 0.5).round() as i64;
        cell_x == 17 && voxel.block_local_coord[0] == 1
    });
    assert!(has_local_one, "sub-block block_local_coord must wrap at d=16");
}

/// The rectangle-detection helper the inspector uses to choose editable
/// Width/Depth vs. a read-only custom-profile note: a `rectangle` profile is
/// detected (returning its in-plane spans) regardless of plane; an L-shape and a
/// degenerate (triangle) profile are not; and a four-point profile whose corners
/// are NOT the bounding-box corners (a non-axis-aligned quad) is rejected.
#[test]
fn rectangle_in_plane_spans_detection() {
    // A genuine rectangle on each plane reports its [width, depth] spans.
    for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
        let extrude = SketchSolid::extrude(Sketch::rectangle(plane, 6, 4), 3);
        assert_eq!(
            extrude.rectangle_in_plane_spans(),
            Some([6, 4]),
            "plane {plane:?} rectangle must report its spans"
        );
    }
    // An L-shape (six vertices) is not a rectangle.
    let l_profile = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(4, 0),
        SketchPoint::new(4, 2),
        SketchPoint::new(2, 2),
        SketchPoint::new(2, 4),
        SketchPoint::new(0, 4),
    ];
    assert_eq!(
        SketchSolid::extrude(Sketch::new(PlaneAxis::Z, l_profile), 1)
            .rectangle_in_plane_spans(),
        None,
        "an L-shape is not a rectangle"
    );
    // A four-point quad whose corners are NOT the bounding-box corners (a
    // diamond) must be rejected — its vertices lie on edge midpoints, not corners.
    let diamond = vec![
        SketchPoint::new(2, 0),
        SketchPoint::new(4, 2),
        SketchPoint::new(2, 4),
        SketchPoint::new(0, 2),
    ];
    assert_eq!(
        SketchSolid::extrude(Sketch::new(PlaneAxis::Z, diamond), 1)
            .rectangle_in_plane_spans(),
        None,
        "a diamond quad is not an axis-aligned rectangle"
    );
    // A degenerate (triangle / <4 point) profile is not a rectangle.
    let triangle = vec![
        SketchPoint::new(0, 0),
        SketchPoint::new(4, 0),
        SketchPoint::new(0, 4),
    ];
    assert_eq!(
        SketchSolid::extrude(Sketch::new(PlaneAxis::Z, triangle), 1)
            .rectangle_in_plane_spans(),
        None,
        "a triangle is not a rectangle"
    );
}

/// A non-rectangular extrude still matches between `grid_dimensions` and the
/// resolved grid's `dimensions`, and respects the voxel cap predicate.
#[test]
fn grid_dimensions_consistent_and_cap() {
    let extrude = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 6, 4), 5);
    let mut grid = VoxelGrid::default();
    extrude.resolve(&mut grid, 16);
    assert_eq!(grid.dimensions, extrude.grid_dimensions());
    assert!(!extrude.exceeds_voxel_cap());
}
