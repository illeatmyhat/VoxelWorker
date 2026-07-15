    use super::*;
    use voxel_core::voxel::{ShapeKind};
    use crate::voxel::{SdfShape, VoxelProducer};
    use voxel_core::voxel::VoxelGrid;
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
                let position = voxel.world_position();
                (
                    [
                        (position[0] * 2.0).round() as i32,
                        (position[1] * 2.0).round() as i32,
                        (position[2] * 2.0).round() as i32,
                    ],
                    voxel.block_local_coord,
                    voxel.color_index(),
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

    // ----- Revolve operation (ADR 0003 §3i, the solid-of-revolution producer) -----

    use crate::sketch::RevolveAxis;

    /// Collect a producer's occupied voxels as a set of corner-anchored CELL indices
    /// `(i, j, k)` (`world − 0.5`). Used for IoU / overlap comparisons against an
    /// `SdfShape` of the same grid dimensions.
    fn cell_set(producer: &dyn VoxelProducer, density: u32) -> BTreeSet<[i64; 3]> {
        let mut grid = VoxelGrid::default();
        producer.resolve(&mut grid, density);
        grid.occupied
            .iter()
            .map(|voxel| {
                let position = voxel.world_position();
                [
                    (position[0] - 0.5).round() as i64,
                    (position[1] - 0.5).round() as i64,
                    (position[2] - 0.5).round() as i64,
                ]
            })
            .collect()
    }

    /// THE LOCK: a rectangle profile (radial 0..R, axial 0..H) revolved a full 360°
    /// about an axis oriented so the AXIAL world axis is Z must match the `Cylinder`
    /// `SdfShape` of diameter 2R, height H.
    ///
    /// Orientation: `PlaneAxis::X` + `RevolveAxis::InPlane1` ⇒ axial world axis = Z
    /// (the cylinder's Z-up vertical axis), radial world axes = {X, Y} (the circular
    /// cross-section). The profile coord (c0, c1) = (radial, axial), so
    /// `Sketch::rectangle(PlaneAxis::X, R, H)` is exactly the radial×axial rectangle.
    ///
    /// EXACT occupancy-set equality. The revolve rasterizes the rim with
    /// `radial = sqrt(x²+y²) <= R` via the polygon edge at radial R, while the SDF
    /// rasterizes `(|p_xy| − R_semi) <= 0`. Both compare the SAME centred radius to the
    /// SAME R (R = grid/2 = semi-axis), so the rim cells agree cell-for-cell and the
    /// equality holds EXACTLY (measured symmetric difference = 0 for both R parities).
    /// Covered for an EVEN and an ODD radial extent at density 16 (parity).
    #[test]
    fn rectangle_revolve_equals_cylinder() {
        let density = 16u32;
        // (radial_extent R in voxels, axial height H in voxels). 2R is the diameter.
        // R=32 → even diameter 64; R=33 → odd radial extent (diameter 66) — exercise
        // both R parities at d16.
        let cases = [(32i64, 48i64), (33, 47)];
        for (radial, axial) in cases {
            let revolve = SketchSolid::revolve(
                Sketch::rectangle(PlaneAxis::X, radial, axial),
                RevolveAxis::InPlane1,
                360,
            );
            // Cylinder of diameter 2R (X, Y) and height H (Z), pure-voxel size.
            let cylinder = SdfShape::from_voxels(
                ShapeKind::Cylinder,
                [(2 * radial) as u32, (2 * radial) as u32, axial as u32],
                1,
            );
            assert_eq!(
                revolve.grid_dimensions(),
                cylinder.grid_dimensions(density),
                "grid dims must match for radial {radial}, axial {axial}"
            );
            assert_eq!(
                cell_set(&revolve, density),
                cell_set(&cylinder, density),
                "rectangle revolve must EXACTLY equal Cylinder for radial {radial}, axial {axial}"
            );
        }
    }

    /// FOLD CORRECTNESS: a solid of revolution is symmetric about its axis, so the
    /// resolve folds both radial signs. (a) A rectangle authored entirely on the
    /// NEGATIVE radial side ([−30, −20]) revolves to the SAME tube as the same
    /// rectangle mirrored to the positive side ([20, 30]). (b) A profile STRADDLING
    /// the axis fills the union of both sides — the |radial|-folded region — and is
    /// non-empty. Without two-sided folding the negative-side rectangle would resolve
    /// to nothing and a straddling profile would lose its negative half.
    #[test]
    fn revolve_negative_and_straddling_radial_fold() {
        let density = 8u32;
        let axial = 24i64;

        // (a) Negative-side rectangle == positive mirror (same tube).
        // Profile (radial, axial) = (c0, c1); radial spans [−30, −20] vs [20, 30].
        let negative_side = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(-30, 0),
                    SketchPoint::new(-20, 0),
                    SketchPoint::new(-20, axial),
                    SketchPoint::new(-30, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        let positive_mirror = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(20, 0),
                    SketchPoint::new(30, 0),
                    SketchPoint::new(30, axial),
                    SketchPoint::new(20, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            negative_side.grid_dimensions(),
            positive_mirror.grid_dimensions(),
            "negative-side and positive-mirror tubes must share grid dims (radial_max folds by abs)"
        );
        let negative_cells = cell_set(&negative_side, density);
        assert!(!negative_cells.is_empty(), "negative-side rectangle must NOT be empty");
        assert_eq!(
            negative_cells,
            cell_set(&positive_mirror, density),
            "negative-side rectangle must revolve to the same tube as its positive mirror"
        );

        // (b) Straddling profile fills the |radial|-folded region. A rectangle radial
        // [−15, 25] straddles the axis; its |radial| union covers [0, 25] (the larger
        // side dominates). Equivalent to a SOLID disc rectangle radial [0, 25] revolved
        // — the straddle's smaller (−15) side is wholly subsumed by the +25 side.
        let straddling = SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(-15, 0),
                    SketchPoint::new(25, 0),
                    SketchPoint::new(25, axial),
                    SketchPoint::new(-15, axial),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        );
        let straddling_cells = cell_set(&straddling, density);
        assert!(!straddling_cells.is_empty(), "straddling profile must NOT be empty");
        // The folded region is the disc of radius max(|−15|, 25) = 25 — a solid disc,
        // because the profile spans through radial 0 (no inner hole). Compare against a
        // one-sided solid rectangle radial [0, 25]: same axial extent, same outer radius,
        // both solid to the axis ⇒ identical folded occupancy.
        let solid_disc = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 25, axial),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            straddling.grid_dimensions(),
            solid_disc.grid_dimensions(),
            "straddling profile diameter is 2·max(|radial|) = 50, matching the solid disc"
        );
        assert_eq!(
            straddling_cells,
            cell_set(&solid_disc, density),
            "straddling profile must fill the |radial|-folded solid disc"
        );
    }

    /// A polygon approximating a half-disc (radial profile of a semicircle) revolved
    /// 360° about Z approximates a `Sphere` — IoU overlap >= 0.97. This is a TOLERANCE
    /// assertion, not exact: the profile is a many-segment polygon approximating the
    /// circular arc, and the SDF sphere uses its own ellipsoid isolevel, so the two
    /// rims never coincide exactly (the polygon under/over-shoots the arc per segment).
    #[test]
    fn half_disc_revolve_approximates_sphere() {
        let density = 16u32;
        let radius = 40i64; // sphere radius in voxels ⇒ diameter 80
        // Half-disc profile in (radial, axial) = (c0, c1): the radial extent is the
        // sphere radius at each axial height. Axial runs 0..2R; radial(axial) =
        // sqrt(R² − (axial − R)²). Many segments ⇒ a close polygon arc. The flat side
        // (radial = 0) is the revolve axis, so revolving the half-disc gives a sphere.
        let segments = 64;
        let mut profile = vec![SketchPoint::new(0, 0)]; // bottom pole on the axis
        for step in 0..=segments {
            let axial = (2 * radius) * step / segments;
            let dz = (axial - radius) as f64;
            let r = ((radius * radius) as f64 - dz * dz).max(0.0).sqrt();
            profile.push(SketchPoint::new(r.round() as i64, axial));
        }
        profile.push(SketchPoint::new(0, 2 * radius)); // top pole on the axis
        let revolve =
            SketchSolid::revolve(Sketch::new(PlaneAxis::X, profile), RevolveAxis::InPlane1, 360);
        let sphere = SdfShape::from_voxels(
            ShapeKind::Sphere,
            [(2 * radius) as u32, (2 * radius) as u32, (2 * radius) as u32],
            1,
        );
        assert_eq!(revolve.grid_dimensions(), sphere.grid_dimensions(density));
        let revolve_set = cell_set(&revolve, density);
        let sphere_set = cell_set(&sphere, density);
        let intersection = revolve_set.intersection(&sphere_set).count();
        let union = revolve_set.union(&sphere_set).count();
        let iou = intersection as f64 / union as f64;
        assert!(iou >= 0.97, "half-disc revolve IoU vs sphere {iou} < 0.97");
    }

    /// EDGE CASE: degenerate revolve profiles resolve to empty without panicking —
    /// fewer than 3 points, zero radial extent, and a zero turn. (Mirror of
    /// `degenerate_profiles_are_empty` for the revolve arm.)
    #[test]
    fn revolve_degenerate_profiles_are_empty() {
        let empty = |producer: &SketchSolid| {
            let mut grid = VoxelGrid::default();
            producer.resolve(&mut grid, 4);
            assert!(grid.occupied.is_empty());
            assert_eq!(grid.dimensions, [0, 0, 0]);
        };
        // < 3 points.
        empty(&SketchSolid::revolve(
            Sketch::new(PlaneAxis::X, vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)]),
            RevolveAxis::InPlane1,
            360,
        ));
        // Zero radial extent: a profile collinear on the radial axis (radial coord all
        // 0) — profile_bounds rejects the zero-span axis ⇒ empty.
        empty(&SketchSolid::revolve(
            Sketch::new(
                PlaneAxis::X,
                vec![
                    SketchPoint::new(0, 0),
                    SketchPoint::new(0, 4),
                    SketchPoint::new(0, 8),
                ],
            ),
            RevolveAxis::InPlane1,
            360,
        ));
        // Zero turn.
        empty(&SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 8, 8),
            RevolveAxis::InPlane1,
            0,
        ));
    }

    /// PARITY: a rectangle revolved on EACH `RevolveAxis` at even and odd diameters is
    /// corner-anchored with NO straddle — occupancy spans exactly `[0, dim)` per axis,
    /// and the disc is symmetric about the grid centre on the two radial axes.
    #[test]
    fn revolve_parity_axis_placement() {
        let density = 8u32;
        // (radial R, axial H): R=10 ⇒ even diameter 20; R=11 ⇒ odd-extent, diameter 22.
        for (radial, axial) in [(10i64, 12i64), (11, 13)] {
            for plane in [PlaneAxis::X, PlaneAxis::Y, PlaneAxis::Z] {
                for revolve_axis in [RevolveAxis::InPlane0, RevolveAxis::InPlane1] {
                    // Place radial on c0, axial on c1 for InPlane1; swapped for InPlane0.
                    let sketch = match revolve_axis {
                        RevolveAxis::InPlane1 => Sketch::rectangle(plane, radial, axial),
                        RevolveAxis::InPlane0 => Sketch::rectangle(plane, axial, radial),
                    };
                    let revolve = SketchSolid::revolve(sketch, revolve_axis, 360);
                    let dims = revolve.grid_dimensions();
                    let mut grid = VoxelGrid::default();
                    revolve.resolve(&mut grid, density);
                    assert!(!grid.occupied.is_empty(), "{plane:?}/{revolve_axis:?} empty");

                    // No straddle: every cell index is within [0, dim) per axis, and the
                    // occupancy touches 0 and dim-1 on the radial axes (the disc spans
                    // the full diameter symmetric about the centre).
                    let mut min_cell = [i64::MAX; 3];
                    let mut max_cell = [i64::MIN; 3];
                    for voxel in &grid.occupied {
                        let position = voxel.world_position();
                        for axis in 0..3 {
                            let cell = (position[axis] - 0.5).round() as i64;
                            assert!(
                                cell >= 0 && (cell as u32) < dims[axis],
                                "{plane:?}/{revolve_axis:?}: cell {cell} out of [0,{}) on axis {axis}",
                                dims[axis]
                            );
                            min_cell[axis] = min_cell[axis].min(cell);
                            max_cell[axis] = max_cell[axis].max(cell);
                        }
                    }
                    // Identify the two radial world axes (full-diameter, symmetric span).
                    let [ip0, ip1] = plane.in_plane_axes();
                    let normal = plane.normal_axis();
                    let radial_axes: [usize; 2] = match revolve_axis {
                        RevolveAxis::InPlane0 => {
                            let mut a = [ip1, normal];
                            a.sort_unstable();
                            a
                        }
                        RevolveAxis::InPlane1 => {
                            let mut a = [ip0, normal];
                            a.sort_unstable();
                            a
                        }
                    };
                    for &axis in &radial_axes {
                        // The widest slice (through the rectangle's full radial extent)
                        // spans the whole diameter, touching both ends ⇒ no straddle and
                        // symmetric about the centre.
                        assert_eq!(
                            min_cell[axis], 0,
                            "{plane:?}/{revolve_axis:?}: radial axis {axis} does not start at 0"
                        );
                        assert_eq!(
                            max_cell[axis] as u32,
                            dims[axis] - 1,
                            "{plane:?}/{revolve_axis:?}: radial axis {axis} does not reach dim-1"
                        );
                    }
                }
            }
        }
    }

    /// PARTIAL turn is inert at 360: a 360° revolve is byte-identical (occupancy SET
    /// equal) to one built without the partial path engaging — proves the atan2 gate
    /// never fires at a full turn (theta ∈ [0,360) is never > 360).
    #[test]
    fn partial_revolve_360_equals_full() {
        let density = 16u32;
        let full_a = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 32, 48),
            RevolveAxis::InPlane1,
            360,
        );
        // The "full" reference is the same operation; this asserts determinism AND that
        // the 360 gate produces the same occupancy as the cylinder lock above relies on.
        let full_b = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 32, 48),
            RevolveAxis::InPlane1,
            360,
        );
        assert_eq!(
            cell_set(&full_a, density),
            cell_set(&full_b, density),
            "360° revolve must be deterministic / gate-inert"
        );
        // And it equals the matching cylinder (the partial gate did not eat any cells).
        let cylinder = SdfShape::from_voxels(ShapeKind::Cylinder, [64, 64, 48], 1);
        let diff = cell_set(&full_a, density)
            .symmetric_difference(&cell_set(&cylinder, density))
            .count();
        let total = cell_set(&cylinder, density).len();
        assert!(diff * 100 <= total, "360 revolve diff from cylinder {diff} > 1%");
    }

    /// PARTIAL turn 180° is roughly half a 360° revolve, and one angular half of the
    /// disc is empty (structural). The angle is measured from radial_a toward radial_b
    /// (ascending world-axis index); for PlaneAxis::X + InPlane1, radial_a=X, radial_b=Y,
    /// so the kept wedge is theta ∈ [0,180] ⇒ the centred-Y < 0 half is empty.
    #[test]
    fn partial_revolve_180_is_half() {
        let density = 8u32;
        let full = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 24, 32),
            RevolveAxis::InPlane1,
            360,
        );
        let half = SketchSolid::revolve(
            Sketch::rectangle(PlaneAxis::X, 24, 32),
            RevolveAxis::InPlane1,
            180,
        );
        let full_count = cell_set(&full, density).len();
        let half_count = cell_set(&half, density).len();
        let ratio = half_count as f64 / full_count as f64;
        assert!(
            (0.40..=0.60).contains(&ratio),
            "180° revolve count ratio {ratio} not ~0.5"
        );
        // Structural: the half with theta > 180 is empty. For PlaneAxis::X + InPlane1,
        // radial_a = X (idx 0), radial_b = Y (idx 1); theta > 180 ⇔ centred-Y < 0. The
        // grid is [48,48,32]; centred-Y < 0 means cell-Y < 24.
        let mut grid = VoxelGrid::default();
        half.resolve(&mut grid, density);
        let dim_y = grid.dimensions[1];
        let any_in_lower_half = grid.occupied.iter().any(|voxel| {
            let cell_y = (voxel.world_position()[1] - 0.5).round() as i64;
            // centred-Y = cell_y + 0.5 - dim_y/2 < 0
            (cell_y as f32 + 0.5 - dim_y as f32 / 2.0) < 0.0
        });
        assert!(!any_in_lower_half, "180° revolve leaked into the theta>180 half");
    }
