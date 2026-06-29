//! Commit-1 windowed-resolve contract: every [`VoxelProducer`] gains a
//! `resolve_into(window)` that writes ONLY the cells whose local voxel index lies
//! inside `window`, while still reporting the producer's FULL `dimensions`. The
//! contract proven here, for EACH producer (SdfShape, SketchSolid-Extrude,
//! SketchSolid-Revolve, DebugCloudField), at EVEN / ODD / MIXED parity sizes:
//!
//! * a windowed resolve is a byte-identical SUBSET of the full resolve —
//!   `resolve_into(window).occupied == { v in full : v.cell ∈ window }`;
//! * `grid.dimensions` ALWAYS equals the full producer dimensions, even when the
//!   window is empty or out of range;
//! * the FULL window (`[0,0,0]..full_dim`) reproduces `resolve` EXACTLY;
//! * a set of DISJOINT windows that TILE `[0, full_dim)` reproduces the full
//!   occupied set as their union (no cell dropped or double-counted).
//!
//! These are the regression net for Commit 2, which switches the call site onto
//! `resolve_into`. Commit 1 changes NO behaviour, so they pass against today's tree.

use crate::spatial_index::VoxelAabb;
use crate::voxel::{Voxel, VoxelGrid, VoxelProducer};
use std::collections::BTreeSet;

/// An occupied voxel keyed for SET comparison independent of emission order. World
/// positions are integer + 0.5, so doubling and rounding gives an EXACT integer key
/// (no float-equality hazard). Includes block-local coord and material so the key
/// captures the full per-cell output, not just position.
type OccupancyKey = ([i64; 3], [u8; 3], u16);

fn key_of(voxel: &Voxel) -> OccupancyKey {
    (
        [
            (voxel.world_position[0] * 2.0).round() as i64,
            (voxel.world_position[1] * 2.0).round() as i64,
            (voxel.world_position[2] * 2.0).round() as i64,
        ],
        voxel.block_local_coord,
        voxel.material_id,
    )
}

/// The integer cell index `floor(world_position)` of an occupied voxel — the corner
/// the centre `idx + 0.5` sits in. Used to decide window membership.
fn cell_index_of(voxel: &Voxel) -> [i64; 3] {
    [
        voxel.world_position[0].floor() as i64,
        voxel.world_position[1].floor() as i64,
        voxel.world_position[2].floor() as i64,
    ]
}

fn cell_in_window(cell: [i64; 3], window: VoxelAabb) -> bool {
    (0..3).all(|axis| window.min[axis] <= cell[axis] && cell[axis] < window.max[axis])
}

fn full_resolve(producer: &dyn VoxelProducer, density: u32) -> VoxelGrid {
    let mut grid = VoxelGrid::default();
    producer.resolve(&mut grid, density);
    grid
}

fn windowed_resolve(producer: &dyn VoxelProducer, density: u32, window: VoxelAabb) -> VoxelGrid {
    let mut grid = VoxelGrid::default();
    producer.resolve_into(&mut grid, density, window);
    grid
}

/// Run the full windowed-subset battery for one producer at one density. `full_dim`
/// is the producer's full dimensions (used to build the windows and assert
/// `grid.dimensions`). `label` distinguishes producers in failure messages.
fn assert_windowed_subset_contract(
    producer: &dyn VoxelProducer,
    density: u32,
    full_dim: [u32; 3],
    label: &str,
) {
    let [fx, fy, fz] = [full_dim[0] as i64, full_dim[1] as i64, full_dim[2] as i64];

    // Reference: the full resolve, its dimensions, and its keyed occupied set.
    let full_grid = full_resolve(producer, density);
    assert_eq!(
        full_grid.dimensions, full_dim,
        "{label}: full resolve must report the full dimensions"
    );
    let full_set: BTreeSet<OccupancyKey> = full_grid.occupied.iter().map(key_of).collect();
    assert_eq!(
        full_set.len(),
        full_grid.occupied.len(),
        "{label}: full resolve emitted a duplicate voxel (key collision)"
    );

    // Per-axis split points for interior / face-clipping / centre-straddling windows.
    let mid = [fx / 2, fy / 2, fz / 2];

    // (a) interior fully inside; (b) clip LOW face on X; (c) clip HIGH face on Z;
    // (d) straddle the centre; (e) fully OUTSIDE [0,full_dim); plus per-axis
    // half-splits used both as standalone windows and as the tiling halves.
    let windows: Vec<(&str, VoxelAabb)> = vec![
        (
            "interior",
            VoxelAabb::new(
                [fx / 4, fy / 4, fz / 4],
                [(3 * fx) / 4, (3 * fy) / 4, (3 * fz) / 4],
            ),
        ),
        (
            "clip-low-x",
            VoxelAabb::new([-5, 0, 0], [fx / 3 + 1, fy, fz]),
        ),
        (
            "clip-high-z",
            VoxelAabb::new([0, 0, (2 * fz) / 3], [fx, fy, fz + 7]),
        ),
        (
            "straddle-centre",
            VoxelAabb::new(
                [mid[0] - 1, mid[1] - 1, mid[2] - 1],
                [mid[0] + 2, mid[1] + 2, mid[2] + 2],
            ),
        ),
        (
            "fully-outside-high",
            VoxelAabb::new([fx + 3, fy + 3, fz + 3], [fx + 9, fy + 9, fz + 9]),
        ),
        (
            "fully-outside-low",
            VoxelAabb::new([-20, -20, -20], [-1, -1, -1]),
        ),
    ];

    for (name, window) in &windows {
        let grid = windowed_resolve(producer, density, *window);
        assert_eq!(
            grid.dimensions, full_dim,
            "{label}/{name}: windowed resolve MUST still report FULL dimensions"
        );
        let got: BTreeSet<OccupancyKey> = grid.occupied.iter().map(key_of).collect();
        assert_eq!(
            got.len(),
            grid.occupied.len(),
            "{label}/{name}: windowed resolve emitted a duplicate voxel"
        );
        let expected: BTreeSet<OccupancyKey> = full_grid
            .occupied
            .iter()
            .filter(|voxel| cell_in_window(cell_index_of(voxel), *window))
            .map(key_of)
            .collect();
        assert_eq!(
            got, expected,
            "{label}/{name}: windowed occupied set must equal the in-window subset of the full set"
        );
    }

    // (e, strict) a window fully outside [0,full_dim) on any axis → EMPTY occupied,
    // FULL dims (the membership filter above already proves the set; pin emptiness).
    let outside = windowed_resolve(
        producer,
        density,
        VoxelAabb::new([fx + 3, 0, 0], [fx + 9, fy, fz]),
    );
    assert!(
        outside.occupied.is_empty(),
        "{label}: a window past the high X face must resolve to EMPTY"
    );
    assert_eq!(outside.dimensions, full_dim, "{label}: empty window keeps full dims");

    // (f) the FULL window equals `resolve` EXACTLY (the clamp makes full-window ≡
    // the historical resolve).
    let full_window = windowed_resolve(producer, density, VoxelAabb::new([0, 0, 0], [fx, fy, fz]));
    let full_window_set: BTreeSet<OccupancyKey> = full_window.occupied.iter().map(key_of).collect();
    assert_eq!(
        full_window.dimensions, full_dim,
        "{label}: full-window resolve_into reports full dims"
    );
    assert_eq!(
        full_window_set, full_set,
        "{label}: full-window resolve_into must reproduce `resolve` exactly"
    );

    // An OVERSIZED window (padded past the grid on every axis) also reproduces the
    // full set — clamping is harmless.
    let oversized = windowed_resolve(
        producer,
        density,
        VoxelAabb::new([-10, -10, -10], [fx + 10, fy + 10, fz + 10]),
    );
    let oversized_set: BTreeSet<OccupancyKey> = oversized.occupied.iter().map(key_of).collect();
    assert_eq!(
        oversized_set, full_set,
        "{label}: an oversized window clamps to the grid and reproduces the full set"
    );

    // TILING-UNION: split [0, full_dim) into 8 disjoint octant windows (the half-cuts
    // on each axis tile the axis), resolve each, and assert the union == full set AND
    // the tiles are pairwise DISJOINT (counts sum exactly). Proves no cell is dropped
    // or double-counted at window seams.
    let x_cuts = [(0, mid[0]), (mid[0], fx)];
    let y_cuts = [(0, mid[1]), (mid[1], fy)];
    let z_cuts = [(0, mid[2]), (mid[2], fz)];
    let mut union: BTreeSet<OccupancyKey> = BTreeSet::new();
    let mut summed_count = 0usize;
    for (x_lo, x_hi) in x_cuts {
        for (y_lo, y_hi) in y_cuts {
            for (z_lo, z_hi) in z_cuts {
                let tile = VoxelAabb::new([x_lo, y_lo, z_lo], [x_hi, y_hi, z_hi]);
                let grid = windowed_resolve(producer, density, tile);
                assert_eq!(
                    grid.dimensions, full_dim,
                    "{label}: tile resolve keeps full dims"
                );
                let tile_set: BTreeSet<OccupancyKey> = grid.occupied.iter().map(key_of).collect();
                summed_count += tile_set.len();
                union.extend(tile_set);
            }
        }
    }
    assert_eq!(
        union, full_set,
        "{label}: union of tiling windows must equal the full occupied set"
    );
    assert_eq!(
        summed_count,
        full_set.len(),
        "{label}: tiling windows must be DISJOINT (per-tile counts sum to the full count)"
    );
}

// ---------------------------------------------------------------------------
// SdfShape
// ---------------------------------------------------------------------------

#[test]
fn sdf_shape_windowed_subset_contract() {
    use crate::voxel::{SdfShape, ShapeKind};
    // EVEN, ODD, MIXED parity voxel sizes; spheres/cylinders to exercise the SDF.
    let cases: [(ShapeKind, [u32; 3], u32); 4] = [
        (ShapeKind::Sphere, [16, 16, 16], 8),  // even
        (ShapeKind::Cylinder, [15, 15, 15], 4), // odd
        (ShapeKind::Box, [14, 9, 12], 4),       // mixed
        (ShapeKind::Sphere, [11, 16, 13], 4),   // mixed
    ];
    for (kind, size_voxels, density) in cases {
        let shape = SdfShape::from_voxels(kind, size_voxels, 1);
        let full_dim = shape.grid_dimensions(density);
        assert_windowed_subset_contract(&shape, density, full_dim, "sdf");
    }
}

// ---------------------------------------------------------------------------
// SketchSolid — Extrude
// ---------------------------------------------------------------------------

#[test]
fn sketch_extrude_windowed_subset_contract() {
    use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
    // Vary the plane so the in-plane↔world axis window mapping is exercised on every
    // normal (X / Y / Z); EVEN, ODD, MIXED profile / height parities.
    let cases: [(PlaneAxis, i64, i64, u32); 4] = [
        (PlaneAxis::Z, 12, 12, 10), // even spans + even height
        (PlaneAxis::Y, 11, 9, 7),   // odd spans + odd height
        (PlaneAxis::X, 13, 8, 6),   // mixed
        (PlaneAxis::Z, 9, 14, 11),  // mixed, tall
    ];
    for (plane, width, height, extrude_height) in cases {
        let solid = SketchSolid::extrude(Sketch::rectangle(plane, width, height), extrude_height);
        let full_dim = solid.grid_dimensions();
        assert_windowed_subset_contract(&solid, 16, full_dim, "extrude");
    }
}

// ---------------------------------------------------------------------------
// SketchSolid — Revolve (incl. partial turn + axis-straddling profile)
// ---------------------------------------------------------------------------

#[test]
fn sketch_revolve_windowed_subset_contract() {
    use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};

    // (1) Full 360° revolve of a one-sided rectangle (radial >= 0): exercises the
    //     non-straddling early-out branch. Even disc diameter.
    let full_revolve = SketchSolid::revolve(Sketch::rectangle(PlaneAxis::Z, 8, 6), RevolveAxis::InPlane0, 360);
    let dim = full_revolve.grid_dimensions();
    assert_windowed_subset_contract(&full_revolve, 16, dim, "revolve-360-onesided");

    // (2) PARTIAL 180° turn: exercises the partial-turn theta gate under windowing.
    let partial = SketchSolid::revolve(Sketch::rectangle(PlaneAxis::Z, 9, 7), RevolveAxis::InPlane0, 180);
    let dim = partial.grid_dimensions();
    assert_windowed_subset_contract(&partial, 16, dim, "revolve-180");

    // (3) A profile that STRADDLES the radial axis (a vertex at negative radial):
    //     exercises the `inside(radius) || inside(-radius)` straddle branch. Radial
    //     coord is c1 for InPlane0, so points span c1 ∈ [-3, 4]; axial c0 ∈ [0, 6].
    let straddle_profile = vec![
        SketchPoint::new(0, -3),
        SketchPoint::new(6, -3),
        SketchPoint::new(6, 4),
        SketchPoint::new(0, 4),
    ];
    let straddle = SketchSolid::revolve(
        Sketch::new(PlaneAxis::Z, straddle_profile),
        RevolveAxis::InPlane0,
        360,
    );
    let dim = straddle.grid_dimensions();
    assert_windowed_subset_contract(&straddle, 16, dim, "revolve-straddle");

    // (4) Partial turn on a different plane/axis → mixed parity + InPlane1 mapping.
    let partial_inplane1 =
        SketchSolid::revolve(Sketch::rectangle(PlaneAxis::Y, 11, 5), RevolveAxis::InPlane1, 270);
    let dim = partial_inplane1.grid_dimensions();
    assert_windowed_subset_contract(&partial_inplane1, 16, dim, "revolve-270-inplane1");
}

// ---------------------------------------------------------------------------
// DebugCloudField
// ---------------------------------------------------------------------------

#[test]
fn debug_cloud_field_windowed_subset_contract() {
    use crate::debug_clouds::DebugCloudField;
    // EVEN, ODD, MIXED dimensions; distinct seeds.
    let cases: [([u32; 3], u32); 4] = [
        ([32, 32, 32], 1), // even
        ([31, 31, 31], 2), // odd
        ([30, 17, 24], 3), // mixed
        ([21, 32, 19], 4), // mixed
    ];
    for (dimensions, seed) in cases {
        let field = DebugCloudField { dimensions, seed };
        assert_windowed_subset_contract(&field, 16, dimensions, "cloud");
    }
}
