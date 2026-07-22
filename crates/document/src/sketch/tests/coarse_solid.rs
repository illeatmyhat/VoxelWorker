use super::*;
use crate::sketch::RevolveAxis;
use crate::voxel::VoxelProducer;
use voxel_core::voxel::VoxelGrid;
use std::collections::BTreeSet;

/// The coarse-solid cell classifier must agree with the per-voxel resolve.
///
/// This is the half of `geom2d` deliberately LEFT at f64 (`orient2d` ->
/// `segments_intersect` -> `segment_intersects_rect` -> `rectangle_inside_polygon` ->
/// `*_cell_is_solid`), so it is the half that must be shown UNMOVED by the narrowing of
/// the other half. The contract is asymmetric and one-directional: a cell claimed SOLID
/// must contain only occupied voxels, because an over-claim fills a cell nothing ever
/// sampled - unsound, not conservative. A cell NOT claimed solid may be anything; that
/// is merely conservative, and stays exact via the per-voxel path.
#[test]
fn coarse_solid_cells_never_over_claim_against_a_per_voxel_sweep() {
    use voxel_core::spatial_index::VoxelAabb;

    let lathe = vec![
        SketchPoint::new(0, 2), SketchPoint::new(9, 2),
        SketchPoint::new(9, 7), SketchPoint::new(4, 5),
        SketchPoint::new(0, 7),
    ];
    let cases: Vec<(&str, SketchSolid)> = vec![
        (
            "extrude",
            SketchSolid::extrude(Sketch::new(PlaneAxis::Z, lathe.clone()), 6),
        ),
        (
            "revolve full",
            SketchSolid::revolve(
                Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane0, 360),
        ),
        (
            "revolve 135",
            SketchSolid::revolve(
                Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane0, 135),
        ),
        (
            "revolve 200",
            SketchSolid::revolve(
                Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane1, 200),
        ),
    ];

    for (label, solid) in cases {
        let dimensions = solid.grid_dimensions();
        // The per-voxel truth: resolve the whole producer and index the occupied set.
        let mut grid = VoxelGrid::default();
        solid.resolve(&mut grid, 1);
        let occupied: BTreeSet<[i32; 3]> =
            grid.occupied.iter().map(|voxel| voxel.local_index).collect();

        // Sweep 2x2x2 cells across the extent and check every SOLID claim voxel by voxel.
        const CELL: i64 = 2;
        let mut solid_claims = 0;
        for z in (0..dimensions[2] as i64).step_by(CELL as usize) {
            for y in (0..dimensions[1] as i64).step_by(CELL as usize) {
                for x in (0..dimensions[0] as i64).step_by(CELL as usize) {
                    let hi = [
                        (x + CELL).min(dimensions[0] as i64),
                        (y + CELL).min(dimensions[1] as i64),
                        (z + CELL).min(dimensions[2] as i64),
                    ];
                    let cell = VoxelAabb::new([x, y, z], hi);
                    let claimed = match solid.operation {
                        Operation::Extrude { .. } => solid.extrude_cell_is_solid(cell),
                        Operation::Revolve { axis, sweep } => {
                            solid.revolve_cell_is_solid(cell, axis, sweep, dimensions)
                        }
                    };
                    if !claimed {
                        continue;
                    }
                    solid_claims += 1;
                    for vz in z..hi[2] {
                        for vy in y..hi[1] {
                            for vx in x..hi[0] {
                                let index = [vx as i32, vy as i32, vz as i32];
                                assert!(
                                    occupied.contains(&index),
                                    "{label}: cell {cell:?} was claimed coarse-SOLID but \
                                     voxel {index:?} is NOT occupied by the per-voxel \
                                     resolve - the classifier over-claims, which is \
                                     unsound (it fills voxels nothing ever sampled)."
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            solid_claims > 0,
            "{label}: no cell was claimed solid at all, so the check proved nothing"
        );
    }
}
