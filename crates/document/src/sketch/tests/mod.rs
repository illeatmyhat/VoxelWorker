#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::explicit_iter_loop,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::manual_midpoint,
    clippy::panic,
    clippy::redundant_clone,
    clippy::unwrap_used
)]

use super::*;
use crate::voxel::VoxelProducer;
use std::collections::BTreeSet;
use voxel_core::voxel::VoxelGrid;

mod arcs;
mod circles;
mod coarse_solid;
mod constraints;
mod drawing;
mod edges;
mod edits;
mod extrude;
mod field;
mod parametric;
mod region_memo;
mod regions;
mod revolve;

/// Collect a producer's occupied voxels as a sorted set of
/// `(world_position_bits, block_local_coord, material_id)` so two producers can
/// be compared for SET equality independent of emission order. World positions
/// are integer + 0.5, so the f32 bit pattern is exact and hashable.
fn occupancy_set(producer: &dyn VoxelProducer, density: u32) -> BTreeSet<([i32; 3], [u8; 3], u16)> {
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
