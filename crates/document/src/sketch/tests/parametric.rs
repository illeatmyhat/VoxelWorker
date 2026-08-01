//! Sub-voxel + parametric coordinates on `SketchPoint`: the floor/fraction split,
//! snapped-path identity (a whole-voxel profile resolves byte-identical to the integer-only
//! producer), a genuinely fractional profile, and the density re-target (a retained
//! `Measurement` re-evaluates losslessly; a plain point rescales its physical position).

use super::ctx;
use crate::sketch::{PlaneAxis, Sketch, SketchPoint, SketchSolid};
use crate::voxel::VoxelProducer;
use ::parametric::units::{ExactRational, Measurement};
use voxel_core::voxel::VoxelGrid;

#[test]
fn from_continuous_splits_floor_and_fraction() {
    let point = SketchPoint::from_continuous(2.75, -1.25);
    assert_eq!(point.offset_voxels, [2, -2], "floor, also below zero");
    assert!(
        (point.offset_local_voxels[0] - 0.75).abs() < 1e-6
            && (point.offset_local_voxels[1] - 0.75).abs() < 1e-6,
        "the fraction is the remainder above the floor, so it stays in [0, 1)"
    );
    assert_eq!(
        point.in_plane(),
        [2.75, -1.25],
        "the split recomposes exactly"
    );
    assert_eq!(
        SketchPoint::from_continuous(f64::NAN, 3.0).offset_local_voxels,
        [0.0, 0.0],
        "a non-finite coordinate sanitizes instead of poisoning equality"
    );
}

#[test]
fn whole_voxel_continuous_point_is_the_snapped_point() {
    assert_eq!(
        SketchPoint::from_continuous(3.0, 4.0),
        SketchPoint::new(3, 4),
        "zero fraction, no retained expression — the same value, so the snapped scene
         resolves through the identical producer"
    );
}

#[test]
fn snapped_profile_resolves_identically_through_the_continuous_path() {
    // The same rectangle authored with integer constructors and with `from_continuous`
    // whole coords must produce the same occupancy — the sub-voxel fields are inert at zero.
    let integer = SketchSolid::extrude(
        Sketch::new(
            PlaneAxis::Z,
            vec![
                SketchPoint::new(1, 1),
                SketchPoint::new(5, 1),
                SketchPoint::new(5, 4),
                SketchPoint::new(1, 4),
            ],
        ),
        2,
    );
    let continuous = SketchSolid::extrude(
        Sketch::new(
            PlaneAxis::Z,
            vec![
                SketchPoint::from_continuous(1.0, 1.0),
                SketchPoint::from_continuous(5.0, 1.0),
                SketchPoint::from_continuous(5.0, 4.0),
                SketchPoint::from_continuous(1.0, 4.0),
            ],
        ),
        2,
    );
    assert_eq!(
        super::occupancy_set(&integer, 8),
        super::occupancy_set(&continuous, 8),
        "byte-identical occupancy"
    );
}

#[test]
fn fractional_profile_resolves_off_the_voxel_grid() {
    // A rectangle 0.4..4.4 × 0.4..3.4 extruded 2. Cell centers land at `min + cell + 0.5`
    // with the floored min 0: in-plane centers 0.5, 1.5, 2.5, 3.5 (4.5 is past 4.4) on
    // axis 0 and 0.5, 1.5, 2.5 on axis 1 — 4×3 cells per layer, NOT the 4×3 the snapped
    // 0..4 × 0..3 box would give at a different position, and NOT 5×4 (the ceiled grid).
    let solid = SketchSolid::extrude(
        Sketch::new(
            PlaneAxis::Z,
            vec![
                SketchPoint::from_continuous(0.4, 0.4),
                SketchPoint::from_continuous(4.4, 0.4),
                SketchPoint::from_continuous(4.4, 3.4),
                SketchPoint::from_continuous(0.4, 3.4),
            ],
        ),
        2,
    );
    assert_eq!(
        solid.grid_dimensions(ctx(16)),
        [5, 4, 2],
        "the grid box is the floor/ceil cover of the fractional profile"
    );
    let mut grid = VoxelGrid::default();
    solid.resolve(&mut grid, 8);
    assert_eq!(
        grid.occupied.len(),
        4 * 3 * 2,
        "occupancy follows the fractional polygon, not the grid box"
    );
}

#[test]
fn coincidence_is_position_only() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let mut measured = SketchPoint::new(2, 3);
    measured.offset_measurements = Some([Measurement::from_voxels(2), Measurement::from_voxels(3)]);
    let id = sketch.add_free_point(measured);
    assert_eq!(
        sketch.point_at(SketchPoint::new(2, 3)),
        Some(id),
        "a retained expression never splits two coincident points into twins"
    );
    assert_eq!(
        sketch.point_at(SketchPoint::from_continuous(2.5, 3.0)),
        None
    );
}

#[test]
fn retarget_reevaluates_a_retained_measurement() {
    // Half a block: 8 voxels at d16, 16 at d32 — the block term scales, losslessly.
    let half_block = Measurement::new(
        ExactRational::new(1, 2).expect("1/2 is a valid rational"),
        0,
    );
    let mut point = SketchPoint::new(8, 3);
    point.offset_measurements = Some([half_block, Measurement::from_voxels(3)]);
    let at_32 = point.retargeted(16, 32);
    assert_eq!(
        at_32.offset_voxels,
        [16, 3],
        "the block term re-evaluates (8 → 16); the pure voxel term stays exact (3 → 3)"
    );
    assert_eq!(
        at_32.offset_measurements, point.offset_measurements,
        "both axes landed on whole voxels, so the authored expressions are kept verbatim"
    );
    // Non-dividing re-target: 1/2 block at d15 = 7.5 voxels → floors to 7 and the
    // retained expression resynthesizes so it can never disagree with the voxels.
    let at_15 = point.retargeted(16, 15);
    assert_eq!(at_15.offset_voxels, [7, 3]);
    assert_eq!(
        at_15.offset_measurements,
        Some([Measurement::from_voxels(7), Measurement::from_voxels(3)]),
        "the offending axis floors and resynthesizes; the landing axis keeps its form"
    );
}

#[test]
fn retarget_rescales_a_plain_point_physically() {
    // No retained expression: the CONTINUOUS position scales by new/old so the point
    // keeps its physical place — fraction included.
    let point = SketchPoint::from_continuous(2.5, 3.0);
    let doubled = point.retargeted(8, 16);
    assert_eq!(doubled.in_plane(), [5.0, 6.0]);
    assert_eq!(doubled.offset_measurements, None, "still non-parametric");

    let mut sketch = Sketch::new(
        PlaneAxis::Z,
        vec![SketchPoint::new(1, 1), SketchPoint::new(3, 2)],
    );
    sketch.retarget_density(8, 16);
    let coords: Vec<[i64; 2]> = sketch.points().iter().map(|p| p.at.offset_voxels).collect();
    assert_eq!(coords, [[2, 2], [6, 4]], "every store point re-targets");
}
