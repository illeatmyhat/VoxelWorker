//! ADR 0027 affine oracle tests. The unified affine has NO golden of its own for the GATHER
//! path (a genuine rotation resamples where nothing did before), so these build oracles: the
//! gather must reproduce the exact permutation for an axis-aligned turn, an isotropic sphere
//! must survive rotation, and the affine's inverse must round-trip.

use crate::cuboid::VoxelRegion;
use crate::two_layer_store::classify::{
    compose_leaf_into_region, gather_rotated_leaf_into_region, leaf_affine, leaf_world_box,
};
use document::scene::{LeafProducer, Scene};
use document::voxel::GeometryParams;
use glam::{Quat, Vec3};
use std::collections::BTreeSet;
use std::f32::consts::FRAC_PI_2;
use substrate::spatial::ProducerLocalVoxelPoint;
use voxel_core::core_geom::MaterialChoice;
use voxel_core::voxel::ShapeKind;

const DENSITY: u32 = 8;

/// The single geometry leaf of a one-shape scene, sized `size_voxels` per axis. Placed at
/// the scene's own (small) world offset — modest coordinates keep the `f32` affine exact.
fn single_leaf(kind: ShapeKind, size_voxels: [u32; 3]) -> LeafProducer {
    let scene = Scene::from_geometry(
        GeometryParams {
            shape: kind,
            size_voxels,
            size_measurements: None,
            voxels_per_block: DENSITY,
            wall_blocks: 1,
        },
        MaterialChoice::Stone,
    );
    scene
        .leaf_producers(DENSITY)
        .into_iter()
        .next()
        .expect("the single-geometry scene has one leaf")
}

/// **Regression (ADR 0027): the builder broadphase box must equal the classifier's rotated
/// box.** `builder::leaf_world_aabb` once computed its extent from the discrete lattice
/// `orientation.turn_extent`, blind to the continuous quaternion — so a leaf with an IDENTITY
/// lattice orientation but a non-identity `rotation` (a tube seated on a curved surface)
/// reserved an UPRIGHT box. The edit broadphase then dropped that leaf from every chunk its
/// tilted body occupied beyond the upright box, classifying them to air and TRUNCATING the
/// tube (the "tubes render upright" bug). The two extents share ONE definition now; this pins
/// that they never diverge under a genuine (lattice-identity) rotation, and that the box
/// actually reflects the tilt rather than staying the axis-aligned lattice box.
#[test]
fn the_broadphase_box_equals_the_rotated_classifier_box() {
    let mut leaf = single_leaf(ShapeKind::Cylinder, [16, 16, 48]);
    // Identity lattice orientation (as `leaf_producers` builds it) but a genuine off-axis
    // tilt — exactly the seated-on-a-curve case the bug truncated.
    leaf.rotation = glam::Quat::from_rotation_x(0.7);
    let broadphase = super::super::builder::leaf_world_aabb(&leaf, DENSITY);
    let classifier = leaf_world_box(&leaf, DENSITY);
    assert_eq!(
        broadphase, classifier,
        "the broadphase AABB must enclose the SAME rotated box the classifier folds through"
    );
    // And it must NOT be the upright axis-aligned box the old lattice path returned: a 48-tall
    // cylinder tilted about X spreads in Y and shrinks in Z, so its extent differs from the
    // untilted [16,16,48].
    let extent = [
        classifier.max[0] - classifier.min[0],
        classifier.max[1] - classifier.min[1],
        classifier.max[2] - classifier.min[2],
    ];
    assert_ne!(extent, [16, 16, 48], "a genuine rotation must change the extent from the upright box");
}

/// The occupied block-local cells of a resolved region.
fn occupied_cells(region: &VoxelRegion) -> BTreeSet<[u32; 3]> {
    let [width, height, depth] = region.extent;
    let mut cells = BTreeSet::new();
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                if region.cell_at(x, y, z).is_some() {
                    cells.insert([x, y, z]);
                }
            }
        }
    }
    cells
}

/// A block window (low corner + cubic edge) that generously encloses a leaf's world box.
fn enclosing_window(leaf: &LeafProducer) -> ([i64; 3], u32) {
    let world_box = leaf_world_box(leaf, DENSITY);
    let pad = 3i64;
    let edge = (0..3)
        .map(|axis| world_box.max[axis] - world_box.min[axis])
        .max()
        .unwrap()
        + 2 * pad;
    let min = std::array::from_fn(|axis| world_box.min[axis] - pad);
    (min, edge as u32)
}

/// **Gather-vs-permutation oracle on a cube.** A cube (equal dims) turned 90° about Z is an
/// AXIS-ALIGNED rotation, so the forward-emit permutation and the inverse-resample gather
/// must produce the IDENTICAL occupied set — the proof that the gather reproduces the exact
/// turn the classifier has always emitted.
#[test]
fn gather_reproduces_the_axis_aligned_permutation_on_a_cube() {
    let mut leaf = single_leaf(ShapeKind::Box, [DENSITY, DENSITY, DENSITY]);
    leaf.rotation = Quat::from_rotation_z(FRAC_PI_2);
    assert!(
        substrate::spatial::is_axis_aligned(leaf.rotation),
        "a 90° turn is one of the 24 lattice rotations"
    );

    let (block_min, edge) = enclosing_window(&leaf);

    // Forward-emit (the permutation path: `compose_leaf_into_region` routes an axis-aligned
    // leaf through `producer_local_voxel_to_abs`).
    let mut forward = VoxelRegion::new_empty([edge; 3]);
    compose_leaf_into_region(&mut forward, &leaf, block_min, edge, DENSITY);

    // The gather path, forced directly.
    let mut gathered = VoxelRegion::new_empty([edge; 3]);
    gather_rotated_leaf_into_region(&mut gathered, &leaf, block_min, edge, DENSITY);

    let forward_cells = occupied_cells(&forward);
    let gathered_cells = occupied_cells(&gathered);
    assert!(!forward_cells.is_empty(), "the cube must occupy cells");
    assert_eq!(
        gathered_cells, forward_cells,
        "the gather must reproduce the exact turned cells the permutation emits"
    );
}

/// **Rotated-sphere sanity.** An isotropic sphere rotated by an arbitrary angle occupies
/// essentially the same cells as the unrotated one at the same centre: the occupied count
/// matches within a few boundary voxels, and the gathered set is centro-symmetric.
#[test]
fn a_rotated_sphere_stays_the_same_sphere() {
    let size = [2 * DENSITY, 2 * DENSITY, 2 * DENSITY];
    let upright = single_leaf(ShapeKind::Sphere, size);
    let (block_min, edge) = enclosing_window(&upright);

    // The upright occupancy through the forward-emit path.
    let mut upright_region = VoxelRegion::new_empty([edge; 3]);
    compose_leaf_into_region(&mut upright_region, &upright, block_min, edge, DENSITY);
    let upright_cells = occupied_cells(&upright_region);
    assert!(!upright_cells.is_empty(), "the sphere must occupy cells");

    // The same sphere rotated by 0.6 rad about Z, resolved by the gather.
    let mut rotated = single_leaf(ShapeKind::Sphere, size);
    rotated.rotation = Quat::from_rotation_z(0.6);
    assert!(!substrate::spatial::is_axis_aligned(rotated.rotation), "0.6 rad is genuinely off-axis");
    let mut rotated_region = VoxelRegion::new_empty([edge; 3]);
    gather_rotated_leaf_into_region(&mut rotated_region, &rotated, block_min, edge, DENSITY);
    let rotated_cells = occupied_cells(&rotated_region);

    // Count matches within a thin voxelization boundary (a sphere is rotation-invariant).
    let upright_count = upright_cells.len() as i64;
    let rotated_count = rotated_cells.len() as i64;
    let tolerance = (upright_count / 20).max(8); // ~5%, floor of a few voxels
    assert!(
        (upright_count - rotated_count).abs() <= tolerance,
        "rotated sphere occupancy {rotated_count} must match upright {upright_count} within {tolerance}"
    );

    // Centro-symmetry: reflect every cell through `2·centroid` (integer for a symmetric
    // set) and count how many mirrors land back in the set. A rotated sphere's centre
    // generally does NOT sit on a half-integer of the ABSOLUTE lattice, so its voxelization
    // is only APPROXIMATELY cell-reflection-symmetric — a thin boundary rim breaks exact
    // reflection. Require the overwhelming majority (a grossly-wrong gather — sheared or
    // truncated — would fail badly), tolerating that rim.
    let sum = rotated_cells.iter().fold([0i64; 3], |acc, cell| {
        std::array::from_fn(|axis| acc[axis] + cell[axis] as i64)
    });
    let twice_centroid: [i64; 3] = std::array::from_fn(|axis| {
        let doubled = 2 * sum[axis];
        (doubled as f64 / rotated_count as f64).round() as i64
    });
    let mirrored = rotated_cells
        .iter()
        .filter(|cell| {
            let reflected: [u32; 3] =
                std::array::from_fn(|axis| (twice_centroid[axis] - cell[axis] as i64) as u32);
            rotated_cells.contains(&reflected)
        })
        .count();
    let mirror_fraction = mirrored as f64 / rotated_count as f64;
    assert!(
        mirror_fraction >= 0.9,
        "rotated sphere must be near-centro-symmetric: only {mirror_fraction:.3} of \
         {rotated_count} cells have their mirror"
    );
}

/// **The affine's inverse round-trips** for a genuinely non-axis-aligned rotation:
/// `local_of(world_of(p)) ≈ p` for arbitrary local points.
#[test]
fn local_of_inverts_world_of_for_a_tilted_rotation() {
    let mut leaf = single_leaf(ShapeKind::Box, [DENSITY, 2 * DENSITY, 3 * DENSITY]);
    leaf.rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);
    let affine = leaf_affine(&leaf, DENSITY);
    for point in [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.5, 9.0, 21.5),
        Vec3::new(7.0, 15.0, 23.0),
        Vec3::new(-2.0, 4.0, 12.0),
    ] {
        let round_tripped = affine
            .local_of(affine.world_of(ProducerLocalVoxelPoint::from_voxels(point)))
            .voxels();
        assert!(
            (round_tripped - point).length() < 1e-3,
            "local_of(world_of({point:?})) = {round_tripped:?} must round-trip"
        );
    }
}
