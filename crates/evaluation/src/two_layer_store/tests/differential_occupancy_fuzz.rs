//! Differential occupancy fuzz — the automatic safety net for the ADR 0027 sub-voxel /
//! off-block / rotated placement bug **class**.
//!
//! ADR 0027 continuous placement drops a leaf at a fractional
//! [`offset_local_voxels`](document::scene::LeafProducer::offset_local_voxels) (e.g.
//! `[0.5, 0.5, 0.0]`) and at an off-block integer
//! [`world_offset_voxels`](document::scene::LeafProducer::world_offset_voxels) (e.g. `-33`
//! at density 16 — not a whole multiple of the block). Older code assumed block / integer
//! alignment, so the SAME leaf's occupancy computed by two code paths could silently drift
//! apart at exactly those placements — the whole class of bugs a full session was spent
//! fixing (`memory/off-block-placement-frame-bugs`, fix 7d3d99e). A single test that
//! asserts the paths agree over a deterministic spread of placements catches that class,
//! once, and every future regression of it.
//!
//! ## Two differentials
//!
//! 1. **Two-layer self-consistency** ([`classify_agrees_with_forced_per_voxel_resolve`]).
//!    The two-layer store's interval **elision** ([`classify_chunk_block`] → AIR /
//!    COARSE-SOLID / BOUNDARY) must produce the identical occupied-cell set as the same
//!    store's **forced** per-voxel resolve ([`resolve_boundary_block`] on *every* block,
//!    trusting no elision) over the same window. Both routes read the ADR 0027 affine
//!    (`leaf_affine`), so both are rotation- and fraction-aware — they must agree for
//!    integer, off-block, fractional AND rotated placements alike. A disagreement means the
//!    conservative classifier called a block AIR/COARSE where the per-voxel truth differs:
//!    exactly the "occupancy drifts at a sub-voxel/off-block/rotated seat" bug. This is the
//!    GREEN net that would have caught the classifier-vs-resolve half of the session's bugs.
//!
//! 2. **Two-layer vs the dense oracle**. Where the two SHOULD agree — integer offsets
//!    (block-aligned AND off-block), identity rotation, zero local slide — a normal green
//!    test ([`two_layer_matches_dense_for_integer_placements`]) pins it. Where the dense
//!    oracle is KNOWN-BLIND — it drops rotation and `offset_local_voxels` entirely (the
//!    deferred "Step 2": `Scene::resolve_region` reads `_rotation` / `_offset_local_voxels`
//!    and ignores them) — the comparison is parked in an `#[ignore]`d test
//!    ([`two_layer_matches_dense_for_rotated_and_fractional_placements`]). That ignored test
//!    is a documented **red-in-waiting**: it fails today because the dense reference has not
//!    yet learned the ADR 0027 affine, and it turns green (delete the `#[ignore]`) the moment
//!    Step 2 teaches the dense oracle to apply rotation + the fractional local offset.
//!
//! Everything here compares occupancy at the **CPU data level** (sets of occupied absolute
//! voxel cells), so no GPU is involved.

use super::*;
use glam::Quat;
use std::collections::BTreeSet;

/// The single material every fuzz leaf stamps — occupancy is all these tests read, so the
/// choice is immaterial beyond being a real id.
const FUZZ_MATERIAL: MaterialChoice = MaterialChoice::Stone;

/// Build ONE placed leaf: an `SdfShape` Tool of `size_blocks³` blocks, then override its
/// carried placement with the exact `world_offset_voxels` / `offset_local_voxels` / `rotation`
/// under test. Mutating the produced [`LeafProducer`] directly (rather than routing through a
/// node transform) is how the affine-oracle tests author a genuine rotation, and it exercises
/// precisely the fields the classifier and the boundary resolve read.
fn placed_single_leaf(
    kind: ShapeKind,
    size_blocks: u32,
    density: u32,
    world_offset_voxels: [i64; 3],
    offset_local_voxels: [f32; 3],
    rotation: Quat,
) -> LeafProducer {
    let shape = SdfShape::from_blocks(kind, [size_blocks; 3], 1, density);
    let node = Node::new(format!("{kind:?}"), NodeContent::Tool { shape, material: FUZZ_MATERIAL });
    let scene = Scene::from_nodes(vec![node]);
    let mut leaf = scene
        .leaf_producers(density)
        .into_iter()
        .next()
        .expect("a single-tool scene yields exactly one leaf");
    leaf.world_offset_voxels = world_offset_voxels;
    leaf.offset_local_voxels = offset_local_voxels;
    leaf.rotation = rotation;
    leaf
}

/// Every block (a `density³` cube on the absolute block lattice) that can touch any leaf,
/// with a one-block pad on each side so the whole rotated / shifted surface is covered. The
/// blocks are block-lattice-aligned (multiples of `density`, `div_euclid` for the negative
/// side), so tiling them is disjoint and gap-free — the union of their per-block occupancy is
/// the leaf's full occupancy exactly once.
fn covering_block_set(leaves: &[&LeafProducer], density: u32) -> BTreeSet<[i64; 3]> {
    let block = density as i64;
    let mut blocks = BTreeSet::new();
    for leaf in leaves {
        let world_box = leaf_world_box(leaf, density);
        let low: [i64; 3] =
            std::array::from_fn(|axis| (world_box.min[axis] - block).div_euclid(block) * block);
        let high: [i64; 3] = std::array::from_fn(|axis| world_box.max[axis] + block);
        let mut block_z = low[2];
        while block_z < high[2] {
            let mut block_y = low[1];
            while block_y < high[1] {
                let mut block_x = low[0];
                while block_x < high[0] {
                    blocks.insert([block_x, block_y, block_z]);
                    block_x += block;
                }
                block_y += block;
            }
            block_z += block;
        }
    }
    blocks
}

/// Expand one boundary block's decomposed cuboids into their absolute occupied cells,
/// inserting into `cells`. The cuboid coordinates are block-local inclusive indices, so the
/// absolute cell is `block_min + local`.
fn insert_boundary_cells(
    cells: &mut BTreeSet<[i64; 3]>,
    geometry: &MicroblockGeometry,
    block_min: [i64; 3],
) {
    for cuboid in &geometry.cuboids {
        for local_z in cuboid.min[2]..=cuboid.max[2] {
            for local_y in cuboid.min[1]..=cuboid.max[1] {
                for local_x in cuboid.min[0]..=cuboid.max[0] {
                    cells.insert([
                        block_min[0] + local_x as i64,
                        block_min[1] + local_y as i64,
                        block_min[2] + local_z as i64,
                    ]);
                }
            }
        }
    }
}

/// The occupancy the two-layer store REPORTS, trusting the interval elision: an AIR block
/// contributes nothing, a COARSE-SOLID block contributes every one of its `density³` cells
/// (no per-voxel data stored), and a BOUNDARY block is resolved per-voxel.
fn classify_elided_occupancy(
    leaves: &[&LeafProducer],
    density: u32,
    blocks: &BTreeSet<[i64; 3]>,
) -> BTreeSet<[i64; 3]> {
    let block = density as i64;
    let mut cells = BTreeSet::new();
    for &block_min in blocks {
        let block_aabb = VoxelAabb::new(
            block_min,
            [block_min[0] + block, block_min[1] + block, block_min[2] + block],
        );
        match classify_chunk_block(leaves, block_aabb, density) {
            BlockClassification::Air => {}
            BlockClassification::CoarseSolid(_) => {
                for local_z in 0..density {
                    for local_y in 0..density {
                        for local_x in 0..density {
                            cells.insert([
                                block_min[0] + local_x as i64,
                                block_min[1] + local_y as i64,
                                block_min[2] + local_z as i64,
                            ]);
                        }
                    }
                }
            }
            BlockClassification::Boundary => {
                let geometry = resolve_boundary_block(leaves, block_min, density, density);
                insert_boundary_cells(&mut cells, &geometry, block_min);
            }
        }
    }
    cells
}

/// The occupancy from FORCING a per-voxel resolve of every block — the always-exact
/// reference that trusts NO elision (it is what a BOUNDARY block would resolve to
/// everywhere).
fn forced_per_voxel_occupancy(
    leaves: &[&LeafProducer],
    density: u32,
    blocks: &BTreeSet<[i64; 3]>,
) -> BTreeSet<[i64; 3]> {
    let mut cells = BTreeSet::new();
    for &block_min in blocks {
        let geometry = resolve_boundary_block(leaves, block_min, density, density);
        insert_boundary_cells(&mut cells, &geometry, block_min);
    }
    cells
}

/// Assert the two two-layer routes agree over the leaf's whole covering window, and return
/// the occupied-cell count (for coverage reporting).
fn assert_classify_matches_forced(leaves: &[&LeafProducer], density: u32, label: &str) -> usize {
    let blocks = covering_block_set(leaves, density);
    let elided = classify_elided_occupancy(leaves, density, &blocks);
    let forced = forced_per_voxel_occupancy(leaves, density, &blocks);
    assert_eq!(
        elided, forced,
        "[{label}] the two-layer interval-elision occupancy (classify → coarse-fill + \
         boundary-resolve) must equal the forced per-voxel resolve over the SAME window — a \
         mismatch is the classifier drifting from the resolve at a sub-voxel / off-block / \
         rotated seat (the ADR 0027 bug class)"
    );
    forced.len()
}

/// **THE GREEN SAFETY NET.** Over a deterministic spread of placements — block-aligned AND
/// off-block integer offsets, zero AND fractional `offset_local_voxels`, identity AND genuine
/// quaternion rotations, three primitive shapes, densities 8 / 16 / 64 (all within the 1..=64
/// bound) — the two-layer classifier's elided occupancy is IDENTICAL to the forced per-voxel
/// resolve. Both paths read the ADR 0027 affine, so both are fraction- and rotation-aware and
/// must agree; a regression that makes either blind to a sub-voxel or off-block seat parts
/// them here.
///
/// Deterministic by construction: an enumerated grid of cases, no RNG and no wall clock.
#[test]
fn classify_agrees_with_forced_per_voxel_resolve() {
    let mut total_cases = 0usize;
    let mut total_solid_cells = 0usize;

    // The genuinely off-axis rotations under test — one simple tilt and one compound turn,
    // both provably non-lattice (they exercise the inverse-resample gather, not the exact
    // permutation).
    let tilt = Quat::from_rotation_x(0.7);
    let compound = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(0.3);

    for density in [8u32, 16] {
        let block = density as i64;
        // (label, integer world offset, fractional local slide, rotation). The offsets mix
        // block-aligned (`k·d`) with off-block (`-33`, `-31`, `d+5` — none a whole multiple
        // of 8 or 16), and the local slides mix zero with the ADR 0027 quarter/half seats.
        let placements: [(&str, [i64; 3], [f32; 3], Quat); 6] = [
            ("block-aligned", [2 * block, 0, block], [0.0, 0.0, 0.0], Quat::IDENTITY),
            ("off-block-integer", [-33, -31, block + 5], [0.0, 0.0, 0.0], Quat::IDENTITY),
            ("block-aligned+fractional", [2 * block, 0, block], [0.5, 0.5, 0.0], Quat::IDENTITY),
            ("off-block+fractional", [-33, -31, block + 5], [0.25, 0.0, 0.75], Quat::IDENTITY),
            ("block-aligned+rotation", [2 * block, 0, block], [0.0, 0.0, 0.0], tilt),
            ("off-block+fractional+rotation", [-33, -31, block + 5], [0.5, 0.5, 0.0], compound),
        ];
        for kind in [ShapeKind::Box, ShapeKind::Cylinder, ShapeKind::Sphere] {
            for (variation, world_offset, local, rotation) in placements {
                let leaf =
                    placed_single_leaf(kind, 2, density, world_offset, local, rotation);
                let label = format!("d{density} {kind:?} {variation}");
                total_solid_cells +=
                    assert_classify_matches_forced(&[&leaf], density, &label);
                total_cases += 1;
            }
        }
    }

    // A couple of density-64 cases (the top of the 1..=64 bound) on a small 1-block shape so
    // the per-voxel window stays cheap: one off-block fractional seat, one rotated seat.
    for (variation, local, rotation) in [
        ("off-block+fractional", [0.5, 0.5, 0.0], Quat::IDENTITY),
        ("block-aligned+rotation", [0.0, 0.0, 0.0], tilt),
    ] {
        let leaf = placed_single_leaf(ShapeKind::Box, 1, 64, [-33, -31, 96], local, rotation);
        total_solid_cells +=
            assert_classify_matches_forced(&[&leaf], 64, &format!("d64 Box {variation}"));
        total_cases += 1;
    }

    eprintln!(
        "differential occupancy fuzz: {total_cases} placements agree \
         (classify-elision == forced per-voxel), {total_solid_cells} solid cells checked"
    );
}

/// **GREEN — two-layer vs the dense oracle where they must agree.** Integer offsets
/// (block-aligned AND off-block), identity rotation, zero local slide: the dense
/// `resolve_region` handles an integer translation exactly, so the two-layer round-trip is
/// bit-identical to it. This pins that the two-layer path did not regress on ordinary
/// (including off-block) integer placement — the half of the bug class where the dense
/// oracle is a valid reference.
#[test]
fn two_layer_matches_dense_for_integer_placements() {
    for density in [8u32, 16] {
        let block = density as i64;

        // A single tool at an OFF-BLOCK integer offset (−33 / −31 are not multiples of 8 or
        // 16). The recentre folds a single leaf onto the origin, so this mainly pins that the
        // covering-range / recentre stay exact for a non-block-aligned single leaf.
        let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [3, 3, 3], 1, density);
        let mut node = Node::new("Cyl", NodeContent::Tool { shape, material: FUZZ_MATERIAL });
        node.transform = NodeTransform::from_offset_voxels([-33, -31, 2 * block]);
        let scene = Scene::from_nodes(vec![node]);
        super::core::assert_two_layer_round_trip_matches_dense(
            &scene,
            density,
            &format!("off-block-int-single-d{density}"),
        );

        // TWO tools at an off-block RELATIVE offset, so the inter-leaf seam lands off the
        // block lattice — the case a block-aligned assumption would have gridded into
        // fragments. Both dense and two-layer see the same integer offsets, so they agree.
        let base = {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, density);
            let mut node = Node::new("Base", NodeContent::Tool { shape, material: FUZZ_MATERIAL });
            node.transform = NodeTransform::from_offset_voxels([0, 0, 0]);
            node
        };
        let offset_partner = {
            let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, density);
            let mut node =
                Node::new("Partner", NodeContent::Tool { shape, material: MaterialChoice::Wood });
            // Off-block relative slide on X (block + 3 voxels), block-aligned on the rest.
            node.transform = NodeTransform::from_offset_voxels([block + 3, 0, 0]);
            node
        };
        let scene = Scene::from_nodes(vec![base, offset_partner]);
        super::core::assert_two_layer_round_trip_matches_dense(
            &scene,
            density,
            &format!("off-block-int-pair-d{density}"),
        );
    }
}

/// **GREEN — two-layer vs the dense oracle over rotated / sub-voxel seats (Step 2 landed).**
/// For a fractional `offset_local_voxels` seat and for a genuine rotation, BOTH paths now fold
/// through substrate's ONE placement affine (`substrate::spatial::LeafPlacement`): the two-layer
/// classifier resamples an out-of-phase leaf by inverse gather, and the dense oracle
/// (`Scene::resolve_region` / `resolve_chunk_rebased`) applies the SAME affine + inverse gather
/// (`document`'s `gather_placed_field_into_grid`) instead of the old translation-only stamp. So
/// the occupied cell sets agree, and this is a permanent green differential — a regression that
/// makes the dense oracle rotation- or fraction-blind again parts them here.
#[test]
fn two_layer_matches_dense_for_rotated_and_fractional_placements() {
    let density = 16u32;

    // (1) A fractional sub-voxel seat: the two-layer path resamples half a voxel over, the
    // dense oracle does not — so the occupied cell sets differ.
    let shape = SdfShape::from_blocks(ShapeKind::Box, [3, 3, 3], 1, density);
    let mut node = Node::new("Box", NodeContent::Tool { shape, material: FUZZ_MATERIAL });
    node.transform.offset_local_voxels = [0.5, 0.5, 0.0];
    let scene = Scene::from_nodes(vec![node]);
    super::core::assert_two_layer_round_trip_matches_dense(&scene, density, "fractional-local");

    // (2) A genuine off-axis rotation: the two-layer path turns the cylinder, the dense
    // oracle leaves it upright — so the occupied cell sets differ.
    let shape = SdfShape::from_blocks(ShapeKind::Cylinder, [2, 2, 4], 1, density);
    let mut node = Node::new("Cyl", NodeContent::Tool { shape, material: FUZZ_MATERIAL });
    node.transform = node.transform.with_rotation(Quat::from_rotation_x(0.7));
    let scene = Scene::from_nodes(vec![node]);
    super::core::assert_two_layer_round_trip_matches_dense(&scene, density, "rotated");
}
