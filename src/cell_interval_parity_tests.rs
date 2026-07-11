//! ADR 0010 E1 — the STANDALONE exactness parity for the conservative cell-interval
//! bound primitive ([`VoxelProducer::cell_field_interval`]).
//!
//! The bound is op-stack math wired to nothing yet; its whole point is that a coarse
//! AIR / SOLID classification derived from the interval can NEVER disagree with a
//! brute-force per-voxel evaluation of the cell:
//!
//! * the interval says **AIR** ([`FieldClassification::Air`]) ⇒ brute force finds ZERO
//!   occupied voxels in the cell;
//! * the interval says **SOLID** ([`FieldClassification::CoarseSolid`]) ⇒ brute force
//!   finds EVERY voxel in the cell occupied;
//! * **BOUNDARY** (straddling) or **`None`** (unboundable) is ALWAYS allowed — it is the
//!   safe fallback (resolve the cell per-voxel), so it can never be wrong.
//!
//! A bound that claims AIR or SOLID where brute force disagrees FAILS the test. The
//! cells fuzzed per producer include ones fully inside, fully outside, straddling the
//! surface, and tiny single-voxel features.
//!
//! Also unit-tests the CSG interval composition (union / subtract / intersect).

use crate::spatial_index::VoxelAabb;
use crate::voxel::{
    union_field_intervals, FieldClassification, FieldInterval, SdfShape, ShapeKind, VoxelGrid,
    VoxelProducer, SURFACE_ISOLEVEL,
};

/// Count the voxels brute force actually fills inside `cell` (in the producer's local
/// voxel-index frame) by resolving JUST that window. Returns `(occupied, total)` where
/// `total` is the number of voxel cells in the (clamped-to-grid) box.
fn brute_force_cell_occupancy(
    producer: &dyn VoxelProducer,
    cell: VoxelAabb,
    voxels_per_block: u32,
) -> (u64, u64) {
    let mut grid = VoxelGrid::default();
    producer.resolve_into(&mut grid, voxels_per_block, cell);
    let dims = grid.dimensions.map(|d| d as i64);
    // The window clamps to `[0, full_dim)`; the brute-force "total" is the count of
    // real voxel cells the clamped window spans (a cell partly outside the grid only
    // owns its in-grid voxels — the producer never emits outside `[0, dim)`).
    let mut total: u64 = 1;
    for (axis, &dim) in dims.iter().enumerate() {
        let lo = cell.min[axis].clamp(0, dim);
        let hi = cell.max[axis].clamp(0, dim).max(lo);
        total *= (hi - lo) as u64;
    }
    (grid.occupied.len() as u64, total)
}

/// Assert the interval classification of `cell` never disagrees with brute force.
fn assert_cell_bound_exact(
    producer: &dyn VoxelProducer,
    cell: VoxelAabb,
    voxels_per_block: u32,
    label: &str,
) {
    let Some(interval) = producer.cell_field_interval(cell, voxels_per_block) else {
        // `None` ⇒ unboundable ⇒ BOUNDARY fallback ⇒ always allowed.
        return;
    };
    let classification = interval.classify(SURFACE_ISOLEVEL);
    let (occupied, total) = brute_force_cell_occupancy(producer, cell, voxels_per_block);
    match classification {
        FieldClassification::Air => assert_eq!(
            occupied, 0,
            "{label}: bound said AIR but brute force found {occupied}/{total} occupied \
             (cell={cell:?}, interval={interval:?})"
        ),
        FieldClassification::CoarseSolid => assert_eq!(
            occupied, total,
            "{label}: bound said SOLID but brute force found only {occupied}/{total} occupied \
             (cell={cell:?}, interval={interval:?})"
        ),
        // Straddling ⇒ resolved per-voxel ⇒ always exact.
        FieldClassification::Boundary => {}
    }
}

/// A tiny deterministic LCG so the fuzz is reproducible without a dev-dependency.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    /// A value in `[lo, hi]` inclusive.
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo {
            return lo;
        }
        let span = (hi - lo + 1) as u64;
        lo + (self.next_u64() % span) as i64
    }
}

/// Generate a varied set of cell boxes over a producer of the given full dimensions:
/// the full grid, every block-sized cell on a coarse stride, single-voxel cells, cells
/// straddling the boundary, and random boxes (including partly/fully out of range).
fn fuzz_cells(full_dimensions: [u32; 3], voxels_per_block: u32, seed: u64) -> Vec<VoxelAabb> {
    let dims = full_dimensions.map(|d| d as i64);
    let mut cells = Vec::new();
    let block = voxels_per_block.max(1) as i64;

    // The whole grid as one cell.
    cells.push(VoxelAabb::new([0, 0, 0], dims));

    // Block-sized cells tiling the grid on the block lattice (this is the real
    // classification granularity ADR 0010 uses), PLUS overhang cells past the edge.
    let mut z = -block;
    while z < dims[2] + block {
        let mut y = -block;
        while y < dims[1] + block {
            let mut x = -block;
            while x < dims[0] + block {
                cells.push(VoxelAabb::new(
                    [x, y, z],
                    [x + block, y + block, z + block],
                ));
                x += block;
            }
            y += block;
        }
        z += block;
    }

    // Every single-voxel cell would be too many; sample a deterministic spread,
    // including the 8 corners + centre (tiny-feature cases).
    let corners = [
        [0, 0, 0],
        [dims[0] - 1, 0, 0],
        [0, dims[1] - 1, 0],
        [0, 0, dims[2] - 1],
        [dims[0] - 1, dims[1] - 1, dims[2] - 1],
        [dims[0] / 2, dims[1] / 2, dims[2] / 2],
    ];
    for corner in corners {
        cells.push(VoxelAabb::new(
            corner,
            [corner[0] + 1, corner[1] + 1, corner[2] + 1],
        ));
    }

    // Random boxes — varied sizes, some straddling edges, some out of range.
    let mut rng = Lcg::new(seed);
    for _ in 0..200 {
        let min = [
            rng.range(-block, dims[0] + block),
            rng.range(-block, dims[1] + block),
            rng.range(-block, dims[2] + block),
        ];
        let extent = [
            rng.range(1, block + 2),
            rng.range(1, block + 2),
            rng.range(1, block + 2),
        ];
        cells.push(VoxelAabb::new(
            min,
            [min[0] + extent[0], min[1] + extent[1], min[2] + extent[2]],
        ));
    }

    cells
}

#[test]
fn sdf_cell_interval_never_misclassifies() {
    // A spread of shapes: isotropic + ANISOTROPIC (the widened-Lipschitz cases),
    // even / odd / mixed-parity sizes, and a thin sliver. density 16 keeps the block
    // lattice the chiseling granularity.
    let density = 16u32;
    let sizes: [[u32; 3]; 7] = [
        [32, 32, 32], // isotropic cube of cells
        [48, 16, 48], // flat disc (anisotropic)
        [16, 48, 16], // tall (anisotropic)
        [33, 17, 49], // all-odd mixed parity
        [40, 24, 16], // fully anisotropic
        [16, 16, 16], // one block per axis
        [64, 8, 8],   // thin sliver (extreme anisotropy)
    ];
    let kinds = [
        ShapeKind::Box,
        ShapeKind::Sphere,
        ShapeKind::Cylinder,
        ShapeKind::Tube,
        ShapeKind::Torus,
    ];

    let mut cases = 0u64;
    let mut air = 0u64;
    let mut solid = 0u64;
    let mut boundary = 0u64;
    for &size in &sizes {
        let cells = fuzz_cells(size, density, 0x5DF_u64 ^ (size[0] as u64));
        for &kind in &kinds {
            let shape = SdfShape::from_voxels(kind, size, 1);
            for &cell in &cells {
                assert_cell_bound_exact(&shape, cell, density, &format!("SDF {kind:?} {size:?}"));
                if let Some(interval) = shape.cell_field_interval(cell, density) {
                    match interval.classify(SURFACE_ISOLEVEL) {
                        FieldClassification::Air => air += 1,
                        FieldClassification::CoarseSolid => solid += 1,
                        FieldClassification::Boundary => boundary += 1,
                    }
                }
                cases += 1;
            }
        }
    }

    // The fuzz must actually exercise ALL THREE verdicts, or it proves nothing about
    // the AIR / SOLID branches (a bound that always straddled would trivially pass).
    assert!(air > 0, "fuzz never produced an AIR verdict");
    assert!(solid > 0, "fuzz never produced a SOLID verdict");
    assert!(boundary > 0, "fuzz never produced a BOUNDARY verdict");
    eprintln!(
        "SDF parity: {cases} cells classified ({air} air, {solid} solid, {boundary} boundary)"
    );
}

#[test]
fn sketch_extrude_cell_interval_never_misclassifies() {
    use crate::sketch::{PlaneAxis, Sketch, SketchSolid};
    let density = 16u32;
    // A rectangle and an L-shape (a concave polygon, so the bbox over-claims).
    let rectangle = Sketch::rectangle(PlaneAxis::Z, 40, 24);
    let l_shape = Sketch::new(
        PlaneAxis::Z,
        vec![
            crate::sketch::SketchPoint::new(0, 0),
            crate::sketch::SketchPoint::new(40, 0),
            crate::sketch::SketchPoint::new(40, 16),
            crate::sketch::SketchPoint::new(16, 16),
            crate::sketch::SketchPoint::new(16, 40),
            crate::sketch::SketchPoint::new(0, 40),
        ],
    );

    let mut cases = 0u64;
    let mut air = 0u64;
    let mut solid = 0u64;
    let mut boundary = 0u64;
    for (label, sketch) in [("rect", rectangle), ("L", l_shape)] {
        let producer = SketchSolid::extrude(sketch, 24);
        let cells = fuzz_cells(producer.grid_dimensions(), density, 0x57E7_u64);
        for &cell in &cells {
            // The over-claim police: any coarse claim (all-inside or the L reflex corner)
            // must match brute force EXACTLY.
            assert_cell_bound_exact(&producer, cell, density, &format!("Sketch-extrude {label}"));
            if let Some(interval) = producer.cell_field_interval(cell, density) {
                match interval.classify(SURFACE_ISOLEVEL) {
                    FieldClassification::Air => air += 1,
                    FieldClassification::CoarseSolid => solid += 1,
                    FieldClassification::Boundary => boundary += 1,
                }
            }
            cases += 1;
        }
    }
    // Interior elision must actually fire (a solid extrude interior is coarse), and the
    // concave L must still leave straddling blocks boundary.
    assert!(air > 0, "sketch-extrude fuzz never produced an AIR verdict");
    assert!(solid > 0, "sketch-extrude fuzz never produced a SOLID verdict (elision dead)");
    assert!(boundary > 0, "sketch-extrude fuzz never produced a BOUNDARY verdict");
    eprintln!(
        "Sketch-extrude parity: {cases} cells classified ({air} air, {solid} solid, {boundary} boundary)"
    );
}

#[test]
fn sketch_revolve_cell_interval_never_misclassifies() {
    use crate::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchSolid};
    let density = 16u32;
    // A full 360° revolve (interior elides) and a PARTIAL 180° revolve (deferred to
    // boundary) — both must never misclassify against brute force.
    let mut cases = 0u64;
    let mut air = 0u64;
    let mut solid = 0u64;
    let mut boundary = 0u64;
    for (label, turn) in [("full-360", 360u32), ("partial-180", 180)] {
        let profile = Sketch::rectangle(PlaneAxis::Z, 24, 16);
        let producer = SketchSolid::revolve(profile, RevolveAxis::InPlane0, turn);
        let cells = fuzz_cells(producer.grid_dimensions(), density, 0x5EF0_u64);
        for &cell in &cells {
            assert_cell_bound_exact(&producer, cell, density, &format!("Sketch-revolve {label}"));
            if let Some(interval) = producer.cell_field_interval(cell, density) {
                match interval.classify(SURFACE_ISOLEVEL) {
                    FieldClassification::Air => air += 1,
                    FieldClassification::CoarseSolid => solid += 1,
                    FieldClassification::Boundary => boundary += 1,
                }
            }
            cases += 1;
        }
    }
    // The full-turn revolve MUST elide some interior (coarse), proving the radius/axial
    // rectangle test fires; the partial turn contributes only air/boundary (deferred).
    assert!(solid > 0, "sketch-revolve fuzz never produced a SOLID verdict (elision dead)");
    assert!(air > 0, "sketch-revolve fuzz never produced an AIR verdict");
    assert!(boundary > 0, "sketch-revolve fuzz never produced a BOUNDARY verdict");
    eprintln!(
        "Sketch-revolve parity: {cases} cells classified ({air} air, {solid} solid, {boundary} boundary)"
    );
}

#[test]
fn debug_cloud_field_is_unboundable() {
    use crate::debug_clouds::DebugCloudField;
    let field = DebugCloudField {
        dimensions: [48, 32, 48],
        seed: 7,
    };
    let cells = fuzz_cells([48, 32, 48], 16, 0xC10D_u64);
    let mut cases = 0u64;
    for &cell in &cells {
        // Unboundable ⇒ always `None` (the safe BOUNDARY fallback). The shared
        // assertion also tolerates `None`, but pin the contract explicitly here.
        assert!(
            field.cell_field_interval(cell, 16).is_none(),
            "DebugCloudField must be unboundable (None) for every cell"
        );
        assert_cell_bound_exact(&field, cell, 16, "DebugCloudField");
        cases += 1;
    }
    eprintln!("DebugCloudField parity: {cases} cells (all unboundable None)");
}

#[test]
fn csg_interval_union_is_min_of_fields() {
    let a = FieldInterval::new(-2.0, 3.0);
    let b = FieldInterval::new(-5.0, 1.0);
    // union = min(a, b) ⇒ [min(min), min(max)].
    assert_eq!(a.union(b), FieldInterval::new(-5.0, 1.0));
    assert_eq!(a.union(b), b.union(a), "union is commutative");
}

#[test]
fn csg_interval_intersect_is_max_of_fields() {
    let a = FieldInterval::new(-2.0, 3.0);
    let b = FieldInterval::new(-5.0, 1.0);
    // intersect = max(a, b) ⇒ [max(min), max(max)].
    assert_eq!(a.intersect(b), FieldInterval::new(-2.0, 3.0));
    assert_eq!(a.intersect(b), b.intersect(a), "intersect is commutative");
}

#[test]
fn csg_interval_subtract_is_intersect_with_negated() {
    let a = FieldInterval::new(-2.0, 3.0);
    let b = FieldInterval::new(-5.0, 1.0);
    // A − B = max(dA, −dB); −b = [−1, 5]; max ⇒ [max(-2,-1), max(3,5)] = [-1, 5].
    assert_eq!(b.negate(), FieldInterval::new(-1.0, 5.0));
    assert_eq!(a.subtract(b), FieldInterval::new(-1.0, 5.0));
}

/// The interval-arithmetic composition must AGREE with the brute-force field over a
/// sampled field range: for any pair of sample fields drawn from `[aMin,aMax]` and
/// `[bMin,bMax]`, the composed value lies inside the composed interval. This is the
/// soundness property the classifier relies on.
#[test]
fn csg_composition_brackets_every_sample() {
    let a = FieldInterval::new(-2.0, 3.0);
    let b = FieldInterval::new(-5.0, 1.0);
    let union = a.union(b);
    let intersect = a.intersect(b);
    let subtract = a.subtract(b);
    let mut rng = Lcg::new(0xC56_u64);
    for _ in 0..5000 {
        let sample_a = a.minimum + (a.maximum - a.minimum) * (rng.next_u64() as f32 / u64::MAX as f32);
        let sample_b = b.minimum + (b.maximum - b.minimum) * (rng.next_u64() as f32 / u64::MAX as f32);
        let union_value = sample_a.min(sample_b);
        let intersect_value = sample_a.max(sample_b);
        let subtract_value = sample_a.max(-sample_b);
        assert!(
            (union.minimum..=union.maximum).contains(&union_value),
            "union {union:?} fails to bracket {union_value}"
        );
        assert!(
            (intersect.minimum..=intersect.maximum).contains(&intersect_value),
            "intersect {intersect:?} fails to bracket {intersect_value}"
        );
        assert!(
            (subtract.minimum..=subtract.maximum).contains(&subtract_value),
            "subtract {subtract:?} fails to bracket {subtract_value}"
        );
    }
}

#[test]
fn union_field_intervals_is_none_when_any_operand_unboundable() {
    let bounded = FieldInterval::new(-1.0, 1.0);
    assert_eq!(
        union_field_intervals([Some(bounded), Some(FieldInterval::new(-3.0, 0.5))]),
        Some(FieldInterval::new(-3.0, 0.5))
    );
    // Any None operand collapses the union to None (the unbounded operand could be
    // occupied anywhere ⇒ the whole union must be treated as boundary).
    assert_eq!(
        union_field_intervals([Some(bounded), None, Some(bounded)]),
        None
    );
    // Empty list ⇒ None.
    assert_eq!(union_field_intervals(std::iter::empty()), None);
}
