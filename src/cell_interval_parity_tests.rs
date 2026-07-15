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
//! The CSG interval algebra itself (union / intersect / subtract / classify) is unit-
//! tested where it now lives, in `substrate::interval::field_interval`; this file keeps only the
//! producer-vs-brute-force exactness gate.

use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{ShapeKind, VoxelGrid, SURFACE_ISOLEVEL};
use document::voxel::{FieldClassification, SdfShape, VoxelProducer};

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
    use document::sketch::{PlaneAxis, Sketch, SketchSolid};
    let density = 16u32;
    // A rectangle and an L-shape (a concave polygon, so the bbox over-claims).
    let rectangle = Sketch::rectangle(PlaneAxis::Z, 40, 24);
    let l_shape = Sketch::new(
        PlaneAxis::Z,
        vec![
            document::sketch::SketchPoint::new(0, 0),
            document::sketch::SketchPoint::new(40, 0),
            document::sketch::SketchPoint::new(40, 16),
            document::sketch::SketchPoint::new(16, 16),
            document::sketch::SketchPoint::new(16, 40),
            document::sketch::SketchPoint::new(0, 40),
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
    use document::sketch::{PlaneAxis, RevolveAxis, Sketch, SketchPoint, SketchSolid};
    // density 8 keeps a 6-block-diameter radius small enough to brute-force every cell of
    // every arc/axis/profile config, while still leaving interior blocks strictly OFF the
    // revolve axis — the only place a partial wedge can go coarse (axis-adjacent blocks are
    // always boundary).
    let density = 8u32;

    // Two profiles: a ONE-SIDED lathe rectangle (radial >= 0) and an AXIS-STRADDLING one
    // (radial spans negative→positive, exercising the resolve's mirrored `−radius` union so
    // a coarse claim must be solid under the SAME union).
    let one_sided = Sketch::rectangle(PlaneAxis::Z, 40, 24);
    let straddling = Sketch::new(
        PlaneAxis::Z,
        vec![
            SketchPoint::new(8, -24),
            SketchPoint::new(40, -24),
            SketchPoint::new(40, 24),
            SketchPoint::new(8, 24),
        ],
    );

    // Arcs sweeping across quadrant boundaries: a tiny sliver, contained in one quadrant,
    // spanning 2 / 3 quadrants, a near-full start>end-class wrap (359), and the full turn.
    // The wedge always opens from theta 0 (the +radial_a ray), so each arc's far edge lands
    // in a different quadrant, testing the angular-containment seam handling.
    let turns = [3u32, 45, 135, 270, 359, 360];
    let axes = [RevolveAxis::InPlane0, RevolveAxis::InPlane1];

    let mut cases = 0u64;
    let mut air = 0u64;
    let mut solid = 0u64;
    let mut boundary = 0u64;
    let mut partial_solid = 0u64;
    for (profile_label, profile) in [("one-sided", &one_sided), ("straddling", &straddling)] {
        for &axis in &axes {
            for &turn in &turns {
                let producer = SketchSolid::revolve(profile.clone(), axis, turn);
                let seed = 0x5EF0_u64 ^ (turn as u64) ^ ((axis as u64) << 20);
                let cells = fuzz_cells(producer.grid_dimensions(), density, seed);
                for &cell in &cells {
                    let label = format!("Sketch-revolve {profile_label} {axis:?} {turn}°");
                    assert_cell_bound_exact(&producer, cell, density, &label);
                    if let Some(interval) = producer.cell_field_interval(cell, density) {
                        match interval.classify(SURFACE_ISOLEVEL) {
                            FieldClassification::Air => air += 1,
                            FieldClassification::CoarseSolid => {
                                solid += 1;
                                if turn < 360 {
                                    partial_solid += 1;
                                }
                            }
                            FieldClassification::Boundary => boundary += 1,
                        }
                    }
                    cases += 1;
                }
            }
        }
    }
    // Both the full turn AND at least one PARTIAL wedge must elide interior blocks to coarse
    // (proving the radius/axial rectangle test AND the new angular-containment test fire); the
    // fuzz must also exercise air + boundary.
    assert!(solid > 0, "sketch-revolve fuzz never produced a SOLID verdict (elision dead)");
    assert!(
        partial_solid > 0,
        "PARTIAL-turn revolve never produced a coarse-solid verdict (angular elision dead)"
    );
    assert!(air > 0, "sketch-revolve fuzz never produced an AIR verdict");
    assert!(boundary > 0, "sketch-revolve fuzz never produced a BOUNDARY verdict");
    eprintln!(
        "Sketch-revolve parity: {cases} cells classified ({air} air, {solid} solid \
         [{partial_solid} partial-turn], {boundary} boundary)"
    );
}

#[test]
fn debug_cloud_field_is_unboundable() {
    use document::debug_clouds::DebugCloudField;
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
