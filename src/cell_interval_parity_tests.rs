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

/// **Issue #62 — interior soundness under an UNDER-ESTIMATED Lipschitz constant.**
///
/// The ADR 0010-core audit flagged that the SDF widening `L = max_semi / min_semi` is not
/// actually an upper bound on the IQ-ellipsoid's gradient: deep inside a thin ellipsoid the
/// true gradient runs far above it. Today that is still SOUND, but only via three
/// circumstances the audit enumerated — (a) field magnitude dominates in the interior, (b)
/// near the surface the gradient stays under the claimed `L`, and (c) for strongly
/// anisotropic shapes the widened `L` makes every block `Boundary` anyway. The coverage gap
/// was that `sdf_cell_interval_never_misclassifies` above fuzzes anisotropy only to 8:1 and
/// asserts nothing about the interior branch.
///
/// One correction to the audit, measured rather than assumed: the existing test does NOT in
/// fact "pass only because those cells come out Boundary" — a crude `L = 1` retune makes it
/// fail on its `CoarseSolid` branch. So the gap is narrower than stated. What this test adds
/// is (i) anisotropy to 32:1, (ii) explicit deep-interior coverage (628 fully-interior cells),
/// and (iii) a guard that fires at the CONTAINMENT level, before any verdict goes wrong —
/// which is a strictly earlier and more diagnosable failure than a misclassification.
///
/// It also **measures which safety net is doing the work per kind**, which differs sharply
/// between them and partly contradicts the issue's framing:
///
/// * **Cylinder / Tube** — the required `L` measured ~1.0 at EVERY anisotropy up to 32:1, so
///   the anisotropy widening was pure waste (headroom 8–33×) and nets (a)/(c) were never
///   load-bearing for them. **This was since acted on:** the elliptical cylinder is
///   *provably* exactly 1-Lipschitz (the `min(ax,ay)` scale factor cancels the worst-case
///   radial gradient — see `cell_field_interval`), and `Tube`'s `outer.max(−inner)` inherits
///   it, so both now take `L = 1` outright rather than an axis-separated anisotropy. A
///   64×4×4 cylinder went from **0 → 128** coarse interior blocks, 32×8×8 from 8 → 744; the
///   isotropic control is unchanged, as it must be.
/// * **Sphere** — the required `L` EXCEEDS the claimed one by ~8.7× (claimed 32, needed 277
///   for a 32:1 ellipsoid). The sphere bound is *already* an under-estimate; it is NOT
///   over-conservative and must NOT be tightened. Its soundness rests entirely on (a): the
///   interval `[f_c ± L·R]` is far wider than the field's whole range, because the range is
///   bounded by the MINOR semi-axis while `L·R` scales with the MAJOR one.
///
/// So the assertion below is deliberately NOT "required L ≤ claimed L" — that is false, and
/// asserting it would fail today. It is the property that actually holds and is load-bearing:
/// **wherever the Lipschitz constant is under-estimated, the verdict must be `Boundary`.** A
/// future tightening (the issue's perf half) that lets such a cell claim `Air`/`CoarseSolid`
/// breaks this test, which is exactly the trap the audit asked to be caught.
#[test]
fn strongly_anisotropic_sdf_cells_stay_sound_where_lipschitz_is_underestimated() {
    use glam::Vec3;
    use voxel_core::voxel::signed_distance;

    let density = 16u32;
    // Anisotropy to 32:1 — the existing fuzz above reaches only 8:1.
    let sizes: [[u32; 3]; 5] = [
        [256, 32, 32],   // 8:1
        [256, 16, 16],   // 16:1
        [512, 16, 16],   // 32:1 — the sphere's worst under-estimate
        [384, 96, 32],   // fully anisotropic, 12:1
        [128, 128, 128], // isotropic control (claimed L == 1 == true L)
    ];
    let kinds = [ShapeKind::Sphere, ShapeKind::Cylinder, ShapeKind::Tube];

    let mut underestimated_cells = 0u64;
    let mut interior_cells = 0u64;
    let mut checked = 0u64;

    for &size in &sizes {
        for &kind in &kinds {
            let shape = SdfShape::from_voxels(kind, size, 1);
            let dims = shape.grid_dimensions(density);
            let half = Vec3::new(
                dims[0] as f32 / 2.0,
                dims[1] as f32 / 2.0,
                dims[2] as f32 / 2.0,
            );
            let wall_voxels = (shape.wall_blocks * density) as f32;
            let block = density as i64;
            let mut z = 0i64;
            while z < dims[2] as i64 {
                let mut y = 0i64;
                while y < dims[1] as i64 {
                    let mut x = 0i64;
                    while x < dims[0] as i64 {
                        let cell = VoxelAabb::new(
                            [x, y, z],
                            [
                                (x + block).min(dims[0] as i64),
                                (y + block).min(dims[1] as i64),
                                (z + block).min(dims[2] as i64),
                            ],
                        );
                        let label = format!("SDF#62 {kind:?} {size:?}");
                        // (1) The verdict must never disagree with brute force — the same
                        // exactness gate as above, now at up to 32:1 anisotropy.
                        assert_cell_bound_exact(&shape, cell, density, &label);

                        if let Some(interval) = shape.cell_field_interval(cell, density) {
                            checked += 1;
                            // The Lipschitz constant this cell's samples ACTUALLY require.
                            let centre = Vec3::new(
                                (cell.min[0] + cell.max[0]) as f32 / 2.0 - half.x,
                                (cell.min[1] + cell.max[1]) as f32 / 2.0 - half.y,
                                (cell.min[2] + cell.max[2]) as f32 / 2.0 - half.z,
                            );
                            let field_at_centre =
                                signed_distance(kind, centre, half, wall_voxels);
                            // Recover the Lipschitz constant the PRODUCTION code actually
                            // used, rather than duplicating its formula here: the interval is
                            // `[f_c ± L·R]`, so `L = (max − min) / 2R`. Reading it back keeps
                            // this guard correct across the very change it exists to police —
                            // a copied formula would go stale the moment the bound is retuned
                            // and would then silently compare against the OLD constant.
                            let extent = Vec3::new(
                                (cell.max[0] - cell.min[0]) as f32,
                                (cell.max[1] - cell.min[1]) as f32,
                                (cell.max[2] - cell.min[2]) as f32,
                            );
                            let circumradius = (extent * 0.5).length();
                            let claimed_lipschitz = if circumradius > 1e-6 {
                                (interval.maximum - interval.minimum) / (2.0 * circumradius)
                            } else {
                                f32::INFINITY
                            };
                            let mut required_lipschitz: f32 = 0.0;
                            let mut every_sample_inside = true;
                            for k in cell.min[2]..cell.max[2] {
                                for j in cell.min[1]..cell.max[1] {
                                    for i in cell.min[0]..cell.max[0] {
                                        let sample = Vec3::new(
                                            i as f32 + 0.5 - half.x,
                                            j as f32 + 0.5 - half.y,
                                            k as f32 + 0.5 - half.z,
                                        );
                                        let field =
                                            signed_distance(kind, sample, half, wall_voxels);
                                        if field > SURFACE_ISOLEVEL {
                                            every_sample_inside = false;
                                        }
                                        // (2) Containment: the interval must bracket every
                                        // in-cell sample. This is what net (a) buys, and it
                                        // holds even where the Lipschitz constant does not.
                                        assert!(
                                            field >= interval.minimum && field <= interval.maximum,
                                            "{label}: sample {field} escaped the interval \
                                             {interval:?} (cell={cell:?})"
                                        );
                                        let travel = (sample - centre).length();
                                        if travel > 1e-6 {
                                            required_lipschitz = required_lipschitz
                                                .max((field - field_at_centre).abs() / travel);
                                        }
                                    }
                                }
                            }
                            if every_sample_inside {
                                interior_cells += 1;
                            }
                            // (3) THE TRAP GUARD. Where the claimed constant is an
                            // under-estimate, the interval is sound only by magnitude
                            // dominance — so it must not be making a coarse claim. If a
                            // future tightening lets one of these decide Air/CoarseSolid,
                            // this fires.
                            // 1% tolerance: for an ISOTROPIC shape the claimed constant is
                            // exactly the true one (a real SDF has unit gradient), so the
                            // measured value lands a few ULPs over it — 1.0000029 vs 1.0 —
                            // which is float noise, not an under-estimate. The real ones are
                            // ~8.7× over, so the separation is not delicate.
                            if required_lipschitz > claimed_lipschitz * 1.01 {
                                underestimated_cells += 1;
                                assert_eq!(
                                    interval.classify(SURFACE_ISOLEVEL),
                                    FieldClassification::Boundary,
                                    "{label}: cell needs Lipschitz {required_lipschitz} but only \
                                     {claimed_lipschitz} is claimed, yet it made a COARSE claim \
                                     — the under-estimate is now load-bearing (cell={cell:?}, \
                                     interval={interval:?})"
                                );
                            }
                        }
                        x += block;
                    }
                    y += block;
                }
                z += block;
            }
        }
    }

    // Non-vacuity, both ways: the fuzz must actually reach cells where the constant is
    // under-estimated (or guard (3) proves nothing), AND must actually cover deep-interior
    // cells (the coverage gap the audit named).
    assert!(
        underestimated_cells > 0,
        "no cell exercised an under-estimated Lipschitz constant — guard (3) is vacuous"
    );
    assert!(
        interior_cells > 0,
        "no fully-interior cell was covered — the audit's interior gap is still open"
    );
    eprintln!(
        "SDF #62 interior soundness: {checked} cells ({interior_cells} fully interior, \
         {underestimated_cells} with an under-estimated Lipschitz constant, all Boundary)"
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

/// The cloud field is BOUNDABLE (ADR 0021) — it was documented as unboundable on the
/// mistaken grounds that fBm cannot be bracketed over a cell. Only the noise's global range
/// is needed; the radial term supplies the rest, so cells classify from puff geometry with
/// no noise evaluation.
///
/// Two things are asserted. Soundness: no coarse verdict may disagree with brute force,
/// which is what would break if the noise range bound were wrong. And usefulness: the fuzz
/// must actually produce AIR verdicts, since a bound that never decides anything would pass
/// the soundness half while delivering exactly the elision the old `None` did.
#[test]
fn debug_cloud_field_is_boundable_and_sound() {
    use document::debug_clouds::DebugCloudField;
    let mut cases = 0u64;
    let mut air = 0u64;
    let mut solid = 0u64;
    let mut boundary = 0u64;
    let mut unbounded = 0u64;

    for seed in [7u32, 11, 23, 1009] {
        let dimensions = [48u32, 32, 48];
        let field = DebugCloudField { dimensions, seed };
        let cells = fuzz_cells(dimensions, 16, 0xC10D_u64 ^ seed as u64);
        for &cell in &cells {
            // The over-claim police: any coarse verdict must match brute force EXACTLY.
            assert_cell_bound_exact(&field, cell, 16, &format!("DebugCloudField seed {seed}"));
            match field.cell_field_interval(cell, 16) {
                None => unbounded += 1,
                Some(interval) => match interval.classify(SURFACE_ISOLEVEL) {
                    FieldClassification::Air => air += 1,
                    FieldClassification::CoarseSolid => solid += 1,
                    FieldClassification::Boundary => boundary += 1,
                },
            }
            cases += 1;
        }
    }

    assert_eq!(unbounded, 0, "DebugCloudField should now bracket every cell");
    assert!(
        air > 0,
        "cloud fuzz produced no AIR verdict — the bound decides nothing and elides nothing"
    );
    assert!(boundary > 0, "cloud fuzz never produced a BOUNDARY verdict; the fuzz misses puffs");
    eprintln!(
        "DebugCloudField parity: {cases} cells classified ({air} air, {solid} solid, \
         {boundary} boundary)"
    );
}
