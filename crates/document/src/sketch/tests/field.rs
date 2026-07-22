use super::*;
use crate::sketch::RevolveAxis;
use crate::voxel::VoxelProducer;
use voxel_core::voxel::VoxelGrid;
use std::collections::BTreeSet;

/// A set of extrude profiles worth stressing: a plain rectangle, a concave L, one with a
/// reflex notch, and one that self-intersects. Each is paired with a plane so all three
/// axis mappings get exercised.
fn extrude_field_cases() -> Vec<(&'static str, SketchSolid)> {
    let l_shape = vec![
        SketchPoint::new(0, 0), SketchPoint::new(6, 0), SketchPoint::new(6, 2),
        SketchPoint::new(2, 2), SketchPoint::new(2, 5), SketchPoint::new(0, 5),
    ];
    let notched = vec![
        SketchPoint::new(0, 0), SketchPoint::new(7, 0), SketchPoint::new(7, 6),
        SketchPoint::new(4, 3), SketchPoint::new(0, 6),
    ];
    let bowtie = vec![
        SketchPoint::new(0, 0), SketchPoint::new(6, 6),
        SketchPoint::new(0, 6), SketchPoint::new(6, 0),
    ];
    vec![
        ("rectangle/Z", SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 5, 3), 4)),
        ("L/X", SketchSolid::extrude(Sketch::new(PlaneAxis::X, l_shape), 3)),
        ("notched/Y", SketchSolid::extrude(Sketch::new(PlaneAxis::Y, notched), 2)),
        ("bowtie/Z", SketchSolid::extrude(Sketch::new(PlaneAxis::Z, bowtie), 3)),
    ]
}

/// The contract the whole field layer rests on (ADR 0019 Decision 4): the field must
/// agree with the resolve, over EVERY voxel of the grid rather than a sample.
///
/// Occupancy is read from the SIGN BIT, not `< 0.0`. A voxel centre can land exactly on a
/// profile edge — a diagonal between integer vertices passes through half-integer points,
/// which the notched case below actually hits at `(4.5, 3.5)` — and there the distance is
/// zero with only its sign carrying the even-odd verdict. `-0.0 < 0.0` is false, so a
/// naive comparison would report air where the resolve reports solid.
#[test]
fn extrude_signed_distance_agrees_with_the_resolve() {
    const DENSITY: u32 = 8;
    for (label, solid) in extrude_field_cases() {
        let mut grid = VoxelGrid::default();
        solid.resolve(&mut grid, DENSITY);
        let occupied: BTreeSet<[i32; 3]> =
            grid.occupied.iter().map(|voxel| voxel.local_index).collect();

        let dimensions = solid.grid_dimensions();
        let mut checked = 0u32;
        let mut inside = 0u32;
        let mut on_boundary = 0u32;
        for x in 0..dimensions[0] {
            for y in 0..dimensions[1] {
                for z in 0..dimensions[2] {
                    let centre =
                        [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    let distance = solid
                        .signed_distance(centre);
                    let field_says_solid = distance.is_sign_negative();
                    let resolve_says_solid =
                        occupied.contains(&[x as i32, y as i32, z as i32]);
                    assert_eq!(
                        field_says_solid, resolve_says_solid,
                        "{label} at {centre:?}: field distance {distance} says \
                         solid={field_says_solid}, resolve says {resolve_says_solid}"
                    );
                    if distance == 0.0 {
                        on_boundary += 1;
                    }
                    checked += 1;
                    inside += u32::from(field_says_solid);
                }
            }
        }
        assert!(checked > 0, "{label}: empty grid, nothing verified");
        assert!(inside > 0, "{label}: nothing solid, the case proves nothing");
        if label == "notched/Y" {
            assert!(
                on_boundary > 0,
                "the notched case is here BECAUSE its diagonal edge puts voxel centres \
                 exactly on the boundary; if that stops happening this test no longer \
                 guards the sign-bit contract"
            );
        }
    }
}

/// The extrude field must be 1-Lipschitz in Chebyshev, which is what makes a cell bound
/// from a single sample sound. Sampled on a fine sub-voxel lattice extending outside the
/// grid, so the exterior and the rim edges are covered too.
#[test]
fn extrude_signed_distance_is_one_lipschitz_in_chebyshev() {
    for (label, solid) in extrude_field_cases() {
        let dimensions = solid.grid_dimensions();
        let mut worst: f32 = 0.0;
        let step = 0.25f32;
        let span = |extent: u32| -> i32 { (extent as f32 / step) as i32 + 8 };
        for xi in -8..span(dimensions[0]) {
            for yi in -8..span(dimensions[1]) {
                for zi in -8..span(dimensions[2]) {
                    let p = [xi as f32 * step, yi as f32 * step, zi as f32 * step];
                    let here = solid.signed_distance(p);
                    for axis in 0..3 {
                        let mut q = p;
                        q[axis] += step;
                        let there = solid.signed_distance(q);
                        worst = worst.max((there - here).abs() / step);
                    }
                }
            }
        }
        assert!(
            worst <= 1.0 + 1e-5,
            "{label}: extrude field is not 1-Lipschitz in Chebyshev (worst {worst})"
        );
    }
}

/// Chebyshev is the metric an extrusion is exact in, and a rectangular prism is where
/// that is checkable by hand: from a point diagonally off a corner, the distance is the
/// larger axis gap, not the hypotenuse.
#[test]
fn extrude_field_is_chebyshev_exact_on_a_prism() {
    use substrate::geom2d::Metric;
    // A 4x4 footprint on Z, extruded 2 — the solid spans [0,4]x[0,4]x[0,2].
    let solid = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 4, 4), 2);
    assert_eq!(solid.field_metric(), Metric::Chebyshev);
    // Diagonally off the (4,4) corner by (3,3): Chebyshev reads 3, not 3*sqrt(2).
    let corner = solid.signed_distance([7.0, 7.0, 1.0]);
    assert!((corner - 3.0).abs() < 1e-4, "corner distance {corner}");
    // Straight out one face by 2.
    let face = solid.signed_distance([6.0, 2.0, 1.0]);
    assert!((face - 2.0).abs() < 1e-4, "face distance {face}");
    // Deepest interior point is 1 from the nearest face (the normal slab is thinnest).
    let centre = solid.signed_distance([2.0, 2.0, 1.0]);
    assert!((centre + 1.0).abs() < 1e-4, "centre distance {centre}");
    // Revolve reports Euclidean instead — the lift decides the metric, not the profile.
    let revolved =
        SketchSolid::revolve(Sketch::rectangle(PlaneAxis::Z, 4, 4), RevolveAxis::InPlane0, 360);
    assert_eq!(revolved.field_metric(), Metric::Euclidean);
    // A degenerate producer is empty, so every point is outside it.
    let degenerate = SketchSolid::extrude(Sketch::rectangle(PlaneAxis::Z, 4, 4), 0);
    assert_eq!(degenerate.signed_distance([1.0, 1.0, 1.0]), f32::INFINITY);
}

/// Revolve cases covering both axis reinterpretations, a full turn and partial turns
/// either side of the half-turn split (where the wedge flips from an intersection of two
/// half-planes to a union), and a profile that straddles the axis so the mirrored query
/// actually matters.
fn revolve_field_cases() -> Vec<(&'static str, SketchSolid)> {
    let lathe = vec![
        SketchPoint::new(0, 2), SketchPoint::new(6, 2),
        SketchPoint::new(6, 5), SketchPoint::new(0, 5),
    ];
    let straddling = vec![
        SketchPoint::new(0, -4), SketchPoint::new(5, -4),
        SketchPoint::new(5, 4), SketchPoint::new(0, 4),
    ];
    vec![
        ("full/InPlane0", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane0, 360)),
        ("full/InPlane1", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Y, lathe.clone()), RevolveAxis::InPlane1, 360)),
        ("quarter", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane0, 90)),
        ("half", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Z, lathe.clone()), RevolveAxis::InPlane0, 180)),
        ("three-quarter", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Z, lathe), RevolveAxis::InPlane0, 270)),
        ("straddling", SketchSolid::revolve(
            Sketch::new(PlaneAxis::Z, straddling), RevolveAxis::InPlane0, 360)),
    ]
}

/// The revolve field must agree with the revolve resolve over every voxel, exactly as the
/// extrude one does. Occupancy is read from the sign bit for the same reason.
#[test]
fn revolve_signed_distance_agrees_with_the_resolve() {
    const DENSITY: u32 = 8;
    for (label, solid) in revolve_field_cases() {
        let mut grid = VoxelGrid::default();
        solid.resolve(&mut grid, DENSITY);
        let occupied: BTreeSet<[i32; 3]> =
            grid.occupied.iter().map(|voxel| voxel.local_index).collect();
        let dimensions = solid.grid_dimensions();
        let mut inside = 0u32;
        for x in 0..dimensions[0] {
            for y in 0..dimensions[1] {
                for z in 0..dimensions[2] {
                    let centre = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
                    let distance = solid.signed_distance(centre);
                    let field_says_solid = distance.is_sign_negative();
                    let resolve_says_solid =
                        occupied.contains(&[x as i32, y as i32, z as i32]);
                    assert_eq!(
                        field_says_solid, resolve_says_solid,
                        "{label} at {centre:?}: field distance {distance} says \
                         solid={field_says_solid}, resolve says {resolve_says_solid}"
                    );
                    inside += u32::from(field_says_solid);
                }
            }
        }
        assert!(inside > 0, "{label}: nothing solid, the case proves nothing");
    }
}

/// Revolve must be 1-Lipschitz in Euclidean — the property its cell bound will rest on.
/// Holds for the partial turns too: the wedge clip is a `max` of half-plane fields, each
/// unit-gradient, and `max` preserves the constant even though it stops being an exact
/// distance near the seam.
#[test]
fn revolve_signed_distance_is_one_lipschitz_in_euclidean() {
    for (label, solid) in revolve_field_cases() {
        let dimensions = solid.grid_dimensions();
        let step = 0.25f32;
        let mut worst: f32 = 0.0;
        for xi in -6..(dimensions[0] as f32 / step) as i32 + 6 {
            for yi in -6..(dimensions[1] as f32 / step) as i32 + 6 {
                for zi in -6..(dimensions[2] as f32 / step) as i32 + 6 {
                    let p = [xi as f32 * step, yi as f32 * step, zi as f32 * step];
                    let here = solid.signed_distance(p);
                    if !here.is_finite() {
                        continue;
                    }
                    for axis in 0..3 {
                        let mut q = p;
                        q[axis] += step;
                        let there = solid.signed_distance(q);
                        worst = worst.max((there - here).abs() / step);
                    }
                }
            }
        }
        assert!(
            worst <= 1.0 + 1e-4,
            "{label}: revolve field is not 1-Lipschitz in Euclidean (worst {worst})"
        );
    }
}

/// The 135 degree closing edge must be INCLUSIVE - the seam the f64 to f32 narrowing
/// repaired.
///
/// Occupancy is `field <= SURFACE_ISOLEVEL`, so a sample lying exactly ON the closing
/// edge of the swept wedge is inside. At `turn = 135` that edge runs along the
/// anti-diagonal: `cos(135) = -sin(135)`, so the wedge term `cos*b - sin*a` collapses
/// to `-k*(a + b)`, which is EXACTLY zero wherever `a = -b`. Centred radial coordinates
/// are half-integers on an even-dimensioned grid, so a whole diagonal line of lattice
/// sites lands precisely there - this is not a measure-zero curiosity, it is a visible
/// seam of voxels.
///
/// In f64 the culprit is that `135.0_f64.to_radians()` is not exactly `3*pi/4`, so
/// `cos` and `sin` come back one ulp off being exact negatives (their sum is
/// `1.11e-16`, not `0`). The cancellation therefore leaves a few ulps of residue whose
/// SIGN varies along the diagonal, and every site that lands positive is dropped. In
/// f32 the two round to exact negatives of each other, their sum is `0.0`, and the term
/// is exactly zero - the true value - so the whole diagonal is kept.
///
/// Measured over the revolve matrix (485,447,064 samples; turns 1-360, three profile
/// scales, profiles clear of and straddling the axis, plus extrude controls), turn=135
/// was the ONLY turn where the two widths disagreed at all, and every one of the 3,639
/// disagreements was a voxel f32 GAINED. Widening this path again re-opens the seam.
#[test]
fn revolve_closing_edge_is_inclusive_at_135_degrees() {
    // A lathe profile clear of the axis: axial 0..=6, radial 2..=8.
    let profile = vec![
        SketchPoint::new(0, 2), SketchPoint::new(6, 2),
        SketchPoint::new(6, 8), SketchPoint::new(0, 8),
    ];
    let solid = SketchSolid::revolve(
        Sketch::new(PlaneAxis::Z, profile),
        RevolveAxis::InPlane0,
        135,
    );
    let dimensions = solid.grid_dimensions();
    // Revolve about in-plane axis 0 (X) puts the radial axes at Y and Z, ascending.
    let (radial_a, radial_b) = (1usize, 2usize);
    let half_a = dimensions[radial_a] as f32 / 2.0;
    let half_b = dimensions[radial_b] as f32 / 2.0;

    // Walk the anti-diagonal `centred_a = -centred_b`. Sample centres are `idx + 0.5`,
    // so centred coords are half-integers and the closing edge passes exactly through
    // them. Keep only radii inside the profile band [2, 8].
    //
    // The 135 degree ray points UP-LEFT: theta is measured from `+radial_a` toward
    // `+radial_b`, so the closing edge is the `(-a, +b)` diagonal. The other diagonal
    // is rejected by the FIRST edge (`-b <= 0`) and says nothing about this seam.
    let mut tested = 0;
    for step in 0..8i32 {
        let centred_b = step as f32 + 0.5;
        let centred_a = -centred_b;
        let radius = (centred_a * centred_a + centred_b * centred_b).sqrt();
        if !(2.0..=8.0).contains(&radius) {
            continue;
        }
        let mut point = [0.0f32; 3];
        point[0] = 3.5; // mid-axial, comfortably inside the profile 0..6 span
        point[radial_a] = centred_a + half_a;
        point[radial_b] = centred_b + half_b;
        let field = solid.signed_distance(point);
        assert!(
            field <= voxel_core::voxel::SURFACE_ISOLEVEL,
            "sample on the 135 degree closing edge at {point:?} (centred {centred_a}, \
             {centred_b}, radius {radius}) reads field {field} - the closing edge must \
             be INCLUSIVE. A positive few-ulp value here means the wedge term stopped \
             cancelling exactly, i.e. this path was widened back to f64."
        );
        tested += 1;
    }
    assert!(tested > 0, "the diagonal walk tested nothing - the fixture drifted");
}
