//! Whole-circle entities: a closed curve is its own loop.

use super::ctx;
use super::*;
use ::parametric::EvaluationContext;
use std::num::NonZeroU32;
use substrate::geom2d::point_in_region;

/// A sketch holding one radius-`r` circle about `(cx, cy)`, and nothing else.
fn lone_circle(cx: i64, cy: i64, radius: i64) -> Sketch {
    Sketch::circle(PlaneAxis::Z, SketchPoint::new(cx, cy), radius)
}

#[test]
fn two_point_circle_uses_the_span_as_its_diameter() {
    let solid = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
    let placement = solid
        .two_point_circle_placement(SketchPoint::new(-3, 2), SketchPoint::new(5, 2))
        .unwrap();
    assert!(placement.center.coincides(&SketchPoint::new(1, 2)));
    assert!((placement.radius.value() - 4.0).abs() < 1e-12);
    let (made, circle) = solid
        .with_two_point_circle(SketchPoint::new(-3, 2), SketchPoint::new(5, 2))
        .unwrap();
    let stored = made
        .sketch
        .circles()
        .iter()
        .find(|item| item.id == circle)
        .unwrap();
    assert_eq!(
        stored.center,
        made.sketch.point_at(placement.center).unwrap()
    );
    assert!((stored.resolved_radius(ctx(16)) - placement.candidate.radius).abs() < 1e-12);
}

#[test]
fn three_point_circle_preview_and_commit_share_the_canonical_circumcircle() {
    let solid = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
    let points = [
        SketchPoint::from_continuous(2.25, 1.5),
        SketchPoint::from_continuous(-0.75, 4.125),
        SketchPoint::from_continuous(-2.0, -1.25),
    ];
    let placement = solid
        .three_point_circle_placement(points[0], points[1], points[2])
        .unwrap();
    let (made, circle) = solid
        .with_three_point_circle(points[0], points[1], points[2])
        .unwrap();
    let stored = made
        .sketch
        .circles()
        .iter()
        .find(|item| item.id == circle)
        .unwrap();
    assert!(made
        .sketch
        .points()
        .iter()
        .find(|point| point.id == stored.center)
        .unwrap()
        .at
        .coincides(&placement.center));
    assert!((stored.resolved_radius(ctx(16)) - placement.candidate.radius).abs() < 1e-12);
}

/// The headline: a circle bounds a region with no help from anything else. An arc has to meet
/// other geometry before it encloses anything; a closed curve does not, which is the whole reason
/// it is its own entity rather than a 360-degree bulge.
#[test]
fn a_circle_bounds_a_face_on_its_own() {
    let sketch = lone_circle(0, 0, 4);
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 1, "one closed curve, one face");
    let expected = std::f64::consts::PI * 16.0;
    assert!(
        (faces[0].area_voxels - expected).abs() < 1e-9,
        "the area is the disc's, exactly — got {}, want {expected}",
        faces[0].area_voxels
    );
}

/// The face's boundary is ONE edge that closes on itself, and the document holds no point on the
/// curve — only the center. A seam is not a vertex: nothing can select it, and moving the circle
/// takes it along with no trace.
#[test]
fn the_boundary_is_one_closed_edge_with_no_on_curve_vertex() {
    let sketch = lone_circle(3, 5, 2);
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces[0].boundary.len(), 1);
    assert!(faces[0].boundary[0].is_closed());
    assert_eq!(
        sketch.points().len(),
        1,
        "the center, and nothing on the curve"
    );
    assert_eq!(sketch.points()[0].lifetime, PointLifetime::CurveAnchored);
    let center = sketch.points()[0].at.in_plane();
    assert_eq!(center, [3.0, 5.0]);
}

/// The region classifies against the true circle, so a sample just inside the rim is solid and one
/// just outside is not — no polygon's corner-cutting in between.
#[test]
fn the_region_classifies_against_the_curve() {
    let sketch = lone_circle(0, 0, 8);
    let region = sketch.region_field_loops(ctx(16));
    assert!(point_in_region(&region, [0.0, 0.0]), "the center is solid");
    assert!(point_in_region(&region, [7.9, 0.0]), "just inside the rim");
    assert!(!point_in_region(&region, [8.1, 0.0]), "just outside it");
    // A diagonal sample a polygon of chords would get wrong: inside the circle, outside the
    // inscribed square.
    assert!(
        point_in_region(&region, [5.6, 5.6]),
        "inside on the diagonal"
    );
    assert!(!point_in_region(&region, [5.7, 5.7]), "outside on it");
}

/// The donut: a circle inside a square, the circle unpicked. The ring is solid, the hole is not.
/// This is the ordered fold doing its job — the smaller face gets its say first.
#[test]
fn an_unpicked_circle_inside_a_square_is_a_hole() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corners = [
        SketchPoint::new(0, 0),
        SketchPoint::new(20, 0),
        SketchPoint::new(20, 20),
        SketchPoint::new(0, 20),
    ]
    .map(|at| sketch.add_free_point(at));
    for index in 0..4 {
        sketch.connect(corners[index], corners[(index + 1) % 4]);
    }
    sketch.add_circle(SketchPoint::new(10, 10), SketchLength::new(5));
    assert_eq!(sketch.faces(ctx(16)).len(), 2, "the square and the disc");

    let solid_everywhere = sketch.region_field_loops(ctx(16));
    assert!(
        point_in_region(&solid_everywhere, [10.0, 10.0]),
        "both picked: the middle is solid"
    );

    let disc = sketch
        .identified_faces(ctx(16))
        .into_iter()
        .min_by(|a, b| a.0.area_voxels.total_cmp(&b.0.area_voxels))
        .expect("a face")
        .1;
    sketch.set_face_picked(disc, false, ctx(16));
    let donut = sketch.region_field_loops(ctx(16));
    assert!(!point_in_region(&donut, [10.0, 10.0]), "the hole is carved");
    assert!(!point_in_region(&donut, [13.0, 10.0]), "still in the hole");
    assert!(point_in_region(&donut, [16.0, 10.0]), "the ring is solid");
    assert!(point_in_region(&donut, [1.0, 1.0]), "so is the corner");
}

/// Two concentric circles about ONE point are two faces — the ring, without a square around it.
/// The second one is not a duplicate: same center, different curve.
#[test]
fn concentric_circles_are_two_faces() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let center = sketch.add_free_point(SketchPoint::new(0, 0));
    sketch
        .circle_about(center, SketchLength::new(9))
        .expect("the outer circle");
    sketch
        .circle_about(center, SketchLength::new(3))
        .expect("the inner one");
    assert_eq!(sketch.faces(ctx(16)).len(), 2);
    assert!(
        sketch.circle_about(center, SketchLength::new(3)).is_none(),
        "the same curve twice is not two curves"
    );
}

/// Two circles that cross bound THREE faces — two crescents and the lens between them. Nothing was
/// drawn at the crossings; the arrangement cut both curves there. Without that cut they would be
/// two overlapping discs bounding one region.
#[test]
fn two_overlapping_circles_bound_three_faces() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch.add_circle(SketchPoint::new(0, 0), SketchLength::new(10));
    sketch.add_circle(SketchPoint::new(12, 0), SketchLength::new(10));

    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 3, "two crescents and the lens");

    // Equal circles, r=10 at a center distance of 12: the lens is
    // 2r^2*acos(d/2r) - (d/2)*sqrt(4r^2 - d^2), and a crescent is a disc less the lens.
    let lens = 200.0 * (0.6f64).acos() - 6.0 * (400.0f64 - 144.0).sqrt();
    let crescent = std::f64::consts::PI * 100.0 - lens;
    let mut areas: Vec<f64> = faces.iter().map(|face| face.area_voxels).collect();
    areas.sort_by(f64::total_cmp);
    for (got, want) in areas.iter().zip([lens, crescent, crescent]) {
        assert!(
            (got - want).abs() < 1e-6,
            "face areas are {areas:?}, want the lens {lens} and two crescents {crescent}"
        );
    }

    // The crossings are at (6, +-8), so the lens spans x in 2..10 on the axis.
    let region = sketch.region_field_loops(ctx(16));
    for inside in [[-5.0, 0.0], [6.0, 0.0], [17.0, 0.0]] {
        assert!(point_in_region(&region, inside), "{inside:?} is solid");
    }

    // The lens is a face like any other, so unpicking it carves a slot neither circle drew — the
    // proof that the three faces are separately addressable and not one region reported thrice.
    let lens_face = sketch
        .identified_faces(ctx(16))
        .into_iter()
        .min_by(|a, b| a.0.area_voxels.total_cmp(&b.0.area_voxels))
        .expect("a face")
        .1;
    sketch.set_face_picked(lens_face, false, ctx(16));
    let carved = sketch.region_field_loops(ctx(16));
    assert!(!point_in_region(&carved, [6.0, 0.0]), "the lens is gone");
    assert!(point_in_region(&carved, [-5.0, 0.0]), "the left crescent");
    assert!(point_in_region(&carved, [17.0, 0.0]), "the right one");
}

/// THE CLOSED CASE, end to end. A line tangent to a circle crosses it once, so the arrangement
/// cuts the circle at exactly one parameter — and a single cut does not open a loop, it only moves
/// where the loop is written from. The piece that comes back is a FULL-TURN arc.
///
/// This is the sketch-level proof that the full turn is legal where the closed case actually lives.
/// The store refuses a 360° *bulge* because endpoints-plus-bulge has a pole there
/// (`the_full_turn_is_where_the_radius_diverges`); substrate's center-radius-sweep form has no pole
/// and carries the closed piece. Get the re-seaming wrong and the disc either splits in two or
/// disappears, so this test fails loudly in both directions.
#[test]
fn a_tangent_line_re_seams_the_circle_without_opening_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch.add_circle(SketchPoint::new(10, 10), SketchLength::new(5));
    let left = sketch.add_free_point(SketchPoint::new(0, 15));
    let right = sketch.add_free_point(SketchPoint::new(30, 15));
    sketch.connect(left, right).expect("the tangent line");

    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 1, "the disc, touched but not cut open");
    let expected = std::f64::consts::PI * 25.0;
    assert!(
        (faces[0].area_voxels - expected).abs() < 1e-9,
        "the whole disc survives the tangency — got {}, want {expected}",
        faces[0].area_voxels
    );
    assert_eq!(faces[0].boundary.len(), 1, "still one edge");
    assert!(faces[0].boundary[0].is_closed(), "still closed");

    let region = sketch.region_field_loops(ctx(16));
    assert!(
        point_in_region(&region, [10.0, 10.0]),
        "the center is solid"
    );
    assert!(
        !point_in_region(&region, [10.0, 16.0]),
        "above the tangency"
    );

    // The secant case is the contrast: two crossings DO open the circle, into two arcs bounding
    // two faces. One cut and two cuts must not behave the same way.
    let mut secant = Sketch::empty(PlaneAxis::Z);
    secant.add_circle(SketchPoint::new(10, 10), SketchLength::new(5));
    let a = secant.add_free_point(SketchPoint::new(0, 10));
    let b = secant.add_free_point(SketchPoint::new(30, 10));
    secant.connect(a, b).expect("the secant");
    assert_eq!(
        secant.faces(ctx(16)).len(),
        2,
        "cut clean through: two halves"
    );
}

/// **The relaxation itself.** The SAME geometry — a full turn about `(2, -3)` at radius 4 — is
/// legal in the form the profile uses and refused in the form the store uses, on purpose.
///
/// `ProfileEdge::circle` is a `sweep_radians: TAU` arc whose chord is zero length. Everything that
/// consumes it reads the solved circle rather than the endpoint-plus-bulge derivation:
/// `interior_points` walks the circle, `signed_area_term` integrates the real sweep, `measured`
/// hands substrate a center and a sweep. None of those has a full-turn guard, and this test fails if
/// one is ever put back.
///
/// `arc_sweep_is_valid` guards a different form on a different path — authoring an `Arc` ENTITY from
/// two endpoints — where the full turn is a pole rather than a policy
/// (`the_full_turn_is_where_the_radius_diverges`).
#[test]
fn a_full_turn_profile_edge_is_the_relaxed_closed_case() {
    let (center, radius) = ([2.0, -3.0], 4.0);
    let edge = ProfileEdge::circle(center, radius);
    assert!(edge.is_closed(), "a loop with no vertex");
    assert_eq!(
        edge.from.in_plane(),
        edge.to.in_plane(),
        "the chord is zero length — the thing the store's form cannot survive"
    );

    // Exact, by Green's theorem over the real sweep rather than over a fan of chords.
    let expected = std::f64::consts::PI * radius * radius;
    assert!(
        (edge.signed_area_term() - expected).abs() < 1e-9,
        "a full turn encloses its disc — got {}, want {expected}",
        edge.signed_area_term()
    );

    // The tessellation walks the whole circle, not a chord's worth of it.
    let interior = edge.interior_points(ARC_SAGITTA_TOLERANCE_VOXELS);
    assert!(
        interior.len() > 8,
        "a full turn needs a fan: {}",
        interior.len()
    );
    for point in &interior {
        let at = point.in_plane();
        let distance = ((at[0] - center[0]).powi(2) + (at[1] - center[1]).powi(2)).sqrt();
        assert!((distance - radius).abs() < 1e-5, "off the circle: {at:?}");
    }
    let bearings: Vec<f64> = interior
        .iter()
        .map(|point| {
            let at = point.in_plane();
            (at[1] - center[1])
                .atan2(at[0] - center[0])
                .rem_euclid(std::f64::consts::TAU)
        })
        .collect();
    for quadrant in 0..4 {
        let low = quadrant as f64 * std::f64::consts::FRAC_PI_2;
        assert!(
            bearings
                .iter()
                .any(|b| (low..low + std::f64::consts::FRAC_PI_2).contains(b)),
            "the fan reaches every quadrant, so the walk is a full turn"
        );
    }

    // The same geometry offered to the store's endpoint-plus-bulge form: no answer.
    let seam = edge.from.in_plane();
    assert_eq!(
        arc_center_radius(seam, seam, 360.0),
        None,
        "the store's form has a pole here; the profile's form does not"
    );
}

/// A circle IS its center plus a radius, so deleting the center deletes the circle — and deleting
/// the circle takes its minted center with it, since nothing else was ever named there.
#[test]
fn the_center_and_the_circle_live_and_die_together() {
    let mut sketch = lone_circle(0, 0, 4);
    let center = sketch.points()[0].id;
    sketch.delete_point_cascade(center);
    assert!(sketch.circles().is_empty());
    assert!(sketch.faces(ctx(16)).is_empty());

    let mut sketch = lone_circle(0, 0, 4);
    let circle = sketch.circles()[0].id;
    sketch.delete_circle(circle);
    assert!(
        sketch.points().is_empty(),
        "the minted center goes with the curve it anchored"
    );
}

#[test]
fn deleting_a_circle_cascades_constraints_that_named_its_center() {
    let mut sketch = lone_circle(0, 0, 4);
    let center = sketch.points()[0].id;
    let circle = sketch.circles()[0].id;
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: center,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("the center is live geometry while its circle exists");

    sketch.delete_circle(circle);

    assert!(sketch.constraints().is_empty());
}

/// A center the author has since drawn to is referenced geometry and survives the circle.
#[test]
fn a_center_something_else_names_survives_the_circle() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let center = sketch.add_free_point(SketchPoint::new(0, 0));
    let far = sketch.add_free_point(SketchPoint::new(10, 0));
    let circle = sketch
        .circle_about(center, SketchLength::new(4))
        .expect("a circle");
    sketch.connect(center, far).expect("a spoke");
    sketch.delete_circle(circle);
    assert_eq!(sketch.points().len(), 2, "the spoke still needs its ends");
}

/// A radius is a length like any other: an authored `1 block` stays one block across a density
/// re-target, where a plain voxel radius simply rescales.
#[test]
fn an_authored_radius_survives_a_density_retarget() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let center = sketch.add_free_point(SketchPoint::new(0, 0));
    let one_block = Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch
        .circle_about(
            center,
            SketchLength {
                voxels: 16,
                local_voxels: 0.0,
                measurement: Some(one_block),
            },
        )
        .expect("a circle");
    sketch.retarget_density(16, 32);
    let at_32 = EvaluationContext::new(NonZeroU32::new(32).expect("non-zero"));
    assert_eq!(
        sketch.circles()[0].resolved_radius(at_32),
        32.0,
        "one block is 32 voxels at d32"
    );

    let mut plain = lone_circle(0, 0, 4);
    plain.retarget_density(16, 32);
    assert_eq!(
        plain.circles()[0].free_radius_value(),
        Some(8.0),
        "a plain radius scales"
    );
}

#[test]
fn free_radius_retarget_preserves_the_exact_ratio() {
    let mut sketch = lone_circle(0, 0, 1);
    sketch.circles_mut_for_test()[0].radius = CircleRadius::free(ResolvedLength::from_rational(
        ::parametric::ExactRational::new(1, 3).expect("non-zero denominator"),
    ));

    sketch.retarget_density(16, 24);

    assert_eq!(
        sketch.circles()[0]
            .radius
            .free_value()
            .expect("the free authority remains free")
            .rational(),
        ::parametric::ExactRational::new(1, 2).expect("non-zero denominator"),
        "the free radius is scaled as an exact 24/16 ratio"
    );
}

#[test]
fn legacy_radius_json_migrates_to_a_fixed_source_without_a_voxel_cache() {
    let mut sketch = lone_circle(0, 0, 4);
    let source = Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    let legacy = SketchLength {
        voxels: 16,
        local_voxels: 0.0,
        measurement: Some(source),
    };
    let mut json = serde_json::to_value(&sketch).expect("serialize");
    json["circles"][0]["radius"] = serde_json::to_value(legacy).expect("legacy shape");
    sketch = serde_json::from_value(json).expect("legacy document loads");

    let radius = &sketch.circles()[0].radius;
    assert_eq!(radius.fixed_source(), Some(&source));
    assert_eq!(
        sketch.circles()[0].resolved_radius(EvaluationContext::new(
            NonZeroU32::new(32).expect("non-zero"),
        )),
        32.0
    );

    let saved = serde_json::to_value(&sketch).expect("new schema serializes");
    assert!(saved["circles"][0]["radius"].get("fixed").is_some());
    assert!(saved["circles"][0]["radius"].get("voxels").is_none());

    let context = EvaluationContext::new(NonZeroU32::new(32).expect("non-zero"));
    assert_eq!(sketch.degrees_of_freedom(context), Ok(2));
}

#[test]
fn legacy_circle_radius_fixtures_preserve_free_and_fixed_authority() {
    let free_fixture = r#"
        {
          "plane": "Z",
          "points": [{ "id": 0, "at": { "offset_voxels": [0, 0] } }],
          "segments": [], "arcs": [],
          "circles": [{
            "id": 1, "center": 0,
            "radius": { "local_voxels": 0.5, "voxels": 3 },
            "origin": 1
          }],
          "unpicked_points": [], "constraints": [], "next_id": 2
        }
    "#;
    let free: Sketch = serde_json::from_str(free_fixture).expect("legacy free radius loads");
    assert_eq!(free.circles()[0].free_radius_value(), Some(3.5));

    let fixed_fixture = r#"
        {
          "plane": "Z",
          "points": [{ "id": 0, "at": { "offset_voxels": [0, 0] } }],
          "segments": [], "arcs": [],
          "circles": [{
            "id": 1, "center": 0,
            "radius": {
              "measurement": {
                "block_term_numerator": 1,
                "block_term_denominator": 1,
                "voxel_term": 0
              },
              "voxels": 16
            },
            "origin": 1
          }],
          "unpicked_points": [], "constraints": [], "next_id": 2
        }
    "#;
    let fixed: Sketch = serde_json::from_str(fixed_fixture).expect("legacy fixed radius loads");
    assert!(fixed.circles()[0].radius.free_value().is_none());
    assert_eq!(fixed.circles()[0].resolved_radius(ctx(32)), 32.0);
}

#[test]
fn malformed_curve_denominator_fixtures_are_rejected_at_deserialization() {
    let bad_free = r#"
        { "plane": "Z", "points": [{ "id": 0, "at": { "offset_voxels": [0, 0] } }],
          "segments": [], "arcs": [],
          "circles": [{ "id": 1, "center": 0,
            "radius": { "free": { "numerator": 1, "denominator": 0 } }, "origin": 1 }],
          "unpicked_points": [], "constraints": [], "next_id": 2 }
    "#;
    let bad_fixed = r#"
        { "plane": "Z", "points": [{ "id": 0, "at": { "offset_voxels": [0, 0] } }],
          "segments": [], "arcs": [],
          "circles": [{ "id": 1, "center": 0,
            "radius": { "fixed": {
              "block_term_numerator": 1, "block_term_denominator": 0, "voxel_term": 0
            } }, "origin": 1 }],
          "unpicked_points": [], "constraints": [], "next_id": 2 }
    "#;

    assert!(serde_json::from_str::<Sketch>(bad_free).is_err());
    assert!(serde_json::from_str::<Sketch>(bad_fixed).is_err());
}

#[test]
fn current_free_and_fixed_circle_authorities_round_trip() {
    let free = lone_circle(0, 0, 4);
    let free_json = serde_json::to_string(&free).expect("serialize free radius");
    let restored_free: Sketch = serde_json::from_str(&free_json).expect("deserialize free radius");
    assert_eq!(restored_free.circles()[0].free_radius_value(), Some(4.0));

    let mut fixed = lone_circle(0, 0, 4);
    fixed.circles_mut_for_test()[0].radius = CircleRadius::fixed(Measurement::new(
        ::parametric::ExactRational::from_integer(1),
        0,
    ));
    let fixed_json = serde_json::to_string(&fixed).expect("serialize fixed radius");
    let restored_fixed: Sketch =
        serde_json::from_str(&fixed_json).expect("deserialize fixed radius");
    assert!(restored_fixed.circles()[0].radius.fixed_source().is_some());
    assert_eq!(restored_fixed.circles()[0].resolved_radius(ctx(16)), 16.0);
}

#[test]
fn fixed_radius_context_drives_region_bounds_field_and_resolve_together() {
    let mut sketch = lone_circle(0, 0, 1);
    let source = Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch.circles_mut_for_test()[0].radius = CircleRadius::fixed(source);
    let d16 = ctx(16);
    let d32 = ctx(32);

    let derived_at_16 = sketch.derived(d16);
    let derived_at_32 = sketch.derived(d32);
    assert!(
        !std::sync::Arc::ptr_eq(&derived_at_16, &derived_at_32),
        "density is part of the resolved-region memo key"
    );

    assert_eq!(sketch.circles()[0].resolved_radius(d16), 16.0);
    assert_eq!(sketch.circles()[0].resolved_radius(d32), 32.0);
    assert_eq!(
        sketch.faces(d16)[0].area_voxels,
        std::f64::consts::PI * 256.0
    );
    assert_eq!(
        sketch.faces(d32)[0].area_voxels,
        std::f64::consts::PI * 1024.0
    );
    assert!(
        substrate::geom2d::point_in_region(&sketch.region_field_loops(d32), [31.5, 0.0]),
        "the field borrows the d32-resolved curve"
    );

    let solid = SketchSolid::extrude(sketch, 1);
    assert_eq!(solid.grid_dimensions(d16), [32, 32, 1]);
    assert_eq!(solid.grid_dimensions(d32), [64, 64, 1]);
    assert!(
        occupancy_set(&solid, 32).len() > occupancy_set(&solid, 16).len(),
        "resolve uses the same d32 radius as the region and bounds"
    );
}

/// Load repair erases a circle with no center or no radius, and reports it — the same policy every
/// other entity gets: erase the invalid, never fail the load.
#[test]
fn repair_erases_a_circle_that_cannot_be_drawn() {
    let mut sketch = lone_circle(0, 0, 4);
    let center = sketch.circles()[0].center;
    sketch.circles_mut_for_test().push(Circle {
        id: 900,
        center: 901,
        radius: CircleRadius::free(ResolvedLength::from_voxels(3)),
        origin: 900,
        role: EntityRole::Real,
    });
    sketch.circles_mut_for_test().push(Circle {
        id: 902,
        center,
        radius: CircleRadius::free(ResolvedLength::from_voxels(0)),
        origin: 902,
        role: EntityRole::Real,
    });
    assert_eq!(
        sketch.repair(ctx(16)),
        2,
        "the dangling one and the zero one"
    );
    assert_eq!(sketch.circles().len(), 1);
}

#[test]
fn repair_removes_a_fixed_circle_that_resolves_invalid_at_its_context() {
    let mut sketch = lone_circle(0, 0, 4);
    let center = sketch.circles()[0].center;
    sketch.circles_mut_for_test().push(Circle {
        id: 900,
        center,
        radius: CircleRadius::fixed(Measurement::from_voxels(0)),
        origin: 900,
        role: EntityRole::Real,
    });

    assert_eq!(sketch.repair(ctx(16)), 1);
    assert_eq!(sketch.circles().len(), 1);
}

#[test]
fn current_schema_rejects_a_nonpositive_free_circle_radius() {
    let sketch = lone_circle(0, 0, 4);
    let mut json = serde_json::to_value(sketch).expect("serialize");
    json["circles"][0]["radius"] = serde_json::json!({
        "free": { "numerator": 0, "denominator": 1 }
    });
    assert!(
        serde_json::from_value::<Sketch>(json).is_err(),
        "a nonpositive current-schema radius must not enter the document"
    );
}

#[test]
fn constructed_invalid_curve_state_reports_geometry_instead_of_panicking() {
    let mut sketch = lone_circle(0, 0, 4);
    let center = sketch.circles()[0].center;
    sketch.circles_mut_for_test().push(Circle {
        id: 900,
        center,
        radius: CircleRadius::free(ResolvedLength::from_voxels(0)),
        origin: 900,
        role: EntityRole::Real,
    });
    assert_eq!(
        sketch.solve(ctx(16)),
        Err(SketchEvaluationError::InvalidDocumentGeometry)
    );
}

/// A zero or negative radius is refused at the door rather than stored and erased later.
#[test]
fn an_impossible_radius_is_refused() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let center = sketch.add_free_point(SketchPoint::new(0, 0));
    assert!(sketch.circle_about(center, SketchLength::new(0)).is_none());
    assert!(sketch.circle_about(center, SketchLength::new(-2)).is_none());
    let circle = sketch
        .circle_about(center, SketchLength::new(4))
        .expect("a real one");
    assert!(!sketch.set_circle_radius(circle, SketchLength::new(0)));
    assert_eq!(
        sketch.circles()[0].free_radius_value(),
        Some(4.0),
        "the refused resize left the curve alone"
    );
    assert!(sketch.set_circle_radius(circle, SketchLength::new(7)));
    assert_eq!(sketch.circles()[0].free_radius_value(), Some(7.0));
}

/// The center-diameter tool keeps its perimeter sample transient: only a center and radius enter
/// the document, and a coincident second click is not a history-worthy edit.
#[test]
fn center_diameter_reuses_or_mints_only_its_center() {
    let empty = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
    let made = empty.with_circle_center_diameter(SketchPoint::new(2, 3), SketchPoint::new(6, 3));
    assert_eq!(
        made.sketch.points().len(),
        1,
        "the perimeter is not a point entity"
    );
    assert_eq!(
        made.sketch.points()[0].lifetime,
        PointLifetime::CurveAnchored
    );
    assert_eq!(made.sketch.circles()[0].free_radius_value(), Some(4.0));
    let circle = made.sketch.circles()[0].id;
    let removed = made.with_circle_deleted(circle);
    assert!(removed.sketch.circles().is_empty());
    assert!(
        removed.sketch.points().is_empty(),
        "deleting a selected circle prunes only its orphan construction center"
    );

    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let center = sketch.add_free_point(SketchPoint::new(2, 3));
    let reused = SketchSolid::extrude(sketch, 3)
        .with_circle_center_diameter(SketchPoint::new(2, 3), SketchPoint::new(2, 8));
    assert_eq!(reused.sketch.points().len(), 1, "the real center is reused");
    assert_eq!(reused.sketch.circles()[0].center, center);

    let unchanged =
        reused.with_circle_center_diameter(SketchPoint::new(2, 3), SketchPoint::new(2, 3));
    assert_eq!(unchanged, reused, "a zero-radius gesture is a pure no-op");
}

/// The full turn stays out of the ARC form on purpose: its chord is zero-length, so there is no
/// circle to recover from it. That is what `Circle` is for.
#[test]
fn a_full_turn_is_not_an_arc_bulge() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    assert!(sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(360))
        .is_none());
    assert!(sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(120))
        .is_some());
}

/// Flattening a circle closes its ring, and every chord stays within tolerance of the curve. The
/// seam is the only place the loop's first and last points meet, and it is not repeated.
#[test]
fn flattening_a_circle_closes_the_ring_within_tolerance() {
    let sketch = lone_circle(0, 0, 8);
    let ring = sketch.flattened_loop(ctx(16));
    assert!(ring.len() >= 8, "a real ring, got {}", ring.len());
    for point in &ring {
        let [x, y] = point.in_plane();
        assert!(
            (x.hypot(y) - 8.0).abs() < 1e-6,
            "every vertex sits on the circle"
        );
    }
    // Each chord's midpoint is within the sagitta tolerance of the curve — including the closing
    // chord from the last point back to the seam, which is what "closed" means here.
    for index in 0..ring.len() {
        let start = ring[index].in_plane();
        let end = ring[(index + 1) % ring.len()].in_plane();
        let mid = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0];
        let sagitta = 8.0 - mid[0].hypot(mid[1]);
        assert!(
            sagitta <= ARC_SAGITTA_TOLERANCE_VOXELS + 1e-9,
            "chord {index} deviates {sagitta}"
        );
    }
}

/// The extent reaches the rim on every side, not the seam's chord — a producer sized off chords
/// would clip the curve it was asked to build.
#[test]
fn the_extent_is_the_circle_not_a_chord() {
    let solid = SketchSolid::extrude(lone_circle(0, 0, 6), 3);
    assert_eq!(
        solid.grid_dimensions(ctx(16)),
        [12, 12, 3],
        "the disc's own 12x12 footprint"
    );
    assert_eq!(solid.profile_bbox_min(ctx(16)), [-6, -6]);
}

/// The resolve produces a disc: the occupied count per layer is the circle's area to within the
/// half-voxel the corner-anchored sampling can differ by, and the shape is symmetric.
#[test]
fn extruding_a_circle_resolves_a_disc() {
    let solid = SketchSolid::extrude(lone_circle(0, 0, 6), 2);
    let mut grid = VoxelGrid::default();
    solid.resolve(&mut grid, 16);
    let per_layer = grid.occupied.len() as f64 / 2.0;
    let expected = std::f64::consts::PI * 36.0;
    assert!(
        (per_layer - expected).abs() < 0.06 * expected,
        "a disc of about {expected} voxels per layer, got {per_layer}"
    );
}
