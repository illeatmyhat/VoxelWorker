//! Whole-circle entities: a closed curve is its own loop (ADR 0035 Decision 7).

use super::*;
use substrate::geom2d::point_in_region;

/// A sketch holding one radius-`r` circle about `(cx, cy)`, and nothing else.
fn lone_circle(cx: i64, cy: i64, radius: i64) -> Sketch {
    Sketch::circle(PlaneAxis::Z, SketchPoint::new(cx, cy), radius)
}

/// The headline: a circle bounds a region with no help from anything else. An arc has to meet
/// other geometry before it encloses anything; a closed curve does not, which is the whole reason
/// it is its own entity rather than a 360-degree bulge.
#[test]
fn a_circle_bounds_a_face_on_its_own() {
    let sketch = lone_circle(0, 0, 4);
    let faces = sketch.faces();
    assert_eq!(faces.len(), 1, "one closed curve, one face");
    let expected = std::f64::consts::PI * 16.0;
    assert!(
        (faces[0].area_voxels - expected).abs() < 1e-9,
        "the area is the disc's, exactly — got {}, want {expected}",
        faces[0].area_voxels
    );
}

/// The face's boundary is ONE edge that closes on itself, and the document holds no point on the
/// curve — only the centre. A seam is not a vertex: nothing can select it, and moving the circle
/// takes it along with no trace.
#[test]
fn the_boundary_is_one_closed_edge_with_no_on_curve_vertex() {
    let sketch = lone_circle(3, 5, 2);
    let faces = sketch.faces();
    assert_eq!(faces[0].boundary.len(), 1);
    assert!(faces[0].boundary[0].is_closed());
    assert_eq!(
        sketch.points().len(),
        1,
        "the centre, and nothing on the curve"
    );
    assert_eq!(sketch.points()[0].role, EntityRole::Construction);
    let centre = sketch.points()[0].at.in_plane();
    assert_eq!(centre, [3.0, 5.0]);
}

/// The region classifies against the true circle, so a sample just inside the rim is solid and one
/// just outside is not — no polygon's corner-cutting in between.
#[test]
fn the_region_classifies_against_the_curve() {
    let sketch = lone_circle(0, 0, 8);
    let region = sketch.region_field_loops();
    assert!(point_in_region(&region, [0.0, 0.0]), "the centre is solid");
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
    assert_eq!(sketch.faces().len(), 2, "the square and the disc");

    let solid_everywhere = sketch.region_field_loops();
    assert!(
        point_in_region(&solid_everywhere, [10.0, 10.0]),
        "both picked: the middle is solid"
    );

    let disc = sketch
        .faces()
        .into_iter()
        .min_by(|a, b| a.area_voxels.total_cmp(&b.area_voxels))
        .expect("a face")
        .key;
    sketch.set_face_picked(disc, false);
    let donut = sketch.region_field_loops();
    assert!(!point_in_region(&donut, [10.0, 10.0]), "the hole is carved");
    assert!(!point_in_region(&donut, [13.0, 10.0]), "still in the hole");
    assert!(point_in_region(&donut, [16.0, 10.0]), "the ring is solid");
    assert!(point_in_region(&donut, [1.0, 1.0]), "so is the corner");
}

/// Two concentric circles about ONE point are two faces — the ring, without a square around it.
/// The second one is not a duplicate: same centre, different curve.
#[test]
fn concentric_circles_are_two_faces() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let centre = sketch.add_free_point(SketchPoint::new(0, 0));
    sketch
        .circle_about(centre, SketchLength::new(9))
        .expect("the outer circle");
    sketch
        .circle_about(centre, SketchLength::new(3))
        .expect("the inner one");
    assert_eq!(sketch.faces().len(), 2);
    assert!(
        sketch.circle_about(centre, SketchLength::new(3)).is_none(),
        "the same curve twice is not two curves"
    );
}

/// A circle IS its centre plus a radius, so deleting the centre deletes the circle — and deleting
/// the circle takes its minted centre with it, since nothing else was ever named there.
#[test]
fn the_centre_and_the_circle_live_and_die_together() {
    let mut sketch = lone_circle(0, 0, 4);
    let centre = sketch.points()[0].id;
    sketch.delete_point_cascade(centre);
    assert!(sketch.circles().is_empty());
    assert!(sketch.faces().is_empty());

    let mut sketch = lone_circle(0, 0, 4);
    let circle = sketch.circles()[0].id;
    sketch.delete_circle(circle);
    assert!(
        sketch.points().is_empty(),
        "the minted centre goes with the curve it anchored"
    );
}

/// A centre the author has since drawn to is referenced geometry and survives the circle.
#[test]
fn a_centre_something_else_names_survives_the_circle() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let centre = sketch.add_free_point(SketchPoint::new(0, 0));
    let far = sketch.add_free_point(SketchPoint::new(10, 0));
    let circle = sketch
        .circle_about(centre, SketchLength::new(4))
        .expect("a circle");
    sketch.connect(centre, far).expect("a spoke");
    sketch.delete_circle(circle);
    assert_eq!(sketch.points().len(), 2, "the spoke still needs its ends");
}

/// A radius is a length like any other: an authored `1 block` stays one block across a density
/// re-target, where a plain voxel radius simply rescales.
#[test]
fn an_authored_radius_survives_a_density_retarget() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let centre = sketch.add_free_point(SketchPoint::new(0, 0));
    let one_block = Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch
        .circle_about(
            centre,
            SketchLength {
                voxels: 16,
                local_voxels: 0.0,
                measurement: Some(one_block),
            },
        )
        .expect("a circle");
    sketch.retarget_density(16, 32);
    assert_eq!(
        sketch.circles()[0].radius.voxels,
        32,
        "one block is 32 voxels at d32"
    );

    let mut plain = lone_circle(0, 0, 4);
    plain.retarget_density(16, 32);
    assert_eq!(plain.circles()[0].radius.voxels, 8, "a plain radius scales");
}

/// Load repair erases a circle with no centre or no radius, and reports it — the same policy every
/// other entity gets (ADR 0030 load policy: erase the invalid, never fail the load).
#[test]
fn repair_erases_a_circle_that_cannot_be_drawn() {
    let mut sketch = lone_circle(0, 0, 4);
    let centre = sketch.circles()[0].center;
    sketch.circles_mut_for_test().push(Circle {
        id: 900,
        center: 901,
        radius: SketchLength::new(3),
        origin: 900,
        role: EntityRole::Real,
    });
    sketch.circles_mut_for_test().push(Circle {
        id: 902,
        center: centre,
        radius: SketchLength::new(0),
        origin: 902,
        role: EntityRole::Real,
    });
    assert_eq!(sketch.repair(), 2, "the dangling one and the zero one");
    assert_eq!(sketch.circles().len(), 1);
}

/// A zero or negative radius is refused at the door rather than stored and erased later.
#[test]
fn an_impossible_radius_is_refused() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let centre = sketch.add_free_point(SketchPoint::new(0, 0));
    assert!(sketch.circle_about(centre, SketchLength::new(0)).is_none());
    assert!(sketch.circle_about(centre, SketchLength::new(-2)).is_none());
    let circle = sketch
        .circle_about(centre, SketchLength::new(4))
        .expect("a real one");
    assert!(!sketch.set_circle_radius(circle, SketchLength::new(0)));
    assert_eq!(
        sketch.circles()[0].radius.value(),
        4.0,
        "the refused resize left the curve alone"
    );
    assert!(sketch.set_circle_radius(circle, SketchLength::new(7)));
    assert_eq!(sketch.circles()[0].radius.value(), 7.0);
}

/// The full turn stays out of the ARC form on purpose (ADR 0035 Decision 7): its chord is
/// zero-length, so there is no circle to recover from it. That is what `Circle` is for.
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
    let ring = sketch.flattened_loop();
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
        solid.grid_dimensions(),
        [12, 12, 3],
        "the disc's own 12x12 footprint"
    );
    assert_eq!(solid.profile_bbox_min(), [-6, -6]);
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
