//! #102 — the arc entity: the 3-point solve, the canonical endpoints+bulge store, chord
//! tessellation through the flattened loop, resolve through an arced profile, delete
//! cascade / repair, and serialization (including a pre-arc document loading clean).

use crate::sketch::{
    arc_center_radius, arc_interior_points, included_angle_through_degrees, PlaneAxis, Sketch,
    SketchPoint, SketchSolid, ARC_SAGITTA_TOLERANCE_VOXELS,
};
use crate::voxel::VoxelProducer;
use voxel_core::units::AngleMeasurement;
use voxel_core::voxel::VoxelGrid;

/// A closed profile: the `[0,4] × [0,3]` rectangle whose BOTTOM edge is replaced by a
/// half-turn arc bulging down to `axis1 = -2` (a half-disc of radius 2 under the box).
fn rounded_bottom_solid(height: u32) -> SketchSolid {
    let mut sketch = Sketch::new(
        PlaneAxis::Z,
        vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 3),
            SketchPoint::new(4, 3),
            SketchPoint::new(4, 0),
        ],
    );
    // The rectangle loop's bottom edge (4 → 0) makes way for the arc.
    let bottom = sketch
        .segments()
        .iter()
        .find(|seg| {
            let ids = [seg.from, seg.to];
            ids.contains(&0) && ids.contains(&3)
        })
        .expect("the wrap segment joins the first and last corners")
        .id;
    sketch.delete_segment(bottom);
    // From (0,0) to (4,0), +180° sweeps counter-clockwise about (2,0): down and around.
    sketch
        .connect_arc(0, 3, AngleMeasurement::from_degrees(180))
        .expect("a fresh arc over an unjoined pair");
    SketchSolid::extrude(sketch, height)
}

#[test]
fn three_point_solve_recovers_the_signed_sweep() {
    // Semicircle through the TOP: from (0,0) to (2,0) via (1,1) — centre (1,0), and the
    // top lies on the CLOCKWISE walk, so the sweep is -180°.
    let sweep = included_angle_through_degrees([0.0, 0.0], [2.0, 0.0], [1.0, 1.0])
        .expect("three non-collinear points");
    assert!((sweep + 180.0).abs() < 1e-9, "got {sweep}");
    // The mirrored through-point flips the sign.
    let mirrored = included_angle_through_degrees([0.0, 0.0], [2.0, 0.0], [1.0, -1.0])
        .expect("three non-collinear points");
    assert!((mirrored - 180.0).abs() < 1e-9, "got {mirrored}");
    // A quarter-ish minor arc: through close to the chord gives a small sweep.
    let minor = included_angle_through_degrees([0.0, 0.0], [2.0, 0.0], [1.0, -0.2])
        .expect("three non-collinear points");
    assert!(
        (0.0..180.0).contains(&minor),
        "a shallow bulge is a minor CCW arc, got {minor}"
    );
    assert_eq!(
        included_angle_through_degrees([0.0, 0.0], [2.0, 0.0], [1.0, 0.0]),
        None,
        "collinear points have no finite circle"
    );
}

#[test]
fn derived_center_and_radius_match_the_canonical_form() {
    let (center, radius) =
        arc_center_radius([0.0, 0.0], [4.0, 0.0], 180.0).expect("a valid half-turn");
    assert!((center[0] - 2.0).abs() < 1e-9 && center[1].abs() < 1e-9);
    assert!((radius - 2.0).abs() < 1e-9);
    assert_eq!(arc_center_radius([1.0, 1.0], [1.0, 1.0], 90.0), None);
    assert_eq!(arc_center_radius([0.0, 0.0], [4.0, 0.0], 0.0), None);
    assert_eq!(arc_center_radius([0.0, 0.0], [4.0, 0.0], 360.0), None);
}

#[test]
fn tessellation_stays_on_the_circle_within_the_sagitta_tolerance() {
    let interior = arc_interior_points([0.0, 0.0], [4.0, 0.0], 180.0);
    assert!(!interior.is_empty(), "a half-turn at r=2 needs chords");
    // Every tessellated vertex sits ON the circle (to f32-fraction precision), below the
    // chord (+180° from (0,0) to (4,0) sweeps through the bottom).
    let mut ring: Vec<[f64; 2]> = vec![[0.0, 0.0]];
    ring.extend(interior.iter().map(|point| point.in_plane()));
    ring.push([4.0, 0.0]);
    for point in &ring[1..ring.len() - 1] {
        let distance = ((point[0] - 2.0).powi(2) + point[1].powi(2)).sqrt();
        assert!((distance - 2.0).abs() < 1e-5, "off the circle: {point:?}");
        assert!(point[1] < 0.0, "the +180° fan bulges below the chord");
    }
    // Sagitta bound: each chord's midpoint deviates from the circle by at most the
    // versioned tolerance.
    for pair in ring.windows(2) {
        let mid = [
            (pair[0][0] + pair[1][0]) / 2.0,
            (pair[0][1] + pair[1][1]) / 2.0,
        ];
        let distance = ((mid[0] - 2.0).powi(2) + mid[1].powi(2)).sqrt();
        assert!(
            2.0 - distance <= ARC_SAGITTA_TOLERANCE_VOXELS + 1e-6,
            "sagitta over tolerance at {mid:?}"
        );
    }
}

#[test]
fn an_arc_closed_profile_extrudes_a_rounded_shape() {
    let solid = rounded_bottom_solid(2);
    let flattened = solid.sketch.flattened_loop();
    assert!(
        flattened.len() > 4,
        "the loop carries the arc's chord fan, got {}",
        flattened.len()
    );
    assert_eq!(
        solid.grid_dimensions(),
        [4, 5, 2],
        "in-plane cover 0..4 × -2..3 (the bulge extends the box), extrude 2"
    );
    let mut grid = VoxelGrid::default();
    solid.resolve(&mut grid, 8);
    // Per layer: the 4×3 rectangle plus the half-disc rows below it (4 cells at the
    // first row down, 2 at the second — cell centres against a radius-2 circle).
    assert_eq!(grid.occupied.len(), (12 + 6) * 2);
}

#[test]
fn arc_edges_join_the_region_graph() {
    // A closed loop of one segment and one arc via distinct point pairs is NOT possible
    // (two points, two edges is the rejected D-shape) — so close a triangle: two straight
    // edges and one arc.
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    let c = sketch.add_free_point(SketchPoint::new(2, 3));
    sketch.connect(a, c).expect("fresh edge");
    sketch.connect(c, b).expect("fresh edge");
    assert!(
        sketch.flattened_loop().is_empty(),
        "two edges of three: still open"
    );
    sketch
        .connect_arc(b, a, AngleMeasurement::from_degrees(120))
        .expect("the arc closes the loop");
    assert!(
        sketch.flattened_loop().len() > 3,
        "closed, with the arc tessellated"
    );
}

#[test]
fn connect_rejects_what_the_store_cannot_hold() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    let quarter = AngleMeasurement::from_degrees(90);
    assert_eq!(sketch.connect_arc(a, a, quarter), None, "self-loop");
    assert_eq!(sketch.connect_arc(a, 99, quarter), None, "unknown endpoint");
    assert_eq!(
        sketch.connect_arc(a, b, AngleMeasurement::from_degrees(0)),
        None,
        "zero bulge is a segment pretending"
    );
    assert_eq!(
        sketch.connect_arc(a, b, AngleMeasurement::from_degrees(360)),
        None,
        "a full turn has no chord-anchored shape"
    );
    sketch.connect_arc(a, b, quarter).expect("first edge lands");
    assert_eq!(
        sketch.connect_arc(b, a, quarter),
        None,
        "second arc over the pair (either direction) is the D-shape"
    );
    assert_eq!(
        sketch.connect(a, b),
        None,
        "a segment over an arced pair is the same D-shape"
    );
}

#[test]
fn delete_cascades_and_repair_cover_arcs() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    let arc = sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(90))
        .expect("fresh arc");

    // Deleting the arc alone leaves both endpoints as free points.
    let solid = SketchSolid::extrude(sketch.clone(), 1);
    let without_arc = solid.with_arc_deleted(arc);
    assert!(without_arc.sketch.arcs().is_empty());
    assert_eq!(without_arc.sketch.points().len(), 2);

    // Deleting an endpoint cascades to the arc.
    sketch.delete_point_cascade(a);
    assert!(sketch.arcs().is_empty(), "the incident arc went with it");
    assert_eq!(sketch.points().len(), 1);

    // Repair erases a dangling arc, a self-loop, and a degenerate bulge — and counts them.
    let mut broken = Sketch::new(PlaneAxis::Z, vec![]);
    let p = broken.add_free_point(SketchPoint::new(0, 0));
    let q = broken.add_free_point(SketchPoint::new(4, 0));
    let good = broken
        .connect_arc(p, q, AngleMeasurement::from_degrees(45))
        .expect("fresh arc");
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 90,
        from: p,
        to: 77, // dangling
        bulge: AngleMeasurement::from_degrees(90),
        origin: 90,
        role: crate::sketch::EntityRole::Real,
    });
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 91,
        from: p,
        to: p, // self-loop
        bulge: AngleMeasurement::from_degrees(90),
        origin: 91,
        role: crate::sketch::EntityRole::Real,
    });
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 92,
        from: q,
        to: p,
        bulge: AngleMeasurement::from_degrees(0), // degenerate bulge
        origin: 92,
        role: crate::sketch::EntityRole::Real,
    });
    assert_eq!(broken.repair(), 3);
    assert_eq!(broken.arcs().len(), 1);
    assert_eq!(broken.arcs()[0].id, good);
}

#[test]
fn arcs_round_trip_through_serde_and_a_pre_arc_document_loads_clean() {
    let solid = rounded_bottom_solid(2);
    let json = serde_json::to_string(&solid.sketch).expect("serialise");
    let restored: Sketch = serde_json::from_str(&json).expect("deserialise");
    assert_eq!(restored, solid.sketch, "the arc store round-trips verbatim");

    // A pre-#102 document has no `arcs` key: strip it and the sketch still loads, with
    // no arcs (the serde default).
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse");
    value
        .as_object_mut()
        .expect("a sketch is a JSON object")
        .remove("arcs")
        .expect("the key was present");
    let pre_arc: Sketch = serde_json::from_value(value).expect("a pre-arc document loads");
    assert!(pre_arc.arcs().is_empty());
    assert_eq!(pre_arc.points(), solid.sketch.points());
}

#[test]
fn quantized_angle_survives_the_float_door() {
    let exact = AngleMeasurement::from_degrees_f64(-180.0).expect("finite");
    assert_eq!(exact, AngleMeasurement::from_degrees(-180));
    let solved = AngleMeasurement::from_degrees_f64(123.4567).expect("finite");
    assert!(
        (solved.to_degrees_f64() - 123.4567).abs() <= 1.0 / 7200.0,
        "arc-second quantization: within half a second of arc"
    );
    assert_eq!(AngleMeasurement::from_degrees_f64(f64::NAN), None);
}
