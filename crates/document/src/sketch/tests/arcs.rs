//! The arc entity: the 3-point solve, the canonical endpoints+bulge store, chord
//! tessellation through the flattened loop, resolve through an arced profile, delete
//! cascade / repair, and serialization (including a pre-arc document loading clean).

use super::ctx;
use crate::sketch::{
    arc_center_radius, arc_interior_points, included_angle_through_degrees, EntityId, PlaneAxis,
    Point, PointLifetime, Sketch, SketchPoint, SketchSolid, ARC_SAGITTA_TOLERANCE_VOXELS,
};
use crate::voxel::VoxelProducer;
use ::parametric::units::AngleMeasurement;
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
    // Semicircle through the TOP: from (0,0) to (2,0) via (1,1) — center (1,0), and the
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
    ring.extend(interior.iter().map(SketchPoint::in_plane));
    ring.push([4.0, 0.0]);
    for point in &ring[1..ring.len() - 1] {
        let distance = ((point[0] - 2.0).powi(2) + point[1].powi(2)).sqrt();
        assert!((distance - 2.0).abs() < 1e-5, "off the circle: {point:?}");
        assert!(point[1] < 0.0, "the +180° fan bulges below the chord");
    }
    // Sagitta bound: each chord's midpoint deviates from the circle by at most the
    // versioned tolerance.
    for pair in ring.array_windows::<2>() {
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
    let flattened = solid.sketch.flattened_loop(ctx(16));
    assert!(
        flattened.len() > 4,
        "the loop carries the arc's chord fan, got {}",
        flattened.len()
    );
    assert_eq!(
        solid.grid_dimensions(ctx(16)),
        [4, 5, 2],
        "in-plane cover 0..4 × -2..3 (the bulge extends the box), extrude 2"
    );
    let mut grid = VoxelGrid::default();
    solid.resolve(&mut grid, 8);
    // Per layer: the 4×3 rectangle plus the half-disc rows below it (4 cells at the
    // first row down, 2 at the second — cell centers against a radius-2 circle).
    assert_eq!(grid.occupied.len(), (12 + 6) * 2);
}

#[test]
fn arc_edges_join_the_region_graph() {
    // A triangle of two straight edges and one arc — the multi-vertex case. The two-vertex
    // D-shape has its own test below.
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    let c = sketch.add_free_point(SketchPoint::new(2, 3));
    sketch.connect(a, c).expect("fresh edge");
    sketch.connect(c, b).expect("fresh edge");
    assert!(
        sketch.flattened_loop(ctx(16)).is_empty(),
        "two edges of three: still open"
    );
    sketch
        .connect_arc(b, a, AngleMeasurement::from_degrees(120))
        .expect("the arc closes the loop");
    assert!(
        sketch.flattened_loop(ctx(16)).len() > 3,
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
        sketch.connect_arc(a, b, quarter),
        None,
        "the identical curve twice is a duplicate"
    );
    assert_eq!(
        sketch.connect_arc(b, a, AngleMeasurement::from_degrees(-90)),
        None,
        "reversed direction AND negated sweep is the same curve"
    );
    let chord = sketch
        .connect(a, b)
        .expect("a chord closes the arc into a D");
    assert_eq!(
        sketch.connect(b, a),
        None,
        "but a SECOND straight edge over the pair is a duplicate"
    );
    sketch.delete_segment(chord);
}

/// Two edges over ONE pair of points are legal in either drawing order: an arc closed by its own
/// chord, and an arc bulging over a pair a segment already joins. The face derivation traces that
/// cycle like any other, so both resolve — a guard rejecting any already-joined pair would refuse
/// the D-shape outright.
#[test]
fn a_chord_and_its_arc_bound_a_d_shape() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));

    // Arc first, then the polyline's chord across it.
    sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(180))
        .expect("the half-circle lands");
    sketch.connect(a, b).expect("the chord closes it");
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 1, "two edges over one pair bound ONE face");
    // Half a radius-2 disc, EXACTLY: the area integrates the arc itself (Green's theorem over the
    // circle), where a tessellated boundary could only inscribe it and land just under.
    let half_disc = std::f64::consts::PI * 2.0;
    let area = faces[0].area_voxels;
    assert!(
        (area - half_disc).abs() < 1e-9,
        "the exact half-disc, got {area} against {half_disc}"
    );
    assert!(!sketch.region(ctx(16)).is_empty(), "and it resolves");

    // The other direction — an arc over a pair a segment already joins — reaches the same store.
    let mut reversed = Sketch::new(PlaneAxis::Z, vec![]);
    let c = reversed.add_free_point(SketchPoint::new(0, 0));
    let d = reversed.add_free_point(SketchPoint::new(4, 0));
    reversed.connect(c, d).expect("the polyline segment lands");
    reversed
        .connect_arc(c, d, AngleMeasurement::from_degrees(180))
        .expect("arcing over it is legal");
    assert_eq!(reversed.faces(ctx(16)).len(), 1);
}

/// Two arcs bulging opposite ways over one pair are a LENS, not a duplicate — the sign of the
/// sweep is what distinguishes them, so the duplicate check cannot be a bare pair test.
#[test]
fn two_arcs_over_one_pair_bound_a_lens() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(120))
        .expect("one side");
    sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(-120))
        .expect("the other side is a different curve");
    let faces = sketch.faces(ctx(16));
    assert_eq!(faces.len(), 1, "the lens is one bounded face");
    assert!(faces[0].area_voxels > 0.0);
}

/// The full turn is a POLE of the endpoint-plus-bulge form, not a policy line. As the sweep
/// approaches 360° the derived radius diverges — a decade closer is a decade larger, without
/// bound — because the chord subtends less and less of the circle it is meant to determine. The
/// guard sits exactly where the arithmetic stops having an answer.
///
/// This matters because the value at 360° itself is FINITE: `sin(PI)` is 1.22e-16 rather than
/// zero, so an unguarded call returns a radius near 4e15 voxels. That would pass every downstream
/// finite check while being nonsense, which is the failure mode the guard exists to make
/// impossible.
///
/// That a closed curve is a `Circle` with no on-curve vertex is enforced separately, by
/// `connect_arc` refusing `from == to`. The two are independent, and this one is arithmetic.
#[test]
fn the_full_turn_is_where_the_radius_diverges() {
    let (from, to) = ([0.0, 0.0], [1.0, 0.0]);
    let mut previous = 0.0;
    for shortfall in [1.0, 1e-2, 1e-4, 1e-6] {
        let (center, radius) =
            arc_center_radius(from, to, 360.0 - shortfall).expect("still short of the turn");
        assert!(
            radius > previous * 50.0,
            "a hundredfold closer to the turn is a hundredfold larger radius — \
             {shortfall} gave {radius}, after {previous}"
        );
        assert!(
            center[1].abs() > radius / 2.0,
            "the center runs away with it, to {center:?}"
        );
        previous = radius;
    }
    assert!(
        previous > 1e6,
        "the last one is already unusable: {previous}"
    );

    for sweep in [360.0, -360.0, 720.0] {
        assert!(
            arc_center_radius(from, to, sweep).is_none(),
            "{sweep}° has no arc to derive"
        );
    }
}

#[test]
fn delete_cascades_and_repair_cover_arcs() {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let a = sketch.add_free_point(SketchPoint::new(0, 0));
    let b = sketch.add_free_point(SketchPoint::new(4, 0));
    let arc = sketch
        .connect_arc(a, b, AngleMeasurement::from_degrees(90))
        .expect("fresh arc");

    // Deleting the arc takes both endpoints with it — nothing else draws them.
    let solid = SketchSolid::extrude(sketch.clone(), 1);
    let without_arc = solid.with_arc_deleted(arc);
    assert!(without_arc.sketch.arcs().is_empty());
    assert!(without_arc.sketch.points().is_empty());

    // Deleting an endpoint cascades to the arc.
    sketch.delete_point_cascade(a);
    assert!(sketch.arcs().is_empty(), "the incident arc went with it");
    assert_eq!(sketch.points().len(), 1);

    // Repair erases a dangling arc, a self-loop, and one whose center is not a point in the
    // store — three arcs that name no circle — and counts them.
    let mut broken = Sketch::new(PlaneAxis::Z, vec![]);
    let p = broken.add_free_point(SketchPoint::new(0, 0));
    let q = broken.add_free_point(SketchPoint::new(4, 0));
    let good = broken
        .connect_arc(p, q, AngleMeasurement::from_degrees(45))
        .expect("fresh arc");
    let seat = broken.add_free_point(SketchPoint::new(2, 2));
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 90,
        from: p,
        to: 77, // dangling
        center: seat,
        origin: 90,
        role: crate::sketch::EntityRole::Real,
    });
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 91,
        from: p,
        to: p, // self-loop
        center: seat,
        origin: 91,
        role: crate::sketch::EntityRole::Real,
    });
    broken.arcs_mut_for_test().push(crate::sketch::Arc {
        id: 92,
        from: q,
        to: p,
        center: 78, // dangling center
        origin: 92,
        role: crate::sketch::EntityRole::Real,
    });
    assert_eq!(broken.repair(ctx(16)), 3);
    assert_eq!(broken.arcs().len(), 1);
    assert_eq!(broken.arcs()[0].id, good);
}

#[test]
fn arcs_round_trip_through_serde_and_a_pre_arc_document_loads_clean() {
    let solid = rounded_bottom_solid(2);
    let json = serde_json::to_string(&solid.sketch).expect("serialize");
    let restored: Sketch = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        restored, *solid.sketch,
        "the arc store round-trips verbatim"
    );

    // A document written before arcs existed has no `arcs` key: strip it and the sketch
    // still loads, with no arcs (the serde default).
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
fn solved_angle_survives_the_exact_float_door() {
    let exact = AngleMeasurement::try_from_degrees_f64(-180.0).expect("representable");
    assert_eq!(exact, AngleMeasurement::from_degrees(-180));
    let value = 123.4567;
    let solved = AngleMeasurement::try_from_degrees_f64(value).expect("representable");
    assert_eq!(solved.to_degrees_f64().to_bits(), value.to_bits());
    assert!(AngleMeasurement::try_from_degrees_f64(f64::NAN).is_err());
}

/// The `[0,0] → [4,0]` half-turn arc: center `[2,0]`, radius 2, bulging down.
fn half_turn() -> (Sketch, EntityId, EntityId, EntityId) {
    let mut sketch = Sketch::new(PlaneAxis::Z, vec![]);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(4, 0));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(180))
        .expect("a legal half turn");
    (sketch, from, to, arc)
}

/// A derived center is a float — the apothem of an exact half turn is `chord / 2 / tan(90°)`,
/// which lands a whisker off zero rather than on it. Compare within a thousandth of a voxel.
fn assert_near(actual: [f64; 2], expected: [f64; 2]) {
    assert!(
        (actual[0] - expected[0]).abs() < 1.0e-3 && (actual[1] - expected[1]).abs() < 1.0e-3,
        "{actual:?} is not {expected:?}"
    );
}

fn center_of(sketch: &Sketch, arc: EntityId) -> Point {
    let center = sketch
        .arcs()
        .iter()
        .find(|candidate| candidate.id == arc)
        .expect("the arc")
        .center;
    *sketch
        .points()
        .iter()
        .find(|point| point.id == center)
        .expect("a reified center point")
}

#[test]
fn an_arc_reifies_its_center_as_a_selectable_point() {
    let (sketch, from, to, arc) = half_turn();
    let center = center_of(&sketch, arc);

    assert_near(center.at.in_plane(), [2.0, 0.0]);
    assert_eq!(center.lifetime, PointLifetime::CurveAnchored);
    assert!(
        ![from, to].contains(&center.id),
        "the center is its own entity, not an endpoint wearing a second hat"
    );
    // It rides in `points()` like any other point, which is the whole ask: the overlay places a
    // handle per point, so the center gets hover, selection and a drag for free.
    assert_eq!(sketch.points().len(), 3);

    // Being construction geometry, it bounds nothing: an isolated point is not a face, and the
    // arc's own single edge still cannot close one.
    assert!(sketch.faces(ctx(16)).is_empty());
}

/// Dragging the center MOVES THE ARC, all three points by the one displacement.
///
/// The center is the place the curve turns about, so taking hold of it is a statement about the
/// curve and not about that one dot — Fusion says the same of the simplest case, "if you drag the
/// center point you will change the position of the arc like in a circle". Under the old derived
/// model a center had one freedom, how far out along the chord's bisector it stood, and a drag of
/// it authored the SWEEP;
/// [ADR 0038](../../../../../docs/adr/0038-a-point-is-placed-never-computed.md) ended that, and
/// what is left is an ordinary point at the middle of a rigid set.
#[test]
fn dragging_a_center_carries_the_whole_arc() {
    let (mut sketch, from, to, arc) = half_turn();
    let center = center_of(&sketch, arc).id;
    let sweep_of = |sketch: &Sketch| {
        sketch
            .arc_form_of(arc)
            .expect("three points that draw an arc")
            .sweep_degrees
    };
    let before = sweep_of(&sketch);

    // Two voxels up, and nothing else said. The half turn's center sits ON the chord, so this is
    // the very drag that used to halve the sweep to 90.
    assert!(sketch
        .move_point(center, SketchPoint::new(2, 2), ctx(16))
        .expect("evaluation context"));

    let position = |id| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .expect("the point")
            .at
            .in_plane()
    };
    assert_near(position(from), [0.0, 2.0]);
    assert_near(position(to), [4.0, 2.0]);
    assert_near(center_of(&sketch, arc).at.in_plane(), [2.0, 2.0]);
    let after = sweep_of(&sketch);
    assert!(
        (after - before).abs() < 1.0e-3,
        "a translation is not a reshape: {before} became {after}"
    );
}

/// A center drag has TWO freedoms now, not one, so it lands where it was put.
///
/// The old rule projected a center onto the chord's perpendicular bisector, because that line was
/// the whole of what a derived center could say. A carried arc has nothing to project: the
/// along-chord component is a real displacement of the shape.
#[test]
fn a_center_lands_where_it_is_put_and_takes_the_arc_with_it() {
    let (mut sketch, from, to, arc) = half_turn();
    let center = center_of(&sketch, arc).id;
    assert!(sketch
        .move_point(center, SketchPoint::new(9, -2), ctx(16))
        .expect("evaluation context"));

    let position = |id| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .expect("the point")
            .at
            .in_plane()
    };
    // Seven along the chord and two across it, and every point of the arc goes the same way.
    assert_near(center_of(&sketch, arc).at.in_plane(), [9.0, -2.0]);
    assert_near(position(from), [7.0, -2.0]);
    assert_near(position(to), [11.0, -2.0]);
}

/// Dragging the center INTO the bulge no longer flips minor to major — it carries the arc down
/// past its own ends, sweep and all. Changing the sweep is a drag of an END or of the rim, which
/// are the gestures that say something about the curve's shape rather than about its place.
#[test]
fn a_center_dragged_into_the_bulge_carries_the_arc_rather_than_flipping_it() {
    let (mut sketch, _from, _to, arc) = half_turn();
    let center = center_of(&sketch, arc).id;
    let sweep_of = |sketch: &Sketch| {
        sketch
            .arc_form_of(arc)
            .expect("three points that draw an arc")
            .sweep_degrees
    };
    let before = sweep_of(&sketch);
    assert!(sketch
        .move_point(center, SketchPoint::new(2, -2), ctx(16))
        .expect("evaluation context"));
    let after = sweep_of(&sketch);
    assert!(
        (after - before).abs() < 1.0e-3,
        "the arc travelled into its own bulge and stayed the same arc: {before} became {after}"
    );
    assert_near(center_of(&sketch, arc).at.in_plane(), [2.0, -2.0]);
}

/// The center is authored, but its ONE freedom is how far out along the chord's bisector it
/// stands — so when an end moves, the seat moves with the bisector rather than staying put and
/// leaving the dot off the curve.
#[test]
fn a_center_re_seats_when_an_endpoint_moves() {
    let (mut sketch, _from, to, arc) = half_turn();
    assert!(sketch
        .move_point(to, SketchPoint::new(2, 0), ctx(16))
        .expect("evaluation context"));
    assert_near(center_of(&sketch, arc).at.in_plane(), [1.0, 0.0]);
}

#[test]
fn a_center_lives_and_dies_with_its_arc() {
    let (mut sketch, from, _to, arc) = half_turn();
    let center = center_of(&sketch, arc).id;

    // Deleting the CENTER takes the arc: there is no arc left for it to be the center of.
    let mut by_center = sketch.clone();
    by_center.delete_point_cascade(center);
    assert!(by_center.arcs().is_empty());
    assert_eq!(
        by_center.points().len(),
        2,
        "both endpoints survive as free"
    );

    // Deleting the ARC takes the center AND both ends: nothing else draws them, and a curve
    // deleted from a drawing must not leave dots behind that the author never placed.
    sketch.delete_arc(arc);
    assert!(sketch.arcs().is_empty());
    assert!(sketch.points().is_empty());
    assert!(!sketch.points().iter().any(|point| point.id == from));

    // A center the author has since drawn TO is referenced geometry, and outlives its arc.
    let (mut kept, from, _to, arc) = half_turn();
    let center = center_of(&kept, arc).id;
    kept.connect(from, center).expect("a radius line");
    kept.delete_arc(arc);
    assert!(kept.points().iter().any(|point| point.id == center));
}

/// A face bounded by an arc keeps the ARC. Its boundary is two straight sides, one more, and the
/// curve — not a fan of chords — and the curve's own circle is what the field measures.
///
/// This is what keeps the wash from reading as a polygon inside its own smooth outline. There is no
/// tolerance to get right here because there is no tessellation: the region is a curve all the way
/// to the measurement, and the only thing that ever flattens is a consumer producing something
/// discrete.
#[test]
fn a_curved_face_keeps_its_arc() {
    let mut sketch = Sketch::new(
        PlaneAxis::Z,
        vec![
            SketchPoint::new(0, 0),
            SketchPoint::new(0, 3),
            SketchPoint::new(4, 3),
            SketchPoint::new(4, 0),
        ],
    );
    let bottom = sketch
        .segments()
        .iter()
        .find(|seg| [seg.from, seg.to].contains(&0) && [seg.from, seg.to].contains(&3))
        .expect("the wrap segment")
        .id;
    sketch.delete_segment(bottom);
    sketch
        .connect_arc(0, 3, AngleMeasurement::from_degrees(180))
        .expect("a fresh arc");

    let region = sketch.region(ctx(16));
    assert_eq!(region.len(), 1);
    let edges = &region[0].edges;
    assert_eq!(
        edges.len(),
        4,
        "three straight sides and one arc, not a chord fan"
    );
    let curved: Vec<_> = edges.iter().filter(|edge| edge.arc.is_some()).collect();
    assert_eq!(curved.len(), 1, "exactly one edge carries a circle");
    let arc = curved[0].arc.expect("the circle");
    // The replaced side spans (0, 0) → (4, 0), so a half turn over it has radius 2 about (2, 0)
    // and bulges a full radius below the rectangle.
    assert!(
        (arc.radius - 2.0).abs() < 1e-9,
        "a half turn across a 4-voxel chord has radius 2, measured {}",
        arc.radius
    );
    assert!(
        (arc.sweep_radians.abs() - std::f64::consts::PI).abs() < 1e-9,
        "a half turn"
    );

    // The bulge is material, and the field measures it against the CIRCLE: a point one voxel in
    // from the curve reads exactly one voxel deep, which a chord approximation cannot say.
    let field = sketch.region_field_loops(ctx(16));
    let under_the_bulge = [arc.center[0] as f32, (arc.center[1] - 1.0) as f32];
    assert!(
        substrate::geom2d::point_in_region(&field, under_the_bulge),
        "under the bulge at {under_the_bulge:?}"
    );
    let gap = substrate::geom2d::signed_distance_to_region(
        &field,
        under_the_bulge,
        substrate::geom2d::Metric::Euclidean,
    );
    assert!(
        (gap + 1.0).abs() < 1e-4,
        "one voxel in from the curve, measured {gap}"
    );

    // Faces are identified by lineage, not geometry.
    let key = sketch.identified_faces(ctx(16)).first().expect("a face").1;
    assert!(sketch.face_is_picked(&key, ctx(16)));
}
