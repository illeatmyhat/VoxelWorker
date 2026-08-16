//! The arc entity: the 3-point solve, the canonical endpoints+bulge store, chord
//! tessellation through the flattened loop, resolve through an arced profile, delete
//! cascade / repair, and serialization (including a pre-arc document loading clean).

use super::ctx;
use crate::sketch::{
    arc_center_radius, arc_interior_points, included_angle_through_degrees, EntityId, PlaneAxis,
    Point, PointLifetime, Sketch, SketchPoint, SketchSolid, ARC_SAGITTA_TOLERANCE,
};
use crate::sketch::{
    wrapped_into_a_half_turn, ConstraintKind, ConstraintRefusal, Dimension, GestureSoFar,
    SketchCurve, SketchLength,
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

/// **An angle can be struck against an arc, and it is struck at an END.**
///
/// The whole reason the arm is a type rather than a segment id: a curve that turns has a different
/// direction at every point, so an angle to one is not a question until a place is named. Here a
/// free line is turned to leave a pinned quarter arc's end at 30 degrees, and what it ends up doing
/// is measured against the TANGENT there — perpendicular to the radius standing at that end.
#[test]
fn an_angle_arm_can_be_an_arcs_own_tangent() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    // A quarter arc about the origin, running counter-clockwise from (10,0) to (0,10). All THREE
    // of its points are pinned — the center included, which the arc minted for itself — so the
    // only thing left free to move is the line.
    let start = sketch.add_free_point(SketchPoint::new(10, 0));
    let finish = sketch.add_free_point(SketchPoint::new(0, 10));
    let arc = sketch
        .connect_arc(start, finish, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc");
    let held_arc = *sketch
        .arcs()
        .iter()
        .find(|held| held.id == arc)
        .expect("the arc");
    for point in [held_arc.center, held_arc.from, held_arc.to] {
        let at = sketch
            .points()
            .iter()
            .find(|held| held.id == point)
            .expect("the point")
            .at;
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("pinning a point where it already is");
    }
    let tail = sketch.add_free_point(SketchPoint::new(10, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 4));
    let line = sketch.connect(tail, head).expect("a free line");
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(10, 0),
            },
            ctx(16),
        )
        .expect("pinning the line's tail at the arc's end");

    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: crate::sketch::AngleArm::ArcEnd {
                    arc,
                    end: crate::sketch::ArcEnd::From,
                },
                second: crate::sketch::AngleArm::Segment { segment: line },
                degrees: AngleMeasurement::from_degrees(30),
                corner: crate::sketch::AngleCorner::Between,
            }),
            ctx(16),
        )
        .expect("a free line can always stand thirty degrees off a pinned tangent");

    let at = |id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|held| held.id == id)
            .expect("the point")
            .at
            .in_plane()
    };
    // The tangent at (10,0) about (0,0) runs straight up, so the line should end up 30 degrees
    // off vertical — and the residual is a sine, so 30 either way is the same claim.
    let (tail_at, head_at) = (at(tail), at(head));
    let line_bearing = (head_at[1] - tail_at[1])
        .atan2(head_at[0] - tail_at[0])
        .to_degrees();
    let off_the_tangent = (line_bearing - 90.0).rem_euclid(180.0);
    let turn = off_the_tangent.min(180.0 - off_the_tangent);
    assert!(
        (turn - 30.0).abs() < 1e-6,
        "thirty degrees off the tangent, got {turn} (line bears {line_bearing})"
    );

    // The OTHER end is a different tangent, so the same pair can state it too — an id pair alone
    // would have called the second one already asserted.
    let other_end = ConstraintKind::Dimension(Dimension::Angle {
        first: crate::sketch::AngleArm::ArcEnd {
            arc,
            end: crate::sketch::ArcEnd::To,
        },
        second: crate::sketch::AngleArm::Segment { segment: line },
        degrees: AngleMeasurement::from_degrees(60),
        corner: crate::sketch::AngleCorner::Between,
    });
    assert!(
        !sketch.constraints()[sketch.constraints().len() - 1]
            .kind
            .is_about_the_same_as(other_end),
        "the two ends of one arc are two different tangents"
    );

    // And an arm on a curve that has gone is refused rather than silently dropped.
    let mut orphaned = sketch.clone();
    orphaned.delete_arc(held_arc.id);
    assert_eq!(
        orphaned.add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: crate::sketch::AngleArm::ArcEnd {
                    arc,
                    end: crate::sketch::ArcEnd::To,
                },
                second: crate::sketch::AngleArm::Segment { segment: line },
                degrees: AngleMeasurement::from_degrees(45),
                corner: crate::sketch::AngleCorner::Between,
            }),
            ctx(16),
        ),
        Err(ConstraintRefusal::UnknownEntity),
        "the arc went with its endpoint"
    );
}

/// **Reversing an arc carries every relation that named one of its ends.**
///
/// An arc runs counter-clockwise from `from` to `to`, so drawing it the other way round is a swap
/// of those two fields — and an [`AngleArm::ArcEnd`](crate::sketch::AngleArm::ArcEnd) names an end
/// by that field, not by a point id. Swap the fields alone and every angle struck at one end
/// silently starts reading the other: a tangent that stood vertical starts standing horizontal
/// under a relation nobody touched.
///
/// Measured both ways here. The stored tag says `To` where it said `From`, and the drawing settles
/// the same: pull the free head right off and the line comes back 30 degrees off the tangent at
/// `(10,0)`, the dot the author struck the angle at. Without the cascade it comes back 30 degrees
/// off the tangent at `(0,10)` instead, which reads 60 at the dot that was named.
#[test]
fn reversing_an_arc_carries_the_angles_struck_at_its_ends() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let start = sketch.add_free_point(SketchPoint::new(10, 0));
    let finish = sketch.add_free_point(SketchPoint::new(0, 10));
    let arc = sketch
        .connect_arc(start, finish, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc");
    let held_arc = *sketch
        .arcs()
        .iter()
        .find(|held| held.id == arc)
        .expect("the arc");
    for point in [held_arc.center, held_arc.from, held_arc.to] {
        let at = sketch
            .points()
            .iter()
            .find(|held| held.id == point)
            .expect("the point")
            .at;
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("pinning a point where it already is");
    }
    let tail = sketch.add_free_point(SketchPoint::new(10, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 4));
    let line = sketch.connect(tail, head).expect("a free line");
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(10, 0),
            },
            ctx(16),
        )
        .expect("pinning the line's tail at the arc's end");
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: crate::sketch::AngleArm::ArcEnd {
                    arc,
                    end: crate::sketch::ArcEnd::From,
                },
                second: crate::sketch::AngleArm::Segment { segment: line },
                degrees: AngleMeasurement::from_degrees(30),
                corner: crate::sketch::AngleCorner::Between,
            }),
            ctx(16),
        )
        .expect("a free line can always stand thirty degrees off a pinned tangent");

    assert!(sketch.reverse_arc(arc), "an arc that is there reverses");
    let reversed = *sketch
        .arcs()
        .iter()
        .find(|held| held.id == arc)
        .expect("the arc");
    assert_eq!(
        (reversed.from, reversed.to),
        (finish, start),
        "the two ends swapped"
    );
    let arm = sketch
        .constraints()
        .iter()
        .find_map(|held| match held.kind {
            ConstraintKind::Dimension(Dimension::Angle { first, .. }) => Some(first),
            _ => None,
        })
        .expect("the angle is still stored");
    assert_eq!(
        arm,
        crate::sketch::AngleArm::ArcEnd {
            arc,
            end: crate::sketch::ArcEnd::To,
        },
        "the arm followed the end it named to its new field"
    );

    // Pull the head well off and let the drawing answer: the angle is imposed at the dot the
    // author struck it at, whichever field that dot is stored in now.
    sketch
        .move_point(head, SketchPoint::from_continuous(30.0, -6.0), ctx(16))
        .expect("evaluation context");
    let at = |id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|held| held.id == id)
            .expect("the point")
            .at
            .in_plane()
    };
    let (tail_at, head_at) = (at(tail), at(head));
    let line_bearing = (head_at[1] - tail_at[1])
        .atan2(head_at[0] - tail_at[0])
        .to_degrees();
    // The tangent at (10,0) about the origin runs straight up, and the residual is a sine, so
    // thirty either side of it is the same claim.
    let off_the_tangent = (line_bearing - 90.0).rem_euclid(180.0);
    let turn = off_the_tangent.min(180.0 - off_the_tangent);
    assert!(
        (turn - 30.0).abs() < 1e-6,
        "thirty degrees off the tangent at (10,0), got {turn} (line bears {line_bearing})"
    );
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
            2.0 - distance <= ARC_SAGITTA_TOLERANCE + 1e-6,
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
    let area = faces[0].area;
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
    assert!(faces[0].area > 0.0);
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

/// **An end dragged onto its own center is refused, and the arc is left standing.**
///
/// The limit of holding the center. With the pin honored, the radius is whatever the dragged end
/// says it is, and an end asked to stand on the center says nothing: a circle of radius zero
/// satisfies both of the arc's rows perfectly and collapses all three points onto one place, from
/// which no author can pull the arc back out. The reading is what refuses it — an end sitting on
/// its center draws no arc, so [`arc_form`](Sketch::arc_form) answers nothing and the frame is
/// dropped whole.
///
/// This used to land somewhere else and land quietly: the center re-seated onto the bisector of
/// the new chord, so the half turn from `[0,0]` to `[2,0]` came back as a half turn of radius 1
/// about `[1,0]`. That was the same re-seat that walked the whole arc across the plane on every
/// endpoint drag, and it is gone.
#[test]
fn an_end_dragged_onto_its_own_center_is_refused() {
    let (mut sketch, from, to, arc) = half_turn();
    assert_eq!(
        sketch.move_point(to, SketchPoint::new(2, 0), ctx(16)),
        Ok(false),
        "the drawing took a frame that leaves it no arc"
    );
    assert_near(center_of(&sketch, arc).at.in_plane(), [2.0, 0.0]);
    assert_near(sketch.point_in_plane(to).expect("the far end"), [4.0, 0.0]);
    assert_near(
        sketch.point_in_plane(from).expect("the near end"),
        [0.0, 0.0],
    );
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

/// A rim drag says ONE thing: how far out the rim now stands. It never moves the center.
///
/// An arc and a circle are the same shape as far as this gesture is concerned, so they are held to
/// the same answer here rather than to two answers that happen to look alike. The arc used to
/// project the travel onto the radial direction and hand the tangential remainder to the center,
/// which slid out from under the shape — pull the rim of a unit-10 arc from `(10,0)` to `(14,6)`
/// and the center went to `(0,6)`; pull it straight sideways to `(10,6)` and the whole arc
/// travelled, radius untouched.
#[test]
fn a_rim_drag_sets_the_radius_from_the_cursor_and_holds_the_center() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(10, 0));
    let to = sketch.add_free_point(SketchPoint::new(0, 10));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc of radius 10 about the origin");
    let circle = sketch
        .add_circle(SketchPoint::new(40, 0), SketchLength::new(10))
        .expect("a lone circle of radius 10");
    let seat = |sketch: &Sketch, id: EntityId| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.at.in_plane())
            .expect("the center survives its own rim drag")
    };
    let center = sketch
        .arcs()
        .iter()
        .find(|held| held.id == arc)
        .map(|held| held.center)
        .expect("the arc center");
    let hub = sketch
        .circles()
        .iter()
        .find(|held| held.id == circle)
        .map(|held| held.center)
        .expect("the circle center");

    // Grabbed on the rim at (10,0) — or (50,0) for the circle, which is the same place relative to
    // its own center — and pulled out, out-and-along, and purely along. The answer each time is
    // the distance from the center to the cursor.
    for (name, pull, want) in [
        ("straight out", [4.0, 0.0], 14.0),
        ("out and along", [4.0, 6.0], 14.0_f64.hypot(6.0)),
        ("purely along", [0.0, 6.0], 10.0_f64.hypot(6.0)),
    ] {
        let mut moved = sketch.clone();
        assert_eq!(
            moved.drag_curve_through(
                SketchCurve::Arc(arc),
                [10.0, 0.0],
                [10.0 + pull[0], pull[1]],
                ctx(16),
            ),
            Ok(true),
            "the arc rim drag {name} was refused"
        );
        let at = seat(&moved, center);
        assert!(
            at[0].hypot(at[1]) < 1.0e-9,
            "an arc rim drag {name} moved the center to {at:?}"
        );
        let end = seat(&moved, from);
        let radius = (end[0] - at[0]).hypot(end[1] - at[1]);
        assert!(
            (radius - want) < 1.0e-6 && (want - radius) < 1.0e-6,
            "an arc rim drag {name} answered radius {radius}, not the {want} the cursor stood at"
        );

        let mut moved = sketch.clone();
        assert_eq!(
            moved.drag_curve_through(
                SketchCurve::Circle(circle),
                [50.0, 0.0],
                [50.0 + pull[0], pull[1]],
                ctx(16),
            ),
            Ok(true),
            "the circle rim drag {name} was refused"
        );
        let at = seat(&moved, hub);
        assert!(
            (at[0] - 40.0).hypot(at[1]) < 1.0e-6,
            "a circle rim drag {name} moved the hub to {at:?}"
        );
        let radius = moved
            .circles()
            .iter()
            .find(|held| held.id == circle)
            .map(|held| held.resolved_radius(ctx(16)))
            .expect("the circle survives its own rim drag");
        assert!(
            (radius - want) < 1.0e-6 && (want - radius) < 1.0e-6,
            "a circle rim drag {name} answered radius {radius}, not {want}"
        );
    }
}

/// One dimension reaches an arc and a circle alike, and it OUTRANKS the rim drag.
///
/// The point of the family: how big a round curve is does not depend on whether it has ends. The
/// arc's radius is a solver column minted beside its three points and the circle's is authored
/// beside its center, and the same relation reads a row against either.
///
/// Held against the gesture, because a dimension that a drag can quietly spend is not a dimension.
/// This is the assertion that would have made a slot's width impossible to collapse.
#[test]
fn one_radius_dimension_holds_an_arc_and_a_circle_against_their_own_rim_drags() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(10, 0));
    let to = sketch.add_free_point(SketchPoint::new(0, 10));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc of radius 10 about the origin");
    let circle = sketch
        .add_circle(SketchPoint::new(40, 0), SketchLength::new(10))
        .expect("a lone circle of radius 10");

    // A straight curve has no center to measure from.
    let segment = {
        let tail = sketch.add_free_point(SketchPoint::new(0, -20));
        let head = sketch.add_free_point(SketchPoint::new(10, -20));
        sketch.connect(tail, head).expect("a segment")
    };
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Dimension(Dimension::Radius {
                curve: SketchCurve::Segment(segment),
                length: SketchLength::new(10),
            }),
            ctx(16),
        ),
        Err(ConstraintRefusal::Impossible),
        "a segment took a radius"
    );

    for (name, curve, grab) in [
        ("arc", SketchCurve::Arc(arc), [10.0, 0.0]),
        ("circle", SketchCurve::Circle(circle), [50.0, 0.0]),
    ] {
        let mut held = sketch.clone();
        assert!(
            held.add_constraint(
                ConstraintKind::Dimension(Dimension::Radius {
                    curve,
                    length: SketchLength::new(10),
                }),
                ctx(16),
            )
            .is_ok(),
            "the {name} refused a radius it already stands at"
        );
        // The same rim drag that grows an undimensioned curve to 14. It is ANSWERED, not refused
        // — the drawing moves and settles back on the dimension, rather than the gesture being
        // turned away and the radius surviving because nothing happened.
        assert_eq!(
            held.drag_curve_through(curve, grab, [grab[0] + 4.0, 0.0], ctx(16)),
            Ok(true),
            "the {name} rim drag was not answered, so this proves nothing"
        );
        let radius = match curve {
            SketchCurve::Circle(id) => held
                .circles()
                .iter()
                .find(|circle| circle.id == id)
                .map(|circle| circle.resolved_radius(ctx(16)))
                .expect("the circle"),
            _ => {
                let seat = |id: EntityId| {
                    held.points()
                        .iter()
                        .find(|point| point.id == id)
                        .map(|point| point.at.in_plane())
                        .expect("the point")
                };
                let center = held
                    .arcs()
                    .iter()
                    .find(|held| held.id == arc)
                    .map(|held| held.center)
                    .expect("the arc");
                let (hub, end) = (seat(center), seat(from));
                (end[0] - hub[0]).hypot(end[1] - hub[1])
            }
        };
        assert!(
            (radius - 10.0).abs() < 1.0e-6,
            "a dimensioned {name} was dragged to {radius}, not held at 10"
        );
    }
}

/// A bare quarter arc from `(-40,0)` to `(40,0)`, which mints its center at `(0,40)` and stands
/// both ends `56.57` out from it. Nothing is asserted about it — that is the whole point, because
/// the drag path an undimensioned arc takes is the one under test.
fn bare_quarter_arc() -> (Sketch, EntityId, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::from_continuous(-40.0, 0.0));
    let to = sketch.add_free_point(SketchPoint::from_continuous(40.0, 0.0));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("a fresh arc over an unjoined pair");
    let center = sketch
        .arcs()
        .iter()
        .find(|held| held.id == arc)
        .expect("the arc")
        .center;
    (sketch, from, to, center)
}

/// **An arc endpoint drag turns about the center, and the center does not move.**
///
/// The gesture already declares this — [`Hand`](crate::sketch::Hand) names the center a `Pin` — and
/// the declaration was honest all along. What was not honest was the path: with nothing asserted
/// about a bare arc, the settle took a write-through shortcut that never ran the solver, so the
/// `Pin` was never turned into a row, and the seat that ran on the very next line projected the
/// center onto the perpendicular bisector of the chord the drag had just changed. Measured, a pull
/// of the near end to `[-60,-20]` walked the center from `[0,40]` to `[-19.23,36.15]` and took the
/// whole arc with it.
///
/// The far-pull clause guards the regression the shortcut was born to fix: an unconstrained end
/// must still land where the cursor is. A solve that drops a Lead it has nothing to trade against
/// fails here rather than in the author's hands.
#[test]
fn an_arc_endpoint_drag_holds_the_center_it_turns_about() {
    let (mut sketch, from, _, center) = bare_quarter_arc();
    assert_eq!(
        sketch.move_point(from, SketchPoint::from_continuous(-60.0, -20.0), ctx(16)),
        Ok(true),
        "the drag was not answered, so this proves nothing"
    );
    let hub = sketch.point_in_plane(center).expect("the center");
    assert!(
        (hub[0]).abs() < 1.0e-6 && (hub[1] - 40.0).abs() < 1.0e-6,
        "the pinned center walked to {hub:?} instead of standing at [0, 40]"
    );
    let end = sketch.point_in_plane(from).expect("the dragged end");
    assert!(
        (end[0] + 60.0).abs() < 1.0e-6 && (end[1] + 20.0).abs() < 1.0e-6,
        "the dragged end landed at {end:?} instead of under the cursor at [-60, -20]"
    );
}

/// **With the center held, the FAR end slides along its own radius.**
///
/// The consequence of holding the center, and the half of the gesture nobody was performing. An arc
/// stores three points and derives everything else from them, so a drag that moves one end without
/// moving the other leaves a triple no circle passes through — and every reader manufactures a
/// circle anyway ([`arc_form`](Sketch::arc_form) re-seats the center on each read), which is how an
/// inconsistent store becomes a moving picture.
///
/// The far end is not a hand. It is FREE, and the arc's own rows — one radius, one row per end —
/// place it: least-norm satisfies `|to - center| = radius` by moving it the shortest way there,
/// which is straight out along the ray it already stands on. So the bearing is the assertion. The
/// sweep changes as a consequence, which is the thing the author asked for.
#[test]
fn an_arc_endpoint_drag_slides_the_far_end_along_its_own_radius() {
    let (mut sketch, from, to, center) = bare_quarter_arc();
    let hub = sketch.point_in_plane(center).expect("the center");
    let stood = sketch.point_in_plane(to).expect("the far end");
    let bearing = (stood[1] - hub[1]).atan2(stood[0] - hub[0]);
    // Walked, because a drag is walked. Each frame's answer is a step taken from a seed the caller
    // has already bent, so the tangential residue is a function of how far one frame reaches: the
    // same pull taken in one jump leaves the far end 1.78 degrees off its ray, and in eight steps
    // it leaves 0.27.
    for step in 1..=8 {
        let travel = f64::from(step) / 8.0;
        assert_eq!(
            sketch.move_point(
                from,
                SketchPoint::from_continuous(-40.0 - 20.0 * travel, -20.0 * travel),
                ctx(16)
            ),
            Ok(true),
            "the drag was not answered, so this proves nothing"
        );
    }
    let hub = sketch.point_in_plane(center).expect("the center");
    let near = sketch.point_in_plane(from).expect("the dragged end");
    let far = sketch.point_in_plane(to).expect("the far end");
    let reach = |at: [f64; 2]| (at[0] - hub[0]).hypot(at[1] - hub[1]);
    assert!(
        (reach(far) - reach(near)).abs() < 1.0e-6,
        "the arc's ends stand {} and {} from the center, so no circle passes through both",
        reach(near),
        reach(far)
    );
    let turned = (far[1] - hub[1]).atan2(far[0] - hub[0]) - bearing;
    assert!(
        turned.abs() < 0.02,
        "the far end swung {turned} radians off its own ray instead of sliding along it"
    );
    assert!(
        reach(far) > 56.6,
        "the far end stands {} from the center, so it did not follow the radius out",
        reach(far)
    );
}

/// **Seating the drawing a second time moves nothing.**
///
/// [`sync_derived_points`](Sketch::sync_derived_points) runs at the end of BOTH settle paths, and it
/// is only ever meant to remove a drift nobody chose. That makes it an identity on any drawing the
/// settle left consistent — so running it twice is the cheapest way to ask whether it is still a
/// corrector anywhere.
///
/// A guard rather than a falsification: it was green before the pin was honored and it is green
/// after, because projecting onto a bisector is idempotent once the chord stops moving. Seen red by
/// deleting the settle's own closing seat, which leaves the solver's `1e-8` residue in the store
/// for the second one to find. What it is here to catch is the day a settle leaves the drawing
/// inconsistent — which surfaces to the author not as a wrong shape but as one that drifts a little
/// further every frame they hold the mouse down.
#[test]
fn seating_the_drawing_twice_moves_nothing_the_first_seat_did_not() {
    let (mut sketch, from, _, _) = bare_quarter_arc();
    assert_eq!(
        sketch.move_point(from, SketchPoint::from_continuous(-60.0, -20.0), ctx(16)),
        Ok(true),
        "the drag was not answered, so this proves nothing"
    );
    let settled: Vec<Point> = sketch.points().to_vec();
    sketch.sync_derived_points();
    let seated: Vec<Point> = sketch.points().to_vec();
    for (before, after) in settled.iter().zip(seated.iter()) {
        assert_eq!(
            before.at.in_plane(),
            after.at.in_plane(),
            "a second seat moved point {:?}, so the first one left the drawing inconsistent",
            before.id
        );
    }
}

/// **An end walked past the other end keeps drawing the same arc, on the other side.**
///
/// The seam an arc's own reading cannot smooth over. `counter_clockwise_sweep` is right that its
/// jump is honest — a hair short of closing is a hair short of a whole turn, and a hair past is a
/// hair past zero, and those are two genuinely different arcs. What is wrong is asking the reading
/// to answer a question about a HAND. The author turning an end past the far one drew one
/// continuous motion, so the curve under it has to be continuous too, and the only way an arc
/// changes direction is for its ends to swap.
///
/// Walked a whole turn and a quarter, so BOTH seams are crossed: the ends meeting at zero, and the
/// ends meeting again a full turn later. Measured before the fix, the first crossing took the
/// sweep from 15 degrees to 342 in one 15-degree step, collapsed the radius from 56.57 to 40, and
/// then left it stuck at 23.91 while the far end's bearing drifted from -45 to -31 under a drag
/// that never asked it to move.
#[test]
fn an_end_walked_past_the_other_end_draws_one_continuous_arc() {
    let (mut sketch, from, to, center) = bare_quarter_arc();
    let arc = sketch.arcs()[0].id;
    let hub = sketch.point_in_plane(center).expect("the center");
    let radius = 56.568_542_494_923_804_f64;
    let mut turns = GestureSoFar::opening_over(&sketch);
    let bearing_of = |sketch: &Sketch, id| {
        let at = sketch.point_in_plane(id).expect("a point");
        (at[1] - hub[1]).atan2(at[0] - hub[0]).to_degrees()
    };
    let far_bearing = bearing_of(&sketch, to);
    let step = 15.0_f64;
    let mut stood = 90.0_f64;
    let mut stands = 0;
    // Thirty-one steps of fifteen degrees: from the end's own -135 round to +330, a whole turn and
    // a quarter. Counted rather than accumulated so the walk lands on the seams exactly.
    for taken in 1..=31 {
        let asked = -135.0 + step * f64::from(taken);
        let hand = [
            hub[0] + radius * asked.to_radians().cos(),
            hub[1] + radius * asked.to_radians().sin(),
        ];
        let answered = sketch
            .move_point_reporting_its_snap(
                from,
                SketchPoint::from_continuous(hand[0], hand[1]),
                ctx(16),
                crate::sketch::SnapReach::UNBOUNDED,
                &mut turns,
            )
            .expect("evaluation context")
            .moved;
        if !answered {
            // The seam itself: the two ends stacked, no piece of the circle to prefer. Stood, and
            // the walk carries on — one refused frame at drag rates is a frame nobody sees.
            stands += 1;
            assert!(
                stands <= 2,
                "more than one frame stood at each of the two seams"
            );
            continue;
        }
        let form = sketch
            .arc_form_of(arc)
            .expect("three points that draw an arc");
        assert!(
            (form.radius - radius).abs() < 0.5,
            "the arc stands at radius {} instead of {radius} at {asked} degrees",
            form.radius
        );
        assert!(
            wrapped_into_a_half_turn(bearing_of(&sketch, to) - far_bearing).abs() < 1.0,
            "the far end swung to {} from {far_bearing} at {asked} degrees, and nothing asked it to",
            bearing_of(&sketch, to)
        );
        assert!(
            (form.sweep_degrees - stood).abs() < step + 2.0,
            "the drawn arc jumped from {stood} to {} in one {step}-degree step at {asked} degrees",
            form.sweep_degrees
        );
        stood = form.sweep_degrees;
    }
    // Two crossings, so the ends are back the way they were drawn.
    let ended = sketch.arcs()[0];
    assert_eq!(
        (ended.from, ended.to),
        (from, to),
        "the arc came back from a whole turn drawn backwards"
    );
}
