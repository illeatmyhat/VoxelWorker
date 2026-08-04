use super::*;

#[test]
fn ellipse_is_one_closed_profile_without_boundary_points() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let ellipse = sketch
        .add_ellipse(
            SketchPoint::new(2, 3),
            SketchPoint::new(8, 3),
            SketchPoint::new(2, 7),
        )
        .expect("valid ellipse");

    assert_eq!(sketch.ellipses()[0].id, ellipse);
    assert_eq!(sketch.points().len(), 3);
    assert!(sketch
        .points()
        .iter()
        .all(|point| point.lifetime == PointLifetime::CurveAnchored));
    assert_eq!(sketch.faces(ctx(16)).len(), 1);

    let restored: Sketch =
        serde_json::from_str(&serde_json::to_string(&sketch).expect("ellipse serializes"))
            .expect("ellipse deserializes");
    assert_eq!(restored, sketch);
}

#[test]
fn conic_and_its_chord_bound_a_profile_and_preserve_exact_rho() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let conic = sketch
        .add_conic(
            SketchPoint::new(-4, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(0, 4),
            0.5,
        )
        .expect("valid conic");
    let held = sketch.conics()[0];
    sketch.connect(held.to, held.from).expect("closing chord");

    assert_eq!(held.id, conic);
    assert_eq!(held.rho.rational(), ExactRational::new(1, 2).unwrap());
    assert_eq!(sketch.faces(ctx(16)).len(), 1);
}

/// The control point is a draggable handle now, so the drag has to decline what the authoring
/// gesture already declines. On the chord midpoint the shoulder track has no length and no conic
/// exists; without this the curve would vanish from the handles and faces while the entity stayed.
#[test]
fn dragging_a_conic_control_point_onto_its_chord_is_refused_and_rolls_back() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_conic(
            SketchPoint::new(0, 0),
            SketchPoint::new(8, 0),
            SketchPoint::new(4, 6),
            0.5,
        )
        .expect("valid conic");
    let control = sketch.conics()[0].control;

    let stood = sketch
        .move_point(control, SketchPoint::new(4, 0), ctx(16))
        .expect("the drag is answered, not errored");

    assert!(!stood, "a collapsed conic is not a drag that stands");
    let held = sketch
        .points()
        .iter()
        .find(|point| point.id == control)
        .expect("the control point survives a refused drag");
    assert_eq!(held.at, SketchPoint::new(4, 6));
    assert_eq!(sketch.faces(ctx(16)).len(), 0, "no chord, so no face");
    assert!(sketch
        .move_point(control, SketchPoint::new(4, 9), ctx(16))
        .expect("the drag is answered"));
}

/// Rho is the one authored freedom of a conic with no other handle on it, so the shoulder is
/// reified and dragging it re-solves rho — the same trade an arc's center makes for its sweep.
/// Moving the CONTROL point instead leaves rho alone and takes the shoulder along with it.
#[test]
fn dragging_a_conic_shoulder_resolves_its_rho_while_the_control_point_keeps_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_conic(
            SketchPoint::new(0, 0),
            SketchPoint::new(8, 0),
            SketchPoint::new(4, 8),
            0.5,
        )
        .expect("valid conic");
    let held = sketch.conics()[0];
    let shoulder_at = |sketch: &Sketch| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == held.shoulder)
            .expect("the shoulder is a real point")
            .at
    };
    assert_eq!(shoulder_at(&sketch), SketchPoint::new(4, 4));
    assert!(sketch.is_derived_point(held.shoulder));

    assert!(sketch
        .move_point(held.shoulder, SketchPoint::new(4, 6), ctx(16))
        .expect("the shoulder drag is answered"));
    assert!((sketch.conics()[0].rho.value() - 0.75).abs() < 1.0e-9);
    assert_eq!(shoulder_at(&sketch), SketchPoint::new(4, 6));

    // The control point authors position, not pull: rho survives and the shoulder follows.
    assert!(sketch
        .move_point(held.control, SketchPoint::new(4, 16), ctx(16))
        .expect("the control drag is answered"));
    assert!((sketch.conics()[0].rho.value() - 0.75).abs() < 1.0e-9);
    assert_eq!(shoulder_at(&sketch), SketchPoint::new(4, 12));
}

#[test]
fn higher_curve_handles_retarget_with_density_but_rho_does_not() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_conic(
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 0),
            SketchPoint::new(2, 2),
            0.75,
        )
        .expect("valid conic");
    let rho = sketch.conics()[0].rho;

    sketch.retarget_density(16, 32);

    assert_eq!(sketch.conics()[0].rho, rho);
    let positions: Vec<_> = sketch
        .points()
        .iter()
        .map(|point| point.at.in_plane())
        .collect();
    // Anchors, control point, and the derived shoulder that rho puts three quarters of the way
    // from the chord midpoint (4, 0) out to the control point (4, 4).
    assert_eq!(
        positions,
        vec![[0.0, 0.0], [8.0, 0.0], [4.0, 4.0], [4.0, 3.0]]
    );
}

#[test]
fn closed_fit_spline_is_one_profile_with_only_authored_fit_points() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let spline = sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(-4, -2),
                SketchPoint::new(4, -2),
                SketchPoint::new(0, 5),
            ],
            true,
        )
        .expect("three distinct points make a closed fit spline");

    assert_eq!(sketch.splines()[0].id, spline);
    assert_eq!(sketch.splines()[0].kind, SplineKind::FitPoint);
    assert!(sketch.splines()[0].closed);
    // Three authored fit points, plus the two-armed lever each one is born with.
    assert_eq!(sketch.points().len(), 9);
    let fit_points = sketch.splines()[0].points.clone();
    assert!(sketch.points().iter().all(|point| {
        let lifetime = if fit_points.contains(&point.id) {
            PointLifetime::Freestanding
        } else {
            PointLifetime::CurveAnchored
        };
        point.lifetime == lifetime
    }));
    assert_eq!(sketch.faces(ctx(16)).len(), 1);

    let restored: Sketch =
        serde_json::from_str(&serde_json::to_string(&sketch).expect("fit spline serializes"))
            .expect("fit spline deserializes");
    assert_eq!(restored, sketch);
}

#[test]
fn control_point_spline_keeps_only_its_endpoints_visible() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 5),
            SketchPoint::new(6, 5),
            SketchPoint::new(8, 0),
        ])
        .expect("four controls make one cubic span");

    let spline = &sketch.splines()[0];
    assert_eq!(spline.kind, SplineKind::ControlPoint);
    assert!(!spline.closed);
    let roles: Vec<_> = spline
        .points
        .iter()
        .map(|id| {
            sketch
                .points()
                .iter()
                .find(|point| point.id == *id)
                .expect("control exists")
                .lifetime
        })
        .collect();
    assert_eq!(
        roles,
        vec![
            PointLifetime::Freestanding,
            PointLifetime::CurveAnchored,
            PointLifetime::CurveAnchored,
            PointLifetime::Freestanding,
        ]
    );
}

#[test]
fn deleting_a_control_simplifies_the_spline_until_only_its_ends_remain() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 5),
            SketchPoint::new(6, 5),
            SketchPoint::new(8, 0),
        ])
        .expect("four controls make one cubic span");
    let controls = sketch.splines()[0].points.clone();

    // An interior control leaves: cubic becomes quadratic, the spline lives.
    sketch.delete_point_cascade(controls[1]);
    assert_eq!(sketch.splines().len(), 1);
    assert_eq!(
        sketch.splines()[0].points,
        vec![controls[0], controls[2], controls[3]]
    );

    // An END leaves: the control behind it inherits the job and is promoted out of Construction,
    // which is the only thing keeping it off `prune_orphan_centers`' sweep.
    sketch.delete_point_cascade(controls[0]);
    assert_eq!(sketch.splines()[0].points, vec![controls[2], controls[3]]);
    let role = |id| {
        sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.lifetime)
    };
    assert_eq!(role(controls[2]), Some(PointLifetime::Freestanding));

    // Two ends is the floor: the last delete has no curve left to simplify to.
    sketch.delete_point_cascade(controls[2]);
    assert!(sketch.splines().is_empty());
}

#[test]
fn deleting_a_fit_point_simplifies_the_spline_and_a_closed_one_opens_no_lower_than_three() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(-4, -2),
                SketchPoint::new(4, -2),
                SketchPoint::new(0, 5),
            ],
            true,
        )
        .expect("three distinct points make a closed fit spline");
    let fits = sketch.splines()[0].points.clone();

    sketch.delete_point_cascade(fits[0]);
    assert!(
        sketch.splines().is_empty(),
        "a closed loop of two is not a loop"
    );

    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(1, 2),
                SketchPoint::new(4, 6),
                SketchPoint::new(8, 2),
            ],
            false,
        )
        .expect("valid open fit spline");
    let fits = sketch.splines()[0].points.clone();

    sketch.delete_point_cascade(fits[1]);
    assert_eq!(sketch.splines()[0].points, vec![fits[0], fits[2]]);
    assert_eq!(
        sketch.faces(ctx(16)).len(),
        0,
        "an open two-point spline is no face"
    );

    sketch.delete_point_cascade(fits[0]);
    assert!(sketch.splines().is_empty());
}

/// Every fit point is born with a handle, and each is minted where the curve ALREADY bends — so a
/// spline with its full set of levers draws exactly the curve it would have drawn with none.
/// Dragging one then bends the curve toward it.
#[test]
fn a_tangent_handle_starts_on_the_natural_tangent_and_steers_once_moved() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit_points = sketch.splines()[0].points.clone();
    let drawn = |sketch: &Sketch| {
        let held = sketch.splines()[0].clone();
        sketch.spline_candidate(&held).expect("the spline draws")
    };
    // What the same points draw with NO tangent authored anywhere — the curve the levers have to
    // leave alone. Not bit-exact: a handle's position round-trips through a point's sub-voxel
    // storage, so "changes nothing" is a claim about the curve, not about the last bit of it.
    let bare = ::parametric::sketch::fit_point_spline(
        &[[0.0, 0.0], [4.0, 4.0], [8.0, 0.0]],
        &[None, None, None],
        false,
    )
    .expect("the natural interpolant draws");
    let born = drawn(&sketch);
    for (was, now) in bare.pieces.iter().zip(&born.pieces) {
        for (was, now) in was.control.iter().zip(&now.control) {
            assert!(
                (was[0] - now[0]).abs() < 1.0e-6 && (was[1] - now[1]).abs() < 1.0e-6,
                "the natural tangent is what the levers were minted holding: {was:?} vs {now:?}"
            );
        }
    }
    let handle = sketch
        .tangent_handle_of(fit_points[0])
        .expect("every fit point is born with a lever")
        .forward;
    assert_eq!(
        sketch
            .points()
            .iter()
            .find(|point| point.id == handle)
            .expect("the handle is a real point")
            .lifetime,
        PointLifetime::CurveAnchored
    );

    // Steering the start tangent straight up puts the curve above the chord it used to leave on.
    assert!(sketch
        .move_point(handle, SketchPoint::new(0, 3), ctx(16))
        .expect("the handle drag is answered"));
    let leaving = drawn(&sketch).pieces[0].point_at(0.2);
    assert!(
        leaving[0] < leaving[1],
        "the curve should leave upward, not along the chord: {leaving:?}"
    );

    // A lever is not the author's to remove: the delete cascade declines it outright, rather than
    // dropping the fit point back to its natural tangent.
    let steered = sketch.point_in_plane(handle).expect("the handle stands");
    sketch.delete_point_cascade(handle);
    assert_eq!(sketch.splines()[0].tangents.len(), fit_points.len());
    assert_eq!(sketch.splines()[0].points, fit_points);
    assert_eq!(
        sketch.point_in_plane(handle),
        Some(steered),
        "the handle did not even move"
    );
}

/// The lever is double-sided and symmetric: its midpoint IS the fit point, and its two arms are
/// the same length. That holds when it is minted and it holds after a drag, because only the
/// forward arm is authored — the back one is put back on the mirror after every edit.
#[test]
fn a_tangent_lever_is_symmetric_about_the_fit_point_it_steers() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit = sketch.splines()[0].points[1];
    let lever = sketch.tangent_handle_of(fit).expect("a lever");
    let mirrored = |sketch: &Sketch| {
        let anchor = sketch.point_in_plane(fit).expect("the fit point stands");
        let forward = sketch
            .point_in_plane(lever.forward)
            .expect("the forward arm stands");
        let backward = sketch
            .point_in_plane(lever.backward)
            .expect("the back arm stands");
        let want = [2.0 * anchor[0] - forward[0], 2.0 * anchor[1] - forward[1]];
        assert!(
            (backward[0] - want[0]).abs() < 1.0e-6 && (backward[1] - want[1]).abs() < 1.0e-6,
            "the back arm is off the mirror: {backward:?} should be {want:?}"
        );
    };
    mirrored(&sketch);

    assert!(sketch
        .move_point(lever.forward, SketchPoint::new(5, 7), ctx(16))
        .expect("the handle drag is answered"));
    mirrored(&sketch);
}

/// The back arm's position is re-derived after every edit, so a relation on it would be met by
/// the solve and then silently overwritten. The door declines it instead — and still accepts one
/// on the FORWARD arm, which is authored and stays where a constraint puts it.
#[test]
fn a_constraint_is_declined_on_the_mirrored_arm_and_taken_on_the_authored_one() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit = sketch.splines()[0].points[0];
    let lever = sketch.tangent_handle_of(fit).expect("a lever");
    let elsewhere = sketch.add_free_point(SketchPoint::new(-6, -6));

    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Coincident {
                first: lever.backward.min(elsewhere),
                second: lever.backward.max(elsewhere),
            },
            ctx(16),
        ),
        Err(ConstraintRefusal::MirroredTangentArm)
    );

    let other = sketch.add_free_point(SketchPoint::new(-9, -9));
    assert!(sketch
        .add_constraint(
            ConstraintKind::Coincident {
                first: lever.forward.min(other),
                second: lever.forward.max(other),
            },
            ctx(16),
        )
        .is_ok());
}

/// Grabbing the BACK arm steers the FORWARD one. The two ends name one quantity, so a drag on
/// either has to land the same lever — and the arm the author grabbed has to end up under their
/// cursor, not at the mirror of it.
#[test]
fn dragging_the_back_arm_of_a_lever_steers_the_front_one() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit = sketch.splines()[0].points[1];
    let lever = sketch.tangent_handle_of(fit).expect("a lever");
    let anchor = sketch.point_in_plane(fit).expect("the fit point stands");

    let grabbed_to = [anchor[0] - 3.0, anchor[1] - 1.0];
    assert!(sketch
        .move_point(
            lever.backward,
            SketchPoint::from_continuous(grabbed_to[0], grabbed_to[1]),
            ctx(16)
        )
        .expect("the back-arm drag is answered"));

    let backward = sketch
        .point_in_plane(lever.backward)
        .expect("the back arm stands");
    assert!(
        (backward[0] - grabbed_to[0]).abs() < 1.0e-6
            && (backward[1] - grabbed_to[1]).abs() < 1.0e-6,
        "the arm the author grabbed did not follow the cursor: {backward:?}"
    );
    let forward = sketch
        .point_in_plane(lever.forward)
        .expect("the forward arm stands");
    let want = [
        2.0 * anchor[0] - grabbed_to[0],
        2.0 * anchor[1] - grabbed_to[1],
    ];
    assert!(
        (forward[0] - want[0]).abs() < 1.0e-6 && (forward[1] - want[1]).abs() < 1.0e-6,
        "the front arm should have taken the mirror of the grab: {forward:?} vs {want:?}"
    );
}

/// A handle belongs to the curve it bends, so it goes when the curve does — and a fit point that
/// is deleted takes only its OWN handle.
#[test]
fn tangent_handles_go_when_their_fit_point_or_their_spline_does() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
                SketchPoint::new(12, 4),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit_points = sketch.splines()[0].points.clone();
    let first = sketch
        .tangent_handle_of(fit_points[1])
        .expect("a lever on the second fit point");
    let second = sketch
        .tangent_handle_of(fit_points[2])
        .expect("a lever on the third fit point");
    // Four fit points, each with a two-armed lever.
    assert_eq!(sketch.points().len(), 12);

    sketch.delete_point_cascade(fit_points[1]);
    assert!(
        first
            .arms()
            .iter()
            .all(|arm| sketch.points().iter().all(|point| point.id != *arm)),
        "the deleted fit point's lever has no fit point left to steer"
    );
    assert_eq!(sketch.splines()[0].tangents.len(), 3);

    sketch.delete_spline(sketch.splines()[0].id);
    assert!(
        second
            .arms()
            .iter()
            .all(|arm| sketch.points().iter().all(|point| point.id != *arm)),
        "the spline took its remaining levers with it"
    );
}

/// A tangent is the vector from a fit point to its handle, so a solve that moves the FIT POINT
/// has to bring the handle along. Left behind, the handle would silently re-aim the tangent as a
/// side effect of solving something else — the author's steering rotated by a constraint that
/// never mentioned it.
#[test]
fn a_solve_that_moves_a_fit_point_carries_its_tangent_handle() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit = sketch.splines()[0].points[0];
    let handle = sketch.tangent_handle_of(fit).expect("a lever").forward;
    let tangent = |sketch: &Sketch| {
        let at = sketch.point_in_plane(fit).expect("the fit point stands");
        let held = sketch.point_in_plane(handle).expect("the handle stands");
        [held[0] - at[0], held[1] - at[1]]
    };
    let before = tangent(&sketch);

    // A constraint that says nothing about the handle, and moves the point it steers.
    let elsewhere = sketch.add_free_point(SketchPoint::new(-6, -6));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                first: fit.min(elsewhere),
                second: fit.max(elsewhere),
            },
            ctx(16),
        )
        .expect("the coincidence is asserted");

    let moved = sketch.point_in_plane(fit).expect("the fit point stands");
    assert!(
        (moved[0] - 0.0).abs() > 0.5 || (moved[1] - 0.0).abs() > 0.5,
        "the solve has to have moved the fit point for this to test anything: {moved:?}"
    );
    let after = tangent(&sketch);
    assert!(
        (before[0] - after[0]).abs() < 1.0e-6 && (before[1] - after[1]).abs() < 1.0e-6,
        "the tangent was re-aimed by a constraint that never named it: {before:?} became {after:?}"
    );
}

/// The other half of the carry rule: a handle a constraint reached is the constraint's to place,
/// and the carry must keep its hands off it — even when the solve's answer was "stay put".
///
/// A pinned handle does not move while its anchor does, which is exactly what an "it did not move,
/// so nobody claimed it" test cannot tell apart from a loose one.
#[test]
fn a_pinned_tangent_handle_stays_pinned_when_its_fit_point_moves() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 4),
                SketchPoint::new(8, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let fit = sketch.splines()[0].points[0];
    let handle = sketch.tangent_handle_of(fit).expect("a lever").forward;

    // Pin the handle to a point of its own, so a relation names it directly.
    let stands = sketch.point_in_plane(handle).expect("the handle stands");
    let pin = sketch.add_free_point(SketchPoint::from_continuous(stands[0], stands[1]));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                first: handle.min(pin),
                second: handle.max(pin),
            },
            ctx(16),
        )
        .expect("the pin is asserted");

    // Now move the ANCHOR with a constraint that says nothing about the handle.
    let elsewhere = sketch.add_free_point(SketchPoint::new(-6, -6));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                first: fit.min(elsewhere),
                second: fit.max(elsewhere),
            },
            ctx(16),
        )
        .expect("the coincidence is asserted");

    let moved = sketch.point_in_plane(fit).expect("the fit point stands");
    assert!(
        (moved[0] - 0.0).abs() > 0.5 || (moved[1] - 0.0).abs() > 0.5,
        "the solve has to have moved the fit point for this to test anything: {moved:?}"
    );
    let held = sketch.point_in_plane(handle).expect("the handle stands");
    let pinned = sketch.point_in_plane(pin).expect("the pin stands");
    assert!(
        (held[0] - pinned[0]).abs() < 1.0e-6 && (held[1] - pinned[1]).abs() < 1.0e-6,
        "the carry dragged a handle off the pin a constraint put it on: \
         {held:?} against {pinned:?}"
    );
}

/// The control frame is derived from the point list, so it names controls in leg order and names
/// nothing at all for a fit spline, whose points are on the curve rather than off it.
#[test]
fn only_a_control_point_spline_reports_a_frame_and_it_runs_in_control_order() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(-6, 0),
                SketchPoint::new(-3, 4),
                SketchPoint::new(0, 0),
            ],
            false,
        )
        .expect("a valid open fit spline");
    assert!(sketch.control_polygons().is_empty());

    let spline = sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(2, 5),
            SketchPoint::new(6, 5),
            SketchPoint::new(8, 0),
        ])
        .expect("four controls make one cubic span");
    let index = sketch
        .splines()
        .iter()
        .position(|held| held.id == spline)
        .expect("the spline stands");
    let controls = sketch.splines()[index].points.clone();
    assert_eq!(sketch.control_polygons(), vec![(spline, controls.clone())]);

    // A closed frame's last leg returns to the first control, so the polygon loops as the curve
    // it carries does.
    sketch.splines[index].closed = true;
    let looped = sketch.control_polygons();
    assert_eq!(looped[0].1.len(), controls.len() + 1);
    assert_eq!(looped[0].1.last(), controls.first());
}

/// A spline has no perpendicular to offset along, so grabbing its body TRANSLATES it: every point
/// moves by the same displacement and the shape is untouched.
#[test]
fn dragging_a_splines_body_carries_every_point_by_the_same_step() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let spline = sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(4, 6),
                SketchPoint::new(10, 2),
            ],
            false,
        )
        .expect("a valid open fit spline");
    let before: Vec<_> = sketch
        .points()
        .iter()
        .map(|point| (point.id, point.at.in_plane()))
        .collect();

    assert!(sketch
        .translate_curve(SketchCurve::Spline(spline), [3.0, -2.0], ctx(16))
        .expect("the translate is answered"));

    for (id, was) in before {
        let now = sketch
            .points()
            .iter()
            .find(|point| point.id == id)
            .expect("every point survives a translation")
            .at
            .in_plane();
        assert!(
            (now[0] - (was[0] + 3.0)).abs() < 1e-6 && (now[1] - (was[1] - 2.0)).abs() < 1e-6,
            "{was:?} went to {now:?}"
        );
    }

    // A handle travels with the spline: a translation that left it behind would re-aim the
    // tangent by exactly the distance the curve moved.
    let steered = sketch.splines()[0].points[0];
    let handle = sketch.tangent_handle_of(steered).expect("a lever").forward;
    let reach = |sketch: &Sketch| {
        let at = sketch
            .point_in_plane(steered)
            .expect("the fit point stands");
        let held = sketch.point_in_plane(handle).expect("the handle stands");
        [held[0] - at[0], held[1] - at[1]]
    };
    let tangent = reach(&sketch);
    assert!(sketch
        .translate_curve(SketchCurve::Spline(spline), [-5.0, 1.0], ctx(16))
        .expect("the translate is answered"));
    let after = reach(&sketch);
    assert!(
        (tangent[0] - after[0]).abs() < 1.0e-6 && (tangent[1] - after[1]).abs() < 1.0e-6,
        "the handle stayed home: {tangent:?} became {after:?}"
    );

    // Only a spline translates. A segment already has a gesture that means something else.
    let tail = sketch.add_free_point(SketchPoint::new(-9, -9));
    let head = sketch.add_free_point(SketchPoint::new(-9, -3));
    let segment = sketch.connect(tail, head).expect("a segment");
    assert!(!sketch
        .translate_curve(SketchCurve::Segment(segment), [1.0, 1.0], ctx(16))
        .expect("the translate is answered"));
}

/// A point's lifetime rode the `role` field, spelled with a curve's role names, until the two
/// quantities were split. Documents written then must still say what they meant — a handle that
/// loaded as Freestanding would stop being swept and litter a dot for every ellipse ever drawn.
#[test]
fn a_points_old_role_loads_as_the_lifetime_it_always_was() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_ellipse(
            SketchPoint::new(2, 3),
            SketchPoint::new(8, 3),
            SketchPoint::new(2, 7),
        )
        .expect("valid ellipse");
    let free = sketch.add_free_point(SketchPoint::new(9, 9));

    let mut raw = serde_json::to_value(&sketch).expect("sketch serializes");
    for point in raw["points"]
        .as_array_mut()
        .expect("points is an array")
        .iter_mut()
    {
        let old = point
            .as_object_mut()
            .expect("a point is an object")
            .remove("lifetime")
            .expect("the point wrote its lifetime");
        let old = match old.as_str().expect("a lifetime is a string") {
            "CurveAnchored" => "Construction",
            _ => "Real",
        };
        point["role"] = serde_json::json!(old);
    }

    let loaded: Sketch = serde_json::from_value(raw).expect("an older document loads");
    assert_eq!(loaded, sketch);
    let lifetime_of = |id| {
        loaded
            .points()
            .iter()
            .find(|point| point.id == id)
            .map(|point| point.lifetime)
    };
    assert_eq!(lifetime_of(free), Some(PointLifetime::Freestanding));
    assert_eq!(
        lifetime_of(loaded.ellipses()[0].center),
        Some(PointLifetime::CurveAnchored)
    );
}

#[test]
fn spline_points_retarget_and_invalid_loaded_splines_repair_atomically() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(1, 2),
                SketchPoint::new(4, 6),
                SketchPoint::new(8, 2),
            ],
            false,
        )
        .expect("valid open fit spline");
    sketch.retarget_density(16, 32);
    // The FIT points; the levers' arms rescale alongside them and are read by their own test.
    let positions: Vec<_> = sketch.splines()[0]
        .points
        .iter()
        .filter_map(|id| sketch.point_in_plane(*id))
        .collect();
    assert_eq!(positions, vec![[2.0, 4.0], [8.0, 12.0], [16.0, 4.0]]);

    let mut raw = serde_json::to_value(&sketch).expect("spline serializes");
    raw["splines"][0]["points"][1] = serde_json::json!(EntityId::MAX);
    let mut loaded: Sketch = serde_json::from_value(raw).expect("structural load succeeds");
    assert_eq!(loaded.repair(ctx(32)), 1);
    assert!(loaded.splines().is_empty());
}
