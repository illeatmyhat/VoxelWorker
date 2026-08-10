//! Constraint entities and the continuous solve.
use super::ctx;

use super::*;

/// A segment from `(0,0)` to `(10,4)` — slanted, so `Horizontal` has something to do.
fn slanted() -> (Sketch, EntityId, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 4));
    let segment = sketch.connect(tail, head).expect("a fresh segment");
    (sketch, tail, head, segment)
}

pub(super) fn position(sketch: &Sketch, id: EntityId) -> [f64; 2] {
    sketch
        .points()
        .iter()
        .find(|point| point.id == id)
        .expect("the point")
        .at
        .in_plane()
}

fn add_and_solve_tangent(sketch: &mut Sketch, kind: ConstraintKind) {
    sketch
        .add_constraint(kind, ctx(16))
        .expect("document adapter accepts tangent");
    sketch
        .solve(ctx(16))
        .expect("document adapter re-solves tangent");
}

/// With nothing asserted, every coordinate is free. This is the baseline "fully constrained" is
/// measured against, and it is read off the store rather than from a solve with no residuals.
#[test]
fn an_unconstrained_sketch_is_all_freedom() {
    let (sketch, _, _, _) = slanted();
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        4,
        "two points, two axes each"
    );
    assert!(sketch.constraints().is_empty());
}

/// `Fix` pins both of a point's coordinates, so it removes exactly two freedoms and moves nothing.
#[test]
fn a_fix_pins_two_freedoms_and_moves_nothing() {
    let (mut sketch, tail, head, _) = slanted();
    let before = position(&sketch, tail);
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("nothing else is asserted, so it cannot conflict");
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        2,
        "the head is still free"
    );
    assert_eq!(
        position(&sketch, tail),
        before,
        "a fix does not move a point"
    );
    assert_eq!(position(&sketch, head), [10.0, 4.0], "nor its neighbor");
}

#[test]
fn quantize_places_a_point_on_its_exact_lattice_and_survives_a_drag() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let point = sketch.add_free_point(SketchPoint::from_continuous(2.6, -1.6));
    sketch
        .add_constraint(
            ConstraintKind::Quantize {
                point,
                pitch: SketchLength::from_continuous(2.0),
                phase: SketchLength::from_continuous(0.5),
            },
            ctx(16),
        )
        .expect("a free point can land on the lattice");
    let at = position(&sketch, point);
    assert!((at[0] - 2.5).abs() < 1e-6);
    assert!((at[1] + 1.5).abs() < 1e-6);
    sketch
        .move_point(point, SketchPoint::from_continuous(5.1, 3.2), ctx(16))
        .expect("quantization remains a standing assertion");
    let at = position(&sketch, point);
    assert!((at[0] - 4.5).abs() < 1e-6, "x snapped after drag: {at:?}");
    assert!((at[1] - 2.5).abs() < 1e-6, "y snapped after drag: {at:?}");
    assert_eq!(sketch.degrees_of_freedom(ctx(16)).unwrap(), 0);
}

#[test]
fn density_retarget_keeps_fix_targets_and_voxel_quantization_sources_consistent() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let fixed = sketch.add_free_point(SketchPoint::new(4, 6));
    let quantized = sketch.add_free_point(SketchPoint::from_continuous(2.4, 3.6));
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: fixed,
                at: SketchPoint::new(4, 6),
            },
            ctx(16),
        )
        .unwrap();
    sketch
        .add_constraint(
            ConstraintKind::Quantize {
                point: quantized,
                pitch: SketchLength::retained_voxels(1),
                phase: SketchLength::retained_voxels(0),
            },
            ctx(16),
        )
        .unwrap();
    sketch.retarget_density(16, 32);
    sketch.solve(ctx(32)).expect("retargeted assertions agree");
    assert_eq!(position(&sketch, fixed), [8.0, 12.0]);
    let at = position(&sketch, quantized);
    assert!((at[0] - at[0].round()).abs() < 1e-6);
    assert!((at[1] - at[1].round()).abs() < 1e-6);
    let ConstraintKind::Quantize { pitch, phase, .. } = sketch.constraints()[1].kind else {
        panic!("quantize retained");
    };
    assert_eq!(pitch.value(), 1.0, "one voxel stays one voxel");
    assert_eq!(phase.value(), 0.0);
}

/// `Horizontal` levels a segment, and the least-squares solve moves the drawing **as little as it
/// can**: neither end is privileged, so they meet in the middle rather than one snapping to the
/// other. That is the property that makes a solve feel like a nudge.
#[test]
fn horizontal_levels_a_segment_by_meeting_in_the_middle() {
    let (mut sketch, tail, head, segment) = slanted();
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("a lone constraint always holds");

    let (a, b) = (position(&sketch, tail), position(&sketch, head));
    assert!((a[1] - b[1]).abs() < 1e-6, "level: {a:?} to {b:?}");
    assert!((a[1] - 2.0).abs() < 1e-6, "the tail rose halfway: {a:?}");
    assert!((b[1] - 2.0).abs() < 1e-6, "the head fell halfway: {b:?}");
    assert_eq!(a[0], 0.0, "nothing pulled sideways");
    assert_eq!(b[0], 10.0);
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        3,
        "one assertion, one freedom"
    );
}

/// A constraint holds through a DRAG, not merely at the moment it was asserted. The grabbed end
/// goes exactly where the hand put it and the free end follows to keep the segment level — the
/// drag is a pin, so the rest of the drawing moves around it.
///
/// `move_point` re-solves the standing system. Writing the coordinate alone would leave every
/// assertion standing only until a drag of the geometry it names tilted it back off.
#[test]
fn a_level_segment_stays_level_when_an_end_is_dragged() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(40, 0));
    let segment = sketch.connect(tail, head).expect("a fresh segment");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("a lone level on a lone segment");

    assert!(sketch
        .move_point(tail, SketchPoint::new(-7, -18), ctx(16))
        .expect("evaluation context"));

    let (dragged, follower) = (position(&sketch, tail), position(&sketch, head));
    assert!(
        (dragged[0] + 7.0).abs() < 1e-6 && (dragged[1] + 18.0).abs() < 1e-6,
        "the hand holds the grabbed end: {dragged:?}"
    );
    assert!(
        (dragged[1] - follower[1]).abs() < 1e-6,
        "still level after the drag: {dragged:?} to {follower:?}"
    );
    assert!(
        (follower[0] - 40.0).abs() < 1e-6,
        "the follower moves only as far as the level asks: {follower:?}"
    );
}

/// A drag the standing constraints cannot admit leaves the drawing alone. `Fix` says the point is
/// where it is; the hand does not outrank it, and the vertex sits still rather than moving and
/// being hauled back.
#[test]
fn a_fixed_point_does_not_move_under_the_hand() {
    let (mut sketch, tail, head, _) = slanted();
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("nothing else is asserted");

    // Near-exactly, not exactly: the drag is now a PULL that the standing system takes back, so
    // the point is re-solved to its fixed place rather than the whole move being discarded, and a
    // re-solved coordinate carries the solver's dust.
    assert!(sketch
        .move_point(tail, SketchPoint::new(25, 25), ctx(16))
        .expect("evaluation context"));
    let held = position(&sketch, tail);
    assert!(
        held[0].abs() < 1e-9 && held[1].abs() < 1e-9,
        "the fix wins: {held:?}"
    );
    assert_eq!(
        position(&sketch, head),
        [10.0, 4.0],
        "and nothing else moved"
    );
}

/// **Geometry the constraint does not name must not be able to break it.** A lone segment gets
/// leveled the same whether it is alone on the plane or surrounded by a drawing.
///
/// The verdict has to be read from the RESIDUALS, never from the solver's `SolveOutcome`: that
/// flag's residual test is absolute while its step test is relative to the size of the whole
/// parameter vector, so free points elsewhere in the sketch — contributing nothing to the residual
/// and everything to the vector's length — make the step test fire first. It then reports
/// `Stalled` with the constraint satisfied to about 1e-10 voxels, and a `Stalled` read as
/// unsatisfiable refuses the assertion. **Two** unrelated points are enough, which is to say every
/// real drawing.
#[test]
fn free_geometry_the_constraint_never_names_cannot_refuse_it() {
    for bystanders in 0..6 {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        for index in 0..bystanders {
            sketch.add_free_point(SketchPoint::new(37 * (index + 1), 53 * (index + 1)));
        }
        let tail = sketch.add_free_point(SketchPoint::new(28, 0));
        let head = sketch.add_free_point(SketchPoint::new(78, 6));
        let segment = sketch.connect(tail, head).expect("a fresh segment");
        assert!(
            sketch
                .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
                .is_ok(),
            "{bystanders} unrelated free point(s) refused a lone level"
        );
        let (a, b) = (position(&sketch, tail), position(&sketch, head));
        assert!((a[1] - b[1]).abs() < 1e-6, "level: {a:?} to {b:?}");
    }
}

/// A distance dimension is met exactly, and the pair moves symmetrically for the same reason.
#[test]
fn a_distance_dimension_is_met() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: tail,
                to: head,
                length: SketchLength::new(6),
            }),
            ctx(16),
        )
        .expect("two free points can always be six apart");

    let (a, b) = (position(&sketch, tail), position(&sketch, head));
    let span = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    assert!((span - 6.0).abs() < 1e-6, "six apart, got {span}");
    assert!((a[0] - 2.0).abs() < 1e-6, "each end came in by two: {a:?}");
    assert!((b[0] - 8.0).abs() < 1e-6, "{b:?}");
}

/// **An angle dimension turns one segment onto the other by the number the author wrote.**
///
/// It also states the two things the family exists for: the value is an angle rather than a
/// length, and a density re-target leaves it exactly where it was, because an angle has no block
/// term to re-target.
#[test]
fn an_angle_dimension_turns_one_segment_onto_the_other() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let along = sketch.add_free_point(SketchPoint::new(10, 0));
    let up = sketch.add_free_point(SketchPoint::new(10, 10));
    // The base is pinned, so the only thing free to move is the arm the angle is measured to.
    for (point, at) in [
        (corner, SketchPoint::new(0, 0)),
        (along, SketchPoint::new(10, 0)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("pinning the base");
    }
    let base = sketch.connect(corner, along).expect("the base");
    let arm = sketch.connect(corner, up).expect("the arm");

    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: AngleArm::Segment { segment: base },
                second: AngleArm::Segment { segment: arm },
                degrees: AngleMeasurement::from_degrees(30),
                corner: AngleCorner::Between,
            }),
            ctx(16),
        )
        .expect("a free arm can always stand thirty degrees off a pinned base");

    let (at_corner, at_up) = (position(&sketch, corner), position(&sketch, up));
    let turn = (at_up[1] - at_corner[1])
        .atan2(at_up[0] - at_corner[0])
        .to_degrees();
    assert!((turn - 30.0).abs() < 1e-6, "thirty degrees off, got {turn}");

    // A quarter turn is Perpendicular's claim written as a number, and it solves to the same
    // place — which is the whole argument for one relation covering both.
    let mut squared = Sketch::empty(PlaneAxis::Z);
    let corner = squared.add_free_point(SketchPoint::new(0, 0));
    let along = squared.add_free_point(SketchPoint::new(10, 0));
    let up = squared.add_free_point(SketchPoint::new(10, 10));
    for (point, at) in [
        (corner, SketchPoint::new(0, 0)),
        (along, SketchPoint::new(10, 0)),
    ] {
        squared
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("pinning the base");
    }
    let base = squared.connect(corner, along).expect("the base");
    let arm = squared.connect(corner, up).expect("the arm");
    squared
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: AngleArm::Segment { segment: base },
                second: AngleArm::Segment { segment: arm },
                degrees: AngleMeasurement::from_degrees(90),
                corner: AngleCorner::Between,
            }),
            ctx(16),
        )
        .expect("ninety is a right angle");
    let (at_corner, at_up) = (position(&squared, corner), position(&squared, up));
    assert!(
        (at_up[0] - at_corner[0]).abs() < 1e-6,
        "square over the corner: {at_up:?}"
    );

    // A segment cannot stand at an angle to itself, and neither can it to one that has died.
    assert_eq!(
        squared.add_constraint(
            ConstraintKind::Dimension(Dimension::Angle {
                first: AngleArm::Segment { segment: base },
                second: AngleArm::Segment { segment: base },
                degrees: AngleMeasurement::from_degrees(45),
                corner: AngleCorner::Between,
            }),
            ctx(16),
        ),
        Err(ConstraintRefusal::Impossible)
    );

    // And the number survives a density re-target untouched: an angle has no density.
    let before = squared.constraints().to_vec();
    squared.retarget_density(16, 32);
    let angle = |from: &[Constraint]| {
        from.iter()
            .find_map(|held| match held.kind {
                ConstraintKind::Dimension(Dimension::Angle { degrees, .. }) => Some(degrees),
                _ => None,
            })
            .expect("the angle is still there")
    };
    assert_eq!(angle(&before), angle(squared.constraints()));
    assert_eq!(
        angle(squared.constraints()),
        AngleMeasurement::from_degrees(90)
    );
}

/// **Unsatisfiable is refused, and refusing changes nothing.** The trial runs on a copy, so the
/// drawing is where it was rather than where a failed solve pushed it.
#[test]
fn a_contradictory_constraint_is_refused_and_leaves_the_drawing_alone() {
    let (mut sketch, tail, head, _) = slanted();
    let mut pins = Vec::new();
    for (point, at) in [
        (tail, SketchPoint::new(0, 0)),
        (head, SketchPoint::new(10, 4)),
    ] {
        pins.push(
            sketch
                .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
                .expect("pinning each end in turn is consistent"),
        );
    }
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        0,
        "fully constrained"
    );
    let before: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();

    // The ends are pinned about 10.77 apart. Five is not a distance they can be.
    let refusal = sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: tail,
                to: head,
                length: SketchLength::new(5),
            }),
            ctx(16),
        )
        .expect_err("five is not a distance those pins allow");
    // And it NAMES what it fights: releasing either pin would let the distance hold, so
    // leave-one-out finds both, and the author is pointed at something they can act on.
    assert_eq!(
        refusal,
        ConstraintRefusal::Unsatisfiable {
            fights: pins.clone()
        }
    );
    assert_eq!(refusal.culprits(), pins);
    assert_eq!(sketch.constraints().len(), 2, "it was not kept");
    let after: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();
    assert_eq!(before, after, "nor did the failed trial move anything");
}

/// **Redundant is accepted and flagged.** An assertion the geometry already implies is insurance
/// against a later edit, so it is marked rather than refused.
///
/// Redundant is not the same as DUPLICATE, and the difference is exactly what this fixture shows:
/// two pinned endpoints already put the segment level, so `Horizontal` adds no information — but
/// it is a different claim, made about different entities, and it survives a later edit that
/// releases a pin. A literal second `Horizontal` would be refused instead.
#[test]
fn a_redundant_constraint_is_kept_and_flagged() {
    let (mut sketch, tail, head, segment) = slanted();
    let pinned_tail = sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("the first pin");
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: head,
                at: SketchPoint::new(10, 0),
            },
            ctx(16),
        )
        .expect("the second pin, level with the first");
    let implied = sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("already true, so redundant rather than refused");

    let flagged = |id: EntityId| {
        sketch
            .constraints()
            .iter()
            .find(|constraint| constraint.id == id)
            .expect("the constraint")
            .redundant
    };
    assert!(!flagged(pinned_tail), "the first raised the rank");
    assert!(flagged(implied), "the last added no information");
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        0,
        "and it took no freedom away, which is what redundant MEANS — the pins took them all"
    );
}

/// The delete cascade reaches constraints: a constraint never outlives the geometry it names, so a
/// residual row can never refer to a shape that is gone.
#[test]
fn deleting_geometry_takes_its_constraints_with_it() {
    let (mut sketch, tail, _, segment) = slanted();
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("a lone fix");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("and a level");
    assert_eq!(sketch.constraints().len(), 2);

    // The line takes its two ends with it — nothing else draws them — so BOTH constraints go:
    // the level names the segment, and the fix names an end that no longer exists.
    sketch.delete_segment(segment);
    assert!(
        sketch.constraints().is_empty(),
        "the level went with the line and the fix went with the end it named"
    );
}

/// Load repair erases a constraint naming geometry the store does not hold, and counts it — the
/// same policy every other entity gets: erase the invalid, never fail the load.
#[test]
fn repair_erases_a_constraint_that_names_nothing() {
    let (mut sketch, _, _, _) = slanted();
    sketch.constraints_mut_for_test().push(Constraint {
        id: 900,
        kind: ConstraintKind::Fix {
            point: 901,
            at: SketchPoint::new(0, 0),
        },
        redundant: false,
        anchor: None,
    });
    sketch.constraints_mut_for_test().push(Constraint {
        id: 902,
        kind: ConstraintKind::Horizontal { segment: 903 },
        redundant: false,
        anchor: None,
    });
    assert_eq!(
        sketch.repair(ctx(16)),
        2,
        "both name entities that are not there"
    );
    assert!(sketch.constraints().is_empty());
}

/// A constraint naming absent geometry cannot be added in the first place — the store is checked
/// before the solver is, because an unknown id is a caller error and not a geometric one.
#[test]
fn a_constraint_naming_absent_geometry_is_refused() {
    let (mut sketch, tail, head, _) = slanted();
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Fix {
                point: 900,
                at: SketchPoint::new(0, 0),
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::UnknownEntity)
    );
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Horizontal { segment: 900 }, ctx(16)),
        Err(ConstraintRefusal::UnknownEntity)
    );
    // A negative length is no drawing's distance, so it never reaches the solver.
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: tail,
                to: head,
                length: SketchLength::new(-3),
            }),
            ctx(16)
        ),
        Err(ConstraintRefusal::Impossible)
    );
    assert!(sketch.constraints().is_empty());
}

/// A refused constraint burns no id: the next entity gets the number the refusal did not take.
#[test]
fn a_refusal_does_not_consume_an_id() {
    let (mut sketch, tail, _, _) = slanted();
    let _ = sketch.add_constraint(ConstraintKind::Horizontal { segment: 900 }, ctx(16));
    let next = sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("a lone fix");
    assert_eq!(next, 3, "after two points and a segment, the next id is 3");
}

/// Solving again from a solution changes nothing — the solve is idempotent, which is what lets it
/// run live during a drag without the drawing creeping.
#[test]
fn solving_a_solved_sketch_moves_nothing() {
    let (mut sketch, _, _, segment) = slanted();
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("a lone constraint");
    let settled: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();

    let report = sketch
        .solve(ctx(16))
        .expect("no fixed source")
        .expect("there is a constraint to solve");
    assert_eq!(report.outcome, SolveOutcome::Converged);
    let again: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();
    assert_eq!(settled, again);
}

/// The PRODUCER door the rail's constraint verbs go through: pure, so the caller
/// holds both drawings and the shell's one-transaction commit has something to commit.
#[test]
fn the_producer_door_asserts_without_touching_the_original() {
    let (sketch, tail, head, segment) = slanted();
    let before = SketchSolid::extrude(sketch, 3);
    let (after, id) = before
        .with_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("nothing else is asserted");

    assert_eq!(position(&before.sketch, head), [10.0, 4.0], "the original");
    // NEAR, not bit-identical. Horizontal is a claim about the two heights agreeing, and a solved
    // drawing agrees to the solver's precision rather than to the last bit — asking for the bit was
    // asking the search to arrive by one particular route.
    assert!(
        (position(&after.sketch, tail)[1] - position(&after.sketch, head)[1]).abs() < 1.0e-9,
        "the copy is leveled: {:?} vs {:?}",
        position(&after.sketch, tail),
        position(&after.sketch, head)
    );
    assert_eq!(after.sketch.constraints().len(), 1);
    assert_eq!(
        after
            .sketch
            .degrees_of_freedom(ctx(16))
            .expect("no fixed source"),
        3
    );

    // Releasing it stops the assertion without undoing what it did — the geometry stays level.
    let released = after.with_constraint_deleted(id);
    assert!(released.sketch.constraints().is_empty());
    assert_eq!(
        position(&released.sketch, tail)[1],
        position(&after.sketch, tail)[1],
        "releasing an assertion is not an undo"
    );
}

/// **Restating a dimension is release-and-assert, so the number goes in by the door that knows how
/// to refuse it** — and the id changes, which is the cost the caller pays for that.
#[test]
fn restating_a_dimension_moves_the_drawing_and_mints_a_new_id() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let span = |length: i64| Dimension::Span {
        from: tail,
        to: head,
        length: SketchLength::new(length),
    };
    let asserted = sketch
        .add_constraint(ConstraintKind::Dimension(span(10)), ctx(16))
        .expect("the drawing already stands ten apart");
    let before = SketchSolid::extrude(sketch, 3);

    let (after, restated) = before
        .with_dimension_restated(asserted, span(6), ctx(16))
        .expect("two points with nothing else asserted can be six apart");
    assert_ne!(restated, asserted, "a fresh assertion carries a fresh id");
    assert_eq!(
        after.sketch.constraints().len(),
        1,
        "the old one went, exactly one dimension remains"
    );
    assert_eq!(after.sketch.constraints()[0].id, restated);
    let (a, b) = (position(&after.sketch, tail), position(&after.sketch, head));
    assert!(
        ((b[0] - a[0]).hypot(b[1] - a[1]) - 6.0).abs() < 1e-6,
        "the drawing moved to the new number: {a:?} to {b:?}"
    );

    // And the original is untouched, because the door is pure like every other one here.
    let (a, b) = (
        position(&before.sketch, tail),
        position(&before.sketch, head),
    );
    assert!(((b[0] - a[0]).hypot(b[1] - a[1]) - 10.0).abs() < 1e-6);
}

/// **A label stays where the author dropped it** — through the restatement that mints a new id,
/// and through a density re-target that moves everything it annotates.
///
/// The anchor is the only piece of a dimension that is neither asserted nor derived, so it is the
/// one piece that can be quietly lost. Restating goes through release-and-assert, which throws the
/// old constraint away; a re-target rescales the drawing out from under a label stored in voxels.
#[test]
fn a_dimension_keeps_the_place_its_annotation_was_dropped() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let span = |length: i64| Dimension::Span {
        from: tail,
        to: head,
        length: SketchLength::new(length),
    };
    let dropped = [4.0, 7.0];
    let asserted = sketch
        .add_constraint_anchored(ConstraintKind::Dimension(span(10)), Some(dropped), ctx(16))
        .expect("the drawing already stands ten apart");
    assert_eq!(sketch.constraints()[0].anchor, Some(dropped));

    // A badge is not a dimension and drops nothing, which is what the `Option` is saying.
    let plain = sketch.add_constraint(ConstraintKind::Horizontal { segment: tail }, ctx(16));
    assert!(plain.is_err() || sketch.constraints()[1].anchor.is_none());

    let before = SketchSolid::extrude(sketch, 3);
    let (after, restated) = before
        .with_dimension_restated(asserted, span(6), ctx(16))
        .expect("two points with nothing else asserted can be six apart");
    assert_ne!(restated, asserted);
    assert_eq!(
        after.sketch.constraints()[0].anchor,
        Some(dropped),
        "changing the number is not moving the label"
    );

    // Doubled density doubles every voxel the drawing is written in, and the label is written in
    // the same voxels. Left alone it would end up half as far out as the author put it.
    let mut retargeted = after.sketch.clone();
    retargeted.retarget_density(16, 32);
    assert_eq!(
        retargeted.constraints()[0].anchor,
        Some([8.0, 14.0]),
        "the label rides with the drawing"
    );
}

/// **Moving a label is not restating a dimension.** The drop point authored the claim once; after
/// that it is where the number sits, and dragging it anywhere at all leaves the claim alone.
///
/// The two cases that could go wrong are the two where the drop point said something in the first
/// place. A run's annotation dropped above it asks for the width and beside it asks for the height,
/// so a hand that drags the width label round to the side must not silently start stating the
/// height. An angle's supplement is a different number from the angle, so a label dragged into a
/// corner of the other size must not restate it either — and the drawing must not move, because a
/// dimension the author never touched has nothing new to say to the solver.
#[test]
fn dragging_a_label_says_nothing_the_dimension_did_not_already_say() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(8, 6));
    let width = sketch
        .add_constraint_anchored(
            ConstraintKind::Dimension(Dimension::SpanAlong {
                from: tail,
                to: head,
                axis: InPlaneAxis::Across,
                length: SketchLength::new(8),
            }),
            // Above the run, which is the region that asks for the width.
            Some([4.0, 11.0]),
            ctx(16),
        )
        .expect("a run eight across states its width");

    // Dragged round to the LEFT of the run, which is the region a height is authored from.
    assert!(sketch.move_annotation(width, [-9.0, 3.0]));
    let held = sketch.constraints()[0];
    assert_eq!(held.anchor, Some([-9.0, 3.0]), "the label went");
    assert_eq!(
        held.kind,
        ConstraintKind::Dimension(Dimension::SpanAlong {
            from: tail,
            to: head,
            axis: InPlaneAxis::Across,
            length: SketchLength::new(8),
        }),
        "and the claim stayed exactly where it was",
    );

    // The same, for the one dimension whose stored corner the drop point chose.
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let corner = sketch.add_free_point(SketchPoint::new(0, 0));
    let along = sketch.add_free_point(SketchPoint::new(10, 0));
    let up = sketch.add_free_point(SketchPoint::new(0, 10));
    let first = sketch.connect(corner, along).expect("one arm");
    let second = sketch.connect(corner, up).expect("the other");
    let stated = sketch
        .add_constraint_anchored(
            ConstraintKind::Dimension(Dimension::Angle {
                first: AngleArm::Segment { segment: first },
                second: AngleArm::Segment { segment: second },
                degrees: AngleMeasurement::from_degrees(90),
                corner: AngleCorner::Between,
            }),
            Some([3.0, 3.0]),
            ctx(16),
        )
        .expect("two arms off one corner make a right angle");
    let drawn = sketch.points().to_vec();

    assert!(sketch.move_annotation(stated, [3.0, -3.0]));
    assert_eq!(
        sketch.constraints()[0].kind,
        ConstraintKind::Dimension(Dimension::Angle {
            first: AngleArm::Segment { segment: first },
            second: AngleArm::Segment { segment: second },
            degrees: AngleMeasurement::from_degrees(90),
            corner: AngleCorner::Between,
        }),
        "a label in the supplement's corner is not a request for the supplement",
    );
    assert_eq!(sketch.points(), drawn, "and no solve was asked for");
}

/// A number the drawing cannot reach is refused, and the refusal costs the author nothing — not
/// the geometry, and not the dimension they were editing. Poking the value in place could not
/// promise that; releasing and re-asserting a COPY can.
#[test]
fn a_restatement_the_drawing_cannot_reach_keeps_the_one_it_had() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    for (point, at) in [
        (tail, SketchPoint::new(0, 0)),
        (head, SketchPoint::new(10, 0)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("pinning a point that is already there");
    }
    let asserted = sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: tail,
                to: head,
                length: SketchLength::new(10),
            }),
            ctx(16),
        )
        .expect("ten apart is what the pins already say");
    let before = SketchSolid::extrude(sketch, 3);

    assert!(
        before
            .with_dimension_restated(
                asserted,
                Dimension::Span {
                    from: tail,
                    to: head,
                    length: SketchLength::new(4),
                },
                ctx(16),
            )
            .is_err(),
        "both ends are pinned ten apart, so four is not on offer"
    );
    assert_eq!(before.sketch.constraints().len(), 3, "nothing was released");
    assert!(
        before
            .sketch
            .constraints()
            .iter()
            .any(|held| held.id == asserted),
        "including the one being edited"
    );
}

/// A refusal at the producer door hands back nothing, so the shell cannot commit half an edit.
#[test]
fn the_producer_door_refuses_without_a_partial_result() {
    let (mut sketch, tail, _, _) = slanted();
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("the first assertion cannot conflict");
    let solid = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        solid
            .with_constraint(ConstraintKind::Horizontal { segment: 900 }, ctx(16))
            .err(),
        Some(ConstraintRefusal::UnknownEntity)
    );
    assert_eq!(solid.sketch.constraints().len(), 1, "unchanged");
}

/// The law the shell's multi-delete leans on: deleting a point CASCADES into the constraints that
/// named it, and deleting an already-gone constraint id is a no-op. Together those let one pass
/// delete a mixed selection — geometry first, assertions after — without asking which of the two
/// took each constraint.
#[test]
fn deleting_geometry_takes_its_constraints_and_the_id_stays_safe_to_delete() {
    let (sketch, tail, _, segment) = slanted();
    let before = SketchSolid::extrude(sketch, 3);
    let (asserted, id) = before
        .with_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("nothing else is asserted");

    let cascaded = asserted.with_point_deleted(tail);
    assert!(
        cascaded.sketch.segments().is_empty(),
        "the segment went with its endpoint"
    );
    assert!(
        cascaded.sketch.constraints().is_empty(),
        "and the assertion went with the segment"
    );

    let twice = cascaded.with_constraint_deleted(id);
    assert_eq!(
        twice, cascaded,
        "deleting a gone constraint changes nothing"
    );
}

/// One constraint of a kind per entity set. The second `Horizontal` on a segment already asserted
/// horizontal says nothing the first did not, so it is refused rather than kept and flagged — and
/// the store is left holding exactly one.
#[test]
fn the_same_assertion_twice_on_one_segment_is_refused() {
    let (mut sketch, _, _, segment) = slanted();
    let first = sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("the first assertion");
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Horizontal { segment }, ctx(16)),
        Err(ConstraintRefusal::AlreadyAsserted { existing: first }),
        "and it names the one already standing, so the answer is a lit badge not a hunt"
    );
    assert_eq!(sketch.constraints().len(), 1);
}

/// The comparison is on kind and geometry, never on the VALUE: a second `Fix` on a fixed point is
/// a re-fix — delete the first, assert the second — and not two live claims about one place.
#[test]
fn refixing_a_fixed_point_somewhere_else_is_still_a_duplicate() {
    let (mut sketch, tail, _, _) = slanted();
    let first = sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("the first assertion");
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(7, 7),
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::AlreadyAsserted { existing: first }),
        "a different place is still the same claim about the same point"
    );
}

/// A distance names an unordered PAIR, so asserting it the other way round is the same assertion.
#[test]
fn a_distance_is_the_same_assertion_in_either_direction() {
    let (mut sketch, tail, head, _) = slanted();
    let apart = |value: f64| {
        ConstraintKind::Dimension(Dimension::Span {
            from: tail,
            to: head,
            length: SketchLength::from_continuous(value),
        })
    };
    let first = sketch
        .add_constraint(apart(9.0), ctx(16))
        .expect("the first");
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: head,
                to: tail,
                length: SketchLength::from_continuous(4.0),
            }),
            ctx(16)
        ),
        Err(ConstraintRefusal::AlreadyAsserted { existing: first })
    );
}

/// **The case convergence cannot report.** `Horizontal` and `Vertical` on one segment DO have a
/// solution — the zero-length segment, where both residuals are exactly zero — so the solver
/// converges and calls it satisfied. The drawing has been destroyed rather than constrained, so
/// the trial refuses it and the segment keeps the length the first assertion left it.
#[test]
fn a_solve_that_collapses_geometry_is_refused() {
    let (mut sketch, tail, head, segment) = slanted();
    let level = sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("leveling a slanted segment is fine");
    let levelled = (position(&sketch, head)[0] - position(&sketch, tail)[0]).abs();
    assert!(levelled > 1.0, "still a line, {levelled} across");

    // Its own refusal, not Unsatisfiable: nothing here fights, the assertions AGREE on an answer
    // that happens to be a singularity. It names the geometry that would vanish and the assertion
    // whose release would save it.
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Vertical { segment }, ctx(16)),
        Err(ConstraintRefusal::WouldCollapse {
            entity: segment,
            implicated: vec![level],
        }),
        "level AND plumb is only meetable by deleting the segment"
    );
    assert_eq!(sketch.constraints().len(), 1, "the refusal kept nothing");
    assert_eq!(
        (position(&sketch, head)[0] - position(&sketch, tail)[0]).abs(),
        levelled,
        "and moved nothing"
    );
}

/// The collapse test is about what the NEW assertion did. A segment already standing at zero
/// length does not veto an unrelated assertion elsewhere in the drawing.
#[test]
fn already_collapsed_geometry_does_not_veto_the_rest() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(0, 0));
    let stub = sketch.connect(tail, head).expect("a zero-length segment");
    let far = sketch.add_free_point(SketchPoint::new(10, 4));
    let real = sketch.connect(head, far).expect("a segment with length");

    assert!(sketch.segments().iter().any(|seg| seg.id == stub));
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment: real }, ctx(16))
        .expect("the collapsed stub is not this assertion's doing");
}

/// The verdict does not depend on the drawing having been pre-solved onto its own assertions —
/// the property the witness reading exists to protect.
#[test]
fn redundancy_reads_the_same_on_a_solved_and_an_unsolved_drawing() {
    let flagged_for = |corner: [f64; 2]| {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let tail = sketch.add_free_point(SketchPoint::new(0, 0));
        let head = sketch.add_free_point(SketchPoint::from_continuous(corner[0], corner[1]));
        let segment = sketch
            .connect(tail, head)
            .expect("two distinct points join");
        for (point, at) in [
            (tail, SketchPoint::new(0, 0)),
            (head, SketchPoint::new(10, 0)),
        ] {
            sketch
                .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
                .expect("pinning each end in turn is consistent");
        }
        let implied = sketch
            .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
            .expect("the pins already put it level");
        sketch
            .constraints()
            .iter()
            .find(|held| held.id == implied)
            .expect("just added")
            .redundant
    };
    // Drawn level to begin with, and drawn slanted so the pins had to move it.
    assert!(flagged_for([10.0, 0.0]));
    assert!(flagged_for([10.0, 4.0]));
}

// ---------------------------------------------------------------------------------------------
// The relations: the constraints that name two pieces of geometry rather than one piece and an
// axis. Every one of them is checked by measuring the drawing afterwards, never by trusting the
// solver's own verdict — see `SATISFIED_RESIDUAL`.
// ---------------------------------------------------------------------------------------------

/// Two segments, drawn apart and slanted differently, for the two-segment relations.
fn two_segments() -> (Sketch, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first_tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let first_head = sketch.add_free_point(SketchPoint::new(20, 4));
    let second_tail = sketch.add_free_point(SketchPoint::new(0, 30));
    let second_head = sketch.add_free_point(SketchPoint::new(12, 44));
    let first = sketch.connect(first_tail, first_head).expect("a segment");
    let second = sketch.connect(second_tail, second_head).expect("a segment");
    (sketch, first, second)
}

/// The direction of a segment, as a unit vector, read off the solved drawing.
fn direction(sketch: &Sketch, segment: EntityId) -> [f64; 2] {
    let span = sketch
        .segments()
        .iter()
        .find(|seg| seg.id == segment)
        .expect("the segment");
    let (tail, head) = (position(sketch, span.from), position(sketch, span.to));
    let delta = [head[0] - tail[0], head[1] - tail[1]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    [delta[0] / length, delta[1] / length]
}

/// The length of a segment, read off the solved drawing.
fn span_length(sketch: &Sketch, segment: EntityId) -> f64 {
    let span = sketch
        .segments()
        .iter()
        .find(|seg| seg.id == segment)
        .expect("the segment");
    let (tail, head) = (position(sketch, span.from), position(sketch, span.to));
    ((head[0] - tail[0]).powi(2) + (head[1] - tail[1]).powi(2)).sqrt()
}

/// Coincident brings two points to one place — and it is a constraint, not a merge, so both ids
/// survive and deleting it lets them part again.
#[test]
fn coincident_brings_two_points_together_without_merging_them() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch.add_free_point(SketchPoint::new(0, 0));
    let second = sketch.add_free_point(SketchPoint::new(10, 6));
    let id = sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: first,
                onto: CoincidentTarget::Point(second),
            },
            ctx(16),
        )
        .expect("two free points can always meet");

    let (here, there) = (position(&sketch, first), position(&sketch, second));
    assert!(
        (here[0] - there[0]).abs() < 1e-6 && (here[1] - there[1]).abs() < 1e-6,
        "coincident: {here:?} vs {there:?}"
    );
    // They meet in the middle, for the same least-squares reason a level segment does.
    assert!((here[0] - 5.0).abs() < 1e-6, "met in the middle: {here:?}");

    sketch.delete_constraint(id);
    assert_eq!(sketch.points().len(), 2, "both ids survived the assertion");
}

/// A point cannot be asserted coincident with itself: the claim has no content.
#[test]
fn coincident_with_itself_is_refused() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let point = sketch.add_free_point(SketchPoint::new(3, 3));
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Coincident {
                point,
                onto: CoincidentTarget::Point(point)
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::Impossible)
    );
}

/// Parallel drives the sine of the angle between two segments to zero, and both segments keep
/// their extent getting there.
///
/// It does NOT preserve length exactly. The residual is an angle, so the solver is free to reach
/// it any way it likes, and the way it likes is the smallest move in the PARAMETERS — which are
/// coordinates, not lengths. A pure rotation would hold both lengths and is a larger coordinate
/// move than the shear-ish answer the solve actually finds. What the normalization buys is
/// conditioning, not rigidity: the residual reads the same on a 3-voxel segment and a 300-voxel
/// one, so neither dominates the step.
#[test]
fn parallel_aligns_two_segments_without_collapsing_them() {
    let (mut sketch, first, second) = two_segments();
    let before_first = span_length(&sketch, first);
    let before_second = span_length(&sketch, second);
    sketch
        .add_constraint(ConstraintKind::Parallel { first, second }, ctx(16))
        .expect("two free segments can always be made parallel");

    let (a, b) = (direction(&sketch, first), direction(&sketch, second));
    assert!(
        (a[0] * b[1] - a[1] * b[0]).abs() < 1e-6,
        "parallel: {a:?} vs {b:?}"
    );
    for (segment, before) in [(first, before_first), (second, before_second)] {
        let after = span_length(&sketch, segment);
        assert!(
            after > before / 2.0,
            "an angle claim kept the extent: {segment} went {before} to {after}"
        );
    }
}

/// Perpendicular drives the cosine to zero.
#[test]
fn perpendicular_squares_two_segments() {
    let (mut sketch, first, second) = two_segments();
    sketch
        .add_constraint(ConstraintKind::Perpendicular { first, second }, ctx(16))
        .expect("two free segments can always be squared");

    let (a, b) = (direction(&sketch, first), direction(&sketch, second));
    assert!(
        (a[0] * b[0] + a[1] * b[1]).abs() < 1e-6,
        "perpendicular: {a:?} vs {b:?}"
    );
}

/// Equal matches two lengths without naming one — the pair settles between them, which is the
/// difference between a relation and a pair of dimensions.
#[test]
fn equal_matches_two_lengths_without_asserting_which() {
    let (mut sketch, first, second) = two_segments();
    let longest = span_length(&sketch, first).max(span_length(&sketch, second));
    let shortest = span_length(&sketch, first).min(span_length(&sketch, second));
    sketch
        .add_constraint(ConstraintKind::Equal { first, second }, ctx(16))
        .expect("two free segments can always match");

    let (a, b) = (span_length(&sketch, first), span_length(&sketch, second));
    assert!((a - b).abs() < 1e-6, "equal: {a} vs {b}");
    assert!(
        a > shortest && a < longest,
        "neither end won outright: {a} is not between {shortest} and {longest}"
    );
}

/// Midpoint pins a point halfway along a segment — both coordinates of halfway.
#[test]
fn midpoint_puts_a_point_halfway_along_a_segment() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 10));
    let segment = sketch.connect(tail, head).expect("a segment");
    let point = sketch.add_free_point(SketchPoint::new(3, 17));
    sketch
        .add_constraint(ConstraintKind::Midpoint { point, segment }, ctx(16))
        .expect("a free point can always reach a midpoint");

    let here = position(&sketch, point);
    let a = position(&sketch, tail);
    let b = position(&sketch, head);
    assert!(
        (here[0] - (a[0] + b[0]) / 2.0).abs() < 1e-6
            && (here[1] - (a[1] + b[1]) / 2.0).abs() < 1e-6,
        "midpoint: {here:?} on {a:?}..{b:?}"
    );
}

/// A segment's own endpoint cannot be its midpoint: it names the collapse rather than solving
/// into one.
#[test]
fn an_endpoint_is_refused_as_its_own_segments_midpoint() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 0));
    let segment = sketch.connect(tail, head).expect("a segment");
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Midpoint {
                point: tail,
                segment
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::Impossible)
    );
}

/// Point-on-curve spends ONE freedom: the point lands on the line and is still free to slide
/// along it. Two points made coincident would have spent two.
#[test]
fn a_point_lands_on_a_segments_line_and_keeps_sliding() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 0));
    let segment = sketch.connect(tail, head).expect("a segment");
    let point = sketch.add_free_point(SketchPoint::new(6, 9));
    let before = sketch.degrees_of_freedom(ctx(16)).expect("no fixed source");

    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point,
                onto: CoincidentTarget::Curve(SketchCurve::Segment(segment)),
            },
            ctx(16),
        )
        .expect("a free point can always reach a line");

    let here = position(&sketch, point);
    assert!(here[1].abs() < 1e-6, "off the line: {here:?}");
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        before - 1
    );
}

/// The support is the whole circle an arc is cut from, not the finite piece. A residual that had
/// to report "past the end" would have a kink there, and the optimizer would be walking a cliff —
/// so a point outside the sweep lands on the circle right where it stands, not dragged to an end.
#[test]
fn a_point_lands_on_the_circle_an_arc_is_cut_from() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(10, 0));
    let to = sketch.add_free_point(SketchPoint::new(0, 10));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc about the origin");
    // Diagonally opposite the quarter the arc covers, and well off its circle.
    let point = sketch.add_free_point(SketchPoint::new(-20, -20));

    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point,
                onto: CoincidentTarget::Curve(SketchCurve::Arc(arc)),
            },
            ctx(16),
        )
        .expect("a free point can always reach a circle");

    let here = position(&sketch, point);
    let form = sketch
        .arc_form_of(arc)
        .expect("the arc still draws a circle");
    let (center, radius) = (form.center, form.radius);
    assert!(
        ((here[0] - center[0]).hypot(here[1] - center[1]) - radius).abs() < 1e-6,
        "off the circle: {here:?} against {center:?} r{radius}"
    );
    assert!(
        here[1] < center[1],
        "it landed below the quarter the arc actually covers, so the support is the whole circle          and not the finite piece: {here:?}"
    );
}

/// An endpoint is already on its own line, so the row could never be violated; a point that SHAPES
/// a spline is the same vacuity one curve up; and a curve the kernel can name no place along has
/// nothing to stand on at all.
#[test]
fn a_vacuous_or_unmodelled_point_on_curve_is_refused() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 0));
    let segment = sketch.connect(tail, head).expect("a segment");
    let spline = sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 8),
            SketchPoint::new(4, 12),
            SketchPoint::new(9, 12),
            SketchPoint::new(13, 8),
        ])
        .expect("a spline");
    let ellipse = sketch
        .add_ellipse(
            SketchPoint::new(0, 40),
            SketchPoint::new(10, 40),
            SketchPoint::new(0, 44),
        )
        .expect("an ellipse");
    let control = sketch
        .splines
        .iter()
        .find(|held| held.id == spline)
        .expect("the spline")
        .points[1];

    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Coincident {
                point: tail,
                onto: CoincidentTarget::Curve(SketchCurve::Segment(segment)),
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::Impossible)
    );
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Coincident {
                point: control,
                onto: CoincidentTarget::Curve(SketchCurve::Spline(spline)),
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::Impossible),
        "a point that draws the curve cannot also be held to it"
    );
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::Coincident {
                point: tail,
                onto: CoincidentTarget::Curve(SketchCurve::Ellipse(ellipse)),
            },
            ctx(16)
        ),
        Err(ConstraintRefusal::UnknownEntity)
    );
    // And the spline itself is not among the refusals: a point that does not draw it can stand
    // on it, which is the whole reason the two predicates differ.
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: tail,
                onto: CoincidentTarget::Curve(SketchCurve::Spline(spline)),
            },
            ctx(16),
        )
        .expect("a free point can stand on a spline");
}

/// Collinear says parallel AND no offset, which is why it spends two freedoms where Parallel
/// spends one.
#[test]
fn collinear_puts_two_segments_on_one_line() {
    let (mut sketch, first, second) = two_segments();
    let before = sketch.degrees_of_freedom(ctx(16)).expect("no fixed source");
    sketch
        .add_constraint(ConstraintKind::Collinear { first, second }, ctx(16))
        .expect("two free segments can always share a line");

    let (a, b) = (direction(&sketch, first), direction(&sketch, second));
    assert!(
        (a[0] * b[1] - a[1] * b[0]).abs() < 1e-6,
        "collinear implies parallel: {a:?} vs {b:?}"
    );
    let datum = sketch
        .segments()
        .iter()
        .find(|seg| seg.id == first)
        .expect("the datum");
    let anchor = position(&sketch, datum.from);
    let normal = [-a[1], a[0]];
    let other = *sketch
        .segments()
        .iter()
        .find(|seg| seg.id == second)
        .expect("the other");
    for end in [other.from, other.to] {
        let here = position(&sketch, end);
        let off = (here[0] - anchor[0]) * normal[0] + (here[1] - anchor[1]) * normal[1];
        assert!(off.abs() < 1e-6, "end {end} stands {off} off the line");
    }
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        before - 2,
        "collinear spends two freedoms"
    );
}

/// **The stride property.** A kind that writes two residuals must be given two rows, or every
/// constraint after it in the list reads the wrong ones. A catch-all `_ => 1` arm in
/// `residual_count` hands a two-row kind one row and corrupts the whole system rather than
/// failing.
///
/// Asserted by stacking a two-row relation BEFORE a one-row one and checking that both still
/// hold — under a wrong stride the second reads the first's spare row and cannot be met.
#[test]
fn a_two_residual_relation_does_not_shift_the_rows_after_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch.add_free_point(SketchPoint::new(0, 0));
    let second = sketch.add_free_point(SketchPoint::new(9, 5));
    let tail = sketch.add_free_point(SketchPoint::new(30, 0));
    let head = sketch.add_free_point(SketchPoint::new(50, 7));
    let segment = sketch.connect(tail, head).expect("a segment");

    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: first,
                onto: CoincidentTarget::Point(second),
            },
            ctx(16),
        )
        .expect("two free points can meet");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("an untouched segment can be leveled");

    let (here, there) = (position(&sketch, first), position(&sketch, second));
    assert!(
        (here[0] - there[0]).abs() < 1e-6 && (here[1] - there[1]).abs() < 1e-6,
        "the earlier two-row relation still holds: {here:?} vs {there:?}"
    );
    let (a, b) = (position(&sketch, tail), position(&sketch, head));
    assert!(
        (a[1] - b[1]).abs() < 1e-6,
        "the later one-row relation still holds: {a:?} to {b:?}"
    );
}

/// Every new relation survives a drag of the geometry it names, for the same reason a level
/// segment does — the drag solves the whole standing system with the grabbed point pinned, and it
/// names no kind.
#[test]
fn the_relations_hold_through_a_drag() {
    let (mut sketch, first, second) = two_segments();
    sketch
        .add_constraint(ConstraintKind::Perpendicular { first, second }, ctx(16))
        .expect("two free segments can be squared");
    let grabbed = sketch
        .segments()
        .iter()
        .find(|seg| seg.id == first)
        .expect("the segment")
        .from;

    assert!(sketch
        .move_point(grabbed, SketchPoint::new(-13, 9), ctx(16))
        .expect("evaluation context"));

    let (a, b) = (direction(&sketch, first), direction(&sketch, second));
    assert!(
        (a[0] * b[0] + a[1] * b[1]).abs() < 1e-6,
        "still square after the drag: {a:?} vs {b:?}"
    );
    let held = position(&sketch, grabbed);
    assert!(
        (held[0] + 13.0).abs() < 1e-6 && (held[1] - 9.0).abs() < 1e-6,
        "the hand still holds its point: {held:?}"
    );
}

/// An arc from `(0,0)` to `(20,0)` sweeping a quarter turn, the id of the center point the sketch
/// derives for it, and a loose point off to the side.
fn arc_with_center() -> (Sketch, EntityId, EntityId, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 0));
    sketch
        .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
        .expect("a quarter turn");
    let center = sketch.arcs()[0].center;
    let loose = sketch.add_free_point(SketchPoint::new(40, 17));
    (sketch, tail, head, center, loose)
}

/// **A constraint on an arc's center still moves the ARC.** The center is a placed point now
/// (ADR 0038) and the solver may choose its coordinates outright — but it cannot choose them
/// alone, because the arc's own row says the center stands the same distance from both ends. So a
/// center pinned somewhere new drags the ends after it.
///
/// The loose point is `Fix`ed so that it is the reference piece and the arc is what travels. Left
/// free, the three-point arc outweighs it under the parametric anchoring policy and the point comes
/// to the arc instead, which is correct but leaves this mechanism unobserved.
#[test]
fn a_constraint_on_an_arcs_center_moves_the_arc() {
    let (mut sketch, tail, head, center, loose) = arc_with_center();
    assert!(sketch.is_arc_center(center));
    let (before_tail, before_head) = (position(&sketch, tail), position(&sketch, head));
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: loose,
                at: SketchPoint::new(40, 17),
            },
            ctx(16),
        )
        .expect("the point the arc must reach");

    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: center,
                onto: CoincidentTarget::Point(loose),
            },
            ctx(16),
        )
        .expect("an arc's center can be pinned to a point");

    let (here, there) = (position(&sketch, center), position(&sketch, loose));
    assert!(
        (here[0] - there[0]).abs() < 1e-6 && (here[1] - there[1]).abs() < 1e-6,
        "the center sits on the point: {here:?} vs {there:?}"
    );
    let moved = position(&sketch, tail) != before_tail || position(&sketch, head) != before_head;
    assert!(moved, "the ends took the correction, not the center's slot");

    // And the stored center still agrees with the arc it belongs to: the solve satisfied the
    // equal-radius row, so seating it onto the chord's bisector afterwards has nothing to move.
    // Not exact: a `SketchPoint` stores an integer voxel plus an f32 fraction, so the seat computed
    // from the ROUND-TRIPPED endpoints lands a storage epsilon away.
    let settled = position(&sketch, center);
    sketch.sync_derived_points();
    let after_sync = position(&sketch, center);
    assert!(
        (settled[0] - after_sync[0]).abs() < 1e-5 && (settled[1] - after_sync[1]).abs() < 1e-5,
        "seating does not move it: {settled:?} vs {after_sync:?}"
    );
}

/// `Fix` one end of the arc, then bring the center onto a point. The fixed end must not move, and
/// the loose point is what takes up the difference — the arc is the heavier piece AND the pinned
/// one, so it is the reference and the point travels to it.
#[test]
fn a_fixed_arc_end_holds_while_the_center_is_brought_to_a_point() {
    let (mut sketch, tail, head, center, loose) = arc_with_center();
    let anchored = position(&sketch, tail);
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::from_continuous(anchored[0], anchored[1]),
            },
            ctx(16),
        )
        .expect("an end can be pinned");
    let before_head = position(&sketch, head);

    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: center,
                onto: CoincidentTarget::Point(loose),
            },
            ctx(16),
        )
        .expect("the center can still be pinned with one end held");

    let held = position(&sketch, tail);
    assert!(
        (held[0] - anchored[0]).abs() < 1e-6 && (held[1] - anchored[1]).abs() < 1e-6,
        "the fixed end did not move: {held:?} vs {anchored:?}"
    );
    let (here, there) = (position(&sketch, center), position(&sketch, loose));
    assert!(
        (here[0] - there[0]).abs() < 1e-6 && (here[1] - there[1]).abs() < 1e-6,
        "the center still reached the point: {here:?} vs {there:?}"
    );
    assert_eq!(
        position(&sketch, head),
        before_head,
        "and the arc held whole: the point is what traveled"
    );
}

/// Every kind with a point slot reads a derived point the same way, because they all go through one
/// `position_of`. Stated for `Fix`, which is the sharpest case: it pins the arc through its center.
#[test]
fn fixing_an_arcs_center_pins_the_arc_through_it() {
    let (mut sketch, tail, _head, center, _loose) = arc_with_center();
    let held = position(&sketch, center);
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: center,
                at: SketchPoint::from_continuous(held[0], held[1]),
            },
            ctx(16),
        )
        .expect("an arc's center can be fixed");

    // Dragging an END now has to respect the fixed center, so the drag settles somewhere that keeps
    // the derived center where it was pinned rather than wherever the raw drag would have put it.
    assert!(sketch
        .move_point(tail, SketchPoint::new(-9, 6), ctx(16))
        .expect("evaluation context"));
    let after = position(&sketch, center);
    assert!(
        (after[0] - held[0]).abs() < 1e-6 && (after[1] - held[1]).abs() < 1e-6,
        "the fixed center held through a drag of the arc's end: {after:?} vs {held:?}"
    );
}

/// A derived point is not a FREEDOM. Counting an arc's center as two free coordinates would say a
/// sketch is under-constrained in ways nothing can take up, because the only way to move a center is
/// to move the arc — which is already counted, at the ends.
#[test]
fn an_arcs_center_is_not_a_degree_of_freedom() {
    let (sketch, _tail, _head, _center, _loose) = arc_with_center();
    assert_eq!(
        sketch.points().len(),
        4,
        "two arc ends, the derived center, and the loose point"
    );
    assert_eq!(
        sketch.degrees_of_freedom(ctx(16)).expect("no fixed source"),
        7,
        "three authored points plus the free arc sweep — the center is not one of them"
    );
}

/// **A drag uses whatever freedom is left, instead of being refused for the freedom that is not.**
///
/// The configuration: an arc whose two ends are both `Fix`ed — so its center is fully determined —
/// a point `Coincident` with that center, and a `Vertical` on the segment reaching down from it.
/// One freedom remains, the segment's LENGTH. Treating the hand as a hard pin makes the far end
/// immovable, because the cursor is essentially never exactly on the line the point may slide
/// along and the pinned system is then refused as unsatisfiable.
#[test]
fn a_point_with_one_freedom_left_slides_along_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let arc_tail = sketch.add_free_point(SketchPoint::new(16, 43));
    let arc_head = sketch.add_free_point(SketchPoint::new(-1, 67));
    sketch
        .connect_arc(arc_tail, arc_head, AngleMeasurement::from_degrees(262))
        .expect("a major arc");
    let center = sketch.arcs()[0].center;
    let at_center = position(&sketch, center);
    let top = sketch.add_free_point(SketchPoint::from_continuous(at_center[0], at_center[1]));
    let bottom = sketch.add_free_point(SketchPoint::from_continuous(
        at_center[0],
        at_center[1] - 36.0,
    ));
    let segment = sketch
        .connect(bottom, top)
        .expect("the line under the center");

    for point in [arc_tail, arc_head] {
        let held = position(&sketch, point);
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(held[0], held[1]),
                },
                ctx(16),
            )
            .expect("both arc ends pin");
    }
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: top,
                onto: CoincidentTarget::Point(center),
            },
            ctx(16),
        )
        .expect("the line's top meets the arc's center");
    sketch
        .add_constraint(ConstraintKind::Vertical { segment }, ctx(16))
        .expect("and the line stands plumb");

    // Drag the free end well off the line it may slide along. It must MOVE — down the line.
    let before = position(&sketch, bottom);
    assert!(sketch
        .move_point(
            bottom,
            SketchPoint::from_continuous(before[0] + 22.0, 4.0),
            ctx(16)
        )
        .expect("evaluation context"));

    let after = position(&sketch, bottom);
    assert!(
        (after[1] - before[1]).abs() > 1.0,
        "the length changed: {before:?} to {after:?}"
    );
    assert!(
        (after[1] - 4.0).abs() < 1e-6,
        "and it followed the cursor as far as it was allowed: {after:?}"
    );
    // The standing constraints are still exactly met — the pull did not buy the move with them.
    let up = position(&sketch, top);
    assert!(
        (after[0] - up[0]).abs() < 1e-6,
        "still plumb: {after:?} under {up:?}"
    );
    let settled_center = position(&sketch, center);
    assert!(
        (up[0] - settled_center[0]).abs() < 1e-6 && (up[1] - settled_center[1]).abs() < 1e-6,
        "still on the arc's center: {up:?} vs {settled_center:?}"
    );
    assert_eq!(
        position(&sketch, arc_tail),
        [16.0, 43.0],
        "and the arc did not move"
    );
}

/// A drag the standing system CAN meet exactly is untouched by the two-stage settle: stage one
/// meets the pull, so stage two starts at a solution and moves nothing.
#[test]
fn an_achievable_drag_lands_exactly_on_the_cursor() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(40, 0));
    let segment = sketch.connect(tail, head).expect("a fresh segment");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment }, ctx(16))
        .expect("a lone level");

    assert!(sketch
        .move_point(tail, SketchPoint::new(-7, -18), ctx(16))
        .expect("evaluation context"));
    let dragged = position(&sketch, tail);
    assert!(
        (dragged[0] + 7.0).abs() < 1e-9 && (dragged[1] + 18.0).abs() < 1e-9,
        "the hand holds the grabbed end exactly: {dragged:?}"
    );
}

/// A closed quad, laid out counter-clockwise from the origin, and its four edges in that order.
fn quad(corners: [[i64; 2]; 4]) -> (Sketch, [EntityId; 4], [EntityId; 4]) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let points = corners.map(|[x, y]| sketch.add_free_point(SketchPoint::new(x, y)));
    let edges = [0, 1, 2, 3].map(|index| {
        sketch
            .connect(points[index], points[(index + 1) % 4])
            .expect("a fresh edge")
    });
    (sketch, points, edges)
}

/// **A constraint moves untouched geometry as a piece, not as a pile of independent points.**
/// Bringing one corner of a square to a point a long way off is met most cheaply by dragging that
/// corner alone and leaving the other three — the least travel, and the maximum deformation.
/// Preferring to keep every edge's span makes the whole square TRANSLATE instead: a rigid motion
/// satisfies both the constraint and the preference at once, so there is nothing to trade.
#[test]
fn a_constraint_translates_a_group_rather_than_deforming_it() {
    let (mut sketch, corners, _) = quad([[0, 0], [20, 0], [20, 20], [0, 20]]);
    let target = sketch.add_free_point(SketchPoint::new(50, 30));
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: target,
                at: SketchPoint::new(50, 30),
            },
            ctx(16),
        )
        .expect("the target is where the square must reach");

    let before = corners.map(|corner| position(&sketch, corner));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: corners[0],
                onto: CoincidentTarget::Point(target),
            },
            ctx(16),
        )
        .expect("one corner meets it");

    let after = corners.map(|corner| position(&sketch, corner));
    for (index, (was, now)) in before.iter().zip(&after).enumerate() {
        let travel = [now[0] - was[0], now[1] - was[1]];
        assert!(
            (travel[0] - 50.0).abs() < 1e-6 && (travel[1] - 30.0).abs() < 1e-6,
            "corner {index} rode along with the rest: {was:?} to {now:?}"
        );
    }
}

/// **The heavier group holds; the lighter one comes to it.**
///
/// Weighing the two pieces is not enough: least squares splits the gap in inverse proportion to
/// their sizes, so a quad meeting a stick would still slide a third of the way to meet it. The
/// heavier piece is anchored outright for the preference pass and does not move at all.
#[test]
fn the_smaller_group_travels_to_the_larger_one() {
    let (mut sketch, corners, _) = quad([[0, 0], [20, 0], [20, 20], [0, 20]]);
    let near = sketch.add_free_point(SketchPoint::new(60, 30));
    let far = sketch.add_free_point(SketchPoint::new(80, 30));
    sketch.connect(near, far).expect("a two-point stick");

    let before: Vec<[f64; 2]> = [corners[1], near]
        .iter()
        .map(|id| position(&sketch, *id))
        .collect();
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: corners[1],
                onto: CoincidentTarget::Point(near),
            },
            ctx(16),
        )
        .expect("the stick's near end meets the quad's corner");
    let after: Vec<[f64; 2]> = [corners[1], near]
        .iter()
        .map(|id| position(&sketch, *id))
        .collect();
    let travel = |index: usize| {
        let (was, now) = (before[index], after[index]);
        ((now[0] - was[0]).powi(2) + (now[1] - was[1]).powi(2)).sqrt()
    };
    assert!(
        travel(0) < 1e-6,
        "the four-corner quad held still: it moved {:.4}",
        travel(0)
    );
    // (20,0) to (60,30) is 50, and the stick covered all of it.
    assert!(
        (travel(1) - 50.0).abs() < 1e-6,
        "the two-point stick came the whole way: {:.4}",
        travel(1)
    );
    // The stick translated by (-40,-30), so its far end went from (80,30) to (40,0).
    let tip = position(&sketch, far);
    assert!(
        (tip[0] - 40.0).abs() < 1e-6 && (tip[1] - 0.0).abs() < 1e-6,
        "and it came as a piece, its far end riding along: {tip:?}"
    );
}

/// The preference never outranks the assertion. Leveling one edge of a closed quad CANNOT leave
/// the other three spans alone, and when the two genuinely fight, the constraint wins outright —
/// the pass that follows the preferred one re-solves the assertions by themselves.
#[test]
fn a_constraint_that_fights_rigidity_is_still_met_exactly() {
    let (mut sketch, corners, edges) = quad([[0, 0], [20, 8], [20, 28], [0, 20]]);
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment: edges[0] }, ctx(16))
        .expect("level the slanted bottom");

    let (tail, head) = (position(&sketch, corners[0]), position(&sketch, corners[1]));
    assert!(
        (tail[1] - head[1]).abs() < 1e-9,
        "exactly level, not nearly: {tail:?} to {head:?}"
    );
}

/// **Deleting a line deletes the points it was drawn between**, unless something else draws them.
/// A line removed from a drawing must leave behind neither dots the author never placed nor a
/// constraint naming them.
#[test]
fn deleting_a_line_takes_the_ends_nothing_else_draws() {
    let (mut sketch, tail, head, segment) = slanted();
    let shared = sketch.add_free_point(SketchPoint::new(30, 30));
    let neighbor = sketch.connect(head, shared).expect("a second line");

    sketch.delete_segment(segment);
    assert!(
        !sketch.points().iter().any(|point| point.id == tail),
        "the lone end went with the line"
    );
    assert!(
        sketch.points().iter().any(|point| point.id == head),
        "the shared end stays: the other line still draws it"
    );
    assert!(
        sketch.points().iter().any(|point| point.id == shared),
        "and so does its far end"
    );
    assert_eq!(sketch.segments().len(), 1, "only the named line went");
    assert_eq!(sketch.segments()[0].id, neighbor);
}

/// A constraint is not a reason for a point to outlive the geometry it was drawn for: the line
/// takes the point, and the cascade takes the constraint.
#[test]
fn a_constraint_does_not_keep_a_deleted_lines_end_alive() {
    let (mut sketch, tail, _head, segment) = slanted();
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("a lone fix");
    assert_eq!(sketch.constraints().len(), 1);

    sketch.delete_segment(segment);
    assert!(sketch.points().is_empty(), "both ends went with the line");
    assert!(
        sketch.constraints().is_empty(),
        "and the fix went with the point it named"
    );
}

#[test]
fn tangent_canonicalizes_member_order_and_rejects_branch_independent_duplicates() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let segment = sketch.connect(tail, head).expect("segment");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
        .expect("circle");
    let requested = ConstraintKind::tangent(
        SketchCurve::Circle(circle),
        SketchCurve::Segment(segment),
        TangentBranch::Line(LineSide::Left),
    );
    let ConstraintKind::Tangent {
        first,
        second,
        branch,
    } = requested
    else {
        panic!("tangent")
    };
    assert_eq!(first.id().min(second.id()), first.id());
    assert_eq!(branch, TangentBranch::Line(LineSide::Left));
    let id = sketch.add_constraint(requested, ctx(16)).expect("tangent");
    assert!(matches!(
        sketch.add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::AlreadyAsserted { existing }) if existing == id
    ));
}

#[test]
#[allow(clippy::too_many_lines)]
fn tangent_load_normalizes_reversed_members_and_repair_drops_malformed_duplicates() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(6))
        .expect("outer circle");
    let second = sketch
        .add_circle(SketchPoint::new(4, 0), SketchLength::new(2))
        .expect("inner circle");
    let segment_from = sketch.add_free_point(SketchPoint::new(-8, 0));
    let segment_to = sketch.add_free_point(SketchPoint::new(-2, 0));
    let segment = sketch.connect(segment_from, segment_to).expect("segment");
    let other_from = sketch.add_free_point(SketchPoint::new(-8, 3));
    let other_to = sketch.add_free_point(SketchPoint::new(-2, 3));
    let other_segment = sketch.connect(other_from, other_to).expect("other segment");
    let mut raw = serde_json::to_value(&sketch).expect("serialize source sketch");
    raw["constraints"] = serde_json::json!([
        {"id": 90, "kind": {"Tangent": {
            "first": {"Circle": second}, "second": {"Circle": first},
            "branch": {"Internal": {"contains": "Second"}}
        }}, "redundant": false},
        {"id": 91, "kind": {"Tangent": {
            "first": {"Circle": first}, "second": {"Circle": second},
            "branch": "External"
        }}, "redundant": false},
        {"id": 92, "kind": {"Tangent": {
            "first": {"Circle": first}, "second": {"Circle": first},
            "branch": "External"
        }}, "redundant": false}
        ,{"id": 93, "kind": {"Tangent": {
            "first": {"Segment": segment}, "second": {"Segment": other_segment},
            "branch": {"Line": "Left"}
        }}, "redundant": false}
        ,{"id": 94, "kind": {"Tangent": {
            "first": {"Segment": segment}, "second": {"Circle": first},
            "branch": "External"
        }}, "redundant": false}
        ,{"id": 95, "kind": {"Tangent": {
            "first": {"Segment": 900}, "second": {"Circle": first},
            "branch": {"Line": "Left"}
        }}, "redundant": false}
        ,{"id": 96, "kind": {"Tangent": {
            "first": {"Arc": 901}, "second": {"Circle": first},
            "branch": "External"
        }}, "redundant": false}
        ,{"id": 97, "kind": {"Tangent": {
            "first": {"Circle": 902}, "second": {"Circle": first},
            "branch": "External"
        }}, "redundant": false}
        ,{"id": 98, "kind": {"Tangent": {
            "first": {"Segment": segment}, "second": {"Circle": first},
            "branch": {"Line": "Left"}
        }}, "redundant": false}
    ]);
    let mut loaded: Sketch = serde_json::from_value(raw).expect("structural load normalizes");
    let ConstraintKind::Tangent {
        first: held_first,
        second: held_second,
        branch,
    } = loaded.constraints()[0].kind
    else {
        panic!("tangent survives load")
    };
    assert_eq!((held_first.id(), held_second.id()), (first, second));
    assert_eq!(
        branch,
        TangentBranch::Internal {
            contains: InternalContainment::First
        }
    );
    let ConstraintKind::Tangent {
        first: line_first,
        second: line_second,
        branch: line_branch,
    } = loaded
        .constraints()
        .iter()
        .find(|constraint| constraint.id == 98)
        .expect("line tangent")
        .kind
    else {
        panic!("line tangent survives load")
    };
    assert_eq!(
        (line_first, line_second),
        (SketchCurve::Circle(first), SketchCurve::Segment(segment))
    );
    assert_eq!(
        line_branch,
        TangentBranch::Line(LineSide::Left),
        "LineSide is not remapped"
    );
    assert_eq!(
        loaded.repair(ctx(16)),
        7,
        "duplicate, self, segment-pair, wrong branch, and every dangling curve kind drop"
    );
    assert_eq!(loaded.constraints().len(), 2);
    assert_eq!(loaded.constraints()[0].id, 90, "stable first survivor");
    assert_eq!(loaded.constraints()[1].id, 98, "later distinct survivor");
    let saved = serde_json::to_value(&loaded).expect("canonical serialize");
    assert_eq!(
        saved["constraints"][0]["kind"]["Tangent"]["first"]["Circle"],
        first
    );
}

#[test]
#[allow(clippy::too_many_lines)]
/// A structurally loaded payload can contain a tangent that is true on an infinite supporting
/// line but misses the finite segment. Both ordinary solve and drag must reject it before
/// write-back.
fn loaded_off_domain_tangent_keeps_solve_and_drag_atomic() {
    let mut source = Sketch::empty(PlaneAxis::Z);
    let from = source.add_free_point(SketchPoint::new(0, 0));
    let to = source.add_free_point(SketchPoint::new(1, 0));
    let segment = source.connect(from, to).expect("short segment");
    let circle = source
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
        .expect("circle tangent to the supporting line");
    let center = source.circles()[0].center;
    for point in [from, to, center] {
        let at = position(&source, point);
        source
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("pin source geometry");
    }
    assert_eq!(
        source
            .add_constraint(
                ConstraintKind::tangent(
                    SketchCurve::Segment(segment),
                    SketchCurve::Circle(circle),
                    TangentBranch::Line(LineSide::Left),
                ),
                ctx(16),
            )
            .expect_err("an infinite-line-only tangent is not authorable"),
        ConstraintRefusal::InvalidTangent {
            constraint: None,
            error: ::parametric::sketch::TangentContactError::OutsideFirstDomain,
        },
        "the finite-domain refusal reaches the document boundary unchanged"
    );
    let valid_from = source.add_free_point(SketchPoint::new(0, 10));
    let valid_to = source.add_free_point(SketchPoint::new(10, 10));
    let valid_segment = source
        .connect(valid_from, valid_to)
        .expect("finite tangent segment");
    let valid_circle = source
        .add_circle(SketchPoint::new(5, 14), SketchLength::new(4))
        .expect("finite tangent circle");
    let valid_center = source
        .circles()
        .iter()
        .find(|held| held.id == valid_circle)
        .expect("stored circle")
        .center;
    for point in [valid_from, valid_to, valid_center] {
        let at = position(&source, point);
        source
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("pin valid tangent geometry");
    }
    source
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(valid_segment),
                SketchCurve::Circle(valid_circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("first standing tangent is valid");
    let mut raw = serde_json::to_value(&source).expect("serialize source");
    raw["constraints"]
        .as_array_mut()
        .expect("constraints array")
        .push(serde_json::json!({"id": 999, "kind": {"Tangent": {
            "first": {"Segment": segment}, "second": {"Circle": circle},
            "branch": {"Line": "Left"}
        }}, "redundant": false}));
    let mut loaded: Sketch = serde_json::from_value(raw).expect("load raw tangent");
    let before = serde_json::to_value(&loaded).expect("snapshot");

    assert_eq!(
        loaded.solve(ctx(16)),
        Err(SketchEvaluationError::InvalidTangent {
            constraint: 999,
            error: ::parametric::sketch::TangentContactError::OutsideFirstDomain,
        })
    );
    assert_eq!(serde_json::to_value(&loaded).expect("after solve"), before);

    let refusal = loaded
        .add_constraint(
            ConstraintKind::Horizontal {
                segment: valid_segment,
            },
            ctx(16),
        )
        .expect_err("a new assertion cannot hide a malformed standing Tangent");
    assert_eq!(
        refusal,
        ConstraintRefusal::InvalidTangent {
            constraint: Some(999),
            error: ::parametric::sketch::TangentContactError::OutsideFirstDomain,
        }
    );
    assert_eq!(refusal.culprits(), vec![999]);
    assert_eq!(
        serde_json::to_value(&loaded).expect("after refused add"),
        before
    );

    assert_eq!(
        loaded.move_point(from, SketchPoint::new(0, 1), ctx(16)),
        Err(SketchEvaluationError::InvalidTangent {
            constraint: 999,
            error: ::parametric::sketch::TangentContactError::OutsideFirstDomain,
        }),
        "a drag of the geometry it names reports the same offending Tangent as solve"
    );
    assert_eq!(serde_json::to_value(&loaded).expect("after drag"), before);

    // A drag answers for what it can REACH. The malformed tangent names the other shape, and no
    // relation or edge connects the two, so this drag could not have broken it and does not solve
    // it — the whole-drawing `solve` above is what reports a corrupt load. The alternative, making
    // every drag anywhere answer for the entire plane, is what made a drag cost the whole drawing.
    assert!(loaded
        .move_point(valid_from, SketchPoint::new(0, 11), ctx(16))
        .expect("an unreachable breakage is not this drag's to report"));
}

#[test]
/// Dragging an unconstrained arc's center moves the ARC and nothing beyond it: its two ends come
/// along by the same displacement, the turn between the three points is unchanged, and an
/// unrelated circle across the plane is byte-exact. A rigid set reaches as far as the shape and
/// no further.
fn dragging_an_arc_center_carries_its_arc_and_nothing_else() {
    let (mut sketch, tail, head, center, _) = arc_with_center();
    sketch
        .add_circle(SketchPoint::new(50, 20), SketchLength::new(7))
        .expect("unrelated circle");
    let before_tail = sketch
        .points()
        .iter()
        .find(|point| point.id == tail)
        .copied()
        .expect("tail");
    let before_head = sketch
        .points()
        .iter()
        .find(|point| point.id == head)
        .copied()
        .expect("head");
    let before_circles = sketch.circles().to_vec();
    let before_center = position(&sketch, center);
    let arc = sketch.arcs()[0].id;
    let sweep_of = |sketch: &Sketch| {
        sketch
            .arc_form_of(arc)
            .expect("the arc draws a circle")
            .sweep_degrees
    };
    let before_sweep = sweep_of(&sketch);

    assert!(sketch
        .move_point(center, SketchPoint::new(10, 20), ctx(16))
        .expect("a center is an ordinary point to drag"));
    let by = [10.0 - before_center[0], 20.0 - before_center[1]];
    assert!(
        (sweep_of(&sketch) - before_sweep).abs() < 1.0e-6,
        "the arc travelled without reshaping"
    );
    let carried = |was: &Point, is: EntityId| {
        let at = position(&sketch, is);
        let stood = was.at.in_plane();
        assert!(
            (at[0] - stood[0] - by[0]).abs() < 1.0e-6 && (at[1] - stood[1] - by[1]).abs() < 1.0e-6,
            "{at:?} is not {stood:?} carried by {by:?}"
        );
    };
    carried(&before_tail, tail);
    carried(&before_head, head);
    assert_eq!(
        sketch.circles(),
        before_circles.as_slice(),
        "circle is byte-exact"
    );
}

#[test]
/// A center under a standing Tangent is dragged, not refused. The old model could only rewrite the
/// arc's sweep and had to give up when no sweep satisfied the relation; a placed center pins where
/// the cursor left it and the settle moves the rest of the drawing to keep the tangency (ADR 0038).
fn an_arc_center_drag_under_a_tangent_settles_the_drawing() {
    let (mut sketch, tail, _head, center, _) = arc_with_center();
    let line_from = sketch.add_free_point(SketchPoint::new(-10, 10));
    let line = sketch
        .connect(line_from, tail)
        .expect("endpoint tangent segment");
    let arc = sketch.arcs()[0].id;
    let tangent = sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Arc(arc),
                SketchCurve::Segment(line),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("arc endpoint is tangent to the line");

    assert_eq!(
        sketch.move_point(center, SketchPoint::new(10, 20), ctx(16)),
        Ok(true)
    );
    assert!(
        sketch.arc_form_of(arc).is_some(),
        "the settled arc still draws a circle"
    );
    let ConstraintKind::Tangent {
        first,
        second,
        branch,
    } = sketch
        .constraints()
        .iter()
        .find(|constraint| constraint.id == tangent)
        .expect("the tangent survives the drag")
        .kind
    else {
        panic!("tangent kind")
    };
    assert!(
        sketch
            .tangent_contact(first, second, branch, ctx(16))
            .is_ok(),
        "the two curves still touch"
    );
}

#[test]
/// Fixed scalar sources are authoritative inputs to every adapter path. Neither accepting a
/// relation nor a later ordinary solve may turn them into free solved values.
fn add_and_solve_preserve_fixed_arc_and_circle_authority_byte_exactly() {
    let (mut sketch, tail, _head, _center, _) = arc_with_center();
    sketch
        .add_circle(SketchPoint::new(50, 20), SketchLength::new(7))
        .expect("circle");
    sketch.circles_mut_for_test()[0].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(7));
    let before_arc = sketch.arcs()[0];
    let before_circle = sketch.circles()[0];
    let at = position(&sketch, tail);

    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::from_continuous(at[0], at[1]),
            },
            ctx(16),
        )
        .expect("current point can be fixed");
    assert_eq!(sketch.arcs()[0], before_arc, "add keeps fixed sweep source");
    assert_eq!(
        sketch.circles()[0],
        before_circle,
        "add keeps fixed radius source"
    );

    sketch.solve(ctx(16)).expect("solve fixed source sketch");
    assert_eq!(
        sketch.arcs()[0],
        before_arc,
        "solve keeps fixed sweep source"
    );
    assert_eq!(
        sketch.circles()[0],
        before_circle,
        "solve keeps fixed radius source"
    );
}

#[test]
/// The drag transaction includes preparation itself. A malformed entity the drag can reach makes
/// preparation fail only after the dragged point has tentatively taken the cursor's position; that
/// error must still put the drawing back exactly where it started.
fn arc_center_prepare_error_restores_the_drag_exactly() {
    let (mut source, _tail, _head, center, _) = arc_with_center();
    // A drag prepares only what it can reach, so the malformed circle has to be tied to the arc —
    // an unrelated one is skipped, and skipping it is the point of scoping the solve.
    let arc = source.arcs()[0].id;
    let circle = source
        .add_circle(SketchPoint::new(50, 20), SketchLength::new(7))
        .expect("circle to corrupt in raw payload");
    source
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Arc(arc), SketchCurve::Circle(circle)),
            ctx(16),
        )
        .expect("concentric");
    let mut raw = serde_json::to_value(&source).expect("serialize source");
    raw["circles"][0]["center"] = serde_json::json!(EntityId::MAX);
    let mut loaded: Sketch = serde_json::from_value(raw).expect("structural load");
    let before = serde_json::to_value(&loaded).expect("snapshot");

    assert!(matches!(
        loaded.move_point(center, SketchPoint::new(10, 20), ctx(16)),
        Err(SketchEvaluationError::InvalidDocumentGeometry)
    ));
    assert_eq!(serde_json::to_value(&loaded).expect("after error"), before);
}

#[test]
/// Tangent reads a fixed circle radius as an immutable source while it is free to move other
/// authored geometry. This is the adapter authority boundary, not merely a scalar serde round-trip.
fn tangent_keeps_a_fixed_circle_radius_and_moves_allowed_geometry() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 1));
    let to = sketch.add_free_point(SketchPoint::new(10, 1));
    let segment = sketch.connect(from, to).expect("segment");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
        .expect("circle");
    sketch.circles_mut_for_test()[0].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(4));
    let source = sketch.circles()[0];
    let before = (
        position(&sketch, from),
        position(&sketch, to),
        position(&sketch, sketch.circles()[0].center),
    );

    sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("fixed-radius tangent");
    assert_eq!(
        sketch.circles()[0],
        source,
        "fixed source remains byte-exact"
    );
    assert_ne!(
        (
            position(&sketch, from),
            position(&sketch, to),
            position(&sketch, sketch.circles()[0].center),
        ),
        before,
        "permitted non-scalar geometry moved"
    );
}

#[test]
/// Conversely, with its center and the finite line fixed, Tangent changes a free circle's one
/// writable scalar and leaves it a free value rather than replacing it with a fixed source.
fn tangent_changes_a_free_circle_radius_when_centers_and_line_are_fixed() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(10, 0));
    let segment = sketch.connect(from, to).expect("segment");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 6), SketchLength::new(4))
        .expect("circle");
    let center = sketch.circles()[0].center;
    for point in [from, to, center] {
        let at = position(&sketch, point);
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("fix authored geometry");
    }
    let before = sketch.circles()[0].radius;

    sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("free radius can satisfy tangent");
    let radius = &sketch.circles()[0].radius;
    assert!(radius.free_value().is_some(), "still a free scalar");
    assert_ne!(*radius, before, "Tangent wrote the free radius");
    assert!((radius.free_value().expect("free").value() - 6.0).abs() < 1e-5);
}

/// With both ends and the line `Fix`ed, the arc has ONE freedom left — where its center stands
/// along the chord's bisector — and tangency is met by spending it (ADR 0038).
#[test]
fn tangent_moves_an_arc_center_when_its_ends_and_the_line_are_fixed() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let arc = sketch
        .connect_arc(tail, head, AngleMeasurement::from_degrees(60))
        .expect("free sweep arc");
    let tangent_y = 5.0 - (50.0_f64).sqrt();
    let line_from = sketch.add_free_point(SketchPoint::from_continuous(0.0, tangent_y));
    let line_to = sketch.add_free_point(SketchPoint::from_continuous(10.0, tangent_y));
    let line = sketch
        .connect(line_from, line_to)
        .expect("target tangent segment");
    for point in [tail, head, line_from, line_to] {
        let at = position(&sketch, point);
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("fix authored geometry");
    }
    let sweep_of = |sketch: &Sketch| {
        sketch
            .arc_form_of(arc)
            .expect("the arc draws a circle")
            .sweep_degrees
    };
    let before = sweep_of(&sketch);

    sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Arc(arc),
                SketchCurve::Segment(line),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("a free center can satisfy endpoint tangent");
    let sweep = sweep_of(&sketch);
    assert!((sweep - before).abs() > 1e-6, "Tangent moved the center");
    assert!((sweep - 90.0).abs() < 1e-5, "{sweep:?}");
}

#[test]
/// Every persisted curve pairing crosses the document adapter, not just the parametric formula
/// layer. Circular cases cover external plus both canonical internal-containment branches.
fn document_tangent_adapter_prepares_adds_and_solves_every_curve_pair() {
    let (mut segment_arc, tail, _head, _center, _) = arc_with_center();
    let line_from = segment_arc.add_free_point(SketchPoint::new(-10, 10));
    let line = segment_arc.connect(line_from, tail).expect("line");
    let arc = segment_arc.arcs()[0].id;
    add_and_solve_tangent(
        &mut segment_arc,
        ConstraintKind::tangent(
            SketchCurve::Segment(line),
            SketchCurve::Arc(arc),
            TangentBranch::Line(LineSide::Left),
        ),
    );

    let mut segment_circle = Sketch::empty(PlaneAxis::Z);
    let from = segment_circle.add_free_point(SketchPoint::new(0, 0));
    let to = segment_circle.add_free_point(SketchPoint::new(10, 0));
    let line = segment_circle.connect(from, to).expect("line");
    let circle = segment_circle
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
        .expect("circle");
    add_and_solve_tangent(
        &mut segment_circle,
        ConstraintKind::tangent(
            SketchCurve::Segment(line),
            SketchCurve::Circle(circle),
            TangentBranch::Line(LineSide::Left),
        ),
    );

    // Two semicircles facing one another, touching at (5, 0).
    let mut arc_arc = Sketch::empty(PlaneAxis::Z);
    let a0 = arc_arc.add_free_point(SketchPoint::new(0, 5));
    let a1 = arc_arc.add_free_point(SketchPoint::new(0, -5));
    let first_arc = arc_arc
        .connect_arc(a0, a1, AngleMeasurement::from_degrees(-180))
        .expect("first arc");
    let b0 = arc_arc.add_free_point(SketchPoint::new(10, -5));
    let b1 = arc_arc.add_free_point(SketchPoint::new(10, 5));
    let second_arc = arc_arc
        .connect_arc(b0, b1, AngleMeasurement::from_degrees(-180))
        .expect("second arc");
    add_and_solve_tangent(
        &mut arc_arc,
        ConstraintKind::tangent(
            SketchCurve::Arc(first_arc),
            SketchCurve::Arc(second_arc),
            TangentBranch::External,
        ),
    );

    let mut arc_circle = Sketch::empty(PlaneAxis::Z);
    let a0 = arc_circle.add_free_point(SketchPoint::new(0, 5));
    let a1 = arc_circle.add_free_point(SketchPoint::new(0, -5));
    let arc = arc_circle
        .connect_arc(a0, a1, AngleMeasurement::from_degrees(-180))
        .expect("arc");
    let circle = arc_circle
        .add_circle(SketchPoint::new(10, 0), SketchLength::new(5))
        .expect("circle");
    add_and_solve_tangent(
        &mut arc_circle,
        ConstraintKind::tangent(
            SketchCurve::Arc(arc),
            SketchCurve::Circle(circle),
            TangentBranch::External,
        ),
    );

    let mut circles = Sketch::empty(PlaneAxis::Z);
    let outer = circles
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(10))
        .expect("outer");
    let inner = circles
        .add_circle(SketchPoint::new(5, 0), SketchLength::new(5))
        .expect("inner");
    add_and_solve_tangent(
        &mut circles,
        ConstraintKind::tangent(
            SketchCurve::Circle(outer),
            SketchCurve::Circle(inner),
            TangentBranch::Internal {
                contains: InternalContainment::First,
            },
        ),
    );

    let mut reversed_circles = Sketch::empty(PlaneAxis::Z);
    let outer = reversed_circles
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(10))
        .expect("outer");
    let inner = reversed_circles
        .add_circle(SketchPoint::new(5, 0), SketchLength::new(5))
        .expect("inner");
    add_and_solve_tangent(
        &mut reversed_circles,
        ConstraintKind::tangent(
            SketchCurve::Circle(inner),
            SketchCurve::Circle(outer),
            TangentBranch::Internal {
                contains: InternalContainment::Second,
            },
        ),
    );
}

#[test]
/// A Tangent that follows completely fixed, already-tangent geometry is retained as durable
/// intent and correctly marked redundant; its trial does not perturb the authored drawing.
fn an_implied_fixed_tangent_is_kept_redundant_without_moving_geometry() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(10, 0));
    let segment = sketch.connect(from, to).expect("segment");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(4))
        .expect("circle");
    let center = sketch.circles()[0].center;
    sketch.circles_mut_for_test()[0].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(4));
    for point in [from, to, center] {
        let at = position(&sketch, point);
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("fixed geometry");
    }
    let before = serde_json::to_value(&sketch).expect("snapshot");
    let tangent = sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect("implied tangent is retained");
    assert!(
        sketch
            .constraints()
            .iter()
            .find(|constraint| constraint.id == tangent)
            .expect("stored tangent")
            .redundant
    );
    let after = serde_json::to_value(&sketch).expect("after tangent");
    assert_eq!(after["points"], before["points"], "no point moved");
    assert_eq!(
        after["circles"], before["circles"],
        "no scalar source changed"
    );
}

#[test]
/// A fixed non-tangent drawing refuses Tangent as an ordinary residual conflict and names the
/// fixes that prevent movement. It is not a finite-contact error and the rejected trial is atomic.
fn a_fixed_non_tangent_reports_tangent_conflict_blame_without_writeback() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(10, 0));
    let segment = sketch.connect(from, to).expect("segment");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 4), SketchLength::new(3))
        .expect("non-tangent circle");
    let center = sketch.circles()[0].center;
    sketch.circles_mut_for_test()[0].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(3));
    let mut pins = Vec::new();
    for point in [from, to, center] {
        let at = position(&sketch, point);
        pins.push(
            sketch
                .add_constraint(
                    ConstraintKind::Fix {
                        point,
                        at: SketchPoint::from_continuous(at[0], at[1]),
                    },
                    ctx(16),
                )
                .expect("fixed geometry"),
        );
    }
    let before = serde_json::to_value(&sketch).expect("snapshot");
    let refusal = sketch
        .add_constraint(
            ConstraintKind::tangent(
                SketchCurve::Segment(segment),
                SketchCurve::Circle(circle),
                TangentBranch::Line(LineSide::Left),
            ),
            ctx(16),
        )
        .expect_err("fixed non-tangent cannot settle");
    assert_eq!(
        refusal,
        ConstraintRefusal::Unsatisfiable {
            fights: pins.clone()
        }
    );
    assert_eq!(refusal.culprits(), pins);
    assert_eq!(
        serde_json::to_value(&sketch).expect("after refusal"),
        before
    );
}

#[test]
/// A block-based fixed radius is re-evaluated through the Tangent adapter at the new density;
/// retargeting cannot leave a cached d16 radius behind in a d32 solve.
fn fixed_block_radius_tangent_retargets_from_density_16_to_32_without_losing_authority() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(-32, 0));
    let to = sketch.add_free_point(SketchPoint::new(32, 0));
    let segment = sketch.connect(from, to).expect("long segment");
    let circle = sketch
        .add_circle(SketchPoint::new(0, 16), SketchLength::new(16))
        .expect("one-block circle");
    let source =
        ::parametric::units::Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch.circles_mut_for_test()[0].radius = CircleRadius::fixed(source);
    let before_source = sketch.circles()[0].radius;

    add_and_solve_tangent(
        &mut sketch,
        ConstraintKind::tangent(
            SketchCurve::Segment(segment),
            SketchCurve::Circle(circle),
            TangentBranch::Line(LineSide::Left),
        ),
    );
    assert!((sketch.circles()[0].resolved_radius(ctx(16)) - 16.0).abs() < 1.0e-6);
    sketch.retarget_density(16, 32);
    sketch.solve(ctx(32)).expect("d32 tangent solve");
    assert_eq!(
        sketch.circles()[0].radius,
        before_source,
        "source bytes remain fixed"
    );
    assert!(sketch.circles()[0].radius.fixed_source().is_some());
    assert_eq!(sketch.circles()[0].resolved_radius(ctx(32)), 32.0);
}

fn add_test_arc(sketch: &mut Sketch, center: [i64; 2]) -> EntityId {
    let from = sketch.add_free_point(SketchPoint::new(center[0] - 3, center[1]));
    let to = sketch.add_free_point(SketchPoint::new(center[0] + 3, center[1]));
    sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(180))
        .expect("arc")
}

#[test]
fn concentric_solves_every_document_circular_pair() {
    for (first_arc, second_arc) in [(true, true), (true, false), (false, false)] {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let first = if first_arc {
            SketchCurve::Arc(add_test_arc(&mut sketch, [0, 0]))
        } else {
            SketchCurve::Circle(
                sketch
                    .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
                    .expect("circle"),
            )
        };
        let second = if second_arc {
            SketchCurve::Arc(add_test_arc(&mut sketch, [12, 8]))
        } else {
            SketchCurve::Circle(
                sketch
                    .add_circle(SketchPoint::new(12, 8), SketchLength::new(7))
                    .expect("circle"),
            )
        };
        let constraint = sketch
            .add_constraint(ConstraintKind::concentric(second, first), ctx(16))
            .expect("circular pair");
        let center = sketch
            .concentric_center(first, second)
            .expect("satisfied circular pair has one center");
        let first_center = sketch.circular_curve_center(first).expect("first center");
        assert!((first_center[0] - center[0]).hypot(first_center[1] - center[1]) < 1e-6);
        let ConstraintKind::Concentric {
            first: stored_first,
            second: stored_second,
        } = sketch.constraints()[0].kind
        else {
            panic!("concentric")
        };
        assert!(
            stored_first.id() < stored_second.id(),
            "stable canonical order"
        );
        assert_eq!(
            sketch.add_constraint(ConstraintKind::concentric(first, second), ctx(16)),
            Err(ConstraintRefusal::AlreadyAsserted {
                existing: constraint
            })
        );
    }
}

#[test]
fn concentric_center_rejects_unsatisfied_or_non_circular_pairs() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
        .expect("first");
    let second = sketch
        .add_circle(SketchPoint::new(4, 0), SketchLength::new(3))
        .expect("second");
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(3, 0));
    let segment = sketch.connect(from, to).expect("segment");

    assert!(sketch
        .concentric_center(SketchCurve::Circle(first), SketchCurve::Circle(second))
        .is_none());
    assert!(sketch
        .concentric_center(SketchCurve::Circle(first), SketchCurve::Circle(first))
        .is_none());
    assert!(sketch
        .concentric_center(SketchCurve::Segment(segment), SketchCurve::Circle(first))
        .is_none());
}

pub(super) fn add_test_segment(
    sketch: &mut Sketch,
    from: [i64; 2],
    to: [i64; 2],
) -> (EntityId, EntityId, EntityId) {
    let from = sketch.add_free_point(SketchPoint::new(from[0], from[1]));
    let to = sketch.add_free_point(SketchPoint::new(to[0], to[1]));
    let segment = sketch.connect(from, to).expect("segment");
    (from, to, segment)
}

#[test]
fn symmetry_uses_the_explicit_axis_as_add_preference_then_allows_axis_drag() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (axis_from, axis_to, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (_, _, first) = add_test_segment(&mut sketch, [-6, 0], [-4, 4]);
    let (_, _, second) = add_test_segment(&mut sketch, [8, 1], [7, 6]);
    let (unrelated_from, unrelated_to, _) = add_test_segment(&mut sketch, [20, 2], [27, 5]);
    let first = SketchCurve::Segment(first);
    let second = SketchCurve::Segment(second);
    let axis_before = (position(&sketch, axis_from), position(&sketch, axis_to));
    let unrelated_before = (
        position(&sketch, unrelated_from),
        position(&sketch, unrelated_to),
    );
    let branch = sketch
        .choose_symmetry_branch(first, second, axis, ctx(16))
        .expect("same-kind subjects and live axis");
    sketch
        .add_constraint(
            ConstraintKind::symmetry(second, first, axis, branch),
            ctx(16),
        )
        .expect("free subjects can mirror about the axis");
    assert_eq!(
        (position(&sketch, axis_from), position(&sketch, axis_to)),
        axis_before,
        "the preference pass keeps the explicit reference axis"
    );
    assert_eq!(
        (
            position(&sketch, unrelated_from),
            position(&sketch, unrelated_to),
        ),
        unrelated_before,
        "unrelated rigidity remains a preference"
    );
    let locus = sketch
        .symmetry_badge_locus(first, second, axis, branch, ctx(16))
        .expect("standing symmetry has a witness");
    assert!(
        locus[0].abs() < 1.0e-6,
        "badge lies on the vertical axis: {locus:?}"
    );

    assert!(sketch
        .move_point(axis_to, SketchPoint::new(3, 11), ctx(16))
        .expect("drag evaluates"));
    assert_ne!(
        position(&sketch, axis_to),
        axis_before.1,
        "the axis is not fixed"
    );
    sketch
        .symmetry_badge_locus(first, second, axis, branch, ctx(16))
        .expect("subjects follow the dragged axis");
}

#[test]
fn symmetry_anchors_only_axis_endpoints_when_a_subject_shares_axis_topology() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let axis_from = sketch.add_free_point(SketchPoint::new(0, -10));
    let shared = sketch.add_free_point(SketchPoint::new(0, 10));
    let axis = sketch.connect(axis_from, shared).expect("axis");
    let movable = sketch.add_free_point(SketchPoint::new(-8, 4));
    let first = sketch.connect(shared, movable).expect("connected subject");
    let fixed_from = sketch.add_free_point(SketchPoint::new(0, 10));
    let fixed_to = sketch.add_free_point(SketchPoint::new(3, 6));
    let second = sketch.connect(fixed_from, fixed_to).expect("fixed subject");
    for (point, at) in [
        (fixed_from, SketchPoint::new(0, 10)),
        (fixed_to, SketchPoint::new(3, 6)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("fix subject endpoint");
    }
    let axis_before = (position(&sketch, axis_from), position(&sketch, shared));
    let movable_before = position(&sketch, movable);
    sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first),
                SketchCurve::Segment(second),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        )
        .expect("the connected subject endpoint remains driven");
    assert_eq!(
        (position(&sketch, axis_from), position(&sketch, shared)),
        axis_before
    );
    assert_ne!(position(&sketch, movable), movable_before);
}

#[test]
fn symmetry_normalizes_subjects_and_deduplicates_independently_of_branch() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (_, _, first) = add_test_segment(&mut sketch, [-4, 0], [-4, 3]);
    let (_, _, second) = add_test_segment(&mut sketch, [4, 0], [4, 3]);
    let first = SketchCurve::Segment(first);
    let second = SketchCurve::Segment(second);
    let constraint = sketch
        .add_constraint(
            ConstraintKind::symmetry(second, first, axis, SymmetryBranch::Direct),
            ctx(16),
        )
        .expect("already symmetric");
    let ConstraintKind::Symmetry {
        first: stored_first,
        second: stored_second,
        axis: stored_axis,
        branch,
    } = sketch.constraints()[0].kind
    else {
        panic!("symmetry")
    };
    assert!(stored_first.id() < stored_second.id());
    assert_eq!(stored_axis, axis);
    assert_eq!(branch, SymmetryBranch::Direct);
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::symmetry(first, second, axis, SymmetryBranch::Reversed),
            ctx(16),
        ),
        Err(ConstraintRefusal::AlreadyAsserted {
            existing: constraint
        })
    );
}

#[test]
fn symmetry_refuses_bad_structure_and_degenerate_axes_before_solving() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (_, _, first_segment) = add_test_segment(&mut sketch, [-4, 0], [-4, 3]);
    let circle = sketch
        .add_circle(SketchPoint::new(4, 1), SketchLength::new(2))
        .expect("circle");
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first_segment),
                SketchCurve::Circle(circle),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::InvalidSymmetry)
    );
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first_segment),
                SketchCurve::Segment(first_segment),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::InvalidSymmetry)
    );
    let (_, _, second_segment) = add_test_segment(&mut sketch, [4, 0], [4, 3]);
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first_segment),
                SketchCurve::Segment(second_segment),
                first_segment,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::InvalidSymmetry)
    );
    let degenerate_from = sketch.add_free_point(SketchPoint::new(20, 20));
    let degenerate_to = sketch.add_free_point(SketchPoint::new(20, 20));
    let degenerate = sketch
        .connect(degenerate_from, degenerate_to)
        .expect("distinct ids");
    assert_eq!(
        sketch.add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first_segment),
                SketchCurve::Segment(second_segment),
                degenerate,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::InvalidSymmetry)
    );
}

#[test]
fn public_symmetry_queries_reject_invalid_persisted_identities() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (_, _, first) = add_test_segment(&mut sketch, [-4, 0], [-4, 3]);
    let (_, _, second) = add_test_segment(&mut sketch, [4, 0], [4, 3]);
    let circle = sketch
        .add_circle(SketchPoint::new(4, 1), SketchLength::new(2))
        .expect("circle");
    assert!(sketch
        .choose_symmetry_branch(
            SketchCurve::Segment(first),
            SketchCurve::Segment(first),
            axis,
            ctx(16),
        )
        .is_err());
    assert!(sketch
        .choose_symmetry_branch(
            SketchCurve::Segment(first),
            SketchCurve::Circle(circle),
            axis,
            ctx(16),
        )
        .is_err());
    assert!(sketch
        .choose_symmetry_branch(
            SketchCurve::Segment(first),
            SketchCurve::Segment(second),
            first,
            ctx(16),
        )
        .is_err());
    assert!(sketch
        .symmetry_badge_locus(
            SketchCurve::Segment(first),
            SketchCurve::Segment(second),
            axis,
            SymmetryBranch::Centers,
            ctx(16),
        )
        .is_err());
}

#[test]
fn circle_symmetry_equalizes_free_radii_and_fixed_disagreement_is_unsatisfiable() {
    let make = || {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let (_, _, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
        let first = sketch
            .add_circle(SketchPoint::new(-5, 0), SketchLength::new(2))
            .expect("circle");
        let second = sketch
            .add_circle(SketchPoint::new(5, 0), SketchLength::new(6))
            .expect("circle");
        (sketch, axis, first, second)
    };
    let (mut free, axis, first, second) = make();
    free.add_constraint(
        ConstraintKind::symmetry(
            SketchCurve::Circle(first),
            SketchCurve::Circle(second),
            axis,
            SymmetryBranch::Centers,
        ),
        ctx(16),
    )
    .expect("free radii can equalize");
    assert!(
        (free.circles()[0].resolved_radius(ctx(16)) - free.circles()[1].resolved_radius(ctx(16)))
            .abs()
            < 1.0e-6
    );

    let (mut fixed, axis, first, second) = make();
    fixed.circles_mut_for_test()[0].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(2));
    fixed.circles_mut_for_test()[1].radius =
        CircleRadius::fixed(::parametric::units::Measurement::from_voxels(6));
    assert!(matches!(
        fixed.add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Circle(first),
                SketchCurve::Circle(second),
                axis,
                SymmetryBranch::Centers,
            ),
            ctx(16),
        ),
        Err(ConstraintRefusal::Unsatisfiable { .. })
    ));
}

#[test]
fn circle_symmetry_reads_fixed_radius_authority_at_each_density() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -40], [0, 40]);
    let free = sketch
        .add_circle(SketchPoint::new(-20, 0), SketchLength::new(3))
        .expect("free circle");
    let fixed = sketch
        .add_circle(SketchPoint::new(20, 0), SketchLength::new(1))
        .expect("fixed circle");
    let source =
        ::parametric::units::Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch.circles_mut_for_test()[1].radius = CircleRadius::fixed(source);
    let fixed_bits = serde_json::to_vec(&sketch.circles()[1].radius).expect("fixed source");
    sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Circle(free),
                SketchCurve::Circle(fixed),
                axis,
                SymmetryBranch::Centers,
            ),
            ctx(16),
        )
        .expect("free radius follows fixed source");
    assert!((sketch.circles()[0].resolved_radius(ctx(16)) - 16.0).abs() < 1.0e-6);
    assert_eq!(
        serde_json::to_vec(&sketch.circles()[1].radius).expect("fixed source"),
        fixed_bits
    );
    sketch.solve(ctx(32)).expect("density re-evaluation");
    assert!((sketch.circles()[0].resolved_radius(ctx(32)) - 32.0).abs() < 1.0e-6);
    assert_eq!(
        serde_json::to_vec(&sketch.circles()[1].radius).expect("fixed source"),
        fixed_bits
    );
}

/// Two arcs held symmetric agree about how far they turn — and an arc has no scalar to be told
/// that with, so the free one gets there by MOVING (ADR 0038).
///
/// Both branches are asked because a reflection reverses the sense of travel and a stored arc has
/// only one sense, so the branch has nothing left to choose between for a pair of arcs. The two
/// answers must therefore be the same answer.
#[test]
fn arc_symmetry_moves_the_free_arc_to_match_its_pinned_partner() {
    for branch in [SymmetryBranch::Direct, SymmetryBranch::Reversed] {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let (_, _, axis) = add_test_segment(&mut sketch, [0, -20], [0, 20]);
        let first_from = sketch.add_free_point(SketchPoint::new(-8, 0));
        let first_to = sketch.add_free_point(SketchPoint::new(-2, 0));
        let first = sketch
            .connect_arc(first_from, first_to, AngleMeasurement::from_degrees(90))
            .expect("free arc");
        let second_from = sketch.add_free_point(SketchPoint::new(2, 0));
        let second_to = sketch.add_free_point(SketchPoint::new(8, 0));
        let second = sketch
            .connect_arc(second_from, second_to, AngleMeasurement::from_degrees(120))
            .expect("the partner arc");
        let pinned = sketch.arcs()[1];
        for point in [pinned.from, pinned.to, pinned.center] {
            let at = position(&sketch, point);
            sketch
                .add_constraint(
                    ConstraintKind::Fix {
                        point,
                        at: SketchPoint::from_continuous(at[0], at[1]),
                    },
                    ctx(16),
                )
                .expect("pin the partner");
        }
        sketch
            .add_constraint(
                ConstraintKind::symmetry(
                    SketchCurve::Arc(first),
                    SketchCurve::Arc(second),
                    axis,
                    branch,
                ),
                ctx(16),
            )
            .expect("the free arc follows its pinned partner");
        let sweep = |id| {
            sketch
                .arc_form_of(id)
                .expect("the arc draws a circle")
                .sweep_degrees
        };
        assert!((sweep(first) - 120.0).abs() < 1.0e-3, "{}", sweep(first));
        assert!((sweep(second) - 120.0).abs() < 1.0e-3, "{}", sweep(second));
    }
}

#[test]
fn symmetry_round_trips_repairs_degenerate_axes_and_cascades_axis_deletion() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (axis_from, axis_to, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (_, _, first) = add_test_segment(&mut sketch, [-4, 0], [-4, 3]);
    let (_, _, second) = add_test_segment(&mut sketch, [4, 0], [4, 3]);
    sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(second),
                SketchCurve::Segment(first),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        )
        .expect("symmetry");
    let json = serde_json::to_string(&sketch).expect("serialize");
    let mut loaded: Sketch = serde_json::from_str(&json).expect("deserialize");
    let ConstraintKind::Symmetry {
        first: stored_first,
        second: stored_second,
        axis: stored_axis,
        branch: SymmetryBranch::Direct,
    } = loaded.constraints()[0].kind
    else {
        panic!("stored symmetry")
    };
    assert!(stored_first.id() < stored_second.id());
    assert_eq!(stored_axis, axis);

    let collapsed_at = loaded
        .points
        .iter()
        .find(|point| point.id == axis_from)
        .expect("axis endpoint")
        .at;
    loaded
        .points
        .iter_mut()
        .find(|point| point.id == axis_to)
        .expect("axis endpoint")
        .at = collapsed_at;
    assert_eq!(
        loaded.repair(ctx(16)),
        1,
        "only the invalid relation is dropped"
    );
    assert!(loaded.constraints().is_empty());

    let mut cascade: Sketch = serde_json::from_str(&json).expect("deserialize again");
    cascade.delete_segment(axis);
    assert!(cascade.constraints().is_empty());
}

#[test]
fn loaded_conflicting_symmetry_keeps_solve_and_drag_byte_atomic_with_sorted_blame() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (_, _, axis) = add_test_segment(&mut sketch, [0, -10], [0, 10]);
    let (first_from, first_to, first) = add_test_segment(&mut sketch, [-4, 0], [-4, 3]);
    let (second_from, second_to, second) = add_test_segment(&mut sketch, [4, 0], [4, 3]);
    let symmetry = sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first),
                SketchCurve::Segment(second),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        )
        .expect("symmetry");
    let mut last_fix = None;
    for point in [first_from, first_to, second_from, second_to] {
        let at = position(&sketch, point);
        last_fix = Some(
            sketch
                .add_constraint(
                    ConstraintKind::Fix {
                        point,
                        at: SketchPoint::from_continuous(at[0], at[1]),
                    },
                    ctx(16),
                )
                .expect("compatible fix"),
        );
    }
    let last_fix = last_fix.expect("fix");
    let held = sketch
        .constraints
        .iter_mut()
        .find(|constraint| constraint.id == last_fix)
        .expect("fix constraint");
    let ConstraintKind::Fix { point, at } = held.kind else {
        panic!("fix")
    };
    held.kind = ConstraintKind::Fix {
        point,
        at: SketchPoint::from_continuous(at.in_plane()[0] + 2.0, at.in_plane()[1]),
    };
    let raw = serde_json::to_vec(&sketch).expect("malformed standing document");
    let mut loaded: Sketch = serde_json::from_slice(&raw).expect("structural load");
    let before = serde_json::to_vec(&loaded).expect("before solve");
    let Err(SketchEvaluationError::Unsatisfied { conflicts }) = loaded.solve(ctx(16)) else {
        panic!("standing conflict")
    };
    assert!(conflicts.windows(2).all(|ids| ids[0] < ids[1]));
    assert!(conflicts.contains(&symmetry) && conflicts.contains(&last_fix));
    assert_eq!(serde_json::to_vec(&loaded).expect("after solve"), before);
    assert_eq!(
        loaded.move_point(first_from, SketchPoint::new(-9, 2), ctx(16)),
        Ok(false)
    );
    assert_eq!(serde_json::to_vec(&loaded).expect("after drag"), before);
}

#[test]
/// Repair seats an arc center before it judges anything that reads the center. The stored dot has
/// drifted along the chord — the one direction that names no different circle (ADR 0038) — onto the
/// other end of a symmetry axis, and an axis with both ends in one place would be erased along with
/// the relation standing on it. Seating first is the difference between a nudge and a deletion.
fn repair_seats_an_arc_center_before_validating_a_symmetry_axis() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    add_test_arc(&mut sketch, [0, 0]);
    let center = sketch.arcs()[0].center;
    let axis_to = sketch.add_free_point(SketchPoint::new(7, 10));
    let axis = sketch.connect(center, axis_to).expect("center-based axis");
    let (_, _, first) = add_test_segment(&mut sketch, [-4, 2], [-4, 5]);
    let (_, _, second) = add_test_segment(&mut sketch, [4, 2], [4, 5]);
    sketch
        .add_constraint(
            ConstraintKind::symmetry(
                SketchCurve::Segment(first),
                SketchCurve::Segment(second),
                axis,
                SymmetryBranch::Direct,
            ),
            ctx(16),
        )
        .expect("symmetry");
    sketch
        .points
        .iter_mut()
        .find(|point| point.id == center)
        .expect("the arc center")
        .at = SketchPoint::new(7, 10);
    let raw = serde_json::to_vec(&sketch).expect("a center adrift along its chord");
    let mut loaded: Sketch = serde_json::from_slice(&raw).expect("structural load");
    assert_eq!(loaded.repair(ctx(16)), 0);
    assert_eq!(loaded.constraints().len(), 1);
    let center = position(&loaded, center);
    assert!(center[0].abs() < 1.0e-12, "seated back onto the bisector");
    assert!((center[1] - 10.0).abs() < 1.0e-12, "and no further");
}

#[test]
fn concentric_keeps_unequal_radius_authorities_exact_across_density() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
        .expect("free circle");
    let second = sketch
        .add_circle(SketchPoint::new(9, 5), SketchLength::new(7))
        .expect("fixed circle");
    let source =
        ::parametric::units::Measurement::new(::parametric::ExactRational::from_integer(1), 0);
    sketch.circles_mut_for_test()[1].radius = CircleRadius::fixed(source);
    let before_free = sketch.circles()[0].radius;
    let before_fixed = sketch.circles()[1].radius;
    sketch
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Circle(first), SketchCurve::Circle(second)),
            ctx(16),
        )
        .expect("concentric circles");
    assert_eq!(sketch.circles()[0].radius, before_free);
    assert_eq!(sketch.circles()[1].radius, before_fixed);
    assert_ne!(
        sketch.circles()[0].resolved_radius(ctx(16)).to_bits(),
        sketch.circles()[1].resolved_radius(ctx(16)).to_bits()
    );

    sketch.retarget_density(16, 32);
    let retargeted_free = sketch.circles()[0].radius;
    sketch.solve(ctx(32)).expect("density-aware preparation");
    assert_eq!(sketch.circles()[0].radius, retargeted_free);
    assert_eq!(sketch.circles()[1].radius, before_fixed);
    assert_eq!(sketch.circles()[1].resolved_radius(ctx(32)), 32.0);
}

/// With its two ends `Fix`ed, an arc held concentric with a fixed-center circle has exactly one
/// place left to put its center — and getting there is the whole of what concentric means to an
/// arc now (ADR 0038).
#[test]
fn concentric_places_an_arcs_center_on_the_circles() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(10, 0));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("free arc");
    let circle = sketch
        .add_circle(SketchPoint::new(5, 0), SketchLength::new(3))
        .expect("fixed-center target");
    let circle_center = sketch.circles()[0].center;
    for (point, at) in [
        (from, [0.0, 0.0]),
        (to, [10.0, 0.0]),
        (circle_center, [5.0, 0.0]),
    ] {
        sketch
            .add_constraint(
                ConstraintKind::Fix {
                    point,
                    at: SketchPoint::from_continuous(at[0], at[1]),
                },
                ctx(16),
            )
            .expect("fixed point");
    }
    sketch
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Arc(arc), SketchCurve::Circle(circle)),
            ctx(16),
        )
        .expect("a free center can be brought onto the circle's");
    let solved = sketch
        .arc_form_of(arc)
        .expect("the arc draws a circle")
        .sweep_degrees;
    assert!((solved - 180.0).abs() < 1e-3, "solved sweep was {solved}");
}

#[test]
fn concentric_leaves_an_already_concentric_arc_alone() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let from = sketch.add_free_point(SketchPoint::new(0, 0));
    let to = sketch.add_free_point(SketchPoint::new(10, 0));
    let arc = sketch
        .connect_arc(from, to, AngleMeasurement::from_degrees(90))
        .expect("arc");
    let center = sketch
        .circular_curve_center(SketchCurve::Arc(arc))
        .expect("derived center");
    let circle = sketch
        .add_circle(
            SketchPoint::from_continuous(center[0], center[1]),
            SketchLength::new(6),
        )
        .expect("circle");
    let before = sketch
        .arc_form_of(arc)
        .expect("the arc draws a circle")
        .sweep_degrees;
    sketch
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Arc(arc), SketchCurve::Circle(circle)),
            ctx(16),
        )
        .expect("already concentric");
    sketch.solve(ctx(32)).expect("solve");
    let after = sketch
        .arc_form_of(arc)
        .expect("the arc draws a circle")
        .sweep_degrees;
    assert!((after - before).abs() < 1e-3, "{before} became {after}");
}

#[test]
fn concentric_repair_canonicalizes_deduplicates_and_cascades() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
        .expect("first");
    let second = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(4))
        .expect("second");
    let from = sketch.add_free_point(SketchPoint::new(-3, 0));
    let to = sketch.add_free_point(SketchPoint::new(3, 0));
    let segment = sketch.connect(from, to).expect("segment");
    let mut raw = serde_json::to_value(&sketch).expect("source");
    raw["constraints"] = serde_json::json!([
        {"id": 90, "kind": {"Concentric": {
            "first": {"Circle": second}, "second": {"Circle": first}
        }}, "redundant": false},
        {"id": 91, "kind": {"Concentric": {
            "first": {"Circle": first}, "second": {"Circle": second}
        }}, "redundant": false},
        {"id": 92, "kind": {"Concentric": {
            "first": {"Circle": first}, "second": {"Circle": first}
        }}, "redundant": false},
        {"id": 93, "kind": {"Concentric": {
            "first": {"Segment": segment}, "second": {"Circle": first}
        }}, "redundant": false},
        {"id": 94, "kind": {"Concentric": {
            "first": {"Circle": 999}, "second": {"Circle": first}
        }}, "redundant": false}
    ]);
    let mut loaded: Sketch = serde_json::from_value(raw).expect("load");
    assert!(matches!(
        loaded.constraints()[0].kind,
        ConstraintKind::Concentric {
            first: SketchCurve::Circle(a),
            second: SketchCurve::Circle(b)
        } if (a, b) == (first, second)
    ));
    assert_eq!(loaded.repair(ctx(16)), 4);
    assert_eq!(loaded.constraints().len(), 1);
    assert_eq!(loaded.constraints()[0].id, 90);
    loaded.delete_circle(first);
    assert!(loaded.constraints().is_empty(), "curve deletion cascades");
}

#[test]
fn fixed_separate_centers_refuse_concentric_with_blame_and_no_writeback() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let first = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
        .expect("first");
    let second = sketch
        .add_circle(SketchPoint::new(10, 0), SketchLength::new(5))
        .expect("second");
    let centers = [sketch.circles()[0].center, sketch.circles()[1].center];
    let mut pins = Vec::new();
    for (point, at) in centers.into_iter().zip([[0.0, 0.0], [10.0, 0.0]]) {
        pins.push(
            sketch
                .add_constraint(
                    ConstraintKind::Fix {
                        point,
                        at: SketchPoint::from_continuous(at[0], at[1]),
                    },
                    ctx(16),
                )
                .expect("pin"),
        );
    }
    let before = serde_json::to_value(&sketch).expect("before");
    let refusal = sketch
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Circle(first), SketchCurve::Circle(second)),
            ctx(16),
        )
        .expect_err("fixed separate centers conflict");
    assert_eq!(refusal, ConstraintRefusal::Unsatisfiable { fights: pins });
    assert_eq!(serde_json::to_value(&sketch).expect("after"), before);
}

#[test]
fn loaded_conflicting_concentric_solve_blames_stable_constraints_without_writeback() {
    let mut source = Sketch::empty(PlaneAxis::Z);
    let first = source
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(2))
        .expect("first");
    let second = source
        .add_circle(SketchPoint::new(10, 0), SketchLength::new(5))
        .expect("second");
    let [first_center, second_center] = [source.circles()[0].center, source.circles()[1].center];
    source.constraints_mut_for_test().extend([
        Constraint {
            id: 40,
            kind: ConstraintKind::Concentric {
                first: SketchCurve::Circle(first),
                second: SketchCurve::Circle(second),
            },
            redundant: false,
            anchor: None,
        },
        Constraint {
            id: 9,
            kind: ConstraintKind::Fix {
                point: first_center,
                at: SketchPoint::new(0, 0),
            },
            redundant: false,
            anchor: None,
        },
        Constraint {
            id: 13,
            kind: ConstraintKind::Fix {
                point: second_center,
                at: SketchPoint::new(10, 0),
            },
            redundant: false,
            anchor: None,
        },
    ]);
    let encoded = serde_json::to_string(&source).expect("persisted source");
    let mut loaded: Sketch = serde_json::from_str(&encoded).expect("loaded conflict");
    let before = serde_json::to_string(&loaded).expect("before solve");

    assert_eq!(
        loaded.solve(ctx(16)),
        Err(SketchEvaluationError::Unsatisfied {
            conflicts: vec![9, 13, 40],
        })
    );
    assert_eq!(serde_json::to_string(&loaded).expect("after solve"), before);
    assert_eq!(
        loaded.move_point(first_center, SketchPoint::new(2, 3), ctx(16)),
        Ok(false)
    );
    assert_eq!(serde_json::to_string(&loaded).expect("after drag"), before);
}

#[test]
fn derived_center_drag_rolls_back_against_loaded_conflicting_concentric() {
    let mut source = Sketch::empty(PlaneAxis::Z);
    let arc = add_test_arc(&mut source, [0, 0]);
    let arc_center = source.arcs()[0].center;
    let circle = source
        .add_circle(SketchPoint::new(10, 0), SketchLength::new(5))
        .expect("circle");
    let circle_center = source.circles()[0].center;
    source.constraints_mut_for_test().extend([
        Constraint {
            id: 40,
            kind: ConstraintKind::Concentric {
                first: SketchCurve::Arc(arc),
                second: SketchCurve::Circle(circle),
            },
            redundant: false,
            anchor: None,
        },
        Constraint {
            id: 9,
            kind: ConstraintKind::Fix {
                point: arc_center,
                at: SketchPoint::new(0, 0),
            },
            redundant: false,
            anchor: None,
        },
        Constraint {
            id: 13,
            kind: ConstraintKind::Fix {
                point: circle_center,
                at: SketchPoint::new(10, 0),
            },
            redundant: false,
            anchor: None,
        },
    ]);
    let encoded = serde_json::to_string(&source).expect("persisted source");
    let mut loaded: Sketch = serde_json::from_str(&encoded).expect("loaded conflict");
    let before = serde_json::to_string(&loaded).expect("before drag");

    assert_eq!(
        loaded.move_point(arc_center, SketchPoint::new(0, 4), ctx(16)),
        Ok(false)
    );
    assert_eq!(serde_json::to_string(&loaded).expect("after drag"), before);
}

#[test]
fn concentric_api_refuses_self_pairs_and_segments_without_mutation() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let circle = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(3))
        .expect("circle");
    let from = sketch.add_free_point(SketchPoint::new(-4, 0));
    let to = sketch.add_free_point(SketchPoint::new(4, 0));
    let segment = sketch.connect(from, to).expect("segment");
    let before = serde_json::to_value(&sketch).expect("before");
    for kind in [
        ConstraintKind::concentric(SketchCurve::Circle(circle), SketchCurve::Circle(circle)),
        ConstraintKind::concentric(SketchCurve::Segment(segment), SketchCurve::Circle(circle)),
    ] {
        assert_eq!(
            sketch.add_constraint(kind, ctx(16)),
            Err(ConstraintRefusal::InvalidConcentric)
        );
        assert_eq!(serde_json::to_value(&sketch).expect("after"), before);
    }
}

#[test]
/// Concentric is what makes a center drag carry its partner. Both centers are placed points now, so
/// the relation is two points holding one spot and the drag moves the pair (ADR 0038).
fn dragging_one_concentric_center_carries_the_other() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let arc = add_test_arc(&mut sketch, [0, 0]);
    let arc_center = sketch.arcs()[0].center;
    let circle = sketch
        .add_circle(SketchPoint::new(0, 0), SketchLength::new(5))
        .expect("circle");
    sketch
        .add_constraint(
            ConstraintKind::concentric(SketchCurve::Arc(arc), SketchCurve::Circle(circle)),
            ctx(16),
        )
        .expect("concentric");
    assert_eq!(
        sketch.move_point(arc_center, SketchPoint::new(0, 4), ctx(16)),
        Ok(true)
    );
    let dragged = position(&sketch, arc_center);
    let carried = position(&sketch, sketch.circles()[0].center);
    assert!((dragged[1] - 4.0).abs() < 1.0e-9, "dragged to {dragged:?}");
    assert!(
        (dragged[0] - carried[0]).hypot(dragged[1] - carried[1]) < 1.0e-9,
        "the circle stayed at {carried:?}"
    );
}

/// The kind-level answer and the geometry-level one are the same fact, so they are held to each
/// other rather than kept in step by hand.
///
/// A gesture refuses an aggregate BEFORE the click lands, using
/// [`SketchCurve::carries_relation_geometry`]; the relations refuse it at solve time by having no
/// geometry to read. If those two ever disagreed, one direction would take a pick it could not
/// apply and the other would turn away a curve that works.
#[test]
fn a_curve_kind_carries_relation_geometry_exactly_when_the_drawing_can_produce_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let every_kind = [
        SketchCurve::Segment(sketch.connect(tail, head).expect("a segment")),
        SketchCurve::Arc(
            sketch
                .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
                .expect("an arc"),
        ),
        SketchCurve::Circle(
            sketch
                .add_circle(SketchPoint::new(0, 20), SketchLength::new(4))
                .expect("a circle"),
        ),
        SketchCurve::Bezier(
            sketch
                .add_cubic_bezier([
                    SketchPoint::new(0, 30),
                    SketchPoint::new(4, 34),
                    SketchPoint::new(8, 34),
                    SketchPoint::new(12, 30),
                ])
                .expect("a bezier"),
        ),
        SketchCurve::Ellipse(
            sketch
                .add_ellipse(
                    SketchPoint::new(0, 40),
                    SketchPoint::new(10, 40),
                    SketchPoint::new(0, 44),
                )
                .expect("an ellipse"),
        ),
        SketchCurve::Conic(
            sketch
                .add_conic(
                    SketchPoint::new(0, 50),
                    SketchPoint::new(10, 50),
                    SketchPoint::new(5, 55),
                    0.5,
                )
                .expect("a conic"),
        ),
        SketchCurve::Spline(
            sketch
                .add_fit_point_spline(
                    &[
                        SketchPoint::new(0, 60),
                        SketchPoint::new(5, 64),
                        SketchPoint::new(10, 60),
                    ],
                    false,
                )
                .expect("a spline"),
        ),
    ];
    for curve in every_kind {
        assert_eq!(
            curve.carries_relation_geometry(),
            sketch.curve_geometry(curve, ctx(16)).is_some(),
            "the two answers disagree for {curve:?}"
        );
    }
}

/// **A point can be held ON a spline, and among the aggregates only on a spline.**
///
/// The sibling of the invariant above, asking the wider question. Reading a shape and standing
/// somewhere along one are different demands: a spline has no center, radius or direction and
/// still has a place everywhere along it, so the two predicates are deliberately not the same.
/// This is where the difference is stated, rather than left to be inferred from which picks the
/// drawing happens to accept.
#[test]
fn a_curve_kind_can_hold_a_point_exactly_when_the_drawing_accepts_the_coincidence() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let every_kind = [
        SketchCurve::Segment(sketch.connect(tail, head).expect("a segment")),
        SketchCurve::Arc(
            sketch
                .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
                .expect("an arc"),
        ),
        SketchCurve::Circle(
            sketch
                .add_circle(SketchPoint::new(0, 20), SketchLength::new(4))
                .expect("a circle"),
        ),
        SketchCurve::Bezier(
            sketch
                .add_cubic_bezier([
                    SketchPoint::new(0, 30),
                    SketchPoint::new(4, 34),
                    SketchPoint::new(8, 34),
                    SketchPoint::new(12, 30),
                ])
                .expect("a bezier"),
        ),
        SketchCurve::Ellipse(
            sketch
                .add_ellipse(
                    SketchPoint::new(0, 40),
                    SketchPoint::new(10, 40),
                    SketchPoint::new(0, 44),
                )
                .expect("an ellipse"),
        ),
        SketchCurve::Conic(
            sketch
                .add_conic(
                    SketchPoint::new(0, 50),
                    SketchPoint::new(10, 50),
                    SketchPoint::new(5, 55),
                    0.5,
                )
                .expect("a conic"),
        ),
        SketchCurve::Spline(
            sketch
                .add_fit_point_spline(
                    &[
                        SketchPoint::new(0, 60),
                        SketchPoint::new(5, 64),
                        SketchPoint::new(10, 60),
                    ],
                    false,
                )
                .expect("a spline"),
        ),
    ];
    for (step, curve) in every_kind.into_iter().enumerate() {
        // A fresh point each time, standing well off every curve so nothing but the coincidence
        // under test decides the answer.
        let standing = sketch.add_free_point(SketchPoint::from_continuous(
            -20.0,
            5.0 * f64::from(u32::try_from(step).expect("a small count")),
        ));
        let held = sketch
            .add_constraint(
                ConstraintKind::Coincident {
                    point: standing,
                    onto: CoincidentTarget::Curve(curve),
                },
                ctx(16),
            )
            .is_ok();
        assert_eq!(
            curve.can_hold_a_point(),
            held,
            "the two answers disagree for {curve:?}"
        );
    }
}

/// **The refused self-reference can be spelled in two steps, and the composed system behaves.**
///
/// A point that shapes a spline is turned away from standing on it, and that refusal is UX rather
/// than a safety rail: a stand-in point coincident to the control point AND to the spline states
/// the same system through a door that stays open. Worth knowing which it is. If the composed
/// spelling fell over, the refusal would be load-bearing and the two-step door a hole; it does
/// not, so the refusal is only the drawing declining to write a claim with no content in it.
#[test]
fn the_self_reference_a_spline_refuses_can_still_be_composed_in_two_steps() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let spline = sketch
        .add_control_point_spline(&[
            SketchPoint::new(0, 0),
            SketchPoint::new(4, 8),
            SketchPoint::new(12, 8),
            SketchPoint::new(16, 0),
        ])
        .expect("a spline");
    let control = sketch
        .splines
        .iter()
        .find(|held| held.id == spline)
        .expect("the spline")
        .points[1];

    let stand_in = sketch.add_free_point(SketchPoint::from_continuous(4.5, 7.0));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: stand_in,
                onto: CoincidentTarget::Curve(SketchCurve::Spline(spline)),
            },
            ctx(16),
        )
        .expect("the stand-in can stand on the spline");
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: stand_in,
                onto: CoincidentTarget::Point(control),
            },
            ctx(16),
        )
        .expect("and can be tied to the control point");

    // The composed system settles, both halves hold, and nothing ran away.
    let (at, on) = (position(&sketch, stand_in), position(&sketch, control));
    assert!(
        (at[0] - on[0]).hypot(at[1] - on[1]) < 1.0e-6,
        "the two points came apart: {at:?} against {on:?}"
    );
    assert!(
        at.iter().all(|value| value.abs() < 1.0e3),
        "the composed system ran away to {at:?}"
    );
}

/// **A point held to a spline rides the curve when the curve is redrawn.**
///
/// Which is the whole of the difference between a coincidence and a snap. A snap puts the point
/// where the curve was when the author clicked; the hold keeps it there afterwards, and the only
/// way to keep a point on a spline is for the solve to be free to choose where along it stands.
#[test]
fn a_point_held_to_a_spline_rides_it_when_the_spline_is_redrawn() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let spline = sketch
        .add_fit_point_spline(
            &[
                SketchPoint::new(0, 0),
                SketchPoint::new(10, 6),
                SketchPoint::new(20, 0),
            ],
            false,
        )
        .expect("a spline");
    let middle = sketch
        .splines
        .iter()
        .find(|held| held.id == spline)
        .expect("the spline")
        .points[1];
    let standing = sketch.add_free_point(SketchPoint::from_continuous(10.0, 2.0));
    sketch
        .add_constraint(
            ConstraintKind::Coincident {
                point: standing,
                onto: CoincidentTarget::Curve(SketchCurve::Spline(spline)),
            },
            ctx(16),
        )
        .expect("a point can stand on a spline");
    let landed = position(&sketch, standing);
    assert!(
        (landed[1] - 2.0).abs() > 1.0,
        "the hold should have pulled the point onto the curve, it sits at {landed:?}"
    );

    // Redraw the spline under it by pinning the middle fit point somewhere else.
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: middle,
                at: SketchPoint::new(10, 16),
            },
            ctx(16),
        )
        .expect("the fit point can be pinned");
    let rode = position(&sketch, standing);
    assert!(
        rode[1] > landed[1] + 1.0,
        "the point should have been carried up with the curve: {landed:?} to {rode:?}"
    );
}

/// A rectangle dragged by a corner RESIZES. It does not slide across the plane.
///
/// The rigidity preference prices each segment by its whole span vector, and the spans around a
/// closed loop sum to zero — so holding the two edges a corner does not touch states the two it
/// does, and the horizontals and verticals then split that statement into four pinned edges. A
/// rigid shape under a pinned hand has exactly one move left, and the rectangle used to make it:
/// all four corners travelled together and the drawing kept the size it was drawn.
///
/// The unit of loosening is the biconnected block the hand stands in, so the whole loop gives way
/// at once. What this test watches is the OPPOSITE corner: it is the one the hand never touches
/// and the one a translation would take along.
#[test]
fn dragging_a_rectangles_corner_resizes_it_rather_than_moving_it() {
    let drawn = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 4)
        .with_rectangle(SketchPoint::new(0, 0), SketchPoint::new(20, 10), ctx(16))
        .expect("a rectangle")
        .sketch
        .as_ref()
        .clone();
    let corner_at = |sketch: &Sketch, want: [f64; 2]| {
        sketch
            .points()
            .iter()
            .find(|point| {
                let at = point.at.in_plane();
                (at[0] - want[0]).hypot(at[1] - want[1]) < 1.0e-9
            })
            .map(|point| point.id)
            .unwrap_or_else(|| panic!("no corner at {want:?}"))
    };
    let grabbed = corner_at(&drawn, [0.0, 0.0]);
    let opposite = corner_at(&drawn, [20.0, 10.0]);
    for cursor in [[-10.0, -6.0], [5.0, 3.0], [-4.0, 7.0_f64]] {
        let mut sketch = drawn.clone();
        assert!(sketch
            .move_point(
                grabbed,
                SketchPoint::from_continuous(cursor[0], cursor[1]),
                ctx(16),
            )
            .expect("the corner drag is answered"));
        let stayed = position(&sketch, opposite);
        assert!(
            (stayed[0] - 20.0).hypot(stayed[1] - 10.0) < 1.0e-6,
            "dragging a corner to {cursor:?} took the opposite corner to {stayed:?}"
        );
        // And the grabbed corner really went where it was asked, so the shape resized to reach it.
        let arrived = position(&sketch, grabbed);
        assert!(
            (arrived[0] - cursor[0]).hypot(arrived[1] - cursor[1]) < 1.0e-6,
            "the corner asked for {cursor:?} and landed at {arrived:?}"
        );
    }
}

/// **A run's length, its width and its height are three different claims**, so asserting two of
/// them is a drawing getting pinned down rather than an author repeating themselves.
///
/// All three answer the same subject pair, which is what makes this worth a test: the ordinary
/// already-asserted check compares the subject and would have refused the second.
#[test]
fn a_span_and_its_two_extents_are_three_separate_claims() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(8, 6));
    let across = ConstraintKind::Dimension(Dimension::SpanAlong {
        from: tail,
        to: head,
        axis: InPlaneAxis::Across,
        length: SketchLength::new(8),
    });
    let up = ConstraintKind::Dimension(Dimension::SpanAlong {
        from: tail,
        to: head,
        axis: InPlaneAxis::Up,
        length: SketchLength::new(6),
    });
    sketch
        .add_constraint(across, ctx(16))
        .expect("the drawing already stands eight across");
    sketch
        .add_constraint(up, ctx(16))
        .expect("its height is a different question from its width");
    assert!(
        matches!(
            sketch.add_constraint(across, ctx(16)),
            Err(ConstraintRefusal::AlreadyAsserted { .. })
        ),
        "the SAME extent twice is still one claim"
    );

    // The length is a third claim, and with both extents already stated it is redundant rather
    // than refused — which is the flag the family has for exactly this.
    let diagonal = sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: tail,
                to: head,
                length: SketchLength::new(10),
            }),
            ctx(16),
        )
        .expect("a diagonal is not either extent");
    assert_eq!(sketch.constraints().len(), 3);
    assert!(sketch
        .constraints()
        .iter()
        .any(|held| held.id == diagonal && held.redundant));
}

/// **An extent moves the drawing along one axis and leaves the other alone.** The row is what the
/// member is for, so the test is a solve rather than a shape check.
#[test]
fn an_extent_states_one_axis_and_says_nothing_about_the_other() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(8, 6));
    sketch
        .add_constraint(
            ConstraintKind::Fix {
                point: tail,
                at: SketchPoint::new(0, 0),
            },
            ctx(16),
        )
        .expect("somewhere to measure from");
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::SpanAlong {
                from: tail,
                to: head,
                axis: InPlaneAxis::Across,
                length: SketchLength::new(20),
            }),
            ctx(16),
        )
        .expect("a width the drawing does not yet have");
    sketch.solve(ctx(16)).expect("one row and two free points");

    let arrived = position(&sketch, head);
    assert!(
        (arrived[0].abs() - 20.0).abs() < 1.0e-6,
        "the width was asserted and the drawing took it: {arrived:?}"
    );
    assert!(
        (arrived[1] - 6.0).abs() < 1.0e-6,
        "the height was not asserted and nothing pulled on it: {arrived:?}"
    );
}

/// An extent is an authored length like any other, so a density re-target rescales it — otherwise
/// a width of one block would silently become a width of half a block.
#[test]
fn an_extent_is_re_targeted_like_every_other_authored_length() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(8, 6));
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::SpanAlong {
                from: tail,
                to: head,
                axis: InPlaneAxis::Across,
                length: SketchLength::new(8),
            }),
            ctx(16),
        )
        .expect("the width the drawing has");
    sketch.retarget_density(16, 32);
    let ConstraintKind::Dimension(Dimension::SpanAlong { length, .. }) =
        sketch.constraints()[0].kind
    else {
        panic!("the extent is still an extent");
    };
    assert!(
        (length.value() - 16.0).abs() < 1.0e-6,
        "half a block stayed half a block: {}",
        length.value()
    );
}
/// **A gap holds a point off a line and lets it slide along that line freely.** That is the whole
/// difference between this member and a span: a span names two PLACES, so walking one of them
/// changes the answer, and a gap names a DIRECTION, so it does not.
#[test]
fn a_gap_holds_a_point_off_a_line_and_says_nothing_about_where_along_it() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let line = sketch
        .connect(tail, head)
        .expect("a line to measure across");
    let stood = sketch.add_free_point(SketchPoint::new(4, 3));
    for (point, at) in [
        (tail, SketchPoint::new(0, 0)),
        (head, SketchPoint::new(10, 0)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("the line stays where it is so the gap is the only thing that can move");
    }
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Gap {
                point: stood,
                segment: line,
                length: SketchLength::new(7),
            }),
            ctx(16),
        )
        .expect("a point three off a line can be asked to stand seven off it");
    sketch.solve(ctx(16)).expect("one row and one free point");

    let arrived = position(&sketch, stood);
    assert!(
        (arrived[1].abs() - 7.0).abs() < 1.0e-6,
        "the point took the gap it was asked for: {arrived:?}"
    );
    assert!(
        (arrived[0] - 4.0).abs() < 1.0e-6,
        "nothing pulled it along the line it is measured against: {arrived:?}"
    );
}

/// **A gap is measured against the whole line, not the run that is drawn.** Two rails cut to
/// different lengths and set past each other still state one offset, which is the ordinary case
/// this member exists for.
#[test]
fn a_gap_is_measured_against_the_line_and_not_the_run_that_is_drawn() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(4, 0));
    let line = sketch.connect(tail, head).expect("a short rail");
    // Stood well past the drawn end, where no perpendicular foot lands on the run at all.
    let stood = sketch.add_free_point(SketchPoint::new(30, 2));
    for (point, at) in [
        (tail, SketchPoint::new(0, 0)),
        (head, SketchPoint::new(4, 0)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at }, ctx(16))
            .expect("the rail stays put");
    }
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Gap {
                point: stood,
                segment: line,
                length: SketchLength::new(9),
            }),
            ctx(16),
        )
        .expect("the line runs on past its drawn end");
    sketch.solve(ctx(16)).expect("one row and one free point");

    let arrived = position(&sketch, stood);
    assert!(
        (arrived[1].abs() - 9.0).abs() < 1.0e-6,
        "measured across the line the run only samples: {arrived:?}"
    );
}

/// **A gap and the same pair's other claims are separate questions.** The subject is a point and a
/// segment, which no other member of the family can name, so the ordinary already-asserted rule
/// separates it without the distance family's special case having to.
#[test]
fn a_gap_is_one_claim_and_the_same_pair_twice_is_still_one() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 0));
    let line = sketch.connect(tail, head).expect("a line");
    let stood = sketch.add_free_point(SketchPoint::new(4, 3));
    let gap = |length| {
        ConstraintKind::Dimension(Dimension::Gap {
            point: stood,
            segment: line,
            length: SketchLength::new(length),
        })
    };
    sketch.add_constraint(gap(3), ctx(16)).expect("an offset");
    assert!(
        matches!(
            sketch.add_constraint(gap(5), ctx(16)),
            Err(ConstraintRefusal::AlreadyAsserted { .. })
        ),
        "a different NUMBER is the same question asked twice"
    );
    // The span to one of the line's own ends is a different question, and takes.
    sketch
        .add_constraint(
            ConstraintKind::Dimension(Dimension::Span {
                from: stood,
                to: tail,
                length: SketchLength::new(5),
            }),
            ctx(16),
        )
        .expect("how far the point stands from an END is not how far it stands off the LINE");

    // And it re-targets like every other authored length.
    sketch.retarget_density(16, 32);
    let ConstraintKind::Dimension(Dimension::Gap { length, .. }) = sketch.constraints()[0].kind
    else {
        panic!("the gap is still a gap");
    };
    assert!((length.value() - 6.0).abs() < 1.0e-6, "{}", length.value());
}

/// A gap needs a line with a direction to be measured across, and a distance that is really one.
#[test]
fn a_gap_refuses_a_line_of_no_length_and_a_length_of_none() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(0, 0));
    let nowhere = sketch.connect(tail, head).expect("a line drawn on itself");
    let stood = sketch.add_free_point(SketchPoint::new(4, 3));
    assert!(matches!(
        sketch.add_constraint(
            ConstraintKind::Dimension(Dimension::Gap {
                point: stood,
                segment: nowhere,
                length: SketchLength::new(3),
            }),
            ctx(16),
        ),
        Err(ConstraintRefusal::Impossible)
    ));

    let far = sketch.add_free_point(SketchPoint::new(10, 0));
    let line = sketch
        .connect(tail, far)
        .expect("a line that goes somewhere");
    assert!(
        matches!(
            sketch.add_constraint(
                ConstraintKind::Dimension(Dimension::Gap {
                    point: stood,
                    segment: line,
                    length: SketchLength::new(0),
                }),
                ctx(16),
            ),
            Err(ConstraintRefusal::Impossible)
        ),
        "a point standing ON a line is `PointOnCurve`, which asserts a place and not a distance"
    );
}

/// **Every member of the dimension family survives being written out and read back.**
///
/// A dimension is the one relation whose whole point is that the AUTHOR stated it, so a member
/// that failed to reload would silently hand back a claim nobody made. The family grew a member at
/// a time and each arrived with its own solver row and its own refusals; what none of them was
/// checked for is the part that only a saved document exercises — a renamed field, a variant that
/// serializes as something an older reader already uses, or a `#[serde(default)]` quietly filling
/// in a corner the author actually chose.
#[test]
fn every_dimension_member_round_trips_through_serde() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let (rail_from, rail_to, rail) = add_test_segment(&mut sketch, [0, 0], [10, 0]);
    // A span along an axis gets its OWN slanted pair. On the horizontal rail the rise is zero, and
    // a rise of zero has no direction to grow in — the row is degenerate, not merely unsatisfied.
    let (stile_from, stile_to, _) = add_test_segment(&mut sketch, [40, 0], [46, 8]);
    let (_, _, arm) = add_test_segment(&mut sketch, [0, 20], [8, 26]);
    let stood = sketch.add_free_point(SketchPoint::new(3, 7));
    let arc_from = sketch.add_free_point(SketchPoint::new(30, 0));
    let arc_to = sketch.add_free_point(SketchPoint::new(20, 10));
    let arc = sketch
        .connect_arc(arc_from, arc_to, AngleMeasurement::from_degrees(90))
        .expect("a quarter arc");
    let hub = sketch.add_free_point(SketchPoint::new(-30, 0));
    let rims = [6, 10].map(|radius| {
        SketchCurve::Circle(
            sketch
                .circle_about(hub, SketchLength::new(radius))
                .expect("a rim about a free point"),
        )
    });

    // One of each, including the two that carry a choice a reader could not otherwise recover:
    // which END of an arc an angle arm reads, and which of the four CORNERS it was struck in.
    let stated = [
        Dimension::Span {
            from: rail_from,
            to: rail_to,
            length: SketchLength::new(10),
        },
        Dimension::SpanAlong {
            from: stile_from,
            to: stile_to,
            axis: InPlaneAxis::Up,
            length: SketchLength::new(8),
        },
        Dimension::Gap {
            point: stood,
            segment: rail,
            length: SketchLength::new(7),
        },
        Dimension::RimGap {
            first: rims[0],
            second: rims[1],
            length: SketchLength::new(4),
        },
        Dimension::Angle {
            first: AngleArm::Segment { segment: arm },
            second: AngleArm::ArcEnd {
                arc,
                end: ArcEnd::To,
            },
            degrees: AngleMeasurement::from_degrees(35),
            corner: AngleCorner::Supplementary,
        },
        Dimension::Radius {
            curve: SketchCurve::Arc(arc),
            length: SketchLength::new(10),
        },
        Dimension::Diameter {
            curve: rims[0],
            length: SketchLength::new(12),
        },
    ];
    for dimension in stated {
        sketch
            .add_constraint(ConstraintKind::Dimension(dimension), ctx(16))
            .unwrap_or_else(|fault| panic!("{dimension:?} was refused: {fault:?}"));
    }

    let json = serde_json::to_string(&sketch).expect("serialize");
    let loaded: Sketch = serde_json::from_str(&json).expect("deserialize");
    let reloaded: Vec<ConstraintKind> = loaded
        .constraints()
        .iter()
        .map(|constraint| constraint.kind)
        .collect();
    for dimension in stated {
        assert!(
            reloaded.contains(&ConstraintKind::Dimension(dimension)),
            "{dimension:?} did not come back: {reloaded:?}"
        );
    }
    assert_eq!(reloaded.len(), stated.len(), "and nothing else came back");

    // A re-target rescales what it should and leaves the angle alone, on the RELOADED document —
    // an authored quantity that lost its density on the way through would only show up here.
    let mut retargeted = loaded;
    retargeted.retarget_density(16, 32);
    for kind in retargeted.constraints().iter().map(|held| held.kind) {
        let ConstraintKind::Dimension(dimension) = kind else {
            panic!("only dimensions were asserted")
        };
        match dimension {
            Dimension::Angle { degrees, .. } => {
                assert!(
                    (degrees.to_degrees_f64() - 35.0).abs() < 1e-9,
                    "{degrees:?}"
                );
            }
            Dimension::Span { length, .. }
            | Dimension::SpanAlong { length, .. }
            | Dimension::Gap { length, .. }
            | Dimension::RimGap { length, .. }
            | Dimension::Radius { length, .. }
            | Dimension::Diameter { length, .. } => {
                assert!(length.value() > 0.0, "a length went missing: {dimension:?}");
            }
        }
    }
}

/// **A drawing saved before the two kinds became one still opens.**
///
/// `Coincident` and `PointOnCurve` were separate `ConstraintKind`s, so an older document spells a
/// point-to-point coincidence `{first, second}` and a point-on-a-curve under its own tag. Both are
/// one kind now, and the reader has to accept all three spellings — the two it will never write
/// again included — or a saved sketch loses assertions on load and the author is never told.
#[test]
fn a_coincidence_saved_before_the_merge_still_loads() {
    let stored = r#"{
        "id": 7,
        "kind": { "Coincident": { "first": 3, "second": 4 } }
    }"#;
    let loaded: Constraint = serde_json::from_str(stored).expect("the pre-merge point pair");
    assert_eq!(
        loaded.kind,
        ConstraintKind::Coincident {
            point: 3,
            onto: CoincidentTarget::Point(4),
        }
    );

    let stored = r#"{
        "id": 8,
        "kind": { "PointOnCurve": { "point": 5, "curve": { "Segment": 6 } } }
    }"#;
    let loaded: Constraint = serde_json::from_str(stored).expect("the pre-merge curve target");
    assert_eq!(
        loaded.kind,
        ConstraintKind::Coincident {
            point: 5,
            onto: CoincidentTarget::Curve(SketchCurve::Segment(6)),
        }
    );

    // And what it writes today reads back as itself, so the migration cannot be the only path in.
    let current = Constraint {
        id: 9,
        kind: ConstraintKind::Coincident {
            point: 1,
            onto: CoincidentTarget::Curve(SketchCurve::Circle(2)),
        },
        redundant: false,
        anchor: None,
    };
    let written = serde_json::to_string(&current).expect("serialize");
    assert!(
        !written.contains("PointOnCurve"),
        "the old tag is read, never written: {written}"
    );
    let reloaded: Constraint = serde_json::from_str(&written).expect("round trip");
    assert_eq!(reloaded.kind, current.kind);
}

/// **A coincidence between two points reads the same both ways round, so it is STORED one way.**
///
/// Which point the author clicked first is not part of the claim. Left unordered, the same
/// assertion is two different values — `!=` under `PartialEq`, different bytes on disk — while
/// `subject` already calls them one, so a drawing would carry a distinction nothing in it
/// means. A curve target is not ordered: there the two operands play different parts.
#[test]
fn a_point_pair_is_stored_in_one_order_and_a_curve_target_is_not() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let earlier = sketch.add_free_point(SketchPoint::new(0, 0));
    let later = sketch.add_free_point(SketchPoint::new(8, 3));

    let clicked_backwards = ConstraintKind::Coincident {
        point: later,
        onto: CoincidentTarget::Point(earlier),
    };
    let clicked_forwards = ConstraintKind::Coincident {
        point: earlier,
        onto: CoincidentTarget::Point(later),
    };
    assert_eq!(
        clicked_backwards.normalized(),
        clicked_forwards,
        "the later id names the target whichever way it was picked"
    );
    assert_eq!(clicked_forwards.normalized(), clicked_forwards);

    // What the store keeps agrees with that, so a saved drawing cannot record the click order.
    let held = sketch
        .add_constraint(clicked_backwards, ctx(16))
        .expect("two free points can always meet");
    assert_eq!(
        sketch
            .constraints()
            .iter()
            .find(|constraint| constraint.id == held)
            .map(|constraint| constraint.kind),
        Some(clicked_forwards)
    );

    // A curve target keeps the order it was given: the point and the curve are not interchangeable.
    let (from, to, segment) = add_test_segment(&mut sketch, [0, 20], [10, 20]);
    let on_a_curve = ConstraintKind::Coincident {
        point: earlier,
        onto: CoincidentTarget::Curve(SketchCurve::Segment(segment)),
    };
    assert_eq!(on_a_curve.normalized(), on_a_curve);
    assert!(
        earlier < from && earlier < to,
        "the point outranks the curve's own ends, so an id sort would have moved it"
    );
}
