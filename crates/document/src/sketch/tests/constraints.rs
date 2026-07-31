//! Constraint entities and the continuous solve (ADR 0035 Decisions 2, 3 and 4).

use super::*;

/// A segment from `(0,0)` to `(10,4)` — slanted, so `Horizontal` has something to do.
fn slanted() -> (Sketch, EntityId, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(10, 4));
    let segment = sketch.connect(tail, head).expect("a fresh segment");
    (sketch, tail, head, segment)
}

fn position(sketch: &Sketch, id: EntityId) -> [f64; 2] {
    sketch
        .points()
        .iter()
        .find(|point| point.id == id)
        .expect("the point")
        .at
        .in_plane()
}

/// With nothing asserted, every coordinate is free. This is the baseline "fully constrained" is
/// measured against, and it is read off the store rather than from a solve with no residuals.
#[test]
fn an_unconstrained_sketch_is_all_freedom() {
    let (sketch, _, _, _) = slanted();
    assert_eq!(sketch.degrees_of_freedom(), 4, "two points, two axes each");
    assert!(sketch.constraints().is_empty());
}

/// `Fix` pins both of a point's coordinates, so it removes exactly two freedoms and moves nothing.
#[test]
fn a_fix_pins_two_freedoms_and_moves_nothing() {
    let (mut sketch, tail, head, _) = slanted();
    let before = position(&sketch, tail);
    sketch
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("nothing else is asserted, so it cannot conflict");
    assert_eq!(sketch.degrees_of_freedom(), 2, "the head is still free");
    assert_eq!(
        position(&sketch, tail),
        before,
        "a fix does not move a point"
    );
    assert_eq!(position(&sketch, head), [10.0, 4.0], "nor its neighbour");
}

/// `Horizontal` levels a segment, and the least-squares solve moves the drawing **as little as it
/// can**: neither end is privileged, so they meet in the middle rather than one snapping to the
/// other. That is the property that makes a solve feel like a nudge.
#[test]
fn horizontal_levels_a_segment_by_meeting_in_the_middle() {
    let (mut sketch, tail, head, segment) = slanted();
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("a lone constraint always holds");

    let (a, b) = (position(&sketch, tail), position(&sketch, head));
    assert!((a[1] - b[1]).abs() < 1e-6, "level: {a:?} to {b:?}");
    assert!((a[1] - 2.0).abs() < 1e-6, "the tail rose halfway: {a:?}");
    assert!((b[1] - 2.0).abs() < 1e-6, "the head fell halfway: {b:?}");
    assert_eq!(a[0], 0.0, "nothing pulled sideways");
    assert_eq!(b[0], 10.0);
    assert_eq!(sketch.degrees_of_freedom(), 3, "one assertion, one freedom");
}

/// **Geometry the constraint does not name must not be able to break it.** A lone segment gets
/// levelled the same whether it is alone on the plane or surrounded by a drawing.
///
/// This is the regression for the bug that made the constraint tools read as simply not working
/// (owner 2026-07-30). The verdict was taken from the solver's `SolveOutcome` rather than from its
/// residuals; that flag's residual test is absolute while its step test is relative to the size of
/// the whole parameter vector, so free points elsewhere in the sketch — contributing nothing to
/// the residual and everything to the vector's length — made the step test fire first. It reported
/// `Stalled` with the constraint satisfied to about 1e-10 voxels, and `Stalled` was refused as
/// unsatisfiable. **Two** unrelated points were enough, which is to say every real drawing.
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
                .add_constraint(ConstraintKind::Horizontal { segment })
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
        .add_constraint(ConstraintKind::Distance {
            from: tail,
            to: head,
            length: SketchLength::new(6),
        })
        .expect("two free points can always be six apart");

    let (a, b) = (position(&sketch, tail), position(&sketch, head));
    let span = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    assert!((span - 6.0).abs() < 1e-6, "six apart, got {span}");
    assert!((a[0] - 2.0).abs() < 1e-6, "each end came in by two: {a:?}");
    assert!((b[0] - 8.0).abs() < 1e-6, "{b:?}");
}

/// Decision 4's first half: **unsatisfiable is refused, and refusing changes nothing.** The trial
/// runs on a copy, so the drawing is where it was rather than where a failed solve pushed it.
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
                .add_constraint(ConstraintKind::Fix { point, at })
                .expect("pinning each end in turn is consistent"),
        );
    }
    assert_eq!(sketch.degrees_of_freedom(), 0, "fully constrained");
    let before: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();

    // The ends are pinned about 10.77 apart. Five is not a distance they can be.
    let refusal = sketch
        .add_constraint(ConstraintKind::Distance {
            from: tail,
            to: head,
            length: SketchLength::new(5),
        })
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

/// Decision 4's second half: **redundant is accepted and flagged.** An assertion the geometry
/// already implies is insurance against a later edit, so it is marked rather than refused.
///
/// Redundant is not the same as DUPLICATE, and the difference is exactly what this fixture shows:
/// two pinned endpoints already put the segment level, so `Horizontal` adds no information — but
/// it is a different claim, made about different entities, and it survives a later edit that
/// releases a pin. A literal second `Horizontal` would be refused instead.
#[test]
fn a_redundant_constraint_is_kept_and_flagged() {
    let (mut sketch, tail, head, segment) = slanted();
    let pinned_tail = sketch
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("the first pin");
    sketch
        .add_constraint(ConstraintKind::Fix {
            point: head,
            at: SketchPoint::new(10, 0),
        })
        .expect("the second pin, level with the first");
    let implied = sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
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
        sketch.degrees_of_freedom(),
        0,
        "and it took no freedom away, which is what redundant MEANS — the pins took them all"
    );
}

/// The delete cascade reaches constraints (Decision 3): a constraint never outlives the geometry
/// it names, so a residual row can never refer to a shape that is gone.
#[test]
fn deleting_geometry_takes_its_constraints_with_it() {
    let (mut sketch, tail, _, segment) = slanted();
    sketch
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("a lone fix");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("and a level");
    assert_eq!(sketch.constraints().len(), 2);

    sketch.delete_segment(segment);
    assert_eq!(
        sketch.constraints().len(),
        1,
        "the level went with the line"
    );
    sketch.delete_point_cascade(tail);
    assert!(
        sketch.constraints().is_empty(),
        "the fix went with the point"
    );
}

/// Load repair erases a constraint naming geometry the store does not hold, and counts it — the
/// same policy every other entity gets (ADR 0030: erase the invalid, never fail the load).
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
    });
    sketch.constraints_mut_for_test().push(Constraint {
        id: 902,
        kind: ConstraintKind::Horizontal { segment: 903 },
        redundant: false,
    });
    assert_eq!(sketch.repair(), 2, "both name entities that are not there");
    assert!(sketch.constraints().is_empty());
}

/// A constraint naming absent geometry cannot be added in the first place — the store is checked
/// before the solver is, because an unknown id is a caller error and not a geometric one.
#[test]
fn a_constraint_naming_absent_geometry_is_refused() {
    let (mut sketch, tail, head, _) = slanted();
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Fix {
            point: 900,
            at: SketchPoint::new(0, 0),
        }),
        Err(ConstraintRefusal::UnknownEntity)
    );
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Horizontal { segment: 900 }),
        Err(ConstraintRefusal::UnknownEntity)
    );
    // A negative length is no drawing's distance, so it never reaches the solver.
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Distance {
            from: tail,
            to: head,
            length: SketchLength::new(-3),
        }),
        Err(ConstraintRefusal::Impossible)
    );
    assert!(sketch.constraints().is_empty());
}

/// A refused constraint burns no id: the next entity gets the number the refusal did not take.
#[test]
fn a_refusal_does_not_consume_an_id() {
    let (mut sketch, tail, _, _) = slanted();
    let _ = sketch.add_constraint(ConstraintKind::Horizontal { segment: 900 });
    let next = sketch
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("a lone fix");
    assert_eq!(next, 3, "after two points and a segment, the next id is 3");
}

/// Solving again from a solution changes nothing — the solve is idempotent, which is what lets it
/// run live during a drag (Decision 11) without the drawing creeping.
#[test]
fn solving_a_solved_sketch_moves_nothing() {
    let (mut sketch, _, _, segment) = slanted();
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("a lone constraint");
    let settled: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();

    let report = sketch.solve().expect("there is a constraint to solve");
    assert_eq!(report.outcome, SolveOutcome::Converged);
    let again: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();
    assert_eq!(settled, again);
}

/// The PRODUCER door the rail's constraint verbs go through (ADR 0035): pure, so the caller
/// holds both drawings and the shell's one-transaction commit has something to commit.
#[test]
fn the_producer_door_asserts_without_touching_the_original() {
    let (sketch, tail, head, segment) = slanted();
    let before = SketchSolid::extrude(sketch, 3);
    let (after, id) = before
        .with_constraint(ConstraintKind::Horizontal { segment })
        .expect("nothing else is asserted");

    assert_eq!(position(&before.sketch, head), [10.0, 4.0], "the original");
    assert_eq!(
        position(&after.sketch, tail)[1],
        position(&after.sketch, head)[1],
        "the copy is levelled"
    );
    assert_eq!(after.sketch.constraints().len(), 1);
    assert_eq!(after.sketch.degrees_of_freedom(), 3);

    // Releasing it stops the assertion without undoing what it did — the geometry stays level.
    let released = after.with_constraint_deleted(id);
    assert!(released.sketch.constraints().is_empty());
    assert_eq!(
        position(&released.sketch, tail)[1],
        position(&after.sketch, tail)[1],
        "releasing an assertion is not an undo"
    );
}

/// A refusal at the producer door hands back nothing, so the shell cannot commit half an edit.
#[test]
fn the_producer_door_refuses_without_a_partial_result() {
    let (mut sketch, tail, _, _) = slanted();
    sketch
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("the first assertion cannot conflict");
    let solid = SketchSolid::extrude(sketch, 3);
    assert_eq!(
        solid
            .with_constraint(ConstraintKind::Horizontal { segment: 900 })
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
        .with_constraint(ConstraintKind::Horizontal { segment })
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

/// One constraint of a kind per entity set (Decision 4). The second `Horizontal` on a segment
/// already asserted horizontal says nothing the first did not, so it is refused rather than kept
/// and flagged — and the store is left holding exactly one.
#[test]
fn the_same_assertion_twice_on_one_segment_is_refused() {
    let (mut sketch, _, _, segment) = slanted();
    let first = sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("the first assertion");
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Horizontal { segment }),
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
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("the first assertion");
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(7, 7),
        }),
        Err(ConstraintRefusal::AlreadyAsserted { existing: first }),
        "a different place is still the same claim about the same point"
    );
}

/// A distance names an unordered PAIR, so asserting it the other way round is the same assertion.
#[test]
fn a_distance_is_the_same_assertion_in_either_direction() {
    let (mut sketch, tail, head, _) = slanted();
    let apart = |value: f64| ConstraintKind::Distance {
        from: tail,
        to: head,
        length: SketchLength::from_continuous(value),
    };
    let first = sketch.add_constraint(apart(9.0)).expect("the first");
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Distance {
            from: head,
            to: tail,
            length: SketchLength::from_continuous(4.0),
        }),
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
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("levelling a slanted segment is fine");
    let levelled = (position(&sketch, head)[0] - position(&sketch, tail)[0]).abs();
    assert!(levelled > 1.0, "still a line, {levelled} across");

    // Its own refusal, not Unsatisfiable: nothing here fights, the assertions AGREE on an answer
    // that happens to be a singularity. It names the geometry that would vanish and the assertion
    // whose release would save it.
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Vertical { segment }),
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
        .add_constraint(ConstraintKind::Horizontal { segment: real })
        .expect("the collapsed stub is not this assertion's doing");
}

/// The witness rank reads the drawing it is HANDED, never a solution it went and computed.
///
/// That is the whole of the fix for a defect the literature names (FreeCAD #5931): rows of the
/// Jacobian can vanish at an exactly-solved configuration, so redundancy read there mistakes a
/// solver's success for a constraint saying nothing. Read at the author's own slanted drawing —
/// a generic configuration — a `Fix` pins two coordinates and a `Horizontal` adds a third
/// independent row.
#[test]
fn the_witness_rank_is_read_at_the_drawing_it_is_given() {
    let (sketch, tail, _, segment) = slanted();
    let ends = sketch.segment_ends();
    let held = |kind| Constraint {
        id: 99,
        kind,
        redundant: false,
    };
    let pin = held(ConstraintKind::Fix {
        point: tail,
        at: SketchPoint::new(0, 0),
    });
    let level = held(ConstraintKind::Horizontal { segment });

    let rank_of =
        |constraints: &[Constraint]| constraint::witness_rank(sketch.points(), &ends, constraints);
    assert_eq!(rank_of(&[]), 0, "no assertions pin nothing");
    assert_eq!(
        rank_of(&[pin]),
        2,
        "a Fix pins both of a point's coordinates"
    );
    assert_eq!(rank_of(&[pin, level]), 3, "and levelling adds a third");
    assert_eq!(rank_of(&[level]), 1);
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
                .add_constraint(ConstraintKind::Fix { point, at })
                .expect("pinning each end in turn is consistent");
        }
        let implied = sketch
            .add_constraint(ConstraintKind::Horizontal { segment })
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
