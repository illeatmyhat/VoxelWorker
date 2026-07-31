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

/// A constraint holds through a DRAG, not merely at the moment it was asserted (ADR 0035
/// Decision 11). The grabbed end goes exactly where the hand put it and the free end follows to
/// keep the segment level — the drag is a pin, so the rest of the drawing moves around it.
///
/// The regression for the second half of the same complaint (owner 2026-07-30): the level was
/// applied, and then moving one of the line's own points tilted it straight back off. `move_point`
/// wrote the coordinate and never re-solved, so every assertion survived exactly until it was
/// tested.
#[test]
fn a_level_segment_stays_level_when_an_end_is_dragged() {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(40, 0));
    let segment = sketch.connect(tail, head).expect("a fresh segment");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("a lone level on a lone segment");

    assert!(sketch.move_point(tail, SketchPoint::new(-7, -18)));

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
        .add_constraint(ConstraintKind::Fix {
            point: tail,
            at: SketchPoint::new(0, 0),
        })
        .expect("nothing else is asserted");

    assert!(sketch.move_point(tail, SketchPoint::new(25, 25)));
    assert_eq!(position(&sketch, tail), [0.0, 0.0], "the fix wins");
    assert_eq!(
        position(&sketch, head),
        [10.0, 4.0],
        "and nothing else moved"
    );
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

// ---------------------------------------------------------------------------------------------
// The relations (ADR 0035 Decision 5): the constraints that name two pieces of geometry rather
// than one piece and an axis. Every one of them is checked by measuring the drawing afterwards,
// never by trusting the solver's own verdict — see `SATISFIED_RESIDUAL`.
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
        .add_constraint(ConstraintKind::Coincident { first, second })
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
        sketch.add_constraint(ConstraintKind::Coincident {
            first: point,
            second: point
        }),
        Err(ConstraintRefusal::Impossible)
    );
}

/// Parallel drives the sine of the angle between two segments to zero, and both segments keep
/// their extent getting there.
///
/// It does NOT preserve length exactly, and the first draft of this test asserted that it did.
/// The residual is an angle, so the solver is free to reach it any way it likes, and the way it
/// likes is the smallest move in the PARAMETERS — which are coordinates, not lengths. A pure
/// rotation would hold both lengths and is a larger coordinate move than the shear-ish answer the
/// solve actually finds. What the normalisation buys is conditioning, not rigidity: the residual
/// reads the same on a 3-voxel segment and a 300-voxel one, so neither dominates the step.
#[test]
fn parallel_aligns_two_segments_without_collapsing_them() {
    let (mut sketch, first, second) = two_segments();
    let before_first = span_length(&sketch, first);
    let before_second = span_length(&sketch, second);
    sketch
        .add_constraint(ConstraintKind::Parallel { first, second })
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
        .add_constraint(ConstraintKind::Perpendicular { first, second })
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
        .add_constraint(ConstraintKind::Equal { first, second })
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
        .add_constraint(ConstraintKind::Midpoint { point, segment })
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
        sketch.add_constraint(ConstraintKind::Midpoint {
            point: tail,
            segment
        }),
        Err(ConstraintRefusal::Impossible)
    );
}

/// Collinear says parallel AND no offset, which is why it spends two freedoms where Parallel
/// spends one.
#[test]
fn collinear_puts_two_segments_on_one_line() {
    let (mut sketch, first, second) = two_segments();
    let before = sketch.degrees_of_freedom();
    sketch
        .add_constraint(ConstraintKind::Collinear { first, second })
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
        sketch.degrees_of_freedom(),
        before - 2,
        "collinear spends two freedoms"
    );
}

/// **The stride property.** A kind that writes two residuals must be given two rows, or every
/// constraint after it in the list reads the wrong ones. This is the regression for the `_ => 1`
/// arm that `residual_count` carried: it would have handed a two-row kind one row and corrupted
/// the whole system rather than failing.
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
        .add_constraint(ConstraintKind::Coincident { first, second })
        .expect("two free points can meet");
    sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("an untouched segment can be levelled");

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
        .add_constraint(ConstraintKind::Perpendicular { first, second })
        .expect("two free segments can be squared");
    let grabbed = sketch
        .segments()
        .iter()
        .find(|seg| seg.id == first)
        .expect("the segment")
        .from;

    assert!(sketch.move_point(grabbed, SketchPoint::new(-13, 9)));

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

/// An arc, and the id of the centre point the sketch derives for it.
fn arc_with_centre() -> (Sketch, EntityId, EntityId) {
    let mut sketch = Sketch::empty(PlaneAxis::Z);
    let tail = sketch.add_free_point(SketchPoint::new(0, 0));
    let head = sketch.add_free_point(SketchPoint::new(20, 0));
    sketch
        .connect_arc(tail, head, AngleMeasurement::from_degrees(90))
        .expect("a quarter turn");
    let centre = sketch.arcs()[0].center;
    let loose = sketch.add_free_point(SketchPoint::new(40, 17));
    (sketch, centre, loose)
}

/// **A constraint cannot name a point the drawing derives.** An arc's centre is owned by
/// `sync_arc_centers`, so an assertion about it would be honoured by the solve and then erased by
/// the next edit — a badge on the drawing claiming something the drawing does not do.
///
/// This is the owner's report of 2026-07-31: a `Coincident` placed on what looked like a circle's
/// centre "didn't work". It was an arc's centre, and it took silently.
#[test]
fn an_arcs_centre_cannot_carry_a_constraint() {
    let (mut sketch, centre, loose) = arc_with_centre();
    assert!(sketch.is_derived_point(centre));
    assert!(
        !sketch.is_derived_point(loose),
        "a free point is the author's"
    );

    assert_eq!(
        sketch.add_constraint(ConstraintKind::Coincident {
            first: centre,
            second: loose,
        }),
        Err(ConstraintRefusal::Derived { point: centre }),
    );
    // The order of the naming is not what makes it derived.
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Coincident {
            first: loose,
            second: centre,
        }),
        Err(ConstraintRefusal::Derived { point: centre }),
    );
    assert!(sketch.constraints().is_empty(), "nothing was recorded");
}

/// The refusal is a property of the POINT, not of one verb — every kind with a point slot inherits
/// it, because the check runs over `ConstraintKind::points()` rather than per variant.
#[test]
fn every_kind_with_a_point_slot_refuses_a_derived_point() {
    let (mut sketch, centre, loose) = arc_with_centre();
    let far = sketch.add_free_point(SketchPoint::new(-30, 6));
    let segment = sketch.connect(loose, far).expect("a fresh segment");
    let at = sketch
        .points()
        .iter()
        .find(|point| point.id == centre)
        .expect("the centre")
        .at;

    for kind in [
        ConstraintKind::Fix { point: centre, at },
        ConstraintKind::Distance {
            from: centre,
            to: loose,
            length: SketchLength::new(12),
        },
        ConstraintKind::Midpoint {
            point: centre,
            segment,
        },
    ] {
        assert_eq!(
            sketch.add_constraint(kind),
            Err(ConstraintRefusal::Derived { point: centre }),
            "{kind:?} named a derived point",
        );
    }
}

/// The refusal is narrow: it is about the centre, not about arcs. A relation between two ordinary
/// points still lands on a sketch that happens to hold an arc.
#[test]
fn an_arc_in_the_drawing_does_not_refuse_ordinary_points() {
    let (mut sketch, _centre, loose) = arc_with_centre();
    let other = sketch.add_free_point(SketchPoint::new(41, 18));
    assert!(sketch
        .add_constraint(ConstraintKind::Coincident {
            first: loose,
            second: other,
        })
        .is_ok());
}
