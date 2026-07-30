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
    for (point, at) in [
        (tail, SketchPoint::new(0, 0)),
        (head, SketchPoint::new(10, 4)),
    ] {
        sketch
            .add_constraint(ConstraintKind::Fix { point, at })
            .expect("pinning each end in turn is consistent");
    }
    assert_eq!(sketch.degrees_of_freedom(), 0, "fully constrained");
    let before: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();

    // The ends are pinned about 10.77 apart. Five is not a distance they can be.
    assert_eq!(
        sketch.add_constraint(ConstraintKind::Distance {
            from: tail,
            to: head,
            length: SketchLength::new(5),
        }),
        Err(ConstraintRefusal::Unsatisfiable)
    );
    assert_eq!(sketch.constraints().len(), 2, "it was not kept");
    let after: Vec<[f64; 2]> = sketch.points().iter().map(|p| p.at.in_plane()).collect();
    assert_eq!(before, after, "nor did the failed trial move anything");
}

/// Decision 4's second half: **redundant is accepted and flagged.** Saying the same thing twice is
/// sometimes the intent — an assertion the geometry already implies is insurance against a later
/// edit — so it is marked rather than refused.
#[test]
fn a_redundant_constraint_is_kept_and_flagged() {
    let (mut sketch, _, _, segment) = slanted();
    let first = sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("the first one says something");
    let second = sketch
        .add_constraint(ConstraintKind::Horizontal { segment })
        .expect("the second is redundant, not refused");

    let flagged = |id: EntityId| {
        sketch
            .constraints()
            .iter()
            .find(|constraint| constraint.id == id)
            .expect("the constraint")
            .redundant
    };
    assert!(!flagged(first), "the first raised the rank");
    assert!(flagged(second), "the second added no information");
    assert_eq!(
        sketch.degrees_of_freedom(),
        3,
        "and it took no freedom away, which is what redundant MEANS"
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
