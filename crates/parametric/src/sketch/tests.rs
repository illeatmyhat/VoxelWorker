//! Cross-module invariants for the public sketch façade.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;

#[test]
/// Builder owner tags guard accidental cross-problem handles; they are process-local validation
/// metadata and must not affect an equivalent problem’s numerical result or diagnostics.
fn owner_tags_do_not_affect_equivalent_solutions_or_diagnostics() {
    let build = || {
        let mut builder = ProblemBuilder::new();
        let first = builder.add_point([0.0, 0.0]);
        let second = builder.add_point([10.0, 4.0]);
        let segment = builder.add_segment(first, second);
        builder.add_constraint(Relation::Horizontal { segment });
        (builder.finish().unwrap(), first, second)
    };
    let (one, one_first, one_second) = build();
    let (two, two_first, two_second) = build();
    let one = one.settle();
    let two = two.settle();
    assert_eq!(
        one.solution.position(one_first),
        two.solution.position(two_first)
    );
    assert_eq!(
        one.solution.position(one_second),
        two.solution.position(two_second)
    );
    assert_eq!(one.diagnostics.report, two.diagnostics.report);
    assert_eq!(one.diagnostics.satisfied, two.diagnostics.satisfied);
}

/// **The undrawn reach runs from the near end out to the point, and only when there is one.**
///
/// It is the whole of what makes a coincidence legible once the solve carries its point off the
/// end, so the two things worth pinning are that an ordinary point produces nothing at all and
/// that a stranded one produces a reach touching BOTH the curve it left and the point it explains.
#[test]
fn an_undrawn_reach_joins_the_near_end_to_a_point_that_left_the_extent() {
    let rail = CurveGeometry::Segment {
        from: [0.0, 0.0],
        to: [10.0, 0.0],
    };
    assert_eq!(
        undrawn_reach_to(rail, [4.0, 0.0], 1.0e-9),
        None,
        "a point on the drawn piece has nothing to explain"
    );
    assert_eq!(
        undrawn_reach_to(rail, [17.0, 0.0], 1.0e-9),
        Some(UndrawnReach::Span {
            from: [10.0, 0.0],
            to: [17.0, 0.0]
        }),
        "past the far end, the reach starts at the far end"
    );
    assert_eq!(
        undrawn_reach_to(rail, [-3.0, 0.0], 1.0e-9),
        Some(UndrawnReach::Span {
            from: [0.0, 0.0],
            to: [-3.0, 0.0]
        }),
        "and short of the near end, at the near end"
    );

    // A quarter arc from due east to due north, so three quarters of its circle went undrawn.
    let quarter = CurveGeometry::Circular(CircularCurve {
        center: [0.0, 0.0],
        radius: 2.0,
        arc: Some(ArcDomain {
            from: [2.0, 0.0],
            to: [0.0, 2.0],
            sweep_radians: std::f64::consts::FRAC_PI_2,
        }),
    });
    assert_eq!(
        undrawn_reach_to(
            quarter,
            [std::f64::consts::SQRT_2, std::f64::consts::SQRT_2],
            1.0e-9
        ),
        None,
        "halfway round the drawn quarter"
    );

    // Due west: a quarter turn on from the drawn END (which stands at north), against three
    // quarters back the other way from its start. The reach takes the quarter.
    let Some(UndrawnReach::Sweep {
        sweep_radians,
        from_radians,
        radius,
        center,
    }) = undrawn_reach_to(quarter, [-2.0, 0.0], 1.0e-9)
    else {
        panic!("due west is off a quarter arc")
    };
    assert_eq!((center, radius), ([0.0, 0.0], 2.0));
    assert!(
        (from_radians - std::f64::consts::FRAC_PI_2).abs() < 1.0e-9,
        "it leaves from the drawn end, at north: {from_radians}"
    );
    assert!(
        (sweep_radians - std::f64::consts::FRAC_PI_2).abs() < 1.0e-9,
        "a quarter turn, not the three quarters the other way: {sweep_radians}"
    );

    // Just past the drawn end: a short reach forward, never the long way round the circle.
    let just_past = [
        2.0 * (std::f64::consts::FRAC_PI_2 + 0.2).cos(),
        2.0 * (std::f64::consts::FRAC_PI_2 + 0.2).sin(),
    ];
    let Some(UndrawnReach::Sweep { sweep_radians, .. }) =
        undrawn_reach_to(quarter, just_past, 1.0e-9)
    else {
        panic!("past the end is off the arc")
    };
    assert!(
        (sweep_radians - 0.2).abs() < 1.0e-9,
        "a fifth of a radian on from the end it left: {sweep_radians}"
    );

    // A whole circle has no ends to be beyond.
    assert_eq!(
        undrawn_reach_to(
            CurveGeometry::Circular(CircularCurve {
                center: [0.0, 0.0],
                radius: 2.0,
                arc: None,
            }),
            [-2.0, 0.0],
            1.0e-9
        ),
        None
    );
}
