//! Cross-module invariants for the public sketch façade.

#![allow(clippy::unwrap_used)]

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
