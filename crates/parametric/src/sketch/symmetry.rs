//! Branch-stable symmetry mathematics for planar sketch curves.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::suboptimal_flops
)]

use super::curve::{
    CircularCurve, CurveGeometry, COLLAPSE_TOLERANCE as DEGENERATE_AXIS_SPAN,
    SATISFACTION_TOLERANCE as SATISFIED_RESIDUAL,
};
const INVALID_RESIDUAL: f64 = 1.0;

/// The durable correspondence between two curves mirrored across an authored axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SymmetryBranch {
    Direct,
    Reversed,
    Centers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryError {
    UnsupportedPair,
    InvalidBranch,
    NonFinite,
    DegenerateAxis,
    Unsatisfied,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetryWitness {
    pub at: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SymmetryResiduals {
    rows: [f64; 5],
    count: usize,
    valid_axis: bool,
}

impl SymmetryResiduals {
    pub(super) fn write_to(self, into: &mut [f64]) -> usize {
        into[..self.count].copy_from_slice(&self.rows[..self.count]);
        self.count
    }

    fn squared_norm(self) -> f64 {
        self.rows[..self.count].iter().map(|row| row * row).sum()
    }

    fn satisfied(self) -> bool {
        self.valid_axis
            && self.rows[..self.count].iter().all(|row| row.is_finite())
            && self.squared_norm().sqrt() <= SATISFIED_RESIDUAL
    }
}

fn finite_point(point: [f64; 2]) -> bool {
    point.into_iter().all(f64::is_finite)
}

fn finite_curve(curve: CurveGeometry) -> bool {
    match curve {
        CurveGeometry::Segment { from, to } => finite_point(from) && finite_point(to),
        CurveGeometry::Circular(CircularCurve {
            center,
            radius,
            arc,
        }) => {
            finite_point(center)
                && radius.is_finite()
                && arc.is_none_or(|arc| {
                    finite_point(arc.from) && finite_point(arc.to) && arc.sweep_radians.is_finite()
                })
        }
    }
}

fn axis_frame(axis: CurveGeometry) -> Result<([f64; 2], [f64; 2]), SymmetryError> {
    let CurveGeometry::Segment { from, to } = axis else {
        return Err(SymmetryError::InvalidBranch);
    };
    if !finite_point(from) || !finite_point(to) {
        return Err(SymmetryError::NonFinite);
    }
    let span = [to[0] - from[0], to[1] - from[1]];
    if !finite_point(span) {
        return Err(SymmetryError::NonFinite);
    }
    let length = span[0].hypot(span[1]);
    if !length.is_finite() {
        return Err(SymmetryError::NonFinite);
    }
    if length <= DEGENERATE_AXIS_SPAN {
        return Err(SymmetryError::DegenerateAxis);
    }
    let along = [span[0] / length, span[1] / length];
    if !finite_point(along) {
        return Err(SymmetryError::NonFinite);
    }
    Ok((from, along))
}

/// Whether a finite segment is a usable infinite symmetry axis.
#[must_use]
pub fn symmetry_axis_is_valid(axis: CurveGeometry) -> bool {
    axis_frame(axis).is_ok()
}

fn reflect_with_frame(point: [f64; 2], origin: [f64; 2], along: [f64; 2]) -> [f64; 2] {
    let delta = [point[0] - origin[0], point[1] - origin[1]];
    let projection = delta[0] * along[0] + delta[1] * along[1];
    [
        origin[0] + 2.0 * projection * along[0] - delta[0],
        origin[1] + 2.0 * projection * along[1] - delta[1],
    ]
}

fn endpoint_rows(
    first: ([f64; 2], [f64; 2]),
    second: ([f64; 2], [f64; 2]),
    reversed: bool,
    origin: [f64; 2],
    along: [f64; 2],
    rows: &mut [f64; 5],
) {
    let targets = if reversed {
        [second.1, second.0]
    } else {
        [second.0, second.1]
    };
    for (index, (point, target)) in [first.0, first.1].into_iter().zip(targets).enumerate() {
        let reflected = reflect_with_frame(point, origin, along);
        rows[index * 2] = reflected[0] - target[0];
        rows[index * 2 + 1] = reflected[1] - target[1];
    }
}

pub(super) fn residuals(
    first: CurveGeometry,
    second: CurveGeometry,
    axis: CurveGeometry,
    branch: SymmetryBranch,
) -> SymmetryResiduals {
    let count = match (first, second, branch) {
        (
            CurveGeometry::Segment { .. },
            CurveGeometry::Segment { .. },
            SymmetryBranch::Direct | SymmetryBranch::Reversed,
        ) => 4,
        (
            CurveGeometry::Circular(CircularCurve { arc: Some(_), .. }),
            CurveGeometry::Circular(CircularCurve { arc: Some(_), .. }),
            SymmetryBranch::Direct | SymmetryBranch::Reversed,
        ) => 5,
        _ => 3,
    };
    let mut result = SymmetryResiduals {
        rows: [INVALID_RESIDUAL; 5],
        count,
        valid_axis: false,
    };
    let Ok((origin, along)) = axis_frame(axis) else {
        return result;
    };
    result.valid_axis = true;
    match (first, second, branch) {
        (
            CurveGeometry::Segment { from: a0, to: a1 },
            CurveGeometry::Segment { from: b0, to: b1 },
            SymmetryBranch::Direct | SymmetryBranch::Reversed,
        ) => endpoint_rows(
            (a0, a1),
            (b0, b1),
            branch == SymmetryBranch::Reversed,
            origin,
            along,
            &mut result.rows,
        ),
        (
            CurveGeometry::Circular(CircularCurve {
                arc: Some(first), ..
            }),
            CurveGeometry::Circular(CircularCurve {
                arc: Some(second), ..
            }),
            SymmetryBranch::Direct | SymmetryBranch::Reversed,
        ) => {
            // A reflection reverses the sense of travel, and a stored arc has only one sense:
            // counter-clockwise from its tail to its head (ADR 0038). So an arc's mirror is
            // ALWAYS the reversed correspondence — the first arc's tail reflects onto the
            // second's head — and the branch, which is only ever a statement about which end
            // answers which, has nothing left to say for this pair. The two turns are equal
            // because a mirror preserves how FAR a curve turns; the direction it reverses is
            // already carried by the swapped ends.
            endpoint_rows(
                (first.from, first.to),
                (second.from, second.to),
                true,
                origin,
                along,
                &mut result.rows,
            );
            result.rows[4] = first.sweep_radians - second.sweep_radians;
        }
        (
            CurveGeometry::Circular(CircularCurve {
                center: first,
                radius: first_radius,
                arc: None,
            }),
            CurveGeometry::Circular(CircularCurve {
                center: second,
                radius: second_radius,
                arc: None,
            }),
            SymmetryBranch::Centers,
        ) => {
            let reflected = reflect_with_frame(first, origin, along);
            result.rows[0] = reflected[0] - second[0];
            result.rows[1] = reflected[1] - second[1];
            result.rows[2] = first_radius - second_radius;
        }
        _ => result.valid_axis = false,
    }
    result
}

/// Choose the stable endpoint correspondence whose current full residual is smallest.
///
/// # Errors
///
/// Returns an error when the axis is invalid or the two curves are not a supported like-kind pair.
pub fn choose_symmetry_branch(
    first: CurveGeometry,
    second: CurveGeometry,
    axis: CurveGeometry,
) -> Result<SymmetryBranch, SymmetryError> {
    axis_frame(axis)?;
    if !finite_curve(first) || !finite_curve(second) {
        return Err(SymmetryError::NonFinite);
    }
    match (first, second) {
        // An arc has one sense of travel and a mirror reverses it (ADR 0038), so there is no
        // correspondence to pick: the answer is the mirrored one, always. Comparing the two
        // branches here would be comparing a value against itself, since the arc arm of
        // `residuals` no longer reads the branch at all.
        (
            CurveGeometry::Circular(CircularCurve { arc: Some(_), .. }),
            CurveGeometry::Circular(CircularCurve { arc: Some(_), .. }),
        ) => {
            let mirrored = residuals(first, second, axis, SymmetryBranch::Reversed);
            if !mirrored.squared_norm().is_finite() {
                return Err(SymmetryError::NonFinite);
            }
            Ok(SymmetryBranch::Reversed)
        }
        (CurveGeometry::Segment { .. }, CurveGeometry::Segment { .. }) => {
            let direct = residuals(first, second, axis, SymmetryBranch::Direct);
            let reversed = residuals(first, second, axis, SymmetryBranch::Reversed);
            let direct_norm = direct.squared_norm();
            let reversed_norm = reversed.squared_norm();
            if !direct_norm.is_finite() || !reversed_norm.is_finite() {
                return Err(SymmetryError::NonFinite);
            }
            Ok(if reversed_norm < direct_norm {
                SymmetryBranch::Reversed
            } else {
                SymmetryBranch::Direct
            })
        }
        (
            CurveGeometry::Circular(CircularCurve { arc: None, .. }),
            CurveGeometry::Circular(CircularCurve { arc: None, .. }),
        ) => {
            let centers = residuals(first, second, axis, SymmetryBranch::Centers);
            if !centers.squared_norm().is_finite() {
                return Err(SymmetryError::NonFinite);
            }
            Ok(SymmetryBranch::Centers)
        }
        _ => Err(SymmetryError::UnsupportedPair),
    }
}

fn representative(curve: CurveGeometry) -> [f64; 2] {
    match curve {
        CurveGeometry::Segment { from, to } => {
            [from[0] * 0.5 + to[0] * 0.5, from[1] * 0.5 + to[1] * 0.5]
        }
        CurveGeometry::Circular(circular) => circular.center,
    }
}

/// Return one validated locus on the authored symmetry axis.
///
/// # Errors
///
/// Returns an error when the axis or branch is invalid, the curves do not match the branch, or
/// the relation is not satisfied within the shared residual tolerance.
pub fn symmetry_witness(
    first: CurveGeometry,
    second: CurveGeometry,
    axis: CurveGeometry,
    branch: SymmetryBranch,
) -> Result<SymmetryWitness, SymmetryError> {
    if !finite_curve(first) || !finite_curve(second) {
        return Err(SymmetryError::NonFinite);
    }
    let evaluated = residuals(first, second, axis, branch);
    if !evaluated.valid_axis {
        axis_frame(axis)?;
        return Err(SymmetryError::InvalidBranch);
    }
    if !evaluated.satisfied() {
        return Err(SymmetryError::Unsatisfied);
    }
    let (origin, along) = axis_frame(axis)?;
    let first = representative(first);
    let second = representative(second);
    let midpoint = [
        first[0] * 0.5 + second[0] * 0.5,
        first[1] * 0.5 + second[1] * 0.5,
    ];
    let delta = [midpoint[0] - origin[0], midpoint[1] - origin[1]];
    let projection = delta[0] * along[0] + delta[1] * along[1];
    let at = [
        origin[0] + projection * along[0],
        origin[1] + projection * along[1],
    ];
    if !finite_point(at) {
        return Err(SymmetryError::NonFinite);
    }
    Ok(SymmetryWitness { at })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::ArcDomain;

    #[test]
    fn chooses_direct_and_reversed_segment_correspondence() {
        let axis = CurveGeometry::Segment {
            from: [0.0, -5.0],
            to: [0.0, 5.0],
        };
        let first = CurveGeometry::Segment {
            from: [-3.0, 1.0],
            to: [-2.0, 4.0],
        };
        let direct = CurveGeometry::Segment {
            from: [3.0, 1.0],
            to: [2.0, 4.0],
        };
        let reversed = CurveGeometry::Segment {
            from: [2.0, 4.0],
            to: [3.0, 1.0],
        };
        assert_eq!(
            choose_symmetry_branch(first, direct, axis),
            Ok(SymmetryBranch::Direct)
        );
        assert_eq!(
            choose_symmetry_branch(first, reversed, axis),
            Ok(SymmetryBranch::Reversed)
        );
        let reversed_axis = CurveGeometry::Segment {
            from: [0.0, 5.0],
            to: [0.0, -5.0],
        };
        assert_eq!(
            choose_symmetry_branch(direct, first, reversed_axis),
            Ok(SymmetryBranch::Direct)
        );
        assert_eq!(
            residuals(first, direct, axis, SymmetryBranch::Direct).count,
            4
        );
    }

    /// An arc pair has no branch to choose. A mirror reverses the sense of travel and a stored arc
    /// only ever runs counter-clockwise (ADR 0038), so the tail of one always answers the head of
    /// the other, and asking for `Direct` cannot make it otherwise.
    #[test]
    fn an_arc_pair_always_mirrors_end_for_end() {
        let arc = |from, to, sweep| {
            CurveGeometry::Circular(CircularCurve {
                center: [0.0, 0.0],
                radius: 1.0,
                arc: Some(ArcDomain {
                    from,
                    to,
                    sweep_radians: sweep,
                }),
            })
        };
        let axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let first = arc([-1.0, 0.0], [-1.0, 1.0], 1.0);
        let second = arc([1.0, 1.0], [1.0, 0.0], 1.0);
        assert_eq!(
            choose_symmetry_branch(first, second, axis),
            Ok(SymmetryBranch::Reversed)
        );
        assert_eq!(
            residuals(first, second, axis, SymmetryBranch::Reversed).count,
            5
        );
        assert!(residuals(first, second, axis, SymmetryBranch::Reversed).satisfied());
        // The stored branch is not read for arcs, so the same pair reads the same either way.
        assert!(residuals(first, second, axis, SymmetryBranch::Direct).satisfied());
        // Ends in the OTHER correspondence are not symmetric at all, whatever branch is named.
        let end_for_end = arc([1.0, 0.0], [1.0, 1.0], 1.0);
        assert!(!residuals(first, end_for_end, axis, SymmetryBranch::Direct).satisfied());
        assert!(!residuals(first, end_for_end, axis, SymmetryBranch::Reversed).satisfied());
    }

    #[test]
    fn circles_use_two_center_rows_and_one_radius_row() {
        let circle = |center, radius| {
            CurveGeometry::Circular(CircularCurve {
                center,
                radius,
                arc: None,
            })
        };
        let axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let equal = residuals(
            circle([-2.0, 1.0], 3.0),
            circle([2.0, 1.0], 3.0),
            axis,
            SymmetryBranch::Centers,
        );
        assert_eq!(equal.count, 3);
        assert!(equal.satisfied());
        assert!(!residuals(
            circle([-2.0, 1.0], 3.0),
            circle([2.0, 1.0], 4.0),
            axis,
            SymmetryBranch::Centers,
        )
        .satisfied());
    }

    #[test]
    fn witness_uses_subject_midpoint_on_an_oblique_axis() {
        let axis = CurveGeometry::Segment {
            from: [-4.0, -4.0],
            to: [4.0, 4.0],
        };
        let first = CurveGeometry::Segment {
            from: [-2.0, 0.0],
            to: [-1.0, 1.0],
        };
        let second = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [1.0, -1.0],
        };
        let result = symmetry_witness(first, second, axis, SymmetryBranch::Direct);
        assert!(result.is_ok());
        let witness = result.unwrap_or(SymmetryWitness { at: [f64::NAN; 2] });
        assert!((witness.at[0] + 0.5).abs() < 1.0e-12);
        assert!((witness.at[1] + 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn witness_remains_finite_for_satisfied_large_finite_segments() {
        let axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let offset = f64::MAX * 0.75;
        let first = CurveGeometry::Segment {
            from: [offset, 0.0],
            to: [offset, 1.0],
        };
        let second = CurveGeometry::Segment {
            from: [-offset, 0.0],
            to: [-offset, 1.0],
        };
        let witness = symmetry_witness(first, second, axis, SymmetryBranch::Direct)
            .unwrap_or(SymmetryWitness { at: [f64::NAN; 2] });
        assert!(witness.at[0].abs() < f64::EPSILON);
        assert!((witness.at[1] - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn witness_rejects_an_arc_with_a_non_finite_outer_radius() {
        let axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let arc = |x, radius, sweep_radians| {
            CurveGeometry::Circular(CircularCurve {
                center: [x, 0.5],
                radius,
                arc: Some(ArcDomain {
                    from: [x, 0.0],
                    to: [x, 1.0],
                    sweep_radians,
                }),
            })
        };
        assert_eq!(
            symmetry_witness(
                arc(-1.0, f64::NAN, 1.0),
                arc(1.0, 1.0, -1.0),
                axis,
                SymmetryBranch::Direct,
            ),
            Err(SymmetryError::NonFinite)
        );
    }

    #[test]
    fn degeneracy_threshold_is_exact_and_never_satisfied() {
        let subject = CurveGeometry::Segment {
            from: [-1.0, 0.0],
            to: [-1.0, 1.0],
        };
        for (length, invalid) in [
            (DEGENERATE_AXIS_SPAN, true),
            (DEGENERATE_AXIS_SPAN + f64::EPSILON, false),
        ] {
            let axis = CurveGeometry::Segment {
                from: [0.0, 0.0],
                to: [length, 0.0],
            };
            let evaluated = residuals(subject, subject, axis, SymmetryBranch::Direct);
            assert_eq!(!evaluated.valid_axis, invalid);
            assert!(!evaluated.satisfied());
        }
    }

    #[test]
    fn branch_choice_rejects_non_finite_segments_arcs_and_circles() {
        let axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let segment = CurveGeometry::Segment {
            from: [-1.0, 0.0],
            to: [-1.0, 1.0],
        };
        let arc = |sweep_radians| {
            CurveGeometry::Circular(CircularCurve {
                center: [-1.0, 0.0],
                radius: 1.0,
                arc: Some(ArcDomain {
                    from: [-1.0, 0.0],
                    to: [-1.0, 1.0],
                    sweep_radians,
                }),
            })
        };
        let circle = |radius| {
            CurveGeometry::Circular(CircularCurve {
                center: [-1.0, 0.0],
                radius,
                arc: None,
            })
        };
        for (first, second) in [
            (
                CurveGeometry::Segment {
                    from: [f64::NAN, 0.0],
                    to: [-1.0, 1.0],
                },
                segment,
            ),
            (arc(f64::INFINITY), arc(1.0)),
            (circle(f64::NAN), circle(1.0)),
        ] {
            assert_eq!(
                choose_symmetry_branch(first, second, axis),
                Err(SymmetryError::NonFinite)
            );
        }
    }

    #[test]
    fn branch_choice_rejects_extreme_finite_axis_and_subject_overflow() {
        let ordinary_axis = CurveGeometry::Segment {
            from: [0.0, -2.0],
            to: [0.0, 2.0],
        };
        let subject = CurveGeometry::Segment {
            from: [-1.0, 0.0],
            to: [-1.0, 1.0],
        };
        let overflowing_axis = CurveGeometry::Segment {
            from: [-f64::MAX, 0.0],
            to: [f64::MAX, 0.0],
        };
        assert_eq!(
            choose_symmetry_branch(subject, subject, overflowing_axis),
            Err(SymmetryError::NonFinite)
        );

        let distant_axis = CurveGeometry::Segment {
            from: [-f64::MAX / 2.0, -1.0],
            to: [-f64::MAX / 2.0, 1.0],
        };
        let extreme_segment = CurveGeometry::Segment {
            from: [f64::MAX, 0.0],
            to: [f64::MAX, 1.0],
        };
        assert_eq!(
            choose_symmetry_branch(extreme_segment, subject, distant_axis),
            Err(SymmetryError::NonFinite)
        );

        let circle = |center| {
            CurveGeometry::Circular(CircularCurve {
                center,
                radius: 1.0,
                arc: None,
            })
        };
        assert_eq!(
            choose_symmetry_branch(circle([f64::MAX, 0.0]), circle([0.0, 0.0]), distant_axis,),
            Err(SymmetryError::NonFinite)
        );

        let arc = |x| {
            CurveGeometry::Circular(CircularCurve {
                center: [x, 0.5],
                radius: 1.0,
                arc: Some(ArcDomain {
                    from: [x, 0.0],
                    to: [x, 1.0],
                    sweep_radians: 1.0,
                }),
            })
        };
        assert_eq!(
            choose_symmetry_branch(arc(f64::MAX), arc(0.0), distant_axis),
            Err(SymmetryError::NonFinite)
        );
        assert!(choose_symmetry_branch(subject, subject, ordinary_axis).is_ok());
    }
}
