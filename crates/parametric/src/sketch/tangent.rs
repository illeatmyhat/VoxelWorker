//! Branch-stable tangent mathematics for planar sketch curves.
//!
//! This module intentionally knows no solver handles or document ids. Callers adapt their curve
//! storage to [`CurveGeometry`], then use the same branch, contact, residual, and finite-domain
//! rules for solving, validation, and presentation.

#![allow(
    clippy::doc_markdown,
    clippy::imprecise_flops,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::suboptimal_flops,
    clippy::too_long_first_doc_paragraph,
    clippy::too_many_lines
)]

use super::curve::{
    CurveGeometry, COLLAPSE_TOLERANCE as COLLAPSED_SPAN,
    SATISFACTION_TOLERANCE as SATISFIED_RESIDUAL,
};

/// Which directed side of a segment's authored `from → to` direction a circular curve touches.
/// This belongs to the segment itself and is consequently unchanged when a relation's members swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LineSide {
    Left,
    Right,
}

/// The ordered relation member that contains the other under an internal tangent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InternalContainment {
    First,
    Second,
}

/// The durable Tangent solution branch. It is a solution choice rather than a transient contact
/// coordinate, and remains stable when a sketch is re-solved or density changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TangentBranch {
    Line(LineSide),
    External,
    Internal { contains: InternalContainment },
}

impl TangentBranch {
    /// Rebind the branch after swapping persisted relation members. Only an internal container
    /// names member position; LineSide continues to mean the segment's stored directed side.
    pub const fn remap_for_swapped_members(self) -> Self {
        match self {
            Self::Internal {
                contains: InternalContainment::First,
            } => Self::Internal {
                contains: InternalContainment::Second,
            },
            Self::Internal {
                contains: InternalContainment::Second,
            } => Self::Internal {
                contains: InternalContainment::First,
            },
            Self::Line(side) => Self::Line(side),
            Self::External => Self::External,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TangentContact {
    pub at: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TangentContactError {
    InvalidBranch,
    NonFinite,
    Degenerate,
    Containment,
    NotCoincident,
    OutsideFirstDomain,
    OutsideSecondDomain,
}

/// A user-facing alias: callers choose a Tangent branch from curve values plus session-only loci.
pub type TangentCurve = CurveGeometry;

/// Why no branch can be chosen from the supplied transient geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchChoiceError {
    UnsupportedPair,
    NonFinite,
    Degenerate,
}

pub(super) fn branch_matches(
    first: CurveGeometry,
    second: CurveGeometry,
    branch: TangentBranch,
) -> bool {
    match branch {
        TangentBranch::Line(_) => matches!(
            (first, second),
            (CurveGeometry::Segment { .. }, CurveGeometry::Circular(_))
                | (CurveGeometry::Circular(_), CurveGeometry::Segment { .. })
        ),
        TangentBranch::External | TangentBranch::Internal { .. } => {
            matches!(
                (first, second),
                (CurveGeometry::Circular(_), CurveGeometry::Circular(_))
            )
        }
    }
}

/// A bounded domain tolerance derived from the solver's residual tolerance and curve magnitude.
pub(super) fn contact_tolerance(scale: f64) -> f64 {
    let scale = scale.max(COLLAPSED_SPAN);
    (SATISFIED_RESIDUAL * 8.0 * scale.min(1.0))
        .max(f64::EPSILON * 64.0 * scale)
        .min(1.0e-3)
}

pub(super) fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub(super) fn residual(first: CurveGeometry, second: CurveGeometry, branch: TangentBranch) -> f64 {
    match branch {
        TangentBranch::Line(side) => {
            let (from, to, circular) = match (first, second) {
                (CurveGeometry::Segment { from, to }, CurveGeometry::Circular(circular))
                | (CurveGeometry::Circular(circular), CurveGeometry::Segment { from, to }) => {
                    (from, to, circular)
                }
                _ => return f64::INFINITY,
            };
            let span = [to[0] - from[0], to[1] - from[1]];
            let length = (span[0] * span[0] + span[1] * span[1]).sqrt();
            if length <= f64::EPSILON {
                return 0.0;
            }
            let left = [-span[1] / length, span[0] / length];
            let sign = match side {
                LineSide::Left => 1.0,
                LineSide::Right => -1.0,
            };
            (circular.center[0] - from[0]) * left[0] + (circular.center[1] - from[1]) * left[1]
                - sign * circular.radius
        }
        TangentBranch::External | TangentBranch::Internal { .. } => {
            let (CurveGeometry::Circular(first), CurveGeometry::Circular(second)) = (first, second)
            else {
                return f64::INFINITY;
            };
            let center_distance = distance(first.center, second.center);
            match branch {
                TangentBranch::External => center_distance - (first.radius + second.radius),
                TangentBranch::Internal {
                    contains: InternalContainment::First,
                } => center_distance - (first.radius - second.radius),
                TangentBranch::Internal {
                    contains: InternalContainment::Second,
                } => center_distance - (second.radius - first.radius),
                TangentBranch::Line(_) => f64::INFINITY,
            }
        }
    }
}

/// The pair of support contacts predicted by one branch. They are intentionally distinct until
/// `contact_of` validates coincidence: branch choice can score a pre-solve drawing that is not yet
/// tangent, while validation cannot accidentally treat the two supports as one contact.
fn branch_contacts(
    first: CurveGeometry,
    second: CurveGeometry,
    branch: TangentBranch,
) -> Result<([f64; 2], [f64; 2]), TangentContactError> {
    if !branch_matches(first, second, branch) {
        return Err(TangentContactError::InvalidBranch);
    }
    let finite = |point: [f64; 2]| point.into_iter().all(f64::is_finite);
    match branch {
        TangentBranch::Line(side) => {
            let (from, to, circular) = match (first, second) {
                (CurveGeometry::Segment { from, to }, CurveGeometry::Circular(circular))
                | (CurveGeometry::Circular(circular), CurveGeometry::Segment { from, to }) => {
                    (from, to, circular)
                }
                _ => return Err(TangentContactError::InvalidBranch),
            };
            let span = [to[0] - from[0], to[1] - from[1]];
            let length = (span[0] * span[0] + span[1] * span[1]).sqrt();
            if length <= COLLAPSED_SPAN || circular.radius <= COLLAPSED_SPAN {
                return Err(TangentContactError::Degenerate);
            }
            let normal = [-span[1] / length, span[0] / length];
            let sign = match side {
                LineSide::Left => 1.0,
                LineSide::Right => -1.0,
            };
            let on_circle = [
                circular.center[0] - sign * circular.radius * normal[0],
                circular.center[1] - sign * circular.radius * normal[1],
            ];
            let along = [span[0] / length, span[1] / length];
            let projection = (circular.center[0] - from[0]) * along[0]
                + (circular.center[1] - from[1]) * along[1];
            let on_line = [
                from[0] + projection * along[0],
                from[1] + projection * along[1],
            ];
            if !finite(on_circle) || !finite(on_line) || !finite(circular.center) {
                return Err(TangentContactError::NonFinite);
            }
            Ok(match first {
                CurveGeometry::Segment { .. } => (on_line, on_circle),
                CurveGeometry::Circular(_) => (on_circle, on_line),
            })
        }
        TangentBranch::External | TangentBranch::Internal { .. } => {
            let (CurveGeometry::Circular(first), CurveGeometry::Circular(second)) = (first, second)
            else {
                return Err(TangentContactError::InvalidBranch);
            };
            let axis = [
                second.center[0] - first.center[0],
                second.center[1] - first.center[1],
            ];
            let axis_length = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
            if axis_length <= COLLAPSED_SPAN
                || first.radius <= COLLAPSED_SPAN
                || second.radius <= COLLAPSED_SPAN
            {
                return Err(TangentContactError::Degenerate);
            }
            let along = [axis[0] / axis_length, axis[1] / axis_length];
            let (first_contact, second_contact) = match branch {
                TangentBranch::External => (
                    [
                        first.center[0] + first.radius * along[0],
                        first.center[1] + first.radius * along[1],
                    ],
                    [
                        second.center[0] - second.radius * along[0],
                        second.center[1] - second.radius * along[1],
                    ],
                ),
                TangentBranch::Internal {
                    contains: InternalContainment::First,
                } => {
                    if first.radius <= second.radius + contact_tolerance(first.radius) {
                        return Err(TangentContactError::Containment);
                    }
                    (
                        [
                            first.center[0] + first.radius * along[0],
                            first.center[1] + first.radius * along[1],
                        ],
                        [
                            second.center[0] + second.radius * along[0],
                            second.center[1] + second.radius * along[1],
                        ],
                    )
                }
                TangentBranch::Internal {
                    contains: InternalContainment::Second,
                } => {
                    if second.radius <= first.radius + contact_tolerance(second.radius) {
                        return Err(TangentContactError::Containment);
                    }
                    (
                        [
                            first.center[0] - first.radius * along[0],
                            first.center[1] - first.radius * along[1],
                        ],
                        [
                            second.center[0] - second.radius * along[0],
                            second.center[1] - second.radius * along[1],
                        ],
                    )
                }
                TangentBranch::Line(_) => return Err(TangentContactError::InvalidBranch),
            };
            if !finite(first_contact) || !finite(second_contact) {
                return Err(TangentContactError::NonFinite);
            }
            Ok((first_contact, second_contact))
        }
    }
}

/// Derive a contact from a solved branch and require its two independently-computed curve points
/// to agree. The midpoint is only a transient numeric witness; no caller may persist it.
pub(super) fn contact_of(
    first: CurveGeometry,
    second: CurveGeometry,
    branch: TangentBranch,
) -> Result<TangentContact, TangentContactError> {
    let (first_contact, second_contact) = branch_contacts(first, second, branch)?;
    let scale = match (first, second) {
        (CurveGeometry::Segment { from, to }, CurveGeometry::Circular(circle))
        | (CurveGeometry::Circular(circle), CurveGeometry::Segment { from, to }) => {
            distance(from, to).max(circle.radius)
        }
        (CurveGeometry::Circular(first), CurveGeometry::Circular(second)) => first
            .radius
            .max(second.radius)
            .max(distance(first.center, second.center)),
        _ => return Err(TangentContactError::InvalidBranch),
    };
    if distance(first_contact, second_contact) > contact_tolerance(scale) {
        return Err(TangentContactError::NotCoincident);
    }
    Ok(TangentContact {
        at: [
            (first_contact[0] + second_contact[0]) * 0.5,
            (first_contact[1] + second_contact[1]) * 0.5,
        ],
    })
}

/// Derive the unique finite authored contact for a stored tangent branch. Solver validation and
/// presentation both call this door so neither can silently use an infinite supporting curve.
pub fn tangent_contact(
    first: CurveGeometry,
    second: CurveGeometry,
    branch: TangentBranch,
) -> Result<TangentContact, TangentContactError> {
    let contact = contact_of(first, second, branch)?;
    if !contains_contact(first, contact.at) {
        return Err(TangentContactError::OutsideFirstDomain);
    }
    if !contains_contact(second, contact.at) {
        return Err(TangentContactError::OutsideSecondDomain);
    }
    Ok(contact)
}

/// Choose a persisted branch from two session-only click loci. Callers canonicalize their stable
/// member order first and pass the matching `(curve, locus)` pairs, so `First` / `Second` and ties
/// remain stable through reversed user pick order, undo, and reload.
pub fn choose_branch(
    first: TangentCurve,
    first_locus: [f64; 2],
    second: TangentCurve,
    second_locus: [f64; 2],
) -> Result<TangentBranch, BranchChoiceError> {
    if !first_locus
        .into_iter()
        .chain(second_locus)
        .all(f64::is_finite)
    {
        return Err(BranchChoiceError::NonFinite);
    }
    let candidates: &[TangentBranch] = match (first, second) {
        (CurveGeometry::Segment { .. }, CurveGeometry::Circular(_))
        | (CurveGeometry::Circular(_), CurveGeometry::Segment { .. }) => &[
            TangentBranch::Line(LineSide::Left),
            TangentBranch::Line(LineSide::Right),
        ],
        (CurveGeometry::Circular(_), CurveGeometry::Circular(_)) => &[
            TangentBranch::External,
            TangentBranch::Internal {
                contains: InternalContainment::First,
            },
            TangentBranch::Internal {
                contains: InternalContainment::Second,
            },
        ],
        (CurveGeometry::Segment { .. }, CurveGeometry::Segment { .. }) => {
            return Err(BranchChoiceError::UnsupportedPair)
        }
    };
    let mut best: Option<(TangentBranch, f64)> = None;
    for branch in candidates {
        let Ok((first_contact, second_contact)) = branch_contacts(first, second, *branch) else {
            continue;
        };
        let score = distance(first_contact, first_locus) + distance(second_contact, second_locus);
        if score.is_finite() && best.is_none_or(|(_, held)| score < held) {
            best = Some((*branch, score));
        }
    }
    best.map(|(branch, _)| branch)
        .ok_or(BranchChoiceError::Degenerate)
}

pub(super) fn contains_contact(curve: CurveGeometry, contact: [f64; 2]) -> bool {
    match curve {
        CurveGeometry::Segment { from, to } => {
            let span = [to[0] - from[0], to[1] - from[1]];
            let length_squared = span[0] * span[0] + span[1] * span[1];
            if length_squared <= COLLAPSED_SPAN.powi(2) {
                return false;
            }
            let length = length_squared.sqrt();
            let tolerance = contact_tolerance(length);
            let delta = [contact[0] - from[0], contact[1] - from[1]];
            let parameter = (delta[0] * span[0] + delta[1] * span[1]) / length_squared;
            let perpendicular = (delta[0] * span[1] - delta[1] * span[0]).abs() / length;
            perpendicular <= tolerance
                && parameter >= -tolerance / length
                && parameter <= 1.0 + tolerance / length
        }
        CurveGeometry::Circular(circular) => {
            if !circular.center.into_iter().all(f64::is_finite) || !circular.radius.is_finite() {
                return false;
            }
            let radial = distance(contact, circular.center);
            if (radial - circular.radius).abs() > contact_tolerance(circular.radius) {
                return false;
            }
            let Some(arc) = circular.arc else {
                return true;
            };
            if !arc.sweep_radians.is_finite() || arc.sweep_radians.abs() <= f64::EPSILON {
                return false;
            }
            let start_angle =
                (arc.from[1] - circular.center[1]).atan2(arc.from[0] - circular.center[0]);
            let contact_angle =
                (contact[1] - circular.center[1]).atan2(contact[0] - circular.center[0]);
            let travel = if arc.sweep_radians.is_sign_positive() {
                (contact_angle - start_angle).rem_euclid(std::f64::consts::TAU)
            } else {
                (start_angle - contact_angle).rem_euclid(std::f64::consts::TAU)
            };
            let scale = circular.radius.max(distance(arc.from, arc.to));
            let angular_tolerance =
                (contact_tolerance(scale) / circular.radius).clamp(f64::EPSILON * 64.0, 1.0e-3);
            // `travel` alone wraps a contact a hair before the start to almost one full turn.
            // Endpoints are authored domain boundaries, so the same bounded tolerance applies on
            // either side of both ends and works for positive and negative sweeps.
            let near_start = (contact_angle - start_angle)
                .abs()
                .min(std::f64::consts::TAU - (contact_angle - start_angle).abs())
                <= angular_tolerance;
            travel <= arc.sweep_radians.abs() + angular_tolerance || near_start
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::{ArcDomain, CircularCurve};

    fn arc(sweep_radians: f64) -> CurveGeometry {
        let (from, to) = if sweep_radians.is_sign_positive() {
            ([4.0, 0.0], [0.0, 4.0])
        } else {
            ([0.0, 4.0], [4.0, 0.0])
        };
        CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 4.0,
            arc: Some(ArcDomain {
                from,
                to,
                sweep_radians,
            }),
        })
    }

    #[test]
    fn arc_domain_accepts_both_endpoints_for_positive_and_negative_sweeps() {
        for sweep in [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
            let CurveGeometry::Circular(CircularCurve {
                arc: Some(domain), ..
            }) = arc(sweep)
            else {
                return;
            };
            assert!(contains_contact(arc(sweep), domain.from));
            assert!(contains_contact(arc(sweep), domain.to));
        }
    }

    #[test]
    fn arc_endpoint_tolerance_is_bounded_on_both_sweep_signs() {
        for sweep in [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2] {
            let CurveGeometry::Circular(CircularCurve {
                radius,
                arc: Some(domain),
                ..
            }) = arc(sweep)
            else {
                return;
            };
            let angular = contact_tolerance(radius) / radius;
            let start = domain.from;
            let start_angle = start[1].atan2(start[0]);
            let direction = sweep.signum();
            let inside = [
                radius * (start_angle + direction * angular * 0.5).cos(),
                radius * (start_angle + direction * angular * 0.5).sin(),
            ];
            let outside = [
                radius * (start_angle - direction * angular * 2.0).cos(),
                radius * (start_angle - direction * angular * 2.0).sin(),
            ];
            assert!(contains_contact(arc(sweep), inside));
            assert!(!contains_contact(arc(sweep), outside));
        }
    }

    #[test]
    fn supporting_circle_contact_outside_an_authored_arc_is_rejected() {
        let curve = arc(std::f64::consts::FRAC_PI_2);
        assert!(!contains_contact(curve, [-4.0, 0.0]));
    }

    #[test]
    fn degenerate_nonfinite_and_containment_boundaries_refuse_deterministically() {
        let line = CurveGeometry::Segment {
            from: [0.0, 0.0],
            to: [0.0, 0.0],
        };
        let circle = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 1.0],
            radius: 1.0,
            arc: None,
        });
        assert_eq!(
            contact_of(line, circle, TangentBranch::Line(LineSide::Left)),
            Err(TangentContactError::Degenerate)
        );
        let live_line = CurveGeometry::Segment {
            from: [-1.0, 0.0],
            to: [1.0, 0.0],
        };
        let nonpositive = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 0.0,
            arc: None,
        });
        assert_eq!(
            contact_of(live_line, nonpositive, TangentBranch::Line(LineSide::Left)),
            Err(TangentContactError::Degenerate)
        );
        let first = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 2.0,
            arc: None,
        });
        let coincident = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 1.0,
            arc: None,
        });
        assert_eq!(
            contact_of(first, coincident, TangentBranch::External),
            Err(TangentContactError::Degenerate)
        );
        let nan = CurveGeometry::Circular(CircularCurve {
            center: [f64::NAN, 0.0],
            radius: 1.0,
            arc: None,
        });
        assert_eq!(
            contact_of(live_line, nan, TangentBranch::Line(LineSide::Left)),
            Err(TangentContactError::NonFinite)
        );
        let tolerance = contact_tolerance(4.0);
        let container = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 4.0,
            arc: None,
        });
        let equal_within = CurveGeometry::Circular(CircularCurve {
            center: [3.0, 0.0],
            radius: 4.0 - tolerance * 0.5,
            arc: None,
        });
        assert_eq!(
            contact_of(
                container,
                equal_within,
                TangentBranch::Internal {
                    contains: InternalContainment::First
                }
            ),
            Err(TangentContactError::Containment)
        );
        let valid = CurveGeometry::Circular(CircularCurve {
            center: [3.0, 0.0],
            radius: 4.0 - tolerance * 2.0,
            arc: None,
        });
        assert!(matches!(
            contact_of(
                container,
                valid,
                TangentBranch::Internal {
                    contains: InternalContainment::First
                }
            ),
            Err(TangentContactError::NotCoincident)
        ));
    }

    #[test]
    fn choose_branch_scores_loci_and_is_stable_after_canonical_member_reordering() {
        let line = CurveGeometry::Segment {
            from: [-10.0, 0.0],
            to: [10.0, 0.0],
        };
        let circle = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 4.0],
            radius: 4.0,
            arc: None,
        });
        assert_eq!(
            choose_branch(line, [0.0, 0.0], circle, [0.0, 0.0]),
            Ok(TangentBranch::Line(LineSide::Left))
        );
        assert_eq!(
            choose_branch(circle, [0.0, 0.0], line, [0.0, 0.0]),
            Ok(TangentBranch::Line(LineSide::Left))
        );

        let outer = CurveGeometry::Circular(CircularCurve {
            center: [0.0, 0.0],
            radius: 6.0,
            arc: None,
        });
        let inner = CurveGeometry::Circular(CircularCurve {
            center: [4.0, 0.0],
            radius: 2.0,
            arc: None,
        });
        assert_eq!(
            choose_branch(outer, [6.0, 0.0], inner, [6.0, 0.0]),
            Ok(TangentBranch::Internal {
                contains: InternalContainment::First
            })
        );
        assert_eq!(
            choose_branch(inner, [6.0, 0.0], outer, [6.0, 0.0]),
            Ok(TangentBranch::Internal {
                contains: InternalContainment::Second
            })
        );
    }
}
