//! Continuous constructions for ellipse and conic sketch tools.
//!
//! These functions know gesture geometry but no document ids, density, or persistence. Their
//! output is the shared rational-cubic substrate representation, so preview and commit cannot
//! disagree about which conic was authored.

use substrate::rational_bezier::RationalBezier;

/// Four exact rational-cubic quarters forming one ellipse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EllipseCandidate {
    pub center: [f64; 2],
    pub major_axis: [f64; 2],
    pub minor_radius: f64,
    pub quarters: [RationalBezier; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipseCandidateError {
    NonFinite,
    CollapsedMajorAxis,
    CollapsedMinorAxis,
}

/// Build an ellipse from its center, a major semi-axis endpoint, and a width pick.
///
/// The third point is projected onto the line perpendicular to the major axis, matching an
/// interactive axis gesture while guaranteeing orthogonal ellipse axes.
///
/// # Errors
///
/// Returns a typed error for non-finite input or either collapsed semi-axis.
pub fn ellipse_candidate(
    center: [f64; 2],
    major_endpoint: [f64; 2],
    width_pick: [f64; 2],
) -> Result<EllipseCandidate, EllipseCandidateError> {
    if ![center, major_endpoint, width_pick]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    {
        return Err(EllipseCandidateError::NonFinite);
    }
    let major_axis = [major_endpoint[0] - center[0], major_endpoint[1] - center[1]];
    let major_radius = major_axis[0].hypot(major_axis[1]);
    if major_radius <= f64::EPSILON {
        return Err(EllipseCandidateError::CollapsedMajorAxis);
    }
    let unit_major = [major_axis[0] / major_radius, major_axis[1] / major_radius];
    let unit_minor = [-unit_major[1], unit_major[0]];
    let width_offset = [width_pick[0] - center[0], width_pick[1] - center[1]];
    let signed_minor = width_offset[0].mul_add(unit_minor[0], width_offset[1] * unit_minor[1]);
    let minor_radius = signed_minor.abs();
    if minor_radius <= f64::EPSILON {
        return Err(EllipseCandidateError::CollapsedMinorAxis);
    }
    let minor_axis = [
        unit_minor[0] * signed_minor.signum() * minor_radius,
        unit_minor[1] * signed_minor.signum() * minor_radius,
    ];
    let at = |major: f64, minor: f64| {
        [
            major_axis[0].mul_add(major, minor_axis[0].mul_add(minor, center[0])),
            major_axis[1].mul_add(major, minor_axis[1].mul_add(minor, center[1])),
        ]
    };
    let quarter = |from: [f64; 2], tangent_intersection: [f64; 2], to: [f64; 2]| {
        RationalBezier::elevated_quadratic(
            [from, tangent_intersection, to],
            [1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        )
    };
    Ok(EllipseCandidate {
        center,
        major_axis,
        minor_radius,
        quarters: [
            quarter(at(1.0, 0.0), at(1.0, 1.0), at(0.0, 1.0)),
            quarter(at(0.0, 1.0), at(-1.0, 1.0), at(-1.0, 0.0)),
            quarter(at(-1.0, 0.0), at(-1.0, -1.0), at(0.0, -1.0)),
            quarter(at(0.0, -1.0), at(1.0, -1.0), at(1.0, 0.0)),
        ],
    })
}

/// One endpoint/vertex/rho conic, exactly represented as a rational cubic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicCandidate {
    pub from: [f64; 2],
    pub to: [f64; 2],
    pub vertex: [f64; 2],
    pub rho: f64,
    pub curve: RationalBezier,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConicCandidateError {
    NonFinite,
    CoincidentEndpoints,
    InvalidRho,
    CollapsedVertex,
}

/// The rho a conic reads at before its control point is dragged.
///
/// A parabola: the exact boundary between the elliptic and hyperbolic halves of the family, and so
/// the reading that presumes least about which half the author is reaching for.
pub const CONIC_PARABOLIC_RHO: f64 = 0.5;

/// How close to degenerate a dragged control point is allowed to get.
///
/// Rho lives on the OPEN interval `(0, 1)`, which has no endpoints to clamp a drag against. Both
/// ends are curves nobody means to author — at 0 the conic is its own chord, at 1 it is a corner —
/// so the gizmo stops just short of each.
const CONIC_RHO_MARGIN: f64 = 1.0e-3;

/// Where a conic is pinned once its control point is first placed.
///
/// Placing the control point does two things at once: it aims the curve, and it fixes the point
/// the curve passes through — the parabolic shoulder halfway out to it. Dragging the control point
/// afterwards moves it along its own ray and changes only how hard it pulls, so the pinned point
/// is what stops that drag from dragging the whole curve with it.
///
/// `None` when the control point falls on the chord midpoint, where there is no curve to pin.
#[must_use]
pub fn conic_parabolic_shoulder(
    from: [f64; 2],
    to: [f64; 2],
    control: [f64; 2],
) -> Option<[f64; 2]> {
    if ![from, to, control]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    {
        return None;
    }
    let midpoint = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
    let reach = (control[0] - midpoint[0]).hypot(control[1] - midpoint[1]);
    (reach > f64::EPSILON).then(|| {
        [
            CONIC_PARABOLIC_RHO.mul_add(control[0] - midpoint[0], midpoint[0]),
            CONIC_PARABOLIC_RHO.mul_add(control[1] - midpoint[1], midpoint[1]),
        ]
    })
}

/// The ray a conic's control point slides on, once the curve is pinned through `shoulder`.
///
/// Returned as `(chord midpoint, unit direction)`. The control point — where the two end tangents
/// meet — always lies on the far side of the on-curve point from the chord, so this ray is the
/// whole authoring space of the last step. How FAR along it the control point sits is rho, by
/// `rho = |midpoint→shoulder| / |midpoint→control|`: close in behind the shoulder is a sharp
/// hyperbola, far away is a flat ellipse.
///
/// `None` when the on-curve point falls on the chord midpoint, where there is no direction to
/// slide along and no conic to shape.
#[must_use]
pub fn conic_control_ray(
    from: [f64; 2],
    to: [f64; 2],
    shoulder: [f64; 2],
) -> Option<([f64; 2], [f64; 2])> {
    if ![from, to, shoulder]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    {
        return None;
    }
    let midpoint = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
    let reach = (shoulder[0] - midpoint[0]).hypot(shoulder[1] - midpoint[1]);
    (reach > f64::EPSILON).then(|| {
        (
            midpoint,
            [
                (shoulder[0] - midpoint[0]) / reach,
                (shoulder[1] - midpoint[1]) / reach,
            ],
        )
    })
}

/// The rho a cursor names when it drags the control point along [`conic_control_ray`].
///
/// The cursor is projected onto the ray and the result is clamped, because the gizmo is captive:
/// the author is choosing how far out along a known direction to sit, not pointing at a free
/// position that might miss. A cursor level with or behind the chord reads as the flattest curve
/// the band allows rather than as a failure. `None` only when there is no ray at all.
#[must_use]
pub fn conic_rho_from_control(
    from: [f64; 2],
    to: [f64; 2],
    shoulder: [f64; 2],
    control: [f64; 2],
) -> Option<f64> {
    if !control.iter().copied().all(f64::is_finite) {
        return None;
    }
    let (midpoint, unit) = conic_control_ray(from, to, shoulder)?;
    let reach = (shoulder[0] - midpoint[0]).hypot(shoulder[1] - midpoint[1]);
    let along = (control[0] - midpoint[0]).mul_add(unit[0], (control[1] - midpoint[1]) * unit[1]);
    if along <= 0.0 {
        return Some(CONIC_RHO_MARGIN);
    }
    Some((reach / along).clamp(CONIC_RHO_MARGIN, 1.0 - CONIC_RHO_MARGIN))
}

/// Where a given rho puts the control point, with the curve pinned through `shoulder`.
///
/// The inverse of [`conic_rho_from_control`], and what turns a rho back into something to draw and
/// grab. Feeding it a clamped rho is what keeps the gizmo on its ray no matter where the cursor
/// wanders.
#[must_use]
pub fn conic_control_from_rho(
    from: [f64; 2],
    to: [f64; 2],
    shoulder: [f64; 2],
    rho: f64,
) -> Option<[f64; 2]> {
    if !rho.is_finite() || rho <= 0.0 {
        return None;
    }
    let (midpoint, unit) = conic_control_ray(from, to, shoulder)?;
    let reach = (shoulder[0] - midpoint[0]).hypot(shoulder[1] - midpoint[1]);
    let along = reach / rho;
    Some([
        along.mul_add(unit[0], midpoint[0]),
        along.mul_add(unit[1], midpoint[1]),
    ])
}

/// Build Fusion's endpoint/vertex/rho conic.
///
/// `rho = 0.5` is parabolic; values below are elliptic and values above are hyperbolic. The open
/// interval keeps the equivalent rational-quadratic weight positive and finite.
///
/// # Errors
///
/// Returns a typed error for non-finite input, coincident endpoints, rho outside `(0, 1)`, or a
/// collapsed vertex construction.
pub fn conic_candidate(
    from: [f64; 2],
    to: [f64; 2],
    vertex: [f64; 2],
    rho: f64,
) -> Result<ConicCandidate, ConicCandidateError> {
    if ![from, to, vertex]
        .into_iter()
        .flatten()
        .chain(std::iter::once(rho))
        .all(f64::is_finite)
    {
        return Err(ConicCandidateError::NonFinite);
    }
    if (to[0] - from[0]).hypot(to[1] - from[1]) <= f64::EPSILON {
        return Err(ConicCandidateError::CoincidentEndpoints);
    }
    if !(0.0..1.0).contains(&rho) {
        return Err(ConicCandidateError::InvalidRho);
    }
    let weight = rho / (1.0 - rho);
    let middle = [
        0.5_f64.mul_add(-(from[0] + to[0]), (1.0 + weight) * vertex[0]) / weight,
        0.5_f64.mul_add(-(from[1] + to[1]), (1.0 + weight) * vertex[1]) / weight,
    ];
    let curve = RationalBezier::elevated_quadratic([from, middle, to], [1.0, weight, 1.0]);
    if !curve.is_valid()
        || curve
            .derivative_at(0.5)
            .iter()
            .all(|value| value.abs() <= f64::EPSILON)
    {
        return Err(ConicCandidateError::CollapsedVertex);
    }
    Ok(ConicCandidate {
        from,
        to,
        vertex,
        rho,
        curve,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::indexing_slicing, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn ellipse_quarters_join_and_hit_both_axes() {
        let ellipse = ellipse_candidate([2.0, 3.0], [6.0, 3.0], [2.0, 5.0]).unwrap();
        assert_eq!(ellipse.quarters[0].control[0], [6.0, 3.0]);
        assert_eq!(ellipse.quarters[0].control[3], [2.0, 5.0]);
        for index in 0..4 {
            assert_eq!(
                ellipse.quarters[index].control[3],
                ellipse.quarters[(index + 1) % 4].control[0]
            );
        }
    }

    #[test]
    fn the_control_ray_runs_outward_from_the_chord_through_the_on_curve_point() {
        let (midpoint, unit) = conic_control_ray([-2.0, 0.0], [2.0, 0.0], [0.0, 6.0]).unwrap();
        assert_eq!(midpoint, [0.0, 0.0]);
        assert_eq!(unit, [0.0, 1.0]);
        assert!(conic_control_ray([-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]).is_none());
    }

    #[test]
    fn a_control_point_drag_is_captive_to_its_ray() {
        let rho =
            |control| conic_rho_from_control([-2.0, 0.0], [2.0, 0.0], [0.0, 2.0], control).unwrap();
        // Twice the on-curve point's reach names the parabola, and sideways drift off the ray does
        // not change that: only the distance ALONG it is the parameter.
        assert!((rho([0.0, 4.0]) - 0.5).abs() < 1.0e-12);
        assert!((rho([9.0, 4.0]) - 0.5).abs() < 1.0e-12);
        // Inside the curve, or behind the chord entirely, the gizmo stops rather than leaving the
        // open interval.
        assert!(rho([0.0, 1.0]) < 1.0);
        assert!(rho([0.0, -50.0]) > 0.0);
    }

    #[test]
    fn a_control_point_and_its_rho_are_inverses_and_pin_the_curve() {
        let (from, to, shoulder) = ([-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]);
        for rho in [0.25, 0.5, 0.75] {
            let control = conic_control_from_rho(from, to, shoulder, rho).unwrap();
            let recovered = conic_rho_from_control(from, to, shoulder, control).unwrap();
            assert!((recovered - rho).abs() < 1.0e-12);
            // Whatever the control point does, the curve keeps passing through the on-curve pick.
            let conic = conic_candidate(from, to, shoulder, rho).unwrap();
            let point = conic.curve.point_at(0.5);
            assert!((point[0] - shoulder[0]).abs() < 1.0e-12);
            assert!((point[1] - shoulder[1]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn conic_passes_through_authored_vertex_at_half_parameter() {
        for rho in [0.25, 0.5, 0.75] {
            let conic = conic_candidate([-2.0, 0.0], [2.0, 0.0], [0.0, 2.0], rho).unwrap();
            let point = conic.curve.point_at(0.5);
            assert!((point[0]).abs() < 1.0e-12);
            assert!((point[1] - 2.0).abs() < 1.0e-12);
        }
    }
}
