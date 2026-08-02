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

/// The rho that makes the conic through `from`/`to`/`vertex` also aim at `apex`.
///
/// The three picks fix the curve's endpoints and one point ON it; rho is the remaining freedom,
/// and on its own it is a bare number with nothing on screen to grab. This is what gives it a
/// handle: the APEX is where the two end tangents meet, the vertex always sits on the segment from
/// the chord midpoint to it, and rho is exactly how far along that segment it sits —
/// `vertex = midpoint + rho * (apex - midpoint)`. So pointing at an apex names a rho, and pulling
/// the apex away from the chord sharpens the curve.
///
/// The cursor is projected onto the midpoint→vertex ray, because only that direction can carry an
/// apex consistent with the vertex already picked. `None` when the projection falls at or behind
/// the vertex, where no rho in the open interval `(0, 1)` answers.
#[must_use]
pub fn conic_rho_from_apex(
    from: [f64; 2],
    to: [f64; 2],
    vertex: [f64; 2],
    apex: [f64; 2],
) -> Option<f64> {
    if ![from, to, vertex, apex]
        .into_iter()
        .flatten()
        .all(f64::is_finite)
    {
        return None;
    }
    let midpoint = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
    let toward_vertex = [vertex[0] - midpoint[0], vertex[1] - midpoint[1]];
    let reach = toward_vertex[0].hypot(toward_vertex[1]);
    if reach <= f64::EPSILON {
        return None;
    }
    let unit = [toward_vertex[0] / reach, toward_vertex[1] / reach];
    let along = (apex[0] - midpoint[0]).mul_add(unit[0], (apex[1] - midpoint[1]) * unit[1]);
    // The apex must lie strictly beyond the vertex, or the vertex is not between the two.
    (along > reach).then(|| reach / along)
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
    fn conic_passes_through_authored_vertex_at_half_parameter() {
        for rho in [0.25, 0.5, 0.75] {
            let conic = conic_candidate([-2.0, 0.0], [2.0, 0.0], [0.0, 2.0], rho).unwrap();
            let point = conic.curve.point_at(0.5);
            assert!((point[0]).abs() < 1.0e-12);
            assert!((point[1] - 2.0).abs() < 1.0e-12);
        }
    }
}
