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

/// One endpoint/control/rho conic, exactly represented as a rational cubic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConicCandidate {
    pub from: [f64; 2],
    pub to: [f64; 2],
    /// Where the end tangents meet — the authored third freedom.
    pub control: [f64; 2],
    /// The on-curve point at `t = 0.5`, DERIVED from the control point and rho.
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

/// The rho a conic starts at before its shoulder is dragged.
///
/// A parabola: the exact boundary between the elliptic and hyperbolic halves of the family, and so
/// the reading that presumes least about which half the author is reaching for.
pub const CONIC_PARABOLIC_RHO: f64 = 0.5;

/// How close to degenerate a dragged shoulder is allowed to get.
///
/// Rho lives on the OPEN interval `(0, 1)`, which has no endpoints to clamp a drag against. Both
/// ends are curves nobody means to author — at 0 the conic is its own chord, at 1 it is a corner
/// at the control point — so the gizmo stops just short of each.
const SHOULDER_RHO_MARGIN: f64 = 1.0e-3;

/// The segment a conic's shoulder gizmo slides along: chord midpoint to control point.
///
/// Every conic through `from` and `to` with its tangents meeting at `apex` puts its on-curve
/// shoulder somewhere on this segment, and rho is exactly how far along it sits. That makes the
/// segment the whole authoring space of the last step, with nothing outside it to refuse. One
/// definition so the drawn track, the rho the cursor names, and the committed vertex agree.
///
/// `None` when the control point falls on the chord midpoint, where the track has no length and no
/// conic exists to shape.
#[must_use]
pub fn conic_shoulder_track(from: [f64; 2], to: [f64; 2], apex: [f64; 2]) -> Option<[[f64; 2]; 2]> {
    if ![from, to, apex].into_iter().flatten().all(f64::is_finite) {
        return None;
    }
    let midpoint = [(from[0] + to[0]) * 0.5, (from[1] + to[1]) * 0.5];
    let reach = (apex[0] - midpoint[0]).hypot(apex[1] - midpoint[1]);
    (reach > f64::EPSILON).then_some([midpoint, apex])
}

/// The rho a cursor names when it drags the shoulder along [`conic_shoulder_track`].
///
/// The cursor is projected onto the track and clamped to it, because the gizmo is captive: the
/// author is choosing how far along a known segment to sit, not pointing at a free position that
/// might miss. `None` only when there is no track at all.
#[must_use]
pub fn conic_rho_from_shoulder(
    from: [f64; 2],
    to: [f64; 2],
    apex: [f64; 2],
    shoulder: [f64; 2],
) -> Option<f64> {
    if !shoulder.iter().copied().all(f64::is_finite) {
        return None;
    }
    let [midpoint, apex] = conic_shoulder_track(from, to, apex)?;
    let track = [apex[0] - midpoint[0], apex[1] - midpoint[1]];
    let length = track[0].hypot(track[1]);
    let unit = [track[0] / length, track[1] / length];
    let along = (shoulder[0] - midpoint[0]).mul_add(unit[0], (shoulder[1] - midpoint[1]) * unit[1]);
    Some((along / length).clamp(SHOULDER_RHO_MARGIN, 1.0 - SHOULDER_RHO_MARGIN))
}

/// Where a given rho puts the on-curve shoulder — `midpoint + rho * (apex - midpoint)`.
///
/// The inverse of [`conic_rho_from_shoulder`], and what turns the gizmo's position back into the
/// vertex [`conic_candidate`] is authored from.
#[must_use]
pub fn conic_vertex_from_rho(
    from: [f64; 2],
    to: [f64; 2],
    apex: [f64; 2],
    rho: f64,
) -> Option<[f64; 2]> {
    if !rho.is_finite() {
        return None;
    }
    let [midpoint, apex] = conic_shoulder_track(from, to, apex)?;
    Some([
        rho.mul_add(apex[0] - midpoint[0], midpoint[0]),
        rho.mul_add(apex[1] - midpoint[1], midpoint[1]),
    ])
}

/// Build a conic from the two points it runs between, the CONTROL point its end tangents meet at,
/// and `rho`.
///
/// The control point is the conic's canonical third freedom — the rational quadratic's middle
/// weighted control point, which is also the handle an author grabs. The on-curve shoulder at
/// `t = 0.5` is derived and reported alongside it; it is a consequence of the control point and
/// rho, not an independent input.
///
/// `rho = 0.5` is parabolic; values below are elliptic and values above are hyperbolic. The open
/// interval keeps the equivalent rational-quadratic weight positive and finite.
///
/// # Errors
///
/// Returns a typed error for non-finite input, coincident endpoints, rho outside `(0, 1)`, or a
/// control point on the chord, where there is no curve to build.
pub fn conic_candidate(
    from: [f64; 2],
    to: [f64; 2],
    control: [f64; 2],
    rho: f64,
) -> Result<ConicCandidate, ConicCandidateError> {
    if ![from, to, control]
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
    let vertex = conic_vertex_from_rho(from, to, control, rho)
        .ok_or(ConicCandidateError::CollapsedVertex)?;
    let weight = rho / (1.0 - rho);
    let curve = RationalBezier::elevated_quadratic([from, control, to], [1.0, weight, 1.0]);
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
        control,
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
    fn the_shoulder_track_runs_from_the_chord_midpoint_to_the_control_point() {
        let [midpoint, apex] = conic_shoulder_track([-2.0, 0.0], [2.0, 0.0], [0.0, 6.0]).unwrap();
        assert_eq!(midpoint, [0.0, 0.0]);
        assert_eq!(apex, [0.0, 6.0]);
        assert!(conic_shoulder_track([-2.0, 0.0], [2.0, 0.0], [0.0, 0.0]).is_none());
    }

    #[test]
    fn a_shoulder_drag_is_captive_to_its_track() {
        let track = |shoulder| {
            conic_rho_from_shoulder([-2.0, 0.0], [2.0, 0.0], [0.0, 4.0], shoulder).unwrap()
        };
        // Halfway up names the parabola, and sideways drift off the track does not change that:
        // only the distance ALONG it is the parameter.
        assert!((track([0.0, 2.0]) - 0.5).abs() < 1.0e-12);
        assert!((track([9.0, 2.0]) - 0.5).abs() < 1.0e-12);
        // Past either end the gizmo stops rather than leaving the open interval.
        assert!(track([0.0, -50.0]) > 0.0);
        assert!(track([0.0, 50.0]) < 1.0);
    }

    #[test]
    fn the_shoulder_a_rho_names_lies_on_the_curve_that_rho_builds() {
        let (from, to, apex) = ([-2.0, 0.0], [2.0, 0.0], [0.0, 4.0]);
        for rho in [0.25, 0.5, 0.75] {
            let vertex = conic_vertex_from_rho(from, to, apex, rho).unwrap();
            let recovered = conic_rho_from_shoulder(from, to, apex, vertex).unwrap();
            assert!((recovered - rho).abs() < 1.0e-12);
            let conic = conic_candidate(from, to, apex, rho).unwrap();
            assert!((conic.vertex[0] - vertex[0]).abs() < 1.0e-12);
            assert!((conic.vertex[1] - vertex[1]).abs() < 1.0e-12);
            let point = conic.curve.point_at(0.5);
            assert!((point[0] - vertex[0]).abs() < 1.0e-12);
            assert!((point[1] - vertex[1]).abs() < 1.0e-12);
        }
    }

    /// The control point is authored; the shoulder is what the curve does about it. Moving the
    /// control point alone — rho untouched — pulls the curve after it, which is what makes it a
    /// handle rather than a vertex.
    #[test]
    fn the_curve_follows_its_control_point_and_meets_the_derived_shoulder() {
        for rho in [0.25, 0.5, 0.75] {
            let conic = conic_candidate([-2.0, 0.0], [2.0, 0.0], [0.0, 4.0], rho).unwrap();
            let point = conic.curve.point_at(0.5);
            assert!(point[0].abs() < 1.0e-12);
            let shoulder_height = 4.0 * rho;
            assert!(
                (point[1] - shoulder_height).abs() < 1.0e-12,
                "{point:?} at {rho}"
            );
            assert_eq!(conic.control, [0.0, 4.0]);

            let pulled = conic_candidate([-2.0, 0.0], [2.0, 0.0], [1.0, 8.0], rho).unwrap();
            assert!(pulled.curve.point_at(0.5)[1] > point[1]);
        }
    }
}
