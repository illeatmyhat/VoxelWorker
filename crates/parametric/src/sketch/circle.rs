//! Continuous construction geometry for point-defined circles.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircleCandidate {
    /// Solved center.
    pub center: [f64; 2],
    /// Positive radius.
    pub radius: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircleCandidateError {
    /// At least one coordinate is not finite.
    NonFinite,
    /// The diameter endpoints coincide.
    CoincidentDiameter,
    /// Three circumference points are coincident or collinear.
    DegenerateCircumcircle,
}

/// Construct the circle whose diameter is `first → second`.
///
/// # Errors
///
/// Refuses non-finite coordinates and coincident endpoints.
pub fn two_point_circle_candidate(
    first: [f64; 2],
    second: [f64; 2],
) -> Result<CircleCandidate, CircleCandidateError> {
    finite_points([first, second])?;
    let span = [second[0] - first[0], second[1] - first[1]];
    let diameter = span[0].hypot(span[1]);
    if diameter <= f64::EPSILON {
        return Err(CircleCandidateError::CoincidentDiameter);
    }
    Ok(CircleCandidate {
        center: [first[0] + span[0] / 2.0, first[1] + span[1] / 2.0],
        radius: diameter / 2.0,
    })
}

/// Construct the unique circle through three circumference points.
///
/// # Errors
///
/// Refuses non-finite coordinates and triples with no finite unique circumcircle.
pub fn three_point_circle_candidate(
    first: [f64; 2],
    second: [f64; 2],
    third: [f64; 2],
) -> Result<CircleCandidate, CircleCandidateError> {
    finite_points([first, second, third])?;
    // Translate to `first` before the determinant. This avoids squaring large absolute document
    // coordinates when only the local triangle controls its circumcircle.
    let b = [second[0] - first[0], second[1] - first[1]];
    let c = [third[0] - first[0], third[1] - first[1]];
    let determinant = 2.0 * b[1].mul_add(-c[0], b[0] * c[1]);
    let scale = b[0]
        .abs()
        .max(b[1].abs())
        .max(c[0].abs())
        .max(c[1].abs())
        .max(1.0);
    if determinant.abs() <= f64::EPSILON * scale * scale * 8.0 {
        return Err(CircleCandidateError::DegenerateCircumcircle);
    }
    let b2 = b[1].mul_add(b[1], b[0] * b[0]);
    let c2 = c[1].mul_add(c[1], c[0] * c[0]);
    let relative_center = [
        b[1].mul_add(-c2, c[1] * b2) / determinant,
        c[0].mul_add(-b2, b[0] * c2) / determinant,
    ];
    let center = [first[0] + relative_center[0], first[1] + relative_center[1]];
    let radius = relative_center[0].hypot(relative_center[1]);
    if !center.into_iter().chain([radius]).all(f64::is_finite) || radius <= f64::EPSILON {
        return Err(CircleCandidateError::DegenerateCircumcircle);
    }
    Ok(CircleCandidate { center, radius })
}

fn finite_points<const N: usize>(points: [[f64; 2]; N]) -> Result<(), CircleCandidateError> {
    points
        .into_iter()
        .flatten()
        .all(f64::is_finite)
        .then_some(())
        .ok_or(CircleCandidateError::NonFinite)
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn two_points_define_the_diameter() {
        let candidate = two_point_circle_candidate([-2.0, 1.0], [4.0, 1.0]).unwrap();
        assert_eq!(candidate.center, [1.0, 1.0]);
        assert_eq!(candidate.radius, 3.0);
    }

    #[test]
    fn three_points_define_the_circumcircle_independent_of_order() {
        for points in [
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            [[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ] {
            let candidate = three_point_circle_candidate(points[0], points[1], points[2]).unwrap();
            assert!(candidate.center[0].abs() < 1e-12);
            assert!(candidate.center[1].abs() < 1e-12);
            assert!((candidate.radius - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn degenerate_point_sets_are_refused() {
        assert_eq!(
            two_point_circle_candidate([0.0, 0.0], [0.0, 0.0]),
            Err(CircleCandidateError::CoincidentDiameter)
        );
        assert_eq!(
            three_point_circle_candidate([0.0, 0.0], [1.0, 0.0], [2.0, 0.0]),
            Err(CircleCandidateError::DegenerateCircumcircle)
        );
    }
}
