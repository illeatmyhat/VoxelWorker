//! Validated continuous construction for a segment named by its midpoint and one endpoint.
//!
//! This is raw planar mathematics: it knows nothing about document ids or the split integer/f32
//! storage of a sketch point. The document adapter canonicalizes the returned coordinates before
//! it promises that preview geometry can be persisted.

/// The raw continuous endpoints of a midpoint-defined segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidpointLineCandidate {
    /// The endpoint supplied by the caller.
    pub endpoint: [f64; 2],
    /// The other endpoint, reflected through the supplied midpoint.
    pub reflected: [f64; 2],
}

/// Why raw midpoint-line construction cannot produce a finite nonzero segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidpointLineCandidateError {
    /// A midpoint or endpoint coordinate is NaN or infinite.
    NonFinite,
    /// Finite inputs overflowed while subtracting or adding the reflected displacement.
    Overflow,
    /// The supplied endpoint is exactly the midpoint.
    Collapsed,
}

/// Reflect `endpoint` through `midpoint`, producing the unique segment whose midpoint is the
/// supplied point. Collapse is exact: arbitrarily short finite `NoSnap` segments remain legal.
///
/// # Errors
///
/// Returns [`MidpointLineCandidateError::NonFinite`] for a nonfinite input,
/// [`MidpointLineCandidateError::Overflow`] when reflection arithmetic overflows, and
/// [`MidpointLineCandidateError::Collapsed`] when the endpoint is exactly the midpoint.
#[allow(clippy::float_cmp)]
pub fn midpoint_line_candidate(
    midpoint: [f64; 2],
    endpoint: [f64; 2],
) -> Result<MidpointLineCandidate, MidpointLineCandidateError> {
    if !midpoint.into_iter().chain(endpoint).all(f64::is_finite) {
        return Err(MidpointLineCandidateError::NonFinite);
    }
    if midpoint == endpoint {
        return Err(MidpointLineCandidateError::Collapsed);
    }

    // `midpoint + (midpoint - endpoint)` delays overflow that the eager
    // `2 * midpoint - endpoint` form would introduce even when the reflected value is finite.
    let delta = [midpoint[0] - endpoint[0], midpoint[1] - endpoint[1]];
    if !delta.into_iter().all(f64::is_finite) {
        return Err(MidpointLineCandidateError::Overflow);
    }
    let reflected = [midpoint[0] + delta[0], midpoint[1] + delta[1]];
    if !reflected.into_iter().all(f64::is_finite) {
        return Err(MidpointLineCandidateError::Overflow);
    }

    Ok(MidpointLineCandidate {
        endpoint,
        reflected,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn reflects_arbitrary_subvoxel_coordinates_with_stable_order() {
        let candidate = midpoint_line_candidate([1.25, -2.5], [-3.75, 4.125]).unwrap();
        assert_eq!(candidate.endpoint, [-3.75, 4.125]);
        assert_eq!(candidate.reflected, [6.25, -9.125]);

        let reversed = midpoint_line_candidate([-4.5, 3.25], [2.0, -8.75]).unwrap();
        assert_eq!(reversed.endpoint, [2.0, -8.75]);
        assert_eq!(reversed.reflected, [-11.0, 15.25]);
    }

    #[test]
    fn collapse_is_exact_not_tolerance_based() {
        assert_eq!(
            midpoint_line_candidate([1.0, 2.0], [1.0, 2.0]),
            Err(MidpointLineCandidateError::Collapsed)
        );
        let tiny = f64::from_bits(1.0f64.to_bits() + 1);
        assert!(midpoint_line_candidate([1.0, 2.0], [tiny, 2.0]).is_ok());
    }

    #[test]
    fn rejects_nonfinite_inputs_separately_from_arithmetic_overflow() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                midpoint_line_candidate([bad, 0.0], [0.0, 1.0]),
                Err(MidpointLineCandidateError::NonFinite)
            );
            assert_eq!(
                midpoint_line_candidate([0.0, 1.0], [bad, 0.0]),
                Err(MidpointLineCandidateError::NonFinite)
            );
        }
    }

    #[test]
    fn reports_subtraction_and_final_addition_overflow() {
        assert_eq!(
            midpoint_line_candidate([f64::MAX, 0.0], [-f64::MAX, 1.0]),
            Err(MidpointLineCandidateError::Overflow)
        );
        assert_eq!(
            midpoint_line_candidate([f64::MAX, 0.0], [f64::MAX / 2.0, 1.0]),
            Err(MidpointLineCandidateError::Overflow)
        );
    }

    #[test]
    fn accepts_large_finite_reflection_when_the_result_is_finite() {
        let candidate = midpoint_line_candidate([f64::MAX / 4.0, 0.0], [0.0, 1.0]).unwrap();
        assert_eq!(candidate.reflected, [f64::MAX / 2.0, -1.0]);
    }
}
