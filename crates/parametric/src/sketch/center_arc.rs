//! Geometry for center-first circular-arc authoring.

use std::f64::consts::TAU;

/// The canonical geometry produced by a center, a start point, and an end direction.
///
/// The direction point is projected onto the start radius. The result always walks
/// counter-clockwise from `start` to `endpoint`, matching center-point arc authoring rather than
/// encoding pointer distance as a second radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CenterArcCandidate {
    pub center: [f64; 2],
    pub start: [f64; 2],
    pub endpoint: [f64; 2],
    pub radius: f64,
    pub sweep_radians: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterArcCandidateError {
    NonFinite,
    CollapsedRadius,
    UndefinedEndDirection,
    CollapsedSweep,
}

/// Construct a center-first arc without document identity or storage concerns.
///
/// # Errors
///
/// Refuses non-finite coordinates, a collapsed start radius, an undefined end direction, or a
/// direction that would collapse the open arc to a zero/full-turn span.
pub fn center_arc_candidate(
    center: [f64; 2],
    start: [f64; 2],
    end_direction: [f64; 2],
) -> Result<CenterArcCandidate, CenterArcCandidateError> {
    if center
        .into_iter()
        .chain(start)
        .chain(end_direction)
        .any(|coordinate| !coordinate.is_finite())
    {
        return Err(CenterArcCandidateError::NonFinite);
    }

    let start_offset = [start[0] - center[0], start[1] - center[1]];
    let radius = start_offset[0].hypot(start_offset[1]);
    if radius <= f64::EPSILON {
        return Err(CenterArcCandidateError::CollapsedRadius);
    }

    let end_offset = [end_direction[0] - center[0], end_direction[1] - center[1]];
    let end_distance = end_offset[0].hypot(end_offset[1]);
    if end_distance <= f64::EPSILON {
        return Err(CenterArcCandidateError::UndefinedEndDirection);
    }
    let endpoint = [
        center[0] + radius * end_offset[0] / end_distance,
        center[1] + radius * end_offset[1] / end_distance,
    ];
    let start_angle = start_offset[1].atan2(start_offset[0]);
    let end_angle = end_offset[1].atan2(end_offset[0]);
    let sweep_radians = (end_angle - start_angle).rem_euclid(TAU);
    if sweep_radians <= f64::EPSILON * 8.0 || TAU - sweep_radians <= f64::EPSILON * 8.0 {
        return Err(CenterArcCandidateError::CollapsedSweep);
    }

    Ok(CenterArcCandidate {
        center,
        start,
        endpoint,
        radius,
        sweep_radians,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_projected_to_the_start_radius_and_sweep_is_counter_clockwise() {
        let candidate = center_arc_candidate([0.0, 0.0], [2.0, 0.0], [0.0, 7.0]).unwrap();
        assert_eq!(candidate.endpoint, [0.0, 2.0]);
        assert_eq!(candidate.radius, 2.0);
        assert!((candidate.sweep_radians.to_degrees() - 90.0).abs() < 1e-12);

        let major = center_arc_candidate([0.0, 0.0], [2.0, 0.0], [0.0, -1.0]).unwrap();
        assert!((major.sweep_radians.to_degrees() - 270.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_distinct_refusals() {
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [0.0, 0.0], [1.0, 0.0]),
            Err(CenterArcCandidateError::CollapsedRadius)
        );
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [1.0, 0.0], [0.0, 0.0]),
            Err(CenterArcCandidateError::UndefinedEndDirection)
        );
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [1.0, 0.0], [2.0, 0.0]),
            Err(CenterArcCandidateError::CollapsedSweep)
        );
    }
}
