//! Geometry for center-first circular-arc authoring.

use std::f64::consts::TAU;

/// Which way round an arc runs from its start point.
///
/// A cursor position cannot answer this on its own — the same point on the circle is reachable
/// either way round — so the caller that watched the cursor get there has to say. See
/// [`substrate::winding::TurnLatch`], which is how an interactive caller works it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcTurn {
    CounterClockwise,
    Clockwise,
}

/// The canonical geometry produced by a center, a start point, and an end direction.
///
/// The direction point is projected onto the start radius. `sweep_radians` is signed by the
/// requested [`ArcTurn`]: positive counter-clockwise, negative clockwise. Its MAGNITUDE comes from
/// where the end direction actually landed, so a pointer that has swung past a half turn gets the
/// long way round without the caller having to say how far.
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
    turn: ArcTurn,
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
    let counter_clockwise = (end_angle - start_angle).rem_euclid(TAU);
    if counter_clockwise <= f64::EPSILON * 8.0 || TAU - counter_clockwise <= f64::EPSILON * 8.0 {
        return Err(CenterArcCandidateError::CollapsedSweep);
    }
    // The same endpoint is the short way round one direction and the long way round the other, so
    // the turn decides which of the two arcs through it the caller meant.
    let sweep_radians = match turn {
        ArcTurn::CounterClockwise => counter_clockwise,
        ArcTurn::Clockwise => counter_clockwise - TAU,
    };

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
        let candidate = center_arc_candidate(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 7.0],
            ArcTurn::CounterClockwise,
        )
        .unwrap();
        assert_eq!(candidate.endpoint, [0.0, 2.0]);
        assert_eq!(candidate.radius, 2.0);
        assert!((candidate.sweep_radians.to_degrees() - 90.0).abs() < 1e-12);

        let major = center_arc_candidate(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, -1.0],
            ArcTurn::CounterClockwise,
        )
        .unwrap();
        assert!((major.sweep_radians.to_degrees() - 270.0).abs() < 1e-12);
    }

    /// The same endpoint, reached the other way round, is the complementary arc — and the sign is
    /// what says so.
    #[test]
    fn the_turn_picks_which_of_the_two_arcs_through_the_endpoint_is_meant() {
        let quarter = center_arc_candidate(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 7.0],
            ArcTurn::CounterClockwise,
        )
        .unwrap();
        let three_quarters =
            center_arc_candidate([0.0, 0.0], [2.0, 0.0], [0.0, 7.0], ArcTurn::Clockwise).unwrap();
        assert_eq!(quarter.endpoint, three_quarters.endpoint);
        assert!((quarter.sweep_radians.to_degrees() - 90.0).abs() < 1e-12);
        assert!((three_quarters.sweep_radians.to_degrees() + 270.0).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_distinct_refusals() {
        let turn = ArcTurn::CounterClockwise;
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [0.0, 0.0], [1.0, 0.0], turn),
            Err(CenterArcCandidateError::CollapsedRadius)
        );
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [1.0, 0.0], [0.0, 0.0], turn),
            Err(CenterArcCandidateError::UndefinedEndDirection)
        );
        assert_eq!(
            center_arc_candidate([0.0, 0.0], [1.0, 0.0], [2.0, 0.0], turn),
            Err(CenterArcCandidateError::CollapsedSweep)
        );
    }
}
