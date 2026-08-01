//! Continuous construction geometry for point-defined rectangles.

/// Four corners in boundary order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectangleCandidate {
    /// Boundary-ordered corners; the closing edge runs from index 3 back to index 0.
    pub corners: [[f64; 2]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RectangleCandidateError {
    /// At least one coordinate is not finite.
    NonFinite,
    /// A center rectangle has zero span on at least one axis.
    CollapsedSpan,
    /// A three-point rectangle's first two corners coincide.
    CollapsedBase,
    /// Its third pick lies on the base line.
    CollapsedWidth,
}

/// Construct an axis-aligned rectangle from its center and one corner.
///
/// # Errors
///
/// Refuses non-finite inputs or a corner sharing either axis coordinate with the center.
pub fn center_rectangle_candidate(
    center: [f64; 2],
    corner: [f64; 2],
) -> Result<RectangleCandidate, RectangleCandidateError> {
    finite([center, corner])?;
    let offset = [corner[0] - center[0], corner[1] - center[1]];
    if offset[0].abs() <= f64::EPSILON || offset[1].abs() <= f64::EPSILON {
        return Err(RectangleCandidateError::CollapsedSpan);
    }
    Ok(RectangleCandidate {
        corners: [
            [center[0] - offset[0], center[1] - offset[1]],
            [center[0] + offset[0], center[1] - offset[1]],
            [center[0] + offset[0], center[1] + offset[1]],
            [center[0] - offset[0], center[1] + offset[1]],
        ],
    })
}

/// Construct an oriented rectangle. `first → second` is its base; `width_point` supplies signed
/// perpendicular width while its along-base component is deliberately ignored.
///
/// # Errors
///
/// Refuses non-finite input, a collapsed base, or zero perpendicular width.
pub fn three_point_rectangle_candidate(
    first: [f64; 2],
    second: [f64; 2],
    width_point: [f64; 2],
) -> Result<RectangleCandidate, RectangleCandidateError> {
    finite([first, second, width_point])?;
    let base = [second[0] - first[0], second[1] - first[1]];
    let length = base[0].hypot(base[1]);
    if length <= f64::EPSILON {
        return Err(RectangleCandidateError::CollapsedBase);
    }
    let normal = [-base[1] / length, base[0] / length];
    let cursor = [width_point[0] - first[0], width_point[1] - first[1]];
    let width = cursor[0].mul_add(normal[0], cursor[1] * normal[1]);
    if width.abs() <= f64::EPSILON * length.max(1.0) {
        return Err(RectangleCandidateError::CollapsedWidth);
    }
    let offset = [normal[0] * width, normal[1] * width];
    Ok(RectangleCandidate {
        corners: [
            first,
            second,
            [second[0] + offset[0], second[1] + offset[1]],
            [first[0] + offset[0], first[1] + offset[1]],
        ],
    })
}

fn finite<const N: usize>(points: [[f64; 2]; N]) -> Result<(), RectangleCandidateError> {
    points
        .into_iter()
        .flatten()
        .all(f64::is_finite)
        .then_some(())
        .ok_or(RectangleCandidateError::NonFinite)
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn center_rectangle_reflects_the_picked_corner() {
        let candidate = center_rectangle_candidate([2.0, 3.0], [5.0, 7.0]).unwrap();
        assert_eq!(
            candidate.corners,
            [[-1.0, -1.0], [5.0, -1.0], [5.0, 7.0], [-1.0, 7.0]]
        );
    }

    #[test]
    fn three_point_rectangle_projects_width_perpendicular_to_its_base() {
        let candidate =
            three_point_rectangle_candidate([0.0, 0.0], [3.0, 4.0], [1.0, 4.25]).unwrap();
        let base: [f64; 2] = [3.0, 4.0];
        let side = [
            candidate.corners[3][0] - candidate.corners[0][0],
            candidate.corners[3][1] - candidate.corners[0][1],
        ];
        assert!(base[1].mul_add(side[1], base[0] * side[0]) < 1e-12);
        assert_eq!(candidate.corners[0], [0.0, 0.0]);
        assert_eq!(candidate.corners[1], [3.0, 4.0]);
    }
}
