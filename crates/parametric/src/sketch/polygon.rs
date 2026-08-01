//! Continuous construction geometry for regular polygons.

use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, PartialEq)]
pub struct PolygonCandidate {
    /// Boundary vertices in traversal order.
    pub vertices: Vec<[f64; 2]>,
    /// Geometric center.
    pub center: [f64; 2],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolygonCandidateError {
    /// At least one coordinate is not finite.
    NonFinite,
    /// Fewer than three sides were requested.
    TooFewSides,
    /// More than the supported 128 sides were requested.
    TooManySides,
    /// The defining radius or edge has zero length.
    CollapsedSize,
    /// Edge Polygon's orientation pick lies on its defining edge.
    UndefinedSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenteredPolygonKind {
    /// Vertices lie on the authored radius.
    Inscribed,
    /// Edge midpoints lie on the authored radius.
    Circumscribed,
}

/// Construct a centered regular polygon from a radius/orientation pick.
///
/// # Errors
///
/// Refuses non-finite or collapsed geometry and side counts outside `3..=128`.
pub fn centered_polygon_candidate(
    kind: CenteredPolygonKind,
    center: [f64; 2],
    radius_point: [f64; 2],
    sides: u16,
) -> Result<PolygonCandidate, PolygonCandidateError> {
    validate_sides(sides)?;
    finite([center, radius_point])?;
    let offset = [radius_point[0] - center[0], radius_point[1] - center[1]];
    let authored_radius = offset[0].hypot(offset[1]);
    if authored_radius <= f64::EPSILON {
        return Err(PolygonCandidateError::CollapsedSize);
    }
    let half_step = PI / f64::from(sides);
    let circumradius = match kind {
        CenteredPolygonKind::Inscribed => authored_radius,
        CenteredPolygonKind::Circumscribed => authored_radius / half_step.cos(),
    };
    let bearing = offset[1].atan2(offset[0]);
    let start = match kind {
        CenteredPolygonKind::Inscribed => bearing,
        CenteredPolygonKind::Circumscribed => bearing - half_step,
    };
    Ok(PolygonCandidate {
        vertices: regular_vertices(center, circumradius, start, TAU / f64::from(sides), sides),
        center,
    })
}

/// Construct a regular polygon from one edge and a pick selecting which side contains its body.
///
/// # Errors
///
/// Refuses non-finite/collapsed geometry, side counts outside `3..=128`, and a side pick on the
/// edge.
pub fn edge_polygon_candidate(
    first: [f64; 2],
    second: [f64; 2],
    side_point: [f64; 2],
    sides: u16,
) -> Result<PolygonCandidate, PolygonCandidateError> {
    validate_sides(sides)?;
    finite([first, second, side_point])?;
    let edge = [second[0] - first[0], second[1] - first[1]];
    let length = edge[0].hypot(edge[1]);
    if length <= f64::EPSILON {
        return Err(PolygonCandidateError::CollapsedSize);
    }
    let side_offset = [side_point[0] - first[0], side_point[1] - first[1]];
    let cross = edge[0].mul_add(side_offset[1], -edge[1] * side_offset[0]);
    if cross.abs() <= f64::EPSILON * length.max(1.0) {
        return Err(PolygonCandidateError::UndefinedSide);
    }
    let sign = cross.signum();
    let normal = [-edge[1] / length * sign, edge[0] / length * sign];
    let apothem = length / (2.0 * (PI / f64::from(sides)).tan());
    let midpoint = [first[0].midpoint(second[0]), first[1].midpoint(second[1])];
    let center = [
        normal[0].mul_add(apothem, midpoint[0]),
        normal[1].mul_add(apothem, midpoint[1]),
    ];
    let start = (first[1] - center[1]).atan2(first[0] - center[0]);
    let mut vertices = regular_vertices(
        center,
        (first[0] - center[0]).hypot(first[1] - center[1]),
        start,
        sign * TAU / f64::from(sides),
        sides,
    );
    // These are authored topology, not merely points on the ideal construction circle. Pin them
    // exactly so the document adapter can reuse existing endpoints and constraints do not inherit
    // trigonometric roundoff.
    if let [canonical_first, canonical_second, ..] = vertices.as_mut_slice() {
        *canonical_first = first;
        *canonical_second = second;
    }
    Ok(PolygonCandidate { vertices, center })
}

fn regular_vertices(
    center: [f64; 2],
    radius: f64,
    start: f64,
    step: f64,
    sides: u16,
) -> Vec<[f64; 2]> {
    (0..sides)
        .map(|index| {
            let angle = step.mul_add(f64::from(index), start);
            [
                radius.mul_add(angle.cos(), center[0]),
                radius.mul_add(angle.sin(), center[1]),
            ]
        })
        .collect()
}

const fn validate_sides(sides: u16) -> Result<(), PolygonCandidateError> {
    match sides {
        0..=2 => Err(PolygonCandidateError::TooFewSides),
        3..=128 => Ok(()),
        129..=u16::MAX => Err(PolygonCandidateError::TooManySides),
    }
}

fn finite<const N: usize>(points: [[f64; 2]; N]) -> Result<(), PolygonCandidateError> {
    points
        .into_iter()
        .flatten()
        .all(f64::is_finite)
        .then_some(())
        .ok_or(PolygonCandidateError::NonFinite)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::manual_midpoint,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    #[test]
    fn centered_variants_put_the_authored_radius_on_the_promised_locus() {
        let inscribed =
            centered_polygon_candidate(CenteredPolygonKind::Inscribed, [0.0, 0.0], [2.0, 0.0], 4)
                .unwrap();
        assert_eq!(inscribed.vertices[0], [2.0, 0.0]);
        let circumscribed = centered_polygon_candidate(
            CenteredPolygonKind::Circumscribed,
            [0.0, 0.0],
            [2.0, 0.0],
            4,
        )
        .unwrap();
        let midpoint = [
            (circumscribed.vertices[0][0] + circumscribed.vertices[1][0]) / 2.0,
            (circumscribed.vertices[0][1] + circumscribed.vertices[1][1]) / 2.0,
        ];
        assert!((midpoint[0] - 2.0).abs() < 1e-12 && midpoint[1].abs() < 1e-12);
    }

    #[test]
    fn edge_polygon_keeps_the_defining_edge_and_obeys_side_pick() {
        let candidate = edge_polygon_candidate([0.0, 0.0], [2.0, 0.0], [0.0, 1.0], 4).unwrap();
        assert!((candidate.vertices[0][0]).abs() < 1e-12);
        assert!((candidate.vertices[1][0] - 2.0).abs() < 1e-12);
        assert!(candidate.center[1] > 0.0);
    }
}
