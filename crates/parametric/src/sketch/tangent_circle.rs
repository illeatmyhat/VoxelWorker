//! Candidate geometry for circles tangent to two or three finite line selections.

#[derive(Debug, Clone, PartialEq)]
pub struct TangentCircleCandidate {
    pub center: [f64; 2],
    pub radius: f64,
    /// One contact point per input line, in the same order.
    pub contacts: Vec<[f64; 2]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TangentCircleCandidateError {
    NonFinite,
    DegenerateLine,
    CollapsedRadius,
    NoFiniteSolution,
    ContactOutsideSegment,
}

#[derive(Debug, Clone, Copy)]
struct NormalLine {
    normal: [f64; 2],
    offset: f64,
    from: [f64; 2],
    to: [f64; 2],
}

/// Construct the radius-selected circle tangent to two lines and nearest the placement witness.
///
/// The radius is the witness's nearest perpendicular distance to either line. Of the four signed
/// offset intersections, the center nearest that witness wins.
///
/// # Errors
///
/// Refuses non-finite/degenerate lines, a collapsed radius, or a candidate whose tangent contact
/// lies outside either finite selected segment. Parallel lines produce the circle on their
/// midline nearest the witness; their separation determines its radius.
pub fn two_tangent_circle_candidate(
    lines: [([f64; 2], [f64; 2]); 2],
    witness: [f64; 2],
) -> Result<TangentCircleCandidate, TangentCircleCandidateError> {
    finite(
        lines
            .into_iter()
            .flat_map(|(from, to)| <[[f64; 2]; 2]>::from((from, to)))
            .chain([witness]),
    )?;
    let lines = [
        normal_line(lines[0].0, lines[0].1)?,
        normal_line(lines[1].0, lines[1].1)?,
    ];
    if parallel(lines[0], lines[1]) {
        return parallel_two_tangent_candidate(lines, witness);
    }
    let radius = lines
        .iter()
        .map(|line| signed_distance(*line, witness).abs())
        .fold(f64::INFINITY, f64::min);
    if radius <= f64::EPSILON {
        return Err(TangentCircleCandidateError::CollapsedRadius);
    }
    let mut best: Option<(f64, TangentCircleCandidate)> = None;
    for first_sign in [-1.0_f64, 1.0] {
        for second_sign in [-1.0_f64, 1.0] {
            let rhs = [
                first_sign.mul_add(radius, lines[0].offset),
                second_sign.mul_add(radius, lines[1].offset),
            ];
            let Some(center) = solve_2x2(lines[0].normal, lines[1].normal, rhs) else {
                continue;
            };
            let Ok(candidate) = candidate_from(center, radius, &lines) else {
                continue;
            };
            let score = squared_distance(center, witness);
            if best
                .as_ref()
                .is_none_or(|(best_score, _)| score < *best_score)
            {
                best = Some((score, candidate));
            }
        }
    }
    best.map(|(_, candidate)| candidate)
        .ok_or(TangentCircleCandidateError::NoFiniteSolution)
}

fn parallel_two_tangent_candidate(
    lines: [NormalLine; 2],
    witness: [f64; 2],
) -> Result<TangentCircleCandidate, TangentCircleCandidateError> {
    let first_offset = lines[0].offset;
    let second_offset =
        lines[0].normal[0].mul_add(lines[1].from[0], lines[0].normal[1] * lines[1].from[1]);
    let radius = (second_offset - first_offset).abs() / 2.0;
    if radius <= f64::EPSILON {
        return Err(TangentCircleCandidateError::CollapsedRadius);
    }
    let midline_offset = first_offset.midpoint(second_offset);
    let witness_offset = lines[0].normal[0].mul_add(witness[0], lines[0].normal[1] * witness[1]);
    let displacement = witness_offset - midline_offset;
    let center = [
        (-lines[0].normal[0]).mul_add(displacement, witness[0]),
        (-lines[0].normal[1]).mul_add(displacement, witness[1]),
    ];
    candidate_from(center, radius, &lines)
}

/// Construct the circle tangent to three selected lines. Each line's selection locus selects
/// among the incircle/excircle branches; the valid center nearest their centroid wins.
///
/// # Errors
///
/// Refuses non-finite/degenerate line sets, singular triples, or candidates whose tangent contact
/// lies outside a finite selected segment.
pub fn three_tangent_circle_candidate(
    lines: [([f64; 2], [f64; 2], [f64; 2]); 3],
) -> Result<TangentCircleCandidate, TangentCircleCandidateError> {
    finite(
        lines
            .into_iter()
            .flat_map(|(from, to, locus)| <[[f64; 2]; 3]>::from((from, to, locus))),
    )?;
    let geometry = [
        normal_line(lines[0].0, lines[0].1)?,
        normal_line(lines[1].0, lines[1].1)?,
        normal_line(lines[2].0, lines[2].1)?,
    ];
    let witness = [
        (lines[0].2[0] + lines[1].2[0] + lines[2].2[0]) / 3.0,
        (lines[0].2[1] + lines[1].2[1] + lines[2].2[1]) / 3.0,
    ];
    let mut best: Option<(f64, TangentCircleCandidate)> = None;
    for first_sign in [-1.0, 1.0] {
        for second_sign in [-1.0, 1.0] {
            for third_sign in [-1.0, 1.0] {
                let matrix = [
                    [geometry[0].normal[0], geometry[0].normal[1], -first_sign],
                    [geometry[1].normal[0], geometry[1].normal[1], -second_sign],
                    [geometry[2].normal[0], geometry[2].normal[1], -third_sign],
                ];
                let rhs = [geometry[0].offset, geometry[1].offset, geometry[2].offset];
                let Some(solution) = solve_3x3(matrix, rhs) else {
                    continue;
                };
                let radius = solution[2];
                if radius <= f64::EPSILON {
                    continue;
                }
                let center = [solution[0], solution[1]];
                let Ok(candidate) = candidate_from(center, radius, &geometry) else {
                    continue;
                };
                let score = squared_distance(center, witness);
                if best
                    .as_ref()
                    .is_none_or(|(best_score, _)| score < *best_score)
                {
                    best = Some((score, candidate));
                }
            }
        }
    }
    best.map(|(_, candidate)| candidate)
        .ok_or(TangentCircleCandidateError::NoFiniteSolution)
}

fn normal_line(from: [f64; 2], to: [f64; 2]) -> Result<NormalLine, TangentCircleCandidateError> {
    let direction = [to[0] - from[0], to[1] - from[1]];
    let length = direction[0].hypot(direction[1]);
    if length <= f64::EPSILON {
        return Err(TangentCircleCandidateError::DegenerateLine);
    }
    let normal = [-direction[1] / length, direction[0] / length];
    Ok(NormalLine {
        normal,
        offset: normal[0].mul_add(from[0], normal[1] * from[1]),
        from,
        to,
    })
}

fn candidate_from(
    center: [f64; 2],
    radius: f64,
    lines: &[NormalLine],
) -> Result<TangentCircleCandidate, TangentCircleCandidateError> {
    if !center.into_iter().chain([radius]).all(f64::is_finite) {
        return Err(TangentCircleCandidateError::NoFiniteSolution);
    }
    let contacts: Vec<[f64; 2]> = lines
        .iter()
        .map(|line| {
            let distance = signed_distance(*line, center);
            [
                (-line.normal[0]).mul_add(distance, center[0]),
                (-line.normal[1]).mul_add(distance, center[1]),
            ]
        })
        .collect();
    if contacts
        .iter()
        .zip(lines)
        .any(|(&contact, line)| !point_on_segment(contact, line.from, line.to))
    {
        return Err(TangentCircleCandidateError::ContactOutsideSegment);
    }
    Ok(TangentCircleCandidate {
        center,
        radius,
        contacts,
    })
}

fn signed_distance(line: NormalLine, point: [f64; 2]) -> f64 {
    line.normal[0].mul_add(point[0], line.normal[1] * point[1]) - line.offset
}

fn parallel(first: NormalLine, second: NormalLine) -> bool {
    first.normal[0]
        .mul_add(second.normal[1], -first.normal[1] * second.normal[0])
        .abs()
        <= f64::EPSILON * 32.0
}

fn point_on_segment(point: [f64; 2], from: [f64; 2], to: [f64; 2]) -> bool {
    let direction = [to[0] - from[0], to[1] - from[1]];
    let length_squared = direction[0].mul_add(direction[0], direction[1] * direction[1]);
    let projection = direction[0].mul_add(point[0] - from[0], direction[1] * (point[1] - from[1]));
    let tolerance = f64::EPSILON * length_squared.max(1.0) * 32.0;
    projection >= -tolerance && projection <= length_squared + tolerance
}

fn solve_2x2(first: [f64; 2], second: [f64; 2], rhs: [f64; 2]) -> Option<[f64; 2]> {
    let determinant = first[0].mul_add(second[1], -first[1] * second[0]);
    if determinant.abs() <= f64::EPSILON * 32.0 {
        return None;
    }
    Some([
        rhs[0].mul_add(second[1], -first[1] * rhs[1]) / determinant,
        first[0].mul_add(rhs[1], -rhs[0] * second[0]) / determinant,
    ])
}

#[allow(clippy::indexing_slicing)]
fn solve_3x3(matrix: [[f64; 3]; 3], rhs: [f64; 3]) -> Option<[f64; 3]> {
    let determinant = determinant_3x3(matrix);
    if determinant.abs() <= f64::EPSILON * 64.0 {
        return None;
    }
    let replace = |column: usize| {
        let mut replaced = matrix;
        for row in 0..3 {
            replaced[row][column] = rhs[row];
        }
        determinant_3x3(replaced) / determinant
    };
    Some([replace(0), replace(1), replace(2)])
}

fn determinant_3x3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0].mul_add(
        matrix[1][1].mul_add(matrix[2][2], -matrix[1][2] * matrix[2][1]),
        -matrix[0][1].mul_add(
            matrix[1][0].mul_add(matrix[2][2], -matrix[1][2] * matrix[2][0]),
            -matrix[0][2] * matrix[1][0].mul_add(matrix[2][1], -matrix[1][1] * matrix[2][0]),
        ),
    )
}

fn squared_distance(first: [f64; 2], second: [f64; 2]) -> f64 {
    let dx = first[0] - second[0];
    let dy = first[1] - second[1];
    dx.mul_add(dx, dy * dy)
}

fn finite(points: impl IntoIterator<Item = [f64; 2]>) -> Result<(), TangentCircleCandidateError> {
    points
        .into_iter()
        .flatten()
        .all(f64::is_finite)
        .then_some(())
        .ok_or(TangentCircleCandidateError::NonFinite)
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    #[test]
    fn two_lines_and_a_witness_select_one_radius_branch() {
        let candidate = two_tangent_circle_candidate(
            [([0.0, 0.0], [10.0, 0.0]), ([0.0, 0.0], [0.0, 10.0])],
            [2.0, 3.0],
        )
        .unwrap();
        assert!((candidate.radius - 2.0).abs() < 1e-12);
        assert_eq!(candidate.center, [2.0, 2.0]);
        assert_eq!(candidate.contacts, vec![[2.0, 0.0], [0.0, 2.0]]);
    }

    #[test]
    fn three_lines_select_the_triangle_incircle() {
        let candidate = three_tangent_circle_candidate([
            ([0.0, 0.0], [10.0, 0.0], [5.0, 0.0]),
            ([10.0, 0.0], [0.0, 10.0], [5.0, 5.0]),
            ([0.0, 10.0], [0.0, 0.0], [0.0, 5.0]),
        ])
        .unwrap();
        let expected = 10.0 - 5.0 * 2.0_f64.sqrt();
        assert!((candidate.center[0] - expected).abs() < 1e-12);
        assert!((candidate.center[1] - expected).abs() < 1e-12);
        assert!((candidate.radius - expected).abs() < 1e-12);
    }

    #[test]
    fn parallel_lines_use_their_midline_and_witness_position() {
        let candidate = two_tangent_circle_candidate(
            [([0.0, 0.0], [10.0, 0.0]), ([10.0, 4.0], [0.0, 4.0])],
            [7.0, 3.0],
        )
        .unwrap();
        assert_eq!(candidate.center, [7.0, 2.0]);
        assert_eq!(candidate.radius, 2.0);
        assert_eq!(candidate.contacts, vec![[7.0, 0.0], [7.0, 4.0]]);
    }
}
