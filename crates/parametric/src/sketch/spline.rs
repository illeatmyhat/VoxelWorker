//! Cubic spline constructions shared by interactive and programmatic sketch authoring.
//!
//! Fit-point splines solve a natural (open) or periodic (closed) cubic interpolant. Control-point
//! splines evaluate a clamped uniform cubic B-spline and convert each knot span exactly into the
//! cubic Bézier representation consumed by the geometry substrate.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::missing_const_for_fn,
    clippy::needless_range_loop,
    clippy::range_plus_one,
    clippy::suboptimal_flops
)]

use substrate::rational_bezier::RationalBezier;

#[derive(Debug, Clone, PartialEq)]
pub struct SplineCandidate {
    pub pieces: Vec<RationalBezier>,
    pub closed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineCandidateError {
    NonFinite,
    TooFewPoints,
    CoincidentPoints,
    Singular,
}

/// Interpolate authored fit points with a C2 cubic spline, honoring any AUTHORED tangents.
///
/// `tangents` runs alongside `points`: `Some(vector)` is the curve's derivative at that fit point,
/// authored by the tangent handle standing there, and `None` leaves it to the interpolant.
///
/// # An authored tangent spends the C2 freedom at its point
///
/// The unauthored system asks for two continuous derivatives everywhere and solves for the one
/// tangent field that delivers it. Naming a tangent replaces that point's equation with the answer
/// — a Dirichlet row — so the curve still meets itself in position and slope there, but its
/// curvature is free to jump across the point. That is exactly what authoring a tangent MEANS:
/// the author has taken the freedom the smoothness condition was spending.
///
/// # Errors
///
/// Returns a typed error for non-finite input, insufficient/distinct points, a tangent that is not
/// finite, or a singular interpolation system.
pub fn fit_point_spline(
    points: &[[f64; 2]],
    tangents: &[Option<[f64; 2]>],
    closed: bool,
) -> Result<SplineCandidate, SplineCandidateError> {
    validate(points, if closed { 3 } else { 2 })?;
    if tangents.len() != points.len()
        || !tangents
            .iter()
            .flatten()
            .flatten()
            .all(|value| value.is_finite())
    {
        return Err(SplineCandidateError::NonFinite);
    }
    let derivatives = if closed {
        periodic_derivatives(points, tangents)?
    } else {
        natural_derivatives(points, tangents)?
    };
    let piece_count = if closed {
        points.len()
    } else {
        points.len().saturating_sub(1)
    };
    let mut pieces = Vec::with_capacity(piece_count);
    for index in 0..piece_count {
        let next = (index + 1) % points.len();
        pieces.push(RationalBezier::cubic([
            points[index],
            add_scaled(points[index], derivatives[index], 1.0 / 3.0),
            add_scaled(points[next], derivatives[next], -1.0 / 3.0),
            points[next],
        ]));
    }
    Ok(SplineCandidate { pieces, closed })
}

/// How much curve one unit of a point's STATION along this spline spends.
///
/// A station is held in length units rather than in pieces, for the reason an arc's radius is
/// held as itself rather than as its logarithm: a solve spends the coordinate whose derivative is
/// largest first, and a station that walked a whole piece per unit would be the cheapest thing in
/// the drawing, so every solve would pay with it instead of with the geometry the author is
/// actually holding. One unit of station is about one unit of travel, and the exchange rate with
/// a position is one to one.
///
/// Measured on the control polygon, which is what makes it cheap enough to ask for on the way in.
/// It is a UNIT, captured once from the curve a constraint was built against and then held fixed
/// while that curve deforms — a scale that drifted with the shape would move the point the author
/// pinned every time a neighbouring fit point did. A curve with no length answers one, because a
/// unit of zero is not a unit.
#[must_use]
pub fn station_length(candidate: &SplineCandidate) -> f64 {
    let total: f64 = candidate
        .pieces
        .iter()
        .map(|piece| {
            let (start, end) = (piece.control[0], piece.control[3]);
            (end[0] - start[0]).hypot(end[1] - start[1])
        })
        .sum();
    let pieces = u32::try_from(candidate.pieces.len()).unwrap_or(u32::MAX);
    let count = f64::from(pieces);
    if !total.is_finite() || total <= f64::EPSILON || count <= 0.0 {
        return 1.0;
    }
    total / count
}

/// Convert a clamped uniform control-point spline into exact cubic Bézier spans.
///
/// # Degree follows the control count
///
/// Cubic wherever there are controls enough for it, and one degree lower for each control short of
/// that: three controls is a quadratic, two is the straight line between them. A clamped uniform
/// B-spline of degree `d` needs `d + 1` controls, so this is the highest degree the polygon
/// actually supports rather than a special case bolted on.
///
/// That is what lets a control point be DELETED. The author's spline simplifies and heals instead
/// of vanishing, down to the two ends — pinning degree at three would make the curve die the
/// moment it dropped to three controls, which is not what removing a point means.
///
/// # Errors
///
/// Returns a typed error for non-finite input, fewer than two controls, or a collapsed control
/// polygon.
pub fn control_point_spline(
    controls: &[[f64; 2]],
) -> Result<SplineCandidate, SplineCandidateError> {
    validate(controls, 2)?;
    let degree = 3.min(controls.len() - 1);
    let span_count = controls.len() - degree;
    let knots = clamped_uniform_knots(controls.len(), degree);
    let mut pieces = Vec::with_capacity(span_count);
    for span in 0..span_count {
        let from = span as f64;
        let to = (span + 1) as f64;
        let (start, start_derivative) = bspline_value_derivative(controls, &knots, degree, from);
        let (end, end_derivative) = if span + 1 == span_count {
            let last = controls.len() - 1;
            (
                controls[last],
                scale(subtract(controls[last], controls[last - 1]), degree as f64),
            )
        } else {
            bspline_value_derivative(controls, &knots, degree, to)
        };
        pieces.push(RationalBezier::cubic([
            start,
            add_scaled(start, start_derivative, 1.0 / 3.0),
            add_scaled(end, end_derivative, -1.0 / 3.0),
            end,
        ]));
    }
    Ok(SplineCandidate {
        pieces,
        closed: false,
    })
}

fn validate(points: &[[f64; 2]], minimum: usize) -> Result<(), SplineCandidateError> {
    if points.len() < minimum {
        return Err(SplineCandidateError::TooFewPoints);
    }
    if !points.iter().flatten().all(|value| value.is_finite()) {
        return Err(SplineCandidateError::NonFinite);
    }
    if points.array_windows::<2>().any(|pair| pair[0] == pair[1]) {
        return Err(SplineCandidateError::CoincidentPoints);
    }
    Ok(())
}

fn natural_derivatives(
    points: &[[f64; 2]],
    tangents: &[Option<[f64; 2]>],
) -> Result<Vec<[f64; 2]>, SplineCandidateError> {
    let count = points.len();
    if count == 2 {
        let chord = subtract(points[1], points[0]);
        return Ok(vec![
            tangents[0].unwrap_or(chord),
            tangents[1].unwrap_or(chord),
        ]);
    }
    let mut lower = vec![1.0; count];
    let mut diagonal = vec![4.0; count];
    let mut upper = vec![1.0; count];
    let mut right = vec![[0.0; 2]; count];
    diagonal[0] = 2.0;
    upper[0] = 1.0;
    lower[count - 1] = 1.0;
    diagonal[count - 1] = 2.0;
    right[0] = scale(subtract(points[1], points[0]), 3.0);
    right[count - 1] = scale(subtract(points[count - 1], points[count - 2]), 3.0);
    for index in 1..count - 1 {
        right[index] = scale(subtract(points[index + 1], points[index - 1]), 3.0);
    }
    for (index, authored) in tangents.iter().enumerate() {
        let Some(authored) = *authored else { continue };
        lower[index] = 0.0;
        diagonal[index] = 1.0;
        upper[index] = 0.0;
        right[index] = authored;
    }
    solve_tridiagonal(&lower, &diagonal, &upper, &right)
}

fn periodic_derivatives(
    points: &[[f64; 2]],
    tangents: &[Option<[f64; 2]>],
) -> Result<Vec<[f64; 2]>, SplineCandidateError> {
    let count = points.len();
    let mut matrix = vec![vec![0.0; count]; count];
    let mut right = vec![[0.0; 2]; count];
    for index in 0..count {
        matrix[index][index] = 4.0;
        matrix[index][(index + count - 1) % count] = 1.0;
        matrix[index][(index + 1) % count] = 1.0;
        right[index] = scale(
            subtract(
                points[(index + 1) % count],
                points[(index + count - 1) % count],
            ),
            3.0,
        );
    }
    for (index, authored) in tangents.iter().enumerate() {
        let Some(authored) = *authored else { continue };
        matrix[index].fill(0.0);
        matrix[index][index] = 1.0;
        right[index] = authored;
    }
    solve_dense(matrix, right)
}

fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &[f64],
    upper: &[f64],
    right: &[[f64; 2]],
) -> Result<Vec<[f64; 2]>, SplineCandidateError> {
    let count = diagonal.len();
    let mut diagonal = diagonal.to_vec();
    let mut right = right.to_vec();
    for index in 1..count {
        if diagonal[index - 1].abs() <= f64::EPSILON {
            return Err(SplineCandidateError::Singular);
        }
        let factor = lower[index] / diagonal[index - 1];
        diagonal[index] -= factor * upper[index - 1];
        right[index] = add_scaled(right[index], right[index - 1], -factor);
    }
    let mut solution = vec![[0.0; 2]; count];
    solution[count - 1] = scale(right[count - 1], 1.0 / diagonal[count - 1]);
    for index in (0..count - 1).rev() {
        solution[index] = scale(
            add_scaled(right[index], solution[index + 1], -upper[index]),
            1.0 / diagonal[index],
        );
    }
    Ok(solution)
}

fn solve_dense(
    mut matrix: Vec<Vec<f64>>,
    mut right: Vec<[f64; 2]>,
) -> Result<Vec<[f64; 2]>, SplineCandidateError> {
    let count = matrix.len();
    for pivot in 0..count {
        let best = (pivot..count)
            .max_by(|&first, &second| {
                matrix[first][pivot]
                    .abs()
                    .total_cmp(&matrix[second][pivot].abs())
            })
            .ok_or(SplineCandidateError::Singular)?;
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return Err(SplineCandidateError::Singular);
        }
        matrix.swap(pivot, best);
        right.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..count {
            matrix[pivot][column] /= divisor;
        }
        right[pivot] = scale(right[pivot], 1.0 / divisor);
        for row in 0..count {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..count {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            right[row] = add_scaled(right[row], right[pivot], -factor);
        }
    }
    Ok(right)
}

fn clamped_uniform_knots(control_count: usize, degree: usize) -> Vec<f64> {
    let end = (control_count - degree) as f64;
    (0..control_count + degree + 1)
        .map(|index| {
            if index <= degree {
                0.0
            } else if index >= control_count {
                end
            } else {
                (index - degree) as f64
            }
        })
        .collect()
}

fn bspline_value_derivative(
    controls: &[[f64; 2]],
    knots: &[f64],
    degree: usize,
    parameter: f64,
) -> ([f64; 2], [f64; 2]) {
    let basis = basis_values(controls.len(), knots, degree, parameter);
    let lower = basis_values(controls.len() + 1, knots, degree - 1, parameter);
    let mut point = [0.0; 2];
    let mut derivative = [0.0; 2];
    for index in 0..controls.len() {
        point = add_scaled(point, controls[index], basis[index]);
        let left_denominator = knots[index + degree] - knots[index];
        let right_denominator = knots[index + degree + 1] - knots[index + 1];
        let left = if left_denominator.abs() <= f64::EPSILON {
            0.0
        } else {
            degree as f64 * lower[index] / left_denominator
        };
        let right = if right_denominator.abs() <= f64::EPSILON {
            0.0
        } else {
            degree as f64 * lower[index + 1] / right_denominator
        };
        derivative = add_scaled(derivative, controls[index], left - right);
    }
    (point, derivative)
}

fn basis_values(count: usize, knots: &[f64], degree: usize, parameter: f64) -> Vec<f64> {
    let mut values = vec![0.0; count + degree];
    for index in 0..values.len() {
        values[index] = if knots[index] <= parameter && parameter < knots[index + 1] {
            1.0
        } else {
            0.0
        };
    }
    for order in 1..=degree {
        for index in 0..values.len() - order {
            let left_denominator = knots[index + order] - knots[index];
            let right_denominator = knots[index + order + 1] - knots[index + 1];
            let left = if left_denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                (parameter - knots[index]) * values[index] / left_denominator
            };
            let right = if right_denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                (knots[index + order + 1] - parameter) * values[index + 1] / right_denominator
            };
            values[index] = left + right;
        }
    }
    values.truncate(count);
    values
}

fn subtract(first: [f64; 2], second: [f64; 2]) -> [f64; 2] {
    [first[0] - second[0], first[1] - second[1]]
}

fn scale(vector: [f64; 2], factor: f64) -> [f64; 2] {
    [vector[0] * factor, vector[1] * factor]
}

fn add_scaled(point: [f64; 2], vector: [f64; 2], factor: f64) -> [f64; 2] {
    [
        vector[0].mul_add(factor, point[0]),
        vector[1].mul_add(factor, point[1]),
    ]
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn fit_spline_passes_every_fit_point_with_c2_joins() {
        let points = [[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [8.0, 4.0]];
        let spline = fit_point_spline(&points, &vec![None; points.len()], false).unwrap();
        assert_eq!(spline.pieces.len(), 3);
        for (index, piece) in spline.pieces.iter().enumerate() {
            assert_eq!(piece.control[0], points[index]);
            assert_eq!(piece.control[3], points[index + 1]);
        }
        for pair in spline.pieces.array_windows::<2>() {
            let first = pair[0].derivative_at(1.0);
            let second = pair[1].derivative_at(0.0);
            assert!((first[0] - second[0]).abs() < 1.0e-10);
            assert!((first[1] - second[1]).abs() < 1.0e-10);
        }
    }

    /// An authored tangent is met exactly, and the points around it still take the smoothest
    /// tangents that can reach it — the condition replaces one equation, it does not rewrite the
    /// system.
    #[test]
    fn an_authored_tangent_is_honored_exactly_and_only_at_its_point() {
        let points = [[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [8.0, 4.0]];
        let authored = [0.0, 6.0];
        let spline = fit_point_spline(&points, &[Some(authored), None, None, None], false).unwrap();

        let leaving = spline.pieces[0].derivative_at(0.0);
        assert!((leaving[0] - authored[0]).abs() < 1.0e-10);
        assert!((leaving[1] - authored[1]).abs() < 1.0e-10);
        // The curve still passes through every fit point, and the joins the author did not touch
        // are still smooth.
        for (index, piece) in spline.pieces.iter().enumerate() {
            assert_eq!(piece.control[0], points[index]);
            assert_eq!(piece.control[3], points[index + 1]);
        }
        for pair in spline.pieces.array_windows::<2>() {
            let first = pair[0].derivative_at(1.0);
            let second = pair[1].derivative_at(0.0);
            assert!((first[0] - second[0]).abs() < 1.0e-10);
            assert!((first[1] - second[1]).abs() < 1.0e-10);
        }
    }

    /// A closed spline honors an authored tangent the same way, through the periodic system.
    #[test]
    fn a_closed_spline_honors_an_authored_tangent() {
        let authored = [1.0, -2.0];
        let spline = fit_point_spline(
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            &[None, Some(authored), None, None],
            true,
        )
        .unwrap();
        let leaving = spline.pieces[1].derivative_at(0.0);
        assert!((leaving[0] - authored[0]).abs() < 1.0e-10);
        assert!((leaving[1] - authored[1]).abs() < 1.0e-10);
    }

    /// A tangent list that does not run alongside the points is not a spline anyone can read.
    #[test]
    fn a_mismatched_or_non_finite_tangent_list_is_refused() {
        let points = [[0.0, 0.0], [2.0, 3.0], [5.0, 1.0]];
        assert_eq!(
            fit_point_spline(&points, &[None, None], false),
            Err(SplineCandidateError::NonFinite)
        );
        assert_eq!(
            fit_point_spline(&points, &[Some([f64::NAN, 0.0]), None, None], false),
            Err(SplineCandidateError::NonFinite)
        );
    }

    #[test]
    fn closed_fit_spline_is_periodic() {
        let spline = fit_point_spline(
            &[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            &[None; 4],
            true,
        )
        .unwrap();
        assert_eq!(spline.pieces.len(), 4);
        assert_eq!(spline.pieces[3].control[3], spline.pieces[0].control[0]);
        let leaving = spline.pieces[0].derivative_at(0.0);
        let arriving = spline.pieces[3].derivative_at(1.0);
        assert!((leaving[0] - arriving[0]).abs() < 1.0e-10);
        assert!((leaving[1] - arriving[1]).abs() < 1.0e-10);
    }

    #[test]
    fn four_control_points_are_one_ordinary_cubic_bezier() {
        let controls = [[0.0, 0.0], [1.0, 3.0], [4.0, 3.0], [5.0, 0.0]];
        let spline = control_point_spline(&controls).unwrap();
        assert_eq!(spline.pieces, vec![RationalBezier::cubic(controls)]);
    }

    /// Three controls is the quadratic through them, degree-elevated — the shape a cubic authoring
    /// path has to produce so that removing a control changes the curve without ending it.
    #[test]
    fn three_control_points_are_the_quadratic_they_describe() {
        let controls = [[0.0, 0.0], [2.0, 4.0], [6.0, 0.0]];
        let spline = control_point_spline(&controls).unwrap();
        assert_eq!(spline.pieces.len(), 1);
        let elevated = spline.pieces[0].control;
        assert_eq!(elevated[0], controls[0]);
        assert_eq!(elevated[3], controls[2]);
        // Cubic elevation of a quadratic puts the inner controls a third of the way from each end
        // toward the shoulder.
        for axis in 0..2 {
            let toward = controls[0][axis] + (controls[1][axis] - controls[0][axis]) * 2.0 / 3.0;
            assert!((elevated[1][axis] - toward).abs() < 1.0e-10);
            let back = controls[2][axis] + (controls[1][axis] - controls[2][axis]) * 2.0 / 3.0;
            assert!((elevated[2][axis] - back).abs() < 1.0e-10);
        }
    }

    /// The floor of the heal: two controls still answer, as the straight line between them.
    #[test]
    fn two_control_points_are_the_segment_between_them() {
        let controls = [[1.0, 1.0], [7.0, 4.0]];
        let spline = control_point_spline(&controls).unwrap();
        assert_eq!(spline.pieces.len(), 1);
        for step in 0..=4 {
            let along = f64::from(step) / 4.0;
            let on_curve = spline.pieces[0].point_at(along);
            for axis in 0..2 {
                let straight = controls[0][axis] + (controls[1][axis] - controls[0][axis]) * along;
                assert!(
                    (on_curve[axis] - straight).abs() < 1.0e-10,
                    "at {along} axis {axis}: {on_curve:?} is not on the chord"
                );
            }
        }
    }

    #[test]
    fn one_control_point_is_not_a_curve() {
        assert_eq!(
            control_point_spline(&[[0.0, 0.0]]),
            Err(SplineCandidateError::TooFewPoints)
        );
    }
}
