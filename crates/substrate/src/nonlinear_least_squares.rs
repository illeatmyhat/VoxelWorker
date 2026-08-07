//! Nonlinear least squares by Powell's Dog Leg over a rank-revealing linear solve, with a rank
//! report — the numerical core a geometric constraint solver runs on.
//!
//! The problem is always the same shape: a vector of PARAMETERS the author is free to move (point
//! coordinates, radii, angles), a vector of RESIDUALS that are zero exactly when every constraint
//! holds, and the question of which parameter vector makes them so. Nothing here knows what a
//! constraint is — a residual is a function, and the caller decides that "these two points are
//! coincident" means `[ax − bx, ay − by]`.
//!
//! ## Why Dog Leg and not plain Gauss-Newton or plain LM
//!
//! Gauss-Newton converges quadratically near the answer and diverges cheerfully far from it.
//! Levenberg–Marquardt fixes that by damping, but its damping parameter has no geometric meaning,
//! so a bad λ shows up as a step that is silently far too short. Powell's Dog Leg keeps an
//! explicit TRUST REGION radius instead: it takes the Gauss-Newton step when that step fits inside
//! the region, the steepest-descent step when even that does not, and the point where the segment
//! between them leaves the region otherwise. The radius is in the same units as the parameters, so
//! "the solver is taking 0.01-voxel steps" is a statement anyone can act on.
//!
//! ## Every step is the MINIMUM-NORM one, whatever shape the system is
//!
//! A sketch's Jacobian is rank-deficient BY CONSTRUCTION — a free degree of freedom is exactly a
//! direction the residuals do not see — so there is no single `h` solving `J h = −r` and there is no
//! point pretending otherwise. Of all the `h` that minimize `‖J h + r‖`, the step taken is the
//! SHORTEST, which is the one that leaves every parameter no relation names nearest where the
//! author put it. [`complete_orthogonal_decomposition`](crate::complete_orthogonal_decomposition)
//! computes it, and does so without ever forming `JᵀJ` — see `gauss_newton_step` for what forming
//! it cost.
//!
//! **Picking the shortest is a GAUGE CHOICE**, in the sense a fluid solver means when it pins the
//! constant mode of a pressure field: the free directions have to be settled by a rule, and the
//! only question is whether the rule is stated or left to rounding error. Stating it is what makes
//! the solve a continuous function of its input, and a drag a continuous function of the cursor.
//!
//! ## The rank report is the diagnosis
//!
//! A solver that only answers "converged" is unusable for authoring, because the two ways a sketch
//! is wrong both converge: an UNDER-constrained drawing has a solution and infinitely many others
//! near it, and a REDUNDANT one has constraints that say the same thing twice. Both are visible in
//! the rank of the Jacobian and nowhere else, so [`SolveReport`] carries
//! [`degrees_of_freedom`](SolveReport::degrees_of_freedom) (parameters the residuals do not pin)
//! and [`redundant_residuals`](SolveReport::redundant_residuals) (residuals that add no
//! information) alongside the outcome.

use crate::complete_orthogonal_decomposition::minimum_norm_least_squares;

/// A system of residual functions of a parameter vector: zero everywhere exactly when every
/// constraint the caller encoded is satisfied.
///
/// The residuals must be at least once differentiable in the parameters wherever the solver is
/// asked to work — the Jacobian is taken by finite differences, so a kink is a place the step
/// direction is a guess. Encode an absolute value or a `min` as two residuals, not one.
pub trait ResidualSystem {
    /// How many parameters the system is a function of.
    fn parameter_count(&self) -> usize;

    /// How many residuals it produces. May be fewer than, equal to, or more than the parameter
    /// count; all three are ordinary.
    fn residual_count(&self) -> usize;

    /// Write the residuals at `parameters` into `into`, which is
    /// [`residual_count`](Self::residual_count) long.
    fn residuals(&self, parameters: &[f64], into: &mut [f64]);
}

/// Why the solve stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveOutcome {
    /// The residuals are as small as the tolerances ask for. The parameters are a solution.
    Converged,
    /// The solver stopped moving without the residuals getting small: either the step it wants is
    /// smaller than the parameters can usefully represent, or the gradient vanished at a point the
    /// residuals are still large at. Both say the same thing — this is the best it can reach from
    /// where it started, and it is the least-squares COMPROMISE rather than a solution, which is
    /// what a sketch with contradictory constraints settles into. Read
    /// [`residual_norm`](SolveReport::residual_norm) to see how far from satisfied they are.
    ///
    /// **This is not the test for whether the constraints hold, and using it as one is a trap.**
    /// [`residual_tolerance`](SolveSettings::residual_tolerance) is absolute while
    /// [`step_tolerance`](SolveSettings::step_tolerance) is relative to the length of the
    /// parameter vector, so on a large system the step test can fire first — stopping with the
    /// residuals far under anything the caller's units can express, and reporting this. The
    /// outcome says why the SEARCH stopped; only `residual_norm` says whether the ANSWER is one.
    /// A caller that knows what "satisfied" means in its own units should ask that question of
    /// the norm and treat the outcome as diagnosis (`document::sketch`'s trial solve does).
    Stalled,
    /// The iteration budget ran out with the residuals still shrinking. Not a failure: it means
    /// the answer is somewhere ahead and the caller decides whether to spend more.
    ExhaustedIterations,
}

/// What a solve did, and what shape the system turned out to be.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    /// Why it stopped.
    pub outcome: SolveOutcome,
    /// How many trust-region iterations it took.
    pub iterations: usize,
    /// The Euclidean norm of the residuals at the parameters it left behind. Near zero means the
    /// constraints hold; anything else on a `Stalled` outcome means they cannot all hold at once.
    pub residual_norm: f64,
    /// Parameters the residuals do not pin down, at the solution: `parameter_count − rank(J)`.
    /// Zero is a fully-constrained sketch; anything above it is how many ways the drawing can
    /// still move.
    pub degrees_of_freedom: usize,
    /// Residuals that add no information at the solution: `residual_count − rank(J)`. Above zero
    /// means constraints say the same thing twice — harmless when they AGREE, and the reason a
    /// contradictory sketch stalls at a non-zero norm when they do not.
    pub redundant_residuals: usize,
}

/// The stopping tolerances and budget of one solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveSettings {
    /// Stop when every component of the gradient `Jᵀr` is under this — a stationary point. Stopping
    /// there is a [`Stalled`](SolveOutcome::Stalled), never a convergence: a flat objective says the
    /// search cannot continue, not that the residuals are zero.
    pub gradient_tolerance: f64,
    /// Stop when the step is under this, RELATIVE to the size of the parameters themselves, so
    /// the same setting means the same thing for a sketch in voxels and one in blocks.
    pub step_tolerance: f64,
    /// Stop when the residual norm is under this. The one tolerance that is about the ANSWER
    /// rather than about the search.
    pub residual_tolerance: f64,
    /// Stop when an accepted step improves the sum of squares by less than this SHARE of it.
    ///
    /// The other three stopping tests all ask whether the search has arrived. This one asks whether
    /// it is still going anywhere, and it is the test that catches an INCOMPATIBLE system — one
    /// where no parameter vector satisfies every residual, so there is a least-squares compromise
    /// to find and no solution to converge to. Gauss-Newton reaches such a compromise only
    /// linearly, and linear convergence past the first few digits is a search spending its whole
    /// budget to move the answer by nothing.
    ///
    /// Relative rather than absolute, because "improved by 1e-9" means something different on a
    /// system compromising at 4e-3 than on one compromising at 40. Ceres calls the same test
    /// `function_tolerance`.
    ///
    /// **The default is a hundred times tighter than Ceres's**, and the difference is measured
    /// rather than cautious. Over a drag of a curved slot — 1005 solves, 349,000 iterations without
    /// this test, 2176 of them spending the whole iteration ceiling — the trade runs:
    ///
    /// | tolerance | iterations | hit the ceiling | error in a scale-invariance check |
    /// | --------- | ---------: | --------------: | --------------------------------: |
    /// | off       |    349,196 |            2176 |                          2.005e-5 |
    /// | **1e-8**  | **269,624**|        **1015** |                      **2.090e-5** |
    /// | 1e-7      |    212,868 |              45 |                          5.936e-5 |
    /// | 1e-6      |    155,487 |               0 |                          1.989e-4 |
    ///
    /// A quarter of the work goes for four percent of the error at `1e-8`. One notch looser triples
    /// the error, and Ceres's own default breaks the check outright — which is what a general
    /// optimizer's default looks like on a problem whose answer a person is looking at.
    ///
    /// **It reports [`Stalled`](SolveOutcome::Stalled), which is the honest outcome and not a
    /// failure**: the search stopped because it had stopped moving, and whether the ANSWER is one is
    /// read off [`residual_norm`](SolveReport::residual_norm) as always. A system that genuinely
    /// converges is claimed by the residual test first — this one can only fire while the residuals
    /// are still large.
    pub improvement_tolerance: f64,
    /// The trust region's starting radius, in parameter units.
    pub initial_trust_radius: f64,
    /// The iteration ceiling.
    pub maximum_iterations: usize,
}

impl Default for SolveSettings {
    /// Tuned for a sketch measured in voxels: converge to well under a thousandth of a voxel, give
    /// up on a search that has stopped moving the answer, and give up entirely after a budget that
    /// is generous for a drawing and instant for a machine.
    fn default() -> Self {
        Self {
            gradient_tolerance: 1.0e-12,
            step_tolerance: 1.0e-12,
            residual_tolerance: 1.0e-10,
            improvement_tolerance: 1.0e-8,
            initial_trust_radius: 1.0,
            maximum_iterations: 100,
        }
    }
}

/// Solve `system` in place from the starting guess in `parameters`.
///
/// The starting guess matters and is not a formality: these systems have many solutions (a
/// distance constraint is a circle of them), and the one this finds is the one NEAREST the guess.
/// For a sketch that is exactly right — the author's drawing is the guess, so the solver moves the
/// geometry as little as it can, which is what makes a solve feel like a nudge rather than a
/// rearrangement.
///
/// # Panics
///
/// Panics if `parameters.len()` does not match `system.parameter_count()`.
pub fn solve(
    system: &dyn ResidualSystem,
    parameters: &mut [f64],
    settings: SolveSettings,
) -> SolveReport {
    let parameter_count = system.parameter_count();
    let residual_count = system.residual_count();
    assert_eq!(
        parameters.len(),
        parameter_count,
        "the starting guess must have one entry per parameter"
    );
    let mut residuals = vec![0.0; residual_count];
    system.residuals(parameters, &mut residuals);
    let mut trust_radius = settings.initial_trust_radius.max(f64::MIN_POSITIVE);
    let mut outcome = SolveOutcome::ExhaustedIterations;
    let mut iterations = 0;
    let mut jacobian_matrix = jacobian(system, parameters);

    for iteration in 0..settings.maximum_iterations {
        iterations = iteration.saturating_add(1);
        if euclidean_norm(&residuals) <= settings.residual_tolerance {
            outcome = SolveOutcome::Converged;
            break;
        }
        // g = Jᵀr, the direction the sum of squares grows fastest in.
        let gradient = transpose_times(
            &jacobian_matrix,
            &residuals,
            residual_count,
            parameter_count,
        );
        if infinity_norm(&gradient) <= settings.gradient_tolerance {
            // A vanishing gradient is a STATIONARY point, not a solution. The residual test a few
            // lines up already claimed every genuine convergence, so arriving here means the sum of
            // squares has flattened out with the residuals still too big — a local minimum the
            // constraints are not satisfied at, which is what a contradictory sketch settles into.
            outcome = SolveOutcome::Stalled;
            break;
        }
        let step = dog_leg_step(
            &jacobian_matrix,
            &residuals,
            &gradient,
            residual_count,
            parameter_count,
            trust_radius,
        );
        let step_size = euclidean_norm(&step);
        if step_size
            <= settings.step_tolerance * (euclidean_norm(parameters) + settings.step_tolerance)
        {
            outcome = SolveOutcome::Stalled;
            break;
        }
        let candidate: Vec<f64> = parameters
            .iter()
            .zip(&step)
            .map(|(at, delta)| at + delta)
            .collect();
        let mut candidate_residuals = vec![0.0; residual_count];
        system.residuals(&candidate, &mut candidate_residuals);
        // The gain ratio: how much of the improvement the LINEAR model promised was actually
        // delivered. Above zero the step is an improvement and is taken; near one the model is
        // trustworthy over this radius and the region may grow.
        let objective = sum_of_squares(&residuals);
        let actual = objective - sum_of_squares(&candidate_residuals);
        let predicted = predicted_reduction(
            &jacobian_matrix,
            &residuals,
            &step,
            residual_count,
            parameter_count,
        );
        let gain = if predicted > 0.0 {
            actual / predicted
        } else {
            // The model promised nothing, so any real improvement is a windfall and any
            // deterioration is a reason to shrink.
            if actual > 0.0 {
                1.0
            } else {
                -1.0
            }
        };
        if gain > 0.0 {
            parameters.copy_from_slice(&candidate);
            residuals = candidate_residuals;
            jacobian_matrix = jacobian(system, parameters);
            // Tested only on an ACCEPTED step, and after taking it. A rejected step leaves the
            // objective alone, so counting it as "no improvement" would stop the search at the
            // first bad guess rather than at the end of its progress — the trust radius collapsing
            // is what says a rejected step will keep being rejected, and that test is below.
            if actual <= settings.improvement_tolerance * objective {
                outcome = SolveOutcome::Stalled;
                break;
            }
        }
        // Madsen–Nielsen–Tingleff's radius update: grow on a well-predicted step, shrink hard on
        // a rejected one, leave it alone in between.
        if gain > 0.75 {
            trust_radius = trust_radius.max(3.0 * step_size);
        } else if gain < 0.25 {
            trust_radius /= 2.0;
            if trust_radius
                <= settings.step_tolerance * (euclidean_norm(parameters) + settings.step_tolerance)
            {
                outcome = SolveOutcome::Stalled;
                break;
            }
        }
    }

    let rank = rank(&jacobian_matrix, residual_count, parameter_count);
    SolveReport {
        outcome,
        iterations,
        residual_norm: euclidean_norm(&residuals),
        degrees_of_freedom: parameter_count.saturating_sub(rank),
        redundant_residuals: residual_count.saturating_sub(rank),
    }
}

/// The Jacobian `∂r_i/∂x_j` at `parameters`, row-major (`residual_count × parameter_count`), by
/// CENTRAL differences.
///
/// Central and not forward on purpose: a forward difference costs one evaluation per parameter
/// instead of two but its error is first order in the step, and near a solution — where every
/// residual is nearly zero and the whole answer is in their differences — that error is the
/// answer. The step is scaled to each parameter's own magnitude so a coordinate of 1000 and one of
/// 0.001 are both differenced sensibly.
#[must_use]
pub fn jacobian(system: &dyn ResidualSystem, parameters: &[f64]) -> Vec<f64> {
    let parameter_count = system.parameter_count();
    let residual_count = system.residual_count();
    let mut matrix = vec![0.0; residual_count.saturating_mul(parameter_count)];
    let mut moved = parameters.to_vec();
    let mut ahead = vec![0.0; residual_count];
    let mut behind = vec![0.0; residual_count];
    for (column, &parameter) in parameters.iter().take(parameter_count).enumerate() {
        let step = DIFFERENCE_STEP * parameter.abs().max(1.0);
        if let Some(slot) = moved.get_mut(column) {
            *slot = parameter + step;
        }
        system.residuals(&moved, &mut ahead);
        if let Some(slot) = moved.get_mut(column) {
            *slot = parameter - step;
        }
        system.residuals(&moved, &mut behind);
        if let Some(slot) = moved.get_mut(column) {
            *slot = parameter;
        }
        for ((row, &ahead_value), &behind_value) in matrix
            .chunks_exact_mut(parameter_count)
            .zip(ahead.iter())
            .zip(behind.iter())
            .take(residual_count)
        {
            if let Some(slot) = row.get_mut(column) {
                *slot = (ahead_value - behind_value) / (2.0 * step);
            }
        }
    }
    matrix
}

/// The finite-difference step, relative to the parameter's own magnitude. The cube root of the
/// machine epsilon is where a central difference's truncation error and its cancellation error
/// meet — smaller is not more accurate, it is less.
const DIFFERENCE_STEP: f64 = 6.0e-6;

/// The rank of a `rows × columns` row-major matrix: how many of its rows are linearly independent,
/// by Gaussian elimination with PARTIAL PIVOTING.
///
/// This is what the degrees-of-freedom report is made of. The tolerance is relative to the largest
/// pivot seen, so it scales with the matrix rather than assuming anything about the units the
/// caller's parameters are in.
#[must_use]
pub fn rank(matrix: &[f64], rows: usize, columns: usize) -> usize {
    if rows == 0 || columns == 0 {
        return 0;
    }
    let mut work = matrix.to_vec();
    let mut row_slices: Vec<&mut [f64]> = work.chunks_exact_mut(columns).take(rows).collect();
    let mut rank = 0;
    let mut largest_pivot = 0.0f64;
    for column in 0..columns {
        if rank == row_slices.len() {
            break;
        }
        // The largest remaining entry in this column is the pivot — anything smaller amplifies
        // rounding error into the rows below it.
        let (pivot_row, pivot) = row_slices
            .iter()
            .enumerate()
            .skip(rank)
            .map(|(row, values)| (row, values.get(column).copied().unwrap_or_default().abs()))
            .fold(
                (rank, 0.0),
                |best, here| if here.1 > best.1 { here } else { best },
            );
        largest_pivot = largest_pivot.max(pivot);
        if pivot <= RANK_TOLERANCE * largest_pivot.max(1.0) {
            continue;
        }
        row_slices.swap(rank, pivot_row);
        let (before_and_rank, below) = row_slices.split_at_mut(rank.saturating_add(1));
        let Some(pivot_values) = before_and_rank.last() else {
            break;
        };
        let Some(&scale) = pivot_values.get(column) else {
            continue;
        };
        for values in below {
            let Some(&value) = values.get(column) else {
                continue;
            };
            let factor = value / scale;
            if factor == 0.0 {
                continue;
            }
            for (target, pivot_value) in values
                .iter_mut()
                .skip(column)
                .zip(pivot_values.iter().skip(column))
            {
                *target = (-factor).mul_add(*pivot_value, *target);
            }
        }
        rank = rank.saturating_add(1);
    }
    rank
}

/// How small a pivot may be, relative to the largest seen, and still count as a real one. Below
/// this it is rounding noise in a column that was already dependent.
const RANK_TOLERANCE: f64 = 1.0e-10;

/// The Dog Leg step within `trust_radius`: the Gauss-Newton step where it fits, the steepest
/// descent step where even that does not, and the point where the segment between them leaves the
/// trust region in between.
fn dog_leg_step(
    jacobian_matrix: &[f64],
    residuals: &[f64],
    gradient: &[f64],
    rows: usize,
    columns: usize,
    trust_radius: f64,
) -> Vec<f64> {
    // The steepest-descent step, at the length that minimizes the linear model along `−g`:
    // α = ‖g‖² / ‖Jg‖².
    let jacobian_gradient = times(jacobian_matrix, gradient, rows, columns);
    let denominator = sum_of_squares(&jacobian_gradient);
    let alpha = if denominator > 0.0 {
        sum_of_squares(gradient) / denominator
    } else {
        0.0
    };
    let steepest: Vec<f64> = gradient.iter().map(|value| -alpha * value).collect();
    let steepest_length = euclidean_norm(&steepest);

    let Some(gauss_newton) = gauss_newton_step(jacobian_matrix, residuals, rows, columns) else {
        // No Gauss-Newton step exists even damped — take what steepest descent offers, clipped.
        return clipped(&steepest, steepest_length, trust_radius);
    };
    let gauss_newton_length = euclidean_norm(&gauss_newton);
    if gauss_newton_length <= trust_radius {
        return gauss_newton;
    }
    if steepest_length >= trust_radius {
        return clipped(&steepest, steepest_length, trust_radius);
    }
    // Somewhere on the segment from the steepest-descent point to the Gauss-Newton one, at the
    // radius: solve ‖h_sd + β(h_gn − h_sd)‖ = Δ for the positive root.
    let leg: Vec<f64> = gauss_newton
        .iter()
        .zip(&steepest)
        .map(|(newton, descent)| newton - descent)
        .collect();
    let a = sum_of_squares(&leg);
    let b = 2.0 * dot(&steepest, &leg);
    let c = trust_radius.mul_add(-trust_radius, steepest_length * steepest_length);
    let discriminant = (4.0 * a).mul_add(-c, b * b).max(0.0);
    let beta = if a > 0.0 {
        (-b + discriminant.sqrt()) / (2.0 * a)
    } else {
        0.0
    };
    steepest
        .iter()
        .zip(&leg)
        .map(|(descent, along)| descent + beta * along)
        .collect()
}

/// `step` scaled to exactly `trust_radius` if it is longer, unchanged if it is not.
fn clipped(step: &[f64], length: f64, trust_radius: f64) -> Vec<f64> {
    if length <= trust_radius || length == 0.0 {
        return step.to_vec();
    }
    let scale = trust_radius / length;
    step.iter().map(|value| value * scale).collect()
}

/// The Gauss-Newton step: the MINIMUM-NORM least-squares solution of `J h = −r`.
///
/// One question and one answer, whatever shape the system is. Over-determined, under-determined and
/// rank-deficient are not three cases here — `h = J⁺(−r)` is all of them, and
/// [`minimum_norm_least_squares`] computes it by complete orthogonal decomposition.
///
/// **Not the normal equations, and that is the whole point.** `JᵀJ h = −Jᵀr` is the textbook form
/// and it was what this did; it squares the condition number, and a sketch's Jacobian routinely
/// carries six or seven digits of conditioning loss, which leaves `JᵀJ` past what a `f64` can
/// factorise at all. Measured on a curved slot mid-drag, the Cholesky failed on 99 iterations out
/// of 100 and every step came out of the damping repair — that is, out of `JᵀJ + λI`, which is a
/// different problem, perturbed hardest in exactly the directions the constraints pinned down
/// least. The free sweep is such a direction, so what the author saw was the drawing swinging
/// hundreds of times the cursor step. Working on `J` itself never squares anything.
///
/// **The minimum-norm choice is a GAUGE CHOICE and is made here on purpose.** An under-constrained
/// drawing has a whole subspace of equally good steps; the shortest one is the answer that leaves
/// every parameter no relation names nearest where the author put it, and — unlike whatever a
/// damped factorisation happens to land on — it is a continuous function of the Jacobian, so
/// neighbouring cursor positions give neighbouring drawings.
fn gauss_newton_step(
    jacobian_matrix: &[f64],
    residuals: &[f64],
    rows: usize,
    columns: usize,
) -> Option<Vec<f64>> {
    let negative_residuals: Vec<f64> = residuals.iter().take(rows).map(|value| -value).collect();
    minimum_norm_least_squares(
        jacobian_matrix,
        &negative_residuals,
        rows,
        columns,
        JACOBIAN_RANK_TOLERANCE,
    )
    .map(|answer| answer.solution)
}

/// How large a direction must be, relative to the largest, for the Jacobian to be believed about
/// it. Below this it is not a weak constraint, it is the FINITE-DIFFERENCE NOISE FLOOR.
///
/// Measured rather than chosen. The Jacobian is taken by central differences with a step of
/// [`DIFFERENCE_STEP`], whose cancellation error is about `ε/h` — four parts in `10¹¹` — so nothing
/// below that carries information. On a curved slot mid-drag the decomposition's diagonal came out
/// as nine directions between 1 and 0.28, three more between `2e-5` and `2e-6`, then one drifting
/// between `3e-10` and `1e-8` from one cursor position to the next, then three at the machine
/// epsilon. The wobbling one is the noise floor showing itself: three orders of magnitude of clear
/// gap sit between it and the last real direction, and this tolerance sits in the middle of that
/// gap. Sweeping it confirms the plateau — every value from `1e-10` to `1e-6` gives the same
/// drawing, and tightening to `1e-13` puts the noise back in and swings the drawing thousands of
/// times the cursor step.
const JACOBIAN_RANK_TOLERANCE: f64 = 1.0e-8;

/// The reduction in the sum of squares the LINEAR model predicts for `step`:
/// `‖r‖² − ‖r + J·step‖²`.
fn predicted_reduction(
    jacobian_matrix: &[f64],
    residuals: &[f64],
    step: &[f64],
    rows: usize,
    columns: usize,
) -> f64 {
    let moved = times(jacobian_matrix, step, rows, columns);
    let after: f64 = residuals
        .iter()
        .zip(&moved)
        .map(|(residual, change)| (residual + change) * (residual + change))
        .sum();
    sum_of_squares(residuals) - after
}

/// `M · v` for a `rows × columns` row-major `M`.
fn times(matrix: &[f64], vector: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    matrix
        .chunks_exact(columns)
        .take(rows)
        .map(|row| {
            row.iter()
                .zip(vector.iter())
                .take(columns)
                .map(|(&matrix_value, &vector_value)| matrix_value * vector_value)
                .sum()
        })
        .collect()
}

/// `Mᵀ · v` for a `rows × columns` row-major `M`.
fn transpose_times(matrix: &[f64], vector: &[f64], rows: usize, columns: usize) -> Vec<f64> {
    (0..columns)
        .map(|column| {
            matrix
                .chunks_exact(columns)
                .take(rows)
                .zip(vector.iter())
                .filter_map(|(row, &vector_value)| {
                    row.get(column)
                        .map(|&matrix_value| matrix_value * vector_value)
                })
                .sum()
        })
        .collect()
}

/// The sum of the squares of a vector's entries.
fn sum_of_squares(vector: &[f64]) -> f64 {
    vector.iter().map(|value| value * value).sum()
}

/// The Euclidean length of a vector.
fn euclidean_norm(vector: &[f64]) -> f64 {
    sum_of_squares(vector).sqrt()
}

/// The largest absolute entry of a vector.
fn infinity_norm(vector: &[f64]) -> f64 {
    vector
        .iter()
        .fold(0.0f64, |best, value| best.max(value.abs()))
}

/// The dot product of two vectors.
fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::imprecise_flops,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::suboptimal_flops,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    /// One residual as a borrowed closure.
    type Residual<'a> = &'a dyn Fn(&[f64]) -> f64;

    /// A residual system built from closures, so a test states its constraints inline.
    struct Closures<'a> {
        parameters: usize,
        residuals: Vec<Residual<'a>>,
    }

    impl ResidualSystem for Closures<'_> {
        fn parameter_count(&self) -> usize {
            self.parameters
        }
        fn residual_count(&self) -> usize {
            self.residuals.len()
        }
        fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
            for (slot, residual) in into.iter_mut().zip(&self.residuals) {
                *slot = residual(parameters);
            }
        }
    }

    /// The headline, in the vocabulary the caller will use it in: two points, a constraint saying
    /// they are 10 apart and another saying the segment between them is horizontal. The solver
    /// moves them the LEAST it can from where they were drawn.
    #[test]
    fn a_distance_and_a_horizontal_constraint_solve_together() {
        let distance = |p: &[f64]| ((p[2] - p[0]).powi(2) + (p[3] - p[1]).powi(2)).sqrt() - 10.0;
        let horizontal = |p: &[f64]| p[3] - p[1];
        let system = Closures {
            parameters: 4,
            residuals: vec![&distance, &horizontal],
        };
        // Drawn roughly right: 8.06 apart and a little off level.
        let mut parameters = vec![0.0, 0.0, 8.0, 1.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert_eq!(report.outcome, SolveOutcome::Converged, "{report:?}");
        assert!(report.residual_norm < 1e-9, "{report:?}");
        let span = ((parameters[2] - parameters[0]).powi(2)
            + (parameters[3] - parameters[1]).powi(2))
        .sqrt();
        assert!((span - 10.0).abs() < 1e-6, "10 apart: {span}");
        assert!((parameters[3] - parameters[1]).abs() < 1e-6, "and level");
        // Two constraints on four parameters: the pair can still slide and the whole thing can
        // still translate, which is two residuals' worth of rank and two free directions.
        assert_eq!(report.degrees_of_freedom, 2, "{report:?}");
        assert_eq!(report.redundant_residuals, 0, "{report:?}");
    }

    /// The solve finds the solution NEAREST the guess, which is what makes it feel like a nudge:
    /// the same system solved from a mirrored drawing lands on the mirrored answer.
    #[test]
    fn the_nearest_solution_wins() {
        let on_unit_circle = |p: &[f64]| p[0] * p[0] + p[1] * p[1] - 1.0;
        let system = Closures {
            parameters: 2,
            residuals: vec![&on_unit_circle],
        };
        let mut right = vec![2.0, 0.1];
        solve(&system, &mut right, SolveSettings::default());
        assert!(right[0] > 0.0, "stayed on its own side: {right:?}");
        let mut left = vec![-2.0, 0.1];
        solve(&system, &mut left, SolveSettings::default());
        assert!(left[0] < 0.0, "and so did the mirrored guess: {left:?}");
    }

    /// A hard start: Rosenbrock's valley, the standard test for whether an optimizer follows a
    /// curved trough or bounces out of it. Gauss-Newton alone diverges from `(-1.2, 1)`.
    #[test]
    fn the_trust_region_follows_a_curved_valley() {
        let curve = |p: &[f64]| 10.0 * (p[1] - p[0] * p[0]);
        let offset = |p: &[f64]| 1.0 - p[0];
        let system = Closures {
            parameters: 2,
            residuals: vec![&curve, &offset],
        };
        let mut parameters = vec![-1.2, 1.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!(
            (parameters[0] - 1.0).abs() < 1e-5,
            "{parameters:?} {report:?}"
        );
        assert!(
            (parameters[1] - 1.0).abs() < 1e-5,
            "{parameters:?} {report:?}"
        );
    }

    /// An UNDER-constrained system: one constraint, two parameters. It solves, and the report says
    /// the drawing can still move in one direction — the number a sketch shows the author.
    #[test]
    fn an_under_constrained_system_reports_its_freedom() {
        let sum = |p: &[f64]| p[0] + p[1] - 3.0;
        let system = Closures {
            parameters: 2,
            residuals: vec![&sum],
        };
        let mut parameters = vec![0.0, 0.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!(report.residual_norm < 1e-9, "{report:?}");
        assert_eq!(report.degrees_of_freedom, 1);
        assert_eq!(report.redundant_residuals, 0);
    }

    /// A REDUNDANT system whose constraints agree: the same thing said twice. It solves, and the
    /// report says one residual carried no information — the warning a sketch shows before the
    /// author adds a third that contradicts.
    #[test]
    fn agreeing_redundancy_solves_and_is_reported() {
        let once = |p: &[f64]| p[0] - 4.0;
        let twice = |p: &[f64]| 2.0 * p[0] - 8.0;
        let system = Closures {
            parameters: 1,
            residuals: vec![&once, &twice],
        };
        let mut parameters = vec![0.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!((parameters[0] - 4.0).abs() < 1e-9, "{parameters:?}");
        assert_eq!(report.degrees_of_freedom, 0);
        assert_eq!(report.redundant_residuals, 1, "{report:?}");
    }

    /// A CONTRADICTORY system: the same parameter told to be two different things. There is no
    /// solution, so the solver settles on the least-squares compromise and stops moving — a
    /// non-zero residual norm with redundancy reported is exactly the "these constraints conflict"
    /// diagnosis.
    #[test]
    fn a_contradiction_settles_and_says_so() {
        let here = |p: &[f64]| p[0] - 1.0;
        let there = |p: &[f64]| p[0] - 5.0;
        let system = Closures {
            parameters: 1,
            residuals: vec![&here, &there],
        };
        let mut parameters = vec![0.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!(
            (parameters[0] - 3.0).abs() < 1e-6,
            "the midpoint: {parameters:?}"
        );
        assert!(
            report.residual_norm > 1.0,
            "and it cannot be solved: {report:?}"
        );
        assert_ne!(report.outcome, SolveOutcome::Converged);
        assert_eq!(report.redundant_residuals, 1, "{report:?}");
    }

    /// A STATIONARY POINT that is not a root: `r(x) = x² + 1` has gradient `2x(x² + 1)`, which
    /// vanishes at `x = 0` where the residual is 1 and never zero anywhere. The gradient test fires
    /// and the solver stops, and stopping there must not be reported as a solution — an
    /// over-constrained sketch reaches exactly this shape, and `Converged` would tell the author
    /// their constraints hold when they do not.
    #[test]
    fn a_stationary_point_that_is_not_a_root_stalls() {
        let never_zero = |p: &[f64]| p[0] * p[0] + 1.0;
        let system = Closures {
            parameters: 1,
            residuals: vec![&never_zero],
        };
        // Started exactly on the stationary point, so the gradient test is what stops it.
        let mut parameters = vec![0.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert_eq!(report.outcome, SolveOutcome::Stalled, "{report:?}");
        assert_eq!(report.iterations, 1, "the gradient test, not the budget");
        assert!(report.residual_norm >= 1.0, "and unsolved: {report:?}");
        // And from off the stationary point it descends into it and still refuses to call it solved.
        let mut approached = vec![0.6];
        let report = solve(&system, &mut approached, SolveSettings::default());
        assert_ne!(report.outcome, SolveOutcome::Converged, "{report:?}");
        assert!(report.residual_norm >= 1.0, "{report:?}");
    }

    /// A system with NO curvature in one direction at all — the Jacobian column is identically
    /// zero, so the direction is outside the decomposition's range and contributes nothing to the
    /// minimum-norm step. The solve lands, and the free parameter is left exactly where it was put.
    #[test]
    fn a_rank_deficient_system_leaves_its_free_direction_alone() {
        let ignores_the_second = |p: &[f64]| p[0] - 7.0;
        let system = Closures {
            parameters: 2,
            residuals: vec![&ignores_the_second],
        };
        let mut parameters = vec![0.0, 42.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!((parameters[0] - 7.0).abs() < 1e-8, "{parameters:?}");
        assert!(
            (parameters[1] - 42.0).abs() < 1e-6,
            "the direction nothing constrains was left alone: {parameters:?}"
        );
        assert_eq!(report.degrees_of_freedom, 1, "{report:?}");
    }

    /// An under-constrained system moves ONLY as far as its one residual asks, and spreads that
    /// motion evenly over the parameters that can supply it.
    ///
    /// One residual over four parameters: `JᵀJ` is four-by-four with rank one, singular by three,
    /// and asking it for a step is asking a matrix for information it does not carry. The least-norm
    /// form asks `JJᵀ` instead — one-by-one, non-singular — and answers with the shortest correction
    /// that does the job. Here the residual wants the sum to fall by four, and the shortest way to
    /// do that is one each.
    ///
    /// The old form answered by factorising the singular matrix anyway, which succeeded whenever
    /// rounding left its last pivot a hair above zero, and then divided by that hair.
    #[test]
    fn an_under_constrained_system_takes_the_shortest_correction() {
        let sum_is_ten = |p: &[f64]| p[0] + p[1] + p[2] + p[3] - 10.0;
        let system = Closures {
            parameters: 4,
            residuals: vec![&sum_is_ten],
        };
        let mut parameters = vec![5.0, 3.0, 4.0, 2.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert_eq!(report.outcome, SolveOutcome::Converged, "{report:?}");
        assert!(report.residual_norm < 1.0e-9, "{report:?}");
        for (moved, was) in parameters.iter().zip([5.0, 3.0, 4.0, 2.0]) {
            assert!(
                (moved - (was - 1.0)).abs() < 1.0e-6,
                "each parameter gave up exactly its quarter: {parameters:?}"
            );
        }
        assert_eq!(report.degrees_of_freedom, 3, "{report:?}");
    }

    /// A pivot that is only rounding dust above zero is treated as zero, so the step goes through
    /// the damped door rather than dividing by the dust.
    ///
    /// Two residuals saying the SAME thing about one parameter, and a second parameter nothing
    /// mentions. Redundancy like this is ordinary in a sketch — two relations that agree — and it
    /// leaves the normal matrix singular in exact arithmetic and a few ulps off it in floating
    /// point. What must not happen is the untouched parameter wandering: whatever the factorisation
    /// does with the direction nothing constrains, it must not be to move it.
    #[test]
    fn a_redundant_system_leaves_the_untouched_parameter_alone() {
        let says_it_once = |p: &[f64]| p[0] - 3.0;
        let says_it_again = |p: &[f64]| 2.0f64.mul_add(p[0], -6.0);
        let system = Closures {
            parameters: 2,
            residuals: vec![&says_it_once, &says_it_again],
        };
        let mut parameters = vec![0.0, 137.5];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert!((parameters[0] - 3.0).abs() < 1.0e-8, "{parameters:?}");
        assert!(
            (parameters[1] - 137.5).abs() < 1.0e-9,
            "the parameter no residual names did not drift: {parameters:?}"
        );
        assert!(report.redundant_residuals >= 1, "{report:?}");
    }

    /// A search still making real progress is NOT cut short by the improvement test.
    ///
    /// `r(x) = e⁻ˣ` has no root and shrinks geometrically: the Gauss-Newton step is exactly one
    /// every iteration, so each step takes a fixed 86% off the sum of squares forever. That is the
    /// shape the test must never fire on — steady progress, however far from done — and it runs all
    /// the way down to the residual tolerance and reports the honest `Converged`.
    /// A search still making real progress is NOT cut short by the improvement test.
    ///
    /// `x² = 2` has a root, so Gauss-Newton is Newton's method on it and converges quadratically:
    /// every step takes almost the whole remaining error, which is the shape the test must never
    /// fire on. Switching the test off changes nothing about the run — same outcome, same five
    /// iterations, same answer — which is the actual claim.
    #[test]
    fn a_search_still_making_progress_is_not_cut_short() {
        let root_of_two = |p: &[f64]| p[0].mul_add(p[0], -2.0);
        let system = Closures {
            parameters: 1,
            residuals: vec![&root_of_two],
        };

        let mut tested = vec![0.5];
        let with_the_test = solve(&system, &mut tested, SolveSettings::default());
        assert_eq!(
            with_the_test.outcome,
            SolveOutcome::Converged,
            "{with_the_test:?}"
        );
        assert!((tested[0] - std::f64::consts::SQRT_2).abs() < 1.0e-10);

        let mut untested = vec![0.5];
        let without_it = solve(
            &system,
            &mut untested,
            SolveSettings {
                improvement_tolerance: 0.0,
                ..SolveSettings::default()
            },
        );
        assert_eq!(
            with_the_test.iterations, without_it.iterations,
            "the test cost the search iterations it needed: {with_the_test:?} against {without_it:?}"
        );
        assert_eq!(tested, untested);
    }

    /// An INCOMPATIBLE system stops when it stops improving, rather than grinding out its budget.
    ///
    /// `x = 1` and `10x² = 0` cannot both hold, so there is a least-squares compromise near 0.161
    /// and no solution. A compromise with residuals this large is reached only LINEARLY — the error
    /// roughly halves per step forever instead of squaring — so the last digits cost exactly what
    /// the first ones did and buy nothing anyone can see.
    ///
    /// Eight iterations against sixty-five, and the two answers agree in their residual norm to a
    /// part in a billion. The fifty-seven extra iterations moved the parameter by 8e-6, in a basin
    /// flat enough that moving it there changed the objective by nothing — which is the whole
    /// argument for the test.
    #[test]
    fn an_incompatible_system_stops_when_it_stops_improving() {
        let wants_one = |p: &[f64]| p[0] - 1.0;
        let wants_zero = |p: &[f64]| 10.0 * p[0] * p[0];
        let system = Closures {
            parameters: 1,
            residuals: vec![&wants_one, &wants_zero],
        };

        let mut stopping = vec![0.5];
        let stopped = solve(&system, &mut stopping, SolveSettings::default());
        assert_eq!(stopped.outcome, SolveOutcome::Stalled, "{stopped:?}");
        assert!(stopped.residual_norm > 0.5, "and unsolved: {stopped:?}");

        let mut grinding = vec![0.5];
        let ground = solve(
            &system,
            &mut grinding,
            SolveSettings {
                improvement_tolerance: 0.0,
                ..SolveSettings::default()
            },
        );
        assert!(
            stopped.iterations * 4 < ground.iterations,
            "it saved next to nothing: {stopped:?} against {ground:?}"
        );
        assert!(
            (stopped.residual_norm - ground.residual_norm).abs() < 1.0e-8,
            "and the extra iterations were not free: {stopped:?} against {ground:?}"
        );
    }

    /// A system already at its solution takes no step and says so immediately.
    #[test]
    fn a_solved_system_converges_at_once() {
        let already = |p: &[f64]| p[0];
        let system = Closures {
            parameters: 1,
            residuals: vec![&already],
        };
        let mut parameters = vec![0.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert_eq!(report.outcome, SolveOutcome::Converged);
        assert_eq!(report.iterations, 1);
        assert_eq!(parameters, vec![0.0]);
    }

    /// The Jacobian by finite differences matches the analytic one to well past what the solver
    /// needs — the whole method rests on this being true.
    #[test]
    fn the_finite_difference_jacobian_matches_the_analytic_one() {
        let product = |p: &[f64]| p[0] * p[1];
        let squared = |p: &[f64]| p[0] * p[0] - p[1];
        let system = Closures {
            parameters: 2,
            residuals: vec![&product, &squared],
        };
        let at = [3.0, -2.0];
        let matrix = jacobian(&system, &at);
        let analytic = [at[1], at[0], 2.0 * at[0], -1.0];
        for (index, expected) in analytic.iter().enumerate() {
            assert!(
                (matrix[index] - expected).abs() < 1e-7,
                "entry {index}: {} vs {expected}",
                matrix[index]
            );
        }
    }

    /// Rank counts independent ROWS, sees through a dependent one, and answers zero for a matrix
    /// with nothing in it.
    #[test]
    fn rank_counts_independent_rows() {
        assert_eq!(rank(&[1.0, 0.0, 0.0, 1.0], 2, 2), 2);
        // The second row is twice the first.
        assert_eq!(rank(&[1.0, 2.0, 2.0, 4.0], 2, 2), 1);
        assert_eq!(rank(&[0.0, 0.0, 0.0, 0.0], 2, 2), 0);
        // More rows than columns: the rank cannot exceed either.
        assert_eq!(rank(&[1.0, 2.0, 3.0], 3, 1), 1);
        assert_eq!(rank(&[], 0, 0), 0);
    }
}
