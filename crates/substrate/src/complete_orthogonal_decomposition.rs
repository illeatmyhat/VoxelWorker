//! The minimum-norm least-squares solution of a dense, possibly RANK-DEFICIENT system, by
//! complete orthogonal decomposition — LAPACK's `xGELSY`, written out.
//!
//! The question is `A x ≈ b` where `A` is `rows × columns` and may be any shape and any rank. Three
//! separate things can be true of it at once, and a solver that answers only the first is unusable
//! on a constraint system:
//!
//! - **More rows than columns**, and no `x` makes the residual zero. The answer is the `x`
//!   minimizing `‖Ax − b‖`.
//! - **Fewer rows than columns**, and infinitely many `x` make it zero. The answer is the SHORTEST
//!   of them.
//! - **Rank-deficient either way**, so the minimizing set is a whole affine subspace. The answer is
//!   again the shortest member — and picking it is a GAUGE CHOICE, made here on purpose rather than
//!   left to whatever a factorisation's rounding error contains.
//!
//! All three are the same answer, `x = A⁺b`, and this computes it.
//!
//! ## Why not the normal equations
//!
//! `AᵀA x = Aᵀb` is one Cholesky away and answers the first case, which is why it is what everyone
//! writes first. Its defect is that forming `AᵀA` SQUARES the condition number: a matrix carrying
//! six digits of conditioning loss becomes one carrying twelve, and a `f64` has sixteen to spend.
//! Past that the factorisation fails outright and the usual repair is to damp — solve `AᵀA + λI`
//! instead — which does succeed, and answers a question nobody asked, with the perturbation landing
//! hardest in exactly the directions the data pinned down least.
//!
//! Working on `A` itself through orthogonal transformations costs the same order of arithmetic and
//! never squares anything, because an orthogonal matrix has condition number one.
//!
//! ## The decomposition
//!
//! Two orthogonal stages, in the order LAPACK does them:
//!
//! 1. **Householder QR with column pivoting** (`xGEQP3`): `A P = Q R`, choosing at each step the
//!    remaining column with the largest residual norm. That ordering is what makes the
//!    factorisation RANK-REVEALING — the diagonal of `R` comes out in decreasing size, so the rank
//!    is read off as the count of diagonal entries above a tolerance instead of guessed.
//! 2. **Trapezoidal reduction from the right** (`xTZRZF`): with `R = [T₁₁ T₁₂; 0 0]` and `T₁₁` an
//!    `r × r` upper triangle, a second family of Householders `Z` annihilates `T₁₂`, leaving
//!    `R Z = [T 0]`.
//!
//! The second stage is the one that earns the name and the one a plain pivoted QR does not have.
//! Without it the natural answer is the BASIC solution — set the last `columns − r` unknowns to
//! zero and back-substitute — which is a legitimate member of the solution set but depends on the
//! pivot order, and pivot order is a discrete choice that can flip between two neighbouring inputs.
//! An answer that jumps when a tie breaks the other way is not a function anyone can build on. With
//! `Z` in hand the answer is the shortest member, and the shortest member is unique.
//!
//! ## Solving, once it is decomposed
//!
//! `A P Z = Q [T 0]`, so with `x = P Z w` the problem reads `[T 0] w ≈ Qᵀb`. Only the leading `r`
//! rows of `Qᵀb` are reachable; the rest is the irreducible residual. `T w₁ = (Qᵀb)₁` by back
//! substitution, `w₂ = 0` because `Z` is orthogonal and so `‖x‖ = ‖w‖` — zeroing the free half is
//! literally what makes the answer shortest — and then `x = P Z w`.
//!
//! ## Every matrix here is ONE buffer, and that is a measurement
//!
//! The natural Rust for a column-major matrix is a `Vec` per column, which is what this held. A
//! sketch drag runs this about a thousand times a frame on a matrix that fits in three kilobytes,
//! and a `Vec` per column made that some sixty heap round-trips per call: `factored`, `working`, and
//! a fresh direction vector for every Householder step. Counted on an arc slot swept round its own
//! centre, one drag frame made **twenty-nine thousand allocations and moved 4.8 MiB** — a quarter of
//! a gigabyte a second of allocator traffic, for one slot.
//!
//! Flattening them cost fifteen percent of the frame and the pivot swap, which used to be two
//! pointers and is now `rows` moves. It is worth being exact about what it did NOT buy: the
//! allocation count only halved, because most of what remained was never in this file.
//!
//! **Nothing about the arithmetic changed, and that is the property to preserve.** Every loop runs
//! in the order it ran in, and no summation was re-associated — a storage change invites re-nesting
//! a loop, and re-nesting a loop reorders a reduction, and a reordered reduction moves the last bits
//! of the answer. Checked as an identity rather than a tolerance: over eight hundred solves of a
//! live drag, every point position and every field of every solve report came back bit for bit.

/// The minimum-norm least-squares solution, and the rank the decomposition found on the way.
#[derive(Debug, Clone, PartialEq)]
pub struct LeastSquaresSolution {
    /// `A⁺b`: among the minimizers of `‖Ax − b‖`, the one of smallest norm.
    pub solution: Vec<f64>,
    /// How many of `A`'s columns the data actually pinned down — the count of `R` diagonal entries
    /// above [`rank_tolerance`](minimum_norm_least_squares) times the largest.
    pub rank: usize,
}

/// Solve `A x ≈ b` for the minimum-norm least-squares `x`, for `A` of any shape and any rank.
///
/// `matrix` is `rows × columns` row-major. `rank_tolerance` is RELATIVE to the largest diagonal of
/// `R`, so it is a statement about significant digits rather than about the caller's units; the
/// conventional value is a small multiple of the machine epsilon.
///
/// `None` when the shapes disagree with the slices handed in, which is a caller bug rather than a
/// numerical outcome. A zero matrix is not that: it has rank zero and the zero solution.
#[must_use]
pub fn minimum_norm_least_squares(
    matrix: &[f64],
    right_hand_side: &[f64],
    rows: usize,
    columns: usize,
    rank_tolerance: f64,
) -> Option<LeastSquaresSolution> {
    if matrix.len() < rows.checked_mul(columns)? || right_hand_side.len() < rows {
        return None;
    }
    if rows == 0 || columns == 0 {
        return Some(LeastSquaresSolution {
            solution: vec![0.0; columns],
            rank: 0,
        });
    }
    // COLUMN-major for the factorisation, in ONE buffer. A Householder reflection from the left
    // mixes rows within one column and leaves the columns independent, so a column is the run the
    // inner loop walks — and in a row-major matrix that run is strided, which costs a bounds check
    // and a cache line per element. Held this way it is a contiguous slice and the pivot norms are
    // plain dot products.
    //
    // One buffer and not a `Vec` per column, which is what this held before: nothing about the
    // arithmetic changes, and a drag frame runs this a thousand times. See the module header.
    let mut factored = vec![0.0; columns.saturating_mul(rows)];
    for (column, run) in factored.chunks_exact_mut(rows).enumerate() {
        for (slot, row) in run.iter_mut().zip(matrix.chunks_exact(columns)) {
            *slot = row.get(column).copied().unwrap_or_default();
        }
    }
    let mut projected: Vec<f64> = right_hand_side.iter().take(rows).copied().collect();
    // The one scratch every reflection writes its direction into, sized once for the longest run.
    let mut direction: Vec<f64> = Vec::with_capacity(rows);
    let ordering =
        pivoted_householder_qr(&mut factored, &mut projected, rows, columns, &mut direction);
    let rank = revealed_rank(&factored, rows, columns, rows.min(columns), rank_tolerance);
    // Back to row-major for what is left. The trapezoidal stage and the back substitution both walk
    // ROWS, both touch only the leading `rank` of them, and both are small enough that the transpose
    // costs less than the strides would.
    let mut working = vec![0.0; rank.saturating_mul(columns)];
    for (row, line) in working.chunks_exact_mut(columns).enumerate() {
        for (slot, column) in line.iter_mut().zip(factored.chunks_exact(rows)) {
            *slot = column.get(row).copied().unwrap_or_default();
        }
    }
    let annihilated = annihilate_the_trailing_block(&mut working, rank, columns);
    let mut weights = back_substitute(&working, &projected, rank, columns);
    for reflector in &annihilated {
        apply_reflector(&mut weights, reflector, rank, columns);
    }
    let mut solution = vec![0.0; columns];
    for (slot, weight) in ordering.iter().zip(weights.iter()) {
        if let Some(target) = solution.get_mut(*slot) {
            *target = *weight;
        }
    }
    Some(LeastSquaresSolution { solution, rank })
}

/// A Householder reflection `I − beta v vᵀ` acting on a SCATTERED set of indices: column `step`,
/// then the whole trailing block `rank..columns`, and nothing between.
///
/// Scattered because that is the shape the trapezoidal stage's reflections have — LAPACK stores it
/// implicitly and pays for it in index arithmetic every time it is applied. Here the run is named by
/// the one index that varies, and the `rank` and `columns` that fix the rest come from the caller
/// that already knows them, so a reflector carries no index vector of its own.
struct Reflector {
    step: usize,
    direction: Vec<f64>,
    beta: f64,
}

/// The indices a reflector acts on, in the order its direction is stored in.
fn reflector_indices(step: usize, rank: usize, columns: usize) -> impl Iterator<Item = usize> {
    core::iter::once(step).chain(rank..columns)
}

/// `y ← y − beta (vᵀy) v`, over the indices the reflector names.
fn apply_reflector(vector: &mut [f64], reflector: &Reflector, rank: usize, columns: usize) {
    let mut projection = 0.0;
    for (index, component) in
        reflector_indices(reflector.step, rank, columns).zip(reflector.direction.iter())
    {
        projection += vector.get(index).copied().unwrap_or_default() * component;
    }
    let scale = reflector.beta * projection;
    if scale == 0.0 {
        return;
    }
    for (index, component) in
        reflector_indices(reflector.step, rank, columns).zip(reflector.direction.iter())
    {
        if let Some(slot) = vector.get_mut(index) {
            *slot = (-scale).mul_add(*component, *slot);
        }
    }
}

/// The scattered counterpart of [`reflection_below`], for the trapezoidal stage: the reflection
/// taking `values` to `(∓‖values‖, 0, …, 0)`, tagged with the column it starts at.
fn reflection_onto_the_first_axis(step: usize, values: &[f64]) -> Option<Reflector> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    let leading = values.first().copied().unwrap_or_default();
    let alpha = if leading >= 0.0 { -norm } else { norm };
    let mut direction = values.to_vec();
    if let Some(slot) = direction.first_mut() {
        *slot -= alpha;
    }
    let squared: f64 = direction.iter().map(|value| value * value).sum();
    if squared <= 0.0 {
        return None;
    }
    Some(Reflector {
        step,
        direction,
        beta: 2.0 / squared,
    })
}

/// Householder QR with column pivoting, in place over a COLUMN-major `factored`: it leaves holding
/// `R`, `projected` leaves holding `Qᵀb`, and the return is where each of `R`'s columns came from in
/// the original matrix.
///
/// The pivot is the remaining column with the largest norm below the current row. Recomputed in
/// full rather than downdated: downdating is the standard optimization and it is also the standard
/// place a rank-revealing QR quietly stops revealing rank, because the downdated norms lose accuracy
/// exactly when the columns are becoming dependent — which is the only moment the rank decision is
/// close. Held column-major the recomputation is one contiguous dot product per remaining column.
fn pivoted_householder_qr(
    factored: &mut [f64],
    projected: &mut [f64],
    rows: usize,
    columns: usize,
    direction: &mut Vec<f64>,
) -> Vec<usize> {
    let mut ordering: Vec<usize> = (0..columns).collect();
    for step in 0..rows.min(columns) {
        let pivot = widest_remaining_column(factored, rows, columns, step);
        if pivot != step {
            swap_columns(factored, rows, step, pivot);
            ordering.swap(step, pivot);
        }
        let Some(beta) = reflection_below(factored.chunks_exact(rows).nth(step), step, direction)
        else {
            continue;
        };
        for column in factored.chunks_exact_mut(rows).take(columns).skip(step) {
            reflect_a_run(column.get_mut(step..), direction, beta);
        }
        reflect_a_run(projected.get_mut(step..), direction, beta);
    }
    ordering
}

/// Exchange two columns of a flat column-major buffer — the pivot swap, which the `Vec`-per-column
/// shape got for two pointers and this one pays `rows` moves for. Cheaper than the allocation it
/// replaces, and it is `rows.min(columns)` of them against a thousand factorisations a frame.
fn swap_columns(factored: &mut [f64], rows: usize, first: usize, second: usize) {
    let (low, high) = if first < second {
        (first, second)
    } else {
        (second, first)
    };
    if low == high {
        return;
    }
    let (before, after) = factored.split_at_mut(high.saturating_mul(rows).min(factored.len()));
    let Some(left) = before
        .get_mut(low.saturating_mul(rows)..)
        .and_then(|run| run.get_mut(..rows))
    else {
        return;
    };
    let Some(right) = after.get_mut(..rows) else {
        return;
    };
    left.swap_with_slice(right);
}

/// `y ← y − beta (vᵀy) v` over one contiguous run — the whole inner loop of the factorisation.
fn reflect_a_run(run: Option<&mut [f64]>, direction: &[f64], beta: f64) {
    let Some(run) = run else {
        return;
    };
    let projection: f64 = run
        .iter()
        .zip(direction.iter())
        .map(|(value, component)| value * component)
        .sum();
    let scale = beta * projection;
    if scale == 0.0 {
        return;
    }
    for (value, component) in run.iter_mut().zip(direction.iter()) {
        *value = (-scale).mul_add(*component, *value);
    }
}

/// The reflection taking `column[from..]` onto its first axis, as a direction and a `beta`.
///
/// The sign is chosen AWAY from the leading entry — `alpha = −sign(x₀)‖x‖` — so the subtraction
/// forming `v` never cancels. Taking the near sign is the classic way to lose every digit of a
/// nearly-aligned vector. `None` when the run is already on its axis, which is not a failure: there
/// is simply nothing to reflect.
fn reflection_below(column: Option<&[f64]>, from: usize, direction: &mut Vec<f64>) -> Option<f64> {
    let values = column?.get(from..)?;
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return None;
    }
    let leading = values.first().copied().unwrap_or_default();
    let alpha = if leading >= 0.0 { -norm } else { norm };
    direction.clear();
    direction.extend_from_slice(values);
    if let Some(slot) = direction.first_mut() {
        *slot -= alpha;
    }
    let squared: f64 = direction.iter().map(|value| value * value).sum();
    if squared <= 0.0 {
        return None;
    }
    Some(2.0 / squared)
}

/// Which of the columns from `step` on has the largest norm below row `step`.
fn widest_remaining_column(factored: &[f64], rows: usize, columns: usize, step: usize) -> usize {
    factored
        .chunks_exact(rows)
        .take(columns)
        .enumerate()
        .skip(step)
        .map(|(index, column)| {
            let norm: f64 = column
                .get(step..)
                .unwrap_or_default()
                .iter()
                .map(|value| value * value)
                .sum();
            (index, norm)
        })
        .fold(
            (step, -1.0),
            |best, here| {
                if here.1 > best.1 {
                    here
                } else {
                    best
                }
            },
        )
        .0
}

/// How many of `R`'s diagonal entries are above `tolerance` times the largest.
///
/// Counted as a PREFIX and stopped at the first entry that fails, not as a total: pivoting has
/// already put the diagonal in decreasing order, so a small entry followed by a large one would
/// mean the ordering broke down, and treating the large one as informative anyway would be reading
/// a triangle that is no longer triangular in the way the count assumes.
fn revealed_rank(
    factored: &[f64],
    rows: usize,
    columns: usize,
    limit: usize,
    tolerance: f64,
) -> usize {
    let diagonal = |index: usize| {
        // Guarded on the COLUMN count as well as the length: past the last column a flat index
        // lands in a neighbouring column rather than off the end, which is the one way this shape
        // can read something the `Vec`-per-column shape would have refused.
        if index >= columns || index >= rows {
            return 0.0;
        }
        factored
            .get(index.saturating_mul(rows).saturating_add(index))
            .copied()
            .unwrap_or_default()
            .abs()
    };
    let largest = diagonal(0);
    if largest == 0.0 {
        return 0;
    }
    (0..limit)
        .take_while(|index| diagonal(*index) > tolerance * largest)
        .count()
}

/// Annihilate `R`'s trailing `columns − rank` block from the right, returning the reflections that
/// did it in the order they must be applied to a solution vector.
///
/// Rows are taken from the BOTTOM up. Each reflection touches column `k` and the trailing block and
/// nothing between, so a row below `k` — zero in column `k` because the triangle says so, and zero
/// across the trailing block because it has already been processed — is left exactly alone. Rows
/// above `k` are disturbed and are put right when their own turn comes.
fn annihilate_the_trailing_block(
    working: &mut [f64],
    rank: usize,
    columns: usize,
) -> Vec<Reflector> {
    if rank >= columns {
        return Vec::new();
    }
    let mut applied = Vec::new();
    let mut values: Vec<f64> = Vec::with_capacity(columns.saturating_sub(rank).saturating_add(1));
    for step in (0..rank).rev() {
        let row = working.chunks_exact(columns).nth(step);
        values.clear();
        values.extend(reflector_indices(step, rank, columns).map(|index| {
            row.and_then(|entries| entries.get(index))
                .copied()
                .unwrap_or_default()
        }));
        let Some(reflector) = reflection_onto_the_first_axis(step, &values) else {
            continue;
        };
        for line in working
            .chunks_exact_mut(columns)
            .take(step.saturating_add(1))
        {
            apply_reflector(line, &reflector, rank, columns);
        }
        applied.push(reflector);
    }
    // Built bottom-up, so `Z = H_{r−1} ⋯ H_0` and a vector meets `H_0` first.
    applied.reverse();
    applied
}

/// `T w = (Qᵀb)₁` by back substitution through the leading `rank × rank` triangle, with the
/// remaining `columns − rank` entries left at zero — which is what makes the result shortest.
fn back_substitute(working: &[f64], projected: &[f64], rank: usize, columns: usize) -> Vec<f64> {
    let mut weights = vec![0.0; columns];
    for row in (0..rank).rev() {
        let mut sum = projected.get(row).copied().unwrap_or_default();
        let values = working.chunks_exact(columns).nth(row);
        for column in row.saturating_add(1)..rank {
            let coefficient = values
                .and_then(|entries| entries.get(column))
                .copied()
                .unwrap_or_default();
            sum = (-coefficient).mul_add(weights.get(column).copied().unwrap_or_default(), sum);
        }
        let diagonal = values
            .and_then(|entries| entries.get(row))
            .copied()
            .unwrap_or_default();
        if let Some(slot) = weights.get_mut(row) {
            *slot = if diagonal == 0.0 { 0.0 } else { sum / diagonal };
        }
    }
    weights
}

#[cfg(test)]
#[allow(
    clippy::arithmetic_side_effects,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1.0e-12;

    fn solved(matrix: &[f64], rhs: &[f64], rows: usize, columns: usize) -> LeastSquaresSolution {
        minimum_norm_least_squares(matrix, rhs, rows, columns, TOLERANCE).expect("well shaped")
    }

    /// `A x` for a row-major `A`, so a test can check the residual rather than the answer.
    fn times(matrix: &[f64], vector: &[f64], rows: usize, columns: usize) -> Vec<f64> {
        (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| matrix[row * columns + column] * vector[column])
                    .sum()
            })
            .collect()
    }

    /// A square, well-conditioned system: the ordinary case, solved exactly.
    #[test]
    fn a_square_system_is_solved_exactly() {
        let matrix = [2.0, 1.0, 1.0, 3.0];
        let answer = solved(&matrix, &[5.0, 10.0], 2, 2);
        assert_eq!(answer.rank, 2);
        assert!((answer.solution[0] - 1.0).abs() < 1e-12, "{answer:?}");
        assert!((answer.solution[1] - 3.0).abs() < 1e-12, "{answer:?}");
    }

    /// An OVER-determined system with no exact answer settles on the least-squares one: fitting a
    /// horizontal line through 1, 2 and 3 gives their mean.
    #[test]
    fn an_over_determined_system_gives_the_least_squares_answer() {
        let matrix = [1.0, 1.0, 1.0];
        let answer = solved(&matrix, &[1.0, 2.0, 3.0], 3, 1);
        assert_eq!(answer.rank, 1);
        assert!((answer.solution[0] - 2.0).abs() < 1e-12, "{answer:?}");
    }

    /// An UNDER-determined system takes the shortest of its infinitely many answers: `x + y + z = 3`
    /// is solved by one each, not by three-and-nothing.
    #[test]
    fn an_under_determined_system_takes_the_shortest_answer() {
        let matrix = [1.0, 1.0, 1.0];
        let answer = solved(&matrix, &[3.0], 1, 3);
        assert_eq!(answer.rank, 1);
        for value in &answer.solution {
            assert!((value - 1.0).abs() < 1e-12, "{answer:?}");
        }
    }

    /// A RANK-DEFICIENT over-determined system: the second column is twice the first, so the
    /// minimizing set is a line and the answer is the point of it nearest the origin.
    ///
    /// The basic solution a plain pivoted QR would give sets one unknown to zero and puts everything
    /// in the other; the minimum-norm one splits it in the ratio the columns ask for. Both minimize
    /// the residual identically, which is exactly why the choice has to be made deliberately.
    #[test]
    fn a_rank_deficient_system_takes_the_minimum_norm_answer() {
        // Columns (1,1) and (2,2): any x with x₀ + 2x₁ = 3 fits.
        let matrix = [1.0, 2.0, 1.0, 2.0];
        let answer = solved(&matrix, &[3.0, 3.0], 2, 2);
        assert_eq!(answer.rank, 1, "{answer:?}");
        let combination = 2.0f64.mul_add(answer.solution[1], answer.solution[0]);
        assert!((combination - 3.0).abs() < 1e-12, "it fits: {answer:?}");
        // The shortest point of `x₀ + 2x₁ = 3` is `3/5 · (1, 2)`.
        assert!((answer.solution[0] - 0.6).abs() < 1e-12, "{answer:?}");
        assert!((answer.solution[1] - 1.2).abs() < 1e-12, "{answer:?}");
    }

    /// The minimum-norm answer does not depend on the ORDER the columns arrive in — which is what
    /// the trapezoidal stage buys, and what a basic solution would fail.
    #[test]
    fn the_answer_does_not_depend_on_column_order() {
        let matrix = [1.0, 2.0, 1.0, 2.0];
        let swapped = [2.0, 1.0, 2.0, 1.0];
        let first = solved(&matrix, &[3.0, 3.0], 2, 2);
        let second = solved(&swapped, &[3.0, 3.0], 2, 2);
        assert!(
            (first.solution[0] - second.solution[1]).abs() < 1e-12,
            "{first:?} {second:?}"
        );
        assert!(
            (first.solution[1] - second.solution[0]).abs() < 1e-12,
            "{first:?} {second:?}"
        );
    }

    /// A matrix of nothing has rank zero and the zero answer, rather than a division by its own
    /// emptiness.
    #[test]
    fn a_zero_matrix_answers_zero() {
        let answer = solved(&[0.0, 0.0, 0.0, 0.0], &[1.0, 2.0], 2, 2);
        assert_eq!(answer.rank, 0);
        assert_eq!(answer.solution, vec![0.0, 0.0]);
    }

    /// The headline claim, measured: a system whose condition number is past what the normal
    /// equations can survive is still solved here.
    ///
    /// `A` has singular values around 1 and around 1e-9, so `AᵀA` has them around 1 and 1e-18 —
    /// under the double epsilon, so its Cholesky cannot succeed and the damped repair answers a
    /// different question. Working on `A` the conditioning is 1e-9, which leaves seven digits.
    #[test]
    fn a_system_the_normal_equations_could_not_survive_is_solved() {
        let tiny = 1.0e-9;
        let matrix = [1.0, 1.0, 0.0, tiny];
        let wanted = [2.0, 3.0 * tiny];
        let answer = solved(&matrix, &wanted, 2, 2);
        assert_eq!(answer.rank, 2, "{answer:?}");
        let residual = times(&matrix, &answer.solution, 2, 2);
        for (got, want) in residual.iter().zip(wanted.iter()) {
            assert!((got - want).abs() < 1e-15, "{answer:?} {residual:?}");
        }
        assert!((answer.solution[1] - 3.0).abs() < 1e-6, "{answer:?}");
    }

    /// Shapes that disagree with the slices are a caller bug and are refused rather than read past.
    #[test]
    fn a_mismatched_shape_is_refused() {
        assert!(minimum_norm_least_squares(&[1.0], &[1.0], 2, 2, TOLERANCE).is_none());
        assert!(
            minimum_norm_least_squares(&[1.0, 0.0, 0.0, 1.0], &[1.0], 2, 2, TOLERANCE).is_none()
        );
    }
}
