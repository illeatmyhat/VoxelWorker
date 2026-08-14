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
//! only question is whether the rule is stated or left to rounding error. The shortest is chosen
//! because it is UNIQUE and does not depend on the pivot order, so no tie breaking the other way
//! can move the answer.
//!
//! It is worth being exact about how much that buys, because the record first said it bought
//! everything. Measured against the basic solution — the other natural gauge, and the one that does
//! depend on pivot order — the two are within four percent on the same drag, and both collapse
//! identically when the rank tolerance is set below the noise floor. **What settles the free
//! direction usefully is DISCARDING the directions the Jacobian cannot see**, not the choice made
//! among the ones it can; that is `JACOBIAN_RANK_TOLERANCE`, and it is worth a factor of a
//! thousand where the gauge is worth a few percent.
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

    /// Write only the rows named in `rows` — ascending and without duplicates — into their own
    /// places in `into`, leaving every other entry holding whatever it held.
    ///
    /// A grouped Jacobian READS only the rows a column declared, and by default it EVALUATES the
    /// whole vector to get them: one difference over a group of nineteen columns costs nineteen
    /// rows' worth of arithmetic to use four. Answering this narrows each pass to the group's own
    /// rows, and the total across a Jacobian falls from one whole residual vector per group to one
    /// entry per non-zero of the sparsity pattern. The default answers the whole vector, which is
    /// always correct and is what a system that cannot evaluate one row without the others keeps.
    ///
    /// **A row this writes must carry the bits [`residuals`](Self::residuals) would have left
    /// there** — the grouped Jacobian's claim is bit-for-bit equality with the column-by-column
    /// one, and a partial pass that rounds differently breaks it as surely as a wrong formula
    /// would. [`first_subset_disagreement`] is the falsifier.
    fn residuals_of_rows(&self, parameters: &[f64], rows: &[usize], into: &mut [f64]) {
        let _ = rows;
        self.residuals(parameters, into);
    }

    /// Which parameters each residual row READS, if the system knows.
    ///
    /// Answering turns the finite-difference Jacobian from one residual pass per parameter into one
    /// per GROUP of structurally independent parameters — see [`ColumnGrouping`]. Answering `None`,
    /// the default, takes the column-by-column path and is always correct.
    ///
    /// **A row that reads a parameter and does not say so silently corrupts the Jacobian**, because
    /// two parameters that row sees will then be perturbed together and their effects added into
    /// one difference. Nothing here can detect that at solve time; it is the caller's claim.
    /// [`first_undeclared_read`] is the falsifier — run it over every shape the system can take,
    /// from a test, before answering anything but `None`.
    ///
    /// Declaring MORE than a row reads is safe and costs only group count, so where a read is hard
    /// to pin down, name the superset.
    fn parameter_reads(&self) -> Option<ResidualReads> {
        None
    }

    /// Which residual rows the system DIFFERENTIATES ITSELF, if any.
    ///
    /// A finite difference is two residual evaluations and a subtraction that throws half the
    /// significant digits away; a row that is LINEAR in the parameters — a fix, a coincidence, one
    /// coordinate against another — has a derivative that is a constant the system already knows.
    /// Naming those rows here moves them off the difference path entirely: they cost no residual
    /// evaluation, they are exact rather than accurate to a part in `10¹¹`, and the columns they
    /// used to conflict over stop forcing the [`ColumnGrouping`] apart, so the rows that are still
    /// differenced are differenced in fewer passes.
    ///
    /// Answering `None`, the default, differences everything and is what every system did before
    /// the seam existed.
    ///
    /// **A row named here and differentiated wrongly is a wrong step direction, not a wrong
    /// answer**, so it does not fail loudly — it makes the search wander. [`first_wrong_analytic_derivative`]
    /// is the falsifier; run it over every shape the system can take, from a test.
    fn analytic_rows(&self) -> Option<AnalyticRows> {
        None
    }

    /// Write the derivative rows named by [`analytic_rows`](Self::analytic_rows) into the row-major
    /// `residual_count × parameter_count` matrix `into`.
    ///
    /// Row `r` owns `into[r * parameter_count .. (r + 1) * parameter_count]` and must be written
    /// WHOLE, zeros included: the entries it does not write hold whatever the finite-difference
    /// pass left there. Rows not named are none of this method's business.
    fn analytic_jacobian(&self, parameters: &[f64], into: &mut [f64]) {
        let _ = (parameters, into);
    }
}

/// The residual rows a system differentiates itself, ascending and without repeats.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnalyticRows {
    rows: Vec<usize>,
}

impl AnalyticRows {
    /// Collect the rows, in any order and with any repeats.
    pub fn from_rows(rows: impl IntoIterator<Item = usize>) -> Self {
        let mut rows: Vec<usize> = rows.into_iter().collect();
        rows.sort_unstable();
        rows.dedup();
        Self { rows }
    }

    /// The rows, ascending.
    #[must_use]
    pub fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// Whether the system named no row at all, which is the same claim as answering `None`.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether one row is among them.
    #[must_use]
    pub fn contains(&self, row: usize) -> bool {
        self.rows.binary_search(&row).is_ok()
    }
}

/// Which parameters each residual row reads — the Jacobian's SPARSITY PATTERN, stated by the
/// system that owns the arithmetic rather than sampled by the solver that consumes it.
///
/// Rows are in residual order and their entries are parameter indices. Duplicates and out-of-range
/// indices are tolerated and ignored; order within a row does not matter.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidualReads {
    /// Every row's columns, rows concatenated.
    columns: Vec<usize>,
    /// Row `index` owns `columns[bounds[index]..bounds[index + 1]]`, so this is one longer than
    /// the row count.
    bounds: Vec<usize>,
}

impl ResidualReads {
    /// Collect one row's columns after another, in residual order.
    pub fn from_rows<Columns: IntoIterator<Item = usize>>(
        rows: impl IntoIterator<Item = Columns>,
    ) -> Self {
        let mut columns = Vec::new();
        let mut bounds = vec![0];
        for row in rows {
            columns.extend(row);
            bounds.push(columns.len());
        }
        Self { columns, bounds }
    }

    /// How many rows were declared. Must equal the system's
    /// [`residual_count`](ResidualSystem::residual_count) for the declaration to be used at all.
    #[must_use]
    pub const fn row_count(&self) -> usize {
        self.bounds.len().saturating_sub(1)
    }

    /// The columns one row declared, or nothing for a row past the end.
    #[must_use]
    pub fn row(&self, index: usize) -> &[usize] {
        let (Some(&start), Some(&end)) = (
            self.bounds.get(index),
            self.bounds.get(index.saturating_add(1)),
        ) else {
            return &[];
        };
        self.columns.get(start..end).unwrap_or_default()
    }
}

/// Parameters partitioned into groups no residual row reads twice — Curtis, Powell and Reid's
/// grouping, and the reason a Jacobian need not cost a pass per column.
///
/// The observation is theirs (1974): differencing along `e_j + e_k` gives, in ONE pass, the `j`
/// column for every row that reads only `j` and the `k` column for every row that reads only `k`.
/// No row reads both, so no difference mixes two derivatives, and the result is not an
/// approximation of the column-by-column Jacobian — it is the same number, bit for bit. A sketch's
/// rows are local (a distance names two points out of forty), so a nineteen-column system colors
/// into single figures and the Jacobian costs what a handful of columns used to.
///
/// The coloring is the classic greedy one over the column-intersection graph, taken LARGEST FIRST:
/// a column is placed in the first group none of whose rows it shares. Optimal coloring is
/// NP-hard and worth nothing here — the grouping only decides how many passes are spent, never
/// what they compute, so a group too many costs a pass and cannot cost an answer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColumnGrouping {
    /// The rows each column is read by, columns concatenated and each ascending.
    rows: Vec<usize>,
    /// Column `index` owns `rows[row_bounds[index]..row_bounds[index + 1]]`.
    row_bounds: Vec<usize>,
    /// The columns of each group, groups concatenated.
    columns: Vec<usize>,
    /// Group `index` owns `columns[column_bounds[index]..column_bounds[index + 1]]`.
    column_bounds: Vec<usize>,
    /// The rows any column of each group is read by, groups concatenated and each ascending.
    group_rows: Vec<usize>,
    /// Group `index` owns `group_rows[group_row_bounds[index]..group_row_bounds[index + 1]]`.
    group_row_bounds: Vec<usize>,
}

impl ColumnGrouping {
    /// Color `parameter_count` columns against the rows that read them.
    #[must_use]
    pub fn curtis_powell_reid(reads: &ResidualReads, parameter_count: usize) -> Self {
        let row_count = reads.row_count();
        let mut rows_of_column: Vec<Vec<usize>> = vec![Vec::new(); parameter_count];
        for row in 0..row_count {
            for &column in reads.row(row) {
                if let Some(bucket) = rows_of_column.get_mut(column) {
                    bucket.push(row);
                }
            }
        }
        for bucket in &mut rows_of_column {
            bucket.sort_unstable();
            bucket.dedup();
        }

        // Largest first: the column hardest to place goes while the groups are still empty. Ties
        // break on index so the coloring is the same on every run of the same drawing.
        let mut order: Vec<usize> = (0..parameter_count).collect();
        order.sort_by(|left, right| {
            let (Some(here), Some(there)) = (rows_of_column.get(*left), rows_of_column.get(*right))
            else {
                return left.cmp(right);
            };
            there.len().cmp(&here.len()).then_with(|| left.cmp(right))
        });

        let mut group_columns: Vec<Vec<usize>> = Vec::new();
        let mut group_rows: Vec<Vec<bool>> = Vec::new();
        for column in order {
            let Some(mine) = rows_of_column.get(column) else {
                continue;
            };
            let landed = group_rows.iter().position(|taken| {
                mine.iter()
                    .all(|row| !taken.get(*row).copied().unwrap_or(false))
            });
            let group = landed.unwrap_or_else(|| {
                group_columns.push(Vec::new());
                group_rows.push(vec![false; row_count]);
                group_rows.len().saturating_sub(1)
            });
            if let (Some(members), Some(taken)) =
                (group_columns.get_mut(group), group_rows.get_mut(group))
            {
                members.push(column);
                for row in mine {
                    if let Some(slot) = taken.get_mut(*row) {
                        *slot = true;
                    }
                }
            }
        }

        let mut rows = Vec::new();
        let mut row_bounds = vec![0];
        for bucket in &rows_of_column {
            rows.extend_from_slice(bucket);
            row_bounds.push(rows.len());
        }
        let mut columns = Vec::new();
        let mut column_bounds = vec![0];
        let mut group_rows = Vec::new();
        let mut group_row_bounds = vec![0];
        let mut union = Vec::new();
        for members in &group_columns {
            columns.extend_from_slice(members);
            column_bounds.push(columns.len());
            union.clear();
            for column in members {
                if let Some(bucket) = rows_of_column.get(*column) {
                    union.extend_from_slice(bucket);
                }
            }
            union.sort_unstable();
            union.dedup();
            group_rows.extend_from_slice(&union);
            group_row_bounds.push(group_rows.len());
        }
        Self {
            rows,
            row_bounds,
            columns,
            column_bounds,
            group_rows,
            group_row_bounds,
        }
    }

    /// How many residual passes a Jacobian over this grouping costs, halved: one group is one
    /// central difference, which is two passes.
    #[must_use]
    pub const fn group_count(&self) -> usize {
        self.column_bounds.len().saturating_sub(1)
    }

    /// The columns one group perturbs together.
    #[must_use]
    pub fn group(&self, index: usize) -> &[usize] {
        let (Some(&start), Some(&end)) = (
            self.column_bounds.get(index),
            self.column_bounds.get(index.saturating_add(1)),
        ) else {
            return &[];
        };
        self.columns.get(start..end).unwrap_or_default()
    }

    /// The rows any column of one group is read by, ascending — every row one central difference
    /// over that group has to evaluate, and no other.
    #[must_use]
    pub fn rows_of_group(&self, index: usize) -> &[usize] {
        let (Some(&start), Some(&end)) = (
            self.group_row_bounds.get(index),
            self.group_row_bounds.get(index.saturating_add(1)),
        ) else {
            return &[];
        };
        self.group_rows.get(start..end).unwrap_or_default()
    }

    /// The rows one column is read by, ascending.
    #[must_use]
    pub fn rows_of(&self, column: usize) -> &[usize] {
        let (Some(&start), Some(&end)) = (
            self.row_bounds.get(column),
            self.row_bounds.get(column.saturating_add(1)),
        ) else {
            return &[];
        };
        self.rows.get(start..end).unwrap_or_default()
    }
}

/// A row whose value MOVED under a parameter it did not declare reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UndeclaredRead {
    /// The residual that moved.
    pub row: usize,
    /// The parameter it moved under.
    pub column: usize,
}

/// The first row that moves under a parameter it did not declare — the falsifier for
/// [`parameter_reads`](ResidualSystem::parameter_reads), and the only thing standing between a
/// grouped Jacobian and a quietly wrong one.
///
/// Perturbs one parameter at a time by `step` scaled to the parameter's own magnitude, exactly as
/// the Jacobian does, and compares every undeclared row's BITS either side. Bits and not a
/// tolerance: the grouping's whole claim is that the grouped Jacobian equals the column-by-column
/// one exactly, and that rests on an undeclared row being not merely close but untouched.
///
/// Answers `None` for a system that declares nothing, since it is then claiming nothing.
#[must_use]
pub fn first_undeclared_read(
    system: &dyn ResidualSystem,
    parameters: &[f64],
    step: f64,
) -> Option<UndeclaredRead> {
    let reads = system.parameter_reads()?;
    let residual_count = system.residual_count();
    let mut here = vec![0.0; residual_count];
    system.residuals(parameters, &mut here);
    let mut declared = vec![false; residual_count];
    let mut moved = parameters.to_vec();
    let mut seen = vec![0.0; residual_count];
    for (column, &parameter) in parameters.iter().enumerate() {
        declared.fill(false);
        for row in 0..reads.row_count() {
            if reads.row(row).contains(&column) {
                if let Some(slot) = declared.get_mut(row) {
                    *slot = true;
                }
            }
        }
        for reach in [step, -step] {
            let Some(slot) = moved.get_mut(column) else {
                continue;
            };
            *slot = reach.mul_add(parameter.abs().max(1.0), parameter);
            system.residuals(&moved, &mut seen);
            let found = here
                .iter()
                .zip(&seen)
                .enumerate()
                .find(|(row, (stood, now))| {
                    !declared.get(*row).copied().unwrap_or(false)
                        && stood.to_bits() != now.to_bits()
                });
            if let Some((row, _)) = found {
                return Some(UndeclaredRead { row, column });
            }
        }
        if let Some(slot) = moved.get_mut(column) {
            *slot = parameter;
        }
    }
    None
}

/// A row a partial residual pass answered differently from a whole one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubsetDisagreement {
    /// The residual whose bits differed.
    pub row: usize,
    /// The subset it was asked for as part of: a group of the system's own Curtis-Powell-Reid
    /// grouping, or `None` where the row was asked for on its own.
    pub group: Option<usize>,
}

/// The first row a partial residual pass gets wrong — the falsifier for
/// [`residuals_of_rows`](ResidualSystem::residuals_of_rows).
///
/// Asks for every group of the system's own grouping, and then for every row on its own, and
/// compares the bits against a whole-vector pass at the rows requested. Bits and not a tolerance,
/// for the same reason [`first_undeclared_read`] uses them: the grouped Jacobian claims to equal
/// the column-by-column one exactly, and a partial pass that reassociates a sum has broken that
/// claim even where it is more accurate.
///
/// Answers `None` for a system that has not overridden the default, since it is then claiming
/// nothing — but it cannot tell, so it does the work either way and simply finds nothing.
#[must_use]
pub fn first_subset_disagreement(
    system: &dyn ResidualSystem,
    parameters: &[f64],
) -> Option<SubsetDisagreement> {
    let residual_count = system.residual_count();
    let mut whole = vec![0.0; residual_count];
    system.residuals(parameters, &mut whole);
    let mut partial = vec![0.0; residual_count];
    let grouping = JacobianPlan::for_system(system).grouping;
    let groups = grouping.as_ref().map_or(0, ColumnGrouping::group_count);
    for group in 0..groups {
        let asked = grouping
            .as_ref()
            .map_or(&[][..], |grouping| grouping.rows_of_group(group));
        if let Some(row) = disagreeing_row(system, parameters, asked, &whole, &mut partial) {
            return Some(SubsetDisagreement {
                row,
                group: Some(group),
            });
        }
    }
    for row in 0..residual_count {
        let asked = [row];
        if let Some(row) = disagreeing_row(system, parameters, &asked, &whole, &mut partial) {
            return Some(SubsetDisagreement { row, group: None });
        }
    }
    None
}

/// The first of `asked` a partial pass answers with different bits than `whole` holds.
fn disagreeing_row(
    system: &dyn ResidualSystem,
    parameters: &[f64],
    asked: &[usize],
    whole: &[f64],
    partial: &mut [f64],
) -> Option<usize> {
    if asked.is_empty() {
        return None;
    }
    partial.fill(f64::NAN);
    system.residuals_of_rows(parameters, asked, partial);
    asked.iter().copied().find(|row| {
        let (Some(&stood), Some(&now)) = (whole.get(*row), partial.get(*row)) else {
            return false;
        };
        stood.to_bits() != now.to_bits()
    })
}

/// An analytic derivative that does not agree with the difference it replaced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WrongDerivative {
    /// The residual whose row it is in.
    pub row: usize,
    /// The parameter it is the derivative with respect to.
    pub column: usize,
    /// What the system said the derivative is.
    pub analytic: f64,
    /// What a central difference of the residual says it is.
    pub differenced: f64,
}

/// The first analytic derivative that disagrees with a central difference of the same residual —
/// the falsifier for [`analytic_rows`](ResidualSystem::analytic_rows).
///
/// A tolerance and not bits, unlike the other two falsifiers here, and the difference is the point:
/// a correct analytic derivative is EXPECTED to differ from the difference that stood in for it, by
/// about the difference's own error. What is being checked is that the two agree to roughly that
/// error and no worse — a sign flip, a swapped pair of columns, a factor of two, a derivative taken
/// with respect to the wrong parameter. `tolerance` is applied as
/// `|analytic − differenced| <= tolerance * (1 + |differenced|)`, so it reads as a relative error on
/// a large derivative and an absolute one on a derivative near zero. A central difference of a
/// well-scaled residual carries about eleven digits, so `1e-6` catches every structural mistake
/// while leaving room for a residual that is merely awkward.
///
/// Answers `None` for a system that names no analytic row, since it is then claiming nothing.
#[must_use]
pub fn first_wrong_analytic_derivative(
    system: &dyn ResidualSystem,
    parameters: &[f64],
    tolerance: f64,
) -> Option<WrongDerivative> {
    let named = system.analytic_rows()?;
    let parameter_count = system.parameter_count();
    let differenced = jacobian_column_by_column(system, parameters);
    let mut analytic = differenced.clone();
    // Poisoned first, so a declared row the system only half writes is caught as loudly as one it
    // writes wrongly.
    for row in named.rows() {
        for column in 0..parameter_count {
            if let Some(slot) =
                analytic.get_mut(row.saturating_mul(parameter_count).saturating_add(column))
            {
                *slot = f64::NAN;
            }
        }
    }
    system.analytic_jacobian(parameters, &mut analytic);
    for row in named.rows() {
        for column in 0..parameter_count {
            let index = row.saturating_mul(parameter_count).saturating_add(column);
            let (Some(&said), Some(&saw)) = (analytic.get(index), differenced.get(index)) else {
                continue;
            };
            if (said - saw).abs() <= tolerance * (1.0 + saw.abs()) {
                continue;
            }
            return Some(WrongDerivative {
                row: *row,
                column,
                analytic: said,
                differenced: saw,
            });
        }
    }
    None
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
    // How the Jacobian will be taken is a fact about the SHAPE of the system, so it is settled once
    // and reused for every Jacobian the search takes. The residuals at the current parameters are
    // already in hand each time one is needed, which is the one extra pass the grouped path would
    // otherwise cost — see `jacobian_in_groups` for what it is for.
    let plan = JacobianPlan::for_system(system);
    let mut jacobian_matrix = jacobian_at(system, parameters, &residuals, &plan);

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
            jacobian_matrix = jacobian_at(system, parameters, &residuals, &plan);
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
///
/// A system that declares its [`parameter_reads`](ResidualSystem::parameter_reads) is differenced a
/// GROUP of columns at a time instead, which is the same matrix in fewer passes, and rows it names
/// in [`analytic_rows`](ResidualSystem::analytic_rows) are not differenced at all. Reaching those
/// paths from here costs one extra residual pass to learn where the system stands; inside [`solve`]
/// that value is already known and is threaded through instead.
#[must_use]
pub fn jacobian(system: &dyn ResidualSystem, parameters: &[f64]) -> Vec<f64> {
    let plan = JacobianPlan::for_system(system);
    let mut here = vec![0.0; system.residual_count()];
    if plan.grouping.is_some() {
        system.residuals(parameters, &mut here);
    }
    jacobian_at(system, parameters, &here, &plan)
}

/// How one system's Jacobian is taken: which rows it differentiates itself, and how the rest are
/// coloured for central differences.
///
/// Both are facts about the SHAPE of the system rather than about where it stands, so they are
/// settled once and reused for every Jacobian a search takes.
#[derive(Debug, Clone, Default)]
struct JacobianPlan {
    /// The rows the system writes itself, if it named any.
    analytic: Option<AnalyticRows>,
    /// The colouring of the columns over the rows that are still DIFFERENCED — the analytic rows
    /// are left out of it, so a column two of them share no longer forces a group apart.
    grouping: Option<ColumnGrouping>,
}

impl JacobianPlan {
    /// Read the system's own declarations.
    ///
    /// A reads-set with the wrong number of rows is REFUSED rather than padded. Rows are matched to
    /// residuals by position, so one row too few is not a smaller claim — it is every later row's
    /// claim attached to the wrong residual, which is precisely the silent corruption the grouping
    /// has to be incapable of.
    fn for_system(system: &dyn ResidualSystem) -> Self {
        let residual_count = system.residual_count();
        let analytic = system.analytic_rows().filter(|named| {
            !named.is_empty() && named.rows().iter().all(|row| *row < residual_count)
        });
        let grouping = system
            .parameter_reads()
            .filter(|reads| reads.row_count() == residual_count)
            .map(|reads| {
                let differenced = analytic.as_ref().map_or_else(
                    || reads.clone(),
                    |named| {
                        ResidualReads::from_rows((0..residual_count).map(|row| {
                            if named.contains(row) {
                                Vec::new()
                            } else {
                                reads.row(row).to_vec()
                            }
                        }))
                    },
                );
                ColumnGrouping::curtis_powell_reid(&differenced, system.parameter_count())
            });
        Self { analytic, grouping }
    }
}

/// The Jacobian at `parameters`, given the residuals there and how the system asked to be
/// differentiated.
///
/// The analytic rows go on LAST, over whatever the difference pass left in them. A row the system
/// writes itself is worth nothing if a difference can land on top of it, and the difference pass has
/// three ways of touching a row it was not asked about — the reconstruction for a non-finite value,
/// a column-by-column pass that cannot narrow, a stale buffer entry — so the order is the guarantee
/// rather than an optimisation.
fn jacobian_at(
    system: &dyn ResidualSystem,
    parameters: &[f64],
    here: &[f64],
    plan: &JacobianPlan,
) -> Vec<f64> {
    let mut matrix = plan.grouping.as_ref().map_or_else(
        || jacobian_column_by_column(system, parameters),
        |grouping| jacobian_in_groups(system, parameters, here, grouping),
    );
    if plan.analytic.is_some() {
        system.analytic_jacobian(parameters, &mut matrix);
    }
    matrix
}

/// The Jacobian one GROUP of columns per central difference, which is the same matrix bit for bit.
///
/// Perturbing a whole group at once works because no row reads two of its columns: a row that reads
/// `j` sees a `moved` vector differing from the column-by-column one only in coordinates it does
/// not touch, so its two evaluations come back with identical bits and its entry is the identical
/// quotient. That is the entire argument, and it stands or falls on the reads-set being honest —
/// [`first_undeclared_read`] is what makes that checkable.
///
/// **The rows a group does not touch are the subtle half.** Column by column, a row that ignores
/// `j` still gets an entry: `(v − v) / 2h`, which is zero for a finite `v` and NaN for one that is
/// not. Where every residual and every parameter is finite that is a zero, and the matrix is
/// already zero, so nothing is written and identity is free. Where something has gone non-finite —
/// a coordinate overflowed, a parameter arrived as NaN — the quotient is reconstructed from `here`
/// rather than assumed, because a solve that has wandered off the finite numbers must produce the
/// same garbage it always did rather than a quietly different garbage.
///
/// Each pass asks for only the rows the group's columns are read by, through
/// [`residuals_of_rows`](ResidualSystem::residuals_of_rows), whose default answers the whole vector
/// anyway. Nothing else in the entries the pass produces changes: the rows it does not ask for are
/// exactly the rows it never reads.
fn jacobian_in_groups(
    system: &dyn ResidualSystem,
    parameters: &[f64],
    here: &[f64],
    grouping: &ColumnGrouping,
) -> Vec<f64> {
    let parameter_count = system.parameter_count();
    let residual_count = system.residual_count();
    let mut matrix = vec![0.0; residual_count.saturating_mul(parameter_count)];
    let mut moved = parameters.to_vec();
    let mut ahead = vec![0.0; residual_count];
    let mut behind = vec![0.0; residual_count];
    let mut steps: Vec<f64> = Vec::new();
    let everything_is_finite = here
        .iter()
        .chain(parameters.iter())
        .all(|value| value.is_finite());
    for group in 0..grouping.group_count() {
        let columns = grouping.group(group);
        steps.clear();
        steps.extend(columns.iter().map(|column| {
            parameters
                .get(*column)
                .map_or(0.0, |parameter| DIFFERENCE_STEP * parameter.abs().max(1.0))
        }));
        // Only the group's own rows are ever read out of `ahead` and `behind` below, so only they
        // are evaluated. A group no row reads is not evaluated at all.
        let touched = grouping.rows_of_group(group);
        for forward in [true, false] {
            for (column, step) in columns.iter().zip(&steps) {
                let Some(&parameter) = parameters.get(*column) else {
                    continue;
                };
                if let Some(slot) = moved.get_mut(*column) {
                    *slot = if forward {
                        parameter + *step
                    } else {
                        parameter - *step
                    };
                }
            }
            if !touched.is_empty() {
                system.residuals_of_rows(
                    &moved,
                    touched,
                    if forward { &mut ahead } else { &mut behind },
                );
            }
        }
        for column in columns {
            let Some(&parameter) = parameters.get(*column) else {
                continue;
            };
            if let Some(slot) = moved.get_mut(*column) {
                *slot = parameter;
            }
        }
        for (column, step) in columns.iter().zip(&steps) {
            for row in grouping.rows_of(*column) {
                let (Some(&ahead_value), Some(&behind_value)) = (ahead.get(*row), behind.get(*row))
                else {
                    continue;
                };
                if let Some(slot) =
                    matrix.get_mut(row.saturating_mul(parameter_count).saturating_add(*column))
                {
                    *slot = (ahead_value - behind_value) / (2.0 * step);
                }
            }
            if everything_is_finite {
                continue;
            }
            let mut read = grouping.rows_of(*column).iter().peekable();
            for (row, value) in here.iter().enumerate().take(residual_count) {
                if read.peek() == Some(&&row) {
                    let _ = read.next();
                    continue;
                }
                if let Some(slot) =
                    matrix.get_mut(row.saturating_mul(parameter_count).saturating_add(*column))
                {
                    // `value - value` is the point, not a typo for zero: it is what the
                    // column-by-column difference computes for a row that ignores this column,
                    // and it is a NaN rather than a zero when the row is not finite.
                    #[allow(clippy::eq_op)]
                    {
                        *slot = (value - value) / (2.0 * step);
                    }
                }
            }
        }
    }
    matrix
}

/// The Jacobian one column per central difference: the always-correct path, and what a grouped one
/// has to reproduce exactly.
fn jacobian_column_by_column(system: &dyn ResidualSystem, parameters: &[f64]) -> Vec<f64> {
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
/// drawing has a whole subspace of equally good steps — on a slot's third pass, 16 residual rows
/// against 19 parameters at rank 12, the subspace is seven-dimensional — and the shortest step is
/// the one that leaves every parameter no relation names nearest where the author put it. Unlike a
/// basic solution it does not depend on the pivot order, so no tie breaking the other way can move
/// the drawing.
///
/// That is a reason to keep it, not the reason the drag is smooth. Swapping it for the basic
/// solution changes the worst gain from 1.52 to 1.45 and changes nothing else; what removed the
/// swinging was truncating the noise directions out of the system, and both gauges are equally
/// useless without it. See [`JACOBIAN_RANK_TOLERANCE`].
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
/// gap.
///
/// **This one constant is the whole of the fix**, which is not what ADR 0047 first claimed and is
/// worth stating where someone might change it. Sweeping it over a five-heading walk, worst gain as
/// a multiple of the cursor step:
///
/// | tolerance | worst gain | what fails |
/// | --------- | ---------: | ---------- |
/// | `1e-3`    |          — | the drag goes DEAD — real directions truncated away |
/// | `1e-4`    |       1.52 | (suite: one test fails by `1e-5`) |
/// | `1e-6`    |       1.52 | nothing |
/// | `1e-7`    |       1.52 | nothing — **here**, the middle of the band |
/// | `1e-8`    |       1.52 | nothing |
/// | `1e-10`   |       3.95 | noise starting to show |
/// | `1e-12`   |       1556 | following the noise outright |
///
/// Failure is two-sided and the band is narrow, so the value belongs in the middle of it rather
/// than at an end: the workspace suite is green from `1e-8` to `1e-6` and red at `1e-5`, and the
/// walk degrades at `1e-10`. A power of ten of margin each way is all there is.
///
/// What does NOT matter, measured the same way: which member of the surviving solution set is
/// picked. Taking the basic solution instead of the shortest gives 1.45 against 1.52 and fails
/// identically at `1e-12`. Weighting the norm by column size — the textbook equilibration — is
/// strictly worse the more of it is applied, because scaling the columns to equal size destroys the
/// very ordering the rank-revealing pivot uses to sort the noise directions last.
const JACOBIAN_RANK_TOLERANCE: f64 = 1.0e-7;

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

    /// A residual system that also states which parameters each row reads, truthfully or not.
    struct Declared<'a> {
        parameters: usize,
        residuals: Vec<Residual<'a>>,
        reads: Vec<Vec<usize>>,
    }

    impl ResidualSystem for Declared<'_> {
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
        fn parameter_reads(&self) -> Option<ResidualReads> {
            Some(ResidualReads::from_rows(self.reads.iter().cloned()))
        }
    }

    /// The coloring's one invariant: no group holds two columns that some row reads together.
    /// Everything the grouped Jacobian claims rests on this and on nothing else.
    #[test]
    fn no_group_holds_two_columns_one_row_reads() {
        // A chain of five points, each row naming a neighbouring pair, plus one row that reads
        // everything — the shape that forces the greedy to spread.
        let reads = ResidualReads::from_rows(vec![
            vec![0, 1],
            vec![1, 2],
            vec![2, 3],
            vec![3, 4],
            vec![0, 4],
            vec![0, 1, 2, 3, 4],
        ]);
        let grouping = ColumnGrouping::curtis_powell_reid(&reads, 5);
        // The row that reads everything makes every pair conflict, so the only valid coloring is
        // one column per group.
        assert_eq!(grouping.group_count(), 5);
        for group in 0..grouping.group_count() {
            for (position, column) in grouping.group(group).iter().enumerate() {
                for other in grouping.group(group).iter().skip(position + 1) {
                    for row in 0..reads.row_count() {
                        let named = reads.row(row);
                        assert!(
                            !(named.contains(column) && named.contains(other)),
                            "row {row} reads both {column} and {other}, and they share a group"
                        );
                    }
                }
            }
        }
    }

    /// Structurally independent columns collapse into ONE group, which is the whole point: five
    /// parameters, five rows that each read one, one central difference.
    #[test]
    fn independent_columns_share_a_single_group() {
        let reads = ResidualReads::from_rows(vec![vec![0], vec![1], vec![2], vec![3], vec![4]]);
        let grouping = ColumnGrouping::curtis_powell_reid(&reads, 5);
        assert_eq!(grouping.group_count(), 1);
        let mut members = grouping.group(0).to_vec();
        members.sort_unstable();
        assert_eq!(members, vec![0, 1, 2, 3, 4]);
    }

    /// A column no row reads still gets a group slot, and one that is read gets its rows back.
    #[test]
    fn a_column_nothing_reads_still_has_a_place() {
        let reads = ResidualReads::from_rows(vec![vec![0], vec![0, 2]]);
        let grouping = ColumnGrouping::curtis_powell_reid(&reads, 4);
        assert_eq!(grouping.rows_of(0), &[0, 1]);
        assert_eq!(grouping.rows_of(1), &[] as &[usize]);
        assert_eq!(grouping.rows_of(2), &[1]);
        let placed: usize = (0..grouping.group_count())
            .map(|group| grouping.group(group).len())
            .sum();
        assert_eq!(placed, 4, "every column is placed exactly once");
    }

    /// The claim the whole change rests on: grouped and column-by-column agree BIT FOR BIT, not
    /// to a tolerance. A tolerance would pass on a coloring that quietly mixed two derivatives.
    #[test]
    fn a_grouped_jacobian_is_the_column_by_column_one_bit_for_bit() {
        let first = |p: &[f64]| p[0] * p[0] + (3.0 * p[1]).sin();
        let second = |p: &[f64]| p[2].exp() - p[3];
        let third = |p: &[f64]| (p[4] - p[0]).hypot(p[5] + 1.0);
        let fourth = |p: &[f64]| p[1] * p[2] * p[4];
        let system = Declared {
            parameters: 6,
            residuals: vec![&first, &second, &third, &fourth],
            reads: vec![vec![0, 1], vec![2, 3], vec![4, 0, 5], vec![1, 2, 4]],
        };
        let at = [0.7, -1.3, 0.25, 4.0, -0.5, 2.75];
        let grouping = ColumnGrouping::curtis_powell_reid(
            &system.parameter_reads().expect("declared"),
            system.parameter_count(),
        );
        assert!(
            grouping.group_count() < system.parameter_count(),
            "the grouping bought nothing: {} groups",
            grouping.group_count()
        );
        let mut here = vec![0.0; system.residual_count()];
        system.residuals(&at, &mut here);
        let grouped = jacobian_in_groups(&system, &at, &here, &grouping);
        let column_by_column = jacobian_column_by_column(&system, &at);
        for (index, (one, other)) in grouped.iter().zip(&column_by_column).enumerate() {
            assert_eq!(
                one.to_bits(),
                other.to_bits(),
                "entry {index}: {one} against {other}"
            );
        }
    }

    /// And it stays bit for bit where a residual has gone INFINITE, which is the one place the
    /// two paths could have disagreed for free: column by column an untouched row differences to
    /// `inf - inf`, and a grouped pass that assumed zero there would answer 0 instead of NaN.
    #[test]
    fn grouping_matches_column_by_column_where_a_row_is_not_finite() {
        let finite = |p: &[f64]| p[0] * 2.0;
        let infinite = |p: &[f64]| f64::INFINITY * p[1].signum();
        let system = Declared {
            parameters: 3,
            residuals: vec![&finite, &infinite],
            reads: vec![vec![0], vec![1]],
        };
        let at = [1.5, 2.5, -3.0];
        let grouping = ColumnGrouping::curtis_powell_reid(
            &system.parameter_reads().expect("declared"),
            system.parameter_count(),
        );
        assert_eq!(grouping.group_count(), 1, "all three columns differ freely");
        let mut here = vec![0.0; system.residual_count()];
        system.residuals(&at, &mut here);
        let grouped = jacobian_in_groups(&system, &at, &here, &grouping);
        let column_by_column = jacobian_column_by_column(&system, &at);
        for (index, (one, other)) in grouped.iter().zip(&column_by_column).enumerate() {
            assert_eq!(
                one.to_bits(),
                other.to_bits(),
                "entry {index}: {one} against {other}"
            );
        }
    }

    /// A declared system that can also evaluate a subset of its rows, and counts the rows it
    /// evaluated either way. `truthful` off makes one row come back a single ulp different when it
    /// is asked for on its own, which is the mistake the subset falsifier exists to catch.
    struct Sparse<'a> {
        parameters: usize,
        residuals: Vec<Residual<'a>>,
        reads: Vec<Vec<usize>>,
        rows_evaluated: std::cell::Cell<usize>,
        truthful: bool,
    }

    impl<'a> Sparse<'a> {
        fn new(parameters: usize, residuals: Vec<Residual<'a>>, reads: Vec<Vec<usize>>) -> Self {
            Self {
                parameters,
                residuals,
                reads,
                rows_evaluated: std::cell::Cell::new(0),
                truthful: true,
            }
        }
    }

    impl ResidualSystem for Sparse<'_> {
        fn parameter_count(&self) -> usize {
            self.parameters
        }
        fn residual_count(&self) -> usize {
            self.residuals.len()
        }
        fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
            self.rows_evaluated
                .set(self.rows_evaluated.get() + self.residuals.len());
            for (slot, residual) in into.iter_mut().zip(&self.residuals) {
                *slot = residual(parameters);
            }
        }
        fn residuals_of_rows(&self, parameters: &[f64], rows: &[usize], into: &mut [f64]) {
            self.rows_evaluated
                .set(self.rows_evaluated.get() + rows.len());
            for row in rows {
                let (Some(residual), Some(slot)) = (self.residuals.get(*row), into.get_mut(*row))
                else {
                    continue;
                };
                *slot = residual(parameters);
                if !self.truthful && rows.len() == 1 {
                    *slot = f64::from_bits(slot.to_bits() ^ 1);
                }
            }
        }
        fn parameter_reads(&self) -> Option<ResidualReads> {
            Some(ResidualReads::from_rows(self.reads.iter().cloned()))
        }
    }

    /// A chain of rows over ten columns, sparse the way a sketch is: the fixture the row-narrowing
    /// claims are measured on.
    fn a_sparse_chain<'a>() -> Sparse<'a> {
        // Ten parameters. Each row reads a neighbouring pair, and two rows read a wider spread.
        Sparse::new(
            10,
            vec![
                &|p: &[f64]| p[0] * p[1] - 1.0,
                &|p: &[f64]| p[1] + p[2].sin(),
                &|p: &[f64]| p[2] * p[3] - 2.0,
                &|p: &[f64]| p[3].hypot(p[4]),
                &|p: &[f64]| p[4] * p[5] - 3.0,
                &|p: &[f64]| p[5] + p[6] * p[6],
                &|p: &[f64]| p[6] * p[7] - 4.0,
                &|p: &[f64]| p[7].hypot(p[8]),
                &|p: &[f64]| p[8] * p[9] - 5.0,
                &|p: &[f64]| p[9] + p[0],
                &|p: &[f64]| p[0] + p[4] + p[8],
                &|p: &[f64]| p[2] + p[6],
                // Two wide rows, the shape a rigidity span has: they are what forces the coloring
                // apart while every other row still reads a neighbouring pair.
                &|p: &[f64]| p[0] + p[2] + p[4] + p[6],
                &|p: &[f64]| p[1] + p[3] + p[5] + p[7] + p[9],
            ],
            vec![
                vec![0, 1],
                vec![1, 2],
                vec![2, 3],
                vec![3, 4],
                vec![4, 5],
                vec![5, 6],
                vec![6, 7],
                vec![7, 8],
                vec![8, 9],
                vec![9, 0],
                vec![0, 4, 8],
                vec![2, 6],
                vec![0, 2, 4, 6],
                vec![1, 3, 5, 7, 9],
            ],
        )
    }

    /// Every group's row set is the union of its columns' rows, ascending and without repeats —
    /// which is what makes it safe to hand to a system as "these rows and no others".
    #[test]
    fn a_groups_rows_are_exactly_its_columns_rows() {
        let system = a_sparse_chain();
        let reads = system.parameter_reads().expect("declared");
        let grouping = ColumnGrouping::curtis_powell_reid(&reads, system.parameter_count());
        for group in 0..grouping.group_count() {
            let mut expected: Vec<usize> = grouping
                .group(group)
                .iter()
                .flat_map(|column| grouping.rows_of(*column).iter().copied())
                .collect();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(grouping.rows_of_group(group), expected, "group {group}");
        }
    }

    /// Narrowing each pass to its group's own rows leaves the Jacobian BIT for bit what a
    /// whole-vector pass produced, and costs a fraction of the row evaluations.
    #[test]
    fn a_narrowed_pass_is_the_wide_one_bit_for_bit_and_cheaper() {
        let at = [0.7, -1.3, 0.25, 4.0, -0.5, 2.75, 1.1, -2.2, 0.33, 3.5];
        let narrow = a_sparse_chain();
        let mut wide = a_sparse_chain();
        // The same system with the seam closed: every pass evaluates the whole vector.
        wide.reads = narrow.reads.clone();
        let grouping = ColumnGrouping::curtis_powell_reid(
            &narrow.parameter_reads().expect("declared"),
            narrow.parameter_count(),
        );
        let mut here = vec![0.0; narrow.residual_count()];
        narrow.residuals(&at, &mut here);
        narrow.rows_evaluated.set(0);
        let narrowed = jacobian_in_groups(&narrow, &at, &here, &grouping);

        let column_by_column = jacobian_column_by_column(&wide, &at);
        for (index, (one, other)) in narrowed.iter().zip(&column_by_column).enumerate() {
            assert_eq!(
                one.to_bits(),
                other.to_bits(),
                "entry {index}: {one} against {other}"
            );
        }
        // What a group-at-a-time pass would have cost with the seam closed: two whole vectors per
        // group. What it cost narrowed: two entries per non-zero of the sparsity pattern.
        let wide_cost = 2 * grouping.group_count() * narrow.residual_count();
        let narrow_cost = narrow.rows_evaluated.get();
        assert!(
            narrow_cost * 2 < wide_cost,
            "narrowing saved nothing: {narrow_cost} rows against {wide_cost}"
        );
    }

    /// And it stays bit for bit through the non-finite door, where the untouched rows are
    /// reconstructed from `here` rather than evaluated at all.
    #[test]
    fn a_narrowed_pass_matches_where_a_row_is_not_finite() {
        let finite = |p: &[f64]| p[0] * 2.0;
        let infinite = |p: &[f64]| f64::INFINITY * p[1].signum();
        let system = Sparse::new(3, vec![&finite, &infinite], vec![vec![0], vec![1]]);
        let at = [1.5, 2.5, -3.0];
        let grouping = ColumnGrouping::curtis_powell_reid(
            &system.parameter_reads().expect("declared"),
            system.parameter_count(),
        );
        let mut here = vec![0.0; system.residual_count()];
        system.residuals(&at, &mut here);
        let narrowed = jacobian_in_groups(&system, &at, &here, &grouping);
        let column_by_column = jacobian_column_by_column(&system, &at);
        for (index, (one, other)) in narrowed.iter().zip(&column_by_column).enumerate() {
            assert_eq!(
                one.to_bits(),
                other.to_bits(),
                "entry {index}: {one} against {other}"
            );
        }
    }

    /// An honest subset pass survives the falsifier, and one ulp of dishonesty does not.
    #[test]
    fn a_subset_pass_that_answers_differently_is_found() {
        let at = [0.7, -1.3, 0.25, 4.0, -0.5, 2.75, 1.1, -2.2, 0.33, 3.5];
        let honest = a_sparse_chain();
        assert_eq!(first_subset_disagreement(&honest, &at), None);

        let mut lying = a_sparse_chain();
        lying.truthful = false;
        let found = first_subset_disagreement(&lying, &at);
        assert!(
            matches!(found, Some(SubsetDisagreement { group: None, .. })),
            "the single-row ask is where it lies: {found:?}"
        );
    }

    /// A system that never overrode the seam is not accused of anything.
    #[test]
    fn the_default_subset_pass_is_beyond_reproach() {
        let first = |p: &[f64]| p[0] - 1.0;
        let second = |p: &[f64]| p[1] * p[2];
        let system = Declared {
            parameters: 3,
            residuals: vec![&first, &second],
            reads: vec![vec![0], vec![1, 2]],
        };
        assert_eq!(first_subset_disagreement(&system, &[1.0, 2.0, 3.0]), None);
    }

    /// A narrowing system solves to the same bits an unnarrowed one does, through `solve` rather
    /// than through the Jacobian alone.
    #[test]
    fn narrowing_the_passes_does_not_move_the_answer() {
        let distance = |p: &[f64]| ((p[2] - p[0]).powi(2) + (p[3] - p[1]).powi(2)).sqrt() - 10.0;
        let horizontal = |p: &[f64]| p[3] - p[1];
        let wide = Declared {
            parameters: 4,
            residuals: vec![&distance, &horizontal],
            reads: vec![vec![0, 1, 2, 3], vec![1, 3]],
        };
        let narrow = Sparse::new(
            4,
            vec![&distance, &horizontal],
            vec![vec![0, 1, 2, 3], vec![1, 3]],
        );
        let mut one = vec![0.0, 0.0, 8.0, 1.0];
        let mut other = vec![0.0, 0.0, 8.0, 1.0];
        let first = solve(&wide, &mut one, SolveSettings::default());
        let second = solve(&narrow, &mut other, SolveSettings::default());
        for (index, (left, right)) in one.iter().zip(&other).enumerate() {
            assert_eq!(left.to_bits(), right.to_bits(), "parameter {index}");
        }
        assert_eq!(first, second);
    }

    /// The falsifier catches a system that reads a parameter it did not declare — the exact
    /// mistake that would make a grouped Jacobian quietly wrong.
    #[test]
    fn a_read_nobody_declared_is_found() {
        let honest = |p: &[f64]| p[0] - 1.0;
        // Says it reads only column 1 and reads column 2 as well.
        let lying = |p: &[f64]| p[1] + 0.5 * p[2];
        let system = Declared {
            parameters: 3,
            residuals: vec![&honest, &lying],
            reads: vec![vec![0], vec![1]],
        };
        assert_eq!(
            first_undeclared_read(&system, &[1.0, 2.0, 3.0], 1.0e-3),
            Some(UndeclaredRead { row: 1, column: 2 })
        );
    }

    /// And says nothing about an honest one.
    #[test]
    fn an_honest_declaration_survives_the_falsifier() {
        let first = |p: &[f64]| p[0] - 1.0;
        let second = |p: &[f64]| p[1] * p[2];
        let system = Declared {
            parameters: 3,
            residuals: vec![&first, &second],
            reads: vec![vec![0], vec![1, 2]],
        };
        assert_eq!(
            first_undeclared_read(&system, &[1.0, 2.0, 3.0], 1.0e-3),
            None
        );
    }

    /// A declaration with the wrong number of ROWS is refused outright rather than used for the
    /// rows it does cover: rows are matched to residuals by position, so one row short is every
    /// later row's claim pinned to the wrong residual.
    #[test]
    fn a_declaration_of_the_wrong_length_is_refused() {
        let first = |p: &[f64]| p[0] - 1.0;
        let second = |p: &[f64]| p[1] - 2.0;
        let system = Declared {
            parameters: 2,
            residuals: vec![&first, &second],
            reads: vec![vec![0]],
        };
        assert!(JacobianPlan::for_system(&system).grouping.is_none());
        // And the Jacobian still comes out right, by the column-by-column door.
        let matrix = jacobian(&system, &[0.0, 0.0]);
        assert!((matrix[0] - 1.0).abs() < 1e-9, "{matrix:?}");
        assert!((matrix[3] - 1.0).abs() < 1e-9, "{matrix:?}");
    }

    /// Two points, a coincidence and a distance: the first two rows are LINEAR and the system
    /// differentiates them itself, the third is a square root and is left to differences. The shape
    /// the sketch layer will eventually take, in four parameters instead of forty.
    struct PartlyAnalytic {
        /// How far apart the pair is asked to stand.
        span: f64,
        /// A deliberate mistake in one analytic entry, for the falsifier to find.
        spoiled: bool,
    }

    impl ResidualSystem for PartlyAnalytic {
        fn parameter_count(&self) -> usize {
            4
        }
        fn residual_count(&self) -> usize {
            3
        }
        fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
            // Rows 0 and 1: the first point sits at the origin. Row 2: the pair stands `span` apart.
            into[0] = parameters[0];
            into[1] = parameters[1];
            into[2] =
                (parameters[2] - parameters[0]).hypot(parameters[3] - parameters[1]) - self.span;
        }
        fn parameter_reads(&self) -> Option<ResidualReads> {
            Some(ResidualReads::from_rows(vec![
                vec![0],
                vec![1],
                vec![0, 1, 2, 3],
            ]))
        }
        fn analytic_rows(&self) -> Option<AnalyticRows> {
            Some(AnalyticRows::from_rows([0, 1]))
        }
        fn analytic_jacobian(&self, _parameters: &[f64], into: &mut [f64]) {
            into[0..4].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
            into[4..8].copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);
            if self.spoiled {
                into[5] = -1.0;
            }
        }
    }

    /// A row the system differentiates itself comes out EXACT, and one it does not is the same
    /// central difference it always was.
    #[test]
    fn an_analytic_row_replaces_its_difference_and_leaves_the_others_alone() {
        let system = PartlyAnalytic {
            span: 10.0,
            spoiled: false,
        };
        let at = [0.3, -0.2, 8.0, 1.0];
        let matrix = jacobian(&system, &at);
        // Exactly one and exactly zero, which a difference of these residuals is not.
        assert_eq!(matrix[0..4], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(matrix[4..8], [0.0, 1.0, 0.0, 0.0]);
        // And the differenced row is untouched by the overlay.
        let span = (at[2] - at[0]).hypot(at[3] - at[1]);
        assert!(
            (matrix[10] - (at[2] - at[0]) / span).abs() < 1e-7,
            "{matrix:?}"
        );
        assert!(
            (matrix[11] - (at[3] - at[1]) / span).abs() < 1e-7,
            "{matrix:?}"
        );
    }

    /// Naming rows analytic frees the colouring: the two linear rows stop conflicting over the
    /// columns they read, so nothing but the differenced row is ever evaluated.
    #[test]
    fn analytic_rows_leave_the_colouring_to_the_rows_that_are_differenced() {
        let system = PartlyAnalytic {
            span: 10.0,
            spoiled: false,
        };
        let plan = JacobianPlan::for_system(&system);
        let grouping = plan.grouping.as_ref().expect("declared");
        for group in 0..grouping.group_count() {
            for row in grouping.rows_of_group(group) {
                assert_eq!(*row, 2, "only the differenced row is ever evaluated");
            }
        }
    }

    /// The falsifier catches a wrong analytic entry, and clears a right one.
    #[test]
    fn a_wrong_analytic_derivative_is_found() {
        let at = [0.3, -0.2, 8.0, 1.0];
        let honest = PartlyAnalytic {
            span: 10.0,
            spoiled: false,
        };
        assert_eq!(first_wrong_analytic_derivative(&honest, &at, 1.0e-6), None);

        let spoiled = PartlyAnalytic {
            span: 10.0,
            spoiled: true,
        };
        let found = first_wrong_analytic_derivative(&spoiled, &at, 1.0e-6);
        assert!(
            matches!(
                found,
                Some(WrongDerivative {
                    row: 1,
                    column: 1,
                    ..
                })
            ),
            "the flipped sign: {found:?}"
        );
    }

    /// A row named analytic and never written is caught too: the poison says so rather than the
    /// difference underneath it standing in.
    #[test]
    fn an_analytic_row_nobody_wrote_is_found() {
        struct NamesAndDoesNotWrite;
        impl ResidualSystem for NamesAndDoesNotWrite {
            fn parameter_count(&self) -> usize {
                1
            }
            fn residual_count(&self) -> usize {
                1
            }
            fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
                into[0] = parameters[0] - 3.0;
            }
            fn analytic_rows(&self) -> Option<AnalyticRows> {
                Some(AnalyticRows::from_rows([0]))
            }
        }
        let found = first_wrong_analytic_derivative(&NamesAndDoesNotWrite, &[1.0], 1.0e-6);
        assert!(
            matches!(
                found,
                Some(WrongDerivative {
                    row: 0,
                    column: 0,
                    ..
                })
            ),
            "{found:?}"
        );
    }

    /// The overlay wins on the column-by-column path too, where the difference pass cannot narrow
    /// and writes an entry into every row whether it was asked to or not.
    #[test]
    fn an_analytic_row_survives_the_column_by_column_path() {
        struct NoReadsDeclared;
        impl ResidualSystem for NoReadsDeclared {
            fn parameter_count(&self) -> usize {
                2
            }
            fn residual_count(&self) -> usize {
                2
            }
            fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
                into[0] = 3.0 * parameters[0] - parameters[1];
                into[1] = parameters[0] * parameters[1];
            }
            fn analytic_rows(&self) -> Option<AnalyticRows> {
                Some(AnalyticRows::from_rows([0]))
            }
            fn analytic_jacobian(&self, _parameters: &[f64], into: &mut [f64]) {
                into[0..2].copy_from_slice(&[3.0, -1.0]);
            }
        }
        let matrix = jacobian(&NoReadsDeclared, &[0.7, -1.3]);
        assert_eq!(matrix[0..2], [3.0, -1.0], "exact, not differenced");
        assert!((matrix[2] - (-1.3)).abs() < 1e-7, "{matrix:?}");
        assert!((matrix[3] - 0.7).abs() < 1e-7, "{matrix:?}");
    }

    /// A system that names no analytic row is not accused of anything, and takes the same Jacobian
    /// it always did.
    #[test]
    fn a_system_with_no_analytic_rows_is_unchanged() {
        let first = |p: &[f64]| p[0] * p[0] + (3.0 * p[1]).sin();
        let second = |p: &[f64]| p[2].exp() - p[3];
        let system = Declared {
            parameters: 4,
            residuals: vec![&first, &second],
            reads: vec![vec![0, 1], vec![2, 3]],
        };
        let at = [0.7, -1.3, 0.25, 4.0];
        assert_eq!(first_wrong_analytic_derivative(&system, &at, 1.0e-6), None);
        let through_the_plan = jacobian(&system, &at);
        let column_by_column = jacobian_column_by_column(&system, &at);
        for (index, (one, other)) in through_the_plan.iter().zip(&column_by_column).enumerate() {
            assert_eq!(one.to_bits(), other.to_bits(), "entry {index}");
        }
    }

    /// And the seam carries through `solve`: the partly-analytic system reaches its answer.
    #[test]
    fn a_partly_analytic_system_solves() {
        let system = PartlyAnalytic {
            span: 10.0,
            spoiled: false,
        };
        let mut parameters = vec![0.3, -0.2, 8.0, 1.0];
        let report = solve(&system, &mut parameters, SolveSettings::default());
        assert_eq!(report.outcome, SolveOutcome::Converged, "{report:?}");
        assert!(parameters[0].abs() < 1.0e-9, "{parameters:?}");
        assert!(parameters[1].abs() < 1.0e-9, "{parameters:?}");
        let span = (parameters[2] - parameters[0]).hypot(parameters[3] - parameters[1]);
        assert!((span - 10.0).abs() < 1.0e-6, "{span}");
    }

    /// A declared system solves to the same answer an undeclared one does, through `solve` rather
    /// than through the Jacobian alone — the grouping must survive the trust-region loop.
    #[test]
    fn declaring_reads_does_not_move_the_answer() {
        let distance = |p: &[f64]| ((p[2] - p[0]).powi(2) + (p[3] - p[1]).powi(2)).sqrt() - 10.0;
        let horizontal = |p: &[f64]| p[3] - p[1];
        let plain = Closures {
            parameters: 4,
            residuals: vec![&distance, &horizontal],
        };
        let declared = Declared {
            parameters: 4,
            residuals: vec![&distance, &horizontal],
            reads: vec![vec![0, 1, 2, 3], vec![1, 3]],
        };
        let mut one = vec![0.0, 0.0, 8.0, 1.0];
        let mut other = vec![0.0, 0.0, 8.0, 1.0];
        let first = solve(&plain, &mut one, SolveSettings::default());
        let second = solve(&declared, &mut other, SolveSettings::default());
        for (index, (left, right)) in one.iter().zip(&other).enumerate() {
            assert_eq!(left.to_bits(), right.to_bits(), "parameter {index}");
        }
        assert_eq!(first, second);
    }
}
