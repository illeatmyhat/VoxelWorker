//! Domain diagnostics kept independent of the numerical substrate.

/// Why the numerical search stopped.
///
/// This is diagnostic context, never the satisfaction verdict: a relative step tolerance can stop
/// a large drawing with a residual well below what the author can express, while a stalled nonzero
/// residual is a least-squares compromise rather than a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveOutcome {
    Converged,
    Stalled,
    ExhaustedIterations,
}

/// Observable result of a continuous solve.
///
/// A small residual means relations hold; [`SolveOutcome`] says only how the search stopped. The
/// substrate report is deliberately mapped here so adapters do not depend on a numerical
/// implementation type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveReport {
    pub outcome: SolveOutcome,
    pub iterations: usize,
    pub residual_norm: f64,
    pub degrees_of_freedom: usize,
    pub redundant_residuals: usize,
}
