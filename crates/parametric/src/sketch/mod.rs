//! Validated continuous planar sketch problems.
//!
//! The public façade owns local handles, typed outcomes, and domain diagnostics. Solver mechanics
//! and substrate numerical types stay private to this subsystem.

mod model;
mod solve;
#[cfg(test)]
mod tests;

pub use model::{SolveOutcome, SolveReport};
pub use solve::{
    Analysis, ArcId, BuildError, CircleId, ConstraintId, CurveKey, Diagnostics, DragOutcome,
    ParameterId, ParameterKind, ParameterValue, PointId, Problem, ProblemBuilder, Relation,
    RequestError, SegmentId, Settled, Solution, TrialAdd, TrialRejection,
};
