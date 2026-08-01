//! Validated continuous planar sketch problems.
//!
//! The public façade owns local handles, typed outcomes, and domain diagnostics. Solver mechanics
//! and substrate numerical types stay private to this subsystem.

mod model;
mod solve;
mod tangent;
#[cfg(test)]
mod tests;

pub use model::{SolveOutcome, SolveReport};
pub use solve::{
    Analysis, ArcId, BuildError, CircleId, ConstraintId, CurrentValidation, Diagnostics,
    DragOutcome, ParameterId, ParameterKind, ParameterValue, PointId, Problem, ProblemBuilder,
    Relation, RequestError, SegmentId, Settled, SketchCurve, Solution, TangentContactFailure,
    TrialAdd, TrialRejection,
};
pub use tangent::{
    choose_branch, tangent_contact, ArcDomain, BranchChoiceError, CircularCurve, CurveGeometry,
    InternalContainment, LineSide, TangentBranch, TangentContact, TangentContactError,
    TangentCurve,
};
