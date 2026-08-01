//! Validated continuous planar sketch problems.
//!
//! The public façade owns local handles, typed outcomes, and domain diagnostics. Solver mechanics
//! and substrate numerical types stay private to this subsystem.

mod curve;
mod model;
mod solve;
mod symmetry;
mod tangent;
#[cfg(test)]
mod tests;

pub use curve::{ArcDomain, CircularCurve, CurveGeometry};
pub use model::{SolveOutcome, SolveReport};
pub use solve::{
    concentric_center, Analysis, ArcId, BuildError, CircleId, ConstraintId, CurrentValidation,
    Diagnostics, DragOutcome, ParameterId, ParameterKind, ParameterValue, PointId, Problem,
    ProblemBuilder, Relation, RequestError, SegmentId, Settled, SketchCurve, Solution,
    TangentContactFailure, TrialAdd, TrialRejection,
};
pub use symmetry::{
    choose_symmetry_branch, symmetry_axis_is_valid, symmetry_witness, SymmetryBranch,
    SymmetryError, SymmetryWitness,
};
pub use tangent::{
    choose_branch, tangent_arc_candidate, tangent_contact, BranchChoiceError, InternalContainment,
    LineSide, TangentArcCandidate, TangentArcCandidateError, TangentBranch, TangentContact,
    TangentContactError, TangentCurve,
};
