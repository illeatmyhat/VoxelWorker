//! Validated continuous planar sketch problems.
//!
//! The public façade owns local handles, typed outcomes, and domain diagnostics. Solver mechanics
//! and substrate numerical types stay private to this subsystem.

mod center_arc;
mod circle;
mod curve;
mod midpoint_line;
mod model;
mod rectangle;
mod solve;
mod symmetry;
mod tangent;
#[cfg(test)]
mod tests;

pub use center_arc::{center_arc_candidate, CenterArcCandidate, CenterArcCandidateError};
pub use circle::{
    three_point_circle_candidate, two_point_circle_candidate, CircleCandidate, CircleCandidateError,
};
pub use curve::{ArcDomain, CircularCurve, CurveGeometry};
pub use midpoint_line::{
    midpoint_line_candidate, MidpointLineCandidate, MidpointLineCandidateError,
};
pub use model::{SolveOutcome, SolveReport};
pub use rectangle::{
    center_rectangle_candidate, three_point_rectangle_candidate, RectangleCandidate,
    RectangleCandidateError,
};
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
