//! Validated continuous planar sketch problems.
//!
//! The public façade owns local handles, typed outcomes, and domain diagnostics. Solver mechanics
//! and substrate numerical types stay private to this subsystem.

mod center_arc;
mod circle;
mod curvature;
mod curve;
mod higher_curve;
mod midpoint_line;
mod model;
mod polygon;
mod rectangle;
mod slot;
mod solve;
mod spline;
mod symmetry;
mod tangent;
mod tangent_circle;
#[cfg(test)]
mod tests;

pub use center_arc::{center_arc_candidate, ArcTurn, CenterArcCandidate, CenterArcCandidateError};
pub use circle::{
    three_point_circle_candidate, two_point_circle_candidate, CircleCandidate, CircleCandidateError,
};
pub use curvature::{
    curvature_arrow_at, curvature_residual, direction_at, direction_residual, JointSpan, SpanEnd,
};
pub use curve::{within_drawn_extent, ArcDomain, CircularCurve, CurveGeometry};
pub use higher_curve::{
    conic_candidate, conic_rho_from_shoulder, conic_shoulder_track, conic_vertex_from_rho,
    ellipse_candidate, ConicCandidate, ConicCandidateError, EllipseCandidate,
    EllipseCandidateError, CONIC_PARABOLIC_RHO,
};
pub use midpoint_line::{
    midpoint_line_candidate, MidpointLineCandidate, MidpointLineCandidateError,
};
pub use model::{SolveOutcome, SolveReport};
pub use polygon::{
    centered_polygon_candidate, edge_polygon_candidate, CenteredPolygonKind, PolygonCandidate,
    PolygonCandidateError,
};
pub use rectangle::{
    center_rectangle_candidate, three_point_rectangle_candidate, RectangleCandidate,
    RectangleCandidateError,
};
pub use slot::{
    center_arc_slot_candidate, center_arc_slot_spine, linear_slot_candidate,
    three_point_arc_slot_candidate, three_point_arc_slot_spine, ArcSlotSpine, LinearSlotKind,
    SlotCandidate, SlotCandidateError, SlotEdgeCandidate, SlotSpine, SlotTurn,
};
pub use solve::{
    concentric_center, Analysis, ArcId, BuildError, CircleId, ConstraintId, CurrentValidation,
    Diagnostics, DragOutcome, Hand, HandRole, KeptQuantity, ParameterId, ParameterKind,
    ParameterValue, PointId, Problem, ProblemBuilder, Relation, RequestError, SegmentId, Settled,
    SketchCurve, SnapReach, Solution, TangentContactFailure, TrialAdd, TrialRejection,
};
pub use spline::{control_point_spline, fit_point_spline, SplineCandidate, SplineCandidateError};
pub use symmetry::{
    choose_symmetry_branch, symmetry_axis_is_valid, symmetry_witness, SymmetryBranch,
    SymmetryError, SymmetryWitness,
};
pub use tangent::{
    choose_branch, tangent_arc_candidate, tangent_contact, BranchChoiceError, InternalContainment,
    LineSide, TangentArcCandidate, TangentArcCandidateError, TangentBranch, TangentContact,
    TangentContactError, TangentCurve,
};
pub use tangent_circle::{
    three_tangent_circle_candidate, two_tangent_circle_candidate, TangentCircleCandidate,
    TangentCircleCandidateError,
};
