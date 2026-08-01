//! Continuous planar sketch solving.
//!
//! This module deliberately knows only local handles and resolved `f64` positions. A document
//! adapter owns stable ids, authored measurements, persistence, and atomic write-back. The solver
//! reads the supplied coordinates as its initial guess and returns coordinates for the adapter to
//! store as authored state; an under-constrained drawing has freedoms only that stored state can
//! remember.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::imprecise_flops,
    clippy::manual_midpoint,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    clippy::use_self
)]

use super::model::{SolveOutcome, SolveReport};
use crate::{AngleMeasurement, ResolvedLength};
use std::sync::atomic::{AtomicU64, Ordering};
use substrate::nonlinear_least_squares::{
    jacobian, rank, solve as solve_nlls, ResidualSystem, SolveOutcome as SubstrateSolveOutcome,
    SolveReport as SubstrateSolveReport, SolveSettings,
};

/// A trial whose residual norm is at or below this tolerance has met its relations.
///
/// The search outcome is deliberately not this test: residual tolerance is absolute while a step
/// tolerance is relative to parameter magnitude, so a large yet satisfied drawing may stop with a
/// `Stalled` status. Search status describes the path; residual norm describes the answer.
const SATISFIED_RESIDUAL: f64 = 1e-6;
/// A span the sketch needs — a segment, arc chord, or arc radius — below this threshold has lost
/// its geometric identity. This is separate from satisfaction even though both current values are
/// `1e-6`: one judges equations and the other judges whether meaningful geometry survived. It is
/// deliberately not a flattening tolerance: flattening chooses a display/discretization scale,
/// while this test rejects loss of the curve or segment identity itself.
const COLLAPSED_SPAN: f64 = 1e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PointId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArcId {
    owner: u64,
    index: usize,
}

/// An opaque local intrinsic-curve parameter. It cannot alias a point coordinate or cross a
/// problem-owner boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId {
    owner: u64,
    index: usize,
}

/// A local whole circle: an authored center point plus an intrinsic radius parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CircleId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveKey {
    Segment(SegmentId),
    Arc(ArcId),
    Circle(CircleId),
}

/// The physical kind of an intrinsic scalar. Typed results prevent an adapter from writing a
/// solved radius through the arc-angle door, or vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKind {
    SignedSweepDegrees,
    PositiveRadius,
}

/// A solved intrinsic scalar paired with its kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterValue {
    SweepDegrees(f64),
    Radius(f64),
}

/// A continuous author relation over local handles.
///
/// Every semantic match on this enum is exhaustive. That is load-bearing rather than stylistic:
/// a new relation must reach validation, resolution, residual evaluation, anchoring, and document
/// mapping. Most critically, residual row counts are the stride consumed by the arithmetic loop;
/// assigning a new two-row relation one row shifts every following residual and silently corrupts
/// the solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Relation {
    /// This point does not move, and `at` is where it does not move to. The target is explicit,
    /// rather than whichever position a later drag reaches, because a fix asserts immovability at
    /// a place rather than a mutable snapshot.
    Fix { point: PointId, at: [f64; 2] },
    /// The endpoints share axis 1, so the segment lies along axis 0.
    Horizontal { segment: SegmentId },
    /// The endpoints share axis 0, so the segment lies along axis 1.
    Vertical { segment: SegmentId },
    /// A dimension carrying an authored resolved length, unlike `Equal`, which names no length.
    Distance {
        from: PointId,
        to: PointId,
        length: f64,
    },
    /// Two independently-addressable points occupy one place. This relation deliberately does not
    /// merge their handles: merging destroys an id, rewrites every segment that named it, and makes
    /// deleting the assertion unable to restore the drawing.
    Coincident { first: PointId, second: PointId },
    /// Parallel uses sine between unit directions, so it is scale independent.
    Parallel { first: SegmentId, second: SegmentId },
    /// Perpendicular uses normalized cosine for the same scale-independent reason.
    Perpendicular { first: SegmentId, second: SegmentId },
    /// Two segments have the same length without asserting which length; that is different from
    /// two `Distance` dimensions that each carry one authored number.
    Equal { first: SegmentId, second: SegmentId },
    /// Midpoint is a place, not merely a line condition, and therefore contributes two rows.
    Midpoint { point: PointId, segment: SegmentId },
    /// Collinear contributes two distances to a datum line: parallel plus zero offset.
    Collinear { first: SegmentId, second: SegmentId },
}

impl Relation {
    /// Keep this exhaustive. It is the residual stride consumed by the arithmetic loop below, so a
    /// wrong answer shifts every later row rather than merely making one relation wrong.
    fn residual_count(self) -> usize {
        match self {
            Self::Fix { .. }
            | Self::Coincident { .. }
            | Self::Midpoint { .. }
            | Self::Collinear { .. } => 2,
            Self::Horizontal { .. }
            | Self::Vertical { .. }
            | Self::Distance { .. }
            | Self::Parallel { .. }
            | Self::Perpendicular { .. }
            | Self::Equal { .. } => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    UnknownPoint,
    UnknownSegment,
    UnknownParameter,
    InvalidParameter,
}

#[derive(Debug, Clone, Copy)]
struct Point {
    at: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
struct Segment {
    from: PointId,
    to: PointId,
}

#[derive(Debug, Clone, Copy)]
struct ArcCenter {
    key: ArcId,
    center: PointId,
    from: PointId,
    to: PointId,
    sweep_degrees: f64,
}

/// One scalar represented in a topology-safe solver coordinate.
///
/// `stored` is the physical value supplied by the document.  A free parameter occupies one
/// numerical column; a fixed one remains available to derived geometry but has no column.  The
/// transform is deliberately here rather than in the document so finite-difference Jacobians see
/// every dependency through the same physical-value read.
#[derive(Debug, Clone, Copy)]
struct Parameter {
    kind: ParameterKind,
    stored: f64,
    free: bool,
    sweep_sign: f64,
}

#[derive(Debug, Clone, Copy)]
struct Circle {
    key: CircleId,
    center: PointId,
    radius: ParameterId,
}

#[derive(Debug, Clone, Copy)]
struct ConstraintEntry {
    key: ConstraintId,
    relation: Relation,
    resolved: Resolved,
}

/// The only construction door for a solver problem.
///
/// It validates every reference and resolves relations once, so residual execution is arithmetic
/// rather than repeated semantic lookup. Handle order is insertion order; adapters that need
/// reproducibility insert stable ids sorted.
#[derive(Debug)]
pub struct ProblemBuilder {
    owner: u64,
    points: Vec<Point>,
    segments: Vec<Segment>,
    arc_centers: Vec<ArcCenter>,
    arc_parameters: Vec<Option<ParameterId>>,
    parameters: Vec<Parameter>,
    circles: Vec<Circle>,
    constraints: Vec<(ConstraintId, Relation)>,
}

impl Default for ProblemBuilder {
    fn default() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        Self {
            owner: NEXT_OWNER.fetch_add(1, Ordering::Relaxed),
            points: Vec::new(),
            segments: Vec::new(),
            arc_centers: Vec::new(),
            arc_parameters: Vec::new(),
            parameters: Vec::new(),
            circles: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl ProblemBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_point(&mut self, at: [f64; 2]) -> PointId {
        let id = PointId {
            owner: self.owner,
            index: self.points.len(),
        };
        self.points.push(Point { at });
        id
    }

    pub fn add_segment(&mut self, from: PointId, to: PointId) -> SegmentId {
        let id = SegmentId {
            owner: self.owner,
            index: self.segments.len(),
        };
        self.segments.push(Segment { from, to });
        id
    }

    /// Add a solver-writable signed arc sweep. Its sign is fixed at creation and the numerical
    /// coordinate represents only the open-turn magnitude, so no solver step can create a zero
    /// or full-turn chord arc.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidParameter`] when `degrees` is not a finite, non-zero sweep
    /// strictly inside one turn.
    pub fn add_free_signed_sweep(&mut self, degrees: f64) -> Result<ParameterId, BuildError> {
        self.add_parameter(ParameterKind::SignedSweepDegrees, degrees, true)
    }

    /// Add a source-owned signed arc sweep. Fixed values are resolved geometry but do not enter
    /// the optimization vector.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidParameter`] when `degrees` is not a finite, non-zero sweep
    /// strictly inside one turn.
    pub fn add_fixed_signed_sweep(&mut self, degrees: f64) -> Result<ParameterId, BuildError> {
        self.add_parameter(ParameterKind::SignedSweepDegrees, degrees, false)
    }

    /// Add a solver-writable strictly-positive radius.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidParameter`] when `radius` is non-finite or not positive.
    pub fn add_free_positive_radius(&mut self, radius: f64) -> Result<ParameterId, BuildError> {
        self.add_parameter(ParameterKind::PositiveRadius, radius, true)
    }

    /// Add a source-owned strictly-positive radius.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidParameter`] when `radius` is non-finite or not positive.
    pub fn add_fixed_positive_radius(&mut self, radius: f64) -> Result<ParameterId, BuildError> {
        self.add_parameter(ParameterKind::PositiveRadius, radius, false)
    }

    fn add_parameter(
        &mut self,
        kind: ParameterKind,
        stored: f64,
        free: bool,
    ) -> Result<ParameterId, BuildError> {
        // Fixed scalars are resolved source geometry, not solver coordinates: retain every
        // exact/domain-valid source value. Free scalars cross the transform on every residual
        // evaluation, so their authored starting value must already lie in that transform's
        // durable-output envelope.
        let source_valid = match kind {
            ParameterKind::SignedSweepDegrees => {
                stored.is_finite()
                    && stored.abs() > 0.0
                    && stored.abs() < 360.0
                    // The inverse is intentionally not clamped: authored values retain their
                    // own coordinate, including values adjacent to either open endpoint.
                    && (stored.abs().ln() - (360.0 - stored.abs()).ln()).is_finite()
                    && AngleMeasurement::try_from_degrees_f64(stored).is_ok()
            }
            ParameterKind::PositiveRadius => {
                stored.is_finite() && stored > 0.0 && ResolvedLength::try_from_f64(stored).is_ok()
            }
        };
        let valid = source_valid
            && (!free
                || match kind {
                    ParameterKind::SignedSweepDegrees => {
                        stored.abs() >= min_exact_positive() && stored.abs() <= max_open_sweep()
                    }
                    ParameterKind::PositiveRadius => {
                        stored >= min_exact_positive() && stored <= max_exact_positive()
                    }
                });
        if !valid {
            return Err(BuildError::InvalidParameter);
        }
        let key = ParameterId {
            owner: self.owner,
            index: self.parameters.len(),
        };
        self.parameters.push(Parameter {
            kind,
            stored,
            free,
            sweep_sign: if matches!(kind, ParameterKind::SignedSweepDegrees) {
                stored.signum()
            } else {
                1.0
            },
        });
        Ok(key)
    }

    pub fn add_arc_center(
        &mut self,
        center: PointId,
        from: PointId,
        to: PointId,
        sweep_degrees: f64,
    ) -> ArcId {
        let key = ArcId {
            owner: self.owner,
            index: self.arc_centers.len(),
        };
        self.arc_centers.push(ArcCenter {
            key,
            center,
            from,
            to,
            sweep_degrees,
        });
        self.arc_parameters.push(None);
        key
    }

    /// Add a chord-defined arc whose sweep is an intrinsic scalar parameter.
    pub fn add_arc(
        &mut self,
        center: PointId,
        from: PointId,
        to: PointId,
        sweep: ParameterId,
    ) -> ArcId {
        let key = ArcId {
            owner: self.owner,
            index: self.arc_centers.len(),
        };
        // Validation at `finish` keeps this construction door cheap and gives foreign parameter
        // ids the same owner check every other local handle receives.
        self.arc_centers.push(ArcCenter {
            key,
            center,
            from,
            to,
            sweep_degrees: f64::NAN,
        });
        // The old ArcCenter layout still stores a scalar for existing callers. Phase 0 replaces
        // it through the side table below when the problem is finished.
        self.arc_parameters.push(Some(sweep));
        key
    }

    /// Add a whole circle with an authored center point and an intrinsic radius parameter.
    pub fn add_circle(&mut self, center: PointId, radius: ParameterId) -> CircleId {
        let key = CircleId {
            owner: self.owner,
            index: self.circles.len(),
        };
        self.circles.push(Circle {
            key,
            center,
            radius,
        });
        key
    }

    pub fn add_constraint(&mut self, relation: Relation) -> ConstraintId {
        let key = ConstraintId {
            owner: self.owner,
            index: self.constraints.len(),
        };
        self.constraints.push((key, relation));
        key
    }

    /// # Errors
    ///
    /// Returns an error when a curve or relation references a foreign or unknown local handle.
    pub fn finish(self) -> Result<Problem, BuildError> {
        let known_point =
            |point: PointId| point.owner == self.owner && point.index < self.points.len();
        let known_segment =
            |segment: SegmentId| segment.owner == self.owner && segment.index < self.segments.len();
        let known_parameter = |parameter: ParameterId| {
            parameter.owner == self.owner && parameter.index < self.parameters.len()
        };
        for segment in &self.segments {
            if !known_point(segment.from) || !known_point(segment.to) {
                return Err(BuildError::UnknownPoint);
            }
        }
        for (arc, parameter) in self.arc_centers.iter().zip(&self.arc_parameters) {
            if !known_point(arc.center) || !known_point(arc.from) || !known_point(arc.to) {
                return Err(BuildError::UnknownPoint);
            }
            if parameter.is_some_and(|parameter| !known_parameter(parameter)) {
                return Err(BuildError::UnknownParameter);
            }
            if let Some(parameter) = parameter {
                if !matches!(
                    self.parameters[parameter.index].kind,
                    ParameterKind::SignedSweepDegrees
                ) {
                    return Err(BuildError::InvalidParameter);
                }
            }
        }
        for circle in &self.circles {
            if !known_point(circle.center) {
                return Err(BuildError::UnknownPoint);
            }
            if !known_parameter(circle.radius) {
                return Err(BuildError::UnknownParameter);
            }
            if !matches!(
                self.parameters[circle.radius.index].kind,
                ParameterKind::PositiveRadius
            ) {
                return Err(BuildError::InvalidParameter);
            }
        }
        for (_, relation) in &self.constraints {
            let points = match *relation {
                Relation::Fix { point, .. } | Relation::Midpoint { point, .. } => vec![point],
                Relation::Distance { from, to, .. } => vec![from, to],
                Relation::Coincident { first, second } => vec![first, second],
                _ => Vec::new(),
            };
            if points.into_iter().any(|point| !known_point(point)) {
                return Err(BuildError::UnknownPoint);
            }
            let segments = match *relation {
                Relation::Horizontal { segment }
                | Relation::Vertical { segment }
                | Relation::Midpoint { segment, .. } => vec![segment],
                Relation::Parallel { first, second }
                | Relation::Perpendicular { first, second }
                | Relation::Equal { first, second }
                | Relation::Collinear { first, second } => vec![first, second],
                _ => Vec::new(),
            };
            if segments.into_iter().any(|segment| !known_segment(segment)) {
                return Err(BuildError::UnknownSegment);
            }
        }
        let raw = Problem {
            owner: self.owner,
            points: self.points,
            segments: self.segments,
            arc_centers: self.arc_centers,
            arc_parameters: self.arc_parameters,
            parameters: self.parameters,
            circles: self.circles,
            constraints: Vec::new(),
        };
        let constraints = self
            .constraints
            .into_iter()
            .map(|(key, relation)| {
                raw.resolve(relation).map(|resolved| ConstraintEntry {
                    key,
                    relation,
                    resolved,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Problem { constraints, ..raw })
    }
}

/// A validated local problem. It intentionally stores no document ids, authored measurements, or
/// persistence policy; a document adapter owns those and atomically applies a returned solution.
#[derive(Debug, Clone)]
pub struct Problem {
    owner: u64,
    /// Resolved point coordinates plus edge topology. Topology and derived arc centers travel
    /// together: a solve that saw an arc's center dependency without its chord, or vice versa,
    /// would be solving a different drawing.
    points: Vec<Point>,
    segments: Vec<Segment>,
    arc_centers: Vec<ArcCenter>,
    arc_parameters: Vec<Option<ParameterId>>,
    parameters: Vec<Parameter>,
    circles: Vec<Circle>,
    constraints: Vec<ConstraintEntry>,
}

impl Problem {
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn relation_count(&self) -> usize {
        self.constraints.len()
    }

    fn scalar_coordinates(&self) -> Vec<f64> {
        self.parameters
            .iter()
            .copied()
            .map(parameter_coordinate)
            .collect()
    }

    fn solution(&self, positions: Vec<[f64; 2]>, scalar_coordinates: &[f64]) -> Solution {
        Solution {
            owner: self.owner,
            positions,
            parameters: self
                .parameters
                .iter()
                .copied()
                .zip(scalar_coordinates)
                .map(|(parameter, coordinate)| match parameter.kind {
                    ParameterKind::SignedSweepDegrees => ParameterValue::SweepDegrees(
                        if coordinate.to_bits() == parameter_coordinate(parameter).to_bits() {
                            parameter.stored
                        } else {
                            physical_parameter_value(parameter, *coordinate)
                        },
                    ),
                    ParameterKind::PositiveRadius => ParameterValue::Radius(
                        if coordinate.to_bits() == parameter_coordinate(parameter).to_bits() {
                            parameter.stored
                        } else {
                            physical_parameter_value(parameter, *coordinate)
                        },
                    ),
                })
                .collect(),
        }
    }

    fn resolve(&self, relation: Relation) -> Result<Resolved, BuildError> {
        let point = |id: PointId| {
            (id.owner == self.owner && id.index < self.points.len())
                .then_some(id.index)
                .ok_or(BuildError::UnknownPoint)
        };
        let segment = |id: SegmentId| {
            (id.owner == self.owner)
                .then(|| self.segments.get(id.index))
                .flatten()
                .map(|segment| SegmentSlots {
                    from: segment.from.index,
                    to: segment.to.index,
                })
                .ok_or(BuildError::UnknownSegment)
        };
        match relation {
            Relation::Fix { point: id, at } => Ok(Resolved::Fix {
                slot: point(id)?,
                at,
            }),
            Relation::Horizontal { segment: id } => {
                let segment = segment(id)?;
                Ok(Resolved::SameCoordinate {
                    from: segment.from,
                    to: segment.to,
                    axis: 1,
                })
            }
            Relation::Vertical { segment: id } => {
                let segment = segment(id)?;
                Ok(Resolved::SameCoordinate {
                    from: segment.from,
                    to: segment.to,
                    axis: 0,
                })
            }
            Relation::Distance { from, to, length } => Ok(Resolved::Distance {
                from: point(from)?,
                to: point(to)?,
                length,
            }),
            Relation::Coincident { first, second } => Ok(Resolved::Coincident {
                first: point(first)?,
                second: point(second)?,
            }),
            Relation::Parallel { first, second } => Ok(Resolved::Parallel {
                first: segment(first)?,
                second: segment(second)?,
            }),
            Relation::Perpendicular { first, second } => Ok(Resolved::Perpendicular {
                first: segment(first)?,
                second: segment(second)?,
            }),
            Relation::Equal { first, second } => Ok(Resolved::Equal {
                first: segment(first)?,
                second: segment(second)?,
            }),
            Relation::Midpoint {
                point: id,
                segment: edge,
            } => Ok(Resolved::Midpoint {
                point: point(id)?,
                segment: segment(edge)?,
            }),
            Relation::Collinear { first, second } => Ok(Resolved::Collinear {
                datum: segment(first)?,
                other: segment(second)?,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::manual_assert_eq,
        clippy::many_single_char_names,
        clippy::panic,
        clippy::unwrap_used
    )]

    use super::*;

    fn two_segments() -> (Problem, PointId, PointId, PointId, SegmentId, SegmentId) {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 4.0]);
        let c = builder.add_point([0.0, 10.0]);
        let d = builder.add_point([10.0, 10.0]);
        let first = builder.add_segment(a, b);
        let second = builder.add_segment(c, d);
        (builder.finish().unwrap(), a, b, c, first, second)
    }

    fn accepts(problem: &Problem, relation: Relation) -> Settled {
        match problem.trial_add(relation).unwrap() {
            TrialAdd::Accepted { settled, .. } => settled,
            TrialAdd::Rejected(rejection) => panic!("expected acceptance: {rejection:?}"),
        }
    }

    #[test]
    /// Each relation reaches the validated residual system, so adding a new variant cannot leave a
    /// silent construction-only path behind. This is the migration guard for exhaustive semantic
    /// seams: a public relation that builds but never writes a row is worse than a compile error.
    fn every_existing_relation_has_a_validated_solver_path() {
        let (problem, a, b, c, first, second) = two_segments();
        accepts(
            &problem,
            Relation::Fix {
                point: a,
                at: [0.0, 0.0],
            },
        );
        accepts(&problem, Relation::Horizontal { segment: first });
        accepts(&problem, Relation::Vertical { segment: first });
        accepts(
            &problem,
            Relation::Distance {
                from: a,
                to: b,
                length: 8.0,
            },
        );
        accepts(
            &problem,
            Relation::Coincident {
                first: a,
                second: c,
            },
        );
        accepts(&problem, Relation::Parallel { first, second });
        accepts(&problem, Relation::Perpendicular { first, second });
        accepts(&problem, Relation::Equal { first, second });
        accepts(
            &problem,
            Relation::Midpoint {
                point: c,
                segment: first,
            },
        );
        accepts(&problem, Relation::Collinear { first, second });
    }

    #[test]
    /// Witness rank belongs to the author’s drawing, where informative gradients have not yet
    /// vanished at an exactly solved configuration; solution rank would make a useful assertion
    /// look redundant after it has driven the drawing to a singular witness.
    fn analysis_reports_rank_and_freedom_at_the_given_witness() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 4.0]);
        let segment = builder.add_segment(a, b);
        builder.add_constraint(Relation::Fix {
            point: a,
            at: [0.0, 0.0],
        });
        builder.add_constraint(Relation::Horizontal { segment });
        let analysis = builder.finish().unwrap().analyze();
        assert_eq!(analysis.witness_rank, 3);
        assert_eq!(analysis.degrees_of_freedom, 1);
    }

    #[test]
    /// Redundancy compares witness rank before and after the candidate, never solver outcome or
    /// solution rank: search progress is not authoring information.
    fn trial_add_flags_a_rank_redundant_relation() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 0.0]);
        let segment = builder.add_segment(a, b);
        builder.add_constraint(Relation::Horizontal { segment });
        let problem = builder.finish().unwrap();
        let TrialAdd::Accepted { redundant, .. } =
            problem.trial_add(Relation::Horizontal { segment }).unwrap()
        else {
            panic!("a duplicate valid row is accepted but redundant");
        };
        assert!(redundant);
    }

    #[test]
    /// A stopped search can still leave a residual below the authoring threshold; status records
    /// the numerical path, while the residual says whether the author’s relations hold.
    fn satisfaction_is_residual_based_not_search_outcome_based() {
        let diagnostics = diagnostics(Some(SolveReport {
            outcome: SolveOutcome::Stalled,
            iterations: 1,
            residual_norm: SATISFIED_RESIDUAL / 2.0,
            degrees_of_freedom: 0,
            redundant_residuals: 0,
        }));
        assert!(diagnostics.satisfied);
    }

    #[test]
    /// Fixed geometry is a visible reference and outranks a larger loose component. This verifies
    /// the strict anchor law rather than an accidental insertion-order tie break.
    fn fixed_piece_wins_anchor_selection_over_a_larger_unfixed_piece() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 0.0]);
        let c = builder.add_point([20.0, 0.0]);
        let d = builder.add_point([30.0, 0.0]);
        let e = builder.add_point([40.0, 0.0]);
        let left = builder.add_segment(a, b);
        let right_one = builder.add_segment(c, d);
        let right_two = builder.add_segment(d, e);
        builder.add_constraint(Relation::Fix {
            point: a,
            at: [0.0, 0.0],
        });
        let problem = builder.finish().unwrap();
        let anchors = problem.anchor_for(Relation::Parallel {
            first: left,
            second: right_one,
        });
        assert_eq!(anchors, vec![a, b]);
        assert!(
            Problem::named_segments(Relation::Parallel {
                first: right_one,
                second: right_two
            })
            .len()
                == 2
        );
    }

    #[test]
    /// The hand temporarily pulls; releasing it restores standing relations exactly. Drag has no
    /// rigidity preference and is not a hard pin, so remaining permitted motion can still occur.
    fn drag_is_a_pull_then_exact_settle() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([40.0, 0.0]);
        let segment = builder.add_segment(a, b);
        builder.add_constraint(Relation::Horizontal { segment });
        let problem = builder.finish().unwrap();
        let DragOutcome::Accepted(settled) = problem.drag(a, [-7.0, -18.0]).unwrap() else {
            panic!("a free horizontal segment can follow the hand");
        };
        let a = settled.solution.position(a).unwrap();
        let b = settled.solution.position(b).unwrap();
        assert!((a[1] - b[1]).abs() < SATISFIED_RESIDUAL);
        assert!((a[0] + 7.0).abs() < SATISFIED_RESIDUAL);
    }

    #[test]
    /// Orthogonal assertions can meet only at a singularity, which must be refused as collapse
    /// rather than reported as a successful zero residual.
    fn collapse_is_rejected_with_a_curve_key() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 4.0]);
        let segment = builder.add_segment(a, b);
        builder.add_constraint(Relation::Horizontal { segment });
        let problem = builder.finish().unwrap();
        let TrialAdd::Rejected(TrialRejection::Collapsed { curve, .. }) =
            problem.trial_add(Relation::Vertical { segment }).unwrap()
        else {
            panic!("orthogonal rows collapse one segment");
        };
        assert_eq!(curve, CurveKey::Segment(segment));
    }

    #[test]
    /// Blame is leave-one-out: only a standing relation whose removal restores a solution is named;
    /// this returns actionable local keys rather than a rank heuristic’s guess.
    fn blame_is_leave_one_out_and_returns_local_constraint_keys() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([10.0, 0.0]);
        let first = builder.add_constraint(Relation::Fix {
            point: a,
            at: [0.0, 0.0],
        });
        let second = builder.add_constraint(Relation::Fix {
            point: b,
            at: [10.0, 0.0],
        });
        let problem = builder.finish().unwrap();
        let TrialAdd::Rejected(TrialRejection::Unsatisfied { conflicts }) = problem
            .trial_add(Relation::Distance {
                from: a,
                to: b,
                length: 20.0,
            })
            .unwrap()
        else {
            panic!("fixed incompatible distance is unsatisfied");
        };
        assert_eq!(conflicts, vec![first, second]);
    }

    #[test]
    /// A handle from another builder never aliases a local slot or reaches an out-of-bounds read.
    fn foreign_handles_are_rejected_at_finish_and_drag() {
        let mut first = ProblemBuilder::new();
        let foreign = first.add_point([0.0, 0.0]);
        let mut second = ProblemBuilder::new();
        let local = second.add_point([1.0, 1.0]);
        second.add_segment(foreign, local);
        assert!(matches!(second.finish(), Err(BuildError::UnknownPoint)));

        let (problem, _, _, _, _, _) = two_segments();
        assert!(matches!(
            problem.drag(foreign, [0.0, 0.0]),
            Err(RequestError::UnknownPoint)
        ));
    }

    #[test]
    /// Fixed intrinsic values are geometry inputs, while free ones are genuine solver freedoms.
    /// A whole circle is sufficient to prove this without coupling the test to a relation kind.
    fn only_free_curve_parameters_contribute_degrees_of_freedom() {
        let mut free = ProblemBuilder::new();
        let center = free.add_point([0.0, 0.0]);
        let radius = free.add_free_positive_radius(4.0).unwrap();
        free.add_circle(center, radius);
        assert_eq!(free.finish().unwrap().analyze().degrees_of_freedom, 3);

        let mut fixed = ProblemBuilder::new();
        let center = fixed.add_point([0.0, 0.0]);
        let radius = fixed.add_fixed_positive_radius(4.0).unwrap();
        fixed.add_circle(center, radius);
        assert_eq!(fixed.finish().unwrap().analyze().degrees_of_freedom, 2);
    }

    #[test]
    fn free_scalar_endpoints_preserve_authored_bits_when_settled() {
        // These are the transform envelope endpoints. No arbitrary transform clamp may rewrite
        // their authored bits before the solver has a relation to satisfy.
        for sweep in [min_exact_positive(), max_open_sweep()] {
            let mut builder = ProblemBuilder::new();
            let parameter = builder.add_free_signed_sweep(sweep).unwrap();
            let settled = builder.finish().unwrap().settle();
            let Some(ParameterValue::SweepDegrees(value)) = settled.solution.parameter(parameter)
            else {
                panic!("the sweep has its declared scalar type");
            };
            assert_eq!(value.to_bits(), sweep.to_bits());
        }
        for radius in [min_exact_positive(), max_exact_positive()] {
            let mut builder = ProblemBuilder::new();
            let parameter = builder.add_free_positive_radius(radius).unwrap();
            let settled = builder.finish().unwrap().settle();
            let Some(ParameterValue::Radius(value)) = settled.solution.parameter(parameter) else {
                panic!("the radius has its declared scalar type");
            };
            assert_eq!(value.to_bits(), radius.to_bits());
        }
    }

    #[test]
    fn fixed_outside_free_envelope_stays_exact_while_free_is_refused() {
        // This exact power of two fits durable rational storage but lies below the transform
        // envelope that guarantees any optimizer-produced neighboring f64 also fits.
        let source_value = f64::from_bits(923_u64 << 52); // 2^-100

        let mut free = ProblemBuilder::new();
        assert!(matches!(
            free.add_free_signed_sweep(source_value),
            Err(BuildError::InvalidParameter)
        ));
        assert!(matches!(
            free.add_free_positive_radius(source_value),
            Err(BuildError::InvalidParameter)
        ));

        let mut fixed = ProblemBuilder::new();
        let sweep = fixed.add_fixed_signed_sweep(source_value).unwrap();
        let radius = fixed.add_fixed_positive_radius(source_value).unwrap();
        let problem = fixed.finish().unwrap();
        assert_eq!(problem.analyze().degrees_of_freedom, 0);
        let settled = problem.settle();
        let Some(ParameterValue::SweepDegrees(solved_sweep)) = settled.solution.parameter(sweep)
        else {
            panic!("the sweep has its declared scalar type");
        };
        let Some(ParameterValue::Radius(solved_radius)) = settled.solution.parameter(radius) else {
            panic!("the radius has its declared scalar type");
        };
        assert_eq!(solved_sweep.to_bits(), source_value.to_bits());
        assert_eq!(solved_radius.to_bits(), source_value.to_bits());

        assert_eq!(
            physical_parameter_value(problem.parameters[sweep.index], f64::INFINITY).to_bits(),
            source_value.to_bits(),
            "fixed sweep bypasses the free-value logistic transform"
        );
        assert_eq!(
            physical_parameter_value(problem.parameters[radius.index], f64::NEG_INFINITY).to_bits(),
            source_value.to_bits(),
            "fixed radius bypasses the free-value exponential transform"
        );
    }

    #[test]
    fn scalar_transforms_stay_topology_safe_and_exactly_writable_at_extremes() {
        let mut builder = ProblemBuilder::new();
        let sweep = builder.add_free_signed_sweep(90.0).unwrap();
        let radius = builder.add_free_positive_radius(1.0).unwrap();
        let problem = builder.finish().unwrap();
        let sweep_parameter = problem.parameters[sweep.index];
        let radius_parameter = problem.parameters[radius.index];

        for coordinate in [f64::NEG_INFINITY, -1.0e300, 1.0e300, f64::INFINITY] {
            let sweep_value = physical_parameter_value(sweep_parameter, coordinate);
            assert!(sweep_value.abs() > 0.0 && sweep_value.abs() < 360.0);
            assert!(AngleMeasurement::try_from_degrees_f64(sweep_value).is_ok());

            let radius_value = physical_parameter_value(radius_parameter, coordinate);
            assert!(radius_value > 0.0 && radius_value.is_finite());
            assert!(
                ResolvedLength::try_from_f64(radius_value).is_ok(),
                "radius {radius_value:?} at coordinate {coordinate:?}"
            );
        }
    }

    #[test]
    /// An arc center reads its sweep through the intrinsic parameter transform. Fixing its ends
    /// and center therefore solves the free sweep rather than treating the reified center as an
    /// independent, stale coordinate.
    fn free_arc_sweep_moves_through_derived_center() {
        let mut builder = ProblemBuilder::new();
        let from = builder.add_point([0.0, 0.0]);
        let to = builder.add_point([10.0, 0.0]);
        let center = builder.add_point([5.0, 5.0]);
        let sweep = builder.add_free_signed_sweep(90.0).unwrap();
        builder.add_arc(center, from, to, sweep);
        builder.add_constraint(Relation::Fix {
            point: from,
            at: [0.0, 0.0],
        });
        builder.add_constraint(Relation::Fix {
            point: to,
            at: [10.0, 0.0],
        });
        builder.add_constraint(Relation::Fix {
            point: center,
            at: [5.0, 0.0],
        });
        let settled = builder.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied);
        let Some(ParameterValue::SweepDegrees(solved)) = settled.solution.parameter(sweep) else {
            panic!("the sweep has its declared scalar type");
        };
        assert!((solved - 180.0).abs() < 1e-4, "solved sweep was {solved}");
    }

    #[test]
    fn intrinsic_parameter_building_rejects_wrong_domains_and_kinds() {
        let mut builder = ProblemBuilder::new();
        assert!(matches!(
            builder.add_free_signed_sweep(0.0),
            Err(BuildError::InvalidParameter)
        ));
        assert!(matches!(
            builder.add_fixed_positive_radius(f64::INFINITY),
            Err(BuildError::InvalidParameter)
        ));
        let point = builder.add_point([0.0, 0.0]);
        let sweep = builder.add_fixed_signed_sweep(90.0).unwrap();
        builder.add_circle(point, sweep);
        assert!(matches!(
            builder.finish(),
            Err(BuildError::InvalidParameter)
        ));
    }
}

#[derive(Debug, Clone)]
pub struct Solution {
    owner: u64,
    positions: Vec<[f64; 2]>,
    parameters: Vec<ParameterValue>,
}

impl Solution {
    pub fn position(&self, point: PointId) -> Option<[f64; 2]> {
        if point.owner != self.owner {
            return None;
        }
        self.positions.get(point.index).copied()
    }

    /// The physical value of an intrinsic parameter, typed by the construction door that made it.
    pub fn parameter(&self, parameter: ParameterId) -> Option<ParameterValue> {
        if parameter.owner != self.owner {
            return None;
        }
        self.parameters.get(parameter.index).copied()
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub report: Option<SolveReport>,
    pub satisfied: bool,
}

#[derive(Debug, Clone)]
pub struct Settled {
    pub solution: Solution,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub solution: Solution,
    pub diagnostics: Diagnostics,
    pub witness_rank: usize,
    pub degrees_of_freedom: usize,
}

#[derive(Debug, Clone)]
pub enum TrialAdd {
    Accepted { settled: Settled, redundant: bool },
    Rejected(TrialRejection),
}

#[derive(Debug, Clone)]
pub enum TrialRejection {
    Unsatisfied {
        conflicts: Vec<ConstraintId>,
    },
    Collapsed {
        curve: CurveKey,
        implicated: Vec<ConstraintId>,
    },
}

#[derive(Debug, Clone)]
pub enum DragOutcome {
    Accepted(Settled),
    Rejected(Diagnostics),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestError {
    UnknownPoint,
    InvalidRelation(BuildError),
}

/// Whether the system carries the **rigidity regularizer**: one row per edge and axis asking that
/// the edge's span come out of the solve as it went in.
///
/// A relation should have a small blast radius — geometry it does not name should move as little
/// as it can, and when it must move, it should move as a piece. Minimizing each point's
/// displacement does the opposite: the cheapest way to bring one corner of a polygon to a far
/// point is to drag that corner alone and leave the rest, which is maximum deformation for minimum
/// travel. Asking instead that every edge keep its **per-axis span** makes a pure translation of a
/// connected group free — every span is unchanged — while any stretch, rotation, or shear is paid
/// for. Length, orientation, and area are what the rows are written in terms of, so they are what
/// gets preserved.
///
/// **The weight is 1 and does not need tuning.** When a rigid motion can satisfy the relations,
/// both blocks reach zero at once and there is no trade to weigh. When they genuinely conflict,
/// the exact pass below runs the relations alone afterwards, so rigidity can only rank answers
/// that satisfy the relations equally well. It is a preference over a null space, never a vote
/// against a relation.
///
/// **The heavier group is anchored, not merely outweighed.** Least squares otherwise splits travel
/// between joined groups in inverse proportion to their sizes; a hard anchor lets the smaller group
/// travel to the visible reference. It is removed from the first parameter vector rather than
/// added as a soft row. The exact pass releases this unreachable anchor so the report describes
/// the real relation system. Drag carries no rigidity because the hand is already the reference.
#[derive(Debug, Clone, Copy)]
enum Rigidity<'a> {
    /// Constraint rows only: used for rank readings and the final exactness pass.
    Ignored,
    /// Preserve every edge span and remove the chosen reference piece from the parameter vector.
    Preferred { anchored: &'a [PointId] },
}

#[derive(Debug, Clone, Copy)]
struct EdgeSpan {
    from: usize,
    to: usize,
    span: [f64; 2],
}
#[derive(Debug, Clone, Copy)]
/// An arc center's derived dependencies, resolved to local coordinate slots.
struct ArcSlots {
    from: usize,
    to: usize,
    sweep_parameter: Option<usize>,
    sweep_degrees: f64,
}
#[derive(Debug, Clone, Copy)]
/// A segment's directional ends. Naming them avoids silently reversing a directional relation.
struct SegmentSlots {
    from: usize,
    to: usize,
}

#[derive(Debug, Clone, Copy)]
enum Resolved {
    Fix {
        slot: usize,
        at: [f64; 2],
    },
    SameCoordinate {
        from: usize,
        to: usize,
        axis: usize,
    },
    Distance {
        from: usize,
        to: usize,
        length: f64,
    },
    Coincident {
        first: usize,
        second: usize,
    },
    Parallel {
        first: SegmentSlots,
        second: SegmentSlots,
    },
    Perpendicular {
        first: SegmentSlots,
        second: SegmentSlots,
    },
    Equal {
        first: SegmentSlots,
        second: SegmentSlots,
    },
    Midpoint {
        point: usize,
        segment: SegmentSlots,
    },
    Collinear {
        datum: SegmentSlots,
        other: SegmentSlots,
    },
}

/// The parameterized residual system, field by field.
///
/// Every point occupies two coordinate slots, not only points named by a relation: unconstrained
/// points are real degrees of freedom. `problem` supplies the topology whose edge spans rigidity
/// preserves; `resolved` holds every relation's endpoints as slots, built once so the residual loop
/// is arithmetic and never searches topology by handle; `derived` says which slots are arc centers;
/// `base` is the pre-solve whole-coordinate vector; and `free` narrows that vector to mutable
/// parameters while anchored coordinates remain in `base`.
///
/// Derived centers occupy slots for simple write-back but are read through their defining function
/// and therefore do not add mutable freedoms.
struct Residuals<'a> {
    /// The validated source of point coordinates and topology.
    problem: &'a Problem,
    /// Each relation resolved once to slots in constraint order.
    resolved: Vec<Resolved>,
    /// Per-point derived-center dependencies; `None` is the ordinary, authored case.
    derived: Vec<Option<ArcSlots>>,
    /// Every author-drawn edge span to preserve during the preference pass, absent in exact mode.
    rigidity: Vec<EdgeSpan>,
    /// Whole-coordinate values at the start of this pass; anchored values remain here unchanged.
    base: Vec<f64>,
    /// Indices into `base` that the numerical solver may alter, in parameter-vector order.
    free: Vec<usize>,
}

/// Read a derived center as the function of its ends and sweep, never as an independent choice.
/// A relation on the center therefore moves the arc ends, not a value later synchronization would
/// overwrite. Its stored slots are inert, zero-Jacobian coordinates; the degree-of-freedom result
/// subtracts them, and write-back uses this same function so the document stays synchronized.
/// Derived-center chains are not authored topology, so this follows exactly one level. A degenerate
/// arc falls back to its stored slot: repair owns invalid-document cleanup, while the solver must
/// retain a finite value to evaluate the rest of a drawing.
fn position_of(
    derived: &[Option<ArcSlots>],
    specifications: &[Parameter],
    parameters: &[f64],
    point_count: usize,
    slot: usize,
) -> [f64; 2] {
    let stored = |slot: usize| [parameters[slot * 2], parameters[slot * 2 + 1]];
    derived[slot].map_or_else(
        || stored(slot),
        |arc| {
            let sweep = arc.sweep_parameter.map_or(arc.sweep_degrees, |index| {
                physical_parameter_value(specifications[index], parameters[point_count * 2 + index])
            });
            arc_center_radius(stored(arc.from), stored(arc.to), sweep)
                .map_or_else(|| stored(slot), |(center, _)| center)
        },
    )
}

fn parameter_coordinate(parameter: Parameter) -> f64 {
    match parameter.kind {
        ParameterKind::SignedSweepDegrees => {
            let magnitude = parameter.stored.abs();
            magnitude.ln() - (360.0 - magnitude).ln()
        }
        ParameterKind::PositiveRadius => parameter.stored.ln(),
    }
}

fn physical_parameter_value(parameter: Parameter, coordinate: f64) -> f64 {
    // Source-owned geometry never participates in optimization. Returning it directly preserves
    // its exact resolved f64 rather than routing it through the free-value topology transform.
    if !parameter.free {
        return parameter.stored;
    }
    match parameter.kind {
        ParameterKind::SignedSweepDegrees => {
            let magnitude = if coordinate >= 0.0 {
                360.0 / (1.0 + (-coordinate).exp())
            } else {
                360.0 * coordinate.exp() / (1.0 + coordinate.exp())
            };
            parameter.sweep_sign * magnitude.clamp(min_exact_positive(), max_open_sweep())
        }
        ParameterKind::PositiveRadius => coordinate
            .clamp(min_exact_positive().ln(), max_exact_positive().ln())
            .exp()
            .clamp(min_exact_positive(), max_exact_positive()),
    }
}

/// The smallest positive bound for which *every* finite `f64` at or above the bound has an exact
/// binary ratio in the bounded `i128` store. Individual values can fit below it, but an optimizer
/// may produce adjacent values whose denominator does not; this envelope keeps every output
/// atomically writable.
const fn min_exact_positive() -> f64 {
    f64::from_bits(949_u64 << 52) // 2^-74; 52 significand bits + 74 exponent bits = 126
}

/// The largest positive IEEE value whose exact binary ratio fits the bounded `i128` rational
/// store: the predecessor of 2^127, which itself is one past the positive `i128` boundary.
const fn max_exact_positive() -> f64 {
    f64::from_bits((1150_u64 << 52) - 1)
}

/// A chord arc has an open sweep domain, so the physical map must not round to a full turn.
const fn max_open_sweep() -> f64 {
    f64::from_bits(360.0_f64.to_bits() - 1)
}

fn arc_center_radius(from: [f64; 2], to: [f64; 2], sweep_degrees: f64) -> Option<([f64; 2], f64)> {
    let sweep = sweep_degrees.to_radians();
    let half = sweep / 2.0;
    let sine = half.sin();
    if sine.abs() <= f64::EPSILON {
        return None;
    }
    let chord = [to[0] - from[0], to[1] - from[1]];
    let chord_length = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt();
    if chord_length <= f64::EPSILON {
        return None;
    }
    let radius = chord_length / (2.0 * sine.abs());
    let midpoint = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
    let normal = [-chord[1] / chord_length, chord[0] / chord_length];
    let apothem = chord_length / (2.0 * half.tan());
    Some((
        [
            midpoint[0] + normal[0] * apothem,
            midpoint[1] + normal[1] * apothem,
        ],
        radius,
    ))
}

/// The length of one resolved segment at the supplied coordinates.
fn length_of(at: &impl Fn(usize) -> [f64; 2], segment: SegmentSlots) -> f64 {
    let (tail, head) = (at(segment.from), at(segment.to));
    let span = [head[0] - tail[0], head[1] - tail[1]];
    (span[0] * span[0] + span[1] * span[1]).sqrt()
}
/// A zero-length segment has no direction. It contributes a zero angular row rather than inventing
/// a direction from noise; collapse validation is responsible for explaining why that otherwise-
/// solved geometry is rejected.
fn unit_along(at: &impl Fn(usize) -> [f64; 2], segment: SegmentSlots) -> [f64; 2] {
    let (tail, head) = (at(segment.from), at(segment.to));
    let span = [head[0] - tail[0], head[1] - tail[1]];
    let length = (span[0] * span[0] + span[1] * span[1]).sqrt();
    if length <= f64::EPSILON {
        [0.0, 0.0]
    } else {
        [span[0] / length, span[1] / length]
    }
}

impl<'a> Residuals<'a> {
    /// Build the parameter layout. Anchored coordinates stay at their pre-solve values and are
    /// omitted from the parameter vector; free coordinates are widened before every geometry read.
    /// This narrowing/widening boundary keeps residual code in one whole-coordinate space while
    /// the numerical substrate sees only actual unknowns.
    fn new(problem: &'a Problem, scalar_coordinates: &[f64], rigidity: Rigidity) -> Option<Self> {
        if scalar_coordinates.len() != problem.parameters.len() {
            return None;
        }
        if problem.points.is_empty() {
            return None;
        }
        let mut derived = vec![None; problem.points.len()];
        for (arc, sweep) in problem.arc_centers.iter().zip(&problem.arc_parameters) {
            derived[arc.center.index] = Some(ArcSlots {
                from: arc.from.index,
                to: arc.to.index,
                sweep_parameter: sweep.map(|parameter| parameter.index),
                sweep_degrees: arc.sweep_degrees,
            });
        }
        let resolved = problem
            .constraints
            .iter()
            .map(|constraint| constraint.resolved)
            .collect();
        let mut base = Vec::with_capacity(problem.points.len() * 2 + problem.parameters.len());
        for point in &problem.points {
            base.extend(point.at);
        }
        base.extend_from_slice(scalar_coordinates);
        let point_coordinates = problem.points.len() * 2;
        let free_scalars = || {
            problem
                .parameters
                .iter()
                .enumerate()
                .filter_map(|(index, parameter)| {
                    parameter.free.then_some(point_coordinates + index)
                })
                .collect::<Vec<_>>()
        };
        let (rigidity, mut free) = match rigidity {
            Rigidity::Ignored => (
                Vec::new(),
                (0..point_coordinates)
                    .filter(|index| derived[index / 2].is_none())
                    .collect::<Vec<_>>(),
            ),
            Rigidity::Preferred { anchored } => {
                let at = |slot| {
                    position_of(
                        &derived,
                        &problem.parameters,
                        &base,
                        problem.points.len(),
                        slot,
                    )
                };
                let spans = problem
                    .segments
                    .iter()
                    .map(|segment| (segment.from.index, segment.to.index))
                    .chain(
                        problem
                            .arc_centers
                            .iter()
                            .map(|arc| (arc.from.index, arc.to.index)),
                    )
                    .map(|(from, to)| {
                        let (tail, head) = (at(from), at(to));
                        EdgeSpan {
                            from,
                            to,
                            span: [head[0] - tail[0], head[1] - tail[1]],
                        }
                    })
                    .collect();
                let held: Vec<_> = anchored.iter().map(|point| point.index).collect();
                (
                    spans,
                    (0..point_coordinates)
                        .filter(|index| {
                            derived[index / 2].is_none() && !held.contains(&(index / 2))
                        })
                        .collect::<Vec<_>>(),
                )
            }
        };
        free.extend(free_scalars());
        Some(Self {
            problem,
            resolved,
            derived,
            rigidity,
            base,
            free,
        })
    }
    fn guess(&self, positions: &[[f64; 2]]) -> Vec<f64> {
        self.free
            .iter()
            .map(|index| {
                if *index < positions.len() * 2 {
                    positions[index / 2][index % 2]
                } else {
                    self.base[*index]
                }
            })
            .collect()
    }
    fn widen(&self, parameters: &[f64]) -> Vec<f64> {
        let mut whole = self.base.clone();
        for (parameter, index) in parameters.iter().zip(&self.free) {
            whole[*index] = *parameter;
        }
        whole
    }
}

impl ResidualSystem for Residuals<'_> {
    fn parameter_count(&self) -> usize {
        self.free.len()
    }
    fn residual_count(&self) -> usize {
        self.problem
            .constraints
            .iter()
            .map(|constraint| constraint.relation.residual_count())
            .sum::<usize>()
            + self.rigidity.len() * 2
    }
    fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
        let whole = self.widen(parameters);
        let at = |slot| {
            position_of(
                &self.derived,
                &self.problem.parameters,
                &whole,
                self.problem.points.len(),
                slot,
            )
        };
        let mut row = 0;
        for relation in &self.resolved {
            match *relation {
                Resolved::Fix { slot, at: target } => {
                    let here = at(slot);
                    into[row] = here[0] - target[0];
                    into[row + 1] = here[1] - target[1];
                    row += 2;
                }
                Resolved::SameCoordinate { from, to, axis } => {
                    into[row] = at(to)[axis] - at(from)[axis];
                    row += 1;
                }
                Resolved::Distance { from, to, length } => {
                    let (tail, head) = (at(from), at(to));
                    into[row] =
                        ((head[0] - tail[0]).powi(2) + (head[1] - tail[1]).powi(2)).sqrt() - length;
                    row += 1;
                }
                Resolved::Coincident { first, second } => {
                    let (a, b) = (at(first), at(second));
                    into[row] = a[0] - b[0];
                    into[row + 1] = a[1] - b[1];
                    row += 2;
                }
                Resolved::Parallel { first, second } => {
                    // Cross(unit directions) is sine(angle): this remains scale independent.
                    let (a, b) = (unit_along(&at, first), unit_along(&at, second));
                    into[row] = a[0] * b[1] - a[1] * b[0];
                    row += 1;
                }
                Resolved::Perpendicular { first, second } => {
                    // Dot(unit directions) is cosine(angle), also independent of segment length.
                    let (a, b) = (unit_along(&at, first), unit_along(&at, second));
                    into[row] = a[0] * b[0] + a[1] * b[1];
                    row += 1;
                }
                Resolved::Equal { first, second } => {
                    into[row] = length_of(&at, first) - length_of(&at, second);
                    row += 1;
                }
                Resolved::Midpoint { point, segment } => {
                    // Both coordinates are constrained because halfway names one exact place.
                    let (p, a, b) = (at(point), at(segment.from), at(segment.to));
                    into[row] = p[0] - (a[0] + b[0]) / 2.0;
                    into[row + 1] = p[1] - (a[1] + b[1]) / 2.0;
                    row += 2;
                }
                Resolved::Collinear { datum, other } => {
                    // Two distances to the datum line state both parallelism and zero offset.
                    let along = unit_along(&at, datum);
                    let normal = [-along[1], along[0]];
                    let anchor = at(datum.from);
                    for (offset, end) in [other.from, other.to].into_iter().enumerate() {
                        let here = at(end);
                        into[row + offset] =
                            (here[0] - anchor[0]) * normal[0] + (here[1] - anchor[1]) * normal[1];
                    }
                    row += 2;
                }
            }
        }
        for edge in &self.rigidity {
            // Per-axis spans intentionally do not leave a group free to rotate. The exact pass
            // handles any genuine disagreement between this preference and a relation.
            let (tail, head) = (at(edge.from), at(edge.to));
            into[row] = (head[0] - tail[0]) - edge.span[0];
            into[row + 1] = (head[1] - tail[1]) - edge.span[1];
            row += 2;
        }
    }
}

fn domain_report(subtrate: SubstrateSolveReport) -> SolveReport {
    let outcome = match subtrate.outcome {
        SubstrateSolveOutcome::Converged => SolveOutcome::Converged,
        SubstrateSolveOutcome::Stalled => SolveOutcome::Stalled,
        SubstrateSolveOutcome::ExhaustedIterations => SolveOutcome::ExhaustedIterations,
    };
    SolveReport {
        outcome,
        iterations: subtrate.iterations,
        residual_norm: subtrate.residual_norm,
        degrees_of_freedom: subtrate.degrees_of_freedom,
        redundant_residuals: subtrate.redundant_residuals,
    }
}

fn run(
    problem: &Problem,
    positions: &mut Vec<[f64; 2]>,
    scalar_coordinates: &mut Vec<f64>,
    rigidity: Rigidity,
) -> Option<SolveReport> {
    // An empty system is already met at its current drawing; callers read unconstrained freedom
    // directly from the validated problem rather than inventing an empty numerical report. A
    // nonempty relation set on no points similarly has no residual system and rank zero.
    if problem.constraints.is_empty() {
        return None;
    }
    let system = Residuals::new(problem, scalar_coordinates, rigidity)?;
    let mut parameters = system.guess(positions);
    let report = solve_nlls(&system, &mut parameters, SolveSettings::default());
    let whole = system.widen(&parameters);
    *positions = (0..problem.points.len())
        .map(|slot| {
            position_of(
                &system.derived,
                &problem.parameters,
                &whole,
                problem.points.len(),
                slot,
            )
        })
        .collect();
    *scalar_coordinates = whole[problem.points.len() * 2..].to_vec();
    Some(domain_report(report))
}

/// Search status diagnoses the numerical path; residual norm decides whether relations hold.
fn diagnostics(report: Option<SolveReport>) -> Diagnostics {
    let satisfied = report
        .as_ref()
        .is_none_or(|report| report.residual_norm <= SATISFIED_RESIDUAL);
    Diagnostics { report, satisfied }
}

/// Read rank at the author's given configuration, not its solution. A row whose gradient vanishes
/// after solving cannot make an informative relation look redundant — a distance between coincident
/// points is the canonical case. Solution rank lies for authoring because the solve can land at a
/// singular configuration. Rigidity is excluded because it is preference, not an assertion; no
/// points or no relations therefore yield rank zero.
fn witness_rank(problem: &Problem) -> usize {
    let scalar_coordinates = problem.scalar_coordinates();
    let Some(system) = Residuals::new(problem, &scalar_coordinates, Rigidity::Ignored) else {
        return 0;
    };
    let positions: Vec<_> = problem.points.iter().map(|point| point.at).collect();
    let guess = system.guess(&positions);
    let matrix = jacobian(&system, &guess);
    rank(&matrix, system.residual_count(), system.parameter_count())
}

impl Problem {
    /// Solve with a full two-pass settle: a preferred pass followed by an exact pass. The second
    /// pass starts at the preferred answer, preserving it wherever preference and relations agree,
    /// but releases anchors and reports the standing relations over the whole drawing. This makes
    /// exactness and degree-of-freedom diagnostics independent of the preference mechanism.
    pub fn settle(&self) -> Settled {
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let mut scalar_coordinates = self.scalar_coordinates();
        run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Preferred { anchored: &[] },
        );
        let report = run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        Settled {
            solution: self.solution(positions, &scalar_coordinates),
            diagnostics: diagnostics(report),
        }
    }

    /// Analyze the exact problem without changing its stored starting configuration. Derived
    /// centers are not freedoms: they are represented in slots solely for simple write-back.
    pub fn analyze(&self) -> Analysis {
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let mut scalar_coordinates = self.scalar_coordinates();
        let report = run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        let degrees_of_freedom = report.as_ref().map_or_else(
            || {
                self.points.len() * 2 - self.arc_centers.len() * 2
                    + self
                        .parameters
                        .iter()
                        .filter(|parameter| parameter.free)
                        .count()
            },
            |report| report.degrees_of_freedom,
        );
        Analysis {
            solution: self.solution(positions, &scalar_coordinates),
            diagnostics: diagnostics(report),
            witness_rank: witness_rank(self),
            degrees_of_freedom,
        }
    }

    /// Try one new relation without mutating the standing problem. A refusal leaves its starting
    /// drawing intact rather than where a failed search pushed it: residuals are equations, not a
    /// search-status verdict. An accepted redundant relation is retained because redundancy can
    /// express durable author intent.
    /// # Errors
    ///
    /// Returns an error if the relation names a handle outside this problem.
    pub fn trial_add(&self, relation: Relation) -> Result<TrialAdd, RequestError> {
        let candidate = self
            .resolve(relation)
            .map_err(RequestError::InvalidRelation)?;
        let anchored = self.anchor_for(relation);
        let candidate_problem = self.with_candidate(relation, candidate);
        let settled = candidate_problem.settle_with(&anchored);
        if !settled.diagnostics.satisfied {
            return Ok(TrialAdd::Rejected(TrialRejection::Unsatisfied {
                conflicts: self.blame(relation, candidate, &anchored),
            }));
        }
        if let Some(curve) = self.collapsed_by(&settled.solution) {
            return Ok(TrialAdd::Rejected(TrialRejection::Collapsed {
                curve,
                implicated: self.constraints_acting_on(curve),
            }));
        }
        let redundant = witness_rank(&candidate_problem) <= witness_rank(self);
        Ok(TrialAdd::Accepted { settled, redundant })
    }

    /// Pull one local point toward a target, then release the hand and settle standing relations.
    /// The hand is not a hard pin: a point free to slide along a relation must still be able to
    /// move even when the cursor is not exactly on that relation. An achievable pull is unchanged;
    /// an impossible pull returns the nearest standing solution or a rejection.
    /// # Errors
    ///
    /// Returns an error if the held point is not local to this problem.
    pub fn drag(&self, held: PointId, at: [f64; 2]) -> Result<DragOutcome, RequestError> {
        if held.owner != self.owner || held.index >= self.points.len() {
            return Err(RequestError::UnknownPoint);
        }
        let pull = Relation::Fix { point: held, at };
        let pulled = self.with_candidate(
            pull,
            self.resolve(pull).map_err(RequestError::InvalidRelation)?,
        );
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let mut scalar_coordinates = self.scalar_coordinates();
        run(
            &pulled,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        let report = run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        let diagnostics = diagnostics(report);
        let settled = Settled {
            solution: self.solution(positions, &scalar_coordinates),
            diagnostics: diagnostics.clone(),
        };
        Ok(if diagnostics.satisfied {
            DragOutcome::Accepted(settled)
        } else {
            DragOutcome::Rejected(diagnostics)
        })
    }

    fn settle_with(&self, anchored: &[PointId]) -> Settled {
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let mut scalar_coordinates = self.scalar_coordinates();
        run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Preferred { anchored },
        );
        let report = run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        Settled {
            solution: self.solution(positions, &scalar_coordinates),
            diagnostics: diagnostics(report),
        }
    }

    fn with_candidate(&self, relation: Relation, resolved: Resolved) -> Self {
        let mut problem = self.clone();
        let key = ConstraintId {
            owner: problem.owner,
            index: problem.constraints.len(),
        };
        problem.constraints.push(ConstraintEntry {
            key,
            relation,
            resolved,
        });
        problem
    }

    /// Anchor only a strict winner among the pieces the candidate joins. A fixed piece outranks
    /// cardinality because it is not going to travel regardless of size. Equal pieces meet in the
    /// middle rather than inheriting an arbitrary pick or id order the author cannot see.
    fn anchor_for(&self, relation: Relation) -> Vec<PointId> {
        let named = self.named_points(relation);
        let pieces = self.connected_pieces();
        let reached: Vec<_> = pieces
            .iter()
            .filter(|piece| piece.iter().any(|point| named.contains(point)))
            .collect();
        if reached.len() < 2 {
            return Vec::new();
        }
        let fixed = |piece: &&Vec<PointId>| {
            self.constraints.iter().any(|constraint| {
            matches!(constraint.relation, Relation::Fix { point, .. } if piece.contains(&point))
        })
        };
        let weight = |piece: &&Vec<PointId>| (fixed(piece), piece.len());
        let Some(winner) = reached.iter().copied().max_by_key(&weight) else {
            return Vec::new();
        };
        if reached
            .iter()
            .filter(|piece| weight(piece) == weight(&winner))
            .count()
            == 1
        {
            winner.clone()
        } else {
            Vec::new()
        }
    }

    /// Connected pieces are author-visible shapes rather than coordinate groups; a lone point is
    /// still a piece. Anchor selection weighs these components when a relation joins them.
    fn connected_pieces(&self) -> Vec<Vec<PointId>> {
        let mut pieces: Vec<Vec<PointId>> = (0..self.points.len())
            .map(|index| PointId {
                owner: self.owner,
                index,
            })
            .map(|point| vec![point])
            .collect();
        let joins = self
            .segments
            .iter()
            .map(|segment| (segment.from, segment.to))
            .chain(
                self.arc_centers
                    .iter()
                    .flat_map(|arc| [(arc.from, arc.to), (arc.from, arc.center)]),
            );
        for (first, second) in joins {
            let left = pieces.iter().position(|piece| piece.contains(&first));
            let right = pieces.iter().position(|piece| piece.contains(&second));
            if let (Some(left), Some(right)) = (left, right) {
                if left != right {
                    let mut joined = pieces[left].clone();
                    joined.extend(pieces[right].iter().copied());
                    let high = left.max(right);
                    let low = left.min(right);
                    pieces.remove(high);
                    pieces.remove(low);
                    pieces.push(joined);
                }
            }
        }
        pieces
    }

    fn named_points(&self, relation: Relation) -> Vec<PointId> {
        let mut points = match relation {
            Relation::Fix { point, .. } | Relation::Midpoint { point, .. } => vec![point],
            Relation::Distance { from, to, .. } => vec![from, to],
            Relation::Coincident { first, second } => vec![first, second],
            _ => Vec::new(),
        };
        for segment in Self::named_segments(relation) {
            if let Some(segment) = self.segments.get(segment.index) {
                points.extend([segment.from, segment.to]);
            }
        }
        points
    }

    fn named_segments(relation: Relation) -> Vec<SegmentId> {
        match relation {
            Relation::Horizontal { segment }
            | Relation::Vertical { segment }
            | Relation::Midpoint { segment, .. } => vec![segment],
            Relation::Parallel { first, second }
            | Relation::Perpendicular { first, second }
            | Relation::Equal { first, second }
            | Relation::Collinear { first, second } => vec![first, second],
            Relation::Fix { .. } | Relation::Distance { .. } | Relation::Coincident { .. } => {
                Vec::new()
            }
        }
    }

    /// Leave-one-out returns only standing constraints whose single removal restores a solution.
    /// At sketch scale the repeated solves are cheap and answer the author’s question more
    /// faithfully than a rank heuristic. An empty list means no single culprit, never harmlessness.
    fn blame(
        &self,
        relation: Relation,
        resolved: Resolved,
        anchored: &[PointId],
    ) -> Vec<ConstraintId> {
        self.constraints
            .iter()
            .filter_map(|standing| {
                let mut without = self.clone();
                without
                    .constraints
                    .retain(|constraint| constraint.key != standing.key);
                without
                    .with_candidate(relation, resolved)
                    .settle_with(anchored)
                    .diagnostics
                    .satisfied
                    .then_some(standing.key)
            })
            .collect()
    }

    /// Reject a newly collapsed segment, arc chord, or arc radius. A singularity can satisfy many
    /// residuals, so this is a property of the result and ignores geometry already degenerate:
    /// generic result-based collapse handles any relation that creates it without maintaining a
    /// list of relation-specific collapse rules.
    fn collapsed_by(&self, solution: &Solution) -> Option<CurveKey> {
        let before = |point: PointId| self.points[point.index].at;
        let span = |from: [f64; 2], to: [f64; 2]| {
            ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt()
        };
        for (index, segment) in self.segments.iter().enumerate() {
            let after = |point| solution.position(point).unwrap_or_else(|| before(point));
            if span(before(segment.from), before(segment.to)) > COLLAPSED_SPAN
                && span(after(segment.from), after(segment.to)) <= COLLAPSED_SPAN
            {
                return Some(CurveKey::Segment(SegmentId {
                    owner: self.owner,
                    index,
                }));
            }
        }
        for arc in &self.arc_centers {
            let after = |point| solution.position(point).unwrap_or_else(|| before(point));
            let collapsed = span(before(arc.from), before(arc.to)) > COLLAPSED_SPAN
                && span(after(arc.from), after(arc.to)) <= COLLAPSED_SPAN
                || span(before(arc.center), before(arc.from)) > COLLAPSED_SPAN
                    && span(after(arc.center), after(arc.from)) <= COLLAPSED_SPAN;
            if collapsed {
                return Some(CurveKey::Arc(arc.key));
            }
        }
        for circle in &self.circles {
            let before = self.parameters[circle.radius.index].stored;
            let after = match solution.parameter(circle.radius) {
                Some(ParameterValue::Radius(value)) => value,
                Some(ParameterValue::SweepDegrees(_)) | None => continue,
            };
            if before > COLLAPSED_SPAN && after <= COLLAPSED_SPAN {
                return Some(CurveKey::Circle(circle.key));
            }
        }
        None
    }

    /// Collapse implication is structural, not experimental: a prior solve may have moved the
    /// drawing, so removing one relation no longer reconstructs the geometry it once produced.
    fn constraints_acting_on(&self, curve: CurveKey) -> Vec<ConstraintId> {
        let points = match curve {
            CurveKey::Segment(segment) => self
                .segments
                .get(segment.index)
                .map(|segment| vec![segment.from, segment.to]),
            CurveKey::Arc(arc) => self
                .arc_centers
                .get(arc.index)
                .map(|arc| vec![arc.from, arc.to, arc.center]),
            CurveKey::Circle(circle) => self
                .circles
                .get(circle.index)
                .map(|circle| vec![circle.center]),
        }
        .unwrap_or_default();
        self.constraints
            .iter()
            .filter(|constraint| {
                Self::named_segments(constraint.relation)
                    .iter()
                    .any(|segment| matches!(curve, CurveKey::Segment(curve) if *segment == curve))
                    || self
                        .named_points(constraint.relation)
                        .iter()
                        .any(|point| points.contains(point))
            })
            .map(|constraint| constraint.key)
            .collect()
    }
}
