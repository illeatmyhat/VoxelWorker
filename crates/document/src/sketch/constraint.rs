//! Persisted constraint entities and the document-to-parametric adapter.
//!
//! A constraint lives in the same stable-id space as a point or a segment: it is selectable,
//! individually deletable, individually undoable, and the delete cascade reaches it when the
//! geometry it names dies. A side table without ids would reindex on every delete and take undo
//! with it.
//!
//! The solver core is pure, continuous, and has no density, lattice, persistence, or document-id
//! vocabulary. This module is the adapter: it flattens the sketch's points into a local problem,
//! carries authored relations across the boundary, and applies accepted solved coordinates back.
//! Solved positions stay **authored** state, never `Derived`: the solver reads them as its initial
//! guess and writes them back, and an under-constrained sketch has free degrees of freedom that
//! only the stored position remembers.

#![allow(
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::use_self
)]

use super::{
    Arc, Circle, CircleRadius, EntityId, Point, Segment, Sketch, SketchLength, SketchPoint,
    ABSENT_CENTER,
};
use parametric::sketch::{
    ArcId, BuildError, CircleId, ConstraintId, PointId, Problem, ProblemBuilder, Relation,
    SegmentId, SketchCurve as ParametricSketchCurve, TangentContactError, TangentContactFailure,
};
pub use parametric::sketch::{InternalContainment, LineSide, SymmetryBranch, TangentBranch};
use parametric::EvaluationContext;

/// A stable reference to one authored curve. This is the document boundary equivalent of the
/// solver's local [`ParametricSketchCurve`]: entity ids persist here, local handles do not.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum SketchCurve {
    Segment(EntityId),
    Arc(EntityId),
    Circle(EntityId),
    /// A rational cubic Bézier piece. Existing circular/linear relations refuse it explicitly;
    /// spline-aware relations such as Curvature consume its control geometry directly.
    Bezier(EntityId),
    /// One closed ellipse aggregate, resolved to four rational cubic spans at geometry seams.
    Ellipse(EntityId),
    /// One endpoint/vertex/rho conic aggregate.
    Conic(EntityId),
    /// One fit-point or control-point spline aggregate, resolved to one or more cubic spans.
    Spline(EntityId),
}

impl SketchCurve {
    pub const fn id(self) -> EntityId {
        match self {
            Self::Segment(id)
            | Self::Arc(id)
            | Self::Circle(id)
            | Self::Bezier(id)
            | Self::Ellipse(id)
            | Self::Conic(id)
            | Self::Spline(id) => id,
        }
    }
}

/// What a constraint asserts. Every reference is a stable document entity id, never a slot.
///
/// These are persisted author claims. The parametric adapter resolves their ids once into local
/// handles; cascade and duplicate-assertion policy remain here.
///
/// **Every match on this enum is exhaustive**, here and at each semantic seam, and that is
/// load-bearing rather than stylistic: it makes adding a variant a compiler error at every place
/// that has to answer for it instead of a silent default. In particular, a new two-residual kind
/// assigned one row shifts every later constraint's row and corrupts the whole system.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintKind {
    /// This point does not move, and `at` is where it does not move to.
    ///
    /// The position is stored rather than read from the point at solve time, because a `Fix`
    /// asserts immovability **at a place**: without it, any other relation that dragged the point
    /// would silently redefine what “fixed” meant.
    Fix { point: EntityId, at: SketchPoint },
    /// The segment lies along in-plane axis 0: its ends share axis 1.
    Horizontal { segment: EntityId },
    /// The segment lies along in-plane axis 1: its ends share axis 0.
    Vertical { segment: EntityId },
    /// Two points stand a given distance apart. A dimension: the length is authored, so it keeps
    /// its [`SketchLength`] and survives a density re-target like every other authored quantity.
    Distance {
        from: EntityId,
        to: EntityId,
        length: SketchLength,
    },
    /// Two points occupy one place.
    ///
    /// It is a CONSTRAINT and not a merge, although a merge is the other design available.
    /// Merging two points into one is destructive in a way the author cannot see afterwards: the
    /// second id is gone, every segment that named it now names the first, and deleting the
    /// coincidence cannot put the drawing back. As an assertion it deletes like any other and the
    /// two points spring apart, which is what “remove this constraint” should mean.
    Coincident { first: EntityId, second: EntityId },
    /// Two segments run the same way. The residual is the SINE of the angle between them, so it is
    /// dimensionless and reads the same on a 3-voxel segment and a 300-voxel one.
    Parallel { first: EntityId, second: EntityId },
    /// Two segments meet at a right angle — the cosine of the angle between them, normalized for
    /// the same reason `Parallel` is.
    Perpendicular { first: EntityId, second: EntityId },
    /// Two segments have equal length without asserting what that shared length is. The pair is
    /// free to settle anywhere, unlike two Distance dimensions that each carry one authored value.
    Equal { first: EntityId, second: EntityId },
    /// The point sits halfway along the segment. Two residuals — it pins both coordinates,
    /// because “halfway” names a place and not merely a line.
    Midpoint { point: EntityId, segment: EntityId },
    /// Two segments lie on one infinite line.
    ///
    /// Two residuals, not one: it says parallel AND no offset, and asking for the distance of each
    /// of `second`'s ends from `first`'s line says both at once without reconciling two
    /// differently-scaled rows.
    Collinear { first: EntityId, second: EntityId },
    /// Two finite authored curves touch at this stable solution branch. `first` and `second` are
    /// canonicalized by stable entity id; an internal branch names that persisted order.
    Tangent {
        first: SketchCurve,
        second: SketchCurve,
        branch: TangentBranch,
    },
    /// Two circular authored curves share one center while retaining independent radii.
    Concentric {
        first: SketchCurve,
        second: SketchCurve,
    },
    /// Two same-kind authored curves mirror across an explicit segment axis.
    Symmetry {
        first: SketchCurve,
        second: SketchCurve,
        axis: EntityId,
        branch: SymmetryBranch,
    },
    /// Both coordinates of this point lie on `phase + n * pitch`. The values are authored sketch
    /// lengths so density retargeting keeps a block lattice physical while a voxel lattice stays
    /// in voxel units.
    Quantize {
        point: EntityId,
        pitch: SketchLength,
        phase: SketchLength,
    },
}

impl ConstraintKind {
    /// Construct a Tangent with deterministic member ordering. Internal containment follows the
    /// members when they swap; LineSide deliberately remains tied to the segment direction.
    /// `EntityId` is minted from Sketch's one document-wide counter, so ids order curves across
    /// Segment/Arc/Circle stores without a kind tie-breaker.
    pub const fn tangent(first: SketchCurve, second: SketchCurve, branch: TangentBranch) -> Self {
        if first.id() <= second.id() {
            Self::Tangent {
                first,
                second,
                branch,
            }
        } else {
            Self::Tangent {
                first: second,
                second: first,
                branch: branch.remap_for_swapped_members(),
            }
        }
    }

    /// Construct a branch-free circular pair in stable entity-id order.
    pub const fn concentric(first: SketchCurve, second: SketchCurve) -> Self {
        if first.id() <= second.id() {
            Self::Concentric { first, second }
        } else {
            Self::Concentric {
                first: second,
                second: first,
            }
        }
    }

    /// Construct Symmetry with canonical subjects while retaining the axis's reference role.
    pub const fn symmetry(
        first: SketchCurve,
        second: SketchCurve,
        axis: EntityId,
        branch: SymmetryBranch,
    ) -> Self {
        if first.id() <= second.id() {
            Self::Symmetry {
                first,
                second,
                axis,
                branch,
            }
        } else {
            Self::Symmetry {
                first: second,
                second: first,
                axis,
                branch,
            }
        }
    }

    pub(super) fn normalized(self) -> Self {
        match self {
            Self::Tangent {
                first,
                second,
                branch,
            } => Self::tangent(first, second, branch),
            Self::Concentric { first, second } => Self::concentric(first, second),
            Self::Symmetry {
                first,
                second,
                axis,
                branch,
            } => Self::symmetry(first, second, axis, branch),
            other => other,
        }
    }
    /// Every point id named directly, for cascade and liveness checks.
    pub(super) fn points(&self) -> Vec<EntityId> {
        match *self {
            Self::Fix { point, .. }
            | Self::Quantize { point, .. }
            | Self::Midpoint { point, .. } => vec![point],
            Self::Distance { from, to, .. } => vec![from, to],
            Self::Coincident { first, second } => vec![first, second],
            Self::Horizontal { .. }
            | Self::Vertical { .. }
            | Self::Parallel { .. }
            | Self::Perpendicular { .. }
            | Self::Equal { .. }
            | Self::Collinear { .. }
            | Self::Tangent { .. }
            | Self::Concentric { .. }
            | Self::Symmetry { .. } => Vec::new(),
        }
    }

    /// Whether two persisted assertions make the same claim about the same geometry. Stored values
    /// deliberately do not participate: two Fixes on a point are the same assertion whether or
    /// not their targets agree, because changing a fix is delete-then-add rather than two claims.
    pub fn is_about_the_same_as(&self, other: Self) -> bool {
        if let (
            Self::Symmetry {
                first,
                second,
                axis,
                ..
            },
            Self::Symmetry {
                first: other_first,
                second: other_second,
                axis: other_axis,
                ..
            },
        ) = (*self, other)
        {
            return first.id() == other_first.id()
                && second.id() == other_second.id()
                && axis == other_axis;
        }
        std::mem::discriminant(self) == std::mem::discriminant(&other)
            && self.subject() == other.subject()
    }

    /// The comparable subject pair. Symmetric pairs are canonicalized, so Distance A→B is the same
    /// assertion as B→A. Midpoint remains ordered because its point and segment belong to different
    /// entity stores and play different semantic roles.
    fn subject(&self) -> [EntityId; 2] {
        match *self {
            Self::Fix { point, .. } | Self::Quantize { point, .. } => [point, point],
            Self::Horizontal { segment } | Self::Vertical { segment } => [segment, segment],
            Self::Distance { from, to, .. } => [from.min(to), from.max(to)],
            Self::Coincident { first, second }
            | Self::Parallel { first, second }
            | Self::Perpendicular { first, second }
            | Self::Equal { first, second }
            | Self::Collinear { first, second } => [first.min(second), first.max(second)],
            Self::Tangent { first, second, .. } | Self::Concentric { first, second } => {
                let (first, second) = (first.id(), second.id());
                [first.min(second), first.max(second)]
            }
            Self::Symmetry { first, second, .. } => [first.id(), second.id()],
            Self::Midpoint { point, segment } => [point, segment],
        }
    }

    /// Every segment id named, for cascade and liveness checks.
    pub(super) fn segments(&self) -> Vec<EntityId> {
        match *self {
            Self::Horizontal { segment }
            | Self::Vertical { segment }
            | Self::Midpoint { segment, .. } => vec![segment],
            Self::Parallel { first, second }
            | Self::Perpendicular { first, second }
            | Self::Equal { first, second }
            | Self::Collinear { first, second } => vec![first, second],
            Self::Tangent { first, second, .. } => [first, second]
                .into_iter()
                .filter_map(|curve| match curve {
                    SketchCurve::Segment(id) => Some(id),
                    SketchCurve::Arc(_)
                    | SketchCurve::Circle(_)
                    | SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => None,
                })
                .collect(),
            Self::Symmetry {
                first,
                second,
                axis,
                ..
            } => std::iter::once(axis)
                .chain([first, second].into_iter().filter_map(|curve| match curve {
                    SketchCurve::Segment(id) => Some(id),
                    SketchCurve::Arc(_)
                    | SketchCurve::Circle(_)
                    | SketchCurve::Bezier(_)
                    | SketchCurve::Ellipse(_)
                    | SketchCurve::Conic(_)
                    | SketchCurve::Spline(_) => None,
                }))
                .collect(),
            Self::Fix { .. }
            | Self::Quantize { .. }
            | Self::Distance { .. }
            | Self::Coincident { .. }
            | Self::Concentric { .. } => Vec::new(),
        }
    }

    /// Every curve id named by a generic curve relation, for cascade/repair.
    pub(super) fn curves(&self) -> Vec<SketchCurve> {
        match *self {
            Self::Tangent { first, second, .. } | Self::Concentric { first, second } => {
                vec![first, second]
            }
            Self::Symmetry { first, second, .. } => vec![first, second],
            _ => Vec::new(),
        }
    }

    pub(super) const fn tangent_is_structurally_valid(&self) -> bool {
        match *self {
            Self::Tangent {
                first,
                second,
                branch,
            } => {
                first.id() != second.id()
                    && matches!(
                        (first, second, branch),
                        (
                            SketchCurve::Segment(_),
                            SketchCurve::Arc(_) | SketchCurve::Circle(_),
                            TangentBranch::Line(_)
                        ) | (
                            SketchCurve::Arc(_) | SketchCurve::Circle(_),
                            SketchCurve::Segment(_),
                            TangentBranch::Line(_)
                        ) | (
                            SketchCurve::Arc(_) | SketchCurve::Circle(_),
                            SketchCurve::Arc(_) | SketchCurve::Circle(_),
                            TangentBranch::External | TangentBranch::Internal { .. }
                        )
                    )
            }
            _ => true,
        }
    }

    pub(super) const fn concentric_is_structurally_valid(&self) -> bool {
        match *self {
            Self::Concentric { first, second } => {
                first.id() != second.id()
                    && matches!(first, SketchCurve::Arc(_) | SketchCurve::Circle(_))
                    && matches!(second, SketchCurve::Arc(_) | SketchCurve::Circle(_))
            }
            _ => true,
        }
    }

    pub(super) const fn symmetry_is_structurally_valid(&self) -> bool {
        match *self {
            Self::Symmetry {
                first,
                second,
                axis,
                branch,
            } => {
                first.id() != second.id()
                    && first.id() != axis
                    && second.id() != axis
                    && matches!(
                        (first, second, branch),
                        (
                            SketchCurve::Segment(_),
                            SketchCurve::Segment(_),
                            SymmetryBranch::Direct | SymmetryBranch::Reversed
                        ) | (
                            SketchCurve::Arc(_),
                            SketchCurve::Arc(_),
                            SymmetryBranch::Direct | SymmetryBranch::Reversed
                        ) | (
                            SketchCurve::Circle(_),
                            SketchCurve::Circle(_),
                            SymmetryBranch::Centers
                        )
                    )
            }
            _ => true,
        }
    }
}

/// A stable, individually selectable and deletable constraint entity.
///
/// Redundancy is retained and flagged rather than refused: an implied assertion can still carry
/// durable author intent, and is a fact the author may want to see rather than lose.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    /// Stable identity, from the same counter as every other entity.
    pub id: EntityId,
    /// What it asserts.
    #[serde(deserialize_with = "deserialize_constraint_kind")]
    pub kind: ConstraintKind,
    /// Whether the solver found it redundant when it was added — it holds, but adds no
    /// information. Redundancy is sometimes the intent, so it is flagged rather than refused.
    #[serde(default)]
    pub redundant: bool,
}

/// Persistence boundary for a stored constraint. Every unordered curve pair is normalized to
/// canonical member order before repair makes the document-specific liveness/type decision.
fn deserialize_constraint_kind<'de, D>(deserializer: D) -> Result<ConstraintKind, D::Error>
where
    D: serde::Deserializer<'de>,
{
    <ConstraintKind as serde::Deserialize>::deserialize(deserializer)
        .map(ConstraintKind::normalized)
}

/// Why a requested assertion cannot be retained by the document — **and what to blame**.
///
/// Every refusal that has a culprit names it. A diagnosis the author cannot act on is barely a
/// diagnosis: “it fights something” leaves them to find the something, and on a drawing carrying
/// twenty assertions that is the whole of the work. Since constraints are selectable entities with
/// badges, an id is all the shell needs to point at one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintRefusal {
    /// A fixed curve source needs the document evaluation context; no cached voxel value is used.
    MissingEvaluationContext,
    /// Tangent is intentionally not a relation between two line segments; Parallel owns that
    /// authoring claim, while a malformed branch/type combination names no meaningful assertion.
    InvalidTangent {
        constraint: Option<EntityId>,
        error: TangentContactError,
    },
    /// Concentric accepts two distinct arcs or circles and no other geometry.
    InvalidConcentric,
    /// Symmetry requires two same-kind curves and one distinct nondegenerate segment axis.
    InvalidSymmetry,
    /// The request names geometry the store does not hold.
    UnknownEntity,
    /// Its own terms cannot be met by any drawing: for example a negative distance or a horizontal
    /// assertion on one segment endpoint twice. There is nothing standing to blame.
    Impossible,
    /// The system it would join has no solution: it fights what is already asserted.
    Unsatisfiable {
        /// Standing constraints it cannot coexist with, found by leave-one-out. **Empty means
        /// undetermined, never innocent** — a conflict needing two removals leaves no single
        /// culprit, and claiming one would be worse than admitting none.
        fights: Vec<EntityId>,
    },
    /// A solution exists only by deleting meaningful geometry. This differs from Unsatisfiable:
    /// the assertions agree on a singular answer, but the answer is not the drawing the author
    /// asked to preserve. Implication is structural rather than experimental because a prior solve
    /// has already moved the drawing; dropping a relation cannot reconstruct the geometry it once
    /// produced, while the relation graph always identifies what still holds the shape.
    WouldCollapse {
        /// The segment or arc that would lose its extent.
        entity: EntityId,
        /// Standing constraints that already act on that geometry. This is structural rather than
        /// experimental: a prior solve has already moved the drawing, and releasing an assertion
        /// does not undo its effect. What the author needs is what else holds the shape, a question
        /// the relation graph can always answer.
        implicated: Vec<EntityId>,
    },
    /// The same kind of assertion already stands on the same geometry. One constraint of a kind
    /// per entity set: a second `Horizontal` says nothing the first did not, and a second `Fix` is
    /// a re-fix, which is a delete and add rather than two claims about one place.
    AlreadyAsserted {
        /// The standing assertion, so “you already have this” lights a badge rather than starts a
        /// hunt.
        existing: EntityId,
    },
}

impl ConstraintRefusal {
    /// Every constraint this refusal blames, for a caller that wants to light them up. Empty when
    /// the refusal has no culprit or none could be isolated.
    pub fn culprits(&self) -> Vec<EntityId> {
        match self {
            Self::InvalidTangent {
                constraint: Some(constraint),
                ..
            } => vec![*constraint],
            Self::MissingEvaluationContext
            | Self::UnknownEntity
            | Self::Impossible
            | Self::InvalidConcentric
            | Self::InvalidSymmetry
            | Self::InvalidTangent {
                constraint: None, ..
            } => Vec::new(),
            Self::Unsatisfiable { fights } => fights.clone(),
            Self::WouldCollapse { implicated, .. } => implicated.clone(),
            Self::AlreadyAsserted { existing } => vec![*existing],
        }
    }
}

/// A validated local problem plus one-way stable-id mappings for atomic write-back and diagnostics.
/// Local owner-tagged handles never enter persistence; only this adapter translates them back to
/// stable document identities after a typed parametric outcome is accepted. The mappings are kept
/// beside the prepared problem so every result — a conflict, a collapsed curve, or a solution —
/// returns to the exact persisted entity that produced it.
pub(super) struct PreparedProblem {
    problem: Problem,
    points: Vec<(EntityId, PointId)>,
    segments: Vec<(EntityId, SegmentId)>,
    arcs: Vec<(EntityId, ArcId, parametric::sketch::ParameterId)>,
    circles: Vec<(EntityId, CircleId, parametric::sketch::ParameterId)>,
    constraints: Vec<(EntityId, ConstraintId)>,
}

pub(super) enum TrialMapError {
    UnmappedGeometry,
    Request(parametric::sketch::RequestError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrepareError {
    MissingEvaluationContext,
    InvalidDocumentGeometry,
    InvalidLocalProblem(BuildError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct StandingTangentFailure {
    pub(super) constraint: EntityId,
    pub(super) error: TangentContactError,
}

/// Why an otherwise accepted local solution cannot be atomically written into document state.
/// This remains separate from evaluation-context failures: a caller must never be told to supply
/// density when the actual problem is an invalid scalar or a mismatched solver handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarWritebackError {
    MissingSolutionPoint,
    MissingSolutionParameter,
    ParameterKindMismatch,
    SweepNotRepresentable,
    RadiusNotRepresentable,
    MissingDocumentEntity,
}

pub(super) struct ApplyPlan {
    points: Vec<Point>,
    arcs: Vec<Arc>,
    circles: Vec<Circle>,
}

impl ApplyPlan {
    pub(super) fn apply(self, sketch: &mut Sketch) {
        sketch.points = self.points;
        sketch.arcs = self.arcs;
        sketch.circles = self.circles;
    }
}

impl PreparedProblem {
    pub(super) fn settle(&self) -> parametric::sketch::Settled {
        self.problem.settle()
    }

    pub(super) fn analyze(&self) -> parametric::sketch::Analysis {
        self.problem.analyze()
    }

    pub(super) fn validate_current(&self) -> parametric::sketch::CurrentValidation {
        self.problem.validate_current()
    }

    /// Map deterministic kernel leave-one-out conflicts back to persistent constraint ids.
    pub(super) fn standing_conflicts(&self) -> Result<Vec<EntityId>, PrepareError> {
        let mut conflicts: Vec<_> = self
            .problem
            .standing_conflicts()
            .into_iter()
            .map(|constraint| {
                self.constraint(constraint)
                    .ok_or(PrepareError::InvalidDocumentGeometry)
            })
            .collect::<Result<_, _>>()?;
        conflicts.sort_unstable();
        Ok(conflicts)
    }

    pub(super) fn standing_tangent_failure(
        &self,
        failure: TangentContactFailure,
    ) -> Result<StandingTangentFailure, PrepareError> {
        self.constraint(failure.constraint)
            .map(|constraint| StandingTangentFailure {
                constraint,
                error: failure.error,
            })
            .ok_or(PrepareError::InvalidDocumentGeometry)
    }

    pub(super) fn first_tangent_contact_failure(
        &self,
        solution: &parametric::sketch::Solution,
    ) -> Result<Option<StandingTangentFailure>, PrepareError> {
        self.problem
            .first_tangent_contact_failure(solution)
            .map(|failure| self.standing_tangent_failure(failure))
            .transpose()
    }

    pub(super) fn trial_add(
        &self,
        kind: ConstraintKind,
    ) -> Result<parametric::sketch::TrialAdd, TrialMapError> {
        let relation = self.relation(kind).ok_or(TrialMapError::UnmappedGeometry)?;
        self.problem
            .trial_add(relation)
            .map_err(TrialMapError::Request)
    }

    pub(super) fn drag(
        &self,
        held: EntityId,
        at: [f64; 2],
    ) -> Result<parametric::sketch::DragOutcome, parametric::sketch::RequestError> {
        let point = self
            .point(held)
            .ok_or(parametric::sketch::RequestError::UnknownPoint)?;
        self.problem.drag(point, at)
    }

    pub(super) fn plan_apply(
        &self,
        points: &[Point],
        arcs: &[Arc],
        circles: &[Circle],
        solution: &parametric::sketch::Solution,
    ) -> Result<ApplyPlan, ScalarWritebackError> {
        let mut points = points.to_vec();
        let mut arcs = arcs.to_vec();
        let mut circles = circles.to_vec();
        for (id, point) in &self.points {
            let at = solution
                .position(*point)
                .ok_or(ScalarWritebackError::MissingSolutionPoint)?;
            let point = points
                .iter_mut()
                .find(|point| point.id == *id)
                .ok_or(ScalarWritebackError::MissingDocumentEntity)?;
            point.at = SketchPoint::from_continuous(at[0], at[1]);
        }
        for (id, _, parameter) in &self.arcs {
            let arc = arcs
                .iter_mut()
                .find(|arc| arc.id == *id)
                .ok_or(ScalarWritebackError::MissingDocumentEntity)?;
            if arc.bulge.free_value().is_none() {
                continue;
            }
            let parametric::sketch::ParameterValue::SweepDegrees(value) = solution
                .parameter(*parameter)
                .ok_or(ScalarWritebackError::MissingSolutionParameter)?
            else {
                return Err(ScalarWritebackError::ParameterKindMismatch);
            };
            let value = parametric::units::AngleMeasurement::try_from_degrees_f64(value)
                .map_err(|_| ScalarWritebackError::SweepNotRepresentable)?;
            arc.replace_free_sweep(value);
        }
        for (id, _, parameter) in &self.circles {
            let circle = circles
                .iter_mut()
                .find(|circle| circle.id == *id)
                .ok_or(ScalarWritebackError::MissingDocumentEntity)?;
            if circle.radius.free_value().is_none() {
                continue;
            }
            let parametric::sketch::ParameterValue::Radius(value) = solution
                .parameter(*parameter)
                .ok_or(ScalarWritebackError::MissingSolutionParameter)?
            else {
                return Err(ScalarWritebackError::ParameterKindMismatch);
            };
            let value = super::ResolvedLength::try_from_f64(value)
                .map_err(|_| ScalarWritebackError::RadiusNotRepresentable)?;
            circle.radius = CircleRadius::free(value);
        }
        Ok(ApplyPlan {
            points,
            arcs,
            circles,
        })
    }

    pub(super) fn point(&self, id: EntityId) -> Option<PointId> {
        self.points
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, point)| *point)
    }

    pub(super) fn constraint(&self, id: ConstraintId) -> Option<EntityId> {
        self.constraints
            .iter()
            .find(|(_, local)| *local == id)
            .map(|(stable, _)| *stable)
    }

    pub(super) fn curve(&self, curve: ParametricSketchCurve) -> Option<EntityId> {
        match curve {
            ParametricSketchCurve::Segment(key) => self
                .segments
                .iter()
                .find(|(_, local)| *local == key)
                .map(|(stable, _)| *stable),
            ParametricSketchCurve::Arc(key) => self
                .arcs
                .iter()
                .find(|(_, local, _)| *local == key)
                .map(|(stable, _, _)| *stable),
            ParametricSketchCurve::Circle(key) => self
                .circles
                .iter()
                .find(|(_, local, _)| *local == key)
                .map(|(stable, _, _)| *stable),
        }
    }

    fn relation(&self, kind: ConstraintKind) -> Option<Relation> {
        relation_for(
            kind,
            &self.points,
            &self.segments,
            &self.arcs,
            &self.circles,
        )
    }
}

#[allow(clippy::too_many_lines)]
fn relation_for(
    kind: ConstraintKind,
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
    arcs: &[(EntityId, ArcId, parametric::sketch::ParameterId)],
    circles: &[(EntityId, CircleId, parametric::sketch::ParameterId)],
) -> Option<Relation> {
    let point = |id| {
        points
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, local)| *local)
    };
    let segment = |id| {
        segments
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, local)| *local)
    };
    let curve = |curve: SketchCurve| match curve {
        SketchCurve::Segment(id) => segment(id).map(ParametricSketchCurve::Segment),
        SketchCurve::Arc(id) => arcs
            .iter()
            .find(|(candidate, _, _)| *candidate == id)
            .map(|(_, local, _)| ParametricSketchCurve::Arc(*local)),
        SketchCurve::Circle(id) => circles
            .iter()
            .find(|(candidate, _, _)| *candidate == id)
            .map(|(_, local, _)| ParametricSketchCurve::Circle(*local)),
        SketchCurve::Bezier(_)
        | SketchCurve::Ellipse(_)
        | SketchCurve::Conic(_)
        | SketchCurve::Spline(_) => None,
    };
    match kind {
        ConstraintKind::Fix { point: id, at } => point(id).map(|point| Relation::Fix {
            point,
            at: at.in_plane(),
        }),
        ConstraintKind::Quantize {
            point: id,
            pitch,
            phase,
        } => point(id).map(|point| Relation::Quantize {
            point,
            pitch: pitch.value(),
            phase: phase.value(),
        }),
        ConstraintKind::Horizontal { segment: id } => {
            segment(id).map(|segment| Relation::Horizontal { segment })
        }
        ConstraintKind::Vertical { segment: id } => {
            segment(id).map(|segment| Relation::Vertical { segment })
        }
        ConstraintKind::Distance { from, to, length } => {
            point(from)
                .zip(point(to))
                .map(|(from, to)| Relation::Distance {
                    from,
                    to,
                    length: length.value(),
                })
        }
        ConstraintKind::Coincident { first, second } => point(first)
            .zip(point(second))
            .map(|(first, second)| Relation::Coincident { first, second }),
        ConstraintKind::Parallel { first, second } => segment(first)
            .zip(segment(second))
            .map(|(first, second)| Relation::Parallel { first, second }),
        ConstraintKind::Perpendicular { first, second } => segment(first)
            .zip(segment(second))
            .map(|(first, second)| Relation::Perpendicular { first, second }),
        ConstraintKind::Equal { first, second } => segment(first)
            .zip(segment(second))
            .map(|(first, second)| Relation::Equal { first, second }),
        ConstraintKind::Midpoint {
            point: id,
            segment: edge,
        } => point(id)
            .zip(segment(edge))
            .map(|(point, segment)| Relation::Midpoint { point, segment }),
        ConstraintKind::Collinear { first, second } => segment(first)
            .zip(segment(second))
            .map(|(first, second)| Relation::Collinear { first, second }),
        ConstraintKind::Tangent {
            first,
            second,
            branch,
        } => curve(first)
            .zip(curve(second))
            .map(|(first, second)| Relation::Tangent {
                first,
                second,
                branch,
            }),
        ConstraintKind::Concentric { first, second } => curve(first)
            .zip(curve(second))
            .map(|(first, second)| Relation::Concentric { first, second }),
        ConstraintKind::Symmetry {
            first,
            second,
            axis,
            branch,
        } => curve(first)
            .zip(curve(second))
            .zip(segment(axis))
            .map(|((first, second), axis)| Relation::Symmetry {
                first,
                second,
                axis,
                branch,
            }),
    }
}

fn add_constraints(
    builder: &mut ProblemBuilder,
    constraints: &[Constraint],
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
    arcs: &[(EntityId, ArcId, parametric::sketch::ParameterId)],
    circles: &[(EntityId, CircleId, parametric::sketch::ParameterId)],
) -> Vec<(EntityId, ConstraintId)> {
    constraints
        .iter()
        .filter_map(|constraint| {
            relation_for(constraint.kind, points, segments, arcs, circles)
                .map(|relation| (constraint.id, builder.add_constraint(relation)))
        })
        .collect()
}

/// Build in stable-id order. The parametric kernel intentionally knows no document ids, density,
/// or authored scalar storage; it receives only resolved positions, topology, and relations.
/// Sorting is not a semantic ordering of the document: it gives the local arithmetic layout a
/// reproducible order while stable ids remain the only identity exposed to callers.
pub(super) fn prepare(
    sketch: &Sketch,
    constraints: &[Constraint],
    context: Option<EvaluationContext>,
) -> Result<PreparedProblem, PrepareError> {
    let mut builder = ProblemBuilder::new();
    let mut ordered_points: Vec<&Point> = sketch.points.iter().collect();
    ordered_points.sort_by_key(|point| point.id);
    let points: Vec<(EntityId, PointId)> = ordered_points
        .into_iter()
        .map(|point| (point.id, builder.add_point(point.at.in_plane())))
        .collect();
    let point = |id| {
        points
            .iter()
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, local)| *local)
    };

    let mut ordered_segments: Vec<&Segment> = sketch.segments.iter().collect();
    ordered_segments.sort_by_key(|segment| segment.id);
    let mut segments = Vec::with_capacity(ordered_segments.len());
    for segment in ordered_segments {
        let (Some(from), Some(to)) = (point(segment.from), point(segment.to)) else {
            return Err(PrepareError::InvalidDocumentGeometry);
        };
        segments.push((segment.id, builder.add_segment(from, to)));
    }
    let mut arcs: Vec<&Arc> = sketch.arcs.iter().collect();
    arcs.sort_by_key(|arc| arc.id);
    let mut local_arcs = Vec::new();
    for arc in arcs {
        if arc.center == ABSENT_CENTER {
            continue;
        }
        let (Some(center), Some(from), Some(to)) =
            (point(arc.center), point(arc.from), point(arc.to))
        else {
            return Err(PrepareError::InvalidDocumentGeometry);
        };
        let sweep = match (arc.bulge.free_value(), arc.bulge.fixed_source()) {
            (Some(value), None) => builder.add_free_signed_sweep(value.to_degrees_f64()),
            (None, Some(source)) => builder.add_fixed_signed_sweep(source.to_degrees_f64()),
            _ => return Err(PrepareError::InvalidDocumentGeometry),
        }
        .map_err(PrepareError::InvalidLocalProblem)?;
        let local = builder.add_arc(center, from, to, sweep);
        local_arcs.push((arc.id, local, sweep));
    }

    let mut circles: Vec<&Circle> = sketch.circles.iter().collect();
    circles.sort_by_key(|circle| circle.id);
    let mut local_circles = Vec::new();
    for circle in circles {
        let center = point(circle.center).ok_or(PrepareError::InvalidDocumentGeometry)?;
        let radius = match (circle.radius.free_value(), circle.radius.fixed_source()) {
            (Some(value), None) => builder.add_free_positive_radius(value.value()),
            (None, Some(source)) => {
                let context = context.ok_or(PrepareError::MissingEvaluationContext)?;
                builder.add_fixed_positive_radius(source.to_voxel_rational(context).to_f64())
            }
            _ => return Err(PrepareError::InvalidDocumentGeometry),
        }
        .map_err(PrepareError::InvalidLocalProblem)?;
        let local = builder.add_circle(center, radius);
        local_circles.push((circle.id, local, radius));
    }

    let local_constraints = add_constraints(
        &mut builder,
        constraints,
        &points,
        &segments,
        &local_arcs,
        &local_circles,
    );
    let problem = builder
        .finish()
        .map_err(PrepareError::InvalidLocalProblem)?;
    Ok(PreparedProblem {
        problem,
        points,
        segments,
        arcs: local_arcs,
        circles: local_circles,
        constraints: local_constraints,
    })
}
