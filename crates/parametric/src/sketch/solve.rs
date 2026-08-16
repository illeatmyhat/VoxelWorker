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

use super::curvature::{curvature_residual, direction_residual, JointSpan, SpanEnd};
use super::curve::{
    ArcDomain, CircularCurve, CurveGeometry, COLLAPSE_TOLERANCE as COLLAPSED_SPAN,
    SATISFACTION_TOLERANCE as SATISFIED_RESIDUAL,
};
use super::model::{SolveOutcome, SolveReport};
use super::spline::{control_point_spline, fit_point_spline, station_length, SplineCandidate};
use super::symmetry::{
    residuals as symmetry_residuals, symmetry_witness, SymmetryBranch, SymmetryError,
    SymmetryWitness,
};
use super::tangent::{
    residual as tangent_residual, tangent_contact, TangentBranch, TangentContact,
    TangentContactError,
};
use crate::ResolvedLength;
use std::sync::atomic::{AtomicU64, Ordering};
use substrate::graph::biconnected_blocks;
#[cfg(test)]
use substrate::nonlinear_least_squares::{
    first_subset_disagreement, first_undeclared_read, ColumnGrouping,
};
use substrate::nonlinear_least_squares::{
    jacobian, rank, search as search_nlls, solve as solve_nlls, ResidualReads, ResidualSystem,
    SearchReport as SubstrateSearchReport, SolveOutcome as SubstrateSolveOutcome,
    SolveReport as SubstrateSolveReport, SolveSettings,
};

/// A trial whose residual norm is at or below this tolerance has met its relations.
///
/// The search outcome is deliberately not this test: residual tolerance is absolute while a step
/// tolerance is relative to parameter magnitude, so a large yet satisfied drawing may stop with a
/// `Stalled` status. Search status describes the path; residual norm describes the answer.
/// The shared center witness for a satisfied Concentric pair. This uses the same residual norm
/// threshold as the solver, so overlays and numerical acceptance cannot disagree at the boundary.
pub fn concentric_center(first: [f64; 2], second: [f64; 2]) -> Option<[f64; 2]> {
    let delta = [first[0] - second[0], first[1] - second[1]];
    (first.into_iter().chain(second).all(f64::is_finite)
        && delta[0].hypot(delta[1]) <= SATISFIED_RESIDUAL)
        .then_some([(first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0])
}

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

/// A local spline: the points that shape it, and how they do.
///
/// The solver stores no curve for it. A spline's shape is a FUNCTION of points it already holds,
/// so it is refit from their live coordinates on every residual pass — which is what makes a
/// finite-difference column through a fit point reach the curve at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SplineId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstraintId {
    owner: u64,
    index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchCurve {
    Segment(SegmentId),
    Arc(ArcId),
    Circle(CircleId),
}

/// One side of a stated angle: something the drawing gives a DIRECTION for.
///
/// A straight run has the same direction at every point on it, so a segment arm names no place. A
/// curve that turns has a different direction everywhere, so an angle to one is not a question
/// until a place is named — and what an arc arm names is an END, because an end is on its own arc
/// by construction. Naming a free point instead would put a coincidence between the arm and the
/// curve that every later solve has to keep agreeing about.
///
/// A whole circle has no ends and so cannot be an arm at all. The thing an author wants there is a
/// tangency, which [`Relation::Tangent`] already states, at a contact the drawing finds for itself.
///
/// Neither arm carries a SENSE, and neither needs to: the angle row is a sine, which repeats every
/// half turn, so flipping an arm end for end leaves the residual exactly where it was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AngleArm {
    /// A straight curve, read along the direction it was drawn in.
    Segment(SegmentId),
    /// The direction an arc leaves at one of its own two ends — perpendicular to the radius
    /// standing there.
    ArcEnd { arc: ArcId, end: SpanEnd },
}

impl AngleArm {
    /// The segment this arm reads, for the liveness checks that are about segments.
    const fn segment(self) -> Option<SegmentId> {
        match self {
            Self::Segment(segment) => Some(segment),
            Self::ArcEnd { .. } => None,
        }
    }

    /// The curve this arm reads, whichever kind it is — what a scope has to include.
    const fn curve(self) -> SketchCurve {
        match self {
            Self::Segment(segment) => SketchCurve::Segment(segment),
            Self::ArcEnd { arc, .. } => SketchCurve::Arc(arc),
        }
    }
}

/// The physical kind of an intrinsic scalar. Typed results prevent an adapter from writing a
/// solved radius through the arc-angle door, or vice versa.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterKind {
    PositiveRadius,
    /// Where along a spline a point standing on it stands, measured in units of that spline's
    /// [`station_length`] so a step along the curve costs about what a step
    /// across the drawing costs.
    SplineStation,
}

/// A solved intrinsic scalar paired with its kind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ParameterValue {
    Radius(f64),
    Station(f64),
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
    /// Two points stand a given distance apart ALONG ONE AXIS of the sketch plane — how far apart
    /// they are across it, or up it, rather than how far apart they are.
    ///
    /// `axis` indexes the plane's own coordinates, so `0` states a width and `1` states a height.
    /// The same convention [`Relation::Horizontal`] resolves to, and for the same reason: a plane
    /// has its own axes and a dimension read against the camera's would answer differently from a
    /// different camera.
    ///
    /// A separate relation and not a [`Distance`](Self::Distance) with a direction, because it is
    /// a different claim with a different Jacobian: this row touches ONE coordinate of each point
    /// and is linear in it, where a distance touches both and is not. Two points can carry both
    /// axes at once and be fully placed by them, which is the ordinary way a drawing gets pinned.
    ///
    /// The residual is `|Δ| − length`, whose slope is defined everywhere the claim can hold —
    /// `Δ = 0` is not on the solution manifold for a positive length. It does not FORBID a solve
    /// from dragging one point clean through the other and settling at `−length`; a stored sign
    /// would, at the cost of a branch the author never chose. Not seen, and cheap to add if it is.
    AxisDistance {
        from: PointId,
        to: PointId,
        axis: usize,
        length: f64,
    },
    /// A point stands a given distance from the LINE a segment draws.
    ///
    /// Measured across that line — the shortest way — and against the whole line rather than the
    /// run the segment happens to occupy: a standoff is a fact about a direction and a point, and
    /// it does not stop being true past where the drawn part ends.
    ///
    /// **Two parallel lines a distance apart is this claim**, taken at one of the second line's
    /// ends, and so is a point standing off a line. They are one relation because they are one
    /// piece of arithmetic; what distinguishes them is which point the author picked, which the
    /// document records rather than the solver inferring.
    ///
    /// The residual is `|cross(unit(to − from), point − from)| − distance`, absolute for the same
    /// reason [`AxisDistance`](Self::AxisDistance) is: the slope is a sign, defined everywhere the
    /// claim can hold, and it does not forbid a solve from carrying the point clean across the
    /// line and settling on the other side of it. A stored side would, at the cost of a branch the
    /// author never chose. Not seen, and cheap to add if it is.
    PointLineDistance {
        point: PointId,
        line: SegmentId,
        distance: f64,
    },
    /// A curve that turns stands a given distance from its own center, everywhere.
    ///
    /// One row, and the same row whether the curve is an arc or a circle: both answer
    /// [`CurveGeometry::Circular`], and both name a radius the solver holds as a column — the
    /// circle's authored one, the arc's minted beside its three points. That is the whole reason
    /// this relation can be written once. A segment has no center and no radius, and the document
    /// refuses to build one against it rather than this arm having to.
    Radius { curve: SketchCurve, length: f64 },
    /// Two curves that turn differ in size by this much.
    ///
    /// Where the two share a center — the drawing this exists for — that difference IS the gap
    /// between the two rims, measured straight out along any radius. The row does not assert the
    /// shared center: that is `Concentric`'s claim, stated separately, and an author who wants
    /// both says both.
    ///
    /// The row is `|r_second - r_first| - distance`, unsigned for the reason `Distance`'s is: the
    /// author asked how far apart the two rims stand and not which of them is the larger, so a
    /// signed row would refuse a drawing that had merely been drawn the other way round. The cost
    /// is that a solve could in principle carry one rim through the other and settle on the far
    /// side. Not observed; recorded here rather than guarded.
    RimGap {
        first: SketchCurve,
        second: SketchCurve,
        distance: f64,
    },
    /// Two independently-addressable points occupy one place. This relation deliberately does not
    /// merge their handles: merging destroys an id, rewrites every segment that named it, and makes
    /// deleting the assertion unable to restore the drawing.
    Coincident { first: PointId, second: PointId },
    /// Parallel uses sine between unit directions, so it is scale independent.
    Parallel { first: SegmentId, second: SegmentId },
    /// Perpendicular uses normalized cosine for the same scale-independent reason.
    Perpendicular { first: SegmentId, second: SegmentId },
    /// Two segments meet at a stated angle, measured turning from `first` to `second`.
    ///
    /// The row is `sin(turn - radians)`: dimensionless, scale independent, and zero exactly when
    /// the turn is the one asked for. `Parallel` and `Perpendicular` are the two values of it that
    /// need no number — at zero the row IS Parallel's cross product, and at a quarter turn it is
    /// Perpendicular's dot product negated. They keep their own relations so the author's word for
    /// what they asked survives into diagnostics, and so the common cases carry no float at all.
    ///
    /// A sine repeats every half turn, so a stated angle and that angle plus 180 degrees are the
    /// same claim. That is not a rounding of the idea: a segment has two ends and no preferred
    /// one, so which way it points is not something the drawing knows. It is the same ambiguity
    /// `Parallel` already lives with, where 0 and 180 are both parallel.
    ///
    /// **Radians, not degrees.** Degrees are an authoring unit and stop at the adapter, the way
    /// voxels do; what crosses into the solver is what its trigonometry takes.
    Angle {
        first: AngleArm,
        second: AngleArm,
        radians: f64,
    },
    /// Two segments have the same length without asserting which length; that is different from
    /// two `Distance` dimensions that each carry one authored number.
    Equal { first: SegmentId, second: SegmentId },
    /// Midpoint is a place, not merely a line condition, and therefore contributes two rows.
    Midpoint { point: PointId, segment: SegmentId },
    /// Collinear contributes two distances to a datum line: parallel plus zero offset.
    Collinear { first: SegmentId, second: SegmentId },
    /// A point lies on the curve's SUPPORT: the infinite line a segment runs along, or the whole
    /// circle an arc is cut from. One row, because a support is one condition and the point keeps
    /// its freedom to slide along it.
    ///
    /// The support and not the finite piece, for the same reason [`Relation::Collinear`] uses the
    /// infinite line: a residual that had to report "off the end" would be discontinuous where the
    /// piece stops, and the optimizer would be walking a cliff. Whether the author meant a point
    /// past the end of an arc is an authoring question, answered before the relation is asserted.
    PointOnCurve { point: PointId, curve: SketchCurve },
    /// A point stands ON a spline. Two rows saying the point IS the curve at `station`, against
    /// the one solver-owned column that says where along it that is.
    ///
    /// The station is the whole reason this is not [`Relation::PointOnCurve`] with a wider curve.
    /// A line or a circle answers "how far off is this point" without being asked where along;
    /// a spline cannot, and the only other way to ask is to search it for the nearest place —
    /// which jumps as soon as two places tie, and hands the finite-difference Jacobian a column
    /// of noise. Naming the place as a coordinate spends one column to buy two smooth rows, and
    /// still removes exactly the one freedom a point-on-curve should.
    PointOnSpline {
        point: PointId,
        spline: SplineId,
        station: ParameterId,
    },
    /// Two curves touch at the persisted branch. The branch is fixed during a solve; it must never
    /// be inferred from a transient contact or switched by the optimizer.
    Tangent {
        first: SketchCurve,
        second: SketchCurve,
        branch: TangentBranch,
    },
    /// Two circular curves share a center while their radii remain independent.
    Concentric {
        first: SketchCurve,
        second: SketchCurve,
    },
    /// Two same-kind curves mirror across an explicit segment axis at a persisted correspondence.
    Symmetry {
        first: SketchCurve,
        second: SketchCurve,
        axis: SegmentId,
        branch: SymmetryBranch,
    },
    /// A spline's lever at `joint` runs along `against`. This is G1 across the joint, expressed
    /// over the lever's ARM rather than over a curve the solver holds: a spline is not one of its
    /// curves, but the arm steering it is one of its points, and the direction the arm gives is the
    /// direction the spline leaves at.
    TangentDirection {
        joint: PointId,
        joint_arm: PointId,
        against: SketchCurve,
    },
    /// A spline's curvature at `joint` matches `against`'s — G2 across the joint.
    ///
    /// The four points name the span the joint belongs to, which is all a spline's shape there
    /// depends on while every fit point carries an authored tangent. See [`JointSpan`] for why
    /// that locality holds and why the comparison is made between curvature ARROWS rather than
    /// signed scalars.
    ///
    /// TWO rows, direction first and then curvature: G2 is G1 plus curvature, and the second does
    /// not imply the first — two curves can agree about how hard they bend while leaving the joint
    /// in different directions, which is a cusp with a matching comb. They travel together as one
    /// relation because they are one authored claim, the way [`Relation::Collinear`] is parallel
    /// plus zero offset rather than two assertions the author has to remember to pair.
    Curvature {
        joint: PointId,
        joint_arm: PointId,
        neighbor: PointId,
        neighbor_arm: PointId,
        end: SpanEnd,
        against: SketchCurve,
    },
    /// Both coordinates of a point lie on the lattice `phase + n * pitch`. Quantize is the
    /// discrete outer tier: one numerical pass freezes the nearest lattice point from its input
    /// configuration, then the next exact pass may choose again from the preferred result.
    Quantize {
        point: PointId,
        pitch: f64,
        phase: f64,
    },
}

impl Relation {
    /// Keep this exhaustive. It is the residual stride consumed by the arithmetic loop below, so a
    /// wrong answer shifts every later row rather than merely making one relation wrong.
    fn residual_count(self) -> usize {
        match self {
            Self::Fix { .. }
            | Self::Quantize { .. }
            | Self::Coincident { .. }
            | Self::Midpoint { .. }
            | Self::Collinear { .. }
            | Self::Concentric { .. }
            // Two rows here are a direction and a curvature rather than an x and a y, and two
            // there are a point meeting a curve at a station it also solves for. A stride is a
            // stride and clippy will not let either distinction have its own arm.
            | Self::Curvature { .. }
            | Self::PointOnSpline { .. } => 2,
            Self::Horizontal { .. }
            | Self::Vertical { .. }
            | Self::Distance { .. }
            | Self::AxisDistance { .. }
            | Self::Radius { .. }
            | Self::RimGap { .. }
            | Self::Parallel { .. }
            | Self::Perpendicular { .. }
            | Self::Angle { .. }
            | Self::Equal { .. }
            | Self::Tangent { .. }
            | Self::TangentDirection { .. }
            | Self::PointLineDistance { .. }
            | Self::PointOnCurve { .. } => 1,
            Self::Symmetry { first, .. } => match first {
                SketchCurve::Segment(_) => 4,
                SketchCurve::Arc(_) => 5,
                SketchCurve::Circle(_) => 3,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildError {
    UnknownPoint,
    UnknownSegment,
    UnknownArc,
    UnknownSpline,
    UnknownParameter,
    InvalidParameter,
    InvalidTangent,
    InvalidConcentric,
    InvalidSymmetry,
    InvalidQuantization,
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

/// One arc: two ends and the point they turn about, all three placed (ADR 0038).
///
/// There is no stored sweep and no stored direction. The arc runs COUNTER-CLOCKWISE from `from` to
/// `to` about `center`, so the endpoint order carries the sense and the swept angle is read off the
/// three positions. The sixth coordinate three points spend on a five-freedom arc is taken back by
/// one equal-radius residual per arc.
#[derive(Debug, Clone, Copy)]
struct ArcCenter {
    key: ArcId,
    center: PointId,
    from: PointId,
    to: PointId,
    /// The arc's radius as a solver COLUMN, the way a circle has always had one.
    ///
    /// `None` only for an arc whose seed geometry is degenerate — an end sitting on the center —
    /// where there is no positive radius to name. That arc keeps the single equal-radius row.
    radius: Option<ParameterId>,
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
}

#[derive(Debug, Clone, Copy)]
struct Circle {
    key: CircleId,
    center: PointId,
    radius: ParameterId,
}

/// How a spline's points shape it.
#[derive(Debug, Clone, PartialEq)]
enum SplineForm {
    /// The curve passes THROUGH each point, steered where an arm stands beside one.
    FitPoint { arms: Vec<Option<PointId>> },
    /// The points stand off the curve and pull it.
    ControlPoint,
}

/// A spline as the solver holds it: the points it is made of, and nothing else.
///
/// Deliberately not a curve. Storing the fitted pieces would freeze a shape the fit points are
/// still free to move, and every residual would then read a curve one iteration out of date.
#[derive(Debug, Clone)]
struct SplineShape {
    key: SplineId,
    points: Vec<PointId>,
    form: SplineForm,
    closed: bool,
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
    parameters: Vec<Parameter>,
    circles: Vec<Circle>,
    splines: Vec<SplineShape>,
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
            parameters: Vec::new(),
            circles: Vec::new(),
            splines: Vec::new(),
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

    /// Construction geometry goes through here too. A scaffold once had its own door, so that its
    /// span could be withheld from the snap; no span is offered to the snap any more, by anything,
    /// and the two doors led to the same place in every other respect.
    pub fn add_segment(&mut self, from: PointId, to: PointId) -> SegmentId {
        let id = SegmentId {
            owner: self.owner,
            index: self.segments.len(),
        };
        self.segments.push(Segment { from, to });
        id
    }

    /// A spline the curve passes THROUGH, with an optional steering arm beside each point.
    ///
    /// `arms` is index-aligned with `points`; a `None` leaves that point's tangent to the fit.
    /// Extra or missing arms are padded to `points`, because a caller that has fewer arms than
    /// points means the rest are unsteered rather than that the spline is malformed.
    pub fn add_fit_point_spline(
        &mut self,
        points: Vec<PointId>,
        arms: Vec<Option<PointId>>,
        closed: bool,
    ) -> SplineId {
        let mut arms = arms;
        arms.resize(points.len(), None);
        self.push_spline(points, SplineForm::FitPoint { arms }, closed)
    }

    /// A spline its points stand OFF and pull, rather than lie on.
    pub fn add_control_point_spline(&mut self, points: Vec<PointId>) -> SplineId {
        self.push_spline(points, SplineForm::ControlPoint, false)
    }

    fn push_spline(&mut self, points: Vec<PointId>, form: SplineForm, closed: bool) -> SplineId {
        let key = SplineId {
            owner: self.owner,
            index: self.splines.len(),
        };
        self.splines.push(SplineShape {
            key,
            points,
            form,
            closed,
        });
        key
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

    /// Add the solver-owned station of a point standing on a spline.
    ///
    /// Internal in the way an arc's radius column is internal: seeded from the place the author's
    /// own pick already landed on, written freely by the solve, never persisted. `seed` is in the
    /// curve's [`station_length`] units, measured from its start.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::InvalidParameter`] when `seed` is not finite.
    pub fn add_free_spline_station(&mut self, seed: f64) -> Result<ParameterId, BuildError> {
        self.add_parameter(ParameterKind::SplineStation, seed, true)
    }

    /// Every point, segment and arc a constraint names is one this problem holds.
    fn check_constraint_handles(&self) -> Result<(), BuildError> {
        let known_point =
            |point: PointId| point.owner == self.owner && point.index < self.points.len();
        let known_segment =
            |segment: SegmentId| segment.owner == self.owner && segment.index < self.segments.len();
        for (_, relation) in &self.constraints {
            let points = match *relation {
                Relation::Fix { point, .. }
                | Relation::Midpoint { point, .. }
                | Relation::PointLineDistance { point, .. }
                | Relation::PointOnCurve { point, .. }
                | Relation::PointOnSpline { point, .. } => vec![point],
                Relation::Distance { from, to, .. } | Relation::AxisDistance { from, to, .. } => {
                    vec![from, to]
                }
                Relation::Coincident { first, second } => vec![first, second],
                Relation::TangentDirection {
                    joint, joint_arm, ..
                } => vec![joint, joint_arm],
                Relation::Curvature {
                    joint,
                    joint_arm,
                    neighbor,
                    neighbor_arm,
                    ..
                } => vec![joint, joint_arm, neighbor, neighbor_arm],
                _ => Vec::new(),
            };
            if points.into_iter().any(|point| !known_point(point)) {
                return Err(BuildError::UnknownPoint);
            }
            let segments = match *relation {
                Relation::Horizontal { segment }
                | Relation::Vertical { segment }
                | Relation::PointLineDistance { line: segment, .. }
                | Relation::Midpoint { segment, .. } => vec![segment],
                Relation::Parallel { first, second }
                | Relation::Perpendicular { first, second }
                | Relation::Equal { first, second }
                | Relation::Collinear { first, second } => vec![first, second],
                // Only the straight arms are segments. The arc arms are checked as arcs, below,
                // because an arc that has gone is a different absence from a segment that has.
                Relation::Angle { first, second, .. } => [first, second]
                    .into_iter()
                    .filter_map(AngleArm::segment)
                    .collect(),
                _ => Vec::new(),
            };
            if segments.into_iter().any(|segment| !known_segment(segment)) {
                return Err(BuildError::UnknownSegment);
            }
            self.check_angle_arms(*relation)?;
        }
        Ok(())
    }

    /// Every point a spline is made of, arms included, is a point this problem holds.
    fn check_spline_points(&self) -> Result<(), BuildError> {
        for spline in &self.splines {
            let arms = match &spline.form {
                SplineForm::FitPoint { arms } => arms.clone(),
                SplineForm::ControlPoint => Vec::new(),
            };
            if spline
                .points
                .iter()
                .copied()
                .chain(arms.into_iter().flatten())
                .any(|point| point.owner != self.owner || self.points.get(point.index).is_none())
            {
                return Err(BuildError::UnknownPoint);
            }
        }
        Ok(())
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
            ParameterKind::PositiveRadius => {
                stored.is_finite() && stored > 0.0 && ResolvedLength::try_from_f64(stored).is_ok()
            }
            // Every finite station is a place, negative ones included: a station short of zero is
            // off the near end of an open spline, which is somewhere the solve is allowed to look
            // and then be told it has gone too far. It is never written back, so it owes the
            // exact-rational store nothing.
            ParameterKind::SplineStation => stored.is_finite(),
        };
        let valid = source_valid
            && (!free
                || match kind {
                    ParameterKind::PositiveRadius => {
                        stored >= min_exact_positive() && stored <= max_exact_positive()
                    }
                    ParameterKind::SplineStation => true,
                });
        if !valid {
            return Err(BuildError::InvalidParameter);
        }
        let key = ParameterId {
            owner: self.owner,
            index: self.parameters.len(),
        };
        self.parameters.push(Parameter { kind, stored, free });
        Ok(key)
    }

    /// Add an arc running counter-clockwise from `from` to `to` about `center`.
    ///
    /// All three are ordinary points the caller placed (ADR 0038), and the arc names its RADIUS as
    /// a solver column beside them — seeded from those points, spent on two rows saying each end
    /// stands that far from the center.
    ///
    /// The column costs nothing in freedom. Six coordinates and one equal-radius row is five, and
    /// so is six coordinates, one column and two rows. What it buys is a name: the radius stops
    /// being a nonlinear function of four coordinates that the solve can only reach through them,
    /// and becomes a quantity a preference can hold, a dimension can pin, and a drag can pull. A
    /// circle has always had one; this is the arc stopping being the exception.
    ///
    /// It is SOLVER-INTERNAL. Nothing writes it back, because the document derives an arc's radius
    /// from its points on demand and never stores it beside them — which is what ADR 0038 means by
    /// a derived quantity, and why naming it here does not persist one. `planegcs` reaches the same
    /// arrangement from the other side, its `Arc` inheriting `rad` from its `Circle`.
    pub fn add_arc(&mut self, center: PointId, from: PointId, to: PointId) -> ArcId {
        let key = ArcId {
            owner: self.owner,
            index: self.arc_centers.len(),
        };
        let seed = self.arc_radius_seed(center, from, to);
        let radius = seed.and_then(|seed| self.add_free_positive_radius(seed).ok());
        self.arc_centers.push(ArcCenter {
            key,
            center,
            from,
            to,
            radius,
        });
        key
    }

    /// The radius to seed a new arc's column with: the mean of what its two ends currently say,
    /// so neither end is privileged when they disagree — and they do, mid-drag, which is exactly
    /// when the seed is read.
    ///
    /// `None` where either end is close enough to the center that no positive radius is meant.
    fn arc_radius_seed(&self, center: PointId, from: PointId, to: PointId) -> Option<f64> {
        let at = |id: PointId| {
            (id.owner == self.owner)
                .then(|| self.points.get(id.index).map(|point| point.at))
                .flatten()
        };
        let (center, from, to) = (at(center)?, at(from)?, at(to)?);
        let reach = |end: [f64; 2]| (end[0] - center[0]).hypot(end[1] - center[1]);
        let mean = f64::midpoint(reach(from), reach(to));
        (mean.is_finite() && mean > 0.0).then_some(mean)
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

    /// The arc half of an angle's liveness. An arc that has gone is a different absence from a
    /// segment that has, so it is answered separately and named separately.
    fn check_angle_arms(&self, relation: Relation) -> Result<(), BuildError> {
        let Relation::Angle { first, second, .. } = relation else {
            return Ok(());
        };
        let live = |arm: AngleArm| match arm {
            AngleArm::Segment(_) => true,
            AngleArm::ArcEnd { arc, .. } => self
                .arc_centers
                .get(arc.index)
                .is_some_and(|held| arc.owner == self.owner && held.key == arc),
        };
        if live(first) && live(second) {
            Ok(())
        } else {
            Err(BuildError::UnknownArc)
        }
    }

    /// # Errors
    ///
    /// Returns an error when a curve or relation references a foreign or unknown local handle.
    pub fn finish(self) -> Result<Problem, BuildError> {
        let known_point =
            |point: PointId| point.owner == self.owner && point.index < self.points.len();
        let known_parameter = |parameter: ParameterId| {
            parameter.owner == self.owner && parameter.index < self.parameters.len()
        };
        for segment in &self.segments {
            if !known_point(segment.from) || !known_point(segment.to) {
                return Err(BuildError::UnknownPoint);
            }
        }
        for arc in &self.arc_centers {
            if !known_point(arc.center) || !known_point(arc.from) || !known_point(arc.to) {
                return Err(BuildError::UnknownPoint);
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
        self.check_spline_points()?;
        self.check_constraint_handles()?;
        let raw = Problem {
            owner: self.owner,
            points: self.points,
            segments: self.segments,
            arc_centers: self.arc_centers,
            parameters: self.parameters,
            circles: self.circles,
            splines: self.splines,
            constraints: Vec::new(),
            snap_reach: SnapReach::UNBOUNDED,
            furthest_the_hand_has_reached: 0.0,
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
    parameters: Vec<Parameter>,
    circles: Vec<Circle>,
    splines: Vec<SplineShape>,
    constraints: Vec<ConstraintEntry>,
    /// How far a snap may carry a hand off the cursor. Not a property of the drawing — it is the
    /// gesture's, and it lives here because the drag is what reads it. A problem walked frame by
    /// frame is cloned from this one, so the ceiling travels with the walk.
    snap_reach: SnapReach,
    /// The furthest the lead hand has stood from where it pressed, over the whole gesture so far.
    /// Zero for a caller that keeps no path, which reads as the frame's own displacement — see
    /// [`Problem::the_hand_having_reached`].
    furthest_the_hand_has_reached: f64,
}

impl Problem {
    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn relation_count(&self) -> usize {
        self.constraints.len()
    }

    /// Derive and validate the contact for one Tangent relation against a solved drawing. The
    /// result is intentionally local and transient; document code maps its stable ids separately.
    ///
    /// # Errors
    ///
    /// Returns a contact error when the relation is not a Tangent or its current geometry is not
    /// one coincident finite-domain contact.
    pub fn tangent_contact(
        &self,
        relation: Relation,
        solution: &Solution,
    ) -> Result<TangentContact, TangentContactError> {
        let Resolved::Tangent {
            first,
            second,
            branch,
        } = self
            .resolve(relation)
            .map_err(|_| TangentContactError::InvalidBranch)?
        else {
            return Err(TangentContactError::InvalidBranch);
        };
        self.contact_for(first, second, branch, solution)
    }

    fn contact_for(
        &self,
        first: ResolvedCurve,
        second: ResolvedCurve,
        branch: TangentBranch,
        solution: &Solution,
    ) -> Result<TangentContact, TangentContactError> {
        if solution.owner != self.owner {
            return Err(TangentContactError::NonFinite);
        }
        if solution.positions.len() != self.points.len() {
            return Err(TangentContactError::NonFinite);
        }
        let scalars =
            scalar_coordinates_of_solution(self, solution).ok_or(TangentContactError::NonFinite)?;
        let mut whole = Vec::with_capacity(solution.positions.len() * 2 + scalars.len());
        for position in &solution.positions {
            whole.extend(*position);
        }
        whole.extend(scalars);
        let at = |slot| solution.positions[slot];
        let first_geometry = curve_geometry(
            first,
            &at,
            &self.parameters,
            Coordinates::of(&whole),
            self.points.len(),
        );
        let second_geometry = curve_geometry(
            second,
            &at,
            &self.parameters,
            Coordinates::of(&whole),
            self.points.len(),
        );
        tangent_contact(first_geometry, second_geometry, branch)
    }

    /// Derive the validated presentation locus for one standing Symmetry relation.
    ///
    /// # Errors
    ///
    /// Returns an error when the relation, solution, branch, or resolved geometry is invalid, or
    /// when the relation is not satisfied within the shared residual tolerance.
    pub fn symmetry_witness(
        &self,
        relation: Relation,
        solution: &Solution,
    ) -> Result<SymmetryWitness, SymmetryError> {
        let Resolved::Symmetry {
            first,
            second,
            axis,
            branch,
        } = self
            .resolve(relation)
            .map_err(|_| SymmetryError::InvalidBranch)?
        else {
            return Err(SymmetryError::InvalidBranch);
        };
        if solution.owner != self.owner {
            return Err(SymmetryError::NonFinite);
        }
        if solution.positions.len() != self.points.len() {
            return Err(SymmetryError::NonFinite);
        }
        let scalars =
            scalar_coordinates_of_solution(self, solution).ok_or(SymmetryError::NonFinite)?;
        let mut whole = Vec::with_capacity(solution.positions.len() * 2 + scalars.len());
        for position in &solution.positions {
            whole.extend(*position);
        }
        whole.extend(scalars);
        let at = |slot| solution.positions[slot];
        let geometry = |curve| {
            curve_geometry(
                curve,
                &at,
                &self.parameters,
                Coordinates::of(&whole),
                self.points.len(),
            )
        };
        symmetry_witness(
            geometry(first),
            geometry(second),
            geometry(ResolvedCurve::Segment(axis)),
            branch,
        )
    }

    /// The first precise finite-contact failure among the standing Tangents, in constraint order.
    /// Document adapters use this before writeback so ordinary settle and drag remain atomic.
    pub fn first_tangent_contact_failure(
        &self,
        solution: &Solution,
    ) -> Option<TangentContactFailure> {
        self.constraints
            .iter()
            .find_map(|constraint| match constraint.resolved {
                Resolved::Tangent {
                    first,
                    second,
                    branch,
                } => self
                    .contact_for(first, second, branch, solution)
                    .err()
                    .map(|error| TangentContactFailure {
                        constraint: constraint.key,
                        error,
                    }),
                _ => None,
            })
    }

    /// Check the authored configuration without invoking the numerical solver. This is for edit
    /// operations whose contract says that they may change one intrinsic value, but may not let
    /// the solver repair that edit by moving other geometry.
    pub fn validate_current(&self) -> CurrentValidation {
        let positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let scalar_coordinates = self.scalar_coordinates();
        let solution = self.solution(positions.clone(), &scalar_coordinates);
        let tangent_failure = self.first_tangent_contact_failure(&solution);
        let satisfied = Residuals::new(self, &scalar_coordinates, Rigidity::Ignored).map_or(
            self.constraints.is_empty(),
            |system| {
                let parameters = system.guess(&positions);
                let mut residuals = vec![0.0; system.residual_count()];
                system.residuals(&parameters, &mut residuals);
                residuals.iter().all(|residual| residual.is_finite())
                    && residuals
                        .iter()
                        .map(|residual| residual * residual)
                        .sum::<f64>()
                        .sqrt()
                        <= SATISFIED_RESIDUAL
            },
        );
        CurrentValidation {
            satisfied,
            collapsed: self.collapsed_by(&solution),
            tangent_failure,
        }
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
                .map(|(parameter, coordinate)| {
                    let settled =
                        if coordinate.to_bits() == parameter_coordinate(parameter).to_bits() {
                            parameter.stored
                        } else {
                            physical_parameter_value(parameter, *coordinate)
                        };
                    match parameter.kind {
                        ParameterKind::PositiveRadius => ParameterValue::Radius(settled),
                        ParameterKind::SplineStation => ParameterValue::Station(settled),
                    }
                })
                .collect(),
        }
    }

    #[allow(clippy::too_many_lines)]
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
        let curve = |curve: SketchCurve| match curve {
            SketchCurve::Segment(id) => segment(id).map(ResolvedCurve::Segment),
            SketchCurve::Arc(id) => self
                .arc_centers
                .get(id.index)
                .filter(|arc| id.owner == self.owner && arc.key == id)
                .map(|arc| {
                    ResolvedCurve::Arc(ArcCurveSlots {
                        center: arc.center.index,
                        from: arc.from.index,
                        to: arc.to.index,
                    })
                })
                .ok_or(BuildError::InvalidTangent),
            SketchCurve::Circle(id) => self
                .circles
                .get(id.index)
                .filter(|circle| id.owner == self.owner && circle.key == id)
                .map(|circle| {
                    ResolvedCurve::Circle(CircleSlots {
                        center: circle.center.index,
                        radius_parameter: circle.radius.index,
                    })
                })
                .ok_or(BuildError::InvalidTangent),
        };
        match relation {
            Relation::Fix { point: id, at } => Ok(Resolved::Fix {
                slot: point(id)?,
                at,
            }),
            Relation::Quantize {
                point: id,
                pitch,
                phase,
            } => {
                if !pitch.is_finite() || pitch <= 0.0 || !phase.is_finite() {
                    return Err(BuildError::InvalidQuantization);
                }
                Ok(Resolved::Quantize {
                    slot: point(id)?,
                    pitch,
                    phase,
                })
            }
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
            Relation::AxisDistance {
                from,
                to,
                axis,
                length,
            } => Ok(Resolved::AxisDistance {
                from: point(from)?,
                to: point(to)?,
                // A plane has two coordinates and nothing names a third, so anything else is a
                // caller bug rather than a drawing the solver should try to honour.
                axis: axis.min(1),
                length,
            }),
            Relation::Coincident { first, second } => Ok(Resolved::Coincident {
                first: point(first)?,
                second: point(second)?,
            }),
            Relation::PointLineDistance {
                point: stood,
                line,
                distance,
            } => Ok(Resolved::PointLineDistance {
                point: point(stood)?,
                line: segment(line)?,
                distance,
            }),
            Relation::Parallel { first, second } => Ok(Resolved::Parallel {
                first: segment(first)?,
                second: segment(second)?,
            }),
            Relation::Angle {
                first,
                second,
                radians,
            } => {
                let arm = |arm: AngleArm| match arm {
                    AngleArm::Segment(id) => segment(id).map(ResolvedAngleArm::Segment),
                    AngleArm::ArcEnd { arc, end } => match curve(SketchCurve::Arc(arc))? {
                        ResolvedCurve::Arc(slots) => Ok(ResolvedAngleArm::ArcEnd {
                            center: slots.center,
                            end: match end {
                                SpanEnd::Start => slots.from,
                                SpanEnd::Finish => slots.to,
                            },
                        }),
                        // Unreachable: the arm's own type says arc, and `curve` answers by kind.
                        ResolvedCurve::Segment(_) | ResolvedCurve::Circle(_) => {
                            Err(BuildError::UnknownArc)
                        }
                    },
                };
                Ok(Resolved::Angle {
                    first: arm(first)?,
                    second: arm(second)?,
                    radians,
                })
            }
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
            Relation::PointOnCurve {
                point: id,
                curve: subject,
            } => Ok(Resolved::PointOnCurve {
                point: point(id)?,
                curve: curve(subject)?,
            }),
            Relation::PointOnSpline {
                point: id,
                spline: shape,
                station,
            } => {
                let held = self
                    .splines
                    .get(shape.index)
                    .filter(|held| shape.owner == self.owner && held.key == shape)
                    .ok_or(BuildError::UnknownSpline)?;
                let known_station = self.parameters.get(station.index).is_some_and(|parameter| {
                    station.owner == self.owner && parameter.kind == ParameterKind::SplineStation
                });
                if !known_station {
                    return Err(BuildError::UnknownParameter);
                }
                Ok(Resolved::PointOnSpline {
                    point: point(id)?,
                    spline: shape.index,
                    station: station.index,
                    per_unit: live_spline(held, &|slot: usize| self.points[slot].at)
                        .as_ref()
                        .map_or(1.0, station_length),
                })
            }
            Relation::Radius {
                curve: subject,
                length,
            } => Ok(Resolved::Radius {
                curve: curve(subject)?,
                length,
            }),
            Relation::RimGap {
                first,
                second,
                distance,
            } => Ok(Resolved::RimGap {
                first: curve(first)?,
                second: curve(second)?,
                distance,
            }),
            Relation::TangentDirection {
                joint,
                joint_arm,
                against,
            } => Ok(Resolved::TangentDirection {
                joint: point(joint)?,
                joint_arm: point(joint_arm)?,
                against: curve(against)?,
            }),
            Relation::Curvature {
                joint,
                joint_arm,
                neighbor,
                neighbor_arm,
                end,
                against,
            } => Ok(Resolved::Curvature {
                joint: point(joint)?,
                joint_arm: point(joint_arm)?,
                neighbor: point(neighbor)?,
                neighbor_arm: point(neighbor_arm)?,
                end,
                against: curve(against)?,
            }),
            Relation::Tangent {
                first,
                second,
                branch,
            } => {
                let (first, second) = (curve(first)?, curve(second)?);
                if !tangent_branch_matches_types(first, second, branch) {
                    return Err(BuildError::InvalidTangent);
                }
                Ok(Resolved::Tangent {
                    first,
                    second,
                    branch,
                })
            }
            Relation::Concentric { first, second } => {
                let center =
                    |subject| match curve(subject).map_err(|_| BuildError::InvalidConcentric)? {
                        ResolvedCurve::Arc(arc) => Ok(arc.center),
                        ResolvedCurve::Circle(circle) => Ok(circle.center),
                        ResolvedCurve::Segment(_) => Err(BuildError::InvalidConcentric),
                    };
                if first == second {
                    return Err(BuildError::InvalidConcentric);
                }
                Ok(Resolved::Concentric {
                    first: center(first)?,
                    second: center(second)?,
                })
            }
            Relation::Symmetry {
                first,
                second,
                axis,
                branch,
            } => {
                if first == second
                    || matches!(first, SketchCurve::Segment(subject) if subject == axis)
                    || matches!(second, SketchCurve::Segment(subject) if subject == axis)
                {
                    return Err(BuildError::InvalidSymmetry);
                }
                let first = curve(first).map_err(|_| BuildError::InvalidSymmetry)?;
                let second = curve(second).map_err(|_| BuildError::InvalidSymmetry)?;
                let axis = segment(axis).map_err(|_| BuildError::InvalidSymmetry)?;
                let matches = matches!(
                    (first, second, branch),
                    (
                        ResolvedCurve::Segment(_),
                        ResolvedCurve::Segment(_),
                        SymmetryBranch::Direct | SymmetryBranch::Reversed
                    ) | (
                        ResolvedCurve::Arc(_),
                        ResolvedCurve::Arc(_),
                        SymmetryBranch::Direct | SymmetryBranch::Reversed
                    ) | (
                        ResolvedCurve::Circle(_),
                        ResolvedCurve::Circle(_),
                        SymmetryBranch::Centers
                    )
                );
                let from = self.points[axis.from].at;
                let to = self.points[axis.to].at;
                if !matches || (to[0] - from[0]).hypot(to[1] - from[1]) <= COLLAPSED_SPAN {
                    return Err(BuildError::InvalidSymmetry);
                }
                Ok(Resolved::Symmetry {
                    first,
                    second,
                    axis,
                    branch,
                })
            }
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
    use crate::sketch::{InternalContainment, LineSide};

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

    /// A three-point fit spline, a point standing beside it, and the relation that holds it on.
    ///
    /// The station is seeded at the middle fit point, which is a starting guess in the way the
    /// pick that authored the relation would have been. The solve is free to slide it.
    fn point_beside_a_spline() -> (ProblemBuilder, Relation, PointId, Vec<PointId>) {
        let places = [[0.0, 0.0], [10.0, 6.0], [20.0, 0.0]];
        let mut builder = ProblemBuilder::new();
        let through: Vec<PointId> = places.iter().map(|at| builder.add_point(*at)).collect();
        let spline = builder.add_fit_point_spline(through.clone(), Vec::new(), false);
        let beside = builder.add_point([10.0, 2.0]);
        let seed = station_length(&fit_point_spline(&places, &[None, None, None], false).unwrap());
        let station = builder.add_free_spline_station(seed).unwrap();
        let relation = Relation::PointOnSpline {
            point: beside,
            spline,
            station,
        };
        (builder, relation, beside, through)
    }

    /// How far `witness` stands off the spline drawn through `places`, by dense sampling.
    fn off_the_spline(places: &[[f64; 2]], witness: [f64; 2]) -> f64 {
        let tangents = vec![None; places.len()];
        let candidate = fit_point_spline(places, &tangents, false).unwrap();
        candidate
            .pieces
            .iter()
            .flat_map(|piece| {
                (0_u16..=400).map(move |step| {
                    let on = piece.point_at(f64::from(step) / 400.0);
                    (on[0] - witness[0]).hypot(on[1] - witness[1])
                })
            })
            .fold(f64::INFINITY, f64::min)
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
        accepts(
            &problem,
            Relation::Quantize {
                point: b,
                pitch: 1.0,
                phase: 0.0,
            },
        );
        // A spline needs a problem that holds one, and the fixture above holds none. A relation
        // that builds but never writes a row is exactly what this test exists to catch, so it
        // gets its own drawing rather than being left out.
        let (builder, on_a_spline, _, _) = point_beside_a_spline();
        accepts(&builder.finish().unwrap(), on_a_spline);
    }

    /// Every handle a relation could want, drawn once, so each relation below is one line.
    ///
    /// Named rather than positional: an array of points with per-index meaning is exactly the
    /// shape a later edit renumbers silently.
    struct Cast {
        corner: PointId,
        across: PointId,
        up: PointId,
        far: PointId,
        loose: PointId,
        arc_start: PointId,
        arc_end: PointId,
        other_end: PointId,
        base: SegmentId,
        lid: SegmentId,
        riser: SegmentId,
        arc: ArcId,
        other_arc: ArcId,
        round: CircleId,
        spline: SplineId,
        joint: PointId,
        joint_arm: PointId,
        neighbor: PointId,
        neighbor_arm: PointId,
        station: ParameterId,
    }

    /// The drawing every relation in the honesty check is asserted against.
    ///
    /// Deliberately generous and deliberately non-degenerate: a collapsed segment or an end
    /// sitting on its own centre sends several residuals down a guard branch that reads nothing,
    /// and a reads-set is only checked by rows that actually run.
    fn cast() -> (ProblemBuilder, Cast) {
        let mut builder = ProblemBuilder::new();
        let corner = builder.add_point([0.0, 0.0]);
        let across = builder.add_point([10.0, 1.0]);
        let up = builder.add_point([1.0, 10.0]);
        let far = builder.add_point([11.0, 12.0]);
        let loose = builder.add_point([4.0, 3.0]);
        let hub = builder.add_point([30.0, 0.0]);
        let arc_start = builder.add_point([36.0, 1.0]);
        let arc_end = builder.add_point([30.5, 6.0]);
        let other_hub = builder.add_point([60.0, 0.0]);
        let other_start = builder.add_point([64.0, 1.0]);
        let other_end = builder.add_point([60.5, 4.0]);
        let round_hub = builder.add_point([90.0, 0.0]);
        let joint = builder.add_point([100.0, 0.0]);
        let joint_arm = builder.add_point([103.0, 1.0]);
        let neighbor = builder.add_point([110.0, 4.0]);
        let neighbor_arm = builder.add_point([107.0, 3.0]);
        let through: Vec<PointId> = [[0.0, 30.0], [10.0, 36.0], [20.0, 30.0]]
            .into_iter()
            .map(|at| builder.add_point(at))
            .collect();
        let base = builder.add_segment(corner, across);
        let lid = builder.add_segment(up, far);
        let riser = builder.add_segment(corner, up);
        let arc = builder.add_arc(hub, arc_start, arc_end);
        let other_arc = builder.add_arc(other_hub, other_start, other_end);
        let radius = builder.add_free_positive_radius(7.0).unwrap();
        let round = builder.add_circle(round_hub, radius);
        let spline = builder.add_fit_point_spline(through, Vec::new(), false);
        let station = builder.add_free_spline_station(1.25).unwrap();
        let held = Cast {
            corner,
            across,
            up,
            far,
            loose,
            arc_start,
            arc_end,
            other_end,
            base,
            lid,
            riser,
            arc,
            other_arc,
            round,
            spline,
            joint,
            joint_arm,
            neighbor,
            neighbor_arm,
            station,
        };
        (builder, held)
    }

    /// One relation of every kind, over the cast above.
    ///
    /// Paired with [`relation_kind`], whose match is exhaustive: adding a `Relation` variant fails
    /// to COMPILE there, and the assertion at the foot of the honesty check refuses to pass until
    /// the new variant is named in `EVERY_RELATION` and appears here. That chain is the only thing
    /// standing between a new relation and a Curtis-Powell-Reid group that perturbs two parameters
    /// one of its rows can see.
    #[allow(clippy::too_many_lines)]
    fn one_of_every_relation(held: &Cast) -> Vec<Relation> {
        vec![
            Relation::Fix {
                point: held.corner,
                at: [0.0, 0.0],
            },
            Relation::Horizontal { segment: held.base },
            Relation::Vertical {
                segment: held.riser,
            },
            Relation::Distance {
                from: held.corner,
                to: held.far,
                length: 12.0,
            },
            Relation::AxisDistance {
                from: held.corner,
                to: held.across,
                axis: 0,
                length: 9.0,
            },
            Relation::PointLineDistance {
                point: held.loose,
                line: held.base,
                distance: 2.0,
            },
            Relation::Radius {
                curve: SketchCurve::Arc(held.arc),
                length: 6.0,
            },
            Relation::RimGap {
                first: SketchCurve::Arc(held.arc),
                second: SketchCurve::Circle(held.round),
                distance: 1.0,
            },
            Relation::Coincident {
                first: held.loose,
                second: held.up,
            },
            Relation::Parallel {
                first: held.base,
                second: held.lid,
            },
            Relation::Perpendicular {
                first: held.base,
                second: held.riser,
            },
            Relation::Angle {
                first: AngleArm::Segment(held.base),
                second: AngleArm::ArcEnd {
                    arc: held.arc,
                    end: SpanEnd::Start,
                },
                radians: 0.7,
            },
            Relation::Equal {
                first: held.base,
                second: held.lid,
            },
            Relation::Midpoint {
                point: held.loose,
                segment: held.riser,
            },
            Relation::Collinear {
                first: held.base,
                second: held.lid,
            },
            Relation::PointOnCurve {
                point: held.loose,
                curve: SketchCurve::Circle(held.round),
            },
            Relation::PointOnSpline {
                point: held.loose,
                spline: held.spline,
                station: held.station,
            },
            Relation::Tangent {
                first: SketchCurve::Arc(held.arc),
                second: SketchCurve::Circle(held.round),
                branch: TangentBranch::External,
            },
            Relation::Concentric {
                first: SketchCurve::Arc(held.arc),
                second: SketchCurve::Arc(held.other_arc),
            },
            Relation::Symmetry {
                first: SketchCurve::Arc(held.arc),
                second: SketchCurve::Arc(held.other_arc),
                axis: held.riser,
                branch: SymmetryBranch::Direct,
            },
            Relation::TangentDirection {
                joint: held.joint,
                joint_arm: held.joint_arm,
                against: SketchCurve::Circle(held.round),
            },
            Relation::Curvature {
                joint: held.joint,
                joint_arm: held.joint_arm,
                neighbor: held.neighbor,
                neighbor_arm: held.neighbor_arm,
                end: SpanEnd::Start,
                against: SketchCurve::Circle(held.round),
            },
            Relation::Quantize {
                point: held.across,
                pitch: 1.0,
                phase: 0.0,
            },
        ]
    }

    /// What kind of relation this is. **EXHAUSTIVE, and that is its entire job.**
    ///
    /// A new `Relation` variant does not compile until it is named here, and once it is named the
    /// honesty check will not pass until it also appears in `EVERY_RELATION` and in
    /// [`one_of_every_relation`]. The alternative is a relation whose rows read a parameter they
    /// never declared, which does not throw and does not fail a tolerance — it converges to a
    /// slightly different drawing.
    fn relation_kind(relation: Relation) -> &'static str {
        match relation {
            Relation::Fix { .. } => "Fix",
            Relation::Horizontal { .. } => "Horizontal",
            Relation::Vertical { .. } => "Vertical",
            Relation::Distance { .. } => "Distance",
            Relation::AxisDistance { .. } => "AxisDistance",
            Relation::PointLineDistance { .. } => "PointLineDistance",
            Relation::Radius { .. } => "Radius",
            Relation::RimGap { .. } => "RimGap",
            Relation::Coincident { .. } => "Coincident",
            Relation::Parallel { .. } => "Parallel",
            Relation::Perpendicular { .. } => "Perpendicular",
            Relation::Angle { .. } => "Angle",
            Relation::Equal { .. } => "Equal",
            Relation::Midpoint { .. } => "Midpoint",
            Relation::Collinear { .. } => "Collinear",
            Relation::PointOnCurve { .. } => "PointOnCurve",
            Relation::PointOnSpline { .. } => "PointOnSpline",
            Relation::Tangent { .. } => "Tangent",
            Relation::Concentric { .. } => "Concentric",
            Relation::Symmetry { .. } => "Symmetry",
            Relation::TangentDirection { .. } => "TangentDirection",
            Relation::Curvature { .. } => "Curvature",
            Relation::Quantize { .. } => "Quantize",
        }
    }

    /// Every relation there is. Bump this when `relation_kind` stops compiling.
    const EVERY_RELATION: [&str; 23] = [
        "Fix",
        "Horizontal",
        "Vertical",
        "Distance",
        "AxisDistance",
        "PointLineDistance",
        "Radius",
        "RimGap",
        "Coincident",
        "Parallel",
        "Perpendicular",
        "Angle",
        "Equal",
        "Midpoint",
        "Collinear",
        "PointOnCurve",
        "PointOnSpline",
        "Tangent",
        "Concentric",
        "Symmetry",
        "TangentDirection",
        "Curvature",
        "Quantize",
    ];

    /// **The falsifier for the grouped Jacobian: no row moves under a parameter it did not
    /// declare, bit for bit.**
    ///
    /// The Curtis-Powell-Reid grouping differences several parameters in one residual pass, which
    /// is exact only while no row reads two of them. Nothing at solve time can notice a row that
    /// reads a parameter its reads-set left out — the Jacobian simply comes back with two
    /// derivatives added into one entry, the search still converges, and it converges somewhere
    /// slightly else. So the claim is checked HERE, against every relation, at both rigidity
    /// modes, at several perturbation sizes, and by comparing BITS rather than a tolerance: a row
    /// that did not declare a parameter must be untouched by it, not merely near where it was.
    ///
    /// Several sizes because one is not enough. A six-millionths step is what the Jacobian uses,
    /// but a dependency that only shows through a branch — a length crossing an epsilon guard, an
    /// arc's sweep crossing its own tail — is invisible at that step and real at a larger one.
    #[test]
    fn every_relation_declares_what_its_rows_read() {
        let mut checked: Vec<&'static str> = Vec::new();
        for index in 0..one_of_every_relation(&cast().1).len() {
            // The cast is rebuilt for each relation because handles are owner-tagged: a relation
            // holding one drawing's points cannot be asserted against another's.
            let (mut builder, held) = cast();
            let relation = one_of_every_relation(&held)[index];
            let kind = relation_kind(relation);
            builder.add_constraint(relation);
            let problem = builder.finish().unwrap();
            let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
            let scalars = problem.scalar_coordinates();
            for rigidity in [
                Rigidity::Ignored,
                Rigidity::Preferred {
                    anchored: &[],
                    flexible_curves: &[],
                    was: &[],
                    opening: &[],
                    reshaping: true,
                },
            ] {
                let system = Residuals::new(&problem, &scalars, rigidity).unwrap();
                let reads = system.parameter_reads().unwrap();
                assert_eq!(
                    reads.row_count(),
                    system.residual_count(),
                    "{kind}: the reads-set is a different shape from the residual vector, which \
                     pins every later row's claim to the wrong residual"
                );
                let guess = system.guess(&positions);
                for step in [6.0e-6, 1.0e-3, 0.25] {
                    assert_eq!(
                        first_undeclared_read(&system, &guess, step),
                        None,
                        "{kind}: a row moved under a parameter it did not declare, at step {step}"
                    );
                }
            }
            checked.push(kind);
        }
        checked.sort_unstable();
        let mut every = EVERY_RELATION.to_vec();
        every.sort_unstable();
        assert_eq!(
            checked, every,
            "every relation must be exercised, and only relations that exist"
        );
    }

    /// **The falsifier for the narrowed pass: a row answers the same alone as it does in company,
    /// bit for bit.**
    ///
    /// A grouped Jacobian reads only the rows a column's group declared, so it now asks for only
    /// those rows. If a row computed on its own rounded differently from the same row computed in
    /// a whole pass, the grouped Jacobian would stop matching the column-by-column one and the
    /// exactness that whole path rests on would be gone — quietly, because both answers would still
    /// be near enough to converge. Checked over every relation, at both rigidity modes, group by
    /// group and then row by row.
    #[test]
    fn every_relations_rows_answer_the_same_alone_as_in_company() {
        for index in 0..one_of_every_relation(&cast().1).len() {
            let (mut builder, held) = cast();
            let relation = one_of_every_relation(&held)[index];
            let kind = relation_kind(relation);
            builder.add_constraint(relation);
            let problem = builder.finish().unwrap();
            let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
            let scalars = problem.scalar_coordinates();
            for rigidity in [
                Rigidity::Ignored,
                Rigidity::Preferred {
                    anchored: &[],
                    flexible_curves: &[],
                    was: &[],
                    opening: &[],
                    reshaping: true,
                },
            ] {
                let system = Residuals::new(&problem, &scalars, rigidity).unwrap();
                let guess = system.guess(&positions);
                assert_eq!(
                    first_subset_disagreement(&system, &guess),
                    None,
                    "{kind}: a row came out differently when it was asked for on its own"
                );
            }
        }
    }

    /// The row table is the only description of the row order, so this is what says it describes
    /// the vector the arithmetic actually writes.
    ///
    /// Three claims. It is as long as the residual vector, or every row past the first mismatch is
    /// attributed to the wrong arm. Each arm's rows are contiguous and start where the arm says,
    /// which is what lets a walk skip an arm it has already written by comparing one number. And
    /// every row is written by some arm: the vector is poisoned to NaN first, so a row no arm
    /// covers is a NaN rather than a stale zero that would read as a satisfied relation.
    #[test]
    fn the_row_table_names_every_row_exactly_once() {
        for index in 0..one_of_every_relation(&cast().1).len() {
            let (mut builder, held) = cast();
            let relation = one_of_every_relation(&held)[index];
            let kind = relation_kind(relation);
            builder.add_constraint(relation);
            let problem = builder.finish().unwrap();
            let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
            let scalars = problem.scalar_coordinates();
            for rigidity in [
                Rigidity::Ignored,
                Rigidity::Preferred {
                    anchored: &[],
                    flexible_curves: &[],
                    was: &[],
                    opening: &[],
                    reshaping: true,
                },
            ] {
                let system = Residuals::new(&problem, &scalars, rigidity).unwrap();
                assert_eq!(
                    system.rows.len(),
                    system.residual_count(),
                    "{kind}: the table is a different length from the residual vector"
                );
                for (row, source) in system.rows.iter().enumerate() {
                    let opens = source.start == row;
                    let carries = row
                        .checked_sub(1)
                        .and_then(|before| system.rows.get(before))
                        .is_some_and(|before| before.start == source.start);
                    assert!(
                        opens || carries,
                        "{kind}: row {row} claims to start at {} without following its own arm",
                        source.start
                    );
                }
                let guess = system.guess(&positions);
                let mut written = vec![f64::NAN; system.residual_count()];
                system.residuals(&guess, &mut written);
                assert!(
                    written.iter().all(|value| !value.is_nan()),
                    "{kind}: a row the table names is written by no arm"
                );
            }
        }
    }

    /// **The falsifier for reading the coordinates instead of building them: every slot answers
    /// what the built vector would have held, bit for bit.**
    ///
    /// The map is the only thing that knows a slot's current value is in the parameter vector
    /// rather than still in `base`. A slot it leaves out reads STALE — the value from before the
    /// pass began — and stale is not a crash, it is a row quietly measuring the wrong drawing while
    /// the search converges on it. The named trap is the scalar range: those rows read `whole`
    /// directly rather than through a point, so a map built only over point coordinates leaves
    /// every radius reading its opening value. Checked at a perturbed parameter vector, because at
    /// the guess the two agree by accident.
    #[test]
    fn every_slot_reads_what_the_built_vector_would_have_held() {
        for index in 0..one_of_every_relation(&cast().1).len() {
            let (mut builder, held) = cast();
            let relation = one_of_every_relation(&held)[index];
            let kind = relation_kind(relation);
            builder.add_constraint(relation);
            let problem = builder.finish().unwrap();
            let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
            let scalars = problem.scalar_coordinates();
            for rigidity in [
                Rigidity::Ignored,
                Rigidity::Preferred {
                    anchored: &[],
                    flexible_curves: &[],
                    was: &[],
                    opening: &[],
                    reshaping: true,
                },
            ] {
                let system = Residuals::new(&problem, &scalars, rigidity).unwrap();
                // Every column moved by a different amount, so a slot reading the wrong column is
                // a disagreement and not a coincidence.
                let mut nudge = 0.125;
                let moved: Vec<f64> = system
                    .guess(&positions)
                    .iter()
                    .map(|at| {
                        nudge += 0.125;
                        at + nudge
                    })
                    .collect();
                let built = system.widen(&moved);
                let read = system.coordinates(&moved);
                assert_eq!(
                    built.len(),
                    system.base.len(),
                    "{kind}: the map covers a shorter vector than the drawing has slots"
                );
                for (slot, stood) in built.iter().enumerate() {
                    assert_eq!(
                        stood.to_bits(),
                        read.get(slot).to_bits(),
                        "{kind}: slot {slot} reads {} where the built vector holds {stood}",
                        read.get(slot)
                    );
                }
            }
        }
    }

    /// The same honesty check over the drawings a DRAG builds, which are the ones the grouping is
    /// there for: an arc slot carries rigidity spans, scalar holds and point stays, and those rows
    /// outnumber the relations three to one.
    #[test]
    fn a_slots_rows_declare_what_they_read_under_every_rigidity() {
        let (mut builder, held) = cast();
        builder.add_constraint(Relation::Concentric {
            first: SketchCurve::Arc(held.arc),
            second: SketchCurve::Arc(held.other_arc),
        });
        builder.add_constraint(Relation::Horizontal { segment: held.base });
        builder.add_constraint(Relation::Coincident {
            first: held.arc_end,
            second: held.other_end,
        });
        let problem = builder.finish().unwrap();
        let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
        let scalars = problem.scalar_coordinates();
        let hands = [(held.arc_start, [37.0, 2.0])];
        for rigidity in [
            Rigidity::Ignored,
            Rigidity::Preferred {
                anchored: &[],
                flexible_curves: &[],
                was: &[],
                opening: &[],
                reshaping: false,
            },
            Rigidity::Preferred {
                anchored: &[],
                flexible_curves: &[SketchCurve::Arc(held.arc)],
                was: &hands,
                opening: &[],
                reshaping: true,
            },
        ] {
            let system = Residuals::new(&problem, &scalars, rigidity).unwrap();
            assert_eq!(
                system.parameter_reads().unwrap().row_count(),
                system.residual_count()
            );
            let guess = system.guess(&positions);
            for step in [6.0e-6, 1.0e-3, 0.25] {
                assert_eq!(first_undeclared_read(&system, &guess, step), None);
            }
            // And the same rows asked for alone. This shape is where the span, scalar-hold and
            // point-stay arms live, which a single relation on the cast never produces enough of.
            assert_eq!(first_subset_disagreement(&system, &guess), None);
        }
    }

    /// What the grouping actually buys on the drawing that prompted it, as a COUNT rather than a
    /// clock: the number of residual passes one Jacobian costs.
    #[test]
    fn a_slot_costs_fewer_residual_passes_than_it_has_parameters() {
        let (mut builder, held) = cast();
        builder.add_constraint(Relation::Concentric {
            first: SketchCurve::Arc(held.arc),
            second: SketchCurve::Arc(held.other_arc),
        });
        builder.add_constraint(Relation::Horizontal { segment: held.base });
        let problem = builder.finish().unwrap();
        let scalars = problem.scalar_coordinates();
        let system = Residuals::new(
            &problem,
            &scalars,
            Rigidity::Preferred {
                anchored: &[],
                flexible_curves: &[],
                was: &[],
                opening: &[],
                reshaping: true,
            },
        )
        .unwrap();
        let grouping = ColumnGrouping::curtis_powell_reid(
            &system.parameter_reads().unwrap(),
            system.parameter_count(),
        );
        assert!(
            grouping.group_count() * 3 < system.parameter_count(),
            "{} groups against {} parameters",
            grouping.group_count(),
            system.parameter_count()
        );
        // And what each of those passes costs, in rows. A group differences a handful of columns,
        // and only the rows those columns appear in can move, so a whole vector per pass is mostly
        // arithmetic thrown away: on this slot, 88 rows against the 295 a whole vector per group
        // would be. This is the count rather than the clock on purpose. The clock reads about two
        // percent, because every narrowed pass still copies the whole coordinate vector to widen
        // it — a fixed cost per pass that the arithmetic shrinking only makes more of the total.
        let asked: usize = (0..grouping.group_count())
            .map(|group| grouping.rows_of_group(group).len())
            .sum();
        let whole = grouping.group_count() * system.residual_count();
        assert!(
            asked * 2 < whole,
            "a Jacobian evaluates {asked} rows where the whole vector each pass would be {whole}"
        );
    }

    /// **A point held to a spline lands on the curve, and stays on it when the curve is redrawn.**
    ///
    /// This is the whole of what "coincident to a spline" means, and neither half is free. Landing
    /// is the two rows; staying is the station being a solver column rather than a place frozen
    /// when the relation was authored — a frozen one would leave the point standing where the
    /// curve USED to be the moment a fit point moved.
    #[test]
    fn a_point_held_to_a_spline_lands_on_it_and_rides_when_the_spline_is_redrawn() {
        let (mut builder, relation, beside, through) = point_beside_a_spline();
        assert_eq!(relation.residual_count(), 2);
        builder.add_constraint(relation);
        let settled = builder.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied, "the point should settle");

        let places: Vec<[f64; 2]> = through
            .iter()
            .map(|point| settled.solution.position(*point).unwrap())
            .collect();
        let landed = settled.solution.position(beside).unwrap();
        assert!(
            off_the_spline(&places, landed) < 1.0e-4,
            "{landed:?} stands off the spline through {places:?}"
        );

        // Now redraw the spline under it: the middle fit point is pulled up, and the held point
        // has to follow the curve rather than stay where the old one ran.
        let (mut builder, relation, beside, through) = point_beside_a_spline();
        builder.add_constraint(relation);
        builder.add_constraint(Relation::Fix {
            point: through[1],
            at: [10.0, 14.0],
        });
        let moved = builder.finish().unwrap().settle();
        assert!(moved.diagnostics.satisfied, "the redraw should settle");
        let places: Vec<[f64; 2]> = through
            .iter()
            .map(|point| moved.solution.position(*point).unwrap())
            .collect();
        let rode = moved.solution.position(beside).unwrap();
        assert!(
            off_the_spline(&places, rode) < 1.0e-4,
            "{rode:?} came off the redrawn spline through {places:?}"
        );
        assert!(
            (rode[1] - landed[1]).abs() > 1.0,
            "the point should have been carried up with the curve: {landed:?} to {rode:?}"
        );
    }

    /// The solver drives a spline's end lever until the spline leaves a circle the way the circle
    /// was going: same direction, same curvature.
    ///
    /// The lever's LENGTH is the freedom this spends. It is a real one — the arm is an ordinary
    /// point in the parameter vector — which is the whole reason curvature can be a true relation
    /// here rather than something written into the geometry after the fact.
    #[test]
    fn a_spline_end_settles_into_curvature_continuity_with_a_circle() {
        let mut builder = ProblemBuilder::new();
        // A circle of radius 5 about the origin, and a spline whose first point sits on it at
        // [5, 0]. Its lever starts pointing the right way but at the wrong LENGTH, so the joint is
        // tangent from the start and the curvature is what has to be found.
        let center = builder.add_point([0.0, 0.0]);
        // The circle is GIVEN, not negotiable: left free, the solver meets the relation by
        // shrinking it to the spline instead of bending the spline to it, which is a valid answer
        // to a question nobody asked.
        let radius = builder.add_fixed_positive_radius(5.0).unwrap();
        let circle = builder.add_circle(center, radius);
        let joint = builder.add_point([5.0, 0.0]);
        let joint_arm = builder.add_point([5.0, 0.35]);
        let neighbor = builder.add_point([4.0, 4.0]);
        let neighbor_arm = builder.add_point([4.0 - 2.1 / 3.0, 4.0 + 1.0 / 3.0]);
        // Pin everything the gesture does not author, so the only way to meet the relation is to
        // move the arm — otherwise least motion is free to drag the circle to the spline.
        builder.add_constraint(Relation::Fix {
            point: center,
            at: [0.0, 0.0],
        });
        builder.add_constraint(Relation::Fix {
            point: joint,
            at: [5.0, 0.0],
        });
        builder.add_constraint(Relation::Fix {
            point: neighbor,
            at: [4.0, 4.0],
        });
        builder.add_constraint(Relation::Fix {
            point: neighbor_arm,
            at: [4.0 - 2.1 / 3.0, 4.0 + 1.0 / 3.0],
        });
        let direction = Relation::TangentDirection {
            joint,
            joint_arm,
            against: SketchCurve::Circle(circle),
        };
        let curvature = Relation::Curvature {
            joint,
            joint_arm,
            neighbor,
            neighbor_arm,
            end: SpanEnd::Start,
            against: SketchCurve::Circle(circle),
        };
        assert_eq!(direction.residual_count(), 1);
        // Curvature carries the direction row itself, so this is G2 on its own.
        assert_eq!(curvature.residual_count(), 2);
        builder.add_constraint(direction);
        builder.add_constraint(curvature);

        let settled = builder.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied, "the joint should settle");

        let arm = settled.solution.position(joint_arm).unwrap();
        // Still tangent: the arm stands straight up from a point on the circle's equator.
        assert!(
            (arm[0] - 5.0).abs() < SATISFIED_RESIDUAL,
            "the lever left the tangent: {arm:?}"
        );
        // And the curvature matches, which for this span happens at a lever of length 1.
        assert!(
            (arm[1].abs() - 1.0).abs() < 1.0e-4,
            "the lever should have found length 1, stands at {arm:?}"
        );
    }

    /// **A closed spline's station wraps, and a point held near the seam is held either side of it.**
    ///
    /// The station of a closed curve is a quantity that WRAPS, which is the family the arc sweep
    /// belongs to and the family that hides seam bugs. A point sitting a hair before the join and
    /// one sitting a hair after it are a hair apart on the drawing, and the solve has to agree —
    /// if the wrap kinked, one of the pair would be dragged the long way round.
    #[test]
    fn a_point_held_near_a_closed_splines_seam_settles_on_either_side_of_it() {
        let places = [[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0], [0.0, -10.0]];
        let landed = |beside: [f64; 2]| {
            let mut builder = ProblemBuilder::new();
            let through: Vec<PointId> = places.iter().map(|at| builder.add_point(*at)).collect();
            let spline = builder.add_fit_point_spline(through.clone(), Vec::new(), true);
            let standing = builder.add_point(beside);
            for (point, at) in through.iter().zip(places) {
                builder.add_constraint(Relation::Fix { point: *point, at });
            }
            let candidate = fit_point_spline(&places, &[None; 4], true).unwrap();
            let seed = station_length(&candidate);
            let station = builder.add_free_spline_station(seed).unwrap();
            builder.add_constraint(Relation::PointOnSpline {
                point: standing,
                spline,
                station,
            });
            let settled = builder.finish().unwrap().settle();
            assert!(settled.diagnostics.satisfied, "{beside:?} did not settle");
            settled.solution.position(standing).unwrap()
        };

        // The seam of this curve is its first fit point, at [10, 0]. Two witnesses a whisker
        // either side of it, both a little outside the loop.
        let before = landed([11.0, -0.6]);
        let after = landed([11.0, 0.6]);
        assert!(
            (before[0] - after[0]).hypot(before[1] - after[1]) < 1.5,
            "the seam pulled them apart: {before:?} against {after:?}"
        );
        for on in [before, after] {
            assert!(
                (on[0].hypot(on[1]) - 10.0).abs() < 0.5,
                "{on:?} is not on the loop"
            );
        }
    }

    #[test]
    fn quantize_places_both_coordinates_on_the_authored_lattice() {
        let mut builder = ProblemBuilder::new();
        let point = builder.add_point([2.6, -1.6]);
        builder.add_constraint(Relation::Quantize {
            point,
            pitch: 2.0,
            phase: 0.5,
        });
        let settled = builder.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied);
        let at = settled.solution.position(point).unwrap();
        assert!((at[0] - 2.5).abs() < SATISFIED_RESIDUAL);
        assert!((at[1] + 1.5).abs() < SATISFIED_RESIDUAL);
    }

    #[test]
    /// A special edit that promises not to move unrelated authored geometry must not accidentally
    /// use `analyze`: it settles a copy. The current-configuration API instead reads the stored
    /// witness exactly as it is, before any numerical repair.
    fn current_validation_does_not_heal_an_unsatisfied_authored_witness() {
        let mut builder = ProblemBuilder::new();
        let from = builder.add_point([0.0, 0.0]);
        let to = builder.add_point([10.0, 4.0]);
        let segment = builder.add_segment(from, to);
        builder.add_constraint(Relation::Horizontal { segment });
        let problem = builder.finish().unwrap();

        assert!(
            !problem.validate_current().satisfied,
            "stored slant is still authored"
        );
        assert!(
            problem.analyze().diagnostics.satisfied,
            "the solver can heal its analysis copy"
        );
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
        let report = Some(SolveReport {
            outcome: SolveOutcome::Stalled,
            iterations: 1,
            residual_norm: SATISFIED_RESIDUAL / 2.0,
            degrees_of_freedom: 0,
            redundant_residuals: 0,
        });
        let diagnostics = Diagnostics {
            satisfied: report
                .as_ref()
                .is_some_and(|report| report.residual_norm <= SATISFIED_RESIDUAL),
            report,
            tangent_contacts_valid: true,
        };
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
        assert_eq!(curve, SketchCurve::Segment(segment));
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
    fn standing_conflicts_returns_each_individually_removable_local_constraint() {
        let mut builder = ProblemBuilder::new();
        let first_center = builder.add_point([0.0, 0.0]);
        let first_radius = builder.add_fixed_positive_radius(2.0).unwrap();
        let first = builder.add_circle(first_center, first_radius);
        let second_center = builder.add_point([10.0, 0.0]);
        let second_radius = builder.add_fixed_positive_radius(5.0).unwrap();
        let second = builder.add_circle(second_center, second_radius);
        let concentric = builder.add_constraint(Relation::Concentric {
            first: SketchCurve::Circle(first),
            second: SketchCurve::Circle(second),
        });
        let first_fix = builder.add_constraint(Relation::Fix {
            point: first_center,
            at: [0.0, 0.0],
        });
        let second_fix = builder.add_constraint(Relation::Fix {
            point: second_center,
            at: [10.0, 0.0],
        });
        let problem = builder.finish().unwrap();

        assert_eq!(
            problem.standing_conflicts(),
            vec![concentric, first_fix, second_fix]
        );
    }

    #[test]
    fn satisfied_standing_problem_has_no_conflicts() {
        let mut builder = ProblemBuilder::new();
        let point = builder.add_point([3.0, 4.0]);
        builder.add_constraint(Relation::Fix {
            point,
            at: [3.0, 4.0],
        });
        let problem = builder.finish().unwrap();

        assert!(problem.settle().diagnostics.satisfied);
        assert!(problem.standing_conflicts().is_empty());
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
            free.add_free_positive_radius(source_value),
            Err(BuildError::InvalidParameter)
        ));

        let mut fixed = ProblemBuilder::new();
        let radius = fixed.add_fixed_positive_radius(source_value).unwrap();
        let problem = fixed.finish().unwrap();
        assert_eq!(problem.analyze().degrees_of_freedom, 0);
        let settled = problem.settle();
        let Some(ParameterValue::Radius(solved_radius)) = settled.solution.parameter(radius) else {
            panic!("the radius has its declared scalar type");
        };
        assert_eq!(solved_radius.to_bits(), source_value.to_bits());

        assert_eq!(
            physical_parameter_value(problem.parameters[radius.index], f64::NEG_INFINITY).to_bits(),
            source_value.to_bits(),
            "fixed radius bypasses the free-value exponential transform"
        );
    }

    #[test]
    fn scalar_transforms_stay_topology_safe_and_exactly_writable_at_extremes() {
        let mut builder = ProblemBuilder::new();
        let radius = builder.add_free_positive_radius(1.0).unwrap();
        let problem = builder.finish().unwrap();
        let radius_parameter = problem.parameters[radius.index];

        for coordinate in [f64::NEG_INFINITY, -1.0e300, 1.0e300, f64::INFINITY] {
            let radius_value = physical_parameter_value(radius_parameter, coordinate);
            assert!(radius_value > 0.0 && radius_value.is_finite());
            assert!(
                ResolvedLength::try_from_f64(radius_value).is_ok(),
                "radius {radius_value:?} at coordinate {coordinate:?}"
            );
        }
    }

    #[test]
    /// The one row an arc contributes (ADR 0038): its center stands the same distance from both
    /// of its ends. Fix the ends, put the center somewhere lopsided, and the settle pulls it onto
    /// the chord's bisector instead of leaving three unrelated points calling themselves an arc.
    fn an_arc_center_settles_the_same_distance_from_both_ends() {
        let mut builder = ProblemBuilder::new();
        let from = builder.add_point([0.0, 0.0]);
        let to = builder.add_point([10.0, 0.0]);
        let center = builder.add_point([2.0, 4.0]);
        builder.add_arc(center, from, to);
        builder.add_constraint(Relation::Fix {
            point: from,
            at: [0.0, 0.0],
        });
        builder.add_constraint(Relation::Fix {
            point: to,
            at: [10.0, 0.0],
        });
        let settled = builder.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied);
        let at = settled.solution.position(center).unwrap();
        let reach = |end: [f64; 2]| (at[0] - end[0]).hypot(at[1] - end[1]);
        assert!(
            (reach([0.0, 0.0]) - reach([10.0, 0.0])).abs() < 1.0e-6,
            "center settled at {at:?}"
        );
    }

    #[test]
    fn intrinsic_parameter_building_rejects_wrong_domains() {
        let mut builder = ProblemBuilder::new();
        assert!(matches!(
            builder.add_free_positive_radius(0.0),
            Err(BuildError::InvalidParameter)
        ));
        assert!(matches!(
            builder.add_fixed_positive_radius(f64::INFINITY),
            Err(BuildError::InvalidParameter)
        ));
    }

    fn circle(builder: &mut ProblemBuilder, center: [f64; 2], radius: f64) -> CircleId {
        let center = builder.add_point(center);
        let radius = builder.add_fixed_positive_radius(radius).unwrap();
        builder.add_circle(center, radius)
    }

    fn arc(builder: &mut ProblemBuilder, from: [f64; 2], to: [f64; 2]) -> ArcId {
        let center_at = [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0];
        let from = builder.add_point(from);
        let to = builder.add_point(to);
        let center = builder.add_point(center_at);
        builder.add_arc(center, from, to)
    }

    #[test]
    fn concentric_has_two_rows_and_never_equalizes_radii() {
        let mut builder = ProblemBuilder::new();
        let first_center = builder.add_point([0.0, 0.0]);
        let first_radius = builder.add_free_positive_radius(2.0).unwrap();
        let first = builder.add_circle(first_center, first_radius);
        let second_center = builder.add_point([6.0, 4.0]);
        let second_radius = builder.add_free_positive_radius(7.0).unwrap();
        let second = builder.add_circle(second_center, second_radius);
        builder.add_constraint(Relation::Concentric {
            first: SketchCurve::Circle(first),
            second: SketchCurve::Circle(second),
        });
        let problem = builder.finish().unwrap();
        let analysis = problem.analyze();
        assert!(analysis.diagnostics.satisfied);
        assert_eq!(analysis.witness_rank, 2);
        assert_eq!(analysis.degrees_of_freedom, 4);
        let Some(ParameterValue::Radius(first_solved)) = analysis.solution.parameter(first_radius)
        else {
            panic!("first radius")
        };
        let Some(ParameterValue::Radius(second_solved)) =
            analysis.solution.parameter(second_radius)
        else {
            panic!("second radius")
        };
        assert_eq!(first_solved.to_bits(), 2.0_f64.to_bits());
        assert_eq!(second_solved.to_bits(), 7.0_f64.to_bits());
    }

    #[test]
    fn concentric_accepts_every_circular_pair_in_either_order() {
        for (first_is_arc, second_is_arc, reversed) in [
            (true, true, false),
            (true, true, true),
            (true, false, false),
            (true, false, true),
            (false, false, false),
            (false, false, true),
        ] {
            let mut builder = ProblemBuilder::new();
            let first = if first_is_arc {
                SketchCurve::Arc(arc(&mut builder, [0.0, 0.0], [4.0, 0.0]))
            } else {
                SketchCurve::Circle(circle(&mut builder, [2.0, 0.0], 2.0))
            };
            let second = if second_is_arc {
                SketchCurve::Arc(arc(&mut builder, [4.0, 6.0], [8.0, 6.0]))
            } else {
                SketchCurve::Circle(circle(&mut builder, [6.0, 6.0], 5.0))
            };
            let (first, second) = if reversed {
                (second, first)
            } else {
                (first, second)
            };
            builder.add_constraint(Relation::Concentric { first, second });
            assert!(builder.finish().unwrap().settle().diagnostics.satisfied);
        }
    }

    #[test]
    fn concentric_refuses_segments_and_self_pairs() {
        let mut segment_pair = ProblemBuilder::new();
        let from = segment_pair.add_point([0.0, 0.0]);
        let to = segment_pair.add_point([1.0, 0.0]);
        let segment = segment_pair.add_segment(from, to);
        let circular = circle(&mut segment_pair, [0.0, 0.0], 1.0);
        segment_pair.add_constraint(Relation::Concentric {
            first: SketchCurve::Segment(segment),
            second: SketchCurve::Circle(circular),
        });
        assert!(matches!(
            segment_pair.finish(),
            Err(BuildError::InvalidConcentric)
        ));

        let mut self_pair = ProblemBuilder::new();
        let circular = circle(&mut self_pair, [0.0, 0.0], 1.0);
        self_pair.add_constraint(Relation::Concentric {
            first: SketchCurve::Circle(circular),
            second: SketchCurve::Circle(circular),
        });
        assert!(matches!(
            self_pair.finish(),
            Err(BuildError::InvalidConcentric)
        ));
    }

    #[test]
    fn concentric_center_uses_the_solver_satisfaction_boundary() {
        for (offset, valid) in [
            (SATISFIED_RESIDUAL, true),
            (f64::from_bits(SATISFIED_RESIDUAL.to_bits() + 1), false),
        ] {
            let first = [0.0, 0.0];
            let second = [offset, 0.0];
            assert_eq!(concentric_center(first, second).is_some(), valid);

            let mut builder = ProblemBuilder::new();
            let first_center = builder.add_point(first);
            let first_radius = builder.add_fixed_positive_radius(2.0).unwrap();
            let first_circle = builder.add_circle(first_center, first_radius);
            let second_center = builder.add_point(second);
            let second_radius = builder.add_fixed_positive_radius(5.0).unwrap();
            let second_circle = builder.add_circle(second_center, second_radius);
            builder.add_constraint(Relation::Concentric {
                first: SketchCurve::Circle(first_circle),
                second: SketchCurve::Circle(second_circle),
            });
            let problem = builder.finish().unwrap();
            let scalar_coordinates = problem.scalar_coordinates();
            let positions = problem
                .points
                .iter()
                .map(|point| point.at)
                .collect::<Vec<_>>();
            let system = Residuals::new(&problem, &scalar_coordinates, Rigidity::Ignored).unwrap();
            let parameters = system.guess(&positions);
            let mut residuals = vec![0.0; system.residual_count()];
            system.residuals(&parameters, &mut residuals);
            let solver_valid = residuals
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                <= SATISFIED_RESIDUAL;
            assert_eq!(solver_valid, valid);
        }
    }

    /// Which end of an arc is called `from` is a LABEL, and a label does not change the equations.
    ///
    /// A wind reverses endpoint order (ADR 0041) and is supposed to change only which way round
    /// the arc is drawn. It did not: an arc's radius was read off its `from` end alone, so the
    /// reversed drawing asked a different question of every tangency touching it. On the author's
    /// own arc slot the same hands over the same seven points converged in two iterations at
    /// 5.1e-12 one way round and exhausted a hundred at 14.1 the other, and the sweep dropped its
    /// snap ghost for thirteen frames together.
    ///
    /// The ends here are deliberately UNEQUAL — reach 10 against reach 14 about the same center.
    /// That is the state the invariant protects and the only state it can be tested in: equal
    /// radius is a ROW the solve is answering, so a converged arc satisfies this vacuously while
    /// every arc mid-pass does not. Their MEAN is 12, so the tangency reads a circle of 12 either
    /// way round; read off one end it reads 10 one way and 14 the other, and the row moves by four.
    ///
    /// Only the arc's own equal-radius row may differ, and only by sign: it says `from` less `to`,
    /// which is the one place endpoint order is the point.
    #[test]
    fn reversing_an_arcs_ends_leaves_the_equations_where_they_stood() {
        let rows_when = |reversed: bool| {
            let mut builder = ProblemBuilder::new();
            let center = builder.add_point([0.0, 0.0]);
            let near = builder.add_point([10.0, 0.0]);
            let far = builder.add_point([0.0, 14.0]);
            let arc = if reversed {
                builder.add_arc(center, far, near)
            } else {
                builder.add_arc(center, near, far)
            };
            // Placed so that neither one-end reading is right: the mean of 12 leaves one unit of
            // residual, the near end three and the far end minus one. Reading a norm would have
            // missed a reading that only flips sign.
            let ring = circle(&mut builder, [31.0, 0.0], 18.0);
            builder.add_constraint(Relation::Tangent {
                first: SketchCurve::Arc(arc),
                second: SketchCurve::Circle(ring),
                branch: TangentBranch::External,
            });
            let problem = builder.finish().unwrap();
            let scalars = problem.scalar_coordinates();
            let positions: Vec<[f64; 2]> = problem.points.iter().map(|point| point.at).collect();
            let system = Residuals::new(&problem, &scalars, Rigidity::Ignored).unwrap();
            let mut rows = vec![f64::NAN; system.residual_count()];
            system.residuals(&system.guess(&positions), &mut rows);
            rows
        };
        let (stood, reversed) = (rows_when(false), rows_when(true));
        assert_eq!(
            stood.len(),
            reversed.len(),
            "reversing an arc changed how many rows the drawing has"
        );
        let magnitudes = |rows: &[f64]| rows.iter().map(|row| row.abs()).collect::<Vec<_>>();
        assert_eq!(
            magnitudes(&stood),
            magnitudes(&reversed),
            "reversing the arc moved the equations: {stood:?} against {reversed:?}"
        );
    }

    #[test]
    fn tangent_has_one_row_and_keeps_the_segment_direction_for_line_side() {
        let mut builder = ProblemBuilder::new();
        let from = builder.add_point([-10.0, 0.0]);
        let to = builder.add_point([10.0, 0.0]);
        let line = builder.add_segment(from, to);
        let circle = circle(&mut builder, [0.0, 4.0], 4.0);
        builder.add_constraint(Relation::Tangent {
            // The line is deliberately second: Left is nevertheless the segment's stored
            // `from → to` left side, never the canonical relation member order.
            first: SketchCurve::Circle(circle),
            second: SketchCurve::Segment(line),
            branch: TangentBranch::Line(LineSide::Left),
        });
        let problem = builder.finish().unwrap();
        let analysis = problem.analyze();
        assert_eq!(problem.relation_count(), 1);
        assert_eq!(analysis.witness_rank, 1);
        assert_eq!(
            analysis.degrees_of_freedom, 5,
            "six coordinates minus one Tangent row"
        );
        assert!(analysis.diagnostics.tangent_contacts_valid);
        let contact = problem
            .tangent_contact(
                Relation::Tangent {
                    first: SketchCurve::Circle(circle),
                    second: SketchCurve::Segment(line),
                    branch: TangentBranch::Line(LineSide::Left),
                },
                &analysis.solution,
            )
            .unwrap();
        assert!((contact.at[0]).abs() < 1e-8 && (contact.at[1]).abs() < 1e-8);
    }

    #[test]
    fn tangent_circular_branches_share_one_contact_formula() {
        let external = {
            let mut builder = ProblemBuilder::new();
            let first = circle(&mut builder, [0.0, 0.0], 4.0);
            let second = circle(&mut builder, [8.0, 0.0], 4.0);
            builder.add_constraint(Relation::Tangent {
                first: SketchCurve::Circle(first),
                second: SketchCurve::Circle(second),
                branch: TangentBranch::External,
            });
            (
                builder.finish().unwrap(),
                first,
                second,
                TangentBranch::External,
                [4.0, 0.0],
            )
        };
        let internal_first = {
            let mut builder = ProblemBuilder::new();
            let first = circle(&mut builder, [0.0, 0.0], 6.0);
            let second = circle(&mut builder, [4.0, 0.0], 2.0);
            let branch = TangentBranch::Internal {
                contains: InternalContainment::First,
            };
            builder.add_constraint(Relation::Tangent {
                first: SketchCurve::Circle(first),
                second: SketchCurve::Circle(second),
                branch,
            });
            (builder.finish().unwrap(), first, second, branch, [6.0, 0.0])
        };
        let internal_second = {
            let mut builder = ProblemBuilder::new();
            let first = circle(&mut builder, [4.0, 0.0], 2.0);
            let second = circle(&mut builder, [0.0, 0.0], 6.0);
            let branch = TangentBranch::Internal {
                contains: InternalContainment::Second,
            };
            builder.add_constraint(Relation::Tangent {
                first: SketchCurve::Circle(first),
                second: SketchCurve::Circle(second),
                branch,
            });
            (builder.finish().unwrap(), first, second, branch, [6.0, 0.0])
        };
        for (problem, first, second, branch, expected) in
            [external, internal_first, internal_second]
        {
            let settled = problem.settle();
            assert!(settled.diagnostics.satisfied && settled.diagnostics.tangent_contacts_valid);
            let contact = problem
                .tangent_contact(
                    Relation::Tangent {
                        first: SketchCurve::Circle(first),
                        second: SketchCurve::Circle(second),
                        branch,
                    },
                    &settled.solution,
                )
                .unwrap();
            assert!(
                ((contact.at[0] - expected[0]).powi(2) + (contact.at[1] - expected[1]).powi(2))
                    .sqrt()
                    < 1e-7,
                "{branch:?} contact was {:?}",
                contact.at
            );
        }
    }

    #[test]
    fn a_supporting_line_contact_outside_a_finite_segment_is_refused_after_trial() {
        let mut builder = ProblemBuilder::new();
        let from = builder.add_point([-1.0, 0.0]);
        let to = builder.add_point([1.0, 0.0]);
        let line = builder.add_segment(from, to);
        let circular = circle(&mut builder, [3.0, 4.0], 4.0);
        let problem = builder.finish().unwrap();
        let rejected = problem
            .trial_add(Relation::Tangent {
                first: SketchCurve::Segment(line),
                second: SketchCurve::Circle(circular),
                branch: TangentBranch::Line(LineSide::Left),
            })
            .unwrap();
        assert!(matches!(
            rejected,
            TrialAdd::Rejected(TrialRejection::InvalidTangent {
                error: TangentContactError::OutsideFirstDomain,
                ..
            })
        ));
    }

    #[test]
    fn segment_segment_tangent_is_not_a_solver_relation() {
        let mut builder = ProblemBuilder::new();
        let a = builder.add_point([0.0, 0.0]);
        let b = builder.add_point([1.0, 0.0]);
        let c = builder.add_point([0.0, 1.0]);
        let d = builder.add_point([1.0, 1.0]);
        let first = builder.add_segment(a, b);
        let second = builder.add_segment(c, d);
        builder.add_constraint(Relation::Tangent {
            first: SketchCurve::Segment(first),
            second: SketchCurve::Segment(second),
            branch: TangentBranch::Line(LineSide::Left),
        });
        assert!(matches!(builder.finish(), Err(BuildError::InvalidTangent)));
    }

    #[test]
    fn symmetry_relation_rows_rank_and_freedom_match_each_curve_kind() {
        let mut segments = ProblemBuilder::new();
        let axis_from = segments.add_point([0.0, -5.0]);
        let axis_to = segments.add_point([0.0, 5.0]);
        let axis = segments.add_segment(axis_from, axis_to);
        let a0 = segments.add_point([-3.0, 0.0]);
        let a1 = segments.add_point([-2.0, 2.0]);
        let b0 = segments.add_point([3.0, 0.0]);
        let b1 = segments.add_point([2.0, 2.0]);
        let first = segments.add_segment(a0, a1);
        let second = segments.add_segment(b0, b1);
        let relation = Relation::Symmetry {
            first: SketchCurve::Segment(first),
            second: SketchCurve::Segment(second),
            axis,
            branch: SymmetryBranch::Direct,
        };
        assert_eq!(relation.residual_count(), 4);
        segments.add_constraint(relation);
        let analysis = segments.finish().unwrap().analyze();
        assert_eq!((analysis.witness_rank, analysis.degrees_of_freedom), (4, 8));

        let mut circles = ProblemBuilder::new();
        let axis_from = circles.add_point([0.0, -5.0]);
        let axis_to = circles.add_point([0.0, 5.0]);
        let axis = circles.add_segment(axis_from, axis_to);
        let first_center = circles.add_point([-3.0, 0.0]);
        let second_center = circles.add_point([3.0, 0.0]);
        let first_radius = circles.add_free_positive_radius(2.0).unwrap();
        let second_radius = circles.add_free_positive_radius(2.0).unwrap();
        let first = circles.add_circle(first_center, first_radius);
        let second = circles.add_circle(second_center, second_radius);
        let relation = Relation::Symmetry {
            first: SketchCurve::Circle(first),
            second: SketchCurve::Circle(second),
            axis,
            branch: SymmetryBranch::Centers,
        };
        assert_eq!(relation.residual_count(), 3);
        circles.add_constraint(relation);
        let analysis = circles.finish().unwrap().analyze();
        assert_eq!((analysis.witness_rank, analysis.degrees_of_freedom), (3, 7));

        let mut arcs = ProblemBuilder::new();
        let axis_from = arcs.add_point([0.0, -5.0]);
        let axis_to = arcs.add_point([0.0, 5.0]);
        let axis = arcs.add_segment(axis_from, axis_to);
        let a0 = arcs.add_point([-3.0, 0.0]);
        let a1 = arcs.add_point([-2.0, 2.0]);
        let ac = arcs.add_point([0.0, 0.0]);
        let b0 = arcs.add_point([3.0, 0.0]);
        let b1 = arcs.add_point([2.0, 2.0]);
        let bc = arcs.add_point([0.0, 0.0]);
        let first = arcs.add_arc(ac, a0, a1);
        let second = arcs.add_arc(bc, b0, b1);
        let relation = Relation::Symmetry {
            first: SketchCurve::Arc(first),
            second: SketchCurve::Arc(second),
            axis,
            branch: SymmetryBranch::Direct,
        };
        assert_eq!(relation.residual_count(), 5);
        arcs.add_constraint(relation);
        let analysis = arcs.finish().unwrap().analyze();
        // Eight points and two arc radii, so eighteen columns, against nine rows — five for the
        // symmetry and two per arc, each end standing its own radius from its center. All nine are
        // independent, leaving nine freedoms.
        //
        // These particular numbers settle into a COLLAPSE: each arc's two ends land on each other
        // about a millionth of a unit apart. That used to cost the reading two of its rows. An arc
        // with no sweep has an equal-radius row `|from - c| = |to - c|` that says nothing once the
        // two ends coincide, and its Jacobian degenerates with it, so the rank came back five and
        // the freedom eleven. Named, the radius keeps the pair honest: `|from - c| = r` and
        // `|to - c| = r` still constrain the ends even where the difference between them does not,
        // and the reading no longer depends on the drawing being non-degenerate to be right.
        //
        // The freedom itself is unchanged, which is the point — a column and a row per arc net to
        // nothing. Sixteen coordinates less seven rows would also have been nine, had the rank
        // reading been able to see it. This test is about row counts and rank, so a degenerate
        // answer serves it; nothing here draws.
        assert_eq!((analysis.witness_rank, analysis.degrees_of_freedom), (9, 9));
    }

    #[test]
    fn symmetry_kernel_validates_identity_type_branch_and_axis() {
        let build = |second_kind: u8, branch, alias_axis: bool, collapsed_axis: bool| {
            let mut builder = ProblemBuilder::new();
            let axis_from = builder.add_point([0.0, 0.0]);
            let axis_to = builder.add_point(if collapsed_axis {
                [0.0, 0.0]
            } else {
                [0.0, 5.0]
            });
            let axis = builder.add_segment(axis_from, axis_to);
            let a = builder.add_point([-2.0, 0.0]);
            let b = builder.add_point([-2.0, 2.0]);
            let first = builder.add_segment(a, b);
            let second = if second_kind == 0 {
                let c = builder.add_point([2.0, 0.0]);
                let d = builder.add_point([2.0, 2.0]);
                SketchCurve::Segment(builder.add_segment(c, d))
            } else {
                SketchCurve::Circle(circle(&mut builder, [2.0, 1.0], 1.0))
            };
            builder.add_constraint(Relation::Symmetry {
                first: SketchCurve::Segment(first),
                second,
                axis: if alias_axis { first } else { axis },
                branch,
            });
            builder.finish()
        };
        assert!(build(0, SymmetryBranch::Direct, false, false).is_ok());
        for invalid in [
            build(1, SymmetryBranch::Direct, false, false),
            build(0, SymmetryBranch::Centers, false, false),
            build(0, SymmetryBranch::Direct, true, false),
            build(0, SymmetryBranch::Direct, false, true),
        ] {
            assert!(matches!(invalid, Err(BuildError::InvalidSymmetry)));
        }
        let mut self_pair = ProblemBuilder::new();
        let axis_from = self_pair.add_point([0.0, -2.0]);
        let axis_to = self_pair.add_point([0.0, 2.0]);
        let axis = self_pair.add_segment(axis_from, axis_to);
        let from = self_pair.add_point([-2.0, 0.0]);
        let to = self_pair.add_point([-2.0, 2.0]);
        let subject = self_pair.add_segment(from, to);
        self_pair.add_constraint(Relation::Symmetry {
            first: SketchCurve::Segment(subject),
            second: SketchCurve::Segment(subject),
            axis,
            branch: SymmetryBranch::Direct,
        });
        assert!(matches!(
            self_pair.finish(),
            Err(BuildError::InvalidSymmetry)
        ));
    }

    #[test]
    fn symmetry_trial_keeps_axis_exact_and_drag_can_move_it_later() {
        let mut builder = ProblemBuilder::new();
        let axis_from = builder.add_point([0.0, -5.0]);
        let axis_to = builder.add_point([0.0, 5.0]);
        let axis = builder.add_segment(axis_from, axis_to);
        let a0 = builder.add_point([-5.0, 0.0]);
        let a1 = builder.add_point([-2.0, 3.0]);
        let b0 = builder.add_point([7.0, 1.0]);
        let b1 = builder.add_point([8.0, 5.0]);
        let first = builder.add_segment(a0, a1);
        let second = builder.add_segment(b0, b1);
        let problem = builder.finish().unwrap();
        let relation = Relation::Symmetry {
            first: SketchCurve::Segment(first),
            second: SketchCurve::Segment(second),
            axis,
            branch: SymmetryBranch::Direct,
        };
        let TrialAdd::Accepted { settled, .. } = problem.trial_add(relation).unwrap() else {
            panic!("free subjects")
        };
        assert_eq!(settled.solution.position(axis_from), Some([0.0, -5.0]));
        assert_eq!(settled.solution.position(axis_to), Some([0.0, 5.0]));

        let mut standing = ProblemBuilder::new();
        let axis_from = standing.add_point([0.0, -5.0]);
        let axis_to = standing.add_point([0.0, 5.0]);
        let axis = standing.add_segment(axis_from, axis_to);
        let a0 = standing.add_point([-3.0, 0.0]);
        let a1 = standing.add_point([-2.0, 2.0]);
        let b0 = standing.add_point([3.0, 0.0]);
        let b1 = standing.add_point([2.0, 2.0]);
        let first = standing.add_segment(a0, a1);
        let second = standing.add_segment(b0, b1);
        standing.add_constraint(Relation::Symmetry {
            first: SketchCurve::Segment(first),
            second: SketchCurve::Segment(second),
            axis,
            branch: SymmetryBranch::Direct,
        });
        let standing = standing.finish().unwrap();
        let DragOutcome::Accepted(dragged) = standing.drag(axis_to, [2.0, 6.0]).unwrap() else {
            panic!("axis drag")
        };
        assert_ne!(dragged.solution.position(axis_to), Some([0.0, 5.0]));
        assert!(standing
            .symmetry_witness(
                Relation::Symmetry {
                    first: SketchCurve::Segment(first),
                    second: SketchCurve::Segment(second),
                    axis,
                    branch: SymmetryBranch::Direct,
                },
                &dragged.solution,
            )
            .is_ok());
    }

    #[test]
    fn symmetry_trial_holds_derived_axis_centers_and_reports_the_retained_solution() {
        let mut builder = ProblemBuilder::new();
        let lower_center = builder.add_point([0.0, -5.0]);
        let lower_from = builder.add_point([-1.0, -5.0]);
        let lower_to = builder.add_point([1.0, -5.0]);
        builder.add_arc(lower_center, lower_from, lower_to);
        let upper_center = builder.add_point([0.0, 5.0]);
        let upper_from = builder.add_point([-1.0, 5.0]);
        let upper_to = builder.add_point([1.0, 5.0]);
        builder.add_arc(upper_center, upper_from, upper_to);
        let axis = builder.add_segment(lower_center, upper_center);
        let a0 = builder.add_point([-5.0, 0.0]);
        let a1 = builder.add_point([-2.0, 3.0]);
        let b0 = builder.add_point([7.0, 1.0]);
        let b1 = builder.add_point([8.0, 5.0]);
        let first = builder.add_segment(a0, a1);
        let second = builder.add_segment(b0, b1);
        let problem = builder.finish().unwrap();
        let relation = Relation::Symmetry {
            first: SketchCurve::Segment(first),
            second: SketchCurve::Segment(second),
            axis,
            branch: SymmetryBranch::Direct,
        };
        let TrialAdd::Accepted { settled, .. } = problem.trial_add(relation).unwrap() else {
            panic!("free subjects")
        };
        for (center, expected) in [(lower_center, [0.0, -5.0]), (upper_center, [0.0, 5.0])] {
            let actual = settled.solution.position(center).unwrap();
            assert!((actual[0] - expected[0]).abs() <= SATISFIED_RESIDUAL);
            assert!((actual[1] - expected[1]).abs() <= SATISFIED_RESIDUAL);
        }

        let resolved = problem.resolve(relation).unwrap();
        let candidate = problem.with_candidate(relation, resolved);
        let mut preferred_positions: Vec<_> =
            candidate.points.iter().map(|point| point.at).collect();
        let mut preferred_scalars = candidate.scalar_coordinates();
        let preferred_trace = run_reporting_only_the_search(
            &candidate,
            &mut preferred_positions,
            &mut preferred_scalars,
            Rigidity::Preferred {
                anchored: &[lower_center, upper_center],
                flexible_curves: &[SketchCurve::Segment(first), SketchCurve::Segment(second)],
                was: &[],
                opening: &[],
                reshaping: false,
            },
        )
        .unwrap();
        let scalars = scalar_coordinates_of_solution(&candidate, &settled.solution).unwrap();
        let measured = exact_report_at(
            &candidate,
            &settled.solution.positions,
            &scalars,
            preferred_trace,
        )
        .unwrap();
        let reported = settled.diagnostics.report.unwrap();
        assert_eq!(reported.outcome, preferred_trace.outcome);
        assert_eq!(reported.iterations, preferred_trace.iterations);
        assert_eq!(
            reported.residual_norm.to_bits(),
            measured.residual_norm.to_bits()
        );
        assert_eq!(
            (reported.degrees_of_freedom, reported.redundant_residuals),
            (measured.degrees_of_freedom, measured.redundant_residuals)
        );
    }

    #[test]
    fn symmetry_direct_reversed_and_member_axis_reversal_are_kernel_invariant() {
        for branch in [SymmetryBranch::Direct, SymmetryBranch::Reversed] {
            for reverse_members in [false, true] {
                for reverse_axis in [false, true] {
                    let mut builder = ProblemBuilder::new();
                    let low = builder.add_point([0.0, -5.0]);
                    let high = builder.add_point([0.0, 5.0]);
                    let axis = if reverse_axis {
                        builder.add_segment(high, low)
                    } else {
                        builder.add_segment(low, high)
                    };
                    let a0 = builder.add_point([-3.0, 0.0]);
                    let a1 = builder.add_point([-2.0, 2.0]);
                    let (second_from, second_to) = if branch == SymmetryBranch::Direct {
                        ([3.0, 0.0], [2.0, 2.0])
                    } else {
                        ([2.0, 2.0], [3.0, 0.0])
                    };
                    let b0 = builder.add_point(second_from);
                    let b1 = builder.add_point(second_to);
                    let first = builder.add_segment(a0, a1);
                    let second = builder.add_segment(b0, b1);
                    let (first, second) = if reverse_members {
                        (second, first)
                    } else {
                        (first, second)
                    };
                    let relation = Relation::Symmetry {
                        first: SketchCurve::Segment(first),
                        second: SketchCurve::Segment(second),
                        axis,
                        branch,
                    };
                    builder.add_constraint(relation);
                    let problem = builder.finish().unwrap();
                    let analysis = problem.analyze();
                    assert!(analysis.diagnostics.satisfied);
                    assert!(problem
                        .symmetry_witness(relation, &analysis.solution)
                        .is_ok());
                }
            }
        }
    }

    #[test]
    fn symmetry_kernel_reads_fixed_and_writes_only_free_curve_scalars() {
        let mut circles = ProblemBuilder::new();
        let low = circles.add_point([0.0, -5.0]);
        let high = circles.add_point([0.0, 5.0]);
        let axis = circles.add_segment(low, high);
        let first_center = circles.add_point([-3.0, 0.0]);
        let second_center = circles.add_point([3.0, 0.0]);
        let free_radius = circles.add_free_positive_radius(2.0).unwrap();
        let fixed_radius = circles.add_fixed_positive_radius(4.0).unwrap();
        let first = circles.add_circle(first_center, free_radius);
        let second = circles.add_circle(second_center, fixed_radius);
        circles.add_constraint(Relation::Symmetry {
            first: SketchCurve::Circle(first),
            second: SketchCurve::Circle(second),
            axis,
            branch: SymmetryBranch::Centers,
        });
        let settled = circles.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied);
        assert!(
            matches!(settled.solution.parameter(free_radius), Some(ParameterValue::Radius(value)) if (value - 4.0).abs() < SATISFIED_RESIDUAL)
        );
        assert!(
            matches!(settled.solution.parameter(fixed_radius), Some(ParameterValue::Radius(value)) if value.to_bits() == 4.0_f64.to_bits())
        );

        // An arc has no scalar of its own left to read or write (ADR 0038) — its radius is
        // wherever its three points stand. So symmetry equalizes two arcs by MOVING them, and the
        // one that starts the wrong size ends up the size of its partner.
        let mut arcs = ProblemBuilder::new();
        let low = arcs.add_point([0.0, -5.0]);
        let high = arcs.add_point([0.0, 5.0]);
        let axis = arcs.add_segment(low, high);
        let a0 = arcs.add_point([-8.0, 0.0]);
        let a1 = arcs.add_point([-2.0, 0.0]);
        let ac = arcs.add_point([-5.0, 0.0]);
        let b0 = arcs.add_point([3.0, 0.0]);
        let b1 = arcs.add_point([7.0, 0.0]);
        let bc = arcs.add_point([5.0, 0.0]);
        let first = arcs.add_arc(ac, a0, a1);
        let second = arcs.add_arc(bc, b0, b1);
        arcs.add_constraint(Relation::Symmetry {
            first: SketchCurve::Arc(first),
            second: SketchCurve::Arc(second),
            axis,
            branch: SymmetryBranch::Direct,
        });
        let settled = arcs.finish().unwrap().settle();
        assert!(settled.diagnostics.satisfied);
        let reach = |center: PointId, end: PointId| {
            let center = settled.solution.position(center).unwrap();
            let end = settled.solution.position(end).unwrap();
            (center[0] - end[0]).hypot(center[1] - end[1])
        };
        assert!((reach(ac, a0) - reach(bc, b0)).abs() < SATISFIED_RESIDUAL);
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

fn scalar_coordinates_of_solution(problem: &Problem, solution: &Solution) -> Option<Vec<f64>> {
    (solution.owner == problem.owner && solution.parameters.len() == problem.parameters.len()).then(
        || {
            problem
                .parameters
                .iter()
                .copied()
                .zip(solution.parameters.iter().copied())
                .map(|(parameter, value)| {
                    let stored = match value {
                        ParameterValue::Radius(value)
                            if parameter.kind == ParameterKind::PositiveRadius =>
                        {
                            value
                        }
                        ParameterValue::Station(value)
                            if parameter.kind == ParameterKind::SplineStation =>
                        {
                            value
                        }
                        ParameterValue::Radius(_) | ParameterValue::Station(_) => f64::NAN,
                    };
                    parameter_coordinate(Parameter {
                        stored,
                        ..parameter
                    })
                })
                .collect()
        },
    )
}

#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub report: Option<SolveReport>,
    pub satisfied: bool,
    /// Numerical residuals can be small while a finite authored curve no longer contains the
    /// derived touching point. This is separate so callers never treat geometric invalidity as a
    /// solver convergence status.
    pub tangent_contacts_valid: bool,
}

/// Validation of the stored configuration, deliberately without numerical settlement.
#[derive(Debug, Clone)]
pub struct CurrentValidation {
    pub satisfied: bool,
    pub collapsed: Option<SketchCurve>,
    pub tangent_failure: Option<TangentContactFailure>,
}

/// One standing Tangent whose current solution has no valid finite contact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TangentContactFailure {
    pub constraint: ConstraintId,
    pub error: TangentContactError,
}

#[derive(Debug, Clone)]
pub struct Settled {
    pub solution: Solution,
    pub diagnostics: Diagnostics,
    /// The quantity a DRAG was pulled onto, where this settle answered one and it snapped.
    ///
    /// Not a fact about the equations — a fact about how the gesture was read, riding home with
    /// the answer because there is nowhere else for it to ride. A plain settle leaves it empty.
    pub kept: Option<KeptQuantity>,
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
        curve: SketchCurve,
        implicated: Vec<ConstraintId>,
    },
    /// The equations settled, but the persisted branch has no finite contact on both authored
    /// curve domains. This is distinct from an unsatisfied system: changing branch or extending
    /// a finite curve is a different author action from removing a fighting constraint.
    InvalidTangent {
        constraint: ConstraintId,
        error: TangentContactError,
    },
}

#[derive(Debug, Clone)]
pub enum DragOutcome {
    Accepted(Settled),
    Rejected(Settled),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestError {
    UnknownPoint,
    InvalidRelation(BuildError),
}

/// Whether the system carries the **rigidity regularizer**: one row per edge and axis asking that
/// the edge's span come out of the solve as it went in, plus one per curve whose shape is a scalar
/// rather than a span — see [`ScalarHold`].
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
    Preferred {
        anchored: &'a [PointId],
        flexible_curves: &'a [SketchCurve],
        /// Where the hand's own points stood BEFORE it moved them, and nothing else.
        ///
        /// A preference describes the shape being PRESERVED, so it has to be measured on the
        /// drawing that still has that shape. A drag writes its hands down before it solves, so by
        /// the time the preference is built the drawing is already distorted around the one point
        /// that led — and a preference read from there asks to keep the distortion.
        ///
        /// Spans survived that only by luck: a span whose end is the hand becomes its own answer,
        /// so it stops asking for anything rather than asking for the wrong thing. An arc's radius
        /// has no such luck. Measured after a center drag, its two ends disagree about how far
        /// away they are, and the average of the two is a radius NEITHER end has — an invented
        /// target the solve then rebuilds the whole arc around.
        ///
        /// Only the hands need naming. Every other point is in `opening`.
        was: &'a [(PointId, [f64; 2])],
        /// Where the whole drawing stood when the GESTURE started, which is what every preference
        /// row is written against. Empty means the drawing in front of the pass is that reference.
        ///
        /// A walked drag hands each step the drawing the last step reached, because a relation has
        /// to be linearized about the configuration in front of it. A PREFERENCE must not follow:
        /// re-aimed at each step's answer it can only ever say "stay where the last step left
        /// you", so whatever that step got slightly wrong becomes the thing being preserved. The
        /// error ratchets, and it drains into whichever quantity nothing else is pricing — on a
        /// slot, its width, which grew 20% over a nine-step sweep and grew FURTHER when the steps
        /// were made finer, which is the signature of a per-step bias rather than of curvature.
        opening: &'a [[f64; 2]],
        /// Whether the gesture is reshaping a curve rather than moving the drawing.
        ///
        /// ONE finger on a point that ENDS a curve is a vertex being reshaped, and the rest of the
        /// drawing then prefers to STAY, not merely to keep its shape. A hand on a point that only
        /// CENTERS curves — a circle's middle, a slot's hub — is the drawing being moved, and so
        /// are the several hands of a carry; travel has to stay free for those or a carry would
        /// deform instead of travelling.
        ///
        /// Spans cannot tell the two apart, because a span is translation-invariant: sliding the
        /// whole drawing under the cursor keeps every one of them exactly, so the preference is
        /// indifferent and the arithmetic settles it by spreading the correction over every
        /// coordinate it touches. This is the one bit that says which gesture it is.
        ///
        /// It is asked of the points STANDING WITH the hand, not of the hand alone — see
        /// [`Problem::standing_together`] — because the dot the author grabs is often a handle
        /// rather than the vertex the curves were drawn through.
        reshaping: bool,
    },
}

#[derive(Debug, Clone, Copy)]
struct EdgeSpan {
    from: usize,
    to: usize,
    span: [f64; 2],
}

/// One curve's SHAPE parameter held at the value it went into the solve with — a segment's span
/// row's counterpart for the curves whose shape is a scalar rather than a displacement.
///
/// This is the whole of what the rigidity preference means for an arc or a circle. A segment says
/// "keep my span"; a curve with a radius column says "keep my radius", and both let the curve
/// travel for free while pricing any change of shape. Nothing here is arc-specific: it holds every
/// free scalar the problem carries, because every one of them is some curve's shape.
///
/// The row is on the COLUMN, not on a distance computed from four coordinates. That is why it is
/// worth having the column at all — the hold is linear, exactly conditioned, and cannot fight the
/// geometry through a badly-scaled derivative. It reads in voxels, like a span row, because
/// [`parameter_coordinate`] keeps a radius in voxels; the two preferences are therefore directly
/// comparable and a drag trades one against the other at par.
#[derive(Debug, Clone, Copy)]
struct ScalarHold {
    slot: usize,
    at: f64,
}

/// What a hand in a drag's set is DOING, said by the caller rather than worked out here.
///
/// A gesture arrives with more hands than the author has fingers, and they do not all mean the
/// same thing. Told apart by their numbers alone the three are indistinguishable at the moment a
/// drag opens — every one of them is a point being asserted somewhere — and every rule that tried
/// to tell them apart afterwards needed a tolerance to do it, because a pin is exact only in exact
/// arithmetic. Commercial solvers do not guess either: D-Cubed, the solver under Fusion and
/// `SolidWorks`, takes a RIGID SET — "collections of geometries which 2D DCM solves as if they are
/// constrained relative to each other" — and `SolveSpace`'s interface names dragging a point and
/// dragging a whole entity as two gestures, not one gesture with an inferred meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandRole {
    /// The point the author has hold of. There is at most one.
    Lead,
    /// A point the lead carries: the rest of a rigid set, moving by the same motion.
    Carried,
    /// A point held STILL for the duration, which is how a reshape names what it turns about.
    Pin,
}

/// One point a drag asserts, and why.
#[derive(Debug, Clone, Copy)]
pub struct Hand {
    /// The point being asserted.
    pub point: PointId,
    /// Where the gesture puts it.
    pub to: [f64; 2],
    /// What it is doing there.
    pub role: HandRole,
}

impl Hand {
    /// The point the author has hold of, where the set names one.
    fn lead_of(hands: &[Self]) -> Option<(usize, Self)> {
        let mut found = None;
        for (index, hand) in hands.iter().enumerate() {
            if hand.role == HandRole::Lead {
                if found.is_some() {
                    return None;
                }
                found = Some((index, *hand));
            }
        }
        found
    }
}

/// The furthest a snap may carry a hand off the cursor, in the drawing's own units.
///
/// A cone that is a share of the hand's TRAVEL is already the same size on screen at every zoom:
/// travel is read from the cursor, so a hundred-pixel gesture is a hundred-pixel gesture however
/// much of the drawing that covers. What it is not is BOUNDED. A long sweep opens a wide cone, and
/// a wide cone can hold the drawing a long way from where the author is pointing — with nothing to
/// say how far, because the cone is an angle and an angle has no length. Every CAD tool that snaps
/// puts a ceiling on that and states it in screen pixels, which is the unit the author's patience
/// is actually measured in.
///
/// It is a ceiling and only a ceiling. [`SnapReach::UNBOUNDED`] is the kernel's own behaviour, so a
/// caller that sets one can only ever narrow a snap, never invent one. It is worth being clear
/// about what it does NOT do: measured over a sweep, capping the cone at a fixed length made a long
/// gesture WORSE rather than steadier, and the instability it was first reached for was cured by
/// holding the quantity instead (ADR 0043, ADR 0044). This bounds how far a snap may take the
/// drawing. It does not make the drag smooth; that is already done.
///
/// The shell owns the number because only the shell has a camera. What arrives here is a length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapReach(f64);

impl SnapReach {
    /// No ceiling: the cone is whatever the gesture opened.
    pub const UNBOUNDED: Self = Self(f64::INFINITY);

    /// A ceiling of `length` drawing units.
    ///
    /// A length that is not a positive, finite number is no ceiling at all rather than a snap that
    /// can never reach — a caller whose camera math degenerated should lose the ceiling, not the
    /// snap.
    #[must_use]
    pub fn of_length(length: f64) -> Self {
        if length.is_finite() && length > 0.0 {
            Self(length)
        } else {
            Self::UNBOUNDED
        }
    }
}

/// The quantity a drag was pulled onto, in enough detail to DRAW.
///
/// A snap is invisible from the outside: the hand goes somewhere slightly other than where the
/// cursor is, and the author is left guessing whether that was the drawing keeping a radius for
/// them or the solve failing to reach. So the drag says which quantity it kept and what it is
/// measured from, and the overlay draws the circle — the author sees the thing they are moving
/// along and can feel the difference between sticking to it and pulling off it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeptQuantity {
    /// What the quantity is measured from, in plane coordinates: an arc's center, or the far end
    /// of the segment whose length is being kept.
    pub about: [f64; 2],
    /// How far the held point stood from it. The radius of the circle the hand is sliding along.
    pub radius: f64,
    /// How far into the cone the hand is standing, as a share of it: zero on the quantity, one at
    /// the rim.
    ///
    /// It rides home so the ring can be drawn at a strength — see [`Self::ghost_ink`], which is
    /// the only thing that reads it.
    pub across_the_cone: f64,
}

impl KeptQuantity {
    /// How strongly to ink the ring: full on the quantity, nothing at the rim of the cone.
    ///
    /// **This reports how much cone is LEFT, not how hard the quantity is being held.** The two
    /// look interchangeable and are not, and drawing the hold was measured to be the wrong choice:
    ///
    /// - The hold is flat over the plateau (`SNAP_HOLD`) and spends its whole range in the
    ///   outer `0.4` of the cone, so the visible fade covered `0.3 * travel` of cursor. This spends
    ///   the whole cone, which is **two and a half times as long a fade** for the same gesture.
    /// - Ink drawn from the hold swings `3.75 / cone` per unit of cursor at the steepest point of
    ///   the falloff; ink drawn from this swings `1 / cone`. At the start of a gesture, where the
    ///   cone is a few screen points across, that difference is the ring strobing against the ring
    ///   dimming.
    /// - The hold cannot warn. It is exactly one until the hand is already 60% of the way out, so
    ///   a ring inked from it is at full strength right up to the moment it starts collapsing. This
    ///   dims from the first step off the quantity, so the ring going grey means "you are running
    ///   out of room" while there is still room.
    ///
    /// What it costs is that a ring at 40% ink may still be holding its quantity *exactly*. That is
    /// the right trade: the author cannot act on how much correction is being applied, and can act
    /// on how much room is left.
    ///
    /// Linear rather than smoothed on purpose — a constant slope is the steadiest ink there is, and
    /// smoothing it would put the peak back at `1.5 / cone`.
    #[must_use]
    pub fn ghost_ink(self) -> f64 {
        (1.0 - self.across_the_cone).clamp(0.0, 1.0)
    }
}

/// What one frame of a walked drag answered: how the solve went, and what the hand was pulled
/// onto if anything. The walk keeps the LAST frame's, because that is the one that delivered.
#[derive(Debug, Clone, Copy, Default)]
struct Frame {
    report: Option<SolveReport>,
    kept: Option<KeptQuantity>,
}

/// A hand pulled onto a quantity its own curve already had — see [`Problem::snapped`].
#[derive(Debug, Clone)]
struct Snap {
    /// What the hand was pulled onto, kept so the overlay can draw it.
    kept: KeptQuantity,
    /// The hand, moved onto the circle the quantity draws.
    hands: Vec<Hand>,
    /// How far around the point the quantity is measured from the hand is asking to go, in
    /// radians.
    turn: f64,
    /// How much of the correction the falloff let through, from one on the quantity to zero at
    /// the rim of the cone. Everything the snap does is scaled by it, so that a snap which is
    /// about to be let go is already doing nothing by the time it is.
    pull: f64,
}

/// The best quantity found so far while looking for one to hold, and how hard it pulls.
#[derive(Debug, Clone, Copy)]
struct Nearest {
    /// The point the quantity is measured from.
    about: [f64; 2],
    /// What to multiply the hand's arm by to land it on the faded quantity.
    scale: f64,
    /// The quantity itself, which is what the overlay draws whatever the pull.
    quantity: f64,
    /// How far off it the hand is, which is how candidates are ranked.
    across: f64,
    /// The same distance as a share of the cone it was measured in, which is what the ring is
    /// inked from — see [`KeptQuantity::ghost_ink`].
    across_the_cone: f64,
    /// The falloff at that distance.
    pull: f64,
    /// How far the hand actually is from `about`, which the faded turn is measured against.
    reach: f64,
}

#[derive(Debug, Clone, Copy)]
struct PointHold {
    slot: usize,
    at: [f64; 2],
}

#[derive(Debug, Clone, Copy)]
/// A segment's directional ends. Naming them avoids silently reversing a directional relation.
struct SegmentSlots {
    from: usize,
    to: usize,
}

/// A locally-resolved authored curve. This is the solver's curve seam: document ids and storage
/// stay outside, while all relation and contact mathematics sees the same three curve forms.
#[derive(Debug, Clone, Copy)]
enum ResolvedCurve {
    Segment(SegmentSlots),
    Arc(ArcCurveSlots),
    Circle(CircleSlots),
}

/// An [`AngleArm`] in slots. Both cases come down to two point slots and how to read a direction
/// from them, which is why the arc case keeps a center rather than a whole curve: the tangent at an
/// end is perpendicular to the radius standing at it, and nothing else about the arc is involved.
#[derive(Debug, Clone, Copy)]
enum ResolvedAngleArm {
    Segment(SegmentSlots),
    ArcEnd { center: usize, end: usize },
}

#[derive(Debug, Clone, Copy)]
struct ArcCurveSlots {
    center: usize,
    from: usize,
    to: usize,
}

#[derive(Debug, Clone, Copy)]
struct CircleSlots {
    center: usize,
    radius_parameter: usize,
}

#[derive(Debug, Clone, Copy)]
enum Resolved {
    Fix {
        slot: usize,
        at: [f64; 2],
    },
    Quantize {
        slot: usize,
        pitch: f64,
        phase: f64,
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
    AxisDistance {
        from: usize,
        to: usize,
        axis: usize,
        length: f64,
    },
    PointLineDistance {
        point: usize,
        line: SegmentSlots,
        distance: f64,
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
    Angle {
        first: ResolvedAngleArm,
        second: ResolvedAngleArm,
        radians: f64,
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
    PointOnCurve {
        point: usize,
        curve: ResolvedCurve,
    },
    PointOnSpline {
        point: usize,
        spline: usize,
        station: usize,
        /// How much curve one unit of `station` spends, captured from the spline the constraint
        /// was built against. See [`station_length`].
        per_unit: f64,
    },
    Radius {
        curve: ResolvedCurve,
        length: f64,
    },
    RimGap {
        first: ResolvedCurve,
        second: ResolvedCurve,
        distance: f64,
    },
    TangentDirection {
        joint: usize,
        joint_arm: usize,
        against: ResolvedCurve,
    },
    Curvature {
        joint: usize,
        joint_arm: usize,
        neighbor: usize,
        neighbor_arm: usize,
        end: SpanEnd,
        against: ResolvedCurve,
    },
    Tangent {
        first: ResolvedCurve,
        second: ResolvedCurve,
        branch: TangentBranch,
    },
    Concentric {
        first: usize,
        second: usize,
    },
    Symmetry {
        first: ResolvedCurve,
        second: ResolvedCurve,
        axis: SegmentSlots,
        branch: SymmetryBranch,
    },
}

/// The parameterized residual system, field by field.
///
/// Every point occupies two coordinate slots, not only points named by a relation: unconstrained
/// points are real degrees of freedom, and an arc's center is one of them (ADR 0038). `problem`
/// supplies the topology whose edge spans rigidity preserves; `resolved` holds every relation's
/// endpoints as slots, built once so the residual loop is arithmetic and never searches topology by
/// handle; `base` is the pre-solve whole-coordinate vector; and `free` narrows that vector to
/// mutable parameters while anchored coordinates remain in `base`.
/// The whole coordinate vector, read without being built.
///
/// Every slot a residual pass reads is one of two things: a coordinate the solver is moving, whose
/// current value is a column of the parameter vector, or one it is not, whose value has not changed
/// since the pass began and is still in `base`. Assembling those into a vector first is a copy of
/// the whole drawing, and a NARROWED pass pays that copy in full for the handful of rows it was
/// asked for — the copy does not shrink with the arithmetic. Answering the question where it is
/// asked costs one branch and no allocation at all.
#[derive(Clone, Copy)]
struct Coordinates<'a> {
    /// Where every slot stood when the pass began.
    base: &'a [f64],
    /// The solver's current values, in parameter-vector order.
    parameters: &'a [f64],
    /// Which column holds each slot's current value, or `None` where `base` still does.
    column_of_slot: &'a [Option<usize>],
}

impl<'a> Coordinates<'a> {
    /// A reading of a vector that is already whole — a solution being examined rather than a pass
    /// being searched, where nothing is moving and every slot is where it stands.
    fn of(whole: &'a [f64]) -> Self {
        Self {
            base: whole,
            parameters: &[],
            column_of_slot: &[],
        }
    }

    fn get(self, slot: usize) -> f64 {
        if let Some(Some(column)) = self.column_of_slot.get(slot) {
            return self.parameters.get(*column).copied().unwrap_or_default();
        }
        self.base.get(slot).copied().unwrap_or_default()
    }

    /// One point's place, which is the pair of slots it occupies.
    fn at(self, point: usize) -> [f64; 2] {
        [self.get(point * 2), self.get(point * 2 + 1)]
    }
}

/// Which arm of the arithmetic a residual row comes out of.
///
/// A drawing's row layout is a fact about the drawing, not about any one evaluation, so it is
/// settled once per pass and BOTH the whole evaluation and a narrowed one walk it. That is the
/// point of it. A second description of the row order, read by only one of the two, could fall
/// behind the walk without failing loudly: every later row would simply be attributed to the wrong
/// arm, and the falsifier would report the disagreement against the innocent relation.
#[derive(Clone, Copy)]
struct RowSource {
    /// The first row this arm writes. Every row of one arm carries the same value, which is how a
    /// walk tells that the arm has already been written.
    start: usize,
    arm: RowArm,
}

/// The arm itself, named rather than numbered wherever the name says anything.
#[derive(Clone, Copy)]
enum RowArm {
    /// One relation of the resolved list, every row of it at once. A symmetry's rows come out of a
    /// single call and a curvature's two share the geometry they are measured against, so this
    /// family is not separable one row at a time.
    Relation(usize),
    /// One arc's form rows: both ends against the arc's own radius column, or the two reaches
    /// equated where the arc has no radius to name.
    ArcForm(usize),
    /// One axis of one author-drawn span.
    Span { edge: usize, axis: usize },
    /// One scalar hold, which is a row on its own.
    Scalar(usize),
    /// One axis of one anchored point.
    Hold { hold: usize, axis: usize },
}

struct Residuals<'a> {
    /// The validated source of point coordinates and topology.
    problem: &'a Problem,
    /// Each relation resolved once to slots in constraint order.
    resolved: Vec<Resolved>,
    /// Every author-drawn edge span to preserve during the preference pass, absent in exact mode.
    rigidity: Vec<EdgeSpan>,
    /// Every curve's scalar shape held through that same pass. See [`ScalarHold`].
    scalars: Vec<ScalarHold>,
    /// Anchored points held at their starting place through the preference pass.
    holds: Vec<PointHold>,
    /// Whole-coordinate values at the start of this pass; anchored values remain here unchanged.
    base: Vec<f64>,
    /// Indices into `base` that the numerical solver may alter, in parameter-vector order.
    free: Vec<usize>,
    /// One entry per residual row, naming the arm that writes it. See [`RowSource`].
    rows: Vec<RowSource>,
    /// Which parameter column holds each slot's current value, or `None` where `base` still does.
    /// See [`Coordinates`], which is what reads it.
    column_of_slot: Vec<Option<usize>>,
}

/// A scalar's SOLVER COORDINATE, which for a length is the length itself.
///
/// The identity is load-bearing, not laziness. A solve picks, among the corrections that satisfy
/// its rows, the SHORTEST one — and "shortest" is measured in whatever coordinates the columns are
/// written in. That makes the transform a statement about relative cost: a coordinate whose
/// derivative is large is a coordinate the solve will spend first, because a little of it goes a
/// long way. A radius held as `ln r` has derivative `r` in every row that reads it, so a forty-
/// voxel arc's radius is forty times cheaper than moving any of its points, and the solve pays with
/// the radius every time. Rescaling one coordinate silently re-prices the whole drawing.
///
/// Held as the radius itself, a voxel of radius costs the same as a voxel of travel, which is the
/// only exchange rate a drag can be reasoned about in. `planegcs` and `SolveSpace` both carry a plain
/// radius for the same reason.
///
/// Positivity, which the logarithm used to guarantee, comes instead from the clamp in
/// [`physical_parameter_value`] and from the geometry: every row that reads a radius equates it to
/// a distance, and a distance cannot pull it below zero.
fn parameter_coordinate(parameter: Parameter) -> f64 {
    match parameter.kind {
        ParameterKind::PositiveRadius | ParameterKind::SplineStation => parameter.stored,
    }
}

fn physical_parameter_value(parameter: Parameter, coordinate: f64) -> f64 {
    // Source-owned geometry never participates in optimization. Returning it directly preserves
    // its exact resolved f64 rather than routing it through the free-value topology transform.
    if !parameter.free {
        return parameter.stored;
    }
    match parameter.kind {
        ParameterKind::PositiveRadius => {
            coordinate.clamp(min_exact_positive(), max_exact_positive())
        }
        // Deliberately unclamped. A station past the end of an open spline is a real answer —
        // the point the author asked for is off the curve — and clamping it would flatten the
        // residual's derivative to nothing exactly where the solve needs to be told so.
        ParameterKind::SplineStation => coordinate,
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

/// The unit direction an [`ResolvedAngleArm`] gives, or zero where the drawing gives none.
///
/// An arc's tangent at an end is perpendicular to the radius standing there, which is
/// [`super::curvature::direction_at`]'s rule for a circular curve written against the two slots
/// this arm kept. An end sitting on its own center has no radius and so no tangent, and answers
/// zero for the same reason a collapsed segment does — collapse validation explains the rejection,
/// the residual does not invent a direction out of noise.
fn unit_of_arm(at: &impl Fn(usize) -> [f64; 2], arm: ResolvedAngleArm) -> [f64; 2] {
    match arm {
        ResolvedAngleArm::Segment(segment) => unit_along(at, segment),
        ResolvedAngleArm::ArcEnd { center, end } => {
            let (center, end) = (at(center), at(end));
            let radius = [end[0] - center[0], end[1] - center[1]];
            let length = radius[0].hypot(radius[1]);
            if length <= f64::EPSILON {
                [0.0, 0.0]
            } else {
                [-radius[1] / length, radius[0] / length]
            }
        }
    }
}

/// The spline `shape` as it stands at these coordinates, refit from its own points.
///
/// Refit on every residual pass, and that is the point rather than the price. A spline's shape is
/// a FUNCTION of points the solve is still moving, so a stored fit would be one iteration stale
/// and a finite-difference column through a fit point would read a curve that did not move when
/// the point did — which is to say, no column at all.
fn live_spline(shape: &SplineShape, at: &impl Fn(usize) -> [f64; 2]) -> Option<SplineCandidate> {
    let places: Vec<[f64; 2]> = shape.points.iter().map(|point| at(point.index)).collect();
    match &shape.form {
        SplineForm::FitPoint { arms } => {
            let tangents: Vec<Option<[f64; 2]>> = arms
                .iter()
                .zip(&places)
                .map(|(arm, place)| {
                    // An arm sits on the cubic's own control point, a third of the derivative out,
                    // the same reading the drawing takes. One dropped on its fit point names no
                    // direction and reads as absent rather than collapsing the curve.
                    let handle = at((*arm)?.index);
                    let tangent = [(handle[0] - place[0]) * 3.0, (handle[1] - place[1]) * 3.0];
                    (tangent[0] != 0.0 || tangent[1] != 0.0).then_some(tangent)
                })
                .collect();
            fit_point_spline(&places, &tangents, shape.closed).ok()
        }
        SplineForm::ControlPoint => control_point_spline(&places).ok(),
    }
}

/// Where on `candidate` the station `along` stands, counted in PIECES rather than in length.
///
/// Off either end the end piece's own cubic is extended rather than the answer clamped: a clamp
/// would report the same place for every station past the end, and a residual that stops changing
/// is a residual that has stopped telling the solve which way it went wrong. A closed spline has
/// no end to go past, so its station wraps — the curve is genuinely periodic there, so wrapping
/// the coordinate costs the residual no smoothness.
fn spline_place(candidate: &SplineCandidate, along: f64) -> Option<[f64; 2]> {
    if !along.is_finite() {
        return None;
    }
    let count = f64::from(u32::try_from(candidate.pieces.len()).ok()?);
    if count <= 0.0 {
        return None;
    }
    let walked = if candidate.closed {
        along.rem_euclid(count)
    } else {
        along
    };
    let (mut index, mut base) = (0_usize, 0.0_f64);
    while index + 1 < candidate.pieces.len() && walked >= base + 1.0 {
        index += 1;
        base += 1.0;
    }
    let landed = candidate.pieces.get(index)?.point_at(walked - base);
    (landed[0].is_finite() && landed[1].is_finite()).then_some(landed)
}

fn curve_geometry(
    curve: ResolvedCurve,
    at: &impl Fn(usize) -> [f64; 2],
    specifications: &[Parameter],
    whole: Coordinates,
    point_count: usize,
) -> CurveGeometry {
    match curve {
        ResolvedCurve::Segment(segment) => CurveGeometry::Segment {
            from: at(segment.from),
            to: at(segment.to),
        },
        ResolvedCurve::Arc(arc) => {
            let (center, from, to) = (at(arc.center), at(arc.from), at(arc.to));
            let reach = |end: [f64; 2]| (end[0] - center[0]).hypot(end[1] - center[1]);
            CurveGeometry::Circular(CircularCurve {
                center,
                // Read off BOTH ends, because which end is called `from` is a label.
                //
                // Equal radius is a ROW the solve is answering, not a property the drawing
                // arrives holding: mid-pass the two ends stand different distances from the
                // center, and taking the first one names a different circle than taking the
                // second. Nothing says which, so a tangency against this arc quietly changed
                // its equations whenever the arc's endpoint order changed — and endpoint order
                // is exactly what a WIND reverses (ADR 0041). Measured on the author's own arc
                // slot: the same hands, the same seven points, the same five relations, and one
                // route converged in two iterations at 5.1e-12 while the relabelled route
                // exhausted a hundred at 14.1, having lost two rows of rank on the way. The
                // caller then read that rejection as a broken tangency and refused the frame, so
                // a sweep across the crossing dropped its snap for thirteen frames together.
                //
                // The mean is the reading the rest of the kernel already uses —
                // [`Problem::arc_radius_seed`] — and it is exact wherever the arc is consistent,
                // which every answer the solve accepts is.
                radius: f64::midpoint(reach(from), reach(to)),
                // The DOMAIN is order-dependent on purpose: an arc is the counter-clockwise
                // sweep from `from` to `to`, so reversing it names the complementary sweep. Only
                // the circle underneath has to be a label away from itself.
                arc: Some(ArcDomain {
                    from,
                    to,
                    sweep_radians: counter_clockwise_sweep(center, from, to),
                }),
            })
        }
        ResolvedCurve::Circle(circle) => CurveGeometry::Circular(CircularCurve {
            center: at(circle.center),
            radius: physical_parameter_value(
                specifications[circle.radius_parameter],
                whole.get(point_count * 2 + circle.radius_parameter),
            ),
            arc: None,
        }),
    }
}

/// The counter-clockwise turn from `from` to `to` about `center`, in radians, within `(0, 2π]`.
///
/// An arc has no stored sweep and no stored direction (ADR 0038): the endpoint ORDER is the
/// direction, so this is the whole of what "how far does it turn" means. A degenerate
/// configuration — an end sitting on the center, or the two ends at one angle — reports a full
/// turn rather than a negative or wrapped value; only a non-finite input answers zero.
///
/// **This JUMPS by a whole turn as the head crosses the tail, and it has to.** A hair short of
/// closing is a hair short of `2π`; a hair past is a hair past zero. That is not a rounding
/// artifact to be smoothed away, it is the two arcs actually being different — a sliver and a
/// curve that goes nearly the whole way round share their endpoints and share nothing else. The
/// range is half-open at the closed configuration for the same reason: `2π` is the left limit, and
/// something has to be reported there.
///
/// The one consumer that could feel it is `Relation::Symmetry` on a pair of arcs, which subtracts
/// two of these readings. It is safe because **a symmetric pair crosses together**: the endpoints
/// are held reflected, so both arcs close in the same frame and the difference never sees the jump.
/// Measured in `a_symmetric_arc_pair_crosses_a_whole_turn_without_a_jump`, where the closing frame
/// costs 0.272 against a walk whose steps run 0.13 to 0.29. Wrapping the difference into `(-π, π]`
/// would make that row continuous and WRONG — it would call the sliver equal to the near-circle.
fn counter_clockwise_sweep(center: [f64; 2], from: [f64; 2], to: [f64; 2]) -> f64 {
    let start = (from[1] - center[1]).atan2(from[0] - center[0]);
    let end = (to[1] - center[1]).atan2(to[0] - center[0]);
    let turn = end - start;
    if turn.is_finite() && turn > 0.0 {
        turn
    } else if turn.is_finite() {
        turn + std::f64::consts::TAU
    } else {
        0.0
    }
}

/// The two whole-coordinate slots a point occupies.
///
/// Every helper below answers in slots of the WIDENED vector, not in parameter columns. The two
/// spaces are different — an anchored coordinate has a slot and no column — and mixing them is the
/// one way a reads-set can be wrong without looking wrong. `Residuals::reads_by_row` does the
/// single translation, at the end, once.
fn point_slots(point: usize) -> [usize; 2] {
    [
        point.saturating_mul(2),
        point.saturating_mul(2).saturating_add(1),
    ]
}

/// Both ends of a segment, which is what every reader of a segment's direction or length touches.
fn segment_slots(segment: SegmentSlots) -> Vec<usize> {
    let mut slots = point_slots(segment.from).to_vec();
    slots.extend(point_slots(segment.to));
    slots
}

/// Everything [`curve_geometry`] reads for one curve.
fn curve_slots(curve: ResolvedCurve, point_count: usize) -> Vec<usize> {
    match curve {
        ResolvedCurve::Segment(segment) => segment_slots(segment),
        ResolvedCurve::Arc(arc) => {
            let mut slots = point_slots(arc.center).to_vec();
            slots.extend(point_slots(arc.from));
            slots.extend(point_slots(arc.to));
            slots
        }
        ResolvedCurve::Circle(circle) => {
            let mut slots = point_slots(circle.center).to_vec();
            slots.push(
                point_count
                    .saturating_mul(2)
                    .saturating_add(circle.radius_parameter),
            );
            slots
        }
    }
}

/// Everything [`unit_of_arm`] reads for one arm of a stated angle.
fn arm_slots(arm: ResolvedAngleArm) -> Vec<usize> {
    match arm {
        ResolvedAngleArm::Segment(segment) => segment_slots(segment),
        ResolvedAngleArm::ArcEnd { center, end } => {
            let mut slots = point_slots(center).to_vec();
            slots.extend(point_slots(end));
            slots
        }
    }
}

/// Every point a spline's shape is a function of — its own points and any steering arms.
///
/// The WHOLE spline, for every row that reads any of it. A spline is refit from its points on
/// every residual pass, so moving any one of them moves the curve everywhere, and a station row
/// two pieces away is not independent of it. See [`live_spline`].
fn spline_slots(shape: &SplineShape) -> Vec<usize> {
    let mut slots: Vec<usize> = shape
        .points
        .iter()
        .flat_map(|point| point_slots(point.index))
        .collect();
    if let SplineForm::FitPoint { arms } = &shape.form {
        slots.extend(arms.iter().flatten().flat_map(|arm| point_slots(arm.index)));
    }
    slots
}

/// The slots each of one relation's residual rows reads, appended in the order
/// [`Residuals::residuals`] writes them.
///
/// **This match is the reads-set, and it must agree with the arithmetic row for row.** Declaring
/// too MUCH is safe and costs only a Curtis-Powell-Reid group; declaring too little is a Jacobian
/// that is quietly wrong, because two parameters one row reads would then be perturbed together
/// and their effects summed into one difference. It is exhaustive over [`Resolved`] so a new
/// relation cannot reach the grouped Jacobian without someone stating what its rows touch, and
/// `every_relation_declares_what_its_rows_read` is what checks the statement is true.
#[allow(clippy::too_many_lines)]
fn push_relation_slots(
    constraint: &ConstraintEntry,
    point_count: usize,
    splines: &[SplineShape],
    rows: &mut Vec<Vec<usize>>,
) {
    let scalar_slot = |parameter: usize| point_count.saturating_mul(2).saturating_add(parameter);
    match constraint.resolved {
        Resolved::Fix { slot, .. } | Resolved::Quantize { slot, .. } => {
            let [across, up] = point_slots(slot);
            rows.push(vec![across]);
            rows.push(vec![up]);
        }
        Resolved::SameCoordinate { from, to, axis }
        | Resolved::AxisDistance { from, to, axis, .. } => rows.push(vec![
            from.saturating_mul(2).saturating_add(axis),
            to.saturating_mul(2).saturating_add(axis),
        ]),
        Resolved::Distance { from, to, .. } => {
            let mut slots = point_slots(from).to_vec();
            slots.extend(point_slots(to));
            rows.push(slots);
        }
        Resolved::PointLineDistance { point, line, .. } => {
            let mut slots = point_slots(point).to_vec();
            slots.extend(segment_slots(line));
            rows.push(slots);
        }
        Resolved::Coincident { first, second } | Resolved::Concentric { first, second } => {
            let ([first_across, first_up], [second_across, second_up]) =
                (point_slots(first), point_slots(second));
            rows.push(vec![first_across, second_across]);
            rows.push(vec![first_up, second_up]);
        }
        Resolved::Parallel { first, second }
        | Resolved::Perpendicular { first, second }
        | Resolved::Equal { first, second } => {
            let mut slots = segment_slots(first);
            slots.extend(segment_slots(second));
            rows.push(slots);
        }
        Resolved::Angle { first, second, .. } => {
            let mut slots = arm_slots(first);
            slots.extend(arm_slots(second));
            rows.push(slots);
        }
        Resolved::Midpoint { point, segment } => {
            let ([across, up], [tail_across, tail_up], [head_across, head_up]) = (
                point_slots(point),
                point_slots(segment.from),
                point_slots(segment.to),
            );
            rows.push(vec![across, tail_across, head_across]);
            rows.push(vec![up, tail_up, head_up]);
        }
        Resolved::Collinear { datum, other } => {
            for end in [other.from, other.to] {
                let mut slots = segment_slots(datum);
                slots.extend(point_slots(end));
                rows.push(slots);
            }
        }
        Resolved::PointOnCurve { point, curve } => {
            let mut slots = point_slots(point).to_vec();
            slots.extend(curve_slots(curve, point_count));
            rows.push(slots);
        }
        Resolved::PointOnSpline {
            point,
            spline,
            station,
            ..
        } => {
            let mut slots = point_slots(point).to_vec();
            slots.push(scalar_slot(station));
            if let Some(shape) = splines.get(spline) {
                slots.extend(spline_slots(shape));
            }
            rows.push(slots.clone());
            rows.push(slots);
        }
        Resolved::Radius { curve, .. } => rows.push(curve_slots(curve, point_count)),
        Resolved::RimGap { first, second, .. } | Resolved::Tangent { first, second, .. } => {
            let mut slots = curve_slots(first, point_count);
            slots.extend(curve_slots(second, point_count));
            rows.push(slots);
        }
        Resolved::TangentDirection {
            joint,
            joint_arm,
            against,
        } => {
            let mut slots = point_slots(joint).to_vec();
            slots.extend(point_slots(joint_arm));
            slots.extend(curve_slots(against, point_count));
            rows.push(slots);
        }
        Resolved::Curvature {
            joint,
            joint_arm,
            neighbor,
            neighbor_arm,
            against,
            ..
        } => {
            let mut direction = point_slots(joint).to_vec();
            direction.extend(point_slots(joint_arm));
            direction.extend(curve_slots(against, point_count));
            let mut bend = direction.clone();
            bend.extend(point_slots(neighbor));
            bend.extend(point_slots(neighbor_arm));
            rows.push(direction);
            rows.push(bend);
        }
        Resolved::Symmetry {
            first,
            second,
            axis,
            ..
        } => {
            let mut slots = curve_slots(first, point_count);
            slots.extend(curve_slots(second, point_count));
            slots.extend(segment_slots(axis));
            for _ in 0..constraint.relation.residual_count() {
                rows.push(slots.clone());
            }
        }
    }
}

fn tangent_branch_matches_types(
    first: ResolvedCurve,
    second: ResolvedCurve,
    branch: TangentBranch,
) -> bool {
    let class = |curve| match curve {
        ResolvedCurve::Segment(_) => 0_u8,
        ResolvedCurve::Arc(_) | ResolvedCurve::Circle(_) => 1_u8,
    };
    match branch {
        TangentBranch::Line(_) => matches!((class(first), class(second)), (0, 1) | (1, 0)),
        TangentBranch::External | TangentBranch::Internal { .. } => {
            class(first) == 1 && class(second) == 1
        }
    }
}

impl<'a> Residuals<'a> {
    /// Build the parameter layout. Anchored coordinates stay at their pre-solve values and are
    /// omitted from the parameter vector; free coordinates are widened before every geometry read.
    /// This narrowing/widening boundary keeps residual code in one whole-coordinate space while
    /// the numerical substrate sees only actual unknowns.
    #[allow(clippy::too_many_lines)]
    fn new(problem: &'a Problem, scalar_coordinates: &[f64], rigidity: Rigidity) -> Option<Self> {
        if scalar_coordinates.len() != problem.parameters.len() {
            return None;
        }
        if problem.points.is_empty() {
            return None;
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
        let (rigidity, scalars, holds, mut free) = match rigidity {
            Rigidity::Ignored => (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                (0..point_coordinates).collect::<Vec<_>>(),
            ),
            Rigidity::Preferred {
                anchored,
                flexible_curves,
                was,
                opening,
                reshaping,
            } => {
                let at = |slot: usize| [base[slot * 2], base[slot * 2 + 1]];
                // The drawing as it stood before the gesture — see
                // [`Rigidity::Preferred::opening`]. Measured for the spans as well as the stays:
                // aiming only the stays at the opening and leaving the spans to follow the walk
                // was tried, and it was worse at both things at once — the hub drifted further AND
                // the slot fattened more — because a span that follows the walk is a span that
                // ratifies whatever the last step did to the shape.
                let shape = |slot: usize| opening.get(slot).copied().unwrap_or_else(|| at(slot));
                let flexible_segment = |index| {
                    flexible_curves.iter().any(
                        |curve| matches!(curve, SketchCurve::Segment(segment) if segment.index == index),
                    )
                };
                let flexible_arc = |index| {
                    flexible_curves
                        .iter()
                        .any(|curve| matches!(curve, SketchCurve::Arc(arc) if arc.index == index))
                };
                let spans = problem
                    .segments
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| !flexible_segment(*index))
                    .map(|(_, segment)| (segment.from.index, segment.to.index))
                    .chain(
                        problem
                            .arc_centers
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| !flexible_arc(*index))
                            .map(|(_, arc)| (arc.from.index, arc.to.index)),
                    )
                    .map(|(from, to)| {
                        let (tail, head) = (shape(from), shape(to));
                        EdgeSpan {
                            from,
                            to,
                            span: [head[0] - tail[0], head[1] - tail[1]],
                        }
                    })
                    .collect();
                // Every free scalar is some curve's shape, so every one is held. See
                // [`ScalarHold`] for why this is the arc's and the circle's span row.
                //
                // An arc's radius is measured off the drawing rather than read off the column,
                // because the column was seeded after the hand landed. A flexible arc is skipped
                // exactly as a flexible segment's span is: that curve is the one being reshaped.
                // `was` names the hands. Membership is the question every rule below asks:
                // what the hand has hold of is what the gesture is authoring, and a preference
                // must not price the very thing the author is setting.
                let in_hand = |point: PointId| was.iter().any(|(held, _)| *held == point);
                let mut scalars: Vec<ScalarHold> = Vec::new();
                let mut spoken_for: Vec<usize> = Vec::new();
                for (index, arc) in problem.arc_centers.iter().enumerate() {
                    let Some(radius) = arc.radius else { continue };
                    let Some(slot) = point_coordinates.checked_add(radius.index) else {
                        continue;
                    };
                    spoken_for.push(slot);
                    // NOT `flexible_arc`. Loosening is about the chord, and an arc grabbed by
                    // one end keeps its radius. Only a gesture holding the arc ENTIRE is authoring
                    // that radius, and then the hands say what it becomes and a hold could only
                    // fight them. Where the hands CARRY the arc instead, the radius is not priced
                    // here either — a row in this sum is traded against every other row in it, and
                    // measured on the slot that prompted all this it lost outright: held as a
                    // preference the cap still went 7.33 to 0.05, within a thousandth of holding
                    // nothing at all. It is held by `quantities_a_carry_holds_still` instead, which
                    // takes the column away for the hand's passes rather than bidding for it.
                    let _ = index;
                    if [arc.center, arc.from, arc.to].iter().copied().all(in_hand) {
                        continue;
                    }
                    // CONCENTRIC arcs are one rail family, and the gap between them is a width —
                    // exactly the freedom a slot keeps on purpose. Holding each rail's radius
                    // holds that gap by the back door, so widening one rail has to buy its way
                    // past every other, and the drawing settles by splitting the difference
                    // instead of moving the rail it was told to move. Measured on a curved slot
                    // pulled two voxels out, the far rail crept 0.29 in; unheld, it stays put and
                    // the width answers the pull exactly. A lone arc — a slot's cap — keeps its
                    // hold, which is what makes a straight slot widen by just what was asked.
                    // ...but only when the gesture has hold of a whole rail. One end in hand is a
                    // corner being pulled, and then the family's radii are the shape being kept;
                    // both ends is the rail itself being moved, and then they are the shape being
                    // authored. The same arc, told apart by how much of it the hand has.
                    let family_is_in_hand = problem.arc_centers.iter().any(|other| {
                        other.center == arc.center && in_hand(other.from) && in_hand(other.to)
                    });
                    let concentric = problem
                        .arc_centers
                        .iter()
                        .filter(|other| other.center == arc.center)
                        .count()
                        > 1;
                    if concentric && family_is_in_hand {
                        continue;
                    }
                    let hub = shape(arc.center.index);
                    let reach = |end: [f64; 2]| (end[0] - hub[0]).hypot(end[1] - hub[1]);
                    let at =
                        f64::midpoint(reach(shape(arc.from.index)), reach(shape(arc.to.index)));
                    if at.is_finite() && at > 0.0 {
                        scalars.push(ScalarHold { slot, at });
                    }
                }
                // A circle's radius is authored rather than derived from points a hand can move,
                // so its own coordinate is already the value to keep.
                for (index, parameter) in problem.parameters.iter().enumerate() {
                    let Some(slot) = point_coordinates.checked_add(index) else {
                        continue;
                    };
                    if !parameter.free || spoken_for.contains(&slot) {
                        continue;
                    }
                    if let Some(stood) = base.get(slot) {
                        scalars.push(ScalarHold { slot, at: *stood });
                    }
                }
                let held: Vec<_> = anchored.iter().map(|point| point.index).collect();
                // An anchored point is removed from the free set outright, so nothing has to hold
                // it with a row, and the holds below skip it for that reason.
                //
                // The rest of the drawing stays where it is while a vertex is reshaped.
                // Measured on a curved slot, a corner pulled six voxels took the far end 3.6 along
                // with it; holding the points the hand does not have makes travel cost what it is
                // worth, and the cheapest answer left is to sweep the corner around a drawing that
                // stays.
                //
                // This is the objective a commercial parametric drag solves — every point weighted
                // toward where it stood, the cursor weighted above them — with the span rows added
                // on top. It seeds only: pass two hands the cursor full authority regardless.
                //
                // A hand's own twins go unheld with it. A handle and the vertex it stands on are
                // one place, so holding one of them still while the other follows the cursor is
                // asking the coincidence to break.
                let hand_and_twins: Vec<PointId> = was
                    .iter()
                    .flat_map(|(held, _)| problem.standing_together(*held))
                    .collect();
                let stays: Vec<PointHold> = if reshaping {
                    (0..problem.points.len())
                        .filter(|slot| !held.contains(slot))
                        .filter(|slot| !hand_and_twins.iter().any(|point| point.index == *slot))
                        .map(|slot| PointHold {
                            slot,
                            at: shape(slot),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                (
                    spans,
                    scalars,
                    stays,
                    (0..point_coordinates)
                        .filter(|index| !held.contains(&(index / 2)))
                        .collect::<Vec<_>>(),
                )
            }
        };
        free.extend(free_scalars());
        let rows = Self::row_sources(problem, &rigidity, &scalars, &holds);
        let mut column_of_slot = vec![None; base.len()];
        for (column, slot) in free.iter().enumerate() {
            if let Some(entry) = column_of_slot.get_mut(*slot) {
                *entry = Some(column);
            }
        }
        Some(Self {
            problem,
            resolved,
            rigidity,
            scalars,
            holds,
            base,
            free,
            rows,
            column_of_slot,
        })
    }

    /// Which arm writes which row, settled once for this pass.
    ///
    /// The strides come from [`Relation::residual_count`], which is the same answer
    /// [`ResidualSystem::residual_count`] sums the vector's length out of, so the table and the
    /// length cannot disagree about the layout — `the_row_table_names_every_row_exactly_once`
    /// holds that over every shape the tests can build.
    fn row_sources(
        problem: &Problem,
        rigidity: &[EdgeSpan],
        scalars: &[ScalarHold],
        holds: &[PointHold],
    ) -> Vec<RowSource> {
        let mut rows: Vec<RowSource> = Vec::new();
        for (index, constraint) in problem.constraints.iter().enumerate() {
            let start = rows.len();
            for _ in 0..constraint.relation.residual_count() {
                rows.push(RowSource {
                    start,
                    arm: RowArm::Relation(index),
                });
            }
        }
        for (index, arc) in problem.arc_centers.iter().enumerate() {
            let start = rows.len();
            for _ in 0..if arc.radius.is_some() { 2 } else { 1 } {
                rows.push(RowSource {
                    start,
                    arm: RowArm::ArcForm(index),
                });
            }
        }
        for edge in 0..rigidity.len() {
            for axis in 0..2 {
                rows.push(RowSource {
                    start: rows.len(),
                    arm: RowArm::Span { edge, axis },
                });
            }
        }
        for index in 0..scalars.len() {
            rows.push(RowSource {
                start: rows.len(),
                arm: RowArm::Scalar(index),
            });
        }
        for hold in 0..holds.len() {
            for axis in 0..2 {
                rows.push(RowSource {
                    start: rows.len(),
                    arm: RowArm::Hold { hold, axis },
                });
            }
        }
        rows
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
    /// A scalar parameter's PHYSICAL value out of a widened coordinate vector. The transform is
    /// the same one every other reader goes through, so a finite-difference Jacobian sees the
    /// dependency by the identical route.
    fn scalar(&self, parameter: ParameterId, whole: Coordinates) -> f64 {
        let Some(specification) = self.problem.parameters.get(parameter.index) else {
            return 0.0;
        };
        let slot = self
            .problem
            .points
            .len()
            .saturating_mul(2)
            .saturating_add(parameter.index);
        physical_parameter_value(*specification, whole.get(slot))
    }

    /// The whole-coordinate slots every residual row reads, in residual order.
    ///
    /// Row for row with [`Residuals::residuals`], including the rows no relation asked for: the
    /// arc's own radius rows, the rigidity spans, the scalar holds and the point stays. Those are
    /// where the grouping is won — a span row reads one axis of two points and nothing else, so
    /// the drawing's `x` columns and its `y` columns fall apart into different groups.
    fn slots_by_row(&self) -> Vec<Vec<usize>> {
        let point_count = self.problem.points.len();
        let scalar_slot =
            |parameter: usize| point_count.saturating_mul(2).saturating_add(parameter);
        let mut rows: Vec<Vec<usize>> = Vec::with_capacity(self.residual_count());
        for constraint in &self.problem.constraints {
            push_relation_slots(constraint, point_count, &self.problem.splines, &mut rows);
        }
        for arc in &self.problem.arc_centers {
            let center = point_slots(arc.center.index);
            if let Some(radius) = arc.radius {
                for end in [arc.from.index, arc.to.index] {
                    let mut slots = center.to_vec();
                    slots.extend(point_slots(end));
                    slots.push(scalar_slot(radius.index));
                    rows.push(slots);
                }
            } else {
                let mut slots = center.to_vec();
                slots.extend(point_slots(arc.from.index));
                slots.extend(point_slots(arc.to.index));
                rows.push(slots);
            }
        }
        for edge in &self.rigidity {
            let ([tail_across, tail_up], [head_across, head_up]) =
                (point_slots(edge.from), point_slots(edge.to));
            rows.push(vec![tail_across, head_across]);
            rows.push(vec![tail_up, head_up]);
        }
        for hold in &self.scalars {
            rows.push(vec![hold.slot]);
        }
        for hold in &self.holds {
            let [across, up] = point_slots(hold.slot);
            rows.push(vec![across]);
            rows.push(vec![up]);
        }
        rows
    }

    /// The same rows in PARAMETER columns: the slots an anchored coordinate occupies drop out,
    /// because the solve never moves them and no finite difference is ever taken along them.
    fn reads_by_row(&self) -> Vec<Vec<usize>> {
        let column_of_slot = &self.column_of_slot;
        self.slots_by_row()
            .into_iter()
            .map(|slots| {
                slots
                    .into_iter()
                    .filter_map(|slot| column_of_slot.get(slot).copied().flatten())
                    .collect()
            })
            .collect()
    }

    /// One arm of the arithmetic, at the rows the table says are its own.
    #[allow(clippy::too_many_lines)]
    fn write_arm(&self, source: RowSource, whole: Coordinates, into: &mut [f64]) {
        let start = source.start;
        let at = |slot: usize| whole.at(slot);
        match source.arm {
            RowArm::Relation(index) => {
                let Some(relation) = self.resolved.get(index) else {
                    return;
                };
                match *relation {
                    Resolved::Fix { slot, at: target } => {
                        let here = at(slot);
                        into[start] = here[0] - target[0];
                        into[start + 1] = here[1] - target[1];
                    }
                    Resolved::Quantize { slot, pitch, phase } => {
                        // The lattice branch is chosen from this pass's immutable starting point,
                        // never from the optimizer's moving iterate. The preferred and exact passes
                        // therefore form a bounded integer outer loop without a discontinuous
                        // residual inside either continuous solve.
                        let stood = [self.base[slot * 2], self.base[slot * 2 + 1]];
                        let target =
                            stood.map(|value| phase + ((value - phase) / pitch).round() * pitch);
                        let here = at(slot);
                        into[start] = here[0] - target[0];
                        into[start + 1] = here[1] - target[1];
                    }
                    Resolved::SameCoordinate { from, to, axis } => {
                        into[start] = at(to)[axis] - at(from)[axis];
                    }
                    Resolved::Distance { from, to, length } => {
                        let (tail, head) = (at(from), at(to));
                        into[start] = ((head[0] - tail[0]).powi(2) + (head[1] - tail[1]).powi(2))
                            .sqrt()
                            - length;
                    }
                    Resolved::AxisDistance {
                        from,
                        to,
                        axis,
                        length,
                    } => {
                        // The absolute value is what makes this the one-axis analogue of the distance
                        // row above: both state a span, and neither carries a direction the author
                        // did not give. Its slope is a sign, so the row is as well conditioned as a
                        // row gets, and it reads the same on a long segment and a short one.
                        into[start] = (at(to)[axis] - at(from)[axis]).abs() - length;
                    }
                    // One pair of rows for both: now that a center is a placed point (ADR 0038),
                    // "these two arcs turn about the same spot" IS "these two points coincide". The
                    // relations stay distinct so the author's word for what they asked survives into
                    // diagnostics, but there is nothing different to solve.
                    Resolved::Coincident { first, second }
                    | Resolved::Concentric { first, second } => {
                        let (a, b) = (at(first), at(second));
                        into[start] = a[0] - b[0];
                        into[start + 1] = a[1] - b[1];
                    }
                    Resolved::PointLineDistance {
                        point,
                        line,
                        distance,
                    } => {
                        // How far ACROSS the line the point stands: the component of its offset from
                        // the line's tail along the line's own normal, which is the cross product with
                        // the unit direction. Measured against the infinite line, so it is unchanged by
                        // where along the line the tail happens to sit.
                        let unit = unit_along(&at, line);
                        let (stood, tail) = (at(point), at(line.from));
                        let across =
                            unit[0] * (stood[1] - tail[1]) - unit[1] * (stood[0] - tail[0]);
                        into[start] = across.abs() - distance;
                    }
                    Resolved::Parallel { first, second } => {
                        // Cross(unit directions) is sine(angle): this remains scale independent.
                        let (a, b) = (unit_along(&at, first), unit_along(&at, second));
                        into[start] = a[0] * b[1] - a[1] * b[0];
                    }
                    Resolved::Perpendicular { first, second } => {
                        // Dot(unit directions) is cosine(angle), also independent of segment length.
                        let (a, b) = (unit_along(&at, first), unit_along(&at, second));
                        into[start] = a[0] * b[0] + a[1] * b[1];
                    }
                    Resolved::Angle {
                        first,
                        second,
                        radians,
                    } => {
                        // sin(turn - asked), expanded so no arctangent has to pick a branch: the two
                        // pieces are Parallel's row and Perpendicular's row, mixed by the stated angle.
                        let (a, b) = (unit_of_arm(&at, first), unit_of_arm(&at, second));
                        let across = a[0] * b[1] - a[1] * b[0];
                        let along = a[0] * b[0] + a[1] * b[1];
                        into[start] = across * radians.cos() - along * radians.sin();
                    }
                    Resolved::Equal { first, second } => {
                        into[start] = length_of(&at, first) - length_of(&at, second);
                    }
                    Resolved::Midpoint { point, segment } => {
                        // Both coordinates are constrained because halfway names one exact place.
                        let (p, a, b) = (at(point), at(segment.from), at(segment.to));
                        into[start] = p[0] - (a[0] + b[0]) / 2.0;
                        into[start + 1] = p[1] - (a[1] + b[1]) / 2.0;
                    }
                    Resolved::Collinear { datum, other } => {
                        // Two distances to the datum line state both parallelism and zero offset.
                        let along = unit_along(&at, datum);
                        let normal = [-along[1], along[0]];
                        let anchor = at(datum.from);
                        for (offset, end) in [other.from, other.to].into_iter().enumerate() {
                            let here = at(end);
                            into[start + offset] = (here[0] - anchor[0]) * normal[0]
                                + (here[1] - anchor[1]) * normal[1];
                        }
                    }
                    Resolved::PointOnCurve { point, curve } => {
                        let here = at(point);
                        into[start] = match curve_geometry(
                            curve,
                            &at,
                            &self.problem.parameters,
                            whole,
                            self.problem.points.len(),
                        ) {
                            // The signed distance to the line, which is zero on either side of it and
                            // has no kink at the ends the way a distance to the finite piece would.
                            CurveGeometry::Segment { from, to } => {
                                let span = [to[0] - from[0], to[1] - from[1]];
                                let length = span[0].hypot(span[1]);
                                if length <= f64::EPSILON {
                                    0.0
                                } else {
                                    (here[1] - from[1])
                                        .mul_add(span[0], -((here[0] - from[0]) * span[1]))
                                        / length
                                }
                            }
                            CurveGeometry::Circular(support) => {
                                (here[0] - support.center[0]).hypot(here[1] - support.center[1])
                                    - support.radius
                            }
                        };
                    }
                    Resolved::PointOnSpline {
                        point,
                        spline,
                        station,
                        per_unit,
                    } => {
                        let here = at(point);
                        let along = physical_parameter_value(
                            self.problem.parameters[station],
                            whole.get(self.problem.points.len() * 2 + station),
                        ) / per_unit;
                        // A spline that cannot be fit at these coordinates says nothing, rather than
                        // pulling the point onto a curve nobody can draw.
                        let landed = self
                            .problem
                            .splines
                            .get(spline)
                            .and_then(|shape| live_spline(shape, &at))
                            .and_then(|candidate| spline_place(&candidate, along));
                        let (across, up) =
                            landed.map_or((0.0, 0.0), |on| (here[0] - on[0], here[1] - on[1]));
                        into[start] = across;
                        into[start + 1] = up;
                    }
                    Resolved::Radius { curve, length } => {
                        into[start] = match curve_geometry(
                            curve,
                            &at,
                            &self.problem.parameters,
                            whole,
                            self.problem.points.len(),
                        ) {
                            CurveGeometry::Circular(support) => support.radius - length,
                            // Unreachable: the document will not build a radius against a straight
                            // curve. Zero rather than a panic, so a hand-built problem misbehaves
                            // by saying nothing instead of by falling over.
                            CurveGeometry::Segment { .. } => 0.0,
                        };
                    }
                    Resolved::RimGap {
                        first,
                        second,
                        distance,
                    } => {
                        let radius = |which| {
                            match curve_geometry(
                                which,
                                &at,
                                &self.problem.parameters,
                                whole,
                                self.problem.points.len(),
                            ) {
                                CurveGeometry::Circular(support) => Some(support.radius),
                                // Unreachable: the document will not build a rim gap against a
                                // straight curve. `None` rather than a panic, so a hand-built problem
                                // misbehaves by saying nothing instead of by falling over.
                                CurveGeometry::Segment { .. } => None,
                            }
                        };
                        into[start] = match (radius(first), radius(second)) {
                            (Some(inner), Some(outer)) => (outer - inner).abs() - distance,
                            _ => 0.0,
                        };
                    }
                    Resolved::Tangent {
                        first,
                        second,
                        branch,
                    } => {
                        into[start] = tangent_residual(
                            curve_geometry(
                                first,
                                &at,
                                &self.problem.parameters,
                                whole,
                                self.problem.points.len(),
                            ),
                            curve_geometry(
                                second,
                                &at,
                                &self.problem.parameters,
                                whole,
                                self.problem.points.len(),
                            ),
                            branch,
                        );
                    }
                    Resolved::TangentDirection {
                        joint,
                        joint_arm,
                        against,
                    } => {
                        into[start] = direction_residual(
                            at(joint),
                            at(joint_arm),
                            curve_geometry(
                                against,
                                &at,
                                &self.problem.parameters,
                                whole,
                                self.problem.points.len(),
                            ),
                        );
                    }
                    Resolved::Curvature {
                        joint,
                        joint_arm,
                        neighbor,
                        neighbor_arm,
                        end,
                        against,
                    } => {
                        let geometry = curve_geometry(
                            against,
                            &at,
                            &self.problem.parameters,
                            whole,
                            self.problem.points.len(),
                        );
                        into[start] = direction_residual(at(joint), at(joint_arm), geometry);
                        into[start + 1] = curvature_residual(
                            JointSpan {
                                joint: at(joint),
                                joint_arm: at(joint_arm),
                                neighbor: at(neighbor),
                                neighbor_arm: at(neighbor_arm),
                                end,
                            },
                            geometry,
                        );
                    }
                    Resolved::Symmetry {
                        first,
                        second,
                        axis,
                        branch,
                    } => {
                        let geometry = |curve| {
                            curve_geometry(
                                curve,
                                &at,
                                &self.problem.parameters,
                                whole,
                                self.problem.points.len(),
                            )
                        };
                        let _ = symmetry_residuals(
                            geometry(first),
                            geometry(second),
                            geometry(ResolvedCurve::Segment(axis)),
                            branch,
                        )
                        .write_to(&mut into[start..]);
                    }
                }
            }
            RowArm::ArcForm(index) => {
                let Some(arc) = self.problem.arc_centers.get(index) else {
                    return;
                };
                // What makes three placed points an ARC rather than three points (ADR 0038): the center
                // stands the same distance from both ends. It is a row and not a projection because the
                // motion it forbids — the center sliding along the chord — is otherwise a gauge freedom
                // the least-squares system cannot see, and a rank-deficient column is worse than a row.
                let (center, from, to) =
                    (at(arc.center.index), at(arc.from.index), at(arc.to.index));
                let reach = |end: [f64; 2]| (end[0] - center[0]).hypot(end[1] - center[1]);
                // Two rows against the arc's own radius column where it has one. Subtract them and the
                // equal-radius condition comes back exactly, so everything that reads an arc through
                // its chord bisector still agrees — but stated this way the drawing also says WHAT the
                // two ends are equidistant to, which is the quantity a preference can hold.
                if let Some(radius) = arc.radius {
                    let named = self.scalar(radius, whole);
                    into[start] = reach(from) - named;
                    into[start + 1] = reach(to) - named;
                } else {
                    into[start] = reach(from) - reach(to);
                }
            }
            RowArm::Span { edge, axis } => {
                // Per-axis spans intentionally do not leave a group free to rotate. The exact pass
                // handles any genuine disagreement between this preference and a relation.
                let Some(edge) = self.rigidity.get(edge) else {
                    return;
                };
                let (tail, head) = (at(edge.from), at(edge.to));
                into[start] = (head[axis] - tail[axis]) - edge.span[axis];
            }
            RowArm::Scalar(index) => {
                let Some(hold) = self.scalars.get(index) else {
                    return;
                };
                into[start] = whole.get(hold.slot) - hold.at;
            }
            RowArm::Hold { hold, axis } => {
                let Some(hold) = self.holds.get(hold) else {
                    return;
                };
                let here = at(hold.slot);
                into[start] = here[axis] - hold.at[axis];
            }
        }
    }

    /// The coordinates this pass reads, without building them. See [`Coordinates`].
    fn coordinates<'p>(&'p self, parameters: &'p [f64]) -> Coordinates<'p> {
        Coordinates {
            base: &self.base,
            parameters,
            column_of_slot: &self.column_of_slot,
        }
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
            + self
                .problem
                .arc_centers
                .iter()
                .map(|arc| if arc.radius.is_some() { 2 } else { 1 })
                .sum::<usize>()
            + self.rigidity.len() * 2
            + self.scalars.len()
            + self.holds.len() * 2
    }
    fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
        let whole = self.coordinates(parameters);
        // Every arm, once each, by walking the same table a narrowed pass walks. There is no second
        // description of the row order for one of them to fall behind.
        let mut written = usize::MAX;
        for source in &self.rows {
            if source.start == written {
                continue;
            }
            written = source.start;
            self.write_arm(*source, whole, into);
        }
    }

    /// Write only the rows named in `rows`, by writing the ARMS those rows come out of.
    ///
    /// An arm whose rows a group only partly asked for is written whole: the extra rows carry the
    /// values [`ResidualSystem::residuals`] would have left there, so nothing downstream can tell,
    /// and the alternative is a second, separable copy of arithmetic that a symmetry and a curvature
    /// cannot be cut into anyway. `rows` arrives ascending, and an arm's rows are contiguous, so
    /// comparing against the arm last written is enough to write each one once.
    fn residuals_of_rows(&self, parameters: &[f64], rows: &[usize], into: &mut [f64]) {
        let whole = self.coordinates(parameters);
        let mut written = usize::MAX;
        for row in rows {
            let Some(source) = self.rows.get(*row) else {
                continue;
            };
            if source.start == written {
                continue;
            }
            written = source.start;
            self.write_arm(*source, whole, into);
        }
    }

    /// What every row reads, so the Jacobian can be taken a GROUP of columns at a time.
    ///
    /// A sketch's rows are local — a distance names two points out of forty, a span row names one
    /// axis of two — so most pairs of parameters have no row in common and can be differenced
    /// together in one pass. The answer is exact rather than approximate: see
    /// [`substrate::nonlinear_least_squares::ColumnGrouping`] for why, and
    /// `every_relation_declares_what_its_rows_read` for the check that keeps it so.
    fn parameter_reads(&self) -> Option<ResidualReads> {
        Some(ResidualReads::from_rows(self.reads_by_row()))
    }
}

/// What a search did, before anything reads the shape of the drawing it landed in.
///
/// A [`SolveReport`]'s other two fields are a reading of that shape, and buying one costs a
/// Jacobian at the answer plus a rank of it. A pass whose answer is on its way somewhere else has
/// no shape worth reading, so it carries this instead.
#[derive(Clone, Copy)]
struct SearchTrace {
    outcome: SolveOutcome,
    iterations: usize,
}

fn domain_outcome(substrate: SubstrateSolveOutcome) -> SolveOutcome {
    match substrate {
        SubstrateSolveOutcome::Converged => SolveOutcome::Converged,
        SubstrateSolveOutcome::Stalled => SolveOutcome::Stalled,
        SubstrateSolveOutcome::ExhaustedIterations => SolveOutcome::ExhaustedIterations,
    }
}

fn domain_report(subtrate: SubstrateSolveReport) -> SolveReport {
    SolveReport {
        outcome: domain_outcome(subtrate.outcome),
        iterations: subtrate.iterations,
        residual_norm: subtrate.residual_norm,
        degrees_of_freedom: subtrate.degrees_of_freedom,
        redundant_residuals: subtrate.redundant_residuals,
    }
}

fn domain_trace(substrate: SubstrateSearchReport) -> SearchTrace {
    SearchTrace {
        outcome: domain_outcome(substrate.outcome),
        iterations: substrate.iterations,
    }
}

/// Seat the answer the search left in `parameters` back onto the drawing's own coordinates.
fn seat(
    problem: &Problem,
    system: &Residuals,
    parameters: &[f64],
    positions: &mut Vec<[f64; 2]>,
    scalar_coordinates: &mut Vec<f64>,
) {
    let whole = system.widen(parameters);
    *positions = (0..problem.points.len())
        .map(|slot| [whole[slot * 2], whole[slot * 2 + 1]])
        .collect();
    *scalar_coordinates = whole[problem.points.len() * 2..].to_vec();
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
    seat(problem, &system, &parameters, positions, scalar_coordinates);
    Some(domain_report(report))
}

/// Move the drawing exactly as [`run`] does, and report only what the search did.
///
/// The answer is [`run`]'s, parameter for parameter; what this does not buy is the reading of the
/// shape at that answer. A drag frame runs three passes and reads the shape of one — the preference
/// seed and the hand's own pass are both on their way somewhere else — and a settle runs two and
/// reads the second. Counted over the sketch suite, one search in three converges on the guess it
/// was handed and so takes no Jacobian at all under this verb; the suite's wall clock came down 6%.
/// See [`substrate::nonlinear_least_squares::search`] for the measurement and its gates.
fn run_reporting_only_the_search(
    problem: &Problem,
    positions: &mut Vec<[f64; 2]>,
    scalar_coordinates: &mut Vec<f64>,
    rigidity: Rigidity,
) -> Option<SearchTrace> {
    if problem.constraints.is_empty() {
        return None;
    }
    let system = Residuals::new(problem, scalar_coordinates, rigidity)?;
    let mut parameters = system.guess(positions);
    let report = search_nlls(&system, &mut parameters, SolveSettings::default());
    seat(problem, &system, &parameters, positions, scalar_coordinates);
    Some(domain_trace(report))
}

fn exact_report_at(
    problem: &Problem,
    positions: &[[f64; 2]],
    scalar_coordinates: &[f64],
    trace: SearchTrace,
) -> Option<SolveReport> {
    let system = Residuals::new(problem, scalar_coordinates, Rigidity::Ignored)?;
    let parameters = system.guess(positions);
    let mut residuals = vec![0.0; system.residual_count()];
    system.residuals(&parameters, &mut residuals);
    let residual_norm = residuals
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        .sqrt();
    if !residual_norm.is_finite() {
        return None;
    }
    let rank = rank(
        &jacobian(&system, &parameters),
        system.residual_count(),
        system.parameter_count(),
    );
    Some(SolveReport {
        outcome: trace.outcome,
        iterations: trace.iterations,
        residual_norm,
        degrees_of_freedom: system.parameter_count().saturating_sub(rank),
        redundant_residuals: system.residual_count().saturating_sub(rank),
    })
}

/// Search status diagnoses the numerical path; residual norm decides whether relations hold.
fn diagnostics(problem: &Problem, solution: &Solution, report: Option<SolveReport>) -> Diagnostics {
    let satisfied = report
        .as_ref()
        .is_none_or(|report| report.residual_norm <= SATISFIED_RESIDUAL);
    Diagnostics {
        report,
        satisfied,
        tangent_contacts_valid: problem.first_tangent_contact_failure(solution).is_none(),
    }
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
        run_reporting_only_the_search(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Preferred {
                anchored: &[],
                flexible_curves: &[],
                was: &[],
                opening: &[],
                reshaping: false,
            },
        );
        let report = run(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Ignored,
        );
        let solution = self.solution(positions, &scalar_coordinates);
        let diagnostics = diagnostics(self, &solution, report);
        Settled {
            solution,
            diagnostics,
            kept: None,
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
                // Every point is a freedom, an arc's center included, and every free scalar is
                // one more. An arc spends two of them on standing its ends its own radius away —
                // which is its radius column back again, so an arc is a net one freedom down,
                // exactly as the single equal-radius row it replaced left it.
                let arc_rows: usize = self
                    .arc_centers
                    .iter()
                    .map(|arc| if arc.radius.is_some() { 2 } else { 1 })
                    .sum();
                (self.points.len() * 2
                    + self
                        .parameters
                        .iter()
                        .filter(|parameter| parameter.free)
                        .count())
                .saturating_sub(arc_rows)
            },
            |report| report.degrees_of_freedom,
        );
        let solution = self.solution(positions, &scalar_coordinates);
        let diagnostics = diagnostics(self, &solution, report);
        Analysis {
            solution,
            diagnostics,
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
        let flexible_curves = match relation {
            Relation::Symmetry { first, second, .. } => vec![first, second],
            _ => Vec::new(),
        };
        let settled = candidate_problem.settle_with(&anchored, &flexible_curves);
        if !settled.diagnostics.satisfied {
            return Ok(TrialAdd::Rejected(TrialRejection::Unsatisfied {
                conflicts: self.blame(relation, candidate, &anchored, &flexible_curves),
            }));
        }
        if let Some(failure) = candidate_problem.first_tangent_contact_failure(&settled.solution) {
            return Ok(TrialAdd::Rejected(TrialRejection::InvalidTangent {
                constraint: failure.constraint,
                error: failure.error,
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

    /// Return the standing constraints whose individual removal restores a satisfied system.
    ///
    /// An empty result means the system already satisfies or no single removal repairs it. The
    /// keys stay local because only the document adapter owns durable constraint identity.
    pub fn standing_conflicts(&self) -> Vec<ConstraintId> {
        if self.settle().diagnostics.satisfied {
            Vec::new()
        } else {
            self.leave_one_out(|without| without.settle().diagnostics.satisfied)
        }
    }

    /// Pull one local point toward a target, then release the hand and settle standing relations.
    /// The hand is not a hard pin: a point free to slide along a relation must still be able to
    /// move even when the cursor is not exactly on that relation. An achievable pull is unchanged;
    /// an impossible pull returns the nearest standing solution or a rejection.
    /// # Errors
    ///
    /// Returns an error if the held point is not local to this problem.
    pub fn drag(&self, held: PointId, at: [f64; 2]) -> Result<DragOutcome, RequestError> {
        self.drag_together(
            &[Hand {
                point: held,
                to: at,
                role: HandRole::Lead,
            }],
            &[],
        )
    }

    /// The rim of the falloff for a hand keeping the RADIUS it stands at, as a share of how far
    /// the hand travelled.
    ///
    /// Read as an angle it is a cone about the direction that keeps the quantity, so a hand moving
    /// nearly ALONG it is understood to be moving along it. Nothing switches AT the rim — see
    /// [`Problem::snapped`] for why the correction has to arrive there already faded to nothing
    /// rather than be dropped when it is crossed.
    ///
    /// **It is not a fixed number of degrees, and the number shrinks as the gesture lengthens.**
    /// `across` is the distance from the cursor to the LOCUS, and a straight line tangent to a
    /// circle of radius `R` leaves it quadratically — about `travel² / 2R` — while the cone grows
    /// only linearly in travel. So a hand that FOLLOWS the locus is held however far it goes, and a
    /// hand that strikes out in a straight line is let go the further it commits. Measured on a
    /// radius of 40 (`a_radius_is_held_within_the_angle_it_is_measured_to_be`):
    ///
    /// | travel | held exactly within |
    /// | --- | --- |
    /// | 2 | 25.5° |
    /// | 10 | 20.5° |
    /// | 30 | 8.7° |
    ///
    /// Holding it is worth more than it looks. Held exactly, the whole rigid set moves by one
    /// similarity and the drawing never has to be reconciled, so the free sweep stops being spent
    /// at random: measured on a curved slot, the far cap's wander across a cursor step of 0.005
    /// went from 2.7 to 2.8e-25.
    ///
    /// **A segment's length once had a cone of its own, and should not have.** It was a third of
    /// this one, on the argument that a length has to be given up readily because dragging an end
    /// is how a length is authored. But the argument concedes the case: a hand pulled onto a
    /// circle no curve draws is a hand pulled onto geometry that is not in the drawing, and the
    /// narrow cone only made it rare. On a rectangle it was not even rare. The horizontals and
    /// verticals hold a dragged corner exactly tangential to the circle about its neighbour, so
    /// the quadratic escape above never happens and the span engaged on every axis-aligned drag —
    /// the corner missed the cursor by up to 2.42 on the author's own drawing, and the ring drawn
    /// about the neighbouring corner is what
    /// `dragging_a_rectangles_corner_resizes_it_rather_than_moving_it` now watches for. Deleting
    /// the candidate cost one test, which existed to measure the constant.
    const SNAP_CONE_KEEPING_A_RADIUS: f64 = 0.75;

    /// The share of the cone over which the quantity holds EXACTLY, before it starts letting go.
    ///
    /// A falloff without a plateau is not a snap. Fading from the moment the hand is off the
    /// quantity means a hand a fifteenth of a radius off it lands a fifteenth off it too, only
    /// slightly pulled in — which is the behaviour the snap exists to replace. The plateau is
    /// where the author is understood to be ON the quantity; the band outside it is where the
    /// snap gives the quantity up, and gives it up smoothly.
    const SNAP_HOLD: f64 = 0.6;

    /// How hard a quantity pulls on a hand standing `across` from it, measured in the `cone` the
    /// gesture opened: one on the plateau, nothing at the rim, a smoothstep between.
    ///
    /// A snap that is simply dropped when the hand leaves its cone makes every drag a spring, and
    /// the author felt it: "small changes in movement of the mouse result in massive swings of
    /// movement back and forth". The snapped and unsnapped answers differ by the WHOLE correction
    /// exactly where the hand crosses between them, so on a six-unit gesture a hundredth of a unit
    /// of mouse swung the drawing 3.79 and swung it back on the next frame, forever — measured at
    /// 189x gain against about 1.7 for every other drag there is. Damping the solve cannot reach
    /// it: the jump is in the question being asked, not in how it is answered, and the numerics
    /// are already a trust region.
    ///
    /// The smoothstep is zero in VALUE and in SLOPE at both ends of the band, so there is nothing
    /// left to cross at the rim and no kink where the holding stops. The plateau is what keeps it
    /// a snap rather than a weak pull toward one — fading from the moment the hand is off the
    /// quantity leaves a hand a fifteenth of a radius off it landing a fifteenth off it too.
    fn pull_toward(across: f64, cone: f64) -> f64 {
        let held_within = Self::SNAP_HOLD * cone;
        if across <= held_within {
            return 1.0;
        }
        let past = (across - held_within) / (cone - held_within);
        1.0 - past * past * past.mul_add(-2.0, 3.0)
    }
    /// How far a walked drag turns in one step, and the most steps it will ever walk.
    const TURN_PER_FRAME: f64 = std::f64::consts::PI / 180.0;
    const MOST_FRAMES: u32 = 16;

    /// Every point standing in the same place as this one, itself included.
    ///
    /// The dot the author can grab is often not the vertex the curves were drawn through. A slot
    /// keeps a HANDLE on its spine — its own point, held onto the geometry by a coincidence —
    /// precisely so that dragging it can mean something different from dragging the derived center
    /// underneath. But a quantity measured to one of two points standing in one place is measured
    /// to the other, so when a drag asks what it has hold of, they answer together. Without this a
    /// slot's end handle belongs to no curve at all and every rule below passes it by, which is
    /// exactly what it did.
    fn standing_together(&self, point: PointId) -> Vec<PointId> {
        let mut together = vec![point];
        let mut grew = true;
        while grew {
            grew = false;
            for constraint in &self.constraints {
                let Relation::Coincident { first, second } = constraint.relation else {
                    continue;
                };
                for (near, far) in [(first, second), (second, first)] {
                    if together.contains(&near) && !together.contains(&far) {
                        together.push(far);
                        grew = true;
                    }
                }
            }
        }
        together
    }

    /// Where a point stood before the hand, which for anything but a hand is where it stands.
    fn stood_of(
        &self,
        point: PointId,
        was: &[(PointId, [f64; 2])],
        positions: &[[f64; 2]],
    ) -> Option<[f64; 2]> {
        was.iter()
            .find(|(named, _)| *named == point)
            .map(|(_, at)| *at)
            .or_else(|| {
                (point.owner == self.owner)
                    .then(|| positions.get(point.index).copied())
                    .flatten()
            })
    }

    /// Pull the LEAD hand onto a quantity its own curve already had, when it moves along one.
    ///
    /// A round curve names a radius and a preference asks for it back. But a hand is an assertion
    /// and a preference is not, so when the cursor sits off the circle that radius draws, the two
    /// cannot both be met and the hand wins: the drawing translates under it instead of the curve
    /// sweeping around a center that stays. No amount of preference fixes that, because
    /// translating satisfies every row exactly and nothing outranks an answer at zero residual.
    ///
    /// So the disagreement is settled before the solve rather than inside it. Snapping the cursor
    /// ONTO the circle makes the sweep an exact answer too, and a cheaper one, and the solve then
    /// finds it on its own. This is why it is a snap and not a mode: the author feels the point
    /// stick to a radius while they move around it, and feels it let go the moment they pull
    /// across. A single frame of a drag is small, so a sweep re-snaps every frame and the quantity
    /// survives the whole gesture, while a deliberate pull leaves the cone at once.
    ///
    /// Every quantity the drawing offers a hand standing at `held`: the point it is measured
    /// from, and the cone that KIND of quantity is held in.
    ///
    /// **Only a curve that DRAWS the circle offers it.** A round curve's end already slides along
    /// its own radius, so keeping it is keeping something the author can see. The circle about a
    /// segment's far end is drawn by nothing: it is a locus this function would be inventing, and
    /// a hand pulled onto an invented locus is a hand pulled onto geometry that is not there. See
    /// [`Problem::SNAP_CONE_KEEPING_A_RADIUS`] for what the span candidate cost while it existed.
    fn quantities_a_hand_could_keep(&self, held: PointId) -> Vec<(PointId, f64)> {
        let together = self.standing_together(held);
        let ends_here = |point: PointId| together.contains(&point);
        self.arc_centers
            .iter()
            .filter_map(|arc| (ends_here(arc.from) || ends_here(arc.to)).then_some(arc.center))
            .map(|pivot| (pivot, Self::SNAP_CONE_KEEPING_A_RADIUS))
            .collect()
    }

    /// `opening` is the drawing as the GESTURE found it, not as a walk has left it: everything a
    /// quantity is measured from that is not itself under a hand is read out of it.
    fn snapped(
        &self,
        hands: &[Hand],
        was: &[(PointId, [f64; 2])],
        opening: &[[f64; 2]],
    ) -> Option<Snap> {
        // The quantity being kept is the LEAD's, because the lead is the point the author has hold
        // of. A carried hand rides whatever the lead does, and a pin is not moving at all.
        let (_, lead) = Hand::lead_of(hands)?;
        let (held, now) = (lead.point, lead.to);
        let stood_at = |point: PointId| self.stood_of(point, was, opening);
        let stood = stood_at(held)?;
        // How far the hand has been from its opening, which is what opened the cone — not how far
        // it stands now. See [`Problem::the_hand_having_reached`]: a hand that has swept an arc
        // end a whole turn is back where it pressed, and reading only this frame shuts the cone
        // on it. The mark defaults to nothing, so a caller that keeps no path reads exactly the
        // displacement it always did.
        let travel = (now[0] - stood[0])
            .hypot(now[1] - stood[1])
            .max(self.furthest_the_hand_has_reached);
        if !travel.is_finite() || travel <= 0.0 {
            return None;
        }
        let mut snap: Option<Nearest> = None;
        for (pivot, share) in self.quantities_a_hand_could_keep(held) {
            let Some(about) = stood_at(pivot) else {
                continue;
            };
            let quantity = (stood[0] - about[0]).hypot(stood[1] - about[1]);
            let reach = (now[0] - about[0]).hypot(now[1] - about[1]);
            if !quantity.is_finite() || !reach.is_finite() || quantity <= 0.0 || reach <= 0.0 {
                continue;
            }
            // The cone the gesture opened, under whatever ceiling the caller set. Taking the
            // smaller keeps the falloff continuous in both — it is still a cone, just a shorter
            // one, and everything below reads the same.
            let cone = (share * travel).min(self.snap_reach.0);
            let across = (reach - quantity).abs();
            if across >= cone {
                continue;
            }
            // The correction FADES to nothing at the rim rather than being let go at full size.
            //
            // A hard cone made every drag a spring, and the author felt it: "small changes in
            // movement of the mouse result in massive swings of movement back and forth". The
            // snapped and unsnapped answers differ by the WHOLE correction exactly where the hand
            // crosses between them, so on a six-unit gesture a hundredth of a unit of mouse swung
            // the drawing 3.79 and swung it back on the next frame, forever. Measured at 189x gain
            // against about 1.7 for every other drag there is. Worse, the correction is a share of
            // travel, so the bang grew with the gesture: travel 3, 6 and 12 let go of 1.90, 3.76
            // and 7.46. Damping the solve cannot reach this — the jump is in the question being
            // asked, not in how it is answered, and the numerics are already a trust region.
            //
            let pull = Self::pull_toward(across, cone);
            let target = (quantity - reach).mul_add(pull, reach);
            if snap.is_none_or(|nearest| across < nearest.across) {
                snap = Some(Nearest {
                    about,
                    scale: target / reach,
                    quantity,
                    across,
                    across_the_cone: across / cone,
                    pull,
                    reach,
                });
            }
        }
        let Nearest {
            about,
            scale,
            quantity,
            across_the_cone: across_the_cone_of_lead,
            pull,
            reach: reach_of_lead,
            ..
        } = snap?;
        let arm = |at: [f64; 2]| [at[0] - about[0], at[1] - about[1]];
        let (from, to) = (arm(stood), arm(now));
        // The snap is a TURN of the whole rigid set, not a correction to one point of it. Moving
        // the lead onto the circle and leaving what it carries where a straight cursor delta put
        // them tears the set apart every step: on a slot swept by one end, the cap's center ran
        // ahead of its own two corners and the cap had to stretch to stay attached, which is a
        // slot that fattens as it sweeps. Pins are handed back untouched — they are what holds the
        // drawing still, and turning them would give the gesture away.
        //
        // Written as a similarity about the pivot: the same complex multiply that takes `stood` to
        // the snapped lead takes every carried point with it, so the set keeps its shape exactly.
        // ONE map, faded once, applied to the lead and to everything it carries alike. The turn
        // fades with the radius: fading only the radius left the set turning through the hand's
        // full angular travel however weakly the quantity pulled, so the drawing was neither where
        // a translation would put it nor where a snap would, and the solve spent a real freedom
        // reconciling the two — the same spring, one layer down. Blending the coefficients of two
        // complex affine maps yields a third, so the faded map is a translation at the rim, the
        // exact similarity of ADR 0042 on the quantity, and a similarity — never a distortion — at
        // every pull between.
        let lead_now = [about[0] + to[0] * scale, about[1] + to[1] * scale];
        let from_arm = arm(stood);
        let denominator = from_arm[0].mul_add(from_arm[0], from_arm[1] * from_arm[1]);
        let similarity = if denominator > 0.0 {
            let held = quantity / reach_of_lead;
            let turned = [
                to[0].mul_add(from_arm[0], to[1] * from_arm[1]) / denominator * held,
                to[1].mul_add(from_arm[0], -(to[0] * from_arm[1])) / denominator * held,
            ];
            [(turned[0] - 1.0).mul_add(pull, 1.0), turned[1] * pull]
        } else {
            [1.0, 0.0]
        };
        let mut snapped = hands.to_vec();
        for hand in &mut snapped {
            match hand.role {
                HandRole::Lead => hand.to = lead_now,
                HandRole::Carried => {
                    let Some(rode) = stood_at(hand.point) else {
                        continue;
                    };
                    let rides = [rode[0] - stood[0], rode[1] - stood[1]];
                    hand.to = [
                        lead_now[0] + similarity[0].mul_add(rides[0], -(similarity[1] * rides[1])),
                        lead_now[1] + similarity[1].mul_add(rides[0], similarity[0] * rides[1]),
                    ];
                }
                HandRole::Pin => {}
            }
        }
        Some(Snap {
            hands: snapped,
            kept: KeptQuantity {
                about,
                radius: quantity,
                across_the_cone: across_the_cone_of_lead,
            },
            turn: (from[0] * to[1] - from[1] * to[0])
                .atan2(from[0] * to[0] + from[1] * to[1])
                .abs(),
            pull,
        })
    }

    /// The hands moved onto the quantity a drag would keep, and that quantity — with no solve.
    ///
    /// A snap is a property of the drawing's GEOMETRY, not of its constraints. A bare arc's end
    /// still stands a radius away from its own center whether or not anything is asserted about
    /// it, and it is still true that the only thing dragging that end can mean is a sweep.
    ///
    /// This exists because the caller short-circuits a drag with nothing standing — no relation to
    /// trade the pull against means the hands ARE the answer — and that path skipped the snap
    /// along with the solve. The shape most obviously "arc-like" was therefore the one place an
    /// end never held its radius and never drew its ghost.
    pub fn snap_the_hands(
        &self,
        hands: &[Hand],
        was: &[(PointId, [f64; 2])],
    ) -> Option<(Vec<Hand>, KeptQuantity)> {
        if hands
            .iter()
            .any(|hand| hand.point.owner != self.owner || hand.point.index >= self.points.len())
        {
            return None;
        }
        let positions: Vec<[f64; 2]> = self.points.iter().map(|point| point.at).collect();
        let snap = self.snapped(hands, was, &positions)?;
        Some((snap.hands, snap.kept))
    }

    /// The same problem, with a ceiling on how far a snap may carry a hand — see [`SnapReach`].
    #[must_use]
    pub const fn holding_a_snap_within(mut self, reach: SnapReach) -> Self {
        self.snap_reach = reach;
        self
    }

    /// The same problem, told how far from its opening the lead hand has ALREADY been.
    ///
    /// The cone a snap is held in is opened by how far the hand has travelled, and a single frame
    /// only knows where the hand stands. Those are the same number until the hand turns back, and
    /// a hand sliding around a pivot turns back at every step of the second half: sweep an arc end
    /// a whole turn and it arrives back where it pressed, so the displacement reads zero and the
    /// cone shuts on a gesture that walked the whole circumference.
    ///
    /// A gesture that has been this far HAS been this far, so what a frame reports is a floor
    /// rather than the whole reading. The caller keeps the mark because only the caller has the
    /// path — the drawing is rebuilt from the press every frame and cannot remember one.
    ///
    /// The high-water of the DISPLACEMENT, not the length of the road: summing each frame's step
    /// would let a hand held still ramp the cone open on tremor alone, which is the one thing the
    /// ramp exists to prevent.
    #[must_use]
    pub fn the_hand_having_reached(mut self, furthest: f64) -> Self {
        self.furthest_the_hand_has_reached = if furthest.is_finite() && furthest > 0.0 {
            furthest
        } else {
            0.0
        };
        self
    }

    /// The curves this gesture is allowed to RESHAPE — the ones whose span the rigidity
    /// preference must stop pricing, because the hand is authoring them.
    ///
    /// A hand on a curve's END is the author saying "change THIS curve". Nothing else loosens: the
    /// rest of the drawing still prefers to travel rather than deform, which is what carries a
    /// shape under one finger.
    ///
    /// Without this the preference is too strong to be useful. Measured pre-hand, a span row is an
    /// honest statement that the shape is rigid, and a rigid drawing under a pinned hand has
    /// exactly one answer — translate — whatever was grabbed and whatever else is asserted. A level
    /// segment dragged by one end took its far end along instead of leaving it where the level
    /// allowed, and a segment whose other end was FIXED translated anyway and stayed translated
    /// once the hand lost.
    ///
    /// A curve's SHAPE parameter is untouched by this: an arc grabbed by an end gives up its chord,
    /// not its radius, which is what lets the end sweep around a center that stays put rather than
    /// dragging the whole arc after it.
    ///
    /// `in_hand` is every point the hands hold, a handle already resolved to the vertex it stands
    /// on — see [`Problem::standing_together`].
    fn curves_the_hands_may_reshape(&self, in_hand: &[PointId]) -> Vec<SketchCurve> {
        let grabbed = |point: PointId| in_hand.contains(&point);
        // Concentric arcs are one rail family, and a preference prices an arc by its CHORD, which
        // is not something a sweep leaves alone: swing one rail's end around and every sibling's
        // chord shortens with it. A rigid sibling therefore outvotes the sweep by itself, and the
        // drawing slides under the cursor instead — the same reasoning that already stops the
        // family's radii being held when a hand has a whole rail, reaching the chords too.
        let swept: Vec<PointId> = self
            .arc_centers
            .iter()
            .filter(|arc| grabbed(arc.from) || grabbed(arc.to))
            .map(|arc| arc.center)
            .collect();
        // A closed loop re-derives whatever a loosening leaves out of it. The spans around a loop
        // sum to zero, so holding every edge but the two a corner joins states those two as well,
        // and a loop with nothing left to give has one motion under a pinned hand: translate. A
        // rectangle dragged by a corner slid across the plane instead of resizing. So the unit of
        // loosening is the BICONNECTED BLOCK the hand stands in — the edges no single point can
        // separate — which is the rule the incident edges were already following: in an open chain
        // every edge is a block of its own and nothing here changes, and in a loop the block is
        // the loop. An ARC's chord is an edge of that graph — it is how a slot's loop closes at
        // all — but it is not loosened by standing in one: an arc chord already has a rule of its
        // own in `swept`, measured against the rail family, and widening it there cost a slot the
        // drag that carries a cap past its partner.
        let chords: Vec<(usize, usize)> = self
            .segments
            .iter()
            .map(|segment| (segment.from.index, segment.to.index))
            .chain(
                self.arc_centers
                    .iter()
                    .map(|arc| (arc.from.index, arc.to.index)),
            )
            .collect();
        let standing_in: Vec<usize> = biconnected_blocks(self.points.len(), &chords)
            .into_iter()
            .filter(|block| {
                block.iter().any(|&edge| {
                    let (from, to) = chords[edge];
                    in_hand
                        .iter()
                        .any(|held| held.index == from || held.index == to)
                })
            })
            .flatten()
            .collect();
        self.segments
            .iter()
            .enumerate()
            .filter(|(index, segment)| {
                standing_in.contains(index) || grabbed(segment.from) || grabbed(segment.to)
            })
            .map(|(index, _)| {
                SketchCurve::Segment(SegmentId {
                    owner: self.owner,
                    index,
                })
            })
            .chain(
                self.arc_centers
                    .iter()
                    .filter(|arc| {
                        grabbed(arc.from) || grabbed(arc.to) || swept.contains(&arc.center)
                    })
                    .map(|arc| SketchCurve::Arc(arc.key)),
            )
            .collect()
    }

    /// Pull SEVERAL local points toward their targets at once, then release and settle.
    ///
    /// One hand asks the drawing to do whatever it likes as long as this point ends up here, and
    /// least motion decides the rest — which is right for grabbing a vertex and wrong for grabbing
    /// a SHAPE. Moving a shape means every point of it goes the same way, and there is no relation
    /// that says so: the freedoms a slot keeps on purpose (its width, its radius) are exactly the
    /// ones a single hand will spend instead of translating. Naming all the points the gesture
    /// holds is how the caller says which motion it meant, without a rigidity relation that would
    /// take those freedoms away for good.
    ///
    /// Each hand joins as one more least-squares row, so they trade off against each other as well
    /// as against everything standing; a set of targets the drawing cannot meet lands as close as
    /// it can rather than failing outright.
    ///
    /// # Errors
    ///
    /// Returns an error if any held point is not local to this problem.
    pub fn drag_together(
        &self,
        hands: &[Hand],
        was: &[(PointId, [f64; 2])],
    ) -> Result<DragOutcome, RequestError> {
        for hand in hands {
            if hand.point.owner != self.owner || hand.point.index >= self.points.len() {
                return Err(RequestError::UnknownPoint);
            }
        }
        let in_hand: Vec<PointId> = hands
            .iter()
            .flat_map(|hand| self.standing_together(hand.point))
            .collect();
        let loosened = &self.curves_the_hands_may_reshape(&in_hand);
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        // The drawing as the gesture FOUND it, kept aside so the walk cannot re-aim the preference
        // at its own intermediate answers — see [`Rigidity::Preferred::opening`]. The problem's own
        // points are not that drawing: the caller writes the hands into the sketch and prepares the
        // problem afterwards, so what arrives here is already bent. `was` is the only record of
        // where the hands stood, which is why every drag sends it.
        let mut opening = positions.clone();
        for (held, stood) in was {
            if let Some(slot) = opening.get_mut(held.index) {
                *slot = *stood;
            }
        }
        let mut scalar_coordinates = self.scalar_coordinates();
        let (origin, frames) = self.walk_of(hands, was, &positions);
        let mut answered = Frame::default();
        for frame in 1..=frames {
            let share = f64::from(frame) / f64::from(frames);
            let target: Vec<Hand> = hands
                .iter()
                .zip(&origin)
                .map(|(hand, (_, from))| Hand {
                    to: [
                        from[0] + (hand.to[0] - from[0]) * share,
                        from[1] + (hand.to[1] - from[1]) * share,
                    ],
                    ..*hand
                })
                .collect();
            // The drawing as the walk has left it. A frame reads its preference off what is in
            // front of it, and a problem carries its own positions as that reference, so a walked
            // frame has to be handed the drawing it walked to rather than the one the gesture
            // started from — otherwise every frame re-asks for the original shape and the walk
            // says nothing the first frame did not.
            let standing = self.standing_at(&positions, &scalar_coordinates);
            // Every step measures from where the GESTURE started, not from where the last step
            // landed. The walk hands out a straight chord while a snapped drag sweeps an arc, so
            // the two part company by the sagitta — a fixed amount of geometry — while a step's own
            // travel shrinks as the walk gets finer. Measured against the previous step, the cone
            // is a fraction of that shrinking travel and loses to the gap partway through: on a
            // nine-step sweep the snap dropped out at step six and again at step nine, and step
            // nine is the one that delivers the answer, so the whole walk was spent to hand over
            // the raw cursor anyway. Against the origin the cone grows with the drag  it keeps.
            //
            // It also stops the answer creeping. A step that measures from the last one snaps to
            // whatever that step settled at, so the quantity it is meant to be keeping drifts a
            // little each time; measured from the origin it is the quantity the author had.
            answered = standing.drag_one_frame(
                &target,
                &origin,
                &opening,
                loosened,
                &mut positions,
                &mut scalar_coordinates,
            )?;
        }
        let solution = self.solution(positions, &scalar_coordinates);
        let diagnostics = diagnostics(self, &solution, answered.report);
        let settled = Settled {
            solution,
            diagnostics: diagnostics.clone(),
            kept: answered.kept,
        };
        Ok(
            if diagnostics.satisfied && diagnostics.tangent_contacts_valid {
                DragOutcome::Accepted(settled)
            } else {
                DragOutcome::Rejected(settled)
            },
        )
    }

    /// Where each hand STARTED, and how many steps the drag should walk to get where it is going.
    ///
    /// A solve is LOCAL, and a snapped drag is a ROTATION — the one motion a linearization is
    /// worst at. Arriving a frame at a time a sweep turns a fraction of a degree and this never
    /// comes up; a fast hand, or a caller that jumps straight to the answer, hands over the whole
    /// turn at once and the pass settles into a mixture of travel and distortion instead. Measured
    /// on a curved slot swept 7.8 degrees, one frame collapsed the rails from 36/40/44 to
    /// 33.5/38.3/43.2, while a degree at a time held them and left the far end inside a twentieth
    /// of a voxel.
    ///
    /// Walking it is therefore what makes a drag's answer independent of how fast the frames
    /// arrived, which is a thing that ought to be true rather than a number worth tuning. Only a
    /// snapped drag pays for it, and only when it turns far enough to have to.
    fn walk_of(
        &self,
        hands: &[Hand],
        was: &[(PointId, [f64; 2])],
        positions: &[[f64; 2]],
    ) -> (Vec<(PointId, [f64; 2])>, u32) {
        let origin: Vec<(PointId, [f64; 2])> = hands
            .iter()
            .map(|hand| {
                let at = was
                    .iter()
                    .find(|(named, _)| *named == hand.point)
                    .map(|(_, at)| *at)
                    .or_else(|| positions.get(hand.point.index).copied())
                    .unwrap_or_default();
                (hand.point, at)
            })
            .collect();
        let frames = self
            .snapped(hands, &origin, positions)
            .map_or(1, |opening| {
                // Stepped by how much of a rotation the snap is actually IMPOSING, not by how far
                // the hand went. The walk earns its cost because a snapped drag forces the set
                // around a pivot and a linearization is worst at exactly that; where the falloff
                // has let go, nothing is being forced and there is nothing to walk. Scaling by the
                // pull is what stops the step count dropping sixteen to one the instant the hand
                // leaves the cone, which was the same spring wearing a second hat: one frame
                // against sixteen is not a rounding difference, it collapsed a slot's rails from
                // 36/40/44 to 33.5/38.3/43.2 over less than eight degrees.
                let forced = opening.pull * opening.turn;
                (1..=Self::MOST_FRAMES)
                    .find(|frames| forced <= f64::from(*frames) * Self::TURN_PER_FRAME)
                    .unwrap_or(Self::MOST_FRAMES)
            });
        (origin, frames)
    }

    /// The same problem standing where a walked drag has got to, which is what the document
    /// itself hands down between one frame of a real drag and the next.
    fn standing_at(&self, positions: &[[f64; 2]], scalars: &[f64]) -> Self {
        let mut walked = self.clone();
        for (point, at) in walked.points.iter_mut().zip(positions) {
            point.at = *at;
        }
        for (parameter, value) in walked.parameters.iter_mut().zip(scalars) {
            parameter.stored = *value;
        }
        walked
    }

    /// One frame of a drag: snap the hand, seed under the preference, then let the hand and then
    /// the drawing have their say. See [`Problem::drag_together`] for why there are three.
    fn drag_one_frame(
        &self,
        hands: &[Hand],
        was: &[(PointId, [f64; 2])],
        opening: &[[f64; 2]],
        loosened: &[SketchCurve],
        positions: &mut Vec<[f64; 2]>,
        scalar_coordinates: &mut Vec<f64>,
    ) -> Result<Frame, RequestError> {
        /// The share of the travel a drawing can cover that a radius hold may give back before it
        /// is dropped. See where it is read for the two measurements that put it here.
        const MOST_OF_THE_TRAVEL: f64 = 0.5;
        // Whose gesture this is, said by the caller rather than measured here. A reshape names
        // what it turns ABOUT, so a set with a pin in it is one — and the drawing has to hold
        // still around the moving vertex, or the whole slot slides over to meet the cursor and
        // nothing gets wider. Asked of the roles and not of the numbers, this survives the snap
        // and does not need a stillness tolerance to decide that a settled pin is still a pin.
        let reshaping = hands.iter().any(|hand| hand.role == HandRole::Pin);
        // The OPENING, not the drawing the walk has got to. `was` names only the hands, so every
        // other point a quantity is measured from — a pivot, an arc's center — falls through to
        // this slice, and handing it the walked positions re-measures the quantity against an
        // answer the previous substep already snapped. It ratchets: on a rectangle corner the span
        // it was meant to be keeping grew 72.5185 -> 74.5786 over sixteen substeps, and the ring
        // was drawn about a corner that had slid nineteen units. The walk states this law four
        // lines above where it hands `origin` to every step, and this read was the one place that
        // was not obeying it.
        let snap = self.snapped(hands, was, opening);
        let kept = snap.as_ref().map(|snap| snap.kept);
        let hands = snap.as_ref().map_or(hands, |snap| snap.hands.as_slice());
        // The hand is written into the guess as well as asserted, so the pass starts from the
        // drawing the author is looking at rather than from the one they left behind.
        for hand in hands {
            if let Some(slot) = positions.get_mut(hand.point.index) {
                *slot = hand.to;
            }
        }
        let mut pulled = self.clone();
        for hand in hands {
            let pull = Relation::Fix {
                point: hand.point,
                at: hand.to,
            };
            pulled = pulled.with_candidate(
                pull,
                self.resolve(pull).map_err(RequestError::InvalidRelation)?,
            );
        }
        // SPIKE: hold every undimensioned metric quantity at what the OPENING measured — arc radii
        // by taking their column away, segment lengths by adding the row that says so.
        if snap.is_some() {
            for (radius, at) in self.radii_a_snapped_gesture_keeps(loosened, opening) {
                if let Some(parameter) = pulled.parameters.get_mut(radius.index) {
                    parameter.stored = at;
                    parameter.free = false;
                }
            }
        }
        // Two passes, and the split is the whole idea. A drawing that is not fully determined has a
        // family of answers under the hand, and picking the one nearest where every point already
        // stood is what makes a shape stretch when it could have travelled: moving a body of N
        // points by d costs N times d squared, so the arithmetic would always rather deform it.
        // The first pass ranks that family by how much the drawing has to CHANGE SHAPE rather than
        // how far it moves, which costs a carry nothing and leaves the author's shape intact.
        //
        // But a preference must not be allowed to cost the author the cursor, so it does not get
        // the last word: it only seeds. The second pass carries the hand at full authority with the
        // preference switched off, and Gauss-Newton from a near seed lands on the answer beside it
        // — the branch the first pass chose, reached exactly.
        //
        // The third pass is the drawing having the last word over the hand. A hand is a pull and
        // not an assertion, so what the author already asserted is the only thing acceptance is
        // measured against: solving it alone leaves an achievable drag exactly where the second
        // pass put it, and takes an impossible one back to where the relations say it belongs
        // rather than reporting the author's own drawing as broken.
        //
        // The second pass does not always have an answer to reach. A hand on an arc's center is
        // incompatible by construction — every cap center of a slot stalls around 1.7e-2 at every
        // reach — so it exits with a compromise, and mirroring the drawing moves that compromise in
        // the third decimal. Nothing reads the stalled residual: the report handed out is the third
        // pass's. What carries is WHERE on the family the third pass re-seats, which is why a knob
        // that ought to mean nothing moves the answer here without the arithmetic being wrong.
        //
        // A point the author fixed does not travel with a body, so the preference is not allowed to
        // weigh carrying it: anchoring drops it out of the pass entirely rather than leaving the
        // spans to argue with the relation and lose the drag to a conflict neither one meant.
        let held = self.points_the_author_fixed();
        let gesture = |hand_problem: &Self,
                       drawing: &Self,
                       positions: &mut Vec<[f64; 2]>,
                       scalars: &mut Vec<f64>| {
            run_reporting_only_the_search(
                hand_problem,
                positions,
                scalars,
                Rigidity::Preferred {
                    anchored: &held,
                    flexible_curves: loosened,
                    was,
                    opening,
                    reshaping,
                },
            );
            run_reporting_only_the_search(hand_problem, positions, scalars, Rigidity::Ignored);
            run(drawing, positions, scalars, Rigidity::Ignored)
        };
        // The gesture is run as the drawing has always run it, and then run again with the radius
        // of every carried arc taken out of the solve. A drawing that does not dimension its own
        // width has a FAMILY of exact answers under the hand — the slot that prompted this settles
        // to 5e-11 with the cap 7.3271 wide and to 5e-11 with it 1.7792 wide, both satisfying every
        // relation it has — and left alone the passes walk to whichever member lies nearest a seed
        // they have already dragged far from the drawing. Holding the radius overrules nothing; it
        // picks the member the author drew out of answers the drawing calls equally good.
        //
        // WHICH IS WHY THE TWO ARE JUDGED AT THE HANDS AND NOT BY THEIR OWN RESIDUALS. A held
        // radius can reach a perfectly converged answer by spending the gesture instead: fix one
        // end of an arc and carry its center, and the drawing honours both the fix and the old
        // radius by putting the center back where it started. That converges to 5e-11, and it is
        // the drag thrown away. Kept only when it costs the author nothing at the cursor, the hold
        // yields to everything the author actually asserted without being ranked against any of it.
        //
        // The held run goes first and the loose one is only paid for when the held one fell
        // short, because a hold that put the lead ON the cursor cannot be beaten by dropping it.
        // Most gestures are that case, and running both regardless cost the suite two and a half
        // times its wall clock for answers that never differed.
        let (seed, seed_scalars) = (positions.clone(), scalar_coordinates.clone());
        let report = match self.quantities_a_carry_holds_still(hands, opening) {
            None => gesture(&pulled, self, positions, scalar_coordinates),
            Some(carried) => {
                let pulled_held = pulled
                    .quantities_a_carry_holds_still(hands, opening)
                    .unwrap_or_else(|| pulled.clone());
                let held_report = gesture(&pulled_held, &carried, positions, scalar_coordinates);
                // The LEAD hand alone, because the cursor is the lead's and no other hand's. Asked
                // of every hand at once the question is drowned by the ones that structurally
                // cannot answer it: carry an arc whose end the author fixed and that end misses its
                // carried place by the whole drag, identically either way, leaving the three
                // thousandths between them to decide it.
                let reached = |settled: &[[f64; 2]]| {
                    hands
                        .iter()
                        .filter(|hand| hand.role == HandRole::Lead)
                        .filter_map(|hand| {
                            let here = settled.get(hand.point.index)?;
                            Some((here[0] - hand.to[0]).hypot(here[1] - hand.to[1]))
                        })
                        .fold(0.0_f64, f64::max)
                };
                // How far the lead hand ASKED to go. No answer can cover more of the travel than
                // was asked for, which is what makes the bound below exact.
                let asked = hands
                    .iter()
                    .filter(|hand| hand.role == HandRole::Lead)
                    .filter_map(|hand| {
                        let stood = opening.get(hand.point.index)?;
                        Some((stood[0] - hand.to[0]).hypot(stood[1] - hand.to[1]))
                    })
                    .fold(0.0_f64, f64::max);
                let covered = |settled: &[[f64; 2]]| asked - reached(settled);
                // The loose run exists only to give the hold something to lose to, and what it is
                // judged by is `covered(held) >= MOST_OF_THE_TRAVEL * covered(loose)`. Since no
                // answer can cover more than `asked`, a hold that already covers that share of
                // `asked` beats every loose answer there could be, and running one can only
                // produce a number it already beats. Skipping it is the SAME test evaluated
                // against the bound instead of against the measurement, so the answer is
                // unchanged bit for bit — not a tolerance and not an approximation. On a slot's
                // cap center, where every drag takes this branch and the hold always wins, it is
                // half the gesture's iterations.
                if held_report.is_some()
                    && (reached(positions) <= SATISFIED_RESIDUAL
                        || covered(positions) >= MOST_OF_THE_TRAVEL * asked)
                {
                    held_report
                } else {
                    let (held, held_scalars) = (positions.clone(), scalar_coordinates.clone());
                    *positions = seed;
                    *scalar_coordinates = seed_scalars;
                    let loose_report = gesture(&pulled, self, positions, scalar_coordinates);
                    // Judged on how much of the DRAG survives, not on whether the hold cost
                    // anything. Holding a width costs the lead something almost always — carried to
                    // twelve a slot lands 4.93 short holding its width and 5.00 short giving it up
                    // — so a hold that has to be free never survives its first gesture. Nor can it
                    // be judged on arriving, because the gestures worth arguing about are the ones
                    // where nobody arrives.
                    //
                    // What separates them is the share of the reachable travel the hold gives back,
                    // and the two cases are nowhere near each other: an arc whose end the author
                    // fixed cannot move its center at all with the radius held, giving back ALL of
                    // the 2.67 the drawing would otherwise have covered, while the slot dragged 163
                    // out gives back 5.2 of 155.3. A hundred percent against three, and the suite
                    // is unchanged at a tenth and at nine tenths.
                    if held_report.is_some()
                        && covered(&held) >= MOST_OF_THE_TRAVEL * covered(positions)
                    {
                        *positions = held;
                        *scalar_coordinates = held_scalars;
                        held_report
                    } else {
                        loose_report
                    }
                }
            }
        };
        Ok(Frame { report, kept })
    }

    /// Every radius a snapped gesture DRAGS BUT DOES NOT AUTHOR, at what the opening measured.
    ///
    /// A drawing that does not dimension its own width has a family of answers under the hand, and
    /// a step wide enough to see the curvature of the circle the snap is turning about lands on the
    /// wrong member of it. Nothing priced leaving that circle, so the error drained into whichever
    /// quantity nothing else was pricing: swept 160 degrees, a slot four units wide came back drawn
    /// seven and a half, and FINER steps only slowed the drift down. Held here it comes back four,
    /// at every angle from twenty degrees to a hundred and eighty.
    ///
    /// The column is taken away rather than bid for, which is the same choice and the same reason
    /// as [`Self::quantities_a_carry_holds_still`]: a bid in this system is traded against every
    /// other bid in the same sum, and this one always loses.
    ///
    /// Two conditions, and both were measured rather than reasoned. The arc must not be one the
    /// hand is RESHAPING, or a gesture whose whole meaning is the radius — a rim drag, a cap pulled
    /// out to lengthen a slot — is refused; that alone was twenty-six failures. And it must share a
    /// point with something the hand does reshape, because an arc the gesture never reaches has its
    /// own relations to answer to: holding a symmetric pair's far arc left the pair with two mirror
    /// answers and the walk jumped four units between them at an ordinary step.
    fn radii_a_snapped_gesture_keeps(
        &self,
        loosened: &[SketchCurve],
        opening: &[[f64; 2]],
    ) -> Vec<(ParameterId, f64)> {
        // Both ends, because which one is called `from` is a label — see the arc arm of
        // [`curve_geometry`], where reading one of them let a wind change the equations. The
        // opening is a converged answer, so the two agree to the solve's own tolerance and this
        // was expected to say nothing; measured, it says nothing.
        let drawn = |arc: &ArcCenter| {
            let hub = opening.get(arc.center.index)?;
            let reach = |end: &[f64; 2]| (end[0] - hub[0]).hypot(end[1] - hub[1]);
            Some(f64::midpoint(
                reach(opening.get(arc.from.index)?),
                reach(opening.get(arc.to.index)?),
            ))
        };
        let mut authored: Vec<PointId> = Vec::new();
        for curve in loosened {
            match *curve {
                SketchCurve::Segment(key) => {
                    if let Some(segment) = self.segments.get(key.index) {
                        authored.extend([segment.from, segment.to]);
                    }
                }
                SketchCurve::Arc(key) => {
                    if let Some(arc) = self.arc_centers.iter().find(|arc| arc.key == key) {
                        authored.extend([arc.center, arc.from, arc.to]);
                    }
                }
                SketchCurve::Circle(_) => {}
            }
        }
        self.arc_centers
            .iter()
            .filter(|arc| !loosened.contains(&SketchCurve::Arc(arc.key)))
            .filter(|arc| {
                [arc.center, arc.from, arc.to]
                    .iter()
                    .any(|point| authored.contains(point))
            })
            .filter_map(|arc| Some((arc.radius?, drawn(arc)?)))
            .collect()
    }

    /// This problem with every quantity the hands CARRY put beyond the solve's reach, or `None`
    /// when they carry none and the problem is already the right one to hand on.
    ///
    /// Two quantities, and the drawing gives them to the solve differently. An arc's radius is a
    /// COLUMN, so holding it means deleting the column and letting the arc's own equal-radius rows
    /// carry its ends. A segment's length is not a column at all — it is whatever its two ends
    /// happen to be apart — so holding it means ADDING the row that says so. A circle needs
    /// neither: its radius is authored rather than derived from points a hand can move, so nothing
    /// can spend it cheaply, and the preference pass already keeps it without ever being skipped.
    /// Take that preference away and a circle slot cannot even be BUILT — the first tangency comes
    /// back `Degenerate` — which is the measurement that says the circle is already looked after.
    ///
    /// Taking the column away rather than bidding for it, because a bid in this system is traded
    /// against every other bid in the same sum and this one always loses. A slot's cap is tangent
    /// to two rails; pull the cap's center a long way and the arithmetic can either turn the rails,
    /// which moves four points, or widen the cap, which moves two — so it widens the cap. Held as a
    /// preference the cap went 7.33 to 0.05 across the drag that prompted this, against 7.33 to
    /// 0.05 holding nothing: the preference bought a thousandth. With no column there is nothing to
    /// spend, and the arc's own equal-radius rows carry its ends along instead.
    ///
    /// It is NOT rigidity, and the difference is which pass sees it. This applies to the hand's
    /// problem only — the two passes where a preference already outranks the cursor — and never to
    /// the drawing's own, which runs last and alone and has the final word. So an undimensioned
    /// radius holds against the gesture and yields to anything the author actually asserted: put a
    /// `Distance` or a radius on that arc and the last pass enforces it over this, exactly as it
    /// already does over every span preference in the pass before.
    ///
    /// A carry is told from an authoring gesture by what the hands DECLARE, never by measuring
    /// where they went (ADR 0042). Two gestures hold an arc entire and they are opposites, so
    /// counting hands cannot separate them — the ROLE of the hand on the center does. Widening a
    /// curve by its body PINS the center and carries the two ends outward along their own radii:
    /// that gesture's whole meaning is the radius, and holding it makes the drawing travel instead
    /// of widen. A carry LEADS or CARRIES the center along with the ends, and then nobody is
    /// authoring the radius at all.
    ///
    /// Measuring instead of asking was tried and cannot work: a point's fraction is stored in an
    /// `f32`, so the radius a carried point reconstructs to disagrees with the one it was carried
    /// from by more than any threshold worth setting, and by an amount that grows with the
    /// drawing's magnitude — a fourfold zoom changed which gesture the arithmetic thought it was
    /// looking at.
    fn quantities_a_carry_holds_still(&self, hands: &[Hand], opening: &[[f64; 2]]) -> Option<Self> {
        /// Whether a segment carried entire keeps the length it was drawn. See where it is read.
        const HOLD_A_CARRIED_SPAN: bool = false;
        let reaching = |role: HandRole| -> Vec<PointId> {
            hands
                .iter()
                .filter(|hand| hand.role == role)
                .flat_map(|hand| self.standing_together(hand.point))
                .collect()
        };
        let in_hand: Vec<PointId> = hands
            .iter()
            .flat_map(|hand| self.standing_together(hand.point))
            .collect();
        let pinned = reaching(HandRole::Pin);
        // Measured at the OPENING, never off the column. By the time this pass runs, two passes
        // have already dragged the drawing about and the column has followed them — on the slot
        // that prompted this the far cap's went 7.3271 to 0.6141 before the drawing was ever asked.
        // The radius the author drew is the only one worth keeping, and `opening` is where it is.
        // Both ends, because which one is called `from` is a label — see the arc arm of
        // [`curve_geometry`], where reading one of them let a wind change the equations. The
        // opening is a converged answer, so the two agree to the solve's own tolerance and this
        // was expected to say nothing; measured, it says nothing.
        let drawn = |arc: &ArcCenter| {
            let hub = opening.get(arc.center.index)?;
            let reach = |end: &[f64; 2]| (end[0] - hub[0]).hypot(end[1] - hub[1]);
            Some(f64::midpoint(
                reach(opening.get(arc.from.index)?),
                reach(opening.get(arc.to.index)?),
            ))
        };
        // A segment carried entire keeps the length it was drawn. Unlike the arc there is no
        // gesture on the other side of this: a segment dragged by its BODY slides sideways, both
        // ends by the same offset, so its length was never what that gesture was setting. Measured
        // on a straight slot's rail, sliding it out drifted 24.0000 to 23.7125 and 24.3920 in
        // different directions for the same shape — small against the arc's 7.33 to 0.05, and the
        // same undimensioned quantity going wherever the arithmetic left it.
        //
        // TEMPORARILY OFF at the owner's request (2026-08-07) while the arc's hold is tried in the
        // app. Flip `HOLD_A_CARRIED_SPAN` back to `true` to restore it — nothing else is stubbed,
        // and `sliding_a_slots_rail_keeps_the_length_it_was_drawn` is `#[ignore]`d to match.
        let spans: Vec<(PointId, PointId, f64)> = self
            .segments
            .iter()
            .filter(|_| HOLD_A_CARRIED_SPAN)
            .filter(|segment| {
                [segment.from, segment.to]
                    .iter()
                    .all(|point| in_hand.contains(point))
            })
            .filter(|segment| {
                ![segment.from, segment.to]
                    .iter()
                    .any(|point| pinned.contains(point))
            })
            .filter_map(|segment| {
                let tail = opening.get(segment.from.index)?;
                let head = opening.get(segment.to.index)?;
                Some((
                    segment.from,
                    segment.to,
                    (head[0] - tail[0]).hypot(head[1] - tail[1]),
                ))
            })
            .collect();
        let carried: Vec<(ParameterId, f64)> = self
            .arc_centers
            .iter()
            .filter(|arc| {
                [arc.center, arc.from, arc.to]
                    .iter()
                    .all(|point| in_hand.contains(point))
            })
            .filter(|arc| !pinned.contains(&arc.center))
            .filter_map(|arc| Some((arc.radius?, drawn(arc)?)))
            .collect();
        if carried.is_empty() && spans.is_empty() {
            return None;
        }
        let mut held = self.clone();
        for (radius, at) in carried {
            if let Some(parameter) = held.parameters.get_mut(radius.index) {
                parameter.stored = at;
                parameter.free = false;
            }
        }
        for (from, to, length) in spans {
            let span = Relation::Distance { from, to, length };
            if let Ok(resolved) = held.resolve(span) {
                held = held.with_candidate(span, resolved);
            }
        }
        Some(held)
    }

    /// The points pinned outright, which a shape-holding preference has to leave where they are.
    fn points_the_author_fixed(&self) -> Vec<PointId> {
        self.constraints
            .iter()
            .filter_map(|constraint| match constraint.relation {
                Relation::Fix { point, .. } => Some(point),
                _ => None,
            })
            .collect()
    }

    fn settle_with(&self, anchored: &[PointId], flexible_curves: &[SketchCurve]) -> Settled {
        let mut positions: Vec<_> = self.points.iter().map(|point| point.at).collect();
        let mut scalar_coordinates = self.scalar_coordinates();
        let preferred_trace = run_reporting_only_the_search(
            self,
            &mut positions,
            &mut scalar_coordinates,
            Rigidity::Preferred {
                anchored,
                flexible_curves,
                was: &[],
                opening: &[],
                reshaping: false,
            },
        );
        let preferred_report = preferred_trace
            .and_then(|trace| exact_report_at(self, &positions, &scalar_coordinates, trace));
        let report =
            if preferred_report.is_some_and(|report| report.residual_norm <= SATISFIED_RESIDUAL) {
                preferred_report
            } else {
                run(
                    self,
                    &mut positions,
                    &mut scalar_coordinates,
                    Rigidity::Ignored,
                )
            };
        let solution = self.solution(positions, &scalar_coordinates);
        let diagnostics = diagnostics(self, &solution, report);
        Settled {
            solution,
            diagnostics,
            kept: None,
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
        if let Relation::Symmetry { axis, .. } = relation {
            let Some(axis) = self
                .segments
                .get(axis.index)
                .filter(|_| axis.owner == self.owner)
            else {
                return Vec::new();
            };
            return vec![axis.from, axis.to];
        }
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
            Relation::Fix { point, .. }
            | Relation::Quantize { point, .. }
            | Relation::Midpoint { point, .. } => vec![point],
            Relation::PointOnCurve { point, curve } => std::iter::once(point)
                .chain(self.points_of_curve(curve))
                .collect(),
            // The WHOLE spline, arms included. Its shape is a function of every one of them, so a
            // scope holding the point to a curve while leaving out a fit point three spans away
            // would be holding it to a curve the solve is free to move out from under it.
            Relation::PointOnSpline { point, spline, .. } => std::iter::once(point)
                .chain(self.points_of_spline(spline))
                .collect(),
            Relation::PointLineDistance { point, line, .. } => std::iter::once(point)
                .chain(self.points_of_curve(SketchCurve::Segment(line)))
                .collect(),
            Relation::Distance { from, to, .. } | Relation::AxisDistance { from, to, .. } => {
                vec![from, to]
            }
            Relation::Coincident { first, second } => vec![first, second],
            Relation::TangentDirection {
                joint,
                joint_arm,
                against,
            } => [joint, joint_arm]
                .into_iter()
                .chain(self.points_of_curve(against))
                .collect(),
            Relation::Curvature {
                joint,
                joint_arm,
                neighbor,
                neighbor_arm,
                end: _,
                against,
            } => [joint, joint_arm, neighbor, neighbor_arm]
                .into_iter()
                .chain(self.points_of_curve(against))
                .collect(),
            Relation::Tangent { first, second, .. }
            | Relation::Concentric { first, second }
            | Relation::RimGap { first, second, .. } => [first, second]
                .into_iter()
                .flat_map(|curve| self.points_of_curve(curve))
                .collect(),
            // An arc arm brings the whole arc into the scope, not only the end it reads: the
            // tangent there is measured against the center, and a scope that left the center out
            // would be solving an arm whose direction nothing could change.
            Relation::Angle { first, second, .. } => [first, second]
                .into_iter()
                .flat_map(|arm| self.points_of_curve(arm.curve()))
                .collect(),
            Relation::Symmetry {
                first,
                second,
                axis: _,
                branch: _,
            } => [first, second]
                .into_iter()
                .flat_map(|curve| self.points_of_curve(curve))
                .collect(),
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
            | Relation::PointLineDistance { line: segment, .. }
            | Relation::Midpoint { segment, .. } => vec![segment],
            Relation::Parallel { first, second }
            | Relation::Perpendicular { first, second }
            | Relation::Equal { first, second }
            | Relation::Collinear { first, second } => vec![first, second],
            Relation::Angle { first, second, .. } => [first, second]
                .into_iter()
                .filter_map(AngleArm::segment)
                .collect(),
            Relation::Tangent { first, second, .. } => [first, second]
                .into_iter()
                .filter_map(|curve| match curve {
                    SketchCurve::Segment(segment) => Some(segment),
                    SketchCurve::Arc(_) | SketchCurve::Circle(_) => None,
                })
                .collect(),
            Relation::Symmetry {
                first,
                second,
                axis,
                branch: _,
            } => std::iter::once(axis)
                .chain([first, second].into_iter().filter_map(|curve| match curve {
                    SketchCurve::Segment(segment) => Some(segment),
                    SketchCurve::Arc(_) | SketchCurve::Circle(_) => None,
                }))
                .collect(),
            Relation::PointOnCurve { curve, .. } => match curve {
                SketchCurve::Segment(segment) => vec![segment],
                SketchCurve::Arc(_) | SketchCurve::Circle(_) => Vec::new(),
            },
            Relation::TangentDirection { against, .. } | Relation::Curvature { against, .. } => {
                match against {
                    SketchCurve::Segment(segment) => vec![segment],
                    SketchCurve::Arc(_) | SketchCurve::Circle(_) => Vec::new(),
                }
            }
            // A radius only ever names a curve that turns, so it never names a segment. Nor
            // does the gap between two of them, nor a spline, which is made of points and
            // nothing else.
            Relation::PointOnSpline { .. }
            | Relation::Fix { .. }
            | Relation::Quantize { .. }
            | Relation::Distance { .. }
            | Relation::AxisDistance { .. }
            | Relation::Radius { .. }
            | Relation::RimGap { .. }
            | Relation::Coincident { .. }
            | Relation::Concentric { .. } => Vec::new(),
        }
    }

    /// Every point a spline's shape depends on: the ones it is made of, and the arms steering them.
    fn points_of_spline(&self, spline: SplineId) -> Vec<PointId> {
        self.splines
            .get(spline.index)
            .filter(|held| spline.owner == self.owner && held.key == spline)
            .map(|held| {
                let arms = match &held.form {
                    SplineForm::FitPoint { arms } => arms.clone(),
                    SplineForm::ControlPoint => Vec::new(),
                };
                held.points
                    .iter()
                    .copied()
                    .chain(arms.into_iter().flatten())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn points_of_curve(&self, curve: SketchCurve) -> Vec<PointId> {
        match curve {
            SketchCurve::Segment(segment) => self
                .segments
                .get(segment.index)
                .filter(|_| segment.owner == self.owner)
                .map(|segment| vec![segment.from, segment.to])
                .unwrap_or_default(),
            SketchCurve::Arc(arc) => self
                .arc_centers
                .get(arc.index)
                .filter(|held| arc.owner == self.owner && held.key == arc)
                .map(|arc| vec![arc.from, arc.to, arc.center])
                .unwrap_or_default(),
            SketchCurve::Circle(circle) => self
                .circles
                .get(circle.index)
                .filter(|held| circle.owner == self.owner && held.key == circle)
                .map(|circle| vec![circle.center])
                .unwrap_or_default(),
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
        flexible_curves: &[SketchCurve],
    ) -> Vec<ConstraintId> {
        self.leave_one_out(|without| {
            without
                .with_candidate(relation, resolved)
                .settle_with(anchored, flexible_curves)
                .diagnostics
                .satisfied
        })
    }

    /// Run one deterministic leave-one-out pass over standing constraints. The insertion order is
    /// the authored relation order supplied by the adapter, so callers can map results faithfully.
    fn leave_one_out(
        &self,
        mut restores_satisfaction: impl FnMut(&Self) -> bool,
    ) -> Vec<ConstraintId> {
        self.constraints
            .iter()
            .map(|constraint| constraint.key)
            .filter(|key| {
                let mut without = self.clone();
                without
                    .constraints
                    .retain(|constraint| constraint.key != *key);
                restores_satisfaction(&without)
            })
            .collect()
    }

    /// Reject a newly collapsed segment, arc chord, or arc radius. A singularity can satisfy many
    /// residuals, so this is a property of the result and ignores geometry already degenerate:
    /// generic result-based collapse handles any relation that creates it without maintaining a
    /// list of relation-specific collapse rules.
    fn collapsed_by(&self, solution: &Solution) -> Option<SketchCurve> {
        let before = |point: PointId| self.points[point.index].at;
        let span = |from: [f64; 2], to: [f64; 2]| {
            ((to[0] - from[0]).powi(2) + (to[1] - from[1]).powi(2)).sqrt()
        };
        for (index, segment) in self.segments.iter().enumerate() {
            let after = |point| solution.position(point).unwrap_or_else(|| before(point));
            if span(before(segment.from), before(segment.to)) > COLLAPSED_SPAN
                && span(after(segment.from), after(segment.to)) <= COLLAPSED_SPAN
            {
                return Some(SketchCurve::Segment(SegmentId {
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
                return Some(SketchCurve::Arc(arc.key));
            }
        }
        for circle in &self.circles {
            let before = self.parameters[circle.radius.index].stored;
            let Some(ParameterValue::Radius(after)) = solution.parameter(circle.radius) else {
                continue;
            };
            if before > COLLAPSED_SPAN && after <= COLLAPSED_SPAN {
                return Some(SketchCurve::Circle(circle.key));
            }
        }
        None
    }

    /// Collapse implication is structural, not experimental: a prior solve may have moved the
    /// drawing, so removing one relation no longer reconstructs the geometry it once produced.
    fn constraints_acting_on(&self, curve: SketchCurve) -> Vec<ConstraintId> {
        let points = match curve {
            SketchCurve::Segment(segment) => self
                .segments
                .get(segment.index)
                .map(|segment| vec![segment.from, segment.to]),
            SketchCurve::Arc(arc) => self
                .arc_centers
                .get(arc.index)
                .map(|arc| vec![arc.from, arc.to, arc.center]),
            SketchCurve::Circle(circle) => self
                .circles
                .get(circle.index)
                .map(|circle| vec![circle.center]),
        }
        .unwrap_or_default();
        self.constraints
            .iter()
            .filter(|constraint| {
                Self::named_segments(constraint.relation).iter().any(
                    |segment| matches!(curve, SketchCurve::Segment(curve) if *segment == curve),
                ) || self
                    .named_points(constraint.relation)
                    .iter()
                    .any(|point| points.contains(point))
            })
            .map(|constraint| constraint.key)
            .collect()
    }
}
