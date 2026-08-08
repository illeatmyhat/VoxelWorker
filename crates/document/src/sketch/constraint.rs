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
    Arc, Circle, CircleRadius, EntityId, EntityRole, Hand, Point, Segment, Sketch, SketchLength,
    SketchPoint, Spline,
};
use parametric::sketch::{
    ArcId, BuildError, CircleId, ConstraintId, PointId, Problem, ProblemBuilder, Relation,
    SegmentId, SketchCurve as ParametricSketchCurve, SpanEnd, TangentContactError,
    TangentContactFailure,
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

    /// Whether the relation system can read this KIND of curve as a whole shape.
    ///
    /// This is the ONE statement of which curves Tangent, Symmetry, and point-on-curve can be
    /// about, and it is here rather than restated by each caller that has to turn a pick away.
    /// [`Sketch::curve_geometry`](super::Sketch::curve_geometry) is the implementation of the same
    /// fact; a test holds the two to each other, because a kind that answers yes here and yields
    /// no geometry there is a gesture that completes and then cannot be applied.
    ///
    /// An aggregate answers no because it has no single center, radius, or direction — the
    /// relations resolve their spans individually, and which span the author meant is not
    /// something the identity carries.
    pub const fn carries_relation_geometry(self) -> bool {
        matches!(self, Self::Segment(_) | Self::Arc(_) | Self::Circle(_))
    }

    /// Whether this curve is a constant radius about a center, which is what Concentric needs on
    /// both sides and what makes an arc and a circle interchangeable to it. An ellipse has a
    /// center too, and is still not this: its radius is not one number.
    pub const fn is_circular(self) -> bool {
        matches!(self, Self::Arc(_) | Self::Circle(_))
    }

    /// Whether two identities name the same KIND of curve — what Symmetry means by mirroring like
    /// onto like, and the reason its second pick narrows once the first one lands.
    pub const fn same_kind_as(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Segment(_), Self::Segment(_))
                | (Self::Arc(_), Self::Arc(_))
                | (Self::Circle(_), Self::Circle(_))
                | (Self::Bezier(_), Self::Bezier(_))
                | (Self::Ellipse(_), Self::Ellipse(_))
                | (Self::Conic(_), Self::Conic(_))
                | (Self::Spline(_), Self::Spline(_))
        )
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
/// Which of an arc's two ends an angle is read at.
///
/// The arc's own vocabulary: an arc is stored as the two points it is drawn between and the point
/// it turns about, so these are those two points and not a start-and-finish imposed on top.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArcEnd {
    From,
    To,
}

/// One side of a stated angle: something the drawing gives a DIRECTION for.
///
/// A straight curve has one direction everywhere on it, so a segment arm names no place. A curve
/// that turns has a different direction at every point, so an angle to one is not a question until
/// a place is named — and what an arc arm names is an END, because an end is on its own arc by
/// construction. Naming a free point instead would put a coincidence between the arm and the curve
/// that every later solve would have to keep agreeing about.
///
/// A whole circle has no ends and so cannot be an arm. What an author wants there is a tangency,
/// which [`ConstraintKind::Tangent`] already states at a contact the drawing finds for itself.
///
/// Neither arm carries a SENSE and neither needs one: the solver's row is a sine, which repeats
/// every half turn, so reading an arm the other way round changes nothing it asserts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AngleArm {
    /// A straight curve, read along the direction it was drawn in.
    Segment { segment: EntityId },
    /// The direction an arc leaves at one of its own ends.
    ArcEnd { arc: EntityId, end: ArcEnd },
}

impl AngleArm {
    /// The entity this arm reads, whichever kind it is.
    pub const fn entity(self) -> EntityId {
        match self {
            Self::Segment { segment } => segment,
            Self::ArcEnd { arc, .. } => arc,
        }
    }

    /// The segment this arm reads, for the cascade and liveness checks that are about segments.
    pub const fn segment(self) -> Option<EntityId> {
        match self {
            Self::Segment { segment } => Some(segment),
            Self::ArcEnd { .. } => None,
        }
    }
}

/// A quantity the AUTHOR states, which the drawing then has to honour.
///
/// One family, three members, because a dimension is a single idea the author has — "this is how
/// big it is" — and the geometry it is stated about is what varies. What does not vary is that the
/// value is authored: it keeps its exact measurement, it survives a density re-target, and it
/// outranks anything the solve merely prefers.
///
/// **Each member carries its own STATICALLY TYPED value** — a span and a radius take a
/// [`SketchLength`], an angle takes a [`parametric::units::AngleMeasurement`], and mixing them
/// does not compile.
/// That is ADR 0035 decision 12: the dynamic `Quantity { value, dimension }` belongs to the
/// expression evaluator, and document fields sit statically typed above it.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Dimension {
    /// Two points stand a given distance apart. The length is authored, so it keeps its
    /// [`SketchLength`] and survives a density re-target like every other authored quantity.
    Span {
        from: EntityId,
        to: EntityId,
        length: SketchLength,
    },
    /// Two segments meet at this angle, measured turning from `first` to `second`.
    ///
    /// The one member whose value is an [`parametric::units::AngleMeasurement`] rather than a
    /// [`SketchLength`], which is the whole reason this family is a family: the author is stating
    /// how big something is either way, and only the kind of quantity differs. It carries no
    /// density and so survives a re-target untouched.
    ///
    /// `Parallel` and `Perpendicular` state the two angles that need no number. They stay their
    /// own kinds — an author asking for a right angle is not authoring the number 90, and a badge
    /// says that where a dimension line cannot.
    ///
    /// A stated angle and that angle plus a half turn are the same claim, because a segment has
    /// two ends and the drawing has no opinion about which one it points from.
    Angle {
        first: AngleArm,
        second: AngleArm,
        degrees: parametric::units::AngleMeasurement,
    },
    /// A curve that turns stands this far from its own center, everywhere.
    ///
    /// One member for the arc and the circle both, because it is one statement about one shape:
    /// an arc is a circle with two ends, and how big it is does not depend on where they are. The
    /// solver holds a radius column for each — the circle's authored, the arc's minted beside its
    /// three points — so this reads the same row against either.
    ///
    /// A straight curve is refused when the relation is added, not here.
    Radius {
        curve: SketchCurve,
        length: SketchLength,
    },
    /// A curve that turns is this wide, measured straight through its own center.
    ///
    /// The SAME claim as [`Radius`](Self::Radius) doubled, and the solver reads it as exactly that
    /// — one radius row against the same column. It is a separate member and not a display flag on
    /// the radius because the number is AUTHORED: an author who wrote "one block across" wrote a
    /// diameter, and storing half of it would throw away the expression they typed and hand back a
    /// halved one at the next density re-target.
    ///
    /// The two cannot both be asserted about one curve. They say the same thing, so the second is
    /// refused as already asserted — which falls out of the subject pair rather than being checked
    /// for, because both name the curve and nothing else.
    Diameter {
        curve: SketchCurve,
        length: SketchLength,
    },
}

impl Dimension {
    /// The authored value, as a length, for the members that measure one.
    pub const fn length(&self) -> Option<SketchLength> {
        match *self {
            Self::Span { length, .. }
            | Self::Radius { length, .. }
            | Self::Diameter { length, .. } => Some(length),
            // An angle is a quantity, but it is not a length, and there is no length to answer
            // with rather than one that happens to be zero.
            Self::Angle { .. } => None,
        }
    }
}

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
    /// An authored quantity the drawing must honour — see [`Dimension`].
    ///
    /// Nested rather than spread across the kinds beside it, so "is this a dimension?" has one
    /// answer. A dimension is the one family that draws as a NUMBER instead of a badge, and both
    /// the gizmos and the panel that lay those numbers out need to ask that question without
    /// naming every member.
    Dimension(Dimension),
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
    /// A point lies on a curve's SUPPORT — the infinite line a segment runs along, or the whole
    /// circle an arc is cut from. One residual: the point keeps its freedom to slide.
    ///
    /// This is what Fusion spells as a Coincident between a point and a curve. It is a separate
    /// kind here rather than a polymorphic `Coincident` because the two assert different things:
    /// two points occupying one place pins both coordinates, and a point on a line pins one.
    PointOnCurve { point: EntityId, curve: SketchCurve },
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
    /// A fit-point spline runs smoothly out of `against` at `joint`: same direction there, and the
    /// same curvature — G2.
    ///
    /// Only the joint and the neighbour are stored. The rest of what the relation needs — the
    /// joint's arm, the next fit point along, that point's arm, and which end of its span the joint
    /// stands at — is DERIVED from the spline when the problem is built, because storing it would
    /// be a second copy of the spline's own structure, free to go stale the moment a point is
    /// inserted beside the joint.
    Curvature {
        joint: EntityId,
        against: SketchCurve,
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
    /// Every point this relation ties to SURVIVING GEOMETRY, rather than merely mentions.
    ///
    /// The distinction decides whether an orphan sweep may take the point. A circle's center is
    /// structural — it anchors the circle and nothing else, so `Fix`ing it must not outlive the
    /// circle. A center rectangle's center is authored: `Midpoint` holds it on a diagonal that is
    /// still there, and that diagonal is the reference. Relations between two points, or between
    /// two curves, anchor nothing — a point they name is held up by whatever draws it, or by
    /// nothing at all.
    pub(super) fn anchored_points(&self) -> Vec<EntityId> {
        match *self {
            // Both hold a point up against geometry that is still drawn: the midpoint of a
            // diagonal, or the arc center a slot's spine runs through.
            Self::Midpoint { point, .. } | Self::PointOnCurve { point, .. } => vec![point],
            // The joint is a spline's own fit point: the spline holds it up, not this relation.
            Self::Curvature { .. }
            | Self::Fix { .. }
            | Self::Quantize { .. }
            | Self::Dimension(_)
            | Self::Coincident { .. }
            | Self::Horizontal { .. }
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

    /// Every point id named directly, for cascade and liveness checks.
    pub(super) fn points(&self) -> Vec<EntityId> {
        match *self {
            Self::Fix { point, .. }
            | Self::Quantize { point, .. }
            | Self::Midpoint { point, .. }
            | Self::PointOnCurve { point, .. } => vec![point],
            Self::Curvature { joint, .. } => vec![joint],
            Self::Dimension(Dimension::Span { from, to, .. }) => vec![from, to],
            Self::Coincident { first, second } => vec![first, second],
            // Both name curves. The points those curves are drawn between are not what
            // either one is about.
            Self::Dimension(
                Dimension::Radius { .. } | Dimension::Diameter { .. } | Dimension::Angle { .. },
            )
            | Self::Horizontal { .. }
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
        // An angle is the second relation whose stored values participate, and for Symmetry's
        // reason: the two arms are the claim. Two angles struck at OPPOSITE ends of one arc against
        // one line are different statements about different tangents, and an id pair alone cannot
        // tell them apart because it holds no room for the end.
        if let (
            Self::Dimension(Dimension::Angle { first, second, .. }),
            Self::Dimension(Dimension::Angle {
                first: other_first,
                second: other_second,
                ..
            }),
        ) = (*self, other)
        {
            // A TOTAL key, so the two ends of one arc do not tie and get ordered by luck.
            let key = |arm: AngleArm| match arm {
                AngleArm::Segment { segment } => (segment, 0_u8),
                AngleArm::ArcEnd {
                    arc,
                    end: ArcEnd::From,
                } => (arc, 1),
                AngleArm::ArcEnd {
                    arc,
                    end: ArcEnd::To,
                } => (arc, 2),
            };
            let pair = |one, another| {
                let mut both = [key(one), key(another)];
                both.sort_unstable();
                both
            };
            return pair(first, second) == pair(other_first, other_second);
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
            Self::Dimension(Dimension::Span { from, to, .. }) => [from.min(to), from.max(to)],
            // Sharing the pair is what refuses a diameter on a curve that already states a
            // radius: it is the same claim written the other way, not a second one.
            Self::Dimension(
                Dimension::Radius { curve, .. } | Dimension::Diameter { curve, .. },
            ) => [curve.id(), curve.id()],
            // Reached only by a caller that is not `is_about_the_same_as`, which answers an
            // angle from its arms before it ever gets here.
            Self::Dimension(Dimension::Angle { first, second, .. }) => {
                let (first, second) = (first.entity(), second.entity());
                [first.min(second), first.max(second)]
            }
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
            Self::PointOnCurve { point, curve } => [point, curve.id()],
            Self::Curvature { joint, against } => [joint, against.id()],
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
            // Only the straight arms are segments. An arc arm names an arc, and an arc that goes
            // takes the constraint by a different cascade.
            Self::Dimension(Dimension::Angle { first, second, .. }) => [first, second]
                .into_iter()
                .filter_map(AngleArm::segment)
                .collect(),
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
            Self::PointOnCurve { curve, .. } => match curve {
                SketchCurve::Segment(id) => vec![id],
                SketchCurve::Arc(_)
                | SketchCurve::Circle(_)
                | SketchCurve::Bezier(_)
                | SketchCurve::Ellipse(_)
                | SketchCurve::Conic(_)
                | SketchCurve::Spline(_) => Vec::new(),
            },
            Self::Curvature { against, .. } => match against {
                SketchCurve::Segment(id) => vec![id],
                SketchCurve::Arc(_)
                | SketchCurve::Circle(_)
                | SketchCurve::Bezier(_)
                | SketchCurve::Ellipse(_)
                | SketchCurve::Conic(_)
                | SketchCurve::Spline(_) => Vec::new(),
            },
            Self::Fix { .. }
            | Self::Quantize { .. }
            | Self::Dimension(_)
            | Self::Coincident { .. }
            | Self::Concentric { .. } => Vec::new(),
        }
    }

    /// Every entity id this relation names, whatever store holds it.
    ///
    /// The other accessors here answer "which segments" or "which points" because cascade and
    /// liveness care which store an id came from. Scoping a solve does not: it asks only which
    /// geometry a relation could speak about, and an answer that omitted a kind would silently
    /// cut a shape in half. One arm per relation, so a new one cannot answer by omission.
    pub(super) fn named_entities(&self) -> Vec<EntityId> {
        match *self {
            Self::Fix { point, .. } | Self::Quantize { point, .. } => vec![point],
            Self::Horizontal { segment } | Self::Vertical { segment } => vec![segment],
            Self::Dimension(Dimension::Span { from, to, .. }) => vec![from, to],
            Self::Dimension(
                Dimension::Radius { curve, .. } | Dimension::Diameter { curve, .. },
            ) => vec![curve.id()],
            Self::Dimension(Dimension::Angle { first, second, .. }) => {
                vec![first.entity(), second.entity()]
            }
            Self::Coincident { first, second }
            | Self::Parallel { first, second }
            | Self::Perpendicular { first, second }
            | Self::Equal { first, second }
            | Self::Collinear { first, second } => vec![first, second],
            Self::Midpoint { point, segment } => vec![point, segment],
            Self::PointOnCurve { point, curve } => vec![point, curve.id()],
            Self::Curvature { joint, against } => vec![joint, against.id()],
            Self::Tangent { first, second, .. } | Self::Concentric { first, second } => {
                vec![first.id(), second.id()]
            }
            Self::Symmetry {
                first,
                second,
                axis,
                ..
            } => vec![first.id(), second.id(), axis],
        }
    }

    /// Every curve id named by a generic curve relation, for cascade/repair.
    pub(super) fn curves(&self) -> Vec<SketchCurve> {
        match *self {
            Self::Tangent { first, second, .. } | Self::Concentric { first, second } => {
                vec![first, second]
            }
            Self::Symmetry { first, second, .. } => vec![first, second],
            Self::Curvature { against, .. } => vec![against],
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
    /// A curvature relation was asked for where there is no joint to speak of: the point is not
    /// the free END of an open fit-point spline, or it does not stand on the curve it is meant to
    /// run out of. Curvature between things that do not meet is not a question with an answer.
    CurvatureNeedsAJoint,
    /// It names the BACK arm of a tangent lever, whose position is re-derived as the mirror of
    /// the forward arm after every edit. A relation on it would be met by the solve and then
    /// silently overwritten, which is worse than declining it.
    MirroredTangentArm,
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
            | Self::MirroredTangentArm
            | Self::CurvatureNeedsAJoint
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
    arcs: Vec<(EntityId, ArcId)>,
    circles: Vec<(EntityId, CircleId, parametric::sketch::ParameterId)>,
    constraints: Vec<(EntityId, ConstraintId)>,
    /// Kept whole because a curvature relation reads its span out of the spline rather than out of
    /// stored fields; a trial has to be able to make the same reading the build made.
    splines: Box<[Spline]>,
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
    RadiusNotRepresentable,
    MissingDocumentEntity,
}

pub(super) struct ApplyPlan {
    points: Vec<Point>,
    circles: Vec<Circle>,
}

impl ApplyPlan {
    pub(super) fn apply(self, sketch: &mut Sketch) {
        sketch.points = self.points;
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

    /// The same prepared problem, holding a snap within `reach`.
    /// See [`parametric::sketch::SnapReach`].
    pub(super) fn holding_a_snap_within(mut self, reach: parametric::sketch::SnapReach) -> Self {
        self.problem = self.problem.holding_a_snap_within(reach);
        self
    }

    /// Snap a drag's hands without solving.
    /// See [`parametric::sketch::Problem::snap_the_hands`].
    pub(super) fn snap_the_hands(
        &self,
        hands: &[Hand],
        was: &[(EntityId, [f64; 2])],
    ) -> Option<(Vec<Hand>, parametric::sketch::KeptQuantity)> {
        let pulling: Vec<parametric::sketch::Hand> = hands
            .iter()
            .map(|hand| {
                self.point(hand.point)
                    .map(|point| parametric::sketch::Hand {
                        point,
                        to: hand.to,
                        role: hand.role,
                    })
            })
            .collect::<Option<Vec<_>>>()?;
        // A point the prepared problem does not carry is DROPPED, not a failure. What arrives is
        // the drawing as the gesture found it, whole, while the problem is scoped to the part the
        // drag can reach; refusing the ones outside that scope would refuse every snap on a plane
        // with a second shape on it.
        let stood: Vec<(parametric::sketch::PointId, [f64; 2])> = was
            .iter()
            .filter_map(|(held, at)| self.point(*held).map(|point| (point, *at)))
            .collect();
        let (snapped, kept) = self.problem.snap_the_hands(&pulling, &stood)?;
        // Back into the document's names. A hand whose point does not map is dropped rather than
        // guessed at, which leaves it where the caller put it.
        let named: Vec<Hand> = snapped
            .iter()
            .filter_map(|hand| {
                let id = self
                    .points
                    .iter()
                    .find(|(_, local)| *local == hand.point)
                    .map(|(id, _)| *id)?;
                Some(Hand {
                    point: id,
                    to: hand.to,
                    role: hand.role,
                })
            })
            .collect();
        Some((named, kept))
    }

    /// Pull one or more points at once. See [`parametric::sketch::Problem::drag_together`].
    pub(super) fn drag_together(
        &self,
        hands: &[Hand],
        was: &[(EntityId, [f64; 2])],
    ) -> Result<parametric::sketch::DragOutcome, parametric::sketch::RequestError> {
        let pulling = hands
            .iter()
            .map(|hand| {
                self.point(hand.point)
                    .map(|point| parametric::sketch::Hand {
                        point,
                        to: hand.to,
                        role: hand.role,
                    })
                    .ok_or(parametric::sketch::RequestError::UnknownPoint)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Out of scope is not unknown. `was` is the whole pre-drag drawing and the problem is only
        // the reachable part of it, so a point the problem does not carry is one the solve could
        // not have consulted anyway.
        let stood: Vec<_> = was
            .iter()
            .filter_map(|(held, at)| self.point(*held).map(|point| (point, *at)))
            .collect();
        self.problem.drag_together(&pulling, &stood)
    }

    /// An arc takes no part here. Its shape is its three placed points (ADR 0038), and those
    /// have already been written back with every other point above.
    pub(super) fn plan_apply(
        &self,
        points: &[Point],
        circles: &[Circle],
        solution: &parametric::sketch::Solution,
    ) -> Result<ApplyPlan, ScalarWritebackError> {
        let mut points = points.to_vec();
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
                .ok_or(ScalarWritebackError::MissingSolutionParameter)?;
            let value = super::ResolvedLength::try_from_f64(value)
                .map_err(|_| ScalarWritebackError::RadiusNotRepresentable)?;
            circle.radius = CircleRadius::free(value);
        }
        Ok(ApplyPlan { points, circles })
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
                .find(|(_, local)| *local == key)
                .map(|(stable, _)| *stable),
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
            &self.splines,
        )
    }
}

#[allow(clippy::too_many_lines)]
fn relation_for(
    kind: ConstraintKind,
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
    arcs: &[(EntityId, ArcId)],
    circles: &[(EntityId, CircleId, parametric::sketch::ParameterId)],
    splines: &[Spline],
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
            .find(|(candidate, _)| *candidate == id)
            .map(|(_, local)| ParametricSketchCurve::Arc(*local)),
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
        ConstraintKind::Dimension(Dimension::Span { from, to, length }) => point(from)
            .zip(point(to))
            .map(|(from, to)| Relation::Distance {
                from,
                to,
                length: length.value(),
            }),
        ConstraintKind::Dimension(Dimension::Radius {
            curve: subject,
            length,
        }) => curve(subject).map(|curve| Relation::Radius {
            curve,
            length: length.value(),
        }),
        // Halved into the one radius relation the solver has, because a diameter is not a
        // different measurement of the shape — it is the same one stated at twice the size, and a
        // second row that said so would be a second way for the two to disagree.
        ConstraintKind::Dimension(Dimension::Diameter {
            curve: subject,
            length,
        }) => curve(subject).map(|curve| Relation::Radius {
            curve,
            length: length.value() / 2.0,
        }),
        ConstraintKind::Dimension(Dimension::Angle {
            first,
            second,
            degrees,
        }) => {
            let arm = |arm: AngleArm| match arm {
                AngleArm::Segment { segment: id } => {
                    segment(id).map(parametric::sketch::AngleArm::Segment)
                }
                AngleArm::ArcEnd { arc: id, end } => arcs
                    .iter()
                    .find(|(candidate, _)| *candidate == id)
                    .map(|(_, local)| parametric::sketch::AngleArm::ArcEnd {
                        arc: *local,
                        end: match end {
                            ArcEnd::From => SpanEnd::Start,
                            ArcEnd::To => SpanEnd::Finish,
                        },
                    }),
            };
            arm(first)
                .zip(arm(second))
                .map(|(first, second)| Relation::Angle {
                    first,
                    second,
                    radians: degrees.to_degrees_f64().to_radians(),
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
        ConstraintKind::PointOnCurve {
            point: id,
            curve: subject,
        } => point(id)
            .zip(curve(subject))
            .map(|(point, curve)| Relation::PointOnCurve { point, curve }),
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
        ConstraintKind::Curvature { joint, against } => {
            let (joint_arm, neighbor, neighbor_arm, end) = curvature_span_of(splines, joint)?;
            Some(Relation::Curvature {
                joint: point(joint)?,
                joint_arm: point(joint_arm)?,
                neighbor: point(neighbor)?,
                neighbor_arm: point(neighbor_arm)?,
                end,
                against: curve(against)?,
            })
        }
    }
}

/// The span a curvature relation reads, derived from the spline whose END the joint is.
///
/// Answers `(joint arm, neighbour, neighbour arm, which end)`. Derived rather than stored so that
/// inserting a point beside the joint cannot leave the relation reading a span that is no longer
/// there — see [`ConstraintKind::Curvature`].
///
/// A closed spline has no end, and a spline of one point has no span, so neither answers.
pub(super) fn curvature_span_of(
    splines: &[Spline],
    joint: EntityId,
) -> Option<(EntityId, EntityId, EntityId, SpanEnd)> {
    splines.iter().find_map(|spline| {
        if spline.closed || spline.points.len() < 2 {
            return None;
        }
        let last = spline.points.len() - 1;
        let (neighbor, end) = match spline.points.iter().position(|id| *id == joint)? {
            0 => (spline.points[1], SpanEnd::Start),
            index if index == last => (spline.points[last - 1], SpanEnd::Finish),
            _ => return None,
        };
        Some((
            spline.tangents.get(&joint)?.forward,
            neighbor,
            spline.tangents.get(&neighbor)?.forward,
            end,
        ))
    })
}

fn add_constraints(
    builder: &mut ProblemBuilder,
    constraints: &[Constraint],
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
    arcs: &[(EntityId, ArcId)],
    circles: &[(EntityId, CircleId, parametric::sketch::ParameterId)],
    splines: &[Spline],
) -> Vec<(EntityId, ConstraintId)> {
    constraints
        .iter()
        .filter_map(|constraint| {
            relation_for(constraint.kind, points, segments, arcs, circles, splines)
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
    prepare_scoped(sketch, constraints, context, None)
}

/// Build the same problem over a NAMED SET of points rather than the whole drawing.
///
/// The kernel's cost is set by how big the problem is, not by how much of it the author is
/// touching: every free coordinate is a Jacobian column, every drawn edge is a rigidity row, and
/// the dense linear algebra over them grows faster than the drawing does. Two shapes with no
/// relation between them cannot influence one another whatever the solver does, so putting both
/// in one system buys nothing and costs the difference between `n³` and two of `(n/2)³`.
///
/// `scope` names the points that may take part; a curve joins when both its ends do. Everything
/// left out simply is not in the problem, so it cannot move and is not written back — which is
/// exactly the guarantee that made leaving it out safe.
pub(super) fn prepare_scoped(
    sketch: &Sketch,
    constraints: &[Constraint],
    context: Option<EvaluationContext>,
    scope: Option<&[EntityId]>,
) -> Result<PreparedProblem, PrepareError> {
    let in_scope = |id: EntityId| scope.is_none_or(|named| named.contains(&id));
    let mut builder = ProblemBuilder::new();
    let mut ordered_points: Vec<&Point> = sketch
        .points
        .iter()
        .filter(|point| in_scope(point.id))
        .collect();
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
        if !in_scope(segment.from) || !in_scope(segment.to) {
            continue;
        }
        let (Some(from), Some(to)) = (point(segment.from), point(segment.to)) else {
            return Err(PrepareError::InvalidDocumentGeometry);
        };
        let local = match segment.role {
            EntityRole::Real => builder.add_segment(from, to),
            EntityRole::Construction => builder.add_scaffolding_segment(from, to),
        };
        segments.push((segment.id, local));
    }
    let mut arcs: Vec<&Arc> = sketch.arcs.iter().collect();
    arcs.sort_by_key(|arc| arc.id);
    let mut local_arcs = Vec::new();
    for arc in arcs {
        if !in_scope(arc.center) || !in_scope(arc.from) || !in_scope(arc.to) {
            continue;
        }
        let (Some(center), Some(from), Some(to)) =
            (point(arc.center), point(arc.from), point(arc.to))
        else {
            return Err(PrepareError::InvalidDocumentGeometry);
        };
        // An arc's RADIUS survives travel around it, so a scaffolding arc still offers one —
        // see `Problem::add_scaffolding_segment` for why only spans are withheld.
        local_arcs.push((arc.id, builder.add_arc(center, from, to)));
    }

    let mut circles: Vec<&Circle> = sketch.circles.iter().collect();
    circles.sort_by_key(|circle| circle.id);
    let mut local_circles = Vec::new();
    for circle in circles {
        if !in_scope(circle.center) {
            continue;
        }
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
        &sketch.splines,
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
        splines: sketch.splines.clone(),
    })
}
