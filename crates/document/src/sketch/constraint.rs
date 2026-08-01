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

use super::{Arc, EntityId, Point, Segment, Sketch, SketchLength, SketchPoint, ABSENT_CENTER};
use parametric::sketch::{
    ArcId, ConstraintId, CurveKey, PointId, Problem, ProblemBuilder, Relation, SegmentId,
};

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
}

impl ConstraintKind {
    /// Every point id named directly, for cascade and liveness checks.
    pub(super) fn points(&self) -> Vec<EntityId> {
        match *self {
            Self::Fix { point, .. } | Self::Midpoint { point, .. } => vec![point],
            Self::Distance { from, to, .. } => vec![from, to],
            Self::Coincident { first, second } => vec![first, second],
            Self::Horizontal { .. }
            | Self::Vertical { .. }
            | Self::Parallel { .. }
            | Self::Perpendicular { .. }
            | Self::Equal { .. }
            | Self::Collinear { .. } => Vec::new(),
        }
    }

    /// Whether two persisted assertions make the same claim about the same geometry. Stored values
    /// deliberately do not participate: two Fixes on a point are the same assertion whether or
    /// not their targets agree, because changing a fix is delete-then-add rather than two claims.
    pub fn is_about_the_same_as(&self, other: Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(&other)
            && self.subject() == other.subject()
    }

    /// The comparable subject pair. Symmetric pairs are canonicalized, so Distance A→B is the same
    /// assertion as B→A. Midpoint remains ordered because its point and segment belong to different
    /// entity stores and play different semantic roles.
    fn subject(&self) -> [EntityId; 2] {
        match *self {
            Self::Fix { point, .. } => [point, point],
            Self::Horizontal { segment } | Self::Vertical { segment } => [segment, segment],
            Self::Distance { from, to, .. } => [from.min(to), from.max(to)],
            Self::Coincident { first, second }
            | Self::Parallel { first, second }
            | Self::Perpendicular { first, second }
            | Self::Equal { first, second }
            | Self::Collinear { first, second } => [first.min(second), first.max(second)],
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
            Self::Fix { .. } | Self::Distance { .. } | Self::Coincident { .. } => Vec::new(),
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
    pub kind: ConstraintKind,
    /// Whether the solver found it redundant when it was added — it holds, but adds no
    /// information. Redundancy is sometimes the intent, so it is flagged rather than refused.
    #[serde(default)]
    pub redundant: bool,
}

/// Why a requested assertion cannot be retained by the document — **and what to blame**.
///
/// Every refusal that has a culprit names it. A diagnosis the author cannot act on is barely a
/// diagnosis: “it fights something” leaves them to find the something, and on a drawing carrying
/// twenty assertions that is the whole of the work. Since constraints are selectable entities with
/// badges, an id is all the shell needs to point at one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintRefusal {
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
            Self::UnknownEntity | Self::Impossible => Vec::new(),
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
    constraints: Vec<(EntityId, ConstraintId)>,
}

pub(super) enum TrialMapError {
    UnmappedGeometry,
    Request(parametric::sketch::RequestError),
}

impl PreparedProblem {
    pub(super) fn settle(&self) -> parametric::sketch::Settled {
        self.problem.settle()
    }

    pub(super) fn analyze(&self) -> parametric::sketch::Analysis {
        self.problem.analyze()
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
        self.problem
            .drag(self.point(held).expect("live drag point"), at)
    }

    pub(super) fn apply(&self, points: &mut [Point], solution: &parametric::sketch::Solution) {
        for (id, point) in &self.points {
            let Some(at) = solution.position(*point) else {
                continue;
            };
            if let Some(point) = points.iter_mut().find(|point| point.id == *id) {
                point.at = SketchPoint::from_continuous(at[0], at[1]);
            }
        }
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

    pub(super) fn curve(&self, curve: CurveKey) -> Option<EntityId> {
        match curve {
            CurveKey::Segment(key) => self
                .segments
                .iter()
                .find(|(_, local)| *local == key)
                .map(|(stable, _)| *stable),
            CurveKey::Arc(key) => self
                .arcs
                .iter()
                .find(|(_, local)| *local == key)
                .map(|(stable, _)| *stable),
        }
    }

    fn relation(&self, kind: ConstraintKind) -> Option<Relation> {
        relation_for(kind, &self.points, &self.segments)
    }
}

fn relation_for(
    kind: ConstraintKind,
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
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
    match kind {
        ConstraintKind::Fix { point: id, at } => point(id).map(|point| Relation::Fix {
            point,
            at: at.in_plane(),
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
    }
}

fn add_constraints(
    builder: &mut ProblemBuilder,
    constraints: &[Constraint],
    points: &[(EntityId, PointId)],
    segments: &[(EntityId, SegmentId)],
) -> Vec<(EntityId, ConstraintId)> {
    constraints
        .iter()
        .filter_map(|constraint| {
            relation_for(constraint.kind, points, segments)
                .map(|relation| (constraint.id, builder.add_constraint(relation)))
        })
        .collect()
}

/// Build in stable-id order. The parametric kernel intentionally knows no document ids, density,
/// or authored scalar storage; it receives only resolved positions, topology, and relations.
/// Sorting is not a semantic ordering of the document: it gives the local arithmetic layout a
/// reproducible order while stable ids remain the only identity exposed to callers.
pub(super) fn prepare(sketch: &Sketch, constraints: &[Constraint]) -> PreparedProblem {
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
    let segments: Vec<(EntityId, SegmentId)> = ordered_segments
        .into_iter()
        .filter_map(|segment| {
            Some((
                segment.id,
                builder.add_segment(point(segment.from)?, point(segment.to)?),
            ))
        })
        .collect();
    let mut arcs: Vec<&Arc> = sketch.arcs.iter().collect();
    arcs.sort_by_key(|arc| arc.id);
    let mut local_arcs = Vec::new();
    for arc in arcs {
        if arc.center == ABSENT_CENTER {
            continue;
        }
        if let (Some(center), Some(from), Some(to)) =
            (point(arc.center), point(arc.from), point(arc.to))
        {
            local_arcs.push((
                arc.id,
                builder.add_arc_center(center, from, to, arc.bulge.to_degrees_f64()),
            ));
        }
    }

    let local_constraints = add_constraints(&mut builder, constraints, &points, &segments);
    let problem = builder
        .finish()
        .expect("document adapter constructs only live local handles");
    PreparedProblem {
        problem,
        points,
        segments,
        arcs: local_arcs,
        constraints: local_constraints,
    }
}
