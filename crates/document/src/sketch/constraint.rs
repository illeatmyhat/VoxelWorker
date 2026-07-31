//! Constraints as entities, and the residual system that hands them to the solver
//! (ADR 0035 Decisions 2, 3 and 4).
//!
//! A constraint lives in the same stable-id space as a point or a segment: it is selectable,
//! individually deletable, individually undoable, and the delete cascade reaches it when the
//! geometry it names dies. A side table without ids would reindex on every delete and take undo
//! with it.
//!
//! The solver core is [`substrate::nonlinear_least_squares`] — pure, continuous, and with no
//! density or lattice vocabulary in it. This module is the adapter: it flattens the sketch's points
//! into a parameter vector, writes one residual per constraint, and puts the solved coordinates
//! back. Solved positions stay **authored** state, never `Derived` (ADR 0022): the solver reads
//! them as its initial guess and writes them back, and an under-constrained sketch has free degrees
//! of freedom that only the stored position remembers.

use super::{EntityId, Point, SketchLength, SketchPoint};
use substrate::nonlinear_least_squares::{
    jacobian, rank, solve, ResidualSystem, SolveReport, SolveSettings,
};

/// What a constraint asserts. Each variant names geometry **by id**, never by index.
///
/// This is the subset ADR 0035 Decision 1 names as the things an author asserts about position
/// directly. Tangent, Perpendicular/Parallel, Equal, Collinear, Midpoint and `Quantize`
/// (Decisions 5 and 14) join it as their residuals are written; the entity, the cascade and the
/// solve path below are the same for all of them.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintKind {
    /// This point does not move, and `at` is where it does not move to.
    ///
    /// The position is stored rather than read from the point at solve time, because a `Fix`
    /// asserts immovability **at a place**: without it, any other constraint that dragged the
    /// point would silently redefine what "fixed" meant.
    Fix { point: EntityId, at: SketchPoint },
    /// The segment's two endpoints share an `axis1` — the segment lies along `axis0`.
    Horizontal { segment: EntityId },
    /// The segment's two endpoints share an `axis0`.
    Vertical { segment: EntityId },
    /// Two points stand a given distance apart. A dimension: the length is authored, so it keeps
    /// its [`SketchLength`] and survives a density re-target like every other authored quantity.
    Distance {
        from: EntityId,
        to: EntityId,
        length: SketchLength,
    },
}

impl ConstraintKind {
    /// Every point id this constraint names directly.
    pub(super) fn points(&self) -> Vec<EntityId> {
        match *self {
            ConstraintKind::Fix { point, .. } => vec![point],
            ConstraintKind::Distance { from, to, .. } => vec![from, to],
            ConstraintKind::Horizontal { .. } | ConstraintKind::Vertical { .. } => Vec::new(),
        }
    }

    /// Whether two constraints make the SAME claim about the SAME geometry — the test behind the
    /// one-of-a-kind-per-entity-set rule (ADR 0035 Decision 4).
    ///
    /// Same claim means same variant, compared through the discriminant so that a variant added
    /// later is covered without an edit here. Same geometry comes from [`Self::subject`], which
    /// every variant answers by construction. The stored VALUES — where a `Fix` fixes to, how far
    /// a `Distance` stands — deliberately play no part: two constraints of one kind on one entity
    /// set are the same assertion whether or not they agree, and if they disagree the answer is
    /// still to replace the first rather than to hold both.
    pub fn is_about_the_same_as(&self, other: ConstraintKind) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(&other)
            && self.subject() == other.subject()
    }

    /// The geometry this constraint is ABOUT, as a comparable pair — a single entity repeated when
    /// it names one, and the pair in a canonical order when it names two (a distance from A to B
    /// is the distance from B to A).
    fn subject(&self) -> [EntityId; 2] {
        match *self {
            ConstraintKind::Fix { point, .. } => [point, point],
            ConstraintKind::Horizontal { segment } | ConstraintKind::Vertical { segment } => {
                [segment, segment]
            }
            ConstraintKind::Distance { from, to, .. } => [from.min(to), from.max(to)],
        }
    }

    /// The segment id this constraint names, if it names one.
    pub(super) fn segment(&self) -> Option<EntityId> {
        match *self {
            ConstraintKind::Horizontal { segment } | ConstraintKind::Vertical { segment } => {
                Some(segment)
            }
            ConstraintKind::Fix { .. } | ConstraintKind::Distance { .. } => None,
        }
    }

    /// How many residuals it contributes. A `Fix` pins two coordinates and so writes two; the
    /// rest write one each.
    fn residual_count(&self) -> usize {
        match *self {
            ConstraintKind::Fix { .. } => 2,
            _ => 1,
        }
    }
}

/// A constraint entity (ADR 0035 Decision 3).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Constraint {
    /// Stable identity, from the same counter as every other entity.
    pub id: EntityId,
    /// What it asserts.
    pub kind: ConstraintKind,
    /// Whether the solver found it redundant when it was added — it holds, but adds no
    /// information (ADR 0035 Decision 4). Redundancy is sometimes the intent, so it is flagged
    /// rather than refused.
    #[serde(default)]
    pub redundant: bool,
}

/// Why a constraint could not be added — **and what to blame** (ADR 0035 Decision 4).
///
/// Every refusal that has a culprit names it. A diagnosis the author cannot act on is barely a
/// diagnosis: "it fights something" leaves them to find the something, and on a drawing carrying
/// twenty assertions that is the whole of the work. Since constraints are selectable entities with
/// badges, an id is all the shell needs to point at one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintRefusal {
    /// It names geometry the store does not hold.
    UnknownEntity,
    /// Its own terms cannot be met by any drawing — a negative distance, a `Horizontal` on a
    /// segment whose ends are the same point. Nothing to blame but the request.
    Impossible,
    /// The system it would join has no solution: it fights what is already asserted.
    Unsatisfiable {
        /// The standing constraints it cannot coexist with, found by leave-one-out (see
        /// `Sketch::blame`). **Empty means undetermined, never innocent** — a conflict that
        /// needs two removals to clear leaves no single culprit, and claiming one would be worse
        /// than admitting none.
        fights: Vec<EntityId>,
    },
    /// The system HAS a solution, and the solution deletes the drawing: this geometry would be
    /// squeezed to nothing. Separated from [`Unsatisfiable`](Self::Unsatisfiable) because it is a
    /// different thing to tell somebody — nothing is fighting, the assertions agree on an answer
    /// that happens to be a singularity.
    WouldCollapse {
        /// The segment or arc that would lose its extent.
        entity: EntityId,
        /// The standing constraints that already act on that geometry.
        ///
        /// Structural rather than experimental, and deliberately so: leave-one-out cannot answer
        /// this one. A previous solve has already MOVED the drawing, and releasing an assertion
        /// does not undo its effect — so dropping the `Horizontal` that levelled a segment leaves
        /// it level, and adding `Vertical` still collapses it. What the author needs is not "which
        /// removal would have helped" but "what else is holding this shape", which is a question
        /// about the constraint graph and always has an answer.
        implicated: Vec<EntityId>,
    },
    /// The same kind of assertion already stands on the same geometry. One constraint of a kind
    /// per entity set: a second `Horizontal` on a segment that is already asserted horizontal says
    /// nothing the first did not, and a second `Fix` on a fixed point is a re-fix, which is a
    /// delete and an add rather than two claims about one place.
    AlreadyAsserted {
        /// The one already standing — so the answer to "you already have this" is a badge lit on
        /// the drawing rather than a hunt.
        existing: EntityId,
    },
}

impl ConstraintRefusal {
    /// Every constraint this refusal blames, for a caller that wants to light them up. Empty when
    /// the refusal has no culprit or none could be isolated.
    pub fn culprits(&self) -> Vec<EntityId> {
        match self {
            ConstraintRefusal::UnknownEntity | ConstraintRefusal::Impossible => Vec::new(),
            ConstraintRefusal::Unsatisfiable { fights } => fights.clone(),
            ConstraintRefusal::WouldCollapse { implicated, .. } => implicated.clone(),
            ConstraintRefusal::AlreadyAsserted { existing } => vec![*existing],
        }
    }
}

/// The sketch's points flattened into a parameter vector, with one entry per constraint residual.
///
/// Parameters are **every** point's two coordinates, not just the constrained ones. That is what
/// makes [`SolveReport::degrees_of_freedom`] mean "how many ways can this drawing still move"
/// rather than "how many ways can the constrained part of it move" — an unconstrained point is a
/// real freedom, and a sketch is fully constrained only when there are none left.
pub(super) struct SketchResiduals<'a> {
    /// The point ids, in parameter order: point `i` owns parameters `2i` and `2i + 1`.
    order: Vec<EntityId>,
    constraints: &'a [Constraint],
    /// Each constraint's endpoints resolved to parameter indices, in the same order as
    /// `constraints`. Resolved once so the residual loop is arithmetic and nothing else.
    resolved: Vec<Resolved>,
}

/// A constraint with its geometry resolved to parameter slots.
#[derive(Debug, Clone, Copy)]
enum Resolved {
    Fix {
        slot: usize,
        at: [f64; 2],
    },
    /// `axis` is the coordinate the two ends must agree on: 1 for horizontal, 0 for vertical.
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
    /// The constraint named geometry that has since gone. It contributes a zero residual rather
    /// than shifting every later constraint's row, and `repair` is what removes it.
    Dangling {
        residuals: usize,
    },
}

impl<'a> SketchResiduals<'a> {
    /// Build the system, or `None` if the sketch holds no points to move.
    pub(super) fn new(
        points: &[Point],
        segments: &[(EntityId, EntityId, EntityId)],
        constraints: &'a [Constraint],
    ) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let order: Vec<EntityId> = points.iter().map(|point| point.id).collect();
        let slot = |id: EntityId| order.iter().position(|other| *other == id);
        let ends = |segment: EntityId| {
            segments
                .iter()
                .find(|(id, _, _)| *id == segment)
                .and_then(|(_, from, to)| Some((slot(*from)?, slot(*to)?)))
        };
        let resolved = constraints
            .iter()
            .map(|constraint| match constraint.kind {
                ConstraintKind::Fix { point, at } => slot(point)
                    .map(|slot| Resolved::Fix {
                        slot,
                        at: at.in_plane(),
                    })
                    .unwrap_or(Resolved::Dangling { residuals: 2 }),
                ConstraintKind::Horizontal { segment } => ends(segment)
                    .map(|(from, to)| Resolved::SameCoordinate { from, to, axis: 1 })
                    .unwrap_or(Resolved::Dangling { residuals: 1 }),
                ConstraintKind::Vertical { segment } => ends(segment)
                    .map(|(from, to)| Resolved::SameCoordinate { from, to, axis: 0 })
                    .unwrap_or(Resolved::Dangling { residuals: 1 }),
                ConstraintKind::Distance { from, to, length } => slot(from)
                    .zip(slot(to))
                    .map(|(from, to)| Resolved::Distance {
                        from,
                        to,
                        length: length.value(),
                    })
                    .unwrap_or(Resolved::Dangling { residuals: 1 }),
            })
            .collect();
        Some(SketchResiduals {
            order,
            constraints,
            resolved,
        })
    }

    /// The starting guess: every point's current position, which is the author's drawing.
    pub(super) fn guess(&self, points: &[Point]) -> Vec<f64> {
        let mut guess = vec![0.0; self.order.len() * 2];
        for point in points {
            if let Some(index) = self.order.iter().position(|id| *id == point.id) {
                let at = point.at.in_plane();
                guess[index * 2] = at[0];
                guess[index * 2 + 1] = at[1];
            }
        }
        guess
    }

    /// The point ids in parameter order, so a caller can write the solution back.
    pub(super) fn order(&self) -> &[EntityId] {
        &self.order
    }
}

impl ResidualSystem for SketchResiduals<'_> {
    fn parameter_count(&self) -> usize {
        self.order.len() * 2
    }

    fn residual_count(&self) -> usize {
        self.constraints
            .iter()
            .map(|constraint| constraint.kind.residual_count())
            .sum()
    }

    fn residuals(&self, parameters: &[f64], into: &mut [f64]) {
        let at = |slot: usize| [parameters[slot * 2], parameters[slot * 2 + 1]];
        let mut row = 0;
        for resolved in &self.resolved {
            match *resolved {
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
                    let span = [head[0] - tail[0], head[1] - tail[1]];
                    into[row] = (span[0] * span[0] + span[1] * span[1]).sqrt() - length;
                    row += 1;
                }
                Resolved::Dangling { residuals } => {
                    into[row..row + residuals].fill(0.0);
                    row += residuals;
                }
            }
        }
    }
}

/// The rank of the constraint system's Jacobian **at the author's own drawing** rather than at the
/// solution — the witness-configuration idea, and the fix for a defect the literature is explicit
/// about (FreeCAD #5931).
///
/// Redundancy is "did this constraint raise the rank", and rank has to be read somewhere. Reading
/// it at the SOLUTION is the obvious choice and the wrong one: rows of the Jacobian vanish at an
/// exactly-solved configuration — a distance residual between two coincident points has a zero
/// gradient — so a perfectly informative constraint can look redundant purely because the solver
/// did its job. Reading it at the pre-solve drawing avoids that: the author's sketch is a generic
/// configuration, which is exactly what a witness is for.
///
/// Zero for a system with no points or no constraints, which is the right answer for both.
pub(super) fn witness_rank(
    points: &[Point],
    segments: &[(EntityId, EntityId, EntityId)],
    constraints: &[Constraint],
) -> usize {
    let Some(system) = SketchResiduals::new(points, segments, constraints) else {
        return 0;
    };
    let at = system.guess(points);
    let matrix = jacobian(&system, &at);
    rank(&matrix, system.residual_count(), system.parameter_count())
}

/// Solve `points` against `constraints`, writing the solution back into the points.
///
/// Returns `None` when there is nothing to solve — no points, or no constraints, in which case
/// every coordinate is free and the drawing is already where the author left it. A caller that
/// wants the degree-of-freedom count for an unconstrained sketch can read it off the point count.
pub(super) fn solve_in_place(
    points: &mut [Point],
    segments: &[(EntityId, EntityId, EntityId)],
    constraints: &[Constraint],
) -> Option<SolveReport> {
    if constraints.is_empty() {
        return None;
    }
    let system = SketchResiduals::new(points, segments, constraints)?;
    let mut parameters = system.guess(points);
    let report = solve(&system, &mut parameters, SolveSettings::default());
    let order: Vec<EntityId> = system.order().to_vec();
    for (index, id) in order.iter().enumerate() {
        if let Some(point) = points.iter_mut().find(|point| point.id == *id) {
            point.at =
                SketchPoint::from_continuous(parameters[index * 2], parameters[index * 2 + 1]);
        }
    }
    Some(report)
}
