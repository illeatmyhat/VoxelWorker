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
use substrate::nonlinear_least_squares::{solve, ResidualSystem, SolveReport, SolveSettings};

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

/// Why a constraint could not be added.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintRefusal {
    /// It names geometry the store does not hold.
    UnknownEntity,
    /// Its own terms cannot be met by any drawing — a negative distance, a `Horizontal` on a
    /// segment whose ends are the same point.
    Impossible,
    /// The system it would join has no solution: it fights what is already asserted. The
    /// constraint it fights is not named yet; that is the tool layer's job once constraints are
    /// selectable in the UI.
    Unsatisfiable,
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
