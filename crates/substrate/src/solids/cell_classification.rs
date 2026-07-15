//! The **black / white / grey cell classification** of the octree-CSG literature: given an
//! ordered sequence of per-operation conservative field intervals over one cell (each tagged with
//! the CSG role by which it combines into the running result), fold them into a single conservative
//! interval under CSG interval arithmetic, then classify that interval against an occupancy
//! threshold into the three-way verdict *empty / full / partial*.
//!
//! This is the pure **kernel** of a coarse-cell classifier: it never sees producers, leaves, world
//! coordinates, or per-voxel evaluation. Its input is a plain iterator of [`CellContribution`]s —
//! an [`Option<FieldInterval>`](FieldInterval) (`None` = *this operation cannot bound the cell*)
//! plus a [`CellCombineOp`]. Its output is the three-way [`FieldClassification`] of
//! [`FieldInterval::classify`], or `None` when the cell **cannot be classified coarsely** — because
//! some operation was unboundable, or because the contribution list was empty. A caller that gets
//! `None` (or the `Boundary`/grey verdict) must resolve the cell by exact per-sample evaluation;
//! both are the always-safe fallback.
//!
//! ## The fold
//!
//! The conservative intervals are combined by a **left fold** in contribution order, seeding the
//! running interval with the first bounded operand (the base solid of the op stack) and combining
//! each subsequent operand into it by that operand's [`CellCombineOp`] — the interval extensions of
//! the Boolean set operations of constructive solid geometry: union `min(a,b)`, intersection
//! `max(a,b)`, difference `A − B = max(A, −B)` (see [`FieldInterval`] for each lifted bound). For
//! the **union** role the fold is order-independent (union of intervals is commutative and
//! associative), so the *geometric* classification of a Union does not depend on operand order; the
//! "later operand wins" rule of a Union is a **material/attribute** resolution that this kernel does
//! not model — it decides only occupancy, and leaves attribute selection to the domain.
//!
//! **Unboundable short-circuit.** The instant any operand's interval is `None`, the whole fold is
//! `None` (cannot classify): an operation whose field cannot be bounded over the cell could place
//! the surface anywhere within it, and no CSG combine with such an operand can be coarsely decided
//! (a union with it may be occupied anywhere; an intersection or difference against it is equally
//! unbounded). An **empty** contribution list is likewise `None` — there is nothing to classify.
//! This reproduces the conservative "any unbounded operand ⇒ resolve per-sample" contract exactly.
//!
//! ## Conservatism (the whole point)
//!
//! Because every operand interval is conservative (never narrower than the operation's true range
//! over the cell) and each CSG interval extension preserves that inclusion property, the folded
//! interval is itself conservative. An "empty" or "full" verdict therefore can **never** disagree
//! with a per-sample evaluation of the same cell; only the always-safe "partial" (grey) verdict —
//! or the `None` cannot-classify — is reported where a per-sample pass might have decided. That
//! one-sided soundness is what lets a caller elide the interior of a solid and the exterior of a
//! void while remaining bit-exact against brute force.
//!
//! ## Literature
//!
//! This is the interval-arithmetic CSG cell classification of **Duff 1992**, *Interval arithmetic
//! and recursive subdivision for implicit functions and constructive solid geometry* (SIGGRAPH) —
//! the direct ancestor: fold a CSG tree's per-primitive interval bounds over a cell, then decide the
//! cell against the surface. The three-way *empty / full / partial* verdict is the **black / white /
//! grey** node classification of the region-octree literature (**Samet 2006**, *Foundations of
//! Multidimensional and Metric Data Structures*, ch. 2 — a node is white/empty, black/full, or
//! grey/mixed). A voxel domain's *air / coarse-solid / boundary* trichotomy **is** exactly this
//! black/white/grey classification; the mapping to [`FieldClassification`]'s `Air` / `CoarseSolid` /
//! `Boundary` lives in the shared [`FieldInterval::classify`] verdict this kernel returns.

use crate::interval::field_interval::{FieldClassification, FieldInterval};

/// The CSG role by which a contribution combines into the running classification interval — the
/// interval extensions of the Boolean set operations of constructive solid geometry. The domain's
/// combine operation maps onto one of these; a voxel v1 that only ever unions supplies only
/// [`CellCombineOp::Union`], and the intersection/difference roles are ready for when a subtract or
/// intersect combine role arrives (a data change at the caller, not a re-architecture here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellCombineOp {
    /// Additive: the running interval becomes the CSG **union** with this operand
    /// ([`FieldInterval::union`], `min` of fields — the nearer surface wins).
    Union,
    /// The running interval becomes the CSG **intersection** with this operand
    /// ([`FieldInterval::intersect`], `max` of fields).
    Intersect,
    /// The running interval becomes the CSG **difference** *running − operand*
    /// ([`FieldInterval::subtract`], `max(field, −operand)`).
    Subtract,
}

/// One operation's contribution to a cell's classification: its conservative [`FieldInterval`] over
/// the cell (or `None` when the operation **cannot bound** the cell — e.g. a producer whose field is
/// not intervalisable there), tagged with the [`CellCombineOp`] by which it folds into the running
/// result. The combine op of the **first** contribution seeds the base and is otherwise unused.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellContribution {
    /// The conservative bound this operation places on the field over the cell, or `None` if the
    /// operation cannot be bounded there (which collapses the whole fold to "cannot classify").
    pub field_interval: Option<FieldInterval>,
    /// How this operation combines into the running classification interval.
    pub combine: CellCombineOp,
}

impl CellContribution {
    /// A contribution that folds by CSG **union** (the only combine role a union-only op stack
    /// emits). `field_interval` is `None` when the operation cannot bound the cell.
    pub fn union(field_interval: Option<FieldInterval>) -> Self {
        Self {
            field_interval,
            combine: CellCombineOp::Union,
        }
    }
}

/// The black/white/grey cell classifier — a namespace for the [`CellClassification::classify`] fold.
/// Zero-sized: it carries no state, only the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellClassification;

impl CellClassification {
    /// Fold `contributions` into one conservative interval under their CSG combine roles, then
    /// classify it against `isolevel` (occupancy convention "inside where `field <= isolevel`").
    ///
    /// Returns:
    /// * `Some(`[`FieldClassification`]`)` — the three-way verdict (`Air` all-outside / `CoarseSolid`
    ///   all-inside / `Boundary` straddling) when every contribution was boundable.
    /// * `None` — **cannot classify coarsely**: the contribution list was empty, or some operation
    ///   was unboundable (`field_interval == None`). The caller must resolve the cell per-sample.
    ///
    /// The fold seeds the running interval with the first bounded operand and combines each
    /// subsequent one by its [`CellCombineOp`]; the first `None` short-circuits to `None`.
    pub fn classify(
        contributions: impl IntoIterator<Item = CellContribution>,
        isolevel: f32,
    ) -> Option<FieldClassification> {
        let mut accumulated: Option<FieldInterval> = None;
        let mut any = false;
        for contribution in contributions {
            any = true;
            // A single unboundable operand collapses the entire fold to "cannot classify".
            let interval = contribution.field_interval?;
            accumulated = Some(match accumulated {
                // The first bounded operand seeds the running interval (the op-stack base); its own
                // combine role is not applied against an empty accumulator.
                None => interval,
                Some(running) => match contribution.combine {
                    CellCombineOp::Union => running.union(interval),
                    CellCombineOp::Intersect => running.intersect(interval),
                    CellCombineOp::Subtract => running.subtract(interval),
                },
            });
        }
        if !any {
            // Empty contribution list ⇒ nothing to classify.
            return None;
        }
        accumulated.map(|interval| interval.classify(isolevel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn union(minimum: f32, maximum: f32) -> CellContribution {
        CellContribution::union(Some(FieldInterval::new(minimum, maximum)))
    }

    #[test]
    fn all_outside_is_empty() {
        // Every operand's interval is strictly above the isolevel ⇒ the whole cell is empty (Air).
        let verdict = CellClassification::classify([union(0.5, 2.0), union(1.0, 3.0)], 0.0);
        assert_eq!(verdict, Some(FieldClassification::Air));
    }

    #[test]
    fn all_inside_is_full() {
        // The union (min-of-fields) is all-at-or-below the isolevel ⇒ the whole cell is full.
        let verdict = CellClassification::classify([union(-3.0, -1.0), union(-2.0, 0.0)], 0.0);
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn straddle_is_partial() {
        // A single operand crossing the isolevel ⇒ the cell straddles the surface (Boundary/grey).
        let verdict = CellClassification::classify([union(-1.0, 1.0)], 0.0);
        assert_eq!(verdict, Some(FieldClassification::Boundary));
    }

    #[test]
    fn union_override_order_does_not_change_the_geometric_verdict() {
        // The Union fold is commutative: the *geometric* classification is identical regardless of
        // contribution order. (A Union's "later operand wins" is a MATERIAL rule the caller applies
        // per-sample; this occupancy kernel is order-independent.)
        let forward = CellClassification::classify([union(-3.0, -1.0), union(0.5, 2.0)], 0.0);
        let reversed = CellClassification::classify([union(0.5, 2.0), union(-3.0, -1.0)], 0.0);
        assert_eq!(forward, reversed);
        // min(min) = -3, min(max) = -1 ⇒ all-inside ⇒ full, whichever order.
        assert_eq!(forward, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn unboundable_operand_surfaces_as_cannot_classify() {
        // Any None operand collapses the fold to None (cannot classify) — even alongside boundable,
        // provably-solid operands, since the unbounded operand could place the surface anywhere.
        let verdict = CellClassification::classify(
            [union(-3.0, -1.0), CellContribution::union(None), union(-2.0, -1.0)],
            0.0,
        );
        assert_eq!(verdict, None);
    }

    #[test]
    fn empty_contribution_list_is_cannot_classify() {
        // Nothing to classify ⇒ None (the caller treats an empty cell however it likes; the kernel
        // makes no occupancy claim).
        let verdict = CellClassification::classify(std::iter::empty(), 0.0);
        assert_eq!(verdict, None);
    }

    #[test]
    fn intersect_role_folds_by_max_of_fields() {
        // Intersection = max(a, b): [max(min), max(max)]. [-3,-1] ∩ [-2,0] ⇒ [-1, 0] ⇒ all-at-or-
        // below-0 ⇒ full. Swapping the first operand's role does not matter (it seeds the base).
        let verdict = CellClassification::classify(
            [
                CellContribution {
                    field_interval: Some(FieldInterval::new(-3.0, -1.0)),
                    combine: CellCombineOp::Union,
                },
                CellContribution {
                    field_interval: Some(FieldInterval::new(-2.0, 0.0)),
                    combine: CellCombineOp::Intersect,
                },
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn subtract_role_folds_by_difference() {
        // A − B = max(A, −B). Base A = [-4, -2] (solid); subtract B = [-4, -3] ⇒ −B = [3, 4];
        // max([-4,-2], [3,4]) = [3, 4] ⇒ all-above-0 ⇒ empty (B carved A entirely away here).
        let verdict = CellClassification::classify(
            [
                CellContribution::union(Some(FieldInterval::new(-4.0, -2.0))),
                CellContribution {
                    field_interval: Some(FieldInterval::new(-4.0, -3.0)),
                    combine: CellCombineOp::Subtract,
                },
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::Air));
    }
}
