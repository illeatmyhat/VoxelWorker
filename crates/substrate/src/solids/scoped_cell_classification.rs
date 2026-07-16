//! The **scoped** black / white / grey cell classification: the
//! [`cell_classification`](super::cell_classification) fold generalized from one linear operand
//! list to a depth-first list with **scope push / pop markers**, evaluated with a stack of
//! accumulators. A scope's operands fold into their own accumulator; when the scope closes, the
//! composed interval folds into the parent accumulator under the SCOPE's combine role — so an
//! operand inside a scope can never affect the classification outside its scope. This is the
//! interval-arithmetic reading of *sealed composition scopes* (a group pre-composes its children
//! into one body, then that body combines as a unit).
//!
//! ## The event list
//!
//! Input is a plain iterator of [`ScopedCellEvent`]s in depth-first order: a
//! [`Contribution`](ScopedCellEvent::Contribution) folds into the CURRENT (innermost open)
//! accumulator under its own role; [`OpenScope`](ScopedCellEvent::OpenScope) pushes a fresh
//! empty accumulator; [`CloseScope`](ScopedCellEvent::CloseScope)`(role)` pops the innermost
//! accumulator and folds it into the new top under `role`. Events MUST be balanced (every open
//! matched by a close before the iterator ends); an unbalanced list is a caller bug
//! (debug-asserted, and the excess is ignored conservatively in release).
//!
//! ## Empty-accumulator (∅) semantics — exact, not conservative
//!
//! Each accumulator level is `Option<FieldInterval>` where `None` is the PROVABLY EMPTY solid ∅
//! (nothing folded in yet, or everything the level held was set-theoretically annihilated).
//! Folding an operand `B` into ∅ follows the Boolean set identities exactly:
//! union `∅ ∪ B = B` (the operand seeds the level), difference `∅ − B = ∅` (removing from
//! nothing leaves nothing), intersection `∅ ∩ B = ∅`. A scope that closes at ∅ contributes
//! nothing to its parent under union/difference and annihilates it under intersection — the
//! same identities, applied to the popped body. The linear kernel's "seed with the first operand
//! regardless of role" shortcut (which its callers compensate for by dropping leading subtracts)
//! is replaced by these identities, so a leading-subtract operand needs no caller-side handling
//! here: it folds into ∅ and stays ∅.
//!
//! A fold whose ROOT ends at ∅ classifies as [`FieldClassification::Air`] — exact (the identities
//! above never widen occupancy), not merely conservative.
//!
//! ## Conservatism
//!
//! Identical to the linear kernel: every operand interval is conservative and each CSG interval
//! extension preserves inclusion, so an *empty*/*full* verdict can never disagree with a
//! per-sample evaluation. The instant any operand is unboundable (`field_interval == None`), the
//! whole fold returns `None` (cannot classify) — an unbounded operand could place the surface
//! anywhere, at any scope depth.
//!
//! ## Literature
//!
//! The per-cell interval fold is **Duff 1992** (*Interval arithmetic and recursive subdivision
//! for implicit functions and constructive solid geometry*, SIGGRAPH) exactly as in
//! [`cell_classification`](super::cell_classification); the three-way verdict is the black /
//! white / grey node classification of the region-octree literature (**Samet 2006**, ch. 2).
//! The *flattened-list-with-stack* evaluation shape — a CSG expression linearized depth-first
//! and evaluated left-to-right with a scope stack instead of tree recursion — is the Boolean
//! list formulation of **Rossignac 1999** (*Blist: a Boolean list formulation of CSG trees*,
//! GVU tech report GIT-GVU-99-04), and is the convergent evaluator of the ordered-edit-list SDF
//! sculptors (Nijhoff's WebGPU SDF editor stores primitives as a depth-first ordered list
//! evaluated with an accumulator plus a group push/pop stack; see
//! `docs/design/csg-prior-art-study.md`, round 2).

use crate::interval::field_interval::{FieldClassification, FieldInterval};

use super::cell_classification::{CellCombineOp, CellContribution};

/// One event of a depth-first scoped operand list (see the module doc): an operand's
/// contribution, or a scope boundary marker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScopedCellEvent {
    /// One operand's conservative interval + combine role, folded into the CURRENT
    /// (innermost open) accumulator.
    Contribution(CellContribution),
    /// A scope opens: push a fresh empty (∅) accumulator; subsequent contributions fold into
    /// it until the matching [`CloseScope`](ScopedCellEvent::CloseScope).
    OpenScope,
    /// The innermost scope closes: pop its accumulator and fold the composed body into the
    /// new top under this role (the SCOPE's own combine role, not any operand's).
    CloseScope(CellCombineOp),
}

/// The scoped black/white/grey cell classifier — a namespace for the
/// [`ScopedCellClassification::classify`] stack fold. Zero-sized: it carries no state, only
/// the algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopedCellClassification;

impl ScopedCellClassification {
    /// Fold a balanced depth-first `events` list into one conservative interval with a scope
    /// stack (see the module doc), then classify it against `isolevel` (occupancy convention
    /// "inside where `field <= isolevel`").
    ///
    /// Returns:
    /// * `Some(`[`FieldClassification`]`)` — the three-way verdict when every contribution was
    ///   boundable. A root accumulator that ends at ∅ (nothing contributed, or everything
    ///   annihilated by the set identities) is `Some(Air)` — exact, per the module doc.
    /// * `None` — **cannot classify coarsely**: some operand was unboundable
    ///   (`field_interval == None`). The caller must resolve the cell per-sample.
    pub fn classify(
        events: impl IntoIterator<Item = ScopedCellEvent>,
        isolevel: f32,
    ) -> Option<FieldClassification> {
        // The accumulator stack; index 0 is the root scope. `None` per level = the provably
        // empty solid ∅ (see the module doc's set identities).
        let mut stack: Vec<Option<FieldInterval>> = vec![None];
        for event in events {
            match event {
                ScopedCellEvent::Contribution(contribution) => {
                    // A single unboundable operand collapses the entire fold to
                    // "cannot classify", regardless of scope depth.
                    let interval = contribution.field_interval?;
                    let top = stack.last_mut().expect("the root level is never popped");
                    *top = fold_into(*top, interval, contribution.combine);
                }
                ScopedCellEvent::OpenScope => stack.push(None),
                ScopedCellEvent::CloseScope(role) => {
                    debug_assert!(
                        stack.len() > 1,
                        "unbalanced ScopedCellEvent list: CloseScope with no open scope"
                    );
                    if stack.len() <= 1 {
                        // Conservative release behaviour for a caller bug: ignore the
                        // excess close rather than corrupt the root accumulator.
                        continue;
                    }
                    let closed = stack.pop().expect("len checked above");
                    let top = stack.last_mut().expect("the root level remains");
                    match closed {
                        // The scope composed a real body: fold it into the parent as one
                        // operand under the SCOPE's role.
                        Some(body) => *top = fold_into(*top, body, role),
                        // The scope composed ∅: `A ∪ ∅ = A`, `A − ∅ = A`, `A ∩ ∅ = ∅`.
                        None => {
                            if role == CellCombineOp::Intersect {
                                *top = None;
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(
            stack.len(),
            1,
            "unbalanced ScopedCellEvent list: {} scope(s) left open",
            stack.len() - 1
        );
        // A root that ended at ∅ is exactly empty (Air); a composed interval classifies
        // three-way as in the linear kernel.
        Some(match stack.pop().flatten() {
            Some(interval) => interval.classify(isolevel),
            None => FieldClassification::Air,
        })
    }
}

/// Fold one operand interval into an accumulator level under `role`, honouring the ∅
/// identities of the module doc: union seeds an empty level; difference/intersection against
/// an empty level stay empty (`∅ − B = ∅`, `∅ ∩ B = ∅`).
fn fold_into(
    accumulated: Option<FieldInterval>,
    operand: FieldInterval,
    role: CellCombineOp,
) -> Option<FieldInterval> {
    match (accumulated, role) {
        (None, CellCombineOp::Union) => Some(operand),
        (None, CellCombineOp::Subtract) | (None, CellCombineOp::Intersect) => None,
        (Some(running), CellCombineOp::Union) => Some(running.union(operand)),
        (Some(running), CellCombineOp::Subtract) => Some(running.subtract(operand)),
        (Some(running), CellCombineOp::Intersect) => Some(running.intersect(operand)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contribution(minimum: f32, maximum: f32, combine: CellCombineOp) -> ScopedCellEvent {
        ScopedCellEvent::Contribution(CellContribution {
            field_interval: Some(FieldInterval::new(minimum, maximum)),
            combine,
        })
    }

    #[test]
    fn flat_union_list_matches_the_linear_kernel() {
        // With no scope markers the stack fold IS the linear left fold: two solid unions ⇒
        // full, exactly as `CellClassification::classify` decides the same list.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-3.0, -1.0, CellCombineOp::Union),
                contribution(-2.0, 0.0, CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn leading_subtract_folds_into_empty_and_stays_empty() {
        // ∅ − B = ∅, then ∅ ∪ A = A: a subtract operand BEFORE the first union operand
        // removes from nothing (the linear kernel's callers drop it by hand; here the ∅
        // identity handles it). The base solid then classifies alone.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-4.0, -3.0, CellCombineOp::Subtract),
                contribution(-3.0, -1.0, CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn subtract_inside_a_scope_cannot_reach_the_outer_body() {
        // Outer body: solid [-3, -1]. A scope opens, contributes a union [-2, 0] and a
        // subtract that annihilates the SCOPE body ([-9, -8] ⇒ −B = [8, 9] ⇒ max = [8, 9],
        // provably empty inside the scope). The scope closes at all-outside, unions into the
        // parent (min keeps the parent's solid bound), so the outer body stays full — the
        // cutter was spent inside its scope.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-3.0, -1.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                contribution(-2.0, 0.0, CellCombineOp::Union),
                contribution(-9.0, -8.0, CellCombineOp::Subtract),
                ScopedCellEvent::CloseScope(CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn a_scope_closing_under_subtract_carves_the_parent() {
        // Parent solid [-4, -2]; a scope composes a deeply-solid body [-4, -3] and closes
        // under Subtract: max([-4, -2], [3, 4]) = [3, 4] ⇒ all-above-0 ⇒ empty. Identical
        // arithmetic to the linear kernel's subtract-role test, with the operand composed
        // in a scope first.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-4.0, -2.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                contribution(-4.0, -3.0, CellCombineOp::Union),
                ScopedCellEvent::CloseScope(CellCombineOp::Subtract),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::Air));
    }

    #[test]
    fn an_empty_scope_contributes_nothing_under_union_and_subtract() {
        for role in [CellCombineOp::Union, CellCombineOp::Subtract] {
            let verdict = ScopedCellClassification::classify(
                [
                    contribution(-3.0, -1.0, CellCombineOp::Union),
                    ScopedCellEvent::OpenScope,
                    ScopedCellEvent::CloseScope(role),
                ],
                0.0,
            );
            assert_eq!(
                verdict,
                Some(FieldClassification::CoarseSolid),
                "an ∅ scope must not change the parent under {role:?}"
            );
        }
    }

    #[test]
    fn a_leading_intersect_operand_folds_into_empty_and_stays_empty() {
        // ∅ ∩ B = ∅: an Intersect OPERAND before anything accumulated (the fold-start
        // edge case of the ordering law) annihilates nothing into nothing — alone it
        // classifies exactly Air, and a union operand AFTER it seeds fresh (∅ ∪ A = A).
        let alone = ScopedCellClassification::classify(
            [contribution(-3.0, -1.0, CellCombineOp::Intersect)],
            0.0,
        );
        assert_eq!(alone, Some(FieldClassification::Air));
        let then_union = ScopedCellClassification::classify(
            [
                contribution(-3.0, -1.0, CellCombineOp::Intersect),
                contribution(-3.0, -1.0, CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(then_union, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn an_empty_scope_annihilates_the_parent_under_intersect() {
        // A ∩ ∅ = ∅: the parent solid intersected with an empty scope body is provably empty.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-3.0, -1.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                ScopedCellEvent::CloseScope(CellCombineOp::Intersect),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::Air));
    }

    #[test]
    fn nested_scopes_fold_innermost_first() {
        // Root: solid [-4, -2]. Outer scope: solid [-4, -3], inner scope carving the OUTER
        // scope's body entirely (subtract of a containing solid), so the outer scope closes
        // at ∅ and the root body survives untouched.
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-4.0, -2.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                contribution(-4.0, -3.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                contribution(-5.0, -4.0, CellCombineOp::Union),
                ScopedCellEvent::CloseScope(CellCombineOp::Subtract),
                ScopedCellEvent::CloseScope(CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(verdict, Some(FieldClassification::CoarseSolid));
    }

    #[test]
    fn unboundable_operand_collapses_to_cannot_classify_at_any_depth() {
        let verdict = ScopedCellClassification::classify(
            [
                contribution(-3.0, -1.0, CellCombineOp::Union),
                ScopedCellEvent::OpenScope,
                ScopedCellEvent::Contribution(CellContribution::union(None)),
                ScopedCellEvent::CloseScope(CellCombineOp::Union),
            ],
            0.0,
        );
        assert_eq!(verdict, None);
    }

    #[test]
    fn an_all_empty_fold_is_exactly_air() {
        // No contributions at all: the root ends at ∅, which is exactly empty (Air) — the
        // scoped kernel's ∅ verdict, distinct from the linear kernel's None-on-empty-list.
        let verdict = ScopedCellClassification::classify(std::iter::empty(), 0.0);
        assert_eq!(verdict, Some(FieldClassification::Air));
    }
}
