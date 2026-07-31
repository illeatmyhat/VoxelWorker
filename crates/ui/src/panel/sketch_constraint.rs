//! Which constraint a selection can carry, and what it asserts (ADR 0035, ADR 0030 §5).
//!
//! A constraint is **not** a drawing tool. Every other rail cell in sketch mode arms a mode and
//! waits for a click in the viewport; a constraint reads what is already picked and applies at
//! once, which is the model Fusion and Onshape both settle on. Making it modal would ask the
//! author to pick the same entities a second time inside the mode, and there is nothing for the
//! second pick to add.
//!
//! That leaves one question the rail cannot answer by itself: *is this verb applicable to what is
//! picked right now?* — the cell is live or dead by the answer, and pressing a live one must not
//! be able to fail for want of geometry. [`ConstraintVerb::kinds`] answers it and builds the
//! assertions in the same pass, so "the cell was enabled" and "these are the constraints" cannot
//! drift apart.
//!
//! Only the three kinds whose residuals ship are here. The other eleven glyphs on the constraint
//! shelf are drawn and named but have no residual behind them yet
//! (`crates/document/src/sketch/constraint.rs`), and a verb with no residual would light a cell
//! that silently asserts nothing.

use document::sketch::{ConstraintKind, EntityId, Sketch};

/// A constraint the rail offers as a verb over the current selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintVerb {
    /// Every picked segment lies along the plane's first in-plane axis.
    Horizontal,
    /// Every picked segment lies along the plane's second in-plane axis.
    Vertical,
    /// Every picked point stays where it is.
    Fix,
}

impl ConstraintVerb {
    /// What this verb asserts about `picked_points` / `picked_segments`, one constraint per
    /// entity it applies to. Empty means **not applicable** — the rail draws the cell dead.
    ///
    /// A multi-selection asserts the verb of each member separately rather than relating them:
    /// two picked segments both told Horizontal is two constraints, not one "these two agree".
    /// The relating verbs (Parallel, Equal, Symmetry) are a different shelf entry and a different
    /// arity, and folding them together here would make the count of constraints depend on which
    /// button was pressed rather than on what was said.
    ///
    /// Ids naming geometry the sketch does not hold are dropped, because a stale selection is a
    /// disabled cell rather than a refusal the author has to read.
    pub fn kinds(
        self,
        sketch: &Sketch,
        picked_points: &[EntityId],
        picked_segments: &[EntityId],
    ) -> Vec<ConstraintKind> {
        match self {
            ConstraintVerb::Horizontal => picked_segments
                .iter()
                .filter(|id| live_segment(sketch, **id))
                .map(|id| ConstraintKind::Horizontal { segment: *id })
                .collect(),
            ConstraintVerb::Vertical => picked_segments
                .iter()
                .filter(|id| live_segment(sketch, **id))
                .map(|id| ConstraintKind::Vertical { segment: *id })
                .collect(),
            // `at` is read from the drawing rather than left implicit: a Fix asserts immovability
            // AT A PLACE, so the place has to be captured at the moment the author asks for it.
            ConstraintVerb::Fix => picked_points
                .iter()
                .filter_map(|id| {
                    let point = sketch.points().iter().find(|point| point.id == *id)?;
                    Some(ConstraintKind::Fix {
                        point: point.id,
                        at: point.at,
                    })
                })
                .collect(),
        }
    }

    /// The rail tooltip, which names the arity the cell is dead without.
    pub fn tooltip(self) -> &'static str {
        match self {
            ConstraintVerb::Horizontal => "Horizontal — pick one or more lines",
            ConstraintVerb::Vertical => "Vertical — pick one or more lines",
            ConstraintVerb::Fix => "Fix — pick one or more points",
        }
    }
}

/// Whether `id` names a segment the sketch still holds and that has two distinct ends — the
/// degenerate case `Sketch::add_constraint` refuses as `Impossible`, screened here so a cell is
/// never live for an assertion that cannot be kept.
fn live_segment(sketch: &Sketch, id: EntityId) -> bool {
    sketch
        .segments()
        .iter()
        .any(|segment| segment.id == id && segment.from != segment.to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, SketchPoint};

    /// Two points joined by one segment, on the ground plane.
    fn one_segment() -> (Sketch, EntityId, EntityId, EntityId) {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let from = sketch.add_free_point(SketchPoint::from_continuous(0.0, 0.0));
        let to = sketch.add_free_point(SketchPoint::from_continuous(8.0, 3.0));
        let segment = sketch.connect(from, to).expect("two distinct points join");
        (sketch, from, to, segment)
    }

    #[test]
    fn a_verb_with_nothing_picked_is_not_applicable() {
        let (sketch, _, _, _) = one_segment();
        for verb in [
            ConstraintVerb::Horizontal,
            ConstraintVerb::Vertical,
            ConstraintVerb::Fix,
        ] {
            assert!(verb.kinds(&sketch, &[], &[]).is_empty());
        }
    }

    #[test]
    fn the_line_verbs_want_segments_and_fix_wants_points() {
        let (sketch, from, _, segment) = one_segment();
        // A picked segment says nothing to Fix, and a picked point says nothing to Horizontal.
        assert!(ConstraintVerb::Fix
            .kinds(&sketch, &[], &[segment])
            .is_empty());
        assert!(ConstraintVerb::Horizontal
            .kinds(&sketch, &[from], &[])
            .is_empty());
        assert_eq!(
            ConstraintVerb::Horizontal.kinds(&sketch, &[], &[segment]),
            vec![ConstraintKind::Horizontal { segment }]
        );
    }

    #[test]
    fn fix_captures_the_position_the_point_is_at() {
        let (sketch, _, to, _) = one_segment();
        let kinds = ConstraintVerb::Fix.kinds(&sketch, &[to], &[]);
        let ConstraintKind::Fix { point, at } = kinds[0] else {
            panic!("Fix builds a Fix");
        };
        assert_eq!(point, to);
        assert_eq!(at.in_plane(), [8.0, 3.0]);
    }

    #[test]
    fn a_stale_id_drops_out_rather_than_refusing() {
        let (mut sketch, from, _, segment) = one_segment();
        sketch.delete_point_cascade(from);
        assert!(ConstraintVerb::Horizontal
            .kinds(&sketch, &[], &[segment])
            .is_empty());
        assert!(ConstraintVerb::Fix.kinds(&sketch, &[from], &[]).is_empty());
    }

    #[test]
    fn a_multi_selection_asserts_the_verb_once_per_entity() {
        let (mut sketch, _, _, first) = one_segment();
        let a = sketch.add_free_point(SketchPoint::from_continuous(0.0, 9.0));
        let b = sketch.add_free_point(SketchPoint::from_continuous(5.0, 9.0));
        let second = sketch.connect(a, b).expect("two distinct points join");
        assert_eq!(
            ConstraintVerb::Vertical.kinds(&sketch, &[], &[first, second]),
            vec![
                ConstraintKind::Vertical { segment: first },
                ConstraintKind::Vertical { segment: second },
            ]
        );
    }
}
