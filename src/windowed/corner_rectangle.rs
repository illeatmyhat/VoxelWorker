//! Shared interaction state for the two-click Rectangle and Center Rectangle commands.

use document::scene::NodeId;
use document::sketch::{RectanglePlacement, SketchPoint, SketchSolid};
use parametric::EvaluationContext;

use document::sketch::SketchTarget;

/// Which corner grammar the first click opened. Held in the pending record rather than read
/// from the live tool at commit time: switching tools mid-gesture would otherwise reinterpret
/// the held click, silently turning a center into a corner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CornerRectangleKind {
    TwoPoint,
    CenterCorner,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingCornerRectangle {
    owner: NodeId,
    kind: CornerRectangleKind,
    first: SketchPoint,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum CornerRectangleEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct CornerRectangleGesture {
    pending: Option<PendingCornerRectangle>,
}

impl CornerRectangleGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .map(|pending| pending.first)
            .collect()
    }

    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<CornerRectangleKind>,
        constraint_is_armed: bool,
        owner: Option<NodeId>,
    ) {
        if constraint_is_armed
            || self.pending.is_some_and(|pending| {
                Some(pending.owner) != owner || Some(pending.kind) != active_kind
            })
        {
            self.reset();
        }
    }

    pub fn cancel_for_escape(
        &mut self,
        active_kind: Option<CornerRectangleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub const fn blocks_enter(
        self,
        active_kind: Option<CornerRectangleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    /// The loop the second click would author, for the preview. Resolves through the same
    /// document placement the commit takes, so what is drawn is what lands.
    pub fn placement(
        self,
        owner: NodeId,
        kind: CornerRectangleKind,
        producer: &SketchSolid,
        cursor: SketchTarget,
    ) -> Option<RectanglePlacement> {
        let pending = self
            .pending
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        match kind {
            CornerRectangleKind::TwoPoint => producer
                .corner_rectangle_placement(pending.first, cursor.at())
                .ok(),
            CornerRectangleKind::CenterCorner => producer
                .center_rectangle_placement(pending.first, cursor.at())
                .ok(),
        }
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: CornerRectangleKind,
        producer: &SketchSolid,
        target: Option<SketchTarget>,
        context: EvaluationContext,
    ) -> CornerRectangleEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingCornerRectangle {
                    owner,
                    kind,
                    first: target.at(),
                });
            }
            return CornerRectangleEdit::InteractionOnly;
        };
        // A click that arrives under a different owner or grammar restarts rather than
        // committing across the change.
        if pending.owner != owner || pending.kind != kind {
            if let Some(target) = target {
                self.pending = Some(PendingCornerRectangle {
                    owner,
                    kind,
                    first: target.at(),
                });
            }
            return CornerRectangleEdit::InteractionOnly;
        }
        let Some(target) = target else {
            return CornerRectangleEdit::InteractionOnly;
        };
        let made = match kind {
            CornerRectangleKind::TwoPoint => {
                producer.with_rectangle(pending.first, target.at(), context)
            }
            CornerRectangleKind::CenterCorner => {
                producer.with_center_rectangle(pending.first, target.at(), context)
            }
        };
        made.map_or(CornerRectangleEdit::InteractionOnly, |next| {
            CornerRectangleEdit::Document(next)
        })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};
    use std::num::NonZeroU32;

    fn target(at: SketchPoint) -> SketchTarget {
        SketchTarget::fresh(at)
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(NonZeroU32::new(16).unwrap())
    }

    fn source() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    #[test]
    fn both_corner_grammars_commit_on_the_second_click_and_repeat_ready() {
        let owner = NodeId(7);
        let source = source();
        for kind in [
            CornerRectangleKind::TwoPoint,
            CornerRectangleKind::CenterCorner,
        ] {
            let mut gesture = CornerRectangleGesture::default();
            let first = gesture.click(
                owner,
                kind,
                &source,
                Some(target(SketchPoint::new(0, 0))),
                context(),
            );
            assert_eq!(first, CornerRectangleEdit::InteractionOnly);
            assert!(source.sketch.points().is_empty());
            let preview = gesture
                .placement(owner, kind, &source, target(SketchPoint::new(4, 3)))
                .unwrap();
            let CornerRectangleEdit::Document(made) = gesture.click(
                owner,
                kind,
                &source,
                Some(target(SketchPoint::new(4, 3))),
                context(),
            ) else {
                panic!("second click completes")
            };
            assert_eq!(
                made.sketch.segments().len(),
                if matches!(kind, CornerRectangleKind::CenterCorner) {
                    6
                } else {
                    4
                }
            );
            // The preview named the very corners the commit authored.
            for corner in preview.corners {
                assert!(made.sketch.point_at(corner).is_some());
            }
            assert!(gesture.pending.is_none());
        }
    }

    #[test]
    fn switching_grammar_mid_gesture_restarts_instead_of_reinterpreting_the_held_click() {
        let owner = NodeId(7);
        let source = source();
        let mut gesture = CornerRectangleGesture::default();
        gesture.click(
            owner,
            CornerRectangleKind::CenterCorner,
            &source,
            Some(target(SketchPoint::new(0, 0))),
            context(),
        );
        // The same second click under the other grammar must not author a rectangle whose
        // center was read as a corner.
        let edit = gesture.click(
            owner,
            CornerRectangleKind::TwoPoint,
            &source,
            Some(target(SketchPoint::new(4, 3))),
            context(),
        );
        assert_eq!(edit, CornerRectangleEdit::InteractionOnly);
        assert!(
            gesture
                .placement(
                    owner,
                    CornerRectangleKind::TwoPoint,
                    &source,
                    target(SketchPoint::new(8, 9))
                )
                .is_some(),
            "the click that switched grammar became the new first corner"
        );
    }

    #[test]
    fn a_pending_corner_does_not_survive_its_sketch() {
        let mut gesture = CornerRectangleGesture::default();
        gesture.click(
            NodeId(7),
            CornerRectangleKind::TwoPoint,
            &source(),
            Some(target(SketchPoint::new(0, 0))),
            context(),
        );
        gesture.retain_for_context(Some(CornerRectangleKind::TwoPoint), false, Some(NodeId(8)));
        assert!(gesture.pending.is_none());
    }
}
