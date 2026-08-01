//! Interaction-transient state for Three-Point Rectangle.

use document::scene::NodeId;
use document::sketch::{RectanglePlacement, SketchPoint, SketchSolid};

use super::sketch_target::ResolvedSketchTarget;

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingRectangle {
    owner: NodeId,
    first: SketchPoint,
    second: Option<SketchPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum ThreePointRectangleEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ThreePointRectangleGesture {
    pending: Option<PendingRectangle>,
}

impl ThreePointRectangleGesture {
    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        tool_is_armed: bool,
        constraint_is_armed: bool,
        owner: Option<NodeId>,
    ) {
        if !tool_is_armed
            || constraint_is_armed
            || self
                .pending
                .is_some_and(|pending| Some(pending.owner) != owner)
        {
            self.reset();
        }
    }

    pub fn cancel_for_escape(&mut self, tool_is_active: bool, constraint_is_armed: bool) -> bool {
        let was_live = self.reset();
        tool_is_active && !constraint_is_armed && was_live
    }

    pub const fn blocks_enter(self, tool_is_active: bool, constraint_is_armed: bool) -> bool {
        tool_is_active && !constraint_is_armed && self.pending.is_some()
    }

    pub fn guide(self, owner: NodeId) -> Option<(SketchPoint, Option<SketchPoint>)> {
        self.pending
            .filter(|pending| pending.owner == owner)
            .map(|pending| (pending.first, pending.second))
    }

    pub fn placement(
        self,
        owner: NodeId,
        producer: &SketchSolid,
        width_point: ResolvedSketchTarget,
    ) -> Option<RectanglePlacement> {
        let pending = self.pending.filter(|pending| pending.owner == owner)?;
        producer
            .three_point_rectangle_placement(pending.first, pending.second?, width_point.at)
            .ok()
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
    ) -> ThreePointRectangleEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingRectangle {
                    owner,
                    first: target.at,
                    second: None,
                });
            }
            return ThreePointRectangleEdit::InteractionOnly;
        };
        if pending.owner != owner {
            return ThreePointRectangleEdit::InteractionOnly;
        }
        let Some(target) = target else {
            return ThreePointRectangleEdit::InteractionOnly;
        };
        let Some(second) = pending.second else {
            if !pending.first.coincides(&target.at) {
                self.pending = Some(PendingRectangle {
                    second: Some(target.at),
                    ..pending
                });
            }
            return ThreePointRectangleEdit::InteractionOnly;
        };
        producer
            .with_three_point_rectangle(pending.first, second, target.at)
            .map_or(ThreePointRectangleEdit::InteractionOnly, |next| {
                ThreePointRectangleEdit::Document(next)
            })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget { at, existing: None }
    }

    #[test]
    fn third_click_commits_one_oriented_loop_and_clears_the_gesture() {
        let owner = NodeId(1);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = ThreePointRectangleGesture::default();
        gesture.click(owner, &source, Some(target(SketchPoint::new(0, 0))));
        gesture.click(owner, &source, Some(target(SketchPoint::new(3, 4))));
        assert!(source.sketch.points().is_empty());
        let preview = gesture
            .placement(owner, &source, target(SketchPoint::new(0, 5)))
            .unwrap();
        let ThreePointRectangleEdit::Document(made) =
            gesture.click(owner, &source, Some(target(SketchPoint::new(0, 5))))
        else {
            panic!("completion")
        };
        assert_eq!(made.sketch.segments().len(), 4);
        assert_eq!(made.sketch.points().len(), 4);
        assert!(gesture.pending.is_none());
        for corner in preview.corners {
            assert!(made.sketch.point_at(corner).is_some());
        }
    }
}
