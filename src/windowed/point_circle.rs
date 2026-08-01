//! Shared interaction state for Two-Point and Three-Point Circle.

use document::scene::NodeId;
use document::sketch::{PointCirclePlacement, SketchPoint, SketchSolid};

use super::sketch_target::ResolvedSketchTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PointCircleKind {
    TwoPoint,
    ThreePoint,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingPointCircle {
    owner: NodeId,
    kind: PointCircleKind,
    first: SketchPoint,
    second: Option<SketchPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PointCircleEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct PointCircleGesture {
    pending: Option<PendingPointCircle>,
}

impl PointCircleGesture {
    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<PointCircleKind>,
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
        active_kind: Option<PointCircleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub const fn blocks_enter(
        self,
        active_kind: Option<PointCircleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    pub fn placement(
        self,
        owner: NodeId,
        kind: PointCircleKind,
        producer: &SketchSolid,
        cursor: ResolvedSketchTarget,
    ) -> Option<PointCirclePlacement> {
        let pending = self
            .pending
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        match (kind, pending.second) {
            (PointCircleKind::TwoPoint, None) => producer
                .two_point_circle_placement(pending.first, cursor.at)
                .ok(),
            (PointCircleKind::ThreePoint, Some(second)) => producer
                .three_point_circle_placement(pending.first, second, cursor.at)
                .ok(),
            (PointCircleKind::TwoPoint, Some(_)) | (PointCircleKind::ThreePoint, None) => None,
        }
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: PointCircleKind,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
    ) -> PointCircleEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingPointCircle {
                    owner,
                    kind,
                    first: target.at,
                    second: None,
                });
            }
            return PointCircleEdit::InteractionOnly;
        };
        if pending.owner != owner || pending.kind != kind {
            if let Some(target) = target {
                self.pending = Some(PendingPointCircle {
                    owner,
                    kind,
                    first: target.at,
                    second: None,
                });
            }
            return PointCircleEdit::InteractionOnly;
        }
        let Some(target) = target else {
            return PointCircleEdit::InteractionOnly;
        };
        match (kind, pending.second) {
            (PointCircleKind::TwoPoint, None) => producer
                .with_two_point_circle(pending.first, target.at)
                .map_or(PointCircleEdit::InteractionOnly, |(next, _)| {
                    PointCircleEdit::Document(next)
                }),
            (PointCircleKind::ThreePoint, None) => {
                if !pending.first.coincides(&target.at) {
                    self.pending = Some(PendingPointCircle {
                        second: Some(target.at),
                        ..pending
                    });
                }
                PointCircleEdit::InteractionOnly
            }
            (PointCircleKind::ThreePoint, Some(second)) => producer
                .with_three_point_circle(pending.first, second, target.at)
                .map_or(PointCircleEdit::InteractionOnly, |(next, _)| {
                    PointCircleEdit::Document(next)
                }),
            (PointCircleKind::TwoPoint, Some(_)) => PointCircleEdit::InteractionOnly,
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget { at, existing: None }
    }

    #[test]
    fn both_point_circle_grammars_commit_atomically_and_repeat_ready() {
        let owner = NodeId(4);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for (kind, points) in [
            (
                PointCircleKind::TwoPoint,
                vec![SketchPoint::new(-2, 0), SketchPoint::new(2, 0)],
            ),
            (
                PointCircleKind::ThreePoint,
                vec![
                    SketchPoint::new(1, 0),
                    SketchPoint::new(0, 1),
                    SketchPoint::new(-1, 0),
                ],
            ),
        ] {
            let mut gesture = PointCircleGesture::default();
            let mut result = PointCircleEdit::InteractionOnly;
            for point in points {
                result = gesture.click(owner, kind, &source, Some(target(point)));
            }
            let PointCircleEdit::Document(made) = result else {
                panic!("final click completes")
            };
            assert_eq!(made.sketch.circles().len(), 1);
            assert!(source.sketch.points().is_empty());
            assert!(gesture.pending.is_none());
        }
    }
}
