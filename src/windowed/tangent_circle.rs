//! Interaction state for Two-Tangent and Three-Tangent Circle.

use document::scene::NodeId;
use document::sketch::{EntityId, SketchPoint, SketchSolid, TangentCirclePlacement};
use parametric::EvaluationContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TangentCircleKind {
    Two,
    Three,
}

#[derive(Debug, Clone, PartialEq)]
struct PendingTangentCircle {
    owner: NodeId,
    kind: TangentCircleKind,
    lines: Vec<(EntityId, SketchPoint)>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum TangentCircleEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct TangentCircleGesture {
    pending: Option<PendingTangentCircle>,
}

impl TangentCircleGesture {
    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<TangentCircleKind>,
        constraint_is_armed: bool,
        owner: Option<NodeId>,
    ) {
        if constraint_is_armed
            || self.pending.as_ref().is_some_and(|pending| {
                Some(pending.owner) != owner || Some(pending.kind) != active_kind
            })
        {
            self.reset();
        }
    }

    pub fn cancel_for_escape(
        &mut self,
        active_kind: Option<TangentCircleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub fn blocks_enter(
        &self,
        active_kind: Option<TangentCircleKind>,
        constraint_is_armed: bool,
    ) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    pub fn placement(
        &self,
        owner: NodeId,
        kind: TangentCircleKind,
        producer: &SketchSolid,
        cursor: SketchPoint,
        hovered_line: Option<EntityId>,
    ) -> Option<TangentCirclePlacement> {
        let pending = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        match (kind, pending.lines.as_slice()) {
            (TangentCircleKind::Two, [(first, _), (second, _)]) => producer
                .two_tangent_circle_placement([*first, *second], cursor)
                .ok(),
            (TangentCircleKind::Three, [first, second]) => {
                let third = hovered_line?;
                if third == first.0 || third == second.0 {
                    return None;
                }
                producer
                    .three_tangent_circle_placement([*first, *second, (third, cursor)])
                    .ok()
            }
            _ => None,
        }
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: TangentCircleKind,
        producer: &SketchSolid,
        line: Option<(EntityId, SketchPoint)>,
        cursor: Option<SketchPoint>,
        context: EvaluationContext,
    ) -> TangentCircleEdit {
        let pending = self.pending.get_or_insert_with(|| PendingTangentCircle {
            owner,
            kind,
            lines: Vec::with_capacity(match kind {
                TangentCircleKind::Two => 2,
                TangentCircleKind::Three => 3,
            }),
        });
        if pending.owner != owner || pending.kind != kind {
            *pending = PendingTangentCircle {
                owner,
                kind,
                lines: Vec::new(),
            };
        }
        match kind {
            TangentCircleKind::Two if pending.lines.len() < 2 => {
                if let Some(line) = line {
                    if !pending.lines.iter().any(|(id, _)| *id == line.0) {
                        pending.lines.push(line);
                    }
                }
                TangentCircleEdit::InteractionOnly
            }
            TangentCircleKind::Two => {
                let Some(cursor) = cursor else {
                    return TangentCircleEdit::InteractionOnly;
                };
                let lines = self
                    .pending
                    .take()
                    .map(|pending| pending.lines)
                    .unwrap_or_default();
                let [first, second] = lines.as_slice() else {
                    return TangentCircleEdit::InteractionOnly;
                };
                producer
                    .with_two_tangent_circle([first.0, second.0], cursor, context)
                    .map_or(
                        TangentCircleEdit::InteractionOnly,
                        TangentCircleEdit::Document,
                    )
            }
            TangentCircleKind::Three => {
                let Some(line) = line else {
                    return TangentCircleEdit::InteractionOnly;
                };
                if pending.lines.iter().any(|(id, _)| *id == line.0) {
                    return TangentCircleEdit::InteractionOnly;
                }
                if pending.lines.len() < 2 {
                    pending.lines.push(line);
                    return TangentCircleEdit::InteractionOnly;
                }
                let lines = self
                    .pending
                    .take()
                    .map(|pending| pending.lines)
                    .unwrap_or_default();
                let [first, second] = lines.as_slice() else {
                    return TangentCircleEdit::InteractionOnly;
                };
                producer
                    .with_three_tangent_circle([*first, *second, line], context)
                    .map_or(
                        TangentCircleEdit::InteractionOnly,
                        TangentCircleEdit::Document,
                    )
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn source() -> (SketchSolid, [EntityId; 3]) {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let mut add = |from, to| {
            let from = sketch.add_free_point(from);
            let to = sketch.add_free_point(to);
            sketch.connect(from, to).unwrap()
        };
        let bottom = add(SketchPoint::new(0, 0), SketchPoint::new(10, 0));
        let diagonal = add(SketchPoint::new(10, 0), SketchPoint::new(0, 10));
        let left = add(SketchPoint::new(0, 10), SketchPoint::new(0, 0));
        (SketchSolid::extrude(sketch, 3), [bottom, diagonal, left])
    }

    #[test]
    fn both_grammars_commit_constraints_only_at_completion() {
        let owner = NodeId(1);
        let context = EvaluationContext::new(std::num::NonZeroU32::new(16).unwrap());
        let (source, [bottom, diagonal, left]) = source();

        let mut two = TangentCircleGesture::default();
        two.click(
            owner,
            TangentCircleKind::Two,
            &source,
            Some((bottom, SketchPoint::new(2, 0))),
            None,
            context,
        );
        two.click(
            owner,
            TangentCircleKind::Two,
            &source,
            Some((left, SketchPoint::new(0, 2))),
            None,
            context,
        );
        let TangentCircleEdit::Document(made) = two.click(
            owner,
            TangentCircleKind::Two,
            &source,
            None,
            Some(SketchPoint::new(2, 3)),
            context,
        ) else {
            panic!("two tangent completion")
        };
        assert_eq!(made.sketch.constraints().len(), 2);

        let mut three = TangentCircleGesture::default();
        three.click(
            owner,
            TangentCircleKind::Three,
            &source,
            Some((bottom, SketchPoint::new(5, 0))),
            None,
            context,
        );
        three.click(
            owner,
            TangentCircleKind::Three,
            &source,
            Some((diagonal, SketchPoint::new(5, 5))),
            None,
            context,
        );
        let TangentCircleEdit::Document(made) = three.click(
            owner,
            TangentCircleKind::Three,
            &source,
            Some((left, SketchPoint::new(0, 5))),
            None,
            context,
        ) else {
            panic!("three tangent completion")
        };
        assert_eq!(made.sketch.constraints().len(), 3);
    }
}
