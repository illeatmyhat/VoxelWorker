//! Interaction-transient state shared by the three regular-polygon tools.

use document::scene::NodeId;
use document::sketch::{PolygonPlacement, SketchPoint, SketchSolid};

use super::sketch_target::ResolvedSketchTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PolygonKind {
    Inscribed,
    Circumscribed,
    Edge,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingPolygon {
    owner: NodeId,
    kind: PolygonKind,
    first: SketchPoint,
    second: Option<SketchPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PolygonEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct PolygonGesture {
    pending: Option<PendingPolygon>,
}

impl PolygonGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .flat_map(|pending| std::iter::once(pending.first).chain(pending.second))
            .collect()
    }

    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<PolygonKind>,
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
        active_kind: Option<PolygonKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub const fn blocks_enter(
        self,
        active_kind: Option<PolygonKind>,
        constraint_is_armed: bool,
    ) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    pub fn guide(
        self,
        owner: NodeId,
        kind: PolygonKind,
    ) -> Option<(SketchPoint, Option<SketchPoint>)> {
        self.pending
            .filter(|pending| pending.owner == owner && pending.kind == kind)
            .map(|pending| (pending.first, pending.second))
    }

    pub fn placement(
        self,
        owner: NodeId,
        kind: PolygonKind,
        producer: &SketchSolid,
        cursor: ResolvedSketchTarget,
        sides: u16,
    ) -> Option<PolygonPlacement> {
        let pending = self
            .pending
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        match (kind, pending.second) {
            (PolygonKind::Inscribed, None) => producer
                .inscribed_polygon_placement(pending.first, cursor.at, sides)
                .ok(),
            (PolygonKind::Circumscribed, None) => producer
                .circumscribed_polygon_placement(pending.first, cursor.at, sides)
                .ok(),
            (PolygonKind::Edge, Some(second)) => producer
                .edge_polygon_placement(pending.first, second, cursor.at, sides)
                .ok(),
            (PolygonKind::Edge, None)
            | (PolygonKind::Inscribed | PolygonKind::Circumscribed, Some(_)) => None,
        }
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: PolygonKind,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
        sides: u16,
    ) -> PolygonEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingPolygon {
                    owner,
                    kind,
                    first: target.at,
                    second: None,
                });
            }
            return PolygonEdit::InteractionOnly;
        };
        if pending.owner != owner || pending.kind != kind {
            if let Some(target) = target {
                self.pending = Some(PendingPolygon {
                    owner,
                    kind,
                    first: target.at,
                    second: None,
                });
            }
            return PolygonEdit::InteractionOnly;
        }
        let Some(target) = target else {
            return PolygonEdit::InteractionOnly;
        };
        match (kind, pending.second) {
            (PolygonKind::Inscribed, None) => producer
                .with_inscribed_polygon(pending.first, target.at, sides)
                .map_or(PolygonEdit::InteractionOnly, PolygonEdit::Document),
            (PolygonKind::Circumscribed, None) => producer
                .with_circumscribed_polygon(pending.first, target.at, sides)
                .map_or(PolygonEdit::InteractionOnly, PolygonEdit::Document),
            (PolygonKind::Edge, None) => {
                if !pending.first.coincides(&target.at) {
                    self.pending = Some(PendingPolygon {
                        second: Some(target.at),
                        ..pending
                    });
                }
                PolygonEdit::InteractionOnly
            }
            (PolygonKind::Edge, Some(second)) => producer
                .with_edge_polygon(pending.first, second, target.at, sides)
                .map_or(PolygonEdit::InteractionOnly, PolygonEdit::Document),
            (PolygonKind::Inscribed | PolygonKind::Circumscribed, Some(_)) => {
                PolygonEdit::InteractionOnly
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget {
            at,
            existing: None,
            on_curve: None,
        }
    }

    #[test]
    fn every_polygon_grammar_commits_atomically_and_repeat_ready() {
        let owner = NodeId(4);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for (kind, points) in [
            (
                PolygonKind::Inscribed,
                vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)],
            ),
            (
                PolygonKind::Circumscribed,
                vec![SketchPoint::new(0, 0), SketchPoint::new(4, 0)],
            ),
            (
                PolygonKind::Edge,
                vec![
                    SketchPoint::new(0, 0),
                    SketchPoint::new(4, 0),
                    SketchPoint::new(2, 3),
                ],
            ),
        ] {
            let mut gesture = PolygonGesture::default();
            let mut result = PolygonEdit::InteractionOnly;
            for point in points {
                result = gesture.click(owner, kind, &source, Some(target(point)), 5);
            }
            let PolygonEdit::Document(made) = result else {
                panic!("final click completes")
            };
            assert_eq!(made.sketch.segments().len(), 5);
            assert!(source.sketch.points().is_empty());
            assert!(gesture.pending.is_none());
        }
    }

    #[test]
    fn preview_and_commit_share_the_same_canonical_vertices() {
        let owner = NodeId(7);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = PolygonGesture::default();
        gesture.click(
            owner,
            PolygonKind::Inscribed,
            &source,
            Some(target(SketchPoint::new(0, 0))),
            8,
        );
        let cursor = target(SketchPoint::new(6, 0));
        let preview = gesture
            .placement(owner, PolygonKind::Inscribed, &source, cursor, 8)
            .unwrap();
        let PolygonEdit::Document(made) =
            gesture.click(owner, PolygonKind::Inscribed, &source, Some(cursor), 8)
        else {
            panic!("completion")
        };
        for vertex in preview.vertices {
            assert!(made.sketch.point_at(vertex).is_some());
        }
    }
}
