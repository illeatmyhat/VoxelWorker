//! Interaction-transient state for the three-click Center Point Arc command.

use document::scene::NodeId;
use document::sketch::{CenterArcPlacement, EntityId, SketchPoint, SketchSolid};
use parametric::EvaluationContext;

use document::sketch::SketchTarget;

#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingCenterArc {
    owner: NodeId,
    center: SketchPoint,
    start: Option<SketchTarget>,
    /// Which way the cursor has been going about the center since the start point landed.
    ///
    /// The arc's direction is a property of the PATH the cursor took, not of where it currently
    /// is: the same point on the circle is reachable either way round. Living inside the pending
    /// record is what keeps it honest — every reset, cancel and context change that drops the
    /// gesture drops this with it, so there is no roster to keep in sync and no way for a stale
    /// direction to outlive the arc it described.
    winding: Option<substrate::winding::TurnLatch>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum CenterArcEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct CenterArcGesture {
    pending: Option<PendingCenterArc>,
}

impl CenterArcGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .flat_map(|pending| {
                std::iter::once(pending.center).chain(pending.start.map(|start| start.at()))
            })
            .collect()
    }

    #[cfg(test)]
    pub const fn is_pending(self) -> bool {
        self.pending.is_some()
    }

    pub fn center(self, owner: NodeId) -> Option<SketchPoint> {
        match self.pending {
            Some(pending) if pending.owner == owner => Some(pending.center),
            _ => None,
        }
    }

    pub fn start(self, owner: NodeId) -> Option<SketchTarget> {
        match self.pending {
            Some(PendingCenterArc {
                owner: pending_owner,
                start,
                ..
            }) if pending_owner == owner => start,
            _ => None,
        }
    }

    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        tool_is_armed: bool,
        constraint_is_armed: bool,
        owner: Option<NodeId>,
        producer: Option<&SketchSolid>,
    ) {
        let valid = self.pending.is_none_or(|pending| {
            tool_is_armed
                && !constraint_is_armed
                && Some(pending.owner) == owner
                && producer.is_some()
                && pending
                    .start
                    .and_then(|start| start.existing())
                    .is_none_or(|id| producer.is_some_and(|solid| point_exists(solid, id)))
        });
        if !valid {
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

    /// Fold this frame's cursor into the winding that decides which way the arc runs.
    ///
    /// Called once per frame while the arc is being aimed, BEFORE the preview is asked for, so the
    /// preview and the click that follows it read the same direction.
    pub fn track_cursor(&mut self, owner: NodeId, cursor: SketchPoint) {
        let Some(pending) = self.pending.as_mut().filter(|p| p.owner == owner) else {
            return;
        };
        let Some(start) = pending.start else {
            return;
        };
        super::arc_winding::track(&mut pending.winding, pending.center, start.at(), cursor);
    }

    pub fn placement(
        self,
        owner: NodeId,
        producer: &SketchSolid,
        direction: SketchTarget,
    ) -> Option<CenterArcPlacement> {
        let pending = self.pending.filter(|pending| pending.owner == owner)?;
        let start = pending.start?;
        producer
            .center_arc_placement(
                pending.center,
                start,
                direction.at(),
                super::arc_winding::turn(pending.winding),
            )
            .ok()
    }

    /// Advance one click. Only the third click can produce a document edit; any invalid later pick
    /// consumes the partial gesture so the still-armed tool is immediately repeat-ready.
    pub fn click(
        &mut self,
        owner: NodeId,
        producer: &SketchSolid,
        target: Option<SketchTarget>,
        context: EvaluationContext,
    ) -> CenterArcEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingCenterArc {
                    owner,
                    center: target.at(),
                    start: None,
                    winding: None,
                });
            }
            return CenterArcEdit::InteractionOnly;
        };
        if pending.owner != owner {
            if let Some(target) = target {
                self.pending = Some(PendingCenterArc {
                    owner,
                    center: target.at(),
                    start: None,
                    winding: None,
                });
            }
            return CenterArcEdit::InteractionOnly;
        }
        let Some(target) = target else {
            return CenterArcEdit::InteractionOnly;
        };
        let Some(start) = pending.start else {
            if !pending.center.coincides(&target.at()) {
                self.pending = Some(PendingCenterArc {
                    start: Some(target),
                    ..pending
                });
            }
            return CenterArcEdit::InteractionOnly;
        };
        // The click's own position is the last reading, so a commit and the preview it replaces
        // cannot disagree about the direction even if no frame rendered in between.
        let mut winding = pending.winding;
        super::arc_winding::track(&mut winding, pending.center, start.at(), target.at());
        producer
            .with_center_arc(
                pending.center,
                start,
                target.at(),
                super::arc_winding::turn(winding),
                context,
            )
            .map_or(CenterArcEdit::InteractionOnly, |(next, _)| {
                CenterArcEdit::Document(next)
            })
    }
}

fn point_exists(producer: &SketchSolid, id: EntityId) -> bool {
    producer.sketch.points().iter().any(|point| point.id == id)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn context() -> EvaluationContext {
        EvaluationContext::new(std::num::NonZeroU32::new(16).unwrap())
    }
    use document::sketch::{PlaneAxis, Sketch};
    use voxel_core::core_geom::MaterialChoice;

    fn empty() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    fn target(at: SketchPoint) -> SketchTarget {
        SketchTarget::fresh(at)
    }

    #[test]
    fn three_click_completion_is_atomic_projected_and_repeat_ready() {
        let owner = NodeId(7);
        let source = empty();
        let mut gesture = CenterArcGesture::default();
        assert_eq!(
            gesture.click(
                owner,
                &source,
                Some(target(SketchPoint::new(0, 0))),
                context()
            ),
            CenterArcEdit::InteractionOnly
        );
        assert_eq!(
            gesture.click(
                owner,
                &source,
                Some(target(SketchPoint::new(4, 0))),
                context()
            ),
            CenterArcEdit::InteractionOnly
        );
        assert!(source.sketch.points().is_empty());
        let preview = gesture
            .placement(owner, &source, target(SketchPoint::new(0, 9)))
            .unwrap();

        let CenterArcEdit::Document(made) = gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(0, 9))),
            context(),
        ) else {
            panic!("third click completes")
        };
        assert!(!gesture.is_pending());
        assert_eq!(made.sketch.arcs().len(), 1);
        let arc = made.sketch.arcs()[0];
        let endpoint = made
            .sketch
            .points()
            .iter()
            .find(|point| point.id == arc.to)
            .unwrap()
            .at;
        assert!(endpoint.coincides(&preview.endpoint));
    }

    /// Two gestures with identical picks, separated only by the route the cursor took.
    #[test]
    fn the_way_the_cursor_went_round_decides_which_arc_is_made() {
        let owner = NodeId(7);
        let source = empty();
        let center = SketchPoint::new(0, 0);
        let start = SketchPoint::new(8, 0);
        let end = SketchPoint::new(-8, 0);

        let mut sweeps = [
            (CenterArcGesture::default(), 1.0_f64),
            (CenterArcGesture::default(), -1.0_f64),
        ];
        for (gesture, sign) in &mut sweeps {
            gesture.click(owner, &source, Some(target(center)), context());
            gesture.click(owner, &source, Some(target(start)), context());
            for step in 1..=8 {
                let angle = *sign * f64::from(step) / 8.0 * std::f64::consts::PI;
                let cursor =
                    SketchPoint::try_from_continuous(8.0 * angle.cos(), 8.0 * angle.sin()).unwrap();
                gesture.track_cursor(owner, cursor);
            }
        }

        let counter_clockwise = sweeps[0]
            .0
            .placement(owner, &source, target(end))
            .unwrap()
            .candidate
            .sweep_radians;
        let clockwise = sweeps[1]
            .0
            .placement(owner, &source, target(end))
            .unwrap()
            .candidate
            .sweep_radians;
        assert!(counter_clockwise > 0.0, "{counter_clockwise}");
        assert!(clockwise < 0.0, "{clockwise}");
    }

    #[test]
    fn cancellation_and_context_invalidation_drop_only_transient_state() {
        let owner = NodeId(7);
        let source = empty();
        let mut gesture = CenterArcGesture::default();
        gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(0, 0))),
            context(),
        );
        assert!(gesture.blocks_enter(true, false));
        assert!(gesture.cancel_for_escape(true, false));
        assert!(!gesture.is_pending());

        gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(0, 0))),
            context(),
        );
        gesture.retain_for_context(true, false, Some(NodeId(8)), Some(&source));
        assert!(!gesture.is_pending());
    }

    #[test]
    fn completion_reaches_app_core_as_one_undoable_edit() {
        let source = empty();
        let mut scene = document::scene::Scene::from_nodes(vec![document::scene::Node::new(
            "Sketch",
            document::scene::NodeContent::SketchTool {
                producer: source.clone(),
                material: MaterialChoice::Stone,
            },
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let owner = scene.roots.first().copied().unwrap();
        let mut core = crate::AppCore::new(camera::OrbitCamera::default());
        let mut selection = ui::panel::Selection::default();
        let mut gesture = CenterArcGesture::default();

        gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(0, 0))),
            context(),
        );
        gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(4, 0))),
            context(),
        );
        assert_eq!(core.undo_depth(), 0);
        let CenterArcEdit::Document(made) = gesture.click(
            owner,
            &source,
            Some(target(SketchPoint::new(0, 4))),
            context(),
        ) else {
            panic!("completion")
        };
        core.apply_transaction(
            &mut scene,
            &mut selection,
            super::super::render::sketch_profile_edit_transaction(
                owner,
                made,
                [0, 0, 0],
                [0, 0, 0],
            ),
        );
        assert_eq!(core.undo_depth(), 1);
        core.undo(&mut scene, &mut selection);
        let restored = match &scene.node_by_id(owner).unwrap().content {
            document::scene::NodeContent::SketchTool { producer, .. } => producer,
            _ => panic!("sketch node"),
        };
        assert_eq!(restored, &source);
    }
}
