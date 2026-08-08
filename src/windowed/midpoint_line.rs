//! Session-only state for the two-click Midpoint Line command.

use document::scene::NodeId;
use document::sketch::{MidpointLinePlacement, SketchPoint, SketchSolid};

use super::sketch_target::ResolvedSketchTarget;

/// The construction midpoint held between clicks. Point identity is deliberately absent: a
/// midpoint is a positional input, never authored geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
struct PendingMidpointLine {
    owner: NodeId,
    at: SketchPoint,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum MidpointLineEdit {
    SessionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct MidpointLineGesture {
    pending: Option<PendingMidpointLine>,
}

impl MidpointLineGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .map(|pending| pending.at)
            .collect()
    }

    #[cfg(test)]
    const fn pending(self) -> Option<PendingMidpointLine> {
        self.pending
    }

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

    /// An active pending midpoint consumes the command-level Enter dispatch while leaving the
    /// gesture untouched. Enter is not a coordinate source for this two-click tool.
    pub const fn blocks_enter(self, tool_is_active: bool, constraint_is_armed: bool) -> bool {
        tool_is_active && !constraint_is_armed && self.pending.is_some()
    }

    /// Resolve the exact canonical preview without changing gesture state. The document adapter
    /// remains the single authority for reflection, refusal, and stored endpoint reuse.
    pub fn placement(
        self,
        owner: NodeId,
        producer: &SketchSolid,
        endpoint: ResolvedSketchTarget,
    ) -> Option<MidpointLinePlacement> {
        let pending = self.pending.filter(|pending| pending.owner == owner)?;
        producer
            .midpoint_line_placement_from_canonical(pending.at, endpoint.at, endpoint.existing)
            .ok()
    }

    /// Advance one stationary click. The first stores only a canonical midpoint. Every later
    /// click consumes that pending input before attempting completion, so a miss or refusal is
    /// repeat-ready and cannot strand a hidden half-gesture.
    pub fn click(
        &mut self,
        owner: NodeId,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
    ) -> MidpointLineEdit {
        let Some(pending) = self.pending.take() else {
            if let Some(target) = target {
                self.pending = Some(PendingMidpointLine {
                    owner,
                    at: target.at,
                });
            }
            return MidpointLineEdit::SessionOnly;
        };
        if pending.owner != owner {
            if let Some(target) = target {
                self.pending = Some(PendingMidpointLine {
                    owner,
                    at: target.at,
                });
            }
            return MidpointLineEdit::SessionOnly;
        }
        let Some(endpoint) = target else {
            return MidpointLineEdit::SessionOnly;
        };
        producer
            .with_midpoint_line_from_canonical(pending.at, endpoint.at, endpoint.existing)
            .map_or(MidpointLineEdit::SessionOnly, |(next, _)| {
                MidpointLineEdit::Document(next)
            })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};
    use parametric::units::Measurement;
    use voxel_core::core_geom::MaterialChoice;

    fn empty() -> SketchSolid {
        SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3)
    }

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget {
            at,
            existing: None,
            on_curve: None,
        }
    }

    #[test]
    fn first_click_is_session_only_and_completion_is_one_segment_then_repeat_ready() {
        let owner = NodeId(10);
        let source = empty();
        let mut gesture = MidpointLineGesture::default();
        assert_eq!(
            gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0)))),
            MidpointLineEdit::SessionOnly
        );
        assert_eq!(
            source.sketch.points().len(),
            0,
            "first click edits no document"
        );
        assert_eq!(gesture.pending().unwrap().at, SketchPoint::new(5, 0));

        let MidpointLineEdit::Document(made) =
            gesture.click(owner, &source, Some(target(SketchPoint::new(8, 0))))
        else {
            panic!("second click completes")
        };
        assert_eq!(made.sketch.points().len(), 2);
        assert_eq!(made.sketch.segments().len(), 1);
        assert!(made.sketch.constraints().is_empty());
        assert_eq!(gesture.pending(), None);

        assert_eq!(
            gesture.click(owner, &made, Some(target(SketchPoint::new(20, 0)))),
            MidpointLineEdit::SessionOnly
        );
        assert_eq!(gesture.pending().unwrap().at, SketchPoint::new(20, 0));
    }

    #[test]
    fn completion_reaches_app_core_as_exactly_one_undoable_edit() {
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
        let mut gesture = MidpointLineGesture::default();

        assert_eq!(
            gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0)))),
            MidpointLineEdit::SessionOnly
        );
        assert_eq!(core.undo_depth(), 0, "first click queues no AppCore edit");

        let MidpointLineEdit::Document(made) =
            gesture.click(owner, &source, Some(target(SketchPoint::new(8, 0))))
        else {
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
        assert_eq!(core.undo_depth(), 1, "completion is one undoable act");
        core.undo(&mut scene, &mut selection);
        let restored = match &scene.node_by_id(owner).unwrap().content {
            document::scene::NodeContent::SketchTool { producer, .. } => producer,
            _ => panic!("sketch node"),
        };
        assert_eq!(restored, &source);
    }

    #[test]
    fn preview_and_commit_share_exact_positions_and_reuse_a_stored_endpoint() {
        let owner = NodeId(10);
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let mut endpoint_at = SketchPoint::new(8, 0);
        endpoint_at.offset_measurements =
            Some([Measurement::from_voxels(8), Measurement::from_voxels(0)]);
        let endpoint = sketch.add_free_point(endpoint_at);
        let source = SketchSolid::extrude(sketch, 3);
        let endpoint_target = ResolvedSketchTarget {
            at: endpoint_at,
            existing: Some(endpoint),
            on_curve: None,
        };
        let mut gesture = MidpointLineGesture::default();
        gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0))));
        let placement = gesture.placement(owner, &source, endpoint_target).unwrap();

        let MidpointLineEdit::Document(made) = gesture.click(owner, &source, Some(endpoint_target))
        else {
            panic!("completion")
        };
        let segment = made.sketch.segments()[0];
        assert_eq!(segment.from, endpoint);
        let at = |id| {
            made.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .unwrap()
                .at
        };
        assert!(placement.endpoint.coincides(&at(segment.from)));
        assert!(placement.reflected.coincides(&at(segment.to)));
    }

    #[test]
    fn large_split_midpoint_stays_exact_from_pending_preview_into_commit() {
        let owner = NodeId(10);
        let source = empty();
        let midpoint = SketchPoint {
            offset_voxels: [1_i64 << 62, -(1_i64 << 62)],
            offset_local_voxels: [0.5, 0.25],
            offset_measurements: None,
        };
        let endpoint = SketchPoint {
            offset_voxels: [(1_i64 << 62) + 1024, -(1_i64 << 62) + 1024],
            offset_local_voxels: [0.5, 0.25],
            offset_measurements: None,
        };
        let mut gesture = MidpointLineGesture::default();
        gesture.click(owner, &source, Some(target(midpoint)));
        assert_eq!(gesture.pending().unwrap().at, midpoint);
        let placement = gesture.placement(owner, &source, target(endpoint)).unwrap();
        assert_eq!(placement.midpoint, midpoint);

        let MidpointLineEdit::Document(made) =
            gesture.click(owner, &source, Some(target(endpoint)))
        else {
            panic!("exact split-coordinate completion")
        };
        let segment = made.sketch.segments()[0];
        let point = |id| {
            made.sketch
                .points()
                .iter()
                .find(|point| point.id == id)
                .unwrap()
                .at
        };
        assert!(placement.endpoint.coincides(&point(segment.from)));
        assert!(placement.reflected.coincides(&point(segment.to)));
    }

    #[test]
    fn refusal_or_missed_second_click_consumes_pending_without_an_edit() {
        let owner = NodeId(10);
        let source = empty();
        for second in [Some(target(SketchPoint::new(5, 0))), None] {
            let mut gesture = MidpointLineGesture::default();
            gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0))));
            assert_eq!(
                gesture.click(owner, &source, second),
                MidpointLineEdit::SessionOnly
            );
            assert_eq!(gesture.pending(), None);
            assert!(source.sketch.points().is_empty());
        }

        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let endpoint = sketch.add_free_point(SketchPoint::new(8, 0));
        let reflected = sketch.add_free_point(SketchPoint::new(2, 0));
        sketch.connect(endpoint, reflected).unwrap();
        let duplicate = SketchSolid::extrude(sketch, 3);
        let before = serde_json::to_string(&duplicate).unwrap();
        let mut gesture = MidpointLineGesture::default();
        gesture.click(owner, &duplicate, Some(target(SketchPoint::new(5, 0))));
        assert_eq!(
            gesture.click(
                owner,
                &duplicate,
                Some(ResolvedSketchTarget {
                    at: SketchPoint::new(8, 0),
                    existing: Some(endpoint),
                    on_curve: None,
                }),
            ),
            MidpointLineEdit::SessionOnly
        );
        assert_eq!(gesture.pending(), None);
        assert_eq!(serde_json::to_string(&duplicate).unwrap(), before);
    }

    #[test]
    fn escape_enter_and_context_changes_have_explicit_lifecycle() {
        let owner = NodeId(10);
        let source = empty();
        let mut gesture = MidpointLineGesture::default();
        gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0))));
        assert!(gesture.blocks_enter(true, false));
        assert!(
            gesture.pending().is_some(),
            "Enter leaves the midpoint intact"
        );
        assert!(gesture.cancel_for_escape(true, false));
        assert!(!gesture.cancel_for_escape(true, false));

        for (armed, constrained, active_owner) in [
            (false, false, Some(owner)),
            (true, true, Some(owner)),
            (true, false, Some(NodeId(11))),
            (true, false, None),
        ] {
            gesture.click(owner, &source, Some(target(SketchPoint::new(5, 0))));
            gesture.retain_for_context(armed, constrained, active_owner);
            assert_eq!(gesture.pending(), None);
        }
    }
}
