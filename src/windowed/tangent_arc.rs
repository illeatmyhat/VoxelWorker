//! Interaction-transient state for the standalone two-click Tangent Arc command.

use document::scene::NodeId;
use document::sketch::{
    EntityId, SketchCurve, SketchSolid, TangentArcPlacement, TangentArcRefusal,
};
use parametric::EvaluationContext;

use super::sketch_target::ResolvedSketchTarget;

/// A supported incoming curve and the endpoint where the tangent arc must leave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TangentArcSource {
    pub curve: SketchCurve,
    pub seam: EntityId,
}

/// Validate a semantic source independently of cursor pixels. Closed circles have no endpoint
/// seam, so the standalone tool deliberately accepts only incident segments and arcs.
pub(super) fn resolve_source(
    producer: &SketchSolid,
    curve: SketchCurve,
    seam: EntityId,
) -> Option<TangentArcSource> {
    let incident = match curve {
        SketchCurve::Segment(id) => producer
            .sketch
            .segments()
            .iter()
            .find(|segment| segment.id == id)
            .is_some_and(|segment| segment.from == seam || segment.to == seam),
        SketchCurve::Arc(id) => producer
            .sketch
            .arcs()
            .iter()
            .find(|arc| arc.id == id)
            .is_some_and(|arc| arc.from == seam || arc.to == seam),
        SketchCurve::Circle(_) | SketchCurve::Bezier(_) => false,
    };
    incident.then_some(TangentArcSource { curve, seam })
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum TangentArcEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct TangentArcGesture {
    pending: Option<(NodeId, TangentArcSource)>,
}

impl TangentArcGesture {
    pub const fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    #[cfg(test)]
    const fn pending(&self) -> Option<(NodeId, TangentArcSource)> {
        self.pending
    }

    pub fn begin(&mut self, owner: NodeId, source: TangentArcSource) {
        self.pending = Some((owner, source));
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
        let valid = self.pending.is_none_or(|(pending_owner, source)| {
            tool_is_armed
                && !constraint_is_armed
                && Some(pending_owner) == owner
                && producer.and_then(|solid| resolve_source(solid, source.curve, source.seam))
                    == Some(source)
        });
        if !valid {
            self.reset();
        }
    }

    pub fn cancel_for_escape(&mut self, tool_is_active: bool, constraint_is_armed: bool) -> bool {
        let was_live = self.reset();
        tool_is_active && !constraint_is_armed && was_live
    }

    pub const fn blocks_enter(&self, tool_is_active: bool, constraint_is_armed: bool) -> bool {
        tool_is_active && !constraint_is_armed && self.pending.is_some()
    }

    pub fn placement(
        &self,
        owner: NodeId,
        producer: &SketchSolid,
        endpoint: ResolvedSketchTarget,
        context: EvaluationContext,
    ) -> Result<TangentArcPlacement, TangentArcRefusal> {
        let (_, source) = self
            .pending
            .filter(|(pending_owner, _)| *pending_owner == owner)
            .ok_or(TangentArcRefusal::UnknownIncoming)?;
        producer.tangent_arc_placement_to(
            source.curve,
            source.seam,
            endpoint.at,
            endpoint.existing,
            context,
        )
    }

    /// Consume the held source before attempting completion. A miss or refusal therefore leaves
    /// the still-armed tool ready for a new first click and records no document edit.
    pub fn complete(
        &mut self,
        owner: NodeId,
        producer: &SketchSolid,
        endpoint: Option<ResolvedSketchTarget>,
        context: EvaluationContext,
    ) -> TangentArcEdit {
        let Some((pending_owner, source)) = self.pending.take() else {
            return TangentArcEdit::InteractionOnly;
        };
        if pending_owner != owner {
            return TangentArcEdit::InteractionOnly;
        }
        let Some(endpoint) = endpoint else {
            return TangentArcEdit::InteractionOnly;
        };
        producer
            .with_tangent_arc_to(
                source.curve,
                source.seam,
                endpoint.at,
                endpoint.existing,
                context,
            )
            .map_or(TangentArcEdit::InteractionOnly, |(next, _)| {
                TangentArcEdit::Document(next)
            })
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch, SketchPoint};
    use std::num::NonZeroU32;
    use voxel_core::core_geom::MaterialChoice;

    fn context() -> EvaluationContext {
        EvaluationContext::new(NonZeroU32::new(16).unwrap())
    }

    fn incoming_segment() -> (SketchSolid, TangentArcSource) {
        let mut sketch = Sketch::empty(PlaneAxis::Z);
        let tail = sketch.add_free_point(SketchPoint::new(0, 0));
        let seam = sketch.add_free_point(SketchPoint::new(10, 0));
        let segment = sketch.connect(tail, seam).unwrap();
        let solid = SketchSolid::extrude(sketch, 3);
        let source = resolve_source(&solid, SketchCurve::Segment(segment), seam).unwrap();
        (solid, source)
    }

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget { at, existing: None }
    }

    #[test]
    fn source_resolution_requires_an_incident_open_curve() {
        let (solid, source) = incoming_segment();
        assert_eq!(
            resolve_source(&solid, source.curve, source.seam),
            Some(source)
        );
        assert_eq!(resolve_source(&solid, source.curve, 9999), None);

        let mut circle_solid = solid;
        let circle = circle_solid
            .sketch
            .add_circle(
                SketchPoint::new(30, 30),
                document::sketch::SketchLength::new(4),
            )
            .unwrap();
        assert_eq!(
            resolve_source(&circle_solid, SketchCurve::Circle(circle), source.seam),
            None
        );
    }

    #[test]
    fn completion_is_atomic_repeat_ready_and_exposes_radius() {
        let owner = NodeId(10);
        let (solid, source) = incoming_segment();
        let mut gesture = TangentArcGesture::default();
        gesture.begin(owner, source);
        let placement = gesture
            .placement(owner, &solid, target(SketchPoint::new(10, 10)), context())
            .unwrap();
        assert!(placement.candidate.radius.is_finite());
        assert!(placement.candidate.radius > 0.0);

        let TangentArcEdit::Document(made) = gesture.complete(
            owner,
            &solid,
            Some(target(SketchPoint::new(10, 10))),
            context(),
        ) else {
            panic!("valid completion")
        };
        assert!(!gesture.is_pending());
        assert_eq!(made.sketch.arcs().len(), 1);
        assert_eq!(made.sketch.constraints().len(), 1);
        assert_eq!(solid.sketch.arcs().len(), 0);
    }

    #[test]
    fn completion_reaches_app_core_as_one_undoable_edit() {
        let (solid, source) = incoming_segment();
        let mut scene = document::scene::Scene::from_nodes(vec![document::scene::Node::new(
            "Sketch",
            document::scene::NodeContent::SketchTool {
                producer: solid.clone(),
                material: MaterialChoice::Stone,
            },
        )]);
        scene.ensure_node_ids();
        scene.ensure_origin_point();
        let owner = scene.roots.first().copied().unwrap();
        let mut core = crate::AppCore::new(camera::OrbitCamera::default());
        let mut selection = ui::panel::Selection::default();
        let mut gesture = TangentArcGesture::default();
        gesture.begin(owner, source);
        assert_eq!(core.undo_depth(), 0);

        let TangentArcEdit::Document(made) = gesture.complete(
            owner,
            &solid,
            Some(target(SketchPoint::new(10, 10))),
            context(),
        ) else {
            panic!("valid completion")
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
        assert_eq!(restored, &solid);
    }

    #[test]
    fn refusal_consumes_pending_and_context_lifecycle_is_explicit() {
        let owner = NodeId(10);
        let (solid, source) = incoming_segment();
        let mut gesture = TangentArcGesture::default();
        gesture.begin(owner, source);
        assert_eq!(
            gesture.complete(
                owner,
                &solid,
                Some(target(SketchPoint::new(20, 0))),
                context(),
            ),
            TangentArcEdit::InteractionOnly
        );
        assert!(!gesture.is_pending());

        gesture.begin(owner, source);
        assert!(gesture.blocks_enter(true, false));
        assert!(gesture.cancel_for_escape(true, false));
        assert!(!gesture.cancel_for_escape(true, false));

        gesture.begin(owner, source);
        gesture.retain_for_context(true, false, Some(NodeId(11)), Some(&solid));
        assert_eq!(gesture.pending(), None);
    }
}
