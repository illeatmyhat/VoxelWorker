//! The workspace selection — one mixed-kind set of picked targets (ADR 0032).
//!
//! ADR 0032 repeals the node side's law (`Scene::active`, document state restored by undo)
//! in favour of the sketch side's: selection is **workspace** state. It never travels in a
//! shared file, never enters undo history, and rides the dump. Edits still *steer* it as an
//! effect — a created node arrives selected, undoing a delete re-selects what came back —
//! but that is a workspace write, not document truth.
//!
//! One set holds every kind of [`SelectionTarget`] rather than one set per kind, so a marquee
//! over a box and a Point returns both. Which kinds may enter is a property of the editing
//! mode (an admission filter), never a second data structure.

use document::scene::{NodeContent, NodeId, PointId, Scene};
use document::sketch::EntityId;

/// One picked thing. ADR 0032 keeps these in ONE set: mode exclusivity is an admission
/// filter, not a reason for parallel structures.
///
/// The sketch variants carry their owning sketch node, not just the entity id, because an
/// [`EntityId`] is minted from a per-sketch counter and means nothing without its scope —
/// the same law ADR 0008 states for spatial values. It also makes a target self-contained
/// for the one consumer that must find the owning producer to edit it (the sketch delete),
/// and it lets a restore sweep drop targets whose sketch left the scene, at the seam where
/// `to_panel_state` already drops a stale `sketch_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectionTarget {
    /// A scene-graph node, at any depth (ADR 0001) — but an instance picks as itself, never
    /// into its definition (ADR 0017).
    Node(NodeId),
    /// A reference Point, by its stable id (ADR 0033) — never its `Vec` slot, which
    /// shifts on `RemovePoint` and would silently re-point at a different Point.
    ReferencePoint(PointId),
    /// A vertex of a sketch profile (ADR 0030), addressable only from inside that sketch.
    SketchPoint {
        /// The sketch node that owns the entity counter this id came from.
        sketch: NodeId,
        /// The point's id within that sketch.
        entity: EntityId,
    },
    /// An edge of a sketch profile (ADR 0030). Deleting one leaves its endpoints as free
    /// points, unlike deleting a vertex — which is why the kind is a variant, not a flag.
    SketchSegment {
        /// The sketch node that owns the entity counter this id came from.
        sketch: NodeId,
        /// The segment's id within that sketch.
        entity: EntityId,
    },
}

impl SelectionTarget {
    /// The sketch this target belongs to, or `None` for a kind that is not a sketch entity.
    /// The admission question — "may this enter while that sketch is open?" — asked of a
    /// target rather than of a parallel data structure.
    pub fn owning_sketch(self) -> Option<NodeId> {
        match self {
            SelectionTarget::SketchPoint { sketch, .. }
            | SelectionTarget::SketchSegment { sketch, .. } => Some(sketch),
            SelectionTarget::Node(_) | SelectionTarget::ReferencePoint(_) => None,
        }
    }
}

/// How a click asked the selection to change (ADR 0032). A VIEW action, not an
/// [`Intent`](document::intent::Intent): selecting is not an edit, so it rides on
/// [`PanelResponse`](super::PanelResponse) and the shell applies it — the same route
/// `focus_node` and `arm_tool` take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionRequest {
    /// A plain click: replace the whole selection with this target.
    Only(SelectionTarget),
    /// A Shift-click: add this target, or drop it if it was already picked. Multi-select
    /// exists the moment this does (ADR 0032) — there is no separate "add" gesture.
    Toggle(SelectionTarget),
    /// A click on empty space, or a deselect: drop everything.
    Clear,
}

/// The set of picked [`SelectionTarget`]s, in pick order (newest last).
///
/// Ordered rather than sorted so "the primary" — what the inspector mirrors and what an
/// active-keyed op acts on — is the most recently picked target of its kind, which is what
/// clicking a second node means.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Selection {
    targets: Vec<SelectionTarget>,
}

impl Selection {
    /// Rebuild a selection from its targets in pick order — the restore half of the
    /// session round-trip (the capture half is [`targets`](Self::targets)).
    pub fn from_targets(targets: impl IntoIterator<Item = SelectionTarget>) -> Self {
        Self {
            targets: targets.into_iter().collect(),
        }
    }

    /// Nothing is picked.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// How many targets are picked (the inspector shows a count summary above 1).
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// Drop the whole selection (a click on empty space).
    pub fn clear(&mut self) {
        self.targets.clear();
    }

    /// Is this target picked?
    pub fn contains(&self, target: SelectionTarget) -> bool {
        self.targets.contains(&target)
    }

    /// Every picked target, in pick order.
    pub fn targets(&self) -> impl Iterator<Item = SelectionTarget> + '_ {
        self.targets.iter().copied()
    }

    /// The most recently picked node, or `None` when no node is picked. The successor of
    /// `Scene::active`: what the inspector mirrors and what an active-keyed op acts on.
    pub fn primary_node_id(&self) -> Option<NodeId> {
        self.targets.iter().rev().find_map(|target| match target {
            SelectionTarget::Node(id) => Some(*id),
            _ => None,
        })
    }

    /// Every picked node id, in pick order — the multi-target verbs' read (Delete acts
    /// on all of these, ADR 0033), where [`primary_node_id`](Self::primary_node_id) is
    /// the single-target consumers'.
    pub fn nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.targets.iter().filter_map(|target| match target {
            SelectionTarget::Node(id) => Some(*id),
            _ => None,
        })
    }

    /// The most recently picked reference Point, or `None`. The successor of
    /// `Scene::active_point`.
    pub fn primary_point_id(&self) -> Option<PointId> {
        self.targets.iter().rev().find_map(|target| match target {
            SelectionTarget::ReferencePoint(id) => Some(*id),
            _ => None,
        })
    }

    /// Land a click's [`SelectionRequest`] — the shell's single door for a view-action
    /// selection change.
    pub fn apply_request(&mut self, request: SelectionRequest) {
        match request {
            SelectionRequest::Only(target) => self.select_only(target),
            SelectionRequest::Toggle(target) => self.toggle(target),
            SelectionRequest::Clear => self.clear(),
        }
    }

    /// Replace the whole selection with one target (a plain click).
    pub fn select_only(&mut self, target: SelectionTarget) {
        self.targets.clear();
        self.targets.push(target);
    }

    /// Toggle a target in / out of the set (a modifier-click — accumulate). Re-picking an
    /// already-picked target drops it.
    pub fn toggle(&mut self, target: SelectionTarget) {
        if let Some(position) = self.targets.iter().position(|held| *held == target) {
            self.targets.remove(position);
        } else {
            self.targets.push(target);
        }
    }

    /// Replace every picked NODE with `node` (or drop them all for `None`), leaving other
    /// kinds untouched. The steer door for a single-node edit effect.
    pub fn set_primary_node(&mut self, node: Option<NodeId>) {
        self.targets
            .retain(|target| !matches!(target, SelectionTarget::Node(_)));
        if let Some(id) = node {
            self.targets.push(SelectionTarget::Node(id));
        }
    }

    /// Replace every picked reference POINT with `point` (or drop them all for `None`),
    /// leaving other kinds untouched.
    pub fn set_primary_point(&mut self, point: Option<PointId>) {
        self.targets
            .retain(|target| !matches!(target, SelectionTarget::ReferencePoint(_)));
        if let Some(id) = point {
            self.targets.push(SelectionTarget::ReferencePoint(id));
        }
    }

    /// The picked VERTEX ids of `sketch`, in pick order. Deletion is id-keyed and no-ops on
    /// an unknown id (`Sketch::delete_point_cascade`), so pick order is as good as any.
    pub fn sketch_points(&self, sketch: NodeId) -> impl Iterator<Item = EntityId> + '_ {
        self.targets.iter().filter_map(move |target| match *target {
            SelectionTarget::SketchPoint {
                sketch: owner,
                entity,
            } if owner == sketch => Some(entity),
            _ => None,
        })
    }

    /// The picked EDGE ids of `sketch`, in pick order.
    pub fn sketch_segments(&self, sketch: NodeId) -> impl Iterator<Item = EntityId> + '_ {
        self.targets.iter().filter_map(move |target| match *target {
            SelectionTarget::SketchSegment {
                sketch: owner,
                entity,
            } if owner == sketch => Some(entity),
            _ => None,
        })
    }

    /// Is anything inside `sketch` picked? What the context menu's Delete is gated on while a
    /// sketch is open.
    pub fn holds_sketch_entities(&self, sketch: NodeId) -> bool {
        self.targets
            .iter()
            .any(|target| target.owning_sketch() == Some(sketch))
    }

    /// Drop every sketch entity, keeping nodes and Points. Entering and leaving a sketch mode
    /// clears the sketch side of the selection WITHOUT disturbing what is picked outside it —
    /// which is why this is not [`clear`](Self::clear).
    pub fn clear_sketch_entities(&mut self) {
        self.targets
            .retain(|target| target.owning_sketch().is_none());
    }

    /// Drop every target that no longer resolves against `scene`, returning whether
    /// anything was dropped. The ADR 0033 rule: the undo stack carries no selection, so
    /// this — run after any document mutation (apply, undo, redo) — is the ONE place
    /// selection learns the document changed. Undoing an add leaves nothing selected,
    /// exactly like Fusion.
    pub fn prune(&mut self, scene: &Scene) -> bool {
        let before = self.targets.len();
        self.targets.retain(|target| match *target {
            SelectionTarget::Node(id) => scene.node_by_id(id).is_some(),
            SelectionTarget::ReferencePoint(id) => scene.points.iter().any(|point| point.id == id),
            SelectionTarget::SketchPoint { sketch, entity } => sketch_of(scene, sketch)
                .is_some_and(|s| s.points().iter().any(|point| point.id == entity)),
            SelectionTarget::SketchSegment { sketch, entity } => sketch_of(scene, sketch)
                .is_some_and(|s| s.segments().iter().any(|segment| segment.id == entity)),
        });
        self.targets.len() != before
    }
}

/// The sketch under `node`, when that node is a sketch producer — the prune's resolver.
fn sketch_of(scene: &Scene, node: NodeId) -> Option<&document::sketch::Sketch> {
    match &scene.node_by_id(node)?.content {
        NodeContent::SketchTool { producer, .. } => Some(&producer.sketch),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST: SelectionTarget = SelectionTarget::Node(NodeId(1));
    const SECOND: SelectionTarget = SelectionTarget::Node(NodeId(2));
    const POINT: SelectionTarget = SelectionTarget::ReferencePoint(PointId(5));

    const SKETCH: NodeId = NodeId(9);
    const OTHER_SKETCH: NodeId = NodeId(10);

    /// A vertex of the fixture sketch.
    fn vertex(entity: EntityId) -> SelectionTarget {
        SelectionTarget::SketchPoint {
            sketch: SKETCH,
            entity,
        }
    }

    /// An edge of the fixture sketch.
    fn edge(entity: EntityId) -> SelectionTarget {
        SelectionTarget::SketchSegment {
            sketch: SKETCH,
            entity,
        }
    }

    /// The primary is the most recently picked target OF ITS KIND, so a mixed selection
    /// answers both questions at once — the reason ADR 0032 holds one set, not two.
    #[test]
    fn primaries_are_per_kind_and_newest_wins() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(POINT);
        selection.toggle(SECOND);
        assert_eq!(selection.primary_node_id(), Some(NodeId(2)));
        assert_eq!(selection.primary_point_id(), Some(PointId(5)));
        assert_eq!(selection.len(), 3);
    }

    /// Steering the node selection leaves picked Points alone (an edit effect is per-kind).
    #[test]
    fn steering_nodes_spares_other_kinds() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(POINT);
        selection.set_primary_node(Some(NodeId(7)));
        assert_eq!(selection.primary_node_id(), Some(NodeId(7)));
        assert_eq!(selection.primary_point_id(), Some(PointId(5)));
        assert!(!selection.contains(FIRST));

        selection.set_primary_node(None);
        assert_eq!(selection.primary_node_id(), None);
        assert_eq!(selection.primary_point_id(), Some(PointId(5)));
    }

    /// Shift-clicking a picked target removes it; a plain click replaces everything.
    #[test]
    fn toggle_removes_and_select_only_replaces() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(SECOND);
        selection.toggle(FIRST);
        assert!(!selection.contains(FIRST));
        assert_eq!(selection.primary_node_id(), Some(NodeId(2)));

        selection.select_only(POINT);
        assert_eq!(selection.len(), 1);
        assert_eq!(selection.primary_node_id(), None);
    }

    /// A plain click **replaces**: selecting a second vertex leaves only it (ADR 0030). Ported
    /// from the retired `SketchSelection`, which stated the same law over its own set.
    #[test]
    fn selecting_a_second_vertex_replaces_the_first() {
        let mut selection = Selection::default();
        selection.select_only(vertex(1));
        selection.select_only(vertex(2));
        assert!(!selection.contains(vertex(1)));
        assert_eq!(selection.sketch_points(SKETCH).collect::<Vec<_>>(), vec![2]);
    }

    /// Shift-click accumulates, and a second toggle removes the same entity.
    #[test]
    fn toggling_vertices_accumulates_then_removes() {
        let mut selection = Selection::default();
        selection.toggle(vertex(1));
        selection.toggle(vertex(2));
        assert_eq!(
            selection.sketch_points(SKETCH).collect::<Vec<_>>(),
            vec![1, 2]
        );
        selection.toggle(vertex(1));
        assert_eq!(selection.sketch_points(SKETCH).collect::<Vec<_>>(), vec![2]);
    }

    /// A point id and a segment id are minted from the SAME per-sketch counter but name
    /// different things, so the KIND has to distinguish them — id 7 as a vertex and id 7 as an
    /// edge are two targets, and each is found by its own query.
    #[test]
    fn a_vertex_and_an_edge_of_the_same_id_are_distinct_targets() {
        let mut selection = Selection::default();
        selection.toggle(vertex(7));
        selection.toggle(edge(7));
        assert_eq!(selection.len(), 2);
        assert_eq!(selection.sketch_points(SKETCH).collect::<Vec<_>>(), vec![7]);
        assert_eq!(
            selection.sketch_segments(SKETCH).collect::<Vec<_>>(),
            vec![7]
        );
    }

    /// The whole point of tagging: the SAME entity id in two different sketches is two
    /// targets, and a query for one sketch never answers with the other's entities.
    #[test]
    fn entity_ids_do_not_collide_across_sketches() {
        let mut selection = Selection::default();
        selection.toggle(vertex(3));
        selection.toggle(SelectionTarget::SketchPoint {
            sketch: OTHER_SKETCH,
            entity: 3,
        });
        assert_eq!(selection.len(), 2);
        assert_eq!(selection.sketch_points(SKETCH).collect::<Vec<_>>(), vec![3]);
        assert_eq!(
            selection.sketch_points(OTHER_SKETCH).collect::<Vec<_>>(),
            vec![3]
        );
        assert!(selection.holds_sketch_entities(OTHER_SKETCH));
    }

    /// Entering or leaving a sketch clears the sketch side ONLY — what is picked outside it
    /// survives, which a plain `clear` would wrongly take with it.
    #[test]
    fn clearing_sketch_entities_spares_nodes_and_points() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(POINT);
        selection.toggle(vertex(1));
        selection.toggle(edge(2));

        selection.clear_sketch_entities();
        assert!(!selection.holds_sketch_entities(SKETCH));
        assert_eq!(selection.primary_node_id(), Some(NodeId(1)));
        assert_eq!(selection.primary_point_id(), Some(PointId(5)));
        assert_eq!(selection.len(), 2);
    }

    /// A sketch entity is not a node or a Point, so it never becomes either primary — the
    /// inspector keeps mirroring the node while a vertex is picked inside its sketch.
    #[test]
    fn sketch_entities_are_never_a_node_or_point_primary() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(vertex(1));
        assert_eq!(selection.primary_node_id(), Some(NodeId(1)));
        assert_eq!(selection.primary_point_id(), None);
    }

    /// ADR 0032: the three requests a click can make, through the one door the shell uses.
    /// `Toggle` is what makes multi-select exist — a Shift-click accumulates, and Shift-clicking
    /// an already-picked target removes it rather than re-picking it.
    #[test]
    fn apply_request_covers_replace_accumulate_and_clear() {
        let mut selection = Selection::default();

        selection.apply_request(SelectionRequest::Only(FIRST));
        assert_eq!(selection.targets().collect::<Vec<_>>(), vec![FIRST]);

        selection.apply_request(SelectionRequest::Toggle(SECOND));
        assert_eq!(
            selection.targets().collect::<Vec<_>>(),
            vec![FIRST, SECOND],
            "a Toggle of an unpicked target ADDS it, and it becomes the primary"
        );
        assert_eq!(selection.primary_node_id(), Some(NodeId(2)));

        selection.apply_request(SelectionRequest::Toggle(FIRST));
        assert_eq!(
            selection.targets().collect::<Vec<_>>(),
            vec![SECOND],
            "a Toggle of a picked target REMOVES it"
        );

        selection.apply_request(SelectionRequest::Only(POINT));
        assert_eq!(
            selection.targets().collect::<Vec<_>>(),
            vec![POINT],
            "a plain click replaces the whole set regardless of kind"
        );

        selection.apply_request(SelectionRequest::Clear);
        assert!(selection.is_empty());
    }
}
