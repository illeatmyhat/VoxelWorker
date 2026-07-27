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

use document::scene::NodeId;

/// One picked thing. ADR 0032 keeps these in ONE set: mode exclusivity is an admission
/// filter, not a reason for parallel structures. Sketch entities join as further variants
/// when [`SketchSelection`](super::SketchSelection) folds in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectionTarget {
    /// A scene-graph node, at any depth (ADR 0001) — but an instance picks as itself, never
    /// into its definition (ADR 0017).
    Node(NodeId),
    /// A reference Point, by its index in `Scene::points`.
    ReferencePoint(usize),
}

/// How a click asked the selection to change (ADR 0032). A VIEW action, not an
/// [`Intent`](document::intent::Intent): selecting is not an edit, so it rides on
/// [`PanelResponse`](super::PanelResponse) and the shell applies it — the same route
/// `focus_node` and `armed_tool` take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionRequest {
    /// A plain click: replace the whole selection with this target.
    Only(SelectionTarget),
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
        Self { targets: targets.into_iter().collect() }
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
            SelectionTarget::ReferencePoint(_) => None,
        })
    }

    /// The most recently picked reference Point index, or `None`. The successor of
    /// `Scene::active_point`.
    pub fn primary_point_index(&self) -> Option<usize> {
        self.targets.iter().rev().find_map(|target| match target {
            SelectionTarget::ReferencePoint(index) => Some(*index),
            SelectionTarget::Node(_) => None,
        })
    }

    /// Land a click's [`SelectionRequest`] — the shell's single door for a view-action
    /// selection change.
    pub fn apply_request(&mut self, request: SelectionRequest) {
        match request {
            SelectionRequest::Only(target) => self.select_only(target),
            SelectionRequest::Clear => self.clear(),
        }
    }

    /// Replace the whole selection with one target (a plain click).
    pub fn select_only(&mut self, target: SelectionTarget) {
        self.targets.clear();
        self.targets.push(target);
    }

    /// Toggle a target in / out of the set (a Shift-click — accumulate). Re-picking an
    /// already-picked target moves it to the end, so it becomes the primary.
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

    /// Replace every picked reference POINT with `index` (or drop them all for `None`),
    /// leaving other kinds untouched.
    pub fn set_primary_point_index(&mut self, index: Option<usize>) {
        self.targets
            .retain(|target| !matches!(target, SelectionTarget::ReferencePoint(_)));
        if let Some(index) = index {
            self.targets.push(SelectionTarget::ReferencePoint(index));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIRST: SelectionTarget = SelectionTarget::Node(NodeId(1));
    const SECOND: SelectionTarget = SelectionTarget::Node(NodeId(2));
    const POINT: SelectionTarget = SelectionTarget::ReferencePoint(0);

    /// The primary is the most recently picked target OF ITS KIND, so a mixed selection
    /// answers both questions at once — the reason ADR 0032 holds one set, not two.
    #[test]
    fn primaries_are_per_kind_and_newest_wins() {
        let mut selection = Selection::default();
        selection.toggle(FIRST);
        selection.toggle(POINT);
        selection.toggle(SECOND);
        assert_eq!(selection.primary_node_id(), Some(NodeId(2)));
        assert_eq!(selection.primary_point_index(), Some(0));
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
        assert_eq!(selection.primary_point_index(), Some(0));
        assert!(!selection.contains(FIRST));

        selection.set_primary_node(None);
        assert_eq!(selection.primary_node_id(), None);
        assert_eq!(selection.primary_point_index(), Some(0));
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
}
