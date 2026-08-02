//! Interaction-transient state shared by Fusion's five slot grammars.

use document::scene::NodeId;
use document::sketch::{SketchPoint, SketchSolid, SlotPlacement};

use super::sketch_target::ResolvedSketchTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SlotKind {
    CenterToCenter,
    Overall,
    CenterPoint,
    ThreePointArc,
    CenterPointArc,
}

impl SlotKind {
    const fn pick_count(self) -> usize {
        match self {
            Self::CenterToCenter | Self::Overall | Self::CenterPoint => 3,
            Self::ThreePointArc | Self::CenterPointArc => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PendingSlot {
    owner: NodeId,
    kind: SlotKind,
    picks: Vec<SketchPoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum SlotEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct SlotGesture {
    pending: Option<PendingSlot>,
}

impl SlotGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .flat_map(|pending| pending.picks.iter().copied())
            .collect()
    }

    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<SlotKind>,
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
        active_kind: Option<SlotKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub fn blocks_enter(&self, active_kind: Option<SlotKind>, constraint_is_armed: bool) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    pub fn guide(&self, owner: NodeId, kind: SlotKind) -> Option<Vec<SketchPoint>> {
        self.pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)
            .map(|pending| pending.picks.clone())
    }

    /// The centerline arc of an arc slot whose spine is settled but whose WIDTH is not — the
    /// intermediate the two arc grammars spend most of their clicks in.
    ///
    /// Without this the width step previewed a straight run through the picks, which looks nothing
    /// like the arc slot it is about to become. The cursor stands in for the last spine pick until
    /// that pick is taken, so the arc is live from the moment it is determined.
    pub fn spine(
        &self,
        owner: NodeId,
        kind: SlotKind,
        cursor: SketchPoint,
    ) -> Option<parametric::sketch::ArcSlotSpine> {
        let pending = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        // Once the spine's own picks are all in, the cursor is driving the width instead.
        let settled: Vec<[f64; 2]> = pending
            .picks
            .iter()
            .chain(std::iter::once(&cursor))
            .take(3)
            .map(SketchPoint::in_plane)
            .collect();
        let [first, second, third] = settled.as_slice() else {
            return None;
        };
        match kind {
            SlotKind::ThreePointArc => {
                parametric::sketch::three_point_arc_slot_spine(*first, *second, *third).ok()
            }
            SlotKind::CenterPointArc => {
                parametric::sketch::center_arc_slot_spine(*first, *second, *third).ok()
            }
            SlotKind::CenterToCenter | SlotKind::Overall | SlotKind::CenterPoint => None,
        }
    }

    pub fn placement(
        &self,
        owner: NodeId,
        kind: SlotKind,
        producer: &SketchSolid,
        cursor: ResolvedSketchTarget,
    ) -> Option<SlotPlacement> {
        let pending = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        placement_from(kind, producer, &pending.picks, cursor.at)
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: SlotKind,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
    ) -> SlotEdit {
        let Some(target) = target else {
            return SlotEdit::InteractionOnly;
        };
        let pending = self.pending.get_or_insert_with(|| PendingSlot {
            owner,
            kind,
            picks: Vec::with_capacity(kind.pick_count() - 1),
        });
        if pending.owner != owner || pending.kind != kind {
            *pending = PendingSlot {
                owner,
                kind,
                picks: Vec::with_capacity(kind.pick_count() - 1),
            };
        }
        if pending.picks.len() + 1 < kind.pick_count() {
            pending.picks.push(target.at);
            return SlotEdit::InteractionOnly;
        }
        let picks = self
            .pending
            .take()
            .map(|pending| pending.picks)
            .unwrap_or_default();
        commit_from(kind, producer, &picks, target.at)
            .map_or(SlotEdit::InteractionOnly, SlotEdit::Document)
    }
}

fn placement_from(
    kind: SlotKind,
    producer: &SketchSolid,
    picks: &[SketchPoint],
    cursor: SketchPoint,
) -> Option<SlotPlacement> {
    match (kind, picks) {
        (SlotKind::CenterToCenter, [first, second]) => producer
            .linear_slot_placement(
                parametric::sketch::LinearSlotKind::CenterToCenter,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::Overall, [first, second]) => producer
            .linear_slot_placement(
                parametric::sketch::LinearSlotKind::Overall,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::CenterPoint, [first, second]) => producer
            .linear_slot_placement(
                parametric::sketch::LinearSlotKind::CenterPoint,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::ThreePointArc, [start, end, through]) => producer
            .three_point_arc_slot_placement(*start, *end, *through, cursor)
            .ok(),
        (SlotKind::CenterPointArc, [center, start, end_direction]) => producer
            .center_arc_slot_placement(*center, *start, *end_direction, cursor)
            .ok(),
        _ => None,
    }
}

fn commit_from(
    kind: SlotKind,
    producer: &SketchSolid,
    picks: &[SketchPoint],
    cursor: SketchPoint,
) -> Option<SketchSolid> {
    match (kind, picks) {
        (SlotKind::CenterToCenter, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::CenterToCenter,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::Overall, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::Overall,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::CenterPoint, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::CenterPoint,
                *first,
                *second,
                cursor,
            )
            .ok(),
        (SlotKind::ThreePointArc, [start, end, through]) => producer
            .with_three_point_arc_slot(*start, *end, *through, cursor)
            .ok(),
        (SlotKind::CenterPointArc, [center, start, end_direction]) => producer
            .with_center_arc_slot(*center, *start, *end_direction, cursor)
            .ok(),
        _ => None,
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
    fn all_five_grammars_commit_only_on_their_final_pick() {
        let owner = NodeId(1);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for (kind, points) in [
            (SlotKind::CenterToCenter, vec![[0, 0], [6, 0], [0, 1]]),
            (SlotKind::Overall, vec![[0, 0], [8, 0], [0, 1]]),
            (SlotKind::CenterPoint, vec![[0, 0], [3, 0], [0, 1]]),
            (
                SlotKind::ThreePointArc,
                vec![[2, 0], [0, 2], [2, 2], [3, 0]],
            ),
            (
                SlotKind::CenterPointArc,
                vec![[0, 0], [2, 0], [0, 2], [3, 0]],
            ),
        ] {
            let mut gesture = SlotGesture::default();
            let mut result = SlotEdit::InteractionOnly;
            for point in points {
                result = gesture.click(
                    owner,
                    kind,
                    &source,
                    Some(target(SketchPoint::new(point[0], point[1]))),
                );
            }
            let SlotEdit::Document(made) = result else {
                panic!("final click completes")
            };
            assert_eq!(made.sketch.segments().len() + made.sketch.arcs().len(), 4);
            assert!(gesture.pending.is_none());
        }
    }
}
