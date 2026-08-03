//! Interaction-transient state shared by Fusion's five slot grammars.

use document::scene::NodeId;
use document::sketch::{SketchPoint, SketchSolid, SlotPlacement};
use parametric::EvaluationContext;

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
    /// Which way the cursor was going about the center while the center-arc spine was being aimed.
    ///
    /// Only [`SlotKind::CenterPointArc`] ever fills this in; the other four grammars have no arc
    /// whose direction is in question. It stops advancing once the direction pick lands, so the
    /// width step cannot reverse a spine the author already settled.
    winding: Option<substrate::winding::TurnLatch>,
}

impl PendingSlot {
    fn starting(owner: NodeId, kind: SlotKind) -> Self {
        Self {
            owner,
            kind,
            picks: Vec::with_capacity(kind.pick_count() - 1),
            winding: None,
        }
    }

    /// The spine picks of a center-arc slot, once they are all in.
    fn center_arc_spine(&self) -> Option<(SketchPoint, SketchPoint)> {
        match (self.kind, self.picks.as_slice()) {
            (SlotKind::CenterPointArc, [center, start, ..]) => Some((*center, *start)),
            _ => None,
        }
    }
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

    /// Fold this frame's cursor into the latch that decides which way a center-arc spine runs.
    ///
    /// Called once per frame before the preview is asked for, and only while the direction pick is
    /// still the one being aimed — after that the cursor is driving the width.
    pub fn track_cursor(&mut self, owner: NodeId, kind: SlotKind, cursor: SketchPoint) {
        let Some(pending) = self
            .pending
            .as_mut()
            .filter(|pending| pending.owner == owner && pending.kind == kind)
        else {
            return;
        };
        if pending.picks.len() != 2 {
            return;
        }
        let Some((center, start)) = pending.center_arc_spine() else {
            return;
        };
        super::arc_winding::track(&mut pending.winding, center, start, cursor);
    }

    fn turn(&self, owner: NodeId, kind: SlotKind) -> parametric::sketch::ArcTurn {
        super::arc_winding::turn(
            self.pending
                .as_ref()
                .filter(|pending| pending.owner == owner && pending.kind == kind)
                .and_then(|pending| pending.winding),
        )
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
            SlotKind::CenterPointArc => parametric::sketch::center_arc_slot_spine(
                *first,
                *second,
                *third,
                self.turn(owner, kind),
            )
            .ok(),
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
        placement_from(
            kind,
            producer,
            &pending.picks,
            super::arc_winding::turn(pending.winding),
            cursor.at,
        )
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: SlotKind,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
        context: EvaluationContext,
    ) -> SlotEdit {
        let Some(target) = target else {
            return SlotEdit::InteractionOnly;
        };
        let pending = self
            .pending
            .get_or_insert_with(|| PendingSlot::starting(owner, kind));
        if pending.owner != owner || pending.kind != kind {
            *pending = PendingSlot::starting(owner, kind);
        }
        // The click's own position is the last winding reading, so a preview and the pick that
        // replaces it cannot disagree about the direction even if no frame rendered in between.
        if pending.picks.len() == 2 {
            if let Some((center, start)) = pending.center_arc_spine() {
                super::arc_winding::track(&mut pending.winding, center, start, target.at);
            }
        }
        if pending.picks.len() + 1 < kind.pick_count() {
            pending.picks.push(target.at);
            return SlotEdit::InteractionOnly;
        }
        let (picks, winding) = self
            .pending
            .take()
            .map(|pending| (pending.picks, pending.winding))
            .unwrap_or_default();
        commit_from(
            kind,
            producer,
            &picks,
            super::arc_winding::turn(winding),
            target.at,
            context,
        )
        .map_or(SlotEdit::InteractionOnly, SlotEdit::Document)
    }
}

fn placement_from(
    kind: SlotKind,
    producer: &SketchSolid,
    picks: &[SketchPoint],
    turn: parametric::sketch::ArcTurn,
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
            .center_arc_slot_placement(*center, *start, *end_direction, turn, cursor)
            .ok(),
        _ => None,
    }
}

fn commit_from(
    kind: SlotKind,
    producer: &SketchSolid,
    picks: &[SketchPoint],
    turn: parametric::sketch::ArcTurn,
    cursor: SketchPoint,
    context: EvaluationContext,
) -> Option<SketchSolid> {
    match (kind, picks) {
        (SlotKind::CenterToCenter, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::CenterToCenter,
                *first,
                *second,
                cursor,
                context,
            )
            .ok(),
        (SlotKind::Overall, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::Overall,
                *first,
                *second,
                cursor,
                context,
            )
            .ok(),
        (SlotKind::CenterPoint, [first, second]) => producer
            .with_linear_slot(
                parametric::sketch::LinearSlotKind::CenterPoint,
                *first,
                *second,
                cursor,
                context,
            )
            .ok(),
        (SlotKind::ThreePointArc, [start, end, through]) => producer
            .with_three_point_arc_slot(*start, *end, *through, cursor, context)
            .ok(),
        (SlotKind::CenterPointArc, [center, start, end_direction]) => producer
            .with_center_arc_slot(*center, *start, *end_direction, turn, cursor, context)
            .ok(),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};
    use std::num::NonZeroU32;

    fn target(at: SketchPoint) -> ResolvedSketchTarget {
        ResolvedSketchTarget { at, existing: None }
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(NonZeroU32::new(16).unwrap())
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
                    context(),
                );
            }
            let SlotEdit::Document(made) = result else {
                panic!("final click completes")
            };
            // Four boundary curves, plus the construction line down an Overall Slot's middle.
            let spine_line = usize::from(kind == SlotKind::Overall);
            assert_eq!(
                made.sketch.segments().len() + made.sketch.arcs().len(),
                4 + spine_line
            );
            assert!(gesture.pending.is_none());
        }
    }
}
