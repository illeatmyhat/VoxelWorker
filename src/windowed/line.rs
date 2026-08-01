//! Session-only state for the connected Line command.

use document::scene::NodeId;
use document::sketch::{EntityId, SketchCurve, SketchPoint, SketchSolid, TangentArcRefusal};
use parametric::EvaluationContext;

#[derive(Debug, Clone, PartialEq)]
pub(super) enum LineEdit {
    SessionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LineChain {
    pub owner: NodeId,
    pub start: EntityId,
    pub end: EntityId,
    pub incoming: Option<SketchCurve>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinePress {
    None,
    Click,
    PendingArc,
    Arc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct LineGesture {
    chain: Option<LineChain>,
    press: LinePress,
}

impl Default for LineGesture {
    fn default() -> Self {
        Self {
            chain: None,
            press: LinePress::None,
        }
    }
}

impl LineGesture {
    pub const fn chain(self) -> Option<LineChain> {
        self.chain
    }

    pub fn start(&mut self, owner: NodeId, point: EntityId) {
        self.chain = Some(LineChain {
            owner,
            start: point,
            end: point,
            incoming: None,
        });
        self.press = LinePress::None;
    }

    pub fn advance(&mut self, end: EntityId, incoming: SketchCurve) -> bool {
        let Some(chain) = &mut self.chain else {
            return false;
        };
        if end == chain.start {
            self.reset();
            return true;
        }
        chain.end = end;
        chain.incoming = Some(incoming);
        self.press = LinePress::None;
        false
    }

    pub fn begin_press(&mut self, hit_live_end: bool) {
        self.press = if hit_live_end && self.chain.is_some_and(|chain| chain.incoming.is_some()) {
            LinePress::PendingArc
        } else {
            LinePress::Click
        };
    }

    pub fn update_drag(&mut self, down: (f64, f64), current: (f64, f64), threshold: f64) {
        let moved =
            (current.0 - down.0).abs() >= threshold || (current.1 - down.1).abs() >= threshold;
        if moved && self.press == LinePress::PendingArc {
            self.press = LinePress::Arc;
        }
    }

    pub const fn arc_is_latched(self) -> bool {
        matches!(self.press, LinePress::Arc)
    }

    pub const fn press_is_live(self) -> bool {
        !matches!(self.press, LinePress::None)
    }

    pub fn end_press(&mut self) {
        self.press = LinePress::None;
    }

    pub fn finish_chain(&mut self) -> bool {
        let had_chain = self.chain.take().is_some();
        self.press = LinePress::None;
        had_chain
    }

    pub fn accept_for_enter(&mut self, line_is_active: bool, constraint_is_armed: bool) -> bool {
        if !line_is_active || constraint_is_armed {
            self.reset();
            return false;
        }
        self.finish_chain()
    }

    pub fn reset(&mut self) -> bool {
        let was_live = self.chain.is_some() || self.press != LinePress::None;
        *self = Self::default();
        was_live
    }

    pub fn retain_if_live(
        &mut self,
        owner: NodeId,
        point_is_live: impl Fn(EntityId) -> bool,
        curve_is_live: impl Fn(SketchCurve) -> bool,
    ) {
        let valid = self.chain.is_none_or(|chain| {
            chain.owner == owner
                && point_is_live(chain.start)
                && point_is_live(chain.end)
                && chain.incoming.is_none_or(curve_is_live)
        });
        if !valid {
            self.reset();
        }
    }

    pub fn retain_for_context(
        &mut self,
        line_is_armed: bool,
        constraint_is_armed: bool,
        owner: Option<NodeId>,
    ) {
        if !line_is_armed
            || constraint_is_armed
            || self.chain.is_some_and(|chain| Some(chain.owner) != owner)
        {
            self.reset();
        }
    }

    pub fn cancel_for_escape(&mut self, line_is_active: bool, constraint_is_armed: bool) -> bool {
        let was_live = self.reset();
        line_is_active && !constraint_is_armed && was_live
    }

    fn placed_point(
        producer: &SketchSolid,
        at: SketchPoint,
        existing: Option<EntityId>,
    ) -> (SketchSolid, EntityId) {
        existing.map_or_else(
            || producer.with_point_placed(at),
            |id| (producer.clone(), id),
        )
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        producer: &SketchSolid,
        at: SketchPoint,
        existing: Option<EntityId>,
    ) -> LineEdit {
        let (with_point, clicked) = Self::placed_point(producer, at, existing);
        let Some(chain) = self.chain else {
            self.start(owner, clicked);
            return if with_point == *producer {
                LineEdit::SessionOnly
            } else {
                LineEdit::Document(with_point)
            };
        };
        if clicked == chain.end {
            self.finish_chain();
            return LineEdit::SessionOnly;
        }
        if clicked == chain.start {
            let edit = with_point
                .with_segment_between_traced(chain.end, clicked)
                .map_or(LineEdit::SessionOnly, |(next, incoming)| {
                    self.advance(clicked, incoming);
                    LineEdit::Document(next)
                });
            self.finish_chain();
            return edit;
        }
        let Some((next, incoming)) = with_point.with_segment_between_traced(chain.end, clicked)
        else {
            return LineEdit::SessionOnly;
        };
        self.advance(clicked, incoming);
        LineEdit::Document(next)
    }

    pub fn append_tangent_arc(
        &mut self,
        producer: &SketchSolid,
        at: SketchPoint,
        existing: Option<EntityId>,
        context: EvaluationContext,
    ) -> Result<SketchSolid, TangentArcRefusal> {
        let Some(chain) = self.chain else {
            return Err(TangentArcRefusal::UnknownIncoming);
        };
        let Some(incoming) = chain.incoming else {
            return Err(TangentArcRefusal::UnknownIncoming);
        };
        let (with_point, clicked) = Self::placed_point(producer, at, existing);
        let (next, arc) =
            with_point.with_tangent_arc_between(incoming, chain.end, clicked, context)?;
        self.advance(clicked, arc);
        Ok(next)
    }

    pub fn tangent_arc_candidate(
        self,
        producer: &SketchSolid,
        target: [f64; 2],
        context: EvaluationContext,
    ) -> Result<parametric::sketch::TangentArcCandidate, TangentArcRefusal> {
        let chain = self.chain.ok_or(TangentArcRefusal::UnknownIncoming)?;
        let incoming = chain.incoming.ok_or(TangentArcRefusal::UnknownIncoming)?;
        producer
            .sketch
            .tangent_arc_candidate(incoming, chain.end, target, context)
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use document::sketch::ConstraintKind;

    fn empty() -> SketchSolid {
        SketchSolid::extrude(
            document::sketch::Sketch::empty(document::sketch::PlaneAxis::Z),
            3,
        )
    }

    fn context() -> EvaluationContext {
        EvaluationContext::new(std::num::NonZeroU32::new(16).unwrap())
    }

    #[test]
    fn arc_latches_only_after_a_curve_and_only_from_the_live_end() {
        let mut gesture = LineGesture::default();
        gesture.start(NodeId(10), 1);
        gesture.begin_press(true);
        gesture.update_drag((0.0, 0.0), (5.0, 0.0), 5.0);
        assert!(!gesture.arc_is_latched());

        gesture.advance(2, SketchCurve::Segment(3));
        gesture.begin_press(false);
        gesture.update_drag((0.0, 0.0), (5.0, 0.0), 5.0);
        assert!(!gesture.arc_is_latched());
        gesture.begin_press(true);
        assert!(!gesture.arc_is_latched());
        gesture.update_drag((0.0, 0.0), (4.9, 4.9), 5.0);
        assert!(!gesture.arc_is_latched());
        gesture.update_drag((0.0, 0.0), (5.0, 0.0), 5.0);
        assert!(gesture.arc_is_latched());
    }

    #[test]
    fn closing_and_reset_drop_all_transient_state() {
        let mut gesture = LineGesture::default();
        gesture.start(NodeId(10), 1);
        gesture.advance(2, SketchCurve::Segment(3));
        assert!(gesture.advance(1, SketchCurve::Arc(4)));
        assert_eq!(gesture.chain(), None);
        assert!(!gesture.reset());
    }

    #[test]
    fn segment_then_arc_keeps_the_created_curve_as_the_next_incoming() {
        let mut gesture = LineGesture::default();
        gesture.start(NodeId(10), 1);
        gesture.advance(2, SketchCurve::Segment(3));
        gesture.advance(4, SketchCurve::Arc(5));
        assert_eq!(gesture.chain().unwrap().incoming, Some(SketchCurve::Arc(5)));
    }

    #[test]
    fn tool_constraint_and_owner_transitions_reset_the_whole_gesture() {
        for (line_armed, constraint_armed, owner) in [
            (false, false, Some(NodeId(10))),
            (true, true, Some(NodeId(10))),
            (true, false, Some(NodeId(11))),
            (true, false, None),
        ] {
            let mut gesture = LineGesture::default();
            gesture.start(NodeId(10), 1);
            gesture.advance(2, SketchCurve::Segment(3));
            gesture.begin_press(true);
            gesture.retain_for_context(line_armed, constraint_armed, owner);
            assert_eq!(gesture, LineGesture::default());
        }
    }

    #[test]
    fn accept_finishes_only_an_active_unoverridden_line_chain() {
        let mut gesture = LineGesture::default();
        gesture.begin_press(false);
        assert!(
            !gesture.accept_for_enter(true, false),
            "a press alone does not consume Enter"
        );
        gesture.start(NodeId(10), 1);
        gesture.advance(2, SketchCurve::Segment(3));
        gesture.begin_press(true);
        assert!(gesture.accept_for_enter(true, false));
        assert_eq!(gesture, LineGesture::default());

        gesture.start(NodeId(10), 1);
        assert!(!gesture.accept_for_enter(true, true));
        assert_eq!(gesture, LineGesture::default());
    }

    #[test]
    fn escape_consumes_the_chain_once_then_falls_through_to_tool_disarm() {
        let mut gesture = LineGesture::default();
        gesture.start(NodeId(10), 1);
        assert!(gesture.cancel_for_escape(true, false));
        assert!(!gesture.cancel_for_escape(true, false));

        gesture.start(NodeId(10), 1);
        assert!(!gesture.cancel_for_escape(true, true));
        assert_eq!(gesture, LineGesture::default());
    }

    #[test]
    fn escape_consumes_active_line_press_and_latch_once() {
        let mut press_only = LineGesture::default();
        press_only.begin_press(false);
        assert!(press_only.cancel_for_escape(true, false));
        assert!(!press_only.cancel_for_escape(true, false));

        let mut latched = LineGesture::default();
        latched.start(NodeId(10), 1);
        latched.advance(2, SketchCurve::Segment(3));
        latched.begin_press(true);
        latched.update_drag((0.0, 0.0), (8.0, 0.0), 8.0);
        assert!(latched.arc_is_latched());
        assert!(latched.cancel_for_escape(true, false));
        assert!(!latched.cancel_for_escape(true, false));
    }

    #[test]
    fn escape_clears_line_state_silently_when_line_is_overridden_or_inactive() {
        for (line_is_active, constraint_is_armed) in [(true, true), (false, false)] {
            let mut gesture = LineGesture::default();
            gesture.begin_press(false);
            assert!(!gesture.cancel_for_escape(line_is_active, constraint_is_armed));
            assert_eq!(gesture, LineGesture::default());
        }
    }

    #[test]
    fn clicks_place_one_point_then_continue_and_close_with_the_start_identity() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("first point edit")
        };
        let start = gesture.chain().unwrap().start;
        assert_eq!(one.sketch.points().len(), 1);

        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("first segment edit")
        };
        let LineEdit::Document(three) = gesture.click(owner, &two, SketchPoint::new(10, 10), None)
        else {
            panic!("second segment edit")
        };
        assert_eq!(three.sketch.segments().len(), 2);

        let start_at = three
            .sketch
            .points()
            .iter()
            .find(|point| point.id == start)
            .unwrap()
            .at;
        let LineEdit::Document(closed) = gesture.click(owner, &three, start_at, Some(start)) else {
            panic!("closure edit")
        };
        assert_eq!(closed.sketch.points().len(), 3, "the start id is reused");
        assert_eq!(closed.sketch.segments().len(), 3);
        assert_eq!(gesture.chain(), None);
    }

    #[test]
    fn clicking_the_live_end_finishes_open_without_an_edit() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let end = gesture.chain().unwrap().end;
        let at = one.sketch.points()[0].at;
        assert_eq!(
            gesture.click(owner, &one, at, Some(end)),
            LineEdit::SessionOnly
        );
        assert_eq!(gesture.chain(), None);
    }

    #[test]
    fn clicking_start_finishes_when_the_closing_segment_already_exists() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let start = gesture.chain().unwrap().start;
        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("segment")
        };
        let LineEdit::Document(three) = gesture.click(owner, &two, SketchPoint::new(10, 10), None)
        else {
            panic!("second segment")
        };
        let end = gesture.chain().unwrap().end;
        let preclosed = three.with_segment_between(end, start);
        let start_at = preclosed
            .sketch
            .points()
            .iter()
            .find(|point| point.id == start)
            .unwrap()
            .at;
        assert_eq!(
            gesture.click(owner, &preclosed, start_at, Some(start)),
            LineEdit::SessionOnly
        );
        assert_eq!(gesture.chain(), None);
        assert_eq!(preclosed.sketch.segments().len(), 3);
    }

    #[test]
    fn tangent_arc_closure_constrains_adjacent_curves_not_the_first_curve() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let start = gesture.chain().unwrap().start;
        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("segment")
        };
        let first_curve = gesture.chain().unwrap().incoming.unwrap();
        let three = gesture
            .append_tangent_arc(&two, SketchPoint::new(10, 10), None, context())
            .unwrap();
        let first_arc = gesture.chain().unwrap().incoming.unwrap();
        assert!(matches!(first_arc, SketchCurve::Arc(_)));
        assert_eq!(three.sketch.constraints().len(), 1);
        let start_at = three
            .sketch
            .points()
            .iter()
            .find(|point| point.id == start)
            .unwrap()
            .at;
        let closed = gesture
            .append_tangent_arc(&three, start_at, Some(start), context())
            .unwrap();
        assert_eq!(gesture.chain(), None);
        assert_eq!(closed.sketch.point_at(start_at), Some(start));
        assert_eq!(
            closed.sketch.constraints().len(),
            2,
            "one Tangent per explicit arc"
        );
        let tangent_pairs: Vec<(SketchCurve, SketchCurve)> = closed
            .sketch
            .constraints()
            .iter()
            .filter_map(|constraint| match constraint.kind {
                ConstraintKind::Tangent { first, second, .. } => Some((first, second)),
                _ => None,
            })
            .collect();
        let closing_arc = tangent_pairs[1].1;
        assert!(tangent_pairs.iter().any(|&(a, b)| {
            (a == first_curve && b == first_arc) || (a == first_arc && b == first_curve)
        }));
        assert!(tangent_pairs.iter().any(|&(a, b)| {
            (a == first_arc && b == closing_arc) || (a == closing_arc && b == first_arc)
        }));
        assert!(!tangent_pairs.iter().any(|&(a, b)| {
            (a == first_curve && b == closing_arc) || (a == closing_arc && b == first_curve)
        }));
    }

    #[test]
    fn ordinary_click_after_an_arc_returns_to_straight_continuation() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("segment")
        };
        let three = gesture
            .append_tangent_arc(&two, SketchPoint::new(10, 10), None, context())
            .unwrap();
        let constraints = three.sketch.constraints().len();
        let LineEdit::Document(four) = gesture.click(owner, &three, SketchPoint::new(20, 10), None)
        else {
            panic!("straight continuation")
        };
        assert_eq!(four.sketch.constraints().len(), constraints);
        assert!(matches!(
            gesture.chain().unwrap().incoming,
            Some(SketchCurve::Segment(_))
        ));
        assert!(gesture.chain().is_some());
    }

    #[test]
    fn refused_arc_keeps_the_chain_and_document_byte_exact() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("segment")
        };
        let before_chain = gesture.chain();
        let before = serde_json::to_string(&two).unwrap();
        assert!(gesture
            .append_tangent_arc(&two, SketchPoint::new(20, 0), None, context())
            .is_err());
        assert_eq!(gesture.chain(), before_chain);
        assert_eq!(serde_json::to_string(&two).unwrap(), before);
    }

    #[test]
    fn grabbed_off_grid_vertex_wins_over_the_snapped_cursor_target() {
        let (producer, grabbed) =
            empty().with_point_placed(SketchPoint::from_continuous(3.25, 4.75));
        let resolved = super::super::sketch_target::resolve_target(
            &producer,
            Some(grabbed),
            Some(SketchPoint::new(3, 5)),
        )
        .unwrap();
        assert_eq!(resolved.existing, Some(grabbed));
        assert_eq!(resolved.at.in_plane(), [3.25, 4.75]);
    }

    #[test]
    fn preview_candidate_sides_and_commit_use_the_same_geometry() {
        let owner = NodeId(10);
        let mut gesture = LineGesture::default();
        let LineEdit::Document(one) = gesture.click(owner, &empty(), SketchPoint::new(0, 0), None)
        else {
            panic!("point")
        };
        let LineEdit::Document(two) = gesture.click(owner, &one, SketchPoint::new(10, 0), None)
        else {
            panic!("segment")
        };
        let upper_minor = gesture
            .tangent_arc_candidate(&two, [15.0, 5.0], context())
            .unwrap();
        let lower_minor = gesture
            .tangent_arc_candidate(&two, [15.0, -5.0], context())
            .unwrap();
        let upper = gesture
            .tangent_arc_candidate(&two, [5.0, 5.0], context())
            .unwrap();
        let lower = gesture
            .tangent_arc_candidate(&two, [5.0, -5.0], context())
            .unwrap();
        assert!(
            upper_minor.sweep_radians > 0.0 && upper_minor.sweep_radians < std::f64::consts::PI
        );
        assert!(
            lower_minor.sweep_radians < 0.0 && lower_minor.sweep_radians > -std::f64::consts::PI
        );
        assert!(upper.sweep_radians > std::f64::consts::PI);
        assert!(lower.sweep_radians < -std::f64::consts::PI);

        let committed = gesture
            .append_tangent_arc(&two, SketchPoint::new(5, 5), None, context())
            .unwrap();
        let created = gesture.chain().unwrap().incoming.unwrap();
        let parametric::sketch::CurveGeometry::Circular(persisted) =
            committed.sketch.curve_geometry(created, context()).unwrap()
        else {
            panic!("persisted circular arc")
        };
        let SketchCurve::Arc(id) = created else {
            panic!("arc identity")
        };
        let arc = committed
            .sketch
            .arcs()
            .iter()
            .find(|arc| arc.id == id)
            .unwrap();
        assert!((persisted.center[0] - upper.center[0]).abs() < 1.0e-10);
        assert!((persisted.center[1] - upper.center[1]).abs() < 1.0e-10);
        assert!((persisted.radius - upper.radius).abs() < 1.0e-10);
        let sweep = arc
            .bulge
            .free_value()
            .or_else(|| arc.bulge.fixed_source())
            .unwrap()
            .to_degrees_f64()
            .to_radians();
        assert!((sweep - upper.sweep_radians).abs() < 1.0e-10);
    }
}
