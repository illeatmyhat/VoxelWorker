//! Interaction state for ellipse, conic, and the two repeated-point spline grammars.
//!
//! The gesture owns only transient picks. A completed command calls the durable document
//! constructor once, so cancellation never leaves control points or half an aggregate behind.

use document::scene::NodeId;
use document::sketch::{SketchPoint, SketchSolid};
use substrate::rational_bezier::RationalBezier;

use super::sketch_target::ResolvedSketchTarget;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HigherCurveKind {
    Ellipse,
    Conic,
    FitPointSpline,
    ControlPointSpline,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum HigherCurveEdit {
    InteractionOnly,
    Document(SketchSolid),
}

#[derive(Debug, Clone, PartialEq)]
struct PendingHigherCurve {
    owner: NodeId,
    kind: HigherCurveKind,
    points: Vec<SketchPoint>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct HigherCurveGesture {
    pending: Option<PendingHigherCurve>,
}

impl HigherCurveGesture {
    /// The points this gesture has already taken, for THIS sketch — the multi-step affordance.
    ///
    /// A tool that has consumed clicks must show what it consumed, or its intermediate steps read
    /// as the tool doing nothing. Empty when idle or when the pending gesture belongs elsewhere.
    pub fn placed_points(&self, owner: NodeId) -> Vec<SketchPoint> {
        self.pending
            .iter()
            .filter(|pending| pending.owner == owner)
            .flat_map(|pending| pending.points.iter().copied())
            .collect()
    }

    pub fn reset(&mut self) -> bool {
        self.pending.take().is_some()
    }

    pub fn retain_for_context(
        &mut self,
        active_kind: Option<HigherCurveKind>,
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
        active_kind: Option<HigherCurveKind>,
        constraint_is_armed: bool,
    ) -> bool {
        let was_live = self.reset();
        active_kind.is_some() && !constraint_is_armed && was_live
    }

    pub fn blocks_enter(
        &self,
        active_kind: Option<HigherCurveKind>,
        constraint_is_armed: bool,
    ) -> bool {
        active_kind.is_some() && !constraint_is_armed && self.pending.is_some()
    }

    pub fn click(
        &mut self,
        owner: NodeId,
        kind: HigherCurveKind,
        producer: &SketchSolid,
        target: Option<ResolvedSketchTarget>,
    ) -> HigherCurveEdit {
        let Some(target) = target else {
            return HigherCurveEdit::InteractionOnly;
        };
        if self
            .pending
            .as_ref()
            .is_none_or(|pending| pending.owner != owner || pending.kind != kind)
        {
            self.pending = Some(PendingHigherCurve {
                owner,
                kind,
                points: vec![target.at],
            });
            return HigherCurveEdit::InteractionOnly;
        }
        // A pick the grammar cannot use is not taken at all. Accepting a conic control point that
        // sits on its own chord would bank a gesture that can never commit, and the author would
        // have to guess that Escape is the way out. This is the same question the preview's
        // refusal mark asks, so what the cursor warned about is what the click declines.
        if self.refuses_cursor(owner, kind, target.at) {
            return HigherCurveEdit::InteractionOnly;
        }
        let Some(pending) = self.pending.as_mut() else {
            return HigherCurveEdit::InteractionOnly;
        };
        // Clicking the same place twice is a no-op almost everywhere — except on the conic's last
        // step, where the cursor is dragging a gizmo and leaving it exactly where it was placed is
        // a real answer: the parabola the previous pick was already previewing.
        let repeats_the_last_pick = pending
            .points
            .last()
            .is_some_and(|point| point.coincides(&target.at));
        let drags_a_gizmo = kind == HigherCurveKind::Conic && pending.points.len() == 3;
        if repeats_the_last_pick && !drags_a_gizmo {
            return HigherCurveEdit::InteractionOnly;
        }
        if kind == HigherCurveKind::FitPointSpline
            && pending.points.len() >= 3
            && pending.points[0].coincides(&target.at)
        {
            return self.finish_with(producer, true);
        }
        pending.points.push(target.at);
        // The conic takes a FOURTH pick the ellipse does not. Two anchors and a control point fix
        // a whole FAMILY of curves, not one: how hard the control point pulls is still free, and
        // that freedom is the difference between an elliptic, parabolic and hyperbolic curve
        // through the same three picks. The fourth pick spends it.
        let arity = match kind {
            HigherCurveKind::Ellipse => 3,
            HigherCurveKind::Conic => 4,
            HigherCurveKind::FitPointSpline | HigherCurveKind::ControlPointSpline => usize::MAX,
        };
        if pending.points.len() == arity {
            let mut restore = pending.clone();
            let edit = self.finish_with(producer, false);
            if matches!(edit, HigherCurveEdit::InteractionOnly) {
                // The completing pick named no curve. Consuming the gesture here would charge the
                // author every earlier pick for one bad cursor position, so it comes back minus
                // the pick that answered nothing, ready for another try at the last step.
                restore.points.pop();
                self.pending = Some(restore);
            }
            return edit;
        }
        HigherCurveEdit::InteractionOnly
    }

    /// Finish an open repeated-point spline on Enter. Fixed-arity tools ignore Enter.
    pub fn finish(&mut self, producer: &SketchSolid) -> HigherCurveEdit {
        self.finish_with(producer, false)
    }

    fn finish_with(&mut self, producer: &SketchSolid, closed: bool) -> HigherCurveEdit {
        let Some(pending) = self.pending.take() else {
            return HigherCurveEdit::InteractionOnly;
        };
        let mut next = producer.clone();
        let made = match pending.kind {
            HigherCurveKind::Ellipse => pending.points.get(..3).and_then(|points| {
                next.sketch
                    .add_ellipse(points[0], points[1], points[2])
                    .ok()
            }),
            HigherCurveKind::Conic => pending.points.get(..4).and_then(|points| {
                // The document stores the conic by the point it passes THROUGH, which the third
                // pick pinned. The control point is the gizmo, never a stored vertex; all the
                // fourth pick contributed is rho.
                let resolved = conic_from_picks(points[0], points[1], points[2], Some(points[3]))?;
                let shoulder =
                    SketchPoint::try_from_continuous(resolved.shoulder[0], resolved.shoulder[1])
                        .ok()?;
                next.sketch
                    .add_conic(points[0], points[1], shoulder, resolved.rho)
                    .ok()
            }),
            HigherCurveKind::FitPointSpline => next
                .sketch
                .add_fit_point_spline(&pending.points, closed)
                .ok(),
            HigherCurveKind::ControlPointSpline => {
                next.sketch.add_control_point_spline(&pending.points).ok()
            }
        };
        made.map_or(HigherCurveEdit::InteractionOnly, |_| {
            HigherCurveEdit::Document(next)
        })
    }

    /// Whether clicking here would author nothing — a conic control point on its own chord
    /// midpoint, which aims at nothing and pins nothing.
    ///
    /// The pick polyline cannot say this on its own: a refused step draws exactly like a gesture
    /// still in progress, so without a mark the author reads a dead cursor as a live one. Only the
    /// step that places the control point can refuse. Dragging it afterwards cannot, because the
    /// gizmo is captive to its ray and every position on the ray is a curve.
    pub fn refuses_cursor(
        &self,
        owner: NodeId,
        kind: HigherCurveKind,
        cursor: SketchPoint,
    ) -> bool {
        let Some(pending) = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)
        else {
            return false;
        };
        match kind {
            HigherCurveKind::Conic => {
                pending.points.len() == 2
                    && pending.points.get(..2).is_some_and(|anchors| {
                        conic_from_picks(anchors[0], anchors[1], cursor, None).is_none()
                    })
            }
            HigherCurveKind::Ellipse
            | HigherCurveKind::FitPointSpline
            | HigherCurveKind::ControlPointSpline => false,
        }
    }

    /// The conic control-point gizmo: the track it rides, then where it sits on that track.
    ///
    /// Live only during the last conic step, when the curve is pinned and the cursor is dragging
    /// the control point in or out along its ray — which is how hard it pulls. Profile space; the
    /// caller projects.
    pub fn conic_control_gizmo(
        &self,
        owner: NodeId,
        kind: HigherCurveKind,
        cursor: SketchPoint,
    ) -> Option<([[f64; 2]; 2], [f64; 2])> {
        if kind != HigherCurveKind::Conic {
            return None;
        }
        let pending = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)?;
        let picks = pending
            .points
            .get(..3)
            .filter(|_| pending.points.len() == 3)?;
        let resolved = conic_from_picks(picks[0], picks[1], picks[2], Some(cursor))?;
        Some((resolved.track, resolved.control))
    }

    /// Profile-space preview through the current cursor. Invalid partial candidates fall back to
    /// their pick polyline, keeping every stage visible without fabricating durable geometry.
    pub fn preview(
        &self,
        owner: NodeId,
        kind: HigherCurveKind,
        cursor: SketchPoint,
    ) -> Vec<[f64; 2]> {
        let Some(pending) = self
            .pending
            .as_ref()
            .filter(|pending| pending.owner == owner && pending.kind == kind)
        else {
            return Vec::new();
        };
        let mut points = pending.points.clone();
        if points.last().is_none_or(|point| !point.coincides(&cursor)) {
            points.push(cursor);
        }
        let continuous: Vec<_> = points.iter().map(SketchPoint::in_plane).collect();
        let curves: Option<Vec<RationalBezier>> = match kind {
            HigherCurveKind::Ellipse if continuous.len() == 3 => {
                parametric::sketch::ellipse_candidate(continuous[0], continuous[1], continuous[2])
                    .ok()
                    .map(|candidate| candidate.quarters.to_vec())
            }
            // A real conic from the moment both anchors are down: while the control point is still
            // moving the curve reads at the parabolic default and visibly bends toward the cursor,
            // rather than a polyline standing in for it. The last step keeps the cursor on that
            // same control point, now captive to its ray, and the same curve keeps answering.
            HigherCurveKind::Conic if points.len() == 3 => {
                conic_from_picks(points[0], points[1], points[2], None)
                    .map(|resolved| vec![resolved.curve])
            }
            HigherCurveKind::Conic if points.len() == 4 => {
                conic_from_picks(points[0], points[1], points[2], Some(points[3]))
                    .map(|resolved| vec![resolved.curve])
            }
            HigherCurveKind::FitPointSpline => {
                parametric::sketch::fit_point_spline(&continuous, false)
                    .ok()
                    .map(|candidate| candidate.pieces)
            }
            HigherCurveKind::ControlPointSpline => {
                parametric::sketch::control_point_spline(&continuous)
                    .ok()
                    .map(|candidate| candidate.pieces)
            }
            HigherCurveKind::Ellipse | HigherCurveKind::Conic => None,
        };
        curves.map_or(continuous, flatten_joined)
    }
}

/// A conic resolved from the gesture's picks: two anchors, the control point that aims it and pins
/// it, and where that control point has since been dragged to.
struct ConicPicks {
    /// Where the control-point gizmo rides — from the pinned on-curve point outward.
    track: [[f64; 2]; 2],
    /// The gizmo itself: the control point, where the two end tangents meet.
    control: [f64; 2],
    /// The point the curve is pinned through, fixed when the control point was first placed.
    shoulder: [f64; 2],
    rho: f64,
    curve: substrate::rational_bezier::RationalBezier,
}

/// Resolve the conic a run of picks names, in the gesture's own pick order.
///
/// One definition behind the preview, the drawn gizmo and the commit, so the curve the author is
/// shaping is the curve the click authors.
///
/// The third pick both aims the curve and pins the point it passes through. `dragged` is the
/// control point's new position after that — `None` while the third pick is itself still moving,
/// where the curve reads as the parabola through the same aim. Because the pin is taken at the
/// parabolic reading, the curve does not jump when the drag begins and the handle starts out
/// exactly under the cursor that just placed it.
///
/// `None` only when the control point falls on the chord midpoint, where there is no ray to ride
/// and no conic to shape.
fn conic_from_picks(
    from: SketchPoint,
    to: SketchPoint,
    aim: SketchPoint,
    dragged: Option<SketchPoint>,
) -> Option<ConicPicks> {
    let (from, to, aim) = (from.in_plane(), to.in_plane(), aim.in_plane());
    let shoulder = parametric::sketch::conic_parabolic_shoulder(from, to, aim)?;
    let rho = dragged.map_or(Some(parametric::sketch::CONIC_PARABOLIC_RHO), |dragged| {
        parametric::sketch::conic_rho_from_control(from, to, shoulder, dragged.in_plane())
    })?;
    let control = parametric::sketch::conic_control_from_rho(from, to, shoulder, rho)?;
    let candidate = parametric::sketch::conic_candidate(from, to, shoulder, rho).ok()?;
    Some(ConicPicks {
        track: [shoulder, control],
        control,
        shoulder,
        rho,
        curve: candidate.curve,
    })
}

fn flatten_joined(curves: Vec<RationalBezier>) -> Vec<[f64; 2]> {
    let mut flattened = Vec::new();
    for (index, curve) in curves.into_iter().enumerate() {
        let mut piece = curve.flatten(document::sketch::ARC_SAGITTA_TOLERANCE_VOXELS);
        if index > 0 && !piece.is_empty() {
            piece.remove(0);
        }
        flattened.extend(piece);
    }
    flattened
}

#[cfg(test)]
#[allow(clippy::panic, clippy::expect_used)]
mod tests {
    use super::*;
    use document::sketch::{PlaneAxis, Sketch};

    fn target(x: i64, y: i64) -> ResolvedSketchTarget {
        ResolvedSketchTarget {
            at: SketchPoint::new(x, y),
            existing: None,
        }
    }

    /// An ellipse is settled by three picks; a conic is not, because rho is still free.
    #[test]
    fn fixed_arity_curves_commit_atomically_on_their_last_pick() {
        let owner = NodeId(1);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for (kind, dragged) in [
            (HigherCurveKind::Ellipse, None),
            // Further out along the ray the control point at (2, 3) already aimed.
            (HigherCurveKind::Conic, Some(target(1, 9))),
        ] {
            let mut gesture = HigherCurveGesture::default();
            gesture.click(owner, kind, &source, Some(target(0, 0)));
            gesture.click(owner, kind, &source, Some(target(5, 0)));
            let third = gesture.click(owner, kind, &source, Some(target(2, 3)));
            let made = match dragged {
                None => third,
                Some(dragged) => {
                    assert!(
                        matches!(third, HigherCurveEdit::InteractionOnly),
                        "placing a conic's control point leaves its pull unchosen"
                    );
                    gesture.click(owner, kind, &source, Some(dragged))
                }
            };
            assert!(matches!(made, HigherCurveEdit::Document(_)));
            assert!(source.sketch.points().is_empty());
        }
    }

    /// Dragging the control point is what chooses how hard it pulls: brought in close behind the
    /// curve it sharpens toward a hyperbola, pushed far away it flattens toward an ellipse. That
    /// is the whole freedom the fourth pick exists to spend.
    #[test]
    fn dragging_the_control_point_chooses_how_hard_it_pulls() {
        let owner = NodeId(3);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        // Anchors (0, 0) and (8, 0) put the chord midpoint at (4, 0). A control point at (4, 8)
        // aims straight up and pins the curve through (4, 4), halfway out.
        let rho_for = |dragged: ResolvedSketchTarget| {
            let mut gesture = HigherCurveGesture::default();
            for point in [target(0, 0), target(8, 0), target(4, 8)] {
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
            }
            let HigherCurveEdit::Document(made) =
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(dragged))
            else {
                panic!("the drag commits")
            };
            made.sketch.conics()[0].rho.value()
        };
        // rho = |midpoint→pin| / |midpoint→control|, so 4/5 against 4/20.
        let close_in = rho_for(target(4, 5));
        let far_out = rho_for(target(4, 20));
        assert!(
            (close_in - 0.8).abs() < 1.0e-9 && (far_out - 0.2).abs() < 1.0e-9,
            "{close_in} vs {far_out}"
        );
        // Left where it was placed, the curve is exactly the parabola the third pick previewed.
        assert!((rho_for(target(4, 8)) - 0.5).abs() < 1.0e-9);
    }

    /// The gizmo is captive: dragged past either end of its ray — inside the pinned point, or
    /// behind the chord entirely — it stops rather than refusing, so the last step cannot fail.
    #[test]
    fn a_control_point_dragged_off_the_end_of_its_ray_still_commits() {
        let owner = NodeId(5);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for overshoot in [target(4, 2), target(4, -40)] {
            let mut gesture = HigherCurveGesture::default();
            for point in [target(0, 0), target(8, 0), target(4, 8)] {
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
            }
            let made = gesture.click(owner, HigherCurveKind::Conic, &source, Some(overshoot));
            let HigherCurveEdit::Document(made) = made else {
                panic!("a clamped control point still commits")
            };
            assert!((0.0..1.0).contains(&made.sketch.conics()[0].rho.value()));
        }
    }

    /// A control point on the chord midpoint aims at nothing and pins nothing. The pick is
    /// declined outright rather than banked into a gesture that could never commit, and the
    /// anchors behind it survive.
    #[test]
    fn a_control_point_on_the_chord_midpoint_is_declined_and_keeps_the_anchors() {
        let owner = NodeId(4);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(8, 0)] {
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
        }
        let declined = gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 0)));
        assert!(matches!(declined, HigherCurveEdit::InteractionOnly));
        assert_eq!(gesture.placed_points(owner).len(), 2);
        gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 8)));
        let made = gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 5)));
        assert!(matches!(made, HigherCurveEdit::Document(_)));
    }

    /// The conic bends toward the cursor while the control point is still being placed, and the
    /// drag gizmo then starts exactly where that pick landed — no jump between the two steps.
    #[test]
    fn the_control_gizmo_starts_where_the_third_pick_landed() {
        let owner = NodeId(6);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(8, 0)] {
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
        }
        let bending = gesture.preview(owner, HigherCurveKind::Conic, SketchPoint::new(4, 8));
        assert!(
            bending.len() > 3,
            "a flattened conic, not the picks: {bending:?}"
        );
        assert!(gesture
            .conic_control_gizmo(owner, HigherCurveKind::Conic, SketchPoint::new(4, 8))
            .is_none());
        gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 8)));
        let (track, control) = gesture
            .conic_control_gizmo(owner, HigherCurveKind::Conic, SketchPoint::new(4, 8))
            .expect("the control gizmo is live once the control point is down");
        assert_eq!(track, [[4.0, 4.0], [4.0, 8.0]]);
        assert!((control[1] - 8.0).abs() < 1.0e-9, "{control:?}");
    }

    #[test]
    fn fit_spline_closes_by_clicking_its_first_point() {
        let owner = NodeId(2);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(5, 0), target(2, 4)] {
            gesture.click(owner, HigherCurveKind::FitPointSpline, &source, Some(point));
        }
        let made = gesture.click(
            owner,
            HigherCurveKind::FitPointSpline,
            &source,
            Some(target(0, 0)),
        );
        let HigherCurveEdit::Document(made) = made else {
            panic!("closing pick commits")
        };
        assert!(made.sketch.splines()[0].closed);
    }
}
