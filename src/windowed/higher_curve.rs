//! Interaction state for ellipse, conic, and the two repeated-point spline grammars.
//!
//! The gesture owns only transient picks. A completed command calls the durable document
//! constructor once, so cancellation never leaves control points or half an aggregate behind.

use document::scene::NodeId;
use document::sketch::{SketchPoint, SketchSolid};
use substrate::rational_bezier::RationalBezier;

use document::sketch::SketchTarget;

/// How many conic picks are banked once the shoulder step begins: two anchors and the control
/// point. The step after them cannot fail, so it is the one step exempt from the gates that
/// decline a pick.
const CONIC_SHOULDER_PICK: usize = 3;

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
        target: Option<SketchTarget>,
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
                points: vec![target.at()],
            });
            return HigherCurveEdit::InteractionOnly;
        }
        // A pick the grammar cannot use is not taken at all. Accepting a conic control point that
        // sits on its own chord would bank a gesture that can never commit, and the author would
        // have to guess that Escape is the way out. This is the same question the preview's
        // refusal mark asks, so what the cursor warned about is what the click declines.
        if self.refuses_cursor(owner, kind, target.at()) {
            return HigherCurveEdit::InteractionOnly;
        }
        let Some(pending) = self.pending.as_mut() else {
            return HigherCurveEdit::InteractionOnly;
        };
        // Repeating the previous pick names no new geometry — except on the conic's shoulder step,
        // where the track RUNS to the control point and clicking it means "pull as hard as it
        // goes". The clamp already keeps that off the degenerate end, and this step is the one the
        // author cannot fail.
        let shoulder_step =
            kind == HigherCurveKind::Conic && pending.points.len() == CONIC_SHOULDER_PICK;
        if !shoulder_step
            && pending
                .points
                .last()
                .is_some_and(|point| point.coincides(&target.at()))
        {
            return HigherCurveEdit::InteractionOnly;
        }
        if kind == HigherCurveKind::FitPointSpline
            && pending.points.len() >= 3
            && pending.points[0].coincides(&target.at())
        {
            return self.finish_with(producer, true);
        }
        pending.points.push(target.at());
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
                // The third pick IS what the document stores: the control point the author placed,
                // which stays grabbable afterwards. The fourth pick slid the shoulder along the
                // track to say how hard it pulls, and all it contributes is rho.
                let resolved = conic_from_picks(points[0], points[1], points[2], Some(points[3]))?;
                next.sketch
                    .add_conic(points[0], points[1], points[2], resolved.rho)
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

    /// Whether clicking here would author nothing — a conic control point on its own chord.
    ///
    /// The pick polyline cannot say this on its own: a refused step draws exactly like a gesture
    /// still in progress, so without a mark the author reads a dead cursor as a live one. Only the
    /// conic's control-point step can refuse. The shoulder step after it cannot, because the gizmo
    /// is captive to its track and every position on the track is a curve.
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

    /// The conic shoulder gizmo: the track it slides on, then its position on that track.
    ///
    /// Live only during the last conic step, when the control point is placed and the cursor is
    /// choosing how hard it pulls. Profile space; the caller projects.
    pub fn conic_shoulder_gizmo(
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
        Some((resolved.track, resolved.shoulder))
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
        // On the shoulder step the cursor always counts, even resting on the control point: drop it
        // there and the preview falls back to the three-pick parabolic default, so hovering the far
        // end of the track flickers the curve away from what the click would make.
        let shoulder_step = kind == HigherCurveKind::Conic && points.len() == CONIC_SHOULDER_PICK;
        if shoulder_step || points.last().is_none_or(|point| !point.coincides(&cursor)) {
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
            // moving it reads at the parabolic default, so the author watches the curve bend under
            // the cursor rather than watching a polyline stand in for it. The last step swaps the
            // cursor from control point to shoulder and the same curve keeps answering.
            HigherCurveKind::Conic if points.len() == 3 => {
                conic_from_picks(points[0], points[1], points[2], None)
                    .map(|resolved| vec![resolved.curve])
            }
            HigherCurveKind::Conic if points.len() == 4 => {
                conic_from_picks(points[0], points[1], points[2], Some(points[3]))
                    .map(|resolved| vec![resolved.curve])
            }
            HigherCurveKind::FitPointSpline => {
                // A spline still being drawn has no handles yet, so every tangent is natural.
                parametric::sketch::fit_point_spline(
                    &continuous,
                    &vec![None; continuous.len()],
                    false,
                )
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

/// A conic resolved from the gesture's picks: two anchors, the control point the end tangents meet
/// at, and where the shoulder sits on the track between them.
struct ConicPicks {
    /// Where the shoulder gizmo slides — chord midpoint to control point.
    track: [[f64; 2]; 2],
    /// The gizmo's position on that track, which is also the curve's point at t = 0.5.
    shoulder: [f64; 2],
    rho: f64,
    curve: substrate::rational_bezier::RationalBezier,
}

/// Resolve the conic a run of picks names, in the gesture's own pick order.
///
/// One definition behind the preview, the drawn gizmo and the commit, so the curve the author is
/// shaping is the curve the click authors.
///
/// `shoulder` is `None` while the control point is still being placed. The curve then reads at the
/// parabolic default, which is what lets the author watch an actual conic bend under the control
/// point instead of waiting for a step that has not happened yet.
///
/// `None` only when the control point falls on the chord midpoint, where there is no track and no
/// conic to shape.
fn conic_from_picks(
    from: SketchPoint,
    to: SketchPoint,
    apex: SketchPoint,
    shoulder: Option<SketchPoint>,
) -> Option<ConicPicks> {
    let (from, to, apex) = (from.in_plane(), to.in_plane(), apex.in_plane());
    let track = parametric::sketch::conic_shoulder_track(from, to, apex)?;
    let rho = shoulder.map_or(Some(parametric::sketch::CONIC_PARABOLIC_RHO), |shoulder| {
        parametric::sketch::conic_rho_from_shoulder(from, to, apex, shoulder.in_plane())
    })?;
    let candidate = parametric::sketch::conic_candidate(from, to, apex, rho).ok()?;
    Some(ConicPicks {
        track,
        shoulder: candidate.vertex,
        rho,
        curve: candidate.curve,
    })
}

fn flatten_joined(curves: Vec<RationalBezier>) -> Vec<[f64; 2]> {
    let mut flattened = Vec::new();
    for (index, curve) in curves.into_iter().enumerate() {
        let mut piece = curve.flatten(document::sketch::ARC_SAGITTA_TOLERANCE);
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

    fn target(x: i64, y: i64) -> SketchTarget {
        SketchTarget::fresh(SketchPoint::new(x, y))
    }

    /// An ellipse is settled by three picks; a conic is not, because rho is still free.
    #[test]
    fn fixed_arity_curves_commit_atomically_on_their_last_pick() {
        let owner = NodeId(1);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for (kind, shoulder) in [
            (HigherCurveKind::Ellipse, None),
            // On the track between the chord midpoint (2.5, 0) and the control point (2, 3).
            (HigherCurveKind::Conic, Some(target(2, 2))),
        ] {
            let mut gesture = HigherCurveGesture::default();
            gesture.click(owner, kind, &source, Some(target(0, 0)));
            gesture.click(owner, kind, &source, Some(target(5, 0)));
            let third = gesture.click(owner, kind, &source, Some(target(2, 3)));
            let made = match shoulder {
                None => third,
                Some(shoulder) => {
                    assert!(
                        matches!(third, HigherCurveEdit::InteractionOnly),
                        "a conic's control point leaves its pull unchosen"
                    );
                    gesture.click(owner, kind, &source, Some(shoulder))
                }
            };
            assert!(matches!(made, HigherCurveEdit::Document(_)));
            assert!(source.sketch.points().is_empty());
        }
    }

    /// The shoulder gizmo names how hard the control point pulls: sliding it toward the control
    /// point sharpens the curve, sliding it back toward the chord flattens it. That is the whole
    /// freedom the fourth pick exists to spend.
    #[test]
    fn the_shoulder_gizmo_chooses_how_hard_the_control_point_pulls() {
        let owner = NodeId(3);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        // Anchors (0, 0) and (8, 0) put the chord midpoint at (4, 0); the control point at (4, 8)
        // makes the track a clean eight voxels of straight up.
        let rho_for = |shoulder: SketchTarget| {
            let mut gesture = HigherCurveGesture::default();
            for point in [target(0, 0), target(8, 0), target(4, 8)] {
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
            }
            let HigherCurveEdit::Document(made) =
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(shoulder))
            else {
                panic!("the shoulder pick commits")
            };
            made.sketch.conics()[0].rho.value()
        };
        let near_the_control_point = rho_for(target(4, 6));
        let near_the_chord = rho_for(target(4, 2));
        assert!(
            near_the_control_point > near_the_chord,
            "toward the control point sharpens: {near_the_control_point} vs {near_the_chord}"
        );
        assert!(
            (0.0..1.0).contains(&near_the_control_point) && (0.0..1.0).contains(&near_the_chord)
        );
    }

    /// The gizmo is captive: dragged past either end of its track it stops rather than refusing,
    /// so the last step of a conic has no way to fail on the author.
    #[test]
    fn a_shoulder_dragged_off_the_end_of_its_track_still_commits() {
        let owner = NodeId(5);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for overshoot in [target(4, -40), target(4, 40)] {
            let mut gesture = HigherCurveGesture::default();
            for point in [target(0, 0), target(8, 0), target(4, 8)] {
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
            }
            let made = gesture.click(owner, HigherCurveKind::Conic, &source, Some(overshoot));
            assert!(matches!(made, HigherCurveEdit::Document(_)));
        }
    }

    /// The far end of the shoulder track IS the control point, so clicking it means "pull as hard
    /// as it goes" — not a repeated pick to swallow. Resting there previews the same curve the
    /// click makes, instead of flickering back to the parabolic default.
    #[test]
    fn a_shoulder_clicked_on_the_control_point_commits_the_hardest_pull() {
        let owner = NodeId(6);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(8, 0), target(4, 8)] {
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
        }
        let resting = gesture.preview(owner, HigherCurveKind::Conic, SketchPoint::new(4, 8));
        let halfway = gesture.preview(owner, HigherCurveKind::Conic, SketchPoint::new(4, 4));
        assert_ne!(resting, halfway, "the track's far end is its own reading");

        let HigherCurveEdit::Document(made) =
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 8)))
        else {
            panic!("the shoulder pick commits even on the control point")
        };
        assert!(made.sketch.conics()[0].rho.value() > 0.99);
    }

    /// A control point on the chord midpoint shapes nothing: no track for the shoulder to slide
    /// on, and no conic to build. The pick is declined outright rather than banked into a gesture
    /// that could never commit, and the anchors behind it survive to be finished properly.
    #[test]
    fn a_control_point_on_the_chord_is_declined_and_keeps_the_anchors() {
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
        let made = gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 4)));
        assert!(matches!(made, HigherCurveEdit::Document(_)));
    }

    /// The conic shows a real curve from the moment its control point starts moving, not a
    /// polyline through the picks — the curve IS the affordance for placing the control point.
    #[test]
    fn a_conic_previews_a_curve_while_its_control_point_is_still_moving() {
        let owner = NodeId(6);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(8, 0)] {
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
        }
        let bending = gesture.preview(owner, HigherCurveKind::Conic, SketchPoint::new(4, 8));
        assert!(
            bending.len() > 3,
            "a flattened conic, not the three picks: {bending:?}"
        );
        // The gizmo only exists once the control point is placed, and then it rides its track.
        assert!(gesture
            .conic_shoulder_gizmo(owner, HigherCurveKind::Conic, SketchPoint::new(4, 4))
            .is_none());
        gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 8)));
        let (track, shoulder) = gesture
            .conic_shoulder_gizmo(owner, HigherCurveKind::Conic, SketchPoint::new(4, 6))
            .expect("the shoulder gizmo is live once the control point is down");
        assert_eq!(track, [[4.0, 0.0], [4.0, 8.0]]);
        assert!((shoulder[1] - 6.0).abs() < 1.0e-9, "{shoulder:?}");
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
