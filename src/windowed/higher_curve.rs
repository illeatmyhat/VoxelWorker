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
        let Some(pending) = self.pending.as_mut() else {
            return HigherCurveEdit::InteractionOnly;
        };
        if pending
            .points
            .last()
            .is_some_and(|point| point.coincides(&target.at))
        {
            return HigherCurveEdit::InteractionOnly;
        }
        if kind == HigherCurveKind::FitPointSpline
            && pending.points.len() >= 3
            && pending.points[0].coincides(&target.at)
        {
            return self.finish_with(producer, true);
        }
        pending.points.push(target.at);
        // The conic takes a FOURTH pick the ellipse does not: its three points leave rho free, and
        // rho is the whole difference between an elliptic, parabolic and hyperbolic curve through
        // the same three points. The fourth pick is the apex that names it.
        let arity = match kind {
            HigherCurveKind::Ellipse => 3,
            HigherCurveKind::Conic => 4,
            HigherCurveKind::FitPointSpline | HigherCurveKind::ControlPointSpline => usize::MAX,
        };
        if pending.points.len() == arity {
            return self.finish_with(producer, false);
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
                let rho = conic_rho(points[0], points[1], points[2], points[3])?;
                next.sketch
                    .add_conic(points[0], points[1], points[2], rho)
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
            // The rho step previews the curve the fourth pick is choosing between; before that
            // pick exists there is no rho, so the three points stand on their own.
            HigherCurveKind::Conic if continuous.len() == 4 => {
                conic_rho(points[0], points[1], points[2], points[3])
                    .and_then(|rho| {
                        parametric::sketch::conic_candidate(
                            continuous[0],
                            continuous[1],
                            continuous[2],
                            rho,
                        )
                        .ok()
                    })
                    .map(|candidate| vec![candidate.curve])
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

/// The rho a conic's fourth pick names, in the gesture's own pick order.
///
/// One definition for the preview and the commit, so the curve the author is aiming at is the
/// curve the click authors. `None` when the apex pick does not lie beyond the vertex, where no rho
/// answers — the tool then shows nothing rather than snapping to an invented sharpness.
fn conic_rho(
    from: SketchPoint,
    to: SketchPoint,
    vertex: SketchPoint,
    apex: SketchPoint,
) -> Option<f64> {
    parametric::sketch::conic_rho_from_apex(
        from.in_plane(),
        to.in_plane(),
        vertex.in_plane(),
        apex.in_plane(),
    )
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
#[allow(clippy::panic)]
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
        for (kind, apex) in [
            (HigherCurveKind::Ellipse, None),
            // Beyond the vertex on the midpoint→vertex ray, so a rho in (0, 1) answers.
            (HigherCurveKind::Conic, Some(target(2, 9))),
        ] {
            let mut gesture = HigherCurveGesture::default();
            gesture.click(owner, kind, &source, Some(target(0, 0)));
            gesture.click(owner, kind, &source, Some(target(5, 0)));
            let third = gesture.click(owner, kind, &source, Some(target(2, 3)));
            let made = match apex {
                None => third,
                Some(apex) => {
                    assert!(
                        matches!(third, HigherCurveEdit::InteractionOnly),
                        "a conic's third pick leaves rho unchosen"
                    );
                    gesture.click(owner, kind, &source, Some(apex))
                }
            };
            assert!(matches!(made, HigherCurveEdit::Document(_)));
            assert!(source.sketch.points().is_empty());
        }
    }

    /// The apex pick names rho, so two different apexes on the same three points are two
    /// different curves — the freedom the fourth pick exists to spend.
    #[test]
    fn the_apex_pick_chooses_the_conics_sharpness() {
        let owner = NodeId(3);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let rho_for = |apex: ResolvedSketchTarget| {
            let mut gesture = HigherCurveGesture::default();
            for point in [target(0, 0), target(8, 0), target(4, 2)] {
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
            }
            let HigherCurveEdit::Document(made) =
                gesture.click(owner, HigherCurveKind::Conic, &source, Some(apex))
            else {
                panic!("the apex pick commits")
            };
            made.sketch.conics()[0].rho.value()
        };
        let near = rho_for(target(4, 3));
        let far = rho_for(target(4, 20));
        assert!(
            near > far,
            "pulling the apex away sharpens: {near} vs {far}"
        );
        assert!((0.0..1.0).contains(&near) && (0.0..1.0).contains(&far));
    }

    /// An apex that does not lie beyond the vertex names no rho, and the tool refuses rather than
    /// inventing a sharpness the author did not point at.
    #[test]
    fn an_apex_short_of_the_vertex_commits_nothing() {
        let owner = NodeId(4);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(8, 0), target(4, 6)] {
            gesture.click(owner, HigherCurveKind::Conic, &source, Some(point));
        }
        let made = gesture.click(owner, HigherCurveKind::Conic, &source, Some(target(4, 2)));
        assert!(matches!(made, HigherCurveEdit::InteractionOnly));
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
