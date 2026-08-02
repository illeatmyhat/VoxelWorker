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
        conic_rho: f64,
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
            return self.finish_with(producer, true, conic_rho);
        }
        pending.points.push(target.at);
        if matches!(kind, HigherCurveKind::Ellipse | HigherCurveKind::Conic)
            && pending.points.len() == 3
        {
            return self.finish_with(producer, false, conic_rho);
        }
        HigherCurveEdit::InteractionOnly
    }

    /// Finish an open repeated-point spline on Enter. Fixed-arity tools ignore Enter.
    pub fn finish(&mut self, producer: &SketchSolid, conic_rho: f64) -> HigherCurveEdit {
        self.finish_with(producer, false, conic_rho)
    }

    fn finish_with(
        &mut self,
        producer: &SketchSolid,
        closed: bool,
        conic_rho: f64,
    ) -> HigherCurveEdit {
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
            HigherCurveKind::Conic => pending.points.get(..3).and_then(|points| {
                next.sketch
                    .add_conic(points[0], points[1], points[2], conic_rho)
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
        conic_rho: f64,
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
            HigherCurveKind::Conic if continuous.len() == 3 => parametric::sketch::conic_candidate(
                continuous[0],
                continuous[1],
                continuous[2],
                conic_rho,
            )
            .ok()
            .map(|candidate| vec![candidate.curve]),
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

    #[test]
    fn fixed_arity_curves_commit_atomically_on_the_third_pick() {
        let owner = NodeId(1);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        for kind in [HigherCurveKind::Ellipse, HigherCurveKind::Conic] {
            let mut gesture = HigherCurveGesture::default();
            gesture.click(owner, kind, &source, Some(target(0, 0)), 0.5);
            gesture.click(owner, kind, &source, Some(target(5, 0)), 0.5);
            let made = gesture.click(owner, kind, &source, Some(target(0, 3)), 0.5);
            assert!(matches!(made, HigherCurveEdit::Document(_)));
            assert!(source.sketch.points().is_empty());
        }
    }

    #[test]
    fn fit_spline_closes_by_clicking_its_first_point() {
        let owner = NodeId(2);
        let source = SketchSolid::extrude(Sketch::empty(PlaneAxis::Z), 3);
        let mut gesture = HigherCurveGesture::default();
        for point in [target(0, 0), target(5, 0), target(2, 4)] {
            gesture.click(
                owner,
                HigherCurveKind::FitPointSpline,
                &source,
                Some(point),
                0.5,
            );
        }
        let made = gesture.click(
            owner,
            HigherCurveKind::FitPointSpline,
            &source,
            Some(target(0, 0)),
            0.5,
        );
        let HigherCurveEdit::Document(made) = made else {
            panic!("closing pick commits")
        };
        assert!(made.sketch.splines()[0].closed);
    }
}
