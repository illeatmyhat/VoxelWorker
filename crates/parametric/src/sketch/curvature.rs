//! Curvature-continuity mathematics for a spline end meeting another curve.
//!
//! Like [`super::tangent`], this module knows no solver handles or document ids: callers adapt
//! their storage to [`CurveGeometry`] and a few plain positions, and get back residuals.
//!
//! # Why a spline end is described by four points
//!
//! A fit-point spline whose every point carries an authored tangent is a chain of cubic HERMITE
//! spans, each decided by its own two ends and their two tangents and by nothing else. So the
//! curve near a joint is not a property of the whole spline; it is a property of the one span the
//! joint belongs to, which four points name completely — the joint, the joint's arm, the next fit
//! point along, and that point's arm.
//!
//! That locality is what makes curvature an ordinary residual here rather than a differentiation
//! through a global interpolation. It holds only while every tangent is authored, which is an
//! invariant the document keeps.

#![allow(
    clippy::imprecise_flops,
    clippy::missing_const_for_fn,
    clippy::must_use_candidate,
    clippy::suboptimal_flops
)]

use super::curve::CurveGeometry;
use substrate::rational_bezier::RationalBezier;

/// Which end of its span a joint stands at.
///
/// A spline's first point starts the span that follows it; its last point finishes the span that
/// precedes it. The difference is which way the span's control points are ordered, and therefore
/// where along it the curvature is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpanEnd {
    Start,
    Finish,
}

/// The four positions naming the span a joint belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JointSpan {
    /// The fit point the relation is asserted at.
    pub joint: [f64; 2],
    /// The forward arm of the joint's tangent lever.
    pub joint_arm: [f64; 2],
    /// The next fit point along the spline.
    pub neighbor: [f64; 2],
    /// That point's forward arm.
    pub neighbor_arm: [f64; 2],
    /// Which end of the span the joint stands at.
    pub end: SpanEnd,
}

impl JointSpan {
    /// The tangent a lever authors: three times the arm's offset from the point it steers.
    ///
    /// The factor of three is the Hermite-to-Bézier conversion and not a taste: a cubic Bézier's
    /// first control point stands one third of the way along the tangent, so an arm placed at
    /// `point + tangent/3` is what makes the drawn lever the derivative it depicts.
    fn tangent(point: [f64; 2], arm: [f64; 2]) -> [f64; 2] {
        [(arm[0] - point[0]) * 3.0, (arm[1] - point[1]) * 3.0]
    }

    /// The span as a cubic, ordered along the spline rather than from the joint.
    ///
    /// Built from the same arithmetic [`super::spline::fit_point_spline`] uses, so the curve this
    /// reads is the curve the drawing draws. Reproducing the formula instead would be a second
    /// definition of the same shape, free to drift from the first.
    pub fn as_cubic(self) -> RationalBezier {
        let (tail, tail_arm, head, head_arm) = match self.end {
            SpanEnd::Start => (self.joint, self.joint_arm, self.neighbor, self.neighbor_arm),
            SpanEnd::Finish => (self.neighbor, self.neighbor_arm, self.joint, self.joint_arm),
        };
        let tail_tangent = Self::tangent(tail, tail_arm);
        let head_tangent = Self::tangent(head, head_arm);
        RationalBezier::cubic([
            tail,
            [
                tail[0] + tail_tangent[0] / 3.0,
                tail[1] + tail_tangent[1] / 3.0,
            ],
            [
                head[0] - head_tangent[0] / 3.0,
                head[1] - head_tangent[1] / 3.0,
            ],
            head,
        ])
    }

    /// Where along the cubic the joint stands.
    fn parameter(self) -> f64 {
        match self.end {
            SpanEnd::Start => 0.0,
            SpanEnd::Finish => 1.0,
        }
    }

    /// The curve's curvature arrow at the joint.
    pub fn curvature_arrow(self) -> [f64; 2] {
        self.as_cubic().curvature_vector_at(self.parameter())
    }

    /// The direction the spline runs at the joint, pointing away from the joint along the lever.
    pub fn direction(self) -> [f64; 2] {
        Self::tangent(self.joint, self.joint_arm)
    }

    /// The distance to the next fit point: the natural length to measure this span's curvature
    /// against when a residual has to be dimensionless.
    fn chord(self) -> f64 {
        (self.neighbor[0] - self.joint[0]).hypot(self.neighbor[1] - self.joint[1])
    }
}

/// The direction a curve runs at `joint`, unnormalized.
///
/// A circular curve's tangent is perpendicular to its radius, which is why a joint standing at the
/// center has no answer — the zero vector it returns leaves the residual reading zero, and the
/// document refuses that configuration before it can be asserted.
pub fn direction_at(geometry: CurveGeometry, joint: [f64; 2]) -> [f64; 2] {
    match geometry {
        CurveGeometry::Segment { from, to } => [to[0] - from[0], to[1] - from[1]],
        CurveGeometry::Circular(circle) => {
            let radius = [joint[0] - circle.center[0], joint[1] - circle.center[1]];
            [-radius[1], radius[0]]
        }
    }
}

/// A curve's curvature arrow at `joint`: toward its center of curvature, `1/radius` long.
///
/// A straight run curves nowhere, so its arrow is zero — which makes G2-to-a-line the ordinary
/// case of this function rather than a branch anyone has to remember.
///
/// `joint` is presumed to lie ON the curve; the caller owes that. For a circular curve the arrow's
/// direction is read from the joint and its length from the radius, so a joint standing off the
/// curve is answered with an arrow that describes no curve at all. Asserting curvature between
/// things that do not meet is not a question with an answer, and the document establishes the
/// coincidence before it offers the relation.
pub fn curvature_arrow_at(geometry: CurveGeometry, joint: [f64; 2]) -> [f64; 2] {
    match geometry {
        CurveGeometry::Segment { .. } => [0.0; 2],
        CurveGeometry::Circular(circle) => {
            let toward = [circle.center[0] - joint[0], circle.center[1] - joint[1]];
            let scale = 1.0 / (circle.radius * circle.radius);
            [toward[0] * scale, toward[1] * scale]
        }
    }
}

/// One row: the sine of the angle between the spline's direction at the joint and the curve's.
///
/// Normalized, for the reason ADR 0035 gives for Parallel — a raw cross product would speak in
/// length-squared and let a long neighbor shout down a short one in the same trust region.
/// Only the joint and its arm are read: a tangent direction is a property of the lever standing at
/// the joint, and does not depend on where the span's far end is.
pub fn direction_residual(joint: [f64; 2], joint_arm: [f64; 2], geometry: CurveGeometry) -> f64 {
    let mine = JointSpan::tangent(joint, joint_arm);
    let theirs = direction_at(geometry, joint);
    let (my_length, their_length) = (mine[0].hypot(mine[1]), theirs[0].hypot(theirs[1]));
    if my_length == 0.0 || their_length == 0.0 {
        return 0.0;
    }
    (mine[0] * theirs[1] - mine[1] * theirs[0]) / (my_length * their_length)
}

/// One row: how far the spline's curvature at the joint stands from the curve's.
///
/// # Arrows rather than signed scalars
///
/// A signed curvature only means something once you have said which way the curve is being walked,
/// so comparing two curves that MEET becomes a question about their two traversal directions before
/// it is a question about their shapes — and every way the pair can be oriented is a case to get
/// right. The curvature ARROW has no such freedom: reversing a curve flips both the sign and the
/// normal it points along, leaving the arrow alone. Two curves are G2 exactly when their arrows
/// agree, whichever way either was drawn.
///
/// The difference is read along the joint's own normal and scaled by the span's chord, which makes
/// the row dimensionless. Curvature is `1/length`, so an unscaled row would be loud on a small
/// drawing and inaudible on a large one.
pub fn curvature_residual(span: JointSpan, geometry: CurveGeometry) -> f64 {
    let mine = span.curvature_arrow();
    if !mine[0].is_finite() || !mine[1].is_finite() {
        return 0.0;
    }
    let theirs = curvature_arrow_at(geometry, span.joint);
    let direction = span.direction();
    let speed = direction[0].hypot(direction[1]);
    if speed == 0.0 {
        return 0.0;
    }
    let normal = [-direction[1] / speed, direction[0] / speed];
    ((mine[0] - theirs[0]) * normal[0] + (mine[1] - theirs[1]) * normal[1]) * span.chord()
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use crate::sketch::curve::CircularCurve;

    fn circle(center: [f64; 2], radius: f64) -> CurveGeometry {
        CurveGeometry::Circular(CircularCurve {
            center,
            radius,
            arc: None,
        })
    }

    /// A span built to match a circle's curvature reads zero, and stretching its lever does not.
    #[test]
    fn a_span_matching_a_circle_reads_zero_and_a_stretched_one_does_not() {
        let span = JointSpan {
            joint: [5.0, 0.0],
            // A tangent of [0, 3] means an arm one third along it.
            joint_arm: [5.0, 1.0],
            neighbor: [4.0, 4.0],
            // Solved so the span's curvature at the joint is exactly 1/5.
            neighbor_arm: [4.0 - 2.1 / 3.0, 4.0 + 1.0 / 3.0],
            end: SpanEnd::Start,
        };
        let against = circle([0.0, 0.0], 5.0);
        assert!(
            curvature_residual(span, against).abs() < 1.0e-9,
            "{}",
            curvature_residual(span, against)
        );
        for scale in [0.5, 0.9, 1.1, 2.0_f64] {
            let stretched = JointSpan {
                joint_arm: [5.0, scale],
                ..span
            };
            assert!(
                curvature_residual(stretched, against).abs() > 1.0e-6,
                "a lever scaled by {scale} should not read as curvature-continuous"
            );
        }
    }

    /// The joint's direction row is blind to which way the neighbor was drawn.
    #[test]
    fn the_direction_row_ignores_how_the_neighbor_was_drawn() {
        let span = JointSpan {
            joint: [0.0, 0.0],
            joint_arm: [1.0, 0.0],
            neighbor: [3.0, 2.0],
            neighbor_arm: [3.5, 2.5],
            end: SpanEnd::Start,
        };
        let forward = CurveGeometry::Segment {
            from: [-2.0, 0.0],
            to: [4.0, 0.0],
        };
        let backward = CurveGeometry::Segment {
            from: [4.0, 0.0],
            to: [-2.0, 0.0],
        };
        assert!(direction_residual(span.joint, span.joint_arm, forward).abs() < 1.0e-12);
        assert!(direction_residual(span.joint, span.joint_arm, backward).abs() < 1.0e-12);
        let across = CurveGeometry::Segment {
            from: [0.0, -1.0],
            to: [0.0, 1.0],
        };
        assert!(
            (direction_residual(span.joint, span.joint_arm, across).abs() - 1.0).abs() < 1.0e-12
        );
    }

    /// One span, read from each of its two ends: `SpanEnd` selects which end the joint stands at,
    /// and each answer matches the cubic's own reading there.
    ///
    /// This is what lets ONE relation serve both ends of a spline. The four points are supplied in
    /// spline order either way — an arm always points forward along the curve — so the only thing
    /// that changes is which end is being asked about.
    #[test]
    fn each_end_of_a_span_reads_its_own_curvature() {
        let (tail, tail_arm) = ([5.0, 0.0], [5.0, 1.0]);
        let (head, head_arm) = ([4.0, 4.0], [4.0 - 2.1 / 3.0, 4.0 + 1.0 / 3.0]);
        let at_start = JointSpan {
            joint: tail,
            joint_arm: tail_arm,
            neighbor: head,
            neighbor_arm: head_arm,
            end: SpanEnd::Start,
        };
        let at_finish = JointSpan {
            joint: head,
            joint_arm: head_arm,
            neighbor: tail,
            neighbor_arm: tail_arm,
            end: SpanEnd::Finish,
        };
        // Both descriptions name the same curve.
        assert_eq!(
            format!("{:?}", at_start.as_cubic().control),
            format!("{:?}", at_finish.as_cubic().control)
        );
        let cubic = at_start.as_cubic();
        for (span, parameter) in [(at_start, 0.0), (at_finish, 1.0_f64)] {
            let reference = cubic.curvature_vector_at(parameter);
            let arrow = span.curvature_arrow();
            for axis in 0..2 {
                assert!(
                    (arrow[axis] - reference[axis]).abs() < 1.0e-9,
                    "at {parameter}: {arrow:?} vs {reference:?}"
                );
            }
        }
    }

    /// A straight neighbor asks for zero curvature, which is the natural end condition and not a
    /// special case in the arithmetic.
    #[test]
    fn a_straight_neighbor_asks_for_no_curvature() {
        let span = JointSpan {
            joint: [0.0, 0.0],
            joint_arm: [1.0, 0.0],
            neighbor: [3.0, 2.0],
            // Solved so the span leaves the joint with zero curvature.
            neighbor_arm: [3.0 + 1.0 / 3.0, 2.0 + 2.0],
            end: SpanEnd::Start,
        };
        let line = CurveGeometry::Segment {
            from: [-1.0, 0.0],
            to: [1.0, 0.0],
        };
        assert!(
            curvature_residual(span, line).abs() < 1.0e-9,
            "{}",
            curvature_residual(span, line)
        );
    }
}
