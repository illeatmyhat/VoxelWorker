//! Curve–curve intersection in the plane: where a segment or a circular arc meets another one.
//!
//! This is the primitive a **geometric arrangement** is built from. Deriving regions from a graph
//! of drawn entities only finds a face where the author put a shared point; deriving them from the
//! arrangement finds one wherever two curves genuinely cross, which is what makes two overlapping
//! circles three regions rather than two. Getting there means being able to ask, of any two curves,
//! exactly where they meet and *how far along each* — the parameter, not just the point, because a
//! crossing is only useful if both curves can be cut at it.
//!
//! ## Why `f64` and not the measurement width
//!
//! The rest of [`geom2d`](crate::geom2d) splits by role: `f64` for the predicates that must be
//! exact, `f32` for the measurements a shader mirrors. Intersection is a predicate's job — a
//! crossing either exists or it does not, and a mistake there changes the *topology* of the
//! result, not its precision. So this half is `f64` throughout, and it carries its own curve type
//! rather than [`RegionEdge`](crate::geom2d::RegionEdge) for that reason.
//!
//! ## The two hard cases
//!
//! Almost all of the difficulty is in the degenerate pair — two curves that are not merely
//! touching but *coincident along a stretch*: collinear overlapping segments, and two arcs of the
//! same circle. A point answer is wrong there; the honest answer is the span they share, reported
//! as its two ends, which is exactly what an arrangement needs in order to cut both curves at the
//! boundary of the shared piece. [`CurveCrossing::overlapping`] marks those so a caller that wants
//! to treat a shared edge differently from a transverse crossing can.
//!
//! Segment–segment is the standard parametric cross-product solve; segment–circle the quadratic
//! substitution; circle–circle the radical-line construction. Each is then clipped to its curve's
//! own parameter range, which is where arcs differ from the full primitives those solves assume.

use std::f64::consts::TAU;

/// How far apart two positions may be and still be one crossing, in the curve's own units
/// (voxels, for a sketch).
///
/// It is an ill-conditioning guard, not a resolution: two curves meeting at a shallow angle have a
/// crossing whose position is genuinely uncertain by about this much, and pretending otherwise
/// produces two crossings where there is one. Nothing downstream inherits it — an arrangement cuts
/// at the crossing it is handed and never asks how precise it was.
pub const CROSSING_EPSILON: f64 = 1.0e-9;

/// The angular slack on "is this bearing within the arc's sweep", in radians. A crossing that
/// lands exactly on an endpoint must count, and floating point will not put it there exactly.
const ANGULAR_EPSILON: f64 = 1.0e-9;

/// One planar curve: a straight span, or a circular arc — including a whole circle, which is the
/// arc that sweeps a full turn.
///
/// The arc form is center-anchored rather than endpoint-anchored, so a closed curve is an ordinary
/// value here rather than a degenerate one. Its endpoints are derived
/// ([`start`](Self::start) / [`end`](Self::end)) and coincide exactly when the sweep is a full
/// turn, which is what "closed" means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlanarCurve {
    /// A straight span.
    Segment {
        /// The tail.
        start: [f64; 2],
        /// The head.
        end: [f64; 2],
    },
    /// A circular arc traveling `sweep_radians` about `center` from the bearing `start_radians` —
    /// counter-clockwise when the sweep is positive, clockwise when it is negative.
    Arc {
        /// The circle's center.
        center: [f64; 2],
        /// The circle's radius.
        radius: f64,
        /// The bearing the arc starts at.
        start_radians: f64,
        /// The signed angle travelled.
        sweep_radians: f64,
    },
}

impl PlanarCurve {
    /// A whole circle: the arc that sweeps a full turn counter-clockwise from bearing zero.
    pub fn circle(center: [f64; 2], radius: f64) -> Self {
        PlanarCurve::Arc {
            center,
            radius,
            start_radians: 0.0,
            sweep_radians: TAU,
        }
    }

    /// The point at `parameter`, which runs `0` at the tail to `1` at the head. Outside that range
    /// the curve is extended — the segment along its line, the arc around its circle — which is
    /// what makes an out-of-range solve reportable rather than silently clamped.
    pub fn point_at(&self, parameter: f64) -> [f64; 2] {
        match *self {
            PlanarCurve::Segment { start, end } => [
                start[0] + (end[0] - start[0]) * parameter,
                start[1] + (end[1] - start[1]) * parameter,
            ],
            PlanarCurve::Arc {
                center,
                radius,
                start_radians,
                sweep_radians,
            } => {
                let bearing = start_radians + sweep_radians * parameter;
                [
                    center[0] + radius * bearing.cos(),
                    center[1] + radius * bearing.sin(),
                ]
            }
        }
    }

    /// The curve's tail.
    pub fn start(&self) -> [f64; 2] {
        self.point_at(0.0)
    }

    /// The curve's head.
    pub fn end(&self) -> [f64; 2] {
        self.point_at(1.0)
    }

    /// Whether the curve closes on itself — a whole circle.
    pub fn is_closed(&self) -> bool {
        match *self {
            PlanarCurve::Segment { .. } => false,
            PlanarCurve::Arc { sweep_radians, .. } => sweep_radians.abs() >= TAU - ANGULAR_EPSILON,
        }
    }

    /// How long the curve is, in the units its coordinates are in.
    pub fn length(&self) -> f64 {
        match *self {
            PlanarCurve::Segment { start, end } => length([end[0] - start[0], end[1] - start[1]]),
            PlanarCurve::Arc {
                radius,
                sweep_radians,
                ..
            } => radius * sweep_radians.abs(),
        }
    }

    /// The stretch of this curve between two parameters, as a curve in its own right.
    ///
    /// An arc keeps its circle and narrows its sweep — it does not become a chord, and it does not
    /// get re-solved from the new endpoints. That is the whole point: cutting a curve produces
    /// pieces of the SAME curve, so nothing is approximated by being split.
    pub fn sub_curve(&self, from: f64, to: f64) -> PlanarCurve {
        match *self {
            PlanarCurve::Segment { .. } => PlanarCurve::Segment {
                start: self.point_at(from),
                end: self.point_at(to),
            },
            PlanarCurve::Arc {
                center,
                radius,
                start_radians,
                sweep_radians,
            } => PlanarCurve::Arc {
                center,
                radius,
                start_radians: start_radians + sweep_radians * from,
                sweep_radians: sweep_radians * (to - from),
            },
        }
    }

    /// This curve cut at each of `parameters`, in order along it.
    ///
    /// Cuts outside `(0, 1)`, cuts too close together to separate, and cuts at the ends are all
    /// dropped — a zero-length piece is not a piece, and an arrangement that grew one would derive
    /// a face with a degenerate edge in its boundary.
    ///
    /// A CLOSED curve with no cuts comes back whole, as one closed piece. That case cannot be
    /// expressed as a chain of pieces between vertices, because it has no vertex; it is a loop
    /// already, and its caller treats it as one.
    pub fn split_at(&self, parameters: &[f64]) -> Vec<PlanarCurve> {
        let curve_length = self.length();
        let slack = if curve_length > CROSSING_EPSILON {
            CROSSING_EPSILON / curve_length
        } else {
            1.0
        };
        // On an OPEN curve the ends are already vertices, so a cut there is not a cut. On a CLOSED
        // one there are no ends: parameter zero is an ordinary place on the curve, and dropping a
        // cut that lands on the seam would leave a circle uncut by a line that plainly crosses it.
        let mut cuts: Vec<f64> = if self.is_closed() {
            parameters
                .iter()
                .map(|parameter| {
                    let wrapped = parameter.rem_euclid(1.0);
                    if wrapped >= 1.0 - slack {
                        0.0
                    } else {
                        wrapped
                    }
                })
                .collect()
        } else {
            parameters
                .iter()
                .copied()
                .filter(|parameter| *parameter > slack && *parameter < 1.0 - slack)
                .collect()
        };
        cuts.sort_by(f64::total_cmp);
        cuts.dedup_by(|later, earlier| (*later - *earlier).abs() <= slack);
        if cuts.is_empty() {
            return vec![*self];
        }
        if self.is_closed() {
            // A closed curve's seam is an artefact of how it was written down, not a place on it.
            // So the pieces run between consecutive cuts and the last one WRAPS through the seam
            // back to the first — otherwise the seam would become a spurious degree-two vertex in
            // the arrangement, splitting one piece into two for no geometric reason. One cut
            // leaves the curve closed; it is merely re-seamed there.
            let first = cuts[0];
            let mut pieces = Vec::with_capacity(cuts.len());
            for window in cuts.windows(2) {
                pieces.push(self.sub_curve(window[0], window[1]));
            }
            pieces.push(self.sub_curve(cuts[cuts.len() - 1], first + 1.0));
            return pieces;
        }
        let mut pieces = Vec::with_capacity(cuts.len() + 1);
        let mut previous = 0.0;
        for cut in cuts.into_iter().chain(std::iter::once(1.0)) {
            pieces.push(self.sub_curve(previous, cut));
            previous = cut;
        }
        pieces
    }

    /// Every place this curve meets `other`, ascending by parameter on `self`.
    ///
    /// A transverse crossing is one entry. A tangency is one entry (the two roots have collapsed).
    /// A coincident stretch is reported as the two ends of the shared span, both flagged
    /// [`overlapping`](CurveCrossing::overlapping) — a caller cutting an arrangement wants exactly
    /// those two cuts, and a caller that treats a shared edge specially can see that it is one.
    ///
    /// A curve never crosses itself here: `self` against `self` is a total overlap, which is true
    /// but useless, so an arrangement filters identical pairs before asking.
    pub fn crossings(&self, other: &PlanarCurve) -> Vec<CurveCrossing> {
        let mut found = match (self, other) {
            (
                PlanarCurve::Segment { start: a0, end: a1 },
                PlanarCurve::Segment { start: b0, end: b1 },
            ) => segment_meets_segment(*a0, *a1, *b0, *b1),
            (PlanarCurve::Segment { start, end }, PlanarCurve::Arc { .. }) => {
                segment_meets_arc(*start, *end, other)
            }
            (PlanarCurve::Arc { .. }, PlanarCurve::Segment { start, end }) => {
                let mut mirrored = segment_meets_arc(*start, *end, self);
                for crossing in &mut mirrored {
                    std::mem::swap(
                        &mut crossing.parameter_on_first,
                        &mut crossing.parameter_on_second,
                    );
                }
                mirrored
            }
            (PlanarCurve::Arc { .. }, PlanarCurve::Arc { .. }) => arc_meets_arc(self, other),
        };
        found.sort_by(|a, b| a.parameter_on_first.total_cmp(&b.parameter_on_first));
        found
    }
}

/// Every curve cut at every crossing with every other, each returned as its ordered pieces.
///
/// This is the arrangement's first half: after it, no two pieces cross anywhere but at a shared
/// endpoint, which is the precondition a planar-graph face walk needs. The second half — matching
/// those endpoints up into vertices and tracing the faces — belongs to whoever owns the graph,
/// because that is where identity lives.
///
/// Quadratic in the number of curves. A sketch is drawn by hand, so the count is small and the
/// constant matters more than the exponent; a sweep-line would be the answer if that ever stopped
/// being true.
pub fn cut_at_crossings(curves: &[PlanarCurve]) -> Vec<Vec<PlanarCurve>> {
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); curves.len()];
    for first in 0..curves.len() {
        for second in (first + 1)..curves.len() {
            for crossing in curves[first].crossings(&curves[second]) {
                cuts[first].push(crossing.parameter_on_first);
                cuts[second].push(crossing.parameter_on_second);
            }
        }
    }
    curves
        .iter()
        .zip(cuts)
        .map(|(curve, parameters)| curve.split_at(&parameters))
        .collect()
}

/// One place two curves meet, located on both of them.
///
/// The parameters are what make this usable: a crossing you cannot cut at is only a yes/no answer,
/// and an arrangement needs to split both curves there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveCrossing {
    /// Where they meet.
    pub point: [f64; 2],
    /// How far along the first curve, `0` at its tail and `1` at its head.
    pub parameter_on_first: f64,
    /// How far along the second curve.
    pub parameter_on_second: f64,
    /// Whether the curves are COINCIDENT here rather than crossing — this is one end of a stretch
    /// they share, not a point they pass through.
    pub overlapping: bool,
}

impl CurveCrossing {
    /// A transverse crossing — the ordinary case.
    fn transverse(point: [f64; 2], parameter_on_first: f64, parameter_on_second: f64) -> Self {
        Self {
            point,
            parameter_on_first,
            parameter_on_second,
            overlapping: false,
        }
    }

    /// One end of a coincident stretch.
    fn shared(point: [f64; 2], parameter_on_first: f64, parameter_on_second: f64) -> Self {
        Self {
            point,
            parameter_on_first,
            parameter_on_second,
            overlapping: true,
        }
    }
}

/// Whether a parameter lies within `[0, 1]` once the epsilon is allowed, clamped into it so a
/// crossing at an endpoint reports exactly `0` or `1` rather than a hair outside.
fn clamped_parameter(parameter: f64, span_length: f64) -> Option<f64> {
    // The epsilon is a DISTANCE, so it converts to a parameter through the curve's own length —
    // otherwise a short curve tolerates a huge overshoot and a long one tolerates none.
    let slack = if span_length > CROSSING_EPSILON {
        CROSSING_EPSILON / span_length
    } else {
        1.0
    };
    (parameter >= -slack && parameter <= 1.0 + slack).then(|| parameter.clamp(0.0, 1.0))
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn length(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

/// Segment against segment: the parametric cross-product solve, with the collinear case answered
/// as the shared span rather than as nothing.
fn segment_meets_segment(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Vec<CurveCrossing> {
    let first = [a1[0] - a0[0], a1[1] - a0[1]];
    let second = [b1[0] - b0[0], b1[1] - b0[1]];
    let offset = [b0[0] - a0[0], b0[1] - a0[1]];
    let denominator = cross(first, second);
    let (first_length, second_length) = (length(first), length(second));
    // The determinant scales with both lengths, so the parallel test has to as well; comparing it
    // against a bare epsilon calls two long near-parallel segments crossing and two short ones
    // parallel, which is exactly backwards.
    let parallel =
        denominator.abs() <= CROSSING_EPSILON * first_length.max(1.0) * second_length.max(1.0);
    if !parallel {
        let on_first = cross(offset, second) / denominator;
        let on_second = cross(offset, first) / denominator;
        let (Some(on_first), Some(on_second)) = (
            clamped_parameter(on_first, first_length),
            clamped_parameter(on_second, second_length),
        ) else {
            return Vec::new();
        };
        let point = [a0[0] + first[0] * on_first, a0[1] + first[1] * on_first];
        return vec![CurveCrossing::transverse(point, on_first, on_second)];
    }
    // Parallel. Off the same line ⇒ nothing; on it ⇒ the span they share.
    if cross(offset, first).abs() > CROSSING_EPSILON * first_length.max(1.0) {
        return Vec::new();
    }
    let squared = dot(first, first);
    if squared <= CROSSING_EPSILON {
        return Vec::new(); // a degenerate first segment has no span to share
    }
    let b0_on_first = dot(offset, first) / squared;
    let b1_on_first = b0_on_first + dot(second, first) / squared;
    let (low, high) = (b0_on_first.min(b1_on_first), b0_on_first.max(b1_on_first));
    let (overlap_low, overlap_high) = (low.max(0.0), high.min(1.0));
    if overlap_high < overlap_low {
        return Vec::new();
    }
    // Back onto the second curve's parameter. A degenerate second segment sits at one point, so
    // both ends map to zero rather than dividing by nothing.
    let to_second = |on_first: f64| {
        if (high - low).abs() <= f64::EPSILON {
            0.0
        } else {
            let raw = (on_first - b0_on_first) / (b1_on_first - b0_on_first);
            raw.clamp(0.0, 1.0)
        }
    };
    let mut ends = vec![CurveCrossing::shared(
        [
            a0[0] + first[0] * overlap_low,
            a0[1] + first[1] * overlap_low,
        ],
        overlap_low,
        to_second(overlap_low),
    )];
    if (overlap_high - overlap_low) * first_length > CROSSING_EPSILON {
        ends.push(CurveCrossing::shared(
            [
                a0[0] + first[0] * overlap_high,
                a0[1] + first[1] * overlap_high,
            ],
            overlap_high,
            to_second(overlap_high),
        ));
    }
    ends
}

/// Segment against arc: substitute the line into the circle, then clip both to their own ranges.
fn segment_meets_arc(a0: [f64; 2], a1: [f64; 2], arc: &PlanarCurve) -> Vec<CurveCrossing> {
    let PlanarCurve::Arc {
        center,
        radius,
        start_radians,
        sweep_radians,
    } = *arc
    else {
        return Vec::new();
    };
    let direction = [a1[0] - a0[0], a1[1] - a0[1]];
    let to_start = [a0[0] - center[0], a0[1] - center[1]];
    let quadratic = dot(direction, direction);
    if quadratic <= CROSSING_EPSILON {
        return Vec::new(); // a degenerate segment is a point, and a point is not a crossing
    }
    let linear = 2.0 * dot(to_start, direction);
    let constant = dot(to_start, to_start) - radius * radius;
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.max(0.0).sqrt();
    let segment_length = length(direction);
    // A tangency has both roots at the same place; emitting it twice would make an arrangement cut
    // the same point twice, so the near-zero discriminant collapses to one.
    let parameters: &[f64] = if root <= CROSSING_EPSILON * quadratic.max(1.0) {
        &[-linear / (2.0 * quadratic)]
    } else {
        &[
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ]
    };
    let mut found = Vec::new();
    for &raw in parameters {
        let Some(on_segment) = clamped_parameter(raw, segment_length) else {
            continue;
        };
        let point = [
            a0[0] + direction[0] * on_segment,
            a0[1] + direction[1] * on_segment,
        ];
        let Some(on_arc) = parameter_on_arc(center, start_radians, sweep_radians, point) else {
            continue;
        };
        found.push(CurveCrossing::transverse(point, on_segment, on_arc));
    }
    found
}

/// Arc against arc: the radical-line construction for two distinct circles, and an angular
/// interval intersection when they are the SAME circle.
fn arc_meets_arc(first: &PlanarCurve, second: &PlanarCurve) -> Vec<CurveCrossing> {
    let (
        PlanarCurve::Arc {
            center: center_a,
            radius: radius_a,
            start_radians: start_a,
            sweep_radians: sweep_a,
        },
        PlanarCurve::Arc {
            center: center_b,
            radius: radius_b,
            start_radians: start_b,
            sweep_radians: sweep_b,
        },
    ) = (*first, *second)
    else {
        return Vec::new();
    };
    let between = [center_b[0] - center_a[0], center_b[1] - center_a[1]];
    let distance = length(between);
    if distance <= CROSSING_EPSILON && (radius_a - radius_b).abs() <= CROSSING_EPSILON {
        return coincident_arcs(center_a, radius_a, (start_a, sweep_a), (start_b, sweep_b));
    }
    // Separate, or one strictly inside the other: no circle meets the other at all.
    if distance > radius_a + radius_b + CROSSING_EPSILON
        || distance < (radius_a - radius_b).abs() - CROSSING_EPSILON
        || distance <= CROSSING_EPSILON
    {
        return Vec::new();
    }
    // The radical line: the crossings sit at `along` down the center line, `off` to either side.
    let along =
        (distance * distance + radius_a * radius_a - radius_b * radius_b) / (2.0 * distance);
    let off = (radius_a * radius_a - along * along).max(0.0).sqrt();
    let unit = [between[0] / distance, between[1] / distance];
    let base = [center_a[0] + unit[0] * along, center_a[1] + unit[1] * along];
    let normal = [-unit[1], unit[0]];
    // Tangent circles have one crossing, not two at the same place.
    let candidates: Vec<[f64; 2]> = if off <= CROSSING_EPSILON {
        vec![base]
    } else {
        vec![
            [base[0] + normal[0] * off, base[1] + normal[1] * off],
            [base[0] - normal[0] * off, base[1] - normal[1] * off],
        ]
    };
    let mut found = Vec::new();
    for point in candidates {
        let (Some(on_first), Some(on_second)) = (
            parameter_on_arc(center_a, start_a, sweep_a, point),
            parameter_on_arc(center_b, start_b, sweep_b, point),
        ) else {
            continue;
        };
        found.push(CurveCrossing::transverse(point, on_first, on_second));
    }
    found
}

/// Two arcs of the SAME circle: the stretches they share, each reported as its two ends.
fn coincident_arcs(
    center: [f64; 2],
    radius: f64,
    first: (f64, f64),
    second: (f64, f64),
) -> Vec<CurveCrossing> {
    let mut found = Vec::new();
    for (begin, span) in shared_angular_spans(
        counter_clockwise_span(first.0, first.1),
        counter_clockwise_span(second.0, second.1),
    ) {
        for bearing in [begin, begin + span] {
            let point = [
                center[0] + radius * bearing.cos(),
                center[1] + radius * bearing.sin(),
            ];
            let (Some(on_first), Some(on_second)) = (
                parameter_on_arc(center, first.0, first.1, point),
                parameter_on_arc(center, second.0, second.1, point),
            ) else {
                continue;
            };
            found.push(CurveCrossing::shared(point, on_first, on_second));
            if span <= ANGULAR_EPSILON {
                break; // a single touching bearing, not a stretch
            }
        }
    }
    found
}

/// An arc as a counter-clockwise span `(begin, length)` with `begin` normalized into `[0, TAU)` —
/// the form two arcs can be compared in regardless of which way each was drawn.
fn counter_clockwise_span(start_radians: f64, sweep_radians: f64) -> (f64, f64) {
    let begin = if sweep_radians < 0.0 {
        start_radians + sweep_radians
    } else {
        start_radians
    };
    (begin.rem_euclid(TAU), sweep_radians.abs().min(TAU))
}

/// The overlaps of two counter-clockwise angular spans on the circle, as `(begin, length)` pairs.
///
/// Two spans on a circle can overlap in TWO disjoint stretches — one at each end of the gap between
/// them — which is why this returns a list. Comparing the second span against the first at three
/// offsets covers every way the wrap can land.
fn shared_angular_spans(first: (f64, f64), second: (f64, f64)) -> Vec<(f64, f64)> {
    let (first_begin, first_span) = first;
    let (second_begin, second_span) = second;
    let mut shared = Vec::new();
    for turn in [-TAU, 0.0, TAU] {
        let low = (second_begin + turn).max(first_begin);
        let high = (second_begin + turn + second_span).min(first_begin + first_span);
        if high >= low - ANGULAR_EPSILON {
            shared.push((low, (high - low).max(0.0)));
        }
    }
    // Two candidate windows can name the same stretch when a span is a full turn; keep the first.
    shared.dedup_by(|a, b| {
        (a.0 - b.0).abs() <= ANGULAR_EPSILON && (a.1 - b.1).abs() <= ANGULAR_EPSILON
    });
    shared
}

/// How far along the arc `point` sits, as a parameter in `[0, 1]`, or `None` when its bearing is
/// off the sweep entirely.
fn parameter_on_arc(
    center: [f64; 2],
    start_radians: f64,
    sweep_radians: f64,
    point: [f64; 2],
) -> Option<f64> {
    let bearing = (point[1] - center[1]).atan2(point[0] - center[0]);
    let magnitude = sweep_radians.abs();
    if magnitude <= ANGULAR_EPSILON {
        return None;
    }
    // Travel is measured in the arc's own direction, so a clockwise arc is the mirror of a
    // counter-clockwise one rather than a second set of comparisons to keep in step.
    let travelled = if sweep_radians < 0.0 {
        (start_radians - bearing).rem_euclid(TAU)
    } else {
        (bearing - start_radians).rem_euclid(TAU)
    };
    // A bearing a hair BEFORE the start wraps to nearly a full turn; it belongs at zero.
    let travelled = if travelled >= TAU - ANGULAR_EPSILON {
        0.0
    } else {
        travelled
    };
    (travelled <= magnitude + ANGULAR_EPSILON).then(|| (travelled / magnitude).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(a: [f64; 2], b: [f64; 2]) -> PlanarCurve {
        PlanarCurve::Segment { start: a, end: b }
    }

    fn arc(center: [f64; 2], radius: f64, start_degrees: f64, sweep_degrees: f64) -> PlanarCurve {
        PlanarCurve::Arc {
            center,
            radius,
            start_radians: start_degrees.to_radians(),
            sweep_radians: sweep_degrees.to_radians(),
        }
    }

    fn assert_near(actual: [f64; 2], expected: [f64; 2]) {
        assert!(
            (actual[0] - expected[0]).abs() < 1e-6 && (actual[1] - expected[1]).abs() < 1e-6,
            "got {actual:?}, want {expected:?}"
        );
    }

    #[test]
    fn two_crossing_segments_meet_once_in_the_middle_of_both() {
        let crossings = segment([0.0, 0.0], [4.0, 4.0]).crossings(&segment([0.0, 4.0], [4.0, 0.0]));
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [2.0, 2.0]);
        assert!((crossings[0].parameter_on_first - 0.5).abs() < 1e-9);
        assert!((crossings[0].parameter_on_second - 0.5).abs() < 1e-9);
        assert!(!crossings[0].overlapping);
    }

    /// The parameter is the point of the whole exercise: a crossing you cannot cut at answers
    /// nothing an arrangement can use.
    #[test]
    fn the_parameter_locates_the_crossing_for_a_cut() {
        let first = segment([0.0, 0.0], [10.0, 0.0]);
        let crossings = first.crossings(&segment([2.0, -1.0], [2.0, 3.0]));
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0].parameter_on_first - 0.2).abs() < 1e-9);
        assert!((crossings[0].parameter_on_second - 0.25).abs() < 1e-9);
        assert_near(first.point_at(crossings[0].parameter_on_first), [2.0, 0.0]);
    }

    #[test]
    fn segments_that_only_would_cross_if_extended_do_not() {
        assert!(segment([0.0, 0.0], [1.0, 0.0])
            .crossings(&segment([5.0, -1.0], [5.0, 1.0]))
            .is_empty());
    }

    #[test]
    fn segments_meeting_at_an_endpoint_report_it_at_the_end() {
        let crossings = segment([0.0, 0.0], [4.0, 0.0]).crossings(&segment([4.0, 0.0], [4.0, 4.0]));
        assert_eq!(crossings.len(), 1);
        assert_eq!(crossings[0].parameter_on_first, 1.0);
        assert_eq!(crossings[0].parameter_on_second, 0.0);
    }

    #[test]
    fn parallel_segments_off_the_same_line_never_meet() {
        assert!(segment([0.0, 0.0], [4.0, 0.0])
            .crossings(&segment([0.0, 1.0], [4.0, 1.0]))
            .is_empty());
    }

    /// The degenerate pair. A point answer would be a lie here: they share a stretch, and its two
    /// ends are where an arrangement has to cut.
    #[test]
    fn collinear_segments_report_the_span_they_share() {
        let crossings =
            segment([0.0, 0.0], [10.0, 0.0]).crossings(&segment([4.0, 0.0], [14.0, 0.0]));
        assert_eq!(crossings.len(), 2);
        assert!(crossings.iter().all(|crossing| crossing.overlapping));
        assert_near(crossings[0].point, [4.0, 0.0]);
        assert_near(crossings[1].point, [10.0, 0.0]);
        assert!((crossings[0].parameter_on_first - 0.4).abs() < 1e-9);
        assert!((crossings[1].parameter_on_first - 1.0).abs() < 1e-9);
    }

    #[test]
    fn collinear_segments_that_only_touch_report_one_point() {
        let crossings = segment([0.0, 0.0], [4.0, 0.0]).crossings(&segment([4.0, 0.0], [8.0, 0.0]));
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [4.0, 0.0]);
    }

    #[test]
    fn a_segment_through_a_circle_meets_it_twice() {
        let crossings =
            segment([-10.0, 0.0], [10.0, 0.0]).crossings(&PlanarCurve::circle([0.0, 0.0], 4.0));
        assert_eq!(crossings.len(), 2);
        assert_near(crossings[0].point, [-4.0, 0.0]);
        assert_near(crossings[1].point, [4.0, 0.0]);
    }

    /// A tangent line touches once. Reporting it twice would make an arrangement cut the same
    /// place twice and derive a zero-length piece.
    #[test]
    fn a_tangent_line_touches_a_circle_once() {
        let crossings =
            segment([-10.0, 4.0], [10.0, 4.0]).crossings(&PlanarCurve::circle([0.0, 0.0], 4.0));
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [0.0, 4.0]);
    }

    /// The clip that makes an ARC different from the circle it lies on: the line still meets the
    /// circle twice, but only one of those is on the quarter that was drawn.
    #[test]
    fn only_the_crossings_on_the_drawn_arc_count() {
        let quarter = arc([0.0, 0.0], 4.0, 0.0, 90.0);
        let crossings = segment([-10.0, 0.0], [10.0, 0.0]).crossings(&quarter);
        assert_eq!(crossings.len(), 1, "the -4 crossing is off the quarter");
        assert_near(crossings[0].point, [4.0, 0.0]);
        assert_eq!(crossings[0].parameter_on_second, 0.0);
    }

    #[test]
    fn a_segment_missing_the_circle_entirely_reports_nothing() {
        assert!(segment([-10.0, 9.0], [10.0, 9.0])
            .crossings(&PlanarCurve::circle([0.0, 0.0], 4.0))
            .is_empty());
    }

    /// The case the whole arrangement exists for: two overlapping circles cross at two points that
    /// belong to neither as a vertex. A graph walk finds nothing here.
    #[test]
    fn two_overlapping_circles_cross_twice() {
        let crossings =
            PlanarCurve::circle([0.0, 0.0], 5.0).crossings(&PlanarCurve::circle([6.0, 0.0], 5.0));
        assert_eq!(crossings.len(), 2);
        for crossing in &crossings {
            assert!((crossing.point[0] - 3.0).abs() < 1e-9);
            assert!((crossing.point[1].abs() - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn circles_that_miss_or_nest_never_meet() {
        let outer = PlanarCurve::circle([0.0, 0.0], 5.0);
        assert!(outer
            .crossings(&PlanarCurve::circle([20.0, 0.0], 5.0))
            .is_empty());
        assert!(
            outer
                .crossings(&PlanarCurve::circle([0.0, 0.0], 2.0))
                .is_empty(),
            "a circle inside another never touches it"
        );
    }

    #[test]
    fn externally_tangent_circles_touch_once() {
        let crossings =
            PlanarCurve::circle([0.0, 0.0], 4.0).crossings(&PlanarCurve::circle([8.0, 0.0], 4.0));
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [4.0, 0.0]);
    }

    /// Two arcs of the same circle: the coincident case, answered as the stretch they share.
    #[test]
    fn arcs_of_one_circle_report_the_stretch_they_share() {
        let crossings =
            arc([0.0, 0.0], 4.0, 0.0, 180.0).crossings(&arc([0.0, 0.0], 4.0, 90.0, 180.0));
        assert_eq!(crossings.len(), 2);
        assert!(crossings.iter().all(|crossing| crossing.overlapping));
        assert_near(crossings[0].point, [0.0, 4.0]);
        assert_near(crossings[1].point, [-4.0, 0.0]);
    }

    /// Arcs of one circle that do not overlap share nothing, even though every point of each is on
    /// the other's circle.
    #[test]
    fn disjoint_arcs_of_one_circle_share_nothing() {
        let crossings =
            arc([0.0, 0.0], 4.0, 0.0, 80.0).crossings(&arc([0.0, 0.0], 4.0, 180.0, 80.0));
        assert!(crossings.is_empty(), "got {crossings:?}");
    }

    /// A clockwise arc is the same curve as its counter-clockwise twin, so the crossings are the
    /// same points — only the parameters run the other way.
    #[test]
    fn direction_moves_the_parameter_not_the_crossing() {
        let forward = arc([0.0, 0.0], 4.0, 0.0, 90.0);
        let backward = arc([0.0, 0.0], 4.0, 90.0, -90.0);
        let cut = segment([0.0, -10.0], [0.0, 10.0]);
        let ahead = cut.crossings(&forward);
        let behind = cut.crossings(&backward);
        assert_eq!(ahead.len(), 1);
        assert_eq!(behind.len(), 1);
        assert_near(ahead[0].point, behind[0].point);
        assert!((ahead[0].parameter_on_second - 1.0).abs() < 1e-9);
        assert!((behind[0].parameter_on_second - 0.0).abs() < 1e-9);
    }

    /// A whole circle is an ordinary curve here, not a degenerate one: it has a parameterisation
    /// and crossings land on it like anywhere else.
    #[test]
    fn a_whole_circle_parameterises_all_the_way_round() {
        let circle = PlanarCurve::circle([0.0, 0.0], 4.0);
        assert!(circle.is_closed());
        assert_near(circle.start(), [4.0, 0.0]);
        assert_near(circle.end(), [4.0, 0.0]);
        assert_near(circle.point_at(0.25), [0.0, 4.0]);
        assert_near(circle.point_at(0.5), [-4.0, 0.0]);
        let crossings = segment([0.0, 0.0], [0.0, 10.0]).crossings(&circle);
        assert_eq!(crossings.len(), 1);
        assert!((crossings[0].parameter_on_second - 0.25).abs() < 1e-9);
    }

    /// Cutting an arc gives back arcs of the SAME circle, not chords. If it did not, splitting a
    /// curve would be a lossy operation and the arrangement would flatten everything it touched.
    #[test]
    fn a_cut_arc_is_still_an_arc_of_its_circle() {
        let half = arc([1.0, 2.0], 5.0, 0.0, 180.0);
        let pieces = half.split_at(&[0.5]);
        assert_eq!(pieces.len(), 2);
        for piece in &pieces {
            let PlanarCurve::Arc { center, radius, .. } = piece else {
                panic!("a cut arc became {piece:?}");
            };
            assert_eq!(*center, [1.0, 2.0]);
            assert_eq!(*radius, 5.0);
        }
        assert_near(pieces[0].start(), half.start());
        assert_near(pieces[0].end(), half.point_at(0.5));
        assert_near(pieces[1].end(), half.end());
    }

    #[test]
    fn cuts_at_the_ends_and_on_top_of_each_other_are_dropped() {
        let span = segment([0.0, 0.0], [10.0, 0.0]);
        assert_eq!(span.split_at(&[0.0, 1.0]).len(), 1, "the ends are not cuts");
        assert_eq!(
            span.split_at(&[0.5, 0.5 + 1.0e-15]).len(),
            2,
            "two cuts at one place are one cut"
        );
        assert_eq!(span.split_at(&[-0.5, 1.5]).len(), 1, "off the curve");
    }

    /// The flagship: two overlapping circles cut each other into four pieces, two apiece — the
    /// decomposition three regions are traced from.
    #[test]
    fn two_overlapping_circles_cut_each_other_in_two() {
        let pieces = cut_at_crossings(&[
            PlanarCurve::circle([0.0, 0.0], 5.0),
            PlanarCurve::circle([6.0, 0.0], 5.0),
        ]);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].len(), 2, "the first circle, cut twice");
        assert_eq!(pieces[1].len(), 2, "and so is the second");
        for arcs in &pieces {
            for piece in arcs {
                assert!(!piece.is_closed(), "a cut circle is no longer closed");
            }
        }
        // The pieces wrap through the seam rather than stopping at it: together they cover the
        // whole circle, and neither ends where the circle happened to be written down from.
        let total: f64 = pieces[0].iter().map(PlanarCurve::length).sum();
        assert!(
            (total - TAU * 5.0).abs() < 1e-9,
            "the pieces cover the circle"
        );
    }

    /// One cut on a closed curve leaves it closed — a tangency does not open a circle up, it only
    /// moves where the loop is written from.
    #[test]
    fn a_single_cut_leaves_a_closed_curve_closed() {
        let pieces = PlanarCurve::circle([0.0, 0.0], 4.0).split_at(&[0.25]);
        assert_eq!(pieces.len(), 1);
        assert!(pieces[0].is_closed());
        assert_near(pieces[0].start(), [0.0, 4.0]);
    }

    /// A circle nothing crosses stays whole. It has no vertex, so it cannot be expressed as a chain
    /// of pieces — it is a loop already.
    #[test]
    fn an_uncrossed_closed_curve_comes_back_whole() {
        let pieces = cut_at_crossings(&[
            PlanarCurve::circle([0.0, 0.0], 5.0),
            segment([20.0, 0.0], [30.0, 0.0]),
        ]);
        assert_eq!(pieces[0].len(), 1);
        assert!(pieces[0][0].is_closed());
    }

    /// A line through a circle cuts BOTH: the circle into two arcs, the line into three spans.
    /// Every piece then meets its neighbors only at endpoints, which is what the face walk needs.
    #[test]
    fn a_line_through_a_circle_cuts_both_of_them() {
        let pieces = cut_at_crossings(&[
            PlanarCurve::circle([0.0, 0.0], 4.0),
            segment([-10.0, 0.0], [10.0, 0.0]),
        ]);
        assert_eq!(pieces[0].len(), 2, "two arcs");
        assert_eq!(pieces[1].len(), 3, "outside, across, outside");
        assert_near(pieces[1][0].end(), [-4.0, 0.0]);
        assert_near(pieces[1][1].end(), [4.0, 0.0]);
    }

    /// A tiny curve and a huge one are held to the same DISTANCE tolerance, not the same parameter
    /// tolerance — otherwise a long segment tolerates no overshoot and a short one tolerates a
    /// visible gap.
    #[test]
    fn the_tolerance_is_a_distance_not_a_parameter() {
        let long = segment([0.0, 0.0], [1.0e6, 0.0]);
        // A hair past the far end: far outside the parameter epsilon, well inside the distance one.
        let just_past = segment([1.0e6 + 1.0e-10, -1.0], [1.0e6 + 1.0e-10, 1.0]);
        assert_eq!(long.crossings(&just_past).len(), 1);
        // A whole voxel past it is a genuine miss.
        let clearly_past = segment([1.0e6 + 1.0, -1.0], [1.0e6 + 1.0, 1.0]);
        assert!(long.crossings(&clearly_past).is_empty());
    }
}
