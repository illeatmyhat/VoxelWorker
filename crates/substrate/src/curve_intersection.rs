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

use crate::rational_bezier::RationalBezier;

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

/// How near a curve a point has to be before the arrangement calls it the SAME PLACE, in the
/// curve's own units.
///
/// This is the arrangement's spatial resolution, and it is one number rather than two on purpose:
/// the face walk welds piece ends into vertices at exactly this distance, and
/// [`cut_at_crossings`] cuts a curve under a foreign endpoint at exactly this distance. The two
/// have to agree, and the direction of the requirement is not symmetric. A cut placed FURTHER
/// than the weld distance from the endpoint that asked for it does not weld to that endpoint, so
/// it manufactures a pair of vertices a hair apart where the author put one thing — strictly
/// worse than not cutting. So the cut tolerance can never exceed the weld tolerance; making them
/// equal is what leaves no band in between where two things are one vertex but no cut was made.
///
/// It is safe because the two populations are nowhere near it. A point a solve has been asked to
/// hold on a curve lands about `1e-8` off, ten thousand times inside; two things an author drew
/// separately are apart by something they could see, which is millions of times outside. Nothing
/// legitimate lives near the threshold, which is why a tolerance is honest here at all.
pub const VERTEX_WELD_EPSILON: f64 = 1.0e-6;

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
    /// A cubic rational Bézier piece. Polynomial cubics, exact conics, and ellipse quarters all
    /// share this representation; a multi-piece spline or closed ellipse is a sequence of these.
    RationalBezier(RationalBezier),
}

impl PlanarCurve {
    /// A whole circle: the arc that sweeps a full turn counter-clockwise from bearing zero.
    #[must_use]
    pub const fn circle(center: [f64; 2], radius: f64) -> Self {
        Self::Arc {
            center,
            radius,
            start_radians: 0.0,
            sweep_radians: TAU,
        }
    }

    /// The point at `parameter`, which runs `0` at the tail to `1` at the head. Outside that range
    /// the curve is extended — the segment along its line, the arc around its circle — which is
    /// what makes an out-of-range solve reportable rather than silently clamped.
    #[must_use]
    pub fn point_at(&self, parameter: f64) -> [f64; 2] {
        match *self {
            Self::Segment { start, end } => [
                (end[0] - start[0]).mul_add(parameter, start[0]),
                (end[1] - start[1]).mul_add(parameter, start[1]),
            ],
            Self::Arc {
                center,
                radius,
                start_radians,
                sweep_radians,
            } => {
                let bearing = sweep_radians.mul_add(parameter, start_radians);
                [
                    radius.mul_add(bearing.cos(), center[0]),
                    radius.mul_add(bearing.sin(), center[1]),
                ]
            }
            Self::RationalBezier(curve) => curve.point_at(parameter),
        }
    }

    /// The curve's tail.
    #[must_use]
    pub fn start(&self) -> [f64; 2] {
        self.point_at(0.0)
    }

    /// The curve's head.
    #[must_use]
    pub fn end(&self) -> [f64; 2] {
        self.point_at(1.0)
    }

    /// Whether the curve closes on itself — a whole circle.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        match *self {
            Self::Arc { sweep_radians, .. } => sweep_radians.abs() >= TAU - ANGULAR_EPSILON,
            Self::Segment { .. } | Self::RationalBezier(_) => false,
        }
    }

    /// How long the curve is, in the units its coordinates are in.
    #[must_use]
    pub fn length(&self) -> f64 {
        match *self {
            Self::Segment { start, end } => length([end[0] - start[0], end[1] - start[1]]),
            Self::Arc {
                radius,
                sweep_radians,
                ..
            } => radius * sweep_radians.abs(),
            Self::RationalBezier(curve) => curve
                .flatten(CROSSING_EPSILON.sqrt())
                .array_windows::<2>()
                .map(|pair| length([pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]]))
                .sum(),
        }
    }

    /// The parameter of the point on this finite curve nearest `witness`.
    ///
    /// Segments use the clamped orthogonal projection. Arcs use the witness bearing when it lies
    /// inside the signed sweep and otherwise choose the nearer endpoint. A whole circle has no
    /// endpoints, so every bearing maps directly into its wrapped `[0, 1)` parameter.
    #[must_use]
    pub fn nearest_parameter(&self, witness: [f64; 2]) -> f64 {
        match *self {
            Self::Segment { start, end } => {
                let direction = [end[0] - start[0], end[1] - start[1]];
                let length_squared =
                    direction[0].mul_add(direction[0], direction[1] * direction[1]);
                if length_squared <= f64::EPSILON {
                    return 0.0;
                }
                (direction[0].mul_add(
                    witness[0] - start[0],
                    direction[1] * (witness[1] - start[1]),
                ) / length_squared)
                    .clamp(0.0, 1.0)
            }
            Self::Arc {
                center,
                start_radians,
                sweep_radians,
                ..
            } => {
                let bearing = (witness[1] - center[1]).atan2(witness[0] - center[0]);
                let travelled = if sweep_radians.is_sign_negative() {
                    (start_radians - bearing).rem_euclid(TAU)
                } else {
                    (bearing - start_radians).rem_euclid(TAU)
                };
                let parameter = travelled / sweep_radians.abs();
                if self.is_closed() || parameter <= 1.0 {
                    return parameter.min(1.0);
                }
                let start_distance = squared_distance(witness, self.start());
                let end_distance = squared_distance(witness, self.end());
                if start_distance <= end_distance {
                    0.0
                } else {
                    1.0
                }
            }
            Self::RationalBezier(curve) => nearest_bezier_parameter(curve, witness),
        }
    }

    /// The stretch of this curve between two parameters, as a curve in its own right.
    ///
    /// An arc keeps its circle and narrows its sweep — it does not become a chord, and it does not
    /// get re-solved from the new endpoints. That is the whole point: cutting a curve produces
    /// pieces of the SAME curve, so nothing is approximated by being split.
    #[must_use]
    pub fn sub_curve(&self, from: f64, to: f64) -> Self {
        match *self {
            Self::Segment { .. } => Self::Segment {
                start: self.point_at(from),
                end: self.point_at(to),
            },
            Self::Arc {
                center,
                radius,
                start_radians,
                sweep_radians,
            } => Self::Arc {
                center,
                radius,
                start_radians: sweep_radians.mul_add(from, start_radians),
                sweep_radians: sweep_radians * (to - from),
            },
            Self::RationalBezier(curve) => Self::RationalBezier(curve.sub_curve(from, to)),
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
    #[must_use]
    pub fn split_at(&self, parameters: &[f64]) -> Vec<Self> {
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
            // A closed curve's seam is an artifact of how it was written down, not a place on it.
            // So the pieces run between consecutive cuts and the last one WRAPS through the seam
            // back to the first — otherwise the seam would become a spurious degree-two vertex in
            // the arrangement, splitting one piece into two for no geometric reason. One cut
            // leaves the curve closed; it is merely re-seamed there.
            let Some(&first) = cuts.first() else {
                return vec![*self];
            };
            let mut pieces = Vec::with_capacity(cuts.len());
            for window in cuts.array_windows::<2>() {
                let [from, to] = window;
                pieces.push(self.sub_curve(*from, *to));
            }
            let Some(&last) = cuts.last() else {
                return vec![*self];
            };
            pieces.push(self.sub_curve(last, first + 1.0));
            return pieces;
        }
        let mut pieces = Vec::with_capacity(cuts.len().saturating_add(1));
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
    #[must_use]
    pub fn crossings(&self, other: &Self) -> Vec<CurveCrossing> {
        let mut found = match (self, other) {
            (Self::Segment { start: a0, end: a1 }, Self::Segment { start: b0, end: b1 }) => {
                segment_meets_segment(*a0, *a1, *b0, *b1)
            }
            (Self::Segment { start, end }, Self::Arc { .. }) => {
                segment_meets_arc(*start, *end, other)
            }
            (Self::Arc { .. }, Self::Segment { start, end }) => {
                let mut mirrored = segment_meets_arc(*start, *end, self);
                for crossing in &mut mirrored {
                    std::mem::swap(
                        &mut crossing.parameter_on_first,
                        &mut crossing.parameter_on_second,
                    );
                }
                mirrored
            }
            (Self::Arc { .. }, Self::Arc { .. }) => arc_meets_arc(self, other),
            (Self::RationalBezier(first), Self::RationalBezier(second)) => {
                rational_bezier_meets_curve(*first, *second)
            }
            (Self::RationalBezier(bezier), _) => bezier_meets_planar(*bezier, *other),
            (_, Self::RationalBezier(bezier)) => {
                let mut mirrored = bezier_meets_planar(*bezier, *self);
                for crossing in &mut mirrored {
                    std::mem::swap(
                        &mut crossing.parameter_on_first,
                        &mut crossing.parameter_on_second,
                    );
                }
                mirrored
            }
        };
        found.sort_by(|a, b| a.parameter_on_first.total_cmp(&b.parameter_on_first));
        found
    }

    /// Every place the infinite support of this curve meets the finite authored span of `other`.
    ///
    /// A segment contributes its supporting line, whose parameter remains `0` at the authored
    /// tail and `1` at the authored head but may fall outside that interval. An arc contributes
    /// its full supporting circle, parameterized counter-clockwise from the positive x axis.
    /// This asymmetric query is the geometric primitive behind operations such as Extend: the
    /// first curve may grow, while the curve it is trying to reach must still be hit as drawn.
    #[must_use]
    pub fn support_crossings_with(&self, other: &Self) -> Vec<CurveSupportCrossing> {
        let mut found = match *self {
            Self::Segment { start, end } => match *other {
                Self::Segment {
                    start: other_start,
                    end: other_end,
                } => line_meets_segment(start, end, other_start, other_end),
                Self::Arc { .. } => line_meets_arc(start, end, other),
                Self::RationalBezier(_) => Vec::new(),
            },
            Self::Arc { center, radius, .. } => Self::circle(center, radius)
                .crossings(other)
                .into_iter()
                .map(CurveSupportCrossing::from_curve_crossing)
                .collect(),
            Self::RationalBezier(_) => Vec::new(),
        };
        found.sort_by(|a, b| a.parameter_on_support.total_cmp(&b.parameter_on_support));
        found
    }
}

/// A bounded numerical search is used only when at least one curve is rational Bézier. Analytic
/// line/circle pairs keep their closed-form paths above; the recursive search works in parameter
/// space and uses exact Bézier subdivision plus conservative control-hull bounds.
const RATIONAL_CROSSING_EPSILON: f64 = 1.0e-8;

fn nearest_bezier_parameter(curve: RationalBezier, witness: [f64; 2]) -> f64 {
    const SAMPLES: u32 = 32;
    let mut best_parameter = 0.0;
    let mut best_distance = squared_distance(curve.point_at(0.0), witness);
    for index in 1..=SAMPLES {
        let parameter = f64::from(index) / f64::from(SAMPLES);
        let distance = squared_distance(curve.point_at(parameter), witness);
        if distance < best_distance {
            best_parameter = parameter;
            best_distance = distance;
        }
    }
    let step = 1.0 / f64::from(SAMPLES);
    let mut low = (best_parameter - step).max(0.0);
    let mut high = (best_parameter + step).min(1.0);
    // Golden-section refinement is derivative-free and remains stable at cusps and endpoints.
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut left = high - ratio * (high - low);
    let mut right = low + ratio * (high - low);
    for _ in 0..48 {
        let left_distance = squared_distance(curve.point_at(left), witness);
        let right_distance = squared_distance(curve.point_at(right), witness);
        if left_distance <= right_distance {
            high = right;
            right = left;
            left = high - ratio * (high - low);
        } else {
            low = left;
            left = right;
            right = low + ratio * (high - low);
        }
    }
    (low + high) * 0.5
}

fn rational_bezier_meets_curve(
    first: RationalBezier,
    second: RationalBezier,
) -> Vec<CurveCrossing> {
    if first == second {
        return vec![
            CurveCrossing::shared(first.control[0], 0.0, 0.0),
            CurveCrossing::shared(first.control[3], 1.0, 1.0),
        ];
    }
    if first == second.reversed() {
        return vec![
            CurveCrossing::shared(first.control[0], 0.0, 1.0),
            CurveCrossing::shared(first.control[3], 1.0, 0.0),
        ];
    }
    numeric_curve_crossings(
        PlanarCurve::RationalBezier(first),
        PlanarCurve::RationalBezier(second),
    )
}

fn bezier_meets_planar(bezier: RationalBezier, other: PlanarCurve) -> Vec<CurveCrossing> {
    numeric_curve_crossings(PlanarCurve::RationalBezier(bezier), other)
}

#[derive(Clone, Copy)]
struct ParameterSpan {
    curve: PlanarCurve,
    from: f64,
    to: f64,
}

impl ParameterSpan {
    fn halves(self) -> [Self; 2] {
        let middle = (self.from + self.to) * 0.5;
        [
            Self {
                curve: self.curve.sub_curve(0.0, 0.5),
                from: self.from,
                to: middle,
            },
            Self {
                curve: self.curve.sub_curve(0.5, 1.0),
                from: middle,
                to: self.to,
            },
        ]
    }

    fn midpoint(self) -> (f64, [f64; 2]) {
        ((self.from + self.to) * 0.5, self.curve.point_at(0.5))
    }
}

fn numeric_curve_crossings(first: PlanarCurve, second: PlanarCurve) -> Vec<CurveCrossing> {
    let mut found = Vec::new();
    intersect_parameter_spans(
        ParameterSpan {
            curve: first,
            from: 0.0,
            to: 1.0,
        },
        ParameterSpan {
            curve: second,
            from: 0.0,
            to: 1.0,
        },
        0,
        &mut found,
    );
    found.sort_by(|a, b| a.parameter_on_first.total_cmp(&b.parameter_on_first));
    found.dedup_by(|later, earlier| {
        (later.parameter_on_first - earlier.parameter_on_first).abs() <= 1.0e-6
            && (later.parameter_on_second - earlier.parameter_on_second).abs() <= 1.0e-6
    });
    found
}

fn intersect_parameter_spans(
    first: ParameterSpan,
    second: ParameterSpan,
    depth: u8,
    found: &mut Vec<CurveCrossing>,
) {
    const MAX_DEPTH: u8 = 52;
    let first_bounds = planar_curve_bounds(first.curve);
    let second_bounds = planar_curve_bounds(second.curve);
    if !bounds_overlap(first_bounds, second_bounds) {
        return;
    }
    let first_size = bounds_size(first_bounds);
    let second_size = bounds_size(second_bounds);
    if depth >= MAX_DEPTH
        || (first_size <= RATIONAL_CROSSING_EPSILON && second_size <= RATIONAL_CROSSING_EPSILON)
    {
        if let Some(crossing) = transverse_parameter_chords(first, second) {
            found.push(crossing);
            return;
        }
        let (first_parameter, first_point) = first.midpoint();
        let (second_parameter, second_point) = second.midpoint();
        if squared_distance(first_point, second_point) <= (RATIONAL_CROSSING_EPSILON * 8.0).powi(2)
        {
            found.push(CurveCrossing::transverse(
                [
                    (first_point[0] + second_point[0]) * 0.5,
                    (first_point[1] + second_point[1]) * 0.5,
                ],
                first_parameter,
                second_parameter,
            ));
        }
        return;
    }
    if first_size >= second_size {
        for half in first.halves() {
            intersect_parameter_spans(half, second, depth.saturating_add(1), found);
        }
    } else {
        for half in second.halves() {
            intersect_parameter_spans(first, half, depth.saturating_add(1), found);
        }
    }
}

/// Intersect two already-converged parameter chords without the authored-segment minimum-length
/// policy. At this recursion leaf both chords are intentionally microscopic, so treating a
/// squared length below the document epsilon as degenerate would discard ordinary crossings.
fn transverse_parameter_chords(
    first: ParameterSpan,
    second: ParameterSpan,
) -> Option<CurveCrossing> {
    const PARAMETER_SLACK: f64 = 1.0e-5;
    let first_start = first.curve.start();
    let second_start = second.curve.start();
    let first_end = first.curve.end();
    let second_end = second.curve.end();
    let first_delta = [first_end[0] - first_start[0], first_end[1] - first_start[1]];
    let second_delta = [
        second_end[0] - second_start[0],
        second_end[1] - second_start[1],
    ];
    let offset = [
        second_start[0] - first_start[0],
        second_start[1] - first_start[1],
    ];
    let denominator = cross(first_delta, second_delta);
    let scale = length(first_delta) * length(second_delta);
    if scale <= f64::MIN_POSITIVE || denominator.abs() <= f64::EPSILON * scale {
        return None;
    }
    let on_first = cross(offset, second_delta) / denominator;
    let on_second = cross(offset, first_delta) / denominator;
    if !(-PARAMETER_SLACK..=1.0 + PARAMETER_SLACK).contains(&on_first)
        || !(-PARAMETER_SLACK..=1.0 + PARAMETER_SLACK).contains(&on_second)
    {
        return None;
    }
    let on_first = on_first.clamp(0.0, 1.0);
    let on_second = on_second.clamp(0.0, 1.0);
    let first_point = [
        first_delta[0].mul_add(on_first, first_start[0]),
        first_delta[1].mul_add(on_first, first_start[1]),
    ];
    let second_point = [
        second_delta[0].mul_add(on_second, second_start[0]),
        second_delta[1].mul_add(on_second, second_start[1]),
    ];
    Some(CurveCrossing::transverse(
        [
            (first_point[0] + second_point[0]) * 0.5,
            (first_point[1] + second_point[1]) * 0.5,
        ],
        (first.to - first.from).mul_add(on_first, first.from),
        (second.to - second.from).mul_add(on_second, second.from),
    ))
}

fn planar_curve_bounds(curve: PlanarCurve) -> ([f64; 2], [f64; 2]) {
    let start = curve.start();
    let end = curve.end();
    let mut low = [start[0].min(end[0]), start[1].min(end[1])];
    let mut high = [start[0].max(end[0]), start[1].max(end[1])];
    match curve {
        PlanarCurve::Segment { .. } => {}
        PlanarCurve::RationalBezier(bezier) => return bezier.control_bounds(),
        PlanarCurve::Arc {
            center,
            radius,
            start_radians,
            sweep_radians,
        } => {
            for bearing in [
                0.0,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                3.0 * std::f64::consts::FRAC_PI_2,
            ] {
                let point = [
                    radius.mul_add(bearing.cos(), center[0]),
                    radius.mul_add(bearing.sin(), center[1]),
                ];
                if parameter_on_arc(center, start_radians, sweep_radians, point).is_some() {
                    low = [low[0].min(point[0]), low[1].min(point[1])];
                    high = [high[0].max(point[0]), high[1].max(point[1])];
                }
            }
        }
    }
    (low, high)
}

fn bounds_overlap(first: ([f64; 2], [f64; 2]), second: ([f64; 2], [f64; 2])) -> bool {
    first.0[0] <= second.1[0] + RATIONAL_CROSSING_EPSILON
        && first.1[0] + RATIONAL_CROSSING_EPSILON >= second.0[0]
        && first.0[1] <= second.1[1] + RATIONAL_CROSSING_EPSILON
        && first.1[1] + RATIONAL_CROSSING_EPSILON >= second.0[1]
}

fn bounds_size(bounds: ([f64; 2], [f64; 2])) -> f64 {
    (bounds.1[0] - bounds.0[0]).max(bounds.1[1] - bounds.0[1])
}

/// Every curve cut at every crossing with every other, each returned as its ordered pieces.
///
/// This is the arrangement's first half: after it, no two pieces cross anywhere but at a shared
/// endpoint, which is the precondition a planar-graph face walk needs. The second half — matching
/// those endpoints up into vertices and tracing the faces — belongs to whoever owns the graph,
/// because that is where identity lives.
///
/// # Why a crossing solve alone does not deliver that
///
/// A curve that ENDS on another one meets it at the very boundary of its own parameter range, and
/// a root solved to be at zero is as likely to come back a hair below it as a hair above. Below,
/// there is no crossing to report, the other curve is never cut, and the endpoint is left sitting
/// in the middle of a piece — which is precisely the postcondition above, failing. It is not a
/// rare case: it is what every T-junction is, and what every point a solve holds ON a curve is,
/// and the side the residual falls on is not a fact about the drawing. So the ends are asked
/// separately, in DISTANCE, which is the quantity that actually decides whether two things are in
/// the same place; see [`VERTEX_WELD_EPSILON`].
///
/// # Both passes reject on bounding boxes first
///
/// Quadratic in the number of curves, and the count is NOT always small — a drawing of 64 arc slots
/// carries 256 curves, and every frame of a drag re-derives the whole arrangement because the
/// region memo is keyed on the store. Measured there, the pair work was 9.0 ms, of which the
/// endpoint pass alone was 8.3: it asked `nearest_parameter` of every curve about every other
/// curve's two ends, including the overwhelming majority that are nowhere near each other.
///
/// So each pass drops a pair its boxes say cannot meet, which is sound for a different reason in
/// each. A CROSSING lies on both curves, so it lies in both boxes. A cut under a foreign endpoint
/// is looser — the endpoint only has to come within [`VERTEX_WELD_EPSILON`] of the other curve —
/// so that reject pads by exactly that, and the pad is load-bearing rather than decorative: an
/// axis-aligned curve's box has ZERO extent across, and a stem landing a hair off it would be
/// dropped by an unpadded test. The boxes are computed ONCE, not per pair; per pair they would cost
/// about what they save. Together the two rejects took the same arrangement from 9.0 ms to 1.0.
///
/// A sweep-line is still the answer if the count grows another order of magnitude.
#[must_use]
pub fn cut_at_crossings(curves: &[PlanarCurve]) -> Vec<Vec<PlanarCurve>> {
    let mut cuts: Vec<Vec<f64>> = vec![Vec::new(); curves.len()];
    let boxes: Vec<([f64; 2], [f64; 2])> =
        curves.iter().copied().map(planar_curve_bounds).collect();
    for (first_index, first_curve) in curves.iter().enumerate() {
        let following_start = first_index.saturating_add(1);
        let (first_cuts, following_cuts) = cuts.split_at_mut(following_start);
        let Some(first_cuts) = first_cuts.last_mut() else {
            continue;
        };
        let Some(first_box) = boxes.get(first_index).copied() else {
            continue;
        };
        for ((second_curve, second_cuts), second_box) in curves
            .iter()
            .skip(following_start)
            .zip(following_cuts.iter_mut())
            .zip(boxes.iter().skip(following_start).copied())
        {
            if !bounds_overlap(first_box, second_box) {
                continue;
            }
            for crossing in first_curve.crossings(second_curve) {
                first_cuts.push(crossing.parameter_on_first);
                second_cuts.push(crossing.parameter_on_second);
            }
        }
    }
    cut_under_foreign_endpoints(curves, &mut cuts);
    curves
        .iter()
        .zip(cuts)
        .map(|(curve, parameters)| curve.split_at(&parameters))
        .collect()
}

/// Cut every curve wherever another curve's END lands on it, which a crossing solve will not do
/// reliably at a parameter boundary.
///
/// Only the curve being LANDED ON is cut. The other one already has a vertex there — that is what
/// being its endpoint means — and [`PlanarCurve::split_at`] discards a cut at the end of an open
/// curve anyway.
///
/// Closed curves are skipped as askers. A circle's `start` is its seam, an arbitrary place with no
/// meaning to the drawing, and cutting a neighbour under it would seat a vertex the author cannot
/// see or account for. A closed curve is still cut BY other curves' ends, which is the direction
/// that carries meaning.
///
/// The dedup is not tidiness. On the frames where the crossing solve does find the root, this pass
/// finds the same place again about a nanometre away, and `split_at`'s own slack is far too fine
/// to merge them — so the curve would come back with a sliver piece between two vertices that the
/// face walk then welds into one, leaving a self-loop where there should be nothing at all. An
/// existing cut within welding distance ALONG the curve already says what this one would say.
fn cut_under_foreign_endpoints(curves: &[PlanarCurve], cuts: &mut [Vec<f64>]) {
    let boxes: Vec<([f64; 2], [f64; 2])> =
        curves.iter().copied().map(planar_curve_bounds).collect();
    for (index, curve) in curves.iter().enumerate() {
        if curve.is_closed() {
            continue;
        }
        for endpoint in [curve.start(), curve.end()] {
            for (landed_on, ((other, found), reach)) in curves
                .iter()
                .zip(cuts.iter_mut())
                .zip(boxes.iter().copied())
                .enumerate()
            {
                if landed_on == index {
                    continue;
                }
                if endpoint[0] < reach.0[0] - VERTEX_WELD_EPSILON
                    || endpoint[0] > reach.1[0] + VERTEX_WELD_EPSILON
                    || endpoint[1] < reach.0[1] - VERTEX_WELD_EPSILON
                    || endpoint[1] > reach.1[1] + VERTEX_WELD_EPSILON
                {
                    continue;
                }
                let parameter = other.nearest_parameter(endpoint);
                if squared_distance(other.point_at(parameter), endpoint)
                    > VERTEX_WELD_EPSILON * VERTEX_WELD_EPSILON
                {
                    continue;
                }
                let length = other.length();
                let already = found.iter().any(|had| {
                    let gap = (had - parameter).abs();
                    let gap = if other.is_closed() {
                        gap.min(1.0 - gap)
                    } else {
                        gap
                    };
                    gap * length <= VERTEX_WELD_EPSILON
                });
                if !already {
                    found.push(parameter);
                }
            }
        }
    }
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

/// One meeting between a curve's unbounded support and another finite curve.
///
/// `parameter_on_support` may be outside `[0, 1]` for a supporting line. For a supporting circle
/// it uses a full counter-clockwise turn from the positive x axis, independent of the authored
/// arc's start and direction. `parameter_on_finite` always lies on the other curve as authored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurveSupportCrossing {
    pub point: [f64; 2],
    pub parameter_on_support: f64,
    pub parameter_on_finite: f64,
    pub overlapping: bool,
}

impl CurveSupportCrossing {
    const fn from_curve_crossing(crossing: CurveCrossing) -> Self {
        Self {
            point: crossing.point,
            parameter_on_support: crossing.parameter_on_first,
            parameter_on_finite: crossing.parameter_on_second,
            overlapping: crossing.overlapping,
        }
    }
}

impl CurveCrossing {
    /// A transverse crossing — the ordinary case.
    const fn transverse(
        point: [f64; 2],
        parameter_on_first: f64,
        parameter_on_second: f64,
    ) -> Self {
        Self {
            point,
            parameter_on_first,
            parameter_on_second,
            overlapping: false,
        }
    }

    /// One end of a coincident stretch.
    const fn shared(point: [f64; 2], parameter_on_first: f64, parameter_on_second: f64) -> Self {
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
    a[1].mul_add(-b[0], a[0] * b[1])
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[1].mul_add(b[1], a[0] * b[0])
}

fn length(a: [f64; 2]) -> f64 {
    a[0].hypot(a[1])
}

fn squared_distance(first: [f64; 2], second: [f64; 2]) -> f64 {
    let delta = [first[0] - second[0], first[1] - second[1]];
    delta[0].mul_add(delta[0], delta[1] * delta[1])
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
        let point = [
            first[0].mul_add(on_first, a0[0]),
            first[1].mul_add(on_first, a0[1]),
        ];
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
            first[0].mul_add(overlap_low, a0[0]),
            first[1].mul_add(overlap_low, a0[1]),
        ],
        overlap_low,
        to_second(overlap_low),
    )];
    if (overlap_high - overlap_low) * first_length > CROSSING_EPSILON {
        ends.push(CurveCrossing::shared(
            [
                first[0].mul_add(overlap_high, a0[0]),
                first[1].mul_add(overlap_high, a0[1]),
            ],
            overlap_high,
            to_second(overlap_high),
        ));
    }
    ends
}

/// Infinite line against a finite segment. Coincidence reports the finite segment's two ends as
/// an overlapping span; callers that require a unique meeting can discard those entries.
fn line_meets_segment(
    line_start: [f64; 2],
    line_end: [f64; 2],
    segment_start: [f64; 2],
    segment_end: [f64; 2],
) -> Vec<CurveSupportCrossing> {
    let line = [line_end[0] - line_start[0], line_end[1] - line_start[1]];
    let segment = [
        segment_end[0] - segment_start[0],
        segment_end[1] - segment_start[1],
    ];
    let offset = [
        segment_start[0] - line_start[0],
        segment_start[1] - line_start[1],
    ];
    let (line_length, segment_length) = (length(line), length(segment));
    if line_length <= CROSSING_EPSILON || segment_length <= CROSSING_EPSILON {
        return Vec::new();
    }
    let denominator = cross(line, segment);
    let parallel =
        denominator.abs() <= CROSSING_EPSILON * line_length.max(1.0) * segment_length.max(1.0);
    if !parallel {
        let on_support = cross(offset, segment) / denominator;
        let Some(on_finite) = clamped_parameter(cross(offset, line) / denominator, segment_length)
        else {
            return Vec::new();
        };
        return vec![CurveSupportCrossing {
            point: [
                line[0].mul_add(on_support, line_start[0]),
                line[1].mul_add(on_support, line_start[1]),
            ],
            parameter_on_support: on_support,
            parameter_on_finite: on_finite,
            overlapping: false,
        }];
    }
    if cross(offset, line).abs() > CROSSING_EPSILON * line_length.max(1.0) {
        return Vec::new();
    }
    let line_squared = dot(line, line);
    [
        (segment_start, 0.0, dot(offset, line) / line_squared),
        (
            segment_end,
            1.0,
            dot(
                [
                    segment_end[0] - line_start[0],
                    segment_end[1] - line_start[1],
                ],
                line,
            ) / line_squared,
        ),
    ]
    .into_iter()
    .map(
        |(point, parameter_on_finite, parameter_on_support)| CurveSupportCrossing {
            point,
            parameter_on_support,
            parameter_on_finite,
            overlapping: true,
        },
    )
    .collect()
}

/// Infinite line against a finite circular arc.
fn line_meets_arc(
    line_start: [f64; 2],
    line_end: [f64; 2],
    arc: &PlanarCurve,
) -> Vec<CurveSupportCrossing> {
    let PlanarCurve::Arc {
        center,
        radius,
        start_radians,
        sweep_radians,
    } = *arc
    else {
        return Vec::new();
    };
    let direction = [line_end[0] - line_start[0], line_end[1] - line_start[1]];
    let to_start = [line_start[0] - center[0], line_start[1] - center[1]];
    let quadratic = dot(direction, direction);
    if quadratic <= CROSSING_EPSILON {
        return Vec::new();
    }
    let linear = 2.0 * dot(to_start, direction);
    let constant = radius.mul_add(-radius, dot(to_start, to_start));
    let discriminant = (4.0 * quadratic).mul_add(-constant, linear * linear);
    if discriminant < 0.0 {
        return Vec::new();
    }
    let root = discriminant.max(0.0).sqrt();
    let parameters: &[f64] = if root <= CROSSING_EPSILON * quadratic.max(1.0) {
        &[-linear / (2.0 * quadratic)]
    } else {
        &[
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ]
    };
    parameters
        .iter()
        .filter_map(|on_support| {
            let point = [
                direction[0].mul_add(*on_support, line_start[0]),
                direction[1].mul_add(*on_support, line_start[1]),
            ];
            parameter_on_arc(center, start_radians, sweep_radians, point).map(|on_finite| {
                CurveSupportCrossing {
                    point,
                    parameter_on_support: *on_support,
                    parameter_on_finite: on_finite,
                    overlapping: false,
                }
            })
        })
        .collect()
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
    let constant = radius.mul_add(-radius, dot(to_start, to_start));
    let discriminant = (4.0 * quadratic).mul_add(-constant, linear * linear);
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
            direction[0].mul_add(on_segment, a0[0]),
            direction[1].mul_add(on_segment, a0[1]),
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
    let along_numerator =
        radius_b.mul_add(-radius_b, radius_a.mul_add(radius_a, distance * distance));
    let along = along_numerator / (2.0 * distance);
    let off = radius_a.mul_add(radius_a, -(along * along)).max(0.0).sqrt();
    let unit = [between[0] / distance, between[1] / distance];
    let base = [
        unit[0].mul_add(along, center_a[0]),
        unit[1].mul_add(along, center_a[1]),
    ];
    let normal = [-unit[1], unit[0]];
    // Tangent circles have one crossing, not two at the same place.
    let candidates: Vec<[f64; 2]> = if off <= CROSSING_EPSILON {
        vec![base]
    } else {
        vec![
            [
                normal[0].mul_add(off, base[0]),
                normal[1].mul_add(off, base[1]),
            ],
            [
                normal[0].mul_add(-off, base[0]),
                normal[1].mul_add(-off, base[1]),
            ],
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
                radius.mul_add(bearing.cos(), center[0]),
                radius.mul_add(bearing.sin(), center[1]),
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
#[allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::imprecise_flops,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::suboptimal_flops,
    clippy::unwrap_used
)]
mod tests {
    /// **A T-junction still cuts its crossbar when the stem lands a hair OFF it.**
    ///
    /// The two passes in [`cut_at_crossings`] skip a pair whose bounding boxes cannot reach each
    /// other, which is what keeps the arrangement from asking every curve about every other. An
    /// axis-aligned crossbar is the case that makes the pad load-bearing rather than decorative:
    /// its box has ZERO height, so a stem whose endpoint is a hair below the line — which is what
    /// arithmetic hands you, not a contrived input — sits outside the raw box and gets skipped, and
    /// the crossbar comes back uncut with the vertex left sitting in the middle of a piece.
    ///
    /// **Seen red**: with the `VERTEX_WELD_EPSILON` terms dropped from the endpoint pass's reject,
    /// this returns one piece instead of two.
    #[test]
    fn a_stem_landing_a_hair_off_an_axis_aligned_crossbar_still_cuts_it() {
        // Inside the weld tolerance, outside the crossbar's own zero-height box.
        let a_hair = VERTEX_WELD_EPSILON / 4.0;
        let crossbar = PlanarCurve::Segment {
            start: [0.0, 0.0],
            end: [10.0, 0.0],
        };
        // STOPPING SHORT of the crossbar, never reaching it. A stem that crossed would be cut by
        // the crossings pass and this would say nothing about the endpoint pass at all — which is
        // what the first draft of this test did, and it stayed green with the pad removed.
        let stem = PlanarCurve::Segment {
            start: [5.0, 4.0],
            end: [5.0, a_hair],
        };
        let cut = cut_at_crossings(&[crossbar, stem]);
        assert_eq!(
            cut[0].len(),
            2,
            "the crossbar should be cut under the stem's end, not left whole"
        );
    }

    /// **Two curves far apart are not asked about each other, and two that touch still are.**
    ///
    /// The reject is only allowed to drop pairs that CANNOT meet. This states both halves at once,
    /// because a filter that drops everything passes the first half on its own.
    #[test]
    fn the_reject_drops_only_pairs_that_cannot_reach_each_other() {
        let here = PlanarCurve::Segment {
            start: [0.0, 0.0],
            end: [1.0, 1.0],
        };
        let far = PlanarCurve::Segment {
            start: [500.0, 500.0],
            end: [501.0, 501.0],
        };
        let crossing = PlanarCurve::Segment {
            start: [0.0, 1.0],
            end: [1.0, 0.0],
        };
        assert_eq!(
            cut_at_crossings(&[here, far])[0].len(),
            1,
            "nothing reaches this curve, so it keeps its whole span"
        );
        assert_eq!(
            cut_at_crossings(&[here, crossing])[0].len(),
            2,
            "a curve crossing it mid-span must still cut it"
        );
        // Externally TANGENT circles are deliberately not asserted here. They come back uncut,
        // and they do so with the reject removed as well — it is how the arrangement has always
        // behaved, not something a bounding box introduced. Whether a tangency should seat a vertex
        // is a question about the arrangement and not about this filter.
    }

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

    /// A curve that ENDS on another one cuts it there, whichever side the rounding falls on.
    ///
    /// A chord of the circle, with the touching end a hair inside, exactly on, and a hair outside
    /// — the band a solved coincidence actually lands in. The three are not equivalent to a
    /// crossing solve: from just outside the chord enters the disc and the root is interior to the
    /// segment, from just inside the whole segment is interior and there is no root to find at
    /// all. Before the endpoint pass those answered differently, and which one a drag produced was
    /// decided by the last digit of the solver's residual.
    ///
    /// A closed curve cut once stays closed — `split_at` re-seams it rather than opening it — so
    /// what says it was cut is WHERE the seam now is. Under the endpoint, not at bearing zero.
    #[test]
    fn a_curve_ending_on_another_cuts_it_whichever_side_the_rounding_falls_on() {
        for reach in [3.999_999_99, 4.0, 4.000_000_01] {
            let curves = [
                PlanarCurve::circle([0.0, 0.0], 4.0),
                segment([0.0, reach], [0.0, -3.0]),
            ];
            let circle = cut_at_crossings(&curves).swap_remove(0);
            assert_eq!(
                circle.len(),
                1,
                "reach {reach} cut the circle more than once"
            );
            let seam = circle[0].start();
            assert!(
                (seam[0] - 0.0).abs() < 1.0e-6 && (seam[1] - 4.0).abs() < 1.0e-6,
                "reach {reach} left the circle seamed at {seam:?}, not under the end on it"
            );
        }
    }

    /// The endpoint pass does not cut a second time where the crossing solve already cut.
    ///
    /// From just outside, the chord enters the disc and the solve finds a genuine crossing a
    /// nanometre along it; the endpoint pass then finds the same place independently. The two cuts
    /// land on the circle about a nanometre apart, which is far too close for `split_at` to merge
    /// and far too far for it to ignore — so without the pass's own dedup the circle comes back in
    /// two pieces, one of them a sliver whose ends the face walk welds into a single vertex: a
    /// self-loop bounding nothing, handed on as though it were geometry.
    ///
    /// The chord leaves at an ANGLE to the radius on purpose. Aimed straight at the centre the two
    /// cuts coincide to the last bit — the crossing and the foot of the perpendicular are the same
    /// place — and `split_at` merges them on its own, so a radial witness would report the dedup
    /// working when it had done nothing.
    #[test]
    fn an_end_the_crossing_solve_already_found_is_not_cut_again() {
        let curves = [
            PlanarCurve::circle([0.0, 0.0], 4.0),
            segment([0.0, 4.000_000_01], [2.0, -3.0]),
        ];
        let circle = cut_at_crossings(&curves).swap_remove(0);
        assert_eq!(
            circle.len(),
            1,
            "the circle was cut twice for one place two curves meet"
        );
    }

    #[test]
    fn nearest_parameter_projects_segments_and_respects_arc_domains() {
        assert!(
            (segment([0.0, 0.0], [10.0, 0.0]).nearest_parameter([4.0, 3.0]) - 0.4).abs() < 1.0e-12
        );
        assert_eq!(
            segment([0.0, 0.0], [10.0, 0.0]).nearest_parameter([-2.0, 1.0]),
            0.0
        );
        let quarter = arc([0.0, 0.0], 5.0, 0.0, 90.0);
        assert!((quarter.nearest_parameter([4.0, 4.0]) - 0.5).abs() < 1.0e-12);
        assert_eq!(quarter.nearest_parameter([-5.0, 0.0]), 1.0);
        let circle = PlanarCurve::circle([0.0, 0.0], 5.0);
        assert!((circle.nearest_parameter([0.0, -8.0]) - 0.75).abs() < 1.0e-12);
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

    #[test]
    fn a_supporting_line_reaches_only_the_other_curve_as_authored() {
        let source = segment([0.0, 0.0], [2.0, 0.0]);
        let crossing = segment([5.0, -1.0], [5.0, 1.0]);
        let found = source.support_crossings_with(&crossing);
        assert_eq!(found.len(), 1);
        assert_near(found[0].point, [5.0, 0.0]);
        assert!((found[0].parameter_on_support - 2.5).abs() < 1.0e-12);
        assert!((found[0].parameter_on_finite - 0.5).abs() < 1.0e-12);

        let finite_miss = segment([5.0, 2.0], [5.0, 4.0]);
        assert!(source.support_crossings_with(&finite_miss).is_empty());
    }

    #[test]
    fn an_arc_support_is_its_whole_circle_but_the_target_stays_finite() {
        let source = arc([0.0, 0.0], 5.0, 0.0, 90.0);
        let target = segment([-6.0, 0.0], [-4.0, 0.0]);
        let found = source.support_crossings_with(&target);
        assert_eq!(found.len(), 1);
        assert_near(found[0].point, [-5.0, 0.0]);
        assert!((found[0].parameter_on_support - 0.5).abs() < 1.0e-12);
        assert!((found[0].parameter_on_finite - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn collinear_support_contacts_are_explicitly_overlapping() {
        let source = segment([0.0, 0.0], [2.0, 0.0]);
        let target = segment([5.0, 0.0], [7.0, 0.0]);
        let found = source.support_crossings_with(&target);
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|crossing| crossing.overlapping));
        assert_near(found[0].point, [5.0, 0.0]);
        assert_near(found[1].point, [7.0, 0.0]);
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

    #[test]
    fn a_rational_bezier_crosses_a_segment_at_curve_parameters() {
        let bezier = PlanarCurve::RationalBezier(RationalBezier::cubic([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]));
        let vertical = segment([1.5, -1.0], [1.5, 1.0]);
        let crossings = bezier.crossings(&vertical);
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [1.5, 0.0]);
        assert!((crossings[0].parameter_on_first - 0.5).abs() <= 1.0e-7);
        assert!((crossings[0].parameter_on_second - 0.5).abs() <= 1.0e-7);
    }

    #[test]
    fn a_curved_bezier_crosses_a_segment_at_non_dyadic_segment_parameter() {
        let bezier = PlanarCurve::RationalBezier(RationalBezier::cubic([
            [0.0, 0.0],
            [3.0, 5.0],
            [7.0, 5.0],
            [10.0, 0.0],
        ]));
        let vertical = segment([5.0, -1.0], [5.0, 6.0]);
        let crossings = bezier.crossings(&vertical);
        assert_eq!(crossings.len(), 1);
        assert_near(crossings[0].point, [5.0, 3.75]);
        assert!((crossings[0].parameter_on_first - 0.5).abs() <= 1.0e-7);
    }

    #[test]
    fn identical_and_reversed_beziers_report_shared_spans() {
        let curve = RationalBezier::cubic([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
        let same =
            PlanarCurve::RationalBezier(curve).crossings(&PlanarCurve::RationalBezier(curve));
        assert_eq!(same.len(), 2);
        assert!(same.iter().all(|crossing| crossing.overlapping));

        let reversed = PlanarCurve::RationalBezier(curve)
            .crossings(&PlanarCurve::RationalBezier(curve.reversed()));
        assert_eq!(reversed.len(), 2);
        assert_eq!(reversed[0].parameter_on_second, 1.0);
        assert_eq!(reversed[1].parameter_on_second, 0.0);
    }
}
