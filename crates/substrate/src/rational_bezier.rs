//! Cubic rational Bézier curves in the plane.
//!
//! A single representation covers ordinary cubic splines (all weights are one), exact conic
//! sections (a rational quadratic elevated to degree three), and exact ellipse quarters. Keeping
//! evaluation, subdivision, derivatives, curvature, and adaptive flattening here prevents every
//! sketch/document/render adapter from growing a slightly different curve implementation.
//!
//! Control points are Euclidean for authoring ergonomics, but every algorithm first lifts them to
//! homogeneous coordinates `(w*x, w*y, w)`. De Casteljau subdivision is then ordinary affine
//! interpolation in that space; converting the result back preserves the rational curve exactly.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::indexing_slicing,
    clippy::many_single_char_names
)]

/// One degree-three rational Bézier curve over parameter `t ∈ [0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RationalBezier {
    /// Euclidean control points, beginning and ending on the curve.
    pub control: [[f64; 2]; 4],
    /// Positive homogeneous weights paired with [`control`](Self::control).
    pub weights: [f64; 4],
}

impl RationalBezier {
    /// An ordinary polynomial cubic Bézier.
    #[must_use]
    pub const fn cubic(control: [[f64; 2]; 4]) -> Self {
        Self {
            control,
            weights: [1.0; 4],
        }
    }

    /// Elevate a rational quadratic to degree three without changing its curve.
    ///
    /// `middle_weight = cos(theta/2)` with the tangent-intersection point in the middle produces
    /// an exact circular or elliptic arc spanning `theta` (for `|theta| < π`).
    #[must_use]
    pub fn elevated_quadratic(control: [[f64; 2]; 3], weights: [f64; 3]) -> Self {
        let q = [
            homogeneous(control[0], weights[0]),
            homogeneous(control[1], weights[1]),
            homogeneous(control[2], weights[2]),
        ];
        let cubic = [
            q[0],
            mix3(q[0], q[1], 2.0 / 3.0),
            mix3(q[1], q[2], 1.0 / 3.0),
            q[2],
        ];
        from_homogeneous(cubic)
    }

    /// Whether every coordinate and weight is finite and every weight is strictly positive.
    /// Positive weights keep the curve inside the control polygon and make its denominator
    /// non-zero throughout the authored domain.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.control
            .iter()
            .flatten()
            .chain(self.weights.iter())
            .all(|value| value.is_finite())
            && self.weights.iter().all(|weight| *weight > 0.0)
    }

    /// Evaluate the rational curve. A malformed denominator returns `[NaN; 2]` rather than
    /// silently selecting an arbitrary projective point.
    #[must_use]
    pub fn point_at(&self, parameter: f64) -> [f64; 2] {
        let [x, y, w] = evaluate_homogeneous(self.homogeneous_control(), parameter);
        if w.abs() <= f64::EPSILON {
            return [f64::NAN; 2];
        }
        [x / w, y / w]
    }

    /// First derivative with respect to the normalized curve parameter.
    #[must_use]
    pub fn derivative_at(&self, parameter: f64) -> [f64; 2] {
        self.derivatives_at(parameter).0
    }

    /// Signed planar curvature. Positive is a left turn, negative a right turn, and a stationary
    /// or malformed point reports `NaN` because it has no defined tangent frame.
    #[must_use]
    pub fn curvature_at(&self, parameter: f64) -> f64 {
        let (first, second) = self.derivatives_at(parameter);
        let speed_squared = first[0].mul_add(first[0], first[1] * first[1]);
        if speed_squared <= f64::EPSILON || !speed_squared.is_finite() {
            return f64::NAN;
        }
        (first[0].mul_add(second[1], -(first[1] * second[0])))
            / (speed_squared * speed_squared.sqrt())
    }

    /// The same curve with its parameter direction reversed.
    #[must_use]
    pub const fn reversed(&self) -> Self {
        Self {
            control: [
                self.control[3],
                self.control[2],
                self.control[1],
                self.control[0],
            ],
            weights: [
                self.weights[3],
                self.weights[2],
                self.weights[1],
                self.weights[0],
            ],
        }
    }

    /// The exact sub-curve over `from..to`. Inputs are clamped into `[0, 1]`; reversed bounds
    /// return the corresponding reversed curve.
    #[must_use]
    pub fn sub_curve(&self, from: f64, to: f64) -> Self {
        let from = from.clamp(0.0, 1.0);
        let to = to.clamp(0.0, 1.0);
        if to < from {
            return self.sub_curve(to, from).reversed();
        }
        if from <= f64::EPSILON && to >= 1.0 - f64::EPSILON {
            return *self;
        }
        let homogeneous = self.homogeneous_control();
        let (_, right) = split_homogeneous(homogeneous, from);
        let remaining = 1.0 - from;
        let local_to = if remaining <= f64::EPSILON {
            0.0
        } else {
            (to - from) / remaining
        };
        let (span, _) = split_homogeneous(right, local_to);
        from_homogeneous(span)
    }

    /// Conservative axis-aligned bounds. Positive rational weights place the entire curve in its
    /// Euclidean control hull, so these bounds may be wider than the curve but never clip it.
    #[must_use]
    pub fn control_bounds(&self) -> ([f64; 2], [f64; 2]) {
        let mut low = self.control[0];
        let mut high = self.control[0];
        for point in self.control.iter().skip(1) {
            low = [low[0].min(point[0]), low[1].min(point[1])];
            high = [high[0].max(point[0]), high[1].max(point[1])];
        }
        (low, high)
    }

    /// Adaptively flatten the curve so every accepted control polygon stays within `tolerance`
    /// of its chord. The first and last outputs are the exact stored endpoints.
    #[must_use]
    pub fn flatten(&self, tolerance: f64) -> Vec<[f64; 2]> {
        let tolerance = tolerance.max(f64::EPSILON);
        let mut points = vec![self.control[0]];
        flatten_recursive(*self, tolerance, 0, &mut points);
        points
    }

    fn homogeneous_control(&self) -> [[f64; 3]; 4] {
        [
            homogeneous(self.control[0], self.weights[0]),
            homogeneous(self.control[1], self.weights[1]),
            homogeneous(self.control[2], self.weights[2]),
            homogeneous(self.control[3], self.weights[3]),
        ]
    }

    fn derivatives_at(&self, parameter: f64) -> ([f64; 2], [f64; 2]) {
        let q = self.homogeneous_control();
        let h = evaluate_homogeneous(q, parameter);
        let first_control = [
            scale3(sub3(q[1], q[0]), 3.0),
            scale3(sub3(q[2], q[1]), 3.0),
            scale3(sub3(q[3], q[2]), 3.0),
        ];
        let second_control = [
            scale3(add3(sub3(q[2], scale3(q[1], 2.0)), q[0]), 6.0),
            scale3(add3(sub3(q[3], scale3(q[2], 2.0)), q[1]), 6.0),
        ];
        let h1 = evaluate_quadratic(first_control, parameter);
        let h2 = mix3(second_control[0], second_control[1], parameter);
        let w = h[2];
        if w.abs() <= f64::EPSILON {
            return ([f64::NAN; 2], [f64::NAN; 2]);
        }
        let point = [h[0] / w, h[1] / w];
        let first = [
            point[0].mul_add(-h1[2], h1[0]) / w,
            point[1].mul_add(-h1[2], h1[1]) / w,
        ];
        let second = [
            (2.0 * h1[2]).mul_add(-first[0], point[0].mul_add(-h2[2], h2[0])) / w,
            (2.0 * h1[2]).mul_add(-first[1], point[1].mul_add(-h2[2], h2[1])) / w,
        ];
        (first, second)
    }
}

fn flatten_recursive(curve: RationalBezier, tolerance: f64, depth: u8, points: &mut Vec<[f64; 2]>) {
    const MAX_DEPTH: u8 = 24;
    if depth >= MAX_DEPTH || curve.flatness() <= tolerance {
        points.push(curve.control[3]);
        return;
    }
    let left = curve.sub_curve(0.0, 0.5);
    let right = curve.sub_curve(0.5, 1.0);
    flatten_recursive(left, tolerance, depth + 1, points);
    flatten_recursive(right, tolerance, depth + 1, points);
}

impl RationalBezier {
    fn flatness(&self) -> f64 {
        let start = self.control[0];
        let end = self.control[3];
        distance_to_line(self.control[1], start, end).max(distance_to_line(
            self.control[2],
            start,
            end,
        ))
    }
}

fn homogeneous(point: [f64; 2], weight: f64) -> [f64; 3] {
    [point[0] * weight, point[1] * weight, weight]
}

fn from_homogeneous(control: [[f64; 3]; 4]) -> RationalBezier {
    let weights = [control[0][2], control[1][2], control[2][2], control[3][2]];
    RationalBezier {
        control: [
            [control[0][0] / weights[0], control[0][1] / weights[0]],
            [control[1][0] / weights[1], control[1][1] / weights[1]],
            [control[2][0] / weights[2], control[2][1] / weights[2]],
            [control[3][0] / weights[3], control[3][1] / weights[3]],
        ],
        weights,
    }
}

fn evaluate_homogeneous(control: [[f64; 3]; 4], t: f64) -> [f64; 3] {
    let a = mix3(control[0], control[1], t);
    let b = mix3(control[1], control[2], t);
    let c = mix3(control[2], control[3], t);
    let d = mix3(a, b, t);
    let e = mix3(b, c, t);
    mix3(d, e, t)
}

fn evaluate_quadratic(control: [[f64; 3]; 3], t: f64) -> [f64; 3] {
    mix3(
        mix3(control[0], control[1], t),
        mix3(control[1], control[2], t),
        t,
    )
}

fn split_homogeneous(control: [[f64; 3]; 4], t: f64) -> ([[f64; 3]; 4], [[f64; 3]; 4]) {
    let a = mix3(control[0], control[1], t);
    let b = mix3(control[1], control[2], t);
    let c = mix3(control[2], control[3], t);
    let d = mix3(a, b, t);
    let e = mix3(b, c, t);
    let point = mix3(d, e, t);
    ([control[0], a, d, point], [point, e, c, control[3]])
}

fn mix3(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        (b[0] - a[0]).mul_add(t, a[0]),
        (b[1] - a[1]).mul_add(t, a[1]),
        (b[2] - a[2]).mul_add(t, a[2]),
    ]
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(value: [f64; 3], scale: f64) -> [f64; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

fn distance_to_line(point: [f64; 2], start: [f64; 2], end: [f64; 2]) -> f64 {
    let span = [end[0] - start[0], end[1] - start[1]];
    let length = span[0].hypot(span[1]);
    if length <= f64::EPSILON {
        return (point[0] - start[0]).hypot(point[1] - start[1]);
    }
    (point[1] - start[1])
        .mul_add(-span[0], (point[0] - start[0]) * span[1])
        .abs()
        / length
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: [f64; 2], expected: [f64; 2]) {
        assert!((actual[0] - expected[0]).abs() <= 1.0e-10, "x: {actual:?}");
        assert!((actual[1] - expected[1]).abs() <= 1.0e-10, "y: {actual:?}");
    }

    #[test]
    fn polynomial_line_evaluates_subdivides_and_reverses_exactly() {
        let line = RationalBezier::cubic([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
        close(line.point_at(0.25), [0.75, 0.0]);
        let middle = line.sub_curve(0.25, 0.75);
        close(middle.control[0], [0.75, 0.0]);
        close(middle.control[3], [2.25, 0.0]);
        close(middle.reversed().point_at(0.0), [2.25, 0.0]);
    }

    #[test]
    fn elevated_rational_quadratic_traces_an_exact_quarter_circle() {
        let quarter = RationalBezier::elevated_quadratic(
            [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [1.0, std::f64::consts::FRAC_1_SQRT_2, 1.0],
        );
        let middle = quarter.point_at(0.5);
        close(middle, [std::f64::consts::FRAC_1_SQRT_2; 2]);
        assert!((quarter.curvature_at(0.5) - 1.0).abs() <= 1.0e-10);
    }

    #[test]
    fn adaptive_flatten_keeps_exact_endpoints_and_refines_a_bend() {
        let curve = RationalBezier::cubic([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]]);
        let coarse = curve.flatten(0.5);
        let fine = curve.flatten(0.01);
        assert_eq!(coarse.first().copied(), Some([0.0, 0.0]));
        assert_eq!(fine.last().copied(), Some([2.0, 0.0]));
        assert!(fine.len() > coarse.len());
    }
}
