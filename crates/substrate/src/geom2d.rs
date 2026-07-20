//! # geom2d — planar computational geometry: exact predicates and measured fields
//!
//! A small kernel of planar geometry over points in the plane, in some caller-chosen
//! coordinate space. It is pure: no domain type appears, and the polygon is any slice
//! of points. The domain adapter (a sketch profile, a brush stroke) converts its own
//! vertices to points and calls in.
//!
//! ## The module is deliberately split across two floating-point widths
//!
//! **This is not an oversight, and the two halves must not be tidied back together.**
//!
//! The **measurement** half ([`Metric`], [`distance_point_to_segment`],
//! [`signed_distance_to_polygon`], [`point_in_polygon`]) is `f32`. It is the half a GPU
//! preview mirrors in WGSL, and **WGSL has no `f64`**. Any `f64` left in the mirrored
//! path is a CPU/GPU divergence that no amount of parity testing can remove, because the
//! shader cannot reproduce the wider arithmetic even in principle — so the widths are
//! matched here, at the source, and parity becomes structural rather than tested.
//! Narrowing costs nothing at the scale this runs at: measured against `f64` over 26.7M
//! samples of realistic sketch geometry, `f32` produced zero occupancy disagreements out
//! to roughly `2.6e5` voxels of coordinate offset, and the only differences found were
//! *repairs* — half-integer lattice sites landing exactly on a closing edge, where `f64`
//! returns a few ulps of positive noise and drops a voxel that `f32` returns as exactly
//! zero and correctly keeps.
//!
//! The **predicate** half ([`orient2d`], [`segments_intersect`],
//! [`segment_intersects_rect`], [`rectangle_inside_polygon`]) stays `f64`. It is CPU-only
//! and will never be mirrored: it answers "is this whole cell inside?", which is the
//! coarse-solid classifier question (ADR 0010), and a shader never asks it — a raymarch
//! asks about points, not cells. It is also where the extra width genuinely earns its
//! keep. Checked against exact `i128` arithmetic, `f32` starts returning **wrong
//! orientation signs from about ±4096 voxels outward** (0 wrong at 2¹², 453 at 2²⁰, 8,817
//! at 2²⁴), while `f64` stays exact past 2³⁰. A wrong sign here does not merely blur a
//! surface: it makes the classifier **over-claim solid**, which is unsound rather than
//! conservative, and a whole cell is then filled without ever being sampled.
//!
//! This is the line ADR 0019 draws: **predicates classify, fields measure.** A predicate
//! must be exact or it lies; a measurement only has to be accurate, and accuracy is
//! cheaper than exactness.
//!
//! ## The primitives
//!
//! - [`orient2d`] — the signed area of the triangle `(a, b, c)`, i.e. the 2D
//!   cross product `(b − a) × (c − a)`. Positive ⇒ `c` is left of the directed
//!   line `a → b` (counter-clockwise turn); negative ⇒ right; zero ⇒ collinear.
//!   This is the atomic orientation test the others build on (Shewchuk 1997,
//!   *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
//!   Predicates*; O'Rourke, *Computational Geometry in C* 1998, the `Area2` /
//!   `Left` predicate). This implementation is the plain non-adaptive
//!   determinant — exact when the inputs are integers-as-`f64` (our sketch
//!   vertices), which is the regime it runs in.
//! - [`segments_intersect`] — whether two closed segments meet, proper crossings
//!   and collinear/endpoint touches alike, decided by the four orientation signs
//!   with a collinear bounding-box fallback (CLRS 3rd ed. §33.1, `SEGMENTS-
//!   INTERSECT` / `ON-SEGMENT`).
//! - [`segment_intersects_rect`] — whether a segment meets a closed axis-aligned
//!   rectangle: an endpoint inside, or a crossing of one of the four edges —
//!   complete for a convex box (Ericson, *Real-Time Collision Detection* 2005).
//! - [`point_in_polygon`] — the crossing-number (ray-crossing) point-in-polygon
//!   test: cast a ray in the `+axis1` direction and count edge crossings; odd ⇒
//!   inside (Franklin's PNPOLY; Shimrat 1962; Preparata & Shamos 1985; Ericson
//!   2005). The polygon is implicitly closed (last vertex → first).
//! - [`rectangle_inside_polygon`] — whether a closed axis-aligned rectangle lies
//!   wholly inside a polygon. Exact by connectedness: if no polygon edge crosses
//!   the rectangle it holds no piece of the boundary, so it is wholly in or out,
//!   and one interior sample (the centre) decides. Conservative on a grazing
//!   edge (counts as crossing ⇒ not-inside, still exact).
//!
//! ## Predicates and measurements
//!
//! [`orient2d`], [`segments_intersect`], [`segment_intersects_rect`] and
//! [`rectangle_inside_polygon`] are **predicates**: they answer yes/no, and they are
//! exact (`f64`, see above). [`point_in_polygon`], [`distance_point_to_segment`] and
//! [`signed_distance_to_polygon`] are **measurements**: they answer how-far — or, for
//! `point_in_polygon`, supply the *sign* of a how-far — in floating point, and cannot be
//! exact in the same sense (`f32`, mirrored in WGSL). The two coexist deliberately — a
//! predicate classifies a region, a measurement gives it a geometry to be offset or
//! displaced — and neither replaces the other. Measurements are taken in a
//! caller-chosen [`Metric`]:
//! `Euclidean` grows a shape by a disc and rounds its corners, `Chebyshev` grows it
//! by a square and keeps them sharp, which is the natural choice on a lattice.

/// The signed area of triangle `(a, b, c)` — twice the area, the determinant
/// `(b − a) × (c − a)`. Positive ⇒ counter-clockwise (`c` left of `a → b`),
/// negative ⇒ clockwise, zero ⇒ collinear. See the module docs for the
/// literature (Shewchuk 1997; O'Rourke 1998).
#[inline]
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

#[inline]
fn orientation_sign(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i32 {
    let value = orient2d(a, b, c);
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

/// Whether the two closed segments `p0→p1` and `q0→q1` intersect — proper
/// crossings AND collinear / endpoint touches — via the four orientation signs
/// with a collinear bounding-box (`on-segment`) fallback. CLRS 3rd ed. §33.1.
pub fn segments_intersect(p0: [f64; 2], p1: [f64; 2], q0: [f64; 2], q1: [f64; 2]) -> bool {
    // `c` (collinear with `a→b`) lies within `a→b`'s bounding box.
    let on_segment = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| -> bool {
        c[0] >= a[0].min(b[0])
            && c[0] <= a[0].max(b[0])
            && c[1] >= a[1].min(b[1])
            && c[1] <= a[1].max(b[1])
    };
    let d1 = orientation_sign(q0, q1, p0);
    let d2 = orientation_sign(q0, q1, p1);
    let d3 = orientation_sign(p0, p1, q0);
    let d4 = orientation_sign(p0, p1, q1);
    if d1 != d2 && d3 != d4 {
        return true;
    }
    (d1 == 0 && on_segment(q0, q1, p0))
        || (d2 == 0 && on_segment(q0, q1, p1))
        || (d3 == 0 && on_segment(p0, p1, q0))
        || (d4 == 0 && on_segment(p0, p1, q1))
}

/// Whether segment `a→b` intersects the CLOSED axis-aligned rectangle
/// `[rect_min, rect_max]` (component-wise `min <= max`). True iff an endpoint is
/// inside the rectangle OR the segment crosses one of the four rectangle edges —
/// complete for a convex box (Ericson 2005).
pub fn segment_intersects_rect(
    a: [f64; 2],
    b: [f64; 2],
    rect_min: [f64; 2],
    rect_max: [f64; 2],
) -> bool {
    let inside = |p: [f64; 2]| {
        p[0] >= rect_min[0] && p[0] <= rect_max[0] && p[1] >= rect_min[1] && p[1] <= rect_max[1]
    };
    if inside(a) || inside(b) {
        return true;
    }
    let corners = [
        [rect_min[0], rect_min[1]],
        [rect_max[0], rect_min[1]],
        [rect_max[0], rect_max[1]],
        [rect_min[0], rect_max[1]],
    ];
    (0..4).any(|edge| segments_intersect(a, b, corners[edge], corners[(edge + 1) % 4]))
}

/// The crossing-number point-in-polygon test: whether `sample` lies inside the
/// polygon `[[axis0, axis1]; n]` (implicitly closed, last vertex → first). Counts
/// how many edges a ray cast in the `+axis1` direction from the sample crosses;
/// an odd count is inside. Franklin's PNPOLY / ray-crossing (see module docs).
///
/// No on-boundary tie-breaking is done: callers that need exactness (e.g. voxel
/// sample centres at half-integer positions against integer vertices) rely on the
/// sample never lying on an edge.
///
/// `f32`, with the rest of the measurement half: this is the boundary authority a WGSL
/// preview must port, and it supplies the sign for [`signed_distance_to_polygon`]. See
/// the module docs for why the width is part of the contract rather than an accident.
pub fn point_in_polygon(polygon: &[[f32; 2]], sample: [f32; 2]) -> bool {
    let mut inside = false;
    let count = polygon.len();
    if count == 0 {
        return false;
    }
    let mut previous = count - 1;
    for current in 0..count {
        let current_0 = polygon[current][0];
        let current_1 = polygon[current][1];
        let previous_0 = polygon[previous][0];
        let previous_1 = polygon[previous][1];
        // Does a ray in the +axis1 direction from the sample cross this edge?
        let straddles = (current_1 > sample[1]) != (previous_1 > sample[1]);
        if straddles {
            // axis0 of the edge at the sample's axis1 height.
            let crossing_0 = (previous_0 - current_0) * (sample[1] - current_1)
                / (previous_1 - current_1)
                + current_0;
            if sample[0] < crossing_0 {
                inside = !inside;
            }
        }
        previous = current;
    }
    inside
}

/// Whether the CLOSED axis-aligned rectangle `[rect_min, rect_max]` lies ENTIRELY
/// inside the polygon (same space [`point_in_polygon`] samples). Exact by
/// connectedness: the rectangle is inside iff **no polygon edge intersects it AND
/// its centre is inside** (see module docs). A rectangle whose edge grazes a
/// polygon edge counts as crossing ⇒ not-inside (conservative, still exact). A
/// degenerate rectangle (`hi == lo` on an axis: a segment or a point) is handled
/// directly — the edge tests run against the degenerate box and the centre
/// reduces to its midpoint/point. Returns `false` for a polygon with fewer than
/// three vertices or an inverted rectangle.
pub fn rectangle_inside_polygon(
    polygon: &[[f64; 2]],
    rect_min: [f64; 2],
    rect_max: [f64; 2],
) -> bool {
    let count = polygon.len();
    if count < 3 || rect_max[0] < rect_min[0] || rect_max[1] < rect_min[1] {
        return false;
    }
    let mut previous = count - 1;
    for current in 0..count {
        if segment_intersects_rect(polygon[current], polygon[previous], rect_min, rect_max) {
            return false;
        }
        previous = current;
    }
    // The edge tests above are the exactness-critical ones and ran in `f64`: a wrong
    // orientation sign there would let a straddled rectangle through as "inside" and
    // over-claim solid. The centre test is a different question and is answered in `f32`,
    // deliberately:
    //
    // - It is only ever REACHED when no polygon edge meets the rectangle, so the centre
    //   is not near the boundary — the case where width would matter has already been
    //   decided by the exact half.
    // - `point_in_polygon` is the same call the per-voxel resolve makes to decide
    //   occupancy. Answering the centre in `f64` here while the resolve answers it in
    //   `f32` would let the coarse claim and the per-voxel truth disagree on a sample
    //   sitting on an edge — exactly the "same set, different rounding" failure this
    //   classifier is supposed to avoid. Sharing the width makes them agree by
    //   construction.
    let narrowed: Vec<[f32; 2]> = polygon
        .iter()
        .map(|point| [point[0] as f32, point[1] as f32])
        .collect();
    let centre = [
        ((rect_min[0] + rect_max[0]) * 0.5) as f32,
        ((rect_min[1] + rect_max[1]) * 0.5) as f32,
    ];
    point_in_polygon(&narrowed, centre)
}

/// Which notion of distance a measurement is taken in.
///
/// The two agree on what is *inside* a shape and disagree on how far away things are, so a
/// classification may use either while an offset must commit to one: `Euclidean` grows a
/// shape by a disc and rounds its convex corners, `Chebyshev` grows it by a square and keeps
/// them sharp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Straight-line distance, `sqrt(dx² + dy²)`. The L2 norm.
    Euclidean,
    /// Largest-axis distance, `max(|dx|, |dy|)`. The L∞ norm — the natural metric of a
    /// square lattice, where it counts axis-aligned steps rather than diagonal reach.
    Chebyshev,
}

impl Metric {
    /// The length of the vector `delta` under this metric.
    #[inline]
    pub fn length(self, delta: [f32; 2]) -> f32 {
        match self {
            Metric::Euclidean => (delta[0] * delta[0] + delta[1] * delta[1]).sqrt(),
            Metric::Chebyshev => delta[0].abs().max(delta[1].abs()),
        }
    }

    /// The distance between two points under this metric.
    #[inline]
    pub fn distance(self, a: [f32; 2], b: [f32; 2]) -> f32 {
        self.length([b[0] - a[0], b[1] - a[1]])
    }
}

/// Distance from `point` to the closed segment `a → b`, under `metric`. Never negative; zero
/// exactly on the segment.
///
/// **Euclidean** is the textbook projection: clamp the parameter of the perpendicular foot to
/// `[0, 1]` and measure to that point (Ericson, *Real-Time Collision Detection* 2005,
/// §5.1.2).
///
/// **Chebyshev** has no such closed form, but it does have an exact one. Writing the segment
/// as `a + t·(b − a)`, the distance is
///
/// ```text
/// f(t) = max(|gx(t)|, |gy(t)|)      gx(t) = px − ax − t·dx,  gy(t) = py − ay − t·dy
/// ```
///
/// Each `|g|` is convex and piecewise linear in `t`, and the maximum of convex functions is
/// convex — so `f` is convex piecewise linear, and its minimum over `[0, 1]` is attained at a
/// breakpoint or an endpoint. The breakpoints are exactly where a term changes slope: where
/// `gx = 0`, where `gy = 0`, and where the two swap dominance (`gx = ±gy`). Evaluating `f` at
/// those four parameters plus both endpoints is therefore **exact**, not an approximation.
///
/// A degenerate (zero-length) segment reduces to the distance to its single point.
pub fn distance_point_to_segment(a: [f32; 2], b: [f32; 2], point: [f32; 2], metric: Metric) -> f32 {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let offset = [point[0] - a[0], point[1] - a[1]];
    // Degenerate segment: the whole thing is the point `a`.
    if delta[0] == 0.0 && delta[1] == 0.0 {
        return metric.length(offset);
    }
    let at = |t: f32| {
        let t = t.clamp(0.0, 1.0);
        metric.length([offset[0] - t * delta[0], offset[1] - t * delta[1]])
    };
    match metric {
        Metric::Euclidean => {
            let length_squared = delta[0] * delta[0] + delta[1] * delta[1];
            at((offset[0] * delta[0] + offset[1] * delta[1]) / length_squared)
        }
        Metric::Chebyshev => {
            let mut best = at(0.0).min(at(1.0));
            // Slope changes of |gx|, |gy|, and of the max between them.
            let breakpoints = [
                (offset[0], delta[0]),                     // gx = 0
                (offset[1], delta[1]),                     // gy = 0
                (offset[0] - offset[1], delta[0] - delta[1]), // gx = gy
                (offset[0] + offset[1], delta[0] + delta[1]), // gx = -gy
            ];
            for (numerator, denominator) in breakpoints {
                if denominator != 0.0 {
                    best = best.min(at(numerator / denominator));
                }
            }
            best
        }
    }
}

/// Signed distance from `point` to the polygon's boundary under `metric` — **negative
/// inside**, positive outside, zero on the boundary. The polygon is implicitly closed (last
/// vertex → first).
///
/// Magnitude is the distance to the nearest edge; the sign comes from [`point_in_polygon`].
/// The two are decided independently, which is what makes this well behaved on inputs a
/// distance function alone would choke on: the field stays continuous through a
/// **self-intersection**, because the sign can only flip where the distance is zero. A
/// self-intersecting or degenerate profile therefore needs no special handling — it gets the
/// same treatment the even-odd rule already gives it.
///
/// Fewer than two vertices has no boundary to measure, and returns `f32::INFINITY`.
pub fn signed_distance_to_polygon(polygon: &[[f32; 2]], point: [f32; 2], metric: Metric) -> f32 {
    if polygon.len() < 2 {
        return f32::INFINITY;
    }
    let mut nearest = f32::INFINITY;
    let mut previous = polygon[polygon.len() - 1];
    for &current in polygon {
        nearest = nearest.min(distance_point_to_segment(previous, current, point, metric));
        previous = current;
    }
    if point_in_polygon(polygon, point) {
        -nearest
    } else {
        nearest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same 4x4 square in both widths — the predicate half is `f64`, the measurement
    /// half `f32`, so a test that spans them needs both. Both are written from the same
    /// integer literals rather than one being cast from the other, matching how the sketch
    /// producer converts its `i64` profile twice from one source.
    const UNIT_SQUARE: [[f64; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    const UNIT_SQUARE_MEASURED: [[f32; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];

    #[test]
    fn orient2d_sign_matches_turn_direction() {
        // Counter-clockwise triple ⇒ positive.
        assert!(orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]) > 0.0);
        // Clockwise triple ⇒ negative.
        assert!(orient2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]) < 0.0);
        // Collinear ⇒ zero.
        assert_eq!(orient2d([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]), 0.0);
    }

    #[test]
    fn segments_intersect_proper_and_touching() {
        // Proper X crossing.
        assert!(segments_intersect(
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0]
        ));
        // Parallel, disjoint.
        assert!(!segments_intersect(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [2.0, 1.0]
        ));
        // Collinear endpoint touch.
        assert!(segments_intersect(
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0]
        ));
        // T-junction (endpoint on the interior of the other).
        assert!(segments_intersect(
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0]
        ));
    }

    #[test]
    fn segment_rect_endpoint_inside_and_edge_crossing() {
        let (lo, hi) = ([0.0, 0.0], [2.0, 2.0]);
        // Endpoint inside.
        assert!(segment_intersects_rect([1.0, 1.0], [5.0, 5.0], lo, hi));
        // Passes through, both endpoints outside.
        assert!(segment_intersects_rect([-1.0, 1.0], [3.0, 1.0], lo, hi));
        // Entirely outside, no crossing.
        assert!(!segment_intersects_rect([3.0, 3.0], [4.0, 4.0], lo, hi));
    }

    #[test]
    fn point_in_polygon_inside_outside() {
        assert!(point_in_polygon(&UNIT_SQUARE_MEASURED, [2.0, 2.0]));
        assert!(!point_in_polygon(&UNIT_SQUARE_MEASURED, [5.0, 2.0]));
        assert!(!point_in_polygon(&UNIT_SQUARE_MEASURED, [-1.0, 2.0]));
        assert!(!point_in_polygon(&[], [0.0, 0.0]));
    }

    #[test]
    fn point_in_polygon_concave_l_shape() {
        // An L: the reflex notch in the upper-right quadrant is OUTSIDE.
        let l_shape: [[f32; 2]; 6] = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [2.0, 2.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ];
        assert!(point_in_polygon(&l_shape, [1.0, 3.0])); // left arm
        assert!(point_in_polygon(&l_shape, [3.0, 1.0])); // bottom arm
        assert!(!point_in_polygon(&l_shape, [3.0, 3.0])); // notch
    }

    #[test]
    fn rectangle_inside_polygon_containment() {
        // Wholly inside.
        assert!(rectangle_inside_polygon(&UNIT_SQUARE, [1.0, 1.0], [3.0, 3.0]));
        // Pokes out the right edge.
        assert!(!rectangle_inside_polygon(&UNIT_SQUARE, [1.0, 1.0], [5.0, 3.0]));
        // Degenerate (single point) inside.
        assert!(rectangle_inside_polygon(&UNIT_SQUARE, [2.0, 2.0], [2.0, 2.0]));
        // Inverted rectangle rejected.
        assert!(!rectangle_inside_polygon(&UNIT_SQUARE, [3.0, 3.0], [1.0, 1.0]));
    }

    /// The Chebyshev segment distance is derived from a breakpoint argument rather than a
    /// projection formula, so check it against brute force: densely sample the segment and
    /// take the nearest sample. If the claim "a convex piecewise-linear minimum is attained
    /// at a breakpoint" were wrong, the closed form would sit ABOVE the sampled minimum.
    #[test]
    fn chebyshev_segment_distance_matches_brute_force() {
        let segments = [
            ([0.0, 0.0], [4.0, 0.0]),   // axis-aligned
            ([0.0, 0.0], [0.0, 3.0]),   // axis-aligned, other axis
            ([0.0, 0.0], [4.0, 4.0]),   // 45°, where |gx| and |gy| swap
            ([1.0, 5.0], [6.0, 2.0]),   // general slope
            ([-3.0, 2.0], [2.0, -4.0]), // crossing the origin region
            ([2.0, 2.0], [2.0, 2.0]),   // degenerate
        ];
        let probes = [
            [0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [-2.0, 3.0],
            [2.5, -1.5], [7.0, 0.5], [0.5, 7.0], [3.3, 3.7],
        ];
        for (a, b) in segments {
            for point in probes {
                const STEPS: u32 = 20_000;
                let closed_form = distance_point_to_segment(a, b, point, Metric::Chebyshev);
                let mut sampled = f32::INFINITY;
                for step in 0..=STEPS {
                    let t = step as f32 / STEPS as f32;
                    let on_segment = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    sampled = sampled.min(Metric::Chebyshev.distance(on_segment, point));
                }
                // The relationship is ONE-SIDED. The closed form is the exact minimum, so it
                // can never exceed any sample; the sampler, stepping discretely, generally
                // lands just above it. (Here the closed form finds 8/11 exactly where 20k
                // samples get within 3e-5 of it.)
                assert!(
                    closed_form <= sampled + 1e-6,
                    "segment {a:?}→{b:?} point {point:?}: closed form {closed_form} is ABOVE \
                     the sampled minimum {sampled} — the breakpoint set is incomplete"
                );
                // And it must not be spuriously low: the sampler cannot miss the true minimum
                // by more than one step's worth of travel, the field being 1-Lipschitz.
                let step_travel = Metric::Chebyshev.length([b[0] - a[0], b[1] - a[1]])
                    / STEPS as f32;
                assert!(
                    sampled - closed_form <= step_travel + 1e-6,
                    "segment {a:?}→{b:?} point {point:?}: closed form {closed_form} is below \
                     the sampled minimum {sampled} by more than one step ({step_travel})"
                );
            }
        }
    }

    #[test]
    fn polygon_signed_distance_signs_and_values() {
        for metric in [Metric::Euclidean, Metric::Chebyshev] {
            // Centre of the 4×4 square is 2 from every edge in both metrics.
            let centre = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [2.0, 2.0], metric);
            assert!((centre + 2.0).abs() < 1e-9, "{metric:?} centre = {centre}");
            // On the boundary ⇒ zero.
            let edge = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [4.0, 2.0], metric);
            assert!(edge.abs() < 1e-9, "{metric:?} edge = {edge}");
            // Straight out from an edge: 1 away in both metrics.
            let outside = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [5.0, 2.0], metric);
            assert!((outside - 1.0).abs() < 1e-9, "{metric:?} outside = {outside}");
            // Inside is negative, outside positive.
            assert!(signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [1.0, 1.0], metric) < 0.0);
            assert!(signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [9.0, 9.0], metric) > 0.0);
        }
        // Diagonally off a corner is where the metrics part company: the corner (4,4) is
        // (3,3) away, so Euclidean reads 3√2 while Chebyshev reads 3.
        let corner = [7.0, 7.0];
        let euclidean = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, corner, Metric::Euclidean);
        let chebyshev = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, corner, Metric::Chebyshev);
        assert!((euclidean - 18.0f32.sqrt()).abs() < 1e-9, "euclidean = {euclidean}");
        assert!((chebyshev - 3.0).abs() < 1e-9, "chebyshev = {chebyshev}");
    }

    /// The property every cell bound rests on: the field must not change faster than
    /// distance does, **in its own metric**. If this fails, classification built on it is
    /// unsound.
    #[test]
    fn polygon_signed_distance_is_one_lipschitz_in_its_own_metric() {
        // A deliberately awkward profile: reflex corner, a spike, and a self-intersection.
        let profile: [[f32; 2]; 7] = [
            [0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [3.0, 2.0],
            [0.0, 6.0], [4.0, -1.0], [1.0, 4.0],
        ];
        for metric in [Metric::Euclidean, Metric::Chebyshev] {
            let mut worst: f32 = 0.0;
            let mut samples = 0u32;
            for xi in -20..=80i32 {
                for yi in -20..=80i32 {
                    let p = [xi as f32 * 0.1, yi as f32 * 0.1];
                    let here = signed_distance_to_polygon(&profile, p, metric);
                    for delta in [[0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.1, -0.1]] {
                        let q = [p[0] + delta[0], p[1] + delta[1]];
                        let there = signed_distance_to_polygon(&profile, q, metric);
                        let ratio = (there - here).abs() / metric.length(delta);
                        worst = worst.max(ratio);
                        samples += 1;
                    }
                }
            }
            // The slack is `f32` rounding, not slack in the property. The ratio divides a
            // field difference by a step of `0.1`, so an absolute error of one `f32` ulp at
            // these magnitudes (~6, i.e. ~5e-7) shows up MAGNIFIED tenfold in the ratio.
            // The observed worst is 1.0000048; anything approaching 1.001 would be a real
            // violation, not arithmetic. (This read `1e-9` while the field was `f64`.)
            assert!(
                worst <= 1.0 + 1e-4,
                "{metric:?} field is not 1-Lipschitz: worst ratio {worst} over {samples} pairs"
            );
        }
    }

    /// The metrics bracket each other: `L∞ <= L2 <= sqrt(2)·L∞` in the plane. A useful
    /// guard that neither implementation has drifted into computing the other.
    #[test]
    fn chebyshev_and_euclidean_bracket_each_other() {
        let profile: [[f32; 2]; 4] = [[0.0, 0.0], [5.0, 1.0], [3.0, 6.0], [-1.0, 4.0]];
        for xi in -10..=15i32 {
            for yi in -10..=15i32 {
                let p = [xi as f32 * 0.5, yi as f32 * 0.5];
                let chebyshev =
                    signed_distance_to_polygon(&profile, p, Metric::Chebyshev).abs();
                let euclidean =
                    signed_distance_to_polygon(&profile, p, Metric::Euclidean).abs();
                // `1e-5` is a few `f32` ulps at these magnitudes (distances run to ~10, and
                // one ulp there is ~1e-6). The tight side is the upper bound, where the
                // sqrt(2) factor is ATTAINED exactly on a 45° diagonal — at [-4.5, -4.5]
                // the two read 6.363961 and 4.5·sqrt(2) = 6.3639603, a one-ulp excess that
                // is the bound being met, not exceeded. (This read `1e-9` under `f64`.)
                assert!(
                    chebyshev <= euclidean + 1e-5,
                    "at {p:?}: chebyshev {chebyshev} exceeds euclidean {euclidean}"
                );
                assert!(
                    euclidean <= chebyshev * 2.0f32.sqrt() + 1e-5,
                    "at {p:?}: euclidean {euclidean} exceeds sqrt(2)·chebyshev {chebyshev}"
                );
            }
        }
    }

    #[test]
    fn degenerate_polygons_have_no_boundary() {
        assert_eq!(
            signed_distance_to_polygon(&[], [0.0, 0.0], Metric::Euclidean),
            f32::INFINITY
        );
        assert_eq!(
            signed_distance_to_polygon(&[[1.0, 1.0]], [0.0, 0.0], Metric::Chebyshev),
            f32::INFINITY
        );
    }
}
