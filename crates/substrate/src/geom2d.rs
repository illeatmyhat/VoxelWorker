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
//! coarse-solid classifier question, and a shader never asks it — a raymarch
//! asks about points, not cells. It is also where the extra width genuinely earns its
//! keep. Checked against exact `i128` arithmetic, `f32` starts returning **wrong
//! orientation signs from about ±4096 voxels outward** (0 wrong at 2¹², 453 at 2²⁰, 8,817
//! at 2²⁴), while `f64` stays exact past 2³⁰. A wrong sign here does not merely blur a
//! surface: it makes the classifier **over-claim solid**, which is unsound rather than
//! conservative, and a whole cell is then filled without ever being sampled.
//!
//! That is the line: **predicates classify, fields measure.** A predicate
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
//! - [`RegionEdge`] — a region's boundary is a loop of these: a straight span or a
//!   circular arc that stays an arc all the way down to the measurement. Distance to an
//!   arc is analytic, and containment splits it at its own turning points so a curved
//!   edge obeys the same crossing rule a straight one does. A polygon is a *drawing* of a
//!   region; this is the region.
//! - [`rectangle_inside_polygon`] — whether a closed axis-aligned rectangle lies
//!   wholly inside a polygon. Exact by connectedness: if no polygon edge crosses
//!   the rectangle it holds no piece of the boundary, so it is wholly in or out,
//!   and one interior sample (the center) decides. Conservative on a grazing
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

/// Twice the signed area of triangle `(a, b, c)`, or `(b − a) × (c − a)`.
/// Positive means a counter-clockwise turn, negative means clockwise, and zero means collinear.
#[inline]
#[must_use]
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[1] - a[1]).mul_add(-(c[0] - a[0]), (b[0] - a[0]) * (c[1] - a[1]))
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

/// Whether two closed segments intersect, including collinear and endpoint touches.
/// The test combines orientation signs with a bounding-box check for collinear points.
#[must_use]
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

/// Whether segment `a→b` intersects the closed axis-aligned rectangle `[rect_min, rect_max]`.
/// An endpoint inside the rectangle or a crossing of one of its edges counts as an intersection.
#[must_use]
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
    let [lower_left, lower_right, upper_right, upper_left] = corners;
    [
        (lower_left, lower_right),
        (lower_right, upper_right),
        (upper_right, upper_left),
        (upper_left, lower_left),
    ]
    .into_iter()
    .any(|(edge_start, edge_end)| segments_intersect(a, b, edge_start, edge_end))
}

/// Whether `sample` lies inside the implicitly closed polygon using an even-odd ray test.
///
/// No on-boundary tie-breaking is done: callers that need exactness (e.g. voxel
/// sample centers at half-integer positions against integer vertices) rely on the
/// sample never lying on an edge.
///
/// `f32`, with the rest of the measurement half: this is the boundary authority a WGSL
/// preview must port, and it supplies the sign for [`signed_distance_to_polygon`]. See
/// the module docs for why the width is part of the contract rather than an accident.
#[must_use]
pub fn point_in_polygon(polygon: &[[f32; 2]], sample: [f32; 2]) -> bool {
    let mut inside = false;
    let Some(mut previous) = polygon.last().copied() else {
        return false;
    };
    for &current in polygon {
        let current_0 = current[0];
        let current_1 = current[1];
        let previous_0 = previous[0];
        let previous_1 = previous[1];
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

/// Whether a closed axis-aligned rectangle lies entirely inside the polygon.
/// The rectangle is inside when no polygon edge intersects it and its center is inside.
#[must_use]
pub fn rectangle_inside_polygon(
    polygon: &[[f64; 2]],
    rect_min: [f64; 2],
    rect_max: [f64; 2],
) -> bool {
    let count = polygon.len();
    if count < 3 || rect_max[0] < rect_min[0] || rect_max[1] < rect_min[1] {
        return false;
    }
    let Some(mut previous) = polygon.last().copied() else {
        return false;
    };
    for &current in polygon {
        if segment_intersects_rect(current, previous, rect_min, rect_max) {
            return false;
        }
        previous = current;
    }
    // The edge tests above are the exactness-critical ones and ran in `f64`: a wrong
    // orientation sign there would let a straddled rectangle through as "inside" and
    // over-claim solid. The center test is a different question and is answered in `f32`,
    // deliberately:
    //
    // - It is only ever REACHED when no polygon edge meets the rectangle, so the center
    //   is not near the boundary — the case where width would matter has already been
    //   decided by the exact half.
    // - `point_in_polygon` is the same call the per-voxel resolve makes to decide
    //   occupancy. Answering the center in `f64` here while the resolve answers it in
    //   `f32` would let the coarse claim and the per-voxel truth disagree on a sample
    //   sitting on an edge — exactly the "same set, different rounding" failure this
    //   classifier is supposed to avoid. Sharing the width makes them agree by
    //   construction.
    let narrowed: Vec<[f32; 2]> = polygon.iter().copied().map(narrow_to_measurement).collect();
    let center = narrow_to_measurement([
        (rect_min[0] + rect_max[0]) * 0.5,
        (rect_min[1] + rect_max[1]) * 0.5,
    ]);
    point_in_polygon(&narrowed, center)
}

/// Convert predicate coordinates at the CPU/GPU boundary.
///
/// The measurement implementation intentionally uses `f32` so it matches the shader. This is
/// the single audited narrowing point for the predicate-to-measurement handoff.
#[allow(clippy::as_conversions, clippy::cast_possible_truncation)]
const fn narrow_to_measurement(point: [f64; 2]) -> [f32; 2] {
    [point[0] as f32, point[1] as f32]
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
    #[must_use]
    pub fn length(self, delta: [f32; 2]) -> f32 {
        match self {
            Self::Euclidean => delta[0].hypot(delta[1]),
            Self::Chebyshev => delta[0].abs().max(delta[1].abs()),
        }
    }

    /// The distance between two points under this metric.
    #[inline]
    #[must_use]
    pub fn distance(self, a: [f32; 2], b: [f32; 2]) -> f32 {
        self.length([b[0] - a[0], b[1] - a[1]])
    }

    /// The circumradius of an axis-aligned 3D cell with the given per-axis `half_extent`, measured
    /// in THIS metric — the radius a field's 1-Lipschitz bound multiplies to bracket the cell's
    /// coarse AIR/SOLID interval. Under **Chebyshev** (L∞) it is the largest
    /// half-extent (`h`), under **Euclidean** (L2) the half-diagonal (`h√3` for a cube) — the
    /// tightening that makes interior elision cheaper for rectilinear bodies. The 3D sibling of
    /// [`length`](Self::length), and the ONE place this metric split lives, so a new `Metric`
    /// variant is a compile error here rather than a silent under-bracket at one of the producers
    /// that share it.
    #[inline]
    #[must_use]
    pub fn cell_circumradius(self, half_extent: [f32; 3]) -> f32 {
        match self {
            Self::Euclidean => half_extent[2]
                .mul_add(
                    half_extent[2],
                    half_extent[1].mul_add(half_extent[1], half_extent[0] * half_extent[0]),
                )
                .sqrt(),
            Self::Chebyshev => half_extent[0].max(half_extent[1]).max(half_extent[2]),
        }
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
#[must_use]
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
            let length_squared = delta[1].mul_add(delta[1], delta[0] * delta[0]);
            at(offset[1].mul_add(delta[1], offset[0] * delta[0]) / length_squared)
        }
        Metric::Chebyshev => {
            let mut best = at(0.0).min(at(1.0));
            // Slope changes of |gx|, |gy|, and of the max between them.
            let breakpoints = [
                (offset[0], delta[0]),                        // gx = 0
                (offset[1], delta[1]),                        // gy = 0
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
#[must_use]
pub fn signed_distance_to_polygon(polygon: &[[f32; 2]], point: [f32; 2], metric: Metric) -> f32 {
    let nearest = nearest_edge_distance(polygon, point, metric);
    if point_in_polygon(polygon, point) {
        -nearest
    } else {
        nearest
    }
}

/// The UNSIGNED distance to the polygon's nearest edge — the magnitude half of
/// [`signed_distance_to_polygon`], split out so a caller that decides the sign for itself walks the
/// edges once instead of twice. `f32::INFINITY` when there is no boundary to measure.
fn nearest_edge_distance(polygon: &[[f32; 2]], point: [f32; 2], metric: Metric) -> f32 {
    if polygon.len() < 2 {
        return f32::INFINITY;
    }
    let mut nearest = f32::INFINITY;
    let Some(mut previous) = polygon.last().copied() else {
        return f32::INFINITY;
    };
    for &current in polygon {
        nearest = nearest.min(distance_point_to_segment(previous, current, point, metric));
        previous = current;
    }
    nearest
}

/// How a boundary loop contributes to a multi-loop region: `Fill` claims its own area, `Hole`
/// leaves it void.
///
/// "Its own area" is the point: a loop governs the ground it encloses that no NARROWER loop
/// encloses. That is why a region is an ORDERED fold (see [`point_in_region`]) and not a global
/// algebra — a `Hole` carves the ground it sits on without reaching into whatever sits inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopRole {
    /// The loop's own area is part of the region.
    Fill,
    /// The loop's own area is left out of it.
    Hole,
}

/// One boundary edge of a region: a straight span, or a circular arc that **stays** a circular arc.
///
/// # Why the region is edges and not vertices
///
/// A polygon is a drawing of a region at some chosen resolution; it is not the region. Once a
/// boundary has been flattened into vertices, every consumer downstream inherits whatever tolerance
/// the flattener happened to pick, and the only way to ask for something better is to pass a
/// tolerance back up — which is how a rendering concern ends up as an argument to a query.
///
/// Carrying the arc instead removes the question. [`signed_distance_to_region`] measures to the
/// true curve, [`point_in_region`] classifies against the true curve, and a length scale enters
/// only where something discrete is actually produced (a voxel grid, a crease polyline). It is also
/// cheaper: one arc stands where twenty-odd chords otherwise would.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegionEdge {
    /// A straight span from `start` to `end`.
    Segment {
        /// The tail.
        start: [f32; 2],
        /// The head.
        end: [f32; 2],
    },
    /// A circular arc from `start` to `end`, traveling `sweep_radians` about `center` —
    /// counter-clockwise when the sweep is positive, clockwise when it is negative.
    ///
    /// The endpoints are carried alongside the center/radius/angle solve rather than recomputed
    /// from it: an endpoint shared with the next edge must be the SAME value on both sides, or the
    /// crossing parity at that vertex can count twice or not at all.
    Arc {
        /// The tail.
        start: [f32; 2],
        /// The head.
        end: [f32; 2],
        /// The circle's center.
        center: [f32; 2],
        /// The circle's radius.
        radius: f32,
        /// The bearing of `start` from `center`, in radians.
        start_radians: f32,
        /// The signed angle travelled from `start` to `end`.
        sweep_radians: f32,
    },
}

impl RegionEdge {
    /// The edge's tail.
    #[inline]
    #[must_use]
    pub const fn start(&self) -> [f32; 2] {
        match self {
            Self::Segment { start, .. } | Self::Arc { start, .. } => *start,
        }
    }

    /// The edge's head.
    #[inline]
    #[must_use]
    pub const fn end(&self) -> [f32; 2] {
        match self {
            Self::Segment { end, .. } | Self::Arc { end, .. } => *end,
        }
    }

    /// The TIGHT axis-aligned bounds of the edge — for an arc the extent of the curve itself,
    /// which reaches past its chord by the sagitta at every bulge.
    ///
    /// This is what an extent measured from a curved profile must use. Bounds taken from a chord
    /// approximation understate the reach, and a producer sized from them clips the bulge it was
    /// asked to build.
    #[must_use]
    pub fn bounds(&self) -> ([f32; 2], [f32; 2]) {
        let (start, end) = (self.start(), self.end());
        let mut low = [start[0].min(end[0]), start[1].min(end[1])];
        let mut high = [start[0].max(end[0]), start[1].max(end[1])];
        if let Self::Arc {
            center,
            radius,
            start_radians,
            sweep_radians,
            ..
        } = self
        {
            // The four compass extremes of the circle, each counted only where the arc reaches it.
            for bearing in [
                0.0,
                std::f32::consts::FRAC_PI_2,
                std::f32::consts::PI,
                std::f32::consts::PI + std::f32::consts::FRAC_PI_2,
            ] {
                if travel_to_bearing(*start_radians, *sweep_radians, bearing).is_none() {
                    continue;
                }
                let reach = [
                    radius.mul_add(bearing.cos(), center[0]),
                    radius.mul_add(bearing.sin(), center[1]),
                ];
                low = [low[0].min(reach[0]), low[1].min(reach[1])];
                high = [high[0].max(reach[0]), high[1].max(reach[1])];
            }
        }
        (low, high)
    }

    /// Distance from `point` to the edge under `metric`. Never negative; zero exactly on it.
    ///
    /// A segment defers to [`distance_point_to_segment`]. An arc whose bearing from `point` falls
    /// within the sweep is `‖point − center‖ − radius` in magnitude under **Euclidean**; otherwise
    /// the nearer endpoint is the closest thing on the curve. **Chebyshev** has no such closed
    /// form and is solved by candidate angles (`chebyshev_distance_to_arc`, private).
    #[must_use]
    pub fn distance(&self, point: [f32; 2], metric: Metric) -> f32 {
        match self {
            Self::Segment { start, end } => distance_point_to_segment(*start, *end, point, metric),
            Self::Arc {
                start,
                end,
                center,
                radius,
                start_radians,
                sweep_radians,
            } => {
                let to_ends = metric
                    .distance(*start, point)
                    .min(metric.distance(*end, point));
                match metric {
                    Metric::Euclidean => {
                        let offset = [point[0] - center[0], point[1] - center[1]];
                        let bearing = offset[1].atan2(offset[0]);
                        if travel_to_bearing(*start_radians, *sweep_radians, bearing).is_some() {
                            (metric.length(offset) - radius).abs()
                        } else {
                            to_ends
                        }
                    }
                    Metric::Chebyshev => to_ends.min(chebyshev_distance_to_arc(
                        *center,
                        *radius,
                        *start_radians,
                        *sweep_radians,
                        point,
                    )),
                }
            }
        }
    }

    /// How many times a ray cast from `sample` in the `+axis0` direction crosses this edge — the
    /// per-edge term of the crossing-number test [`point_in_polygon`] runs over a vertex list.
    ///
    /// A segment uses the textbook half-open rule (an edge counts when exactly one endpoint is
    /// strictly above the sample), which is what makes a vertex shared by two edges count once. An
    /// arc can cross the ray's line twice, so it is first cut at its own top and bottom — the only
    /// places where the tangent turns horizontal — leaving pieces that are `axis1`-monotone and
    /// obey that SAME rule. The cut points are the arc's, not the sampler's, so the parity is
    /// independent of where the ray happens to sit.
    fn crossings(&self, sample: [f32; 2]) -> u32 {
        match self {
            Self::Segment { start, end } => segment_crossings(*start, *end, sample),
            Self::Arc {
                start,
                end,
                center,
                radius,
                start_radians,
                sweep_radians,
            } => {
                let span = sweep_radians.abs();
                let mut cuts = vec![0.0, span];
                for extreme in [std::f32::consts::FRAC_PI_2, -std::f32::consts::FRAC_PI_2] {
                    if let Some(travel) = travel_to_bearing(*start_radians, *sweep_radians, extreme)
                    {
                        if travel > 0.0 && travel < span {
                            cuts.push(travel);
                        }
                    }
                }
                cuts.sort_by(f32::total_cmp);
                let direction: f32 = if *sweep_radians < 0.0 { -1.0 } else { 1.0 };
                let at = |travel: f32| {
                    let bearing = direction.mul_add(travel, *start_radians);
                    [
                        radius.mul_add(bearing.cos(), center[0]),
                        radius.mul_add(bearing.sin(), center[1]),
                    ]
                };
                let mut crossings: u32 = 0;
                for piece in cuts.array_windows::<2>() {
                    let [entry, exit] = piece;
                    let (entry, exit) = (*entry, *exit);
                    if exit <= entry {
                        continue;
                    }
                    // The outer ends are the STORED endpoints, so a vertex shared with the next
                    // edge is the same value on both sides of the join.
                    let low = if entry.abs() <= f32::EPSILON {
                        *start
                    } else {
                        at(entry)
                    };
                    let high = if (exit - span).abs() <= f32::EPSILON {
                        *end
                    } else {
                        at(exit)
                    };
                    if (low[1] > sample[1]) == (high[1] > sample[1]) {
                        continue;
                    }
                    // The piece is monotone, so its half of the circle decides which root of
                    // `axis0 = center ± √(r² − dy²)` it crosses at.
                    let rise = sample[1] - center[1];
                    let half_chord = (*radius).mul_add(*radius, -(rise * rise)).max(0.0).sqrt();
                    let middle = direction.mul_add((entry + exit) * 0.5, *start_radians);
                    let crossing_0 = if middle.cos() >= 0.0 {
                        center[0] + half_chord
                    } else {
                        center[0] - half_chord
                    };
                    if sample[0] < crossing_0 {
                        crossings = crossings.saturating_add(1);
                    }
                }
                crossings
            }
        }
    }
}

/// How far along a sweep the bearing `bearing` sits, in radians of travel from the start, or `None`
/// when the bearing is off the arc. Direction-agnostic: travel is always non-negative and compares
/// against `|sweep|`, so a clockwise arc is the mirror of a counter-clockwise one rather than a
/// second set of comparisons to keep in step.
fn travel_to_bearing(start_radians: f32, sweep_radians: f32, bearing: f32) -> Option<f32> {
    let turn = std::f32::consts::TAU;
    let travelled = if sweep_radians < 0.0 {
        (start_radians - bearing).rem_euclid(turn)
    } else {
        (bearing - start_radians).rem_euclid(turn)
    };
    (travelled <= sweep_radians.abs()).then_some(travelled)
}

/// The Chebyshev (L∞) distance from `point` to the arc's CURVE, ignoring its endpoints (the caller
/// folds those in).
///
/// Writing the arc as `center + radius·(cos t, sin t)`, the distance is
///
/// ```text
/// f(t) = max(|gx(t)|, |gy(t)|)   gx(t) = cx + r·cos t − px,  gy(t) = cy + r·sin t − py
/// ```
///
/// which is smooth except where a term changes sign or the two swap dominance. Its minimum over the
/// sweep is therefore attained at an end of the sweep or at one of those breakpoints: `gy` turns at
/// `t ∈ {0, π}`, `gx` at `t ∈ {π/2, 3π/2}`, and the swap `|gx| = |gy|` solves in closed form as
/// `√2·r·cos(t ± π/4) = (px − cx) ∓ (py − cy)`. Evaluating those candidates is **exact**, in the
/// same way [`distance_point_to_segment`]'s Chebyshev branch is.
///
/// CPU-only, like the rest of the Chebyshev branch: it is the lattice metric an outset measures in,
/// and the WGSL mirror only ever wants the round one.
fn chebyshev_distance_to_arc(
    center: [f32; 2],
    radius: f32,
    start_radians: f32,
    sweep_radians: f32,
    point: [f32; 2],
) -> f32 {
    let offset = [point[0] - center[0], point[1] - center[1]];
    let mut nearest = f32::INFINITY;
    let mut consider = |bearing: f32| {
        if travel_to_bearing(start_radians, sweep_radians, bearing).is_none() {
            return;
        }
        nearest = nearest.min(Metric::Chebyshev.length([
            radius.mul_add(bearing.cos(), -offset[0]),
            radius.mul_add(bearing.sin(), -offset[1]),
        ]));
    };
    for bearing in [
        0.0,
        std::f32::consts::FRAC_PI_2,
        std::f32::consts::PI,
        std::f32::consts::PI + std::f32::consts::FRAC_PI_2,
    ] {
        consider(bearing);
    }
    let amplitude = radius * std::f32::consts::SQRT_2;
    if amplitude > 0.0 {
        for sign in [1.0_f32, -1.0_f32] {
            let ratio = sign.mul_add(-offset[1], offset[0]) / amplitude;
            if ratio.abs() > 1.0 {
                continue;
            }
            let base = ratio.acos();
            for direction in [1.0, -1.0] {
                consider(sign.mul_add(-std::f32::consts::FRAC_PI_4, direction * base));
            }
        }
    }
    nearest
}

/// Whether a ray cast from `sample` in the `+axis0` direction crosses the segment `a → b`. The
/// per-edge term [`point_in_polygon`] inlines over a vertex list, kept identical here so a region
/// of edges and a polygon of vertices classify a shared boundary the same way.
fn segment_crossings(a: [f32; 2], b: [f32; 2], sample: [f32; 2]) -> u32 {
    if (b[1] > sample[1]) == (a[1] > sample[1]) {
        return 0;
    }
    let crossing_0 = (a[0] - b[0]) * (sample[1] - b[1]) / (a[1] - b[1]) + b[0];
    u32::from(sample[0] < crossing_0)
}

/// Whether `sample` is inside the closed loop of possibly curved edges.
/// The test uses the same crossing-number rule as [`point_in_polygon`].
#[must_use]
pub fn point_in_edge_loop(edges: &[RegionEdge], sample: [f32; 2]) -> bool {
    let crossings: u32 = edges.iter().map(|edge| edge.crossings(sample)).sum();
    crossings % 2 == 1
}

/// The UNSIGNED distance to the loop's nearest edge. `f32::INFINITY` for an empty loop.
#[must_use]
pub fn nearest_boundary_distance(edges: &[RegionEdge], point: [f32; 2], metric: Metric) -> f32 {
    edges
        .iter()
        .map(|edge| edge.distance(point, metric))
        .fold(f32::INFINITY, f32::min)
}

/// Whether `sample` is inside the region `loops` — decided by the FIRST loop that contains it.
///
/// # The ordering is the contract
///
/// `loops` must run INNERMOST-FIRST: a loop appears before every loop that contains it. A caller
/// deriving loops from nested boundaries gets that for free by sorting on enclosed area ascending,
/// since strict containment means strictly smaller area.
///
/// Given that order, "the first containing loop wins" means each loop decides its OWN area and
/// nothing deeper. This is not a detail of the algorithm, it is the meaning of a nested profile: a
/// carved region does not carve what is nested inside it, so a `Hole` around a `Fill` leaves that
/// `Fill` standing as an island. An algebra over the whole loop set cannot express that — union
/// then subtract makes an outer `Hole` veto everything within it, which is a different (and, for
/// authoring, a surprising) shape.
///
/// It is also NOT a global crossing parity: two loops that touch or share an edge each keep their
/// own area, where even-odd would cancel them.
///
/// A sample in no loop at all is outside, so an empty region contains nothing.
#[must_use]
pub fn point_in_region(loops: &[(LoopRole, Vec<RegionEdge>)], sample: [f32; 2]) -> bool {
    for (role, edges) in loops {
        if point_in_edge_loop(edges, sample) {
            return *role == LoopRole::Fill;
        }
    }
    false
}

/// Signed distance from `point` to the region's boundary under `metric` — negative inside,
/// positive outside. The field reading of [`point_in_region`], and it takes the same
/// innermost-first `loops`.
///
/// The sign comes from the predicate; the magnitude is the distance to the nearest loop boundary of
/// any kind. Every boundary of the region is one of those, so the field is zero wherever the sign
/// flips and stays 1-Lipschitz and continuous — what the interval bounds need. Where a loop edge is
/// INTERIOR to the region (two adjacent `Fill` loops sharing it) the magnitude UNDERSTATES the true
/// clearance, which narrows a coarse claim and never widens one (CONSERVATIVE-NEVER-NARROW).
///
/// An empty region is `f32::INFINITY` (everywhere outside), matching the composite fold's empty
/// accumulator.
#[must_use]
pub fn signed_distance_to_region(
    loops: &[(LoopRole, Vec<RegionEdge>)],
    point: [f32; 2],
    metric: Metric,
) -> f32 {
    let mut nearest = f32::INFINITY;
    let mut inside = None;
    for (role, edges) in loops {
        nearest = nearest.min(nearest_boundary_distance(edges, point, metric));
        // The innermost containing loop decides, so only the first one to answer counts.
        if inside.is_none() && point_in_edge_loop(edges, point) {
            inside = Some(*role == LoopRole::Fill);
        }
    }
    if inside.unwrap_or(false) {
        -nearest
    } else {
        nearest
    }
}

/// The point of the region's interior FARTHEST from its boundary — its pole of inaccessibility —
/// together with that clearance. `None` when the region encloses nothing.
///
/// # Why the deepest point and not the centroid
///
/// This is an identity, not a display position: something names a face by a point inside it and
/// must still name the same face after the boundary moves a little. A centroid leaves the interior
/// entirely for any crescent or L-shape, and even where it stays inside it can sit a hair from an
/// edge, so a small edit walks it out. The deepest point is the one with the most room to survive,
/// and `clearance` is exactly how much of an edit it can absorb.
///
/// # The search
///
/// Garcia-Castellanos & Lombardo's pole of inaccessibility (2007), by the quadtree refinement
/// Mapbox's *polylabel* (2016) popularised: cover the bounds in square cells, and repeatedly
/// subdivide whichever cell has the best *possible* answer left in it — its center's clearance
/// plus its own half-diagonal, since the field is 1-Lipschitz and cannot climb faster than the
/// distance travelled. That bound is what makes the search exhaustive rather than lucky: a sliver
/// no coarse sample lands in still has a cell whose optimiztic bound outranks the current best, so
/// it gets subdivided rather than missed. It stops once no cell can beat the best by more than
/// `precision`.
///
/// Unlike the published algorithm this measures to CURVES, not to a flattened polygon
/// ([`signed_distance_to_region`] over [`RegionEdge`]s), so a disc's pole is its center exactly
/// rather than the center of a chord approximation. It also takes a REGION and not a single loop,
/// for the same reason the identity wants the deepest point in the first place: the pole of a ring
/// has to be in the ring, not in the hole the ring is drawn around. `loops` is innermost-first,
/// the order [`point_in_region`] states.
#[must_use]
pub fn deepest_interior_point(
    loops: &[(LoopRole, Vec<RegionEdge>)],
    precision: f32,
) -> Option<([f32; 2], f32)> {
    let (low, high) = region_bounds(loops)?;
    let (width, height) = (high[0] - low[0], high[1] - low[1]);
    let side = width.min(height);
    // NaN bounds and a degenerate span alike: there is no interior to search.
    if side.is_nan() || side <= 0.0 {
        return None;
    }
    let depth = |point: [f32; 2]| -signed_distance_to_region(loops, point, Metric::Euclidean);
    // Seed with the bounds' center so a convex loop is answered before any subdivision, and so
    // there is always a best to compare optimiztic bounds against.
    let mut best = [low[0] + width / 2.0, low[1] + height / 2.0];
    let mut best_depth = depth(best);
    // Each cell carries the depth at its center, measured once when it is created — the search
    // spends its whole cost in `depth`, so re-reading it while ranking would square that.
    let cell = |center: [f32; 2], half: f32| Cell {
        center,
        half,
        bound: half.mul_add(std::f32::consts::SQRT_2, depth(center)),
    };
    let mut queue: std::collections::BinaryHeap<Cell> = std::collections::BinaryHeap::new();
    let mut x = low[0];
    while x.total_cmp(&high[0]).is_lt() {
        let mut y = low[1];
        while y.total_cmp(&high[1]).is_lt() {
            queue.push(cell([x + side / 2.0, y + side / 2.0], side / 2.0));
            y += side;
        }
        x += side;
    }
    // A budget, not a convergence criterion: the loop below terminates on its own because every
    // subdivision halves `half`. This only bounds the cost of a pathological boundary.
    for _ in 0..MAXIMUM_POLE_CELLS {
        let Some(popped) = queue.pop() else {
            break;
        };
        // The heap's head is the best any surviving cell could do, so once IT cannot beat the
        // incumbent by `precision`, nothing can and the search is over.
        if popped.bound - best_depth <= precision {
            break;
        }
        let here = popped.half.mul_add(-std::f32::consts::SQRT_2, popped.bound);
        if here > best_depth {
            best_depth = here;
            best = popped.center;
        }
        let quarter = popped.half / 2.0;
        for (dx, dy) in [
            (-1.0_f32, -1.0_f32),
            (1.0_f32, -1.0_f32),
            (-1.0_f32, 1.0_f32),
            (1.0_f32, 1.0_f32),
        ] {
            queue.push(cell(
                [
                    dx.mul_add(quarter, popped.center[0]),
                    dy.mul_add(quarter, popped.center[1]),
                ],
                quarter,
            ));
        }
    }
    (best_depth > 0.0).then_some((best, best_depth))
}

/// One square of the [`deepest_interior_point`] search, ordered by the best clearance it could
/// still hold: its center's, plus how far its corner reaches. That ordering is the whole point —
/// the search always subdivides the most promising square left, so it is a max-heap of `bound`
/// and nothing else participates in the comparison.
struct Cell {
    center: [f32; 2],
    half: f32,
    bound: f32,
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bound.total_cmp(&other.bound)
    }
}

/// The cost ceiling on one [`deepest_interior_point`] search.
const MAXIMUM_POLE_CELLS: usize = 4096;

/// The union of every edge's [`RegionEdge::bounds`] across every loop, or `None` for an empty
/// region.
fn region_bounds(loops: &[(LoopRole, Vec<RegionEdge>)]) -> Option<([f32; 2], [f32; 2])> {
    let mut bounds: Option<([f32; 2], [f32; 2])> = None;
    for edge in loops.iter().flat_map(|(_, edges)| edges) {
        let (low, high) = edge.bounds();
        bounds = Some(match bounds {
            None => (low, high),
            Some((was_low, was_high)) => (
                [was_low[0].min(low[0]), was_low[1].min(low[1])],
                [was_high[0].max(high[0]), was_high[1].max(high[1])],
            ),
        });
    }
    bounds
}

/// Whether a closed axis-aligned rectangle lies entirely inside the region.
/// The innermost loop that touches it must be a `Fill` containing it whole.
///
/// **Conservative**: it never claims a rectangle that is not wholly solid, but it declines
/// rectangles that are (one spanning two adjacent `Fill` loops, say), because the coarse
/// classifier's contract is to narrow work, never to narrow truth.
///
/// # Why this half still takes a polygon
///
/// This is the one consumer that genuinely wants vertices: it is the exact-`f64` cell classifier
/// (see the module docs on the width split), and its exactness rests on [`orient2d`] signs over
/// straight edges. So the caller flattens for it — the terminal-adapter case, where a discrete
/// artifact is actually being produced — and hands over `curve_bounds`, the bounds of every edge
/// the flattening approximated. **Any rectangle meeting one of those is declined outright.** The
/// chord/curve discrepancy lives strictly inside those bounds, so it can never sit inside a
/// rectangle this claims, and the connectedness argument holds unchanged everywhere else. Without
/// that guard a chord cutting to the void side of a concave curve would let the classifier fill a
/// cell that is not wholly solid, which is unsound rather than merely coarse.
#[must_use]
pub fn rectangle_inside_region(
    loops: &[(LoopRole, Vec<[f64; 2]>)],
    curve_bounds: &[([f64; 2], [f64; 2])],
    rect_min: [f64; 2],
    rect_max: [f64; 2],
) -> bool {
    let overlaps = |low: &[f64; 2], high: &[f64; 2]| {
        low[0] <= rect_max[0]
            && high[0] >= rect_min[0]
            && low[1] <= rect_max[1]
            && high[1] >= rect_min[1]
    };
    if curve_bounds.iter().any(|(low, high)| overlaps(low, high)) {
        return false;
    }
    for (role, polygon) in loops {
        if !rectangle_meets_polygon(polygon, rect_min, rect_max) {
            continue;
        }
        // The innermost loop with any claim on this rectangle. It decides the whole rectangle only
        // if it is a `Fill` that swallows it; a loop that merely crosses the rectangle leaves part
        // of it to something narrower, which is not a claim this may build on.
        return *role == LoopRole::Fill && rectangle_inside_polygon(polygon, rect_min, rect_max);
    }
    false
}

/// Whether the closed rectangle touches the polygon's interior or boundary at all — an edge
/// crossing, or a center inside it. The negation of "provably disjoint", so a `false` is the
/// only answer [`rectangle_inside_region`] is allowed to build a solid claim on.
fn rectangle_meets_polygon(polygon: &[[f64; 2]], rect_min: [f64; 2], rect_max: [f64; 2]) -> bool {
    let count = polygon.len();
    if count < 3 {
        return false;
    }
    let Some(mut previous) = polygon.last().copied() else {
        return false;
    };
    for &current in polygon {
        if segment_intersects_rect(current, previous, rect_min, rect_max) {
            return true;
        }
        previous = current;
    }
    // No edge crosses, so the rectangle is wholly inside or wholly outside: its center decides,
    // in the same `f32` the per-voxel resolve uses (see `rectangle_inside_polygon`).
    let narrowed: Vec<[f32; 2]> = polygon.iter().copied().map(narrow_to_measurement).collect();
    point_in_polygon(
        &narrowed,
        narrow_to_measurement([
            (rect_min[0] + rect_max[0]) * 0.5,
            (rect_min[1] + rect_max[1]) * 0.5,
        ]),
    )
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
    use super::*;

    /// The same 4x4 square in both widths — the predicate half is `f64`, the measurement
    /// half `f32`, so a test that spans them needs both. Both are written from the same
    /// integer literals rather than one being cast from the other, matching how the sketch
    /// producer converts its `i64` profile twice from one source.
    const UNIT_SQUARE: [[f64; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
    const UNIT_SQUARE_MEASURED: [[f32; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];

    /// A 12x12 square with a 4x4 square centered in it — the donut every hole test needs.
    const OUTER_MEASURED: [[f32; 2]; 4] = [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0]];
    const INNER_MEASURED: [[f32; 2]; 4] = [[4.0, 4.0], [8.0, 4.0], [8.0, 8.0], [4.0, 8.0]];
    const OUTER: [[f64; 2]; 4] = [[0.0, 0.0], [12.0, 0.0], [12.0, 12.0], [0.0, 12.0]];
    const INNER: [[f64; 2]; 4] = [[4.0, 4.0], [8.0, 4.0], [8.0, 8.0], [4.0, 8.0]];

    /// A vertex list as a closed loop of straight edges — the shape a region takes now that its
    /// boundary is edges. Only the tests that are ABOUT curvature build arcs.
    fn closed_loop(points: &[[f32; 2]]) -> Vec<RegionEdge> {
        (0..points.len())
            .map(|index| RegionEdge::Segment {
                start: points[index],
                end: points[(index + 1) % points.len()],
            })
            .collect()
    }

    /// A full circle as ONE arc edge — degenerate as a polygon, exact as a curve.
    fn circle(center: [f32; 2], radius: f32) -> Vec<RegionEdge> {
        let seam = [center[0] + radius, center[1]];
        vec![RegionEdge::Arc {
            start: seam,
            end: seam,
            center,
            radius,
            start_radians: 0.0,
            sweep_radians: std::f32::consts::TAU,
        }]
    }

    /// A hole is carved, not parity-canceled: the ring is inside and the pocket is not. Loops run
    /// innermost-first, so the pocket gets its say before the square it sits in.
    #[test]
    fn a_hole_is_carved_out_of_its_fill() {
        let region = [
            (LoopRole::Hole, closed_loop(&INNER_MEASURED)),
            (LoopRole::Fill, closed_loop(&OUTER_MEASURED)),
        ];
        assert!(point_in_region(&region, [1.0, 1.0]), "the ring is solid");
        assert!(!point_in_region(&region, [6.0, 6.0]), "the pocket is not");
        assert!(!point_in_region(&region, [-1.0, 6.0]), "outside is not");
    }

    /// Where even-odd cancels, the explicit fold does not: two NESTED `Fill` loops are one
    /// solid disc, because inclusion is per-loop state rather than a global crossing parity.
    #[test]
    fn nested_fills_do_not_cancel_each_other() {
        let region = [
            (LoopRole::Fill, closed_loop(&INNER_MEASURED)),
            (LoopRole::Fill, closed_loop(&OUTER_MEASURED)),
        ];
        assert!(point_in_region(&region, [6.0, 6.0]));
        // The even-odd rule over the same loop soup says the opposite.
        let soup: Vec<[f32; 2]> = OUTER_MEASURED
            .iter()
            .chain(&INNER_MEASURED)
            .copied()
            .collect();
        assert!(!point_in_polygon(&soup, [6.0, 6.0]));
    }

    /// The field agrees with the predicate on which side of the boundary a point is, and the
    /// hole's own wall measures as a real surface (distance 1 one voxel outside the pocket).
    #[test]
    fn the_region_field_signs_match_the_predicate() {
        let region = [
            (LoopRole::Hole, closed_loop(&INNER_MEASURED)),
            (LoopRole::Fill, closed_loop(&OUTER_MEASURED)),
        ];
        let at = |point| signed_distance_to_region(&region, point, Metric::Euclidean);
        assert!(at([1.0, 6.0]) < 0.0, "in the ring");
        assert!(at([6.0, 6.0]) > 0.0, "in the pocket");
        assert!(
            (at([3.0, 6.0]) + 1.0).abs() < 1e-5,
            "one voxel in from the hole wall"
        );
        assert_eq!(
            signed_distance_to_region(&[], [0.0, 0.0], Metric::Euclidean),
            f32::INFINITY,
            "an empty region is everywhere outside"
        );
    }

    /// The coarse classifier never claims a cell the hole touches — the conservative direction
    /// is the sound one (over-claiming solid would carve nothing where a hole belongs).
    #[test]
    fn the_coarse_region_claim_declines_anything_a_hole_touches() {
        let region = [
            (LoopRole::Hole, INNER.to_vec()),
            (LoopRole::Fill, OUTER.to_vec()),
        ];
        assert!(
            rectangle_inside_region(&region, &[], [1.0, 1.0], [3.0, 3.0]),
            "wholly in the ring"
        );
        assert!(
            !rectangle_inside_region(&region, &[], [5.0, 5.0], [7.0, 7.0]),
            "wholly in the pocket"
        );
        assert!(
            !rectangle_inside_region(&region, &[], [3.0, 3.0], [5.0, 5.0]),
            "straddling the hole wall"
        );
        assert!(
            !rectangle_inside_region(
                &[(LoopRole::Hole, INNER.to_vec())],
                &[],
                [5.0, 5.0],
                [7.0, 7.0]
            ),
            "a region with no fill claims nothing"
        );
    }

    /// The classifier reads a flattened polygon, so anywhere a curve was approximated it must
    /// decline outright — that is what keeps a chord cutting to the void side of a concave arc from
    /// filling a cell that is not wholly solid.
    #[test]
    fn the_coarse_region_claim_declines_anything_a_curve_reaches() {
        let region = [(LoopRole::Fill, OUTER.to_vec())];
        let near_the_corner = [([0.0, 0.0], [3.0, 3.0])];
        assert!(
            rectangle_inside_region(&region, &near_the_corner, [5.0, 5.0], [7.0, 7.0]),
            "far from the approximated edge"
        );
        assert!(
            !rectangle_inside_region(&region, &near_the_corner, [1.0, 1.0], [2.0, 2.0]),
            "inside the polygon, but where a curve was flattened"
        );
    }

    /// An arc is measured as a curve: every point one voxel outside a circle of radius four is one
    /// voxel from the boundary, wherever on the circle it sits. A chord approximation cannot say
    /// that — its error is largest exactly midway between two vertices.
    #[test]
    fn the_arc_field_measures_the_curve_and_not_its_chords() {
        let region = [(LoopRole::Fill, circle([0.0, 0.0], 4.0))];
        for step in 0..16 {
            let bearing = step as f32 / 16.0 * std::f32::consts::TAU;
            let outside = [5.0 * bearing.cos(), 5.0 * bearing.sin()];
            let distance = signed_distance_to_region(&region, outside, Metric::Euclidean);
            assert!(
                (distance - 1.0).abs() < 1e-4,
                "a voxel outside the circle at {bearing} measured {distance}"
            );
        }
        assert!(
            signed_distance_to_region(&region, [0.0, 0.0], Metric::Euclidean) < 0.0,
            "the center is inside"
        );
    }

    /// The crossing count over a curved edge: a full circle is entered once and left once from any
    /// interior point, whatever direction the ray leaves in. The arc is cut at its own top and
    /// bottom, so a ray grazing either one still counts a boundary exactly once.
    #[test]
    fn a_curved_loop_classifies_by_the_curve() {
        let region = [(LoopRole::Fill, circle([6.0, 6.0], 4.0))];
        assert!(point_in_region(&region, [6.0, 6.0]), "the center");
        assert!(point_in_region(&region, [9.0, 6.0]), "inside, off-center");
        assert!(!point_in_region(&region, [11.0, 6.0]), "past the rim");
        assert!(!point_in_region(&region, [6.0, 12.0]), "above it");
        // A ray leaving exactly at the circle's topmost point — the cut the monotone split makes.
        assert!(!point_in_region(&region, [0.0, 10.0]), "grazing the top");
        // The bulge reaches past its chord: a point the chord of a half-circle would call outside.
        let half = [(
            LoopRole::Fill,
            vec![
                RegionEdge::Arc {
                    start: [10.0, 6.0],
                    end: [2.0, 6.0],
                    center: [6.0, 6.0],
                    radius: 4.0,
                    start_radians: 0.0,
                    sweep_radians: std::f32::consts::PI,
                },
                RegionEdge::Segment {
                    start: [2.0, 6.0],
                    end: [10.0, 6.0],
                },
            ],
        )];
        assert!(point_in_region(&half, [6.0, 9.0]), "under the bulge");
        assert!(!point_in_region(&half, [6.0, 3.0]), "below the chord");
    }

    /// The Chebyshev branch is a real distance, not a Euclidean one with a different name: on the
    /// axis a square and a disc agree, and on the diagonal the square reaches further.
    #[test]
    fn the_arc_field_has_an_exact_chebyshev_branch() {
        let arc = RegionEdge::Arc {
            start: [4.0, 0.0],
            end: [4.0, 0.0],
            center: [0.0, 0.0],
            radius: 4.0,
            start_radians: 0.0,
            sweep_radians: std::f32::consts::TAU,
        };
        assert!(
            (arc.distance([6.0, 0.0], Metric::Chebyshev) - 2.0).abs() < 1e-4,
            "straight out along an axis, the two metrics agree"
        );
        // The nearest point on the circle to (6, 6) is at 45°, i.e. (2√2, 2√2) ≈ (2.83, 2.83), and
        // L∞ measures the larger axis gap: 6 − 2.83.
        let diagonal = arc.distance([6.0, 6.0], Metric::Chebyshev);
        let expected = 6.0 - 4.0 * std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (diagonal - expected).abs() < 1e-3,
            "on the diagonal expected {expected}, measured {diagonal}"
        );
        // From the center, the L∞ ball is a square whose CORNER reaches the circle first, so the
        // distance is `radius/√2` and not the radius. Getting this wrong is how a Euclidean
        // measurement wearing the Chebyshev name goes unnoticed.
        let from_center = arc.distance([0.0, 0.0], Metric::Chebyshev);
        assert!(
            (from_center - 4.0 * std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-3,
            "from the center, measured {from_center}"
        );
    }

    /// An arc's bounds are the curve's, not its chord's: a half-turn bulges a full radius past the
    /// line joining its ends, and an extent measured from the chord would clip it.
    #[test]
    fn arc_bounds_follow_the_bulge() {
        let half = RegionEdge::Arc {
            start: [4.0, 0.0],
            end: [-4.0, 0.0],
            center: [0.0, 0.0],
            radius: 4.0,
            start_radians: 0.0,
            sweep_radians: std::f32::consts::PI,
        };
        let (low, high) = half.bounds();
        // The compass extremes come out of `cos`/`sin`, so they land a few ulps off the axis.
        assert!(
            (low[0] + 4.0).abs() < 1e-5 && low[1].abs() < 1e-5,
            "{low:?}"
        );
        assert!(
            (high[0] - 4.0).abs() < 1e-5 && (high[1] - 4.0).abs() < 1e-5,
            "{high:?}"
        );
        let straight = RegionEdge::Segment {
            start: [4.0, 0.0],
            end: [-4.0, 0.0],
        };
        assert_eq!(straight.bounds(), ([-4.0, 0.0], [4.0, 0.0]));
    }

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
        assert!(rectangle_inside_polygon(
            &UNIT_SQUARE,
            [1.0, 1.0],
            [3.0, 3.0]
        ));
        // Pokes out the right edge.
        assert!(!rectangle_inside_polygon(
            &UNIT_SQUARE,
            [1.0, 1.0],
            [5.0, 3.0]
        ));
        // Degenerate (single point) inside.
        assert!(rectangle_inside_polygon(
            &UNIT_SQUARE,
            [2.0, 2.0],
            [2.0, 2.0]
        ));
        // Inverted rectangle rejected.
        assert!(!rectangle_inside_polygon(
            &UNIT_SQUARE,
            [3.0, 3.0],
            [1.0, 1.0]
        ));
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
            [0.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [-2.0, 3.0],
            [2.5, -1.5],
            [7.0, 0.5],
            [0.5, 7.0],
            [3.3, 3.7],
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
                let step_travel =
                    Metric::Chebyshev.length([b[0] - a[0], b[1] - a[1]]) / STEPS as f32;
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
            // Center of the 4×4 square is 2 from every edge in both metrics.
            let center = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [2.0, 2.0], metric);
            assert!((center + 2.0).abs() < 1e-9, "{metric:?} center = {center}");
            // On the boundary ⇒ zero.
            let edge = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [4.0, 2.0], metric);
            assert!(edge.abs() < 1e-9, "{metric:?} edge = {edge}");
            // Straight out from an edge: 1 away in both metrics.
            let outside = signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [5.0, 2.0], metric);
            assert!(
                (outside - 1.0).abs() < 1e-9,
                "{metric:?} outside = {outside}"
            );
            // Inside is negative, outside positive.
            assert!(signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [1.0, 1.0], metric) < 0.0);
            assert!(signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, [9.0, 9.0], metric) > 0.0);
        }
        // Diagonally off a corner is where the metrics part company: the corner (4,4) is
        // (3,3) away, so Euclidean reads 3√2 while Chebyshev reads 3.
        let corner = [7.0, 7.0];
        let euclidean =
            signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, corner, Metric::Euclidean);
        let chebyshev =
            signed_distance_to_polygon(&UNIT_SQUARE_MEASURED, corner, Metric::Chebyshev);
        assert!(
            (euclidean - 18.0f32.sqrt()).abs() < 1e-9,
            "euclidean = {euclidean}"
        );
        assert!((chebyshev - 3.0).abs() < 1e-9, "chebyshev = {chebyshev}");
    }

    /// The property every cell bound rests on: the field must not change faster than
    /// distance does, **in its own metric**. If this fails, classification built on it is
    /// unsound.
    #[test]
    fn polygon_signed_distance_is_one_lipschitz_in_its_own_metric() {
        // A deliberately awkward profile: reflex corner, a spike, and a self-intersection.
        let profile: [[f32; 2]; 7] = [
            [0.0, 0.0],
            [6.0, 0.0],
            [6.0, 6.0],
            [3.0, 2.0],
            [0.0, 6.0],
            [4.0, -1.0],
            [1.0, 4.0],
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
                let chebyshev = signed_distance_to_polygon(&profile, p, Metric::Chebyshev).abs();
                let euclidean = signed_distance_to_polygon(&profile, p, Metric::Euclidean).abs();
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

    /// A vertex list as a one-loop region — what a pole is asked for most of the time.
    fn fill(points: &[[f32; 2]]) -> Vec<(LoopRole, Vec<RegionEdge>)> {
        vec![(LoopRole::Fill, closed_loop(points))]
    }

    /// The square's pole is its center, and the clearance is the inradius.
    #[test]
    fn a_squares_pole_is_its_center() {
        let (pole, clearance) =
            deepest_interior_point(&fill(&UNIT_SQUARE_MEASURED), 1e-3).expect("a pole");
        assert!(
            (pole[0] - 2.0).abs() < 1e-2 && (pole[1] - 2.0).abs() < 1e-2,
            "{pole:?}"
        );
        assert!((clearance - 2.0).abs() < 1e-2, "{clearance}");
    }

    /// A disc's pole is its center measured to the CURVE, so the clearance is the radius exactly
    /// rather than the apothem of some chord approximation.
    #[test]
    fn a_discs_pole_reads_the_curve_not_a_chord() {
        let radius = 8.0;
        let circle = vec![RegionEdge::Arc {
            start: [radius, 0.0],
            end: [radius, 0.0],
            center: [0.0, 0.0],
            radius,
            start_radians: 0.0,
            sweep_radians: std::f32::consts::TAU,
        }];
        let (pole, clearance) =
            deepest_interior_point(&[(LoopRole::Fill, circle)], 1e-3).expect("a pole");
        assert!(pole[0].hypot(pole[1]) < 1e-2, "{pole:?}");
        assert!((clearance - radius).abs() < 1e-2, "{clearance}");
    }

    /// The case a centroid gets wrong: a C whose centroid sits in the notch, outside the shape.
    /// The pole is in one of the arms, and it is genuinely inside.
    #[test]
    fn a_crescents_pole_is_inside_it() {
        let c = closed_loop(&[
            [0.0, 0.0],
            [12.0, 0.0],
            [12.0, 3.0],
            [3.0, 3.0],
            [3.0, 9.0],
            [12.0, 9.0],
            [12.0, 12.0],
            [0.0, 12.0],
        ]);
        let centroid = [6.0, 6.0];
        assert!(
            !point_in_edge_loop(&c, centroid),
            "the centroid is in the notch"
        );
        let (pole, clearance) =
            deepest_interior_point(&[(LoopRole::Fill, c.clone())], 1e-3).expect("a pole");
        assert!(point_in_edge_loop(&c, pole), "{pole:?} is outside");
        assert!(
            clearance > 1.4,
            "the deepest point has real room: {clearance}"
        );
    }

    /// A sliver no coarse sample lands in is still found — the optimiztic bound keeps its cells
    /// alive until they are small enough to sample it.
    #[test]
    fn a_sliver_is_found_not_missed() {
        let sliver = closed_loop(&[[0.0, 0.0], [64.0, 0.0], [64.0, 0.4], [0.0, 0.4]]);
        let (pole, clearance) =
            deepest_interior_point(&[(LoopRole::Fill, sliver.clone())], 1e-3).expect("a pole");
        assert!(point_in_edge_loop(&sliver, pole), "{pole:?}");
        assert!((clearance - 0.2).abs() < 1e-2, "{clearance}");
    }

    /// The case the identity actually needs the REGION for: a ring's pole is in the ring, not in
    /// the hole it is drawn around, even though the hole is the roomiest place inside the outer
    /// loop.
    #[test]
    fn a_rings_pole_is_in_the_ring() {
        let ring = vec![
            (LoopRole::Hole, closed_loop(&INNER_MEASURED)),
            (LoopRole::Fill, closed_loop(&OUTER_MEASURED)),
        ];
        let (pole, clearance) = deepest_interior_point(&ring, 1e-3).expect("a pole");
        assert!(point_in_region(&ring, pole), "{pole:?} is in the hole");
        // Comfortably more than the 2 the plain 4-wide band offers: the roomiest spot backs away
        // DIAGONALLY from one of the hole's corners, where the clearance is the corner distance.
        assert!((clearance - 2.343).abs() < 0.05, "{clearance}");
    }

    /// A loop that encloses nothing has no interior to name.
    #[test]
    fn a_degenerate_loop_has_no_pole() {
        assert!(deepest_interior_point(&[], 1e-3).is_none());
        assert!(
            deepest_interior_point(&fill(&[[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]), 1e-3).is_none()
        );
    }
}
