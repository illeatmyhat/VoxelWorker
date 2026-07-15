//! # geom2d — planar computational-geometry predicates
//!
//! A small kernel of exact predicates over points in the plane, each a point
//! `[f64; 2]` in some caller-chosen coordinate space. They are pure: no domain
//! type appears, and the polygon is any slice of points. The domain adapter (a
//! sketch profile, a brush stroke) converts its own vertices to `[f64; 2]` and
//! calls in.
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
pub fn point_in_polygon(polygon: &[[f64; 2]], sample: [f64; 2]) -> bool {
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
    let centre = [
        (rect_min[0] + rect_max[0]) * 0.5,
        (rect_min[1] + rect_max[1]) * 0.5,
    ];
    point_in_polygon(polygon, centre)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_SQUARE: [[f64; 2]; 4] = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];

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
        assert!(point_in_polygon(&UNIT_SQUARE, [2.0, 2.0]));
        assert!(!point_in_polygon(&UNIT_SQUARE, [5.0, 2.0]));
        assert!(!point_in_polygon(&UNIT_SQUARE, [-1.0, 2.0]));
        assert!(!point_in_polygon(&[], [0.0, 0.0]));
    }

    #[test]
    fn point_in_polygon_concave_l_shape() {
        // An L: the reflex notch in the upper-right quadrant is OUTSIDE.
        let l_shape = [
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
}
