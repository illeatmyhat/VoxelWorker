//! A parametric ray and the slab-method ray–box intersection.
//!
//! `Ray` is the textbook parametric half-line `p(t) = origin + t · direction`,
//! `t >= 0`. It is plain geometry over `glam::Vec3` and names nothing about who
//! cast it; a camera unprojection produces one and a volume traversal consumes
//! one, and neither has to know about the other.
//!
//! ## Ray–box intersection: the slab method
//!
//! [`Ray::intersect_box_slab`] tests the ray against an axis-aligned box given by
//! its two float corners and returns the entry/exit parameters `(t_enter, t_exit)`
//! of the overlapped segment. It is the **slab method** (Kay & Kajiya, "Ray Tracing
//! Complex Scenes", SIGGRAPH 1986; Ericson, *Real-Time Collision Detection* 2005
//! §5.3.3): an AABB is the intersection of three axis-aligned slabs, so the ray's
//! interval inside the box is the intersection of its three per-axis slab
//! intervals. For each axis the two slab planes are hit at
//! `t = (corner - origin) / direction`; the near/far pair is the ordered
//! `(min, max)` of those two, and the box interval is
//! `t_enter = max over axes of the nears` (clamped to `0` so a ray that starts
//! inside the box enters at `t = 0`), `t_exit = min over axes of the fars`. The
//! ray hits the box iff `t_exit >= t_enter`.
//!
//! ## Zero direction components (the load-bearing robustness detail)
//!
//! A direction component of exactly `0` makes the ray parallel to that pair of
//! slab planes, and `(corner - origin) / 0` is an IEEE infinity. Williams, Barrs,
//! Morley & Shirley ("An Efficient and Robust Ray–Box Intersection Algorithm",
//! JGT 2005) note the genuine hazard is not the infinities themselves — a signed
//! infinity still orders correctly under `min`/`max` — but the `0 · inf = NaN`
//! that appears when the origin lies *exactly on* a slab plane (`corner - origin`
//! is `0` on an axis whose reciprocal is infinite), since `NaN` silently defeats
//! every subsequent comparison.
//!
//! This implementation follows the variant its GPU shader mirror uses (see the
//! ray–volume traversal chapter of `docs/architecture`): rather than form true
//! infinities and special-case the `NaN`, it **nudges any near-zero direction
//! component to a tiny positive magnitude** ([`SLAB_ZERO_DIRECTION_GUARD`]) before
//! taking the reciprocal. The reciprocal is then a large but finite number, so
//! `(corner - origin) · huge_finite` stays finite for every origin: a ray parallel
//! to a slab and *outside* it yields a huge-magnitude interval of the correct sign
//! that forces `t_exit < t_enter` (a miss), while one parallel and *inside* the
//! slab yields a huge interval that does not constrain the result — and an origin
//! sitting on the plane produces a clean `0` rather than a `NaN`. Reproducing this
//! exact arithmetic (guard-then-reciprocal, `max(…, 0)` on entry) is what lets the
//! CPU traversal march and its WGSL mirror stay byte-comparable under the parity
//! suite.

use glam::Vec3;

/// The magnitude any near-zero ray-direction component is nudged to before the
/// slab test takes its reciprocal, so a component of exactly `0` becomes a large
/// finite reciprocal instead of an IEEE infinity (which would then risk a
/// `0 · inf = NaN` when the origin lies on a slab plane). See the module docs for
/// why the guard is preferred over the true-infinity Williams et al. variant.
pub const SLAB_ZERO_DIRECTION_GUARD: f32 = 1e-20;

/// A parametric ray `p(t) = origin + t · direction`. The direction is stored as
/// handed in (callers that need a unit ray normalize before constructing); the
/// slab test does not assume it is normalized, but the returned `t` values are in
/// units of `direction`'s length when it is not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// The ray's start point `p(0)`.
    pub origin: Vec3,
    /// The ray's direction. Not required to be unit length.
    pub direction: Vec3,
}

/// The parameter interval a [`Ray`] spends inside an axis-aligned box: it enters at
/// `t_enter` and leaves at `t_exit`, with `t_enter <= t_exit`. `t_enter` is clamped
/// to be non-negative, so a ray that starts inside the box reports `t_enter == 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayBoxIntersection {
    /// Parameter at which the ray enters the box (`0` if it starts inside).
    pub t_enter: f32,
    /// Parameter at which the ray exits the box.
    pub t_exit: f32,
}

impl Ray {
    /// A ray from `origin` along `direction`.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// The componentwise reciprocal of the direction, with any component whose
    /// magnitude is below [`SLAB_ZERO_DIRECTION_GUARD`] first nudged up to that
    /// guard so the reciprocal stays finite (see the module docs). Exposed so a
    /// traversal that reuses the same reciprocal for its stepping seeds derives it
    /// identically to the slab test.
    pub fn slab_inverse_direction(&self) -> Vec3 {
        let guard = |component: f32| -> f32 {
            if component.abs() < SLAB_ZERO_DIRECTION_GUARD {
                SLAB_ZERO_DIRECTION_GUARD
            } else {
                component
            }
        };
        Vec3::new(
            guard(self.direction.x),
            guard(self.direction.y),
            guard(self.direction.z),
        )
        .recip()
    }

    /// Intersect the ray with the axis-aligned box `[box_min, box_max]` by the slab
    /// method, returning the entry/exit parameters of the overlapped segment, or
    /// `None` if the ray misses the box. `t_enter` is clamped to `0`, so a ray
    /// starting inside the box enters at `t = 0`. Zero direction components are
    /// handled by the guard described in the module docs (no `NaN`).
    pub fn intersect_box_slab(&self, box_min: Vec3, box_max: Vec3) -> Option<RayBoxIntersection> {
        let inverse = self.slab_inverse_direction();
        let t_a = (box_min - self.origin) * inverse;
        let t_b = (box_max - self.origin) * inverse;
        let t_near = t_a.min(t_b);
        let t_far = t_a.max(t_b);
        let t_enter = t_near.x.max(t_near.y).max(t_near.z).max(0.0);
        let t_exit = t_far.x.min(t_far.y).min(t_far.z);
        if t_exit < t_enter {
            None
        } else {
            Some(RayBoxIntersection { t_enter, t_exit })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT_BOX_MIN: Vec3 = Vec3::ZERO;
    const UNIT_BOX_MAX: Vec3 = Vec3::ONE;

    /// A ray fired straight through the middle of the unit box enters at the near
    /// face and exits at the far face, both parameters finite and ordered.
    #[test]
    fn hits_through_the_middle() {
        let ray = Ray::new(Vec3::new(-5.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hit = ray.intersect_box_slab(UNIT_BOX_MIN, UNIT_BOX_MAX).unwrap();
        assert_eq!(hit.t_enter, 5.0);
        assert_eq!(hit.t_exit, 6.0);
    }

    /// A ray parallel to the X slab but offset off the box in Y misses: the Y slab
    /// interval never overlaps the X one, forcing `t_exit < t_enter`.
    #[test]
    fn misses_when_parallel_and_outside() {
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.5), Vec3::new(1.0, 0.0, 0.0));
        assert!(ray.intersect_box_slab(UNIT_BOX_MIN, UNIT_BOX_MAX).is_none());
    }

    /// A ray grazing along the box's lower edge (origin on the Y=0 and Z=0 slab
    /// planes, direction parallel to both) still intersects — the guard keeps the
    /// on-plane `0 · huge` finite at `0` rather than producing a `NaN` that would
    /// defeat the comparison.
    #[test]
    fn edge_parallel_ray_grazes_without_nan() {
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = ray.intersect_box_slab(UNIT_BOX_MIN, UNIT_BOX_MAX).unwrap();
        assert_eq!(hit.t_enter, 5.0);
        assert_eq!(hit.t_exit, 6.0);
    }

    /// A ray whose origin is inside the box enters at `t = 0` (the `max(…, 0)`
    /// clamp) and exits at the far face.
    #[test]
    fn ray_starting_inside_enters_at_zero() {
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let hit = ray.intersect_box_slab(UNIT_BOX_MIN, UNIT_BOX_MAX).unwrap();
        assert_eq!(hit.t_enter, 0.0);
        assert_eq!(hit.t_exit, 0.5);
    }

    /// A fully degenerate zero direction (parallel to all three slab pairs) never
    /// yields a `NaN`: the guard makes every reciprocal finite, so an origin inside
    /// the box still returns a finite, ordered interval starting at `0`.
    #[test]
    fn degenerate_zero_direction_never_nans() {
        let ray = Ray::new(Vec3::new(0.5, 0.5, 0.5), Vec3::ZERO);
        let hit = ray.intersect_box_slab(UNIT_BOX_MIN, UNIT_BOX_MAX).unwrap();
        assert!(hit.t_enter.is_finite() && hit.t_exit.is_finite());
        assert_eq!(hit.t_enter, 0.0);
        assert!(hit.t_exit >= hit.t_enter);
    }
}
