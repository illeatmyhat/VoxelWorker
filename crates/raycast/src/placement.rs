//! **Where an armed tool would drop its node** — the picked point, and the four answers a
//! viewport has to be able to give (`docs/design/direct-manipulation.md`).
//!
//! The grammar says the picked point is *the nearer of the ray's hit on existing geometry and
//! its hit on the ground plane*. The geometry half is bounded by the geometry. **The ground
//! half is not**: the plane is infinite, so a ray a few degrees off level meets it hundreds of
//! blocks away, and a click there drops a node somewhere absurd. That asymmetry is the whole
//! reason this module exists.
//!
//! ## Why clamping the ray parameter is wrong
//!
//! The obvious fix — cap `t` along the ray — produces a point that is no longer **on the
//! plane**. A preview floating above the ground is a worse answer than one that is merely far
//! away, because it is not even a legal placement.
//!
//! So the clamp runs *along the ground*: keep the plane, slide the point toward the anchor the
//! user is working around until it is exactly at the limit. [`clamp_along_plane`] solves that
//! in closed form rather than stepping toward it.
//!
//! ## Four answers, and two of them are different messages
//!
//! [`PlacementTarget`] distinguishes *nothing to hit* from *too far to author*. Both draw no
//! preview, and collapsing them into one greyed-out cursor would be a real loss: the first
//! means **point at something**, the second means **zoom in**. Those are different actions, and
//! a viewport that cannot say which is which leaves the user to guess.
//!
//! A geometry hit is never clamped. Clicking a surface you can see is unambiguous intent, and
//! the surface is wherever it is; the clamp exists for the unbounded case only.

use glam::Vec3;
use substrate::spatial::Ray;

/// Where an armed tool would place, or why it cannot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementTarget {
    /// On existing geometry, with the face the ray entered through. Never clamped — the
    /// surface is where it is, and clicking a visible one is unambiguous.
    OnSurface {
        /// The placement point.
        point: Vec3,
        /// The entered face's outward normal, an exact `±1` axis vector. Tools that orient to
        /// the surface (the sketch plane) read this.
        face_normal: [i32; 3],
    },
    /// On the ground plane, within the authorable limit.
    OnGround {
        /// The placement point.
        point: Vec3,
    },
    /// On the ground plane, pulled back along it to the authorable limit. Still a legal
    /// placement — the viewport should show the preview AND mark that it was clamped, or the
    /// user will not understand why it stopped following the cursor.
    Clamped {
        /// The placement point, on the plane and exactly at the limit.
        point: Vec3,
    },
    /// The ray meets neither geometry nor the ground — it is pointing at the sky, or lies
    /// parallel to the plane. **Point at something.**
    NoSurface,
    /// The camera is far enough out that a block is too small to author against, so nothing
    /// the cursor could reach is placeable. **Zoom in.** Distinct from
    /// [`NoSurface`](Self::NoSurface) on purpose.
    TooFar,
}

/// Where `ray` meets the horizontal plane at height `plane_height`, or `None` when it never
/// does — pointing away from the plane, or lying parallel to it.
///
/// The parallel case is a genuine miss rather than a hit at infinity, and it is reported as
/// such so the caller can say "point at something" instead of placing at an absurd distance.
/// The `t > 0` requirement discards the plane *behind* the eye.
pub fn ground_plane_hit(ray: Ray, plane_height: f32) -> Option<Vec3> {
    let direction_z = ray.direction.z;
    // Exactly parallel never meets the plane; near-parallel meets it so far away that the
    // authorable limit would reject it anyway, so no separate epsilon is needed here.
    if direction_z == 0.0 {
        return None;
    }
    let t = (plane_height - ray.origin.z) / direction_z;
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    Some(ray.origin + ray.direction * t)
}

/// Slide `point` along its plane toward `anchor` until it is exactly `limit` from `eye`,
/// returning `None` when it is already inside the limit.
///
/// Closed form: with `offset = point − eye` and `direction = anchor − point`, solve
/// `|offset + s·direction|² = limit²` for the smallest `s ∈ [0, 1]`. Both `point` and `anchor`
/// lie on the plane, so every point on that segment does too — which is the property the naive
/// "cap `t` along the view ray" fix destroys.
///
/// `None` also covers the degenerate cases: an anchor that is itself outside the limit, and an
/// anchor coincident with the point. Callers treat that as "no legal clamp" rather than
/// inventing a position.
pub fn clamp_along_plane(point: Vec3, anchor: Vec3, eye: Vec3, limit: f32) -> Option<Vec3> {
    let offset = point - eye;
    if offset.length() <= limit {
        return None;
    }
    let direction = anchor - point;
    let a = direction.length_squared();
    if a <= f32::EPSILON {
        return None;
    }
    let b = 2.0 * offset.dot(direction);
    let c = offset.length_squared() - limit * limit;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    // The smaller root is the first crossing walking from `point` toward `anchor`, i.e. the
    // furthest legal placement — which is the one to keep.
    let smaller = (-b - root) / (2.0 * a);
    let larger = (-b + root) / (2.0 * a);
    let s = [smaller, larger].into_iter().find(|s| (0.0..=1.0).contains(s))?;
    Some(point + direction * s)
}

/// Resolve the four answers from a geometry hit, a ray, and the camera's authorable limit.
///
/// `surface` is the geometry hit if the ray found one (its point and entered face). `anchor` is
/// the point the camera orbits, projected onto the plane — the clamp slides toward it because
/// that is where the user is working.
pub fn resolve_placement(
    surface: Option<(Vec3, [i32; 3])>,
    ray: Ray,
    eye: Vec3,
    anchor_on_plane: Vec3,
    plane_height: f32,
    authorable_limit: f32,
    camera_can_author: bool,
) -> PlacementTarget {
    // Asked first, because it does not depend on the ray: if the point the camera orbits is
    // itself too far to work at, nothing nearer the cursor can be either.
    if !camera_can_author {
        return PlacementTarget::TooFar;
    }
    if let Some((point, face_normal)) = surface {
        return PlacementTarget::OnSurface { point, face_normal };
    }
    let Some(ground) = ground_plane_hit(ray, plane_height) else {
        return PlacementTarget::NoSurface;
    };
    match clamp_along_plane(ground, anchor_on_plane, eye, authorable_limit) {
        None if (ground - eye).length() <= authorable_limit => PlacementTarget::OnGround { point: ground },
        // Beyond the limit with no legal clamp — the anchor could not rescue it.
        None => PlacementTarget::NoSurface,
        Some(point) => PlacementTarget::Clamped { point },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMIT: f32 = 100.0;

    fn ray(origin: [f32; 3], direction: [f32; 3]) -> Ray {
        Ray::new(Vec3::from(origin), Vec3::from(direction).normalize())
    }

    /// A ray parallel to the plane MISSES rather than hitting at infinity. This is the
    /// grazing case that motivated the module: reported as a miss, the viewport says "point at
    /// something"; reported as a hit, it would place at an absurd distance.
    #[test]
    fn a_parallel_ray_misses_the_ground() {
        assert_eq!(ground_plane_hit(ray([0.0, 0.0, 10.0], [1.0, 0.0, 0.0]), 0.0), None);
        // Pointing away from the plane is also a miss, not a hit behind the eye.
        assert_eq!(ground_plane_hit(ray([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]), 0.0), None);
    }

    /// **The property the naive fix breaks.** A clamped point must still be ON the plane —
    /// capping the ray parameter instead would leave the preview floating above the ground,
    /// which is not a legal placement at all.
    #[test]
    fn a_clamped_point_stays_on_the_plane_and_lands_at_the_limit() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let far = Vec3::new(5000.0, 0.0, 0.0);
        let anchor = Vec3::ZERO;
        let clamped = clamp_along_plane(far, anchor, eye, LIMIT).expect("a clamp exists");
        assert!(clamped.z.abs() < 1e-3, "left the plane: z = {}", clamped.z);
        assert!(
            ((clamped - eye).length() - LIMIT).abs() < 1e-2,
            "landed at {} from the eye, expected {LIMIT}",
            (clamped - eye).length()
        );
    }

    /// A point already inside the limit is not moved — the clamp only ever pulls back.
    #[test]
    fn a_point_inside_the_limit_is_untouched() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let near = Vec3::new(20.0, 0.0, 0.0);
        assert_eq!(clamp_along_plane(near, Vec3::ZERO, eye, LIMIT), None);
    }

    /// **The distinction the viewport hangs on.** Nothing-to-hit and too-far both draw no
    /// preview, but they are different messages — "point at something" versus "zoom in" — so
    /// they must be different values, not one shared empty state.
    #[test]
    fn nothing_to_hit_and_too_far_are_different_answers() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let skyward = ray([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);

        let nothing =
            resolve_placement(None, skyward, eye, Vec3::ZERO, 0.0, LIMIT, true);
        assert_eq!(nothing, PlacementTarget::NoSurface);

        let too_far =
            resolve_placement(None, skyward, eye, Vec3::ZERO, 0.0, LIMIT, false);
        assert_eq!(too_far, PlacementTarget::TooFar);

        assert_ne!(nothing, too_far, "the two must never collapse into one state");
    }

    /// Too-far is decided before the ray is consulted, so a camera zoomed past the limit
    /// reports TooFar even when the cursor is over solid geometry. Zooming in is the only
    /// thing that helps, and the message should say so rather than pretending the surface is
    /// placeable.
    #[test]
    fn too_far_outranks_a_geometry_hit() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let down = ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let surface = Some((Vec3::ZERO, [0, 0, 1]));
        assert_eq!(
            resolve_placement(surface, down, eye, Vec3::ZERO, 0.0, LIMIT, false),
            PlacementTarget::TooFar
        );
    }

    /// A geometry hit is never clamped, however far away it is: the surface is where it is,
    /// and clicking a visible one is unambiguous intent. The clamp exists for the unbounded
    /// ground plane, not for geometry.
    #[test]
    fn a_geometry_hit_is_never_clamped() {
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let down = ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let distant = Vec3::new(9000.0, 0.0, 0.0);
        assert_eq!(
            resolve_placement(Some((distant, [0, 0, 1])), down, eye, Vec3::ZERO, 0.0, LIMIT, true),
            PlacementTarget::OnSurface { point: distant, face_normal: [0, 0, 1] }
        );
    }

    /// The ordinary case: a downward ray inside the limit places where it lands, unclamped.
    #[test]
    fn a_near_ray_places_where_it_lands() {
        let eye = Vec3::new(0.0, 0.0, 50.0);
        let down = ray([0.0, 0.0, 50.0], [0.0, 0.0, -1.0]);
        match resolve_placement(None, down, eye, Vec3::ZERO, 0.0, LIMIT, true) {
            PlacementTarget::OnGround { point } => {
                assert!(point.z.abs() < 1e-3 && point.length() < 1e-3, "landed at {point}");
            }
            other => panic!("expected OnGround, got {other:?}"),
        }
    }

    /// A grazing ray — the motivating case — clamps rather than flying to the horizon, and the
    /// result is still on the plane and inside the limit.
    #[test]
    fn a_grazing_ray_clamps_instead_of_reaching_the_horizon() {
        let eye = Vec3::new(0.0, 0.0, 20.0);
        let grazing = ray([0.0, 0.0, 20.0], [1.0, 0.0, -0.02]);
        let unclamped = ground_plane_hit(grazing, 0.0).expect("a grazing ray does meet the plane");
        assert!(unclamped.x > 900.0, "the unclamped hit should be far out: {unclamped}");

        match resolve_placement(None, grazing, eye, Vec3::ZERO, 0.0, LIMIT, true) {
            PlacementTarget::Clamped { point } => {
                assert!(point.z.abs() < 1e-3, "left the plane: {point}");
                assert!((point - eye).length() <= LIMIT + 1e-2, "still beyond the limit: {point}");
            }
            other => panic!("expected Clamped, got {other:?}"),
        }
    }
}
