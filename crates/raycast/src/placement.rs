//! **Where an armed tool would drop its node** — the picked point, its orientation, and the
//! answers a viewport has to be able to give (`docs/design/direct-manipulation.md`,
//! `docs/design/placement-prior-art.md`).
//!
//! ## Position and orientation are two questions
//!
//! * *Position* — **where does the node land?** A geometry hit answers it directly; failing that,
//!   the cursor ray is intersected with a placement plane.
//! * *Orientation* — **which way does it face?** A geometry face and a user-created plane set it;
//!   the built-in world planes do **not** (see below).
//!
//! Blender ships this separation and names it — `Depth: Surface | Cursor Plane` and
//! `Orientation: Surface | Default` are independent dropdowns.
//!
//! ## The three world planes, ground privileged (2026-07-20)
//!
//! The fixed infinite ground plane was abandoned for a view-aligned plane, then that in turn was
//! replaced — the reasoning and prior art are in `placement-prior-art.md`. This app authors
//! architectural assets *for worlds that have a ground*, so a ground plane is authoring in
//! context; the view-aligned plane's wandering normal was the part that felt unpredictable.
//!
//! When the ray hits no geometry it is intersected with one of **three axis-aligned planes through
//! the world origin** — the ground (`z = 0`, normal `+Z`; we are Z-up) and two verticals. The
//! **ground is privileged**: it is chosen across the whole cone where the ray faces it well enough,
//! and a vertical takes over only when the ground *grazes* (see [`select_world_plane`]). The planes
//! are pinned at the origin, not movable — `0,0,0` means `0,0,0`, and a user who wants a plane
//! elsewhere makes their own reference plane.
//!
//! **This cannot graze.** A unit ray direction cannot have all three components small, so the
//! best-faced of three orthogonal planes always has a healthy denominator — the totality the
//! wandering view-aligned normal used to provide, now from three *fixed* normals and a selection
//! (invariant swept in the tests). No horizon flight, nothing to clamp.
//!
//! ## The built-in planes position but never orient — verticality is preserved
//!
//! A node located via any world plane keeps its **world-vertical** orientation. The vertical
//! fallback that catches a grazing ground ray is a *positioning* device; it never tips what is
//! placed. Only a geometry face ([`PlacementTarget::OnSurface`]) or a user-created plane sets
//! orientation. This is Blender's `Orientation: Default` for the built-ins.
//!
//! ## Looking away from all of them
//!
//! Point at the sky and the selected world plane can sit *behind* the ray — there is nothing in
//! front to place on. That is reported as [`PlacementTarget::NoSurface`] ("point toward the
//! ground"), the honest answer, rather than inventing a depth. It is reachable again here where the
//! view-aligned plane had made it unreachable — the deliberate cost of a fixed ground plane, and a
//! rare one, since the geometry path covers you the moment anything is built.
//!
//! ## How far is too far
//!
//! [`resolve_placement`] takes an injected `depth_is_authorable` predicate rather than a
//! camera-level yes/no, and asks it on **both** paths at the depth each landed at — a block face
//! can sit arbitrarily far away and still be hit, and a world plane can be reached at a distance a
//! block is no longer worth authoring at. `camera`'s `depth_is_authorable` is the intended
//! argument (this crate never depends on `camera`).
//!
//! ## The frame
//!
//! Every position and direction here is in the **one render/world frame the caller is already
//! working in**, and travels as a value in that frame rather than being re-derived (ADR 0008).

use glam::Vec3;
use substrate::spatial::Ray;

/// Where an armed tool would place, how it should face, or why it cannot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementTarget {
    /// On existing geometry, with the face the ray entered through. Orientation comes from the
    /// face — its outward normal is an exact `±1` axis vector.
    OnSurface {
        /// The placement point.
        point: Vec3,
        /// The entered face's outward normal, an exact `±1` axis vector.
        face_normal: [i32; 3],
    },
    /// On one of the three built-in world planes, in empty space. **Orientation is world-vertical,
    /// not the plane's normal** — the built-in planes position only (see module docs), so the
    /// caller orients the node upright regardless of which plane caught the ray. `plane` is carried
    /// for the preview/affordance, not for orientation.
    OnWorldPlane {
        /// The placement point.
        point: Vec3,
        /// Which of the three planes the ray landed on.
        plane: WorldPlane,
    },
    /// The ray points away from every world plane — the selected plane is behind the eye, so there
    /// is nothing in front to place against. **Point toward the ground.** Distinct from
    /// [`TooFar`](Self::TooFar): pointing elsewhere fixes this, zooming does not.
    NoSurface,
    /// The depth the cursor resolved to is far enough out that a block there is too small to author
    /// against. **Zoom in.** Asked per hit, on both the geometry and the world-plane paths.
    TooFar,
}

/// One of the three axis-aligned planes through the world origin — the built-in placement planes.
///
/// They are a **positioning device only**: a node located via any of them keeps its world-vertical
/// orientation, so the fallback that catches a grazing ground ray never tips what is placed. That
/// is what preserves verticality (a product value). Only a geometry face or a user-created plane
/// sets orientation; these do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldPlane {
    /// `z = 0`, normal `+Z`. The ground — we are Z-up. Privileged: chosen across the whole cone
    /// where the ray faces it well enough, not merely when it is the best-faced of the three.
    Ground,
    /// `x = 0`, normal `+X`. A vertical fallback, taken when the ground grazes and the ray faces
    /// this plane more squarely than the other vertical.
    VerticalFacingX,
    /// `y = 0`, normal `+Y`. The other vertical fallback.
    VerticalFacingY,
}

impl WorldPlane {
    /// The plane's unit outward normal.
    pub fn normal(self) -> Vec3 {
        match self {
            WorldPlane::Ground => Vec3::Z,
            WorldPlane::VerticalFacingX => Vec3::X,
            WorldPlane::VerticalFacingY => Vec3::Y,
        }
    }
}

/// Choose which world plane an empty-space placement lands on: the **ground**, unless the ray
/// grazes it more shallowly than `min_ground_facing`, in which case whichever **vertical** the ray
/// faces more squarely.
///
/// `min_ground_facing` is `sin(grazing angle)` — the smallest `|ray·Ẑ|` at which the ground is
/// still worth placing on. The ground keeps priority: it is chosen on the entire cone
/// `|ray·Ẑ| >= min_ground_facing`, not merely when it is best-faced. Ties between the two verticals
/// break toward `VerticalFacingX`; the choice is immaterial (their facings are equal at the tie).
///
/// **The invariant this exists to guarantee** (swept by the
/// `the_selected_world_plane_is_always_well_faced` test): for any unit `ray_direction` and any
/// `min_ground_facing` in `[0, 1/√3]`, the returned plane's normal has
/// `|ray_direction · normal| >= min_ground_facing`. So the ray-plane denominator is bounded away
/// from zero by the threshold itself, the intersection is always well-conditioned, and there is no
/// grazing case to clamp. The bound holds because a unit vector cannot have all three components
/// small: when the ground is rejected (`|z| < m`), `x² + y² > 1 − m²`, so the larger of `|x|, |y|`
/// is at least `√((1−m²)/2) >= m` exactly when `m <= 1/√3`.
pub fn select_world_plane(ray_direction: Vec3, min_ground_facing: f32) -> WorldPlane {
    if ray_direction.z.abs() >= min_ground_facing {
        WorldPlane::Ground
    } else if ray_direction.x.abs() >= ray_direction.y.abs() {
        WorldPlane::VerticalFacingX
    } else {
        WorldPlane::VerticalFacingY
    }
}

/// Intersect `ray` with the world `plane` (through the origin). Returns the point and the signed
/// ray parameter `t`; `t < 0` means the plane lies **behind** the ray origin.
///
/// Call only on a plane [`select_world_plane`] chose for this ray, whose denominator the selection
/// bounds away from zero — that is what makes the division safe. With a unit ray direction `t` is
/// the distance from the origin to the point.
pub fn world_plane_hit(ray: Ray, plane: WorldPlane) -> (Vec3, f32) {
    let normal = plane.normal();
    // Plane through the origin, so its offset term is zero: (origin + t·dir)·n = 0.
    let t = -ray.origin.dot(normal) / ray.direction.dot(normal);
    (ray.origin + ray.direction * t, t)
}

/// Resolve the picked point from a geometry hit, the cursor ray, the grazing threshold, and a
/// per-depth authorability predicate.
///
/// `surface` is the geometry hit if the ray found one — its point and the face it entered through.
/// When it is `None` the ray hit nothing and the point is invented on a world plane.
/// `min_ground_facing` is [`select_world_plane`]'s threshold. `depth_is_authorable` is asked at
/// whichever depth the answer landed at; `camera::OrbitCamera::depth_is_authorable` is the intended
/// argument.
///
/// Custom (user-created) planes are a future second tier between geometry and the world planes;
/// they are not wired here yet.
pub fn resolve_placement(
    surface: Option<(Vec3, [i32; 3])>,
    ray: Ray,
    min_ground_facing: f32,
    depth_is_authorable: impl Fn(f32) -> bool,
) -> PlacementTarget {
    // Work with a unit direction so `t` is a distance and the facing threshold is in true sine.
    let ray = Ray::new(ray.origin, ray.direction.normalize());

    // Tier 1 — geometry. Clicking a visible surface is unambiguous; nothing moves it.
    if let Some((point, face_normal)) = surface {
        let depth = (point - ray.origin).dot(ray.direction);
        return if depth_is_authorable(depth) {
            PlacementTarget::OnSurface { point, face_normal }
        } else {
            PlacementTarget::TooFar
        };
    }

    // Tier 3 — the built-in world planes. (Tier 2, custom planes, is not wired yet.)
    let plane = select_world_plane(ray.direction, min_ground_facing);
    let (point, depth) = world_plane_hit(ray, plane);
    if depth <= 0.0 {
        // The selected plane is behind the ray: pointing at the sky, nothing in front.
        return PlacementTarget::NoSurface;
    }
    if !depth_is_authorable(depth) {
        return PlacementTarget::TooFar;
    }
    PlacementTarget::OnWorldPlane { point, plane }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: [f32; 3], direction: [f32; 3]) -> Ray {
        Ray::new(Vec3::from(origin), Vec3::from(direction).normalize())
    }

    /// Everything is authorable, for the tests that are not about the limit.
    fn anywhere(_depth: f32) -> bool {
        true
    }

    /// A default-ish grazing threshold: `sin(20°) ≈ 0.342`.
    const GRAZE: f32 = 0.342;

    /// **The three-plane totality invariant, swept.** For any view direction and any grazing
    /// threshold up to `1/√3`, the plane `select_world_plane` returns is faced by at least the
    /// threshold — so the ray-plane denominator can never collapse and there is no grazing case to
    /// clamp. This is the property that lets three *fixed* normals replace the wandering
    /// view-aligned one. Swept over a fine grid of unit directions; the algebra is in
    /// `select_world_plane`'s docs. (A Kani harness would state it over the whole sphere, but the
    /// bound is a squares-only algebraic fact and the sweep is decisive and in-gate.)
    #[test]
    fn the_selected_world_plane_is_always_well_faced() {
        // 1/√3 is the largest threshold the guarantee holds for; test at and below it.
        for &min_ground_facing in &[0.1_f32, 0.342, 0.5, 0.577] {
            let mut worst_facing = f32::INFINITY;
            for yaw_step in 0..360 {
                for pitch_step in -89..=89 {
                    let yaw = (yaw_step as f32).to_radians();
                    let pitch = (pitch_step as f32).to_radians();
                    let direction = Vec3::new(
                        pitch.cos() * yaw.cos(),
                        pitch.cos() * yaw.sin(),
                        pitch.sin(),
                    )
                    .normalize();
                    let chosen = select_world_plane(direction, min_ground_facing);
                    let facing = direction.dot(chosen.normal()).abs();
                    worst_facing = worst_facing.min(facing);
                    assert!(
                        facing >= min_ground_facing - 1e-6,
                        "plane {chosen:?} faced only {facing} at yaw {yaw_step} pitch {pitch_step}, \
                         threshold {min_ground_facing}"
                    );
                }
            }
            // And the guarantee is tight: some direction sits right at the threshold.
            assert!(
                worst_facing <= min_ground_facing + 0.02,
                "bound is looser than claimed: worst {worst_facing} vs {min_ground_facing}"
            );
        }
    }

    /// The ground is **privileged**, not merely best-faced: a ray that faces a vertical more
    /// squarely than the ground still lands on the ground as long as the ground is not grazing.
    #[test]
    fn the_ground_wins_whenever_it_is_not_grazing() {
        // Looking down fairly steeply but with a strong sideways component: |z| beats the
        // threshold though |x| is larger. Ground must still win.
        let direction = Vec3::new(0.8, 0.0, -0.4).normalize();
        assert!(direction.x.abs() > direction.z.abs(), "set up so a vertical is better-faced");
        assert_eq!(select_world_plane(direction, GRAZE), WorldPlane::Ground);
        // Now graze the ground (nearly horizontal): a vertical must take over.
        let grazing = Vec3::new(0.98, 0.1, -0.05).normalize();
        assert_eq!(select_world_plane(grazing, GRAZE), WorldPlane::VerticalFacingX);
    }

    /// The ordinary empty-space case: looking down at the ground places on it, under the cursor,
    /// upright. The reported plane is the ground; orientation (world-vertical) is the caller's.
    #[test]
    fn an_empty_space_ray_places_on_the_ground_under_the_cursor() {
        // Eye above the ground looking straight down; cursor at world (5, 7).
        let cursor = ray([5.0, 7.0, 50.0], [0.0, 0.0, -1.0]);
        match resolve_placement(None, cursor, GRAZE, anywhere) {
            PlacementTarget::OnWorldPlane { point, plane } => {
                assert_eq!(plane, WorldPlane::Ground);
                assert!((point - Vec3::new(5.0, 7.0, 0.0)).length() < 1e-4, "landed at {point}");
            }
            other => panic!("expected OnWorldPlane(Ground), got {other:?}"),
        }
    }

    /// A grazing ground ray falls back to a vertical plane rather than flying to a horizon, and
    /// lands on that plane in front of the eye.
    #[test]
    fn a_grazing_ray_falls_back_to_a_vertical_plane() {
        // Nearly horizontal, aimed mostly along +X so the x=0 plane is the one it faces.
        let cursor = ray([-30.0, 4.0, 6.0], [1.0, 0.05, -0.04]);
        match resolve_placement(None, cursor, GRAZE, anywhere) {
            PlacementTarget::OnWorldPlane { point, plane } => {
                assert_eq!(plane, WorldPlane::VerticalFacingX);
                assert!(point.x.abs() < 1e-3, "not on the x=0 plane: {point}");
                assert!(point.is_finite(), "flew off: {point}");
            }
            other => panic!("expected OnWorldPlane(VerticalFacingX), got {other:?}"),
        }
    }

    /// **Pointing at the sky is `NoSurface`, not an invented depth.** Eye above the ground looking
    /// up: the ground is behind, so there is nothing in front to place on.
    #[test]
    fn pointing_away_from_every_plane_is_no_surface() {
        let skyward = ray([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
        assert_eq!(resolve_placement(None, skyward, GRAZE, anywhere), PlacementTarget::NoSurface);
    }

    /// A geometry hit is reported exactly, normal and all — the surface is where it is, and
    /// clicking a visible one is unambiguous intent.
    #[test]
    fn a_geometry_hit_is_reported_exactly() {
        let cursor = ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let hit = Vec3::new(0.0, 0.0, 2.0);
        assert_eq!(
            resolve_placement(Some((hit, [0, 0, 1])), cursor, GRAZE, anywhere),
            PlacementTarget::OnSurface { point: hit, face_normal: [0, 0, 1] }
        );
    }

    /// **Too-far is answered per hit.** A face far down the ray is still sub-pixel and not worth
    /// authoring against, so a distant geometry hit reports `TooFar` while a near one places.
    #[test]
    fn a_geometry_hit_beyond_the_authorable_depth_is_too_far() {
        let cursor = ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let within_100 = |depth: f32| depth <= 100.0;
        let near = Vec3::new(0.0, 0.0, -20.0); // depth 30 from the eye
        let far = Vec3::new(0.0, 0.0, -900.0); // depth 910 from the eye
        assert_eq!(
            resolve_placement(Some((near, [0, 0, 1])), cursor, GRAZE, within_100),
            PlacementTarget::OnSurface { point: near, face_normal: [0, 0, 1] }
        );
        assert_eq!(
            resolve_placement(Some((far, [0, 0, 1])), cursor, GRAZE, within_100),
            PlacementTarget::TooFar,
            "a hit past the limit is too far"
        );
    }

    /// **The world plane is too-far exactly when its own depth is.** Looking down from far out
    /// past the limit reports `TooFar` ("zoom in"); from within it, a placement.
    #[test]
    fn the_world_plane_is_too_far_when_its_depth_exceeds_the_limit() {
        let within_100 = |depth: f32| depth <= 100.0;
        let near = ray([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        assert!(matches!(
            resolve_placement(None, near, GRAZE, within_100),
            PlacementTarget::OnWorldPlane { .. }
        ));
        let far = ray([0.0, 0.0, 4000.0], [0.0, 0.0, -1.0]);
        assert_eq!(resolve_placement(None, far, GRAZE, within_100), PlacementTarget::TooFar);
    }

    /// **The distinction the viewport hangs on.** `NoSurface` ("point toward the ground") and
    /// `TooFar` ("zoom in") name different corrective actions, so they must be different values —
    /// and both differ from a placement. The pair the old view-aligned plane had made unreachable
    /// (`NoSurface`) is reachable again under the fixed ground plane.
    #[test]
    fn the_three_negative_and_placed_answers_are_all_distinct() {
        let skyward = ray([0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
        let downward = ray([0.0, 0.0, 40.0], [0.0, 0.0, -1.0]);
        let no_surface = resolve_placement(None, skyward, GRAZE, anywhere);
        let placed = resolve_placement(None, downward, GRAZE, anywhere);
        let too_far = resolve_placement(None, downward, GRAZE, |_| false);
        assert_eq!(no_surface, PlacementTarget::NoSurface);
        assert_eq!(too_far, PlacementTarget::TooFar);
        assert!(matches!(placed, PlacementTarget::OnWorldPlane { .. }));
        assert_ne!(no_surface, too_far);
        assert_ne!(placed, too_far);
        assert_ne!(placed, no_surface);
    }
}
