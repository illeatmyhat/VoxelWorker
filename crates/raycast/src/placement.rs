//! **Where an armed tool would drop its node** — the picked point, and the answers a viewport
//! has to be able to give (`docs/design/direct-manipulation.md`).
//!
//! ## Two questions, not one
//!
//! Surface placement and the picking plane look like one problem and are not:
//!
//! * *Surface placement* answers **the ray hit something; where and how does the node sit
//!   against it?** The hit face and its normal answer it, and nothing here can improve on that.
//! * *The picking plane* answers **the ray hit nothing; what depth do I invent?**
//!
//! Conflating them is what made a fixed infinite ground plane look load-bearing. Blender ships
//! the separation and names it in the interface — `Depth: Surface | Cursor Plane` and
//! `Orientation: Surface | Default` are two independent dropdowns. **The fixed ground plane is
//! gone** (decided 2026-07-20; evidence in `docs/design/placement-prior-art.md`). The ground is
//! a surface if something is there and nothing if not.
//!
//! ## The picking plane is perpendicular to the view axis
//!
//! Blender's `ED_view3d_win_to_3d` uses `rv3d->viewinv[2]` — the view axis — as the picking
//! plane's normal, through the user-movable 3D cursor rather than world Z=0. Taking the same
//! shape has one consequence that removes a whole class of code: **the ray-plane denominator is
//! `dot(ray_direction, view_direction)`, and no camera can drive it to zero.** Under perspective
//! every ray lies inside a frustum whose half-angle is well under 90°; under orthographic every
//! ray *is* the view axis and meets the plane perpendicularly.
//!
//! So there is no grazing case, no horizon flight, and nothing to clamp. The distance clamp this
//! module used to carry — which slid a point along an unbounded ground plane toward the orbit
//! pivot, and past its limit produced a dead zone where large mouse movement moved no preview —
//! was solving a problem that the plane choice deletes outright. It is gone, not ported.
//!
//! ## The anchor makes this one primitive rather than two modes
//!
//! [`AnchorPlane`] carries the depth the plane sits at. **A surface hit is expected to update
//! that anchor, and this module does not do it** — it takes the anchor as an argument and has no
//! opinion on where it is stored. Place against a face, drag off its edge into empty space, and
//! the next placement should continue at the depth you were just working at rather than snapping
//! back to the orbit pivot. Anchor precedence, most recent wins: last placement, else the orbit
//! pivot. There is no world-origin term.
//!
//! **The precondition that comes with it:** the anchor must sit at a depth the viewer can
//! actually see into — for the orbit pivot that holds by construction, but a last-placement
//! anchor can end up *behind* the eye after an orbit, and a plane behind the eye would put the
//! preview behind the viewer. That is the anchor policy's problem, not this module's; it is
//! recorded here because this module cannot detect it (under orthographic the ray origin sits on
//! an arbitrary near plane, so the sign of the depth carries no information about the viewer).
//!
//! ## How far is too far
//!
//! [`resolve_placement`] takes an injected `depth_is_authorable` predicate rather than a
//! camera-level yes/no, and asks it on **both** paths at the depth each one landed at. That is
//! the crate's usual injection discipline (this crate never depends on `camera`), and it is
//! also the more honest question: the anchor plane is authorable exactly when its own depth is,
//! while a block face can sit arbitrarily far behind the anchor and still be hit. `camera`'s
//! `depth_is_authorable` is the intended argument, and at the orbit pivot's depth it is
//! identically its `can_author_at_all`.
//!
//! ## The frame
//!
//! Every position and direction here — the ray, the view axis, the anchor — is in the **one
//! render/world frame the caller is already working in**, and travels as a value in that frame
//! rather than being re-derived from anything (ADR 0008, the carry half). `AnchorPlane` exists
//! so the plane's two halves arrive together and cannot be assembled from a height and a guess.

use glam::Vec3;
use substrate::spatial::Ray;

/// Where an armed tool would place, or why it cannot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlacementTarget {
    /// On existing geometry, with the face the ray entered through.
    OnSurface {
        /// The placement point.
        point: Vec3,
        /// The entered face's outward normal, an exact `±1` axis vector. Tools that orient to
        /// the surface (the sketch plane) read this.
        face_normal: [i32; 3],
    },
    /// On the view-aligned plane through the depth anchor — the ray hit nothing, so the depth
    /// was invented at the anchor rather than found.
    OnAnchorPlane {
        /// The placement point.
        point: Vec3,
    },
    /// The depth the cursor resolved to is far enough out that a block there is too small to
    /// author against. **Zoom in.**
    ///
    /// This is per-hit, not per-camera. On the anchor-plane path it fires when the anchor
    /// itself is out of reach; on the geometry path it fires for a face hundreds of blocks
    /// behind the anchor, which is still sub-pixel however unambiguous the click was.
    TooFar,
}

/// The picking plane: perpendicular to the view axis, through a movable depth anchor.
///
/// Both fields are in the caller's render/world frame and arrive together, so the plane cannot
/// be reassembled from a height plus an assumption (ADR 0008).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorPlane {
    /// The unit view axis, pointing from the eye into the scene. This is the plane's normal —
    /// which is the whole reason the intersection below cannot degenerate.
    pub view_direction: Vec3,
    /// The depth anchor the plane passes through: the last placement, else the orbit pivot.
    /// A surface hit is expected to move it, elsewhere.
    pub anchor_point: Vec3,
}

impl AnchorPlane {
    /// The depth of `point` along the view axis, measured from `ray_origin`.
    ///
    /// Under perspective the ray origin is the eye and this is the distance the authorability
    /// rule is quoted in. Under orthographic the rays are parallel and apparent size does not
    /// fall off with depth, so the value is inert there by construction.
    pub fn depth_of(&self, point: Vec3, ray_origin: Vec3) -> f32 {
        (point - ray_origin).dot(self.view_direction)
    }
}

/// Where `ray` meets the view-aligned plane through the anchor. **Total** — every camera ray
/// meets it.
///
/// The denominator is `dot(ray_direction, view_direction)`, which a perspective frustum bounds
/// away from zero (its half-angle is far below 90°) and which orthographic pins at exactly 1.
/// There is therefore no miss to report, no parallel case, and no `Option` for a caller to
/// handle — a state no input can reach would be a lie in the type.
///
/// Two deliberate non-guards. **The sign of `t` is not tested:** under orthographic the ray
/// origin sits on an arbitrary near plane, so a negative parameter says something about where
/// the caller chose to start the ray and nothing about the viewer — Blender's orthographic
/// branch likewise takes a signed `ray_point_factor_v3` with no forward test. And the
/// **mathematically** degenerate case, a ray exactly perpendicular to the view axis, is not
/// producible by any camera; should one arrive, the anchor is itself a legal point on the plane
/// and is the depth reference the plane was built from, which is also what Blender does in
/// orthographic when the invisible axis cannot be resolved: leave the coordinate unchanged.
pub fn anchor_plane_hit(ray: Ray, plane: AnchorPlane) -> Vec3 {
    let denominator = ray.direction.dot(plane.view_direction);
    if denominator == 0.0 {
        return plane.anchor_point;
    }
    let t = (plane.anchor_point - ray.origin).dot(plane.view_direction) / denominator;
    if !t.is_finite() {
        return plane.anchor_point;
    }
    ray.origin + ray.direction * t
}

/// Resolve the picked point from a geometry hit, the cursor ray, the picking plane, and a
/// per-depth authorability predicate.
///
/// `surface` is the geometry hit if the ray found one — its point and the face it entered
/// through. When it is `None` the ray hit nothing and the depth is invented on `plane`.
///
/// `depth_is_authorable` is asked at whichever depth the answer landed at, on both paths;
/// `camera::OrbitCamera::depth_is_authorable` is the intended argument.
pub fn resolve_placement(
    surface: Option<(Vec3, [i32; 3])>,
    ray: Ray,
    plane: AnchorPlane,
    depth_is_authorable: impl Fn(f32) -> bool,
) -> PlacementTarget {
    let (point, face_normal) = match surface {
        Some((point, face_normal)) => (point, Some(face_normal)),
        None => (anchor_plane_hit(ray, plane), None),
    };
    if !depth_is_authorable(plane.depth_of(point, ray.origin)) {
        return PlacementTarget::TooFar;
    }
    match face_normal {
        Some(face_normal) => PlacementTarget::OnSurface { point, face_normal },
        None => PlacementTarget::OnAnchorPlane { point },
    }
}

// ============================================================================================
// Three-world-plane model (2026-07-20, `placement-prior-art.md` "Superseded" section).
//
// The view-aligned `AnchorPlane` above is being replaced by three axis-aligned world planes,
// the ground privileged. This block carries the piece that is settled and depends only on the
// ray *direction* — which plane the empty-space placement lands on. The intersection, the
// orientation split, and the behind-the-eye/"nothing in front" residual case are wired into a
// new `resolve_placement` once those decisions land, at which point the `AnchorPlane` path above
// is deleted, not kept.
// ============================================================================================

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
/// **The invariant this exists to guarantee** (swept by the `the_selected_world_plane_is_always_well_faced`
/// test): for any unit `ray_direction` and any `min_ground_facing` in `[0, 1/√3]`, the returned plane's normal has
/// `|ray_direction · normal| >= min_ground_facing`. So the ray-plane denominator is bounded away
/// from zero by the threshold itself, the intersection is always well-conditioned, and there is no
/// grazing case to clamp — the property the wandering view-aligned normal used to provide, now from
/// three *fixed* normals. The bound holds because a unit vector cannot have all three components
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ray(origin: [f32; 3], direction: [f32; 3]) -> Ray {
        Ray::new(Vec3::from(origin), Vec3::from(direction).normalize())
    }

    fn plane(view_direction: [f32; 3], anchor_point: [f32; 3]) -> AnchorPlane {
        AnchorPlane {
            view_direction: Vec3::from(view_direction).normalize(),
            anchor_point: Vec3::from(anchor_point),
        }
    }

    /// Everything is authorable, for the tests that are not about the limit.
    fn anywhere(_depth: f32) -> bool {
        true
    }

    /// **The grazing case cannot arise.** Its predecessor asserted that a ray parallel to the
    /// ground plane missed, and that a near-parallel one flew to the horizon and had to be
    /// clamped back. Against a plane perpendicular to the view axis there is no such ray: over
    /// a sweep far wider than any real frustum the denominator stays bounded away from zero,
    /// every ray meets the plane, and every result lands exactly at the anchor's depth. That is
    /// the property that deleted the clamp.
    #[test]
    fn no_camera_ray_can_graze_the_anchor_plane() {
        let view = Vec3::new(0.0, 1.0, 0.0);
        let picking_plane = plane([0.0, 1.0, 0.0], [0.0, 40.0, 0.0]);
        let eye = Vec3::ZERO;
        // ±60° off-axis in two directions — a 120° field, roughly twice the widest sane one.
        for horizontal in -60..=60 {
            for vertical in -60..=60 {
                let yaw = (horizontal as f32).to_radians();
                let pitch = (vertical as f32).to_radians();
                let direction =
                    Vec3::new(yaw.sin(), yaw.cos() * pitch.cos(), pitch.sin()).normalize();
                let denominator = direction.dot(view);
                // Worst case is the corner, a fifth after normalisation — small, but bounded
                // away from zero, and that is with a field twice as wide as any real one.
                assert!(denominator > 0.19, "denominator collapsed to {denominator}");
                let point = anchor_plane_hit(Ray::new(eye, direction), picking_plane);
                assert!(point.is_finite(), "flew to infinity at {yaw}/{pitch}");
                assert!(
                    (picking_plane.depth_of(point, eye) - 40.0).abs() < 1e-2,
                    "left the plane: {point}"
                );
            }
        }
    }

    /// The ordinary empty-space case: the ray hits nothing and lands on the plane at the
    /// anchor's depth, directly under the cursor.
    #[test]
    fn an_empty_space_ray_places_on_the_anchor_plane() {
        let cursor = ray([0.0, 0.0, 50.0], [0.0, 0.0, -1.0]);
        let picking_plane = plane([0.0, 0.0, -1.0], [0.0, 0.0, 10.0]);
        match resolve_placement(None, cursor, picking_plane, anywhere) {
            PlacementTarget::OnAnchorPlane { point } => {
                assert!((point - Vec3::new(0.0, 0.0, 10.0)).length() < 1e-3, "landed at {point}");
            }
            other => panic!("expected OnAnchorPlane, got {other:?}"),
        }
    }

    /// **Orthographic is the case the old ground plane got wrong.** Every ray is the view axis,
    /// so the intersection is a perpendicular one and the answer is the cursor's own lateral
    /// position at the anchor's depth — clicking empty space leaves the depth coordinate
    /// exactly where it was, which is Blender's documented Front-Ortho behaviour.
    #[test]
    fn an_orthographic_ray_lands_at_the_anchor_depth_under_the_cursor() {
        let view = [0.0, 1.0, 0.0];
        let picking_plane = plane(view, [0.0, 25.0, 0.0]);
        // Parallel rays, offset laterally, starting from an arbitrary near plane.
        for lateral in [-30.0_f32, 0.0, 7.5] {
            let cursor = ray([lateral, -100.0, 3.0], view);
            match resolve_placement(None, cursor, picking_plane, anywhere) {
                PlacementTarget::OnAnchorPlane { point } => {
                    assert!((point.x - lateral).abs() < 1e-3, "lateral drift: {point}");
                    assert!((point.z - 3.0).abs() < 1e-3, "lateral drift: {point}");
                    assert!((point.y - 25.0).abs() < 1e-3, "left the anchor depth: {point}");
                }
                other => panic!("expected OnAnchorPlane, got {other:?}"),
            }
        }
    }

    /// A geometry hit is reported exactly, normal and all — the surface is where it is, and
    /// clicking a visible one is unambiguous intent. Nothing slides it toward the anchor.
    #[test]
    fn a_geometry_hit_is_reported_exactly() {
        let cursor = ray([0.0, 0.0, 10.0], [0.0, 0.0, -1.0]);
        let picking_plane = plane([0.0, 0.0, -1.0], [0.0, 0.0, 0.0]);
        let hit = Vec3::new(0.0, 0.0, 2.0);
        assert_eq!(
            resolve_placement(Some((hit, [0, 0, 1])), cursor, picking_plane, anywhere),
            PlacementTarget::OnSurface { point: hit, face_normal: [0, 0, 1] }
        );
    }

    /// **Too-far is now answered per hit, not per camera.** A face far behind the anchor is
    /// still sub-pixel and still not worth authoring against, so the angular rule keeps a job on
    /// the geometry path — the one path where the ray can reach a depth the anchor does not
    /// bound.
    #[test]
    fn a_geometry_hit_beyond_the_authorable_depth_is_too_far() {
        let cursor = ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let picking_plane = plane([0.0, 1.0, 0.0], [0.0, 40.0, 0.0]);
        let near = Vec3::new(0.0, 30.0, 0.0);
        let far = Vec3::new(0.0, 900.0, 0.0);
        let within_100 = |depth: f32| depth <= 100.0;
        assert_eq!(
            resolve_placement(Some((near, [0, -1, 0])), cursor, picking_plane, within_100),
            PlacementTarget::OnSurface { point: near, face_normal: [0, -1, 0] }
        );
        assert_eq!(
            resolve_placement(Some((far, [0, -1, 0])), cursor, picking_plane, within_100),
            PlacementTarget::TooFar,
            "a hit past the limit is too far even though the anchor is not"
        );
    }

    /// **And the anchor plane is authorable exactly when its own depth is.** The empty-space
    /// path can no longer fly to a horizon, so the only way it reports too-far is the camera
    /// being zoomed out past the anchor itself — which is the "zoom in" message, unchanged in
    /// meaning and now reachable by one route instead of two.
    #[test]
    fn the_anchor_plane_is_too_far_exactly_when_the_anchor_is() {
        let cursor = ray([0.0, 0.0, 0.0], [0.1, 1.0, 0.0]);
        let within_100 = |depth: f32| depth <= 100.0;
        let near_anchor = plane([0.0, 1.0, 0.0], [0.0, 40.0, 0.0]);
        assert!(matches!(
            resolve_placement(None, cursor, near_anchor, within_100),
            PlacementTarget::OnAnchorPlane { .. }
        ));
        let far_anchor = plane([0.0, 1.0, 0.0], [0.0, 4000.0, 0.0]);
        assert_eq!(
            resolve_placement(None, cursor, far_anchor, within_100),
            PlacementTarget::TooFar
        );
    }

    /// **On-plane precision holds far past the authorable range.** This is the property CBMC
    /// cannot discharge in bounded time (proving a tight f32 bound bit-blasts the division), so it
    /// is measured instead: a deterministic sweep out to ±1e3 — ten times the authorable range —
    /// of the worst-case depth residue, `(point − anchor)·view`. It stays under `1e-3`, and the
    /// adversarial f32 worst case found offline was ~1.8e-4. Well within it. The Kani harness
    /// below proves the *finiteness* half over the same domain; this covers the *depth* half.
    #[test]
    fn the_anchor_plane_hit_stays_on_the_plane_at_large_magnitude() {
        let view = Vec3::new(0.0, 1.0, 0.0);
        let mut worst = 0.0_f32;
        // Non-round steps so the samples do not all land on tidy f32 values that cancel exactly.
        let mut origin_y = -1000.0_f32;
        while origin_y <= 1000.0 {
            let mut anchor_y = -1000.0_f32;
            while anchor_y <= 1000.0 {
                // direction.y across the guaranteed denominator range [0.5, 1.0]; x/z off-axis.
                for direction_y_step in 0..=10 {
                    let direction_y = 0.5 + direction_y_step as f32 * 0.05;
                    let direction = Vec3::new(0.37, direction_y, -0.21);
                    let picking_plane = AnchorPlane {
                        view_direction: view,
                        anchor_point: Vec3::new(11.0, anchor_y, -7.0),
                    };
                    let origin = Vec3::new(3.0, origin_y, 5.0);
                    let point = anchor_plane_hit(Ray::new(origin, direction), picking_plane);
                    let residue = (point - picking_plane.anchor_point).dot(view).abs();
                    worst = worst.max(residue);
                }
                anchor_y += 13.3;
            }
            origin_y += 13.3;
        }
        assert!(worst < 1e-3, "depth residue grew to {worst} at ±1e3");
    }

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
        assert_eq!(select_world_plane(direction, 0.342), WorldPlane::Ground);
        // Now graze the ground (nearly horizontal): a vertical must take over.
        let grazing = Vec3::new(0.98, 0.1, -0.05).normalize();
        assert_eq!(select_world_plane(grazing, 0.342), WorldPlane::VerticalFacingX);
    }

    /// **The distinction the viewport hangs on.** Too-far draws no preview and a placement
    /// does; they must be different values, or the viewport cannot say "zoom in" instead of
    /// silently doing nothing. The pair that used to be tested here included "nothing to hit",
    /// which the view-aligned plane made unreachable.
    #[test]
    fn too_far_is_never_confused_with_a_placement() {
        let cursor = ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let picking_plane = plane([0.0, 1.0, 0.0], [0.0, 40.0, 0.0]);
        let placed = resolve_placement(None, cursor, picking_plane, anywhere);
        let too_far = resolve_placement(None, cursor, picking_plane, |_| false);
        assert_eq!(too_far, PlacementTarget::TooFar);
        assert_ne!(placed, too_far, "the two must never collapse into one state");
    }
}

/// Kani bounded-model-checking proof that [`anchor_plane_hit`] is **total**: every camera ray
/// meets the view-aligned plane at a *finite* point, so the function never has a miss to report
/// and its `Vec3` return (no `Option`) is honest. This is exactly the claim the deleted
/// `NoSurface` state used to hedge against — if it failed, `NoSurface` would have to come back.
///
/// **Status: PROVEN.** `VERIFICATION:- SUCCESSFUL`, all 16 checks (including every glam
/// `mul.NaN` / division check CBMC auto-inserts) discharged in ~1.5 s solver time. Run under WSL:
/// `cargo kani -p raycast --harness anchor_plane_hit_is_total`.
///
/// **What this harness does *not* prove, and the record that has to be corrected.** An earlier
/// commit (`252c8e9`) reported that a combined harness "found a counterexample" and that totality
/// might be false. **That was a misread and it is retracted.** The line it read as a failure —
/// `SAT checker: instance is SATISFIABLE` — is CBMC's *reachability* solve reporting that the
/// assertions are reachable, which is expected and good; the proven harness above prints the same
/// line and still verifies. No run ever produced a `Status: FAILURE` verdict.
///
/// The combined harness *also* asserted the point lands exactly at the anchor's **depth**, a
/// floating-point equality-within-tolerance. That is a different kind of claim: proving a tight
/// f32 bound requires CBMC to bit-blast the division and multiplication, and it **does not
/// terminate** here (it hangs in the property solve even at a ±2 input bound — the expensive
/// *proving* direction, cf. "refuting is cheap, proving is not"). So the depth property is **not**
/// carried by Kani. It is covered instead by [`no_camera_ray_can_graze_the_anchor_plane`] above,
/// which sweeps a 121×121 ray grid and asserts `depth_of(point)` stays on the plane to `1e-2` —
/// and by a native f32 sweep (6 M samples) measuring the worst-case depth residue at ~1.8e-4 out
/// to ±1e3, well inside that. The two together are the on-plane evidence; the algebra in
/// [`anchor_plane_hit`]'s own docs is why it is exact in reals.
///
/// `#[cfg(kani)]` keeps it out of ordinary builds/tests.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A finite, magnitude-bounded symbolic `f32`.
    fn finite_f32(max_abs: f32) -> f32 {
        let value: f32 = kani::any();
        kani::assume(value.is_finite() && value.abs() <= max_abs);
        value
    }

    /// **Totality.** Every camera ray meets the plane at a finite point.
    ///
    /// The angular bound is stated on the **unnormalised** direction and that matters: with each
    /// component in `[-1, 1]`, `direction.dot(view) >= 0.5` on a vector whose magnitude reaches
    /// `√3` admits rays up to `acos(0.5/√3) ≈ 73°` off the view axis — *wider* than any real
    /// frustum, not the 60° a normalised `dot >= 0.5` would suggest. So the guaranteed denominator
    /// (`>= 0.5`) is exact and the field is conservatively wide. Held at the full `±1e3` anchor
    /// range because totality must hold everywhere, not just near the origin.
    #[kani::proof]
    fn anchor_plane_hit_is_total() {
        let view_direction = Vec3::new(0.0, 1.0, 0.0); // representative axis; the claim is rotation invariant
        let direction = Vec3::new(finite_f32(1.0), finite_f32(1.0), finite_f32(1.0));
        kani::assume(direction.dot(view_direction) >= 0.5);
        let origin = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let anchor_point = Vec3::new(finite_f32(1e3), finite_f32(1e3), finite_f32(1e3));
        let plane = AnchorPlane { view_direction, anchor_point };
        let point = anchor_plane_hit(Ray::new(origin, direction), plane);
        assert!(point.is_finite());
    }
}
