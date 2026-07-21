//! **Sliding a contact on a composed signed-distance surface** — the CPU counterpart of the
//! GPU placement ghost's sphere-trace (`crates/display/src/shaders/placement_ghost.wgsl`,
//! `fn trace`), and the continuous-placement solver ADR 0027 §5 calls for.
//!
//! Continuous placement (ADR 0027) needs three answers off a surface the caller composes at
//! runtime — the **contact point**, its **normal**, and where a wanted normal **slides the
//! contact to** — and it must get them off the *composed* field (a boolean / `Part` result),
//! not a single analytic primitive. So the whole module speaks one abstraction: a
//! caller-supplied **signed-distance closure** `field: impl Fn(Vec3) -> f32`, negative inside,
//! positive outside, zero on the surface, evaluated at a world point. This crate never
//! constructs that closure — it consumes it — which keeps `raycast` below the domain layer
//! (the graphics-crate boundary law, ADR 0015): no `document` / `evaluation` / `voxel_core`
//! field type crosses the edge, only a `Fn`.
//!
//! ## One mechanism: gradient + damped Newton
//!
//! ADR 0027 §5's ruling is that *one* predicate serves the hit, the normal, the contact
//! projection, the voxel-snap re-projection, and the angle-to-position slide — sliding on the
//! composed field via `p -= field(p) * gradient / |gradient|^2`. This module is that predicate,
//! spelled out:
//!
//! * [`raymarch`] — sphere-trace the closure from a ray to the first surface crossing.
//! * [`gradient_normal`] — the central-difference gradient, normalized: the surface normal.
//! * [`project_to_surface`] — damped, iteration-capped Newton onto `field == 0`.
//! * [`snap_slide_to_normal`] — slide the contact *along* the surface toward a wanted normal.
//! * [`snap_to_lattice_then_reproject`] — round to a lattice granule, then re-seat.
//!
//! ## Why the `/|gradient|^2` form and the damping
//!
//! The gradient of a **true** distance field is the exact unit normal, so on the primitives the
//! Newton step is a closed-form inversion — it lands on the surface in one step. A
//! **non-true-distance** field (an L-infinity box, a post-outset / emboss field) still gives a
//! correct normal *direction* but a non-unit gradient magnitude; dividing by `|gradient|^2`
//! absorbs that magnitude so the step size stays honest. The damping is scale-free: the Newton
//! step is never allowed to move further than the distance the field itself reports at the
//! point (its magnitude is clamped to `field(p).abs()`), so on a Lipschitz-1 field it cannot
//! overshoot the surface, and on a badly non-Lipschitz field the per-step move stays bounded
//! and the iteration cap ends it. No per-primitive analytic inversion + validity twin to drift
//! — Newton on the composed field physically cannot converge into a carved-away region, because
//! the field is positive there.
//!
//! ## Parity with the GPU sphere-trace
//!
//! [`raymarch`] mirrors `placement_ghost.wgsl`'s `fn trace`: the same 0.7 step relaxation, the
//! same 192-step ceiling, the same `distance < tolerance` hit test, and the same
//! central-difference normal (never `fwidth` — that pass runs in non-uniform control flow). It
//! deliberately does **not** carry the shader's AABB pre-clip or its shape-offset uniform: this
//! is the readable spec of the marching *loop*, given a bare ray and a closure, and the caller
//! supplies the surface tolerance rather than deriving it from a shape scale. See
//! [`MarchParams`] for the correspondence.
//!
//! ## The frame
//!
//! Every point and direction here is in the **one world frame the caller already works in** and
//! travels as a value in that frame (ADR 0008) — the closure is evaluated at world points, the
//! hit and slide come back in world points.

use glam::Vec3;

/// A surface crossing found by [`raymarch`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHit {
    /// The world point on the surface (`field(point) ~= 0`).
    pub point: Vec3,
    /// The ray parameter at the hit — the distance travelled from the ray origin along the
    /// (unit) direction. May be small but is always non-negative; the march starts at the
    /// origin and steps forward.
    pub distance_travelled: f32,
    /// The outward surface normal at [`SurfaceHit::point`], from [`gradient_normal`].
    pub normal: Vec3,
}

/// Marching parameters for [`raymarch`], with defaults matched to the GPU sphere-trace in
/// `placement_ghost.wgsl` (`fn trace`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarchParams {
    /// The hit tolerance: the march reports a hit when `field(p) < surface_epsilon`, and the
    /// step never shrinks below this so the loop cannot stall on a near-zero field. The GPU
    /// derives this from the shape scale (`max(scale * 1e-4, 1e-3)`); here the caller supplies
    /// it directly, defaulting to that floor.
    pub surface_epsilon: f32,
    /// The step is `max(field(p) * step_relaxation, surface_epsilon)`. Relaxation below 1.0
    /// guards against overshooting a thin feature of a non-exact distance field — the GPU uses
    /// `0.7`, the factor at which every ghost fixture traces cleanly.
    pub step_relaxation: f32,
    /// The march gives up after this many steps (the GPU's `MAX_TRACE_STEPS`).
    pub max_steps: usize,
    /// The march gives up once the ray parameter passes this (there is no AABB pre-clip on the
    /// CPU side, so a far-range ceiling stands in for the shader's bounding interval).
    pub max_distance: f32,
}

impl Default for MarchParams {
    fn default() -> Self {
        Self {
            surface_epsilon: 1.0e-3,
            step_relaxation: 0.7,
            max_steps: 192,
            max_distance: 1.0e4,
        }
    }
}

/// The epsilon of the central-difference gradient. Matches the GPU shading normal's fine probe
/// (`max(scale * 1e-3, 1e-3)`) at unit-ish scale; small enough to read curvature, large enough
/// to stay clear of closure noise.
const GRADIENT_EPSILON: f32 = 1.0e-3;

/// Newton stops once `field(p).abs()` falls below this.
const PROJECTION_SURFACE_TOLERANCE: f32 = 1.0e-4;

/// The Newton iteration ceiling — the always-on safety against a non-Lipschitz field.
const PROJECTION_MAX_ITERATIONS: usize = 64;

/// The slide's iteration ceiling.
const SLIDE_MAX_ITERATIONS: usize = 96;

/// The slide has converged (or stalled) once the tangential component of the normal error falls
/// below this.
const SLIDE_ALIGNMENT_TOLERANCE: f32 = 1.0e-4;

/// The slide's first tangential step length; backtracking halves it as it homes in.
const SLIDE_INITIAL_STEP: f32 = 1.0;

/// The slide gives up backtracking once the step shrinks below this.
const SLIDE_MIN_STEP: f32 = 1.0e-5;

/// The raw (**un-normalized**) central-difference gradient of the field at `point`.
///
/// This is `vec3(f(p+ex) - f(p-ex), f(p+ey) - f(p-ey), f(p+ez) - f(p-ez))` before scaling —
/// its direction is the outward normal and its magnitude is the field's local slope (unit for a
/// true distance field). [`gradient_normal`] normalizes it; the Newton step divides by its
/// squared length so the non-unit magnitude of a non-true-distance field cancels.
fn field_gradient(point: Vec3, field: &impl Fn(Vec3) -> f32) -> Vec3 {
    let dx = Vec3::new(GRADIENT_EPSILON, 0.0, 0.0);
    let dy = Vec3::new(0.0, GRADIENT_EPSILON, 0.0);
    let dz = Vec3::new(0.0, 0.0, GRADIENT_EPSILON);
    Vec3::new(
        field(point + dx) - field(point - dx),
        field(point + dy) - field(point - dy),
        field(point + dz) - field(point - dz),
    )
}

/// The unit outward surface normal at `point` — the normalized central-difference gradient of
/// the field.
///
/// For a **true** distance field the gradient is already the exact unit normal (a sphere's is
/// its radial direction). A degenerate gradient (a flat interior, a discontinuity where both
/// samples agree) falls back to world-up `+Z`, matching the GPU shader's guard.
pub fn gradient_normal(point: Vec3, field: impl Fn(Vec3) -> f32) -> Vec3 {
    let gradient = field_gradient(point, &field);
    let magnitude = gradient.length();
    if magnitude < 1.0e-12 {
        return Vec3::Z;
    }
    gradient / magnitude
}

/// A single damped Newton step toward `field == 0`, shared by the projection and the slide.
///
/// `p -= field(p) * gradient / |gradient|^2`, with the step magnitude clamped to
/// `field(p).abs()` — the scale-free damping (see the module docs). Returns the moved point, or
/// the input unchanged when the gradient has collapsed.
fn newton_step_to_surface(point: Vec3, value: f32, field: &impl Fn(Vec3) -> f32) -> Vec3 {
    let gradient = field_gradient(point, field);
    let squared_length = gradient.dot(gradient);
    if squared_length < 1.0e-24 {
        return point;
    }
    let mut step = value * gradient / squared_length;
    // Damping: never move further than the distance the field itself reports at this point. On
    // a Lipschitz-1 field this cannot overshoot the surface; on a worse field it stays bounded.
    let max_magnitude = value.abs();
    let magnitude = step.length();
    if magnitude > max_magnitude && magnitude > 1.0e-20 {
        step *= max_magnitude / magnitude;
    }
    point - step
}

/// Damped, iteration-capped Newton projection of `point` onto the surface `field == 0`.
///
/// Returns the converged point (`field(result).abs()` below tolerance, or the best point within
/// the iteration cap). For a true distance field this lands exactly in one step; for a
/// non-true-distance field the `/|gradient|^2` form and the step-magnitude clamp keep it stable.
/// Because it descends the composed field it cannot converge into a carved-away region — the
/// field is positive there.
pub fn project_to_surface(point: Vec3, field: impl Fn(Vec3) -> f32) -> Vec3 {
    let mut current = point;
    for _ in 0..PROJECTION_MAX_ITERATIONS {
        let value = field(current);
        if value.abs() < PROJECTION_SURFACE_TOLERANCE {
            break;
        }
        let moved = newton_step_to_surface(current, value, &field);
        if moved == current {
            break;
        }
        current = moved;
    }
    current
}

/// **The angle-to-position slide** (ADR 0027 §2, position-dominant precedence): given a `seat`
/// point on the surface and a wanted `target_normal` direction, slide the contact *along* the
/// surface to where the surface normal best matches `target_normal`, and return that seated
/// point.
///
/// At each step it reads the current normal, projects `target_normal` onto the local tangent
/// plane to get the direction that rotates the normal toward the target, moves the contact that
/// way, and re-[`project_to_surface`]s so the point stays seated. The tangential step
/// backtracks (halves) whenever a move would *not* improve the alignment, so it converges
/// without a scene-scale parameter and settles right at the matching contact.
///
/// If `target_normal` is unreachable on this surface — a flat face exposes only one normal, and
/// the exact antipode of the current normal has no tangential direction — the slide makes no
/// progress and returns the best-effort point (the nearest reachable contact / the seat), per
/// ADR 0027 §5. It works on any composed field (sphere, cylinder, their booleans) because it
/// only calls the closure.
pub fn snap_slide_to_normal(seat: Vec3, target_normal: Vec3, field: impl Fn(Vec3) -> f32) -> Vec3 {
    let target = {
        let length = target_normal.length();
        if length < 1.0e-12 {
            return seat;
        }
        target_normal / length
    };

    let mut current = project_to_surface(seat, &field);
    let mut step = SLIDE_INITIAL_STEP;

    for _ in 0..SLIDE_MAX_ITERATIONS {
        let normal = gradient_normal(current, &field);
        let alignment = normal.dot(target);
        // The part of the target that lies in the tangent plane — the direction along the
        // surface that most reduces the normal error.
        let tangential = target - normal * alignment;
        let tangential_length = tangential.length();
        if tangential_length < SLIDE_ALIGNMENT_TOLERANCE {
            break;
        }
        let direction = tangential / tangential_length;

        let candidate = project_to_surface(current + direction * step, &field);
        let candidate_alignment = gradient_normal(candidate, &field).dot(target);
        if candidate_alignment > alignment {
            current = candidate;
        } else {
            step *= 0.5;
            if step < SLIDE_MIN_STEP {
                break;
            }
        }
    }
    current
}

/// Round `hit_point` to the nearest multiple of `lattice_step` on every axis, then
/// [`project_to_surface`] so it stays seated — the position Voxel / Block snap of ADR 0027 §2.
///
/// `lattice_step` is a caller-supplied granule (one voxel or one block, in world units). A
/// non-positive step is a no-op guard: the raw point is projected and returned. Rounding pulls
/// the contact to the lattice; the re-projection pulls it back onto the surface, so the snapped
/// contact is both quantized and seated.
pub fn snap_to_lattice_then_reproject(
    hit_point: Vec3,
    lattice_step: f32,
    field: impl Fn(Vec3) -> f32,
) -> Vec3 {
    if lattice_step > 0.0 {
        let snapped = (hit_point / lattice_step).round() * lattice_step;
        project_to_surface(snapped, field)
    } else {
        // A non-positive (or NaN) step is a no-op guard: seat the raw point and return.
        project_to_surface(hit_point, field)
    }
}

/// Sphere-trace the signed-distance `field` from a ray, returning the first surface crossing or
/// `None` on a miss.
///
/// The march starts at `origin` and steps forward along the normalized `direction` by the field
/// value (relaxed by [`MarchParams::step_relaxation`]), reporting a hit the first time
/// `field(p) < surface_epsilon` — so a ray starting *inside* the surface hits immediately, the
/// same signed test the GPU shader uses. It misses after [`MarchParams::max_steps`] or once the
/// ray parameter passes [`MarchParams::max_distance`]. A zero-length direction is a guaranteed
/// miss.
pub fn raymarch(
    origin: Vec3,
    direction: Vec3,
    field: impl Fn(Vec3) -> f32,
    params: &MarchParams,
) -> Option<SurfaceHit> {
    let direction_length = direction.length();
    if direction_length < 1.0e-20 {
        return None;
    }
    let unit_direction = direction / direction_length;

    let mut travelled = 0.0_f32;
    for _ in 0..params.max_steps {
        if travelled > params.max_distance {
            return None;
        }
        let point = origin + unit_direction * travelled;
        let distance = field(point);
        if distance < params.surface_epsilon {
            return Some(SurfaceHit {
                point,
                distance_travelled: travelled,
                normal: gradient_normal(point, &field),
            });
        }
        travelled += (distance * params.step_relaxation).max(params.surface_epsilon);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A true distance field for a sphere: `|p - center| - radius`. Its gradient is the exact
    /// unit outward normal everywhere off the center.
    fn sphere_field(center: Vec3, radius: f32) -> impl Fn(Vec3) -> f32 {
        move |point: Vec3| (point - center).length() - radius
    }

    /// A true distance field for an infinite cylinder about the world Z axis: the 2D distance
    /// in the XY plane, `radius` from the axis. Gradient is the radial XY unit normal.
    fn z_axis_cylinder_field(radius: f32) -> impl Fn(Vec3) -> f32 {
        move |point: Vec3| (point.x * point.x + point.y * point.y).sqrt() - radius
    }

    #[test]
    fn gradient_normal_on_sphere_is_radial() {
        let center = Vec3::new(1.0, -2.0, 0.5);
        let radius = 3.0;
        let field = sphere_field(center, radius);
        // A point on the surface in a known direction.
        let direction = Vec3::new(0.3, 0.4, 0.866_025_4).normalize();
        let surface_point = center + direction * radius;
        let normal = gradient_normal(surface_point, &field);
        assert!(
            (normal - direction).length() < 1.0e-3,
            "expected radial normal {direction:?}, got {normal:?}"
        );
    }

    #[test]
    fn gradient_normal_on_cylinder_is_radial_in_plane() {
        let radius = 2.5;
        let field = z_axis_cylinder_field(radius);
        // On the surface at an angle, at some height (height is irrelevant to the normal).
        let angle = 0.9_f32;
        let surface_point = Vec3::new(radius * angle.cos(), radius * angle.sin(), 4.0);
        let normal = gradient_normal(surface_point, &field);
        let expected = Vec3::new(angle.cos(), angle.sin(), 0.0);
        assert!(
            (normal - expected).length() < 1.0e-3,
            "expected radial-in-plane normal {expected:?}, got {normal:?}"
        );
    }

    #[test]
    fn project_to_surface_lands_on_sphere() {
        let center = Vec3::new(-1.0, 0.5, 2.0);
        let radius = 4.0;
        let field = sphere_field(center, radius);
        // Start well off the surface, both outside and inside.
        for start in [
            center + Vec3::new(1.0, 0.0, 0.0) * (radius + 2.7),
            center + Vec3::new(0.2, -0.5, 0.3).normalize() * (radius - 1.5),
            center + Vec3::new(-3.0, 2.0, -1.0),
        ] {
            let landed = project_to_surface(start, &field);
            let distance_to_center = (landed - center).length();
            assert!(
                (distance_to_center - radius).abs() < 1.0e-3,
                "projected point {landed:?} is {distance_to_center} from center, want {radius}"
            );
            assert!(field(landed).abs() < 1.0e-3, "field at landed = {}", field(landed));
        }
    }

    #[test]
    fn snap_slide_to_normal_reaches_closed_form_on_sphere() {
        let center = Vec3::new(2.0, 2.0, -1.0);
        let radius = 5.0;
        let field = sphere_field(center, radius);
        // Seat at the north pole; ask for several target normals across the sphere.
        let seat = center + Vec3::Z * radius;
        for target in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.3, 0.4, 0.866_025_4).normalize(),
            Vec3::new(-0.5, -0.5, std::f32::consts::FRAC_1_SQRT_2).normalize(),
            Vec3::new(0.0, 1.0, 0.2).normalize(),
        ] {
            let slid = snap_slide_to_normal(seat, target, &field);
            // The closed-form answer: the contact where the normal equals the target is
            // c + r * target.
            let expected_point = center + target * radius;
            assert!(
                (slid - expected_point).length() < 1.0e-2,
                "target {target:?}: slid to {slid:?}, want {expected_point:?}"
            );
            // Its actual normal matches the target, and it is on the sphere.
            let actual_normal = gradient_normal(slid, &field);
            assert!(
                (actual_normal - target).length() < 5.0e-3,
                "target {target:?}: actual normal {actual_normal:?}"
            );
            assert!(
                ((slid - center).length() - radius).abs() < 1.0e-2,
                "target {target:?}: slid point is off the sphere"
            );
        }
    }

    #[test]
    fn snap_slide_on_flat_face_is_best_effort() {
        // A flat field: the surface z == 0, normal +Z everywhere. No tangential move rotates
        // the normal, so asking for +X must not blow up — it returns a seated point.
        let field = |point: Vec3| point.z;
        let seat = Vec3::new(3.0, -2.0, 0.0);
        let slid = snap_slide_to_normal(seat, Vec3::X, field);
        assert!(field(slid).abs() < 1.0e-3, "best-effort point must stay seated on z==0");
    }

    #[test]
    fn snap_to_lattice_reprojects_onto_sphere() {
        let center = Vec3::ZERO;
        let radius = 4.0;
        let field = sphere_field(center, radius);
        let hit = Vec3::new(0.1, 0.2, radius - 0.05); // near the north pole, off-lattice
        let lattice_step = 0.5;
        let snapped = snap_to_lattice_then_reproject(hit, lattice_step, &field);
        // Still seated on the sphere after the lattice round.
        assert!(
            ((snapped - center).length() - radius).abs() < 1.0e-3,
            "lattice-snapped point {snapped:?} is off the sphere"
        );
    }

    #[test]
    fn snap_to_lattice_nonpositive_step_is_a_projection() {
        let field = sphere_field(Vec3::ZERO, 2.0);
        let hit = Vec3::new(3.0, 0.0, 0.0);
        let result = snap_to_lattice_then_reproject(hit, 0.0, &field);
        assert!((result.length() - 2.0).abs() < 1.0e-3, "zero step should still project");
    }

    #[test]
    fn raymarch_hits_sphere_at_analytic_entry() {
        let center = Vec3::new(0.0, 0.0, 10.0);
        let radius = 2.0;
        let field = sphere_field(center, radius);
        // Ray down +Z from the origin: analytic entry at z = 10 - 2 = 8, distance 8.
        let hit = raymarch(Vec3::ZERO, Vec3::Z, &field, &MarchParams::default())
            .expect("ray should hit the sphere");
        assert!(
            (hit.distance_travelled - 8.0).abs() < 2.0e-3,
            "entry distance {} want ~8.0",
            hit.distance_travelled
        );
        assert!((hit.point.z - 8.0).abs() < 2.0e-3, "entry z {} want ~8.0", hit.point.z);
        // The entry normal faces back toward the ray origin: -Z.
        assert!(
            (hit.normal - Vec3::new(0.0, 0.0, -1.0)).length() < 5.0e-3,
            "entry normal {:?} want -Z",
            hit.normal
        );
    }

    #[test]
    fn raymarch_misses_when_pointed_away() {
        let field = sphere_field(Vec3::new(0.0, 0.0, 10.0), 2.0);
        // Point away from the sphere.
        assert!(raymarch(Vec3::ZERO, -Vec3::Z, &field, &MarchParams::default()).is_none());
    }

    #[test]
    fn raymarch_zero_direction_misses() {
        let field = sphere_field(Vec3::ZERO, 1.0);
        assert!(raymarch(Vec3::new(5.0, 0.0, 0.0), Vec3::ZERO, &field, &MarchParams::default())
            .is_none());
    }
}

/// A cheap Kani invariant (ADR 0027 says one is welcome, not required): [`project_to_surface`]'s
/// output seats onto a sphere — its field magnitude is within tolerance. Runs only under
/// `cargo kani`, never in `cargo test`.
#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    fn project_to_surface_seats_on_sphere() {
        let radius = 3.0_f32;
        // A bounded start point off the surface.
        let x: f32 = kani::any();
        let y: f32 = kani::any();
        let z: f32 = kani::any();
        kani::assume(x.is_finite() && x.abs() < 8.0);
        kani::assume(y.is_finite() && y.abs() < 8.0);
        kani::assume(z.is_finite() && z.abs() < 8.0);
        let start = Vec3::new(x, y, z);
        // Stay clear of the center where the gradient is undefined.
        kani::assume(start.length() > 0.5);
        let field = move |point: Vec3| point.length() - radius;
        let landed = project_to_surface(start, &field);
        assert!(field(landed).abs() < 1.0e-2);
    }
}
