//! **Free Orbit** — the trackball half of the rig, and the seam between the two representations.
//!
//! [`OrbitCamera`] stores its orientation in the spherical chart (`orbit_theta` / `orbit_phi` /
//! `roll`). That chart is the **primary** representation and stays exact: a snap writes a literal
//! angle, Home restores one, and an idle frame changes nothing. The trackball is the
//! **secondary** representation — a quaternion in [`OrbitCamera::free_orientation`], authoritative
//! only while Free Orbit is the active type.
//!
//! The design and the reasons live in `docs/design/tool-modes-and-navigation.md` (camera
//! representation, owner-resolved 2026-07-27, reversing the quaternion-primary line). The short
//! version: quaternion-primary would force every `theta`/`phi` reader — `HomeView`, `SnapTween`,
//! the view-cube snap tables, `is_face_constrained`, persistence, `shot`'s CLI — to convert on day
//! one, and that migration is the expensive part of this work, not the integrator.
//!
//! **One authority at a time.** `free_orientation.is_some()` IS the authority bit; there is no
//! separate flag that could disagree with the data it describes. While it is `Some`, the chart is
//! stale *by definition* and no consumer may read it raw — [`OrbitCamera::direction`] and
//! [`OrbitCamera::up_vector`] dispatch, and everything else is built on those two. Only the
//! ORIENTATION forks: `target`, `orbit_distance`, `orbit_center` and `projection_mode` are shared
//! and mean the same thing under both types.
//!
//! **The seam is exact in both directions.** `theta`/`phi`/`roll` is a proper chart of SO(3), so
//! converting is not lossy. At a pole `theta` and `roll` are not individually determined — only
//! their combination is — which is gauge freedom, not information loss: every consistent choice
//! reproduces a bit-identical view. [`OrbitCamera::ensure_constrained`] resolves it by taking the
//! `theta` nearest the one already stored and letting `roll` absorb the remainder, the same rule
//! [`nearest_equivalent_theta`] applies to snaps. Gimbal lock never enters, because lock is a
//! failure of continuity *along a trajectory* and a type switch is a single point — which is also
//! why the type may not change mid-gesture.

use glam::{Quat, Vec3};

use crate::orbit::{OrbitCamera, OrbitType};
use crate::tween::nearest_equivalent_theta;

/// Radians of rotation per pixel of drag. The same gain the chart drag uses (`* 0.01`), so the
/// two types feel identical for a small drag away from the poles — pinned by
/// `a_small_free_drag_matches_the_constrained_drag`.
///
/// Unlike the chart drag this needs **no damping**: the trackball integrates in the tangent
/// space, where screen-drag → rotation is uniform by construction. The `sin(phi)` damping and its
/// floor exist only because the chart's map degenerates at the poles.
const RADIANS_PER_PIXEL: f32 = 0.01;

/// Below this the projection of world-up onto the view plane is too short to normalise: the
/// camera is looking straight down (or up) the vertical axis.
const POLE_PROJECTION_EPSILON: f32 = 1e-6;

impl OrbitCamera {
    /// Turn the camera by a screen-space drag, as `orbit_type` — the ONE door both types go
    /// through, so a caller cannot pick a type and reach the wrong integrator.
    ///
    /// Converts the representation first if the active type changed. Both `ensure_` calls are
    /// idempotent, so the common case (same type as last time) is a branch and nothing else.
    pub fn orbit_by_drag_as(&mut self, orbit_type: OrbitType, delta_x: f32, delta_y: f32) {
        match orbit_type {
            OrbitType::Constrained => {
                self.ensure_constrained();
                self.orbit_by_drag(delta_x, delta_y);
            }
            OrbitType::Free => {
                self.ensure_free();
                self.orbit_free_by_drag(delta_x, delta_y);
            }
        }
    }

    /// Turn the model about a screen-space drag with **no constraint**: the drag rotates the
    /// camera about its own right/up axes and roll accumulates freely.
    ///
    /// The rotation is composed in the camera's LOCAL frame (post-multiplied), which is what
    /// makes it a trackball: the axes are the ones on screen right now, so the same drag always
    /// produces the same on-screen motion regardless of where the camera has got to. No pole, no
    /// damping, no clamp — the degeneracies being avoided are the chart's, not the rotation's.
    ///
    /// Signs match [`OrbitCamera::orbit_by_drag`]: dragging right swings the camera the same way
    /// round the model, dragging down raises it. That equivalence is a pinned test, not a
    /// coincidence — it is the property that makes switching types feel continuous.
    ///
    /// The turn composes the two screen axes as an ORDERED pair, which is exact only in the
    /// small-delta limit. The deltas are per-`CursorMoved` cursor diffs (tens of pixels →
    /// milliradian commutator error), and a finite trackball drag has no path-independent
    /// answer anyway, so the ordered form is fine. If input handling ever coalesces deltas
    /// per FRAME, switch to one `Quat::from_axis_angle` about `(-delta_y, -delta_x, 0)` —
    /// at a large single delta the two forms visibly diverge.
    pub fn orbit_free_by_drag(&mut self, delta_x: f32, delta_y: f32) {
        self.ensure_free();
        let Some(orientation) = self.free_orientation else {
            return;
        };
        let turn = Quat::from_rotation_y(-delta_x * RADIANS_PER_PIXEL)
            * Quat::from_rotation_x(-delta_y * RADIANS_PER_PIXEL);
        self.free_orientation = Some((orientation * turn).normalize());
    }

    /// Make the trackball authoritative, seeding it from the chart if it is not already.
    ///
    /// Idempotent: calling it while already free is a no-op, which is what lets the shell treat
    /// its type variable as *policy* and this `Option` as *mechanism* without the two ever having
    /// to be proven in sync — a mismatch converges on the next gesture instead of corrupting.
    ///
    /// Returns whether it converted.
    pub fn ensure_free(&mut self) -> bool {
        if self.free_orientation.is_some() {
            return false;
        }
        // `view_basis` is already representation-generic and its columns are exactly
        // (right, screen_up, back) — the same frame a quaternion means by (X, Y, Z).
        self.free_orientation = Some(Quat::from_mat3(&self.view_basis()).normalize());
        true
    }

    /// Make the chart authoritative, decomposing the trackball into it if it is not already.
    ///
    /// `theta` and `phi` come from the view direction; `roll` takes the residual twist, so the
    /// view is reproduced EXACTLY rather than approximately — nothing is discarded here. The
    /// caller decides what to do about a non-zero `roll` afterwards: the owner ruling is that
    /// switching type re-levels it, ANIMATED (an eased [`SnapTween`], the same machinery Home
    /// already uses to re-upright), because a hard cut reads as a glitch. This function does not
    /// animate — `crates/camera` has no frame clock — it just reports that a conversion happened
    /// so the shell can start the tween.
    ///
    /// Idempotent, like its opposite. Returns whether it converted.
    ///
    /// [`SnapTween`]: crate::tween::SnapTween
    pub fn ensure_constrained(&mut self) -> bool {
        let Some(orientation) = self.free_orientation.take() else {
            return false;
        };
        let direction = (orientation * Vec3::Z).normalize();
        let screen_up = orientation * Vec3::Y;

        self.orbit_phi = direction.z.clamp(-1.0, 1.0).acos();
        // The gauge choice at (and near) a pole: of all the equivalent `theta` values, take the
        // one nearest where the chart already was, so a conversion never whips the azimuth
        // through most of a turn to describe the same view.
        let raw_theta = direction.y.atan2(direction.x);
        self.orbit_theta = nearest_equivalent_theta(self.orbit_theta, raw_theta);

        // `roll` is the signed twist from the chart's roll-free up to the actual one, measured
        // about the axis the camera looks along — the exact inverse of what `up_vector` applies,
        // so writing it here reproduces the view rather than approximating it. Measured against
        // `up_vector_base` and NOT `roll_free_up_for`: at a pole the base up is a function of the
        // theta just written, which is the vector `up_vector` will roll, while `roll_free_up_for`
        // answers a different question there (see its docs) and still carries the stale roll.
        self.roll = signed_twist_about(-direction, self.up_vector_base(), screen_up);
        true
    }

    /// A copy of this camera with the seam already closed: the chart authoritative, the
    /// trackball gone, the same view.
    ///
    /// For the read-only consumers that speak in ANGLES and cannot dispatch — the persisted
    /// config, the F9 repro dump, `shot`'s `--from-config`. Persisting the live camera while Free
    /// Orbit is active would write whatever `theta`/`phi` were last true, which is a view the user
    /// left some time ago; a repro that opens on a different pose than the one the bug was seen
    /// at is worse than no repro. Non-mutating on purpose, so a capture stays a capture.
    pub fn as_chart(&self) -> Self {
        let mut settled = *self;
        settled.ensure_constrained();
        settled
    }

    /// The canonical **roll-free** up for a view direction: world-up projected onto the view
    /// plane. Representation-generic — it takes the direction rather than reading the chart — so
    /// the per-frame consumers that ask "is this view upright" work under both types.
    ///
    /// At a pole the projection vanishes, and there the question has no answer to give: `theta`
    /// and `roll` are interchangeable there, so *every* up is a canonical up. Returning the
    /// actual up says exactly that, and makes "upright" true at the top and bottom face views —
    /// which is right, because those are face views like any other.
    pub(crate) fn roll_free_up_for(&self, direction: Vec3) -> Vec3 {
        let vertical = Vec3::Z - direction * Vec3::Z.dot(direction);
        if vertical.length_squared() > POLE_PROJECTION_EPSILON {
            vertical.normalize()
        } else {
            self.up_vector()
        }
    }
}

/// The signed angle from `from` to `to` about `axis`, both measured in the plane perpendicular to
/// it. Matches the sense `up_vector` rolls in (`Quat::from_axis_angle(forward, roll) * base`), so
/// feeding the result back through it reproduces `to`.
fn signed_twist_about(axis: Vec3, from: Vec3, to: Vec3) -> f32 {
    let flatten = |vector: Vec3| (vector - axis * vector.dot(axis)).normalize_or_zero();
    let from = flatten(from);
    let to = flatten(to);
    if from == Vec3::ZERO || to == Vec3::ZERO {
        return 0.0;
    }
    axis.dot(from.cross(to)).atan2(from.dot(to))
}

/// The quaternion for a view basis, for tests that need to build one by hand.
#[cfg(test)]
fn orientation_of(basis: glam::Mat3) -> Quat {
    Quat::from_mat3(&basis).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit::ProjectionMode;
    use glam::Mat3;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn upright_equator_camera() -> OrbitCamera {
        OrbitCamera {
            orbit_theta: 0.7,
            orbit_phi: FRAC_PI_2,
            ..OrbitCamera::default()
        }
    }

    /// How far apart two view bases are, as the largest per-axis angle. Bases, never angles:
    /// `theta` and `roll` have gauge freedom at the poles, so comparing them is wrong by
    /// construction there, while the basis is what the user actually sees.
    fn basis_distance(left: Mat3, right: Mat3) -> f32 {
        (0..3)
            .map(|column| left.col(column).angle_between(right.col(column)))
            .fold(0.0_f32, f32::max)
    }

    /// **The seam, chart → trackball → chart.** The round trip must reproduce the VIEW, at every
    /// orientation including the poles. This is the property that makes two representations
    /// honest rather than two truths.
    #[test]
    fn the_seam_round_trips_every_orientation() {
        let orientations = [
            (0.0, FRAC_PI_2, 0.0),
            (0.7, 1.1, 0.0),
            (-2.4, 0.3, 0.9),
            (1.2, PI - 0.02, -1.4),
            // The poles themselves, where theta and roll stop being individually determined.
            (0.6, 0.0, 0.0),
            (0.6, 0.0, 1.0),
            (-1.8, PI, 0.4),
        ];
        for (theta, phi, roll) in orientations {
            let mut camera = OrbitCamera {
                orbit_theta: theta,
                orbit_phi: phi,
                roll,
                ..OrbitCamera::default()
            };
            let before = camera.view_basis();

            assert!(camera.ensure_free(), "the chart camera was not yet free");
            assert!(
                basis_distance(camera.view_basis(), before) < 1e-3,
                "going free changed the view at ({theta}, {phi}, {roll})"
            );

            assert!(camera.ensure_constrained(), "it was free, so it converts");
            assert!(
                basis_distance(camera.view_basis(), before) < 1e-3,
                "coming back changed the view at ({theta}, {phi}, {roll})"
            );
        }
    }

    /// **The seam, trackball → chart → trackball.** The other direction, seeded from a
    /// quaternion that no chart produced — including one with a rolled horizon, which is the
    /// state Free Orbit actually leaves behind.
    #[test]
    fn the_seam_round_trips_an_arbitrary_orientation() {
        let mut camera = upright_equator_camera();
        camera.ensure_free();
        // Drag all over the place, well past where any chart would have gone.
        for (delta_x, delta_y) in [(140.0, -60.0), (-35.0, 210.0), (90.0, 90.0)] {
            camera.orbit_free_by_drag(delta_x, delta_y);
        }
        let before = camera.view_basis();

        assert!(camera.ensure_constrained(), "it was free");
        assert!(
            basis_distance(camera.view_basis(), before) < 1e-3,
            "the decomposition lost the view"
        );
        assert!(
            camera.roll.abs() > 1e-3,
            "a free drag rolls the horizon; roll must carry the residual, got {}",
            camera.roll
        );

        assert!(camera.ensure_free(), "it was constrained");
        assert!(
            basis_distance(camera.view_basis(), before) < 1e-3,
            "re-seeding the trackball lost the view"
        );
    }

    /// **The gauge rule.** Converting near a pole must take the `theta` nearest the one already
    /// stored, not the raw `atan2` branch — otherwise the azimuth whips most of a turn to
    /// describe a view that did not move.
    #[test]
    fn the_seam_takes_the_nearest_equivalent_theta() {
        // A theta well outside `atan2`'s range, so the naive answer would be ~2π away.
        let mut camera = OrbitCamera {
            orbit_theta: 6.5,
            orbit_phi: 0.4,
            ..OrbitCamera::default()
        };
        camera.ensure_free();
        camera.ensure_constrained();
        assert!(
            (camera.orbit_theta - 6.5).abs() < 0.2,
            "theta whipped from 6.5 to {}",
            camera.orbit_theta
        );
    }

    /// **Feel continuity — the test that catches a sign flip.** Away from the poles a SMALL free
    /// drag must produce the same view as the same small constrained drag. An inverted `dy`, a
    /// swapped axis or a handedness error all compile clean and all fail here.
    #[test]
    fn a_small_free_drag_matches_the_constrained_drag() {
        // winit's y grows DOWNWARD, so a positive `delta_y` is the cursor moving down the screen.
        for (delta_x, delta_y) in [(4.0, 0.0), (0.0, 4.0), (-3.0, 2.0), (2.0, -3.0)] {
            let mut constrained = upright_equator_camera();
            constrained.orbit_by_drag(delta_x, delta_y);

            let mut free = upright_equator_camera();
            free.orbit_free_by_drag(delta_x, delta_y);

            assert!(
                basis_distance(free.view_basis(), constrained.view_basis()) < 2e-3,
                "a ({delta_x}, {delta_y}) drag diverges between the two types"
            );
        }
    }

    /// The trackball keeps turning where the chart clamps. A vertical drag big enough to hit the
    /// pole stops the chart dead; Free carries straight over the top and comes down the far side,
    /// which is the whole point of having it.
    ///
    /// "Past the pole" is read off the AZIMUTH, not the height: 200px is 2 radians, which crosses
    /// the pole at 90° and ends 25° down the other side — still above the horizon, but facing the
    /// opposite way. A camera that stopped at the pole would keep its original azimuth.
    #[test]
    fn free_orbit_passes_over_the_pole() {
        let horizontal_before = upright_equator_camera().direction().truncate();

        let mut camera = upright_equator_camera();
        camera.orbit_free_by_drag(0.0, 200.0);
        let direction = camera.direction();
        assert!(
            direction.z > 0.0 && direction.truncate().dot(horizontal_before) < 0.0,
            "a 200px drag must carry past the top pole and come down the far side, got \
             {direction:?}"
        );

        let mut chart = upright_equator_camera();
        chart.orbit_by_drag(0.0, 200.0);
        assert!(
            chart.orbit_phi.abs() < 1e-6,
            "the chart clamps AT the pole, got {}",
            chart.orbit_phi
        );
    }

    /// Both `ensure_` calls are no-ops when already in that representation, and neither moves the
    /// view. This is what lets the shell's type variable be policy and the `Option` be mechanism
    /// without the two ever being proven in sync.
    #[test]
    fn the_transitions_are_idempotent() {
        let mut camera = upright_equator_camera();
        assert!(!camera.ensure_constrained(), "already constrained");
        assert!(camera.ensure_free(), "converted");
        let basis = camera.view_basis();
        assert!(!camera.ensure_free(), "already free");
        assert!(
            basis_distance(camera.view_basis(), basis) < 1e-6,
            "a redundant ensure_free moved the view"
        );
        assert!(camera.ensure_constrained(), "converted back");
        assert!(!camera.ensure_constrained(), "already constrained");
    }

    /// The trackball is ORIENTATION only. Everything else the camera holds means the same thing
    /// under both types and must survive a switch untouched — a pivot that moved when the user
    /// changed how the camera turns would be the two-pivot confusion all over again.
    #[test]
    fn only_the_orientation_forks() {
        let mut camera = OrbitCamera {
            target: Vec3::new(3.0, -4.0, 5.0),
            orbit_distance: 42.0,
            orbit_center: Vec3::new(-7.0, 1.0, 2.0),
            projection_mode: ProjectionMode::Orthographic,
            ..upright_equator_camera()
        };
        camera.ensure_free();
        camera.orbit_free_by_drag(50.0, 30.0);
        camera.ensure_constrained();

        assert_eq!(camera.target, Vec3::new(3.0, -4.0, 5.0), "target moved");
        assert_eq!(camera.orbit_distance, 42.0, "distance moved");
        assert_eq!(
            camera.orbit_center,
            Vec3::new(-7.0, 1.0, 2.0),
            "the orbit center moved"
        );
        assert_eq!(
            camera.projection_mode,
            ProjectionMode::Orthographic,
            "the projection changed"
        );
    }

    /// `orbit_about_point` holds its pivot fixed on screen under BOTH types — the invariant that
    /// makes a placed orbit center mean anything.
    #[test]
    fn a_free_orbit_about_a_point_leaves_that_point_fixed() {
        for orbit_type in [OrbitType::Constrained, OrbitType::Free] {
            let mut camera = upright_equator_camera();
            let pivot = Vec3::new(2.0, -1.0, 3.0);
            let before = camera.eye() - pivot;
            camera.orbit_about_point_as(orbit_type, pivot, 37.0, -21.0);
            let after = camera.eye() - pivot;
            assert!(
                (before.length() - after.length()).abs() < 1e-2,
                "{orbit_type:?}: the eye's distance to the pivot changed, {} -> {}",
                before.length(),
                after.length()
            );
        }
    }

    /// A quaternion built straight from a basis reproduces that basis — the assumption
    /// `ensure_free` rests on.
    #[test]
    fn a_basis_survives_the_quaternion() {
        let camera = OrbitCamera {
            orbit_theta: -1.3,
            orbit_phi: 2.2,
            roll: 0.6,
            ..OrbitCamera::default()
        };
        let basis = camera.view_basis();
        let rebuilt = Mat3::from_quat(orientation_of(basis));
        assert!(
            basis_distance(basis, rebuilt) < 1e-4,
            "the basis did not survive"
        );
    }
}
