//! Eased camera snaps and angle normalisation.
//!
//! A [`SnapTween`] carries the camera's orbit `(theta, phi)` and `roll` from their
//! current values to a target over a fixed duration, easing the interpolation with
//! [`ease_in_out_quad`] so a ViewCube click or a Home snap accelerates out and
//! decelerates in rather than starting and stopping abruptly. Two small pieces of
//! angle hygiene travel with it: [`nearest_equivalent_theta`] picks the target
//! azimuth's `mod 2π` representative nearest the current one so a snap never spins
//! the long way round, and [`normalize_roll`] keeps accumulated roll bounded.
//!
//! Cite: `easeInOutQuad` is the quadratic ease from the standard easing catalogue
//! (Penner, *Robert Penner's Programming Macromedia Flash MX*, 2002); the
//! nearest-representative choice is the shortest-arc rule for interpolating on the
//! circle (Akenine-Möller, Haines & Hoffman, *Real-Time Rendering*).

use crate::orbit::OrbitCamera;
use crate::view_cube::{CubeFace, RollDir, ViewCubeElement};

/// Pick the equivalent of `target_theta` (mod 2π) nearest to `current_theta`, so a
/// snap never spins the long way round ("add/sub 2π before tweening").
pub fn nearest_equivalent_theta(current_theta: f32, target_theta: f32) -> f32 {
    use std::f32::consts::PI;
    let mut chosen = target_theta;
    while chosen - current_theta > PI {
        chosen -= 2.0 * PI;
    }
    while chosen - current_theta < -PI {
        chosen += 2.0 * PI;
    }
    chosen
}

/// easeInOutQuad over `t` in `[0, 1]`.
pub fn ease_in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

/// An in-progress eased camera snap from `(theta, phi)` toward a target view.
///
/// The windowed app advances it each frame; a headless path skips the tween and
/// applies the final angles directly. `theta_to` is already the nearest-equivalent
/// target (no long spins).
#[derive(Debug, Clone, Copy)]
pub struct SnapTween {
    pub theta_from: f32,
    pub phi_from: f32,
    pub theta_to: f32,
    pub phi_to: f32,
    /// Roll at the tween start (radians).
    pub roll_from: f32,
    /// Roll target (radians). Face/edge/corner/Home snaps re-upright (target 0);
    /// a roll arrow tweens it by ∓π/2 off `roll_from`.
    pub roll_to: f32,
    /// Seconds elapsed since the tween started.
    pub elapsed_seconds: f32,
    /// Total duration in seconds (~0.38 s).
    pub duration_seconds: f32,
}

impl SnapTween {
    /// Tween duration in seconds (~380 ms).
    pub const DEFAULT_DURATION_SECONDS: f32 = 0.38;

    /// Begin a snap to a view-cube element (face, edge or corner). `theta_to` is
    /// resolved to the nearest equivalent so the camera takes the short way.
    pub fn to_element(camera: &OrbitCamera, element: ViewCubeElement) -> Self {
        let (target_theta, target_phi) = element.snap_angles();
        Self {
            theta_from: camera.orbit_theta,
            phi_from: camera.orbit_phi,
            theta_to: nearest_equivalent_theta(camera.orbit_theta, target_theta),
            phi_to: target_phi,
            // A face/edge/corner snap re-uprights: tween roll back to 0.
            roll_from: camera.roll,
            roll_to: 0.0,
            elapsed_seconds: 0.0,
            duration_seconds: Self::DEFAULT_DURATION_SECONDS,
        }
    }

    /// Begin a snap from the camera's current angles to a face. `theta_to` is
    /// resolved to the nearest equivalent so the camera takes the short way.
    pub fn to_face(camera: &OrbitCamera, face: CubeFace) -> Self {
        let (target_theta, target_phi) = face.snap_angles();
        Self {
            theta_from: camera.orbit_theta,
            phi_from: camera.orbit_phi,
            theta_to: nearest_equivalent_theta(camera.orbit_theta, target_theta),
            phi_to: target_phi,
            // A face snap re-uprights: tween roll back to 0.
            roll_from: camera.roll,
            roll_to: 0.0,
            elapsed_seconds: 0.0,
            duration_seconds: Self::DEFAULT_DURATION_SECONDS,
        }
    }

    /// Begin a **roll** tween: twist the view by ∓π/2 about the view axis without
    /// moving the orbit angles. `Cw` rolls the view clockwise on screen, `Ccw`
    /// counter-clockwise. The orbit angles are held (`*_to == *_from`) so only
    /// `roll` animates. The target is kept CONTINUOUS (it accumulates off the live
    /// roll) so repeated arrow presses tween smoothly; a later face/Home snap
    /// re-uprights to 0.
    pub fn roll(camera: &OrbitCamera, direction: RollDir) -> Self {
        use std::f32::consts::FRAC_PI_2;
        // Screen convention: a Cw roll arrow twists the view clockwise, which is a
        // NEGATIVE rotation of the up vector about the forward (look) axis in a
        // right-handed frame; Ccw is positive.
        let delta = match direction {
            RollDir::Cw => -FRAC_PI_2,
            RollDir::Ccw => FRAC_PI_2,
        };
        Self {
            theta_from: camera.orbit_theta,
            phi_from: camera.orbit_phi,
            theta_to: camera.orbit_theta,
            phi_to: camera.orbit_phi,
            roll_from: camera.roll,
            roll_to: camera.roll + delta,
            elapsed_seconds: 0.0,
            duration_seconds: Self::DEFAULT_DURATION_SECONDS,
        }
    }

    /// Advance by `delta_seconds` and write the eased angles into `camera`.
    /// Returns `true` once the tween has finished (so the caller can drop it).
    pub fn advance(&mut self, camera: &mut OrbitCamera, delta_seconds: f32) -> bool {
        self.elapsed_seconds += delta_seconds;
        let progress = (self.elapsed_seconds / self.duration_seconds).clamp(0.0, 1.0);
        let eased = ease_in_out_quad(progress);
        camera.orbit_theta = self.theta_from + (self.theta_to - self.theta_from) * eased;
        camera.orbit_phi = self.phi_from + (self.phi_to - self.phi_from) * eased;
        camera.roll = self.roll_from + (self.roll_to - self.roll_from) * eased;
        let finished = progress >= 1.0;
        if finished {
            // Normalise the settled roll to (−π, π] so repeated arrow presses never
            // let it grow unbounded (the up vector is 2π-periodic anyway). Only at
            // rest, so the in-flight interpolation stays continuous (no mid-tween jump).
            camera.roll = normalize_roll(camera.roll);
        }
        finished
    }
}

/// Normalise a roll angle to the half-open interval `(−π, π]`. Keeps accumulated
/// roll bounded after repeated arrow presses without affecting the rendered
/// orientation (the up vector is 2π-periodic in roll).
pub fn normalize_roll(roll: f32) -> f32 {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    // Map into [0, 2π) robustly (no float drift from repeated subtraction), then
    // shift the upper half into the negative side so the result lands in (−π, π].
    let wrapped = roll.rem_euclid(two_pi);
    if wrapped > PI {
        wrapped - two_pi
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn nearest_theta_avoids_long_spin() {
        // Current near 3.0 rad, target 0 → should pick -2π (i.e. ≈ -0… within π).
        let chosen = nearest_equivalent_theta(3.0, 0.0);
        assert!((chosen - 3.0).abs() <= PI + 1e-4);
        // Target chosen must be congruent to 0 mod 2π.
        let remainder = chosen.rem_euclid(2.0 * PI);
        assert!(approx(remainder, 0.0) || approx(remainder, 2.0 * PI));
    }

    #[test]
    fn nearest_theta_picks_plus_two_pi_when_closer() {
        // current -3.0, target +π/2: +π/2 is 4.57 away; +π/2 - 2π = -4.71 (1.71 away).
        let chosen = nearest_equivalent_theta(-3.0, FRAC_PI_2);
        assert!((chosen - (-3.0)).abs() <= PI + 1e-4);
    }

    #[test]
    fn ease_in_out_quad_endpoints_and_midpoint() {
        assert!(approx(ease_in_out_quad(0.0), 0.0));
        assert!(approx(ease_in_out_quad(1.0), 1.0));
        assert!(approx(ease_in_out_quad(0.5), 0.5));
    }

    #[test]
    fn tween_to_element_lands_on_edge_target() {
        let mut camera = OrbitCamera::default();
        let element = ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top);
        let mut tween = SnapTween::to_element(&camera, element);
        let (target_theta, target_phi) = (tween.theta_to, tween.phi_to);
        assert!(tween.advance(&mut camera, 1.0));
        assert!(approx(camera.orbit_theta, target_theta));
        assert!(approx(camera.orbit_phi, target_phi));
    }

    #[test]
    fn tween_lands_exactly_on_target() {
        let mut camera = OrbitCamera::default();
        let mut tween = SnapTween::to_face(&camera, CubeFace::Front);
        let target_theta = tween.theta_to;
        let target_phi = tween.phi_to;
        // Step well past the duration.
        let finished = tween.advance(&mut camera, 1.0);
        assert!(finished);
        assert!(approx(camera.orbit_theta, target_theta));
        assert!(approx(camera.orbit_phi, target_phi));
    }

    /// A `RollArrow(Cw)` click tweens roll by −π/2; `Ccw` by +π/2. The orbit angles
    /// are held (theta/phi targets == sources).
    #[test]
    fn roll_arrow_tweens_roll_by_quarter_turn() {
        let camera = OrbitCamera::default();
        let cw = SnapTween::roll(&camera, RollDir::Cw);
        assert!(approx(cw.roll_to, -FRAC_PI_2), "Cw roll_to {}", cw.roll_to);
        assert!(approx(cw.theta_to, camera.orbit_theta), "Cw must hold theta");
        assert!(approx(cw.phi_to, camera.orbit_phi), "Cw must hold phi");
        let ccw = SnapTween::roll(&camera, RollDir::Ccw);
        assert!(approx(ccw.roll_to, FRAC_PI_2), "Ccw roll_to {}", ccw.roll_to);

        // Advancing a Cw roll tween lands roll exactly on −π/2.
        let mut camera = OrbitCamera::default();
        let mut tween = SnapTween::roll(&camera, RollDir::Cw);
        assert!(tween.advance(&mut camera, 1.0));
        assert!(approx(camera.roll, -FRAC_PI_2), "settled roll {}", camera.roll);
    }

    /// A face snap tweens roll back to 0 (re-uprights), regardless of the start roll.
    #[test]
    fn face_snap_resets_roll_to_zero() {
        let mut camera = OrbitCamera { roll: FRAC_PI_2, ..OrbitCamera::default() };
        let mut tween = SnapTween::to_face(&camera, CubeFace::Front);
        assert!(approx(tween.roll_to, 0.0), "face snap roll_to {}", tween.roll_to);
        assert!(tween.advance(&mut camera, 1.0));
        assert!(approx(camera.roll, 0.0), "roll after face snap {}", camera.roll);

        // An element (edge) snap also re-uprights.
        let mut camera = OrbitCamera { roll: -0.9, ..OrbitCamera::default() };
        let element = ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top);
        let mut tween = SnapTween::to_element(&camera, element);
        assert!(approx(tween.roll_to, 0.0));
        assert!(tween.advance(&mut camera, 1.0));
        assert!(approx(camera.roll, 0.0));
    }

    /// Roll normalises to (−π, π] at rest, so repeated arrow presses never grow it
    /// unbounded. Four Ccw quarter-turns net to ≈0, not 2π.
    #[test]
    fn roll_normalizes_and_does_not_grow_unbounded() {
        assert!(approx(normalize_roll(0.0), 0.0));
        assert!(approx(normalize_roll(PI), PI));
        // Add/subtract 2π is a no-op (mod 2π); avoid the exact ±π boundary, which is
        // float-ambiguous between +π and −π.
        assert!(approx(normalize_roll(2.0), 2.0));
        assert!(approx(normalize_roll(2.0 + 2.0 * PI), 2.0));
        assert!(approx(normalize_roll(2.0 - 2.0 * PI), 2.0));
        assert!(approx(normalize_roll(-2.5), -2.5));
        assert!(approx(normalize_roll(-2.5 + 4.0 * PI), -2.5));
        assert!(approx(normalize_roll(2.0 * PI), 0.0));
        // Result is always within (−π, π].
        for &r in &[5.0f32, -5.0, 10.0, -10.0, 100.0] {
            let n = normalize_roll(r);
            assert!(n > -PI - 1e-4 && n <= PI + 1e-4, "normalize_roll({r}) = {n} out of range");
        }

        // Four Ccw quarter-turns: settled roll wraps back near 0, magnitude ≤ π.
        let mut camera = OrbitCamera::default();
        for _ in 0..4 {
            let mut tween = SnapTween::roll(&camera, RollDir::Ccw);
            assert!(tween.advance(&mut camera, 1.0));
            assert!(camera.roll.abs() <= PI + 1e-4, "roll grew unbounded: {}", camera.roll);
        }
        assert!(approx(camera.roll, 0.0), "four quarter-turns should net ~0, got {}", camera.roll);
    }
}
