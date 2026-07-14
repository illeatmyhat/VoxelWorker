//! The spherical **orbit camera** rig and its control math.
//!
//! [`OrbitCamera`] parameterises the eye as a spherical orbit around a fixed
//! `target`: an azimuth `theta` in the XY ground plane, a polar angle `phi` from
//! the +Z (vertical) axis, and a `distance`. The drag/pan/zoom controls are the
//! arcball-family gestures of a modelling viewer — a pole-clamped orbit with
//! `sin(phi)` azimuth damping, a cursor-locked view-plane pan, and a multiplicative
//! zoom — plus a `roll` degree of freedom about the view axis. The load-bearing
//! subtlety is the **up vector**: near the poles a naive `Vec3::Z` up makes
//! `look_at_rh` degenerate, so [`OrbitCamera::up_vector_base`] blends smoothly to an
//! azimuth-derived horizontal up across a small band, giving a singular-frame up
//! that is continuous right through the pole.
//!
//! [`HomeView`] and [`OrbitCamera::focus_target_and_distance`] are the framing-fit
//! helpers: they frame a bounding box by the longest-axis rule so a Fit/Focus/Home
//! reproduces the same on-screen size at any model scale.
//!
//! Cite: the orbit gestures are the arcball lineage (Shoemake, "ARCBALL: A User
//! Interface for Specifying Three-Dimensional Orientation Using a Mouse", *Graphics
//! Interface* 1992); the look-at frame and the Gram–Schmidt screen-up construction
//! are standard viewing geometry (Akenine-Möller, Haines & Hoffman, *Real-Time
//! Rendering*). The projection matrices this rig feeds live in [`crate::projection`].

use glam::{Quat, Vec3};

use crate::tween::{nearest_equivalent_theta, normalize_roll, SnapTween};
use crate::view_cube::{CubeFace, CUBE_FACES};

/// Field of view (vertical) for the perspective projection, in radians. Shared with
/// [`crate::projection`], which builds the perspective matrix, and with the pan math
/// below, which derives the cursor-locked world-per-pixel from the same frustum.
pub(crate) const PERSPECTIVE_FOV_Y: f32 = std::f32::consts::FRAC_PI_4; // 45°

/// Historical pole epsilon. **No longer used by the camera math** — the snaps and
/// the drag clamp now reach the EXACT poles (`0` / `π`) and rely on
/// [`OrbitCamera::up_vector`] for a true singular-frame up instead of nudging `phi`
/// a hair short. Retained as a public constant for back-compat.
pub const POLE_EPSILON: f32 = 0.0001;

/// Drag clamp for `orbit_phi`. The drag now reaches the EXACT poles (`0.0` / `π`)
/// and stops there: the view matrix no longer degenerates at the pole because
/// [`OrbitCamera::up_vector`] supplies a true singular-frame up.
const PHI_MIN: f32 = 0.0;
const PHI_MAX: f32 = std::f32::consts::PI;

/// Half-width of the smoothstep band (in `phi` radians) over which the up vector
/// blends from `Vec3::Z` to the azimuth-derived horizontal up. Inside `[0, BAND]`
/// of the top pole (and `[π−BAND, π]` of the bottom) the blend runs; outside it
/// the up is exactly `Vec3::Z`. Small enough to be invisible, wide enough that the
/// blend is smooth (no 1-frame flip) right through the singular frame.
const UP_BLEND_BAND: f32 = 0.05;

/// Orthographic half-height factor relative to `orbit_distance` (`vh = distance *
/// 0.42`, chosen so toggling perspective ↔ orthographic keeps roughly the same
/// framing at the target). Shared with [`crate::projection`] and the pan math.
pub(crate) const ORTHO_HALF_HEIGHT_FACTOR: f32 = 0.42;

/// Which projection the orbit rig produces in [`OrbitCamera::view_projection`].
///
/// A display-only parameter: switching it never moves the camera — only the
/// projection matrix changes. Serialization of this enum for config persistence is
/// handled at the application seam (this crate carries no serde dependency).
///
/// [`OrbitCamera::view_projection`]: crate::projection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionMode {
    /// 45° vertical field-of-view perspective.
    #[default]
    Perspective,
    /// Orthographic frustum whose half-height tracks `orbit_distance`.
    Orthographic,
}

/// The saved "home" view: the orbit angles + distance the Home button returns to.
/// Defaults to the camera defaults; `from_camera` overwrites it from the live
/// camera, and the application persists it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomeView {
    pub theta: f32,
    pub phi: f32,
    pub distance: f32,
    /// Did the USER explicitly capture this home? `false` for the default home — in
    /// which case the Home button FRAMES the model (re-fits the distance) instead of
    /// using the canned default distance, so Home never zooms in too close on a
    /// model of a different size. `true` once [`Self::from_camera`] records the live
    /// view, and then the saved distance is honoured verbatim.
    pub explicitly_set: bool,
}

impl Default for HomeView {
    fn default() -> Self {
        let camera = OrbitCamera::default();
        Self {
            theta: camera.orbit_theta,
            phi: camera.orbit_phi,
            distance: camera.orbit_distance,
            explicitly_set: false,
        }
    }
}

impl HomeView {
    /// Capture the live camera's orbit angles + distance as the new home. This is
    /// an EXPLICIT user home, so its saved distance is honoured by Home (no re-fit).
    pub fn from_camera(camera: &OrbitCamera) -> Self {
        Self {
            theta: camera.orbit_theta,
            phi: camera.orbit_phi,
            distance: camera.orbit_distance,
            explicitly_set: true,
        }
    }

    /// Begin an eased snap from `camera`'s current angles toward this home view
    /// (nearest-equivalent theta, so no long spin). The caller advances the returned
    /// tween each frame and separately lerps/sets `orbit_distance` (the tween
    /// animates angles only, matching [`SnapTween`]).
    pub fn snap_tween(&self, camera: &OrbitCamera) -> SnapTween {
        SnapTween {
            theta_from: camera.orbit_theta,
            phi_from: camera.orbit_phi,
            theta_to: nearest_equivalent_theta(camera.orbit_theta, self.theta),
            phi_to: self.phi,
            // Home re-uprights too: tween roll back to 0.
            roll_from: camera.roll,
            roll_to: 0.0,
            elapsed_seconds: 0.0,
            duration_seconds: SnapTween::DEFAULT_DURATION_SECONDS,
        }
    }
}

/// Spherical orbit camera around `target`.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    /// Point the camera looks at.
    pub target: Vec3,
    /// Azimuth in the XY ground plane, radians.
    pub orbit_theta: f32,
    /// Polar angle from +Z (the vertical/up axis), radians (clamped to
    /// `[PHI_MIN, PHI_MAX]`).
    pub orbit_phi: f32,
    /// Distance from `target` to the camera eye.
    pub orbit_distance: f32,
    /// Roll about the forward/view axis (radians). Default 0 = the pole-aware base up
    /// (`Vec3::Z` away from the poles). The ViewCube roll arrows tween this by ∓90°;
    /// any face/edge/corner/Home snap re-uprights it to 0. Transient view state —
    /// NOT persisted (default 0 on load).
    pub roll: f32,
    /// Active projection (perspective by default). Display-only param.
    pub projection_mode: ProjectionMode,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            // Home view = the TOP-FRONT-LEFT iso corner: the eye sits in the
            // (−X, −Y, +Z) octant looking at the corner where the Top/Front/Left
            // faces meet (Z-up: front = −Y). `theta = −3π/4` aims the eye toward
            // (−X, −Y); `phi = acos(1/√3) ≈ 54.7°` from +Z is the symmetric corner
            // elevation so all three faces are seen equally.
            orbit_theta: -3.0 * std::f32::consts::FRAC_PI_4,
            orbit_phi: (1.0 / 3.0_f32.sqrt()).acos(),
            orbit_distance: 10.0,
            roll: 0.0,
            projection_mode: ProjectionMode::Perspective,
        }
    }
}

impl OrbitCamera {
    /// Auto-frame the camera for a grid of the given voxel dimensions:
    /// `distance = max(grid_x, grid_y, grid_z) * 1.9`.
    pub fn auto_framed_distance(grid_dimensions: [u32; 3]) -> f32 {
        let longest = grid_dimensions[0]
            .max(grid_dimensions[1])
            .max(grid_dimensions[2]) as f32;
        longest * 1.9
    }

    /// Frame a single node's AABB for the "Focus" view action. Given the node's
    /// recentred AABB `centre` (the gizmo pivot, in the recentred render frame) and
    /// its voxel `extent`, returns the `(target, distance)` the camera should adopt:
    /// the target is the node centre, and the distance reuses the SAME
    /// [`auto_framed_distance`](Self::auto_framed_distance) fit math
    /// (`longest_axis * 1.9`) so a focused node is framed exactly like a whole-scene
    /// Fit, scoped to that node. The orbit angles are left to the caller (Focus
    /// moves the pivot + distance only, like Fit). A zero-extent node yields a
    /// floored minimum distance so the camera never collapses onto the target.
    pub fn focus_target_and_distance(centre: Vec3, extent: [f32; 3]) -> (Vec3, f32) {
        let extent_dimensions = [
            extent[0].round().max(0.0) as u32,
            extent[1].round().max(0.0) as u32,
            extent[2].round().max(0.0) as u32,
        ];
        let distance = Self::auto_framed_distance(extent_dimensions).max(0.1);
        (centre, distance)
    }

    /// Unit direction from the target toward the camera eye (Z-up spherical:
    /// `phi` is the polar angle from +Z, `theta` the azimuth in the XY plane).
    pub fn direction(&self) -> Vec3 {
        let (sin_phi, cos_phi) = self.orbit_phi.sin_cos();
        let (sin_theta, cos_theta) = self.orbit_theta.sin_cos();
        Vec3::new(sin_phi * cos_theta, sin_phi * sin_theta, cos_phi)
    }

    /// Camera eye position: `target + direction * distance`.
    pub fn eye(&self) -> Vec3 {
        self.target + self.direction() * self.orbit_distance
    }

    /// The cube **face** the camera is currently looking at most head-on: the face
    /// whose outward normal is nearest (largest dot product) to the eye direction
    /// (target→eye). Used by the rotate arrows to pick the face a 90° step rotates
    /// *from*. Ties (exactly edge-/corner-on views) resolve to the first face in
    /// `CUBE_FACES` order, which is deterministic and good enough — the rotate target
    /// is then `adjacent_face` of that face.
    pub fn nearest_face(&self) -> CubeFace {
        let direction = self.direction();
        CUBE_FACES
            .iter()
            .map(|(face, _)| *face)
            .max_by(|a, b| {
                let da = direction.dot(a.normal());
                let db = direction.dot(b.normal());
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(CubeFace::Front)
    }

    /// Is the view currently **constrained to a single face** (looking nearly
    /// head-on at one of the 6 faces, upright)? The ViewCube rotate arrows are a
    /// face-relative 90°-step affordance, so — matching Fusion — they are only
    /// offered when the view is face-on. An edge/corner/arbitrary-orbit view returns
    /// `false` (no rotate arrows). The test: the eye direction is within ~8° of the
    /// nearest face's outward normal AND the view is roughly upright (roll ≈ 0), so
    /// the four screen-aligned arrows map cleanly to the face's four neighbours.
    pub fn is_face_constrained(&self) -> bool {
        // cos(8°) ≈ 0.990 — a tight cone around the face normal.
        const FACE_ALIGN_COS: f32 = 0.990;
        let aligned = self.direction().dot(self.nearest_face().normal()) >= FACE_ALIGN_COS;
        // Upright within ~8° of roll (the rotate arrows assume a screen-aligned face).
        let upright = normalize_roll(self.roll).abs() <= 0.14;
        aligned && upright
    }

    /// The up vector for `look_at_rh`, well-defined and CONTINUOUS through the
    /// poles (no `look_at` degeneracy, no roll-flip).
    ///
    /// Away from the poles this is just `Vec3::Z`. Within [`UP_BLEND_BAND`] of a
    /// pole it smoothly blends to an **azimuth-derived horizontal up** — the exact
    /// limit of "`Vec3::Z` projected onto the view plane, normalised" as
    /// `phi → 0/π`. That limit is `(−cos θ, −sin θ, 0)` at the top pole and
    /// `(cos θ, sin θ, 0)` at the bottom, so the screen "up" the user sees is the
    /// direction the camera would tilt toward, and it never jumps as the drag
    /// crosses the singular frame.
    ///
    /// At the exact TOP snap (`θ = −π/2`, `phi = 0`) this yields up `≈ (0, 1, 0)`,
    /// consistent with the Z-up TOP/BOTTOM snap convention (looking straight down
    /// at the XY ground, screen-up points toward +Y = the BACK edge).
    ///
    /// This is the **base** up (roll-free). [`Self::up_vector`] rotates it by
    /// `roll` about the forward axis; both view matrices route through that one so
    /// roll twists the scene AND the small cube together.
    pub fn up_vector_base(&self) -> Vec3 {
        use std::f32::consts::PI;
        // Distance (in phi) from the nearest pole.
        let phi = self.orbit_phi;
        let distance_from_pole = phi.min(PI - phi);
        if distance_from_pole >= UP_BLEND_BAND {
            return Vec3::Z;
        }
        // Horizontal up: the limit of projected-Z as phi → the near pole (in the XY
        // ground plane). Top pole (phi≈0): (−cosθ, −sinθ, 0); bottom (phi≈π):
        // (cosθ, sinθ, 0).
        let (sin_theta, cos_theta) = self.orbit_theta.sin_cos();
        let near_top = phi < PI - phi;
        let horizontal_up = if near_top {
            Vec3::new(-cos_theta, -sin_theta, 0.0)
        } else {
            Vec3::new(cos_theta, sin_theta, 0.0)
        };
        // smoothstep from horizontal_up (at the pole) to Vec3::Z (at the band edge).
        let t = (distance_from_pole / UP_BLEND_BAND).clamp(0.0, 1.0);
        let weight = t * t * (3.0 - 2.0 * t); // smoothstep
        let blended = horizontal_up.lerp(Vec3::Z, weight);
        // The two endpoints are orthogonal unit vectors, so the lerp is never
        // zero-length; normalise so `look_at_rh` gets a clean unit up.
        blended.normalize()
    }

    /// The up vector fed to `look_at_rh`, including the **roll DOF**.
    ///
    /// Composition: take the pole-aware base up ([`Self::up_vector_base`]) and
    /// rotate it by `roll` radians about the FORWARD axis (`normalize(target −
    /// eye)`), the axis the camera looks along. Rolling about forward keeps the up
    /// perpendicular to the view direction, so `look_at_rh` never degenerates — the
    /// whole view simply twists in screen space. `roll = 0` returns the base up
    /// unchanged (existing goldens stay byte-identical). Both
    /// [`OrbitCamera::view_projection`] and [`OrbitCamera::view_cube_view_projection`]
    /// use THIS, so the scene and the small ViewCube roll in lockstep.
    ///
    /// [`OrbitCamera::view_projection`]: crate::projection
    /// [`OrbitCamera::view_cube_view_projection`]: crate::projection
    pub fn up_vector(&self) -> Vec3 {
        let base = self.up_vector_base();
        if self.roll == 0.0 {
            return base;
        }
        // Forward axis = direction the camera looks (target − eye), i.e. the
        // negation of `direction()` (which points target → eye).
        let forward = -self.direction().normalize();
        // Roll twists the SCREEN up. `look_at_rh` only uses the component of the up
        // perpendicular to forward, so project base onto the view plane first
        // (Gram–Schmidt) and roll THAT. This makes the roll a true in-screen
        // rotation: at roll=π/2 the effective up is exactly 90° from roll=0, and the
        // result is guaranteed ⊥ forward (no `look_at` degeneracy). If base happens
        // to be parallel to forward (never, given the pole-blend), fall back to base.
        let screen_up = base - forward * base.dot(forward);
        let screen_up = screen_up.normalize_or_zero();
        if screen_up == Vec3::ZERO {
            return base;
        }
        let rolled = Quat::from_axis_angle(forward, self.roll) * screen_up;
        rolled.normalize()
    }

    /// Orbit by a screen-space drag delta (left-drag): `phi -= dy * 0.01`, with
    /// `phi` clamped to `[0, π]` — the drag reaches the EXACT poles and stops there
    /// (Fusion "Constrained Orbit"). No degeneracy: [`Self::up_vector`] supplies a
    /// true singular-frame up at the pole.
    ///
    /// Azimuth (`theta`) is damped by `sin(phi)` so the view doesn't "whip" sideways
    /// as it approaches a pole: the same horizontal drag sweeps a smaller arc the
    /// closer the eye is to straight-up/down (where azimuth is degenerate).
    pub fn orbit_by_drag(&mut self, delta_x: f32, delta_y: f32) {
        let azimuth_damping = self.orbit_phi.sin().max(0.0);
        self.orbit_theta -= delta_x * 0.01 * azimuth_damping;
        self.orbit_phi = (self.orbit_phi - delta_y * 0.01).clamp(PHI_MIN, PHI_MAX);
    }

    /// Pan by a screen-space drag delta (middle-drag): slide `target` (and with it
    /// the eye, since `eye = target + direction·distance`) within the camera's view
    /// plane, so the grabbed point stays locked under the cursor as the model
    /// translates without rotating.
    ///
    /// The view-plane basis is the SAME orthonormal frame `look_at_rh` builds:
    /// `right = forward × up_vector()`, then the true screen up is `right × forward`.
    /// Using the orthogonalised screen up — NOT the raw `up_vector()`, which is only
    /// perpendicular to `forward` when the view is level — is what keeps a vertical
    /// drag moving straight up ON SCREEN at any tilt. With the raw up the pan drifts
    /// along world-Z and the cursor slips off the grabbed point (worst in
    /// orthographic, where there's no perspective foreshortening to hide the drift).
    ///
    /// `viewport_height_px` makes the pan cursor-locked (1:1): `world_per_pixel` is
    /// the world span of one screen pixel at the target plane, derived per
    /// projection (the ortho half-height, or the perspective frustum height at
    /// `orbit_distance`). Both evaluate to ≈`0.83·distance / height`, so the SAME
    /// drag tracks the cursor identically in either mode.
    ///
    /// Dragging right (`delta_x > 0`) grabs the scene and pulls it right, so the
    /// target slides LEFT (`−right`). Winit's screen Y is down-positive, so dragging
    /// down (`delta_y > 0`) pulls the scene down and the target slides UP
    /// (`+screen_up`).
    pub fn pan_by_drag(&mut self, delta_x: f32, delta_y: f32, viewport_height_px: f32) {
        let forward = -self.direction();
        let right = forward.cross(self.up_vector()).normalize_or_zero();
        let screen_up = right.cross(forward).normalize_or_zero();
        let height = viewport_height_px.max(1.0);
        let world_per_pixel = match self.projection_mode {
            ProjectionMode::Orthographic => {
                2.0 * self.orbit_distance * ORTHO_HALF_HEIGHT_FACTOR / height
            }
            ProjectionMode::Perspective => {
                2.0 * self.orbit_distance * (PERSPECTIVE_FOV_Y * 0.5).tan() / height
            }
        };
        self.target += (-right * delta_x + screen_up * delta_y) * world_per_pixel;
    }

    /// Zoom by a wheel step: `distance *= 1 ± 0.08`. Positive `scroll_lines`
    /// zooms in (closer).
    pub fn zoom_by_wheel(&mut self, scroll_lines: f32) {
        let factor = if scroll_lines > 0.0 { 1.0 - 0.08 } else { 1.0 + 0.08 };
        self.orbit_distance = (self.orbit_distance * factor).max(0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view_cube::{CubeFace, ViewCubeElement, CUBE_FACES};
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn drag_clamp_reaches_exact_poles() {
        // Dragging straight up from near-pole should clamp to the EXACT top pole
        // (phi = 0), not POLE_EPSILON and not the old 0.05 floor.
        let mut camera = OrbitCamera {
            orbit_phi: 0.2,
            ..OrbitCamera::default()
        };
        // A big upward drag (negative dy reduces phi toward the top pole).
        camera.orbit_by_drag(0.0, 1000.0);
        assert!(approx(camera.orbit_phi, 0.0), "phi = {}", camera.orbit_phi);

        // And the bottom pole — exact π.
        let mut camera = OrbitCamera::default();
        camera.orbit_by_drag(0.0, -1000.0);
        assert!(approx(camera.orbit_phi, PI), "phi = {}", camera.orbit_phi);
    }

    #[test]
    fn pan_moves_along_the_screen_basis_and_scales_with_distance() {
        // The default view is TILTED (phi ≈ 1.05), so up_vector() (≈ +Z) is NOT
        // perpendicular to the look direction — the exact case the screen-up
        // orthogonalisation fixes. The pan must follow the look_at_rh frame:
        // right = forward × up, screen_up = right × forward.
        let height = 1000.0;
        let camera = OrbitCamera::default();
        let forward = -camera.direction();
        let right = forward.cross(camera.up_vector()).normalize();
        let screen_up = right.cross(forward).normalize();

        // A pure horizontal drag right grabs the scene and pulls it right, so the
        // target slides LEFT (−right), with NO vertical (screen_up) component and
        // nothing along the look axis.
        let mut horizontal = OrbitCamera::default();
        horizontal.pan_by_drag(10.0, 0.0, height);
        let moved = horizontal.target;
        assert!(moved.dot(right) < 0.0, "drag right slides the target left");
        assert!(approx(moved.dot(screen_up), 0.0), "a horizontal drag has no vertical pan");
        assert!(approx(moved.dot(forward), 0.0), "pan stays in the view plane");

        // A pure vertical drag DOWN (winit screen-y is down-positive) pulls the
        // scene down, so the target slides UP — along SCREEN up, in the view plane,
        // with no horizontal (right) component. The regression guard: it must NOT be
        // pure world-Z (the vertical axis), which on this tilted view has screen_up·Z < 1.
        let mut vertical = OrbitCamera::default();
        vertical.pan_by_drag(0.0, 10.0, height);
        let moved = vertical.target;
        assert!(moved.dot(screen_up) > 0.0, "drag down slides the target up-screen");
        assert!(approx(moved.dot(right), 0.0), "a vertical drag has no horizontal pan");
        assert!(approx(moved.dot(forward), 0.0), "pan stays in the view plane");
        // The vertical pan is genuinely TILTED off world-Z (the bug was using raw up).
        assert!(
            moved.normalize().dot(Vec3::Z) < 0.999,
            "vertical pan must track screen-up, not pure world-Z",
        );

        // Pan scales with orbit_distance: the SAME drag at twice the distance moves
        // the target exactly twice as far (cursor-locked at any zoom).
        let mut near = OrbitCamera { orbit_distance: 5.0, ..OrbitCamera::default() };
        let mut far = OrbitCamera { orbit_distance: 10.0, ..OrbitCamera::default() };
        near.pan_by_drag(7.0, -3.0, height);
        far.pan_by_drag(7.0, -3.0, height);
        assert!(
            approx(far.target.length(), 2.0 * near.target.length()),
            "pan must scale linearly with orbit_distance",
        );
    }

    #[test]
    fn focus_target_and_distance_centres_and_fits_node() {
        // Focus sets the target to the node centre and fits the distance from the
        // longest extent axis via the same `auto_framed_distance` math (longest×1.9).
        let centre = Vec3::new(3.0, -2.0, 5.0);
        let (target, distance) = OrbitCamera::focus_target_and_distance(centre, [4.0, 12.0, 6.0]);
        assert_eq!(target, centre);
        assert!(approx(distance, 12.0 * 1.9), "distance = {distance}");
    }

    #[test]
    fn focus_target_and_distance_floors_zero_extent() {
        // A node with no resolvable extent must not collapse the camera onto the
        // target (distance 0) — it is floored to a small minimum.
        let centre = Vec3::ZERO;
        let (target, distance) = OrbitCamera::focus_target_and_distance(centre, [0.0, 0.0, 0.0]);
        assert_eq!(target, centre);
        assert!(distance >= 0.1, "distance = {distance}");
    }

    /// At every phi the up vector must be finite, unit-length, and not parallel
    /// to the view direction (else `look_at_rh` degenerates).
    #[test]
    fn up_vector_is_finite_unit_and_non_parallel_to_view() {
        for &phi in &[0.0f32, 0.0001, 0.04, 0.06, FRAC_PI_2, PI - 0.0001, PI] {
            let camera = OrbitCamera {
                orbit_theta: -FRAC_PI_2,
                orbit_phi: phi,
                ..OrbitCamera::default()
            };
            let up = camera.up_vector();
            assert!(up.is_finite(), "up not finite at phi={phi}: {up:?}");
            assert!(approx(up.length(), 1.0), "up not unit at phi={phi}: len {}", up.length());
            // View direction is -direction() (camera looks from eye toward target).
            let view_dir = -camera.direction();
            let parallelism = up.dot(view_dir).abs();
            assert!(
                parallelism < 0.999,
                "up parallel to view at phi={phi}: dot {parallelism}"
            );
        }
    }

    /// Continuity across the blend band: the up vector must be CONTINUOUS through
    /// the singular frame — no 1-frame flip.
    #[test]
    fn up_vector_is_continuous_across_pole_band() {
        let make = |phi: f32| OrbitCamera {
            orbit_theta: 0.7,
            orbit_phi: phi,
            ..OrbitCamera::default()
        }
        .up_vector();
        // Straddle the band edge (UP_BLEND_BAND = 0.05): close, never a flip.
        let inside = make(0.04);
        let outside = make(0.06);
        assert!(
            (inside - outside).length() < 0.15,
            "up flipped across band edge: {inside:?} vs {outside:?}"
        );
        // Dense sweep: every 0.002-rad step changes up by less than the smooth
        // Lipschitz bound (no jump).
        const STEP_CEILING: f32 = 0.12;
        let mut previous = make(0.0);
        let mut phi = 0.0f32;
        while phi <= UP_BLEND_BAND + 0.02 {
            let current = make(phi);
            assert!(
                (current - previous).length() < STEP_CEILING,
                "up jumped between phi steps near {phi}: {previous:?} -> {current:?}"
            );
            previous = current;
            phi += 0.002;
        }
        // Bottom pole sweep behaves identically.
        let mut previous = make(PI);
        let mut phi = PI;
        while phi >= PI - UP_BLEND_BAND - 0.02 {
            let current = make(phi);
            assert!(
                (current - previous).length() < STEP_CEILING,
                "bottom up jumped near {phi}: {previous:?} -> {current:?}"
            );
            previous = current;
            phi -= 0.002;
        }
    }

    /// Z-up: at the exact TOP snap (theta=-π/2, phi=0) the up limit is (0,1,0) and
    /// at the BOTTOM snap (theta=-π/2, phi=π) it is (0,-1,0) — the documented
    /// convention (screen-up points toward ±Y in the ground plane).
    #[test]
    fn up_vector_at_exact_pole_snaps_matches_convention() {
        let top = OrbitCamera {
            orbit_theta: -FRAC_PI_2,
            orbit_phi: 0.0,
            ..OrbitCamera::default()
        }
        .up_vector();
        // (-cos(-π/2), -sin(-π/2), 0) = (0, 1, 0).
        assert!(approx(top.x, 0.0) && approx(top.y, 1.0) && approx(top.z, 0.0), "{top:?}");

        let bottom = OrbitCamera {
            orbit_theta: -FRAC_PI_2,
            orbit_phi: PI,
            ..OrbitCamera::default()
        }
        .up_vector();
        // (cos(-π/2), sin(-π/2), 0) = (0, -1, 0).
        assert!(
            approx(bottom.x, 0.0) && approx(bottom.y, -1.0) && approx(bottom.z, 0.0),
            "{bottom:?}"
        );
    }

    /// Away from the poles the up vector is exactly Vec3::Z (Z-up: the vertical axis).
    #[test]
    fn up_vector_away_from_poles_is_exactly_z() {
        for &phi in &[0.1f32, 0.5, 1.05, FRAC_PI_2, 2.5, PI - 0.1] {
            let up = OrbitCamera {
                orbit_phi: phi,
                ..OrbitCamera::default()
            }
            .up_vector();
            assert_eq!(up, Vec3::Z, "phi={phi} should give exact Vec3::Z");
        }
    }

    /// At each face's snap orientation the camera's nearest face is that face.
    #[test]
    fn nearest_face_at_each_face_snap_is_that_face() {
        for (face, _) in CUBE_FACES {
            let (theta, phi) = face.snap_angles();
            let camera = OrbitCamera {
                orbit_theta: theta,
                orbit_phi: phi,
                ..OrbitCamera::default()
            };
            assert_eq!(camera.nearest_face(), face, "nearest at {face:?} snap");
        }
    }

    /// `roll = π/2` rotates the SCREEN up exactly 90° about the forward axis.
    #[test]
    fn up_vector_roll_quarter_is_perpendicular_to_unrolled() {
        let camera0 = OrbitCamera { roll: 0.0, ..OrbitCamera::default() };
        let rolled = OrbitCamera { roll: FRAC_PI_2, ..OrbitCamera::default() }.up_vector();
        // The unrolled SCREEN up: base up projected ⊥ forward (what look_at uses).
        let forward = -camera0.direction().normalize();
        let base = camera0.up_vector_base();
        let screen_up0 = (base - forward * base.dot(forward)).normalize();
        assert!(rolled.is_finite(), "rolled up not finite: {rolled:?}");
        assert!(approx(rolled.length(), 1.0), "rolled up not unit: {}", rolled.length());
        // 90° apart → dot ≈ 0.
        assert!(
            screen_up0.dot(rolled).abs() < 1e-3,
            "roll=π/2 up not perpendicular to roll=0 screen up: dot {}",
            screen_up0.dot(rolled)
        );
    }

    /// `roll = 0` returns the base up unchanged (goldens stay byte-identical).
    #[test]
    fn up_vector_roll_zero_equals_base() {
        for &phi in &[0.1f32, 0.5, 1.05, FRAC_PI_2, 2.5] {
            let camera = OrbitCamera { orbit_phi: phi, roll: 0.0, ..OrbitCamera::default() };
            assert_eq!(camera.up_vector(), camera.up_vector_base(), "phi={phi}");
        }
    }

    /// The rolled up stays unit-length, finite and perpendicular to the view
    /// direction across a range of roll angles (so `look_at_rh` never degenerates).
    #[test]
    fn up_vector_finite_unit_and_non_parallel_under_roll() {
        for &roll in &[0.0f32, FRAC_PI_2, PI, -FRAC_PI_2, 0.3] {
            let camera = OrbitCamera { roll, ..OrbitCamera::default() };
            let up = camera.up_vector();
            assert!(up.is_finite(), "up not finite at roll={roll}: {up:?}");
            assert!(approx(up.length(), 1.0), "up not unit at roll={roll}: {}", up.length());
            // For any NONZERO roll the up is orthogonalised against forward (rolled
            // screen-up), so it is exactly ⊥ the view direction — `look_at_rh` never
            // degenerates. (roll=0 keeps the raw base up, which look_at re-orthogonalises
            // itself; it need not be ⊥ forward, matching pre-roll behaviour.)
            if roll != 0.0 {
                let view_dir = -camera.direction();
                assert!(
                    up.dot(view_dir).abs() < 1e-3,
                    "rolled up not perpendicular to view at roll={roll}: {}",
                    up.dot(view_dir)
                );
            }
        }
    }

    /// The view is face-constrained at each face snap (upright, head-on) and NOT at
    /// an edge/corner view or a rolled view.
    #[test]
    fn is_face_constrained_only_at_upright_face_views() {
        for (face, _) in CUBE_FACES {
            let (theta, phi) = face.snap_angles();
            let camera = OrbitCamera { orbit_theta: theta, orbit_phi: phi, ..OrbitCamera::default() };
            assert!(camera.is_face_constrained(), "should be face-on at {face:?}");
        }
        // An edge view (front-top) is NOT face-constrained.
        let (theta, phi) =
            ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top).snap_angles();
        let edge_camera = OrbitCamera { orbit_theta: theta, orbit_phi: phi, ..OrbitCamera::default() };
        assert!(!edge_camera.is_face_constrained(), "edge view must not be face-on");
        // A corner view is NOT face-constrained.
        let (theta, phi) =
            ViewCubeElement::from_corner(CubeFace::Front, CubeFace::Top, CubeFace::Right)
                .snap_angles();
        let corner_camera = OrbitCamera { orbit_theta: theta, orbit_phi: phi, ..OrbitCamera::default() };
        assert!(!corner_camera.is_face_constrained(), "corner view must not be face-on");
        // Face-on but ROLLED 90° is not constrained (the screen arrows wouldn't align).
        let (theta, phi) = CubeFace::Front.snap_angles();
        let rolled = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            roll: FRAC_PI_2,
            ..OrbitCamera::default()
        };
        assert!(!rolled.is_face_constrained(), "rolled face view must not be face-on");
    }

    /// The default home is NOT explicitly set (so Home re-fits), while a home
    /// captured from the camera IS explicitly set (so Home uses its distance).
    #[test]
    fn home_view_explicit_flag_tracks_origin() {
        assert!(!HomeView::default().explicitly_set, "default home is implicit");
        let camera = OrbitCamera::default();
        assert!(HomeView::from_camera(&camera).explicitly_set, "captured home is explicit");
    }

    #[test]
    fn home_view_default_matches_camera_defaults() {
        let home = HomeView::default();
        let camera = OrbitCamera::default();
        assert!(approx(home.theta, camera.orbit_theta));
        assert!(approx(home.phi, camera.orbit_phi));
        assert!(approx(home.distance, camera.orbit_distance));
    }

    /// Z-up convention pins (the central guarantee of the reorientation):
    ///  * the base up away from the poles is exactly `Vec3::Z`;
    ///  * a top-down view's nearest face is `Top` with outward normal +Z;
    ///  * the ground-plane / front-face normals are +Z / −Y respectively.
    #[test]
    fn z_up_convention_holds() {
        // (1) up_vector_base ≈ Vec3::Z at a normal (non-pole) tilt.
        let camera = OrbitCamera { orbit_phi: 1.0, ..OrbitCamera::default() };
        assert_eq!(camera.up_vector_base(), Vec3::Z, "base up must be +Z");

        // (2) A top-down view (snapped to TOP) has nearest face Top, normal +Z.
        let (theta, phi) = CubeFace::Top.snap_angles();
        let top_down = OrbitCamera { orbit_theta: theta, orbit_phi: phi, ..OrbitCamera::default() };
        assert_eq!(top_down.nearest_face(), CubeFace::Top, "top-down nearest face");
        assert_eq!(CubeFace::Top.normal(), Vec3::Z, "TOP normal is +Z");

        // (3) FRONT normal = −Y (the front view looks along +Y); ground plane up +Z.
        assert_eq!(CubeFace::Front.normal(), Vec3::NEG_Y, "FRONT normal is −Y");
        assert_eq!(CubeFace::Bottom.normal(), Vec3::NEG_Z, "BOTTOM normal is −Z");

        // (4) The default view looks DOWN at the XY ground (eye above, +Z), 3/4-ish.
        let eye_dir = OrbitCamera::default().direction();
        assert!(eye_dir.z > 0.0, "default eye is above the ground (Z>0): {eye_dir:?}");
    }

    #[test]
    fn set_home_then_move_then_snap_targets_saved_angles() {
        // Capture home at a custom view, then move the camera elsewhere.
        let mut camera = OrbitCamera {
            orbit_theta: 1.2,
            orbit_phi: 0.8,
            orbit_distance: 25.0,
            ..OrbitCamera::default()
        };
        let home = HomeView::from_camera(&camera);
        camera.orbit_theta = 3.0;
        camera.orbit_phi = 1.5;
        // Snapping home must land back on the saved angles.
        let mut tween = home.snap_tween(&camera);
        assert!(tween.advance(&mut camera, 1.0));
        assert!(approx(camera.orbit_theta, home.theta), "theta {}", camera.orbit_theta);
        assert!(approx(camera.orbit_phi, home.phi), "phi {}", camera.orbit_phi);
    }
}
