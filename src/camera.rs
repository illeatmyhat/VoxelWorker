//! The orbit camera rig (ARCHITECTURE.md §4).
//!
//! A spherical orbit around a fixed `target` (the origin in M2). Milestone 2
//! ships only the perspective projection; the orthographic branch and the view
//! cube arrive later. The rig produces a single `view_projection` matrix that is
//! uploaded to the shader uniform — render-target-agnostic, identical for the
//! window and the headless capture.

use glam::{Mat4, Vec3};

/// Field of view (vertical) for the perspective projection, in radians.
const PERSPECTIVE_FOV_Y: f32 = std::f32::consts::FRAC_PI_4; // 45°

/// Tiny epsilon used at the poles so the look direction never becomes exactly
/// parallel to the up vector (which would make `look_at_rh` degenerate). Both the
/// view-cube TOP/BOTTOM snaps and the Fusion-style "constrained orbit" drag clamp
/// land on this value, so a drag can reach effectively-dead-center top/bottom and
/// STOP there (no inversion past the pole).
pub const POLE_EPSILON: f32 = 0.0001;

/// Drag clamp for `orbit_phi`. Unlike the old `[0.05, π−0.05]` clamp (which
/// stopped a drag well short of the poles), the drag now reaches `POLE_EPSILON`
/// from each pole — the same epsilon the snaps target — so dragging can park the
/// camera looking straight down/up and stay there.
const PHI_MIN: f32 = POLE_EPSILON;
const PHI_MAX: f32 = std::f32::consts::PI - POLE_EPSILON;

/// Orthographic half-height factor relative to `orbit_distance`
/// (ARCHITECTURE.md §4: `vh = distance * 0.42`, chosen so toggling perspective ↔
/// orthographic keeps roughly the same framing at the target).
const ORTHO_HALF_HEIGHT_FACTOR: f32 = 0.42;

/// The six view-cube faces, in `materialIndex` order (+X, -X, +Y, -Y, +Z, -Z).
///
/// Index order matches the prototype's `CUBELABELS` / `FACE_VIEW` arrays so a
/// raycast hit's material index maps straight to a [`CubeFace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFace {
    /// +X — RIGHT.
    Right,
    /// -X — LEFT.
    Left,
    /// +Y — TOP.
    Top,
    /// -Y — BOTTOM.
    Bottom,
    /// +Z — FRONT.
    Front,
    /// -Z — BACK.
    Back,
}

/// The view-cube faces in `materialIndex` order, with their human labels.
pub const CUBE_FACES: [(CubeFace, &str); 6] = [
    (CubeFace::Right, "RIGHT"),
    (CubeFace::Left, "LEFT"),
    (CubeFace::Top, "TOP"),
    (CubeFace::Bottom, "BOTTOM"),
    (CubeFace::Front, "FRONT"),
    (CubeFace::Back, "BACK"),
];

impl CubeFace {
    /// Map a 0..5 material index (raycast hit) to a face, matching the prototype
    /// `materialIndex` order (+X, -X, +Y, -Y, +Z, -Z).
    pub fn from_material_index(index: usize) -> Option<Self> {
        CUBE_FACES.get(index).map(|(face, _)| *face)
    }

    /// The snap target `(theta, phi)` for this face (ARCHITECTURE.md §4
    /// `FACE_VIEW`). Polar values use a tiny epsilon at the poles so the view
    /// matrix never degenerates (look direction parallel to up).
    pub fn snap_angles(self) -> (f32, f32) {
        ViewCubeElement::from_face(self).snap_angles()
    }

    /// The outward unit normal of this face (+X,-X,+Y,-Y,+Z,-Z).
    pub fn normal(self) -> Vec3 {
        match self {
            CubeFace::Right => Vec3::X,
            CubeFace::Left => Vec3::NEG_X,
            CubeFace::Top => Vec3::Y,
            CubeFace::Bottom => Vec3::NEG_Y,
            CubeFace::Front => Vec3::Z,
            CubeFace::Back => Vec3::NEG_Z,
        }
    }
}

/// A clickable element of the view cube: a single **face** (1 normal), an **edge**
/// (2 adjacent face normals) or a **corner** (3 face normals). The standard
/// Autodesk ViewCube hot-zone model divides each face into a 3×3 grid — the centre
/// zone is the face, the 4 edge zones are edges (45° edge-on views shared with the
/// neighbour) and the 4 corner zones are corners (isometric 3-face views).
///
/// One element thus addresses any of the 26 cube orientations (6 faces + 12 edges
/// + 8 corners) uniformly through [`ViewCubeElement::snap_direction`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewCubeElement {
    /// The 1–3 faces whose normals define this element. Only `normals[..count]`
    /// are meaningful.
    faces: [CubeFace; 3],
    /// How many of `faces` are populated (1 = face, 2 = edge, 3 = corner).
    count: u8,
}

impl ViewCubeElement {
    /// A single-face element (centre zone of a face).
    pub fn from_face(face: CubeFace) -> Self {
        Self { faces: [face, face, face], count: 1 }
    }

    /// An edge element shared by two adjacent faces.
    pub fn from_edge(first: CubeFace, second: CubeFace) -> Self {
        Self { faces: [first, second, second], count: 2 }
    }

    /// A corner element shared by three mutually-adjacent faces.
    pub fn from_corner(first: CubeFace, second: CubeFace, third: CubeFace) -> Self {
        Self { faces: [first, second, third], count: 3 }
    }

    /// The faces composing this element (`&faces[..count]`).
    pub fn faces(&self) -> &[CubeFace] {
        &self.faces[..self.count as usize]
    }

    /// Is this a pure pole element (the TOP-only or BOTTOM-only face)? At the
    /// poles azimuth is undefined, so we special-case theta below.
    fn is_pole(&self) -> bool {
        self.count == 1 && matches!(self.faces[0], CubeFace::Top | CubeFace::Bottom)
    }

    /// The unnormalised view direction: the sum of the element's face normals.
    /// Pointing from the target toward the eye, so the camera looks back along it.
    pub fn snap_direction(&self) -> Vec3 {
        self.faces()
            .iter()
            .fold(Vec3::ZERO, |sum, face| sum + face.normal())
    }

    /// Convert this element's direction into orbit `(theta, phi)`.
    ///
    /// Unified spherical conversion `phi = acos(dir.y)`, `theta = atan2(dir.z,
    /// dir.x)` — works for faces, edges AND corners. The pure poles (dir = ±Y)
    /// special-case theta (undefined there) and clamp phi to `POLE_EPSILON` /
    /// `π − POLE_EPSILON` so the view matrix never degenerates, matching the
    /// historical TOP/BOTTOM snap table.
    pub fn snap_angles(&self) -> (f32, f32) {
        use std::f32::consts::{FRAC_PI_2, PI};
        if self.is_pole() {
            return match self.faces[0] {
                CubeFace::Top => (-FRAC_PI_2, POLE_EPSILON),
                _ => (-FRAC_PI_2, PI - POLE_EPSILON),
            };
        }
        let direction = self.snap_direction().normalize();
        let phi = direction.y.clamp(-1.0, 1.0).acos();
        let theta = direction.z.atan2(direction.x);
        (theta, phi)
    }
}

/// Pick the equivalent of `target_theta` (mod 2π) nearest to `current_theta`,
/// so a snap never spins the long way round (ARCHITECTURE.md §4: "add/sub 2π
/// before tweening"). Mirrors the prototype `snapTo` loop.
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

/// easeInOutQuad over `t` in `[0, 1]` (prototype `applyTween`).
pub fn ease_in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

/// An in-progress eased camera snap from `(theta, phi)` toward a face's view.
///
/// The windowed app advances it each frame; the headless `shot` path skips the
/// tween and applies the final angles directly. `theta_to` is already the
/// nearest-equivalent target (no long spins).
#[derive(Debug, Clone, Copy)]
pub struct SnapTween {
    pub theta_from: f32,
    pub phi_from: f32,
    pub theta_to: f32,
    pub phi_to: f32,
    /// Seconds elapsed since the tween started.
    pub elapsed_seconds: f32,
    /// Total duration in seconds (~0.38 s, ARCHITECTURE.md §4).
    pub duration_seconds: f32,
}

impl SnapTween {
    /// Tween duration in seconds (~380 ms, ARCHITECTURE.md §4).
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
        progress >= 1.0
    }
}

/// Which projection the orbit rig produces in [`OrbitCamera::view_projection`].
///
/// A display-only param (ARCHITECTURE.md §4): switching it never rebuilds the
/// voxel grid and never moves the camera — only the projection matrix changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ProjectionMode {
    /// 45° vertical field-of-view perspective.
    #[default]
    Perspective,
    /// Orthographic frustum whose half-height tracks `orbit_distance`.
    Orthographic,
}

/// Spherical orbit camera around `target`.
#[derive(Debug, Clone, Copy)]
pub struct OrbitCamera {
    /// Point the camera looks at (origin in M2).
    pub target: Vec3,
    /// Azimuth, radians.
    pub orbit_theta: f32,
    /// Polar angle from +Y, radians (clamped to `[PHI_MIN, PHI_MAX]`).
    pub orbit_phi: f32,
    /// Distance from `target` to the camera eye.
    pub orbit_distance: f32,
    /// Active projection (perspective by default). Display-only param.
    pub projection_mode: ProjectionMode,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            orbit_theta: 0.7,
            orbit_phi: 1.05,
            orbit_distance: 10.0,
            projection_mode: ProjectionMode::Perspective,
        }
    }
}

impl OrbitCamera {
    /// Auto-frame the camera for a grid of the given voxel dimensions:
    /// `distance = max(grid_x, grid_y, grid_z) * 1.9` (ARCHITECTURE.md §4).
    pub fn auto_framed_distance(grid_dimensions: [u32; 3]) -> f32 {
        let longest = grid_dimensions[0]
            .max(grid_dimensions[1])
            .max(grid_dimensions[2]) as f32;
        longest * 1.9
    }

    /// Unit direction from the target toward the camera eye.
    pub fn direction(&self) -> Vec3 {
        let (sin_phi, cos_phi) = self.orbit_phi.sin_cos();
        let (sin_theta, cos_theta) = self.orbit_theta.sin_cos();
        Vec3::new(sin_phi * cos_theta, cos_phi, sin_phi * sin_theta)
    }

    /// Camera eye position: `target + direction * distance`.
    pub fn eye(&self) -> Vec3 {
        self.target + self.direction() * self.orbit_distance
    }

    /// Orbit by a screen-space drag delta (left-drag): `phi -= dy * 0.01`, with
    /// `phi` clamped to `[PHI_MIN, PHI_MAX]` (now `POLE_EPSILON`-tight, so a drag
    /// reaches the poles and stops there — Fusion "Constrained Orbit").
    ///
    /// Azimuth (`theta`) is damped by `sin(phi)` so the view doesn't "whip"
    /// sideways as it approaches a pole: the same horizontal drag sweeps a smaller
    /// arc the closer the eye is to straight-up/down (where azimuth is degenerate).
    pub fn orbit_by_drag(&mut self, delta_x: f32, delta_y: f32) {
        let azimuth_damping = self.orbit_phi.sin().max(0.0);
        self.orbit_theta -= delta_x * 0.01 * azimuth_damping;
        self.orbit_phi = (self.orbit_phi - delta_y * 0.01).clamp(PHI_MIN, PHI_MAX);
    }

    /// Zoom by a wheel step: `distance *= 1 ± 0.08`. Positive `scroll_lines`
    /// zooms in (closer).
    pub fn zoom_by_wheel(&mut self, scroll_lines: f32) {
        let factor = if scroll_lines > 0.0 { 1.0 - 0.08 } else { 1.0 + 0.08 };
        self.orbit_distance = (self.orbit_distance * factor).max(0.1);
    }

    /// Build the combined `view_projection` matrix for an aspect ratio (w/h).
    ///
    /// The projection branch is chosen by [`OrbitCamera::projection_mode`]; the
    /// orthographic frustum tracks `orbit_distance` so zoom keeps working and the
    /// framing is preserved when toggling (ARCHITECTURE.md §4).
    pub fn view_projection(&self, aspect_ratio: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        // Near/far chosen generously relative to the auto-framed distance so the
        // grid never clips at any zoom we allow.
        let near = (self.orbit_distance * 0.01).max(0.05);
        let far = self.orbit_distance * 10.0 + 1000.0;
        let projection = match self.projection_mode {
            ProjectionMode::Perspective => {
                Mat4::perspective_rh(PERSPECTIVE_FOV_Y, aspect_ratio, near, far)
            }
            ProjectionMode::Orthographic => {
                let half_height = self.orbit_distance * ORTHO_HALF_HEIGHT_FACTOR;
                let half_width = half_height * aspect_ratio;
                Mat4::orthographic_rh(
                    -half_width,
                    half_width,
                    -half_height,
                    half_height,
                    near,
                    far,
                )
            }
        };
        projection * view
    }

    /// View-projection for the corner view cube (ARCHITECTURE.md §4): an
    /// orthographic camera whose eye copies the MAIN camera's *direction*
    /// (`pos = direction * 4`, look at origin), so the small cube mirrors the
    /// current main view. Independent of `orbit_distance` / projection mode.
    pub fn view_cube_view_projection(&self) -> Mat4 {
        let eye = self.direction() * 4.0;
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        // Half-extent 1.35 frames a cube of side 1.4 with a little margin
        // (prototype `OrthographicCamera(-1.35, 1.35, 1.35, -1.35, …)`).
        let projection = Mat4::orthographic_rh(-1.35, 1.35, -1.35, 1.35, 0.1, 100.0);
        projection * view
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
    fn material_index_maps_to_faces_in_order() {
        assert_eq!(CubeFace::from_material_index(0), Some(CubeFace::Right));
        assert_eq!(CubeFace::from_material_index(2), Some(CubeFace::Top));
        assert_eq!(CubeFace::from_material_index(5), Some(CubeFace::Back));
        assert_eq!(CubeFace::from_material_index(6), None);
    }

    #[test]
    fn front_face_points_down_positive_z() {
        // FRONT = +Z: snapping should put the eye on +Z looking back at origin.
        let (theta, phi) = CubeFace::Front.snap_angles();
        assert!(approx(theta, FRAC_PI_2));
        assert!(approx(phi, FRAC_PI_2));
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let direction = camera.direction();
        // direction = (sin·cos, cos, sin·sin) → ~(0, 0, 1).
        assert!(approx(direction.x, 0.0));
        assert!(approx(direction.y, 0.0));
        assert!(approx(direction.z, 1.0));
    }

    #[test]
    fn top_face_points_down_positive_y() {
        let (theta, phi) = CubeFace::Top.snap_angles();
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let direction = camera.direction();
        assert!(approx(direction.y, 1.0));
        assert!(direction.x.abs() < 1e-3 && direction.z.abs() < 1e-3);
    }

    #[test]
    fn right_face_points_down_positive_x() {
        let (theta, phi) = CubeFace::Right.snap_angles();
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let direction = camera.direction();
        assert!(approx(direction.x, 1.0));
        assert!(direction.y.abs() < 1e-3 && direction.z.abs() < 1e-3);
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
    fn drag_clamp_reaches_pole_epsilon_not_old_limit() {
        // Dragging straight up from near-pole should clamp to POLE_EPSILON (the
        // tight, snap-matching limit), NOT the old 0.05 floor.
        let mut camera = OrbitCamera {
            orbit_phi: 0.2,
            ..OrbitCamera::default()
        };
        // A big upward drag (negative dy reduces phi toward the top pole).
        camera.orbit_by_drag(0.0, 1000.0);
        assert!(approx(camera.orbit_phi, POLE_EPSILON), "phi = {}", camera.orbit_phi);
        assert!(camera.orbit_phi < 0.05, "drag must now pass the old 0.05 floor");

        // And the bottom pole.
        let mut camera = OrbitCamera::default();
        camera.orbit_by_drag(0.0, -1000.0);
        assert!(approx(camera.orbit_phi, PI - POLE_EPSILON));
    }

    #[test]
    fn faces_match_old_snap_table() {
        // The unified element snap must reproduce the historical face angles.
        let expected = [
            (CubeFace::Right, (0.0, FRAC_PI_2)),
            (CubeFace::Left, (PI, FRAC_PI_2)),
            (CubeFace::Top, (-FRAC_PI_2, POLE_EPSILON)),
            (CubeFace::Bottom, (-FRAC_PI_2, PI - POLE_EPSILON)),
            (CubeFace::Front, (FRAC_PI_2, FRAC_PI_2)),
            (CubeFace::Back, (-FRAC_PI_2, FRAC_PI_2)),
        ];
        for (face, (theta, phi)) in expected {
            let (got_theta, got_phi) = face.snap_angles();
            assert!(approx(got_theta, theta), "{face:?} theta {got_theta} != {theta}");
            assert!(approx(got_phi, phi), "{face:?} phi {got_phi} != {phi}");
        }
    }

    #[test]
    fn edge_snap_direction_front_top() {
        // FRONT (+Z) + TOP (+Y) → dir (0, .707, .707): phi = π/4, theta = π/2.
        let element = ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top);
        let (theta, phi) = element.snap_angles();
        assert!(approx(phi, std::f32::consts::FRAC_PI_4), "phi = {phi}");
        assert!(approx(theta, FRAC_PI_2), "theta = {theta}");
        // Order-independence: the same edge the other way round.
        let (theta2, phi2) = ViewCubeElement::from_edge(CubeFace::Top, CubeFace::Front).snap_angles();
        assert!(approx(theta, theta2) && approx(phi, phi2));
    }

    #[test]
    fn corner_snap_direction_front_top_right_is_isometric() {
        // FRONT (+Z) + TOP (+Y) + RIGHT (+X) → dir (1,1,1)/√3: all components
        // positive, an isometric view. phi = acos(1/√3) ≈ 0.9553.
        let element =
            ViewCubeElement::from_corner(CubeFace::Front, CubeFace::Top, CubeFace::Right);
        let direction = element.snap_direction().normalize();
        assert!(direction.x > 0.0 && direction.y > 0.0 && direction.z > 0.0);
        let (theta, phi) = element.snap_angles();
        // Rebuild the eye direction from the snapped angles and confirm it matches.
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let rebuilt = camera.direction();
        assert!(approx(rebuilt.x, direction.x));
        assert!(approx(rebuilt.y, direction.y));
        assert!(approx(rebuilt.z, direction.z));
        assert!(approx(phi, (1.0f32 / 3.0f32.sqrt()).acos()));
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
}
