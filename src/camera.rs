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

/// Historical pole epsilon. **No longer used by the camera math** — the snaps and
/// the drag clamp now reach the EXACT poles (`0` / `π`) and rely on
/// [`OrbitCamera::up_vector`] for a true singular-frame up instead of nudging
/// `phi` a hair short. Retained as a public constant for back-compat; a later
/// step (#13) may remove it once nothing downstream references it.
pub const POLE_EPSILON: f32 = 0.0001;

/// Drag clamp for `orbit_phi`. The drag now reaches the EXACT poles (`0.0` /
/// `π`) and stops there: the view matrix no longer degenerates at the pole
/// because [`OrbitCamera::up_vector`] supplies a true singular-frame up. The old
/// `[POLE_EPSILON, π−POLE_EPSILON]` clamp (which stopped a hair short) is gone.
const PHI_MIN: f32 = 0.0;
const PHI_MAX: f32 = std::f32::consts::PI;

/// Half-width of the smoothstep band (in `phi` radians) over which the up vector
/// blends from `Vec3::Y` to the azimuth-derived horizontal up. Inside `[0, BAND]`
/// of the top pole (and `[π−BAND, π]` of the bottom) the blend runs; outside it
/// the up is exactly `Vec3::Y`. Small enough to be invisible, wide enough that
/// the blend is smooth (no 1-frame flip) right through the singular frame.
const UP_BLEND_BAND: f32 = 0.05;

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
    /// special-case theta (undefined there) and snap phi to the EXACT pole
    /// (`0` / `π`): the view matrix no longer degenerates there because
    /// [`OrbitCamera::up_vector`] supplies a true singular-frame up. Theta keeps
    /// the historical TOP/BOTTOM convention (`−π/2`) so the pole-up limit
    /// `(−cos θ, 0, −sin θ)` lands on a stable screen orientation.
    pub fn snap_angles(&self) -> (f32, f32) {
        use std::f32::consts::{FRAC_PI_2, PI};
        if self.is_pole() {
            return match self.faces[0] {
                CubeFace::Top => (-FRAC_PI_2, 0.0),
                _ => (-FRAC_PI_2, PI),
            };
        }
        let direction = self.snap_direction().normalize();
        let phi = direction.y.clamp(-1.0, 1.0).acos();
        let theta = direction.z.atan2(direction.x);
        (theta, phi)
    }
}

/// Screen direction of a ViewCube **rotate arrow** (90° step to the adjacent
/// face). `Up`/`Down`/`Left`/`Right` are *screen-relative* — the direction the
/// arrow points in the cube's gutter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrowDir {
    Up,
    Down,
    Left,
    Right,
}

impl ArrowDir {
    /// The opposite screen direction (Up↔Down, Left↔Right).
    pub fn opposite(self) -> ArrowDir {
        match self {
            ArrowDir::Up => ArrowDir::Down,
            ArrowDir::Down => ArrowDir::Up,
            ArrowDir::Left => ArrowDir::Right,
            ArrowDir::Right => ArrowDir::Left,
        }
    }
}

/// Screen direction of a ViewCube **roll arrow** (90° roll about the view axis).
/// `Cw` = clockwise, `Ccw` = counter-clockwise (as seen on screen).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollDir {
    Cw,
    Ccw,
}

/// A compass heading on the ViewCube base ring (AutoCAD/Inventor style). The
/// mapping to faces / theta is pinned in [`compass_heading_to_theta`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Heading {
    North,
    East,
    South,
    West,
}

/// The neighbour face reached by a 90° ViewCube rotate in screen direction
/// `dir`, starting from the current nearest face.
///
/// **Convention** (pinned here; Step 2/3 rendering + wiring MUST match). A
/// rotate arrow steps the view 90° along one of two great circles:
///
///   * **Up / Down** walk the *vertical* great circle through the front:
///     `FRONT → TOP → BACK → BOTTOM → FRONT` (Up advances forward, Down back).
///   * **Left / Right** walk the *horizontal equator*:
///     `FRONT → RIGHT → BACK → LEFT → FRONT` (Right advances, Left back).
///
/// The faces *off* each circle fall back to its poles: an equatorial face's
/// Up/Down reach TOP/BOTTOM, and TOP/BOTTOM's Left/Right reach the LEFT/RIGHT
/// equatorial faces (a spin about the vertical axis). Full table:
///
///   * FRONT (+Z): Up→Top,  Down→Bottom, Left→Left,  Right→Right
///   * BACK (−Z):  Up→Bottom, Down→Top,  Left→Right, Right→Left
///   * RIGHT (+X): Up→Top,  Down→Bottom, Left→Front, Right→Back
///   * LEFT (−X):  Up→Top,  Down→Bottom, Left→Back,  Right→Front
///   * TOP (+Y):   Up→Back, Down→Front,  Left→Left,  Right→Right
///   * BOTTOM(−Y): Up→Front, Down→Back,  Left→Left,  Right→Right
///
/// Properties (proven in tests): the four neighbours of any face are distinct
/// (never the face itself); four Up steps cycle the vertical circle and four
/// Right steps cycle the equator; Up↔Down are mutual inverses on the four
/// vertical-circle faces, and Left↔Right are mutual inverses on the four
/// equatorial faces. (A *full* memoryless inverse over all 6×4 is geometrically
/// impossible — stepping off and back onto a circle rolls the cube — so the
/// inverse property is asserted only on each direction's own great circle.)
pub fn adjacent_face(current: CubeFace, dir: ArrowDir) -> CubeFace {
    use ArrowDir as A;
    use CubeFace as F;
    match (current, dir) {
        (F::Front, A::Up) => F::Top,
        (F::Front, A::Down) => F::Bottom,
        (F::Front, A::Left) => F::Left,
        (F::Front, A::Right) => F::Right,

        (F::Back, A::Up) => F::Bottom,
        (F::Back, A::Down) => F::Top,
        (F::Back, A::Left) => F::Right,
        (F::Back, A::Right) => F::Left,

        (F::Right, A::Up) => F::Top,
        (F::Right, A::Down) => F::Bottom,
        (F::Right, A::Left) => F::Front,
        (F::Right, A::Right) => F::Back,

        (F::Left, A::Up) => F::Top,
        (F::Left, A::Down) => F::Bottom,
        (F::Left, A::Left) => F::Back,
        (F::Left, A::Right) => F::Front,

        (F::Top, A::Up) => F::Back,
        (F::Top, A::Down) => F::Front,
        (F::Top, A::Left) => F::Left,
        (F::Top, A::Right) => F::Right,

        (F::Bottom, A::Up) => F::Front,
        (F::Bottom, A::Down) => F::Back,
        (F::Bottom, A::Left) => F::Left,
        (F::Bottom, A::Right) => F::Right,
    }
}

/// The azimuth (`theta`) a compass heading snaps the camera to, consistent with
/// the face theta convention in [`ViewCubeElement::snap_angles`]
/// (FRONT/+Z→π/2, RIGHT/+X→0, BACK/−Z→−π/2, LEFT/−X→π). The heading→face map is
/// the natural ground-compass one: **N = Front, E = Right, S = Back, W = Left**.
///
/// The four headings give distinct thetas exactly 90° apart, stepping by −π/2
/// in N→E→S→W order: `N=π/2, E=0, S=−π/2, W=−π` (W ≡ Left's `π` mod 2π).
pub fn compass_heading_to_theta(heading: Heading) -> f32 {
    use std::f32::consts::FRAC_PI_2;
    match heading {
        Heading::North => FRAC_PI_2,       // Front (+Z)
        Heading::East => 0.0,              // Right (+X)
        Heading::South => -FRAC_PI_2,      // Back (−Z)
        Heading::West => -std::f32::consts::PI, // Left (−X), ≡ +π mod 2π
    }
}

/// A rectangle in window pixels (the ViewCube's on-screen region). `x`/`y` are
/// the top-left corner; `size` is the side length (the cube viewport is square).
/// Used by [`classify_cube_point`] so the chrome hit-zones and the Step-2
/// renderer share one layout definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubeRect {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

/// A classified hit zone within (or just around) the ViewCube's screen rect.
/// Step 1 only computes these; Step 2 renders the chrome in the SAME rects and
/// Step 3 wires them to mouse events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeChromeZone {
    /// The cube body proper — resolved to a face/edge/corner by the caller's
    /// raycast picker (the existing `pick_view_cube_element`).
    Element(ViewCubeElement),
    /// A rotate-to-adjacent-face arrow in one of the four gutters.
    RotateArrow(ArrowDir),
    /// A roll arrow at a top corner.
    RollArrow(RollDir),
    /// A compass heading on the base ring.
    Compass(Heading),
    /// The Home button (top-left corner badge).
    HomeButton,
    /// The Fit button (next to Home).
    FitButton,
}

/// **ViewCube chrome layout** — pure screen-space hit-testing over the cube's
/// square `rect`. All zones are expressed as fractions of `rect.size` so Step 2
/// draws them in the identical pixels. Documented fractions (origin = rect
/// top-left, x right, y down):
///
/// ```text
///   ┌─────────────────────────────┐  0.00
///   │ H F      [ ▲ ]         ⟲  ⟳ │  Home/Fit badges (TL), roll arrows (TR)
///   │  ┌───────────────────────┐  │  0.15   rotate-UP gutter  : x∈[.35,.65] y∈[.04,.15]
///   │  │                       │  │         roll arrows       : y∈[.00,.13]
///   │[◀]      cube body      [▶]│         CUBE BODY (raycast): [.15,.85]²
///   │  │                       │  │         rotate-L/R gutters: y∈[.30,.70]
///   │  └───────────────────────┘  │  0.85
///   │ N    [ ▼ ]   E   S   W      │  rotate-DOWN gutter, then compass ring
///   └─────────────────────────────┘  1.00   compass ring: y∈[.88,1.00]
/// ```
///
/// Precedence (first match wins): Home/Fit badges, roll arrows, the four rotate
/// gutters, the compass ring, then the central cube body (delegated to
/// `body_picker`). A point inside `rect` that matches no chrome zone and whose
/// `body_picker` returns `None` yields `None`; a point outside `rect` is `None`.
///
/// `body_picker` is the caller's raycast (`pick_view_cube_element`); it is
/// invoked only for the central body region. In tests a stub picker stands in,
/// keeping this function fully headless.
pub fn classify_cube_point(
    rect: CubeRect,
    cursor_x: f32,
    cursor_y: f32,
    body_picker: impl FnOnce() -> Option<ViewCubeElement>,
) -> Option<CubeChromeZone> {
    // Normalised position within the rect (0..1 across each axis).
    let u = (cursor_x - rect.x) / rect.size;
    let v = (cursor_y - rect.y) / rect.size;
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None; // outside the cube rect entirely.
    }

    // --- Home / Fit badges: two small squares in the top-left corner. ---
    const BADGE_TOP: f32 = 0.00;
    const BADGE_BOTTOM: f32 = 0.12;
    if (BADGE_TOP..BADGE_BOTTOM).contains(&v) {
        if (0.00..0.12).contains(&u) {
            return Some(CubeChromeZone::HomeButton);
        }
        if (0.12..0.24).contains(&u) {
            return Some(CubeChromeZone::FitButton);
        }
    }

    // --- Roll arrows: two small rects in the top-right corner. ---
    const ROLL_TOP: f32 = 0.00;
    const ROLL_BOTTOM: f32 = 0.13;
    if (ROLL_TOP..ROLL_BOTTOM).contains(&v) {
        if (0.74..0.87).contains(&u) {
            return Some(CubeChromeZone::RollArrow(RollDir::Ccw));
        }
        if (0.87..1.00).contains(&u) {
            return Some(CubeChromeZone::RollArrow(RollDir::Cw));
        }
    }

    // --- Rotate arrows: 4 rects in the gutters just outside the cube body. ---
    // The cube body occupies the central [.15, .85]² region; the gutters are the
    // bands between the body and the rect edge, centred on each side.
    const BODY_MIN: f32 = 0.15;
    const BODY_MAX: f32 = 0.85;
    const GUTTER_LO: f32 = 0.35; // along-side span of each rotate arrow
    const GUTTER_HI: f32 = 0.65;
    // UP gutter: above the body, horizontally centred.
    if (0.04..BODY_MIN).contains(&v) && (GUTTER_LO..GUTTER_HI).contains(&u) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Up));
    }
    // DOWN gutter: below the body but above the compass ring.
    if (BODY_MAX..0.88).contains(&v) && (GUTTER_LO..GUTTER_HI).contains(&u) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Down));
    }
    // LEFT gutter: left of the body, vertically centred.
    if (0.02..BODY_MIN).contains(&u) && (GUTTER_LO..GUTTER_HI).contains(&v) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Left));
    }
    // RIGHT gutter: right of the body, vertically centred.
    if (BODY_MAX..0.98).contains(&u) && (GUTTER_LO..GUTTER_HI).contains(&v) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Right));
    }

    // --- Compass ring: 4 rects along a band at the base (N/E/S/W L→R). ---
    const COMPASS_TOP: f32 = 0.88;
    if v >= COMPASS_TOP {
        if (0.05..0.30).contains(&u) {
            return Some(CubeChromeZone::Compass(Heading::North));
        }
        if (0.30..0.50).contains(&u) {
            return Some(CubeChromeZone::Compass(Heading::East));
        }
        if (0.50..0.70).contains(&u) {
            return Some(CubeChromeZone::Compass(Heading::South));
        }
        if (0.70..0.95).contains(&u) {
            return Some(CubeChromeZone::Compass(Heading::West));
        }
        return None;
    }

    // --- Cube body: the central region, resolved by the caller's raycast. ---
    if (BODY_MIN..=BODY_MAX).contains(&u) && (BODY_MIN..=BODY_MAX).contains(&v) {
        return body_picker().map(CubeChromeZone::Element);
    }

    None
}

/// The saved "home" view (#13): the orbit angles + distance the Home button
/// returns to. Defaults to the camera defaults (theta≈0.7, phi≈1.05, dist 10);
/// `set_home_to_current` overwrites it from the live camera, and it persists via
/// `AppConfig` (`home_theta`/`home_phi`/`home_distance`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomeView {
    pub theta: f32,
    pub phi: f32,
    pub distance: f32,
}

impl Default for HomeView {
    fn default() -> Self {
        let camera = OrbitCamera::default();
        Self {
            theta: camera.orbit_theta,
            phi: camera.orbit_phi,
            distance: camera.orbit_distance,
        }
    }
}

impl HomeView {
    /// Capture the live camera's orbit angles + distance as the new home.
    pub fn from_camera(camera: &OrbitCamera) -> Self {
        Self {
            theta: camera.orbit_theta,
            phi: camera.orbit_phi,
            distance: camera.orbit_distance,
        }
    }

    /// Begin an eased snap from `camera`'s current angles toward this home view
    /// (nearest-equivalent theta, so no long spin). The caller advances the
    /// returned tween each frame and separately lerps/sets `orbit_distance`
    /// (the tween animates angles only, matching `SnapTween`).
    pub fn snap_tween(&self, camera: &OrbitCamera) -> SnapTween {
        SnapTween {
            theta_from: camera.orbit_theta,
            phi_from: camera.orbit_phi,
            theta_to: nearest_equivalent_theta(camera.orbit_theta, self.theta),
            phi_to: self.phi,
            elapsed_seconds: 0.0,
            duration_seconds: SnapTween::DEFAULT_DURATION_SECONDS,
        }
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

    /// The up vector for `look_at_rh`, well-defined and CONTINUOUS through the
    /// poles (no `look_at` degeneracy, no roll-flip).
    ///
    /// Away from the poles this is just `Vec3::Y`. Within [`UP_BLEND_BAND`] of a
    /// pole it smoothly blends to an **azimuth-derived horizontal up** — the exact
    /// limit of "`Vec3::Y` projected onto the view plane, normalised" as
    /// `phi → 0/π`. That limit is `(−cos θ, 0, −sin θ)` at the top pole and
    /// `(cos θ, 0, sin θ)` at the bottom, so the screen "up" the user sees is the
    /// direction the camera would tilt toward, and it never jumps as the drag
    /// crosses the singular frame.
    ///
    /// At the exact TOP snap (`θ = −π/2`, `phi = 0`) this yields up `≈ (0, 0, 1)`,
    /// consistent with the historical TOP/BOTTOM snap convention.
    pub fn up_vector(&self) -> Vec3 {
        use std::f32::consts::PI;
        // Distance (in phi) from the nearest pole.
        let phi = self.orbit_phi;
        let distance_from_pole = phi.min(PI - phi);
        if distance_from_pole >= UP_BLEND_BAND {
            return Vec3::Y;
        }
        // Horizontal up: the limit of projected-Y as phi → the near pole.
        // Top pole (phi≈0): (−cosθ, 0, −sinθ); bottom (phi≈π): (cosθ, 0, sinθ).
        let (sin_theta, cos_theta) = self.orbit_theta.sin_cos();
        let near_top = phi < PI - phi;
        let horizontal_up = if near_top {
            Vec3::new(-cos_theta, 0.0, -sin_theta)
        } else {
            Vec3::new(cos_theta, 0.0, sin_theta)
        };
        // smoothstep from horizontal_up (at the pole) to Vec3::Y (at the band edge).
        let t = (distance_from_pole / UP_BLEND_BAND).clamp(0.0, 1.0);
        let weight = t * t * (3.0 - 2.0 * t); // smoothstep
        let blended = horizontal_up.lerp(Vec3::Y, weight);
        // The two endpoints are orthogonal unit vectors, so the lerp is never
        // zero-length; normalise so `look_at_rh` gets a clean unit up.
        blended.normalize()
    }

    /// Orbit by a screen-space drag delta (left-drag): `phi -= dy * 0.01`, with
    /// `phi` clamped to `[0, π]` — the drag reaches the EXACT poles and stops
    /// there (Fusion "Constrained Orbit"). No degeneracy: [`Self::up_vector`]
    /// supplies a true singular-frame up at the pole.
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
        let view = Mat4::look_at_rh(self.eye(), self.target, self.up_vector());
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
        // MUST share the main camera's up (`up_vector`) or the cube and the scene
        // desync at the pole — same singular-frame up, same orientation.
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, self.up_vector());
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
    fn faces_match_old_snap_table() {
        // The unified element snap must reproduce the historical face angles.
        let expected = [
            (CubeFace::Right, (0.0, FRAC_PI_2)),
            (CubeFace::Left, (PI, FRAC_PI_2)),
            (CubeFace::Top, (-FRAC_PI_2, 0.0)),
            (CubeFace::Bottom, (-FRAC_PI_2, PI)),
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
    /// the singular frame — no 1-frame flip. Two proofs:
    ///  1. A close pair straddling the band edge (0.04 inside, 0.06 outside) stays
    ///     close — contrast the OLD epsilon-clamp, which would flip up ~180° on
    ///     crossing the pole.
    ///  2. A dense sweep over the whole [0, band+ε] range: no adjacent sample pair
    ///     ever jumps more than a small Lipschitz bound (proves smoothness, not
    ///     just endpoint agreement).
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
        // Lipschitz bound (no jump). The up traces a quarter-arc (π/2) across the
        // band via smoothstep; max |d(up)/dphi| ≈ (π/2)·1.5/BAND, so a 0.002 step
        // moves the chord by at most ~0.1. A real flip would be ~2.0 (180°), so a
        // 0.12 ceiling proves continuity while staying far below any flip.
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

    /// At the exact TOP snap (theta=-π/2, phi=0) the up limit is (0,0,1) and at
    /// the BOTTOM snap (theta=-π/2, phi=π) it is (0,0,-1) — the documented
    /// convention.
    #[test]
    fn up_vector_at_exact_pole_snaps_matches_convention() {
        let top = OrbitCamera {
            orbit_theta: -FRAC_PI_2,
            orbit_phi: 0.0,
            ..OrbitCamera::default()
        }
        .up_vector();
        // (-cos(-π/2), 0, -sin(-π/2)) = (0, 0, 1).
        assert!(approx(top.x, 0.0) && approx(top.y, 0.0) && approx(top.z, 1.0), "{top:?}");

        let bottom = OrbitCamera {
            orbit_theta: -FRAC_PI_2,
            orbit_phi: PI,
            ..OrbitCamera::default()
        }
        .up_vector();
        // (cos(-π/2), 0, sin(-π/2)) = (0, 0, -1).
        assert!(
            approx(bottom.x, 0.0) && approx(bottom.y, 0.0) && approx(bottom.z, -1.0),
            "{bottom:?}"
        );
    }

    /// Away from the poles the up vector is exactly Vec3::Y (no behavioural
    /// change to non-pole views — goldens stay byte-identical).
    #[test]
    fn up_vector_away_from_poles_is_exactly_y() {
        for &phi in &[0.1f32, 0.5, 1.05, FRAC_PI_2, 2.5, PI - 0.1] {
            let up = OrbitCamera {
                orbit_phi: phi,
                ..OrbitCamera::default()
            }
            .up_vector();
            assert_eq!(up, Vec3::Y, "phi={phi} should give exact Vec3::Y");
        }
    }

    /// Both view matrices are all-finite (no NaN/inf) at the exact poles — the
    /// old degeneracy is gone.
    #[test]
    fn view_matrices_finite_at_exact_poles() {
        for &phi in &[0.0f32, PI] {
            let camera = OrbitCamera {
                orbit_theta: -FRAC_PI_2,
                orbit_phi: phi,
                ..OrbitCamera::default()
            };
            let vp = camera.view_projection(1.0);
            assert!(
                vp.to_cols_array().iter().all(|v| v.is_finite()),
                "view_projection not finite at phi={phi}: {vp:?}"
            );
            let cube = camera.view_cube_view_projection();
            assert!(
                cube.to_cols_array().iter().all(|v| v.is_finite()),
                "view_cube_view_projection not finite at phi={phi}: {cube:?}"
            );
        }
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

    // ---- #13 Step 1: ViewCube chrome hit-math + Home/Fit logic ----

    /// A square cube rect for the layout tests: top-left (100, 200), side 128px.
    fn test_cube_rect() -> CubeRect {
        CubeRect { x: 100.0, y: 200.0, size: 128.0 }
    }

    /// Window pixel at fractional `(u, v)` of the rect (matches the layout doc).
    fn at(rect: CubeRect, u: f32, v: f32) -> (f32, f32) {
        (rect.x + u * rect.size, rect.y + v * rect.size)
    }

    fn no_body() -> Option<ViewCubeElement> {
        None
    }

    #[test]
    fn classify_rotate_arrows_in_each_gutter() {
        let rect = test_cube_rect();
        // UP gutter (top centre), DOWN (bottom centre), LEFT, RIGHT.
        let cases = [
            (0.50, 0.10, ArrowDir::Up),
            (0.50, 0.86, ArrowDir::Down),
            (0.08, 0.50, ArrowDir::Left),
            (0.92, 0.50, ArrowDir::Right),
        ];
        for (u, v, dir) in cases {
            let (x, y) = at(rect, u, v);
            assert_eq!(
                classify_cube_point(rect, x, y, no_body),
                Some(CubeChromeZone::RotateArrow(dir)),
                "({u},{v}) should be RotateArrow({dir:?})"
            );
        }
    }

    #[test]
    fn classify_roll_arrows_at_top_corners() {
        let rect = test_cube_rect();
        let (x, y) = at(rect, 0.80, 0.06);
        assert_eq!(
            classify_cube_point(rect, x, y, no_body),
            Some(CubeChromeZone::RollArrow(RollDir::Ccw))
        );
        let (x, y) = at(rect, 0.93, 0.06);
        assert_eq!(
            classify_cube_point(rect, x, y, no_body),
            Some(CubeChromeZone::RollArrow(RollDir::Cw))
        );
    }

    #[test]
    fn classify_compass_ring_headings_left_to_right() {
        let rect = test_cube_rect();
        let cases = [
            (0.17, Heading::North),
            (0.40, Heading::East),
            (0.60, Heading::South),
            (0.82, Heading::West),
        ];
        for (u, heading) in cases {
            let (x, y) = at(rect, u, 0.94);
            assert_eq!(
                classify_cube_point(rect, x, y, no_body),
                Some(CubeChromeZone::Compass(heading)),
                "u={u} should be Compass({heading:?})"
            );
        }
    }

    #[test]
    fn classify_home_and_fit_badges() {
        let rect = test_cube_rect();
        let (x, y) = at(rect, 0.05, 0.06);
        assert_eq!(
            classify_cube_point(rect, x, y, no_body),
            Some(CubeChromeZone::HomeButton)
        );
        let (x, y) = at(rect, 0.18, 0.06);
        assert_eq!(
            classify_cube_point(rect, x, y, no_body),
            Some(CubeChromeZone::FitButton)
        );
    }

    #[test]
    fn classify_central_point_delegates_to_body_picker() {
        let rect = test_cube_rect();
        let (x, y) = at(rect, 0.5, 0.5);
        // A stub picker returns a known element; the body case wraps it.
        let element = ViewCubeElement::from_face(CubeFace::Front);
        let zone = classify_cube_point(rect, x, y, || Some(element));
        assert_eq!(zone, Some(CubeChromeZone::Element(element)));
        // If the raycast misses (e.g. a corner of the body square off the cube),
        // the body case yields None rather than a bogus chrome zone.
        assert_eq!(classify_cube_point(rect, x, y, no_body), None);
    }

    #[test]
    fn classify_outside_rect_is_none() {
        let rect = test_cube_rect();
        // Left of, above, right of, below the rect.
        for (dx, dy) in [(-10.0, 0.0), (0.0, -10.0), (200.0, 0.0), (0.0, 200.0)] {
            assert_eq!(
                classify_cube_point(rect, rect.x + dx, rect.y + dy, no_body),
                None
            );
        }
    }

    #[test]
    fn adjacent_face_spot_checks() {
        use ArrowDir::*;
        assert_eq!(adjacent_face(CubeFace::Front, Right), CubeFace::Right);
        assert_eq!(adjacent_face(CubeFace::Front, Up), CubeFace::Top);
        assert_eq!(adjacent_face(CubeFace::Front, Left), CubeFace::Left);
        assert_eq!(adjacent_face(CubeFace::Front, Down), CubeFace::Bottom);
        assert_eq!(adjacent_face(CubeFace::Right, Left), CubeFace::Front);
        assert_eq!(adjacent_face(CubeFace::Top, Down), CubeFace::Front);
    }

    #[test]
    fn adjacent_face_four_up_steps_cycle() {
        // FRONT --Up--> Top --Up--> Back --Up--> Bottom --Up--> FRONT.
        let mut face = CubeFace::Front;
        let visited: Vec<CubeFace> = (0..4)
            .map(|_| {
                face = adjacent_face(face, ArrowDir::Up);
                face
            })
            .collect();
        assert_eq!(
            visited,
            vec![CubeFace::Top, CubeFace::Back, CubeFace::Bottom, CubeFace::Front]
        );
    }

    #[test]
    fn adjacent_face_neighbours_are_distinct_and_exclude_self() {
        let faces = [
            CubeFace::Right,
            CubeFace::Left,
            CubeFace::Top,
            CubeFace::Bottom,
            CubeFace::Front,
            CubeFace::Back,
        ];
        let dirs = [ArrowDir::Up, ArrowDir::Down, ArrowDir::Left, ArrowDir::Right];
        for face in faces {
            for dir in dirs {
                assert_ne!(adjacent_face(face, dir), face, "{face:?} {dir:?} is a no-op");
            }
            // The four neighbours of a face are pairwise distinct.
            let mut sorted: Vec<String> =
                dirs.iter().map(|&d| format!("{:?}", adjacent_face(face, d))).collect();
            sorted.sort();
            sorted.dedup();
            assert_eq!(sorted.len(), 4, "{face:?} must have 4 distinct neighbours");
        }
    }

    #[test]
    fn adjacent_face_inverses_hold_on_each_great_circle() {
        // Up↔Down are mutual inverses on the VERTICAL circle (Front/Top/Back/Bottom).
        for &face in &[CubeFace::Front, CubeFace::Top, CubeFace::Back, CubeFace::Bottom] {
            let up = adjacent_face(face, ArrowDir::Up);
            assert_eq!(adjacent_face(up, ArrowDir::Down), face, "Up/Down inverse at {face:?}");
            let down = adjacent_face(face, ArrowDir::Down);
            assert_eq!(adjacent_face(down, ArrowDir::Up), face, "Down/Up inverse at {face:?}");
        }
        // Left↔Right are mutual inverses on the EQUATOR (Front/Right/Back/Left).
        for &face in &[CubeFace::Front, CubeFace::Right, CubeFace::Back, CubeFace::Left] {
            let right = adjacent_face(face, ArrowDir::Right);
            assert_eq!(adjacent_face(right, ArrowDir::Left), face, "R/L inverse at {face:?}");
            let left = adjacent_face(face, ArrowDir::Left);
            assert_eq!(adjacent_face(left, ArrowDir::Right), face, "L/R inverse at {face:?}");
        }
    }

    #[test]
    fn adjacent_face_four_right_steps_cycle_the_equator() {
        // FRONT --Right--> Right --> Back --> Left --> FRONT.
        let mut face = CubeFace::Front;
        let visited: Vec<CubeFace> = (0..4)
            .map(|_| {
                face = adjacent_face(face, ArrowDir::Right);
                face
            })
            .collect();
        assert_eq!(
            visited,
            vec![CubeFace::Right, CubeFace::Back, CubeFace::Left, CubeFace::Front]
        );
    }

    #[test]
    fn compass_headings_are_distinct_and_90_apart() {
        let n = compass_heading_to_theta(Heading::North);
        let e = compass_heading_to_theta(Heading::East);
        let s = compass_heading_to_theta(Heading::South);
        let w = compass_heading_to_theta(Heading::West);
        // Each consecutive step is −π/2 (N→E→S→W clockwise).
        assert!(approx(n - e, FRAC_PI_2));
        assert!(approx(e - s, FRAC_PI_2));
        assert!(approx(s - w, FRAC_PI_2));
        // Distinct mod 2π.
        let thetas = [n, e, s, w];
        for i in 0..4 {
            for j in (i + 1)..4 {
                let diff = (thetas[i] - thetas[j]).rem_euclid(2.0 * PI);
                assert!(!approx(diff, 0.0), "headings {i},{j} collide");
            }
        }
        // N maps to Front's theta, E to Right's, etc. (the pinned face map).
        assert!(approx(n, CubeFace::Front.snap_angles().0));
        assert!(approx(e, CubeFace::Right.snap_angles().0));
        assert!(approx(s, CubeFace::Back.snap_angles().0));
    }

    #[test]
    fn home_view_default_matches_camera_defaults() {
        let home = HomeView::default();
        let camera = OrbitCamera::default();
        assert!(approx(home.theta, camera.orbit_theta));
        assert!(approx(home.phi, camera.orbit_phi));
        assert!(approx(home.distance, camera.orbit_distance));
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
