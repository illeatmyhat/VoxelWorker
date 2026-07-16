//! The **Autodesk ViewCube** orientation model and its on-screen chrome.
//!
//! A ViewCube addresses any of the 26 canonical orientations of a cube — 6 faces,
//! 12 edges, 8 corners — as a single [`ViewCubeElement`], and maps each to the
//! orbit `(theta, phi)` the camera should snap to. On top of the cube body sit the
//! screen affordances Autodesk's widget is known for: rotate arrows that step the
//! view 90° to an adjacent face along a great circle ([`adjacent_face`]), roll
//! arrows, and Home/Fit badges. [`classify_cube_point`] is the pure screen-space
//! hit-test over that layout, and [`chrome_zone_left_click_action`] is the pure
//! zone→action dispatch (it never mutates the camera; the windowed caller executes
//! the returned [`ChromeClickAction`]).
//!
//! Everything here is pure geometry and screen arithmetic — no winit, no egui, no
//! rendering. The cube's rendered chrome and its mouse plumbing stay in the app.
//!
//! Cite: Khan, Mordatch, Fitzmaurice, Matejka & Kurtenbach, "ViewCube: a 3D
//! orientation indicator and controller" (ACM I3D 2008), for the 26-orientation
//! hot-zone model; the spherical `(theta, phi)` conversion is the standard Z-up
//! spherical parameterisation (Akenine-Möller, Haines & Hoffman, *Real-Time
//! Rendering*).

use glam::Vec3;

use crate::orbit::OrbitCamera;
use crate::tween::SnapTween;

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
    /// +Z — TOP (Z-up).
    Top,
    /// -Z — BOTTOM (Z-up).
    Bottom,
    /// -Y — FRONT (Z-up: the front view looks along +Y at the −Y face).
    Front,
    /// +Y — BACK (Z-up).
    Back,
}

/// The view-cube faces in `materialIndex` order, with their human labels.
///
/// `materialIndex` order is the GEOMETRIC cube-axis order `(+X, -X, +Y, -Y, +Z,
/// -Z)` (the raycast computes it as `axis*2 + sign`). Under the Z-up convention the
/// geometric +Z face is TOP, -Z is BOTTOM, -Y is FRONT (the front view looks along
/// +Y at the −Y face), and +Y is BACK — so the semantic [`CubeFace`] at each
/// geometric index is RIGHT/LEFT/BACK/FRONT/TOP/BOTTOM.
pub const CUBE_FACES: [(CubeFace, &str); 6] = [
    (CubeFace::Right, "RIGHT"),   // +X
    (CubeFace::Left, "LEFT"),     // -X
    (CubeFace::Back, "BACK"),     // +Y
    (CubeFace::Front, "FRONT"),   // -Y
    (CubeFace::Top, "TOP"),       // +Z
    (CubeFace::Bottom, "BOTTOM"), // -Z
];

impl CubeFace {
    /// Map a 0..5 material index (raycast hit) to a face, matching the GEOMETRIC
    /// `materialIndex` order (+X, -X, +Y, -Y, +Z, -Z) under the Z-up convention.
    pub fn from_material_index(index: usize) -> Option<Self> {
        CUBE_FACES.get(index).map(|(face, _)| *face)
    }

    /// The snap target `(theta, phi)` for this face. Polar values use the exact
    /// poles at TOP/BOTTOM; the view matrix never degenerates there because
    /// [`OrbitCamera::up_vector`] supplies a true singular-frame up.
    pub fn snap_angles(self) -> (f32, f32) {
        ViewCubeElement::from_face(self).snap_angles()
    }

    /// The outward unit normal of this face, Z-up: RIGHT/LEFT = ±X, TOP/BOTTOM =
    /// ±Z, FRONT/BACK = ∓Y (front = −Y, the front view looks along +Y).
    pub fn normal(self) -> Vec3 {
        match self {
            CubeFace::Right => Vec3::X,
            CubeFace::Left => Vec3::NEG_X,
            CubeFace::Top => Vec3::Z,
            CubeFace::Bottom => Vec3::NEG_Z,
            CubeFace::Front => Vec3::NEG_Y,
            CubeFace::Back => Vec3::Y,
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

    /// The per-axis sign selector `[sx, sy, sz]` (each in `{-1, 0, +1}`) describing
    /// this element geometrically for the renderer's hover highlight. For each cube
    /// axis, the value is the sign of the element's face normal along it, or `0` if no
    /// face of the element touches that axis. A face has one non-zero entry, an edge
    /// two, a corner three (e.g. the FRONT·TOP·RIGHT corner → `[+1, -1, +1]`: Right +X,
    /// Front −Y, Top +Z). The renderer highlights a face fragment at cube position `p`
    /// iff, on every axis `a`, `p[a]` lies on the selector's side of the centre patch
    /// (`sel[a]·p[a] ≥ threshold`, or `|p[a]| ≤ threshold` when `sel[a] = 0`) — which
    /// lights exactly the 1/2/3 across-the-fold facets of the hovered element.
    pub fn axis_selectors(&self) -> [f32; 3] {
        let mut selectors = [0.0f32; 3];
        for face in self.faces() {
            let normal = face.normal();
            if normal.x != 0.0 {
                selectors[0] = normal.x.signum();
            }
            if normal.y != 0.0 {
                selectors[1] = normal.y.signum();
            }
            if normal.z != 0.0 {
                selectors[2] = normal.z.signum();
            }
        }
        selectors
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
    /// Unified spherical conversion (Z-up) `phi = acos(dir.z)`, `theta =
    /// atan2(dir.y, dir.x)` — works for faces, edges AND corners. The pure poles
    /// (dir = ±Z) special-case theta (undefined there) and snap phi to the EXACT
    /// pole (`0` / `π`): the view matrix no longer degenerates there because
    /// [`OrbitCamera::up_vector`] supplies a true singular-frame up. Theta keeps
    /// the historical TOP/BOTTOM convention (`−π/2`) so the pole-up limit
    /// `(−cos θ, −sin θ, 0)` lands on a stable screen orientation.
    pub fn snap_angles(&self) -> (f32, f32) {
        use std::f32::consts::{FRAC_PI_2, PI};
        if self.is_pole() {
            return match self.faces[0] {
                CubeFace::Top => (-FRAC_PI_2, 0.0),
                _ => (-FRAC_PI_2, PI),
            };
        }
        let direction = self.snap_direction().normalize();
        let phi = direction.z.clamp(-1.0, 1.0).acos();
        let theta = direction.y.atan2(direction.x);
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

/// The neighbour face reached by a 90° ViewCube rotate in screen direction
/// `dir`, starting from the current nearest face.
///
/// **Convention** (pinned here; the renderer + wiring MUST match). A rotate arrow
/// steps the view 90° along one of two great circles:
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
///   * FRONT (−Y): Up→Top,  Down→Bottom, Left→Left,  Right→Right
///   * BACK (+Y):  Up→Bottom, Down→Top,  Left→Right, Right→Left
///   * RIGHT (+X): Up→Top,  Down→Bottom, Left→Front, Right→Back
///   * LEFT (−X):  Up→Top,  Down→Bottom, Left→Back,  Right→Front
///   * TOP (+Z):   Up→Back, Down→Front,  Left→Left,  Right→Right
///   * BOTTOM(−Z): Up→Front, Down→Back,  Left→Left,  Right→Right
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

/// A rectangle in window pixels (the ViewCube's on-screen region). `x`/`y` are
/// the top-left corner; `size` is the side length (the cube viewport is square).
/// Used by [`classify_cube_point`] so the chrome hit-zones and the renderer share
/// one layout definition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubeRect {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

/// A classified hit zone within (or just around) the ViewCube's screen rect. The
/// renderer draws the chrome in the SAME rects and the app wires them to mouse
/// events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeChromeZone {
    /// The cube body proper — resolved to a face/edge/corner by the caller's
    /// raycast picker.
    Element(ViewCubeElement),
    /// A rotate-to-adjacent-face arrow in one of the four gutters.
    RotateArrow(ArrowDir),
    /// A roll arrow at a top corner.
    RollArrow(RollDir),
    /// The Home button (top-left corner badge).
    HomeButton,
    /// The Fit button (next to Home).
    FitButton,
}

/// The readout order rank of a face for the [`view_cube_zone_readout`] label:
/// vertical faces first (TOP/BOTTOM), then depth (FRONT/BACK), then horizontal
/// (RIGHT/LEFT), so a corner reads `TOP·FRONT·RIGHT` (Signal spec order).
fn face_readout_rank(face: CubeFace) -> u8 {
    match face {
        CubeFace::Top | CubeFace::Bottom => 0,
        CubeFace::Front | CubeFace::Back => 1,
        CubeFace::Right | CubeFace::Left => 2,
    }
}

/// The human label of a face (`RIGHT`/`LEFT`/`BACK`/`FRONT`/`TOP`/`BOTTOM`) — the
/// [`CUBE_FACES`] vocabulary.
fn face_label(face: CubeFace) -> &'static str {
    CUBE_FACES
        .iter()
        .find(|(f, _)| *f == face)
        .map(|(_, label)| *label)
        .unwrap_or("")
}

/// The dot-joined zone name for a hovered chrome `zone`, e.g. `TOP·FRONT` (edge) or
/// `TOP·FRONT·RIGHT` (corner), or a single face name. Returns `None` for the
/// non-element chrome zones (arrows / badges), which have no cube-zone readout. Used
/// by the Signal view cube's faint readout line under the cube (`ADR 0018` Decision 8
/// / `docs/design/viewport-chrome-signal.md`). Faces are ordered vertical → depth →
/// horizontal so the label reads TOP·FRONT·RIGHT regardless of pick order.
pub fn view_cube_zone_readout(zone: CubeChromeZone) -> Option<String> {
    let CubeChromeZone::Element(element) = zone else {
        return None;
    };
    let mut faces: Vec<CubeFace> = element.faces().to_vec();
    faces.sort_by_key(|face| face_readout_rank(*face));
    Some(
        faces
            .iter()
            .map(|face| face_label(*face))
            .collect::<Vec<_>>()
            .join("·"),
    )
}

/// **ViewCube chrome layout** — pure screen-space hit-testing over the cube's
/// square `rect`. All zones are expressed as fractions of `rect.size` so the
/// renderer draws them in the identical pixels. Documented fractions (origin = rect
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
///   │        [ ▼ ]                │  rotate-DOWN gutter
///   └─────────────────────────────┘  1.00
/// ```
///
/// Precedence (first match wins): Home/Fit badges, roll arrows, the four rotate
/// gutters, then the central cube body (delegated to `body_picker`). A point inside
/// `rect` that matches no chrome zone and whose `body_picker` returns `None` yields
/// `None`; a point outside `rect` is `None`.
///
/// `body_picker` is the caller's raycast; it is invoked only for the central body
/// region. In tests a stub picker stands in, keeping this function fully headless.
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
    // bands between the body and the rect edge, centred on each side. The rotate
    // arrows are pushed OUT to the rect edge (a wider gutter band) so they no longer
    // crowd the cube — they sit clearly in the margin, Fusion-style.
    const BODY_MIN: f32 = 0.15;
    const BODY_MAX: f32 = 0.85;
    const GUTTER_LO: f32 = 0.38; // along-side span of each rotate arrow
    const GUTTER_HI: f32 = 0.62;
    // UP gutter: hugging the TOP rect edge, horizontally centred.
    if (0.00..0.13).contains(&v) && (GUTTER_LO..GUTTER_HI).contains(&u) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Up));
    }
    // DOWN gutter: hugging the BOTTOM rect edge.
    if (0.87..1.00).contains(&v) && (GUTTER_LO..GUTTER_HI).contains(&u) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Down));
    }
    // LEFT gutter: hugging the LEFT rect edge, vertically centred.
    if (0.00..0.13).contains(&u) && (GUTTER_LO..GUTTER_HI).contains(&v) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Left));
    }
    // RIGHT gutter: hugging the RIGHT rect edge, vertically centred.
    if (0.87..1.00).contains(&u) && (GUTTER_LO..GUTTER_HI).contains(&v) {
        return Some(CubeChromeZone::RotateArrow(ArrowDir::Right));
    }

    // --- Cube body: the central region, resolved by the caller's raycast. ---
    if (BODY_MIN..=BODY_MAX).contains(&u) && (BODY_MIN..=BODY_MAX).contains(&v) {
        return body_picker().map(CubeChromeZone::Element);
    }

    None
}

/// The camera-side outcome of a **left-click on a ViewCube chrome zone**. This is
/// the PURE half of the click dispatch: given a classified [`CubeChromeZone`] and
/// the current camera, [`chrome_zone_left_click_action`] resolves *what should
/// happen* without touching any winit/state plumbing, so the mapping is
/// unit-testable headlessly. The windowed caller then EXECUTES the action (starts
/// the returned tween, or runs the home / fit logic).
///
///   * [`RotateArrow`](CubeChromeZone::RotateArrow) → a face snap toward
///     `adjacent_face(nearest_face, dir)`.
///   * [`Element`](CubeChromeZone::Element) → the existing element snap.
///   * [`HomeButton`](CubeChromeZone::HomeButton) → `Home` (caller runs the home
///     tween, which also sets the home distance).
///   * [`FitButton`](CubeChromeZone::FitButton) → `Fit` (caller frames the model).
///   * [`RollArrow`](CubeChromeZone::RollArrow) → a roll tween: twist the view ∓90°
///     about the view axis (the real roll DOF). `Cw`/`Ccw` set the sign; the orbit
///     angles are held.
#[derive(Debug, Clone, Copy)]
pub enum ChromeClickAction {
    /// Start this eased snap tween (face / element / roll).
    Snap(SnapTween),
    /// Run the Home action (eased snap to the saved home view + home distance).
    Home,
    /// Run the Fit-to-view action (re-frame the model).
    Fit,
}

/// Resolve a left-click on a ViewCube chrome `zone` into a [`ChromeClickAction`]
/// against the current `camera` (the PURE dispatch). See [`ChromeClickAction`] for
/// the per-zone mapping. This never mutates the camera; the caller executes the
/// returned action.
pub fn chrome_zone_left_click_action(
    zone: CubeChromeZone,
    camera: &OrbitCamera,
) -> ChromeClickAction {
    match zone {
        CubeChromeZone::Element(element) => {
            ChromeClickAction::Snap(SnapTween::to_element(camera, element))
        }
        CubeChromeZone::RotateArrow(dir) => {
            let target = adjacent_face(camera.nearest_face(), dir);
            ChromeClickAction::Snap(SnapTween::to_face(camera, target))
        }
        CubeChromeZone::HomeButton => ChromeClickAction::Home,
        CubeChromeZone::FitButton => ChromeClickAction::Fit,
        // The real roll DOF — twist the view ∓90° about the view axis.
        CubeChromeZone::RollArrow(direction) => {
            ChromeClickAction::Snap(SnapTween::roll(camera, direction))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tween::nearest_equivalent_theta;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn axis_selectors_encode_face_edge_corner() {
        // A face has one non-zero selector, an edge two, a corner three; each entry is
        // the sign of the element's face normal along that axis (Z-up: Right +X, Front
        // −Y, Top +Z).
        assert_eq!(
            ViewCubeElement::from_face(CubeFace::Right).axis_selectors(),
            [1.0, 0.0, 0.0]
        );
        assert_eq!(
            ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top).axis_selectors(),
            [0.0, -1.0, 1.0]
        );
        assert_eq!(
            ViewCubeElement::from_corner(CubeFace::Front, CubeFace::Top, CubeFace::Right)
                .axis_selectors(),
            [1.0, -1.0, 1.0]
        );
    }

    #[test]
    fn zone_readout_orders_vertical_depth_horizontal() {
        // Faces read vertical → depth → horizontal regardless of pick order.
        let top_front = CubeChromeZone::Element(ViewCubeElement::from_edge(
            CubeFace::Front,
            CubeFace::Top,
        ));
        assert_eq!(view_cube_zone_readout(top_front).as_deref(), Some("TOP·FRONT"));
        let corner = CubeChromeZone::Element(ViewCubeElement::from_corner(
            CubeFace::Right,
            CubeFace::Front,
            CubeFace::Top,
        ));
        assert_eq!(view_cube_zone_readout(corner).as_deref(), Some("TOP·FRONT·RIGHT"));
        let face = CubeChromeZone::Element(ViewCubeElement::from_face(CubeFace::Right));
        assert_eq!(view_cube_zone_readout(face).as_deref(), Some("RIGHT"));
        // Non-element chrome zones have no zone readout.
        assert_eq!(view_cube_zone_readout(CubeChromeZone::HomeButton), None);
        assert_eq!(
            view_cube_zone_readout(CubeChromeZone::RotateArrow(ArrowDir::Left)),
            None
        );
    }

    #[test]
    fn material_index_maps_to_faces_in_order() {
        // Z-up geometric (+X,-X,+Y,-Y,+Z,-Z) → Right,Left,Back,Front,Top,Bottom.
        assert_eq!(CubeFace::from_material_index(0), Some(CubeFace::Right));
        assert_eq!(CubeFace::from_material_index(2), Some(CubeFace::Back));
        assert_eq!(CubeFace::from_material_index(4), Some(CubeFace::Top));
        assert_eq!(CubeFace::from_material_index(5), Some(CubeFace::Bottom));
        assert_eq!(CubeFace::from_material_index(6), None);
    }

    #[test]
    fn front_face_eye_on_negative_y() {
        // Z-up: FRONT = −Y. Snapping puts the eye on −Y looking back along +Y at
        // the origin. Front snap: theta = atan2(−1, 0) = −π/2, phi = acos(0) = π/2.
        let (theta, phi) = CubeFace::Front.snap_angles();
        assert!(approx(theta, -FRAC_PI_2));
        assert!(approx(phi, FRAC_PI_2));
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let direction = camera.direction();
        // direction = (sinφcosθ, sinφsinθ, cosφ) → ~(0, −1, 0).
        assert!(approx(direction.x, 0.0));
        assert!(approx(direction.y, -1.0));
        assert!(approx(direction.z, 0.0));
    }

    #[test]
    fn top_face_eye_on_positive_z() {
        // Z-up: TOP = +Z. The eye sits straight above, looking down at the XY ground.
        let (theta, phi) = CubeFace::Top.snap_angles();
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let direction = camera.direction();
        assert!(approx(direction.z, 1.0));
        assert!(direction.x.abs() < 1e-3 && direction.y.abs() < 1e-3);
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
    fn faces_match_old_snap_table() {
        // The unified element snap must reproduce the historical face angles.
        // Z-up snap angles: Right/Left/Top/Bottom unchanged; Front (−Y) → theta
        // −π/2 and Back (+Y) → theta +π/2 (swapped from the old Y-up +Z/−Z front/back).
        let expected = [
            (CubeFace::Right, (0.0, FRAC_PI_2)),
            (CubeFace::Left, (PI, FRAC_PI_2)),
            (CubeFace::Top, (-FRAC_PI_2, 0.0)),
            (CubeFace::Bottom, (-FRAC_PI_2, PI)),
            (CubeFace::Front, (-FRAC_PI_2, FRAC_PI_2)),
            (CubeFace::Back, (FRAC_PI_2, FRAC_PI_2)),
        ];
        for (face, (theta, phi)) in expected {
            let (got_theta, got_phi) = face.snap_angles();
            assert!(approx(got_theta, theta), "{face:?} theta {got_theta} != {theta}");
            assert!(approx(got_phi, phi), "{face:?} phi {got_phi} != {phi}");
        }
    }

    #[test]
    fn edge_snap_direction_front_top() {
        // Z-up: FRONT (−Y) + TOP (+Z) → dir (0, −.707, .707): phi = π/4, theta = −π/2.
        let element = ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top);
        let (theta, phi) = element.snap_angles();
        assert!(approx(phi, std::f32::consts::FRAC_PI_4), "phi = {phi}");
        assert!(approx(theta, -FRAC_PI_2), "theta = {theta}");
        // Order-independence: the same edge the other way round.
        let (theta2, phi2) = ViewCubeElement::from_edge(CubeFace::Top, CubeFace::Front).snap_angles();
        assert!(approx(theta, theta2) && approx(phi, phi2));
    }

    #[test]
    fn corner_snap_direction_front_top_right_is_isometric() {
        // Z-up: FRONT (−Y) + TOP (+Z) + RIGHT (+X) → dir (1,−1,1)/√3: an isometric
        // view (X+, Y−, Z+). phi = acos(1/√3) ≈ 0.9553.
        let element =
            ViewCubeElement::from_corner(CubeFace::Front, CubeFace::Top, CubeFace::Right);
        let direction = element.snap_direction().normalize();
        assert!(direction.x > 0.0 && direction.y < 0.0 && direction.z > 0.0);
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

    // ---- ViewCube chrome hit-math + Home/Fit logic ----

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
            (0.50, 0.06, ArrowDir::Up),
            (0.50, 0.94, ArrowDir::Down),
            (0.06, 0.50, ArrowDir::Left),
            (0.94, 0.50, ArrowDir::Right),
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

    // ---- pure chrome-click dispatch (zone → action) ----

    /// A RotateArrow(Right) click from a FRONT-facing camera tweens toward
    /// `adjacent_face(Front, Right) = Right`'s angles (shortest theta).
    #[test]
    fn rotate_arrow_right_from_front_targets_right_face() {
        let (theta, phi) = CubeFace::Front.snap_angles();
        let camera = OrbitCamera {
            orbit_theta: theta,
            orbit_phi: phi,
            ..OrbitCamera::default()
        };
        let action = chrome_zone_left_click_action(
            CubeChromeZone::RotateArrow(ArrowDir::Right),
            &camera,
        );
        let ChromeClickAction::Snap(tween) = action else {
            panic!("expected Snap, got {action:?}");
        };
        let (right_theta, right_phi) = CubeFace::Right.snap_angles();
        assert!(approx(tween.theta_to, nearest_equivalent_theta(theta, right_theta)));
        assert!(approx(tween.phi_to, right_phi));
    }

    /// An Element click reproduces the existing element snap.
    #[test]
    fn element_click_matches_element_snap() {
        let camera = OrbitCamera::default();
        let element = ViewCubeElement::from_edge(CubeFace::Front, CubeFace::Top);
        let action =
            chrome_zone_left_click_action(CubeChromeZone::Element(element), &camera);
        let ChromeClickAction::Snap(tween) = action else {
            panic!("expected Snap, got {action:?}");
        };
        let reference = SnapTween::to_element(&camera, element);
        assert!(approx(tween.theta_to, reference.theta_to));
        assert!(approx(tween.phi_to, reference.phi_to));
    }

    /// Home / Fit map to their dedicated (non-tween) actions; a roll arrow maps to a
    /// real roll tween.
    #[test]
    fn home_fit_roll_zones_map_to_their_actions() {
        let camera = OrbitCamera::default();
        assert!(matches!(
            chrome_zone_left_click_action(CubeChromeZone::HomeButton, &camera),
            ChromeClickAction::Home
        ));
        assert!(matches!(
            chrome_zone_left_click_action(CubeChromeZone::FitButton, &camera),
            ChromeClickAction::Fit
        ));
        // A roll arrow click is now a Snap tween that targets ∓π/2 of roll.
        let action = chrome_zone_left_click_action(
            CubeChromeZone::RollArrow(RollDir::Cw),
            &camera,
        );
        let ChromeClickAction::Snap(tween) = action else {
            panic!("expected Snap (roll tween), got {action:?}");
        };
        assert!(approx(tween.roll_to, -FRAC_PI_2), "Cw should target -π/2, got {}", tween.roll_to);
    }
}
