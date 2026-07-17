//! The shell's ViewCube + framing camera actions: Home snap/set, Fit, the cube viewport
//! hit-rect, chrome-zone dispatch, and the ray-cast element picker that maps a cube click to a
//! face/edge/corner. Split out of `windowed/mod.rs` (ADR 0016).

use super::*;

/// The [`CubeFace`] whose outward normal lies along the GEOMETRIC cube `axis`
/// (0=X,1=Y,2=Z) with the given sign. Z-up: +X→Right, −X→Left, +Y→Back, −Y→Front
/// (front = −Y), +Z→Top, −Z→Bottom.
fn face_for_axis_sign(axis: usize, positive: bool) -> CubeFace {
    match (axis, positive) {
        (0, true) => CubeFace::Right,
        (0, false) => CubeFace::Left,
        (1, true) => CubeFace::Back,
        (1, false) => CubeFace::Front,
        (2, true) => CubeFace::Top,
        _ => CubeFace::Bottom,
    }
}

impl WindowedState {
    /// #13: save the live camera orbit as the new Home view (the right-click
    /// "set current view as home" context-menu action; Step 3).
    pub(super) fn set_home_to_current(&mut self) {
        self.home_view = HomeView::from_camera(&self.app_core.camera);
    }

    /// #13: begin an eased snap toward the saved Home view and set the home
    /// distance directly (the tween animates the orbit angles; distance is a
    /// non-orbit param so it is applied immediately, matching the face-snap
    /// path which never tweens distance). Wired to the Home button + context-menu
    /// Home item in Step 3; pure-ish here so the logic is testable.
    ///
    /// #13 Step 6.4: a USER-set home (`explicitly_set`) restores its saved distance
    /// verbatim. The DEFAULT home (never set by the user) instead FRAMES the model —
    /// the canned default distance (10) zooms in far too close on a large model — so
    /// Home re-fits the auto-framed distance, matching the Fit button's distance.
    pub(super) fn home_snap_tween(&mut self) -> SnapTween {
        let tween = self.home_view.snap_tween(&self.app_core.camera);
        self.app_core.camera.orbit_distance = if self.home_view.explicitly_set {
            self.home_view.distance
        } else {
            let region_dimensions = AppCore::region_dimensions_for(
                &self.panel_state.scene,
                self.panel_state.geometry.voxels_per_block,
            );
            self.app_core.camera.target = glam::Vec3::ZERO;
            OrbitCamera::auto_framed_distance(region_dimensions)
        };
        tween
    }

    /// #13: frame the model (the "Fit to view" action). Recompute the auto-frame
    /// distance from the scene's region dimensions and recentre the target on the
    /// model centroid — the recentred composite always sits at the world origin
    /// (`resolve_region` centres it), so the centroid is `Vec3::ZERO`. No geometry
    /// rebuild: only the camera distance + target change. The distance math is the
    /// same `auto_framed_distance` covered by camera tests.
    pub(super) fn fit_to_view(&mut self) {
        let region_dimensions = AppCore::region_dimensions_for(
            &self.panel_state.scene,
            self.panel_state.geometry.voxels_per_block,
        );
        self.app_core.camera.target = glam::Vec3::ZERO;
        self.app_core.camera.orbit_distance = OrbitCamera::auto_framed_distance(region_dimensions);
    }

    /// Is the pixel `(x, y)` inside the view-cube viewport? Signal (#86): the cube is
    /// in the **top-right** of the central 3D viewport rect (cached from the last
    /// frame). The corner comes from the SAME [`view_cube_corner`] the renderer draws
    /// with, so the hit rect always coincides with the drawn cube. When the viewport
    /// is below the minimum size the cube isn't drawn, so no pixel is inside it.
    pub(super) fn position_in_view_cube(&self, x: f64, y: f64) -> bool {
        let Some((corner_x, corner_y)) =
            view_cube_corner(self.last_viewport_px, self.last_cube_right_inset)
        else {
            return false;
        };
        let (corner_x, corner_y) = (corner_x as f64, corner_y as f64);
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f64;
        x >= corner_x && x <= corner_x + size && y >= corner_y && y <= corner_y + size
    }

    /// Is the pixel `(x, y)` inside the Signal chrome (the floating display stack or the
    /// icon rail)? Rects are cached from the last rendered frame
    /// (`PreparedEguiFrame::chrome_rects_px`), like the cube's. The camera gate treats
    /// pointer input inside them as chrome — with the stack no longer allocating in
    /// egui's root ui (the #88 dead-band regression), egui's own pointer-consumption
    /// heuristic no longer covers this chrome, so the shell reserves it here.
    pub(super) fn position_in_signal_chrome(&self, x: f64, y: f64) -> bool {
        let (x, y) = (x as f32, y as f32);
        self.last_chrome_rects_px.iter().any(|[rx, ry, rw, rh]| {
            x >= *rx && x < rx + rw && y >= *ry && y < ry + rh
        })
    }

    /// The ViewCube's on-screen square in window pixels, so the chrome hit-math
    /// ([`classify_cube_point`]) shares the SAME rect as [`Self::position_in_view_cube`]
    /// and the renderer (both via [`view_cube_corner`]). A degenerate rect (size 0) is
    /// returned when the cube isn't drawn (viewport below the minimum size), so every
    /// chrome hit-test cleanly misses.
    pub(super) fn cube_rect(&self) -> CubeRect {
        match view_cube_corner(self.last_viewport_px, self.last_cube_right_inset) {
            Some((corner_x, corner_y)) => CubeRect {
                x: corner_x as f32,
                y: corner_y as f32,
                size: VIEW_CUBE_VIEWPORT_PIXELS as f32,
            },
            None => CubeRect { x: 0.0, y: 0.0, size: 0.0 },
        }
    }

    /// Execute a [`ChromeClickAction`] resolved from a chrome-zone left-click
    /// (#13 Step 3). The pure mapping lives in `chrome_zone_left_click_action`; this
    /// only carries out the side effects (start a tween, run Home/Fit). A roll-arrow
    /// click resolves to a roll `Snap` tween (#13 Step 5: the real roll DOF).
    pub(super) fn run_chrome_action(&mut self, action: ChromeClickAction) {
        match action {
            ChromeClickAction::Snap(tween) => self.snap_tween = Some(tween),
            ChromeClickAction::Home => self.snap_tween = Some(self.home_snap_tween()),
            ChromeClickAction::Fit => self.fit_to_view(),
        }
    }

    /// Ray-cast a click inside the view-cube viewport against the cube and return
    /// the hit [`ViewCubeElement`] (face / edge / corner). NDC is computed within
    /// the cube's screen rect, then unprojected through the view-cube matrix; the
    /// entry face is found by a slab intersection, and the 3D hit point's in-plane
    /// coordinates pick one of the face's 9 hot zones (3×3 grid at the Signal 68 %
    /// centre patch, ±`VIEW_CUBE_ZONE_THRESHOLD`): centre → the face, an edge zone →
    /// this face + the neighbour the zone points toward, a corner zone → this face +
    /// both neighbours.
    pub(super) fn pick_view_cube_element(&self, x: f64, y: f64) -> Option<ViewCubeElement> {
        // Signal (#86): the cube's corner is the top-right of the central viewport rect
        // (shared with the renderer via `view_cube_corner`).
        let (corner_x, corner_y) = view_cube_corner(self.last_viewport_px, self.last_cube_right_inset)?;
        let (corner_x, corner_y) = (corner_x as f32, corner_y as f32);
        let size = VIEW_CUBE_VIEWPORT_PIXELS as f32;
        // NDC inside the cube rect (y up).
        let ndc_x = ((x as f32 - corner_x) / size) * 2.0 - 1.0;
        let ndc_y = -(((y as f32 - corner_y) / size) * 2.0 - 1.0);

        // Unproject the NDC point through the view-cube matrix into a world ray
        // (inverse-VP through the near/far clip planes; the generic form lives in the
        // camera crate), then resolve the hit face + hot zone with the pure ViewCube
        // slab picker in the `raycast` crate. This function is the thin adapter: it reads
        // the viewport/camera state and maps the picker's axes+signs to CubeFace.
        let view_projection = self.app_core.camera.view_cube_view_projection();
        let ray = camera::unproject_screen_point_to_ray(view_projection, ndc_x, ndc_y)?;

        // Slab intersection against the cube [-HALF, HALF]^3; the entry face's dominant
        // axis + sign give the material index / CubeFace.
        let hit = raycast::pick_view_cube_slab(ray, raycast::VIEW_CUBE_HALF_EXTENT)?;
        // Map (axis, sign) → material index (+X,-X,+Y,-Y,+Z,-Z) → CubeFace.
        let material_index = hit.entry_axis * 2 + if hit.entry_sign > 0.0 { 0 } else { 1 };
        let face = CubeFace::from_material_index(material_index)?;

        // The two in-plane axes' 3×3 hot zones (split at ±HALF/3) point toward the
        // neighbouring faces; the combined set of faces resolves the element (Z-up:
        // +X→Right, +Y→Back, +Z→Top).
        let neighbours: Vec<CubeFace> =
            raycast::view_cube_hot_zone_neighbours(&hit, raycast::VIEW_CUBE_ZONE_THRESHOLD)
                .into_iter()
                .map(|(axis, positive)| face_for_axis_sign(axis, positive))
                .collect();

        Some(match neighbours.as_slice() {
            [] => ViewCubeElement::from_face(face),
            [a] => ViewCubeElement::from_edge(face, *a),
            [a, b] => ViewCubeElement::from_corner(face, *a, *b),
            _ => ViewCubeElement::from_face(face),
        })
    }
}
