//! Projection matrices and the inverse — screen-point → world-ray unprojection.
//!
//! [`OrbitCamera::view_projection`] composes the look-at view with either a
//! perspective or an orthographic projection into the single `view_projection`
//! matrix the shaders consume, choosing the near/far planes to ENCLOSE the scene's
//! bounding sphere so no in-scene geometry is ever depth-clipped.
//! [`OrbitCamera::view_cube_view_projection`] is the small independent orthographic
//! matrix for the corner `ViewCube`. [`unproject_screen_point_to_ray`] runs the
//! inverse: it maps a normalized-device-coordinate screen point back through the
//! inverse `view_projection` to the pair of world points on the near and far clip
//! planes, and returns the [`Ray`] through them — the generic operation behind
//! every "what did the cursor point at" pick.
//!
//! Cite: look-at + perspective/orthographic projection and the inverse-VP
//! unprojection are standard viewing geometry (Akenine-Möller, Haines & Hoffman,
//! *Real-Time Rendering*, the viewing/projection chapters; the OpenGL/`gluUnProject`
//! screen→world convention). The `glam` `_rh` builders use the wgpu clip-space
//! convention (z ∈ [0, 1], y up in NDC).

use glam::{Mat4, Vec3, Vec4};
use substrate::spatial::Ray;

use crate::orbit::{OrbitCamera, ProjectionMode, ORTHO_HALF_HEIGHT_FACTOR, PERSPECTIVE_FOV_Y};

/// The scene camera matrices one frame renders with.
///
/// They are bundled so the full and
/// camera-relative forms can never drift apart: `view_projection` for
/// forward-projected geometry (the view multiply makes vertices eye-relative before
/// any precision-losing arithmetic), `camera_relative_view_projection` for the
/// passes that UNPROJECT per fragment (brick raymarch ray setup, analytic infinite
/// grid) — inverting a matrix that carries a ~10^5-voxel eye translation melts the
/// `/w` divide — and the `camera_eye` those passes add back OUTSIDE the matrix math.
#[derive(Debug, Clone, Copy)]
pub struct SceneMatrices {
    pub view_projection: Mat4,
    /// The forward matrix of the RAY FRAME — the frame the per-fragment-unprojecting
    /// passes (brick raymarch, analytic infinite grid) run their ray + `frag_depth`
    /// math in. PERSPECTIVE anchors it at the eye (rotation-only view): the full
    /// matrix's inverse mixes a ~10^5-voxel eye translation into the `/w` divide and
    /// melts at wide-baseline coordinates. ORTHOGRAPHIC keeps the plain render frame
    /// (this is just `view_projection`): affine unprojection carries no `/w` and is
    /// precise at any coordinate, and reusing the exact pre-existing arithmetic keeps
    /// the GPU march bit-agreeing with its CPU mirror (the zero-tolerance parity net).
    pub ray_view_projection: Mat4,
    /// The RAY-RECONSTRUCTION matrix (same frame as `ray_view_projection`), whose
    /// inverse the passes unproject through. Under PERSPECTIVE its near/far come from
    /// the `CAMERA` (a `2·orbit_distance` sphere around the target), not the scene: a
    /// compact scene viewed from 10^5 voxels away makes the scene bracket a thin
    /// distant slab whose z=0/z=1 unprojections are two huge nearly-equal points —
    /// their difference, the ray direction, cancels catastrophically. Ray geometry is
    /// independent of clip planes, so the conditioned bracket is free; depth still
    /// projects through `ray_view_projection`. Under ORTHOGRAPHIC this is the scene
    /// matrix itself.
    pub ray_unprojection: Mat4,
    /// The ray frame's origin in the recentered render frame: the eye under
    /// PERSPECTIVE, zero under ORTHOGRAPHIC. Consumers add it back outside the
    /// matrix math.
    pub ray_eye: Vec3,
    /// The projection's NDC-depth mapping, for passes that compare two HARDWARE
    /// depth values with a world-length tolerance (the selection outline+wash).
    pub ndc_depth: NdcDepthMapping,
}

/// The two z-row coefficients of the projection matrix.
///
/// They are exposed so a screen-space
/// pass can convert a world-length tolerance into an NDC-depth tolerance AT a
/// sampled depth — comparing hardware z against hardware z directly. Never
/// linearize a hardware depth to compare in view space: the inverse map amplifies
/// quantization noise by `view_depth² / near`, exploding at wide baselines; the
/// forward slope below degrades gracefully instead.
///
/// With `d` = view depth (distance along the forward axis, positive in front) and
/// the glam `_rh` wgpu convention (z ∈ [0, 1], NOT reversed):
///
/// * perspective:  `ndc(d) = depth_offset / d − depth_scale`, so
///   `|∂ndc/∂d| = |depth_offset| / d²` and `d = depth_offset / (ndc + depth_scale)`.
/// * orthographic: `ndc(d) = depth_scale · (−d) + depth_offset`, so
///   `|∂ndc/∂d| = |depth_scale|` (constant).
///
/// An NDC tolerance for a world half-width `h` at a sampled `ndc` is then
/// `h · |∂ndc/∂d|`, floored by a few depth-buffer ULPs.
#[derive(Debug, Clone, Copy)]
pub struct NdcDepthMapping {
    /// The projection matrix's `z_axis.z` (the view-z multiplier of clip z).
    pub depth_scale: f32,
    /// The projection matrix's `w_axis.z` (the constant term of clip z).
    pub depth_offset: f32,
    /// Which branch of the mapping applies (`clip_w` is 1 vs `−view_z`).
    pub orthographic: bool,
}

impl OrbitCamera {
    /// Build the combined `view_projection` matrix for an aspect ratio (w/h), with
    /// the near/far planes derived to ENCLOSE the scene's bounding sphere
    /// (`scene_center` + `scene_radius`, render-frame units) so no in-scene geometry
    /// is ever depth-clipped.
    ///
    /// Keying near/far off `orbit_distance` alone (`near = distance·0.01`) clips the
    /// moment another object sits closer to the eye than the auto-framed target: Focus
    /// shrinks `orbit_distance` to fit one node, so a second node 15 blocks toward the
    /// camera falls in front of the near plane.
    /// Instead, project the bounding-sphere center onto the view axis and place the
    /// planes a sphere-radius (plus a small margin) either side: the WHOLE scene is
    /// then always within `[near, far]`, making near-plane clipping of scene
    /// geometry unrepresentable. Orthographic tolerates a near plane behind the eye,
    /// so its guarantee is absolute; perspective requires a positive near, so it
    /// clamps to a small floor (only reachable by zooming the eye inside the sphere,
    /// where a perspective near-clip is unavoidable anyway).
    ///
    /// The projection branch is chosen by [`OrbitCamera::projection_mode`]; the
    /// orthographic frustum's half-height tracks `orbit_distance` so zoom keeps
    /// working and the framing is preserved when toggling.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn view_projection(
        &self,
        aspect_ratio: f32,
        scene_center: Vec3,
        scene_radius: f32,
    ) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, self.up_vector());
        self.projection_enclosing_sphere(aspect_ratio, scene_center, scene_radius) * view
    }

    /// [`view_projection`](Self::view_projection) with the camera EYE at the frame
    /// origin: the identical projection (same near/far derivation) composed with the
    /// view's ROTATION only. Unprojecting through this matrix's inverse yields
    /// EYE-RELATIVE points — small numbers even when the render frame puts the eye at
    /// ~10^5 voxels — which keeps per-fragment unprojection (the `/w` divide) precise
    /// at wide-baseline coordinates. Recover render-frame positions by adding the eye
    /// back OUTSIDE the matrix math; symmetrically, forward-project points translated
    /// by `−eye`.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn camera_relative_view_projection(
        &self,
        aspect_ratio: f32,
        scene_center: Vec3,
        scene_radius: f32,
    ) -> Mat4 {
        // look_at_rh = [R | R·(−eye)]; zeroing the translation column leaves exactly
        // the rotation the full view matrix applies — same bits, no re-derivation.
        let mut rotation_only_view = Mat4::look_at_rh(self.eye(), self.target, self.up_vector());
        rotation_only_view.w_axis = Vec4::W;
        self.projection_enclosing_sphere(aspect_ratio, scene_center, scene_radius)
            * rotation_only_view
    }

    /// Both scene matrices + the eye as one [`SceneMatrices`] bundle — the form the
    /// per-fragment-unprojecting display passes consume.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn scene_matrices(
        &self,
        aspect_ratio: f32,
        scene_center: Vec3,
        scene_radius: f32,
    ) -> SceneMatrices {
        let view_projection = self.view_projection(aspect_ratio, scene_center, scene_radius);
        // The SAME projection the view_projection above composed (identical args →
        // identical near/far), so the mapping can never drift from the matrix.
        let projection = self.projection_enclosing_sphere(aspect_ratio, scene_center, scene_radius);
        let ndc_depth = NdcDepthMapping {
            depth_scale: projection.z_axis.z,
            depth_offset: projection.w_axis.z,
            orthographic: self.projection_mode == ProjectionMode::Orthographic,
        };
        match self.projection_mode {
            ProjectionMode::Perspective => SceneMatrices {
                view_projection,
                ray_view_projection: self.camera_relative_view_projection(
                    aspect_ratio,
                    scene_center,
                    scene_radius,
                ),
                // Camera-sized bracket: the near clamps to the 0.05 floor (eye inside
                // the sphere), giving a near point AT the eye — no cancellation in
                // `far − near` at any scene distance.
                ray_unprojection: self.camera_relative_view_projection(
                    aspect_ratio,
                    self.target,
                    (self.orbit_distance * 2.0).max(1.0),
                ),
                ray_eye: self.eye(),
                ndc_depth,
            },
            // Ortho: the plain render frame, bit-identical to the historical path.
            ProjectionMode::Orthographic => SceneMatrices {
                view_projection,
                ray_view_projection: view_projection,
                ray_unprojection: view_projection,
                ray_eye: Vec3::ZERO,
                ndc_depth,
            },
        }
    }

    /// The projection half of [`view_projection`](Self::view_projection): near/far
    /// placed a sphere-radius (plus margin) either side of the bounding-sphere
    /// center's view depth. Eye-position-independent apart from that depth, so the
    /// full and camera-relative view-projections share it verbatim.
    #[allow(clippy::arithmetic_side_effects, clippy::suboptimal_flops)]
    fn projection_enclosing_sphere(
        &self,
        aspect_ratio: f32,
        scene_center: Vec3,
        scene_radius: f32,
    ) -> Mat4 {
        // Signed depth from the eye to the bounding-sphere center along the view
        // axis (forward = the unit look direction, target − eye = −direction()).
        let forward = -self.direction();
        let center_depth = (scene_center - self.eye()).dot(forward);
        // A hair of slack so faces exactly on the sphere don't sit on a plane.
        let margin = scene_radius * 0.05 + 0.5;
        let mut near = center_depth - scene_radius - margin;
        let mut far = center_depth + scene_radius + margin;
        match self.projection_mode {
            ProjectionMode::Perspective => {
                // Perspective needs near > 0; clamp to a small floor and keep far
                // strictly beyond it (the matrix stays finite even when the scene
                // is behind the camera).
                near = near.max(0.05);
                far = far.max(near + 0.1);
                Mat4::perspective_rh(PERSPECTIVE_FOV_Y, aspect_ratio, near, far)
            }
            ProjectionMode::Orthographic => {
                // far − near = 2·(radius + margin) > 0 always, so the planes are
                // never inverted; a negative `near` (eye inside the sphere) is fine.
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
        }
    }

    /// View-projection for the corner view cube: an orthographic camera whose eye
    /// copies the MAIN camera's *direction* (`pos = direction * 4`, look at origin),
    /// so the small cube mirrors the current main view. Independent of
    /// `orbit_distance` / projection mode.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn view_cube_view_projection(&self) -> Mat4 {
        let eye = self.direction() * 4.0;
        // MUST share the main camera's up (`up_vector`) or the cube and the scene
        // desync at the pole — same singular-frame up, same orientation.
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, self.up_vector());
        // Half-extent 1.35 frames a cube of side 1.4 with a little margin.
        let projection = Mat4::orthographic_rh(-1.35, 1.35, -1.35, 1.35, 0.1, 100.0);
        projection * view
    }
}

/// Unproject a normalized-device-coordinate screen point through the inverse of a
/// `view_projection` matrix into the world-space [`Ray`] the cursor points along.
///
/// `ndc_x` / `ndc_y` are in clip space `[-1, 1]` with **y up** (a caller working in
/// window pixels first maps the pixel into the target rect and flips y). The two
/// NDC points `(ndc_x, ndc_y, 0)` and `(ndc_x, ndc_y, 1)` are the near and far clip
/// planes (the wgpu z ∈ [0, 1] convention `glam`'s `_rh` projections produce);
/// pushing both through `view_projection.inverse()` and dividing by `w` recovers
/// their world positions, and the ray runs from the near point toward the far one.
/// Returns `None` only if the two unprojected points coincide (a degenerate matrix),
/// so the direction cannot be normalized.
#[must_use]
#[allow(clippy::arithmetic_side_effects)]
pub fn unproject_screen_point_to_ray(view_projection: Mat4, ndc_x: f32, ndc_y: f32) -> Option<Ray> {
    let inverse = view_projection.inverse();
    let near = inverse * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inverse * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let near = near.truncate() / near.w;
    let far = far.truncate() / far.w;
    let direction = (far - near).normalize_or_zero();
    if direction == Vec3::ZERO {
        None
    } else {
        Some(Ray::new(near, direction))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::suboptimal_flops)]

    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    #[test]
    fn orthographic_near_far_enclose_the_whole_scene_sphere() {
        // A small orbit_distance (eye close to the target) with a scene far larger
        // than that distance: an `orbit_distance·0.01` near plane would sit in front
        // of a second object 15+ blocks toward the camera. Orthographic tolerates a
        // near plane BEHIND the eye, so the entire bounding sphere lands inside the
        // depth frustum and near-plane clipping of scene geometry is unrepresentable.
        let radius = 30.0;
        let camera = OrbitCamera {
            orbit_distance: 2.0, // eye very close to the target, deep inside the sphere
            projection_mode: ProjectionMode::Orthographic,
            ..OrbitCamera::default()
        };
        let vp = camera.view_projection(1.0, Vec3::ZERO, radius);
        let forward = -camera.direction();
        let up = camera.up_vector();
        let right = forward.cross(up).normalize();
        // Sample the sphere's six axis-extreme surface points (center = ZERO). The
        // two along the view axis are the binding ones; all must map to a depth
        // inside [0, 1] (glam's `_rh` projections use the wgpu [0,1] z range).
        for dir in [forward, -forward, up, -up, right, -right] {
            let clip = vp * (dir * radius).extend(1.0);
            let ndc_z = clip.z / clip.w;
            assert!(
                (0.0..=1.0).contains(&ndc_z),
                "sphere surface point {dir:?} depth-clipped (ndc_z={ndc_z})",
            );
        }
    }

    /// Both view matrices are all-finite (no NaN/inf) at the exact poles.
    #[test]
    fn view_matrices_finite_at_exact_poles() {
        for &phi in &[0.0f32, PI] {
            let camera = OrbitCamera {
                orbit_theta: -FRAC_PI_2,
                orbit_phi: phi,
                ..OrbitCamera::default()
            };
            let vp = camera.view_projection(1.0, Vec3::ZERO, 10.0);
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

    /// Both view matrices are all-finite at every roll in {0, π/2, π, −π/2}.
    #[test]
    fn view_matrices_finite_under_roll() {
        for &roll in &[0.0f32, FRAC_PI_2, PI, -FRAC_PI_2] {
            let camera = OrbitCamera {
                roll,
                ..OrbitCamera::default()
            };
            let vp = camera.view_projection(1.0, Vec3::ZERO, 10.0);
            assert!(
                vp.to_cols_array().iter().all(|v| v.is_finite()),
                "view_projection not finite at roll={roll}"
            );
            let cube = camera.view_cube_view_projection();
            assert!(
                cube.to_cols_array().iter().all(|v| v.is_finite()),
                "view_cube_view_projection not finite at roll={roll}"
            );
        }
    }

    /// The unprojected ray round-trips: its near-plane origin projects back to the
    /// same NDC point, and the ray points INTO the scene (away from the eye) — so a
    /// center-screen pick aims at the look target.
    #[test]
    fn unprojected_ray_round_trips_and_points_into_the_scene() {
        let camera = OrbitCamera::default();
        let vp = camera.view_projection(1.0, Vec3::ZERO, 10.0);
        // Center of the screen (NDC origin) unprojects to a ray toward the target.
        let ray = unproject_screen_point_to_ray(vp, 0.0, 0.0).expect("non-degenerate VP");
        // The direction must run from the eye toward the target (the forward axis).
        let forward = (camera.target - camera.eye()).normalize();
        assert!(
            ray.direction.dot(forward) > 0.999,
            "center pick should aim along forward: dot {}",
            ray.direction.dot(forward)
        );
        // Re-project the ray origin: it lands back at NDC (0, 0) on the near plane.
        let clip = vp * ray.origin.extend(1.0);
        assert!(
            (clip.x / clip.w).abs() < 1e-3 && (clip.y / clip.w).abs() < 1e-3,
            "origin should re-project to NDC center: {clip:?}"
        );
    }

    /// A degenerate (non-invertible → zero-direction) matrix yields `None` rather
    /// than a NaN ray.
    #[test]
    fn unproject_degenerate_matrix_is_none() {
        assert!(unproject_screen_point_to_ray(Mat4::ZERO, 0.0, 0.0).is_none());
    }

    /// `NdcDepthMapping` reproduces the depth the full `view_projection` assigns:
    /// project points at several view depths through the matrix and through the
    /// mapping's closed form — they must agree in both projection modes. Also pins
    /// the perspective inverse (`d = depth_offset / (ndc + depth_scale)`) and the
    /// analytic slope against a numeric derivative, since the selection outline's
    /// epsilon is built from exactly those forms.
    #[test]
    fn ndc_depth_mapping_matches_the_matrix_in_both_modes() {
        for mode in [ProjectionMode::Perspective, ProjectionMode::Orthographic] {
            let camera = OrbitCamera {
                projection_mode: mode,
                ..OrbitCamera::default()
            };
            let matrices = camera.scene_matrices(1.5, Vec3::new(2.0, -3.0, 1.0), 25.0);
            let mapping = matrices.ndc_depth;
            assert_eq!(mapping.orthographic, mode == ProjectionMode::Orthographic);
            let forward = -camera.direction();
            for fraction in [0.2f32, 0.5, 0.9] {
                // A point on the view axis at a depth inside the scene bracket.
                let depth = camera.orbit_distance * (0.5 + fraction);
                let point = camera.eye() + forward * depth;
                let clip = matrices.view_projection * point.extend(1.0);
                let matrix_ndc = clip.z / clip.w;
                let mapping_ndc = if mapping.orthographic {
                    mapping.depth_scale * (-depth) + mapping.depth_offset
                } else {
                    mapping.depth_offset / depth - mapping.depth_scale
                };
                assert!(
                    (matrix_ndc - mapping_ndc).abs() < 1e-4,
                    "{mode:?} depth {depth}: matrix ndc {matrix_ndc} vs mapping {mapping_ndc}"
                );
                // Slope: analytic |∂ndc/∂d| vs a central numeric derivative.
                let analytic_slope = if mapping.orthographic {
                    mapping.depth_scale.abs()
                } else {
                    // The inverse the shader runs: recover d from the sampled ndc.
                    let recovered = mapping.depth_offset / (mapping_ndc + mapping.depth_scale);
                    assert!(
                        (recovered - depth).abs() < 1e-2,
                        "{mode:?}: recovered depth {recovered} vs {depth}"
                    );
                    mapping.depth_offset.abs() / (depth * depth)
                };
                let step = 0.01;
                let ndc_at = |d: f32| {
                    let p = camera.eye() + forward * d;
                    let c = matrices.view_projection * p.extend(1.0);
                    c.z / c.w
                };
                let numeric_slope =
                    ((ndc_at(depth + step) - ndc_at(depth - step)) / (2.0 * step)).abs();
                assert!(
                    (analytic_slope - numeric_slope).abs() < numeric_slope * 0.01 + 1e-6,
                    "{mode:?} depth {depth}: analytic slope {analytic_slope} vs numeric {numeric_slope}"
                );
            }
        }
    }
}
