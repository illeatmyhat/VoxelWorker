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

/// Polar-angle clamp so the camera never flips over the poles.
const PHI_MIN: f32 = 0.05;
const PHI_MAX: f32 = std::f32::consts::PI - 0.05;

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
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            orbit_theta: 0.7,
            orbit_phi: 1.05,
            orbit_distance: 10.0,
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

    /// Orbit by a screen-space drag delta (left-drag): `theta -= dx * 0.01`,
    /// `phi -= dy * 0.01`, with `phi` clamped to `[PHI_MIN, PHI_MAX]`.
    pub fn orbit_by_drag(&mut self, delta_x: f32, delta_y: f32) {
        self.orbit_theta -= delta_x * 0.01;
        self.orbit_phi = (self.orbit_phi - delta_y * 0.01).clamp(PHI_MIN, PHI_MAX);
    }

    /// Zoom by a wheel step: `distance *= 1 ± 0.08`. Positive `scroll_lines`
    /// zooms in (closer).
    pub fn zoom_by_wheel(&mut self, scroll_lines: f32) {
        let factor = if scroll_lines > 0.0 { 1.0 - 0.08 } else { 1.0 + 0.08 };
        self.orbit_distance = (self.orbit_distance * factor).max(0.1);
    }

    /// Build the combined `view_projection` matrix for an aspect ratio (w/h).
    pub fn view_projection(&self, aspect_ratio: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        // Near/far chosen generously relative to the auto-framed distance so the
        // grid never clips at any zoom we allow.
        let near = (self.orbit_distance * 0.01).max(0.05);
        let far = self.orbit_distance * 10.0 + 1000.0;
        let projection = Mat4::perspective_rh(PERSPECTIVE_FOV_Y, aspect_ratio, near, far);
        projection * view
    }
}
