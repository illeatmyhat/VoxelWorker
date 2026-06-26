//! Frustum culling for the chunked voxel renderer (ADR 0002 E2, part of #19).
//!
//! Extracts the six clip-space frustum planes from a `view_projection` matrix
//! (the Gribb–Hartmann method) and tests each chunk's world-space axis-aligned
//! bounding box against them. A chunk is drawn only when its AABB intersects the
//! frustum, so at small scale every chunk is visible (identical output) while a
//! large scene viewed partially off-screen draws only the chunks on screen.
//!
//! This module is pure CPU maths (no GPU types) so it is unit-testable without a
//! device — see the tests at the bottom.

use glam::{Mat4, Vec3, Vec4};

/// A world-space axis-aligned bounding box (inclusive min/max corners).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    /// An empty/degenerate box that grows to fit points via [`Aabb::expand`].
    /// `min` starts at +∞ and `max` at −∞ so the first expansion sets both.
    pub fn empty() -> Self {
        Self {
            min: Vec3::splat(f32::INFINITY),
            max: Vec3::splat(f32::NEG_INFINITY),
        }
    }

    /// Grow the box to include `point`.
    pub fn expand(&mut self, point: Vec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
    }
}

/// The six planes of a view frustum, each stored as `(a, b, c, d)` where
/// `a*x + b*y + c*z + d >= 0` for points on the inside (the normal points
/// inward). Extracted from a `view_projection` matrix via Gribb–Hartmann.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extract the six frustum planes from a combined `view_projection` matrix
    /// (Gribb–Hartmann, 2001). glam stores matrices column-major; the rows of
    /// the matrix are recovered from the columns. The planes are normalised so a
    /// later signed-distance test is in world units (not strictly required for a
    /// pure inside/outside test, but it keeps the maths well-conditioned).
    pub fn from_view_projection(view_projection: Mat4) -> Self {
        // Row i of the matrix: the i-th component of each of the four columns.
        let c = view_projection.to_cols_array_2d();
        let row = |i: usize| Vec4::new(c[0][i], c[1][i], c[2][i], c[3][i]);
        let row0 = row(0);
        let row1 = row(1);
        let row2 = row(2);
        let row3 = row(3);

        // Inward-pointing planes (normal · p + d >= 0 inside):
        //   left   = row3 + row0      right = row3 - row0
        //   bottom = row3 + row1      top   = row3 - row1
        //   near   = row3 + row2      far   = row3 - row2
        let raw = [
            row3 + row0,
            row3 - row0,
            row3 + row1,
            row3 - row1,
            row3 + row2,
            row3 - row2,
        ];
        let mut planes = [Vec4::ZERO; 6];
        for (out, plane) in planes.iter_mut().zip(raw) {
            let normal_length = plane.truncate().length();
            *out = if normal_length > 1e-6 {
                plane / normal_length
            } else {
                plane
            };
        }
        Self { planes }
    }

    /// Does `aabb` intersect (or lie inside) the frustum? Uses the standard
    /// "positive vertex" test: for each plane, pick the AABB corner farthest
    /// along the plane's inward normal; if even that corner is outside the
    /// plane, the whole box is outside and we reject. This can produce false
    /// positives for boxes straddling a frustum corner, but NEVER a false
    /// negative — so no on-screen geometry is ever wrongly culled.
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        for plane in &self.planes {
            let normal = plane.truncate();
            // The corner of the box farthest in the direction of the inward
            // normal (max where the normal is positive, min where negative).
            let positive_vertex = Vec3::new(
                if normal.x >= 0.0 { aabb.max.x } else { aabb.min.x },
                if normal.y >= 0.0 { aabb.max.y } else { aabb.min.y },
                if normal.z >= 0.0 { aabb.max.z } else { aabb.min.z },
            );
            if normal.dot(positive_vertex) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    /// A simple perspective camera looking down −Z from `eye`.
    fn camera_at(eye: Vec3) -> Mat4 {
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let projection =
            Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 1000.0);
        projection * view
    }

    fn unit_box_at(center: Vec3) -> Aabb {
        Aabb {
            min: center - Vec3::splat(0.5),
            max: center + Vec3::splat(0.5),
        }
    }

    #[test]
    fn box_at_origin_is_visible() {
        let frustum = Frustum::from_view_projection(camera_at(Vec3::new(0.0, 0.0, 10.0)));
        assert!(frustum.intersects_aabb(&unit_box_at(Vec3::ZERO)));
    }

    #[test]
    fn box_behind_camera_is_culled() {
        // Camera at +Z=10 looking at origin (down −Z): a box far in +Z (behind
        // the eye) must be rejected.
        let frustum = Frustum::from_view_projection(camera_at(Vec3::new(0.0, 0.0, 10.0)));
        assert!(!frustum.intersects_aabb(&unit_box_at(Vec3::new(0.0, 0.0, 100.0))));
    }

    #[test]
    fn box_far_off_to_the_side_is_culled() {
        // A box far in +X is outside the ~45° FOV frustum.
        let frustum = Frustum::from_view_projection(camera_at(Vec3::new(0.0, 0.0, 10.0)));
        assert!(!frustum.intersects_aabb(&unit_box_at(Vec3::new(1000.0, 0.0, 0.0))));
    }

    #[test]
    fn enclosing_aabb_intersects() {
        // A box that fully contains the camera + target still intersects.
        let frustum = Frustum::from_view_projection(camera_at(Vec3::new(0.0, 0.0, 10.0)));
        let big = Aabb {
            min: Vec3::splat(-1000.0),
            max: Vec3::splat(1000.0),
        };
        assert!(frustum.intersects_aabb(&big));
    }
}
