//! Picking and dragging a translate gizmo's axis handles — the ray/line closest-approach
//! test, and the lattice snap that turns a gesture into an exact voxel count.
//!
//! ## The concept and its literature
//!
//! Both halves are the classic **closest points of two skew lines** (Eberly, *3D Game Engine
//! Design*, §"Distance Between Lines"; Ericson, *Real-Time Collision Detection*, §5.1.8). For
//! a cursor ray `P(s) = o + s·d` and a handle axis `Q(t) = pivot + t·a`, the parameters of
//! mutual closest approach solve a 2×2 system whose determinant is `1 − (d·a)²` once both
//! directions are unit. That determinant vanishing IS the degenerate case — the cursor is
//! sighting straight down the handle, where the gesture carries no information about position
//! along it — so it is reported rather than clamped away.
//!
//! Picking clamps `t` to the drawn handle segment before measuring; dragging does not, because
//! a drag legitimately continues past the handle's drawn end.
//!
//! ## Why the snap lives here rather than at the call site
//!
//! A drag is only as precise as where it lands, and this project's whole premise is exact
//! voxel counts. Snapping in the same place as the solve keeps "what the cursor points at" and
//! "what that means on the lattice" in one testable unit — a caller cannot accidentally
//! commit an unsnapped float. Everything here is in VOXELS: the caller works in the recentred
//! render frame, and a value carries the frame it was authored in rather than having one
//! re-derived downstream.

use glam::Vec3;
use substrate::spatial::Ray;

/// Which handle of the translate gizmo a gesture is on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
}

impl GizmoAxis {
    /// The unit direction this handle runs along.
    pub fn direction(self) -> Vec3 {
        match self {
            GizmoAxis::X => Vec3::X,
            GizmoAxis::Y => Vec3::Y,
            GizmoAxis::Z => Vec3::Z,
        }
    }

    /// The index this axis occupies in a `[_; 3]` — Z-up world, so Z is index 2.
    pub fn index(self) -> usize {
        match self {
            GizmoAxis::X => 0,
            GizmoAxis::Y => 1,
            GizmoAxis::Z => 2,
        }
    }

    /// Every handle, in axis order.
    pub const ALL: [GizmoAxis; 3] = [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z];
}

/// Below this, the cursor ray and the handle are treated as parallel: sighting down a handle
/// says nothing about position along it, so there is no gesture to read.
const PARALLEL_EPSILON: f32 = 1.0e-4;

/// The parameters of mutual closest approach between `ray` and the line through `pivot` along
/// `axis`, as `(t_on_axis, separation)`. `None` when the two are parallel to within
/// [`PARALLEL_EPSILON`].
///
/// `t_on_axis` is a signed distance from `pivot` in the caller's units (voxels, here).
fn closest_approach(ray: &Ray, pivot: Vec3, axis: GizmoAxis) -> Option<(f32, f32)> {
    let d = ray.direction.normalize_or_zero();
    if d == Vec3::ZERO {
        return None;
    }
    let a = axis.direction();
    let w0 = ray.origin - pivot;

    // The 2x2 solve: with |d| = |a| = 1 the determinant reduces to 1 - (d·a)².
    let b = d.dot(a);
    let determinant = 1.0 - b * b;
    if determinant.abs() < PARALLEL_EPSILON {
        return None;
    }
    let d_dot_w0 = d.dot(w0);
    let a_dot_w0 = a.dot(w0);

    let t = (a_dot_w0 - b * d_dot_w0) / determinant;
    let s = (b * a_dot_w0 - d_dot_w0) / determinant;

    let on_ray = ray.origin + d * s;
    let on_axis = pivot + a * t;
    Some((t, (on_ray - on_axis).length()))
}

/// Which handle the cursor is over, if any.
///
/// `handle_length` is how far each handle extends from `pivot` (voxels); `pick_radius` is how
/// close the ray must pass to count as a hit. The nearest qualifying handle wins, so the
/// crowded region near the pivot — where all three handles are within a radius of each other —
/// resolves to whichever the cursor is genuinely closest to rather than to axis order.
pub fn pick_gizmo_axis(
    ray: &Ray,
    pivot: Vec3,
    handle_length: f32,
    pick_radius: f32,
) -> Option<GizmoAxis> {
    let mut best: Option<(GizmoAxis, f32)> = None;
    for axis in GizmoAxis::ALL {
        let Some((t, separation)) = closest_approach(ray, pivot, axis) else {
            continue;
        };
        // Measure against the DRAWN handle: a ray passing close to the axis LINE far beyond
        // the handle's end is not over the handle.
        let clamped = t.clamp(0.0, handle_length);
        let separation = if (clamped - t).abs() > f32::EPSILON {
            let on_axis = pivot + axis.direction() * clamped;
            let d = ray.direction.normalize_or_zero();
            let w = on_axis - ray.origin;
            let along = w.dot(d);
            (w - d * along).length()
        } else {
            separation
        };
        if separation <= pick_radius && best.is_none_or(|(_, b)| separation < b) {
            best = Some((axis, separation));
        }
    }
    best.map(|(axis, _)| axis)
}

/// How far along `axis` the cursor currently points, in voxels from `pivot`.
///
/// Unclamped: a drag may legitimately run past the handle's drawn end. `None` when the cursor
/// is sighting down the handle, where the gesture is degenerate — the caller should hold the
/// last good value rather than snapping the object to an arbitrary place.
pub fn drag_distance_along_axis(ray: &Ray, pivot: Vec3, axis: GizmoAxis) -> Option<f32> {
    closest_approach(ray, pivot, axis).map(|(t, _)| t)
}

/// Snap a voxel distance to whole `step` voxels, rounding half away from zero.
///
/// This is what reconciles direct manipulation with exact authoring: the gesture is
/// continuous, the result is an integer voxel count, and there is no float left for a caller
/// to commit by accident. `step` of 0 is treated as 1 (voxel granularity).
pub fn snap_voxels(distance_voxels: f32, step_voxels: u32) -> i64 {
    let step = step_voxels.max(1) as f32;
    let steps = (distance_voxels / step).abs() + 0.5;
    let magnitude = (steps.floor() as i64) * step_voxels.max(1) as i64;
    if distance_voxels.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray aimed at a point, from far enough away that the origin never lands inside the
    /// geometry under test.
    fn ray_at(from: Vec3, toward: Vec3) -> Ray {
        Ray::new(from, (toward - from).normalize())
    }

    #[test]
    fn a_cursor_over_the_x_handle_picks_x() {
        // Sighting from +Z down at a point partway along the X handle.
        let ray = ray_at(Vec3::new(4.0, 0.0, 30.0), Vec3::new(4.0, 0.0, 0.0));
        assert_eq!(
            pick_gizmo_axis(&ray, Vec3::ZERO, 10.0, 1.0),
            Some(GizmoAxis::X)
        );
    }

    #[test]
    fn a_cursor_over_the_z_handle_picks_z() {
        let ray = ray_at(Vec3::new(0.0, 30.0, 4.0), Vec3::new(0.0, 0.0, 4.0));
        assert_eq!(
            pick_gizmo_axis(&ray, Vec3::ZERO, 10.0, 1.0),
            Some(GizmoAxis::Z)
        );
    }

    #[test]
    fn a_cursor_far_from_every_handle_picks_nothing() {
        let ray = ray_at(Vec3::new(40.0, 40.0, 30.0), Vec3::new(40.0, 40.0, 0.0));
        assert_eq!(pick_gizmo_axis(&ray, Vec3::ZERO, 10.0, 1.0), None);
    }

    #[test]
    fn a_cursor_past_the_handles_end_does_not_pick_it() {
        // On the X LINE, but well beyond the 10-voxel handle: not over the drawn handle.
        let ray = ray_at(Vec3::new(40.0, 0.0, 30.0), Vec3::new(40.0, 0.0, 0.0));
        assert_eq!(pick_gizmo_axis(&ray, Vec3::ZERO, 10.0, 1.0), None);
    }

    #[test]
    fn the_nearest_handle_wins_when_two_are_in_range() {
        // Nearer the X handle than the Y handle, with a radius generous enough for both.
        let ray = ray_at(Vec3::new(5.0, 1.0, 30.0), Vec3::new(5.0, 1.0, 0.0));
        assert_eq!(
            pick_gizmo_axis(&ray, Vec3::ZERO, 10.0, 8.0),
            Some(GizmoAxis::X)
        );
    }

    #[test]
    fn the_pivot_is_honoured_rather_than_assumed_to_be_the_origin() {
        let pivot = Vec3::new(100.0, -40.0, 7.0);
        let ray = ray_at(pivot + Vec3::new(4.0, 0.0, 30.0), pivot + Vec3::X * 4.0);
        assert_eq!(pick_gizmo_axis(&ray, pivot, 10.0, 1.0), Some(GizmoAxis::X));
    }

    #[test]
    fn dragging_reads_the_distance_along_the_axis() {
        // Sighting straight down -Z at x = 6 gives t = 6 on the X handle.
        let ray = ray_at(Vec3::new(6.0, 0.0, 30.0), Vec3::new(6.0, 0.0, 0.0));
        let t = drag_distance_along_axis(&ray, Vec3::ZERO, GizmoAxis::X).expect("not parallel");
        assert!((t - 6.0).abs() < 1.0e-3, "t = {t}");
    }

    #[test]
    fn dragging_reads_a_negative_distance_on_the_far_side() {
        let ray = ray_at(Vec3::new(-6.0, 0.0, 30.0), Vec3::new(-6.0, 0.0, 0.0));
        let t = drag_distance_along_axis(&ray, Vec3::ZERO, GizmoAxis::X).expect("not parallel");
        assert!((t + 6.0).abs() < 1.0e-3, "t = {t}");
    }

    #[test]
    fn sighting_down_the_handle_is_degenerate_and_reports_none() {
        // Looking along +X at the X handle: the gesture cannot say where along it we are.
        let ray = Ray::new(Vec3::new(-30.0, 0.0, 0.0), Vec3::X);
        assert_eq!(
            drag_distance_along_axis(&ray, Vec3::ZERO, GizmoAxis::X),
            None
        );
        // ... but the OTHER handles are still readable from that viewpoint.
        assert!(drag_distance_along_axis(&ray, Vec3::ZERO, GizmoAxis::Y).is_some());
    }

    #[test]
    fn snapping_rounds_to_whole_voxels() {
        assert_eq!(snap_voxels(3.4, 1), 3);
        assert_eq!(snap_voxels(3.6, 1), 4);
        assert_eq!(snap_voxels(0.0, 1), 0);
    }

    #[test]
    fn snapping_rounds_half_away_from_zero_symmetrically() {
        // The sign must not change the magnitude — an asymmetric round would make a drag
        // out and back land somewhere other than where it started.
        assert_eq!(snap_voxels(2.5, 1), 3);
        assert_eq!(snap_voxels(-2.5, 1), -3);
        assert_eq!(snap_voxels(-3.4, 1), -3);
        assert_eq!(snap_voxels(-3.6, 1), -4);
    }

    #[test]
    fn snapping_to_blocks_lands_on_block_multiples() {
        // A 16-voxel block: everything must land on a multiple of 16.
        assert_eq!(snap_voxels(20.0, 16), 16);
        assert_eq!(snap_voxels(25.0, 16), 32);
        assert_eq!(snap_voxels(-20.0, 16), -16);
        for distance in [0.0_f32, 7.0, 9.0, 100.0, -100.0] {
            assert_eq!(snap_voxels(distance, 16) % 16, 0, "distance {distance}");
        }
    }

    #[test]
    fn a_zero_step_is_treated_as_voxel_granularity() {
        assert_eq!(snap_voxels(3.6, 0), 4);
    }
}
