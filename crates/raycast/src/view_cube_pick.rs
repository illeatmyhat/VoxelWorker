//! The ViewCube element picker's ray-slab-with-entry-axis test.
//!
//! Casting a ray at the small on-screen orientation cube and deciding which element it
//! hit — a face, an edge, or a corner — is pure ray geometry: a slab intersection
//! against the cube `[-half, half]³` gives the entered face (its dominant axis and
//! sign), and the 3×3 grid of hot zones on that face, split at the ±(0.68·half)
//! thresholds, decides whether the pick is the face's centre (→ the face), an edge
//! column/row (→ the face plus one in-plane neighbour), or a corner (→ the face plus
//! two). This module holds that geometry; the app maps the axes and signs to its
//! ViewCube face vocabulary.
//!
//! ## Zone proportion (Signal spec)
//!
//! The centre patch spans **68 %** of each face (16 % edge strips on either side) — the
//! `docs/design/viewport-chrome-signal.md` proportion. The pick threshold and the
//! renderer's drawn slice lines are BOTH derived from [`VIEW_CUBE_CENTRE_PATCH_FRACTION`]
//! so the picture is the hit-map: a fragment on a face is in the central zone exactly
//! when its in-plane coordinate is within `±(0.68·half)` (the drawn 16 %/84 % slice
//! lines sit on those same planes).
//!
//! ## Literature
//!
//! The intersection is the same **slab method** as [`substrate::spatial::Ray::intersect_box_slab`]
//! (Kay & Kajiya 1986; Ericson 2005), but retained here in the variant that also reports
//! the entered face's *axis and sign* — which the box-interval primitive does not surface
//! — and that uses the picker's own `1e-6` parallel-axis guard. The cube itself is the
//! Autodesk ViewCube widget (its orientation model lives in the `camera` crate).

use glam::Vec3;
use substrate::spatial::Ray;

/// The ViewCube's half-extent: the cube spans `[-0.7, 0.7]` on each axis.
pub const VIEW_CUBE_HALF_EXTENT: f32 = 0.7;

/// The fraction of a face spanned by its central (face-centre) zone — the Signal
/// spec's **68 % centre patch**, leaving two 16 % edge strips. The renderer draws the
/// 3×3 slice lines at this same proportion so the drawn partition IS the pick partition.
pub const VIEW_CUBE_CENTRE_PATCH_FRACTION: f32 = 0.68;

/// The hot-zone threshold: the half-width of the 68 %-centre patch, in cube units. An
/// in-plane hit coordinate beyond `±VIEW_CUBE_ZONE_THRESHOLD` falls in that axis's
/// edge/corner (16 %) strip rather than the central face zone. Derived from
/// [`VIEW_CUBE_CENTRE_PATCH_FRACTION`] so a retune moves both the pick and the drawn
/// slices together: a centre patch covering fraction `f` of the full `2·half` face
/// extends `±(f·half)` from the face centre.
pub const VIEW_CUBE_ZONE_THRESHOLD: f32 =
    VIEW_CUBE_HALF_EXTENT * VIEW_CUBE_CENTRE_PATCH_FRACTION;

/// The parallel-axis guard: a direction component below this magnitude is treated as
/// parallel to that pair of slab planes (mirrors the picker's original `1e-6`).
const PARALLEL_GUARD: f32 = 1e-6;

/// Where a ray entered the ViewCube: the dominant entry-face axis (0=x, 1=y, 2=z), its
/// sign (`-1.0` entered the `-half` face, `+1.0` the `+half` face), and the 3D hit point
/// on that face in cube space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewCubeSlabHit {
    /// The entered face's dominant axis (0=x, 1=y, 2=z).
    pub entry_axis: usize,
    /// The entered face's sign: `-1.0` for the `-half` face, `+1.0` for the `+half` face.
    pub entry_sign: f32,
    /// The 3D hit point on the entered face, in cube space.
    pub hit_point: Vec3,
}

/// Intersect a ray with the ViewCube `[-half_extent, half_extent]³` by the slab method,
/// returning the entered face (axis + sign) and the hit point, or `None` if the ray
/// misses the cube (or only meets it behind the origin). Reproduces the picker's exact
/// arithmetic: the `1e-6` parallel-axis guard, the entry face = the axis with the largest
/// near-slab parameter, and the miss test `t_entry > t_exit || t_exit < 0`.
pub fn pick_view_cube_slab(ray: Ray, half_extent: f32) -> Option<ViewCubeSlabHit> {
    let origin = ray.origin;
    let direction = ray.direction;
    let mut t_entry = f32::NEG_INFINITY;
    let mut entry_axis = 0usize;
    let mut entry_sign = 1.0f32;
    let mut t_exit = f32::INFINITY;
    for axis in 0..3 {
        let o = origin[axis];
        let d = direction[axis];
        if d.abs() < PARALLEL_GUARD {
            if !(-half_extent..=half_extent).contains(&o) {
                return None; // parallel and outside the slab
            }
            continue;
        }
        let mut t0 = (-half_extent - o) / d;
        let mut t1 = (half_extent - o) / d;
        let mut sign = -1.0; // entering the -half face
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
            sign = 1.0; // entering the +half face
        }
        if t0 > t_entry {
            t_entry = t0;
            entry_axis = axis;
            entry_sign = sign;
        }
        t_exit = t_exit.min(t1);
    }
    if t_entry > t_exit || t_exit < 0.0 {
        return None;
    }
    Some(ViewCubeSlabHit {
        entry_axis,
        entry_sign,
        hit_point: origin + direction * t_entry,
    })
}

/// The in-plane neighbour axes+signs the hit's 3×3 hot zones trigger. For each of the two
/// axes NOT equal to the entered face's axis, if the hit's coordinate on that axis exceeds
/// `+threshold` the zone points toward that axis's positive face (`(axis, true)`); below
/// `-threshold`, its negative face (`(axis, false)`); within the band, nothing. Zero
/// neighbours ⇒ the face centre, one ⇒ an edge, two ⇒ a corner. The app maps each
/// `(axis, positive)` to its ViewCube face vocabulary.
pub fn view_cube_hot_zone_neighbours(hit: &ViewCubeSlabHit, threshold: f32) -> Vec<(usize, bool)> {
    let mut neighbours = Vec::with_capacity(2);
    for axis in 0..3 {
        if axis == hit.entry_axis {
            continue;
        }
        let coordinate = hit.hit_point[axis];
        if coordinate > threshold {
            neighbours.push((axis, true));
        } else if coordinate < -threshold {
            neighbours.push((axis, false));
        }
    }
    neighbours
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ray fired straight down −z at the cube's top enters the +z face (axis 2, sign +),
    /// near the face centre, so it triggers no in-plane neighbours.
    #[test]
    fn straight_on_hit_picks_the_face_centre() {
        let ray = Ray::new(Vec3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = pick_view_cube_slab(ray, VIEW_CUBE_HALF_EXTENT).expect("hit");
        assert_eq!(hit.entry_axis, 2);
        assert_eq!(hit.entry_sign, 1.0);
        assert!(view_cube_hot_zone_neighbours(&hit, VIEW_CUBE_ZONE_THRESHOLD).is_empty());
    }

    /// A ray aimed at the +z face but offset far in +x lands in that face's +x edge zone,
    /// yielding exactly one neighbour: (x, positive).
    #[test]
    fn offset_hit_picks_an_edge_neighbour() {
        let ray = Ray::new(Vec3::new(0.6, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = pick_view_cube_slab(ray, VIEW_CUBE_HALF_EXTENT).expect("hit");
        assert_eq!(hit.entry_axis, 2);
        let neighbours = view_cube_hot_zone_neighbours(&hit, VIEW_CUBE_ZONE_THRESHOLD);
        assert_eq!(neighbours, vec![(0usize, true)]);
    }

    /// A ray that passes the cube by misses.
    #[test]
    fn ray_past_the_cube_misses() {
        let ray = Ray::new(Vec3::new(5.0, 5.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(pick_view_cube_slab(ray, VIEW_CUBE_HALF_EXTENT).is_none());
    }
}
