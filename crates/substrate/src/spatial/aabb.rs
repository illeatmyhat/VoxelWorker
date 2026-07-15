//! Axis-aligned bounding boxes: the half-open integer lattice box and its closed
//! continuous (f32) twin.
//!
//! The textbook axis-aligned bounding box (AABB) appears here in the TWO variants a
//! voxel application actually needs, co-located so their deliberately different
//! conventions can be read side by side:
//!
//! * [`LatticeAabb`] — **integer** corners under the **half-open** `[min, max)`
//!   convention: a box owns the integer cells `min[axis] <= c < max[axis]`, and two
//!   boxes that merely touch on a face (one's `max` equalling the other's `min`) do
//!   **not** overlap. That half-open rule is what makes lattice boxes tile space
//!   without double-counting the shared boundary cell — the same reason a grid of
//!   cells indexed by `floor(position / extent)` partitions cleanly. This is the
//!   *ownership/broadphase* box: an edit broadphase asking "which boxes share a
//!   cell" must NOT report neighbours that only abut.
//!
//! * [`RealAabb`] — **f32** corners under the **closed** `[min, max]` convention: a
//!   continuous region of render/world space with inclusive faces, so two boxes
//!   that touch on a face DO intersect. This is the *conservative-culling* box: a
//!   frustum or overlap test over continuous space must err toward intersection
//!   (a face-on-plane chunk must still be drawn), the exact opposite bias from the
//!   lattice box's disjoint-tiling rule. It grows from an ±∞-sentinel
//!   [`RealAabb::empty`] via [`RealAabb::expand`], the standard fold for bounding a
//!   point set.
//!
//! `LatticeAabb` corners are `i64` so the box can address a large integer coordinate
//! range without overflow; nothing in either type is parameterised by the meaning of
//! a coordinate.
//!
//! Cite: Ericson, *Real-Time Collision Detection* (2005), ch. 4 (AABBs and the
//! separating-axis overlap test). `RealAabb` is the textbook floating-point form;
//! `LatticeAabb` deviates by integer cells and the half-open `[min, max)` ownership
//! convention, so its `intersects` is a strict-inequality test on every axis and
//! touching faces do not overlap.

use glam::Vec3;

/// A half-open integer box `[min, max)`. A box owns the integer cells with
/// `min[axis] <= c < max[axis]` on every axis; it is empty when any
/// `min[axis] >= max[axis]`. Touching boxes are DISJOINT (see the module docs for
/// the contrast with the closed [`RealAabb`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatticeAabb {
    /// Inclusive minimum corner. `i64` so a far-flung box (a coordinate scaled up by
    /// a large factor) cannot silently truncate.
    pub min: [i64; 3],
    /// Exclusive maximum corner.
    pub max: [i64; 3],
}

impl LatticeAabb {
    /// A box spanning `[min, max)`.
    pub fn new(min: [i64; 3], max: [i64; 3]) -> Self {
        Self { min, max }
    }

    /// Whether the box is empty (owns no cell on some axis).
    pub fn is_empty(&self) -> bool {
        (0..3).any(|axis| self.min[axis] >= self.max[axis])
    }

    /// Whether two half-open boxes overlap (share at least one integer cell). Touching
    /// faces (one box's `max` equals the other's `min`) do **not** overlap — the
    /// half-open convention. This is the per-axis separating-axis test of Ericson 2005
    /// ch. 4, with strict inequalities because the boxes are half-open.
    pub fn intersects(&self, other: &LatticeAabb) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        (0..3).all(|axis| self.min[axis] < other.max[axis] && other.min[axis] < self.max[axis])
    }

    /// Whether this box fully CONTAINS `other` (every cell of `other` lies inside
    /// `self`). Half-open: `self.min <= other.min` and `other.max <= self.max` on every
    /// axis. An empty `other` is never contained (it owns no cell to be contained).
    pub fn contains_box(&self, other: &LatticeAabb) -> bool {
        if other.is_empty() || self.is_empty() {
            return false;
        }
        (0..3).all(|axis| self.min[axis] <= other.min[axis] && other.max[axis] <= self.max[axis])
    }

    /// The smallest box containing both inputs (an empty box contributes nothing).
    pub fn union(&self, other: &LatticeAabb) -> LatticeAabb {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        LatticeAabb {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }
}

/// A closed continuous box `[min, max]` with `f32` corners (inclusive min/max), in
/// render/world-space units. Touching boxes INTERSECT — the conservative bias a
/// continuous culling or overlap test needs (see the module docs for the contrast
/// with the half-open [`LatticeAabb`]). Consumed by the `camera` crate's frustum
/// culling test and by [`crate::spatial::Ray::intersect_box_slab`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealAabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl RealAabb {
    /// An empty/degenerate box that grows to fit points via [`RealAabb::expand`].
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_is_half_open() {
        let a = LatticeAabb::new([0, 0, 0], [10, 10, 10]);
        // Overlapping box.
        assert!(a.intersects(&LatticeAabb::new([5, 5, 5], [15, 15, 15])));
        // Touching faces (b.min == a.max) do NOT overlap (half-open).
        assert!(!a.intersects(&LatticeAabb::new([10, 0, 0], [20, 10, 10])));
        // Fully separate.
        assert!(!a.intersects(&LatticeAabb::new([100, 0, 0], [110, 10, 10])));
        // Empty box never intersects.
        assert!(!a.intersects(&LatticeAabb::new([0, 0, 0], [0, 0, 0])));
    }

    #[test]
    fn union_ignores_empty() {
        let empty = LatticeAabb::new([0, 0, 0], [0, 0, 0]);
        let b = LatticeAabb::new([3, 3, 3], [7, 7, 7]);
        assert_eq!(empty.union(&b), b);
        assert_eq!(b.union(&empty), b);
        let a = LatticeAabb::new([-2, 0, 0], [4, 4, 4]);
        assert_eq!(a.union(&b), LatticeAabb::new([-2, 0, 0], [7, 7, 7]));
    }

    #[test]
    fn contains_box_is_half_open() {
        let outer = LatticeAabb::new([0, 0, 0], [10, 10, 10]);
        assert!(outer.contains_box(&LatticeAabb::new([0, 0, 0], [10, 10, 10]))); // equal is contained
        assert!(outer.contains_box(&LatticeAabb::new([2, 2, 2], [8, 8, 8])));
        assert!(!outer.contains_box(&LatticeAabb::new([2, 2, 2], [11, 8, 8]))); // pokes out
        assert!(!outer.contains_box(&LatticeAabb::new([0, 0, 0], [0, 0, 0]))); // empty never contained
    }

    /// The empty sentinel grows to exactly bound the points folded into it, and the
    /// first expansion sets both corners (±∞ start).
    #[test]
    fn real_aabb_empty_expands_to_bound_points() {
        let mut aabb = RealAabb::empty();
        assert!(aabb.min.x.is_infinite() && aabb.max.x.is_infinite());
        aabb.expand(Vec3::new(1.0, -2.0, 3.0));
        assert_eq!(aabb.min, Vec3::new(1.0, -2.0, 3.0));
        assert_eq!(aabb.max, Vec3::new(1.0, -2.0, 3.0));
        aabb.expand(Vec3::new(-4.0, 5.0, 0.5));
        assert_eq!(aabb.min, Vec3::new(-4.0, -2.0, 0.5));
        assert_eq!(aabb.max, Vec3::new(1.0, 5.0, 3.0));
    }
}
