//! A half-open, integer axis-aligned bounding box.
//!
//! `Aabb` is the textbook axis-aligned bounding box (AABB) specialised to
//! **integer** coordinates under the **half-open** `[min, max)` convention: a box
//! owns the integer cells `min[axis] <= c < max[axis]` on every axis, and two boxes
//! that merely touch on a face (one's `max` equalling the other's `min`) do **not**
//! overlap. That half-open rule is what makes the boxes tile space without double-
//! counting the shared boundary cell — the same reason a grid of cells indexed by
//! `floor(position / extent)` partitions cleanly.
//!
//! The corners are `i64` so the box can address a large integer coordinate range
//! without overflow; nothing here is parameterised by the meaning of a coordinate.
//!
//! Cite: Ericson, *Real-Time Collision Detection* (2005), ch. 4 (AABBs and the
//! separating-axis overlap test). Deviation from the textbook floating-point box:
//! integer cells and the half-open `[min, max)` ownership convention, so `intersects`
//! is a strict-inequality test on every axis and touching faces do not overlap.

/// A half-open integer box `[min, max)`. A box owns the integer cells with
/// `min[axis] <= c < max[axis]` on every axis; it is empty when any
/// `min[axis] >= max[axis]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aabb {
    /// Inclusive minimum corner. `i64` so a far-flung box (a coordinate scaled up by
    /// a large factor) cannot silently truncate.
    pub min: [i64; 3],
    /// Exclusive maximum corner.
    pub max: [i64; 3],
}

impl Aabb {
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
    pub fn intersects(&self, other: &Aabb) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        (0..3).all(|axis| self.min[axis] < other.max[axis] && other.min[axis] < self.max[axis])
    }

    /// Whether this box fully CONTAINS `other` (every cell of `other` lies inside
    /// `self`). Half-open: `self.min <= other.min` and `other.max <= self.max` on every
    /// axis. An empty `other` is never contained (it owns no cell to be contained).
    pub fn contains_box(&self, other: &Aabb) -> bool {
        if other.is_empty() || self.is_empty() {
            return false;
        }
        (0..3).all(|axis| self.min[axis] <= other.min[axis] && other.max[axis] <= self.max[axis])
    }

    /// The smallest box containing both inputs (an empty box contributes nothing).
    pub fn union(&self, other: &Aabb) -> Aabb {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Aabb {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersects_is_half_open() {
        let a = Aabb::new([0, 0, 0], [10, 10, 10]);
        // Overlapping box.
        assert!(a.intersects(&Aabb::new([5, 5, 5], [15, 15, 15])));
        // Touching faces (b.min == a.max) do NOT overlap (half-open).
        assert!(!a.intersects(&Aabb::new([10, 0, 0], [20, 10, 10])));
        // Fully separate.
        assert!(!a.intersects(&Aabb::new([100, 0, 0], [110, 10, 10])));
        // Empty box never intersects.
        assert!(!a.intersects(&Aabb::new([0, 0, 0], [0, 0, 0])));
    }

    #[test]
    fn union_ignores_empty() {
        let empty = Aabb::new([0, 0, 0], [0, 0, 0]);
        let b = Aabb::new([3, 3, 3], [7, 7, 7]);
        assert_eq!(empty.union(&b), b);
        assert_eq!(b.union(&empty), b);
        let a = Aabb::new([-2, 0, 0], [4, 4, 4]);
        assert_eq!(a.union(&b), Aabb::new([-2, 0, 0], [7, 7, 7]));
    }

    #[test]
    fn contains_box_is_half_open() {
        let outer = Aabb::new([0, 0, 0], [10, 10, 10]);
        assert!(outer.contains_box(&Aabb::new([0, 0, 0], [10, 10, 10]))); // equal is contained
        assert!(outer.contains_box(&Aabb::new([2, 2, 2], [8, 8, 8])));
        assert!(!outer.contains_box(&Aabb::new([2, 2, 2], [11, 8, 8]))); // pokes out
        assert!(!outer.contains_box(&Aabb::new([0, 0, 0], [0, 0, 0]))); // empty never contained
    }
}
