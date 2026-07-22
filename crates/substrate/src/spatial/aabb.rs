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

/// The whole-**block** box that tightly ENCLOSES a voxel span, given as
/// `(low_block_corner, high_block_corner)` in block units. The low corner FLOORS
/// `voxel_min` to its block and the high corner CEILS `voxel_max` to its block, each
/// axis handled INDEPENDENTLY.
///
/// This exists to prevent a recurring off-by-a-block bug: computing an enclosing box
/// as `low = floor(voxel_min / density)` then `high = low + block_size` CLIPS the
/// high side whenever the span is not block-aligned, because the high corner must be
/// ceiled on its own — an off-block span touches one more block than its block size.
/// Rounding both corners outward here means the returned block box, scaled back to
/// voxels as `[low·density, high·density)`, ALWAYS contains `[voxel_min, voxel_max)`.
/// A block-aligned span has no remainder, so `high == low + (voxel_max − voxel_min) /
/// density` exactly.
///
/// `voxels_per_block` is clamped to `>= 1` so a zero/negative density cannot divide by
/// zero. The floor uses `div_euclid` (rounds toward −∞); the ceil uses the identity
/// `ceil(a / d) == −((−a).div_euclid(d))` for a positive divisor, because signed
/// `i64::div_ceil` is still unstable on this toolchain.
///
/// Both corners are `[i64; 3]` so a far-flung span (a coordinate scaled up by a large
/// density) cannot silently truncate.
pub fn enclosing_block_aabb(
    voxel_min: [i64; 3],
    voxel_max: [i64; 3],
    voxels_per_block: i64,
) -> ([i64; 3], [i64; 3]) {
    let density = voxels_per_block.max(1);
    let low_block_corner = std::array::from_fn(|axis| voxel_min[axis].div_euclid(density));
    let high_block_corner = std::array::from_fn(|axis| -((-voxel_max[axis]).div_euclid(density)));
    (low_block_corner, high_block_corner)
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

/// Machine-checked proofs of the [`enclosing_block_aabb`] containment + tightness
/// contract. `#[cfg(kani)]` keeps them inactive in ordinary builds. Run under WSL:
/// `cargo kani -p substrate`.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A symbolic voxel coordinate confined to a bounded range so the model checker
    /// terminates and the `·density` products below cannot overflow `i64` (the
    /// production callers never scale near the type's edge). Wide enough to exercise
    /// both signs and non-block-aligned remainders.
    fn voxel_coordinate_in_range() -> [i64; 3] {
        let x: i64 = kani::any();
        let y: i64 = kani::any();
        let z: i64 = kani::any();
        kani::assume(x >= -1_000_000 && x <= 1_000_000);
        kani::assume(y >= -1_000_000 && y <= 1_000_000);
        kani::assume(z >= -1_000_000 && z <= 1_000_000);
        [x, y, z]
    }

    /// **The enclosing block box tightly contains the voxel span.** For every bounded
    /// voxel span and every valid density, the returned whole-block corners scaled back
    /// to voxels (`[low·d, high·d)`) contain `[voxel_min, voxel_max)`, AND each corner is
    /// the tightest whole block: `low·d <= voxel_min < (low+1)·d` and `(high−1)·d <
    /// voxel_max <= high·d`. The tightness clauses are what rule out both the clip bug
    /// (an over-tight high corner) and a needlessly loose cage.
    #[kani::proof]
    fn enclosing_block_box_tightly_contains_the_voxel_span() {
        let voxel_min = voxel_coordinate_in_range();
        let voxel_max = voxel_coordinate_in_range();
        let density: i64 = kani::any();
        kani::assume(density >= 1 && density <= 64);
        let (low_block_corner, high_block_corner) =
            enclosing_block_aabb(voxel_min, voxel_max, density);
        for axis in 0..3 {
            let low = low_block_corner[axis];
            let high = high_block_corner[axis];
            // Containment: the block box scaled to voxels spans the voxel span.
            assert!(low * density <= voxel_min[axis]);
            assert!(voxel_max[axis] <= high * density);
            // Tightness of the FLOOR: no larger block starts at or below voxel_min.
            assert!(voxel_min[axis] < (low + 1) * density);
            // Tightness of the CEIL: no smaller block reaches voxel_max.
            assert!((high - 1) * density < voxel_max[axis]);
        }
    }

    /// **A block-aligned span keeps `high == low + block_size` exactly.** When both
    /// corners are exact block multiples the remainder vanishes, so the enclosing box is
    /// the naive `low + size` box — proving the primitive does not perturb the aligned
    /// case the goldens pin (it only widens the OFF-block case).
    #[kani::proof]
    fn a_block_aligned_span_has_no_ceil_slack() {
        let low_block: [i64; 3] = voxel_coordinate_in_range();
        let size_blocks: [i64; 3] = {
            let sx: i64 = kani::any();
            let sy: i64 = kani::any();
            let sz: i64 = kani::any();
            kani::assume(sx >= 0 && sx <= 1_000);
            kani::assume(sy >= 0 && sy <= 1_000);
            kani::assume(sz >= 0 && sz <= 1_000);
            [sx, sy, sz]
        };
        let density: i64 = kani::any();
        kani::assume(density >= 1 && density <= 64);
        let voxel_min = std::array::from_fn(|axis| low_block[axis] * density);
        let voxel_max = std::array::from_fn(|axis| (low_block[axis] + size_blocks[axis]) * density);
        let (low_block_corner, high_block_corner) =
            enclosing_block_aabb(voxel_min, voxel_max, density);
        for axis in 0..3 {
            assert!(low_block_corner[axis] == low_block[axis]);
            assert!(high_block_corner[axis] == low_block[axis] + size_blocks[axis]);
        }
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

    /// An OFF-block span must ceil its high corner independently — the naive `low +
    /// size` box would clip the high side. A 1-voxel translate that crosses a block
    /// boundary grows the enclosing box by exactly one whole block.
    #[test]
    fn enclosing_block_box_ceils_the_off_block_high_corner() {
        // A 2-block span (density 4 → 8 voxels) shifted 1 voxel off the block lattice.
        let density = 4;
        let (low, high) = enclosing_block_aabb([1, 0, 0], [9, 8, 8], density);
        // Low floors 1→0; high ceils 9→3 (touches 3 blocks, not 2), 8→2 (aligned).
        assert_eq!(low, [0, 0, 0]);
        assert_eq!(high, [3, 2, 2]);
        // The block box scaled back to voxels fully contains the span.
        assert!(low[0] * density <= 1 && 9 <= high[0] * density);
    }

    /// A block-aligned span has no remainder, so the enclosing box is the naive `low +
    /// size` box (the goldens' invariant).
    #[test]
    fn enclosing_block_box_is_exact_when_aligned() {
        let (low, high) = enclosing_block_aabb([8, -12, 0], [24, 0, 16], 4);
        assert_eq!(low, [2, -3, 0]);
        assert_eq!(high, [6, 0, 4]);
    }

    /// A zero/negative density is clamped to 1, never dividing by zero.
    #[test]
    fn enclosing_block_box_clamps_nonpositive_density() {
        let (low, high) = enclosing_block_aabb([3, 0, 0], [7, 0, 0], 0);
        assert_eq!(low, [3, 0, 0]);
        assert_eq!(high, [7, 0, 0]);
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
