//! Continuous leaf placement (ADR 0027): the world↔producer-local affine that a rotated,
//! corner-anchored leaf folds through. **ONE definition** shared by every voxel sink — the dense
//! reference oracle (`document`), the two-layer classifier (`evaluation`) and the brick field — so
//! they can never disagree on where a producer's cells land. Splitting this into a per-sink
//! reimplementation is what let the dense path silently drop the rotation (it lived below the
//! evaluation-layer `LeafAffine` it needed and only ever had a translation); hoisting the math to
//! substrate lets both layers construct the identical placement.
//!
//! **Corner-anchor convention.** A producer emits its cells in its LOCAL box `[0, full]`. The
//! placement rotates that box and RE-ANCHORS its lowest rotated corner back onto `world_offset`,
//! so `world_of(min_corner) == world_offset` exactly: a leaf occupies
//! `[world_offset, world_offset + span_of_rotated_box)`. This is the same anchor
//! [`seat_centre_at`] inverts when it seats a producer's centre onto a surface contact.

use crate::spatial::voxel_frames::{ProducerLocalVoxelPoint, TrueWorldVoxelPoint};
use glam::{Quat, Vec3};

/// The 8 corners of the local box `[0, full]`, in a fixed order.
pub fn box_corners(full: Vec3) -> [Vec3; 8] {
    [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(full.x, 0.0, 0.0),
        Vec3::new(0.0, full.y, 0.0),
        Vec3::new(full.x, full.y, 0.0),
        Vec3::new(0.0, 0.0, full.z),
        Vec3::new(full.x, 0.0, full.z),
        Vec3::new(0.0, full.y, full.z),
        Vec3::new(full.x, full.y, full.z),
    ]
}

/// Round each component to the nearest integer voxel index — used where the affine is KNOWN to
/// land on integer coordinates (an axis-aligned leaf) so float round-off never grows a box by a
/// voxel.
/// Snap-to-integer tolerance for the conservative box-bound helpers below: a corner within this of
/// an integer is treated as landing exactly on it (absorbs `world_of`/`local_of` float round-off).
const CORNER_SNAP_TOLERANCE: f32 = 1e-3;

/// The integer `[min, max)` box conservatively enclosing the real interval `[low, high]` per axis:
/// a coordinate within [`CORNER_SNAP_TOLERANCE`] of an integer snaps to it (so an integer-landing
/// box is recovered EXACTLY — bit-identical to round-to-nearest, every golden holds); otherwise the
/// min FLOORS and the max CEILS so the box never sheds a boundary voxel. Unlike a plain round, a
/// half-integer box (an axis-aligned leaf under an ADR 0027 sub-voxel seat) WIDENS rather than
/// rounding-to-nearest, which would drop the boundary chunk/voxel on the shrunk side.
fn conservative_box(low: Vec3, high: Vec3) -> ([i64; 3], [i64; 3]) {
    let snap_floor = |value: f32| {
        let nearest = value.round();
        if (value - nearest).abs() < CORNER_SNAP_TOLERANCE {
            nearest as i64
        } else {
            value.floor() as i64
        }
    };
    let snap_ceil = |value: f32| {
        let nearest = value.round();
        if (value - nearest).abs() < CORNER_SNAP_TOLERANCE {
            nearest as i64
        } else {
            value.ceil() as i64
        }
    };
    (
        [snap_floor(low.x), snap_floor(low.y), snap_floor(low.z)],
        [snap_ceil(high.x), snap_ceil(high.y), snap_ceil(high.z)],
    )
}

/// Whether `rotation` is one of the 24 axis-aligned lattice turns (to a `1e-4` tolerance): each of
/// `rotation · {X, Y, Z}` lands on a signed unit axis (exactly one component `≈ ±1`, the other two
/// `≈ 0`). An axis-aligned leaf takes the EXACT integer paths (byte-identical to the ADR 0026
/// permutation — the whole existing golden suite); a genuinely-rotated one resamples (ADR 0027 §4).
pub fn is_axis_aligned(rotation: Quat) -> bool {
    const TOLERANCE: f32 = 1e-4;
    [Vec3::X, Vec3::Y, Vec3::Z].into_iter().all(|axis| {
        let image = rotation * axis;
        let near_unit = image
            .to_array()
            .iter()
            .filter(|component| (component.abs() - 1.0).abs() <= TOLERANCE)
            .count();
        let near_zero =
            image.to_array().iter().filter(|component| component.abs() <= TOLERANCE).count();
        near_unit == 1 && near_zero == 2
    })
}

/// Whether a placed leaf is **in phase** with the absolute voxel lattice (ADR 0027): an
/// axis-aligned rotation AND a whole-voxel `offset_local` (no sub-voxel slide). An in-phase leaf
/// emits one-cell-per-absolute-cell by a pure translation — the byte-identical ADR 0026
/// forward-emit path; an out-of-phase one (a genuine rotation OR a fractional `offset_local`) must
/// be inverse-resampled by gather. The dense oracle (`document`) and the live two-layer classifier
/// (`evaluation`) MUST agree on this split or the two paths disagree on rotated / sub-voxel seats,
/// so the predicate lives beside [`is_axis_aligned`] as the ONE definition both fold through.
pub fn is_in_phase(rotation: Quat, offset_local_voxels: [f32; 3]) -> bool {
    is_axis_aligned(rotation) && offset_local_voxels.iter().all(|slide| slide.fract() == 0.0)
}

/// The corner-anchored world↔producer-local affine of a placed leaf (ADR 0027). Pure `glam`
/// arithmetic over the leaf's `rotation`, local `full` extent, and a **wandering-origin** world
/// offset split into an integer [`origin_voxels`](Self::origin_voxels) and a fractional
/// `offset_local`; construct it via [`from_origin_and_local`](Self::from_origin_and_local) (the
/// precision-preserving door) or [`new`](Self::new) (a combined-`f32` convenience).
///
/// **Wandering origin (ADR 0027 §1 / ADR 0008).** The integer origin is carried as `i64` and every
/// large translation is done in `i64` (exact), so the rotation math only ever touches
/// origin-*relative* coordinates (magnitude `≈ full`). A leaf placed millions of voxels from the
/// world origin therefore keeps full sub-voxel precision — collapsing the origin into an `f32` up
/// front (the old field) silently lost it, the `(c)`-re-derive failure ADR 0008 forbids.
#[derive(Clone, Copy, Debug)]
pub struct LeafPlacement {
    rotation: Quat,
    /// The lowest of the 8 rotated local corners — the re-anchoring term that lands the box's low
    /// corner exactly on the world offset.
    min_rotated_corner: Vec3,
    /// The INTEGER part of the leaf's world offset (`world_offset_voxels`), in absolute voxels —
    /// the wandering i64 origin kept exact so a far-out translation never downcasts through `f32`.
    origin_voxels: [i64; 3],
    /// The FRACTIONAL sub-voxel slide on top of `origin_voxels` (`offset_local_voxels`, typically
    /// in `[0, 1)`), in voxels. The full world offset is `origin_voxels + offset_local`.
    offset_local: Vec3,
    /// The producer's local box extent `full`, in voxels.
    full: Vec3,
}

impl LeafPlacement {
    /// Build the placement from a leaf's `rotation`, local `full` extent, integer
    /// `origin_voxels` (the wandering i64 origin — a node's `world_offset_voxels`) and fractional
    /// `offset_local` (its `offset_local_voxels` sub-voxel slide). This is the precision-preserving
    /// door: the integer origin stays `i64`, so a leaf arbitrarily far from the world origin never
    /// loses sub-voxel precision (ADR 0027 §1). `min_rotated_corner` is derived so the low rotated
    /// corner anchors on the world offset.
    pub fn from_origin_and_local(
        rotation: Quat,
        full: Vec3,
        origin_voxels: [i64; 3],
        offset_local: [f32; 3],
    ) -> Self {
        Self {
            rotation,
            min_rotated_corner: min_rotated_corner(rotation, full),
            origin_voxels,
            offset_local: Vec3::from_array(offset_local),
            full,
        }
    }

    /// Build the placement from a combined-`f32` `world_offset` (a [`TrueWorldVoxelPoint`]) — the
    /// convenience door for the near-frame / zero-origin callers (the ghost preview, the
    /// span-at-origin extent helpers) where the offset is small and precision is not at stake. The
    /// combined offset is split into its integer floor (the wandering origin) and fractional
    /// remainder; a far-out caller should use [`from_origin_and_local`](Self::from_origin_and_local)
    /// with the offset already split so no precision is lost in the split.
    pub fn new(rotation: Quat, full: Vec3, world_offset: TrueWorldVoxelPoint) -> Self {
        let offset = world_offset.voxels();
        let floor = offset.floor();
        Self::from_origin_and_local(
            rotation,
            full,
            [floor.x as i64, floor.y as i64, floor.z as i64],
            (offset - floor).to_array(),
        )
    }

    /// The leaf's rotation.
    pub fn rotation(&self) -> Quat {
        self.rotation
    }

    /// The producer's local `full` extent, in voxels.
    pub fn full(&self) -> Vec3 {
        self.full
    }

    /// The wandering integer origin (`world_offset_voxels`), in absolute voxels.
    pub fn origin_voxels(&self) -> [i64; 3] {
        self.origin_voxels
    }

    /// The integer origin as an `f32` vector — the single downcast, used only at the near-frame
    /// point crossings ([`world_of`](Self::world_of) / [`local_of`](Self::local_of)) whose result
    /// or input a caller has already accepted as small. The precision-critical integer paths
    /// (`world_aabb`, `world_cell_of_local_centre`, …) add the origin in `i64` instead.
    fn origin_vec3(&self) -> Vec3 {
        Vec3::new(
            self.origin_voxels[0] as f32,
            self.origin_voxels[1] as f32,
            self.origin_voxels[2] as f32,
        )
    }

    /// The forward affine in the ORIGIN-RELATIVE frame: a producer-local point mapped to its world
    /// position **minus** `origin_voxels` (`rotation·local − min_rotated_corner + offset_local`).
    /// Magnitude `≈ full`, so it never loses precision however far out the leaf sits; the integer
    /// origin is re-added afterwards (in `i64` where the result must stay exact).
    fn forward_local_relative(&self, local: Vec3) -> Vec3 {
        self.rotation * local - self.min_rotated_corner + self.offset_local
    }

    /// The inverse affine from an ORIGIN-RELATIVE world position (`world − origin_voxels`) back to
    /// the producer-local frame. The caller rebases the world point against `origin_voxels` in
    /// `i64` first (exact), so this only ever rotates a small residual.
    fn local_of_relative(&self, world_relative: Vec3) -> Vec3 {
        self.rotation.inverse() * (world_relative - self.offset_local + self.min_rotated_corner)
    }

    /// A [`ProducerLocalVoxelPoint`] mapped to its [`TrueWorldVoxelPoint`] — the frame types make a
    /// producer-local/true-world mix-up a compile error. Near-frame convenience (the origin is
    /// re-added through `f32`); the precision-critical integer twin is
    /// [`world_cell_of_local_centre`](Self::world_cell_of_local_centre).
    pub fn world_of(&self, local: ProducerLocalVoxelPoint) -> TrueWorldVoxelPoint {
        TrueWorldVoxelPoint::from_voxels(self.forward_local_relative(local.voxels()) + self.origin_vec3())
    }

    /// The inverse: a [`TrueWorldVoxelPoint`] mapped back to the producer-LOCAL frame.
    /// `local_of(world_of(p)) ≈ p` for every `p` (a rotation's inverse is exact up to float
    /// round-off, which the classifier's `+0.5` centre-sample margins absorb). Near-frame
    /// convenience; the precision-critical twin is
    /// [`local_of_abs_cell_centre`](Self::local_of_abs_cell_centre).
    pub fn local_of(&self, world: TrueWorldVoxelPoint) -> ProducerLocalVoxelPoint {
        ProducerLocalVoxelPoint::from_voxels(self.local_of_relative(world.voxels() - self.origin_vec3()))
    }

    /// The absolute voxel cell a producer-LOCAL cell's CENTRE lands in — `world_of(index + 0.5)`
    /// floored, computed with the integer origin re-added in `i64` so it stays exact arbitrarily
    /// far from the world origin (the wandering-origin fold). Bit-identical to the near-frame
    /// `world_of(index + 0.5).floor()` for a small origin, and the precision-preserving replacement
    /// for it everywhere a far-out leaf's cells are emitted.
    pub fn world_cell_of_local_centre(&self, local_index: [i32; 3]) -> [i64; 3] {
        let centre = Vec3::new(
            local_index[0] as f32 + 0.5,
            local_index[1] as f32 + 0.5,
            local_index[2] as f32 + 0.5,
        );
        let relative = self.forward_local_relative(centre);
        [
            self.origin_voxels[0] + relative.x.floor() as i64,
            self.origin_voxels[1] + relative.y.floor() as i64,
            self.origin_voxels[2] + relative.z.floor() as i64,
        ]
    }

    /// The producer-LOCAL coordinate of an ABSOLUTE voxel cell's CENTRE (`abs_cell + 0.5`),
    /// rebasing the cell against the integer origin in `i64` first (exact), so the inverse affine
    /// only rotates a small residual and keeps full precision however far out the leaf sits. The
    /// precision-preserving replacement for `local_of((abs_cell + 0.5).into())` at the resample
    /// gather sites.
    pub fn local_of_abs_cell_centre(&self, abs_cell: [i64; 3]) -> ProducerLocalVoxelPoint {
        let relative = Vec3::new(
            (abs_cell[0] - self.origin_voxels[0]) as f32 + 0.5,
            (abs_cell[1] - self.origin_voxels[1]) as f32 + 0.5,
            (abs_cell[2] - self.origin_voxels[2]) as f32 + 0.5,
        );
        ProducerLocalVoxelPoint::from_voxels(self.local_of_relative(relative))
    }

    /// The integer world AABB `[min, max)` (in absolute voxels) enclosing the placed box — the ONE
    /// extent every sink and the coverage/broadphase walk must agree on. For an AXIS-ALIGNED
    /// rotation with a whole-voxel offset the corners land on integers, recovered exactly (bit-
    /// identical to the pre-0027 `turn_extent` permutation, so every golden holds); an axis-aligned
    /// leaf under an ADR 0027 sub-voxel seat lands on half-integers and WIDENS (floor-min/ceil-max)
    /// so the fractional side never sheds its boundary chunk; a genuine rotation likewise floors the
    /// min and ceils the max to conservatively enclose the rotated box (SOUND: the true occupied set
    /// ⊆ this AABB, ADR 0027 §4).
    ///
    /// The enclosing box is computed in the ORIGIN-RELATIVE frame and the integer origin is added
    /// back in `i64`, so a far-out leaf's extent stays exact (the wandering-origin fold).
    pub fn world_aabb(&self) -> ([i64; 3], [i64; 3]) {
        let (rel_min, rel_max) = enclosing_box(self.rotation, box_corners(self.full), |corner| {
            self.forward_local_relative(corner)
        });
        (
            [
                rel_min[0] + self.origin_voxels[0],
                rel_min[1] + self.origin_voxels[1],
                rel_min[2] + self.origin_voxels[2],
            ],
            [
                rel_max[0] + self.origin_voxels[0],
                rel_max[1] + self.origin_voxels[1],
                rel_max[2] + self.origin_voxels[2],
            ],
        )
    }

    /// The producer-LOCAL integer box `[min, max)` enclosing the inverse image of an absolute voxel
    /// box `[abs_min, abs_max)` — the frame `resolve_into` / `cell_field_interval` expect (the
    /// producer never learns the leaf is turned). The inverse of [`world_aabb`](Self::world_aabb):
    /// a whole-phase axis-aligned leaf recovers integers exactly (bit-identical to the pre-0027
    /// unturn); a sub-voxel-seated or genuinely-rotated one floors the min and ceils the max to
    /// conservatively enclose the preimage (SOUND — the isometry keeps the cell radius invariant,
    /// ADR 0027 §4; the box may fall partly outside `[0, full]`, which the producer bounds/clamps
    /// exactly as before).
    ///
    /// Each absolute corner is rebased against the integer origin in `i64` before the rotation, so
    /// a far-out box maps to producer-local without precision loss (the output is producer-local,
    /// magnitude `≈ full`, so no origin is re-added).
    pub fn local_aabb(&self, abs_min: [i64; 3], abs_max: [i64; 3]) -> ([i64; 3], [i64; 3]) {
        // The 8 integer corners of `[abs_min, abs_max]`, rebased against the origin in i64 (exact)
        // then rotated into the local frame — the same enclosing floor/ceil as `world_aabb`.
        let corners: [Vec3; 8] = std::array::from_fn(|i| {
            let pick = |axis: usize| if (i >> axis) & 1 == 0 { abs_min[axis] } else { abs_max[axis] };
            self.local_of_relative(Vec3::new(
                (pick(0) - self.origin_voxels[0]) as f32,
                (pick(1) - self.origin_voxels[1]) as f32,
                (pick(2) - self.origin_voxels[2]) as f32,
            ))
        });
        // `enclosing_box`'s transform is identity here — the corners are already the local images.
        enclosing_box(self.rotation, corners, |mapped| mapped)
    }
}

/// The low corner of the box `[0, full]` after `rotation` — the anchoring term the
/// whole corner-anchor convention depends on (module docs call it load-bearing), so
/// [`LeafPlacement::new`] and [`seat_centre_at`] read it from ONE definition rather
/// than each re-running the fold.
fn min_rotated_corner(rotation: Quat, full: Vec3) -> Vec3 {
    let mut low = Vec3::splat(f32::INFINITY);
    for corner in box_corners(full) {
        low = low.min(rotation * corner);
    }
    low
}

/// Fold the eight `corners` through `transform`, then snap the enclosing float box to
/// an integer `[min, max)`: an axis-aligned `rotation` recovers exact integers (a
/// whole-phase leaf) or half-integer widens via [`conservative_box`]; a genuine
/// rotation floors the min and ceils the max to conservatively enclose the box (SOUND,
/// ADR 0027 §4). The shared skeleton of [`LeafPlacement::world_aabb`] (forward
/// transform) and [`LeafPlacement::local_aabb`] (inverse) — the floor/ceil-vs-
/// `conservative_box` dispatch most at risk of silent drift now lives once.
fn enclosing_box(
    rotation: Quat,
    corners: [Vec3; 8],
    transform: impl Fn(Vec3) -> Vec3,
) -> ([i64; 3], [i64; 3]) {
    let mut low = Vec3::splat(f32::INFINITY);
    let mut high = Vec3::splat(f32::NEG_INFINITY);
    for corner in corners {
        let mapped = transform(corner);
        low = low.min(mapped);
        high = high.max(mapped);
    }
    if is_axis_aligned(rotation) {
        conservative_box(low, high)
    } else {
        (
            [low.x.floor() as i64, low.y.floor() as i64, low.z.floor() as i64],
            [high.x.ceil() as i64, high.y.ceil() as i64, high.z.ceil() as i64],
        )
    }
}

/// The world offset (in ABSOLUTE voxels) that seats a producer of local dimensions `full`, rotated
/// by `rotation`, so its local CENTRE `full/2` lands at world `target_centre` under the SAME
/// corner-anchored [`LeafPlacement`] the classifier folds through (ADR 0027 §5 placement). It is
/// the inverse of [`LeafPlacement::new`]`(rotation, full, result).world_of(full/2) == target_centre`.
pub fn seat_centre_at(rotation: Quat, full: Vec3, target_centre: Vec3) -> Vec3 {
    target_centre - rotation * (full * 0.5) + min_rotated_corner(rotation, full)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A voxel offset (`> 2^24`) at which the OLD collapse of the integer origin into an `f32`
    /// world offset loses whole voxels of precision — the wandering-origin fold must survive it.
    const FAR: i64 = 100_000_000;

    /// The wandering-origin fold (ADR 0027 §1): mapping an absolute cell back to producer-local is
    /// TRANSLATION-INVARIANT — a leaf placed `FAR` from the world origin resolves the identical
    /// local coordinate as the same leaf at the origin. The pre-fold path (integer origin + fraction
    /// collapsed to one `f32`) failed this: `abs_centre − world_offset` cancelled catastrophically
    /// far out, so a placed body drifted / fragmented past ~16M voxels.
    #[test]
    fn local_of_abs_cell_is_translation_invariant_arbitrarily_far_out() {
        let rotation = Quat::from_rotation_z(0.37) * Quat::from_rotation_x(0.11);
        let full = Vec3::new(3.0, 5.0, 2.0);
        let offset_local = [0.25, 0.5, 0.75];

        let near = LeafPlacement::from_origin_and_local(rotation, full, [0, 0, 0], offset_local);
        let far = LeafPlacement::from_origin_and_local(rotation, full, [FAR, -FAR, FAR], offset_local);

        // The same cell, expressed relative to each leaf's own origin, must map to the SAME local
        // point (the affine only ever sees the small residual).
        for cell in [[1, 2, 0], [7, -3, 4], [0, 0, 0]] {
            let near_local = near.local_of_abs_cell_centre(cell).voxels();
            let far_cell = [cell[0] + FAR, cell[1] - FAR, cell[2] + FAR];
            let far_local = far.local_of_abs_cell_centre(far_cell).voxels();
            assert!(
                (near_local - far_local).length() < 1e-4,
                "far-out local {far_local:?} drifted from near local {near_local:?}"
            );
        }
    }

    /// The forward twin: the absolute cell a producer-local centre lands in tracks the integer
    /// origin EXACTLY, however far out — `world_cell_of_local_centre(idx)` at `FAR` equals the near
    /// answer shifted by the origin, to the voxel.
    #[test]
    fn world_cell_tracks_the_integer_origin_exactly_far_out() {
        let rotation = Quat::from_rotation_z(0.6);
        let full = Vec3::new(4.0, 4.0, 4.0);
        let offset_local = [0.3, 0.1, 0.0];

        let near = LeafPlacement::from_origin_and_local(rotation, full, [0, 0, 0], offset_local);
        let far = LeafPlacement::from_origin_and_local(rotation, full, [FAR, FAR, -FAR], offset_local);

        for idx in [[0, 0, 0], [2, 1, 3], [3, 3, 3]] {
            let near_cell = near.world_cell_of_local_centre(idx);
            let far_cell = far.world_cell_of_local_centre(idx);
            assert_eq!(
                far_cell,
                [near_cell[0] + FAR, near_cell[1] + FAR, near_cell[2] - FAR],
                "world cell for local {idx:?} did not track the integer origin exactly"
            );
            // And the world AABB shifts by exactly the origin (no f32 rounding of the extent).
        }
        let (near_min, near_max) = near.world_aabb();
        let (far_min, far_max) = far.world_aabb();
        assert_eq!(far_min, [near_min[0] + FAR, near_min[1] + FAR, near_min[2] - FAR]);
        assert_eq!(far_max, [near_max[0] + FAR, near_max[1] + FAR, near_max[2] - FAR]);
    }

    /// The split constructor and the combined-`f32` [`LeafPlacement::new`] agree bit-for-bit at a
    /// near/zero origin, so the convenience door (ghost, extent-at-origin) is unchanged by the fold.
    #[test]
    fn new_and_split_agree_near_the_origin() {
        let rotation = Quat::from_rotation_y(0.9);
        let full = Vec3::new(6.0, 2.0, 3.0);
        let origin = [12, -4, 7];
        let offset_local = [0.5, 0.25, 0.0];

        let split = LeafPlacement::from_origin_and_local(rotation, full, origin, offset_local);
        let combined = Vec3::new(
            origin[0] as f32 + offset_local[0],
            origin[1] as f32 + offset_local[1],
            origin[2] as f32 + offset_local[2],
        );
        let via_new = LeafPlacement::new(rotation, full, TrueWorldVoxelPoint::from_voxels(combined));

        assert_eq!(split.world_aabb(), via_new.world_aabb());
        for idx in [[0, 0, 0], [3, 1, 2]] {
            assert_eq!(
                split.world_cell_of_local_centre(idx),
                via_new.world_cell_of_local_centre(idx)
            );
        }
    }
}
