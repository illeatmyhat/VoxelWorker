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
fn round_to_i64(vector: Vec3) -> [i64; 3] {
    [vector.x.round() as i64, vector.y.round() as i64, vector.z.round() as i64]
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

/// The corner-anchored world↔producer-local affine of a placed leaf (ADR 0027). Pure `glam`
/// arithmetic over the leaf's `rotation`, local `full` extent, and integer-plus-fraction
/// `world_offset`; construct it via [`new`](Self::new) from either layer.
#[derive(Clone, Copy, Debug)]
pub struct LeafPlacement {
    rotation: Quat,
    /// The lowest of the 8 rotated local corners — the re-anchoring term that lands the box's low
    /// corner exactly on `world_offset`.
    min_rotated_corner: Vec3,
    /// The leaf's continuous world offset (integer `world_offset_voxels` plus the float
    /// `offset_local_voxels`), in absolute voxels.
    world_offset: Vec3,
    /// The producer's local box extent `full`, in voxels.
    full: Vec3,
}

impl LeafPlacement {
    /// Build the placement from a leaf's `rotation`, local `full` extent and continuous
    /// `world_offset` (all in the absolute voxel frame). `min_rotated_corner` is derived so the
    /// low rotated corner anchors on `world_offset`.
    pub fn new(rotation: Quat, full: Vec3, world_offset: Vec3) -> Self {
        let mut min_rotated_corner = Vec3::splat(f32::INFINITY);
        for corner in box_corners(full) {
            min_rotated_corner = min_rotated_corner.min(rotation * corner);
        }
        Self { rotation, min_rotated_corner, world_offset, full }
    }

    /// The leaf's rotation.
    pub fn rotation(&self) -> Quat {
        self.rotation
    }

    /// The producer's local `full` extent, in voxels.
    pub fn full(&self) -> Vec3 {
        self.full
    }

    /// A producer-LOCAL voxel coordinate mapped to its ABSOLUTE voxel coordinate.
    pub fn world_of(&self, local: Vec3) -> Vec3 {
        self.rotation * local - self.min_rotated_corner + self.world_offset
    }

    /// The inverse: an ABSOLUTE voxel coordinate mapped back to the producer-LOCAL frame.
    /// `local_of(world_of(p)) ≈ p` for every `p` (a rotation's inverse is exact up to float
    /// round-off, which the classifier's `+0.5` centre-sample margins absorb).
    pub fn local_of(&self, world: Vec3) -> Vec3 {
        self.rotation.inverse() * (world - self.world_offset + self.min_rotated_corner)
    }

    /// The integer world AABB `[min, max)` (in absolute voxels) enclosing the placed box — the ONE
    /// extent every sink and the coverage/broadphase walk must agree on. For an AXIS-ALIGNED
    /// rotation the corners land on integers, recovered by rounding (bit-identical to the pre-0027
    /// `turn_extent` permutation, so every golden holds); for a genuine rotation the min is FLOORED
    /// and the max CEILED to conservatively enclose the rotated box (SOUND: the true occupied set ⊆
    /// this AABB, ADR 0027 §4).
    pub fn world_aabb(&self) -> ([i64; 3], [i64; 3]) {
        let mut world_min = Vec3::splat(f32::INFINITY);
        let mut world_max = Vec3::splat(f32::NEG_INFINITY);
        for corner in box_corners(self.full) {
            let world = self.world_of(corner);
            world_min = world_min.min(world);
            world_max = world_max.max(world);
        }
        if is_axis_aligned(self.rotation) {
            (round_to_i64(world_min), round_to_i64(world_max))
        } else {
            (
                [world_min.x.floor() as i64, world_min.y.floor() as i64, world_min.z.floor() as i64],
                [world_max.x.ceil() as i64, world_max.y.ceil() as i64, world_max.z.ceil() as i64],
            )
        }
    }

    /// The producer-LOCAL integer box `[min, max)` enclosing the inverse image of an absolute voxel
    /// box `[abs_min, abs_max)` — the frame `resolve_into` / `cell_field_interval` expect (the
    /// producer never learns the leaf is turned). The inverse of [`world_aabb`](Self::world_aabb):
    /// axis-aligned rounds (bit-identical to the pre-0027 unturn); a genuine rotation floors the
    /// min and ceils the max to conservatively enclose the rotated preimage (SOUND — the isometry
    /// keeps the cell radius invariant, ADR 0027 §4; the box may fall partly outside `[0, full]`,
    /// which the producer bounds/clamps exactly as before).
    pub fn local_aabb(&self, abs_min: [i64; 3], abs_max: [i64; 3]) -> ([i64; 3], [i64; 3]) {
        let abs_origin = Vec3::new(abs_min[0] as f32, abs_min[1] as f32, abs_min[2] as f32);
        let abs_full = Vec3::new(
            (abs_max[0] - abs_min[0]) as f32,
            (abs_max[1] - abs_min[1]) as f32,
            (abs_max[2] - abs_min[2]) as f32,
        );
        let mut local_min = Vec3::splat(f32::INFINITY);
        let mut local_max = Vec3::splat(f32::NEG_INFINITY);
        for corner in box_corners(abs_full) {
            let local = self.local_of(abs_origin + corner);
            local_min = local_min.min(local);
            local_max = local_max.max(local);
        }
        if is_axis_aligned(self.rotation) {
            (round_to_i64(local_min), round_to_i64(local_max))
        } else {
            (
                [local_min.x.floor() as i64, local_min.y.floor() as i64, local_min.z.floor() as i64],
                [local_max.x.ceil() as i64, local_max.y.ceil() as i64, local_max.z.ceil() as i64],
            )
        }
    }
}

/// The world offset (in ABSOLUTE voxels) that seats a producer of local dimensions `full`, rotated
/// by `rotation`, so its local CENTRE `full/2` lands at world `target_centre` under the SAME
/// corner-anchored [`LeafPlacement`] the classifier folds through (ADR 0027 §5 placement). It is
/// the inverse of [`LeafPlacement::new`]`(rotation, full, result).world_of(full/2) == target_centre`.
pub fn seat_centre_at(rotation: Quat, full: Vec3, target_centre: Vec3) -> Vec3 {
    let mut min_rotated_corner = Vec3::splat(f32::INFINITY);
    for corner in box_corners(full) {
        min_rotated_corner = min_rotated_corner.min(rotation * corner);
    }
    target_centre - rotation * (full * 0.5) + min_rotated_corner
}
