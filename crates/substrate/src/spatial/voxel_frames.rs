//! Voxel COORDINATE-FRAME newtypes — the frame-invariant law: a spatial value CARRIES
//! the frame it was authored in, and the ONLY way to cross frames is an explicit, named conversion
//! that adds or subtracts the [`RecenterVoxels`] offset (or, for the producer-local crossing, folds
//! through the [`LeafPlacement`] affine). Making the frames distinct **zero-cost**
//! (`#[repr(transparent)]`) types turns a whole class of frame confusion — adding a producer-local
//! voxel to a true-world one, or treating a render-local `world_position + grid_half_extent` as if
//! it were the true-world coordinate — into a COMPILE error rather than a silent mis-placement.
//!
//! ## The frames
//! * [`TrueWorldVoxelPoint`] — the absolute producer/world voxel coordinate.
//! * [`RecenteredVoxelPoint`] — `true_world − recenter`, the frame a rebuild's resolved grid lives
//!   in (the floating-origin recenter keeps a far scene `f32`-exact).
//! * [`ProducerLocalVoxelPoint`] — a coordinate inside a leaf's local `[0, full)` box, BEFORE the
//!   placement affine. The producer-local ↔ true-world crossing is NOT a pure translation (it folds
//!   through a rotation + corner anchor), so it lives on [`LeafPlacement`]; this module owns only
//!   the pure-translation recenter crossing and the render grid-cage offset.
//!
//! [`LeafPlacement`]: crate::spatial::LeafPlacement

use glam::Vec3;

/// The composite floating-origin recenter, in voxels — the offset a rebuild's resolved grid was
/// shifted by so a far scene stays `f32`-exact (`recentered = true_world − recenter`). It is the
/// frame value every display artifact of one rebuild is resolved in, carried end-to-end (resolve →
/// orchestrator → the async worker channels → the GPU install) so the compiler enforces that the
/// install uses the request's recenter rather than a same-shaped `[i64; 3]` from somewhere else.
///
/// The one PRODUCTION mint point is `Scene::recenter_voxels_for_resolve`, which returns this newtype
/// directly; [`new`](RecenterVoxels::new) remains for the boundary/test sites that carry a KNOWN
/// recenter from a raw triple (the `shot` oracle grid's carried field, the parity tests). It is
/// `Copy`, and [`voxels`](RecenterVoxels::voxels) is the ONE way back to the raw triple — unwrapped
/// only at the point of actual positional ARITHMETIC and at the GPU uniform packing.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RecenterVoxels([i64; 3]);

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
const fn i64_to_f32(value: i64) -> f32 {
    value as f32
}

#[allow(clippy::as_conversions, clippy::cast_precision_loss)]
const fn u32_to_f32(value: u32) -> f32 {
    value as f32
}

impl RecenterVoxels {
    /// Carry a known recenter triple as its frame value — the boundary/test constructor for a
    /// recenter that arrives as a raw `[i64; 3]`.
    #[must_use]
    pub const fn new(voxels: [i64; 3]) -> Self {
        Self(voxels)
    }

    /// The raw voxel triple — the single consumption door, called only at the point of positional
    /// arithmetic, at the GPU uniform packing, and at the raw-by-rule oracle / cache / delta values.
    #[must_use]
    pub const fn voxels(&self) -> [i64; 3] {
        self.0
    }

    /// The translation that carries a point expressed in THIS frame into `viewer`'s frame: a
    /// point `p` recorded here stands at `p + this_offset` for someone measuring in `viewer`.
    /// Equal to `self − viewer`, and exactly `ZERO` when the two frames agree.
    ///
    /// The frames differ by a pure translation, so a consumer holding data baked in one frame and
    /// a camera built for another has a choice of which side to move. Moving the camera is the
    /// cheap side: it is one matrix concat against however many vertices, and it leaves the data
    /// untouched — which matters when the data is a GPU buffer somebody else is still filling.
    /// The subtraction is taken in `i64` before either term is an `f32`, so a far scene loses
    /// nothing to the downcast that its own coordinates would not have lost anyway.
    #[must_use]
    pub fn a_point_of_this_frame_seen_from(self, viewer: Self) -> Vec3 {
        let here = self.voxels();
        let there = viewer.voxels();
        Vec3::from_array(std::array::from_fn(|axis| {
            i64_to_f32(
                here.get(axis)
                    .copied()
                    .unwrap_or_default()
                    .wrapping_sub(there.get(axis).copied().unwrap_or_default()),
            )
        }))
    }
}

/// The TRUE-WORLD voxel coordinate of a region's LOW CORNER — the point a region-scoped
/// consumer's 0-based index space counts from (`index = true_world − low_corner`).
///
/// This is one quantity that five call sites used to spell for themselves, in two disguises. The
/// plain one is `recenter − floor(dim/2)`. The other cancels the recenter away entirely: a
/// consumer holding a RENDER-frame position writes `index = world_render + floor(dim/2)`, because
///
/// ```text
/// index = true_world − low = (world_render + recenter) − (recenter − floor(dim/2))
/// ```
///
/// Nobody who wrote that made a mistake. They used an identity, and the identity holds because
/// the origin policy puts the render origin at the region's midpoint. It is the policy that
/// guarantees it, and the policy is what a floating origin eventually has to change.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegionLowCorner([i64; 3]);

impl RegionLowCorner {
    /// Carry a known low corner — the constructor for a low corner that arrives already
    /// established as a fact about the geometry (`Scene::placed_composite_low_corner_voxels`).
    #[must_use]
    pub const fn new(voxels: [i64; 3]) -> Self {
        Self(voxels)
    }

    /// The low corner of a region that is CENTERED ON THE RENDER ORIGIN — `recenter −
    /// floor(dim/2)`.
    ///
    /// The assumption is in the name because the assumption is the whole risk. A cage built
    /// this way contains the composite only while the origin sits at the composite's midpoint;
    /// the day the origin becomes sticky or quantized, every caller of this function is a site
    /// whose region has quietly stopped containing what it frames. That makes the call itself
    /// the to-do list — grep for it — rather than a subtraction spread across five files where
    /// nothing marks which ones share a fate.
    ///
    /// The subtraction is `i64`, and the cast comes later at
    /// [`as_render_offset`](Self::as_render_offset). Doing it the other way round — cast each
    /// term, then subtract — costs a far scene precision that neither term would have lost on
    /// its own, which is the rule [`RecenterVoxels::a_point_of_this_frame_seen_from`] states
    /// and which this function used to break.
    ///
    /// The FLOOR is integer division before any cast, matching [`GridHalfExtent`]: an odd
    /// dimension halved in floating point sits half a voxel off and mis-snaps the whole index
    /// space by one on that axis.
    #[must_use]
    #[allow(clippy::as_conversions)]
    pub fn of_origin_centered_region(recenter: RecenterVoxels, dimensions: [u32; 3]) -> Self {
        Self(std::array::from_fn(|axis| {
            let half = i64::from(dimensions.get(axis).copied().unwrap_or_default() / 2);
            recenter
                .voxels()
                .get(axis)
                .copied()
                .unwrap_or_default()
                .wrapping_sub(half)
        }))
    }

    /// The raw voxel triple — the consumption door, at the point of positional arithmetic.
    #[must_use]
    pub const fn voxels(self) -> [i64; 3] {
        self.0
    }

    /// The per-axis `f32` offset that carries a shader's RENDER-local
    /// `voxel_absolute_position` (`= world_position + grid_half_extent`, a 0-based index into
    /// the cage) into the TRUE world voxel frame: `true_world = render_absolute + low_corner`.
    /// See `shaders/cuboid.wgsl`'s `overlay_world_offset`.
    ///
    /// The single downcast, taken once at the uniform-packing door and never before.
    #[must_use]
    pub fn as_render_offset(self) -> [f32; 3] {
        self.0.map(i64_to_f32)
    }
}

/// Half the render grid's voxel dimensions, floored per axis (`floor(dim / 2)`) — the grid cage's
/// corner-anchoring term. The mesh centers its cage on the origin (its low corner sits at
/// `−grid_half_extent`), so a shader recovers the RENDER-LOCAL absolute voxel index with
/// `world_position + grid_half_extent`.
///
/// **That index role is all this type is for.** The same floored half also appeared inside the
/// frame derivation `recenter − floor(dim/2)`, which made it look like a frame term; it is not,
/// and that derivation now lives once in [`RegionLowCorner::of_origin_centered_region`]. What
/// remains here is a fact about how the CAGE was baked — the vertices really are centered on the
/// origin — and it stays true of a given mesh no matter what the origin policy later does, which
/// is exactly why it must not be confused with a frame.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GridHalfExtent([f32; 3]);

impl GridHalfExtent {
    /// The floored half-extent of a grid `dimensions` (voxels): `floor(dim / 2)` per axis, cast to
    /// `f32`. The integer division BEFORE the cast reproduces the corner-anchoring the shader relies
    /// on — a `dim / 2.0` would sit half a voxel off for an ODD dimension, mis-snapping the overlay
    /// and the Z-band clip.
    #[must_use]
    pub fn of_grid_dimensions(dimensions: [u32; 3]) -> Self {
        Self(dimensions.map(|dimension| u32_to_f32(dimension / 2)))
    }

    /// The raw per-axis half-extent — the GPU uniform packing door (`grid_half_extent`).
    #[must_use]
    pub const fn voxels(&self) -> [f32; 3] {
        self.0
    }
}

/// The recenter offset as an `f32` vector — the single place the `[i64; 3]` recenter is downcast for
/// the pure-translation point crossings below.
fn recenter_as_vec3(recenter: RecenterVoxels) -> Vec3 {
    let voxels = recenter.voxels();
    Vec3::from_array(voxels.map(i64_to_f32))
}

/// The absolute producer/world voxel coordinate. Cross into another frame ONLY via a
/// named conversion — [`to_recentered`](Self::to_recentered) for the recenter translation, or
/// [`LeafPlacement::local_of`](crate::spatial::LeafPlacement::local_of) for the producer-local
/// affine.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TrueWorldVoxelPoint(Vec3);

impl TrueWorldVoxelPoint {
    /// Carry a `Vec3` as a true-world voxel point — the boundary constructor where a raw absolute
    /// coordinate (a leaf's world offset, an absolute cell center) enters the typed frame world.
    #[must_use]
    pub const fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door, called where the true-world coordinate leaves the
    /// typed world for arithmetic (a `floor` to an integer cell, a GPU coordinate).
    #[must_use]
    pub const fn voxels(self) -> Vec3 {
        self.0
    }

    /// Translate into the [`RecenteredVoxelPoint`] frame by SUBTRACTING the recenter — one of the two
    /// audited recenter crossings (`recentered = true_world − recenter`).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn to_recentered(self, recenter: RecenterVoxels) -> RecenteredVoxelPoint {
        RecenteredVoxelPoint(self.0 - recenter_as_vec3(recenter))
    }
}

/// A voxel coordinate in the recentered frame (`true_world − recenter`) — the frame a rebuild's
/// resolved grid and its mesh vertices are expressed in.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RecenteredVoxelPoint(Vec3);

impl RecenteredVoxelPoint {
    /// Carry a `Vec3` as a recentered voxel point — the boundary constructor for a coordinate born in
    /// the resolved-grid frame.
    #[must_use]
    pub const fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door out of the recentered frame.
    #[must_use]
    pub const fn voxels(self) -> Vec3 {
        self.0
    }

    /// Translate into the [`TrueWorldVoxelPoint`] frame by ADDING the recenter — the inverse of
    /// [`TrueWorldVoxelPoint::to_recentered`] (`true_world = recentered + recenter`).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub fn to_true_world(self, recenter: RecenterVoxels) -> TrueWorldVoxelPoint {
        TrueWorldVoxelPoint(self.0 + recenter_as_vec3(recenter))
    }
}

/// A voxel coordinate inside a leaf's local `[0, full)` box, BEFORE the placement affine.
/// It reaches the true-world frame only by folding through
/// [`LeafPlacement::world_of`](crate::spatial::LeafPlacement::world_of) — never by a bare add.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ProducerLocalVoxelPoint(Vec3);

impl ProducerLocalVoxelPoint {
    /// Carry a `Vec3` as a producer-local voxel point — the boundary constructor for a local cell
    /// center / box corner the placement affine will fold through.
    #[must_use]
    pub const fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door, called where the producer-local coordinate is handed
    /// to the producer's field sampler (`signed_distance` / `material_at`).
    #[must_use]
    pub const fn voxels(self) -> Vec3 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::all,
        clippy::arithmetic_side_effects,
        clippy::as_conversions,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::pedantic,
        clippy::nursery,
        clippy::unwrap_used
    )]
    use super::*;

    #[test]
    fn recenter_round_trips_a_true_world_point() {
        let recenter = RecenterVoxels::new([100, -40, 7]);
        let point = TrueWorldVoxelPoint::from_voxels(Vec3::new(3.5, 9.0, -2.0));
        let there_and_back = point.to_recentered(recenter).to_true_world(recenter);
        assert_eq!(there_and_back.voxels(), point.voxels());
    }

    #[test]
    fn recentered_is_true_world_minus_recenter() {
        let recenter = RecenterVoxels::new([10, 20, 30]);
        let recentered = TrueWorldVoxelPoint::from_voxels(Vec3::new(11.0, 22.0, 33.0))
            .to_recentered(recenter)
            .voxels();
        assert_eq!(recentered, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn grid_half_extent_floors_an_odd_dimension() {
        // floor(7/2) = 3, not 3.5 — the corner-anchoring the overlay relies on.
        assert_eq!(
            GridHalfExtent::of_grid_dimensions([7, 8, 9]).voxels(),
            [3.0, 4.0, 4.0]
        );
    }

    #[test]
    fn an_origin_centered_low_corner_is_the_recenter_less_the_floored_half() {
        let recenter = RecenterVoxels::new([64, 0, -16]);
        // floor(16/2) = 8 on every axis; the odd axis floors to 3, not 3.5.
        assert_eq!(
            RegionLowCorner::of_origin_centered_region(recenter, [16, 16, 7]).voxels(),
            [64 - 8, 0 - 8, -16 - 3]
        );
    }

    /// **The subtraction happens in `i64`, and a far scene is why.**
    ///
    /// Past 2^24 an `f32` no longer names every voxel, so casting each term FIRST and
    /// subtracting after rounds both sides to the same representable value and loses the
    /// difference between them — a difference that is itself small and perfectly representable.
    /// Subtracting first and casting once keeps it. The overlay offset this feeds used to be
    /// computed the other way round, and this is the case that told them apart.
    #[test]
    fn a_far_low_corner_keeps_the_voxel_the_cast_order_would_have_eaten() {
        // 2^24 = 16_777_216: the first magnitude at which consecutive integers stop being
        // distinct f32 values.
        let recenter = RecenterVoxels::new([16_777_217, 0, 0]);
        let low = RegionLowCorner::of_origin_centered_region(recenter, [2, 2, 2]);

        assert_eq!(low.voxels()[0], 16_777_216);
        assert_eq!(low.as_render_offset()[0], 16_777_216.0);

        // What the cast-first order produced: 16_777_217 is not representable, so it rounds to
        // 16_777_216 before the subtraction, and the answer comes out a whole voxel short.
        #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
        let cast_first = 16_777_217_i64 as f32 - 1.0_f32;
        assert_eq!(cast_first, 16_777_215.0);
    }

    /// A point baked in one frame, walked into another, lands where that other frame would have
    /// put it in the first place — for a world coordinate far enough out that the walk is the
    /// whole difference between drawing it and misdrawing it.
    ///
    /// This is the arithmetic under a display pass whose buffers were emitted before the
    /// floating origin last moved. Agreeing frames give exactly `ZERO`, which is what lets the
    /// consumer skip the matrix concat and stay bit-identical in the steady state.
    #[test]
    fn a_point_walked_between_frames_lands_where_the_other_frame_puts_it() {
        let baked = RecenterVoxels::new([640, -20, 8]);
        let current = RecenterVoxels::new([1280, -20, -12]);
        let world = Vec3::new(5000.0, 300.0, -70.0);

        let in_baked = TrueWorldVoxelPoint::from_voxels(world)
            .to_recentered(baked)
            .voxels();
        let in_current = TrueWorldVoxelPoint::from_voxels(world)
            .to_recentered(current)
            .voxels();
        assert_ne!(in_baked, in_current, "the two frames must actually differ");
        assert_eq!(
            in_baked + baked.a_point_of_this_frame_seen_from(current),
            in_current,
        );

        assert_eq!(
            baked.a_point_of_this_frame_seen_from(baked),
            Vec3::ZERO,
            "a frame seen from itself asks for no walk at all",
        );
    }
}
