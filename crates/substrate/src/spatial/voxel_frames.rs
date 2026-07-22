//! Voxel COORDINATE-FRAME newtypes (the frame-invariant law, ADR 0008): a spatial value CARRIES
//! the frame it was authored in, and the ONLY way to cross frames is an explicit, named conversion
//! that adds or subtracts the [`RecentreVoxels`] offset (or, for the producer-local crossing, folds
//! through the [`LeafPlacement`] affine). Making the frames distinct **zero-cost**
//! (`#[repr(transparent)]`) types turns a whole class of frame confusion — adding a producer-local
//! voxel to a true-world one, or treating a render-local `world_position + grid_half_extent` as if
//! it were the true-world coordinate — into a COMPILE error rather than a silent mis-placement.
//!
//! ## The frames
//! * [`TrueWorldVoxelPoint`] — the absolute producer/world voxel coordinate.
//! * [`RecentredVoxelPoint`] — `true_world − recentre`, the frame a rebuild's resolved grid lives
//!   in (the floating-origin recentre keeps a far scene `f32`-exact).
//! * [`ProducerLocalVoxelPoint`] — a coordinate inside a leaf's local `[0, full)` box, BEFORE the
//!   placement affine. The producer-local ↔ true-world crossing is NOT a pure translation (it folds
//!   through a rotation + corner anchor), so it lives on [`LeafPlacement`]; this module owns only
//!   the pure-translation recentre crossing and the render grid-cage offset.
//!
//! [`LeafPlacement`]: crate::spatial::LeafPlacement

use glam::Vec3;

/// The composite floating-origin recentre, in voxels — the offset a rebuild's resolved grid was
/// shifted by so a far scene stays `f32`-exact (`recentred = true_world − recentre`). It is the
/// frame value every display artifact of one rebuild is resolved in, carried end-to-end (resolve →
/// orchestrator → the async worker channels → the GPU install) so the compiler enforces that the
/// install uses the request's recentre rather than a same-shaped `[i64; 3]` from somewhere else.
///
/// The one PRODUCTION mint point is `Scene::recentre_voxels_for_resolve`, which returns this newtype
/// directly; [`new`](RecentreVoxels::new) remains for the boundary/test sites that carry a KNOWN
/// recentre from a raw triple (the `shot` oracle grid's carried field, the parity tests). It is
/// `Copy`, and [`voxels`](RecentreVoxels::voxels) is the ONE way back to the raw triple — unwrapped
/// only at the point of actual positional ARITHMETIC and at the GPU uniform packing.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RecentreVoxels([i64; 3]);

impl RecentreVoxels {
    /// Carry a known recentre triple as its frame value — the boundary/test constructor for a
    /// recentre that arrives as a raw `[i64; 3]`.
    pub fn new(voxels: [i64; 3]) -> Self {
        Self(voxels)
    }

    /// The raw voxel triple — the single consumption door, called only at the point of positional
    /// arithmetic, at the GPU uniform packing, and at the raw-by-rule oracle / cache / delta values.
    pub fn voxels(&self) -> [i64; 3] {
        self.0
    }

    /// The per-axis `f32` offset the on-face grid overlay adds to a shader's RENDER-local
    /// `voxel_absolute_position` (`= world_position + grid_half_extent`) to recover the TRUE world
    /// voxel frame: `true_world = render_absolute + (recentre − grid_half_extent)` (see
    /// `shaders/cuboid.wgsl`'s `overlay_world_offset`). This is the ONE audited place the
    /// `recentre − grid_half_extent` subtraction happens — so no call site can treat
    /// `world_position + grid_half_extent` (a render-local index) as if it were the true-world
    /// coordinate, and a [`GridHalfExtent`] can never be swapped for a `RecentreVoxels` here.
    pub fn render_absolute_to_true_world_offset(self, grid_half_extent: GridHalfExtent) -> [f32; 3] {
        std::array::from_fn(|axis| self.0[axis] as f32 - grid_half_extent.0[axis])
    }
}

/// Half the render grid's voxel dimensions, floored per axis (`floor(dim / 2)`) — the grid cage's
/// corner-anchoring term. The mesh centres its cage on the origin (its low corner sits at
/// `−grid_half_extent`), so a shader recovers the render-local absolute voxel index with
/// `world_position + grid_half_extent`. A distinct type from [`RecentreVoxels`] so the two frame
/// terms of the overlay offset can never be swapped.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GridHalfExtent([f32; 3]);

impl GridHalfExtent {
    /// The floored half-extent of a grid `dimensions` (voxels): `floor(dim / 2)` per axis, cast to
    /// `f32`. The integer division BEFORE the cast reproduces the corner-anchoring the shader relies
    /// on — a `dim / 2.0` would sit half a voxel off for an ODD dimension, mis-snapping the overlay
    /// and the Z-band clip.
    pub fn of_grid_dimensions(dimensions: [u32; 3]) -> Self {
        Self(std::array::from_fn(|axis| (dimensions[axis] / 2) as f32))
    }

    /// The raw per-axis half-extent — the GPU uniform packing door (`grid_half_extent`).
    pub fn voxels(&self) -> [f32; 3] {
        self.0
    }
}

/// The recentre offset as an `f32` vector — the single place the `[i64; 3]` recentre is downcast for
/// the pure-translation point crossings below.
fn recentre_as_vec3(recentre: RecentreVoxels) -> Vec3 {
    let voxels = recentre.voxels();
    Vec3::new(voxels[0] as f32, voxels[1] as f32, voxels[2] as f32)
}

/// The absolute producer/world voxel coordinate (ADR 0008). Cross into another frame ONLY via a
/// named conversion — [`to_recentred`](Self::to_recentred) for the recentre translation, or
/// [`LeafPlacement::local_of`](crate::spatial::LeafPlacement::local_of) for the producer-local
/// affine.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TrueWorldVoxelPoint(Vec3);

impl TrueWorldVoxelPoint {
    /// Carry a `Vec3` as a true-world voxel point — the boundary constructor where a raw absolute
    /// coordinate (a leaf's world offset, an absolute cell centre) enters the typed frame world.
    pub fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door, called where the true-world coordinate leaves the
    /// typed world for arithmetic (a `floor` to an integer cell, a GPU coordinate).
    pub fn voxels(self) -> Vec3 {
        self.0
    }

    /// Translate into the [`RecentredVoxelPoint`] frame by SUBTRACTING the recentre — one of the two
    /// audited recentre crossings (`recentred = true_world − recentre`).
    pub fn to_recentred(self, recentre: RecentreVoxels) -> RecentredVoxelPoint {
        RecentredVoxelPoint(self.0 - recentre_as_vec3(recentre))
    }
}

/// A voxel coordinate in the recentred frame (`true_world − recentre`) — the frame a rebuild's
/// resolved grid and its mesh vertices are expressed in.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RecentredVoxelPoint(Vec3);

impl RecentredVoxelPoint {
    /// Carry a `Vec3` as a recentred voxel point — the boundary constructor for a coordinate born in
    /// the resolved-grid frame.
    pub fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door out of the recentred frame.
    pub fn voxels(self) -> Vec3 {
        self.0
    }

    /// Translate into the [`TrueWorldVoxelPoint`] frame by ADDING the recentre — the inverse of
    /// [`TrueWorldVoxelPoint::to_recentred`] (`true_world = recentred + recentre`).
    pub fn to_true_world(self, recentre: RecentreVoxels) -> TrueWorldVoxelPoint {
        TrueWorldVoxelPoint(self.0 + recentre_as_vec3(recentre))
    }
}

/// A voxel coordinate inside a leaf's local `[0, full)` box, BEFORE the placement affine (ADR 0027).
/// It reaches the true-world frame only by folding through
/// [`LeafPlacement::world_of`](crate::spatial::LeafPlacement::world_of) — never by a bare add.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ProducerLocalVoxelPoint(Vec3);

impl ProducerLocalVoxelPoint {
    /// Carry a `Vec3` as a producer-local voxel point — the boundary constructor for a local cell
    /// centre / box corner the placement affine will fold through.
    pub fn from_voxels(point: Vec3) -> Self {
        Self(point)
    }

    /// The raw `Vec3` — the consumption door, called where the producer-local coordinate is handed
    /// to the producer's field sampler (`signed_distance` / `material_at`).
    pub fn voxels(self) -> Vec3 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recentre_round_trips_a_true_world_point() {
        let recentre = RecentreVoxels::new([100, -40, 7]);
        let point = TrueWorldVoxelPoint::from_voxels(Vec3::new(3.5, 9.0, -2.0));
        let there_and_back = point.to_recentred(recentre).to_true_world(recentre);
        assert_eq!(there_and_back.voxels(), point.voxels());
    }

    #[test]
    fn recentred_is_true_world_minus_recentre() {
        let recentre = RecentreVoxels::new([10, 20, 30]);
        let recentred = TrueWorldVoxelPoint::from_voxels(Vec3::new(11.0, 22.0, 33.0))
            .to_recentred(recentre)
            .voxels();
        assert_eq!(recentred, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn grid_half_extent_floors_an_odd_dimension() {
        // floor(7/2) = 3, not 3.5 — the corner-anchoring the overlay relies on.
        assert_eq!(GridHalfExtent::of_grid_dimensions([7, 8, 9]).voxels(), [3.0, 4.0, 4.0]);
    }

    #[test]
    fn overlay_offset_is_recentre_minus_half_extent() {
        let recentre = RecentreVoxels::new([64, 0, -16]);
        let half = GridHalfExtent::of_grid_dimensions([16, 16, 16]); // floor(16/2) = 8
        assert_eq!(
            recentre.render_absolute_to_true_world_offset(half),
            [64.0 - 8.0, 0.0 - 8.0, -16.0 - 8.0]
        );
    }
}
