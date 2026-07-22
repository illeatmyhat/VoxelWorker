//! Outset: a producer decorator that dilates a body before it folds (ADR 0019 Decision 7).

use super::{Field, FieldInterval, VoxelProducer};
use voxel_core::spatial_index::VoxelAabb;
use voxel_core::voxel::{BlockAttrs, BlockId, Voxel, VoxelGrid, SURFACE_ISOLEVEL};

/// A producer whose body is the inner producer DILATED by `outset_voxels` (ADR 0019
/// Decision 7). A negative outset insets, eroding instead.
///
/// **Outset is a decorator, not a fold arm, and that is the point.** ADR 0020 Decision 7
/// warns that the fold exists twice — over voxel sets in `scene::producers` and over
/// intervals in `substrate::solids::cell_classification` — and that the two diverge silently
/// if only one learns a new operation. Wrapping the *producer* sidesteps that hazard by
/// construction: both folds consume this type through the ordinary [`VoxelProducer`]
/// interface, so there is exactly one implementation of what outset MEANS and no second arm
/// to keep in sync. `stamp_producer`, `mask_producer` (carve/intersect) and the
/// classifier's `cell_field_interval` call are all unchanged.
///
/// # The frame
///
/// Dilation grows the body by `N` on every side, so the wrapper's `[0, full_dim)` frame is
/// the inner one shifted: wrapper coordinate `x` is inner coordinate `x − N`, and the
/// wrapper's dimensions are the inner's plus `2N`. Callers anchor the wrapper by subtracting
/// `N` from the leaf's world offset, which keeps the dilated body centred on the original
/// (ADR 0008 — the frame is carried, never re-derived).
///
/// # Why this needs a field
///
/// Dilation is `d(p) <= N`, so it is meaningless without a distance. ADR 0020 Decision 1
/// makes that structural: a producer with no [`Field`] cannot be outset, and
/// [`wrap`](Self::wrap) returns the inner producer untouched rather than inventing a
/// geometry for it. `DebugCloudField` is the case that matters — it brackets cells exactly
/// but has no usable pointwise distance, and its cell intervals are still sign-only
/// sentinels, so shifting them would resurrect exactly the ADR 0019 Decision 1 trap that
/// metricising `SketchSolid` closed.
pub struct OutsetProducer {
    inner: Box<dyn VoxelProducer>,
    outset_voxels: i64,
}

impl OutsetProducer {
    /// Wrap `inner` so its body dilates by `outset_voxels`, or hand back `inner` unchanged
    /// when the outset is zero or the producer has no field to dilate.
    ///
    /// Returning the inner producer for a fieldless one is ADR 0020 Decision 1's "outset is
    /// unavailable there", not a silent failure to dilate: nothing in the document can set a
    /// nonzero outset on such a node today, and the type is what bars it.
    pub fn wrap(inner: Box<dyn VoxelProducer>, outset_voxels: i64) -> Box<dyn VoxelProducer> {
        if outset_voxels == 0 || inner.as_field().is_none() {
            return inner;
        }
        Box::new(Self { inner, outset_voxels })
    }

    /// How far the wrapper's frame origin sits BELOW the inner producer's, per axis.
    fn origin_shift(&self) -> f32 {
        self.outset_voxels as f32
    }

    /// The inner-frame point a wrapper-frame point denotes.
    fn to_inner_point(&self, point: [f32; 3]) -> [f32; 3] {
        let shift = self.origin_shift();
        [point[0] - shift, point[1] - shift, point[2] - shift]
    }
}

impl VoxelProducer for OutsetProducer {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [x, y, z] = self.full_dimensions(voxels_per_block);
        self.resolve_into(
            grid,
            voxels_per_block,
            VoxelAabb::new([0, 0, 0], [x as i64, y as i64, z as i64]),
        );
    }

    /// Fill by testing the dilated field directly: a voxel is occupied iff
    /// `d(centre) <= N`, the closed Minkowski dilation of the body by a ball of radius `N`
    /// in the inner producer's own metric.
    ///
    /// This resolves through the FIELD rather than through the inner producer's occupancy
    /// predicate, which is the only way to reach the cells the dilation ADDS — they lie
    /// outside the inner producer's extent entirely, where its resolve emits nothing.
    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: VoxelAabb,
    ) {
        let dimensions = self.full_dimensions(voxels_per_block);
        // FULL dimensions even for a windowed fill — downstream decode reads indices
        // against the whole extent (the `resolve_into` contract).
        grid.dimensions = dimensions;
        let Some(field) = self.inner.as_field() else {
            return;
        };
        let outset = self.outset_voxels as f32;

        // Clamp the window to `[0, full_dim)` per axis, so an oversized window is harmless
        // and a full-window call reproduces the whole body exactly (the `resolve_into`
        // contract).
        let low: [i64; 3] =
            std::array::from_fn(|axis| window_local_voxels.min[axis].clamp(0, dimensions[axis] as i64));
        let high: [i64; 3] = std::array::from_fn(|axis| {
            window_local_voxels.max[axis].clamp(low[axis], dimensions[axis] as i64)
        });

        let mut occupied = Vec::new();
        for k in low[2]..high[2] {
            for j in low[1]..high[1] {
                for i in low[0]..high[0] {
                    // Sample at the voxel centre in the wrapper frame, mapped back into the
                    // inner producer's frame.
                    let centre =
                        self.to_inner_point([i as f32 + 0.5, j as f32 + 0.5, k as f32 + 0.5]);
                    if field.signed_distance(centre, voxels_per_block) - outset
                        <= SURFACE_ISOLEVEL
                    {
                        occupied.push(Voxel {
                            local_index: [i as i32, j as i32, k as i32],
                            block_local_coord: [
                                (i % voxels_per_block as i64) as u8,
                                (j % voxels_per_block as i64) as u8,
                                (k % voxels_per_block as i64) as u8,
                            ],
                            // A composed Part carries per-voxel materials, and the dilated
                            // shell must inherit the material of the surface it grew from
                            // rather than flattening the Part to one colour — so the
                            // material is sampled at the SAME inner point as the distance.
                            // A single-material producer answers `None` here and its leaf
                            // override stamps instead, exactly as before.
                            block_id: self
                                .inner
                                .material_at(centre, voxels_per_block)
                                .unwrap_or(BlockId::DEFAULT),
                            attrs: BlockAttrs::DEFAULT,
                            grid_overlay: false,
                        });
                    }
                }
            }
        }
        grid.occupied = occupied;
    }

    /// The Lipschitz bracket of THIS producer's own field over the cell.
    ///
    /// **It deliberately does not delegate to the inner producer's bracket, and that is a
    /// correctness requirement rather than a preference.** A producer may measure its field
    /// in one metric while bracketing cells in another: `SdfShape` resolves and brackets a
    /// Box through the free Euclidean-magnitude `signed_distance`, but its [`Field`] impl
    /// measures a Box in Chebyshev. Those two "agree in sign everywhere", which is all an
    /// un-outset classification needs — sign is the whole verdict.
    ///
    /// An outset shift compares MAGNITUDES, so sign agreement stops being enough. Shifting
    /// the delegated Euclidean bracket while resolving through the Chebyshev field reports
    /// AIR for cells the dilated body actually fills: at a box's corner voxel the Chebyshev
    /// distance is `3.5` (inside after an outset of `4`) while the Euclidean is `6.06`
    /// (bracketed to air). The parity gate catches it, which is exactly what it exists for.
    ///
    /// Bracketing the wrapper's own field keeps the measure and the fill in one metric by
    /// construction. The cost is the inner producer's structural refinements — a sketch's
    /// provably-solid closure, its extent clearance — which do not survive the change of
    /// metric and would have to be re-derived per metric to be reused.
    fn cell_field_interval(
        &self,
        cell_local_voxels: VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<FieldInterval> {
        if cell_local_voxels.is_empty() {
            return None;
        }
        // Occupancy is decided at voxel CENTRES (`index + 0.5`), so the region to bracket is
        // `[min + 0.5, max − 0.5]` — the same samples `resolve_into` visits.
        let mut centre = [0.0f32; 3];
        let mut half_extent = [0.0f32; 3];
        for axis in 0..3 {
            let low = cell_local_voxels.min[axis] as f32 + 0.5;
            let high = (cell_local_voxels.max[axis] - 1) as f32 + 0.5;
            centre[axis] = 0.5 * (low + high);
            half_extent[axis] = 0.5 * (high - low);
        }
        // The circumradius belongs to the metric the field is 1-Lipschitz in: under
        // Chebyshev the largest half-extent, under Euclidean the half-diagonal.
        let circumradius = match self.metric() {
            substrate::geom2d::Metric::Chebyshev => {
                half_extent.iter().copied().fold(0.0f32, f32::max)
            }
            substrate::geom2d::Metric::Euclidean => half_extent
                .iter()
                .map(|extent| extent * extent)
                .sum::<f32>()
                .sqrt(),
        };
        Some(FieldInterval::from_lipschitz_center(
            self.signed_distance(centre, voxels_per_block),
            circumradius,
        ))
    }

    /// Delegated at the corresponding inner point, so a dilated Part's materials line up
    /// with its undilated ones.
    fn material_at(
        &self,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> Option<voxel_core::core_geom::BlockId> {
        self.inner
            .material_at(self.to_inner_point(point_local_voxels), voxels_per_block)
    }

    fn as_field(&self) -> Option<&dyn Field> {
        Some(self)
    }

    /// The inner extent grown by `N` on every side (so `2N` per axis), floored at zero — an
    /// inset deeper than the body's own half-extent erodes it away entirely.
    fn full_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        let inner = self.inner.full_dimensions(voxels_per_block);
        std::array::from_fn(|axis| {
            (inner[axis] as i64 + 2 * self.outset_voxels).max(0) as u32
        })
    }
}

impl Field for OutsetProducer {
    /// The dilated field: `d_inner(p) − N`. Subtracting a constant preserves the Lipschitz
    /// bound exactly, so the classifier's soundness argument carries over unchanged.
    fn signed_distance(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> f32 {
        match self.inner.as_field() {
            Some(field) => {
                field.signed_distance(self.to_inner_point(point_local_voxels), voxels_per_block)
                    - self.outset_voxels as f32
            }
            None => f32::INFINITY,
        }
    }

    /// Unchanged by the offset: dilating by a constant moves the surface, never the norm the
    /// distance is measured in. This is what makes ADR 0019 Decision 6's rule — "outset's
    /// shape follows the body's category" — fall out rather than need enforcing: a box
    /// outsets square because a box MEASURES square.
    fn metric(&self) -> substrate::geom2d::Metric {
        match self.inner.as_field() {
            Some(field) => field.metric(),
            None => substrate::geom2d::Metric::Euclidean,
        }
    }
}
