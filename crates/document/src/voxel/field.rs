//! The field seam: a producer's signed distance field.

/// A producer's signed distance field: negative inside, zero on the surface, and changing no
/// faster than distance does in the metric it declares.
///
/// **This carries no cell bracket, deliberately.** Bracketing a cell and measuring a distance
/// are separable capabilities that do not always co-occur: [`DebugCloudField`] brackets every
/// cell exactly while having no usable pointwise distance — its field is a normalized radial
/// falloff plus an fBm displacement, whose Lipschitz constant is far above 1 and could only be
/// normalized with a *gradient* bound on the noise, where only a *range* bound is proven.
///
/// So cell bracketing is a **classification** capability and stays on
/// [`VoxelProducer::cell_field_interval`](crate::voxel::VoxelProducer::cell_field_interval),
/// which every producer may implement; a distance field is a **geometry** capability, and only
/// a genuine field has one. Predicates classify, fields measure.
///
/// [`DebugCloudField`]: crate::debug_clouds::DebugCloudField
pub trait PreparedField: Send + Sync {
    /// Signed distance at a point after density-dependent setup has been performed.
    fn signed_distance(&self, point_local_voxels: [f32; 3]) -> f32;

    /// The metric of the prepared field.
    fn metric(&self) -> substrate::geom2d::Metric;

    /// Whether this evaluator preserves an interval capability the producer previously exposed.
    /// A generic signed-distance evaluator is not, by itself, authorization to turn a formerly
    /// boundary-only producer into a coarse classifier.
    fn preserves_native_interval(&self) -> bool {
        false
    }

    /// Conservative signed-distance interval over a cell after setup has been performed.
    ///
    /// A field is 1-Lipschitz in [`Self::metric`], so the metric-center bracket is sound for
    /// every field implementation. Producers with a sharper structural proof may still expose it
    /// through their legacy `VoxelProducer` interval door; this companion is the prepared seam
    /// used by a coarse-classification sweep so child setup is not repeated for every block.
    fn native_cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) -> crate::voxel::FieldInterval {
        crate::voxel::metric_cell_bracket(cell_local_voxels, self.metric(), |center| {
            self.signed_distance(center)
        })
    }
}

struct BorrowedPreparedField<'a, FieldT: Field + ?Sized> {
    field: &'a FieldT,
    voxels_per_block: u32,
}

impl<FieldT: Field + ?Sized> PreparedField for BorrowedPreparedField<'_, FieldT> {
    fn signed_distance(&self, point_local_voxels: [f32; 3]) -> f32 {
        self.field
            .signed_distance(point_local_voxels, self.voxels_per_block)
    }

    fn metric(&self) -> substrate::geom2d::Metric {
        self.field.metric()
    }

    fn preserves_native_interval(&self) -> bool {
        self.field.has_native_interval()
    }

    fn native_cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) -> crate::voxel::FieldInterval {
        self.field
            .native_cell_field_interval(cell_local_voxels, self.voxels_per_block)
    }
}

pub trait Field: Send + Sync {
    /// Signed distance at `point_local_voxels`, a point in the producer's own
    /// `[0, full_dim)` voxel frame — the frame is carried, never re-derived.
    ///
    /// `voxels_per_block` is carried because a producer's field can depend on the document
    /// density: `Tube`'s wall is authored in whole blocks, so its geometry is not fixed until
    /// density is known.
    fn signed_distance(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> f32;

    /// The metric [`signed_distance`](Self::signed_distance) is exact in — which decides the
    /// shape of an offset, so it is visible geometry rather than an implementation detail.
    /// The enum names a norm family, not a dimension.
    fn metric(&self) -> substrate::geom2d::Metric;

    /// Whether the owning [`VoxelProducer`](crate::voxel::VoxelProducer) already classifies
    /// cells with a native interval proof. This keeps preparation behavior-preserving: a field
    /// that was previously boundary-only remains boundary-only.
    fn has_native_interval(&self) -> bool {
        false
    }

    /// Conservative field interval used by the borrowing prepared adapter.
    ///
    /// The default is the metric-center bracket. Producers with a stronger existing interval
    /// proof override this so preparation preserves classification behavior exactly.
    fn native_cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        voxels_per_block: u32,
    ) -> crate::voxel::FieldInterval {
        crate::voxel::metric_cell_bracket(cell_local_voxels, self.metric(), |center| {
            self.signed_distance(center, voxels_per_block)
        })
    }

    /// Prepare density-dependent state once for a sampling operation. Producers without a
    /// specialized evaluator keep the legacy behavior through this borrowing adapter.
    fn prepare(&self, voxels_per_block: u32) -> Box<dyn PreparedField + '_> {
        Box::new(BorrowedPreparedField {
            field: self,
            voxels_per_block,
        })
    }
}
