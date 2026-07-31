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
}
