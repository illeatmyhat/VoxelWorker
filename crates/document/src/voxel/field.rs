//! The field seam (ADR 0019): a producer's signed distance field.

/// A producer's signed distance field: negative inside, zero on the surface, and changing no
/// faster than distance does in the metric it declares (ADR 0019, ADR 0020 Decision 1).
///
/// **This is deliberately narrower than ADR 0020's sketch, which also carried the cell
/// bracket.** Implementing the field layer showed those are separable capabilities that do
/// not always co-occur: [`DebugCloudField`] brackets every cell exactly (ADR 0021) while
/// having no usable pointwise distance â€” its field is a normalised radial falloff plus an fBm
/// displacement, whose Lipschitz constant is far above 1 and could only be normalised with a
/// *gradient* bound on the noise, where only a *range* bound is proven.
///
/// The split is ADR 0019's own rule showing up in the type system. Cell bracketing is a
/// **classification** capability and stays on
/// [`VoxelProducer::cell_field_interval`](crate::voxel::VoxelProducer::cell_field_interval), which
/// every producer may implement; a distance field is a **geometry** capability, and only a
/// genuine field has one. Predicates classify, fields measure.
///
/// [`DebugCloudField`]: crate::debug_clouds::DebugCloudField
pub trait Field: Send + Sync {
    /// Signed distance at `point_local_voxels`, a point in the producer's own
    /// `[0, full_dim)` voxel frame (ADR 0008 â€” the frame is carried, never re-derived).
    ///
    /// `voxels_per_block` is carried because a producer's field can depend on the document
    /// density: `Tube`'s wall is authored in whole blocks, so its geometry is not fixed until
    /// density is known. (ADR 0020's illustrative signature omitted it.)
    fn signed_distance(&self, point_local_voxels: [f32; 3], voxels_per_block: u32) -> f32;

    /// The metric [`signed_distance`](Self::signed_distance) is exact in â€” which decides the
    /// shape of an offset, so it is visible geometry rather than an implementation detail
    /// (ADR 0019 Decision 6). The enum names a norm family, not a dimension.
    fn metric(&self) -> substrate::geom2d::Metric;
}
