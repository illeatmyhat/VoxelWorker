//! The producers that fill the resolved voxel grid.
//!
//! ## Coordinate convention (PROJECT-WIDE — Z-up, right-handed)
//!
//! **Vertical / up = +Z** ([`glam::Vec3::Z`], array index **2**) EVERYWHERE in this
//! project — camera, SDFs, onion skin, layers, diameter, mesh and `.vox` export all
//! agree. The ground plane is **XY** (normal +Z); **front = −Y** (the front view looks
//! along +Y); LEFT/RIGHT = ±X; TOP/BOTTOM = ±Z. Panel X/Y/Z fields map directly to
//! indices 0/1/2 with Z genuinely the vertical axis — no relabel shim.
//!
//! Consequences pinned by tests: a tall cylinder/tube/torus has its axis along Z
//! (`size_voxels[2]` is the vertical extent), layer slices are Z-slices, the onion
//! band is a Z-range, and the `.vox` export writes our Z straight to vox-Z with
//! NO axis swap (the `.vox` format is itself Z-up).
//!
//! ## The producer seam
//!
//! This module implements the architectural seam **the renderer never calls the SDF
//! directly**: instead a [`VoxelProducer`] resolves a parametric shape (or a sub-assembly,
//! or a sketch, or a sculpt overlay) into a [`VoxelGrid`] — the one consumed truth. The
//! renderer, the layer-range diameter readout and the `.vox` export all read the grid, so
//! a new producer costs nothing downstream.
//!
//! ## The value ⊥ producer split
//!
//! This is the **document-bound** producer half. It depends DOWNWARD on the
//! foundational value vocabulary in the `voxel_core` crate (the resolved
//! [`Voxel`](voxel_core::voxel::Voxel),
//! its [`VoxelGrid`], the frame-bearing recenter, the primitive-kind tag and the pure
//! signed-distance functions) and on `voxel_core`'s `units` / `spatial_index`; the
//! value crate never names anything here. That ⊥ is compile-enforced by the crate
//! boundary: `voxel_core` cannot import the document layer.

use voxel_core::voxel::VoxelGrid;

// The conservative cell-interval bound and its coarse classification are pure interval
// arithmetic under CSG lattice ops. The domain reads it with the occupancy convention
// "inside where `field <= SURFACE_ISOLEVEL`": `FieldInterval::classify(SURFACE_ISOLEVEL)`
// yields AIR / COARSE-SOLID / BOUNDARY for a whole block-sized cell, and
// `substrate::interval::union_field_intervals` composes a Union of producers
// (min-of-fields). The conservative-never-narrow property is why a coarse verdict can
// never disagree with a brute-force per-voxel evaluation — the boundary-residency
// classifier's soundness (see `docs/architecture/02-evaluation.md`).
pub use substrate::interval::{FieldClassification, FieldInterval};

/// Anything that can resolve itself into the shared [`VoxelGrid`].
///
/// The trait is the renderer's only door onto geometry, so a new kind of body (a sketch
/// solid, a composed scope, a sculpt overlay) reaches the display without the renderer
/// learning about it.
// `Send + Sync`: every implementor is plain immutable data, so a boxed producer can be
// SHARED read-only across rayon threads. The chunk build computes the leaf list ONCE and
// shares the boxed producers across the parallel per-chunk build — this bound is what lets
// `&[LeafProducer]` be `Sync`.
pub trait VoxelProducer: Send + Sync {
    /// Write occupied voxels into `grid`. The grid's `dimensions` are assumed to
    /// already be set by the caller (so multiple producers can target one grid).
    /// `voxels_per_block` is the document-level density (one grid fineness for the whole
    /// plan) — used to fill each voxel's `block_local_coord` (and, for a sized producer,
    /// its grid extent).
    ///
    /// This is the full-window convenience wrapper over [`resolve_into`]: each impl
    /// computes its own FULL grid dimensions and calls `resolve_into` with the window
    /// `[0, full_dim)` on every axis, so it writes EVERY in-range cell.
    ///
    /// [`resolve_into`]: VoxelProducer::resolve_into
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32);

    /// Resolve only the cells whose LOCAL voxel index lies inside `window_local_voxels`
    /// (a half-open `[min, max)` box in the producer's own voxel-index frame
    /// `[0, full_dim)`), writing JUST those in-window cells into `grid.occupied`.
    ///
    /// Two invariants every implementor upholds (so a windowed resolve is a
    /// byte-identical SUBSET of the full resolve):
    ///
    /// * **`grid.dimensions` is ALWAYS the producer's FULL dimensions**, never the
    ///   window size. Downstream decode (`widest_run_in_band`, the 2D slice, `.vox`
    ///   export) recover indices against the full extent, so the dimensions must
    ///   describe the whole producer even when only a sub-region's cells are written.
    /// * Each impl **CLAMPs** the window to `[0, full_dim)` per axis before iterating,
    ///   so an oversized / partly-out-of-range window is harmless and a full-window
    ///   call (`[0,0,0]..full_dim`) reproduces the full resolve EXACTLY.
    ///
    /// Every producer's per-cell output depends ONLY on the cell index and the FULL
    /// dimensions (centered sample `idx + 0.5 − full_dim/2`; corner-anchored store
    /// `idx + 0.5`; revolve radius/axial from the full extent; cloud puffs scattered
    /// from the full extent) — never on which window is being filled. So restricting
    /// the iteration to `window ∩ [0, full_dim)` produces a byte-identical subset.
    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    );

    /// CONSERVATIVE bound on the producer's SIGNED field over a block-sized cell — the
    /// primitive the chunk classifier elides whole cells with. `cell_local_voxels` is a
    /// half-open `[min, max)` box in the producer's OWN local voxel-index frame
    /// `[0, full_dim)` — the SAME frame [`resolve_into`]'s window uses; the frame is
    /// carried, never re-derived.
    ///
    /// Returns `Some([minimum, maximum])` whenever the producer can bracket its field
    /// over the whole cell (see [`FieldInterval`] for the conservative-never-narrow
    /// rule), or `None` when it cannot (e.g. the fBm-displaced cloud field) — a `None`
    /// consumer treats the cell as BOUNDARY and resolves it per-voxel, still exact, just
    /// unelided.
    ///
    /// The default is `None` (the always-safe fallback): a producer opts INTO coarse
    /// classification by overriding this.
    ///
    /// [`resolve_into`]: VoxelProducer::resolve_into
    fn cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<FieldInterval> {
        let _ = (cell_local_voxels, voxels_per_block);
        None
    }

    /// The material this producer stamps at a point in its own `[0, full_dim)` voxel frame,
    /// for a producer that carries per-voxel materials rather than one override.
    ///
    /// The default `None` means "I have no opinion" — the leaf's single-material override
    /// answers instead, which is the case for every Tool and sketch solid. A
    /// [`CompositeProducer`] overrides it because a composed Part's material varies across
    /// the body, and an outset shell has to inherit the material of the surface it grew from
    /// rather than flattening the Part to one color.
    ///
    /// [`CompositeProducer`]: crate::voxel::CompositeProducer
    fn material_at(
        &self,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> Option<voxel_core::core_geom::BlockId> {
        let _ = (point_local_voxels, voxels_per_block);
        None
    }

    /// Which node authored the geometry at a point in this producer's own `[0, full_dim)`
    /// voxel frame — what a viewport pick landing there selects.
    ///
    /// The default `None` means "I am one node's body, ask the leaf" — the case for every
    /// Tool and sketch solid, whose leaf already names its node. Only a
    /// [`CompositeProducer`] answers: a pre-composed scope is ONE leaf to the walk, so
    /// without this a pick anywhere inside it could name only the scope.
    ///
    /// **The pick follows the material.** The rule here is the one
    /// [`material_at`](VoxelProducer::material_at) uses — last containing `Union` member
    /// inside the body, nearest one out in an outset shell — so the node you select is the
    /// node that colored the voxel you clicked. Two answers here would be two answers to
    /// "whose voxel is this", and the user can see the material.
    ///
    /// [`CompositeProducer`]: crate::voxel::CompositeProducer
    fn origin_at(
        &self,
        point_local_voxels: [f32; 3],
        voxels_per_block: u32,
    ) -> Option<crate::scene::LeafOrigin> {
        let _ = (point_local_voxels, voxels_per_block);
        None
    }

    /// This producer's signed distance field, when it has one.
    ///
    /// `None` is not a failure — it is the honest answer for a producer whose occupancy is
    /// real but whose *geometry* is not a distance. Operations that need to measure (outset,
    /// emboss, displacement) are then unavailable on it, enforced by the type rather than
    /// discovered at runtime.
    fn as_field(&self) -> Option<&dyn Field> {
        None
    }

    /// The producer's ANALYTIC feature-edge polylines in its own `[0, full_dim)` local
    /// voxel frame — only edges the AUTHORED geometry actually has, never anything derived
    /// from the voxel surface. `circle_segments` tessellates one full rim turn (fixed, not
    /// screen-adaptive, so the polyline is world-stable under orbit).
    ///
    /// The default empty answer is honest for a producer with no authored creases: a
    /// voxel body, a composed scope, an outset wrapper (whose dilated surface has left
    /// the authored edges behind).
    fn edge_polylines_local(
        &self,
        voxels_per_block: u32,
        circle_segments: u32,
    ) -> Vec<Vec<[f32; 3]>> {
        let _ = (voxels_per_block, circle_segments);
        Vec::new()
    }

    /// The producer's FULL grid dimensions in voxels (its `[0, full_dim)` local frame).
    /// This is the span [`resolve`] writes into and the AABB the classifier / chunk
    /// window clip against. A sized producer (an SDF Tool, a sketch solid) returns its
    /// intrinsic extent; a region-sized producer (the cloud field) returns the region it
    /// was constructed for. The chunk build reads this to bound each leaf's contribution
    /// to a chunk block.
    ///
    /// [`resolve`]: VoxelProducer::resolve
    fn full_dimensions(&self, voxels_per_block: u32) -> [u32; 3];
}

/// Clamp a producer window to `[0, full_dim)` per axis and return the per-axis
/// iteration bounds `[lo, hi)` as `u32` (already intersected with the grid). When the
/// window lies fully outside the grid on any axis the returned range is EMPTY
/// (`lo >= hi`), so the iteration writes nothing. Shared by every `resolve_into`.
#[inline]
pub(crate) fn clamp_window_to_grid(
    window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    full_dimensions: [u32; 3],
) -> [(u32, u32); 3] {
    let mut bounds = [(0u32, 0u32); 3];
    for axis in 0..3 {
        let full = full_dimensions[axis] as i64;
        let lo = window_local_voxels.min[axis].clamp(0, full) as u32;
        let hi = window_local_voxels.max[axis].clamp(0, full) as u32;
        // `hi >= lo` always holds after clamping a half-open box to a non-negative
        // range, but a degenerate (min > max) input box could invert — guard it so
        // the range is never reversed (which would panic the `par_iter`).
        bounds[axis] = (lo, hi.max(lo));
    }
    bounds
}

/// The metric Lipschitz bracket of a field over a cell — the ONE bracket the composite,
/// outset, and sketch producers share for their `cell_field_interval`.
///
/// Occupancy is decided at voxel CENTERS (`index + 0.5`), so the region to bracket is the
/// center span `[min + 0.5, max − 0.5]` — the exact samples `resolve_into` visits, tighter than
/// the whole cell box. The field is sampled at that span's center (via `sample_center`, the only
/// per-producer difference) and widened by the cell circumradius in the field's `metric`, which
/// the field is 1-Lipschitz in ([`Metric::cell_circumradius`](substrate::geom2d::Metric::cell_circumradius)).
/// Callers keep their own empties/no-field guard and any post-refinement.
#[inline]
pub(crate) fn metric_cell_bracket(
    cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
    metric: substrate::geom2d::Metric,
    sample_center: impl FnOnce([f32; 3]) -> f32,
) -> FieldInterval {
    let mut center = [0.0f32; 3];
    let mut half_extent = [0.0f32; 3];
    for axis in 0..3 {
        let low = cell_local_voxels.min[axis] as f32 + 0.5;
        let high = (cell_local_voxels.max[axis] - 1) as f32 + 0.5;
        center[axis] = 0.5 * (low + high);
        half_extent[axis] = 0.5 * (high - low);
    }
    FieldInterval::from_lipschitz_center(
        sample_center(center),
        metric.cell_circumradius(half_extent),
    )
}

mod composite;
mod field;
mod outset;
mod sdf_shape;

pub use composite::{CompositeMember, CompositeProducer};
pub use field::Field;
pub use outset::OutsetProducer;
pub use sdf_shape::{GeometryParams, SdfShape};
