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
//! NO axis swap (MagicaVoxel is itself Z-up).
//!
//! ## The producer seam (`docs/adr/0006-authoring-truth-and-gpu-boundary.md`)
//!
//! This module implements the architectural seam **the renderer never calls the SDF
//! directly**: instead a [`VoxelProducer`] resolves a parametric shape (or a sub-assembly,
//! or a sketch, or a sculpt overlay) into a [`VoxelGrid`] — the one consumed truth. The
//! renderer, the layer-range diameter readout (issue #12) and the `.vox` export all read
//! the grid, so adding a second producer touched nothing downstream — proven out since:
//! [`SdfShape`] (which runs the sampling triple-loop transcribed from the original
//! prototype) was the first implementor, and `SketchSolid`, `DebugCloudField`,
//! `CompositeProducer` and `OutsetProducer` have since joined it.
//!
//! ## The value ⊥ producer split (ADR 0016)
//!
//! This is the **document-bound** producer half. It depends DOWNWARD on the
//! foundational value vocabulary in the `voxel_core` crate (the resolved
//! [`Voxel`](voxel_core::voxel::Voxel),
//! its [`VoxelGrid`], the frame-bearing recentre, the primitive-kind tag and the pure
//! signed-distance functions) and on `voxel_core`'s `units` / `spatial_index`; the
//! value crate never names anything here. That ⊥ is compile-enforced by the crate
//! boundary: `voxel_core` cannot import the document layer.

use voxel_core::voxel::VoxelGrid;

// The conservative cell-interval bound and its coarse classification are pure interval
// arithmetic under CSG lattice ops — substrate's [`substrate::interval::FieldInterval`]. The
// domain reads it with the occupancy convention "inside where `field <= SURFACE_ISOLEVEL`":
// `FieldInterval::classify(SURFACE_ISOLEVEL)` yields AIR / COARSE-SOLID / BOUNDARY for a
// whole block-sized cell, and `substrate::interval::union_field_intervals` composes a Union of producers
// (min-of-fields). The conservative-never-narrow property is why a coarse verdict can
// never disagree with a brute-force per-voxel evaluation — the boundary-residency
// classifier's soundness (see the Boundary-residency material in
// `docs/architecture/02-evaluation.md`, proven by the E1 parity gate in
// `cell_interval_parity_tests`). The interval algebra, the Lipschitz-centre bound, and
// the classify threshold-parameter live in the substrate module doc.
pub use substrate::interval::{FieldClassification, FieldInterval};

/// Anything that can resolve itself into the shared [`VoxelGrid`].
///
/// v1 has a single implementor ([`SdfShape`]); the trait exists so a sculpt
/// overlay (the sparse-override option, `docs/adr/0003` §3g) can be added later
/// without changing the renderer.
// `Send + Sync`: every implementor ([`SdfShape`], the sketch producer, [`DebugCloudField`])
// is plain immutable data, so a boxed producer can be SHARED read-only across rayon threads.
// The #63 hoisted two-layer build computes the leaf list ONCE and shares the boxed producers
// across the parallel per-chunk build — this bound is what lets `&[LeafProducer]` be `Sync`.
pub trait VoxelProducer: Send + Sync {
    /// Write occupied voxels into `grid`. The grid's `dimensions` are assumed to
    /// already be set by the caller (so multiple producers can target one grid).
    /// `voxels_per_block` is the document-level density (ADR 0003 §3f(0): one grid
    /// fineness for the whole plan, no longer a per-producer field) — used to fill
    /// each voxel's `block_local_coord` (and, for a sized producer, its grid extent).
    ///
    /// This is the full-window convenience wrapper over [`resolve_into`]: each impl
    /// computes its own FULL grid dimensions and calls `resolve_into` with the window
    /// `[0, full_dim)` on every axis. It therefore writes EVERY in-range cell — i.e.
    /// it is exactly the historical (pre-windowing) resolve.
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
    ///   call (`[0,0,0]..full_dim`) reproduces the historical resolve EXACTLY.
    ///
    /// Every producer's per-cell output depends ONLY on the cell index and the FULL
    /// dimensions (centred sample `idx + 0.5 − full_dim/2`; corner-anchored store
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
    /// classification primitive of ADR 0010 Decision 2 (the E1 slice). `cell_local_voxels`
    /// is a half-open `[min, max)` box in the producer's OWN local voxel-index frame
    /// `[0, full_dim)` (the SAME frame [`resolve_into`]'s window uses, ADR 0008 — the
    /// frame is carried, never re-derived).
    ///
    /// Returns `Some([minimum, maximum])` whenever the producer can bracket its field
    /// over the whole cell (see [`FieldInterval`] for the conservative-never-narrow
    /// rule), or `None` when it cannot (e.g. the fBm-displaced cloud field) — a `None`
    /// consumer treats the cell as BOUNDARY and resolves it per-voxel, still exact, just
    /// unelided.
    ///
    /// The default is `None` (the always-safe fallback): a producer opts INTO coarse
    /// classification by overriding this. Wired to nothing yet (E1 stands alone with its
    /// own exactness gate); it is op-stack math independent of any payload change.
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
    /// rather than flattening the Part to one colour.
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

    /// This producer's signed distance field, when it has one (ADR 0020 Decision 1).
    ///
    /// `None` is not a failure — it is the honest answer for a producer whose occupancy is
    /// real but whose *geometry* is not a distance. Operations that need to measure (outset,
    /// emboss, displacement) are then unavailable on it, enforced by the type rather than
    /// discovered at runtime.
    fn as_field(&self) -> Option<&dyn Field> {
        None
    }

    /// The producer's FULL grid dimensions in voxels (its `[0, full_dim)` local frame).
    /// This is the span [`resolve`] writes into and the AABB the classifier / chunk
    /// window clip against. A sized producer (an SDF Tool, a sketch solid) returns its
    /// intrinsic extent; a region-sized producer (the cloud field) returns the region it
    /// was constructed for. ADR 0010 E2 reads this to bound each leaf's contribution to a
    /// chunk block.
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

mod composite;
mod field;
mod outset;
mod sdf_shape;

pub use composite::{CompositeMember, CompositeProducer};
pub use field::Field;
pub use outset::OutsetProducer;
pub use sdf_shape::{GeometryParams, SdfShape};
