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
//! ## The producer seam (`REPRESENTATION.md`)
//!
//! This module implements the architectural seam **the renderer never calls the SDF
//! directly**: instead a [`VoxelProducer`] resolves a parametric shape (or, later, a
//! sculpt overlay) into a [`VoxelGrid`] — the one consumed truth. The renderer, the
//! layer-range diameter readout (issue #12) and the `.vox` export all read the grid,
//! so adding a second producer touches nothing downstream. The first (and, in M2,
//! only) implementor is [`SdfShape`], which runs the sampling triple-loop transcribed
//! from `ARCHITECTURE.md` §1/§2.
//!
//! ## The value ⊥ producer split (ADR 0016)
//!
//! This is the **document-bound** producer half. It depends DOWNWARD on the
//! foundational value vocabulary in the `voxel_core` crate (the resolved [`Voxel`],
//! its [`VoxelGrid`], the frame-bearing recentre, the primitive-kind tag and the pure
//! signed-distance functions) and on `voxel_core`'s `units` / `spatial_index`; the
//! value crate never names anything here. That ⊥ is compile-enforced by the crate
//! boundary: `voxel_core` cannot import the document layer.

use glam::Vec3;
use rayon::prelude::*;

use voxel_core::voxel::{
    signed_distance, BlockAttrs, BlockId, ShapeKind, Voxel, VoxelGrid, MAX_GRID_VOXELS,
    SURFACE_ISOLEVEL,
};

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
/// overlay (REPRESENTATION.md option 2) can be added later without changing the
/// renderer.
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

/// Geometry parameters — the *only* params that trigger a voxel rebuild.
///
/// The UI-side mirror of [`SdfShape`] (the panel edits this; `SdfShape::from_geometry`
/// turns it into a producer).
///
/// **Size is voxel-granular** (ADR 0003 §3f(0)): the canonical [`size_voxels`] is the
/// bounding-box span in VOXELS at the document density, and [`size_measurements`]
/// retains the authored blocks+voxels expression the inspector typed (so a density
/// re-target is lossless). A whole-block size has `size_voxels = blocks · d`, so the
/// resolved geometry is identical to the old block-granular path.
///
/// `voxels_per_block` is the **transient UI control value** for the density slider
/// only — density is a document-level attribute on [`Scene`](crate::scene::Scene)
/// (ADR 0003 §3f(0)), so this field is mirrored from / written to the scene via
/// [`Intent::SetDensity`](crate::intent::Intent::SetDensity) and is NOT copied onto
/// the produced [`SdfShape`]. Fineness only — it never changes the object's physical
/// size (DATA.md "the density bug").
///
/// [`size_voxels`]: GeometryParams::size_voxels
/// [`size_measurements`]: GeometryParams::size_measurements
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeometryParams {
    /// Selected primitive.
    pub shape: ShapeKind,
    /// Bounding-box size in **voxels** (X, Y, Z) at the document density — the
    /// canonical size the producer resolves (a whole-block size is `blocks · d`).
    pub size_voxels: [u32; 3],
    /// The RETAINED authored size expression per axis (ADR 0003 §3f(0)), or `None`
    /// when the size carries no parametric block expression (a pure-voxel size). The
    /// canonical `size_voxels` always wins for geometry; this is retention/display
    /// only, kept so a density re-target re-evaluates losslessly.
    pub size_measurements: Option<Box<[voxel_core::units::Measurement; 3]>>,
    /// Voxels per block (chisel fineness): the density slider's transient UI value,
    /// mirrored to/from [`Scene::voxels_per_block`](crate::scene::Scene). Default 16.
    pub voxels_per_block: u32,
    /// Tube wall thickness in whole blocks (used by [`ShapeKind::Tube`] only).
    pub wall_blocks: u32,
}

impl Default for GeometryParams {
    fn default() -> Self {
        // Default size 5×1×5 BLOCKS at the default density 16 → voxel-granular canonical.
        Self {
            shape: ShapeKind::Cylinder,
            // 5×1×5 BLOCKS at the default density 16 → voxel-granular canonical.
            size_voxels: [80, 16, 80],
            size_measurements: None,
            voxels_per_block: 16,
            wall_blocks: 1,
        }
    }
}

/// A single parametric SDF primitive: the first (and, in M2, only) producer.
///
/// **Size is voxel-granular** (ADR 0003 §3f(0)): the canonical [`size_voxels`] is
/// the bounding-box span in VOXELS at the document density. Density
/// (`voxels_per_block`) is NOT stored here — it is a document-level attribute on
/// [`Scene`](crate::scene::Scene) (one grid fineness for the whole plan), passed in
/// to the size / resolve methods. A whole-block size is `blocks · d`, so the
/// resolved grid is identical to the old block-granular store (goldens unchanged).
///
/// [`size_measurements`] RETAINS the authored blocks+voxels expression (parametric)
/// alongside the canonical voxels, mirroring
/// [`NodeTransform::offset_measurements`](crate::scene::NodeTransform::offset_measurements):
/// `size_voxels` is the source of truth for ALL geometry / resolve; the retained
/// expression is read only by the inspector (seed/undo) and the density re-target
/// ([`Intent::SetDensity`](crate::intent::Intent::SetDensity)).
///
/// [`size_voxels`]: SdfShape::size_voxels
/// [`size_measurements`]: SdfShape::size_measurements
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SdfShape {
    #[serde(default = "default_shape_kind")]
    pub kind: ShapeKind,
    /// Bounding-box size in **voxels** (X, Y, Z) at the document density — the
    /// canonical span the producer resolves over. Always `>= 1` per axis.
    #[serde(default = "default_shape_size_voxels")]
    pub size_voxels: [u32; 3],
    /// Tube wall thickness in whole blocks (used by [`ShapeKind::Tube`] only).
    #[serde(default = "default_shape_wall")]
    pub wall_blocks: u32,
    /// The RETAINED authored size expression per axis (ADR 0003 §3f(0)).
    ///
    /// `serde(default)` makes this `None` on an OLD document predating the field, so
    /// old scenes still load; the accessor [`size_measurements`](SdfShape::size_measurements)
    /// then SYNTHESISES a pure-voxel measurement from `size_voxels`. Boxed so the
    /// common (`None`) case keeps `SdfShape` small.
    #[serde(default)]
    size_measurements: Option<Box<[voxel_core::units::Measurement; 3]>>,
}

/// Persistence defaults for a partial [`SdfShape`] (a missing field falls back to
/// a sane non-zero value so a tolerant config load never yields a degenerate
/// zero-size shape).
fn default_shape_kind() -> ShapeKind {
    ShapeKind::Cylinder
}
/// The default canonical voxel size for a config load missing `size_voxels`: the
/// historical 5×1×5-block default at the default density 16.
fn default_shape_size_voxels() -> [u32; 3] {
    [80, 16, 80]
}
fn default_shape_wall() -> u32 {
    1
}

/// Clamp a per-axis voxel size so every axis is at least 1 voxel (a 0-voxel axis
/// would resolve an empty / degenerate grid). The UI rejects sub-1 sizes before
/// emitting; this is the constructor-side guard so a `from_*` caller can never
/// build a degenerate shape (ADR 0003 §3f(0)).
fn clamp_size_voxels(size_voxels: [u32; 3]) -> [u32; 3] {
    [size_voxels[0].max(1), size_voxels[1].max(1), size_voxels[2].max(1)]
}

impl SdfShape {
    /// Build the shape from the UI-side [`GeometryParams`].
    ///
    /// This is the single place geometry params become a producer; the split in
    /// `panel.rs` guarantees display/camera params never reach here. The canonical
    /// `size_voxels` and the retained `size_measurements` ride straight across (the
    /// inspector already validated the size lands on a whole voxel ≥ 1). Density is
    /// NOT copied — it lives on the [`Scene`](crate::scene::Scene), not the shape.
    pub fn from_geometry(geometry: GeometryParams) -> Self {
        let size_voxels = clamp_size_voxels(geometry.size_voxels);
        Self {
            kind: geometry.shape,
            size_voxels,
            wall_blocks: geometry.wall_blocks,
            size_measurements: Self::retained_or_none(geometry.size_measurements, size_voxels),
        }
    }

    /// Build a shape from a whole-**block** size at density `voxels_per_block`
    /// (`size_voxels = blocks · d`). The terse whole-block entry point for demos,
    /// tests and `GroupSpec` placement (mirrors
    /// [`NodeTransform::from_blocks`](crate::scene::NodeTransform::from_blocks)). It
    /// retains each axis as a whole-block measurement so a later density re-target
    /// scales it losslessly. Each axis is clamped to `>= 1` block.
    pub fn from_blocks(
        kind: ShapeKind,
        size_blocks: [u32; 3],
        wall_blocks: u32,
        voxels_per_block: u32,
    ) -> Self {
        use voxel_core::units::{ExactRational, Measurement};
        let density = voxels_per_block.max(1);
        let blocks = [size_blocks[0].max(1), size_blocks[1].max(1), size_blocks[2].max(1)];
        let size_voxels =
            clamp_size_voxels([blocks[0] * density, blocks[1] * density, blocks[2] * density]);
        let measurements = [
            Measurement::new(ExactRational::from_integer(blocks[0] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[1] as i128), 0),
            Measurement::new(ExactRational::from_integer(blocks[2] as i128), 0),
        ];
        Self {
            kind,
            size_voxels,
            wall_blocks,
            size_measurements: Self::retained_or_none(Some(Box::new(measurements)), size_voxels),
        }
    }

    /// Build a shape from a pure-**voxel** size with NO retained authored expression
    /// (the synthesis / integer-rescale path — e.g. an old document, or a density
    /// re-target of a size that had no parametric block expression). Each axis is
    /// clamped to `>= 1` voxel. The retained field stays `None`, so its measurement
    /// is synthesised from `size_voxels` (re-evaluates to the same voxels at any
    /// density). Mirrors the `from_voxels` synthesis on the offset side.
    pub fn from_voxels(kind: ShapeKind, size_voxels: [u32; 3], wall_blocks: u32) -> Self {
        Self {
            kind,
            size_voxels: clamp_size_voxels(size_voxels),
            wall_blocks,
            size_measurements: None,
        }
    }

    /// Build a shape from a per-axis authored [`Measurement`](voxel_core::units::Measurement)
    /// size at density `voxels_per_block` (ADR 0003 §3f(0)). The canonical voxel size
    /// is DERIVED via [`Measurement::to_voxels`](voxel_core::units::Measurement::to_voxels)
    /// and clamped to `>= 1`; the measurements are RETAINED for lossless density
    /// re-targeting. Mirrors
    /// [`NodeTransform::from_measurements`](crate::scene::NodeTransform::from_measurements),
    /// including the self-consistency rule: a non-landing axis floors AND
    /// resynthesises its retained measurement to the pure-voxel form, so
    /// `size_voxels` and the retained expression never disagree. (A size that floors
    /// below 1 voxel is clamped to 1 and resynthesised to the pure-voxel `1`.)
    pub fn from_measurements(
        kind: ShapeKind,
        measurements: [voxel_core::units::Measurement; 3],
        wall_blocks: u32,
        voxels_per_block: u32,
    ) -> Self {
        use voxel_core::units::{Measurement, MeasurementError};
        let resolve_axis = |measurement: Measurement| -> (u32, Measurement) {
            let raw = match measurement.to_voxels(voxels_per_block) {
                Ok(voxels) => (voxels, Some(measurement)),
                Err(MeasurementError::BlockTermNotWholeVoxels { nearest_floor_voxels, .. }) => {
                    (nearest_floor_voxels, None)
                }
                Err(MeasurementError::ZeroDensity) => (measurement.voxel_term(), None),
            };
            // A size must be at least 1 voxel: clamp negatives / zero up to 1. If the
            // authored measurement landed cleanly AND is >= 1 keep it verbatim; any
            // floor or clamp resynthesises to the pure-voxel form of the final value.
            let clamped = raw.0.max(1) as u32;
            let landed_exact = raw.1.is_some() && raw.0 == clamped as i64;
            if landed_exact {
                (clamped, measurement)
            } else {
                (clamped, Measurement::from_voxels(clamped as i64))
            }
        };
        let (vx_x, m_x) = resolve_axis(measurements[0]);
        let (vx_y, m_y) = resolve_axis(measurements[1]);
        let (vx_z, m_z) = resolve_axis(measurements[2]);
        let size_voxels = [vx_x, vx_y, vx_z];
        Self {
            kind,
            size_voxels,
            wall_blocks,
            size_measurements: Self::retained_or_none(Some(Box::new([m_x, m_y, m_z])), size_voxels),
        }
    }

    /// Normalise the retained measurements to `None` when every axis is exactly the
    /// pure-voxel measurement of its derived voxels — i.e. there is NO parametric
    /// block content beyond the voxel count. Keeps a pure-voxel size in the same
    /// canonical form as a freshly-loaded shape (`None`) so apply→undo is
    /// byte-identical and serde gains no redundant husk. Mirrors
    /// `NodeTransform::retained_or_none`.
    fn retained_or_none(
        measurements: Option<Box<[voxel_core::units::Measurement; 3]>>,
        size_voxels: [u32; 3],
    ) -> Option<Box<[voxel_core::units::Measurement; 3]>> {
        use voxel_core::units::Measurement;
        let measurements = measurements?;
        let is_synthesisable = (0..3)
            .all(|axis| measurements[axis] == Measurement::from_voxels(size_voxels[axis] as i64));
        if is_synthesisable {
            None
        } else {
            Some(measurements)
        }
    }

    /// The RETAINED per-axis authored size measurement (ADR 0003 §3f(0)). When the
    /// shape carries no stored expression (an OLD scene, or a pure-voxel size), this
    /// SYNTHESISES a pure-voxel measurement equal to `size_voxels` per axis (correct
    /// at any density, just non-parametric). Mirrors
    /// `NodeTransform::offset_measurements`.
    pub fn size_measurements(&self) -> [voxel_core::units::Measurement; 3] {
        use voxel_core::units::Measurement;
        match &self.size_measurements {
            Some(measurements) => **measurements,
            None => [
                Measurement::from_voxels(self.size_voxels[0] as i64),
                Measurement::from_voxels(self.size_voxels[1] as i64),
                Measurement::from_voxels(self.size_voxels[2] as i64),
            ],
        }
    }

    /// Whether this shape carries a GENUINELY retained authored size expression
    /// (the stored field is `Some`) versus a pure-voxel size whose measurement is
    /// only SYNTHESISED. The density re-target uses this to choose between
    /// re-evaluating the authored block expression and an integer rescale that
    /// preserves physical size. Mirrors `NodeTransform::has_retained_measurements`.
    pub fn has_retained_size_measurements(&self) -> bool {
        self.size_measurements.is_some()
    }

    /// Grid dimensions in voxels: the canonical `size_voxels` directly (ADR 0003
    /// §3f(0); size is now voxel-granular, so density no longer scales it here — a
    /// whole-block size already stored `blocks · d`). The `voxels_per_block` argument
    /// is retained for call-site symmetry but unused.
    pub fn grid_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        let _ = voxels_per_block;
        self.size_voxels
    }

    /// Total number of sampling-grid voxels (`grid_x * grid_y * grid_z`), as
    /// `u64` so it can't overflow at large sizes/densities.
    pub fn grid_voxel_count(&self, voxels_per_block: u32) -> u64 {
        let [grid_x, grid_y, grid_z] = self.grid_dimensions(voxels_per_block);
        grid_x as u64 * grid_y as u64 * grid_z as u64
    }

    /// Whether this shape's sampling grid exceeds [`MAX_GRID_VOXELS`] and so the
    /// 3D rebuild should be skipped (ARCHITECTURE.md §7).
    pub fn exceeds_voxel_cap(&self, voxels_per_block: u32) -> bool {
        self.grid_voxel_count(voxels_per_block) > MAX_GRID_VOXELS
    }
}

impl VoxelProducer for SdfShape {
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [full_x, full_y, full_z] = self.grid_dimensions(voxels_per_block);
        self.resolve_into(
            grid,
            voxels_per_block,
            voxel_core::spatial_index::VoxelAabb::new(
                [0, 0, 0],
                [full_x as i64, full_y as i64, full_z as i64],
            ),
        );
    }

    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: voxel_core::spatial_index::VoxelAabb,
    ) {
        profiling::scope!("sdf_resolve");
        let [grid_x, grid_y, grid_z] = self.grid_dimensions(voxels_per_block);
        // FULL dimensions even when only a window is written (downstream decode /
        // slice / export read against the whole producer extent).
        grid.dimensions = [grid_x, grid_y, grid_z];

        // Shape inscribed in the box: semi-axes are half the voxel-space dims. ALL
        // per-cell math is derived from the FULL dims — the window only narrows the
        // iteration range, never the sampling frame.
        let semi_axes = Vec3::new(
            grid_x as f32 / 2.0,
            grid_y as f32 / 2.0,
            grid_z as f32 / 2.0,
        );
        let wall_voxels = (self.wall_blocks * voxels_per_block) as f32;

        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;

        // Clamp the window to `[0, full_dim)`; a full-window call reproduces the
        // historical `0..grid_*` loops exactly.
        let [(win_x_lo, win_x_hi), (win_y_lo, win_y_hi), (win_z_lo, win_z_hi)] =
            clamp_window_to_grid(window_local_voxels, [grid_x, grid_y, grid_z]);

        // The outer `j` slices are order-independent (each samples a disjoint set
        // of voxels and writes nothing shared), so M8 parallelises them with
        // rayon: each slice produces a local `Vec<Voxel>` and the results are
        // concatenated. The voxel ORDER may differ from the serial version, but
        // the SET is identical — the renderer doesn't care about order, and the
        // 2D slice / `.vox` export recover indices from each voxel's position.
        // Windowing parallelises over the WINDOWED outer-axis range.
        let kind = self.kind;
        grid.occupied = (win_y_lo..win_y_hi)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut local = Vec::new();
                for k in win_z_lo..win_z_hi {
                    for i in win_x_lo..win_x_hi {
                        // The shape geometry is still inscribed symmetric about the
                        // grid's centre, so SAMPLE the SDF at the centred coordinate
                        // (`idx + 0.5 − grid/2`). But STORE the voxel CORNER-ANCHORED
                        // (`idx + 0.5`): the local occupied span is `[0, grid)` and the
                        // centre is a HALF-INTEGER for any grid size, so it always sits
                        // inside its voxel cell `[idx, idx+1)` — on the global voxel
                        // lattice at any parity. (Was centred at `idx + 0.5 − grid/2`,
                        // which lands on integers for an odd grid and straddles cells.)
                        let sample = Vec3::new(
                            i as f32 + 0.5 - half_x,
                            j as f32 + 0.5 - half_y,
                            k as f32 + 0.5 - half_z,
                        );

                        if signed_distance(kind, sample, semi_axes, wall_voxels)
                            <= SURFACE_ISOLEVEL
                        {
                            local.push(Voxel {
                                local_index: [i as i32, j as i32, k as i32],
                                block_local_coord: [
                                    (i % voxels_per_block) as u8,
                                    (j % voxels_per_block) as u8,
                                    (k % voxels_per_block) as u8,
                                ],
                                block_id: BlockId::DEFAULT,
                                attrs: BlockAttrs::DEFAULT,
                                grid_overlay: false,
                            });
                        }
                    }
                }
                local
            })
            .collect();
    }

    /// Conservative 1-Lipschitz field interval over a cell (ADR 0010 Decision 2). The
    /// resolve samples the SDF at the CENTRED coordinate `idx + 0.5 − full_dim/2`, so
    /// this maps the cell box (local voxel-index frame, ADR 0008) into that SAME centred
    /// frame, evaluates the field at the cell's geometric centre, and brackets the
    /// variation over the cell by the (widened) circumradius.
    ///
    /// `signed_distance_box` and the torus SDF are exactly 1-Lipschitz, but the IQ
    /// ellipsoid and the elliptical-cylinder/tube SDFs have gradient magnitude up to
    /// the semi-axis ANISOTROPY `max_semi / min_semi` (≥ 1; = 1 for an isotropic shape).
    /// To stay conservative for EVERY kind we WIDEN the circumradius by that anisotropy
    /// factor — never narrower than the true field range, so a coarse AIR/SOLID verdict
    /// can never misclassify (proven by the E1 parity gate).
    fn cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        voxels_per_block: u32,
    ) -> Option<FieldInterval> {
        if cell_local_voxels.is_empty() {
            return None;
        }
        let [grid_x, grid_y, grid_z] = self.grid_dimensions(voxels_per_block);
        let semi_axes = Vec3::new(grid_x as f32 / 2.0, grid_y as f32 / 2.0, grid_z as f32 / 2.0);
        let wall_voxels = (self.wall_blocks * voxels_per_block) as f32;
        let half = semi_axes;

        // The cell's geometric centre in the producer's CENTRED sampling frame: a cell
        // sample at integer index `idx` sits at `idx + 0.5 − half`, so the centre of the
        // half-open cell box `[min, max)` is `(min + max) / 2 − half`.
        let center = Vec3::new(
            (cell_local_voxels.min[0] + cell_local_voxels.max[0]) as f32 / 2.0 - half.x,
            (cell_local_voxels.min[1] + cell_local_voxels.max[1]) as f32 / 2.0 - half.y,
            (cell_local_voxels.min[2] + cell_local_voxels.max[2]) as f32 / 2.0 - half.z,
        );

        // Circumradius = half the cell's space-diagonal. The brute-force seam SAMPLES
        // each voxel at its own centre `idx + 0.5 − half`, so the farthest sample from
        // the cell centre is half the diagonal across the SPAN OF SAMPLE CENTRES — which
        // is `(extent − 1)` voxels per axis. Using the full extent (`extent`) is strictly
        // wider, so we keep it: a wider radius is always conservative.
        let extent = Vec3::new(
            (cell_local_voxels.max[0] - cell_local_voxels.min[0]) as f32,
            (cell_local_voxels.max[1] - cell_local_voxels.min[1]) as f32,
            (cell_local_voxels.max[2] - cell_local_voxels.min[2]) as f32,
        );
        let circumradius = (extent * 0.5).length();

        // Conservative Lipschitz constant. Always >= the true constant ⇒ never narrows.
        let lipschitz_constant = match self.kind {
            // The elliptical CYLINDER and TUBE are exactly 1-Lipschitz, so they belong here
            // with the box and torus rather than carrying the anisotropy widening (issue #62).
            // The radial term is `(k − 1)·m` with `k = |(x/ax, y/ay)|` and `m = min(ax, ay)`;
            // writing `u = (x/ax, y/ay)`, its gradient is
            //     |∇k| = |(uₓ/ax, u_y/ay)| / |u| ≤ max(1/ax, 1/ay) = 1/m
            // so `|∇radial| = m·|∇k| ≤ 1` — the `min(ax, ay)` scale factor exactly cancels the
            // worst-case gradient along the SHORTER cross-section axis, which is precisely
            // where the old widening feared it steepened. The axial term `|z| − half_height`
            // is 1-Lipschitz outright, and `max` / `min` / the positive-part norm / negation
            // all preserve the constant — so the tube's `outer.max(−inner)` is 1-Lipschitz too.
            //
            // Empirically confirmed before the change: the constant these kinds actually
            // REQUIRE measures 0.93–1.00 across anisotropies to 32:1, never above 1. The old
            // `max_semi / min_semi` was over-conservative by exactly the anisotropy factor
            // (8–33× headroom), which is what suppressed interior elision for long cylinders.
            ShapeKind::Box | ShapeKind::Torus | ShapeKind::Cylinder | ShapeKind::Tube => 1.0,
            // The IQ ellipsoid is a genuine APPROXIMATION, not a true distance field, and its
            // gradient really does blow up deep inside a thin shape — measured at 277 against
            // a claimed 32 for a 32:1 ellipsoid, i.e. this widening is ALREADY an
            // under-estimate. It survives on magnitude dominance (the field's whole range is
            // bounded by the minor semi-axis while `L·R` scales with the major one), which the
            // `strongly_anisotropic_sdf_cells_stay_sound_where_lipschitz_is_underestimated`
            // parity test pins. Do NOT tighten this one; if anything it wants widening.
            ShapeKind::Sphere => {
                let largest = semi_axes.x.max(semi_axes.y).max(semi_axes.z);
                let smallest = semi_axes.x.min(semi_axes.y).min(semi_axes.z);
                if smallest > 0.0 {
                    (largest / smallest).max(1.0)
                } else {
                    // A degenerate zero-thickness axis: fall back to BOUNDARY (None) — we
                    // cannot bound the gradient, so let the per-voxel seam decide.
                    return None;
                }
            }
        };

        let field_at_center = signed_distance(self.kind, center, semi_axes, wall_voxels);
        Some(FieldInterval::from_lipschitz_center(
            field_at_center,
            circumradius * lipschitz_constant,
        ))
    }

    fn full_dimensions(&self, voxels_per_block: u32) -> [u32; 3] {
        self.grid_dimensions(voxels_per_block)
    }
}

/// ADR 0003 §3f(0): voxel-granular Size with parametric Measurement retention,
/// mirroring the Offset tests in `scene.rs`. These pin the canonical
/// `size_voxels`, the retained-expression round-trip, the density re-target, serde
/// back-compat, and (the high-risk area) the occupied-voxel set / centring at
/// ODD / EVEN / MIXED-parity voxel-granular sizes.
#[cfg(test)]
mod sdf_size_units_tests {
    use super::*;
    use voxel_core::units::{DisplayUnit, ExactRational, Measurement};

    /// A whole-**block** size built via `from_blocks` derives `size_voxels =
    /// blocks · d` (byte-identical to the OLD block-granular store), and retains
    /// each axis as a whole-block measurement so a density re-target is lossless.
    #[test]
    fn from_blocks_matches_legacy_block_size() {
        let shape = SdfShape::from_blocks(ShapeKind::Box, [5, 1, 5], 1, 16);
        assert_eq!(shape.size_voxels, [80, 16, 80], "blocks · d, identical to the old store");
        // grid_dimensions returns the canonical voxels directly.
        assert_eq!(shape.grid_dimensions(16), [80, 16, 80]);
        // The retained expression re-evaluates losslessly at a denser document.
        let dense = SdfShape::from_measurements(ShapeKind::Box, shape.size_measurements(), 1, 32);
        assert_eq!(dense.size_voxels, [160, 32, 160], "5 blocks · 32 = 160 (lossless block refine)");
    }

    /// `from_measurements` derives the canonical voxel size from a per-axis authored
    /// expression and retains it. `3.5 blocks` lands on `3.5·d`; a `2 blocks 8
    /// voxels` axis is `2·d + 8`; a pure-voxel axis is exact.
    #[test]
    fn from_measurements_derives_voxels_and_retains_expression() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(83),                           // 83 voxels (odd, pure-voxel)
            Measurement::new(ExactRational::from_integer(2), 8),    // 2 blocks 8 voxels
        ];
        let shape = SdfShape::from_measurements(ShapeKind::Box, measurements, 1, 16);
        assert_eq!(shape.size_voxels, [56, 83, 40]);
        assert_eq!(shape.size_measurements(), measurements, "expression retained verbatim");
        assert!(shape.has_retained_size_measurements());
        // The SAME measurements refine at a denser document: 3.5·32 = 112; the
        // pure-voxel 83 stays 83; 2·32 + 8 = 72.
        let dense = SdfShape::from_measurements(ShapeKind::Box, measurements, 1, 32);
        assert_eq!(dense.size_voxels, [112, 83, 72]);
    }

    /// A `2 blocks 8 voxels` size (56 vx at d16) re-evaluated at the integer-multiple
    /// d32 keeps the VOXEL TERM EXACT: 2·32 + 8 = 72, NOT the integer rescale 112.
    #[test]
    fn from_measurements_integer_multiple_density_keeps_voxel_term_exact() {
        let measurements = [
            Measurement::new(ExactRational::from_integer(2), 8), // 2 blocks 8 voxels
            Measurement::from_voxels(16),
            Measurement::from_voxels(16),
        ];
        let at16 = SdfShape::from_measurements(ShapeKind::Box, measurements, 1, 16);
        assert_eq!(at16.size_voxels[0], 40);
        let at32 = SdfShape::from_measurements(ShapeKind::Box, at16.size_measurements(), 1, 32);
        assert_eq!(at32.size_voxels[0], 72, "2·32 + 8, NOT the integer rescale 80");
        assert_eq!(at32.size_measurements()[0], measurements[0], "expression preserved");
    }

    /// A `3.5 blocks` size re-evaluated at the NON-dividing d15 (3.5·15 = 52.5) must
    /// not panic, floors to a whole voxel, and resynthesises its retained measurement
    /// to stay CONSISTENT with `size_voxels` (the self-consistency rule).
    #[test]
    fn from_measurements_non_dividing_density_stays_self_consistent() {
        let measurements = [
            Measurement::new(ExactRational::new(7, 2).unwrap(), 0), // 3.5 blocks
            Measurement::from_voxels(16),
            Measurement::from_voxels(16),
        ];
        let at15 = SdfShape::from_measurements(ShapeKind::Box, measurements, 1, 15);
        assert_eq!(at15.size_voxels[0], 52, "3.5·15 = 52.5 floored to 52, no panic");
        let retained = at15.size_measurements();
        assert_eq!(
            retained[0].to_voxels(15).unwrap(),
            at15.size_voxels[0] as i64,
            "retained measurement must agree with the floored canonical voxels"
        );
    }

    /// Size must be at least 1 voxel: a 0 / negative / sub-1 authored size clamps to
    /// 1 voxel and resynthesises to the pure-voxel `1` (the constructor-side guard).
    #[test]
    fn size_clamps_to_at_least_one_voxel() {
        // A `0 voxels` axis clamps to 1.
        let zero = SdfShape::from_measurements(
            ShapeKind::Box,
            [Measurement::from_voxels(0), Measurement::from_voxels(5), Measurement::from_voxels(5)],
            1,
            16,
        );
        assert_eq!(zero.size_voxels[0], 1, "0-voxel axis clamps up to 1");
        assert_eq!(zero.size_measurements()[0], Measurement::from_voxels(1));
        // `from_blocks` with a 0-block axis clamps to 1 block.
        let zero_block = SdfShape::from_blocks(ShapeKind::Box, [0, 2, 2], 1, 16);
        assert_eq!(zero_block.size_voxels[0], 16, "0 blocks clamps to 1 block = 16 voxels");
        // `from_voxels` clamps too.
        let pure = SdfShape::from_voxels(ShapeKind::Box, [0, 0, 0], 1);
        assert_eq!(pure.size_voxels, [1, 1, 1]);
    }

    /// A pure-voxel size (no parametric block term) normalises its retained field to
    /// `None`, so it is in the same canonical form as a freshly-loaded shape and
    /// serde gains no redundant husk.
    #[test]
    fn pure_voxel_size_retains_none() {
        let pure = SdfShape::from_measurements(
            ShapeKind::Box,
            [Measurement::from_voxels(83), Measurement::from_voxels(17), Measurement::from_voxels(80)],
            1,
            16,
        );
        assert!(!pure.has_retained_size_measurements(), "pure-voxel size is synthesisable → None");
        // The accessor still synthesises the correct per-axis pure-voxel measurement.
        assert_eq!(pure.size_measurements()[0], Measurement::from_voxels(83));
    }

    /// parse(format(size)) round-trips for voxel-granular sizes through the
    /// blocks+voxels display the Size panel uses.
    #[test]
    fn size_format_parse_round_trips() {
        for voxels in [1_i64, 16, 56, 80, 83, 257] {
            let text = voxel_core::units::format(voxels, 16, DisplayUnit::BlocksAndVoxels);
            let reparsed = voxel_core::units::parse(&text).expect("re-parses");
            assert_eq!(reparsed.to_voxels(16).unwrap(), voxels, "round-trip via `{text}`");
        }
    }

    /// An OLD `SdfShape` JSON predating `size_measurements` (and even predating
    /// `size_voxels`, carrying the legacy `size_blocks`... NO — the legacy field is
    /// gone; the realistic old-document shape carries `size_voxels` but NO
    /// `size_measurements`) deserialises (serde default → `None`) and the accessor
    /// synthesises a pure-voxel measurement from `size_voxels`.
    #[test]
    fn serde_back_compat_synthesises_measurements_from_voxels() {
        let old_json = r#"{ "kind": "Box", "size_voxels": [83, 17, 80], "wall_blocks": 1 }"#;
        let restored: SdfShape =
            serde_json::from_str(old_json).expect("old shape without size_measurements must load");
        assert_eq!(restored.size_voxels, [83, 17, 80]);
        assert!(!restored.has_retained_size_measurements());
        for (axis, &voxels) in restored.size_voxels.iter().enumerate() {
            assert_eq!(restored.size_measurements()[axis], Measurement::from_voxels(voxels as i64));
        }
    }

    /// A shape carrying retained size measurements round-trips through serde
    /// unchanged (the new field persists for a forward-saved document).
    #[test]
    fn serde_round_trips_with_retained_size() {
        let shape = SdfShape::from_measurements(
            ShapeKind::Box,
            [
                Measurement::new(ExactRational::new(7, 2).unwrap(), 0),
                Measurement::from_voxels(17),
                Measurement::new(ExactRational::from_integer(2), 8),
            ],
            1,
            16,
        );
        let json = serde_json::to_string(&shape).expect("serialises");
        let restored: SdfShape = serde_json::from_str(&json).expect("deserialises");
        assert_eq!(restored, shape);
        assert_eq!(restored.size_measurements(), shape.size_measurements());
    }

    /// Resolve a Box of the given canonical VOXEL size at the origin and return the
    /// occupied-voxel integer-index bounding box `(min, max_exclusive)` + count.
    fn box_voxel_extent(size_voxels: [u32; 3], density: u32) -> ([i64; 3], [i64; 3], usize) {
        let shape = SdfShape::from_voxels(ShapeKind::Box, size_voxels, 1);
        let mut grid = VoxelGrid::new(size_voxels);
        shape.resolve(&mut grid, density);
        let mut min = [i64::MAX; 3];
        let mut max = [i64::MIN; 3];
        for voxel in &grid.occupied {
            for axis in 0..3 {
                let index = voxel.local_index[axis] as i64;
                min[axis] = min[axis].min(index);
                max[axis] = max[axis].max(index + 1);
            }
        }
        (min, max, grid.occupied.len())
    }

    /// PARITY: a Box fully fills its bounding box, so a voxel-granular size of ANY
    /// parity (odd / even / mixed) emits EXACTLY `prod(size_voxels)` voxels spanning
    /// `[0, size_voxels)` per axis in the producer-true (corner-anchored) frame — no
    /// straddle, no drop. This covers whole-block (even), odd, and mixed sizes.
    #[test]
    fn voxel_granular_box_fills_its_exact_extent_all_parities() {
        let cases: [[u32; 3]; 5] = [
            [80, 16, 80],  // whole-block 5×1×5 @ d16 (all even)
            [81, 17, 81],  // all odd
            [83, 17, 80],  // mixed: odd, odd, even
            [56, 1, 1],    // a flat axis (1 voxel) + even
            [1, 1, 1],     // the minimal box
        ];
        for size in cases {
            let (min, max, count) = box_voxel_extent(size, 16);
            let expected = size[0] as usize * size[1] as usize * size[2] as usize;
            assert_eq!(count, expected, "size {size:?}: a Box fills prod(size) voxels");
            for axis in 0..3 {
                assert_eq!(min[axis], 0, "size {size:?} axis {axis}: corner-anchored min is 0");
                assert_eq!(
                    max[axis], size[axis] as i64,
                    "size {size:?} axis {axis}: spans [0, size) exactly"
                );
            }
        }
    }
}
