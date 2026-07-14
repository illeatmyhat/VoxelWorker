//! The resolved voxel grid and the producers that fill it.
//!
//! ## Coordinate convention (PROJECT-WIDE — Z-up, right-handed)
//!
//! **Vertical / up = +Z** ([`glam::Vec3::Z`], array index **2**) EVERYWHERE in this
//! crate — camera, SDFs, fog, layers, diameter, mesh and `.vox` export all agree.
//! The ground plane is **XY** (normal +Z); **front = −Y** (the front view looks
//! along +Y); LEFT/RIGHT = ±X; TOP/BOTTOM = ±Z. Panel X/Y/Z fields map directly to
//! indices 0/1/2 with Z genuinely the vertical axis — no relabel shim.
//!
//! Consequences pinned by tests: a tall cylinder/tube/torus has its axis along Z
//! (`size_voxels[2]` is the vertical extent), layer slices are Z-slices, the onion
//! fog band is a Z-range, and the `.vox` export writes our Z straight to vox-Z with
//! NO axis swap (MagicaVoxel is itself Z-up).
//!
//! This module implements the architectural seam required by `REPRESENTATION.md`:
//! **the renderer never calls the SDF directly.** Instead a [`VoxelProducer`]
//! resolves a parametric shape (or, in a later milestone, a sculpt overlay) into
//! a [`VoxelGrid`] — the one consumed truth. The renderer, the layer-range
//! diameter readout (issue #12) and the `.vox` export (M8) all read the grid, so
//! adding a second producer later touches nothing downstream.
//!
//! Milestone 2 has exactly one producer: [`SdfShape`], which runs the sampling
//! triple-loop transcribed from `ARCHITECTURE.md` §1/§2 and writes occupied
//! voxels into the grid.

use glam::Vec3;
use rayon::prelude::*;

/// CPU-only iso-surface threshold. A voxel is kept when its signed distance is
/// at or below this level. NOT a uniform and NOT a UI slider (DEV_NOTES).
pub const SURFACE_ISOLEVEL: f32 = 0.0;

/// Stability cap on the sampling grid volume (ARCHITECTURE.md §7). If
/// `grid_x * grid_y * grid_z` exceeds this, the 3D rebuild is skipped (the panel
/// shows a warning) so dragging a sphere to 16×16×16 @32 can't freeze the app.
///
/// **Issue #27 S2 — no longer a whole-scene total cap.** The resolve is now
/// chunked + lazy (see [`crate::chunk_cache`]), so the guard moved to a *per-chunk*
/// bound: [`MAX_CHUNK_VOXELS`]. A scene whose TOTAL voxel count is far beyond this
/// 6M figure now resolves fine, as long as each individual chunk is small. This
/// constant is retained because [`exceeds_voxel_cap`](SdfShape::exceeds_voxel_cap)
/// still uses it as a single-shape sanity guard (a lone shape resolved outside the
/// chunk path), and the S2 tests reference it as the OLD total ceiling.
pub const MAX_GRID_VOXELS: u64 = 6_000_000;

/// Per-chunk voxel bound (ADR 0002 Decision 3, issue #27 S2): the most voxels a
/// SINGLE chunk may hold. The deep chunked resolve ([`crate::chunk_cache`]) caps
/// each chunk, not the whole scene — so total scene size is bounded only by how
/// many chunks resolve, not by one 6M ceiling.
///
/// One chunk's voxel CAPACITY is `(CHUNK_BLOCKS × voxels_per_block)³`: at the app
/// default density 16 that is `64³ = 262_144` voxels, comfortably under this bound.
/// The bound exists so a pathological density (where one chunk's capacity alone
/// would blow memory) is still rejected — see [`chunk_extent_exceeds_bound`].
pub const MAX_CHUNK_VOXELS: u64 = 6_000_000;

/// Whether one chunk's voxel CAPACITY at `voxels_per_block`
/// (`(CHUNK_BLOCKS × voxels_per_block)³`) exceeds the per-chunk bound
/// [`MAX_CHUNK_VOXELS`] (issue #27 S2). The chunked-resolve call sites reject a
/// density this large (a single chunk alone would exceed the bound) instead of
/// resolving it.
pub fn chunk_extent_exceeds_bound(voxels_per_block: u32) -> bool {
    let extent = (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as u64;
    extent.saturating_mul(extent).saturating_mul(extent) > MAX_CHUNK_VOXELS
}

/// The parametric primitive kinds (ARCHITECTURE.md §2 dispatcher).
///
/// Milestone 2 only renders [`ShapeKind::Cylinder`], but the full set is
/// implemented now because M3 needs them and the cost is trivial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShapeKind {
    Cylinder,
    Tube,
    Sphere,
    Torus,
    Box,
}

pub use crate::core_geom::{BlockAttrs, BlockId};

/// The composite floating-origin recentre, in voxels — the frame value every display
/// artifact of one rebuild is resolved in.
///
/// **The frame law (docs/architecture, the voxel-frame invariant).** A spatial value
/// CARRIES the frame it was authored in; consumers decode with it and never re-derive it.
/// A build's install must use the recentre THAT build was resolved at — so the recentre
/// travels end-to-end (resolve → orchestrator → the async worker channels → the GPU
/// install) as this newtype, and the compiler enforces that the install uses the request's
/// recentre rather than a same-shaped `[i64; 3]` from somewhere else.
///
/// **The one mint point** is [`Scene::recentre_voxels_for_resolve`](crate::scene::Scene::recentre_voxels_for_resolve),
/// which returns this newtype directly — so a build's recentre is born already carrying its
/// frame. Transport only this increment: it is `Copy`, has no arithmetic, and [`voxels`] is
/// the ONE way back to the raw triple — unwrapped only at the point of actual positional
/// ARITHMETIC (a leaf stamp's recentre subtraction, a chunk rebase's index offset), at the
/// GPU uniform packing, and at the raw-BY-RULE values (the dense-oracle grid's carried
/// field, a recentre-shift delta, a comparison/cache key). The mesh / two-layer / scene
/// transport signatures now speak this newtype. It lives in the spatial-primitive layer (this
/// module) alongside the other frame-bearing primitives; [`new`](RecentreVoxels::new)
/// remains for the boundary/test sites that mint a KNOWN recentre from a raw triple (e.g.
/// the `shot` oracle grid's carried field).
///
/// [`voxels`]: RecentreVoxels::voxels
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RecentreVoxels([i64; 3]);

impl RecentreVoxels {
    /// Carry a known recentre triple as its frame value — the boundary/test constructor
    /// for a recentre that arrives as a raw `[i64; 3]` (the `shot` oracle grid's carried
    /// field, the parity tests' known recentre). The PRODUCTION mint is
    /// [`Scene::recentre_voxels_for_resolve`](crate::scene::Scene::recentre_voxels_for_resolve),
    /// which returns this newtype directly.
    pub fn new(voxels: [i64; 3]) -> Self {
        Self(voxels)
    }

    /// The raw voxel triple — the single consumption door, called only at the point of
    /// positional arithmetic, at the GPU uniform packing, and at the raw-by-rule oracle /
    /// cache / delta values.
    pub fn voxels(&self) -> [i64; 3] {
        self.0
    }
}

/// One occupied voxel in the resolved grid (ADR 0003 §3a — the chunk-local integer +
/// categorical block-palette cell).
///
/// **The per-voxel record carries an INTEGER index, never an f32 position.** ADR 0003
/// §3a / ADR 0008 (the voxel-frame invariant): the absolute i64 origin lives ONLY in
/// the grid's carried frame (the chunk key / `recentre_voxels`), and each cell stores
/// its voxel index `[i, j, k]` *within that frame*. f32 is produced ONLY at consumption
/// via [`world_position`](Voxel::world_position) (`index + 0.5`), reproducing exactly
/// the half-integer voxel centre the old f32 payload stored — but exactly, with no f32
/// magnitude loss for a far-placed (origin-rebased) chunk. The stamp keeps the integer
/// in i64 right up to the downcast to the field, so a far scene is exact rather than
/// merely "exact for near scenes".
///
/// `block_local_coord` is `(i % voxels_per_block, …)` — the voxel's position *within*
/// its block, needed by the M4 texture-slice shader. `block_id` is the categorical
/// block-palette index (replacing the old 3-value `material_id` enum); `attrs` is the
/// minimal forward-compat [`BlockAttrs`] placeholder (the typed stair-facing /
/// connection schema of ADR 0003 §3a-bis stays deferred). The `GRID_OVERLAY_BIT`
/// render flag is **no longer in this payload** — it is a per-draw / per-box render
/// attribute (ADR 0003 §3c).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Voxel {
    /// Voxel index within the grid's CARRIED frame (ADR 0008): the absolute origin
    /// (chunk key / `recentre_voxels`) lives on the grid, this is the local integer
    /// index. `i32` carries any region-scoped index (recentred grids place index 0 at
    /// a negative position) with full precision and no f32 rounding.
    pub local_index: [i32; 3],
    /// Coordinate within the owning block: `(i % d, j % d, k % d)`.
    pub block_local_coord: [u8; 3],
    /// Categorical block-palette id (ADR 0003 §3a). For the three procedural materials
    /// this is the old `material_id` value (Stone/Wood/Plain ⇒ 0/1/2), so existing
    /// scenes resolve byte-identically; the rich VS palette content stays deferred.
    pub block_id: BlockId,
    /// Typed per-`block_id` attributes (ADR 0003 §3a-bis). A minimal forward-compat
    /// placeholder here; the orientation / variant / connection schema is deferred.
    pub attrs: BlockAttrs,
    /// **Transient render marker — NOT part of the categorical cell** (ADR 0003 §3c).
    /// The owning node's `grids.voxel_grid_on_faces` flag, carried so the cuboid mesher
    /// can split a box on it and the draw can enable the on-face grid overlay. This is
    /// the per-node render concern §3c removed from `block_id` (where the old
    /// `GRID_OVERLAY_BIT` jammed it): it never rides the chunk-storage codec, the `.vox`
    /// export, or the categorical id — it is a resolve→mesh render hint only, surfaced
    /// to the shader as a dedicated overlay attribute, never masked out of the material.
    pub grid_overlay: bool,
}

impl Voxel {
    /// The voxel centre as an f32 position in the grid's carried frame — `index + 0.5`,
    /// EXACTLY the half-integer centre the retired `world_position: [f32; 3]` field
    /// stored (ADR 0003 §3a: f32 produced only at consumption). Every consumer that
    /// decoded the old f32 back to an integer (`floor`, `round(world + half − 0.5)`, …)
    /// keeps working byte-identically because this reproduces the same `index + 0.5`.
    #[inline]
    pub fn world_position(&self) -> [f32; 3] {
        [
            self.local_index[0] as f32 + 0.5,
            self.local_index[1] as f32 + 0.5,
            self.local_index[2] as f32 + 0.5,
        ]
    }

    /// The categorical block id as the colour / atlas index the renderer + `.vox`
    /// export use (today the 3-value palette maps 1:1 to the colour index). Replaces
    /// the old `material_id_color_index` mask now that the render flag is gone.
    #[inline]
    pub fn color_index(&self) -> u16 {
        self.block_id.0
    }
}

/// The resolved truth consumed by the renderer / slice / export.
///
/// Sparse representation: grid dimensions in voxels plus a `Vec` of the occupied
/// voxels only. For a filled 5×1×5@16 disc this is ~800k entries which is
/// memory-friendly compared with a dense 80×16×80 bitfield-plus-payload, and it
/// is exactly the iteration set the instance buffer needs.
#[derive(Debug, Default, Clone)]
pub struct VoxelGrid {
    /// Grid dimensions in voxels (the producer's voxel-granular size, already at
    /// document density — e.g. `SdfShape::size_voxels`).
    pub dimensions: [u32; 3],
    /// The integer voxel offset this grid's world positions were RECENTRED by
    /// ([`Scene::resolve_region`](crate::scene::Scene::resolve_region) subtracts it from
    /// every voxel). **ADR 0008 — the carried frame.** A placed composite is recentred by
    /// `(min+max)/2` (= `floor(dim/2)` for a lone producer); a Part-only / bare-producer
    /// grid is corner-anchored, so this is `[0,0,0]`. Carrying it lets every consumer
    /// decode `world → index` correctly WITHOUT re-deriving the centring (the assumption
    /// that, hard-coded as `floor(dim/2)`, made the fog drop a corner-anchored cloud
    /// field). Default `[0,0,0]` is correct for any un-recentred grid.
    pub recentre_voxels: [i64; 3],
    /// The occupied voxels (sparse).
    pub occupied: Vec<Voxel>,
}

impl VoxelGrid {
    /// Create an empty grid with the given voxel dimensions (un-recentred:
    /// `recentre_voxels = [0,0,0]`; a recentred resolve sets it explicitly).
    pub fn new(dimensions: [u32; 3]) -> Self {
        Self {
            dimensions,
            recentre_voxels: [0, 0, 0],
            occupied: Vec::new(),
        }
    }

    /// Number of occupied voxels.
    pub fn occupied_count(&self) -> usize {
        self.occupied.len()
    }

    /// The local `[0, dimensions)` voxel index of the voxel whose centre sits at
    /// `world_position`, decoded with this grid's CARRIED [`recentre_voxels`] rather than
    /// a re-derived `floor(dim/2)`.
    ///
    /// **ADR 0008 (the voxel-frame invariant): the SINGLE world→index decode authority.**
    /// A producer corner-anchors each centre at `idx + 0.5`, then the resolve subtracts
    /// `recentre_voxels`, so `world = idx + 0.5 − recentre`; this inverts that exactly for
    /// any dimension parity. Because the recentre is *carried* (not assumed), a centred
    /// placed Tool (`recentre = floor(dim/2)`) and a corner-anchored Part-only
    /// `DebugClouds` (`recentre = [0,0,0]`) BOTH decode into `[0, dimensions)` — the
    /// divergence that, with a hard-coded `floor(dim/2)`, dropped ~7/8 of a corner-
    /// anchored cloud field. For a centred grid this reduces to the historical
    /// `round(world + floor(dim/2) − 0.5)`, so placed scenes are byte-identical. The
    /// result may fall outside `[0, dimensions)` for a stray position — callers
    /// bounds-check.
    ///
    /// [`recentre_voxels`]: VoxelGrid::recentre_voxels
    pub fn voxel_index_of(&self, world_position: [f32; 3]) -> [i64; 3] {
        [
            (world_position[0] + self.recentre_voxels[0] as f32 - 0.5).round() as i64,
            (world_position[1] + self.recentre_voxels[1] as f32 - 0.5).round() as i64,
            (world_position[2] + self.recentre_voxels[2] as f32 - 0.5).round() as i64,
        ]
    }

    /// Measure the widest occupied voxel run (the diameter readout, issue #12),
    /// restricted to the layers `[band_min, band_max]` (inclusive) along Z (Z-up:
    /// layers are Z-slices). The "widest run" is the longest contiguous span of
    /// occupied voxels along X within any single `(z, y)` row of the band — the same
    /// measure the old 2D slice reported, but taken over the active band instead of
    /// the mid-vertical layer.
    ///
    /// Reads the RESOLVED grid — NOT the SDF — per REPRESENTATION.md. Cheap: one
    /// pass over the sparse occupied list bucketed into per-(z,y)-row bitsets.
    pub fn widest_run_in_band(&self, band_min: u32, band_max: u32) -> u32 {
        let [grid_x, grid_y, grid_z] = self.dimensions;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            return 0;
        }
        let width = grid_x as usize;
        // Corner-anchoring decode: the grid's low corner in the recentred frame is
        // `−floor(dim/2)`, so `idx = round(world − region_low − 0.5) = round(world +
        // floor(dim/2) − 0.5)`. Use FLOORED half (`dim/2` integer division), NOT
        // `dim/2.0`, so the decode is exact for an ODD dim too (world is half-integer).
        let half_x = (grid_x / 2) as f32;
        let half_y = (grid_y / 2) as f32;
        let half_z = (grid_z / 2) as f32;

        // One occupancy row (length grid_x) per (z, y) row that touches the band.
        // Keyed by a flat (z, y) index; built sparsely so an empty grid is cheap.
        // Z-up: the band is a Z-layer (index 2) range; `k` (Z) is the layer scan.
        let mut rows: std::collections::HashMap<u64, Vec<bool>> = std::collections::HashMap::new();
        for voxel in &self.occupied {
            let position = voxel.world_position();
            let k = (position[2] + half_z - 0.5).round() as i64;
            if k < band_min as i64 || k > band_max as i64 {
                continue;
            }
            let i = (position[0] + half_x - 0.5).round() as i64;
            let j = (position[1] + half_y - 0.5).round() as i64;
            if i < 0 || i >= width as i64 || j < 0 || j >= grid_y as i64 {
                continue;
            }
            let key = (k as u64) << 32 | (j as u64);
            let row = rows.entry(key).or_insert_with(|| vec![false; width]);
            row[i as usize] = true;
        }

        let mut widest = 0u32;
        for row in rows.values() {
            let mut run = 0u32;
            for &occupied in row {
                if occupied {
                    run += 1;
                    widest = widest.max(run);
                } else {
                    run = 0;
                }
            }
        }
        widest
    }
}

/// **Region-scoped diameter readout (issue #20 S6d).** Compute the SAME value as
/// [`VoxelGrid::widest_run_in_band`] would return for the whole region, but from a
/// SET of per-chunk grids instead of one assembled monolithic grid — so the
/// scrubber/diameter consumer no longer needs the whole grid materialised once the
/// S6c monolithic bridge is gone.
///
/// `region_dimensions` are the region's voxel dimensions (`[grid_x, grid_y,
/// grid_z]`), exactly what the assembled monolithic grid's `dimensions` would be —
/// they define the X-axis width of each scan row and the half-extents used to
/// recover integer grid indices from a voxel's centred `world_position`. The
/// `chunk_grids` iterator yields each covering per-chunk grid whose voxels are in
/// the SAME (recentred) coordinate frame the monolithic grid uses; only their
/// `occupied` lists are read (each chunk's own `dimensions` are irrelevant here).
///
/// ## How runs are stitched across chunk seams (the subtle part)
///
/// A run of occupied voxels that crosses a chunk boundary must count as ONE run,
/// not two. We do not merge per-chunk partial runs after the fact (that would need
/// careful seam bookkeeping and is easy to get subtly wrong); instead we bucket
/// **every** voxel from **every** chunk into a SINGLE shared occupancy row per
/// `(y, z)` keyed by the voxel's GLOBAL X index (`i = round(world_x + grid_x/2 −
/// 0.5)`), the very same index the whole-grid function computes. Because two
/// voxels straddling a chunk seam land at adjacent global X positions in the same
/// shared row bitset, the seam simply vanishes — the contiguous-run scan sees one
/// uninterrupted span. The result is therefore identical to the whole-grid
/// computation by construction: the set of bucketed voxels is the union of the
/// chunk occupied sets (= the monolithic occupied set), and the bucketing /
/// run-scan arithmetic is byte-for-byte the same as
/// [`VoxelGrid::widest_run_in_band`].
pub fn widest_run_in_band_over_chunks<'grid>(
    region_dimensions: [u32; 3],
    chunk_grids: impl IntoIterator<Item = &'grid VoxelGrid>,
    band_min: u32,
    band_max: u32,
) -> u32 {
    let [grid_x, grid_y, grid_z] = region_dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return 0;
    }
    let width = grid_x as usize;
    // Corner-anchoring decode: FLOORED half (`dim/2` integer division), exact for an
    // odd dim — see `widest_run_in_band`.
    let half_x = (grid_x / 2) as f32;
    let half_y = (grid_y / 2) as f32;
    let half_z = (grid_z / 2) as f32;

    // ONE shared occupancy row (length grid_x) per (z, y) row that touches the
    // band — shared across ALL chunks, so a run spanning a chunk seam is one
    // contiguous span in the same bitset. Keyed by a flat (z, y) index, built
    // sparsely so an empty band is cheap. Z-up: the band is a Z-layer range.
    let mut rows: std::collections::HashMap<u64, Vec<bool>> = std::collections::HashMap::new();
    for grid in chunk_grids {
        for voxel in &grid.occupied {
            let position = voxel.world_position();
            let k = (position[2] + half_z - 0.5).round() as i64;
            if k < band_min as i64 || k > band_max as i64 {
                continue;
            }
            let i = (position[0] + half_x - 0.5).round() as i64;
            let j = (position[1] + half_y - 0.5).round() as i64;
            if i < 0 || i >= width as i64 || j < 0 || j >= grid_y as i64 {
                continue;
            }
            let key = (k as u64) << 32 | (j as u64);
            let row = rows.entry(key).or_insert_with(|| vec![false; width]);
            row[i as usize] = true;
        }
    }

    let mut widest = 0u32;
    for row in rows.values() {
        let mut run = 0u32;
        for &occupied in row {
            if occupied {
                run += 1;
                widest = widest.max(run);
            } else {
                run = 0;
            }
        }
    }
    widest
}

// The conservative cell-interval bound and its coarse classification are pure interval
// arithmetic under CSG lattice ops — substrate's [`substrate::FieldInterval`]. The
// domain reads it with the occupancy convention "inside where `field <= SURFACE_ISOLEVEL`":
// `FieldInterval::classify(SURFACE_ISOLEVEL)` yields AIR / COARSE-SOLID / BOUNDARY for a
// whole block-sized cell, and `substrate::union_field_intervals` composes a Union of producers
// (min-of-fields). The conservative-never-narrow property is why a coarse verdict can
// never disagree with a brute-force per-voxel evaluation — the boundary-residency
// classifier's soundness (see the Boundary-residency material in
// `docs/architecture/02-evaluation.md`, proven by the E1 parity gate in
// `cell_interval_parity_tests`). The interval algebra, the Lipschitz-centre bound, and
// the classify threshold-parameter live in the substrate module doc.
pub use substrate::{FieldClassification, FieldInterval};

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
        window_local_voxels: crate::spatial_index::VoxelAabb,
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
        cell_local_voxels: crate::spatial_index::VoxelAabb,
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
    window_local_voxels: crate::spatial_index::VoxelAabb,
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
    pub size_measurements: Option<Box<[crate::units::Measurement; 3]>>,
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
    size_measurements: Option<Box<[crate::units::Measurement; 3]>>,
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
        use crate::units::{ExactRational, Measurement};
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

    /// Build a shape from a per-axis authored [`Measurement`](crate::units::Measurement)
    /// size at density `voxels_per_block` (ADR 0003 §3f(0)). The canonical voxel size
    /// is DERIVED via [`Measurement::to_voxels`](crate::units::Measurement::to_voxels)
    /// and clamped to `>= 1`; the measurements are RETAINED for lossless density
    /// re-targeting. Mirrors
    /// [`NodeTransform::from_measurements`](crate::scene::NodeTransform::from_measurements),
    /// including the self-consistency rule: a non-landing axis floors AND
    /// resynthesises its retained measurement to the pure-voxel form, so
    /// `size_voxels` and the retained expression never disagree. (A size that floors
    /// below 1 voxel is clamped to 1 and resynthesised to the pure-voxel `1`.)
    pub fn from_measurements(
        kind: ShapeKind,
        measurements: [crate::units::Measurement; 3],
        wall_blocks: u32,
        voxels_per_block: u32,
    ) -> Self {
        use crate::units::{Measurement, MeasurementError};
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
        measurements: Option<Box<[crate::units::Measurement; 3]>>,
        size_voxels: [u32; 3],
    ) -> Option<Box<[crate::units::Measurement; 3]>> {
        use crate::units::Measurement;
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
    pub fn size_measurements(&self) -> [crate::units::Measurement; 3] {
        use crate::units::Measurement;
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
            crate::spatial_index::VoxelAabb::new(
                [0, 0, 0],
                [full_x as i64, full_y as i64, full_z as i64],
            ),
        );
    }

    fn resolve_into(
        &self,
        grid: &mut VoxelGrid,
        voxels_per_block: u32,
        window_local_voxels: crate::spatial_index::VoxelAabb,
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
        cell_local_voxels: crate::spatial_index::VoxelAabb,
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

        // Conservative Lipschitz constant: 1 for the truly-1-Lipschitz kinds, the
        // semi-axis anisotropy for the ellipsoid / cylinder / tube whose gradient can
        // steepen along the shorter axis. Always >= the true constant ⇒ never narrows.
        let lipschitz_constant = match self.kind {
            ShapeKind::Box | ShapeKind::Torus => 1.0,
            ShapeKind::Sphere | ShapeKind::Cylinder | ShapeKind::Tube => {
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

/// Signed distance to an axis-aligned box with half-extents `box_half`.
///
/// `sdBox` in ARCHITECTURE.md §2, descriptive names.
pub fn signed_distance_box(point: Vec3, box_half: Vec3) -> f32 {
    let q = point.abs() - box_half;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Signed distance to an inscribed ellipsoid (IQ approximation).
///
/// `sdEllipsoid` in ARCHITECTURE.md §2.
pub fn signed_distance_ellipsoid(point: Vec3, semi_axes: Vec3) -> f32 {
    let scaled = point / semi_axes;
    let distance_to_unit = scaled.length();
    if distance_to_unit == 0.0 {
        return -semi_axes.x.min(semi_axes.y.min(semi_axes.z));
    }
    let scaled_squared = point / (semi_axes * semi_axes);
    let gradient = scaled_squared.length();
    distance_to_unit * (distance_to_unit - 1.0) / gradient
}

/// Signed distance to an elliptical cylinder with its axis along Z (Z-up).
///
/// `sdCylE(p, ax, ay, az)` in ARCHITECTURE.md §2: `semi_axis_x`/`semi_axis_y`
/// are the cross-section radii (the cylinder's circular cross-section lies in the
/// XY ground plane), `half_height` is the Z (vertical) half-extent.
pub fn signed_distance_elliptical_cylinder(
    point: Vec3,
    semi_axis_x: f32,
    semi_axis_y: f32,
    half_height: f32,
) -> f32 {
    let radial = (glam::Vec2::new(point.x / semi_axis_x, point.y / semi_axis_y).length() - 1.0)
        * semi_axis_x.min(semi_axis_y);
    let vertical = point.z.abs() - half_height;
    radial.max(vertical).min(0.0)
        + glam::Vec2::new(radial.max(0.0), vertical.max(0.0)).length()
}

/// Dispatch to the right SDF for a shape kind (ARCHITECTURE.md §2 `sdf(p)`).
///
/// `semi_axes` are the inscribed half-extents `(AX, AY, AZ)`; `wall_voxels` is
/// `wall * density` (Tube only).
pub fn signed_distance(
    shape: ShapeKind,
    point: Vec3,
    semi_axes: Vec3,
    wall_voxels: f32,
) -> f32 {
    let semi_axis_x = semi_axes.x;
    let semi_axis_y = semi_axes.y;
    let semi_axis_z = semi_axes.z;

    match shape {
        ShapeKind::Cylinder => {
            // Z-up: axis along Z. Cross-section radii are X/Y; `semi_axis_z` is the
            // vertical half-height.
            signed_distance_elliptical_cylinder(point, semi_axis_x, semi_axis_y, semi_axis_z)
        }
        ShapeKind::Tube => {
            let outer =
                signed_distance_elliptical_cylinder(point, semi_axis_x, semi_axis_y, semi_axis_z);
            let inner = signed_distance_elliptical_cylinder(
                point,
                (semi_axis_x - wall_voxels).max(0.01),
                (semi_axis_y - wall_voxels).max(0.01),
                semi_axis_z + 1.0,
            );
            outer.max(-inner)
        }
        ShapeKind::Sphere => signed_distance_ellipsoid(point, semi_axes),
        ShapeKind::Torus => {
            // Z-up: the ring lies in the XY ground plane, swept around the +Z axis;
            // the tube minor radius is the vertical (Z) extent.
            let tube_radius = semi_axis_z;
            let ring_radius = (semi_axis_x.min(semi_axis_y) - tube_radius).max(0.0);
            let radial = glam::Vec2::new(point.x, point.y).length() - ring_radius;
            glam::Vec2::new(radial, point.z).length() - tube_radius
        }
        ShapeKind::Box => signed_distance_box(point, semi_axes),
    }
}

#[cfg(test)]
mod categorical_block_id_tests {
    use super::*;

    /// ADR 0003 §3a/§3c: the per-voxel cell carries the categorical `block_id` ONLY —
    /// the colour index IS the block id (no render flag sharing the field, no mask). The
    /// three procedural materials keep their old ids (Stone/Wood/Plain ⇒ 0/1/2), so an
    /// existing scene resolves byte-identically.
    #[test]
    fn color_index_is_the_block_id_no_flag_in_the_field() {
        for id in 0u16..=2 {
            let voxel = Voxel {
                local_index: [0, 0, 0],
                block_local_coord: [0, 0, 0],
                block_id: BlockId(id),
                attrs: BlockAttrs::DEFAULT,
                grid_overlay: false,
            };
            assert_eq!(voxel.color_index(), id, "the colour index is the block id verbatim");
            assert!(voxel.color_index() <= 2, "the procedural ids stay in the shader's colour range");
        }
    }

    /// The reconstructed f32 centre is exactly `index + 0.5` (ADR 0003 §3a: f32 produced
    /// only at consumption), so `floor` recovers the stored integer index losslessly.
    #[test]
    fn world_position_reconstructs_index_plus_half() {
        for index in [[0, 0, 0], [3, 5, 7], [-4, -1, -9], [1234, -5678, 9012]] {
            let voxel = Voxel {
                local_index: index,
                block_local_coord: [0, 0, 0],
                block_id: BlockId::DEFAULT,
                attrs: BlockAttrs::DEFAULT,
                grid_overlay: false,
            };
            let position = voxel.world_position();
            for axis in 0..3 {
                assert_eq!(position[axis], index[axis] as f32 + 0.5);
                assert_eq!(position[axis].floor() as i32, index[axis]);
            }
        }
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
    use crate::units::{DisplayUnit, ExactRational, Measurement};

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
            let text = crate::units::format(voxels, 16, DisplayUnit::BlocksAndVoxels);
            let reparsed = crate::units::parse(&text).expect("re-parses");
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
