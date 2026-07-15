//! The foundational voxel VALUE layer: the resolved cell, its sparse grid, the
//! frame-bearing recentre, the primitive-kind tag, and the pure signed-distance
//! functions the producers sample.
//!
//! This is the `voxel_core` value layer: it depends only DOWNWARD (on `core_geom`)
//! and NEVER on the producer half — no `SdfShape`, no `VoxelProducer`, no
//! `GeometryParams` (those live in the app-crate `voxel` module). That ⊥ is
//! load-bearing: `voxel_core` cannot import the document layer, and the crate
//! boundary now compile-enforces it.
//!
//! Every value here obeys the project-wide Z-up coordinate convention (vertical = +Z,
//! array index 2; ground = XY; front = −Y) — see `docs/architecture/01-document.md`.

use glam::Vec3;

/// CPU-only iso-surface threshold. A voxel is kept when its signed distance is
/// at or below this level. NOT a uniform and NOT a UI slider (DEV_NOTES).
pub const SURFACE_ISOLEVEL: f32 = 0.0;

/// Stability cap on the sampling grid volume (ARCHITECTURE.md §7). If
/// `grid_x * grid_y * grid_z` exceeds this, the 3D rebuild is skipped (the panel
/// shows a warning) so dragging a sphere to 16×16×16 @32 can't freeze the app.
///
/// **Issue #27 S2 — no longer a whole-scene total cap.** The resolve is now
/// chunked + lazy (see the app-crate `chunk_cache`), so the guard moved to a *per-chunk*
/// bound: [`MAX_CHUNK_VOXELS`]. A scene whose TOTAL voxel count is far beyond this
/// 6M figure now resolves fine, as long as each individual chunk is small. This
/// constant is retained because the single-shape `exceeds_voxel_cap` guard still
/// uses it (a lone shape resolved outside the chunk path), and the S2 tests
/// reference it as the OLD total ceiling.
pub const MAX_GRID_VOXELS: u64 = 6_000_000;

/// Per-chunk voxel bound (ADR 0002 Decision 3, issue #27 S2): the most voxels a
/// SINGLE chunk may hold. The deep chunked resolve (the app-crate `chunk_cache`) caps
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
/// **The one mint point** is `Scene::recentre_voxels_for_resolve` (in the app-crate scene),
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
    /// `Scene::recentre_voxels_for_resolve` (in the app-crate scene),
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

    /// Compose this voxel's cuboid region-cell key: the clean categorical colour index
    /// in the low bits, the transient on-face-grid overlay marker in the high bit (ADR
    /// 0003 §3c). The overlay bit lives ONLY in this render-side key — never in the
    /// persistent [`Voxel`] payload, the chunk-storage codec, or the `.vox` export. The
    /// cuboid mesher and every region builder decompose against this [`CellKey`], so a
    /// box splits across differing overlay flags without a render flag entering the
    /// categorical id.
    ///
    /// [`CellKey`]: crate::core_geom::CellKey
    #[inline]
    pub fn cell_key(&self) -> crate::core_geom::CellKey {
        crate::core_geom::CellKey::compose(self.color_index(), self.grid_overlay)
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
    /// (`Scene::resolve_region` subtracts it from
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
