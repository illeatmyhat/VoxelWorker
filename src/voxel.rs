//! The resolved voxel grid and the producers that fill it.
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

/// Per-object on-face-grid flag bit packed into a voxel's `material_id`
/// (issue #29 S4). The material id only ever carries a small enum value
/// (Stone/Wood/Plain ⇒ 0/1/2; the shaders clamp it to ≤2 before any colour
/// lookup), so the high bit is free to flag "draw the on-face voxel grid on this
/// voxel's faces". The resolver ORs this bit into a voxel's `material_id` iff the
/// producing node has `grids.voxel_grid_on_faces`; the GPU-upload path strips it
/// again when the scene-wide `master_voxel_grid` is OFF (the master AND); and the
/// both mesh shaders read `(material_id & GRID_OVERLAY_BIT) != 0` to gate the
/// on-face grid branch, masking the bit OFF (via [`material_id_color_index`])
/// before any atlas / base-colour lookup so the flag never corrupts the colour.
///
/// **This constant is mirrored verbatim in `shaders/cuboid.wgsl` and
/// `shaders/cuboid_loaded.wgsl`** (`const GRID_OVERLAY_BIT: u32 = 32768u;`) — keep
/// all three in sync.
pub const GRID_OVERLAY_BIT: u16 = 1 << 15;

/// Strip the [`GRID_OVERLAY_BIT`] from a `material_id`, leaving only the real
/// material handle used for the colour / atlas lookup. The shaders perform the
/// same mask (`material_id & ~GRID_OVERLAY_BIT`, then clamp to ≤2); this is the
/// CPU mirror so tests can assert the colour index round-trips.
#[inline]
pub fn material_id_color_index(material_id: u16) -> u16 {
    material_id & !GRID_OVERLAY_BIT
}

/// One occupied voxel in the resolved grid.
///
/// `block_local_coord` is `(i % voxels_per_block, …)` — the voxel's position
/// *within* its block, needed by the M4 texture-slice shader. `material_id`
/// carries the real material handle in its low bits plus the optional
/// [`GRID_OVERLAY_BIT`] flag (issue #29 S4) in its high bit.
#[derive(Debug, Clone, Copy)]
pub struct Voxel {
    /// World-centred voxel-grid coordinate of the voxel centre.
    pub world_position: [f32; 3],
    /// Coordinate within the owning block: `(i % d, j % d, k % d)`.
    pub block_local_coord: [u8; 3],
    /// Reserved material handle (unused in M2).
    pub material_id: u16,
}

/// The resolved truth consumed by the renderer / slice / export.
///
/// Sparse representation: grid dimensions in voxels plus a `Vec` of the occupied
/// voxels only. For a filled 5×1×5@16 disc this is ~800k entries which is
/// memory-friendly compared with a dense 80×16×80 bitfield-plus-payload, and it
/// is exactly the iteration set the instance buffer needs.
#[derive(Debug, Default, Clone)]
pub struct VoxelGrid {
    /// Grid dimensions in voxels: `size_blocks * voxels_per_block`.
    pub dimensions: [u32; 3],
    /// The occupied voxels (sparse).
    pub occupied: Vec<Voxel>,
}

impl VoxelGrid {
    /// Create an empty grid with the given voxel dimensions.
    pub fn new(dimensions: [u32; 3]) -> Self {
        Self {
            dimensions,
            occupied: Vec::new(),
        }
    }

    /// Number of occupied voxels.
    pub fn occupied_count(&self) -> usize {
        self.occupied.len()
    }

    /// Measure the widest occupied voxel run (the diameter readout, issue #12),
    /// restricted to the layers `[band_min, band_max]` (inclusive) along Y. The
    /// "widest run" is the longest contiguous span of occupied voxels along X
    /// within any single `(y, z)` row of the band — the same measure the old 2D
    /// slice reported, but taken over the active band instead of the mid-Y layer.
    ///
    /// Reads the RESOLVED grid — NOT the SDF — per REPRESENTATION.md. Cheap: one
    /// pass over the sparse occupied list bucketed into per-(y,z)-row bitsets.
    pub fn widest_run_in_band(&self, band_min: u32, band_max: u32) -> u32 {
        let [grid_x, grid_y, grid_z] = self.dimensions;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            return 0;
        }
        let width = grid_x as usize;
        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;

        // One occupancy row (length grid_x) per (y, z) row that touches the band.
        // Keyed by a flat (y, z) index; built sparsely so an empty grid is cheap.
        let mut rows: std::collections::HashMap<u64, Vec<bool>> = std::collections::HashMap::new();
        for voxel in &self.occupied {
            let j = (voxel.world_position[1] + half_y - 0.5).round() as i64;
            if j < band_min as i64 || j > band_max as i64 {
                continue;
            }
            let i = (voxel.world_position[0] + half_x - 0.5).round() as i64;
            let k = (voxel.world_position[2] + half_z - 0.5).round() as i64;
            if i < 0 || i >= width as i64 || k < 0 || k >= grid_z as i64 {
                continue;
            }
            let key = (j as u64) << 32 | (k as u64);
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
    let half_x = grid_x as f32 / 2.0;
    let half_y = grid_y as f32 / 2.0;
    let half_z = grid_z as f32 / 2.0;

    // ONE shared occupancy row (length grid_x) per (y, z) row that touches the
    // band — shared across ALL chunks, so a run spanning a chunk seam is one
    // contiguous span in the same bitset. Keyed by a flat (y, z) index, built
    // sparsely so an empty band is cheap.
    let mut rows: std::collections::HashMap<u64, Vec<bool>> = std::collections::HashMap::new();
    for grid in chunk_grids {
        for voxel in &grid.occupied {
            let j = (voxel.world_position[1] + half_y - 0.5).round() as i64;
            if j < band_min as i64 || j > band_max as i64 {
                continue;
            }
            let i = (voxel.world_position[0] + half_x - 0.5).round() as i64;
            let k = (voxel.world_position[2] + half_z - 0.5).round() as i64;
            if i < 0 || i >= width as i64 || k < 0 || k >= grid_z as i64 {
                continue;
            }
            let key = (j as u64) << 32 | (k as u64);
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

/// Anything that can resolve itself into the shared [`VoxelGrid`].
///
/// v1 has a single implementor ([`SdfShape`]); the trait exists so a sculpt
/// overlay (REPRESENTATION.md option 2) can be added later without changing the
/// renderer.
pub trait VoxelProducer {
    /// Write occupied voxels into `grid`. The grid's `dimensions` are assumed to
    /// already be set by the caller (so multiple producers can target one grid).
    fn resolve(&self, grid: &mut VoxelGrid);
}

/// Geometry parameters — the *only* params that trigger a voxel rebuild.
///
/// The UI-side mirror of [`SdfShape`] (the panel edits this; `SdfShape::from_geometry`
/// turns it into a producer). Sizes are in **whole blocks**; `voxels_per_block` is
/// fineness only and never changes the object's block size (DATA.md "the density bug").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryParams {
    /// Selected primitive.
    pub shape: ShapeKind,
    /// Bounding-box size in whole blocks (X, Y, Z).
    pub size_blocks: [u32; 3],
    /// Voxels per block (chisel fineness). Default 16.
    pub voxels_per_block: u32,
    /// Tube wall thickness in whole blocks (used by [`ShapeKind::Tube`] only).
    pub wall_blocks: u32,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            shape: ShapeKind::Cylinder,
            size_blocks: [5, 1, 5],
            voxels_per_block: 16,
            wall_blocks: 1,
        }
    }
}

/// A single parametric SDF primitive: the first (and, in M2, only) producer.
///
/// Sizes are stored in **whole blocks**; `voxels_per_block` (density) is fineness
/// only and never changes object size (DATA.md "the density bug").
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SdfShape {
    #[serde(default = "default_shape_kind")]
    pub kind: ShapeKind,
    /// Bounding-box size in whole blocks (X, Y, Z).
    #[serde(default = "default_shape_size")]
    pub size_blocks: [u32; 3],
    /// Voxels per block (chisel fineness). Default 16.
    #[serde(default = "default_shape_density")]
    pub voxels_per_block: u32,
    /// Tube wall thickness in whole blocks (used by [`ShapeKind::Tube`] only).
    #[serde(default = "default_shape_wall")]
    pub wall_blocks: u32,
}

/// Persistence defaults for a partial [`SdfShape`] (a missing field falls back to
/// a sane non-zero value so a tolerant config load never yields a degenerate
/// zero-size shape).
fn default_shape_kind() -> ShapeKind {
    ShapeKind::Cylinder
}
fn default_shape_size() -> [u32; 3] {
    [5, 1, 5]
}
fn default_shape_density() -> u32 {
    16
}
fn default_shape_wall() -> u32 {
    1
}

impl SdfShape {
    /// Build the shape from the UI-side [`GeometryParams`].
    ///
    /// This is the single place geometry params become a producer; the split in
    /// `panel.rs` guarantees display/camera params never reach here.
    pub fn from_geometry(geometry: GeometryParams) -> Self {
        Self {
            kind: geometry.shape,
            size_blocks: geometry.size_blocks,
            voxels_per_block: geometry.voxels_per_block,
            wall_blocks: geometry.wall_blocks,
        }
    }

    /// Grid dimensions in voxels: `size_blocks * voxels_per_block`.
    pub fn grid_dimensions(&self) -> [u32; 3] {
        [
            self.size_blocks[0] * self.voxels_per_block,
            self.size_blocks[1] * self.voxels_per_block,
            self.size_blocks[2] * self.voxels_per_block,
        ]
    }

    /// Total number of sampling-grid voxels (`grid_x * grid_y * grid_z`), as
    /// `u64` so it can't overflow at large sizes/densities.
    pub fn grid_voxel_count(&self) -> u64 {
        let [grid_x, grid_y, grid_z] = self.grid_dimensions();
        grid_x as u64 * grid_y as u64 * grid_z as u64
    }

    /// Whether this shape's sampling grid exceeds [`MAX_GRID_VOXELS`] and so the
    /// 3D rebuild should be skipped (ARCHITECTURE.md §7).
    pub fn exceeds_voxel_cap(&self) -> bool {
        self.grid_voxel_count() > MAX_GRID_VOXELS
    }
}

impl VoxelProducer for SdfShape {
    fn resolve(&self, grid: &mut VoxelGrid) {
        let [grid_x, grid_y, grid_z] = self.grid_dimensions();
        grid.dimensions = [grid_x, grid_y, grid_z];

        // Shape inscribed in the box: semi-axes are half the voxel-space dims.
        let semi_axes = Vec3::new(
            grid_x as f32 / 2.0,
            grid_y as f32 / 2.0,
            grid_z as f32 / 2.0,
        );
        let wall_voxels = (self.wall_blocks * self.voxels_per_block) as f32;
        let voxels_per_block = self.voxels_per_block;

        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;

        // The outer `j` slices are order-independent (each samples a disjoint set
        // of voxels and writes nothing shared), so M8 parallelises them with
        // rayon: each slice produces a local `Vec<Voxel>` and the results are
        // concatenated. The voxel ORDER may differ from the serial version, but
        // the SET is identical — the renderer doesn't care about order, and the
        // 2D slice / `.vox` export recover indices from each voxel's position.
        let kind = self.kind;
        grid.occupied = (0..grid_y)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut local = Vec::new();
                for k in 0..grid_z {
                    for i in 0..grid_x {
                        // World-centred sample point at the voxel centre.
                        let point = Vec3::new(
                            i as f32 + 0.5 - half_x,
                            j as f32 + 0.5 - half_y,
                            k as f32 + 0.5 - half_z,
                        );

                        if signed_distance(kind, point, semi_axes, wall_voxels)
                            <= SURFACE_ISOLEVEL
                        {
                            local.push(Voxel {
                                world_position: [point.x, point.y, point.z],
                                block_local_coord: [
                                    (i % voxels_per_block) as u8,
                                    (j % voxels_per_block) as u8,
                                    (k % voxels_per_block) as u8,
                                ],
                                material_id: 0,
                            });
                        }
                    }
                }
                local
            })
            .collect();
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

/// Signed distance to an elliptical cylinder with its axis along Y.
///
/// `sdCylE(p, ax, az, ay)` in ARCHITECTURE.md §2: `semi_axis_x`/`semi_axis_z`
/// are the cross-section radii, `half_height` is the Y half-extent.
pub fn signed_distance_elliptical_cylinder(
    point: Vec3,
    semi_axis_x: f32,
    semi_axis_z: f32,
    half_height: f32,
) -> f32 {
    let radial = (glam::Vec2::new(point.x / semi_axis_x, point.z / semi_axis_z).length() - 1.0)
        * semi_axis_x.min(semi_axis_z);
    let vertical = point.y.abs() - half_height;
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
            signed_distance_elliptical_cylinder(point, semi_axis_x, semi_axis_z, semi_axis_y)
        }
        ShapeKind::Tube => {
            let outer =
                signed_distance_elliptical_cylinder(point, semi_axis_x, semi_axis_z, semi_axis_y);
            let inner = signed_distance_elliptical_cylinder(
                point,
                (semi_axis_x - wall_voxels).max(0.01),
                (semi_axis_z - wall_voxels).max(0.01),
                semi_axis_y + 1.0,
            );
            outer.max(-inner)
        }
        ShapeKind::Sphere => signed_distance_ellipsoid(point, semi_axes),
        ShapeKind::Torus => {
            let tube_radius = semi_axis_y;
            let ring_radius = (semi_axis_x.min(semi_axis_z) - tube_radius).max(0.0);
            let radial = glam::Vec2::new(point.x, point.z).length() - ring_radius;
            glam::Vec2::new(radial, point.y).length() - tube_radius
        }
        ShapeKind::Box => signed_distance_box(point, semi_axes),
    }
}

#[cfg(test)]
mod grid_overlay_bit_tests {
    use super::*;

    /// Issue #29 S4: the flag bit is the high bit (1 << 15), well clear of every
    /// real material handle (Stone/Wood/Plain ⇒ 0/1/2), so masking it off recovers
    /// the real id for the colour lookup and the bit round-trips independently.
    #[test]
    fn flag_bit_is_high_and_masks_cleanly() {
        assert_eq!(GRID_OVERLAY_BIT, 0x8000);
        for material in 0u16..=2 {
            // The bit never collides with a real material id.
            assert_eq!(material & GRID_OVERLAY_BIT, 0, "material {material} must not set the flag bit");
            // OR the bit on, then mask it off → the original material id.
            let flagged = material | GRID_OVERLAY_BIT;
            assert_ne!(flagged, material, "the bit must change the raw id");
            assert_eq!(
                material_id_color_index(flagged),
                material,
                "masking the flag bit off must recover the real material id"
            );
            // The unflagged id masks to itself (idempotent).
            assert_eq!(material_id_color_index(material), material);
        }
        // The masked id always clamps into the shader's [0, 2] colour range.
        for raw in [GRID_OVERLAY_BIT, GRID_OVERLAY_BIT | 2, 2] {
            assert!(material_id_color_index(raw).min(2) <= 2);
        }
    }
}
