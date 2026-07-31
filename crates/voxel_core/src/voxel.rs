//! The foundational voxel VALUE layer: the resolved cell, its sparse grid, the
//! frame-bearing recenter, the primitive-kind tag, and the pure signed-distance
//! functions the producers sample.
//!
//! This is the `voxel_core` value layer: it depends only DOWNWARD (on `core_geom`)
//! and NEVER on the producer half — no `SdfShape`, no `VoxelProducer`, no
//! `GeometryParams` (those live in the app-crate `voxel` module). That ⊥ is
//! load-bearing: `voxel_core` cannot import the document layer, and the crate
//! boundary compile-enforces it.
//!
//! Every value here obeys the project-wide Z-up coordinate convention (vertical = +Z,
//! array index 2; ground = XY; front = −Y) — see `docs/architecture/01-document.md`.

use glam::Vec3;

/// CPU-only iso-surface threshold. A voxel is kept when its signed distance is
/// at or below this level. NOT a uniform and NOT a UI slider.
pub const SURFACE_ISOLEVEL: f32 = 0.0;

/// Stability cap on a single shape's sampling grid volume. If
/// `grid_x * grid_y * grid_z` exceeds this, the 3D rebuild is skipped (the panel
/// shows a warning) so dragging a sphere to 16×16×16 @32 can't freeze the app.
///
/// This bounds a lone shape resolved outside the chunk path (`exceeds_voxel_cap`); the
/// chunked resolve is bounded per chunk instead, by [`MAX_CHUNK_VOXELS`].
pub const MAX_GRID_VOXELS: u64 = 6_000_000;

/// Per-chunk voxel bound: the most voxels a SINGLE chunk may hold. The deep chunked
/// resolve (the app-crate `chunk_cache`) caps each chunk, not the whole scene — so total
/// scene size is bounded only by how many chunks resolve.
///
/// One chunk's voxel CAPACITY is `(CHUNK_BLOCKS × voxels_per_block)³`: at the app
/// default density 16 that is `64³ = 262_144` voxels, comfortably under this bound.
/// The bound exists so a pathological density (where one chunk's capacity alone
/// would blow memory) is still rejected — see [`chunk_extent_exceeds_bound`].
pub const MAX_CHUNK_VOXELS: u64 = 6_000_000;

/// Whether one chunk's voxel CAPACITY at `voxels_per_block`
/// (`(CHUNK_BLOCKS × voxels_per_block)³`) exceeds the per-chunk bound
/// [`MAX_CHUNK_VOXELS`]. The chunked-resolve call sites reject a
/// density this large (a single chunk alone would exceed the bound) instead of
/// resolving it.
pub fn chunk_extent_exceeds_bound(voxels_per_block: u32) -> bool {
    let extent = (crate::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as u64;
    extent.saturating_mul(extent).saturating_mul(extent) > MAX_CHUNK_VOXELS
}

/// The parametric primitive kinds (the shape dispatcher).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ShapeKind {
    Cylinder,
    Tube,
    Sphere,
    Torus,
    Box,
}

impl ShapeKind {
    /// A sensible default bounding box for a freshly-armed primitive of this kind, in whole
    /// BLOCKS `[X, Y, Z]` (Z-up). A single shared default squashes every kind to one shape;
    /// each kind instead wants bounds matching its SDF's axis roles (see
    /// [`signed_distance`]):
    /// * **Box / Sphere** are symmetric — a cube of bounds reads as a plain box / a round ball.
    /// * **Cylinder / Tube** take their long axis on Z (the SDF's half-height), so a round
    ///   cross-section (X = Y) taller than it is wide reads as a pillar / pipe.
    /// * **Torus** sweeps its ring in the XY ground plane about +Z with `tube_radius` on Z, so
    ///   it wants wide X/Y and a small Z — a flat donut.
    pub const fn default_size_blocks(self) -> [u32; 3] {
        match self {
            ShapeKind::Box => [4, 4, 4],
            ShapeKind::Sphere => [4, 4, 4],
            ShapeKind::Cylinder => [4, 4, 6],
            ShapeKind::Tube => [4, 4, 6],
            ShapeKind::Torus => [6, 6, 2],
        }
    }
}

pub use crate::core_geom::{BlockAttrs, BlockId};

/// The composite floating-origin recenter, in voxels — the frame value every display artifact of
/// one rebuild is resolved in. A substrate frame primitive, co-located with the other
/// coordinate-frame newtypes ([`TrueWorldVoxelPoint`](substrate::spatial::TrueWorldVoxelPoint) and
/// friends) so the recenter point-crossings live in ONE audited place.
pub use substrate::spatial::RecenterVoxels;

/// One occupied voxel in the resolved grid: the chunk-local integer index plus the
/// categorical block-palette cell.
///
/// **The per-voxel record carries an INTEGER index, never an f32 position.** The absolute
/// i64 origin lives ONLY in the grid's carried frame (the chunk key / `recenter_voxels`),
/// and each cell stores its voxel index `[i, j, k]` *within that frame*. f32 is produced
/// ONLY at consumption via [`world_position`](Voxel::world_position) (`index + 0.5`). The
/// stamp keeps the integer in i64 right up to the downcast to the field, so a far-placed
/// chunk is exact rather than merely "exact for near scenes".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Voxel {
    /// Voxel index within the grid's CARRIED frame: the absolute origin
    /// (chunk key / `recenter_voxels`) lives on the grid, this is the local integer
    /// index. `i32` carries any region-scoped index (recentered grids place index 0 at
    /// a negative position) with full precision and no f32 rounding.
    pub local_index: [i32; 3],
    /// Coordinate within the owning block: `(i % d, j % d, k % d)`.
    pub block_local_coord: [u8; 3],
    /// Categorical block-palette id. The three procedural materials are Stone/Wood/Plain
    /// ⇒ 0/1/2.
    pub block_id: BlockId,
    /// Typed per-`block_id` attributes.
    pub attrs: BlockAttrs,
    /// **Transient render marker — NOT part of the categorical cell.**
    /// The owning node's `grids.voxel_grid_on_faces` flag, carried so the cuboid mesher
    /// can split a box on it and the draw can enable the on-face grid overlay. It never
    /// rides the chunk-storage codec, the `.vox` export, or the categorical id — it is a
    /// resolve→mesh render hint only, surfaced to the shader as a dedicated overlay
    /// attribute, never masked out of the material.
    pub grid_overlay: bool,
}

impl Voxel {
    /// The voxel center as an f32 position in the grid's carried frame — `index + 0.5`.
    /// This is the only place f32 is produced; the stored index stays integer.
    #[inline]
    pub fn world_position(&self) -> [f32; 3] {
        [
            self.local_index[0] as f32 + 0.5,
            self.local_index[1] as f32 + 0.5,
            self.local_index[2] as f32 + 0.5,
        ]
    }

    /// The categorical block id as the color / atlas index the renderer + `.vox`
    /// export use — the 3-value palette maps 1:1 to the color index.
    #[inline]
    pub fn color_index(&self) -> u16 {
        self.block_id.0
    }

    /// Compose this voxel's cuboid region-cell key: the clean categorical color index
    /// in the low bits, the transient on-face-grid overlay marker in the high bit.
    /// The overlay bit lives ONLY in this render-side key — never in the
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
    /// The integer voxel offset this grid's world positions were RECENTERED by
    /// (`Scene::resolve_region` subtracts it from
    /// every voxel) — **the carried frame**. A placed composite is recentered by
    /// `(min+max)/2` (= `floor(dim/2)` for a lone producer); a VoxelBody-only / bare-producer
    /// grid is corner-anchored, so this is `[0,0,0]`. Carrying it lets every consumer
    /// decode `world → index` correctly WITHOUT re-deriving the centering, which a
    /// hard-coded `floor(dim/2)` gets wrong for a corner-anchored grid. Default
    /// `[0,0,0]` is correct for any un-recentered grid.
    pub recenter_voxels: [i64; 3],
    /// The occupied voxels (sparse).
    pub occupied: Vec<Voxel>,
}

impl VoxelGrid {
    /// Create an empty grid with the given voxel dimensions (un-recentered:
    /// `recenter_voxels = [0,0,0]`; a recentered resolve sets it explicitly).
    pub fn new(dimensions: [u32; 3]) -> Self {
        Self {
            dimensions,
            recenter_voxels: [0, 0, 0],
            occupied: Vec::new(),
        }
    }

    /// Number of occupied voxels.
    pub fn occupied_count(&self) -> usize {
        self.occupied.len()
    }

    // Nothing decodes world→index, because nothing holds a world position to decode:
    // `Voxel` stores an integer `local_index` and f32 is produced only at consumption. Do
    // not re-inline a `round(world + floor(dim/2) − 0.5)` at a call site — that is the
    // frame re-derivation `recenter_voxels` exists to prevent.

    /// Measure the widest occupied voxel run (the diameter readout),
    /// restricted to the layers `[band_min, band_max]` (inclusive) along Z (Z-up:
    /// layers are Z-slices). The "widest run" is the longest contiguous span of
    /// occupied voxels along X within any single `(z, y)` row of the band.
    ///
    /// Reads the RESOLVED grid — NOT the SDF. Cheap: one
    /// pass over the sparse occupied list bucketed into per-(z,y)-row bitsets (the
    /// shared [`widest_run_over`] kernel, fed this grid's own `occupied` list).
    pub fn widest_run_in_band(&self, band_min: u32, band_max: u32) -> u32 {
        widest_run_over(self.occupied.iter(), self.dimensions, band_min, band_max)
    }
}

/// Shared kernel for the two diameter readouts: bucket every
/// voxel in `voxels` into ONE occupancy row per `(z, y)`, keyed by its GLOBAL X index
/// so a run crossing a chunk seam stays a single contiguous span in the same bitset,
/// then return the widest contiguous X run within the Z-band `[band_min, band_max]`
/// (inclusive). Z-up: the band is a Z-layer range; `k` (Z) is the layer scan.
///
/// `dimensions` (`[grid_x, grid_y, grid_z]`) gives the row width and the FLOORED
/// half-extents used to decode a voxel's centered `world_position` to integer grid
/// indices: the grid's low corner in the recentered frame is `−floor(dim/2)`, so
/// `idx = round(world + floor(dim/2) − 0.5)`. FLOORED half (`dim/2` integer division,
/// NOT `dim/2.0`) keeps the decode exact for an ODD dim too (world is half-integer).
///
/// Both [`VoxelGrid::widest_run_in_band`] (one grid's `occupied`) and
/// [`widest_run_in_band_over_chunks`] (many chunk grids' `occupied` lists flattened)
/// are thin sources over this same bucket-and-scan arithmetic — one definition, so the
/// seam-stitching decode can never drift between them.
fn widest_run_over<'voxel>(
    voxels: impl Iterator<Item = &'voxel Voxel>,
    dimensions: [u32; 3],
    band_min: u32,
    band_max: u32,
) -> u32 {
    let [grid_x, grid_y, grid_z] = dimensions;
    if grid_x == 0 || grid_y == 0 || grid_z == 0 {
        return 0;
    }
    let width = grid_x as usize;
    let half_x = (grid_x / 2) as f32;
    let half_y = (grid_y / 2) as f32;
    let half_z = (grid_z / 2) as f32;

    // Sparse (z, y)-keyed rows, built lazily so an empty band is cheap.
    let mut rows: std::collections::HashMap<u64, Vec<bool>> = std::collections::HashMap::new();
    for voxel in voxels {
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

/// **Region-scoped diameter readout.** Compute the SAME value as
/// [`VoxelGrid::widest_run_in_band`] would return for the whole region, but from a
/// SET of per-chunk grids instead of one assembled monolithic grid, so no consumer
/// needs the whole grid materialized.
///
/// `region_dimensions` are the region's voxel dimensions (`[grid_x, grid_y,
/// grid_z]`) — they define the X-axis width of each scan row and the half-extents used
/// to recover integer grid indices from a voxel's centered `world_position`. The
/// `chunk_grids` iterator yields each covering per-chunk grid whose voxels are in
/// the SAME (recentered) coordinate frame; only their `occupied` lists are read (each
/// chunk's own `dimensions` are irrelevant here).
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
/// [`VoxelGrid::widest_run_in_band`] — because it IS the same code: both funnel their
/// voxels through the shared [`widest_run_over`] kernel.
pub fn widest_run_in_band_over_chunks<'grid>(
    region_dimensions: [u32; 3],
    chunk_grids: impl IntoIterator<Item = &'grid VoxelGrid>,
    band_min: u32,
    band_max: u32,
) -> u32 {
    // Flatten every chunk's occupied list into one voxel stream: the kernel buckets
    // them into a SINGLE shared row per (z, y) keyed by GLOBAL X, so a run straddling
    // a chunk seam lands as adjacent bits in the same bitset — the seam vanishes.
    widest_run_over(
        chunk_grids
            .into_iter()
            .flat_map(|grid| grid.occupied.iter()),
        region_dimensions,
        band_min,
        band_max,
    )
}

/// Signed distance to an axis-aligned box with half-extents `box_half`.
pub fn signed_distance_box(point: Vec3, box_half: Vec3) -> f32 {
    let q = point.abs() - box_half;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Signed distance to an inscribed ellipsoid — the standard gradient-normalized
/// approximation, which is bounded but not exact.
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
/// `semi_axis_x`/`semi_axis_y` are the cross-section radii (the cylinder's circular
/// cross-section lies in the XY ground plane), `half_height` is the Z (vertical)
/// half-extent.
pub fn signed_distance_elliptical_cylinder(
    point: Vec3,
    semi_axis_x: f32,
    semi_axis_y: f32,
    half_height: f32,
) -> f32 {
    let radial = (glam::Vec2::new(point.x / semi_axis_x, point.y / semi_axis_y).length() - 1.0)
        * semi_axis_x.min(semi_axis_y);
    let vertical = point.z.abs() - half_height;
    radial.max(vertical).min(0.0) + glam::Vec2::new(radial.max(0.0), vertical.max(0.0)).length()
}

/// Dispatch to the right SDF for a shape kind.
///
/// `semi_axes` are the inscribed half-extents `(AX, AY, AZ)`; `wall_voxels` is
/// `wall * density` (Tube only).
pub fn signed_distance(shape: ShapeKind, point: Vec3, semi_axes: Vec3, wall_voxels: f32) -> f32 {
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
mod default_size_tests {
    use super::*;

    /// The per-kind default bounds read as the RIGHT shape, not one shared slab: box and
    /// sphere cubic, cylinder/tube taller than wide (a pillar on Z), torus flat and wide
    /// (a donut in the XY plane). Pins the intent by RELATIONSHIP, not by literal numbers,
    /// so retuning the sizes stays free while a kind that regresses to a wrong proportion
    /// (a flat sphere, a tall torus) fails here.
    #[test]
    fn each_kind_defaults_to_its_own_proportion() {
        let [bx, by, bz] = ShapeKind::Box.default_size_blocks();
        assert!(
            bx == by && by == bz,
            "a box default is a cube, got {bx}×{by}×{bz}"
        );
        let [sx, sy, sz] = ShapeKind::Sphere.default_size_blocks();
        assert!(
            sx == sy && sy == sz,
            "a sphere default is cubic, got {sx}×{sy}×{sz}"
        );
        for kind in [ShapeKind::Cylinder, ShapeKind::Tube] {
            let [x, y, z] = kind.default_size_blocks();
            assert_eq!(x, y, "{kind:?} has a round cross-section (X == Y)");
            assert!(
                z > x,
                "{kind:?} stands taller than wide (Z > X), got {x}×{y}×{z}"
            );
        }
        let [tx, ty, tz] = ShapeKind::Torus.default_size_blocks();
        assert_eq!(tx, ty, "a torus ring is round in the XY plane (X == Y)");
        assert!(tz < tx, "a torus is flat (Z < X), got {tx}×{ty}×{tz}");
    }
}

#[cfg(test)]
mod categorical_block_id_tests {
    use super::*;

    /// The per-voxel cell carries the categorical `block_id` ONLY — the color index IS
    /// the block id, with no render flag sharing the field and no mask.
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
            assert_eq!(
                voxel.color_index(),
                id,
                "the color index is the block id verbatim"
            );
            assert!(
                voxel.color_index() <= 2,
                "the procedural ids stay in the shader's color range"
            );
        }
    }

    /// The reconstructed f32 center is exactly `index + 0.5`, so `floor` recovers the
    /// stored integer index losslessly.
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
