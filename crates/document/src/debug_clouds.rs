//! Debug cloud field: a [`VoxelProducer`] that fills the grid with several
//! visually distinct, billowy cloud blobs separated by empty space. It exists to
//! exercise the renderer and the onion skin with richer content than the parametric
//! shapes — a single connected SDF can't show how the pipeline handles many disjoint
//! objects scattered through a large, mostly-empty volume. (The onion skin is now
//! ghost-shaded clip-slab passes rather than the volumetric fog ADR 0012 deleted; the
//! feature this exercises is live, only its implementation changed.)
//!
//! Recipe (the standard one for cloud-like volumes): each cloud is a soft RADIAL
//! FALLOFF (so it stays a bounded, separate puff with gaps around it) whose
//! surface is then displaced by FRACTAL PERLIN NOISE (fBm — summed octaves of
//! gradient noise). Gradient/Perlin fBm reads soft and fluffy, which suits a
//! cloud; the alternative, Worley/cellular (Voronoi) noise, gives lumpier,
//! more "cauliflower" clouds and is worth trying if you want a different look,
//! but fBm is the better default here. Each cloud takes its own noise offset and
//! radius so no two read alike.
//!
//! Deterministic: a fixed `seed` drives both the cloud placement and the noise
//! permutation, so the same parameters always resolve to the same field (good for
//! a reproducible debug object and for golden-image tests later).

use voxel_core::voxel::{Voxel, VoxelGrid};
use crate::voxel::{VoxelProducer};
use glam::Vec3;
use rayon::prelude::*;
use substrate::noise::{PerlinNoise, SmallRng};

/// How far past the radial edge the fBm displacement may push the surface, as a
/// fraction of each cloud's radius. Keeps clouds bounded (so the gaps survive)
/// while making the edges billow.
const CLOUD_EDGE_BILLOW: f32 = 0.42;

/// fBm octave count / shaping. Four octaves is plenty for a readable cloud at
/// these grid sizes; more just adds sub-voxel detail.
const CLOUD_NOISE_OCTAVES: u32 = 4;
const CLOUD_NOISE_LACUNARITY: f32 = 2.0;
const CLOUD_NOISE_GAIN: f32 = 0.5;

/// Noise wavelength as a fraction of a cloud's radius. ~0.6 puts a few billows
/// across each cloud (wavelength a bit smaller than the cloud), which reads as
/// fluffy rather than either smooth (too large) or noisy (too small).
const CLOUD_NOISE_WAVELENGTH_FRACTION: f32 = 0.6;

/// The PROVEN bound on `|fractal_noise|`, from `substrate::noise::perlin` (ADR 0021
/// Decision 1): noise is a convex combination of gradient dot-products each bounded by 2,
/// and fBm normalises by its amplitude sum so it inherits the same bound for ANY octave
/// count, lacunarity or gain. Deliberately loose — the observed extreme is around 0.87 —
/// but sound without depending on an unproven literature constant. It sets only how deep a
/// puff's provably-solid core reaches; the air side does not use it.
const NOISE_RANGE_BOUND: f32 = 2.0;

/// How far a puff can CLAIM, in units of `CLOUD_EDGE_BILLOW`, irrespective of the noise's
/// true range. [`cloud_field_is_solid`]'s `radial < -CLOUD_EDGE_BILLOW` early-out is
/// **semantics, not an optimisation**: a puff is skipped outright beyond that radius, so it
/// can never claim a point past `radius * (1 + CLOUD_EDGE_BILLOW)` even if the noise were to
/// exceed 1. That is what makes the AIR verdict exact and independent of
/// [`NOISE_RANGE_BOUND`]. Deleting the early-out as "just a fast path" would both change the
/// resolved geometry and make this classifier's air bound unsound.
const NOISE_CLAIM_REACH: f32 = 1.0;

/// A single cloud puff.
#[derive(Debug, Clone, Copy)]
struct CloudPuff {
    /// World-centred centre (same coordinate frame as `Voxel::world_position()`).
    center: Vec3,
    /// Base radius in voxels (before noise displacement).
    radius: f32,
    /// Per-cloud offset into the noise field, so each cloud looks different.
    noise_offset: Vec3,
}

/// A field of distinct noise-displaced cloud puffs scattered through the grid.
#[derive(Debug, Clone, Copy)]
pub struct DebugCloudField {
    /// Voxel-space grid dimensions (X, Y, Z).
    pub dimensions: [u32; 3],
    /// Seed for the deterministic placement + noise permutation.
    pub seed: u32,
}

// `CloudPuffParams`, `gpu_puffs` and `permutation_table` were DELETED 2026-07-18. They
// existed only to flatten this producer's puffs + Perlin table for the ADR 0007 GPU
// view-resolve to stream into WGSL; ADR 0012 deleted that evaluator (it was the fog's, and
// the fog went with it), leaving all three with zero callers. The CPU `resolve_into` below
// computes the same puffs via `scatter_cloud_puffs` and is the only path. Restore from git
// history if a GPU producer mirror returns.

impl VoxelProducer for DebugCloudField {
    /// `voxels_per_block` is the document-level density (ADR 0003 §3f(0)) — only
    /// used to fill each voxel's `block_local_coord` so the block lattice / per-face
    /// texturing stay consistent with the shapes.
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [full_x, full_y, full_z] = self.dimensions;
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
        let [grid_x, grid_y, grid_z] = self.dimensions;
        // FULL dimensions even when only a window is written.
        grid.dimensions = self.dimensions;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            grid.occupied = Vec::new();
            return;
        }

        // ALL field math (half_*, the extent driving puff placement, the noise) is
        // derived from the FULL `self.dimensions` — the window only narrows the
        // iterated cell range, so a windowed resolve is a byte-identical subset.
        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;
        let extent = Vec3::new(grid_x as f32, grid_y as f32, grid_z as f32);

        let noise = PerlinNoise::new(self.seed);
        let clouds = scatter_cloud_puffs(self.seed, extent);
        let voxels_per_block = voxels_per_block.max(1);

        // Clamp the window to `[0, full_dim)`; a full-window call reproduces the
        // historical `0..grid_*` loops exactly.
        let [(win_x_lo, win_x_hi), (win_y_lo, win_y_hi), (win_z_lo, win_z_hi)] =
            crate::voxel::clamp_window_to_grid(window_local_voxels, self.dimensions);

        // The outer `j` slices are disjoint and order-independent, so parallelise
        // them with rayon (same pattern as `SdfShape::resolve`): each slice builds
        // a local `Vec<Voxel>` and the results are concatenated. The SET is what
        // matters downstream, not the order. Windowing parallelises over the
        // WINDOWED j range.
        grid.occupied = (win_y_lo..win_y_hi)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut local = Vec::new();
                for k in win_z_lo..win_z_hi {
                    for i in win_x_lo..win_x_hi {
                        // SAMPLE the field at the centred coordinate (`idx + 0.5 −
                        // half`) so the cloud geometry is unchanged, but STORE the voxel
                        // CORNER-ANCHORED (`idx + 0.5`) exactly like `SdfShape` /
                        // `SketchSolid`: the centre is a half-integer for any grid
                        // size → always on the global voxel lattice, and the cloud
                        // resolves in the SAME frame as the Tools it mixes with.
                        let sample = Vec3::new(
                            i as f32 + 0.5 - half_x,
                            j as f32 + 0.5 - half_y,
                            k as f32 + 0.5 - half_z,
                        );
                        if cloud_field_is_solid(sample, &clouds, &noise) {
                            local.push(Voxel {
                                local_index: [
                                    i as i32,
                                    j as i32,
                                    k as i32,
                                ],
                                block_local_coord: [
                                    (i % voxels_per_block) as u8,
                                    (j % voxels_per_block) as u8,
                                    (k % voxels_per_block) as u8,
                                ],
                                block_id: voxel_core::core_geom::BlockId::DEFAULT,
                                attrs: voxel_core::core_geom::BlockAttrs::DEFAULT,
                                grid_overlay: false,
                            });
                        }
                    }
                }
                local
            })
            .collect();
    }

    /// Conservative bracket on the cloud field over a block cell (ADR 0021).
    ///
    /// This producer was long documented as UNBOUNDABLE on the grounds that fBm "has no
    /// cheap conservative bracket over a cell". That reasoning was wrong, and it cost real
    /// interior elision. Bracketing the fBm *over a cell* is indeed hard; it is also never
    /// needed. Only the noise's GLOBAL RANGE is required, after which the radial term does
    /// all the work — so a cell is classified from puff geometry alone, with **no noise
    /// evaluation at all**.
    ///
    /// Per puff, with `radial = 1 − d/R` and the solidity test `radial + BILLOW·fbm > 0`:
    ///
    /// * **AIR** when every puff's NEAREST approach exceeds `R(1 + BILLOW)`. Exact, and
    ///   independent of the noise bound — see [`NOISE_CLAIM_REACH`].
    /// * **COARSE-SOLID** when some puff's FARTHEST reach is within `R(1 − BILLOW·B)`, so
    ///   even a worst-case negative billow cannot retract past the cell.
    /// * **BOUNDARY** otherwise, resolved per-voxel — still exact, just unelided.
    ///
    /// Returned intervals are the genuine bracket on the field (negated, since this field is
    /// POSITIVE inside while [`FieldInterval`] is negative-inside), never sentinel values.
    ///
    /// [`FieldInterval`]: crate::voxel::FieldInterval
    fn cell_field_interval(
        &self,
        cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        _voxels_per_block: u32,
    ) -> Option<crate::voxel::FieldInterval> {
        let dimensions = self.dimensions;
        // Clamp to the resolved index range: voxels outside `[0, full_dim)` are never
        // written, so they are air and cannot make a cell solid.
        let mut lo_index = [0i64; 3];
        let mut hi_index = [0i64; 3];
        for axis in 0..3 {
            lo_index[axis] = cell_local_voxels.min[axis].max(0);
            hi_index[axis] = cell_local_voxels.max[axis].min(dimensions[axis] as i64);
            if lo_index[axis] >= hi_index[axis] {
                // No resolvable voxel in this cell ⇒ provably AIR.
                return Some(crate::voxel::FieldInterval::new(1.0, 2.0));
            }
        }
        // A cell that pokes outside the extent holds clamped-away air, so it can never be
        // claimed COARSE-SOLID however deep inside a puff it sits.
        let fully_inside_extent = (0..3).all(|axis| {
            cell_local_voxels.min[axis] >= 0
                && cell_local_voxels.max[axis] <= dimensions[axis] as i64
        });

        // The resolve samples index `i` at `i + 0.5 − half` (see `resolve_into`), so the
        // sampled points of this cell span the CLOSED box below — its exact convex hull,
        // not the continuous cell. Bracketing over the samples is what classification needs.
        let half = Vec3::new(
            dimensions[0] as f32 / 2.0,
            dimensions[1] as f32 / 2.0,
            dimensions[2] as f32 / 2.0,
        );
        let sample_lo = Vec3::new(
            lo_index[0] as f32 + 0.5 - half.x,
            lo_index[1] as f32 + 0.5 - half.y,
            lo_index[2] as f32 + 0.5 - half.z,
        );
        let sample_hi = Vec3::new(
            hi_index[0] as f32 - 0.5 - half.x,
            hi_index[1] as f32 - 0.5 - half.y,
            hi_index[2] as f32 - 0.5 - half.z,
        );

        let extent = Vec3::new(
            dimensions[0] as f32,
            dimensions[1] as f32,
            dimensions[2] as f32,
        );
        let clouds = scatter_cloud_puffs(self.seed, extent);

        // Bracket `max over puffs of (radial + BILLOW·fbm)` — the value the resolve tests
        // against zero. Seeded below any real contribution: with no puffs the field is
        // everywhere unclaimed.
        let mut strongest_lower = f32::NEG_INFINITY;
        let mut strongest_upper = f32::NEG_INFINITY;
        for cloud in &clouds {
            if cloud.radius <= 0.0 {
                continue;
            }
            let (nearest, farthest) = box_distance_bounds(cloud.center, sample_lo, sample_hi);
            // Worst-case billow shrinks the puff; the reject caps how far it can grow.
            let lower = 1.0 - farthest / cloud.radius - CLOUD_EDGE_BILLOW * NOISE_RANGE_BOUND;
            let upper = 1.0 - nearest / cloud.radius + CLOUD_EDGE_BILLOW * NOISE_CLAIM_REACH;
            strongest_lower = strongest_lower.max(lower);
            strongest_upper = strongest_upper.max(upper);
        }
        if !strongest_lower.is_finite() || !strongest_upper.is_finite() {
            // No puffs contribute ⇒ provably AIR.
            return Some(crate::voxel::FieldInterval::new(1.0, 2.0));
        }

        // Negate into the negative-inside convention, rounding each endpoint OUTWARD (the
        // never-narrower contract). The resolve is solid on `field > 0` STRICTLY while
        // `classify` is inside on `field <= isolevel`, so a bracket that merely touches zero
        // must not read as solid — the outward rounding on `maximum` guarantees that.
        let mut minimum = (-strongest_upper).next_down();
        let mut maximum = (-strongest_lower).next_up();
        if !fully_inside_extent && maximum <= 0.0 {
            // Clamped-away air forbids a solid verdict; keep the interval straddling.
            maximum = 0.0f32.next_up();
        }
        if minimum > maximum {
            minimum = maximum;
        }
        Some(crate::voxel::FieldInterval::new(minimum, maximum))
    }

    fn full_dimensions(&self, _voxels_per_block: u32) -> [u32; 3] {
        self.dimensions
    }
}

/// Nearest and farthest Euclidean distance from `point` to the closed axis-aligned box
/// `[lo, hi]`. Used to bracket a puff's radial term over a whole cell without sampling it.
///
/// Nearest: per axis the excursion outside the slab (zero when the point is within it),
/// combined as a length — the standard point-to-AABB distance. Farthest: per axis the larger
/// of the two face gaps, which is attained at some corner, so the combined length is exact.
fn box_distance_bounds(point: Vec3, lo: Vec3, hi: Vec3) -> (f32, f32) {
    let mut nearest_squared = 0.0f32;
    let mut farthest_squared = 0.0f32;
    for axis in 0..3 {
        let (p, l, h) = (point[axis], lo[axis], hi[axis]);
        let outside = (l - p).max(p - h).max(0.0);
        nearest_squared += outside * outside;
        let widest = (p - l).abs().max((p - h).abs());
        farthest_squared += widest * widest;
    }
    (nearest_squared.sqrt(), farthest_squared.sqrt())
}

/// Whether `point` lands inside any cloud puff. The field is the per-cloud radial
/// falloff (1 at the centre, 0 at the base radius, negative beyond) plus the fBm
/// displacement; the voxel is solid when the strongest cloud's field clears zero.
/// Taking the MAX across clouds keeps the puffs separate (they only merge where
/// they actually overlap).
fn cloud_field_is_solid(point: Vec3, clouds: &[CloudPuff], noise: &PerlinNoise) -> bool {
    for cloud in clouds {
        let distance = (point - cloud.center).length();
        let radial = 1.0 - distance / cloud.radius;
        // Cheap reject: if even a full positive billow couldn't reach the iso,
        // this cloud can't claim the point — skip the noise evaluation.
        if radial < -CLOUD_EDGE_BILLOW {
            continue;
        }
        let wavelength = cloud.radius * CLOUD_NOISE_WAVELENGTH_FRACTION;
        let frequency = 1.0 / wavelength.max(1.0);
        let billow = noise.fractal_noise(
            (point + cloud.noise_offset) * frequency,
            CLOUD_NOISE_OCTAVES,
            CLOUD_NOISE_LACUNARITY,
            CLOUD_NOISE_GAIN,
        );
        if radial + CLOUD_EDGE_BILLOW * billow > 0.0 {
            return true;
        }
    }
    false
}

/// Place the cloud puffs on the eight octant centres of the volume, each jittered
/// and sized deterministically from the seed so they read as eight distinct clouds
/// with clear gaps between them. Radii stay small enough (relative to the
/// half-octant spacing) that even fully billowed they don't bridge the gaps.
fn scatter_cloud_puffs(seed: u32, extent: Vec3) -> Vec<CloudPuff> {
    let mut random = SmallRng::new(seed ^ 0x9e37_79b9);
    let min_extent = extent.x.min(extent.y).min(extent.z);

    let mut clouds = Vec::with_capacity(8);
    for octant in 0..8u32 {
        // Octant centre at ±0.25 of the extent on each axis.
        let sign = |bit: u32| if (octant >> bit) & 1 == 0 { -1.0 } else { 1.0 };
        let base = Vec3::new(
            sign(0) * extent.x * 0.25,
            sign(1) * extent.y * 0.25,
            sign(2) * extent.z * 0.25,
        );
        // Jitter within the octant so the lattice doesn't read as a grid.
        let jitter = Vec3::new(
            random.signed_unit() * extent.x * 0.06,
            random.signed_unit() * extent.y * 0.06,
            random.signed_unit() * extent.z * 0.06,
        );
        // Radius 10–15% of the smallest extent: distinct sizes, generous gaps.
        let radius = min_extent * (0.10 + 0.05 * random.unit());
        clouds.push(CloudPuff {
            center: base + jitter,
            radius,
            noise_offset: Vec3::new(
                random.unit() * 100.0,
                random.unit() * 100.0,
                random.unit() * 100.0,
            ),
        });
    }
    clouds
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::voxel::VoxelGrid;

    #[test]
    fn resolves_some_but_not_all_voxels() {
        // A cloud field should be mostly empty space with distinct solid blobs:
        // neither empty nor a filled box.
        let field = DebugCloudField {
            dimensions: [64, 64, 64],
            seed: 1,
        };
        let mut grid = VoxelGrid::new(field.dimensions);
        field.resolve(&mut grid, 16);

        let total = 64 * 64 * 64;
        let filled = grid.occupied_count();
        assert!(filled > 0, "cloud field resolved to empty");
        assert!(
            filled < total / 3,
            "cloud field too dense ({filled}/{total}); expected lots of empty space"
        );
    }

    #[test]
    fn is_deterministic() {
        let field = DebugCloudField {
            dimensions: [48, 48, 48],
            seed: 7,
        };
        let mut a = VoxelGrid::new(field.dimensions);
        let mut b = VoxelGrid::new(field.dimensions);
        field.resolve(&mut a, 16);
        field.resolve(&mut b, 16);
        assert_eq!(a.occupied_count(), b.occupied_count());
    }
}
