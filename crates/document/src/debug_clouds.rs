//! Debug cloud field: a [`VoxelProducer`] that fills the grid with several
//! visually distinct, billowy cloud blobs separated by empty space. It exists to
//! exercise the renderer and the onion-skin fog with richer content than the five
//! parametric shapes — a single connected SDF can't show how the pipeline handles
//! many disjoint objects scattered through a large, mostly-empty volume.
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
/// while making the edges billow. < 1.0 guarantees a cloud never reaches more
/// than `radius * (1 + this)` from its centre.
///
/// `pub` so the GPU view-resolve (ADR 0007) shares the EXACT same constant with its
/// WGSL port (one source of truth — a drift would break the §6 exact-parity net).
pub const CLOUD_EDGE_BILLOW: f32 = 0.42;

/// fBm octave count / shaping. Four octaves is plenty for a readable cloud at
/// these grid sizes; more just adds sub-voxel detail. `pub`: shared with the GPU port.
pub const CLOUD_NOISE_OCTAVES: u32 = 4;
pub const CLOUD_NOISE_LACUNARITY: f32 = 2.0;
pub const CLOUD_NOISE_GAIN: f32 = 0.5;

/// Noise wavelength as a fraction of a cloud's radius. ~0.6 puts a few billows
/// across each cloud (wavelength a bit smaller than the cloud), which reads as
/// fluffy rather than either smooth (too large) or noisy (too small). `pub`: shared
/// with the GPU port.
pub const CLOUD_NOISE_WAVELENGTH_FRACTION: f32 = 0.6;

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

/// One cloud puff's resolved parameters, flattened for the GPU view-resolve (ADR
/// 0007). The producer evaluates `distance + fBm` against these on the GPU exactly as
/// [`cloud_field_is_solid`] does on the CPU.
#[derive(Debug, Clone, Copy)]
pub struct CloudPuffParams {
    /// World-centred centre (same frame as the centred SDF sample `idx + 0.5 - grid/2`).
    pub center: [f32; 3],
    /// Base radius in voxels (before noise displacement).
    pub radius: f32,
    /// Per-cloud offset into the noise field.
    pub noise_offset: [f32; 3],
}

impl DebugCloudField {
    /// The resolved cloud puffs (the GPU view-resolve streams these), computed from
    /// `seed` + `dimensions` EXACTLY as [`resolve_into`](DebugCloudField::resolve_into)
    /// does — same `scatter_cloud_puffs`, so the GPU eval matches the CPU bit-for-bit.
    pub fn gpu_puffs(&self) -> Vec<CloudPuffParams> {
        let extent = Vec3::new(
            self.dimensions[0] as f32,
            self.dimensions[1] as f32,
            self.dimensions[2] as f32,
        );
        scatter_cloud_puffs(self.seed, extent)
            .into_iter()
            .map(|cloud| CloudPuffParams {
                center: cloud.center.to_array(),
                radius: cloud.radius,
                noise_offset: cloud.noise_offset.to_array(),
            })
            .collect()
    }

    /// The seed-shuffled Perlin permutation table (the GPU view-resolve streams it so
    /// its WGSL noise indexes the SAME table as the CPU `PerlinNoise`).
    pub fn permutation_table(&self) -> [u8; 512] {
        PerlinNoise::new(self.seed).permutation()
    }
}

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

    /// The cloud field is displaced by FRACTAL PERLIN NOISE (fBm), whose value over a
    /// cell has no cheap conservative bracket — so this producer is UNBOUNDABLE and
    /// returns `None` (ADR 0010 Decision 2). A `None` consumer treats every cell as
    /// BOUNDARY and resolves it per-voxel: still EXACT, just unelided. (This is also the
    /// trait default; the explicit override documents the intent at the producer.)
    fn cell_field_interval(
        &self,
        _cell_local_voxels: voxel_core::spatial_index::VoxelAabb,
        _voxels_per_block: u32,
    ) -> Option<crate::voxel::FieldInterval> {
        None
    }

    fn full_dimensions(&self, _voxels_per_block: u32) -> [u32; 3] {
        self.dimensions
    }
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
