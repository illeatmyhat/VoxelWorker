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

use crate::voxel::{Voxel, VoxelGrid, VoxelProducer};
use glam::Vec3;
use rayon::prelude::*;

/// How far past the radial edge the fBm displacement may push the surface, as a
/// fraction of each cloud's radius. Keeps clouds bounded (so the gaps survive)
/// while making the edges billow. < 1.0 guarantees a cloud never reaches more
/// than `radius * (1 + this)` from its centre.
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

/// A single cloud puff.
#[derive(Debug, Clone, Copy)]
struct CloudPuff {
    /// World-centred centre (same coordinate frame as `Voxel::world_position`).
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

impl VoxelProducer for DebugCloudField {
    /// `voxels_per_block` is the document-level density (ADR 0003 §3f(0)) — only
    /// used to fill each voxel's `block_local_coord` so the block lattice / per-face
    /// texturing stay consistent with the shapes.
    fn resolve(&self, grid: &mut VoxelGrid, voxels_per_block: u32) {
        let [grid_x, grid_y, grid_z] = self.dimensions;
        grid.dimensions = self.dimensions;
        if grid_x == 0 || grid_y == 0 || grid_z == 0 {
            grid.occupied = Vec::new();
            return;
        }

        let half_x = grid_x as f32 / 2.0;
        let half_y = grid_y as f32 / 2.0;
        let half_z = grid_z as f32 / 2.0;
        let extent = Vec3::new(grid_x as f32, grid_y as f32, grid_z as f32);

        let noise = PerlinNoise::new(self.seed);
        let clouds = scatter_cloud_puffs(self.seed, extent);
        let voxels_per_block = voxels_per_block.max(1);

        // The outer `j` slices are disjoint and order-independent, so parallelise
        // them with rayon (same pattern as `SdfShape::resolve`): each slice builds
        // a local `Vec<Voxel>` and the results are concatenated. The SET is what
        // matters downstream, not the order.
        grid.occupied = (0..grid_y)
            .into_par_iter()
            .flat_map_iter(|j| {
                let mut local = Vec::new();
                for k in 0..grid_z {
                    for i in 0..grid_x {
                        let point = Vec3::new(
                            i as f32 + 0.5 - half_x,
                            j as f32 + 0.5 - half_y,
                            k as f32 + 0.5 - half_z,
                        );
                        if cloud_field_is_solid(point, &clouds, &noise) {
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

// ---------------------------------------------------------------------------
// Deterministic small RNG (LCG) — just enough for placement/jitter. Not for
// anything that needs statistical quality.
// ---------------------------------------------------------------------------

struct SmallRng {
    state: u32,
}

impl SmallRng {
    fn new(seed: u32) -> Self {
        Self {
            state: seed.wrapping_mul(2_654_435_761).wrapping_add(1),
        }
    }

    /// Next raw u32 (Numerical Recipes LCG constants).
    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// Uniform f32 in [0, 1).
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// Uniform f32 in [-1, 1).
    fn signed_unit(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Improved Perlin gradient noise (Ken Perlin, 2002) + fBm. Self-contained so the
// project takes no new dependency for a debug feature.
// ---------------------------------------------------------------------------

struct PerlinNoise {
    /// 0..255 permutation, duplicated to 512 to avoid index wrapping in `noise`.
    permutation: [u8; 512],
}

impl PerlinNoise {
    fn new(seed: u32) -> Self {
        // Identity table, shuffled deterministically (Fisher–Yates with an LCG).
        let mut table: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut random = SmallRng::new(seed);
        for i in (1..256).rev() {
            let j = (random.next_u32() as usize) % (i + 1);
            table.swap(i, j);
        }
        let mut permutation = [0u8; 512];
        for i in 0..512 {
            permutation[i] = table[i & 255];
        }
        Self { permutation }
    }

    /// Improved-Perlin 3D noise in roughly [-1, 1].
    fn noise(&self, point: Vec3) -> f32 {
        let xi = point.x.floor();
        let yi = point.y.floor();
        let zi = point.z.floor();
        let cube_x = (xi as i32 & 255) as usize;
        let cube_y = (yi as i32 & 255) as usize;
        let cube_z = (zi as i32 & 255) as usize;

        let fx = point.x - xi;
        let fy = point.y - yi;
        let fz = point.z - zi;

        let u = fade(fx);
        let v = fade(fy);
        let w = fade(fz);

        let p = &self.permutation;
        let a = p[cube_x] as usize + cube_y;
        let aa = p[a] as usize + cube_z;
        let ab = p[a + 1] as usize + cube_z;
        let b = p[cube_x + 1] as usize + cube_y;
        let ba = p[b] as usize + cube_z;
        let bb = p[b + 1] as usize + cube_z;

        let x1 = lerp(
            grad(p[aa], fx, fy, fz),
            grad(p[ba], fx - 1.0, fy, fz),
            u,
        );
        let x2 = lerp(
            grad(p[ab], fx, fy - 1.0, fz),
            grad(p[bb], fx - 1.0, fy - 1.0, fz),
            u,
        );
        let y1 = lerp(x1, x2, v);

        let x3 = lerp(
            grad(p[aa + 1], fx, fy, fz - 1.0),
            grad(p[ba + 1], fx - 1.0, fy, fz - 1.0),
            u,
        );
        let x4 = lerp(
            grad(p[ab + 1], fx, fy - 1.0, fz - 1.0),
            grad(p[bb + 1], fx - 1.0, fy - 1.0, fz - 1.0),
            u,
        );
        let y2 = lerp(x3, x4, v);

        lerp(y1, y2, w)
    }

    /// Fractional Brownian motion: summed octaves of `noise`, normalised back to
    /// roughly [-1, 1]. This is the "fractal" that turns one smooth gradient field
    /// into a cloud-like surface.
    fn fractal_noise(&self, point: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
        let mut frequency = 1.0;
        let mut amplitude = 1.0;
        let mut sum = 0.0;
        let mut normalization = 0.0;
        for _ in 0..octaves {
            sum += amplitude * self.noise(point * frequency);
            normalization += amplitude;
            amplitude *= gain;
            frequency *= lacunarity;
        }
        if normalization == 0.0 {
            0.0
        } else {
            sum / normalization
        }
    }
}

fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Perlin's gradient: pick one of 12 edge directions from the low hash bits.
fn grad(hash: u8, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let u_term = if h & 1 == 0 { u } else { -u };
    let v_term = if h & 2 == 0 { v } else { -v };
    u_term + v_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::VoxelGrid;

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
