//! Improved Perlin gradient noise (Ken Perlin, 2002, *Improving Noise*) and the
//! fractional Brownian motion (fBm) built on it. Self-contained — the only
//! dependency is glam for the sample point and this crate's [`SmallRng`] for the
//! permutation shuffle.
//!
//! The classic construction: a 256-entry permutation table (seed-shuffled,
//! duplicated to 512 to avoid index wrapping), the quintic fade `6t⁵ − 15t⁴ +
//! 10t³`, and Perlin's 12-edge gradient hash. [`PerlinNoise::fractal_noise`] sums
//! octaves at rising frequency and falling amplitude (Mandelbrot–Van Ness 1968;
//! Musgrave's fBm) to turn one smooth field into fractal detail.
//!
//! This module is the readable CPU specification of the noise WGSL port; a
//! parity net keeps the two byte-identical (the permutation table and constants
//! are streamed to the shader rather than duplicated).

use super::rng::SmallRng;
use glam::Vec3;

/// Seed-shuffled improved-Perlin gradient noise over 3D space.
pub struct PerlinNoise {
    /// 0..255 permutation, duplicated to 512 to avoid index wrapping in [`noise`](Self::noise).
    permutation: [u8; 512],
}

impl PerlinNoise {
    /// Build the noise from a seed: an identity table shuffled deterministically
    /// (Fisher–Yates with an LCG), then duplicated to 512.
    pub fn new(seed: u32) -> Self {
        let mut table: [u8; 256] = std::array::from_fn(|i| i as u8);
        let mut random = SmallRng::new(seed);
        random.shuffle(&mut table);
        let mut permutation = [0u8; 512];
        for i in 0..512 {
            permutation[i] = table[i & 255];
        }
        Self { permutation }
    }

    /// The seed-shuffled permutation table (512 entries). Exposed so a GPU mirror
    /// can index the SAME table as this CPU implementation.
    pub fn permutation(&self) -> [u8; 512] {
        self.permutation
    }

    /// Improved-Perlin 3D noise in roughly `[-1, 1]`.
    pub fn noise(&self, point: Vec3) -> f32 {
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

        let x1 = lerp(grad(p[aa], fx, fy, fz), grad(p[ba], fx - 1.0, fy, fz), u);
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

    /// Fractional Brownian motion: summed octaves of [`noise`](Self::noise) at
    /// frequency scaled by `lacunarity` and amplitude by `gain` each octave,
    /// normalised back to roughly `[-1, 1]`.
    pub fn fractal_noise(&self, point: Vec3, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
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

/// The quintic fade curve `6t⁵ − 15t⁴ + 10t³` (Perlin 2002) — C² continuous, so
/// the interpolated field has no second-derivative creases at cell boundaries.
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

    #[test]
    fn fade_endpoints_and_midpoint() {
        assert_eq!(fade(0.0), 0.0);
        assert_eq!(fade(1.0), 1.0);
        assert!((fade(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn noise_is_deterministic_and_bounded() {
        let noise = PerlinNoise::new(1);
        let p = Vec3::new(1.5, -2.25, 3.75);
        assert_eq!(noise.noise(p), noise.noise(p));
        // Improved Perlin stays within a small constant of [-1, 1].
        for i in 0..500 {
            let t = i as f32 * 0.37;
            let value = noise.noise(Vec3::new(t, t * 0.5, -t));
            assert!(value.abs() <= 1.5, "noise out of range: {value}");
        }
    }

    #[test]
    fn integer_lattice_points_are_zero() {
        // Gradient noise vanishes at integer lattice points by construction.
        let noise = PerlinNoise::new(9);
        for &p in &[Vec3::ZERO, Vec3::new(3.0, -7.0, 12.0), Vec3::new(-5.0, 5.0, 0.0)] {
            assert!(noise.noise(p).abs() < 1e-6);
        }
    }

    #[test]
    fn fractal_noise_normalised_and_seed_sensitive() {
        let a = PerlinNoise::new(1);
        let b = PerlinNoise::new(2);
        let p = Vec3::new(0.3, 0.7, 1.1);
        assert!(a.fractal_noise(p, 4, 2.0, 0.5).abs() <= 1.5);
        // Zero octaves normalises to zero (no divide-by-zero).
        assert_eq!(a.fractal_noise(p, 0, 2.0, 0.5), 0.0);
        // Different seeds give different fields.
        assert_ne!(a.fractal_noise(p, 4, 2.0, 0.5), b.fractal_noise(p, 4, 2.0, 0.5));
    }

    #[test]
    fn permutation_is_seeded() {
        assert_ne!(PerlinNoise::new(1).permutation(), PerlinNoise::new(2).permutation());
    }
}
