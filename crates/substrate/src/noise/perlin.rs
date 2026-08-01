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
    #[must_use]
    pub fn new(seed: u32) -> Self {
        let mut table: [u8; 256] = std::array::from_fn(|i| u8::try_from(i).unwrap_or_default());
        let mut random = SmallRng::new(seed);
        random.shuffle(&mut table);
        let mut permutation = [0u8; 512];
        for (index, slot) in permutation.iter_mut().enumerate() {
            *slot = table.get(index & 255).copied().unwrap_or_default();
        }
        Self { permutation }
    }

    /// The seed-shuffled permutation table (512 entries). Exposed so a GPU mirror
    /// can index the SAME table as this CPU implementation.
    #[must_use]
    pub const fn permutation(&self) -> [u8; 512] {
        self.permutation
    }

    /// Improved-Perlin 3D noise in roughly `[-1, 1]`.
    #[must_use]
    pub fn noise(&self, point: Vec3) -> f32 {
        let xi = point.x.floor();
        let yi = point.y.floor();
        let zi = point.z.floor();
        let cube_x = lattice_index(xi);
        let cube_y = lattice_index(yi);
        let cube_z = lattice_index(zi);

        let fx = point.x - xi;
        let fy = point.y - yi;
        let fz = point.z - zi;

        let fade_x = fade(fx);
        let fade_y = fade(fy);
        let fade_z = fade(fz);

        let permutation = &self.permutation;
        let cube_base = usize::from(permutation_value(permutation, cube_x)).saturating_add(cube_y);
        let cube_base_z =
            usize::from(permutation_value(permutation, cube_base)).saturating_add(cube_z);
        let cube_base_next_z =
            usize::from(permutation_value(permutation, cube_base.saturating_add(1)))
                .saturating_add(cube_z);
        let next_cube_base = usize::from(permutation_value(permutation, cube_x.saturating_add(1)))
            .saturating_add(cube_y);
        let next_cube_base_z =
            usize::from(permutation_value(permutation, next_cube_base)).saturating_add(cube_z);
        let next_cube_base_next_z = usize::from(permutation_value(
            permutation,
            next_cube_base.saturating_add(1),
        ))
        .saturating_add(cube_z);

        let x1 = lerp(
            grad(permutation_value(permutation, cube_base_z), fx, fy, fz),
            grad(
                permutation_value(permutation, next_cube_base_z),
                fx - 1.0,
                fy,
                fz,
            ),
            fade_x,
        );
        let x2 = lerp(
            grad(
                permutation_value(permutation, cube_base_next_z),
                fx,
                fy - 1.0,
                fz,
            ),
            grad(
                permutation_value(permutation, next_cube_base_next_z),
                fx - 1.0,
                fy - 1.0,
                fz,
            ),
            fade_x,
        );
        let y1 = lerp(x1, x2, fade_y);

        let x3 = lerp(
            grad(
                permutation_value(permutation, cube_base_z.saturating_add(1)),
                fx,
                fy,
                fz - 1.0,
            ),
            grad(
                permutation_value(permutation, next_cube_base_z.saturating_add(1)),
                fx - 1.0,
                fy,
                fz - 1.0,
            ),
            fade_x,
        );
        let x4 = lerp(
            grad(
                permutation_value(permutation, cube_base_next_z.saturating_add(1)),
                fx,
                fy - 1.0,
                fz - 1.0,
            ),
            grad(
                permutation_value(permutation, next_cube_base_next_z.saturating_add(1)),
                fx - 1.0,
                fy - 1.0,
                fz - 1.0,
            ),
            fade_x,
        );
        let y2 = lerp(x3, x4, fade_y);

        lerp(y1, y2, fade_z)
    }

    /// Fractional Brownian motion: summed octaves of [`noise`](Self::noise) at
    /// frequency scaled by `lacunarity` and amplitude by `gain` each octave,
    /// normalized back to roughly `[-1, 1]`.
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
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
        if normalization.abs() <= f32::EPSILON {
            0.0
        } else {
            sum / normalization
        }
    }
}

/// The quintic fade curve `6t⁵ − 15t⁴ + 10t³` (Perlin 2002) — C² continuous, so
/// the interpolated field has no second-derivative creases at cell boundaries.
fn fade(t: f32) -> f32 {
    let cubic = t.mul_add(t.mul_add(6.0, -15.0), 10.0);
    t * t * t * cubic
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    t.mul_add(b - a, a)
}

#[allow(
    clippy::as_conversions,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn lattice_index(value: f32) -> usize {
    value.floor().rem_euclid(256.0) as usize
}

fn permutation_value(permutation: &[u8; 512], index: usize) -> u8 {
    permutation.get(index).copied().unwrap_or_default()
}

/// Perlin's gradient: pick one of 12 edge directions from the low hash bits.
fn grad(hash: u8, x_offset: f32, y_offset: f32, z_offset: f32) -> f32 {
    let hash_low = hash & 15;
    let first_component = if hash_low < 8 { x_offset } else { y_offset };
    let second_component = if hash_low < 4 {
        y_offset
    } else if hash_low == 12 || hash_low == 14 {
        x_offset
    } else {
        z_offset
    };
    let first_term = if hash_low & 1 == 0 {
        first_component
    } else {
        -first_component
    };
    let second_term = if hash_low & 2 == 0 {
        second_component
    } else {
        -second_component
    };
    first_term + second_term
}

#[cfg(test)]
#[allow(
    clippy::all,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::pedantic,
    clippy::nursery,
    clippy::unwrap_used
)]
mod tests {
    #![allow(
        clippy::all,
        clippy::arithmetic_side_effects,
        clippy::as_conversions,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::pedantic,
        clippy::nursery,
        clippy::unwrap_used
    )]
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
        for &p in &[
            Vec3::ZERO,
            Vec3::new(3.0, -7.0, 12.0),
            Vec3::new(-5.0, 5.0, 0.0),
        ] {
            assert!(noise.noise(p).abs() < 1e-6);
        }
    }

    #[test]
    fn fractal_noise_normalized_and_seed_sensitive() {
        let a = PerlinNoise::new(1);
        let b = PerlinNoise::new(2);
        let p = Vec3::new(0.3, 0.7, 1.1);
        assert!(a.fractal_noise(p, 4, 2.0, 0.5).abs() <= 1.5);
        // Zero octaves normalizes to zero (no divide-by-zero).
        assert_eq!(a.fractal_noise(p, 0, 2.0, 0.5), 0.0);
        // Different seeds give different fields.
        assert_ne!(
            a.fractal_noise(p, 4, 2.0, 0.5),
            b.fractal_noise(p, 4, 2.0, 0.5)
        );
    }

    #[test]
    fn permutation_is_seeded() {
        assert_ne!(
            PerlinNoise::new(1).permutation(),
            PerlinNoise::new(2).permutation()
        );
    }

    /// The proven range bound: `|noise| <= NOISE_BOUND`, and hence
    /// `|fractal_noise| <= NOISE_BOUND` for every octave count, lacunarity and gain.
    ///
    /// The argument is four steps and needs no literature constant:
    ///
    /// 1. [`noise`](PerlinNoise::noise) is nested [`lerp`]s whose weights `u, v, w` are
    ///    [`fade`] outputs in `[0, 1]`. `lerp(a, b, t) = a + t(b − a)` with `t ∈ [0,1]` is a
    ///    convex combination, so the composed result is a convex combination of the eight
    ///    corner [`grad`] values — never larger in magnitude than the largest of them.
    /// 2. Each corner offset component is `f` or `f − 1` with `f ∈ [0, 1)`, so every
    ///    component lies in `[-1, 1]`.
    /// 3. [`grad`] returns `±u ± v` where `u` and `v` are each one of those components, so
    ///    `|grad| <= 2`. With step 1, `|noise| <= 2`.
    /// 4. [`fractal_noise`](PerlinNoise::fractal_noise) divides the octave sum by the sum of
    ///    its amplitudes, making it a convex combination of `noise` samples — so it inherits
    ///    the same bound, INDEPENDENT of octaves/lacunarity/gain.
    ///
    /// The bound is deliberately loose: step 3 ignores that a corner's fade weight goes to
    /// zero exactly as its offset grows, which is why the observed maximum is far below 2
    /// (see `observed_noise_extreme_is_well_inside_the_proven_bound`). A tighter PROVEN
    /// constant would buy larger elided regions for displaced bodies; it is not needed for
    /// soundness and is left as a follow-up.
    #[test]
    fn noise_and_fractal_noise_respect_the_proven_bound() {
        const NOISE_BOUND: f32 = 2.0;
        for seed in [0u32, 1, 7, 42, 9001] {
            let noise = PerlinNoise::new(seed);
            let mut point = Vec3::new(0.017, -0.033, 0.011);
            for step in 0..40_000 {
                // Walk irrationally so samples never repeat a lattice alignment, and
                // deliberately cross negative coordinates and cell boundaries.
                point += Vec3::new(0.2113, -0.1471, 0.3079);
                if step % 997 == 0 {
                    point = -point * 1.618;
                }
                let single = noise.noise(point);
                assert!(
                    single.abs() <= NOISE_BOUND,
                    "noise({point:?}) = {single} exceeds the proven bound {NOISE_BOUND}"
                );
                // The fBm bound is octave-independent, so vary the shaping too.
                for (octaves, lacunarity, gain) in [
                    (1u32, 2.0f32, 0.5f32),
                    (4, 2.0, 0.5),
                    (8, 2.7, 0.9),
                    (3, 1.3, 1.0),
                ] {
                    let fractal = noise.fractal_noise(point, octaves, lacunarity, gain);
                    assert!(
                        fractal.abs() <= NOISE_BOUND,
                        "fractal_noise({point:?}, {octaves}, {lacunarity}, {gain}) = {fractal} \
                         exceeds the proven bound {NOISE_BOUND}"
                    );
                }
            }
        }
    }

    /// Records how much headroom the proven bound of 2 leaves, so a future tightening effort
    /// knows what it is chasing. This is an OBSERVATION, not a guarantee: the asserted value
    /// is a loose regression guard, NOT a bound anything may rely on. Only the constant in
    /// `noise_and_fractal_noise_respect_the_proven_bound` is sound to build on.
    #[test]
    fn observed_noise_extreme_is_well_inside_the_proven_bound() {
        let mut worst: f32 = 0.0;
        for seed in [0u32, 1, 7, 42, 9001] {
            let noise = PerlinNoise::new(seed);
            let mut point = Vec3::new(0.017, -0.033, 0.011);
            for _ in 0..60_000 {
                point += Vec3::new(0.2113, -0.1471, 0.3079);
                worst = worst.max(noise.noise(point).abs());
            }
        }
        // Sampling finds ~0.87 (consistent with the sqrt(3)/2 figure usually quoted for 3D
        // improved Perlin). Guard loosely so this records the observation without becoming
        // a brittle exact-value test.
        assert!(
            worst < 1.0,
            "observed noise extreme {worst} — if this ever exceeds 1.0, any code that assumed \
             a bound below the proven 2.0 must be re-audited"
        );
        assert!(
            worst > 0.5,
            "sampling found only {worst}; the walk is not exploring the field"
        );
    }
}
