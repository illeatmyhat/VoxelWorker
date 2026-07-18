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

    /// The PROVEN range bound (ADR 0021 Decision 1): `|noise| <= NOISE_BOUND`, and hence
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
                for (octaves, lacunarity, gain) in
                    [(1u32, 2.0f32, 0.5f32), (4, 2.0, 0.5), (8, 2.7, 0.9), (3, 1.3, 1.0)]
                {
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
        assert!(worst > 0.5, "sampling found only {worst}; the walk is not exploring the field");
    }
}
