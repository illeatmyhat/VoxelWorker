//! A tiny deterministic pseudo-random generator — a linear congruential generator
//! (LCG). Just enough for placement, jitter, and permutation shuffles; NOT for
//! anything that needs statistical quality or cryptographic strength.
//!
//! Literature: Lehmer 1951 (the LCG); the multiplier/increment are the Numerical
//! Recipes constants (Press et al., *Numerical Recipes* 3rd ed. §7.1); the
//! [`shuffle`](SmallRng::shuffle) is the Fisher–Yates / Knuth shuffle (Knuth
//! TAOCP vol. 2 §3.4.2, Algorithm P).

/// A seedable linear congruential generator over `u32` state.
pub struct SmallRng {
    state: u32,
}

impl SmallRng {
    /// Seed the generator. The seed is pre-mixed (Knuth multiplicative hash) so
    /// nearby seeds diverge immediately.
    pub fn new(seed: u32) -> Self {
        Self {
            state: seed.wrapping_mul(2_654_435_761).wrapping_add(1),
        }
    }

    /// The next raw `u32` (advances the state with the Numerical Recipes LCG
    /// constants).
    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    /// A uniform `f32` in `[0, 1)` (24-bit mantissa's worth of precision).
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    /// A uniform `f32` in `[-1, 1)`.
    pub fn signed_unit(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// Fisher–Yates in place: for `i` from `len − 1` down to `1`, swap element `i`
    /// with a uniformly-chosen element in `0..=i` (index drawn as
    /// `next_u32() % (i + 1)`). Every permutation is equally likely under a uniform
    /// generator; the result is deterministic for a given seed.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let len = slice.len();
        for i in (1..len).rev() {
            let j = (self.next_u32() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_a_seed() {
        let mut a = SmallRng::new(42);
        let mut b = SmallRng::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn unit_ranges() {
        let mut random = SmallRng::new(1);
        for _ in 0..1000 {
            let u = random.unit();
            assert!((0.0..1.0).contains(&u));
            let s = random.signed_unit();
            assert!((-1.0..1.0).contains(&s));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut random = SmallRng::new(7);
        let mut values: Vec<u32> = (0..256).collect();
        random.shuffle(&mut values);
        assert_eq!(values.len(), 256);
        let mut sorted = values.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..256).collect::<Vec<_>>());
        // With 256! orderings, a real shuffle almost surely moves the identity.
        assert_ne!(values, (0..256).collect::<Vec<_>>());
    }
}
