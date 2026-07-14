//! An exact rational number: a sign-normalized, gcd-reduced ratio of two `i128`.
//!
//! [`Rational`] represents a number exactly as `numerator / denominator`, kept in
//! **canonical form**: the sign lives on the numerator (the denominator is always
//! positive) and the pair is divided through by their greatest common divisor, so equal
//! values have identical representations and `PartialEq`/`Eq` are exact value equality.
//! This is the textbook rational-arithmetic type — exact where binary floating point
//! drifts (`1/10 + 2/10` is precisely `3/10`, never `0.30000000000000004`).
//!
//! Reduction uses **Euclid's algorithm** for the gcd, on unsigned magnitudes;
//! multiplication and addition reduce their result so magnitudes stay small and the
//! form stays canonical.
//!
//! Cite: Knuth, *The Art of Computer Programming*, vol. 2, §4.5 (rational arithmetic and
//! the Euclidean gcd). Deviation: `i128` limbs (no bignum), so a long chain of
//! operations can overflow — intended for small exact ratios (measurement expressions),
//! not arbitrary-precision arithmetic.

/// An exact, always-reduced rational number backed by `i128`. The sign is normalised
/// onto the numerator (denominator always `>= 1`) and the pair is gcd-reduced, so equal
/// rationals compare equal bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rational {
    numerator: i128,
    denominator: i128,
}

impl Rational {
    /// A reduced rational from a raw numerator/denominator. Returns `None` on a
    /// zero denominator (the only un-representable case). The sign is normalised
    /// onto the numerator so the denominator is always positive, and both are
    /// divided through by their greatest common divisor.
    pub fn new(numerator: i128, denominator: i128) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        let sign = if denominator < 0 { -1 } else { 1 };
        let mut numerator = numerator * sign;
        let mut denominator = denominator * sign;
        let divisor =
            greatest_common_divisor(numerator.unsigned_abs(), denominator.unsigned_abs()) as i128;
        if divisor > 1 {
            numerator /= divisor;
            denominator /= divisor;
        }
        Some(Self {
            numerator,
            denominator,
        })
    }

    /// A whole-number rational (`value / 1`).
    pub fn from_integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    /// The reduced numerator (sign lives here; denominator is always positive).
    pub fn numerator(self) -> i128 {
        self.numerator
    }

    /// The reduced denominator (always `>= 1`).
    pub fn denominator(self) -> i128 {
        self.denominator
    }

    /// `self * other`, reduced.
    pub fn times(self, other: Rational) -> Rational {
        // Operands are already reduced; reducing again after the cross-multiply
        // keeps the magnitudes small and the result canonical.
        Rational::new(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators multiply to a non-zero denominator")
    }

    /// `self + other`, reduced.
    pub fn plus(self, other: Rational) -> Rational {
        Rational::new(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators add to a non-zero denominator")
    }

    /// `true` when this rational is a whole number (denominator reduced to 1).
    pub fn is_integer(self) -> bool {
        self.denominator == 1
    }

    /// The whole-number value when [`is_integer`](Self::is_integer); otherwise
    /// `None`.
    pub fn to_integer(self) -> Option<i128> {
        if self.is_integer() {
            Some(self.numerator)
        } else {
            None
        }
    }

    /// The largest integer `<= self` (toward negative infinity).
    pub fn floor(self) -> i128 {
        // Truncating division rounds toward zero; for a negative non-integer that
        // is one too large, so step down.
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator < 0 {
            truncated - 1
        } else {
            truncated
        }
    }

    /// The smallest integer `>= self` (toward positive infinity).
    pub fn ceil(self) -> i128 {
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator > 0 {
            truncated + 1
        } else {
            truncated
        }
    }
}

/// Euclid's algorithm on unsigned magnitudes. `gcd(x, 0) == x`, so a `0`
/// numerator reduces against any denominator to leave the denominator as the
/// divisor (giving the canonical `0/1`).
fn greatest_common_divisor(mut first: u128, mut second: u128) -> u128 {
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    first.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rationals_do_not_drift_like_floats() {
        // 0.1 + 0.2 is the canonical f64 trap (== 0.30000000000000004). As exact
        // rationals 1/10 + 2/10 is precisely 3/10.
        let tenth = Rational::new(1, 10).unwrap();
        let fifth = Rational::new(2, 10).unwrap();
        assert_eq!(tenth.plus(fifth), Rational::new(3, 10).unwrap());
    }

    #[test]
    fn rational_floor_and_ceil_handle_signs() {
        let half = Rational::new(1, 2).unwrap();
        assert_eq!(half.floor(), 0);
        assert_eq!(half.ceil(), 1);
        let negative_half = Rational::new(-1, 2).unwrap();
        assert_eq!(negative_half.floor(), -1);
        assert_eq!(negative_half.ceil(), 0);
        let whole = Rational::from_integer(5);
        assert_eq!(whole.floor(), 5);
        assert_eq!(whole.ceil(), 5);
    }

    #[test]
    fn new_normalizes_sign_and_reduces() {
        // Sign moves onto the numerator; the pair reduces by its gcd.
        let r = Rational::new(2, -4).unwrap();
        assert_eq!(r.numerator(), -1);
        assert_eq!(r.denominator(), 2);
        // Zero denominator is the only un-representable case.
        assert_eq!(Rational::new(1, 0), None);
    }
}
