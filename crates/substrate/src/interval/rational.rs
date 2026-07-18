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
    /// A reduced rational from a raw numerator/denominator. The sign is normalised onto the
    /// numerator so the denominator is always positive, and both are divided through by their
    /// greatest common divisor.
    ///
    /// Returns `None` for input with no canonical `i128` form:
    /// - a **zero denominator** (the ordinary case), or
    /// - a reduced value whose magnitude overflows the asymmetric two's-complement range — a
    ///   positive numerator or a denominator of `2^127` (`|i128::MIN|`, one past `i128::MAX`).
    ///   `Rational::new(1, i128::MIN)` is `-1/2^127`, whose denominator is unrepresentable, so it is
    ///   `None`; `Rational::new(i128::MIN, -1)` is `+2^127`, likewise `None`. The mirror cases DO
    ///   have a form and are returned: `Rational::new(i128::MIN, 1)` is `i128::MIN / 1`, and
    ///   `Rational::new(i128::MIN, i128::MIN)` reduces to `1/1`.
    pub fn new(numerator: i128, denominator: i128) -> Option<Self> {
        if denominator == 0 {
            return None;
        }
        // Normalize in UNSIGNED magnitudes, never by multiplying through by a sign. `|i128::MIN|`
        // is 2^127 — one past `i128::MAX` — so `numerator * -1` overflows for the most-negative
        // input, which would panic before this `Option` could reject it. `unsigned_abs` carries
        // that magnitude exactly, and the reconstruction below rejects the results that genuinely
        // have no `i128` form.
        let negative = (numerator < 0) != (denominator < 0);
        let numerator_magnitude = numerator.unsigned_abs();
        let denominator_magnitude = denominator.unsigned_abs();
        let divisor = greatest_common_divisor(numerator_magnitude, denominator_magnitude);
        let numerator_magnitude = numerator_magnitude / divisor;
        let denominator_magnitude = denominator_magnitude / divisor;
        // The denominator is always positive, so it must fit in `i128::MAX`; only a NEGATIVE
        // numerator can use the extra step down to `i128::MIN`.
        let numerator = if negative {
            negated_from_magnitude(numerator_magnitude)?
        } else {
            i128::try_from(numerator_magnitude).ok()?
        };
        let denominator = i128::try_from(denominator_magnitude).ok()?;
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

    /// Render this rational as a **terminating** decimal string, or `None` when it
    /// has no finite base-10 expansion. Pure integer arithmetic — no `f64` anywhere,
    /// so the result is exact (`1/8` → `"0.125"`, `1/3` → `None`).
    ///
    /// A reduced fraction `p/q` terminates in base 10 iff `q` is **2/5-smooth** — its
    /// only prime factors are 2 and 5 (the prime factors of the base). The method
    /// strips factors of 2 and 5 from the denominator; if anything remains it does not
    /// terminate. Otherwise it scales the numerator up to a power of ten and splits off
    /// the fractional digits. Textbook elementary number theory (the terminating-decimal
    /// criterion; Hardy & Wright, *An Introduction to the Theory of Numbers*).
    pub fn to_terminating_decimal(self) -> Option<String> {
        if self.is_integer() {
            return Some(self.numerator.to_string());
        }
        // Strip factors of 2 and 5 from the denominator; whatever remains must be 1
        // for the decimal to terminate.
        let mut denominator = self.denominator;
        let mut factor_twos = 0;
        let mut factor_fives = 0;
        while denominator % 2 == 0 {
            denominator /= 2;
            factor_twos += 1;
        }
        while denominator % 5 == 0 {
            denominator /= 5;
            factor_fives += 1;
        }
        if denominator != 1 {
            return None;
        }
        // Scale numerator/denominator up to a power of ten, then split off the
        // fractional digits.
        let fractional_digits = factor_twos.max(factor_fives);
        let mut scaled_numerator = self.numerator;
        for _ in 0..(fractional_digits - factor_twos) {
            scaled_numerator *= 2;
        }
        for _ in 0..(fractional_digits - factor_fives) {
            scaled_numerator *= 5;
        }
        let scale = 10i128.pow(fractional_digits as u32);
        let negative = scaled_numerator < 0;
        let magnitude = scaled_numerator.unsigned_abs();
        let whole_part = (magnitude / scale as u128) as i128;
        let fraction_part = (magnitude % scale as u128) as i128;
        let mut fraction_text =
            format!("{fraction_part:0width$}", width = fractional_digits as usize);
        while fraction_text.ends_with('0') {
            fraction_text.pop();
        }
        let sign = if negative { "-" } else { "" };
        if fraction_text.is_empty() {
            Some(format!("{sign}{whole_part}"))
        } else {
            Some(format!("{sign}{whole_part}.{fraction_text}"))
        }
    }
}

/// Rebuild a NEGATIVE `i128` from its unsigned magnitude, or `None` when no `i128` represents it.
///
/// The two's-complement range is asymmetric: `i128::MIN` is `-2^127` but `i128::MAX` is only
/// `2^127 - 1`, so the magnitude `2^127` is representable as a negative value and NOT as a positive
/// one. That asymmetry is the whole reason [`Rational::new`] normalizes in magnitudes — negating
/// through `* -1` would overflow on exactly this value.
fn negated_from_magnitude(magnitude: u128) -> Option<i128> {
    /// `|i128::MIN|` — one past `i128::MAX`, so it needs the explicit case below.
    const MOST_NEGATIVE_MAGNITUDE: u128 = i128::MAX as u128 + 1;
    if magnitude == MOST_NEGATIVE_MAGNITUDE {
        return Some(i128::MIN);
    }
    Some(-(i128::try_from(magnitude).ok()?))
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

/// Kani bounded-model-checking probes of the `i128` arithmetic — the overflow edge the deductive
/// (Verus) and algebraic (Lean) tiers deliberately do NOT cover (exact `Rat` reasoning cannot see a
/// limb overflow). Two questions: is the arithmetic overflow-free and correct across the INTENDED
/// measurement domain (small exact ratios), and where exactly does the raw `i128` boundary bite?
/// `#[cfg(kani)]` keeps them out of ordinary builds. Run under WSL: `cargo kani -p substrate`.
///
/// ## Runtime — read before wiring these into CI
///
/// Measured 2026-07-18, all four together under `-j` (solve time only; the build was ~1.4 s):
/// **550 s wall-clock**, the slowest harness ~547 s, and the concrete-input boundary harness
/// 0.16 s. (Terse output does not label which thread ran which harness, so per-harness attribution
/// beyond the concrete one is inference; the total is exact.)
///
/// The two binary-operator harnesses dominate because each chains DATA-DEPENDENT Euclid loops —
/// `greatest_common_divisor` does `first % second` with a SYMBOLIC divisor, the single worst shape
/// for CBMC — and unwinding multiplies against the symbolic domain.
///
/// What did NOT help, recorded so nobody re-tries it: tightening the domain `±200`→`±8` (the cost is
/// the loop chain, not the bound), and `CARGO_TARGET_DIR` on a Linux FS (a WSL-only fix for the slow
/// `/mnt/c` mount — the build is ~1.4 s either way, and a native CI runner is unaffected).
///
/// What DID help: removing gcd chains from the harness rather than from the code. Building operands
/// straight from their fields instead of through `new`, and dropping a reduced-form assertion that
/// `new`'s own harness already proves, took the chain from five Euclid loops to two and roughly
/// halved the worst harness. The remaining cost is the gcd INSIDE `times`/`plus`, which can only go
/// by changing production code (a binary/Stein gcd, trading Knuth-cited Euclid for a subtler
/// algorithm in COLD code purely to please a verifier) or by `#[kani::stub]`-ing the gcd (which stops
/// verifying the real one). Neither is worth it at an epic cadence.
///
/// So: these belong in a proof job run **at EPIC boundaries, NOT per-commit** (an epic is this repo's
/// unit of work, and it ties the pass to when the proven code actually changed — nightly would burn
/// runner minutes on days nothing in `substrate` moved). For scale, the whole `substrate` unit suite
/// is 118 tests in 0.02 s. Run the pass with `cargo kani -p substrate -j --output-format=terse`
/// (`-j` verifies harnesses on parallel threads and REQUIRES terse output; it turned ~21 min of
/// serial solving into ~9 min wall-clock).
///
/// These proofs do not replace the unit tests below: they are `#[cfg(kani)]`, so they are invisible
/// to `cargo test`/`clippy`/CI, and the tests remain the only always-on check that the shipping
/// binary still implements what is proved here.
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// A reduced rational drawn from the intended measurement domain: small numerator, small
    /// positive denominator. The bound is deliberately TIGHT — each `times`/`plus` harness unwinds
    /// several of Euclid's loops symbolically, and the properties they check (commutativity,
    /// canonical form) are structural, so a wider domain costs solver time without covering a new
    /// case. Overflow-freedom is NOT the interesting claim at this bound (it is trivially true);
    /// the real `i128` boundary is probed separately below.
    /// Built DIRECTLY from its fields rather than through `Rational::new`, which would run an
    /// Euclid loop per operand. `times`/`plus` read the raw fields and re-canonicalize through
    /// `new`, so their commutativity does not depend on the operands already being reduced — only
    /// on the type's structural invariant, a positive denominator. Skipping `new` here removes two
    /// of the harness's data-dependent loop chains, which is what the solver actually pays for.
    fn measurement_rational() -> Rational {
        let numerator: i128 = kani::any();
        let denominator: i128 = kani::any();
        kani::assume(numerator >= -8 && numerator <= 8);
        kani::assume(denominator >= 1 && denominator <= 8);
        Rational {
            numerator,
            denominator,
        }
    }

    /// `new` is overflow-free and produces canonical form (positive, gcd-reduced denominator) over
    /// the whole measurement domain — the reduction proved abstractly in `RationalReduce.lean`, now
    /// on the real `i128` code with the overflow checks live.
    #[kani::proof]
    #[kani::unwind(31)]
    fn new_is_overflow_free_and_reduced_in_the_measurement_domain() {
        let numerator: i128 = kani::any();
        let denominator: i128 = kani::any();
        kani::assume(numerator >= -200 && numerator <= 200);
        kani::assume(denominator >= 1 && denominator <= 200);
        let reduced = Rational::new(numerator, denominator).unwrap();
        assert!(reduced.denominator() >= 1);
        assert!(
            greatest_common_divisor(
                reduced.numerator().unsigned_abs(),
                reduced.denominator().unsigned_abs()
            ) == 1
        );
    }

    /// `times` is commutative over the measurement domain. The result's canonical form is NOT
    /// re-asserted here — `times` returns `new`'s output, and
    /// `new_is_overflow_free_and_reduced_in_the_measurement_domain` already proves `new` yields
    /// canonical form. Dropping that assertion removes another Euclid loop from the chain.
    #[kani::proof]
    #[kani::unwind(11)]
    fn times_is_commutative() {
        let a = measurement_rational();
        let b = measurement_rational();
        assert!(a.times(b) == b.times(a));
    }

    /// `plus` is commutative over the measurement domain (canonical form covered as above).
    #[kani::proof]
    #[kani::unwind(11)]
    fn plus_is_commutative() {
        let a = measurement_rational();
        let b = measurement_rational();
        assert!(a.plus(b) == b.plus(a));
    }

    /// The raw-boundary probe that FOUND the `i128::MIN` overflow (`new` used to sign-normalize by
    /// `numerator * sign` / `denominator * sign`, and `i128::MIN * -1` is unrepresentable, so it
    /// panicked before the `Option` guard could reject it). Now magnitude-normalized, `new` returns
    /// a value where one exists and `None` where none does — asserted here at every corner of the
    /// asymmetric two's-complement range. Concrete inputs, so this solves in seconds.
    #[kani::proof]
    fn new_handles_the_i128_min_boundary_without_overflow() {
        // Unrepresentable ⇒ None. `1/i128::MIN` needs denominator 2^127; `i128::MIN/-1` is +2^127.
        assert!(Rational::new(1, i128::MIN).is_none());
        assert!(Rational::new(i128::MIN, -1).is_none());
        // Representable ⇒ Some, in canonical form.
        let most_negative = Rational::new(i128::MIN, 1).expect("i128::MIN / 1 is representable");
        assert!(most_negative.numerator() == i128::MIN && most_negative.denominator() == 1);
        // MIN/MIN reduces to 1/1 — this also exercises a gcd of 2^127, which the old
        // `greatest_common_divisor(..) as i128` cast wrapped to a NEGATIVE divisor.
        let unity = Rational::new(i128::MIN, i128::MIN).expect("MIN/MIN is 1");
        assert!(unity.numerator() == 1 && unity.denominator() == 1);
        // Zero keeps its canonical 0/1 form even against the most-negative denominator.
        let zero = Rational::new(0, i128::MIN).expect("0/MIN is 0");
        assert!(zero.numerator() == 0 && zero.denominator() == 1);
        // The zero denominator remains the ordinary rejection.
        assert!(Rational::new(i128::MIN, 0).is_none());
    }
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
    fn terminating_decimal_expansion() {
        let dec = |n, d| Rational::new(n, d).unwrap().to_terminating_decimal();
        assert_eq!(dec(1, 8), Some("0.125".to_string())); // 2-smooth
        assert_eq!(dec(1, 10), Some("0.1".to_string())); // 2·5
        assert_eq!(dec(3, 4), Some("0.75".to_string()));
        assert_eq!(dec(-7, 20), Some("-0.35".to_string())); // sign carried
        assert_eq!(dec(5, 1), Some("5".to_string())); // integer
        assert_eq!(Rational::from_integer(42).to_terminating_decimal(), Some("42".to_string()));
        // Non-2/5-smooth denominators do not terminate.
        assert_eq!(dec(1, 3), None);
        assert_eq!(dec(2, 7), None);
        assert_eq!(dec(1, 6), None); // 6 = 2·3, the 3 blocks it
    }

    /// The asymmetric two's-complement boundary. `|i128::MIN|` is `2^127`, one past `i128::MAX`, so
    /// normalizing the sign by multiplying through by `-1` used to OVERFLOW here (a panic escaping a
    /// `pub fn` whose contract is to return `None` instead). Found by the Kani harness
    /// `new_handles_the_i128_min_boundary_without_overflow`; `new` now normalizes in magnitudes.
    #[test]
    fn new_handles_i128_min_without_overflowing() {
        // No canonical i128 form ⇒ None (rather than a panic).
        assert_eq!(Rational::new(1, i128::MIN), None, "-1/2^127: denominator unrepresentable");
        assert_eq!(Rational::new(i128::MIN, -1), None, "+2^127: numerator unrepresentable");
        // The mirror cases DO have a form, and keep it.
        let most_negative = Rational::new(i128::MIN, 1).expect("i128::MIN / 1 is representable");
        assert_eq!((most_negative.numerator(), most_negative.denominator()), (i128::MIN, 1));
        // Reduces to 1/1 — and exercises a gcd of 2^127, which the old `as i128` cast wrapped
        // negative.
        let unity = Rational::new(i128::MIN, i128::MIN).expect("MIN/MIN is 1");
        assert_eq!((unity.numerator(), unity.denominator()), (1, 1));
        // Zero stays canonical 0/1 even against the most-negative denominator.
        let zero = Rational::new(0, i128::MIN).expect("0/MIN is 0");
        assert_eq!((zero.numerator(), zero.denominator()), (0, 1));
        // A zero denominator is still the ordinary rejection.
        assert_eq!(Rational::new(i128::MIN, 0), None);
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
