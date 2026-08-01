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

/// An exact, always-reduced rational backed by `i128`.
/// The denominator is positive, and equal values have identical representations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Rational {
    numerator: i128,
    denominator: i128,
}

/// Why an IEEE-754 value cannot enter the bounded exact-rational representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RationalFromF64Error {
    /// `NaN` and infinities do not name rational values.
    NonFinite,
    /// The exact binary ratio needs an `i128` numerator or denominator outside this type's range.
    OutOfRange,
}

const F64_FRACTION_BITS: u32 = 52;
const F64_FRACTION_BITS_I32: i32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;
const F64_HIDDEN_BIT: u64 = 1_u64 << F64_FRACTION_BITS;
const F64_FRACTION_MASK: u64 = F64_HIDDEN_BIT - 1;

impl Rational {
    /// A reduced rational from a raw numerator/denominator. The sign is normalized onto the
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
    #[must_use]
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
        let numerator_magnitude = numerator_magnitude.checked_div(divisor).unwrap_or_default();
        let denominator_magnitude = denominator_magnitude
            .checked_div(divisor)
            .unwrap_or_default();
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
    pub const fn from_integer(value: i128) -> Self {
        Self {
            numerator: value,
            denominator: 1,
        }
    }

    /// Rebuild the exact binary value of a finite `f64`, without decimal conversion or rounding.
    ///
    /// IEEE-754 stores a finite value as a signed integer significand times a power of two. Powers
    /// of two are stripped from the significand before this type's canonical constructor receives
    /// the resulting ratio. `-0.0` is the same rational as `0.0`, so both become canonical `0/1`.
    ///
    /// # Errors
    ///
    /// Returns [`RationalFromF64Error::NonFinite`] for `NaN` or infinity, and
    /// [`RationalFromF64Error::OutOfRange`] when the exact ratio cannot fit the bounded `i128`
    /// representation. Finiteness alone does not guarantee that bound: a nonzero subnormal needs
    /// a denominator of at least `2^1023`, while this type can represent at most `2^126`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_from_f64_exact(value: f64) -> Result<Self, RationalFromF64Error> {
        if !value.is_finite() {
            return Err(RationalFromF64Error::NonFinite);
        }
        if value == 0.0 {
            return Ok(Self::from_integer(0));
        }

        let bits = value.to_bits();
        let negative = bits >> 63 != 0;
        let raw_exponent = (bits >> F64_FRACTION_BITS) & 0x7ff;
        let fraction = bits & F64_FRACTION_MASK;
        let (significand, exponent) = if raw_exponent == 0 {
            (fraction, -1074)
        } else {
            let exponent = i32::try_from(raw_exponent)
                .map_err(|_| RationalFromF64Error::OutOfRange)?
                .checked_sub(F64_EXPONENT_BIAS)
                .and_then(|exponent| exponent.checked_sub(F64_FRACTION_BITS_I32))
                .ok_or(RationalFromF64Error::OutOfRange)?;
            (F64_HIDDEN_BIT | fraction, exponent)
        };
        let removed_twos = significand.trailing_zeros();
        let significand = significand >> removed_twos;
        let exponent = exponent
            .checked_add(i32::try_from(removed_twos).map_err(|_| RationalFromF64Error::OutOfRange)?)
            .ok_or(RationalFromF64Error::OutOfRange)?;

        if exponent >= 0 {
            let magnitude = u128::from(significand)
                .checked_shl(u32::try_from(exponent).map_err(|_| RationalFromF64Error::OutOfRange)?)
                .ok_or(RationalFromF64Error::OutOfRange)?;
            let numerator = if negative {
                negated_from_magnitude(magnitude).ok_or(RationalFromF64Error::OutOfRange)?
            } else {
                i128::try_from(magnitude).map_err(|_| RationalFromF64Error::OutOfRange)?
            };
            Self::new(numerator, 1).ok_or(RationalFromF64Error::OutOfRange)
        } else {
            let denominator = 1_u128
                .checked_shl(exponent.unsigned_abs())
                .ok_or(RationalFromF64Error::OutOfRange)?;
            let denominator =
                i128::try_from(denominator).map_err(|_| RationalFromF64Error::OutOfRange)?;
            let numerator = i128::from(significand);
            let numerator = if negative {
                numerator
                    .checked_neg()
                    .ok_or(RationalFromF64Error::OutOfRange)?
            } else {
                numerator
            };
            Self::new(numerator, denominator).ok_or(RationalFromF64Error::OutOfRange)
        }
    }

    /// The nearest IEEE-754 value to this exact ratio.
    #[must_use]
    #[allow(clippy::as_conversions, clippy::cast_precision_loss)]
    pub fn to_f64(self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// The reduced numerator (sign lives here; denominator is always positive).
    #[must_use]
    pub const fn numerator(self) -> i128 {
        self.numerator
    }

    /// The reduced denominator (always `>= 1`).
    #[must_use]
    pub const fn denominator(self) -> i128 {
        self.denominator
    }

    /// `self * other`, reduced.
    ///
    /// # Panics
    ///
    /// Panics if the cross-products do not fit in `i128` or the reduced result cannot be
    /// represented by this type.
    #[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
    pub fn times(self, other: Self) -> Self {
        // Operands are already reduced; reducing again after the cross-multiply
        // keeps the magnitudes small and the result canonical.
        Self::new(
            self.numerator * other.numerator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators multiply to a non-zero denominator")
    }

    /// `self + other`, reduced.
    ///
    /// # Panics
    ///
    /// Panics if the cross-products or their sum do not fit in `i128`, or the reduced result
    /// cannot be represented by this type.
    #[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
    pub fn plus(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator + other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators add to a non-zero denominator")
    }

    /// `self - other`, reduced.
    ///
    /// # Panics
    ///
    /// Panics if the cross-products or their difference do not fit in `i128`, or the reduced
    /// result cannot be represented by this type.
    #[allow(clippy::arithmetic_side_effects, clippy::expect_used)]
    pub fn minus(self, other: Self) -> Self {
        Self::new(
            self.numerator * other.denominator - other.numerator * self.denominator,
            self.denominator * other.denominator,
        )
        .expect("non-zero denominators subtract to a non-zero denominator")
    }

    /// `-self`.
    ///
    /// `None` for `i128::MIN / 1` alone: its magnitude is one past `i128::MAX`, so the
    /// negation has no `i128` form — the same asymmetry [`new`](Self::new) documents. The
    /// value is already reduced and negating cannot change that, so this rebuilds directly
    /// rather than going through a subtraction that would overflow before it could report.
    #[must_use]
    pub fn negated(self) -> Option<Self> {
        Some(Self {
            numerator: self.numerator.checked_neg()?,
            denominator: self.denominator,
        })
    }

    /// `self / other`, reduced. `None` when `other` is zero, or when the reduced result has
    /// no `i128` form.
    ///
    /// Unlike [`times`](Self::times) and [`plus`](Self::plus) this can genuinely fail, so it
    /// returns an `Option` rather than asserting: dividing by a user-authored expression that
    /// evaluates to zero is an ordinary authoring mistake, not a bug.
    #[must_use]
    pub fn divided_by(self, other: Self) -> Option<Self> {
        let numerator = self.numerator.checked_mul(other.denominator)?;
        let denominator = self.denominator.checked_mul(other.numerator)?;
        Self::new(numerator, denominator)
    }

    /// `true` when this rational is a whole number (denominator reduced to 1).
    #[must_use]
    pub const fn is_integer(self) -> bool {
        self.denominator == 1
    }

    /// The whole-number value when [`is_integer`](Self::is_integer); otherwise
    /// `None`.
    #[must_use]
    pub const fn to_integer(self) -> Option<i128> {
        if self.is_integer() {
            Some(self.numerator)
        } else {
            None
        }
    }

    /// The largest integer `<= self` (toward negative infinity).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub const fn floor(self) -> i128 {
        // Truncating division rounds toward zero; for a negative non-integer that
        // is one too large, so step down.
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator < 0 {
            truncated.saturating_sub(1)
        } else {
            truncated
        }
    }

    /// The smallest integer `>= self` (toward positive infinity).
    #[must_use]
    #[allow(clippy::arithmetic_side_effects)]
    pub const fn ceil(self) -> i128 {
        let truncated = self.numerator / self.denominator;
        if self.numerator % self.denominator != 0 && self.numerator > 0 {
            truncated.saturating_add(1)
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
    #[must_use]
    pub fn to_terminating_decimal(self) -> Option<String> {
        if self.is_integer() {
            return Some(self.numerator.to_string());
        }
        // Strip factors of 2 and 5 from the denominator; whatever remains must be 1
        // for the decimal to terminate.
        let mut denominator = self.denominator;
        let mut factor_twos: usize = 0;
        let mut factor_fives: usize = 0;
        while denominator.checked_rem(2).unwrap_or_default() == 0 {
            denominator = denominator.checked_div(2).unwrap_or_default();
            factor_twos = factor_twos.saturating_add(1);
        }
        while denominator.checked_rem(5).unwrap_or_default() == 0 {
            denominator = denominator.checked_div(5).unwrap_or_default();
            factor_fives = factor_fives.saturating_add(1);
        }
        if denominator != 1 {
            return None;
        }
        // Scale numerator/denominator up to a power of ten, then split off the
        // fractional digits.
        let fractional_digits = factor_twos.max(factor_fives);
        let mut scaled_numerator = self.numerator;
        for _ in 0..fractional_digits.saturating_sub(factor_twos) {
            scaled_numerator = scaled_numerator.checked_mul(2)?;
        }
        for _ in 0..fractional_digits.saturating_sub(factor_fives) {
            scaled_numerator = scaled_numerator.checked_mul(5)?;
        }
        let exponent = u32::try_from(fractional_digits).unwrap_or_default();
        let scale = 10i128.pow(exponent);
        let negative = scaled_numerator < 0;
        let magnitude = scaled_numerator.unsigned_abs();
        let scale = u128::try_from(scale).unwrap_or_default();
        let whole_part = magnitude.checked_div(scale)?;
        let fraction_part = magnitude.checked_rem(scale)?;
        let mut fraction_text = format!("{fraction_part:0fractional_digits$}");
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
    const MOST_NEGATIVE_MAGNITUDE: u128 = 170_141_183_460_469_231_731_687_303_715_884_105_728u128;
    if magnitude == MOST_NEGATIVE_MAGNITUDE {
        return Some(i128::MIN);
    }
    i128::try_from(magnitude).ok()?.checked_neg()
}

/// Euclid's algorithm on unsigned magnitudes. `gcd(x, 0) == x`, so a `0`
/// numerator reduces against any denominator to leave the denominator as the
/// divisor (giving the canonical `0/1`).
fn greatest_common_divisor(mut first: u128, mut second: u128) -> u128 {
    while second != 0 {
        let remainder = first.checked_rem(second).unwrap_or_default();
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
/// ## Runtime — and the cost lesson that shaped these
///
/// These are cheap (seconds). They were NOT: two earlier harnesses proved `times`/`plus` commutative
/// and cost ~606 s and ~666 s. The lesson is worth keeping, because the fix was not a faster solver:
///
/// * **Cost tracks DATA-DEPENDENT LOOP CHAINS, not bounds.** `greatest_common_divisor` does
///   `first % second` with a SYMBOLIC divisor — the worst shape for CBMC, and on `i128` it builds a
///   full 128-bit division circuit per unwound iteration REGARDLESS of how tightly the inputs are
///   assumed. That is why tightening the domain `±200`→`±8` bought almost nothing.
/// * **`CARGO_TARGET_DIR` on a Linux FS is a WSL-only fix** for the slow `/mnt/c` mount — the build
///   is ~1.4 s either way, and a native CI runner is unaffected. It never touched solve time.
/// * **The real fix was asking what the harness proved.** Swapping the operands of `times`/`plus`
///   yields the *same argument expressions* to `new`, so commutativity follows from `i128` `*`/`+`
///   commuting and the gcd is irrelevant to it — ~21 minutes of solving for no information about
///   this code. Replaced by a unit test (which catches the transposition typo that was the only real
///   risk) plus the overflow-envelope harness above, which proves something genuinely unknown.
/// * **`unwind` must be DERIVED, not guessed — this dominated everything else.** A guessed
///   `unwind(31)` over `±200` made the reduction anchor take **462 s, 99% of the entire `substrate`
///   Kani tier**; Lamé's bound gives `unwind(10)` at `±64`, taking it to 72 s and the whole
///   three-tier battery from 479 s to 94 s. Guessing LOW is safe (loud unwinding-assertion failure);
///   guessing HIGH is silently expensive.
/// * **Profile before theorizing.** The envelope harness above was predicted to cost ~290 s from
///   arithmetic on totals; measured, it is **3.8 s**. Pair each `Thread N: Checking harness <name>`
///   line with that thread's `Verification Time` — terse output does carry the attribution.
///
/// So: before optimizing a slow harness, check it is worth running at all; then cut loop chains out
/// of the HARNESS before touching production code. Rejected on purpose: a binary/Stein gcd (a subtler
/// algorithm in COLD code purely to please a verifier) and `#[kani::stub]`-ing the gcd (which stops
/// verifying the real one).
///
/// Run the tier with `cargo kani -p substrate -j --output-format=terse` — `-j` verifies harnesses on
/// parallel threads and REQUIRES terse output. Or run all three tiers via `verification/run-all.sh`.
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
    /// `new` is overflow-free and produces canonical form (positive, gcd-reduced denominator) over
    /// the measurement domain. This is the ANCHOR tying `lean/RationalReduce.lean` — which proves
    /// coprime reduction for ALL naturals — to the real `i128` code with the overflow checks live,
    /// so a modest domain does the job; the unbounded claim lives in the Lean proof and the extremes
    /// in `new_handles_the_i128_min_boundary_without_overflow`.
    ///
    /// The `unwind` bound is DERIVED, not guessed. By Lamé's theorem, Euclid's worst case for a
    /// given magnitude is a consecutive Fibonacci pair; under 64 that is `(55, 34)`, which takes 8
    /// iterations, so 10 covers it with margin. This matters: every surplus unwind inlines another
    /// full 128-bit division circuit, and this harness runs TWO gcd chains (inside `new`, and the
    /// coprimality assertion). A guessed `unwind(31)` over `±200` cost 462 s — 99% of the whole
    /// `substrate` Kani tier. Setting it too low fails loudly with an unwinding assertion, so
    /// deriving it is safe.
    #[kani::proof]
    #[kani::unwind(10)]
    fn new_is_overflow_free_and_reduced_in_the_measurement_domain() {
        let numerator: i128 = kani::any();
        let denominator: i128 = kani::any();
        kani::assume(numerator >= -64 && numerator <= 64);
        kani::assume(denominator >= 1 && denominator <= 64);
        let reduced = Rational::new(numerator, denominator).unwrap();
        assert!(reduced.denominator() >= 1);
        assert!(
            greatest_common_divisor(
                reduced.numerator().unsigned_abs(),
                reduced.denominator().unsigned_abs()
            ) == 1
        );
    }

    /// **The overflow envelope** — the one genuinely unverified thing about `times`/`plus`, and the
    /// source's own documented deviation ("a long chain of operations can overflow"). Both operators
    /// CROSS-MULTIPLY before reducing, so the products, not the reduction, are where `i128` gives
    /// out. This establishes the safe operating envelope:
    ///
    /// > if every component of both operands fits an `i64`, neither `times` nor `plus` can overflow.
    ///
    /// That bound is tight enough to be useful and tight enough to be true only just: `plus` forms
    /// `an·bd + bn·ad`, whose magnitude reaches `2^127 − 2^64`, a hair under `i128::MAX = 2^127 − 1`.
    ///
    /// This mirrors the exact argument expressions rather than calling `times`/`plus`, because the
    /// real calls route through `new`'s Euclid loop and gcd over `2^63`-wide operands needs ~90
    /// unwound iterations of a 128-bit division circuit — not BMC-tractable. The mirror is anchored
    /// to the real operators by `times_and_plus_are_the_cross_multiply_expressions` in the unit tests
    /// below. (Same mirror-and-anchor shape as `ValueCube`'s `row_major_index` proof.)
    #[kani::proof]
    fn i64_bounded_components_cannot_overflow_times_or_plus() {
        let (a_numerator, a_denominator): (i128, i128) = (kani::any(), kani::any());
        let (b_numerator, b_denominator): (i128, i128) = (kani::any(), kani::any());
        kani::assume(a_numerator >= i64::MIN as i128 && a_numerator <= i64::MAX as i128);
        kani::assume(b_numerator >= i64::MIN as i128 && b_numerator <= i64::MAX as i128);
        // Denominators are positive by the type's invariant.
        kani::assume(a_denominator >= 1 && a_denominator <= i64::MAX as i128);
        kani::assume(b_denominator >= 1 && b_denominator <= i64::MAX as i128);

        // `times` computes exactly these two products ...
        let _ = a_numerator * b_numerator;
        let _ = a_denominator * b_denominator;
        // ... and `plus` these (the denominator product is shared). Kani's arithmetic-overflow
        // checks on each are the proof; no assertion is needed.
        let _ = a_numerator * b_denominator + b_numerator * a_denominator;
    }

    /// The raw-boundary probe of the `i128::MIN` corner. Sign-normalizing by
    /// `numerator * sign` / `denominator * sign` would panic on `i128::MIN * -1` before the
    /// `Option` guard could reject it; `new` normalizes in magnitudes, so it returns a value
    /// where one exists and `None` where none does — asserted here at every corner of the
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
    #![allow(clippy::unwrap_used)]
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
        assert_eq!(
            Rational::from_integer(42).to_terminating_decimal(),
            Some("42".to_string())
        );
        // Non-2/5-smooth denominators do not terminate.
        assert_eq!(dec(1, 3), None);
        assert_eq!(dec(2, 7), None);
        assert_eq!(dec(1, 6), None); // 6 = 2·3, the 3 blocks it
    }

    /// `times`/`plus` are commutative. This was briefly a Kani harness, which cost ~10 minutes each
    /// to prove something that follows from commutativity of `i128` `*` and `+`: swapping the
    /// operands yields the *same argument expressions* to `new`, so the gcd inside it is irrelevant
    /// to the property and the solver was re-deriving school arithmetic. What that harness could
    /// actually catch was a TRANSPOSITION TYPO (`other.den * other.den` for `self.den * other.den`),
    /// which asymmetric operands catch here for free. Deliberately asymmetric in numerator,
    /// denominator, and sign so a swapped term cannot coincidentally agree.
    #[test]
    fn times_and_plus_are_commutative() {
        let a = Rational::new(-3, 7).unwrap();
        let b = Rational::new(5, 11).unwrap();
        assert_eq!(a.times(b), b.times(a));
        assert_eq!(a.plus(b), b.plus(a));

        // A second pair whose cross terms differ in magnitude AND sign.
        let c = Rational::new(9, 2).unwrap();
        let d = Rational::new(-4, 13).unwrap();
        assert_eq!(c.times(d), d.times(c));
        assert_eq!(c.plus(d), d.plus(c));
    }

    /// Anchors the Kani harness `i64_bounded_components_cannot_overflow_times_or_plus`, which proves
    /// the overflow envelope on MIRRORED cross-multiply expressions (the real calls route through
    /// `new`'s Euclid loop, untractable at `2^63`-wide operands). This pins that the mirror is what
    /// `times`/`plus` actually compute, so the envelope transfers to the real operators.
    #[test]
    fn times_and_plus_are_the_cross_multiply_expressions() {
        let a = Rational::new(-3, 7).unwrap();
        let b = Rational::new(5, 11).unwrap();
        let (an, ad) = (a.numerator(), a.denominator());
        let (bn, bd) = (b.numerator(), b.denominator());

        assert_eq!(a.times(b), Rational::new(an * bn, ad * bd).unwrap());
        assert_eq!(
            a.plus(b),
            Rational::new(an * bd + bn * ad, ad * bd).unwrap()
        );
    }

    /// The asymmetric two's-complement boundary. `|i128::MIN|` is `2^127`, one past `i128::MAX`, so
    /// normalizing the sign by multiplying through by `-1` would overflow — a panic escaping a
    /// `pub fn` whose contract is to return `None` instead. `new` normalizes in magnitudes.
    #[test]
    fn new_handles_i128_min_without_overflowing() {
        // No canonical i128 form ⇒ None (rather than a panic).
        assert_eq!(
            Rational::new(1, i128::MIN),
            None,
            "-1/2^127: denominator unrepresentable"
        );
        assert_eq!(
            Rational::new(i128::MIN, -1),
            None,
            "+2^127: numerator unrepresentable"
        );
        // The mirror cases DO have a form, and keep it.
        let most_negative = Rational::new(i128::MIN, 1).expect("i128::MIN / 1 is representable");
        assert_eq!(
            (most_negative.numerator(), most_negative.denominator()),
            (i128::MIN, 1)
        );
        // Reduces to 1/1 — and exercises a gcd of 2^127, which an `as i128` cast wraps
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

    #[test]
    fn subtraction_and_division_are_exact() {
        let third = Rational::new(1, 3).expect("non-zero denominator");
        let two_thirds = Rational::new(2, 3).expect("non-zero denominator");
        assert_eq!(two_thirds.minus(third), third);
        // 1/3 divided by 1/3 is exactly one, where the f64 chain drifts.
        assert_eq!(third.divided_by(third), Some(Rational::from_integer(1)));
        assert_eq!(
            two_thirds.divided_by(Rational::from_integer(2)),
            Some(third)
        );
    }

    #[test]
    fn dividing_by_zero_reports_rather_than_panicking() {
        // An authored expression that evaluates to zero is an ordinary mistake, so this is
        // the one arithmetic door that returns an Option instead of asserting.
        let half = Rational::new(1, 2).expect("non-zero denominator");
        assert_eq!(half.divided_by(Rational::from_integer(0)), None);
    }

    #[test]
    fn negation_is_exact_and_refuses_only_the_unrepresentable_extreme() {
        let half = Rational::new(1, 2).expect("non-zero denominator");
        assert_eq!(half.negated(), Rational::new(-1, 2));
        assert_eq!(half.negated().and_then(Rational::negated), Some(half));
        // |i128::MIN| is one past i128::MAX, so its negation has no form. Rebuilding
        // directly (rather than subtracting from zero) is what lets this REPORT instead of
        // overflowing on the way to the check.
        assert_eq!(Rational::from_integer(i128::MIN).negated(), None);
    }

    #[test]
    fn finite_f64_conversion_rebuilds_accepted_binary_values_exactly() {
        let awkward_solver_value = f64::from_bits(0x405e_dd2f_1a9f_be77);
        for value in [
            -180.0,
            -0.125,
            123.4567,
            awkward_solver_value,
            2f64.powi(-126),
        ] {
            let rational = Rational::try_from_f64_exact(value).expect("fits i128 rational");
            assert_eq!(
                rational.to_f64().to_bits(),
                value.to_bits(),
                "{value} must round-trip through its exact binary ratio"
            );
        }
    }

    #[test]
    fn finite_f64_conversion_canonicalizes_zero_and_rejects_nonfinite_values() {
        for value in [0.0, -0.0] {
            let rational = Rational::try_from_f64_exact(value).expect("zero is finite");
            assert_eq!((rational.numerator(), rational.denominator()), (0, 1));
            assert_eq!(rational.to_f64().to_bits(), 0.0f64.to_bits());
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                Rational::try_from_f64_exact(value),
                Err(RationalFromF64Error::NonFinite)
            );
        }
    }

    #[test]
    fn finite_f64_conversion_reports_the_i128_representation_boundary() {
        assert_eq!(
            Rational::try_from_f64_exact(f64::from_bits(1)),
            Err(RationalFromF64Error::OutOfRange),
            "a nonzero subnormal needs a denominator beyond i128"
        );
        assert_eq!(
            Rational::try_from_f64_exact(2f64.powi(-127)),
            Err(RationalFromF64Error::OutOfRange),
            "2^127 is one past the largest positive denominator"
        );
        assert_eq!(
            Rational::try_from_f64_exact(2f64.powi(127)),
            Err(RationalFromF64Error::OutOfRange),
            "+2^127 has no positive i128 numerator"
        );
        let negative = Rational::try_from_f64_exact(-2f64.powi(127))
            .expect("the negative i128 extreme is representable");
        assert_eq!(
            (negative.numerator(), negative.denominator()),
            (i128::MIN, 1)
        );
    }

    #[test]
    fn finite_f64_conversion_covers_a_deterministic_ieee_normal_matrix() {
        const EXPONENTS: [u64; 5] = [949, 1000, 1023, 1075, 1149];
        const FRACTIONS: [u64; 3] = [0, 1, F64_FRACTION_MASK];

        for sign in [0_u64, 1_u64 << 63] {
            for exponent in EXPONENTS {
                for fraction in FRACTIONS {
                    let value = f64::from_bits(sign | (exponent << F64_FRACTION_BITS) | fraction);
                    let rational = Rational::try_from_f64_exact(value)
                        .expect("the representative normal fits the bounded rational");
                    assert_eq!(rational.to_f64().to_bits(), value.to_bits(), "{value:?}");
                }
            }
        }
    }

    #[test]
    fn finite_f64_conversion_rejects_deliberately_outside_ieee_values() {
        let denominator_overflow = f64::from_bits((948_u64 << F64_FRACTION_BITS) | 1);
        let numerator_overflow = f64::from_bits((1150_u64 << F64_FRACTION_BITS) | 1);
        for value in [
            f64::from_bits(1),
            denominator_overflow,
            -denominator_overflow,
            numerator_overflow,
            -numerator_overflow,
        ] {
            assert_eq!(
                Rational::try_from_f64_exact(value),
                Err(RationalFromF64Error::OutOfRange),
                "{value:?} is finite but outside the i128 ratio envelope"
            );
        }

        let positive_limit = Rational::try_from_f64_exact(2f64.powi(126))
            .expect("the largest positive power of two below 2^127 fits");
        assert_eq!(
            (positive_limit.numerator(), positive_limit.denominator()),
            (1_i128 << 126, 1)
        );
        let negative_fraction = Rational::try_from_f64_exact(-2f64.powi(-126))
            .expect("the negative denominator boundary fits");
        assert_eq!(
            (
                negative_fraction.numerator(),
                negative_fraction.denominator()
            ),
            (-1, 1_i128 << 126)
        );
    }
}
