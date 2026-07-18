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
        assert_eq!(a.plus(b), Rational::new(an * bd + bn * ad, ad * bd).unwrap());
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
