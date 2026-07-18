/-
  Lean 4 (core only, no mathlib) — `Rational::new`'s gcd reduction produces CANONICAL form: the
  Euclid loop computes the gcd, dividing through by it leaves a coprime numerator/denominator pair,
  the division is exact (no truncation), and the denominator stays ≥ 1. Canonical form is what makes
  equal rationals compare equal bit-for-bit (`PartialEq`/`Eq` are exact value equality).

  Source: crates/substrate/src/interval/rational.rs — `greatest_common_divisor` (Euclid on unsigned
  magnitudes, `first.max(1)` guarding gcd(0,0)) and `new` (divide numerator and denominator by it).
  Magnitudes are `Nat` here (`i128::unsigned_abs` in the source); the sign lives on the numerator and
  is orthogonal to reduction, so the reduction theorem is about the magnitudes.

  Lean's `Nat.gcd` IS the Euclidean algorithm, so the loop is modelled by `euclid` below and proved
  equal to `Nat.gcd`; the canonical-form facts then come from core `Nat.gcd` lemmas. Holds for ALL
  natural inputs (unbounded), the algebraic tier's remit.

  Checked with:  lean verification/lean/RationalReduce.lean   (Lean 4.32.0; see ../README.md)
-/

namespace Substrate

/-- The source's `greatest_common_divisor` loop, ported: repeatedly replace `(a, b)` with
    `(b, a % b)` until the second reaches zero. (The `.max 1` tail is applied at the use sites
    below.) -/
set_option linter.unusedVariables false in
def euclid (a b : Nat) : Nat :=
  if h : b = 0 then a else euclid b (a % b)
termination_by b
decreasing_by exact Nat.mod_lt a (Nat.pos_of_ne_zero h)

/-- The loop's invariant, and the heart of its correctness: one Euclid step preserves the gcd
    (`gcd a b = gcd b (a % b)`). -/
theorem gcd_step (a b : Nat) : Nat.gcd a b = Nat.gcd b (a % b) := by
  rw [Nat.gcd_comm a b, Nat.gcd_rec b a, Nat.gcd_comm (a % b) b]

/-- The ported loop computes exactly `Nat.gcd`. -/
theorem euclid_eq_gcd (a b : Nat) : euclid a b = Nat.gcd a b := by
  induction a, b using euclid.induct with
  | case1 a => rw [euclid]; simp [Nat.gcd_zero_right]
  | case2 a b hb ih =>
    rw [euclid]
    simp only [hb, dif_neg, not_false_eq_true]
    rw [ih, ← gcd_step]

/-- Reduction divides both magnitudes EXACTLY — no truncation, the reduced pair still represents the
    same ratio. -/
theorem gcd_divides (a b : Nat) : Nat.gcd a b ∣ a ∧ Nat.gcd a b ∣ b :=
  ⟨Nat.gcd_dvd_left a b, Nat.gcd_dvd_right a b⟩

/-- Canonical form: after dividing by the gcd, numerator and denominator are COPRIME — so equal
    values reduce to identical representations. -/
theorem reduced_is_coprime (a b : Nat) (h : 0 < Nat.gcd a b) :
    Nat.Coprime (a / Nat.gcd a b) (b / Nat.gcd a b) :=
  Nat.coprime_div_gcd_div_gcd h

/-- A non-zero denominator stays ≥ 1 through reduction (the source keeps `denominator >= 1`). -/
theorem reduced_denominator_positive (a b : Nat) (hb : 1 ≤ b) : 1 ≤ b / Nat.gcd a b := by
  have hg : 0 < Nat.gcd a b := Nat.gcd_pos_of_pos_right a hb
  have hle : Nat.gcd a b ≤ b := Nat.le_of_dvd hb (Nat.gcd_dvd_right a b)
  exact Nat.one_le_div_iff hg |>.mpr hle

/-- The gcd guard: the source returns `first.max(1)`, so the reported divisor is ALWAYS ≥ 1 (it is
    never zero, even for `gcd(0, 0)`), so `new` never divides by zero. -/
theorem guarded_gcd_positive (a b : Nat) : 1 ≤ max (euclid a b) 1 := Nat.le_max_right _ 1

end Substrate
