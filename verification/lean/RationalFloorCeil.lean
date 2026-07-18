/-
  Lean 4 (core only, no mathlib) — the `Rational::floor` / `Rational::ceil` algorithm proved to
  compute the TRUE mathematical floor and ceiling, over EVERY integer numerator.

  Source: crates/substrate/src/interval/rational.rs. A `Rational` is `numerator / denominator` in
  canonical form with `denominator ≥ 1`. Rust integer `/` and `%` truncate toward zero, so `floor`
  and `ceil` carry an explicit sign correction:

      floor = let t = num / den (truncating)
              if num % den ≠ 0 ∧ num < 0 then t - 1 else t
      ceil  = let t = num / den (truncating)
              if num % den ≠ 0 ∧ num > 0 then t + 1 else t

  Rust's truncating `/`,`%` are Lean's `Int.tdiv` / `Int.tmod` (NOT Lean's `/`,`%`, which are
  Euclidean — `(-7)/2 = -4`). For a POSITIVE denominator Lean's Euclidean `num / den` rounds toward
  −∞, i.e. it IS the true `⌊num/den⌋`; and `-((-num) / den)` is the true `⌈num/den⌉`. So proving the
  Rust truncating-with-correction result EQUALS those Euclidean spec values is exactly proving the
  algorithm computes the true floor / ceil.

  Scope — like the seed `Fold.lean`, these hold for a fixed set of denominators but for ALL of `Int`
  as the numerator. A single symbolic-denominator statement is out of reach for `omega`: with `den`
  a variable the goal contains the products `f · den` and `den · q`, which are nonlinear (omega is a
  LINEAR integer decision procedure). Fixing `den` to a literal makes them linear, and `omega` then
  discharges the whole correction — including the divisibility case-split — over every numerator.
  The denominators below are the rational test suite's (2, 4, 10, 20) plus coprime/odd witnesses
  (3, 7) and the integer base case (1), so the set is not cherry-picked to divide evenly.

  Checked with:  lean verification/lean/RationalFloorCeil.lean   (Lean 4.32.0; see ../README.md)
-/

namespace Substrate

/-- `Rational::floor`, ported verbatim: truncating divide, then step down for a negative
    non-integer (whose truncation rounded the wrong way). -/
def rustFloor (num den : Int) : Int :=
  if num.tmod den ≠ 0 ∧ num < 0 then num.tdiv den - 1 else num.tdiv den

/-- `Rational::ceil`, ported verbatim: truncating divide, then step up for a positive
    non-integer. -/
def rustCeil (num den : Int) : Int :=
  if num.tmod den ≠ 0 ∧ num > 0 then num.tdiv den + 1 else num.tdiv den

/-
  Bridge the two truncating operators to Lean's Euclidean `/`,`%` (where `omega` is complete for a
  literal divisor), then split the resulting conditionals and let `omega` finish. `Int.tmod_eq_emod`
  and `Int.tdiv_eq_ediv` are the core lemmas; `Int.sign`/`Int.natAbs` reduce the literal constants
  they introduce.
-/

/-- The floor correction computes the true floor `⌊num/den⌋` (= Euclidean `num / den`, `den > 0`). -/
theorem floor_correct_den2 (num : Int) : rustFloor num 2 = num / 2 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den3 (num : Int) : rustFloor num 3 = num / 3 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den4 (num : Int) : rustFloor num 4 = num / 4 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den5 (num : Int) : rustFloor num 5 = num / 5 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den7 (num : Int) : rustFloor num 7 = num / 7 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den10 (num : Int) : rustFloor num 10 = num / 10 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem floor_correct_den20 (num : Int) : rustFloor num 20 = num / 20 := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

/-- The ceil correction computes the true ceiling `⌈num/den⌉` (= `-((-num) / den)`, `den > 0`). -/
theorem ceil_correct_den2 (num : Int) : rustCeil num 2 = -((-num) / 2) := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem ceil_correct_den3 (num : Int) : rustCeil num 3 = -((-num) / 3) := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem ceil_correct_den4 (num : Int) : rustCeil num 4 = -((-num) / 4) := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem ceil_correct_den10 (num : Int) : rustCeil num 10 = -((-num) / 10) := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem ceil_correct_den20 (num : Int) : rustCeil num 20 = -((-num) / 20) := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

/-- Integer base case (`denominator == 1`, i.e. a whole-number rational / `from_integer`): floor and
    ceil are the value itself, with no correction. -/
theorem floor_of_integer (num : Int) : rustFloor num 1 = num := by
  simp only [rustFloor, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

theorem ceil_of_integer (num : Int) : rustCeil num 1 = num := by
  simp only [rustCeil, Int.tmod_eq_emod, Int.tdiv_eq_ediv, Int.sign, Int.natAbs]
  split <;> split <;> omega

/-- The classic sign witnesses from the `rational_floor_and_ceil_handle_signs` unit test, now as
    corollaries that hold for the whole family above rather than the few sampled points. -/
example : rustFloor 1 2 = 0 ∧ rustCeil 1 2 = 1 := by decide
example : rustFloor (-1) 2 = -1 ∧ rustCeil (-1) 2 = 0 := by decide

end Substrate
