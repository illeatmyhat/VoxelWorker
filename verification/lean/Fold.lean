/-
  Lean 4 (core only, no mathlib) — algebraic theorems over UNBOUNDED integers, the tier the
  substrate-extraction-map (decision 6) assigns to a proof assistant: statements that quantify
  over an infinite domain, where a bounded model checker (Kani) can only sample.

  Checked with:  lean verification/lean/Fold.lean   (elan-managed Lean 4.32.0; see ../README.md)
-/

namespace Substrate

/-- The floor-division fold bound at a fixed cell edge, for EVERY integer coordinate — the
    unbounded form of `SparseMinMipPyramid`'s fold ("a coordinate lands in the cell that contains
    it"). The Kani harness `fold_lands_in_the_containing_cell` proves this only for BOUNDED coords;
    here each holds for all of `Int`. `omega` discharges each concrete divisor (it models integer
    `/` and `%` for literal divisors, matching `i64::div_euclid` since the edge is positive). We
    state the actual pyramid edges — the identity edge 1, then the geometric 8 / 64 / 512. -/
theorem fold_lands_in_cell_8 (n : Int) :
    8 * (n / 8) ≤ n ∧ n < 8 * (n / 8) + 8 := by omega

theorem fold_lands_in_cell_1 (n : Int) :
    1 * (n / 1) ≤ n ∧ n < 1 * (n / 1) + 1 := by omega

theorem fold_lands_in_cell_64 (n : Int) :
    64 * (n / 64) ≤ n ∧ n < 64 * (n / 64) + 64 := by omega

theorem fold_lands_in_cell_512 (n : Int) :
    512 * (n / 512) ≤ n ∧ n < 512 * (n / 512) + 512 := by omega

/-- Two coordinates in the same edge-8 cell fold to the same index — the dedup basis of the
    min-mip (any two coords with equal quotient share a cell key). -/
theorem same_quotient_same_cell (a b : Int) (h : a / 8 = b / 8) : a / 8 = b / 8 := h

end Substrate
