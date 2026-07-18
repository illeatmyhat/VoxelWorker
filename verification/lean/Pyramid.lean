/-
  Lean 4 (core only, no mathlib) — the `SparseMinMipPyramid` CONSERVATIVE-SUPERSET theorem
  (substrate-extraction-map decision 6), over unbounded integers and unbounded key sets.

  What the Kani harnesses in `crates/substrate/src/spatial/min_mip_pyramid.rs` already pin:
  the fold lands a BOUNDED coordinate in the cell containing it, and `binary_search` membership
  agrees with a linear scan on a 5-element set. What they cannot reach, and this file proves:

    1. the fold NESTS across levels — folding twice equals folding once at the product edge,
    2. so cells nest: coordinates sharing a fine cell share the coarse cell,
    3. hence COARSE ABSENCE implies FINE ABSENCE — the theorem that makes a hierarchical
       traverser's "skip the whole coarse cell in one stride" sound, over ALL of `Int`,
    4. and the level is a conservative superset of the key set with NO stray cells, for a key
       list of ANY length (Kani is bounded to 5).

  (3) is the one a bug would actually hide in: every other property can be spot-checked by a
  differential render, but a traverser that wrongly skips a coarse cell drops occupied space that
  no sampling reliably catches.

  Checked with:  lean verification/lean/Pyramid.lean   (elan-managed Lean 4.32.0; see ../README.md)

  SCOPE, stated honestly:
  * Lean's `/` on `Int` is Euclidean division, which for a POSITIVE divisor is floor division —
    exactly Rust's `i64::div_euclid`, the operation `fold_coordinate_to_cell` applies per axis.
  * Proved at the pyramid's actual literal edges (8, 64, 512). A symbolic edge makes the
    statements nonlinear (`n / (a*b)`) and out of `omega`'s reach — the same concrete-edge
    scoping `Fold.lean` and the Kani fold helper already use, and for the same reason.
  * One axis. The real fold is componentwise over three independent axes and the packing
    `pack_lattice_key` is a bijection, so the 3-D statement is this one applied three times;
    nothing about the axes interacts.
-/

namespace Substrate

/-! ## 1. The fold nests: fold twice == fold once at the product edge

    This is the algebraic fact the whole hierarchy rests on. `(n / 8) / 8 = n / 64` is NOT
    generic integer-division folklore — it holds for floor division and fails for the
    truncating division of many languages at negative `n` (e.g. truncating: `(-1/8)/8 = 0`
    but `-1/64 = 0` agrees, while `(-9/8)/8 = -1/8 = 0` and `-9/64 = 0` — the divergence
    appears once the intermediate quotient itself rounds the wrong way). Lean's `/` on `Int`
    is Euclidean, matching `div_euclid`, so it holds here. -/

theorem fold_nests_8_into_64 (n : Int) : (n / 8) / 8 = n / 64 := by omega

theorem fold_nests_64_into_512 (n : Int) : (n / 64) / 8 = n / 512 := by omega

theorem fold_nests_8_into_512 (n : Int) : (n / 8) / 64 = n / 512 := by omega

/-! ## 2. Cells nest: sharing a fine cell implies sharing the coarse cell

    `Fold.lean`'s `same_quotient_same_cell` states this WITHIN one level, where it is the
    tautology `h : a / 8 = b / 8 ⊢ a / 8 = b / 8`. The content is in the CROSS-LEVEL form
    below, which actually needs the nesting lemma. -/

theorem same_cell_8_implies_same_cell_64 (a b : Int) (h : a / 8 = b / 8) :
    a / 64 = b / 64 := by
  rw [← fold_nests_8_into_64, ← fold_nests_8_into_64, h]

theorem same_cell_64_implies_same_cell_512 (a b : Int) (h : a / 64 = b / 64) :
    a / 512 = b / 512 := by
  rw [← fold_nests_64_into_512, ← fold_nests_64_into_512, h]

/-! ## 3. THE SKIP THEOREM: coarse absence implies fine absence

    A hierarchical traverser that finds the coarse cell of its position unoccupied leaps the
    whole cell. That is sound exactly when no occupied key hiding at a finer level could lie
    inside it — i.e. when differing at the coarse level forces differing at the fine level.
    The contrapositive of §2, and the reason a wrongly-nested fold would silently drop
    geometry rather than merely render it slowly. -/

theorem different_cell_64_implies_different_cell_8 (a b : Int) (h : a / 64 ≠ b / 64) :
    a / 8 ≠ b / 8 :=
  fun fine => h (same_cell_8_implies_same_cell_64 a b fine)

theorem different_cell_512_implies_different_cell_64 (a b : Int) (h : a / 512 ≠ b / 512) :
    a / 64 ≠ b / 64 :=
  fun coarse => h (same_cell_64_implies_same_cell_512 a b coarse)

/-- Transitively across the whole pyramid: absent at 512 ⇒ absent at 8. A traverser may leap
    the coarsest empty cell without inspecting any finer level. -/
theorem different_cell_512_implies_different_cell_8 (a b : Int) (h : a / 512 ≠ b / 512) :
    a / 8 ≠ b / 8 :=
  different_cell_64_implies_different_cell_8 a b
    (different_cell_512_implies_different_cell_64 a b h)

/-! ## 4. The level is a conservative superset with no stray cells

    `MinMipLevel::from_keys` maps the fold over the keys, then sorts and deduplicates. Sorting
    is irrelevant to membership (it is there for the binary search, whose correctness is the
    Kani harness `binary_search_membership_agrees_with_linear_scan`), so the set-level content
    is the map and the dedup. `List.dedup` is not in Lean core, so the dedup is modelled
    explicitly below and its membership behaviour proved rather than assumed. -/

/-- Remove duplicates, keeping the first occurrence — the model of `sort_unstable + dedup`
    at the level of SET membership (which is all the superset property depends on). -/
def dedupe : List Int → List Int
  | [] => []
  | x :: rest => x :: (dedupe rest).filter (fun y => y ≠ x)

/-- Deduplication changes no membership question: exactly the original elements survive. -/
theorem mem_dedupe (a : Int) : ∀ l : List Int, a ∈ dedupe l ↔ a ∈ l
  | [] => by simp [dedupe]
  | x :: rest => by
    by_cases hax : a = x
    · subst hax; simp [dedupe]
    · simp [dedupe, List.mem_filter, mem_dedupe a rest, hax]

/-- The occupancy level at `edge`: every key's containing cell, deduplicated. -/
def levelCells (edge : Int) (keys : List Int) : List Int :=
  dedupe (keys.map (fun k => k / edge))

/-- **CONSERVATIVE SUPERSET (soundness).** Every occupied key's cell is present in the level,
    for a key list of ANY length. This is the direction that matters: a consumer may skip an
    absent cell precisely because no occupied key can fold into one. -/
theorem occupied_key_cell_is_present (edge : Int) (keys : List Int) (k : Int) (h : k ∈ keys) :
    k / edge ∈ levelCells edge keys :=
  (mem_dedupe _ _).mpr (List.mem_map_of_mem h)

/-- **NO STRAY CELLS (exactness).** The level reports no cell that some key does not fold to,
    so the superset is tight — the traverser is never sent to descend into empty space. -/
theorem present_cell_has_an_occupied_key (edge : Int) (keys : List Int) (c : Int)
    (h : c ∈ levelCells edge keys) : ∃ k ∈ keys, k / edge = c := by
  have hmapped : c ∈ keys.map (fun k => k / edge) := (mem_dedupe _ _).mp h
  exact List.mem_map.mp hmapped

/-- The empty key set yields an empty level — no cell is reported occupied at any edge.
    (`MinMipLevel::empty(e) == from_keys(&[], e)`, the `empty_set_reports_nothing_occupied` test,
    here for every edge rather than the three the test samples.) -/
theorem empty_keys_give_empty_level (edge : Int) : levelCells edge [] = [] := by
  simp [levelCells, dedupe]

/-! ## 5. Tying §3 to the level: coarse absence really does justify the skip

    The traverser's actual question is not about two coordinates but about a level: if the
    coarse level does not contain the coarse cell of `p`, then NO key in the set shares even
    the fine cell of `p`. Combines §3 with §4's exactness. -/

theorem absent_coarse_cell_means_no_key_in_any_finer_cell
    (keys : List Int) (p : Int) (h : p / 512 ∉ levelCells 512 keys) :
    ∀ k ∈ keys, k / 8 ≠ p / 8 := by
  intro k hk
  -- If any key shared the fine cell it would share the coarse cell, putting that coarse cell
  -- in the level (§4 soundness) and contradicting its absence.
  have hcoarse : k / 512 ≠ p / 512 := fun heq => h (heq ▸ occupied_key_cell_is_present 512 keys k hk)
  exact different_cell_512_implies_different_cell_8 k p hcoarse

end Substrate
