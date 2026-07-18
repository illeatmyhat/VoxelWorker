# `verification/` — machine-checked construction (ADR 0014 decision 6)

The proof tier of the doctrine in [`docs/architecture/05-proof.md`](../docs/architecture/05-proof.md)
§"Construction, types, and machine-checked proof". Three tools, each matched to what it proves;
the tool assignment per component lives in
[`docs/design/substrate-extraction-map.md`](../docs/design/substrate-extraction-map.md) decision-6.

None of these run in the ordinary `cargo` gate — they are **on-demand proofs run in WSL** (CBMC,
Z3, and Lean have no native Windows story here). The oracle/parity gates in the normal test suite
remain permanent regardless; a theorem verifies the mathematics, the gate verifies the shipping
binary still implements it.

## The three tiers

| Tier | Tool | Lives | Proves |
| --- | --- | --- | --- |
| **BMC** | **Kani** 0.67.0 | inline `#[cfg(kani)] mod kani_proofs` in `crates/{substrate,raycast}` | finite bit/index/arithmetic kernels over their whole *bounded* input space |
| **Deductive** | **Verus** 0.2026.07.12 | `verification/verus/*.rs` | **stateful** invariants via loop/data invariants (unbounded), where BMC's unrolling explodes |
| **Algebraic** | **Lean** 4.32.0 | `verification/lean/*.lean` | theorems over **unbounded/exact** domains (all `Int`, rationals, fold/frame algebra) |

**Why Verus, not Creusot** (the map leaves it open): this box has no passwordless `sudo`, which
makes Creusot's Why3 + opam + SMT-solver platform painful to stand up. Verus ships a prebuilt
release that bundles Z3 and installs entirely under `$HOME` — self-contained, no root.

## Toolchains (WSL Ubuntu, user `kai_yuu`, all under `$HOME`, no root)

- **Kani** — `cargo kani`; see the `kani-wsl-toolchain` memory for the full story.
- **Verus** — prebuilt `verus-0.2026.07.12` unzipped to `$HOME/verus-dist/`; bundles `z3`. Needs
  rust toolchain `1.96.0` (`rustup install 1.96.0-x86_64-unknown-linux-gnu`, already done). The
  `verus` launcher is `$(find $HOME/verus-dist -name verus)`.
- **Lean** — `elan` (Lean toolchain manager) under `$HOME/.elan/`; toolchain `stable` = Lean
  4.32.0 + Lake 5.0.0. Core-only proofs check with `lean file.lean` (no mathlib yet — a Lake
  project with a mathlib dependency gets stood up when the rational-arithmetic proofs need it).

## Running (from Git Bash on Windows)

Run through a **script file**, never an inline `bash -lc '…$VAR…'` — the GitBash→wsl.exe→bash
quoting layers silently blank in-script shell variables (loop vars, `$?`). A script sidesteps it.

```bash
# Verus — verify one file
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc \
  'source $HOME/.cargo/env; V=$(find $HOME/verus-dist -name verus); \
   "$V" /mnt/c/Users/Kai_Yuu/Documents/VoxelWorker/verification/verus/widest_span.rs'

# Lean — check one file (no output + exit 0 == all theorems accepted)
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc \
  'export PATH="$HOME/.elan/bin:$PATH"; \
   lean /mnt/c/Users/Kai_Yuu/Documents/VoxelWorker/verification/lean/Fold.lean'
```

## Running the battery

```bash
./verification/run-all.sh              # all three tiers; exits non-zero if any proof fails
./verification/run-all.sh --quick      # Lean + Verus only — ~8 s, safe per-commit
./verification/run-all.sh --kani-only  # just the BMC tier
```

It DISCOVERS proof files rather than listing them, so a new proof is picked up automatically. Set
`KANI_TARGET_DIR` under WSL (e.g. `$HOME/rc-kani-target`) — building on `/mnt/c` is slow; leave it
unset on a native CI runner, where it buys nothing.

Use it rather than running tiers by hand: `lean/RationalReduce.lean` once sat BROKEN across a commit
because a cosmetic edit was never re-checked, and only a full-battery run caught it.

## Cadence: run the full battery at EPIC boundaries, not per commit

Measured 2026-07-18 (`-j`, warm build):

| tier | time | scope |
| --- | --- | --- |
| Lean | 4 s | 3 files |
| Verus | 4 s | 4 files, 26 obligations |
| Kani `raycast` | 4 s | 5 harnesses |
| **Kani `substrate`** | **467 s** | 16 harnesses |
| **total** | **479 s** | |

**The deductive and algebraic tiers are effectively free** — 8 s for all 30 proofs, so `--quick` can
run per commit without anyone noticing. The entire cost is Kani on `substrate`, and within it a
couple of harnesses dominate (see the cost lesson in `interval/rational.rs`'s `kani_proofs` doc). A
multi-minute pass on every PR gets resented and then disabled, and nightly would burn runner minutes
on days nothing in the proven crates moved. An **epic boundary** is the right trigger for the full
run: it is this repo's unit of work, and it fires exactly when the proven code has actually changed.

The everyday `cargo` gate keeps its own job — a theorem verifies the mathematics, the gate verifies
the shipping binary still implements it. The Kani harnesses are `#[cfg(kani)]` and the Verus/Lean
proofs are models under `verification/`, so **none of them are visible to `cargo test`/`clippy`**;
the unit tests remain the only always-on regression check and must not be deleted in favour of a
proof that only runs on demand.

When wiring a CI job:

- **Kani** — `cargo kani -p substrate -j --output-format=terse`. `-j` verifies harnesses on parallel
  threads and **requires** `--output-format=terse`; it cut the two expensive harnesses from ~21 min
  serial to ~11 min wall-clock. Budget each harness a ceiling and tighten or cut anything that
  blows it — cost scales with data-dependent LOOP CHAINS (each `Rational::new` gcd), not with the
  symbolic bound, so widening a loop-free bound is nearly free while adding a loop is not.
- Toolchain install dominates a cold run and should be **cached**: `cargo kani setup` fetches CBMC
  plus a pinned nightly; Verus is a prebuilt zip needing rust 1.96.0; Lean comes via `elan`.
- `CARGO_TARGET_DIR=$HOME/rc-kani-target` is a **WSL-only** workaround for the slow `/mnt/c` 9p
  mount — irrelevant on a native Linux runner, so do not carry it into CI.

## What is proved here so far

**Verus (deductive) — the seed plus all three decision-6 stateful targets:**

- **`verus/widest_span.rs`** — a model of `DisjointIntervalSet::widest_span`: the max stored width
  over a sequence, discharged from a `while`-loop invariant. Establishes the loop-invariant
  machinery the real stateful targets need.
- **`verus/disjoint_interval_set_insert.rs`** — `DisjointIntervalSet::insert` preserves the
  normalization invariant (non-empty ∧ strict gap between consecutive ⇒ sorted ∧ disjoint ∧
  non-touching) across every path: the three O(1) fast paths and the general skip-left + merge
  splice. THE target Kani couldn't reach (`Vec::splice` exploded BMC); the splice is modelled as an
  explicit prefix ++ [merged] ++ suffix rebuild yielding the identical sequence.
- **`verus/slot_free_list.rs`** — `SlotFreeList` safety: `allocate` never returns a slot still in
  the free set (no double-allocation) and every allocated/free index is `< len` (no out-of-bounds
  `slots[slot]`), from the strictly-increasing-and-in-range free-set invariant; `free` modelled as
  the sorted-unique insert (the faithful model of `sort_unstable + dedup`).
- **`verus/generation_supersede.rs`** — `GenerationTracker`: generations strictly increase,
  acceptance is unique to the newest, a superseded generation is discarded (stale never swaps in over
  fresher state), nothing is accepted before any dispatch — plus a burst tying the theorems to the
  real API.

**Lean (algebraic) — the seed plus the `Rational` floor/ceil + reduction targets, all core-only:**

- **`lean/Fold.lean`** — the floor-division fold bound (`edge·(n/edge) ≤ n < edge·(n/edge)+edge`)
  for **every** `Int` at each pyramid edge {1,8,64,512} — the unbounded form of the Kani
  `fold_lands_in_the_containing_cell` harness, which could only sample bounded coordinates.
- **`lean/RationalFloorCeil.lean`** — the shipping truncating sign-corrected `Rational::floor` /
  `::ceil` equal the true `⌊·⌋` / `⌈·⌉` for **every** integer numerator, at a spread of literal
  denominators (a symbolic denominator is nonlinear, out of `omega`'s reach — same concrete-edge
  scoping as `Fold.lean`). Rust truncation is Lean's `Int.tdiv`/`Int.tmod`, bridged to Euclidean
  `/`,`%` (which IS the true floor for a positive denominator) then `omega`-discharged.
- **`lean/RationalReduce.lean`** — `Rational::new`'s gcd reduction yields canonical form: the Euclid
  loop (proved `= Nat.gcd`) divides both magnitudes exactly, the reduced pair is coprime (⇒ equal
  values have identical representations), and a non-zero denominator stays ≥ 1.

## Next targets (decision-6 tool assignment)

All decision-6 Verus targets, and the `Rational` floor/ceil + reduction on the Lean side, are done.

**`Rational` field laws — RULED OUT, not proved (2026-07-17).** The times/plus assoc/comm/distrib
laws are properties of ℚ (a field by textbook), not of this code. A mathlib proof would be a
*refinement* — our i128 cross-multiply-then-reduce agrees with mathlib's `Rat` — after which the
laws fall out of mathlib's `Field Rat` instance. That re-derives school-book algebra and pulls in a
multi-GB `mathlib` cache to anchor a property nobody doubts. The two things that could actually be
wrong here are already covered elsewhere: **canonicalization** (equal values reduce to identical
structs, so `==` is real value equality) is exactly `lean/RationalReduce.lean`'s coprime-reduction
theorem, which `times`/`plus` inherit by routing through `new`; and **i128 overflow** is a BMC-shaped
concern a field-law proof over exact `Rat` would not catch anyway, and is a documented accepted
deviation in the source. So `mathlib` stays unwired.

Remaining Lean targets that *would* still justify wiring `mathlib` (a Lake project +
`lake exe cache get`) when tackled: the voxel-frame algebra (ADR 0008) compose/invert laws;
`SparseMinMipPyramid`'s conservative-superset theorem; `FieldInterval` conservatism (Duff 1992).
