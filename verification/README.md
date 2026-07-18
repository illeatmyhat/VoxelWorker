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

## What is proved here so far

- **`verus/widest_span.rs`** — a model of `DisjointIntervalSet::widest_span`: the max stored width
  over a sequence, discharged from a `while`-loop invariant. This establishes the loop-invariant
  machinery the real stateful targets need.
- **`lean/Fold.lean`** — the floor-division fold bound (`edge·(n/edge) ≤ n < edge·(n/edge)+edge`)
  for **every** `Int` at each pyramid edge {1,8,64,512} — the unbounded form of the Kani
  `fold_lands_in_the_containing_cell` harness, which could only sample bounded coordinates.

## Next targets (decision-6 tool assignment)

- **Verus (deductive):** `DisjointRunList`/`DisjointIntervalSet` insert keeps sorted ∧ disjoint ∧
  non-touching (the invariant Kani couldn't reach — `Vec::splice` exploded BMC); `SlotFreeList`
  no-double-allocation; generation-supersede newest-wins.
- **Lean (algebraic):** `Rational` field laws + `floor`/`ceil` (needs mathlib); the voxel-frame
  algebra (ADR 0008) compose/invert laws; `SparseMinMipPyramid`'s conservative-superset theorem.
