#!/usr/bin/env bash
#
# Run the whole three-tier proof battery (see ./README.md): Lean (algebraic) → Verus (deductive) →
# Kani (bounded model checking). Prints a per-file/per-crate timing table and exits NON-ZERO if any
# tier fails, so CI can call it directly.
#
# Why this exists: the tiers were previously run from ad-hoc one-off commands, and a proof file
# (`lean/RationalReduce.lean`) sat BROKEN across a commit because a cosmetic edit was never
# re-checked. One command that runs everything is the fix. It DISCOVERS proof files rather than
# listing them, so a newly added proof is picked up automatically instead of being silently skipped.
#
#   ./verification/run-all.sh              # everything (Kani dominates; see --quick)
#   ./verification/run-all.sh --quick      # Lean + Verus only — seconds, fine per-commit
#   ./verification/run-all.sh --kani-only  # just the BMC tier
#
# Cadence: the full battery belongs at EPIC boundaries, not per commit — Lean+Verus are ~5 s
# combined, but the Kani tier is minutes and dominated by a couple of harnesses. `--quick` is the
# per-commit-safe subset.
#
# Environment:
#   VERUS               path to the verus launcher (default: found under $HOME/verus-dist)
#   KANI_TARGET_DIR     CARGO_TARGET_DIR for the Kani tier. Set this under WSL to a Linux-FS path
#                       (e.g. $HOME/rc-kani-target) — building on /mnt/c is slow. Leave UNSET on a
#                       native Linux CI runner, where it buys nothing.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFICATION_DIR="$REPO_ROOT/verification"

# Crates carrying `#[cfg(kani)]` harnesses. Both are wgpu-free, hence CBMC-analyzable.
KANI_CRATES=(substrate raycast)

run_lean=1
run_verus=1
run_kani=1
case "${1:-}" in
    --quick | --no-kani) run_kani=0 ;;
    --kani-only) run_lean=0; run_verus=0 ;;
    "") ;;
    *) echo "usage: $0 [--quick|--no-kani|--kani-only]" >&2; exit 2 ;;
esac

failures=0
summary=()

# Record one result row and count failures. $1=label $2=verdict $3=seconds $4=detail
record() {
    summary+=("$(printf '  %-46s %-8s %5ss  %s' "$1" "$2" "$3" "${4:-}")")
    [ "$2" = "FAIL" ] && failures=$((failures + 1))
    return 0
}

if [ "$run_lean" = 1 ]; then
    echo "===== TIER 1: Lean (algebraic — unbounded/exact domains) ====="
    export PATH="$HOME/.elan/bin:$PATH"
    for proof in "$VERIFICATION_DIR"/lean/*.lean; do
        [ -e "$proof" ] || continue
        name="$(basename "$proof")"
        start=$SECONDS
        # No output and exit 0 == every theorem accepted.
        if output="$(lean "$proof" 2>&1)"; then
            record "lean/$name" "OK" "$((SECONDS - start))"
        else
            record "lean/$name" "FAIL" "$((SECONDS - start))" "$(echo "$output" | head -1)"
        fi
    done
fi

if [ "$run_verus" = 1 ]; then
    echo "===== TIER 2: Verus (deductive — stateful invariants) ====="
    # shellcheck disable=SC1091
    [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
    verus_bin="${VERUS:-$(find "$HOME/verus-dist" -name verus -type f 2>/dev/null | head -1)}"
    if [ -z "$verus_bin" ]; then
        record "verus (toolchain)" "FAIL" "0" "no verus launcher; set \$VERUS"
    else
        for proof in "$VERIFICATION_DIR"/verus/*.rs; do
            [ -e "$proof" ] || continue
            name="$(basename "$proof")"
            start=$SECONDS
            output="$("$verus_bin" "$proof" 2>&1)"
            status=$?
            detail="$(echo "$output" | grep -E 'verification results' | head -1)"
            if [ $status -eq 0 ]; then
                record "verus/$name" "OK" "$((SECONDS - start))" "$detail"
            else
                record "verus/$name" "FAIL" "$((SECONDS - start))" "${detail:-$(echo "$output" | head -1)}"
            fi
        done
    fi
fi

if [ "$run_kani" = 1 ]; then
    echo "===== TIER 3: Kani (BMC — machine-integer/index kernels) ====="
    echo "  (minutes: a couple of harnesses dominate — see README 'Cadence')"
    [ -n "${KANI_TARGET_DIR:-}" ] && export CARGO_TARGET_DIR="$KANI_TARGET_DIR"
    cd "$REPO_ROOT" || exit 1
    for crate in "${KANI_CRATES[@]}"; do
        start=$SECONDS
        # -j verifies harnesses on parallel threads and REQUIRES terse output.
        output="$(cargo kani -p "$crate" -j --output-format=terse 2>&1)"
        status=$?
        detail="$(echo "$output" | grep -E 'Complete -' | tail -1)"
        if [ $status -eq 0 ]; then
            record "kani/$crate" "OK" "$((SECONDS - start))" "$detail"
        else
            record "kani/$crate" "FAIL" "$((SECONDS - start))" \
                "${detail:-$(echo "$output" | grep -E 'Failed Checks' | head -1)}"
        fi
    done
fi

echo
echo "===== PROOF BATTERY SUMMARY ====="
printf '%s\n' "${summary[@]}"
echo
if [ "$failures" -eq 0 ]; then
    echo "ALL PROOFS PASSED"
    exit 0
fi
echo "$failures FAILED"
exit 1
