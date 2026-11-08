#!/usr/bin/env bash
#
# Compiling the suites runs them. Usage: constexpr_check.sh [compiler]

set -euo pipefail

CXX="${1:-${CXX:-g++}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/shrimp_constexpr_checks"

# Default budgets are too small for a forward pass.
case "$("${CXX}" --version 2>&1 | head -1)" in
    *clang*)
        BUDGET=(-fconstexpr-steps=100000000 -fconstexpr-depth=4096)
        ;;
    *)
        BUDGET=(-fconstexpr-ops-limit=2000000000 -fconstexpr-depth=4096
                -fconstexpr-loop-limit=100000000)
        ;;
esac

echo "compiler : $("${CXX}" --version 2>&1 | head -1)"
echo "budget   : ${BUDGET[*]}"
echo

total_start=$(date +%s)
for suite in constexpr_checks induction_checks gpt2_checks; do
    printf '%-18s ' "${suite}"
    start=$(date +%s)
    "${CXX}" -std=c++26 -Wall -Wextra -Wpedantic \
        -I "${ROOT}/include" \
        "${BUDGET[@]}" \
        "${ROOT}/tests/${suite}.cpp" \
        -o "${OUT}_${suite}"
    echo "ok ($(($(date +%s) - start))s)"
done
total_end=$(date +%s)

echo
echo "all compile-time checks passed in $((total_end - total_start))s"
echo "binaries do nothing when run; the work happened during translation"
