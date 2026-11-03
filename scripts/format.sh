#!/usr/bin/env bash
set -euo pipefail

mapfile -t files < <(find include src tests benchmarks -type f \( -name '*.hpp' -o -name '*.h' -o -name '*.cpp' \) | sort)

if [[ ${#files[@]} -eq 0 ]]; then
    exit 0
fi

case "${1:-}" in
    --check)
        clang-format --dry-run --Werror "${files[@]}"
        ;;
    --diff)
        clang-format -n --Werror "${files[@]}"
        ;;
    "")
        clang-format -i "${files[@]}"
        ;;
    *)
        echo "unknown argument: ${1}" >&2
        exit 1
        ;;
esac
