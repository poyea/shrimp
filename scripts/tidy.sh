#!/usr/bin/env bash
set -euo pipefail

preset="${1:-debug}"
build_dir="build/${preset}"

cmake --preset "${preset}" -DSHRIMP_ENABLE_CLANG_TIDY=ON

cmake --build "${build_dir}" --parallel
