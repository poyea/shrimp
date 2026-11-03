#!/usr/bin/env bash
set -euo pipefail

preset="${1:-debug}"
build_dir="build/${preset}"

cmake --preset "${preset}" -DCPP_PACKAGE_BOILERPLATE_ENABLE_CLANG_TIDY=ON

cmake --build "${build_dir}" --parallel
