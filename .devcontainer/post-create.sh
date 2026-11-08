#!/usr/bin/env bash
set -euo pipefail

if ! command -v clang-format >/dev/null 2>&1; then
	sudo apt-get update
	sudo apt-get install -y clang-format
fi

cmake --version
clang-format --version
cmake -S . -B build/dev -G Ninja -DSHRIMP_BUILD_BENCHMARKS=OFF -DSHRIMP_BUILD_DOCS=OFF
