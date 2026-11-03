# cpp-package-boilerplate [![ci](https://github.com/poyea/cpp-package-boilerplate/actions/workflows/ci.yml/badge.svg)](https://github.com/poyea/cpp-package-boilerplate/actions/workflows/ci.yml)

Modern C++ package boilerplate with CMake, tests, benchmarks, docs, devcontainers, Docker, and GitHub Actions.

## What this template supports

- C++26 by default, with compiler checks for recent toolchains.
- Compiled library, header-only library, CLI, or mixed library plus CLI layouts.
- GoogleTest-based unit tests and Google Benchmark integration.
- Doxygen documentation generation.
- Dev container and Codespaces setup with a repository-owned toolchain.
- Docker build support for a runnable runtime image.
- Clang-format, strict warnings, sanitizers, and release LTO.

## Prerequisites

**Required:**

| Tool | Minimum version | Purpose |
|---|---|---|
| CMake | 4.3 | Build system |
| GCC | 16.1 | C++ compiler (Linux/Windows) |
| Clang | 18 | C++ compiler (alternative) |
| AppleClang | 16 | C++ compiler (macOS) |
| MSVC | 19.40 (VS 2022 17.10) | C++ compiler (Windows) |
| Ninja | 1.11 | Build tool |
| Git | 2 | Source control |

**Feature-specific tools:**

| Tool | Purpose |
|---|---|
| clang-format | Code formatting (`scripts/format.sh`) |
| clang-tidy | Static analysis (`scripts/tidy.sh`) |
| vcpkg | C/C++ dependency management for tests and benchmarks |
| Doxygen + Graphviz | API documentation (`BUILD_DOCS=ON`) |
| Valgrind | Profiling (`scripts/build.sh --profile callgrind`) |
| perf | Profiling (`scripts/build.sh --profile perf`, Linux only) |
| Docker | Runtime container image |

**Install on Ubuntu/Debian:**

```bash
bash scripts/install_tools.sh
```

After installation, export `VCPKG_ROOT` if needed:

```bash
export VCPKG_ROOT="${HOME}/.local/vcpkg"
```

**Dev container / Codespaces:** all tools are pre-installed — no manual setup needed.

## Quick start

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Use vcpkg manifest mode:

```bash
cmake --preset debug-vcpkg
cmake --build --preset debug-vcpkg
ctest --preset debug-vcpkg
```

Or use the helper script:

`bash scripts/build.sh --preset debug --test`

## Project options

Key CMake options:

- `CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY=ON|OFF`
- `CPP_PACKAGE_BOILERPLATE_BUILD_CLI=ON|OFF`
- `CPP_PACKAGE_BOILERPLATE_HEADER_ONLY=ON|OFF`
- `CPP_PACKAGE_BOILERPLATE_BUILD_TESTS=ON|OFF`
- `CPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS=ON|OFF`
- `CPP_PACKAGE_BOILERPLATE_BUILD_DOCS=ON|OFF`

Examples:

```bash
cmake -S . -B build/header-only -G Ninja -DCPP_PACKAGE_BOILERPLATE_HEADER_ONLY=ON
cmake -S . -B build/library-only -G Ninja -DCPP_PACKAGE_BOILERPLATE_BUILD_CLI=OFF
cmake -S . -B build/full -G Ninja -DCPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS=ON -DCPP_PACKAGE_BOILERPLATE_BUILD_DOCS=ON
```

## Repository layout

- `cmake/`: shared compiler and toolchain settings
- `include/`: public headers
- `src/`: compiled library and CLI sources
- `tests/`: unit tests
- `benchmarks/`: performance benchmarks
- `docs/`: Doxygen configuration
- `.devcontainer/`: development container for VS Code and Codespaces
- `.github/workflows/`: CI, docs, and Docker automation
- `scripts/`: local developer helpers

## Formatting

```bash
bash scripts/format.sh
bash scripts/format.sh --check
cmake --build --preset debug --target format
cmake --build --preset debug --target format-check
```

## Static analysis

```bash
bash scripts/tidy.sh
cmake --preset debug-vcpkg -DCPP_PACKAGE_BOILERPLATE_ENABLE_CLANG_TIDY=ON
cmake --build --preset debug-vcpkg
```

## Benchmarks and profiling

```bash
bash scripts/build.sh --benchmark
bash scripts/build.sh --benchmark --benchmark_filter=BM_Add
bash scripts/build.sh --profile perf
bash scripts/build.sh --profile callgrind
```

## Documentation

```bash
cmake -S . -B build/docs -G Ninja -DCPP_PACKAGE_BOILERPLATE_BUILD_TESTS=OFF -DCPP_PACKAGE_BOILERPLATE_BUILD_DOCS=ON
cmake --build build/docs --target docs
```

The GitHub Pages workflow publishes generated Doxygen output from the default branch.

## Container workflows

- Dev container: open the repository in a compatible devcontainer client or GitHub Codespaces.
- Runtime image: `docker build -t cpp-package-boilerplate:latest .`

## License

MIT
