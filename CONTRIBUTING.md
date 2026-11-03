# Contributing

## Local workflow

1. Configure with `cmake --preset debug`.
2. Build with `cmake --build --preset debug`.
3. Run tests with `ctest --preset debug`.
4. Format with `bash scripts/format.sh`.
5. Run static analysis with `bash scripts/tidy.sh`.
6. For dependency-manager validation use `cmake --preset debug-vcpkg` then `cmake --build --preset debug-vcpkg`.
7. Run benchmarks with `bash scripts/build.sh --benchmark` when performance-sensitive code changes.

## Expectations

- Keep the codebase warning-free.
- Add or update tests for functional changes.
- Run formatting before opening a pull request.
- Run clang-tidy for non-trivial C++ changes.
- Keep public headers documented and stable.
- Use `bash scripts/build.sh --profile perf` or `bash scripts/build.sh --profile callgrind` when investigating regressions.
