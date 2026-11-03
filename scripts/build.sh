#!/usr/bin/env bash
# Usage:
#   build.sh [--preset debug|release|ci] [--test] [--docs]
#            [--benchmark [BENCHMARK_ARGS...]]
#            [--profile perf|callgrind [BENCHMARK_ARGS...]]
set -euo pipefail

preset="debug"
run_tests="false"
build_docs="false"
action=""          # benchmark | profile
profile_tool="perf"
extra_args=()
toolchain_args=()

if [[ -n "${VCPKG_ROOT:-}" ]]; then
    toolchain_args=(-DCMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)    preset="$2"; shift 2 ;;
        --test)      run_tests="true"; shift ;;
        --docs)      build_docs="true"; shift ;;
        --benchmark) action="benchmark"; shift; extra_args=("$@"); break ;;
        --profile)   action="profile"; profile_tool="${2:-perf}"; shift 2; extra_args=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# benchmark / profile path
if [[ "$action" == "benchmark" || "$action" == "profile" ]]; then
    if [[ "$action" == "profile" ]]; then
        build_dir="build/profile"
        build_type="RelWithDebInfo"
    else
        build_dir="build/benchmarks"
        build_type="Release"
    fi

    cmake -S . -B "$build_dir" -G Ninja \
        "${toolchain_args[@]}" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_TESTS=OFF \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS=ON \
        -DCPP_PACKAGE_BOILERPLATE_BUILD_DOCS=OFF

    cmake --build "$build_dir" --parallel --target cpp_package_boilerplate_benchmarks

    bin="$build_dir/benchmarks/cpp_package_boilerplate_benchmarks"

    if [[ "$action" == "benchmark" ]]; then
        "$bin" "${extra_args[@]+"${extra_args[@]}"}"
    else
        case "$profile_tool" in
            perf)       perf record --call-graph dwarf "$bin" "${extra_args[@]+"${extra_args[@]}"}" ;;
            callgrind)  valgrind --tool=callgrind "$bin" "${extra_args[@]+"${extra_args[@]}"}" ;;
            *)          echo "unsupported profiling tool: $profile_tool (supported: perf, callgrind)" >&2; exit 1 ;;
        esac
    fi
    exit 0
fi

# standard build path
build_dir="build/${preset}"

case "$preset" in
    release) build_type="Release" ;;
    ci)      build_type="RelWithDebInfo" ;;
    *)       build_type="Debug" ;;
esac

cmake -S . -B "$build_dir" -G Ninja \
    "${toolchain_args[@]}" \
    -DCMAKE_BUILD_TYPE="$build_type" \
    -DCPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS=OFF \
    -DCPP_PACKAGE_BOILERPLATE_BUILD_DOCS="$build_docs"

cmake --build "$build_dir" --parallel

if [[ "$run_tests" == "true" ]]; then
    ctest --test-dir "$build_dir" --output-on-failure
fi
