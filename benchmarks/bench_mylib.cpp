#include "cpp_package_boilerplate/package.hpp"

#include <benchmark/benchmark.h>

static void BM_Add(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(cpp_package_boilerplate::add(20, 22));
    }
}

BENCHMARK(BM_Add);
BENCHMARK_MAIN();
