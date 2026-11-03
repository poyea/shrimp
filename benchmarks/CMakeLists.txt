if(
    CPP_PACKAGE_BOILERPLATE_USE_PACKAGE_MANAGER_DEPS
    AND (
        DEFINED VCPKG_TARGET_TRIPLET
        OR CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg"
    ))
    find_package(benchmark CONFIG QUIET)
endif()

if(NOT TARGET benchmark::benchmark)
    include(FetchContent)

    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz)

    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(${PROJECT_NAME}_benchmarks bench_mylib.cpp)
target_link_libraries(${PROJECT_NAME}_benchmarks PRIVATE ${PROJECT_NAME}::${PROJECT_NAME} benchmark::benchmark)
cpp_package_boilerplate_apply_target_options(${PROJECT_NAME}_benchmarks)
