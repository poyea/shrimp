if(
    SHRIMP_USE_PACKAGE_MANAGER_DEPS
    AND (
        DEFINED VCPKG_TARGET_TRIPLET
        OR CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg"
    ))
    find_package(GTest CONFIG QUIET)
endif()

if(NOT TARGET GTest::gtest_main)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.16.0.tar.gz)

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(${PROJECT_NAME}_tests test_mylib.cpp)
target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${PROJECT_NAME}::${PROJECT_NAME} GTest::gtest_main)
shrimp_apply_target_options(${PROJECT_NAME}_tests)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)

# static_assert suites: the build is the test run.
set(shrimp_constexpr_suites
    constexpr_checks
    induction_checks
    training_checks
    linalg_grad_checks
    norm_grad_checks
    act_grad_checks
    rope_table_checks
    ffn_backward_checks
    attn_backward_checks
    backward_consistency_checks
    backward_checks
    train_checks
    gpt2_checks)

foreach(suite IN LISTS shrimp_constexpr_suites)
    add_executable(${PROJECT_NAME}_${suite} ${suite}.cpp)
    target_include_directories(
        ${PROJECT_NAME}_${suite} PRIVATE ${PROJECT_SOURCE_DIR}/include)

    # Defaults are too small for a forward pass.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${PROJECT_NAME}_${suite}
            PRIVATE -fconstexpr-steps=100000000 -fconstexpr-depth=4096)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${PROJECT_NAME}_${suite}
            PRIVATE
                -fconstexpr-ops-limit=2000000000
                -fconstexpr-depth=4096
                -fconstexpr-loop-limit=100000000)
    elseif(MSVC)
        target_compile_options(
            ${PROJECT_NAME}_${suite}
            PRIVATE /constexpr:depth4096 /constexpr:steps100000000)
    endif()

    add_test(NAME ${suite} COMMAND ${PROJECT_NAME}_${suite})
endforeach()
