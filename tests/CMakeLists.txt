if(
    CPP_PACKAGE_BOILERPLATE_USE_PACKAGE_MANAGER_DEPS
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
cpp_package_boilerplate_apply_target_options(${PROJECT_NAME}_tests)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)
