cmake_minimum_required(VERSION 4.3)

project(
    cpp_package_boilerplate
    VERSION 0.1.4
    DESCRIPTION "Modern C++ package boilerplate"
    LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/std:c++latest)
    set(CMAKE_CXX_STANDARD 23)
endif()

option(CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY "Build the library target" ON)
option(CPP_PACKAGE_BOILERPLATE_BUILD_CLI "Build the CLI target" ON)
option(CPP_PACKAGE_BOILERPLATE_HEADER_ONLY "Build the library as a header-only target" OFF)
option(CPP_PACKAGE_BOILERPLATE_BUILD_TESTS "Build unit tests" ${BUILD_TESTING})
option(CPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(CPP_PACKAGE_BOILERPLATE_BUILD_DOCS "Build API documentation" OFF)
option(CPP_PACKAGE_BOILERPLATE_ENABLE_IPO "Enable interprocedural optimization for release builds" ON)
option(CPP_PACKAGE_BOILERPLATE_ENABLE_SANITIZERS "Enable AddressSanitizer and UBSan in debug builds" ON)
option(CPP_PACKAGE_BOILERPLATE_ENABLE_CLANG_TIDY "Run clang-tidy during compilation" OFF)
option(CPP_PACKAGE_BOILERPLATE_USE_PACKAGE_MANAGER_DEPS "Prefer package manager dependencies over FetchContent" ON)

if(NOT CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY AND NOT CPP_PACKAGE_BOILERPLATE_BUILD_CLI)
    message(FATAL_ERROR "Enable at least one target: library or CLI.")
endif()

if(CPP_PACKAGE_BOILERPLATE_HEADER_ONLY AND NOT CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY)
    message(FATAL_ERROR "CPP_PACKAGE_BOILERPLATE_HEADER_ONLY requires CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY=ON.")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16.1)
    message(FATAL_ERROR "GCC 16.1 or newer is required for the default C++26 configuration.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
    message(FATAL_ERROR "Clang 18 or newer is required for the default C++26 configuration.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
    message(FATAL_ERROR "AppleClang 16 or newer is required for the default C++26 configuration.")
elseif(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.40)
    message(FATAL_ERROR "MSVC 19.40 (VS 2022 17.10) or newer is required for the default C++26 configuration.")
endif()

include(cmake/CompilerOptions.cmake)

if(CPP_PACKAGE_BOILERPLATE_ENABLE_CLANG_TIDY)
    find_program(
        CPP_PACKAGE_BOILERPLATE_CLANG_TIDY_EXE
        NAMES clang-tidy clang-tidy-22 clang-tidy-21 clang-tidy-20 clang-tidy-19 clang-tidy-18 clang-tidy-17
        REQUIRED)
    set(
        CMAKE_CXX_CLANG_TIDY
        "${CPP_PACKAGE_BOILERPLATE_CLANG_TIDY_EXE};--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy"
        CACHE STRING
        "clang-tidy command"
        FORCE)
    set(CMAKE_CXX_CLANG_TIDY_EXPORT_FIXES_DIR "${CMAKE_BINARY_DIR}/__clang-tidy-fixes")
    message(STATUS "clang-tidy enabled: ${CPP_PACKAGE_BOILERPLATE_CLANG_TIDY_EXE}")
endif()

find_program(CPP_PACKAGE_BOILERPLATE_CLANG_FORMAT_EXE NAMES clang-format)
if(CPP_PACKAGE_BOILERPLATE_CLANG_FORMAT_EXE)
    file(
        GLOB_RECURSE CPP_PACKAGE_BOILERPLATE_FORMAT_FILES
        CONFIGURE_DEPENDS
        ${PROJECT_SOURCE_DIR}/include/*.h
        ${PROJECT_SOURCE_DIR}/include/*.hpp
        ${PROJECT_SOURCE_DIR}/src/*.h
        ${PROJECT_SOURCE_DIR}/src/*.hpp
        ${PROJECT_SOURCE_DIR}/src/*.cpp
        ${PROJECT_SOURCE_DIR}/tests/*.h
        ${PROJECT_SOURCE_DIR}/tests/*.hpp
        ${PROJECT_SOURCE_DIR}/tests/*.cpp
        ${PROJECT_SOURCE_DIR}/benchmarks/*.h
        ${PROJECT_SOURCE_DIR}/benchmarks/*.hpp
        ${PROJECT_SOURCE_DIR}/benchmarks/*.cpp)

    if(CPP_PACKAGE_BOILERPLATE_FORMAT_FILES)
        add_custom_target(
            format
            COMMAND ${CPP_PACKAGE_BOILERPLATE_CLANG_FORMAT_EXE} -i ${CPP_PACKAGE_BOILERPLATE_FORMAT_FILES}
            COMMENT "Formatting C++ sources with clang-format")
        add_custom_target(
            format-check
            COMMAND ${CPP_PACKAGE_BOILERPLATE_CLANG_FORMAT_EXE} --dry-run --Werror ${CPP_PACKAGE_BOILERPLATE_FORMAT_FILES}
            COMMENT "Checking C++ formatting with clang-format")
    endif()
endif()

add_subdirectory(src)

if(CPP_PACKAGE_BOILERPLATE_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CPP_PACKAGE_BOILERPLATE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CPP_PACKAGE_BOILERPLATE_BUILD_DOCS)
    add_subdirectory(docs)
endif()
