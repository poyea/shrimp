set(public_headers
    ${PROJECT_SOURCE_DIR}/include/cpp_package_boilerplate/package.hpp)

if(CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY)
    if(CPP_PACKAGE_BOILERPLATE_HEADER_ONLY)
        add_library(${PROJECT_NAME} INTERFACE)
        target_compile_definitions(${PROJECT_NAME} INTERFACE CPP_PACKAGE_BOILERPLATE_HEADER_ONLY=1)
        set_target_properties(
            ${PROJECT_NAME}
            PROPERTIES
                CXX_STANDARD ${CMAKE_CXX_STANDARD}
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF)
        target_include_directories(
            ${PROJECT_NAME}
            INTERFACE
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_sources(
            ${PROJECT_NAME}
            INTERFACE
                FILE_SET HEADERS
                BASE_DIRS ${PROJECT_SOURCE_DIR}/include
                FILES ${public_headers})
    else()
        add_library(${PROJECT_NAME})
        target_sources(
            ${PROJECT_NAME}
            PRIVATE
                mylib.cpp
            PUBLIC
                FILE_SET HEADERS
                BASE_DIRS ${PROJECT_SOURCE_DIR}/include
                FILES ${public_headers})
        target_include_directories(
            ${PROJECT_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        cpp_package_boilerplate_apply_target_options(${PROJECT_NAME})
    endif()

    add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()

if(CPP_PACKAGE_BOILERPLATE_BUILD_CLI)
    add_executable(${PROJECT_NAME}_cli main.cpp)

    if(CPP_PACKAGE_BOILERPLATE_BUILD_LIBRARY)
        target_link_libraries(${PROJECT_NAME}_cli PRIVATE ${PROJECT_NAME})
    else()
        target_include_directories(${PROJECT_NAME}_cli PRIVATE ${PROJECT_SOURCE_DIR}/include)
        target_sources(${PROJECT_NAME}_cli PRIVATE mylib.cpp)
    endif()

    cpp_package_boilerplate_apply_target_options(${PROJECT_NAME}_cli)
endif()
