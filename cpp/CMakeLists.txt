cmake_minimum_required(VERSION 3.16)
project(turboquant_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TURBOQUANT_BUILD_TESTS "Build unit tests" ON)
option(TURBOQUANT_BUILD_PYTHON "Build pybind11 _native extension" OFF)

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
# Header-only: do not add_subdirectory(Eigen) — avoids Eigen's full CMake
# (unit tests, SuiteSparse, Boost, CUDA probes, ~30s+ configure noise).
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()

add_library(turboquant STATIC
  src/lloyd_max.cpp
  src/turboquant.cpp
)
target_include_directories(turboquant
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)
target_compile_features(turboquant PUBLIC cxx_std_17)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(turboquant PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(turboquant PUBLIC TURBOQUANT_USE_OPENMP=1)
else()
  target_compile_definitions(turboquant PUBLIC TURBOQUANT_USE_OPENMP=0)
endif()

if(MSVC)
  target_compile_options(turboquant PRIVATE /O2)
else()
  target_compile_options(turboquant PRIVATE -O3 -ffast-math)
endif()

if(TURBOQUANT_BUILD_TESTS)
  enable_testing()
  add_executable(turboquant_tests tests/test_basic.cpp)
  target_link_libraries(turboquant_tests PRIVATE turboquant)
  add_test(NAME turboquant_tests COMMAND turboquant_tests)
endif()

if(TURBOQUANT_BUILD_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_native MODULE bindings/module.cpp)
  target_link_libraries(_native PRIVATE turboquant)
  target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  if(MSVC)
    target_compile_options(_native PRIVATE /O2)
  else()
    target_compile_options(_native PRIVATE -O3 -ffast-math)
  endif()
  install(
    TARGETS _native
    LIBRARY DESTINATION turboquant
    ARCHIVE DESTINATION turboquant
    RUNTIME DESTINATION turboquant)
endif()
