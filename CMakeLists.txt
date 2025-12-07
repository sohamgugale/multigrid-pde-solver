cmake_minimum_required(VERSION 3.18)
project(MultigridPDESolver VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

option(USE_MPI "Enable MPI parallelization" OFF)
option(USE_OPENMP "Enable OpenMP parallelization" ON)
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)

if(USE_MPI)
    find_package(MPI REQUIRED)
    add_definitions(-DUSE_MPI)
endif()

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DUSE_OPENMP)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    add_definitions(-DUSE_CUDA)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

set(COMMON_SOURCES
    src/common/matrix.cpp
    src/utils/timer.cpp
    src/utils/logger.cpp
)

set(CPU_SOURCES
    src/cpu/fd_discretizer.cpp
    src/cpu/smoothers.cpp
    src/solvers/multigrid.cpp
)

add_library(mglib STATIC ${COMMON_SOURCES} ${CPU_SOURCES})

if(USE_MPI)
    target_link_libraries(mglib PUBLIC MPI::MPI_CXX)
endif()

if(USE_OPENMP)
    target_link_libraries(mglib PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test_mg_2d tests/test_mg_2d.cpp)
target_link_libraries(test_mg_2d mglib)

if(BUILD_BENCHMARKS)
    add_executable(benchmark_scaling benchmarks/scaling_benchmark.cpp)
    target_link_libraries(benchmark_scaling mglib)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

message(STATUS "==================================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "==================================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "MPI: ${USE_MPI}")
message(STATUS "OpenMP: ${USE_OPENMP}")
message(STATUS "CUDA: ${USE_CUDA}")
message(STATUS "==================================================")
