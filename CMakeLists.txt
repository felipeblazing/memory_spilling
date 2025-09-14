cmake_minimum_required(VERSION 3.20)

# Set CUDA compiler path
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(rmm_example 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Find RMM, cuDF, and Catch2 packages
find_package(rmm REQUIRED)
find_package(cudf REQUIRED)
find_package(Catch2 REQUIRED)

# Create unified test executable
add_executable(tests 
    tests/main.cpp
    tests/simple_test.cpp
    tests/cudf_to_host_test.cpp
    tests/host_to_cudf_test.cpp
    tests/custom_memory_resource_test.cpp
    tests/multiple_blocks_test.cpp
    src/fixed_size_host_memory_resource.cpp
    src/cudf_table_converter.cpp
)
# Add benchmark executables
add_executable(benchmark_converter 
    benchmarks/benchmark_converter.cpp
    benchmarks/benchmark_output.cpp
    src/fixed_size_host_memory_resource.cpp
    src/cudf_table_converter.cpp
)
add_executable(performance_comparison 
    benchmarks/performance_comparison.cpp
    benchmarks/benchmark_output.cpp
    src/fixed_size_host_memory_resource.cpp
    src/cudf_table_converter.cpp
)

# Include directories for tests
target_include_directories(tests PRIVATE include)

# Include directories for benchmarks
target_include_directories(benchmark_converter PRIVATE include)
target_include_directories(performance_comparison PRIVATE include)

# Link libraries for test executable
target_link_libraries(tests 
    PRIVATE 
    rmm::rmm
    cudf::cudf
    Catch2::Catch2WithMain
)

# Link libraries for benchmark executables
target_link_libraries(benchmark_converter 
    PRIVATE 
    rmm::rmm
    cudf::cudf
)

target_link_libraries(performance_comparison 
    PRIVATE 
    rmm::rmm
    cudf::cudf
)


# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75)  # For RTX 20xx series, adjust as needed

# Enable CTest
enable_testing()

# Add Catch2 test discovery
include(CTest)
include(Catch)
catch_discover_tests(tests)

# Print some information
message(STATUS "RMM version: ${RMM_VERSION}")
message(STATUS "RMM found: ${RMM_FOUND}")
message(STATUS "Catch2 found: ${Catch2_FOUND}")
