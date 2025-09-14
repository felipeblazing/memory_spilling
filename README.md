# RMM Memory Management Project

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-25.04-orange.svg)](https://rapids.ai/)

This project demonstrates custom memory management using RAPIDS Memory Manager (RMM) with cuDF integration, featuring high-performance host memory spilling capabilities for GPU data processing workflows.

## Project Structure

```
memory/
├── src/                          # Core source files
│   ├── fixed_size_host_memory_resource.cpp
│   └── cudf_table_converter.cpp
├── include/                      # Header files
│   ├── fixed_size_host_memory_resource.hpp
│   └── cudf_table_converter.hpp
├── tests/                        # Test files (Catch2 framework)
│   ├── main.cpp                  # Basic RMM test
│   ├── simple_test.cpp           # Simple cuDF to host test
│   ├── cudf_to_host_test.cpp     # cuDF to host functionality tests
│   ├── host_to_cudf_test.cpp     # Host to cuDF functionality tests
│   ├── custom_memory_resource_test.cpp
│   └── multiple_blocks_test.cpp
├── benchmarks/                   # Performance benchmarks
│   ├── benchmark_converter.cpp   # Concurrent benchmark (2 streams)
│   ├── performance_comparison.cpp # Single vs concurrent comparison
│   ├── benchmark_output.hpp      # Benchmark output utilities (header)
│   └── benchmark_output.cpp      # Benchmark output utilities (implementation)
├── build/                        # Build directory
├── CMakeLists.txt               # Build configuration
└── README.md                    # This file
```

## Features

### Spilling Memory Resource
- **Fixed-size host memory resource** with dynamic pool expansion
- **Thread-safe** allocation and deallocation
- **RAII wrapper** for multi-block allocations
- **Configurable block size** (default: 4MB) and pool size (default: 256 blocks)
- **Namespace**: `spilling` for clear identification of memory spilling functionality

### cuDF Integration
- **Table conversion** from GPU to host memory
- **Metadata preservation** for table recreation
- **Stream-based** asynchronous operations
- **Multi-block allocation** for large datasets

### Performance Benchmarks
- **Concurrent processing** with 2 streams and separate data
- **Performance comparison** between single-threaded and concurrent approaches
- **Large dataset testing** (500MB+ per stream)

## Prerequisites

### Conda Dependencies
Install the required RAPIDS and CUDA dependencies using conda:

```bash
conda install -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf=25.04 cmake
```

### CUDA Compiler
Ensure your CUDA compiler is available in one of these ways:

1. **Default location**: CUDA compiler should be at `/usr/local/cuda/bin/nvcc`
2. **Environment variable**: Set `CMAKE_CUDA_COMPILER` to point to your nvcc installation:
   ```bash
   export CMAKE_CUDA_COMPILER=/path/to/your/cuda/bin/nvcc
   ```

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Running Tests

### All Tests
```bash
# Run all tests
./tests

# Run specific test categories
./tests "[cudf]"                    # All cuDF tests
./tests "[to_host]"                 # cuDF to host tests
./tests "[host_to_cudf]"            # Host to cuDF tests
./tests "[memory]"                  # Memory resource tests
./tests "~[rmm] ~[memory]"          # Exclude problematic tests
```

### Test Structure
- **cuDF to Host Tests**: Convert cuDF tables to host memory
- **Host to cuDF Tests**: Convert host memory back to cuDF tables
- **Memory Resource Tests**: Test custom memory allocation
- **RMM Tests**: Basic RMM functionality

### Benchmarks
```bash
# Concurrent benchmark (2 streams, 4MB blocks)
./benchmark_converter

# Performance comparison (single vs concurrent)
./performance_comparison
```

## Performance Results

### 4MB Block Size (256 blocks = 1GB total)
- **Convert to host**: ~8,000-8,800 MB/s
- **Recreate from host**: ~9,200-9,900 MB/s
- **Round-trip**: ~2,600-2,700 MB/s
- **Memory efficiency**: 98% utilization

### Concurrent Performance (2 streams)
- **Aggregate throughput**: ~8,000-9,000 MB/s
- **Data integrity**: 100% verified
- **Scalability**: Efficient parallel processing

## Dependencies

The following dependencies are required and can be installed via conda:

- **RMM**: RAPIDS Memory Manager (included in libcudf)
- **cuDF**: CUDA DataFrames (libcudf=25.04)
- **CUDA**: NVIDIA CUDA Toolkit (included in libcudf)
- **CMake**: Build system
- **C++20**: Modern C++ standard for enhanced features

### Installation
```bash
conda install -c rapidsai -c conda-forge -c nvidia rapidsai::libcudf=25.04 cmake
```

## Configuration

The project uses 4MB blocks by default with 256 blocks total (1GB capacity). This can be modified in the benchmark files:

```cpp
constexpr std::size_t block_size = 4 * 1024 * 1024;  // 4MB blocks
constexpr std::size_t pool_size = 256;               // 256 blocks (1GB total)
```