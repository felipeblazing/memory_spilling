# Profiling Script Documentation

## Overview
The `profile.sh` script is a comprehensive tool for profiling the cuDF Table Converter benchmarks using NVIDIA Nsight Systems. It combines all profiling functionality into a single, easy-to-use script.

## Features
- **Unified Interface**: All profiling operations in one script
- **Timestamped Directories**: Automatic organization of results
- **Multiple Profiling Modes**: Simple and full profiling options
- **Result Management**: View, migrate, cleanup, and list results
- **Statistics Generation**: Automatic performance analysis
- **Error Handling**: Comprehensive error checking and user feedback

## Usage

### Basic Commands
```bash
# Show help
./profile.sh help

# Run simple profiling (default)
./profile.sh run

# Run full profiling with all options
./profile.sh run full

# View latest results in nsys-ui (launches GUI)
./profile.sh view

# List all profiling directories
./profile.sh list

# Generate statistics for latest run
./profile.sh stats

# Generate statistics for specific directory
./profile.sh stats profiling_results/20250914_070859
```

### Advanced Commands
```bash
# Clean up old results (interactive)
./profile.sh cleanup

# Clean up keeping only 3 latest runs
./profile.sh cleanup --keep-n 3

# Run profiling with custom output directory
./profile.sh run --output-dir /path/to/custom/dir
```

## Directory Structure
```
profiling_results/
├── 20250914_070859/          # Latest profiling run
│   ├── benchmark_converter.nsys-rep
│   ├── performance_comparison.nsys-rep
│   ├── benchmark_converter.sqlite
│   └── performance_comparison.sqlite
├── 20250914_070650/          # Previous run
│   └── ...
```

## Profiling Modes

### Simple Profiling
- Basic CUDA and NVTX tracing
- Memory usage tracking
- Faster execution
- Good for regular performance monitoring

### Full Profiling
- Complete CUDA, NVTX, and OSRT tracing
- Capture range profiling
- More detailed analysis
- Slower execution

## Output Files

### .nsys-rep Files
- Raw profiling data from Nsight Systems
- Can be opened with `nsys-ui` for visual analysis
- Contains detailed timeline and performance metrics

### .sqlite Files
- Processed statistics and reports
- Generated automatically when running `stats` command
- Contains performance summaries and analysis

## Examples

### Quick Performance Check
```bash
# Run simple profiling and view results in nsys-ui
./profile.sh run
./profile.sh view
```

### Detailed Analysis
```bash
# Run full profiling with detailed analysis
./profile.sh run full
./profile.sh stats
```

### Cleanup Old Results
```bash
# Keep only the latest 5 profiling runs
./profile.sh cleanup --keep-n 5
```

### View Specific Results
```bash
# List all available results
./profile.sh list

# View latest results in nsys-ui (automatically opens both files)
./profile.sh view

# View specific directory manually
nsys-ui profiling_results/20250914_070859/benchmark_converter.nsys-rep
```

## Requirements
- NVIDIA Nsight Systems installed
- Built project in `build/` directory
- `benchmark_converter` and `performance_comparison` executables

## Troubleshooting

### Common Issues
1. **"nsys not found"**: Install NVIDIA Nsight Systems or add nsys to your path
2. **"build directory not found"**: Run `mkdir build && cd build && cmake .. && make`
3. **"No profiling directories found"**: Run profiling first with `./profile.sh run`

### Getting Help
```bash
# Show detailed help
./profile.sh help

# Check if everything is set up correctly
./profile.sh list
```

## Performance Tips
- Use simple profiling for regular monitoring
- Use full profiling for detailed analysis
- Clean up old results regularly to save disk space
- Generate statistics after profiling for quick analysis
