#!/bin/bash

# Comprehensive Profiling Script for cuDF Table Converter
# Combines all profiling functionality into one script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "cuDF Table Converter Profiling Script"
    echo "====================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  run [simple|full]     Run profiling (default: simple)"
    echo "  view                  View latest profiling results in nsys-ui"
    echo "  cleanup               Clean up old profiling results"
    echo "  list                  List all profiling directories"
    echo "  stats [dir]           Generate statistics for specific directory"
    echo "  help                  Show this help message"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR      Specify custom output directory"
    echo "  --keep-n N            Keep only N latest profiling runs (cleanup)"
    echo ""
    echo "Examples:"
    echo "  $0 run                # Run simple profiling"
    echo "  $0 run full           # Run full profiling with all options"
    echo "  $0 view               # View latest results in nsys-ui"
    echo "  $0 cleanup --keep-n 3 # Keep only 3 latest runs"
    echo "  $0 stats 20250914_070650 # Generate stats for specific directory"
}

# Function to check if nsys is available
check_nsys() {
    if ! command -v nsys &> /dev/null; then
        print_error "nsys not found. Please install NVIDIA Nsight Systems."
        exit 1
    fi
}

# Function to check if build directory exists
check_build() {
    if [ ! -d "build" ]; then
        print_error "build directory not found. Please run 'mkdir build && cd build && cmake .. && make' first."
        exit 1
    fi
    
    if [ ! -f "build/benchmark_converter" ] || [ ! -f "build/performance_comparison" ]; then
        print_error "Benchmark executables not found. Please build the project first."
        exit 1
    fi
}

# Function to run profiling
run_profiling() {
    local profile_type=${1:-"simple"}
    local output_dir=${2:-""}
    
    print_status "Starting profiling (type: $profile_type)..."
    
    # Create timestamped directory
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    if [ -n "$output_dir" ]; then
        PROFILING_DIR="$output_dir/${TIMESTAMP}"
    else
        PROFILING_DIR="profiling_results/${TIMESTAMP}"
    fi
    mkdir -p "$PROFILING_DIR"
    
    print_status "Profiling results will be saved to: $PROFILING_DIR"
    echo ""
    
    # Choose profiling options based on type
    if [ "$profile_type" = "full" ]; then
        NSYS_OPTS="--trace=cuda,nvtx,osrt --cuda-memory-usage=true"
    else
        NSYS_OPTS="--trace=cuda,nvtx --cuda-memory-usage=true"
    fi
    
    # Profile benchmark_converter
    print_status "Profiling benchmark_converter..."
    nsys profile \
        --output="$PROFILING_DIR/benchmark_converter" \
        --force-overwrite=true \
        $NSYS_OPTS \
        build/benchmark_converter
    
    # Profile performance_comparison
    print_status "Profiling performance_comparison..."
    nsys profile \
        --output="$PROFILING_DIR/performance_comparison" \
        --force-overwrite=true \
        $NSYS_OPTS \
        build/performance_comparison
    
    print_success "Profiling completed!"
    print_status "Results saved in $PROFILING_DIR/ directory"
    
    # List generated files
    echo ""
    print_status "Generated files:"
    ls -lh "$PROFILING_DIR"/*.nsys-rep
    
    echo ""
    print_status "To view the results:"
    echo "  nsys-ui $PROFILING_DIR/benchmark_converter.nsys-rep"
    echo "  nsys-ui $PROFILING_DIR/performance_comparison.nsys-rep"
    echo ""
    print_status "To generate statistics:"
    echo "  $0 stats $PROFILING_DIR"
}

# Function to view latest results
view_results() {
    print_status "Finding latest profiling results..."
    
    # Check if profiling_results directory exists
    if [ ! -d "profiling_results" ]; then
        print_error "No profiling_results directory found. Run profiling first."
        exit 1
    fi
    
    # Find the most recent profiling directory
    LATEST_DIR=$(find profiling_results -type d -name "20*" | sort | tail -1)
    
    if [ -z "$LATEST_DIR" ]; then
        print_error "No timestamped profiling directories found."
        print_status "Available directories:"
        ls -la profiling_results/
        exit 1
    fi
    
    print_success "Latest profiling results: $LATEST_DIR"
    echo ""
    
    # Check if nsys-ui is available
    if ! command -v nsys-ui &> /dev/null; then
        print_error "nsys-ui not found. Please install NVIDIA Nsight Systems GUI."
        print_status "Profiling data files are available in $LATEST_DIR/ directory."
        print_status "To install nsys-ui, install the full NVIDIA Nsight Systems package."
        exit 1
    fi
    
    # Check if .nsys-rep files exist
    if [ ! -f "$LATEST_DIR/benchmark_converter.nsys-rep" ] && [ ! -f "$LATEST_DIR/performance_comparison.nsys-rep" ]; then
        print_error "No .nsys-rep files found in $LATEST_DIR"
        exit 1
    fi
    
    print_status "Available profiling results in $LATEST_DIR:"
    ls -lh "$LATEST_DIR"/*.nsys-rep 2>/dev/null
    
    echo ""
    print_status "Launching nsys-ui..."
    
    # Launch nsys-ui with available files
    if [ -f "$LATEST_DIR/benchmark_converter.nsys-rep" ] && [ -f "$LATEST_DIR/performance_comparison.nsys-rep" ]; then
        print_status "Opening both benchmark files in nsys-ui..."
        nsys-ui "$LATEST_DIR/benchmark_converter.nsys-rep" "$LATEST_DIR/performance_comparison.nsys-rep" &
    elif [ -f "$LATEST_DIR/benchmark_converter.nsys-rep" ]; then
        print_status "Opening benchmark_converter in nsys-ui..."
        nsys-ui "$LATEST_DIR/benchmark_converter.nsys-rep" &
    elif [ -f "$LATEST_DIR/performance_comparison.nsys-rep" ]; then
        print_status "Opening performance_comparison in nsys-ui..."
        nsys-ui "$LATEST_DIR/performance_comparison.nsys-rep" &
    fi
    
    print_success "nsys-ui launched! Check your desktop for the profiler window."
    echo ""
    print_status "To generate statistics:"
    echo "  $0 stats $LATEST_DIR"
}


# Function to clean up old results
cleanup_results() {
    local keep_n=${1:-3}
    
    print_status "Cleaning up old profiling results (keeping latest $keep_n runs)..."
    
    # Check if profiling_results directory exists
    if [ ! -d "profiling_results" ]; then
        print_warning "No profiling_results directory found."
        exit 0
    fi
    
    print_status "Current profiling directories:"
    ls -la profiling_results/
    
    # Get all directories except the current ones, sort by time, and remove all but the latest N
    local dirs_to_remove=$(find profiling_results -type d -name "20*" -o -name "migration_*" | sort | head -n -$keep_n)
    
    if [ -z "$dirs_to_remove" ]; then
        print_warning "No directories found to remove."
    else
        echo ""
        print_status "Directories to be removed:"
        echo "$dirs_to_remove"
        echo ""
        read -p "Are you sure you want to remove these directories? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            echo "$dirs_to_remove" | xargs rm -rf
            print_success "Cleanup completed! Removed $(echo "$dirs_to_remove" | wc -l) directories."
        else
            print_status "Cleanup cancelled."
        fi
    fi
    
    echo ""
    print_status "Remaining profiling directories:"
    ls -la profiling_results/
}

# Function to list all profiling directories
list_results() {
    print_status "All profiling directories:"
    
    if [ ! -d "profiling_results" ]; then
        print_warning "No profiling_results directory found."
        exit 0
    fi
    
    ls -la profiling_results/
    
    echo ""
    print_status "Directory sizes:"
    du -sh profiling_results/* 2>/dev/null || print_warning "No subdirectories found."
}

# Function to generate statistics
generate_stats() {
    local target_dir=${1:-""}
    
    if [ -z "$target_dir" ]; then
        # Find the most recent profiling directory
        target_dir=$(find profiling_results -type d -name "20*" | sort | tail -1)
        if [ -z "$target_dir" ]; then
            print_error "No profiling directories found. Run profiling first."
            exit 1
        fi
    fi
    
    if [ ! -d "$target_dir" ]; then
        print_error "Directory $target_dir not found."
        exit 1
    fi
    
    print_status "Generating statistics for $target_dir..."
    
    # Generate stats for benchmark_converter
    if [ -f "$target_dir/benchmark_converter.nsys-rep" ]; then
        print_status "Generating stats for benchmark_converter..."
        nsys stats "$target_dir/benchmark_converter.nsys-rep"
    fi
    
    # Generate stats for performance_comparison
    if [ -f "$target_dir/performance_comparison.nsys-rep" ]; then
        print_status "Generating stats for performance_comparison..."
        nsys stats "$target_dir/performance_comparison.nsys-rep"
    fi
    
    print_success "Statistics generation completed!"
}

# Main script logic
main() {
    local command=${1:-"help"}
    local arg1=${2:-""}
    local arg2=${3:-""}
    
    case $command in
        "run")
            check_nsys
            check_build
            run_profiling "$arg1" "$arg2"
            ;;
        "view")
            check_nsys
            view_results
            ;;
        "cleanup")
            cleanup_results "$arg1"
            ;;
        "list")
            list_results
            ;;
        "stats")
            check_nsys
            generate_stats "$arg1"
            ;;
        "help"|"--help"|"-h")
            show_usage
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Parse command line arguments
OUTPUT_DIR=""
KEEP_N=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --keep-n)
            KEEP_N="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Run main function with all arguments
main "$@"
