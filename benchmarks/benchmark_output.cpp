#include "benchmark_output.hpp"

namespace benchmark_output {

void print_memory_resource_info(const std::string& title, 
                               std::size_t block_size, 
                               std::size_t total_blocks, 
                               std::size_t free_blocks) {
    std::cout << title << std::endl;
    std::cout << "  Block size: " << block_size << " bytes" << std::endl;
    std::cout << "  Pool size: " << total_blocks << " blocks" << std::endl;
    std::cout << "  Total capacity: " << (total_blocks * block_size) / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Free blocks: " << free_blocks << std::endl;
}

void print_dataset_info(const std::string& title,
                       std::size_t target_size_mb,
                       std::size_t num_rows,
                       std::size_t expected_blocks,
                       std::size_t num_columns,
                       std::size_t num_rows_actual) {
    std::cout << title << std::endl;
    std::cout << "  Target size: " << target_size_mb << " MB" << std::endl;
    std::cout << "  Number of rows: " << num_rows << std::endl;
    std::cout << "  Expected blocks needed: " << expected_blocks << std::endl;
    std::cout << "  Table created with " << num_columns << " columns and " 
              << num_rows_actual << " rows" << std::endl;
}

void print_benchmark_header(const std::string& title) {
    std::cout << title << std::endl;
    std::cout << std::string(title.length(), '=') << std::endl;
}

void print_section_header(const std::string& title, char separator) {
    std::cout << "\n" << std::string(60, separator) << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, separator) << std::endl;
}

void print_conversion_results(const std::string& operation,
                             std::size_t blocks_allocated,
                             std::size_t data_size,
                             std::size_t free_blocks_after,
                             const std::chrono::microseconds& duration,
                             double throughput_mb_s) {
    std::cout << "  " << operation << " completed!" << std::endl;
    std::cout << "  Allocated " << blocks_allocated << " blocks" << std::endl;
    std::cout << "  Data size: " << data_size << " bytes" << std::endl;
    std::cout << "  Free blocks after conversion: " << free_blocks_after << std::endl;
    std::cout << "  Conversion time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Conversion time: " << std::fixed << std::setprecision(3) 
              << (duration.count() / 1000.0) << " milliseconds" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << throughput_mb_s << " MB/s" << std::endl;
}

void print_concurrent_conversion_results(const std::string& operation,
                                        std::size_t blocks_per_conversion,
                                        std::size_t data_size_per_conversion,
                                        std::size_t free_blocks_after,
                                        const std::chrono::microseconds& total_duration,
                                        double aggregate_throughput_mb_s) {
    std::cout << "  " << operation << " completed!" << std::endl;
    std::cout << "  Allocated " << blocks_per_conversion << " blocks per conversion" << std::endl;
    std::cout << "  Data size: " << data_size_per_conversion << " bytes per conversion" << std::endl;
    std::cout << "  Free blocks after conversion: " << free_blocks_after << std::endl;
    std::cout << "  Total conversion time: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "  Total conversion time: " << std::fixed << std::setprecision(3) 
              << (total_duration.count() / 1000.0) << " milliseconds" << std::endl;
    std::cout << "  Aggregate throughput: " << std::fixed << std::setprecision(2) 
              << aggregate_throughput_mb_s << " MB/s" << std::endl;
}

void print_recreation_results(const std::string& operation,
                             std::size_t num_columns,
                             std::size_t num_rows,
                             const std::chrono::microseconds& duration,
                             double throughput_mb_s) {
    std::cout << "  " << operation << " completed!" << std::endl;
    std::cout << "  Recreated table has " << num_columns << " columns and " 
              << num_rows << " rows" << std::endl;
    std::cout << "  Recreation time: " << duration.count() << " microseconds" << std::endl;
    std::cout << "  Recreation time: " << std::fixed << std::setprecision(3) 
              << (duration.count() / 1000.0) << " milliseconds" << std::endl;
    std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
              << throughput_mb_s << " MB/s" << std::endl;
}

void print_concurrent_recreation_results(const std::string& operation,
                                        std::size_t num_columns,
                                        std::size_t num_rows,
                                        const std::chrono::microseconds& total_duration,
                                        double aggregate_throughput_mb_s) {
    std::cout << "  " << operation << " completed!" << std::endl;
    std::cout << "  Recreated tables have " << num_columns << " columns and " 
              << num_rows << " rows each" << std::endl;
    std::cout << "  Total recreation time: " << total_duration.count() << " microseconds" << std::endl;
    std::cout << "  Total recreation time: " << std::fixed << std::setprecision(3) 
              << (total_duration.count() / 1000.0) << " milliseconds" << std::endl;
    std::cout << "  Aggregate throughput: " << std::fixed << std::setprecision(2) 
              << aggregate_throughput_mb_s << " MB/s" << std::endl;
}

void print_data_integrity_results(bool data_matches, const std::string& error_message) {
    std::cout << "\nVerifying data integrity..." << std::endl;
    if (data_matches) {
        std::cout << "  ✓ Data integrity verified!" << std::endl;
    } else {
        std::cout << "  ✗ Data integrity check failed!" << std::endl;
        if (!error_message.empty()) {
            std::cout << "  " << error_message << std::endl;
        }
    }
}

void print_performance_comparison_table(const std::string& operation,
                                       double single_time_ms,
                                       double concurrent_time_ms,
                                       double speedup,
                                       double efficiency) {
    std::cout << std::setw(25) << operation 
              << std::setw(15) << std::fixed << std::setprecision(3) << single_time_ms
              << std::setw(15) << std::fixed << std::setprecision(3) << concurrent_time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "x"
              << std::setw(15) << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
}

void print_throughput_comparison_table(const std::string& operation,
                                      double single_throughput,
                                      double concurrent_throughput,
                                      double improvement) {
    std::cout << std::setw(25) << operation 
              << std::setw(20) << std::fixed << std::setprecision(2) << single_throughput
              << std::setw(20) << std::fixed << std::setprecision(2) << concurrent_throughput
              << std::setw(15) << std::fixed << std::setprecision(2) << improvement << "x" << std::endl;
}

void print_benchmark_summary(const std::string& title,
                            std::size_t dataset_size_mb,
                            std::size_t num_streams,
                            std::size_t total_data_mb,
                            std::size_t blocks_per_conversion,
                            double convert_time_ms,
                            double convert_throughput,
                            double recreate_time_ms,
                            double recreate_throughput) {
    std::cout << "\n" << title << std::endl;
    std::cout << std::string(title.length(), '=') << std::endl;
    std::cout << "Dataset size per stream: " << dataset_size_mb << " MB" << std::endl;
    std::cout << "Number of concurrent streams: " << num_streams << std::endl;
    std::cout << "Total data processed: " << total_data_mb << " MB" << std::endl;
    std::cout << "Blocks used per conversion: " << blocks_per_conversion << std::endl;
    std::cout << "Concurrent convert to host: " << std::fixed << std::setprecision(3) 
              << convert_time_ms << " ms (" 
              << std::setprecision(2) << convert_throughput << " MB/s)" << std::endl;
    std::cout << "Concurrent recreate from host: " << std::fixed << std::setprecision(3) 
              << recreate_time_ms << " ms (" 
              << std::setprecision(2) << recreate_throughput << " MB/s)" << std::endl;
    std::cout << "\nBenchmark completed successfully!" << std::endl;
}

void print_table_separator(char separator, int width) {
    std::cout << std::string(width, separator) << std::endl;
}

void print_error(const std::string& message) {
    std::cerr << "Error: " << message << std::endl;
}

void print_success(const std::string& message) {
    std::cout << message << std::endl;
}

} // namespace benchmark_output
