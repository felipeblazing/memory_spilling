#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>

namespace benchmark_output {

// Memory resource information
void print_memory_resource_info(const std::string& title, 
                               std::size_t block_size, 
                               std::size_t total_blocks, 
                               std::size_t free_blocks);

// Dataset creation information
void print_dataset_info(const std::string& title,
                       std::size_t target_size_mb,
                       std::size_t num_rows,
                       std::size_t expected_blocks,
                       std::size_t num_columns,
                       std::size_t num_rows_actual);

// Benchmark section headers
void print_benchmark_header(const std::string& title);
void print_section_header(const std::string& title, char separator = '=');

// Conversion results
void print_conversion_results(const std::string& operation,
                             std::size_t blocks_allocated,
                             std::size_t data_size,
                             std::size_t free_blocks_after,
                             const std::chrono::microseconds& duration,
                             double throughput_mb_s);

// Concurrent conversion results
void print_concurrent_conversion_results(const std::string& operation,
                                        std::size_t blocks_per_conversion,
                                        std::size_t data_size_per_conversion,
                                        std::size_t free_blocks_after,
                                        const std::chrono::microseconds& total_duration,
                                        double aggregate_throughput_mb_s);

// Recreation results
void print_recreation_results(const std::string& operation,
                             std::size_t num_columns,
                             std::size_t num_rows,
                             const std::chrono::microseconds& duration,
                             double throughput_mb_s);

// Concurrent recreation results
void print_concurrent_recreation_results(const std::string& operation,
                                        std::size_t num_columns,
                                        std::size_t num_rows,
                                        const std::chrono::microseconds& total_duration,
                                        double aggregate_throughput_mb_s);

// Data integrity verification
void print_data_integrity_results(bool data_matches, 
                                 const std::string& error_message = "");

// Performance comparison table
void print_performance_comparison_table(const std::string& operation,
                                       double single_time_ms,
                                       double concurrent_time_ms,
                                       double speedup,
                                       double efficiency);

// Throughput comparison table
void print_throughput_comparison_table(const std::string& operation,
                                      double single_throughput,
                                      double concurrent_throughput,
                                      double improvement);

// Summary information
void print_benchmark_summary(const std::string& title,
                            std::size_t dataset_size_mb,
                            std::size_t num_streams,
                            std::size_t total_data_mb,
                            std::size_t blocks_per_conversion,
                            double convert_time_ms,
                            double convert_throughput,
                            double recreate_time_ms,
                            double recreate_throughput);

// Table separators
void print_table_separator(char separator = '-', int width = 80);

// Error messages
void print_error(const std::string& message);
void print_success(const std::string& message);

} // namespace benchmark_output
