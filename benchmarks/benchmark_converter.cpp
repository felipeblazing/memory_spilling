#include <iostream>
#include <memory>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <future>
#include <atomic>

#include "cudf_table_converter.hpp"
#include "fixed_size_host_memory_resource.hpp"
#include "benchmark_output.hpp"
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/contiguous_split.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/device_buffer.hpp>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

// Helper function to perform conversion to host in a separate thread
std::pair<spilling::table_allocation, std::chrono::microseconds> 
convert_to_host_async(const cudf::table_view& table_view, 
                      spilling::fixed_size_host_memory_resource* mr,
                      rmm::cuda_stream_view stream) {
    nvtx3::scoped_range range{"convert_to_host_async", nvtx3::rgb{255, 0, 255}}; // Magenta for async convert
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto table_alloc = spilling::cudf_table_converter::convert_to_host(
        table_view, mr, stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return {std::move(table_alloc), duration};
}

// Helper function to perform recreation from host in a separate thread
std::pair<cudf::table, std::chrono::microseconds> 
recreate_from_host_async(const spilling::table_allocation& table_alloc,
                         rmm::cuda_stream_view stream) {
    nvtx3::scoped_range range{"recreate_from_host_async", nvtx3::rgb{0, 255, 255}}; // Cyan for async recreate
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto recreated_table = spilling::cudf_table_converter::recreate_table(
        table_alloc, stream);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    return {std::move(recreated_table), duration};
}

int main() {
    benchmark_output::print_benchmark_header("cuDF Table Converter Benchmark");

    try {
        // Create our custom fixed-size memory resource
        constexpr std::size_t block_size = 4 * 1024 * 1024;  // 4MB blocks
        constexpr std::size_t pool_size = 256;               // 256 blocks (1GB total)
        
        auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
            block_size, pool_size);

        benchmark_output::print_memory_resource_info(
            "Created custom memory resource:",
            custom_mr->get_block_size(),
            custom_mr->get_total_blocks(),
            custom_mr->get_free_blocks()
        );

        // Create a large dataset (500MB)
        constexpr std::size_t target_size_mb = 500;
        constexpr std::size_t target_size_bytes = target_size_mb * 1024 * 1024;
        constexpr std::size_t int32_size = sizeof(int32_t);
        constexpr std::size_t num_rows = target_size_bytes / int32_size;
        
        benchmark_output::print_dataset_info(
            "\nCreating benchmark dataset...",
            target_size_mb,
            num_rows,
            (target_size_bytes + block_size - 1) / block_size,
            1, // num_columns
            num_rows // num_rows_actual
        );

        // Create test data
        std::vector<int32_t> test_data(num_rows);
        for (size_t i = 0; i < num_rows; ++i) {
            test_data[i] = static_cast<int32_t>(i);
        }

        // Create cuDF column using device buffer
        cudaStream_t default_stream;
        cudaStreamCreate(&default_stream);
        rmm::device_buffer gpu_buffer(test_data.data(), test_data.size() * sizeof(int32_t), rmm::cuda_stream_view(default_stream));
        
        // Create column view first, then create column from it
        auto column_view = cudf::column_view(
            cudf::data_type{cudf::type_id::INT32}, 
            num_rows, 
            gpu_buffer.data(),
            nullptr,  // null mask
            0,        // null count
            0,        // offset
            std::vector<cudf::column_view>{}  // children
        );
        
        auto column = std::make_unique<cudf::column>(column_view);

        // Create table
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.push_back(std::move(column));
        cudf::table original_table(std::move(columns));
        cudf::table_view table_view = original_table.view();

        // Table info already printed in print_dataset_info

        // Create CUDA streams for concurrent operations
        constexpr int num_streams = 2;
        std::vector<rmm::cuda_stream_view> streams;
        for (int i = 0; i < num_streams; ++i) {
            // Create actual separate CUDA streams
            cudaStream_t cuda_stream;
            cudaStreamCreate(&cuda_stream);
            streams.push_back(rmm::cuda_stream_view(cuda_stream));
        }

        // Create separate datasets for each stream to avoid contention
        std::vector<cudf::table_view> table_views;
        std::vector<std::unique_ptr<rmm::device_buffer>> device_buffers; // Keep device buffers alive
        std::vector<std::unique_ptr<cudf::table>> tables; // Keep tables alive
        
        for (int i = 0; i < num_streams; ++i) {
            // Create separate test data for each stream
            std::vector<int32_t> stream_test_data(num_rows);
            for (size_t j = 0; j < num_rows; ++j) {
                stream_test_data[j] = static_cast<int32_t>(j + i * 1000000); // Offset data for each stream
            }

            // Create cuDF column using device buffer - keep buffer alive
            auto stream_gpu_buffer = std::make_unique<rmm::device_buffer>(
                stream_test_data.data(), stream_test_data.size() * sizeof(int32_t), streams[i]);
            device_buffers.push_back(std::move(stream_gpu_buffer));
            
            // Create column view first, then create column from it
            auto stream_column_view = cudf::column_view(
                cudf::data_type{cudf::type_id::INT32}, 
                num_rows, 
                device_buffers[i]->data(),
                nullptr,  // null mask
                0,        // null count
                0,        // offset
                std::vector<cudf::column_view>{}  // children
            );
            
            auto stream_column = std::make_unique<cudf::column>(stream_column_view);

            // Create table - keep table alive
            std::vector<std::unique_ptr<cudf::column>> stream_columns;
            stream_columns.push_back(std::move(stream_column));
            auto stream_table = std::make_unique<cudf::table>(std::move(stream_columns));
            tables.push_back(std::move(stream_table));
            table_views.push_back(tables[i]->view());
        }

        std::cout << "  Created " << num_streams << " separate datasets for concurrent processing" << std::endl;

        // Benchmark: Concurrent Convert to host memory (2 threads, 2 streams)
        std::cout << "\nBenchmarking: Concurrent Convert to host memory (2 threads, 2 streams)..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch 2 concurrent conversions with separate data
        std::vector<std::future<std::pair<spilling::table_allocation, std::chrono::microseconds>>> convert_futures;
        for (int i = 0; i < num_streams; ++i) {
            convert_futures.push_back(std::async(std::launch::async, [&](int stream_idx) {
                return convert_to_host_async(table_views[stream_idx], custom_mr.get(), streams[stream_idx]);
            }, i));
        }
        
        // Wait for all conversions to complete
        std::vector<spilling::table_allocation> table_allocs;
        std::vector<std::chrono::microseconds> convert_durations;
        for (auto& future : convert_futures) {
            auto [alloc, duration] = future.get();
            table_allocs.push_back(std::move(alloc));
            convert_durations.push_back(duration);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_convert_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double aggregate_convert_throughput_mb_s = (target_size_mb * num_streams * 1000000.0) / total_convert_duration.count();
        benchmark_output::print_concurrent_conversion_results(
            "Concurrent conversion",
            table_allocs[0].allocation.size(),
            table_allocs[0].data_size,
            custom_mr->get_free_blocks(),
            total_convert_duration,
            aggregate_convert_throughput_mb_s
        );

        // Benchmark: Concurrent Recreate from host memory (2 threads, 2 streams)
        std::cout << "\nBenchmarking: Concurrent Recreate from host memory (2 threads, 2 streams)..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        
        // Launch 2 concurrent recreations
        std::vector<std::future<std::pair<cudf::table, std::chrono::microseconds>>> recreate_futures;
        for (int i = 0; i < num_streams; ++i) {
            recreate_futures.push_back(std::async(std::launch::async, [&](int stream_idx) {
                return recreate_from_host_async(table_allocs[stream_idx], streams[stream_idx]);
            }, i));
        }
        
        // Wait for all recreations to complete
        std::vector<cudf::table> recreated_tables;
        std::vector<std::chrono::microseconds> recreate_durations;
        for (auto& future : recreate_futures) {
            auto [table, duration] = future.get();
            recreated_tables.push_back(std::move(table));
            recreate_durations.push_back(duration);
        }
        
        end_time = std::chrono::high_resolution_clock::now();
        auto total_recreate_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        double aggregate_recreate_throughput_mb_s = (target_size_mb * num_streams * 1000000.0) / total_recreate_duration.count();
        benchmark_output::print_concurrent_recreation_results(
            "Concurrent recreation",
            recreated_tables[0].num_columns(),
            recreated_tables[0].num_rows(),
            total_recreate_duration,
            aggregate_recreate_throughput_mb_s
        );


        // Verify data integrity
        std::vector<int32_t> original_sample(1000);
        std::vector<int32_t> recreated_sample(1000);
        
        cudaMemcpy(original_sample.data(), table_view.column(0).data<int32_t>(), 
                   1000 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(recreated_sample.data(), recreated_tables[0].view().column(0).data<int32_t>(), 
                   1000 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        bool data_matches = true;
        std::string error_message;
        for (size_t i = 0; i < 1000; ++i) {
            if (original_sample[i] != recreated_sample[i]) {
                data_matches = false;
                error_message = "Data mismatch at index " + std::to_string(i) + 
                              ": original=" + std::to_string(original_sample[i]) + 
                              ", recreated=" + std::to_string(recreated_sample[i]);
                break;
            }
        }
        
        benchmark_output::print_data_integrity_results(data_matches, error_message);
        if (!data_matches) {
            return 1;
        }

        // Summary
        benchmark_output::print_benchmark_summary(
            "Concurrent Benchmark Summary (2 streams, separate data)",
            target_size_mb,
            num_streams,
            target_size_mb * num_streams,
            table_allocs[0].allocation.size(),
            total_convert_duration.count() / 1000.0,
            aggregate_convert_throughput_mb_s,
            total_recreate_duration.count() / 1000.0,
            aggregate_recreate_throughput_mb_s
        );

        // Clean up CUDA streams
        cudaStreamDestroy(default_stream);
        for (auto& stream : streams) {
            cudaStreamDestroy(stream.value());
        }

    } catch (const std::exception& e) {
        benchmark_output::print_error(e.what());
        return 1;
    }

    return 0;
}
