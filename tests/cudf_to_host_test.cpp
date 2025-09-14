#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "cudf_table_converter.hpp"
#include "fixed_size_host_memory_resource.hpp"
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/contiguous_split.hpp>

TEST_CASE("cuDF to Host Basic Functionality", "[cudf][to_host][basic]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size);

    std::vector<int32_t> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    rmm::device_buffer gpu_buffer(test_data.size() * sizeof(int32_t), rmm::cuda_stream_view(stream));
    
    cudaMemcpy(gpu_buffer.data(), test_data.data(), 
               test_data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    auto column_view = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32}, 
        test_data.size(), 
        gpu_buffer.data(),
        nullptr,
        0,
        0,
        std::vector<cudf::column_view>{}
    );
    
    std::vector<cudf::column_view> column_views = {column_view};
    cudf::table_view table_view(column_views);

    SECTION("Test cuDF pack functionality") {
        auto packed_data = cudf::pack(table_view);
        
        REQUIRE(packed_data.gpu_data != nullptr);
        REQUIRE(packed_data.metadata != nullptr);
        REQUIRE(packed_data.gpu_data->size() > 0);
        REQUIRE(packed_data.metadata->size() > 0);
    }

    SECTION("Convert table to host memory") {
        auto table_alloc = spilling::cudf_table_converter::convert_to_host(
            table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        REQUIRE(table_alloc.data_size > 0);
        REQUIRE(table_alloc.metadata != nullptr);
        REQUIRE(table_alloc.metadata->size() > 0);
        REQUIRE(table_alloc.allocation.size() > 0);
        REQUIRE(table_alloc.allocation.size() * custom_mr->get_block_size() >= table_alloc.data_size);
    }

    SECTION("Test automatic cleanup") {
        const auto initial_free_blocks = custom_mr->get_free_blocks();
        
        {
            auto test_table_alloc = spilling::cudf_table_converter::convert_to_host(
                table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
            REQUIRE(custom_mr->get_free_blocks() < initial_free_blocks);
        }
        
        REQUIRE(custom_mr->get_free_blocks() == initial_free_blocks);
    }

    cudaStreamDestroy(stream);
}

TEST_CASE("cuDF to Host Large Dataset", "[cudf][to_host][large]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size);

    std::vector<int32_t> large_data(500000);
    for (size_t i = 0; i < large_data.size(); ++i) {
        large_data[i] = static_cast<int32_t>(i);
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    rmm::device_buffer large_gpu_buffer(large_data.size() * sizeof(int32_t), rmm::cuda_stream_view(stream));
    cudaMemcpy(large_gpu_buffer.data(), large_data.data(), 
               large_data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    auto large_column_view = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32}, 
        large_data.size(), 
        large_gpu_buffer.data(),
        nullptr,
        0,
        0,
        std::vector<cudf::column_view>{}
    );
    
    std::vector<cudf::column_view> large_column_views = {large_column_view};
    cudf::table_view large_table_view(large_column_views);

    SECTION("Convert large table to host memory") {
        auto large_table_alloc = spilling::cudf_table_converter::convert_to_host(
            large_table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        REQUIRE(large_table_alloc.data_size > 0);
        REQUIRE(large_table_alloc.metadata != nullptr);
        REQUIRE(large_table_alloc.metadata->size() > 0);
        REQUIRE(large_table_alloc.allocation.size() > 1);
        
        const auto expected_blocks = (large_data.size() * sizeof(int32_t) + custom_mr->get_block_size() - 1) / custom_mr->get_block_size();
        REQUIRE(large_table_alloc.allocation.size() >= expected_blocks);
    }

    SECTION("Recreate large table from host memory") {
        auto large_table_alloc = spilling::cudf_table_converter::convert_to_host(
            large_table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        auto recreated_table = spilling::cudf_table_converter::recreate_table(large_table_alloc, rmm::cuda_stream_view(stream));
        
        REQUIRE(recreated_table.num_columns() == 1);
        REQUIRE(recreated_table.num_rows() == large_data.size());
        
        std::vector<int32_t> recreated_data(large_data.size());
        cudaMemcpy(recreated_data.data(), recreated_table.view().column(0).data<int32_t>(), 
                   large_data.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < std::min(1000UL, large_data.size()); ++i) {
            REQUIRE(recreated_data[i] == large_data[i]);
        }
    }

    cudaStreamDestroy(stream);
}