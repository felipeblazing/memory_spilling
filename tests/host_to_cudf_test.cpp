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

TEST_CASE("Host to cuDF", "[cudf][host_to_cudf]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size);

    std::vector<int32_t> int_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> float_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.0f};
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    rmm::device_buffer int_buffer(int_data.size() * sizeof(int32_t), rmm::cuda_stream_view(stream));
    rmm::device_buffer float_buffer(float_data.size() * sizeof(float), rmm::cuda_stream_view(stream));
    
    cudaMemcpy(int_buffer.data(), int_data.data(), 
               int_data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(float_buffer.data(), float_data.data(), 
               float_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    auto int_column_view = cudf::column_view(
        cudf::data_type{cudf::type_id::INT32}, 
        int_data.size(), 
        int_buffer.data(),
        nullptr,
        0,
        0,
        std::vector<cudf::column_view>{}
    );
    
    auto float_column_view = cudf::column_view(
        cudf::data_type{cudf::type_id::FLOAT32}, 
        float_data.size(), 
        float_buffer.data(),
        nullptr,
        0,
        0,
        std::vector<cudf::column_view>{}
    );
    
    std::vector<cudf::column_view> column_views = {int_column_view, float_column_view};
    cudf::table_view original_table(column_views);

    SECTION("Pack table using cuDF") {
        auto packed_data = cudf::pack(original_table);
        
        REQUIRE(packed_data.gpu_data != nullptr);
        REQUIRE(packed_data.metadata != nullptr);
        REQUIRE(packed_data.gpu_data->size() > 0);
        REQUIRE(packed_data.metadata->size() > 0);
    }

    SECTION("Convert table to host memory") {
        auto table_alloc = spilling::cudf_table_converter::convert_to_host(
            original_table, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        REQUIRE(table_alloc.data_size > 0);
        REQUIRE(table_alloc.metadata != nullptr);
        REQUIRE(table_alloc.metadata->size() > 0);
        REQUIRE(table_alloc.allocation.size() > 0);
    }

    SECTION("Recreate table from host memory") {
        auto table_alloc = spilling::cudf_table_converter::convert_to_host(
            original_table, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        auto recreated_table = spilling::cudf_table_converter::recreate_table(table_alloc, rmm::cuda_stream_view(stream));
        
        REQUIRE(recreated_table.num_columns() == original_table.num_columns());
        REQUIRE(recreated_table.num_rows() == original_table.num_rows());
        
        for (std::size_t i = 0; i < original_table.num_columns(); ++i) {
            REQUIRE(recreated_table.view().column(i).type() == original_table.column(i).type());
        }
        
        std::vector<int32_t> original_int_data(int_data.size());
        std::vector<int32_t> recreated_int_data(int_data.size());
        
        cudaMemcpy(original_int_data.data(), original_table.column(0).data<int32_t>(), 
                   int_data.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(recreated_int_data.data(), recreated_table.view().column(0).data<int32_t>(), 
                   int_data.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        for (std::size_t i = 0; i < int_data.size(); ++i) {
            REQUIRE(original_int_data[i] == recreated_int_data[i]);
        }
        
        std::vector<float> original_float_data(float_data.size());
        std::vector<float> recreated_float_data(float_data.size());
        
        cudaMemcpy(original_float_data.data(), original_table.column(1).data<float>(), 
                   float_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(recreated_float_data.data(), recreated_table.view().column(1).data<float>(), 
                   float_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        for (std::size_t i = 0; i < float_data.size(); ++i) {
            REQUIRE(std::abs(original_float_data[i] - recreated_float_data[i]) <= 1e-6f);
        }
    }

    SECTION("Test automatic cleanup") {
        const auto initial_free_blocks = custom_mr->get_free_blocks();
        
        {
            auto test_table_alloc = spilling::cudf_table_converter::convert_to_host(
                original_table, custom_mr.get(), rmm::cuda_stream_view(stream));
            REQUIRE(custom_mr->get_free_blocks() < initial_free_blocks);
        }
        
        REQUIRE(custom_mr->get_free_blocks() == initial_free_blocks);
    }

    cudaStreamDestroy(stream);
}