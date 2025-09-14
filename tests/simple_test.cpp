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

TEST_CASE("Simple cuDF to Host Test", "[cudf][to_host][simple]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size);

    std::vector<int32_t> test_data = {42, 43, 44, 45, 46};
    
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

    SECTION("Convert table to host memory") {
        auto table_alloc = spilling::cudf_table_converter::convert_to_host(
            table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        REQUIRE(table_alloc.data_size > 0);
        REQUIRE(table_alloc.metadata != nullptr);
        REQUIRE(table_alloc.metadata->size() > 0);
        REQUIRE(table_alloc.allocation.size() > 0);
        
        const uint8_t* host_data = static_cast<const uint8_t*>(table_alloc.allocation[0]);
        REQUIRE(host_data != nullptr);
    }

    SECTION("Recreate table from host memory") {
        auto table_alloc = spilling::cudf_table_converter::convert_to_host(
            table_view, custom_mr.get(), rmm::cuda_stream_view(stream));
        
        auto recreated_table = spilling::cudf_table_converter::recreate_table(table_alloc, rmm::cuda_stream_view(stream));
        
        REQUIRE(recreated_table.num_columns() == 1);
        REQUIRE(recreated_table.num_rows() == test_data.size());
        
        std::vector<int32_t> recreated_data(test_data.size());
        cudaMemcpy(recreated_data.data(), recreated_table.view().column(0).data<int32_t>(), 
                   test_data.size() * sizeof(int32_t), cudaMemcpyDeviceToHost);
        
        for (size_t i = 0; i < test_data.size(); ++i) {
            REQUIRE(recreated_data[i] == test_data[i]);
        }
    }

    cudaStreamDestroy(stream);
}
