#include <catch2/catch_test_macros.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

TEST_CASE("RMM Basic Memory Management", "[rmm][basic]") {
    cudaError_t cuda_status = cudaSetDevice(0);
    REQUIRE(cuda_status == cudaSuccess);
    
    auto cuda_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    
    auto pool_mr = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
        cuda_mr.get(), 1024 * 1024 * 1024
    );
    
    rmm::mr::set_current_device_resource(pool_mr.get());
    
    SECTION("Device buffer creation and allocation") {
        constexpr size_t buffer_size = 1024 * 1024;
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        rmm::device_buffer buffer(buffer_size, rmm::cuda_stream_view(stream));
        
        REQUIRE(buffer.size() == buffer_size);
        REQUIRE(buffer.data() != nullptr);
        
        cudaStreamDestroy(stream);
    }
}
