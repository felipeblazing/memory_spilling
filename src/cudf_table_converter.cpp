#include "cudf_table_converter.hpp"
#include <cudf/contiguous_split.hpp>

#include <cuda_runtime.h>
#include <cstring>
#include <stdexcept>
#include <nvtx3/nvtx3.hpp>

namespace spilling {

spilling::table_allocation 
cudf_table_converter::convert_to_host(const cudf::table_view& table, 
                                      spilling::fixed_size_host_memory_resource* mr,
                                      rmm::cuda_stream_view stream) {
    nvtx3::scoped_range range{"cudf_table_converter::convert_to_host", nvtx3::rgb{0, 0, 255}};
    
    if (table.num_columns() == 0) {
        auto empty_allocation = mr->allocate_multiple_blocks(0);
        auto empty_metadata = std::make_unique<std::vector<uint8_t>>();
        return spilling::table_allocation(std::move(empty_allocation), std::move(empty_metadata), 0);
    }

    nvtx3::scoped_range pack_range{"cudf::pack"};
    auto packed_data = cudf::pack(table);

    auto metadata = std::make_unique<std::vector<uint8_t>>(*packed_data.metadata);

    nvtx3::scoped_range copy_range{"copy_data_to_host"};
    std::size_t data_size;
    auto allocation = copy_data_to_host(packed_data.gpu_data.get(), mr, data_size, stream);

    return spilling::table_allocation(std::move(allocation), std::move(metadata), data_size);
}


spilling::fixed_size_host_memory_resource::multiple_blocks_allocation 
cudf_table_converter::copy_data_to_host(const rmm::device_buffer* gpu_data, 
                                        spilling::fixed_size_host_memory_resource* mr,
                                        std::size_t& data_size,
                                        rmm::cuda_stream_view stream) {
    nvtx3::scoped_range range{"cudf_table_converter::copy_data_to_host"};
    
    data_size = gpu_data->size();
    
    spilling::fixed_size_host_memory_resource::multiple_blocks_allocation allocation = mr->allocate_multiple_blocks(data_size);
    
    if (allocation.size() == 0) {
        return allocation;
    }

    nvtx3::scoped_range copy_range{"gpu_to_host_copy_loop"};
    std::size_t remaining_bytes = data_size;
    std::size_t block_index = 0;
    std::size_t block_offset = 0;
    const std::size_t block_size = mr->get_block_size();
    
    const uint8_t* gpu_data_ptr = static_cast<const uint8_t*>(gpu_data->data());
    
    while (remaining_bytes > 0) {
        std::size_t bytes_to_copy = std::min(remaining_bytes, block_size - block_offset);
        
        void* block_ptr = allocation[block_index];
        void* dest_ptr = static_cast<char*>(block_ptr) + block_offset;
        
        std::size_t source_offset = data_size - remaining_bytes;
        
        cudaMemcpyAsync(dest_ptr, gpu_data_ptr + source_offset, bytes_to_copy, cudaMemcpyDeviceToHost, stream.value());
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
        }
        
        remaining_bytes -= bytes_to_copy;
        block_offset += bytes_to_copy;
        
        if (block_offset >= block_size) {
            block_index++;
            block_offset = 0;
        }
    }
    
    stream.synchronize();

    return allocation;
}


cudf::table cudf_table_converter::recreate_table(const spilling::table_allocation& table_alloc, 
                                                 rmm::cuda_stream_view stream) {
    nvtx3::scoped_range range{"cudf_table_converter::recreate_table", nvtx3::rgb{128, 0, 128}};
    
    if (table_alloc.allocation.size() == 0) {
        return cudf::table(std::vector<std::unique_ptr<cudf::column>>{});
    }

    rmm::device_buffer gpu_data;
    {
        nvtx3::scoped_range buffer_range{"create_device_buffer"};
        gpu_data = rmm::device_buffer(table_alloc.data_size, stream);
    }
    
    if (table_alloc.data_size > 0) {
        nvtx3::scoped_range copy_range{"host_to_gpu_copy_loop"};
        std::size_t remaining_data = table_alloc.data_size;
        std::size_t block_index = 0;
        std::size_t block_offset = 0;
        const std::size_t block_size = table_alloc.allocation.block_size;
        
        while (remaining_data > 0) {
            std::size_t bytes_to_copy = std::min(remaining_data, block_size - block_offset);
            
            const void* block_ptr = table_alloc.allocation[block_index];
            const uint8_t* source_ptr = static_cast<const uint8_t*>(block_ptr) + block_offset;
            
            std::size_t dest_offset = table_alloc.data_size - remaining_data;
            void* gpu_dest_ptr = static_cast<uint8_t*>(gpu_data.data()) + dest_offset;
            
            nvtx3::scoped_range memcpy_range{"cudaMemcpyAsync_host_to_device", nvtx3::rgb{0, 255, 0}};
            cudaMemcpyAsync(gpu_dest_ptr, source_ptr, bytes_to_copy, cudaMemcpyHostToDevice, stream.value());
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                throw std::runtime_error("CUDA memcpy failed during recreation: " + std::string(cudaGetErrorString(err)));
            }
            
            remaining_data -= bytes_to_copy;
            block_offset += bytes_to_copy;
            if (block_offset >= block_size) {
                block_index++;
                block_offset = 0;
            }
        }
    }
    
    {
        nvtx3::scoped_range sync_range{"stream_synchronize_recreate", nvtx3::rgb{255, 0, 0}};
        stream.synchronize();
    }

    nvtx3::scoped_range packed_range{"create_packed_columns"};
    cudf::packed_columns packed_data;
    packed_data.metadata = std::make_unique<std::vector<uint8_t>>(*table_alloc.metadata);
    packed_data.gpu_data = std::make_unique<rmm::device_buffer>(std::move(gpu_data));

    cudf::table_view table_view;
    {
        nvtx3::scoped_range unpack_range{"cudf::unpack"};
        table_view = cudf::unpack(packed_data);
    }
    
    nvtx3::scoped_range column_range{"create_columns_from_view"};
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (cudf::size_type i = 0; i < table_view.num_columns(); ++i) {
        columns.push_back(std::make_unique<cudf::column>(table_view.column(i)));
    }
    
    stream.synchronize();

    return cudf::table(std::move(columns));
}


} // namespace spilling
