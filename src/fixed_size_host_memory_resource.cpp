#include "fixed_size_host_memory_resource.hpp"
#include <nvtx3/nvtx3.hpp>

namespace spilling {

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t initial_pools)
    : block_size_(rmm::align_up(block_size, alignof(std::max_align_t))),
      pool_size_(pool_size)
{
    for (std::size_t i = 0; i < initial_pools; ++i) {
        expand_pool();
    }
}

fixed_size_host_memory_resource::fixed_size_host_memory_resource(
    std::unique_ptr<rmm::mr::host_memory_resource> upstream_mr,
    std::size_t block_size,
    std::size_t pool_size,
    std::size_t initial_pools)
    : block_size_(rmm::align_up(block_size, alignof(std::max_align_t))),
      pool_size_(pool_size),
      upstream_mr_(std::move(upstream_mr))
{
    for (std::size_t i = 0; i < initial_pools; ++i) {
        expand_pool();
    }
}

fixed_size_host_memory_resource::~fixed_size_host_memory_resource() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& block : allocated_blocks_) {
        const std::size_t dealloc_size = block_size_ * pool_size_;
        
        if (upstream_mr_) {
            upstream_mr_->deallocate(block, dealloc_size);
        } else {
            rmm::mr::pinned_host_memory_resource::deallocate(block, dealloc_size);
        }
    }
    allocated_blocks_.clear();
    free_blocks_.clear();
}

std::size_t fixed_size_host_memory_resource::get_block_size() const noexcept {
    return block_size_;
}

std::size_t fixed_size_host_memory_resource::get_free_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_.size();
}

std::size_t fixed_size_host_memory_resource::get_total_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_blocks_.size() * pool_size_;
}

rmm::mr::host_memory_resource* fixed_size_host_memory_resource::get_upstream_resource() const noexcept {
    return upstream_mr_.get();
}

fixed_size_host_memory_resource::multiple_blocks_allocation fixed_size_host_memory_resource::allocate_multiple_blocks(std::size_t total_bytes) {
    nvtx3::scoped_range range{"fixed_size_host_memory_resource::allocate_multiple_blocks"};
    
    if (total_bytes == 0) {
        return multiple_blocks_allocation({}, this, block_size_);
    }

    const std::size_t num_blocks = (total_bytes + block_size_ - 1) / block_size_;
    
    std::vector<void*> allocated_blocks;
    allocated_blocks.reserve(num_blocks);

    nvtxRangePush("mutex_lock");
    std::lock_guard<std::mutex> lock(mutex_);
    nvtxRangePop();

    nvtx3::scoped_range alloc_loop_range{"block_allocation_loop"};
    for (std::size_t i = 0; i < num_blocks; ++i) {
        if (free_blocks_.empty()) {
            nvtx3::scoped_range expand_range{"expand_pool", nvtx3::rgb{255, 255, 0}};
            expand_pool();
        }

        if (free_blocks_.empty()) {
            nvtx3::scoped_range cleanup_range{"cleanup_on_failure", nvtx3::rgb{255, 0, 0}};
            for (void* ptr : allocated_blocks) {
                free_blocks_.push_back(ptr);
            }
            throw std::bad_alloc();
        }

        nvtx3::scoped_range block_alloc_range{"allocate_single_block"};
        void* ptr = free_blocks_.back();
        free_blocks_.pop_back();
        allocated_blocks.push_back(ptr);
    }

    return multiple_blocks_allocation(std::move(allocated_blocks), this, block_size_);
}

void* fixed_size_host_memory_resource::do_allocate(std::size_t bytes, std::size_t alignment) {
    RMM_FUNC_RANGE();
    
    if (bytes == 0) {
        return nullptr;
    }

    if (bytes > block_size_) {
        throw std::bad_alloc();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    
    if (free_blocks_.empty()) {
        expand_pool();
    }

    if (free_blocks_.empty()) {
        throw std::bad_alloc();
    }

    void* ptr = free_blocks_.back();
    free_blocks_.pop_back();
    
    return ptr;
}

void fixed_size_host_memory_resource::do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) {
    RMM_FUNC_RANGE();
    
    if (ptr == nullptr) {
        return;
    }

    if (bytes > block_size_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push_back(ptr);
}

bool fixed_size_host_memory_resource::do_is_equal(const rmm::mr::host_memory_resource& other) const noexcept {
    return this == &other;
}

void fixed_size_host_memory_resource::expand_pool() {
    nvtx3::scoped_range range{"fixed_size_host_memory_resource::expand_pool", nvtx3::rgb{255, 255, 0}};
    

    const std::size_t total_size = block_size_ * pool_size_;
    
    void* large_allocation;
    {
        nvtx3::scoped_range alloc_range{"upstream_allocation"};
        if (upstream_mr_) {
            large_allocation = upstream_mr_->allocate(total_size);
        } else {
            large_allocation = rmm::mr::pinned_host_memory_resource::allocate(total_size);
        }
    }
    
    allocated_blocks_.push_back(large_allocation);
    
    nvtx3::scoped_range split_range{"split_into_blocks"};
    for (std::size_t i = 0; i < pool_size_; ++i) {
        void* block = static_cast<char*>(large_allocation) + (i * block_size_);
        free_blocks_.push_back(block);
    }
}

} // namespace spilling
