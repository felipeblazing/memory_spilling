#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "fixed_size_host_memory_resource.hpp"

TEST_CASE("Custom Memory Resource", "[memory][resource]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    constexpr std::size_t initial_pools = 4;
    constexpr std::size_t total_initial_blocks = pool_size * initial_pools;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size, initial_pools);

    SECTION("Single block allocation") {
        void* ptr1 = custom_mr->allocate(512 * 1024);
        
        REQUIRE(ptr1 != nullptr);
        REQUIRE(custom_mr->get_free_blocks() < total_initial_blocks);
        
        custom_mr->deallocate(ptr1, 512 * 1024);
        REQUIRE(custom_mr->get_free_blocks() == total_initial_blocks);
    }

    SECTION("Multiple block allocation") {
        void* ptr2 = custom_mr->allocate(1024 * 1024);
        
        REQUIRE(ptr2 != nullptr);
        REQUIRE(custom_mr->get_free_blocks() < total_initial_blocks);
        
        custom_mr->deallocate(ptr2, 1024 * 1024);
        REQUIRE(custom_mr->get_free_blocks() == total_initial_blocks);
    }

    SECTION("Pool expansion") {
        std::vector<void*> ptrs;
        for (int i = 0; i < total_initial_blocks; ++i) {
            void* ptr = custom_mr->allocate(1024 * 1024);
            REQUIRE(ptr != nullptr);
            ptrs.push_back(ptr);
        }
        
        REQUIRE(custom_mr->get_free_blocks() == 0);
        
        void* extra_ptr = custom_mr->allocate(1024 * 1024);
        
        REQUIRE(extra_ptr != nullptr);
        REQUIRE(custom_mr->get_free_blocks() > 0);
        
        for (void* ptr : ptrs) {
            custom_mr->deallocate(ptr, 1024 * 1024);
        }
        custom_mr->deallocate(extra_ptr, 1024 * 1024);
    }

    SECTION("Multiple blocks allocation") {
        auto multi_alloc = custom_mr->allocate_multiple_blocks(3 * 1024 * 1024);
        
        REQUIRE(multi_alloc.size() > 0);
        REQUIRE(multi_alloc.size() >= 3);
    }

    SECTION("Memory resource properties") {
        REQUIRE(custom_mr->get_block_size() == block_size);
        REQUIRE(custom_mr->get_total_blocks() >= total_initial_blocks);
        REQUIRE(custom_mr->get_free_blocks() == total_initial_blocks);
    }
}