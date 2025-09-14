#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "fixed_size_host_memory_resource.hpp"

TEST_CASE("Multiple Blocks Allocation", "[memory][multiple_blocks]") {
    constexpr std::size_t block_size = 1024 * 1024;
    constexpr std::size_t pool_size = 10;
    
    auto custom_mr = std::make_unique<spilling::fixed_size_host_memory_resource>(
        block_size, pool_size);

    SECTION("Allocate multiple blocks for large request") {
        auto multi_alloc = custom_mr->allocate_multiple_blocks(3 * 1024 * 1024);
        
        REQUIRE(multi_alloc.size() >= 3);
        REQUIRE(multi_alloc.size() * custom_mr->get_block_size() >= 3 * 1024 * 1024);
        
        for (size_t i = 0; i < multi_alloc.size(); ++i) {
            REQUIRE(multi_alloc[i] != nullptr);
        }
    }

    SECTION("Allocate multiple blocks for exact block size") {
        auto multi_alloc = custom_mr->allocate_multiple_blocks(1024 * 1024);
        
        REQUIRE(multi_alloc.size() == 1);
        REQUIRE(multi_alloc[0] != nullptr);
    }

    SECTION("Allocate multiple blocks for small request") {
        auto multi_alloc = custom_mr->allocate_multiple_blocks(512 * 1024);
        
        REQUIRE(multi_alloc.size() == 1);
        REQUIRE(multi_alloc[0] != nullptr);
    }

    SECTION("Test automatic cleanup") {
        const auto initial_free_blocks = custom_mr->get_free_blocks();
        
        {
            auto multi_alloc = custom_mr->allocate_multiple_blocks(2 * 1024 * 1024);
            REQUIRE(multi_alloc.size() >= 2);
            REQUIRE(custom_mr->get_free_blocks() < initial_free_blocks);
        }
        
        REQUIRE(custom_mr->get_free_blocks() == initial_free_blocks);
    }

    SECTION("Test multiple allocations") {
        const auto initial_free_blocks = custom_mr->get_free_blocks();
        
        auto alloc1 = custom_mr->allocate_multiple_blocks(1024 * 1024);
        auto alloc2 = custom_mr->allocate_multiple_blocks(2 * 1024 * 1024);
        auto alloc3 = custom_mr->allocate_multiple_blocks(512 * 1024);
        
        REQUIRE(alloc1.size() >= 1);
        REQUIRE(alloc2.size() >= 2);
        REQUIRE(alloc3.size() >= 1);
        
        REQUIRE(custom_mr->get_free_blocks() <= initial_free_blocks - 4);
    }
}