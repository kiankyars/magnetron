// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(allocators, fixed_intrusive_pool_alloc_free) {
    mag_fixed_intrusive_pool pool {};
    mag_fixed_intrusive_pool_init(&pool, sizeof(int), alignof(int), 8);
    ASSERT_EQ(pool.num_allocs, 0);
    ASSERT_EQ(pool.num_chunks, 1);
    ASSERT_EQ(pool.num_freelist_hits, 0);
    ASSERT_EQ(pool.num_pool_hits, 0);
    for (int i = 0; i < 8192; ++i) {
        int* x = static_cast<int*>(mag_fixed_intrusive_pool_malloc(&pool));
        ASSERT_EQ(pool.num_allocs, i+1);
        if (i >= 1) {
            ASSERT_EQ(pool.num_freelist_hits, i);
        }
        *x = -1;
        ASSERT_EQ(*x, -1);
        mag_fixed_intrusive_pool_free(&pool, x);
    }
    ASSERT_EQ(pool.num_chunks, 1);
    ASSERT_EQ(pool.num_pool_hits, 1);
    mag_fixed_intrusive_pool_destroy(&pool);
}

TEST(allocators, fixed_intrusive_pool_exhaust_pool) {
    mag_fixed_intrusive_pool pool {};
    mag_fixed_intrusive_pool_init(&pool, sizeof(int), alignof(int), 8);
    ASSERT_EQ(pool.num_allocs, 0);
    ASSERT_EQ(pool.num_chunks, 1);
    ASSERT_EQ(pool.num_freelist_hits, 0);
    ASSERT_EQ(pool.num_pool_hits, 0);
    for (int i = 0; i < 8192; ++i) {
        [[maybe_unused]] int* volatile x = static_cast<int*>(mag_fixed_intrusive_pool_malloc(&pool));
    }
    ASSERT_EQ(pool.num_chunks, 8192/8);
    //ASSERT_EQ(pool.num_pool_hits, 1);
    ASSERT_EQ(pool.num_freelist_hits, 0);
    mag_fixed_intrusive_pool_destroy(&pool);
}

#ifndef _MSC_VER // MSVC fucks around with linking a __declspex(dllexport) ed function ptr. TODO: fix

TEST(allocators, alloc_small) {
    int* i = static_cast<int*>((*mag_alloc)(nullptr, sizeof(int)));
    ASSERT_NE(i, nullptr);
    *i = -1;
    ASSERT_EQ(*i, -1);
    (*mag_alloc)(i, 0);
}

TEST(allocators, alloc_large) {
    void* i = (*mag_alloc)(nullptr, 1ull<<30);
    ASSERT_NE(i, nullptr);
    std::memset(i, 5, 1ull<<30);
    (*mag_alloc)(i, 0);
}

#endif

TEST(allocators, alloc_aligned) {
    std::size_t align = alignof(int);
    for (; align <= 8192; align <<= 1) {
        int* i = static_cast<int*>(mag_alloc_aligned(sizeof(int), align));
        ASSERT_NE(i, nullptr);
        *i = -1;
        ASSERT_EQ(*i, -1);
        ASSERT_EQ(reinterpret_cast<std::uintptr_t>(i) % align, 0);
        mag_free_aligned(i);
    }
}
