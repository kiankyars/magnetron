// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <thread>
#include <atomic>
#include <chrono>

#include "prelude.hpp"
#include "magnetron_cpu.c"

using namespace std::chrono_literals;

TEST(threading, create_thread) {
    std::atomic<bool> executed = false;
    static volatile std::atomic_uintptr_t main_tid {mag_thread_id()};
    static volatile std::atomic_uintptr_t second_tid {};

    std::cout << "Main thread ID: " << std::hex << main_tid.load(std::memory_order_seq_cst) << std::endl;

    auto thread_func = [](void* arg) -> void* {
        std::this_thread::sleep_for(1s); // Simulate some work
        second_tid.store(mag_thread_id(), std::memory_order_seq_cst);
        static_cast<std::atomic<bool>*>(arg)->store(true, std::memory_order_seq_cst);
        std::cout << "Second thread ID: " << std::hex << second_tid.load(std::memory_order_seq_cst) << std::endl;
        return NULL;
    };

    mag_thread_t thread;
    mag_thread_create(&thread, nullptr, thread_func, &executed);
    ASSERT_NE(thread, nullptr);
    mag_thread_join(thread, nullptr);
    EXPECT_EQ(true, executed.load(std::memory_order_seq_cst));
    ASSERT_NE(main_tid.load(std::memory_order_seq_cst), second_tid.load(std::memory_order_seq_cst));
}

TEST(thread_pool, create_destroy) {
    mag_threadpool_t* pool = mag_threadpool_create(4, MAG_THREAD_SCHED_PRIO_NORMAL);
    ASSERT_NE(pool, nullptr);
    mag_threadpool_destroy(pool);
}

TEST(thread_pool, simulated_work_some_tasks) {
    constexpr std::size_t num_tasks = 4;
    mag_threadpool_t* pool = mag_threadpool_create(2, MAG_THREAD_SCHED_PRIO_NORMAL);
    ASSERT_NE(pool, nullptr);
    for (std::size_t i = 0; i < num_tasks; ++i) {
        mag_threadpool_enqueue_task(pool, [](void* arg) -> void* {
            std::cout << "Working on thread: " << std::hex << mag_thread_id() << std::endl;
            return nullptr;
        }, pool);
    }
    mag_threadpool_barrier(pool);
    mag_threadpool_destroy(pool);
}

TEST(thread_pool, simulated_heavy_workload_some_slow_tasks) {
    constexpr std::size_t num_tasks = 10;
    mag_threadpool_t* pool = mag_threadpool_create(std::max(1u, std::thread::hardware_concurrency()), MAG_THREAD_SCHED_PRIO_NORMAL);
    ASSERT_NE(pool, nullptr);
    for (std::size_t i = 0; i < num_tasks; ++i) {
        mag_threadpool_enqueue_task(pool, [](void* arg) -> void* {
            volatile auto target = reinterpret_cast<std::uintptr_t>(arg);
            volatile std::uintptr_t result = 0;
            for (volatile std::size_t i = 0; i < target; ++i) {
                result += i;
            }
            return reinterpret_cast<void*>(result);
        }, reinterpret_cast<void*>(250'000'000));
    }
    mag_threadpool_barrier(pool);
    mag_threadpool_destroy(pool);
}

TEST(thread_pool, simulated_heavy_workload_many_fast_tasks) {
    constexpr std::size_t num_tasks = 8192;
    mag_threadpool_t* pool = mag_threadpool_create(std::max(1u, std::thread::hardware_concurrency()), MAG_THREAD_SCHED_PRIO_NORMAL);
    ASSERT_NE(pool, nullptr);
    for (std::size_t i = 0; i < num_tasks; ++i) {
        mag_threadpool_enqueue_task(pool, [](void* arg) -> void* {
            volatile auto target = reinterpret_cast<std::uintptr_t>(arg);
            volatile std::uintptr_t result = 0;
            for (volatile std::size_t i = 0; i < target; ++i) {
                result += i;
            }
            return reinterpret_cast<void*>(result);
        }, reinterpret_cast<void*>(1'000));
    }
    mag_threadpool_barrier(pool);
    mag_threadpool_destroy(pool);
}

TEST(threading, StoreLoadTest) {
    mag_atomic_t val = 0;
    mag_atomic_store(&val, 42, MAG_MO_RELAXED);
    int loaded = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(loaded, 42);
}

TEST(threading, FetchAddTest) {
    mag_atomic_t val = 10;
    int old = mag_atomic_fetch_add(&val, 5, MAG_MO_RELAXED);
    EXPECT_EQ(old, 10);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 15);
}

TEST(threading, FetchSubTest) {
    mag_atomic_t val = 20;
    int old = mag_atomic_fetch_sub(&val, 3, MAG_MO_RELAXED);
    EXPECT_EQ(old, 20);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 17);
}

TEST(threading, FetchAndTest) {
    mag_atomic_t val = 0xF0;
    int old = mag_atomic_fetch_and(&val, 0x0F, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xF0);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x00);
}

TEST(threading, FetchOrTest) {
    mag_atomic_t val = 0x0F;
    int old = mag_atomic_fetch_or(&val, 0xF0, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0x0F);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0xFF);
}

TEST(threading, FetchXorTest) {
    mag_atomic_t val = 0xAA;
    int old = mag_atomic_fetch_xor(&val, 0xFF, MAG_MO_RELAXED);
    EXPECT_EQ(old, 0xAA);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 0x55);
}

TEST(threading, ExchangeTest) {
    mag_atomic_t val = 100;
    int old = mag_atomic_exchange(&val, 200, MAG_MO_RELAXED);
    EXPECT_EQ(old, 100);
    int new_val = mag_atomic_load(&val, MAG_MO_RELAXED);
    EXPECT_EQ(new_val, 200);
}

TEST(threading, CompareExchangeWeakSuccessTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 10;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 20);
    EXPECT_EQ(expected, 10);
}

TEST(threading, CompareExchangeWeakFailTest) {
    mag_atomic_t val = 10;
    mag_atomic_t expected = 5;
    mag_atomic_t desired = 20;
    bool success = mag_atomic_compare_exchange_weak(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 10);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 10);
}

TEST(threading, CompareExchangeStrongSuccessTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 30;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_TRUE(success);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 40);
    EXPECT_EQ(expected, 30);
}

TEST(threading, CompareExchangeStrongFailTest) {
    mag_atomic_t val = 30;
    mag_atomic_t expected = 25;
    mag_atomic_t desired = 40;
    bool success = mag_atomic_compare_exchange_strong(&val, &expected, &desired, MAG_MO_RELAXED, MAG_MO_RELAXED);
    EXPECT_FALSE(success);
    EXPECT_EQ(expected, 30);
    EXPECT_EQ(mag_atomic_load(&val, MAG_MO_RELAXED), 30);
}